"""Strict one-time depth migration (notes/solo/deepen-spec.md 2): append transformer blocks to the main token trunk.

Each appended ``Block`` has ``proj`` and ``ff2`` (weights and biases) exactly zero, so both of its residual branches
add exactly +0 and the block is the identity map; its other parameters (qkv, ff1, film) get the model's ordinary
initialisation (``SpireNet.reset_parameters``: weights N(0, init_std), biases 0), drawn from a fixed generator.
Online and EMA weights get the same new values. Old weights and Adam histories stay exact; the new parameters
start with zero moments and step 0.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from .model import OUTPUT_KEYS, SpireNet
from .tokenize import Sample, Tokenizer

DEPTH_INIT_SEED = 0x44455054  # "DEPT"
ZERO_MODULES = ("proj", "ff2")  # the two residual outputs of a pre-norm block
BLOCK_MODULES = {"qkv", "proj", "ff1", "ff2", "film"}


def new_block_names(net: SpireNet, old_layers: int) -> list[str]:
    assert len(net.blocks) == net.config.n_layers > old_layers >= 1, "depth migration must add blocks"
    return [f"blocks.{i}.{name}" for i in range(old_layers, net.config.n_layers)
            for name, _ in net.blocks[i].named_parameters()]


def new_block_state(net: SpireNet, old_layers: int) -> dict[str, Tensor]:
    """Values of the appended blocks' parameters (identity blocks; deterministic)."""
    g = torch.Generator().manual_seed(DEPTH_INIT_SEED)
    out: dict[str, Tensor] = {}
    for i in range(old_layers, net.config.n_layers):
        params = dict(net.blocks[i].named_parameters())
        assert {n.split(".")[0] for n in params} == BLOCK_MODULES and len(params) == 2 * len(BLOCK_MODULES), params
        for name, p in params.items():
            module, kind = name.split(".")
            assert kind in ("weight", "bias"), name
            if module in ZERO_MODULES or kind == "bias":
                value = torch.zeros(p.shape, dtype=p.dtype)
            else:
                value = torch.empty(p.shape, dtype=p.dtype).normal_(0.0, net.config.init_std, generator=g)
            out[f"blocks.{i}.{name}"] = value
    assert list(out) == new_block_names(net, old_layers)
    return out


def migrate_depth_weights(state: Mapping[str, Tensor], net: SpireNet, new: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Online or EMA weights of the shallower model -> the deeper model's (outputs unchanged)."""
    expected = net.state_dict()
    assert set(state) == set(expected) - set(new) and not set(state) & set(new), \
        "migration requires exactly the shallower model's keys"
    out: dict[str, Tensor] = {}
    for name, ref in expected.items():
        value = state[name] if name in state else new[name]
        assert value.shape == ref.shape and value.dtype == ref.dtype, f"{name}: shape / dtype mismatch"
        out[name] = value.detach().clone()
    return out


def migrate_depth_optimizer(saved: dict[str, Any], net: SpireNet, old_layers: int,
                            optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Preserve every old Adam step / moment (old parameter IDs remapped by position within their unchanged group);
    the appended blocks' parameters get zero moments and step 0."""
    assert not optimizer.state, "depth migration must precede optimizer updates"
    added_names = set(new_block_names(net, old_layers))
    names = {id(p): name for name, p in net.named_parameters()}
    current = optimizer.state_dict()
    assert len(saved["param_groups"]) == len(current["param_groups"]), "migration cannot change optimizer groups"
    remap: dict[int, tuple[int, str, torch.nn.Parameter, bool]] = {}
    added: list[tuple[int, torch.nn.Parameter, bool]] = []
    groups = []
    for old, new, live in zip(saved["param_groups"], current["param_groups"], optimizer.param_groups):
        existing = []
        for after, parameter in zip(new["params"], live["params"]):
            if names[id(parameter)] in added_names:
                added.append((after, parameter, bool(old["amsgrad"])))
            else:
                existing.append((after, names[id(parameter)], parameter, bool(old["amsgrad"])))
        assert len(old["params"]) == len(existing), "migration cannot change existing optimizer groups"
        for before, target in zip(old["params"], existing):
            assert before not in remap, "duplicate optimizer parameter ID"
            remap[before] = target
        groups.append({**old, "params": new["params"]})
    assert len(added) == len(added_names)
    assert set(saved["state"]) == set(remap), "optimizer state must cover exactly the existing parameters"
    state: dict[int, dict[str, Tensor]] = {}
    step: Tensor | None = None
    for before, entry in saved["state"].items():
        after, name, parameter, amsgrad = remap[before]
        expected = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if amsgrad else set())
        assert set(entry) == expected, f"{name}: invalid Adam state keys"
        assert isinstance(entry["step"], Tensor) and entry["step"].ndim == 0, f"{name}: invalid Adam step"
        for key, value in entry.items():
            assert key == "step" or value.shape == parameter.shape, f"{name}.{key}: Adam state does not fit"
        state[after] = {key: value.detach().clone() for key, value in entry.items()}
        step = entry["step"]
    assert step is not None, "the source optimizer has no state"
    for new_id, parameter, amsgrad in added:
        keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if amsgrad else ("exp_avg", "exp_avg_sq")
        state[new_id] = {"step": torch.zeros_like(step),
                         **{key: torch.zeros(parameter.shape, dtype=parameter.dtype) for key in keys}}
    return {"state": state, "param_groups": groups}


@torch.no_grad()
def equivalence(samples: list[Sample], tok: Tokenizer, before: SpireNet, after: SpireNet) -> dict[str, Any]:
    """Max |difference| per output, bit-identical share and greedy-argmax agreement (one sample at a time)."""
    worst = {k: 0.0 for k in OUTPUT_KEYS}
    exact = same_argmax = n_dec = 0
    for s in samples:
        batch = tok.collate([s])
        x, y = before(batch), after(batch)
        ok = True
        for k in OUTPUT_KEYS:
            fin = torch.isfinite(x[k])
            assert torch.equal(fin, torch.isfinite(y[k])), k
            if fin.any():
                worst[k] = max(worst[k], float((x[k][fin] - y[k][fin]).abs().max()))
            ok &= torch.equal(x[k], y[k])
        exact += ok
        if s.n_actions:
            n_dec += 1
            same_argmax += int(x["logits"][0, : s.n_actions].argmax() == y["logits"][0, : s.n_actions].argmax())
    return {"frames": len(samples), "decisions": n_dec, "bit_identical": exact, "max_abs_diff": worst,
            "argmax_agree": same_argmax}
