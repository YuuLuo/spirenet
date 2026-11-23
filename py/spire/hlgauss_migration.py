"""Strict one-time HL-Gauss migration: convert a scalar ``value_head`` checkpoint into a ``value_bins > 0``
distributional value head (``val_dist``) while preserving policy, aux heads, trunk weights and Adam state.

Mathematical projection
-----------------------
Old ``value_head = _norm_mlp(d, 1)`` computes ``v(g) = w_out @ h + b_out`` where
``h = GELU(W0 @ LN(g) + b0)``. New ``val_dist = _norm_mlp(d, K)`` copies ``(W0, b0)`` verbatim so ``h`` is
identical, and sets the final linear layer to the exact Gaussian log-density over bin centers ``c_k``::

    W_dist[k, :] = ((c_k - v0) / s^2) * w_out
    b_dist[k]    = ((c_k - v0) / s^2) * (b_out - v0) - (c_k - v0)^2 / (2 * s^2)

with ``v0 = clip(b_out, vmin, vmax)`` and width ``s = init_sigma``. Because softmax cancels additive terms
independent of ``k``, ``softmax(W_dist @ h + b_dist)_k`` is proportional to ``exp(-(c_k - v(g))^2 / (2 s^2))``,
so the expected value ``sum_k p_k c_k`` matches the old scalar ``value_head(g)`` to within ``< 0.01`` across
``[vmin + 2s, vmax - 2s]`` while keeping logits near 0 at the peak bin (high bf16 accuracy).

Optimizer state is preserved strictly for every parameter except ``val_dist.*``, whose Adam moments and step
are zeroed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .model import ModelConfig, SpireNet, model_config_dict

OLD_PREFIX = "value_head."
NEW_PREFIX = "val_dist."


def default_init_sigma(cfg: ModelConfig) -> float:
    """Default Gaussian projection width for HL-Gauss weight migration: smooth initial distribution with
    minimal boundary bias and small bf16 logit magnitude."""
    assert cfg.value_bins > 0
    return max(cfg.value_sigma, min(0.10, (cfg.value_vmax - cfg.value_vmin) / 30.0))


def migrate_hlgauss_weights(state: Mapping[str, Tensor], net: SpireNet, *,
                            init_sigma: float | None = None) -> dict[str, Tensor]:
    """Convert a scalar ``value_head`` state dict into ``net``'s HL-Gauss ``val_dist`` state dict.

    All non-value-head parameters are copied bit-for-bit; ``val_dist.1.0`` copies ``value_head.1.0``;
    ``val_dist.1.2`` projects ``value_head.1.2`` onto Gaussian bin logits over ``net.val_centers``.
    """
    cfg = net.config
    assert cfg.value_bins > 0 and net.val_dist is not None and net.value_head is None, \
        "HL-Gauss migration requires a target model with model.value_bins > 0"
    sigma = default_init_sigma(cfg) if init_sigma is None else float(init_sigma)
    assert sigma > 0.0, f"init_sigma must be positive, got {sigma}"
    expected = net.state_dict()
    old_keys = {k for k in state if k.startswith(OLD_PREFIX)}
    new_keys = {k for k in expected if k.startswith(NEW_PREFIX)}
    want_old = {"value_head.1.0.weight", "value_head.1.0.bias", "value_head.1.2.weight", "value_head.1.2.bias"}
    want_new = {"val_dist.1.0.weight", "val_dist.1.0.bias", "val_dist.1.2.weight", "val_dist.1.2.bias"}
    assert old_keys == want_old, f"source state must have scalar value_head keys {sorted(want_old)}, got {sorted(old_keys)}"
    assert new_keys == want_new, f"target model val_dist keys {sorted(new_keys)} != {sorted(want_new)}"
    assert not any(k.startswith(NEW_PREFIX) for k in state), "source state already has val_dist.*"
    shared_src = set(state) - old_keys
    shared_dst = set(expected) - new_keys
    assert shared_src == shared_dst, (
        f"non-value keys differ: missing {sorted(shared_dst - shared_src)[:5]}, "
        f"unexpected {sorted(shared_src - shared_dst)[:5]}"
    )
    out: dict[str, Tensor] = {}
    for k in shared_dst:
        assert state[k].shape == expected[k].shape and state[k].dtype == expected[k].dtype, \
            f"{k}: shape/dtype mismatch ({state[k].shape}, {state[k].dtype}) vs ({expected[k].shape}, {expected[k].dtype})"
        out[k] = state[k].detach().clone()

    # Hidden layer of value MLP: exact copy so hidden activation h is identical
    for suffix in ("1.0.weight", "1.0.bias"):
        src_t = state[OLD_PREFIX + suffix]
        dst_ref = expected[NEW_PREFIX + suffix]
        assert src_t.shape == dst_ref.shape and src_t.dtype == dst_ref.dtype, suffix
        out[NEW_PREFIX + suffix] = src_t.detach().clone()

    # Final linear layer: exact exponential-family Gaussian projection over bin centers
    w_out = state["value_head.1.2.weight"].detach().to(torch.float64)  # (1, d)
    b_out = float(state["value_head.1.2.bias"].detach().to(torch.float64).item())
    centers = net.val_centers.detach().cpu().to(torch.float64)  # (K,)
    v0 = float(min(max(b_out, cfg.value_vmin), cfg.value_vmax))
    u = centers - v0  # (K,)
    inv_var = 1.0 / (sigma * sigma)
    w_dist = (u * inv_var).unsqueeze(1) * w_out  # (K, d)
    b_dist = (u * inv_var) * (b_out - v0) - 0.5 * (u * u) * inv_var  # (K,)
    out["val_dist.1.2.weight"] = w_dist.to(expected["val_dist.1.2.weight"].dtype)
    out["val_dist.1.2.bias"] = b_dist.to(expected["val_dist.1.2.bias"].dtype)
    return {k: out[k] for k in expected}


def migrate_hlgauss_optimizer(saved: dict[str, Any], net: SpireNet,
                              optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Preserve every Adam step and moment except for ``val_dist.*``, whose entries are zeroed (step 0).

    ``value_head.*`` and ``val_dist.*`` occupy the exact same parameter positions in their respective decay /
    no-decay groups (2 weights in decay, 2 biases in no-decay).
    """
    assert net.val_dist is not None, "HL-Gauss optimizer migration requires model.value_bins > 0"
    assert not optimizer.state, "HL-Gauss migration must precede optimizer updates"
    names = {id(p): name for name, p in net.named_parameters()}
    current = optimizer.state_dict()
    assert len(saved["param_groups"]) == len(current["param_groups"]), "migration cannot change optimizer groups"
    state: dict[int, dict[str, Tensor]] = {}
    groups = []
    step_ref: Tensor | None = None
    for e in saved["state"].values():
        step_ref = e["step"]
        break
    val_ids: set[int] = set()
    for old, new, live in zip(saved["param_groups"], current["param_groups"], optimizer.param_groups):
        assert len(old["params"]) == len(new["params"]) == len(live["params"]), \
            "migration cannot change parameter group counts"
        amsgrad = bool(old["amsgrad"])
        for old_i, new_i, param in zip(old["params"], new["params"], live["params"]):
            name = names[id(param)]
            if name.startswith(NEW_PREFIX):
                val_ids.add(new_i)
                if step_ref is not None:
                    keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if amsgrad else ("exp_avg", "exp_avg_sq")
                    state[new_i] = {"step": torch.zeros_like(step_ref),
                                    **{k: torch.zeros(param.shape, dtype=param.dtype) for k in keys}}
            elif old_i in saved["state"]:
                entry = saved["state"][old_i]
                for k, v in entry.items():
                    assert k == "step" or v.shape == param.shape, f"{name}.{k}: Adam state shape mismatch"
                state[new_i] = {k: v.detach().clone() for k, v in entry.items()}
        groups.append({**old, "params": new["params"]})
    assert len(val_ids) == 4, f"expected 4 val_dist parameters in optimizer, found {len(val_ids)}"
    return {"state": state, "param_groups": groups}


def migrate_checkpoint(src: Path, dst: Path, *, value_bins: int, value_vmin: float = -0.5,
                       value_vmax: float = 2.5, value_sigma: float = 0.05,
                       init_sigma: float | None = None, vocab_path: Path | None = None,
                       catalog_path: Path | None = None) -> dict[str, Any]:
    """Migrate a full ``spire.train`` checkpoint on CPU from scalar value head to HL-Gauss."""
    from .evalpanel import load_catalog
    from .learner import _param_groups
    from .tokenize import Tokenizer, TokenizerConfig
    from .train import build
    from .vocab import Vocab, make_synthetic_vocab

    assert not dst.exists(), f"{dst} already exists; migration never overwrites a checkpoint"
    ck = torch.load(src, map_location="cpu", weights_only=True)
    assert ck["format"] == 1
    extra = dict(ck["extra"])
    before = build(ModelConfig, extra["model_config"])
    assert before.value_bins == 0, f"source checkpoint already has value_bins={before.value_bins}"
    after = replace(before, value_bins=value_bins, value_vmin=value_vmin, value_vmax=value_vmax,
                    value_sigma=value_sigma)
    vocab_src = str(vocab_path) if vocab_path is not None else extra["vocab"]
    catalog_src = str(catalog_path) if catalog_path is not None else extra["catalog"]
    vocab = Vocab(make_synthetic_vocab()) if vocab_src == "synthetic" else Vocab.load(vocab_src)
    assert vocab.hash == extra["vocab_hash"], "source vocabulary hash mismatch"
    catalog = load_catalog(catalog_src, vocab)
    assert (0 if catalog is None else catalog.hash) == extra["catalog_hash"], "source catalog hash mismatch"
    tok = Tokenizer(vocab, build(TokenizerConfig, extra["tokenizer"]))
    net = SpireNet(tok.layout, after, catalog).eval()
    migrated = {k: migrate_hlgauss_weights(ck[k], net, init_sigma=init_sigma) for k in ("model", "ema")}
    opt = torch.optim.AdamW(_param_groups(net, float(ck["config"]["weight_decay"])))
    optimizer = migrate_hlgauss_optimizer(ck["optimizer"], net, opt)
    opt.load_state_dict(optimizer)
    extra["model_config"] = model_config_dict(after)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, **migrated, "optimizer": optimizer, "extra": extra}, dst)
    return {"source": str(src), "output": str(dst), "value_bins": value_bins,
            "value_vmin": value_vmin, "value_vmax": value_vmax, "value_sigma": value_sigma,
            "init_sigma": default_init_sigma(after) if init_sigma is None else init_sigma}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate a scalar-value checkpoint to HL-Gauss value_bins")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--value-bins", type=int, default=51)
    ap.add_argument("--vmin", type=float, default=-0.5)
    ap.add_argument("--vmax", type=float, default=2.5)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--init-sigma", type=float, default=None)
    ap.add_argument("--vocab", type=Path, default=None)
    ap.add_argument("--catalog", type=Path, default=None)
    a = ap.parse_args(argv)
    info = migrate_checkpoint(a.checkpoint, a.output, value_bins=a.value_bins, value_vmin=a.vmin,
                              value_vmax=a.vmax, value_sigma=a.sigma, init_sigma=a.init_sigma,
                              vocab_path=a.vocab, catalog_path=a.catalog)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
