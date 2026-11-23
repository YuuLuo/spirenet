"""Strict one-time inputs-v2a migration (notes/inputs2/v2a.md 3); old weights and Adam histories stay exact.

A calc network (``map_tokens`` "frame", no route summary, no second-boss table) becomes the network of the config
``map_tokens = "no_combat"``, ``route_summary = true``, ``second_boss_table = true``:

1. ``ids.weight`` (the only tensor indexed by learned row) gains the 256-row region ``encounter#second`` after the
   last learned region; its rows are a copy of the ``encounter`` region's rows (the static-prior regions, computed
   from the catalog, shift by 256 with no weights of their own);
2. ``type_emb`` gains row 24 and ``ahead_in`` is new, both from the fixed generator of contract 2.4
   (``model.ahead_init``); ``ahead_gate`` = 0;
3. nothing else changes, byte for byte.

Optimizer: the Adam moments of the copied rows are copied with them, ``type_emb``'s new row and the new parameters
get zero moments (step 0); steps, schedules and the param groups' options are unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
from torch import Tensor

from .model import SpireNet, ahead_init
from .tokenize import N_ENT_CALC, Ent, TokenizerConfig, TokenLayout

ADDED = ("ahead_in.weight", "ahead_gate")
KEYS = ("map_tokens", "route_summary", "second_boss_table")
SECOND = "encounter#second"


def config_before(after: TokenizerConfig) -> TokenizerConfig:
    """The source network's tokenizer config of a migrated config (the three keys off)."""
    assert (after.map_tokens, after.route_summary, after.second_boss_table) == ("no_combat", True, True), \
        "inputs2 migration targets map_tokens = 'no_combat', route_summary and second_boss_table"
    return replace(after, map_tokens="frame", route_summary=False, second_boss_table=False)


def check_source(cfg: TokenizerConfig) -> None:
    """Asserts that a source checkpoint's tokenizer config can be migrated: ``calc`` and ``pool_children`` on,
    ``map_tokens`` absent (its default) or "frame", no route summary, no second-boss table."""
    assert cfg.calc and cfg.pool_children, "inputs2 migration needs a calc network (tokenizer.calc, pool_children)"
    assert (cfg.map_tokens, cfg.route_summary, cfg.second_boss_table) == ("frame", False, False), \
        f"already migrated or not a v1-map network: {[getattr(cfg, k) for k in KEYS]}"


def _region_copy(before: TokenLayout, after: TokenLayout) -> tuple[int, int, int]:
    """(old learned rows, first row of the ``encounter`` region, its rows) - asserting the layout change of contract
    4: one region ``encounter#second`` of the encounter's size appended after the old learned rows."""
    off, rows = after.regions["encounter"]
    assert before.regions["encounter"] == (off, rows)
    assert {k: v for k, v in after.regions.items() if k != SECOND} == before.regions, "existing regions must not move"
    assert after.regions[SECOND] == (before.n_learned, rows) and after.n_learned == before.n_learned + rows
    return before.n_learned, off, rows


def _extend(name: str, value: Tensor, shape: torch.Size, copy: tuple[int, int, int], fill: Tensor | None) -> Tensor:
    """``value`` unchanged, except: ``ids.weight`` (weights and Adam moments alike) with a copy of its ``encounter``
    rows appended; ``type_emb.weight`` with row 24 appended (``fill``; zeros for the Adam moments)."""
    if name == "ids.weight":
        n_old, off, rows = copy
        assert value.shape == (n_old, shape[1]) and shape[0] == n_old + rows, f"{name}: {tuple(value.shape)}"
        return torch.cat([value.detach(), value.detach()[off : off + rows].clone()])
    if name == "type_emb.weight":
        assert value.shape == (N_ENT_CALC, shape[1]) and shape[0] == N_ENT_CALC + 1, \
            f"migration requires the {N_ENT_CALC}-row type embedding, got {tuple(value.shape)}"
        row = torch.zeros(1, shape[1], dtype=value.dtype) if fill is None else fill.reshape(1, -1).to(value.dtype)
        return torch.cat([value.detach(), row])
    assert value.shape == shape, f"{name}: migration cannot change existing shapes"
    return value.detach().clone()


def migrate_inputs2_weights(state: Mapping[str, Tensor], net: SpireNet, before: TokenLayout,
                            after: TokenLayout) -> dict[str, Tensor]:
    """Online or EMA weights of a calc network -> the inputs-v2a network's (outputs unchanged, ``ahead_gate`` 0)."""
    assert after.route_summary and after.second_boss_table and net.ahead_in is not None and net.ahead_gate is not None
    assert not before.route_summary and not before.second_boss_table and before.calc
    expected = net.state_dict()
    assert set(state) == set(expected) - set(ADDED), "migration requires exactly the pre-inputs2 model keys"
    copy = _region_copy(before, after)
    row, w = ahead_init(net.config.d_model, net.ahead_in.weight.shape, net.config.init_std)
    out = {name: _extend(name, value, expected[name].shape, copy, row if name == "type_emb.weight" else None)
           for name, value in state.items()}
    out["ahead_in.weight"] = w.to(expected["ahead_in.weight"].dtype)
    out["ahead_gate"] = torch.zeros_like(expected["ahead_gate"], device="cpu")
    assert torch.equal(out["type_emb.weight"][int(Ent.AHEAD)], row.to(out["type_emb.weight"].dtype))
    return {k: out[k] for k in expected}


def migrate_inputs2_optimizer(saved: dict[str, Any], net: SpireNet, before: TokenLayout, after: TokenLayout,
                              optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Every old Adam step / moment preserved (remapped by parameter name); ``ids.weight`` moments with the copied
    rows' moments appended, ``type_emb`` moments with a zero row, the new parameters zero moments and step 0."""
    assert not optimizer.state, "inputs2 migration must precede optimizer updates"
    copy = _region_copy(before, after)
    names = {id(p): name for name, p in net.named_parameters()}
    current = optimizer.state_dict()
    assert len(saved["param_groups"]) == len(current["param_groups"]), "migration cannot change optimizer groups"
    remap: dict[int, tuple[int, str, torch.nn.Parameter, bool]] = {}
    added: list[tuple[int, torch.nn.Parameter, bool]] = []
    groups = []
    for old, new, live in zip(saved["param_groups"], current["param_groups"], optimizer.param_groups):
        existing = []
        for after_id, parameter in zip(new["params"], live["params"]):
            if names[id(parameter)] in ADDED:
                added.append((after_id, parameter, bool(old["amsgrad"])))
            else:
                existing.append((after_id, names[id(parameter)], parameter, bool(old["amsgrad"])))
        assert len(old["params"]) == len(existing), "migration cannot change existing optimizer groups"
        for before_id, target in zip(old["params"], existing):
            assert before_id not in remap, "duplicate optimizer parameter ID"
            remap[before_id] = target
        groups.append({**old, "params": new["params"]})
    assert len(added) == len(ADDED)
    assert set(saved["state"]) <= set(remap), "optimizer state references an unknown parameter"
    state: dict[int, dict[str, Tensor]] = {}
    step: Tensor | None = None
    for before_id, entry in saved["state"].items():
        after_id, name, parameter, amsgrad = remap[before_id]
        expected = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if amsgrad else set())
        assert set(entry) == expected, f"{name}: invalid Adam state keys"
        assert isinstance(entry["step"], Tensor) and entry["step"].ndim == 0, f"{name}: invalid Adam step"
        state[after_id] = {key: (value.detach().clone() if key == "step" else
                                 _extend(name, value, parameter.shape, copy, None)) for key, value in entry.items()}
        step = entry["step"]
    assert step is not None, "the source optimizer has no state"
    for new_id, new_parameter, new_amsgrad in added:
        keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if new_amsgrad else ("exp_avg", "exp_avg_sq")
        state[new_id] = {"step": torch.zeros_like(step),
                         **{key: torch.zeros_like(new_parameter, device="cpu") for key in keys}}
    return {"state": state, "param_groups": groups}
