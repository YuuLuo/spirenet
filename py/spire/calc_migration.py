"""Strict one-time calc migration (notes/calc/contract.md 4); old weights and Adam histories stay exact.

The migrated network takes the TURN / FOE / ROUTE kid rows: ``type_emb`` grows to 24 rows (old rows exact, the new
rows drawn from a fixed generator), ``calc_in`` is drawn from the same generator and ``calc_gate`` is 0, so every
calc kid adds exactly +-0 to its owner token until training opens the gates. The source may have the forecast on
or off (it is kept); forecast first, then calc.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from .model import CALC_INIT_SEED, SpireNet, calc_type_rows
from .tokenize import N_ENT_CALC, TokenLayout

ADDED = ("calc_in.weight", "calc_gate")


def _extend(name: str, value: Tensor, shape: torch.Size, old_rows: int, fill: Tensor | None) -> Tensor:
    """``value`` unchanged, or for ``type_emb.weight`` with the new rows appended (``fill``; zeros if None)."""
    if name == "type_emb.weight":
        assert value.shape == (old_rows, shape[1]) and shape[0] > old_rows, \
            f"migration requires the {old_rows}-row type embedding, got {tuple(value.shape)}"
        rows = torch.zeros(shape[0] - old_rows, shape[1], dtype=value.dtype) if fill is None else fill
        return torch.cat([value.detach(), rows.to(value.dtype)])
    assert value.shape == shape, f"{name}: migration cannot change existing shapes"
    return value.detach().clone()


def calc_init(net: SpireNet, layout: TokenLayout) -> tuple[Tensor, Tensor]:
    """(new type_emb rows, calc_in.weight) as SpireNet draws them (generator CALC_INIT_SEED, rows first)."""
    d = net.config.d_model
    old = calc_type_rows(layout.forecast)
    g = torch.Generator().manual_seed(CALC_INIT_SEED)
    assert net.calc_in is not None
    rows = torch.empty(N_ENT_CALC - old, d).normal_(0.0, net.config.init_std, generator=g)
    w = torch.empty(net.calc_in.weight.shape).normal_(0.0, net.config.init_std, generator=g)
    return rows, w


def migrate_calc_weights(state: Mapping[str, Tensor], net: SpireNet, layout: TokenLayout) -> dict[str, Tensor]:
    """Online or EMA weights of a pre-calc model -> the calc model's (outputs unchanged, gates 0)."""
    assert layout.calc and net.calc_in is not None and net.calc_gate is not None, \
        "calc migration requires tokenizer.calc=true"
    expected = net.state_dict()
    assert set(state) == set(expected) - set(ADDED), "migration requires exactly the pre-calc model keys"
    old = calc_type_rows(layout.forecast)
    rows, w = calc_init(net, layout)
    out = {name: _extend(name, value, expected[name].shape, old, rows if name == "type_emb.weight" else None)
           for name, value in state.items()}
    out["calc_in.weight"] = w.to(expected["calc_in.weight"].dtype)
    out["calc_gate"] = torch.zeros_like(expected["calc_gate"], device="cpu")
    return {k: out[k] for k in expected}


def migrate_calc_optimizer(saved: dict[str, Any], net: SpireNet, layout: TokenLayout,
                           optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Preserve every old Adam step/moment (remapped by parameter name); the type_emb moments get zero rows and the
    new parameters zero moments with step 0."""
    assert layout.calc
    assert not optimizer.state, "calc migration must precede optimizer updates"
    old_rows = calc_type_rows(layout.forecast)
    names = {id(p): name for name, p in net.named_parameters()}
    current = optimizer.state_dict()
    assert len(saved["param_groups"]) == len(current["param_groups"]), "migration cannot change optimizer groups"
    remap: dict[int, tuple[int, str, torch.nn.Parameter, bool]] = {}
    added: list[tuple[int, torch.nn.Parameter, bool]] = []
    groups = []
    for old, new, live in zip(saved["param_groups"], current["param_groups"], optimizer.param_groups):
        existing = []
        for after, parameter in zip(new["params"], live["params"]):
            if names[id(parameter)] in ADDED:
                added.append((after, parameter, bool(old["amsgrad"])))
            else:
                existing.append((after, names[id(parameter)], parameter, bool(old["amsgrad"])))
        assert len(old["params"]) == len(existing), "migration cannot change existing optimizer groups"
        for before, target in zip(old["params"], existing):
            assert before not in remap, "duplicate optimizer parameter ID"
            remap[before] = target
        groups.append({**old, "params": new["params"]})
    assert len(added) == len(ADDED)
    assert set(saved["state"]) <= set(remap), "optimizer state references an unknown parameter"
    state: dict[int, dict[str, Tensor]] = {}
    step: Tensor | None = None
    for before, entry in saved["state"].items():
        after, name, parameter, amsgrad = remap[before]
        expected = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if amsgrad else set())
        assert set(entry) == expected, f"{name}: invalid Adam state keys"
        assert isinstance(entry["step"], Tensor) and entry["step"].ndim == 0, f"{name}: invalid Adam step"
        state[after] = {key: (value.detach().clone() if key == "step" else
                             _extend(name, value, parameter.shape, old_rows, None)) for key, value in entry.items()}
        step = entry["step"]
    assert step is not None, "the source optimizer has no state"
    for new_id, new_parameter, new_amsgrad in added:
        keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if new_amsgrad else ("exp_avg", "exp_avg_sq")
        state[new_id] = {"step": torch.zeros_like(step),
                         **{key: torch.zeros_like(new_parameter, device="cpu") for key in keys}}
    return {"state": state, "param_groups": groups}
