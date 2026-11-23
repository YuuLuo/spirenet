"""Strict one-time forecast migration (notes/forecast/contract.md 4); old weights and Adam histories stay exact.

The migrated network takes the FORECAST kid rows: ``type_emb`` gets row ``Ent.FORECAST`` (a copy of the INTENT
row), ``forecast_in`` is drawn from a fixed generator and ``forecast_gate`` is 0, so every FORECAST kid adds
exactly +-0 to its creature token until training opens the gate.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from .model import SpireNet
from .tokenize import N_ENT_LEGACY, Ent, TokenLayout

FORECAST_INIT_SEED = 0x46435354  # "FCST"
ADDED = ("forecast_in.weight", "forecast_gate")


def _extend(name: str, value: Tensor, shape: torch.Size, fill: Tensor | None) -> Tensor:
    """``value`` unchanged, or for ``type_emb.weight`` with the FORECAST row appended (``fill``; zeros if None)."""
    if name == "type_emb.weight":
        assert value.shape == (N_ENT_LEGACY, shape[1]) and shape[0] == N_ENT_LEGACY + 1, \
            "migration requires the legacy type embedding"
        row = torch.zeros_like(value[:1]) if fill is None else fill.reshape(1, -1)
        return torch.cat([value.detach(), row.to(value.dtype)])
    assert value.shape == shape, f"{name}: migration cannot change existing shapes"
    return value.detach().clone()


def migrate_forecast_weights(state: Mapping[str, Tensor], net: SpireNet, layout: TokenLayout) -> dict[str, Tensor]:
    """Online or EMA weights of a pre-forecast model -> the forecast model's (outputs unchanged, gate 0)."""
    assert layout.forecast and net.forecast_in is not None, "forecast migration requires tokenizer.forecast=true"
    assert not layout.calc, "forecast migration first, then calc (notes/calc/contract.md 4)"
    expected = net.state_dict()
    assert set(state) == set(expected) - set(ADDED), "migration requires exactly the pre-forecast model keys"
    out = {name: _extend(name, value, expected[name].shape,
                         state["type_emb.weight"][int(Ent.INTENT)] if name == "type_emb.weight" else None)
           for name, value in state.items()}
    g = torch.Generator().manual_seed(FORECAST_INIT_SEED)
    w = expected["forecast_in.weight"]
    out["forecast_in.weight"] = torch.empty(w.shape, dtype=w.dtype).normal_(0.0, net.config.init_std, generator=g)
    out["forecast_gate"] = torch.zeros_like(expected["forecast_gate"], device="cpu")
    return {k: out[k] for k in expected}


def migrate_forecast_optimizer(saved: dict[str, Any], net: SpireNet, layout: TokenLayout,
                               optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Preserve every old Adam step/moment (remapped by parameter name: ``forecast_gate`` is a direct parameter of
    the net, so it sits among the other direct parameters, not last); the type_emb moments get a zero row and the
    new parameters zero moments with step 0."""
    assert layout.forecast
    assert not optimizer.state, "forecast migration must precede optimizer updates"
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
                             _extend(name, value, parameter.shape, None)) for key, value in entry.items()}
        step = entry["step"]
    assert step is not None, "the source optimizer has no state"
    for new_id, new_parameter, new_amsgrad in added:
        keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if new_amsgrad else ("exp_avg", "exp_avg_sq")
        state[new_id] = {"step": torch.zeros_like(step),
                         **{key: torch.zeros_like(new_parameter, device="cpu") for key in keys}}
    return {"state": state, "param_groups": groups}
