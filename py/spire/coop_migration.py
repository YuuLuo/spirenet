"""Strict one-time Stage B migration; old weights and Adam histories stay exact."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from .model import SpireNet
from .tokenize import TokenLayout


def _extend(name: str, value: Tensor, shape: torch.Size, old_width: int) -> Tensor:
    if name == "in_proj.weight":
        assert value.shape == (shape[0], old_width), "migration requires the legacy input width"
        out = value.new_zeros(shape)
        out[:, :old_width].copy_(value)
        return out
    assert value.shape == shape, f"{name}: migration cannot change existing shapes"
    return value.detach().clone()


def migrate_coop_features_weights(state: Mapping[str, Tensor], net: SpireNet,
                                  layout: TokenLayout) -> dict[str, Tensor]:
    """Append zero numeric columns and a separate zero seat-screen embedding.

    The separate embedding leaves the old ID table, static rows and embedding-bag
    reduction intact. The model applies new numeric columns as a residual, so the
    legacy projection keeps the exact same reduction width.
    """
    assert layout.coop_features or layout.mend_target, \
        "co-op migration requires tokenizer.coop_features=true or tokenizer.mend_target=true"
    expected = net.state_dict()
    added = ({"coop_screen.weight"} if layout.coop_features else set()) | \
            ({"mend_target_gate"} if layout.mend_target else set())
    assert set(state) == set(expected) - added, "migration requires exactly the pre-coop-features model keys"
    old_width = net.config.d_id + layout.legacy_n_wide
    out = {name: _extend(name, value, expected[name].shape, old_width) for name, value in state.items()}
    for key in added:
        out[key] = torch.zeros_like(expected[key], device="cpu")
    return out


def migrate_coop_features_optimizer(saved: dict[str, Any], net: SpireNet, layout: TokenLayout,
                                    optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Preserve every old Adam step/moment; new columns and screen moments are zero."""
    assert layout.coop_features or layout.mend_target
    assert not optimizer.state, "co-op migration must precede optimizer updates"
    names = {id(p): name for name, p in net.named_parameters()}
    current = optimizer.state_dict()
    assert len(saved["param_groups"]) == len(current["param_groups"]), "migration cannot change optimizer groups"
    remap: dict[int, tuple[int, str, torch.nn.Parameter, bool]] = {}
    added: list[tuple[int, torch.nn.Parameter, bool]] = []
    groups = []
    for old, new, live in zip(saved["param_groups"], current["param_groups"], optimizer.param_groups):
        existing = []
        for after, parameter in zip(new["params"], live["params"]):
            if names[id(parameter)] in {"coop_screen.weight", "mend_target_gate"}:
                added.append((after, parameter, bool(old["amsgrad"])))
            else:
                existing.append((after, names[id(parameter)], parameter, bool(old["amsgrad"])))
        assert len(old["params"]) == len(existing), "migration cannot change existing optimizer groups"
        for before, target in zip(old["params"], existing):
            assert before not in remap, "duplicate optimizer parameter ID"
            remap[before] = target
        groups.append({**old, "params": new["params"]})
    assert len(added) == int(layout.coop_features) + int(layout.mend_target)
    assert set(saved["state"]) <= set(remap), "optimizer state references an unknown parameter"
    old_width = net.config.d_id + layout.legacy_n_wide
    state: dict[int, dict[str, Tensor]] = {}
    for before, entry in saved["state"].items():
        after, name, parameter, amsgrad = remap[before]
        expected = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if amsgrad else set())
        assert set(entry) == expected, f"{name}: invalid Adam state keys"
        assert isinstance(entry["step"], Tensor) and entry["step"].ndim == 0, f"{name}: invalid Adam step"
        assert all(isinstance(value, Tensor) for value in entry.values()), f"{name}: invalid Adam tensors"
        state[after] = {key: (value.detach().clone() if key == "step" else
                             _extend(name, value, parameter.shape, old_width)) for key, value in entry.items()}
        if name == "ids.weight":
            for new_id, new_parameter, new_amsgrad in added:
                keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if new_amsgrad else ("exp_avg", "exp_avg_sq")
                state[new_id] = {"step": torch.zeros_like(entry["step"]),
                                 **{key: torch.zeros_like(new_parameter, device="cpu") for key in keys}}
    return {"state": state, "param_groups": groups}
