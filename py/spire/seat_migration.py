"""Explicit one-time single-player -> H5 parameter migration, without new input features."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from .model import SpireNet
from .schema import IntentCol, MapNodeCol, PowerCol, ViewMode
from .tokenize import HISTORY_STATS, SPECS, Ent, Mode, TokenLayout


def seat_column_groups(layout: TokenLayout, d_id: int) -> dict[str, tuple[int, ...]]:
    """Every per-seat numeric group already present in the Stage A feature layout."""
    groups: dict[str, tuple[int, ...]] = {}
    for name, ent, mode, src, start in (
        ("power_visibility", Ent.POWER, Mode.BIT, int(PowerCol.VISIBLE_MASK), 0),
        ("map_votes", Ent.MAP_NODE, Mode.BIT, int(MapNodeCol.FLAGS), 4),
    ):
        cols = tuple(d_id + layout.num_offsets[ent] + j for j, f in enumerate(SPECS[ent].nums)
                     if f.src == src and f.mode == mode and start <= f.arg < start + 4)
        assert len(cols) == 4, name
        groups[name] = cols
    for name, mode in (("intent_damage", Mode.SYM), ("intent_missing", Mode.MISS),
                       ("intent_total_damage", Mode.SYMPROD)):
        cols = tuple(d_id + layout.num_offsets[Ent.INTENT] + j
                     for j, f in enumerate(SPECS[Ent.INTENT].nums)
                     if f.mode == mode and isinstance(f.src, int)
                     and IntentCol.DMG_SEAT0 <= f.src <= IntentCol.DMG_SEAT3)
        assert len(cols) == 4, name
        groups[name] = cols
    return groups


def migrate_seat_symmetry_weights(state: Mapping[str, Tensor], net: SpireNet, layout: TokenLayout,
                                  *, source_view_mode: int = ViewMode.TEAM) -> dict[str, Tensor]:
    """Copy seat-zero semantics into all live seats; preserve legacy SP missing constants.

    Requires the old architecture exactly and an explicitly enabled target architecture.
    This function does not mutate its input. The matching optimizer migration preserves
    existing histories and copies the source moments for every copied row and column.
    """
    assert net.config.seat_symmetry and net.multiseat_intent_missing is not None, \
        "seat symmetry migration requires model.seat_symmetry=true"
    assert source_view_mode in (ViewMode.TEAM, ViewMode.SEAT)
    key = "multiseat_intent_missing"
    expected = net.state_dict()
    assert set(state) == set(expected) - {key}, "migration requires exactly the pre-seat-symmetry model keys"
    assert all(state[k].shape == expected[k].shape for k in state), "migration cannot change existing shapes"
    out = {k: _copy_seat_tensor(k, v, net, layout, source_view_mode) for k, v in state.items()}
    missing = seat_column_groups(layout, net.config.d_id)["intent_missing"]
    out[key] = out["in_proj.weight"][:, missing[:1]].repeat(1, 4)
    if net.history_proj is not None:
        # history.stats[19:26] stores the raw action. stats[20] is its numeric seat
        # ordinal: it was always zero in SP, so its symmetric counterpart is zero.
        out["history_proj.weight"][:, -HISTORY_STATS + 20].zero_()
    return out


def _copy_seat_tensor(name: str, value: Tensor, net: SpireNet, layout: TokenLayout,
                      source_view_mode: int) -> Tensor:
    """The same row/column transfer for a weight or an Adam moment."""
    out = value.detach().clone()
    if name == "ids.weight":
        for region, source, targets in (("p:seat", 1, (2, 3, 4)), ("p:n_seats", 1, (2, 3, 4)),
                                       ("p:view_mode", source_view_mode, (1 - source_view_mode,))):
            off, _ = layout.regions[region]
            for target in targets:
                out[off + target].copy_(out[off + source])
    elif name == "cond_seats.weight":
        for seat_count in (2, 3, 4):
            out[seat_count].copy_(out[1])
    elif name == "in_proj.weight":
        for group, cols in seat_column_groups(layout, net.config.d_id).items():
            if group != "intent_missing":
                for target in cols[1:]:
                    out[:, target].copy_(out[:, cols[0]])
    return out


def migrate_seat_symmetry_optimizer(saved: dict[str, Any], net: SpireNet, layout: TokenLayout,
                                    optimizer: torch.optim.Optimizer, *,
                                    source_view_mode: int = ViewMode.TEAM,
                                    reset: bool = False) -> dict[str, Any]:
    """Remap Adam parameter IDs and transfer source histories alongside their weights.

    Existing per-parameter steps and group settings persist. The new missing-indicator
    matrix inherits in_proj's step and source-column moments. A source without Adam
    state remains uninitialized, including its copy. Reset explicitly discards all state.
    The optional history ordinal coefficient keeps its old moments even though the
    weight migration zeroes it: no history was copied from another parameter there.
    """
    assert net.config.seat_symmetry and net.multiseat_intent_missing is not None
    assert source_view_mode in (ViewMode.TEAM, ViewMode.SEAT)
    assert not optimizer.state, "seat migration must precede optimizer updates"
    names = {id(p): name for name, p in net.named_parameters()}
    current = optimizer.state_dict()
    old_groups, new_groups = saved["param_groups"], current["param_groups"]
    assert len(old_groups) == len(new_groups), "migration cannot change optimizer groups"
    remap: dict[int, tuple[int, str, torch.nn.Parameter, bool]] = {}
    groups = []
    new_matrix_id: int | None = None
    for old, new, live in zip(old_groups, new_groups, optimizer.param_groups):
        existing = []
        for after, parameter in zip(new["params"], live["params"]):
            if parameter is net.multiseat_intent_missing:
                assert new_matrix_id is None
                new_matrix_id = after
            else:
                existing.append((after, names[id(parameter)], parameter, bool(old["amsgrad"])))
        assert len(old["params"]) == len(existing), "migration cannot change existing optimizer groups"
        for before, target in zip(old["params"], existing):
            assert before not in remap, "duplicate optimizer parameter ID"
            remap[before] = target
        groups.append({**old, "params": new["params"]})
    assert new_matrix_id is not None
    assert set(saved["state"]) <= set(remap), "optimizer state references an unknown parameter"
    state: dict[int, dict[str, Tensor]] = {}
    missing = seat_column_groups(layout, net.config.d_id)["intent_missing"]
    for before, entry in saved["state"].items():
        after, name, parameter, amsgrad = remap[before]
        expected = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if amsgrad else set())
        assert set(entry) == expected, f"{name}: invalid Adam state keys"
        assert isinstance(entry["step"], Tensor) and entry["step"].ndim == 0, f"{name}: invalid Adam step"
        assert all(isinstance(value, Tensor) and value.shape == parameter.shape
                   for key, value in entry.items() if key != "step"), f"{name}: invalid Adam moment shape"
        state[after] = {key: (value.detach().clone() if key == "step" else
                             _copy_seat_tensor(name, value, net, layout, source_view_mode))
                        for key, value in entry.items()}
        if name == "in_proj.weight":
            state[new_matrix_id] = {key: (value.detach().clone() if key == "step" else
                                         value[:, missing[:1]].repeat(1, 4))
                                    for key, value in entry.items()}
    return {"state": {} if reset else state, "param_groups": groups}
