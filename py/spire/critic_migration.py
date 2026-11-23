"""Strict one-time critic migration (notes/critic/contract.md 5); old weights and Adam histories stay exact.

The migrated network gains only the tower ``critic.*`` (build-v1.md 3: no ``death_head`` in v1), drawn from its own
generator exactly as ``SpireNet`` draws it. ``forward`` never calls the tower, so logits, value and the aux outputs
of the migrated network are those of the source bit for bit. Optimizer: the source's groups (the trunk) keep every
step and moment and their settings; the tower's two groups follow with zero moments and step 0.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from .critic import CriticTower
from .model import SpireNet

PREFIX = "critic."


def tower_init_state(net: SpireNet) -> dict[str, Tensor]:
    """The tower's initial weights, ``critic.*`` keys: a fresh :class:`CriticTower` of ``net``'s config (its own
    generator; the global RNG is untouched)."""
    cfg = net.config
    assert cfg.critic is not None and net.critic is not None, "critic migration requires model.critic"
    with torch.random.fork_rng(devices=[]):
        tower = CriticTower(cfg.critic, cfg.d_model, cfg.d_cond, cfg.init_std)
    return {PREFIX + k: v.detach().clone() for k, v in tower.state_dict().items()}


def migrate_critic_weights(state: Mapping[str, Tensor], net: SpireNet) -> dict[str, Tensor]:
    """Online or EMA weights of a model without the tower -> the critic model's (every old key exact)."""
    expected = net.state_dict()
    new = tower_init_state(net)
    assert set(new) == {k for k in expected if k.startswith(PREFIX)}, "tower keys differ from the network's"
    assert not any(k.startswith(PREFIX) for k in state), "the source already has a critic tower"
    assert set(state) == set(expected) - set(new), "migration requires exactly the pre-critic model keys"
    for k, v in state.items():
        assert v.shape == expected[k].shape and v.dtype == expected[k].dtype, f"{k}: migration cannot change it"
    out = {k: v.detach().clone() for k, v in state.items()}
    out.update({k: v.to(expected[k].dtype) for k, v in new.items()})
    return {k: out[k] for k in expected}


def migrate_critic_optimizer(saved: dict[str, Any], net: SpireNet, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """``optimizer`` is built with :func:`spire.critic.optimizer_groups` of ``net``: its trunk groups are the saved
    groups (same parameters in the same order), then the tower's groups. Every saved entry and group setting is
    kept; the tower's parameters get zero moments and step 0 and their groups the current settings."""
    assert not optimizer.state, "critic migration must precede optimizer updates"
    current = optimizer.state_dict()
    live = optimizer.param_groups
    n_old = len(saved["param_groups"])
    assert [bool(g.get("critic")) for g in live] == [False] * n_old + [True] * (len(live) - n_old) and \
        len(live) > n_old, "the optimizer must hold the saved trunk groups, then the tower's own groups"
    tower = {id(p) for p in net.critic.parameters()}  # type: ignore[union-attr]
    groups = []
    for old, new, lv in zip(saved["param_groups"], current["param_groups"][:n_old], live[:n_old]):
        assert len(old["params"]) == len(new["params"]) and not any(id(p) in tower for p in lv["params"]), \
            "migration cannot change the trunk's optimizer groups"
        assert old["params"] == new["params"], "trunk parameter ids must enumerate in the saved order"
        groups.append({**old, "params": new["params"]})
    groups += current["param_groups"][n_old:]
    state: dict[int, dict[str, Tensor]] = {k: {n: v.detach().clone() for n, v in e.items()}
                                           for k, e in saved["state"].items()}
    assert state, "the source optimizer has no state"
    step = next(iter(saved["state"].values()))["step"]
    amsgrad = [bool(g["amsgrad"]) for g in current["param_groups"][n_old:]]
    for g, lv, ams in zip(current["param_groups"][n_old:], live[n_old:], amsgrad):
        for i, p in zip(g["params"], lv["params"]):
            assert i not in state and id(p) in tower
            keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if ams else ("exp_avg", "exp_avg_sq")
            state[i] = {"step": torch.zeros_like(step), **{k: torch.zeros_like(p, device="cpu") for k in keys}}
    return {"state": state, "param_groups": groups}
