"""Quiet combat sampling: a per-step behaviour temperature in fights, exact PPO accounting
(notes/next/temperature-spec.md v1).

* A COMBAT STEP is a decision taken while the frame's IN_COMBAT fact is set and whose decision kind is COMBAT or
  CARD_SELECT (the predicate of alt/solo/noise/noise_eval.py). :func:`combat_steps` is that predicate; the policy
  evaluates it on the global token of the sample it samples from (the tokenizer's copy of the frame's GLOBAL values),
  the actor on the frame's facts, and the actor asserts that both agree at every decision.
* T(step) = ``ppo.combat_temperature`` on a combat step, 1.0 elsewhere (:func:`step_temperatures`); the behaviour is
  q = (1 - eps_kind) softmax(z / T) + eps_kind reference. The actor (:class:`spire.actor.TorchPolicy`) decides T and
  stores it per step in the rollout column ``beh_temp``; the learner recomputes log q(a) with the STORED temperature
  (:func:`tempered_log_probs`), never with the config value.
* Entropy bonus, uniform barrier, kickstart, value / auxiliary losses, greedy evaluation: untempered z, unchanged.
"""
from __future__ import annotations

import numpy as np
from torch import Tensor

from .distributions import legal_uniform_log_probs
from .schema import Kind

COMBAT_KINDS = (int(Kind.COMBAT), int(Kind.CARD_SELECT))


def combat_steps(kind: np.ndarray, in_combat: np.ndarray) -> np.ndarray:
    """Combat-step flags: IN_COMBAT set and decision kind COMBAT or CARD_SELECT (noise_eval's predicate)."""
    return (np.asarray(in_combat) != 0) & np.isin(np.asarray(kind), COMBAT_KINDS)


def step_temperatures(combat: np.ndarray, combat_temperature: float) -> np.ndarray:
    """T(step) per step (float32): ``combat_temperature`` on combat steps, 1.0 elsewhere."""
    return np.where(combat, np.float32(combat_temperature), np.float32(1.0)).astype(np.float32)


def tempered_log_probs(logits: Tensor, beh_temp: Tensor, epsilon: Tensor | None = None,
                       reference: Tensor | None = None) -> Tensor:
    """log q of every row for q = (1 - eps) softmax(logits / beh_temp) + eps reference (0 at illegal rows)."""
    return legal_uniform_log_probs(logits / beh_temp[:, None], epsilon, reference)


def check_temperature(combat_temperature: float, advice: tuple, taste_on: bool, decoupled_clip_anchor: bool) -> None:
    """Start-up rules of ``ppo.combat_temperature`` (spec 3): > 0; with a value other than 1, no advice, no taste
    (their terms were written for T = 1) and no decoupled clip anchor (it re-derives log q from the untempered z)."""
    assert np.isfinite(combat_temperature) and combat_temperature > 0, "ppo.combat_temperature must be finite and > 0"
    if combat_temperature != 1.0:
        assert not advice, "ppo.combat_temperature != 1 requires ppo.advice to be empty"
        assert not taste_on, "ppo.combat_temperature != 1 requires taste to be off"
        assert not decoupled_clip_anchor, "ppo.combat_temperature != 1 requires ppo.decoupled_clip_anchor = false"
