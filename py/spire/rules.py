"""Rules out, version 1.2 (notes/solo/rules-out-spec.md, notes/next/p4.md): teach the removal rule, then lift it.

The interface offers outcomes; no-op paths are not offered. The cycle ban and the missing CANCEL of upgrade / enchant
prompts (commit_select) are therefore INTERFACE and mask in every mode. The one rule that chooses for the network is
the removal rule (spire.removal): ``TokenizerConfig.rules_mode`` decides what it does.

* ``"mask"`` (default): an out-of-combat removal prompt keeps the rule's row only (the base tree, bit for bit).
* ``"teach"``: the same acting distribution; the learner adds ``T_REMOVAL = -log P(rule row)`` on removal prompts,
  ``P`` = softmax of the scores over the rows the host offers minus the commit / cycle masked rows.
* ``"free"``: the removal prompt offers every such row (CANCEL and fewer-than-allowed selections included); the term
  continues with its coefficient fading to a floor (:func:`coefficient`).

The six ban slots keep one encoding in every mode (spire.cycle): positive entries = masked rows (commit CANCEL first,
then cycle rows), slot 0 negative = ``-(row + 1)`` of the removal rule's row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

RULES_MODES = ("mask", "teach", "free")
NEG = -1e9  # stands in for -inf inside the term (finite: no NaN gradients through samples without a removal prompt)
N_RULES_ACC = 4
"""Device sums of an update's removal metrics: removal prompts, P(rule row), arg-max = rule row, the term."""


@dataclass(frozen=True)
class RulesConfig:
    """``ppo.rules``: the removal teach term's coefficient and its fade in ``"free"``."""

    coef_removal: float = 0.05
    fade_updates: int = 1000  # "free": coef_removal fades to floor_removal x coef_removal over this many updates
    floor_removal: float = 0.1

    def __post_init__(self) -> None:
        assert self.coef_removal >= 0, "ppo.rules.coef_removal must be >= 0"
        assert self.fade_updates > 0, "ppo.rules.fade_updates must be positive"
        assert 0 <= self.floor_removal <= 1, "ppo.rules.floor_removal must be in [0, 1]"


def coefficient(cfg: RulesConfig, mode: str, free_updates: int | None) -> float:
    """The removal term's coefficient of one update. ``free_updates``: in ``"free"`` the number of updates before this
    one since the first update in ``"free"`` (0 at that first update), else None. In ``"free"``:
    ``coef_removal x (1 - (1 - floor_removal) x min(1, free_updates / fade_updates))``."""
    assert mode in RULES_MODES, mode
    if mode == "mask":
        assert free_updates is None
        return 0.0
    if mode == "teach":
        assert free_updates is None
        return cfg.coef_removal
    assert free_updates is not None and free_updates >= 0, free_updates
    f = min(1.0, free_updates / cfg.fade_updates)
    return cfg.coef_removal * (1.0 - (1.0 - cfg.floor_removal) * f)


def mask_legal(slots: np.ndarray, n_actions: int) -> np.ndarray:
    """The rows ``"mask"`` / ``"teach"`` acting leaves legal from the six ban slots."""
    if slots[0] < 0:
        return np.asarray([-int(slots[0]) - 1])
    return np.delete(np.arange(n_actions), slots[slots > 0] - 1)


def free_legal(slots: np.ndarray, n_actions: int) -> np.ndarray:
    """The rows ``"free"`` acting leaves legal: every row but the masked (positive) slots; slot 0 < 0 restricts
    nothing."""
    return np.delete(np.arange(n_actions), slots[slots > 0] - 1)


def removal_terms(score: Tensor, keep: Tensor, rule: Tensor) -> dict[str, Tensor]:
    """Per sample, over the rows ``keep`` (host-legal minus the commit / cycle masked rows; ``score``: (B, A) float32,
    ``rule``: (B, A) bool, the removal rule's row): ``teach_removal = -log P(rule row)`` with ``P = softmax`` of
    ``score`` over ``keep`` (0 where the sample has no removal prompt); ``rules_covered`` (the sample has one),
    ``rules_p`` (P(rule row)), ``rules_agree`` (arg-max over ``keep`` is the rule's row), each (B,) float."""
    z = torch.where(keep, score, NEG)
    cov = rule.any(1)
    row = rule.float().argmax(1, keepdim=True)
    t = torch.where(cov, torch.logsumexp(z, 1) - z.gather(1, row).squeeze(1), 0.0)
    agree = (z.argmax(1, keepdim=True) == row).squeeze(1) & cov
    return {"teach_removal": t, "rules_covered": cov.float(), "rules_p": torch.where(cov, torch.exp(-t.detach()), 0.0),
            "rules_agree": agree.float()}


def teach_loss(out: Mapping[str, Tensor], valid: Tensor, den_all: Tensor | float, coef: Tensor, epoch0: Tensor | float,
               acc: Tensor) -> Tensor:
    """``coef x (T_REMOVAL x valid).sum() / den_all`` of one micro-batch; adds the epoch-0 sums of the metrics
    (removal prompts, P(rule row), arg-max = rule row, term) to ``acc`` in place (capturable)."""
    t = out["teach_removal"] * valid
    loss = coef * t.sum() / den_all
    with torch.no_grad():
        acc.add_(torch.stack([(out["rules_covered"] * valid).sum(), (out["rules_p"] * valid).sum(),
                              (out["rules_agree"] * valid).sum(), t.detach().sum()]) * epoch0)
    return loss


def row_metrics(mode: str, coef: float, acc: list[float]) -> dict[str, object]:
    """The learner row's ``rules`` block from the update's coefficient and its :data:`N_RULES_ACC` sums."""
    n = acc[0]
    return {"mode": mode, "coef_removal": coef, "n": int(round(n)), "z": acc[1] / n if n else None,
            "agree": acc[2] / n if n else None, "t": acc[3] / n if n else None}
