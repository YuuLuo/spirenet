"""The entropy coefficient of a sample is per single-seat-equivalent decision when ent_coef_seat_scaled is set."""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from torch import nn

from ppo_stubs import StubTokenizer
from spire.learner import Learner, PPOConfig
from spire.rollout import RolloutBatch
from spire.schema import Kind


def _learner(**kw: object) -> Learner:
    cfg = PPOConfig(ent_coef=0.01, ent_coef_by_kind=((int(Kind.REST), 0.03),), bf16=False, **kw)  # type: ignore[arg-type]
    return Learner(nn.Linear(2, 2), StubTokenizer(), cfg, device="cpu")


def _batch(kinds: list[int], seats: list[int]) -> RolloutBatch:
    return cast(RolloutBatch, SimpleNamespace(kinds=np.asarray(kinds, np.int8), n_seats=np.asarray(seats, np.int8)))


def test_default_keeps_the_coefficient_by_kind_for_every_seat_count() -> None:
    rb = _batch([Kind.COMBAT, Kind.REST, Kind.COMBAT, Kind.REST], [1, 1, 4, 4])
    np.testing.assert_array_equal(_learner()._ent_coef(rb), np.asarray([0.01, 0.03, 0.01, 0.03], np.float32))


def test_seat_scaling_divides_by_the_seat_count_and_leaves_one_seat_alone() -> None:
    rb = _batch([Kind.COMBAT, Kind.REST, Kind.COMBAT, Kind.REST, Kind.COMBAT], [1, 1, 4, 4, 2])
    coef = _learner(ent_coef_seat_scaled=True)._ent_coef(rb)
    assert coef.dtype == np.float32
    np.testing.assert_array_equal(coef[:2], np.asarray([0.01, 0.03], np.float32))  # single-seat samples: bit-identical
    np.testing.assert_allclose(coef[2:], [0.0025, 0.0075, 0.005], rtol=1e-6)


def test_seat_scaling_needs_a_seat_count_per_sample() -> None:
    rb = _batch([Kind.COMBAT, Kind.REST], [4])
    with pytest.raises(AssertionError):
        _learner(ent_coef_seat_scaled=True)._ent_coef(rb)
