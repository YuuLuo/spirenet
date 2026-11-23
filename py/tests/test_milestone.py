"""Milestone checkpoints are due when the update count has crossed a multiple, not only when it equals one."""
from __future__ import annotations

from spire.train import milestone_index


def test_milestone_index_counts_crossed_boundaries() -> None:
    assert [milestone_index(u, 500) for u in (0, 1, 499, 500, 501, 999, 1000, 1788)] == [0, 0, 0, 1, 1, 1, 2, 3]
    assert milestone_index(12345, 0) == 0  # off


def test_timed_saves_keep_one_milestone_per_crossing() -> None:
    # the learner saves every 10 minutes, about 73 updates apart: a save lands on an exact multiple almost never
    last, kept = milestone_index(818, 500), []
    for updates in range(818, 2400, 73):
        if milestone_index(updates, 500) > last:
            last = milestone_index(updates, 500)
            kept.append(updates)
    assert kept == [1037, 1548, 2059]
    assert not any(u % 500 == 0 for u in range(818, 2400, 73))  # the old rule would have kept none of them
