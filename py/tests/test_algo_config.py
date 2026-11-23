"""Candidate wiring and the shaping ablation, including the unchanged default objective."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ppo_stubs import obs_frame, terminal_frame
from spire.rewards import RewardConfig, RoomIds, episode_targets, frame_facts
from spire.schema import Kind, Result
from spire.train import REPO, TrainConfig, build, load_config
from spire.vocab import Vocab


def test_hp_potential_does_not_change_complete_episode_preference(vocab: Vocab) -> None:
    rooms = RoomIds.from_vocab(vocab)
    totals = []
    for after_hp in (40, 70):
        facts = [frame_facts(f) for f in [
            obs_frame(Kind.REST, 12, 0, False, [40]),
            obs_frame(Kind.MAP, 12, 0, False, [after_hp]),
            terminal_frame(Result.LOSS, 20, 0, [0])]]
        shaped = episode_targets(facts, rooms, RewardConfig(gamma=1.0))
        base = episode_targets(facts, rooms, RewardConfig(gamma=1.0, hp_coef=0))
        assert shaped.rewards.sum() == pytest.approx(base.rewards.sum() - .04)
        totals.append(float(shaped.rewards.sum()))
    assert totals[0] == pytest.approx(totals[1])


