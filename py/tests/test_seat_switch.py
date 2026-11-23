"""Numerical checks for heldout switch metrics; production PPO is tested elsewhere."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from spire.learner import PPOConfig
from spire.seat_switch import categorical_kl, prediction_change


def test_full_kl_uses_switch_policy_and_ignores_illegal_actions() -> None:
    before = torch.tensor([[math.log(0.75), math.log(0.25), -math.inf], [2.0, -math.inf, -math.inf]])
    after = torch.tensor([[math.log(0.5), math.log(0.5), -math.inf], [-1.0, -math.inf, -math.inf]])
    result = categorical_kl(before, after)
    assert result[0].item() == pytest.approx(0.75 * math.log(1.5) + 0.25 * math.log(0.5), abs=1e-7)
    assert result[1].item() == 0
    assert torch.equal(categorical_kl(before, before), torch.zeros(2, dtype=torch.float64))


def test_change_metrics_average_frames_and_retain_policy_only_diagnostic() -> None:
    left = [{"logits": torch.tensor([[0.0, 0.0], [0.0, -math.inf]]), "value": torch.tensor([1.0, 3.0])}]
    right = [{"logits": torch.tensor([[0.0, math.log(3)], [0.0, -math.inf]]), "value": torch.tensor([2.0, 1.0])}]
    result = prediction_change(left, right, np.asarray([1, 1], np.int8), PPOConfig())
    assert result["frames"] == 2 and result["policy_frames"] == 1
    assert result["mean_kl"] == pytest.approx(result["policy_frames_mean_kl"] / 2)
    assert result["behavior_mean_kl"] == result["mean_kl"]
    assert result["mean_abs_value_change"] == 1.5
