"""Canonical lineups, per-seat caps, public death records and team-level rewards."""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

from ppo_stubs import obs_frame, terminal_frame
from spire.actor import EpisodeDist, EpisodeJob, EpisodeSampler, _Env, parse_lineup, validate_distribution
from spire.evalpanel import Panel
from spire.distributions import legal_uniform_log_probs
from spire.host import Host
from spire.metrics import EpisodeRecord, OutcomeTracker, canonical_lineup
from spire.rewards import RewardConfig, RoomIds, episode_targets, frame_facts
from spire.proto import parse_frame
from spire.schema import FrameFlag, G, Kind, MsgType, RESET_FLAGS_DEFAULT, Result, TerminalReason
from spire.synth import FrameBuilder
from spire.vocab import Vocab


def test_canonical_multiset_used_for_training_panels_metrics_and_weights(vocab: Vocab) -> None:
    original = "IRONCLAD+DEFECT+IRONCLAD+DEFECT"
    canonical = "DEFECT+DEFECT+IRONCLAD+IRONCLAD"
    assert canonical_lineup(original) == canonical
    dist = validate_distribution({"lineups": [[original, 1.0]]}, vocab)
    assert dist.lineups == ((canonical, 1.0),)
    sampler = EpisodeSampler(EpisodeDist(lineups=((original, 1.0),)), vocab, seed=3)
    job = sampler.next_job(0)
    assert job.lineup == canonical
    assert job.characters == tuple(vocab.id("character", name) for name in canonical.split("+"))
    panel = Panel("four", ("000000000001",), (original,), (10,))
    assert panel.lineups == (canonical,)
    assert panel.jobs(vocab)[0].characters == job.characters
    tracker = OutcomeTracker()
    tracker.add(EpisodeRecord(original, 10, "win"))
    tracker.add(EpisodeRecord(canonical, 10, "loss"))
    assert tracker.summary()[canonical + "|A10"]["n"] == 2
    record = EpisodeRecord("IRONCLAD+DEFECT", 10, "win", final_hp=(70, 60), final_alive=(True, False))
    assert record.lineup == "DEFECT+IRONCLAD"
    assert record.final_hp == (60, 70) and record.final_alive == (False, True)
    with pytest.raises(ValueError, match="duplicate canonical"):
        validate_distribution({"lineups": [[original, 1.0], [canonical, 1.0]]}, vocab)
    with pytest.raises(AssertionError, match="duplicate canonical"):
        Panel("duplicate", ("000000000001",), (original, canonical), (10,))


@pytest.mark.parametrize("seats", [1, 2, 3, 4])
def test_decision_cap_is_per_seat(vocab: Vocab, seats: int) -> None:
    lineup = "+".join(["DEFECT"] * seats)
    job = EpisodeJob(parse_lineup(lineup, vocab), lineup)
    assert job.reset_cmd(int(RESET_FLAGS_DEFAULT), 30000).max_decisions == 30000 * seats
    assert job.reset_cmd(int(RESET_FLAGS_DEFAULT), 0).max_decisions == 0


def test_observed_deaths_count_revivals_and_team_wipe(vocab: Vocab) -> None:
    lineup = "DEFECT+DEFECT+IRONCLAD+IRONCLAD"
    env = _Env(0, 0, cast(Host, None), job=EpisodeJob(parse_lineup(lineup, vocab), lineup))
    for hps in ([80, 80, 80, 80], [80, 0, 80, 80], [80, 1, 80, 80], [80, 0, 80, 80]):
        env.observe_seats(obs_frame(Kind.COMBAT, 1, 0, True, hps))
    assert env.seat_deaths == 2
    env.observe_seats(terminal_frame(Result.LOSS, 1, 0, [0, 0, 0, 0]))
    assert env.seat_deaths == 5  # repeated seat-1 death plus each seat's final death
    assert env.final_hp == (0, 0, 0, 0) and env.final_alive == (False,) * 4


def test_win_terminal_zeroing_is_recorded_without_inventing_deaths(vocab: Vocab) -> None:
    lineup = "DEFECT+DEFECT+IRONCLAD+IRONCLAD"
    env = _Env(0, 0, cast(Host, None), job=EpisodeJob(parse_lineup(lineup, vocab), lineup))
    env.observe_seats(obs_frame(Kind.REWARD, 1, 0, False, [70, 60, 50, 40]))
    env.observe_seats(terminal_frame(Result.WIN, 1, 0, [0, 0, 0, 0]))
    assert env.seat_deaths == 0 and env.final_hp == (0,) * 4


def test_bare_aborted_terminal_retains_last_observed_seats(vocab: Vocab) -> None:
    lineup = "DEFECT+DEFECT+IRONCLAD+IRONCLAD"
    env = _Env(0, 0, cast(Host, None), job=EpisodeJob(parse_lineup(lineup, vocab), lineup))
    env.observe_seats(obs_frame(Kind.COMBAT, 1, 0, True, [70, 0, 50, 40]))
    before = env.final_hp, env.final_alive, env.seat_deaths
    builder = FrameBuilder(MsgType.TERMINAL)
    builder.flags = int(FrameFlag.TERMINAL)
    builder.result = int(Result.ABORTED)
    builder.glob[G.KIND] = Kind.TERMINAL
    builder.glob[G.N_SEATS] = 4
    builder.glob[G.TERMINAL_REASON] = TerminalReason.ABORTED
    env.observe_seats(parse_frame(builder.build()))
    assert (env.final_hp, env.final_alive, env.seat_deaths) == before
    builder.result = int(Result.LOSS)
    with pytest.raises(AssertionError, match="only aborted terminals"):
        env.observe_seats(parse_frame(builder.build()))


def test_rewards_read_all_seats_and_wait_for_team_terminal(vocab: Vocab) -> None:
    frames = [obs_frame(Kind.COMBAT, 1, 0, True, hp) for hp in
              ([80, 80, 80, 80], [80, 0, 80, 80], [80, 1, 80, 80])]
    facts = [frame_facts(frame) for frame in frames]
    assert [fact.hp_frac for fact in facts] == pytest.approx([0.8, 0.6, 0.6025])
    cfg = RewardConfig(gamma=1.0)
    targets = episode_targets(facts, RoomIds.from_vocab(vocab), cfg)
    np.testing.assert_allclose(targets.rewards, [-0.02, 0.00025], atol=1e-7)
    terminal = frame_facts(terminal_frame(Result.LOSS, 1, 0, [0, 0, 0, 0]))
    terminal_targets = episode_targets([*facts, terminal], RoomIds.from_vocab(vocab), cfg)
    assert terminal_targets.rewards[-1] == pytest.approx(-0.1 - 0.06025)


def test_union_combat_exploration_is_uniform_over_action_rows_not_seats() -> None:
    logits = torch.arange(10, dtype=torch.float64)[None, :]
    epsilon = torch.tensor([0.2], dtype=torch.float64)
    mixed = legal_uniform_log_probs(logits, epsilon).exp()[0]
    uniform_part = mixed - (1 - epsilon) * logits.softmax(-1)[0]
    # Union frame seats have 1, 2, 3, 4 legal actions: epsilon's seat mass is proportional
    # to that count. An ordinary seat-major frame contains just one of these seat groups.
    action_seats = torch.repeat_interleave(torch.arange(4), torch.arange(1, 5))
    mass = torch.stack([uniform_part[action_seats == seat].sum() for seat in range(4)])
    seat_share = torch.arange(1, 5, dtype=torch.float64) / 10
    # legal.sum() is integer; its .log() uses float32 even with float64 logits/epsilon.
    # Bound that scalar log-normalizer rounding while checking the seat proportions strictly.
    torch.testing.assert_close(mass, epsilon * seat_share,
                               rtol=2 * torch.finfo(torch.float32).eps, atol=1e-15)
    torch.testing.assert_close(mass / mass.sum(), seat_share, rtol=1e-12, atol=1e-12)
