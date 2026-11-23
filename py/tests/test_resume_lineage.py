"""Crash resume cannot expose an abandoned publication to the new actors."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeSampler, TorchPolicy
from spire.learner import FileWeightPublisher, FileWeightSource, Learner
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment, build_batch
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.train import Supervisor, build_net
from spire.vocab import Vocab

from test_learner import FAST, Harness, params
from test_train import small_config


def _close(supervisor: Supervisor) -> None:
    for writer in (supervisor.metrics, supervisor.agg.episodes, supervisor.agg.errors):
        writer.close()


@pytest.mark.parametrize("resume", [False, True])
def test_every_supervisor_start_removes_acting_publications(tmp_path: Path, resume: bool) -> None:
    cfg = small_config(tmp_path, "startup")
    weights = cfg.run_dir / "weights"
    publisher = FileWeightPublisher(weights)
    publisher.publish({"weight": torch.tensor([3.0])}, 3)
    publisher.publish({"weight": torch.tensor([10.0])}, 10)
    unrelated = weights / "keep.txt"
    unrelated.write_text("not an acting publication", encoding="utf-8")

    supervisor = Supervisor(cfg, resume=resume)
    try:
        assert not (weights / "LATEST").exists()
        assert list(weights.glob("acting-*.pt")) == []
        assert unrelated.read_text(encoding="utf-8") == "not an acting publication"
    finally:
        _close(supervisor)


def test_resume_older_checkpoint_never_adopts_abandoned_branch(vocab: Vocab, tmp_path: Path) -> None:
    cfg = small_config(tmp_path, "lineage")
    tok = Tokenizer(vocab)
    publisher = FileWeightPublisher(cfg.run_dir / "weights")
    old = Learner(build_net(cfg, tok, None), tok, cfg.ppo, device="cpu", publisher=publisher)
    old.version = old.updates = 3
    checkpoint = cfg.run_dir / "ckpt" / "latest.pt"
    old.save(checkpoint)
    old.publish()
    with torch.no_grad():
        next(old.model.parameters()).add_(1.0)
    publisher.publish(old.acting_state_dict(), 10)
    source = FileWeightSource(cfg.run_dir / "weights")
    assert source.latest_version() == 10

    supervisor = Supervisor(cfg, resume=True)
    policy = TorchPolicy(build_net(cfg, tok, None), tok, use_graphs=False)
    # Drive the actual poll/adoption path synchronously: the actor wins the startup race.
    policy._source = source
    try:
        policy._poll_once()
        assert not policy.refresh() and policy._adopted == -1
        learner = Learner(build_net(cfg, tok, None), tok, cfg.ppo, device="cpu", publisher=publisher)
        learner.load(checkpoint)
        learner.publish()
        policy._poll_once()
        assert policy.refresh() and policy.version == learner.version == 3
        assert all(torch.equal(a, b) for a, b in zip(policy.net.state_dict().values(),
                                                    learner.model.state_dict().values()))

        segments: list[Segment] = []
        store = RolloutStore(segments.append, RoomIds.from_vocab(vocab),
                             config=RolloutConfig(segment_len=32))
        hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=16, hashes=False,
                                                             abort_prob=0.0, error_prob=0.0)) for i in range(2)]
        Actor(hosts, policy, tok, EpisodeSampler(cfg.episodes, vocab, 0), ActorConfig(), store=store).run(
            max_decisions=128)
        assert segments
        batch = build_batch(segments, cfg.ppo.gae, learner.version)
        assert np.all(batch.staleness >= 0)
        assert all(np.all(s.policy_version == 3) for s in segments)
        metrics = learner.update(segments)
        assert metrics["staleness_mean"] == 0
        assert metrics["segment_staleness_hist"] == {0: len(segments)}
        assert learner.negative_staleness_samples == 0
    finally:
        policy.close()
        _close(supervisor)


def test_learner_counts_and_refuses_any_future_step_before_update(vocab: Vocab,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(vocab, seed=4, cfg=replace(FAST, recompute_values=True), n_envs=2)
    segments = harness.segments(32)
    segment = next(s for s in segments if len(s) >= 3)
    segment.policy_version[-2:] = 1
    assert segment.version_min == 0  # checking only the segment's oldest version misses the violation
    learner = harness.learner
    before = params(learner.model)
    recompute = Mock(side_effect=AssertionError("must refuse before value recomputation"))
    monkeypatch.setattr(learner, "_recompute", recompute)
    with pytest.raises(AssertionError, match="negative staleness: 2 samples.*learner version 0"):
        learner.update(segments)
    recompute.assert_not_called()
    assert learner.negative_staleness_samples == 2
    assert learner.state()["counters"]["negative_staleness_samples"] == 2
    assert learner.version == learner.updates == learner.samples_seen == learner.ema_steps == 0
    assert not learner.opt.state
    assert all(torch.equal(a, b) for a, b in zip(before, params(learner.model)))
