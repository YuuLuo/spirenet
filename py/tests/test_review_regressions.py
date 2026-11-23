"""Regression tests for the bugs found in the py-review (notes/py-review.md)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest
import torch

import spire.actor as actor_mod
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeJob, EpisodeSampler, JobList, PolicyOut, \
    TorchPolicy, UniformPolicy
from spire.host import HostStateError
from spire.learner import FileWeightPublisher, FileWeightSource, WeightSource
from spire.model import ModelConfig, SpireNet, export_weights
from spire.rewards import RoomIds, team_hp_fraction
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import ResetFlag, SeatCol
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import TokenLayout, Tokenizer
from spire.vocab import Vocab, make_synthetic_vocab

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, d_id=16, d_cond=16, d_static=16)


# --------------------------------------------------------------------------- learner: weight publication


def test_resumed_publisher_drops_files_from_a_run_that_went_further(tmp_path: Any) -> None:
    """Resume from a checkpoint older than the newest published weights: the stale higher-version
    files sorted after (and so evicted) every freshly published file, actors never got weights,
    the learner never got samples (deadlock)."""
    first = FileWeightPublisher(tmp_path, keep=3)
    for v in (5, 6, 7):  # published after the last checkpoint, then the learner died
        first.publish({"w": torch.full((2,), float(v))}, v)
    resumed = FileWeightPublisher(tmp_path, keep=3)
    src = FileWeightSource(tmp_path)
    for v in (3, 4, 5, 6, 7, 8):  # learner resumed from the v3 checkpoint
        resumed.publish({"w": torch.full((2,), float(v))}, v)
        got = src.poll(v - 1)
        assert got is not None and got[0] == v and float(got[1]["w"][0]) == v
    assert sorted(p.name for p in tmp_path.glob("acting-*.pt")) == [f"acting-{v:010d}.pt" for v in (6, 7, 8)]


# --------------------------------------------------------------------------- actor: version stamping


class _LaggingPolicy(UniformPolicy):
    """Uniform actions; ``logp`` carries the version the batch was launched with; each batch is still
    in flight during one housekeeping pass, whose ``refresh`` adopts a new version."""

    def launch(self, samples: Sequence[Any], n_actions: np.ndarray) -> list[Any]:  # type: ignore[override]
        out = super().launch(samples, n_actions)
        out.logp = np.full(len(n_actions), -float(self.version), np.float32)
        return [out, 0]

    def done(self, handle: list[Any]) -> bool:  # type: ignore[override]
        handle[1] += 1
        return bool(handle[1] > 1)

    def finish(self, handle: list[Any]) -> PolicyOut:  # type: ignore[override]
        return handle[0]

    def refresh(self) -> bool:
        self.version += 1
        return True


def test_steps_carry_the_version_their_batch_was_launched_with(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=30, hashes=False)) for i in range(3)]
    actor = Actor(hosts, _LaggingPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
                  ActorConfig(max_batch=4, min_group=1), store=store)
    actor.run(max_decisions=400)
    versions = np.concatenate([s.policy_version for s in segs])
    launched = -np.concatenate([s.logp for s in segs]).astype(np.int64)
    assert len(np.unique(versions)) > 10  # weights did change between launch and finish
    np.testing.assert_array_equal(versions, launched)


# --------------------------------------------------------------------------- actor: weight polling


class _Source(WeightSource):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.latest = 0
        self.state = state

    def poll(self, have_version: int) -> tuple[int, dict[str, torch.Tensor]] | None:
        return None if self.latest <= have_version else (self.latest, self.state)


def test_weights_cleared_by_a_racing_refresh_are_fetched_again(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    net = SpireNet(tok.layout, TINY)
    src = _Source(export_weights(net))
    pol = TorchPolicy(net, tok, use_graphs=False, weights=src, poll_interval=3600.0)
    try:
        assert pol.wait_for_weights(10.0) and pol.version == 0
        src.latest = 2
        pol._poll_once()
        # refresh() read an older staged entry, the poll thread staged v2, refresh() cleared it:
        pol._staged = None
        pol._poll_once()  # before the fix: have = last *fetched* (2) -> v2 never fetched again
        assert pol.refresh() and pol.version == 2
        assert not pol.refresh()
    finally:
        pol.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_refresh_copies_weights_on_the_inference_stream(vocab: Vocab, monkeypatch: pytest.MonkeyPatch) -> None:
    tok = Tokenizer(vocab)
    net = SpireNet(tok.layout, TINY).cuda()
    pol = TorchPolicy(net, tok, high_priority=True)
    streams: list[torch.cuda.Stream] = []
    monkeypatch.setattr(actor_mod, "import_weights", lambda n, sd: streams.append(torch.cuda.current_stream()))
    pol._staged = (1, export_weights(net))
    assert pol.refresh()
    assert pol.stream is not None and streams == [pol.stream]


# --------------------------------------------------------------------------- actor: host state errors


def test_reset_on_a_busy_host_raises_instead_of_spinning(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    host = FakeHost(vocab, seed=0, config=FakeHostConfig(hashes=False))
    job = EpisodeJob((vocab.id("character", "IRONCLAD"),), "IRONCLAD")
    jobs = JobList([job] * 50)
    actor = Actor([host], UniformPolicy(0), tok, jobs, ActorConfig(mode="eval"))
    host.send_reset(job.reset_cmd(int(ResetFlag.AUTO_SINGLE)))  # a reply is now outstanding
    with pytest.raises(HostStateError):
        actor.run(max_decisions=10)  # before: every job "lost" in a tight retry loop (infinite with a sampler)
    assert jobs._next == 1


# --------------------------------------------------------------------------- tokenizer / rewards


def test_layout_refuses_rows_beyond_int16() -> None:
    """Derived ("@") id slots store global embedding rows in int16 Sample.raw; a larger layout used to
    saturate them silently (every overflowing card / relic / power id -> the same row)."""
    with pytest.raises(ValueError, match="int16"):
        TokenLayout(Vocab(make_synthetic_vocab(scale=4.0)))
    assert TokenLayout(Vocab(make_synthetic_vocab(scale=3.0))).n_rows - 1 <= np.iinfo(np.int16).max


def test_team_hp_fraction_matches_vectorised_reference() -> None:
    rng = np.random.default_rng(0)
    for n in range(0, 5):
        for _ in range(200):
            seats = np.zeros((n, 32), np.int32)
            seats[:, SeatCol.HP] = rng.integers(-20, 120, n)
            seats[:, SeatCol.MAX_HP] = rng.integers(-1, 100, n)
            hp = np.maximum(seats[:, SeatCol.HP], 0).astype(np.float64)
            mx = seats[:, SeatCol.MAX_HP].astype(np.float64)
            ref = float(np.minimum(np.divide(hp, mx, out=np.zeros_like(hp), where=mx > 0), 1.0).mean()) if n else 0.0
            assert team_hp_fraction(seats) == pytest.approx(ref, abs=1e-12)
