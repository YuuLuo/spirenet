"""The actor's scheduling (batching, per-group finishing, polling instead of blocking, pipeline depth)
never changes what is decided or recorded.

A deterministic policy (the action, log-prob and value are functions of the frame bytes alone) drives
in-process fake hosts with per-env episode lists. Whatever the scheduling configuration, every env
must play exactly the same episodes and the rollout store must receive bit-identical segments.
The policy reports "not done" for a while after each launch (like a GPU batch in flight), so finishes
happen group by group, interleaved with launches and polls.
"""
from __future__ import annotations

import zlib
from typing import Any, Sequence

import numpy as np
import pytest

from spire.actor import Actor, ActorConfig, EpisodeJob, PolicyOut, parse_lineup, random_seed
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Sample, Tokenizer
from spire.vocab import Vocab


class PerEnvJobs:
    """Finite episode source: env ``k`` plays its own fixed list of episodes."""

    def __init__(self, vocab: Vocab, n_env: int, per_env: int) -> None:
        rng = np.random.default_rng(5)
        chars = parse_lineup("IRONCLAD", vocab)
        self.jobs = {k: [EpisodeJob(chars, "IRONCLAD", 0, random_seed(rng)) for _ in range(per_env)]
                     for k in range(n_env)}

    @property
    def exhausted(self) -> bool:
        return not any(self.jobs.values())

    def next_job(self, env_id: int) -> EpisodeJob | None:
        lst = self.jobs.get(env_id)
        return lst.pop(0) if lst else None


class HashPolicy:
    """Decisions depend only on the frame: action = crc32(sample bytes) mod legal actions."""

    needs_samples = True
    version = 0

    def __init__(self, delay_polls: int) -> None:
        self.delay = delay_polls
        self.launched = self.finished = 0

    def launch(self, samples: Sequence[Sample], n_actions: np.ndarray) -> list[Any]:
        h = np.array([zlib.crc32(s.raw.tobytes() + s.types.tobytes()) for s in samples], np.int64)
        n = np.maximum(np.asarray(n_actions, np.int64), 1)
        out = PolicyOut(h % n, (h % 1000).astype(np.float32) / -1000.0, (h % 777).astype(np.float32) / 777.0,
                        np.zeros(len(samples), np.float32))
        self.launched += 1
        return [out, self.delay]

    def done(self, handle: list[Any]) -> bool:
        handle[1] -= 1
        return handle[1] < 0

    def finish(self, handle: list[Any]) -> PolicyOut:
        self.finished += 1
        return handle[0]

    def refresh(self) -> bool:
        return False


def run(vocab: Vocab, tok: Tokenizer, cfg: ActorConfig, delay: int) -> tuple[list[Segment], dict[int, list[Any]]]:
    n_env = 7
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=25, hashes=False, abort_prob=0.1))
             for i in range(n_env)]
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16))
    recs: dict[int, list[Any]] = {}
    pol = HashPolicy(delay)
    actor = Actor(hosts, pol, tok, PerEnvJobs(vocab, n_env, 4), cfg, store=store,
                  on_episode=lambda r: recs.setdefault(r.env, []).append((r.seed, r.result, r.decisions)))
    actor.run(max_seconds=60)
    assert pol.launched == pol.finished > 0
    assert store.stats.steps_added == actor.counters.decisions == store.stats.steps_released + store.stats.steps_dropped
    return segs, recs


CONFIGS = [
    ActorConfig(max_batch=1, min_group=1, max_inflight=1, spin_wait=False),
    ActorConfig(max_batch=128, min_group=1, max_inflight=1, spin_wait=False),
    ActorConfig(max_batch=4, min_group=2, max_inflight=3, spin_wait=True),
    ActorConfig(max_batch=128, min_group=16, max_inflight=2, spin_wait=True),
    ActorConfig(max_batch=3, min_group=1, max_inflight=4, spin_wait=True, max_wait_ms=0.0),
]


@pytest.mark.parametrize("delay", [0, 3])
def test_scheduling_does_not_change_decisions_or_records(vocab: Vocab, delay: int) -> None:
    tok = Tokenizer(vocab)
    ref_segs, ref_recs = run(vocab, tok, CONFIGS[0], 0)
    assert sum(len(v) for v in ref_recs.values()) == 7 * 4
    def key(s: Segment) -> tuple[int, int, int]:
        return s.meta.env_id, s.meta.seed_id, s.start

    ref = sorted(ref_segs, key=key)
    for cfg in CONFIGS[1:]:
        segs, recs = run(vocab, tok, cfg, delay)
        assert recs == ref_recs, cfg
        got = sorted(segs, key=key)
        assert len(got) == len(ref), cfg
        for a, b in zip(ref, got):
            assert (a.meta.env_id, a.meta.seed_id, a.start, a.end) == (b.meta.env_id, b.meta.seed_id, b.start, b.end)
            for f in ("actions", "logp", "values", "kinds", "n_legal", "policy_version", "rewards", "combat_id",
                      "win", "win_mask", "floors_left", "hp_post", "hp_mask"):
                x, y = getattr(a, f), getattr(b, f)
                assert x.dtype == y.dtype and np.array_equal(x, y), (cfg, f)
            assert a.bootstrap_value == b.bootstrap_value
            for sa, sb in zip(a.samples, b.samples):
                for f in ("types", "raw", "rel", "edges", "cond"):
                    assert np.array_equal(getattr(sa, f), getattr(sb, f)), (cfg, f)
                assert tuple(sa.counts) == tuple(sb.counts)
