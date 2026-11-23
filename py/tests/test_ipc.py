"""Segment blobs and the shared-memory segment ring (in-process and across a spawned process)."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.ipc import ShmRing, decode_segment, pack_segment
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment, SegmentEnd
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.vocab import Vocab, make_synthetic_vocab


def collect_segments(vocab: Vocab, n_decisions: int = 1500, seg_len: int = 24) -> list[Segment]:
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=seg_len))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False)) for i in range(4)]
    Actor(hosts, UniformPolicy(0), Tokenizer(vocab), EpisodeSampler(EpisodeDist(), vocab, 0), ActorConfig(),
          store=store).run(max_decisions=n_decisions)
    return segs


def assert_same(a: Segment, b: Segment) -> None:
    assert a.meta == b.meta and a.start == b.start and a.end == b.end and a.bootstrap_value == b.bootstrap_value
    for name in ("actions", "logp", "values", "kinds", "n_legal", "policy_version", "rewards", "combat_id", "win",
                 "win_mask", "floors_left", "hp_post", "hp_mask"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and np.array_equal(x, y), name
    pairs = list(zip(a.samples, b.samples))
    if a.bootstrap_sample is not None:
        pairs.append((a.bootstrap_sample, b.bootstrap_sample))
    else:
        assert b.bootstrap_sample is None
    assert len(a.samples) == len(b.samples)
    for x, y in pairs:
        assert x.counts == y.counts
        for f in ("types", "raw", "rel", "edges", "cond"):
            assert np.array_equal(getattr(x, f), getattr(y, f)) and getattr(x, f).dtype == getattr(y, f).dtype


@pytest.fixture(scope="module")
def segs(vocab: Vocab) -> list[Segment]:
    s = collect_segments(vocab)
    assert {SegmentEnd.CUT, SegmentEnd.TERMINAL} <= {x.end for x in s}
    return s


def test_blob_roundtrip_and_collate(segs: list[Segment], vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    for s in segs:
        b = decode_segment(pack_segment(s))
        assert_same(s, b)
    a, b = segs[0], decode_segment(pack_segment(segs[0]))
    ca, cb = tok.collate(a.samples), tok.collate(b.samples)
    assert all(np.array_equal(ca[k].numpy(), cb[k].numpy()) for k in ca)


def test_ring_wraps_and_drops_when_full(segs: list[Segment]) -> None:
    sizes = [len(pack_segment(s)) for s in segs[:6]]
    owner, spec = ShmRing.create(int(max(sizes) * 2.5))
    rx, tx = mp.Pipe(duplex=False)
    prod, cons = ShmRing.attach(spec, tx), ShmRing.attach(spec, rx)
    try:
        for _ in range(3):  # many laps around a ring that holds ~2 blobs: exercises the wrap
            for s in segs[:6]:
                assert prod.put(s)
                assert_same(s, cons.get(rx.recv()))
        assert prod.put(segs[0]) and prod.put(segs[1])
        full = prod.stats.dropped
        ok = prod.put(segs[2], timeout=0.05) and prod.put(segs[3], timeout=0.05)
        assert not ok and prod.stats.dropped == full + 1  # counted, never silent
        assert_same(segs[0], cons.get(rx.recv()))
        assert_same(segs[1], cons.get(rx.recv()))
        st = prod.stats.take()
        assert st["segments"] >= 20 and st["bytes"] > 0 and prod.stats.segments == 0
    finally:
        prod.close()
        cons.close()
        owner.close()


def _producer(spec, conn, n: int) -> None:  # type: ignore[no-untyped-def]
    vocab = Vocab(make_synthetic_vocab())
    ring = ShmRing.attach(spec, conn)
    for s in collect_segments(vocab, 400, 16)[:n]:
        assert ring.put(s)
    conn.send(("done", ring.stats.segments))
    ring.close()


def test_ring_across_processes(vocab: Vocab) -> None:
    ctx = mp.get_context("spawn")
    owner, spec = ShmRing.create(32 << 20)
    rx, tx = ctx.Pipe(duplex=False)
    cons = ShmRing.attach(spec, rx)
    p = ctx.Process(target=_producer, args=(spec, tx, 10))
    p.start()
    try:
        got = []
        while True:
            assert rx.poll(120)
            msg = rx.recv()
            if msg[0] == "done":
                break
            got.append(cons.get(msg))
        p.join(30)
        assert p.exitcode == 0 and msg[1] == len(got) == 10
        ref = collect_segments(vocab, 400, 16)[:10]  # same deterministic fake run
        for a, b in zip(ref, got):
            assert_same(a, b)
    finally:
        if p.is_alive():
            p.kill()
        cons.close()
        owner.close()
