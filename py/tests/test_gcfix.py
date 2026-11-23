"""notes/next/round9-1119.md A2 / A3: the collector policy (spire.sysutil.GcPolicy, SPIRE_GC_POLICY), the receiver's
backlog write after every put (R1) and the ring's numpy copy-out (R2)."""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spire.ipc import ShmRing, decode_segment, pack_segment
from spire.rollout import Segment, SegmentEnd, SegmentRing
from spire.train import _receive_segments, actor_gc_fields
from spire.vocab import Vocab
from test_ipc import collect_segments

TREE = Path(__file__).resolve().parents[1]


def run_fresh(code: str, policy: str | None) -> subprocess.CompletedProcess[str]:
    """``code`` in a fresh interpreter (the policy is installed once per process) with SPIRE_GC_POLICY = ``policy``
    (None = unset)."""
    env = {k: v for k, v in os.environ.items() if k != "SPIRE_GC_POLICY"}
    env["PYTHONPATH"] = str(TREE)
    if policy is not None:
        env["SPIRE_GC_POLICY"] = policy
    return subprocess.run([sys.executable, "-B", "-c", textwrap.dedent(code)], env=env, cwd=str(TREE),
                          capture_output=True, text=True, timeout=120)


OFF = """
    import gc
    before, n_cb = gc.get_threshold(), len(gc.callbacks)
    from spire.sysutil import gc_policy
    p = gc_policy()
    p.settle()
    gc.collect(2)
    assert gc.get_threshold() == before, gc.get_threshold()
    assert gc.get_freeze_count() == 0
    assert len(gc.callbacks) == n_cb
    assert p.fields() == {"policy": "off", "frozen": False, "gen1_n": 0, "gen1_s": 0.0, "gen2_n": 0, "gen2_s": 0.0}
    assert gc_policy() is p and not p.pending
    print("OK")
"""


def test_policy_off_changes_nothing() -> None:
    r = run_fresh(OFF, "off")
    assert r.returncode == 0 and r.stdout.strip() == "OK", r.stderr


FREEZE = """
    import gc
    n_cb = len(gc.callbacks)
    from spire.sysutil import GC_THRESHOLD, gc_policy
    p = gc_policy()
    assert p.policy == "freeze" and p.pending and not p.frozen
    assert gc.get_threshold() == GC_THRESHOLD == (50_000, 20, 100), gc.get_threshold()
    assert len(gc.callbacks) == n_cb + 1
    junk = [[i] for i in range(100_000)]  # something to walk
    c0 = p.counters()
    gc.collect(1)
    gc.collect(2)
    c1 = p.counters()
    assert c1["gen1_n"] == c0["gen1_n"] + 1 and c1["gen2_n"] == c0["gen2_n"] + 1, (c0, c1)
    assert p.s[2] > 0.0 and p.s[1] > 0.0
    assert gc.get_freeze_count() == 0
    p.settle()
    frozen = gc.get_freeze_count()
    assert frozen > 100_000 and p.frozen and not p.pending, frozen
    assert p.counters()["gen2_n"] == c1["gen2_n"] + 1  # the settle's own collection
    p.settle()  # once per process
    assert gc.get_freeze_count() == frozen and p.counters()["gen2_n"] == c1["gen2_n"] + 1
    assert gc_policy() is p and len(gc.callbacks) == n_cb + 1
    f = p.fields()
    assert f["policy"] == "freeze" and f["frozen"] is True and set(f) == {"policy", "frozen", "gen1_n", "gen1_s",
                                                                       "gen2_n", "gen2_s"}
    print("OK")
"""


@pytest.mark.parametrize("policy", [None, "freeze"])
def test_policy_freeze(policy: str | None) -> None:
    r = run_fresh(FREEZE, policy)
    assert r.returncode == 0 and r.stdout.strip() == "OK", r.stderr


def test_policy_unknown_fails() -> None:
    r = run_fresh("from spire.sysutil import gc_policy\ngc_policy()\n", "Off")
    assert r.returncode != 0 and "SPIRE_GC_POLICY 'Off'" in r.stderr, r.stderr


def test_actor_gc_fields() -> None:
    g = {"frozen": True, "gen1_n": 3, "gen1_s": 0.5, "gen2_n": 1, "gen2_s": 0.25}
    reps = [{"slot": 0, "gc": {**g, "gen1_n": 1}}, {"slot": 0, "gc": g}, {"slot": 1, "gc": {**g, "frozen": False}},
            {"slot": 2}]
    assert actor_gc_fields(reps) == {"actors": 2, "frozen": 1, "gen1_n": 6, "gen1_s": 1.0, "gen2_n": 2, "gen2_s": 0.5}


# ----------------------------------------------------------------------------------------------- R1 / R2


@pytest.fixture(scope="module")
def segs(vocab: Vocab) -> list[Segment]:
    s = collect_segments(vocab)
    assert {SegmentEnd.CUT, SegmentEnd.TERMINAL} <= {x.end for x in s}
    assert any(x.bootstrap_sample is not None for x in s) and any(x.bootstrap_sample is None for x in s)
    return s


class Events(list):  # type: ignore[type-arg]
    pass


class RecordingRing(SegmentRing):
    def __init__(self, events: Events) -> None:
        super().__init__()
        self.events = events

    def put(self, seg: Segment) -> None:
        super().put(seg)
        self.events.append(("put", self.n_samples))


class RecordingBacklog:
    """A shared ``Value`` stand-in that logs its writes."""

    def __init__(self, events: Events) -> None:
        self.events = events

    @property
    def value(self) -> int:
        raise AssertionError("the receiver only writes the backlog")

    @value.setter
    def value(self, v: int) -> None:
        self.events.append(("set", v))


def test_receiver_writes_backlog_after_every_put(segs: list[Segment]) -> None:
    owner, spec = ShmRing.create(32 << 20)
    rx, tx = mp.Pipe(duplex=False)
    prod, cons = ShmRing.attach(spec, tx), ShmRing.attach(spec, rx)
    events = Events()
    out = RecordingRing(events)
    n = 8
    for s in segs[:n]:  # all announced before the receiver starts: one sweep drains them
        assert prod.put(s)
    stop = threading.Event()
    th = threading.Thread(target=_receive_segments, args=([cons], out, stop, RecordingBacklog(events)), daemon=True)
    th.start()
    try:
        deadline = time.monotonic() + 30
        while len(out) < n:
            assert time.monotonic() < deadline, "receiver did not drain the ring"
            time.sleep(0.01)
        stop.set()
        th.join(10)
        assert not th.is_alive()
    finally:
        stop.set()
        prod.close()
        cons.close()
        owner.close()
    puts = [i for i, e in enumerate(events) if e[0] == "put"]
    assert len(puts) == n
    for i in puts:
        assert events[i + 1] == ("set", events[i][1]), events[i : i + 2]
    assert events[puts[-1]][1] == sum(len(s) for s in segs[:n])


def assert_identical(a: Any, b: Any, path: str = "seg") -> None:
    """Same type; arrays: dtype, shape, values and writeability; slots / dataclass fields / sequences recursively."""
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), path
        assert a.flags.writeable == b.flags.writeable, (path, a.flags.writeable, b.flags.writeable)
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            assert_identical(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif hasattr(type(a), "__slots__") and type(a).__slots__:
        for k in type(a).__slots__:
            assert_identical(getattr(a, k), getattr(b, k), f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_identical(x, y, f"{path}[{i}]")
    else:
        assert a == b, (path, a, b)


def test_ring_get_numpy_copy_equals_bytes_path(segs: list[Segment]) -> None:
    owner, spec = ShmRing.create(32 << 20)
    rx, tx = mp.Pipe(duplex=False)
    prod, cons = ShmRing.attach(spec, tx), ShmRing.attach(spec, rx)
    try:
        for s in segs:
            assert prod.put(s)
            start, n = ann = rx.recv()
            p = start % cons.capacity
            old = decode_segment(bytes(cons.data[p : p + n]))  # the path before R2
            new = cons.get(ann)
            assert_identical(old, new)
            assert not new.actions.flags.writeable and not new.samples.raw.flags.writeable
            if new.bootstrap_sample is not None:
                assert not new.bootstrap_sample.raw.flags.writeable
            assert_identical(decode_segment(pack_segment(s)), new)
    finally:
        prod.close()
        cons.close()
        owner.close()
