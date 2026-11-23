"""The "pipe" transport on Linux (spire.pipes.PosixPipeMux: the hosts' stdin / stdout read through one epoll in
the actor thread): framing, EOF, restarts, a shared mux, the actor."""
from __future__ import annotations

import os
import struct
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.host import HostConfig, HostCrashed, HostProcess, HostStateError, HostTimeout
from spire.pipes import EOF, PosixPipeMux
from spire.proto import ProtocolError
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.vocab import Vocab, make_synthetic_vocab

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="PosixPipeMux is the Linux pipe transport")

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "fake_spirehost.py")
_LEN = struct.Struct("<I")


def msg(b: bytes) -> bytes:
    return _LEN.pack(len(b)) + b


def make(tmp_path: Path, *args: str, mux: PosixPipeMux | None = None, **cfg: object) -> HostProcess:
    opts: dict[str, object] = dict(start_timeout=30.0, reset_timeout=5.0, step_timeout=5.0, quit_timeout=5.0,
                                   log_path=str(tmp_path / f"host{len(args)}.stderr.log"), name="p", transport="pipe")
    opts.update(cfg)
    return HostProcess(HostConfig(command=(sys.executable, "-B", SCRIPT, "--serve", *args), **opts),  # type: ignore[arg-type]
                       mux=mux)


def play(h: Any, seed: int | None, policy: int, max_steps: int = 10_000) -> list[bytes]:
    rng = np.random.default_rng(policy)
    f = h.reset(seed, (4,), 0)
    out = [bytes(f.payload)]
    for _ in range(max_steps):
        if f.is_terminal:
            break
        f = h.step(int(rng.integers(f.n_actions)))
        out.append(bytes(f.payload))
    return out


def test_random_streams_arrive_intact_then_eof() -> None:
    """Random messages (0 B .. 300 KB) in random chunks on 12 connections: each delivers its messages in order,
    then EOF when the writer closes."""
    rng = np.random.default_rng(7)
    mux = PosixPipeMux()
    got: list[list[object]] = [[] for _ in range(12)]
    conns = [mux.open(f"s{i}", got[i].append) for i in range(12)]
    want, writers = [], []
    for c in conns:
        msgs = [bytes(rng.integers(0, 256, int(n), dtype=np.uint8))
                for n in rng.choice([0, 1, 7, 100, 4096, 70_000, 300_000], size=12)]
        want.append(msgs)
        stream = b"".join(msg(m) for m in msgs)
        cuts = np.sort(rng.integers(0, len(stream), size=20))
        chunks = [stream[a:b] for a, b in zip([0, *cuts], [*cuts, len(stream)]) if b > a]
        in_r, out_w = c.child_fds
        os.close(in_r)
        c.child_fds = ()

        def write(fd: int = out_w, chunks: list[bytes] = chunks) -> None:
            for ch in chunks:
                os.write(fd, ch)
            os.close(fd)
        writers.append(threading.Thread(target=write))
    for t in writers:
        t.start()
    for _ in range(100_000):
        if all(len(g) == len(w) + 1 for g, w in zip(got, want)):
            break
        mux.poll(0.01)
    for t in writers:
        t.join()
    for g, w in zip(got, want):
        assert g[:-1] == w and g[-1] is EOF
    assert mux.n_open == 0
    mux.close()


def test_oversized_message_fails_the_connection() -> None:
    mux = PosixPipeMux(max_bytes=1000)
    got: list[object] = []
    c = mux.open("big", got.append)
    os.write(c.child_fds[1], _LEN.pack(1001))
    mux.poll(1.0)
    assert len(got) == 1 and isinstance(got[0], ProtocolError) and mux.n_open == 0
    mux.close()


def test_pipe_host_matches_in_process_fake(tmp_path: Path) -> None:
    vocab = Vocab(make_synthetic_vocab())
    with make(tmp_path, "--seed", "1", "--mean-decisions", "40", expected_hashes=(vocab.hash, 0),
              verify_hashes=True) as h:
        a = [play(h, None, 0), play(h, 99, 1)]
        assert h.stats.finished == 2 and h._conn is not None and h._conn.client_pid == h.pid
        assert not h.reader_threads
        proc = h._proc
    assert proc is not None and proc.returncode == 0 and not h.alive
    ref = FakeHost(vocab, seed=1, config=FakeHostConfig(mean_decisions=40))
    assert a == [play(ref, None, 0), play(ref, 99, 1)]


def test_pipe_hang_and_crash_restart(tmp_path: Path) -> None:
    with make(tmp_path, "--hang-after", "3", "--mean-decisions", "500", step_timeout=0.5) as h:
        pid0 = h.pid
        h.reset(1, (4,), 0)
        h.step(0)
        h.step(0)
        with pytest.raises(HostTimeout):
            h.step(0)
        assert (h.stats.timeouts, h.stats.restarts, h.stats.lost) == (1, 1, 1)
        assert h.pid != pid0 and h._conn is not None and h._conn.client_pid == h.pid
        with pytest.raises(HostStateError):
            h.step(0)
        assert h.reset(2, (4,), 0).step_id == 0
    with make(tmp_path, "--crash-after", "2", "--mean-decisions", "500") as h:
        h.reset(1, (4,), 0)
        h.step(0)
        with pytest.raises(HostCrashed) as e2:
            h.step(0)
        assert any("injected crash" in line for line in e2.value.stderr_tail)
        assert (h.stats.crashes, h.stats.restarts) == (1, 1)
        assert h.reset(1, (4,), 0).step_id == 0


def test_shared_mux_closes_with_its_last_host(tmp_path: Path) -> None:
    mux = PosixPipeMux()
    hosts = [make(tmp_path, "--seed", str(i), mux=mux, name=f"m{i}") for i in range(3)]
    for h in hosts:
        h.reset(None, (4,), 0)
    assert mux.n_open == 3
    for h in hosts[:2]:
        h.close()
    assert not mux._closed and mux.n_open == 1
    hosts[2].step(0)
    hosts[2].close()
    assert mux._closed
    assert all(h._proc is None for h in hosts)


def test_actor_over_pipes_accounting_crash_and_hang(tmp_path: Path) -> None:
    vocab = Vocab(make_synthetic_vocab())
    tok = Tokenizer(vocab)
    mux = PosixPipeMux()
    opts: dict[str, Any] = dict(step_timeout=1.0, max_restarts=20, mux=mux)
    hosts = [make(tmp_path, "--no-hashes", "--mean-decisions", "30", "--crash-after", "25", **{**opts, "max_restarts": 3},
                  name="crash"),
             make(tmp_path, "--no-hashes", "--mean-decisions", "30", "--hang-after", "40", "--seed", "1", **opts,
                  name="hang"),
             *[make(tmp_path, "--no-hashes", "--mean-decisions", "30", "--seed", str(s), **opts, name=f"ok{s}")
               for s in (2, 3)]]
    try:
        segs: list[Segment] = []
        store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
        recs: list[Any] = []
        actor = Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
                      ActorConfig(max_batch=8, min_group=2), store=store, on_episode=recs.append)
        assert actor.mux is mux
        actor.run(max_seconds=12.0)
        details = " ".join(r.detail for r in recs if r.result == "lost")
        assert "HostCrashed" in details and "HostTimeout" in details and "host retired" in details
        assert hosts[0].stats.restarts == 3 and actor.counters.hosts_retired == 1 and actor.envs[0].dead
        assert hosts[1].stats.timeouts >= 1 and not actor.envs[1].dead
        st = store.stats
        assert st.steps_added == actor.counters.decisions == st.steps_released + st.steps_dropped
        res = Counter(r.result for r in recs)
        assert res["win"] + res["loss"] > 10 and res["win"] + res["loss"] == st.wins + st.losses
        assert sum(len(s) for s in segs) == st.steps_released
    finally:
        for h in hosts:
            h.close()
    assert mux._closed
