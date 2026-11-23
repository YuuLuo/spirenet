"""Named-pipe transport (spire.pipes, notes/transport-spec.md): framing, HELLO, faults, actor use."""
from __future__ import annotations

import struct
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from spire.actor import STAGES, Actor, ActorConfig, EpisodeDist, EpisodeSampler, RunTotals, UniformPolicy
from spire.host import ActionRejected, HostConfig, HostCrashed, HostProcess, HostStateError, HostTimeout
from spire.pipes import EOF, HELLO, HELLO_MAGIC, READ_BUFFER, TRANSPORT_VERSION, PipeMux, hello_message
from spire.proto import ProtocolError
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import MAX_FRAME_BYTES, ErrorCode
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.vocab import Vocab, make_synthetic_vocab

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="named-pipe transport is Windows-only")


@pytest.fixture(autouse=True, params=["native", "python"])
def impl(request: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """Every test runs against both I/O implementations of PipeMux (spire._pipemux and pure Python)."""
    from spire import pipes
    if request.param == "native" and pipes._pipemux is None:
        pytest.skip("spire._pipemux is not built (python -B native/build.py)")
    monkeypatch.setattr(pipes, "NATIVE_DEFAULT", request.param == "native")
    return str(request.param)

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "fake_spirehost.py")
_LEN = struct.Struct("<I")


def make(tmp_path: Path, *args: str, mux: PipeMux | None = None, **cfg: object) -> HostProcess:
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


# --------------------------------------------------------------------------- framing / handshake


class Client(threading.Thread):
    """A raw pipe client (what SpireHost does): HELLO, then the given writes."""

    def __init__(self, name: str, writes: list[bytes], hello: bytes | None = None, linger: float = 0.0) -> None:
        super().__init__(daemon=True)
        self.name_, self.writes, self.hello, self.linger = name, writes, hello, linger
        self.received = b""

    def run(self) -> None:
        with open(self.name_, "r+b", buffering=0) as f:
            f.write(hello_message() if self.hello is None else self.hello)
            for w in self.writes:
                f.write(w)
                time.sleep(0.001)
            time.sleep(self.linger)


def drive(mux: PipeMux, until: Any, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while not until() and time.monotonic() < end:
        mux.poll(0.01)
    assert until(), "timed out"


def msg(payload: bytes) -> bytes:
    return _LEN.pack(len(payload)) + payload


def test_framing_split_coalesced_and_large_messages(impl: str) -> None:
    mux = PipeMux()
    assert mux.native == (impl == "native")
    got: list[object] = []
    conn = mux.open("framing", got.append)
    big = bytes(np.random.default_rng(0).integers(0, 256, 2 * READ_BUFFER + 12345, dtype=np.uint8))  # grows the buffer
    assert len(big) < MAX_FRAME_BYTES
    small = [b"a" * 8, b"b" * 4, b""]
    stream = msg(small[0]) + msg(small[1]) + msg(small[2])  # three messages in one write
    split = msg(b"split!!!")
    writes = [stream, split[:3], split[3:6], split[6:], msg(big), msg(b"tail")]
    c = Client(conn.name, writes)
    c.start()
    drive(mux, lambda: len(got) >= 6)
    c.join(5)  # the client closes its end: EOF
    drive(mux, lambda: len(got) >= 7)
    assert got == [*small, b"split!!!", big, b"tail", EOF]
    assert conn.state == "closed" and conn.client_pid > 0 and mux.n_open == 0
    mux.close()


def test_bad_hello_and_oversized_message_are_protocol_errors() -> None:
    mux = PipeMux(max_bytes=1 << 16)  # oversized = beyond the mux limit (MAX_FRAME_BYTES by default)
    got: list[object] = []
    bad = mux.open("bad-hello", got.append)
    Client(bad.name, [], hello=msg(HELLO.pack(HELLO_MAGIC, TRANSPORT_VERSION + 1, 0, 0)), linger=0.5).start()
    drive(mux, lambda: bool(got))
    assert isinstance(got[0], ProtocolError) and "HELLO" in str(got[0])
    got2: list[object] = []
    big = mux.open("too-big", got2.append)
    Client(big.name, [msg(b"x" * (1 << 17))], linger=0.5).start()
    drive(mux, lambda: bool(got2))
    assert isinstance(got2[0], ProtocolError) and "exceeds" in str(got2[0])
    mux.close()


def test_commands_sent_before_connect_are_flushed() -> None:
    mux = PipeMux()
    got: list[object] = []
    conn = mux.open("early", got.append)
    conn.send(msg(b"first"))
    conn.send(msg(b"second"))
    seen: list[bytes] = []

    def client() -> None:
        with open(conn.name, "r+b", buffering=0) as f:
            f.write(hello_message())
            buf = b""
            while len(buf) < 2 * 4 + 11:
                buf += f.read(64)
            seen.append(buf)

    t = threading.Thread(target=client, daemon=True)
    t.start()
    drive(mux, lambda: bool(seen))
    t.join(5)
    assert seen[0] == msg(b"first") + msg(b"second")
    mux.close()


# --------------------------------------------------------------------------- HostProcess over pipes


def test_pipe_host_matches_in_process_fake(tmp_path: Path) -> None:
    vocab = Vocab(make_synthetic_vocab())
    with make(tmp_path, "--seed", "1", "--mean-decisions", "40", expected_hashes=(vocab.hash, 0),
              verify_hashes=True) as h:
        a = [play(h, None, 0), play(h, 99, 1)]
        assert h.stats.finished == 2 and h._conn is not None and h._conn.client_pid > 0
        proc = h._proc
    assert proc is not None and proc.returncode == 0 and not h.alive
    ref = FakeHost(vocab, seed=1, config=FakeHostConfig(mean_decisions=40))
    assert a == [play(ref, None, 0), play(ref, 99, 1)]


def test_pipe_error_resend(tmp_path: Path) -> None:
    with make(tmp_path, "--mean-decisions", "30") as h:
        f = h.reset(5, (4,), 0)
        with pytest.raises(ActionRejected) as e:
            h.step(f.n_actions + 3)
        assert e.value.code == ErrorCode.BAD_INDEX and e.value.frame is not None
        assert bytes(e.value.frame.payload) == bytes(f.payload)
        assert h.step(0).step_id == 1


def test_pipe_hang_and_crash_restart_with_a_new_pipe(tmp_path: Path) -> None:
    with make(tmp_path, "--hang-after", "3", "--mean-decisions", "500", step_timeout=0.5) as h:
        name0, pid0 = h._conn.name if h._conn else "", h.pid
        h.reset(1, (4,), 0)
        h.step(0)
        h.step(0)
        with pytest.raises(HostTimeout):
            h.step(0)
        assert (h.stats.timeouts, h.stats.restarts, h.stats.lost) == (1, 1, 1)
        assert h.pid != pid0 and h._conn is not None and h._conn.name != name0
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
    mux = PipeMux()
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


# --------------------------------------------------------------------------- the actor over pipes


def test_actor_over_pipes_accounting_crash_and_hang(tmp_path: Path) -> None:
    vocab = Vocab(make_synthetic_vocab())
    tok = Tokenizer(vocab)
    mux = PipeMux()
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
        reports: list[dict] = []
        actor = Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
                      ActorConfig(max_batch=8, min_group=2), store=store, on_episode=recs.append,
                      on_report=reports.append)
        assert actor.mux is mux
        actor.run(max_seconds=12.0)
        lost = [r for r in recs if r.result == "lost"]
        details = " ".join(r.detail for r in lost)
        assert "HostCrashed" in details and "HostTimeout" in details and "host retired" in details
        assert hosts[0].stats.restarts == 3 and actor.counters.hosts_retired == 1 and actor.envs[0].dead
        assert hosts[1].stats.timeouts >= 1 and not actor.envs[1].dead
        st = store.stats
        assert st.steps_added == actor.counters.decisions == st.steps_released + st.steps_dropped
        assert len(recs) == actor.counters.episodes_started == actor.counters.episodes_ended
        res = Counter(r.result for r in recs)
        assert res["win"] + res["loss"] > 10 and res["win"] + res["loss"] == st.wins + st.losses
        assert sum(len(s) for s in segs) == st.steps_released
        assert sum(r["decisions"] for r in reports) == actor.counters.decisions
    finally:
        for h in hosts:
            h.close()
    assert mux._closed


def test_actor_rejects_mixed_transports(tmp_path: Path) -> None:
    vocab = Vocab(make_synthetic_vocab())
    h = make(tmp_path, "--no-hashes")
    try:
        with pytest.raises(ValueError, match="PipeMux"):
            Actor([h, FakeHost(vocab, seed=0)], UniformPolicy(0), Tokenizer(vocab),
                  EpisodeSampler(EpisodeDist(), vocab, 0), ActorConfig(mode="eval"))
    finally:
        h.close()


# --------------------------------------------------------------------------- randomized stress (both implementations)


def test_random_streams_on_many_connections_arrive_intact() -> None:
    """Random messages (0 B .. 300 KB), cut into random chunks, on 12 concurrent connections: every
    connection delivers exactly its messages, in order, then EOF."""
    rng = np.random.default_rng(7)
    mux = PipeMux()
    n_conn = 12
    got: list[list[object]] = [[] for _ in range(n_conn)]
    conns = [mux.open(f"stress{i}", got[i].append) for i in range(n_conn)]
    want: list[list[bytes]] = []
    clients = []
    for i, c in enumerate(conns):
        sizes = [int(x) for x in rng.choice([0, 1, 7, 100, 4096, 70_000, 300_000], size=12)]
        msgs = [bytes(rng.integers(0, 256, n, dtype=np.uint8)) for n in sizes]
        want.append(msgs)
        stream = b"".join(msg(m) for m in msgs)
        cuts = np.sort(rng.integers(0, len(stream), size=20))
        chunks = [stream[a:b] for a, b in zip([0, *cuts], [*cuts, len(stream)]) if b > a]
        clients.append(Client(c.name, chunks))
    for cl in clients:
        cl.start()
    drive(mux, lambda: all(len(g) == len(w) + 1 for g, w in zip(got, want)), timeout=30)
    for g, w in zip(got, want):
        assert g[:-1] == w and g[-1] is EOF
    assert mux.n_open == 0
    mux.close()


def test_writes_beyond_the_pipe_buffer_complete_later() -> None:
    """Writes the host does not read yet pend (the pipe buffer is full) and complete on the port;
    the host then reads every byte in order."""
    mux = PipeMux()
    conn = mux.open("pending", lambda item: None)
    seen: list[bytes] = []
    total = 40 * 64 * 1024
    ready = threading.Event()

    def client() -> None:
        with open(conn.name, "r+b", buffering=0) as f:
            f.write(hello_message())
            ready.wait(10)
            buf = bytearray()
            while len(buf) < total + 40 * 4:
                buf += f.read(1 << 16)
            seen.append(bytes(buf))

    t = threading.Thread(target=client, daemon=True)
    t.start()
    drive(mux, lambda: conn.connected)
    payloads = [bytes([i]) * (64 * 1024) for i in range(40)]
    for p in payloads:
        conn.send(msg(p))
    assert mux.writes_pending > 0  # 2.5 MB into a 256 KiB pipe buffer
    ready.set()
    drive(mux, lambda: bool(seen) and mux.writes_pending == 0, timeout=20)
    t.join(5)
    assert seen[0] == b"".join(msg(p) for p in payloads)
    mux.close()


def test_actor_books_empty_spin_polls_as_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Classify exact poll/loop durations independently of OS scheduling and host throughput."""
    from spire import actor as actor_module

    vocab = Vocab(make_synthetic_vocab())
    actor = Actor([], UniformPolicy(0), Tokenizer(vocab), EpisodeSampler(EpisodeDist(), vocab, 0),
                  ActorConfig(mode="eval", max_wait_ms=1e9, yield_gil=False))
    timer = SimpleNamespace(now=1_000_000)

    class ScriptedMux:
        # (completed messages, blocked wait, total poll duration), all durations in ns.
        steps = ((0, 0, 100), (0, 60, 100), (1, 30, 70))
        wait_ns = 83  # nonzero baseline: the actor must book the increment, not the total
        polls = 0

        def poll(self, timeout: float) -> int:
            assert timeout == 0.0  # a pending batch and spin_wait=True must not block
            got, wait, duration = self.steps[self.polls]
            self.polls += 1
            self.wait_ns += wait
            timer.now += duration
            return got

    mux = ScriptedMux()
    actor.mux = cast(PipeMux, mux)
    actor._busy.add(0)
    # Only presence/length are used while this deliberately long batching window stays open.
    actor.ready = [cast(Any, object())]

    def housekeeping() -> None:
        timer.now += 11

    monkeypatch.setattr(actor_module, "time", SimpleNamespace(perf_counter_ns=lambda: timer.now))
    monkeypatch.setattr(actor, "_housekeeping", housekeeping)
    monkeypatch.setattr(actor, "_shutdown", lambda: None)
    actor.run(stop=lambda: mux.polls == len(mux.steps))
    stage = actor.clock.take()
    expected = dict.fromkeys(STAGES, 0)
    expected.update(wait=90, recv=40, idle=162)
    assert stage == expected  # empty polls: 100 + (100-60), plus two 11ns loop remainders
    assert actor._pump_ns == 70
    assert timer.now - 1_000_000 - sum(stage.values()) == 11  # non-idle iteration's housekeeping


def test_actor_reports_accounted_work_with_slow_pipe_hosts(tmp_path: Path) -> None:
    """Real transport integration checks report conservation; stage timing ratios depend on machine load."""
    vocab = Vocab(make_synthetic_vocab())
    tok = Tokenizer(vocab)
    mux = PipeMux()
    hosts = [make(tmp_path, "--no-hashes", "--mean-decisions", "40", "--latency-ms", "2", "--seed", str(s), mux=mux,
                  name=f"slow{s}") for s in range(3)]
    try:
        reports: list[dict] = []
        store = RolloutStore(lambda seg: None, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
        actor = Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
                      ActorConfig(max_batch=64, min_group=1, max_wait_ms=1.0, report_interval=0.5), store=store,
                      on_report=reports.append)
        actor.run(max_decisions=300, max_seconds=30.0)
    finally:
        for h in hosts:
            h.close()
    tot = RunTotals()
    for r in reports:
        tot.add(r)
    us = tot.us_per_decision()
    assert tot.decisions == actor.counters.decisions >= 300
    assert all(np.isfinite(value) and value >= 0 for value in us.values()), us
    assert us["recv"] > 0 and us["python"] > 0, us
    assert store.stats.steps_added == tot.decisions == store.stats.steps_released + store.stats.steps_dropped
