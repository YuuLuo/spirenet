"""HostProcess against scripts/fake_spirehost.py (a real subprocess speaking the protocol)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from spire.host import (
    ActionRejected,
    HostConfig,
    HostCrashed,
    HostProcess,
    HostStateError,
    HostTimeout,
    HostUnrecoverable,
)
from spire.schema import ErrorCode, Result
from spire.synth import FakeHost, FakeHostConfig
from spire.vocab import Vocab, make_synthetic_vocab

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "fake_spirehost.py")


def make(tmp_path: Path, *args: str, **cfg: object) -> HostProcess:
    opts: dict[str, object] = dict(start_timeout=30.0, reset_timeout=5.0, step_timeout=5.0, quit_timeout=5.0,
                                   log_path=str(tmp_path / "host.stderr.log"), name="t")
    opts.update(cfg)
    return HostProcess(HostConfig(command=(sys.executable, "-B", SCRIPT, "--serve", *args), **opts))  # type: ignore[arg-type]


def play(h, seed: int | None, policy: int, max_steps: int = 10_000) -> list[bytes]:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(policy)
    f = h.reset(seed, (4,), 0)
    out = [bytes(f.payload)]
    for _ in range(max_steps):
        if f.is_terminal:
            break
        f = h.step(int(rng.integers(f.n_actions)))
        out.append(bytes(f.payload))
    return out


def test_subprocess_matches_in_process_fake(tmp_path: Path) -> None:
    vocab = Vocab(make_synthetic_vocab())
    with make(tmp_path, "--seed", "1", "--mean-decisions", "40", "--stderr-every", "10",
              expected_hashes=(vocab.hash, 0), verify_hashes=True) as h:
        a = [play(h, None, 0), play(h, 99, 1)]
        assert h.stats.finished == 2 and h.stats.decisions == sum(len(x) - 1 for x in a)
        assert h.last_episode is not None and h.last_episode.seed == 99
        assert h.last_episode.result in (Result.WIN, Result.LOSS, Result.ABORTED)
        proc = h._proc
    assert proc is not None and proc.returncode == 0 and not h.alive
    ref = FakeHost(vocab, seed=1, config=FakeHostConfig(mean_decisions=40))
    assert a == [play(ref, None, 0), play(ref, 99, 1)]
    log = (tmp_path / "host.stderr.log").read_text(encoding="utf-8")
    assert "fake_spirehost" in log and "diag:" in log and "--- started pid" in log


def test_error_frame_resend_over_pipe(tmp_path: Path) -> None:
    with make(tmp_path, "--mean-decisions", "30") as h:
        f = h.reset(5, (4,), 0)
        with pytest.raises(ActionRejected) as e:
            h.step(f.n_actions + 3)
        assert e.value.code == ErrorCode.BAD_INDEX and e.value.frame is not None
        assert bytes(e.value.frame.payload) == bytes(f.payload)
        assert h.step(0).step_id == 1 and h.stats.errors == 1


def test_timeout_kills_and_restarts(tmp_path: Path) -> None:
    with make(tmp_path, "--hang-after", "3", "--mean-decisions", "500", step_timeout=0.5) as h:
        pid0 = h.pid
        f = h.reset(1, (4,), 0)
        f = h.step(0)
        f = h.step(0)
        with pytest.raises(HostTimeout):
            h.step(0)
        s = h.stats
        assert (s.timeouts, s.restarts, s.lost) == (1, 1, 1) and h.episode is None and h.last_frame is None
        assert h.pid != pid0 and h.alive
        with pytest.raises(HostStateError):
            h.step(0)
        assert h.reset(2, (4,), 0).step_id == 0  # the restarted process serves a new episode


def test_crash_is_detected_with_stderr_tail(tmp_path: Path) -> None:
    with make(tmp_path, "--crash-after", "2", "--mean-decisions", "500") as h:
        h.reset(1, (4,), 0)
        h.step(0)
        with pytest.raises(HostCrashed) as e:
            h.step(0)
        assert e.value.returncode == 17 and any("injected crash" in line for line in e.value.stderr_tail)
        assert (h.stats.crashes, h.stats.restarts, h.stats.lost) == (1, 1, 1)
        h.reset(1, (4,), 0)
        h.step(0)  # the new process counts its own steps: crashes again on its 2nd STEP
        with pytest.raises(HostCrashed):
            h.step(0)
        assert h.stats.restarts == 2


def test_restart_budget_exhaustion(tmp_path: Path) -> None:
    with make(tmp_path, "--crash-after", "1", max_restarts=1) as h:
        h.reset(1, (4,), 0)
        with pytest.raises(HostCrashed):
            h.step(0)
        h.reset(1, (4,), 0)
        with pytest.raises(HostUnrecoverable):
            h.step(0)
        with pytest.raises(HostUnrecoverable):
            h.reset(1, (4,), 0)
        assert h.stats.crashes == 2 and h.stats.restarts == 1


def test_start_timeout_covers_slow_boot(tmp_path: Path) -> None:
    with make(tmp_path, "--boot-delay", "1.0", reset_timeout=0.2, step_timeout=0.2) as h:
        f = h.reset(1, (4,), 0)  # first reply after start may take start_timeout
        assert f.step_id == 0
        h.step(0)


def test_unstartable_command(tmp_path: Path) -> None:
    with pytest.raises(HostUnrecoverable, match="cannot start"):
        HostProcess(HostConfig(command=(str(tmp_path / "missing.exe"), "--serve"),
                               log_path=str(tmp_path / "x.log")))


def test_close_sends_quit(tmp_path: Path) -> None:
    h = make(tmp_path)
    proc = h._proc
    assert proc is not None
    h.reset(None, (4,), 0)
    h.close()
    assert proc.returncode == 0
    h.close()  # idempotent
    with pytest.raises(HostStateError):
        h.reset(None, (4,), 0)
