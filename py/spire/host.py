"""Host wrappers: a common :class:`Host` base (episode protocol + accounting) and
:class:`HostProcess`, the real SpireHost subprocess over binary stdin/stdout pipes.

Protocol per episode (schema 3, 7)::

    RESET  -> OBS(step 0) | TERMINAL
    STEP   -> OBS(step+1) | TERMINAL | ERROR followed by the same OBS re-sent
    QUIT   -> (process exits)

``Host`` turns this into ``reset(...) -> Frame`` / ``step(i) -> Frame`` and keeps per-episode
and cumulative statistics. Transport failures (timeout, process death, broken framing) raise
:class:`HostTimeout` / :class:`HostCrashed`; ``HostProcess`` restarts the process before
raising, so the caller only has to start a new episode.

Asynchronous use (the actor drives many hosts from one thread): :meth:`Host.route` sends every
reply / transport event to a shared sink as ``sink(generation, item)`` instead of the host's own
queue; the caller then uses :meth:`Host.send_reset` / :meth:`Host.send_step` (send only),
hands each routed item to :meth:`Host.deliver` (protocol + accounting, same checks as the
blocking API) and calls :meth:`Host.expire` when :meth:`Host.reply_timeout` has elapsed.
Items whose ``generation`` differs from :attr:`Host.generation` come from a process that was
already replaced and must be ignored.
"""
from __future__ import annotations

import ctypes
import io
import logging
import logging.handlers
import os
import queue
import subprocess
import sys
import threading
import time
import tomllib
import weakref
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import IO, Callable, Mapping, NoReturn, Self, Sequence

from .proto import (
    Frame,
    ProtocolError,
    ResetCmd,
    encode_quit,
    encode_reset,
    encode_step,
    parse_frame,
    read_message,
    write_message,
)
from .pipes import EOF as PIPE_EOF
from .pipes import PipeConn, PipeMux, PosixPipeConn, PosixPipeMux, new_mux
from .schema import (
    RESET_FLAGS_DEFAULT,
    TERMINAL_RESULT,
    TRUNCATION_REASONS,
    FrameFlag,
    G,
    MsgType,
    ResetFlag,
    Result,
    SchemaError,
    ViewMode,
)
from .sysutil import affinity_restricted, apply_placement
from .vocab import check_hashes

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO / "runs" / "host-logs"
REAL_HOST_CONFIG = REPO / "py" / "configs" / "real_host.toml"
REAL_HOST_CONFIG_ENV = "SPIRE_REAL_HOST_CONFIG"

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- errors


class HostError(RuntimeError):
    """Base class of host wrapper errors."""


class HostTimeout(HostError):
    """No reply within the per-call timeout (the process is killed and restarted)."""


class HostCrashed(HostError):
    """The process died or its stdout framing broke (it is restarted)."""

    def __init__(self, msg: str, returncode: int | None = None, stderr_tail: Sequence[str] = ()) -> None:
        tail = "\n".join(stderr_tail[-20:])
        super().__init__(f"{msg} (returncode={returncode})" + (f"\n--- stderr tail ---\n{tail}" if tail else ""))
        self.returncode = returncode
        self.stderr_tail = tuple(stderr_tail)


class HostUnrecoverable(HostError):
    """Restart budget exhausted; the wrapper refuses further calls."""


class HostStateError(HostError):
    """API misuse, e.g. ``step`` without a pending OBS frame."""


class ActionRejected(HostError):
    """The host answered with an ERROR frame (schema 7.3).

    ``frame`` is the OBS frame the host re-sent afterwards (``None`` for a rejected RESET);
    the wrapper is back in that state, so the caller may retry or start a new episode.
    """

    def __init__(self, code: int, error_frame: Frame, frame: Frame | None) -> None:
        super().__init__(f"host rejected command: error code {code}")
        self.code = code
        self.error_frame = error_frame
        self.frame = frame


# --------------------------------------------------------------------------- accounting


@dataclass
class EpisodeStats:
    """One episode's reproducibility record and counters."""

    seed: int | None
    characters: tuple[int, ...]
    ascension: int
    episode_id: int | None = None
    decisions: int = 0
    auto_steps: int = 0
    errors: int = 0
    overflow: bool = False
    result: int = Result.IN_PROGRESS
    actions: list[int] = field(default_factory=list)
    # the STEP that was outstanding when the episode was lost (timeout / crash / bad frame); -1 = none. With
    # ``actions`` it is the complete action sequence that reproduces the loss (seed + RESET + actions + unanswered).
    unanswered: int = -1
    # from the TERMINAL frame (errata-v1 section 5): TerminalReason, AbortReason, win counters
    terminal_reason: int = 0
    abort_reason: int = 0
    combats_won: int = 0
    elites_won: int = 0
    bosses_won: int = 0


@dataclass
class HostStats:
    episodes: int = 0  # RESETs sent
    finished: int = 0  # TERMINAL frames received (= wins + losses + truncated + aborted)
    wins: int = 0
    losses: int = 0  # TERMINAL with RESULT=LOSS that is a death (TERMINAL_REASON DEATH, or NONE)
    truncated: int = 0  # TERMINAL with RESULT=LOSS and a truncation reason (decision / combat-turn cap ...)
    aborted: int = 0  # TERMINAL with RESULT=ABORTED
    abandoned: int = 0  # episode replaced by a RESET before TERMINAL
    lost: int = 0  # episode lost to a timeout / crash / malformed frame
    decisions: int = 0  # OBS frames delivered
    auto_steps: int = 0
    errors: int = 0  # ERROR frames
    overflow_episodes: int = 0
    bad_frames: int = 0
    timeouts: int = 0
    crashes: int = 0
    restarts: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


# --------------------------------------------------------------------------- base

_MSG_OBS, _MSG_ERROR = int(MsgType.OBS), int(MsgType.ERROR)
_F_OVERFLOW = int(FrameFlag.OVERFLOW)
_G_REASON, _G_ABORT, _G_DECISIONS = int(G.TERMINAL_REASON), int(G.ABORT_REASON), int(G.POLICY_DECISIONS)
_G_COMBATS, _G_ELITES, _G_BOSSES = int(G.COMBATS_WON), int(G.ELITES_WON), int(G.BOSSES_WON)
_TRUNCATIONS = frozenset(int(r) for r in TRUNCATION_REASONS)
_TERMINAL_RESULT = {int(k): int(v) for k, v in TERMINAL_RESULT.items()}


class Pending(IntEnum):
    """The reply a host owes (one outstanding request at a time)."""

    NONE = 0
    RESET = 1  # OBS step 0 | TERMINAL | ERROR
    STEP = 2  # OBS | TERMINAL | ERROR
    RESEND = 3  # the OBS frame re-sent after an ERROR


_P_NONE, _P_STEP = Pending.NONE, Pending.STEP

RouteSink = Callable[[int, object], None]
"""``sink(generation, item)``: item is a reply payload, or a transport event (EOF marker / exception)."""


class Host(ABC):
    """Episode protocol shared by :class:`HostProcess` and ``synth.FakeHost``.

    Subclasses provide the transport (``_send`` a complete wire message, ``_recv`` one
    payload within a timeout, optional routing of replies to a shared sink) and ``close``.
    """

    def __init__(
        self,
        *,
        reset_timeout: float,
        step_timeout: float,
        verify_hashes: bool = False,
        expected_hashes: tuple[int, int] | None = None,
        training_hidden: bool = False,
    ) -> None:
        # notes/critic/contract.md 1.2 (the Python lock): only a host made with training_hidden=True (the training
        # actors of a critic run, spire.train) may send RESET flag HIDDEN
        self.training_hidden = training_hidden
        self.reset_timeout = reset_timeout
        self.step_timeout = step_timeout
        self.verify_hashes = verify_hashes
        self.expected_hashes = expected_hashes
        self.stats = HostStats()
        self.episode: EpisodeStats | None = None
        self.last_episode: EpisodeStats | None = None
        self.last_frame: Frame | None = None
        self.generation = 0  # bumped by every process (re)start; tags routed items
        self.pending = Pending.NONE
        self._pending_action = -1
        self._resend_of: Frame | None = None
        self._sink: RouteSink | None = None
        self._fresh = False  # no reply received since the last (re)start (HostProcess: boot timeout)

    # transport -----------------------------------------------------------------
    @abstractmethod
    def _send(self, message: bytes) -> None: ...

    @abstractmethod
    def _recv(self, timeout: float) -> bytes | bytearray: ...

    @abstractmethod
    def close(self) -> None: ...

    def route(self, sink: RouteSink | None) -> None:
        """Deliver replies to ``sink(generation, item)`` (``None``: back to blocking use).
        Call only while no request is outstanding."""
        if self.pending != Pending.NONE:
            raise HostStateError("route() while a reply is outstanding")
        self._sink = sink

    def restart(self, reason: str) -> None:
        """Bring the transport back to a clean state after a protocol violation (the episode is
        lost). Processes are killed and respawned; in-process hosts only invalidate replies in
        flight (new generation)."""
        self._lose_episode()
        self.generation += 1

    def reply_timeout(self) -> float:
        """Seconds the outstanding request may take."""
        return self.reset_timeout if self.pending == Pending.RESET else self.step_timeout

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # blocking episode API ------------------------------------------------------
    def reset(
        self,
        seed: int | None,
        characters: Sequence[int],
        ascension: int = 0,
        *,
        view_mode: int = ViewMode.TEAM,
        view_seat: int = -1,
        flags: int = int(RESET_FLAGS_DEFAULT),
    ) -> Frame:
        """Start an episode. ``characters`` are ``character`` vocab ids, one per seat."""
        return self.reset_with(ResetCmd(tuple(characters), ascension, seed, view_mode, view_seat, flags))

    def reset_with(self, cmd: ResetCmd) -> Frame:
        self._check_unrouted()
        self.send_reset(cmd)
        f = self.receive(self._recv_guarded(self.reset_timeout))
        if f.is_error:
            raise ActionRejected(f.error_code, f, None)
        return f

    def step(self, action_index: int, *, step_id: int | None = None) -> Frame:
        """Answer the pending OBS frame with ``action_index`` (its ACTIONS row).

        ``step_id`` defaults to the pending frame's STEP_ID. Raises :class:`ActionRejected`
        if the host replies with ERROR (the re-sent OBS becomes the pending frame).
        """
        self._check_unrouted()
        self.send_step(action_index, step_id=step_id)
        f = self.receive(self._recv_guarded(self.step_timeout))
        if f.is_error:
            resent = self.receive(self._recv_guarded(self.step_timeout))
            raise ActionRejected(f.error_code, f, resent)
        return f

    # asynchronous episode API --------------------------------------------------
    def send_reset(self, cmd: ResetCmd) -> None:
        """Send RESET without waiting; the reply goes to :meth:`receive` / :meth:`deliver`."""
        if self.pending != Pending.NONE:
            raise HostStateError(f"send_reset() while {self.pending.name} reply is outstanding")
        assert not cmd.flags & int(ResetFlag.HIDDEN) or self.training_hidden, \
            "RESET flag HIDDEN (1024) needs a host constructed with training_hidden=True"
        if self.episode is not None:
            self.stats.abandoned += 1
        self.episode = EpisodeStats(cmd.seed, cmd.characters, cmd.ascension)
        self.last_frame = None
        self.stats.episodes += 1
        self._send_guarded(encode_reset(cmd))
        self.pending = Pending.RESET

    def send_step(self, action_index: int, *, step_id: int | None = None) -> None:
        """Send STEP for the pending OBS frame without waiting."""
        last, ep = self.last_frame, self.episode
        if last is None or last.msg_type != _MSG_OBS or ep is None or self.pending != _P_NONE:
            raise HostStateError("step needs a pending OBS frame and no outstanding reply; call reset() first")
        msg = encode_step(last.step_id if step_id is None else step_id, action_index)
        try:
            self._send(msg)
        except (HostError, ProtocolError):
            self._lose_episode()
            raise
        self.pending = Pending.STEP
        self._pending_action = action_index

    def receive(self, payload: bytes | bytearray) -> Frame:
        """Process one reply payload to the outstanding request.

        Returns the OBS / TERMINAL frame, or an ERROR frame (counted). After a STEP's ERROR the
        host owes the re-sent OBS (``pending == RESEND``); after a RESET's ERROR the episode is
        gone. Malformed or unexpected frames raise (the episode is lost).
        """
        self._fresh = False
        w = self.pending
        if w == Pending.NONE:
            self._lose_episode()
            raise ProtocolError("unsolicited message from host")
        if w == Pending.RESEND:
            self.pending = Pending.NONE
            return self._accept(payload, resend_of=self._resend_of)
        ep = self.episode
        f = self._parse(payload)
        last = self.last_frame
        if (w == _P_STEP and f.msg_type == _MSG_OBS and ep is not None and last is not None
                and f.episode_id == ep.episode_id and f.step_id == last.step_id + 1 and not f.flags & _F_OVERFLOW):
            # the per-decision case, with exactly the bookkeeping of the general path below
            st = self.stats
            ep.decisions += 1
            st.decisions += 1
            ep.auto_steps += f.auto_steps
            st.auto_steps += f.auto_steps
            self.last_frame = f
            self.pending = Pending.NONE
            ep.actions.append(self._pending_action)
            return f
        f = self._accept(payload, frame=f)
        if f.is_error:
            self._count_error()
            if w == Pending.RESET:
                self.episode = None
                self.pending = Pending.NONE
            else:
                self.pending = Pending.RESEND
                self._resend_of = self.last_frame
            return f
        self.pending = Pending.NONE
        if w == Pending.STEP and ep is not None:
            ep.actions.append(self._pending_action)
        return f

    def deliver(self, item: object) -> Frame:
        """Routed counterpart of the blocking receive: ``item`` from the route sink."""
        if type(item) is bytes or isinstance(item, (bytes, bytearray)):
            return self.receive(item)  # type: ignore[arg-type]
        self._lose_episode()
        self._transport_failure(item)

    def expire(self) -> NoReturn:
        """The outstanding request exceeded :meth:`reply_timeout` (routed use)."""
        self._lose_episode()
        self._timeout_failure()

    def _transport_failure(self, item: object) -> NoReturn:
        self.generation += 1
        raise HostCrashed(f"transport failure: {item!r}")

    def _timeout_failure(self) -> NoReturn:
        self.generation += 1  # a late reply must not be taken for the next request's
        raise HostTimeout("no reply within the timeout")

    # internals -----------------------------------------------------------------
    def _check_unrouted(self) -> None:
        if self._sink is not None:
            raise HostStateError("blocking reset()/step() on a routed host; use send_*/deliver")

    def _send_guarded(self, message: bytes) -> None:
        try:
            self._send(message)
        except (HostError, ProtocolError):
            self._lose_episode()
            raise

    def _recv_guarded(self, timeout: float) -> bytes | bytearray:
        try:
            return self._recv(timeout)
        except (HostError, ProtocolError):
            self._lose_episode()
            raise

    def _lose_episode(self) -> None:
        if self.episode is not None:
            self.stats.lost += 1
            if self.pending == _P_STEP:
                self.episode.unanswered = self._pending_action
            self.last_episode = self.episode
            self.episode = None
        self.last_frame = None
        self.pending = Pending.NONE

    def _count_error(self) -> None:
        self.stats.errors += 1
        if self.episode is not None:
            self.episode.errors += 1

    def _parse(self, payload: bytes | bytearray) -> Frame:
        try:
            f = parse_frame(payload, verify_hashes=self.verify_hashes)
            if self.expected_hashes is not None and f.msg_type != _MSG_ERROR:
                check_hashes(f.vocab_hash, f.catalog_hash, *self.expected_hashes)
        except SchemaError:
            self.stats.bad_frames += 1
            self._lose_episode()
            raise
        return f

    def _check_terminal(self, f: Frame) -> None:
        """errata-v1 P-5 / PY-10: a TERMINAL's STEP_ID is the episode's policy-decision count, i.e. the
        last OBS STEP_ID + 1 (0 without decisions) and GLOBAL[POLICY_DECISIONS]; RESULT agrees with
        GLOBAL[TERMINAL_REASON] (NONE = a host predating errata-v1: only the STEP_ID rule applies)."""
        want = 0 if self.last_frame is None else self.last_frame.step_id + 1
        g = f.gvals
        reason = g[_G_REASON]
        problem = ""
        if f.step_id != want:
            problem = f"TERMINAL step_id {f.step_id}, expected {want} (last OBS + 1)"
        elif reason != 0 and g[_G_DECISIONS] != f.step_id:
            problem = f"TERMINAL GLOBAL[POLICY_DECISIONS] {g[_G_DECISIONS]} != step_id {f.step_id}"
        elif reason != 0 and _TERMINAL_RESULT.get(reason) != f.result:
            problem = f"TERMINAL result {f.result} with terminal_reason {reason}"
        if problem:
            self._lose_episode()
            raise ProtocolError(problem)

    def _accept(self, payload: bytes | bytearray, resend_of: Frame | None = None, frame: Frame | None = None) -> Frame:
        f = self._parse(payload) if frame is None else frame
        if f.is_error:
            if resend_of is not None:
                self._lose_episode()
                raise ProtocolError("ERROR frame where the re-sent OBS frame was expected")
            return f
        ep = self.episode
        if ep is None:
            self._lose_episode()
            raise ProtocolError(f"unexpected frame outside an episode: {f!r}")
        if ep.episode_id is None:
            ep.episode_id = f.episode_id
        elif f.episode_id != ep.episode_id:
            self._lose_episode()
            raise ProtocolError(f"episode id changed {ep.episode_id} -> {f.episode_id}")
        if resend_of is not None:
            if not f.is_obs or f.step_id != resend_of.step_id:
                self._lose_episode()
                raise ProtocolError(f"after ERROR expected OBS step {resend_of.step_id}, got {f!r}")
            self.last_frame = f
            return f
        if f.is_obs:
            want = 0 if self.last_frame is None else self.last_frame.step_id + 1
            if f.step_id != want:
                self._lose_episode()
                raise ProtocolError(f"OBS step_id {f.step_id}, expected {want}")
            ep.decisions += 1
            self.stats.decisions += 1
            if f.overflow and not ep.overflow:
                ep.overflow = True
                self.stats.overflow_episodes += 1
                log.warning("episode %s (seed %s): frame overflow - episode must be discarded", ep.episode_id, ep.seed)
        if f.is_terminal:
            self._check_terminal(f)
        ep.auto_steps += f.auto_steps
        self.stats.auto_steps += f.auto_steps
        if f.is_terminal:
            g = f.gvals
            ep.result = f.result
            ep.terminal_reason, ep.abort_reason = g[_G_REASON], g[_G_ABORT]
            ep.combats_won, ep.elites_won, ep.bosses_won = g[_G_COMBATS], g[_G_ELITES], g[_G_BOSSES]
            self.stats.finished += 1
            if f.result == Result.WIN:
                self.stats.wins += 1
            elif f.result == Result.LOSS:
                if ep.terminal_reason in _TRUNCATIONS:
                    self.stats.truncated += 1
                else:
                    self.stats.losses += 1
            else:
                self.stats.aborted += 1
            self.last_episode = ep
            self.episode = None
        self.last_frame = f
        return f


# --------------------------------------------------------------------------- subprocess host

_EOF = PIPE_EOF  # end-of-stream marker of both transports


@dataclass(frozen=True)
class RealHostSpec:
    """How to launch the real SpireHost protocol server and which vocab / catalog it speaks.

    Read from a TOML file (:func:`real_host_spec`), not hard-coded: until the H1 integration the
    server is the obs team's build (``dotnet ...\\SpireHost.Obs.dll --serve``, errata-v1 PY-7);
    afterwards it is the main ``SpireHost.exe --serve`` with ``vocab\\*.json`` next to it (contract
    request O4) - only the file changes. ``{repo}`` in any string = the repository root.
    """

    command: tuple[str, ...]  # complete --serve command line (the wrapper appends --pipe NAME)
    vocab: Path
    catalog: Path
    env: Mapping[str, str] | None = None  # extra environment variables for the host process
    source: Path | None = None  # the TOML file

    def missing(self) -> list[str]:
        """Files this spec needs that do not exist (empty: the host can be started)."""
        out = [str(p) for p in (self.vocab, self.catalog) if not p.is_file()]
        exe = [Path(c) for c in self.command if c.lower().endswith((".dll", ".exe")) and ("\\" in c or "/" in c)]
        out += [str(p) for p in exe if not p.is_file()]
        if not self.command:
            out.append("<empty command>")
        return out

    def command_with(self, *extra: str) -> tuple[str, ...]:
        return (*self.command, *extra)

    def process_env(self) -> dict[str, str] | None:
        """The host's environment: ours plus :attr:`env` (``None`` = inherit unchanged)."""
        if not self.env:
            return None
        return {**os.environ, **self.env}


def real_host_spec(path: str | Path | None = None) -> RealHostSpec:
    """Load the real-host launch spec: ``path``, else ``$SPIRE_REAL_HOST_CONFIG``, else
    ``py/configs/real_host.toml``. Keys: ``command`` (list of strings), ``vocab``, ``catalog``
    (paths), optional ``[env]`` table; relative paths are relative to the TOML file."""
    p = Path(str(path).replace("{repo}", str(REPO))) if path is not None else         Path(os.environ.get(REAL_HOST_CONFIG_ENV) or REAL_HOST_CONFIG)
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    unknown = set(raw) - {"command", "vocab", "catalog", "env"}
    if unknown:
        raise ValueError(f"{p}: unknown keys {sorted(unknown)}")

    def sub(s: str) -> str:
        return str(s).replace("{repo}", str(REPO))

    def path_of(s: str) -> Path:
        q = Path(sub(s))
        return q if q.is_absolute() else (p.parent / q).resolve()

    cmd = raw.get("command")
    if not isinstance(cmd, list) or not cmd or not all(isinstance(c, str) for c in cmd):
        raise ValueError(f"{p}: 'command' must be a non-empty list of strings")
    holes = [str(v) for v in (*cmd, raw.get("vocab", ""), raw.get("catalog", "")) if "<FILL" in str(v)]
    if holes:
        raise ValueError(f"{p}: the launch spec still has placeholders {holes}: fill in the host command and the "
                         "vocab / catalog paths of the integrated build (notes/l1-runbook.md)")
    env = {str(k): sub(v) for k, v in (raw.get("env") or {}).items()}
    return RealHostSpec(tuple(sub(c) for c in cmd), path_of(raw["vocab"]), path_of(raw["catalog"]), env or None, p)


def _default_command() -> tuple[str, ...]:
    try:
        return real_host_spec().command
    except (OSError, ValueError, KeyError):
        return ()


@dataclass(frozen=True)
class HostConfig:
    command: tuple[str, ...] = field(default_factory=_default_command)  # default: the real host (real_host_spec)
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    start_timeout: float = 120.0  # first reply after a (re)start: engine boot
    reset_timeout: float = 30.0
    step_timeout: float = 10.0
    quit_timeout: float = 5.0
    max_restarts: int = 5  # within restart_window seconds
    restart_window: float = 600.0
    log_path: str | None = None  # default: <repo>/runs/host-logs/<name>.stderr.log
    log_max_bytes: int = 8 << 20
    log_backups: int = 3
    stderr_tail: int = 200
    verify_hashes: bool = False
    expected_hashes: tuple[int, int] | None = None  # (vocab_hash, catalog_hash)
    name: str = "host0"
    transport: str = "stdio"  # "stdio": frames on stdin/stdout; "pipe": named pipe (notes/transport-spec.md)
    priority: str = ""  # process priority class of the host ("" = inherit; spire.sysutil.PRIORITY_CLASSES)
    cpus: str = ""  # CPU affinity of the host ("" = any; "p" / "e" / "0,1,10-13", spire.sysutil.parse_cpus)


TRANSPORTS = ("stdio", "pipe")


class HostProcess(Host):
    """The SpireHost process.

    ``transport="stdio"``: unbuffered binary stdin/stdout pipes; stdout is read by a thread that
    pushes whole messages into a queue (so every call has a real timeout).
    ``transport="pipe"``: frames travel over a named pipe owned by a :class:`~spire.pipes.PipeMux`
    (one completion port per actor thread, no reader threads); the host gets ``--pipe <name>``
    appended to its command line. Pass the actor's shared ``mux``; without one the host makes a
    private mux (blocking use). On Linux the "pipe" transport is a :class:`~spire.pipes.PosixPipeMux`:
    the host keeps plain stdin / stdout (no ``--pipe``), the actor thread reads them through epoll.

    stderr lines go to a rotating log file and an in-memory tail (one low-traffic thread).
    On timeout, death or broken framing the process is killed and restarted (bounded by
    ``max_restarts`` per ``restart_window``) and the error is raised to the caller.
    """

    def __init__(self, config: HostConfig = HostConfig(), *, mux: PipeMux | PosixPipeMux | None = None,
                 training_hidden: bool = False) -> None:
        super().__init__(
            reset_timeout=config.reset_timeout,
            step_timeout=config.step_timeout,
            verify_hashes=config.verify_hashes,
            expected_hashes=config.expected_hashes,
            training_hidden=training_hidden,
        )
        if config.transport not in TRANSPORTS:
            raise ValueError(f"unknown transport {config.transport!r}; expected one of {TRANSPORTS}")
        if not config.command:
            raise HostUnrecoverable(f"no host command: pass HostConfig.command or create {REAL_HOST_CONFIG}")
        self.cfg = config
        self.mux: PipeMux | PosixPipeMux | None = None
        if config.transport == "pipe":
            self.mux = mux if mux is not None else new_mux()
            self.mux.acquire()
        self._conn: PipeConn | None = None
        self.stderr_tail: deque[str] = deque(maxlen=config.stderr_tail)
        self._restarts: deque[float] = deque()
        self._closed = False
        self._dead = False
        self._proc: subprocess.Popen[bytes] | None = None
        self._q: queue.Queue[object] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._finalizer: weakref.finalize | None = None
        self._fresh = True
        self._logger, self._handler = self._make_logger()
        self._spawn()

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    @property
    def reader_threads(self) -> bool:
        """True for the stdio transport: replies arrive on a reader thread (it needs the GIL)."""
        return self.mux is None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # lifecycle -----------------------------------------------------------------
    def _make_logger(self) -> tuple[logging.Logger, logging.Handler]:
        path = Path(self.cfg.log_path) if self.cfg.log_path else DEFAULT_LOG_DIR / f"{self.cfg.name}.stderr.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        lg = logging.getLogger(f"spire.host.stderr.{self.cfg.name}.{id(self):x}")
        lg.propagate = False
        lg.setLevel(logging.INFO)
        h = logging.handlers.RotatingFileHandler(
            path, maxBytes=self.cfg.log_max_bytes, backupCount=self.cfg.log_backups, encoding="utf-8"
        )
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        lg.addHandler(h)
        return lg, h

    def _spawn(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]
        gen = self.generation + 1
        command = list(self.cfg.command)
        conn: PipeConn | None = None
        if self.mux is not None:
            conn = self.mux.open(self.cfg.name, lambda item: self._dispatch(gen, item))
            if not isinstance(conn, PosixPipeConn):
                command += ["--pipe", conn.name]
        pipe = self.mux is not None
        stdin: int = subprocess.DEVNULL if pipe else subprocess.PIPE
        stdout: int = subprocess.DEVNULL if pipe else subprocess.PIPE
        if isinstance(conn, PosixPipeConn):  # the host's stdin / stdout are the mux's pipes
            stdin, stdout = conn.child_fds
        try:
            proc = subprocess.Popen(
                command, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE,
                bufsize=0, cwd=self.cfg.cwd, env=dict(self.cfg.env) if self.cfg.env is not None else None,
                creationflags=flags,
            )
        except OSError as e:
            if conn is not None:
                conn.close()
            self._dead = True
            raise HostUnrecoverable(f"cannot start {self.cfg.command!r}: {e}") from e
        if isinstance(conn, PosixPipeConn):
            conn.spawned()
            conn.client_pid = proc.pid
        _bind_to_job(proc)
        # a host inherits its creator's affinity: an actor pinned to the P-cores must not pin its hosts too
        cpus = self.cfg.cpus or ("all" if affinity_restricted() else "")
        if self.cfg.priority or cpus:
            apply_placement(self.cfg.priority, cpus, int(proc._handle) if sys.platform == "win32" else proc.pid)  # type: ignore[attr-defined]
        self._proc = proc
        self._conn = conn
        self._q = queue.Queue()
        self.generation = gen
        assert proc.stderr is not None
        self._threads = [
            threading.Thread(target=_pump_stderr, args=(proc.stderr, self._logger, self.stderr_tail), daemon=True,
                             name=f"{self.cfg.name}-stderr"),
        ]
        if conn is None:
            assert proc.stdout is not None
            # buffered: a whole frame (prefix + payload) usually arrives in one read call
            self._threads.append(threading.Thread(
                target=_pump_stdout, args=(io.BufferedReader(proc.stdout, 1 << 18),  # type: ignore[arg-type]
                                           lambda item: self._dispatch(gen, item)),
                daemon=True, name=f"{self.cfg.name}-stdout"))
        for t in self._threads:
            t.start()
        self._finalizer = weakref.finalize(self, _kill_quietly, proc)
        self._fresh = True
        self._logger.info("--- started pid %d: %s", proc.pid, " ".join(command))

    def _kill(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.error("%s: pid %d did not exit after kill", self.cfg.name, proc.pid)
        self._join_and_close(proc)

    def _join_and_close(self, proc: subprocess.Popen[bytes]) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        for t in self._threads:
            t.join(timeout=2)
        for s in (proc.stdin, proc.stdout, proc.stderr):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        if self._finalizer is not None:
            self._finalizer.detach()
        self._proc = None

    def _dispatch(self, gen: int, item: object) -> None:
        """Reader-thread delivery: to the route sink if one is set, else the private queue."""
        sink = self._sink
        if sink is not None:
            sink(gen, item)
        elif gen == self.generation:
            self._q.put(item)

    def _fail(self, exc: HostError) -> NoReturn:
        if isinstance(exc, HostTimeout):
            self.stats.timeouts += 1
        else:
            self.stats.crashes += 1
        self._logger.info("--- failure: %s", str(exc).splitlines()[0])
        self._respawn(exc)
        raise exc

    def _respawn(self, cause: BaseException | None) -> None:
        """Kill the process and start a new one (bounded by the restart budget)."""
        self._kill()
        if self._closed:
            return
        now = time.monotonic()
        while self._restarts and now - self._restarts[0] > self.cfg.restart_window:
            self._restarts.popleft()
        if len(self._restarts) >= self.cfg.max_restarts:
            self._dead = True
            raise HostUnrecoverable(
                f"{self.cfg.name}: {len(self._restarts)} restarts within {self.cfg.restart_window:.0f}s"
            ) from cause
        self._restarts.append(now)
        self.stats.restarts += 1
        self._spawn()

    def restart(self, reason: str) -> None:
        self._lose_episode()
        self.stats.crashes += 1
        self._logger.info("--- restart: %s", reason)
        self._respawn(None)

    def reply_timeout(self) -> float:
        t = super().reply_timeout()
        return max(t, self.cfg.start_timeout) if self._fresh else t

    def _transport_failure(self, item: object) -> NoReturn:
        proc = self._proc
        rc = None if proc is None else proc.poll()
        if item is _EOF:
            if proc is not None and rc is None:
                try:
                    rc = proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            self._fail(HostCrashed(f"{self.cfg.name}: stdout closed", rc, tuple(self.stderr_tail)))
        self._fail(HostCrashed(f"{self.cfg.name}: broken stdout framing: {item}", rc, tuple(self.stderr_tail)))

    def _timeout_failure(self) -> NoReturn:
        self._fail(HostTimeout(f"{self.cfg.name}: no reply within {self.reply_timeout():.1f}s"))

    def close(self) -> None:
        """Send QUIT, wait ``quit_timeout`` for exit, then kill. Idempotent."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None:
            if proc.poll() is None:
                try:
                    if self._conn is not None:
                        if self._conn.connected:
                            self._conn.send(encode_quit())
                    else:
                        assert proc.stdin is not None
                        write_message(proc.stdin, encode_quit())
                except (OSError, ValueError):
                    pass
                try:
                    proc.wait(timeout=self.cfg.quit_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            self._join_and_close(proc)
        if self.mux is not None:
            self.mux.release()
        self._logger.removeHandler(self._handler)
        self._handler.close()

    # transport -----------------------------------------------------------------
    def _check_usable(self) -> subprocess.Popen[bytes]:
        if self._closed:
            raise HostStateError(f"{self.cfg.name} is closed")
        if self._dead or self._proc is None:
            raise HostUnrecoverable(f"{self.cfg.name} gave up after repeated failures")
        return self._proc

    def _send(self, message: bytes) -> None:
        conn = self._conn
        if conn is not None and conn.state == "connected" and not self._closed and not self._dead:
            try:  # pipe transport, connected: the per-decision case
                conn.mux._write(conn, message)
                return
            except OSError as e:
                proc = self._proc
                self._fail(HostCrashed(f"{self.cfg.name}: write failed: {e}", None if proc is None else proc.poll(),
                                       tuple(self.stderr_tail)))
        proc = self._check_usable()
        try:
            if self._conn is not None:
                self._conn.send(message)
            else:
                assert proc.stdin is not None
                write_message(proc.stdin, message)
        except (OSError, ValueError) as e:
            self._fail(HostCrashed(f"{self.cfg.name}: write failed: {e}", proc.poll(), tuple(self.stderr_tail)))

    def _recv(self, timeout: float) -> bytes | bytearray:
        proc = self._check_usable()
        if self._fresh:
            timeout = max(timeout, self.cfg.start_timeout)
        try:
            if self.mux is not None:  # drive the completion port until this host's reply arrives
                end = time.monotonic() + timeout
                while self._q.empty():
                    left = end - time.monotonic()
                    if left <= 0:
                        raise queue.Empty
                    self.mux.poll(min(left, 0.05))
                item = self._q.get_nowait()
            else:
                item = self._q.get(timeout=timeout)
        except queue.Empty:
            self._fail(HostTimeout(f"{self.cfg.name}: no reply within {timeout:.1f}s"))
        if item is _EOF:
            try:
                rc: int | None = proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                rc = None
            self._fail(HostCrashed(f"{self.cfg.name}: stdout closed", rc, tuple(self.stderr_tail)))
        if isinstance(item, BaseException):
            self._fail(HostCrashed(f"{self.cfg.name}: broken stdout framing: {item}", proc.poll(),
                                   tuple(self.stderr_tail)))
        self._fresh = False
        assert isinstance(item, (bytes, bytearray))
        return item


def _pump_stdout(stream: IO[bytes], put: Callable[[object], None]) -> None:
    while True:
        try:
            msg = read_message(stream)
        except (ProtocolError, OSError, ValueError) as e:
            put(e)
            return
        if msg is None:
            put(_EOF)
            return
        put(msg)


def _pump_stderr(stream: IO[bytes], logger: logging.Logger, tail: "deque[str]") -> None:
    try:
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", "replace").rstrip("\r\n")
            tail.append(text)
            logger.info(text)
    except (OSError, ValueError):
        pass


def _kill_quietly(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()


# --------------------------------------------------------------------------- Windows job object

_JOB: int | None = None
_JOB_LOCK = threading.Lock()
_JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_EXTENDED_LIMIT_INFORMATION = 9


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("Basic", _BasicLimits), ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _bind_to_job(proc: subprocess.Popen[bytes]) -> None:
    """Windows: put the host into a kill-on-close job owned by this process, so hosts die with
    their owner (an actor killed by the watchdog or crashing never leaves orphan hosts)."""
    global _JOB
    if sys.platform != "win32":
        return
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    with _JOB_LOCK:
        if _JOB is None:
            k32.CreateJobObjectW.restype = ctypes.c_void_p
            job = k32.CreateJobObjectW(None, None)
            info = _ExtendedLimits()
            info.Basic.LimitFlags = _JOB_LIMIT_KILL_ON_JOB_CLOSE
            if not job or not k32.SetInformationJobObject(ctypes.c_void_p(job), _JOB_EXTENDED_LIMIT_INFORMATION,
                                                          ctypes.byref(info), ctypes.sizeof(info)):
                log.warning("kill-on-close job object unavailable (error %d)", ctypes.get_last_error())
                _JOB = 0
            else:
                _JOB = job
    if _JOB and not k32.AssignProcessToJobObject(ctypes.c_void_p(_JOB), ctypes.c_void_p(int(proc._handle))):  # type: ignore[attr-defined]
        log.warning("could not assign pid %d to the job object (error %d)", proc.pid, ctypes.get_last_error())
