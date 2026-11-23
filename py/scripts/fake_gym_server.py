"""Fake ``SpireHost --serve --search`` of the gym host for the practice-fight tests (Linux): ``spire.synth.FakeGame``
over stdio plus FORK (notes/search/protocol.md, flags GAME / DIE_WITH_PARENT of notes/gym/host.md) with a real
``os.fork``. FORK is valid only at a decision inside a fight (else ERROR 4 + the same OBS, as the real host).

Fight mode (flag GAME clear, the practice children): a child continues the parent's FakeGame on its socket until the
first decision outside the forked fight (IN_COMBAT 0 or another TOTAL_FLOOR) or the episode's end, then sends
FIGHT_END (protocol 3.4) + that frame and exits. GAME: to the episode's TERMINAL, then exits. EXACT keeps the episode
seed (frames for the same actions are byte-identical to the parent's), RESAMPLE replaces it by the fork seed.
DIE_WITH_PARENT: the child gets SIGKILL when the server dies (``prctl(PR_SET_PDEATHSIG)``).

Fault injection: ``--refuse-fork`` (every FORK is answered with ERROR 4 + the same OBS), ``--die-index I
--die-after N`` (child I of every FORK exits without a word after N steps), ``--mute-index I`` (child I of every
FORK exits before it connects), ``--hang-index I --hang-after N`` (child I stops answering after N steps).

    python -B scripts/fake_gym_server.py [--seed N] [--mean-decisions 80] [--allow-debug-options] ...
"""
from __future__ import annotations

import argparse
import ctypes
import os
import signal
import socket
import struct
import sys
import time
from pathlib import Path
from typing import IO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from spire import gym  # noqa: E402
from spire.proto import QuitCmd, StepCmd, decode_command, frame_message, parse_frame, read_message  # noqa: E402
from spire.schema import CMD_MAGIC, FRAME_MAGIC, SCHEMA_VERSION, ErrorCode, G, Result, SeatCol, Table, TerminalReason  # noqa: E402
from spire.synth import FakeGame, FakeHostConfig  # noqa: E402

_U32 = 0xFFFFFFFF
PR_SET_PDEATHSIG = 1


def encode_forked(f: object, tag: int, mode: int, abnormal: int, pids: list[int]) -> bytes:
    n = len(pids)
    words = [FRAME_MAGIC, SCHEMA_VERSION, gym.MSG_FORKED, f.vocab_hash, f.catalog_hash, f.episode_id,  # type: ignore[attr-defined]
             f.step_id, (20 + n) * 4, 0, 0, 0, 0, 0, 0, 7, 0, tag, n, mode, abnormal, *pids]  # type: ignore[attr-defined]
    return struct.pack(f"<{20 + n}I", *[x & _U32 for x in words])


def encode_fight_end(f: object, tag: int, index: int, depth: int, round_fork: int, decisions: int,
                     fork_seats: object) -> bytes:
    """FIGHT_END (MSG_TYPE 5) before frame ``f`` (protocol 3.4); seats' HP from the SEATS tables."""
    if f.is_terminal:  # type: ignore[attr-defined]
        reason = f.gvals[G.TERMINAL_REASON]  # type: ignore[attr-defined]
        outcome = (1 if f.result == Result.WIN else 4 if f.result == Result.ABORTED  # type: ignore[attr-defined]
                   else 3 if reason in (TerminalReason.DECISION_CAP, TerminalReason.COMBAT_TURN_CAP) else 2)
    else:
        outcome = 1
    seats = f.table(Table.SEATS)  # type: ignore[attr-defined]
    n = len(fork_seats)  # type: ignore[arg-type]
    words = [FRAME_MAGIC, SCHEMA_VERSION, 5, f.vocab_hash, f.catalog_hash, f.episode_id, f.step_id,  # type: ignore[attr-defined]
             (28 + 10 * n) * 4, 0, 0, 0, 0, 0, 0, 0, 0,
             tag, index, outcome, n, int(f.is_terminal), round_fork, f.gvals[G.COMBAT_ROUND] if not f.is_terminal else 0,  # type: ignore[attr-defined]
             decisions, decisions, 0, 0, depth]
    for s in range(n):
        end = seats[s] if len(seats) else fork_seats[s]  # type: ignore[index]
        fs = fork_seats[s]  # type: ignore[index]
        words += [int(end[SeatCol.HP]), int(end[SeatCol.MAX_HP]), int(end[SeatCol.HP] > 0), int(fs[SeatCol.HP]),
                  int(fs[SeatCol.MAX_HP]), 0, 0, 0, 0, 0]
    return struct.pack(f"<{len(words)}I", *[x & _U32 for x in words])


def encode_hello(pid: int, parent: int, tag: int, index: int, mode: int, depth: int, f: object, seed: int,
                 root: int, flags: int) -> bytes:
    return struct.pack("<10iII4i", gym.HELLO_MAGIC, gym.HELLO_VERSION, pid, parent, tag, index, mode, depth,
                       f.episode_id, f.step_id, seed & _U32, seed >> 32, root, flags, 0, 0)  # type: ignore[attr-defined]


class Stream:
    def __init__(self, r: IO[bytes], w: IO[bytes]) -> None:
        self.r, self.w = r, w

    def read(self) -> bytearray | None:
        return read_message(self.r)

    def write(self, payload: bytes) -> None:
        self.w.write(frame_message(payload))
        self.w.flush()


class SockIO:
    """Minimal binary file over a connected socket (readinto / write / flush)."""

    def __init__(self, s: socket.socket) -> None:
        self.s = s

    def readinto(self, view: memoryview) -> int:
        return self.s.recv_into(view)

    def write(self, data: bytes) -> int:
        self.s.sendall(data)
        return len(data)

    def flush(self) -> None:
        pass


class Server:
    def __init__(self, game: FakeGame, a: argparse.Namespace) -> None:
        self.game = game
        self.a = a
        self.depth = 0
        self.root_pid = os.getpid()
        self.abnormal = 0
        self.kids: set[int] = set()
        self.child: tuple[int, int] | None = None  # (tag, index) of a fork child
        self.since_fork = 0
        self.fight_mode = False
        self.fork_floor = -1
        self.fork_round = 0
        self.fork_seats: object = None

    def reap(self) -> None:
        for pid in list(self.kids):
            got, status = os.waitpid(pid, os.WNOHANG)
            if got:
                self.kids.discard(pid)
                if status != 0 and not (os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL):
                    self.abnormal += 1
                    print(f"[search] child {pid} exited {status}", file=sys.stderr, flush=True)

    def serve(self, io: Stream) -> None:
        a = self.a
        while True:
            msg = io.read()
            self.reap()
            if msg is None:
                return
            typ = struct.unpack_from("<2i", msg, 0)[1]
            if typ == gym.FORK_CMD:
                self.fork(io, bytes(msg))
                continue
            cmd = decode_command(msg)
            if self.child is not None and not isinstance(cmd, StepCmd):
                if isinstance(cmd, QuitCmd):
                    os._exit(0)
                for p in self.game._error(ErrorCode.BAD_MESSAGE, resend=True):  # RESET in a child
                    io.write(p)
                continue
            out = self.game.handle(cmd)
            if self.game.quit:
                return
            if self.child is not None:
                f = parse_frame(out[-1])
                if not f.is_error:
                    self.since_fork += 1
                    if self.child[1] == a.die_index and self.since_fork >= a.die_after:
                        os._exit(0)
                    if self.child[1] == a.hang_index and self.since_fork >= a.hang_after:
                        time.sleep(3600)
                if self.fight_mode and not f.is_error and (
                        f.is_terminal or f.gvals[G.IN_COMBAT] != 1 or f.gvals[G.TOTAL_FLOOR] != self.fork_floor):
                    assert len(out) == 1
                    io.write(encode_fight_end(f, self.child[0], self.child[1], self.depth, self.fork_round,
                                              self.since_fork, self.fork_seats))
                    io.write(out[0])
                    os._exit(0)
                for p in out:
                    io.write(p)
                if f.is_terminal:  # GAME: the TERMINAL frame is the child's last message
                    os._exit(0)
                continue
            for p in out:
                io.write(p)

    def fork(self, io: Stream, msg: bytes) -> None:
        w = struct.unpack_from("<8i", msg, 0)
        assert w[0] == CMD_MAGIC and w[1] == gym.FORK_CMD
        _, _, step_id, n, mode, tag, plen, flags = w
        ep = self.game.ep
        last = ep.last_obs if ep is not None else b""
        bad = (not 1 <= n <= 256 or mode not in (0, 1) or not 1 <= plen <= 107 or flags & ~gym.KNOWN_FLAGS
               or len(msg) != 32 + 8 * n + 4 * ((plen + 3) // 4) or (mode == 1 and not self.a.allow_debug_options))
        if bad or ep is None:
            for p in self.game._error(ErrorCode.BAD_MESSAGE, resend=ep is not None):
                io.write(p)
            return
        if step_id != ep.ctx.step_id:
            for p in self.game._error(ErrorCode.STALE_STEP, resend=True):
                io.write(p)
            return
        if self.a.refuse_fork or parse_frame(last).gvals[G.IN_COMBAT] != 1:
            for p in self.game._error(gym.FORK_REFUSED, resend=True):
                io.write(p)
            return
        seeds = [lo | (hi << 32) for lo, hi in (struct.unpack_from("<II", msg, 32 + 8 * i) for i in range(n))]
        path = msg[32 + 8 * n:32 + 8 * n + plen].decode()
        pids = []
        me = os.getpid()
        for i in range(n):
            pid = os.fork()
            if pid == 0:
                self.become_child(path, tag, i, mode, seeds[i], last, flags, me)
                os._exit(0)
            pids.append(pid)
            self.kids.add(pid)
        io.write(encode_forked(parse_frame(last), tag, mode, self.abnormal, pids))

    def become_child(self, path: str, tag: int, index: int, mode: int, seed: int, last: bytes, flags: int,
                     forker: int) -> None:
        if flags & gym.FLAG_DIE_WITH_PARENT:
            assert ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) == 0
            if os.getppid() != forker:
                os._exit(0)
        dn = os.open(os.devnull, os.O_RDWR)
        os.dup2(dn, 0)
        os.dup2(dn, 1)
        if index == self.a.mute_index:
            os._exit(0)
        ep = self.game.ep
        assert ep is not None
        self.depth += 1
        self.kids = set()
        self.abnormal = 0
        self.child = (tag, index)
        if mode == gym.MODE_RESAMPLE:
            ep.seed = seed & ((1 << 63) - 1)
        self.since_fork = 0
        lf = parse_frame(last)
        self.fight_mode = not flags & gym.FLAG_GAME
        self.fork_floor, self.fork_round = lf.gvals[G.TOTAL_FLOOR], lf.gvals[G.COMBAT_ROUND]
        self.fork_seats = lf.table(Table.SEATS).copy()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # type: ignore[attr-defined]
        s.connect(path)
        sio = SockIO(s)
        stream = Stream(sio, sio)  # type: ignore[arg-type]
        stream.write(encode_hello(os.getpid(), forker, tag, index, mode, self.depth, parse_frame(last),
                                  seed if mode == gym.MODE_RESAMPLE else 0, self.root_pid, flags))
        stream.write(last)
        self.serve(stream)
        os._exit(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mean-decisions", type=float, default=80.0)
    ap.add_argument("--min-decisions", type=int, default=30)
    ap.add_argument("--win-prob", type=float, default=0.5)
    ap.add_argument("--allow-debug-options", action="store_true")
    ap.add_argument("--refuse-fork", action="store_true")
    ap.add_argument("--die-index", type=int, default=-1)
    ap.add_argument("--die-after", type=int, default=3)
    ap.add_argument("--hang-index", type=int, default=-1)
    ap.add_argument("--hang-after", type=int, default=3)
    ap.add_argument("--mute-index", type=int, default=-1)
    ap.add_argument("--marker", default="", help="ignored: makes the processes of one test findable in ps")
    a = ap.parse_args()
    game = FakeGame(seed=a.seed, config=FakeHostConfig(
        mean_decisions=a.mean_decisions, min_decisions=a.min_decisions, win_prob=a.win_prob, abort_prob=0.0,
        hashes=False))
    Server(game, a).serve(Stream(sys.stdin.buffer, sys.stdout.buffer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
