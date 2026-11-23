"""Practice fights (notes/gym/practice-spec.md v1 + v1.1): extra training samples of the hard fights from RESAMPLE
fork copies of the training games, each playing one fight.

During TRAINING, on a practice actor (the last ``practice.actors`` actor slots), at the FIRST decision of every fight
of a parent game the actor draws ``k`` copies from the measured difficulty of that encounter (:class:`Difficulty`,
:class:`Controller`) and its host forks them (search wire protocol, notes/search/protocol.md: FORK mode RESAMPLE,
flags DIE_WITH_PARENT only = the host's fight mode). Every copy (a child) is one more env of the actor
(:class:`spire.actor.Actor`), plays that fight with the training behaviour and ends with FIGHT_END + one frame: a
TERMINAL (death, cap, the run's last boss) or the first OBS after the fight, whose value bootstraps the child's
truncated episode. The parent game is never touched: its frame stream does not change (protocol 2.2), the children
are sampled from their own generator in their own inference groups, and their outcomes go to their own metrics.
With ``scouts`` (v1.1) every parent is a scout: it plays combat greedily, the rest without exploration, from the
practice actor's own generator, and is never trained on (spire.actor, spire.rollout).
With ``game_classes`` (v1.2) the children of a fight of a listed class are GAME children (flags GAME |
DIE_WITH_PARENT): ordinary trained episodes from the fork to the run's TERMINAL, without FIGHT_END.

This module holds what the actor needs and nothing of the loop itself:

* :class:`PracticeConfig` - the ``practice`` section of the run config (default: off);
* :class:`Difficulty` - the per-actor table of (encounter, ascension) -> n, death rate, HP lost, shrunk to a prior;
* :class:`Controller` - the per-actor gain ``g`` that holds the children's share of the decisions at ``q_a``;
* :class:`SeedStream` - the children's seeds, SplitMix64 of an actor seed and a counter (fair-play rule: a seed is
  never derived from game state or from any other generator of the actor);
* the wire codec of FORK / FORKED / HELLO / FIGHT_END;
* :class:`ChildHost` - a fork child as a :class:`~spire.host.Host` on its socket connection;
* :class:`GymPool` - the actor's Unix-socket listener; one thread reads every child's connection and hands whole
  messages to the actor's queue as ``(env index, generation, item)``, like the stdout readers of the hosts.

Linux only (AF_UNIX sockets, the host's ``fork``).
"""
from __future__ import annotations

import math
import os
import selectors
import signal
import socket
import struct
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, NoReturn

from .host import EpisodeStats, Host, HostCrashed, HostStateError, HostTimeout, Pending
from .pipes import EOF
from .proto import Frame, parse_frame
from .schema import CMD_MAGIC, FRAME_MAGIC, MAX_FRAME_BYTES_HIDDEN, SCHEMA_VERSION, H, MsgType

# =========================================================================== configuration


@dataclass(frozen=True)
class Priors:
    """Prior difficulty ``p(class)`` of an encounter without measurements (spec 3.1): hallway, elite, boss, the
    second boss of an act (class suffix ``|2``); other room types count as Monster."""

    Monster: float = 0.10
    Elite: float = 0.35
    Boss: float = 0.55
    Boss2: float = 0.90


@dataclass(frozen=True)
class PracticeConfig:
    """``practice`` section of the run config (spec 7; v1.1 A, B). ``actors`` = how many of the run's LAST actor
    slots are practice actors (0 = off: nothing in the run changes). ``host_config``: the fork-capable launch spec
    of the dist of ``hosts.real_config`` (a ``--search`` server). Copies per fork: ``k = round_stochastic(g x
    max(0, s - s_min) / s_unit)`` capped at ``k_max``; ``s`` = (n x (d + c_hp x h) + prior_n x p) / (n + prior_n)
    over the last ``window`` parent fights of the encounter. Caps: ``per_parent`` children alive per parent host,
    ``max_children`` per actor (0 = ceil(1.6 x envs_per_actor x q_a)). ``scouts`` (v1.1): the parents play quietly
    and are not trained on; ``q_a`` is then the children's share of a practice actor's decisions directly (without
    scouts it is ``share x actor.actors / actors``). ``paired_seeds``: children 2j and 2j+1 of a FORK share a seed.
    The listener: ``<socket_dir>/g<pid>-<env id base>.sock`` with ``backlog`` pending connections; FORKED and every
    child's HELLO + first frame must arrive within ``accept_timeout`` seconds."""

    actors: int = 0
    host_config: str = ""
    share: float = 0.25
    ascensions: tuple[int, ...] = ()  # () = all
    k_max: int = 8
    per_parent: int = 12
    max_children: int = 0
    c_hp: float = 0.5
    prior_n: float = 20.0
    window: int = 100
    s_min: float = 0.05
    s_unit: float = 0.10
    priors: Priors = field(default_factory=Priors)
    g_min: float = 0.25
    g_max: float = 8.0
    socket_dir: str = "/tmp/spire-gym"
    backlog: int = 64
    accept_timeout: float = 30.0
    scouts: bool = False
    q_a: float = 0.5
    paired_seeds: bool = False
    game_classes: tuple[str, ...] = ()  # v1.2: the children of a fight of these classes play to the run's end
    rloo_alpha: float = 0.0  # p12: sibling leave-one-out door-baseline blend in [0, 1] (0 = off)
    rloo_by_pair: bool = False  # p12: True = group by (fork.tag, kid.pair) with paired_seeds; False = all k siblings

    def __post_init__(self) -> None:
        assert self.actors >= 0
        # k_max = 0: the cost run of spec 9 (practice actors on the fork-capable spec, no FORK ever sent)
        assert 0 <= self.k_max <= MAX_FORK and self.per_parent >= 1 and self.max_children >= 0
        assert self.prior_n > 0 and self.window >= 1 and self.s_unit > 0 and 0 < self.g_min <= 1.0 <= self.g_max
        assert self.backlog >= 1 and self.accept_timeout > 0
        assert math.isfinite(self.rloo_alpha) and 0.0 <= self.rloo_alpha <= 1.0, \
            f"practice.rloo_alpha {self.rloo_alpha}: 0 <= alpha <= 1"
        assert not self.rloo_by_pair or self.paired_seeds, "practice.rloo_by_pair requires practice.paired_seeds"
        for cls in self.game_classes:
            parts = cls.split("|")
            assert len(parts) in (2, 3) and parts[0].isdigit() and int(parts[0]) >= 1 and parts[1] \
                and (len(parts) == 2 or parts[2] == "2"), f"practice.game_classes: {cls!r} is not a class"
        assert len(set(self.game_classes)) == len(self.game_classes), self.game_classes

    @property
    def enabled(self) -> bool:
        return self.actors > 0

    def fork_flags(self, cls: str) -> int:
        """FORK word 7 of a fork at a fight of class ``cls`` (v1.2 A): GAME children for a listed class."""
        return GAME_FLAGS if cls in self.game_classes else PRACTICE_FLAGS

    def q_target(self, n_actors: int) -> float:
        """The children's share of a practice actor's decisions (spec 3.3; v1.1 A.7: given directly with scouts)."""
        assert 1 <= self.actors <= n_actors, (self.actors, n_actors)
        q = self.q_a if self.scouts else self.share * n_actors / self.actors
        assert 0.2 <= q <= 0.6, f"practice: q_a = {q} outside [0.2, 0.6]"
        return q

    def parent_envs(self, envs_per_actor: int, n_actors: int) -> int:
        return round(envs_per_actor * (1 - self.q_target(n_actors)))

    def children_cap(self, envs_per_actor: int, n_actors: int) -> int:
        return self.max_children or math.ceil(1.6 * envs_per_actor * self.q_target(n_actors))


# =========================================================================== difficulty, copies, gain


def encounter_class(act: int, room: str, second_boss: bool) -> str:
    """``"<act>|<room>"`` (+ ``"|2"`` for the second boss fight of an act); ``act`` counts from 1."""
    return f"{act}|{room}" + ("|2" if second_boss else "")


def key_class(key: str) -> str:
    """The class of an encounter key ``"<act>|<room>|<enemies>"`` (+ ``"|2"``)."""
    parts = key.split("|")
    assert len(parts) in (3, 4) and (len(parts) == 3 or parts[3] == "2"), key
    return encounter_class(int(parts[0]), parts[1], len(parts) == 4)


def prior_of(cls: str, priors: Priors) -> float:
    parts = cls.split("|")
    room = parts[1]
    if room == "Boss":
        return priors.Boss2 if len(parts) == 3 else priors.Boss
    if room == "Elite":
        return priors.Elite
    return priors.Monster


class Difficulty:
    """Per (encounter key, ascension): ``[n, d, h]`` = fights seen, death rate, mean HP-lost fraction (spec 3.1);
    the arithmetic mean of the first ``window`` fights, then an exponential mean of about the last ``window``."""

    def __init__(self, cfg: PracticeConfig) -> None:
        self.cfg = cfg
        self.table: dict[tuple[str, int], list[float]] = {}
        self.cls: dict[tuple[str, int], str] = {}

    def update(self, key: str, cls: str, asc: int, death: bool, hp_lost: float) -> None:
        assert 0.0 <= hp_lost <= 1.0, hp_lost
        e = self.table.setdefault((key, asc), [0.0, 0.0, 0.0])
        self.cls[key, asc] = cls
        e[0] += 1
        w = 1.0 / min(e[0], self.cfg.window)
        e[1] += w * (float(death) - e[1])
        e[2] += w * (hp_lost - e[2])

    def s(self, key: str, cls: str, asc: int) -> float:
        cfg = self.cfg
        p = prior_of(cls, cfg.priors)
        e = self.table.get((key, asc))
        if e is None:
            return p
        n, d, h = e
        return (n * (d + cfg.c_hp * h) + cfg.prior_n * p) / (n + cfg.prior_n)

    def x(self, s: float, g: float) -> float:
        """Expected copies of a fight of difficulty ``s`` at gain ``g`` (before the cap)."""
        return g * max(0.0, s - self.cfg.s_min) / self.cfg.s_unit

    def copies(self, x: float, u: float) -> int:
        """``min(k_max, floor(x) + Bernoulli(x - floor(x)))`` with the uniform ``u`` of the gym's k generator."""
        f = math.floor(x)
        return min(self.cfg.k_max, f + (u < x - f))

    def dump(self) -> list[dict[str, object]]:
        return [{"key": k, "ascension": a, "class": self.cls[k, a], "n": e[0], "d": e[1], "h": e[2],
                 "s": self.s(k, self.cls[k, a], a)} for (k, a), e in sorted(self.table.items())]


CONTROL_PERIOD_S: Final = 60.0
CONTROL_BUCKETS: Final = 10  # the share is measured over the last 600 s
CONTROL_START_BUCKETS: Final = 5  # g moves once 300 s exist


class Controller:
    """The gain ``g`` of one practice actor (spec 3.3): every 60 s, once 300 s of counts exist, ``g <- clip(g x
    clip(q_a / max(q, 0.02), 0.8, 1.25), g_min, g_max)`` with ``q`` = child decisions / all decisions of the actor
    over the last 600 s. ``tick`` takes the actor's cumulative counts."""

    def __init__(self, q_a: float, g_min: float, g_max: float, t0: float) -> None:
        self.q_a, self.g_min, self.g_max = q_a, g_min, g_max
        self.g = 1.0
        self.q: float | None = None
        self.t_next = t0 + CONTROL_PERIOD_S
        self.marks: deque[tuple[int, int]] = deque([(0, 0)], maxlen=CONTROL_BUCKETS + 1)

    def tick(self, now: float, child: int, total: int) -> None:
        while now >= self.t_next:
            self.t_next += CONTROL_PERIOD_S
            self.marks.append((child, total))
            if len(self.marks) > CONTROL_START_BUCKETS:
                c0, a0 = self.marks[0]
                q = (child - c0) / (total - a0) if total > a0 else 0.0
                self.q = q
                step = min(1.25, max(0.8, self.q_a / max(q, 0.02)))
                self.g = min(self.g_max, max(self.g_min, self.g * step))


# =========================================================================== seeds

_M64: Final = (1 << 64) - 1
_GOLDEN: Final = 0x9E3779B97F4A7C15


def splitmix64(x: int) -> int:
    z = x & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def fork_group_id(gym_seed: int, tag: int, pair: int = 0) -> int:
    """Deterministic positive signed-int63 group id shared by sibling practice children of one FORK
    (or one seed pair when ``practice.rloo_by_pair`` is True). Never 0."""
    h = splitmix64((gym_seed & _M64) ^ ((splitmix64(tag + 1) * _GOLDEN) & _M64) ^ splitmix64(pair + 0x517CC1B727220A95))
    fg = (h & ((1 << 63) - 1)) | 1
    assert fg > 0
    return fg


class SeedStream:
    """SplitMix64: the n-th seed is ``splitmix64(seed + n * golden)``, a function of the actor's gym seed and a
    private counter only (protocol 2.1: never of a game seed or of anything a host sent)."""

    def __init__(self, seed: int) -> None:
        self.seed = seed & _M64
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return splitmix64(self.seed + self.n * _GOLDEN)

    def fork_seeds(self, k: int, paired: bool) -> tuple[int, ...]:
        """The seeds of one FORK of ``k`` children; ``paired``: children 2j and 2j+1 share one (v1.1 B)."""
        if not paired:
            return tuple(self.next() for _ in range(k))
        out: list[int] = []
        while len(out) < k:
            s = self.next()
            out += [s, s][:k - len(out)]
        return tuple(out)


# =========================================================================== wire (notes/search/protocol.md)

FORK_CMD: Final = 4  # command type (RESET 1, STEP 2, QUIT 3)
MSG_FORKED: Final = 4  # frame header MSG_TYPE of FORKED
MSG_FIGHT_END: Final = 5  # frame header MSG_TYPE of FIGHT_END (protocol 3.4)
HELLO_MAGIC: Final = 0x4B465053  # "SPFK"
HELLO_VERSION: Final = 1
HELLO_BYTES: Final = 64
MODE_RESAMPLE: Final = 0
MODE_EXACT: Final = 1  # debug servers only; never in training (protocol 5)
MAX_FORK: Final = 256
FORK_REFUSED: Final = 4  # ERROR code: not at a decision inside a fight
# FORK word 7 (notes/gym/host.md): GAME = the child plays to the end of the game; clear = fight mode (FIGHT_END +
# one frame); DIE_WITH_PARENT = SIGKILL when the process that forked it dies
FLAG_GAME: Final = 1
FLAG_DIE_WITH_PARENT: Final = 2
KNOWN_FLAGS: Final = FLAG_GAME | FLAG_DIE_WITH_PARENT
PRACTICE_FLAGS: Final = FLAG_DIE_WITH_PARENT  # fight mode, dies with its host
GAME_FLAGS: Final = FLAG_GAME | FLAG_DIE_WITH_PARENT  # v1.2: plays to the run's TERMINAL, dies with its host
# FIGHT_END word 18
FIGHT_WON, FIGHT_WIPED, FIGHT_TRUNCATED, FIGHT_ABORTED = 1, 2, 3, 4

_LEN = struct.Struct("<I")
_U32: Final = 0xFFFFFFFF
_HEAD3 = struct.Struct("<3i")
_MSG_TERMINAL = int(MsgType.TERMINAL)


def msg_type(payload: bytes | bytearray | memoryview) -> int:
    """MSG_TYPE of a host message (word 2) after checking the frame magic."""
    magic, _, mt = _HEAD3.unpack_from(payload, 0)
    assert magic & _U32 == FRAME_MAGIC, f"bad frame magic 0x{magic & _U32:08X}"
    return mt  # type: ignore[no-any-return]


def encode_fork(step_id: int, seeds: tuple[int, ...], mode: int, tag: int, path: str, flags: int) -> bytes:
    """Complete wire message (length prefix included) of FORK (protocol 2.1, word 7 = ``flags``)."""
    n = len(seeds)
    assert 1 <= n <= MAX_FORK, n
    assert mode in (MODE_RESAMPLE, MODE_EXACT), mode
    assert flags & ~KNOWN_FLAGS == 0, flags
    p = path.encode("utf-8")
    assert 1 <= len(p) <= 107, path
    assert all(0 <= s <= _M64 for s in seeds)
    body = struct.pack("<8i", CMD_MAGIC, FORK_CMD, step_id, n, mode, tag, len(p), flags)
    body += b"".join(struct.pack("<II", s & _U32, s >> 32) for s in seeds)
    body += p + b"\0" * ((-len(p)) % 4)
    assert len(body) == 32 + 8 * n + 4 * ((len(p) + 3) // 4)
    return _LEN.pack(len(body)) + body


@dataclass(frozen=True)
class Forked:
    episode_id: int
    step_id: int
    host_us: int
    tag: int
    n: int
    mode: int
    abnormal_exits: int
    pids: tuple[int, ...]


def _header(payload: bytes | bytearray | memoryview, mt: int) -> tuple[int, ...]:
    h = struct.unpack_from("<16i", payload, 0)
    assert h[H.MAGIC] & _U32 == FRAME_MAGIC and h[H.SCHEMA_VERSION] == SCHEMA_VERSION, h[:2]
    assert h[H.MSG_TYPE] == mt and h[H.FRAME_BYTES] == len(payload), (h[H.MSG_TYPE], h[H.FRAME_BYTES], len(payload))
    return h


def parse_forked(payload: bytes | bytearray | memoryview) -> Forked:
    h = _header(payload, MSG_FORKED)
    n = struct.unpack_from("<i", payload, 17 * 4)[0]
    assert len(payload) == (20 + n) * 4, (len(payload), n)
    w = struct.unpack_from(f"<{4 + n}i", payload, 16 * 4)
    return Forked(h[H.EPISODE_ID], h[H.STEP_ID], h[H.HOST_US], w[0], w[1], w[2], w[3], tuple(w[4:]))


@dataclass(frozen=True)
class SeatEnd:
    """FIGHT_END per-seat block (protocol 3.4, words 28 + 10 s ..)."""

    hp_end: int
    max_hp_end: int
    alive_end: int
    hp_fork: int
    max_hp_fork: int
    potions_fork: int
    potions_end: int
    potions_used: int
    gold_fork: int
    gold_end: int


@dataclass(frozen=True)
class FightEnd:
    episode_id: int
    step_id: int  # the STEP_ID of the frame that follows
    tag: int
    index: int
    outcome: int  # FIGHT_WON / FIGHT_WIPED / FIGHT_TRUNCATED / FIGHT_ABORTED
    n_seats: int
    terminal: int  # 1: a TERMINAL frame follows, 0: an OBS
    round_fork: int
    round_end: int
    decisions: int
    points: int
    enemies_alive: int
    enemy_hp: int
    depth: int
    seats: tuple[SeatEnd, ...]


def parse_fight_end(payload: bytes | bytearray | memoryview) -> FightEnd:
    h = _header(payload, MSG_FIGHT_END)
    w = struct.unpack_from("<12i", payload, 16 * 4)
    n = w[3]
    assert 1 <= n <= 4 and len(payload) == (28 + 10 * n) * 4, (n, len(payload))
    assert w[2] in (FIGHT_WON, FIGHT_WIPED, FIGHT_TRUNCATED, FIGHT_ABORTED) and w[4] in (0, 1), w
    s = struct.unpack_from(f"<{10 * n}i", payload, 28 * 4)
    seats = tuple(SeatEnd(*s[10 * i:10 * i + 10]) for i in range(n))
    return FightEnd(h[H.EPISODE_ID], h[H.STEP_ID], *w, seats)


@dataclass(frozen=True)
class Hello:
    pid: int
    parent_pid: int
    tag: int
    index: int
    mode: int
    depth: int
    episode_id: int
    step_id: int
    seed: int
    root_pid: int
    flags: int


def parse_hello(payload: bytes | bytearray | memoryview) -> Hello:
    assert len(payload) == HELLO_BYTES, len(payload)
    w = struct.unpack_from("<10iII4i", payload, 0)
    assert w[0] & _U32 == HELLO_MAGIC, f"bad HELLO magic 0x{w[0] & _U32:08X}"
    assert w[1] == HELLO_VERSION, w[1]
    return Hello(w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10] | (w[11] << 32), w[12], w[13])


def same_obs(a: bytes | bytearray | memoryview, b: bytes | bytearray | memoryview) -> bool:
    """Two OBS payloads equal except HOST_US (protocol 3.1 step 3). With the HDRAW trailer (notes/critic/contract.md
    1.4: a RESAMPLE child re-draws its pile): bytes [0, XH) equal except HOST_US and FRAME_BYTES, and the two HDRAW
    tables hold the same multiset of (SEAT, OWNER_ROW)."""
    o = H.HOST_US * 4
    fa, fb = parse_frame(a), parse_frame(b)
    assert (fa.hd_entry < 0) == (fb.hd_entry < 0), "same_obs: one frame has the HDRAW trailer, the other not"
    if fa.hd_entry < 0:
        return len(a) == len(b) and bytes(a[:o]) == bytes(b[:o]) and bytes(a[o + 4:]) == bytes(b[o + 4:])
    xh, fbw = fa.hd_entry * 4, H.FRAME_BYTES * 4
    if fb.hd_entry * 4 != xh:
        return False
    pa, pb = bytes(a[:xh]), bytes(b[:xh])
    if pa[:fbw] != pb[:fbw] or pa[fbw + 4 : o] != pb[fbw + 4 : o] or pa[o + 4:] != pb[o + 4:]:
        return False
    ha, hb = fa.hdraw, fb.hdraw
    assert ha is not None and hb is not None
    return sorted(map(tuple, ha[:, [0, 2]].tolist())) == sorted(map(tuple, hb[:, [0, 2]].tolist()))


# =========================================================================== children


@dataclass(frozen=True)
class Arrival:
    """Pool -> actor: a child connected and said HELLO (its first frame follows as the next item)."""

    conn: "_Conn"
    hello: Hello


class _Conn:
    """One accepted connection; owned by the pool's thread, which alone closes the socket."""

    __slots__ = ("sock", "buf", "idx", "gen", "ended", "closed", "fight_end", "final", "game")

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buf = bytearray()
        self.idx = -1  # the actor's env index once HELLO named the child
        self.gen = 0
        self.ended = False  # nothing more is forwarded (the final frame delivered, or dropped by the actor)
        self.closed = False
        self.fight_end = False  # FIGHT_END forwarded: exactly one frame follows
        self.final = False  # the final frame forwarded (after FIGHT_END; a GAME child's TERMINAL): nothing more
        self.game = False  # a GAME child (v1.2): its TERMINAL comes without FIGHT_END


def child_alive(pid: int, forker: int) -> bool:
    """The process ``pid`` exists, is not a zombie and is still the child of ``forker`` (the process that forked
    it: never another process that got the pid later)."""
    path = f"/proc/{pid}/stat"
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        s = fh.read().decode()
    state, ppid = s[s.rindex(")") + 2:].split()[:2]
    return state != "Z" and int(ppid) == forker


class ChildHost(Host):
    """A fork child on its socket connection (protocol 3): one slot of the actor, re-used for one child after
    another (:meth:`begin` ... :meth:`close`). Routed use only: the pool's thread delivers the frames. The slot
    counts in :attr:`generation`, so items of an earlier child are stale."""

    reader_threads = True

    def __init__(self, pool: "GymPool", *, step_timeout: float, accept_timeout: float,
                 expected_hashes: tuple[int, int] | None) -> None:
        super().__init__(reset_timeout=accept_timeout, step_timeout=step_timeout, expected_hashes=expected_hashes)
        self.pool = pool
        self.conn: _Conn | None = None
        self.pid = 0  # the child process (FORKED / HELLO), 0 until one of them named it
        self.forker = 0  # the process that forked it (the parent game's host)

    def begin(self, seed: int | None, characters: tuple[int, ...], ascension: int, episode_id: int) -> None:
        """A new child is expected on this slot: the wrapper is inside the parent's episode, no frame yet."""
        assert self.conn is None and self.episode is None, "the slot still holds a child"
        self.generation += 1
        self.episode = EpisodeStats(seed, characters, ascension, episode_id)
        self.last_frame = None
        self.pid = self.forker = 0
        self.stats.episodes += 1

    def first(self, payload: bytes) -> Frame:
        """The child's first frame, the OBS of the fork decision (the caller compared it with the parent's)."""
        f = self._parse(payload)
        ep = self.episode
        assert ep is not None and self.conn is not None and f.is_obs and f.episode_id == ep.episode_id, f
        self.last_frame = f
        return f

    def _send(self, message: bytes) -> None:
        conn = self.conn
        assert conn is not None, "STEP for a child that is not there"
        try:
            conn.sock.sendall(message)
        except OSError as e:
            self.generation += 1
            raise HostCrashed(f"child {self.pid}: write failed: {e}") from e

    def _recv(self, timeout: float) -> NoReturn:
        raise HostStateError("a fork child is routed: its frames come from the pool")

    def _timeout_failure(self) -> NoReturn:
        """No HELLO / first frame within the accept timeout, or no reply within the step timeout: the child is
        killed if it is still there (it never times out by itself, protocol 3.3; its forking host reaps it)."""
        self.generation += 1
        if self.pid and child_alive(self.pid, self.forker):
            os.kill(self.pid, signal.SIGKILL)
        if self.conn is None:
            raise HostTimeout(f"fork child {self.pid or '?'} did not arrive within the accept timeout")
        raise HostTimeout(f"child {self.pid}: no reply within {self.step_timeout:.1f}s (killed)")

    def close(self) -> None:
        """Leave the child: EOF at a message boundary makes it exit (protocol 3.3). The slot is free again."""
        if self.conn is not None:
            self.pool.drop(self.conn)
            self.conn = None
        self.episode = None
        self.last_frame = None
        self.pending = Pending.NONE  # a STEP may be outstanding (the actor stopped, a timeout)


@dataclass
class PoolStats:
    accepted: int = 0
    unexpected: int = 0  # connections whose HELLO named no expected child (closed)
    messages: int = 0


class GymPool:
    """One listening AF_UNIX socket and one thread for all children of an actor.

    The actor thread: :meth:`expect` a child before it sends the FORK that creates it, :meth:`forget` one it stops
    waiting for, :meth:`drop` a connection it is done with. The pool thread: accepts, reads HELLO, binds the
    connection to the expected env and from then on hands every complete message to ``put`` as ``(env index,
    generation, item)``: :class:`Arrival`, frame payloads (FIGHT_END included), or ``pipes.EOF`` for a child that
    closed before the frame that follows its FIGHT_END. An exception of the pool thread is handed over as ``(-1, 0,
    exception)`` (the actor raises it).
    """

    def __init__(self, path: str, put: Callable[[tuple[int, int, object]], None], *, backlog: int) -> None:
        assert len(path.encode("utf-8")) <= 107, f"socket path too long: {path}"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(path):
            os.unlink(path)
        self.path = path
        self.put = put
        self.stats = PoolStats()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # type: ignore[attr-defined]
        self.listener.bind(path)
        self.listener.listen(backlog)
        # (tag, index) -> (env index, generation, GAME child)
        self._expect: dict[tuple[int, int], tuple[int, int, bool]] = {}
        self._lock = threading.Lock()  # drop() vs the pool thread closing a socket
        self._wake_r, self._wake_w = os.pipe()
        self._stop = False
        self._sel = selectors.DefaultSelector()
        self._sel.register(self.listener, selectors.EVENT_READ, None)
        self._sel.register(self._wake_r, selectors.EVENT_READ, None)
        self._thread = threading.Thread(target=self._run, daemon=True, name="gym-pool")
        self._thread.start()

    # ------------------------------------------------------------------ actor thread
    def expect(self, tag: int, index: int, idx: int, gen: int, game: bool) -> None:
        self._expect[tag, index] = (idx, gen, game)

    def forget(self, tag: int, index: int) -> None:
        self._expect.pop((tag, index), None)  # absent: the child connected meanwhile (its Arrival is stale)

    def drop(self, conn: _Conn) -> None:
        with self._lock:
            conn.ended = True
            if not conn.closed:
                conn.sock.shutdown(socket.SHUT_RDWR)  # the child reads EOF and exits; the pool thread closes

    def close(self) -> None:
        """Stop the thread; every connection and the listener are closed, the socket file removed."""
        self._stop = True
        os.write(self._wake_w, b"x")
        self._thread.join()
        os.close(self._wake_r)
        os.close(self._wake_w)
        os.unlink(self.path)

    # ------------------------------------------------------------------ pool thread
    def _run(self) -> None:
        try:
            self._serve()
        except BaseException as e:
            self.put((-1, 0, e))
            raise

    def _serve(self) -> None:
        sel = self._sel
        listener, wake = self.listener, self._wake_r
        while True:
            for key, _ in sel.select():
                if key.fileobj is listener:
                    sock, _addr = listener.accept()
                    self.stats.accepted += 1
                    sel.register(sock, selectors.EVENT_READ, _Conn(sock))
                elif key.fileobj == wake:
                    assert self._stop
                    for k in list(sel.get_map().values()):
                        if k.data is not None:
                            self._close(k.data)
                    sel.close()
                    listener.close()
                    return
                else:
                    self._read(key.data)

    def _close(self, conn: _Conn) -> None:
        with self._lock:
            self._sel.unregister(conn.sock)
            conn.sock.close()
            conn.closed = True

    def _read(self, conn: _Conn) -> None:
        try:
            data = conn.sock.recv(1 << 18)
        except ConnectionResetError:  # the child was killed with unread data on its side
            data = b""
        if not data:
            if conn.idx >= 0 and not conn.ended:
                self.put((conn.idx, conn.gen, EOF))
            self._close(conn)
            return
        assert not conn.final, f"child of env {conn.idx}: a message after its final frame"
        if conn.ended:
            return
        buf = conn.buf
        buf += data
        pos = 0
        while len(buf) - pos >= 4:
            (n,) = _LEN.unpack_from(buf, pos)
            assert n <= MAX_FRAME_BYTES_HIDDEN, f"child message of {n} bytes"
            if len(buf) - pos - 4 < n:
                break
            self._message(conn, bytes(buf[pos + 4:pos + 4 + n]))
            pos += 4 + n
            if conn.final:
                assert pos == len(buf), f"child of env {conn.idx}: a message after its final frame"
            if conn.closed or conn.ended:
                return
        del buf[:pos]

    def _message(self, conn: _Conn, m: bytes) -> None:
        self.stats.messages += 1
        if conn.idx < 0:
            hello = parse_hello(m)
            slot = self._expect.pop((hello.tag, hello.index), None)
            if slot is None:  # its fork was given up meanwhile: EOF tells the child to exit
                self.stats.unexpected += 1
                self._close(conn)
                return
            conn.idx, conn.gen, conn.game = slot
            self.put((conn.idx, conn.gen, Arrival(conn, hello)))
            return
        mt = msg_type(m)
        if conn.game:  # v1.2 B: a GAME child ends with its TERMINAL (then exits; its EOF is not an event)
            assert mt != MSG_FIGHT_END, "a GAME child sent FIGHT_END"
            if mt == _MSG_TERMINAL:
                conn.final = conn.ended = True
        elif conn.fight_end:  # the one frame after FIGHT_END: the child exits after it, its EOF is not an event
            assert mt != MSG_FIGHT_END, "a second FIGHT_END"
            conn.final = conn.ended = True
        elif mt == MSG_FIGHT_END:
            conn.fight_end = True
        else:
            assert mt != _MSG_TERMINAL, "a fight-mode child sent TERMINAL without FIGHT_END"
        self.put((conn.idx, conn.gen, m))


# =========================================================================== actor-side counters

GAME_CLASS_FIELDS: Final = ("game_children", "game_finished", "game_wins", "game_deaths_first", "game_deaths_later",
                            "game_decisions")  # v1.2 D
CLASS_FIELDS: Final = ("forks", "children", "finished", "fight_wins", "deaths", "decisions", "hp_lost",
                       "parent_fights", "parent_deaths", "parent_hp_lost", "forked_w", "forked_deaths_w",
                       "forked_hp_lost_w", *GAME_CLASS_FIELDS)
# v1.2 D: the table practice.classes_asc, keys "<class>|A9" / "<class>|A10" (fight wins and deaths: of the forked fight)
ASC_FIELDS: Final = ("parent_fights", "parent_deaths", "children", "fight_wins", "game_wins", "deaths")
TOP_ASCENSIONS: Final = (9, 10)


@dataclass
class PracticeCounters:
    """Since the previous actor report."""

    forks: int = 0  # FORK commands answered with FORKED
    children: int = 0  # children of those FORKs
    fork_failures: int = 0  # FORK refused by the host (ERROR), or its parent's episode ended before FORKED
    capped: int = 0  # fights whose copies were trimmed to 0 by the caps
    trimmed: int = 0  # fights whose copies were trimmed to fewer
    started: int = 0  # children that arrived with their first frame
    lost: int = 0  # children lost (EOF without FIGHT_END + frame, timeout, never connected, error)
    decisions: int = 0  # policy decisions of children
    scout_decisions: int = 0  # decisions of scouts (v1.1)
    fork_ns: list[int] = field(default_factory=list)  # FORK sent -> FORKED (the parent's frame is held that long)
    arrive_ns: list[int] = field(default_factory=list)  # FORK sent -> a child's first frame is ready
    host_us: int = 0  # FORKED HOST_US: time inside the host's fork() calls
    classes: dict[str, dict[str, float]] = field(default_factory=dict)  # per class: CLASS_FIELDS
    keys: dict[tuple[str, int], list[float]] = field(default_factory=dict)  # children, fight wins, deaths, HP lost
    classes_asc: dict[str, dict[str, int]] = field(default_factory=dict)  # "<class>|A<asc>": ASC_FIELDS (v1.2 D)

    def cls(self, name: str) -> dict[str, float]:
        c = self.classes.get(name)
        if c is None:
            c = self.classes[name] = dict.fromkeys(CLASS_FIELDS, 0)
        return c

    def asc(self, name: str, asc: int) -> dict[str, int]:
        """The ``classes_asc`` row of class ``name`` at ascension ``asc`` (one of TOP_ASCENSIONS)."""
        assert asc in TOP_ASCENSIONS, asc
        key = f"{name}|A{asc}"
        c = self.classes_asc.get(key)
        if c is None:
            c = self.classes_asc[key] = dict.fromkeys(ASC_FIELDS, 0)
        return c
