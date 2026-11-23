"""Per-step rewards and auxiliary targets derived from consecutive public frames of one episode.

Only public frame fields are read (schema v1 + errata-v1):

=====================  =====================================================================
field                  use
=====================  =====================================================================
HEADER MSG_TYPE        OBS (decision frame) vs TERMINAL (final observation)
HEADER RESULT          TERMINAL outcome: WIN / LOSS / ABORTED
HEADER FLAGS bit1      overflow -> the whole episode is dropped
GLOBAL KIND            decision kind (stored per step, not used by the reward)
GLOBAL TOTAL_FLOOR     floor progress; combat boundaries
GLOBAL ROOM_TYPE       vocab ``enum:RoomType`` of the current room (kill inference, diagnostics)
GLOBAL IN_COMBAT       combat membership of a decision
GLOBAL ELITES_WON,     the host's counters of won Elite / Boss combats (errata-v1 section 5):
       BOSSES_WON      kill bonuses
GLOBAL TERMINAL_REASON TERMINAL only: death vs decision cap / combat-turn cap / other truncation
SEATS hp, max_hp       team HP fraction (potential term, post-combat HP target)
HEADER N_ACTIONS       legal action count (stored per step for entropy normalisation)
=====================  =====================================================================

Reward of the transition ``frame_t --action_t--> frame_{t+1}`` (``frame_{t+1}`` may be the
TERMINAL frame), a single **team** scalar shared by all seats::

    r_t = floor_t + elite_t + boss_t + terminal_t + hp_t

* ``floor``    ``+0.01 * max(0, TOTAL_FLOOR_{t+1} - TOTAL_FLOOR_t)``.
* ``elite`` / ``boss``  ``+0.02`` / ``+0.1`` per Elite / Boss combat won at this transition, read
  from the host's explicit counters (errata-v1 PY-5): ``0.02 * dG[ELITES_WON] + 0.1 *
  dG[BOSSES_WON]`` with ``d`` = frame ``t+1`` minus frame ``t``. TERMINAL frames carry the
  counters too, so the final boss kill is credited on the transition into TERMINAL(WIN), and each
  boss of the A10 double boss counts. A fight that ends without a win (escape, death) never
  increments a counter, and event-started fights are Monster-room fights (errata E-22): no bonus.
  ``RewardConfig(kills="rooms")`` restores the older inference (:func:`inferred_kills`, kept for
  cross-checks): a combat is a maximal run of consecutive decision frames with ``IN_COMBAT = 1``
  and the same ``TOTAL_FLOOR``; one that ends in an Elite / Boss room with the team alive counts.
* ``terminal`` ``+1`` for RESULT=WIN, ``-0.1`` for a death (RESULT=LOSS with TERMINAL_REASON DEATH,
  or NONE from hosts predating errata-v1), or a DECISION_CAP / COMBAT_TURN_CAP failure, on the last
  transition. ``RewardConfig.caps_as_failure=False`` restores the caps' truncation handling.
  RESULT=ABORTED (host exception)
  is **not** a death: no terminal reward, and the episode is dropped by the rollout layer
  (:attr:`Outcome.dropped`). Same for an overflow frame.
* **Truncations** (RESULT=LOSS with TERMINAL_REASON OTHER_TRUNCATION, or caps when
  ``caps_as_failure=False``) count as failures in every outcome
  statistic, but the game did not reach a terminal state: :attr:`Outcome.TRUNCATED`. The TERMINAL
  frame is then the next *state*, not a terminal one: the last transition gets its floor / kill /
  shaping terms with ``Phi`` of that frame's real HP and no terminal reward; the value target
  bootstraps (rollout layer); ``win`` / ``floors_left`` are masked (the outcome is censored); a
  combat still running at the cut (combat-turn cap) gets no ``hp_post``.
* ``hp``       potential-based shaping (Ng, Harada & Russell 1999)
  ``gamma * Phi(s_{t+1}) - Phi(s_t)`` with ``Phi(s) = hp_coef * team_hp(s)`` on decision
  frames and ``Phi(TERMINAL) = 0``; ``team_hp`` = mean over seats of
  ``clip(hp, 0) / max_hp`` (dead seats count 0). ``gamma`` must be the per-decision discount
  of the advantage estimator; then the discounted sum of the shaping terms over an episode
  telescopes to ``-Phi(s_0)``, a constant, so it cannot change which policy is optimal.

Auxiliary targets (model heads ``pwin_logit`` / ``floors_pred`` / ``hp_pred``):

* ``win``          1.0 if the episode ended with RESULT=WIN else 0.0 (WIN / death / cap failures);
* ``floors_left``  ``TOTAL_FLOOR(TERMINAL) - TOTAL_FLOOR_t`` (WIN / death / cap failures);
* ``hp_post``      for decisions inside a combat: ``team_hp`` of the first frame after that
  combat. A death's TERMINAL carries HP 0. The real host also encodes HP 0 on WIN, so the won
  final fight uses the last decision's HP instead. A cap does not resolve a fight still in progress:
  its HP target stays masked. Targets are also masked outside combats and unfinished combats.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from .proto import Frame
from .schema import (
    GLOBAL_OFFSET,
    MAX_ROWS,
    N_COLS,
    TRUNCATION_REASONS,
    VOCAB_UNK,
    FrameFlag,
    G,
    MsgType,
    Result,
    SeatCol,
    Table,
    TerminalReason,
)
from .vocab import Vocab

F32 = NDArray[np.float32]
I32 = NDArray[np.int32]


class Facts(NamedTuple):
    """The public scalars of one OBS / TERMINAL frame that rewards and targets depend on."""

    msg_type: int
    result: int
    kind: int
    total_floor: int
    room_type: int
    in_combat: bool
    hp_frac: float
    n_actions: int
    overflow: bool
    elites_won: int = 0  # GLOBAL[ELITES_WON]
    bosses_won: int = 0  # GLOBAL[BOSSES_WON]
    terminal_reason: int = 0  # GLOBAL[TERMINAL_REASON] of a TERMINAL frame (0 for OBS frames)

    @property
    def terminal(self) -> bool:
        return self.msg_type == MsgType.TERMINAL

    @property
    def truncated(self) -> bool:
        """TERMINAL of an episode the host stopped (decision / combat-turn cap ...), not a death."""
        return self.msg_type == _TERMINAL and self.result == _LOSS and self.terminal_reason in _TRUNC


_TRUNC = frozenset(int(r) for r in TRUNCATION_REASONS)
_CAPS = frozenset((int(TerminalReason.DECISION_CAP), int(TerminalReason.COMBAT_TURN_CAP)))
_LOSS = int(Result.LOSS)
(_MSG, _RES, _KIND, _FLOOR, _ROOM, _IC, _HP, _NACT, _OVF, _ELITES, _BOSSES, _REASON) = range(len(Facts._fields))
_HP_COLS = [int(SeatCol.HP), int(SeatCol.MAX_HP)]


def team_hp_fraction(seats: NDArray[np.int32]) -> float:
    """Mean over seats of ``clip(hp, 0) / max_hp`` (a seat with ``max_hp <= 0`` counts 0)."""
    if not len(seats):
        return 0.0
    # <= 4 seats: plain Python beats six numpy calls on 1-4 element arrays (per-decision hot path)
    tot = 0.0
    for hp, mx in seats[:, _HP_COLS].tolist():
        if mx > 0 and hp > 0:
            tot += min(hp / mx, 1.0)
    return tot / len(seats)


def _team_hp_lists(hps: Sequence[int], mxs: Sequence[int]) -> float:
    """:func:`team_hp_fraction` of seats given as hp / max_hp lists (same arithmetic, same order)."""
    if not hps:
        return 0.0
    tot = 0.0
    for hp, mx in zip(hps, mxs):
        if mx > 0 and hp > 0:
            tot += min(hp / mx, 1.0)
    return tot / len(hps)


_SEATS_T, _SEAT_W = int(Table.SEATS), N_COLS[Table.SEATS]
_S_HP, _S_MAX = int(SeatCol.HP), int(SeatCol.MAX_HP)
_G_KIND, _G_FLOOR, _G_ROOM, _G_IC = int(G.KIND), int(G.TOTAL_FLOOR), int(G.ROOM_TYPE), int(G.IN_COMBAT)
_G_ELITES, _G_BOSSES, _G_REASON = int(G.ELITES_WON), int(G.BOSSES_WON), int(G.TERMINAL_REASON)
_OBS, _TERMINAL, _F_OVF = int(MsgType.OBS), int(MsgType.TERMINAL), int(FrameFlag.OVERFLOW)
_SEAT_ROWS = [struct.Struct(f"<{k * _SEAT_W}i") for k in range(MAX_ROWS[Table.SEATS] + 1)]


def _gather_struct(indices: Sequence[int]) -> tuple[struct.Struct, int]:
    """A struct reading the int32 words ``indices`` (ascending) of a block, and its byte offset."""
    fmt, pos = "<", indices[0]
    for i in indices:
        fmt += f"{(i - pos) * 4}xi" if i > pos else "i"
        pos = i + 1
    return struct.Struct(fmt), GLOBAL_OFFSET + indices[0] * 4


assert _G_KIND < _G_FLOOR < _G_ROOM < _G_IC < _G_ELITES < _G_BOSSES < _G_REASON
_FACT_G, _FACT_OFF = _gather_struct((_G_KIND, _G_FLOOR, _G_ROOM, _G_IC, _G_ELITES, _G_BOSSES, _G_REASON))


def frame_facts(frame: Frame) -> Facts:
    """Extract :class:`Facts` from an OBS or TERMINAL frame."""
    msg = frame.msg_type
    if msg != _OBS and msg != _TERMINAL:
        raise ValueError(f"facts need an OBS or TERMINAL frame, got {frame!r}")
    kind, floor, room, ic, elites, bosses, reason = _FACT_G.unpack_from(frame.payload, _FACT_OFF)
    n = frame.rows[_SEATS_T]
    if n >= 0 and frame._tables is None:  # SEATS straight from the payload bytes
        v = _SEAT_ROWS[n].unpack_from(frame.payload, frame.offsets[_SEATS_T] * 4)
        hp = _team_hp_lists(v[_S_HP::_SEAT_W], v[_S_MAX::_SEAT_W])
    else:
        hp = team_hp_fraction(frame.table(Table.SEATS))
    return Facts(msg, frame.result, kind, floor, room, bool(ic), hp, frame.n_actions, bool(frame.flags & _F_OVF),
                 elites, bosses, reason if msg == _TERMINAL else 0)


@dataclass(frozen=True)
class RoomIds:
    """Vocab ids (category ``enum:RoomType``) of the rooms whose combats earn kill bonuses."""

    elite: int
    boss: int

    @classmethod
    def from_vocab(cls, vocab: Vocab) -> "RoomIds":
        ids = {n: vocab.id("enum:RoomType", n) for n in ("Elite", "Boss")}
        missing = [n for n, i in ids.items() if i == VOCAB_UNK]
        if missing:
            raise ValueError(f"vocab enum:RoomType lacks {missing}")
        return cls(ids["Elite"], ids["Boss"])


@dataclass(frozen=True)
class RewardConfig:
    floor: float = 0.01
    elite: float = 0.02
    boss: float = 0.1
    win: float = 1.0
    death: float = -0.1
    hp_coef: float = 0.1
    gamma: float = 0.9998  # must equal the advantage estimator's per-decision gamma
    kills: str = "counters"  # "counters": host GLOBAL[ELITES_WON / BOSSES_WON]; "rooms": inferred (old rule)
    caps_as_failure: bool = True  # policy-caused caps: death reward, zero terminal potential, no bootstrap

    def __post_init__(self) -> None:
        if self.kills not in ("counters", "rooms"):
            raise ValueError(f"RewardConfig.kills must be 'counters' or 'rooms', got {self.kills!r}")


class Outcome(IntEnum):
    ONGOING = 0
    WIN = 1
    LOSS = 2  # a death or a policy-caused cap with RewardConfig.caps_as_failure
    ABORTED = 3
    OVERFLOW = 4
    TRUNCATED = 5  # RESULT=LOSS with a truncation reason: a failure, but no terminal game state

    @property
    def finished(self) -> bool:
        """Resolved objective (WIN / death / cap failure): reward, no bootstrap, outcome targets."""
        return self in (Outcome.WIN, Outcome.LOSS)

    @property
    def dropped(self) -> bool:
        return self in (Outcome.ABORTED, Outcome.OVERFLOW)


REWARD_COMPONENTS = ("floor", "elite", "boss", "terminal", "hp")


@dataclass
class EpisodeTargets:
    """Rewards and auxiliary targets of one (possibly unfinished) episode.

    With ``m`` decision frames: ``rewards`` has one entry per transition whose next frame is
    known - ``m`` for an episode with its TERMINAL frame (WIN / LOSS / TRUNCATED), ``m - 1``
    otherwise (ABORTED / OVERFLOW / ONGOING). Per-decision arrays (``combat_id``, ``hp_post``,
    ``hp_mask``, ``floors_left``) have ``m``.
    """

    outcome: Outcome
    rewards: F32
    components: dict[str, F32]
    combat_id: I32  # -1 outside combat; ids number the combats of the episode from 0
    hp_post: F32  # NaN where masked
    hp_mask: NDArray[np.bool_]
    win: float | None = None  # WIN / LOSS episodes only
    floors_left: F32 | None = None  # WIN / LOSS episodes only
    n_combats: int = field(default=0)
    # the five components in float64 (the critic's component rewards are summed from these: notes/critic/contract.md 4.1)
    components64: dict[str, NDArray[np.float64]] = field(default_factory=dict)


def episode_targets(facts: Sequence[Facts], rooms: RoomIds, cfg: RewardConfig = RewardConfig()) -> EpisodeTargets:
    """Rewards and targets from the facts of an episode prefix, in frame order.

    ``facts`` = the decision frames of the episode so far, optionally followed by its TERMINAL
    frame. Vectorised over the episode; pure function of the facts, so a prefix gives the same
    values for the transitions it covers as the full episode.
    """
    if not len(facts):
        raise ValueError("episode_targets needs at least one frame")
    F = np.asarray(facts, dtype=np.float64)
    term = F[:, _MSG] == MsgType.TERMINAL
    if term[:-1].any():
        raise ValueError("TERMINAL frame before the end of the episode")
    ended = bool(term[-1])  # the last frame is the TERMINAL frame
    result = int(F[-1, _RES]) if ended else int(Result.IN_PROGRESS)
    cap_failure = ended and result == Result.LOSS and int(F[-1, _REASON]) in _CAPS and cfg.caps_as_failure
    if F[:, _OVF].any():
        outcome = Outcome.OVERFLOW
    elif ended and result == Result.ABORTED:
        outcome = Outcome.ABORTED
    elif ended and result == Result.LOSS and int(F[-1, _REASON]) in _TRUNC and not cap_failure:
        outcome = Outcome.TRUNCATED
    elif ended:
        outcome = Outcome.WIN if result == Result.WIN else Outcome.LOSS
    else:
        outcome = Outcome.ONGOING
    if ended and outcome.dropped:  # an ABORTED terminal is not a transition
        F, term, ended = F[:-1], term[:-1], False
    if outcome == Outcome.TRUNCATED:  # the TERMINAL frame is the next state, not a terminal one
        term = np.zeros_like(term)
    finished = outcome.finished
    n = len(F)
    m = n - int(ended)  # decision frames
    if m == 0:
        raise ValueError("episode without decision frames")

    floor = F[:, _FLOOR]
    ic = (F[:, _IC] != 0) & ~term
    if cap_failure:
        ic[-1] = bool(F[-1, _IC])  # failure ends the objective, but does not observe post-combat HP
    hp = F[:, _HP]
    phi = cfg.hp_coef * np.where(term, 0.0, hp)

    same_floor = floor[1:] == floor[:-1]
    continues = ic[:-1] & ic[1:] & same_floor  # frames t and t+1 are in the same combat
    ends = ic[:-1] & ~continues  # a combat ends at transition t
    if cfg.kills == "counters":
        elite_k = np.maximum(F[1:, _ELITES] - F[:-1, _ELITES], 0.0)
        boss_k = np.maximum(F[1:, _BOSSES] - F[:-1, _BOSSES], 0.0)
    else:
        elite_k, boss_k = _inferred(F, ends, term, outcome, rooms)
    comp = {
        "floor": cfg.floor * np.maximum(floor[1:] - floor[:-1], 0.0),
        "elite": cfg.elite * elite_k,
        "boss": cfg.boss * boss_k,
        "terminal": np.zeros(n - 1),
        "hp": cfg.gamma * phi[1:] - phi[:-1],
    }
    if finished:
        comp["terminal"][-1] = cfg.win if outcome == Outcome.WIN else cfg.death
    rewards = sum(comp.values(), np.zeros(n - 1))

    icd = ic[:m]
    start = icd & ~np.concatenate(([False], continues[: m - 1]))
    combat_id = np.where(icd, np.cumsum(start) - 1, -1).astype(np.int32)
    n_combats = int(start.sum())
    post = np.full(n_combats, np.nan)
    end_t = np.flatnonzero(ends)
    post[combat_id[end_t]] = hp[end_t + 1]
    if outcome == Outcome.WIN and ends[-1]:
        post[combat_id[-1]] = hp[m - 1]  # WIN TERMINAL's zero HP is a host encoding artifact
    hp_post = np.where(combat_id >= 0, post[np.maximum(combat_id, 0)] if n_combats else np.nan, np.nan)
    out = EpisodeTargets(
        outcome=outcome,
        rewards=rewards.astype(np.float32),
        components={k: v.astype(np.float32) for k, v in comp.items()},
        combat_id=combat_id,
        hp_post=hp_post.astype(np.float32),
        hp_mask=np.isfinite(hp_post),
        n_combats=n_combats,
        components64=comp,
    )
    if finished:
        out.win = float(outcome == Outcome.WIN)
        out.floors_left = np.maximum(floor[-1] - floor[:m], 0.0).astype(np.float32)
    return out


def _inferred(F: NDArray[np.float64], ends: NDArray[np.bool_], term: NDArray[np.bool_], outcome: Outcome,
              rooms: RoomIds) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Elite / boss kills per transition inferred from ROOM_TYPE (the rule before errata-v1 PY-5)."""
    room = F[:, _ROOM]
    kill = ends & ~(term[1:] & (outcome == Outcome.LOSS))
    return (kill & (room[:-1] == rooms.elite)).astype(np.float64), (kill & (room[:-1] == rooms.boss)).astype(np.float64)


def inferred_kills(facts: Sequence[Facts], rooms: RoomIds) -> tuple[int, int]:
    """``(elite, boss)`` kills of an episode inferred from ROOM_TYPE / IN_COMBAT / TOTAL_FLOOR (the
    pre-errata rule): a cross-check of the host counters, which agreed on 30/30 golden episodes."""
    t = episode_targets(facts, rooms, RewardConfig(kills="rooms", elite=1.0, boss=1.0))
    return int(round(float(t.components["elite"].sum()))), int(round(float(t.components["boss"].sum())))
