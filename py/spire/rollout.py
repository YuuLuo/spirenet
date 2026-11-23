"""Trajectory storage for asynchronous actors, GAE, and the learner's segment ring.

Data flow::

    actor:  RolloutStore.begin_episode / add_step / end_episode | abort_episode | truncate_episode
              -> sealed Segment objects -> sink (e.g. SegmentRing.put, or a queue to the learner)
    learner: SegmentRing.take(>= N samples) -> build_batch (GAE) -> PPO update

Segments
    A :class:`Segment` holds up to ``segment_len`` consecutive decisions of **one** episode of
    one environment (an episode of length L becomes ceil(L / segment_len) segments; segments
    never span episodes, so dropping an episode drops whole segments). How it ends:

    * ``CUT``       the episode continues in the next segment; ``bootstrap_value`` = V(s_T),
                    the behaviour value of the next segment's first state;
    * ``TERMINAL``  the episode ended (WIN / death / cap failure) after the last step; no bootstrap;
    * ``TRUNCATED`` the episode was stopped without a terminal state (actor shutdown, the
                    already released part of an episode that later aborted, or a host truncation:
                    TERMINAL(LOSS) with OTHER_TRUNCATION, or a cap with caps_as_failure=False);
                    bootstrap V(s_T).

Release policy (episode gate)
    Segments are held per environment until their episode resolves, because the auxiliary
    targets (final win, floors remaining) and the "drop aborted episodes" rule need the end of
    the episode. On WIN/LOSS/TRUNCATED all pending segments get rewards + targets and are released; on
    ABORTED / overflow / host loss they are dropped and counted. If an episode keeps more than
    ``max_pending_steps`` steps pending, its sealed segments are force-released with the
    targets known so far (win / floors masked) - should it abort later, only the unreleased
    remainder is dropped and the released part stands as a truncated trajectory.

GAE semantics (:func:`gae`)
    ``delta_t = r_t + gamma * V(s_{t+1}) * [s_{t+1} not terminal] - V(s_t)``;
    ``A_t = delta_t + gamma * lam * A_{t+1}`` within a segment; the recursion stops at the
    segment's last step, where ``V(s_{t+1})`` is the bootstrap value (CUT / TRUNCATED) or 0
    (TERMINAL). Returns (value targets) are ``A_t + V(s_t)``: the TD(lambda) return.

    Optional macro/micro mode (``GAEConfig.macro_micro``, off by default): decisions inside a
    combat (``combat_id >= 0``) form one stream per combat - per-decision GAE(gamma, lam_micro)
    whose trace stops at the combat's last decision, bootstrapping V of the first post-combat
    state. All other (map-level) decisions form one stream: each macro step's TD target spans
    up to the next macro step as an option (semi-MDP): ``R = sum_k gamma^k r_{t+k}`` over the
    k decisions in between, discount ``gamma^k``, bootstrap V(next macro state), trace decay
    ``gamma^k * lam_macro``. Each episode uses ``lam_macro ** (1/n_seats)`` and
    ``lam_micro ** (1/n_seats)`` so trace decay is per single-seat-equivalent decision.
    A macro successor may be another seat's next room decision; combat remains one union stream.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .rewards import EpisodeTargets, Facts, Outcome, RewardConfig, RoomIds, episode_targets
from .schema import MsgType
from .tokenize import Sample, SampleBlock

_TERMINAL = int(MsgType.TERMINAL)
_time = time.time

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
Floats = NDArray[np.floating[Any]]


class SegmentEnd(IntEnum):
    CUT = 0
    TERMINAL = 1
    TRUNCATED = 2


@dataclass(frozen=True)
class EpisodeMeta:
    env_id: int
    episode_uid: int  # unique within a RolloutStore (assigned by begin_episode)
    characters: tuple[int, ...]  # character vocab id per seat
    ascension: int
    seed_id: int = -1  # the trainer's seed / panel id of the run (never observed by the policy)
    fork_group: int = 0  # p12: non-zero positive group id shared by sibling practice children (0 = off / plain)
    fork_start: int = 0  # p12: episode step index of the fork decision (door step) for fork_group != 0

    def __post_init__(self) -> None:
        assert 1 <= len(self.characters) <= 4
        assert self.fork_group >= 0 and self.fork_start >= 0


@dataclass
class Segment:
    """Up to ``segment_len`` consecutive decisions of one episode (see module docstring)."""

    meta: EpisodeMeta
    start: int  # index of the first step within its episode
    samples: Sequence[Any]  # opaque Tokenizer samples (a SampleBlock for tokenizer samples)
    actions: NDArray[np.int64]
    logp: F32  # behaviour log-prob of the chosen action
    values: F32  # behaviour value estimate V(s_t)
    kinds: NDArray[np.int8]  # GLOBAL KIND per step
    n_legal: NDArray[np.int16]  # legal action count per step
    policy_version: NDArray[np.int64]  # acting weights version per step
    end: SegmentEnd = SegmentEnd.CUT
    bootstrap_value: float = 0.0
    bootstrap_sample: Any = None  # sample of s_T for value recomputation (CUT / TRUNCATED)
    rewards: F32 = field(default_factory=lambda: np.zeros(0, np.float32))
    combat_id: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, np.int32))
    win: F32 = field(default_factory=lambda: np.zeros(0, np.float32))
    win_mask: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, bool))
    floors_left: F32 = field(default_factory=lambda: np.zeros(0, np.float32))
    hp_post: F32 = field(default_factory=lambda: np.zeros(0, np.float32))
    hp_mask: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, bool))
    t_first: float = 0.0  # wall-clock time (time.time()) of the first / last decision: policy lag in seconds
    t_last: float = 0.0
    # advice record (spire.advice; None when the store records none): candidate ACTIONS row per step (-1 = none)
    # and the acting seat's HP / max HP at candidate steps (0 elsewhere)
    advice_row: NDArray[np.int16] | None = None
    advice_hp: F32 | None = None
    # habit-rule record (spire.advice; None when the store records none): index of the rule that judged the step
    # (-1 = none) and its sign (+1 / -1; 0 where no rule judged)
    advice_rule: NDArray[np.int8] | None = None
    advice_sign: NDArray[np.int8] | None = None
    # taste record (spire.taste; None when the store records none): t_keep = 1 - eps where taste was active (else 1)
    # and t_add = eps [a in F] / |F| (else 0) per step
    t_keep: F32 | None = None
    t_add: F32 | None = None
    # behaviour temperature record (spire.temperature; None when the store records none): T(step) the actor sampled
    # the step with (ppo.combat_temperature on combat steps, 1.0 elsewhere)
    beh_temp: F32 | None = None
    # critic record (notes/critic/contract.md 3; None when the store records none): the tower's behaviour values
    # (V_WIN, V_PROG, V_SURV) per step, the side input CRIT per step, the tower's bootstrap values of s_T (zeros for
    # TERMINAL) and the component rewards (R_WIN, R_PROG, R_SURV) per step (filled at release)
    values_c: F32 | None = None
    crit: NDArray[np.int16] | None = None
    bootstrap_c: F32 | None = None
    rewards_c: F32 | None = None
    # p12: floor + elite + boss portion of rewards per step (filled at release iff shaping fade is active; else None)
    rewards_prog: F32 | None = None

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def done(self) -> NDArray[np.bool_]:
        d = np.zeros(len(self), bool)
        d[-1] = self.end == SegmentEnd.TERMINAL
        return d

    @property
    def truncated(self) -> NDArray[np.bool_]:
        d = np.zeros(len(self), bool)
        d[-1] = self.end == SegmentEnd.TRUNCATED
        return d

    @property
    def version_min(self) -> int:
        return int(self.policy_version.min())

    def _fill(self, t: EpisodeTargets, *, record_prog: bool = False) -> None:
        sl = slice(self.start, self.start + len(self))
        self.rewards = t.rewards[sl].copy()
        self.combat_id = t.combat_id[sl].copy()
        self.hp_post = np.nan_to_num(t.hp_post[sl], nan=0.0)
        self.hp_mask = t.hp_mask[sl].copy()
        n = len(self)
        if t.win is not None and t.floors_left is not None:
            self.win = np.full(n, t.win, np.float32)
            self.win_mask = np.ones(n, bool)
            self.floors_left = t.floors_left[sl].copy()
        else:
            self.win = np.zeros(n, np.float32)
            self.win_mask = np.zeros(n, bool)
            self.floors_left = np.zeros(n, np.float32)
        assert len(self.rewards) == n, "rewards missing for a released step"
        if self.values_c is not None:
            self.rewards_c = reward_groups(t)[sl].astype(np.float32)
        if record_prog:
            c = t.components64
            self.rewards_prog = (c["floor"] + c["elite"] + c["boss"])[sl].astype(np.float32)


def reward_groups(t: EpisodeTargets) -> F64:
    """(transitions, 3) float64 component rewards (R_WIN, R_PROG, R_SURV) of ``t`` (notes/critic/contract.md 4.1):
    R_WIN = terminal if WIN, R_PROG = floor + elite + boss, R_SURV = hp + (terminal if LOSS). ``terminal`` is non-zero
    on at most one step with one outcome, so the three sum to ``rewards`` exactly in real numbers."""
    c = t.components64
    term = c["terminal"]
    zero = np.zeros_like(term)
    win = term if t.outcome == Outcome.WIN else zero
    surv = c["hp"] + (term if t.outcome == Outcome.LOSS else zero)
    return np.stack([win, c["floor"] + c["elite"] + c["boss"], surv], 1)


# =========================================================================== actor-side store


@dataclass(frozen=True)
class RolloutConfig:
    segment_len: int = 256
    max_pending_steps: int = 4096  # per environment; beyond it sealed segments are force-released
    keep_bootstrap_sample: bool = True


@dataclass
class RolloutStats:
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    aborted: int = 0
    overflow: int = 0
    lost: int = 0  # abort_episode() with reason "lost" (host timeout / crash / ERROR)
    shutdown: int = 0  # abort_episode() with reason "shutdown" (actor stopped mid-episode)
    truncated: int = 0  # truncate_episode() (stopped by the caller)
    host_truncated: int = 0  # TERMINAL with RESULT=LOSS and a truncation reason (decision / turn cap)
    aborted_after_release: int = 0  # dropped episodes whose head was already force-released
    steps_added: int = 0
    steps_released: int = 0
    steps_dropped: int = 0
    segments_released: int = 0
    forced_releases: int = 0
    # R07: steps shipped by a forced release (win / floors labels masked for good), by the episode's final outcome
    censored_win_steps: int = 0
    censored_loss_steps: int = 0
    win_steps: int = 0  # all steps of won / lost episodes (the denominators)
    loss_steps: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class _Open:
    __slots__ = ("start", "samples", "actions", "logp", "values", "kinds", "n_legal", "versions", "t_first", "t_last",
                 "advice_row", "advice_hp", "advice_rule", "advice_sign", "t_keep", "t_add", "beh_temp", "values_c",
                 "crit")

    def __init__(self, start: int) -> None:
        self.start = start
        self.t_first = self.t_last = 0.0
        self.samples: list[Any] = []
        self.actions: list[int] = []
        self.logp: list[float] = []
        self.values: list[float] = []
        self.kinds: list[int] = []
        self.n_legal: list[int] = []
        self.versions: list[int] = []
        self.advice_row: list[int] = []
        self.advice_hp: list[float] = []
        self.advice_rule: list[int] = []
        self.advice_sign: list[int] = []
        self.t_keep: list[float] = []
        self.t_add: list[float] = []
        self.beh_temp: list[float] = []
        self.values_c: list[NDArray[np.float32]] = []
        self.crit: list[NDArray[np.int16]] = []

    def __len__(self) -> int:
        return len(self.actions)

    def seal(self, meta: EpisodeMeta, end: SegmentEnd, boot_value: float, boot_sample: Any, advice: bool,
             judged: bool, taste: bool, beh_temp: bool, *, critic: bool = False,
             boot_c: NDArray[np.float32] | None = None) -> Segment:
        """``critic``: the segment carries the critic record; ``boot_c`` = the tower's values of s_T (None: zeros, a
        TERMINAL segment)."""
        samples: Sequence[Any] = self.samples
        if samples and type(samples[0]) is Sample and all(type(x) is Sample for x in samples):
            # compact: the actor's samples are views into whole inference batches
            samples = SampleBlock.from_samples(self.samples)
        assert critic or boot_c is None
        if critic:
            assert (boot_c is None) == (end == SegmentEnd.TERMINAL), "a tower bootstrap exactly when not TERMINAL"
            crit_seg = dict(values_c=np.stack(self.values_c).astype(np.float32), crit=np.stack(self.crit),
                            bootstrap_c=(np.zeros(3, np.float32) if boot_c is None
                                         else np.asarray(boot_c, np.float32).copy()))
        else:
            crit_seg = {}
        return Segment(**crit_seg,
            meta=meta, start=self.start, samples=samples,
            actions=np.asarray(self.actions, np.int64), logp=np.asarray(self.logp, np.float32),
            values=np.asarray(self.values, np.float32), kinds=np.asarray(self.kinds, np.int8),
            n_legal=np.asarray(self.n_legal, np.int16), policy_version=np.asarray(self.versions, np.int64),
            end=end, bootstrap_value=float(boot_value), bootstrap_sample=boot_sample,
            t_first=self.t_first, t_last=self.t_last,
            advice_row=np.asarray(self.advice_row, np.int16) if advice else None,
            advice_hp=np.asarray(self.advice_hp, np.float32) if advice else None,
            advice_rule=np.asarray(self.advice_rule, np.int8) if judged else None,
            advice_sign=np.asarray(self.advice_sign, np.int8) if judged else None,
            t_keep=np.asarray(self.t_keep, np.float32) if taste else None,
            t_add=np.asarray(self.t_add, np.float32) if taste else None,
            beh_temp=np.asarray(self.beh_temp, np.float32) if beh_temp else None,
        )


class _Track:
    __slots__ = ("meta", "facts", "open", "pending", "n_steps", "released", "base", "stats", "fight_only", "scout")

    def __init__(self, meta: EpisodeMeta, stats: RolloutStats, fight_only: bool = False, scout: bool = False) -> None:
        self.meta = meta
        self.facts: list[Facts] = []
        self.open = _Open(0)
        self.pending: list[Segment] = []
        self.n_steps = 0
        self.released = 0  # steps already force-released
        self.base = 0  # a fork child (fork_episode): the steps before the fork, which are its parent's
        self.stats = stats  # RolloutStore.stats, gym_stats for a fork child, scout_stats for a scout
        # a fight-only practice child: it knows the run's outcome only when it dies, so its win / floors targets are
        # masked (a GAME child, practice v1.2, plays to the run's end: its labels are the true ones)
        self.fight_only = fight_only
        self.scout = scout  # a scout (practice v1.1): facts only (its children's prefix), never released

    @property
    def n_pending(self) -> int:
        # every step is in a pending segment, the open one, or already released
        return self.n_steps - self.released


class RolloutStore:
    """Per-actor trajectory builder for many environments (not thread-safe; one per actor loop).

    For every decision the actor calls :meth:`add_step` with the frame's :class:`Facts`, the
    opaque sample it fed the policy, the chosen action, its behaviour log-prob and value, and
    the version of the acting weights. When the episode ends it calls :meth:`end_episode` with
    the TERMINAL frame's facts (an ABORTED result is routed to :meth:`abort_episode`).
    ``advice``: every step carries an advice record ``(row, hp)`` (:mod:`spire.advice`) into its segment;
    ``judged``: every step carries a habit-rule record ``(rule, sign)``.
    Fork children (practice fights, :meth:`fork_episode`) are counted in ``gym_stats``, scouts
    (``begin_episode(scout=True)``, :meth:`add_scout_step`: facts only, never released) in ``scout_stats``;
    ``stats`` stays trained full games only.
    """

    def __init__(self, sink: Callable[[Segment], None], rooms: RoomIds, *,
                 config: RolloutConfig = RolloutConfig(), rewards: RewardConfig = RewardConfig(),
                 advice: bool = False, judged: bool = False, taste: bool = False, beh_temp: bool = False,
                 critic_k: int = 0, record_prog: bool = False) -> None:
        """``critic_k`` > 0 (a critic run's training actor, notes/critic/contract.md 3): every step carries the
        tower's values (3 floats) and its CRIT row (2 + critic_k int16), every non-TERMINAL segment the tower's
        bootstrap values, every released segment its component rewards."""
        if config.segment_len < 1:
            raise ValueError("segment_len must be >= 1")
        assert critic_k >= 0
        self.critic_k = critic_k
        self.critic = critic_k > 0
        self.record_prog = record_prog
        self.sink = sink
        self.rooms = rooms
        self.cfg = config
        self.reward_cfg = rewards
        self.advice = advice
        self.judged = judged
        self.taste = taste
        self.beh_temp = beh_temp  # every step carries its behaviour temperature (spire.temperature)
        self.stats = RolloutStats()
        self.gym_stats = RolloutStats()
        self.scout_stats = RolloutStats()
        self._tracks: dict[int, _Track] = {}
        self._next_uid = 0

    def in_episode(self, env_id: int) -> bool:
        return env_id in self._tracks

    def begin_episode(self, env_id: int, characters: Sequence[int], ascension: int = 0,
                      seed_id: int = -1, *, scout: bool = False) -> EpisodeMeta:
        if env_id in self._tracks:
            raise RuntimeError(f"env {env_id}: begin_episode while an episode is open")
        meta = EpisodeMeta(env_id, self._next_uid, tuple(int(c) for c in characters), int(ascension), int(seed_id))
        self._next_uid += 1
        stats = self.scout_stats if scout else self.stats
        self._tracks[env_id] = _Track(meta, stats, scout=scout)
        stats.episodes += 1
        return meta

    def fork_point(self, env_id: int) -> tuple[EpisodeMeta, list[Facts]]:
        """The open episode of ``env_id`` as fork children start from it (boss gym): its metadata and the facts
        of its decisions so far (a copy; the next decision is the fork decision)."""
        tr = self._track(env_id)
        return tr.meta, list(tr.facts)

    def fork_episode(self, point: tuple[EpisodeMeta, list[Facts]], env_id: int, *, game: bool,
                     fork_group: int = 0) -> EpisodeMeta:
        """Open the episode of a fork child of the episode ``point`` (:meth:`fork_point`) was taken from. The
        child is an episode of its own (``env_id``, a new uid) whose steps continue the parent's numbering: the
        parent's facts are its prefix, so its rewards (potentials), combat ids and targets are those of a game
        played from the start, and its first segment starts at the fork decision. The parent's steps are never
        released through the child. ``game``: a GAME child (practice v1.2) whose win / floors labels are trained;
        otherwise a fight-only child (masked)."""
        if env_id in self._tracks:
            raise RuntimeError(f"env {env_id}: fork_episode while an episode is open")
        src, facts = point
        meta = EpisodeMeta(env_id, self._next_uid, src.characters, src.ascension, src.seed_id,
                           fork_group, len(facts) if fork_group else 0)
        self._next_uid += 1
        tr = _Track(meta, self.gym_stats, fight_only=not game)
        tr.facts = list(facts)
        tr.base = tr.n_steps = tr.released = len(facts)
        tr.open = _Open(len(facts))
        self._tracks[env_id] = tr
        self.gym_stats.episodes += 1
        return meta

    def add_step(self, env_id: int, facts: Facts, sample: Any, action: int, logp: float, value: float,
                 policy_version: int, advice: tuple[int, float] | None = None,
                 judged: tuple[int, int] | None = None, *, taste: tuple[float, float] | None = None,
                 beh_temp: float | None = None,
                 critic: tuple[NDArray[np.float32], NDArray[np.int16]] | None = None) -> bool:
        """Record the decision taken at a frame (``advice`` / ``judged``: its advice / habit-rule record, exactly when
        the store records them; ``critic``: the tower's values (3,) and the CRIT row (2 + K,) of the frame, exactly
        when the store records them). Returns False if the episode was dropped instead (the frame carries the
        overflow flag); the actor should then reset the env."""
        assert (advice is not None) == self.advice, "advice record given iff the store records advice"
        assert (judged is not None) == self.judged, "habit-rule record given iff the store records it"
        assert (taste is not None) == self.taste, "taste record given iff the store records it"
        assert (beh_temp is not None) == self.beh_temp, "behaviour temperature given iff the store records it"
        assert (critic is not None) == self.critic, "critic record given iff the store records it"
        if critic is not None:
            assert critic[0].shape == (3,) and critic[1].shape == (2 + self.critic_k,) and critic[1].dtype == np.int16
        tr = self._tracks.get(env_id)
        if tr is None:
            raise RuntimeError(f"env {env_id}: no open episode")
        assert not tr.scout, "a scout's steps are add_scout_step"
        if facts.msg_type == _TERMINAL:
            raise ValueError("add_step with a TERMINAL frame; use end_episode")
        if facts.overflow:
            self.abort_episode(env_id, "overflow")
            return False
        op = tr.open
        cfg = self.cfg
        if len(op.actions) == cfg.segment_len:
            tr.pending.append(op.seal(tr.meta, SegmentEnd.CUT, value, sample if cfg.keep_bootstrap_sample else None,
                                      self.advice, self.judged, self.taste, self.beh_temp, critic=self.critic,
                                      boot_c=None if critic is None else critic[0]))
            tr.open = op = _Open(tr.n_steps)
        now = _time()
        if not op.actions:
            op.t_first = now
        op.t_last = now
        op.samples.append(sample)
        op.actions.append(int(action))
        op.logp.append(float(logp))
        op.values.append(float(value))
        op.kinds.append(facts.kind)
        op.n_legal.append(facts.n_actions)
        op.versions.append(int(policy_version))
        if advice is not None:
            op.advice_row.append(advice[0])
            op.advice_hp.append(advice[1])
        if judged is not None:
            op.advice_rule.append(judged[0])
            op.advice_sign.append(judged[1])
        if taste is not None:
            op.t_keep.append(taste[0])
            op.t_add.append(taste[1])
        if beh_temp is not None:
            op.beh_temp.append(beh_temp)
        if critic is not None:
            op.values_c.append(critic[0])
            op.crit.append(critic[1])
        tr.facts.append(facts)
        tr.n_steps += 1
        tr.stats.steps_added += 1
        if tr.pending and tr.n_steps - tr.released > cfg.max_pending_steps:
            self._force_release(tr)
        return True

    def add_scout_step(self, env_id: int, facts: Facts) -> bool:
        """A scout's decision (practice v1.1): only its facts are kept (the prefix of its children's episodes).
        False: the frame overflowed and the episode was dropped."""
        tr = self._track(env_id)
        assert tr.scout, "add_scout_step on a trained episode"
        if facts.msg_type == _TERMINAL:
            raise ValueError("add_scout_step with a TERMINAL frame; use end_episode")
        if facts.overflow:
            self.abort_episode(env_id, "overflow")
            return False
        tr.facts.append(facts)
        tr.n_steps += 1
        tr.stats.steps_added += 1
        return True

    def end_episode(self, env_id: int, terminal: Facts, *, bootstrap_value: float | None = None,
                    bootstrap_sample: Any = None, bootstrap_c: NDArray[np.float32] | None = None) -> Outcome:
        """Close the episode with its TERMINAL frame; releases (WIN / LOSS / TRUNCATED) or drops
        (ABORTED / overflow).

        Policy-caused caps are terminal failures by default. Other host truncations (and caps with
        ``caps_as_failure=False``) end ``TRUNCATED`` and bootstrap with
        ``bootstrap_value`` = V(TERMINAL frame) when the caller has it, else with the behaviour value
        of the last decision (V(s_{T-1}) for V(s_T): the actor does not run the policy on TERMINAL
        frames; a one-step approximation on rare episodes); win / floors targets are masked. A critic store
        bootstraps the tower the same way: ``bootstrap_c`` with ``bootstrap_value``, else the last decision's values.
        """
        assert bootstrap_c is None or (self.critic and bootstrap_value is not None), \
            "bootstrap_c comes with bootstrap_value, in a critic store"
        assert not (self.critic and bootstrap_value is not None) or bootstrap_c is not None, \
            "a critic store's bootstrap_value needs bootstrap_c"
        if not terminal.terminal:
            raise ValueError("end_episode needs the TERMINAL frame's facts")
        tr = self._track(env_id)
        if tr.n_steps == tr.base:
            self._drop(env_id, "aborted")
            return Outcome.ABORTED
        targets = episode_targets([*tr.facts, terminal], self.rooms, self.reward_cfg)
        if tr.scout:  # nothing to release: counted in scout_stats
            del self._tracks[env_id]
            if targets.outcome.dropped:
                name = "overflow" if targets.outcome == Outcome.OVERFLOW else "aborted"
                setattr(tr.stats, name, getattr(tr.stats, name) + 1)
            elif terminal.truncated:
                tr.stats.host_truncated += 1
            elif targets.outcome == Outcome.WIN:
                tr.stats.wins += 1
            elif targets.outcome == Outcome.LOSS:
                tr.stats.losses += 1
            return targets.outcome
        if targets.outcome.dropped:
            self._drop(env_id, "overflow" if targets.outcome == Outcome.OVERFLOW else "aborted")
            return targets.outcome
        if targets.outcome == Outcome.TRUNCATED:
            op = tr.open
            if bootstrap_value is None:  # V(s_{T-1}) and its sample stand in for V(s_T)
                bootstrap_value, bootstrap_sample = op.values[-1], op.samples[-1]
                if self.critic:
                    bootstrap_c = op.values_c[-1]
            seg = op.seal(tr.meta, SegmentEnd.TRUNCATED, bootstrap_value,
                          bootstrap_sample if self.cfg.keep_bootstrap_sample else None, self.advice, self.judged, self.taste, self.beh_temp,
                          critic=self.critic, boot_c=bootstrap_c)
        else:
            seg = tr.open.seal(tr.meta, SegmentEnd.TERMINAL, 0.0, None, self.advice, self.judged, self.taste, self.beh_temp,
                               critic=self.critic)
        tr.pending.append(seg)
        censored = tr.released - tr.base  # before this release: the force-released steps (a child's prefix excluded)
        self._release(tr, targets, len(tr.pending))
        del self._tracks[env_id]
        if terminal.truncated:  # retain the host-reason diagnostic even when caps train as failures
            tr.stats.host_truncated += 1
        elif targets.outcome == Outcome.WIN:
            tr.stats.wins += 1
            tr.stats.censored_win_steps += censored
            tr.stats.win_steps += tr.n_steps - tr.base
        elif targets.outcome == Outcome.LOSS:
            tr.stats.losses += 1
            tr.stats.censored_loss_steps += censored
            tr.stats.loss_steps += tr.n_steps - tr.base
        return targets.outcome

    def truncate_episode(self, env_id: int, next_facts: Facts, next_value: float, next_sample: Any = None,
                         next_values_c: NDArray[np.float32] | None = None) -> None:
        """Stop the episode without a terminal state at the not-yet-acted frame ``next_facts``
        (value ``next_value``; a critic store: the tower's values ``next_values_c`` too): every step is released,
        the last segment bootstraps."""
        assert (next_values_c is not None) == self.critic, "the tower's bootstrap values iff a critic store"
        tr = self._track(env_id)
        if next_facts.terminal:
            raise ValueError("truncate_episode needs a non-terminal frame; use end_episode")
        if next_facts.overflow or any(f.overflow for f in tr.facts):
            self._drop(env_id, "overflow")
            return
        if tr.n_steps > tr.base:
            targets = episode_targets([*tr.facts, next_facts], self.rooms, self.reward_cfg)
            tr.pending.append(tr.open.seal(tr.meta, SegmentEnd.TRUNCATED, next_value,
                                           next_sample if self.cfg.keep_bootstrap_sample else None, self.advice, self.judged, self.taste, self.beh_temp,
                                           critic=self.critic, boot_c=next_values_c))
            self._release(tr, targets, len(tr.pending))
        del self._tracks[env_id]
        tr.stats.truncated += 1

    def abort_episode(self, env_id: int, reason: str = "aborted") -> None:
        """Drop the episode's unreleased steps. ``reason``: "aborted" (RESULT=ABORTED),
        "overflow" (FLAGS overflow), "lost" (host timeout / crash / ERROR frame) or
        "shutdown" (the actor stops; nobody would consume a truncated tail)."""
        if reason not in ("aborted", "overflow", "lost", "shutdown"):
            raise ValueError(f"unknown abort reason {reason!r}")
        self._track(env_id)
        self._drop(env_id, reason)

    def close_all(self, reason: str = "lost") -> None:
        for env_id in list(self._tracks):
            self.abort_episode(env_id, reason)

    # ------------------------------------------------------------------ internals
    def _track(self, env_id: int) -> _Track:
        tr = self._tracks.get(env_id)
        if tr is None:
            raise RuntimeError(f"env {env_id}: no open episode")
        return tr

    def _drop(self, env_id: int, reason: str) -> None:
        tr = self._tracks.pop(env_id)
        setattr(tr.stats, reason, getattr(tr.stats, reason) + 1)
        tr.stats.steps_dropped += tr.n_steps - tr.released
        if tr.released > tr.base:
            tr.stats.aborted_after_release += 1

    def _release(self, tr: _Track, targets: EpisodeTargets, n_segments: int) -> None:
        assert not tr.scout, "a scout's segments are never shipped"
        for seg in tr.pending[:n_segments]:
            seg._fill(targets, record_prog=self.record_prog)
            if tr.fight_only:  # spec 2.3 (b): the run's outcome is known to a fight-only child only when it dies
                seg.win = np.zeros(len(seg), np.float32)
                seg.win_mask = np.zeros(len(seg), bool)
                seg.floors_left = np.zeros(len(seg), np.float32)
            self.sink(seg)
            tr.stats.steps_released += len(seg)
            tr.stats.segments_released += 1
            tr.released += len(seg)
        del tr.pending[:n_segments]

    def _force_release(self, tr: _Track) -> None:
        # Every sealed segment has its next step recorded, so its rewards are known.
        self._release(tr, episode_targets(tr.facts, self.rooms, self.reward_cfg), len(tr.pending))
        tr.stats.forced_releases += 1


# =========================================================================== GAE


@dataclass(frozen=True)
class GAEConfig:
    gamma: float = 0.9998
    lam: float = 0.99
    macro_micro: bool = False
    lam_macro: float = 0.95
    lam_micro: float = 0.95
    stitch_segments: bool = False  # trace through adjacent, contiguous fragments of the same episode


@dataclass
class _Padded:
    r: F64  # (N, T)
    v: F64
    valid: NDArray[np.bool_]
    cid: NDArray[np.int32]
    boot: F64  # (N,) value after each segment's last step (0 for TERMINAL)
    lengths: NDArray[np.int64]
    n_seats: NDArray[np.int8]


def _pad(segments: Sequence[Segment], values: Floats | None, bootstrap: Floats | None,
         rewards: Floats | None = None) -> _Padded:
    if not segments:
        raise ValueError("no segments")
    lengths = np.fromiter((len(s) for s in segments), np.int64, len(segments))
    if values is not None and len(values) != int(lengths.sum()):
        raise ValueError("values override has the wrong length")
    if bootstrap is not None and len(bootstrap) != len(segments):
        raise ValueError("bootstrap override has the wrong length")
    assert rewards is None or len(rewards) == int(lengths.sum()), "rewards override has the wrong length"
    n, t = len(segments), int(lengths.max())
    valid = np.arange(t)[None, :] < lengths[:, None]
    r = np.zeros((n, t))
    v = np.zeros((n, t))
    cid = np.full((n, t), -1, np.int32)
    r[valid] = np.concatenate([s.rewards for s in segments]) if rewards is None else rewards
    v[valid] = values if values is not None else np.concatenate([s.values for s in segments])
    cid[valid] = np.concatenate([s.combat_id for s in segments])
    boot = np.fromiter((s.bootstrap_value for s in segments), np.float64, n) if bootstrap is None \
        else np.asarray(bootstrap, np.float64).copy()
    boot[np.fromiter((s.end == SegmentEnd.TERMINAL for s in segments), bool, n)] = 0.0
    n_seats = np.fromiter((len(s.meta.characters) for s in segments), np.int8, len(segments))
    return _Padded(r, v, valid, cid, boot, lengths, n_seats)


def seat_trace_decay(lam: float, n_seats: int) -> float:
    """One controller decision is ``1 / n_seats`` of a single-seat-equivalent decision."""
    assert 1 <= n_seats <= 4 and 0.0 <= lam <= 1.0
    return lam if n_seats == 1 else lam ** (1.0 / n_seats)


def gae(segments: Sequence[Segment], cfg: GAEConfig, values: Floats | None = None,
        bootstrap: Floats | None = None, rewards: Floats | None = None) -> tuple[F32, F32]:
    """Advantages and value targets for ``segments``, flattened in segment order.

    ``values`` / ``bootstrap`` optionally replace the stored behaviour values (flat, one per
    step) and per-segment bootstrap values, e.g. recomputed with the learner's weights; ``rewards`` the stored
    rewards. With a trailing component axis - ``values`` (steps, C), ``bootstrap`` (segments, C), ``rewards``
    (steps, C), all three given - every column is one estimate of its own (:func:`gae_columns`) and the results
    are (steps, C).
    """
    if values is not None and np.ndim(values) == 2:
        adv, v = gae_columns(segments, cfg, values, bootstrap, rewards)  # type: ignore[arg-type]
        return adv.astype(np.float32), (adv + v).astype(np.float32)
    adv, v = _gae64(segments, cfg, values, bootstrap, rewards)
    return adv.astype(np.float32), (adv + v).astype(np.float32)


def _gae64(segments: Sequence[Segment], cfg: GAEConfig, values: Floats | None, bootstrap: Floats | None,
           rewards: Floats | None) -> tuple[F64, F64]:
    """Flat float64 advantages and the float64 values they are relative to (one column)."""
    if cfg.stitch_segments:
        return _gae_stitched(segments, cfg, values, bootstrap, rewards)
    p = _pad(segments, values, bootstrap, rewards)
    adv = _gae_macro_micro(p, cfg) if cfg.macro_micro else _gae_standard(p, cfg.gamma, cfg.lam)
    return adv[p.valid], p.v[p.valid]


def gae_columns(segments: Sequence[Segment], cfg: GAEConfig, values: Floats, bootstrap: Floats,
                rewards: Floats) -> tuple[F64, F64]:
    """GAE of C reward / value streams over the same segments (notes/critic/contract.md 4.2): ``values`` (steps, C),
    ``bootstrap`` (segments, C; ignored at TERMINAL segments), ``rewards`` (steps, C). Returns float64 advantages
    (steps, C) and the float64 values (steps, C); returns = advantages + values. The estimator is linear in (r, V)
    and its trace structure depends only on the segments, so every column runs the one-column code of :func:`gae`
    (column c is exactly ``gae(segments, cfg, values[:, c], bootstrap[:, c], rewards[:, c])`` in float64) and the
    sum of the columns' advantages is the advantage of the summed rewards under the summed values."""
    n = sum(len(s) for s in segments)
    values, bootstrap, rewards = np.asarray(values), np.asarray(bootstrap), np.asarray(rewards)
    C = values.shape[1]
    assert values.shape == (n, C) and rewards.shape == (n, C) and bootstrap.shape == (len(segments), C), \
        (values.shape, rewards.shape, bootstrap.shape)
    adv = np.empty((n, C))
    v = np.empty((n, C))
    for c in range(C):
        adv[:, c], v[:, c] = _gae64(segments, cfg, values[:, c], bootstrap[:, c], rewards[:, c])
    return adv, v


def stitch_joins(segments: Sequence[Segment]) -> NDArray[np.bool_]:
    """``joins[i]``: segment ``i`` continues into segment ``i + 1`` of the same batch (a storage CUT followed by the
    next fragment of the same episode, adjacent in release order). The trace of :func:`_gae_stitched` crosses
    exactly these boundaries; the learner reports their count (``gae_joined_cuts``)."""
    n = len(segments)
    joins = np.zeros(n, bool)
    for i in range(n - 1):
        s, t = segments[i], segments[i + 1]
        joins[i] = s.end == SegmentEnd.CUT and s.meta == t.meta and s.start + len(s) == t.start
    return joins


def _gae_stitched(segments: Sequence[Segment], cfg: GAEConfig, values: Floats | None,
                  bootstrap: Floats | None, rewards: Floats | None = None) -> tuple[F64, F64]:
    """Carry traces over storage cuts, stopping at missing fragments or episode boundaries.

    Segments arrive in release order. Only adjacent fragments with matching episode metadata,
    consecutive positions and a CUT between them form a chain. A scalar scan avoids padding
    every short episode to the length of a rare 30,000-decision episode.
    """
    assert segments
    lengths = np.fromiter((len(s) for s in segments), np.int64, len(segments))
    ends = lengths.cumsum()
    n = int(ends[-1])
    assert values is None or len(values) == n
    assert bootstrap is None or len(bootstrap) == len(segments)
    assert rewards is None or len(rewards) == n
    r = np.concatenate([s.rewards for s in segments]).astype(np.float64) if rewards is None else \
        np.asarray(rewards, np.float64)
    v = np.asarray(values, np.float64) if values is not None else \
        np.concatenate([s.values for s in segments]).astype(np.float64)
    cid = np.concatenate([s.combat_id for s in segments])
    lam_macro = np.repeat([seat_trace_decay(cfg.lam_macro, len(s.meta.characters)) for s in segments], lengths)
    lam_micro = np.repeat([seat_trace_decay(cfg.lam_micro, len(s.meta.characters)) for s in segments], lengths)
    # The last transition of each unjoined fragment uses its own bootstrap, never the
    # first value of an unrelated episode that happens to follow it in the minibatch.
    boundary: dict[int, float] = {}
    joins = stitch_joins(segments)
    for i, s in enumerate(segments):
        if not joins[i]:
            boundary[int(ends[i]) - 1] = (0.0 if s.end == SegmentEnd.TERMINAL else
                                         float(bootstrap[i]) if bootstrap is not None else s.bootstrap_value)
    adv = np.zeros(n, np.float64)
    a_mu = s_acc = dist = hv = ha = next_v = 0.0
    have_macro = False
    g = cfg.gamma
    for t in range(n - 1, -1, -1):
        if t in boundary:
            a_mu = s_acc = dist = ha = 0.0
            have_macro = False
            next_v = hv = boundary[t]
        delta = r[t] + g * next_v - v[t]
        if not cfg.macro_micro:
            a_mu = delta + g * cfg.lam * a_mu
            adv[t] = a_mu
        else:
            same_combat = t not in boundary and cid[t] >= 0 and cid[t + 1] == cid[t]
            a_mu = delta + g * lam_micro[t] * a_mu * same_combat
            if cid[t] < 0:
                disc = g ** (1.0 + dist)
                a_mac = r[t] + g * s_acc + disc * hv - v[t] + disc * lam_macro[t] * have_macro * ha
                adv[t] = a_mac
                s_acc = dist = 0.0
                hv, ha, have_macro = v[t], a_mac, True
            else:
                adv[t] = a_mu
                s_acc = r[t] + g * s_acc
                dist += 1.0
        next_v = v[t]
    return adv, v


def _next_values(p: _Padded) -> F64:
    nv = np.zeros_like(p.v)
    nv[:, :-1] = p.v[:, 1:]
    nv[np.arange(len(nv)), p.lengths - 1] = p.boot
    return nv * p.valid


def _gae_standard(p: _Padded, gamma: float, lam: float) -> F64:
    delta = (p.r + gamma * _next_values(p) - p.v) * p.valid
    cont = np.zeros_like(p.valid)
    cont[:, :-1] = p.valid[:, 1:]
    adv = np.zeros_like(delta)
    carry = np.zeros(len(delta))
    k = gamma * lam
    for t in range(delta.shape[1] - 1, -1, -1):
        carry = delta[:, t] + k * cont[:, t] * carry
        adv[:, t] = carry
    return adv


def _gae_macro_micro(p: _Padded, cfg: GAEConfig) -> F64:
    g = cfg.gamma
    valid, cid = p.valid, p.cid
    micro = valid & (cid >= 0)
    macro = valid & (cid < 0)
    # micro streams: per-decision GAE whose trace stops when the next decision leaves the combat
    delta = (p.r + g * _next_values(p) - p.v) * valid
    cont = np.zeros_like(valid)
    cont[:, :-1] = micro[:, :-1] & micro[:, 1:] & (cid[:, 1:] == cid[:, :-1])
    n, tmax = delta.shape
    adv = np.zeros_like(delta)
    a_mu = np.zeros(n)
    # macro stream state for "the position after t": discounted reward sum up to the next macro
    # step (exclusive), its distance, V and A of that macro step, and whether one exists.
    s_acc = np.zeros(n)
    dist = np.zeros(n)
    hv = p.boot.copy()
    ha = np.zeros(n)
    hm = np.zeros(n, bool)
    k_mu = g * np.asarray([seat_trace_decay(cfg.lam_micro, int(n)) for n in p.n_seats])
    lam_macro = np.asarray([seat_trace_decay(cfg.lam_macro, int(n)) for n in p.n_seats])
    for t in range(tmax - 1, -1, -1):
        a_mu = delta[:, t] + k_mu * cont[:, t] * a_mu
        mac, mic = macro[:, t], micro[:, t]
        k = 1.0 + dist
        disc = g ** k
        a_mac = p.r[:, t] + g * s_acc + disc * hv - p.v[:, t] + disc * lam_macro * hm * ha
        adv[:, t] = np.where(mac, a_mac, np.where(mic, a_mu, 0.0))
        s_acc = np.where(mac, 0.0, np.where(mic, p.r[:, t] + g * s_acc, s_acc))
        dist = np.where(mac, 0.0, np.where(mic, 1.0 + dist, dist))
        hv = np.where(mac, p.v[:, t], hv)
        ha = np.where(mac, a_mac, ha)
        hm = hm | mac
    return adv


# =========================================================================== learner-side batch


@dataclass
class RolloutBatch:
    """Flattened steps of a set of segments plus advantages (segment order)."""

    samples: list[Any]
    actions: NDArray[np.int64]
    logp: F32
    values: F32
    advantages: F32
    returns: F32
    kinds: NDArray[np.int8]
    n_legal: NDArray[np.int16]
    in_combat: NDArray[np.bool_]
    rewards: F32
    win: F32
    win_mask: NDArray[np.bool_]
    floors_left: F32
    hp_post: F32
    hp_mask: NDArray[np.bool_]
    staleness: NDArray[np.int64]  # learner version - acting version, per step
    characters: NDArray[np.int32]  # (steps, 4), canonical lineup's character ids; absent seats = 0
    ascension: NDArray[np.int16]
    n_seats: NDArray[np.int8]
    episode_uid: NDArray[np.int64]
    env_id: NDArray[np.int32]
    segment_staleness: NDArray[np.int64]  # learner version - oldest acting version, per segment
    advice_row: NDArray[np.int16] | None = None  # Segment.advice_row / advice_hp (None: the segments carry none)
    advice_hp: F32 | None = None
    advice_rule: NDArray[np.int8] | None = None  # Segment.advice_rule / advice_sign (None: the segments carry none)
    advice_sign: NDArray[np.int8] | None = None
    t_keep: F32 | None = None  # Segment.t_keep / t_add (None: the segments carry none)
    t_add: F32 | None = None
    beh_temp: F32 | None = None  # Segment.beh_temp (None: the segments carry none)
    # critic segments (notes/critic/contract.md 4.2; None otherwise): the tower's values and CRIT per step, the
    # component rewards, and the float64 advantages A_k / returns RET_k of the components (columns 1-3 of the
    # 4-column GAE; ``advantages`` / ``returns`` are column 0, the public stream, bit for bit today's)
    values_c: F32 | None = None
    crit: NDArray[np.int16] | None = None
    rewards_c: F32 | None = None
    adv_c: F64 | None = None
    ret_c: F64 | None = None
    # p12: per-step combat_id, practice fork_group (0 = plain / off) and step offset from the fork door (-1 if fg == 0)
    combat_id: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, np.int32))
    fork_group: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
    fork_step: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))

    def __len__(self) -> int:
        return len(self.actions)


def critic_streams(segments: Sequence[Segment]) -> tuple[F64, F64, F64]:
    """The 4 GAE streams of critic segments (notes/critic/contract.md 4.2): values (steps, 4), bootstrap (segments,
    4), rewards (steps, 4); column 0 = (stored public values, public bootstrap, r), columns 1-3 = (values_c[:, k],
    bootstrap_c[:, k], R_k)."""
    assert all(s.values_c is not None and s.rewards_c is not None and s.bootstrap_c is not None for s in segments), \
        "critic streams need segments with the critic record"
    values = np.concatenate([np.concatenate([s.values[:, None], s.values_c], 1) for s in segments]).astype(np.float64)
    rewards = np.concatenate([np.concatenate([s.rewards[:, None], s.rewards_c], 1)  # type: ignore[list-item]
                              for s in segments]).astype(np.float64)
    boot = np.asarray([[s.bootstrap_value, *s.bootstrap_c.tolist()] for s in segments], np.float64)  # type: ignore[union-attr]
    return values, boot, rewards


def _cat(segments: Sequence[Segment], name: str) -> NDArray[Any]:
    return np.concatenate([getattr(s, name) for s in segments])


def build_batch(segments: Sequence[Segment], gae_cfg: GAEConfig, current_version: int,
                values: Floats | None = None, bootstrap: Floats | None = None, *,
                with_samples: bool = True, rewards: Floats | None = None) -> RolloutBatch:
    """Flatten ``segments`` and compute GAE. ``with_samples=False`` leaves ``samples`` empty (the learner's
    fast path reads the segments' sample blocks directly)."""
    if not segments:
        raise ValueError("build_batch needs at least one segment")
    critic = segments[0].values_c is not None
    assert all((s.values_c is not None) == critic for s in segments), "segments with and without the critic record"
    crit_cols: dict[str, Any] = {}
    if critic:  # one 4-column GAE (notes/critic/contract.md 4.2); column 0 is today's public stream bit for bit
        assert values is None and bootstrap is None and rewards is None, \
            "critic segments: no value recomputation or reward override (contract 3)"
        a4, v4 = gae_columns(segments, gae_cfg, *critic_streams(segments))
        adv, ret = a4[:, 0].astype(np.float32), (a4[:, 0] + v4[:, 0]).astype(np.float32)
        crit_cols = dict(values_c=_cat(segments, "values_c"), crit=_cat(segments, "crit"),
                         rewards_c=_cat(segments, "rewards_c"), adv_c=a4[:, 1:], ret_c=a4[:, 1:] + v4[:, 1:])
    else:
        adv, ret = gae(segments, gae_cfg, values, bootstrap, rewards)
    lens = np.fromiter((len(s) for s in segments), np.int64, len(segments))

    def per_seg(fn: Callable[[Segment], int], dtype: type) -> NDArray[Any]:
        return np.repeat(np.fromiter((fn(s) for s in segments), dtype, len(segments)), lens)

    versions = _cat(segments, "policy_version")
    combat_id = _cat(segments, "combat_id")
    fork_group = per_seg(lambda s: s.meta.fork_group, np.int64)
    if (fork_group != 0).any():
        fsteps = []
        for s in segments:
            if s.meta.fork_group:
                assert s.start >= s.meta.fork_start, (s.start, s.meta.fork_start)
                fsteps.append(np.arange(s.start - s.meta.fork_start, s.start - s.meta.fork_start + len(s), dtype=np.int64))
            else:
                fsteps.append(np.full(len(s), -1, np.int64))
        fork_step = np.concatenate(fsteps)
    else:
        fork_step = np.full(int(lens.sum()), -1, np.int64)
    advice = segments[0].advice_row is not None
    assert all((s.advice_row is not None) == advice for s in segments), "segments with and without advice records"
    judged = segments[0].advice_rule is not None
    assert all((s.advice_rule is not None) == judged for s in segments), "segments with and without habit-rule records"
    taste = segments[0].t_keep is not None
    assert all((s.t_keep is not None) == taste for s in segments), "segments with and without taste records"
    temp = segments[0].beh_temp is not None
    assert all((s.beh_temp is not None) == temp for s in segments), "segments with and without beh_temp records"
    return RolloutBatch(
        samples=[x for s in segments for x in s.samples] if with_samples else [],
        actions=_cat(segments, "actions"), logp=_cat(segments, "logp"),
        values=values.astype(np.float32) if values is not None else _cat(segments, "values"),
        advantages=adv, returns=ret,
        kinds=_cat(segments, "kinds"), n_legal=_cat(segments, "n_legal"),
        in_combat=combat_id >= 0,
        rewards=np.asarray(rewards, np.float32) if rewards is not None else _cat(segments, "rewards"),
        win=_cat(segments, "win"), win_mask=_cat(segments, "win_mask"),
        floors_left=_cat(segments, "floors_left"),
        hp_post=_cat(segments, "hp_post"), hp_mask=_cat(segments, "hp_mask"),
        staleness=current_version - versions,
        characters=np.repeat(np.asarray([(*s.meta.characters, *((0,) * (4 - len(s.meta.characters))))
                                         for s in segments], np.int32), lens, axis=0),
        ascension=per_seg(lambda s: s.meta.ascension, np.int16),
        n_seats=per_seg(lambda s: len(s.meta.characters), np.int8),
        episode_uid=per_seg(lambda s: s.meta.episode_uid, np.int64),
        env_id=per_seg(lambda s: s.meta.env_id, np.int32),
        segment_staleness=current_version - np.fromiter((s.version_min for s in segments), np.int64, len(segments)),
        advice_row=_cat(segments, "advice_row") if advice else None,
        advice_hp=_cat(segments, "advice_hp") if advice else None,
        advice_rule=_cat(segments, "advice_rule") if judged else None,
        advice_sign=_cat(segments, "advice_sign") if judged else None,
        t_keep=_cat(segments, "t_keep") if taste else None,
        t_add=_cat(segments, "t_add") if taste else None,
        beh_temp=_cat(segments, "beh_temp") if temp else None,
        combat_id=combat_id,
        fork_group=fork_group,
        fork_step=fork_step,
        **crit_cols,
    )


def apply_rloo(rb: RolloutBatch, gae_cfg: GAEConfig, alpha: float) -> tuple[int, int, float]:
    """Sibling-centered Leave-One-Out (RLOO) door-baseline shift for practice children in ``rb`` (p12).

    For every ``fork_group > 0`` with ``m >= 2`` distinct child episodes having a door step (``fork_step == 0``)
    in this batch:
    * child ``i``'s door return (float64) is ``G_i = A_{i,0} + V_{i,0}``;
    * leave-one-out sibling mean is ``G_{-i} = (sum_j G_j - G_i) / (m - 1)``;
    * door-baseline residual shift is ``delta_i = alpha * (V_{i,0} - G_{-i})``;
    * every step ``t = fork_step >= 0`` of child ``i``'s forked fight (same ``combat_id`` as its door step)
      gets ``adv[idx] += delta_i * (gamma * lam_eff) ** t``.

    Shifts ONLY ``rb.advantages`` in place (never ``rb.returns``: value head targets stay unbiased).
    Returns ``(rloo_groups, rloo_children, rloo_shift_std)``.
    """
    assert np.isfinite(alpha) and 0.0 <= alpha <= 1.0, f"rloo_alpha {alpha}: 0 <= alpha <= 1"
    if alpha == 0.0 or not len(rb):
        return 0, 0, 0.0
    fg = rb.fork_group
    if not (fg != 0).any():
        return 0, 0, 0.0
    door_idx = np.flatnonzero((fg != 0) & (rb.fork_step == 0))
    if not len(door_idx):
        return 0, 0, 0.0
    by_group: dict[int, list[int]] = {}
    for p in door_idx.tolist():
        by_group.setdefault(int(fg[p]), []).append(p)
    adv64 = rb.advantages.astype(np.float64)
    val64 = rb.values.astype(np.float64)
    child_shift: dict[tuple[int, int, int], tuple[float, int, float]] = {}
    shifts: list[float] = []
    n_groups = 0
    lam_base = gae_cfg.lam_micro if gae_cfg.macro_micro else gae_cfg.lam
    for g, ps in by_group.items():
        keys = [(g, int(rb.env_id[p]), int(rb.episode_uid[p])) for p in ps]
        assert len(set(keys)) == len(keys), f"RLOO: duplicate door steps in fork_group {g}"
        m = len(ps)
        if m < 2:
            continue
        n_groups += 1
        G = np.array([adv64[p] + val64[p] for p in ps], dtype=np.float64)
        G_sum = float(G.sum())
        for k, p in enumerate(ps):
            g_loo = (G_sum - float(G[k])) / (m - 1)
            delta = alpha * (float(val64[p]) - g_loo)
            shifts.append(delta)
            lam_eff = seat_trace_decay(lam_base, int(rb.n_seats[p]))
            child_shift[keys[k]] = (delta, int(rb.combat_id[p]), gae_cfg.gamma * lam_eff)
    if not child_shift:
        return 0, 0, 0.0
    nz = np.flatnonzero(fg != 0)
    fg_z, ev_z, ep_z = fg[nz], rb.env_id[nz], rb.episode_uid[nz]
    brk = np.empty(len(nz), bool)
    brk[0] = True
    brk[1:] = (fg_z[1:] != fg_z[:-1]) | (ev_z[1:] != ev_z[:-1]) | (ep_z[1:] != ep_z[:-1])
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:], len(nz))
    for s0, s1 in zip(starts.tolist(), ends.tolist()):
        p0 = int(nz[s0])
        info = child_shift.get((int(fg[p0]), int(rb.env_id[p0]), int(rb.episode_uid[p0])))
        if info is None:
            continue
        delta, cid0, decay = info
        sl = nz[s0:s1]
        m_fight = rb.combat_id[sl] == cid0
        if m_fight.any():
            idx = sl[m_fight]
            t = rb.fork_step[idx]
            assert (t >= 0).all()
            adv64[idx] += delta * (decay ** t.astype(np.float64))
    rb.advantages = adv64.astype(np.float32)
    return n_groups, len(shifts), float(np.std(shifts))


# =========================================================================== segment ring


@dataclass
class RingStats:
    put_segments: int = 0
    put_samples: int = 0
    taken_samples: int = 0
    evicted_samples: int = 0  # oldest segments dropped because the ring was full
    stale_dropped_samples: int = 0  # dropped by take(max_staleness=...)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class SegmentRing:
    """Thread-safe FIFO of completed segments bounded by a sample capacity.

    Actor threads (or a queue-draining thread) :meth:`put`; the learner :meth:`take` s an
    iteration of at least ``min_samples`` samples, oldest first.
    """

    def __init__(self, capacity_samples: int = 1 << 20) -> None:
        self.capacity = capacity_samples
        self.stats = RingStats()
        self._q: deque[Segment] = deque()
        self._n = 0
        self._cv = threading.Condition()

    def __len__(self) -> int:
        return len(self._q)

    @property
    def n_samples(self) -> int:
        return self._n

    def put(self, seg: Segment) -> None:
        with self._cv:
            self._q.append(seg)
            self._n += len(seg)
            self.stats.put_segments += 1
            self.stats.put_samples += len(seg)
            while self._n > self.capacity and len(self._q) > 1:
                old = self._q.popleft()
                self._n -= len(old)
                self.stats.evicted_samples += len(old)
            self._cv.notify_all()

    def take(self, min_samples: int, *, current_version: int | None = None, max_staleness: int | None = None,
             timeout: float | None = None) -> list[Segment] | None:
        """Pop the oldest segments until they hold >= ``min_samples`` samples.

        Segments whose oldest step is more than ``max_staleness`` versions behind
        ``current_version`` are discarded (counted). Blocks up to ``timeout`` seconds
        (``None`` = forever); returns ``None`` on timeout, leaving the ring untouched.
        """
        if max_staleness is not None and current_version is None:
            raise ValueError("max_staleness needs current_version")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while True:
                if max_staleness is not None:
                    assert current_version is not None
                    self._evict_stale(current_version - max_staleness)
                if self._n >= min_samples:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cv.wait(remaining)
            out: list[Segment] = []
            got = 0
            while got < min_samples:
                seg = self._q.popleft()
                out.append(seg)
                got += len(seg)
            self._n -= got
            self.stats.taken_samples += got
            return out

    def _evict_stale(self, min_version: int) -> None:
        keep = deque(s for s in self._q if s.version_min >= min_version)
        dropped = self._n - sum(len(s) for s in keep)
        if dropped:
            self._q = keep
            self._n -= dropped
            self.stats.stale_dropped_samples += dropped
