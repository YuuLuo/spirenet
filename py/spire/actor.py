"""Actor / inference loop: E hosts, batched GPU inference, rollout recording (py-stack-spec 1).

Data flow of one decision::

    host reader thread --(env, generation, payload)--> SimpleQueue --> Actor._pump: Host.deliver
      -> ready list --> Actor._launch: frame_facts, Tokenizer.encode, group by token bucket,
         Policy.launch per group (collate, CUDA-graph replay, async device->host copy)
      -> Actor._finish: Policy.finish (wait, sample on the host), RolloutStore.add_step (train),
         Host.send_step (pipe write)

* Every host is :meth:`~spire.host.Host.route`-d into one queue; the reader threads of
  :class:`~spire.host.HostProcess` block on the pipes, the loop never does. Items of replaced
  processes (stale ``generation``) are ignored.
* Batching: after draining the queue the loop launches inference when ``max_batch`` frames are
  ready, when no host has a reply outstanding, or ``max_wait_ms`` after the oldest ready frame.
  Ready frames are grouped by padded token bucket (small groups merge upwards). Up to
  ``max_inflight`` launched batches overlap with receiving / encoding further frames; the
  oldest is finished as soon as the device is done or there is nothing else to do.
* Robustness: a host timeout / crash / broken framing (host restarted by the wrapper), a
  protocol violation (host restarted here), an ERROR frame (reproduction record, then RESET
  once the host re-sent its frame), RESULT=ABORTED and overflow frames all end the episode:
  its unreleased steps are dropped from training (RolloutStore counts them) and an
  :class:`~spire.metrics.EpisodeRecord` with the failure reason is emitted - nothing is
  silently dropped from the statistics. A host whose restart budget is exhausted is retired.
* Pile cap (``ActorConfig.pile_cap``, a backstop for fights that pile up thousands of generated cards, whose
  reshuffles outlast the step timeout): a decision frame over the cap ends the episode like a decision cap.
* ``train`` mode samples from the policy and records transitions; ``eval`` mode plays a finite
  list of jobs (greedy or with a temperature) and only emits episode records.
* Practice fights (``practice``, :mod:`spire.gym`; train mode, off by default): at the first decision of a fight
  of a parent game the host forks ``k`` RESAMPLE copies (k from the measured difficulty); each copy is one more env
  (a child slot after the hosts' envs) from its first frame, the parent's pending frame, to the end of that fight
  (FIGHT_END + one frame: TERMINAL, or the post-fight OBS whose value bootstraps the child's truncated episode).
  Children are launched in inference groups of their own and sampled from the gym's generator, so the full games
  see the batches and the sampling stream they would see without practice; their episodes carry
  ``EpisodeRecord.gym`` (the encounter key) and are counted apart (``report["practice"]``). With scouts (v1.1)
  the parents play combat greedily and the rest without exploration, from their own generator, untrained.
  With ``practice.game_classes`` (v1.2) the children of a listed fight class are GAME children: ordinary trained
  episodes from the fork to the run's TERMINAL (no FIGHT_END, no final-frame evaluation).
* Time per decision is accounted by stage (:data:`STAGES`); ``wait`` is time blocked on hosts.
"""
from __future__ import annotations

import json
import os
import queue
import secrets
import threading
import time
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, cast

import numpy as np
import torch

from .advice import Advisor
from .distributions import choice_uniform_probs_numpy, exploration_table, sample_choice_keys
from .taste import Taste
from .temperature import combat_steps, step_temperatures
from .cycle import CycleBan
from .removal import CLASSES as REMOVAL_CLASSES, load_rule
from .rules import free_legal, mask_legal
from .gym import (
    MODE_EXACT,
    MODE_RESAMPLE,
    MSG_FIGHT_END,
    MSG_FORKED,
    GAME_FLAGS,
    TOP_ASCENSIONS,
    Arrival,
    ChildHost,
    Controller,
    Difficulty,
    FightEnd,
    GymPool,
    Hello,
    PracticeConfig,
    PracticeCounters,
    SeedStream,
    encode_fork,
    encounter_class,
    fork_group_id,
    key_class,
    msg_type,
    parse_fight_end,
    parse_forked,
    same_obs,
)
from .host import EpisodeStats, Host, HostError, HostProcess, HostStateError, HostUnrecoverable, Pending
from .history import PublicHistory
from .critic import crit_width, empty_crit, side_batch, truncated
from .learner import FileWeightSource, WeightSource
from .metrics import LONG_EPISODE_DECISIONS, ActionStats, EpisodeRecord, StageClock, canonical_lineup, percentiles
from .model import GraphRunner, PinnedState, SpireNet, import_weights, packed_layout, packed_views
from .pipes import PipeMux, PosixPipeMux
from .proto import Frame, ProtocolError, ResetCmd, parse_frame
from .rewards import Facts, frame_facts
from .rollout import EpisodeMeta, Outcome, RolloutStore
from .schema import (
    RESET_FLAGS_DEFAULT,
    ActionCol,
    CardCol,
    CreatureCol,
    CreatureKind,
    FrameFlag,
    G,
    Kind,
    MsgType,
    ResetFlag,
    Result,
    SchemaError,
    SeatCol,
    Table,
    TerminalReason,
    Verb,
    Zone,
)
from .tokenize import CYCLE_BAN_SLOTS, BatchShape, EncodedBatch, Ent, Sample, Tokenizer, batch_to
from .vocab import Catalog, Vocab

STAGES = ("wait", "recv", "facts", "encode", "collate", "infer", "gpu", "record", "advice", "send", "idle", "other")
"""Actor time accounting: ``wait`` = blocked on hosts; ``gpu`` = blocked waiting for the device
(forward + readback still running); ``infer`` = the CPU side of inference (graph replay launch,
host->device copies enqueued, host-side sampling); ``advice`` = the advice records of the recorded steps
(:class:`spire.advice.Advisor`; 0 without advice rules); ``idle`` = spin-polling loop iterations that found
nothing to do (no frame, no launch, no finished group: hosts or the batching window are the limit - CPU
time, but no work); the rest is Python-side work."""
_WAIT, _RECV, _FACTS, _ENCODE, _COLLATE, _INFER, _GPU, _RECORD, _ADVICE, _SEND, _IDLE = range(len(STAGES) - 1)
PYTHON_STAGES = ("recv", "facts", "encode", "collate", "infer", "record", "advice", "send", "other")
_VERB = int(ActionCol.VERB)  # plain int index (IntEnum indices go through __index__ per element)
_INF = float("inf")
_MSG_OBS, _MSG_ERROR = int(MsgType.OBS), int(MsgType.ERROR)
_F_OVERFLOW = int(FrameFlag.OVERFLOW)
_PENDING_NONE = int(Pending.NONE)
_BATCH_KEYS = ("types", "raw", "rel", "dest", "edges", "tok_mask", "act_mask", "kid_mask", "var_mask", "cond", "history")

# =========================================================================== seeds

SEED_ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # SeedHelper._characters (no I, no O)
SEED_LEN = 12
_SEED_INDEX = {c: i for i, c in enumerate(SEED_ALPHABET)}


def canonical_seed(seed: str) -> str:
    """``SeedHelper.CanonicalizeSeed``: upper case, O -> 0, I -> 1, trimmed."""
    return seed.upper().replace("O", "0").replace("I", "1").strip()


def random_seed(rng: np.random.Generator | None = None) -> str:
    """A 12-character game seed; ``rng=None`` draws from the OS CSPRNG."""
    if rng is None:
        return "".join(secrets.choice(SEED_ALPHABET) for _ in range(SEED_LEN))
    return "".join(SEED_ALPHABET[i] for i in rng.integers(0, len(SEED_ALPHABET), SEED_LEN))


def seed_to_wire(seed: str) -> int:
    """RESET seed word: the canonical 12-character seed read as a base-34 number (< 2**62)."""
    s = canonical_seed(seed)
    if len(s) != SEED_LEN or any(c not in _SEED_INDEX for c in s):
        raise ValueError(f"not a {SEED_LEN}-character game seed: {seed!r}")
    x = 0
    for c in s:
        x = x * len(SEED_ALPHABET) + _SEED_INDEX[c]
    return x


def wire_to_seed(x: int) -> str:
    """Inverse of :func:`seed_to_wire`."""
    if not 0 <= x < len(SEED_ALPHABET) ** SEED_LEN:
        raise ValueError(f"wire seed out of range: {x}")
    out = []
    for _ in range(SEED_LEN):
        x, r = divmod(x, len(SEED_ALPHABET))
        out.append(SEED_ALPHABET[r])
    return "".join(reversed(out))


# =========================================================================== episode configuration


@dataclass(frozen=True)
class EpisodeJob:
    """One episode to play: lineup (character vocab id per seat), ascension and seed."""

    characters: tuple[int, ...]
    lineup: str  # sorted character names joined by "+" in canonical seat order
    ascension: int = 0
    seed: str | None = None  # 12-character game seed; None = the host draws one
    seed_id: str = ""  # panel entry id (eval) / free-form tag

    def __post_init__(self) -> None:
        assert self.lineup == canonical_lineup(self.lineup), "EpisodeJob needs canonical seat order"
        assert len(self.characters) == len(self.lineup.split("+"))

    def reset_cmd(self, flags: int, max_decisions: int = 0) -> ResetCmd:
        """RESET with a per-seat cap: ``max_decisions * n_seats`` (0 = host default)."""
        assert max_decisions >= 0
        return ResetCmd(self.characters, self.ascension, None if self.seed is None else seed_to_wire(self.seed),
                        flags=flags, max_decisions=max_decisions * len(self.characters))


class EpisodeSource(Protocol):
    def next_job(self, env_id: int) -> EpisodeJob | None:
        """The next episode for ``env_id``, or None when there is none (env goes idle)."""
        ...

    @property
    def exhausted(self) -> bool: ...


def parse_lineup(lineup: str, vocab: Vocab) -> tuple[int, ...]:
    """Character ids in sorted character-name order (duplicate characters allowed)."""
    names = [n.strip() for n in lineup.split("+")]
    if not 1 <= len(names) <= 4 or not all(names):
        raise ValueError(f"lineup needs 1-4 characters: {lineup!r}")
    ids = tuple(vocab.id("character", n) for n in sorted(names))
    if any(i <= 1 for i in ids):
        raise ValueError(f"unknown character in lineup {lineup!r}")
    return ids


@dataclass(frozen=True)
class EpisodeDist:
    """Training episode distribution: weighted lineups x weighted ascensions, seed source.

    ``seeds``: ``"rng"`` - reproducible 12-character seeds from ``seed`` (per actor stream);
    ``"csprng"`` - fresh OS-entropy seeds; ``"host"`` - no seed sent, the host draws one.
    ``lineup_ascensions``: ascension weights of single lineups (an ascension curriculum for the lineups that
    are ready for it); a lineup listed here ignores ``ascensions``.
    """

    lineups: tuple[tuple[str, float], ...] = (("IRONCLAD", 0.7), ("SILENT", 0.3))
    ascensions: tuple[tuple[int, float], ...] = ((0, 1.0),)
    seeds: Literal["rng", "csprng", "host"] = "rng"
    lineup_ascensions: tuple[tuple[str, tuple[tuple[int, float], ...]], ...] = ()

    def __post_init__(self) -> None:
        lineups = tuple((canonical_lineup(name), weight) for name, weight in self.lineups)
        if len({name for name, _ in lineups}) != len(lineups):
            raise ValueError("lineups: duplicate canonical lineups")
        object.__setattr__(self, "lineups", lineups)
        own = tuple((canonical_lineup(name), tuple(ascs)) for name, ascs in self.lineup_ascensions)
        if len({name for name, _ in own}) != len(own):
            raise ValueError("lineup_ascensions: duplicate canonical lineups")
        if missing := {name for name, _ in own} - {name for name, _ in lineups}:
            raise ValueError(f"lineup_ascensions: {sorted(missing)} not in lineups")
        object.__setattr__(self, "lineup_ascensions", own)


MAX_ASCENSION = 20


def validate_distribution(data: Mapping[str, Any], vocab: Vocab, base: EpisodeDist | None = None) -> EpisodeDist:
    """An :class:`EpisodeDist` from plain data ``{"lineups": [[name, weight], ...], "ascensions": [[asc,
    weight], ...], "lineup_ascensions": [[name, [[asc, weight], ...]], ...]}`` (the sampling-weights file;
    missing keys keep ``base``'s values, the seed source is never changed here). Raises ``ValueError`` with a
    readable reason for anything invalid: unknown keys, an empty list, an unknown character, a lineup of more
    than 4 seats, a negative / non-finite weight, all weights zero, an ascension outside 0..20, duplicates, an
    ascension list for a lineup that is not in ``lineups``."""
    if not isinstance(data, Mapping):
        raise ValueError("expected a JSON object")
    unknown = set(data) - {"lineups", "ascensions", "lineup_ascensions", "comment", "version"}
    if unknown:
        raise ValueError(f"unknown keys {sorted(unknown)} (allowed: lineups, ascensions, lineup_ascensions, comment)")
    base = base or EpisodeDist()

    def pairs(key: str, default: tuple[tuple[Any, float], ...], raw: Any = None) -> list[tuple[Any, float]]:
        raw = data.get(key) if raw is None else raw
        if raw is None:
            return list(default)
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError(f"{key}: expected a non-empty list of [value, weight] pairs")
        out = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"{key}: {item!r} is not a [value, weight] pair")
            w = item[1]
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not np.isfinite(w) or w < 0:
                raise ValueError(f"{key}: weight {w!r} of {item[0]!r} must be a finite number >= 0")
            out.append((item[0], float(w)))
        if sum(w for _, w in out) <= 0:
            raise ValueError(f"{key}: all weights are zero")
        values = [v for v, _ in out]
        if len(set(map(str, values))) != len(values):
            raise ValueError(f"{key}: duplicate entries")
        return out

    lineups = pairs("lineups", base.lineups)
    for name, _ in lineups:
        if not isinstance(name, str):
            raise ValueError(f"lineups: {name!r} is not a lineup string like \"IRONCLAD\" or \"IRONCLAD+SILENT\"")
        parse_lineup(name, vocab)  # ValueError for unknown characters / seat counts
    lineups = [(canonical_lineup(name), weight) for name, weight in lineups]
    if len({name for name, _ in lineups}) != len(lineups):
        raise ValueError("lineups: duplicate canonical lineups")

    def ascension_pairs(key: str, default: tuple[tuple[int, float], ...], raw: Any = None
                        ) -> tuple[tuple[int, float], ...]:
        ascs = pairs(key, default, raw)
        for a, _ in ascs:
            if isinstance(a, bool) or not isinstance(a, int) or not 0 <= a <= MAX_ASCENSION:
                raise ValueError(f"{key}: {a!r} is not an ascension 0..{MAX_ASCENSION}")
        return tuple((int(a), w) for a, w in ascs)

    own = data.get("lineup_ascensions")
    if own is None:
        lineup_ascensions = base.lineup_ascensions
    else:
        if not isinstance(own, (list, tuple)) or any(
                not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[0], str) for item in own):
            raise ValueError("lineup_ascensions: expected a list of [lineup, [[ascension, weight], ...]] entries")
        for name, _ in own:
            parse_lineup(name, vocab)
        lineup_ascensions = tuple((canonical_lineup(name), ascension_pairs(f"lineup_ascensions[{name}]", (), raw or []))
                                  for name, raw in own)
    # the dataclass refuses duplicate lineups and ascension lists of lineups that are not sampled (ValueError)
    return EpisodeDist(lineups=tuple((str(n), w) for n, w in lineups),
                       ascensions=ascension_pairs("ascensions", base.ascensions), seeds=base.seeds,
                       lineup_ascensions=lineup_ascensions)


class EpisodeSampler:
    """Infinite :class:`EpisodeSource` drawing from an :class:`EpisodeDist`; seeds in
    ``exclude`` (e.g. every evaluation panel seed) are never used for training.

    ``weights_file``: a JSON file ``{"version": n, "lineups": [...], "ascensions": [...]}`` (the supervisor's
    validated copy of the run's sampling-weights file) re-read when its modification time changes, at most
    every ``poll_s`` seconds; the distribution then changes for every following episode. A file that fails
    :func:`validate_distribution` is ignored (the supervisor already logged why)."""

    def __init__(self, dist: EpisodeDist, vocab: Vocab, seed: int | Sequence[int] = 0,
                 exclude: frozenset[str] = frozenset(), *, weights_file: str | Path | None = None,
                 poll_s: float = 10.0) -> None:
        if not dist.lineups or not dist.ascensions:
            raise ValueError("empty lineup / ascension distribution")
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)
        self.exclude = frozenset(canonical_seed(s) for s in exclude)
        self.weights_file = Path(weights_file) if weights_file else None
        self.poll_s = poll_s
        self.weights_version = 0
        self._mtime = -1.0
        self._t_poll = 0.0
        self._apply(dist)
        if self.weights_file is not None:
            self.reload()

    def _apply(self, dist: EpisodeDist) -> None:
        dist = validate_distribution({}, self.vocab, dist)
        self.dist = dist
        self.lineups = [(name, parse_lineup(name, self.vocab)) for name, _ in dist.lineups]
        self.p_lineup = _normalise([w for _, w in dist.lineups])
        self.ascensions = [int(a) for a, _ in dist.ascensions]
        self.p_asc = _normalise([w for _, w in dist.ascensions])
        own = {name: ([int(a) for a, _ in ascs], _normalise([w for _, w in ascs]))
               for name, ascs in dist.lineup_ascensions}
        self.asc_by_lineup = [own.get(name, (self.ascensions, self.p_asc)) for name, _ in dist.lineups]

    def reload(self) -> bool:
        """Re-read ``weights_file`` if it changed; True if a new distribution was adopted."""
        self._t_poll = time.monotonic()
        if self.weights_file is None:
            return False
        try:
            mtime = self.weights_file.stat().st_mtime
            if mtime == self._mtime:
                return False
            data = json.loads(self.weights_file.read_text(encoding="utf-8"))
            dist = validate_distribution({k: data[k] for k in ("lineups", "ascensions", "lineup_ascensions")
                                          if k in data}, self.vocab, self.dist)
        except (OSError, ValueError):  # missing, half-written (retried next poll) or invalid
            return False
        self._mtime = mtime
        self.weights_version = int(data.get("version", self.weights_version + 1))
        self._apply(dist)
        return True

    @property
    def exhausted(self) -> bool:
        return False

    def next_job(self, env_id: int) -> EpisodeJob:
        if self.weights_file is not None and time.monotonic() - self._t_poll >= self.poll_s:
            self.reload()
        i = int(self.rng.choice(len(self.lineups), p=self.p_lineup))
        name, chars = self.lineups[i]
        ascensions, p_asc = self.asc_by_lineup[i]
        asc = ascensions[int(self.rng.choice(len(ascensions), p=p_asc))]
        seed: str | None = None
        if self.dist.seeds != "host":
            while True:
                seed = random_seed(self.rng if self.dist.seeds == "rng" else None)
                if seed not in self.exclude:
                    break
        return EpisodeJob(chars, name, asc, seed)


def _normalise(w: Sequence[float]) -> np.ndarray:
    a = np.asarray(w, np.float64)
    if (a < 0).any() or a.sum() <= 0:
        raise ValueError(f"bad weights {list(w)}")
    return a / a.sum()


class JobList:
    """Finite :class:`EpisodeSource`: every job is handed out exactly once (evaluation panels)."""

    def __init__(self, jobs: Sequence[EpisodeJob]) -> None:
        self.jobs = list(jobs)
        self._next = 0
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        return self._next >= len(self.jobs)

    def next_job(self, env_id: int) -> EpisodeJob | None:
        with self._lock:
            if self._next >= len(self.jobs):
                return None
            job = self.jobs[self._next]
            self._next += 1
            return job


# =========================================================================== policies


@dataclass
class PolicyOut:
    actions: np.ndarray  # (n,) int64 ACTIONS row per frame
    logp: np.ndarray  # (n,) float32 behaviour log-prob of the chosen action
    value: np.ndarray  # (n,) float32
    entropy: np.ndarray | None  # (n,) float32 behaviour entropy
    collate_ns: int = 0
    gpu_ns: int = 0  # launch + waiting for the device + host-side sampling
    wait_ns: int = 0  # part of gpu_ns spent blocked on the device
    logits: np.ndarray | None = None  # (n, A) the scores the actions were chosen from (-inf: illegal / banned)
    logp_all: np.ndarray | None = None  # (n, A) behaviour log-probs of every row (0 at illegal rows; spire.taste)
    beh_temp: np.ndarray | None = None  # (n,) float32 behaviour temperature T(step) of each row (spire.temperature)
    combat: np.ndarray | None = None  # (n,) the policy's combat-step flags (spire.temperature)
    values_c: np.ndarray | None = None  # (n, 3) float32 the critic tower's (V_WIN, V_PROG, V_SURV) (a critic policy)
    pwin_logit: np.ndarray | None = None  # (n,) float32 (when TorchPolicy(include_pwin=True))


class Policy(Protocol):
    """Two-phase inference so the actor can overlap device work with host work:
    :meth:`launch` starts a batch and returns a handle, :meth:`done` polls it, :meth:`finish`
    waits for it and samples. ``rng`` (passed only for practice children / scouts): the generator this batch
    is sampled from instead of the policy's own. ``greedy_rows`` (scouts only): no exploration mixture, arg-max
    on the rows marked True, the base softmax elsewhere."""

    version: int
    needs_samples: bool

    def launch(self, samples: Sequence[Sample], n_actions: np.ndarray, rng: np.random.Generator | None = None,
               greedy_rows: np.ndarray | None = None) -> Any: ...

    def done(self, handle: Any) -> bool: ...

    def finish(self, handle: Any) -> PolicyOut: ...

    def refresh(self) -> bool:
        """Adopt newer published weights if any (between batches); True if it did."""
        ...


class UniformPolicy:
    """Uniformly random legal action (baselines, actor overhead benchmarks); no model. ``rules_mode`` (spire.rules):
    "mask" / "teach" decode the ban slots as masks with the removal rule's one row, "free" without it."""

    needs_samples = False

    def __init__(self, seed: int = 0, *, rules_mode: str = "mask") -> None:
        self.rng = np.random.default_rng(seed)
        self.version = 0
        self.rules_mode = rules_mode

    def launch(self, samples: Sequence[Sample], n_actions: np.ndarray, rng: np.random.Generator | None = None,
               greedy_rows: np.ndarray | None = None) -> PolicyOut:
        assert greedy_rows is None, "scouts need the policy's logits"
        rng = self.rng if rng is None else rng
        if samples and any(s.history[-CYCLE_BAN_SLOTS:].any() for s in samples):
            decode = free_legal if self.rules_mode == "free" else mask_legal
            legal = [decode(s.history[-CYCLE_BAN_SLOTS:], s.n_actions) for s in samples]
            n = np.asarray([len(row) for row in legal], np.int64)
            assert (n > 0).all()
            chosen = (rng.random(len(n)) * n).astype(np.int64)
            a = np.asarray([row[index] for row, index in zip(legal, chosen)], np.int64)
            return PolicyOut(a, -np.log(n).astype(np.float32), np.zeros(len(n), np.float32),
                             np.log(n).astype(np.float32))
        n = np.maximum(n_actions, 1)
        a = np.minimum((rng.random(len(n)) * n).astype(np.int64), n - 1)
        return PolicyOut(a, -np.log(n).astype(np.float32), np.zeros(len(n), np.float32),
                         np.log(n).astype(np.float32))

    def done(self, handle: PolicyOut) -> bool:
        return True

    def finish(self, handle: PolicyOut) -> PolicyOut:
        return handle

    def refresh(self) -> bool:
        return False


@dataclass
class _Launched:
    out: torch.Tensor | np.ndarray  # (n, A + 1): logits | value; pinned host tensor while in flight
    event: torch.cuda.Event | None  # recorded after the device->host copy (None: already on the host)
    shape: BatchShape | None
    staging: dict[str, Any] | None  # pinned buffers of this launch, recycled by finish()
    collate_ns: int
    launch_ns: int
    exploration: np.ndarray | None = None  # per-row mixture epsilon, retained through asynchronous inference
    choice_keys: np.ndarray | None = None
    combat: np.ndarray | None = None  # per-row combat-step flags (spire.temperature)
    rng: np.random.Generator | None = None  # samples this batch instead of the policy's generator (practice)
    greedy_rows: np.ndarray | None = None  # scouts: arg-max on these rows, no exploration mixture (practice v1.1)


_N_VC = 3  # the critic tower's outputs per row, after logits | value (notes/critic/contract.md 3)


class TorchPolicy:
    """:class:`SpireNet` policy.

    CUDA: collate, copy into recycled pinned staging buffers (so the host->device copies are truly
    asynchronous), replay a CUDA graph (one graph per (batch, token bucket), see :meth:`_shape`),
    copy (logits, value) asynchronously into pinned host memory and record an event;
    :meth:`finish` waits for the event and samples on the host (Gumbel-max for
    ``mode="sample"`` with optional temperature, argmax for ``mode="greedy"``) - the logits are a
    few KB, and host sampling saves a dozen kernel launches per batch. Without graphs (CPU,
    tests) the forward runs eagerly under ``inference_mode`` + autocast at launch.

    ``exploration_by_kind`` mixes a configured fraction of legal-uniform probability into
    the policy. Sampling, stored log-probabilities and entropy all describe that mixture.
    Its decision kind comes from the observation's public global token.
    ``exploration_choice_uniform`` instead splits that mass equally between source/verb/seat
    choices, then equally among each choice's legal targets.

    With a :class:`~spire.learner.WeightSource`, a background thread polls and loads new weight
    files every ``poll_interval`` seconds; :meth:`refresh` copies staged weights into the network
    (in place, so captured graphs stay valid) between batches. ``frozen`` (kickstart phase 0: the
    teacher acts): the network's weights are never replaced; the :class:`~spire.learner.FileWeightSource`
    only supplies the learner's published version, which stamps the steps.
    """

    needs_samples = True

    def __init__(self, net: SpireNet, tokenizer: Tokenizer, *, mode: Literal["sample", "greedy"] = "sample",
                 temperature: float = 1.0, use_graphs: bool = True, weights: WeightSource | None = None,
                 poll_interval: float = 0.5, seed: int = 0, version: int = 0, high_priority: bool = False,
                 max_graphs: int = 64, exploration_by_kind: tuple[tuple[int, float], ...] = (),
                 exploration_choice_uniform: bool = False, compile_model: bool = False, frozen: bool = False,
                 combat_temperature: float = 1.0, critic: bool = False, include_pwin: bool = False) -> None:
        """``critic`` (a critic run's training actor, notes/critic/contract.md 3): every launch takes the frames'
        CRIT rows and runs ``SpireNet.forward_critic``; the readback row is logits | value | vc[3] and
        :class:`PolicyOut` carries ``values_c``. The logits are computed before CRIT is read (FP2)."""
        assert not frozen or isinstance(weights, FileWeightSource), "a frozen policy reads versions from weight files"
        assert not critic or (net.critic is not None and not frozen), \
            "a critic policy: a network with model.critic, not the kickstart teacher"
        self.critic = critic
        self.include_pwin = include_pwin
        self.draw_k = net.critic.cfg.draw_k if critic and net.critic is not None else 0
        self._n_out = 1 + (_N_VC if critic else 0) + (1 if include_pwin else 0)  # columns after logits: value (| vc) (| pwin)
        self.frozen = frozen
        if temperature <= 0:
            raise ValueError("temperature must be > 0 (use mode='greedy')")
        assert not exploration_by_kind or temperature == 1.0, "PPO mixture accounting requires temperature=1"
        # spire.temperature: softmax(z / combat_temperature) on combat steps, a separate per-step path
        assert np.isfinite(combat_temperature) and combat_temperature > 0
        assert combat_temperature == 1.0 or (mode == "sample" and temperature == 1.0), \
            "a combat temperature is a sampling-policy behaviour on top of temperature 1"
        self.combat_temperature = float(combat_temperature)
        self.net = net.eval()
        self.tok = tokenizer
        self.device = next(net.parameters()).device
        self.cuda = self.device.type == "cuda"
        self.mode, self.temperature = mode, float(temperature)
        self._exploration = exploration_table(exploration_by_kind) if exploration_by_kind else None
        self._choice_uniform = exploration_choice_uniform
        self._verb_slot = tokenizer.layout.slot(Ent.ACTION, ActionCol.VERB)
        self._kind_slot = tokenizer.layout.slot(Ent.GLOBAL, G.KIND)
        self._in_combat_slot = tokenizer.layout.slot(Ent.GLOBAL, G.IN_COMBAT)
        # compile_model: the graphs replay a torch.compile'd forward (dynamic shapes; GraphRunner warms up eagerly
        # before every capture, so compilation never happens inside one). Weights still load into ``net``. A critic
        # policy compiles its graphs' entry ``forward_critic`` (torch.compile wraps ``forward`` only; tree p7).
        fwd = torch.compile(net, dynamic=True).eval() if compile_model and not critic else net
        critic_fwd = torch.compile(net.forward_critic, dynamic=True) if compile_model and critic else None
        self.runner = (GraphRunner(fwd, tokenizer, max_graphs=max_graphs, critic=critic,  # type: ignore[arg-type]
                                   critic_forward=critic_fwd, include_pwin=include_pwin)
                       if use_graphs and self.cuda else None)
        self._caps: dict[tuple[int, int], BatchShape] = {}
        self._staging: dict[BatchShape, list[dict[str, Any]]] = {}
        self.rng = np.random.default_rng(seed)
        self.stream = torch.cuda.Stream(self.device, priority=-1) if high_priority and self.cuda else None
        self.version = version
        self.weight_loads = 0
        self.weight_load_ns = 0
        self._source = weights
        self._adopted = -1  # newest version adopted from ``weights``
        self._staged: tuple[int, Mapping[str, torch.Tensor] | PinnedState] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if weights is not None:
            self._thread = threading.Thread(target=self._poll_loop, args=(poll_interval,), daemon=True,
                                            name="weight-poll")
            self._thread.start()

    # ------------------------------------------------------------------ weights
    def _poll_loop(self, interval: float) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(interval)

    def _poll_once(self) -> None:
        """Stage weights newer than the adopted / already staged ones. ``have`` is derived from those
        rather than from what was fetched, so weights that a concurrent :meth:`refresh` cleared
        before adopting them (it reads, then clears ``_staged``) are fetched again."""
        assert self._source is not None
        staged = self._staged
        if self.frozen:  # the learner's version only (no weights are read)
            v = cast(FileWeightSource, self._source).latest_version()
            if v > max(self._adopted, -1 if staged is None else staged[0]):
                self._staged = (v, None)  # type: ignore[assignment]
            return
        got = self._source.poll(max(self._adopted, -1 if staged is None else staged[0]))
        if got is not None:
            if self.cuda:  # pinned staging here, off the hot loop: refresh() then queues the copies only
                self._staged = (got[0], PinnedState(got[1]))
            else:
                self._staged = got

    def wait_for_weights(self, timeout: float) -> bool:
        """Block until the first published weights are staged and adopted (train start)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.refresh():
                return True
            time.sleep(0.05)
        return False

    def refresh(self) -> bool:
        staged = self._staged
        if staged is None:
            return False
        self._staged = None
        version, sd = staged
        if version <= self._adopted:
            return False
        if self.frozen:
            self.version = self._adopted = version
            return True
        t = time.perf_counter_ns()
        # on the inference stream: batches already launched there finish with the old weights
        with torch.no_grad(), torch.cuda.stream(self.stream) if self.stream is not None else nullcontext():
            if isinstance(sd, PinnedState):
                sd.load_into(self.net)
            else:
                import_weights(self.net, sd)
        self.version = self._adopted = version
        self.weight_loads += 1
        self.weight_load_ns += time.perf_counter_ns() - t
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------ inference
    def _shape(self, samples: Sequence[Sample]) -> BatchShape:
        """One graph shape per (batch, token bucket) that only ever grows.

        ``GraphRunner.shape_for`` rounds every dimension (actions, kid / var / edge / entity rows)
        to a power of two independently, which yields hundreds of distinct shapes over a run and
        thrashes the graph cache (measured: 563 captures in 35 s). Keeping per (B, T) the
        element-wise maximum of the shapes seen so far converges to one graph per (B, T); the
        superseded graph is dropped.
        """
        assert self.runner is not None
        return self._grow_shape(self.runner.shape_for(samples))

    def _grow_shape(self, need: BatchShape) -> BatchShape:
        assert self.runner is not None
        key = (need.batch, need.tokens)
        cap = self._caps.get(key)
        if cap is None or not cap.fits(need):
            if cap is not None:
                self.runner.graphs.pop(cap, None)
                self._staging.pop(cap, None)
            cap = need if cap is None else BatchShape(*(max(a, b) for a, b in zip(cap.astuple(), need.astuple())))
            self._caps[key] = cap
            self._staging.setdefault(cap, [])
        return cap

    def launch(self, samples: Sequence[Sample], n_actions: np.ndarray, rng: np.random.Generator | None = None,
               greedy_rows: np.ndarray | None = None, crit: np.ndarray | None = None) -> _Launched:
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                return self._launch(samples, rng, greedy_rows, crit)
        return self._launch(samples, rng, greedy_rows, crit)

    def _crit_rows(self, crit: np.ndarray | None, n: int, batch: int) -> np.ndarray | None:
        """The (batch, 2 + K) CRIT input of a launch of ``n`` frames (rows n.. = graph padding); None without the
        critic. ``crit`` is given iff the policy is a critic policy."""
        assert (crit is not None) == self.critic, "CRIT rows are given iff the policy runs the critic tower"
        if crit is None:
            return None
        assert crit.shape == (n, crit_width(self.draw_k)) and crit.dtype == np.int16, crit.shape
        out = empty_crit(batch, self.draw_k)
        out[:n] = crit
        return out

    def _launch(self, samples: Sequence[Sample], rng: np.random.Generator | None = None,
                greedy_rows: np.ndarray | None = None, crit: np.ndarray | None = None) -> _Launched:
        t0 = time.perf_counter_ns()
        n = len(samples)
        exploration = (None if self._exploration is None or greedy_rows is not None else
                       self._exploration[np.fromiter((s.raw[0, self._kind_slot] for s in samples), np.int64, n)])
        combat = combat_steps(np.fromiter((s.raw[0, self._kind_slot] for s in samples), np.int64, n),
                              np.fromiter((s.raw[0, self._in_combat_slot] for s in samples), np.int64, n))
        if self.runner is not None:
            shape = self._shape(samples)
            batch = self.tok.collate(samples, shape=shape)
            crit_b = self._crit_rows(crit, n, shape.batch)
            if crit_b is not None:
                batch["crit"] = torch.from_numpy(crit_b)
            free = self._staging.setdefault(shape, [])
            stage = free.pop() if free else {
                **{k: torch.empty_like(v, pin_memory=True) for k, v in batch.items()},
                "_out": torch.empty((shape.batch, shape.actions + self._n_out), dtype=torch.float32, pin_memory=True)}
            for k, v in batch.items():
                stage[k].copy_(v)
            t1 = time.perf_counter_ns()
            out = self.runner.run({k: stage[k] for k in batch}, shape, n, clone=False)
            host = stage["_out"][:n]
            lv_t = [out["logits"], out["value"][:, None]]
            if self.critic:
                lv_t.append(out["vc"].float())
            if self.include_pwin:
                lv_t.append(out["pwin_logit"].float()[:, None])
            host.copy_(torch.cat(lv_t, 1), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            choices = sample_choice_keys(samples, self._verb_slot, shape.actions) if self._choice_uniform else None
            return _Launched(host, ev, shape, stage, t1 - t0, time.perf_counter_ns() - t1, exploration, choices, combat, rng,
                             greedy_rows)
        batch = self.tok.collate(samples)
        t1 = time.perf_counter_ns()
        lv = self._eager(batch, self._crit_rows(crit, n, n))
        choices = (sample_choice_keys(samples, self._verb_slot, lv.shape[1] - self._n_out) if self._choice_uniform
                   else None)
        return _Launched(lv, None, None, None, t1 - t0, time.perf_counter_ns() - t1, exploration, choices, combat, rng,
                         greedy_rows)

    def _eager(self, batch: Mapping[str, torch.Tensor], crit: np.ndarray | None) -> np.ndarray:
        """The eager forward of a collated batch (no graphs): (n, A + 1) logits | value, or with the critic
        (n, A + 4) logits | value | vc (plus optional trailing pwin_logit column when ``include_pwin=True``)."""
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cuda):
            b = batch_to(batch, self.device)
            o = self.net(b) if crit is None else self.net.forward_critic(b, torch.from_numpy(crit).to(self.device))
            cols = [o["logits"].float(), o["value"].float()[:, None]]
            if crit is not None:
                cols.append(o["vc"].float())
            if self.include_pwin:
                cols.append(o["pwin_logit"].float()[:, None])
            return torch.cat(cols, 1).cpu().numpy()

    def launch_encoded(self, enc: EncodedBatch, lo: int, hi: int, n_actions: np.ndarray,
                       rng: np.random.Generator | None = None, greedy_rows: np.ndarray | None = None,
                       crit: np.ndarray | None = None) -> _Launched:
        """:meth:`launch` for frames ``lo:hi`` of a batch-tokenized :class:`EncodedBatch`: collated
        straight into the recycled pinned staging buffers (no per-sample work, no intermediate
        tensors); the same batch tensors as ``launch(enc.samples()[lo:hi])``. ``crit``: the CRIT rows of frames
        ``lo:hi`` (a critic policy)."""
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                return self._launch_encoded(enc, lo, hi, rng, greedy_rows, crit)
        return self._launch_encoded(enc, lo, hi, rng, greedy_rows, crit)

    def _launch_encoded(self, enc: EncodedBatch, lo: int, hi: int, rng: np.random.Generator | None = None,
                        greedy_rows: np.ndarray | None = None, crit: np.ndarray | None = None) -> _Launched:
        t0 = time.perf_counter_ns()
        n = hi - lo
        exploration = (None if self._exploration is None or greedy_rows is not None else
                       self._exploration[enc.raw[enc.row0[lo:hi], self._kind_slot]])
        g0 = enc.row0[lo:hi]
        combat = combat_steps(enc.raw[g0, self._kind_slot], enc.raw[g0, self._in_combat_slot])
        if self.runner is None:
            batch = self.tok.collate_encoded(enc, lo, hi)
            t1 = time.perf_counter_ns()
            lv = self._eager(batch, self._crit_rows(crit, n, n))
            choices = (sample_choice_keys([enc.sample(i) for i in range(lo, hi)], self._verb_slot,
                                          lv.shape[1] - self._n_out) if self._choice_uniform else None)
            return _Launched(lv, None, None, None, t1 - t0, time.perf_counter_ns() - t1, exploration, choices, combat, rng,
                             greedy_rows)
        nat = self.tok.shape_encoded(enc, lo, hi)
        shape = self._grow_shape(self.runner.pad_shape(nat))
        free = self._staging.get(shape)
        stage = free.pop() if free else self._new_stage(shape)
        self.tok.collate_encoded_into(enc, lo, hi, shape, stage["_np"], nat=nat)
        crit_b = self._crit_rows(crit, n, shape.batch)
        if crit_b is not None:
            stage["_np"]["crit"][:] = crit_b
        t1 = time.perf_counter_ns()
        dev = self.runner.run_packed(stage["_buf"], shape, n)  # (n, A + 1): logits | value (| vc), on the device
        host = stage["_out"][:n]
        host.copy_(dev, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        choices = (sample_choice_keys([enc.sample(i) for i in range(lo, hi)], self._verb_slot, shape.actions)
                   if self._choice_uniform else None)
        return _Launched(host, ev, shape, stage, t1 - t0, time.perf_counter_ns() - t1, exploration, choices, combat, rng,
                         greedy_rows)

    def _new_stage(self, shape: BatchShape) -> dict[str, Any]:
        """Pinned host buffers for one in-flight batch of ``shape``: every batch tensor packed into
        one uint8 buffer (``_buf``, one host->device copy per replay; ``_np`` = numpy views the
        collation writes into) and the (logits | value) readback buffer ``_out``."""
        layout, size = packed_layout(shape, self.tok.width, self.draw_k)
        buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        views = packed_views(buf, layout)
        return {"_buf": buf, "_np": {k: v.numpy() for k, v in views.items()},
                "_out": torch.empty((shape.batch, shape.actions + self._n_out), dtype=torch.float32, pin_memory=True)}

    def done(self, handle: _Launched) -> bool:
        return handle.event is None or bool(handle.event.query())

    def finish(self, handle: _Launched) -> PolicyOut:
        t0 = time.perf_counter_ns()
        if handle.event is not None:
            handle.event.synchronize()
        t_wait = time.perf_counter_ns() - t0
        out = handle.out
        lv = out.numpy() if isinstance(out, torch.Tensor) else out
        assert handle.combat is not None
        temps = step_temperatures(handle.combat, self.combat_temperature)
        na = lv.shape[1] - self._n_out  # actions: logits | value (| vc)
        a, logp, ent, logp_all = self._sample_all(lv[:, :na], handle.exploration, handle.choice_keys,
                                                  temps if self.combat_temperature != 1.0 else None, handle.rng,
                                                  handle.greedy_rows)
        vc_end = na + 1 + (_N_VC if self.critic else 0)
        res = PolicyOut(a, logp, lv[:, na].copy(), ent, handle.collate_ns, handle.launch_ns + time.perf_counter_ns() - t0,
                        t_wait, lv[:, :na].copy(), logp_all, temps, handle.combat,
                        lv[:, na + 1 : vc_end].copy() if self.critic else None,
                        lv[:, vc_end].copy() if self.include_pwin else None)
        if handle.staging is not None and handle.shape is not None:
            free = self._staging.get(handle.shape)
            if free is not None:
                free.append(handle.staging)  # its copies have run: reusable
        return res

    def _sample(self, logits: np.ndarray, exploration: np.ndarray | None = None, choice_keys: np.ndarray | None = None,
                rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(action, log-prob, entropy) per row; illegal / padded actions are -inf. ``rng``: the generator to
        sample from instead of the policy's own."""
        return self._sample_all(logits, exploration, choice_keys, rng=rng)[:3]

    def _sample_all(self, logits: np.ndarray, exploration: np.ndarray | None = None,
                    choice_keys: np.ndarray | None = None, temps: np.ndarray | None = None,
                    rng: np.random.Generator | None = None, greedy_rows: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """:meth:`_sample` plus the log-probs of every row (0 at illegal rows). ``temps``: the per-row behaviour
        temperature (spire.temperature; the base policy of a row is softmax(logits / T)). ``greedy_rows`` (a scout
        batch, no exploration mixture): those rows take the arg-max of the masked untempered logits."""
        z = logits if self.mode == "greedy" or self.temperature == 1.0 else logits / self.temperature
        z_base = z  # untempered: a scout's arg-max (notes/next/p2: regardless of the combat temperature)
        if temps is not None:
            z = z / temps[:, None]
        legal = np.isfinite(z)
        zm = np.where(legal, z, -np.inf)
        mx = zm.max(1, keepdims=True)
        e = np.where(legal, np.exp(zm - mx), 0.0)
        s = e.sum(1, keepdims=True)
        logp_all = np.where(legal, zm - mx - np.log(s), 0.0)
        probs = e / s
        if exploration is not None:
            eps = exploration[:, None]
            log_eps = np.full_like(eps, -np.inf)
            np.log(eps, out=log_eps, where=eps > 0)
            log_uniform = log_eps - np.log(legal.sum(1, keepdims=True))
            if choice_keys is not None:
                reference = choice_uniform_probs_numpy(legal, choice_keys)
                log_uniform = log_eps + np.log(np.where(legal, reference, 1.0))
            logp_all = np.where(legal, np.logaddexp(logp_all + np.log1p(-eps), log_uniform), 0.0)
            probs = np.where(legal, np.exp(logp_all), 0.0)
            zm = np.where(legal, logp_all, -np.inf)
        if self.mode == "greedy":
            a = zm.argmax(1)
        else:
            g = -np.log(-np.log((self.rng if rng is None else rng).random(z.shape).clip(1e-12, 1.0 - 1e-7)))
            a = np.where(legal, zm + g, -np.inf).argmax(1)
            if greedy_rows is not None:
                assert exploration is None and self.temperature == 1.0, "a scout samples the base softmax"
                a = np.where(greedy_rows, np.where(legal, z_base, -np.inf).argmax(1), a)
        rows = np.arange(len(a))
        ent = -(probs * logp_all).sum(1)
        return a.astype(np.int64), logp_all[rows, a].astype(np.float32), ent.astype(np.float32), logp_all


# =========================================================================== actor

# Evaluation-only action override (habit study, notes/habit): (env id, frame, sample, the policy's scores of the frame's
# actions (-inf = illegal / cycle-banned), banned rows) -> ACTIONS row to take instead, or None for the policy's choice.
Override = Callable[[int, Frame, Sample, np.ndarray, Sequence[int]], "int | None"]



@dataclass(frozen=True)
class ActorConfig:
    mode: Literal["train", "eval"] = "train"
    max_batch: int = 128
    max_wait_ms: float = 1.0
    min_group: int = 16  # token-bucket groups smaller than this are merged into the next bucket
    reset_flags: int = int(RESET_FLAGS_DEFAULT)
    max_inflight: int = 2  # batches on the device at once (overlap of device and host work)
    # while the oldest batch is still on the device and hosts owe replies, keep polling the hosts
    # instead of blocking on the device, and poll while waiting to fill a batch (False: block as soon
    # as a poll comes back empty). Only used when no host has reader threads (stdio transport).
    spin_wait: bool = True
    report_interval: float = 5.0
    deadline_interval: float = 0.05
    env_id_base: int = 0  # env ids = base + index (unique across actor processes)
    max_episode_decisions: int = 0  # per seat; RESET word 14 = this * n_seats (0 = host default)
    # an idle spin iteration releases the GIL (time.sleep(0)) so a learner thread of the same process (threads
    # topology) gets to run; off by default (one process per actor: nobody else needs the GIL)
    yield_gil: bool = False
    # backstop for degenerate fights (cards that add cards on every play): a decision frame in which a seat's own
    # combat piles (seat_pile_totals) hold more cards than this ends the episode as a policy-caused cap, result
    # "pile_cap" (Actor._pile_cap); 0 = off
    pile_cap: int = 0
    gym_env_id_base: int = 0  # env ids of the practice child slots = base + slot (unique across actor processes)

    def __post_init__(self) -> None:
        assert self.pile_cap >= 0


@dataclass(eq=False)
class _Env:
    idx: int
    env_id: int
    host: Host
    history: PublicHistory | None = None  # only with tokenizer.public_history
    cycle: CycleBan | None = None
    job: EpisodeJob | None = None
    frame: Frame | None = None  # OBS frame awaiting a decision
    t_ready: int = 0  # perf_counter_ns when it became ready
    t_sent: int = 0
    deadline: float = 0.0  # monotonic seconds
    discard: bool = False  # waiting for the re-sent OBS after an ERROR, then RESET
    dead: bool = False
    t_start: float = 0.0
    decisions: int = 0
    auto_steps: int = 0
    version_first: int = -1
    last_facts: Facts | None = None
    last_frame: Frame | None = None
    last_combat: Frame | None = None
    terminal: Frame | None = None  # the episode's TERMINAL frame once it arrived
    act1: int = -1  # act vocab id of the first decision frame (Act 1 variant, e.g. Overgrowth / Underdocks)
    action_verbs: Counter[int] = field(default_factory=Counter)
    decision_kinds: Counter[int] = field(default_factory=Counter)
    final_hp: tuple[int, ...] = ()
    final_alive: tuple[bool, ...] = ()
    seat_deaths: int = 0
    removal: Counter[str] = field(default_factory=Counter)  # removal rule: "steps" and removed cards by class
    rules: Counter[str] = field(default_factory=Counter)  # rules_mode teach / free: the episode's RULES_KEYS counts
    # practice: a scout parent (v1.1), the parent's fight in progress, the FORK in flight (a parent waiting for
    # FORKED, a child that has not arrived), what makes the env a child
    scout: bool = False
    fight: "_Fight | None" = None
    fork: "_Fork | None" = None
    kid: "_Kid | None" = None

    def observe_seats(self, frame: Frame) -> None:
        """Keep exact public seat rows and count observable deaths, including repeated deaths after revive.

        WIN/ABORTED terminal rows may zero HP during teardown, so they cannot establish a death.
        A death followed by an automatic revive without an intermediate frame is unobservable.
        """
        seats = frame.table(Table.SEATS)
        if not len(seats):
            # Protocol P-5 explicitly allows a bare ABORTED terminal when game state
            # cannot be encoded. Keep the last observed rows, rather than inventing deaths.
            assert frame.is_terminal and frame.result == Result.ABORTED, "only aborted terminals may omit SEATS"
            return
        hp = tuple(int(value) for value in seats[:, SeatCol.HP])
        alive = tuple(bool(value) for value in seats[:, SeatCol.ALIVE])
        assert self.job is not None and len(alive) == len(self.job.characters)
        previous = self.final_alive or (True,) * len(alive)
        if not frame.is_terminal or (frame.result == Result.LOSS and
                                    frame.gvals[G.TERMINAL_REASON] in (TerminalReason.NONE, TerminalReason.DEATH)):
            self.seat_deaths += sum(before and not after for before, after in zip(previous, alive))
        self.final_hp, self.final_alive = hp, alive


@dataclass(eq=False)
class _Batch:
    envs: list[_Env]
    frames: list[Frame]
    facts: list[Facts]
    samples: list[Sample]
    groups: list[tuple[int, int, Any]]  # (first row, end row, policy handle)
    version: int  # policy weights version the batch was launched with
    enc: EncodedBatch | None = None
    finished: int = 0  # groups finished so far (they finish in launch order)
    kind: int = 0  # _PLAIN, _CHILDREN (practice children) or _SCOUTS (practice v1.1)
    crit: np.ndarray | None = None  # (frames, 2 + K) CRIT rows in batch order (a critic actor; scouts: no hidden rows)


_PLAIN, _CHILDREN, _SCOUTS = 0, 1, 2
_COMBAT_KINDS = (int(Kind.COMBAT), int(Kind.CARD_SELECT))  # with IN_COMBAT: a scout's greedy steps
# child results of practice_episodes.jsonl (spec 8)
_CAP_RESULTS = frozenset(("decision_cap", "turn_cap", "pile_cap", "truncated"))


@dataclass(eq=False)
class _Fight:
    """A parent's fight in progress (practice): opened at its first decision, closed at the first decision after it
    or at the episode's TERMINAL (it updates the difficulty table), dropped with a lost episode."""

    key: str  # encounter key "<act>|<room>|<enemies>" (+ "|2")
    cls: str  # "<act>|<room>" (+ "|2")
    asc: int
    floor: int  # TOTAL_FLOOR of the fight
    hp_in: float  # Facts.hp_frac at its first decision
    k: int = 0  # copies forked at its first decision (0 when none, refused or given up)


@dataclass(eq=False)
class _Kid:
    """What makes an env a practice child: its parent's fight, its place in the FORK, its end."""

    fight: _Fight
    index: int
    seed: int
    pair: int  # seed pair index (practice.paired_seeds; = index otherwise)
    fork_step: int  # STEP_ID of the fork decision
    parent: EpisodeStats | None  # the parent's wrapper record: its first ``fork_step`` actions lead to the fork
    parent_idx: int  # env index of the parent's host (per_parent cap)
    game: bool  # a GAME child (v1.2): plays to the run's TERMINAL; otherwise fight-only
    hello: Hello | None = None
    fight_end: FightEnd | None = None  # FIGHT_END received: exactly one frame follows
    final: bool = False  # the post-fight OBS is pending: evaluated once, never acted on
    post_hp: float = 0.0  # Facts.hp_frac of the first decision after the forked fight
    passed: bool = False  # a GAME child reached a decision after the forked fight (it won that fight)

    def repro(self) -> dict[str, Any]:
        """How to reproduce the child: replay ``prefix`` on its game's seed, FORK there with ``fork_seed``."""
        return {"gym": self.fight.key, "fork_step": self.fork_step, "fork_seed": self.seed, "fork_index": self.index,
                "pair": self.pair, "prefix": [] if self.parent is None else self.parent.actions[:self.fork_step]}


@dataclass(eq=False)
class _Fork:
    """One FORK: the parent's pending frame and the parent's state at that decision that a child takes when it
    arrives (the parent may have moved on by then): the rollout prefix and the history / cycle trackers, which
    have not observed the frame yet."""

    fight: _Fight
    frame: Frame
    seeds: tuple[int, ...]
    tag: int
    t0: int  # perf_counter_ns when FORK was sent
    job: EpisodeJob
    track: tuple[EpisodeMeta, list[Facts]]  # RolloutStore.fork_point: the reward / target prefix
    history: PublicHistory | None
    cycle: CycleBan | None
    flags: int  # FORK word 7: PRACTICE_FLAGS, or GAME_FLAGS for a fight of practice.game_classes (v1.2)
    kids: list[_Env] = field(default_factory=list)  # the child slots, in child index order
    error: int = 0  # ERROR code of a refused FORK (the same OBS follows)
    forker: int = 0  # pid of the parent's host process (the children's parent)


@dataclass(eq=False)
class _Gym:
    """The actor's practice state (spire.gym; lost at an actor restart)."""

    cfg: PracticeConfig
    q_a: float
    seeds: SeedStream  # the children's re-deal seeds
    rng: np.random.Generator  # samples the children's actions (the policy's generator is the full games')
    krng: np.random.Generator  # the copies' Bernoulli draws
    srng: np.random.Generator  # samples the scouts' actions
    pool: GymPool
    kids: list[_Env]  # the child slots (envs after the hosts' envs)
    free: list[int]  # env indices of the free slots
    table: Difficulty
    ctl: Controller
    alive: list[int]  # children alive per host env index (FORK to reap)
    counters: PracticeCounters = field(default_factory=PracticeCounters)
    child_total: int = 0  # child decisions since the actor started (the controller's input)
    keys: dict[tuple[str, int], list[float]] = field(default_factory=dict)  # cumulative per (key, asc): children,
    # fight wins, deaths, HP lost
    t_table: float = 0.0  # monotonic time of the last table dump in a report
    mode: int = MODE_RESAMPLE  # EXACT exists for tests on a debug server only: never a config option
    tag: int = 0


def group_ranges(buckets: np.ndarray, min_group: int, max_batch: int) -> list[tuple[int, int]]:
    """Inference groups of bucket-sorted frames as ``(lo, hi)`` ranges: consecutive buckets are
    merged until a group holds ``min_group`` frames (the last group takes the rest), then split
    into chunks of at most ``max_batch`` - the grouping of :meth:`Actor._groups`."""
    b = buckets.tolist()
    n = len(b)
    out: list[tuple[int, int]] = []
    lo = i = 0
    while i < n:
        j = i + 1
        while j < n and b[j] == b[i]:
            j += 1
        if j - lo >= min_group or j == n:
            out.extend((s, min(s + max_batch, j)) for s in range(lo, j, max_batch))
            lo = j
        i = j
    return out


CRITIC_KEYS = ("frames", "combat_frames", "hdraw_rows", "rows_kept", "truncated")
"""The actor report's ``critic`` counts (a critic actor): frames given CRIT (scouts excluded), of them in a fight, their
HDRAW rows, the rows CRIT kept, the rows it dropped (POS >= K: ``critic_draw_truncated``)."""

RULES_KEYS = ("removal_prompts", "removal_rule", "removal_other", "removal_cancel", "removal_fewer")
"""Counts of rules_mode teach / free (spire.rules): removal prompts, and the choice there: the rule's row, another
row (another card), CANCEL, CONFIRM while the rule selects a card (fewer than allowed)."""


@dataclass
class ActorCounters:
    decisions: int = 0
    card_offer_reopens: int = 0
    cycle_ban_decisions: int = 0
    cycle_ban_all_banned: int = 0
    cycle_ban_capacity_hits: int = 0
    batches: int = 0
    episodes_started: int = 0
    episodes_ended: int = 0
    stale_items: int = 0
    protocol_errors: int = 0
    errors_recorded: int = 0
    lost_recorded: int = 0  # reproduction records of episodes lost to a timeout / crash (on_error, kind "lost")
    pile_caps: int = 0  # episodes ended by ActorConfig.pile_cap
    hosts_retired: int = 0
    weight_loads: int = 0
    # spire.temperature: decisions on combat / other steps and those whose action is not the arg-max of the scores
    combat_steps: int = 0
    combat_nonargmax: int = 0
    noncombat_steps: int = 0
    noncombat_nonargmax: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class Actor:
    """Drives ``hosts`` with ``policy`` (see module docstring).

    Callbacks: ``on_episode(EpisodeRecord)`` for every ended episode (finished or failed),
    ``on_decision(env_id, frame, action)`` for every policy decision (in decision order per host),
    ``on_error(dict)`` with a reproduction record for every ERROR frame, ``on_report(dict)``
    every ``report_interval`` seconds with per-stage times, latencies, counters and host /
    rollout statistics accumulated since the previous report.
    """

    def __init__(self, hosts: Sequence[Host], policy: Policy, tokenizer: Tokenizer, source: EpisodeSource,
                 config: ActorConfig = ActorConfig(), *, store: RolloutStore | None = None,
                 on_episode: Callable[[EpisodeRecord], None] | None = None,
                 on_decision: Callable[[int, Frame, int], None] | None = None,
                 on_error: Callable[[dict[str, Any]], None] | None = None,
                 on_report: Callable[[dict[str, Any]], None] | None = None,
                 throttle: Callable[[], bool] | None = None,
                 advice: Sequence[Sequence[Any]] = (), catalog: Catalog | None = None,
                 override: Override | None = None, taste: Taste | None = None,
                 practice: PracticeConfig | None = None, practice_q: float = 0.0, practice_children: int = 0,
                 gym_seed: int = 0) -> None:
        """``throttle``: back-pressure predicate (e.g. the learner's segment backlog is above its high-water
        mark); while it returns True no new inference batch is launched - in-flight batches finish, frames keep
        arriving and wait - and the time is booked as ``throttled_s`` in the reports. ``advice``: the rules of
        ``PPOConfig.advice``; every recorded step then carries its advice record (old-format rules) and its
        habit-rule record (habit rules; :class:`spire.advice.Advisor`, the store must record them; ``catalog``: the
        card classes some habit judges read), timed as the ``advice`` stage. ``override``: the evaluation-only action
        override (:data:`Override`). ``taste``: per-episode card-family exploration (:mod:`spire.taste`; train mode,
        the store records its per-step terms). ``practice``: practice fights (train mode, hosts that implement FORK
        on the stdio transport; ``practice_q`` = the children's target share of this actor's decisions,
        ``practice_children`` = child slots); ``gym_seed`` seeds the children's re-deal seeds, the copies' draws and
        the children's / scouts' action sampling, nothing else."""
        if config.mode == "train" and store is None:
            raise ValueError("train mode needs a RolloutStore")
        assert override is None or config.mode == "eval", "action overrides are an evaluation-only measurement device"
        self.override = override
        self.advisor = Advisor(tokenizer.vocab, advice, catalog) if advice else None
        assert store is None or store.advice == (self.advisor is not None and bool(self.advisor.rules)), \
            "advice rules iff the store records advice"
        assert store is None or store.judged == (self.advisor is not None and bool(self.advisor.habits)), \
            "habit rules iff the store records habit-rule records"
        assert store is None or store.taste == (taste is not None), "taste iff the store records taste terms"
        assert not (isinstance(policy, TorchPolicy) and policy.combat_temperature != 1.0) or \
            (store is not None and store.beh_temp), \
            "a combat temperature needs a store that records beh_temp"
        assert taste is None or (config.mode == "train" and isinstance(policy, TorchPolicy) and policy.mode == "sample"), \
            "taste is a training-time behaviour exploration of a sampling TorchPolicy"
        self.taste = taste
        # the critic tower (notes/critic/contract.md 3): a critic store iff a critic policy, in train mode, on frames
        # with the hidden table; the hidden rows reach only CRIT (critic.side), never the tokenizer's Sample
        self.critic = store is not None and store.critic
        assert self.critic == bool(getattr(policy, "critic", False)), "a critic policy iff a critic store"
        assert not self.critic or (config.mode == "train" and tokenizer.allow_hidden and store is not None
                                   and store.critic_k == cast(TorchPolicy, policy).draw_k), \
            "a critic actor: train mode, tokenizer.allow_hidden, the store's K = the tower's draw_k"
        self.critic_counts: Counter[str] = Counter()  # report "critic": frames, combat frames, HDRAW rows, truncated
        self.cfg = config
        self.policy = policy
        self.tok = tokenizer
        self.vocab = tokenizer.vocab
        self.source = source
        self.store = store
        self.on_episode = on_episode
        self.on_decision = on_decision
        self.on_error = on_error
        self.on_report = on_report
        self.q: queue.SimpleQueue[tuple[int, int, object]] = queue.SimpleQueue()
        muxes = {id(m): m for m in (getattr(h, "mux", None) for h in hosts)}
        if len(muxes) > 1:
            raise ValueError("the hosts of one actor must all share one PipeMux or all use stdio / in-process")
        self.mux: PipeMux | PosixPipeMux | None = next(iter(muxes.values())) if muxes else None
        assert tokenizer.config.cycle_ban or not tokenizer.config.commit_select, "commit_select uses the cycle-ban mask"
        removal = load_rule(self.vocab, tokenizer.config.removal_ratings) if tokenizer.config.removal_rule else None
        self.removal_rule = removal is not None
        # spire.rules: "teach" / "free" count the removal prompts and the choices there (report "rules", the episode
        # record's ``rules``); in "free" the policy chooses the removal prompt's row itself
        self.rules_mode = tokenizer.config.rules_mode
        self.rules_counts: Counter[str] = Counter()
        self.envs = [_Env(i, config.env_id_base + i, h,
                         PublicHistory(tokenizer.width) if tokenizer.config.public_history else None,
                         CycleBan(self.vocab, commit_select=tokenizer.config.commit_select, removal=removal,
                                  mode=self.rules_mode)
                         if tokenizer.config.cycle_ban else None)
                     for i, h in enumerate(hosts)]
        for env in self.envs:  # pipe transport: replies are handled inside PipeMux.poll, no queue
            env.host.route(self._direct(env.idx) if self.mux is not None else self._sink(env.idx))
        self.ready: list[_Env] = []
        self.inflight: deque[_Batch] = deque()
        self._busy: set[int] = set()  # envs whose host owes a reply
        self.clock = StageClock(STAGES)
        self.counters = ActorCounters()
        self.actions = ActionStats()
        self._lat_actor: list[int] = []  # ns from frame arrival to STEP sent
        self._lat_host: list[int] = []  # ns from STEP sent to reply processed
        self._lat_gpu: list[int] = []  # ns per policy batch
        self._batch_sizes: list[int] = []
        self._host_prev = self._host_totals()
        self._roll_prev: dict[str, int] = store.stats.as_dict() if store is not None else {}
        self._gym_roll_prev: dict[str, int] = {}
        self._cnt_prev = self.counters.as_dict()
        self._t_report = time.monotonic()
        self._t_deadline = 0.0
        self._pump_ns = 0  # ns booked by the last _pump call
        self.throttle = throttle
        self._throttled = False
        self._throttled_ns = 0
        self._t_throttle = 0
        self._room_names = self.vocab["enum:RoomType"].entries if "enum:RoomType" in self.vocab.categories else []
        self._monster_names = self.vocab["monster"].entries if "monster" in self.vocab.categories else []
        self._act_names = self.vocab["act"].entries if "act" in self.vocab.categories else []
        self._stopping = False  # the final report (practice: it carries the difficulty table)
        self._all = self.envs  # every env by index: the hosts' envs, then the practice child slots
        self.gym: _Gym | None = None
        if practice is not None and practice.enabled:
            self._gym_init(practice, gym_seed, practice_q, practice_children)
        else:
            assert practice_q == 0.0 and practice_children == 0

    def _gym_init(self, cfg: PracticeConfig, seed: int, q_a: float, n_kids: int) -> None:
        """Practice state of a practice actor (spec 2.3 (c))."""
        assert self.cfg.mode == "train" and self.store is not None, "practice is a training device"
        assert self.mux is None, "practice needs the stdio transport: the children's frames arrive on the queue"
        assert not self.cfg.reset_flags & ResetFlag.ELIDE_UNCHANGED, "a child's first frame must be complete"
        assert self.store.cfg.keep_bootstrap_sample, "practice needs rollout.keep_bootstrap_sample"
        assert all(isinstance(e.host, HostProcess) for e in self.envs), "practice forks host processes"
        assert isinstance(self.policy, TorchPolicy) and self.policy.mode == "sample", "practice trains a sampling policy"
        assert 0.2 <= q_a <= 0.6 and n_kids >= 1, (q_a, n_kids)
        # v1.2 B: a GAME child meets CARD_REWARD decisions; steering them is not specified
        assert not cfg.game_classes or self.taste is None, "practice.game_classes needs a run without taste"
        n = len(self.envs)
        assert not {e.env_id for e in self.envs} & set(range(self.cfg.gym_env_id_base,
                                                               self.cfg.gym_env_id_base + n_kids)), \
            "gym_env_id_base: the child slots' env ids overlap the hosts'"
        pool = GymPool(f"{cfg.socket_dir}/g{os.getpid()}-{self.cfg.gym_env_id_base}.sock", self.q.put,
                       backlog=cfg.backlog)
        hosts = [e.host for e in self.envs]
        kids = [_Env(n + j, self.cfg.gym_env_id_base + j,
                     ChildHost(pool, step_timeout=max(h.step_timeout for h in hosts),
                               accept_timeout=cfg.accept_timeout, expected_hashes=hosts[0].expected_hashes),
                     dead=True) for j in range(n_kids)]
        self._all = self.envs + kids
        for e in self.envs:
            e.scout = cfg.scouts
        now = time.monotonic()
        self.gym = _Gym(cfg, q_a, SeedStream(seed), np.random.default_rng((seed, 1)), np.random.default_rng((seed, 2)),
                        np.random.default_rng((seed, 3)), pool, kids, [k.idx for k in reversed(kids)],
                        Difficulty(cfg), Controller(q_a, cfg.g_min, cfg.g_max, now), [0] * n, t_table=now)

    def _sink(self, idx: int) -> Callable[[int, object], None]:
        put = self.q.put
        return lambda gen, item: put((idx, gen, item))

    def _direct(self, idx: int) -> Callable[[int, object], None]:
        handle = self._handle
        return lambda gen, item: handle(idx, gen, item)

    # ------------------------------------------------------------------ main loop
    def run(self, stop: Callable[[], bool] = lambda: False, *, max_decisions: int | None = None,
            max_seconds: float | None = None) -> ActorCounters:
        """Play until ``stop()``, a limit, or (finite sources) every job is done; then shut down."""
        t_end = None if max_seconds is None else time.monotonic() + max_seconds
        for env in self.envs:
            self._start_episode(env)
        cfg, policy = self.cfg, self.policy
        max_wait = int(cfg.max_wait_ms * 1e6)
        # polling instead of blocking only without reader threads (pipe transport / in-process hosts):
        # a spinning loop would keep the GIL from stdio reader threads for a whole switch interval
        spin = cfg.spin_wait and not any(getattr(e.host, "reader_threads", False) for e in self.envs)
        yield_gil = cfg.yield_gil
        ready_since = 0
        try:
            while True:
                t_it = time.perf_counter_ns()
                if stop() or (max_decisions is not None and self.counters.decisions >= max_decisions) or \
                        (t_end is not None and time.monotonic() >= t_end):
                    break
                if not (self.ready or self.inflight or self._busy) and \
                        (self.source.exhausted or all(e.dead for e in self.envs)):
                    break
                had_ready = bool(self.ready)
                if self.inflight or (had_ready and spin):
                    # sub-millisecond scheduling: a blocking wait on Windows sleeps to the next timer tick
                    # (1-15.6 ms), so batching waits and device waits poll instead of blocking
                    timeout = 0.0
                elif had_ready:
                    timeout = max(0.0, (ready_since + max_wait - time.perf_counter_ns()) / 1e9) if self._busy else 0.0
                else:
                    timeout = cfg.deadline_interval
                got = self._pump(timeout)
                worked = got
                gpu_q = 0
                if self.throttle is not None and t_it - self._t_throttle >= 1_000_000:  # at most every 1 ms
                    self._t_throttle = t_it
                    self._throttled = self.throttle()
                if self._throttled and not self.inflight:
                    # back-pressure: hold the ready frames (their hosts wait) and sleep instead of spinning
                    t_s = time.perf_counter_ns()
                    time.sleep(0.002)
                    self._throttled_ns += time.perf_counter_ns() - t_s
                    self._t_throttle = 0  # re-evaluate right away
                    self._housekeeping()
                    continue
                if self.ready and not self._throttled:
                    now = time.perf_counter_ns()
                    if not had_ready:
                        ready_since = now
                    if len(self.inflight) < cfg.max_inflight and (
                            len(self.ready) >= cfg.max_batch or now - ready_since >= max_wait or not self._busy):
                        self._launch()
                        got = worked = True
                # Finish the oldest batch group by group (in launch order) as each group is done, or
                # block when there is nothing else to do: the hosts of a finished group get their
                # actions while the device still works on the later groups.
                while self.inflight:
                    b = self.inflight[0]
                    oldest = b.groups[b.finished][2]
                    if got:
                        if not policy.done(oldest):
                            break
                    elif spin and self._busy:
                        # Nothing arrived, but hosts owe replies: poll them again instead of blocking on
                        # the device, so frames arriving meanwhile are received (and the next batch
                        # launched) while the GPU works. Empty polls count as ``idle``, this query as ``gpu``.
                        t_q = time.perf_counter_ns()
                        done = policy.done(oldest)
                        gpu_q = time.perf_counter_ns() - t_q
                        self.clock.add(_GPU, gpu_q)
                        if not done:
                            break
                    self._finish_group(b)
                    if b.finished == len(b.groups):
                        self.inflight.popleft()
                    got = worked = True
                self._housekeeping()
                if not worked:  # a spin iteration without work: all of it is idle (minus what is booked)
                    if yield_gil:
                        time.sleep(0)
                    self.clock.add(_IDLE, time.perf_counter_ns() - t_it - self._pump_ns - gpu_q)
        finally:
            self._shutdown()
        return self.counters

    def _pump(self, timeout: float) -> bool:
        """Process routed items, blocking up to ``timeout`` seconds for the first. True if any.
        ``self._pump_ns`` = the ns this call booked (``wait`` + ``recv`` / ``idle``)."""
        q, clock = self.q, self.clock
        t0 = time.perf_counter_ns()
        mux = self.mux
        if mux is not None:  # pipe transport: poll() hands every complete frame to _handle directly
            w0 = mux.wait_ns
            got = mux.poll(timeout)
            wait = mux.wait_ns - w0
            clock.add(_WAIT, wait)
            dt = time.perf_counter_ns() - t0
            clock.add(_RECV if got else _IDLE, dt - wait)  # an empty poll is idle
            self._pump_ns = dt
            return got > 0
        else:
            try:
                item = q.get(timeout=timeout) if timeout > 0 else q.get_nowait()
            except queue.Empty:
                self._pump_ns = dt = time.perf_counter_ns() - t0
                clock.add(_WAIT, dt)
                return False
            t1 = time.perf_counter_ns()
            clock.add(_WAIT, t1 - t0)
        handle = self._handle
        while True:
            handle(*item)
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
        t2 = time.perf_counter_ns()
        clock.add(_RECV, t2 - t1)
        self._pump_ns = t2 - t0
        return True

    def _mark(self, env: _Env) -> None:
        """Keep ``_busy`` in sync with the host's outstanding-reply state."""
        if env.host.pending != Pending.NONE and not env.dead:
            self._busy.add(env.idx)
        else:
            self._busy.discard(env.idx)

    def _handle(self, idx: int, gen: int, item: object) -> None:
        if idx < 0:  # the boss gym's pool thread failed (spire.gym.GymPool)
            raise cast(BaseException, item)
        env = self._all[idx]
        host = env.host
        if gen != host.generation or env.dead:
            self.counters.stale_items += 1
            if type(item) is Arrival:  # a child whose fork was given up: EOF makes it exit
                cast(_Gym, self.gym).pool.drop(cast(Arrival, item).conn)
            return
        if env.fork is not None and self._fork_reply(env, item):
            return
        if env.kid is not None and type(item) is bytes and msg_type(item) == MSG_FIGHT_END:
            self._fight_end(env, item)
            return
        try:
            f = host.receive(item) if type(item) is bytes else host.deliver(item)  # type: ignore[arg-type]
        except HostUnrecoverable as e:
            self._retire(env, e)
            return
        except HostError as e:  # timeout / crash: the wrapper already restarted the process
            self._fail_episode(env, "lost", e)
            return
        except (ProtocolError, SchemaError) as e:
            self.counters.protocol_errors += 1
            try:
                host.restart(f"protocol violation: {e}")
            except HostUnrecoverable as e2:
                self._retire(env, e2)
                return
            self._fail_episode(env, "lost", e)
            return
        self._lat_host.append(time.perf_counter_ns() - env.t_sent)
        if f.msg_type == _MSG_OBS and not env.discard and not f.flags & _F_OVERFLOW:  # the common case
            env.observe_seats(f)
            env.auto_steps += f.auto_steps
            if self.cfg.pile_cap and self._pile_cap(env, f):
                self._mark(env)
                return
            env.frame = f
            env.t_ready = time.perf_counter_ns()
            if self.gym is not None and self._practice_frame(env, f):  # a FORK: the frame waits for FORKED
                return
            self.ready.append(env)
            if host.pending == _PENDING_NONE:
                self._busy.discard(env.idx)
            else:
                self._busy.add(env.idx)
            return
        self._on_frame(env, f)
        self._mark(env)

    def _on_frame(self, env: _Env, f: Frame) -> None:
        if f.is_error:
            self._record_error(env, f)
            if env.host.pending == Pending.RESEND:
                env.discard = True  # the host re-sends its OBS frame; RESET after that
                self._end_episode(env, "error", f"ERROR code {f.error_code}", store_reason="lost")
                env.deadline = _deadline(env.host)
            else:  # rejected RESET: no episode exists
                self._end_episode(env, "error", f"RESET rejected, code {f.error_code}", store_reason="lost")
                self._start_episode(env)
            return
        if env.discard:
            env.discard = False
            self._start_episode(env)
            return
        if f.is_terminal:
            if env.kid is not None and not env.kid.game:  # a GAME child's TERMINAL comes without (spire.gym.GymPool)
                fe = env.kid.fight_end
                assert fe is not None and fe.terminal == 1 and fe.step_id == f.step_id, "a child's TERMINAL without FIGHT_END"
            env.observe_seats(f)
            facts = frame_facts(f)
            env.last_facts = facts
            env.auto_steps += f.auto_steps
            env.terminal = f
            result = terminal_result_name(f.result, facts.terminal_reason)
            if self.store is not None and self.store.in_episode(env.env_id):
                outcome = self.store.end_episode(env.env_id, facts)
                if outcome == Outcome.OVERFLOW:
                    result = "overflow"
            if env.fight is not None:
                self._fight_terminal(env, result)
            self._end_episode(env, result, "")
            self._start_episode(env)
            return
        env.auto_steps += f.auto_steps
        if f.overflow:
            self._end_episode(env, "overflow", f"overflow at step {f.step_id}", store_reason="overflow")
            self._start_episode(env)
            return
        env.observe_seats(f)
        if self.cfg.pile_cap and self._pile_cap(env, f):
            return
        env.frame = f
        env.t_ready = time.perf_counter_ns()
        self.ready.append(env)

    def _pile_cap(self, env: _Env, f: Frame) -> bool:
        """End the episode at decision frame ``f`` if a seat's combat piles exceed ``cfg.pile_cap``; True if it did.

        Mirrors the host's decision cap: ``f`` stands in for the TERMINAL frame (RESULT=LOSS, TERMINAL_REASON
        DECISION_CAP), so the store applies the same cap rules (``RewardConfig.caps_as_failure``: death reward,
        no bootstrap); the record says "pile_cap" with ``f``'s room, enemies and win counters (TERMINAL_REASON 0:
        no host TERMINAL); the host is RESET for the next episode (it counts the open one as abandoned)."""
        if not f.glob[G.IN_COMBAT]:
            return False
        piles = seat_pile_totals(f)
        seat = int(piles.argmax())
        if piles[seat] <= self.cfg.pile_cap:
            return False
        facts = frame_facts(f)
        env.last_facts = facts
        env.last_frame = env.last_combat = env.terminal = f  # the record shows the fight that hit the cap
        result = "pile_cap"
        if self.store is not None and self.store.in_episode(env.env_id):
            cap = facts._replace(msg_type=_MSG_TERMINAL, result=_RES_LOSS, terminal_reason=_DECISION_CAP)
            if self.store.end_episode(env.env_id, cap) == Outcome.OVERFLOW:
                result = "overflow"
        self.counters.pile_caps += 1
        if env.fight is not None:
            self._fight_terminal(env, result)
        self._end_episode(env, result, f"seat {seat}: {int(piles[seat])} cards in its combat piles > pile_cap "
                                       f"{self.cfg.pile_cap} at step {f.step_id}")
        self._start_episode(env)
        return True

    # ------------------------------------------------------------------ decisions
    def _launch(self) -> None:
        """Launch the ready frames (:meth:`_launch_rows`). Practice children and scouts form batches of their own,
        sampled from the gym's generators: the batch of the full games is the one they would form without them."""
        ready, self.ready = self.ready, []
        gym = self.gym
        if gym is not None:
            plain = [e for e in ready if e.kid is None and not e.scout]
            scouts = [e for e in ready if e.scout]
            kids = [e for e in ready if e.kid is not None]
            if plain:
                self._launch_rows(plain, None, _PLAIN)
            if scouts:
                self._launch_rows(scouts, gym.srng, _SCOUTS)
            if kids:
                self._launch_rows(kids, gym.rng, _CHILDREN)
            return
        self._launch_rows(ready, None, _PLAIN)

    def _launch_rows(self, ready: list[_Env], rng: np.random.Generator | None, kind: int) -> None:
        """Facts + batch tokenization of the frames of ``ready``, inference launched per token-bucket group
        (``rng``: the generator a batch of children / scouts is sampled from; ``kind``: _PLAIN, _CHILDREN, _SCOUTS).

        ``encode_batch(sort=True)`` orders the frames by token bucket (stably), so every group is a
        contiguous range ``lo:hi`` of the batch - the same groups, in the same order and with the
        same members in the same order, as grouping the unsorted frames by bucket."""
        clock, policy, tok = self.clock, self.policy, self.tok
        t0 = time.perf_counter_ns()
        need = policy.needs_samples or self.store is not None or tok.config.cycle_ban
        frames = [cast(Frame, e.frame) for e in ready]
        enc = tok.encode_batch(frames, sort=need)
        if enc.order is not None:
            order = enc.order.tolist()
            ready = [ready[i] for i in order]
            frames = [frames[i] for i in order]
        samples = enc.samples()  # views share the encoded side record used by inference/storage
        for i, (env, frame) in enumerate(zip(ready, frames)):
            # The history tracker costs about 20 us per decision (measured in run l1-third): it runs only
            # when the model reads it. Without it the re-open counter stays 0 (cycle_ban_share covers it).
            if env.history is not None:
                enc.history[i] = env.history.observe(frame)
            if env.cycle is not None:
                env.cycle.observe(frame, samples[i])
        t1 = time.perf_counter_ns()
        clock.add(_ENCODE, t1 - t0)
        facts = [frame_facts(f) for f in frames]
        n_act = np.fromiter((f.n_actions for f in frames), np.int64, len(frames))
        t2 = time.perf_counter_ns()
        clock.add(_FACTS, t2 - t1)
        groups: list[tuple[int, int, Any]] = []
        kw: dict[str, Any] = {} if rng is None else {"rng": rng}
        greedy = None
        if kind == _SCOUTS:  # a scout: arg-max on its combat steps (spec v1.1 A.2)
            assert need
            greedy = np.fromiter((fa.in_combat and fa.kind in _COMBAT_KINDS for fa in facts), bool, len(facts))
        crit = None
        if self.critic:  # CRIT of every frame (critic.side; frames are in enc order); scouts: no rows (not trained)
            K = cast(TorchPolicy, policy).draw_k
            crit = (empty_crit(len(frames), K) if kind == _SCOUTS else
                    side_batch(frames, enc.card_base, enc.n_tokens, K))
            if kind == _SCOUTS:
                crit[:, 0] = [fa.kind for fa in facts]
            else:
                cc = self.critic_counts
                cc["frames"] += len(frames)
                cc["combat_frames"] += sum(fa.in_combat for fa in facts)
                cc["hdraw_rows"] += sum(0 if f.hdraw is None else int(f.hdraw.shape[0]) for f in frames)
                cc["rows_kept"] += int(crit[:, 1].sum())
                cc["truncated"] += sum(truncated(f, c) for f, c in zip(frames, crit))
        if not need:
            groups.append((0, len(frames), policy.launch([], n_act, **kw)))
        else:
            launch_enc = getattr(policy, "launch_encoded", None)
            for lo, hi in group_ranges(tok.bucket_of(enc.n_tokens), self.cfg.min_group, self.cfg.max_batch):
                if greedy is not None:
                    kw["greedy_rows"] = greedy[lo:hi]
                if crit is not None:
                    kw["crit"] = crit[lo:hi]
                h = launch_enc(enc, lo, hi, n_act[lo:hi], **kw) if launch_enc is not None else \
                    policy.launch(samples[lo:hi], n_act[lo:hi], **kw)
                groups.append((lo, hi, h))
        clock.add(_INFER, time.perf_counter_ns() - t2)
        # refresh() may adopt new weights before this batch is finished; its logits are the old ones
        self.inflight.append(_Batch(ready, frames, facts, samples, groups, policy.version, enc, kind=kind, crit=crit))

    def _finish(self, b: _Batch) -> None:
        """Wait for every remaining group of a launched batch, send the actions and record."""
        while b.finished < len(b.groups):
            self._finish_group(b)

    def _finish_group(self, b: _Batch) -> None:
        """Wait for the next unfinished group of ``b`` (groups finish in launch order), send its
        actions and record its transitions. Per batch this is exactly the old whole-batch finish:
        the policy samples group by group in the same order, rows are sent and recorded in the same
        order, and a failed send is handled after its group was recorded."""
        clock, policy = self.clock, self.policy
        lo, hi, handle = b.groups[b.finished]
        b.finished += 1
        t0 = time.perf_counter_ns()
        out = policy.finish(handle)
        actions, logp, value = out.actions, out.logp, out.value
        if self.override is not None:
            actions = self._override(b, lo, hi, actions, out.logits)
        ent = out.entropy if out.entropy is not None else np.zeros(hi - lo, np.float32)
        clock.add(_COLLATE, out.collate_ns)
        clock.add(_INFER, -out.collate_ns)  # _launch booked the whole launch (collate included)
        self._lat_gpu.append(out.collate_ns + out.gpu_ns)
        self._batch_sizes.append(hi - lo)
        self.counters.batches += 1
        t1 = time.perf_counter_ns()
        clock.add(_GPU, out.wait_ns)
        clock.add(_INFER, t1 - t0 - out.wait_ns)
        store, version = self.store, b.version
        envs, frames, facts = b.envs, b.frames, b.facts
        live = [i for i in range(lo, hi) if envs[i].frame is frames[i]]  # not reset meanwhile
        finals: list[int] = []  # practice children's post-fight frames: evaluated, never acted on
        if b.kind == _CHILDREN:
            finals = [i for i in live if cast(_Kid, envs[i].kid).final]
            if finals:
                live = [i for i in live if not cast(_Kid, envs[i].kid).final]
        all_live = len(live) == hi - lo
        sel = None if all_live else np.asarray(live, np.int64) - lo
        taste_rec: list[tuple[float, float]] | None = None  # (t_keep, t_add) per live row (spire.taste)
        if self.taste is not None and b.kind == _PLAIN:
            actions, logp, taste_rec = self._steer(b, lo, live, actions, logp, out)
        elif self.taste is not None and b.kind == _CHILDREN:  # a child's batch is not steered (spec 6.7)
            assert all(facts[i].kind != Kind.CARD_REWARD for i in live), "a practice child at a CARD_REWARD"
            taste_rec = [(1.0, 0.0)] * len(live)
        a_list = (actions if sel is None else actions[sel]).tolist()
        # 1. send first: hosts start on their next step while the transitions are recorded. A failed
        #    send is handled after recording, exactly as if it had happened after the record (old order).
        lat = self._lat_actor
        busy = self._busy
        failed: list[tuple[_Env, HostError]] = []
        for k, i in enumerate(live):
            env = envs[i]
            if env.cycle is not None:
                assert a_list[k] not in env.cycle.active_bans, "policy selected a cycle-banned action"
            env.frame = None
            host = env.host
            try:
                host.send_step(a_list[k])
            except HostError as e:  # HostUnrecoverable included
                failed.append((env, e))
                continue
            now = time.perf_counter_ns()
            env.t_sent = now
            lat.append(now - env.t_ready)
            t = host.reply_timeout()
            env.deadline = time.monotonic() + t if t > 0 else _INF
            busy.add(env.idx)
        t2 = time.perf_counter_ns()
        clock.add(_SEND, t2 - t1)
        # 2. record
        if b.enc is not None:  # verbs of the chosen actions straight from the batch words
            verbs = b.enc.action_column(_VERB, actions, lo)
            if sel is not None:
                verbs = verbs[sel]
        else:
            verbs = np.fromiter((frames[i].actions[int(actions[i - lo]), _VERB] for i in live), np.int64, len(live))
        kinds = np.fromiter((facts[i].kind for i in live), np.int64, len(live))
        # spire.temperature: combat steps by the frame's facts; the policy's flags (from the sample's global token) agree
        combat = combat_steps(kinds, np.fromiter((facts[i].in_combat for i in live), bool, len(live)))
        if out.combat is not None:
            assert np.array_equal(out.combat if sel is None else out.combat[sel], combat), \
                "policy and frame facts disagree on combat steps"
        if out.logits is not None and b.kind != _SCOUTS:  # a scout's arg-max is not sampled behaviour (notes/next/p2)
            greedy = out.logits.argmax(1) if sel is None else out.logits[sel].argmax(1)
            off = np.asarray(a_list, np.int64) != greedy
            n_c = int(combat.sum())
            self.counters.combat_steps += n_c
            self.counters.combat_nonargmax += int((off & combat).sum())
            self.counters.noncombat_steps += len(live) - n_c
            self.counters.noncombat_nonargmax += int((off & ~combat).sum())
        t_advice = 0
        if store is not None and b.kind == _SCOUTS:  # a scout's facts only (its children's prefix), no segment
            for i in live:
                cycle = envs[i].cycle
                store.add_scout_step(envs[i].env_id, facts[i] if cycle is None else
                                     facts[i]._replace(n_actions=facts[i].n_actions - len(cycle.active_bans)))
        elif store is not None:
            add = store.add_step
            samples = b.samples
            lp, vl = (logp if sel is None else logp[sel]).tolist(), (value if sel is None else value[sel]).tolist()
            advisor = self.advisor
            n_live = len(live)
            advice: list[Any] = [None] * n_live
            judged: list[Any] = [None] * n_live
            if advisor is not None:
                ta = time.perf_counter_ns()
                v_list = verbs.tolist()
                for k, i in enumerate(live):
                    cycle = envs[i].cycle
                    bans = () if cycle is None else cycle.active_bans
                    if advisor.rules:
                        advice[k] = advisor.candidate(frames[i], bans, a_list[k])
                    if advisor.habits:
                        judged[k] = advisor.judge(frames[i], bans, a_list[k], v_list[k])
                t_advice = time.perf_counter_ns() - ta
            # per-step keyword records, given exactly when the store records them (spire.taste, spire.temperature)
            extra: list[dict[str, Any]] = [{} for _ in live]
            if taste_rec is not None:
                for k in range(n_live):
                    extra[k]["taste"] = taste_rec[k]
            if store.beh_temp:
                assert out.beh_temp is not None, "the store records beh_temp: the policy must report it"
                for k, x in enumerate((out.beh_temp if sel is None else out.beh_temp[sel]).tolist()):
                    extra[k]["beh_temp"] = x
            if store.critic:  # the tower's values and the CRIT row of each step (notes/critic/contract.md 3)
                assert out.values_c is not None and b.crit is not None, "a critic store needs the tower's values"
                for k, i in enumerate(live):
                    extra[k]["critic"] = (out.values_c[i - lo], b.crit[i])
            for k, i in enumerate(live):
                cycle = envs[i].cycle
                stored_facts = facts[i] if cycle is None else \
                    facts[i]._replace(n_actions=facts[i].n_actions - len(cycle.active_bans))
                add(envs[i].env_id, stored_facts, samples[i], a_list[k], lp[k], vl[k], version, advice[k], judged[k],
                    **extra[k])
        on_decision = self.on_decision
        for k, i in enumerate(live):
            env, f, fa = envs[i], frames[i], facts[i]
            if on_decision is not None:
                on_decision(env.env_id, f, a_list[k])
            if env.history is not None:
                self.counters.card_offer_reopens += env.history.record(f, b.samples[i], a_list[k])
            if env.cycle is not None:
                status = env.cycle.status
                self.counters.cycle_ban_decisions += status.banned
                self.counters.cycle_ban_all_banned += status.all_banned
                self.counters.cycle_ban_capacity_hits += status.capacity_hits
                if status.removal_rule:
                    assert self.rules_mode == "free" or a_list[k] == env.cycle.rule_row
                    if a_list[k] == env.cycle.rule_row:  # "free": the steps on which the policy chose the rule's row
                        env.removal["steps"] += 1
                        if env.cycle.rule_class is not None:
                            env.removal[env.cycle.rule_class] += 1
                    if self.rules_mode != "mask":
                        self._rules_step(env, a_list[k] == env.cycle.rule_row, int(verbs[k]))
                env.cycle.record(f, a_list[k])
            env.decisions += 1
            env.action_verbs[int(verbs[k])] += 1
            env.decision_kinds[fa.kind] += 1
            if env.version_first < 0:
                env.version_first = version
                if env.kid is None:  # a fork child keeps the first act of its game
                    env.act1 = int(f.glob[G.ACT_ID]) if f.glob[G.ACT_INDEX] == 0 else -1
            env.last_facts = fa
            env.last_frame = f
            if fa.in_combat:
                env.last_combat = f
        self.actions.add(kinds, verbs, ent if sel is None else ent[sel])
        t3 = time.perf_counter_ns()
        clock.add(_RECORD, t3 - t2 - t_advice)
        clock.add(_ADVICE, t_advice)
        # 3. failed sends: the episode ends (its steps, this one included, are dropped)
        if failed:
            for env, err in failed:
                if isinstance(err, HostUnrecoverable):
                    self._retire(env, err)
                else:
                    self._fail_episode(env, "lost", err)
            clock.add(_SEND, time.perf_counter_ns() - t3)
        self.counters.decisions += len(live)
        if b.kind == _CHILDREN:
            gym = cast(_Gym, self.gym)
            gym.counters.decisions += len(live)
            gym.child_total += len(live)
            for i in finals:
                self._fight_won(envs[i], facts[i], float(value[i - lo]), b.samples[i],
                                None if out.values_c is None else out.values_c[i - lo])
        elif b.kind == _SCOUTS:
            cast(_Gym, self.gym).counters.scout_decisions += len(live)

    def _steer(self, b: _Batch, lo: int, live: list[int], actions: np.ndarray, logp: np.ndarray, out: PolicyOut
               ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
        """Taste (:mod:`spire.taste`) on the live rows of group ``lo:``: the actions, their behaviour log-probs and
        the per-step ``(t_keep, t_add)`` records (in ``live`` order)."""
        taste = self.taste
        assert taste is not None and out.logits is not None and out.logp_all is not None, \
            "taste needs the policy's scores and log-probs"
        actions, logp = actions.copy(), logp.copy()
        rec = []
        for i in live:
            r = i - lo
            a, lp, t_keep, t_add = taste.steer(b.envs[i].env_id, b.frames[i], out.logits[r], out.logp_all[r],
                                               int(actions[r]))
            if t_keep != 1.0:
                actions[r], logp[r] = a, lp
            rec.append((t_keep, t_add))
        return actions, logp, rec

    def _rules_step(self, env: _Env, rule: bool, verb: int) -> None:
        """rules_mode teach / free: count one removal prompt and its choice (:data:`RULES_KEYS`) for the episode and the
        report; a choice other than the rule's row is CANCEL, CONFIRM (fewer than allowed: the rule's row is CONFIRM
        once the selection is full) or another row."""
        key = "removal_rule" if rule else ("removal_cancel" if verb == Verb.CANCEL else "removal_fewer"
                                           if verb == Verb.CONFIRM_SELECTION else "removal_other")
        c = Counter({"removal_prompts": 1, key: 1})
        env.rules.update(c)
        self.rules_counts.update(c)

    def _override(self, b: _Batch, lo: int, hi: int, actions: np.ndarray, logits: np.ndarray | None) -> np.ndarray:
        """``actions`` of group ``lo:hi`` with ``self.override``'s choices (live frames only)."""
        assert self.override is not None and logits is not None, "an override needs the policy's scores"
        actions = actions.copy()
        for i in range(lo, hi):
            env, f = b.envs[i], b.frames[i]
            if env.frame is not f:
                continue
            banned = env.cycle.active_bans if env.cycle is not None else ()
            scores = logits[i - lo, :f.n_actions]
            assert int(np.argmax(scores)) == int(actions[i - lo]), "override needs greedy scores"
            a = self.override(env.env_id, f, b.samples[i], scores, banned)
            if a is not None:
                assert 0 <= a < f.n_actions and a not in banned and np.isfinite(scores[a]), (a, f.n_actions, banned)
                actions[i - lo] = a
        return actions

    def _groups(self, samples: Sequence[Sample]) -> list[list[int]]:
        """Reference grouping of unsorted samples by token bucket (small groups merge upwards);
        :func:`group_ranges` is the same grouping on bucket-sorted frames."""
        bucket = self.tok.config.bucket
        by: dict[int, list[int]] = {}
        for i, s in enumerate(samples):
            by.setdefault(bucket(s.n_tokens), []).append(i)
        keys = sorted(by)
        out: list[list[int]] = []
        carry: list[int] = []
        mb = self.cfg.max_batch
        for k in keys:
            carry += by[k]
            if len(carry) >= self.cfg.min_group or k == keys[-1]:
                out.extend(carry[j : j + mb] for j in range(0, len(carry), mb))
                carry = []
        return out

    # ------------------------------------------------------------------ episodes
    def _start_episode(self, env: _Env) -> None:
        self._busy.discard(env.idx)
        while not env.dead:
            job = self.source.next_job(env.env_id)
            env.job = job
            if job is None:
                return
            env.t_start = time.monotonic()
            env.decisions = env.auto_steps = 0
            env.final_hp = env.final_alive = ()
            env.seat_deaths = 0
            env.action_verbs.clear()
            env.decision_kinds.clear()
            env.removal.clear()
            env.rules.clear()
            if env.history is not None:
                env.history.clear()
            if env.cycle is not None:
                env.cycle.clear()
            env.version_first = -1
            env.act1 = -1
            env.last_facts = env.last_frame = env.last_combat = env.terminal = None
            env.fight = None
            if self.store is not None:
                self.store.begin_episode(env.env_id, job.characters, job.ascension,
                                         -1 if job.seed is None else seed_to_wire(job.seed), scout=env.scout)
            if self.taste is not None and not env.scout:
                self.taste.begin(env.env_id, job.characters)
            self.counters.episodes_started += 1
            try:
                env.host.send_reset(job.reset_cmd(self.cfg.reset_flags, self.cfg.max_episode_decisions))
            except HostUnrecoverable as e:
                self._retire(env, e)
                return
            except HostStateError:  # actor / wrapper out of sync: retrying would spin through every job
                raise
            except HostError as e:
                self._end_episode(env, "lost", f"RESET failed: {e}", store_reason="lost")
                continue
            env.t_sent = time.perf_counter_ns()
            env.deadline = _deadline(env.host)
            self._busy.add(env.idx)
            return

    def _fail_episode(self, env: _Env, result: str, exc: BaseException) -> None:
        self._record_lost(env, exc)
        self._end_episode(env, result, f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}", store_reason="lost")
        env.discard = False
        self._start_episode(env)

    def _end_episode(self, env: _Env, result: str, detail: str, store_reason: str | None = None) -> None:
        if env.job is None:
            return
        if store_reason is not None and self.store is not None and self.store.in_episode(env.env_id):
            self.store.abort_episode(env.env_id, store_reason)
        if env.frame is not None:
            if env in self.ready:
                self.ready.remove(env)
            env.frame = None  # a launched batch skips it (identity check in _finish)
        if env.fork is not None and env.kid is None:  # the parent of a FORK in flight
            self._fork_drop(env, f"{result}: {detail}" if detail else result, count=result != "shutdown")
        env.fight = None  # a parent's fight not closed by its TERMINAL: the episode was lost
        practice = self._kid_done(env, result) if env.kid is not None else {}
        rec = self._record(env, result, detail, practice)
        if self.taste is not None and env.kid is None and not env.scout:
            self.taste.end(env.env_id, result, rec.floor, rec.ascension, rec.bosses_won)
        env.job = None
        if env.kid is not None:  # a practice child's episode is not a game: counted by its record's gym key
            self._reap(env)
        elif not env.scout:  # a scout's neither (v1.1 A.4)
            self.counters.episodes_ended += 1
        if self.on_episode is not None:
            self.on_episode(rec)

    def _record(self, env: _Env, result: str, detail: str, practice: dict[str, Any] | None = None) -> EpisodeRecord:
        job = env.job
        assert job is not None
        fa = env.last_facts
        enemies: tuple[str, ...] = ()
        if env.last_combat is not None and result != "win":
            cre = env.last_combat.table(Table.CREATURES)
            mids = cre[(cre[:, CreatureCol.KIND] == CreatureKind.ENEMY), CreatureCol.MODEL_ID].tolist()
            enemies = tuple(sorted(self._name(self._monster_names, m) for m in mids))
        last = env.last_frame
        term = env.terminal
        tg = term.gvals if term is not None else None
        return EpisodeRecord(
            lineup=job.lineup, ascension=job.ascension, result=result,
            floor=fa.total_floor if fa is not None else 0,
            act=int(last.glob[G.ACT_INDEX]) if last is not None else 0,
            room=self._name(self._room_names, int(last.glob[G.ROOM_TYPE])) if last is not None else "",
            in_combat=bool(last is not None and last.glob[G.IN_COMBAT]), enemies=enemies,
            decisions=env.decisions, auto_steps=env.auto_steps, seed=job.seed or "", seed_id=job.seed_id,
            env=env.env_id, version_first=env.version_first, version_last=self.policy.version,
            seconds=round(time.monotonic() - env.t_start, 3), detail=detail,
            terminal_reason=tg[_G_REASON] if tg is not None else -1, abort_reason=tg[_G_ABORT] if tg is not None else 0,
            combats_won=tg[_G_COMBATS] if tg is not None else -1, elites_won=tg[_G_ELITES] if tg is not None else -1,
            bosses_won=tg[_G_BOSSES] if tg is not None else -1,
            final_hp=env.final_hp, final_alive=env.final_alive, seat_deaths=env.seat_deaths,
            act1=self._name(self._act_names, env.act1) if env.act1 >= 0 else "",
            action_verbs={Verb(v).name: n for v, n in env.action_verbs.items()}
            if env.decisions > LONG_EPISODE_DECISIONS else {},
            decision_kinds={Kind(k).name: n for k, n in env.decision_kinds.items()}
            if env.decisions > LONG_EPISODE_DECISIONS else {},
            removal={"steps": env.removal["steps"], **{c: env.removal[c] for c in REMOVAL_CLASSES}}
            if self.removal_rule else {},
            rules={k: env.rules[k] for k in RULES_KEYS} if self.rules_mode != "mask" else {},
            gym=env.kid.fight.key if env.kid is not None else "", scout=env.scout, practice=practice or {},
        )

    @staticmethod
    def _name(names: Sequence[str], idx: int) -> str:
        return names[idx] if 0 <= idx < len(names) else str(idx)

    def _record_error(self, env: _Env, f: Frame) -> None:
        self.counters.errors_recorded += 1
        ep = env.host.episode or env.host.last_episode
        job = env.job
        if self.on_error is not None:
            self.on_error({
                "env": env.env_id, "code": f.error_code, "step_id": f.step_id, "episode_id": f.episode_id,
                "seed": None if job is None else job.seed, "lineup": None if job is None else job.lineup,
                "ascension": None if job is None else job.ascension,
                "actions": [] if ep is None else list(ep.actions), "policy_version": self.policy.version,
            })

    def _record_lost(self, env: _Env, exc: BaseException) -> None:
        """Reproduction record of an episode lost to a timeout / crash / bad frame (``on_error``, kind "lost"):
        the exact RESET and every action index sent, the unanswered one last. Replaying RESET + ``actions`` +
        ``unanswered`` against the same host build reproduces the failing request."""
        job = env.job
        if job is None:  # the episode already ended (ERROR resend pending): nothing was lost here
            return
        ep = env.host.last_episode
        cmd = job.reset_cmd(self.cfg.reset_flags, self.cfg.max_episode_decisions)
        assert ep is not None and ep.seed == cmd.seed and ep.characters == cmd.characters, \
            "the host's lost episode is not this env's job"
        sent = len(ep.actions) + (ep.unanswered >= 0)
        assert sent in (env.decisions, env.decisions - 1), (sent, env.decisions)  # -1: a STEP whose send failed
        self.counters.lost_recorded += 1
        if self.on_error is not None:
            self.on_error({
                "kind": "lost", "env": env.env_id, "detail": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}",
                "episode_id": ep.episode_id, "seed": job.seed, "lineup": job.lineup, "characters": list(cmd.characters),
                "ascension": cmd.ascension, "reset_flags": cmd.flags, "max_decisions": cmd.max_decisions,
                "decisions": env.decisions, "actions": list(ep.actions), "unanswered": ep.unanswered,
                "policy_version": self.policy.version,
                **({} if env.kid is None else env.kid.repro()),  # a fork child: ``actions`` are its own
            })

    def _retire(self, env: _Env, exc: BaseException) -> None:
        self._record_lost(env, exc)
        self._end_episode(env, "lost", f"host retired: {exc}", store_reason="lost")
        env.dead = True
        self._busy.discard(env.idx)
        self.counters.hosts_retired += 1

    # ------------------------------------------------------------------ practice fights
    def _practice_frame(self, env: _Env, f: Frame) -> bool:
        """A decision frame of a practice actor's env. A child: its frames before FIGHT_END are decisions of the
        forked fight; the one after it is the post-fight OBS (marked ``final``); a GAME child's frames go on to the
        run's end (its first decision after the forked fight marks that fight won). A parent: closes its fight at the
        first decision after it; at the first decision of a fight draws the copies and forks (True: the frame
        waits for FORKED)."""
        g = f.glob
        kid = env.kid
        if kid is not None:
            inside = g[G.IN_COMBAT] == 1 and g[G.TOTAL_FLOOR] == kid.fight.floor
            if kid.game:  # never forks (depth 1)
                if not inside and not kid.passed:
                    kid.passed = True
                    kid.post_hp = frame_facts(f).hp_frac
                return False
            if kid.fight_end is None:
                assert inside, f"child {env.env_id}: a frame outside its fight before FIGHT_END"
            else:
                fe = kid.fight_end
                assert not kid.final and fe.terminal == 0 and fe.step_id == f.step_id, (fe, f.step_id)
                assert not inside, f"child {env.env_id}: the post-fight frame continues the fight"
                kid.final = True
            return False
        gym = cast(_Gym, self.gym)
        fight = env.fight
        floor = int(g[G.TOTAL_FLOOR])
        if fight is not None:
            if g[G.IN_COMBAT] == 1 and floor == fight.floor:
                return False
            self._fight_close(env, False, max(0.0, fight.hp_in - frame_facts(f).hp_frac))
        if g[G.IN_COMBAT] != 1:
            return False
        job = cast(EpisodeJob, env.job)
        act = int(g[G.ACT_INDEX])
        room = self._name(self._room_names, int(g[G.ROOM_TYPE]))
        second = room == "Boss" and int(g[G.BOSSES_WON]) > act
        cre = f.table(Table.CREATURES)
        enemies = "+".join(sorted(self._name(self._monster_names, m) for m in
                                  cre[cre[:, CreatureCol.KIND] == CreatureKind.ENEMY, CreatureCol.MODEL_ID].tolist()))
        cls = encounter_class(act + 1, room, second)
        key = f"{act + 1}|{room}|{enemies}" + ("|2" if second else "")
        fight = env.fight = _Fight(key, cls, job.ascension, floor, frame_facts(f).hp_frac)
        cfg = gym.cfg
        if cfg.ascensions and job.ascension not in cfg.ascensions:
            return False
        want = gym.table.copies(gym.table.x(gym.table.s(key, cls, job.ascension), gym.ctl.g), float(gym.krng.random()))
        if want == 0:
            return False
        k = min(want, cfg.per_parent - gym.alive[env.idx], len(gym.free))
        if k <= 0:
            gym.counters.capped += 1
            return False
        if k < want:
            gym.counters.trimmed += 1
        return self._fork(env, f, fight, k)

    def _fight_close(self, env: _Env, death: bool, hp_lost: float) -> None:
        """A parent's fight ended with its episode still tracked: the difficulty table and the class totals."""
        gym, fight = cast(_Gym, self.gym), cast(_Fight, env.fight)
        env.fight = None
        gym.table.update(fight.key, fight.cls, fight.asc, death, hp_lost)
        c = gym.counters.cls(fight.cls)
        c["parent_fights"] += 1
        c["parent_deaths"] += death
        c["parent_hp_lost"] += hp_lost
        c["forked_w"] += fight.k
        c["forked_deaths_w"] += fight.k * death
        c["forked_hp_lost_w"] += fight.k * hp_lost
        if fight.asc in TOP_ASCENSIONS:
            ca = gym.counters.asc(fight.cls, fight.asc)
            ca["parent_fights"] += 1
            ca["parent_deaths"] += death

    def _fight_terminal(self, env: _Env, result: str) -> None:
        """A parent's TERMINAL (or pile cap) inside its fight: the run's last boss won (HP lost to the last
        decision: the TERMINAL carries HP 0), a death or a cap (everything that was left)."""
        fight = cast(_Fight, env.fight)
        if result == "win":  # a TERMINAL follows a STEP: the last acted frame is a decision of this fight
            last = cast(Frame, env.last_frame)
            assert last.glob[G.TOTAL_FLOOR] == fight.floor and last.glob[G.IN_COMBAT] == 1
            self._fight_close(env, False, max(0.0, fight.hp_in - frame_facts(last).hp_frac))
        elif result == "loss" or result in _CAP_RESULTS:
            self._fight_close(env, True, fight.hp_in)

    def _fork(self, env: _Env, f: Frame, fight: _Fight, k: int) -> bool:
        """Fork ``k`` children of the game of ``env`` at its pending frame ``f`` (the first decision of ``fight``):
        the child slots are taken, FORK is sent and the frame is held until FORKED arrives (:meth:`_fork_reply`)."""
        gym = cast(_Gym, self.gym)
        assert env.kid is None and env.job is not None and self.store is not None
        gym.tag += 1
        host = env.host
        seeds = gym.seeds.fork_seeds(k, gym.cfg.paired_seeds)
        # the parent's state at the fork decision that the children take when they arrive (the parent may have
        # moved on by then): the rollout prefix and the history / cycle-ban trackers, which have recorded the
        # previous decision and not yet observed ``f``
        flags = gym.cfg.fork_flags(fight.cls)
        game = flags == GAME_FLAGS
        fork = env.fork = _Fork(
            fight, f, seeds, gym.tag, time.perf_counter_ns(), env.job, self.store.fork_point(env.env_id),
            None if env.history is None else env.history.fork(), None if env.cycle is None else env.cycle.fork(),
            flags, forker=cast(int, cast(HostProcess, host).pid))
        fight.k = k
        deadline = time.monotonic() + gym.cfg.accept_timeout
        seed = None if env.job.seed is None else seed_to_wire(env.job.seed)
        for i in range(k):  # expected before the FORK: a child may connect before FORKED arrives
            kid = self._all[gym.free.pop()]
            child = cast(ChildHost, kid.host)
            child.begin(seed, env.job.characters, env.job.ascension, f.episode_id)
            kid.job, kid.fork = env.job, fork
            kid.kid = _Kid(fight, i, seeds[i], i // 2 if gym.cfg.paired_seeds else i, f.step_id, host.episode,
                           env.idx, game)
            kid.dead = kid.discard = kid.scout = False
            kid.frame = kid.terminal = None
            kid.fight = None
            # Copied from the parent: the seat rows (deaths are counted against them), the first act, the last
            # recorded frames (the record's place if the child is lost before its first decision). Counted from
            # the fork: decisions, auto steps (the fork frame's are the parent's), deaths, time, verbs, kinds.
            kid.final_hp, kid.final_alive, kid.act1 = env.final_hp, env.final_alive, env.act1
            kid.last_facts, kid.last_frame, kid.last_combat = env.last_facts, env.last_frame, env.last_combat
            kid.t_start = time.monotonic()
            kid.decisions = kid.auto_steps = kid.seat_deaths = 0
            kid.version_first = -1
            kid.action_verbs.clear()
            kid.decision_kinds.clear()
            kid.removal.clear()
            kid.rules.clear()
            kid.deadline = deadline
            gym.pool.expect(fork.tag, i, kid.idx, child.generation, game)
            self._busy.add(kid.idx)
            fork.kids.append(kid)
        gym.alive[env.idx] += k
        try:
            host._send_guarded(encode_fork(f.step_id, seeds, gym.mode, fork.tag, gym.pool.path, flags))
        except HostUnrecoverable as e:  # the episode ends; _end_episode gives the fork up
            self._retire(env, e)
            return True
        except HostError as e:
            self._fail_episode(env, "lost", e)
            return True
        env.deadline = deadline
        self._busy.add(env.idx)
        return True

    def _fork_reply(self, env: _Env, item: object) -> bool:
        """An item for an env with a FORK in flight: the parent's FORKED, or ERROR and the same OBS again
        (refused); a child's :class:`~spire.gym.Arrival` and then its first frame. False: ``item`` is not part
        of the fork (a transport failure) and takes the normal path, where the episode ends - which gives the fork
        up (parent) or counts the child as lost."""
        gym, fork = cast(_Gym, self.gym), cast(_Fork, env.fork)
        kid = env.kid
        if kid is not None:
            child = cast(ChildHost, env.host)
            if type(item) is Arrival:
                h = cast(Arrival, item).hello
                assert kid.hello is None and (h.tag, h.index, h.mode, h.depth, h.flags) == \
                    (fork.tag, kid.index, gym.mode, 1, fork.flags), h
                assert (h.episode_id, h.step_id) == (fork.frame.episode_id, fork.frame.step_id), h
                assert gym.mode == MODE_EXACT or h.seed == kid.seed, (h.seed, kid.seed)
                assert child.pid in (0, h.pid) and child.forker in (0, h.parent_pid), (h, child.pid, child.forker)
                assert h.parent_pid == h.root_pid == fork.forker, h
                kid.hello = h
                child.pid, child.forker = h.pid, h.parent_pid
                child.conn = cast(Arrival, item).conn
                return True
            if type(item) is not bytes or kid.hello is None:  # the pool hands over bytes
                return False
            assert same_obs(item, fork.frame.payload), f"child {kid.hello.pid}: first frame differs from the parent's"
            self._adopt(env, fork, child.first(item))
            return True
        if type(item) is not bytearray:  # the stdout reader hands over bytearrays (proto.read_message)
            return False
        mt = msg_type(item)
        if fork.error:  # the OBS re-sent after the ERROR: the game is where it was
            assert same_obs(item, fork.frame.payload), "after a FORK ERROR the same OBS must be re-sent"
            self._fork_drop(env, f"FORK refused: ERROR code {fork.error}")
        elif mt == _MSG_ERROR:
            fork.error = parse_frame(item).error_code
            env.host._count_error()
            return True
        elif mt == MSG_FORKED:
            fk = parse_forked(item)
            assert (fk.tag, fk.n, fk.mode, fk.episode_id, fk.step_id) == \
                (fork.tag, len(fork.seeds), gym.mode, fork.frame.episode_id, fork.frame.step_id), fk
            assert len(set(fk.pids)) == fk.n and all(p > 0 for p in fk.pids), fk.pids
            for k in fork.kids:
                if k.fork is fork:  # not arrived yet, or arrived and named by its HELLO
                    child = cast(ChildHost, k.host)
                    assert child.pid in (0, fk.pids[cast(_Kid, k.kid).index]), (child.pid, fk.pids)
                    child.pid, child.forker = fk.pids[cast(_Kid, k.kid).index], fork.forker
            c = gym.counters
            c.forks += 1
            c.children += fk.n
            cc = c.cls(fork.fight.cls)
            cc["forks"] += 1
            cc["children"] += fk.n
            if fork.flags == GAME_FLAGS:
                cc["game_children"] += fk.n
            if fork.fight.asc in TOP_ASCENSIONS:
                c.asc(fork.fight.cls, fork.fight.asc)["children"] += fk.n
            for d in (c.keys, gym.keys):
                d.setdefault((fork.fight.key, fork.fight.asc), [0, 0, 0, 0.0])[0] += fk.n
            c.fork_ns.append(time.perf_counter_ns() - fork.t0)
            c.host_us += fk.host_us
            env.fork = None
        else:
            return False
        env.t_ready = time.perf_counter_ns()  # the held frame goes on untouched
        self.ready.append(env)
        self._busy.discard(env.idx)
        return True

    def _adopt(self, env: _Env, fork: _Fork, f: Frame) -> None:
        """A child arrived with its first frame ``f`` (= the parent's frame at the fork): it becomes a playing
        env. Its trackers (history, cycle ban incl. commit_select and the removal rule) are copies of the parent's
        at the fork decision; its rewards, potentials, combat ids and targets are re-derived by the store from the
        parent's facts (RolloutStore.fork_episode)."""
        gym = cast(_Gym, self.gym)
        assert self.store is not None
        env.fork = None
        env.history = None if fork.history is None else fork.history.fork()
        env.cycle = None if fork.cycle is None else fork.cycle.fork()
        if gym.cfg.rloo_alpha > 0.0:
            kid = cast(_Kid, env.kid)
            fg = fork_group_id(gym.seeds.seed, fork.tag, kid.pair if gym.cfg.rloo_by_pair else 0)
            self.store.fork_episode(fork.track, env.env_id, game=fork.flags == GAME_FLAGS, fork_group=fg)
        else:
            self.store.fork_episode(fork.track, env.env_id, game=fork.flags == GAME_FLAGS)
        env.frame = f
        env.t_ready = time.perf_counter_ns()
        self.ready.append(env)
        self._busy.discard(env.idx)
        gym.counters.started += 1
        gym.counters.arrive_ns.append(env.t_ready - fork.t0)

    def _fight_end(self, env: _Env, item: bytes) -> None:
        """A child's FIGHT_END (protocol 3.4): exactly one frame follows (TERMINAL or the post-fight OBS)."""
        kid = cast(_Kid, env.kid)
        fe = parse_fight_end(item)
        assert not kid.game, "a GAME child sent FIGHT_END"
        assert kid.hello is not None and kid.fight_end is None, "FIGHT_END before the first frame or twice"
        assert (fe.tag, fe.index, fe.depth, fe.episode_id) == (kid.hello.tag, kid.index, 1, kid.hello.episode_id), fe
        kid.fight_end = fe

    def _fight_won(self, env: _Env, fa: Facts, value: float, sample: Sample, values_c: np.ndarray | None) -> None:
        """A child's post-fight OBS was evaluated: its episode is truncated there with that frame's value and
        sample as the bootstrap (no step, no STEP, no tracker record; the sampled action is discarded); a critic
        actor's child also bootstraps the tower with that frame's ``values_c`` (notes/critic/contract.md 3)."""
        assert self.store is not None
        kid = cast(_Kid, env.kid)
        cycle = env.cycle
        stored = fa if cycle is None else fa._replace(n_actions=fa.n_actions - len(cycle.active_bans))
        self.store.truncate_episode(env.env_id, stored, value, sample, values_c)
        kid.post_hp = fa.hp_frac
        env.frame = None
        self._end_episode(env, "fight_won", "")

    def _kid_done(self, env: _Env, result: str) -> dict[str, Any]:
        """A child's end: its class / key totals and the practice part of its record (spec 8). For a GAME child
        (v1.2) ``hp_lost``, fight wins and deaths are those of the forked fight; its run's end is counted apart."""
        gym, kid = cast(_Gym, self.gym), cast(_Kid, env.kid)
        fight = kid.fight
        if result in ("fight_won", "win", "loss"):
            res = result
        elif result in _CAP_RESULTS:
            res = "cap"
        elif result == "shutdown":
            res = "shutdown"
        else:
            res = "lost"
        assert res != "fight_won" or not kid.game, "a GAME child is never truncated after its fight"
        hp_lost: float | None = None
        if res == "fight_won" or (kid.passed and res in ("win", "loss", "cap")):  # a GAME child after its fight
            hp_lost = max(0.0, fight.hp_in - kid.post_hp)
        elif res == "win":  # the run's last boss: the TERMINAL carries HP 0, the last decision counts
            hp_lost = max(0.0, fight.hp_in - frame_facts(cast(Frame, env.last_frame)).hp_frac)
        elif res in ("loss", "cap"):
            hp_lost = fight.hp_in
        c = gym.counters
        cc = c.cls(fight.cls)
        cc["decisions"] += env.decisions
        if kid.game:
            cc["game_decisions"] += env.decisions
        if hp_lost is not None:
            # the forked fight: won (fight_won, the run's last boss, or a GAME child that got past it) or a death
            won = res in ("fight_won", "win") or kid.passed
            died = not won
            if kid.game:
                cc["game_finished"] += 1
                cc["game_wins"] += res == "win"
                cc["game_deaths_first"] += died
                cc["game_deaths_later"] += kid.passed and res in ("loss", "cap")
            if fight.asc in TOP_ASCENSIONS:
                ca = c.asc(fight.cls, fight.asc)
                ca["fight_wins"] += won
                ca["game_wins"] += res == "win"
                ca["deaths"] += died
            cc["finished"] += 1
            cc["fight_wins"] += won
            cc["deaths"] += died
            cc["hp_lost"] += hp_lost
            for d in (c.keys, gym.keys):
                e = d.setdefault((fight.key, fight.asc), [0, 0, 0, 0.0])
                e[1] += won
                e[2] += died
                e[3] += hp_lost
        elif res == "lost":
            c.lost += 1
        fe = kid.fight_end
        return {"class": fight.cls, "mode": "game" if kid.game else "fight", "result": res, "hp_in": fight.hp_in,
                "hp_lost": hp_lost,
                "rounds": None if fe is None else [fe.round_fork, fe.round_end],
                "fight_end": None if fe is None else fe.outcome, "fork_step": kid.fork_step, "seed": kid.seed,
                "index": kid.index, "pair": kid.pair, "parent_env": self.envs[kid.parent_idx].env_id}

    def _fork_drop(self, env: _Env, detail: str, count: bool = True) -> None:
        """Give up the FORK in flight of parent ``env`` (refused by the host, or the episode ended first):
        counted and logged; the slots of the children that have not arrived are freed (one that still connects
        is closed by the pool; one that did arrive is a valid copy and plays on)."""
        gym, fork = cast(_Gym, self.gym), cast(_Fork, env.fork)
        env.fork = None
        fork.fight.k = 0
        for k in fork.kids:
            if k.fork is fork:
                k.job = None
                self._reap(k)
        if not count:
            return
        gym.counters.fork_failures += 1
        if self.on_error is not None:
            self.on_error({"kind": "gym_fork", "env": env.env_id, "key": fork.fight.key, "detail": detail,
                           "episode_id": fork.frame.episode_id, "step_id": fork.frame.step_id,
                           "seed": fork.job.seed, "lineup": fork.job.lineup, "ascension": fork.job.ascension,
                           "children": len(fork.seeds)})

    def _reap(self, env: _Env) -> None:
        """Free a child slot: the connection is closed (the child exits at EOF), late items become stale."""
        gym, kid = cast(_Gym, self.gym), cast(_Kid, env.kid)
        if env.fork is not None:  # it never arrived
            gym.pool.forget(env.fork.tag, kid.index)
            env.fork = None
        env.host.close()
        env.kid = None
        env.dead = True
        self._busy.discard(env.idx)
        gym.alive[kid.parent_idx] -= 1
        assert gym.alive[kid.parent_idx] >= 0
        gym.free.append(env.idx)

    def _gym_report(self, now: float) -> dict[str, Any]:
        """Practice since the previous report (spec 8): forks, children, failures, caps, the children's / scouts'
        decisions and rollout statistics, the gain, the class totals and the keys with the most children;
        ``alive`` = children from FORK to reap now; every 600 s the full difficulty table."""
        gym = cast(_Gym, self.gym)
        assert self.store is not None
        c, gym.counters = gym.counters, PracticeCounters()
        roll = {"gym": self.store.gym_stats.as_dict(), "scout": self.store.scout_stats.as_dict()}
        prev = self._gym_roll_prev
        roll_d = {w: {k: v - prev.get(w, {}).get(k, 0) for k, v in d.items()} for w, d in roll.items()}
        self._gym_roll_prev = roll  # type: ignore[assignment]
        table = gym.table
        top = []
        for (key, asc), (n_kids, wins, deaths, hp) in sorted(c.keys.items(), key=lambda kv: -kv[1][0])[:20]:
            e = table.table.get((key, asc), [0.0, 0.0, 0.0])
            top.append({"key": key, "ascension": asc, "n": e[0], "d": e[1], "h": e[2],
                        "s": table.s(key, key_class(key), asc),
                        "children": n_kids, "fight_wins": wins, "deaths": deaths, "hp_lost": hp})
        rep: dict[str, Any] = {
            "forks": c.forks, "children": c.children, "fork_failures": c.fork_failures, "capped": c.capped,
            "trimmed": c.trimmed, "started": c.started, "lost": c.lost, "decisions": c.decisions,
            "scout_decisions": c.scout_decisions, "alive": sum(gym.alive), "g": gym.ctl.g, "q": gym.ctl.q,
            "rollout": roll_d["gym"], "scout_rollout": roll_d["scout"], "host_us": c.host_us,
            "fork_ms": [None if x is None else x / 1e6 for x in percentiles(c.fork_ns, (50, 99))],
            "arrive_ms": [None if x is None else x / 1e6 for x in percentiles(c.arrive_ns, (50, 99))],
            "classes": c.classes, "classes_asc": c.classes_asc, "top": top}
        if now - gym.t_table >= 600.0 or self._stopping:
            gym.t_table = now
            kids = {f"{k}#{a}": v for (k, a), v in gym.keys.items()}
            rep["table"] = [{**row, "kids": kids.get(f"{row['key']}#{row['ascension']}", [0, 0, 0, 0.0])}
                            for row in table.dump()]
        return rep

    # ------------------------------------------------------------------ periodic work
    def _housekeeping(self) -> None:
        now = time.monotonic()
        if now >= self._t_deadline:
            self._t_deadline = now + self.cfg.deadline_interval
            for idx in [i for i in self._busy if now > self._all[i].deadline]:
                env = self._all[idx]
                try:
                    env.host.expire()
                except HostUnrecoverable as e:
                    self._retire(env, e)
                except HostError as e:
                    self._busy.discard(idx)
                    self._fail_episode(env, "lost", e)
        if self.policy.refresh():
            self.counters.weight_loads += 1
        if self.gym is not None:
            self.gym.ctl.tick(now, self.gym.child_total, self.counters.decisions)
        if now - self._t_report >= self.cfg.report_interval:
            self._report(now)

    def _host_totals(self) -> dict[str, int]:
        tot: dict[str, int] = {}
        for env in self.envs:
            for k, v in env.host.stats.as_dict().items():
                tot[k] = tot.get(k, 0) + v
        return tot

    def take_report(self) -> dict[str, Any]:
        """Everything accumulated since the previous report (also sent to ``on_report``)."""
        return self._report(time.monotonic(), emit=False)

    def _report(self, now: float, emit: bool = True) -> dict[str, Any]:
        dt = now - self._t_report
        self._t_report = now
        stage = self.clock.take()
        # loop overhead, housekeeping (back-pressure sleeps are reported separately as throttled_s)
        stage["other"] = max(0, int(dt * 1e9) - sum(stage.values()) - self._throttled_ns)
        host = self._host_totals()
        host_d = {k: v - self._host_prev.get(k, 0) for k, v in host.items()}
        self._host_prev = host
        cnt = self.counters.as_dict()
        cnt_d = {k: v - self._cnt_prev.get(k, 0) for k, v in cnt.items()}
        self._cnt_prev = cnt
        roll_d: dict[str, int] = {}
        if self.store is not None:
            roll = self.store.stats.as_dict()
            roll_d = {k: v - self._roll_prev.get(k, 0) for k, v in roll.items()}
            self._roll_prev = roll
        la, lh, lg = self._lat_actor, self._lat_host, self._lat_gpu
        self._lat_actor, self._lat_host, self._lat_gpu = [], [], []
        bs, self._batch_sizes = self._batch_sizes, []
        rep = {
            "seconds": dt, "decisions": cnt_d["decisions"], "stage_ns": stage, "counters": cnt_d,
            "host": host_d, "rollout": roll_d, "actions": self.actions.take(),
            "actor_latency_ms": [None if x is None else x / 1e6 for x in percentiles(la, (50, 99))],
            "host_latency_ms": [None if x is None else x / 1e6 for x in percentiles(lh, (50, 99))],
            "infer_ms": [None if x is None else x / 1e6 for x in percentiles(lg, (50, 99))],
            "batch_mean": float(np.mean(bs)) if bs else None, "batch_max": max(bs) if bs else None,
            "version": self.policy.version,
            "envs_alive": sum(not e.dead for e in self.envs), "envs": len(self.envs),
            "throttled_s": self._throttled_ns / 1e9,
            "sampling_version": getattr(self.source, "weights_version", None),
        }
        if self.taste is not None:
            rep["taste"] = self.taste.take()
        if self.rules_mode != "mask":
            rep["rules"] = {k: self.rules_counts[k] for k in RULES_KEYS}
            self.rules_counts.clear()
        if self.gym is not None:
            rep["practice"] = self._gym_report(now)
        if self.critic:  # CRIT of the trained frames (notes/critic/contract.md 4.4): rows seen / kept / truncated
            rep["critic"] = {k: self.critic_counts[k] for k in CRITIC_KEYS}
            self.critic_counts.clear()
        self._throttled_ns = 0
        if emit and self.on_report is not None:
            self.on_report(rep)
        return rep

    def _shutdown(self) -> None:
        self.inflight.clear()
        for env in self._all:
            if env.job is not None:
                self._end_episode(env, "shutdown", "actor stopped", store_reason="shutdown")
        self.ready = []
        if self.gym is not None:
            self.gym.pool.close()
        self._stopping = True
        if self.on_report is not None:
            self._report(time.monotonic())


_RESULT_NAMES = {int(Result.WIN): "win", int(Result.LOSS): "loss", int(Result.ABORTED): "aborted"}
_TRUNCATION_NAMES = {int(TerminalReason.DECISION_CAP): "decision_cap", int(TerminalReason.COMBAT_TURN_CAP): "turn_cap",
                     int(TerminalReason.OTHER_TRUNCATION): "truncated"}
_G_REASON, _G_ABORT = int(G.TERMINAL_REASON), int(G.ABORT_REASON)
_G_COMBATS, _G_ELITES, _G_BOSSES = int(G.COMBATS_WON), int(G.ELITES_WON), int(G.BOSSES_WON)


def terminal_result_name(result: int, reason: int) -> str:
    """:class:`~spire.metrics.EpisodeRecord` result of a TERMINAL frame: "win", "loss" (a death),
    "decision_cap" / "turn_cap" / "truncated" (RESULT=LOSS host truncations, errata-v1 PY-4: failures
    inside the win-rate denominator) or "aborted"."""
    if result == Result.LOSS and reason in _TRUNCATION_NAMES:
        return _TRUNCATION_NAMES[reason]
    return _RESULT_NAMES.get(result, "aborted")


_MSG_TERMINAL, _RES_LOSS, _DECISION_CAP = int(MsgType.TERMINAL), int(Result.LOSS), int(TerminalReason.DECISION_CAP)
_S_PILES = [int(SeatCol.HAND_COUNT), int(SeatCol.DRAW_COUNT), int(SeatCol.DISCARD_COUNT), int(SeatCol.EXHAUST_COUNT)]
_S_SEAT, _S_VISIBLE = int(SeatCol.SEAT), int(SeatCol.PRIVATE_VISIBLE)
_C_ZONE, _C_SEAT, _C_COUNT, _Z_PLAY = int(CardCol.ZONE), int(CardCol.SEAT), int(CardCol.COUNT), int(Zone.PLAY)


def seat_pile_totals(frame: Frame) -> np.ndarray:
    """Per SEATS row: the cards of that seat's own combat piles, HAND + DRAW + DISCARD + EXHAUST counts of its row
    plus its PLAY rows of CARDS; -1 for a seat whose private piles the frame hides (a teammate in SEAT view)."""
    seats = frame.table(Table.SEATS)
    assert (seats[:, _S_SEAT] == np.arange(len(seats))).all(), "SEATS rows in seat order"
    tot = seats[:, _S_PILES].sum(1, dtype=np.int64)
    cards = frame.table(Table.CARDS)
    play = cards[cards[:, _C_ZONE] == _Z_PLAY]
    np.add.at(tot, play[:, _C_SEAT], play[:, _C_COUNT])
    hidden = seats[:, _S_VISIBLE] == 0
    assert (tot[~hidden] >= 0).all(), "negative pile count of a visible seat"
    tot[hidden] = -1
    return tot


def _deadline(host: Host) -> float:
    """Monotonic deadline of the host's outstanding request (in-process hosts: none)."""
    t = host.reply_timeout()
    return time.monotonic() + t if t > 0 else float("inf")


def stage_us_per_decision(stage_ns: dict[str, int], decisions: int) -> dict[str, float]:
    """Per-stage microseconds per decision, plus ``python`` (all Python-side stages)."""
    d = max(1, decisions)
    out = {k: v / 1e3 / d for k, v in stage_ns.items()}
    out["python"] = sum(out.get(k, 0.0) for k in PYTHON_STAGES)
    return out


@dataclass
class RunTotals:
    """Convenience accumulator of actor reports (benchmarks, tests)."""

    decisions: int = 0
    seconds: float = 0.0
    stage_ns: dict[str, int] = field(default_factory=lambda: {s: 0 for s in STAGES})

    def add(self, rep: dict[str, Any]) -> None:
        self.decisions += rep["decisions"]
        self.seconds += rep["seconds"]
        for k, v in rep["stage_ns"].items():
            self.stage_ns[k] += v

    def us_per_decision(self) -> dict[str, float]:
        return stage_us_per_decision(self.stage_ns, self.decisions)
