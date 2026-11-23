"""Run instrumentation: append-only ``metrics.jsonl`` / ``episodes.jsonl`` and the aggregators
behind them (design-review-rl section 6).

Record kinds written to ``metrics.jsonl`` (one JSON object per line, ``kind`` + ``t`` unix
time + ``elapsed`` run seconds, cumulative across resumes):

``train``    every ``metrics_interval`` s by the supervisor: totals (decisions, finished games),
             throughput over the interval (decisions/s, games/hour, actor stage us/decision,
             inference / host / actor latency percentiles, batch sizes), robustness counters
             (aborted, overflow, lost, timeouts, crashes, ERROR frames, restarts, actor restarts),
             rolling outcomes per lineup x ascension (window of finished training games: win rate
             + Wilson 95%, mean final floor, death-floor / room histograms, top killers),
             action-type frequencies and behaviour entropy per decision kind, acting staleness.
``learner``  after every PPO update: losses, entropy (overall / per kind), approx KL, clip
             fraction (also epoch 0), explained variance (overall / per kind), staleness, timing,
             segment transfer cost.
``gpu``      nvidia-smi samples (utilisation, memory, temperature, power).
``checkpoint`` / ``event``  checkpoints written, actor restarts, stop reasons.

``episodes.jsonl`` holds one :class:`EpisodeRecord` per finished or failed training episode
(reproducibility: seed, lineup, ascension, result, final floor, death location, versions).
"""
from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .schema import Kind, Verb

N_KIND = max(int(k) for k in Kind) + 1
N_VERB = max(int(v) for v in Verb) + 1
FINISHED = ("win", "loss")  # the game reached a terminal state ("loss" = a death)
# host truncations (TERMINAL RESULT=LOSS with TERMINAL_REASON DECISION_CAP / COMBAT_TURN_CAP /
# OTHER_TRUNCATION, errata-v1 PY-4) and the actor's pile cap (ActorConfig.pile_cap): failures that count in every
# win-rate denominator
TRUNCATIONS = ("decision_cap", "turn_cap", "truncated", "pile_cap")
FAILURES = ("aborted", "overflow", "lost", "error", "shutdown", *TRUNCATIONS)
COUNTED = (*FINISHED, *TRUNCATIONS)  # results inside the win-rate denominator of training statistics
LONG_EPISODE_DECISIONS = 5000


# =========================================================================== statistics


def wilson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval for ``k`` successes out of ``n`` (``(0, 1)`` if n=0)."""
    if n <= 0:
        return 0.0, 1.0
    z = _z(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def wilson_lower(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided lower Wilson bound at confidence ``conf`` (formal "rate > target" claims)."""
    return wilson_interval(k, n, 2 * conf - 1)[0]


def _z(q: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9)."""
    if not 0 < q < 1:
        raise ValueError("quantile must be in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    lo = 0.02425
    if q < lo or q > 1 - lo:
        s = math.sqrt(-2 * math.log(q if q < lo else 1 - q))
        x = (((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) / \
            ((((d[0] * s + d[1]) * s + d[2]) * s + d[3]) * s + 1)
        return x if q < lo else -x
    r = q - 0.5
    t = r * r
    return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * r / \
        (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1)


def percentiles(values: Sequence[float] | np.ndarray, qs: Sequence[float] = (50, 99)) -> list[float | None]:
    if not len(values):
        return [None] * len(qs)
    return [float(x) for x in np.percentile(np.asarray(values, np.float64), qs)]


# =========================================================================== files


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _clean(o: Any) -> Any:
    """NaN / inf -> None (strict JSON)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class JsonlWriter:
    """Append-only JSON-lines file (one writer per file; flushed per record)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(_clean(dict(record)), default=_default, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.close()


class MetricsWriter(JsonlWriter):
    """``metrics.jsonl``: :meth:`emit` stamps ``kind``, ``t`` and ``elapsed`` (run seconds,
    offset by ``elapsed_offset`` so a resumed run continues its wall-clock axis)."""

    def __init__(self, path: str | Path, *, elapsed_offset: float = 0.0) -> None:
        super().__init__(path)
        self.t0 = time.time() - elapsed_offset

    def elapsed(self) -> float:
        return time.time() - self.t0

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        now = time.time()
        rec = {"kind": kind, "t": round(now, 3), "elapsed": round(now - self.t0, 3), **fields}
        self.write(rec)
        return rec


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Records of a JSON-lines file; a torn last line (writer killed mid-write) is skipped."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# =========================================================================== actor-side counters


class StageClock:
    """Nanosecond accumulators per actor stage; :meth:`take` returns and resets them."""

    def __init__(self, stages: Sequence[str]) -> None:
        self.stages = tuple(stages)
        self.ns = [0] * len(self.stages)

    def add(self, i: int, dt_ns: int) -> None:
        self.ns[i] += dt_ns

    def take(self) -> dict[str, int]:
        out = dict(zip(self.stages, self.ns))
        self.ns = [0] * len(self.stages)
        return out


class ActionStats:
    """Chosen action verb counts and behaviour entropy per decision kind."""

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.counts = np.zeros((N_KIND, N_VERB), np.int64)
        self.ent = np.zeros(N_KIND, np.float64)
        self.n = np.zeros(N_KIND, np.int64)

    def add(self, kinds: np.ndarray, verbs: np.ndarray, entropy: np.ndarray | None) -> None:
        np.add.at(self.counts, (kinds, verbs), 1)
        if entropy is not None:
            np.add.at(self.ent, kinds, entropy)
            np.add.at(self.n, kinds, 1)

    def take(self) -> dict[str, Any]:
        out = {"counts": self.counts.tolist(), "entropy_sum": self.ent.tolist(), "entropy_n": self.n.tolist()}
        self._reset()
        return out


def merge_action_stats(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-kind verb frequencies and mean behaviour entropy from :meth:`ActionStats.take` dicts."""
    counts = np.zeros((N_KIND, N_VERB), np.int64)
    ent = np.zeros(N_KIND)
    n = np.zeros(N_KIND, np.int64)
    for r in reports:
        counts += np.asarray(r["counts"], np.int64)
        ent += np.asarray(r["entropy_sum"])
        n += np.asarray(r["entropy_n"], np.int64)
    freq: dict[str, dict[str, float]] = {}
    for k in range(N_KIND):
        tot = int(counts[k].sum())
        if tot:
            freq[Kind(k).name] = {Verb(v).name: round(int(counts[k, v]) / tot, 4)
                                  for v in range(N_VERB) if counts[k, v]}
            freq[Kind(k).name]["_n"] = tot
    entropy = {Kind(k).name: float(ent[k] / n[k]) for k in range(N_KIND) if n[k]}
    return {"action_freq": freq, "entropy_by_kind": entropy}


# =========================================================================== episodes


@dataclass
class EpisodeRecord:
    """One training / evaluation episode as seen by the actor (public information only)."""

    lineup: str  # sorted character-name multiset, also the canonical seat order
    ascension: int
    result: str  # win | loss (a death) | decision_cap | turn_cap | truncated | pile_cap | aborted | overflow | lost
    #             | error | shutdown
    floor: int = 0  # TOTAL_FLOOR of the last frame
    act: int = 0
    room: str = ""  # room type of the last decision frame
    in_combat: bool = False
    enemies: tuple[str, ...] = ()  # monsters of the last combat frame (death location)
    decisions: int = 0
    auto_steps: int = 0
    seed: str = ""
    seed_id: str = ""  # panel entry / training seed tag
    env: int = -1
    version_first: int = -1
    version_last: int = -1
    seconds: float = 0.0
    detail: str = ""  # failure detail (exception text, error code)
    # host TERMINAL counters (errata-v1 section 5; -1 = no TERMINAL frame was received)
    terminal_reason: int = -1  # TerminalReason
    abort_reason: int = 0  # SpireHost.Driver.AbortReason of an aborted episode
    combats_won: int = -1
    elites_won: int = -1
    bosses_won: int = -1
    act1: str = ""  # act of the first decision frame (Act 1 variant: panel stratification)
    action_verbs: dict[str, int] = field(default_factory=dict)  # populated for episodes > 5,000 decisions
    decision_kinds: dict[str, int] = field(default_factory=dict)
    final_hp: tuple[int, ...] = ()  # latest received SEATS rows, terminal rows when available
    final_alive: tuple[bool, ...] = ()
    seat_deaths: int = 0  # observed alive -> dead transitions; automatic death/revive gaps are unobservable
    removal: dict[str, int] = field(default_factory=dict)  # removal rule on: its steps, removed cards by class
    rules: dict[str, int] = field(default_factory=dict)  # rules_mode teach / free: spire.actor.RULES_KEYS counts
    # practice fights (spire.gym): the encounter key of a child's fight ("" = a full game) and the rest of its
    # practice_episodes.jsonl record (class, result, hp_in, hp_lost, rounds, fork step / seed / index / pair). A
    # child's decisions / auto_steps / seconds / seat_deaths count from the fork. ``scout``: a scout's game (v1.1).
    # Neither ever enters the outcome windows (the supervisor writes them to practice_ / scout_episodes.jsonl).
    gym: str = ""
    scout: bool = False
    practice: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [name.strip() for name in self.lineup.split("+")]
        if self.final_hp or self.final_alive:
            assert len(self.final_hp) == len(self.final_alive) == len(names)
            order = sorted(range(len(names)), key=names.__getitem__)
            self.final_hp = tuple(self.final_hp[index] for index in order)
            self.final_alive = tuple(self.final_alive[index] for index in order)
        self.lineup = canonical_lineup(self.lineup)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not self.removal:  # removal rule off: the record of the tree without the rule
            del d["removal"]
        if not self.rules:  # rules_mode "mask": the record of the tree without rules modes
            del d["rules"]
        if not self.gym:  # the record of a full game is what it was before practice
            del d["gym"], d["practice"]
        if not self.scout:
            del d["scout"]
        return d

    @property
    def key(self) -> str:
        return lineup_key(self.lineup, self.ascension)


def canonical_lineup(lineup: str) -> str:
    """The multiset's sorted character names define both its key and its seat order."""
    names = [name.strip() for name in lineup.split("+")]
    assert 1 <= len(names) <= 4 and all(names), f"lineup needs 1-4 characters: {lineup!r}"
    return "+".join(sorted(names))


def lineup_key(lineup: str, ascension: int) -> str:
    return f"{canonical_lineup(lineup)}|A{ascension}"


def win_decision_percentiles(counts: Mapping[str, int]) -> dict[str, float | int | None]:
    """Exact linear percentiles from the cumulative win-length histogram, without expanding it."""
    if not counts:  # no won episodes yet
        return {"p50": None, "p99": None, "max": None}
    lengths = sorted(int(k) for k in counts)
    cumulative = np.cumsum([counts[str(k)] for k in lengths])
    ranks = (int(cumulative[-1]) - 1) * np.array([0.5, 0.99])
    low = np.asarray(lengths)[np.searchsorted(cumulative, np.floor(ranks), side="right")]
    high = np.asarray(lengths)[np.searchsorted(cumulative, np.ceil(ranks), side="right")]
    values = low + (high - low) * (ranks - np.floor(ranks))
    return {"p50": float(values[0]), "p99": float(values[1]), "max": lengths[-1]}


@dataclass
class _Window:
    finished: deque[tuple[bool, int, str, tuple[str, ...], str]]  # (win, floor, room, enemies, result)
    failures: Counter[str] = field(default_factory=Counter)
    total_finished: int = 0
    total_wins: int = 0
    decisions: int = 0  # every decision of every episode of this configuration (compute spent on it)
    episodes: int = 0


class OutcomeTracker:
    """Rolling training outcomes per lineup x ascension (DESIGN 3.3 / design-review T0).

    The win-rate window holds every episode that ended with a TERMINAL frame other than ABORTED:
    wins, deaths and host truncations (:data:`TRUNCATIONS`: decision cap / combat-turn cap - failures
    in the denominator, errata-v1 PY-4). Death statistics (floor / room / killers) use deaths only;
    truncations are also counted per reason in ``failures``, next to the episodes outside the
    denominator (aborted, overflow, lost, error, shutdown).
    """

    def __init__(self, window: int = 2000) -> None:
        self.window = window
        self.keys: dict[str, _Window] = {}

    def add(self, rec: EpisodeRecord) -> None:
        w = self.keys.get(rec.key)
        if w is None:
            w = self.keys[rec.key] = _Window(deque(maxlen=self.window))
        w.decisions += rec.decisions
        w.episodes += 1
        if rec.result in COUNTED:
            win = rec.result == "win"
            w.finished.append((win, rec.floor, rec.room, rec.enemies, rec.result))
            w.total_finished += 1
            w.total_wins += win
        if rec.result not in FINISHED:
            w.failures[rec.result] += 1

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        all_dec = max(1, sum(w.decisions for w in self.keys.values()))
        for key, w in sorted(self.keys.items()):
            n = len(w.finished)
            wins = sum(1 for x in w.finished if x[0])
            lo, hi = wilson_interval(wins, n)
            deaths = [x for x in w.finished if x[4] == "loss"]
            killers = Counter("+".join(x[3]) for x in deaths if x[3])
            out[key] = {
                "n": n, "wins": wins, "win_rate": wins / n if n else None, "lo": lo, "hi": hi,
                "mean_floor": float(np.mean([x[1] for x in w.finished])) if n else None,
                "deaths": len(deaths), "truncated": n - wins - len(deaths),
                "death_floor": dict(sorted(Counter(x[1] for x in deaths).items())),
                "death_room": dict(Counter(x[2] for x in deaths)),
                "top_killers": killers.most_common(8),
                "total_finished": w.total_finished, "total_wins": w.total_wins,
                "failures": dict(w.failures),
                "decisions": w.decisions, "decision_share": w.decisions / all_dec, "episodes": w.episodes,
            }
        return out


# =========================================================================== GPU sampling


def query_gpu(timeout: float = 5.0) -> list[dict[str, float]] | None:
    """One nvidia-smi sample per GPU, or ``None`` if nvidia-smi is unavailable."""
    fields = ("utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw")
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=timeout, check=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for i, line in enumerate(out.strip().splitlines()):
        vals = [v.strip() for v in line.split(",")]
        row: dict[str, float] = {"index": i}
        for k, v in zip(("util", "mem_used_mb", "mem_total_mb", "temp_c", "power_w"), vals):
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = float("nan")
        rows.append(row)
    return rows


class GpuSampler:
    """Background thread calling ``emit(samples)`` every ``interval`` seconds."""

    def __init__(self, emit: Callable[[list[dict[str, float]]], None], interval: float = 10.0) -> None:
        self.emit = emit
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)

    def start(self) -> "GpuSampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            rows = query_gpu()
            if rows is None:
                return
            self.emit(rows)
