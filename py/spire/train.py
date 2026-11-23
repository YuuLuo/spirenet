"""Training entry point: config file -> actor process(es) + learner process + supervisor.

Usage::

    python -B -m spire.train --config run.toml [--resume] [--duration 600]

Topology (``topology`` in the config):

``processes`` (default; multiprocessing ``spawn``)
    supervisor (this process): config, metrics aggregation (``metrics.jsonl``,
    ``episodes.jsonl``, ``errors.jsonl``), nvidia-smi sampling, watchdog, stop handling;
    ``actors`` actor processes, each driving ``hosts.envs_per_actor`` hosts with its own GPU
    inference; one learner process. Segments travel actor -> learner through one
    :class:`~spire.ipc.ShmRing` per actor (binary blobs in shared memory, a pipe carries only
    ``(offset, length)``); acting weights go learner -> actors as files
    (:class:`~spire.learner.FileWeightPublisher`, polled by a background thread in each actor);
    small messages (reports, episode records) go to the supervisor over one queue.
``threads``
    the same actor and learner loops as threads of one process sharing a
    :class:`~spire.rollout.SegmentRing` (no IPC); inference on a high-priority CUDA stream. For
    the single-process vs two-process GPU comparison; the watchdog cannot restart threads.

Stopping: ``duration_s`` / ``max_updates`` reached, a ``STOP`` file in the run directory, or
Ctrl+C: actors end their episodes (counted as ``shutdown``), the learner writes a final
checkpoint, the supervisor writes ``train_state.json``. ``--resume`` continues from
``ckpt/latest.pt`` (weights, optimiser, counters, RNG) and the saved totals (decision count,
wall-clock axis of the metrics). The watchdog restarts an actor process that died or stopped
heart-beating (its in-flight episodes are counted as lost); a dead learner stops the run.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import tomllib
import types
import typing
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np
import torch

from .actor import (
    CRITIC_KEYS,
    RULES_KEYS,
    Actor,
    ActorConfig,
    EpisodeDist,
    EpisodeSampler,
    TorchPolicy,
    stage_us_per_decision,
    validate_distribution,
)
from .advice import Advisor, parse_advice
from .gym import ASC_FIELDS, CLASS_FIELDS, GAME_CLASS_FIELDS, PracticeConfig
from .host import Host, HostConfig, HostProcess, real_host_spec
from .inputs2_migration import KEYS as INPUTS2_KEYS, config_before
from .ipc import Conn, RingSpec, ShmRing, take_ring_stats
from .kickstart import load_teacher
from .learner import FileWeightPublisher, FileWeightSource, Learner, PPOConfig, Prepared, _EXIT_CONFIG_DEFAULTS
from .metrics import (
    COUNTED,
    FAILURES,
    FINISHED,
    LONG_EPISODE_DECISIONS,
    EpisodeRecord,
    GpuSampler,
    JsonlWriter,
    MetricsWriter,
    OutcomeTracker,
    canonical_lineup,
    merge_action_stats,
    percentiles,
    win_decision_percentiles,
)
from .critic import CriticConfig, carries_hidden, training_reset_flags
from .model import PRESETS, ModelConfig, SpireNet, model_config_dict
from .pipes import new_mux
from .replica import normalize_replica_devices
from .rewards import RewardConfig, RoomIds
from .rollout import RolloutConfig, RolloutStore, Segment, SegmentRing
from .schema import RESET_FLAGS_DEFAULT, Kind, ResetFlag
from .synth import FakeHostConfig
from .sysutil import apply_placement, cuda_blocking_sync, gc_policy, set_gpu_priority
from .tokenize import Tokenizer, TokenizerConfig
from .taste import Taste, TasteStats, load_table
from .removal import CLASSES as REMOVAL_CLASSES, load_rule
from .vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab

REPO = Path(__file__).resolve().parents[2]
FAKE_HOST_SCRIPT = REPO / "py" / "scripts" / "fake_spirehost.py"
PANEL_DIR = REPO / "eval" / "panels"


# =========================================================================== configuration


@dataclass(frozen=True)
class HostSection:
    """Which hosts to run. ``real = true``: the real SpireHost as described by the launch spec
    ``real_config`` (a TOML path; empty = ``py/configs/real_host.toml`` / ``$SPIRE_REAL_HOST_CONFIG``,
    :func:`spire.host.real_host_spec`) plus ``extra_args`` (e.g. ``["--hp", "99999"]``). Otherwise
    ``command`` = an explicit host command line; empty = ``scripts/fake_spirehost.py`` with ``fake``."""

    command: tuple[str, ...] = ()
    real: bool = False
    real_config: str = ""
    extra_args: tuple[str, ...] = ()
    fake: FakeHostConfig = field(default_factory=lambda: FakeHostConfig(hashes=False))
    fake_latency_ms: float = 0.0
    envs_per_actor: int = 16
    start_timeout: float = 120.0
    reset_timeout: float = 30.0
    step_timeout: float = 10.0
    max_restarts: int = 5
    restart_window: float = 600.0
    transport: str = "stdio"  # "stdio" | "pipe" (named pipes on one completion port, notes/transport-spec.md)
    priority: str = ""  # host process priority class ("" = inherit; spire.sysutil.PRIORITY_CLASSES)
    cpus: str = ""  # host CPU affinity ("" = any, "p" / "e" = performance / efficiency cores, "0,1,10-13")


@dataclass(frozen=True)
class ModelSection:
    preset: str = "2.5m"
    override: ModelConfig | None = None  # full explicit config instead of the preset
    seat_symmetry: bool = False  # explicit Stage A architecture; requires checkpoint migration on warm start
    critic: CriticConfig | None = None  # [model.critic]: the training-only critic tower (notes/critic/contract.md 5)
    value_bins: int = 0  # p12: HL-Gauss distributional value head (0 = scalar value_head)
    value_vmin: float = -0.5
    value_vmax: float = 2.5
    value_sigma: float = 0.05
    char_experts: int = 0  # p12: character-routed FFN experts per block (0 = off)
    char_expert_ff: int = 0

    def config(self) -> ModelConfig:
        config = self.override if self.override is not None else PRESETS[self.preset]
        config = dataclasses.replace(config, seat_symmetry=True) if self.seat_symmetry else config
        assert self.critic is None or config.critic is None, "model.critic and model.override.critic: set one"
        config = dataclasses.replace(config, critic=self.critic) if self.critic is not None else config
        if self.value_bins != 0:
            assert self.override is None or self.override.value_bins == 0, \
                "model.value_bins and model.override.value_bins: set one"
            config = dataclasses.replace(config, value_bins=self.value_bins, value_vmin=self.value_vmin,
                                         value_vmax=self.value_vmax, value_sigma=self.value_sigma)
        else:
            assert (self.value_vmin, self.value_vmax, self.value_sigma) == (-0.5, 2.5, 0.05), \
                "model.value_vmin/vmax/sigma require model.value_bins > 0"
        if self.char_experts != 0 or self.char_expert_ff != 0:
            assert self.override is None or self.override.char_experts == 0, \
                "model.char_experts and model.override.char_experts: set one"
            config = dataclasses.replace(config, char_experts=self.char_experts, char_expert_ff=self.char_expert_ff)
        return config


@dataclass(frozen=True)
class ActorSection:
    actors: int = 1
    device: str = "cuda"
    devices: tuple[str, ...] = ()  # per-actor devices: actor slot k uses devices[k % len]; () = ``device`` for all
    use_graphs: bool = True
    temperature: float = 1.0
    max_batch: int = 128
    max_wait_ms: float = 1.0
    min_group: int = 16
    max_inflight: int = 2  # inference batches on the device at once (ActorConfig.max_inflight)
    report_interval: float = 5.0
    weight_poll_interval: float = 0.5
    reset_flags: int = int(RESET_FLAGS_DEFAULT)
    max_episode_decisions: int = 0  # per seat; RESET word 14 receives this times n_seats (0 = host default)
    pile_cap: int = 0  # cards in one seat's combat piles that end the episode as "pile_cap" (ActorConfig; 0 = off)
    priority: str = ""  # actor process priority class ("" = inherit; spire.sysutil.PRIORITY_CLASSES)
    cpus: str = ""  # actor CPU affinity ("" = any, "p" = performance cores, "0,1", spire.sysutil.parse_cpus)
    gpu_priority: str = ""  # WDDM GPU scheduling priority class of the actor process ("" = default "normal")
    switch_interval_ms: float = 0.0  # threads topology: sys.setswitchinterval (GIL hand-over; 0 = Python default)
    compile: bool = False  # torch.compile the inference forward replayed by the CUDA graphs (alt/speed)


@dataclass(frozen=True)
class LearnerSection:
    device: str = "cuda"
    allow_ppo_config_change: bool = False  # explicit resume-time algorithm change, weights/optimizer stay strict
    iteration_samples: int = 65536
    ring_capacity_samples: int = 1 << 20
    max_staleness: int | None = 4
    checkpoint_interval_s: float = 900.0
    milestone_every_updates: int = 0  # also keep ckpt/u<updates>.pt every N updates (0 = off)
    micro_tokens: int | None = 65536
    priority: str = ""  # learner process priority class ("" = inherit)
    cpus: str = ""  # learner CPU affinity ("" = any, "p" = performance cores, ...)
    gpu_priority: str = ""  # WDDM GPU scheduling priority class of the learner process ("" = default)
    blocking_sync: bool = True  # processes topology: CUDA waits block instead of spinning (frees 1-2 cores)
    fast: bool = True  # CUDA: asynchronous C-collation micro-steps (spire.trainstep)
    cuda_graphs: bool = True  # CUDA + fast: one CUDA graph per padded micro-batch shape
    compile: bool = False  # CUDA + fast: torch.compile the training forward (dynamic shapes; alt/speed: 1.5-1.7x)
    bag_atomic: bool = False  # atomic embedding-bag table gradient (model._BagSum; alt/speed: 1.1x)
    # back-pressure when the learner is the bottleneck (the segment backlog = samples received, not yet trained):
    #   "throttle"    actors stop launching inference while backlog >= backlog_high_iterations x iteration_samples
    #                 and resume at <= backlog_low_iterations x iteration_samples (hosts wait; no data is lost)
    #   "drop_oldest" the backlog is capped at backlog_high_iterations x iteration_samples, the oldest segments
    #                 are dropped (counted: ring.evicted_samples)
    #   "none"        backlog up to ring_capacity_samples; staleness bounded only by max_staleness (counted drops)
    backpressure: Literal["throttle", "drop_oldest", "none"] = "throttle"
    backlog_high_iterations: float = 2.0
    backlog_low_iterations: float = 1.0
    heartbeat_s: float = 900.0  # a learner loop silent this long (hung) stops the run like a dead learner
    # notes/next/pipeline-spec.md: a preparer thread takes and prepares (Learner.prepare: block, GAE, columns) the next
    # iteration while the card trains the current one; off = the sequential loop (config.json without the key)
    pipeline_prepare: bool = False
    # notes/next/dual-spec.md: "cuda:N" or ["cuda:1", "cuda:2", ...] = model replicas on those cards splitting every
    # minibatch's micro-batches across all 1 + N learner cards (spire.replica); "" = one learner card (omitted from
    # config.json)
    replica_device: str | tuple[str, ...] = ""
    # notes/next/round13-1119.md: the weights the actors act with, "online" (the trained model) or "ema" (its EMA,
    # ppo.ema_decay; the learner trains the online weights either way); "online" = config.json without the key
    acting_weights: Literal["online", "ema"] = "online"
    # notes/next/round21-1121.md: a taken sample whose behaviour probability is >= confident_tau goes into the
    # minibatches with probability confident_keep (Learner._kept_rows); iteration_samples still counts the TAKEN
    # samples; keep 1.0 = off (config.json without either key at its default)
    confident_keep: float = 1.0
    confident_tau: float = 0.95
    confident_exempt_kinds: tuple[int, ...] = ()  # p12: Kind ints always kept regardless of confident_keep

    def __post_init__(self) -> None:
        devs = normalize_replica_devices(self.replica_device)
        norm: str | tuple[str, ...] = "" if not devs else (devs[0] if len(devs) == 1 else devs)
        if norm != self.replica_device:
            object.__setattr__(self, "replica_device", norm)


@dataclass(frozen=True)
class EvalSection:
    """Periodic panel evaluation during training (a separate process with its own hosts; eval games never enter
    training data, panel seeds are excluded from training). Every round plays the panel until complete or its
    wall-clock budget expires, reporting completed episodes and whether the round is partial, with the
    latest checkpoint's ``weights`` (EMA by default) greedily (``temperature = 0``), at most every ``interval_s``
    and only while eval decisions stay below ``max_fraction`` of the training decisions since the run started.
    The panel ``panel`` (``eval/panels/<name>.json``) is created before training if missing, from
    ``panel_seeds`` CSPRNG seeds x ``lineups`` x ``ascensions`` (empty = the training distribution's)."""

    enabled: bool = False
    panel: str = ""
    panel_seeds: int = 128
    lineups: tuple[str, ...] = ()
    ascensions: tuple[int, ...] = ()
    interval_s: float = 1800.0
    first_after_s: float = 600.0
    envs: int = 4  # host processes of the evaluator (count them against the machine's host budget)
    max_fraction: float = 0.03
    weights: Literal["ema", "online"] = "ema"
    temperature: float = 0.0  # 0 = greedy
    device: str = "cpu"  # "cpu" keeps the evaluator off the GPU (no third WDDM context)
    max_episode_decisions: int = 0  # per seat; use the same backstop as actor.max_episode_decisions
    round_budget_s: float = 1200.0  # stop the round cleanly after this wall-clock budget
    priority: str = "below_normal"
    results_dir: str = ""  # default: <repo>/eval/results

    def __post_init__(self) -> None:
        assert self.max_episode_decisions >= 0
        assert self.round_budget_s > 0
        canonical = tuple(canonical_lineup(lineup) for lineup in self.lineups)
        assert len(set(canonical)) == len(canonical), "duplicate canonical evaluation lineups"
        object.__setattr__(self, "lineups", canonical)


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    run_root: str = str(REPO / "runs")
    topology: Literal["processes", "threads"] = "processes"
    seed: int = 0
    duration_s: float | None = None
    max_updates: int | None = None
    vocab: str = "synthetic"  # "synthetic", "host" (real-host launch spec) or a vocab.json path
    catalog: str = "synthetic"  # "synthetic", "host", "none" or a catalog.json path
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelSection = field(default_factory=ModelSection)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    episodes: EpisodeDist = field(default_factory=EpisodeDist)
    actor: ActorSection = field(default_factory=ActorSection)
    hosts: HostSection = field(default_factory=HostSection)
    learner: LearnerSection = field(default_factory=LearnerSection)
    shm_mb: int = 256  # shared-memory ring per actor
    metrics_interval_s: float = 30.0
    gpu_interval_s: float = 15.0
    outcome_window: int = 2000
    heartbeat_timeout_s: float = 120.0
    max_actor_restarts: int = 20
    eval: EvalSection = field(default_factory=EvalSection)
    # practice fights (spire.gym, notes/gym/practice-spec.md): the last practice.actors actor slots fork RESAMPLE
    # copies of their games at the first decision of every fight (by measured difficulty); off with actors = 0.
    # Practice hosts: practice.host_config (a --search server of the dist of hosts.real_config), stdio transport
    practice: PracticeConfig = field(default_factory=PracticeConfig)
    # live sampling weights: the supervisor re-reads this JSON file ({"lineups": [[name, w], ...], "ascensions":
    # [[asc, w], ...]}; relative paths are inside the run directory) every sampling_poll_s, validates it, logs the
    # change and publishes it to the actors (sampling.active.json); an invalid file is rejected and logged
    sampling_file: str = "sampling.json"
    sampling_poll_s: float = 30.0
    dashboard_interval_s: float = 0.0  # > 0: the supervisor rewrites <run>/dashboard.html this often (and at the end)
    run_start: int = 0  # set by the supervisor: number of starts of this run (1 = fresh); seeds the episode streams
    resume_from: str = ""  # --resume-from: checkpoint to resume from instead of ckpt/latest.pt (e.g. a crash-*.pt)
    migrate_public_history: bool = False  # one-time CLI migration; never persisted as a resume default
    migrate_seat_symmetry: bool = False  # one-time CLI migration; never persisted as a resume default
    migrate_coop_features: bool = False  # one-time Stage B migration; cleared in persisted resume config
    migrate_forecast: bool = False  # one-time forecast migration (notes/forecast); cleared in persisted resume config
    migrate_calc: bool = False  # one-time calc migration (notes/calc); cleared in persisted resume config
    migrate_inputs2: bool = False  # one-time inputs-v2a migration (notes/inputs2/v2a.md 3); cleared likewise
    # one-time rules-mode changes (spire.rules); cleared in persisted resume config
    migrate_rules: bool = False  # tokenizer.rules_mode "mask" -> "teach" (no kickstart teacher; weights untouched)
    rules_free: bool = False  # "teach" -> "free" (weights untouched; the fade starts)
    reset_seat_symmetry_optimizer: bool = False  # explicit ablation; requires the one-time migration
    # one-time critic migration (notes/critic/contract.md 5): adds the tower to a checkpoint without it; cleared in the
    # persisted resume config, and config.json has no such key (tree p4's)
    migrate_critic: bool = False

    @property
    def run_dir(self) -> Path:
        return Path(self.run_root) / self.run_name


def build(cls: Any, data: Any) -> Any:
    """Recursively build dataclass ``cls`` from plain data (TOML / JSON); unknown keys raise."""
    if data is None:
        return None
    if isinstance(cls, type) and dataclasses.is_dataclass(cls):
        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            raise TypeError(f"{cls.__name__}: expected a table, got {data!r}")
        hints = typing.get_type_hints(cls)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
        return cls(**{k: build(hints[k], v) for k, v in data.items()})
    origin = typing.get_origin(cls)
    args = typing.get_args(cls)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        return build(inner[0], data) if len(inner) == 1 else data
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(build(args[0], x) for x in data)
        return tuple(build(a, x) for a, x in zip(args, data))
    if cls is float and isinstance(data, int):
        return float(data)
    return data


def load_config(path: str | Path, **overrides: Any) -> TrainConfig:
    p = Path(path)
    raw = tomllib.loads(p.read_text(encoding="utf-8")) if p.suffix == ".toml" else json.loads(p.read_text("utf-8"))
    raw.update({k: v for k, v in overrides.items() if v is not None})
    cfg = build(TrainConfig, raw)
    check_forecast_flags(cfg)
    check_practice(cfg)
    check_critic(cfg)
    check_pipeline(cfg)
    check_dual(cfg)
    check_acting(cfg)
    check_confident(cfg)
    return cfg


def _card(name: str) -> int | None:
    """The CUDA index of a device name ("cuda" = 0, "cuda:N" = N), None for any other device."""
    d = torch.device(name)
    return (d.index or 0) if d.type == "cuda" else None


def check_dual(cfg: TrainConfig) -> None:
    """``learner.replica_device`` (notes/next/dual-spec.md) is for the plain production configuration of the main line
    (fast path, CUDA graphs, compile, any rules mode, inputs v2a, the pipelined prepare): it refuses, one message each,
    what it was not built for and any replica card that duplicates another or collides with another part of the run."""
    rds = normalize_replica_devices(cfg.learner.replica_device)
    if not rds:
        return
    p, ls = cfg.ppo, cfg.learner
    assert not p.kickstart.on, "learner.replica_device refuses an active kickstart (ppo.kickstart.on)"
    assert p.critic is None and cfg.model.config().critic is None, \
        "learner.replica_device refuses the critic (ppo.critic / model.critic)"
    assert not p.taste.on, "learner.replica_device refuses ppo.taste.on"
    assert not p.advice, "learner.replica_device refuses ppo.advice (advice and habit rules)"
    assert not p.recompute_values, "learner.replica_device refuses ppo.recompute_values"
    assert not p.decoupled_clip_anchor, "learner.replica_device refuses ppo.decoupled_clip_anchor"
    assert not p.normalize_returns, "learner.replica_device refuses ppo.normalize_returns"
    assert ls.fast, "learner.replica_device refuses the slow path (learner.fast false)"
    assert ls.cuda_graphs, "learner.replica_device refuses learner.cuda_graphs false"
    assert cfg.topology == "processes", f"learner.replica_device refuses topology {cfg.topology!r}"
    parsed: list[tuple[str, torch.device]] = []
    seen_cards: set[int] = set()
    for rd in rds:
        try:
            r = torch.device(rd)
        except RuntimeError as exc:
            raise AssertionError(f"learner.replica_device {rd!r}: a CUDA device cuda:N") from exc
        assert r.type == "cuda" and r.index is not None, f"learner.replica_device {rd!r}: a CUDA device cuda:N"
        assert r.index not in seen_cards, f"learner.replica_device {rd!r}: duplicate CUDA device in {list(rds)}"
        seen_cards.add(r.index)
        parsed.append((rd, r))
    learner_card = _card(ls.device)
    assert learner_card is not None, f"learner.replica_device refuses learner.device {ls.device!r} (not CUDA)"
    actor_devices = cfg.actor.devices or (cfg.actor.device,)
    actor_cards = {_card(d) for d in actor_devices}
    eval_card = _card(cfg.eval.device)
    for rd, r in parsed:
        assert r.index != learner_card, f"learner.replica_device {rd!r} is the learner's device {ls.device!r}"
        assert r.index not in actor_cards, \
            f"learner.replica_device {rd!r} is an actor's device {list(actor_devices)}"
        assert r.index != eval_card, f"learner.replica_device {rd!r} is the evaluator's device {cfg.eval.device!r}"


def check_acting(cfg: TrainConfig) -> None:
    """``learner.acting_weights`` "ema" (notes/next/round13-1119.md) is for the plain production configuration: the
    actors act with the EMA weights and store their log-probabilities and values; it refuses, one message each, what
    acts with other weights (kickstart: the teacher in phase 0) or re-derives the behaviour log-probability / values
    from the online weights, and the critic tower (whose stored values the contract defines as the online tower's)."""
    aw = cfg.learner.acting_weights
    assert aw in ("online", "ema"), f"learner.acting_weights {aw!r}: 'online' or 'ema'"
    if aw == "online":
        return
    p = cfg.ppo
    assert not p.kickstart.on, "learner.acting_weights 'ema' refuses an active kickstart (ppo.kickstart.on)"
    assert not p.recompute_values, "learner.acting_weights 'ema' refuses ppo.recompute_values"
    assert not p.decoupled_clip_anchor, "learner.acting_weights 'ema' refuses ppo.decoupled_clip_anchor"
    assert p.critic is None and cfg.model.config().critic is None, \
        "learner.acting_weights 'ema' refuses the critic (ppo.critic / model.critic)"


def check_confident(cfg: TrainConfig) -> None:
    """``learner.confident_keep`` < 1 (notes/next/round21-1121.md) is for the plain production configuration: it
    refuses, one message each, what trains on other targets than the stored behaviour of every sample (kickstart's
    distillation, values recomputed with the online weights, the critic tower's mixed advantages)."""
    ls = cfg.learner
    assert 0.0 < ls.confident_keep <= 1.0, f"learner.confident_keep {ls.confident_keep}: 0 < keep <= 1"
    assert 0.0 < ls.confident_tau <= 1.0, f"learner.confident_tau {ls.confident_tau}: 0 < tau <= 1"
    max_kind = max(int(k) for k in Kind)
    for k in ls.confident_exempt_kinds:
        assert isinstance(k, int) and not isinstance(k, bool) and 0 <= k <= max_kind, \
            f"learner.confident_exempt_kinds: invalid Kind {k!r}"
    if ls.confident_keep == 1.0:
        return
    p = cfg.ppo
    assert not p.kickstart.on, "learner.confident_keep < 1 refuses an active kickstart (ppo.kickstart.on)"
    assert not p.recompute_values, "learner.confident_keep < 1 refuses ppo.recompute_values"
    assert p.critic is None and cfg.model.config().critic is None, \
        "learner.confident_keep < 1 refuses the critic (ppo.critic / model.critic)"


def check_pipeline(cfg: TrainConfig) -> None:
    """``learner.pipeline_prepare`` (notes/next/pipeline-spec.md) is for the plain production configuration of the
    main line (fast path, CUDA graphs, compile, any rules mode, inputs v2a, the critic): it refuses, one message each,
    what reads or writes learner state in the prepare part, or what the option was not built for."""
    if not cfg.learner.pipeline_prepare:
        return
    p = cfg.ppo
    assert not p.recompute_values, "learner.pipeline_prepare refuses ppo.recompute_values"
    assert not p.decoupled_clip_anchor, "learner.pipeline_prepare refuses ppo.decoupled_clip_anchor"
    assert not p.normalize_returns, "learner.pipeline_prepare refuses ppo.normalize_returns"
    assert not p.kickstart.on, "learner.pipeline_prepare refuses an active kickstart (ppo.kickstart.on)"
    assert not p.taste.on, "learner.pipeline_prepare refuses ppo.taste.on"
    assert not p.advice, "learner.pipeline_prepare refuses ppo.advice (advice and habit offsets)"
    assert cfg.learner.fast, "learner.pipeline_prepare refuses the slow path (learner.fast false)"
    assert cfg.learner.device != "cpu", "learner.pipeline_prepare refuses the slow path (learner.device cpu)"
    assert cfg.topology == "processes", f"learner.pipeline_prepare refuses topology {cfg.topology!r}"


def check_critic(cfg: TrainConfig) -> None:
    """Start-up asserts of the critic tower (notes/critic/contract.md 1.2, 3, 5): ``ppo.critic`` iff ``model.critic``;
    RESET bit 10 never in ``actor.reset_flags`` (the training actor alone adds it, :func:`spire.critic.
    training_reset_flags`); real hosts of a critic run (and its practice hosts) are launch specs with
    ``--training-hidden``; no value recomputation (asserted by PPOConfig)."""
    assert not carries_hidden(cfg.actor.reset_flags), "actor.reset_flags must not carry bit 10 (HIDDEN, 1024)"
    on = cfg.ppo.critic is not None
    assert on == (cfg.model.config().critic is not None), "ppo.critic and model.critic come together"
    assert not cfg.migrate_critic or on, "--migrate-critic needs ppo.critic and model.critic"
    # kickstart phase 0: the teacher acts and has no tower, so no actor could store the tower's values (not built)
    assert not (on and cfg.ppo.kickstart.on and cfg.ppo.kickstart.phase == 0), \
        "ppo.critic refuses kickstart phase 0 (the teacher acts without a tower)"
    if on and cfg.hosts.real:
        assert "--training-hidden" in real_host_spec(cfg.hosts.real_config or None).command, \
            "a critic run's hosts.real_config must be a launch spec with --training-hidden"
        if cfg.practice.enabled:
            assert "--training-hidden" in real_host_spec(cfg.practice.host_config).command, \
                "a critic run's practice.host_config must be the fork-capable spec with --training-hidden"


def check_practice(cfg: TrainConfig) -> None:
    """Start-up asserts of practice fights (spec 2.2, 2.3 (c), 6.7): the identity of a child's numbers needs the
    per-fight GAE (macro_micro) and the post-fight sample; stdio transport, complete first frames; the practice
    hosts are the fork-capable launch spec of the same dist (same vocab / catalog files) as hosts.real_config."""
    pr = cfg.practice
    if not pr.enabled:
        assert pr.rloo_alpha == 0.0 and not pr.rloo_by_pair, \
            "practice.rloo_alpha / rloo_by_pair need practice.actors > 0"
        return
    assert cfg.ppo.gae.macro_micro, "practice requires ppo.gae.macro_micro"
    # notes/next/p2: the teacher's games are not forked in this version
    assert not (cfg.ppo.kickstart.on and cfg.ppo.kickstart.phase == 0), "practice is refused in kickstart phase 0"
    assert cfg.rollout.keep_bootstrap_sample, "practice requires rollout.keep_bootstrap_sample"
    assert cfg.hosts.transport == "stdio", "practice requires the stdio host transport"
    assert not cfg.actor.reset_flags & int(ResetFlag.ELIDE_UNCHANGED), "practice requires complete frames"
    pr.q_target(cfg.actor.actors)  # asserts 1 <= practice.actors <= actor.actors and 0.2 <= q_a <= 0.6
    # v1.2 B: a GAME child meets CARD_REWARD decisions; steering them is not specified
    assert not pr.game_classes or not cfg.ppo.taste.on, "practice.game_classes needs a run without taste"
    if cfg.hosts.real:
        assert pr.host_config, "practice.host_config: the fork-capable launch spec of hosts.real_config's dist"
        a, b = real_host_spec(cfg.hosts.real_config or None), real_host_spec(pr.host_config)
        dist = lambda spec: {str(Path(c).parent) for c in spec.command if c.endswith(".dll")}  # noqa: E731
        assert dist(a) == dist(b) and len(dist(a)) == 1, f"practice.host_config: another dist {dist(a)} {dist(b)}"
        assert (a.vocab, a.catalog) == (b.vocab, b.catalog), "practice.host_config: another vocab / catalog"
        assert "--search" in b.command and "--search" not in a.command, "practice hosts fork, plain hosts do not"
        assert "--allow-debug-options" not in b.command, "practice hosts never allow EXACT"
    else:
        assert not pr.host_config, "practice.host_config is a real-host launch spec"


def check_forecast_flags(cfg: TrainConfig) -> None:
    """The FORECAST frames and the forecast tokenizer come together (notes/forecast/contract.md 3.4)."""
    assert bool(cfg.actor.reset_flags & int(ResetFlag.FORECAST)) == cfg.tokenizer.forecast,         "actor.reset_flags must include 256 (FORECAST) iff tokenizer.forecast"
    # the calc trailers and the calc tokenizer come together; CALC needs TARGET_PREVIEWS (notes/calc/contract.md 3.4)
    assert bool(cfg.actor.reset_flags & int(ResetFlag.CALC)) == cfg.tokenizer.calc, \
        "actor.reset_flags must include 512 (CALC) iff tokenizer.calc"
    assert not cfg.tokenizer.calc or cfg.actor.reset_flags & int(ResetFlag.TARGET_PREVIEWS), \
        "tokenizer.calc needs actor.reset_flags bit 16 (TARGET_PREVIEWS)"
    # the map in fights (RESET bit 1) iff a network of the run has the route summary - the run's tokenizer; a kickstart
    # teacher's three inputs2 keys equal the run's (spire.kickstart.load_teacher, notes/inputs2/v2a.md 1) - and then no
    # network reads map tokens in fights (notes/inputs2/contract.md 1)
    map_in_combat = bool(cfg.actor.reset_flags & int(ResetFlag.MAP_IN_COMBAT))
    assert map_in_combat == cfg.tokenizer.route_summary, \
        "actor.reset_flags must include 2 (MAP_IN_COMBAT) iff tokenizer.route_summary"
    assert not map_in_combat or cfg.tokenizer.map_tokens != "frame", \
        "actor.reset_flags bit 2 (MAP_IN_COMBAT) needs tokenizer.map_tokens != 'frame'"



def config_dict(cfg: TrainConfig) -> dict[str, Any]:
    d: dict[str, Any] = json.loads(json.dumps(asdict(cfg)))
    if not cfg.practice.enabled:  # a run without practice has the config.json it had before practice existed
        del d["practice"]
    else:
        if cfg.practice.rloo_alpha == 0.0:
            del d["practice"]["rloo_alpha"]
        if not cfg.practice.rloo_by_pair:
            del d["practice"]["rloo_by_pair"]
    if cfg.model.critic is None:  # a run without the critic has the config.json of tree p4
        del d["model"]["critic"]
    if cfg.model.value_bins == 0:
        for k in ("value_bins", "value_vmin", "value_vmax", "value_sigma"):
            del d["model"][k]
    if cfg.model.char_experts == 0:
        for k in ("char_experts", "char_expert_ff"):
            del d["model"][k]
    if cfg.model.override is not None:
        if cfg.model.override.critic is None:
            del d["model"]["override"]["critic"]
        if cfg.model.override.value_bins == 0:
            for k in ("value_bins", "value_vmin", "value_vmax", "value_sigma"):
                del d["model"]["override"][k]
        if cfg.model.override.char_experts == 0:
            for k in ("char_experts", "char_expert_ff"):
                del d["model"]["override"][k]
    if cfg.ppo.critic is None:  # likewise ppo.critic and the critic migration flag
        del d["ppo"]["critic"]
    if not cfg.migrate_critic:
        del d["migrate_critic"]
    if not cfg.learner.pipeline_prepare:  # a run without the pipelined prepare has the config.json of tree p5
        del d["learner"]["pipeline_prepare"]
    if not cfg.learner.replica_device:  # a run on one learner card has the config.json of tree p7b
        del d["learner"]["replica_device"]
    if cfg.learner.acting_weights == "online":  # actors acting with the online weights: the config.json of tree p9
        del d["learner"]["acting_weights"]
    if cfg.learner.confident_keep == 1.0:  # every sample trained: the config.json of tree p10
        del d["learner"]["confident_keep"]
    if cfg.learner.confident_tau == 0.95:
        del d["learner"]["confident_tau"]
    if not cfg.learner.confident_exempt_kinds:
        del d["learner"]["confident_exempt_kinds"]
    if cfg.ppo.kickstart.floor == 0.0:  # a kickstart without a floor has the config.json of tree p6
        del d["ppo"]["kickstart"]["floor"]
    if cfg.ppo.shaping_end_scale == 1.0:
        del d["ppo"]["shaping_end_scale"]
    if cfg.ppo.shaping_fade_updates == 0:
        del d["ppo"]["shaping_fade_updates"]
    for k, v in _EXIT_CONFIG_DEFAULTS:
        if d["ppo"].get(k) == v:
            del d["ppo"][k]
    return d


GYM_ENV_ID_BASE = 1 << 20  # env ids of the practice child slots start here (hosts' envs: slot * envs_per_actor + i)


def gym_seed(cfg: TrainConfig, slot: int, incarnation: int) -> int:
    """The practice seed of one actor start: the root of its children's re-deal seeds, the copies' draws and the
    children's / scouts' action sampling. A function of the run seed, the actor slot and its start counters -
    never of a game."""
    return int(np.random.SeedSequence((cfg.seed, slot, incarnation, cfg.run_start, 0x67796D)).generate_state(
        1, np.uint64)[0])


@dataclass(frozen=True)
class ActorLayout:
    """What actor slot ``slot`` runs (spec 5): its host section and host count; for a practice actor (the last
    ``practice.actors`` slots) the practice config, the children's target share and the child slots."""

    hosts: HostSection
    n_hosts: int
    practice: PracticeConfig | None = None
    q_a: float = 0.0
    children: int = 0


def actor_layout(cfg: TrainConfig, slot: int) -> ActorLayout:
    pr, n = cfg.practice, cfg.actor.actors
    E = cfg.hosts.envs_per_actor
    if not pr.enabled or slot < n - pr.actors:
        return ActorLayout(cfg.hosts, E)
    sec = dataclasses.replace(cfg.hosts, real_config=pr.host_config) if cfg.hosts.real else cfg.hosts
    return ActorLayout(sec, pr.parent_envs(E, n), pr, pr.q_target(n), pr.children_cap(E, n))


# =========================================================================== shared builders


def load_vocab(cfg: TrainConfig) -> tuple[Vocab, Catalog | None]:
    """``vocab`` / ``catalog``: "synthetic", "host" (the files of the real-host launch spec,
    ``hosts.real_config``), a path; ``catalog`` may also be "none"."""
    spec = real_host_spec(cfg.hosts.real_config or None) if "host" in (cfg.vocab, cfg.catalog) else None
    if cfg.vocab == "synthetic":
        vocab = Vocab(make_synthetic_vocab())
    else:
        vocab = Vocab.load(spec.vocab if spec is not None and cfg.vocab == "host" else cfg.vocab)
    if cfg.catalog == "none":
        return vocab, None
    if cfg.catalog == "synthetic":
        return vocab, Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
    return vocab, Catalog.load(spec.catalog if spec is not None and cfg.catalog == "host" else cfg.catalog, vocab)


def build_net(cfg: TrainConfig, tok: Tokenizer, catalog: Catalog | None) -> SpireNet:
    return SpireNet(tok.layout, cfg.model.config(), catalog)


def resolved_vocab_paths(cfg: TrainConfig) -> tuple[str, str]:
    """``cfg.vocab`` / ``cfg.catalog`` with "host" replaced by the launch spec's files (checkpoint metadata must
    not depend on which launch spec is current when it is read)."""
    if "host" not in (cfg.vocab, cfg.catalog):
        return cfg.vocab, cfg.catalog
    spec = real_host_spec(cfg.hosts.real_config or None)
    return (str(spec.vocab) if cfg.vocab == "host" else cfg.vocab,
            str(spec.catalog) if cfg.catalog == "host" else cfg.catalog)


def milestone_index(updates: int, every: int) -> int:
    """Number of milestone boundaries (multiples of ``every`` updates) passed; 0 when milestones are off.

    A milestone checkpoint ``ckpt/u<updates>.pt`` is written at the first timed save after this index grew: saves
    are timed, so the update count is almost never an exact multiple (runs l1-first .. l1-third, which tested
    ``updates % every == 0``, kept no milestone at all)."""
    return updates // every if every else 0


def checkpoint_extra(cfg: TrainConfig, vocab: Vocab, catalog: Catalog | None) -> dict[str, Any]:
    """What :func:`spire.evalpanel.load_policy_net` needs to rebuild the network."""
    v, c = resolved_vocab_paths(cfg)
    return {"model_config": model_config_dict(cfg.model.config()),
            "tokenizer": json.loads(json.dumps(asdict(cfg.tokenizer))),
            "vocab": v, "catalog": c, "vocab_hash": vocab.hash,
            "catalog_hash": 0 if catalog is None else catalog.hash, "run": cfg.run_name}


def fake_host_command(fake: FakeHostConfig, seed: int, latency_ms: float = 0.0) -> tuple[str, ...]:
    args = [sys.executable, "-B", str(FAKE_HOST_SCRIPT), "--serve", "--seed", str(seed),
            "--win-prob", str(fake.win_prob), "--mean-decisions", str(fake.mean_decisions),
            "--abort-prob", str(fake.abort_prob), "--error-prob", str(fake.error_prob),
            "--end-turn-penalty", str(fake.end_turn_penalty), "--frame-pool", str(fake.frame_pool)]
    if fake.learnable:
        args.append("--learnable")
    if not fake.hashes:
        args.append("--no-hashes")
    if fake.size != "typical":
        args += ["--size", fake.size]
    if latency_ms:
        args += ["--latency-ms", str(latency_ms)]
    return tuple(args)


def make_hosts(sec: HostSection, n: int, *, seed: int, log_dir: Path, name: str,
               expected_hashes: tuple[int, int] | None = None, training_hidden: bool = False) -> list[Host]:
    """``n`` host processes (the real SpireHost, the configured command, or fake hosts).

    ``expected_hashes`` = (vocab hash, catalog hash) every frame header must carry (real hosts:
    pass the hashes of the vocab / catalog the tokenizer was built from). ``training_hidden``: the hosts accept
    RESET bit 10 (the Python lock of notes/critic/contract.md 1.2; only a critic run's training actor passes it)."""
    hosts: list[Host] = []
    env: Mapping[str, str] | None = None
    if sys.platform == "linux":  # each host holds 3-4 pipe fds here: the default soft limit 1024 ends near 250 hosts
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    if sec.real:
        spec = real_host_spec(sec.real_config or None)
        base: tuple[str, ...] = spec.command_with(*sec.extra_args)
        env = spec.process_env()
    else:
        base = (*sec.command, *sec.extra_args) if sec.command else ()
    mux = new_mux() if sec.transport == "pipe" else None  # shared by the hosts; closed with the last one
    try:
        for i in range(n):
            cmd = base or fake_host_command(sec.fake, seed * 1000 + i, sec.fake_latency_ms)
            hosts.append(HostProcess(HostConfig(
                command=tuple(cmd), env=env, start_timeout=sec.start_timeout, reset_timeout=sec.reset_timeout,
                step_timeout=sec.step_timeout, max_restarts=sec.max_restarts, restart_window=sec.restart_window,
                log_path=str(log_dir / f"{name}-h{i:02d}.stderr.log"), name=f"{name}-h{i:02d}",
                transport=sec.transport, expected_hashes=expected_hashes, priority=sec.priority, cpus=sec.cpus),
                mux=mux, training_hidden=training_hidden))
    except BaseException:
        for h in hosts:
            h.close()
        raise
    return hosts


def eval_panel_seeds(cfg: TrainConfig) -> frozenset[str]:
    """Seeds of the periodic-evaluation panel when it lives outside ``eval/panels`` (a path in ``eval.panel``)."""
    ev = cfg.eval
    if not ev.enabled or not ev.panel.endswith(".json") or not Path(ev.panel).exists():
        return frozenset()
    try:
        return frozenset(json.loads(Path(ev.panel).read_text(encoding="utf-8")).get("seeds", []))
    except (OSError, json.JSONDecodeError):
        return frozenset()


def panel_seeds(directory: Path = PANEL_DIR) -> frozenset[str]:
    """Every seed of every evaluation panel (excluded from training)."""
    out: set[str] = set()
    for p in sorted(directory.glob("*.json")) if directory.exists() else []:
        try:
            out.update(json.loads(p.read_text(encoding="utf-8")).get("seeds", []))
        except (OSError, json.JSONDecodeError):
            continue
    return frozenset(out)


class Beat:
    """Heartbeat cell for the threads topology (``multiprocessing.Value`` stand-in)."""

    def __init__(self, value: float | None = None) -> None:
        self.value = time.time() if value is None else value


class RingBacklog:
    """Threads topology: the learner's backlog read straight from the shared :class:`SegmentRing`."""

    def __init__(self, ring: SegmentRing) -> None:
        self.ring = ring

    @property
    def value(self) -> int:
        return self.ring.n_samples

    @value.setter
    def value(self, v: int) -> None:  # the learner's writes are redundant here
        pass


def backlog_limits(cfg: TrainConfig) -> tuple[int, int]:
    """(high, low) backlog marks in samples of the learner's back-pressure policy."""
    ls = cfg.learner
    high = max(1, int(ls.backlog_high_iterations * ls.iteration_samples))
    low = min(high, max(0, int(ls.backlog_low_iterations * ls.iteration_samples)))
    return high, low


def make_throttle(cfg: TrainConfig, backlog: Any) -> Callable[[], bool] | None:
    """The actor's back-pressure predicate (hysteresis between the high and low marks), or None."""
    if cfg.learner.backpressure != "throttle":
        return None
    high, low = backlog_limits(cfg)
    state = [False]

    def throttled() -> bool:
        b = backlog.value
        state[0] = b > low if state[0] else b >= high
        return state[0]
    return throttled


def select_device(name: str) -> torch.device:
    """The torch device ``name`` of a training process ("cuda" without CUDA: the CPU). An indexed CUDA device
    ("cuda:2") also becomes the process's current device: CUDA graph captures, events and default streams use
    the current device, so a process working on another GPU than cuda:0 must switch to it first."""
    device = torch.device(name if name != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    return device


def void(_: object) -> None:
    """Discard a value (adapts emitters to ``Callable[..., None]``)."""


def _parent_alive() -> bool:
    parent = mp.parent_process()
    return parent is None or parent.is_alive()


def _process_entry(fn: Any, *args: Any) -> None:
    """Child-process target: Ctrl+C reaches every process of the console; only the supervisor
    reacts to it (it sets the stop event), so children finish cleanly (final checkpoint)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    fn(*args)


# =========================================================================== actor process


def acting_policy(cfg: TrainConfig, tok: Tokenizer, catalog: Catalog | None, device: torch.device, slot: int,
                  incarnation: int) -> TorchPolicy:
    """The policy an actor acts with: the run's own network fed by the learner's published weights, or in kickstart
    phase 0 (notes/solo/kickstart-spec-v2.md 1) the TEACHER - the network of the teacher checkpoint's own model config
    with its EMA weights (asserted to be fed what the run feeds, :func:`spire.kickstart.load_teacher`), never replaced;
    the published versions only stamp the steps. Exploration, temperature, seed and graphs are the run's own."""
    a = cfg.actor
    ks = cfg.ppo.kickstart
    frozen = ks.on and ks.phase == 0
    if frozen:
        net = load_teacher(ks.teacher, tok, catalog, cfg.model.config())[0].to(device)
    else:
        net = build_net(cfg, tok, catalog).to(device)
    return TorchPolicy(net, tok, mode="sample", temperature=a.temperature, use_graphs=a.use_graphs,
                       weights=FileWeightSource(cfg.run_dir / "weights"), poll_interval=a.weight_poll_interval,
                       seed=(cfg.seed << 16) + (slot << 8) + incarnation,
                       high_priority=cfg.topology == "threads", exploration_by_kind=cfg.ppo.exploration_by_kind,
                       exploration_choice_uniform=cfg.ppo.exploration_choice_uniform, compile_model=a.compile,
                       frozen=frozen, combat_temperature=cfg.ppo.combat_temperature,
                       critic=cfg.ppo.critic is not None)


def actor_main(cfg_d: dict[str, Any], slot: int, incarnation: int, channel: tuple[RingSpec, Conn] | SegmentRing,
               msg_q: Any, stop: Any, beat: Any, backlog: Any = None) -> None:
    """One actor: E hosts, GPU inference, rollout recording -> learner (shm ring or shared ring)."""
    cfg = build(TrainConfig, cfg_d)
    check_practice(cfg)
    check_critic(cfg)
    torch.set_num_threads(1)  # intra-op worker spinning slows the single-threaded loop by ~40% (measured)
    a = cfg.actor
    gcp = gc_policy() if cfg.topology == "processes" else None  # settled at the 200th inference round
    placed = apply_placement(a.priority, a.cpus) if cfg.topology == "processes" else {}
    if cfg.topology == "processes" and a.gpu_priority:
        placed["gpu_priority"] = set_gpu_priority(a.gpu_priority)
    run_dir = cfg.run_dir
    vocab, catalog = load_vocab(cfg)
    tok = Tokenizer(vocab, cfg.tokenizer)
    # the critic tower (notes/critic/contract.md 1.2, 3): this training actor alone sets RESET bit 10, on hosts
    # made with training_hidden, and its tokenizer accepts (and never reads) the hidden table
    critic = cfg.ppo.critic is not None
    reset_flags = a.reset_flags
    if critic:
        assert cfg.ppo.critic is not None and cfg.model.config().critic is not None
        reset_flags = training_reset_flags(a.reset_flags)
        tok.allow_hidden = True
    device = select_device(a.devices[slot % len(a.devices)] if a.devices else a.device)
    torch.manual_seed(cfg.seed * 7919 + slot)
    if a.compile:
        import torch._dynamo as dynamo  # not "import torch._dynamo": that makes torch a local name of the function
        dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    policy = acting_policy(cfg, tok, catalog, device, slot, incarnation)
    ring: ShmRing | None = None
    hosts: list[Host] = []
    try:
        if not policy.wait_for_weights(timeout=600.0):
            raise RuntimeError("no weights published by the learner within 600 s")
        if isinstance(channel, SegmentRing):
            sink = channel.put
        else:
            ring = ShmRing.attach(*channel)
            put = ring.put

            def sink(seg: Segment) -> None:
                put(seg)
        E = cfg.hosts.envs_per_actor
        lay = actor_layout(cfg, slot)
        hosts = make_hosts(lay.hosts, lay.n_hosts, seed=(cfg.seed * 100 + slot) * 100 + incarnation,
                           log_dir=run_dir / "host-logs", name=f"a{slot}",
                           expected_hashes=(vocab.hash, catalog.hash if catalog is not None else 0)
                           if cfg.hosts.real else None, training_hidden=critic)
        old_advice, habit_advice = parse_advice(cfg.ppo.advice)
        crit_cfg = cfg.model.config().critic
        store = RolloutStore(sink, RoomIds.from_vocab(vocab), config=cfg.rollout,
                             rewards=dataclasses.replace(cfg.rewards, gamma=cfg.ppo.gae.gamma),
                             advice=bool(old_advice), judged=bool(habit_advice), taste=cfg.ppo.taste.on,
                             beh_temp=True, critic_k=crit_cfg.draw_k if critic and crit_cfg is not None else 0,
                             record_prog=cfg.ppo.shaping_fade_updates > 0 and cfg.ppo.shaping_end_scale != 1.0)
        # its own generator (the policy's random stream stays the one of a run without taste)
        # the taste hash key: with the env id and the env's episode count it names every episode of the run uniquely
        taste = (Taste(cfg.ppo.taste, vocab, (cfg.ppo.taste.seed, cfg.seed, cfg.run_start, slot, incarnation))
                 if cfg.ppo.taste.on else None)
        # the seed stream includes the run's start count: a resumed run never replays the episodes of an earlier
        # start (py-review 5.6)
        sampler = EpisodeSampler(cfg.episodes, vocab, seed=(cfg.seed, slot, incarnation, cfg.run_start),
                                 exclude=panel_seeds() | eval_panel_seeds(cfg),
                                 weights_file=run_dir / "sampling.active.json",
                                 poll_s=min(10.0, cfg.sampling_poll_s))
        acfg = ActorConfig(mode="train", max_batch=a.max_batch, max_wait_ms=a.max_wait_ms, min_group=a.min_group,
                           max_inflight=a.max_inflight,
                           reset_flags=reset_flags, report_interval=a.report_interval, env_id_base=slot * E,
                           max_episode_decisions=a.max_episode_decisions, yield_gil=cfg.topology == "threads",
                           pile_cap=a.pile_cap, gym_env_id_base=GYM_ENV_ID_BASE + slot * lay.children)
        assert not critic or acfg.mode == "train", "RESET bit 10 is a training actor's only"

        def on_report(rep: dict[str, Any]) -> None:
            rep["weight_loads"] = policy.weight_loads
            rep["weight_load_ms"] = policy.weight_load_ns / 1e6
            rep["pid"] = os.getpid()
            rep["placement"] = placed
            if ring is not None:
                rep["transfer"] = ring.stats.take()
            if device.type == "cuda":
                rep["cuda_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            if gcp is not None:
                rep["gc"] = {"frozen": gcp.frozen, **gcp.counters()}
            msg_q.put(("actor", slot, rep))

        actor = Actor(hosts, policy, tok, sampler, acfg, store=store,
                      on_episode=lambda r: msg_q.put(("episode", slot, r.as_dict())),
                      on_error=lambda d: msg_q.put(("error", slot, d)), on_report=on_report,
                      throttle=make_throttle(cfg, backlog) if backlog is not None else None, advice=cfg.ppo.advice,
                      catalog=catalog, taste=taste, practice=lay.practice, practice_q=lay.q_a,
                      practice_children=lay.children,
                      gym_seed=gym_seed(cfg, slot, incarnation) if lay.practice is not None else 0)
        t_check = [0.0]

        def should_stop() -> bool:
            now = time.time()
            beat.value = now
            if gcp is not None and gcp.pending and actor.counters.batches >= 200:  # warm-up over (the rounds come
                gcp.settle()  # after the first weights: wait_for_weights above)
            if now - t_check[0] > 1.0:
                t_check[0] = now
                if not _parent_alive():
                    return True
            return stop.is_set()

        actor.run(should_stop)
    finally:
        policy.close()
        for h in hosts:
            h.close()
        if ring is not None:
            ring.close()


# =========================================================================== learner process


def learner_main(cfg_d: dict[str, Any], channels: list[tuple[RingSpec, Conn]] | SegmentRing, msg_q: Any,
                 stop: Any, resume: bool, backlog: Any = None, beat: Any = None) -> None:
    """The learner; any exception is reported to the supervisor (``learner_error`` with the traceback) and a
    best-effort ``ckpt/crash-u<updates>.pt`` is written before the process exits non-zero (the supervisor then
    stops the run; ``--resume`` continues from ``ckpt/latest.pt``)."""
    state: dict[str, Any] = {}
    try:
        _learner_loop(cfg_d, channels, msg_q, stop, resume, backlog, beat, state)
    except BaseException as e:
        import traceback
        crash = ""
        lr = state.get("learner")
        if lr is not None and not isinstance(e, KeyboardInterrupt):
            try:
                path = Path(state["ckpt"]).with_name(f"crash-u{lr.updates:06d}.pt")
                lr.save(path, extra=state.get("extra"))
                crash = str(path)
            except Exception:  # noqa: BLE001 - best effort only
                crash = ""
        try:
            msg_q.put(("learner_error", 0, {"error": f"{type(e).__name__}: {e}"[:2000],
                                            "traceback": traceback.format_exc()[-8000:], "crash_checkpoint": crash,
                                            "updates": lr.updates if lr is not None else None}))
        except Exception:  # noqa: BLE001
            pass
        raise


def _learner_loop(cfg_d: dict[str, Any], channels: list[tuple[RingSpec, Conn]] | SegmentRing, msg_q: Any,
                  stop: Any, resume: bool, backlog: Any, beat: Any, state: dict[str, Any]) -> None:
    cfg = build(TrainConfig, cfg_d)
    check_pipeline(cfg)
    check_dual(cfg)
    check_acting(cfg)
    check_confident(cfg)
    ls = cfg.learner
    gcp = gc_policy() if cfg.topology == "processes" else None  # settled after the first completed update
    placed = apply_placement(ls.priority, ls.cpus) if cfg.topology == "processes" else {}
    if cfg.topology == "processes" and ls.gpu_priority:
        placed["gpu_priority"] = set_gpu_priority(ls.gpu_priority)
    rdevs = normalize_replica_devices(ls.replica_device)
    if cfg.topology == "processes" and ls.blocking_sync and rdevs:  # replicas' cards first: the learner's last
        bs_rep = [cuda_blocking_sync(torch.device(rd).index) for rd in rdevs]
        placed["blocking_sync_replica"] = bs_rep[0] if len(bs_rep) == 1 else bs_rep
    if cfg.topology == "processes" and ls.blocking_sync and ls.device != "cpu":
        placed["blocking_sync"] = cuda_blocking_sync(torch.device(ls.device).index or 0)  # before the CUDA context exists
    run_dir = cfg.run_dir
    vocab, catalog = load_vocab(cfg)
    tok = Tokenizer(vocab, cfg.tokenizer)
    torch.manual_seed(cfg.seed)
    device = select_device(ls.device)
    ppo = dataclasses.replace(cfg.ppo, micro_tokens=ls.micro_tokens) if ls.micro_tokens else cfg.ppo
    if ppo.exit_coef > 0.0 and ppo.exit_dir and not Path(ppo.exit_dir).is_absolute():
        ppo = dataclasses.replace(ppo, exit_dir=str(run_dir / ppo.exit_dir))
    net = build_net(cfg, tok, catalog)
    net.bag_atomic = ls.bag_atomic
    if ls.compile:
        import torch._dynamo as dynamo
        dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    teacher, teacher_sha256 = (load_teacher(ppo.kickstart.teacher, tok, catalog, cfg.model.config())
                               if ppo.kickstart.on else (None, ""))
    learner = Learner(net, tok, ppo, device=device,
                      size_fn=lambda s: tok.config.bucket(s.n_tokens),
                      publisher=FileWeightPublisher(run_dir / "weights", keep=3), fast=ls.fast,
                      graphs=ls.cuda_graphs, compile_model=ls.compile, teacher=teacher, teacher_sha256=teacher_sha256,
                      release_file=run_dir / "KICK_RELEASE" if ppo.kickstart.on else None,
                      replica_device=ls.replica_device or None, acting_weights=ls.acting_weights,
                      confident_keep=ls.confident_keep, confident_tau=ls.confident_tau, confident_seed=cfg.seed,
                      confident_exempt_kinds=ls.confident_exempt_kinds,
                      rloo_alpha=cfg.practice.rloo_alpha if cfg.practice.enabled else 0.0)
    ckpt = run_dir / "ckpt" / "latest.pt"
    src = Path(cfg.resume_from) if cfg.resume_from else ckpt
    if resume and cfg.resume_from and not src.exists():
        raise FileNotFoundError(f"--resume-from {src}: no such checkpoint")
    if cfg.migrate_public_history:
        assert resume and src.is_file(), "public-history migration requires an existing resume checkpoint"
    if cfg.migrate_seat_symmetry:
        assert resume and src.is_file(), "seat-symmetry migration requires an existing resume checkpoint"
    if cfg.migrate_coop_features:
        assert resume and src.is_file(), "co-op migration requires an existing resume checkpoint"
    if cfg.migrate_forecast:
        assert resume and src.is_file(), "forecast migration requires an existing resume checkpoint"
    if cfg.migrate_calc:
        assert resume and src.is_file(), "calc migration requires an existing resume checkpoint"
    if cfg.migrate_inputs2:
        assert resume and src.is_file(), "inputs2 migration requires an existing resume checkpoint"
    if cfg.migrate_rules or cfg.rules_free:
        assert resume and src.is_file(), "a rules-mode change requires an existing resume checkpoint"
    if cfg.migrate_critic:
        assert resume and src.is_file(), "critic migration requires an existing resume checkpoint"
    if resume and src.exists():
        ck_extra = learner.load(src, allow_config_change=cfg.learner.allow_ppo_config_change,
                                migrate_public_history=cfg.migrate_public_history,
                                migrate_seat_symmetry=cfg.migrate_seat_symmetry,
                                migrate_coop_features=cfg.migrate_coop_features,
                                migrate_forecast=cfg.migrate_forecast, migrate_calc=cfg.migrate_calc,
                                migrate_inputs2=cfg.migrate_inputs2, migrate_rules=cfg.migrate_rules,
                                rules_free=cfg.rules_free,
                                reset_seat_symmetry_optimizer=cfg.reset_seat_symmetry_optimizer,
                                migrate_critic=cfg.migrate_critic)
        # the inputs2 keys are the checkpoint's (older checkpoints: their defaults), or its migration's source state
        # (notes/inputs2/contract.md 1)
        saved_tok = build(TokenizerConfig, ck_extra["tokenizer"])
        want = config_before(cfg.tokenizer) if cfg.migrate_inputs2 else cfg.tokenizer
        assert [getattr(saved_tok, k) for k in INPUTS2_KEYS] == [getattr(want, k) for k in INPUTS2_KEYS], \
            f"{src}: tokenizer {INPUTS2_KEYS} {[getattr(saved_tok, k) for k in INPUTS2_KEYS]} != the run's " \
            f"{[getattr(want, k) for k in INPUTS2_KEYS]}" + ("" if cfg.migrate_inputs2 else " (--migrate-inputs2?)")
    extra = checkpoint_extra(cfg, vocab, catalog)
    state.update(learner=learner, ckpt=str(ckpt), extra=extra)
    if os.environ.get("SPIRE_TEST_LEARNER_FAIL_AT"):  # tests: make the learner die after N updates
        fail_at = int(os.environ["SPIRE_TEST_LEARNER_FAIL_AT"])
    else:
        fail_at = -1
    if ls.pipeline_prepare:
        assert learner.stepper is not None, \
            "learner.pipeline_prepare refuses the slow path (no fast-path stepper: no CUDA or no entity tokenizer)"
    learner.publish()
    msg_q.put(("learner_ready", 0, {"version": learner.version, "n_params": learner.model.n_params, "pid": os.getpid(),
                         "placement": placed}))
    rings: list[ShmRing] = []
    recv_stop = threading.Event()
    recv_thread: threading.Thread | None = None
    high, _low = backlog_limits(cfg)
    if isinstance(channels, SegmentRing):
        seg_ring = channels
        if ls.backpressure == "drop_oldest":
            seg_ring.capacity = high
    else:
        seg_ring = SegmentRing(high if ls.backpressure == "drop_oldest" else ls.ring_capacity_samples)
        rings = [ShmRing.attach(spec, conn) for spec, conn in channels]
        recv_thread = threading.Thread(target=_receive_segments, args=(rings, seg_ring, recv_stop, backlog),
                                       daemon=True, name="segment-receiver")
        recv_thread.start()
    last_ckpt = time.monotonic()
    t_wait = 0.0
    milestone = [milestone_index(learner.updates, ls.milestone_every_updates)]

    def save(reason: str, **more: Any) -> None:
        learner.save(ckpt, extra=extra)
        if milestone_index(learner.updates, ls.milestone_every_updates) > milestone[0]:
            milestone[0] = milestone_index(learner.updates, ls.milestone_every_updates)
            learner.save(ckpt.with_name(f"u{learner.updates:06d}.pt"), extra=extra)
        msg_q.put(("checkpoint", 0, {"path": str(ckpt), "version": learner.version, "updates": learner.updates,
                                     "samples_seen": learner.samples_seen, "reason": reason, **more}))

    prof = None
    if os.environ.get("SPIRE_PROFILE_LEARNER"):  # diagnostics: cProfile of the learner loop
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    pipe: Preparer | None = None
    try:
        if ls.pipeline_prepare:
            prep_fail_at = int(os.environ.get("SPIRE_TEST_PREPARE_FAIL_AT", "-1"))  # tests: the preparer dies
            pipe = Preparer(learner, seg_ring, ls.iteration_samples, ls.max_staleness, backlog, prep_fail_at)
        while not stop.is_set() and _parent_alive():
            if beat is not None:
                beat.value = time.time()
            t0 = time.perf_counter()
            if pipe is None:
                segs = seg_ring.take(ls.iteration_samples, current_version=learner.version,
                                     max_staleness=ls.max_staleness, timeout=0.5)
                t_wait += time.perf_counter() - t0
                if backlog is not None:
                    backlog.value = seg_ring.n_samples
                if segs is None:
                    continue
                if learner.updates == fail_at:
                    raise RuntimeError(f"injected learner failure at update {fail_at} (SPIRE_TEST_LEARNER_FAIL_AT)")
                m = learner.update(segs)
            else:
                got = pipe.get(timeout=0.5)
                t_wait += time.perf_counter() - t0
                if got is None:
                    continue
                if learner.updates == fail_at:
                    raise RuntimeError(f"injected learner failure at update {fail_at} (SPIRE_TEST_LEARNER_FAIL_AT)")
                m = learner.update_prepared(got.prepared)
                pipe.done()
                m.update(pipeline=True, prep_wait_s=got.prep_wait_s, handoff_wait_s=got.handoff_wait_s)
            if gcp is not None:
                gcp.settle()
                m["gc"] = gcp.fields()
            m["wait_s"] = t_wait
            t_wait = 0.0
            m["ring"] = seg_ring.stats.as_dict()
            m["ring_samples"] = seg_ring.n_samples
            m["backpressure"] = ls.backpressure
            m["acting_weights"] = ls.acting_weights
            if rings:
                m["transfer"] = take_ring_stats(rings)
            if device.type == "cuda":
                m["cuda_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            if rdevs:
                mems = [torch.cuda.max_memory_allocated(torch.device(rd)) / 2**20 for rd in rdevs]
                m["cuda_mem_mb_replica"] = max(mems)
                if len(mems) > 1:
                    m["cuda_mem_mb_replicas"] = mems
            msg_q.put(("learner", 0, m))
            if time.monotonic() - last_ckpt >= ls.checkpoint_interval_s:
                save("interval")
                last_ckpt = time.monotonic()
            if cfg.max_updates is not None and learner.updates >= cfg.max_updates:
                msg_q.put(("learner_done", 0, {"updates": learner.updates}))
                break
        if pipe is None:
            save("final")
        else:  # the preparer ends first; a batch it prepared (or was preparing) is dropped and counted
            dropped, pipe = pipe.close(), None
            save("final", pipeline=True, pipeline_dropped_samples=dropped)
    finally:
        if pipe is not None:  # an exception left the loop: end the preparer (its own pending error is raised too)
            pipe.close()
        if prof is not None:
            prof.disable()
            prof.dump_stats(os.environ["SPIRE_PROFILE_LEARNER"])
        recv_stop.set()
        if recv_thread is not None:
            recv_thread.join(timeout=5)
        for r in rings:
            r.close()


@dataclass
class Handover:
    """A prepared batch in the preparer's slot, with the preparer's time inside ``take`` for it and its time blocked
    on the full slot before that take."""

    prepared: Prepared
    prep_wait_s: float
    handoff_wait_s: float


class Preparer:
    """``learner.pipeline_prepare`` (notes/next/pipeline-spec.md): one thread of the learner process that takes the
    next iteration from the segment ring and runs :meth:`Learner.prepare` on it while the main thread trains the
    current one; a hand-over slot of capacity 1, so at most one batch is prepared ahead.

    The thread takes only while the slot is empty, with ``cv`` = the version the learner will have when it consumes
    the batch: the learner's version, + 1 while an update is in flight. The main thread changes the pair
    (``base``, ``in_flight``) under the lock (:meth:`get` takes a batch: in flight; :meth:`done`: the new version, not
    in flight), so the preparer never reads a torn pair; while the slot is empty no new update can start, so ``cv``
    stays right during the take. The ring's staleness eviction and the batch's staleness are therefore the sequential
    loop's. An exception in the thread is raised in the main thread at the next :meth:`get` (or at :meth:`close`)."""

    def __init__(self, learner: Learner, ring: SegmentRing, iteration_samples: int, max_staleness: int | None,
                 backlog: Any = None, fail_at: int = -1) -> None:
        self.learner, self.ring, self.backlog = learner, ring, backlog
        self.iteration_samples, self.max_staleness = iteration_samples, max_staleness
        self.fail_at = fail_at  # tests (SPIRE_TEST_PREPARE_FAIL_AT): the preparer raises before its batch fail_at
        self.lock = threading.Condition()
        self.slot: Handover | None = None
        self.base = learner.version
        self.in_flight = False
        self.closing = False
        self.closed = False
        self.error: BaseException | None = None
        self.raised = False
        self.handed = 0  # batches handed over
        learner.ensure_shaping_start()  # lock start_update before background prepare thread starts
        self.thread = threading.Thread(target=self._run, daemon=True, name="pipeline-preparer")
        self.thread.start()

    def _run(self) -> None:
        try:
            self._loop()
        except BaseException as e:  # noqa: BLE001 - handed to the main thread, raised there
            with self.lock:
                self.error = e
                self.lock.notify_all()

    def _loop(self) -> None:
        while True:
            t0 = time.perf_counter()
            with self.lock:
                self.lock.wait_for(lambda: self.slot is None or self.closing)
                if self.closing:
                    return
                cv = self.base + int(self.in_flight)
            t1 = time.perf_counter()
            segs = None
            while segs is None:
                if self.closing:
                    return
                segs = self.ring.take(self.iteration_samples, current_version=cv, max_staleness=self.max_staleness,
                                      timeout=0.5)
                if self.backlog is not None:
                    self.backlog.value = self.ring.n_samples
            t2 = time.perf_counter()
            if self.handed == self.fail_at:
                raise RuntimeError(f"injected preparer failure at batch {self.fail_at} (SPIRE_TEST_PREPARE_FAIL_AT)")
            prepared = self.learner.prepare(segs, cv)
            with self.lock:
                self.slot = Handover(prepared, t2 - t1, t1 - t0)
                self.handed += 1
                self.lock.notify_all()

    def _raise(self) -> None:
        assert self.error is not None
        self.raised = True
        raise self.error

    def get(self, timeout: float) -> Handover | None:
        """The prepared batch (the learner is in flight from here to :meth:`done`), or None after ``timeout``."""
        with self.lock:
            self.lock.wait_for(lambda: self.slot is not None or self.error is not None, timeout)
            if self.error is not None:
                self._raise()
            if self.slot is None:
                return None
            got, self.slot = self.slot, None
            assert not self.in_flight and self.base == self.learner.version
            self.in_flight = True
            self.lock.notify_all()
            return got

    def done(self) -> None:
        """The update of the batch of the last :meth:`get` is over: its version."""
        with self.lock:
            assert self.in_flight and self.learner.version == self.base + 1
            self.base = self.learner.version
            self.in_flight = False

    def close(self) -> int:
        """End the thread; returns the samples of a prepared batch never consumed (dropped). Raises the thread's
        error if the main thread has not raised it yet."""
        if not self.closed:
            with self.lock:
                self.closing = True
                self.lock.notify_all()
            self.thread.join()
            self.closed = True
        dropped = 0 if self.slot is None else len(self.slot.prepared)
        self.slot = None
        if self.error is not None and not self.raised:
            self._raise()
        return dropped


def _receive_segments(rings: list[ShmRing], out: SegmentRing, stop: threading.Event, backlog: Any = None) -> None:
    by_conn: dict[Any, ShmRing] = {r.conn: r for r in rings}
    while not stop.is_set():
        for conn in wait(list(by_conn), timeout=0.2):
            ring = by_conn[conn]
            c: Any = conn
            try:
                while c.poll():
                    out.put(ring.get(c.recv()))
                    if backlog is not None:  # at once: the actors' throttle reads it (a sweep can take seconds)
                        backlog.value = out.n_samples
            except (EOFError, OSError):
                continue
        if backlog is not None:
            backlog.value = out.n_samples


# =========================================================================== supervisor


@dataclass
class _ActorSlot:
    slot: int
    spec: RingSpec | None = None
    tx: Conn | None = None
    proc: Any = None
    beat: Any = None
    incarnation: int = 0
    started: float = 0.0


class Aggregator:
    """Supervisor-side metrics: folds actor / learner messages into ``metrics.jsonl`` records."""

    COUNTERS = ("timeouts", "crashes", "restarts", "errors", "bad_frames", "overflow_episodes", "abandoned")

    def __init__(self, cfg: TrainConfig, metrics: MetricsWriter, episodes: JsonlWriter, errors: JsonlWriter,
                 totals: Mapping[str, Any], practice_episodes: JsonlWriter | None = None,
                 scout_episodes: JsonlWriter | None = None) -> None:
        """``practice_episodes`` / ``scout_episodes``: where the records of practice children / scouts go (given
        iff practice is on / with scouts) - they are not training games, so ``episodes.jsonl``, the outcome
        windows and every games / episode counter never see them."""
        assert (practice_episodes is not None) == cfg.practice.enabled
        assert (scout_episodes is not None) == (cfg.practice.enabled and cfg.practice.scouts)
        self.cfg = cfg
        self.metrics, self.episodes, self.errors = metrics, episodes, errors
        self.practice_episodes, self.scout_episodes = practice_episodes, scout_episodes
        # cumulative practice totals (spec 8): per class CLASS_FIELDS; results of the children; scout results
        self.practice: dict[str, Any] = {"classes": {}, "results": {}, "scout_results": {}, "forks": 0,
                                         "children": 0, "fork_failures": 0, "capped": 0, "trimmed": 0, "lost": 0,
                                         "decisions": 0, "scout_decisions": 0, "classes_asc": {}}
        self.practice.update(totals.get("practice", {}))
        for c in self.practice["classes"].values():  # a v1.1 train_state.json lacks the v1.2 game counters
            missing = set(CLASS_FIELDS) - set(c)
            assert not missing or missing == set(GAME_CLASS_FIELDS), f"practice totals: class fields {missing}"
            c.update(dict.fromkeys(missing, 0))
        self.practice_tables: dict[str, Any] = {}  # actor slot -> its last difficulty table
        self.outcomes = OutcomeTracker(cfg.outcome_window)
        self.totals: dict[str, Any] = {"decisions": 0, "games": 0, "episodes_ended": 0, "version": 0, "updates": 0,
                                       "samples_seen": 0, "actor_restarts": 0, "actor_lost_episodes": 0, "starts": 0,
                                       "episode_decisions": 0, "long_episode_decisions": 0,
                                       "win_decision_counts": {},
                                       **{f: 0 for f in FAILURES}, **{c: 0 for c in self.COUNTERS}}
        self.totals.update({k: v for k, v in totals.items() if k in self.totals})
        self._reports: list[dict[str, Any]] = []
        self._t_emit = time.monotonic()
        self.last_learner: dict[str, Any] = {}
        self.learner_done = False
        self.learner_error: dict[str, Any] | None = None
        self.sampling: dict[str, Any] = {}
        self.last_eval: dict[str, Any] = {}
        self.eval_decisions = int(totals.get("eval_decisions", 0))
        self.taste = TasteStats(cfg.outcome_window) if cfg.ppo.taste.on else None  # spire.taste train-row table
        # removal rule: lineup -> [finished episodes, rule steps, removed curse / strike / defend / other]
        self.removal: dict[str, list[int]] | None = {} if cfg.tokenizer.removal_rule else None

    def handle(self, kind: str, slot: int, payload: dict[str, Any]) -> None:
        t = self.totals
        if kind == "episode" and payload.get("gym"):
            self._practice_episode(slot, payload)
        elif kind == "episode" and payload.get("scout"):
            assert self.scout_episodes is not None
            self.scout_episodes.write({"t": round(time.time(), 3), "actor": slot, **payload})
            r = self.practice["scout_results"]
            r[payload["result"]] = r.get(payload["result"], 0) + 1
        elif kind == "episode":
            rec = EpisodeRecord(**{**payload, "enemies": tuple(payload["enemies"])})
            self.outcomes.add(rec)
            self.episodes.write({"t": round(time.time(), 3), "actor": slot, **payload})
            t["episodes_ended"] += 1
            t["episode_decisions"] += rec.decisions
            if rec.decisions > LONG_EPISODE_DECISIONS:
                t["long_episode_decisions"] += rec.decisions
            if rec.result == "win":
                lengths = t["win_decision_counts"]
                length = str(rec.decisions)
                lengths[length] = lengths.get(length, 0) + 1
            if rec.result in COUNTED:  # games played to a TERMINAL frame: wins, deaths, host truncations
                t["games"] += 1
            if rec.result not in FINISHED:
                t[rec.result] = t.get(rec.result, 0) + 1
            elif self.removal is not None:
                r = self.removal.setdefault(rec.lineup, [0] * 6)
                for i, x in enumerate((1, rec.removal["steps"], *(rec.removal[c] for c in REMOVAL_CLASSES))):
                    r[i] += x
        elif kind == "actor":
            payload["slot"] = slot
            self._reports.append(payload)
            t["decisions"] += payload["decisions"]
            if self.taste is not None:
                self.taste.add(payload["taste"])
            for c in self.COUNTERS:
                t[c] += payload["host"].get(c, 0)
        elif kind == "learner":
            t["version"], t["updates"], t["samples_seen"] = payload["version"], payload["updates"], \
                payload["samples_seen"]
            self.last_learner = payload
            self.metrics.emit("learner", decisions=t["decisions"], **_learner_fields(payload))
        elif kind == "error":
            self.errors.write({"t": round(time.time(), 3), "actor": slot, **payload})
        elif kind == "checkpoint":
            self.metrics.emit("checkpoint", **payload)
            self.write_state()
        elif kind == "learner_done":
            self.learner_done = True
        elif kind == "learner_ready":
            self.metrics.emit("event", what="learner_ready", **payload)
        elif kind == "learner_error":
            self.learner_error = payload
            self.metrics.emit("event", what="learner_error", **payload)
        elif kind == "eval":
            self.last_eval = payload
            self.eval_decisions += int(payload.get("decisions", 0))
            self.metrics.emit("eval", **payload)
        elif kind == "eval_event":
            self.metrics.emit("event", what="eval", **payload)
        else:
            raise ValueError(f"unknown message kind {kind!r}")

    def _practice_episode(self, slot: int, payload: dict[str, Any]) -> None:
        """The record of a practice child (``gym`` = its encounter key): practice_episodes.jsonl and totals only."""
        assert self.practice_episodes is not None
        self.practice_episodes.write({"t": round(time.time(), 3), "actor": slot, **payload})
        r = self.practice["results"]
        res = payload["practice"]["result"]
        r[res] = r.get(res, 0) + 1

    def _practice_fields(self, reps: list[dict[str, Any]], dec: int, span: float) -> dict[str, Any]:
        """The practice part of a ``train`` row (spec 8): the children's share of the trained decisions (child
        decisions / (all decisions - scout decisions)), forks per hour, failures, caps, lost children; per practice
        actor its gain; cumulative per class totals with the health reading's rates; the interval's top keys."""
        prs = [r for r in reps if "practice" in r]
        P = self.practice
        q = [r["practice"] for r in prs]
        for x in q:
            for k in ("forks", "children", "fork_failures", "capped", "trimmed", "lost", "decisions",
                      "scout_decisions"):
                P[k] += x[k]
            for name, c in x["classes"].items():
                tot = P["classes"].setdefault(name, dict.fromkeys(CLASS_FIELDS, 0))
                for k, v in c.items():
                    tot[k] += v
            for name, c in x["classes_asc"].items():
                tot = P["classes_asc"].setdefault(name, dict.fromkeys(ASC_FIELDS, 0))
                for k, v in c.items():
                    tot[k] += v
        for r in prs:
            if "table" in r["practice"]:
                self.practice_tables[str(r["slot"])] = r["practice"]["table"]
        if self.practice_tables and any("table" in x for x in q):
            tmp = self.cfg.run_dir / "practice_table.json.tmp"
            tmp.write_text(json.dumps(self.practice_tables), encoding="utf-8")
            os.replace(tmp, self.cfg.run_dir / "practice_table.json")
        latest = [r for r in _latest_per_slot(reps) if "practice" in r]
        roll: dict[str, dict[str, int]] = {"gym": {}, "scout": {}}
        for x in q:
            for w, key in (("gym", "rollout"), ("scout", "scout_rollout")):
                for k, v in x[key].items():
                    roll[w][k] = roll[w].get(k, 0) + v
        top: dict[tuple[str, int], dict[str, Any]] = {}
        for x in q:
            for t in x["top"]:
                e = top.setdefault((t["key"], t["ascension"]), {"key": t["key"], "ascension": t["ascension"], "n": 0,
                                                                "d_n": 0.0, "h_n": 0.0, "s_n": 0.0, "children": 0,
                                                                "fight_wins": 0, "deaths": 0, "hp_lost": 0.0})
                for k in ("n", "children", "fight_wins", "deaths", "hp_lost"):
                    e[k] += t[k]
                e["d_n"] += t["d"] * t["n"]
                e["h_n"] += t["h"] * t["n"]
                e["s_n"] += t["s"] * max(t["n"], 1)
        top_rows = []
        for e in sorted(top.values(), key=lambda e: -e["children"])[:20]:
            n = e.pop("n")
            d_n, h_n, s_n = e.pop("d_n"), e.pop("h_n"), e.pop("s_n")
            top_rows.append({**e, "n": n, "d": d_n / n if n else None, "h": h_n / n if n else None,
                             "s": s_n / max(n, 1)})

        def rate(a: float, b: float) -> float | None:
            return a / b if b else None

        classes = {name: {**c, "death_rate": rate(c["deaths"], c["finished"]),
                          "hp_lost_mean": rate(c["hp_lost"], c["finished"]),
                          "forked_death_rate": rate(c["forked_deaths_w"], c["forked_w"]),
                          "forked_hp_lost_mean": rate(c["forked_hp_lost_w"], c["forked_w"]),
                          "parent_death_rate": rate(c["parent_deaths"], c["parent_fights"])}
                   for name, c in sorted(P["classes"].items())}

        def ms(name: str, i: int) -> float | None:
            vals = [x[name][i] for x in q if x[name][i] is not None]
            return float(np.mean(vals)) if vals else None

        child = sum(x["decisions"] for x in q)
        scout = sum(x["scout_decisions"] for x in q)
        forks = sum(x["forks"] for x in q)
        return {"practice_decision_share": child / (dec - scout) if dec - scout > 0 else None,
                "practice_forks_per_hour": forks / span * 3600 if span > 0 else 0.0,
                "practice_fork_failures": sum(x["fork_failures"] for x in q),
                "practice_capped": sum(x["capped"] for x in q), "practice_trimmed": sum(x["trimmed"] for x in q),
                "practice_lost": sum(x["lost"] for x in q),
                "practice": {"g": {str(r["slot"]): r["practice"]["g"] for r in latest},
                             "q": {str(r["slot"]): r["practice"]["q"] for r in latest},
                             "alive": sum(r["practice"]["alive"] for r in latest),
                             "k_mean": rate(sum(x["children"] for x in q), forks),
                             "decisions": child, "scout_decisions": scout,
                             "fork_ms_p50": ms("fork_ms", 0), "fork_ms_p99": ms("fork_ms", 1),
                             "arrive_ms_p50": ms("arrive_ms", 0), "arrive_ms_p99": ms("arrive_ms", 1),
                             "fork_host_ms": sum(x["host_us"] for x in q) / 1e3,
                             "rollout": roll["gym"], "scout_rollout": roll["scout"],
                             "totals": {k: P[k] for k in ("forks", "children", "fork_failures", "capped", "trimmed",
                                                          "lost", "decisions", "scout_decisions")},
                             "results": P["results"], "scout_results": P["scout_results"],
                             "classes": classes, "classes_asc": dict(sorted(P["classes_asc"].items())),
                             "top": top_rows}}

    def actor_restarted(self, slot: int, reason: str, lost_envs: int) -> None:
        self.totals["actor_restarts"] += 1
        self.totals["actor_lost_episodes"] += lost_envs
        self.metrics.emit("event", what="actor_restart", slot=slot, reason=reason, lost_episodes_bound=lost_envs)

    def maybe_emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._t_emit >= self.cfg.metrics_interval_s:
            self.emit_train(now - self._t_emit)
            self._t_emit = now

    def emit_train(self, interval: float) -> None:
        reps, self._reports = self._reports, []
        dec = sum(r["decisions"] for r in reps)
        reopens = sum(r["counters"]["card_offer_reopens"] for r in reps)
        cycle_decisions = sum(r["counters"]["cycle_ban_decisions"] for r in reps)
        stage = {k: sum(r["stage_ns"].get(k, 0) for r in reps) for k in (reps[0]["stage_ns"] if reps else {})}
        by_slot: dict[int, float] = {}
        for r in reps:
            by_slot[r["slot"]] = by_slot.get(r["slot"], 0.0) + r["seconds"]
        span = max(by_slot.values()) if by_slot else interval
        roll: dict[str, int] = {}
        for r in reps:
            for k, v in r.get("rollout", {}).items():
                roll[k] = roll.get(k, 0) + v
        games = roll.get("wins", 0) + roll.get("losses", 0) + roll.get("host_truncated", 0)
        transfer: dict[str, int] = {}
        for r in reps:
            for k, v in r.get("transfer", {}).items():
                transfer[k] = transfer.get(k, 0) + v
        t = self.totals
        version_lag = [t["version"] - r["version"] for r in reps]
        # spire.temperature: share of sampled actions that are not the arg-max of the scores, combat / other steps
        steer = {k: sum(r["counters"][k] for r in reps)
                 for k in ("combat_steps", "combat_nonargmax", "noncombat_steps", "noncombat_nonargmax")}
        practice = self._practice_fields(reps, dec, span) if self.cfg.practice.enabled else {}
        # R09b: the interval's decisions by origin: plain games, practice children, scouts (never trained)
        child = sum(r["practice"]["decisions"] for r in reps if "practice" in r)
        scout = sum(r["practice"]["scout_decisions"] for r in reps if "practice" in r)
        by_origin = {"plain": dec - child - scout, "child": child, "scout": scout}
        assert by_origin["plain"] >= 0, by_origin
        released = roll.get("steps_released", 0) + \
            (practice["practice"]["rollout"].get("steps_released", 0) if practice else 0)

        def p(name: str, i: int) -> float | None:
            vals = [r[name][i] for r in reps if r.get(name) and r[name][i] is not None]
            return float(np.mean(vals)) if vals else None

        self.metrics.emit(
            "train", decisions=t["decisions"], games=t["games"], episodes_ended=t["episodes_ended"],
            card_offer_reopens=reopens, card_offer_reopen_share=reopens / dec if dec else None,
            cycle_ban_decisions=cycle_decisions, cycle_ban_share=cycle_decisions / dec if dec else None,
            cycle_ban_all_banned=sum(r["counters"]["cycle_ban_all_banned"] for r in reps),
            cycle_ban_capacity_hits=sum(r["counters"]["cycle_ban_capacity_hits"] for r in reps),
            combat_nonargmax=steer["combat_nonargmax"] / steer["combat_steps"] if steer["combat_steps"] else None,
            noncombat_nonargmax=(steer["noncombat_nonargmax"] / steer["noncombat_steps"]
                                 if steer["noncombat_steps"] else None),
            combat_steps=steer["combat_steps"], noncombat_steps=steer["noncombat_steps"],
            sample_counts_by_kind=self.last_learner.get("sample_counts_by_kind"),
            sample_share_by_kind=self.last_learner.get("sample_share_by_kind"),
            sample_kind_version=self.last_learner.get("version"),
            version=t["version"], updates=t["updates"], samples_seen=t["samples_seen"],
            decisions_per_s=dec / span if span > 0 else 0.0, games_per_hour=games / span * 3600 if span > 0 else 0.0,
            decisions_by_origin=by_origin,
            decisions_per_s_trained=(dec - scout) / span if span > 0 else 0.0,
            stage_us=stage_us_per_decision(stage, dec) if dec else {},
            actor_latency_ms_p50=p("actor_latency_ms", 0), actor_latency_ms_p99=p("actor_latency_ms", 1),
            host_latency_ms_p50=p("host_latency_ms", 0), host_latency_ms_p99=p("host_latency_ms", 1),
            infer_ms_p50=p("infer_ms", 0), infer_ms_p99=p("infer_ms", 1),
            batch_mean=float(np.mean([r["batch_mean"] for r in reps if r["batch_mean"]]))
            if any(r["batch_mean"] for r in reps) else None,
            envs_alive=sum(r["envs_alive"] for r in _latest_per_slot(reps)),
            acting_staleness_mean=float(np.mean(version_lag)) if version_lag else None,
            transfer_us_per_step=transfer.get("encode_ns", 0) / 1e3 / max(1, released),
            transfer_bytes=transfer.get("bytes", 0), transfer_dropped=transfer.get("dropped", 0),
            weight_loads=sum(r.get("weight_loads", 0) for r in _latest_per_slot(reps)),
            weight_load_ms_total=sum(r.get("weight_load_ms", 0.0) for r in _latest_per_slot(reps)),
            actor_cuda_mem_mb=max((r.get("cuda_mem_mb", 0.0) for r in reps), default=None),
            actor_pids=sorted({r["pid"] for r in reps if r.get("pid")}),
            **({"actor_gc": actor_gc_fields(reps)} if any("gc" in r for r in reps) else {}),
            counts={k: t[k] for k in (*FAILURES, *self.COUNTERS, "actor_restarts", "actor_lost_episodes")},
            rollout=roll, outcomes=self.outcomes.summary(),
            episode_decisions=t["episode_decisions"], long_episode_decisions=t["long_episode_decisions"],
            long_episode_decision_share=t["long_episode_decisions"] / t["episode_decisions"]
            if t["episode_decisions"] else None,
            win_episode_decisions=win_decision_percentiles(t["win_decision_counts"]),
            throttled_frac=sum(r.get("throttled_s", 0.0) for r in reps) / max(1e-9, sum(by_slot.values()))
            if reps else 0.0,
            backlog_samples=self.last_learner.get("ring_samples"),
            sampling=self.sampling, eval_decisions=self.eval_decisions,
            eval_fraction=self.eval_decisions / max(1, t["decisions"]),
            **merge_action_stats([r["actions"] for r in reps]),
            **({"taste": self.taste.summary()} if self.taste is not None else {}),
            **({"removal": self.removal_summary()} if self.removal is not None else {}), **practice,
            **(rules_train_fields(reps) if self.cfg.tokenizer.rules_mode == "free" else {}),
            **(critic_train_fields(reps) if self.cfg.ppo.critic is not None else {}),
            **({"acting": "teacher" if self.cfg.ppo.kickstart.phase == 0 else "student"}
               if self.cfg.ppo.kickstart.on else {}),
        )

    def removal_summary(self) -> dict[str, Any]:
        """Per lineup, finished episodes since this supervisor started: rule steps and removed cards by class."""
        assert self.removal is not None
        return {k: {"episodes": r[0], "rule_steps": r[1], "rule_steps_per_episode": round(r[1] / r[0], 3),
                    "removed": dict(zip(REMOVAL_CLASSES, r[2:])),
                    "removed_per_episode": round(sum(r[2:]) / r[0], 3)}
                for k, r in sorted(self.removal.items())}

    def write_state(self) -> None:
        state = {**self.totals, "elapsed": self.metrics.elapsed(), "t": time.time(),
                 "eval_decisions": self.eval_decisions,
                 **({"practice": self.practice} if self.cfg.practice.enabled else {})}
        tmp = self.cfg.run_dir / "train_state.json.tmp"
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.cfg.run_dir / "train_state.json")


def rules_train_fields(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """rules_mode "free" (spire.rules): the interval's removal prompts and those whose sampled action was not the
    rule's row, split into another card / CANCEL / fewer than allowed (spire.actor.RULES_KEYS)."""
    tot = {k: sum(r["rules"][k] for r in reps) for k in RULES_KEYS}
    return {"rules_removal_prompts": tot["removal_prompts"], "rules_removal_other": tot["removal_other"],
            "rules_removal_cancel": tot["removal_cancel"], "rules_removal_fewer": tot["removal_fewer"]}


def critic_train_fields(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """A critic run (notes/critic/contract.md 4.4): the interval's CRIT counts of the training actors
    (spire.actor.CRITIC_KEYS): HDRAW rows dropped by K (``critic_draw_truncated``), mean HDRAW rows per combat
    frame, the share of frames given CRIT that are in a fight."""
    tot = {k: sum(r["critic"][k] for r in reps if "critic" in r) for k in CRITIC_KEYS}
    return {"critic_draw_truncated": tot["truncated"], "critic_frames": tot["frames"],
            "critic_hdraw_rows_per_combat_frame": tot["hdraw_rows"] / tot["combat_frames"] if tot["combat_frames"]
            else None}


def actor_gc_fields(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """The actors' collector counters (spire.sysutil.GcPolicy; cumulative per process): the latest report of each
    actor, summed, with the number of actors reporting and of those frozen."""
    last = [r["gc"] for r in _latest_per_slot(reps) if "gc" in r]
    return {"actors": len(last), "frozen": sum(g["frozen"] for g in last),
            **{k: round(sum(g[k] for g in last), 4) for k in ("gen1_n", "gen1_s", "gen2_n", "gen2_s")}}


def _latest_per_slot(reps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last: dict[int, dict[str, Any]] = {}
    for r in reps:
        last[r["slot"]] = r
    return list(last.values())


def _learner_fields(m: dict[str, Any]) -> dict[str, Any]:
    keep = ("loss_pi", "loss_v", "loss_win", "loss_floors", "loss_hp", "loss_ent_bonus", "loss_uniform_ce",
            "behavior_kl_before_update", "behavior_importance_p99", "behavior_importance_max",
            "gae_cut_segments", "gae_joined_cuts",
            "entropy", "entropy_norm", "entropy_base", "entropy_base_norm",
            "entropy_by_kind", "approx_kl", "clipfrac", "clipfrac_epoch0", "ratio_dev", "ratio_dev_epoch0",
            "first_minibatch_kl", "kl_lr_scale_mean", "kl_lr_scale_min",
            "explained_variance", "explained_variance_by_kind", "grad_norm", "nonfinite_skipped", "illegal_chosen",
            "lr", "samples", "version", "updates", "samples_seen", "staleness_mean", "staleness_max",
            "segment_staleness_hist", "return_mean", "value_mean", "reward_mean", "adv_std", "frac_single_action",
            "sample_counts_by_kind", "sample_share_by_kind",
            "time_prep", "time_train", "samples_per_s", "wait_s", "ring", "ring_samples", "transfer", "cuda_mem_mb",
            "early_stop", "epochs_done", "lag_s", "fast_path", "stepper", "advice", "taste", "calc_gate", "ahead_gate",
            "kickstart",
            "beh_temp_mean", "entropy_beh_combat", "entropy_base_combat", "combat_policy_samples", "rules", "critic",
            "pipeline", "prep_wait_s", "handoff_wait_s", "dual", "cuda_mem_mb_replica", "cuda_mem_mb_replicas", "gc", "acting_weights",
            "conf_keep", "conf_tau", "conf_taken", "conf_dropped", "kept", "conf_share_by_kind",
            "conf_exempt_kept", "rloo_groups", "rloo_children", "rloo_shift_std", "shaping_scale",
            "exit_loss", "exit_buffer_n", "exit_ingested", "exit_evicted", "exit_corrupt_skipped",
            "exit_target_prob_mean", "exit_top1_acc", "exit_step_kl", "exit_grad_norm", "exit_lr_scale",
            "exit_step_ms", "exit_live_n", "exit_legacy_n", "exit_live_share", "exit_live_top1_acc",
            "exit_active_frac")
    return {k: m[k] for k in keep if k in m}


class Supervisor:
    def __init__(self, cfg: TrainConfig, *, resume: bool = False) -> None:
        check_practice(cfg)
        check_critic(cfg)
        check_pipeline(cfg)
        check_dual(cfg)
        check_acting(cfg)
        check_confident(cfg)
        self.cfg = cfg
        self.resume = resume
        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Actors must wait for this learner's first publication, including when resuming an older checkpoint.
        weights = self.run_dir / "weights"
        (weights / "LATEST").unlink(missing_ok=True)
        for acting in weights.glob("acting-*.pt"):
            acting.unlink()
        state_path = self.run_dir / "train_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if resume and state_path.exists() else {}
        if not resume:
            for name in ("metrics.jsonl", "episodes.jsonl", "errors.jsonl", "train_state.json", "sampling.active.json",
                         "practice_episodes.jsonl", "scout_episodes.jsonl", "practice_table.json"):
                (self.run_dir / name).unlink(missing_ok=True)
        (self.run_dir / "STOP").unlink(missing_ok=True)
        (self.run_dir / "FAILED.json").unlink(missing_ok=True)
        # every start of the run gets its own episode seed streams (cfg.run_start), recorded before anything runs
        self.starts = int(state.get("starts", 0)) + 1
        self.cfg = cfg = dataclasses.replace(cfg, run_start=self.starts)
        # config.json = what --resume restarts from: a one-off --resume-from is not part of it
        (self.run_dir / "config.json").write_text(json.dumps(config_dict(dataclasses.replace(
            cfg, resume_from="", migrate_public_history=False, migrate_seat_symmetry=False,
            migrate_coop_features=False, migrate_forecast=False, migrate_calc=False, migrate_inputs2=False,
            migrate_rules=False, rules_free=False, reset_seat_symmetry_optimizer=False, migrate_critic=False)),
                                                             indent=1), encoding="utf-8")
        self.metrics = MetricsWriter(self.run_dir / "metrics.jsonl", elapsed_offset=float(state.get("elapsed", 0.0)))
        self.agg = Aggregator(cfg, self.metrics, JsonlWriter(self.run_dir / "episodes.jsonl"),
                              JsonlWriter(self.run_dir / "errors.jsonl"), state,
                              JsonlWriter(self.run_dir / "practice_episodes.jsonl") if cfg.practice.enabled else None,
                              JsonlWriter(self.run_dir / "scout_episodes.jsonl")
                              if cfg.practice.enabled and cfg.practice.scouts else None)
        self.agg.totals["starts"] = self.starts
        self.agg.write_state()
        self.stop_reason = ""
        self.slots: list[_ActorSlot] = []
        self.sampling = SamplingWeights(cfg, self.run_dir, self.metrics)
        if cfg.ppo.advice:  # unknown option / habit names fail at start-up, not in an actor
            Advisor(self.sampling.vocab, cfg.ppo.advice, load_vocab(cfg)[1])
        if cfg.ppo.taste.on:  # unknown character / card names fail at start-up, not in an actor
            version, table = load_table(cfg.ppo.taste.file, self.sampling.vocab)
            self.metrics.emit("event", what="taste_table", file=cfg.ppo.taste.file, version=version,
                              families={self.sampling.vocab["character"].entries[c]: [f.name for f in fams]
                                        for c, fams in table.items()})
        if cfg.tokenizer.removal_rule:  # a table of another vocabulary or without starter cards fails at start-up
            rule = load_rule(self.sampling.vocab, cfg.tokenizer.removal_ratings)
            self.metrics.emit("event", what="removal_rule", file=cfg.tokenizer.removal_ratings,
                              rated=int((rule.rating >= 0).sum()),
                              characters=sorted(rule.char_names[c] for c in rule.classes))
        self.agg.sampling = self.sampling.active
        self.failed: dict[str, Any] | None = None
        self._t_dash = time.monotonic()

    def request_stop(self, reason: str = "requested") -> None:
        self.stop_reason = self.stop_reason or reason

    def run(self) -> int:
        cfg = self.cfg
        self.metrics.emit("event", what="start", resume=self.resume, topology=cfg.topology, pid=os.getpid(),
                          actors=cfg.actor.actors, envs_per_actor=cfg.hosts.envs_per_actor)
        gpu = GpuSampler(lambda rows: void(self.metrics.emit("gpu", gpus=rows)), cfg.gpu_interval_s).start()
        main = threading.current_thread() is threading.main_thread()
        prev = signal.signal(signal.SIGINT, self._on_sigint) if main else None
        try:
            code = self._run_processes() if cfg.topology == "processes" else self._run_threads()
        finally:
            if main:
                signal.signal(signal.SIGINT, prev)
            gpu.stop()
        self.agg.maybe_emit(force=True)
        self.agg.write_state()
        if code != 0:
            self._write_failure(code)
        self.metrics.emit("event", what="stop", reason=self.stop_reason, code=code)
        for w in (self.metrics, self.agg.episodes, self.agg.errors, self.agg.practice_episodes,
                  self.agg.scout_episodes):
            if w is not None:
                w.close()
        if self.cfg.dashboard_interval_s > 0:
            self._dashboard()
        return code

    def _on_sigint(self, *_: object) -> None:
        self.stop_reason = self.stop_reason or "interrupt"

    def _write_failure(self, code: int) -> None:
        """``FAILED.json``: why the run stopped with an error and how to continue it."""
        err = self.agg.learner_error or {}
        cfg_path = self.run_dir / "config.json"
        info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "code": code, "reason": self.stop_reason,
                "learner_error": err.get("error"), "traceback": err.get("traceback"),
                "crash_checkpoint": err.get("crash_checkpoint") or None,
                "resume_from": str(self.run_dir / "ckpt" / "latest.pt") if (self.run_dir / "ckpt" / "latest.pt").exists()
                else None,
                "resume_command": f"python -B -m spire.train --config {cfg_path} --resume",
                "updates": self.agg.totals.get("updates"), "decisions": self.agg.totals.get("decisions")}
        self.failed = info
        tmp = self.run_dir / "FAILED.json.tmp"
        tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
        os.replace(tmp, self.run_dir / "FAILED.json")
        self.metrics.emit("event", what="fatal", reason=self.stop_reason, learner_error=err.get("error"),
                          crash_checkpoint=info["crash_checkpoint"], resume_from=info["resume_from"])

    def _poll_sampling(self) -> None:
        if self.sampling.poll():
            self.agg.sampling = self.sampling.active
        if self.cfg.dashboard_interval_s > 0 and time.monotonic() - self._t_dash >= self.cfg.dashboard_interval_s:
            self._dashboard()

    def _dashboard(self) -> None:
        self._t_dash = time.monotonic()
        try:
            from .dashboard import write_dashboard
            write_dashboard(self.run_dir)
        except Exception as e:  # noqa: BLE001 - a broken dashboard never stops training
            self.metrics.emit("event", what="dashboard_error", error=f"{type(e).__name__}: {e}"[:300])

    def _should_stop(self, t_start: float) -> bool:
        cfg = self.cfg
        self._poll_sampling()
        if not self.stop_reason:
            if cfg.duration_s is not None and time.monotonic() - t_start >= cfg.duration_s:
                self.stop_reason = "duration"
            elif self.agg.learner_done:
                self.stop_reason = "max_updates"
            elif (self.run_dir / "STOP").exists():
                self.stop_reason = "stop_file"
        return bool(self.stop_reason)

    def _drain(self, q: Any, timeout: float) -> None:
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            return
        while True:
            self.agg.handle(*msg)
            try:
                msg = q.get_nowait()
            except queue.Empty:
                return

    # ------------------------------------------------------------------ processes
    def _run_processes(self) -> int:
        cfg = self.cfg
        ctx = mp.get_context("spawn")
        stop = ctx.Event()
        msg_q = ctx.Queue()
        cfg_d = config_dict(cfg)
        backlog = ctx.Value("q", 0, lock=False)  # learner: samples received and not yet trained on
        lbeat = ctx.Value("d", time.time(), lock=False)  # learner loop heartbeat
        train_dec = ctx.Value("q", int(self.agg.totals["decisions"]), lock=False)  # for the evaluator's budget
        self._shared = {"backlog": backlog, "lbeat": lbeat, "train_decisions": train_dec}
        eval_panel = None
        if cfg.eval.enabled:  # before any actor starts: actors read the panel seeds to exclude them from training
            from .evalpanel import ensure_eval_panel
            eval_panel = ensure_eval_panel(cfg)
        owners: list[ShmRing] = []
        slots = self.slots
        readers: list[tuple[RingSpec, Conn]] = []
        for i in range(cfg.actor.actors):
            owner, spec = ShmRing.create(cfg.shm_mb << 20)
            rx, tx = ctx.Pipe(duplex=False)
            owners.append(owner)
            readers.append((spec, rx))
            slots.append(_ActorSlot(i, spec, tx, beat=ctx.Value("d", time.time(), lock=False)))
        learner = ctx.Process(target=_process_entry, args=(learner_main, cfg_d, readers, msg_q, stop, self.resume,
                                                           backlog, lbeat), name="learner")
        learner.start()
        for s in slots:
            self._spawn_actor(ctx, s, cfg_d, msg_q, stop)
        evaluator = self._spawn_evaluator(ctx, cfg_d, msg_q, stop, eval_panel)
        t_start = time.monotonic()
        code = 0
        try:
            while not self._should_stop(t_start):
                self._drain(msg_q, 0.25)
                train_dec.value = int(self.agg.totals["decisions"])
                self.agg.maybe_emit()
                if not learner.is_alive():
                    self._drain(msg_q, 1.0)  # the learner's last words (learner_error with the traceback)
                    err = self.agg.learner_error
                    self.stop_reason = f"learner died (exit code {learner.exitcode})" + \
                        (f": {err['error']}" if err else "")
                    code = 1
                    break
                if time.time() - lbeat.value > cfg.learner.heartbeat_s:
                    learner.kill()
                    self.stop_reason = f"learner hung: no heartbeat for {cfg.learner.heartbeat_s:.0f} s (killed)"
                    code = 1
                    break
                for s in slots:
                    self._watch(ctx, s, cfg_d, msg_q, stop)
                if evaluator is not None and not evaluator.is_alive() and evaluator.exitcode not in (0, None):
                    self.metrics.emit("event", what="eval", status="evaluator died", exit_code=evaluator.exitcode)
                    evaluator = None  # training continues without periodic evaluation
        finally:
            stop.set()
            deadline = time.monotonic() + 60
            procs = [s.proc for s in slots] + [learner] + ([evaluator] if evaluator is not None else [])
            while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
                self._drain(msg_q, 0.25)
            for p in procs:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=10)
            self._drain(msg_q, 0.5)
            for o in owners:
                o.close()
        return code

    def _spawn_actor(self, ctx: Any, s: _ActorSlot, cfg_d: dict[str, Any], msg_q: Any, stop: Any) -> None:
        assert s.spec is not None and s.tx is not None
        s.beat.value = time.time()
        s.proc = ctx.Process(target=_process_entry,
                             args=(actor_main, cfg_d, s.slot, s.incarnation, (s.spec, s.tx), msg_q, stop, s.beat,
                                   self._shared["backlog"]),
                             name=f"actor{s.slot}")
        s.proc.start()
        s.started = time.monotonic()

    def _spawn_evaluator(self, ctx: Any, cfg_d: dict[str, Any], msg_q: Any, stop: Any, panel: Any) -> Any:
        """The periodic panel evaluator (``eval.enabled``; ``panel`` created before the actors started), or None."""
        if panel is None:
            return None
        from .evalpanel import eval_main
        self.metrics.emit("event", what="eval", status="evaluator started", panel=panel.name, seeds=len(panel.seeds),
                          configs=[f"{lu}|A{a}" for lu, a in panel.configs], envs=self.cfg.eval.envs)
        proc = ctx.Process(target=_process_entry, args=(eval_main, cfg_d, msg_q, stop, self._shared["train_decisions"]),
                           name="evaluator")
        proc.start()
        return proc

    def _watch(self, ctx: Any, s: _ActorSlot, cfg_d: dict[str, Any], msg_q: Any, stop: Any) -> None:
        cfg = self.cfg
        hung = time.time() - s.beat.value > cfg.heartbeat_timeout_s and \
            time.monotonic() - s.started > cfg.heartbeat_timeout_s
        if s.proc.is_alive() and not hung:
            return
        reason = "heartbeat timeout" if s.proc.is_alive() else f"exit code {s.proc.exitcode}"
        if s.proc.is_alive():
            s.proc.kill()
        s.proc.join(timeout=10)
        if self.agg.totals["actor_restarts"] >= cfg.max_actor_restarts:
            self.stop_reason = f"actor {s.slot} failed ({reason}); restart budget exhausted"
            return
        self.agg.actor_restarted(s.slot, reason, actor_layout(cfg, s.slot).n_hosts)
        s.incarnation += 1
        self._spawn_actor(ctx, s, cfg_d, msg_q, stop)

    # ------------------------------------------------------------------ threads
    def _run_threads(self) -> int:
        cfg = self.cfg
        if cfg.eval.enabled:
            self.metrics.emit("event", what="eval", status="periodic evaluation needs the processes topology: off")
        if cfg.actor.switch_interval_ms > 0:
            sys.setswitchinterval(cfg.actor.switch_interval_ms / 1000.0)
        stop = threading.Event()
        msg_q: queue.Queue[Any] = queue.Queue()
        cfg_d = config_dict(cfg)
        ring = SegmentRing(cfg.learner.ring_capacity_samples)
        backlog = RingBacklog(ring)
        lbeat = Beat()
        errors: list[BaseException] = []

        def guarded(fn: Any, *args: Any) -> None:
            try:
                fn(*args)
            except BaseException as e:  # surfaced by the supervisor loop
                errors.append(e)
                stop.set()

        learner = threading.Thread(target=guarded, args=(learner_main, cfg_d, ring, msg_q, stop, self.resume, backlog,
                                                         lbeat), name="learner", daemon=True)
        learner.start()
        actors = [threading.Thread(target=guarded, args=(actor_main, cfg_d, i, 0, ring, msg_q, stop, Beat(), backlog),
                                   name=f"actor{i}", daemon=True) for i in range(cfg.actor.actors)]
        for a in actors:
            a.start()
        t_start = time.monotonic()
        try:
            while not self._should_stop(t_start) and not errors:
                self._drain(msg_q, 0.25)
                self.agg.maybe_emit()
        finally:
            stop.set()
            for th in [*actors, learner]:
                while th.is_alive():
                    self._drain(msg_q, 0.25)
            self._drain(msg_q, 0.1)
        if errors:
            self.stop_reason = self.stop_reason or f"thread failed: {errors[0]!r}"
            return 1
        return 0


class SamplingWeights:
    """The run's live sampling weights (``cfg.sampling_file``): created from the config's distribution if missing,
    re-read when it changes (at most every ``cfg.sampling_poll_s``), validated
    (:func:`spire.actor.validate_distribution`), logged as ``event`` records (``what="sampling"``, applied or
    rejected with the reason) and published to the actors as ``sampling.active.json`` (atomic replace, with a
    version number the actors report back)."""

    def __init__(self, cfg: TrainConfig, run_dir: Path, metrics: MetricsWriter) -> None:
        self.cfg = cfg
        self.metrics = metrics
        p = Path(cfg.sampling_file)
        self.path = p if p.is_absolute() else run_dir / p
        self.active_path = run_dir / "sampling.active.json"
        self.vocab, _ = load_vocab(cfg)
        self.active: dict[str, Any] = {}
        self._mtime = -1.0
        self._t_poll = 0.0
        prev = {}
        if self.active_path.exists():
            try:
                prev = json.loads(self.active_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prev = {}
        self.version = int(prev.get("version", 0))
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "comment": "training sampling weights; edit and save - the supervisor applies valid changes within "
                           f"{cfg.sampling_poll_s:.0f} s (see notes/l1-runbook.md)",
                "lineups": [list(x) for x in cfg.episodes.lineups],
                "ascensions": [list(x) for x in cfg.episodes.ascensions]}, indent=1), encoding="utf-8")
        if not self.poll(force=True):  # an invalid file at start: run with the config's distribution
            self._publish(cfg.episodes, source="config")

    def poll(self, force: bool = False) -> bool:
        """Apply the weights file if it changed; True if a new distribution was published."""
        now = time.monotonic()
        if not force and now - self._t_poll < self.cfg.sampling_poll_s:
            return False
        self._t_poll = now
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        if mtime == self._mtime:
            return False
        self._mtime = mtime
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            dist = validate_distribution(data, self.vocab, self.cfg.episodes)
        except (OSError, ValueError) as e:  # json.JSONDecodeError is a ValueError
            self.metrics.emit("event", what="sampling", applied=False, file=str(self.path), error=str(e)[:500],
                              active=self.active)
            return False
        self._publish(dist, source=str(self.path))
        return True

    def _publish(self, dist: EpisodeDist, source: str) -> None:
        self.version += 1
        self.active = {"version": self.version, "lineups": [list(x) for x in dist.lineups],
                       "ascensions": [list(x) for x in dist.ascensions],
                       "lineup_ascensions": [[name, [list(x) for x in ascs]] for name, ascs in dist.lineup_ascensions],
                       "source": source, "t": time.time()}
        tmp = self.active_path.with_name(self.active_path.name + ".tmp")
        tmp.write_text(json.dumps(self.active, indent=1), encoding="utf-8")
        os.replace(tmp, self.active_path)
        self.metrics.emit("event", what="sampling", applied=True, **self.active)


# =========================================================================== CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SpireHost PPO training")
    ap.add_argument("--config", required=True, help="TOML or JSON training config")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/ckpt/latest.pt")
    ap.add_argument("--resume-from", default=None, help="with --resume: this checkpoint instead (e.g. ckpt/crash-*.pt)")
    ap.add_argument("--migrate-public-history", action="store_true",
                    help="once, with --resume: add zero public-history weights and preserve existing Adam states")
    ap.add_argument("--migrate-seat-symmetry", action="store_true",
                    help="once, with --resume: copy seat-0 weights and Adam histories for TEAM warm start")
    ap.add_argument("--migrate-coop-features", action="store_true",
                    help="once, with --resume: add zero seat-screen/status/vote weights and preserve Adam")
    ap.add_argument("--migrate-forecast", action="store_true",
                    help="once, with --resume: add the zero-gated move-forecast input and preserve Adam")
    ap.add_argument("--migrate-calc", action="store_true",
                    help="once, with --resume: add the zero-gated turn-end / per-enemy / route input and preserve Adam")
    ap.add_argument("--migrate-inputs2", action="store_true",
                    help="once, with --resume: add the second-boss table and the zero-gated route summary "
                         "(notes/inputs2/v2a.md 3) and preserve Adam")
    ap.add_argument("--migrate-rules", action="store_true",
                    help="once, with --resume: tokenizer.rules_mode mask -> teach (spire.rules; no kickstart teacher)")
    ap.add_argument("--rules-free", action="store_true",
                    help="once, with --resume: tokenizer.rules_mode teach -> free (spire.rules)")
    ap.add_argument("--migrate-critic", action="store_true",
                    help="once, with --resume: add the critic tower (notes/critic/contract.md 5; model.critic and "
                         "ppo.critic set) and preserve every old weight and Adam state")
    ap.add_argument("--reset-seat-symmetry-optimizer", action="store_true",
                    help="with --migrate-seat-symmetry: discard all Adam moments and steps")
    ap.add_argument("--duration", type=float, default=None, help="override duration_s")
    ap.add_argument("--max-updates", type=int, default=None, help="override max_updates")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, duration_s=a.duration, max_updates=a.max_updates)
    if a.migrate_public_history:
        if not a.resume or not cfg.tokenizer.public_history:
            ap.error("--migrate-public-history needs --resume and tokenizer.public_history=true")
        cfg = dataclasses.replace(cfg, migrate_public_history=True)
    if a.migrate_seat_symmetry:
        if not a.resume or not cfg.model.config().seat_symmetry:
            ap.error("--migrate-seat-symmetry needs --resume and model.seat_symmetry=true")
        cfg = dataclasses.replace(cfg, migrate_seat_symmetry=True)
    if a.migrate_coop_features:
        if not a.resume or not (cfg.tokenizer.coop_features or cfg.tokenizer.mend_target):
            ap.error("--migrate-coop-features needs --resume and tokenizer.coop_features=true or mend_target=true")
        cfg = dataclasses.replace(cfg, migrate_coop_features=True)
    if a.migrate_forecast:
        if not a.resume or not cfg.tokenizer.forecast:
            ap.error("--migrate-forecast needs --resume and tokenizer.forecast=true")
        cfg = dataclasses.replace(cfg, migrate_forecast=True)
    if a.migrate_calc:
        if not a.resume or not cfg.tokenizer.calc:
            ap.error("--migrate-calc needs --resume and tokenizer.calc=true")
        cfg = dataclasses.replace(cfg, migrate_calc=True)
    if a.migrate_inputs2:
        if not a.resume or not cfg.tokenizer.route_summary or not cfg.tokenizer.second_boss_table:
            ap.error("--migrate-inputs2 needs --resume and tokenizer.route_summary / second_boss_table = true")
        cfg = dataclasses.replace(cfg, migrate_inputs2=True)
    if a.migrate_rules:
        if not a.resume or cfg.tokenizer.rules_mode != "teach" or cfg.ppo.kickstart.on:
            ap.error("--migrate-rules needs --resume, tokenizer.rules_mode=teach and ppo.kickstart off")
        cfg = dataclasses.replace(cfg, migrate_rules=True)
    if a.rules_free:
        if not a.resume or cfg.tokenizer.rules_mode != "free":
            ap.error("--rules-free needs --resume and tokenizer.rules_mode=free")
        cfg = dataclasses.replace(cfg, rules_free=True)
    if a.migrate_critic:
        if not a.resume or cfg.ppo.critic is None or cfg.model.config().critic is None:
            ap.error("--migrate-critic needs --resume, model.critic and ppo.critic")
        cfg = dataclasses.replace(cfg, migrate_critic=True)
    if a.reset_seat_symmetry_optimizer:
        if not a.migrate_seat_symmetry:
            ap.error("--reset-seat-symmetry-optimizer needs --migrate-seat-symmetry")
        cfg = dataclasses.replace(cfg, reset_seat_symmetry_optimizer=True)
    if a.resume_from:
        if not a.resume:
            ap.error("--resume-from needs --resume")
        cfg = dataclasses.replace(cfg, resume_from=str(Path(a.resume_from).resolve()))
    return Supervisor(cfg, resume=a.resume).run()


if __name__ == "__main__":
    sys.exit(main())
