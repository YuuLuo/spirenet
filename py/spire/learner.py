"""PPO learner: clipped-surrogate update over rollout segments, EMA weights, checkpoints and
acting-weight publication.

Model / tokenizer contract (``spire.tokenize`` / ``spire.model``)::

    Tokenizer.collate(list[Sample]) -> Batch                  # dict of CPU tensors
    SpireNet.forward(Batch) -> {"logits": (B, A_max) with illegal / padded actions = -inf,
                                "value": (B,), "pwin_logit": (B,), "floors_pred": (B,),
                                "hp_pred": (B,)}

Head targets used here: ``value`` ~ discounted return (raw units); ``pwin_logit`` ~ logit of
P(win) (BCE); ``floors_pred`` ~ floors remaining / ``PPOConfig.floors_scale`` (MSE);
``hp_pred`` ~ post-combat team HP fraction in [0, 1] (MSE, combat decisions only).

One :meth:`Learner.update` = one PPO iteration over the given segments:

1. optional value recomputation with the current weights (``recompute_values``);
2. GAE (:func:`spire.rollout.build_batch`); the policy advantages get the advice offset (``advice``,
   :mod:`spire.advice`; also habit rules ``(habit spec, bonus)``: ``+bonus`` / ``-bonus`` where the habit judges the
   step +1 / -1) and are normalised over the whole iteration;
3. ``epochs`` passes over random minibatches of ~``minibatch`` samples; each minibatch is
   sorted by sample size (``size_fn``) and cut into micro-batches (``micro_batch`` samples /
   ``micro_tokens`` padded tokens) that go through ``Tokenizer.collate`` and accumulate
   gradients, so micro-batches are size-homogeneous (little padding) while minibatches stay
   unbiased random draws;
4. per minibatch: grad-norm clipping, optional KL-dependent Adam learning-rate damping,
   AdamW step (skipped if loss / grad is not finite), EMA update; optional early stop
   after the step when its pre-step approximate KL exceeds ``target_kl``. KL is relative
   to behaviour unless ``decoupled_clip_anchor`` selects the frozen pre-update anchor.

Decisions with a single legal action carry no policy gradient and are excluded from the
policy / entropy terms (they still train value and auxiliary heads).
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .advice import advice_metrics, advice_offset, habit_metrics, habit_offset, parse_advice
from . import critic as critic_mod
from .critic import CriticPPOConfig
from .critic_migration import migrate_critic_optimizer, migrate_critic_weights, tower_init_state
from .distributions import (batch_choice_keys, choice_uniform_probs, exploration_table, kind_coefficients,
                            legal_uniform_log_probs, uniform_cross_entropy)
from .exit_buffer import ExitBuffer
from .hlgauss_migration import migrate_hlgauss_optimizer, migrate_hlgauss_weights
from .model import SpireNet, hl_gauss_probs, migrate_public_history_weights, state_to_cpu
from .rollout import GAEConfig, RolloutBatch, Segment, SegmentEnd, apply_rloo, build_batch, stitch_joins
from .taste import TasteConfig, taste_learner_metrics, taste_log_prob
from .temperature import COMBAT_KINDS, check_temperature, tempered_log_probs
from .schema import ActionCol, Kind
from .seat_migration import migrate_seat_symmetry_optimizer, migrate_seat_symmetry_weights
from .coop_migration import migrate_coop_features_optimizer, migrate_coop_features_weights
from .calc_migration import migrate_calc_optimizer, migrate_calc_weights
from .inputs2_migration import config_before, migrate_inputs2_optimizer, migrate_inputs2_weights
from .forecast_migration import migrate_forecast_optimizer, migrate_forecast_weights
from .kickstart import (A_ERR, A_ERR2, A_LOSS, A_RET, A_RET2, N_ACC, KickstartConfig, coefficients, distil_sums,
                        kick_loss, refade_after_resume, smooth_table, teacher_active)
from . import rules as rules_mod
from .rules import N_RULES_ACC, RulesConfig, teach_loss
from .replica import Replica, normalize_replica_devices
from .tokenize import Ent, SampleBlock, Tokenizer
from .trainstep import NF, NF_CRITIC, PF_COLS, MicroStepper, bucket_micros, critic_loss, ema_weight

Batch = Mapping[str, Tensor]
SizeFn = Callable[[Any], int]
N_KIND_SLOTS = max(int(k) for k in Kind) + 1


class TokenizerLike(Protocol):
    def collate(self, samples: Sequence[Any]) -> Batch: ...


# =========================================================================== config


@dataclass(frozen=True)
class PPOConfig:
    # iteration structure
    epochs: int = 2
    minibatch: int = 4096
    micro_batch: int = 1024
    micro_tokens: int | None = None  # padded-token cap per micro-batch (needs size_fn)
    # surrogate / regularisers
    clip: float = 0.2
    adv_norm: bool = True
    ent_coef: float = 0.01
    ent_coef_by_kind: tuple[tuple[int, float], ...] = ()  # (Kind, coef) overrides
    # Divide a sample's entropy coefficient by its episode's seat count. One controller decision of an n-seat game
    # is 1 / n of a single-seat-equivalent decision (as for the trace decay): its action-value gaps are about n
    # times smaller, so an unscaled bonus flattens the multi-seat policy. Measured in run l4-first: 4-seat COMBAT
    # entropy / log(n) rose from 0.62 to 0.81 in two hours and sampled 4-seat play fell from 20% to 5% wins while
    # greedy play rose to 60%.
    ent_coef_seat_scaled: bool = False
    exploration_by_kind: tuple[tuple[int, float], ...] = ()  # uniform legal-action mass in behavior policy
    exploration_choice_uniform: bool = False  # mixture and uniform-KL reference: choices, then their targets
    uniform_kl_by_kind: tuple[tuple[int, float], ...] = ()  # KL(uniform || base policy) barrier
    ent_normalize: bool = False  # divide entropy by log(n_legal) in the bonus
    vf_coef: float = 0.5
    value_clip: float | None = None  # clip V around the behaviour value (None = off)
    normalize_returns: bool = False  # value loss in units of the running return std
    aux_win: float = 0.1
    aux_floors: float = 0.1
    aux_hp: float = 0.1
    floors_scale: float = 50.0
    # optimiser
    lr: float = 3e-4
    lr_schedule: Literal["constant", "linear", "cosine"] = "constant"
    warmup_updates: int = 0
    total_updates: int = 0  # horizon of the linear / cosine decay
    min_lr_frac: float = 0.1
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-5
    grad_clip: float = 1.0
    target_kl: float | None = None
    kl_lr_target: float | None = None  # damp Adam above this KL; retain the lowest scale through the iteration
    ema_decay: float = 0.998
    bf16: bool = True  # bf16 autocast on CUDA (no GradScaler needed); fp32 on CPU
    recompute_values: bool = False
    decoupled_clip_anchor: bool = False  # clip around pre-update policy; retain behavior IS denominator
    # advice rules (kind name, option key name, hp_min, bonus[, below]), spire.advice: the policy-loss advantage of an
    # advised step is A + bonus if it chose the advised option, A - bonus if another option of its kind (other actions
    # are not advised steps; raw units, before adv_norm); ``below`` is the same against the option under hp_min;
    # () = off. Rules are variable-length: train.build would cut a fixed-length tuple annotation to its length
    advice: tuple[tuple[Any, ...], ...] = ()
    # per-episode card-family exploration at CARD_REWARD (spire.taste): behaviour only, exact accounting; off by default
    taste: TasteConfig = field(default_factory=TasteConfig)
    # distillation toward a frozen teacher checkpoint that fades out (spire.kickstart); teacher "" = off
    kickstart: KickstartConfig = field(default_factory=KickstartConfig)
    # spire.rules: the removal teach term while tokenizer.rules_mode is "teach" / "free" (unused in "mask")
    rules: RulesConfig = field(default_factory=RulesConfig)
    # behaviour temperature of combat steps (spire.temperature): the actor samples softmax(z / T) in fights and stores
    # T per step (rollout column beh_temp); the learner's log q(a) uses the stored value. 1.0 = off
    combat_temperature: float = 1.0
    gae: GAEConfig = field(default_factory=GAEConfig)
    seed: int = 0
    # the training-only critic tower (notes/critic/contract.md 4-5; needs model.critic): None = off, and then neither
    # the checkpoint's config nor config.json has the key (tree p4)
    critic: CriticPPOConfig | None = None
    # p12: smooth linear decay of non-potential progress shaping (floor + elite + boss) from 1.0 -> shaping_end_scale
    # over shaping_fade_updates updates starting at the first update (0 / 1.0 = off, omitted from config dicts)
    shaping_end_scale: float = 1.0
    shaping_fade_updates: int = 0
    # p13: Async ExIt Search Teacher bounded cross-entropy aux loss (0.0 = off, omitted from config dicts when default)
    exit_coef: float = 0.0
    exit_dir: str = ""
    exit_batch_size: int = 64
    exit_min_samples: int = 64
    exit_buffer_size: int = 8192
    exit_max_age_updates: int = 2000
    exit_min_gain: float = 0.025
    exit_gain_scale: float = 0.08
    exit_weight_min: float = 0.5
    exit_weight_max: float = 3.0
    exit_max_kl_guard: float = 0.02
    # p13.1 (ExIt v2, all defaults OFF = bit-identical to p13): recency/legacy prioritization,
    # residual margin cap, Boltzmann soft K-candidate targets, legal_mask enforcement, momentum-free step
    exit_recency_halflife: int = 0
    exit_legacy_weight: float = 1.0
    exit_retire_legacy_at: int = 0
    exit_target_cap: float = 0.0
    exit_soft_beta: float = 0.0
    exit_precond: bool = False
    exit_apply_legal_mask: bool = False

    def __post_init__(self) -> None:
        assert self.kl_lr_target is None or (math.isfinite(self.kl_lr_target) and self.kl_lr_target > 0), \
            "kl_lr_target must be finite and positive"
        parse_advice(self.advice)  # unknown kinds / habits / bad numbers fail here, names at the vocab (Advisor)
        assert not self.taste.on or not (self.recompute_values or self.decoupled_clip_anchor), \
            "taste: recompute_values and decoupled_clip_anchor re-derive the behaviour log-prob; both must be off"
        # kickstart phase 0 (the teacher acts): one epoch, so that every minibatch is unseen data (held-out metrics)
        assert not (self.kickstart.on and self.kickstart.phase == 0) or self.epochs == 1, \
            "kickstart phase 0 trains one epoch: set ppo.epochs = 1"
        # R10e: phase 0 builds the returns from the TEACHER's stored values (kickstart-spec-v2 phase 0); recompute
        # would replace values and bootstraps by the student's before build_batch (update(): _recompute -> build_batch)
        assert not (self.kickstart.on and self.kickstart.phase == 0 and self.recompute_values), \
            "kickstart phase 0 uses the teacher's stored values: set ppo.recompute_values = false"
        check_temperature(self.combat_temperature, self.advice, self.taste.on, self.decoupled_clip_anchor)
        if isinstance(self.critic, Mapping):  # PPOConfig(**saved) of a checkpoint / config round trip
            object.__setattr__(self, "critic", CriticPPOConfig(**{**self.critic, "coef": tuple(self.critic["coef"])}))
        # contract 3: the actors compute the tower's values; a recomputation would refresh the public value only
        assert self.critic is None or not self.recompute_values, "ppo.critic refuses ppo.recompute_values"
        assert math.isfinite(self.shaping_end_scale) and 0.0 <= self.shaping_end_scale <= 1.0, \
            f"shaping_end_scale {self.shaping_end_scale}: 0 <= scale <= 1"
        assert (isinstance(self.shaping_fade_updates, int) and not isinstance(self.shaping_fade_updates, bool)
                and self.shaping_fade_updates >= 0), f"shaping_fade_updates {self.shaping_fade_updates}: >= 0"
        assert self.shaping_fade_updates > 0 or self.shaping_end_scale == 1.0, \
            "shaping_end_scale != 1.0 requires shaping_fade_updates > 0"
        assert self.critic is None or not (self.shaping_fade_updates > 0 and self.shaping_end_scale != 1.0), \
            "shaping_fade_updates refuses ppo.critic"
        assert math.isfinite(self.exit_coef) and self.exit_coef >= 0.0, f"exit_coef {self.exit_coef}: >= 0"
        assert self.exit_coef == 0.0 or bool(self.exit_dir), "ppo.exit_coef > 0 requires ppo.exit_dir"
        assert (isinstance(self.exit_batch_size, int) and not isinstance(self.exit_batch_size, bool)
                and self.exit_batch_size >= 1), f"exit_batch_size {self.exit_batch_size}: >= 1"
        assert (isinstance(self.exit_min_samples, int) and not isinstance(self.exit_min_samples, bool)
                and self.exit_min_samples >= 1), f"exit_min_samples {self.exit_min_samples}: >= 1"
        assert (isinstance(self.exit_buffer_size, int) and not isinstance(self.exit_buffer_size, bool)
                and self.exit_buffer_size >= 1), f"exit_buffer_size {self.exit_buffer_size}: >= 1"
        assert (isinstance(self.exit_max_age_updates, int) and not isinstance(self.exit_max_age_updates, bool)
                and self.exit_max_age_updates >= 0), f"exit_max_age_updates {self.exit_max_age_updates}: >= 0"
        assert math.isfinite(self.exit_min_gain) and self.exit_min_gain >= 0.0, f"exit_min_gain {self.exit_min_gain}"
        assert math.isfinite(self.exit_gain_scale) and self.exit_gain_scale > 0.0, \
            f"exit_gain_scale {self.exit_gain_scale}"
        assert (math.isfinite(self.exit_weight_min) and math.isfinite(self.exit_weight_max)
                and 0.0 < self.exit_weight_min <= self.exit_weight_max), \
            f"exit_weight_min/max ({self.exit_weight_min}, {self.exit_weight_max})"
        assert math.isfinite(self.exit_max_kl_guard) and self.exit_max_kl_guard > 0.0, \
            f"exit_max_kl_guard {self.exit_max_kl_guard}"
        assert (isinstance(self.exit_recency_halflife, int) and not isinstance(self.exit_recency_halflife, bool)
                and self.exit_recency_halflife >= 0), f"exit_recency_halflife {self.exit_recency_halflife}: >= 0"
        assert math.isfinite(self.exit_legacy_weight) and 0.0 <= self.exit_legacy_weight <= 1.0, \
            f"exit_legacy_weight {self.exit_legacy_weight}: 0 <= w <= 1"
        assert (isinstance(self.exit_retire_legacy_at, int) and not isinstance(self.exit_retire_legacy_at, bool)
                and self.exit_retire_legacy_at >= 0), f"exit_retire_legacy_at {self.exit_retire_legacy_at}: >= 0"
        assert math.isfinite(self.exit_target_cap) and 0.0 <= self.exit_target_cap <= 1.0, \
            f"exit_target_cap {self.exit_target_cap}: 0 <= cap <= 1"
        assert math.isfinite(self.exit_soft_beta) and self.exit_soft_beta >= 0.0, \
            f"exit_soft_beta {self.exit_soft_beta}: >= 0"
        assert isinstance(self.exit_precond, bool), self.exit_precond
        assert isinstance(self.exit_apply_legal_mask, bool), self.exit_apply_legal_mask

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.critic is None:  # a learner without the critic has the config of tree p4
            del d["critic"]
        if self.kickstart.floor == 0.0:  # a kickstart without a floor has the config of tree p6
            del d["kickstart"]["floor"]
        if self.shaping_end_scale == 1.0:
            del d["shaping_end_scale"]
        if self.shaping_fade_updates == 0:
            del d["shaping_fade_updates"]
        for k, v in _EXIT_CONFIG_DEFAULTS:
            if d[k] == v:
                del d[k]
        return d


_EXIT_CONFIG_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("exit_coef", 0.0),
    ("exit_dir", ""),
    ("exit_batch_size", 64),
    ("exit_min_samples", 64),
    ("exit_buffer_size", 8192),
    ("exit_max_age_updates", 2000),
    ("exit_min_gain", 0.025),
    ("exit_gain_scale", 0.08),
    ("exit_weight_min", 0.5),
    ("exit_weight_max", 3.0),
    ("exit_max_kl_guard", 0.02),
    ("exit_recency_halflife", 0),
    ("exit_legacy_weight", 1.0),
    ("exit_retire_legacy_at", 0),
    ("exit_target_cap", 0.0),
    ("exit_soft_beta", 0.0),
    ("exit_precond", False),
    ("exit_apply_legal_mask", False),
)


def lr_at(cfg: PPOConfig, update: int) -> float:
    """Learning rate for PPO iteration ``update`` (0-based)."""
    if cfg.warmup_updates and update < cfg.warmup_updates:
        return cfg.lr * (update + 1) / cfg.warmup_updates
    if cfg.lr_schedule == "constant" or cfg.total_updates <= cfg.warmup_updates:
        return cfg.lr
    x = min(1.0, (update - cfg.warmup_updates) / (cfg.total_updates - cfg.warmup_updates))
    lo = cfg.min_lr_frac
    if cfg.lr_schedule == "linear":
        return cfg.lr * (1.0 - (1.0 - lo) * x)
    return cfg.lr * (lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * x)))


# =========================================================================== weight publication


class WeightPublisher(ABC):
    """Learner side: make acting weights + version visible to actors."""

    @abstractmethod
    def publish(self, state_dict: Mapping[str, Tensor], version: int) -> None: ...


class WeightSource(ABC):
    """Actor side: fetch acting weights newer than the ones held."""

    @abstractmethod
    def poll(self, have_version: int) -> tuple[int, dict[str, Tensor]] | None: ...


def _retry(fn: Callable[[], Any], attempts: int = 50, delay: float = 0.01) -> Any:
    """Windows refuses to replace / open a file another process holds open; retry briefly."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


class FileWeightPublisher(WeightPublisher):
    """Atomic file publication: ``acting-<version>.pt`` (written to a temp name, then renamed),
    then a ``LATEST`` pointer ``{"version", "file"}`` replaced atomically. Readers never see a
    partial file; the newest ``keep`` weight files are retained so a reader that just read the
    pointer can still open its file. A shared-memory publisher implements the same interface."""

    def __init__(self, directory: str | Path, *, keep: int = 3, dtype: torch.dtype | None = None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = max(1, keep)
        self.dtype = dtype

    def publish(self, state_dict: Mapping[str, Tensor], version: int) -> None:
        sd = {k: (v.detach().to("cpu", self.dtype) if self.dtype is not None and v.is_floating_point()
                  else v.detach().cpu()) for k, v in state_dict.items()}
        name = f"acting-{version:010d}.pt"
        tmp = self.dir / (name + ".tmp")
        torch.save({"version": version, "state_dict": sd}, tmp)
        _retry(lambda: os.replace(tmp, self.dir / name))
        ptr = self.dir / "LATEST.tmp"
        ptr.write_text(json.dumps({"version": version, "file": name}), encoding="utf-8")
        _retry(lambda: os.replace(ptr, self.dir / "LATEST"))
        # Files newer than this version were left by a run that went past the checkpoint being
        # resumed: they would sort after (and so evict) every new file until the version caught up.
        files = sorted(self.dir.glob("acting-*.pt"))
        newer = [p for p in files if _weight_version(p) > version]
        for old in newer + [p for p in files if p not in newer][:-self.keep]:
            try:
                old.unlink()
            except OSError:  # still open by a reader; removed on a later publish
                pass


def _weight_version(path: Path) -> int:
    """Version of an ``acting-<version>.pt`` file."""
    return int(path.stem.split("-", 1)[1])


class FileWeightSource(WeightSource):
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def latest_version(self) -> int:
        try:
            return int(json.loads(_retry((self.dir / "LATEST").read_text))["version"])
        except FileNotFoundError:
            return -1

    def poll(self, have_version: int) -> tuple[int, dict[str, Tensor]] | None:
        try:
            ptr = json.loads(_retry((self.dir / "LATEST").read_text))
            if int(ptr["version"]) <= have_version:
                return None
            obj = _retry(lambda: torch.load(self.dir / ptr["file"], map_location="cpu", weights_only=True))
        except FileNotFoundError:  # nothing published yet, or pruned between pointer and file
            return None
        return int(obj["version"]), obj["state_dict"]


# =========================================================================== helpers


class RunningMoments:
    """Running mean / variance (Chan et al. parallel update), float64."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: np.ndarray) -> None:
        if not len(x):
            return
        n_b, mean_b, var_b = len(x), float(np.mean(x)), float(np.var(x))
        n = self.n + n_b
        d = mean_b - self.mean
        self.mean += d * n_b / n
        self.m2 += var_b * n_b + d * d * self.n * n_b / n
        self.n = n

    @property
    def var(self) -> float:
        return self.m2 / self.n if self.n > 1 else 1.0

    def state(self) -> dict[str, float]:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    def load(self, s: Mapping[str, float]) -> None:
        self.n, self.mean, self.m2 = int(s["n"]), float(s["mean"]), float(s["m2"])


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = float(np.var(target))
    return float("nan") if var == 0 or not len(target) else 1.0 - float(np.var(target - pred)) / var


TD_CLASSES = ("td_end_won", "td_enter_fight", "td_in_fight", "td_reward_screens")
_REWARD_KINDS = (int(Kind.REWARD), int(Kind.CARD_REWARD))


def transition_td(segments: Sequence[Segment], gamma: float) -> dict[str, Any]:
    """One-step TD errors r_t + gamma V(s_{t+1}) - V(s_t) of the public head and of the tower total (V_WIN + V_PROG +
    V_SURV, rewards R_WIN + R_PROG + R_SURV) on every step of critic segments, and the transition classes of
    notes/next/signal.md P1 (``alt/next/signal/seg_probe.py``): s_{t+1} is the next step of the segment, of the next
    fragment of the same episode when adjacent (:func:`spire.rollout.stitch_joins`), else the segment's bootstrap
    (its kind unknown: the step is in no class) or the terminal (no class).

    * ``td_in_fight``: a fight decision whose next decision is in the same fight;
    * ``td_end_won``: a fight decision whose next decision is outside it (the first frame after a won fight);
    * ``td_enter_fight``: a decision outside fights whose next decision is a fight's first;
    * ``td_reward_screens``: a REWARD / CARD_REWARD decision whose next decision is outside fights."""
    lens = np.fromiter((len(s) for s in segments), np.int64, len(segments))
    ends = lens.cumsum() - 1
    n = int(lens.sum())
    vp = np.concatenate([s.values for s in segments]).astype(np.float64)
    vt = np.concatenate([s.values_c for s in segments]).astype(np.float64).sum(1)  # type: ignore[misc]
    rp = np.concatenate([s.rewards for s in segments]).astype(np.float64)
    rt = np.concatenate([s.rewards_c for s in segments]).astype(np.float64).sum(1)  # type: ignore[misc]
    cid = np.concatenate([s.combat_id for s in segments])
    kinds = np.concatenate([s.kinds for s in segments]).astype(np.int64)
    nvp, nvt = np.zeros(n), np.zeros(n)
    nvp[:-1], nvt[:-1] = vp[1:], vt[1:]
    ncid = np.full(n, -9, np.int64)
    ncid[:-1] = cid[1:]
    known = np.ones(n, bool)
    joins = stitch_joins(segments)
    for i, s in enumerate(segments):
        if joins[i]:
            continue
        e = int(ends[i])
        known[e] = False  # terminal (no next state) or a bootstrap whose kind is unknown
        ncid[e] = -9
        if s.end == SegmentEnd.TERMINAL:
            nvp[e] = nvt[e] = 0.0
        else:
            nvp[e], nvt[e] = s.bootstrap_value, float(np.asarray(s.bootstrap_c, np.float64).sum())
    inc = cid >= 0
    same = inc & (ncid == cid) & known
    classes = {"td_end_won": inc & ~same & known, "td_enter_fight": ~inc & (ncid >= 0) & known, "td_in_fight": same,
               "td_reward_screens": ~inc & (ncid < 0) & known & np.isin(kinds, _REWARD_KINDS)}
    return {"public": rp + gamma * nvp - vp, "tower": rt + gamma * nvt - vt, "classes": classes}


def switch_test(window: Sequence[Sequence[float]], c: CriticPPOConfig) -> dict[str, Any]:
    """The switch conditions of notes/critic/contract.md 5 on the pooled statistics of the last ``switch_window``
    updates (``window`` rows: won-fight-end TD count and sum of the tower, combat decisions, sum and sum of squares of
    A_PUB, of A_TOWER): |mean TD of the tower at combat -> fight won| <= switch_td_end and var(A_TOWER) / var(A_PUB)
    over combat decisions <= switch_var_ratio. ``ready``: the window is full and both quantities are defined."""
    w = np.asarray(window, np.float64).reshape(-1, 7).sum(0)
    n_won, s_won, n_c, s_p, ss_p, s_t, ss_t = w.tolist()
    var_p = ss_p / n_c - (s_p / n_c) ** 2 if n_c else float("nan")
    var_t = ss_t / n_c - (s_t / n_c) ** 2 if n_c else float("nan")
    td = s_won / n_won if n_won else float("nan")
    ratio = var_t / var_p if n_c and var_p > 0 else float("nan")
    ready = len(window) >= c.switch_window and math.isfinite(td) and math.isfinite(ratio)
    return {"ready": bool(ready), "td_end_won": td, "var_ratio": ratio, "n_won": int(n_won), "n_combat": int(n_c),
            "pass": bool(ready and abs(td) <= c.switch_td_end and ratio <= c.switch_var_ratio)}


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    no_decay: set[int] = set()
    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            no_decay.update(id(p) for p in mod.parameters(recurse=False))
        elif getattr(mod, "char_experts", 0) > 0:
            no_decay.add(id(mod.exp_b1))
            no_decay.add(id(mod.exp_b2))
    decay: list[nn.Parameter] = []
    other: list[nn.Parameter] = []
    for p in model.parameters():
        if p.requires_grad:
            (other if p.ndim < 2 or id(p) in no_decay else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": other, "weight_decay": 0.0}]
    return [g for g in groups if g["params"]]


def _migrate_history_optimizer(saved: dict[str, Any], model: SpireNet,
                               optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Preserve old Adam states by parameter position within each strict, unchanged parameter group.

    Serialized IDs of the no-decay group move when the new matrix joins the decay group. Remap them;
    the new projection has no state and Adam initializes its moments and step on its first update.
    """
    assert model.history_proj is not None
    assert not optimizer.state, "migration must precede optimizer updates"
    current = optimizer.state_dict()
    old_groups, new_groups = saved["param_groups"], current["param_groups"]
    assert len(old_groups) == len(new_groups)
    remap: dict[int, int] = {}
    groups = []
    for old, new, live in zip(old_groups, new_groups, optimizer.param_groups):
        old_params = [i for i, p in zip(new["params"], live["params"]) if p is not model.history_proj.weight]
        assert len(old["params"]) == len(old_params), "migration cannot change existing optimizer groups"
        for before, after in zip(old["params"], old_params):
            assert before not in remap, "duplicate optimizer parameter ID"
            remap[before] = after
        groups.append({**old, "params": new["params"]})
    assert set(saved["state"]) <= set(remap), "optimizer state references an unknown parameter"
    return {"state": {remap[k]: v for k, v in saved["state"].items()}, "param_groups": groups}


@dataclass
class _Tensors:
    """Per-step training tensors of one iteration, on the learner device."""

    actions: Tensor
    logp_b: Tensor
    logp_ref: Tensor
    adv: Tensor
    ret: Tensor
    v_old: Tensor
    pol_mask: Tensor
    log_legal: Tensor
    ent_coef: Tensor
    kinds: Tensor
    win: Tensor
    win_mask: Tensor
    floors: Tensor
    floors_mask: Tensor
    hp: Tensor
    hp_mask: Tensor
    # spire.temperature: T(step) of the behaviour (ones for segments without a beh_temp record) and the combat steps
    # (IN_COMBAT and kind COMBAT / CARD_SELECT, for entropy_beh_combat); None only in tensors built by hand (tests)
    beh_temp: Tensor | None = field(default=None, kw_only=True)
    combat: Tensor | None = field(default=None, kw_only=True)


@dataclass
class _TasteTensors(_Tensors):
    """:class:`_Tensors` of a learner with taste on: the per-step spire.taste terms."""

    t_keep: Tensor
    t_add: Tensor


@dataclass
class Prepared:
    """One iteration made ready for :meth:`Learner.update_prepared` by :meth:`Learner.prepare`, without the model's
    weights and without the card (notes/next/pipeline-spec.md): the segments, the version the learner will have when it
    consumes them, the GAE batch, the fast path's sample block, columns and buckets (or the generic path's sizes), the
    behaviour temperatures, the critic's weight-free part and the time it took."""

    segments: Sequence[Segment]
    consume_version: int
    rb: RolloutBatch
    blk: SampleBlock | None
    temps: np.ndarray
    tempered: bool
    reference: np.ndarray | None
    cols: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]] | None
    buckets: np.ndarray | None
    sizes: np.ndarray | None
    critic: dict[str, Any] | None
    time_prep: float
    rloo: tuple[int, int, float] | None = None
    shaping_scale: float | None = None

    def __len__(self) -> int:
        return len(self.rb)


def policy_lag_seconds(segments: Sequence[Segment], now: float) -> dict[str, float] | None:
    """Age of the iteration's decisions when the update starts (seconds, sample-weighted; a segment's
    steps are spread evenly between its first and last decision time). None if no segment carries times."""
    segs = [s for s in segments if s.t_first > 0]
    if not segs:
        return None
    ages = np.concatenate([now - np.linspace(s.t_first, s.t_last, len(s)) for s in segs])
    p50, p90, p99 = np.percentile(ages, (50, 90, 99)).tolist()
    return {"p50": round(p50, 3), "p90": round(p90, 3), "p99": round(p99, 3), "mean": round(float(ages.mean()), 3),
            "max": round(float(ages.max()), 3)}


# =========================================================================== learner


class Learner:
    """Owns the online model, its EMA copy, the optimiser and the update / version counters.

    ``version`` = number of completed updates = version tag of the published weights; actors act
    with the online weights (stamping steps with the version they used), the EMA weights are
    for evaluation only (DECISIONS-H1 G.3) - or, with ``acting_weights`` "ema"
    (notes/next/round13-1119.md), actors act with the EMA weights of that version.
    """

    def __init__(self, model: nn.Module, tokenizer: TokenizerLike, cfg: PPOConfig = PPOConfig(), *,
                 device: str | torch.device | None = None, size_fn: SizeFn | None = None,
                 publisher: WeightPublisher | None = None, fast: bool = True, graphs: bool = False,
                 compile_model: bool = False, teacher: nn.Module | None = None, teacher_sha256: str = "",
                 release_file: str | Path | None = None,
                 replica_device: str | torch.device | Sequence[str | torch.device] | None = None,
                 acting_weights: str = "online", confident_keep: float = 1.0, confident_tau: float = 0.95,
                 confident_seed: int = 0, confident_exempt_kinds: Sequence[int] = (),
                 rloo_alpha: float = 0.0) -> None:
        """``fast`` (CUDA + a :class:`~spire.tokenize.Tokenizer`): the asynchronous C-collation path of
        :mod:`spire.trainstep` for segments whose samples are :class:`~spire.tokenize.SampleBlock` s;
        ``graphs``: in that path, replay one CUDA graph per padded micro-batch shape. Otherwise (CPU, stub
        tokenizers, plain sample lists) the generic eager path. ``teacher`` (with ``cfg.kickstart`` on, and only
        then): the frozen network of :func:`spire.kickstart.load_teacher` and its checkpoint's sha256;
        ``release_file``: the kickstart release file (``runs/<run>/KICK_RELEASE``; needed in phase 1 with
        ``fade_updates``). ``replica_device`` ("cuda:N", notes/next/dual-spec.md): the fast path's micro-batches are
        split between this device and a replica of the model there (:mod:`spire.replica`); None = one device.
        ``acting_weights``: what :meth:`acting_state_dict` publishes, "online" (``self.model``) or "ema"
        (``self.ema``, on ``device``); the optimizer trains ``self.model`` either way. ``confident_keep`` /
        ``confident_tau`` / ``confident_seed`` (notes/next/round21-1121.md): a taken sample whose behaviour probability
        exp(logp_b) is >= tau goes into the minibatches with probability keep (:meth:`_kept_rows`); keep 1 = all."""
        assert cfg.kickstart.on == (teacher is not None), "a teacher iff ppo.kickstart is on"
        assert acting_weights in ("online", "ema"), f"acting_weights {acting_weights!r}: 'online' or 'ema'"
        self.acting_weights = acting_weights
        assert 0.0 < confident_keep <= 1.0, f"confident_keep {confident_keep}: 0 < keep <= 1"
        assert 0.0 < confident_tau <= 1.0, f"confident_tau {confident_tau}: 0 < tau <= 1"
        assert confident_keep == 1.0 or not (cfg.kickstart.on or cfg.recompute_values or cfg.critic is not None), \
            "confident_keep < 1 refuses a kickstart, recompute_values and the critic"
        self.confident_keep = confident_keep
        self.confident_tau = confident_tau
        self.confident_seed = confident_seed
        assert all(isinstance(k, int) and not isinstance(k, bool) and 0 <= k < N_KIND_SLOTS
                   for k in confident_exempt_kinds), f"confident_exempt_kinds {confident_exempt_kinds!r}: invalid Kind"
        self.confident_exempt_kinds = tuple(int(k) for k in confident_exempt_kinds)
        self._confident_exempt = np.zeros(N_KIND_SLOTS, bool)
        if self.confident_exempt_kinds:
            self._confident_exempt[list(self.confident_exempt_kinds)] = True
        assert math.isfinite(rloo_alpha) and 0.0 <= rloo_alpha <= 1.0, f"rloo_alpha {rloo_alpha}: 0 <= alpha <= 1"
        self.rloo_alpha = float(rloo_alpha)
        assert (teacher is None) == (not teacher_sha256), "a teacher needs its checkpoint's sha256"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = cfg
        self.model = model.to(self.device)
        mcfg = getattr(self.model, "config", None)
        self._hl_gauss = bool(mcfg is not None and getattr(mcfg, "value_bins", 0) > 0)
        if self._hl_gauss:
            assert cfg.value_clip is None, "model.value_bins > 0 refuses ppo.value_clip"
            assert not cfg.normalize_returns, "model.value_bins > 0 refuses ppo.normalize_returns"
            self._val_edges: Tensor | None = self.model.val_edges  # type: ignore[assignment]
            self._val_vmin = float(mcfg.value_vmin)  # type: ignore[union-attr]
            self._val_vmax = float(mcfg.value_vmax)  # type: ignore[union-attr]
            self._val_sigma = float(mcfg.value_sigma)  # type: ignore[union-attr]
        else:
            self._val_edges = None
            self._val_vmin = self._val_vmax = self._val_sigma = 0.0
        self._shaping_active = cfg.shaping_fade_updates > 0 and cfg.shaping_end_scale != 1.0
        self.shaping_start_update: int | None = None
        self._updates_offset = 0  # self.updates - self.version; immutable during training (read by prepare thread)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.tokenizer = tokenizer
        if cfg.exploration_choice_uniform:
            assert isinstance(tokenizer, Tokenizer), "choice-uniform exploration requires entity samples"
            self._choice_verb_slot = tokenizer.layout.slot(Ent.ACTION, ActionCol.VERB)
        self.size_fn = size_fn
        self.publisher = publisher
        cuda = self.device.type == "cuda"
        # the critic tower (notes/critic/contract.md 4.3): its own AdamW groups (lr ppo.critic.lr, constant, no KL
        # brake), its own gradient clip; the trunk's groups are those of a network without the tower
        self.critic_cfg = cfg.critic
        if cfg.critic is not None:
            assert isinstance(model, SpireNet) and model.critic is not None, "ppo.critic needs model.critic"
            groups = critic_mod.optimizer_groups(self.model, cfg.weight_decay, cfg.critic.lr)
        else:
            assert getattr(model, "critic", None) is None, "model.critic needs ppo.critic"
            groups = _param_groups(self.model, cfg.weight_decay)
        self.opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.adam_eps, fused=cuda)
        self.autocast = cuda and cfg.bf16
        self.rng = np.random.default_rng(cfg.seed)
        self.ret_moments = RunningMoments()
        self.version = 0
        self.updates = 0
        self.samples_seen = 0
        self.negative_staleness_samples = 0  # refusals in this learner process, saved for crash diagnosis
        self._ent_by_kind = np.full(N_KIND_SLOTS, cfg.ent_coef, np.float32)
        for k, c in cfg.ent_coef_by_kind:
            self._ent_by_kind[int(k)] = c
        self._exploration = (torch.as_tensor(exploration_table(cfg.exploration_by_kind), device=self.device)
                             if cfg.exploration_by_kind else None)
        self._uniform_kl = (torch.as_tensor(kind_coefficients(cfg.uniform_kl_by_kind), device=self.device)
                            if cfg.uniform_kl_by_kind else None)
        self._advice, self._habits = parse_advice(cfg.advice)
        self._taste = cfg.taste.on
        self._tempered = False  # this iteration has steps with a behaviour temperature != 1 (spire.temperature)
        self.ema_steps = 0  # EMA updates so far (warm-up: decay_t = min(decay, (1 + t) / (10 + t)))
        # the trunk's parameters (clipped together, as in a network without the tower) and the tower's
        self._params = [p for g in self.opt.param_groups if not g.get("critic") for p in g["params"]]
        self._cparams = [p for g in self.opt.param_groups if g.get("critic") for p in g["params"]]
        # critic state (notes/critic/contract.md 5): the update at which the tower began (its warm-up counts from
        # there), the update of the switch (None = not yet), the last switch_window updates' switch statistics, the
        # device sums of the critic losses
        self.critic_first: int | None = None
        self.switched_at: int | None = None
        self._switch_window: list[list[float]] = []
        self._beta = 0.0
        self._critic_acc = torch.zeros(3, device=self.device) if cfg.critic is not None else None
        # kickstart (spire.kickstart): student updates since it began (checkpoint counter), the update that first saw
        # the release file (phase 1 with fade_updates; None = not yet), the teacher while it is evaluated, this update's
        # coefficients and the device sums (spire.kickstart.N_ACC) of the metrics; phase 0 = distillation only
        self.kick_u = 0
        self.kick_release: int | None = None
        self.kick_refade: tuple[int, float] | None = None  # a resume that lowered kickstart.floor (tree p7)
        self.release_file = None if release_file is None else Path(release_file)
        ks = cfg.kickstart
        assert not (ks.on and ks.phase == 1 and ks.fade_updates is not None) or self.release_file is not None, \
            "kickstart fade_updates needs the release file's path"
        self.distil = ks.on and ks.phase == 0
        self.teacher_sha256 = teacher_sha256
        self.teacher = None if teacher is None else teacher.to(self.device).eval().requires_grad_(False)
        self._kick_coef = (0.0, 0.0)
        self._kick_acc = torch.zeros(N_ACC, device=self.device)
        self._smooth = smooth_table(ks, self.device)
        # spire.rules: the tokenizer's rules_mode (stub tokenizers of the tests have none: "mask"), the first update in
        # "free" (checkpoint counter; the fade counts from it), this update's coefficient (host float and the live
        # device scalar the captured graphs read) and the device sums of the row's rules block
        self.rules_mode: str = tokenizer.config.rules_mode if isinstance(tokenizer, Tokenizer) else "mask"
        self.rules_free_first: int | None = None
        self._rules_coef = 0.0
        rules_on = self.rules_mode != "mask"
        self._rules_coef_dev = torch.zeros((), device=self.device) if rules_on else None
        self._rules_acc = torch.zeros(N_RULES_ACC, device=self.device) if rules_on else None
        self.stepper: MicroStepper | None = None
        if fast and cuda and isinstance(tokenizer, Tokenizer):
            for p in self._params + self._cparams:  # persistent gradient buffers (the graphs hold their addresses)
                p.grad = torch.zeros_like(p)
            self._facc = _Accum(self.device)
            self.stepper = MicroStepper(self.model, tokenizer, cfg, self.device, self._facc, graphs=graphs,
                                        autocast=self.autocast, compile_model=compile_model, teacher=self.teacher,
                                        kick_acc=self._kick_acc, rules_coef=self._rules_coef_dev,
                                        rules_acc=self._rules_acc, critic_acc=self._critic_acc)
        self.replica: Replica | None = None
        rdevs = normalize_replica_devices(replica_device)
        if rdevs:
            assert self.stepper is not None, "learner.replica_device needs the fast path (CUDA, an entity tokenizer)"
            assert graphs, "learner.replica_device needs CUDA graphs"
            assert self.teacher is None and cfg.critic is None, "learner.replica_device refuses a teacher and the critic"
            devs = [torch.device(d) for d in rdevs]
            accs = [_Accum(d) for d in devs]
            self.replica = Replica(self.model, tokenizer, cfg, devs, accs, graphs=graphs,  # type: ignore[arg-type]
                                   autocast=self.autocast, compile_model=compile_model, rules_on=rules_on)
            assert {id(p) for p in self.replica.params_a} == {id(p) for p in self._params}
            self._grads_a: list[Tensor] = [p.grad for p in self.replica.params_a]  # type: ignore[misc]
        # p13: Async ExIt Search Teacher sliding buffer (only allocated when exit_coef > 0)
        self._exit_buf: ExitBuffer | None = None
        self._exit_rng: np.random.Generator | None = None
        self._exit_lr_scale: float = 1.0
        if cfg.exit_coef > 0.0:
            vocab_hash = int(tokenizer.vocab.hash) if isinstance(tokenizer, Tokenizer) else None
            self._exit_buf = ExitBuffer(
                cfg.exit_dir,
                capacity=cfg.exit_buffer_size,
                max_age_updates=cfg.exit_max_age_updates,
                min_gain=cfg.exit_min_gain,
                expected_vocab_hash=vocab_hash,
                recency_halflife=cfg.exit_recency_halflife,
                legacy_weight=cfg.exit_legacy_weight,
                retire_legacy_at=cfg.exit_retire_legacy_at,
            )
            self._exit_rng = np.random.default_rng(cfg.seed + 0x45784974)

    # ------------------------------------------------------------------ forward helpers
    def _forward(self, model: nn.Module, samples: Sequence[Any], teacher: nn.Module | None = None,
                 crit: np.ndarray | None = None) -> dict[str, Tensor]:
        """``teacher``: also its logits / value on the same collated batch (``teacher_logits`` / ``teacher_value``);
        ``crit`` (the samples' CRIT rows): ``forward_critic`` instead of ``forward`` (``vc``, ``win_logit_c``)."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in self.tokenizer.collate(samples).items()}
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.autocast):
            out = model(batch) if crit is None else \
                model.forward_critic(batch, torch.as_tensor(crit, device=self.device))  # type: ignore[operator]
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(batch)
                out = {**out, "teacher_logits": t_out["logits"], "teacher_value": t_out["value"]}
        result = {k: v.float() for k, v in out.items()}
        if self.cfg.exploration_choice_uniform:
            result["choice_keys"] = batch_choice_keys(batch, self._choice_verb_slot)
        return result

    def _micro_batches(self, idx: np.ndarray, sizes: np.ndarray | None) -> Iterator[np.ndarray]:
        cfg = self.cfg
        if sizes is None:
            yield from (idx[i:i + cfg.micro_batch] for i in range(0, len(idx), cfg.micro_batch))
            return
        idx = idx[np.argsort(-sizes[idx], kind="stable")]
        s = sizes[idx]
        start = 0
        while start < len(idx):
            stop = min(len(idx), start + cfg.micro_batch)
            if cfg.micro_tokens is not None:  # rows sorted by size: padded cost = count * s[start]
                stop = min(stop, start + max(1, cfg.micro_tokens // max(1, int(s[start]))))
            yield idx[start:stop]
            start = stop

    @torch.no_grad()
    def predict_values(self, samples: Sequence[Any], model: nn.Module | None = None) -> np.ndarray:
        """Value-head predictions (float32 numpy) for ``samples``, in micro-batches."""
        m = model or self.model
        was_training = m.training
        m.eval()
        sizes = self._sizes(samples)
        order = np.arange(len(samples))
        out = np.zeros(len(samples), np.float32)
        for mb in self._micro_batches(order, sizes):
            out[mb] = self._forward(m, [samples[i] for i in mb])["value"].cpu().numpy()
        m.train(was_training)
        return out

    def _sizes(self, samples: Sequence[Any]) -> np.ndarray | None:
        if self.size_fn is None:
            return None
        return np.fromiter((self.size_fn(s) for s in samples), np.int64, len(samples))

    def ensure_shaping_start(self) -> None:
        """Lock ``shaping_start_update`` to ``self.updates`` on first use (call before starting a preparer thread)."""
        if self._shaping_active and self.shaping_start_update is None:
            self.shaping_start_update = self.updates

    def shaping_scale(self, update: int) -> float:
        """Linear scale in [shaping_end_scale, 1.0] applied to floor + elite + boss rewards at iteration ``update``."""
        if not self._shaping_active:
            return 1.0
        start = self.updates if self.shaping_start_update is None else self.shaping_start_update
        frac = min(1.0, max(0.0, (update - start) / self.cfg.shaping_fade_updates))
        return 1.0 + (self.cfg.shaping_end_scale - 1.0) * frac

    # ------------------------------------------------------------------ update
    def update(self, segments: Sequence[Segment]) -> dict[str, Any]:
        """Run one PPO iteration on ``segments``; returns diagnostics. = ``update_prepared(prepare(segments,
        version))``; the weight-dependent recomputation (``recompute_values`` / ``decoupled_clip_anchor``) runs first
        and its results go into :meth:`prepare`."""
        self._refuse_future(segments)  # before the recomputation, as before the split (update_prepared asserts again)
        self.ensure_shaping_start()
        t0 = time.perf_counter()
        cfg = self.cfg
        recomputed = self._recompute(segments) if cfg.recompute_values or cfg.decoupled_clip_anchor else None
        prepared = self.prepare(segments, self.version, recomputed)
        prepared.time_prep = time.perf_counter() - t0
        return self.update_prepared(prepared)

    def _refuse_future(self, segments: Sequence[Segment]) -> None:
        """Count and refuse steps acted with weights newer than the learner's (negative staleness)."""
        future = sum(int(np.count_nonzero(s.policy_version > self.version)) for s in segments)
        self.negative_staleness_samples += future
        assert future == 0, (f"negative staleness: {future} samples have policy versions newer than "
                             f"learner version {self.version}")

    def prepare(self, segments: Sequence[Segment], consume_version: int,
                recomputed: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None) -> Prepared:
        """Everything of an update before the minibatch loop that needs neither the model's weights nor the card
        (notes/next/pipeline-spec.md): the sample block, the GAE batch at ``consume_version`` (the version the learner
        will have when :meth:`update_prepared` consumes it), the behaviour temperatures, the fast path's columns and
        buckets (the generic path's sizes), the critic's weight-free part. No CUDA call; no learner state is written
        (it may run in a thread while the learner trains). ``recomputed``: :meth:`_recompute`'s output (update only)."""
        t0 = time.perf_counter()
        cfg = self.cfg
        values = boot = reference = None
        if recomputed is not None:
            current_values, current_boot, reference = recomputed
            if cfg.recompute_values:
                values, boot = current_values, current_boot
        rewards = None
        shaping_scale: float | None = None
        if self._shaping_active:
            assert all(s.rewards_prog is not None for s in segments), \
                "shaping_fade_updates > 0 requires segments with rewards_prog (RolloutStore record_prog=True)"
            shaping_scale = self.shaping_scale(consume_version + self._updates_offset)
            if shaping_scale != 1.0:
                raw_r = np.concatenate([s.rewards for s in segments]).astype(np.float64)
                prog_r = np.concatenate([s.rewards_prog for s in segments]).astype(np.float64)  # type: ignore[misc]
                rewards = (raw_r - (1.0 - shaping_scale) * prog_r).astype(np.float32)
        blk = self._block(segments)
        rb = build_batch(segments, cfg.gae, consume_version, values, boot, with_samples=blk is None, rewards=rewards)
        rloo = apply_rloo(rb, cfg.gae, self.rloo_alpha) if self.rloo_alpha > 0.0 else None
        crit = self._critic_prepare(rb, segments) if cfg.critic is not None else None
        temps, _, tempered = self._temps(rb)
        cols = buckets = sizes = None
        if blk is not None:
            cols = self._columns(rb, reference)  # with beta > 0 (critic) update_prepared builds them again
            buckets = self.tokenizer.bucket_of(blk.counts[:, 0])  # type: ignore[attr-defined]
        else:
            sizes = self._sizes(rb.samples)
        return Prepared(segments, consume_version, rb, blk, temps, tempered, reference, cols, buckets, sizes, crit,
                        time.perf_counter() - t0, rloo, shaping_scale)

    def update_prepared(self, prepared: Prepared) -> dict[str, Any]:
        """The PPO iteration of a :meth:`prepare` d batch: the staleness asserts against the learner's version, the
        device transfers, the learning rate, the minibatch loop, the counters, publication and the metrics."""
        segments, rb, blk, reference = prepared.segments, prepared.rb, prepared.blk, prepared.reference
        self._refuse_future(segments)
        assert prepared.consume_version == self.version, \
            f"a batch prepared for version {prepared.consume_version} consumed at version {self.version}"
        self.ensure_shaping_start()
        if self._shaping_active:
            assert prepared.shaping_scale == self.shaping_scale(self.updates), \
                f"shaping_scale mismatch: prepared {prepared.shaping_scale} vs expected {self.shaping_scale(self.updates)}"
        cfg = self.cfg
        t0 = time.perf_counter()
        wall = time.time()
        n = len(rb)
        crit_stats = self._critic_mix(prepared.critic) if prepared.critic is not None else None
        self._set_tempered(prepared.tempered)
        temps = prepared.temps
        lr = lr_at(cfg, self.updates)
        for g in self.opt.param_groups:
            g["lr"] = lr if not g.get("critic") else cfg.critic.lr  # type: ignore[union-attr]
        if cfg.normalize_returns:
            self.ret_moments.update(rb.returns.astype(np.float64))
        rows, conf = self._kept_rows(rb)
        n_kept = len(rows)
        n_mb = max(1, round(n_kept / cfg.minibatch))
        early_stop = False
        epochs_done = 0.0
        if cfg.kickstart.on:
            self._kick_begin()
        if self.rules_mode != "mask":
            self._rules_begin()
        self.model.train()
        stats: _Accum
        if blk is not None:
            stats = self._facc
            stats.reset()
            if self.replica is not None:  # B's sums zeroed, the tempered flag and the removal coefficient written
                self.replica.begin(self._tempered, self._rules_coef if self.rules_mode != "mask" else None)
            assert prepared.cols is not None and prepared.buckets is not None
            # the critic's beta > 0 changed the advantages (and the public target): the columns again, on the mixed rb
            pf, pi, masks = self._columns(rb, reference) if crit_stats is not None and self._beta > 0.0 else prepared.cols
            buckets = prepared.buckets
            if self.stepper is not None:
                self.stepper.next_micros = None
                self.stepper.stop_pending = False
                self.stepper.allow_lookahead = self.stepper.replay_is_pure_backward()
            t_prep = time.perf_counter()
            for epoch in range(cfg.epochs):
                parts = np.array_split(rows[self.rng.permutation(n_kept)], n_mb)
                for j, mb in enumerate(parts):
                    ahead = parts[j + 1] if j + 1 < n_mb else None
                    completed = self._minibatch_fast(blk, mb, buckets, pf, pi, masks, stats, epoch, rb.crit, ahead)
                    if completed is not None:
                        early_stop = True
                        epochs_done = epoch + (j + completed) / n_mb
                        break
                if early_stop:
                    break
                epochs_done = epoch + 1.0
            torch.cuda.current_stream(self.device).synchronize()
            if self.replica is not None:  # B's micro-step sums and rules sums into A's (read once below)
                self.replica.merge_into(stats, self._rules_acc)
        else:
            assert self.replica is None, "learner.replica_device: segments without sample blocks (the generic path)"
            tens = self._tensors(rb, reference)
            # a critic learner: the tower's targets (steps, 3) = RET_WIN (clamped to [0, 1]), RET_PROG, RET_SURV
            ret_c = (None if cfg.critic is None else
                     torch.as_tensor(self._critic_targets(rb)).to(self.device, torch.float32))
            sizes = prepared.sizes
            stats = _Accum(self.device)
            t_prep = time.perf_counter()
            for epoch in range(cfg.epochs):
                parts = np.array_split(rows[self.rng.permutation(n_kept)], n_mb)
                for j, mb in enumerate(parts):
                    kl = self._minibatch(rb.samples, mb, sizes, tens, stats, epoch, rb.crit, ret_c)
                    if not self.distil and cfg.target_kl is not None and kl > cfg.target_kl:
                        early_stop = True
                        epochs_done = epoch + (j + 1) / n_mb
                        break
                if early_stop:
                    break
                epochs_done = epoch + 1.0
        exit_metrics = self._exit_step(lr) if cfg.exit_coef > 0.0 else None
        t_train = time.perf_counter()
        self.version += 1
        self.updates += 1
        self.samples_seen += n
        if self.publisher is not None:
            self.publish()
        metrics = stats.result()
        if exit_metrics is not None:
            metrics.update(exit_metrics)
        if cfg.kickstart.on:
            metrics["kickstart"] = self._kick_end()
        if self.rules_mode != "mask":
            assert self._rules_acc is not None
            metrics["rules"] = rules_mod.row_metrics(self.rules_mode, self._rules_coef, self._rules_acc.tolist())
        if crit_stats is not None:
            metrics["critic"] = self._critic_end(crit_stats, stats)
        metrics.update(self._batch_metrics(rb))
        if conf is not None:
            kinds, counts = np.unique(rb.kinds, return_counts=True)
            metrics.update(conf_keep=self.confident_keep, conf_tau=self.confident_tau, conf_taken=int(conf.sum()),
                           conf_dropped=n - n_kept, kept=n_kept,
                           conf_share_by_kind={Kind(int(k)).name: float(conf[rb.kinds == k].sum() / c)
                                               for k, c in zip(kinds, counts)})
            if self.confident_exempt_kinds:
                exempt = self._confident_exempt[rb.kinds.astype(np.int64)]
                metrics["conf_exempt_kept"] = int((conf & exempt).sum())
        if prepared.rloo is not None:
            metrics["rloo_groups"], metrics["rloo_children"], metrics["rloo_shift_std"] = prepared.rloo
        if prepared.shaping_scale is not None:
            metrics["shaping_scale"] = prepared.shaping_scale
        if self.distil:  # the stored values are the teacher's: the row's explained variance is the student's
            metrics["kickstart"]["ev_teacher"] = metrics["explained_variance"]
            metrics["kickstart"]["ev_teacher_by_kind"] = metrics.pop("explained_variance_by_kind")
            metrics["explained_variance"] = metrics["kickstart"]["ev_student"]
        metrics["beh_temp_mean"] = float(temps.mean())
        if self._advice:
            assert rb.advice_row is not None and rb.advice_hp is not None
            metrics["advice"] = advice_metrics(self._advice, rb.kinds, rb.actions, rb.advice_row, rb.advice_hp,
                                               rb.advantages)
        if self._habits:
            assert rb.advice_rule is not None and rb.advice_sign is not None
            metrics.setdefault("advice", {}).update(habit_metrics(self._habits, rb.kinds, rb.advice_rule,
                                                                  rb.advice_sign, rb.advantages))
        if self._taste:
            assert rb.t_keep is not None and rb.t_add is not None, "taste needs segments with taste records"
            metrics["taste"] = taste_learner_metrics(rb.kinds, rb.t_keep, rb.t_add, rb.advantages)
        if reference is not None:
            drift = reference - rb.logp
            metrics["behavior_kl_before_update"] = float(np.mean(np.expm1(drift) - drift))
            importance = np.exp(drift)
            metrics["behavior_importance_p99"] = float(np.percentile(importance, 99))
            metrics["behavior_importance_max"] = float(importance.max())
        # storage cuts in this iteration and how many of them the stitched GAE trace crossed (0 when stitching is off)
        metrics["gae_cut_segments"] = sum(1 for s in segments if s.end == SegmentEnd.CUT)
        metrics["gae_joined_cuts"] = int(stitch_joins(segments).sum()) if cfg.gae.stitch_segments else 0
        metrics["lag_s"] = policy_lag_seconds(segments, wall)
        calc_gate = getattr(self.model, "calc_gate", None)
        if calc_gate is not None:  # the three calc gates (TURN, FOE, ROUTE): 0 at migration, opened by training
            metrics["calc_gate"] = [float(g) for g in calc_gate.detach().float().cpu()]
        ahead_gate = getattr(self.model, "ahead_gate", None)
        if ahead_gate is not None:  # the route summary's gate (notes/inputs2/contract.md 2.4): 0 at birth / migration
            metrics["ahead_gate"] = float(ahead_gate.detach().float().cpu())
        metrics["fast_path"] = blk is not None
        if blk is not None and self.stepper is not None:
            metrics["stepper"] = self.stepper.stats()
        if self.replica is not None:
            metrics["dual"] = self.replica.row()
        metrics.update(
            lr=lr, samples=n, minibatches=n_mb, epochs_done=epochs_done, early_stop=early_stop,
            version=self.version, updates=self.updates, samples_seen=self.samples_seen,
            time_prep=prepared.time_prep + (t_prep - t0), time_train=t_train - t_prep,
            samples_per_s=n * max(epochs_done, 1e-9)
            / max(t_train - t_prep, 1e-9),
        )
        return metrics

    def _kept_rows(self, rb: RolloutBatch) -> tuple[np.ndarray, np.ndarray | None]:
        """The rows of the iteration that go into the minibatches and the confident mask (None with keep 1: every row,
        the minibatches of a learner without the option). A confident row (exp(logp_b) >= tau, the behaviour
        probability the PPO ratio uses) is kept with probability keep, each independently, by a generator of the run
        seed and the update index: the same rows on a resume from the same checkpoint and for both learner cards.
        Everything that needs all rows (GAE, the advantage normalisation, the row's per-kind statistics) has seen them
        before. No re-weighting: the losses average over the kept rows as they average over all rows without the
        option, so a confident sample weighs keep relative to the others - the intended objective."""
        n = len(rb)
        if self.confident_keep == 1.0:
            return np.arange(n), None
        conf = np.exp(rb.logp.astype(np.float64)) >= self.confident_tau
        draw = np.random.default_rng((self.confident_seed, self.updates, 0x636F6E66)).random(n)
        if self.confident_exempt_kinds:
            exempt = self._confident_exempt[rb.kinds.astype(np.int64)]
            rows = np.flatnonzero(~conf | exempt | (draw < self.confident_keep))
        else:
            rows = np.flatnonzero(~conf | (draw < self.confident_keep))
        assert len(rows) > 0, "confident_keep dropped every row of the iteration"
        return rows, conf

    # ------------------------------------------------------------------ fast path (spire.trainstep)
    def _block(self, segments: Sequence[Segment]) -> SampleBlock | None:
        """The iteration's samples as one block, if the fast path applies to these segments."""
        if self.stepper is None or not segments or not all(isinstance(s.samples, SampleBlock) for s in segments):
            return None
        return SampleBlock.concat([s.samples for s in segments])  # type: ignore[misc]

    def _temperature(self, rb: RolloutBatch) -> tuple[np.ndarray, np.ndarray]:
        """The behaviour temperature T(step) of every step (float32; ones for segments without a beh_temp record,
        which only an untempered learner accepts) and the combat-step mask; sets :attr:`_tempered` (and the
        stepper's flag) to whether any step of the iteration was sampled with T != 1 (spire.temperature)."""
        temps, combat, tempered = self._temps(rb)
        self._set_tempered(tempered)
        return temps, combat

    def _set_tempered(self, tempered: bool) -> None:
        self._tempered = tempered
        if self.stepper is not None:
            self.stepper.tempered = tempered

    def _temps(self, rb: RolloutBatch) -> tuple[np.ndarray, np.ndarray, bool]:
        """:meth:`_temperature` without writing the learner's state: (T(step), combat mask, tempered)."""
        combat = rb.in_combat & np.isin(rb.kinds, COMBAT_KINDS)
        if rb.beh_temp is None:
            assert self.cfg.combat_temperature == 1.0, "ppo.combat_temperature != 1 needs segments with beh_temp records"
            temps = np.ones(len(rb), np.float32)
        else:
            temps = rb.beh_temp
            assert ((temps == 1) | combat).all(), "a non-combat step carries a behaviour temperature != 1"
        tempered = bool((temps != 1).any())
        assert not tempered or not (self._taste or self.cfg.decoupled_clip_anchor or self._advice or self._habits), \
            "tempered steps with taste, advice or the decoupled clip anchor"
        return temps, combat, tempered

    def _ent_coef(self, rb: RolloutBatch) -> np.ndarray:
        """Entropy coefficient per sample: by decision kind, divided by the episode's seat count when
        ``ent_coef_seat_scaled`` is set (unchanged for single-seat episodes)."""
        coef = self._ent_by_kind[rb.kinds.astype(np.int64)]
        if not self.cfg.ent_coef_seat_scaled:
            return coef
        assert rb.n_seats.shape == rb.kinds.shape and (rb.n_seats >= 1).all()
        return (coef / rb.n_seats.astype(np.float32)).astype(np.float32)

    def _policy_advantages(self, rb: RolloutBatch) -> np.ndarray:
        """GAE advantages (float32) plus the advice offsets (:func:`spire.advice.advice_offset` of the old-format rules,
        :func:`spire.advice.habit_offset` of the habit rules; never both at one step), before normalisation."""
        adv = rb.advantages.astype(np.float32)
        if self._advice:
            assert rb.advice_row is not None and rb.advice_hp is not None, "advice rules need segments with advice records"
            adv = adv + advice_offset(self._advice, rb.kinds, rb.actions, rb.advice_row, rb.advice_hp)
        if self._habits:
            assert rb.advice_rule is not None and rb.advice_sign is not None, \
                "habit rules need segments with habit-rule records"
            adv = adv + habit_offset(self._habits, rb.advice_rule, rb.advice_sign)
        return adv

    def _columns(self, rb: RolloutBatch, reference: np.ndarray | None = None) \
            -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Per-sample training columns (:data:`spire.trainstep.PF_COLS`, float32; action and kind, int64) and
        the host masks of the minibatch denominators - the quantities of :meth:`_tensors`. Writes no learner state
        (:meth:`prepare`)."""
        cfg = self.cfg
        temps, combat, _ = self._temps(rb)
        adv = self._policy_advantages(rb)
        pol = rb.n_legal > 1
        if cfg.adv_norm and pol.sum() > 1:
            a = adv[pol]
            adv = (adv - a.mean()) / (a.std() + 1e-8)
        n = len(rb)
        pf = np.empty((n, NF if cfg.critic is None else NF_CRITIC), np.float32)
        col = PF_COLS.index
        pf[:, col("logp_b")] = rb.logp
        pf[:, col("logp_ref")] = rb.logp if reference is None else reference
        pf[:, col("adv")] = adv
        pf[:, col("ret")] = rb.returns
        pf[:, col("v_old")] = rb.values
        pf[:, col("pol")] = pol
        pf[:, col("ent_coef")] = self._ent_coef(rb)
        pf[:, col("log_legal")] = np.log(np.maximum(rb.n_legal, 2).astype(np.float32))
        pf[:, col("win")] = rb.win
        pf[:, col("win_mask")] = rb.win_mask
        pf[:, col("floors")] = rb.floors_left / cfg.floors_scale
        pf[:, col("hp")] = rb.hp_post
        pf[:, col("hp_mask")] = rb.hp_mask
        pf[:, col("valid")] = 1.0
        if self._taste:
            assert rb.t_keep is not None and rb.t_add is not None, "taste needs segments with taste records"
            pf[:, col("t_keep")] = rb.t_keep
            pf[:, col("t_add")] = rb.t_add
        else:
            pf[:, col("t_keep")] = 1.0
            pf[:, col("t_add")] = 0.0
        pf[:, col("beh_temp")] = temps
        pf[:, col("combat")] = combat
        if cfg.critic is not None:  # the tower's targets (CRITIC_PF_COLS, after PF_COLS)
            pf[:, NF:NF_CRITIC] = self._critic_targets(rb)
        pi = np.stack([rb.actions.astype(np.int64), rb.kinds.astype(np.int64)], 1)
        return pf, pi, {"pol": pol, "win": rb.win_mask.astype(bool), "hp": rb.hp_mask.astype(bool)}

    def _minibatch_fast(self, blk: SampleBlock, mb: np.ndarray, buckets: np.ndarray, pf: np.ndarray,
                        pi: np.ndarray, masks: dict[str, np.ndarray], acc: "_Accum", epoch: int,
                        crit: np.ndarray | None = None, ahead: np.ndarray | None = None) -> int | None:
        """On early stop, return 0 if the preceding minibatch stopped or 1 if this one did."""
        cfg = self.cfg
        st = self.stepper
        assert st is not None
        rep = self.replica
        saved_a = None
        saved_reps: list[tuple[int, int]] | None = None
        old_micro, old_samples = acc.micro, acc.samples
        if getattr(st, "stop_pending", False):
            st.stop_pending = False
            if st.kl_ready.query():
                if st.should_stop():
                    return 0
            else:
                # Previous weights have already been committed. Only a warmed
                # backward is allowed ahead of its stop verdict; no optimizer,
                # EMA, shape capture, or random draw is speculative.
                saved_a = st.save_micro_state()
                if rep is not None:
                    saved_reps = rep.save_micro_states()
                    old_split = (rep.micro.copy(), rep.load.copy())
        n_pol = float(max(1, int(masks["pol"][mb].sum())))
        win = float(max(1, int(masks["win"][mb].sum())))
        den = (n_pol, win, win, float(max(1, int(masks["hp"][mb].sum()))), float(len(mb)),
               1.0 / self.ret_moments.var if cfg.normalize_returns else 1.0, epoch == 0)
        st.set_den(*den)
        st.mb_kl.zero_()
        if getattr(st, "next_micros", None) is not None and st.next_micros[0] is mb:
            _, micros = st.next_micros
            st.next_micros = None
        else:
            micros = bucket_micros(mb, buckets, cfg)
        pipeline = (self.device.type == "cuda" and st.graphs and crit is None
                    and not cfg.kickstart.on and not st.shared_input)
        if pipeline and st.copy_stream is None:
            st.copy_stream = torch.cuda.Stream(device=st.device)
            if rep is not None:
                rep.ensure_copy_streams()
        if rep is not None:
            rep.set_den(*den)
        if rep is None:
            for bucket, idx in micros:
                if crit is None:
                    st.run(blk, idx, pf[idx], pi[idx], bucket)
                else:
                    st.run(blk, idx, pf[idx], pi[idx], bucket, crit[idx])
                acc.micro += 1
                acc.samples += len(idx)
        else:
            sides = rep.assign(micros)
            futures = rep.enqueue_all(blk, micros, sides, pf, pi)
            primary_err: BaseException | None = None
            worker_err: BaseException | None = None
            try:
                for (bucket, idx), side in zip(micros, sides):
                    if side == 0:
                        st.run(blk, idx, pf[idx], pi[idx], bucket)
            except BaseException as exc:
                primary_err = exc
                raise
            finally:
                for fut in futures:
                    try:
                        fut.result()
                    except BaseException as wexc:
                        if worker_err is None:
                            worker_err = wexc
                if primary_err is None and worker_err is not None:
                    raise worker_err
            acc.micro += len(micros)
            acc.samples += len(mb)
        if saved_a is not None and st.should_stop():
            self.opt.zero_grad(set_to_none=False)
            st.restore_micro_state(saved_a)
            if rep is not None:
                assert saved_reps is not None
                rep.zero_grads()
                rep.restore_micro_states(saved_reps)
                rep.micro, rep.load = old_split
            acc.micro, acc.samples = old_micro, old_samples
            return 0
        # the minibatch's KL sum (with replicas: replicas' gradients added to A's, replicas' KL sums to A's)
        mb_kl = st.mb_kl if rep is None else rep.combine(self._grads_a, st.mb_kl)
        if pipeline and cfg.target_kl is not None:
            st.stage_stop(mb_kl, n_pol, cfg.target_kl)
        gn = torch.nn.utils.clip_grad_norm_(self._params, cfg.grad_clip)
        found = (~torch.isfinite(gn)).to(torch.float32).reshape(())
        if self._cparams:  # the tower: its own clip; a non-finite tower gradient skips the step too (one optimizer)
            gn_c = torch.nn.utils.clip_grad_norm_(self._cparams, cfg.grad_clip)
            found = torch.maximum(found, (~torch.isfinite(gn_c)).to(torch.float32).reshape(()))
            with torch.no_grad():
                acc.critic_gn += torch.where(torch.isfinite(gn_c), gn_c, torch.zeros_like(gn_c)).reshape(())
        self._minibatch_lr(mb_kl / n_pol, acc)
        self.opt.found_inf = found  # type: ignore[attr-defined]  # fused AdamW skips the step on the device
        try:
            self.opt.step()
        finally:
            del self.opt.found_inf  # type: ignore[attr-defined]
        self._ema_step(found)
        with torch.no_grad():
            acc.gn_sum += torch.where(found > 0, torch.zeros_like(gn), gn).reshape(())
            acc.gn_n += 1.0 - found
            acc.skipped_dev += found
        self.opt.zero_grad(set_to_none=False)
        if rep is not None:  # replicas' parameters = A's, replicas' gradients zeroed in place
            rep.sync()
        acc.minibatches += 1
        if cfg.target_kl is None or self.distil:  # phase 0: no early stop (and no host synchronisation)
            return None
        if pipeline and ahead is not None and not st.kl_ready.query():
            following = bucket_micros(ahead, buckets, cfg)
            st.next_micros = (ahead, following)
            sides = rep.assign(following, record=False) if rep is not None else [0] * len(following)
            if (st.allow_lookahead
                    and all((st if side == 0 or rep is None else rep.workers[side - 1].stepper)
                            .already_captured(blk, idx, bucket)
                            for (bucket, idx), side in zip(following, sides))):
                st.stop_pending = True
                return None
            for (bucket, idx), side in zip(following, sides):
                if st.kl_ready.query():
                    break
                if side == 0 or rep is None:
                    st.prepack(blk, idx, pf, pi, bucket)
                else:
                    w = rep.workers[side - 1]
                    with torch.cuda.device(w.device):
                        w.stepper.prepack(blk, idx, pf, pi, bucket)
        stop = st.should_stop() if pipeline else float(mb_kl) / n_pol > cfg.target_kl
        return 1 if stop else None

    def _minibatch_lr(self, kl: Tensor, acc: "_Accum") -> None:
        """Set this step's Adam learning rate from KL already measured by its forward pass.

        A device scalar LR scales the complete Adam displacement, including existing momentum,
        without a GPU-to-host read. The optimizer itself is outside the captured micro-step.
        The scale can only decrease within an iteration and resets to one next iteration.
        A wholly below-threshold iteration uses full LR; no sample or gradient is discarded.
        A NaN KL belongs to a minibatch whose step the optimizer skips (non-finite gradient): it leaves
        the retained scale as it is, so the later minibatches of the iteration do not get a NaN rate.
        """
        if acc.minibatches == 0:
            acc.first_kl.copy_(kl.detach())
        target = self.cfg.kl_lr_target
        if target is None or self.distil:  # phase 0: no behaviour ratio to bound, the configured learning rate
            return
        kl = kl.detach()
        scale = torch.minimum(acc.lr_scale_min, target / kl.clamp_min(target))
        scale = torch.where(torch.isnan(kl), acc.lr_scale_min, scale)
        for group in self.opt.param_groups:
            if not group.get("critic"):  # the tower is not braked (notes/critic/contract.md 4.3)
                group["lr"] = scale * lr_at(self.cfg, self.updates)
        acc.lr_scale_sum += scale
        acc.lr_scale_min.copy_(scale)
        acc.lr_scale_n += 1

    # ------------------------------------------------------------------ kickstart (spire.kickstart)
    def _kick_begin(self) -> None:
        """This update's coefficients (phase 1 with ``fade_updates``: the first update that sees the release file
        becomes ``release_update``; the file is not looked at again); the teacher is dropped for good (with the graphs
        captured with it) once the schedule is over."""
        ks = self.cfg.kickstart
        if ks.phase == 1 and ks.fade_updates is not None and self.kick_release is None:
            assert self.release_file is not None
            if self.release_file.exists():
                self.kick_release = self.kick_u
        if self.teacher is not None and not teacher_active(ks, self.kick_u, self.kick_release, self.kick_refade):
            self.teacher = None
            if self.stepper is not None:
                self.stepper.set_teacher(None)
                torch.cuda.empty_cache()
        self._kick_coef = coefficients(ks, self.kick_u, self.kick_release, self.kick_refade)
        if self.stepper is not None:
            self.stepper.set_kick(*self._kick_coef)
        self._kick_acc.zero_()

    def _rules_begin(self) -> None:
        """This update's removal coefficient (spire.rules.coefficient; the first update in "free" is recorded)."""
        assert self._rules_coef_dev is not None and self._rules_acc is not None
        free = None
        if self.rules_mode == "free":
            if self.rules_free_first is None:
                self.rules_free_first = self.updates
            free = self.updates - self.rules_free_first
        self._rules_coef = rules_mod.coefficient(self.cfg.rules, self.rules_mode, free)
        self._rules_coef_dev.fill_(self._rules_coef)
        self._rules_acc.zero_()

    # ------------------------------------------------------------------ the critic tower (notes/critic/contract.md 4-5)
    def critic_beta(self) -> float:
        """beta of the update about to run (contract 5): ``adv_source`` "public" 0, "tower" 1; "auto" 0 until the
        switch, then (updates - switched_at) / blend_updates, at most 1 (one-way)."""
        c = self.critic_cfg
        assert c is not None
        if c.adv_source == "public":
            return 0.0
        if c.adv_source == "tower":
            return 1.0
        if self.switched_at is None:
            return 0.0
        return min(1.0, (self.updates - self.switched_at) / c.blend_updates)

    def _critic_targets(self, rb: RolloutBatch) -> np.ndarray:
        """(steps, 3) float32 targets of the tower: RET_WIN (asserted within 1e-9 of [0, 1] in float64, then
        clamped), RET_PROG, RET_SURV (contract 4.3). The targets are never weighted (contract 4.2)."""
        assert rb.ret_c is not None
        r = rb.ret_c.copy()
        w = r[:, 0]
        assert bool(((w >= -1e-9) & (w <= 1.0 + 1e-9)).all()), f"RET_WIN outside [0, 1]: {w.min()} .. {w.max()}"
        r[:, 0] = np.clip(w, 0.0, 1.0)
        return r.astype(np.float32)

    def _critic_begin(self, rb: RolloutBatch, segments: Sequence[Segment]) -> dict[str, Any]:
        """This update's advantage A = (1 - beta) A_PUB + beta A_TOWER and (beta > 0, ``public_target`` "tower") the
        public head's target RET_WIN + RET_PROG + RET_SURV, written into ``rb``; A_TOWER = sum_k w_k(ascension) A_k
        with w = (1, 1, 1) asserted (v1). Returns what :meth:`_critic_end` reports: A_PUB, A_TOWER, RET_PUB, the
        one-step TD errors by transition class. beta 0 leaves ``rb`` untouched (the trunk's update is tree p4's).
        = :meth:`_critic_mix` of :meth:`_critic_prepare`."""
        return self._critic_mix(self._critic_prepare(rb, segments))

    def _critic_prepare(self, rb: RolloutBatch, segments: Sequence[Segment]) -> dict[str, Any]:
        """The weight-free, state-free part of :meth:`_critic_begin` (:meth:`prepare`): A_PUB, A_TOWER, RET_PUB, the
        TD errors."""
        c = self.critic_cfg
        assert c is not None
        assert rb.values_c is not None and rb.adv_c is not None and rb.ret_c is not None and rb.crit is not None, \
            "a critic learner needs segments with the critic record (contract 3)"
        asc, inv = np.unique(rb.ascension, return_inverse=True)
        w = np.asarray([c.weights(int(a)) for a in asc], np.float64)[inv]
        assert bool((w == 1.0).all()), "v1 keeps the tower's advantage weights at (1, 1, 1) (build-v1.md 3)"
        return {"a_pub": rb.advantages, "a_tower": (rb.adv_c * w).sum(1), "ret_pub": rb.returns, "rb": rb,
                "td": transition_td(segments, self.cfg.gae.gamma)}

    def _critic_mix(self, st: dict[str, Any]) -> dict[str, Any]:
        """The learner-state part of :meth:`_critic_begin` (:meth:`update_prepared`): the tower's start, the device
        sums, beta of this update and the mixing into ``st["rb"]``."""
        c = self.critic_cfg
        assert c is not None
        rb: RolloutBatch = st["rb"]
        assert rb.ret_c is not None
        if self.critic_first is None:  # a fresh tower (migration or a fresh critic run): its warm-up starts here
            self.critic_first = self.updates
        assert self._critic_acc is not None
        self._critic_acc.zero_()
        beta = self._beta = self.critic_beta()
        if beta == 1.0:
            rb.advantages = st["a_tower"].astype(np.float32)
        elif beta > 0.0:
            rb.advantages = ((1.0 - beta) * st["a_pub"].astype(np.float64) + beta * st["a_tower"]).astype(np.float32)
        if beta > 0.0 and c.public_target == "tower":
            rb.returns = rb.ret_c.sum(1).astype(np.float32)
        return st

    def _critic_end(self, st: dict[str, Any], acc: "_Accum") -> dict[str, Any]:
        """The learner row's ``critic`` record (contract 4.4) after the update; the switch test (contract 5)."""
        c = self.critic_cfg
        assert c is not None and self._critic_acc is not None and self.critic_first is not None
        rb: RolloutBatch = st["rb"]
        assert rb.values_c is not None and rb.ret_c is not None and rb.crit is not None
        mbs = max(1, acc.minibatches)
        l_win, l_prog, l_surv = (x / mbs for x in self._critic_acc.tolist())
        inc = rb.in_combat
        a_pub, a_tower = st["a_pub"].astype(np.float64), st["a_tower"]
        td = st["td"]
        won = td["classes"]["td_end_won"]
        row: dict[str, Any] = {
            "beta": self._beta, "switched_at": self.switched_at, "first_update": self.critic_first,
            "switch_overdue": int(self.switched_at is None and c.adv_source == "auto"
                                  and self.updates - self.critic_first >= 900),
            "loss_win": l_win, "loss_prog": l_prog, "loss_surv": l_surv, "grad_norm": float(acc.critic_gn) / mbs,
            "ev_win": explained_variance(rb.values_c[:, 0], rb.ret_c[:, 0]),
            "ev_prog": explained_variance(rb.values_c[:, 1], rb.ret_c[:, 1]),
            "ev_surv": explained_variance(rb.values_c[:, 2], rb.ret_c[:, 2]),
            "ev_total": explained_variance(rb.values_c.astype(np.float64).sum(1), rb.ret_c.sum(1)),
            "ev_public": explained_variance(rb.values, st["ret_pub"]),
            "var_adv_combat_pub": float(a_pub[inc].var()) if inc.any() else None,
            "var_adv_combat_tower": float(a_tower[inc].var()) if inc.any() else None,
            "var_adv_macro_pub": float(a_pub[~inc].var()) if (~inc).any() else None,
            "var_adv_macro_tower": float(a_tower[~inc].var()) if (~inc).any() else None,
            "draw_rows_combat_mean": float(rb.crit[inc, 1].mean()) if inc.any() else None,
            "ret_win_range": [float(rb.ret_c[:, 0].min()), float(rb.ret_c[:, 0].max())],
            "td": {k: {"n": int(m.sum()), "tower": float(td["tower"][m].mean()) if m.any() else None,
                       "public": float(td["public"][m].mean()) if m.any() else None}
                   for k, m in td["classes"].items()},
        }
        self._critic_acc.zero_()
        # the switch statistics of this update: won-fight-end TD of the tower (count, sum), combat advantages of
        # both baselines (count, sums, sums of squares)
        row.update(self.switch_step([float(won.sum()), float(td["tower"][won].sum()), float(inc.sum()),
                                     float(a_pub[inc].sum()), float(np.square(a_pub[inc]).sum()),
                                     float(a_tower[inc].sum()), float(np.square(a_tower[inc]).sum())]))
        return row

    def switch_step(self, stats: list[float]) -> dict[str, Any]:
        """After an update (``self.updates`` already counts it): add its switch statistics (:func:`switch_test`'s
        row) to the window of the last ``switch_window`` updates and, with ``adv_source`` "auto", not yet switched and
        at least ``warmup_updates`` updates since the tower began, switch at the first update where both conditions
        hold: ``switched_at`` = this update count (one-way; beta = 0 at the next update, then rises over
        ``blend_updates``). Returns the row fields ``switch_test`` (and ``switched_now``)."""
        c = self.critic_cfg
        assert c is not None and self.critic_first is not None and len(stats) == 7
        self._switch_window.append(list(stats))
        del self._switch_window[:-c.switch_window]
        test = switch_test(self._switch_window, c)
        out: dict[str, Any] = {"switch_test": test}
        if (c.adv_source == "auto" and self.switched_at is None and self.updates - self.critic_first >= c.warmup_updates
                and test["pass"]):
            self.switched_at = self.updates
            out["switched_now"] = True
        return out

    def _kick_end(self) -> dict[str, Any]:
        """The learner row's ``kickstart`` record of this update; advances the schedule."""
        u = self.kick_u
        self.kick_u += 1
        row: dict[str, Any] = {"u": u, "phase": self.cfg.kickstart.phase, "pi_coef": self._kick_coef[0],
                               "v_coef": self._kick_coef[1], "teacher_evaluated": self.teacher is not None,
                               "release_update": self.kick_release}
        if self.kick_refade is not None:
            row["refade"] = list(self.kick_refade)
        if self.teacher is not None:
            a = self._kick_acc.tolist()
            kl, v2, agree, n_pol, n_valid = a[:5]
            row.update(kick_kl=kl / max(1.0, n_pol), kick_v=v2 / max(1.0, n_valid), kick_agree=agree / max(1.0, n_pol))
            if self.distil:  # the student's explained variance on the (unseen) minibatches, the summed loss
                var_ret = a[A_RET2] / n_valid - (a[A_RET] / n_valid) ** 2
                var_err = a[A_ERR2] / n_valid - (a[A_ERR] / n_valid) ** 2
                row.update(ev_student=1.0 - var_err / var_ret, loss_total=a[A_LOSS])
        return row

    @torch.no_grad()
    def _recompute(self, segments: Sequence[Segment]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """One frozen pre-update pass for both GAE values and the proximal policy anchor.

        Stored behavior probabilities remain the importance-sampling denominator. The anchor
        only sets PPO's clipping interval and therefore does not discard behavior-policy lag.
        """
        samples = [x for s in segments for x in s.samples]
        values = np.empty(len(samples), np.float32)
        reference = np.empty(len(samples), np.float32) if self.cfg.decoupled_clip_anchor else None
        actions = np.concatenate([s.actions for s in segments])
        kinds = np.concatenate([s.kinds for s in segments]).astype(np.int64)
        was_training = self.model.training
        self.model.eval()
        for mb in self._micro_batches(np.arange(len(samples)), self._sizes(samples)):
            out = self._forward(self.model, [samples[i] for i in mb])
            values[mb] = out["value"].cpu().numpy()
            if reference is not None:
                eps = None if self._exploration is None else self._exploration[kinds[mb]]
                reference_probs = (choice_uniform_probs(torch.isfinite(out["logits"]), out["choice_keys"])
                                   if self.cfg.exploration_choice_uniform else None)
                logp = legal_uniform_log_probs(out["logits"], eps, reference_probs)
                a = torch.as_tensor(actions[mb], device=self.device)
                assert bool(torch.isfinite(out["logits"].gather(1, a[:, None])).all())
                reference[mb] = logp.gather(1, a[:, None]).squeeze(1).cpu().numpy()
        self.model.train(was_training)
        boot = np.fromiter((s.bootstrap_value for s in segments), np.float32, len(segments))
        if self.cfg.recompute_values:
            need = [i for i, s in enumerate(segments) if s.end != SegmentEnd.TERMINAL]
            assert all(segments[i].bootstrap_sample is not None for i in need), \
                "recompute_values requires bootstrap samples for nonterminal segments"
            if need:
                boot[need] = self.predict_values([segments[i].bootstrap_sample for i in need])
        return values, boot, reference

    def _tensors(self, rb: RolloutBatch, reference: np.ndarray | None = None) -> _Tensors:
        cfg = self.cfg
        temps, combat = self._temperature(rb)
        adv = self._policy_advantages(rb)
        pol = rb.n_legal > 1
        if cfg.adv_norm and pol.sum() > 1:
            a = adv[pol]
            adv = (adv - a.mean()) / (a.std() + 1e-8)
        d = self.device

        def t(x: np.ndarray, dtype: torch.dtype = torch.float32) -> Tensor:
            return torch.as_tensor(np.ascontiguousarray(x)).to(d, dtype)

        cols = (dict(t_keep=t(rb.t_keep), t_add=t(rb.t_add)) if self._taste and rb.t_keep is not None and rb.t_add is not None
                else {})
        assert self._taste == bool(cols), "taste needs segments with taste records"
        return (_TasteTensors if cols else _Tensors)(
            actions=t(rb.actions, torch.int64), logp_b=t(rb.logp), adv=t(adv), ret=t(rb.returns),
            logp_ref=t(rb.logp if reference is None else reference),
            v_old=t(rb.values), pol_mask=t(pol, torch.bool),
            log_legal=t(np.log(np.maximum(rb.n_legal, 2).astype(np.float32))),
            ent_coef=t(self._ent_coef(rb)), kinds=t(rb.kinds, torch.int64),
            win=t(rb.win), win_mask=t(rb.win_mask, torch.bool),
            floors=t(rb.floors_left / cfg.floors_scale), floors_mask=t(rb.win_mask, torch.bool),
            hp=t(rb.hp_post), hp_mask=t(rb.hp_mask, torch.bool),
            beh_temp=t(temps), combat=t(combat, torch.bool),
            **cols,
        )

    def _minibatch(self, samples: list[Any], mb: np.ndarray, sizes: np.ndarray | None, T: _Tensors,
                   acc: "_Accum", epoch: int, crit: np.ndarray | None = None, ret_c: Tensor | None = None) -> float:
        """``crit`` / ``ret_c`` (a critic learner): the iteration's CRIT rows and the tower's targets (steps, 3)."""
        cfg = self.cfg
        assert (crit is not None) == (cfg.critic is not None) == (ret_c is not None), "CRIT and targets iff critic"
        mb_t = torch.as_tensor(mb, device=self.device)
        den = {k: max(1.0, float(m[mb_t].sum())) for k, m in
               (("pol", T.pol_mask), ("win", T.win_mask), ("floors", T.floors_mask), ("hp", T.hp_mask))}
        den["all"] = float(len(mb))
        v_scale = 1.0 / self.ret_moments.var if cfg.normalize_returns else 1.0
        self.opt.zero_grad(set_to_none=True)
        kl_sum = torch.zeros((), device=self.device)
        ok = True
        for micro in self._micro_batches(mb, sizes):
            i = torch.as_tensor(micro, device=self.device)
            out = self._forward(self.model, [samples[k] for k in micro], self.teacher,
                                None if crit is None else crit[micro])
            logits = out["logits"]
            legal = torch.isfinite(logits)
            eps = None if self._exploration is None else self._exploration[T.kinds[i]]
            reference_probs = (choice_uniform_probs(legal, out["choice_keys"])
                               if cfg.exploration_choice_uniform else None)
            logp_all = legal_uniform_log_probs(logits, eps, reference_probs)
            # the behaviour's log q of every row, with the stored temperature of the step (spire.temperature)
            logp_beh = tempered_log_probs(logits, T.beh_temp[i], eps, reference_probs) if self._tempered else logp_all
            a = T.actions[i]
            chosen_legal = legal.gather(1, a[:, None]).squeeze(1)
            pm = T.pol_mask[i] & chosen_legal
            logp_b = T.logp_b[i]
            chosen = logp_beh.gather(1, a[:, None]).squeeze(1)
            if self._taste:
                assert isinstance(T, _TasteTensors)
                chosen = taste_log_prob(chosen, T.t_keep[i], T.t_add[i])
            logp = torch.where(pm, chosen, logp_b)
            log_ratio = logp - logp_b
            ratio = log_ratio.exp()
            adv = T.adv[i]
            if cfg.decoupled_clip_anchor:
                logp_ref = torch.where(pm, T.logp_ref[i], logp_b)
                proximal_log_ratio = logp - logp_ref
                proximal_ratio = proximal_log_ratio.exp()
                anchor_weight = (logp_ref - logp_b).exp()
                clipped = anchor_weight * proximal_ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                surr = torch.minimum(ratio * adv, clipped * adv)
            else:
                proximal_log_ratio, proximal_ratio = log_ratio, ratio
                surr = torch.minimum(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
            pmf = pm.float()
            loss_pi = -(surr * pmf).sum() / den["pol"]
            if self.distil:  # phase 0: no PPO surrogate, entropy bonus or uniform barrier
                loss_pi = logits.new_zeros(())
            probs = torch.where(legal, logp_all.exp(), 0.0)
            ent = -(probs * logp_all).sum(-1)
            ent_n = ent / T.log_legal[i]
            bonus = ((ent_n if cfg.ent_normalize else ent) * T.ent_coef[i] * pmf).sum() / den["pol"]
            if self.distil:
                bonus = logits.new_zeros(())
            uniform_ce = logits.new_zeros(())
            if self._uniform_kl is not None and not self.distil:
                uniform_ce = (uniform_cross_entropy(logits, reference_probs)
                              * self._uniform_kl[T.kinds[i]] * pmf).sum() / den["pol"]

            v = out["value"]
            ret = T.ret[i]
            if self._hl_gauss:
                assert self._val_edges is not None
                p_tgt = hl_gauss_probs(ret, self._val_edges, self._val_vmin, self._val_vmax, self._val_sigma)
                loss_v = -(p_tgt * F.log_softmax(out["value_logits"], dim=-1)).sum(-1).sum() / den["all"]
            else:
                err2 = (v - ret).square()
                if cfg.value_clip is not None:
                    v_old = T.v_old[i]
                    v_c = v_old + (v - v_old).clamp(-cfg.value_clip, cfg.value_clip)
                    err2 = torch.maximum(err2, (v_c - ret).square())
                loss_v = 0.5 * err2.sum() / den["all"] * v_scale
            wm, fm, hm = T.win_mask[i].float(), T.floors_mask[i].float(), T.hp_mask[i].float()
            loss_win = (F.binary_cross_entropy_with_logits(out["pwin_logit"], T.win[i], reduction="none")
                        * wm).sum() / den["win"]
            loss_fl = ((out["floors_pred"] - T.floors[i]).square() * fm).sum() / den["floors"]
            loss_hp = ((out["hp_pred"] - T.hp[i]).square() * hm).sum() / den["hp"]
            loss = (loss_pi - bonus + uniform_ce + cfg.vf_coef * loss_v + cfg.aux_win * loss_win
                    + cfg.aux_floors * loss_fl + cfg.aux_hp * loss_hp)
            if self.teacher is not None:
                loss = loss + kick_loss(out, pmf, torch.ones_like(pmf), den["pol"], den["all"], *self._kick_coef,
                                        self._kick_acc, None if self._smooth is None else self._smooth[T.kinds[i]])
            if self.distil:
                distil_sums(v, ret, torch.ones_like(pmf), loss, self._kick_acc)
            if self._rules_coef_dev is not None:  # spire.rules: the removal teach term
                assert self._rules_acc is not None
                loss = loss + teach_loss(out, torch.ones_like(pmf), den["all"], self._rules_coef_dev,
                                         1.0 if epoch == 0 else 0.0, self._rules_acc)
            if cfg.critic is not None:  # the tower's losses (spire.trainstep.critic_loss, the captured step's terms)
                assert ret_c is not None and self._critic_acc is not None
                rc = ret_c[i]
                loss = loss + critic_loss(out, rc[:, 0], rc[:, 1], rc[:, 2], torch.ones_like(pmf), den["all"],
                                          cfg.critic.coef, self._critic_acc)
            if not torch.isfinite(loss):
                ok = False
                break
            loss.backward()
            with torch.no_grad():
                base_logp = legal_uniform_log_probs(logits) if self._exploration is not None else logp_all
                base_ent = -(torch.where(legal, base_logp.exp(), 0.0) * base_logp).sum(-1)
                kl = ((proximal_ratio - 1) - proximal_log_ratio) * pmf
                if cfg.decoupled_clip_anchor:
                    kl = kl * anchor_weight
                if self.distil:  # no behaviour ratio: zeros for the PPO terms
                    kl, proximal_ratio = torch.zeros_like(pmf), torch.ones_like(pmf)
                kl_sum += kl.sum()
                if T.combat is not None:
                    cm = T.combat[i].float() * pmf
                    acc.add_combat(-(torch.where(legal, logp_beh.exp(), 0.0) * logp_beh).sum(-1) * cm, base_ent * cm, cm)
                acc.add_micro(1.0 if epoch == 0 else 0.0, loss_pi, loss_v, loss_win, loss_fl, loss_hp, bonus,
                              kl.sum(), ((proximal_ratio - 1).abs() > cfg.clip).float().mul(pmf).sum(),
                              (proximal_ratio - 1).abs().mul(pmf).sum(), pmf.sum(), T.kinds[i], ent * pmf, ent_n * pmf, pmf,
                              (~chosen_legal).sum(), len(micro), uniform_ce,
                              base_ent * pmf, base_ent / T.log_legal[i] * pmf)
                acc.micro += 1
                acc.samples += len(micro)
        params = [p for g in self.opt.param_groups if not g.get("critic") for p in g["params"]]
        if ok:
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            ok = bool(torch.isfinite(gn))
            acc.grad_norm(float(gn) if ok else float("nan"))
        if ok and self._cparams:  # the tower: its own clip (notes/critic/contract.md 4.3)
            gn_c = torch.nn.utils.clip_grad_norm_(self._cparams, cfg.grad_clip)
            ok = bool(torch.isfinite(gn_c))
            with torch.no_grad():
                acc.critic_gn += gn_c.reshape(()) if ok else 0.0
        if ok:
            self._minibatch_lr(kl_sum / den["pol"], acc)
            self.opt.step()
            self._ema_step()
        else:
            self.opt.zero_grad(set_to_none=True)
            acc.skipped += 1
        acc.minibatches += 1
        return float(kl_sum) / den["pol"]

    def _exit_step(self, lr: float) -> dict[str, Any]:
        """One bounded ExIt cross-entropy / soft-MPO auxiliary step from teacher shards (never alters PPO CUDA graphs)."""
        assert self._exit_buf is not None and self._exit_rng is not None
        cfg = self.cfg
        t0 = time.perf_counter()
        v2_on = (
            cfg.exit_recency_halflife > 0
            or cfg.exit_legacy_weight < 1.0
            or cfg.exit_retire_legacy_at > 0
            or cfg.exit_target_cap > 0.0
            or cfg.exit_soft_beta > 0.0
            or cfg.exit_precond
            or cfg.exit_apply_legal_mask
        )
        ingested, evicted, corrupt = self._exit_buf.poll_and_evict(self.updates)
        buf_n = len(self._exit_buf)
        row: dict[str, Any] = {
            "exit_loss": None,
            "exit_buffer_n": buf_n,
            "exit_ingested": ingested,
            "exit_evicted": evicted,
            "exit_corrupt_skipped": corrupt,
            "exit_target_prob_mean": None,
            "exit_top1_acc": None,
            "exit_step_kl": None,
            "exit_grad_norm": None,
            "exit_lr_scale": self._exit_lr_scale,
            "exit_step_ms": 0.0,
        }
        if v2_on:
            n_live, n_legacy = self._exit_buf.counts()
            row.update(
                exit_live_n=n_live,
                exit_legacy_n=n_legacy,
                exit_live_share=None,
                exit_live_top1_acc=None,
                exit_active_frac=None,
            )
        if buf_n < cfg.exit_min_samples:
            row["exit_step_ms"] = (time.perf_counter() - t0) * 1e3
            return row

        recs, is_leg = self._exit_buf.sample_with_flags(
            cfg.exit_batch_size, self._exit_rng, current_update=self.updates
        )
        if not recs:
            row["exit_step_ms"] = (time.perf_counter() - t0) * 1e3
            return row
        samples = [r.sample for r in recs]
        tgt = torch.as_tensor([r.target_action for r in recs], dtype=torch.long, device=self.device)
        gains = torch.as_tensor([r.gain for r in recs], dtype=torch.float32, device=self.device)
        w_np = np.clip(
            np.asarray([r.gain / cfg.exit_gain_scale for r in recs], dtype=np.float32),
            cfg.exit_weight_min,
            cfg.exit_weight_max,
        )
        weights = torch.as_tensor(w_np, dtype=torch.float32, device=self.device)
        live_mask = torch.as_tensor([not leg for leg in is_leg], dtype=torch.bool, device=self.device)
        if v2_on:
            row["exit_live_share"] = float(live_mask.float().mean())

        # Preserve persistent gradient buffer addresses on the CUDA-graph fast path,
        # and run forward+backward on MicroStepper's capture stream so AccumulateGrad nodes
        # never see a cross-stream mismatch if a later PPO micro-batch captures a new graph shape.
        set_to_none = self.stepper is None
        use_side = bool(self.stepper is not None and self.stepper.graphs and self.device.type == "cuda")
        self.opt.zero_grad(set_to_none=set_to_none)
        cur_stream = torch.cuda.current_stream(self.device) if use_side else None
        side_stream: torch.cuda.Stream | None = None
        if use_side:
            assert self.stepper is not None
            if self.stepper._side is None:
                self.stepper._side = torch.cuda.Stream(self.device)
            side_stream = self.stepper._side
            assert cur_stream is not None
            side_stream.wait_stream(cur_stream)

        def _apply_rec_masks(logits: Tensor) -> Tensor:
            if not cfg.exit_apply_legal_mask:
                return logits
            B, A = logits.shape
            mask = torch.ones((B, A), dtype=torch.bool, device=logits.device)
            any_mask = False
            for i, r in enumerate(recs):
                if r.legal_mask is not None:
                    any_mask = True
                    m = r.legal_mask
                    k = min(len(m), A)
                    mask[i, :k] = torch.as_tensor(m[:k], dtype=torch.bool, device=logits.device)
                    if k < A:
                        mask[i, k:] = False
            return logits.masked_fill(~mask, -torch.inf) if any_mask else logits

        def _fwd_bwd() -> tuple[float | None, float, float, float | None, float | None, Tensor, Tensor, Tensor, bool]:
            out = self._forward(self.model, samples)
            logits = _apply_rec_masks(out["logits"])
            legal = torch.isfinite(logits)
            n_act_pad = logits.shape[1]
            in_bounds = (tgt >= 0) & (tgt < n_act_pad)
            safe_tgt = tgt.clamp(0, max(0, n_act_pad - 1))
            valid = in_bounds & legal.gather(1, safe_tgt[:, None]).squeeze(1)
            if not bool(valid.any()):
                return None, 0.0, 0.0, None, None, torch.empty(0), torch.empty(0), valid, False
            logp = legal_uniform_log_probs(logits)
            tgt_logp = logp.gather(1, safe_tgt[:, None]).squeeze(1)
            with torch.no_grad():
                probs_pre = torch.where(legal, logp.exp(), 0.0).detach()
                logp_pre = logp.detach()
                tp_row = tgt_logp.detach().exp()
                tp_mean = float(tp_row[valid].mean())
                top1_ok = logits.detach().argmax(dim=-1) == safe_tgt
                t1_acc = float(top1_ok[valid].float().mean())
                lv_valid = valid & live_mask
                live_t1 = float(top1_ok[lv_valid].float().mean()) if bool(lv_valid.any()) else None

            # Per-row objective: optional Boltzmann soft K-candidate MPO target, else 1-hot / margin-capped CE
            per_row_loss = -tgt_logp
            eff_w = weights * valid.float()
            active_frac: float | None = None
            if cfg.exit_soft_beta > 0.0:
                soft_mask = torch.zeros(len(recs), dtype=torch.bool, device=self.device)
                soft_loss = torch.zeros(len(recs), dtype=torch.float32, device=self.device)
                for i, r in enumerate(recs):
                    if not bool(valid[i]) or not r.cand_scores or len(r.cand_scores) != len(r.cand_rows):
                        continue
                    c_idx = [c for c in r.cand_rows if 0 <= c < n_act_pad and bool(legal[i, c])]
                    if len(c_idx) < 2:
                        continue
                    c_sc = [s for c, s in zip(r.cand_rows, r.cand_scores) if 0 <= c < n_act_pad and bool(legal[i, c])]
                    sc_t = torch.tensor(c_sc, dtype=torch.float32, device=self.device)
                    q_t = F.softmax(cfg.exit_soft_beta * (sc_t - sc_t.max()), dim=-1)
                    idx_t = torch.tensor(c_idx, dtype=torch.long, device=self.device)
                    soft_loss[i] = -(q_t * logp[i, idx_t]).sum()
                    soft_mask[i] = True
                per_row_loss = torch.where(soft_mask, soft_loss, per_row_loss)

            if cfg.exit_target_cap > 0.0:
                # Gain-adaptive target cap: zero gradient once p(c*) >= p_cap(gain), full gradient on blind spots
                p_cap = (0.55 + 2.0 * gains).clamp(max=cfg.exit_target_cap).clamp_min(0.1)
                margin = ((p_cap - tp_row).clamp_min(0.0) / p_cap)
                if cfg.exit_soft_beta > 0.0:
                    # Keep soft-target rows active (they naturally self-stop at pi == q_t)
                    margin = torch.where(soft_mask, torch.ones_like(margin), margin)
                eff_w = eff_w * margin
                active_frac = float((eff_w[valid] > 0.0).float().mean())
            elif v2_on:
                active_frac = 1.0

            w_sum = eff_w.sum()
            if float(w_sum.detach()) <= 0.0:
                return 0.0, tp_mean, t1_acc, live_t1, active_frac, probs_pre, logp_pre, valid, False

            ce = (per_row_loss * eff_w).sum() / w_sum.clamp_min(1e-8)
            loss_exit = ce if cfg.exit_precond else (cfg.exit_coef * ce)
            if not bool(torch.isfinite(loss_exit)):
                return None, tp_mean, t1_acc, live_t1, active_frac, probs_pre, logp_pre, valid, True
            loss_val = float((cfg.exit_coef * ce).detach())
            loss_exit.backward()
            return loss_val, tp_mean, t1_acc, live_t1, active_frac, probs_pre, logp_pre, valid, True

        if side_stream is not None:
            assert cur_stream is not None
            with torch.cuda.stream(side_stream):
                (loss_val, target_prob_mean, top1_acc, live_top1_acc, active_frac,
                 probs_pre, logp_pre, valid, has_valid) = _fwd_bwd()
            cur_stream.wait_stream(side_stream)
        else:
            (loss_val, target_prob_mean, top1_acc, live_top1_acc, active_frac,
             probs_pre, logp_pre, valid, has_valid) = _fwd_bwd()

        if v2_on:
            row["exit_live_top1_acc"] = live_top1_acc
            row["exit_active_frac"] = active_frac
        if not has_valid:
            if loss_val == 0.0:
                row.update(
                    exit_loss=0.0,
                    exit_target_prob_mean=target_prob_mean,
                    exit_top1_acc=top1_acc,
                    exit_step_kl=0.0,
                    exit_grad_norm=0.0,
                )
            row["exit_step_ms"] = (time.perf_counter() - t0) * 1e3
            return row
        if loss_val is None:
            self.opt.zero_grad(set_to_none=set_to_none)
            row.update(
                exit_target_prob_mean=target_prob_mean,
                exit_top1_acc=top1_acc,
                exit_step_ms=(time.perf_counter() - t0) * 1e3,
            )
            return row

        gn = torch.nn.utils.clip_grad_norm_(self._params, cfg.grad_clip)
        gn_f = float(gn)
        if not math.isfinite(gn_f):
            self.opt.zero_grad(set_to_none=set_to_none)
            row.update(
                exit_loss=loss_val,
                exit_target_prob_mean=target_prob_mean,
                exit_top1_acc=top1_acc,
                exit_grad_norm=float("nan"),
                exit_step_ms=(time.perf_counter() - t0) * 1e3,
            )
            return row

        if cfg.exit_precond:
            # Momentum-free Adam-variance-preconditioned ExIt step: reads PPO's exp_avg_sq read-only,
            # never mutates PPO's exp_avg/exp_avg_sq/step, and never coasts on stale PPO momentum.
            exit_step_lr = lr * cfg.exit_coef * self._exit_lr_scale
            beta2 = cfg.betas[1]
            eps = cfg.adam_eps
            with torch.no_grad():
                for p in self._params:
                    g = p.grad
                    if g is None:
                        continue
                    st = self.opt.state.get(p)
                    if st is not None and "exp_avg_sq" in st:
                        step_val = float(st["step"])
                        bc2 = max(1e-12, 1.0 - (beta2 ** step_val)) if step_val > 0.0 else 1.0
                        denom = (st["exp_avg_sq"] / bc2).sqrt().add_(eps)
                    else:
                        denom = g.pow(2).mean().sqrt().clamp_min(eps)
                    p.addcdiv_(g, denom, value=-exit_step_lr)
        else:
            saved_lrs = [g["lr"] for g in self.opt.param_groups]
            exit_lr = lr * self._exit_lr_scale
            for g in self.opt.param_groups:
                if not g.get("critic"):
                    g["lr"] = exit_lr
            try:
                self.opt.step()
            finally:
                for g, old_lr in zip(self.opt.param_groups, saved_lrs):
                    g["lr"] = old_lr
        self.opt.zero_grad(set_to_none=set_to_none)
        self._ema_step()
        if self.replica is not None:
            self.replica.sync()

        with torch.no_grad():
            out_post = self._forward(self.model, samples)
            logits_post = _apply_rec_masks(out_post["logits"])
            logp_post = legal_uniform_log_probs(logits_post)
            kl_row = (probs_pre * (logp_pre - logp_post)).sum(-1)
            step_kl = float(kl_row[valid].mean().clamp_min(0.0))
        if step_kl > cfg.exit_max_kl_guard:
            ratio = cfg.exit_max_kl_guard / max(step_kl, 1e-12)
            self._exit_lr_scale = max(0.05, self._exit_lr_scale * max(0.5, ratio))
        else:
            self._exit_lr_scale = min(1.0, self._exit_lr_scale * 1.05)

        row.update(
            exit_loss=loss_val,
            exit_target_prob_mean=target_prob_mean,
            exit_top1_acc=top1_acc,
            exit_step_kl=step_kl,
            exit_grad_norm=gn_f,
            exit_lr_scale=self._exit_lr_scale,
            exit_step_ms=(time.perf_counter() - t0) * 1e3,
        )
        return row

    @torch.no_grad()
    def _ema_step(self, found: Tensor | None = None) -> None:
        """EMA of the online weights with warm-up (``decay_t = min(decay, (1 + t) / (10 + t))``, py-review 5.1:
        without it a short run's EMA keeps a large share of the random initialisation). ``found`` (device
        scalar, 1 = this minibatch's step was skipped as non-finite) zeroes the update without a sync."""
        w = ema_weight(self.cfg.ema_decay, self.ema_steps)
        self.ema_steps += 1
        ema_p: list[Tensor] = list(self.ema.parameters())
        on_p: list[Tensor] = [p.detach() for p in self.model.parameters()]
        if found is None:
            for e, p in zip(ema_p, on_p):
                e.lerp_(p, w)
        else:
            d: Any = torch._foreach_sub(on_p, ema_p)  # type: ignore[arg-type]
            torch._foreach_mul_(d, (1.0 - found) * w)
            torch._foreach_add_(ema_p, d)  # type: ignore[arg-type]
        for eb, b in zip(self.ema.buffers(), self.model.buffers()):
            eb.copy_(b)

    def _batch_metrics(self, rb: RolloutBatch) -> dict[str, Any]:
        ev_kind = {}
        kinds, counts = np.unique(rb.kinds, return_counts=True)
        for k in np.unique(rb.kinds):
            sel = rb.kinds == k
            if sel.sum() > 1:
                ev_kind[Kind(int(k)).name] = explained_variance(rb.values[sel], rb.returns[sel])
        return {
            "explained_variance": explained_variance(rb.values, rb.returns),
            "explained_variance_by_kind": ev_kind,
            "sample_counts_by_kind": {Kind(int(k)).name: int(n) for k, n in zip(kinds, counts)},
            "sample_share_by_kind": {Kind(int(k)).name: float(n / len(rb.kinds)) for k, n in zip(kinds, counts)},
            "adv_mean": float(rb.advantages.mean()), "adv_std": float(rb.advantages.std()),
            "return_mean": float(rb.returns.mean()), "value_mean": float(rb.values.mean()),
            "reward_mean": float(rb.rewards.mean()),
            "frac_single_action": float((rb.n_legal <= 1).mean()),
            "staleness_mean": float(rb.staleness.mean()), "staleness_max": int(rb.staleness.max()),
            "segment_staleness_hist": {int(k): int(c) for k, c in zip(*np.unique(rb.segment_staleness,
                                                                                  return_counts=True))},
            "return_std_running": math.sqrt(self.ret_moments.var) if self.cfg.normalize_returns else None,
        }

    # ------------------------------------------------------------------ weights / checkpoints
    def acting_state_dict(self) -> dict[str, Tensor]:
        """CPU copy of the weights actors act with (one device->host copy per dtype): the online weights, or with
        ``acting_weights`` "ema" the EMA weights (the same network: key set and dtypes of the online weights)."""
        return state_to_cpu(self.ema if self.acting_weights == "ema" else self.model)

    def publish(self) -> None:
        if self.publisher is None:
            raise RuntimeError("no publisher configured")
        self.publisher.publish(self.acting_state_dict(), self.version)

    def state(self) -> dict[str, Any]:
        rng: dict[str, Any] = {"torch": torch.get_rng_state(), "numpy": self.rng.bit_generator.state}
        if torch.cuda.is_initialized():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        counters: dict[str, Any] = {"version": self.version, "updates": self.updates, "samples_seen": self.samples_seen,
                                    "ema_steps": self.ema_steps,
                                    "negative_staleness_samples": self.negative_staleness_samples}
        if self.cfg.kickstart.on:  # schedule position, phase, release and the teacher's identity (never its weights)
            counters["kickstart"] = {"u": self.kick_u, "teacher": self.cfg.kickstart.teacher,
                                     "teacher_sha256": self.teacher_sha256, "phase": self.cfg.kickstart.phase,
                                     "release_update": self.kick_release}
            if self.kick_refade is not None:  # tree p7: the second fade of a lowered floor
                counters["kickstart"]["refade"] = list(self.kick_refade)
        if self.rules_mode != "mask":  # spire.rules: the mode and the first update in "free" (None = not yet)
            counters["rules"] = {"mode": self.rules_mode, "free_first_update": self.rules_free_first}
        if self.critic_cfg is not None:  # the tower's start, the one-way switch and the switch window (contract 5)
            counters["critic"] = {"first_update": self.critic_first, "switched_at": self.switched_at,
                                  "window": [list(r) for r in self._switch_window]}
        if self._shaping_active:  # p12: smooth progress-shaping fade start update
            counters["shaping"] = {"start_update": self.shaping_start_update}
        return {
            "format": 1, "config": self.cfg.to_dict(),
            "model": self.model.state_dict(), "ema": self.ema.state_dict(), "optimizer": self.opt.state_dict(),
            "counters": counters,
            "ret_moments": self.ret_moments.state(), "rng": rng,
        }

    def save(self, path: str | Path, *, extra: Mapping[str, Any] | None = None) -> None:
        """Atomic checkpoint: model, EMA, optimiser, counters, config, RNG states, plus an optional
        ``extra`` dict of plain data (e.g. the model config needed to rebuild the network)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        st = self.state()
        if extra is not None:
            st["extra"] = dict(extra)
        torch.save(st, tmp)
        _retry(lambda: os.replace(tmp, path))

    def load(self, path: str | Path, *, allow_config_change: bool = False,
             migrate_public_history: bool = False, migrate_seat_symmetry: bool = False,
             reset_seat_symmetry_optimizer: bool = False,
             migrate_coop_features: bool = False, migrate_forecast: bool = False,
             migrate_calc: bool = False, migrate_inputs2: bool = False, migrate_rules: bool = False,
             rules_free: bool = False, migrate_critic: bool = False,
             migrate_hlgauss: bool = False) -> dict[str, Any]:
        """Restore strictly, with explicit one-time history, seat-symmetry or co-op migration.

        Migration is one-way and explicit. Old online/EMA weights and Adam states are preserved;
        the history matrix is zero and receives fresh Adam moments on its first update.
        Seat-symmetry migration copies Adam moments with their source weights and preserves
        steps and saved group settings. Its full optimizer reset requires an explicit option.
        Co-op migration appends zero input columns, screen embeddings and/or a MEND gate;
        existing Adam state remains exact and new entries start at zero. Forecast migration adds the
        FORECAST type row, ``forecast_in`` and the zero ``forecast_gate`` the same way; calc migration the TURN / FOE /
        ROUTE type rows, ``calc_in`` and the zero ``calc_gate`` (notes/calc/contract.md 4); inputs2 migration the
        ``encounter#second`` id rows (copies), the AHEAD type row, ``ahead_in`` and the zero ``ahead_gate``, and it may
        switch a finished kickstart off (notes/inputs2/v2a.md 3).
        spire.rules: the checkpoint's rules_mode (none recorded = "mask") must be this learner's; ``migrate_rules``
        ("mask" -> "teach", no active kickstart teacher) and ``rules_free`` ("teach" -> "free") change nothing but the
        mode (no weight or optimizer state is added).
        ``migrate_critic`` (notes/critic/contract.md 5): a checkpoint without the tower into a critic learner; the
        tower's weights are drawn as at birth, its Adam groups start empty, every old entry is exact; ``ppo.critic``
        is the only accepted config difference.
        Online/EMA, counters, return moments and RNG are otherwise retained.
        """
        ck = torch.load(Path(path), map_location="cpu", weights_only=True)
        if ck.get("format") != 1:
            raise ValueError(f"{path}: unknown checkpoint format {ck.get('format')}")
        # Missing optional fields in an older checkpoint mean their declared defaults.
        # Reconstructing the configs also rejects unknown fields instead of ignoring them.
        saved = dict(ck["config"])
        saved_gae = GAEConfig(**saved.pop("gae"))
        saved_taste = TasteConfig(**saved.pop("taste")) if "taste" in saved else TasteConfig()  # pre-taste checkpoints
        saved_kick = KickstartConfig(**saved.pop("kickstart")) if "kickstart" in saved else KickstartConfig()
        saved_rules = RulesConfig(**saved.pop("rules")) if "rules" in saved else RulesConfig()  # pre-rules checkpoints
        saved_cfg = PPOConfig(**saved, gae=saved_gae, taste=saved_taste, kickstart=saved_kick, rules=saved_rules)
        compared = saved_cfg
        if saved_kick.on and self.cfg.kickstart.on and saved_kick.phase == 1 and \
                saved_kick.floor != self.cfg.kickstart.floor:
            # tree p7: a changed kickstart.floor is the one accepted kickstart change (refade_after_resume: before the
            # release any floor, after it only a lower one). Tree p7b: not for a phase-0 checkpoint (it cannot hold a
            # floor); its floor comes with the phase switch, which needs allow_config_change as in tree p7
            compared = replace(compared, kickstart=replace(compared.kickstart, floor=self.cfg.kickstart.floor))
        if migrate_inputs2 and saved_kick.on and not self.cfg.kickstart.on:
            # v2a.md 3: the migration switches the kickstart off; its schedule must be over (asserted below)
            compared = replace(saved_cfg, kickstart=self.cfg.kickstart)
        if migrate_critic:
            assert saved_cfg.critic is None and self.cfg.critic is not None, \
                "--migrate-critic: a checkpoint without ppo.critic into a learner with it"
            compared = replace(compared, critic=self.cfg.critic)
        if not allow_config_change and _normalise(compared.to_dict()) != _normalise(self.cfg.to_dict()):
            raise ValueError(f"{path}: checkpoint config differs from the learner's (allow_config_change=True "
                             "to override)")
        model_state, ema_state, optimizer_state = ck["model"], ck["ema"], ck["optimizer"]
        assert sum((migrate_public_history, migrate_seat_symmetry, migrate_coop_features, migrate_forecast,
                    migrate_calc, migrate_inputs2, migrate_rules, rules_free, migrate_critic,
                    migrate_hlgauss)) <= 1, \
            "perform checkpoint migrations separately"
        saved_rules_c = ck["counters"].get("rules")  # spire.rules: none recorded = a "mask" checkpoint
        saved_mode = "mask" if saved_rules_c is None else saved_rules_c["mode"]
        if migrate_rules:
            assert (saved_mode, self.rules_mode) == ("mask", "teach"), \
                f"--migrate-rules is mask -> teach, not {saved_mode} -> {self.rules_mode}"
            assert self.teacher is None, "--migrate-rules refuses an active kickstart teacher"
        elif rules_free:
            assert (saved_mode, self.rules_mode) == ("teach", "free"), \
                f"--rules-free is teach -> free, not {saved_mode} -> {self.rules_mode}"
        else:
            assert saved_mode == self.rules_mode, \
                f"{path}: checkpoint rules_mode {saved_mode}, learner {self.rules_mode} (--migrate-rules / --rules-free)"
        self.rules_free_first = None if saved_rules_c is None else saved_rules_c["free_first_update"]
        assert not reset_seat_symmetry_optimizer or migrate_seat_symmetry, \
            "optimizer reset requires seat-symmetry migration"
        if migrate_public_history:
            assert isinstance(self.model, SpireNet) and isinstance(self.ema, SpireNet)
            model_state = migrate_public_history_weights(model_state, self.model)
            ema_state = migrate_public_history_weights(ema_state, self.ema)
            optimizer_state = _migrate_history_optimizer(optimizer_state, self.model, self.opt)
        if migrate_seat_symmetry:
            assert isinstance(self.model, SpireNet) and isinstance(self.ema, SpireNet)
            assert isinstance(self.tokenizer, Tokenizer)
            assert not self.opt.state, "seat migration must precede optimizer updates"
            model_state = migrate_seat_symmetry_weights(model_state, self.model, self.tokenizer.layout)
            ema_state = migrate_seat_symmetry_weights(ema_state, self.ema, self.tokenizer.layout)
            optimizer_state = migrate_seat_symmetry_optimizer(
                optimizer_state, self.model, self.tokenizer.layout, self.opt, reset=reset_seat_symmetry_optimizer)
        if migrate_coop_features:
            assert isinstance(self.model, SpireNet) and isinstance(self.ema, SpireNet)
            assert isinstance(self.tokenizer, Tokenizer)
            model_state = migrate_coop_features_weights(model_state, self.model, self.tokenizer.layout)
            ema_state = migrate_coop_features_weights(ema_state, self.ema, self.tokenizer.layout)
            optimizer_state = migrate_coop_features_optimizer(optimizer_state, self.model,
                                                               self.tokenizer.layout, self.opt)
        if migrate_forecast:
            assert isinstance(self.model, SpireNet) and isinstance(self.ema, SpireNet)
            assert isinstance(self.tokenizer, Tokenizer)
            model_state = migrate_forecast_weights(model_state, self.model, self.tokenizer.layout)
            ema_state = migrate_forecast_weights(ema_state, self.ema, self.tokenizer.layout)
            optimizer_state = migrate_forecast_optimizer(optimizer_state, self.model, self.tokenizer.layout, self.opt)
        if migrate_calc:
            assert isinstance(self.model, SpireNet) and isinstance(self.ema, SpireNet)
            assert isinstance(self.tokenizer, Tokenizer)
            model_state = migrate_calc_weights(model_state, self.model, self.tokenizer.layout)
            ema_state = migrate_calc_weights(ema_state, self.ema, self.tokenizer.layout)
            optimizer_state = migrate_calc_optimizer(optimizer_state, self.model, self.tokenizer.layout, self.opt)
        if migrate_inputs2:
            assert isinstance(self.model, SpireNet) and isinstance(self.ema, SpireNet)
            assert isinstance(self.tokenizer, Tokenizer)
            assert not self.cfg.kickstart.on, "inputs2 migration: ppo.kickstart must be off (v2a.md 3)"
            after = self.tokenizer.layout
            before = Tokenizer(self.tokenizer.vocab, config_before(self.tokenizer.config), native=False).layout
            model_state = migrate_inputs2_weights(model_state, self.model, before, after)
            ema_state = migrate_inputs2_weights(ema_state, self.ema, before, after)
            optimizer_state = migrate_inputs2_optimizer(optimizer_state, self.model, before, after, self.opt)
        if migrate_critic:
            assert isinstance(self.model, SpireNet) and isinstance(self.ema, SpireNet)
            if any(k.startswith("critic.") for k in model_state):
                # migrated on the CPU by scripts/migrate_critic.py (its proof in extra["critic_migration"]): the
                # tower must be the one at birth and the optimizer already hold its groups; nothing is changed
                assert "critic_migration" in ck.get("extra", {}), f"{path}: a tower without the CPU migration's proof"
                birth = tower_init_state(self.model)
                for st in (model_state, ema_state):
                    assert all(torch.equal(st[k], v) for k, v in birth.items()), f"{path}: the tower is not at birth"
                assert [bool(g.get("critic")) for g in optimizer_state["param_groups"]] == \
                    [bool(g.get("critic")) for g in self.opt.param_groups], f"{path}: optimizer groups"
            else:
                model_state = migrate_critic_weights(model_state, self.model)
                ema_state = migrate_critic_weights(ema_state, self.ema)
                optimizer_state = migrate_critic_optimizer(optimizer_state, self.model, self.opt)
        if migrate_hlgauss:
            assert isinstance(self.model, SpireNet) and isinstance(self.ema, SpireNet)
            model_state = migrate_hlgauss_weights(model_state, self.model)
            ema_state = migrate_hlgauss_weights(ema_state, self.ema)
            optimizer_state = migrate_hlgauss_optimizer(optimizer_state, self.model, self.opt)
        self.model.load_state_dict(model_state)
        self.ema.load_state_dict(ema_state)
        self.opt.load_state_dict(optimizer_state)
        c = ck["counters"]
        self.version, self.updates, self.samples_seen = int(c["version"]), int(c["updates"]), int(c["samples_seen"])
        self._updates_offset = self.updates - self.version
        if self._shaping_active:
            sc = c.get("shaping")
            self.shaping_start_update = None if sc is None or sc["start_update"] is None else int(sc["start_update"])
        else:
            self.shaping_start_update = None
        self.ema_steps = int(c.get("ema_steps", 1 << 30))  # older checkpoints: warm-up long over
        ks = self.cfg.kickstart
        if "kickstart" in c:  # a kickstarted run: the same teacher, the schedule continues (or is over)
            kc = c["kickstart"]
            if ks.on:
                assert kc["teacher_sha256"] == self.teacher_sha256, \
                    f"{path}: kickstart teacher changed ({kc['teacher_sha256']} -> {self.teacher_sha256})"
                assert int(kc["phase"]) <= ks.phase, f"{path}: kickstart phase {kc['phase']} -> {ks.phase} refused"
                self.kick_u = int(kc["u"])
                self.kick_release = None if kc["release_update"] is None else int(kc["release_update"])
                refade = None if kc.get("refade") is None else (int(kc["refade"][0]), float(kc["refade"][1]))
                self.kick_refade = refade_after_resume(saved_kick, ks, self.kick_u, self.kick_release, refade)
            else:
                assert not teacher_active(saved_kick, int(kc["u"]), kc["release_update"],
                                          None if kc.get("refade") is None else tuple(kc["refade"])), \
                    f"{path}: kickstart switched off at update {kc['u']} (phase {kc['phase']}, release " \
                    f"{kc['release_update']}) before its schedule is over"
        else:
            assert not ks.on, f"{path}: kickstart on, but the checkpoint has no kickstart counters (v1: a fresh student)"
        if self.critic_cfg is None:
            assert "critic" not in c, f"{path}: a critic checkpoint, the learner has no ppo.critic"
        elif migrate_critic:
            assert "critic" not in c, f"{path}: already a critic checkpoint"
            self.critic_first = self.switched_at = None  # the warm-up starts at the first update of this learner
            self._switch_window = []
        else:
            assert "critic" in c, f"{path}: no critic state (--migrate-critic?)"
            cc = c["critic"]
            self.critic_first, self.switched_at = cc["first_update"], cc["switched_at"]
            self._switch_window = [list(map(float, r)) for r in cc["window"]]
        self.ret_moments.load(ck["ret_moments"])
        rng = ck["rng"]
        torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            saved_cuda = list(rng["cuda"])
            n_dev = torch.cuda.device_count()
            if len(saved_cuda) == n_dev:
                torch.cuda.set_rng_state_all(saved_cuda)
            elif saved_cuda:
                for idx in range(n_dev):
                    torch.cuda.set_rng_state(saved_cuda[idx % len(saved_cuda)], idx)
        self.rng.bit_generator.state = rng["numpy"]
        if self.replica is not None:  # the replica is never saved: a copy of the loaded model
            self.replica.copy_from(self.model)
        return dict(ck.get("extra", {}))


def _normalise(d: Any) -> Any:
    return json.loads(json.dumps(d))  # tuples -> lists, for config comparison


class _Accum:
    """Device-side sums of per-micro-batch diagnostics; one host sync at the end.

    :meth:`add_micro` only does in-place tensor arithmetic (it is captured into the fast path's CUDA graphs);
    host-side counts (micro-batches, minibatches, samples) are kept by the caller."""

    def __init__(self, device: torch.device) -> None:
        z = lambda: torch.zeros((), device=device)  # noqa: E731
        self.loss = {k: z() for k in ("pi", "v", "win", "floors", "hp", "ent_bonus", "uniform_ce")}
        self.kl, self.clipfrac, self.ratio_dev, self.n_pol = z(), z(), z(), z()
        self.ratio_dev0, self.n_pol0, self.clip0 = z(), z(), z()
        self.ent_k = torch.zeros(N_KIND_SLOTS, device=device)
        self.ent_nk = torch.zeros(N_KIND_SLOTS, device=device)
        self.ent_base_k = torch.zeros(N_KIND_SLOTS, device=device)
        self.ent_base_nk = torch.zeros(N_KIND_SLOTS, device=device)
        self.n_k = torch.zeros(N_KIND_SLOTS, device=device)
        self.ent_beh_c, self.ent_base_c, self.n_c = z(), z(), z()  # combat policy steps (spire.temperature)
        self.illegal_chosen = z()
        self.gn_sum, self.gn_n, self.skipped_dev = z(), z(), z()  # device-side optimiser steps (fast path)
        self.first_kl, self.lr_scale_sum, self.lr_scale_n = z(), z(), z()
        self.lr_scale_min = torch.ones((), device=device)
        self.critic_gn = z()  # the tower's gradient norms, summed over the minibatches (a critic learner)
        self.micro = 0
        self.minibatches = 0
        self.skipped = 0
        self.samples = 0
        self._gn: list[float] = []

    def tensors(self) -> list[Tensor]:
        return [*self.loss.values(), self.kl, self.clipfrac, self.ratio_dev, self.n_pol, self.ratio_dev0, self.n_pol0,
                self.clip0, self.ent_k, self.ent_nk, self.ent_base_k, self.ent_base_nk,
                self.n_k, self.illegal_chosen, self.gn_sum, self.gn_n,
                self.skipped_dev, self.first_kl, self.lr_scale_sum, self.lr_scale_n, self.lr_scale_min,
                self.ent_beh_c, self.ent_base_c, self.n_c]

    def snapshot(self) -> list[Tensor]:
        return [t.clone() for t in self.tensors()]

    @torch.no_grad()
    def restore(self, snap: list[Tensor]) -> None:
        for t, s in zip(self.tensors(), snap):
            t.copy_(s)

    @torch.no_grad()
    def reset(self) -> None:
        for t in self.tensors():
            t.zero_()
        self.critic_gn.zero_()
        self.lr_scale_min.fill_(1.0)
        self.micro = self.minibatches = self.skipped = self.samples = 0
        self._gn = []

    def add_micro(self, e0: float | Tensor, l_pi: Tensor, l_v: Tensor, l_win: Tensor, l_fl: Tensor, l_hp: Tensor,
                  bonus: Tensor, kl: Tensor, clipped: Tensor, ratio_dev: Tensor, n_pol: Tensor, kinds: Tensor,
                  ent: Tensor, ent_n: Tensor, pmf: Tensor, illegal: Tensor, n: int, uniform_ce: Tensor,
                  base_ent: Tensor, base_ent_n: Tensor) -> None:
        """``e0`` = 1 in the first epoch, else 0 (a float or a device scalar)."""
        for k, v in (("pi", l_pi), ("v", l_v), ("win", l_win), ("floors", l_fl), ("hp", l_hp),
                     ("ent_bonus", bonus), ("uniform_ce", uniform_ce)):
            self.loss[k] += v.detach()
        self.kl += kl
        self.clipfrac += clipped
        self.ratio_dev += ratio_dev
        self.n_pol += n_pol
        self.ratio_dev0 += ratio_dev * e0
        self.n_pol0 += n_pol * e0
        self.clip0 += clipped * e0
        self.ent_k.index_add_(0, kinds, ent)
        self.ent_nk.index_add_(0, kinds, ent_n)
        self.ent_base_k.index_add_(0, kinds, base_ent)
        self.ent_base_nk.index_add_(0, kinds, base_ent_n)
        self.n_k.index_add_(0, kinds, pmf)
        self.illegal_chosen += illegal

    def add_combat(self, ent_beh: Tensor, base_ent: Tensor, mask: Tensor) -> None:
        """Entropy of the (tempered) behaviour and of the base policy on the combat policy steps (``mask`` = 1)."""
        self.ent_beh_c += ent_beh.sum()
        self.ent_base_c += base_ent.sum()
        self.n_c += mask.sum()

    def grad_norm(self, g: float) -> None:
        self._gn.append(g)

    def result(self) -> dict[str, Any]:
        # one device->host transfer for every accumulator (each separate read is a synchronous copy; under WDDM
        # contention each waits for a time slice of this process)
        flat = torch.cat([t.reshape(-1).float() for t in self.tensors()]).cpu().numpy().astype(np.float64)
        vals: dict[str, Any] = {}
        off = 0
        names = [*(f"loss_{k}" for k in self.loss), "kl", "clipfrac", "ratio_dev", "n_pol", "ratio_dev0", "n_pol0",
                 "clip0", "ent_k", "ent_nk", "ent_base_k", "ent_base_nk", "n_k", "illegal", "gn_sum", "gn_n", "skipped_dev",
                 "first_kl", "lr_scale_sum", "lr_scale_n", "lr_scale_min", "ent_beh_c", "ent_base_c", "n_c"]
        for name, t in zip(names, self.tensors()):
            n = t.numel()
            vals[name] = flat[off:off + n] if t.ndim else float(flat[off])
            off += n
        skipped = self.skipped + int(round(vals["skipped_dev"]))
        mbs = max(1, self.minibatches - skipped)
        n_pol = max(1.0, vals["n_pol"])
        n_pol0 = max(1.0, vals["n_pol0"])
        ent_k, ent_nk, n_k = vals["ent_k"], vals["ent_nk"], vals["n_k"]
        ent_base_k, ent_base_nk = vals["ent_base_k"], vals["ent_base_nk"]
        by_kind = {Kind(k).name: {"entropy": float(ent_k[k] / n_k[k]), "entropy_norm": float(ent_nk[k] / n_k[k]),
                                  "base_entropy": float(ent_base_k[k] / n_k[k]),
                                  "base_entropy_norm": float(ent_base_nk[k] / n_k[k]),
                                  "samples": int(n_k[k])} for k in range(N_KIND_SLOTS) if n_k[k] > 0}
        gn = [g for g in self._gn if math.isfinite(g)]
        gn_n = vals["gn_n"]
        gn_mean = (sum(gn) + vals["gn_sum"]) / (len(gn) + gn_n) if gn or gn_n else float("nan")
        return {
            **{f"loss_{k}": vals[f"loss_{k}"] / mbs for k in self.loss},
            "entropy": float(ent_k.sum() / max(1.0, n_k.sum())),
            "entropy_norm": float(ent_nk.sum() / max(1.0, n_k.sum())),
            "entropy_base": float(ent_base_k.sum() / max(1.0, n_k.sum())),
            "entropy_base_norm": float(ent_base_nk.sum() / max(1.0, n_k.sum())),
            "entropy_by_kind": by_kind,
            "entropy_beh_combat": vals["ent_beh_c"] / vals["n_c"] if vals["n_c"] else None,
            "entropy_base_combat": vals["ent_base_c"] / vals["n_c"] if vals["n_c"] else None,
            "combat_policy_samples": int(vals["n_c"]),
            "approx_kl": vals["kl"] / n_pol, "clipfrac": vals["clipfrac"] / n_pol,
            "first_minibatch_kl": vals["first_kl"],
            "kl_lr_scale_mean": (vals["lr_scale_sum"] / vals["lr_scale_n"] if vals["lr_scale_n"] else 1.0),
            "kl_lr_scale_min": vals["lr_scale_min"],
            "ratio_dev": vals["ratio_dev"] / n_pol,
            "ratio_dev_epoch0": vals["ratio_dev0"] / n_pol0, "clipfrac_epoch0": vals["clip0"] / n_pol0,
            "grad_norm": gn_mean,
            "nonfinite_skipped": skipped, "illegal_chosen": int(round(vals["illegal"])),
            "micro_batches": self.micro,
        }
