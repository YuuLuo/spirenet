"""Standalone Async ExIt Search Teacher Worker (py-stage-p13).

Runs 100% fair-play ``MODE_RESAMPLE`` fight-mode lookahead on live scout episodes alongside
PPO training, polling the latest EMA weights from ``<run_dir>/ckpt/latest.pt`` and/or
``<run_dir>/weights/LATEST`` every ``--poll-s`` seconds, and writing verified high-gain
tactical corrections as atomic shards into ``--exit-dir``.

STRICT FAIR-PLAY:
  - Uses ONLY ``MODE_RESAMPLE`` + ``PRACTICE_FLAGS`` (never ``MODE_EXACT``, never ``--allow-debug-options``).
  - Episode seeds and fork resample seeds derive exclusively from a private SplitMix64 counter
    (never from game state or any formal/holdout/acceptance evaluation panel).
  - Acting policy and search observe strictly player-visible public frames.

Usage::

    python -B -m spire.exit_teacher \\
        --run-dir ../runs/S35-silent-19m-exit \\
        --host-config configs/hosts/linux_search.toml \\
        --removal-ratings configs/removal-ratings.json \\
        --exit-dir ../runs/exit-bank/silent \\
        --char SILENT --ascension 10 --worker-id 0 --device cuda
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
import signal
import socket
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actor import EpisodeJob, TorchPolicy, group_ranges, parse_lineup
from .cycle import CycleBan
from .evalpanel import load_policy_net
from .exit_buffer import ExitRecord, _retry_fs, save_exit_shard
from .gamelog import GameLogger, _View
from .gym import (
    FIGHT_ABORTED,
    FIGHT_WON,
    FORK_REFUSED,
    GAME_FLAGS,
    MODE_RESAMPLE,
    MSG_FIGHT_END,
    MSG_FORKED,
    PRACTICE_FLAGS,
    SeedStream,
    encode_fork,
    msg_type,
    parse_fight_end,
    parse_forked,
    parse_hello,
    same_obs,
    splitmix64,
)
from .history import PublicHistory
from .host import HostConfig, HostCrashed, HostError, HostProcess, HostTimeout, real_host_spec
from .learner import FileWeightSource
from .native_batch import NativeBatchHost, NativeTransportError
from .proto import Frame, encode_step, parse_frame
from .removal import load_rule
from .schema import (
    ActionCol,
    CardCol,
    CardSelectSub,
    CreatureCol,
    CreatureKind,
    EventFlag,
    G,
    G_ERROR_CODE,
    ItemZone,
    Kind,
    MsgType,
    PotionCol,
    Result,
    SeatCol,
    Table,
    Verb,
    reset_flags_for,
)
from .tokenize import Sample, batch_to

GOLDEN = 0x9E3779B97F4A7C15
M64 = (1 << 64) - 1
MAX_DECISIONS = 30000
GROUP = 16
CHUNK_FORK = 16
CHILD_TIMEOUT = 120.0
_SEED_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_scout_seed(stream: SeedStream) -> str:
    """Generate a 12-char alphanumeric STS2 run seed purely from a private SplitMix64 counter."""
    x = stream.next()
    chars: list[str] = []
    for _ in range(12):
        chars.append(_SEED_ALPHABET[x % 36])
        x //= 36
        if x < 36:
            x = stream.next()
    return "".join(chars)


# =========================================================================== teacher config & filters


# Non-combat kinds eligible for fair-play Macro ExIt (Kind.CRYSTAL & Kind.REWARD strictly excluded!)
MACRO_ELIGIBLE_KINDS = frozenset(
    {
        Kind.CARD_REWARD,
        Kind.EVENT,
        Kind.SHOP,
        Kind.REST,
        Kind.MAP,
        Kind.CARD_SELECT,
        Kind.BUNDLE,
        Kind.TREASURE,
    }
)
MACRO_CARD_SELECT_SUBS = frozenset(
    {
        CardSelectSub.DECK_UPGRADE,
        CardSelectSub.DECK_REMOVAL,
        CardSelectSub.DECK_TRANSFORM,
        CardSelectSub.DECK_ENCHANT,
    }
)


@dataclasses.dataclass(frozen=True)
class ExitTeacherConfig:
    """Per-turn budgeted trigger & high-precision emission filter (pwin-search-probe.md §5)."""

    m1: int = 4                      # Stage 1: 4 cands x 4 worlds = 16 children (1 GROUP=16 batch)
    m2: int = 8                      # Stage 2: 2 cands x 8 worlds = 16 children (M_total=12)
    max_k: int = 4                   # [greedy, top-1, END_TURN, best_potion]
    skip_pwin: float = 0.94          # Skip search when cur_pwin >= 0.94
    danger_pwin: float = 0.72        # Act-2/3 Elites & Act-2 Boss eligibility threshold
    act1_danger_pwin: float = 0.55   # Act-1 Elite & Act-1 Boss eligibility threshold
    boss3_risk_pwin: float = 0.85    # Boss-3 mid-turn END_TURN safety check threshold
    boss3_safety_pwin: float = 0.92  # Boss-3 potion & CARD_SELECT check threshold
    greedy_uncertain: float = 0.90   # Boss-3 uncertain policy trigger
    other_uncertain: float = 0.85    # Non-Boss-3 uncertain policy trigger
    max_decs_per_turn: int = 3       # Per-turn pacing cap (fixes front-loaded cap bug!)
    boss3_max_decs: int = 60         # Per-fight backstop for Act-3 bosses
    other_max_decs: int = 25         # Per-fight backstop for non-Boss-3 danger fights
    s1_gate: float = 0.015           # Stage-1 -> Stage-2 progression margin
    # High-precision emission thresholds
    boss3_min_gain: float = 0.025
    boss3_min_t: float = 1.8
    boss3_min_s2: float = 0.015
    other_min_gain: float = 0.040
    other_min_t: float = 2.0
    other_min_s2: float = 0.020
    require_dv_positive: bool = True
    # p13.1 optional expansions (defaults preserve exact p13 behavior)
    max_k_card_select: int = 0       # 0 = use max_k; > 0 (e.g. 6) widens CARD_SELECT candidate pool
    emergency_hallway: bool = False  # True: also trigger on hallway/event fights with door_pwin < emergency_pwin
    emergency_pwin: float = 0.45
    combat_max_child_steps: int = 1200  # Anti-stalemate cap per forked combat child (0 = MAX_DECISIONS)
    # p13.2 optional fair-play Macro ExIt (default False = bit-identical to Combat-only workers)
    enable_macro: bool = False
    macro_m1: int = 6
    macro_m2: int = 6
    macro_max_k: int = 5
    macro_horizon_floors: int = 4
    macro_max_child_steps: int = 240
    macro_skip_pwin: float = 0.95
    macro_card_uncertain: float = 0.88
    macro_card_pwin: float = 0.78
    macro_shop_min_gold: int = 75
    macro_shop_uncertain: float = 0.90
    macro_shop_pwin: float = 0.82
    macro_shop_max_per_visit: int = 2
    macro_rest_uncertain: float = 0.96
    macro_enable_map: bool = False
    macro_map_uncertain: float = 0.70
    macro_event_uncertain: float = 0.85
    macro_select_uncertain: float = 0.85
    macro_s1_gate: float = 0.020
    macro_min_gain: float = 0.035
    macro_min_t: float = 2.00
    macro_min_s2: float = 0.020
    macro_max_per_episode: int = 24
    macro_blindspot_prob: float = 0.0        # Opt-in probe rate on confident CARD_REWARD with <2% blind-spot cards
    macro_blindspot_max_prob: float = 0.02   # Policy group_prob threshold defining an ignored blind-spot card
    # p13.3 optional Macro Teacher v3 (accelerate-90-1123.md §3.2; default False = bit-identical!)
    enable_combat_search: bool = True
    macro_v3: bool = False
    macro_v3_max_k: int = 4
    macro_v3_world_stages: tuple[int, ...] = (16, 32, 64)
    macro_v3_min_worlds: int = 48
    macro_v3_min_gain: float = 0.020
    macro_v3_min_t: float = 2.50
    macro_v3_t_lo: float = 1.0
    macro_v3_t_hi: float = 3.0
    macro_v3_max_child_steps: int = 2500
    macro_v3_max_per_episode: int = 40


def should_trigger_fight(
    act_idx: int,
    room_name: str,
    door_pwin: float,
    cfg: ExitTeacherConfig,
) -> tuple[bool, str]:
    """High-yield fight filter: Act-3 Bosses (BA/BB), Act-1/2 Bosses, Act-1/2/3 Elites, optional emergency hallways."""
    is_boss = room_name == "Boss"
    if act_idx == 2 and is_boss:
        return True, "boss3"
    if act_idx in (0, 1) and is_boss:
        thr = cfg.act1_danger_pwin if act_idx == 0 else cfg.danger_pwin
        return (door_pwin < thr), f"boss_act{act_idx + 1}"
    if room_name == "Elite":
        thr = cfg.act1_danger_pwin if act_idx == 0 else cfg.danger_pwin
        return (door_pwin < thr), f"danger_act{act_idx + 1}_elite"
    if cfg.emergency_hallway and door_pwin < cfg.emergency_pwin:
        return True, f"emerg_act{act_idx + 1}_{room_name.lower()}"
    return False, ""


def should_trigger_decision(
    *,
    is_boss3: bool,
    cur_pwin: float,
    greedy_gp: float,
    is_turn_start: bool,
    is_card_select: bool,
    has_potion: bool,
    has_end_turn: bool,
    greedy_is_end: bool,
    turn_decs_searched: int,
    fight_decs_searched: int,
    checked_card_select_turn: bool,
    checked_potion_turn: bool,
    checked_end_turn_turn: bool,
    cfg: ExitTeacherConfig,
) -> tuple[bool, str]:
    """Per-turn budgeted trigger (<= max_decs_per_turn per round, skipping cur_pwin >= skip_pwin)."""
    fight_cap = cfg.boss3_max_decs if is_boss3 else cfg.other_max_decs
    if fight_decs_searched >= fight_cap:
        return False, ""
    if turn_decs_searched >= cfg.max_decs_per_turn:
        return False, ""
    if cur_pwin >= cfg.skip_pwin:
        return False, ""

    if is_boss3:
        if is_turn_start:
            return True, "turn_start"
        if is_card_select and (not checked_card_select_turn) and cur_pwin < cfg.boss3_safety_pwin:
            return True, "card_select"
        if greedy_gp < cfg.greedy_uncertain:
            return True, "uncertain"
        if has_potion and (not checked_potion_turn) and cur_pwin < cfg.boss3_safety_pwin:
            return True, "potion_check"
        if has_end_turn and (not greedy_is_end) and (not checked_end_turn_turn) and cur_pwin < cfg.boss3_risk_pwin:
            return True, "end_turn_check"
        return False, ""

    # Non-Boss-3 danger fight (Act-1/2 Boss or Act-1/2/3 Elite)
    if is_turn_start and cur_pwin < cfg.danger_pwin:
        return True, "turn_start"
    if is_card_select and (not checked_card_select_turn) and cur_pwin < cfg.danger_pwin:
        return True, "card_select"
    if greedy_gp < cfg.other_uncertain and cur_pwin < cfg.boss3_risk_pwin:
        return True, "uncertain"
    if has_potion and (not checked_potion_turn) and cur_pwin < cfg.danger_pwin:
        return True, "potion_check"
    if has_end_turn and (not greedy_is_end) and (not checked_end_turn_turn) and cur_pwin < cfg.danger_pwin:
        return True, "end_turn_check"
    return False, ""


def passes_emission_filter(
    *,
    is_boss3: bool,
    gain_all: float,
    t_all: float,
    gain_s2: float,
    dv_after: float,
    cfg: ExitTeacherConfig,
) -> bool:
    """High-precision emission gate for ExIt teacher supervision labels."""
    if cfg.require_dv_positive and not (dv_after > 0.0):
        return False
    if is_boss3:
        return (gain_all > cfg.boss3_min_gain) and (t_all >= cfg.boss3_min_t) and (gain_s2 > cfg.boss3_min_s2)
    return (gain_all > cfg.other_min_gain) and (t_all >= cfg.other_min_t) and (gain_s2 > cfg.other_min_s2)


def should_trigger_macro(
    *,
    kind: int,
    kind_sub: int,
    room_name: str,
    event_flags: int,
    gold: int,
    n_unique: int,
    greedy_gp: float,
    cur_pwin: float,
    macro_decs_episode: int,
    shop_decs_visit: int,
    cfg: ExitTeacherConfig,
    has_blindspot_card: bool = False,
    probe_u: float = 1.0,
    act_idx: int = 0,
) -> tuple[bool, str]:
    """Fair-play Macro ExIt trigger (excludes CRYSTAL/REWARD; MAP triggers ONLY on greedy_gp < 0.70)."""
    if not (cfg.enable_macro or cfg.macro_v3) or n_unique <= 1:
        return False, ""
    if kind not in MACRO_ELIGIBLE_KINDS:
        return False, ""
    max_ep = cfg.macro_v3_max_per_episode if cfg.macro_v3 else cfg.macro_max_per_episode
    if macro_decs_episode >= max_ep:
        return False, ""

    if cfg.macro_v3:
        # Macro v3 (§3.2): EVERY Ancient / Relic EVENT, and EVERY Act-3 REST, CARD_REWARD, and funded SHOP
        # (never skipped by greedy_gp or cur_pwin >= macro_skip_pwin).
        if kind == Kind.EVENT and bool(event_flags & int(EventFlag.ANCIENT)):
            return True, "ancient_relic"
        if act_idx == 2:
            if kind == Kind.REST:
                return True, "act3_rest"
            if kind == Kind.CARD_REWARD:
                return True, "act3_card_reward"
            if kind == Kind.SHOP:
                if shop_decs_visit >= cfg.macro_shop_max_per_visit:
                    return False, ""
                if gold >= cfg.macro_shop_min_gold:
                    return True, "act3_shop"
                return False, ""

    if cur_pwin >= cfg.macro_skip_pwin:
        return False, ""

    if kind == Kind.EVENT:
        if bool(event_flags & int(EventFlag.ANCIENT)):
            return True, "ancient_relic"
        if greedy_gp < cfg.macro_event_uncertain:
            return True, "event_uncertain"
        return False, ""

    if kind == Kind.REST:
        if greedy_gp < cfg.macro_rest_uncertain:
            return True, "rest_choice"
        return False, ""

    if kind == Kind.MAP:
        if cfg.macro_enable_map and greedy_gp < cfg.macro_map_uncertain:
            return True, "map_fork"
        return False, ""

    if kind == Kind.CARD_REWARD:
        if room_name in ("Boss", "Elite"):
            return True, f"card_reward_{room_name.lower()}"
        if greedy_gp < cfg.macro_card_uncertain or cur_pwin < cfg.macro_card_pwin:
            return True, "card_reward_uncertain"
        if cfg.macro_blindspot_prob > 0.0 and has_blindspot_card and probe_u < cfg.macro_blindspot_prob:
            return True, "card_reward_blindspot"
        return False, ""

    if kind == Kind.SHOP:
        if shop_decs_visit >= cfg.macro_shop_max_per_visit:
            return False, ""
        if gold >= cfg.macro_shop_min_gold and (greedy_gp < cfg.macro_shop_uncertain or cur_pwin < cfg.macro_shop_pwin):
            return True, "shop_spend"
        return False, ""

    if kind == Kind.CARD_SELECT:
        if kind_sub in MACRO_CARD_SELECT_SUBS and greedy_gp < cfg.macro_select_uncertain:
            return True, "deck_select"
        return False, ""

    if kind in (Kind.BUNDLE, Kind.TREASURE):
        if greedy_gp < cfg.macro_card_uncertain:
            return True, Kind(int(kind)).name.lower()
        return False, ""

    return False, ""


def select_macro_candidates(reps: list[dict[str, Any]], kind: int, max_k: int) -> list[dict[str, Any]]:
    """Select up to ``max_k`` non-combat candidates, reserving SKIP/LEAVE/CANCEL alongside top-policy actions."""
    chosen: dict[int, dict[str, Any]] = {}

    def add(rep: dict[str, Any], reason: str) -> None:
        r = rep["row"]
        if r not in chosen:
            chosen[r] = dict(rep, reason=reason)

    greedy_rep = next(x for x in reps if x["is_greedy"])
    add(greedy_rep, "greedy")
    non_greedy = sorted(
        [x for x in reps if (not x["is_greedy"]) and x["verb"] != Verb.DISCARD_POTION],
        key=lambda x: (-x["group_prob"], -x["logit"], x["row"]),
    )

    reserve_verbs: tuple[int, ...] = ()
    if kind in (Kind.CARD_REWARD, Kind.BUNDLE, Kind.TREASURE):
        reserve_verbs = (Verb.SKIP,)
    elif kind == Kind.SHOP:
        reserve_verbs = (Verb.PROCEED, Verb.SKIP)
    elif kind == Kind.CARD_SELECT:
        reserve_verbs = (Verb.CANCEL,)

    for rv in reserve_verbs:
        for rep in non_greedy:
            if rep["verb"] == rv and len(chosen) < max_k:
                add(rep, "reserve")
                break

    for rep in non_greedy:
        if len(chosen) >= max_k:
            break
        if rep["row"] not in chosen:
            add(rep, "top_policy")

    return [chosen[greedy_rep["row"]]] + [v for k, v in chosen.items() if k != greedy_rep["row"]]


def passes_macro_emission_filter(
    *,
    is_ancient: bool,
    gain_all: float,
    t_all: float,
    gain_s2: float,
    dv_after: float,
    cfg: ExitTeacherConfig,
) -> bool:
    """High-precision Macro ExIt emission gate (unconditional dv_after > 0 required, including Ancient relics!)."""
    _ = is_ancient
    if not (gain_all > cfg.macro_min_gain and t_all >= cfg.macro_min_t and gain_s2 > cfg.macro_min_s2):
        return False
    if dv_after <= 0.0:
        return False
    return True


def is_macro_horizon_reached(
    fr: Frame,
    *,
    start_act: int,
    start_floor: int,
    steps: int,
    cfg: ExitTeacherConfig,
) -> bool:
    """True once a GAME_FLAGS macro rollout reaches a clean non-combat valuation horizon."""
    if not fr.is_obs:
        return True
    if int(fr.gvals[G.IN_COMBAT]) != 0:
        return False
    act = int(fr.gvals[G.ACT_INDEX])
    floor = int(fr.gvals[G.TOTAL_FLOOR])
    if act > start_act:
        return True
    if floor >= start_floor + cfg.macro_horizon_floors:
        return True
    if steps >= cfg.macro_max_child_steps:
        return True
    return False


def is_macro_v3_horizon_reached(
    fr: Frame,
    *,
    start_act: int,
    ascension: int,
    steps: int = 0,
    cfg: ExitTeacherConfig | None = None,
) -> bool:
    """Macro v3 deep valuation horizon (accelerate-90-1123.md §3.2):
      - Act 3 (start_act >= 2), A10+: roll out through Boss A to Boss B's exact door
        (act == 2, bosses_won >= 3, in_combat == 1) where raw pwin has Brier 0.03.
      - Act 3 (start_act >= 2), A<10: roll out to terminal / final-boss win.
      - Acts 1-2 (start_act in (0, 1)): roll out to the first non-combat decision of the next Act.
    Step-cap safety backstop is enforced via ``_ChildAbortedError`` in the rollout loop so stuck
    mid-combat frames are never scored as a false horizon.
    """
    _ = (steps, cfg)
    if not fr.is_obs:
        return True
    act = int(fr.gvals[G.ACT_INDEX])
    in_cmb = int(fr.gvals[G.IN_COMBAT]) != 0
    bw = int(fr.gvals[G.BOSSES_WON])
    if start_act >= 2:
        if ascension >= 10:
            return (act == 2) and (bw >= 3) and in_cmb
        return False
    return (act > start_act) and (not in_cmb)


def _is_macro_v3_boss_b_door(
    fr: Frame,
    *,
    start_act: int,
    ascension: int,
    cfg: ExitTeacherConfig,
) -> bool:
    return (
        cfg.macro_v3
        and start_act >= 2
        and ascension >= 10
        and fr.is_obs
        and int(fr.gvals[G.ACT_INDEX]) == 2
        and int(fr.gvals[G.BOSSES_WON]) >= 3
        and int(fr.gvals[G.IN_COMBAT]) != 0
    )


def _is_macro_horizon_hit(
    fr: Frame,
    *,
    start_act: int,
    start_floor: int,
    ascension: int,
    steps: int,
    cfg: ExitTeacherConfig,
) -> bool:
    if cfg.macro_v3:
        return is_macro_v3_horizon_reached(
            fr, start_act=start_act, ascension=ascension, steps=steps, cfg=cfg
        )
    return is_macro_horizon_reached(
        fr, start_act=start_act, start_floor=start_floor, steps=steps, cfg=cfg
    )


def _macro_step_cap(cfg: ExitTeacherConfig) -> int:
    if cfg.macro_v3:
        return cfg.macro_v3_max_child_steps
    return (
        cfg.combat_max_child_steps + cfg.macro_max_child_steps
        if cfg.combat_max_child_steps > 0
        else 0
    )


def _score_macro_horizon_frame(
    fr: Frame,
    vh: dict[str, float],
    *,
    start_act: int,
    ascension: int,
    cfg: ExitTeacherConfig,
    calibrator: OnlinePwinCalibrator | None,
) -> float:
    """Score a Macro horizon frame: raw pwin at Act-3 A10 Boss-B door (Brier 0.03), else calibrated if enabled."""
    if _is_macro_v3_boss_b_door(fr, start_act=start_act, ascension=ascension, cfg=cfg):
        return float(vh["pwin"])
    if calibrator is not None:
        return calibrator.predict(int(fr.gvals[G.ACT_INDEX]), vh["pwin_logit"])
    return float(vh["pwin"])


def should_expand_macro_v3_stage(
    *,
    mean_d: float,
    t_stat: float,
    n_worlds: int,
    max_worlds: int,
    cfg: ExitTeacherConfig,
) -> bool:
    """Adaptive paired CRN continuation rule for Macro Teacher v3 (16 -> 32 -> 64).

    Stops immediately on non-positive mean or noise (|t| <= t_lo).
    Continues to the next stage when:
      - signal is in the undecided band (t_lo < |t| < t_hi), OR
      - signal already clears (min_gain, min_t) but needs >= macro_v3_min_worlds confirmation.
    """
    if n_worlds >= max_worlds:
        return False
    if math.isnan(mean_d) or math.isnan(t_stat) or not (mean_d > 0.0):
        return False
    abs_t = abs(t_stat)
    if abs_t <= cfg.macro_v3_t_lo:
        return False
    if cfg.macro_v3_t_lo < abs_t < cfg.macro_v3_t_hi:
        return True
    if (
        mean_d >= cfg.macro_v3_min_gain
        and t_stat >= cfg.macro_v3_min_t
        and n_worlds < cfg.macro_v3_min_worlds
    ):
        return True
    return False


def passes_macro_v3_emission_filter(
    *,
    n_worlds: int,
    gain_all: float,
    t_all: float,
    dv_after: float,
    cfg: ExitTeacherConfig,
) -> bool:
    """High-precision Macro v3 emission gate: >= min_worlds (48), +2pp, t >= 2.5, unconditional dv_after > 0."""
    if n_worlds < cfg.macro_v3_min_worlds:
        return False
    if not (gain_all >= cfg.macro_v3_min_gain and t_all >= cfg.macro_v3_min_t):
        return False
    if not (dv_after > 0.0):
        return False
    return True


class OnlinePwinCalibrator:
    """Rolling per-Act Platt calibrator ``sigmoid(a * raw_logit + b)`` trained on live greedy EMA scout runs.

    Eliminates the -15 to -20 pp Act-1/2 downward bias of ``pwin_logit`` (caused by training rollouts running
    at temp=1.0 + exploration_by_kind) with zero changes to model weights or checkpoints.
    """

    def __init__(self, *, window: int = 300, min_samples: int = 25, l2: float = 1.0) -> None:
        assert window >= min_samples >= 5 and l2 > 0.0
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.l2 = float(l2)
        self._data: dict[int, collections.deque[tuple[float, float]]] = {}
        self._params: dict[int, tuple[float, float]] = {}  # act -> (a, b), prior (1.0, 0.0)

    def observe(self, act_idx: int, raw_logit: float, won: bool) -> None:
        if not math.isfinite(raw_logit):
            return
        q = self._data.setdefault(int(act_idx), collections.deque(maxlen=self.window))
        q.append((float(raw_logit), 1.0 if won else 0.0))
        if len(q) >= self.min_samples:
            self._refit(int(act_idx))

    def _refit(self, act_idx: int) -> None:
        arr = np.asarray(self._data[act_idx], dtype=np.float64)
        x, y = arr[:, 0], arr[:, 1]
        # Regularize toward identity (a=1, b=0) so small samples stay well-behaved
        a, b = self._params.get(act_idx, (1.0, 0.0))
        for _ in range(15):
            z = np.clip(a * x + b, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            w = np.maximum(p * (1.0 - p), 1e-6)
            ga = float(np.sum((p - y) * x) + self.l2 * (a - 1.0))
            gb = float(np.sum(p - y) + self.l2 * b)
            haa = float(np.sum(w * x * x) + self.l2)
            hab = float(np.sum(w * x))
            hbb = float(np.sum(w) + self.l2)
            det = haa * hbb - hab * hab
            if det <= 1e-12:
                break
            da = (hbb * ga - hab * gb) / det
            db = (-hab * ga + haa * gb) / det
            a = float(np.clip(a - da, 0.25, 4.0))
            b = float(np.clip(b - db, -4.0, 4.0))
            if abs(da) + abs(db) < 1e-6:
                break
        self._params[act_idx] = (a, b)

    def predict(self, act_idx: int, raw_logit: float) -> float:
        a, b = self._params.get(int(act_idx), (1.0, 0.0))
        z = max(-30.0, min(30.0, a * float(raw_logit) + b))
        return 1.0 / (1.0 + math.exp(-z))

    def params(self, act_idx: int) -> tuple[float, float]:
        return self._params.get(int(act_idx), (1.0, 0.0))


# =========================================================================== policy, context & weight polling


@dataclasses.dataclass
class TeacherCtx:
    tok: Any
    net: Any
    policy: TorchPolicy
    gl: GameLogger
    spec: Any
    flags: int
    removal: Any
    vocab_hash: int
    catalog_hash: int
    version: int
    room_names: tuple[str, ...]
    device: torch.device


class LiveWeightPoller:
    """Polls ``<run_dir>/ckpt/latest.pt`` (EMA) and optional ``<run_dir>/weights/LATEST`` for newer weights."""

    def __init__(
        self,
        *,
        ckpt_path: Path | None,
        weights_dir: Path | None,
        poll_s: float = 60.0,
        have_version: int = 0,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.wsrc = FileWeightSource(weights_dir) if weights_dir is not None else None
        self.poll_s = max(1.0, float(poll_s))
        self.version = int(have_version)
        self._last_poll = time.monotonic()
        self._last_ckpt_mtime = -1.0

    def maybe_refresh(self, net: torch.nn.Module, *, force: bool = False) -> bool:
        """If ``poll_s`` elapsed and newer weights exist, copy them into ``net`` in-place (graph-safe)."""
        now = time.monotonic()
        if not force and (now - self._last_poll) < self.poll_s:
            return False
        self._last_poll = now

        best_v = self.version
        best_sd: dict[str, torch.Tensor] | None = None

        # 1. Check weights/LATEST (fast acting weights published every learner update when acting_weights="ema")
        if self.wsrc is not None:
            got = self.wsrc.poll(best_v)
            if got is not None and got[0] > best_v:
                best_v, best_sd = got

        # 2. Check ckpt/latest.pt (EMA weights saved periodically)
        if self.ckpt_path is not None and self.ckpt_path.is_file():
            try:
                mt = self.ckpt_path.stat().st_mtime
                if mt != self._last_ckpt_mtime:
                    ck = _retry_fs(lambda: torch.load(self.ckpt_path, map_location="cpu", weights_only=True))
                    self._last_ckpt_mtime = mt
                    cv = int(ck.get("counters", {}).get("version", -1))
                    if cv > best_v and "ema" in ck:
                        best_v = cv
                        best_sd = ck["ema"]
            except (FileNotFoundError, OSError, RuntimeError, ValueError):
                pass

        if best_sd is None or best_v <= self.version:
            return False
        with torch.no_grad():
            cur = net.state_dict()
            for k, v in cur.items():
                if k in best_sd:
                    v.copy_(best_sd[k], non_blocking=False)
        self.version = best_v
        return True


class Player:
    """Per-game trackers matching spire.actor._Env."""

    def __init__(self, ctx: TeacherCtx | None, history: PublicHistory | None = None, cycle: CycleBan | None = None) -> None:
        if ctx is not None:
            c = ctx.tok.config
            history = PublicHistory(ctx.tok.width) if c.public_history else None
            cycle = CycleBan(ctx.tok.vocab, commit_select=c.commit_select, removal=ctx.removal, mode=c.rules_mode)
        self.history, self.cycle = history, cycle

    def fork(self) -> "Player":
        return Player(None, None if self.history is None else self.history.fork(), self.cycle.fork())

    def record(self, f: Frame, sample: Sample, a: int) -> None:
        assert a not in self.cycle.active_bans, f"step {f.step_id}: action {a} is cycle-banned"
        if self.history is not None:
            self.history.record(f, sample, a)
        self.cycle.record(f, a)


def _sigmoid_f64(x: float) -> float:
    z = max(-40.0, min(40.0, float(x)))
    return 1.0 / (1.0 + math.exp(-z))


def decide_with_heads(
    ctx: TeacherCtx, items: list[tuple[Frame, Player]]
) -> tuple[list[int], list[np.ndarray], list[Sample], list[dict[str, float]]]:
    """Run ``decide`` and return ``(value, pwin_logit, pwin)`` from the same CUDA-graphed forward pass."""
    tok, policy = ctx.tok, ctx.policy
    frames = [f for f, _ in items]
    enc = tok.encode_batch(frames, sort=True)
    order = list(range(len(frames))) if enc.order is None else enc.order.tolist()
    samples = enc.samples()
    for j, i in enumerate(order):
        f, p = items[i]
        if p.history is not None:
            enc.history[j] = p.history.observe(f)
        p.cycle.observe(f, samples[j])
    n_act = np.fromiter((frames[i].n_actions for i in order), np.int64, len(order))
    launched = [
        (lo, hi, policy.launch_encoded(enc, lo, hi, n_act[lo:hi]))
        for lo, hi in group_ranges(tok.bucket_of(enc.n_tokens), GROUP, GROUP)
    ]
    acts: list[int] = [-1] * len(frames)
    scores: list[np.ndarray] = [np.zeros(0)] * len(frames)
    samp: list[Sample] = [None] * len(frames)  # type: ignore[list-item]
    heads: list[dict[str, float]] = [{} for _ in range(len(frames))]
    has_pwin = True
    for lo, hi, h in launched:
        out = policy.finish(h)
        pw = out.pwin_logit
        if pw is None:
            has_pwin = False
        for r in range(lo, hi):
            i = order[r]
            k = r - lo
            acts[i] = int(out.actions[k])
            scores[i] = out.logits[k, : n_act[r]].copy()
            samp[i] = samples[r]
            if pw is not None:
                pl = float(pw[k])
                heads[i] = {"value": float(out.value[k]), "pwin_logit": pl, "pwin": _sigmoid_f64(pl)}
    if not has_pwin:
        heads = eval_heads_batch(ctx, items)
    return acts, scores, samp, heads


def decide(ctx: TeacherCtx, items: list[tuple[Frame, Player]]) -> tuple[list[int], list[np.ndarray], list[Sample]]:
    acts, scores, samp, _ = decide_with_heads(ctx, items)
    return acts, scores, samp


def eval_heads_batch(ctx: TeacherCtx, items: list[tuple[Frame, Player]]) -> list[dict[str, float]]:
    if not items:
        return []
    frames = [f for f, _ in items]
    enc = ctx.tok.encode_batch(frames, sort=True)
    order = list(range(len(frames))) if enc.order is None else enc.order.tolist()
    samples = enc.samples()
    for j, i in enumerate(order):
        f, p = items[i]
        pf = p.fork()
        if pf.history is not None:
            enc.history[j] = pf.history.observe(f)
        pf.cycle.observe(f, samples[j])
    out_list: list[dict[str, float]] = [{} for _ in range(len(frames))]
    if getattr(ctx.policy, "include_pwin", False):
        n_act = np.fromiter((frames[i].n_actions for i in order), np.int64, len(order))
        launched = [
            (lo, hi, ctx.policy.launch_encoded(enc, lo, hi, n_act[lo:hi]))
            for lo, hi in group_ranges(ctx.tok.bucket_of(enc.n_tokens), GROUP, GROUP)
        ]
        for lo, hi, h in launched:
            out = ctx.policy.finish(h)
            assert out.pwin_logit is not None
            for r in range(lo, hi):
                i = order[r]
                k = r - lo
                pl = float(out.pwin_logit[k])
                out_list[i] = {"value": float(out.value[k]), "pwin": _sigmoid_f64(pl), "pwin_logit": pl}
        return out_list
    use_amp = ctx.device.type == "cuda"
    for lo, hi in group_ranges(ctx.tok.bucket_of(enc.n_tokens), 8, 8):
        batch = ctx.tok.collate_encoded(enc, lo, hi)
        with torch.inference_mode():
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = ctx.net(batch_to(batch, ctx.device))
            else:
                o = ctx.net(batch_to(batch, ctx.device))
        vals = o["value"].float().cpu().numpy()
        logits_pwin = o["pwin_logit"].float().cpu().numpy()
        pwins = torch.sigmoid(o["pwin_logit"].float()).cpu().numpy()
        for r in range(lo, hi):
            i = order[r]
            k = r - lo
            out_list[i] = {"value": float(vals[k]), "pwin": float(pwins[k]), "pwin_logit": float(logits_pwin[k])}
    return out_list


def eval_heads_one(ctx: TeacherCtx, f: Frame, player: Player) -> dict[str, float]:
    return eval_heads_batch(ctx, [(f, player)])[0]


# =========================================================================== candidate selection & scoring


def decode_actions(ctx: TeacherCtx, f: Frame) -> list[dict[str, Any]]:
    v = _View(f)
    acts = v.t(Table.ACTIONS)
    return [ctx.gl._action(v, a) for a in acts]


def action_canonical_sig(f: Frame, dec: dict[str, Any], row: int) -> str:
    acts = f.table(Table.ACTIONS)
    st = int(acts[row, ActionCol.SRC_TABLE])
    sr = int(acts[row, ActionCol.SRC_ROW])
    extra: Any = None
    if st == Table.CARDS:
        c = f.table(Table.CARDS)[sr]
        extra = [
            int(c[CardCol.KEYWORD_BITS]),
            int(c[CardCol.ENCHANT_AMOUNT]),
            int(c[CardCol.AFFLICTION_AMOUNT]),
            int(c[CardCol.REPLAY_COUNT]),
            int(c[CardCol.SEL_COUNT]),
            int(c[CardCol.AUX]),
        ]
    return json.dumps({"d": dec, "x": extra}, sort_keys=True, separators=(",", ":"))


def legal_action_stats(
    ctx: TeacherCtx, f: Frame, scores: np.ndarray, bans: set[int], greedy_row: int
) -> tuple[list[dict[str, Any]], float, bool, bool, np.ndarray]:
    n_act = f.n_actions
    assert len(scores) == n_act
    masked = np.array(scores, dtype=np.float64)
    for b in bans:
        if 0 <= b < n_act:
            masked[b] = -np.inf
    finite = np.isfinite(masked)
    assert np.any(finite), f"step {f.step_id}: no finite legal action"
    mx = float(np.max(masked[finite]))
    exps = np.where(finite, np.exp(masked - mx), 0.0)
    probs = exps / float(np.sum(exps))
    assert np.isfinite(masked[greedy_row])

    decs = decode_actions(ctx, f)
    acts_tbl = f.table(Table.ACTIONS)
    groups: dict[str, list[int]] = {}
    for r in range(n_act):
        if r in bans or not np.isfinite(masked[r]):
            continue
        sig = action_canonical_sig(f, decs[r], r)
        groups.setdefault(sig, []).append(r)

    reps: list[dict[str, Any]] = []
    greedy_gp = float(probs[greedy_row])
    has_end_turn = False
    has_potion = False
    for _, rows in groups.items():
        is_greedy = greedy_row in rows
        rep = greedy_row if is_greedy else max(rows, key=lambda r: (float(probs[r]), -r))
        verb = int(acts_tbl[rep, ActionCol.VERB])
        if verb == Verb.END_TURN:
            has_end_turn = True
        elif verb == Verb.USE_POTION:
            has_potion = True
        gp = float(sum(probs[r] for r in rows))
        if is_greedy:
            greedy_gp = gp
        reps.append(
            {
                "row": rep,
                "verb": verb,
                "verb_name": decs[rep].get("verb", Verb(verb).name),
                "prob": float(probs[rep]),
                "group_prob": gp,
                "logit": float(masked[rep]),
                "is_greedy": is_greedy,
            }
        )
    return reps, greedy_gp, has_end_turn, has_potion, finite.astype(np.bool_)


def select_candidates(reps: list[dict[str, Any]], max_k: int) -> list[dict[str, Any]]:
    """Select up to ``max_k`` canonical candidates: greedy, top-1 alt, END_TURN, best_potion, then fill."""
    chosen: dict[int, dict[str, Any]] = {}

    def add(rep: dict[str, Any], reason: str) -> None:
        r = rep["row"]
        if r not in chosen:
            chosen[r] = dict(rep, reason=reason)

    greedy_rep = next(x for x in reps if x["is_greedy"])
    add(greedy_rep, "greedy")
    non_greedy = sorted(
        [x for x in reps if (not x["is_greedy"]) and x["verb"] != Verb.DISCARD_POTION],
        key=lambda x: (-x["group_prob"], -x["prob"], x["row"]),
    )
    n_top = 2 if max_k >= 5 else 1
    for rep in non_greedy[:n_top]:
        if len(chosen) < max_k:
            add(rep, "top_policy")
    for rep in non_greedy:
        if rep["verb"] == Verb.END_TURN and rep["row"] not in chosen and len(chosen) < max_k:
            add(rep, "end_turn")
            break
    pot_reps = sorted(
        [x for x in non_greedy if x["verb"] == Verb.USE_POTION and x["row"] not in chosen],
        key=lambda x: (-x["logit"], x["row"]),
    )
    if pot_reps and len(chosen) < max_k:
        add(pot_reps[0], "best_potion")
    if len(pot_reps) > 1 and len(chosen) < max_k:
        add(pot_reps[1], "potion_2")
    for rep in non_greedy:
        if len(chosen) >= max_k:
            break
        if rep["row"] not in chosen:
            add(rep, "fill_policy")
    return [chosen[greedy_rep["row"]]] + [v for k, v in chosen.items() if k != greedy_rep["row"]]


def paired_stats(alt_scores: list[float], base_scores: list[float]) -> tuple[float, float, float]:
    diffs = np.asarray(alt_scores, dtype=np.float64) - np.asarray(base_scores, dtype=np.float64)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    if n <= 1:
        return mean_d, 0.0, (math.inf if mean_d > 0 else (-math.inf if mean_d < 0 else 0.0))
    sd = float(np.std(diffs, ddof=1))
    se = sd / math.sqrt(n)
    if se == 0.0:
        t_stat = math.inf if mean_d > 0 else (-math.inf if mean_d < 0 else 0.0)
    else:
        t_stat = mean_d / se
    return mean_d, se, t_stat


# =========================================================================== host & fair RESAMPLE forks


def make_search_host(ctx: TeacherCtx, log_dir: Path, name: str) -> HostProcess:
    spec = ctx.spec
    assert "--search" in spec.command, f"exit_teacher requires --search host: {spec.command}"
    assert "--allow-debug-options" not in spec.command, "FORBIDDEN: --allow-debug-options in exit_teacher"
    log_dir.mkdir(parents=True, exist_ok=True)
    return HostProcess(
        HostConfig(
            command=spec.command,
            env=spec.process_env(),
            start_timeout=180.0,
            reset_timeout=60.0,
            step_timeout=60.0,
            log_path=str(log_dir / f"{name}.log"),
            name=name,
            max_restarts=0,
            expected_hashes=(ctx.vocab_hash, ctx.catalog_hash),
        )
    )


def _recv_exact(s: socket.socket, n: int) -> bytearray:
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise EOFError("child closed socket inside message")
        buf += chunk
    return buf


def _recv_msg(s: socket.socket) -> bytes:
    (n,) = struct.unpack("<I", _recv_exact(s, 4))
    return bytes(_recv_exact(s, n))


def _alive_pid(pid: int) -> bool:
    p = f"/proc/{pid}/stat"
    if not os.path.exists(p):
        return False
    try:
        s = open(p, "rb").read().decode()
    except FileNotFoundError:
        return False
    return s[s.rindex(")") + 2 :].split()[0] != "Z"


def fork_fight_kids(
    host: HostProcess,
    f: Frame,
    seeds: list[int],
    tag: int,
    listener: socket.socket,
    sock_path: str,
) -> list[tuple[socket.socket, int, bytes]]:
    """Strictly fair-play ``MODE_RESAMPLE`` fight-mode fork (``PRACTICE_FLAGS``)."""
    host._send_guarded(encode_fork(f.step_id, tuple(seeds), MODE_RESAMPLE, tag, sock_path, PRACTICE_FLAGS))
    reply = host._recv_guarded(60.0)
    mt = msg_type(reply)
    assert mt == MSG_FORKED, f"FORK answered with msg type {mt}: {parse_frame(reply)!r}"
    fk = parse_forked(reply)
    assert (fk.tag, fk.n, fk.mode, fk.step_id) == (tag, len(seeds), MODE_RESAMPLE, f.step_id), fk
    got: dict[int, tuple[socket.socket, int, bytes]] = {}
    deadline = time.monotonic() + CHILD_TIMEOUT
    prev_to = listener.gettimeout()
    listener.settimeout(0.5)
    try:
        while len(got) < len(seeds):
            try:
                s, _ = listener.accept()
            except TimeoutError:
                missing_pids = [fk.pids[i] for i in range(len(seeds)) if i not in got]
                if any(not _alive_pid(p) for p in missing_pids):
                    raise _ChildAbortedError(
                        f"combat fork child exited before connecting (missing={len(missing_pids)})"
                    )
                if time.monotonic() >= deadline:
                    raise _ChildAbortedError("combat fork accept timeout")
                continue
            try:
                s.settimeout(CHILD_TIMEOUT)
                h = parse_hello(_recv_msg(s))
                assert (h.tag, h.mode, h.depth, h.flags, h.step_id) == (
                    tag,
                    MODE_RESAMPLE,
                    1,
                    PRACTICE_FLAGS,
                    f.step_id,
                ), h
                assert h.parent_pid == h.root_pid == host.pid and h.pid == fk.pids[h.index], h
                assert h.seed == seeds[h.index], h
                first = _recv_msg(s)
                assert same_obs(first, f.payload), f"child {h.pid}: first frame differs from parent"
                assert h.index not in got
                got[h.index] = (s, h.pid, first)
            except BaseException:
                s.close()
                raise
    except (EOFError, TimeoutError, OSError) as exc:
        for sock, _, _ in got.values():
            sock.close()
        raise _ChildAbortedError(f"combat fork handshake error: {exc}") from exc
    except BaseException:
        for sock, _, _ in got.values():
            sock.close()
        raise
    finally:
        listener.settimeout(prev_to)
    return [got[i] for i in range(len(seeds))]


def _is_final_boss_win_obs(fr: Frame, ascension: int) -> bool:
    if not fr.is_obs or int(fr.gvals[G.IN_COMBAT]) != 0:
        return False
    bosses_needed = 4 if ascension >= 10 else 3
    return int(fr.gvals[G.BOSSES_WON]) >= bosses_needed or int(fr.gvals[G.TOTAL_FLOOR]) >= 49


class _ChildAbortedError(RuntimeError):
    """Raised when a forked search child aborts/crashes (e.g., unsupported Rust edge).

    Never scored as a loss (0.0) so ExIt labels have zero bias against unsupported edges.
    """


def eval_candidates_on_seeds(
    ctx: TeacherCtx,
    host: HostProcess | NativeBatchHost,
    f: Frame,
    player: Player,
    cand_rows: list[int],
    seeds: list[int],
    ascension: int,
    listener: socket.socket | None,
    sock_path: str,
    tag_box: list[int],
    pids: list[int],
    calibrator: OnlinePwinCalibrator | None = None,
    max_child_steps: int = 1200,
) -> dict[int, list[dict[str, float]]]:
    assert f.is_obs and int(f.gvals[G.IN_COMBAT]) == 1
    if isinstance(host, NativeBatchHost):
        return _eval_candidates_native_batch(
            ctx, host, f, player, cand_rows, seeds, ascension, calibrator=calibrator, max_child_steps=max_child_steps
        )
    assert listener is not None
    out: dict[int, list[dict[str, float]]] = {r: [] for r in cand_rows}
    jobs = [(r, w) for r in cand_rows for w in range(len(seeds))]
    for lo in range(0, len(jobs), CHUNK_FORK):
        chunk = jobs[lo : lo + CHUNK_FORK]
        tag_box[0] += 1
        got = fork_fight_kids(host, f, [seeds[w] for _, w in chunk], tag_box[0], listener, sock_path)
        batch: list[dict[str, Any]] = []
        for (row, w), (s, pid, first) in zip(chunk, got):
            pids.append(pid)
            batch.append(
                {
                    "sock": s,
                    "pid": pid,
                    "frame": parse_frame(first),
                    "player": player.fork(),
                    "cand_row": row,
                    "w": w,
                    "steps": 0,
                }
            )
        need_pwin: list[tuple[dict[str, Any], Frame, Player]] = []
        done_recs: list[dict[str, Any]] = []
        try:
            while batch:
                acts, _, samples = decide(ctx, [(e["frame"], e["player"]) for e in batch])
                for i, e in enumerate(batch):
                    fr = e["frame"]
                    a = e["cand_row"] if e["steps"] == 0 else acts[i]
                    e["player"].record(fr, samples[i], a)
                    try:
                        e["sock"].sendall(encode_step(fr.step_id, a))
                    except (EOFError, TimeoutError, OSError) as exc:
                        raise _ChildAbortedError(f"combat child {e['pid']} send error: {exc}") from exc
                    e["steps"] += 1
                    assert e["steps"] < MAX_DECISIONS
                nxt_batch: list[dict[str, Any]] = []
                for e in batch:
                    try:
                        raw = _recv_msg(e["sock"])
                    except (EOFError, TimeoutError, OSError) as exc:
                        raise _ChildAbortedError(f"combat child {e['pid']} transport error: {exc}") from exc
                    mt = msg_type(raw)
                    if mt == MSG_FIGHT_END:
                        fe = parse_fight_end(raw)
                        try:
                            final_fr = parse_frame(_recv_msg(e["sock"]))
                        except (EOFError, TimeoutError, OSError) as exc:
                            raise _ChildAbortedError(f"combat child {e['pid']} post-fight error: {exc}") from exc
                        e["sock"].close()
                        rec: dict[str, Any] = {"cand_row": e["cand_row"], "w": e["w"]}
                        if fe.outcome == FIGHT_ABORTED:
                            raise _ChildAbortedError(f"combat child {e['pid']} returned FIGHT_ABORTED")
                        if fe.outcome != FIGHT_WON:
                            rec["score"] = 0.0
                            rec["v_after"] = 0.0
                            done_recs.append(rec)
                        elif final_fr.is_terminal:
                            if final_fr.result not in (Result.WIN, Result.LOSS):
                                raise _ChildAbortedError(
                                    f"combat child {e['pid']} non-WL terminal: {final_fr.result}"
                                )
                            won = bool(final_fr.result == Result.WIN)
                            rec["score"] = 1.0 if won else 0.0
                            rec["v_after"] = 1.0 if won else 0.0
                            done_recs.append(rec)
                        elif _is_final_boss_win_obs(final_fr, ascension):
                            rec["score"] = 1.0
                            rec["v_after"] = 1.0
                            done_recs.append(rec)
                        else:
                            if not final_fr.is_obs:
                                raise _ChildAbortedError(
                                    f"combat child {e['pid']} non-obs post-fight frame: msg={final_fr.msg_type}"
                                )
                            need_pwin.append((rec, final_fr, e["player"]))
                    else:
                        fr = parse_frame(raw)
                        if fr.is_terminal and fr.result not in (Result.WIN, Result.LOSS):
                            raise _ChildAbortedError(f"combat child {e['pid']} aborted terminal: {fr.result}")
                        if not fr.is_obs:
                            raise _ChildAbortedError(
                                f"combat child {e['pid']} non-obs frame: msg={fr.msg_type} result={fr.result}"
                            )
                        if max_child_steps > 0 and e["steps"] >= max_child_steps:
                            raise _ChildAbortedError(
                                f"combat child {e['pid']} exceeded max_child_steps={max_child_steps}"
                            )
                        e["frame"] = fr
                        nxt_batch.append(e)
                batch = nxt_batch
        finally:
            for e in batch:
                e["sock"].close()
        if need_pwin:
            heads = eval_heads_batch(ctx, [(fr, pl) for _, fr, pl in need_pwin])
            for (rec, fr, _), vh in zip(need_pwin, heads):
                if calibrator is not None:
                    rec["score"] = calibrator.predict(int(fr.gvals[G.ACT_INDEX]), vh["pwin_logit"])
                else:
                    rec["score"] = vh["pwin"]
                rec["v_after"] = vh["value"]
                done_recs.append(rec)
        for rec in done_recs:
            out[rec["cand_row"]].append(rec)
    for r in cand_rows:
        out[r].sort(key=lambda x: x["w"])
        assert len(out[r]) == len(seeds)
    return out


def fork_macro_kids(
    host: HostProcess,
    f: Frame,
    seeds: list[int],
    tag: int,
    listener: socket.socket,
    sock_path: str,
) -> list[tuple[socket.socket, int, bytes]] | None:
    """Strictly fair-play ``MODE_RESAMPLE | GAME_FLAGS`` non-combat fork. Returns ``None`` on ERROR 4."""
    host._send_guarded(encode_fork(f.step_id, tuple(seeds), MODE_RESAMPLE, tag, sock_path, GAME_FLAGS))
    reply = host._recv_guarded(60.0)
    mt = msg_type(reply)
    if mt == int(MsgType.ERROR):
        err_fr = parse_frame(reply)
        assert int(err_fr.gvals[G_ERROR_CODE]) == FORK_REFUSED, f"unexpected macro FORK error: {err_fr!r}"
        resent = host._recv_guarded(60.0)
        assert same_obs(resent, f.payload), "resent OBS after macro FORK_REFUSED differed from pending frame"
        return None
    assert mt == MSG_FORKED, f"macro FORK answered with msg type {mt}: {parse_frame(reply)!r}"
    fk = parse_forked(reply)
    assert (fk.tag, fk.n, fk.mode, fk.step_id) == (tag, len(seeds), MODE_RESAMPLE, f.step_id), fk
    got: dict[int, tuple[socket.socket, int, bytes]] = {}
    deadline = time.monotonic() + CHILD_TIMEOUT
    prev_to = listener.gettimeout()
    listener.settimeout(0.5)
    try:
        while len(got) < len(seeds):
            try:
                s, _ = listener.accept()
            except TimeoutError:
                missing_pids = [fk.pids[i] for i in range(len(seeds)) if i not in got]
                if any(not _alive_pid(p) for p in missing_pids):
                    raise _ChildAbortedError(
                        f"macro fork child exited before connecting (missing={len(missing_pids)})"
                    )
                if time.monotonic() >= deadline:
                    raise _ChildAbortedError("macro fork accept timeout")
                continue
            try:
                s.settimeout(CHILD_TIMEOUT)
                h = parse_hello(_recv_msg(s))
                assert (h.tag, h.mode, h.depth, h.flags, h.step_id) == (
                    tag,
                    MODE_RESAMPLE,
                    1,
                    GAME_FLAGS,
                    f.step_id,
                ), h
                assert h.parent_pid == h.root_pid == host.pid and h.pid == fk.pids[h.index], h
                assert h.seed == seeds[h.index], h
                first = _recv_msg(s)
                assert same_obs(first, f.payload), f"macro child {h.pid}: first frame differs from parent"
                assert h.index not in got
                got[h.index] = (s, h.pid, first)
            except BaseException:
                s.close()
                raise
    except (EOFError, TimeoutError, OSError) as exc:
        for sock, _, _ in got.values():
            sock.close()
        raise _ChildAbortedError(f"macro fork handshake error: {exc}") from exc
    except BaseException:
        for sock, _, _ in got.values():
            sock.close()
        raise
    finally:
        listener.settimeout(prev_to)
    return [got[i] for i in range(len(seeds))]


def _eval_candidates_native_batch(
    ctx: TeacherCtx,
    host: NativeBatchHost,
    f: Frame,
    player: Player,
    cand_rows: list[int],
    seeds: list[int],
    ascension: int,
    calibrator: OnlinePwinCalibrator | None = None,
    max_child_steps: int = 1200,
) -> dict[int, list[dict[str, float]]]:
    """Option B in-process parallel ``NativeBatch`` combat rollout (zero fork/socket IPC)."""
    start_combats = int(f.gvals[G.COMBATS_WON])
    out: dict[int, list[dict[str, float]]] = {r: [] for r in cand_rows}
    jobs = [(r, w) for r in cand_rows for w in range(len(seeds))]
    for lo in range(0, len(jobs), CHUNK_FORK):
        chunk = jobs[lo : lo + CHUNK_FORK]
        try:
            kids = host.fork_resample(f, [seeds[w] for _, w in chunk], flags=0)
        except NativeTransportError as exc:
            raise _ChildAbortedError(f"native combat fork error: {exc}") from exc
        all_handles = [h for h, _ in kids]
        batch: list[dict[str, Any]] = []
        for (row, w), (h, kf) in zip(chunk, kids):
            batch.append(
                {
                    "handle": h,
                    "frame": kf,
                    "player": player.fork(),
                    "cand_row": row,
                    "w": w,
                    "steps": 0,
                }
            )
        need_pwin: list[tuple[dict[str, Any], Frame, Player]] = []
        done_recs: list[dict[str, Any]] = []
        try:
            while batch:
                acts, _, samples = decide(ctx, [(e["frame"], e["player"]) for e in batch])
                step_reqs: list[tuple[int, int, int]] = []
                for i, e in enumerate(batch):
                    fr = e["frame"]
                    a = e["cand_row"] if e["steps"] == 0 else acts[i]
                    e["player"].record(fr, samples[i], a)
                    step_reqs.append((e["handle"], a, fr.step_id))
                    e["steps"] += 1
                    assert e["steps"] < MAX_DECISIONS
                try:
                    sres = host.step_batch(step_reqs)
                except NativeTransportError as exc:
                    raise _ChildAbortedError(f"native combat step_batch error: {exc}") from exc
                nxt_batch: list[dict[str, Any]] = []
                for e, sr in zip(batch, sres):
                    if sr.status < 0 or not sr.frames:
                        raise _ChildAbortedError(f"native combat child {e['handle']} failed: {sr.status} {sr.error}")
                    rec: dict[str, Any] = {"cand_row": e["cand_row"], "w": e["w"]}
                    if msg_type(sr.frames[0]) == MSG_FIGHT_END:
                        fe = parse_fight_end(sr.frames[0])
                        final_fr = parse_frame(sr.frames[-1])
                        if fe.outcome == FIGHT_ABORTED:
                            raise _ChildAbortedError(f"native combat child {e['handle']} FIGHT_ABORTED")
                        if fe.outcome != FIGHT_WON:
                            rec["score"] = 0.0
                            rec["v_after"] = 0.0
                            done_recs.append(rec)
                        elif final_fr.is_terminal:
                            if final_fr.result not in (Result.WIN, Result.LOSS):
                                raise _ChildAbortedError(f"native combat child non-WL terminal: {final_fr.result}")
                            won = bool(final_fr.result == Result.WIN)
                            rec["score"] = 1.0 if won else 0.0
                            rec["v_after"] = 1.0 if won else 0.0
                            done_recs.append(rec)
                        elif _is_final_boss_win_obs(final_fr, ascension):
                            rec["score"] = 1.0
                            rec["v_after"] = 1.0
                            done_recs.append(rec)
                        else:
                            if not final_fr.is_obs:
                                raise _ChildAbortedError(f"native combat child non-obs post-fight: {final_fr.msg_type}")
                            need_pwin.append((rec, final_fr, e["player"]))
                        continue
                    fr = parse_frame(sr.frames[-1])
                    if fr.is_terminal:
                        if fr.result not in (Result.WIN, Result.LOSS):
                            raise _ChildAbortedError(f"native combat child non-WL terminal: {fr.result}")
                        won = bool(fr.result == Result.WIN)
                        rec["score"] = 1.0 if won else 0.0
                        rec["v_after"] = 1.0 if won else 0.0
                        done_recs.append(rec)
                    elif fr.is_obs and (int(fr.gvals[G.IN_COMBAT]) == 0 or int(fr.gvals[G.COMBATS_WON]) > start_combats):
                        if _is_final_boss_win_obs(fr, ascension):
                            rec["score"] = 1.0
                            rec["v_after"] = 1.0
                            done_recs.append(rec)
                        else:
                            need_pwin.append((rec, fr, e["player"]))
                    elif not fr.is_obs:
                        raise _ChildAbortedError(f"native combat child non-obs frame: msg={fr.msg_type} res={fr.result}")
                    else:
                        if max_child_steps > 0 and e["steps"] >= max_child_steps:
                            raise _ChildAbortedError(
                                f"native combat child {e['handle']} exceeded max_child_steps={max_child_steps}"
                            )
                        e["frame"] = fr
                        nxt_batch.append(e)
                batch = nxt_batch
        finally:
            host.release_handles(all_handles)
        if need_pwin:
            heads = eval_heads_batch(ctx, [(fr, pl) for _, fr, pl in need_pwin])
            for (rec, fr, _), vh in zip(need_pwin, heads):
                if calibrator is not None:
                    rec["score"] = calibrator.predict(int(fr.gvals[G.ACT_INDEX]), vh["pwin_logit"])
                else:
                    rec["score"] = vh["pwin"]
                rec["v_after"] = vh["value"]
                done_recs.append(rec)
        for rec in done_recs:
            out[rec["cand_row"]].append(rec)
    for r in cand_rows:
        out[r].sort(key=lambda x: x["w"])
        assert len(out[r]) == len(seeds)
    return out


def _eval_macro_candidates_native_batch(
    ctx: TeacherCtx,
    host: NativeBatchHost,
    f: Frame,
    player: Player,
    cand_rows: list[int],
    seeds: list[int],
    ascension: int,
    cfg: ExitTeacherConfig,
    calibrator: OnlinePwinCalibrator | None = None,
) -> dict[int, list[dict[str, float]]] | None:
    """Option B in-process parallel ``NativeBatch`` 6-vector Macro rollout (``FLAG_GAME=1``)."""
    start_act = int(f.gvals[G.ACT_INDEX])
    start_floor = int(f.gvals[G.TOTAL_FLOOR])
    macro_step_cap = _macro_step_cap(cfg)
    out: dict[int, list[dict[str, float]]] = {r: [] for r in cand_rows}
    jobs = [(r, w) for r in cand_rows for w in range(len(seeds))]
    for lo in range(0, len(jobs), CHUNK_FORK):
        chunk = jobs[lo : lo + CHUNK_FORK]
        try:
            kids = host.fork_resample(f, [seeds[w] for _, w in chunk], flags=1)
        except NativeTransportError as exc:
            raise _ChildAbortedError(f"native macro fork error: {exc}") from exc
        all_handles = [h for h, _ in kids]
        batch: list[dict[str, Any]] = []
        for (row, w), (h, kf) in zip(chunk, kids):
            batch.append(
                {
                    "handle": h,
                    "frame": kf,
                    "player": player.fork(),
                    "cand_row": row,
                    "w": w,
                    "steps": 0,
                }
            )
        need_pwin: list[tuple[dict[str, Any], Frame, Player]] = []
        done_recs: list[dict[str, Any]] = []
        try:
            while batch:
                acts, _, samples = decide(ctx, [(e["frame"], e["player"]) for e in batch])
                step_reqs: list[tuple[int, int, int]] = []
                for i, e in enumerate(batch):
                    fr = e["frame"]
                    a = e["cand_row"] if e["steps"] == 0 else acts[i]
                    e["player"].record(fr, samples[i], a)
                    step_reqs.append((e["handle"], a, fr.step_id))
                    e["steps"] += 1
                    assert e["steps"] < MAX_DECISIONS
                try:
                    sres = host.step_batch(step_reqs)
                except NativeTransportError as exc:
                    raise _ChildAbortedError(f"native macro step_batch error: {exc}") from exc
                nxt_batch: list[dict[str, Any]] = []
                for e, sr in zip(batch, sres):
                    if sr.status < 0 or not sr.frames:
                        raise _ChildAbortedError(f"native macro child {e['handle']} failed: {sr.status} {sr.error}")
                    fr = parse_frame(sr.frames[-1])
                    rec: dict[str, Any] = {"cand_row": e["cand_row"], "w": e["w"]}
                    if fr.is_terminal:
                        if fr.result not in (Result.WIN, Result.LOSS):
                            raise _ChildAbortedError(f"native macro child non-WL terminal: {fr.result}")
                        won = bool(fr.result == Result.WIN)
                        rec["score"] = 1.0 if won else 0.0
                        rec["v_after"] = 1.0 if won else 0.0
                        done_recs.append(rec)
                    elif _is_final_boss_win_obs(fr, ascension):
                        rec["score"] = 1.0
                        rec["v_after"] = 1.0
                        done_recs.append(rec)
                    elif not fr.is_obs:
                        raise _ChildAbortedError(f"native macro child non-obs frame: msg={fr.msg_type} res={fr.result}")
                    elif _is_macro_horizon_hit(
                        fr, start_act=start_act, start_floor=start_floor, ascension=ascension, steps=e["steps"], cfg=cfg
                    ):
                        need_pwin.append((rec, fr, e["player"]))
                    else:
                        if macro_step_cap > 0 and e["steps"] >= macro_step_cap:
                            raise _ChildAbortedError(f"native macro child {e['handle']} exceeded max_steps={macro_step_cap}")
                        e["frame"] = fr
                        nxt_batch.append(e)
                batch = nxt_batch
        finally:
            host.release_handles(all_handles)
        if need_pwin:
            heads = eval_heads_batch(ctx, [(fr, pl) for _, fr, pl in need_pwin])
            for (rec, fr, _), vh in zip(need_pwin, heads):
                rec["score"] = _score_macro_horizon_frame(
                    fr, vh, start_act=start_act, ascension=ascension, cfg=cfg, calibrator=calibrator
                )
                rec["v_after"] = vh["value"]
                done_recs.append(rec)
        for rec in done_recs:
            out[rec["cand_row"]].append(rec)
    for r in cand_rows:
        out[r].sort(key=lambda x: x["w"])
        assert len(out[r]) == len(seeds)
    return out


def eval_macro_candidates_on_seeds(
    ctx: TeacherCtx,
    host: HostProcess | NativeBatchHost,
    f: Frame,
    player: Player,
    cand_rows: list[int],
    seeds: list[int],
    ascension: int,
    cfg: ExitTeacherConfig,
    listener: socket.socket | None,
    sock_path: str,
    tag_box: list[int],
    pids: list[int],
    calibrator: OnlinePwinCalibrator | None = None,
) -> dict[int, list[dict[str, float]]] | None:
    """Truncated-horizon ``GAME_FLAGS`` macro rollout; returns ``None`` if host refuses macro fork (ERROR 4)."""
    assert f.is_obs and int(f.gvals[G.IN_COMBAT]) == 0
    if isinstance(host, NativeBatchHost):
        return _eval_macro_candidates_native_batch(
            ctx, host, f, player, cand_rows, seeds, ascension, cfg, calibrator=calibrator
        )
    assert listener is not None
    start_act = int(f.gvals[G.ACT_INDEX])
    start_floor = int(f.gvals[G.TOTAL_FLOOR])
    out: dict[int, list[dict[str, float]]] = {r: [] for r in cand_rows}
    jobs = [(r, w) for r in cand_rows for w in range(len(seeds))]
    for lo in range(0, len(jobs), CHUNK_FORK):
        chunk = jobs[lo : lo + CHUNK_FORK]
        tag_box[0] += 1
        got = fork_macro_kids(host, f, [seeds[w] for _, w in chunk], tag_box[0], listener, sock_path)
        if got is None:
            return None
        batch: list[dict[str, Any]] = []
        for (row, w), (s, pid, first) in zip(chunk, got):
            pids.append(pid)
            batch.append(
                {
                    "sock": s,
                    "pid": pid,
                    "frame": parse_frame(first),
                    "player": player.fork(),
                    "cand_row": row,
                    "w": w,
                    "steps": 0,
                }
            )
        need_pwin: list[tuple[dict[str, Any], Frame, Player]] = []
        done_recs: list[dict[str, Any]] = []
        try:
            while batch:
                acts, _, samples = decide(ctx, [(e["frame"], e["player"]) for e in batch])
                for i, e in enumerate(batch):
                    fr = e["frame"]
                    a = e["cand_row"] if e["steps"] == 0 else acts[i]
                    e["player"].record(fr, samples[i], a)
                    try:
                        e["sock"].sendall(encode_step(fr.step_id, a))
                    except (EOFError, TimeoutError, OSError) as exc:
                        raise _ChildAbortedError(f"macro child {e['pid']} send error: {exc}") from exc
                    e["steps"] += 1
                    assert e["steps"] < MAX_DECISIONS
                nxt_batch: list[dict[str, Any]] = []
                for e in batch:
                    try:
                        raw = _recv_msg(e["sock"])
                    except (EOFError, TimeoutError, OSError) as exc:
                        raise _ChildAbortedError(f"macro child {e['pid']} transport error: {exc}") from exc
                    fr = parse_frame(raw)
                    rec: dict[str, Any] = {"cand_row": e["cand_row"], "w": e["w"]}
                    if fr.is_terminal:
                        e["sock"].close()
                        if fr.result not in (Result.WIN, Result.LOSS):
                            raise _ChildAbortedError(f"macro child {e['pid']} non-WL terminal: {fr.result}")
                        won = bool(fr.result == Result.WIN)
                        rec["score"] = 1.0 if won else 0.0
                        rec["v_after"] = 1.0 if won else 0.0
                        done_recs.append(rec)
                    elif _is_final_boss_win_obs(fr, ascension):
                        e["sock"].close()
                        rec["score"] = 1.0
                        rec["v_after"] = 1.0
                        done_recs.append(rec)
                    elif not fr.is_obs:
                        raise _ChildAbortedError(
                            f"macro child {e['pid']} non-obs frame: msg={fr.msg_type} result={fr.result}"
                        )
                    elif _is_macro_horizon_hit(
                        fr, start_act=start_act, start_floor=start_floor, ascension=ascension, steps=e["steps"], cfg=cfg
                    ):
                        e["sock"].close()
                        need_pwin.append((rec, fr, e["player"]))
                    else:
                        macro_step_cap = _macro_step_cap(cfg)
                        if macro_step_cap > 0 and e["steps"] >= macro_step_cap:
                            raise _ChildAbortedError(
                                f"macro child {e['pid']} exceeded max_steps={macro_step_cap}"
                            )
                        e["frame"] = fr
                        nxt_batch.append(e)
                batch = nxt_batch
        finally:
            for e in batch:
                e["sock"].close()
        if need_pwin:
            heads = eval_heads_batch(ctx, [(fr, pl) for _, fr, pl in need_pwin])
            for (rec, fr, _), vh in zip(need_pwin, heads):
                rec["score"] = _score_macro_horizon_frame(
                    fr, vh, start_act=start_act, ascension=ascension, cfg=cfg, calibrator=calibrator
                )
                rec["v_after"] = vh["value"]
                done_recs.append(rec)
        for rec in done_recs:
            out[rec["cand_row"]].append(rec)
    for r in cand_rows:
        out[r].sort(key=lambda x: x["w"])
        assert len(out[r]) == len(seeds)
    return out


# =========================================================================== scout episode loop


def run_scout_episode(
    ctx: TeacherCtx,
    host: HostProcess | NativeBatchHost,
    char: str,
    ascension: int,
    run_seed: str,
    fork_stream: SeedStream,
    cfg: ExitTeacherConfig,
    listener: socket.socket | None,
    sock_path: str,
    tag_box: list[int],
    pids: list[int],
    calibrator: OnlinePwinCalibrator | None = None,
    observe_calibrator: bool = True,
) -> tuple[list[ExitRecord], dict[str, Any]]:
    t0 = time.monotonic()
    job = EpisodeJob(parse_lineup(char, ctx.tok.vocab), char, ascension, run_seed, run_seed)
    f = host.reset_with(job.reset_cmd(ctx.flags, MAX_DECISIONS))
    player = Player(ctx)

    emitted: list[ExitRecord] = []
    door_logits: list[tuple[int, float]] = []
    cur_fight_key: tuple[int, int] | None = None
    cur_fight_triggered = False
    cur_fight_is_boss3 = False
    cur_fight_stage = ""
    fight_decs_searched = 0
    entered_b3a = False

    seen_rounds: set[int] = set()
    seen_combat_rounds: set[int] = set()
    turn_decs: dict[int, int] = {}
    checked_sel_round: set[int] = set()
    checked_pot_round: set[int] = set()
    checked_end_round: set[int] = set()

    total_decs = 0
    searched_decs = 0
    s2_decs = 0
    macro_supported = bool(cfg.enable_macro or cfg.macro_v3)
    macro_decs_ep = 0
    macro_s2_ep = 0
    macro_v3_s3_ep = 0
    macro_v3_worlds_ep = 0
    macro_emitted_ep = 0
    macro_bs_probed_ep = 0
    macro_bs_emitted_ep = 0
    child_aborts_ep = 0
    cur_shop_key: tuple[int, int] | None = None
    shop_decs_visit = 0

    while f.is_obs:
        gv = f.gvals
        act = int(gv[G.ACT_INDEX])
        floor = int(gv[G.TOTAL_FLOOR])
        in_combat = int(gv[G.IN_COMBAT]) == 1
        combats_won = int(gv[G.COMBATS_WON])
        bosses_won = int(gv[G.BOSSES_WON])
        rid = int(gv[G.ROOM_TYPE])
        rname = ctx.room_names[rid] if 0 <= rid < len(ctx.room_names) else f"Room({rid})"

        if not in_combat:
            cur_fight_key = None
            cur_fight_triggered = False
        else:
            fkey = (floor, combats_won)
            if fkey != cur_fight_key:
                cur_fight_key = fkey
                fight_decs_searched = 0
                seen_rounds = set()
                seen_combat_rounds = set()
                turn_decs = {}
                checked_sel_round = set()
                checked_pot_round = set()
                checked_end_round = set()
                is_boss3 = (act == 2) and (rname == "Boss")
                if is_boss3:
                    entered_b3a = True
                cur_fight_is_boss3 = is_boss3
                if is_boss3:
                    cur_fight_stage = "A3-BB" if (ascension >= 10 and (bosses_won >= 3 or floor >= 49)) else "A3-BA"
                else:
                    cur_fight_stage = f"A{act + 1}-{rname}"
                need_door_eval = cfg.enable_combat_search or (calibrator is not None and observe_calibrator)
                if need_door_eval:
                    vh_door = eval_heads_one(ctx, f, player)
                    door_logits.append((act, vh_door["pwin_logit"]))
                    door_p = calibrator.predict(act, vh_door["pwin_logit"]) if calibrator is not None else vh_door["pwin"]
                    cur_fight_triggered, _ = (
                        should_trigger_fight(act, rname, door_p, cfg) if cfg.enable_combat_search else (False, "")
                    )
                else:
                    cur_fight_triggered = False

        if in_combat and cur_fight_triggered and f.n_actions > 1:
            probe_player = player.fork()
            acts, scs, probe_samp, probe_heads = decide_with_heads(ctx, [(f, probe_player)])
            greedy_row = acts[0]
            bans = set(probe_player.cycle.active_bans)
            reps, greedy_gp, has_end_turn, has_potion, legal_mask = legal_action_stats(
                ctx, f, scs[0], bans, greedy_row
            )
            chosen_row = greedy_row

            r_num = int(gv[G.COMBAT_ROUND])
            is_turn_start = (r_num not in seen_rounds) or (f.kind == Kind.COMBAT and r_num not in seen_combat_rounds)
            seen_rounds.add(r_num)
            if f.kind == Kind.COMBAT:
                seen_combat_rounds.add(r_num)

            if len(reps) > 1:
                vh_cur = probe_heads[0]
                cur_pwin = calibrator.predict(act, vh_cur["pwin_logit"]) if calibrator is not None else vh_cur["pwin"]
                greedy_is_end = any(x["is_greedy"] and x["verb"] == Verb.END_TURN for x in reps)
                is_sel = f.kind == Kind.CARD_SELECT

                do_search, _ = should_trigger_decision(
                    is_boss3=cur_fight_is_boss3,
                    cur_pwin=cur_pwin,
                    greedy_gp=greedy_gp,
                    is_turn_start=is_turn_start,
                    is_card_select=is_sel,
                    has_potion=has_potion,
                    has_end_turn=has_end_turn,
                    greedy_is_end=greedy_is_end,
                    turn_decs_searched=turn_decs.get(r_num, 0),
                    fight_decs_searched=fight_decs_searched,
                    checked_card_select_turn=(r_num in checked_sel_round),
                    checked_potion_turn=(r_num in checked_pot_round),
                    checked_end_turn_turn=(r_num in checked_end_round),
                    cfg=cfg,
                )

                if do_search:
                    turn_decs[r_num] = turn_decs.get(r_num, 0) + 1
                    fight_decs_searched += 1
                    searched_decs += 1
                    if is_sel:
                        checked_sel_round.add(r_num)
                    if has_potion:
                        checked_pot_round.add(r_num)
                    if has_end_turn and (not greedy_is_end) and (not is_turn_start):
                        checked_end_round.add(r_num)

                    eff_k = cfg.max_k_card_select if (is_sel and cfg.max_k_card_select > 0) else cfg.max_k
                    cands = select_candidates(reps, eff_k)
                    cand_rows = [c["row"] for c in cands]
                    if len(cand_rows) < 2:
                        cands = []
                    best_cand = None
                    best_s1_diff = -1e9
                    if cands:
                        seeds_s1 = [fork_stream.next() for _ in range(cfg.m1)]
                        try:
                            res_s1 = eval_candidates_on_seeds(
                                ctx, host, f, player, cand_rows, seeds_s1, ascension, listener, sock_path, tag_box, pids,
                                calibrator=calibrator, max_child_steps=cfg.combat_max_child_steps,
                            )
                        except _ChildAbortedError:
                            child_aborts_ep += 1
                            res_s1 = None
                        if res_s1 is not None:
                            base_s1 = [x["score"] for x in res_s1[greedy_row]]
                            base_s1_mean = float(np.mean(base_s1))
                            for c in cands[1:]:
                                alt_s1 = [x["score"] for x in res_s1[c["row"]]]
                                d1 = float(np.mean(alt_s1)) - base_s1_mean
                                if d1 > best_s1_diff:
                                    best_s1_diff = d1
                                    best_cand = c

                    if best_cand is not None and best_s1_diff > cfg.s1_gate and cfg.m2 > 0:
                        s2_decs += 1
                        seeds_s2 = [fork_stream.next() for _ in range(cfg.m2)]
                        try:
                            res_s2 = eval_candidates_on_seeds(
                                ctx, host, f, player, [greedy_row, best_cand["row"]], seeds_s2, ascension,
                                listener, sock_path, tag_box, pids, calibrator=calibrator,
                                max_child_steps=cfg.combat_max_child_steps,
                            )
                        except _ChildAbortedError:
                            child_aborts_ep += 1
                            res_s2 = None
                        if res_s2 is None:
                            best_cand = None
                    if best_cand is not None and best_s1_diff > cfg.s1_gate and cfg.m2 > 0:
                        base_s2 = [x["score"] for x in res_s2[greedy_row]]
                        alt_s2 = [x["score"] for x in res_s2[best_cand["row"]]]
                        d2_mean, _, _ = paired_stats(alt_s2, base_s2)

                        base_all = base_s1 + base_s2
                        alt_all = [x["score"] for x in res_s1[best_cand["row"]]] + alt_s2
                        d_all, _, t_all = paired_stats(alt_all, base_all)

                        base_v = [x["v_after"] for x in res_s1[greedy_row]] + [x["v_after"] for x in res_s2[greedy_row]]
                        alt_v = [x["v_after"] for x in res_s1[best_cand["row"]]] + [x["v_after"] for x in res_s2[best_cand["row"]]]
                        dv_all = float(np.mean(alt_v) - np.mean(base_v))

                        if passes_emission_filter(
                            is_boss3=cur_fight_is_boss3,
                            gain_all=d_all,
                            t_all=t_all,
                            gain_s2=d2_mean,
                            dv_after=dv_all,
                            cfg=cfg,
                        ):
                            chosen_row = best_cand["row"]
                            cand_scores = []
                            for c in cands:
                                cr = c["row"]
                                if cr == greedy_row:
                                    cand_scores.append(float(np.mean(base_all)))
                                elif cr == chosen_row:
                                    cand_scores.append(float(np.mean(alt_all)))
                                else:
                                    cand_scores.append(float(np.mean([x["score"] for x in res_s1[cr]])))
                            cand_p_old = tuple(float(c["prob"]) for c in cands)
                            # Clone sample arrays so they detach from the encoded batch buffer
                            s0 = probe_samp[0]
                            s_copy = Sample(
                                s0.types.copy(), s0.raw.copy(), s0.rel.copy(), s0.edges.copy(),
                                s0.cond.copy(), s0.counts, s0.history.copy(),
                            )
                            emitted.append(
                                ExitRecord(
                                    sample=s_copy,
                                    target_action=chosen_row,
                                    legal_mask=legal_mask.copy(),
                                    cand_rows=tuple(cand_rows),
                                    greedy_action=greedy_row,
                                    gain=float(d_all),
                                    gain_s2=float(d2_mean),
                                    t_stat=float(min(t_all, 999.0)),
                                    dv_after=float(dv_all),
                                    p_old=float(best_cand["prob"]),
                                    p_group_old=float(best_cand["group_prob"]),
                                    teacher_version=int(ctx.version),
                                    stage=cur_fight_stage,
                                    verb=str(best_cand["verb_name"]),
                                    char=char,
                                    ascension=ascension,
                                    ts=time.time(),
                                    cand_scores=tuple(cand_scores),
                                    cand_p_old=cand_p_old,
                                )
                            )

            probe_player.record(f, probe_samp[0], chosen_row)
            player = probe_player
            f = host.step(chosen_row)
        elif (not in_combat) and macro_supported and f.n_actions > 1 and f.kind in MACRO_ELIGIBLE_KINDS:
            if f.kind == Kind.SHOP:
                skey = (act, floor)
                if skey != cur_shop_key:
                    cur_shop_key = skey
                    shop_decs_visit = 0
            probe_player = player.fork()
            acts, scs, probe_samp, probe_heads = decide_with_heads(ctx, [(f, probe_player)])
            greedy_row = acts[0]
            bans = set(probe_player.cycle.active_bans)
            reps, greedy_gp, _, _, legal_mask = legal_action_stats(ctx, f, scs[0], bans, greedy_row)
            chosen_row = greedy_row

            if len(reps) > 1:
                seats_tbl = f.table(Table.SEATS)
                gold = int(seats_tbl[0, SeatCol.GOLD]) if len(seats_tbl) > 0 else 0
                vh_cur = probe_heads[0]
                cur_pwin = calibrator.predict(act, vh_cur["pwin_logit"]) if calibrator is not None else vh_cur["pwin"]
                has_bs = (
                    f.kind == Kind.CARD_REWARD
                    and cfg.macro_blindspot_prob > 0.0
                    and any(
                        (not r["is_greedy"])
                        and r["verb"] == Verb.PICK_OFFER_CARD
                        and r["group_prob"] < cfg.macro_blindspot_max_prob
                        for r in reps
                    )
                )
                probe_u = ((fork_stream.next() >> 11) * (1.0 / (1 << 53))) if has_bs else 1.0
                do_macro, macro_reason = should_trigger_macro(
                    kind=f.kind,
                    kind_sub=int(gv[G.KIND_SUB]),
                    room_name=rname,
                    event_flags=int(gv[G.EVENT_FLAGS]),
                    gold=gold,
                    n_unique=len(reps),
                    greedy_gp=greedy_gp,
                    cur_pwin=cur_pwin,
                    macro_decs_episode=macro_decs_ep,
                    shop_decs_visit=shop_decs_visit,
                    cfg=cfg,
                    has_blindspot_card=has_bs,
                    probe_u=probe_u,
                    act_idx=act,
                )
                if do_macro:
                    if macro_reason == "card_reward_blindspot":
                        macro_bs_probed_ep += 1
                    eff_macro_k = cfg.macro_v3_max_k if cfg.macro_v3 else cfg.macro_max_k
                    cands = select_macro_candidates(reps, f.kind, eff_macro_k)
                    cand_rows = [c["row"] for c in cands]
                    if len(cand_rows) < 2:
                        cands = []
                    if cfg.macro_v3:
                        stages = cfg.macro_v3_world_stages
                        assert len(stages) >= 1 and stages[0] > 0
                        s1_n = stages[0]
                        max_w = stages[-1]
                        macro_aborted = False
                        if cands:
                            seeds_s1 = [fork_stream.next() for _ in range(s1_n)]
                            try:
                                res_s1 = eval_macro_candidates_on_seeds(
                                    ctx, host, f, player, cand_rows, seeds_s1, ascension, cfg,
                                    listener, sock_path, tag_box, pids, calibrator=calibrator,
                                )
                            except _ChildAbortedError:
                                child_aborts_ep += 1
                                macro_aborted = True
                                res_s1 = None
                        else:
                            res_s1 = None
                        if (not cands) or macro_aborted:
                            pass
                        elif res_s1 is None:
                            macro_supported = False
                        else:
                            macro_decs_ep += 1
                            macro_v3_worlds_ep += s1_n
                            if f.kind == Kind.SHOP:
                                shop_decs_visit += 1
                            base_s1 = [x["score"] for x in res_s1[greedy_row]]
                            base_s1_mean = float(np.mean(base_s1))
                            best_cand = None
                            best_s1_diff = -1e9
                            for c in cands[1:]:
                                alt_s1 = [x["score"] for x in res_s1[c["row"]]]
                                d1 = float(np.mean(alt_s1)) - base_s1_mean
                                if d1 > best_s1_diff:
                                    best_s1_diff = d1
                                    best_cand = c

                            if best_cand is not None:
                                base_all = list(base_s1)
                                alt_all = [x["score"] for x in res_s1[best_cand["row"]]]
                                base_v = [x["v_after"] for x in res_s1[greedy_row]]
                                alt_v = [x["v_after"] for x in res_s1[best_cand["row"]]]
                                d_all, _, t_all = paired_stats(alt_all, base_all)
                                cur_w = len(base_all)
                                aborted_later = False

                                for st_idx, target_w in enumerate(stages[1:], start=2):
                                    if not should_expand_macro_v3_stage(
                                        mean_d=d_all, t_stat=t_all, n_worlds=cur_w, max_worlds=max_w, cfg=cfg
                                    ):
                                        break
                                    delta_w = target_w - cur_w
                                    if delta_w <= 0:
                                        continue
                                    if st_idx == 2:
                                        macro_s2_ep += 1
                                    else:
                                        macro_v3_s3_ep += 1
                                    seeds_more = [fork_stream.next() for _ in range(delta_w)]
                                    try:
                                        res_more = eval_macro_candidates_on_seeds(
                                            ctx, host, f, player, [greedy_row, best_cand["row"]], seeds_more,
                                            ascension, cfg, listener, sock_path, tag_box, pids, calibrator=calibrator,
                                        )
                                    except _ChildAbortedError:
                                        child_aborts_ep += 1
                                        aborted_later = True
                                        break
                                    if res_more is None:
                                        macro_supported = False
                                        aborted_later = True
                                        break
                                    base_all.extend(x["score"] for x in res_more[greedy_row])
                                    alt_all.extend(x["score"] for x in res_more[best_cand["row"]])
                                    base_v.extend(x["v_after"] for x in res_more[greedy_row])
                                    alt_v.extend(x["v_after"] for x in res_more[best_cand["row"]])
                                    cur_w = len(base_all)
                                    macro_v3_worlds_ep += delta_w
                                    d_all, _, t_all = paired_stats(alt_all, base_all)

                                if not aborted_later:
                                    dv_all = float(np.mean(alt_v) - np.mean(base_v))
                                    if passes_macro_v3_emission_filter(
                                        n_worlds=cur_w, gain_all=d_all, t_all=t_all, dv_after=dv_all, cfg=cfg
                                    ):
                                        d_confirm = (
                                            paired_stats(alt_all[s1_n:], base_all[s1_n:])[0]
                                            if cur_w > s1_n
                                            else d_all
                                        )
                                        chosen_row = best_cand["row"]
                                        macro_emitted_ep += 1
                                        if macro_reason == "card_reward_blindspot":
                                            macro_bs_emitted_ep += 1
                                        cand_scores = []
                                        for c in cands:
                                            cr = c["row"]
                                            if cr == greedy_row:
                                                cand_scores.append(float(np.mean(base_all)))
                                            elif cr == chosen_row:
                                                cand_scores.append(float(np.mean(alt_all)))
                                            else:
                                                cand_scores.append(float(np.mean([x["score"] for x in res_s1[cr]])))
                                        cand_p_old = tuple(float(c["prob"]) for c in cands)
                                        s0 = probe_samp[0]
                                        s_copy = Sample(
                                            s0.types.copy(), s0.raw.copy(), s0.rel.copy(), s0.edges.copy(),
                                            s0.cond.copy(), s0.counts, s0.history.copy(),
                                        )
                                        kname = Kind(int(f.kind)).name
                                        emitted.append(
                                            ExitRecord(
                                                sample=s_copy,
                                                target_action=chosen_row,
                                                legal_mask=legal_mask.copy(),
                                                cand_rows=tuple(cand_rows),
                                                greedy_action=greedy_row,
                                                gain=float(d_all),
                                                gain_s2=float(d_confirm),
                                                t_stat=float(min(t_all, 999.0)),
                                                dv_after=float(dv_all),
                                                p_old=float(best_cand["prob"]),
                                                p_group_old=float(best_cand["group_prob"]),
                                                teacher_version=int(ctx.version),
                                                stage=f"MACRO-A{act + 1}-{kname}",
                                                verb=str(best_cand["verb_name"]),
                                                char=char,
                                                ascension=ascension,
                                                ts=time.time(),
                                                cand_scores=tuple(cand_scores),
                                                cand_p_old=cand_p_old,
                                            )
                                        )
                    else:
                        macro_aborted = False
                        if cands:
                            seeds_s1 = [fork_stream.next() for _ in range(cfg.macro_m1)]
                            try:
                                res_s1 = eval_macro_candidates_on_seeds(
                                    ctx, host, f, player, cand_rows, seeds_s1, ascension, cfg,
                                    listener, sock_path, tag_box, pids, calibrator=calibrator,
                                )
                            except _ChildAbortedError:
                                child_aborts_ep += 1
                                macro_aborted = True
                                res_s1 = None
                        else:
                            res_s1 = None
                        if (not cands) or macro_aborted:
                            pass
                        elif res_s1 is None:
                            macro_supported = False
                        else:
                            macro_decs_ep += 1
                            if f.kind == Kind.SHOP:
                                shop_decs_visit += 1
                            base_s1 = [x["score"] for x in res_s1[greedy_row]]
                            base_s1_mean = float(np.mean(base_s1))
                            best_cand = None
                            best_s1_diff = -1e9
                            for c in cands[1:]:
                                alt_s1 = [x["score"] for x in res_s1[c["row"]]]
                                d1 = float(np.mean(alt_s1)) - base_s1_mean
                                if d1 > best_s1_diff:
                                    best_s1_diff = d1
                                    best_cand = c

                            if best_cand is not None and best_s1_diff > cfg.macro_s1_gate and cfg.macro_m2 > 0:
                                macro_s2_ep += 1
                                seeds_s2 = [fork_stream.next() for _ in range(cfg.macro_m2)]
                                try:
                                    res_s2 = eval_macro_candidates_on_seeds(
                                        ctx, host, f, player, [greedy_row, best_cand["row"]], seeds_s2, ascension, cfg,
                                        listener, sock_path, tag_box, pids, calibrator=calibrator,
                                    )
                                except _ChildAbortedError:
                                    child_aborts_ep += 1
                                    res_s2 = None
                                if res_s2 is not None:
                                    base_s2 = [x["score"] for x in res_s2[greedy_row]]
                                    alt_s2 = [x["score"] for x in res_s2[best_cand["row"]]]
                                    d2_mean, _, _ = paired_stats(alt_s2, base_s2)

                                    base_all = base_s1 + base_s2
                                    alt_all = [x["score"] for x in res_s1[best_cand["row"]]] + alt_s2
                                    d_all, _, t_all = paired_stats(alt_all, base_all)

                                    base_v = [x["v_after"] for x in res_s1[greedy_row]] + [x["v_after"] for x in res_s2[greedy_row]]
                                    alt_v = [x["v_after"] for x in res_s1[best_cand["row"]]] + [x["v_after"] for x in res_s2[best_cand["row"]]]
                                    dv_all = float(np.mean(alt_v) - np.mean(base_v))
                                    is_anc = f.kind == Kind.EVENT and bool(int(gv[G.EVENT_FLAGS]) & int(EventFlag.ANCIENT))

                                    if passes_macro_emission_filter(
                                        is_ancient=is_anc,
                                        gain_all=d_all,
                                        t_all=t_all,
                                        gain_s2=d2_mean,
                                        dv_after=dv_all,
                                        cfg=cfg,
                                    ):
                                        chosen_row = best_cand["row"]
                                        macro_emitted_ep += 1
                                        if macro_reason == "card_reward_blindspot":
                                            macro_bs_emitted_ep += 1
                                        cand_scores = []
                                        for c in cands:
                                            cr = c["row"]
                                            if cr == greedy_row:
                                                cand_scores.append(float(np.mean(base_all)))
                                            elif cr == chosen_row:
                                                cand_scores.append(float(np.mean(alt_all)))
                                            else:
                                                cand_scores.append(float(np.mean([x["score"] for x in res_s1[cr]])))
                                        cand_p_old = tuple(float(c["prob"]) for c in cands)
                                        s0 = probe_samp[0]
                                        s_copy = Sample(
                                            s0.types.copy(), s0.raw.copy(), s0.rel.copy(), s0.edges.copy(),
                                            s0.cond.copy(), s0.counts, s0.history.copy(),
                                        )
                                        kname = Kind(int(f.kind)).name
                                        emitted.append(
                                            ExitRecord(
                                                sample=s_copy,
                                                target_action=chosen_row,
                                                legal_mask=legal_mask.copy(),
                                                cand_rows=tuple(cand_rows),
                                                greedy_action=greedy_row,
                                                gain=float(d_all),
                                                gain_s2=float(d2_mean),
                                                t_stat=float(min(t_all, 999.0)),
                                                dv_after=float(dv_all),
                                                p_old=float(best_cand["prob"]),
                                                p_group_old=float(best_cand["group_prob"]),
                                                teacher_version=int(ctx.version),
                                                stage=f"MACRO-A{act + 1}-{kname}",
                                                verb=str(best_cand["verb_name"]),
                                                char=char,
                                                ascension=ascension,
                                                ts=time.time(),
                                                cand_scores=tuple(cand_scores),
                                                cand_p_old=cand_p_old,
                                            )
                                        )

            probe_player.record(f, probe_samp[0], chosen_row)
            player = probe_player
            f = host.step(chosen_row)
        else:
            acts, _, samp = decide(ctx, [(f, player)])
            a_row = acts[0]
            player.record(f, samp[0], a_row)
            f = host.step(a_row)

        total_decs += 1
        assert total_decs <= MAX_DECISIONS + 5

    assert f.is_terminal
    won = bool(f.result == Result.WIN)
    if calibrator is not None and observe_calibrator and f.result in (Result.WIN, Result.LOSS):
        for d_act, d_logit in door_logits:
            calibrator.observe(d_act, d_logit, won)
    stats: dict[str, Any] = {
        "seed": run_seed,
        "char": char,
        "ascension": ascension,
        "won": won,
        "floor": int(f.gvals[G.TOTAL_FLOOR]),
        "b3a": entered_b3a,
        "decisions": total_decs,
        "searched_decs": searched_decs,
        "s2_decs": s2_decs,
        "emitted": len(emitted),
        "child_aborts": child_aborts_ep,
        "version": ctx.version,
        "wall_s": round(time.monotonic() - t0, 2),
    }
    if cfg.enable_macro or cfg.macro_v3:
        stats.update(
            {
                "macro_supported": macro_supported,
                "macro_searched": macro_decs_ep,
                "macro_s2": macro_s2_ep,
                "macro_emitted": macro_emitted_ep,
            }
        )
        if cfg.macro_v3:
            stats.update(
                {
                    "macro_v3_s2": macro_s2_ep,
                    "macro_v3_s3": macro_v3_s3_ep,
                    "macro_v3_worlds_total": macro_v3_worlds_ep,
                }
            )
        if cfg.macro_blindspot_prob > 0.0:
            stats.update(
                {
                    "macro_bs_probed": macro_bs_probed_ep,
                    "macro_bs_emitted": macro_bs_emitted_ep,
                }
            )
    return emitted, stats


def run_greedy_control_episode(
    ctx: TeacherCtx,
    host: HostProcess | NativeBatchHost,
    char: str,
    ascension: int,
    run_seed: str,
    collect_door_logits: bool = False,
) -> tuple[dict[str, Any], list[tuple[int, float]]]:
    """Pure greedy 0-fork Control episode on ``run_seed`` for paired A/B lift measurement.

    Returns ``(stats, door_logits)`` without mutating any calibrator so the caller can observe
    control outcomes AFTER both Control and Treatment finish on the same seed (zero leakage!).
    """
    t0 = time.monotonic()
    job = EpisodeJob(parse_lineup(char, ctx.tok.vocab), char, ascension, run_seed, run_seed)
    f = host.reset_with(job.reset_cmd(ctx.flags, MAX_DECISIONS))
    player = Player(ctx)
    door_logits: list[tuple[int, float]] = []
    cur_fight_key: tuple[int, int] | None = None
    entered_b3a = False
    total_decs = 0

    while f.is_obs:
        gv = f.gvals
        act = int(gv[G.ACT_INDEX])
        floor = int(gv[G.TOTAL_FLOOR])
        in_combat = int(gv[G.IN_COMBAT]) == 1
        combats_won = int(gv[G.COMBATS_WON])
        rid = int(gv[G.ROOM_TYPE])
        rname = ctx.room_names[rid] if 0 <= rid < len(ctx.room_names) else f"Room({rid})"
        is_new_fight = False

        if not in_combat:
            cur_fight_key = None
        else:
            fkey = (floor, combats_won)
            if fkey != cur_fight_key:
                cur_fight_key = fkey
                is_new_fight = True
                if act == 2 and rname == "Boss":
                    entered_b3a = True

        acts, _, samp, heads = decide_with_heads(ctx, [(f, player)])
        if is_new_fight and collect_door_logits:
            door_logits.append((act, heads[0]["pwin_logit"]))
        a_row = acts[0]
        player.record(f, samp[0], a_row)
        f = host.step(a_row)
        total_decs += 1
        assert total_decs <= MAX_DECISIONS + 5

    assert f.is_terminal
    won = bool(f.result == Result.WIN)
    stats: dict[str, Any] = {
        "seed": run_seed,
        "char": char,
        "ascension": ascension,
        "won": won,
        "wl_valid": f.result in (Result.WIN, Result.LOSS),
        "floor": int(f.gvals[G.TOTAL_FLOOR]),
        "b3a": entered_b3a,
        "decisions": total_decs,
        "version": ctx.version,
        "wall_s": round(time.monotonic() - t0, 2),
    }
    return stats, door_logits


def format_ab_pair_event(
    *,
    worker_id: str,
    pair_idx: int,
    run_seed: str,
    version: int,
    off_stats: dict[str, Any],
    on_stats: dict[str, Any],
    cum_off: int,
    cum_on: int,
    cum_gained: int,
    cum_lost: int,
) -> dict[str, Any]:
    """Build structured ``ab_pair`` telemetry with exact paired McNemar z-stat."""
    won_off = bool(off_stats["won"])
    won_on = bool(on_stats["won"])
    diff = int(won_on) - int(won_off)
    n_pairs = max(1, int(pair_idx))
    lift_pp = 100.0 * (cum_gained - cum_lost) / n_pairs
    cum_z = (cum_gained - cum_lost) / math.sqrt(max(1, cum_gained + cum_lost))
    return {
        "event": "ab_pair",
        "worker": worker_id,
        "pair": int(pair_idx),
        "seed": run_seed,
        "version": int(version),
        "won_off": won_off,
        "won_on": won_on,
        "diff": diff,
        "floor_off": int(off_stats["floor"]),
        "floor_on": int(on_stats["floor"]),
        "b3a_off": bool(off_stats.get("b3a", False)),
        "b3a_on": bool(on_stats.get("b3a", False)),
        "cum_off": int(cum_off),
        "cum_on": int(cum_on),
        "cum_gained": int(cum_gained),
        "cum_lost": int(cum_lost),
        "cum_lift_pp": round(lift_pp, 3),
        "cum_z": round(cum_z, 3),
        "emitted": int(on_stats.get("emitted", 0)),
        "macro_emitted": int(on_stats.get("macro_emitted", 0)),
        "wall_s_off": float(off_stats["wall_s"]),
        "wall_s_on": float(on_stats["wall_s"]),
    }


# =========================================================================== CLI entrypoint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Async ExIt Search Teacher Worker (py-stage-p13)")
    ap.add_argument("--run-dir", type=Path, required=True, help="PPO training run directory (polls ckpt/latest.pt & weights/LATEST)")
    ap.add_argument("--init-ckpt", type=Path, default=None, help="Fallback initial checkpoint if ckpt/latest.pt does not exist yet")
    ap.add_argument("--host-config", required=True, help="Search-capable host TOML (--search, no --allow-debug-options)")
    ap.add_argument("--removal-ratings", required=True, help="removal-ratings.json path")
    ap.add_argument("--exit-dir", type=Path, default=None, help="Output directory for atomic ExIt shards (required unless --no-shards)")
    ap.add_argument("--char", default="SILENT", help="Character lineup (SILENT, FOOL, etc.)")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--worker-id", default="0")
    ap.add_argument("--seed", type=int, default=2026112113)
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--poll-weights-dir", action="store_true", help="Also poll <run_dir>/weights/LATEST (use when acting_weights=ema)")
    ap.add_argument("--shard-size", type=int, default=32, help="Flush shard once pending records reach this count")
    ap.add_argument("--flush-s", type=float, default=60.0, help="Flush non-empty pending records at least this often")
    ap.add_argument("--max-shards", type=int, default=500, help="Prune oldest shards beyond this cap (<100 MB)")
    ap.add_argument("--max-games", type=int, default=0, help="Stop after N scout games or A/B pairs (0 = run until STOP/SIGINT)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-gb", type=float, default=1.6)
    ap.add_argument("--sock-dir", type=Path, default=Path("/dev/shm/exit-teacher"))
    ap.add_argument("--max-k", type=int, default=4, help="Candidate count per searched decision (default 4)")
    ap.add_argument("--max-k-select", type=int, default=0, help="Wider candidate count on CARD_SELECT (0 = same as --max-k)")
    ap.add_argument("--calibrate-pwin", action="store_true", help="Enable online per-Act Platt pwin calibration from greedy scouts")
    ap.add_argument("--emergency-hallway", action="store_true", help="Also search hallway/event fights with door_pwin < 0.45")
    ap.add_argument("--combat-max-child-steps", type=int, default=1200, help="Anti-stalemate step cap per forked combat child (default 1200; 0 = unlimited)")
    ap.add_argument("--enable-macro", action="store_true", help="Enable fair-play Macro ExIt (requires macro-fork host dist; default False)")
    ap.add_argument("--macro-m1", type=int, default=6, help="Macro Stage-1 CRN world count (default 6)")
    ap.add_argument("--macro-m2", type=int, default=6, help="Macro Stage-2 CRN world count (default 6)")
    ap.add_argument("--macro-horizon-floors", type=int, default=4, help="Macro truncated rollout horizon in floors (default 4)")
    ap.add_argument("--macro-max-per-episode", type=int, default=None, help="Max macro decisions searched per scout episode (default 24; 40 with --macro-v3)")
    ap.add_argument("--macro-blindspot-prob", type=float, default=0.0, help="Opt-in probe rate on confident CARD_REWARD screens containing <2%% blind-spot cards (default 0.0)")
    ap.add_argument("--macro-blindspot-max-prob", type=float, default=0.02, help="Policy group_prob threshold defining a blind-spot card (default 0.02)")
    ap.add_argument("--macro-v3", action="store_true", help="Enable Macro Teacher v3 (§3.2: deep Act/Boss-B horizon, adaptive 16->32->64 CRN, Act-3/Ancient scope; auto-enables macro)")
    ap.add_argument("--macro-v3-max-k", type=int, default=4, help="Max candidates per Macro v3 decision (default 4)")
    ap.add_argument("--macro-v3-min-worlds", type=int, default=48, help="Minimum paired worlds required to emit a Macro v3 label (default 48)")
    ap.add_argument("--macro-v3-min-gain", type=float, default=0.020, help="Minimum paired pwin gain to emit a Macro v3 label (default 0.020)")
    ap.add_argument("--macro-v3-min-t", type=float, default=2.50, help="Minimum paired t-statistic to emit a Macro v3 label (default 2.50)")
    ap.add_argument("--macro-v3-max-child-steps", type=int, default=2500, help="Anti-stalemate step cap per forked Macro v3 child (default 2500)")
    ap.add_argument("--no-combat-search", "--macro-only", dest="no_combat_search", action="store_true", help="Disable combat search (isolate pure Macro lift)")
    ap.add_argument("--ab-paired-lift", action="store_true", help="Alternate pure greedy Control (OFF) and Search (ON) on the exact same seed and emit ab_pair telemetry")
    ap.add_argument("--no-shards", action="store_true", help="Do not write ExIt shard files (useful for pure paired A/B lift benchmarking)")
    ap.add_argument("--native-batch-so", type=Path, default=None, help="Option B: path to Cut #42+ libfast_native_runtime.so for in-process parallel NativeBatch")
    ap.add_argument("--native-batch-threads", type=int, default=8, help="Rust BatchPool worker threads for --native-batch-so (default 8)")
    a = ap.parse_args(argv)
    if (not a.no_shards) and a.exit_dir is None:
        ap.error("--exit-dir is required unless --no-shards is set")
    out_dir = a.exit_dir if a.exit_dir is not None else (a.sock_dir / f"noshards-{a.worker_id}")

    stop_flag = False

    def _on_sig(_signum: int, _frame: Any) -> None:
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, _on_sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_sig)

    torch.set_num_threads(2)
    device = torch.device(a.device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        frac = min(0.95, (a.gpu_gb * (2**30)) / props.total_memory)
        torch.cuda.set_per_process_memory_fraction(frac)

    ckpt_latest = a.run_dir / "ckpt" / "latest.pt"
    init_path = ckpt_latest if ckpt_latest.is_file() else a.init_ckpt
    while (init_path is None or not init_path.is_file()) and not stop_flag:
        if (a.run_dir / "STOP").exists():
            return 0
        time.sleep(2.0)
        init_path = ckpt_latest if ckpt_latest.is_file() else a.init_ckpt
    if stop_flag or init_path is None:
        return 0

    spec = real_host_spec(a.host_config)
    if a.native_batch_so is None:
        assert "--search" in spec.command, f"host-config must use --search: {spec.command}"
        assert "--allow-debug-options" not in spec.command, "FORBIDDEN: --allow-debug-options"
    net, tok, version, extra = load_policy_net(
        init_path, device=a.device, weights="ema", vocab_path=spec.vocab, catalog_path=spec.catalog
    )
    tok.config = dataclasses.replace(
        tok.config,
        commit_select=True,
        removal_rule=True,
        removal_ratings=a.removal_ratings,
        rules_mode="free",
    )
    net.rules_mode = "free"
    policy = TorchPolicy(
        net, tok, mode="greedy", seed=a.seed, version=version,
        use_graphs=(device.type == "cuda"), max_graphs=32, include_pwin=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    gl = GameLogger(tok.vocab, out_dir / f".gl-{a.worker_id}", {})
    flags = reset_flags_for(tok.config.forecast, calc=tok.config.calc, map_in_combat=tok.config.route_summary)
    ctx = TeacherCtx(
        tok=tok,
        net=net,
        policy=policy,
        gl=gl,
        spec=spec,
        flags=flags,
        removal=load_rule(tok.vocab, a.removal_ratings),
        vocab_hash=int(tok.vocab.hash),
        catalog_hash=int(extra["catalog_hash"]),
        version=version,
        room_names=tuple(tok.vocab["enum:RoomType"].entries),
        device=device,
    )

    # Check if run_dir/config.json uses acting_weights="ema" so weights/LATEST is safe EMA
    use_wdir = a.poll_weights_dir
    cfg_json = a.run_dir / "config.json"
    if not use_wdir and cfg_json.is_file():
        try:
            rcfg = json.loads(cfg_json.read_text(encoding="utf-8"))
            if rcfg.get("learner", {}).get("acting_weights") == "ema":
                use_wdir = True
        except Exception:  # noqa: BLE001
            pass

    poller = LiveWeightPoller(
        ckpt_path=ckpt_latest,
        weights_dir=(a.run_dir / "weights") if use_wdir else None,
        poll_s=a.poll_s,
        have_version=version,
    )

    wid_hash = sum((i + 1) * ord(ch) for i, ch in enumerate(str(a.worker_id)))
    run_seed_stream = SeedStream(splitmix64((a.seed + wid_hash * GOLDEN + 0x52554E) & M64))
    fork_stream = SeedStream(splitmix64((a.seed + wid_hash * GOLDEN + 0x46524B) & M64))

    use_native_batch = a.native_batch_so is not None
    sock_path = ""
    listener: socket.socket | None = None
    if not use_native_batch:
        a.sock_dir.mkdir(parents=True, exist_ok=True)
        sock_path = str(a.sock_dir / f"exit-{a.worker_id}-{os.getpid()}.sock")
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(sock_path)
        listener.listen(128)
        listener.settimeout(CHILD_TIMEOUT)

    def _open_host() -> HostProcess | NativeBatchHost:
        if use_native_batch:
            assert a.native_batch_so is not None
            return NativeBatchHost(
                a.native_batch_so,
                threads=a.native_batch_threads,
                expected_hashes=(ctx.vocab_hash, ctx.catalog_hash),
            )
        return make_search_host(ctx, out_dir / "logs", f"teacher-{a.worker_id}")

    host: HostProcess | NativeBatchHost = _open_host()
    host_pid = host.pid
    last_pids: list[int] = []
    tag_box = [(wid_hash % 1000) * 1000000]
    eff_enable_macro = bool(a.enable_macro or a.macro_v3)
    eff_macro_cap = (
        int(a.macro_max_per_episode)
        if a.macro_max_per_episode is not None
        else (40 if a.macro_v3 else 24)
    )
    cfg = ExitTeacherConfig(
        max_k=a.max_k,
        max_k_card_select=a.max_k_select,
        emergency_hallway=a.emergency_hallway,
        combat_max_child_steps=a.combat_max_child_steps,
        enable_combat_search=(not a.no_combat_search),
        enable_macro=eff_enable_macro,
        macro_m1=a.macro_m1,
        macro_m2=a.macro_m2,
        macro_horizon_floors=a.macro_horizon_floors,
        macro_max_per_episode=eff_macro_cap,
        macro_blindspot_prob=a.macro_blindspot_prob,
        macro_blindspot_max_prob=a.macro_blindspot_max_prob,
        macro_v3=a.macro_v3,
        macro_v3_max_k=a.macro_v3_max_k,
        macro_v3_min_worlds=a.macro_v3_min_worlds,
        macro_v3_min_gain=a.macro_v3_min_gain,
        macro_v3_min_t=a.macro_v3_min_t,
        macro_v3_max_child_steps=a.macro_v3_max_child_steps,
        macro_v3_max_per_episode=eff_macro_cap,
    )
    calibrator = OnlinePwinCalibrator() if a.calibrate_pwin else None

    pending: list[ExitRecord] = []
    shard_seq = 0
    last_flush = time.monotonic()
    games_done = 0
    ab_cum_off = 0
    ab_cum_on = 0
    ab_cum_gained = 0
    ab_cum_lost = 0

    def _flush() -> None:
        nonlocal shard_seq, last_flush
        if a.no_shards or not pending:
            return
        assert a.exit_dir is not None
        shard_seq += 1
        p = save_exit_shard(
            a.exit_dir,
            a.worker_id,
            shard_seq,
            pending,
            max_shards=a.max_shards,
            vocab_hash=ctx.vocab_hash,
            catalog_hash=ctx.catalog_hash,
        )
        print(json.dumps({"event": "shard", "worker": a.worker_id, "path": p.name, "n": len(pending), "v": ctx.version}), flush=True)
        pending.clear()
        last_flush = time.monotonic()

    try:
        while not stop_flag:
            if (a.run_dir / "STOP").exists():
                break
            if poller.maybe_refresh(ctx.net):
                ctx.version = poller.version
                print(json.dumps({"event": "weights", "worker": a.worker_id, "version": ctx.version}), flush=True)
            rseed = make_scout_seed(run_seed_stream)
            game_pids: list[int] = []
            try:
                if a.ab_paired_lift:
                    off_stats, off_door_logits = run_greedy_control_episode(
                        ctx, host, a.char, a.ascension, rseed, collect_door_logits=(calibrator is not None),
                    )
                    recs, stats = run_scout_episode(
                        ctx, host, a.char, a.ascension, rseed, fork_stream, cfg, listener, sock_path, tag_box, game_pids,
                        calibrator=calibrator, observe_calibrator=False,
                    )
                else:
                    off_stats = None
                    off_door_logits = []
                    recs, stats = run_scout_episode(
                        ctx, host, a.char, a.ascension, rseed, fork_stream, cfg, listener, sock_path, tag_box, game_pids,
                        calibrator=calibrator,
                    )
            except (HostError, EOFError, BrokenPipeError, OSError) as exc:
                print(
                    json.dumps(
                        {
                            "event": "host_recovered",
                            "worker": a.worker_id,
                            "seed": rseed,
                            "error": str(exc)[:240],
                        }
                    ),
                    flush=True,
                )
                try:
                    host.close()
                except Exception:  # noqa: BLE001
                    pass
                host = _open_host()
                host_pid = host.pid
                continue
            last_pids = game_pids[-512:]
            if not a.no_shards:
                pending.extend(recs)
            games_done += 1
            if a.ab_paired_lift and off_stats is not None:
                if calibrator is not None and off_stats.get("wl_valid", True):
                    for d_act, d_logit in off_door_logits:
                        calibrator.observe(d_act, d_logit, bool(off_stats["won"]))
                won_off = bool(off_stats["won"])
                won_on = bool(stats["won"])
                ab_cum_off += int(won_off)
                ab_cum_on += int(won_on)
                if won_on and not won_off:
                    ab_cum_gained += 1
                elif won_off and not won_on:
                    ab_cum_lost += 1
                ab_ev = format_ab_pair_event(
                    worker_id=str(a.worker_id),
                    pair_idx=games_done,
                    run_seed=rseed,
                    version=ctx.version,
                    off_stats=off_stats,
                    on_stats=stats,
                    cum_off=ab_cum_off,
                    cum_on=ab_cum_on,
                    cum_gained=ab_cum_gained,
                    cum_lost=ab_cum_lost,
                )
                print(json.dumps(ab_ev), flush=True)
            print(json.dumps({"event": "game", "worker": a.worker_id, "game": games_done, **stats}), flush=True)
            if (not a.no_shards) and (len(pending) >= a.shard_size or (pending and (time.monotonic() - last_flush) >= a.flush_s)):
                _flush()
            if a.max_games > 0 and games_done >= a.max_games:
                break
        _flush()
    finally:
        if listener is not None:
            listener.close()
        if sock_path and os.path.exists(sock_path):
            os.unlink(sock_path)
        host.close()
        ctx.policy.close()

    if not use_native_batch:
        time.sleep(1.0)
        left = [p for p in ([host_pid] + last_pids) if _alive_pid(p)]
        assert not left, f"exit_teacher worker {a.worker_id} leaked PIDs: {left}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
