"""Habit study hook (spire.habit): every habit fires where its definition says on recorded 4-seat frames, returns only
legal, not banned rows, its judge() agrees with its choose() and grades only its own choice set, the round-1
verifier's defects stay fixed, the event page table (r3 events), and the actor's evaluation-only override hook. The
tactical combat habits: test_habit_tactics.py."""
from __future__ import annotations

import copy
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeJob, EpisodeSampler, TorchPolicy, UniformPolicy
from spire.advice import NO_JUDGE, Advisor, advice_offset, habit_offset
from spire.evalpanel import _NoticeJobs
from spire.habit import Call, HabitOverride, HabitRecorder, PageTable, _Names, parse_habit
from spire.model import ModelConfig, SpireNet
from spire.proto import Frame, parse_frame
from spire.rewards import RoomIds
from spire.rollout import RolloutStore
from spire.schema import ActionCol, CardCol, CardSelectSub, CreatureCol, CreatureKind, G, Kind, MapNodeCol, OptionCol, \
    OptionKind, PowerCol, SeatCol, Table, Verb, Zone
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Catalog, Vocab
from test_advice import ROOT

GOLDEN_DIR = ROOT / "bench" / "h5" / "golden4"
DIST = ROOT / "host" / "dist" / "20261115-194451-h5-4p" / "vocab"
FILES = ("frames4_nat.bin", "frames4_hp.bin")  # 4-seat natural games; frames4_hp: more REST / SHOP decisions
# card-reward preferences over cards the recorded games offer often (AFTERLIFE ... COOLHEADED: 9-13 offers each)
PREFS = "prefs:list:AFTERLIFE,BEGONE,POISONED_STAB,HOTFIX/DAGGER_SPRAY,IRON_WAVE,POMMEL_STRIKE,COLD_SNAP,BLIGHT_STRIKE"
AVOID_ONLY = "prefs:list:/AFTERLIFE,BEGONE,POISONED_STAB,HOTFIX,DAGGER_SPRAY,IRON_WAVE,POMMEL_STRIKE,COLD_SNAP"
JUDGED = ("smith:0.75", "smith:0.5", "smith_pre:0.75:1.0", "no_elite:1-2", "no_elite:1-3", "potion_hold", "potion_boss",
          "potion_r1:Elite,Boss", "potion_r1:Monster", "enrage:none", "enrage:block", "shop_relic", "shop_nocard",
          "shop_nopotion", "deck_cap:12", "deck_cap:16", PREFS, AVOID_ONLY, "heal_late:1.0:0:3", "heal_late:1.0:0:5",
          "potion_none")
# specs the recorded games exercise only thinly -> the minimum count of judged rows: the games reach round 1 of an Elite
# / Boss fight with a usable potion on 2 frames (2 rows); test_potion_r1_on_injected_rooms covers it on 206 frames
THIN = {"potion_r1:Elite,Boss": 2}
TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)


class Golden:
    def __init__(self) -> None:
        if not GOLDEN_DIR.exists() or not (DIST / "catalog.json").exists():
            pytest.skip(f"recorded 4-seat frames {GOLDEN_DIR} / vocab {DIST} not present")
        self.vocab = Vocab.load(DIST / "vocab.json")
        self.catalog = Catalog.load(DIST / "catalog.json", self.vocab)
        self.names = _Names(self.vocab, self.catalog)
        self.frames: list[Frame] = []  # OBS frames of both files, in recorded order
        for name in FILES:
            with (GOLDEN_DIR / name).open("rb") as fh:
                while head := fh.read(4):
                    (size,) = struct.unpack("<I", head)
                    f = parse_frame(fh.read(size))
                    if f.msg_type == 1:
                        self.frames.append(f)
        assert len(self.frames) == 19365 + 21182 and all(f.vocab_hash == self.vocab.hash for f in self.frames[:50])
        assert all(f.gvals[G.N_SEATS] == 4 for f in self.frames)

    def of(self, *kinds: Kind) -> list[Frame]:
        return [f for f in self.frames if f.gvals[G.KIND] in kinds]

    # ------------------------------------------------------------------ names, row by row (independent of habit.py)

    def option(self, f: Frame, row: int) -> tuple[str, str] | None:
        """(option kind name, key name) of a CHOOSE_OPTION row of the OPTIONS table, else None."""
        a = f.actions[row]
        if a[ActionCol.VERB] != Verb.CHOOSE_OPTION or a[ActionCol.SRC_TABLE] != Table.OPTIONS:
            return None
        o = f.table(Table.OPTIONS)[a[ActionCol.SRC_ROW]]
        kind = OptionKind(int(o[OptionCol.KIND]))
        cat = {OptionKind.REST_OPTION: "rest_option", OptionKind.SHOP_ENTRY: "shop_section",
               OptionKind.CARD_REWARD_ALT: "card_reward_alt"}.get(kind)
        return kind.name, (self.vocab.name(cat, int(o[OptionCol.KEY_ID])) if cat else "")

    def room(self, f: Frame) -> str:
        return self.vocab.name("enum:RoomType", int(f.gvals[G.ROOM_TYPE]))

    def card(self, f: Frame, row: int) -> dict[str, Any] | None:
        """Catalog entry (level 0) of the card a PLAY_CARD row plays, else None."""
        a = f.actions[row]
        if a[ActionCol.VERB] != Verb.PLAY_CARD:
            return None
        cid = int(f.table(Table.CARDS)[a[ActionCol.SRC_ROW], CardCol.CARD_ID])
        return self.catalog.entries["card"][cid][0]

    def offered(self, f: Frame, row: int) -> str | None:
        """Card name of a PICK_OFFER_CARD row, else None."""
        a = f.actions[row]
        if a[ActionCol.VERB] != Verb.PICK_OFFER_CARD:
            return None
        return self.vocab.name("card", int(f.table(Table.CARDS)[a[ActionCol.SRC_ROW], CardCol.CARD_ID]))

    def deck_size(self, f: Frame) -> int:
        """Cards in the acting seat's deck (CARDS rows of zone DECK)."""
        c = f.table(Table.CARDS)
        return int(c[(c[:, CardCol.ZONE] == Zone.DECK) & (c[:, CardCol.SEAT] == f.gvals[G.ACTING_SEAT]),
                     CardCol.COUNT].sum())

    def point(self, f: Frame, row: int) -> str:
        node = f.table(Table.MAP_NODES)[f.actions[row, ActionCol.SRC_ROW]]
        return self.vocab.name("enum:MapPointType", int(node[MapNodeCol.POINT_TYPE]))


@pytest.fixture(scope="module")
def golden() -> Golden:
    return Golden()


def is_rest_option(gd: Golden, f: Frame, row: int) -> bool:
    return (gd.option(f, row) or ("",))[0] == "REST_OPTION"


def reference(gd: Golden, spec: str) -> Callable[[Frame, int, tuple[int, ...]], int]:
    """The habit's judgement written out from its definition (module docstring of spire.habit), row by row."""
    name, _, arg = spec.partition(":")

    def allowed(f: Frame, banned: tuple[int, ...]) -> list[int]:
        return [r for r in range(f.n_actions) if r not in banned]

    if name in ("smith", "smith_pre"):
        t, t_pre = (float(arg), None) if name == "smith" else (float(x) for x in arg.split(":"))

        def smith(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            g = f.gvals
            if g[G.KIND] == Kind.REST:
                rows = [r for r in allowed(f, banned) if gd.option(f, r) == ("REST_OPTION", "SMITH")]
                if not rows or not is_rest_option(gd, f, action):
                    return 0  # only a choice among the rest options is judged
                thr = t
                if t_pre is not None:
                    nodes = f.table(Table.MAP_NODES)
                    boss_rows = [int(n[MapNodeCol.ROW]) for n in nodes
                                 if gd.vocab.name("enum:MapPointType", int(n[MapNodeCol.POINT_TYPE])) == "Boss"]
                    thr = t_pre if g[G.CUR_ROW] == min(boss_rows) - 1 else t
                seat = f.table(Table.SEATS)[g[G.ACTING_SEAT]]
                up = int(seat[SeatCol.HP]) >= thr * int(seat[SeatCol.MAX_HP])
                return (1 if action == rows[0] else -1) * (1 if up else -1)
            if (g[G.KIND] == Kind.CARD_SELECT and g[G.KIND_SUB] == CardSelectSub.DECK_UPGRADE and not g[G.IN_COMBAT]
                    and gd.room(f) == "RestSite"):
                confirm = [r for r in allowed(f, banned) if f.actions[r, ActionCol.VERB] == Verb.CONFIRM_SELECTION]
                if not confirm:
                    return 0
                return 1 if action == confirm[0] else -1 if f.actions[action, ActionCol.VERB] == Verb.CANCEL else 0
            return 0
        return smith
    if name == "heal_late":
        t_s, *rest = arg.split(":")
        t, act, below = float(t_s), int(rest[0]) if rest else 3, int(rest[1]) if len(rest) > 1 else 1

        def heal_late(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            g = f.gvals
            if g[G.KIND] != Kind.REST or (act and g[G.ACT_INDEX] + 1 != act):
                return 0
            heal = [r for r in allowed(f, banned) if gd.option(f, r) == ("REST_OPTION", "HEAL")]
            if not heal or not is_rest_option(gd, f, action):
                return 0
            boss_rows = [int(n[MapNodeCol.ROW]) for n in f.table(Table.MAP_NODES)
                         if gd.vocab.name("enum:MapPointType", int(n[MapNodeCol.POINT_TYPE])) == "Boss"]
            seat = f.table(Table.SEATS)[g[G.ACTING_SEAT]]
            if g[G.CUR_ROW] < min(boss_rows) - below or not int(seat[SeatCol.HP]) < t * int(seat[SeatCol.MAX_HP]):
                return 0
            return 1 if action == heal[0] else -1
        return heal_late
    if name == "no_elite":
        lo, hi = (int(x) for x in arg.split("-"))

        def no_elite(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            g = f.gvals
            if g[G.KIND] != Kind.MAP or not lo <= g[G.ACT_INDEX] + 1 <= hi:
                return 0
            travel = [r for r in allowed(f, banned) if f.actions[r, ActionCol.VERB] == Verb.TRAVEL]
            elite = {r for r in travel if gd.point(f, r) == "Elite"}
            if not elite or len(elite) == len(travel):
                return 0
            return -1 if action in elite else 0
        return no_elite
    if name in ("potion_hold", "potion_boss", "potion_none"):
        rooms = {"potion_hold": ("Monster",), "potion_boss": ("Monster", "Elite"),
                 "potion_none": ("Monster", "Elite", "Boss")}[name]

        def potion(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            return -1 if (f.gvals[G.KIND] == Kind.COMBAT and gd.room(f) in rooms
                          and f.actions[action, ActionCol.VERB] == Verb.USE_POTION) else 0
        return potion
    if name == "potion_r1":
        r1_rooms = arg.split(",")

        def potion_r1(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            return 1 if (f.gvals[G.KIND] == Kind.COMBAT and f.gvals[G.COMBAT_ROUND] == 1 and gd.room(f) in r1_rooms
                         and f.actions[action, ActionCol.VERB] == Verb.USE_POTION) else 0
        return potion_r1
    if name == "enrage":
        def enrage(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            if f.gvals[G.KIND] != Kind.COMBAT:
                return 0
            enraged = any(gd.vocab.name("power", int(p[PowerCol.POWER_ID])) == "ENRAGE_POWER" and p[PowerCol.AMOUNT] > 0
                          for p in f.table(Table.POWERS))
            card = gd.card(f, action)
            skill = card is not None and card["type"] == "Skill" and (arg == "none" or not card["gains_block"])
            return -1 if enraged and skill else 0
        return enrage
    if name == "shop_relic":
        def shop_relic(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            if f.gvals[G.KIND] != Kind.SHOP:
                return 0
            relic = [r for r in allowed(f, banned) if gd.option(f, r) == ("SHOP_ENTRY", "RELIC")]
            if not relic:
                return 0
            return 1 if action in relic else -1 if (gd.option(f, action) or ("",))[0] == "SHOP_ENTRY" else 0
        return shop_relic
    if name in ("shop_nocard", "shop_nopotion"):
        sections = ("CHARACTER_CARD", "COLORLESS_CARD") if name == "shop_nocard" else ("POTION",)

        def shop_ban(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            o = gd.option(f, action)
            return -1 if f.gvals[G.KIND] == Kind.SHOP and o is not None and o[0] == "SHOP_ENTRY" and o[1] in sections \
                else 0
        return shop_ban
    if name == "deck_cap":
        cap = int(arg)

        def deck_cap(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            if f.gvals[G.KIND] != Kind.CARD_REWARD or gd.deck_size(f) < cap:
                return 0
            skip = [r for r in allowed(f, banned) if gd.option(f, r) == ("CARD_REWARD_ALT", "SKIP")]
            if not skip:
                return 0
            return 1 if action == skip[0] else -1 if gd.offered(f, action) is not None else 0
        return deck_cap
    if name == "prefs":
        mode, _, lists = arg.partition(":")
        assert mode == "list"
        prefer, avoid = (set(filter(None, x.split(","))) for x in lists.split("/"))

        def prefs(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            if f.gvals[G.KIND] != Kind.CARD_REWARD:
                return 0
            ok = allowed(f, banned)
            liked = [r for r in ok if gd.offered(f, r) in prefer]
            if liked:
                other = gd.offered(f, action) is not None or gd.option(f, action) == ("CARD_REWARD_ALT", "SKIP")
                return 1 if action in liked else -1 if other else 0
            avoided = [r for r in ok if gd.offered(f, r) in avoid]
            return -1 if action in avoided and len(ok) > len(avoided) else 0
        return prefs
    raise AssertionError(spec)


def bans_of(f: Frame, rng: np.random.Generator) -> tuple[int, ...]:
    """A random cycle-ban mask over the rows that can close a cycle: never END_TURN or PROCEED (they leave the
    public state: a new round, the next room), so at least one row stays."""
    can = np.flatnonzero(~np.isin(f.actions[:, ActionCol.VERB], (Verb.END_TURN, Verb.PROCEED)))
    if not len(can) or rng.random() < 0.5:
        return ()
    k = int(rng.integers(1, len(can) + 1))
    return tuple(sorted(int(x) for x in rng.choice(can, k, replace=False)))


def frames_for(gd: Golden, h: Any, combat_every: int) -> list[Frame]:
    """The frames of the habit's decision kinds; COMBAT (16k + 18k frames) thinned to every ``combat_every``-th plus
    every COMBAT frame with a USE_POTION row."""
    out = []
    for i, f in enumerate(gd.of(*h.KINDS)):
        if f.gvals[G.KIND] == Kind.COMBAT and i % combat_every and \
                not (f.actions[:, ActionCol.VERB] == Verb.USE_POTION).any():
            continue
        out.append(f)
    return out


# =========================================================================== judge: where each habit fires


@pytest.mark.parametrize("spec", JUDGED)
def test_judge_fires_exactly_where_the_definition_says(golden: Golden, spec: str) -> None:
    (h,) = parse_habit(spec)
    ref = reference(golden, spec)
    rng = np.random.default_rng(1)
    fired = 0
    for f in frames_for(golden, h, 1):
        for banned in ((), bans_of(f, rng)):
            for a in range(f.n_actions):
                if a in banned:
                    continue
                got = h.judge(golden.names, f, f.gvals, a, banned)
                assert got == ref(f, a, banned), (spec, f, a, banned)
                fired += got != 0
                # the Advisor's precondition: a non-zero judgement only for actions of the habit's VERBS
                assert not got or h.VERBS is None or int(f.actions[a, ActionCol.VERB]) in h.VERBS
    # frames outside the habit's kinds are never judged
    for f in golden.frames[::7]:
        if f.gvals[G.KIND] not in h.KINDS:
            assert all(h.judge(golden.names, f, f.gvals, a, ()) == 0 for a in range(f.n_actions))
    if spec.startswith("enrage"):
        assert fired == 0  # no creature holds ENRAGE_POWER in these games: see test_enrage_on_injected_power
    else:
        assert fired >= THIN.get(spec, 21), fired


# =========================================================================== choose: legal rows; judge agrees


def scores_of(f: Frame, banned: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    s = rng.standard_normal(f.n_actions).astype(np.float32)
    s[list(banned)] = -np.inf
    return s


def check_choose_and_judge(gd: Golden, spec: str, frames: list[Frame], rng: np.random.Generator) -> dict[str, int]:
    """For every frame, a random ban mask and every possible greedy action: choose() returns None or a legal, not
    banned row (potion habits: never a USE / DISCARD_POTION row); where it overrides the greedy action, the greedy
    action is judged -1 or the chosen row +1; where it returns None for every greedy action, every action is judged 0
    (except the deliberate below-threshold side of smith)."""
    (h,) = parse_habit(spec)
    n = gd.names
    c = {"frames": 0, "overrides": 0, "silent": 0, "two_sided": 0}
    for f in frames:
        g = f.gvals
        banned = bans_of(f, rng)
        base = scores_of(f, banned, rng)
        ok = [a for a in range(f.n_actions) if a not in banned]
        judge = {a: h.judge(n, f, g, a, banned) for a in ok}
        any_choice = False
        for greedy in ok:
            s = base.copy()
            s[greedy] = base[ok].max() + 1.0
            chosen = h.choose(n, f, g, s, greedy, Call(0, None))
            if chosen is None:
                continue
            any_choice = True
            assert 0 <= chosen < f.n_actions and chosen not in banned and np.isfinite(s[chosen]), (spec, chosen, banned)
            if spec in ("potion_hold", "potion_boss", "potion_none"):
                assert f.actions[chosen, ActionCol.VERB] not in (Verb.USE_POTION, Verb.DISCARD_POTION)
            if spec.startswith("potion_r1"):
                assert f.actions[chosen, ActionCol.VERB] == Verb.USE_POTION
            if chosen != greedy:
                c["overrides"] += 1
                assert judge[greedy] == -1 or judge[chosen] == 1, (spec, f, greedy, chosen, judge)
            else:
                assert judge[chosen] == 1, (spec, f, chosen, judge)  # it asks for what the policy chose
        c["frames"] += 1
        if not any_choice:
            if spec.startswith("smith") and g[G.KIND] == Kind.REST and any(judge.values()):
                # SMITH offered below the threshold: -1 SMITH, +1 the other rest options, 0 every other row
                c["two_sided"] += 1
                assert sum(v < 0 for v in judge.values()) == 1
                assert all(bool(v) == is_rest_option(gd, f, a) for a, v in judge.items())
            else:
                assert not any(judge.values()), (spec, f, judge)
                c["silent"] += 1
    return c


@pytest.mark.parametrize("spec", JUDGED)
def test_choose_returns_legal_rows_and_judge_agrees(golden: Golden, spec: str) -> None:
    (h,) = parse_habit(spec)
    rng = np.random.default_rng(2)
    c = check_choose_and_judge(golden, spec, frames_for(golden, h, 6), rng)
    assert c["frames"] > 100
    if not spec.startswith("enrage"):
        assert c["overrides"] > (0 if spec in THIN else 10), c
    if spec.startswith("smith"):
        assert c["two_sided"] > 5, c


def inject_enrage(gd: Golden, f: Frame, amount: int) -> Frame:
    """``f`` with its first POWERS row turned into ENRAGE_POWER of ``amount`` (no recorded game reached TEST_SUBJECT)."""
    out = copy.copy(f)
    tables = list(f.tables)
    powers = tables[Table.POWERS].copy()
    powers[0, PowerCol.POWER_ID] = gd.vocab.id("power", "ENRAGE_POWER")
    powers[0, PowerCol.AMOUNT] = amount
    tables[Table.POWERS] = powers
    out.tables = tables
    return out


def test_enrage_on_injected_power(golden: Golden) -> None:
    rng = np.random.default_rng(3)
    names = golden.names
    frames = []
    for f in golden.of(Kind.COMBAT)[::5]:
        if len(f.table(Table.POWERS)) and any((golden.card(f, a) or {}).get("type") == "Skill" for a in range(f.n_actions)):
            frames.append(inject_enrage(golden, f, int(rng.integers(1, 4))))
    assert len(frames) > 500
    for spec in ("enrage:none", "enrage:block"):
        (h,) = parse_habit(spec)
        ref = reference(golden, spec)
        fired = 0
        for f in frames:
            for a in range(f.n_actions):
                got = h.judge(names, f, f.gvals, a, ())
                assert got == ref(f, a, ())
                fired += got != 0
        c = check_choose_and_judge(golden, spec, frames, rng)
        assert fired > 300 and c["overrides"] > 300, (spec, fired, c)
        zero = [inject_enrage(golden, f, 0) for f in frames[:100]]  # amount 0: no habit
        assert all(h.judge(names, f, f.gvals, a, ()) == 0 for f in zero for a in range(f.n_actions))
    block = parse_habit("enrage:block")[0]  # block Skills stay allowed under enrage:block
    blocking = [(f, a) for f in frames for a in range(f.n_actions)
                if (golden.card(f, a) or {}).get("type") == "Skill" and golden.card(f, a)["gains_block"]]
    assert blocking and all(block.judge(names, f, f.gvals, a, ()) == 0 for f, a in blocking)


def test_smith_two_sided_and_pre_boss_threshold(golden: Golden) -> None:
    names = golden.names
    rest = [f for f in golden.of(Kind.REST) if parse_habit("smith:0.75")[0]._offer(names, f, f.gvals,
                                                                                    np.ones(f.n_actions, bool))]
    assert len(rest) > 100
    s75, pre = parse_habit("smith:0.75")[0], parse_habit("smith_pre:0.75:1.0")[0]
    differ = 0
    for f in rest:
        row, up = s75._offer(names, f, f.gvals, np.ones(f.n_actions, bool))
        other = next(a for a in range(f.n_actions) if a != row and is_rest_option(golden, f, a))
        assert s75.judge(names, f, f.gvals, row, ()) == (1 if up else -1)
        assert s75.judge(names, f, f.gvals, other, ()) == (-1 if up else 1)
        assert s75.judge(names, f, f.gvals, other, (row,)) == 0  # a banned SMITH is not on offer
        assert all(s75.judge(names, f, f.gvals, a, ()) == 0 for a in range(f.n_actions)
                   if not is_rest_option(golden, f, a))
        differ += pre.judge(names, f, f.gvals, row, ()) != s75.judge(names, f, f.gvals, row, ())
    assert differ > 3  # rest sites below the boss row with 0.75 <= HP < max HP


# =========================================================================== choice sets (notes/advfix)

REST_RULE = ("REST", "SMITH", 0.75, 0.05, 0.05)  # the live runs' old-format rule


def in_choice_set(gd: Golden, spec: str, f: Frame, row: int) -> bool:
    """``row`` belongs to the rule's own choice set at ``f`` (written out from the definitions, independent of
    habit.py): the only rows a judgement may grade non-zero."""
    name, _, arg = spec.partition(":")
    verb = f.actions[row, ActionCol.VERB]
    o = gd.option(f, row)
    if name in ("smith", "smith_pre", "REST"):
        return is_rest_option(gd, f, row) if f.gvals[G.KIND] == Kind.REST else \
            verb in (Verb.CONFIRM_SELECTION, Verb.CANCEL)
    if name == "heal_late":
        return is_rest_option(gd, f, row)
    if name.startswith("shop"):
        return o is not None and o[0] == "SHOP_ENTRY"
    if name == "no_elite":
        return verb == Verb.TRAVEL
    if name == "potion_lethal":  # the lethal seat's END_TURN or its saving potion use
        return verb in (Verb.END_TURN, Verb.USE_POTION)
    if name in ("potion_boss_use", "lethal_block", "lethal_eot"):  # what a seat does with its turn
        return verb in (Verb.PLAY_CARD, Verb.END_TURN, Verb.USE_POTION)
    if name.startswith("potion"):  # potion_hold / _boss / _none / _r1 / _reserve2 / _boss_take
        return verb == Verb.USE_POTION
    if name == "enrage":
        return verb == Verb.PLAY_CARD
    if name in ("kill_dangerous", "weak_attacker"):  # the aim of one card (/ WEAK_POTION) at an enemy
        tgt = int(f.actions[row, ActionCol.TGT_ROW])
        aims = tgt >= 0 and f.table(Table.CREATURES)[tgt, CreatureCol.KIND] == CreatureKind.ENEMY
        return aims and (verb == Verb.PLAY_CARD or (name == "weak_attacker" and not arg and verb == Verb.USE_POTION))
    if name == "explore":  # a non-relic option of the event page
        return o is not None and o[0] == "EVENT_OPTION" and gd.vocab.name(
            "event_option_key", int(f.table(Table.OPTIONS)[f.actions[row, ActionCol.SRC_ROW], OptionCol.KEY_ID])) \
            != "RELIC_OPTION"
    assert name in ("deck_cap", "prefs"), spec
    return verb == Verb.PICK_OFFER_CARD or o == ("CARD_REWARD_ALT", "SKIP")


@pytest.mark.parametrize("spec", JUDGED)
def test_judgements_grade_only_the_rules_own_choice_set(golden: Golden, spec: str) -> None:
    """Every recorded frame of the habit's kinds, every row: a non-zero judgement only for a row of the rule's choice
    set; at the judged frames the rows outside it (potion use / discard, PROCEED, other options, ...) get 0."""
    (h,) = parse_habit(spec)
    judged = outside = 0
    for f in frames_for(golden, h, 1):
        s = [h.judge(golden.names, f, f.gvals, a, ()) for a in range(f.n_actions)]
        for a, x in enumerate(s):
            assert not x or in_choice_set(golden, spec, f, a), (spec, f, a, x)
        judged += sum(map(bool, s))
        outside += any(s) and sum(not in_choice_set(golden, spec, f, a) for a in range(f.n_actions))
    if not spec.startswith("enrage"):  # enrage: test_enrage_on_injected_power
        assert judged >= THIN.get(spec, 21), judged
    # rows outside the choice set at judged frames, the rows the "-1 otherwise" judgements used to grade: smith the 5
    # DISCARD_POTION rows next to SMITH (PROCEED and USE_POTION: test_rest_frame_potion_and_proceed_rows_...); prefs:
    # no recorded PREFER frame has one (test_prefs_does_not_judge_the_sacrifice)
    if spec.split(":")[0] in ("smith", "smith_pre"):
        assert outside == 5, outside
    elif spec.split(":")[0] in ("shop_relic", "potion_r1"):
        assert outside > (5 if spec in THIN else 20), outside


def test_old_rest_rule_records_only_rest_option_choices(golden: Golden) -> None:
    """The old-format rule (Advisor.candidate) on every recorded frame and row: a record exactly for the rest-option
    choices of REST frames offering SMITH; the DISCARD_POTION / PROCEED rows of REST frames carry none. frames4_nat
    reproduces the production hot-fix's check (441 rest-option rows with a record, 3 DISCARD_POTION and 77 PROCEED
    rows without); PROCEED is offered only at REST frames whose rest choices are made (no SMITH row)."""
    adv = Advisor(golden.vocab, [REST_RULE])
    n_nat = 19365  # OBS frames of frames4_nat.bin, the first file
    c: Counter[tuple[str, str, bool]] = Counter()
    for k, f in enumerate(golden.frames):
        rest = f.gvals[G.KIND] == Kind.REST
        offered = rest and any(golden.option(f, a) == ("REST_OPTION", "SMITH") for a in range(f.n_actions))
        for a in range(f.n_actions):
            row, _ = adv.candidate(f, (), a)
            assert (row >= 0) == (offered and is_rest_option(golden, f, a)), (f, a)
            if rest:
                what = "REST_OPTION" if is_rest_option(golden, f, a) else Verb(int(f.actions[a, ActionCol.VERB])).name
                c[("nat" if k < n_nat else "hp", what, row >= 0)] += 1
    assert c == {("nat", "REST_OPTION", True): 441, ("nat", "DISCARD_POTION", False): 3, ("nat", "PROCEED", False): 77,
                 ("hp", "REST_OPTION", True): 541, ("hp", "DISCARD_POTION", False): 5, ("hp", "PROCEED", False): 97}, c


def with_hp(f: Frame, hp: int) -> Frame:
    """``f`` with the acting seat's HP set to ``hp``."""
    out = copy.copy(f)
    tables = list(f.tables)
    seats = tables[Table.SEATS].copy()
    seats[f.gvals[G.ACTING_SEAT], SeatCol.HP] = hp
    tables[Table.SEATS] = seats
    out.tables = tables
    return out


def as_verb(f: Frame, verb: Verb) -> Frame:
    """``f`` with its DISCARD_POTION rows turned into ``verb`` rows (no recorded REST frame offers a usable potion, or
    PROCEED next to SMITH)."""
    out = copy.copy(f)
    tables = list(f.tables)
    acts = tables[Table.ACTIONS].copy()
    acts[acts[:, ActionCol.VERB] == Verb.DISCARD_POTION, ActionCol.VERB] = verb
    tables[Table.ACTIONS] = acts
    out.tables = tables
    return out


def test_rest_frame_potion_and_proceed_rows_get_no_offset_on_both_sides_of_the_threshold(golden: Golden) -> None:
    """REST frames offering DISCARD_POTION (recorded: 5 frames; also turned into USE_POTION and PROCEED rows) next to
    HEAL / SMITH, the acting seat at full HP and at 1 HP: those rows get offset 0 in the old path (candidate -> advice_offset) and in the habit path
    (smith:0.75 -> habit_offset); the rest-option rows get SMITH +b / HEAL -b above and SMITH -b / HEAL +b below, equal
    in both paths."""
    b = 0.05
    old, new = Advisor(golden.vocab, [REST_RULE]), Advisor(golden.vocab, [("smith:0.75", b)])
    base = [f for f in golden.of(Kind.REST)
            if {golden.option(f, a) for a in range(f.n_actions)} >= {("REST_OPTION", "SMITH"), ("REST_OPTION", "HEAL")}
            and (f.actions[:, ActionCol.VERB] == Verb.DISCARD_POTION).any()]
    frames = base + [as_verb(f, v) for f in base for v in (Verb.USE_POTION, Verb.PROCEED)]
    zero = Counter()
    for f0 in frames:
        max_hp = int(f0.table(Table.SEATS)[f0.gvals[G.ACTING_SEAT], SeatCol.MAX_HP])
        for hp, up in ((max_hp, True), (1, False)):
            f = with_hp(f0, hp)
            for a in range(f.n_actions):
                row, frac = old.candidate(f, (), a)
                o_old = advice_offset(old.rules, np.array([Kind.REST], np.int8), np.array([a], np.int64),
                                      np.array([row], np.int16), np.array([frac], np.float32))[0]
                i, s = new.judge(f, (), a, int(f.actions[a, ActionCol.VERB]))
                o_new = habit_offset(new.habits, np.array([i], np.int8), np.array([s], np.int8))[0]
                key = golden.option(f, a)
                assert o_old == o_new, (f, a, hp)
                if key == ("REST_OPTION", "SMITH"):
                    assert o_old == np.float32(b if up else -b)
                elif key == ("REST_OPTION", "HEAL"):
                    assert o_old == np.float32(-b if up else b)
                elif not is_rest_option(golden, f, a):
                    assert o_old == 0 and row == -1 and i == -1, (f, a, hp)
                    zero[(Verb(int(f.actions[a, ActionCol.VERB])).name, up)] += 1
    assert len(base) == 5 and all(zero[(v, up)] > 0 for v in ("DISCARD_POTION", "USE_POTION", "PROCEED")
                                  for up in (True, False)), (len(base), zero)


ROLLOUT = [list(REST_RULE), ["potion_r1:Elite,Boss", 0.05]]  # the rule set to roll out on N4-spear


def check_rollout(gd: Golden, frames: list[Frame]) -> Counter[str]:
    """The roll-out rule set through the Advisor on every row of ``frames``: a REST record only for rest-option rows,
    a potion_r1 judgement only +1 and only for USE_POTION rows in round 1 of Elite / Boss fights, never both."""
    adv = Advisor(gd.vocab, ROLLOUT, gd.catalog)
    n: Counter[str] = Counter()
    for f in frames:
        g = f.gvals
        r1 = g[G.KIND] == Kind.COMBAT and g[G.COMBAT_ROUND] == 1 and gd.room(f) in ("Elite", "Boss")
        for a in range(f.n_actions):
            verb = int(f.actions[a, ActionCol.VERB])
            row, _ = adv.candidate(f, (), a)
            judged = adv.judge(f, (), a, verb)
            assert row < 0 or is_rest_option(gd, f, a)
            assert judged == ((0, 1) if r1 and verb == Verb.USE_POTION else NO_JUDGE), (f, a, judged)
            assert row < 0 or judged == NO_JUDGE
            n["rest"] += row >= 0
            n["potion_r1"] += judged != NO_JUDGE
    return n


def test_rollout_rule_set_judges_only_rest_options_and_round1_potions(golden: Golden) -> None:
    n = check_rollout(golden, golden.frames)
    assert n["rest"] == 441 + 541 and n["potion_r1"] == THIN["potion_r1:Elite,Boss"], n


def as_room(gd: Golden, f: Frame, room: str) -> Frame:
    """``f`` with GLOBAL ROOM_TYPE set to ``room``."""
    out = copy.copy(f)
    g = list(f.gvals)
    g[G.ROOM_TYPE] = gd.vocab.id("enum:RoomType", room)
    out._gvals = tuple(g)
    return out


@pytest.fixture(scope="module")
def r1_frames(golden: Golden) -> list[Frame]:
    """Round-1 Monster COMBAT frames with a USE_POTION row, moved to an Elite and to a Boss room (the recorded games
    reach round 1 of those fights with a usable potion on 2 frames only)."""
    base = [f for f in golden.of(Kind.COMBAT) if f.gvals[G.COMBAT_ROUND] == 1 and golden.room(f) == "Monster"
            and (f.actions[:, ActionCol.VERB] == Verb.USE_POTION).any()]
    assert len(base) == 103
    return [as_room(golden, f, r) for r in ("Elite", "Boss") for f in base]


def test_potion_r1_on_injected_rooms(golden: Golden, r1_frames: list[Frame]) -> None:
    """potion_r1:Elite,Boss on 206 round-1 Elite / Boss frames with potions: judge == the definition, +1 only (USE_POTION
    rows, the choice set), choose agrees with judge, the roll-out rule set judges every potion row there."""
    spec = "potion_r1:Elite,Boss"
    (h,) = parse_habit(spec)
    ref = reference(golden, spec)
    rng = np.random.default_rng(7)
    plus = 0
    for f in r1_frames:
        for banned in ((), bans_of(f, rng)):
            for a in (a for a in range(f.n_actions) if a not in banned):
                got = h.judge(golden.names, f, f.gvals, a, banned)
                assert got == ref(f, a, banned) and got >= 0 and (not got or in_choice_set(golden, spec, f, a))
                plus += got
    c = check_choose_and_judge(golden, spec, r1_frames, rng)
    n = check_rollout(golden, r1_frames)
    assert plus > 600 and c["overrides"] > 1000 and n["potion_r1"] > 600 and n["rest"] == 0, (plus, c, n)


def as_act(f: Frame, act_index: int) -> Frame:
    """``f`` with GLOBAL ACT_INDEX set to ``act_index`` (0-based)."""
    out = copy.copy(f)
    g = list(f.gvals)
    g[G.ACT_INDEX] = act_index
    out._gvals = tuple(g)
    return out


def test_heal_late_default_act3_on_injected_frames(golden: Golden) -> None:
    """heal_late:<t> (act 3, the last row below the Boss node): the recorded games reach no act-3 rest site in that
    row, so every REST frame offering HEAL is moved to act 3 (ACT_INDEX 2): judge == the definition, only rest options
    graded, choose agrees with judge; the same frames in their own act are not judged; heal_late:<t>:0 (every act) is
    the same habit as heal_late:<t>:3 on the moved frames."""
    base = [f for f in golden.of(Kind.REST) if any(golden.option(f, a) == ("REST_OPTION", "HEAL")
                                                   for a in range(f.n_actions))]
    moved = [as_act(f, 2) for f in base]
    rng = np.random.default_rng(9)
    for spec in ("heal_late:1.0", "heal_late:1.0:3:3"):
        (h,) = parse_habit(spec)
        ref = reference(golden, spec)
        fired = 0
        for f in moved:
            for banned in ((), bans_of(f, rng)):
                for a in (a for a in range(f.n_actions) if a not in banned):
                    got = h.judge(golden.names, f, f.gvals, a, banned)
                    assert got == ref(f, a, banned) and (not got or in_choice_set(golden, spec, f, a))
                    fired += got != 0
        c = check_choose_and_judge(golden, spec, moved, rng)
        assert fired > 20 and c["overrides"] > 10, (spec, fired, c)
        assert all(h.judge(golden.names, f, f.gvals, a, ()) == 0 for f in base if f.gvals[G.ACT_INDEX] != 2
                   for a in range(f.n_actions))
    every, act3 = parse_habit("heal_late:1.0:0")[0], parse_habit("heal_late:1.0:3")[0]
    assert all(every.judge(golden.names, f, f.gvals, a, ()) == act3.judge(golden.names, f, f.gvals, a, ())
               for f in moved for a in range(f.n_actions))
    for bad in ("heal_late:1.5", "heal_late:0", "heal_late:0.9:3:0", "heal_late:0.9:-1", "heal_late:0.9:3:1:1"):
        with pytest.raises(AssertionError, match="heal_late|expected"):
            parse_habit(bad)


def test_potion_r1_choose(golden: Golden, r1_frames: list[Frame]) -> None:
    """potion_r1:Elite,Boss on the recorded COMBAT frames with a USE_POTION row and the injected round-1 Elite / Boss
    ones: in round 1 of an Elite / Boss fight a greedy non-potion action becomes the policy's best USE_POTION row
    (banned rows excluded), a greedy potion use is left alone; any other round or room: None. Combat rounds count
    from 1."""
    (h,) = parse_habit("potion_r1:Elite,Boss")
    n = golden.names
    combat = golden.of(Kind.COMBAT)
    assert min(f.gvals[G.COMBAT_ROUND] for f in combat) == 1
    rng = np.random.default_rng(6)
    c = Counter()
    for f in [f for f in combat if (f.actions[:, ActionCol.VERB] == Verb.USE_POTION).any()] + r1_frames:
        g = f.gvals
        r1 = g[G.COMBAT_ROUND] == 1 and golden.room(f) in ("Elite", "Boss")
        verbs = f.actions[:, ActionCol.VERB]
        banned = bans_of(f, rng)
        base = scores_of(f, banned, rng)
        use = [a for a in np.flatnonzero(verbs == Verb.USE_POTION).tolist() if a not in banned]
        for greedy in (a for a in range(f.n_actions) if a not in banned):
            s = base.copy()
            s[greedy] = base[np.isfinite(base)].max() + 1.0
            got = h.choose(n, f, g, s, greedy, Call(0, None))
            if not r1 or verbs[greedy] == Verb.USE_POTION or not use:
                assert got is None
            else:
                assert got == max(use, key=lambda a: s[a])
                c["forced"] += 1
        c["r1_frames" if r1 else "other_frames"] += 1
    assert c["r1_frames"] == 208 and c["forced"] > 1000 and c["other_frames"] > 100, c


# =========================================================================== card rewards: prefs file / list, deck_cap


def test_prefs_file_equals_the_inline_list(golden: Golden, tmp_path: Path) -> None:
    """``prefs:file:<json>`` (the screening's table format) and ``prefs:list:`` are one habit; the screening's tables
    resolve on the 4-seat vocab; unknown names fail on first use; malformed specs fail at parse."""
    _, _, lists = PREFS.split(":", 2)
    prefer, avoid = (x.split(",") for x in lists.split("/"))
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"prefer": prefer, "avoid": avoid, "note": "test"}), encoding="utf-8")
    (a,), (b,) = parse_habit(f"prefs:file:{path}"), parse_habit(PREFS)
    assert a == b and a.prefer == tuple(prefer) and a.avoid == tuple(avoid)
    rng = np.random.default_rng(4)
    for f in golden.of(Kind.CARD_REWARD):
        banned = bans_of(f, rng)
        assert all(a.judge(golden.names, f, f.gvals, x, banned) == b.judge(golden.names, f, f.gvals, x, banned)
                   for x in range(f.n_actions) if x not in banned)
    tables = sorted((ROOT / "alt" / "screen" / "tables").glob("T*.json"))
    assert len(tables) >= 3, "the screening's tables T1-T3 (alt/screen/tables under SPIRE_REPO)"
    for table in tables:
        (h,) = parse_habit(f"prefs:file:{table}")
        assert (h.values(golden.names) != 0).sum() == len(h.prefer) + len(h.avoid) > 0
    with pytest.raises(AssertionError, match="vocab card lacks NOT_A_CARD"):
        parse_habit("prefs:list:NOT_A_CARD/")[0].values(golden.names)
    for bad, msg in (("prefs:list:A/A", "both PREFER and AVOID"), ("prefs:list:/", "empty table"),
                     ("prefs:list:A,A/", "repeated card"), ("prefs:list:A", "expected file"),
                     ("prefs:rand:1:0.1", "expected file")):
        with pytest.raises(AssertionError, match=msg):
            parse_habit(bad)


def test_prefs_does_not_judge_the_sacrifice(golden: Golden) -> None:
    """The one recorded CARD_REWARD frame with a third alternative (SACRIFICE), made a PREFER frame by a table of its
    first offered card: +1 that pick, -1 the other picks and the Skip, 0 the SACRIFICE (not a pick / skip choice); the
    Advisor gives the same."""
    (f,) = [f for f in golden.of(Kind.CARD_REWARD) if any(golden.option(f, a) == ("CARD_REWARD_ALT", "SACRIFICE")
                                                         for a in range(f.n_actions))]
    picks = [a for a in range(f.n_actions) if golden.offered(f, a) is not None]
    assert len(picks) >= 2 and len({golden.offered(f, a) for a in picks}) == len(picks)
    spec = f"prefs:list:{golden.offered(f, picks[0])}/"
    (h,) = parse_habit(spec)
    adv = Advisor(golden.vocab, [[spec, 0.05]], golden.catalog)
    for a in range(f.n_actions):
        o = golden.option(f, a)
        want = 1 if a == picks[0] else -1 if a in picks or o == ("CARD_REWARD_ALT", "SKIP") else 0
        assert (o == ("CARD_REWARD_ALT", "SACRIFICE")) == (want == 0 and a not in picks)
        assert h.judge(golden.names, f, f.gvals, a, ()) == want
        assert adv.judge(f, (), a, int(f.actions[a, ActionCol.VERB])) == ((0, want) if want else NO_JUDGE)


def test_deck_cap_threshold_and_skip(golden: Golden) -> None:
    """deck_cap:<n> takes the Skip exactly at the CARD_REWARD frames of decks >= n (22: the round-2 arm value; the
    recorded decks reach 25), never without a legal Skip."""
    n = golden.names
    frames = golden.of(Kind.CARD_REWARD)
    for cap in (12, 22, 26):
        (h,) = parse_habit(f"deck_cap:{cap}")
        hit = 0
        for f in frames:
            skip = next(r for r in range(f.n_actions) if golden.option(f, r) == ("CARD_REWARD_ALT", "SKIP"))
            s = greedy_scores(f, 0)
            got = h.choose(n, f, f.gvals, s, 0, Call(0, None))
            assert got == (skip if golden.deck_size(f) >= cap else None)
            hit += got is not None
            s[skip] = -np.inf  # a banned Skip: nothing to do
            assert h.choose(n, f, f.gvals, s, 0, Call(0, None)) is None
            assert all(h.judge(n, f, f.gvals, a, (skip,)) == 0 for a in range(f.n_actions) if a != skip)
        assert hit == sum(golden.deck_size(f) >= cap for f in frames)
        assert hit > 100 if cap == 12 else hit == {22: 4, 26: 0}[cap], (cap, hit)


PAIR = f"{PREFS}+deck_cap:14"


def test_deck_cap_has_precedence_over_prefs_in_evaluation_and_advice(golden: Golden) -> None:
    """Both judge CARD_REWARD picks. Evaluation: a composite answers with deck_cap where it applies, in either spec
    order. Advice: the prefs rule is silent at those frames (Advisor, YIELDS), so every step has at most one judgement
    (without the precedence some steps would have two), and the Advisor mirrors the composite override."""
    vocab, n = golden.vocab, golden.names
    cap, prefs = parse_habit("deck_cap:14")[0], parse_habit(PREFS)[0]
    for spec in (PAIR, f"deck_cap:14+{PREFS}"):
        assert [type(h) for h in parse_habit(spec)] == [type(cap), type(prefs)]
    for rules in ([[PREFS, 0.05], ["deck_cap:14", 0.03]], [["deck_cap:14", 0.03], [PREFS, 0.05]]):
        adv = Advisor(vocab, rules, golden.catalog)
        i_cap, i_prefs = (0, 1) if rules[0][0] == "deck_cap:14" else (1, 0)
        rng = np.random.default_rng(5)
        both = shadowed = by_prefs = 0
        for f in golden.of(Kind.CARD_REWARD):
            g = f.gvals
            banned = bans_of(f, rng)
            applies = cap.applies(n, f, g, banned)
            for a in range(f.n_actions):
                if a in banned:
                    continue
                verb = int(f.actions[a, ActionCol.VERB])
                sc, sp = cap.judge(n, f, g, a, banned), prefs.judge(n, f, g, a, banned)
                both += bool(sc and sp)
                got = adv.judge(f, banned, a, verb)
                if applies:
                    assert got == ((i_cap, sc) if sc else NO_JUDGE)
                    shadowed += bool(sp)
                else:
                    assert sc == 0 and got == ((i_prefs, sp) if sp else NO_JUDGE)
                    by_prefs += bool(sp)
            # the composite override and the advice agree: where it overrides greedy, greedy -1 or chosen +1
            ov = HabitOverride(PAIR, vocab, golden.catalog)
            base = scores_of(f, banned, rng)
            for greedy in (a for a in range(f.n_actions) if a not in banned):
                s = base.copy()
                s[greedy] = base[np.isfinite(base)].max() + 1.0
                chosen = ov(0, f, None, s, banned)
                assert (ov.last[0][4] == "deck_cap") == (applies and chosen is not None)
                if chosen is not None and chosen != greedy:
                    j = {x: adv.judge(f, banned, x, int(f.actions[x, ActionCol.VERB])) for x in (greedy, chosen)}
                    assert j[greedy][1] == -1 or j[chosen][1] == 1, (f, greedy, chosen, j)
        assert both > 20 and shadowed > 20 and by_prefs > 20, (both, shadowed, by_prefs)


# =========================================================================== round-1 verifier's defects (verify.md)


def test_potion_hold_never_redirects_to_a_discard(golden: Golden) -> None:
    """Defect 1: the fallback of potion_hold / potion_boss is neither USE_POTION nor DISCARD_POTION."""
    n = golden.names
    frames = [f for f in golden.of(Kind.COMBAT) if (f.actions[:, ActionCol.VERB] == Verb.DISCARD_POTION).any()
              and (f.actions[:, ActionCol.VERB] == Verb.USE_POTION).any() and golden.room(f) == "Monster"]
    assert len(frames) > 20
    (h,) = parse_habit("potion_hold")
    for f in frames:
        verbs = f.actions[:, ActionCol.VERB]
        greedy = int(np.flatnonzero(verbs == Verb.USE_POTION)[0])
        s = np.zeros(f.n_actions, np.float32)
        s[verbs == Verb.DISCARD_POTION] = 5.0  # the policy's second choice is to discard
        s[greedy] = 10.0
        chosen = h.choose(n, f, f.gvals, s, greedy, Call(0, None))
        assert chosen is not None and verbs[chosen] not in (Verb.USE_POTION, Verb.DISCARD_POTION)


def removal_episodes(gd: Golden) -> list[tuple[Frame, int, list[Frame]]]:
    """(SHOP frame, its CARD_REMOVAL row, the DECK_REMOVAL prompt frames that the purchase opened) in recorded order."""
    out = []
    fr = gd.frames
    for i, f in enumerate(fr[:-1]):
        nxt = fr[i + 1]
        g = nxt.gvals
        if f.gvals[G.KIND] == Kind.SHOP and g[G.KIND] == Kind.CARD_SELECT and g[G.KIND_SUB] == CardSelectSub.DECK_REMOVAL \
                and gd.room(nxt) == "Shop" and g[G.SEL_COUNT] == 0:
            row = next(r for r in range(f.n_actions) if gd.option(f, r) == ("SHOP_ENTRY", "CARD_REMOVAL"))
            prompts = [nxt] + [p for p in fr[i + 2:i + 4] if p.gvals[G.KIND] == Kind.CARD_SELECT
                               and p.gvals[G.KIND_SUB] == CardSelectSub.DECK_REMOVAL and p.episode_id == f.episode_id]
            out.append((f, row, prompts))
    return out


def greedy_scores(f: Frame, row: int) -> np.ndarray:
    s = np.zeros(f.n_actions, np.float32)
    s[row] = 1.0
    return s


def test_removal_prompt_override_needs_the_purchase(golden: Golden) -> None:
    """Defect 2: remove_basic's prompt part (and the recorder's ``removal_selected:*``) acts only on the DECK_REMOVAL
    prompt opened by buying the card removal, not on every Shop-room removal prompt."""
    eps = removal_episodes(golden)
    assert len(eps) >= 10
    for shop, row, prompts in eps:
        prompt = prompts[0]
        worst = 0  # the policy's greedy SELECT_CARD: the first selectable row
        sel = [r for r in range(prompt.n_actions) if prompt.actions[r, ActionCol.VERB] == Verb.SELECT_CARD]
        ov = HabitOverride("remove_basic", golden.vocab, golden.catalog)
        # without the purchase (the prompt alone, e.g. a relic's): the policy's prompt
        assert ov(7, prompt, None, greedy_scores(prompt, sel[worst]), ()) is None
        # after the purchase (by the policy: the habit is silent at the shop frame only when it buys itself)
        ov = HabitOverride("remove_basic", golden.vocab, golden.catalog)
        got = ov(7, shop, None, greedy_scores(shop, row), ())
        assert got in (None, row)
        assert ov.removal_seat == {7: shop.gvals[G.ACTING_SEAT]}
        cls = golden.names.card_class[prompt.table(Table.CARDS)[prompt.actions[sel, ActionCol.SRC_ROW], CardCol.CARD_ID]]
        chosen = ov(7, prompt, None, greedy_scores(prompt, sel[worst]), ())
        if (cls > 0).any():
            best = min(int(c) for c in cls if c > 0)
            assert chosen is not None and golden.names.card_class[
                prompt.table(Table.CARDS)[prompt.actions[chosen, ActionCol.SRC_ROW], CardCol.CARD_ID]] == best
        else:
            assert chosen is None
        # another frame of the env closes the prompt: the next removal prompt is the policy's again
        m = golden.of(Kind.MAP)[0]
        ov(7, m, None, greedy_scores(m, 0), ())
        assert 7 not in ov.removal_seat and ov(7, prompt, None, greedy_scores(prompt, sel[worst]), ()) is None


def feed(rec: HabitRecorder, ov: HabitOverride, env: int, f: Frame, action: int) -> None:
    ov(env, f, None, greedy_scores(f, action), ())
    rec.observe(env, f, action)


def test_recorder_counts_bought_removals_only(golden: Golden) -> None:
    shop, row, prompts = removal_episodes(golden)[0]
    prompt = prompts[0]
    sel = next(r for r in range(prompt.n_actions) if prompt.actions[r, ActionCol.VERB] == Verb.SELECT_CARD)
    ov = HabitOverride("baseline", golden.vocab, golden.catalog)
    rec = HabitRecorder(ov)
    feed(rec, ov, 0, prompt, sel)  # a removal prompt nobody bought
    assert not any(k.startswith("removal_selected:") for k in rec._open[0].c)
    assert sum(v for k, v in rec._open[0].c.items() if k.startswith("removal_selected_unbought:")) == 1
    feed(rec, ov, 0, shop, row)
    feed(rec, ov, 0, prompt, sel)
    c = rec._open[0].c
    assert c["removal_bought"] == 1 and sum(v for k, v in c.items() if k.startswith("removal_selected:")) == 1


def test_elite_with_alternative_counts_the_travel_into_the_elite(golden: Golden) -> None:
    """Defect 3: ``elites_entered_with_alternative_*`` reads the TRAVEL that led into the elite (an Elite node chosen
    while a non-Elite one was offered), not merely the last MAP decision."""
    gd = golden

    def travel(f: Frame, elite: bool) -> list[int]:
        return [r for r in range(f.n_actions) if f.actions[r, ActionCol.VERB] == Verb.TRAVEL
                and (gd.point(f, r) == "Elite") == elite]

    found = None  # one game: a MAP frame offering an Elite and a non-Elite node, two elite fights of the same act
    for ep in sorted({f.episode_id for f in gd.frames}):
        frames = [f for f in gd.frames if f.episode_id == ep]
        for m in (f for f in frames if f.gvals[G.KIND] == Kind.MAP and travel(f, True) and travel(f, False)):
            fights: dict[tuple[int, int], Frame] = {}
            for f in frames:
                if f.gvals[G.KIND] == Kind.COMBAT and gd.room(f) == "Elite" and f.gvals[G.ACT_INDEX] == m.gvals[G.ACT_INDEX]:
                    fights.setdefault((f.gvals[G.TOTAL_FLOOR], f.gvals[G.COMBATS_WON]), f)
            if len(fights) >= 2:
                found = m, list(fights.values())[:2]
                break
        if found:
            break
    assert found is not None
    m, (e1, e2) = found
    act = m.gvals[G.ACT_INDEX] + 1
    elite_rows, other_rows = travel(m, True), travel(m, False)
    ov = HabitOverride("baseline", gd.vocab, gd.catalog)
    rec = HabitRecorder(ov)
    feed(rec, ov, 0, m, other_rows[0])  # chose a non-Elite node ...
    feed(rec, ov, 0, e1, 0)  # ... and still ended in an elite fight (e.g. an unknown room): no elite "with alternative"
    feed(rec, ov, 0, m, elite_rows[0])
    feed(rec, ov, 0, e2, 0)
    c = rec._open[0].c
    assert c[f"elites_entered_act{act}"] == 2 and c[f"elites_entered_with_alternative_act{act}"] == 1


# =========================================================================== the actor's evaluation-only hook


class _Force:
    """Override: every other decision the lowest legal row that is not the greedy one; remembers what it did."""

    def __init__(self) -> None:
        self.forced: dict[tuple[int, int, int], int] = {}
        self.calls = 0

    def __call__(self, env_id: int, f: Frame, sample: Any, scores: np.ndarray, banned: Any) -> int | None:
        self.calls += 1
        greedy = int(np.argmax(scores))
        ok = [a for a in np.flatnonzero(np.isfinite(scores)).tolist() if a != greedy]
        if self.calls % 2 or not ok:
            return None
        self.forced[(env_id, f.episode_id, f.step_id)] = ok[0]
        return ok[0]


def test_override_hook_sends_the_override_and_is_evaluation_only(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, commit_select=True))
    torch.manual_seed(0)
    pol = TorchPolicy(SpireNet(tok.layout, TINY), tok, mode="greedy", use_graphs=False)
    force = _Force()
    seen: list[tuple[int, Frame, int]] = []
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=80, hashes=False, abort_prob=0.0))
             for i in range(3)]
    try:
        Actor(hosts, pol, tok, EpisodeSampler(EpisodeDist(lineups=(("IRONCLAD+SILENT", 1.0),)), vocab, 0),
              ActorConfig(mode="eval"), on_decision=lambda e, f, a: seen.append((e, f, a)), override=force).run(
            max_decisions=600)
    finally:
        pol.close()
    hits = [(e, f, a) for e, f, a in seen if (e, f.episode_id, f.step_id) in force.forced]
    assert len(hits) > 100 and all(a == force.forced[(e, f.episode_id, f.step_id)] for e, f, a in hits)
    store = RolloutStore(lambda s: None, RoomIds.from_vocab(vocab))
    with pytest.raises(AssertionError, match="evaluation-only"):
        Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0), ActorConfig(), store=store,
              override=force)


# =========================================================================== event page table (r3 events, notes/gym)

CHEESE = "explore:page:ROOM_FULL_OF_CHEESE.INITIAL=ROOM_FULL_OF_CHEESE.pages.INITIAL.options.SEARCH"  # r3 events' one


def page_of(gd: Golden, f: Frame) -> str:
    """Short name of an EVENT frame's page (``X.pages.P.description`` -> ``X.P``)."""
    return gd.vocab.name("event_page", int(f.gvals[G.EVENT_PAGE_KEY])).replace(".pages.", ".").removesuffix(".description")


def event_option(gd: Golden, f: Frame, row: int) -> str | None:
    """Option key vocab name of a non-relic EVENT_OPTION row, else None."""
    o = gd.option(f, row)
    if o is None or o[0] != "EVENT_OPTION":
        return None
    key = gd.vocab.name("event_option_key", int(f.table(Table.OPTIONS)[f.actions[row, ActionCol.SRC_ROW], OptionCol.KEY_ID]))
    return None if key == "RELIC_OPTION" else key


@pytest.fixture(scope="module")
def event_pages(golden: Golden) -> dict[str, list[str]]:
    """Recorded event pages with a frame of >= 2 distinct non-relic option keys -> those keys (sorted)."""
    out: dict[str, set[str]] = {}
    for f in golden.of(Kind.EVENT):
        keys = {k for k in (event_option(golden, f, a) for a in range(f.n_actions)) if k}
        if len(keys) >= 2:
            out.setdefault(page_of(golden, f), set()).update(keys)
    return {p: sorted(k) for p, k in sorted(out.items())}


def page_reference(gd: Golden, table: dict[str, str]) -> Callable[[Frame, int, tuple[int, ...]], int]:
    """The page table's judgement written out from its definition (spire.habit.PageTable), row by row."""

    def ref(f: Frame, action: int, banned: tuple[int, ...]) -> int:
        if f.gvals[G.KIND] != Kind.EVENT or page_of(gd, f) not in table:
            return 0
        opts = {r: k for r in range(f.n_actions) if r not in banned and (k := event_option(gd, f, r))}
        if action not in opts or len(set(opts.values())) < 2 or table[page_of(gd, f)] not in opts.values():
            return 0
        return 1 if opts[action] == table[page_of(gd, f)] else -1
    return ref


JOB = EpisodeJob((1, 1), "NECROBINDER+NECROBINDER", 10, "ABCDEFGHIJKL")


def check_choose_and_judge_started(gd: Golden, spec: str, frames: list[Frame],
                                   rng: np.random.Generator) -> dict[str, int]:
    """check_choose_and_judge for a habit whose choose() needs the game's table (``HabitOverride.on_job``)."""
    ov = HabitOverride(spec, gd.vocab, gd.catalog)
    ov.on_job(0, JOB)
    (h,) = ov.habits
    n = ov.names
    c = {"frames": 0, "overrides": 0}
    for f in frames:
        g = f.gvals
        banned = bans_of(f, rng)
        base = scores_of(f, banned, rng)
        ok = [a for a in range(f.n_actions) if a not in banned]
        judge = {a: h.judge(n, f, g, a, banned) for a in ok}
        any_choice = False
        for greedy in ok:
            s = base.copy()
            s[greedy] = base[ok].max() + 1.0
            chosen = h.choose(n, f, g, s, greedy, Call(0, None))
            if chosen is None:
                continue
            any_choice = True
            assert chosen not in banned and np.isfinite(s[chosen]), (spec, chosen, banned)
            if chosen != greedy:
                c["overrides"] += 1
                assert judge[greedy] == -1 or judge[chosen] == 1, (spec, f.episode_id, f.step_id, greedy, chosen)
            else:
                assert judge[chosen] == 1, (spec, f.episode_id, f.step_id, chosen)
        if not any_choice:
            assert not any(judge.values()), (spec, f.episode_id, f.step_id, judge)
        c["frames"] += 1
    return c


def test_page_table_judge_equals_the_definition_and_grades_only_its_pages(golden: Golden, tmp_path: Path,
                                                                          event_pages: dict[str, list[str]]) -> None:
    """Single-page tables (first and last option of every recorded page; each page's own frames and ~300 other EVENT
    frames) and one table of all recorded pages (every EVENT frame), with and without a random ban mask: judge equals
    the definition on every row; +1 only for the listed option, -1 only for another non-relic option of that page
    (PROCEED, potion actions, relic options 0); choose() agrees with judge()."""
    assert len(event_pages) > 20, len(event_pages)
    frames = golden.of(Kind.EVENT)
    rng = np.random.default_rng(8)
    path = tmp_path / "pages.json"
    every = {p: k[0] for p, k in event_pages.items()}
    path.write_text(json.dumps({"pages": every, "note": "test"}), encoding="utf-8")
    tables = [(f"explore:file:{path}", every, frames)]
    for p, keys in event_pages.items():
        own = [f for f in frames if page_of(golden, f) == p]
        for k in (keys[0], keys[-1]):
            tables.append((f"explore:page:{p}={k}", {p: k}, own + frames[::max(1, len(frames) // 300)]))
    c: Counter[str] = Counter()
    for spec, table, pool in tables:
        (h,) = parse_habit(spec)
        assert isinstance(h, PageTable) and h.pages == table and h.KINDS == {Kind.EVENT}
        ref = page_reference(golden, table)
        for f in pool:
            for banned in ((), bans_of(f, rng)):
                want = {a: ref(f, a, banned) for a in range(f.n_actions) if a not in banned}
                for a, w in want.items():
                    got = h.judge(golden.names, f, f.gvals, a, banned)
                    assert got == w, (spec, f.episode_id, f.step_id, a, banned)
                    assert not got or (in_choice_set(golden, spec, f, a) and int(f.actions[a, ActionCol.VERB]) in h.VERBS)
                    c["plus" if got > 0 else "minus" if got < 0 else "zero"] += 1
                    # PROCEED, potion and relic rows at a judged frame
                    c["zero_beside"] += any(want.values()) and event_option(golden, f, a) is None
    assert c["plus"] > 200 and c["minus"] > 200 and c["zero_beside"] > 20, c
    cc = check_choose_and_judge_started(golden, f"explore:file:{path}", frames, rng)
    assert cc["overrides"] > 50, cc


def test_page_table_specs_and_the_advice_start_up(golden: Golden, tmp_path: Path,
                                                  event_pages: dict[str, list[str]]) -> None:
    """The inline single page equals the file table of that page; r3 events' cheese rule resolves on the 4-seat vocab
    and shares a config with the REST rule and potion_r1 (EVENT vs REST / COMBAT); an unknown page or option fails at
    the Advisor's start-up; malformed inline specs fail at parse; a table with relics is evaluation-only (Explore)."""
    p, keys = next(iter(event_pages.items()))
    path = tmp_path / "one.json"
    path.write_text(json.dumps({"pages": {p: keys[0]}}), encoding="utf-8")
    assert parse_habit(f"explore:file:{path}") == parse_habit(f"explore:page:{p}={keys[0]}")
    (cheese,) = parse_habit(CHEESE)
    xg = cheese.resolve(golden.names)
    assert xg.page_forced.sum() == 1 and set(xg.page_arm.values()) == {
        golden.vocab.id("event_option_key", "ROOM_FULL_OF_CHEESE.pages.INITIAL.options.SEARCH")}
    adv = Advisor(golden.vocab, [list(REST_RULE), ["potion_r1:Elite,Boss", 0.1], [CHEESE, 0.05]], golden.catalog)
    assert [h.spec for h in adv.habits] == ["potion_r1:Elite,Boss", CHEESE]
    for bad, msg in ((f"explore:page:NOT_A_PAGE.INITIAL={keys[0]}", "event_page lacks NOT_A_PAGE"),
                     (f"explore:page:{p}=NOT_AN_OPTION", "event_option_key lacks NOT_AN_OPTION")):
        with pytest.raises(AssertionError, match=msg):
            Advisor(golden.vocab, [[bad, 0.05]], golden.catalog)
    for bad, msg in (("explore:page:ROOM_FULL_OF_CHEESE.INITIAL", "expected page"), ("explore:page:=X", "expected page"),
                     ("explore:pages:X=Y", "expected rand")):
        with pytest.raises(AssertionError, match=msg):
            parse_habit(bad)
    relic = tmp_path / "relic.json"
    relic.write_text(json.dumps({"pages": {p: keys[0]}, "prefer": ["FUR_COAT"]}), encoding="utf-8")
    with pytest.raises(AssertionError, match="Explore is evaluation-only"):
        Advisor(golden.vocab, [[f"explore:file:{relic}", 0.05]], golden.catalog)


def test_explore_table_starts_per_game_and_is_recorded(golden: Golden, event_pages: dict[str, list[str]]) -> None:
    """``evaluate(on_job=)`` (_NoticeJobs) tells the override every job an env starts; the explore habit makes that
    game's table, a listed page frame is overridden to the listed option, and the recorder's game carries the table."""
    p, keys = next(iter(event_pages.items()))
    ov = HabitOverride(f"explore:page:{p}={keys[0]}", golden.vocab, golden.catalog)
    jobs = _NoticeJobs([JOB], ov.on_job)
    assert jobs.next_job(3) == JOB and jobs.next_job(3) is None
    assert ov.explore is not None and set(ov.explore.games) == {3} and ov.explore.games[3].seed == JOB.seed
    f = next(f for f in golden.of(Kind.EVENT) if page_of(golden, f) == p
             and len({event_option(golden, f, a) for a in range(f.n_actions)} - {None}) >= 2
             and any(event_option(golden, f, a) == keys[0] for a in range(f.n_actions)))
    other = next(a for a in range(f.n_actions) if event_option(golden, f, a) not in (None, keys[0]))
    chosen = ov(3, f, None, greedy_scores(f, other), ())
    assert chosen is not None and event_option(golden, f, chosen) == keys[0] and ov.last[3][4] == "explore"
    rec = HabitRecorder(ov)
    rec.observe(3, f, chosen)
    g = rec._open[3]
    assert g.explore is ov.explore.games[3] and g.explore.pages[p][:3] == [keys[0].rpartition(".options.")[2], 1, 1]
    assert g.c["changed:explore"] == 1 and g.c["changed:explore:out"] == 1
