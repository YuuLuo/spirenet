"""Tactical combat habits (spire.habit, notes/r3/tactics.md): judge() equals a row-by-row reference written from each
definition (names, not ids), grades only the rows of the habit's own choice set (notes/advfix), choose() returns legal,
not banned rows and agrees with judge() (the pattern of test_habit.py), on the recorded 4-seat golden frames and on
real NECROBINDER x4 ascension-10 frames captured on the measurement box (bench/r3/tactics/frames_tac.bin: COMBAT frames
with a legal potion use in act 3 or with a seat facing lethal at factor 0.5); the advice start-up accepts the training
forms and refuses the pairs that can judge one step (notes/gym/merge.md)."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from spire.advice import NO_JUDGE, Advisor, parse_advice
from spire.habit import Call, parse_habit
from spire.proto import Frame, parse_frame
from spire.schema import (ActionCol, CardCol, CreatureCol, CreatureKind, G, IntentCol, Kind, PotionCol, SeatCol, Table,
                          VarCol, Verb)
from test_advice import ROOT
from test_habit import REST_RULE, Golden, bans_of, in_choice_set, scores_of

CAPTURED = ROOT / "bench" / "r3" / "tactics" / "frames_tac.bin"
SPECS = ("lethal_block", "lethal_block:0.5", "kill_dangerous", "weak_attacker", "potion_reserve2", "potion_reserve2:all",
         "potion_lethal", "lethal_eot", "potion_boss_use", "potion_boss_use:all", "potion_boss_use:late",
         "potion_boss_take", "weak_attacker:cards")
GUARD = ("BLOCK_POTION", "SHIP_IN_A_BOTTLE", "BONE_BREW", "HEART_OF_IRON", "GHOST_IN_A_JAR", "LUCKY_TONIC", "FORTIFIER")
HEAL = ("BLOOD_POTION", "AMBERGRIS", "REGEN_POTION", "FRUIT_JUICE")


def read_frames(path: Path) -> list[Frame]:
    out = []
    with path.open("rb") as fh:
        while head := fh.read(4):
            (size,) = struct.unpack("<I", head)
            out.append(parse_frame(fh.read(size)))
    return out


class Frames:
    def __init__(self, gd: Golden) -> None:
        if not CAPTURED.exists():
            pytest.skip(f"captured frames {CAPTURED} not present")
        self.gd = gd
        self.captured = read_frames(CAPTURED)
        assert len(self.captured) > 250 and all(f.vocab_hash == gd.vocab.hash for f in self.captured[:20])
        self.combat = gd.of(Kind.COMBAT)[::10] + self.captured


@pytest.fixture(scope="module")
def golden() -> Golden:
    return Golden()


@pytest.fixture(scope="module")
def frames(golden: Golden) -> Frames:
    return Frames(golden)


# =========================================================================== the definitions, row by row


class Ref:
    """The definitions of notes/r3/tactics.md written out with names."""

    def __init__(self, gd: Golden, f: Frame) -> None:
        self.gd, self.f = gd, f
        v = gd.vocab
        seats, cr, it = f.table(Table.SEATS), f.table(Table.CREATURES), f.table(Table.INTENTS)
        n = len(seats)
        self.n = n
        self.hp = [int(s[SeatCol.HP]) for s in seats]
        self.block = [int(s[SeatCol.BLOCK]) for s in seats]
        self.alive = [bool(s[SeatCol.ALIVE]) and s[SeatCol.HP] > 0 for s in seats]
        self.incoming = [0] * n
        self.on = {}  # enemy creature row -> per seat damage
        for x in it:
            e = int(x[IntentCol.OWNER_ROW])
            self.on.setdefault(e, [0] * n)
            for s in range(n):
                d = int(x[IntentCol.DMG_SEAT0 + s])
                if d >= 0:
                    self.incoming[s] += d * max(int(x[IntentCol.HITS]), 1)
                    self.on[e][s] += d * max(int(x[IntentCol.HITS]), 1)
        osty = [0] * n
        for c in cr:
            if c[CreatureCol.KIND] == CreatureKind.PET and v.name("monster", int(c[CreatureCol.MODEL_ID])) == "OSTY" \
                    and c[CreatureCol.ALIVE]:
                osty[c[CreatureCol.SEAT]] += max(int(c[CreatureCol.HP]), 0)
        self.reach = [max(0, self.incoming[s] - self.block[s] - osty[s]) for s in range(n)]
        self.cr = cr
        # end-of-turn damage of hand cards with has_turn_end_in_hand_effect: Damage (blockable), HpLoss (not)
        dmg, loss = [0] * n, [0] * n
        for row, c in enumerate(f.table(Table.CARDS)):
            if c[CardCol.ZONE] != 1:  # HAND
                continue
            e = gd.catalog.entries["card"][int(c[CardCol.CARD_ID])][0]
            if e["has_turn_end_in_hand_effect"]:
                dmg[c[CardCol.SEAT]] += self.shown(Table.CARDS, row, "Damage", -1) or 0
                loss[c[CardCol.SEAT]] += self.shown(Table.CARDS, row, "HpLoss", -1) or 0
        self.reach_eot = [loss[s] + max(0, dmg[s] - self.block[s])
                          + max(0, self.incoming[s] - max(0, self.block[s] - dmg[s]) - osty[s]) for s in range(n)]

    def lethal(self, s: int, fac: float, eot: bool = False) -> bool:
        return self.alive[s] and self.incoming[s] > 0 and (self.reach_eot if eot else self.reach)[s] >= fac * self.hp[s]

    def danger(self, e: int) -> int:
        return sum(d for s, d in enumerate(self.on.get(e, [0] * self.n)) if self.alive[s])

    def seat_of(self, row: int) -> int | None:
        c = self.cr[row]
        return None if c[CreatureCol.KIND] == CreatureKind.ENEMY else int(c[CreatureCol.SEAT])

    def shown(self, table: int, row: int, var: str, target: int) -> int | None:
        va = [r for r in self.f.table(Table.VARS) if r[VarCol.OWNER_TABLE] == table and r[VarCol.OWNER_ROW] == row
              and self.gd.vocab.name("var_name", int(r[VarCol.VAR_ID])) == var]
        t = [r for r in va if r[VarCol.TARGET_ROW] == target] or [r for r in va if r[VarCol.TARGET_ROW] == -1]
        return int(t[0][VarCol.SHOWN]) if t else None

    def card(self, row: int) -> tuple[str, dict[str, Any]]:
        c = self.f.table(Table.CARDS)[self.f.actions[row, ActionCol.SRC_ROW]]
        levels = self.gd.catalog.entries["card"][int(c[CardCol.CARD_ID])]
        return self.gd.vocab.name("card", int(c[CardCol.CARD_ID])), levels[min(int(c[CardCol.UPGRADE_LEVEL]), len(levels) - 1)]

    def potion(self, row: int) -> str:
        p = self.f.table(Table.POTIONS)[self.f.actions[row, ActionCol.SRC_ROW]]
        return self.gd.vocab.name("potion", int(p[PotionCol.POTION_ID]))

    def guards(self, row: int) -> set[int]:
        """Seats a row adds defence to this turn (guard_rows' definition)."""
        a = self.f.actions[row]
        tgt = int(a[ActionCol.TGT_ROW])
        if a[ActionCol.VERB] == Verb.PLAY_CARD:
            name, e = self.card(row)
            if not (e["gains_block"] or ("Summon" in e["vars"] and name != "INVOKE")):
                return set()
            vals = [x for x in (self.shown(Table.CARDS, int(a[ActionCol.SRC_ROW]), k, tgt)
                                for k in ("Block", "CalculatedBlock", "Summon")) if x is not None]
            if vals and sum(vals) <= 0:
                return set()
            if name in ("LIFT", "INTERCEPT"):
                s = self.seat_of(tgt) if tgt >= 0 else None
                return set() if s is None else {s}
            if e["target_type"] == "AllAllies":
                return set(range(self.n))
            return {int(a[ActionCol.SEAT])}
        if a[ActionCol.VERB] == Verb.USE_POTION and self.potion(row) in GUARD and tgt >= 0:
            s = self.seat_of(tgt)
            if s is None or (self.potion(row) == "FORTIFIER" and self.block[s] <= 0):
                return set()
            return {s}
        return set()

    def saves(self, row: int, s: int) -> bool:
        a = self.f.actions[row]
        if a[ActionCol.VERB] != Verb.USE_POTION or a[ActionCol.SEAT] != s:
            return False
        p, tgt = self.potion(row), int(a[ActionCol.TGT_ROW])
        if p in GUARD or p in HEAL:
            return tgt >= 0 and self.seat_of(tgt) == s and (p != "FORTIFIER" or self.block[s] > 0)
        if p in ("WEAK_POTION", "BEETLE_JUICE"):
            return tgt >= 0 and self.seat_of(tgt) is None and self.on.get(tgt, [0] * self.n)[s] > 0
        if p in ("SHACKLING_POTION", "POTION_OF_BINDING"):
            return any(d[s] > 0 for d in self.on.values())
        return False

    def kills(self, row: int) -> bool:
        a = self.f.actions[row]
        tgt, src = int(a[ActionCol.TGT_ROW]), int(a[ActionCol.SRC_ROW])
        dmg = [x for x in (self.shown(Table.CARDS, src, k, tgt) for k in ("Damage", "CalculatedDamage", "OstyDamage"))
               if x is not None]
        hits = [x for x in (self.shown(Table.CARDS, src, k, tgt) for k in ("CalculatedHits", "HitCount")) if x is not None]
        c = self.cr[tgt]
        return max(dmg, default=0) * (hits[0] if hits else 1) >= c[CreatureCol.HP] + c[CreatureCol.BLOCK]

    def siblings(self, row: int, allowed: list[int]) -> list[int]:
        a = self.f.actions
        return [r for r in allowed if all(a[r, k] == a[row, k] for k in (ActionCol.VERB, ActionCol.SEAT, ActionCol.SRC_ROW))
                and a[r, ActionCol.TGT_ROW] >= 0 and self.seat_of(int(a[r, ActionCol.TGT_ROW])) is None]

    def aimed(self, rows: list[int]) -> list[int] | None:
        tg = [int(self.f.actions[r, ActionCol.TGT_ROW]) for r in rows]
        if len(set(tg)) < 2:
            return None
        d = [self.danger(t) for t in tg]
        if max(d) <= 0 or min(d) == max(d):
            return None
        return [r for r, x in zip(rows, d) if x == max(d)]


def reference(gd: Golden, spec: str) -> Callable[[Frame, int, tuple[int, ...]], int]:
    name, _, arg = spec.partition(":")
    memo: dict[Any, Any] = {}  # per frame (and ban mask): the definition's sets, computed once

    def ref(f: Frame) -> Ref:
        key = ("ref", id(f))
        if key not in memo:
            memo[key] = Ref(gd, f)
        return memo[key]

    def allowed(f: Frame, banned: tuple[int, ...]) -> list[int]:
        return [r for r in range(f.n_actions) if r not in banned]

    def combat(f: Frame) -> bool:
        return f.gvals[G.KIND] == Kind.COMBAT and len(f.table(Table.INTENTS)) > 0

    if name in ("lethal_block", "lethal_eot"):
        fac = float(arg) if arg else 1.0
        eot = name == "lethal_eot"

        def lethal_block(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            if not combat(f):
                return 0
            key = ("guard", id(f), banned)
            if key not in memo:
                r = ref(f)
                need = {s for s in range(r.n) if r.lethal(s, fac, eot)}
                memo[key] = [x for x in allowed(f, banned) if r.guards(x) & need]
            rows = memo[key]
            if not rows:
                return 0
            # -1 only for what the seats do with their turn instead: another card play or END_TURN (notes/advfix)
            return 1 if action in rows else -1 if f.actions[action, ActionCol.VERB] in (Verb.PLAY_CARD, Verb.END_TURN)                 else 0
        return lethal_block
    if name in ("kill_dangerous", "weak_attacker"):
        def aim(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            a = f.actions[action]
            if not combat(f) or a[ActionCol.TGT_ROW] < 0:
                return 0
            r = ref(f)
            if name == "kill_dangerous":
                if a[ActionCol.VERB] != Verb.PLAY_CARD:
                    return 0
                e = gd.catalog.entries["card"][int(f.table(Table.CARDS)[a[ActionCol.SRC_ROW], CardCol.CARD_ID])][0]
                if not (e["type"] == "Attack" and e["target_type"] == "AnyEnemy"):
                    return 0
            elif a[ActionCol.VERB] == Verb.PLAY_CARD:
                e = gd.catalog.entries["card"][int(f.table(Table.CARDS)[a[ActionCol.SRC_ROW], CardCol.CARD_ID])][0]
                if not ("WeakPower" in e["vars"] and e["target_type"] == "AnyEnemy"):
                    return 0
            elif a[ActionCol.VERB] != Verb.USE_POTION or r.potion(action) != "WEAK_POTION" or arg == "cards":
                return 0
            rows = r.siblings(action, allowed(f, banned))
            best = r.aimed(rows)
            if best is None:
                return 0
            if name == "kill_dangerous":
                kill = {x for x in rows if r.kills(x)}
                if kill & set(best):
                    best = [x for x in best if x in kill]
                elif kill:
                    return 0
            return 1 if action in best else -1
        return aim
    if name == "potion_reserve2":
        def reserve(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            g = f.gvals
            if g[G.KIND] != Kind.COMBAT or f.actions[action, ActionCol.VERB] != Verb.USE_POTION or g[G.ACT_INDEX] != 2 \
                    or g[G.SECOND_BOSS_ENCOUNTER] <= 1:
                return 0
            room = gd.room(f)
            if room == "Boss":
                r = ref(f)
                return -1 if g[G.BOSSES_WON] == 2 and not any(r.lethal(s, 1.0) for s in range(r.n)) else 0
            return -1 if arg == "all" and room in ("Monster", "Elite") else 0
        return reserve
    if name in ("potion_boss_use", "potion_boss_take"):
        def boss_use(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            g = f.gvals
            a = f.actions[action]
            if g[G.KIND] != Kind.COMBAT or gd.room(f) != "Boss" or (arg != "all" and g[G.ACT_INDEX] != 2):
                return 0
            if arg == "late" and g[G.BOSSES_WON] != 3:
                r = ref(f)
                if not (len(f.table(Table.INTENTS)) and any(r.lethal(x, 1.0) for x in range(r.n))):
                    return 0
            if a[ActionCol.VERB] not in (Verb.PLAY_CARD, Verb.END_TURN, Verb.USE_POTION):
                return 0
            s = int(a[ActionCol.SEAT])
            if f.table(Table.SEATS)[s, SeatCol.CARDS_PLAYED_TURN] != 0:
                return 0
            if not any(f.actions[x, ActionCol.VERB] == Verb.USE_POTION and f.actions[x, ActionCol.SEAT] == s
                       for x in allowed(f, banned)):
                return 0
            return 1 if a[ActionCol.VERB] == Verb.USE_POTION else -1 if name == "potion_boss_use" else 0
        return boss_use
    if name == "potion_lethal":
        def potion_lethal(f: Frame, action: int, banned: tuple[int, ...]) -> int:
            a = f.actions[action]
            if not combat(f) or a[ActionCol.VERB] not in (Verb.END_TURN, Verb.USE_POTION):
                return 0
            s = int(a[ActionCol.SEAT])
            ok = allowed(f, banned)
            if not any(f.actions[x, ActionCol.VERB] == Verb.END_TURN and f.actions[x, ActionCol.SEAT] == s for x in ok):
                return 0
            r = ref(f)
            if not r.lethal(s, 1.0):
                return 0
            rows = [x for x in ok if r.saves(x, s)]
            if not rows:
                return 0
            return -1 if a[ActionCol.VERB] == Verb.END_TURN else 1 if action in rows else 0
        return potion_lethal
    raise AssertionError(spec)


# =========================================================================== tests


@pytest.mark.parametrize("spec", SPECS)
def test_judge_equals_the_definition(golden: Golden, frames: Frames, spec: str) -> None:
    (h,) = parse_habit(spec)
    ref = reference(golden, spec)
    rng = np.random.default_rng(11)
    fired = {"golden": 0, "captured": 0}
    pool = frames.combat if spec.split(":")[0] in ("lethal_block", "lethal_eot", "potion_lethal", "potion_reserve2",
                                                   "potion_boss_use", "potion_boss_take") else frames.combat[::3]
    captured = {id(f) for f in frames.captured}
    for f in pool:
        for banned in ((), bans_of(f, rng)):
            for a in range(f.n_actions):
                if a in banned:
                    continue
                got = h.judge(golden.names, f, f.gvals, a, banned)
                assert got == ref(f, a, banned), (spec, f.episode_id, f.step_id, a, banned, got)
                fired["captured" if id(f) in captured else "golden"] += got != 0
                assert not got or h.VERBS is None or int(f.actions[a, ActionCol.VERB]) in h.VERBS
    assert fired["captured"] + fired["golden"] > 20 and fired["captured"] > 5, fired
    for f in golden.frames[::50]:  # never outside COMBAT
        if f.gvals[G.KIND] != Kind.COMBAT:
            assert all(h.judge(golden.names, f, f.gvals, a, ()) == 0 for a in range(f.n_actions))


def check_choose_and_judge(gd: Golden, spec: str, frames: list[Frame], rng: np.random.Generator) -> dict[str, int]:
    """test_habit.check_choose_and_judge for the tactical habits (their potion habits may choose a USE_POTION row; only
    potion_reserve2 must not): for every frame, a random ban mask and every possible greedy action, choose() returns
    None or a legal, not banned, finite-score row; where it overrides the greedy action, the greedy action is judged -1
    or the chosen row +1; where it returns the greedy action itself, that action is judged +1; where it returns None
    for every greedy action, every action is judged 0."""
    (h,) = parse_habit(spec)
    n = gd.names
    c = {"frames": 0, "overrides": 0, "silent": 0}
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
            if spec.startswith("potion_reserve2"):
                assert f.actions[chosen, ActionCol.VERB] not in (Verb.USE_POTION, Verb.DISCARD_POTION)
            if chosen != greedy:
                c["overrides"] += 1
                assert judge[greedy] == -1 or judge[chosen] == 1, (spec, f, greedy, chosen, judge)
            else:
                assert judge[chosen] == 1, (spec, f, chosen, judge)
        c["frames"] += 1
        if not any_choice:
            assert not any(judge.values()), (spec, f, judge)
            c["silent"] += 1
    return c


@pytest.mark.parametrize("spec", SPECS + ("kill_dangerous:1.0",))
def test_choose_returns_legal_rows_and_judge_agrees(golden: Golden, frames: Frames, spec: str) -> None:
    rng = np.random.default_rng(12)
    pool = frames.captured + golden.of(Kind.COMBAT)[::24]
    if spec == "kill_dangerous:1.0":  # judge of the margin variant = the plain habit's; choose overrides less often
        (h,) = parse_habit(spec)
        (plain,) = parse_habit("kill_dangerous")
        n = golden.names
        changed = 0
        for f in pool:
            banned = bans_of(f, rng)
            s = rng.standard_normal(f.n_actions).astype(np.float32)
            s[list(banned)] = -np.inf
            greedy = int(np.argmax(s))
            got, base = h.choose(n, f, f.gvals, s, greedy, None), plain.choose(n, f, f.gvals, s, greedy, None)
            assert got is None or got == base
            changed += got is not None
        assert changed > 5
        return
    c = check_choose_and_judge(golden, spec, pool, rng)
    assert c["overrides"] > (3 if spec == "potion_reserve2" else 10), c


def test_potion_classes_match_the_catalog(golden: Golden) -> None:
    """Every classified potion exists, aims at what its class says (catalog target_type) and no potion with a Block,
    Summon or heal var is left out."""
    cat = golden.catalog.entries["potion"]
    tt = {golden.vocab.name("potion", i): e["target_type"] for i, e in cat.items()}
    for p in GUARD + HEAL:
        assert tt[p] == "AnyPlayer", p
    for p in ("WEAK_POTION", "BEETLE_JUICE"):
        assert tt[p] == "AnyEnemy", p
    for p in ("SHACKLING_POTION", "POTION_OF_BINDING"):
        assert tt[p] == "AllEnemies", p
    for i, e in cat.items():
        name = golden.vocab.name("potion", i)
        if set(e["vars"]) & {"Block", "Summon", "HealPercent", "MaxHp", "RegenPower", "PlatingPower",
                             "IntangiblePower", "BufferPower"}:
            assert name in GUARD + HEAL, name


# =========================================================================== choice sets and the advice start-up


@pytest.mark.parametrize("spec", SPECS)
def test_judgements_grade_only_the_rules_own_choice_set(golden: Golden, frames: Frames, spec: str) -> None:
    """The census of notes/advfix for the tactical habits: every row of every captured frame and of every 5th golden
    COMBAT frame, a non-zero judgement only for a row of the habit's own choice set (test_habit.in_choice_set); at
    the judged frames lethal_block / lethal_eot leave the potion uses that are no guard rows and the discards at 0."""
    (h,) = parse_habit(spec)
    judged = outside = 0
    for f in golden.of(Kind.COMBAT)[::5] + frames.captured:
        s = [h.judge(golden.names, f, f.gvals, a, ()) for a in range(f.n_actions)]
        for a, x in enumerate(s):
            assert not x or in_choice_set(golden, spec, f, a), (spec, f.episode_id, f.step_id, a, x)
        judged += sum(map(bool, s))
        outside += any(s) and sum(f.actions[a, ActionCol.VERB] in (Verb.USE_POTION, Verb.DISCARD_POTION) and not s[a]
                                  for a in range(f.n_actions))
    assert judged > 20, judged
    if spec.split(":")[0] in ("lethal_block", "lethal_eot"):
        assert outside > 20, outside  # the potion rows the r3 version graded -1


ROLLOUT_WEAK = [list(REST_RULE), ["potion_r1:Elite,Boss", 0.1], ["weak_attacker:cards", 0.05]]


def test_advice_start_up_accepts_the_training_forms_and_refuses_shared_steps(golden: Golden) -> None:
    """Accepted: the roll-out set + weak_attacker:cards (USE_POTION vs PLAY_CARD), potion_boss_take + weak_attacker:cards
    (+ enrage:none instead), potion_lethal + weak_attacker:cards, every judging spec alone (names resolve on the 4-seat
    vocab and catalog). Refused: potion_boss_use next to any other COMBAT rule (it judges card plays, END_TURN and
    potion uses), the USE_POTION rules pairwise, the PLAY_CARD rules pairwise, the evaluation-only margin variant."""
    vocab, cat = golden.vocab, golden.catalog
    old, new = parse_advice(ROLLOUT_WEAK)
    assert [r.name for r in old] == ["REST:SMITH"] and [r.spec for r in new] == ["potion_r1:Elite,Boss", "weak_attacker:cards"]
    assert [r.bonus for r in new] == [0.1, 0.05]
    Advisor(vocab, ROLLOUT_WEAK, cat)
    for ok in ([list(REST_RULE), ["potion_boss_take", 0.05], ["weak_attacker:cards", 0.05]],
               [list(REST_RULE), ["potion_boss_take", 0.05], ["enrage:none", 0.05]],
               [["potion_lethal", 0.05], ["weak_attacker:cards", 0.05]],
               [["potion_reserve2", 0.05], ["kill_dangerous", 0.05]]):
        Advisor(vocab, ok, cat)
    for spec in SPECS:
        Advisor(vocab, [[spec, 0.05]], cat)
    others = ("enrage:none", "potion_hold", "potion_boss", "potion_none", "potion_r1:Elite,Boss", "potion_lethal",
              "weak_attacker", "weak_attacker:cards", "kill_dangerous", "lethal_block", "lethal_eot",
              "potion_reserve2", "potion_boss_take")
    for other in others:
        for pair in ([["potion_boss_use", 0.05], [other, 0.05]], [[other, 0.05], ["potion_boss_use", 0.05]]):
            with pytest.raises(AssertionError, match="can judge one step"):
                parse_advice(pair)
    for a, b in (("potion_boss_take", "potion_r1:Elite,Boss"), ("potion_boss_take", "potion_lethal"),
                 ("potion_lethal", "potion_r1:Elite,Boss"), ("potion_reserve2", "potion_hold"),
                 ("weak_attacker", "potion_r1:Elite,Boss"), ("weak_attacker:cards", "enrage:none"),
                 ("kill_dangerous", "weak_attacker:cards"), ("lethal_block", "potion_lethal"),
                 ("potion_boss_take", "potion_boss_take:late")):
        with pytest.raises(AssertionError, match="can judge one step"):
            parse_advice([[a, 0.05], [b, 0.05]])
    with pytest.raises(AssertionError, match="evaluation-only"):
        parse_advice([["kill_dangerous:1.0", 0.05]])


def test_rollout_set_with_weak_attacker_cards_judges_only_its_rows(golden: Golden, frames: Frames) -> None:
    """``[[REST, SMITH, 0.75, 0.05, 0.05], [potion_r1:Elite,Boss, 0.1], [weak_attacker:cards, 0.05]]`` through the
    Advisor on every row of every golden frame and every captured frame: a REST record only for rest options; potion_r1
    only +1 on round-1 Elite / Boss USE_POTION rows; weak_attacker:cards only on PLAY_CARD rows of a Weak card aimed at
    an enemy, equal to the habit's own judge; never two at one step."""
    adv = Advisor(golden.vocab, ROLLOUT_WEAK, golden.catalog)
    (weak,) = parse_habit("weak_attacker:cards")
    c: dict[str, int] = {"rest": 0, "r1": 0, "weak+": 0, "weak-": 0}
    for f in golden.frames + frames.captured:
        g = f.gvals
        r1 = g[G.KIND] == Kind.COMBAT and g[G.COMBAT_ROUND] == 1 and golden.room(f) in ("Elite", "Boss")
        for a in range(f.n_actions):
            verb = int(f.actions[a, ActionCol.VERB])
            row, _ = adv.candidate(f, (), a)
            i, s = adv.judge(f, (), a, verb)
            assert row < 0 or (golden.option(f, a) or ("",))[0] == "REST_OPTION"
            assert row < 0 or i < 0
            w = weak.judge(golden.names, f, g, a, ()) if g[G.KIND] == Kind.COMBAT else 0
            if r1 and verb == Verb.USE_POTION:
                assert (i, s) == (0, 1), (f.episode_id, f.step_id, a)
            elif w:
                assert verb == Verb.PLAY_CARD and "WeakPower" in golden.card(f, a)["vars"] and (i, s) == (1, w), \
                    (f.episode_id, f.step_id, a)
            else:
                assert (i, s) == NO_JUDGE, (f.episode_id, f.step_id, a)
            c["rest"] += row >= 0
            c["r1"] += i == 0
            c["weak+"] += i == 1 and s > 0
            c["weak-"] += i == 1 and s < 0
    assert c["rest"] == 441 + 541 and c["r1"] >= 2 and c["weak+"] > 20 and c["weak-"] > 20, c
