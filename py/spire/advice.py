"""Advice: a temporary, explicit training bias toward one option of one decision kind (notes/advice/build.md).

Why: forcing SMITH at rest sites whenever the acting seat has at least 50% HP raised the greedy win rate of run
N4-spear's frozen network by 10-12 points at ascensions 9 and 10 (notes/habit/progress.md, 400 paired seeds), yet the
policy never chooses it greedily: one upgrade is worth ~0.4 win-rate points, far below the noise of one game's return.
The bias is removed again (``bonus`` annealed to 0) once the policy has adopted the habit; the policy holds no rule.

A rule is ``(kind name, option key name, hp_min, bonus[, below])`` (``PPOConfig.advice``). At a decision of that kind
whose frame offers the option as a legal, not cycle-banned CHOOSE_OPTION row, whose chosen action is a CHOOSE_OPTION
row of the rule's option kind (REST_OPTION: HEAL, SMITH, MEND, ...) and whose ACTING seat has ``hp / max_hp >= hp_min``
(an "advised step"), the advantage that enters the policy loss is ``A + bonus`` if the chosen action is the option's
row and ``A - bonus`` if it is another option of the kind (raw advantage units, before the learner's normalisation).
Any other action taken at such a frame (potion use or discard, PROCEED, ...) carries no record: the first version
pushed against them too, and within 40 minutes of going live three runs stopped using potions in combat (greedy: 43
-> 0.4 uses per game, about 8 win-rate points at ascension 10; notes/advfix). Returns, value / auxiliary targets, GAE,
the behaviour policy and its masks are unchanged.

``below`` (default 0 = none) makes the rule two-sided: where the option is on offer but the seat's HP is under
``hp_min`` the offset is ``-below`` for the option and ``+below`` for another option of the kind. Measured on
2026-11-16: ten minutes after the one-sided rule went live on N4-spear (hp_min 0.75) the SMITH share rose from 0.47 to
0.68 at HP >= 0.75 and from 0.44 to 0.60 below it - the push generalised across HP - while upgrading at any HP costs
9.8 win-rate points and the gain peaks at the 0.75 threshold (+16.5 at ascension 9, +14.0 at ascension 10; 0.5: +10.5,
0.9: +13.5, 1.0: +10.8).

Split of the work: the actor (:class:`Advisor`, it has the frame) records per step the candidate row (-1 = none: not
a rule kind, option not offered, banned, or the chosen action is not an option of the kind) and the acting seat's HP
fraction; the record travels in ``Segment.advice_row`` / ``Segment.advice_hp`` (also through the shared-memory ring,
:mod:`spire.ipc`); the learner applies ``hp_min`` and ``bonus`` (:func:`advice_offset`) and reports
:func:`advice_metrics`.

Habit rules (notes/unify/build.md): a rule ``(habit spec, bonus)`` (e.g. ``("potion_hold", 0.05)``) names one habit of
:mod:`spire.habit` that has a ``judge``. The actor grades every recorded step with every habit rule of the step's
decision kind (:meth:`Advisor.judge`: +1 = what the habit asks for, -1 = what it forbids, 0 = nothing to say, from the
same code as the habit's evaluation ``choose``; non-zero only for rows of the habit's own choice set,
:mod:`spire.habit`) and stores the index of the rule that judged it non-zero (-1 = none) and the sign
(``Segment.advice_rule`` / ``advice_sign``, int8); the learner adds ``sign * bonus`` of that rule to the
policy advantage at the same place as the offset above (:func:`habit_offset`) and reports :func:`habit_metrics`.
At most one rule may judge a step (asserted in the actor): the rules in use forbid or ask for disjoint actions, and a
step graded twice would count one habit measurement twice. Both formats may appear in one config; an old rule and a
habit rule that can judge the same decision kind are refused (:func:`parse_advice`), so every step has at most one
offset. The one combination of habit rules of one decision kind with overlapping verbs is ordered by :data:`YIELDS`:
``prefs`` + ``deck_cap`` (both CARD_REWARD); the prefs rule does not judge the frames where the deck_cap rule applies,
as in a composite evaluation habit (``spire.habit.parse_habit``: deck_cap answers first).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .habit import (ALIASES, DeckCap, Enrage, HealLate, KillDangerous, LethalBlock, NoElite, PageTable, PotionBoss,
                    PotionBossTake, PotionHold, PotionLethal, PotionReserve, PotionRound1, Prefs, ShopBan, ShopPrefer,
                    Smith, SmithPre, WeakAttacker, WeakAttackerCards, _Names, parse_habit)
from .proto import Frame
from .schema import ActionCol, G, Kind, OptionCol, OptionKind, SeatCol, Table, Verb
from .vocab import Catalog, Vocab

# decision kind -> (OPTIONS row kind, vocab category of its KEY_ID) of the options a rule of that kind can advise
ADVICE_OPTIONS = {Kind.REST: (OptionKind.REST_OPTION, "rest_option")}
NO_ADVICE = (-1, 0.0)
NO_JUDGE = (-1, 0)
# the _Names groups each judging habit reads (resolved by the Advisor at start-up: a missing name fails there)
JUDGE_NAMES = {Smith: ("smith", "room"), SmithPre: ("smith", "room", "boss_point"), NoElite: ("elite_point",),
               PotionHold: ("room",), PotionRound1: ("room",), Enrage: ("enrage", "card_skill", "card_skill_no_block"),
               ShopPrefer: ("shop",), ShopBan: ("shop",), DeckCap: ("card_skip",),
               Prefs: ("card_skip",),  # Prefs: also its own card names (Prefs.values)
               HealLate: ("heal", "boss_point"),
               LethalBlock: ("osty", "tvars", "potion_class", "fortifier", "card_defence", "card_eot", "eot_vars"),
               KillDangerous: ("osty", "tvars", "card_attack_one"),
               WeakAttacker: ("osty", "card_weak_one", "weak_potion"), WeakAttackerCards: ("osty", "card_weak_one"),
               PotionReserve: ("room", "osty"), PotionLethal: ("osty", "potion_class", "fortifier"),
               PotionBoss: ("room", "osty"), PotionBossTake: ("room", "osty"),
               PageTable: ("relic_option", "page_names")}  # PageTable: also its pages / options (PageTable.resolve)
# precedence among habit rules that judge one decision kind: a rule of the key's class does not judge a frame where a
# rule of one of the listed classes applies (``applies``), so those pairs may share a config (habit_rules)
YIELDS: dict[type, tuple[type, ...]] = {Prefs: (DeckCap,)}
_VERB, _SRC_TABLE, _SRC_ROW = int(ActionCol.VERB), int(ActionCol.SRC_TABLE), int(ActionCol.SRC_ROW)
_CHOOSE, _OPTIONS = int(Verb.CHOOSE_OPTION), int(Table.OPTIONS)
_O_KIND, _O_KEY = int(OptionCol.KIND), int(OptionCol.KEY_ID)
_HP, _MAX_HP = int(SeatCol.HP), int(SeatCol.MAX_HP)


@dataclass(frozen=True)
class AdviceRule:
    kind: Kind
    option: str
    hp_min: float
    bonus: float
    below: float = 0.0  # bonus against the option at offered steps with HP below hp_min (0 = none)

    @property
    def name(self) -> str:
        return f"{self.kind.name}:{self.option}"


@dataclass(frozen=True)
class HabitRule:
    spec: str  # one habit of spire.habit with a judge, e.g. "potion_hold", "smith:0.75", "enrage:none"
    bonus: float
    habit: Any

    @property
    def name(self) -> str:
        return self.spec

    @property
    def kinds(self) -> frozenset[Kind]:
        """The decision kinds the habit can judge."""
        return self.habit.KINDS

    @property
    def combat(self) -> bool:
        return Kind.COMBAT in self.habit.KINDS


def advice_rules(spec: Sequence[Sequence[Any]]) -> tuple[AdviceRule, ...]:
    """Validated old-format rules (option names are checked against the vocab by :class:`Advisor`)."""
    rules = []
    for item in spec:
        assert len(item) in (4, 5), \
            f"advice rule {item!r}: expected [kind, option, hp_min, bonus(, below)] or [habit, bonus]"
        kind, option, hp_min, bonus = item[:4]
        below = item[4] if len(item) == 5 else 0.0
        assert math.isfinite(below) and below >= 0.0, f"advice rule {item!r}: below must be finite and >= 0"
        assert kind in Kind.__members__ and Kind[kind] in ADVICE_OPTIONS, \
            f"advice rule {item!r}: kind must be one of {sorted(k.name for k in ADVICE_OPTIONS)}"
        assert isinstance(option, str) and option, f"advice rule {item!r}: option must be an option key name"
        assert 0.0 <= hp_min <= 1.0, f"advice rule {item!r}: hp_min must be in [0, 1]"
        assert math.isfinite(bonus) and bonus >= 0.0, f"advice rule {item!r}: bonus must be finite and >= 0"
        rules.append(AdviceRule(Kind[kind], option, float(hp_min), float(bonus), float(below)))
    assert len({r.kind for r in rules}) == len(rules), "advice: at most one rule per decision kind"
    return tuple(rules)


def habit_rules(spec: Sequence[Sequence[Any]]) -> tuple[HabitRule, ...]:
    """Validated habit rules ``(habit spec, bonus)`` (vocab / catalog names are checked by :class:`Advisor`)."""
    rules = []
    for item in spec:
        assert len(item) == 2, f"advice rule {item!r}: expected [habit, bonus]"
        habit, bonus = item
        assert isinstance(habit, str) and habit and habit not in ALIASES and "+" not in habit, \
            f"advice rule {item!r}: one habit per rule"
        (h,) = parse_habit(habit)
        assert type(h) in JUDGE_NAMES, f"advice rule {item!r}: {type(h).__name__} is evaluation-only (no judge)"
        assert math.isfinite(bonus) and bonus >= 0.0, f"advice rule {item!r}: bonus must be finite and >= 0"
        rules.append(HabitRule(habit, float(bonus), h))
    # two rules that can judge one step are refused here, not by the actor's assert in training: variants of one habit
    # (potion_hold + potion_boss, smith + smith_pre) and habits of one decision kind (shop_relic + shop_nocard: buying a
    # card while a relic is affordable). Rules of one kind are allowed only with disjoint VERBS (potion_hold + enrage,
    # potion_r1 + weak_attacker:cards) or an order of precedence (YIELDS: prefs + deck_cap). potion_boss_use judges card
    # plays, END_TURN and potion uses, so it shares a verb with every other COMBAT rule (enrage, potion_hold / _boss /
    # _none / _r1, potion_lethal, weak_attacker, kill_dangerous, lethal_block, ...) and is accepted only alone among them.
    for i, a in enumerate(rules):
        for b in rules[:i]:
            va, vb = a.habit.VERBS, b.habit.VERBS
            ordered = type(b.habit) in YIELDS.get(type(a.habit), ()) or type(a.habit) in YIELDS.get(type(b.habit), ())
            assert not a.kinds & b.kinds or ordered or (va is not None and vb is not None and not va & vb), \
                f"advice: {b.spec} and {a.spec} can judge one step"
    assert len(rules) <= 127, "advice_rule is int8"
    return tuple(rules)


def parse_advice(spec: Sequence[Sequence[Any]]) -> tuple[tuple[AdviceRule, ...], tuple[HabitRule, ...]]:
    """``PPOConfig.advice`` -> (old-format rules, habit rules); an old rule and a habit rule must not share a decision
    kind (a step would get two offsets)."""
    old = advice_rules([x for x in spec if len(x) != 2])
    new = habit_rules([x for x in spec if len(x) == 2])
    for r in old:
        for h in new:
            assert r.kind not in h.kinds, f"advice: {r.name} and {h.spec} both judge {r.kind.name} decisions"
    return old, new


class Advisor:
    """Actor side: the candidate ACTIONS row of a frame and the acting seat's HP fraction (old-format rules), and the
    judgement of the taken action by the habit rules (module docstring)."""

    def __init__(self, vocab: Vocab, spec: Sequence[Sequence[Any]], catalog: Catalog | None = None) -> None:
        self.rules, self.habits = parse_advice(spec)
        assert self.rules or self.habits, "an Advisor needs at least one rule"
        self.names = _Names(vocab, catalog)
        for h in self.habits:
            for attr in JUDGE_NAMES[type(h.habit)]:
                getattr(self.names, attr)
            if isinstance(h.habit, Prefs):
                h.habit.values(self.names)
            if isinstance(h.habit, PageTable):
                h.habit.resolve(self.names)
        # per decision kind: (rule index, habit, verbs it can grade non-zero or None = any, habits of the rules it
        # yields to) of the habit rules that can judge it; the verb of the sent action is the cheap precondition of
        # the COMBAT habits (most steps stop here)
        self._judges: dict[int, tuple[tuple[int, Any, frozenset[int] | None, tuple[Any, ...]], ...]] = {
            int(k): tuple((i, h.habit, h.habit.VERBS, tuple(o.habit for o in self.habits if k in o.kinds
                                                              and type(o.habit) in YIELDS.get(type(h.habit), ())))
                          for i, h in enumerate(self.habits) if k in h.kinds) for k in Kind}
        self._keys: dict[int, tuple[int, int]] = {}
        for r in self.rules:
            option_kind, category = ADVICE_OPTIONS[r.kind]
            assert r.option in vocab[category].index, f"advice rule {r.name}: {r.option!r} is not a {category} key"
            self._keys[int(r.kind)] = (int(option_kind), vocab.id(category, r.option))

    def candidate(self, frame: Frame, banned: Sequence[int], action: int) -> tuple[int, float]:
        """``(row, hp / max_hp)`` if the frame's kind has a rule, offers its option as a legal row not in ``banned``
        (the cycle-ban / commit mask of this decision) and the taken ``action`` is a row of the rule's option kind;
        else :data:`NO_ADVICE`."""
        g = frame.gvals
        key = self._keys.get(g[G.KIND])
        if key is None:
            return NO_ADVICE
        acts = frame.actions
        rows = np.flatnonzero((acts[:, _VERB] == _CHOOSE) & (acts[:, _SRC_TABLE] == _OPTIONS))
        opts = frame.table(Table.OPTIONS)[acts[rows, _SRC_ROW]]
        # only a choice among the rule's own options is advised: a potion use / discard or PROCEED taken at this
        # frame carries no record (the first version pushed against them and the runs stopped using potions)
        if action not in rows[opts[:, _O_KIND] == key[0]]:
            return NO_ADVICE
        rows = rows[(opts[:, _O_KIND] == key[0]) & (opts[:, _O_KEY] == key[1])]
        assert len(rows) <= 1, "one advised option row per decision"
        if not len(rows) or int(rows[0]) in banned:
            return NO_ADVICE
        seat = frame.table(Table.SEATS)[g[G.ACTING_SEAT]]
        assert seat[_MAX_HP] > 0, "the acting seat has a max HP"
        return int(rows[0]), float(seat[_HP]) / float(seat[_MAX_HP])

    def judge(self, frame: Frame, banned: Sequence[int], action: int, verb: int) -> tuple[int, int]:
        """``(rule index, sign)`` of the habit rule that judges ``action`` (of verb ``verb``) at the frame non-zero,
        else :data:`NO_JUDGE`; ``banned`` = the decision's cycle-ban / commit mask."""
        g = frame.gvals
        out = NO_JUDGE
        for i, habit, verbs, over in self._judges[g[0]]:
            if verbs is not None and verb not in verbs:
                continue
            if over and any(o.applies(self.names, frame, g, banned) for o in over):
                continue
            s = habit.judge(self.names, frame, g, action, banned)
            if s:
                assert out[0] < 0, f"advice rules {self.habits[out[0]].spec} and {self.habits[i].spec} judge one step"
                out = (i, s)
        return out


def _advised(r: AdviceRule, kinds: NDArray[Any], row: NDArray[np.int16], hp: NDArray[np.float32]) -> NDArray[np.bool_]:
    # hp is the float32 of the exact fraction; float32 rounding is monotonic, so the comparison in float32 keeps
    # every fraction >= hp_min (HP values below 2**12 keep distinct fractions apart)
    return (kinds == int(r.kind)) & (row >= 0) & (hp >= np.float32(r.hp_min))


def advice_offset(rules: Sequence[AdviceRule], kinds: NDArray[Any], actions: NDArray[np.int64],
                  row: NDArray[np.int16], hp: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per step: ``+bonus`` at advised steps whose chosen action is the advised row, ``-bonus`` at the other advised
    steps (another option of the kind: the actor records no other action, :meth:`Advisor.candidate`), 0 elsewhere."""
    off = np.zeros(len(kinds), np.float32)
    for r in rules:
        adv = _advised(r, kinds, row, hp)
        b = np.float32(r.bonus)
        off[adv] = np.where(actions[adv] == row[adv], b, -b)
        if r.below:
            low = (kinds == int(r.kind)) & (row >= 0) & ~adv
            c = np.float32(r.below)
            off[low] = np.where(actions[low] == row[low], -c, c)
    return off


def _share(x: NDArray[np.bool_]) -> float | None:
    return round(float(x.mean()), 4) if len(x) else None


def _mean(x: NDArray[np.floating[Any]]) -> float | None:
    return round(float(x.mean()), 5) if len(x) else None


def advice_metrics(rules: Sequence[AdviceRule], kinds: NDArray[Any], actions: NDArray[np.int64],
                   row: NDArray[np.int16], hp: NDArray[np.float32], adv: NDArray[np.float32]) -> dict[str, Any]:
    """Per rule (``"REST:SMITH"``): advised ``steps``; ``chosen`` = share of them that chose the advised row, overall
    and per HP bucket (``chosen_by_hp`` / ``steps_by_hp``: [hp_min, 0.75), [0.75, 1.0), 1.0); ``raw_adv_chosen`` /
    ``raw_adv_other`` = mean GAE advantage (before the offset) of advised steps that chose / did not choose it;
    reference ``below_steps`` / ``below_chosen``: steps with the option on offer but HP below ``hp_min``, and the share
    that chose it. Shares and means of no steps are None."""
    out: dict[str, Any] = {}
    for r in rules:
        offered = (kinds == int(r.kind)) & (row >= 0)
        adv_m = _advised(r, kinds, row, hp)
        chosen = actions == row
        below = offered & ~adv_m
        h = hp[adv_m]
        c = chosen[adv_m]
        lo, hi = np.float32(0.75), np.float32(1.0)
        buckets = {"lo": h < lo, "hi": (h >= lo) & (h < hi), "full": h >= hi}
        out[r.name] = {
            "hp_min": r.hp_min, "bonus": r.bonus, "steps": int(adv_m.sum()), "chosen": _share(c),
            "steps_by_hp": {k: int(m.sum()) for k, m in buckets.items()},
            "chosen_by_hp": {k: _share(c[m]) for k, m in buckets.items()},
            "raw_adv_chosen": _mean(adv[adv_m & chosen]), "raw_adv_other": _mean(adv[adv_m & ~chosen]),
            "below_steps": int(below.sum()), "below_chosen": _share(chosen[below]),
        }
    return out


def habit_offset(rules: Sequence[HabitRule], rule: NDArray[np.int8], sign: NDArray[np.int8]) -> NDArray[np.float32]:
    """Per step: ``sign * bonus`` of the habit rule that judged it, 0 at steps no rule judged."""
    bonus = np.array([r.bonus for r in rules], np.float32)
    off = np.zeros(len(rule), np.float32)
    m = rule >= 0
    off[m] = np.where(sign[m] > 0, bonus[rule[m]], -bonus[rule[m]])
    return off


def habit_metrics(rules: Sequence[HabitRule], kinds: NDArray[Any], rule: NDArray[np.int8], sign: NDArray[np.int8],
                  adv: NDArray[np.float32]) -> dict[str, Any]:
    """Per habit rule (named by its spec): ``judged`` steps, ``plus`` / ``minus`` = those judged +1 / -1, the mean
    raw GAE advantage (before the offset) of the +1 and of the -1 steps (None without such steps), and for rules that
    judge COMBAT decisions ``per_combat`` = judged steps / COMBAT samples of the update (how often the rule bites)."""
    out: dict[str, Any] = {}
    n_combat = int((kinds == int(Kind.COMBAT)).sum())
    for i, r in enumerate(rules):
        mine = rule == i
        plus, minus = mine & (sign > 0), mine & (sign < 0)
        m: dict[str, Any] = {"bonus": r.bonus, "judged": int(mine.sum()), "plus": int(plus.sum()),
                             "minus": int(minus.sum()), "raw_adv_plus": _mean(adv[plus]),
                             "raw_adv_minus": _mean(adv[minus])}
        if r.combat:
            m["per_combat"] = round(m["judged"] / n_combat, 6) if n_combat else None
        out[r.name] = m
    return out
