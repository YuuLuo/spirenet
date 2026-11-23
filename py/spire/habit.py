"""Habit study (notes/habit): fixed strategic habits forced on top of the greedy policy, and their per-game counters.

A habit is a measurement device (``spire.actor.Override``, evaluation only): it reads only what a player sees in the
frame (HP, room type, legal actions, deck contents, the map, powers) plus the policy's own scores, and returns an ACTIONS
row or None (None = the policy's greedy choice). In training a habit can only be advice (:mod:`spire.advice`): its
``judge`` grades the action that was taken (+1 what the habit asks for, -1 what it forbids, 0 nothing to say), from the
same core as ``choose`` (``ok`` = the legal, not banned rows: finite scores in ``choose``, rows outside the cycle-ban /
commit mask in ``judge``). Habits without ``judge`` are evaluation-only. A judge grades non-zero only the rows of the
habit's own choice set at the frame (``VERBS`` narrows it to verbs): "-1 otherwise" means "another row of that set",
never a potion use or discard, PROCEED or another kind of option taken at the frame, which are judged 0 (a REST rule
that pushed against every other action stopped three runs using potions within 40 minutes, notes/advfix).

Habits (``parse_habit``; ``+`` composes, the first component with an answer wins; components touch disjoint frames):

* ``smith:<t>`` - REST frame whose acting seat has a legal SMITH option and HP >= t * max HP: SMITH. In the following
  DECK_UPGRADE prompt of a RestSite room the policy picks its own best card (CANCEL is masked by commit_select) and
  CONFIRM_SELECTION is taken as soon as it is legal. judge: REST frame with that row: HP >= t * max HP: +1 for the
  row, -1 for another rest option (HEAL, MEND, ...); below: -1 for the row, +1 for another rest option (two-sided,
  notes/advice); other actions 0. RestSite DECK_UPGRADE prompt with a legal CONFIRM_SELECTION: +1 for it, -1 for CANCEL
  (SELECT_CARD 0).
* ``smith_pre:<t>:<t_pre>`` - as ``smith:<t>``, but at a rest site in the map row directly below the act's Boss node the
  HP threshold is ``t_pre`` (1.0 = upgrade only at full HP, i.e. heal before the boss whenever hurt). judge: likewise.
* ``no_elite:<a>-<b>`` - MAP frame in acts a..b (1-based) whose legal TRAVEL targets include an Elite node and a
  non-Elite node, when the policy's choice is an Elite TRAVEL: the policy's best non-Elite TRAVEL instead. judge: -1
  for an Elite TRAVEL at such a frame.
* ``potion_hold`` - COMBAT frame of a Monster room, policy's choice USE_POTION: its best action that is neither
  USE_POTION nor DISCARD_POTION. ``potion_boss``: the same in Monster and Elite rooms (potions only at bosses). judge:
  -1 for USE_POTION in those rooms.
* ``potion_r1:<rooms>`` - COMBAT frame in round 1 (GLOBAL COMBAT_ROUND) of a room of the listed types (comma list, e.g.
  ``Elite,Boss``) with an allowed USE_POTION row, policy's choice not USE_POTION: the policy's best USE_POTION row (the
  network drank its combat potions early until the REST rule's push against potion rows, notes/advfix). judge: +1 for
  USE_POTION at such a frame, never -1 (positive-only: nothing else is pushed down).
* ``enrage:none`` - COMBAT frame in which some creature holds ENRAGE_POWER with amount > 0 (TEST_SUBJECT form 1: every
  Skill played by anyone gives it that much Strength, game source EnragePower.AfterCardPlayed): when the policy's choice
  is to play a Skill, its best action that is not a Skill play (Attacks, Powers, potions, END_TURN). ``enrage:block``:
  the same, but Skills that gain block (catalog ``gains_block``) stay allowed. judge: -1 for playing a forbidden Skill.
* ``shop_relic`` - SHOP frame with a legal (affordable) RELIC entry: buy the policy's best-scored relic entry first.
  judge: +1 for a RELIC entry at such a frame, -1 for another SHOP_ENTRY purchase, 0 otherwise (leaving, potions).
* ``shop_nocard`` - SHOP frame, policy's choice is a CHARACTER_CARD / COLORLESS_CARD entry: its best other action.
  ``shop_nopotion``: the same for POTION entries. judge: -1 for such an entry.
* ``remove_basic`` - SHOP frame where buying the CARD_REMOVAL entry is legal and the acting seat's deck holds a
  removable curse (type Curse, not Eternal), a starter Strike or a starter Defend (catalog ``is_basic_strike_or_defend``
  with tag Strike / Defend): buy the removal. The DECK_REMOVAL prompt that this purchase opens (by habit or policy; the
  next frames of the env while they are Shop-room DECK_REMOVAL prompts of the buying seat): select a removable curse
  (policy's best of them), else a starter Strike, else a starter Defend (policy's best copy), then CONFIRM_SELECTION.
  Other shop decisions and other removal prompts stay the policy's. Evaluation-only (the prompt part needs the
  purchase, which a frame does not show).
* ``deck_cap:<n>`` - CARD_REWARD frame whose acting seat's deck holds at least n cards (SEATS DECK_COUNT) and whose
  Skip alternative (CHOOSE_OPTION of the CARD_REWARD_ALT option SKIP) is legal: take the Skip (round 2's habit).
  judge: at such a frame +1 for the Skip, -1 for a card pick, 0 otherwise (REROLL / SACRIFICE).
* ``prefs:file:<json>`` / ``prefs:list:<prefer>/<avoid>`` - card-reward preferences from a fixed table: a JSON file
  ``{"prefer": [card names], "avoid": [card names]}`` (alt/screen/add_prefs.py's "file" mode), or the same lists
  inline, comma-separated card names (``prefs:list:CACOPHONY,DEBILITATE/THE_BALL``; either side may be empty; the
  inline form keeps a run's config.json self-contained). CARD_REWARD frame: some allowed pick offers a PREFER card ->
  the policy's best-scored PREFER pick; else, when the policy's choice is an AVOID pick, its best allowed action that
  is not an AVOID pick (other picks, SKIP / REROLL / SACRIFICE; none left: the policy's choice). judge: at a frame
  with an allowed PREFER pick +1 for a PREFER pick, -1 for another pick or the Skip (REROLL / SACRIFICE 0); at one
  without, -1 for an AVOID pick when another allowed action exists, else 0.
* Precedence: deck_cap before prefs. Where deck_cap applies (its Skip is taken), prefs is silent: ``parse_habit``
  orders a composite's prefs component after deck_cap, and an advice rule of prefs does not judge the frames where a
  deck_cap rule of the same config applies (``spire.advice.YIELDS``).
* ``combo`` = smith:0.5+potion_hold+remove_basic; ``combo2`` = combo+no_elite:1-2.
* ``heal_late:<t>[:<act>[:<rows>]]`` - REST frame at a rest site in the last ``rows`` (default 1) map rows below the
  act's first Boss node, in act ``act`` (1-based, default 3; 0 = every act), acting seat below t * max HP with a legal
  HEAL option: HEAL (the lead's add-on: enter the two act-3 bosses at full HP). judge: at such a frame +1 for HEAL, -1
  for another rest option (SMITH, MEND, ...), other actions 0.
* ``potion_none`` - potion_hold in every combat room (Monster, Elite, Boss; the lead's potion ablation). judge: -1 for
  USE_POTION in combat.

Tactical combat habits (notes/r3/tactics.md; the frame quantities they read are defined at ``Threat``):

* ``lethal_block[:<f>]`` - some seat faces lethal (f x HP, default 1) and an allowed row adds defence to such a seat
  (``guard_rows``): such a row instead of any other choice. judge: +1 such a row, -1 another card play or END_TURN
  there (potion uses that are not guard rows, discards: 0).
* ``kill_dangerous`` - an Attack aimed at one enemy with targets that differ in danger: aim at the most dangerous one
  when the card kills it or kills no target. ``kill_dangerous:<m>``: only within m (logits) of the policy's choice,
  evaluation-only. judge: +1 / -1 for the aim of such a play (the rows of the same card on other enemies).
* ``weak_attacker[:cards]`` - a Weak card / WEAK_POTION (``:cards``: cards only) aimed at one enemy: aim at the most
  dangerous one. judge: likewise.
* ``potion_reserve2[:all]`` - no USE_POTION in the first act-3 boss fight of the A10 double boss unless a seat faces
  lethal (``:all``: none in act-3 Monster / Elite rooms either). judge: -1 for such a USE_POTION.
* ``potion_lethal`` - a seat facing lethal that would end its turn while it holds a potion that saves it
  (``saving_potion_rows``) uses that potion first. judge: -1 that END_TURN, +1 such a use.
* habit 6 (loss reading): ``lethal_eot`` - lethal_block with the end-of-turn hand damage (WITHER ...) counted;
  ``potion_boss_use[:all|:late]`` - in act-3 (any; late: the 2nd act-3 boss, the 1st only while a seat faces lethal)
  Boss rooms a seat that has played no card yet this turn drinks its potions first (the policy's best use). judge: -1
  for its card play / END_TURN, +1 for its USE_POTION (two-sided on the seat's card plays and END_TURN: as an advice
  rule ``spire.advice`` refuses it next to every other combat rule, they all share one of these verbs).
  ``potion_boss_take``: the same choose, one-sided judge (+1 only, VERBS USE_POTION) for training.

Event tables (notes/r3/events.md; :class:`Explore`):

* ``explore:rand:<master>:<cfg json>`` / ``explore:file:<json>`` - event-page, relic and rare-option table per game
  (r3 task A's screening). Evaluation-only, except a table of event pages alone (:class:`PageTable`).
* ``explore:file:<json>`` with only ``pages`` / ``explore:page:<page>=<option key>`` (inline, one page, e.g.
  ``explore:page:ROOM_FULL_OF_CHEESE.INITIAL=ROOM_FULL_OF_CHEESE.pages.INITIAL.options.SEARCH``: page by ``short_page``
  name, option by its event_option_key vocab name) - the page table (:class:`PageTable`): at a listed page whose
  option is allowed, that option. judge: at such a page frame +1 for the option, -1 for another (non-relic) option of
  that page; PROCEED, potion actions, relic options 0.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .proto import Frame
from .schema import (ActionCol, CardCol, CardSelectSub, CreatureCol, CreatureKind, G, IntentCol, ItemZone, Kind,
                     MapNodeCol, OptionCol, OptionFlag, OptionKind, PotionCol, PowerCol, RefCategory, RefCol, RelicCol,
                     SeatCol, Table, VarCol, Verb, Zone)
from .tokenize import Sample
from .vocab import Catalog, Vocab

if TYPE_CHECKING:  # actor -> advice -> habit
    from .actor import EpisodeJob

ALIASES = {"baseline": "", "combo": "smith:0.5+potion_hold+remove_basic",
           "combo2": "smith:0.5+potion_hold+remove_basic+no_elite:1-2"}
# plain ints for the judges (they run for every recorded training step)
_KIND, _KIND_SUB, _SEAT, _ROOM, _IN_COMBAT, _ROUND = int(G.KIND), int(G.KIND_SUB), int(G.ACTING_SEAT), \
    int(G.ROOM_TYPE), int(G.IN_COMBAT), int(G.COMBAT_ROUND)
_VERB, _SRC_ROW = int(ActionCol.VERB), int(ActionCol.SRC_ROW)
_COMBAT, _REST, _MAP, _SHOP, _CARD_SELECT, _CARD_REWARD = (int(k) for k in (
    Kind.COMBAT, Kind.REST, Kind.MAP, Kind.SHOP, Kind.CARD_SELECT, Kind.CARD_REWARD))
_USE_POTION, _PLAY_CARD, _PICK_OFFER_CARD, _CHOOSE_OPTION, _CONFIRM, _CANCEL = (int(v) for v in (
    Verb.USE_POTION, Verb.PLAY_CARD, Verb.PICK_OFFER_CARD, Verb.CHOOSE_OPTION, Verb.CONFIRM_SELECTION, Verb.CANCEL))
_END_TURN, _DISCARD_POTION = int(Verb.END_TURN), int(Verb.DISCARD_POTION)
_EVENT, _TREASURE = int(Kind.EVENT), int(Kind.TREASURE)
_PET, _ENEMY = int(CreatureKind.PET), int(CreatureKind.ENEMY)
_COMBAT_ROOMS = ("Monster", "Elite", "Boss")

# potion classes, from the game source (Models.Potions/<Potion>.OnUse and the powers they apply)
POTION_GUARD = ("BLOCK_POTION", "SHIP_IN_A_BOTTLE", "BONE_BREW", "HEART_OF_IRON", "GHOST_IN_A_JAR", "LUCKY_TONIC",
                "FORTIFIER")  # target player: GainBlock / Osty Summon / Plating (block at turn end) / Intangible /
#                               Buffer (next HP loss 0) / FORTIFIER doubles the target's block (only with block > 0)
POTION_HEAL = ("BLOOD_POTION", "AMBERGRIS", "REGEN_POTION", "FRUIT_JUICE")  # Heal / Regen (heals at the player turn's
#                                                                            end) / GainMaxHp (heals the gained HP)
POTION_ENEMY = ("WEAK_POTION", "BEETLE_JUICE")  # one enemy: Weak (x0.75) / Shrink (-30% damage)
POTION_ALL_ENEMIES = ("SHACKLING_POTION", "POTION_OF_BINDING")  # -7 Strength until the enemy turn's end / Weak 1
ALLY_GUARD_CARDS = ("LIFT", "INTERCEPT")  # AnyAlly cards whose defence goes to the target (MIMIC's goes to the player)
DELAYED_SUMMON = ("INVOKE",)  # Summon next turn (SummonNextTurnPower)
_DEF_NONE, _DEF_SELF, _DEF_ALL, _DEF_TARGET = 0, 1, 2, 3
_PC_NONE, _PC_GUARD, _PC_HEAL, _PC_ENEMY, _PC_ALL = 0, 1, 2, 3, 4


class _Names:
    """Vocab ids and catalog card classes the habits and the recorder need (each group resolved on first use: a habit
    asserts only the names it reads, so the advice of a training run on a vocab without e.g. ENRAGE_POWER can use the
    other habits)."""

    def __init__(self, vocab: Vocab, catalog: Catalog | None) -> None:
        self.vocab = vocab
        self.catalog = catalog
        self.room_names = vocab["enum:RoomType"].entries
        self.rest_names = vocab["rest_option"].entries
        self.point_names = vocab["enum:MapPointType"].entries

    def vid(self, cat: str, name: str) -> int:
        assert name in self.vocab[cat].index, f"vocab {cat} lacks {name}"
        return self.vocab.id(cat, name)

    @cached_property
    def smith(self) -> int:
        return self.vid("rest_option", "SMITH")

    @cached_property
    def heal(self) -> int:
        return self.vid("rest_option", "HEAL")

    @cached_property
    def removal(self) -> int:
        return self.vid("shop_section", "CARD_REMOVAL")

    @cached_property
    def shop(self) -> dict[str, int]:
        return {s: self.vid("shop_section", s) for s in ("CHARACTER_CARD", "COLORLESS_CARD", "POTION", "RELIC")}

    @cached_property
    def enrage(self) -> int:
        return self.vid("power", "ENRAGE_POWER")

    @cached_property
    def elite_point(self) -> int:
        return self.vid("enum:MapPointType", "Elite")

    @cached_property
    def boss_point(self) -> int:
        return self.vid("enum:MapPointType", "Boss")

    @cached_property
    def card_skip(self) -> int:
        return self.vid("card_reward_alt", "SKIP")

    @cached_property
    def room(self) -> dict[str, int]:
        return {n: self.vid("enum:RoomType", n) for n in ("Monster", "Elite", "Boss", "RestSite", "Shop")}

    def _cards(self) -> dict[int, Any]:
        assert self.catalog is not None, "card classes need the catalog"
        return self.catalog.entries["card"]

    @cached_property
    def card_skill(self) -> np.ndarray:
        """card id -> is a Skill (level 0 of the catalog)"""
        out = np.zeros(self.vocab.size("card"), bool)
        for i, levels in self._cards().items():
            out[i] = levels[0]["type"] == "Skill"
        return out

    @cached_property
    def card_skill_no_block(self) -> np.ndarray:
        """card id -> is a Skill that gains no block"""
        out = np.zeros(self.vocab.size("card"), bool)
        for i, levels in self._cards().items():
            out[i] = levels[0]["type"] == "Skill" and not levels[0]["gains_block"]
        return out

    @cached_property
    def card_class(self) -> np.ndarray:
        """card id -> 1 removable curse, 2 starter Strike, 3 starter Defend, 0 other (removal priority order)"""
        out = np.zeros(self.vocab.size("card"), np.int8)
        for i, levels in self._cards().items():
            e = levels[0]
            if e["type"] == "Curse" and "Eternal" not in e["keywords"]:
                out[i] = 1
            elif e["is_basic_strike_or_defend"]:
                assert ("Strike" in e["tags"]) != ("Defend" in e["tags"]), (i, e["tags"])
                out[i] = 2 if "Strike" in e["tags"] else 3
        assert (out == 2).sum() == (out == 3).sum() >= 6, "one starter Strike / Defend per character"
        return out

    # ------------------------------------------------------------------ tactical combat habits (notes/r3/tactics.md)
    @cached_property
    def osty(self) -> int:
        return self.vid("monster", "OSTY")

    @cached_property
    def tvars(self) -> dict[str, int]:
        return {v: self.vid("var_name", v) for v in ("Block", "CalculatedBlock", "Summon", "Damage", "CalculatedDamage",
                                                     "OstyDamage", "CalculatedHits", "HitCount")}

    @cached_property
    def potion_class(self) -> np.ndarray:
        """potion id -> _PC_*"""
        out = np.zeros(self.vocab.size("potion"), np.int8)
        for names, c in ((POTION_GUARD, _PC_GUARD), (POTION_HEAL, _PC_HEAL), (POTION_ENEMY, _PC_ENEMY),
                         (POTION_ALL_ENEMIES, _PC_ALL)):
            for p in names:
                out[self.vid("potion", p)] = c
        return out

    @cached_property
    def fortifier(self) -> int:
        return self.vid("potion", "FORTIFIER")

    @cached_property
    def weak_potion(self) -> int:
        return self.vid("potion", "WEAK_POTION")

    @cached_property
    def card_defence(self) -> np.ndarray:
        """(card id, upgrade level) -> _DEF_*: whom the card's immediate defence (catalog ``gains_block``, or a Summon
        var other than INVOKE's next-turn Summon) protects: the player, all players (target AllAllies) or the target
        (LIFT, INTERCEPT)."""
        cards = self._cards()
        out = np.zeros((self.vocab.size("card"), max(len(e) for e in cards.values())), np.int8)
        target = {self.vid("card", c) for c in ALLY_GUARD_CARDS}
        delayed = {self.vid("card", c) for c in DELAYED_SUMMON}
        for i, levels in cards.items():
            for lv in range(out.shape[1]):
                e = levels[min(lv, len(levels) - 1)]
                if not (e["gains_block"] or ("Summon" in e["vars"] and i not in delayed)):
                    continue
                out[i, lv] = _DEF_TARGET if i in target else _DEF_ALL if e["target_type"] == "AllAllies" else _DEF_SELF
        return out

    @cached_property
    def card_attack_one(self) -> np.ndarray:
        """card id -> an Attack aimed at one enemy (target AnyEnemy)"""
        out = np.zeros(self.vocab.size("card"), bool)
        for i, levels in self._cards().items():
            out[i] = levels[0]["type"] == "Attack" and levels[0]["target_type"] == "AnyEnemy"
        return out

    @cached_property
    def card_weak_one(self) -> np.ndarray:
        """card id -> applies Weak (a WeakPower var) to one enemy (target AnyEnemy)"""
        out = np.zeros(self.vocab.size("card"), bool)
        for i, levels in self._cards().items():
            out[i] = "WeakPower" in levels[0]["vars"] and levels[0]["target_type"] == "AnyEnemy"
        return out

    @cached_property
    def card_eot(self) -> np.ndarray:
        """card id -> deals damage / HP loss at the end of its owner's turn while in hand (catalog
        has_turn_end_in_hand_effect: BURN, DECAY, INFECTION, TOXIC, WITHER deal shown Damage, BAD_LUCK / BECKON shown
        HpLoss)"""
        out = np.zeros(self.vocab.size("card"), bool)
        for i, levels in self._cards().items():
            out[i] = levels[0]["has_turn_end_in_hand_effect"]
        return out

    @cached_property
    def eot_vars(self) -> tuple[int, int]:
        return self.vid("var_name", "Damage"), self.vid("var_name", "HpLoss")

    # ------------------------------------------------------------------ event tables (notes/r3/events.md)
    @cached_property
    def relic_option(self) -> int:
        return self.vid("event_option_key", "RELIC_OPTION")

    @cached_property
    def relic_section(self) -> int:
        return self.vid("shop_section", "RELIC")

    @cached_property
    def relic_names(self) -> list[str]:
        return self.vocab["relic"].entries

    @cached_property
    def page_names(self) -> list[str]:
        return [short_page(x) for x in self.vocab["event_page"].entries]

    @cached_property
    def opt_names(self) -> list[str]:
        return [short_option(x) for x in self.vocab["event_option_key"].entries]

    @cached_property
    def reward_names(self) -> list[str]:
        return self.vocab["enum:RewardType"].entries

    @cached_property
    def section_names(self) -> list[str]:
        return self.vocab["shop_section"].entries

    @cached_property
    def alt_names(self) -> list[str]:
        return self.vocab["card_reward_alt"].entries

    @cached_property
    def confirm_names(self) -> list[str]:
        return self.vocab["confirm_key"].entries


CLASS_NAMES = ("other", "curse", "strike", "defend")


@dataclass
class Call:
    """One override call: the env, the frame's sample, and whether the frame is a DECK_REMOVAL prompt opened by buying
    the shop's card removal (``HabitOverride`` keeps that state per env)."""
    env_id: int
    sample: Sample
    removal_bought: bool = False


def allowed_rows(f: Frame, banned: Sequence[int]) -> np.ndarray:
    """``ok`` of ``judge``: every ACTIONS row of the frame is legal; the cycle-ban / commit mask removes ``banned``."""
    ok = np.ones(f.n_actions, bool)
    ok[list(banned)] = False
    return ok


def _rows(f: Frame, verb: int, ok: np.ndarray) -> np.ndarray:
    """Allowed ACTIONS rows of ``verb``."""
    return np.flatnonzero((f.actions[:, ActionCol.VERB] == verb) & ok)


def _best(rows: np.ndarray, scores: np.ndarray) -> int:
    assert len(rows), "no candidate row"
    return int(rows[np.argmax(scores[rows])])


def _option_rows(f: Frame, ok: np.ndarray, kind: int, key: int) -> np.ndarray:
    rows = _rows(f, Verb.CHOOSE_OPTION, ok)
    acts = f.actions
    opts = f.table(Table.OPTIONS)
    o = opts[acts[rows, ActionCol.SRC_ROW]]
    return rows[(o[:, OptionCol.KIND] == kind) & (o[:, OptionCol.KEY_ID] == key)]


def _is_option(f: Frame, action: int, kind: int) -> bool:
    """``action`` is a CHOOSE_OPTION row of an OPTIONS row of ``kind`` (the choice set of the option habits)."""
    a = f.actions[action]
    return bool(a[_VERB] == _CHOOSE_OPTION and f.table(Table.OPTIONS)[a[_SRC_ROW], OptionCol.KIND] == kind)


def deck_classes(f: Frame, names: _Names, seat: int) -> Counter[int]:
    """Copies per card class (``CLASS_NAMES`` index) in ``seat``'s deck (out-of-combat frame)."""
    cards = f.table(Table.CARDS)
    deck = cards[(cards[:, CardCol.ZONE] == Zone.DECK) & (cards[:, CardCol.SEAT] == seat)]
    assert len(deck), "an out-of-combat frame carries every seat's deck"
    out: Counter[int] = Counter()
    for cid, n in zip(deck[:, CardCol.CARD_ID].tolist(), deck[:, CardCol.COUNT].tolist()):
        out[int(names.card_class[cid])] += n
    return out


def _upgrade_prompt(n: _Names, g: tuple[int, ...]) -> bool:
    return (g[_KIND] == _CARD_SELECT and g[_KIND_SUB] == CardSelectSub.DECK_UPGRADE and not g[_IN_COMBAT]
            and g[_ROOM] == n.room["RestSite"])


@dataclass
class Smith:
    t: float
    name: str = "smith"
    KINDS = frozenset((Kind.REST, Kind.CARD_SELECT))  # the decision kinds ``judge`` can grade
    VERBS = frozenset((_CHOOSE_OPTION, _CONFIRM, _CANCEL))  # the verbs of the actions ``judge`` can grade non-zero

    def threshold(self, n: _Names, f: Frame, g: tuple[int, ...]) -> float:
        return self.t

    def _offer(self, n: _Names, f: Frame, g: tuple[int, ...], ok: np.ndarray) -> tuple[int, bool] | None:
        """REST frame: (the allowed SMITH row, HP >= threshold * max HP), or None without such a row."""
        rows = _option_rows(f, ok, OptionKind.REST_OPTION, n.smith)
        if not len(rows):
            return None
        assert len(rows) == 1, "one SMITH option per seat"
        seat = f.table(Table.SEATS)[g[_SEAT]]
        return int(rows[0]), bool(seat[SeatCol.HP] >= self.threshold(n, f, g) * seat[SeatCol.MAX_HP])

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if g[_KIND] == _REST:
            offer = self._offer(n, f, g, np.isfinite(scores))
            return offer[0] if offer is not None and offer[1] else None
        if _upgrade_prompt(n, g):
            confirm = _rows(f, Verb.CONFIRM_SELECTION, np.isfinite(scores))
            return int(confirm[0]) if len(confirm) else None
        return None

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        if g[_KIND] == _REST:
            offer = self._offer(n, f, g, allowed_rows(f, banned))
            if offer is None or not _is_option(f, action, OptionKind.REST_OPTION):
                return 0
            return (1 if action == offer[0] else -1) if offer[1] else (-1 if action == offer[0] else 1)
        if _upgrade_prompt(n, g):
            confirm = _rows(f, Verb.CONFIRM_SELECTION, allowed_rows(f, banned))
            if not len(confirm):
                return 0
            return 1 if action == int(confirm[0]) else -1 if f.actions[action, _VERB] == _CANCEL else 0
        return 0


@dataclass
class SmithPre(Smith):
    t_pre: float = 1.0
    name: str = "smith"

    def threshold(self, n: _Names, f: Frame, g: tuple[int, ...]) -> float:
        nodes = f.table(Table.MAP_NODES)
        boss = nodes[nodes[:, MapNodeCol.POINT_TYPE] == n.boss_point]
        assert len(boss), "an out-of-combat frame carries the act's map with its Boss node"
        return self.t_pre if g[G.CUR_ROW] == boss[:, MapNodeCol.ROW].min() - 1 else self.t


@dataclass
class HealLate:
    t: float
    act: int = 3  # 1-based, 0 = every act
    rows: int = 1
    name: str = "heal_late"
    KINDS = frozenset((Kind.REST,))
    VERBS = frozenset((_CHOOSE_OPTION,))

    def _heal(self, n: _Names, f: Frame, g: tuple[int, ...], ok: np.ndarray) -> int | None:
        """REST frame of the habit (act, rows below the Boss node, HP < t * max HP): the allowed HEAL row, else None."""
        if g[_KIND] != _REST or (self.act and g[G.ACT_INDEX] + 1 != self.act):
            return None
        rows = _option_rows(f, ok, OptionKind.REST_OPTION, n.heal)
        if not len(rows):
            return None
        assert len(rows) == 1, "one HEAL option per seat"
        nodes = f.table(Table.MAP_NODES)
        boss = nodes[nodes[:, MapNodeCol.POINT_TYPE] == n.boss_point]
        assert len(boss), "an out-of-combat frame carries the act's map with its Boss node"
        if g[G.CUR_ROW] < boss[:, MapNodeCol.ROW].min() - self.rows:
            return None
        seat = f.table(Table.SEATS)[g[_SEAT]]
        return int(rows[0]) if seat[SeatCol.HP] < self.t * seat[SeatCol.MAX_HP] else None

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        return self._heal(n, f, g, np.isfinite(scores))

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        heal = self._heal(n, f, g, allowed_rows(f, banned))
        if heal is None or not _is_option(f, action, OptionKind.REST_OPTION):
            return 0
        return 1 if action == heal else -1


@dataclass
class ShopPrefer:
    section: str
    name: str = "shop_prefer"
    KINDS = frozenset((Kind.SHOP,))
    VERBS = frozenset((_CHOOSE_OPTION,))

    def _rows(self, n: _Names, f: Frame, ok: np.ndarray) -> np.ndarray:
        return _option_rows(f, ok, OptionKind.SHOP_ENTRY, n.shop[self.section])

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if g[_KIND] != _SHOP:
            return None
        rows = self._rows(n, f, np.isfinite(scores))
        return _best(rows, scores) if len(rows) and greedy not in rows else None

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        if g[_KIND] != _SHOP:
            return 0
        rows = self._rows(n, f, allowed_rows(f, banned))
        if not len(rows):
            return 0
        return 1 if action in rows else -1 if _is_option(f, action, OptionKind.SHOP_ENTRY) else 0


@dataclass
class ShopBan:
    sections: tuple[str, ...]
    name: str = "shop_ban"
    KINDS = frozenset((Kind.SHOP,))
    VERBS = frozenset((int(Verb.CHOOSE_OPTION),))

    def _banned(self, n: _Names, f: Frame, ok: np.ndarray) -> np.ndarray:
        return np.concatenate([_option_rows(f, ok, OptionKind.SHOP_ENTRY, n.shop[s]) for s in self.sections])

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if g[_KIND] != _SHOP:
            return None
        banned = self._banned(n, f, np.isfinite(scores))
        if greedy not in banned:
            return None
        allowed = np.isfinite(scores)
        allowed[banned] = False
        return _best(np.flatnonzero(allowed), scores)  # leaving the shop is always legal

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        if g[_KIND] != _SHOP:
            return 0
        return -1 if action in self._banned(n, f, allowed_rows(f, banned)) else 0


@dataclass
class NoElite:
    acts: tuple[int, int]  # 1-based, inclusive
    name: str = "no_elite"
    KINDS = frozenset((Kind.MAP,))
    VERBS = frozenset((int(Verb.TRAVEL),))

    def _split(self, n: _Names, f: Frame, g: tuple[int, ...], ok: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(allowed TRAVEL rows, which of them target an Elite) of a MAP frame in the acts with both kinds, else None."""
        if g[_KIND] != _MAP or not self.acts[0] <= g[G.ACT_INDEX] + 1 <= self.acts[1]:
            return None
        rows = _rows(f, Verb.TRAVEL, ok)
        elite = map_targets_elite(f, rows, n)
        return (rows, elite) if elite.any() and (~elite).any() else None

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        split = self._split(n, f, g, np.isfinite(scores))
        if split is None:
            return None
        rows, elite = split
        return _best(rows[~elite], scores) if greedy in rows[elite] else None

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        if g[_KIND] != _MAP:
            return 0
        split = self._split(n, f, g, allowed_rows(f, banned))
        return -1 if split is not None and action in split[0][split[1]] else 0


@dataclass
class PotionHold:
    rooms: tuple[str, ...]
    name: str = "potion_hold"
    KINDS = frozenset((Kind.COMBAT,))
    VERBS = frozenset((_USE_POTION,))

    def _in_room(self, n: _Names, g: tuple[int, ...]) -> bool:
        return g[_KIND] == _COMBAT and g[_ROOM] in [n.room[r] for r in self.rooms]

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if not self._in_room(n, g) or f.actions[greedy, ActionCol.VERB] != Verb.USE_POTION:
            return None
        verbs = f.actions[:, ActionCol.VERB]
        other = np.flatnonzero((verbs != Verb.USE_POTION) & (verbs != Verb.DISCARD_POTION) & np.isfinite(scores))
        return _best(other, scores)  # END_TURN is always legal in a COMBAT frame of a seat that can act

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        # the action's verb first: almost every COMBAT step ends here
        return -1 if f.actions[action, _VERB] == _USE_POTION and self._in_room(n, g) else 0


@dataclass
class PotionRound1:
    rooms: tuple[str, ...]
    name: str = "potion_r1"
    KINDS = frozenset((Kind.COMBAT,))
    VERBS = frozenset((_USE_POTION,))

    def _round1(self, n: _Names, g: tuple[int, ...]) -> bool:
        return g[_KIND] == _COMBAT and g[_ROUND] == 1 and g[_ROOM] in [n.room[r] for r in self.rooms]

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if not self._round1(n, g) or f.actions[greedy, ActionCol.VERB] == Verb.USE_POTION:
            return None
        use = _rows(f, Verb.USE_POTION, np.isfinite(scores))
        return _best(use, scores) if len(use) else None

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        # positive-only: the taken USE_POTION row is allowed, so "an allowed USE_POTION row exists" holds
        return 1 if f.actions[action, _VERB] == _USE_POTION and self._round1(n, g) else 0


@dataclass
class Enrage:
    mode: str  # "none": no Skill at all; "block": Skills that gain block stay allowed
    name: str = "enrage"
    KINDS = frozenset((Kind.COMBAT,))
    VERBS = frozenset((_PLAY_CARD,))

    def _forbidden(self, n: _Names) -> np.ndarray:
        return n.card_skill if self.mode == "none" else n.card_skill_no_block

    @staticmethod
    def _enraged(n: _Names, f: Frame) -> bool:
        powers = f.table(Table.POWERS)
        return bool(((powers[:, PowerCol.POWER_ID] == n.enrage) & (powers[:, PowerCol.AMOUNT] > 0)).any())

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if g[_KIND] != _COMBAT or not self._enraged(n, f):
            return None
        acts = f.actions
        play = np.flatnonzero(acts[:, ActionCol.VERB] == Verb.PLAY_CARD)
        assert (acts[play, ActionCol.SRC_TABLE] == Table.CARDS).all()
        ids = f.table(Table.CARDS)[acts[play, ActionCol.SRC_ROW], CardCol.CARD_ID]
        banned = play[self._forbidden(n)[ids]]
        if greedy not in banned:
            return None
        allowed = np.isfinite(scores)
        allowed[banned] = False
        return _best(np.flatnonzero(allowed), scores)  # END_TURN is always legal for a seat that can act

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        # the played card first (cheap), the POWERS scan only for a forbidden Skill
        a = f.actions[action]
        if g[_KIND] != _COMBAT or a[_VERB] != _PLAY_CARD:
            return 0
        if not self._forbidden(n)[f.table(Table.CARDS)[a[_SRC_ROW], CardCol.CARD_ID]]:
            return 0
        return -1 if self._enraged(n, f) else 0


@dataclass
class DeckCap:
    cap: int
    name: str = "deck_cap"
    KINDS = frozenset((Kind.CARD_REWARD,))
    VERBS = frozenset((_CHOOSE_OPTION, _PICK_OFFER_CARD))

    def skip(self, n: _Names, f: Frame, g: tuple[int, ...], ok: np.ndarray) -> int | None:
        """CARD_REWARD frame of a seat whose deck holds >= cap cards: its allowed Skip row, else None."""
        if g[_KIND] != _CARD_REWARD or f.table(Table.SEATS)[g[_SEAT], SeatCol.DECK_COUNT] < self.cap:
            return None
        rows = _option_rows(f, ok, OptionKind.CARD_REWARD_ALT, n.card_skip)  # the Skip button (Rewards.cs)
        assert len(rows) <= 1, "one Skip per card reward"
        return int(rows[0]) if len(rows) else None

    def applies(self, n: _Names, f: Frame, g: tuple[int, ...], banned: Sequence[int]) -> bool:
        """The habit decides the frame (precedence over prefs, ``spire.advice.YIELDS``)."""
        return self.skip(n, f, g, allowed_rows(f, banned)) is not None

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        return self.skip(n, f, g, np.isfinite(scores))

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        skip = self.skip(n, f, g, allowed_rows(f, banned))
        if skip is None:
            return 0
        return 1 if action == skip else -1 if f.actions[action, _VERB] == _PICK_OFFER_CARD else 0


@dataclass
class Prefs:
    prefer: tuple[str, ...]
    avoid: tuple[str, ...]
    name: str = "prefs"
    KINDS = frozenset((Kind.CARD_REWARD,))
    VERBS = frozenset((_CHOOSE_OPTION, _PICK_OFFER_CARD))
    _table: tuple[_Names, np.ndarray] | None = field(default=None, compare=False, repr=False)

    @staticmethod
    def parse(arg: str) -> "Prefs":
        """``file:<json>`` or ``list:<prefer>/<avoid>`` (module docstring)."""
        mode, _, rest = arg.partition(":")
        if mode == "file":
            d = json.loads(open(rest, encoding="utf-8").read())
            assert set(d) <= {"prefer", "avoid", "note"}, d.keys()
            prefer, avoid = tuple(d.get("prefer", [])), tuple(d.get("avoid", []))
        else:
            assert mode == "list" and rest.count("/") == 1, f"prefs:{arg}: expected file:<json> or list:<prefer>/<avoid>"
            p, a = rest.split("/")
            prefer, avoid = tuple(filter(None, p.split(","))), tuple(filter(None, a.split(",")))
        assert prefer or avoid, f"prefs:{arg}: an empty table"
        assert len(set(prefer)) == len(prefer) and len(set(avoid)) == len(avoid), f"prefs:{arg}: a repeated card"
        assert not set(prefer) & set(avoid), f"prefs:{arg}: {sorted(set(prefer) & set(avoid))} both PREFER and AVOID"
        return Prefs(prefer, avoid)

    def values(self, n: _Names) -> np.ndarray:
        """card id -> +1 PREFER, -1 AVOID, 0 (resolved on first use: an unknown card name fails here)."""
        if self._table is None or self._table[0] is not n:
            t = np.zeros(n.vocab.size("card"), np.int8)
            t[[n.vid("card", c) for c in self.prefer]] = 1
            t[[n.vid("card", c) for c in self.avoid]] = -1
            self._table = (n, t)
        return self._table[1]

    def _offer(self, n: _Names, f: Frame, g: tuple[int, ...], ok: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """CARD_REWARD frame: (allowed PREFER pick rows, allowed AVOID pick rows), else None."""
        if g[_KIND] != _CARD_REWARD:
            return None
        picks = _rows(f, Verb.PICK_OFFER_CARD, ok)
        acts = f.actions
        assert (acts[picks, ActionCol.SRC_TABLE] == Table.CARDS).all()
        v = self.values(n)[f.table(Table.CARDS)[acts[picks, _SRC_ROW], CardCol.CARD_ID]]
        return picks[v > 0], picks[v < 0]

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        offer = self._offer(n, f, g, np.isfinite(scores))
        if offer is None:
            return None
        prefer, avoid = offer
        if len(prefer):
            return _best(prefer, scores)
        if greedy not in avoid:
            return None
        allowed = np.isfinite(scores)
        allowed[avoid] = False
        rows = np.flatnonzero(allowed)
        return _best(rows, scores) if len(rows) else None  # only AVOID picks allowed: the policy's choice

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        ok = allowed_rows(f, banned)
        offer = self._offer(n, f, g, ok)
        if offer is None:
            return 0
        prefer, avoid = offer
        if len(prefer):
            if action in prefer:
                return 1
            other = f.actions[action, _VERB] == _PICK_OFFER_CARD  # another pick, or the Skip; REROLL / SACRIFICE 0
            return -1 if other or action in _option_rows(f, ok, OptionKind.CARD_REWARD_ALT, n.card_skip) else 0
        return -1 if action in avoid and ok.sum() > len(avoid) else 0


@dataclass
class RemoveBasic:
    name: str = "remove_basic"

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if g[_KIND] == _SHOP:
            rows = _option_rows(f, np.isfinite(scores), OptionKind.SHOP_ENTRY, n.removal)
            if not len(rows):
                return None
            assert len(rows) == 1, "one card-removal entry per shop"
            classes = deck_classes(f, n, g[_SEAT])
            return int(rows[0]) if classes[1] or classes[2] or classes[3] else None
        if x.removal_bought:
            ok = np.isfinite(scores)
            confirm = _rows(f, Verb.CONFIRM_SELECTION, ok)
            if g[G.SEL_COUNT] > 0:
                return int(confirm[0]) if len(confirm) else None
            sel = _rows(f, Verb.SELECT_CARD, ok)
            cards = f.table(Table.CARDS)[f.actions[sel, ActionCol.SRC_ROW]]
            cls = n.card_class[cards[:, CardCol.CARD_ID]]
            for c in (1, 2, 3):
                if (cls == c).any():
                    return _best(sel[cls == c], scores)
            return None  # the shop decision did not trigger (no curse / Strike / Defend): the policy's prompt
        return None


def removal_prompt(n: _Names, g: tuple[int, ...]) -> bool:
    """A DECK_REMOVAL prompt of a Shop room (the card removal's, or one of e.g. a relic bought there)."""
    return (g[_KIND] == _CARD_SELECT and g[_KIND_SUB] == CardSelectSub.DECK_REMOVAL and not g[_IN_COMBAT]
            and g[_ROOM] == n.room["Shop"])


def bought_removal(n: _Names, f: Frame, g: tuple[int, ...], action: int) -> bool:
    """``action`` buys the shop's card removal."""
    if g[_KIND] != _SHOP or f.actions[action, ActionCol.VERB] != Verb.CHOOSE_OPTION:
        return False
    o = f.table(Table.OPTIONS)[f.actions[action, ActionCol.SRC_ROW]]
    return bool(o[OptionCol.KIND] == OptionKind.SHOP_ENTRY and o[OptionCol.KEY_ID] == n.removal)


def map_targets_elite(f: Frame, rows: np.ndarray, n: _Names) -> np.ndarray:
    nodes = f.table(Table.MAP_NODES)
    assert (f.actions[rows, ActionCol.SRC_TABLE] == Table.MAP_NODES).all()
    return nodes[f.actions[rows, ActionCol.SRC_ROW], MapNodeCol.POINT_TYPE] == n.elite_point


# =========================================================================== tactical combat habits (notes/r3/tactics.md)
#
# What a COMBAT frame shows and these habits read: per seat HP, block and life (SEATS); per creature kind, seat, HP,
# block, life (CREATURES; OSTY is the NECROBINDER's PET row of the seat); per living enemy its intents (INTENTS: type,
# hits, shown damage per seat, -1 for non-attacks); the cards in hand with their shown numbers (VARS; per-target previews
# with target_row >= 0 where they differ from the untargeted value); potions (POTIONS); the legal actions (ACTIONS:
# verb, acting seat, source row, target creature row).
#
# Incoming damage of seat s this enemy turn: T_s = sum over the intents of the living enemies of shown damage[s] x hits
# (every monster attack hits every player, AttackCommand.FromMonster -> TargetingAllOpponents). It is absorbed by the
# seat's block, then by its living OSTY (DieForYouPower redirects unblocked powered damage, overkill to the owner):
# damage reaching HP = max(0, T_s - block_s - osty_hp_s). Seat s "faces lethal" at factor f when it is alive, T_s > 0
# and damage reaching HP >= f x HP (f = 1: it dies unless something changes).
# Danger of an enemy = its intents' damage x hits summed over the living seats (the team damage it deals this turn).

@dataclass
class Threat:
    """The incoming attack of one COMBAT frame (see above): per seat, the damage reaching HP and the HP; per creature
    row, the enemy's danger (0 for non-enemies); the creature row -> seat map of the players."""
    to_hp: np.ndarray
    hp: np.ndarray
    alive: np.ndarray
    attacked: np.ndarray  # T_s > 0
    danger: np.ndarray
    hits_seat: np.ndarray  # creature row x seat: that enemy's attack damage on the seat this turn
    seat_of: np.ndarray  # creature row -> seat of a PLAYER / PET row, -1 for enemies

    def lethal(self, f: float) -> np.ndarray:
        return self.alive & self.attacked & (self.to_hp >= f * self.hp)


def eot_damage(n: _Names, f: Frame, ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Per seat: (blockable Damage, unblockable HpLoss) its hand cards deal it at the end of its turn (``card_eot``)."""
    cards = f.table(Table.CARDS)
    dv, lv = n.eot_vars
    blockable, loss = np.zeros(ns, np.int64), np.zeros(ns, np.int64)
    for r in np.flatnonzero((cards[:, CardCol.ZONE] == Zone.HAND) & n.card_eot[cards[:, CardCol.CARD_ID]]):
        s = int(cards[r, CardCol.SEAT])
        blockable[s] += _shown(f, Table.CARDS, int(r), dv, -1) or 0
        loss[s] += _shown(f, Table.CARDS, int(r), lv, -1) or 0
    return blockable, loss


def threat(n: _Names, f: Frame, eot: bool = False) -> Threat:
    """``eot``: the seat's end-of-turn hand damage comes first (``eot_damage``: Damage absorbed by block, HpLoss not),
    then the attacks (remaining block, then Osty)."""
    seats, cr, it = f.table(Table.SEATS), f.table(Table.CREATURES), f.table(Table.INTENTS)
    ns = len(seats)
    dmg = it[:, IntentCol.DMG_SEAT0:IntentCol.DMG_SEAT0 + ns].astype(np.int64)
    per = np.maximum(dmg, 0) * np.maximum(it[:, IntentCol.HITS], 1)[:, None]
    owner = it[:, IntentCol.OWNER_ROW]
    assert (cr[owner, CreatureCol.KIND] == _ENEMY).all() and cr[owner, CreatureCol.ALIVE].all(), "intents of live enemies"
    hits_seat = np.zeros((len(cr), ns), np.int64)
    np.add.at(hits_seat, owner, per)
    total = hits_seat.sum(axis=0)
    osty = np.zeros(ns, np.int64)
    pets = (cr[:, CreatureCol.KIND] == _PET) & (cr[:, CreatureCol.MODEL_ID] == n.osty) & (cr[:, CreatureCol.ALIVE] > 0)
    for row in np.flatnonzero(pets):
        osty[cr[row, CreatureCol.SEAT]] += max(int(cr[row, CreatureCol.HP]), 0)
    hp = seats[:, SeatCol.HP].astype(np.int64)
    block = seats[:, SeatCol.BLOCK].astype(np.int64)
    if eot:
        d, loss = eot_damage(n, f, ns)
        to_hp = loss + np.maximum(d - block, 0) + np.maximum(total - np.maximum(block - d, 0) - osty, 0)
    else:
        to_hp = np.maximum(total - block - osty, 0)
    alive = (seats[:, SeatCol.ALIVE] > 0) & (hp > 0)
    danger = (hits_seat * alive[None, :]).sum(axis=1)
    seat_of = np.where(cr[:, CreatureCol.KIND] == _ENEMY, -1, cr[:, CreatureCol.SEAT])
    return Threat(to_hp, hp, alive, total > 0, danger, hits_seat, seat_of)


def _shown(f: Frame, table: int, row: int, var: int, target: int) -> int | None:
    """Shown value of ``var`` of an entity row: the per-target preview for creature row ``target`` if the frame has one,
    else the untargeted value; None if the entity has no such var."""
    va = f.table(Table.VARS)
    m = (va[:, VarCol.OWNER_TABLE] == table) & (va[:, VarCol.OWNER_ROW] == row) & (va[:, VarCol.VAR_ID] == var)
    rows = va[m]
    if not len(rows):
        return None
    t = rows[rows[:, VarCol.TARGET_ROW] == target]
    return int((t if len(t) else rows[rows[:, VarCol.TARGET_ROW] == -1])[0, VarCol.SHOWN])


def _card_of(f: Frame, row: int) -> tuple[int, int]:
    c = f.table(Table.CARDS)[f.actions[row, ActionCol.SRC_ROW]]
    return int(c[CardCol.CARD_ID]), int(c[CardCol.UPGRADE_LEVEL])


def guard_rows(n: _Names, f: Frame, t: Threat, need: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Allowed rows that add defence this turn to a seat with ``need``: a card of the seat with immediate defence for the
    player (its shown Block / CalculatedBlock / Summon > 0 where it has one), an all-player defence card of any seat, a
    LIFT / INTERCEPT aimed at the seat, a POTION_GUARD potion (of anyone) aimed at the seat (FORTIFIER: seat block > 0)."""
    acts = f.actions
    out = []
    v = n.tvars
    seats = f.table(Table.SEATS)
    for r in np.flatnonzero(ok & ((acts[:, ActionCol.VERB] == Verb.PLAY_CARD) | (acts[:, ActionCol.VERB] == Verb.USE_POTION))):
        a = acts[r]
        tgt = int(a[ActionCol.TGT_ROW])
        if a[ActionCol.VERB] == Verb.PLAY_CARD:
            cid, lv = _card_of(f, r)
            d = n.card_defence[cid, min(lv, n.card_defence.shape[1] - 1)]
            if d == _DEF_NONE:
                continue
            shown = [x for x in (_shown(f, Table.CARDS, int(a[ActionCol.SRC_ROW]), v[k], tgt)
                                 for k in ("Block", "CalculatedBlock", "Summon")) if x is not None]
            if shown and sum(shown) <= 0:
                continue
            if d == _DEF_SELF:
                covered = need[a[ActionCol.SEAT]]
            elif d == _DEF_ALL:
                covered = need.any()
            else:
                covered = tgt >= 0 and t.seat_of[tgt] >= 0 and need[t.seat_of[tgt]]
        else:
            pid = int(f.table(Table.POTIONS)[a[ActionCol.SRC_ROW], PotionCol.POTION_ID])
            if n.potion_class[pid] != _PC_GUARD or tgt < 0 or t.seat_of[tgt] < 0:
                continue
            s = t.seat_of[tgt]
            covered = need[s] and (pid != n.fortifier or seats[s, SeatCol.BLOCK] > 0)
        if covered:
            out.append(r)
    return np.array(out, np.int64)


def saving_potion_rows(n: _Names, f: Frame, t: Threat, s: int, ok: np.ndarray) -> np.ndarray:
    """Allowed USE_POTION rows of seat ``s``'s own potions that prevent or heal damage for ``s``: POTION_GUARD /
    POTION_HEAL aimed at ``s`` (FORTIFIER: block > 0), POTION_ENEMY aimed at an enemy that attacks ``s``,
    POTION_ALL_ENEMIES while some enemy attacks ``s``."""
    acts = f.actions
    rows = np.flatnonzero(ok & (acts[:, ActionCol.VERB] == Verb.USE_POTION) & (acts[:, ActionCol.SEAT] == s))
    pots = f.table(Table.POTIONS)
    out = []
    for r in rows:
        pid = int(pots[acts[r, ActionCol.SRC_ROW], PotionCol.POTION_ID])
        c = n.potion_class[pid]
        tgt = int(acts[r, ActionCol.TGT_ROW])
        if c in (_PC_GUARD, _PC_HEAL):
            good = tgt >= 0 and t.seat_of[tgt] == s and (pid != n.fortifier or f.table(Table.SEATS)[s, SeatCol.BLOCK] > 0)
        elif c == _PC_ENEMY:
            good = tgt >= 0 and t.seat_of[tgt] < 0 and t.hits_seat[tgt, s] > 0
        else:
            good = c == _PC_ALL and bool(t.hits_seat[:, s].any())
        if good:
            out.append(r)
    return np.array(out, np.int64)


@dataclass
class LethalBlock:
    """``lethal_block[:<f>]``: COMBAT frame in which some seat faces lethal at factor f (default 1) and an allowed row
    adds defence to such a seat (``guard_rows``): when the policy's choice is not such a row, its best such row.
    judge: +1 for such a row, -1 for another PLAY_CARD or an END_TURN at such a frame (its choice set: what the seats
    do with their turn; a potion use that is not a guard row and a discard are 0, notes/advfix). ``lethal_eot`` (habit
    6, from the loss reading): the same at f = 1 with the end-of-turn hand damage counted (``threat(eot=True)``)."""
    f: float = 1.0
    eot: bool = False
    name: str = "lethal_block"
    KINDS = frozenset((Kind.COMBAT,))
    VERBS = frozenset((_PLAY_CARD, _END_TURN, _USE_POTION))

    def _rows(self, n: _Names, f: Frame, g: tuple[int, ...], ok: np.ndarray) -> np.ndarray | None:
        if g[_KIND] != _COMBAT or not len(f.table(Table.INTENTS)):
            return None
        t = threat(n, f, self.eot)
        need = t.lethal(self.f)
        if not need.any():
            return None
        rows = guard_rows(n, f, t, need, ok)
        return rows if len(rows) else None

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        rows = self._rows(n, f, g, np.isfinite(scores))
        return _best(rows, scores) if rows is not None and greedy not in rows else None

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        verb = f.actions[action, _VERB]
        if verb not in self.VERBS:
            return 0
        rows = self._rows(n, f, g, allowed_rows(f, banned))
        return 0 if rows is None else 1 if action in rows else -1 if verb != _USE_POTION else 0


def _same_source(f: Frame, row: int, ok: np.ndarray) -> np.ndarray:
    """Allowed rows of the same verb, seat and source as ``row`` that aim at an enemy creature."""
    acts = f.actions
    cr = f.table(Table.CREATURES)
    m = ok & (acts[:, ActionCol.VERB] == acts[row, ActionCol.VERB]) & (acts[:, ActionCol.SEAT] == acts[row, ActionCol.SEAT]) \
        & (acts[:, ActionCol.SRC_ROW] == acts[row, ActionCol.SRC_ROW]) & (acts[:, ActionCol.TGT_ROW] >= 0)
    rows = np.flatnonzero(m)
    return rows[cr[acts[rows, ActionCol.TGT_ROW], CreatureCol.KIND] == _ENEMY]


def _aim(f: Frame, t: Threat, rows: np.ndarray) -> np.ndarray | None:
    """Of target rows of one card / potion: those aimed at the most dangerous target, or None when there is no choice
    (fewer than two targets, no target attacks, or all targets are equally dangerous)."""
    tg = f.actions[rows, ActionCol.TGT_ROW]
    if len(np.unique(tg)) < 2:
        return None
    d = t.danger[tg]
    if d.max() <= 0 or d.min() == d.max():
        return None
    return rows[d == d.max()]


@dataclass
class KillDangerous:
    """``kill_dangerous`` (``kill_dangerous:<m>``: evaluation only, re-aim only when the policy scores the new row within
    m of its choice): COMBAT frame, the policy's choice (judge: the action) plays an Attack card aimed at one enemy
    (catalog type Attack, target AnyEnemy) with allowed rows on at least two enemies that differ in danger; D = the
    rows on the most dangerous of them. The card kills a target if its shown damage there (per-target preview of
    Damage / CalculatedDamage / OstyDamage) x shown CalculatedHits or HitCount (1 if absent) >= the target's HP + block.
    If the card kills a D target, or kills no target: aim at D (the policy's best D row; killing D rows first). judge:
    such a play +1 aimed at D, -1 aimed elsewhere (its choice set: the rows of that card on the enemies)."""
    margin: float | None = None
    name: str = "kill_dangerous"
    KINDS = frozenset((Kind.COMBAT,))
    VERBS = frozenset((_PLAY_CARD,))

    def _target(self, n: _Names, f: Frame, row: int, ok: np.ndarray) -> np.ndarray | None:
        """The rows the habit asks for when ``row`` is played, or None if it has nothing to say."""
        a = f.actions[row]
        if a[_VERB] != _PLAY_CARD or a[ActionCol.TGT_ROW] < 0 or not len(f.table(Table.INTENTS)):
            return None
        if not n.card_attack_one[_card_of(f, row)[0]]:
            return None
        rows = _same_source(f, row, ok)
        best = _aim(f, threat(n, f), rows)
        if best is None:
            return None
        kills = self._kills(n, f, rows)
        dk = kills[np.isin(rows, best)]
        if dk.any():
            return best[dk]
        return best if not kills.any() else None

    @staticmethod
    def _kills(n: _Names, f: Frame, rows: np.ndarray) -> np.ndarray:
        v = n.tvars
        cr = f.table(Table.CREATURES)
        src = int(f.actions[rows[0], ActionCol.SRC_ROW])
        out = np.zeros(len(rows), bool)
        for i, r in enumerate(rows):
            tg = int(f.actions[r, ActionCol.TGT_ROW])
            dmg = [x for x in (_shown(f, Table.CARDS, src, v[k], tg) for k in ("Damage", "CalculatedDamage", "OstyDamage"))
                   if x is not None]
            hits = [x for x in (_shown(f, Table.CARDS, src, v[k], tg) for k in ("CalculatedHits", "HitCount"))
                    if x is not None]
            total = max(dmg, default=0) * (hits[0] if hits else 1)
            out[i] = total >= cr[tg, CreatureCol.HP] + cr[tg, CreatureCol.BLOCK]
        return out

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if g[_KIND] != _COMBAT:
            return None
        want = self._target(n, f, greedy, np.isfinite(scores))
        if want is None or greedy in want:
            return None
        r = _best(want, scores)
        return r if self.margin is None or scores[r] >= scores[greedy] - self.margin else None

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        assert self.margin is None, "kill_dangerous:<m> is evaluation-only (the margin reads the policy's scores)"
        if g[_KIND] != _COMBAT or f.actions[action, _VERB] != _PLAY_CARD:
            return 0
        want = self._target(n, f, action, allowed_rows(f, banned))
        return 0 if want is None else 1 if action in want else -1


@dataclass
class KillDangerousMargin(KillDangerous):
    """``kill_dangerous:<m>``: evaluation-only (not in spire.advice.JUDGE_NAMES)."""
    name: str = "kill_dangerous"


@dataclass
class WeakAttacker:
    """``weak_attacker``: COMBAT frame, the policy's choice (judge: the action) plays a card that applies Weak to one
    enemy (catalog: WeakPower var, target AnyEnemy) or uses a WEAK_POTION, with allowed rows on at least two enemies that
    differ in danger: aim at the most dangerous one (the policy's best such row). judge: +1 aimed there, -1 elsewhere
    (its choice set: the rows of that card / potion on the enemies)."""
    name: str = "weak_attacker"
    KINDS = frozenset((Kind.COMBAT,))
    VERBS = frozenset((_PLAY_CARD, _USE_POTION))

    def _target(self, n: _Names, f: Frame, row: int, ok: np.ndarray) -> np.ndarray | None:
        a = f.actions[row]
        if a[ActionCol.TGT_ROW] < 0 or not len(f.table(Table.INTENTS)):
            return None
        if a[_VERB] == _PLAY_CARD:
            if not n.card_weak_one[_card_of(f, row)[0]]:
                return None
        elif a[_VERB] == _USE_POTION:
            if f.table(Table.POTIONS)[a[ActionCol.SRC_ROW], PotionCol.POTION_ID] != n.weak_potion:
                return None
        else:
            return None
        return _aim(f, threat(n, f), _same_source(f, row, ok))

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if g[_KIND] != _COMBAT:
            return None
        want = self._target(n, f, greedy, np.isfinite(scores))
        return _best(want, scores) if want is not None and greedy not in want else None

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        if g[_KIND] != _COMBAT or f.actions[action, _VERB] not in self.VERBS:
            return 0
        want = self._target(n, f, action, allowed_rows(f, banned))
        return 0 if want is None else 1 if action in want else -1


@dataclass
class WeakAttackerCards(WeakAttacker):
    """``weak_attacker:cards``: weak_attacker for Weak cards only (no WEAK_POTION; VERBS = PLAY_CARD, so it combines with
    the USE_POTION habits in spire.advice)."""
    VERBS = frozenset((_PLAY_CARD,))

    def _target(self, n: _Names, f: Frame, row: int, ok: np.ndarray) -> np.ndarray | None:
        return None if f.actions[row, _VERB] != _PLAY_CARD else super()._target(n, f, row, ok)


@dataclass
class PotionReserve:
    """``potion_reserve2`` (A10 double boss): COMBAT frame of the FIRST act-3 boss fight (room Boss, act 3, the frame
    names a second boss encounter, BOSSES_WON == 2) in which no seat faces lethal (f = 1): when the policy's choice is
    USE_POTION, its best action that is neither USE_POTION nor DISCARD_POTION. ``potion_reserve2:all``: also no
    USE_POTION at all in act-3 Monster and Elite rooms. judge: -1 for USE_POTION at such a frame."""
    hallway: bool = False
    name: str = "potion_reserve2"
    KINDS = frozenset((Kind.COMBAT,))
    VERBS = frozenset((_USE_POTION,))

    def _holds(self, n: _Names, f: Frame, g: tuple[int, ...]) -> bool:
        if g[_KIND] != _COMBAT or g[G.ACT_INDEX] != 2 or g[G.SECOND_BOSS_ENCOUNTER] <= 1:
            return False
        if g[_ROOM] == n.room["Boss"]:
            assert g[G.BOSSES_WON] in (2, 3), g[G.BOSSES_WON]
            return g[G.BOSSES_WON] == 2 and not (len(f.table(Table.INTENTS)) and threat(n, f).lethal(1.0).any())
        return self.hallway and g[_ROOM] in (n.room["Monster"], n.room["Elite"])

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if f.actions[greedy, ActionCol.VERB] != Verb.USE_POTION or not self._holds(n, f, g):
            return None
        verbs = f.actions[:, ActionCol.VERB]
        other = np.flatnonzero((verbs != Verb.USE_POTION) & (verbs != Verb.DISCARD_POTION) & np.isfinite(scores))
        return _best(other, scores)  # END_TURN is always legal in a COMBAT frame of a seat that can act

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        return -1 if f.actions[action, _VERB] == _USE_POTION and self._holds(n, f, g) else 0


@dataclass
class PotionLethal:
    """``potion_lethal``: COMBAT frame in which seat s faces lethal (f = 1) and holds a potion with an allowed use that
    prevents or heals damage for s (``saving_potion_rows``): when the policy's choice is s's END_TURN, the best such
    use instead. judge: at such a frame with s's END_TURN allowed, -1 for that END_TURN, +1 for such a use (its choice
    set: s's END_TURN and saving uses; other potion uses 0)."""
    name: str = "potion_lethal"
    KINDS = frozenset((Kind.COMBAT,))
    VERBS = frozenset((_END_TURN, _USE_POTION))

    def _rows(self, n: _Names, f: Frame, g: tuple[int, ...], s: int, ok: np.ndarray) -> np.ndarray | None:
        """Seat s's saving potion rows when s faces lethal and its END_TURN is allowed, else None."""
        if g[_KIND] != _COMBAT or not len(f.table(Table.INTENTS)):
            return None
        acts = f.actions
        if not (ok & (acts[:, ActionCol.VERB] == _END_TURN) & (acts[:, ActionCol.SEAT] == s)).any():
            return None
        t = threat(n, f)
        if not t.lethal(1.0)[s]:
            return None
        rows = saving_potion_rows(n, f, t, s, ok)
        return rows if len(rows) else None

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        a = f.actions[greedy]
        if a[_VERB] != _END_TURN:
            return None
        rows = self._rows(n, f, g, int(a[ActionCol.SEAT]), np.isfinite(scores))
        return None if rows is None else _best(rows, scores)

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        a = f.actions[action]
        if a[_VERB] not in self.VERBS or g[_KIND] != _COMBAT:
            return 0
        rows = self._rows(n, f, g, int(a[ActionCol.SEAT]), allowed_rows(f, banned))
        if rows is None:
            return 0
        return -1 if a[_VERB] == _END_TURN else 1 if action in rows else 0


@dataclass
class PotionBoss:
    """``potion_boss_use`` (habit 6, from the loss reading): COMBAT frame of an act-3 Boss room (``potion_boss_use:all``:
    any Boss room; ``potion_boss_use:late``: the second act-3 boss of the A10 double boss, and the first one only while
    some seat faces lethal at f = 1), the policy's choice is a PLAY_CARD or END_TURN of seat s, s has played no card
    this turn (SEATS cards_played_turn == 0) and has an allowed USE_POTION row: the policy's best USE_POTION row of s
    instead (potions are drunk at the start of the seat's turn, before its cards). judge: at such a frame, -1 for s's
    PLAY_CARD / END_TURN, +1 for a USE_POTION of s (its choice set: what s does first in its turn; DISCARD_POTION 0).
    Two-sided on card plays: kept for evaluation; as an advice rule it shares verbs with every other combat rule
    (spire.advice refuses the pairs), the training form is ``potion_boss_take``."""
    mode: str = "act3"  # "act3" | "all" | "late"
    name: str = "potion_boss_use"
    KINDS = frozenset((Kind.COMBAT,))
    VERBS = frozenset((_PLAY_CARD, _END_TURN, _USE_POTION))

    def _boss(self, n: _Names, f: Frame, g: tuple[int, ...]) -> bool:
        if g[_KIND] != _COMBAT or g[_ROOM] != n.room["Boss"]:
            return False
        if self.mode == "all":
            return True
        if g[G.ACT_INDEX] != 2:
            return False
        if self.mode == "act3":
            return True
        assert g[G.SECOND_BOSS_ENCOUNTER] > 1 and g[G.BOSSES_WON] in (2, 3), "potion_boss_use:late needs the double boss"
        return g[G.BOSSES_WON] == 3 or bool(len(f.table(Table.INTENTS)) and threat(n, f).lethal(1.0).any())

    def _rows(self, n: _Names, f: Frame, g: tuple[int, ...], s: int, ok: np.ndarray) -> np.ndarray | None:
        if not self._boss(n, f, g):
            return None
        if f.table(Table.SEATS)[s, SeatCol.CARDS_PLAYED_TURN] != 0:
            return None
        acts = f.actions
        rows = np.flatnonzero(ok & (acts[:, ActionCol.VERB] == _USE_POTION) & (acts[:, ActionCol.SEAT] == s))
        return rows if len(rows) else None

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        a = f.actions[greedy]
        if a[_VERB] not in (_PLAY_CARD, _END_TURN):
            return None
        rows = self._rows(n, f, g, int(a[ActionCol.SEAT]), np.isfinite(scores))
        return None if rows is None else _best(rows, scores)

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        a = f.actions[action]
        if a[_VERB] not in self.VERBS:
            return 0
        rows = self._rows(n, f, g, int(a[ActionCol.SEAT]), allowed_rows(f, banned))
        return 0 if rows is None else 1 if a[_VERB] == _USE_POTION else -1


@dataclass
class PotionBossTake(PotionBoss):
    """``potion_boss_take[:all|:late]``: ``potion_boss_use`` with a one-sided judge for training (+1 for a USE_POTION of
    s at such a frame, 0 for everything else; VERBS = USE_POTION only, so it combines with enrage:none in
    spire.advice); choose is the same."""
    name: str = "potion_boss_take"
    VERBS = frozenset((_USE_POTION,))

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        a = f.actions[action]
        if a[_VERB] != _USE_POTION:
            return 0
        return 0 if self._rows(n, f, g, int(a[ActionCol.SEAT]), allowed_rows(f, banned)) is None else 1


# =========================================================================== event tables (notes/r3/events.md)
def short_page(name: str) -> str:
    """``X.pages.P.description`` -> ``X.P`` (event page vocab names)."""
    return name.replace(".pages.", ".").removesuffix(".description")


def short_option(name: str) -> str:
    """``X.pages.P.options.K`` -> ``K`` (event option key vocab names; keys without ``.options.`` stay whole)."""
    return name.rpartition(".options.")[2]


_POTION_VERBS = (_USE_POTION, _DISCARD_POTION)


def row_idents(n: _Names, f: Frame, g: tuple[int, ...]) -> list[str]:
    """Choice identity of every ACTIONS row of an out-of-combat frame ("" = a potion action, not a choice of the
    room): ``ev:<page>:<option>`` (``ev:<page>:RELIC`` for an event's RELIC_OPTION), ``rest:<option>``,
    ``reward:<type>``, ``cr:PICK`` / ``cr:<alternative>``, ``shop:<section>``, ``treasure:RELIC``, ``bundle:OPTION``,
    ``confirm:<key>``, ``map:<point type>``, ``cs:<sub>:<verb>``, else ``<kind>:<verb>`` (PROCEED, SKIP, ...)."""
    kind = g[_KIND]
    kname = Kind(kind).name.lower()
    opts = f.table(Table.OPTIONS)
    nodes = f.table(Table.MAP_NODES) if kind == _MAP else None
    page = n.page_names[g[G.EVENT_PAGE_KEY]] if kind == _EVENT else ""
    out = []
    for verb, src_row in f.actions[:, [ActionCol.VERB, ActionCol.SRC_ROW]].tolist():
        if verb in _POTION_VERBS:
            out.append("")
        elif verb == Verb.CHOOSE_OPTION:
            o = opts[src_row]
            ok, key = o[OptionCol.KIND], o[OptionCol.KEY_ID]
            if ok == OptionKind.EVENT_OPTION:
                out.append(f"ev:{page}:{'RELIC' if key == n.relic_option else n.opt_names[key]}")
            elif ok == OptionKind.REST_OPTION:
                out.append(f"rest:{n.rest_names[key]}")
            elif ok == OptionKind.REWARD_ITEM:
                out.append(f"reward:{n.reward_names[key]}")
            elif ok == OptionKind.SHOP_ENTRY:
                out.append(f"shop:{n.section_names[key]}")
            elif ok == OptionKind.CARD_REWARD_ALT:
                out.append(f"cr:{n.alt_names[key]}")
            elif ok == OptionKind.TREASURE_RELIC:
                out.append("treasure:RELIC")
            elif ok == OptionKind.BUNDLE:
                out.append("bundle:OPTION")
            else:
                assert ok == OptionKind.CONFIRM_CHOICE, ok
                out.append(f"confirm:{n.confirm_names[key]}")
        elif verb == Verb.PICK_OFFER_CARD:
            out.append("cr:PICK")
        elif verb == Verb.TRAVEL:
            assert nodes is not None
            out.append(f"map:{n.point_names[nodes[src_row, MapNodeCol.POINT_TYPE]]}")
        elif kind == _CARD_SELECT:
            out.append(f"cs:{CardSelectSub(g[_KIND_SUB]).name}:{Verb(verb).name}")
        elif kind == _EVENT:
            out.append(f"ev:{page}:{Verb(verb).name}")
        else:
            out.append(f"{kname}:{Verb(verb).name}")
    return out


def relic_rows(n: _Names, f: Frame, g: tuple[int, ...], ok: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    """(domain, allowed rows, relic ids) of the rows that choose a relic among alternatives: an event's RELIC_OPTION
    rows (domain event: the Ancients, DOLL_ROOM's dolls), the TREASURE_RELIC rows (treasure), the shop's RELIC entries
    (shop); ("", [], []) else."""
    kind = g[_KIND]
    empty = np.zeros(0, np.int64)
    if kind not in (_EVENT, _TREASURE, _SHOP):
        return "", empty, empty
    rows = _rows(f, Verb.CHOOSE_OPTION, ok)
    o = f.table(Table.OPTIONS)[f.actions[rows, ActionCol.SRC_ROW]]
    if kind == _EVENT:
        dom, m = "event", (o[:, OptionCol.KIND] == OptionKind.EVENT_OPTION) & (o[:, OptionCol.KEY_ID] == n.relic_option)
    elif kind == _TREASURE:
        dom, m = "treasure", o[:, OptionCol.KIND] == OptionKind.TREASURE_RELIC
    else:
        dom, m = "shop", (o[:, OptionCol.KIND] == OptionKind.SHOP_ENTRY) & (o[:, OptionCol.KEY_ID] == n.relic_section)
    o = o[m]
    if not len(o):
        return "", empty, empty
    return dom, rows[m], np.array([option_relic(f, x) for x in o], np.int64)


def option_relic(f: Frame, o: np.ndarray) -> int:
    """Relic id of a relic option row: its RELICS row (Ancients, treasure, shop), else its first reference, which is
    a relic (event relic options without a RELICS row, e.g. DOLL_ROOM's dolls)."""
    if o[OptionCol.REF_TABLE] == Table.RELICS:
        return int(f.table(Table.RELICS)[o[OptionCol.REF_ROW], RelicCol.RELIC_ID])
    assert o[OptionCol.REF_TABLE] < 0 and o[OptionCol.REF_CNT] >= 1, ("a relic option names its relic", o.tolist())
    ref = f.table(Table.REFS)[o[OptionCol.REF_OFF]]
    assert ref[RefCol.CATEGORY] == RefCategory.RELIC, ("a relic option's first reference is its relic", o.tolist())
    return int(ref[RefCol.VOCAB_ID])


@dataclass
class _XGame:
    """One game of :class:`Explore`: its table and what came up / fired."""
    seed: str
    page_arm: dict[int, int]  # page id -> -1 (the policy's choice) or the forced option key id (resolved lazily)
    page_forced: np.ndarray  # page id -> forced (rand: drawn; file: in the table)
    page_v: np.ndarray  # page id -> uniform draw of the forced alternative (rand)
    relic: np.ndarray  # relic id -> +1 PREFER / -1 AVOID / 0
    rare_on: dict[str, bool]
    pages: dict[str, list[Any]] = field(default_factory=dict)  # page -> [arm name or "P", frames, fired, unavailable]
    relics: dict[str, list[int]] = field(default_factory=dict)  # relic -> [value, event, treasure, shop, fired]
    rare: dict[str, list[int]] = field(default_factory=dict)  # option -> [on, frames offered, fired]

    def record(self, n: _Names) -> dict[str, Any]:
        return {"seed": self.seed, "pages": self.pages, "relics": self.relics, "rare": self.rare,
                "table": {"pages_forced": [n.page_names[i] for i in np.flatnonzero(self.page_forced)],
                          "prefer": [n.relic_names[i] for i in np.flatnonzero(self.relic > 0)],
                          "avoid": [n.relic_names[i] for i in np.flatnonzero(self.relic < 0)],
                          "rare_on": sorted(k for k, v in self.rare_on.items() if v)}}


_DOMAIN_COL = {"event": 1, "treasure": 2, "shop": 3}


class Explore:
    """Screening table per game (r3 task A; evaluation-only, a table of pages alone is a :class:`PageTable`). Three
    parts, tried in this order on the policy's greedy choice:

    * event pages (EVENT frame, greedy = an event option that is not a RELIC_OPTION, at least two distinct allowed
      such option keys): page forced with probability ``p_page`` (rand) / listed (file); its option = one of the
      allowed keys other than the greedy one at the page's first frame of the game (uniform by ``page_v``; file: the
      listed key), forced at every frame of that page in the game (every seat; a shared event's team vote is one
      frame) where it is allowed (else the policy's choice, counted unavailable).
    * relics (``relic_rows``): PREFER with probability ``p_relic``, AVOID with ``p_relic`` (rand) / listed (file).
      Event / treasure (greedy one of the relic rows): a PREFER relic offered -> the policy's best PREFER row; else
      greedy AVOID -> its best non-AVOID relic row (if any). Shop: an allowed PREFER relic -> greedy if it buys a
      PREFER relic, else the best PREFER relic; else greedy buys an AVOID relic -> its best action that buys none.
      A frame a part applies to (its answer may be greedy itself) is not offered to the later parts.
      Treasure votes: each seat's vote frame gets the same rule (all seats together).
    * rare options (``rare``: ``row_idents`` names): each on with probability ``p_rare`` (rand) / listed (file);
      greedy not a potion action and an allowed row of an on option whose identity greedy does not have -> the best
      row of the first such option in ``rare`` order.

    ``rand:<master>:<cfg json>`` - cfg ``{"p_page", "p_relic", "p_rare", "rare": [...]}``; per game
    ``default_rng(SeedSequence([master, *seed bytes]))`` draws in this order: forced pages (u < p_page over the
    event_page vocab), page_v, relics (u < p_relic PREFER, < 2 p_relic AVOID over the relic vocab; PAD / UNK neutral),
    rare (u < p_rare, in ``rare`` order).
    ``file:<json>`` - ``{"pages": {page: full option key vocab name}, "prefer": [...], "avoid": [...], "rare": [...]}``
    (pages by ``short_page`` name; records name options by ``short_option``, unique within a page).
    The per-game table is made by ``start`` (``HabitOverride.on_job``, before the env's RESET)."""

    name = "explore"
    KINDS = frozenset((Kind.EVENT, Kind.TREASURE, Kind.SHOP, Kind.REST, Kind.REWARD, Kind.CARD_REWARD, Kind.CONFIRM,
                       Kind.CARD_SELECT, Kind.BUNDLE, Kind.MAP))

    def __init__(self, mode: str, master: int = 0, cfg: Mapping[str, Any] | None = None,
                 fixed: Mapping[str, Any] | None = None) -> None:
        self.mode = mode
        self.fixed: Mapping[str, Any] = fixed or {}
        if mode == "rand":
            assert cfg is not None and set(cfg) <= {"p_page", "p_relic", "p_rare", "rare", "note"}, cfg
            self.master = master
            self.p_page, self.p_relic, self.p_rare = float(cfg["p_page"]), float(cfg["p_relic"]), float(cfg["p_rare"])
            assert 0 <= self.p_page <= 1 and 0 <= 2 * self.p_relic <= 1 and 0 <= self.p_rare <= 1, cfg
            self.rare: list[str] = list(cfg["rare"])
        else:
            assert mode == "file" and set(self.fixed) <= {"pages", "prefer", "avoid", "rare", "note"}, self.fixed.keys()
            assert not set(self.fixed.get("prefer", [])) & set(self.fixed.get("avoid", [])), "relic PREFER and AVOID"
            self.rare = list(self.fixed.get("rare", []))
        assert len(set(self.rare)) == len(self.rare), self.rare
        self.n: _Names | None = None  # set by HabitOverride
        self.games: dict[int, _XGame] = {}  # env -> its current game

    @staticmethod
    def parse(arg: str) -> "Explore":
        """``rand:<master>:<cfg json>``, ``file:<json>`` or ``page:<page>=<option>`` (module docstring); a file table of
        pages alone and the inline page are a :class:`PageTable`."""
        mode, _, rest = arg.partition(":")
        if mode == "rand":
            master, path = rest.split(":", 1)
            return Explore("rand", int(master), cfg=json.loads(open(path, encoding="utf-8").read()))
        if mode == "page":
            assert rest.count("=") == 1 and all(rest.split("=")), f"explore:{arg}: expected page:<page>=<option key>"
            page, key = rest.split("=")
            return PageTable({page: key})
        assert mode == "file", f"explore:{arg}: expected rand:<master>:<cfg>, file:<json> or page:<page>=<option key>"
        d = json.loads(open(rest, encoding="utf-8").read())
        if set(d) <= {"pages", "note"}:
            assert d.get("pages"), f"explore:{arg}: an empty table"
            return PageTable(d["pages"])
        return Explore("file", fixed=d)

    def _fixed_game(self, n: _Names, seed: str) -> _XGame:
        for p in self.fixed.get("pages", {}):
            assert p in n.page_names, f"vocab event_page lacks {p}"
        pages = {n.page_names.index(p): n.vid("event_option_key", k) for p, k in self.fixed.get("pages", {}).items()}
        forced = np.zeros(len(n.page_names), bool)
        forced[list(pages)] = True
        relic = np.zeros(len(n.relic_names), np.int8)
        for v, key in ((1, "prefer"), (-1, "avoid")):
            for r in self.fixed.get(key, []):
                relic[n.vid("relic", r)] = v
        return _XGame(seed, dict(pages), forced, np.zeros(len(n.page_names)), relic, {r: True for r in self.rare})

    def start(self, env_id: int, job: EpisodeJob) -> None:
        n = self.n
        assert n is not None and job.seed is not None, "an exploration table needs the game seed"
        if self.mode != "rand":
            self.games[env_id] = self._fixed_game(n, job.seed)
            return
        rng = np.random.default_rng(np.random.SeedSequence([self.master, *job.seed.encode()]))
        forced = rng.random(len(n.page_names)) < self.p_page
        v = rng.random(len(n.page_names))
        u = rng.random(len(n.relic_names))
        relic = np.where(u < self.p_relic, 1, np.where(u < 2 * self.p_relic, -1, 0)).astype(np.int8)
        relic[:2] = 0  # PAD / UNK
        rare = rng.random(len(self.rare)) < self.p_rare
        self.games[env_id] = _XGame(job.seed, {}, forced, v, relic, dict(zip(self.rare, rare.tolist())))

    # ------------------------------------------------------------------ the three parts (``ok`` = allowed rows)
    def _page(self, n: _Names, f: Frame, g: tuple[int, ...], ok: np.ndarray, greedy: int,
              xg: _XGame) -> tuple[np.ndarray, np.ndarray, int] | None:
        """EVENT page frame of a choice: (allowed non-relic option rows, their keys, the page's arm), else None."""
        if g[_KIND] != _EVENT:
            return None
        rows = _rows(f, Verb.CHOOSE_OPTION, ok)
        o = f.table(Table.OPTIONS)[f.actions[rows, ActionCol.SRC_ROW]]
        m = (o[:, OptionCol.KIND] == OptionKind.EVENT_OPTION) & (o[:, OptionCol.KEY_ID] != n.relic_option)
        rows, keys = rows[m], o[m, OptionCol.KEY_ID]
        if greedy not in rows or len(set(keys.tolist())) < 2:
            return None
        page = int(g[G.EVENT_PAGE_KEY])
        arm = xg.page_arm.get(page)
        if arm is None:
            if not xg.page_forced[page]:
                arm = -1
            else:
                gk = int(keys[rows == greedy][0])
                cands = sorted(set(keys.tolist()) - {gk})
                arm = cands[int(xg.page_v[page] * len(cands))]
            xg.page_arm[page] = arm
        return rows, keys, arm

    @staticmethod
    def _relic_pick(dom: str, rows: np.ndarray, v: np.ndarray, scores: np.ndarray, greedy: int) -> int | None:
        if dom == "shop":
            if (v > 0).any():  # greedy buying a PREFER relic already: applied, unchanged (no later part)
                return greedy if greedy in rows[v > 0] else _best(rows[v > 0], scores)
            if greedy in rows[v < 0]:
                allowed = np.isfinite(scores)
                allowed[rows[v < 0]] = False
                return _best(np.flatnonzero(allowed), scores)  # leaving the shop is always allowed
            return None
        if greedy not in rows:
            return None
        if (v > 0).any():
            return _best(rows[v > 0], scores)
        if greedy in rows[v < 0] and (v >= 0).any():
            return _best(rows[v >= 0], scores)
        return None

    def _rare_rows(self, ids: Sequence[str], ok: np.ndarray, greedy: int, on: Mapping[str, bool]) -> np.ndarray:
        """Allowed rows of the first on rare option greedy does not take ([] = none applies)."""
        if not ids[greedy]:
            return np.zeros(0, np.int64)
        for r in self.rare:
            if on[r] and ids[greedy] != r:
                rows = np.flatnonzero(np.array([i == r for i in ids]) & ok)
                if len(rows):
                    return rows
        return np.zeros(0, np.int64)

    def choose(self, n: _Names, f: Frame, g: tuple[int, ...], scores: np.ndarray, greedy: int, x: Call) -> int | None:
        if g[_IN_COMBAT] or g[_KIND] not in self.KINDS:
            return None
        xg = self.games[x.env_id]
        ok = np.isfinite(scores)
        page = self._page(n, f, g, ok, greedy, xg)
        if page is not None:
            rows, keys, arm = page
            rec = xg.pages.setdefault(n.page_names[g[G.EVENT_PAGE_KEY]], ["P" if arm < 0 else n.opt_names[arm], 0, 0, 0])
            rec[1] += 1
            if arm < 0:
                return None
            target = rows[keys == arm]
            if not len(target):
                rec[3] += 1
                return None
            best = _best(target, scores)
            rec[2] += best != greedy
            return best
        dom, rows, rel = relic_rows(n, f, g, ok)
        if dom:
            for r in set(rel.tolist()):
                rec = xg.relics.setdefault(n.relic_names[r], [int(xg.relic[r]), 0, 0, 0, 0])
                rec[_DOMAIN_COL[dom]] += 1
            best = self._relic_pick(dom, rows, xg.relic[rel], scores, greedy)
            if best is not None:
                if best != greedy:  # fired: the PREFER relic taken, else the AVOID relic greedy wanted
                    r = int(rel[rows == best][0]) if best in rows and xg.relic[rel[rows == best][0]] > 0 \
                        else int(rel[rows == greedy][0])
                    xg.relics[n.relic_names[r]][4] += 1
                return best
        if not self.rare:
            return None
        ids = row_idents(n, f, g)
        for r in set(ids) & set(self.rare):
            xg.rare.setdefault(r, [int(xg.rare_on[r]), 0, 0])[1] += 1
        target = self._rare_rows(ids, ok, greedy, xg.rare_on)
        if not len(target):
            return None
        best = _best(target, scores)
        xg.rare[ids[best]][2] += 1
        return best


class PageTable(Explore):
    """The page table: ``explore:file:<json>`` with only ``pages``, or ``explore:page:<page>=<option key>`` (one page;
    the inline form keeps a run's config.json self-contained). Pages by ``short_page`` name, options by their full
    event_option_key vocab name (an option may belong to another page of the event, e.g. ``ABYSSAL_BATHS.pages.ALL``).
    ``choose`` is :class:`Explore`'s page part. ``judge`` (the advice rule; the table is resolved by ``resolve`` at the
    Advisor's start-up): at an EVENT frame of a listed page with at least two distinct allowed non-relic option keys
    and the listed option allowed, +1 for a row of the listed option, -1 for a row of another non-relic option of that
    page; everything else 0 (PROCEED, potion actions, the page's RELIC_OPTION rows, other pages)."""

    KINDS = frozenset((Kind.EVENT,))
    VERBS = frozenset((_CHOOSE_OPTION,))

    def __init__(self, pages: Mapping[str, str]) -> None:
        super().__init__("file", fixed={"pages": dict(pages)})
        self.pages = dict(pages)
        self._table: tuple[_Names, _XGame] | None = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PageTable) and self.pages == other.pages

    def resolve(self, n: _Names) -> _XGame:
        """The table on ``n``'s vocab (resolved on first use: an unknown page / option fails here)."""
        if self._table is None or self._table[0] is not n:
            self._table = (n, self._fixed_game(n, ""))
        return self._table[1]

    def judge(self, n: _Names, f: Frame, g: tuple[int, ...], action: int, banned: Sequence[int]) -> int:
        if g[_KIND] != _EVENT or f.actions[action, _VERB] != _CHOOSE_OPTION:
            return 0
        xg = self.resolve(n)
        if not xg.page_forced[g[G.EVENT_PAGE_KEY]]:
            return 0
        page = self._page(n, f, g, allowed_rows(f, banned), action, xg)
        if page is None:
            return 0
        rows, keys, arm = page
        return (1 if keys[rows == action][0] == arm else -1) if (keys == arm).any() else 0


def parse_habit(spec: str) -> list[Any]:
    spec = ALIASES.get(spec, spec)
    out: list[Any] = []
    for part in filter(None, spec.split("+")):
        name, _, arg = part.partition(":")
        if name == "smith_pre":
            t, t_pre = (float(x) for x in arg.split(":"))
            out.append(SmithPre(t, t_pre=t_pre))
        elif name == "smith":
            out.append(Smith(float(arg)))
        elif name == "no_elite":
            a, b = (int(x) for x in arg.split("-"))
            out.append(NoElite((a, b)))
        elif name == "potion_hold":
            out.append(PotionHold(("Monster",)))
        elif name == "potion_boss":
            out.append(PotionHold(("Monster", "Elite"), name="potion_boss"))
        elif name == "potion_r1":
            rooms = tuple(arg.split(","))
            assert len(set(rooms)) == len(rooms) and set(rooms) <= set(_COMBAT_ROOMS), \
                f"{part}: rooms must be distinct names of {_COMBAT_ROOMS}"
            out.append(PotionRound1(rooms))
        elif name == "enrage":
            assert arg in ("none", "block"), part
            out.append(Enrage(arg))
        elif name == "shop_relic":
            out.append(ShopPrefer("RELIC", name="shop_relic"))
        elif name == "shop_nocard":
            out.append(ShopBan(("CHARACTER_CARD", "COLORLESS_CARD"), name="shop_nocard"))
        elif name == "shop_nopotion":
            out.append(ShopBan(("POTION",), name="shop_nopotion"))
        elif name == "remove_basic":
            out.append(RemoveBasic())
        elif name == "deck_cap":
            out.append(DeckCap(int(arg)))
        elif name == "prefs":
            out.append(Prefs.parse(arg))
        elif name == "heal_late":
            t, *rest = arg.split(":")
            assert len(rest) <= 2, f"{part}: expected heal_late:<t>[:<act>[:<rows>]]"
            h = HealLate(float(t), *(int(x) for x in rest))
            assert 0.0 < h.t <= 1.0 and 0 <= h.act and h.rows >= 1, \
                f"{part}: expected t in (0, 1], act >= 0 (0 = every act), rows >= 1"
            out.append(h)
        elif name == "potion_none":
            assert not arg, part
            out.append(PotionHold(_COMBAT_ROOMS, name="potion_none"))
        elif name == "lethal_block":
            out.append(LethalBlock(float(arg)) if arg else LethalBlock())
        elif name == "kill_dangerous":
            out.append(KillDangerousMargin(float(arg)) if arg else KillDangerous())
        elif name == "weak_attacker":
            assert arg in ("", "cards"), part
            out.append(WeakAttackerCards() if arg else WeakAttacker())
        elif name == "potion_reserve2":
            assert arg in ("", "all"), part
            out.append(PotionReserve(arg == "all"))
        elif name == "lethal_eot":
            assert not arg, part
            out.append(LethalBlock(1.0, eot=True, name="lethal_eot"))
        elif name in ("potion_boss_use", "potion_boss_take"):
            assert arg in ("", "all", "late"), part
            out.append((PotionBoss if name == "potion_boss_use" else PotionBossTake)(arg or "act3"))
        elif name == "potion_lethal":
            assert not arg, part
            out.append(PotionLethal())
        elif name == "explore":
            out.append(Explore.parse(arg))
        else:
            raise ValueError(f"unknown habit {part!r}")
    assert len({h.name for h in out}) == len(out), spec
    out.sort(key=lambda h: isinstance(h, Prefs))  # precedence: deck_cap decides before prefs (stable otherwise)
    return out


class HabitOverride:
    """``spire.actor.Override`` of a habit list; remembers per env the last (step, greedy, chosen, habit name) and the
    seat whose card-removal purchase opened the current DECK_REMOVAL prompt; starts the per-game table of an explore
    habit (``on_job``)."""

    def __init__(self, spec: str, vocab: Vocab, catalog: Catalog) -> None:
        self.spec = spec
        self.habits = parse_habit(spec)
        self.names = _Names(vocab, catalog)
        self.last: dict[int, tuple[int, int, int, int, str]] = {}  # env -> (episode, step, greedy, chosen, habit)
        self.removal_seat: dict[int, int] = {}  # env -> seat that bought the removal whose prompt is open
        explore = [h for h in self.habits if isinstance(h, Explore)]
        self.explore: Explore | None = explore[0] if explore else None
        if self.explore is not None:
            self.explore.n = self.names

    def on_job(self, env_id: int, job: EpisodeJob | None) -> None:
        """``evaluate(on_job=...)``: the env's next game (None: the env goes idle)."""
        if self.explore is not None and job is not None:
            self.explore.start(env_id, job)

    def __call__(self, env_id: int, f: Frame, sample: Sample, scores: np.ndarray, banned: Sequence[int]) -> int | None:
        greedy = int(np.argmax(scores))
        g = f.gvals
        n = self.names
        seat = self.removal_seat.pop(env_id, -1)
        x = Call(env_id, sample, removal_bought=seat == g[_SEAT] and removal_prompt(n, g))
        chosen, by = None, ""
        for h in self.habits:
            chosen = h.choose(n, f, g, scores, greedy, x)
            if chosen is not None:
                by = h.name
                break
        action = greedy if chosen is None else chosen
        if x.removal_bought or bought_removal(n, f, g, action):
            self.removal_seat[env_id] = int(g[_SEAT])
        self.last[env_id] = (f.episode_id, f.step_id, greedy, action, by)
        return chosen


@dataclass
class _Game:
    episode_id: int
    decisions: int = 0
    c: Counter[str] = field(default_factory=Counter)
    combat_key: tuple[int, int] | None = None
    boss_hp: list[float] = field(default_factory=list)
    prev_room: int = -1  # ROOM_TYPE of the previous decision
    deck_up: list[int] | None = None  # per seat: sum of upgrade levels of the deck (last out-of-combat frame)
    deck_n: list[int] | None = None
    # the last MAP decision until a combat consumes it: (act, elite offered, non-elite offered, chosen point type)
    last_map: tuple[int, bool, bool, int] | None = None
    removal_seat: int = -1  # seat whose card-removal purchase opened the current DECK_REMOVAL prompt (-1 = none)
    potions_held: int = 0
    deck_count: list[int] = field(default_factory=list)
    forced: list[list[Any]] = field(default_factory=list)  # [decision index, kind, greedy row, chosen row, habit]
    visits: set[tuple[int, int]] = field(default_factory=set)  # (floor, seat) shop visits counted
    # confirmed rest upgrade / shop removal awaiting its check at the next out-of-combat frame:
    # (label, seat, card id, level, copies of (id, level) in the seat's deck, copies of (id, level + 1))
    pending: tuple[str, int, int, int, int, int] | None = None
    explore: _XGame | None = None  # the game's exploration table (habit explore)


def fight_label(n: _Names, g: tuple[int, ...]) -> str:
    """Room name + act (1-based) of an in-combat frame, ".1" / ".2" for the two act-3 bosses of the A10 double boss
    (BOSSES_WON 2 / 3); "out" outside combat."""
    if not g[_IN_COMBAT]:
        return "out"
    label = f"{n.room_names[g[_ROOM]]}{g[G.ACT_INDEX] + 1}"
    if g[_ROOM] == n.room["Boss"] and g[G.ACT_INDEX] == 2 and g[G.SECOND_BOSS_ENCOUNTER] > 1:
        label += f".{g[G.BOSSES_WON] - 1}"
    return label


class HabitRecorder:
    """Per-game habit counters from the decision stream (``evaluate(on_decision=...)``) and the override's memory."""

    def __init__(self, override: HabitOverride) -> None:
        self.ov = override
        self.n = override.names
        self._open: dict[int, _Game] = {}
        self._done: dict[int, list[_Game]] = {}

    def observe(self, env_id: int, f: Frame, action: int) -> None:
        n, g = self.n, f.gvals
        e = self._open.get(env_id)
        if e is None or e.episode_id != f.episode_id:
            if e is not None:
                self._done.setdefault(env_id, []).append(self._open.pop(env_id))
            e = self._open[env_id] = _Game(f.episode_id)
            if self.ov.explore is not None:
                e.explore = self.ov.explore.games[env_id]
        ep, step, greedy, chosen, by = self.ov.last[env_id]
        assert (ep, step, chosen) == (f.episode_id, f.step_id, action), "override memory out of step"
        c = e.c
        i = e.decisions
        e.decisions += 1
        if by:
            c[f"applied:{by}"] += 1
            if chosen != greedy:
                c[f"changed:{by}"] += 1
                c[f"changed:{by}:{fight_label(n, g)}"] += 1
                e.forced.append([i, Kind(g[G.KIND]).name, greedy, chosen, by])
        acts, seats = f.actions, f.table(Table.SEATS)
        verb = int(acts[action, ActionCol.VERB])
        kind, room = g[G.KIND], g[G.ROOM_TYPE]
        potions = f.table(Table.POTIONS)
        e.potions_held = int((potions[:, PotionCol.ZONE] == ItemZone.OWNED).sum())
        e.deck_count = seats[:, SeatCol.DECK_COUNT].tolist()
        bought = e.removal_seat == g[G.ACTING_SEAT] and removal_prompt(n, g)
        e.removal_seat = int(g[G.ACTING_SEAT]) if bought or bought_removal(n, f, g, action) else -1
        if g[G.IN_COMBAT]:
            key = (g[G.TOTAL_FLOOR], g[G.COMBATS_WON])
            if e.combat_key != key:
                e.combat_key = key
                rname = n.room_names[room]
                c[f"combats:{rname}"] += 1
                if room == n.room["Boss"]:
                    e.boss_hp.append(round(float(seats[:, SeatCol.HP].sum() / seats[:, SeatCol.MAX_HP].sum()), 4))
                lm, e.last_map = e.last_map, None
                if room == n.room["Elite"]:
                    c[f"elites_entered_act{g[G.ACT_INDEX] + 1}"] += 1
                    # the TRAVEL that led into this elite chose an Elite node while a non-Elite one was offered
                    if lm is not None and lm[0] == g[G.ACT_INDEX] and lm[1] and lm[2] and lm[3] == n.elite_point:
                        c[f"elites_entered_with_alternative_act{g[G.ACT_INDEX] + 1}"] += 1
            if kind == Kind.COMBAT and verb == Verb.USE_POTION:
                c[f"potions_used:{n.room_names[room]}"] += 1
        else:
            e.combat_key = None
            self._deck(e, f, seats)
        if kind == Kind.REST:
            if verb == Verb.CHOOSE_OPTION:
                key_id = int(f.table(Table.OPTIONS)[acts[action, ActionCol.SRC_ROW], OptionCol.KEY_ID])
                c[f"rest:{n.rest_names[key_id]}"] += 1
            else:
                c[f"rest:{Verb(verb).name}"] += 1
        elif kind == Kind.CARD_SELECT and not g[G.IN_COMBAT]:
            sub = CardSelectSub(g[G.KIND_SUB]).name
            if verb == Verb.CONFIRM_SELECTION:
                c[f"confirm:{sub}:{n.room_names[room]}"] += 1
                label = {(CardSelectSub.DECK_UPGRADE, n.room["RestSite"]): "upgrade",
                         (CardSelectSub.DECK_REMOVAL, n.room["Shop"]): "removal"}.get((g[G.KIND_SUB], room))
                if label is not None:
                    e.pending = self._pending(f, int(g[G.ACTING_SEAT]), label)
            elif verb == Verb.CANCEL:
                c[f"cancel:{sub}:{n.room_names[room]}"] += 1
            elif verb == Verb.SELECT_CARD and g[G.KIND_SUB] == CardSelectSub.DECK_REMOVAL and room == n.room["Shop"]:
                card = f.table(Table.CARDS)[acts[action, ActionCol.SRC_ROW]]
                cls = CLASS_NAMES[n.card_class[card[CardCol.CARD_ID]]]
                c[f"removal_selected:{cls}" if bought else f"removal_selected_unbought:{cls}"] += 1
        elif kind == Kind.SHOP:
            seat = int(g[G.ACTING_SEAT])
            opts = f.table(Table.OPTIONS)
            rem = opts[(opts[:, OptionCol.KIND] == OptionKind.SHOP_ENTRY) & (opts[:, OptionCol.KEY_ID] == n.removal)]
            assert len(rem) <= 1
            visit = (int(g[G.TOTAL_FLOOR]), seat)
            if visit not in e.visits:
                e.visits.add(visit)
                classes = deck_classes(f, n, seat)
                stocked = bool(len(rem) and rem[0, OptionCol.FLAGS] & OptionFlag.STOCKED)
                affordable = stocked and bool(rem[0, OptionCol.FLAGS] & OptionFlag.AFFORDABLE)
                target = bool(classes[1] or classes[2] or classes[3])
                c["shop_visits"] += 1
                c["removal_offered"] += stocked
                c["removal_affordable"] += affordable
                c["removal_affordable_with_target"] += affordable and target
            if verb == Verb.CHOOSE_OPTION:
                o = opts[acts[action, ActionCol.SRC_ROW]]
                if o[OptionCol.KIND] == OptionKind.SHOP_ENTRY and o[OptionCol.KEY_ID] == n.removal:
                    c["removal_bought"] += 1
        elif kind == Kind.MAP:
            rows = np.flatnonzero(acts[:, ActionCol.VERB] == Verb.TRAVEL)
            elite = map_targets_elite(f, rows, n)
            c[f"map_decisions_act{g[G.ACT_INDEX] + 1}"] += 1
            c[f"map_seat:{g[G.ACTING_SEAT]}"] += 1
            point = -1
            if verb == Verb.TRAVEL:
                point = int(f.table(Table.MAP_NODES)[acts[action, ActionCol.SRC_ROW], MapNodeCol.POINT_TYPE])
                c[f"map_travel:{n.point_names[point]}"] += 1
                if elite.any() and (~elite).any():
                    c[f"map_choice_elite_vs_other_act{g[G.ACT_INDEX] + 1}"] += 1
                    c[f"map_chose_elite_when_alternative_act{g[G.ACT_INDEX] + 1}"] += point == n.elite_point
            e.last_map = (int(g[G.ACT_INDEX]), bool(elite.any()), bool((~elite).any()), point)
        e.prev_room = room

    @staticmethod
    def _copies(deck: np.ndarray, seat: int, card: int, level: int) -> int:
        m = (deck[:, CardCol.SEAT] == seat) & (deck[:, CardCol.CARD_ID] == card) & (deck[:, CardCol.UPGRADE_LEVEL] == level)
        return int(deck[m, CardCol.COUNT].sum())

    def _pending(self, f: Frame, seat: int, label: str) -> tuple[str, int, int, int, int, int]:
        cards = f.table(Table.CARDS)
        sel = cards[(cards[:, CardCol.ZONE] == Zone.CHOICE) & (cards[:, CardCol.SEL_COUNT] > 0)]
        assert len(sel) == 1 and sel[0, CardCol.SEL_COUNT] == 1, "one card confirmed"
        cid, lv = int(sel[0, CardCol.CARD_ID]), int(sel[0, CardCol.UPGRADE_LEVEL])
        deck = cards[cards[:, CardCol.ZONE] == Zone.DECK]
        return label, seat, cid, lv, self._copies(deck, seat, cid, lv), self._copies(deck, seat, cid, lv + 1)

    def _deck(self, e: _Game, f: Frame, seats: np.ndarray) -> None:
        """Deck changes since the last out-of-combat frame, booked to the room of the previous decision; the check
        of a pending confirmed upgrade (the card's copies at its level -1, at level + 1 +1) / removal (copies -1)."""
        cards = f.table(Table.CARDS)
        deck = cards[cards[:, CardCol.ZONE] == Zone.DECK]
        if e.pending is not None:
            label, seat, cid, lv, same, nxt = e.pending
            e.pending = None
            ok = self._copies(deck, seat, cid, lv) == same - 1
            if label == "upgrade":
                ok = ok and self._copies(deck, seat, cid, lv + 1) == nxt + 1
            e.c[f"{label}_verified" if ok else f"{label}_not_verified"] += 1
        ns = len(seats)
        up = [int((deck[deck[:, CardCol.SEAT] == s, CardCol.UPGRADE_LEVEL]
                   * deck[deck[:, CardCol.SEAT] == s, CardCol.COUNT]).sum()) for s in range(ns)]
        cnt = [int(deck[deck[:, CardCol.SEAT] == s, CardCol.COUNT].sum()) for s in range(ns)]
        if e.deck_up is not None and e.deck_n is not None and e.prev_room >= 0:
            rname = self.n.room_names[e.prev_room]
            for s in range(ns):
                if up[s] > e.deck_up[s]:
                    e.c[f"deck_upgrade_levels_gained:{rname}"] += up[s] - e.deck_up[s]
                if cnt[s] < e.deck_n[s]:
                    e.c[f"deck_cards_lost:{rname}"] += e.deck_n[s] - cnt[s]
        e.deck_up, e.deck_n = up, cnt

    def rows(self, episodes: Sequence[Mapping[str, Any]], keep_forced: int) -> list[dict[str, Any]]:
        """One row per finished game, joined in order per host to the evaluation's episode records (like
        ``DebuffTelemetry.report``); ``forced`` lists only for seed ids below ``keep_forced``."""
        for env_id in list(self._open):
            self._done.setdefault(env_id, []).append(self._open.pop(env_id))
        records: dict[int, list[Mapping[str, Any]]] = {}
        for rec in episodes:
            records.setdefault(int(rec["env"]), []).append(rec)
        out = []
        for env_id in sorted(set(records) | set(self._done)):
            games = list(reversed(self._done.get(env_id, [])))
            for rec in records.get(env_id, []):
                if rec["decisions"] == 0:
                    continue
                e = games.pop()
                assert e.decisions == rec["decisions"], (env_id, e.decisions, rec)
                assert e.explore is None or e.explore.seed == rec["seed"], (env_id, e.explore.seed, rec["seed"])
                lost = rec["result"] != "win"
                out.append({
                    "seed": rec["seed"], "seed_id": int(rec["seed_id"]), "lineup": rec["lineup"],
                    "ascension": rec["ascension"], "result": rec["result"], "win": rec["result"] == "win",
                    "floor": rec["floor"], "bosses_won": rec["bosses_won"], "elites_won": rec["elites_won"],
                    "combats_won": rec["combats_won"], "act": rec["act"] + 1 if lost else None,
                    "room": rec["room"] if lost else None, "enemies": list(rec["enemies"]) if lost else None,
                    "decisions": rec["decisions"], "seconds": rec["seconds"],
                    "potions_held_at_end": e.potions_held, "deck_count_at_end": e.deck_count,
                    "boss_start_team_hp": e.boss_hp, "counts": dict(sorted(e.c.items())),
                    **({"explore": e.explore.record(self.n)} if e.explore is not None else {}),
                    **({"forced": e.forced} if int(rec["seed_id"]) < keep_forced else {})})
            assert len(games) <= 1, f"env {env_id}: {len(games)} games without an episode record"
        self._done.clear()
        return out


def write_rows(rows: Sequence[Mapping[str, Any]], path: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: r["seed_id"]):
            fh.write(json.dumps(r) + "\n")
