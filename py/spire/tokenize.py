"""Frame -> compact per-sample entity record (:class:`Sample`) and bucketed collation (:class:`Batch`).

Entities
--------
Every frame becomes a list of *entity rows*, in four contiguous groups:

* **tokens** - the transformer sequence: ``GLOBAL`` (index 0, run state), ``DECISION`` (index 1,
  prompt/event/selection state), then one token per row of SEATS, CARDS, CREATURES, RELICS,
  POTIONS, ORBS, ARROWS, MAP_NODES, OPTIONS, CELLS (and, with ``pool_children=False``, of
  POWERS, INTENTS, REFS and the untargeted VARS);
* **kids** - child rows pooled into an owner token before the transformer: POWERS/INTENTS
  (owner = creature) and REFS (owner = option); with ``forecast=True`` also the FORECAST rows
  (owner = creature; notes/forecast/contract.md 3), after REFS; with ``calc=True`` the TURN / FOE rows (owner =
  creature) and ROUTE rows (owner = map node) after FORECAST (notes/calc/contract.md 3); with ``route_summary=True``
  one AHEAD row (owner = GLOBAL) after ROUTE (notes/inputs2/contract.md 2.3);
* **vars** - VARS rows (schema 5.3) pooled into their owner (card/relic/potion/option token or
  power kid); per-target VARS (``target_row >= 0``) are routed to the matching
  ``(src_row, tgt_row)`` action instead (schema 13);
* **actions** - one row per ACTIONS row.

MAP_EDGES rows carry no features of their own: each edge is stored twice in ``Sample.edges``
(out-edge pooled into the from-node, in-edge into the to-node, each linked to the node at the
other end).

Each entity row carries its type (:class:`Ent`), ``W`` raw int16 *slots* (the table columns
the feature spec needs plus a few derived columns, see :data:`SPECS`) and three relation
indices into the sample's own entity rows (``-1`` = none):

======== ============================ ============================ =================
group    rel[0]                       rel[1]                       rel[2]
======== ============================ ============================ =================
token    owner (child tokens only)    related token (see below)    -
kid      owner token (pool target)    related token                -
var      pool target (token/kid/act)  -                            -
action   source token                 target creature / MEND seat  acting seat token
======== ============================ ============================ =================

Related tokens: SEATS -> its creature, CARDS -> ``link_row`` (CHOICE source, PREVIEW subject),
OPTIONS -> ``ref_row`` entity, POWERS -> applier creature, REFS -> referenced entity row.
With ``mend_target=True``, MEND target relations point to SEATS and the model gates this new
contribution. With ``coop_features=True``, SEATS packs acts/screen/status in its existing
derived slot and OPTION adds a symmetric vote count, preserving the compact record width.

Normalisation (schema 13: symlog with +-8 clip, ratios, ``-1`` split into value 0 + missing bit,
bit masks split into bits, positions and small enums as embeddings) is specified here per
entity type and executed on the device by :class:`FeatureDecoder`, so a sample stays a small
integer record and the per-frame CPU work is a handful of vectorised copies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
from operator import attrgetter
from typing import Any, Iterator, Mapping, Sequence, Union, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .proto import Frame
from .schema import (
    CALC_N_COLS,
    CALC_TABLES,
    CARD_CHOICE_AUX_VOCAB,
    CREATURE_MODEL_VOCAB,
    CRYSTAL_ACTION_AUX_VOCAB,
    DATA_OFFSET,
    DIR_OFFSET,
    DIR_STRIDE,
    FORECAST_N_COLS,
    FORECAST_TABLE,
    FOE_TABLE,
    GLOBAL_LEN,
    GLOBAL_OFFSET,
    HDRAW_N_COLS,
    HDRAW_TABLE,
    N_COLS,
    N_TABLES,
    N_TABLES_FORECAST,
    OPTION_KEY_VOCAB,
    REF_CATEGORY_VOCAB,
    ROUTE_TABLE,
    TURN_TABLE,
    WORD,
    ActionCol,
    ArrowCol,
    CardCol,
    CellCol,
    CreatureCol,
    CreatureKind,
    FoeCol,
    ForecastCol,
    FrameFlag,
    G,
    H,
    IntentCol,
    Kind,
    MapEdgeCol,
    MapNodeCol,
    MapNodeFlag,
    MsgType,
    OptionCol,
    OptionKind,
    OrbCol,
    PotionCol,
    PowerCol,
    RefCol,
    RelicCol,
    RouteCol,
    SeatCol,
    Table,
    TurnCol,
    VarCol,
    Zone,
    n_tables_for,
)
from .vocab import Vocab

try:  # compiled kernels (python -B native/build.py); the numpy batch path is the identical fallback
    from . import _native  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on the build
    _native = None

I32 = NDArray[np.int32]
I64 = NDArray[np.int64]
Batch = dict[str, Tensor]

SYMLOG_CLIP = 8.0
STATIC_LEVELS = 4  # card static-prior rows per card id (upgrade levels 0..3, higher clamp to 3)
HISTORY_ROWS = 5  # previous action, source, source's related entity, target, acting seat
HISTORY_STATS = 32
CYCLE_BAN_SLOTS = 6  # final history statistics: action index + 1, zero means empty


def history_width(width: int) -> int:
    return HISTORY_ROWS * (1 + width) + HISTORY_STATS


class Ent(IntEnum):
    """Entity (token / child / action) type ids. 0 is padding."""

    PAD = 0
    GLOBAL = 1
    DECISION = 2
    SEAT = 3
    CARD = 4
    CREATURE = 5
    RELIC = 6
    POTION = 7
    ORB = 8
    ARROW = 9
    MAP_NODE = 10
    OPTION = 11
    CELL = 12
    POWER = 13
    INTENT = 14
    REF = 15
    VAR = 16
    EDGE_OUT = 17
    EDGE_IN = 18
    ACTION = 19
    FORECAST = 20  # kid rows of the FORECAST table (TokenizerConfig.forecast); no type_emb row without it
    TURN = 21  # kid rows of the calc tables (TokenizerConfig.calc; notes/calc/contract.md 3.2)
    FOE = 22
    ROUTE = 23
    AHEAD = 24  # the route summary row, a kid of the GLOBAL token (TokenizerConfig.route_summary; notes/inputs2)


N_ENT = len(Ent)
N_ENT_LEGACY = int(Ent.FORECAST)  # type_emb rows of a layout without forecast
N_ENT_FORECAST = int(Ent.TURN)  # type_emb rows of a layout with forecast, without calc
N_ENT_CALC = int(Ent.AHEAD)  # type_emb rows of a layout with calc, without the route summary
CALC_ENTS = (Ent.TURN, Ent.FOE, Ent.ROUTE)


class Mode(IntEnum):
    """Numeric feature transforms (executed by :class:`FeatureDecoder`)."""

    LIN = 0  # x * scale
    SYM = 1  # symlog(x * scale), clipped to +-8
    BIT = 2  # bit ``arg`` of x
    EQ = 3  # x == arg
    MISS = 4  # x == -1
    RATIO = 5  # x / max(y, 1), clipped to +-2
    SYMDIFF = 6  # symlog((x - y) * scale)
    SYMPROD = 7  # symlog(x * y * scale)


Src = Union[int, str]  # a table column, or the name of a derived column


@dataclass(frozen=True)
class Id:
    """A categorical slot: ``clip(raw + add, 0, rows - 1)`` indexes embedding table ``table``.

    ``table``: a vocab category (rows = ``Vocab.embedding_rows``), ``"<cat>#<tag>"`` (a separate
    table sized like ``cat``), ``"p:<name>"`` (protocol table, :data:`PROTOCOL_ROWS`) or ``"@"``
    (the slot already holds a global row index, resolved during encoding).
    """

    src: Src
    table: str
    add: int = 0


@dataclass(frozen=True)
class Num:
    src: Src
    mode: Mode = Mode.LIN
    scale: float = 1.0
    arg: int = 0
    src2: Src | None = None
    null: bool = False  # src == -1 -> feature 0 (pair with a MISS feature)


@dataclass(frozen=True)
class EntitySpec:
    ids: tuple[Id, ...]
    nums: tuple[Num, ...]
    derived: tuple[str, ...] = ()


def _lin(c: Src, s: float = 1.0, null: bool = False) -> Num:
    return Num(c, Mode.LIN, s, null=null)


def _sym(c: Src, s: float = 1.0, null: bool = False) -> Num:
    return Num(c, Mode.SYM, s, null=null)


def _miss(c: Src) -> Num:
    return Num(c, Mode.MISS)


def _bits(c: Src, n: int) -> tuple[Num, ...]:
    return tuple(Num(c, Mode.BIT, arg=i) for i in range(n))


PROTOCOL_ROWS: Mapping[str, int] = {
    "p:kind": 16, "p:kind_sub": 16 * 16, "p:n_seats": 5, "p:view_mode": 2, "p:ascension": 16,
    "p:act_index": 4, "p:seat": 5, "p:pos": 33, "p:zone": 10, "p:group": 17, "p:creature_kind": 4,
    "p:option_kind": 9, "p:ref_category": 10, "p:map_state": 4, "p:map_col": 9, "p:grid": 12,
    "p:verb": 16, "p:table": N_TABLES, "p:item_zone": 3, "p:treasure_phase": 3, "p:intent_idx": 9,
}
STATIC_CATEGORIES: Mapping[str, int] = {"card": STATIC_LEVELS, "relic": 1, "potion": 1, "power": 1, "monster": 1}

_S, _C, _P, _O, _R = SeatCol, CardCol, PowerCol, OptionCol, RelicCol

SPECS: Mapping[Ent, EntitySpec] = {
    Ent.GLOBAL: EntitySpec(
        ids=(Id(G.KIND, "p:kind"), Id("kind_sub", "p:kind_sub"), Id(G.N_SEATS, "p:n_seats"),
             Id(G.VIEW_MODE, "p:view_mode"), Id(G.ASCENSION, "p:ascension"), Id(G.GAME_MODE, "enum:GameMode"),
             Id(G.ACT_INDEX, "p:act_index"), Id(G.ACT_ID, "act"), Id(G.ROOM_TYPE, "enum:RoomType"),
             Id(G.BOSS_ENCOUNTER, "encounter"), Id(G.SECOND_BOSS_ENCOUNTER, "encounter"),
             Id(G.ANCIENT_EVENT, "event"), Id(G.LAST_WON_ROOM_TYPE, "enum:RoomType#lastwon")),
        nums=(_lin(G.ACT_FLOOR, 1 / 17), _lin(G.TOTAL_FLOOR, 1 / 50), _lin(G.MAP_ROWS, 1 / 20),
              _lin(G.CUR_ROW, 1 / 20, True), _miss(G.CUR_ROW), _lin(G.CUR_COL, 1 / 7, True), _miss(G.CUR_COL),
              _lin(G.IN_COMBAT), _lin(G.COMBAT_ROUND, 1 / 20), _lin(G.PLAYERS_ALIVE, 1 / 4),
              # public win history (errata-v1 section 5); 49..51 are TERMINAL-only protocol slots, never features
              _sym(G.COMBATS_WON), _lin(G.ELITES_WON, 0.2), _lin(G.BOSSES_WON, 0.25),
              _lin(G.LAST_WON_TOTAL_FLOOR, 1 / 50, True), _miss(G.LAST_WON_TOTAL_FLOOR)),
        derived=("kind_sub",)),
    Ent.DECISION: EntitySpec(
        ids=(Id(G.KIND, "p:kind"), Id(G.COMBAT_SIDE, "enum:CombatSide"), Id(G.PLAYER_PHASE, "enum:PlayerTurnPhase"),
             Id(G.PROMPT_KEY, "prompt_key"), Id("prompt_source", "@"), Id(G.EVENT_ID, "event"),
             Id(G.EVENT_PAGE_KEY, "event_page"), Id(G.CONFIRM_KEY, "confirm_key"),
             Id(G.CRYSTAL_TOOL, "enum:CrystalSphereToolType"), Id(G.ACTING_SEAT, "p:seat", 1),
             Id(G.TREASURE_PHASE, "p:treasure_phase")),
        nums=(_lin(G.SEL_MIN, 0.1), _lin(G.SEL_MAX, 0.1), _lin(G.SEL_COUNT, 0.1), *_bits(G.SEL_FLAGS, 6),
              *_bits(G.EVENT_FLAGS, 3), *_bits(G.REST_FLAGS, 2), *_bits(G.REWARD_FLAGS, 1),
              *_bits(G.SHOP_FLAGS, 1), *_bits(G.CARD_REWARD_FLAGS, 2), *_bits(G.BUNDLE_FLAGS, 1),
              _lin(G.CRYSTAL_DIVINATIONS, 0.1)),
        derived=("prompt_source",)),
    Ent.SEAT: EntitySpec(
        ids=(Id(_S.SEAT, "p:seat", 1), Id(_S.CHARACTER_ID, "character")),
        nums=(_lin(_S.PRIVATE_VISIBLE), _lin(_S.ALIVE), Num(_S.HP, Mode.RATIO, src2=_S.MAX_HP, null=True),
              _sym(_S.HP, null=True), _sym(_S.MAX_HP, null=True), _sym(_S.BLOCK), _sym(_S.GOLD, 0.1),
              _lin(_S.POTION_SLOTS, 0.2), _lin(_S.ENERGY, 0.2, True), _miss(_S.ENERGY), _lin(_S.MAX_ENERGY, 0.2),
              _sym(_S.STARS), _lin(_S.ORB_CAPACITY, 0.1), _lin(_S.HAND_COUNT, 0.1), _sym(_S.DRAW_COUNT, null=True),
              _sym(_S.DISCARD_COUNT, null=True), _sym(_S.EXHAUST_COUNT, null=True), _miss(_S.DRAW_COUNT),
              _sym(_S.DECK_COUNT), _lin(_S.READY_TO_END_TURN), _lin(_S.TURN_NUMBER, 0.05),
              _lin(_S.ARROW_SLOTS, 0.2), _lin(_S.ARROWS_AVAILABLE, 0.2), *_bits(_S.MELODY_BITS, 3),
              *(_lin(c, 0.1) for c in range(_S.CARDS_PLAYED_TURN, _S.HP_LOSS_EVENTS_COMBAT + 1)),
              _lin("acts")),
        derived=("acts",)),
    Ent.CARD: EntitySpec(
        ids=(Id(_C.ZONE, "p:zone"), Id(_C.SEAT, "p:seat", 1), Id(_C.POS, "p:pos", 1), Id(_C.CARD_ID, "card"),
             Id("static", "@"), Id(_C.ENERGY_COST_COLOR, "enum:CardCostColor"),
             Id(_C.STAR_COST_COLOR, "enum:CardCostColor#star"), Id(_C.ENCHANT_ID, "enchantment"),
             Id(_C.ENCHANT_STATUS, "enum:EnchantmentStatus"), Id(_C.AFFLICTION_ID, "affliction"),
             Id("pile", CARD_CHOICE_AUX_VOCAB), Id("group", "p:group", 1)),
        nums=(_sym(_C.COUNT), _lin(_C.UPGRADE_LEVEL, 0.5), _lin(_C.ENERGY_COST, 0.2),
              *(Num(_C.ENERGY_COST, Mode.EQ, arg=v) for v in range(6)), *_bits(_C.COST_FLAGS, 5),
              _lin(_C.STAR_COST, 0.2, True), _miss(_C.STAR_COST), *_bits(_C.KEYWORD_BITS, 7),
              _sym(_C.ENCHANT_AMOUNT, null=True), _miss(_C.ENCHANT_AMOUNT), _sym(_C.AFFLICTION_AMOUNT),
              _lin(_C.REPLAY_COUNT, 1 / 3), *_bits(_C.HAND_FLAGS, 4), *_bits(_C.UNPLAYABLE_REASON, 12),
              Num(_C.SEL_COUNT, Mode.RATIO, src2=_C.COUNT), _lin(_C.SEL_COUNT, 0.2)),
        derived=("static", "pile", "group")),
    Ent.CREATURE: EntitySpec(
        ids=(Id(CreatureCol.KIND, "p:creature_kind"), Id(CreatureCol.SEAT, "p:seat", 1),
             Id(CreatureCol.POS, "p:pos", 1), Id("model", "@"), Id("static", "@"),
             Id(CreatureCol.HP_DISPLAY, "enum:HpDisplay"), Id(CreatureCol.SLOT_ID, "slot_name")),
        nums=(Num(CreatureCol.HP, Mode.RATIO, src2=CreatureCol.MAX_HP, null=True), _sym(CreatureCol.HP, null=True),
              _miss(CreatureCol.HP), _sym(CreatureCol.MAX_HP, null=True), _sym(CreatureCol.BLOCK),
              _lin(CreatureCol.ALIVE), _lin(CreatureCol.HITTABLE), _sym(CreatureCol.POISON_NEXT_TURN),
              _lin("poison_lethal"), _lin(CreatureCol.POWER_CNT, 0.1), _lin(CreatureCol.INTENT_CNT, 1 / 3)),
        derived=("model", "static", "poison_lethal")),
    Ent.RELIC: EntitySpec(
        ids=(Id(_R.SEAT, "p:seat", 1), Id(_R.ZONE, "p:item_zone"), Id(_R.RELIC_ID, "relic"), Id("static", "@"),
             Id(_R.STATUS, "enum:RelicStatus")),
        nums=(_lin(_R.POS, 1 / 30), _sym(_R.COUNTER, null=True), _miss(_R.COUNTER), *_bits(_R.FLAGS, 3)),
        derived=("static",)),
    Ent.POTION: EntitySpec(
        ids=(Id(PotionCol.SEAT, "p:seat", 1), Id(PotionCol.ZONE, "p:item_zone"), Id(PotionCol.SLOT, "p:pos", 1),
             Id(PotionCol.POTION_ID, "potion"), Id("static", "@")),
        nums=_bits(PotionCol.FLAGS, 3),
        derived=("static",)),
    Ent.ORB: EntitySpec(
        ids=(Id(OrbCol.SEAT, "p:seat", 1), Id(OrbCol.POS, "p:pos", 1), Id(OrbCol.ORB_ID, "orb")),
        nums=(_sym(OrbCol.PASSIVE), _sym(OrbCol.EVOKE))),
    Ent.ARROW: EntitySpec(
        ids=(Id(ArrowCol.SEAT, "p:seat", 1), Id(ArrowCol.SLOT, "p:pos", 1), Id(ArrowCol.STATUS, "enum:ArrowStatus")),
        nums=()),
    Ent.MAP_NODE: EntitySpec(
        ids=(Id(MapNodeCol.COL, "p:map_col", 1), Id(MapNodeCol.POINT_TYPE, "enum:MapPointType"),
             Id(MapNodeCol.STATE, "p:map_state"), Id(MapNodeCol.RESOLVED_ROOM, "enum:RoomType#resolved")),
        nums=(_lin(MapNodeCol.ROW, 0.05), _lin("drow", 0.05), *_bits(MapNodeCol.FLAGS, 8)),
        derived=("drow",)),
    Ent.OPTION: EntitySpec(
        ids=(Id(_O.KIND, "p:option_kind"), Id(_O.SEAT, "p:seat", 1), Id(_O.POS, "p:pos", 1), Id("key", "@"),
             Id(_O.GROUP, "p:group", 1)),
        nums=(*_bits(_O.FLAGS, 7), _sym(_O.PRICE, null=True), _miss(_O.PRICE), _sym(_O.AMOUNT, null=True),
              _miss(_O.AMOUNT), _miss(_O.REF_TABLE)),
        derived=("key",)),
    Ent.CELL: EntitySpec(
        ids=(Id(CellCol.X, "p:grid", 1), Id(CellCol.Y, "p:grid#y", 1),
             Id(CellCol.ITEM_TYPE, "enum:CrystalSphereItemType"), Id(CellCol.ITEM_IDX, "p:group", 1)),
        nums=(_lin(CellCol.HIDDEN), _lin(CellCol.ITEM_GOOD, null=True), _miss(CellCol.ITEM_GOOD))),
    Ent.POWER: EntitySpec(
        ids=(Id(_P.POWER_ID, "power"), Id("static", "@"), Id(_P.STACK_TYPE, "enum:PowerStackType"),
             Id(_P.TYPE_NOW, "enum:PowerType"), Id("applier_kind", "p:creature_kind")),
        nums=(_sym(_P.AMOUNT), *_bits(_P.VISIBLE_MASK, 4), _lin(_P.ORDER, 0.1), _miss(_P.APPLIER_ROW)),
        derived=("static", "applier_kind")),
    Ent.INTENT: EntitySpec(
        ids=(Id(IntentCol.INTENT_TYPE, "enum:IntentType"), Id(IntentCol.IDX, "p:intent_idx", 1)),
        nums=(_lin(IntentCol.HITS, 0.2), _lin(IntentCol.CARD_COUNT, 0.2),
              *(_sym(c, null=True) for c in range(IntentCol.DMG_SEAT0, IntentCol.DMG_SEAT3 + 1)),
              *(_miss(c) for c in range(IntentCol.DMG_SEAT0, IntentCol.DMG_SEAT3 + 1)),
              *(Num(c, Mode.SYMPROD, src2=IntentCol.HITS, null=True)
                for c in range(IntentCol.DMG_SEAT0, IntentCol.DMG_SEAT3 + 1)),
              *_bits(IntentCol.FLAGS, 1))),
    Ent.REF: EntitySpec(
        ids=(Id(RefCol.CATEGORY, "p:ref_category"), Id("vocab", "@")),
        nums=(_miss(RefCol.ENT_TABLE),),
        derived=("vocab",)),
    Ent.VAR: EntitySpec(
        ids=(Id(VarCol.VAR_ID, "var_name"), Id(VarCol.OWNER_TABLE, "p:table")),
        nums=(_sym(VarCol.SHOWN), Num(VarCol.SHOWN, Mode.SYMDIFF, src2=VarCol.ENCHANTED),
              Num(VarCol.ENCHANTED, Mode.SYMDIFF, src2=VarCol.BASE), *_bits(VarCol.FLAGS, 3),
              _miss(VarCol.TARGET_ROW))),
    Ent.EDGE_OUT: EntitySpec(ids=(), nums=()),
    Ent.EDGE_IN: EntitySpec(ids=(), nums=()),
    Ent.ACTION: EntitySpec(
        ids=(Id(ActionCol.VERB, "p:verb"), Id(ActionCol.SEAT, "p:seat", 1),
             Id(ActionCol.AUX, CRYSTAL_ACTION_AUX_VOCAB)),
        nums=(_miss(ActionCol.SRC_ROW), _miss(ActionCol.TGT_ROW))),
}

# FORECAST rows (notes/forecast/contract.md 3.2): not in SPECS, so the legacy wide columns stay where they are.
# nums 0-14 mirror INTENT's 15 on the mirrored columns (cols 0-9 of FORECAST are INTENTS cols 0-9) and reuse its
# wide columns; nums 15-26 get 12 new wide columns after every other column (TokenLayout.forecast_cols). (The
# contract says "19 / 31": INTENT has 15 nums - the list in the contract is the same.)
_F = ForecastCol
FORECAST_SPEC: EntitySpec = EntitySpec(
    ids=(Id(_F.INTENT_TYPE, "enum:IntentType"), Id(_F.IDX, "p:intent_idx", 1)),
    nums=(*SPECS[Ent.INTENT].nums,
          Num(_F.FLAGS, Mode.BIT, arg=1), Num(_F.FLAGS, Mode.BIT, arg=2), Num(_F.AHEAD, Mode.EQ, arg=2),
          Num(_F.CLASS, Mode.EQ, arg=1), Num(_F.CLASS, Mode.EQ, arg=2), Num(_F.CLASS, Mode.EQ, arg=3),
          _lin(_F.PROB, 1e-4), _lin(_F.N_CAND, 0.25),
          *(Num(c, Mode.SYMPROD, 1e-4, src2=_F.PROB, null=True) for c in range(_F.DMG_SEAT0, _F.DMG_SEAT3 + 1))))
FORECAST_NEW_NUMS = 12
assert len(SPECS[Ent.INTENT].nums) + FORECAST_NEW_NUMS == len(FORECAST_SPEC.nums) == 27

# Calc rows (notes/calc/contract.md 3.2): no ids, no legacy column; 17 + 10 + 13 = 40 new wide columns after every
# other column (TokenLayout.calc_cols), projected by the model's own zero-gated matrix.
_TC, _FC, _RC = TurnCol, FoeCol, RouteCol
CALC_SPECS: Mapping[Ent, EntitySpec] = {
    Ent.TURN: EntitySpec(ids=(), nums=(
        _sym(_TC.HP_LOSS_NOW), _sym(_TC.HP_LOSS_MIN), _sym(_TC.INCOMING), _sym(_TC.BLOCK), _sym(_TC.BLOCK_REACHABLE),
        _sym(_TC.END_BLOCK), _sym(_TC.END_DMG), _sym(_TC.END_LOSS),
        Num(_TC.HP_LOSS_NOW, Mode.RATIO, src2=_TC.HP), Num(_TC.HP_LOSS_MIN, Mode.RATIO, src2=_TC.HP),
        Num(_TC.INCOMING, Mode.SYMDIFF, src2=_TC.BLOCK), Num(_TC.HP_LOSS_NOW, Mode.SYMDIFF, src2=_TC.HP_LOSS_MIN),
        *_bits(_TC.FLAGS, 5))),
    Ent.FOE: EntitySpec(ids=(), nums=(
        _sym(_FC.TO_KILL, null=True), _miss(_FC.TO_KILL), _sym(_FC.DMG_REACHABLE), _sym(_FC.INCOMING_FROM),
        Num(_FC.TO_KILL, Mode.SYMDIFF, src2=_FC.DMG_REACHABLE, null=True), *_bits(_FC.FLAGS, 5))),
    Ent.ROUTE: EntitySpec(ids=(), nums=(
        _lin(_RC.ROWS_TO_BOSS, 1 / 16), _lin(_RC.REST_DIST, 1 / 8, True), _miss(_RC.REST_DIST),
        _lin(_RC.SHOP_DIST, 1 / 8, True), _miss(_RC.SHOP_DIST), _lin(_RC.ELITES_TO_REST, 1 / 2, True),
        _lin(_RC.FIGHTS_TO_REST, 1 / 4, True), _lin(_RC.ELITES_TO_BOSS, 1 / 4), _lin(_RC.FIGHTS_TO_BOSS, 1 / 8),
        _lin(_RC.N_REST, 1 / 4), _lin(_RC.N_SHOP, 1 / 2), _lin(_RC.N_ELITE, 1 / 4), _lin(_RC.N_UNKNOWN, 1 / 8))),
}
CALC_N_NUMS = 40
assert tuple(len(CALC_SPECS[e].nums) for e in CALC_ENTS) == (17, 10, 13)

# AHEAD (notes/inputs2/contract.md 2.2-2.3, v2a.md 2): one kid row of the GLOBAL token per OBS / TERMINAL frame; its
# 30 slots are counts, minima and maxima over the frame's ROUTE rows (read before the DROP rule) and the POINT_TYPE /
# FLAGS of their nodes. No ids, no legacy column: 35 wide columns after the calc block (TokenLayout.ahead_cols).
# Slots: 0 N; 1-6 NEXT_<type> (AHEAD_TYPES); 7 NEXT_BOSS; 8 BOSS_ROWS_MIN; 9-28 MIN / MAX of AHEAD_PAIRS; 29 NEXT_QUEST.
AHEAD_SLOTS = 30
AHEAD_TYPES = ("Monster", "Elite", "RestSite", "Shop", "Treasure", "Unknown")  # enum:MapPointType names, slots 1-6
AHEAD_PAIRS = (_RC.REST_DIST, _RC.SHOP_DIST, _RC.ELITES_TO_REST, _RC.FIGHTS_TO_REST, _RC.ELITES_TO_BOSS,
               _RC.FIGHTS_TO_BOSS, _RC.N_REST, _RC.N_SHOP, _RC.N_ELITE, _RC.N_UNKNOWN)  # slots 9 + 2k (MIN), 10 + 2k (MAX)
AHEAD_SPEC: EntitySpec = EntitySpec(ids=(), nums=(
    _lin(0, 1 / 4), Num(0, Mode.EQ, arg=0), *(_lin(k, 1 / 2) for k in range(1, 8)), _lin(8, 1 / 16, True),
    _lin(9, 1 / 8, True), _miss(9), _lin(10, 1 / 8, True), _miss(10),
    _lin(11, 1 / 8, True), _miss(11), _lin(12, 1 / 8, True), _miss(12),
    _lin(13, 1 / 2, True), _lin(14, 1 / 2, True), _lin(15, 1 / 4, True), _lin(16, 1 / 4, True),
    _lin(17, 1 / 4, True), _lin(18, 1 / 4, True), _lin(19, 1 / 8, True), _lin(20, 1 / 8, True),
    _lin(21, 1 / 4, True), _lin(22, 1 / 4, True), _lin(23, 1 / 2, True), _lin(24, 1 / 2, True),
    _lin(25, 1 / 4, True), _lin(26, 1 / 4, True), _lin(27, 1 / 8, True), _lin(28, 1 / 8, True),
    _lin(29, 1 / 2)))
AHEAD_N_NUMS = 35
assert len(AHEAD_SPEC.nums) == AHEAD_N_NUMS and len(AHEAD_PAIRS) == 10

# Vocab categories reached only through derived ("@") slots.
_RESOLVED_CATEGORIES = tuple(dict.fromkeys((*CREATURE_MODEL_VOCAB.values(), *OPTION_KEY_VOCAB.values(),
                                             *REF_CATEGORY_VOCAB.values())))


# =========================================================================== layout


class TokenLayout:
    """Embedding-row layout and per-entity-type slot programs, derived from the vocab.

    Global embedding row 0 is an all-zero row; every table key of :data:`SPECS` (plus the vocab
    categories reached through polymorphic columns) gets a contiguous region of learned rows,
    followed by the static-prior regions (:data:`STATIC_CATEGORIES`, fed from the catalog).
    """

    def __init__(self, vocab: Vocab, *, public_history: bool = False, cycle_ban: bool = False,
                 coop_features: bool = False, mend_target: bool = False, forecast: bool = False,
                 calc: bool = False, route_summary: bool = False, second_boss_table: bool = False,
                 rules_mode: str = "mask") -> None:
        self.public_history = public_history
        self.cycle_ban = cycle_ban
        self.rules_mode = rules_mode  # spire.rules: the model's removal teach term (and "free" acting) unless "mask"
        self.coop_features = coop_features
        self.mend_target = mend_target
        self.forecast = forecast
        self.calc = calc
        self.route_summary = route_summary
        self.second_boss_table = second_boss_table
        assert not route_summary or calc, "the route summary needs calc"
        # type_emb rows (with calc: 24, row 20 unused without the forecast; with the route summary: 25)
        self.n_types = N_ENT if route_summary else N_ENT_CALC if calc else N_ENT_FORECAST if forecast else N_ENT_LEGACY
        self.specs = dict(SPECS)
        if second_boss_table:  # notes/inputs2/contract.md 4: the second boss id indexes a table of its own
            g = SPECS[Ent.GLOBAL]
            ids = tuple(Id(i.src, "encounter#second") if i.src == G.SECOND_BOSS_ENCOUNTER else i for i in g.ids)
            assert sum(i.table == "encounter#second" for i in ids) == 1
            self.specs[Ent.GLOBAL] = EntitySpec(ids, g.nums, g.derived)
        if coop_features:
            seat, option = SPECS[Ent.SEAT], SPECS[Ent.OPTION]
            # Pack acts | screen<<1 | status<<5 into the existing acts slot to keep
            # the compact record width unchanged. Screen uses a separate model
            # embedding, preserving every existing ID and ID reduction operation.
            assert seat.nums[-1] == _lin("acts")
            self.specs[Ent.SEAT] = EntitySpec(
                seat.ids, seat.nums[:-1] + (Num("acts", Mode.BIT),) +
                tuple(Num("acts", Mode.BIT, arg=5 + bit) for bit in range(6)), seat.derived)
            self.specs[Ent.OPTION] = EntitySpec(option.ids, option.nums + (_lin("vote_count", 0.25),),
                                               option.derived + ("vote_count",))
        if forecast:
            self.specs[Ent.FORECAST] = FORECAST_SPEC
        if calc:
            self.specs.update(CALC_SPECS)
        if route_summary:
            self.specs[Ent.AHEAD] = AHEAD_SPEC
        self.vocab_hash = vocab.hash
        self.regions: dict[str, tuple[int, int]] = {}
        n = 1
        keys = [i.table for e in Ent for i in SPECS.get(e, EntitySpec((), ())).ids if i.table != "@"]
        # tables only this layout's specs name ("encounter#second") follow every other learned region (contract 4)
        own = [i.table for e in Ent for i in self.specs.get(e, EntitySpec((), ())).ids if i.table != "@"]
        for key in dict.fromkeys([*keys, *_RESOLVED_CATEGORIES, *own]):
            stem = key.split("#")[0]
            rows = PROTOCOL_ROWS[stem] if stem.startswith("p:") else vocab.embedding_rows(stem)
            self.regions[key] = (n, rows)
            n += rows
        self.n_learned = n
        self.static: dict[str, tuple[int, int, int]] = {}  # cat -> (offset, id rows, levels)
        for cat, levels in STATIC_CATEGORIES.items():
            rows = vocab.embedding_rows(cat)
            self.static[cat] = (n, rows, levels)
            n += rows * levels
        if n - 1 > _I16_MAX:  # "@" slots carry global row indices through int16 Sample.raw
            raise ValueError(f"embedding layout has {n} rows; derived id slots are int16 (max {_I16_MAX})")
        self.n_rows = n
        self.character_rows = vocab.embedding_rows("character")

        # Polymorphic resolution tables (discriminator value -> region offset / rows).
        self.opt_key_off, self.opt_key_rows = self._poly(OPTION_KEY_VOCAB, 9)
        self.ref_off, self.ref_rows = self._poly(REF_CATEGORY_VOCAB, 10)
        self.creature_off, self.creature_rows = self._poly(CREATURE_MODEL_VOCAB, 4)

        # Raw slot program per entity type.
        self.slot_cols: dict[Ent, I64] = {}  # table/global columns copied into slots 0..k-1
        self.derived_slot: dict[Ent, dict[str, int]] = {}
        width = 1
        for e in Ent:
            if e == Ent.PAD or e not in self.specs:
                self.slot_cols[e], self.derived_slot[e] = np.zeros(0, np.int64), {}
                continue
            spec = self.specs[e]
            srcs = [i.src for i in spec.ids] + [s for nm in spec.nums for s in (nm.src, nm.src2) if s is not None]
            unknown = {s for s in srcs if isinstance(s, str)} - set(spec.derived)
            if unknown:
                raise ValueError(f"{e.name}: undeclared derived columns {unknown}")
            cols = list(dict.fromkeys(int(s) for s in srcs if not isinstance(s, str)))
            self.slot_cols[e] = np.array(cols, np.int64)
            self.derived_slot[e] = {d: len(cols) + j for j, d in enumerate(spec.derived)}
            width = max(width, len(cols) + len(spec.derived))
        self.width = width
        # FORECAST adds no slot width: Sample.raw and history_width stay the same (contract 3.2)
        assert not forecast or len(self.slot_cols[Ent.FORECAST]) == 13 <= max(
            len(self.slot_cols[e]) + len(self.derived_slot[e]) for e in Ent if e != Ent.FORECAST)
        # calc adds no slot width either (contract 3.2): 10 / 4 / 11 slot columns
        assert not calc or tuple(len(self.slot_cols[e]) for e in CALC_ENTS) == (10, 4, 11) and max(
            len(self.slot_cols[e]) for e in CALC_ENTS) <= max(
            len(self.slot_cols[e]) + len(self.derived_slot[e]) for e in Ent if e not in CALC_ENTS)
        # nor does AHEAD (notes/inputs2/contract.md 2.2): its 30 slots fit the record width of the other types
        assert not route_summary or len(self.slot_cols[Ent.AHEAD]) == AHEAD_SLOTS <= max(
            len(self.slot_cols[e]) + len(self.derived_slot[e]) for e in Ent if e != Ent.AHEAD)
        self.n_id_slots = max(len(s.ids) for s in SPECS.values())
        self.n_num = max(len(s.nums) for s in self.specs.values())
        self.num_offsets: dict[Ent, int] = {}
        f = 0
        for e in Ent:
            self.num_offsets[e] = f
            f += len(SPECS[e].nums) if e in SPECS else 0
        self.legacy_n_wide = f
        self.num_columns = {e: list(range(self.num_offsets[e], self.num_offsets[e] + len(s.nums)))
                            for e, s in SPECS.items()}
        # Append extension columns after the entire legacy block, never between entity
        # blocks: stage-A columns (including INTENT missingness) retain their indices.
        for e, spec in self.specs.items():
            if e == Ent.FORECAST or e in CALC_ENTS or e == Ent.AHEAD:
                continue
            extra = len(spec.nums) - len(SPECS[e].nums)
            self.num_columns[e].extend(range(f, f + extra))
            f += extra
        # FORECAST: INTENT's columns, then its own block after everything else (the model projects it with
        # its own zero-gated matrix, so in_proj keeps its width: wide[:, :forecast_cols[0]]).
        self.forecast_cols = (f, f)
        if forecast:
            assert len(self.num_columns[Ent.INTENT]) == len(SPECS[Ent.INTENT].nums)
            self.num_columns[Ent.FORECAST] = self.num_columns[Ent.INTENT] + list(range(f, f + FORECAST_NEW_NUMS))
            self.forecast_cols = (f, f + FORECAST_NEW_NUMS)
            f += FORECAST_NEW_NUMS
        # calc: 40 columns after everything, the forecast block included (TURN +0..16, FOE +17..26, ROUTE +27..39)
        self.calc_cols = (f, f)
        if calc:
            for e in CALC_ENTS:
                k = len(CALC_SPECS[e].nums)
                self.num_columns[e] = list(range(f, f + k))
                f += k
            self.calc_cols = (self.forecast_cols[1], f)
            assert f - self.forecast_cols[1] == CALC_N_NUMS
        # AHEAD: 35 columns after the calc block (notes/inputs2/contract.md 2.3, v2a.md 2)
        self.ahead_cols = (f, f)
        if route_summary:
            self.num_columns[Ent.AHEAD] = list(range(f, f + AHEAD_N_NUMS))
            self.ahead_cols = (self.calc_cols[1], f + AHEAD_N_NUMS)
            f += AHEAD_N_NUMS
        self.n_wide = f

    def _poly(self, mapping: Mapping[Any, str], n: int) -> tuple[I64, I64]:
        off, rows = np.zeros(n, np.int64), np.zeros(n, np.int64)
        for k, cat in mapping.items():
            off[int(k)], rows[int(k)] = self.regions[cat]
        return off, rows

    def slot(self, e: Ent, src: Src) -> int:
        if isinstance(src, str):
            return self.derived_slot[e][src]
        return int(np.flatnonzero(self.slot_cols[e] == int(src))[0])

    def decoder_tables(self) -> dict[str, NDArray[Any]]:
        """Per-type programs for :class:`FeatureDecoder` (arrays of shape (N_ENT, K) / (N_ENT, F))."""
        K, F = self.n_id_slots, self.n_num
        t: dict[str, NDArray[Any]] = {
            "id_slot": np.zeros((N_ENT, K), np.int64), "id_add": np.zeros((N_ENT, K), np.int64),
            "id_max": np.zeros((N_ENT, K), np.int64), "id_off": np.zeros((N_ENT, K), np.int64),
            "num_src": np.zeros((N_ENT, F), np.int64), "num_src2": np.zeros((N_ENT, F), np.int64),
            "num_mode": np.zeros((N_ENT, F), np.int64), "num_arg": np.zeros((N_ENT, F), np.int32),
            "num_scale": np.zeros((N_ENT, F), np.float32), "num_null": np.zeros((N_ENT, F), np.bool_),
            "num_wcol": np.full((N_ENT, F), self.n_wide, np.int64),
        }
        for e, spec in self.specs.items():
            for k, i in enumerate(spec.ids):
                t["id_slot"][e, k] = self.slot(e, i.src)
                if i.table == "@":
                    t["id_max"][e, k] = self.n_rows - 1
                else:
                    off, rows = self.regions[i.table]
                    t["id_add"][e, k], t["id_max"][e, k], t["id_off"][e, k] = i.add, rows - 1, off
            for j, nm in enumerate(spec.nums):
                t["num_src"][e, j] = self.slot(e, nm.src)
                t["num_src2"][e, j] = self.slot(e, nm.src2) if nm.src2 is not None else 0
                t["num_mode"][e, j], t["num_arg"][e, j] = nm.mode, nm.arg
                t["num_scale"][e, j], t["num_null"][e, j] = nm.scale, nm.null
                t["num_wcol"][e, j] = self.num_columns[e][j]
        return t


def _resolve(off: I64, rows: I64, disc: NDArray[np.integer], ids: NDArray[np.integer]) -> I64:
    """Global row of vocab id ``ids`` in the region selected by discriminator ``disc``.

    0 (the zero row) for PAD ids and for discriminators without a vocab category. Ids are trusted
    to lie inside their category (the frame's vocab hash is checked against the loaded vocab, and
    every region has >= 128 rows of slack).
    """
    o = off.take(disc, mode="clip")
    r = rows.take(disc, mode="clip")
    return np.where((r > 0) & (ids > 0), o + ids, 0)


def _ref(base: I64, table: NDArray[np.integer], row: NDArray[np.integer]) -> I64:
    """Entity row of (table, row) references; -1 for null references or tables without entities."""
    b = base.take(table, mode="clip")
    return np.where((table >= 0) & (b >= 0), b + row, -1)


# =========================================================================== samples


@dataclass(frozen=True)
class TokenizerConfig:
    pool_children: bool = True
    """Pool POWERS / INTENTS / REFS / untargeted VARS into their owners (default) or emit them as tokens."""
    token_buckets: tuple[int, ...] = (32, 48, 64, 80, 112, 160, 224, 320, 448, 640)
    """Padded token counts; longer samples round up to a multiple of 64."""
    action_align: int = 8
    child_align: int = 64
    public_history: bool = False
    cycle_ban: bool = False
    mend_target: bool = False
    """Link MEND actions to the target SEATS token behind the model's zero-initialized gate.

    Preserve their legacy auxiliary IDs for a function-preserving checkpoint introduction.
    """
    coop_features: bool = False
    """Expose each seat's screen/status and the permutation-invariant OPTION vote count."""
    commit_select: bool = False
    """No CANCEL in an out-of-combat upgrade / enchant card prompt (needs ``cycle_ban``: same mask slots).

    The seat opened the prompt on purpose (rest-site SMITH); backing out only returns to the screen it came from.
    """
    forecast: bool = False
    """Encode the FORECAST trailer table (moves of the next two enemy turns) as kid rows pooled into their creature.

    Frames must carry it (RESET with ``ResetFlag.FORECAST``: ``schema.reset_flags_for``) iff this is set; needs
    ``pool_children``. notes/forecast/contract.md.
    """
    calc: bool = False
    """Encode the calc trailer tables TURN / FOE / ROUTE (turn-end and per-enemy sums, route summary) as kid rows
    pooled into the acting player's creature, the enemy's creature and the map node.

    Frames must carry them (RESET with ``ResetFlag.CALC``: ``schema.reset_flags_for(..., calc=True)``) iff this is
    set; needs ``pool_children``. notes/calc/contract.md.
    """
    removal_rule: bool = False
    """Out-of-combat DECK_REMOVAL prompts keep one legal row, the removal rule's (spire.removal; needs ``cycle_ban``:
    same mask slots, and ``removal_ratings``)."""
    removal_ratings: str = ""
    """The rule's table (removal-ratings.json, copied into the run directory); set iff ``removal_rule``."""
    map_tokens: str = "frame"
    """``"frame"``: the map tables as the frame carries them. ``"no_combat"``: the DROP rule - a frame with
    GLOBAL[IN_COMBAT] = 1 and GLOBAL[KIND] != MAP is tokenized as if MAP_NODES, MAP_EDGES and ROUTE had 0 rows, with
    assertion A1 (no ACTIONS / OPTIONS / REFS row references a MAP_NODES row there). For a network fed by hosts with
    RESET bit 1 (MAP_IN_COMBAT). notes/inputs2/contract.md 1; ``"decision"`` is not built (notes/inputs2/v2a.md 1)."""
    route_summary: bool = False
    """One AHEAD kid row of the GLOBAL token in every frame: what lies ahead on the map, from the ROUTE rows and their
    nodes (notes/inputs2/contract.md 2, v2a.md 2). Needs ``calc``, ``pool_children``, ``map_tokens = "no_combat"``
    and frames with the map in fights (RESET bit 1: ``schema.reset_flags_for(..., map_in_combat=True)``)."""
    second_boss_table: bool = False
    """GLOBAL[SECOND_BOSS_ENCOUNTER] indexes its own embedding table ``encounter#second`` (notes/inputs2/contract.md
    4), so the two act-3 boss ids are no longer summed from one table."""
    rules_mode: str = "mask"
    """What the removal rule does (spire.rules, notes/next/p4.md): "mask" (its one legal row) | "teach" (the same
    acting, plus the removal teach term in the loss) | "free" (the removal prompt offers every row the host offers and
    the commit / cycle masks leave; the term continues, fading). "teach" and "free" need ``cycle_ban`` and
    ``removal_rule`` (the rule is the teacher)."""

    def __post_init__(self) -> None:
        assert self.removal_rule == bool(self.removal_ratings), "tokenizer.removal_rule iff tokenizer.removal_ratings"
        assert self.cycle_ban or not self.removal_rule, "removal_rule uses the cycle-ban mask"
        assert self.rules_mode in ("mask", "teach", "free"), f"tokenizer.rules_mode {self.rules_mode!r}"
        assert self.rules_mode == "mask" or (self.cycle_ban and self.removal_rule),             "tokenizer.rules_mode teach / free needs cycle_ban and removal_rule"
        assert self.map_tokens in ("frame", "no_combat"), \
            f"tokenizer.map_tokens {self.map_tokens!r}: 'frame' or 'no_combat' (notes/inputs2/v2a.md 1)"
        assert not self.route_summary or (self.calc and self.pool_children and self.map_tokens == "no_combat"), \
            "tokenizer.route_summary needs calc, pool_children and map_tokens = 'no_combat'"

    def bucket(self, n_tokens: int) -> int:
        for e in self.token_buckets:
            if n_tokens <= e:
                return e
        return -(-n_tokens // 64) * 64


class Sample:
    """Compact per-decision record (numpy only; no reference to the frame payload).

    * ``types`` (N,) int8 - entity type (:class:`Ent`) of the N entity rows, in group order
      tokens | kids | vars | actions (``counts`` rows each);
    * ``raw`` (N, W) int16 - raw slots (table columns + derived columns, saturated to int16);
    * ``rel`` (N, 3) int16 - relation indices into the sample's rows, -1 = none (module docstring);
    * ``edges`` (2E, 3) int16 - map edges as (owner node token, other node token, direction 0 out / 1 in);
    * ``cond`` (6,) int16 - ascension, n_seats, character vocab id of seats 0..3 (0 = no seat).
    """

    __slots__ = ("types", "raw", "rel", "edges", "cond", "counts", "history")

    def __init__(self, types: NDArray[np.int8], raw: NDArray[np.int16], rel: NDArray[np.int16],
                 edges: NDArray[np.int16], cond: NDArray[np.int16], counts: tuple[int, int, int, int],
                 history: NDArray[np.int16] | None = None) -> None:
        self.types, self.raw, self.rel, self.edges, self.cond, self.counts = types, raw, rel, edges, cond, counts
        self.history = np.zeros(history_width(raw.shape[1]), np.int16) if history is None else history

    @property
    def n_tokens(self) -> int:
        return self.counts[0]

    @property
    def n_actions(self) -> int:
        return self.counts[3]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.types, self.raw, self.rel, self.edges, self.cond, self.history))

    def __repr__(self) -> str:
        t, k, v, a = self.counts
        return f"Sample(tokens={t} kids={k} vars={v} actions={a} edges={self.edges.shape[0]})"


class SampleBlock(Sequence[Sample]):
    """Several :class:`Sample` records concatenated into shared arrays (one rollout segment).

    Items are views into the block (``block[i]`` is equal to the i-th original sample). The rollout
    store compacts every sealed segment into a block, so per-frame sample views of a whole
    inference batch (:meth:`EncodedBatch.samples`) never keep that batch alive for longer than the
    open segment; :mod:`spire.ipc` ships the block arrays without re-concatenating them.
    """

    __slots__ = ("types", "raw", "rel", "edges", "cond", "counts", "row0", "edge0", "_rows", "_edges", "_cl", "history")

    def __init__(self, types: NDArray[np.int8], raw: NDArray[np.int16], rel: NDArray[np.int16],
                 edges: NDArray[np.int16], cond: NDArray[np.int16], counts: NDArray[np.integer],
                 history: NDArray[np.int16] | None = None) -> None:
        self.types, self.raw, self.rel, self.edges, self.cond = types, raw, rel, edges, cond
        self.counts = np.asarray(counts).reshape(-1, 4)
        self.history = (np.zeros((len(self.counts), history_width(raw.shape[1])), np.int16)
                        if history is None else history)
        n_rows = self.counts.sum(1, dtype=np.int64)
        self.row0 = np.zeros(len(n_rows) + 1, np.int64)
        np.cumsum(n_rows, out=self.row0[1:])
        if self.row0[-1] != types.shape[0]:
            raise ValueError("sample counts do not match the concatenated rows")
        self._rows = self.row0.tolist()
        self._cl = self.counts.tolist()
        self.edge0 = np.zeros(len(n_rows) + 1, np.int64)
        self._edges: list[int] = []

    @classmethod
    def from_samples(cls, samples: Sequence[Sample], edge_counts: Sequence[int] | None = None) -> "SampleBlock":
        if not samples:
            raise ValueError("empty block")
        ty, raw, rel, ed, cond, cnt = zip(*map(_SAMPLE_FIELDS, samples))
        counts = np.fromiter(chain.from_iterable(cnt), np.int64, 4 * len(cnt))
        concat = getattr(_native, "concat_into", None)
        blk: SampleBlock
        if concat is not None and ty[0].dtype == np.int8 and raw[0].dtype == rel[0].dtype == ed[0].dtype == \
                cond[0].dtype == np.int16 and all(a.ndim == 1 for a in cond):
            # one C call per array instead of np.concatenate (~3.5x faster for 256 small views; the
            # same bytes: back-to-back copies of C-contiguous arrays, sizes checked by the kernel)
            n_rows = int(counts.sum())
            n_edges = sum(map(len, ed))
            out_ty = np.empty(n_rows, np.int8)
            out_raw = np.empty((n_rows, raw[0].shape[1]), np.int16)
            out_rel = np.empty((n_rows, rel[0].shape[1]), np.int16)
            out_ed = np.empty((n_edges, ed[0].shape[1]), np.int16)
            out_cond = np.empty((len(cond), cond[0].shape[0]), np.int16)
            try:
                for src, dst in ((ty, out_ty), (raw, out_raw), (rel, out_rel), (ed, out_ed), (cond, out_cond)):
                    concat(src, dst)
            except (BufferError, ValueError):  # non-contiguous or odd-shaped input: the numpy path below
                concat = None
            else:
                blk = cls(out_ty, out_raw, out_rel, out_ed, out_cond, counts)
        if concat is None:
            blk = cls(np.concatenate(ty), np.concatenate(raw), np.concatenate(rel), np.concatenate(ed),
                      np.stack(cond), counts)
        blk.set_edge_counts(list(map(len, ed)) if edge_counts is None else edge_counts)
        blk.history = np.stack([s.history for s in samples])
        return blk

    @classmethod
    def concat(cls, blocks: Sequence["SampleBlock"]) -> "SampleBlock":
        """One block holding every sample of ``blocks`` in order (a learner iteration: the samples of all
        its segments back to back, for :meth:`Tokenizer.collate_block_into`)."""
        if not blocks:
            raise ValueError("empty block list")
        blk = cls(np.concatenate([b.types for b in blocks]), np.concatenate([b.raw for b in blocks]),
                  np.concatenate([b.rel for b in blocks]), np.concatenate([b.edges for b in blocks]),
                  np.concatenate([b.cond.reshape(-1, b.cond.shape[-1]) for b in blocks]),
                  np.concatenate([b.counts for b in blocks]).astype(np.int64),
                  np.concatenate([b.history for b in blocks]))
        blk.set_edge_counts(np.concatenate([np.diff(b.edge0) for b in blocks]))
        return blk

    def set_edge_counts(self, edge_counts: Sequence[int] | NDArray[np.integer]) -> None:
        np.cumsum(np.asarray(edge_counts, np.int64), out=self.edge0[1:])
        if self.edge0[-1] != self.edges.shape[0]:
            raise ValueError("edge counts do not match the concatenated edges")
        self._edges = self.edge0.tolist()

    def __len__(self) -> int:
        return len(self._cl)

    def _item(self, i: int) -> Sample:
        r, e, c = self._rows, self._edges, self._cl[i]
        a, b = r[i], r[i + 1]
        return Sample(self.types[a:b], self.raw[a:b], self.rel[a:b], self.edges[e[i]:e[i + 1]], self.cond[i],
                      (c[0], c[1], c[2], c[3]), self.history[i])

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return [self._item(j) for j in range(*i.indices(len(self)))]
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        return self._item(i)

    def __iter__(self) -> Iterator[Sample]:
        return (self._item(i) for i in range(len(self)))


_SAMPLE_FIELDS = attrgetter("types", "raw", "rel", "edges", "cond", "counts")
_I16_MIN, _I16_MAX = -(1 << 15), (1 << 15) - 1
_NO_EDGES = np.zeros((0, 3), np.int16)

# Hot-path integer constants (IntEnum indexing is measurably slower in numpy).
(_SEATS, _CARDS, _VARS, _CREATURES, _POWERS, _INTENTS, _RELICS, _POTIONS, _ORBS, _ARROWS, _NODES, _EDGES,
 _OPTIONS, _REFS, _CELLS, _ACTIONS) = (int(t) for t in Table)
_A_SEAT, _A_SRC_T, _A_SRC_R, _A_TGT = (int(c) for c in (ActionCol.SEAT, ActionCol.SRC_TABLE, ActionCol.SRC_ROW,
                                                          ActionCol.TGT_ROW))
_V_OWNER_T, _V_OWNER_R, _V_TGT = int(VarCol.OWNER_TABLE), int(VarCol.OWNER_ROW), int(VarCol.TARGET_ROW)
_K_ZONE, _K_ID, _K_UP, _K_LINK, _K_AUX = (int(c) for c in (CardCol.ZONE, CardCol.CARD_ID, CardCol.UPGRADE_LEVEL,
                                                             CardCol.LINK_ROW, CardCol.AUX))
_CR_KIND, _CR_MODEL, _CR_HP, _CR_POISON = (int(c) for c in (CreatureCol.KIND, CreatureCol.MODEL_ID, CreatureCol.HP,
                                                             CreatureCol.POISON_NEXT_TURN))
_P_ID, _P_OWNER, _P_APPLIER = int(PowerCol.POWER_ID), int(PowerCol.OWNER_ROW), int(PowerCol.APPLIER_ROW)
_O_KIND, _O_KEY, _O_REF_T, _O_REF_R = (int(c) for c in (OptionCol.KIND, OptionCol.KEY_ID, OptionCol.REF_TABLE,
                                                         OptionCol.REF_ROW))
_ZONE_CHOICE, _ZONE_OFFER, _PLAYER = int(Zone.CHOICE), int(Zone.OFFER), int(CreatureKind.PLAYER)

# ---- batch encoder constants
_G_KIND, _G_SUB, _G_PCAT, _G_PSID, _G_CUR_ROW, _G_ASC, _G_NSEATS = (
    int(g) for g in (G.KIND, G.KIND_SUB, G.PROMPT_SOURCE_CAT, G.PROMPT_SOURCE_ID, G.CUR_ROW, G.ASCENSION, G.N_SEATS))
_S_CROW, _S_CHAR = int(SeatCol.CREATURE_ROW), int(SeatCol.CHARACTER_ID)
_RL_ID, _PT_ID, _MN_ROW = int(RelicCol.RELIC_ID), int(PotionCol.POTION_ID), int(MapNodeCol.ROW)
_I_OWNER = int(IntentCol.OWNER_ROW)
_RF_CAT, _RF_VOC, _RF_OT, _RF_OR, _RF_ET, _RF_ER = (int(c) for c in (RefCol.CATEGORY, RefCol.VOCAB_ID,
                                                                    RefCol.OWNER_TABLE, RefCol.OWNER_ROW,
                                                                    RefCol.ENT_TABLE, RefCol.ENT_ROW))
_E_FROM, _E_TO = int(MapEdgeCol.FROM_NODE_ROW), int(MapEdgeCol.TO_NODE_ROW)
_DIR_W0, _GLOB_W0, _DATA_W0 = DIR_OFFSET // WORD, GLOBAL_OFFSET // WORD, DATA_OFFSET // WORD
_DIR_IDX = np.arange(_DIR_W0, _DIR_W0 + N_TABLES * DIR_STRIDE, dtype=np.int64)
_DIR_CNT_IDX = _DIR_IDX[1::DIR_STRIDE].copy()  # directory word holding each table's row count
_GLOB_IDX = np.arange(_GLOB_W0, _GLOB_W0 + GLOBAL_LEN, dtype=np.int64)
_NCOLS_A = np.array(N_COLS, np.int64)
_F_ELIDED, _MSG_OBS, _MSG_TERMINAL = int(FrameFlag.ELIDED), int(MsgType.OBS), int(MsgType.TERMINAL)
# frame index * _FRAME_KEY separates the (src_table, src_row, tgt) match keys of different frames:
# |key| < 2**49.01 for any int32 column values, so keys of different frames never collide.
_FRAME_KEY = 1 << 51
_MAX_ENCODE_BATCH = (1 << 12) - 1
_FORECAST = FORECAST_TABLE  # 16: the trailer table, a 17th table to the batch encoder when forecast is on
_FC_OWNER = int(ForecastCol.OWNER_ROW)
_H_NTAB = int(H.N_TABLES)
_NCOLS_A17 = np.array((*N_COLS, FORECAST_N_COLS), np.int64)
_TURN, _FOE, _ROUTE = TURN_TABLE, FOE_TABLE, ROUTE_TABLE  # 17, 18, 19: calc trailer tables
_N_TAB_ALL = ROUTE_TABLE + 1  # 20: v1 + FORECAST + calc (the batch encoder's table count with calc)
_AHEAD = _N_TAB_ALL  # 20: the AHEAD row, one more "table" of the batch encoder (1 row per frame, AHEAD_SLOTS columns)
_N_TAB_AH = _AHEAD + 1
_NCOLS_A21 = np.array((*N_COLS, FORECAST_N_COLS, *CALC_N_COLS, AHEAD_SLOTS), np.int64)
_NCOLS_A20 = np.array((*N_COLS, FORECAST_N_COLS, *CALC_N_COLS), np.int64)
_TU_OWNER, _FO_OWNER, _RO_OWNER = int(TurnCol.OWNER_ROW), int(FoeCol.OWNER_ROW), int(RouteCol.OWNER_ROW)
_G_IN_COMBAT, _KIND_MAP, _H_STEP = int(G.IN_COMBAT), int(Kind.MAP), int(H.STEP_ID)
_MN_PT, _MN_FLAGS = int(MapNodeCol.POINT_TYPE), int(MapNodeCol.FLAGS)
_FL_BOSS = int(MapNodeFlag.BOSS | MapNodeFlag.SECOND_BOSS)
_FL_QUEST = int(MapNodeFlag.QUEST)
_RC_TO_BOSS = int(RouteCol.ROWS_TO_BOSS)
_RC_PAIRS = tuple(int(c) for c in AHEAD_PAIRS)
_ROUTE_NC = CALC_N_COLS[2]
assert CALC_TABLES == (_TURN, _FOE, _ROUTE) and _FORECAST + 1 == _TURN
_TOK_TABS = (_SEATS, _CARDS, _CREATURES, _RELICS, _POTIONS, _ORBS, _ARROWS, _NODES, _OPTIONS, _CELLS)
_KID_TABS = (_POWERS, _INTENTS, _REFS)
_TOK_TABS_A = np.array(_TOK_TABS, np.int64)
_BT_TABS = _TOK_TABS + _KID_TABS  # tables whose rows are entity rows, in segment order after GLOBAL/DECISION
_BT_TABS_A = np.array(_BT_TABS, np.int64)
# pooled:  GLOBAL DECISION | tok tables | kid tables | VARS | ACTIONS
_SEG_TABS_POOLED = _BT_TABS + (_VARS, _ACTIONS)
_SEG_TABS_POOLED_A = np.array(_SEG_TABS_POOLED, np.int64)
_SEG_FIRST_KID = 2 + len(_TOK_TABS)
# tokens:  GLOBAL DECISION | tok tables | kid tables | token VARS | (target-matched) VARS | ACTIONS
_SEG_TABS_TOKENS = _BT_TABS + (_VARS, _VARS, _ACTIONS)
_SEG_TABS_TOKENS_A = np.array(_SEG_TABS_TOKENS, np.int64)
_SEG_VT, _SEG_VG = 2 + len(_BT_TABS), 3 + len(_BT_TABS)
_TAB_ENT = {_SEATS: Ent.SEAT, _CARDS: Ent.CARD, _CREATURES: Ent.CREATURE, _RELICS: Ent.RELIC,
            _POTIONS: Ent.POTION, _ORBS: Ent.ORB, _ARROWS: Ent.ARROW, _NODES: Ent.MAP_NODE, _OPTIONS: Ent.OPTION,
            _CELLS: Ent.CELL, _POWERS: Ent.POWER, _INTENTS: Ent.INTENT, _REFS: Ent.REF, _VARS: Ent.VAR,
            _ACTIONS: Ent.ACTION, _FORECAST: Ent.FORECAST, _TURN: Ent.TURN, _FOE: Ent.FOE, _ROUTE: Ent.ROUTE,
            _AHEAD: Ent.AHEAD}
_ENT_OF_TAB = tuple((t, int(_TAB_ENT[t])) for t in (*_BT_TABS, _VARS, _ACTIONS))
# forecast (pooled only): GLOBAL DECISION | tok tables | POWERS INTENTS REFS FORECAST | VARS | ACTIONS
_BT_TABS_FC = _BT_TABS + (_FORECAST,)
_SEG_TABS_FC = _BT_TABS_FC + (_VARS, _ACTIONS)
_ENT_OF_TAB_FC = tuple((t, int(_TAB_ENT[t])) for t in (*_BT_TABS_FC, _VARS, _ACTIONS))
_BT_TABS_FC_A, _SEG_TABS_FC_A = np.array(_BT_TABS_FC, np.int64), np.array(_SEG_TABS_FC, np.int64)
# calc (pooled only): ... | POWERS INTENTS REFS FORECAST TURN FOE ROUTE | VARS | ACTIONS (FORECAST: 0 rows when off)
_BT_TABS_CC = _BT_TABS_FC + (_TURN, _FOE, _ROUTE)
_SEG_TABS_CC = _BT_TABS_CC + (_VARS, _ACTIONS)
_ENT_OF_TAB_CC = tuple((t, int(_TAB_ENT[t])) for t in (*_BT_TABS_CC, _VARS, _ACTIONS))
_BT_TABS_CC_A, _SEG_TABS_CC_A = np.array(_BT_TABS_CC, np.int64), np.array(_SEG_TABS_CC, np.int64)
# route summary (pooled only): ... | POWERS INTENTS REFS FORECAST TURN FOE ROUTE AHEAD | VARS | ACTIONS
_BT_TABS_AH = _BT_TABS_CC + (_AHEAD,)
_SEG_TABS_AH = _BT_TABS_AH + (_VARS, _ACTIONS)
_ENT_OF_TAB_AH = tuple((t, int(_TAB_ENT[t])) for t in (*_BT_TABS_AH, _VARS, _ACTIONS))
_BT_TABS_AH_A, _SEG_TABS_AH_A = np.array(_BT_TABS_AH, np.int64), np.array(_SEG_TABS_AH, np.int64)


def _seg_arrays(seg_tabs: tuple[int, ...], pooled: bool, n_tab: int = N_TABLES) -> tuple[NDArray[np.int8], I64]:
    ents = [int(Ent.GLOBAL), int(Ent.DECISION)] + [int(_TAB_ENT[t]) for t in seg_tabs]
    seg_of_tab = np.zeros(n_tab, np.int64)  # segment column holding each table (VARS: the pooled / token one)
    for j, t in reversed(list(enumerate(seg_tabs))):
        seg_of_tab[t] = 2 + j
    return np.array(ents, np.int8), seg_of_tab


_SEG_ENT_POOLED, _SEG_OF_TAB_POOLED = _seg_arrays(_SEG_TABS_POOLED, True)
_SEG_ENT_TOKENS, _SEG_OF_TAB_TOKENS = _seg_arrays(_SEG_TABS_TOKENS, False)
_SEG_ENT_FC, _SEG_OF_TAB_FC = _seg_arrays(_SEG_TABS_FC, True, N_TABLES_FORECAST)
_SEG_ENT_CC, _SEG_OF_TAB_CC = _seg_arrays(_SEG_TABS_CC, True, _N_TAB_ALL)
_SEG_ENT_AH, _SEG_OF_TAB_AH = _seg_arrays(_SEG_TABS_AH, True, _N_TAB_AH)

# table columns read by the derived slots / relations of each entity type (besides its slot columns)
_EXTRA: Mapping[Ent, tuple[int, ...]] = {
    Ent.SEAT: (_S_CROW,),
    Ent.CARD: (_K_ZONE, _K_ID, _K_UP, _K_LINK, _K_AUX),
    Ent.CREATURE: (_CR_KIND, _CR_MODEL, _CR_HP, _CR_POISON),
    Ent.RELIC: (_RL_ID,),
    Ent.POTION: (_PT_ID,),
    Ent.MAP_NODE: (_MN_ROW,),
    Ent.OPTION: (_O_KIND, _O_KEY, _O_REF_T, _O_REF_R),
    Ent.POWER: (_P_ID, _P_OWNER, _P_APPLIER),
    Ent.INTENT: (_I_OWNER,),
    Ent.FORECAST: (_FC_OWNER,),
    Ent.TURN: (_TU_OWNER,),
    Ent.FOE: (_FO_OWNER,),
    Ent.ROUTE: (_RO_OWNER,),
    Ent.REF: (_RF_CAT, _RF_VOC, _RF_OT, _RF_OR, _RF_ET, _RF_ER),
    Ent.VAR: (_V_OWNER_T, _V_OWNER_R),
    Ent.ACTION: (_A_SRC_T, _A_SRC_R, _A_TGT, _A_SEAT),
}


def _frame_bytes(f: Frame) -> bytes | bytearray | memoryview:
    """The frame's payload with every table inside it (resolved elided tables are copied in; a FORECAST
    trailer and the calc trailers are re-appended after the rebuilt v1 data)."""
    if f.msg_type != _MSG_OBS and f.msg_type != _MSG_TERMINAL:
        raise ValueError("only OBS / TERMINAL frames can be tokenized")
    if not f.flags & _F_ELIDED:  # every table lives in the payload (unless a caller removed one)
        if f._tables is not None and any(a is None for a in f._tables):
            raise ValueError("frame has elided tables; call resolve_elided(prev) first")
        p = f.payload
        return p.cast("B") if type(p) is memoryview else p
    tables = f.tables
    if any(a is None for a in tables):
        raise ValueError("frame has elided tables; call resolve_elided(prev) first")
    head = f.words[:_DATA_W0].copy()
    d = head[_DIR_W0:].reshape(N_TABLES, DIR_STRIDE)
    off = DATA_OFFSET
    parts: list[I32] = [head]
    for t, a in enumerate(cast(list[I32], tables)):
        d[t, 1], d[t, 3] = a.shape[0], off
        off += a.size * WORD
        parts.append(a.ravel())
    fc = f.forecast
    if fc is not None:
        parts.append(np.array([_FORECAST, fc.shape[0], FORECAST_N_COLS, off + DIR_STRIDE * WORD, f.fc_hash], np.int32))
        parts.append(fc.ravel())
        off += DIR_STRIDE * WORD + fc.size * WORD
    if f.cc_rows is not None:
        assert f.cc_hash is not None
        for k, tid in enumerate(CALC_TABLES):
            a = cast(I32, f.calc_table(k))
            parts.append(np.array([tid, a.shape[0], CALC_N_COLS[k], off + DIR_STRIDE * WORD, f.cc_hash[k]], np.int32))
            parts.append(a.ravel())
            off += DIR_STRIDE * WORD + a.size * WORD
    hd = f.hdraw
    if hd is not None:  # the HDRAW trailer closes the chain (never read by the tokenizer: notes/critic/contract.md 1.2)
        parts.append(np.array([HDRAW_TABLE, hd.shape[0], HDRAW_N_COLS, off + DIR_STRIDE * WORD, f.hd_hash], np.int32))
        parts.append(hd.ravel())
        off += DIR_STRIDE * WORD + hd.size * WORD
    return np.concatenate(parts).astype("<i4").tobytes()


_NATIVE_VERSION = 6


def _native_params(tok: "Tokenizer") -> tuple[I64, I64, I64, I64]:
    """``(params, prog, progn, poly)`` for ``spire._native.encode_*`` (layout: NP_* in spire_native.c)."""
    L = tok.layout
    d = L.derived_slot
    st = {cat: off for cat, (off, _, _) in L.static.items()}
    vals: list[int] = [
        0, _NATIVE_VERSION, tok.width, N_ENT, L.static["card"][2],
        st["card"], st["monster"], st["relic"], st["potion"], st["power"],
        d[Ent.GLOBAL]["kind_sub"], d[Ent.DECISION]["prompt_source"], d[Ent.SEAT]["acts"],
        d[Ent.CARD]["static"], d[Ent.CARD]["pile"], d[Ent.CARD]["group"],
        d[Ent.CREATURE]["model"], d[Ent.CREATURE]["static"], d[Ent.CREATURE]["poison_lethal"],
        d[Ent.RELIC]["static"], d[Ent.POTION]["static"], d[Ent.MAP_NODE]["drow"], d[Ent.OPTION]["key"],
        d[Ent.POWER]["static"], d[Ent.POWER]["applier_kind"], d[Ent.REF]["vocab"],
        _ZONE_CHOICE, _ZONE_OFFER, _PLAYER,
        _G_KIND, _G_SUB, _G_PCAT, _G_PSID, _G_CUR_ROW, _G_ASC, _G_NSEATS,
        _S_CROW, _S_CHAR,
        _K_ZONE, _K_ID, _K_UP, _K_LINK, _K_AUX,
        _CR_KIND, _CR_MODEL, _CR_HP, _CR_POISON,
        _RL_ID, _PT_ID, _MN_ROW,
        _O_KIND, _O_KEY, _O_REF_T, _O_REF_R,
        _P_ID, _P_OWNER, _P_APPLIER,
        _I_OWNER,
        _RF_CAT, _RF_VOC, _RF_OT, _RF_OR, _RF_ET, _RF_ER,
        _V_OWNER_T, _V_OWNER_R, _V_TGT,
        _A_SEAT, _A_SRC_T, _A_SRC_R, _A_TGT,
        _E_FROM, _E_TO,
        _SEATS, _CARDS, _VARS, _CREATURES, _POWERS, _INTENTS, _RELICS, _POTIONS,
        _ORBS, _ARROWS, _NODES, _EDGES, _OPTIONS, _REFS, _CELLS, _ACTIONS,
        _DIR_W0, _GLOB_W0, DIR_STRIDE,
        int(Ent.GLOBAL), int(Ent.DECISION),
        len(L.creature_off), len(L.opt_key_off), len(L.ref_off),
        *N_COLS, FORECAST_N_COLS, *CALC_N_COLS, AHEAD_SLOTS,
        *_SEG_TABS_AH,
        *(int(_TAB_ENT[t]) for t in _SEG_TABS_AH),
        int(tok.config.mend_target), int(tok.config.coop_features),
        d[Ent.OPTION]["vote_count"] if tok.config.coop_features else -1,
        int(G.SEAT_SCREEN0), int(G.SEAT_STATUS0), int(SeatCol.SEAT), int(OptionCol.FLAGS),
        int(OptionKind.REST_OPTION), tok._mend_key, int(ActionCol.AUX),
        # version 3: FORECAST trailer (17th table; 0 rows / not read when off)
        int(tok.config.forecast), _FORECAST, FORECAST_N_COLS, int(Ent.FORECAST), _FC_OWNER, _H_NTAB, _DATA_W0,
        # version 4: calc trailers TURN / FOE / ROUTE (tables 17-19; 0 rows / not read when off)
        int(tok.config.calc), _TURN, _FOE, _ROUTE, _TU_OWNER, _FO_OWNER, _RO_OWNER, _CREATURES, _CREATURES, _NODES,
        # version 5 (notes/inputs2): the DROP rule, A1 and the AHEAD row (1 row of "table" _AHEAD when on)
        int(tok._drop), int(tok.config.route_summary), _G_IN_COMBAT, _KIND_MAP, _H_STEP, _MN_PT, _MN_FLAGS,
        _FL_BOSS, _FL_QUEST, _AHEAD, int(Ent.AHEAD), _RC_TO_BOSS, *_RC_PAIRS, *tok._pt_ids,
        # version 6 (notes/critic/contract.md 1.2-1.3): the frames carry the HDRAW trailer iff allow_hidden (checked
        # in the chain, never read); encode_fill writes card_base
        int(tok.allow_hidden), HDRAW_TABLE, HDRAW_N_COLS,
    ]
    vals[0] = len(vals)
    if _native is not None and (len(vals) != _native.N_PARAMS or _native.VERSION != _NATIVE_VERSION):
        raise RuntimeError("spire._native was built from another source version; run python -B native/build.py")
    width = tok.width
    prog = np.full((N_ENT, width), -1, np.int64)
    progn = np.zeros(N_ENT, np.int64)
    for e in Ent:
        cols = [int(c) for c in L.slot_cols[e]]
        prog[e, : len(cols)] = cols
        progn[e] = len(cols)
    poly = np.concatenate([L.creature_off, L.creature_rows, L.opt_key_off, L.opt_key_rows, L.ref_off, L.ref_rows])
    return np.array(vals, np.int64), prog.ravel(), progn, poly.astype(np.int64)


def _drop_frames(GL: I32) -> NDArray[np.bool_]:
    """DROP(frame) of ``map_tokens = "no_combat"``: GLOBAL[IN_COMBAT] = 1 and GLOBAL[KIND] != MAP (one per row of the
    (B, GLOBAL_LEN) globals ``GL``)."""
    return (GL[:, _G_IN_COMBAT] == 1) & (GL[:, _G_KIND] != _KIND_MAP)


def _rows_of(C: I64, A: I64, t: int, nc: int, frames: NDArray[np.bool_]) -> tuple[I64, I64, I64]:
    """(frame, row index, first word) of every row of table ``t`` in the selected frames."""
    n = np.where(frames, C[:, t], 0)
    f = np.repeat(np.arange(len(n)), n)
    loc = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    return f, loc, A[f, t] + loc * nc


def _assert_a1(W: I32, C: I64, A: I64, fb: I64, drop: NDArray[np.bool_]) -> None:
    """Assertion A1 (notes/inputs2/contract.md 1): in a DROP frame no ACTIONS row has SRC_TABLE = MAP_NODES, no
    OPTIONS row REF_TABLE = MAP_NODES and no REFS row ENT_TABLE = MAP_NODES (with 0 node rows such a reference would
    resolve to a row of the next segment)."""
    for t, c, name in ((_ACTIONS, _A_SRC_T, "ACTIONS row {} has SRC_TABLE"), (_OPTIONS, _O_REF_T, "OPTIONS row {} has "
                       "REF_TABLE"), (_REFS, _RF_ET, "REFS row {} has ENT_TABLE")):
        f, loc, w = _rows_of(C, A, t, N_COLS[t], drop)
        bad = np.flatnonzero(W[w + c] == _NODES)
        if bad.size:
            i = int(bad[0])
            raise AssertionError(f"A1: frame step {int(W[fb[f[i]] + _H_STEP])} (IN_COMBAT = 1, KIND != MAP: map tables "
                                 f"dropped): {name.format(int(loc[i]))} = MAP_NODES")


def _ahead_slots(W: I32, C: I64, A: I64, pt_ids: tuple[int, ...]) -> I32:
    """(B, AHEAD_SLOTS) AHEAD slots of every frame from its ROUTE rows and their MAP_NODES rows (notes/inputs2/
    contract.md 2.2, v2a.md 2): MINV(c) = min over the rows with c >= 0, else -1; MAXV(c) = -1 if there is no row or a
    row has c = -1, else the max."""
    B = C.shape[0]
    n = C[:, _ROUTE]
    out = np.zeros((B, AHEAD_SLOTS), np.int64)
    out[:, 8:29] = -1
    if not n.any():
        return out.astype(np.int32)
    f, _, w = _rows_of(C, A, _ROUTE, _ROUTE_NC, np.ones(B, np.bool_))
    node = A[f, _NODES] + W[w + _RO_OWNER].astype(np.int64) * N_COLS[_NODES]
    ty, fl = W[node + _MN_PT], W[node + _MN_FLAGS]

    def count(x: NDArray[np.bool_]) -> I64:
        return np.bincount(f[x], minlength=B).astype(np.int64)

    def minv(v: I32) -> I64:
        m = np.full(B, np.iinfo(np.int64).max, np.int64)
        ok = v >= 0
        np.minimum.at(m, f[ok], v[ok])
        return np.where(m == np.iinfo(np.int64).max, -1, m)

    def maxv(v: I32) -> I64:
        m = np.full(B, np.iinfo(np.int64).min, np.int64)
        np.maximum.at(m, f, v)
        return np.where((n == 0) | (count(v == -1) > 0), -1, m)

    out[:, 0] = n
    for k, tid in enumerate(pt_ids):
        out[:, 1 + k] = count(ty == tid)
    out[:, 7] = count((fl & _FL_BOSS) != 0)
    out[:, 8] = minv(W[w + _RC_TO_BOSS])
    for k, c in enumerate(_RC_PAIRS):
        v = W[w + c]
        out[:, 9 + 2 * k], out[:, 10 + 2 * k] = minv(v), maxv(v)
    out[:, 29] = count((fl & _FL_QUEST) != 0)
    return out.astype(np.int32)


class EncodedBatch:
    """Tokenized frames of one batch (:meth:`Tokenizer.encode_batch`).

    ``types`` / ``raw`` / ``rel`` hold the entity rows of all frames concatenated in frame order
    (frame ``i`` owns rows ``row0[i]:row0[i+1]``; ``rel`` is sample-relative, as in
    :class:`Sample`), ``edges`` the map edges (``edge0``), ``cond`` and ``counts`` one row per
    frame. ``order`` maps output frames to input indices when the batch was sorted (else None).
    ``words`` / ``act0`` (the concatenated frame words and the word of each frame's ACTIONS row 0)
    let callers read action columns without the frames (:meth:`action_column`). ``card_base`` (B,) int32: the local
    token index of CARDS row 0 of each frame (notes/critic/contract.md 2.1; the base the calc kid rows use).
    """

    __slots__ = ("types", "raw", "rel", "edges", "cond", "counts", "row0", "edge0", "order", "words", "act0", "history",
                 "card_base")

    def __init__(self, types: NDArray[np.int8], raw: NDArray[np.int16], rel: NDArray[np.int16],
                 edges: NDArray[np.int16], cond: NDArray[np.int16], counts: I64, row0: I64, edge0: I64,
                 order: I64 | None, words: I32, act0: I64, card_base: I32) -> None:
        self.types, self.raw, self.rel, self.edges, self.cond = types, raw, rel, edges, cond
        self.counts, self.row0, self.edge0, self.order, self.words, self.act0 = counts, row0, edge0, order, words, act0
        self.card_base = card_base
        self.history = np.zeros((len(counts), history_width(raw.shape[1])), np.int16)

    def __len__(self) -> int:
        return self.cond.shape[0]

    @property
    def n_tokens(self) -> I64:
        return self.counts[:, 0]

    def sample(self, i: int) -> Sample:
        a, b = int(self.row0[i]), int(self.row0[i + 1])
        e0, e1 = int(self.edge0[i]), int(self.edge0[i + 1])
        c = self.counts[i].tolist()
        return Sample(self.types[a:b], self.raw[a:b], self.rel[a:b], self.edges[e0:e1], self.cond[i],
                      (c[0], c[1], c[2], c[3]), self.history[i])

    def samples(self, copy: bool = False) -> list[Sample]:
        """Per-frame :class:`Sample` records: views into the batch arrays, or (``copy=True``) owned
        copies that do not keep the whole batch alive."""
        r, e, cl = self.row0.tolist(), self.edge0.tolist(), self.counts.tolist()
        ty, raw, rel, ed, cond = self.types, self.raw, self.rel, self.edges, self.cond
        if copy:
            return [Sample(ty[r[i]:r[i + 1]].copy(), raw[r[i]:r[i + 1]].copy(), rel[r[i]:r[i + 1]].copy(),
                           ed[e[i]:e[i + 1]].copy(), cond[i].copy(), (c[0], c[1], c[2], c[3]), self.history[i].copy())
                    for i, c in enumerate(cl)]
        return [Sample(ty[r[i]:r[i + 1]], raw[r[i]:r[i + 1]], rel[r[i]:r[i + 1]], ed[e[i]:e[i + 1]], cond[i],
                       (c[0], c[1], c[2], c[3]), self.history[i])
                for i, c in enumerate(cl)]

    def action_column(self, col: int, rows: NDArray[np.integer], lo: int = 0) -> I32:
        """ACTIONS column ``col`` of action row ``rows[i]`` of frame ``lo + i`` (0 where the row is out
        of the frame's ACTIONS range, e.g. a frame without legal actions)."""
        r = np.asarray(rows, np.int64)
        hi = lo + len(r)
        ok = (r >= 0) & (r < self.counts[lo:hi, 3])
        idx = self.act0[lo:hi] + np.where(ok, r, 0) * N_COLS[_ACTIONS] + col
        w = self.words
        return np.where(ok, w[np.minimum(idx, w.shape[0] - 1)], 0)


class Tokenizer:
    """Frame -> :class:`Sample` (:meth:`encode`) and list of samples -> :class:`Batch` (:meth:`collate`)."""

    def __init__(self, vocab: Vocab, config: TokenizerConfig = TokenizerConfig(), *, native: bool = True) -> None:
        """``native=False`` forces the numpy batch encoder even when ``spire._native`` is built."""
        self.vocab = vocab
        self.config = config
        assert not config.forecast or config.pool_children, "tokenizer.forecast needs pool_children"
        assert not config.calc or config.pool_children, "tokenizer.calc needs pool_children"
        self.layout = L = TokenLayout(vocab, public_history=config.public_history, cycle_ban=config.cycle_ban,
                                     coop_features=config.coop_features, mend_target=config.mend_target,
                                     forecast=config.forecast, calc=config.calc,
                                     route_summary=config.route_summary, second_boss_table=config.second_boss_table,
                                     rules_mode=config.rules_mode)
        # tables the batch encoder reads (with calc all 20: FORECAST has 0 rows when off; the AHEAD row is the 21st)
        self._n_tab = (_N_TAB_AH if config.route_summary else _N_TAB_ALL if config.calc else
                       N_TABLES_FORECAST if config.forecast else N_TABLES)
        self._drop = config.map_tokens == "no_combat"  # the DROP rule (notes/inputs2/contract.md 1)
        # vocab ids of the six room types of AHEAD slots 1-6 (unused, 0, without the route summary)
        self._pt_ids = (tuple(vocab.id("enum:MapPointType", n) for n in AHEAD_TYPES) if config.route_summary
                        else (0,) * len(AHEAD_TYPES))
        self._n_hdr = n_tables_for(config.forecast, config.calc)  # header N_TABLES of the frames (strict check)
        self._allow_hidden = False  # run-time only (notes/critic/contract.md 1.2): see allow_hidden
        self._mend_key = vocab["rest_option"].index["MEND"] if config.mend_target else -1
        self.width = L.width
        self._pooled = config.pool_children
        self._cols = [L.slot_cols[e] for e in Ent]
        d = L.derived_slot
        self._d_kind_sub, self._d_prompt = d[Ent.GLOBAL]["kind_sub"], d[Ent.DECISION]["prompt_source"]
        self._d_acts = d[Ent.SEAT]["acts"]
        self._d_card = (d[Ent.CARD]["static"], d[Ent.CARD]["pile"], d[Ent.CARD]["group"])
        self._d_cre = (d[Ent.CREATURE]["model"], d[Ent.CREATURE]["static"], d[Ent.CREATURE]["poison_lethal"])
        self._d_relic, self._d_potion = d[Ent.RELIC]["static"], d[Ent.POTION]["static"]
        self._d_drow, self._d_key = d[Ent.MAP_NODE]["drow"], d[Ent.OPTION]["key"]
        self._d_power = (d[Ent.POWER]["static"], d[Ent.POWER]["applier_kind"])
        self._d_ref = d[Ent.REF]["vocab"]
        self._st = {cat: off for cat, (off, _, _) in L.static.items()}
        self._card_levels = L.static["card"][2]
        self._ref_c = (int(RefCol.CATEGORY), int(RefCol.VOCAB_ID), int(RefCol.OWNER_TABLE), int(RefCol.OWNER_ROW),
                       int(RefCol.ENT_TABLE), int(RefCol.ENT_ROW))
        self._tok_ents = [int(e) for e in (Ent.SEAT, Ent.CARD, Ent.CREATURE, Ent.RELIC, Ent.POTION, Ent.ORB,
                                           Ent.ARROW, Ent.MAP_NODE, Ent.OPTION, Ent.CELL)]
        self._tok_tabs = [_SEATS, _CARDS, _CREATURES, _RELICS, _POTIONS, _ORBS, _ARROWS, _NODES, _OPTIONS, _CELLS]
        self._buckets = np.array(config.token_buckets, np.int64)
        # batch-encoder gather programs per entity type: the slot columns, one placeholder per derived
        # slot (overwritten), then the extra table columns the derived slots / relations read.
        self._prog: list[I64] = []
        self._k: list[int] = []  # raw slots written per row (slot columns + derived)
        self._extra: list[dict[int, int]] = []  # table column -> position in the gathered row
        for e in Ent:
            cols = [int(c) for c in L.slot_cols[e]]
            prog = cols + [0] * len(L.derived_slot[e])
            ex: dict[int, int] = {}
            for j, c in enumerate(cols):
                ex.setdefault(c, j)
            for c in _EXTRA.get(e, ()):
                if c not in ex:
                    ex[c] = len(prog)
                    prog.append(c)
            self._prog.append(np.array(prog, np.int64))
            self._k.append(len(cols) + len(L.derived_slot[e]))
            self._extra.append(ex)
        self.native = _native is not None and self._pooled and native
        self._native_collate = _native is not None and native and hasattr(_native, "collate_fill")
        self._nat: tuple[I64, I64, I64, I64] | None = _native_params(self) if self.native else None

    @property
    def allow_hidden(self) -> bool:
        """The frames carry the HDRAW trailer (notes/critic/contract.md 1.2). A run-time attribute, not a
        TokenizerConfig key and never stored: only the training actor of a critic run sets it (spire.train). The
        strict table count becomes 16 + [forecast] + 3 [calc] + [allow_hidden]; the table itself is never read."""
        return self._allow_hidden

    @allow_hidden.setter
    def allow_hidden(self, on: bool) -> None:
        self._allow_hidden = bool(on)
        self._n_hdr = n_tables_for(self.config.forecast, self.config.calc, self._allow_hidden)
        if self._nat is not None:
            self._nat = _native_params(self)

    # ------------------------------------------------------------------ encode
    def encode(self, frame: Frame) -> Sample:
        """Encode one OBS or TERMINAL frame (elided tables must be resolved first).

        The batch path with B = 1 (:meth:`encode_batch`); bit-identical to the original
        per-frame implementation kept in :mod:`spire.tokenize_ref`."""
        return self.encode_batch((frame,)).sample(0)

    def encode_batch(self, frames: Sequence[Frame], *, sort: bool = False) -> "EncodedBatch":
        """Tokenize the OBS / TERMINAL ``frames`` of an inference batch in one vectorised pass.

        The tables of all frames are addressed in one concatenated int32 word array with per-frame
        offsets, so each gather / scatter / derived-column computation runs once per batch instead
        of once per frame. The result holds the concatenated :class:`Sample` arrays (row-major in
        frame order, exactly what :meth:`collate` would concatenate), from which
        :meth:`EncodedBatch.sample` makes per-frame ``Sample`` views and :meth:`collate_encoded`
        builds batches without re-concatenating.

        ``sort=True`` reorders the frames by padded token bucket (stable: input order inside a
        bucket); ``EncodedBatch.order[j]`` is the input index of output frame ``j``.
        """
        B = len(frames)
        if not B:
            raise ValueError("empty batch")
        if B > _MAX_ENCODE_BATCH:  # var-matching keys carry the frame index in their top bits
            raise ValueError(f"encode_batch takes at most {_MAX_ENCODE_BATCH} frames, got {B}")
        parts = [_frame_bytes(f) for f in frames]
        lens = np.fromiter(map(len, parts), np.int64, B)
        lens >>= 2
        W = np.frombuffer(b"".join(parts), "<i4")  # one copy of every frame, in order
        fb = np.cumsum(lens) - lens
        # strict: a frame carries the FORECAST trailer iff the tokenizer encodes it (contract 3.2)
        assert (W[fb + _H_NTAB] == self._n_hdr).all(),             f"frames with N_TABLES {np.unique(W[fb + _H_NTAB]).tolist()} for tokenizer.forecast={self.config.forecast} calc={self.config.calc} allow_hidden={self._allow_hidden}"
        if self._nat is not None:
            return self._encode_native(W, fb, lens, sort)
        if not sort:
            return self._encode(W, fb, None)
        if self._pooled:
            D = W[fb[:, None] + _DIR_CNT_IDX]  # (B, 16) row counts
            n_tok = 2 + D[:, _TOK_TABS_A].sum(1, dtype=np.int64)
            if self._drop:  # a DROP frame has no MAP_NODE tokens
                n_tok -= np.where(_drop_frames(W[fb[:, None] + _GLOB_IDX]), D[:, _NODES], 0)
            order = self._bucket_order(n_tok)
            return self._encode(W, fb[order], order) if order is not None else self._encode(W, fb, None)
        enc = self._encode(W, fb, None)
        order = self._bucket_order(enc.counts[:, 0])
        return self._encode(W, fb[order], order) if order is not None else enc

    def _encode_native(self, W: I32, fb: I64, lens: I64, sort: bool) -> "EncodedBatch":
        """Pooled layout via ``spire._native`` (one C loop over the frames; same output)."""
        assert self._nat is not None and _native is not None
        params, prog, progn, poly = self._nat
        B = fb.shape[0]
        counts = np.empty((B, 4), np.int64)
        n_rows = np.empty(B, np.int64)
        n_edges = np.empty(B, np.int64)
        _native.encode_plan(W, fb, lens, params, counts, n_rows, n_edges)
        order = self._bucket_order(counts[:, 0]) if sort else None
        if order is not None:
            fb, lens, counts, n_rows, n_edges = fb[order], lens[order], counts[order], n_rows[order], n_edges[order]
        row0 = np.zeros(B + 1, np.int64)
        np.cumsum(n_rows, out=row0[1:])
        edge0 = np.zeros(B + 1, np.int64)
        np.cumsum(n_edges, out=edge0[1:])
        M, E = int(row0[-1]), int(edge0[-1])
        types = np.empty(M, np.int8)
        raw = np.zeros((M, self.width), np.int16)
        rel = np.empty((M, 3), np.int16)
        edges = np.empty((E, 3), np.int16)
        cond = np.zeros((B, 6), np.int16)
        act0 = np.empty(B, np.int64)
        card_base = np.empty(B, np.int32)
        _native.encode_fill(W, fb, lens, params, prog, progn, poly, row0, edge0, types, raw, rel, edges, cond, act0,
                            card_base)
        return EncodedBatch(types, raw, rel, edges, cond, counts, row0, edge0, order, W, act0, card_base)

    def _bucket_order(self, n_tok: I64) -> I64 | None:
        """Stable permutation sorting frames by padded token bucket (None if already sorted)."""
        b = self.bucket_of(n_tok)
        if (b[1:] >= b[:-1]).all():
            return None
        return np.argsort(b, kind="stable")

    def bucket_of(self, n_tokens: NDArray[np.integer]) -> I64:
        """Vectorised :meth:`TokenizerConfig.bucket`."""
        bk = self._buckets
        n = np.asarray(n_tokens, np.int64)
        i = np.searchsorted(bk, n, side="left")
        return np.where(i < len(bk), bk[np.minimum(i, len(bk) - 1)], -(-n // 64) * 64)

    def _encode(self, W: I32, fb: I64, order: I64 | None) -> "EncodedBatch":
        """Encode the frames starting at word offsets ``fb`` of the concatenated words ``W``."""
        B = fb.shape[0]
        L, pooled = self.layout, self._pooled
        NT = self._n_tab
        ar_b = np.arange(B)
        W0 = W  # the frames' words (EncodedBatch.words); W gains the AHEAD rows below with the route summary
        # ---- per-frame directory and global vector
        D = W[fb[:, None] + _DIR_IDX].reshape(B, N_TABLES, DIR_STRIDE)
        C = D[:, :, 1].astype(np.int64)  # rows per table
        A = fb[:, None] + (D[:, :, 3] >> 2)  # absolute word of each table's row 0
        ncols = _NCOLS_A
        if self.config.calc:  # the trailer chain at X0: [FORECAST] TURN FOE ROUTE become tables 16-19
            x = fb + _DATA_W0 + (C * _NCOLS_A).sum(1)
            CX = np.zeros((B, 4), np.int64)
            AX = np.repeat(fb[:, None], 4, 1)
            want = ((_FORECAST,) if self.config.forecast else ()) + CALC_TABLES
            for tid in want:
                DX = W[x[:, None] + np.arange(DIR_STRIDE)]
                k = tid - _FORECAST
                assert (DX[:, 0] == tid).all() and (DX[:, 2] == _NCOLS_A20[tid]).all()
                CX[:, k] = DX[:, 1]
                AX[:, k] = fb + (DX[:, 3] >> 2)
                x = AX[:, k] + CX[:, k] * _NCOLS_A20[tid]
            C = np.concatenate([C, CX], 1)
            A = np.concatenate([A, AX], 1)
            ncols = _NCOLS_A20
        elif self.config.forecast:  # the trailer entry at X0 (end of the v1 data) becomes table 16
            x0 = fb + _DATA_W0 + (C * _NCOLS_A).sum(1)
            DF = W[x0[:, None] + np.arange(DIR_STRIDE)]
            assert (DF[:, 0] == _FORECAST).all() and (DF[:, 2] == FORECAST_N_COLS).all()
            C = np.concatenate([C, DF[:, 1:2].astype(np.int64)], 1)
            A = np.concatenate([A, fb[:, None] + (DF[:, 3:4] >> 2)], 1)
            ncols = _NCOLS_A17
        GL = W[fb[:, None] + _GLOB_IDX]  # (B, GLOBAL_LEN) int32
        if self.config.route_summary:  # AHEAD from the ROUTE rows before the DROP rule (notes/inputs2/contract.md 2.1)
            combat = GL[:, _G_IN_COMBAT] == 1
            assert (C[combat, _NODES] > 0).all(), \
                "tokenizer.route_summary: a frame with IN_COMBAT = 1 has no MAP_NODES rows (hosts need RESET bit 1)"
            AH = _ahead_slots(W, C, A, self._pt_ids)
        if self._drop:  # the DROP rule: the three map tables of a DROP frame have 0 rows (after A1)
            drop = _drop_frames(GL)
            if drop.any():
                _assert_a1(W, C, A, fb, drop)
                C[drop, _NODES] = 0
                C[drop, _EDGES] = 0
                if self.config.calc:
                    C[drop, _ROUTE] = 0
        if self.config.route_summary:  # the AHEAD rows become table _AHEAD: one row per frame after the frames' words
            C = np.concatenate([C, np.ones((B, 1), np.int64)], 1)
            A = np.concatenate([A, (W.shape[0] + AHEAD_SLOTS * ar_b)[:, None]], 1)
            W = np.concatenate([W, AH.ravel()])
            ncols = _NCOLS_A21

        # ---- every table row of every frame, table-major then frame (one pass for all tables)
        L_run = C.T.ravel()
        tab_tot = C.sum(0)
        tab_start = (np.cumsum(tab_tot) - tab_tot).tolist()
        tab_n = tab_tot.tolist()
        run0 = np.cumsum(L_run) - L_run
        n_all = int(tab_tot.sum())
        pos = np.arange(n_all) - np.repeat(run0, L_run)  # row index within its frame's table
        fr = np.repeat(np.tile(ar_b, NT), L_run)  # frame of each row
        ws = np.repeat(A.T.ravel(), L_run) + pos * np.repeat(np.repeat(ncols, B), L_run)  # row's first word

        def sl(t: int) -> slice:
            return slice(tab_start[t], tab_start[t] + tab_n[t])

        def col(t: int, c: int) -> I32:
            return W[ws[sl(t)] + c]

        # ---- per-target VARS -> ACTIONS matching (schema 13)
        nv, na = tab_n[_VARS], tab_n[_ACTIONS]
        sv, sa = sl(_VARS), sl(_ACTIONS)
        v_ws, v_fr = ws[sv], fr[sv]
        var_act = np.full(nv, -1, np.int64)
        if nv and na:
            vt = W[v_ws + _V_TGT]
            targeted = vt >= 0
            if targeted.any():
                a_ws, a_fr = ws[sa], fr[sa]
                a_st, a_tg = W[a_ws + _A_SRC_T], W[a_ws + _A_TGT]
                key_base_a = a_fr * _FRAME_KEY
                a_key = key_base_a + np.where((a_st >= 0) & (a_tg >= 0),
                                              (a_st.astype(np.int64) * 4096 + W[a_ws + _A_SRC_R]) * 64 + a_tg, -1)
                v_key = v_fr * _FRAME_KEY + (W[v_ws + _V_OWNER_T].astype(np.int64) * 4096 + W[v_ws + _V_OWNER_R]) * 64 + vt
                o = np.argsort(a_key, kind="stable")
                sk = a_key[o]
                p = np.minimum(np.searchsorted(sk, v_key), na - 1)
                hit = targeted & (sk[p] == v_key)
                a0 = np.cumsum(C[:, _ACTIONS]) - C[:, _ACTIONS]  # first action of each frame (flat)
                gi = o[p[hit]]
                var_act[hit] = gi - a0[a_fr[gi]]

        # ---- segment layout per frame
        bt_tabs, ent_of_tab = _BT_TABS_A, _ENT_OF_TAB
        if pooled:
            seg_tabs = _SEG_TABS_POOLED_A
            seg_ent, seg_of_tab = _SEG_ENT_POOLED, _SEG_OF_TAB_POOLED
            if self.config.route_summary:
                seg_tabs, bt_tabs, ent_of_tab = _SEG_TABS_AH_A, _BT_TABS_AH_A, _ENT_OF_TAB_AH
                seg_ent, seg_of_tab = _SEG_ENT_AH, _SEG_OF_TAB_AH
            elif self.config.calc:
                seg_tabs, bt_tabs, ent_of_tab = _SEG_TABS_CC_A, _BT_TABS_CC_A, _ENT_OF_TAB_CC
                seg_ent, seg_of_tab = _SEG_ENT_CC, _SEG_OF_TAB_CC
            elif self.config.forecast:
                seg_tabs, bt_tabs, ent_of_tab = _SEG_TABS_FC_A, _BT_TABS_FC_A, _ENT_OF_TAB_FC
                seg_ent, seg_of_tab = _SEG_ENT_FC, _SEG_OF_TAB_FC
            SC = np.empty((B, len(seg_tabs) + 2), np.int64)
            SC[:, :2] = 1
            SC[:, 2:] = C[:, seg_tabs]
            tok_var = None
            n_tokvar = None
        else:
            tok_var = var_act < 0
            n_tokvar = np.bincount(v_fr[tok_var], minlength=B).astype(np.int64)
            SC = np.empty((B, len(_SEG_TABS_TOKENS) + 2), np.int64)
            SC[:, :2] = 1
            SC[:, 2:] = C[:, _SEG_TABS_TOKENS_A]
            SC[:, _SEG_VT] = n_tokvar
            SC[:, _SEG_VG] = C[:, _VARS] - n_tokvar
            seg_ent, seg_of_tab = _SEG_ENT_TOKENS, _SEG_OF_TAB_TOKENS
        SS = np.cumsum(SC, 1) - SC  # sample-relative first row of each segment
        n_rows = SS[:, -1] + SC[:, -1]
        row0 = np.cumsum(n_rows) - n_rows
        M = int(n_rows.sum())
        bt = np.full((B, NT), -1, np.int64)  # sample-relative row of each table's row 0 (-1: no entity)
        bt[:, bt_tabs] = SS[:, 2 : 2 + len(bt_tabs)]
        if not pooled:
            bt[:, _VARS] = SS[:, _SEG_VT]
        dest = np.repeat((row0[:, None] + SS[:, seg_of_tab]).T.ravel(), L_run) + pos  # batch row of each table row

        types = np.repeat(np.tile(seg_ent, B), SC.ravel())
        raw = np.zeros((M, self.width), np.int16)
        rel = np.full((M, 3), -1, np.int16)
        bt_flat = bt.ravel()
        bt_cre, bt_seat = bt[:, _CREATURES], bt[:, _SEATS]

        def ref(f: I64, table: I32, row: I32) -> I64:
            b = bt_flat[f * NT + np.clip(table, 0, N_TABLES - 1)]  # references name v1 tables only
            return np.where((table >= 0) & (b >= 0), b + row, -1)

        def put(r: I64, Y: I32, k: int) -> None:
            """Saturate the first ``k`` slot columns of ``Y`` to int16 and store them at rows ``r``."""
            y = Y[:, :k]
            np.clip(y, _I16_MIN, _I16_MAX, out=y)
            raw[r, :k] = y

        # ---- GLOBAL / DECISION rows
        prog = self._prog
        g0 = row0
        Y = GL[:, prog[int(Ent.GLOBAL)]]
        Y[:, self._d_kind_sub] = GL[:, _G_KIND] * 16 + np.clip(GL[:, _G_SUB], 0, 15)
        put(g0, Y, self._k[int(Ent.GLOBAL)])
        Y = GL[:, prog[int(Ent.DECISION)]]
        cat, sid = GL[:, _G_PCAT], GL[:, _G_PSID]
        cc = np.clip(cat, 0, len(L.ref_off) - 1)
        ok = (cat > 0) & (cat < len(L.ref_off)) & (L.ref_rows[cc] > 0) & (sid > 0)
        Y[:, self._d_prompt] = np.where(ok, L.ref_off[cc] + sid, 0)
        put(g0 + 1, Y, self._k[int(Ent.DECISION)])

        # ---- token / child tables: raw slots + derived slots + relations
        lv = self._card_levels
        st = self._st
        for t, e in ent_of_tab:
            n = tab_n[t]
            if not n:
                continue
            s = sl(t)
            Y = W[ws[s][:, None] + prog[e]]  # slot columns, derived placeholders, extra columns
            ex = self._extra[e]
            f = fr[s]
            r = dest[s]
            if t == _SEATS:
                if na:  # "acts": the seat has a legal action (old: bincount(seat)[:n_seats] > 0)
                    a_seat = W[ws[sa] + _A_SEAT].astype(np.int64)
                    a_fr = fr[sa]
                    ok_s = (a_seat >= 0) & (a_seat < C[a_fr, _SEATS])
                    seat0 = np.cumsum(C[:, _SEATS]) - C[:, _SEATS]
                    hit_s = np.zeros(n, np.bool_)
                    hit_s[seat0[a_fr[ok_s]] + a_seat[ok_s]] = True
                    Y[:, self._d_acts] = hit_s
                else:
                    Y[:, self._d_acts] = 0
                crow = Y[:, ex[_S_CROW]]
                rel[r, 1] = np.where(crow >= 0, bt_cre[f] + crow, -1)
                if self.config.coop_features:
                    seat_ids = Y[:, ex[int(SeatCol.SEAT)]]
                    Y[:, self._d_acts] |= ((GL[f, int(G.SEAT_SCREEN0) + seat_ids] << 1) |
                                           (GL[f, int(G.SEAT_STATUS0) + seat_ids] << 5))
            elif t == _CARDS:
                ds, dp, dg = self._d_card
                zone, aux = Y[:, ex[_K_ZONE]], Y[:, ex[_K_AUX]]
                Y[:, ds] = st["card"] + Y[:, ex[_K_ID]] * lv + np.minimum(Y[:, ex[_K_UP]], lv - 1)
                Y[:, dp] = np.where(zone == _ZONE_CHOICE, aux, 0)
                Y[:, dg] = np.where(zone == _ZONE_OFFER, aux, -1)
                lr = Y[:, ex[_K_LINK]]
                rel[r, 1] = np.where(lr >= 0, bt[f, _CARDS] + lr, -1)
            elif t == _CREATURES:
                dm, dms, dl = self._d_cre
                kind, mid = Y[:, ex[_CR_KIND]], Y[:, ex[_CR_MODEL]]
                hp, poison = Y[:, ex[_CR_HP]], Y[:, ex[_CR_POISON]]
                lethal = (poison > 0) & (poison >= hp) & (hp > 0)
                Y[:, dms] = np.where(kind == _PLAYER, 0, st["monster"] + mid)
                Y[:, dm] = _resolve(L.creature_off, L.creature_rows, kind, mid)
                Y[:, dl] = lethal
            elif t == _RELICS:
                Y[:, self._d_relic] = st["relic"] + Y[:, ex[_RL_ID]]
            elif t == _POTIONS:
                Y[:, self._d_potion] = st["potion"] + Y[:, ex[_PT_ID]]
            elif t == _NODES:
                Y[:, self._d_drow] = Y[:, ex[_MN_ROW]] - GL[f, _G_CUR_ROW]
            elif t == _OPTIONS:
                Y[:, self._d_key] = _resolve(L.opt_key_off, L.opt_key_rows, Y[:, ex[_O_KIND]], Y[:, ex[_O_KEY]])
                rel[r, 1] = ref(f, Y[:, ex[_O_REF_T]], Y[:, ex[_O_REF_R]])
                if self.config.coop_features:
                    votes = (Y[:, ex[int(OptionCol.FLAGS)]] >> 7) & 15
                    Y[:, L.derived_slot[Ent.OPTION]["vote_count"]] = (
                        (votes & 1) + ((votes >> 1) & 1) + ((votes >> 2) & 1) + ((votes >> 3) & 1))
            elif t == _POWERS:
                dps, dak = self._d_power
                app = Y[:, ex[_P_APPLIER]]
                has = app >= 0
                ncre = C[f, _CREATURES]
                cre0 = (np.cumsum(C[:, _CREATURES]) - C[:, _CREATURES])[f]
                ak = W[ws[sl(_CREATURES)][np.clip(cre0 + np.clip(app, 0, ncre - 1), 0, max(tab_n[_CREATURES] - 1, 0))]
                       + _CR_KIND] if tab_n[_CREATURES] else np.zeros(n, np.int32)
                Y[:, dps] = st["power"] + Y[:, ex[_P_ID]]
                Y[:, dak] = np.where(has, ak, 0)
                bc = bt_cre[f]
                rel[r, 0] = bc + Y[:, ex[_P_OWNER]]
                rel[r, 1] = np.where(has, bc + app, -1)
            elif t == _INTENTS:
                rel[r, 0] = bt_cre[f] + Y[:, ex[_I_OWNER]]
            elif t == _FORECAST:
                rel[r, 0] = bt_cre[f] + Y[:, ex[_FC_OWNER]]
            elif t == _TURN:
                rel[r, 0] = bt_cre[f] + Y[:, ex[_TU_OWNER]]
            elif t == _FOE:
                rel[r, 0] = bt_cre[f] + Y[:, ex[_FO_OWNER]]
            elif t == _ROUTE:
                rel[r, 0] = bt[f, _NODES] + Y[:, ex[_RO_OWNER]]
            elif t == _AHEAD:
                rel[r, 0] = 0  # the sample's GLOBAL token
            elif t == _REFS:
                Y[:, self._d_ref] = _resolve(L.ref_off, L.ref_rows, Y[:, ex[_RF_CAT]], Y[:, ex[_RF_VOC]])
                rel[r, 0] = ref(f, Y[:, ex[_RF_OT]], Y[:, ex[_RF_OR]])
                rel[r, 1] = ref(f, Y[:, ex[_RF_ET]], Y[:, ex[_RF_ER]])
            elif t == _VARS:
                own = ref(f, Y[:, ex[_V_OWNER_T]], Y[:, ex[_V_OWNER_R]])
                act_base = SS[f, -1]
                dst = np.where(var_act >= 0, act_base + var_act, own)
                if tok_var is not None:
                    assert n_tokvar is not None
                    tv = tok_var.astype(np.int64)
                    v0 = np.cumsum(C[:, _VARS]) - C[:, _VARS]
                    t0 = np.cumsum(n_tokvar) - n_tokvar
                    e_tok = np.cumsum(tv) - tv - t0[f]  # earlier token vars of the same frame
                    loc = np.arange(n) - v0[f]
                    r = row0[f] + np.where(tok_var, SS[f, _SEG_VT] + e_tok, SS[f, _SEG_VG] + loc - e_tok)
                    dst = np.where(tok_var, own, dst)
                rel[r, 0] = dst
            elif t == _ACTIONS:
                rel[r, 0] = ref(f, Y[:, ex[_A_SRC_T]], Y[:, ex[_A_SRC_R]])
                tg = Y[:, ex[_A_TGT]]
                rel[r, 1] = np.where(tg >= 0, bt_cre[f] + tg, -1)
                seat = Y[:, ex[_A_SEAT]]
                rel[r, 2] = np.where(seat < C[f, _SEATS], bt_seat[f] + seat, -1)
                if self.config.mend_target:
                    candidates = np.flatnonzero(Y[:, ex[_A_SRC_T]] == _OPTIONS)
                    if candidates.size:
                        cf = f[candidates]
                        source = Y[candidates, ex[_A_SRC_R]]
                        assert ((source >= 0) & (source < C[cf, _OPTIONS])).all()
                        option_words = A[cf, _OPTIONS] + source * N_COLS[_OPTIONS]
                        is_mend = ((W[option_words + _O_KIND] == int(OptionKind.REST_OPTION)) &
                                   (W[option_words + _O_KEY] == self._mend_key))
                        mend = candidates[is_mend]
                        aux_slot = L.slot(Ent.ACTION, ActionCol.AUX)
                        target = Y[mend, aux_slot]
                        assert ((target >= 0) & (target < C[f[mend], _SEATS]) & (target != seat[mend])).all()
                        rel[r[mend], 1] = bt_seat[f[mend]] + target
            put(r, Y, self._k[e])

        # ---- map edges: (from, to, 0) for every edge, then (to, from, 1), per frame
        ne_f = C[:, _EDGES]
        edge0 = np.zeros(B + 1, np.int64)
        np.cumsum(2 * ne_f, out=edge0[1:])
        edges = np.zeros((int(edge0[-1]), 3), np.int16)
        ne = tab_n[_EDGES]
        if ne:
            s = sl(_EDGES)
            f = fr[s]
            bn = bt[f, _NODES]
            a_, b_ = W[ws[s] + _E_FROM] + bn, W[ws[s] + _E_TO] + bn
            out_i = edge0[f] + pos[s]
            in_i = out_i + ne_f[f]
            edges[out_i, 0], edges[out_i, 1] = a_, b_
            edges[in_i, 0], edges[in_i, 1], edges[in_i, 2] = b_, a_, 1

        # ---- conditioning vector: ascension, n_seats, character of seats 0..3
        cond = np.zeros((B, 6), np.int16)
        cond[:, 0], cond[:, 1] = GL[:, _G_ASC], GL[:, _G_NSEATS]
        if tab_n[_SEATS]:
            s = sl(_SEATS)
            p4 = pos[s] < 4
            cond[fr[s][p4], 2 + pos[s][p4]] = W[ws[s][p4] + _S_CHAR]

        counts = np.empty((B, 4), np.int64)
        if pooled:
            counts[:, 0] = SS[:, _SEG_FIRST_KID]
            counts[:, 1] = SS[:, 2 + len(bt_tabs)] - SS[:, _SEG_FIRST_KID]
        else:
            counts[:, 0] = SS[:, _SEG_VG]
            counts[:, 1] = 0
        counts[:, 2] = SC[:, -2]
        counts[:, 3] = SC[:, -1]
        r0 = np.empty(B + 1, np.int64)
        r0[:B], r0[B] = row0, M
        act0 = A[:, _ACTIONS]
        return EncodedBatch(types, raw, rel, edges, cond, counts, r0, edge0, order, W0, act0,
                            bt[:, _CARDS].astype(np.int32))

    # ------------------------------------------------------------------ collate
    def shape_of(self, samples: Sequence[Sample]) -> "BatchShape":
        """The natural (smallest) padded shape :meth:`collate` uses for ``samples``."""
        cfg = self.config
        c = np.array([s.counts for s in samples], np.int64)
        n_edge = sum(s.edges.shape[0] for s in samples)
        return BatchShape(len(samples), cfg.bucket(int(c[:, 0].max())), _align(int(c[:, 3].max()), cfg.action_align),
                          _align(int(c[:, 1].sum()), cfg.child_align), _align(int(c[:, 2].sum()), cfg.child_align),
                          _align(n_edge, cfg.child_align), int(c.sum()))

    def collate(self, samples: Sequence[Sample], *, shape: "BatchShape | None" = None,
                pin_memory: bool = False) -> Batch:
        """Batch ``samples`` into one :data:`Batch` (dict of CPU tensors; move with :func:`batch_to`).

        The ``M`` entity rows of all samples are sent unpadded (``types``, ``raw``, ``rel``) with
        their position ``dest`` in the padded layout ``[tokens B*T | kids Nk | vars Nv | actions
        B*A]`` of ``N`` rows, which the model materialises on the device. By default tokens are
        padded to the bucket of the longest sample, actions to a multiple of ``action_align`` and
        kid / var / edge counts to a multiple of ``child_align`` (:meth:`shape_of`); ``shape`` fixes
        all sizes instead (CUDA graphs) - extra batch rows are empty samples, extra entity rows go to
        the sink row ``N + 1``. Relation indices are rows of the padded layout, ``N`` = "none".

        Keys: ``types`` (M,) int64, ``raw`` (M, W) int16, ``rel`` (M, 3) int64, ``dest`` (M,) int64,
        ``edges`` (Ne, 3) int64 (owner token, other token, direction), ``tok_mask`` (B, T) bool,
        ``act_mask`` (B, A) bool, ``kid_mask`` (Nk,) bool, ``var_mask`` (Nv,) bool, ``cond`` (B, 6)
        int64.
        """
        nat = self.shape_of(samples) if samples else None
        if nat is None:
            raise ValueError("empty batch")
        if shape is None:
            shape = nat
        elif not shape.fits(nat):
            raise ValueError(f"batch {nat} does not fit {shape}")
        B, T, A, Nk, Nv = shape.batch, shape.tokens, shape.actions, shape.kids, shape.vars
        b = len(samples)
        counts = np.array([s.counts for s in samples], np.int64)
        nt, nk, nv, na = counts.T
        BT = B * T
        N = BT + Nk + Nv + B * A
        ar = np.arange(b, dtype=np.int64)
        gbase = np.stack([ar * T, BT + np.cumsum(nk) - nk, BT + Nk + np.cumsum(nv) - nv, BT + Nk + Nv + ar * A], 1)
        sizes = counts.ravel()
        total = nat.rows
        M = shape.rows
        dest = np.full(M, N + 1, np.int64)
        dest[:total] = np.repeat(gbase.ravel() - (np.cumsum(sizes) - sizes), sizes) + np.arange(total)
        dest_ext = np.append(dest[:total], N)
        n_rows = counts.sum(1)
        row0 = np.cumsum(n_rows) - n_rows

        raw = np.zeros((M, self.width), np.int16)
        raw[:total] = np.concatenate([s.raw for s in samples])
        types = np.zeros(M, np.int64)
        types[:total] = np.concatenate([s.types for s in samples])
        rel = np.full((M, 3), N, np.int64)
        r = np.concatenate([s.rel for s in samples]).astype(np.int64)
        rel[:total] = dest_ext[np.where(r >= 0, r + np.repeat(row0, n_rows)[:, None], total)]

        n_edge = np.array([s.edges.shape[0] for s in samples], np.int64)
        se = int(n_edge.sum())
        edges = np.zeros((shape.edges, 3), np.int64)
        edges[:, :2] = N
        if se:
            e_cat = np.concatenate([s.edges for s in samples]).astype(np.int64)
            off = np.repeat(row0, n_edge)
            edges[:se, 0] = dest[e_cat[:, 0] + off]
            edges[:se, 1] = dest[e_cat[:, 1] + off]
            edges[:se, 2] = e_cat[:, 2]
        cond = np.zeros((B, 6), np.int64)
        cond[:b] = np.stack([s.cond for s in samples])
        history = np.zeros((B, history_width(self.width)), np.int16)
        history[:b] = np.stack([s.history for s in samples])
        tok_mask = np.zeros((B, T), np.bool_)
        tok_mask[:b] = np.arange(T)[None, :] < nt[:, None]
        tok_mask[b:, 0] = True  # empty filler samples attend to their (zero) global token: no NaN rows
        act_mask = np.zeros((B, A), np.bool_)
        act_mask[:b] = np.arange(A)[None, :] < na[:, None]
        out = {
            "types": torch.from_numpy(types), "raw": torch.from_numpy(raw), "rel": torch.from_numpy(rel),
            "dest": torch.from_numpy(dest), "edges": torch.from_numpy(edges),
            "tok_mask": torch.from_numpy(tok_mask), "act_mask": torch.from_numpy(act_mask),
            "kid_mask": torch.from_numpy(np.arange(Nk) < int(nk.sum())),
            "var_mask": torch.from_numpy(np.arange(Nv) < int(nv.sum())),
            "cond": torch.from_numpy(cond),
            "history": torch.from_numpy(history),
        }
        if pin_memory:
            out = {k: v.pin_memory() for k, v in out.items()}
        return out

    def shape_encoded(self, enc: "EncodedBatch", lo: int = 0, hi: int | None = None) -> "BatchShape":
        """:meth:`shape_of` for frames ``lo:hi`` of an :class:`EncodedBatch`."""
        hi = len(enc) if hi is None else hi
        cfg = self.config
        c = enc.counts[lo:hi]
        t_max, _, _, a_max = c.max(0).tolist()
        _, k_sum, v_sum, _ = c.sum(0).tolist()
        ca = cfg.child_align
        return BatchShape(hi - lo, cfg.bucket(t_max), _align(a_max, cfg.action_align), _align(k_sum, ca),
                          _align(v_sum, ca), _align(int(enc.edge0[hi] - enc.edge0[lo]), ca),
                          int(enc.row0[hi] - enc.row0[lo]))

    def collate_encoded(self, enc: "EncodedBatch", lo: int = 0, hi: int | None = None, *,
                        shape: "BatchShape | None" = None) -> Batch:
        """:meth:`collate` of the samples of frames ``lo:hi`` of ``enc``, without per-sample work.

        Bit-identical to ``collate(enc.samples()[lo:hi], shape=shape)``: the frames' rows are
        already contiguous in ``enc``. :meth:`collate_encoded_into` fills caller-owned buffers.
        """
        hi = len(enc) if hi is None else hi
        if hi <= lo:
            raise ValueError("empty batch")
        nat = self.shape_encoded(enc, lo, hi)
        if shape is None:
            shape = nat
        B, T, A, M = shape.batch, shape.tokens, shape.actions, shape.rows
        o: dict[str, NDArray[Any]] = {"types": np.empty(M, np.int64), "raw": np.empty((M, self.width), np.int16),
             "rel": np.empty((M, 3), np.int64), "dest": np.empty(M, np.int64),
             "edges": np.empty((shape.edges, 3), np.int64), "tok_mask": np.empty((B, T), np.bool_),
             "act_mask": np.empty((B, A), np.bool_), "kid_mask": np.empty(shape.kids, np.bool_),
             "var_mask": np.empty(shape.vars, np.bool_), "cond": np.empty((B, 6), np.int64),
             "history": np.empty((B, history_width(self.width)), np.int16)}
        self.collate_encoded_into(enc, lo, hi, shape, o, nat=nat)
        return {k: torch.from_numpy(v) for k, v in o.items()}

    def collate_encoded_into(self, enc: "EncodedBatch", lo: int, hi: int, shape: "BatchShape",
                             o: Mapping[str, NDArray[Any]], *, nat: "BatchShape | None" = None) -> None:
        """Collate frames ``lo:hi`` of ``enc`` with ``shape`` into the arrays ``o`` (every key of a
        :data:`Batch`, exact collated shapes and dtypes; e.g. numpy views of pinned staging
        memory). Every element is written, so recycled buffers need no clearing."""
        if hi <= lo:
            raise ValueError("empty batch")
        if nat is None:
            nat = self.shape_encoded(enc, lo, hi)
        if not shape.fits(nat):
            raise ValueError(f"batch {nat} does not fit {shape}")
        B, T, A, Nk, Nv = shape.batch, shape.tokens, shape.actions, shape.kids, shape.vars
        o["history"][:hi - lo] = enc.history[lo:hi]
        o["history"][hi - lo:] = 0
        if self._native_collate:
            assert _native is not None
            _native.collate_fill(enc.types, enc.raw, enc.rel, enc.edges, enc.cond, enc.counts, enc.row0, enc.edge0,
                                 lo, hi, T, A, Nk, Nv, o["types"], o["raw"], o["rel"], o["dest"], o["edges"],
                                 o["tok_mask"], o["act_mask"], o["kid_mask"], o["var_mask"], o["cond"])
            return
        b = hi - lo
        counts = enc.counts[lo:hi]
        nt, nk, nv, na = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
        BT = B * T
        N = BT + Nk + Nv + B * A
        R0, R1 = int(enc.row0[lo]), int(enc.row0[hi])
        E0, E1 = int(enc.edge0[lo]), int(enc.edge0[hi])
        total, se = R1 - R0, E1 - E0
        n_rows = enc.row0[lo + 1 : hi + 1] - enc.row0[lo:hi]
        row0 = enc.row0[lo:hi] - R0
        dest, rel, edges = o["dest"], o["rel"], o["edges"]
        ar = np.arange(b, dtype=np.int64)
        snk, snv = int(nk.sum()), int(nv.sum())
        gbase = np.empty((b, 4), np.int64)
        gbase[:, 0] = ar * T
        np.cumsum(nk, out=gbase[:, 1])
        gbase[:, 1] += BT - nk
        np.cumsum(nv, out=gbase[:, 2])
        gbase[:, 2] += BT + Nk - nv
        gbase[:, 3] = ar * A + (BT + Nk + Nv)
        sizes = counts.ravel()
        starts = np.cumsum(sizes) - sizes
        d = dest[:total]
        np.add(np.repeat(gbase.ravel() - starts, sizes), np.arange(total), out=d)
        dest[total:] = N + 1
        dest_ext = np.empty(total + 1, np.int64)
        dest_ext[:total] = d
        dest_ext[total] = N
        o["raw"][:total] = enc.raw[R0:R1]
        o["raw"][total:] = 0
        o["types"][:total] = enc.types[R0:R1]
        o["types"][total:] = 0
        r = enc.rel[R0:R1].astype(np.int64)
        r += np.repeat(row0, n_rows)[:, None]
        np.copyto(r, total, where=enc.rel[R0:R1] < 0)
        np.take(dest_ext, r, out=rel[:total])
        rel[total:] = N
        edges[:, :2] = N
        edges[:, 2] = 0
        if se:
            e_cat = enc.edges[E0:E1].astype(np.int64)
            off = np.repeat(row0, np.diff(enc.edge0[lo : hi + 1]))
            edges[:se, 0] = dest[e_cat[:, 0] + off]
            edges[:se, 1] = dest[e_cat[:, 1] + off]
            edges[:se, 2] = e_cat[:, 2]
        cond = o["cond"]
        cond[:b] = enc.cond[lo:hi]
        cond[b:] = 0
        tok_mask, act_mask = o["tok_mask"], o["act_mask"]
        np.less(_arange(T)[None, :], nt[:, None], out=tok_mask[:b])
        tok_mask[b:] = False
        tok_mask[b:, 0] = True  # empty filler samples attend to their (zero) global token: no NaN rows
        np.less(_arange(A)[None, :], na[:, None], out=act_mask[:b])
        act_mask[b:] = False
        np.less(_arange(Nk), snk, out=o["kid_mask"])
        np.less(_arange(Nv), snv, out=o["var_mask"])

    def shape_block(self, blk: SampleBlock, idx: NDArray[np.integer]) -> "BatchShape":
        """:meth:`shape_of` of the samples ``idx`` of ``blk``."""
        cfg = self.config
        idx = np.asarray(idx, np.int64)
        c = blk.counts[idx]
        t_max, _, _, a_max = c.max(0).tolist()
        _, k_sum, v_sum, _ = c.sum(0).tolist()
        n_edge = int((blk.edge0[idx + 1] - blk.edge0[idx]).sum())
        ca = cfg.child_align
        return BatchShape(len(idx), cfg.bucket(t_max), _align(a_max, cfg.action_align), _align(k_sum, ca),
                          _align(v_sum, ca), _align(n_edge, ca), int(c.sum()))

    def collate_block_into(self, blk: SampleBlock, idx: NDArray[np.integer], shape: "BatchShape",
                           o: Mapping[str, NDArray[Any]]) -> None:
        """:meth:`collate` of ``[blk[i] for i in idx]`` with ``shape`` into the arrays ``o`` (every key of a
        :data:`Batch`, exact shapes and dtypes, e.g. views of pinned staging memory; every element is
        written). The C kernel copies straight from the block's concatenated arrays without the GIL; the
        fallback goes through :meth:`collate`. Both are bit-identical to :meth:`collate`."""
        idx = np.ascontiguousarray(idx, np.int64)
        o["history"][:len(idx)] = blk.history[idx]
        o["history"][len(idx):] = 0
        if _native is not None and hasattr(_native, "collate_idx") and self._native_collate:
            _native.collate_idx(blk.types, blk.raw, blk.rel, blk.edges, blk.cond, blk.counts, blk.row0, blk.edge0,
                                idx, shape.tokens, shape.actions, shape.kids, shape.vars, o["types"], o["raw"],
                                o["rel"], o["dest"], o["edges"], o["tok_mask"], o["act_mask"], o["kid_mask"],
                                o["var_mask"], o["cond"])
            return
        batch = self.collate([blk[int(i)] for i in idx], shape=shape)
        for k, v in batch.items():
            np.copyto(o[k], v.numpy())

    def bucket_batches(self, samples: Sequence[Sample], max_tokens: int,
                       max_batch: int | None = None) -> list[list[int]]:
        """Split sample indices into same-bucket sub-batches with ``B * T_pad <= max_tokens``.

        For learner minibatches (one optimizer step over all sub-batches) and actor inference.
        Order inside a bucket follows the input order.
        """
        by: dict[int, list[int]] = {}
        for i, s in enumerate(samples):
            by.setdefault(self.config.bucket(s.counts[0]), []).append(i)
        out: list[list[int]] = []
        for t in sorted(by):
            cap = max(1, max_tokens // t)
            if max_batch is not None:
                cap = min(cap, max_batch)
            idx = by[t]
            out.extend(idx[j : j + cap] for j in range(0, len(idx), cap))
        return out


@dataclass(frozen=True)
class BatchShape:
    """Padded sizes of a batch: samples, tokens and actions per sample, kid / var / edge / entity rows."""

    batch: int
    tokens: int
    actions: int
    kids: int
    vars: int
    edges: int
    rows: int

    def fits(self, other: "BatchShape") -> bool:
        return (self.batch >= other.batch and self.tokens >= other.tokens and self.actions >= other.actions
                and self.kids >= other.kids and self.vars >= other.vars and self.edges >= other.edges
                and self.rows >= other.rows)

    def astuple(self) -> tuple[int, int, int, int, int, int, int]:
        return (self.batch, self.tokens, self.actions, self.kids, self.vars, self.edges, self.rows)


_ARANGE: dict[int, I64] = {}


def _arange(n: int) -> I64:
    """Cached ``np.arange(n)`` (read-only)."""
    a = _ARANGE.get(n)
    if a is None:
        a = np.arange(n, dtype=np.int64)
        a.flags.writeable = False
        if len(_ARANGE) < 512:
            _ARANGE[n] = a
    return a


def _align(n: int, a: int) -> int:
    return max(a, -(-n // a) * a)


def batch_to(batch: Batch, device: torch.device | str, non_blocking: bool = True) -> Batch:
    """Move every tensor of a batch to ``device``."""
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


# =========================================================================== device-side features


def symlog(x: Tensor) -> Tensor:
    return (torch.sign(x) * torch.log1p(torch.abs(x))).clamp_(-SYMLOG_CLIP, SYMLOG_CLIP)


class FeatureDecoder(nn.Module):
    """Raw entity slots -> (global embedding row ids, normalised wide numeric features).

    ``ids`` (N, K) int64 rows into the layout's embedding space; ``wide`` (N, n_wide) float32,
    where each entity type owns a disjoint block of columns, so a single linear layer on
    ``wide`` is a per-type projection. No learned parameters; all tables are non-persistent buffers.
    """

    id_slot: Tensor
    id_add: Tensor
    id_max: Tensor
    id_off: Tensor
    num_src: Tensor
    num_src2: Tensor
    num_mode: Tensor
    num_arg: Tensor
    num_scale: Tensor
    num_null: Tensor
    num_wcol: Tensor

    def __init__(self, layout: TokenLayout) -> None:
        super().__init__()
        self.n_wide = layout.n_wide
        for k, v in layout.decoder_tables().items():
            self.register_buffer(k, torch.from_numpy(v), persistent=False)

    def forward(self, types: Tensor, raw: Tensor) -> tuple[Tensor, Tensor]:
        t = types.long()
        raw = raw.int()
        ids = torch.gather(raw, 1, self.id_slot[t]).long() + self.id_add[t]
        ids = torch.minimum(ids.clamp_(min=0), self.id_max[t]) + self.id_off[t]
        xi = torch.gather(raw, 1, self.num_src[t])
        yi = torch.gather(raw, 1, self.num_src2[t])
        x, y = xi.float(), yi.float()
        mode, scale = self.num_mode[t], self.num_scale[t]
        sx = x * scale
        out = torch.where(mode == Mode.LIN, sx, symlog(sx))
        out = torch.where(mode == Mode.BIT, ((xi >> self.num_arg[t]) & 1).float(), out)
        out = torch.where(mode == Mode.EQ, (xi == self.num_arg[t]).float(), out)
        missing = xi == -1
        out = torch.where(mode == Mode.MISS, missing.float(), out)
        out = torch.where(mode == Mode.RATIO, (x / y.clamp(min=1.0)).clamp(-2.0, 2.0), out)
        out = torch.where(mode == Mode.SYMDIFF, symlog((x - y) * scale), out)
        out = torch.where(mode == Mode.SYMPROD, symlog(sx * y), out)
        out = out.masked_fill(self.num_null[t] & missing, 0.0)
        wide = out.new_zeros(out.shape[0], self.n_wide + 1).scatter_(1, self.num_wcol[t], out)
        return ids, wide[:, : self.n_wide]
