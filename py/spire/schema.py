"""Constants of the SpireHost observation/action wire protocol, schema v1.

Single source of truth on the Python side for ``notes/obs-schema-v1.md``
(sections 3-8, 11, 12 and appendix A). No other module may hard-code a column
number, header index or enum value: import them from here.

Layout recap (all little-endian int32, 4-byte aligned)::

    offset   0  HEADER    16 int32
    offset  64  GLOBAL    64 int32
    offset 320  TABLEDIR  16 x 5 int32  (table_id, n_rows, n_cols, byte_offset, fnv1a32)
    offset 640  TABLE DATA, row-major, tightly packed in table-id order
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Final, Mapping, Sequence

SCHEMA_VERSION: Final = 1
FRAME_MAGIC: Final = 0x424F5053  # bytes 53 50 4F 42 = "SPOB"
CMD_MAGIC: Final = 0x4D435053  # bytes 53 50 43 4D = "SPCM"

HEADER_LEN: Final = 16
GLOBAL_LEN: Final = 64
N_TABLES: Final = 16
DIR_STRIDE: Final = 5
WORD: Final = 4

GLOBAL_OFFSET: Final = HEADER_LEN * WORD  # 64
DIR_OFFSET: Final = GLOBAL_OFFSET + GLOBAL_LEN * WORD  # 320
DATA_OFFSET: Final = DIR_OFFSET + N_TABLES * DIR_STRIDE * WORD  # 640

FNV_OFFSET_BASIS: Final = 2166136261
FNV_PRIME: Final = 16777619
# The int32 value of the FNV offset basis as it appears in the (int32) directory.
FNV_EMPTY_I32: Final = FNV_OFFSET_BASIS - (1 << 32)


# --------------------------------------------------------------------------- header / global


class H(IntEnum):
    """Header word indices (schema 3.2)."""

    MAGIC = 0
    SCHEMA_VERSION = 1
    MSG_TYPE = 2
    VOCAB_HASH = 3
    CATALOG_HASH = 4
    EPISODE_ID = 5
    STEP_ID = 6
    FRAME_BYTES = 7
    N_TABLES = 8
    GLOBAL_LEN = 9
    FLAGS = 10
    RESULT = 11
    N_ACTIONS = 12
    AUTO_STEPS = 13
    HOST_US = 14
    RESERVED = 15


# Header words ignored by leak / determinism byte comparisons (schema 3.2).
HEADER_VOLATILE: Final = (H.EPISODE_ID, H.STEP_ID, H.AUTO_STEPS, H.HOST_US)


class G(IntEnum):
    """GLOBAL vector indices: 44..51 errata-v1, 52..59 H5, 60..63 reserved.

    44..48 are public history valid in every OBS / TERMINAL frame; 49..51 are protocol slots set
    only in TERMINAL frames (0 in OBS frames) and must never be fed to the policy.
    """

    KIND = 0
    KIND_SUB = 1
    N_SEATS = 2
    VIEW_MODE = 3
    VIEW_SEAT = 4
    ACTING_SEAT = 5
    ASCENSION = 6
    GAME_MODE = 7
    ACT_INDEX = 8
    ACT_ID = 9
    ACT_FLOOR = 10
    TOTAL_FLOOR = 11
    MAP_ROWS = 12
    CUR_ROW = 13
    CUR_COL = 14
    ROOM_TYPE = 15
    BOSS_ENCOUNTER = 16
    SECOND_BOSS_ENCOUNTER = 17
    ANCIENT_EVENT = 18
    IN_COMBAT = 19
    COMBAT_ROUND = 20
    COMBAT_SIDE = 21
    PLAYER_PHASE = 22
    RESERVED_23 = 23
    PROMPT_KEY = 24
    PROMPT_SOURCE_CAT = 25
    PROMPT_SOURCE_ID = 26
    SEL_MIN = 27
    SEL_MAX = 28
    SEL_COUNT = 29
    SEL_FLAGS = 30
    EVENT_ID = 31
    EVENT_PAGE_KEY = 32
    EVENT_FLAGS = 33
    REST_FLAGS = 34
    REWARD_FLAGS = 35
    SHOP_FLAGS = 36
    TREASURE_PHASE = 37
    CARD_REWARD_FLAGS = 38
    BUNDLE_FLAGS = 39
    CONFIRM_KEY = 40
    CRYSTAL_DIVINATIONS = 41
    CRYSTAL_TOOL = 42
    PLAYERS_ALIVE = 43
    COMBATS_WON = 44  # combats won so far this run (CombatManager.CombatWon, event fights included)
    ELITES_WON = 45  # of which in a room of type Elite
    BOSSES_WON = 46  # of which in a room of type Boss (each boss of the A10 double boss counts)
    LAST_WON_ROOM_TYPE = 47  # enum:RoomType vocab id of the last won combat's room; 0 before the first win
    LAST_WON_TOTAL_FLOOR = 48  # TOTAL_FLOOR of the last won combat; -1 before the first win
    TERMINAL_REASON = 49  # TERMINAL only: TerminalReason; 0 in OBS frames
    ABORT_REASON = 50  # TERMINAL with RESULT=ABORTED only: SpireHost.Driver.AbortReason; else 0
    POLICY_DECISIONS = 51  # TERMINAL only: policy decisions of the episode (= the frame's STEP_ID)
    SEAT_SCREEN0 = 52  # H5: current screen Kind (0 = none); Stage A does not tokenize these slots
    SEAT_SCREEN1 = 53
    SEAT_SCREEN2 = 54
    SEAT_SCREEN3 = 55
    SEAT_STATUS0 = 56  # H5: SeatStatus bits
    SEAT_STATUS1 = 57
    SEAT_STATUS2 = 58
    SEAT_STATUS3 = 59


# GLOBAL slots that only carry protocol information in TERMINAL frames (never policy features).
G_TERMINAL_ONLY: Final = (G.TERMINAL_REASON, G.ABORT_REASON, G.POLICY_DECISIONS)

# In ERROR frames global[0] carries the error code instead of the decision kind (schema 7.3).
G_ERROR_CODE: Final = 0

# GLOBAL entries holding vocab ids, with their vocab category (schema 4).
GLOBAL_VOCAB: Final[Mapping[G, str]] = {
    G.GAME_MODE: "enum:GameMode",
    G.ACT_ID: "act",
    G.ROOM_TYPE: "enum:RoomType",
    G.BOSS_ENCOUNTER: "encounter",
    G.SECOND_BOSS_ENCOUNTER: "encounter",
    G.ANCIENT_EVENT: "event",
    G.COMBAT_SIDE: "enum:CombatSide",
    G.PLAYER_PHASE: "enum:PlayerTurnPhase",
    G.PROMPT_KEY: "prompt_key",
    G.EVENT_ID: "event",
    G.EVENT_PAGE_KEY: "event_page",
    G.CONFIRM_KEY: "confirm_key",
    G.CRYSTAL_TOOL: "enum:CrystalSphereToolType",
    G.LAST_WON_ROOM_TYPE: "enum:RoomType",
}


# --------------------------------------------------------------------------- protocol enums (appendix A)


class MsgType(IntEnum):
    OBS = 1
    TERMINAL = 2
    ERROR = 3


class CmdType(IntEnum):
    RESET = 1
    STEP = 2
    QUIT = 3


class Result(IntEnum):
    IN_PROGRESS = 0
    WIN = 1
    LOSS = 2
    ABORTED = 3


class TerminalReason(IntEnum):
    """GLOBAL[TERMINAL_REASON] of a TERMINAL frame (errata-v1 section 5, P-5 / P-8).

    RESULT = WIN <-> WIN; RESULT = LOSS <-> DEATH or one of the truncations (DECISION_CAP:
    the RESET / ``--max-decisions`` policy-decision cap; COMBAT_TURN_CAP: ``--turn-cap`` combat
    rounds; OTHER_TRUNCATION); RESULT = ABORTED <-> ABORTED (GLOBAL[ABORT_REASON] says why).
    """

    NONE = 0
    WIN = 1
    DEATH = 2
    DECISION_CAP = 3
    COMBAT_TURN_CAP = 4
    OTHER_TRUNCATION = 5
    ABORTED = 6


TRUNCATION_REASONS: Final = frozenset(
    {TerminalReason.DECISION_CAP, TerminalReason.COMBAT_TURN_CAP, TerminalReason.OTHER_TRUNCATION})
"""Episodes the host stopped without a terminal game state: RESULT = LOSS (they count as failures
in evaluation statistics) but not deaths (no death reward, bootstrapped value target)."""

# RESULT each TERMINAL_REASON must come with (TerminalReason.NONE: hosts predating errata-v1).
TERMINAL_RESULT: Final[Mapping[TerminalReason, int]] = {
    TerminalReason.WIN: 1, TerminalReason.DEATH: 2, TerminalReason.DECISION_CAP: 2,
    TerminalReason.COMBAT_TURN_CAP: 2, TerminalReason.OTHER_TRUNCATION: 2, TerminalReason.ABORTED: 3,
}


class ErrorCode(IntEnum):
    BAD_INDEX = 1
    STALE_STEP = 2
    BAD_MESSAGE = 3


class Kind(IntEnum):
    """Decision kind, GLOBAL[KIND] (A.1)."""

    NONE = 0
    MAP = 1
    COMBAT = 2
    CARD_SELECT = 3
    REWARD = 4
    CARD_REWARD = 5
    EVENT = 6
    SHOP = 7
    REST = 8
    TREASURE = 9
    CONFIRM = 10
    BUNDLE = 11
    CRYSTAL = 12
    TERMINAL = 13


DECISION_KINDS: Final = tuple(k for k in Kind if Kind.MAP <= k <= Kind.CRYSTAL)


class CardSelectSub(IntEnum):
    """GLOBAL[KIND_SUB] for CARD_SELECT (A.2)."""

    SIMPLE_GRID = 1
    SIMPLE_GRID_REWARDS = 2
    COMBAT_PILE = 3
    HAND = 4
    HAND_DISCARD = 5
    HAND_UPGRADE = 6
    DECK_UPGRADE = 7
    DECK_TRANSFORM = 8
    DECK_ENCHANT = 9
    DECK_REMOVAL = 10
    DECK_GENERIC = 11
    CHOOSE_A_CARD = 12


class TreasurePhase(IntEnum):
    """GLOBAL[KIND_SUB] and GLOBAL[TREASURE_PHASE] for TREASURE."""

    CHEST = 1
    RELIC = 2


class ViewMode(IntEnum):
    TEAM = 0
    SEAT = 1


class Zone(IntEnum):
    """CARDS.zone (A.3)."""

    NONE = 0
    HAND = 1
    DRAW = 2
    DISCARD = 3
    EXHAUST = 4
    PLAY = 5
    DECK = 6
    CHOICE = 7
    OFFER = 8
    PREVIEW = 9


# Zones whose rows are canonical multisets (schema 5.2.3). CHOICE rows with pos == -1 are too.
MULTISET_ZONES: Final = frozenset({Zone.DRAW, Zone.DISCARD, Zone.EXHAUST, Zone.DECK})
PRIVATE_ZONES: Final = frozenset(
    {Zone.HAND, Zone.DRAW, Zone.DISCARD, Zone.EXHAUST, Zone.PLAY, Zone.CHOICE}
)


class ItemZone(IntEnum):
    """RELICS.zone / POTIONS.zone."""

    OWNED = 1
    OFFER = 2


class CreatureKind(IntEnum):
    PLAYER = 1
    PET = 2
    ENEMY = 3


class OptionKind(IntEnum):
    EVENT_OPTION = 1
    REST_OPTION = 2
    REWARD_ITEM = 3
    SHOP_ENTRY = 4
    CARD_REWARD_ALT = 5
    TREASURE_RELIC = 6
    BUNDLE = 7
    CONFIRM_CHOICE = 8


class Verb(IntEnum):
    INVALID = 0
    END_TURN = 1
    UNDO_END_TURN = 2
    PLAY_CARD = 3
    USE_POTION = 4
    DISCARD_POTION = 5
    TRAVEL = 6
    SELECT_CARD = 7
    CONFIRM_SELECTION = 8
    CANCEL = 9
    SKIP = 10
    CHOOSE_OPTION = 11
    PICK_OFFER_CARD = 12
    PROCEED = 13
    OPEN_CHEST = 14
    CRYSTAL_CLICK = 15


class RefCategory(IntEnum):
    """REFS.category and GLOBAL[PROMPT_SOURCE_CAT] (values ruled in errata-v1 E-1)."""

    UNKNOWN = 0
    CARD = 1
    RELIC = 2
    POTION = 3
    POWER = 4
    ENCHANTMENT = 5
    AFFLICTION = 6
    ORB = 7
    MONSTER = 8
    EVENT = 9


REF_CATEGORY_VOCAB: Final[Mapping[RefCategory, str]] = {
    RefCategory.CARD: "card",
    RefCategory.RELIC: "relic",
    RefCategory.POTION: "potion",
    RefCategory.POWER: "power",
    RefCategory.ENCHANTMENT: "enchantment",
    RefCategory.AFFLICTION: "affliction",
    RefCategory.ORB: "orb",
    RefCategory.MONSTER: "monster",
    RefCategory.EVENT: "event",
}


class MapNodeState(IntEnum):
    NORMAL = 0
    REACHABLE = 1
    VISITED = 2
    CURRENT = 3


# --------------------------------------------------------------------------- flag words


class FrameFlag(IntFlag):
    """HEADER[FLAGS]."""

    TERMINAL = 1 << 0
    OVERFLOW = 1 << 1
    ELIDED = 1 << 2
    HAS_MAP = 1 << 3
    DECK_IN_COMBAT = 1 << 4
    TEAM_VIEW = 1 << 5
    UPGRADE_PREVIEW = 1 << 6
    TARGET_PREVIEW = 1 << 7


class ResetFlag(IntFlag):
    """RESET.flags (schema 7.1)."""

    AUTO_SINGLE = 1 << 0
    MAP_IN_COMBAT = 1 << 1
    DECK_IN_COMBAT = 1 << 2
    UPGRADE_PREVIEWS = 1 << 3
    TARGET_PREVIEWS = 1 << 4
    ELIDE_UNCHANGED = 1 << 5
    # bits 6, 7: host H5 AskEverySeat / OfferUndoEndTurn (2-4 seats); not named here
    FORECAST = 1 << 8  # notes/forecast/contract.md 2.1: FORECAST trailer table on OBS / TERMINAL frames
    CALC = 1 << 9  # notes/calc/contract.md 2.1: TURN / FOE / ROUTE trailer tables on OBS / TERMINAL frames
    # notes/critic/contract.md 1.2: the HDRAW trailer (draw order). Training only: set by spire.critic.
    # training_reset_flags alone, accepted only by a host started with --training-hidden; never by reset_flags_for
    HIDDEN = 1 << 10


# Flags the Python side sends by default. DECISIONS-H1 F.2 overrides schema 7.1 here:
# combat frames carry neither the map nor the deck by default (schema 7.1 lists both bits as 1).
RESET_FLAGS_DEFAULT: Final = ResetFlag.AUTO_SINGLE | ResetFlag.UPGRADE_PREVIEWS | ResetFlag.TARGET_PREVIEWS
# The literal schema 7.1 defaults, for reference / host-side tests.
RESET_FLAGS_SCHEMA_DEFAULT: Final = RESET_FLAGS_DEFAULT | ResetFlag.MAP_IN_COMBAT | ResetFlag.DECK_IN_COMBAT


def reset_flags_for(forecast: bool, base: int = RESET_FLAGS_DEFAULT, *, calc: bool = False,
                    map_in_combat: bool = False) -> int:
    """RESET flags for a tokenizer config: ``base`` with the FORECAST bit iff ``forecast``, the CALC bit iff
    ``calc`` (CALC needs TARGET_PREVIEWS: notes/calc/contract.md 2.1) and the MAP_IN_COMBAT bit iff
    ``map_in_combat`` (callers pass ``tokenizer.route_summary``: notes/inputs2/contract.md 1)."""
    out = (int(base) & ~int(ResetFlag.FORECAST)) | (int(ResetFlag.FORECAST) if forecast else 0)
    out = (out & ~int(ResetFlag.CALC)) | (int(ResetFlag.CALC) if calc else 0)
    out = (out & ~int(ResetFlag.MAP_IN_COMBAT)) | (int(ResetFlag.MAP_IN_COMBAT) if map_in_combat else 0)
    assert not calc or out & int(ResetFlag.TARGET_PREVIEWS), "ResetFlag.CALC needs TARGET_PREVIEWS (bit 16)"
    # the hidden trailer cannot come from here (notes/critic/contract.md 1.2): no parameter, and not via ``base``
    assert not out & int(ResetFlag.HIDDEN), "reset_flags_for: ResetFlag.HIDDEN (bit 1024) is training-only"
    return out


class SelFlag(IntFlag):
    """GLOBAL[SEL_FLAGS]."""

    CAN_SKIP = 1 << 0
    CANCELABLE = 1 << 1
    MANUAL_CONFIRM = 1 << 2
    UNPOWERED_PREVIEWS = 1 << 3
    PRETEND_PLAYABLE = 1 << 4
    UPGRADE_PREVIEW = 1 << 5


class EventFlag(IntFlag):
    SHARED = 1 << 0
    ANCIENT = 1 << 1
    FINISHED = 1 << 2


class RestFlag(IntFlag):
    HAS_OPTIONS = 1 << 0
    CHOSEN = 1 << 1


class RewardFlag(IntFlag):
    NO_SKIP = 1 << 0


class ShopFlag(IntFlag):
    FAKE_MERCHANT = 1 << 0


class CardRewardFlag(IntFlag):
    CAN_SKIP = 1 << 0
    CAN_REROLL = 1 << 1


class BundleFlag(IntFlag):
    CAN_SKIP = 1 << 0


class CostFlag(IntFlag):
    """CARDS.cost_flags."""

    COSTS_X = 1 << 0
    COST_JUST_UPGRADED = 1 << 1
    STAR_X = 1 << 2
    STAR_JUST_UPGRADED = 1 << 3
    STAR_VISIBLE = 1 << 4


class HandFlag(IntFlag):
    """CARDS.hand_flags (HAND rows only)."""

    CAN_PLAY = 1 << 0
    GLOW_GOLD = 1 << 1
    GLOW_RED = 1 << 2
    UNPLAYABLE_ICON = 1 << 3


N_KEYWORD_BITS: Final = 7  # CARDS.keyword_bits bit (k-1) = CardKeyword k, k = 1..7


class VarFlag(IntFlag):
    JUST_UPGRADED = 1 << 0
    LOCSTRING = 1 << 1
    ENCHANTMENT = 1 << 2


class RelicFlag(IntFlag):
    USED_UP = 1 << 0
    WAX = 1 << 1
    MELTED = 1 << 2


class PotionFlag(IntFlag):
    USABLE = 1 << 0
    DISCARDABLE = 1 << 1
    QUEUED = 1 << 2


class IntentFlag(IntFlag):
    HAS_TIP = 1 << 0


class MapNodeFlag(IntFlag):
    QUEST = 1 << 0
    BOSS = 1 << 1
    SECOND_BOSS = 1 << 2
    START = 1 << 3
    VOTE_SEAT0 = 1 << 4
    VOTE_SEAT1 = 1 << 5
    VOTE_SEAT2 = 1 << 6
    VOTE_SEAT3 = 1 << 7


class OptionFlag(IntFlag):
    LOCKED = 1 << 0
    LETHAL = 1 << 1
    ON_SALE = 1 << 2
    STOCKED = 1 << 3
    AFFORDABLE = 1 << 4
    IS_PROCEED = 1 << 5
    WAS_CHOSEN = 1 << 6
    VOTE_SEAT0 = 1 << 7  # H5 treasure/shared-event votes; ignored by the Stage A decoder
    VOTE_SEAT1 = 1 << 8
    VOTE_SEAT2 = 1 << 9
    VOTE_SEAT3 = 1 << 10


class SeatStatus(IntFlag):
    PROCEEDED = 1 << 0
    VOTED_MAP = 1 << 1
    ACT_READY = 1 << 2
    VOTED_SHARED_EVENT = 1 << 3
    VOTED_TREASURE = 1 << 4
    REST_DONE = 1 << 5


# --------------------------------------------------------------------------- tables and columns


class Table(IntEnum):
    """Table ids (A.7); also the order of tables in the directory and data area."""

    SEATS = 0
    CARDS = 1
    VARS = 2
    CREATURES = 3
    POWERS = 4
    INTENTS = 5
    RELICS = 6
    POTIONS = 7
    ORBS = 8
    ARROWS = 9
    MAP_NODES = 10
    MAP_EDGES = 11
    OPTIONS = 12
    REFS = 13
    CELLS = 14
    ACTIONS = 15


class SeatCol(IntEnum):
    SEAT = 0
    CHARACTER_ID = 1
    CREATURE_ROW = 2
    PRIVATE_VISIBLE = 3
    ALIVE = 4
    HP = 5
    MAX_HP = 6
    BLOCK = 7
    GOLD = 8
    POTION_SLOTS = 9
    ENERGY = 10
    MAX_ENERGY = 11
    STARS = 12
    ORB_CAPACITY = 13
    HAND_COUNT = 14
    DRAW_COUNT = 15
    DISCARD_COUNT = 16
    EXHAUST_COUNT = 17
    DECK_COUNT = 18
    READY_TO_END_TURN = 19
    TURN_NUMBER = 20
    ARROW_SLOTS = 21
    ARROWS_AVAILABLE = 22
    MELODY_BITS = 23
    CARDS_PLAYED_TURN = 24
    ATTACKS_PLAYED_TURN = 25
    SKILLS_PLAYED_TURN = 26
    POWERS_PLAYED_TURN = 27
    CARDS_DRAWN_TURN = 28
    ENERGY_SPENT_TURN = 29
    STARS_GAINED_TURN = 30
    HP_LOSS_EVENTS_COMBAT = 31


class CardCol(IntEnum):
    ZONE = 0
    SEAT = 1
    POS = 2
    COUNT = 3
    CARD_ID = 4
    UPGRADE_LEVEL = 5
    ENERGY_COST = 6
    COST_FLAGS = 7
    ENERGY_COST_COLOR = 8
    STAR_COST = 9
    STAR_COST_COLOR = 10
    KEYWORD_BITS = 11
    ENCHANT_ID = 12
    ENCHANT_AMOUNT = 13
    ENCHANT_STATUS = 14
    AFFLICTION_ID = 15
    AFFLICTION_AMOUNT = 16
    REPLAY_COUNT = 17
    HAND_FLAGS = 18
    UNPLAYABLE_REASON = 19
    VAR_OFF = 20
    VAR_CNT = 21
    LINK_ROW = 22
    AUX = 23
    SEL_COUNT = 24


class VarCol(IntEnum):
    OWNER_TABLE = 0
    OWNER_ROW = 1
    TARGET_ROW = 2
    VAR_ID = 3
    SHOWN = 4
    BASE = 5
    ENCHANTED = 6
    FLAGS = 7


class CreatureCol(IntEnum):
    KIND = 0
    SEAT = 1
    POS = 2
    MODEL_ID = 3
    HP = 4
    MAX_HP = 5
    HP_DISPLAY = 6
    BLOCK = 7
    ALIVE = 8
    HITTABLE = 9
    SLOT_ID = 10
    POWER_OFF = 11
    POWER_CNT = 12
    INTENT_OFF = 13
    INTENT_CNT = 14
    POISON_NEXT_TURN = 15


class PowerCol(IntEnum):
    OWNER_ROW = 0
    POWER_ID = 1
    AMOUNT = 2
    STACK_TYPE = 3
    TYPE_NOW = 4
    APPLIER_ROW = 5
    VISIBLE_MASK = 6
    VAR_OFF = 7
    VAR_CNT = 8
    ORDER = 9


class IntentCol(IntEnum):
    OWNER_ROW = 0
    IDX = 1
    INTENT_TYPE = 2
    HITS = 3
    CARD_COUNT = 4
    DMG_SEAT0 = 5
    DMG_SEAT1 = 6
    DMG_SEAT2 = 7
    DMG_SEAT3 = 8
    FLAGS = 9


class RelicCol(IntEnum):
    SEAT = 0
    ZONE = 1
    POS = 2
    RELIC_ID = 3
    COUNTER = 4
    STATUS = 5
    FLAGS = 6
    VAR_OFF = 7
    VAR_CNT = 8


class PotionCol(IntEnum):
    SEAT = 0
    ZONE = 1
    SLOT = 2
    POTION_ID = 3
    FLAGS = 4
    VAR_OFF = 5
    VAR_CNT = 6


class OrbCol(IntEnum):
    SEAT = 0
    POS = 1
    ORB_ID = 2
    PASSIVE = 3
    EVOKE = 4


class ArrowCol(IntEnum):
    SEAT = 0
    SLOT = 1
    STATUS = 2


class MapNodeCol(IntEnum):
    ROW = 0
    COL = 1
    POINT_TYPE = 2
    STATE = 3
    RESOLVED_ROOM = 4
    FLAGS = 5


class MapEdgeCol(IntEnum):
    FROM_NODE_ROW = 0
    TO_NODE_ROW = 1


class OptionCol(IntEnum):
    KIND = 0
    SEAT = 1
    POS = 2
    KEY_ID = 3
    FLAGS = 4
    PRICE = 5
    AMOUNT = 6
    REF_TABLE = 7
    REF_ROW = 8
    GROUP = 9
    VAR_OFF = 10
    VAR_CNT = 11
    REF_OFF = 12
    REF_CNT = 13


class RefCol(IntEnum):
    OWNER_TABLE = 0
    OWNER_ROW = 1
    CATEGORY = 2
    VOCAB_ID = 3
    ENT_TABLE = 4
    ENT_ROW = 5


class CellCol(IntEnum):
    X = 0
    Y = 1
    HIDDEN = 2
    ITEM_TYPE = 3
    ITEM_GOOD = 4
    ITEM_IDX = 5


class ActionCol(IntEnum):
    VERB = 0
    SEAT = 1
    SRC_TABLE = 2
    SRC_ROW = 3
    TGT_ROW = 4
    AUX = 5
    FLAGS = 6


COLUMNS: Final[tuple[type[IntEnum], ...]] = (
    SeatCol,
    CardCol,
    VarCol,
    CreatureCol,
    PowerCol,
    IntentCol,
    RelicCol,
    PotionCol,
    OrbCol,
    ArrowCol,
    MapNodeCol,
    MapEdgeCol,
    OptionCol,
    RefCol,
    CellCol,
    ActionCol,
)
"""Column enum per table, indexed by table id."""

N_COLS: Final[tuple[int, ...]] = tuple(len(c) for c in COLUMNS)

COLUMN_INDEX: Final[Mapping[Table, Mapping[str, int]]] = {
    Table(t): {m.name.lower(): int(m) for m in cols} for t, cols in enumerate(COLUMNS)
}
"""table -> lower-case column name -> column index (names as in the schema document)."""

MAX_ROWS: Final[tuple[int, ...]] = (4, 1024, 4096, 32, 256, 64, 160, 48, 40, 16, 96, 192, 256, 256, 121, 1024)
"""v1 hard row caps per table (schema 11); exceeding them sets FrameFlag.OVERFLOW."""

MAX_FRAME_BYTES: Final = DATA_OFFSET + WORD * sum(r * c for r, c in zip(MAX_ROWS, N_COLS))


# --------------------------------------------------------------------------- extension table FORECAST (X1)
# notes/forecast/contract.md 2.2-2.3: with ResetFlag.FORECAST the frame carries N_TABLES = 17 and one more
# 5-word directory entry at X0 (the end of the v1 data), followed by the FORECAST data. Table / N_TABLES /
# COLUMNS / MAX_ROWS above stay the 16 v1 tables.


class ForecastCol(IntEnum):
    OWNER_ROW = 0
    IDX = 1
    INTENT_TYPE = 2
    HITS = 3
    CARD_COUNT = 4
    DMG_SEAT0 = 5
    DMG_SEAT1 = 6
    DMG_SEAT2 = 7
    DMG_SEAT3 = 8
    FLAGS = 9
    AHEAD = 10
    CAND = 11
    N_CAND = 12
    CLASS = 13
    PROB = 14
    MOVE_ORD = 15
    SHOWN_ORD = 16


class ForecastFlag(IntFlag):
    HAS_TIP = 1 << 0
    NO_INTENTS = 1 << 1
    SAME_AS_SHOWN = 1 << 2


class ForecastClass(IntEnum):
    FIXED = 0
    CONDITIONAL = 1
    RANDOM = 2
    UNKNOWN = 3


FORECAST_TABLE: Final = 16
FORECAST_N_COLS: Final = len(ForecastCol)
FORECAST_MAX_ROWS: Final = 256
FORECAST_MAX_CAND: Final = 6
FORECAST_PROB_ONE: Final = 10000
N_TABLES_FORECAST: Final = N_TABLES + 1
# The trailer: one directory entry + the FORECAST data.
FORECAST_MAX_BYTES: Final = DIR_STRIDE * WORD + WORD * FORECAST_MAX_ROWS * FORECAST_N_COLS
MAX_FRAME_BYTES_FORECAST: Final = MAX_FRAME_BYTES + FORECAST_MAX_BYTES

# --------------------------------------------------------------------------- extension tables TURN / FOE / ROUTE
# notes/calc/contract.md 2.2-2.3: with ResetFlag.CALC three more 5-word directory entries + data follow the v1
# data and the FORECAST trailer (if that flag is also set), in table-id order 17, 18, 19.


class TurnCol(IntEnum):
    OWNER_ROW = 0
    HP = 1
    BLOCK = 2
    INCOMING = 3
    END_BLOCK = 4
    END_DMG = 5
    END_LOSS = 6
    HP_LOSS_NOW = 7
    BLOCK_REACHABLE = 8
    HP_LOSS_MIN = 9
    FLAGS = 10


class TurnFlag(IntFlag):
    LETHAL_NOW = 1 << 0
    LETHAL_UNAVOIDABLE = 1 << 1
    UNKNOWN_INCOMING = 1 << 2
    REDUCED = 1 << 3
    CAPPED = 1 << 4


class FoeCol(IntEnum):
    OWNER_ROW = 0
    TO_KILL = 1
    DMG_REACHABLE = 2
    INCOMING_FROM = 3
    FLAGS = 4


class FoeFlag(IntFlag):
    KILL = 1 << 0
    GUARDED = 1 << 1
    HP_HIDDEN = 1 << 2
    CAPPED = 1 << 3
    UNHITTABLE = 1 << 4


class RouteCol(IntEnum):
    OWNER_ROW = 0
    ROWS_TO_BOSS = 1
    REST_DIST = 2
    SHOP_DIST = 3
    ELITES_TO_REST = 4
    FIGHTS_TO_REST = 5
    ELITES_TO_BOSS = 6
    FIGHTS_TO_BOSS = 7
    N_REST = 8
    N_SHOP = 9
    N_ELITE = 10
    N_UNKNOWN = 11


TURN_TABLE: Final = 17
FOE_TABLE: Final = 18
ROUTE_TABLE: Final = 19
CALC_TABLES: Final = (TURN_TABLE, FOE_TABLE, ROUTE_TABLE)
TURN_N_COLS: Final = len(TurnCol)
FOE_N_COLS: Final = len(FoeCol)
ROUTE_N_COLS: Final = len(RouteCol)
CALC_N_COLS: Final = (TURN_N_COLS, FOE_N_COLS, ROUTE_N_COLS)
TURN_MAX_ROWS: Final = 4
FOE_MAX_ROWS: Final = 32
ROUTE_MAX_ROWS: Final = 96
CALC_MAX_ROWS: Final = (TURN_MAX_ROWS, FOE_MAX_ROWS, ROUTE_MAX_ROWS)
CALC_MAX_ITEMS: Final = 12
N_TABLES_CALC_EXTRA: Final = 3  # header N_TABLES = 16 + [forecast] + 3 * [calc]
CALC_MAX_BYTES: Final = len(CALC_TABLES) * DIR_STRIDE * WORD + WORD * sum(
    r * c for r, c in zip(CALC_MAX_ROWS, CALC_N_COLS))
MAX_FRAME_BYTES_CALC: Final = MAX_FRAME_BYTES_FORECAST + CALC_MAX_BYTES  # the read limit with both trailers


def n_tables_for(forecast: bool, calc: bool, hidden: bool = False) -> int:
    """Header N_TABLES of a frame carrying the FORECAST / calc / HDRAW trailers (notes/calc/contract.md 2.2,
    notes/critic/contract.md 1.3)."""
    return N_TABLES + int(forecast) + N_TABLES_CALC_EXTRA * int(calc) + int(hidden)


# --------------------------------------------------------------------------- extension table HDRAW (hidden)
# notes/critic/contract.md 1.3: with ResetFlag.HIDDEN one more 5-word directory entry + data follow the calc tables
# (table-id order 16, 17-19, 20). Training only: the draw pile of every visible seat in draw order.


class HDrawCol(IntEnum):
    SEAT = 0  # seat of the pile (0..3)
    POS = 1  # index in the draw pile; 0 = the next card drawn
    OWNER_ROW = 2  # the CARDS row (ZONE = DRAW, SEAT = this seat) encoding this card instance


HDRAW_TABLE: Final = 20
HDRAW_N_COLS: Final = len(HDrawCol)
HDRAW_MAX_ROWS: Final = 1024  # the CARDS cap
# The trailer: one directory entry + the HDRAW data (the contract's "MAX_FRAME_BYTES_HIDDEN = 12,308").
HDRAW_MAX_BYTES: Final = DIR_STRIDE * WORD + WORD * HDRAW_MAX_ROWS * HDRAW_N_COLS
MAX_FRAME_BYTES_HIDDEN: Final = MAX_FRAME_BYTES_CALC + HDRAW_MAX_BYTES  # the read limit with every trailer

# Actions canonical order key columns (schema 5.13).
ACTION_SORT_COLS: Final = (
    ActionCol.VERB,
    ActionCol.SEAT,
    ActionCol.SRC_TABLE,
    ActionCol.SRC_ROW,
    ActionCol.TGT_ROW,
    ActionCol.AUX,
)

# CARDS columns excluded from the multiset content key (schema 5.2.3).
CARD_KEY_EXCLUDED: Final = frozenset(
    {CardCol.POS, CardCol.COUNT, CardCol.VAR_OFF, CardCol.VAR_CNT, CardCol.LINK_ROW, CardCol.SEL_COUNT}
)
CARD_KEY_COLS: Final = tuple(c for c in CardCol if c not in CARD_KEY_EXCLUDED)
# VARS columns appended to the content key, in key order (schema 5.2.3).
VAR_KEY_COLS: Final = (
    VarCol.VAR_ID,
    VarCol.TARGET_ROW,
    VarCol.SHOWN,
    VarCol.BASE,
    VarCol.ENCHANTED,
    VarCol.FLAGS,
)


# --------------------------------------------------------------------------- relations


@dataclass(frozen=True)
class SliceSpec:
    """A CSR slice: owner rows carry (off, cnt) into a contiguous run of child rows."""

    owner: Table
    off_col: int
    cnt_col: int
    child: Table
    child_owner_row_col: int
    child_owner_table_col: int | None  # None when the child table has a single owner table


SLICES: Final[tuple[SliceSpec, ...]] = (
    SliceSpec(Table.CARDS, CardCol.VAR_OFF, CardCol.VAR_CNT, Table.VARS, VarCol.OWNER_ROW, VarCol.OWNER_TABLE),
    SliceSpec(Table.CREATURES, CreatureCol.POWER_OFF, CreatureCol.POWER_CNT, Table.POWERS, PowerCol.OWNER_ROW, None),
    SliceSpec(Table.CREATURES, CreatureCol.INTENT_OFF, CreatureCol.INTENT_CNT, Table.INTENTS, IntentCol.OWNER_ROW, None),
    SliceSpec(Table.POWERS, PowerCol.VAR_OFF, PowerCol.VAR_CNT, Table.VARS, VarCol.OWNER_ROW, VarCol.OWNER_TABLE),
    SliceSpec(Table.RELICS, RelicCol.VAR_OFF, RelicCol.VAR_CNT, Table.VARS, VarCol.OWNER_ROW, VarCol.OWNER_TABLE),
    SliceSpec(Table.POTIONS, PotionCol.VAR_OFF, PotionCol.VAR_CNT, Table.VARS, VarCol.OWNER_ROW, VarCol.OWNER_TABLE),
    SliceSpec(Table.OPTIONS, OptionCol.VAR_OFF, OptionCol.VAR_CNT, Table.VARS, VarCol.OWNER_ROW, VarCol.OWNER_TABLE),
    SliceSpec(Table.OPTIONS, OptionCol.REF_OFF, OptionCol.REF_CNT, Table.REFS, RefCol.OWNER_ROW, RefCol.OWNER_TABLE),
)

VAR_OWNER_TABLES: Final = frozenset(s.owner for s in SLICES if s.child == Table.VARS)

# Row pointers into a fixed table (-1 = null).
POINTERS: Final[Mapping[tuple[Table, int], Table]] = {
    (Table.SEATS, SeatCol.CREATURE_ROW): Table.CREATURES,
    (Table.CARDS, CardCol.LINK_ROW): Table.CARDS,
    (Table.VARS, VarCol.TARGET_ROW): Table.CREATURES,
    (Table.POWERS, PowerCol.OWNER_ROW): Table.CREATURES,
    (Table.POWERS, PowerCol.APPLIER_ROW): Table.CREATURES,
    (Table.INTENTS, IntentCol.OWNER_ROW): Table.CREATURES,
    (Table.MAP_EDGES, MapEdgeCol.FROM_NODE_ROW): Table.MAP_NODES,
    (Table.MAP_EDGES, MapEdgeCol.TO_NODE_ROW): Table.MAP_NODES,
    (Table.ACTIONS, ActionCol.TGT_ROW): Table.CREATURES,
}

# Row pointers whose target table is given by another column of the same row:
# (table, table_col, row_col). -1 in the table column = no reference.
DYNAMIC_POINTERS: Final[tuple[tuple[Table, int, int], ...]] = (
    (Table.VARS, VarCol.OWNER_TABLE, VarCol.OWNER_ROW),
    (Table.OPTIONS, OptionCol.REF_TABLE, OptionCol.REF_ROW),
    (Table.REFS, RefCol.OWNER_TABLE, RefCol.OWNER_ROW),
    (Table.REFS, RefCol.ENT_TABLE, RefCol.ENT_ROW),
    (Table.ACTIONS, ActionCol.SRC_TABLE, ActionCol.SRC_ROW),
)

# Columns holding vocab ids with a fixed category.
VOCAB_COLUMNS: Final[Mapping[tuple[Table, int], str]] = {
    (Table.SEATS, SeatCol.CHARACTER_ID): "character",
    (Table.CARDS, CardCol.CARD_ID): "card",
    (Table.CARDS, CardCol.ENERGY_COST_COLOR): "enum:CardCostColor",
    (Table.CARDS, CardCol.STAR_COST_COLOR): "enum:CardCostColor",
    (Table.CARDS, CardCol.ENCHANT_ID): "enchantment",
    (Table.CARDS, CardCol.ENCHANT_STATUS): "enum:EnchantmentStatus",
    (Table.CARDS, CardCol.AFFLICTION_ID): "affliction",
    (Table.VARS, VarCol.VAR_ID): "var_name",
    (Table.CREATURES, CreatureCol.HP_DISPLAY): "enum:HpDisplay",
    (Table.CREATURES, CreatureCol.SLOT_ID): "slot_name",
    (Table.POWERS, PowerCol.POWER_ID): "power",
    (Table.POWERS, PowerCol.STACK_TYPE): "enum:PowerStackType",
    (Table.POWERS, PowerCol.TYPE_NOW): "enum:PowerType",
    (Table.INTENTS, IntentCol.INTENT_TYPE): "enum:IntentType",
    (Table.RELICS, RelicCol.RELIC_ID): "relic",
    (Table.RELICS, RelicCol.STATUS): "enum:RelicStatus",
    (Table.POTIONS, PotionCol.POTION_ID): "potion",
    (Table.ORBS, OrbCol.ORB_ID): "orb",
    (Table.ARROWS, ArrowCol.STATUS): "enum:ArrowStatus",
    (Table.MAP_NODES, MapNodeCol.POINT_TYPE): "enum:MapPointType",
    (Table.MAP_NODES, MapNodeCol.RESOLVED_ROOM): "enum:RoomType",
    (Table.CELLS, CellCol.ITEM_TYPE): "enum:CrystalSphereItemType",
}

# Polymorphic vocab columns: category depends on a discriminator column.
CREATURE_MODEL_VOCAB: Final[Mapping[CreatureKind, str]] = {
    CreatureKind.PLAYER: "character",
    CreatureKind.PET: "monster",
    CreatureKind.ENEMY: "monster",
}
OPTION_KEY_VOCAB: Final[Mapping[OptionKind, str]] = {
    OptionKind.EVENT_OPTION: "event_option_key",
    OptionKind.REST_OPTION: "rest_option",
    OptionKind.REWARD_ITEM: "enum:RewardType",
    OptionKind.SHOP_ENTRY: "shop_section",
    OptionKind.CARD_REWARD_ALT: "card_reward_alt",
    OptionKind.CONFIRM_CHOICE: "confirm_key",
}
CARD_CHOICE_AUX_VOCAB: Final = "enum:PileType"  # CARDS.aux for CHOICE rows
CRYSTAL_ACTION_AUX_VOCAB: Final = "enum:CrystalSphereToolType"  # Stage A also embeds H5 MEND target seat here

# --------------------------------------------------------------------------- vocab categories (schema 12.1)

VOCAB_MODEL_CATEGORIES: Final = (
    "card", "relic", "potion", "power", "monster", "encounter", "event", "enchantment",
    "affliction", "orb", "act", "character", "modifier", "card_pool", "relic_pool", "potion_pool",
)
VOCAB_ENUM_CATEGORIES: Final = tuple(
    "enum:" + n
    for n in (
        "CardType", "CardRarity", "TargetType", "CardKeyword", "CardTag", "CardCostColor", "PileType",
        "UnplayableReason", "OrbEvokeType", "CardMultiplayerConstraint", "IntentType", "PowerType",
        "PowerStackType", "PowerInstanceType", "RelicRarity", "RelicStatus", "PotionRarity", "PotionUsage",
        "EnchantmentStatus", "HpDisplay", "CombatSide", "PlayerTurnPhase", "MapPointType", "MapPointState",
        "RoomType", "RewardType", "GameMode", "AscensionLevel", "ArrowStatus", "CrystalSphereItemType",
        "CrystalSphereToolType", "StaticHoverTip",
    )
)
VOCAB_STRING_CATEGORIES: Final = (
    "var_name", "event_option_key", "event_page", "rest_option", "card_reward_alt",
    "prompt_key", "confirm_key", "slot_name", "shop_section",
)
VOCAB_CATEGORIES: Final = VOCAB_MODEL_CATEGORIES + VOCAB_ENUM_CATEGORIES + VOCAB_STRING_CATEGORIES
VOCAB_PAD: Final = 0
VOCAB_UNK: Final = 1


# --------------------------------------------------------------------------- self-checks


class SchemaError(ValueError):
    """A frame does not conform to schema v1 (bad magic/version/directory ...)."""


def validate_directory(directory: Sequence[int], frame_bytes: int, flags: int) -> None:
    """Validate a flat table directory (16 x 5 ints) against the v1 constants.

    Checks: table ids in order; ``n_cols`` equal to the schema; ``n_rows`` within
    ``[0, MAX_ROWS]`` (``-1`` only for elided tables when ``FrameFlag.ELIDED`` is set);
    tables tightly packed in table-id order starting at ``DATA_OFFSET`` (an empty
    table's offset is the next free offset); data ending exactly at ``frame_bytes``.
    Raises ``SchemaError`` on the first violation.
    """
    if len(directory) != N_TABLES * DIR_STRIDE:
        raise SchemaError(f"directory has {len(directory)} words, expected {N_TABLES * DIR_STRIDE}")
    elided_ok = bool(flags & int(FrameFlag.ELIDED))  # int: avoids IntFlag.__rand__ on the hot path
    expect = DATA_OFFSET
    for t in range(N_TABLES):
        tid, n, c, off, _h = directory[t * DIR_STRIDE : t * DIR_STRIDE + DIR_STRIDE]
        if tid != t:
            raise SchemaError(f"directory entry {t} has table_id {tid}")
        if c != N_COLS[t]:
            raise SchemaError(f"table {Table(t).name}: n_cols {c} != {N_COLS[t]}")
        if n == -1:
            if not elided_ok:
                raise SchemaError(f"table {Table(t).name}: n_rows=-1 without ELIDED flag")
            if off != 0:
                raise SchemaError(f"table {Table(t).name}: elided table with byte_offset {off}")
            continue
        if not 0 <= n <= MAX_ROWS[t]:
            raise SchemaError(f"table {Table(t).name}: n_rows {n} outside [0, {MAX_ROWS[t]}]")
        if off != expect:
            raise SchemaError(f"table {Table(t).name}: byte_offset {off} != expected {expect}")
        expect += n * c * WORD
    if expect != frame_bytes:
        raise SchemaError(f"table data ends at {expect}, frame has {frame_bytes} bytes")


def _check_constants() -> None:
    assert len(COLUMNS) == N_TABLES == len(Table) == len(MAX_ROWS)
    for cols in COLUMNS:
        assert [int(m) for m in cols] == list(range(len(cols))), cols
    assert N_COLS == (32, 25, 8, 16, 10, 10, 9, 7, 5, 3, 6, 2, 14, 6, 6, 7), N_COLS
    assert [int(m) for m in H] == list(range(HEADER_LEN))
    assert max(G) < GLOBAL_LEN
    assert DATA_OFFSET == 640
    assert [int(m) for m in ForecastCol] == list(range(17)) and FORECAST_N_COLS == 17
    assert FORECAST_MAX_BYTES == 17428
    assert CALC_N_COLS == (11, 5, 12) and CALC_MAX_BYTES == 5484
    assert HDRAW_N_COLS == 3 and HDRAW_MAX_BYTES == 12308 and HDRAW_TABLE == ROUTE_TABLE + 1
    assert all(int(ForecastCol[c.name]) == int(c) for c in IntentCol)  # cols 0-9 mirror INTENTS
    assert len(set(VOCAB_CATEGORIES)) == len(VOCAB_CATEGORIES)
    for cat in (*VOCAB_COLUMNS.values(), *GLOBAL_VOCAB.values(), *CREATURE_MODEL_VOCAB.values(),
                *OPTION_KEY_VOCAB.values(), *REF_CATEGORY_VOCAB.values(), CARD_CHOICE_AUX_VOCAB,
                CRYSTAL_ACTION_AUX_VOCAB):
        assert cat in VOCAB_CATEGORIES, cat


_check_constants()
