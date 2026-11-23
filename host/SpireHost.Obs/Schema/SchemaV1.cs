// =========================================================================================================
//  obs-schema-v1 constants (notes/obs-schema-v1.md sections 3-8, 11 and appendix A, as amended by
//  notes/schema-errata-v1.md). Single source of truth on the C# side; mirrors py/spire/schema.py.
//  Values must never be renumbered (schema version bump instead).
// =========================================================================================================

namespace SpireHost.Obs.Schema;

public static class SchemaV1
{
    public const int Version = 1;
    public const int FrameMagic = 0x424F5053;   // "SPOB"
    public const int CommandMagic = 0x4D435053; // "SPCM"
    public const int HeaderLen = 16;
    public const int GlobalLen = 64;
    public const int NTables = 16;
    public const int DirStride = 5;
    public const int GlobalOffsetWords = HeaderLen;                         // 16
    public const int DirOffsetWords = HeaderLen + GlobalLen;                // 80
    public const int DataOffsetWords = DirOffsetWords + NTables * DirStride; // 160 (= byte 640)
    public const uint FnvOffsetBasis = 2166136261;
    public const uint FnvPrime = 16777619;

    /// <summary>Column count per table id (schema 3.3).</summary>
    public static readonly int[] NCols = { 32, 25, 8, 16, 10, 10, 9, 7, 5, 3, 6, 2, 14, 6, 6, 7 };

    /// <summary>v1 hard row caps per table (schema 11). Exceeding one sets FrameFlag.Overflow.</summary>
    public static readonly int[] MaxRows = { 4, 1024, 4096, 32, 256, 64, 160, 48, 40, 16, 96, 192, 256, 256, 121, 1024 };

    public static readonly string[] TableNames =
    {
        "SEATS", "CARDS", "VARS", "CREATURES", "POWERS", "INTENTS", "RELICS", "POTIONS", "ORBS", "ARROWS",
        "MAP_NODES", "MAP_EDGES", "OPTIONS", "REFS", "CELLS", "ACTIONS",
    };

    public static int MaxFrameBytes
    {
        get
        {
            int words = DataOffsetWords;
            for (int t = 0; t < NTables; t++)
            {
                words += MaxRows[t] * NCols[t];
            }
            return words * 4;
        }
    }
}

public static class T
{
    public const int Seats = 0, Cards = 1, Vars = 2, Creatures = 3, Powers = 4, Intents = 5, Relics = 6, Potions = 7,
        Orbs = 8, Arrows = 9, MapNodes = 10, MapEdges = 11, Options = 12, Refs = 13, Cells = 14, Actions = 15;
}

/// <summary>Header word indices (schema 3.2).</summary>
public static class Hd
{
    public const int Magic = 0, SchemaVersion = 1, MsgType = 2, VocabHash = 3, CatalogHash = 4, EpisodeId = 5, StepId = 6,
        FrameBytes = 7, NTables = 8, GlobalLen = 9, Flags = 10, Result = 11, NActions = 12, AutoSteps = 13, HostUs = 14, Reserved = 15;
}

/// <summary>GLOBAL vector indices (schema 4; 44.. are errata-v1 additions in formerly reserved slots).</summary>
public static class Gl
{
    public const int Kind = 0, KindSub = 1, NSeats = 2, ViewMode = 3, ViewSeat = 4, ActingSeat = 5, Ascension = 6, GameMode = 7,
        ActIndex = 8, ActId = 9, ActFloor = 10, TotalFloor = 11, MapRows = 12, CurRow = 13, CurCol = 14, RoomType = 15,
        BossEncounter = 16, SecondBossEncounter = 17, AncientEvent = 18, InCombat = 19, CombatRound = 20, CombatSide = 21,
        PlayerPhase = 22, Reserved23 = 23, PromptKey = 24, PromptSourceCat = 25, PromptSourceId = 26, SelMin = 27, SelMax = 28,
        SelCount = 29, SelFlags = 30, EventId = 31, EventPageKey = 32, EventFlags = 33, RestFlags = 34, RewardFlags = 35,
        ShopFlags = 36, TreasurePhase = 37, CardRewardFlags = 38, BundleFlags = 39, ConfirmKey = 40, CrystalDivinations = 41,
        CrystalTool = 42, PlayersAlive = 43;

    // ---- errata-v1 additions (formerly reserved, all frames unless noted) ----

    /// <summary>Combats won in this run so far (public history: CombatManager.CombatWon).</summary>
    public const int CombatsWon = 44;

    /// <summary>Of those, in rooms of type Elite.</summary>
    public const int ElitesWon = 45;

    /// <summary>Of those, in rooms of type Boss.</summary>
    public const int BossesWon = 46;

    /// <summary>vocab enum:RoomType of the room of the most recent won combat; 0 = none yet.</summary>
    public const int LastWonRoomType = 47;

    /// <summary>TOTAL_FLOOR at which the most recent won combat happened; -1 = none yet.</summary>
    public const int LastWonTotalFloor = 48;

    /// <summary>TERMINAL only: <see cref="TerminalReason"/>; 0 in OBS frames.</summary>
    public const int TerminalReasonIdx = 49;

    /// <summary>TERMINAL with RESULT=ABORTED: SpireHost.Driver.AbortReason (int); else 0.</summary>
    public const int AbortReasonIdx = 50;

    /// <summary>TERMINAL only: policy decisions answered in the episode (= STEP_ID of the TERMINAL frame).</summary>
    public const int PolicyDecisionsIdx = 51;

    /// <summary>
    /// Erratum H5 (multi-seat frames only; 0 in every single-seat frame): GLOBAL[52 + s] = DecisionKind of the screen
    /// seat s has open (0 = none / waiting), GLOBAL[56 + s] = SpireHost.Driver.SeatStatus bits of seat s.
    /// </summary>
    public const int SeatScreen0 = 52, SeatStatus0 = 56;

    /// <summary>ERROR frames only: the error code (also in global[0] per schema 7.3).</summary>
    public const int ErrorCode = 0;
}

public static class MsgType
{
    public const int Obs = 1, Terminal = 2, Error = 3;
}

public static class Result
{
    public const int InProgress = 0, Win = 1, Loss = 2, Aborted = 3;
}

public static class ErrorCode
{
    public const int BadIndex = 1, StaleStep = 2, BadMessage = 3;
}

public static class CmdType
{
    public const int Reset = 1, Step = 2, Quit = 3;
}

/// <summary>GLOBAL[TERMINAL_REASON] (errata-v1).</summary>
public static class TerminalReason
{
    public const int None = 0, Win = 1, Death = 2, DecisionCap = 3, CombatTurnCap = 4, OtherTruncation = 5, Aborted = 6;
}

public static class Zone
{
    public const int None = 0, Hand = 1, Draw = 2, Discard = 3, Exhaust = 4, Play = 5, Deck = 6, Choice = 7, Offer = 8, Preview = 9;
}

public static class ItemZone
{
    public const int Owned = 1, Offer = 2;
}

public static class CreatureKind
{
    public const int Player = 1, Pet = 2, Enemy = 3;
}

public static class OptionKind
{
    public const int EventOption = 1, RestOption = 2, RewardItem = 3, ShopEntry = 4, CardRewardAlt = 5, TreasureRelic = 6, Bundle = 7, ConfirmChoice = 8;
}

/// <summary>REFS.category and GLOBAL[PROMPT_SOURCE_CAT] (errata-v1: the Python provisional numbering adopted).</summary>
public static class RefCategory
{
    public const int Unknown = 0, Card = 1, Relic = 2, Potion = 3, Power = 4, Enchantment = 5, Affliction = 6, Orb = 7, Monster = 8, Event = 9;
}

public static class MapNodeState
{
    public const int Normal = 0, Reachable = 1, Visited = 2, Current = 3;
}

public static class FrameFlag
{
    public const int Terminal = 1 << 0, Overflow = 1 << 1, Elided = 1 << 2, HasMap = 1 << 3, DeckInCombat = 1 << 4, TeamView = 1 << 5,
        UpgradePreview = 1 << 6, TargetPreview = 1 << 7;
}

public static class ResetFlag
{
    public const int AutoSingle = 1 << 0, MapInCombat = 1 << 1, DeckInCombat = 1 << 2, UpgradePreviews = 1 << 3, TargetPreviews = 1 << 4,
        ElideUnchanged = 1 << 5;

    /// <summary>
    /// Erratum H5, RESET with n_seats 2-4 only (a single-seat RESET that sets one of them is refused as before).
    /// Bit 6: ask EVERY seat for its team votes (map, shared-event page) instead of asking once and casting the vote
    /// for all seats (the default, lead ruling Q5). Bit 7: offer UNDO_END_TURN (off by default, lead ruling Q6).
    /// </summary>
    public const int AskEverySeat = 1 << 6, OfferUndoEndTurn = 1 << 7;

    /// <summary>
    /// Move forecast (notes/forecast/contract.md 2.1; 1 to 4 seats): every OBS / TERMINAL frame of the episode carries the
    /// extension table FORECAST in a trailer after the 16 v1 tables (<see cref="SchemaX1"/>).
    /// </summary>
    public const int Forecast = 1 << 8;

    /// <summary>
    /// Turn-end / per-enemy / route sums (notes/calc/contract.md 2.1; 1 to 4 seats; needs <see cref="TargetPreviews"/>): every
    /// OBS / TERMINAL frame of the episode carries the trailer tables TURN, FOE, ROUTE after the v1 data and the FORECAST
    /// trailer (<see cref="SchemaX1"/>).
    /// </summary>
    public const int Calc = 1 << 9;

    /// <summary>Errata-v1: the default Python sends (DECISIONS-H1 F.2: combat frames carry neither map nor deck).</summary>
    public const int Default = AutoSingle | UpgradePreviews | TargetPreviews;
}

public static class SelFlag
{
    public const int CanSkip = 1 << 0, Cancelable = 1 << 1, ManualConfirm = 1 << 2, UnpoweredPreviews = 1 << 3, PretendPlayable = 1 << 4, UpgradePreview = 1 << 5;
}

public static class CostFlag
{
    public const int CostsX = 1 << 0, CostJustUpgraded = 1 << 1, StarX = 1 << 2, StarJustUpgraded = 1 << 3, StarVisible = 1 << 4;
}

public static class HandFlag
{
    public const int CanPlay = 1 << 0, GlowGold = 1 << 1, GlowRed = 1 << 2, UnplayableIcon = 1 << 3;
}

public static class VarFlag
{
    public const int JustUpgraded = 1 << 0, LocString = 1 << 1, Enchantment = 1 << 2;
}

public static class RelicFlag
{
    public const int UsedUp = 1 << 0, Wax = 1 << 1, Melted = 1 << 2;
}

public static class PotionFlag
{
    public const int Usable = 1 << 0, Discardable = 1 << 1, Queued = 1 << 2;
}

public static class MapNodeFlag
{
    public const int Quest = 1 << 0, Boss = 1 << 1, SecondBoss = 1 << 2, Start = 1 << 3;

    /// <summary>Schema 5.10, written since erratum H5: bit (4 + s) = seat s voted for this node (multi-seat MAP frames).</summary>
    public const int VoteSeat0 = 1 << 4;
}

public static class OptionFlag
{
    public const int Locked = 1 << 0, Lethal = 1 << 1, OnSale = 1 << 2, Stocked = 1 << 3, Affordable = 1 << 4, IsProceed = 1 << 5, WasChosen = 1 << 6;

    /// <summary>Erratum H5: bit (7 + s) = seat s voted for this option (TREASURE_RELIC rows; EVENT_OPTION rows of a shared event).</summary>
    public const int VoteSeat0 = 1 << 7;
}

// Column indices, one class per table (names as in the schema document / py/spire/schema.py).

public static class SeatC
{
    public const int Seat = 0, CharacterId = 1, CreatureRow = 2, PrivateVisible = 3, Alive = 4, Hp = 5, MaxHp = 6, Block = 7, Gold = 8,
        PotionSlots = 9, Energy = 10, MaxEnergy = 11, Stars = 12, OrbCapacity = 13, HandCount = 14, DrawCount = 15, DiscardCount = 16,
        ExhaustCount = 17, DeckCount = 18, ReadyToEndTurn = 19, TurnNumber = 20, ArrowSlots = 21, ArrowsAvailable = 22, MelodyBits = 23,
        CardsPlayedTurn = 24, AttacksPlayedTurn = 25, SkillsPlayedTurn = 26, PowersPlayedTurn = 27, CardsDrawnTurn = 28,
        EnergySpentTurn = 29, StarsGainedTurn = 30, HpLossEventsCombat = 31;
}

public static class CardC
{
    public const int Zone = 0, Seat = 1, Pos = 2, Count = 3, CardId = 4, UpgradeLevel = 5, EnergyCost = 6, CostFlags = 7, EnergyCostColor = 8,
        StarCost = 9, StarCostColor = 10, KeywordBits = 11, EnchantId = 12, EnchantAmount = 13, EnchantStatus = 14, AfflictionId = 15,
        AfflictionAmount = 16, ReplayCount = 17, HandFlags = 18, UnplayableReason = 19, VarOff = 20, VarCnt = 21, LinkRow = 22, Aux = 23,
        SelCount = 24;
}

public static class VarC
{
    public const int OwnerTable = 0, OwnerRow = 1, TargetRow = 2, VarId = 3, Shown = 4, Base = 5, Enchanted = 6, Flags = 7;
}

public static class CreatureC
{
    public const int Kind = 0, Seat = 1, Pos = 2, ModelId = 3, Hp = 4, MaxHp = 5, HpDisplay = 6, Block = 7, Alive = 8, Hittable = 9,
        SlotId = 10, PowerOff = 11, PowerCnt = 12, IntentOff = 13, IntentCnt = 14, PoisonNextTurn = 15;
}

public static class PowerC
{
    public const int OwnerRow = 0, PowerId = 1, Amount = 2, StackType = 3, TypeNow = 4, ApplierRow = 5, VisibleMask = 6, VarOff = 7, VarCnt = 8, Order = 9;
}

public static class IntentC
{
    public const int OwnerRow = 0, Idx = 1, IntentType = 2, Hits = 3, CardCount = 4, DmgSeat0 = 5, Flags = 9;
}

/// <summary>
/// FORECAST (extension table 16, notes/forecast/contract.md 2.3): columns 0-9 mean what INTENTS columns 0-9 mean, for the
/// intents of a candidate move of the next (AHEAD 1) or the one after (AHEAD 2) enemy turn.
/// </summary>
public static class ForecastC
{
    public const int OwnerRow = 0, Idx = 1, IntentType = 2, Hits = 3, CardCount = 4, DmgSeat0 = 5, Flags = 9, Ahead = 10, Cand = 11, NCand = 12,
        Class = 13, Prob = 14, MoveOrd = 15, ShownOrd = 16;
}

public static class ForecastFlag
{
    public const int HasTip = 1 << 0, NoIntents = 1 << 1, SameAsShown = 1 << 2;
}

public static class ForecastClass
{
    public const int Fixed = 0, Conditional = 1, Random = 2, Unknown = 3;
}

/// <summary>
/// Schema v1 + X1 (notes/forecast/contract.md 2.2): with <see cref="ResetFlag.Forecast"/> the header says N_TABLES = 17 and
/// the frame ends with one 5-word directory entry (table 16) at X0 = the end of the v1 data, followed by its rows.
/// </summary>
public static class SchemaX1
{
    public const int NTables = 17;
    public const int ForecastTable = 16, ForecastNCols = 17, ForecastMaxRows = 256, ForecastMaxCand = 6, ProbOne = 10000;

    // calc trailer tables (notes/calc/contract.md 2.2-2.3): after FORECAST, in table-id order, iff ResetFlag.Calc
    public const int TurnTable = 17, FoeTable = 18, RouteTable = 19;
    public const int TurnNCols = 11, FoeNCols = 5, RouteNCols = 12;
    public const int TurnMaxRows = 4, FoeMaxRows = 32, RouteMaxRows = 96;
    public const int CalcMaxItems = 12;
}

/// <summary>TURN (trailer table 17, notes/calc/contract.md 2.3): one row for the acting seat's player.</summary>
public static class TurnC
{
    public const int OwnerRow = 0, Hp = 1, Block = 2, Incoming = 3, EndBlock = 4, EndDmg = 5, EndLoss = 6, HpLossNow = 7, BlockReachable = 8,
        HpLossMin = 9, Flags = 10;
}

public static class TurnFlag
{
    public const int LethalNow = 1 << 0, LethalUnavoidable = 1 << 1, UnknownIncoming = 1 << 2, Reduced = 1 << 3, Capped = 1 << 4;
}

/// <summary>FOE (trailer table 18): one row per living enemy.</summary>
public static class FoeC
{
    public const int OwnerRow = 0, ToKill = 1, DmgReachable = 2, IncomingFrom = 3, Flags = 4;
}

public static class FoeFlag
{
    public const int Kill = 1 << 0, Guarded = 1 << 1, HpHidden = 1 << 2, Capped = 1 << 3, Unhittable = 1 << 4;
}

/// <summary>ROUTE (trailer table 19): one row per REACHABLE map node.</summary>
public static class RouteC
{
    public const int OwnerRow = 0, RowsToBoss = 1, RestDist = 2, ShopDist = 3, ElitesToRest = 4, FightsToRest = 5, ElitesToBoss = 6,
        FightsToBoss = 7, NRest = 8, NShop = 9, NElite = 10, NUnknown = 11;
}

public static class RelicC
{
    public const int Seat = 0, Zone = 1, Pos = 2, RelicId = 3, Counter = 4, Status = 5, Flags = 6, VarOff = 7, VarCnt = 8;
}

public static class PotionC
{
    public const int Seat = 0, Zone = 1, Slot = 2, PotionId = 3, Flags = 4, VarOff = 5, VarCnt = 6;
}

public static class OrbC
{
    public const int Seat = 0, Pos = 1, OrbId = 2, Passive = 3, Evoke = 4;
}

public static class ArrowC
{
    public const int Seat = 0, Slot = 1, Status = 2;
}

public static class MapNodeC
{
    public const int Row = 0, Col = 1, PointType = 2, State = 3, ResolvedRoom = 4, Flags = 5;
}

public static class MapEdgeC
{
    public const int From = 0, To = 1;
}

public static class OptionC
{
    public const int Kind = 0, Seat = 1, Pos = 2, KeyId = 3, Flags = 4, Price = 5, Amount = 6, RefTable = 7, RefRow = 8, Group = 9,
        VarOff = 10, VarCnt = 11, RefOff = 12, RefCnt = 13;
}

public static class RefC
{
    public const int OwnerTable = 0, OwnerRow = 1, Category = 2, VocabId = 3, EntTable = 4, EntRow = 5;
}

public static class CellC
{
    public const int X = 0, Y = 1, Hidden = 2, ItemType = 3, ItemGood = 4, ItemIdx = 5;
}

public static class ActionC
{
    public const int Verb = 0, Seat = 1, SrcTable = 2, SrcRow = 3, TgtRow = 4, Aux = 5, Flags = 6;
}
