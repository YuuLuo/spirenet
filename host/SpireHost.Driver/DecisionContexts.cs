using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;

namespace SpireHost.Driver;

// ---------------------------------------------------------------------------------------------------------
//  Decision contexts (Decision.Context). Plain data for the observation encoder and for tools; live game
//  objects. Owners: the module that builds the decision fills them COMPLETELY; the encoder reads them and
//  nothing else of the builder (ruling D6 of the H1 integration: no side tables, no reference from the encoder
//  to another module). Add fields, never repurpose. Members marked HOST-ONLY carry hidden information and
//  must never be read by an encoder.
// ---------------------------------------------------------------------------------------------------------

public sealed class MapContext
{
    public ActMap Map = null!;

    /// <summary>Points the real UI would mark Travelable (NMapScreen.RecalculateTravelability), canonical (row, col) order.</summary>
    public readonly List<MapPoint> Travelable = new List<MapPoint>(8);

    /// <summary>
    /// MULTI-SEAT (phase H5): the map vote of every seat as the map shows it (NMapScreen.OnPlayerVoteChanged, :698-735;
    /// MapSelectionSynchronizer.GetVote): index = seat, null = that seat has not voted for the current location. Public
    /// to every player. Empty in a single-seat run (the vote resolves at once).
    /// </summary>
    public readonly List<MapPoint?> SeatVotes = new List<MapPoint?>(4);
}

// =========================================================================================================
//  COMBAT
// =========================================================================================================

/// <summary>Why the engine, or one seat, is not waiting for combat input. <see cref="Ready"/> = it is. (driver-spec-combat.md 1.5)</summary>
public enum NotReady
{
    Ready = 0,

    // ---- engine-wide (no seat can act) ----
    NoCombat,                // CombatManager.IsInProgress is false / no turn state
    CombatOverOrEnding,      // IsOverOrEnding: pending loss, or no living primary enemy (Combat/CombatManager.cs:108-139, 242-261)
    EnemySide,               // CombatState.CurrentSide != Player
    SynchronizerNotPlayPhase,// ActionQueueSynchronizer.CombatState != PlayPhase (CombatManager.cs:618): player queues are paused
    ExecutorPaused,          // ActionExecutor.IsPaused (CombatManager.cs:380 .. 617): an enqueued action would poll, not run
    ExecutorRunning,         // ActionExecutor.IsRunning / CurrentlyRunningAction != null (GameActions/ActionExecutor.cs:30-47)
    QueueNotEmpty,           // ActionQueueSet.IsEmpty is false (ActionQueueSet.cs:55-65)
    EndingPlayerTurn,        // EndingPlayerTurnPhaseOne / PhaseTwo (CombatManager.cs:100-102)

    // ---- per seat ----
    NoCombatState,           // Player.PlayerCombatState is null
    PhaseNotPlay,            // PlayerCombatState.Phase != Play (CombatManager.cs:645): Start / AutoPrePlay / AutoPostPlay / End / None
    ReadyToEndTurn,          // IsPlayerReadyToEndTurn: the UI disables hand, potions and the end-turn button (PlayerCmd.cs:215-218)
    Dead,                    // !Creature.IsAlive (NEndTurnButton.cs:286-297 PlayerCanTakeAction)
    NotInExtraTurn,          // PlayersTakingExtraTurn is not empty and does not contain the seat (NPlayerHand.cs:879-893)
    ExecutingEffect,         // IsExecutingCardOrPotionEffect (CombatManager.cs:214-217)
    QueuePaused,             // the seat's action queue is paused (ActionQueueSet.cs:190-194)
    QueueCancelling,         // the seat's queue cancels what is enqueued (ActionQueueSet.cs:105-131): the action would be rejected
    ActionsDisabledLocally,  // CombatManager.PlayerActionsDisabled for the local seat (set by OnEndedTurnLocally)
}

/// <summary>What the COMBAT builder evaluated for one seat (all of it public: greyed-out cards, disabled buttons).</summary>
public sealed class CombatSeatInfo
{
    public int Seat;
    public Player Player = null!;

    /// <summary>Ready = this seat may play cards / use potions / end its turn in this decision; else why not.</summary>
    public NotReady State;

    /// <summary>The seat has ended its turn (hand, belt and button disabled by the UI).</summary>
    public bool ReadyToEndTurn;

    /// <summary>Multi-seat only: the seat has ended its turn and UNDO_END_TURN is offered.</summary>
    public bool CanUndoEndTurn;

    /// <summary>Per hand card (index = hand position): CardModel.CanPlay, its reason bits and the model whose Hook.ShouldPlay vetoed. Filled for seats that can act.</summary>
    public readonly List<bool> HandPlayable = new List<bool>(10);
    public readonly List<UnplayableReason> HandReason = new List<UnplayableReason>(10);
    public readonly List<AbstractModel?> HandPreventer = new List<AbstractModel?>(10);

    /// <summary>Per potion slot: a USE_POTION action / the DISCARD_POTION action is offered in this decision.</summary>
    public readonly List<bool> PotionUsable = new List<bool>(5);
    public readonly List<bool> PotionDiscardable = new List<bool>(5);

    public void Clear()
    {
        State = NotReady.NoCombatState;
        CanUndoEndTurn = false;
        ReadyToEndTurn = false;
        HandPlayable.Clear();
        HandReason.Clear();
        HandPreventer.Clear();
        PotionUsable.Clear();
        PotionDiscardable.Clear();
    }
}

/// <summary>
/// COMBAT decision context (combat request C1). One instance per handler, refilled at every COMBAT decision,
/// valid until the next one.
/// </summary>
public sealed class CombatContext
{
    public CombatState State = null!;

    /// <summary>Index = seat. Which seats may act in this decision and, if not, why.</summary>
    public readonly List<CombatSeatInfo> Seats = new List<CombatSeatInfo>(4);

    /// <summary>CombatState.Creatures without the allocation: allies, then enemies. Index = the Creature ordinal of GameRef.</summary>
    public readonly List<Creature> Creatures = new List<Creature>(12);

    /// <summary>Per creature: Creature.IsHittable as evaluated for this decision (one hook dispatch); 0 = not asked, 1 = yes, 2 = no.</summary>
    public readonly List<byte> Hittable = new List<byte>(12);

    public int AllyCount;

    /// <summary>The (cleared) info slot of a seat; grows the list as needed.</summary>
    public CombatSeatInfo SeatSlot(int seat, Player player)
    {
        while (Seats.Count <= seat)
        {
            Seats.Add(new CombatSeatInfo());
        }
        CombatSeatInfo s = Seats[seat];
        s.Clear();
        s.Seat = seat;
        s.Player = player;
        return s;
    }
}

// =========================================================================================================
//  CARD_SELECT
// =========================================================================================================

/// <summary>
/// The 15 public overloads behind the 12 CardSelectCmd.From* names that go through ICardSelector
/// (Commands/CardSelectCmd.cs). Values are stable (logs, test output). The 13th name, FromChooseABundleScreen
/// (:657), never reaches a selector: its gate and its BUNDLE frame are the events module's.
/// </summary>
public enum CardSelectEntry
{
    None = 0,
    ChooseACardScreen = 1,         // :48
    SimpleGridForRewards = 2,      // :105
    SimpleGrid = 3,                // :165
    CombatPile = 4,                // :223 (4 arguments, delegates to the filter overload)
    CombatPileFiltered = 5,        // :228
    DeckForUpgrade = 6,            // :315
    DeckForTransformation = 7,     // :358
    DeckForEnchantment = 8,        // :405 (player, enchantment, amount, prefs) -> filter overload
    DeckForEnchantmentFiltered = 9,// :410 -> card-list overload
    DeckForEnchantmentCards = 10,  // :418
    DeckForRemoval = 11,           // :476 -> FromDeckGeneric
    DeckGeneric = 12,              // :483
    Hand = 13,                     // :536
    HandForDiscard = 14,           // :593 -> FromHand
    HandForUpgrade = 15,           // :608
}

/// <summary>How the game will signal the choice (which PlayerChoiceContext subclass the caller handed in).</summary>
public enum ChoiceContextKind
{
    None = 0,          // entry point has no context (the deck screens)
    GameAction = 1,    // card / potion being played: the action pauses (GameActionPlayerChoiceContext)
    Hook = 2,          // start-of-turn / shuffle hooks: a GenericHookGameAction is generated (HookPlayerChoiceContext)
    Blocking = 3,      // nothing pauses, the caller simply awaits (KnowledgeDemon, relic pickup, events)
    Other = 4,
}

public sealed class CardSelectContext
{
    public CardSelectKind Kind;
    public Player Player = null!;

    /// <summary>Candidates in canonical order (the ordinal of a SELECT_CARD action indexes this list).</summary>
    public IReadOnlyList<CardModel> Options = Array.Empty<CardModel>();

    /// <summary>For merged candidates (pile-sourced prompts): how many identical instances each option stands for; null = all 1.</summary>
    public IReadOnlyList<int>? OptionCounts;

    /// <summary>Effective bounds after applying the UI-only rules (not the raw numbers the game hands to ICardSelector: see <see cref="RawMin"/>).</summary>
    public int MinSelect;
    public int MaxSelect;

    public bool Cancelable;
    public bool CanSkip;
    public bool RequireManualConfirmation;

    /// <summary>
    /// Cards picked so far in this prompt, in pick order: the INSTANCES handed to the game. For a merged option the
    /// second pick is a different instance than <c>Options[i]</c>; use <see cref="OptionPicked"/> / <see cref="PickedOrdinals"/>
    /// to count picks per option.
    /// </summary>
    public readonly List<CardModel> Picked = new List<CardModel>(8);

    /// <summary>The game's CardSelectorPrefs when the entry point has one.</summary>
    public MegaCrit.Sts2.Core.CardSelection.CardSelectorPrefs? Prefs;

    /// <summary>The model that asked (card being played, relic, power, potion, event ...), when captured.</summary>
    public AbstractModel? SourceModel;

    /// <summary>Pile the candidates come from, when the entry point names one.</summary>
    public PileType? Pile;

    /// <summary>True while a combat is in progress (the encoder then includes the combat tables).</summary>
    public bool InCombat;

    // ---- combat request C2 ----

    /// <summary>Which CardSelectCmd overload raised the prompt (outermost call).</summary>
    public CardSelectEntry Entry;

    /// <summary>How the game signals the choice: card / potion action, hook action, or a plain blocking await.</summary>
    public ChoiceContextKind ContextKind;

    /// <summary>The raw numbers the game handed to ICardSelector (before the UI-only rules; e.g. FromChooseACardScreen always passes (0, 1)).</summary>
    public int RawMin;
    public int RawMax;

    /// <summary>Number of candidate INSTANCES (= sum of <see cref="OptionCounts"/>).</summary>
    public int InstanceCount;

    /// <summary>True when identical candidates were merged into classes and re-ordered canonically (pile-sourced prompts).</summary>
    public bool Merged;

    /// <summary>Ordinal (index in <see cref="Options"/>) of every pick so far, in pick order. Parallel to <see cref="Picked"/>.</summary>
    public readonly List<int> PickedOrdinals = new List<int>(8);

    /// <summary>Per option: how many of its instances are already picked (obs-schema-v1 CARDS col 24 sel_count).</summary>
    public int[] OptionPicked = Array.Empty<int>();

    /// <summary>FromDeckForEnchantment: what the picked cards will be enchanted with (shown by the real screen).</summary>
    public EnchantmentModel? Enchantment;
    public int EnchantmentAmount;

    /// <summary>
    /// The header the real screen shows, as localisation table + key (never the text): CardSelectorPrefs.Prompt when
    /// the entry point has prefs, else the fixed header of the screen: "gameplay_ui" / "CHOOSE_CARD_HEADER"
    /// (FromChooseACardScreen, NChooseACardSelectionScreen.cs:164) and "gameplay_ui" / "CHOOSE_CARD_UPGRADE_HEADER"
    /// (FromHandForUpgrade, CardSelectCmd.cs:641). obs-schema-v1 GLOBAL 24 prompt_key. Empty = no header.
    /// </summary>
    public string PromptTable = "";
    public string PromptKey = "";

    /// <summary>True when the candidates were re-ordered with CardSelectorPrefs.Comparison, as NSimpleCardSelectScreen.Create sorts what it displays (:81-85, :96-100).</summary>
    public bool SortedByComparison;

    /// <summary>The prompt was raised while it is not this seat's play phase (start-of-turn hook, shuffle hook, enemy turn).</summary>
    public bool OutsidePlayPhase;

    /// <summary>
    /// The prompt was raised during the enemy side's turn (KnowledgeDemon): the UI has hidden the intents of the enemies
    /// that already began their move (DECISIONS-H1 F.6); query them with <see cref="RunContext.IntentPerformed"/>.
    /// </summary>
    public bool DuringEnemyTurn;
}

/// <summary>
/// HOST-ONLY view of a CARD_SELECT frame, implemented by the frame (<c>Decision.Source</c>) - deliberately NOT part of
/// the context, so that an encoder (which reads contexts only) cannot reach it. For verification tools that
/// replay choices recorded in the real client ("the first card the screen shows"). For draw-pile prompts the raw
/// order is hidden information.
/// </summary>
public interface ICardSelectHostView
{
    /// <summary>The candidates exactly as the game handed them to ICardSelector.GetSelectedCards (the order the real screen displays).</summary>
    IReadOnlyList<CardModel> RawOptions { get; }

    /// <summary>The option ordinal (index in CardSelectContext.Options) whose class contains this raw instance; -1 if it is not a candidate.</summary>
    int OrdinalOf(CardModel instance);

    /// <summary>
    /// TOOL ONLY. Makes <paramref name="instance"/> the instance that the NEXT SELECT_CARD on its option hands to the
    /// game. A merged option (identical cards of a pile) otherwise hands out its instances in the order the game listed
    /// them; the real screen shows every one of them as its own holder, so a choice recorded on the real screen names
    /// one particular instance, and a verification tool that replays it must be able to hand the game the same one
    /// (which identical card leaves a pile decides the order of what remains). Changes nothing a policy or an encoder
    /// sees: the options, their counts and the action list are the same. False = not an unpicked candidate.
    /// </summary>
    bool PreferInstance(CardModel instance);
}

// =========================================================================================================
//  REWARD / CARD_REWARD / SHOP / REST / TREASURE
// =========================================================================================================

public sealed class RewardContext
{
    public RewardsSet Set = null!;

    /// <summary>Terminal = offered by a combat room (proceeding leaves the room: map / next act / back to the parent event).</summary>
    public bool Terminal;
}

public sealed class CardRewardContext
{
    public CardReward Reward = null!;
    public IReadOnlyList<CardCreationResult> Options = Array.Empty<CardCreationResult>();

    /// <summary>
    /// The buttons of the screen: the list the game passes, minus a REROLL that CardRewardAlternative.Generate would no
    /// longer produce (!CardReward.CanReroll; the real screen regenerates its buttons after a reroll). The ordinal of a
    /// CardRewardAlt action indexes this list.
    /// </summary>
    public IReadOnlyList<CardRewardAlternative> Alternatives = Array.Empty<CardRewardAlternative>();
}

public sealed class ShopContext
{
    public MerchantInventory Inventory = null!;

    /// <summary>True for the FakeMerchant event (relics only, no card removal, no price hooks).</summary>
    public bool FakeMerchant;
}

public sealed class RestContext
{
    public IReadOnlyList<RestSiteOption> Options = Array.Empty<RestSiteOption>();

    /// <summary>Successful choices made in this rest site so far (leaving is legal once this is > 0).</summary>
    public int Successes;
}

/// <summary>
/// TREASURE pages (Decision.SubKind = TreasurePhase): CHEST = the single action OPEN_CHEST (auto-advanced when it is
/// the only action; asked when a frame extension added potion actions); RELIC = the offered relics + SKIP, or - once
/// <see cref="Finished"/> - the PROCEED-only page.
/// </summary>
public sealed class TreasureContext
{
    public MegaCrit.Sts2.Core.Rooms.TreasureRoom Room = null!;

    /// <summary>The relics offered by the opened chest (empty on the CHEST page and on the PROCEED-only page).</summary>
    public IReadOnlyList<RelicModel> Relics = Array.Empty<RelicModel>();
    public int GoldGained;

    /// <summary>Relic picking finished (or the chest was empty): the PROCEED-only page (rooms request R1).</summary>
    public bool Finished;

    /// <summary>The chest had no relic (SilverCrucible / Hook.ShouldGenerateTreasure false).</summary>
    public bool Empty;

    /// <summary>Relics obtained from this chest so far (public: shown when taken).</summary>
    public IReadOnlyList<RelicModel> Awarded = Array.Empty<RelicModel>();

    /// <summary>
    /// MULTI-SEAT (phase H5), RELIC page: the vote of every seat as the relic collection shows it
    /// (NTreasureRoomRelicCollection, TreasureRoomRelicSynchronizer.GetPlayerVote): index = seat, value = index in
    /// <see cref="Relics"/> of the relic that seat voted for, -1 = not voted. Empty in a single-seat run and on the other pages.
    /// </summary>
    public readonly List<int> SeatVotes = new List<int>(4);
}

public sealed class ConfirmContext
{
    /// <summary>What is being confirmed: a vocabulary name of category confirm_key (every key a module uses is listed in <see cref="ConfirmKeys"/>).</summary>
    public string Key = "";
    public AbstractModel? SourceModel;
}

/// <summary>
/// Every <see cref="ConfirmContext.Key"/> used by any module (obs request O7): the vocabulary category confirm_key
/// is built from this list (plus the two button names), and a self-test enumerates it. Append only.
/// </summary>
public static class ConfirmKeys
{
    /// <summary>Trial.DoubleDown: the header of NAbandonRunConfirmPopup (NAbandonRunConfirmPopup.cs:65), events module.</summary>
    public const string AbandonRun = "main_menu_ui/ABANDON_RUN_CONFIRMATION.header";

    /// <summary>The CONFIRM buttons (ConfirmChoice ordinal 0 / 1).</summary>
    public const string No = "NO";
    public const string Yes = "YES";

    /// <summary>Every context key a module may put into ConfirmContext.Key, in registration order.</summary>
    public static readonly string[] All = { AbandonRun };
}

public sealed class BundleContext
{
    public Player Player = null!;
    public IReadOnlyList<IReadOnlyList<CardModel>> Bundles = Array.Empty<IReadOnlyList<CardModel>>();
    public bool CanSkip;
}

// =========================================================================================================
//  EVENT (events request R1)
// =========================================================================================================

/// <summary>One option button as the UI shows it (NEventOptionButton._Ready / OnFocus, Nodes.Events/NEventOptionButton.cs:181-214, 266, 337).</summary>
public sealed class EventOptionInfo
{
    /// <summary>Index in EventModel.CurrentOptions (= the GameRef ordinal of its CHOOSE_OPTION action); -1 for the synthesized PROCEED.</summary>
    public int Index;

    public EventOption? Option;

    /// <summary>EventOption.TextKey; "PROCEED" for the synthesized proceed button.</summary>
    public string TextKey = "";

    /// <summary>Red text, not clickable (NEventOptionButton.cs:192, 262).</summary>
    public bool IsLocked;

    /// <summary>The UI dispatches it with option.Chosen() directly (NEventRoom.cs:240-244).</summary>
    public bool IsProceed;

    public bool WasChosen;

    /// <summary>WillKillPlayer(owner): shown as a red glow (NEventOptionButton.cs:214, 266, 338) - public.</summary>
    public bool WillKillPlayer;

    /// <summary>True when the option has an action in the decision.</summary>
    public bool Legal;

    /// <summary>The UI-synthesized PROCEED of a finished event (NEventRoom.cs:217-224).</summary>
    public bool Synthesized;

    /// <summary>EventOption.Relic: the relic icon of Ancient options (NEventOptionButton.cs:203-209).</summary>
    public RelicModel? Relic;

    /// <summary>Numeric variables printed by the option's title / description templates: (name, shown value), in template order (title first).</summary>
    public readonly List<KeyValuePair<string, int>> Vars = new List<KeyValuePair<string, int>>(4);

    /// <summary>Cards of the option's hover tips (CardHoverTip.Card), in hover-tip order.</summary>
    public readonly List<CardModel> Cards = new List<CardModel>(2);

    /// <summary>Canonical models of the other hover tips (relic, potion, power, enchantment ...), in hover-tip order. Static keyword tips are omitted.</summary>
    public readonly List<AbstractModel> Models = new List<AbstractModel>(2);

    internal void Clear()
    {
        Index = 0;
        Option = null;
        TextKey = "";
        IsLocked = IsProceed = WasChosen = WillKillPlayer = Legal = Synthesized = false;
        Relic = null;
        Vars.Clear();
        Cards.Clear();
        Models.Clear();
    }
}

/// <summary>The event screen as the UI shows it (NEventRoom.SetTitle / SetDescription / SetOptions).</summary>
public sealed class EventContext
{
    public EventModel Event = null!;

    /// <summary>True for the synthesized "event finished, proceed" page.</summary>
    public bool Finished;

    /// <summary>EventModel.Id.Entry.</summary>
    public string EventId = "";

    /// <summary>LocEntryKey of the current page description, or null.</summary>
    public string? PageKey;

    public bool IsShared;
    public bool IsAncient;

    /// <summary>Numeric variables printed by the page description: (name, shown value), in template order.</summary>
    public readonly List<KeyValuePair<string, int>> PageVars = new List<KeyValuePair<string, int>>(4);

    /// <summary>Every displayed button in display order (locked ones included); the synthesized PROCEED (Index -1) when <see cref="Finished"/>.</summary>
    public readonly List<EventOptionInfo> Options = new List<EventOptionInfo>(4);

    /// <summary>
    /// MULTI-SEAT (phase H5), shared events only: the vote of every seat on the current page as the event screen shows it
    /// (EventSynchronizer.PlayerVoteChanged / GetPlayerVote): index = seat, value = EventOptionInfo.Index of the option
    /// that seat voted for, -1 = not voted. Empty otherwise.
    /// </summary>
    public readonly List<int> SeatVotes = new List<int>(4);

    private readonly List<EventOptionInfo> _pool = new List<EventOptionInfo>(4);

    /// <summary>Clears the page and fills the page-level fields from the event (white-listed public members only).</summary>
    public void BeginPage(EventModel ev)
    {
        Event = ev;
        Finished = ev.IsFinished;
        EventId = ev.Id.Entry;
        PageKey = ev.Description == null || ev.Description.IsEmpty ? null : ev.Description.LocEntryKey;
        IsShared = ev.IsShared;
        IsAncient = ev is AncientEventModel;
        PageVars.Clear();
        LocText.PrintedVars(ev.DynamicVars, ev.Description, PageVars);
        Options.Clear();
    }

    /// <summary>Appends a (cleared, pooled) option slot.</summary>
    public EventOptionInfo NextOption()
    {
        int n = Options.Count;
        if (_pool.Count <= n)
        {
            _pool.Add(new EventOptionInfo());
        }
        EventOptionInfo o = _pool[n];
        o.Clear();
        Options.Add(o);
        return o;
    }

    /// <summary>
    /// Appends option <paramref name="index"/> of the event exactly as its button shows it: flags, relic icon, the numbers
    /// its title / description print (NEventOptionButton._Ready adds Event.DynamicVars to both, :181-184) and - for a
    /// button that is not locked - the kill warning and the hover-tip entities. <c>Legal</c> stays false: the handler sets
    /// it when it adds the action.
    /// </summary>
    public EventOptionInfo AddOption(EventOption option, int index, bool willKillPlayer)
    {
        EventOptionInfo o = NextOption();
        o.Index = index;
        o.Option = option;
        o.TextKey = option.TextKey;
        o.IsLocked = option.IsLocked;
        o.IsProceed = option.IsProceed;
        o.WasChosen = option.WasChosen;
        // Verifier finding R-14: a locked button shows neither the kill warning nor its hover tips. NEventOptionButton._Ready gives a
        // locked option the locked look INSTEAD of the kill glow (Nodes.Events/NEventOptionButton.cs:210-217), and OnFocus - which shows
        // the hover tips and pulses the glow - does nothing for a locked option (:327-338). The relic icon is shown either way (:203-209).
        bool locked = option.IsLocked;
        o.WillKillPlayer = willKillPlayer && !locked;
        o.Relic = option.Relic;
        LocText.PrintedVars(Event.DynamicVars, option.Title, o.Vars);
        LocText.PrintedVars(Event.DynamicVars, option.Description, o.Vars);
        if (!locked && option.HoverTips != null)
        {
            foreach (MegaCrit.Sts2.Core.HoverTips.IHoverTip tip in option.HoverTips)
            {
                if (tip is MegaCrit.Sts2.Core.HoverTips.CardHoverTip ct)
                {
                    o.Cards.Add(ct.Card);
                }
                else if (tip.CanonicalModel != null)
                {
                    o.Models.Add(tip.CanonicalModel);
                }
            }
        }
        return o;
    }

    /// <summary>Appends the UI-synthesized PROCEED of a finished event (no OPTIONS row: its action has no source).</summary>
    public EventOptionInfo AddSynthesizedProceed()
    {
        EventOptionInfo o = NextOption();
        o.Index = -1;
        o.TextKey = "PROCEED";
        o.IsProceed = true;
        o.Synthesized = true;
        o.Legal = true;
        return o;
    }
}

// =========================================================================================================
//  CRYSTAL (events request R2)
// =========================================================================================================

/// <summary>One cell of the crystal sphere, public part only (obs-schema-v1 5.12).</summary>
public struct CrystalCellView
{
    public int X;
    public int Y;
    public bool Hidden;

    /// <summary>Non-hidden: -1 = empty, else (int)CrystalSphereItemType of the item under the cleared cell. Hidden: -2 (unknown).</summary>
    public int ItemType;

    /// <summary>Non-hidden with an item: IsGood (0/1); else -1.</summary>
    public int ItemGood;

    /// <summary>Non-hidden with an item: index of that item in <see cref="CrystalContext.Exposed"/>; else -1.</summary>
    public int ItemIndex;
}

public sealed class CrystalContext
{
    /// <summary>
    /// HOST-ONLY. The live minigame: its <c>cells[x, y].Item</c> and <c>Items</c> expose the HIDDEN item layout
    /// (CrystalSphereItem.PlaceItem, CrystalSphereMinigame.cs:145-189; obs-schema-v1 5.12). For the frame that applies
    /// the click only. An encoder reads the public view below and never this field.
    /// </summary>
    public CrystalSphereMinigame Minigame = null!;

    // ---- the public view ----

    public int Width;
    public int Height;

    /// <summary>Remaining clicks (NCrystalSphereScreen.cs:250, DivinationsLeft label).</summary>
    public int Divinations;

    /// <summary>The selected tool button (Small / Big).</summary>
    public CrystalSphereMinigame.CrystalSphereToolType Tool;

    /// <summary>All cells in (y, x) order: index = y * Width + x = the CrystalCell ordinal of the actions. Hidden cells carry no item information.</summary>
    public readonly List<CrystalCellView> Cells = new List<CrystalCellView>(121);

    /// <summary>Items with at least one cleared cell, ordered by their first cleared cell in (y, x) scan order.</summary>
    public readonly List<CrystalSphereItem> Exposed = new List<CrystalSphereItem>(8);

    /// <summary>Per exposed item: every one of its cells is cleared (shape and position are public).</summary>
    public readonly List<bool> ExposedComplete = new List<bool>(8);

    /// <summary>Rebuilds the public view from <see cref="Minigame"/>. An item is read only through a cell that is NOT hidden.</summary>
    public void FillPublicView()
    {
        CrystalSphereMinigame m = Minigame;
        CrystalSphereCell[,] cells = m.cells;
        int w = cells.GetLength(0);
        int h = cells.GetLength(1);
        Width = w;
        Height = h;
        Divinations = m.DivinationCount;
        Tool = m.CrystalSphereTool;
        Cells.Clear();
        Exposed.Clear();
        ExposedComplete.Clear();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                CrystalSphereCell c = cells[x, y];
                CrystalCellView v = new CrystalCellView { X = x, Y = y, Hidden = c.IsHidden, ItemType = -2, ItemGood = -1, ItemIndex = -1 };
                if (!c.IsHidden)
                {
                    CrystalSphereItem? item = c.Item;
                    if (item == null)
                    {
                        v.ItemType = -1;
                    }
                    else
                    {
                        int idx = Exposed.IndexOf(item);
                        if (idx < 0)
                        {
                            idx = Exposed.Count;
                            Exposed.Add(item);
                            ExposedComplete.Add(IsComplete(cells, item));
                        }
                        v.ItemType = (int)ItemTypeOf(item);
                        v.ItemGood = item.IsGood ? 1 : 0;
                        v.ItemIndex = idx;
                    }
                }
                Cells.Add(v);
            }
        }
    }

    private static bool IsComplete(CrystalSphereCell[,] cells, CrystalSphereItem item)
    {
        for (int i = 0; i < item.Size.X; i++)
        {
            for (int j = 0; j < item.Size.Y; j++)
            {
                if (cells[item.Position.X + i, item.Position.Y + j].IsHidden)
                {
                    return false;
                }
            }
        }
        return true;
    }

    public static CrystalSphereItemType ItemTypeOf(CrystalSphereItem item) => item switch
    {
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItems.CrystalSphereCardReward => CrystalSphereItemType.CardReward,
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItems.CrystalSphereCurse => CrystalSphereItemType.Curse,
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItems.CrystalSphereGold => CrystalSphereItemType.Gold,
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItems.CrystalSpherePotion => CrystalSphereItemType.Potion,
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItems.CrystalSphereRelic => CrystalSphereItemType.Relic,
        _ => throw new InvalidOperationException("Unknown crystal sphere item " + item.GetType().Name),
    };
}
