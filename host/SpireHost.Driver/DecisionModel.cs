using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;

namespace SpireHost.Driver;

// ---------------------------------------------------------------------------------------------------------
//  Protocol enums. Numeric values are obs-schema-v1 appendix A; do not renumber.
// ---------------------------------------------------------------------------------------------------------

/// <summary>obs-schema-v1 A.1.</summary>
public enum DecisionKind
{
    None = 0,
    Map = 1,
    Combat = 2,
    CardSelect = 3,
    Reward = 4,
    CardReward = 5,
    Event = 6,
    Shop = 7,
    Rest = 8,
    Treasure = 9,
    Confirm = 10,
    Bundle = 11,
    Crystal = 12,
    Terminal = 13,
}

/// <summary>obs-schema-v1 A.2: sub-kind of a CARD_SELECT decision = which CardSelectCmd.From* entry raised it.</summary>
public enum CardSelectKind
{
    Unknown = 0,
    SimpleGrid = 1,
    SimpleGridRewards = 2,
    CombatPile = 3,
    Hand = 4,
    HandDiscard = 5,
    HandUpgrade = 6,
    DeckUpgrade = 7,
    DeckTransform = 8,
    DeckEnchant = 9,
    DeckRemoval = 10,
    DeckGeneric = 11,
    ChooseACard = 12,
}

/// <summary>
/// Public per-seat status bits of a multi-seat run (obs erratum H5, GLOBAL 56-59). Numeric values are part of the
/// observation format; do not renumber.
/// </summary>
[Flags]
public enum SeatStatus
{
    None = 0,

    /// <summary>The seat left its room screen: its map is open.</summary>
    Proceeded = 1 << 0,

    /// <summary>The seat cast its map vote for the current location.</summary>
    VotedMap = 1 << 1,

    /// <summary>The seat pressed the boss PROCEED: its vote to move to the next act is cast.</summary>
    ActReady = 1 << 2,

    /// <summary>The seat voted on the current page of a shared event.</summary>
    VotedSharedPage = 1 << 3,

    /// <summary>The seat voted for a relic of the treasure chest.</summary>
    VotedTreasure = 1 << 4,

    /// <summary>The seat made a successful choice at the rest site.</summary>
    RestDone = 1 << 5,
}

/// <summary>Sub-kind of a TREASURE decision.</summary>
public enum TreasurePhase
{
    Chest = 1,
    Relic = 2,
}

/// <summary>obs-schema-v1 A.6.</summary>
public enum Verb
{
    Invalid = 0,
    EndTurn = 1,
    UndoEndTurn = 2,
    PlayCard = 3,
    UsePotion = 4,
    DiscardPotion = 5,
    Travel = 6,
    SelectCard = 7,
    ConfirmSelection = 8,
    Cancel = 9,
    Skip = 10,
    ChooseOption = 11,
    PickOfferCard = 12,
    Proceed = 13,
    OpenChest = 14,
    CrystalClick = 15,
}

/// <summary>What a <see cref="GameRef"/> points at. Option-like kinds mirror obs-schema-v1 A.5 (option_kind).</summary>
public enum RefKind
{
    None = 0,

    /// <summary>A live CardModel instance (hand card, selection candidate, offered card).</summary>
    Card,

    /// <summary>A live Creature (player, pet or enemy) in the current combat.</summary>
    Creature,

    /// <summary>A potion slot of a seat: Object = Player, Index = slot index.</summary>
    PotionSlot,

    /// <summary>A MapPoint of the current act map.</summary>
    MapPoint,

    /// <summary>An EventOption of the local event (Index = index in EventModel.CurrentOptions).</summary>
    EventOption,

    /// <summary>A RestSiteOption (Index = index in RestSiteSynchronizer.GetLocalOptions()).</summary>
    RestOption,

    /// <summary>A Reward of the reward set on top of the stack (Index = index in RewardsSet.Rewards).</summary>
    Reward,

    /// <summary>A MerchantEntry (Index = index in MerchantInventory.AllEntries).</summary>
    ShopEntry,

    /// <summary>A CardRewardAlternative (skip / reroll / sacrifice) of a card reward.</summary>
    CardRewardAlt,

    /// <summary>A relic offered by an opened treasure chest (Index = index in CurrentRelics).</summary>
    TreasureRelic,

    /// <summary>A card bundle (Index = bundle index); Object = the bundle's card list.</summary>
    Bundle,

    /// <summary>A button of a confirmation popup: Index 0 = no, 1 = yes.</summary>
    ConfirmChoice,

    /// <summary>A Crystal Sphere cell: Object = the cell, Index = y * width + x.</summary>
    CrystalCell,
}

/// <summary>
/// A typed reference to a live game object plus its PUBLIC ordinal. The ordinal is what canonical action
/// ordering uses (never object identity, never a hidden order):
///   Card (hand)        index in PlayerCombatState.Hand.Cards (hand order is public)
///   Card (candidate)   index in the canonical candidate list of the prompt (draw-pile candidates: after
///                      merging identical cards and canonical re-sorting, see DRIVER-API.md section 6)
///   Creature           index in CombatState.Creatures (allies, then enemies in slot order)
///   PotionSlot         slot index
///   MapPoint           row * 64 + col
///   everything else    index in the list named in <see cref="RefKind"/>
/// </summary>
public readonly struct GameRef
{
    public readonly RefKind Kind;
    public readonly object? Object;
    public readonly int Index;

    public GameRef(RefKind kind, object? obj, int index)
    {
        Kind = kind;
        Object = obj;
        Index = index;
    }

    public static readonly GameRef None = new GameRef(RefKind.None, null, -1);

    public bool IsNone => Kind == RefKind.None;

    public static GameRef OfCard(CardModel card, int ordinal) => new GameRef(RefKind.Card, card, ordinal);

    public static GameRef OfCreature(Creature creature, int ordinal) => new GameRef(RefKind.Creature, creature, ordinal);

    public static GameRef OfPotionSlot(Player owner, int slot) => new GameRef(RefKind.PotionSlot, owner, slot);

    public static GameRef OfMapPoint(MapPoint point) => new GameRef(RefKind.MapPoint, point, point.coord.row * 64 + point.coord.col);

    public static GameRef OfEventOption(EventOption option, int index) => new GameRef(RefKind.EventOption, option, index);

    public static GameRef OfRestOption(RestSiteOption option, int index) => new GameRef(RefKind.RestOption, option, index);

    public static GameRef OfReward(Reward reward, int index) => new GameRef(RefKind.Reward, reward, index);

    public static GameRef OfShopEntry(MerchantEntry entry, int index) => new GameRef(RefKind.ShopEntry, entry, index);

    public static GameRef OfCardRewardAlt(CardRewardAlternative alternative, int index) => new GameRef(RefKind.CardRewardAlt, alternative, index);

    public static GameRef OfTreasureRelic(RelicModel relic, int index) => new GameRef(RefKind.TreasureRelic, relic, index);

    public static GameRef OfBundle(IReadOnlyList<CardModel> cards, int index) => new GameRef(RefKind.Bundle, cards, index);

    public static GameRef OfConfirm(bool yes) => new GameRef(RefKind.ConfirmChoice, null, yes ? 1 : 0);

    public static GameRef OfCrystalCell(object cell, int index) => new GameRef(RefKind.CrystalCell, cell, index);

    public CardModel Card => (CardModel)Object!;

    public Creature Creature => (Creature)Object!;

    public Player PotionOwner => (Player)Object!;

    public MapPoint MapPoint => (MapPoint)Object!;

    public EventOption EventOption => (EventOption)Object!;

    public RestSiteOption RestOption => (RestSiteOption)Object!;

    public Reward Reward => (Reward)Object!;

    public MerchantEntry ShopEntry => (MerchantEntry)Object!;

    public CardRewardAlternative CardRewardAlt => (CardRewardAlternative)Object!;

    public RelicModel Relic => (RelicModel)Object!;

    public override string ToString() => Kind == RefKind.None ? "-" : Kind + "#" + Index;
}

[Flags]
public enum ActionFlags
{
    None = 0,

    /// <summary>
    /// Taking a gold reward. With EpisodeOptions.AutoGold the driver core applies the lowest-index action
    /// carrying this flag without asking the policy (DECISIONS-H1 C).
    /// </summary>
    AutoGold = 1,

    /// <summary>
    /// Equivalent to giving up the whole run (Trial double-down confirmation, any "abandon"). Removed from
    /// the action list by the driver core unless EpisodeOptions.AllowAbandon (DECISIONS-H1 B.6).
    /// </summary>
    Abandon = 2,
}

/// <summary>
/// One legal action = verb + typed references + the data its source needs to apply it. A struct on purpose:
/// no per-action allocation, no closure. The action is applied by <see cref="Owner"/> if set, else by
/// <see cref="Decision.Source"/>.
/// </summary>
public struct LegalAction
{
    public Verb Verb;

    /// <summary>Seat (index in RunState.Players) that performs the action.</summary>
    public int Seat;

    /// <summary>The object acted with (card, potion slot, map point, option ...), or None (END_TURN, PROCEED, CONFIRM ...).</summary>
    public GameRef Source;

    /// <summary>The targeted creature, or None.</summary>
    public GameRef Target;

    /// <summary>CRYSTAL: tool; otherwise 0.</summary>
    public int Aux;

    public ActionFlags Flags;

    /// <summary>Opaque data for the source's Apply (never read by the core, the encoder or a policy).</summary>
    public object? Payload;

    /// <summary>
    /// Who applies THIS action when it is not the decision's Source: set by the core for actions that a
    /// room-frame extension added (e.g. out-of-combat potion actions on a MAP / EVENT / SHOP frame).
    /// Null = <see cref="Decision.Source"/>.
    /// </summary>
    public IDecisionSource? Owner;

    public LegalAction(Verb verb, int seat, GameRef source = default, GameRef target = default, int aux = 0, ActionFlags flags = ActionFlags.None, object? payload = null)
    {
        Verb = verb;
        Seat = seat;
        Source = source.Kind == RefKind.None ? GameRef.None : source;
        Target = target.Kind == RefKind.None ? GameRef.None : target;
        Aux = aux;
        Flags = flags;
        Payload = payload;
    }

    /// <summary>Canonical order: (verb, seat, source ordinal, target ordinal, aux). See DRIVER-API.md section 6.</summary>
    public static int Compare(in LegalAction a, in LegalAction b)
    {
        int c = ((int)a.Verb).CompareTo((int)b.Verb);
        if (c != 0)
        {
            return c;
        }
        c = a.Seat.CompareTo(b.Seat);
        if (c != 0)
        {
            return c;
        }
        c = ((int)a.Source.Kind).CompareTo((int)b.Source.Kind);
        if (c != 0)
        {
            return c;
        }
        c = a.Source.Index.CompareTo(b.Source.Index);
        if (c != 0)
        {
            return c;
        }
        c = a.Target.Index.CompareTo(b.Target.Index);
        if (c != 0)
        {
            return c;
        }
        return a.Aux.CompareTo(b.Aux);
    }

    public override string ToString() => $"{Verb}(seat {Seat}, {Source}{(Target.IsNone ? "" : " -> " + Target)}{(Aux != 0 ? ", aux " + Aux : "")})";
}

/// <summary>Whoever built a decision applies the chosen action. Implemented by room handlers and pending frames.</summary>
public interface IDecisionSource
{
    /// <summary>Short stable name for logs / diagnostics ("map", "combat:combat", "rooms:reward" ...).</summary>
    string Name { get; }

    /// <summary>
    /// Applies <c>decision.Actions[actionIndex]</c> on the engine thread. May start game work (enqueue an
    /// action, call a synchronizer, complete a TaskCompletionSource); must NOT pump, wait or block. Tasks
    /// returned by game APIs are handed to <see cref="RunContext.Track"/>. The driver core settles afterwards.
    /// </summary>
    void Apply(RunContext ctx, Decision decision, int actionIndex);
}

/// <summary>
/// A decision point: what kind of screen, who may act, the prompt context and the ordered list of legal
/// actions. A Decision object is valid only until the next decision is asked (sources may reuse instances);
/// policies and observers must not keep references to it or to its action list.
/// </summary>
public sealed class Decision
{
    public DecisionKind Kind;

    /// <summary>Kind-specific: CardSelect -> <see cref="CardSelectKind"/>, Treasure -> <see cref="TreasurePhase"/>, otherwise 0.</summary>
    public int SubKind;

    /// <summary>The single seat this screen belongs to, or -1 when several seats may act (see <see cref="SeatMask"/>).</summary>
    public int ActingSeat;

    /// <summary>Bit i set = seat i has at least one legal action. Computed by <see cref="Seal"/>.</summary>
    public int SeatMask;

    /// <summary>Bit v set = at least one action with Verb v is offered. Computed by <see cref="Seal"/>.</summary>
    public int VerbMask;

    /// <summary>True when the decision is asked from inside a synchronous game callback (no pump re-entry; see RunContext.DecideNested).</summary>
    public bool Nested;

    /// <summary>Who applies the chosen action. Null only for nested decisions, whose caller interprets the answer itself.</summary>
    public IDecisionSource? Source;

    /// <summary>
    /// Typed prompt / screen context for the observation encoder: one of the *Context classes below, matching
    /// <see cref="Kind"/>. Never read by the driver core.
    /// </summary>
    public object? Context;

    public readonly List<LegalAction> Actions = new List<LegalAction>(32);

    /// <summary>Index of this decision point within the episode (auto-advanced ones included). Set by the core.</summary>
    public long Sequence;

    /// <summary>The episode this decision belongs to (run state, seats, counters). Set by the core before a policy or observer sees the decision.</summary>
    public RunContext Episode = null!;

    public bool Sealed { get; private set; }

    public Decision()
    {
    }

    public Decision(DecisionKind kind, IDecisionSource? source, int actingSeat = 0, int subKind = 0, object? context = null)
    {
        Reset(kind, source, actingSeat, subKind, context);
    }

    /// <summary>Re-initialises a reused instance.</summary>
    public Decision Reset(DecisionKind kind, IDecisionSource? source, int actingSeat = 0, int subKind = 0, object? context = null)
    {
        Kind = kind;
        Source = source;
        ActingSeat = actingSeat;
        SubKind = subKind;
        Context = context;
        Nested = false;
        SeatMask = 0;
        VerbMask = 0;
        Sequence = 0;
        Sealed = false;
        Actions.Clear();
        return this;
    }

    public void Add(in LegalAction action)
    {
        if (Sealed)
        {
            throw new InvalidOperationException("Decision is sealed; build a new one.");
        }
        Actions.Add(action);
    }

    public void Add(Verb verb, int seat, GameRef source = default, GameRef target = default, int aux = 0, ActionFlags flags = ActionFlags.None, object? payload = null)
    {
        Add(new LegalAction(verb, seat, source, target, aux, flags, payload));
    }

    /// <summary>
    /// For test code that builds synthetic decisions (obs request O6): exactly what the driver core does before a
    /// decision is shown to anyone - abandon filter off, canonical order, duplicate check, seat / verb masks.
    /// </summary>
    public Decision SealForTest(bool allowAbandon = true)
    {
        Seal(allowAbandon);
        return this;
    }

    /// <summary>
    /// Called once by the driver core before a decision is shown to anyone: removes abandon-class actions
    /// (unless allowed), puts the list into canonical order (stable insertion sort; lists are short), rejects
    /// duplicates of the sort key and computes <see cref="SeatMask"/> / <see cref="VerbMask"/>. Sources must not sort themselves.
    /// </summary>
    internal void Seal(bool allowAbandon)
    {
        if (Sealed)
        {
            return;
        }
        List<LegalAction> a = Actions;
        if (!allowAbandon)
        {
            for (int i = a.Count - 1; i >= 0; i--)
            {
                if ((a[i].Flags & ActionFlags.Abandon) != 0)
                {
                    a.RemoveAt(i);
                }
            }
        }
        for (int i = 1; i < a.Count; i++)
        {
            LegalAction x = a[i];
            int j = i - 1;
            while (j >= 0 && LegalAction.Compare(a[j], x) > 0)
            {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = x;
        }
        int mask = 0;
        int verbs = 0;
        for (int i = 0; i < a.Count; i++)
        {
            if (i > 0 && LegalAction.Compare(a[i - 1], a[i]) == 0)
            {
                throw new InvalidOperationException($"Decision {Kind} from {Source?.Name}: two actions share the canonical key {a[i]}; give them distinct ordinals.");
            }
            int seat = a[i].Seat;
            if (seat >= 0 && seat < 31)
            {
                mask |= 1 << seat;
            }
            verbs |= 1 << (int)a[i].Verb;
        }
        SeatMask = mask;
        VerbMask = verbs;
        Sealed = true;
    }

    /// <summary>Order-sensitive hash of the decision's public shape (kind, sub-kind, seats, every action key). Part of the determinism trace.</summary>
    public ulong ShapeHash()
    {
        ulong h = 14695981039346656037UL;
        void Mix(long v)
        {
            h ^= unchecked((ulong)v);
            h *= 1099511628211UL;
            h ^= h >> 29;
        }
        Mix((int)Kind);
        Mix(SubKind);
        Mix(ActingSeat);
        Mix(Actions.Count);
        for (int i = 0; i < Actions.Count; i++)
        {
            LegalAction a = Actions[i];
            Mix((int)a.Verb);
            Mix(a.Seat);
            Mix((int)a.Source.Kind);
            Mix(a.Source.Index);
            Mix((int)a.Target.Kind);
            Mix(a.Target.Index);
            Mix(a.Aux);
        }
        return h;
    }

    public override string ToString() => $"{Kind}{(SubKind != 0 ? "/" + SubKind : "")} seat {ActingSeat} [{Actions.Count} actions] from {Source?.Name ?? "nested"}";
}
