using System;
using System.Collections.Generic;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using SpireHost.Patches;

namespace SpireHost.Driver.Combat;

/// <summary>
/// Everything the game told CardSelectCmd about one selection prompt and that ICardSelector.GetSelectedCards
/// does NOT receive (TestSupport/ICardSelector.cs:11 only has options, min, max). Captured read-only by the
/// prefixes below; consumed by <see cref="HostCardSelector"/>.
/// </summary>
public sealed class SelectionPrompt
{
    public long Serial;

    /// <summary>The outermost From* overload the game called.</summary>
    public CardSelectEntry Entry;

    /// <summary>The innermost From* overload reached (differs from Entry for the delegating wrappers).</summary>
    public CardSelectEntry InnerEntry;

    public CardSelectKind Kind;
    public Player Player = null!;

    public bool HasPrefs;
    public CardSelectorPrefs Prefs;

    /// <summary>FromChooseACardScreen: the skip button is shown.</summary>
    public bool CanSkip;

    /// <summary>FromCombatPile: the pile object.</summary>
    public CardPile? Pile;

    /// <summary>Pile the candidates come from: the combat pile, Hand for the hand entries, Deck for the deck entries; null for generated lists.</summary>
    public PileType? PileType;

    /// <summary>The model that asked: the explicit <c>source</c> argument, else the model on top of the choice context's stack.</summary>
    public AbstractModel? SourceModel;

    public PlayerChoiceContext? ChoiceContext;
    public ChoiceContextKind ContextKind;

    /// <summary>The card list handed to the entry point (ChooseACard, SimpleGrid, SimpleGridForRewards, DeckForEnchantment(cards)).</summary>
    public IReadOnlyList<CardModel>? GivenCards;

    public List<CardCreationResult>? GivenResults;

    public bool HasFilter;

    /// <summary>
    /// The candidate filter the innermost entry received: the caller-supplied filter of the hand / combat-pile entries,
    /// the filter argument of FromDeckGeneric (for FromDeckForRemoval the lambda that wrapper builds, CardSelectCmd.cs:480).
    /// Null = none (the 4-argument FromCombatPile passes a constant "accept all", which is recorded as none).
    /// The game filters and hands the result to the local selector; the driver evaluates the same predicate itself only
    /// to rebuild the candidate list of a prompt (<see cref="RemoteCandidates"/>: the list for a REMOTE seat, whose
    /// prompt never reaches a selector, and the cross-check of that rebuild against every local prompt).
    /// </summary>
    public Func<CardModel, bool>? Filter;

    /// <summary>FromDeckGeneric: the sortingOrder argument (FromDeckForRemoval passes "curses first, then deck order", CardSelectCmd.cs:480). Null = none.</summary>
    public Func<CardModel, int>? SortingOrder;

    /// <summary>MULTI-SEAT: the prompt is a remote seat's (the game waits in PlayerChoiceSynchronizer.WaitForRemoteChoice) and this is its choice id.</summary>
    public bool Remote;
    public uint RemoteChoiceId;

    public EnchantmentModel? Enchantment;
    public int EnchantmentAmount;

    /// <summary>What the game will pass to ICardSelector.GetSelectedCards for this entry point.</summary>
    public int ExpectMin;
    public int ExpectMax;

    /// <summary>The Task the outermost From* call returned (null while that call is still on the stack).</summary>
    public Task? Task;

    /// <summary>The selector was asked for this prompt (false = the game answered itself, or has not reached the selector yet).</summary>
    public bool Asked;

    internal CardSelectEntry AwaitingInner;
    internal CardSelectEntry AwaitingInner2;

    public override string ToString() => $"#{Serial} {Entry}/{Kind} player {Player?.NetId} min {ExpectMin} max {ExpectMax}";
}

/// <summary>Prefix -> postfix state of one patched From* call (Harmony __state).</summary>
public sealed class SelectionScope
{
    public SelectionPrompt Prompt = null!;
    public SelectionPrompt? Previous;
    public bool Created;
}

/// <summary>
/// Read-only Harmony prefixes / postfixes on every public CardSelectCmd.From* overload that asks an
/// ICardSelector (DECISIONS-H1 B.3; the bundle entry is the events module's, see <see cref="CardSelectEntry"/>).
///
/// Association prompt -> selector call. The prefix runs synchronously in the async method's stub, before
/// the state machine starts, and stores the prompt in an AsyncLocal. The state machine therefore starts -
/// and, after every suspension, resumes - with an ExecutionContext in which <see cref="Current"/> is this
/// prompt, and the selector, which CardSelectCmd calls from inside that state machine, reads exactly the
/// prompt of ITS call, however many From* calls are suspended at the same time (a hook-raised prompt waits
/// for its hook action while other hooks run). The postfix restores the caller's value, so nothing leaks
/// into the caller's flow. No "last write wins" guessing is involved.
///
/// Wrappers (FromHandForDiscard -> FromHand, FromDeckForRemoval -> FromDeckGeneric, the 4-argument
/// FromCombatPile, the two player overloads of FromDeckForEnchantment) call their inner overload
/// synchronously: the inner prefix sees the wrapper's prompt in <see cref="Current"/>, expecting exactly
/// this inner overload, and merges into it instead of creating a second prompt.
///
/// The patches never change an argument or a result.
/// </summary>
public static class SelectionCapture
{
    private static readonly AsyncLocal<SelectionPrompt?> _current = new AsyncLocal<SelectionPrompt?>();
    private static long _serial;
    private static bool _installed;

    /// <summary>Number of captured overloads (= highest <see cref="CardSelectEntry"/> value).</summary>
    public const int EntryCount = 15;

    /// <summary>Calls seen per entry point since process start (index = CardSelectEntry).</summary>
    public static readonly long[] EntryCalls = new long[EntryCount + 1];

    /// <summary>Prompts for which the selector was asked, per entry point.</summary>
    public static readonly long[] EntryAsked = new long[EntryCount + 1];

    /// <summary>The prompt of the From* call the current async flow is inside of, or null.</summary>
    public static SelectionPrompt? Current => _current.Value;

    public static int PatchCount { get; private set; }

    /// <summary>Called at run set-up and at episode end: nothing of a previous run may be associated with a new one.</summary>
    public static void ResetForRun()
    {
        _current.Value = null;
    }

    public static void Install()
    {
        if (_installed)
        {
            return;
        }
        _installed = true;
        Type t = typeof(CardSelectCmd);
        Type ctx = typeof(PlayerChoiceContext);
        Type prefs = typeof(CardSelectorPrefs);
        Type player = typeof(Player);
        Type cards = typeof(IReadOnlyList<CardModel>);

        Patch(CardSelectEntry.ChooseACardScreen, nameof(CardSelectCmd.FromChooseACardScreen), nameof(PreChooseACard), "CardSelectCmd.cs:48",
            ctx, cards, player, typeof(bool));
        Patch(CardSelectEntry.SimpleGridForRewards, nameof(CardSelectCmd.FromSimpleGridForRewards), nameof(PreSimpleGridForRewards), "CardSelectCmd.cs:105",
            ctx, typeof(List<CardCreationResult>), player, prefs);
        Patch(CardSelectEntry.SimpleGrid, nameof(CardSelectCmd.FromSimpleGrid), nameof(PreSimpleGrid), "CardSelectCmd.cs:165",
            ctx, cards, player, prefs);
        Patch(CardSelectEntry.CombatPile, nameof(CardSelectCmd.FromCombatPile), nameof(PreCombatPile), "CardSelectCmd.cs:223",
            ctx, typeof(CardPile), player, prefs);
        Patch(CardSelectEntry.CombatPileFiltered, nameof(CardSelectCmd.FromCombatPile), nameof(PreCombatPileFiltered), "CardSelectCmd.cs:228",
            ctx, typeof(CardPile), player, prefs, typeof(Func<CardModel, bool>));
        Patch(CardSelectEntry.DeckForUpgrade, nameof(CardSelectCmd.FromDeckForUpgrade), nameof(PreDeckForUpgrade), "CardSelectCmd.cs:315",
            player, prefs);
        Patch(CardSelectEntry.DeckForTransformation, nameof(CardSelectCmd.FromDeckForTransformation), nameof(PreDeckForTransformation), "CardSelectCmd.cs:358",
            player, prefs, typeof(Func<CardModel, CardTransformation>));
        Patch(CardSelectEntry.DeckForEnchantment, nameof(CardSelectCmd.FromDeckForEnchantment), nameof(PreDeckForEnchantment), "CardSelectCmd.cs:405",
            player, typeof(EnchantmentModel), typeof(int), prefs);
        Patch(CardSelectEntry.DeckForEnchantmentFiltered, nameof(CardSelectCmd.FromDeckForEnchantment), nameof(PreDeckForEnchantmentFiltered), "CardSelectCmd.cs:410",
            player, typeof(EnchantmentModel), typeof(int), typeof(Func<CardModel, bool>), prefs);
        Patch(CardSelectEntry.DeckForEnchantmentCards, nameof(CardSelectCmd.FromDeckForEnchantment), nameof(PreDeckForEnchantmentCards), "CardSelectCmd.cs:418",
            cards, typeof(EnchantmentModel), typeof(int), prefs);
        // ORDER MATTERS (found by the rebuild cross-check, H5): FromDeckGeneric is patched BEFORE its wrapper FromDeckForRemoval.
        // The wrapper is not async - it calls FromDeckGeneric directly (:480) - and Harmony compiles a patched method's
        // replacement with full optimisation at the moment it is patched: a callee that is not patched yet (so not yet flagged
        // "do not inline") is inlined into it, and its prefix never runs on that path. The async wrappers (FromHandForDiscard,
        // the 4-argument FromCombatPile, the FromDeckForEnchantment chain) call their inner overload from their state machine,
        // which is compiled after every patch is in place. RemoteCandidates.Rebuild needs the arguments of the INNER call
        // (the filter / sortingOrder lambdas the removal wrapper builds) and fails closed when the inner prefix did not run.
        Patch(CardSelectEntry.DeckGeneric, nameof(CardSelectCmd.FromDeckGeneric), nameof(PreDeckGeneric), "CardSelectCmd.cs:483",
            player, prefs, typeof(Func<CardModel, bool>), typeof(Func<CardModel, int>));
        Patch(CardSelectEntry.DeckForRemoval, nameof(CardSelectCmd.FromDeckForRemoval), nameof(PreDeckForRemoval), "CardSelectCmd.cs:476",
            player, prefs, typeof(Func<CardModel, bool>));
        Patch(CardSelectEntry.Hand, nameof(CardSelectCmd.FromHand), nameof(PreHand), "CardSelectCmd.cs:536",
            ctx, player, prefs, typeof(Func<CardModel, bool>), typeof(AbstractModel));
        Patch(CardSelectEntry.HandForDiscard, nameof(CardSelectCmd.FromHandForDiscard), nameof(PreHandForDiscard), "CardSelectCmd.cs:593",
            ctx, player, prefs, typeof(Func<CardModel, bool>), typeof(AbstractModel));
        Patch(CardSelectEntry.HandForUpgrade, nameof(CardSelectCmd.FromHandForUpgrade), nameof(PreHandForUpgrade), "CardSelectCmd.cs:608",
            ctx, player, typeof(AbstractModel));

        // CardSelectCmd.FromChooseABundleScreen (:657) is deliberately left alone: it does not go through
        // ICardSelector (TestMode picks bundles[0], :670) and the events team owns both its gate patch and the
        // BUNDLE frame (notes\host-h1-events.md section 9). Tests\Suites.cs asserts that nothing of this module
        // is attached to it.
        if (PatchCount != EntryCount)
        {
            throw new InvalidOperationException($"SelectionCapture installed {PatchCount} of {EntryCount} CardSelectCmd.From* captures.");
        }
    }

    private static void Patch(CardSelectEntry entry, string method, string prefix, string site, params Type[] signature)
    {
        MethodInfo? target = AccessTools.Method(typeof(CardSelectCmd), method, signature);
        PatchRegistry.PrefixPostfix("combat-select-capture-" + EntryId(entry), target,
            PatchRegistry.M(typeof(SelectionCapture), prefix), PatchRegistry.M(typeof(SelectionCapture), nameof(Post)),
            PatchCategory.Instrumentation,
            "Read-only: captures the prompt context of CardSelectCmd." + method + " (" + site + ": entry kind, prefs, canSkip, pile, source model, player) that ICardSelector.GetSelectedCards does not receive; the postfix only records the returned Task. Arguments and result untouched.");
        PatchCount++;
    }

    public static string EntryId(CardSelectEntry entry)
    {
        return entry switch
        {
            CardSelectEntry.ChooseACardScreen => "choose-a-card",
            CardSelectEntry.SimpleGridForRewards => "simple-grid-rewards",
            CardSelectEntry.SimpleGrid => "simple-grid",
            CardSelectEntry.CombatPile => "combat-pile",
            CardSelectEntry.CombatPileFiltered => "combat-pile-filtered",
            CardSelectEntry.DeckForUpgrade => "deck-upgrade",
            CardSelectEntry.DeckForTransformation => "deck-transform",
            CardSelectEntry.DeckForEnchantment => "deck-enchant",
            CardSelectEntry.DeckForEnchantmentFiltered => "deck-enchant-filtered",
            CardSelectEntry.DeckForEnchantmentCards => "deck-enchant-cards",
            CardSelectEntry.DeckForRemoval => "deck-removal",
            CardSelectEntry.DeckGeneric => "deck-generic",
            CardSelectEntry.Hand => "hand",
            CardSelectEntry.HandForDiscard => "hand-discard",
            CardSelectEntry.HandForUpgrade => "hand-upgrade",
            _ => "none",
        };
    }

    // -----------------------------------------------------------------------------------------------------
    //  Scope handling
    // -----------------------------------------------------------------------------------------------------

    /// <summary>Starts (or, for the inner overload of a wrapper, continues) the prompt of this From* call.</summary>
    private static SelectionPrompt Enter(CardSelectEntry entry, Player? player, out SelectionScope scope)
    {
        EntryCalls[(int)entry]++;
        SelectionPrompt? previous = _current.Value;
        SelectionPrompt prompt;
        bool created;
        if (previous != null && !previous.Asked && previous.Task == null
            && (previous.AwaitingInner == entry || previous.AwaitingInner2 == entry)
            && (player == null || ReferenceEquals(previous.Player, player)))
        {
            // The wrapper that created 'previous' is still on the stack and is calling its inner overload.
            prompt = previous;
            prompt.AwaitingInner = CardSelectEntry.None;
            prompt.AwaitingInner2 = CardSelectEntry.None;
            created = false;
        }
        else
        {
            prompt = new SelectionPrompt { Serial = ++_serial, Entry = entry };
            if (player != null)
            {
                prompt.Player = player;
            }
            created = true;
        }
        prompt.InnerEntry = entry;
        scope = new SelectionScope { Prompt = prompt, Previous = previous, Created = created };
        _current.Value = prompt;
        return prompt;
    }

    /// <summary>Shared postfix: remembers the Task of the outermost call and gives the caller's flow its own value back.</summary>
    private static void Post(object __result, SelectionScope __state)
    {
        if (__state == null)
        {
            return;
        }
        if (__state.Created)
        {
            __state.Prompt.Task = __result as Task;
        }
        _current.Value = __state.Previous;
    }

    private static void SetContext(SelectionPrompt p, PlayerChoiceContext? context, AbstractModel? explicitSource)
    {
        p.ChoiceContext = context;
        p.ContextKind = KindOf(context, p.Player);
        AbstractModel? source = explicitSource ?? context?.LastInvolvedModel;
        if (source == null)
        {
            source = context switch
            {
                HookPlayerChoiceContext hook => hook.Source,
                BranchingPlayerChoiceContext branching => branching.Source,
                _ => null,
            };
        }
        if (source != null)
        {
            p.SourceModel = source;
        }
    }

    /// <summary>
    /// Which signalling path the context will take for this chooser. A card's OnPlay always receives a
    /// BranchingPlayerChoiceContext around the context of whatever played it (Models/CardModel.cs:1696); it
    /// delegates to that context unless the chooser is not its owner, in which case it creates a hook context
    /// (GameActions.Multiplayer/BranchingPlayerChoiceContext.cs:45-73).
    /// </summary>
    private static ChoiceContextKind KindOf(PlayerChoiceContext? context, Player? chooser)
    {
        for (int depth = 0; depth < 8; depth++)
        {
            switch (context)
            {
                case null:
                    return ChoiceContextKind.None;
                case GameActionPlayerChoiceContext:
                    return ChoiceContextKind.GameAction;
                case HookPlayerChoiceContext:
                    return ChoiceContextKind.Hook;
                case BlockingPlayerChoiceContext:
                    return ChoiceContextKind.Blocking;
                case BranchingPlayerChoiceContext branching:
                {
                    PlayerChoiceContext original = branching._originalContext;
                    if (branching._createdContext != null)
                    {
                        return ChoiceContextKind.Hook;
                    }
                    if (!original.OwnerId.HasValue || !original.GameActionType.HasValue || chooser == null || chooser.NetId == original.OwnerId)
                    {
                        context = original;
                        continue;
                    }
                    return ChoiceContextKind.Hook;
                }
                default:
                    return ChoiceContextKind.Other;
            }
        }
        return ChoiceContextKind.Other;
    }

    private static void SetPrefs(SelectionPrompt p, CardSelectorPrefs prefs)
    {
        p.HasPrefs = true;
        p.Prefs = prefs;
        p.ExpectMin = prefs.MinSelect;
        p.ExpectMax = prefs.MaxSelect;
    }

    // -----------------------------------------------------------------------------------------------------
    //  Prefixes (parameter names must equal the game's parameter names)
    // -----------------------------------------------------------------------------------------------------

    private static void PreChooseACard(PlayerChoiceContext context, IReadOnlyList<CardModel> cards, Player player, bool canSkip, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.ChooseACardScreen, player, out __state);
        p.Kind = CardSelectKind.ChooseACard;
        p.CanSkip = canSkip;
        p.GivenCards = cards;
        p.ExpectMin = 0; // CardSelectCmd.cs:64/75: the selector is asked for (0, 1) whatever canSkip is
        p.ExpectMax = 1;
        SetContext(p, context, null);
    }

    private static void PreSimpleGridForRewards(PlayerChoiceContext context, List<CardCreationResult> cards, Player player, CardSelectorPrefs prefs, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.SimpleGridForRewards, player, out __state);
        p.Kind = CardSelectKind.SimpleGridRewards;
        p.GivenResults = cards;
        SetPrefs(p, prefs);
        SetContext(p, context, null);
    }

    private static void PreSimpleGrid(PlayerChoiceContext context, IReadOnlyList<CardModel> cardsIn, Player player, CardSelectorPrefs prefs, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.SimpleGrid, player, out __state);
        p.Kind = CardSelectKind.SimpleGrid;
        p.GivenCards = cardsIn;
        SetPrefs(p, prefs);
        SetContext(p, context, null);
    }

    private static void PreCombatPile(PlayerChoiceContext context, CardPile pile, Player player, CardSelectorPrefs prefs, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.CombatPile, player, out __state);
        p.Kind = CardSelectKind.CombatPile;
        p.Pile = pile;
        p.PileType = pile?.Type;
        SetPrefs(p, prefs);
        SetContext(p, context, null);
        p.AwaitingInner = CardSelectEntry.CombatPileFiltered;
    }

    private static void PreCombatPileFiltered(PlayerChoiceContext context, CardPile pile, Player player, CardSelectorPrefs prefs, Func<CardModel, bool>? filter, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.CombatPileFiltered, player, out __state);
        p.Kind = CardSelectKind.CombatPile;
        p.Pile = pile;
        p.PileType = pile?.Type;
        // The 4-argument overload passes a constant "accept all" lambda (:225); only a caller-supplied filter counts.
        p.HasFilter = filter != null && p.Entry == CardSelectEntry.CombatPileFiltered;
        p.Filter = p.HasFilter ? filter : null;
        SetPrefs(p, prefs);
        SetContext(p, context, null);
    }

    private static void PreDeckForUpgrade(Player player, CardSelectorPrefs prefs, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.DeckForUpgrade, player, out __state);
        p.Kind = CardSelectKind.DeckUpgrade;
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck;
        SetPrefs(p, prefs);
    }

    private static void PreDeckForTransformation(Player player, CardSelectorPrefs prefs, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.DeckForTransformation, player, out __state);
        p.Kind = CardSelectKind.DeckTransform;
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck;
        SetPrefs(p, prefs);
    }

    private static void PreDeckForEnchantment(Player player, EnchantmentModel enchantment, int amount, CardSelectorPrefs prefs, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.DeckForEnchantment, player, out __state);
        p.Kind = CardSelectKind.DeckEnchant;
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck;
        p.Enchantment = enchantment;
        p.EnchantmentAmount = amount;
        SetPrefs(p, prefs);
        p.AwaitingInner = CardSelectEntry.DeckForEnchantmentFiltered;
    }

    private static void PreDeckForEnchantmentFiltered(Player player, EnchantmentModel enchantment, int amount, Func<CardModel?, bool>? additionalFilter, CardSelectorPrefs prefs, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.DeckForEnchantmentFiltered, player, out __state);
        p.Kind = CardSelectKind.DeckEnchant;
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck;
        p.Enchantment = enchantment;
        p.EnchantmentAmount = amount;
        p.HasFilter = additionalFilter != null;
        SetPrefs(p, prefs);
        p.AwaitingInner = CardSelectEntry.DeckForEnchantmentCards;
    }

    private static void PreDeckForEnchantmentCards(IReadOnlyList<CardModel> cards, EnchantmentModel enchantment, int amount, CardSelectorPrefs prefs, out SelectionScope __state)
    {
        // This overload has no player argument: the chooser is the owner of the cards (:432 cards[0].Owner).
        Player? owner = cards != null && cards.Count > 0 ? cards[0].Owner : null;
        SelectionPrompt p = Enter(CardSelectEntry.DeckForEnchantmentCards, owner, out __state);
        p.Kind = CardSelectKind.DeckEnchant;
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck;
        p.Enchantment = enchantment;
        p.EnchantmentAmount = amount;
        p.GivenCards = cards;
        SetPrefs(p, prefs);
    }

    private static void PreDeckForRemoval(Player player, CardSelectorPrefs prefs, Func<CardModel, bool>? filter, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.DeckForRemoval, player, out __state);
        p.Kind = CardSelectKind.DeckRemoval;
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck;
        p.HasFilter = filter != null;
        SetPrefs(p, prefs);
        p.AwaitingInner = CardSelectEntry.DeckGeneric;
    }

    private static void PreDeckGeneric(Player player, CardSelectorPrefs prefs, Func<CardModel, bool>? filter, Func<CardModel, int>? sortingOrder, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.DeckGeneric, player, out __state);
        if (p.Entry == CardSelectEntry.DeckGeneric)
        {
            p.Kind = CardSelectKind.DeckGeneric;
            p.HasFilter = filter != null;
        }
        p.Filter = filter;
        p.SortingOrder = sortingOrder;
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Deck;
        SetPrefs(p, prefs);
    }

    private static void PreHand(PlayerChoiceContext context, Player player, CardSelectorPrefs prefs, Func<CardModel, bool>? filter, AbstractModel source, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.Hand, player, out __state);
        if (p.Entry == CardSelectEntry.Hand)
        {
            p.Kind = CardSelectKind.Hand;
        }
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Hand;
        p.HasFilter = filter != null;
        p.Filter = filter;
        SetPrefs(p, prefs); // the wrapper's prefs plus ShouldGlowGold (:595)
        SetContext(p, context, source);
    }

    private static void PreHandForDiscard(PlayerChoiceContext context, Player player, CardSelectorPrefs prefs, Func<CardModel, bool>? filter, AbstractModel source, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.HandForDiscard, player, out __state);
        p.Kind = CardSelectKind.HandDiscard;
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Hand;
        p.HasFilter = filter != null;
        p.Filter = filter;
        SetPrefs(p, prefs);
        SetContext(p, context, source);
        p.AwaitingInner = CardSelectEntry.Hand;
    }

    private static void PreHandForUpgrade(PlayerChoiceContext context, Player player, AbstractModel source, out SelectionScope __state)
    {
        SelectionPrompt p = Enter(CardSelectEntry.HandForUpgrade, player, out __state);
        p.Kind = CardSelectKind.HandUpgrade;
        p.PileType = MegaCrit.Sts2.Core.Entities.Cards.PileType.Hand;
        p.ExpectMin = 1; // CardSelectCmd.cs:627/634
        p.ExpectMax = 1;
        SetContext(p, context, source);
    }
}
