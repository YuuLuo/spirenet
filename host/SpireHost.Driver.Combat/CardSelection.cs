using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;

namespace SpireHost.Driver.Combat;

/// <summary>Process-wide counters of the selection machinery (reported by the combat self-tests).</summary>
public static class CardSelectStats
{
    public static long Frames;
    public static long FramesMerged;          // prompts whose candidates were merged into classes
    public static long ClassesMerged;         // instances that disappeared into a class of size > 1
    public static long Picks;
    public static long Completed;
    public static long CompletedEmpty;        // skip / cancel / confirm with nothing picked
    public static long AutoCompleted;         // completed by the last pick (no manual confirmation)
    public static long SoftlockAvoided;       // prompt the real screen could not complete and could not cancel (defensive: take all)
    public static long OnlyCancel;            // prompt the real screen could only cancel
    public static long MaxFrameDepth;

    /// <summary>MULTI-SEAT: frames of remote seats (the game waits for the seat's client) and the answers sent as that client's PlayerChoiceMessage.</summary>
    public static long RemoteFrames;
    public static long RemoteAnswers;

    /// <summary>MULTI-SEAT: hand prompts that closed themselves with an empty selection because no selectable card was left (the combat ended, or the hand changed) while they were open.</summary>
    public static long RevalidatedEmpty;
    public static readonly long[] FramesByKind = new long[13];
    public static readonly long[] FramesByContext = new long[5];

    public static object Snapshot() => new
    {
        frames = Frames,
        frames_by_kind = KindCounts(FramesByKind),
        frames_by_context = new { none = FramesByContext[0], game_action = FramesByContext[1], hook = FramesByContext[2], blocking = FramesByContext[3], other = FramesByContext[4] },
        frames_with_merged_classes = FramesMerged,
        instances_merged_away = ClassesMerged,
        picks = Picks,
        completed = Completed,
        completed_empty = CompletedEmpty,
        auto_completed = AutoCompleted,
        softlock_avoided = SoftlockAvoided,
        only_cancel = OnlyCancel,
    };

    public static Dictionary<string, long> KindCounts(long[] byKind)
    {
        Dictionary<string, long> d = new Dictionary<string, long>();
        for (int i = 0; i < byKind.Length; i++)
        {
            if (byKind[i] != 0)
            {
                d[((CardSelectKind)i).ToString()] = byKind[i];
            }
        }
        return d;
    }
}

/// <summary>
/// The run's ICardSelector for one seat. Installed with CardSelectorStack.UseLocalSelector for the local seat
/// (DECISIONS-H1 B.2: the game then runs ReserveChoiceId / SignalPlayerChoiceBegun / SignalPlayerChoiceEnded
/// exactly as with the real screens) and published as RunContext.CardSelector. Never the global _testSelector.
///
/// MULTI-SEAT (phase H5). A remote seat has an instance too, but it is never installed in the game: a remote player's
/// prompt takes the game's remote branch (ReserveChoiceId, SignalPlayerChoiceBegun, WaitForRemoteChoice,
/// SignalPlayerChoiceEnded - e.g. CardSelectCmd.cs:66-99), never a selector (a per-player stack selector would skip
/// the choice id and the pause, CardSelectCmd.cs:60-64). The instance only owns the seat's open prompt:
/// <see cref="OpenRemote"/> parks the CARD_SELECT frame when the game starts to wait, and the frame answers with the
/// seat's PlayerChoiceMessage.
/// </summary>
public sealed class HostCardSelector : ICardSelector
{
    private readonly RunContext _ctx;
    private readonly int _seat;
    private CardSelectFrame? _pending;

    public HostCardSelector(RunContext ctx, int seat)
    {
        _ctx = ctx;
        _seat = seat;
    }

    public int Seat => _seat;

    public CardSelectFrame? Pending => _pending;

    /// <summary>
    /// Called by CardSelectCmd from inside the From* state machine. Parks the request on the decision stack and
    /// returns the incomplete task; the driver core asks the policy at the next quiescent point (DRIVER-API 7.4).
    /// </summary>
    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        RunContext ctx = _ctx;
        SelectionPrompt? prompt = SelectionCapture.Current;
        if (prompt == null)
        {
            Fault(AbortReason.HandlerFault, "card_select_without_context",
                $"ICardSelector.GetSelectedCards(min {minSelect}, max {maxSelect}) was called outside a captured CardSelectCmd.From* call.");
        }
        if (prompt.Asked)
        {
            Fault(AbortReason.HandlerFault, "card_select_context_reused", "The selector was asked twice for the same CardSelectCmd call: " + prompt);
        }
        prompt.Asked = true;
        SelectionCapture.EntryAsked[(int)prompt.Entry]++;
        List<CardModel> list = new List<CardModel>(options);
        string? mismatch = Validate(ctx, prompt, list, minSelect, maxSelect);
        if (mismatch != null)
        {
            Fault(AbortReason.HandlerFault, "card_select_context_mismatch", mismatch + " (" + prompt + ")");
        }
        // The rebuild that gives a REMOTE seat its candidates (RemoteCandidates) is checked against the game on every local prompt.
        RemoteCandidates.LocalChecks++;
        List<CardModel>? rebuilt = RemoteCandidates.Rebuild(prompt);
        int differs = rebuilt == null ? 0 : RemoteCandidates.FirstDifference(rebuilt, list);
        if (rebuilt == null || differs >= 0)
        {
            RemoteCandidates.LocalMismatches++;
            Fault(AbortReason.HandlerFault, "card_select_rebuild_mismatch",
                (rebuilt == null ? "the candidate list of this entry cannot be rebuilt" : $"the rebuilt candidate list ({rebuilt.Count} cards) differs from the game's ({list.Count} cards) at position {differs}") + " (" + prompt + ")");
        }
        if (_pending != null)
        {
            // driver-spec-combat.md 4.4-4: with the local-selector path a seat has at most one open prompt (the
            // action or hook that asked is paused until it is answered).
            Fault(AbortReason.HandlerFault, "card_select_overlap", "A second selection prompt (" + prompt + ") was raised for seat " + _seat + " while " + _pending.Prompt + " is still unanswered.");
        }
        CardSelectFrame frame = new CardSelectFrame(this, _seat, prompt, list, minSelect, maxSelect, null, 0);
        _pending = frame;
        ctx.Push(frame);
        if (ctx.PendingCount > CardSelectStats.MaxFrameDepth)
        {
            CardSelectStats.MaxFrameDepth = ctx.PendingCount;
        }
        return frame.Completion.Task;
    }

    /// <summary>
    /// MULTI-SEAT. The game started to wait for this (remote) seat's answer to <paramref name="prompt"/>
    /// (PlayerChoiceSynchronizer.WaitForRemoteChoice, choice id <paramref name="choiceId"/>): the seat's own client is
    /// showing the prompt. Parks the CARD_SELECT frame; its candidates are what the client's screen receives
    /// (<see cref="RemoteCandidates.Rebuild"/>), its bounds the ones the entry point hands to a selector.
    /// Called from inside the game call (the remote-choice router): pushes the frame, starts nothing.
    /// </summary>
    internal void OpenRemote(SelectionPrompt prompt, VirtualClient client, uint choiceId)
    {
        RunContext ctx = _ctx;
        prompt.Asked = true;
        prompt.Remote = true;
        prompt.RemoteChoiceId = choiceId;
        SelectionCapture.EntryAsked[(int)prompt.Entry]++;
        if (client.Seat != _seat || ctx.SeatOf(prompt.Player) != _seat)
        {
            Fault(AbortReason.HandlerFault, "card_select_context_mismatch", $"remote prompt of player {prompt.Player.NetId} reached the prompt slot of seat {_seat} ({prompt})");
        }
        List<CardModel>? list = RemoteCandidates.Rebuild(prompt);
        if (list == null)
        {
            Fault(AbortReason.HandlerFault, "card_select_rebuild_mismatch", "the candidate list of this entry cannot be rebuilt for a remote seat (" + prompt + ")");
        }
        if (list.Count == 0)
        {
            // Every entry answers an empty candidate list itself, before it waits (e.g. CardSelectCmd.cs:263-266, :556-559).
            Fault(AbortReason.HandlerFault, "card_select_remote_empty", "the game waits for a remote choice among zero candidates (" + prompt + ")");
        }
        if (_pending != null)
        {
            Fault(AbortReason.HandlerFault, "card_select_overlap", "A second selection prompt (" + prompt + ") was raised for seat " + _seat + " while " + _pending.Prompt + " is still unanswered.");
        }
        CardSelectFrame frame = new CardSelectFrame(this, _seat, prompt, list, prompt.ExpectMin, prompt.ExpectMax, client, choiceId);
        _pending = frame;
        ctx.Push(frame);
        CardSelectStats.RemoteFrames++;
        if (ctx.PendingCount > CardSelectStats.MaxFrameDepth)
        {
            CardSelectStats.MaxFrameDepth = ctx.PendingCount;
        }
    }

    /// <summary>
    /// CardReward.OnSelect only reads the per-player selector stack (Rewards/CardReward.cs:185), never the local
    /// selector: card rewards are answered by the reward owner's own stacked selector, which forwards
    /// GetSelectedCards to this object.
    /// </summary>
    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        throw new InvalidOperationException("The run's card selector was asked for a card reward; CardReward.OnSelect only reads the per-player selector stack (the reward owner pushes its own selector).");
    }

    internal void Completed(CardSelectFrame frame)
    {
        if (ReferenceEquals(_pending, frame))
        {
            _pending = null;
        }
    }

    internal void Reset()
    {
        _pending = null;
    }

    /// <summary>
    /// The selector is called from game code, whose task captures the exception; some callers swallow faults.
    /// The failure is therefore also handed to the driver core as a faulted tracked operation, so the episode
    /// aborts with this reason whatever the game does with the exception.
    /// </summary>
    [System.Diagnostics.CodeAnalysis.DoesNotReturn]
    private void Fault(AbortReason reason, string detail, string message)
    {
        EpisodeAbortException e = new EpisodeAbortException(reason, detail, message);
        _ctx.Post(() => Task.FromException(e), "card selection fault");
        throw e;
    }

    /// <summary>The options must be what the captured entry point would hand over; a mismatch means the capture is wrong.</summary>
    private string? Validate(RunContext ctx, SelectionPrompt p, List<CardModel> options, int min, int max)
    {
        if (ctx.SeatOf(p.Player) != _seat)
        {
            return $"prompt of player {p.Player.NetId} reached the selector of seat {_seat}";
        }
        if (min != p.ExpectMin || max != p.ExpectMax)
        {
            return $"selector asked for ({min}, {max}) but the entry point implies ({p.ExpectMin}, {p.ExpectMax})";
        }
        switch (p.Kind)
        {
            case CardSelectKind.ChooseACard:
                if (p.GivenCards == null || p.GivenCards.Count != options.Count)
                {
                    return "ChooseACard options differ from the cards handed to the entry point";
                }
                for (int i = 0; i < options.Count; i++)
                {
                    if (!ReferenceEquals(p.GivenCards[i], options[i]))
                    {
                        return "ChooseACard options differ from the cards handed to the entry point";
                    }
                }
                break;
            case CardSelectKind.Hand:
            case CardSelectKind.HandDiscard:
            case CardSelectKind.HandUpgrade:
            {
                IReadOnlyList<CardModel>? hand = p.Player.PlayerCombatState?.Hand.Cards;
                if (hand == null)
                {
                    return "hand prompt without a combat state";
                }
                // Hand order must be preserved (CardSelectCmd.cs:551 filters the hand in order).
                int at = 0;
                for (int i = 0; i < options.Count; i++)
                {
                    while (at < hand.Count && !ReferenceEquals(hand[at], options[i]))
                    {
                        at++;
                    }
                    if (at >= hand.Count)
                    {
                        return "hand prompt option is not a hand card in hand order";
                    }
                    at++;
                }
                break;
            }
            case CardSelectKind.CombatPile:
            {
                CardPile? pile = p.Pile;
                if (pile == null)
                {
                    return "combat pile prompt without a pile";
                }
                IReadOnlyList<CardModel> cards = pile.Cards;
                for (int i = 0; i < options.Count; i++)
                {
                    if (!Contains(cards, options[i]))
                    {
                        return "combat pile prompt option is not in the " + pile.Type + " pile";
                    }
                }
                break;
            }
            case CardSelectKind.DeckUpgrade:
            case CardSelectKind.DeckTransform:
            case CardSelectKind.DeckEnchant:
            case CardSelectKind.DeckRemoval:
            case CardSelectKind.DeckGeneric:
            {
                IReadOnlyList<CardModel> deck = p.Player.Deck.Cards;
                for (int i = 0; i < options.Count; i++)
                {
                    if (!Contains(deck, options[i]))
                    {
                        return "deck prompt option is not in the deck";
                    }
                }
                break;
            }
        }
        for (int i = 1; i < options.Count; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (ReferenceEquals(options[i], options[j]))
                {
                    return "the game handed over the same card instance twice";
                }
            }
        }
        return null;
    }

    internal static bool Contains(IReadOnlyList<CardModel> list, CardModel card)
    {
        for (int i = 0; i < list.Count; i++)
        {
            if (ReferenceEquals(list[i], card))
            {
                return true;
            }
        }
        return false;
    }
}

/// <summary>
/// One selection prompt = one pending frame. Ordered multi-select is a sequence of single picks plus an
/// explicit completion (obs-schema-v1 7.2); the cards are handed to the game in pick order (the order is
/// meaningful: Glimmer / PhotonCut put them back on the draw pile in that order).
///
/// Effective rules (what the real screen lets the player do; the engine itself validates nothing):
///
///   entry                         final count k (n = candidates)                  other exits
///   ChooseACard                   1                                               SKIP iff canSkip         (NChooseACardSelectionScreen.cs:193-203, 288)
///   HandUpgrade                   1                                               -                        (CardSelectCmd.cs:627)
///   Hand / HandDiscard            Min .. min(Max,n)                               -                        (NPlayerHand.cs:1011-1021)
///   SimpleGrid / ..Rewards        Min .. min(Max,n)                               -                        (NSimpleCardSelectScreen.cs:118-124, 205-232)
///   CombatPile                    min(Min,n) .. min(Max,n)                        -                        (NCombatPileCardSelectScreen.cs:215-241)
///   DeckGeneric / Removal /       Min != Max: Min .. min(Max,n); else exactly Max CANCEL iff prefs.Cancelable (NDeckCardSelectScreen.cs:133-141, 165-173, 247-275;
///   DeckTransform                                                                                           NDeckTransformSelectScreen.cs:179, 238)
///   DeckEnchant                   as above, but never 0                           CANCEL iff Cancelable    (NDeckEnchantSelectScreen.cs:240, 324, 334)
///   DeckUpgrade                   exactly Max                                     CANCEL iff Cancelable    (NDeckUpgradeSelectScreen.cs:296-311: completes only at Max)
///
///   k = 0 is offered as SKIP when the lower bound is 0 (not when CANCEL exists: both hand the game an empty list).
///   Cancelable is honoured only by the four deck screens; the other screens have no close button.
///   Completion: prompts without manual confirmation complete with the pick that reaches the upper bound
///   (CheckIfSelectionComplete); with manual confirmation (or a real range) a CONFIRM action appears once k
///   is in range. When the interval is empty (fewer candidates than the required count) the real screen can
///   only be cancelled - then CANCEL is the only action - or, if it is not cancelable, soft-locks; in that
///   case the frame takes as many cards as there are (the game's own rule when no confirmation is needed,
///   CardSelectCmd: "count &lt;= MinSelect -> all") and counts it in CardSelectStats.SoftlockAvoided.
///
/// Candidates. Hand prompts keep hand order; generated lists (ChooseACard, SimpleGrid) keep the order the
/// game shows them in (the order handed over, or - when CardSelectorPrefs.Comparison is set - that list sorted
/// with the comparison, exactly as NSimpleCardSelectScreen.Create does). Candidates that come from a pile (draw / discard / exhaust pile, deck) are merged into
/// classes of identical cards (<see cref="CardClassKey"/>) and the classes are put in canonical key order:
/// for the draw pile this is mandatory (the game's stable sort leaks the pile order among identical cards,
/// CardSelectCmd.cs:273-291), for the other piles it follows obs-schema-v1 5.2.4 and removes duplicate
/// actions. A SELECT on a class takes its first not-yet-picked instance in the order the game handed over.
/// </summary>
public sealed class CardSelectFrame : IPendingDecision, ICardSelectHostView
{
    private readonly HostCardSelector _selector;
    private readonly int _seat;
    private readonly List<CardModel> _raw;
    private readonly int _rawMin;
    private readonly int _rawMax;
    private readonly CardSelectContext _context = new CardSelectContext();
    private readonly Decision _decision = new Decision();

    private readonly VirtualClient? _remoteClient;              // MULTI-SEAT: the seat's client answers (null = the local selector's task)
    private readonly uint _remoteChoiceId;

    private bool _initialised;
    private List<CardModel> _options = new List<CardModel>();   // canonical representatives
    private List<int>? _counts;                                  // null = every class has one instance
    private List<List<CardModel>>? _members;                     // per option, instances in engine order (merged prompts only)
    private int _effMin;
    private int _effMax;
    private bool _needConfirm;
    private bool _cancelable;
    private bool _zeroAllowed;

    public readonly TaskCompletionSource<IEnumerable<CardModel>> Completion = new TaskCompletionSource<IEnumerable<CardModel>>();

    public SelectionPrompt Prompt { get; }

    public CardSelectContext Context => _context;

    /// <summary>ICardSelectHostView: the candidates exactly as the game handed them over (HOST ONLY: for draw-pile prompts this order is hidden information).</summary>
    public IReadOnlyList<CardModel> RawOptions => _raw;

    /// <summary>ICardSelectHostView: the option ordinal whose class contains this raw instance; -1 if it is not a candidate.</summary>
    public int OrdinalOf(CardModel instance)
    {
        if (!_initialised)
        {
            return -1;
        }
        if (_members == null)
        {
            for (int i = 0; i < _options.Count; i++)
            {
                if (ReferenceEquals(_options[i], instance))
                {
                    return i;
                }
            }
            return -1;
        }
        for (int i = 0; i < _members.Count; i++)
        {
            List<CardModel> m = _members[i];
            for (int k = 0; k < m.Count; k++)
            {
                if (ReferenceEquals(m[k], instance))
                {
                    return i;
                }
            }
        }
        return -1;
    }

    /// <summary>
    /// ICardSelectHostView, TOOL ONLY: the next SELECT_CARD on the option that contains <paramref name="instance"/> hands
    /// exactly this instance to the game (the members of a merged option are otherwise handed out in engine order).
    /// Only the order of the not-yet-picked members of that one option changes; nothing a policy or an encoder sees does.
    /// </summary>
    public bool PreferInstance(CardModel instance)
    {
        if (!_initialised)
        {
            return false;
        }
        int[] taken = _context.OptionPicked;
        if (_members == null)
        {
            for (int i = 0; i < _options.Count; i++)
            {
                if (ReferenceEquals(_options[i], instance))
                {
                    return taken[i] == 0;
                }
            }
            return false;
        }
        for (int i = 0; i < _members.Count; i++)
        {
            List<CardModel> m = _members[i];
            int already = taken[i];
            for (int k = already; k < m.Count; k++)
            {
                if (ReferenceEquals(m[k], instance))
                {
                    if (k != already)
                    {
                        m[k] = m[already];
                        m[already] = instance;
                    }
                    return true;
                }
            }
        }
        return false;
    }

    /// <summary>HOST / TEST USE ONLY: the instance a pick of option <paramref name="ordinal"/> hands to the game when <paramref name="alreadyPicked"/> of its instances are taken.</summary>
    internal CardModel InstanceOf(int ordinal, int alreadyPicked) => _members != null ? _members[ordinal][alreadyPicked] : _options[ordinal];

    internal CardSelectFrame(HostCardSelector selector, int seat, SelectionPrompt prompt, List<CardModel> raw, int rawMin, int rawMax, VirtualClient? remoteClient, uint remoteChoiceId)
    {
        _selector = selector;
        _seat = seat;
        Prompt = prompt;
        _raw = raw;
        _rawMin = rawMin;
        _rawMax = rawMax;
        _remoteClient = remoteClient;
        _remoteChoiceId = remoteChoiceId;
        CardSelectStats.Frames++;
        CardSelectStats.FramesByKind[(int)prompt.Kind]++;
        CardSelectStats.FramesByContext[(int)prompt.ContextKind]++;
    }

    public string Name => "combat:card-select";

    /// <summary>True for prompts whose candidates come out of a pile whose instances are interchangeable for the player.</summary>
    public static bool MergesCandidates(SelectionPrompt p)
    {
        switch (p.Kind)
        {
            case CardSelectKind.CombatPile:
            case CardSelectKind.DeckUpgrade:
            case CardSelectKind.DeckTransform:
            case CardSelectKind.DeckEnchant:
            case CardSelectKind.DeckRemoval:
            case CardSelectKind.DeckGeneric:
                return true;
            default:
                return false;
        }
    }

    private void Initialise(RunContext ctx)
    {
        _initialised = true;
        SelectionPrompt p = Prompt;
        int n = _raw.Count;

        // ---- candidates ----
        if (MergesCandidates(p))
        {
            bool unpowered = p.HasPrefs && p.Prefs.UnpoweredPreviews;
            CanonicalCandidates.Build(ctx, _raw, p.PileType ?? PileType.None, _seat, unpowered, out _options, out _counts, out _members);
            if (_counts != null)
            {
                CardSelectStats.FramesMerged++;
                CardSelectStats.ClassesMerged += n - _options.Count;
            }
        }
        else
        {
            _options = _raw;
            _counts = null;
            _members = null;
            if ((p.Kind == CardSelectKind.SimpleGrid || p.Kind == CardSelectKind.SimpleGridRewards) && p.HasPrefs && p.Prefs.Comparison != null && n > 1)
            {
                // The simple grid screen displays the cards sorted with prefs.Comparison when there is one
                // (NSimpleCardSelectScreen.Create :81-85 / :96-100: List.Sort on a copy of the list the selector
                // receives). The public order is the displayed one, so the same call is made on a copy here;
                // List.Sort is deterministic for a given input order, so ties come out as on the real screen.
                List<CardModel> shown = new List<CardModel>(_raw);
                shown.Sort(p.Prefs.Comparison);
                _options = shown;
                _context.SortedByComparison = true;
            }
        }

        // ---- effective bounds: what the real screen of this entry point lets the player finish with ----
        SelectionRules rules = SelectionRules.Compute(p, n);
        if (rules.OnlyCancel)
        {
            CardSelectStats.OnlyCancel++;
        }
        if (rules.SoftlockAvoided)
        {
            // Never silent: the real screen would soft-lock here (see SelectionRules). Not observed in any test or fuzz run.
            CardSelectStats.SoftlockAvoided++;
            SpireHost.Boot.HostLog.Error($"combat:card-select: {p} has {n} candidates, fewer than the real screen needs and it cannot be closed; taking {rules.EffMax} (driver-spec-combat.md 4.4-2).");
        }
        _cancelable = rules.Cancelable;
        _effMin = rules.EffMin;
        _effMax = rules.EffMax;
        _needConfirm = rules.NeedConfirm;
        _zeroAllowed = rules.ZeroAllowed;
        int min = rules.ContextMin;
        int max = rules.ContextMax;

        // ---- context for the encoder ----
        _context.Kind = p.Kind;
        _context.Player = p.Player;
        _context.Options = _options;
        _context.OptionCounts = _counts;
        _context.MinSelect = min;
        _context.MaxSelect = max;
        _context.Cancelable = _cancelable;
        _context.CanSkip = _zeroAllowed;
        _context.RequireManualConfirmation = _needConfirm;
        _context.Prefs = p.HasPrefs ? p.Prefs : null;
        _context.SourceModel = p.SourceModel;
        _context.Pile = p.PileType;

        // ---- contract C2: prompt identity, raw bounds, picks per option ----
        _context.Entry = p.Entry;
        _context.ContextKind = p.ContextKind;
        _context.RawMin = _rawMin;
        _context.RawMax = _rawMax;
        _context.InstanceCount = n;
        _context.Merged = _counts != null;
        _context.OptionPicked = new int[_options.Count];
        _context.Enchantment = p.Enchantment;
        _context.EnchantmentAmount = p.EnchantmentAmount;
        if (p.HasPrefs && p.Prefs.Prompt != null)
        {
            _context.PromptTable = p.Prefs.Prompt.LocTable ?? "";
            _context.PromptKey = p.Prefs.Prompt.LocEntryKey ?? "";
        }
        else if (p.Kind == CardSelectKind.ChooseACard)
        {
            _context.PromptTable = "gameplay_ui";         // NChooseACardSelectionScreen.cs:164
            _context.PromptKey = "CHOOSE_CARD_HEADER";
        }
        else if (p.Kind == CardSelectKind.HandUpgrade)
        {
            _context.PromptTable = "gameplay_ui";         // CardSelectCmd.cs:641
            _context.PromptKey = "CHOOSE_CARD_UPGRADE_HEADER";
        }
    }

    public Decision? Build(RunContext ctx)
    {
        if (!_initialised)
        {
            Initialise(ctx);
        }
        if (ctx.SeatCount > 1 && (Prompt.Kind is CardSelectKind.Hand or CardSelectKind.HandDiscard or CardSelectKind.HandUpgrade) && RevalidateHand(ctx))
        {
            return null; // the screen closed itself (the frame is popped, the game has its answer)
        }
        bool inCombat = ctx.CombatManager.IsInProgress;
        _context.InCombat = inCombat;
        if (inCombat)
        {
            MegaCrit.Sts2.Core.Combat.CombatState? state = ctx.CombatManager.DebugOnlyGetState();
            PlayerCombatState? pcs = Prompt.Player.PlayerCombatState;
            _context.DuringEnemyTurn = state != null && state.CurrentSide != MegaCrit.Sts2.Core.Combat.CombatSide.Player;
            _context.OutsidePlayPhase = _context.DuringEnemyTurn || pcs == null || pcs.Phase != MegaCrit.Sts2.Core.Combat.PlayerTurnPhase.Play;
        }
        else
        {
            _context.DuringEnemyTurn = false;
            _context.OutsidePlayPhase = false;
        }

        int seat = _seat;
        int k = _context.Picked.Count;
        Decision d = _decision.Reset(DecisionKind.CardSelect, this, actingSeat: seat, subKind: (int)_context.Kind, context: _context);
        if (k < _effMax)
        {
            int[] taken = _context.OptionPicked;
            for (int i = 0; i < _options.Count; i++)
            {
                int size = _counts != null ? _counts[i] : 1;
                if (taken[i] < size)
                {
                    d.Add(Verb.SelectCard, seat, GameRef.OfCard(_options[i], i));
                }
            }
        }
        if (k == 0)
        {
            if (_zeroAllowed)
            {
                d.Add(Verb.Skip, seat);
            }
        }
        else if (k >= _effMin && _needConfirm)
        {
            d.Add(Verb.ConfirmSelection, seat);
        }
        if (_cancelable)
        {
            d.Add(Verb.Cancel, seat);
        }
        return d;
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        List<CardModel> picked = _context.Picked;
        switch (a.Verb)
        {
            case Verb.SelectCard:
            {
                int ordinal = a.Source.Index;
                if ((uint)ordinal >= (uint)_options.Count)
                {
                    ctx.Fail(AbortReason.HostBug, "card_select_bad_ordinal", "SELECT_CARD ordinal " + ordinal + " of " + _options.Count);
                }
                int already = _context.OptionPicked[ordinal];
                CardModel instance = _members != null ? _members[ordinal][already] : _options[ordinal];
                _context.OptionPicked[ordinal] = already + 1;
                _context.PickedOrdinals.Add(ordinal);
                picked.Add(instance);
                CardSelectStats.Picks++;
                if (picked.Count >= _effMax && !_needConfirm)
                {
                    CardSelectStats.AutoCompleted++;
                    Complete(ctx, picked, empty: false);
                }
                return; // otherwise: same prompt, next pick
            }
            case Verb.ConfirmSelection:
                Complete(ctx, picked, empty: false);
                return;
            case Verb.Skip:
            case Verb.Cancel:
                Complete(ctx, picked, empty: true);
                return;
            default:
                ctx.Fail(AbortReason.HostBug, "card_select_bad_verb", "Unexpected verb " + a.Verb + " on a CARD_SELECT frame.");
                return;
        }
    }

    /// <summary>
    /// MULTI-SEAT (phase H5). The hand screen tracks the hand while it is open: on every combat state change
    /// NPlayerHand.RevalidateSelectionAfterStateChange (Nodes.Combat/NPlayerHand.cs:963-995) drops what no longer passes
    /// the prompt's filter, and when nothing is selected and no card in hand is selectable any more it completes the
    /// selection with an EMPTY list by itself (:988-991) - whatever the prompt's minimum is. With one seat the hand
    /// cannot change while a prompt is open (the prompt pauses the only player's actions). With several seats it can:
    /// the hook prompts of different players are open at the same time (e.g. two end-of-turn prompts), and answering one
    /// can end the combat or change another player's hand. Called when the frame is about to be asked:
    ///   no selectable card in the owner's hand and nothing picked  -> the screen closes itself: the answer is the empty
    ///       selection, sent as the local branch sends it (the client's PlayerChoiceMessage for a remote seat). True.
    ///   the selectable cards are no longer the ones the frame holds -> not reproduced yet: the episode fails
    ///       (HandlerFault/card_select_hand_changed) instead of offering cards the real screen would not show.
    /// </summary>
    private bool RevalidateHand(RunContext ctx)
    {
        SelectionPrompt p = Prompt;
        bool handExists = ctx.CombatManager.IsInProgress && p.Player.PlayerCombatState != null;
        List<CardModel>? live = handExists ? RemoteCandidates.Rebuild(p) : null;
        if ((live == null || live.Count == 0) && _context.Picked.Count == 0)
        {
            CardSelectStats.RevalidatedEmpty++;
            CardSelectStats.Completed++;
            ctx.Pop(this);
            _selector.Completed(this);
            if (_remoteClient != null)
            {
                CardSelectStats.RemoteAnswers++;
                _remoteClient.SendChoice(_remoteChoiceId, RemoteCandidates.Encode(p, _raw, Array.Empty<CardModel>()));
            }
            else
            {
                Completion.SetResult(Array.Empty<CardModel>());
            }
            return true;
        }
        if (live == null || RemoteCandidates.FirstDifference(live, _raw) >= 0)
        {
            ctx.Fail(AbortReason.HandlerFault, "card_select_hand_changed",
                "The hand of seat " + _seat + " changed while its hand prompt was open (" + (live?.Count ?? 0) + " selectable cards now, " + _raw.Count + " when the prompt opened, " + _context.Picked.Count + " picked): " + p);
        }
        return false;
    }

    /// <summary>Host-side validation (the engine uses whatever the selector returns), then the answer goes to the game.</summary>
    private void Complete(RunContext ctx, List<CardModel> picked, bool empty)
    {
        CardModel[] result;
        if (empty)
        {
            if (!_zeroAllowed && !_cancelable)
            {
                ctx.Fail(AbortReason.HostBug, "card_select_invalid_result", "Empty selection on a prompt that can be neither skipped nor cancelled: " + Prompt);
            }
            result = Array.Empty<CardModel>();
            CardSelectStats.CompletedEmpty++;
        }
        else
        {
            int k = picked.Count;
            if (k < _effMin || k > _effMax || k == 0)
            {
                ctx.Fail(AbortReason.HostBug, "card_select_invalid_result", $"Selection of {k} cards outside [{_effMin}, {_effMax}]: " + Prompt);
            }
            for (int i = 0; i < k; i++)
            {
                CardModel c = picked[i];
                if (!HostCardSelector.Contains(_raw, c))
                {
                    ctx.Fail(AbortReason.HostBug, "card_select_invalid_result", "Picked card is not a candidate: " + Prompt);
                }
                for (int j = 0; j < i; j++)
                {
                    if (ReferenceEquals(picked[j], c))
                    {
                        ctx.Fail(AbortReason.HostBug, "card_select_invalid_result", "Card picked twice: " + Prompt);
                    }
                }
            }
            string? stale = StillAvailable(picked);
            if (stale != null)
            {
                ctx.Fail(AbortReason.HandlerFault, "card_select_stale_option", stale + ": " + Prompt);
            }
            result = picked.ToArray(); // pick order = the order handed to the game
        }
        CardSelectStats.Completed++;
        ctx.Pop(this);
        _selector.Completed(this);
        if (_remoteClient != null)
        {
            // The seat's client sends its answer exactly as the local branch of the entry point encodes it; the host's
            // WaitForRemoteChoice completes on it and the game resumes inline. (The client's resume request for the paused
            // action follows from the pump: mp-remote-resume-*.)
            CardSelectStats.RemoteAnswers++;
            _remoteClient.SendChoice(_remoteChoiceId, RemoteCandidates.Encode(Prompt, _raw, result));
            return;
        }
        // The answer encoding of remote seats is checked against the game on every local answer.
        string? garbled = RemoteCandidates.CheckEncoding(ctx, Prompt, _raw, result);
        if (garbled != null)
        {
            ctx.Fail(AbortReason.HandlerFault, "card_select_encoding_mismatch", garbled + ": " + Prompt);
        }
        Completion.SetResult(result); // the game resumes inline; the driver core settles afterwards
    }

    /// <summary>The real pile / hand screens track their source while they are open; a picked card must still be where it was offered.</summary>
    private string? StillAvailable(List<CardModel> picked)
    {
        SelectionPrompt p = Prompt;
        IReadOnlyList<CardModel>? source = null;
        switch (p.Kind)
        {
            case CardSelectKind.Hand:
            case CardSelectKind.HandDiscard:
            case CardSelectKind.HandUpgrade:
                source = p.Player.PlayerCombatState?.Hand.Cards;
                if (source == null)
                {
                    return "the combat ended while a hand prompt was open";
                }
                break;
            case CardSelectKind.CombatPile:
                source = p.Pile?.Cards;
                break;
            case CardSelectKind.DeckUpgrade:
            case CardSelectKind.DeckTransform:
            case CardSelectKind.DeckEnchant:
            case CardSelectKind.DeckRemoval:
            case CardSelectKind.DeckGeneric:
                source = p.Player.Deck.Cards;
                break;
        }
        if (source == null)
        {
            return null;
        }
        for (int i = 0; i < picked.Count; i++)
        {
            if (!HostCardSelector.Contains(source, picked[i]))
            {
                return "a picked card left its pile while the prompt was open";
            }
        }
        return null;
    }
}

/// <summary>
/// The effective selection rules of one prompt: what the real screen behind the CardSelectCmd entry point lets
/// the player finish with (the table in the comment of <see cref="CardSelectFrame"/>; the engine itself
/// validates nothing). A pure function of the captured prompt and the number of candidate INSTANCES, so that it
/// can be evaluated without a frame: the .mcr cross-check applies it to the prompts of the real client's
/// recordings and checks that every answer the real screens produced is one these rules allow.
/// </summary>
public struct SelectionRules
{
    /// <summary>Final counts k > 0 the screen can complete with: EffMin..EffMax. EffMin > EffMax = none (the prompt can only be cancelled).</summary>
    public int EffMin;
    public int EffMax;

    /// <summary>A CONFIRM action is needed to finish (manual confirmation, or a real range); otherwise the pick that reaches EffMax completes the prompt.</summary>
    public bool NeedConfirm;

    /// <summary>The close button of a deck screen exists (prefs.Cancelable): CANCEL = empty result.</summary>
    public bool Cancelable;

    /// <summary>Finishing with nothing picked is allowed without a close button: SKIP (skip button of ChooseACard, or a lower bound of 0).</summary>
    public bool ZeroAllowed;

    /// <summary>The bounds published in CardSelectContext.MinSelect / MaxSelect.</summary>
    public int ContextMin;
    public int ContextMax;

    /// <summary>The real screen could not be completed (fewer candidates than the required count) but can be closed: CANCEL is the only action.</summary>
    public bool OnlyCancel;

    /// <summary>The real screen could neither be completed nor closed (it would soft-lock): the frame takes as many cards as there are (driver-spec-combat.md 4.4-2, "defensively min(Min, n)").</summary>
    public bool SoftlockAvoided;

    /// <summary>Is "the prompt ends with k cards" (0 = skip / cancel) an outcome these rules can produce?</summary>
    public readonly bool Allows(int k) => k == 0 ? ZeroAllowed || Cancelable : k >= EffMin && k <= EffMax;

    public static bool IsDeckKind(CardSelectKind kind)
    {
        return kind == CardSelectKind.DeckUpgrade || kind == CardSelectKind.DeckTransform || kind == CardSelectKind.DeckEnchant
            || kind == CardSelectKind.DeckRemoval || kind == CardSelectKind.DeckGeneric;
    }

    public static SelectionRules Compute(SelectionPrompt p, int n)
    {
        // The interval [lo, hi] of final counts the real screen can complete with.
        bool manual;
        int lo;
        int hi;
        int prefMin = p.HasPrefs ? Math.Max(0, p.Prefs.MinSelect) : 0;
        int prefMax = p.HasPrefs ? p.Prefs.MaxSelect : 0;
        switch (p.Kind)
        {
            case CardSelectKind.ChooseACard:
                lo = 1; // the (0, 1) the game passes is not the rule: no skip button unless canSkip
                hi = 1;
                manual = false;
                break;
            case CardSelectKind.HandUpgrade:
                lo = 1;
                hi = 1;
                manual = false;
                break;
            case CardSelectKind.CombatPile:
                lo = Math.Min(prefMin, n);   // NCombatPileCardSelectScreen.UpdateConfirmButton / CheckIfSelectionComplete
                hi = Math.Min(prefMax, n);
                manual = p.Prefs.RequireManualConfirmation;
                break;
            case CardSelectKind.DeckUpgrade:
                lo = prefMax;                // completes only at MaxSelect (NDeckUpgradeSelectScreen.cs:306)
                hi = prefMax;
                manual = p.Prefs.RequireManualConfirmation;
                break;
            case CardSelectKind.DeckEnchant:
            case CardSelectKind.DeckTransform:
            case CardSelectKind.DeckRemoval:
            case CardSelectKind.DeckGeneric:
                // Preview opens at MaxSelect; the confirm button exists only when MinSelect != MaxSelect.
                lo = prefMin;
                hi = prefMin != prefMax ? Math.Min(prefMax, n) : prefMax;
                if (p.Kind == CardSelectKind.DeckEnchant && lo < 1)
                {
                    lo = 1;                  // NDeckEnchantSelectScreen.ConfirmSelection: never with zero cards
                }
                manual = p.Prefs.RequireManualConfirmation;
                break;
            default:                          // Hand, HandDiscard, SimpleGrid, SimpleGridRewards
                lo = prefMin;
                hi = Math.Min(prefMax, n);
                manual = p.HasPrefs && p.Prefs.RequireManualConfirmation;
                break;
        }
        SelectionRules r = default;
        r.Cancelable = p.HasPrefs && p.Prefs.Cancelable && IsDeckKind(p.Kind);
        bool completable = lo <= hi && hi <= n;
        if (!completable)
        {
            // The real screen cannot be completed here (fewer candidates than the required count).
            // Cancelable: the close button is the only exit. Otherwise the real game soft-locks; the frame takes
            // as many cards as there are - the game's own rule when no confirmation is required
            // (CardSelectCmd: "count <= MinSelect -> all") - and counts it.
            if (r.Cancelable)
            {
                r.OnlyCancel = true;
                lo = 1;
                hi = 0;
            }
            else
            {
                r.SoftlockAvoided = true;
                hi = Math.Min(Math.Max(prefMax, 1), n);
                lo = hi;
            }
        }
        r.EffMin = lo;
        r.EffMax = hi;
        bool fixedCount = p.Kind == CardSelectKind.ChooseACard || p.Kind == CardSelectKind.HandUpgrade;
        r.NeedConfirm = !fixedCount && (manual || lo != hi);
        r.ZeroAllowed = (lo == 0 || (p.Kind == CardSelectKind.ChooseACard && p.CanSkip)) && !r.Cancelable;
        r.ContextMin = Math.Max(0, Math.Min(lo, hi));
        r.ContextMax = Math.Max(0, hi);
        if (p.Kind == CardSelectKind.ChooseACard && p.CanSkip)
        {
            r.ContextMin = 0;
        }
        return r;
    }
}

/// <summary>
/// Classes of identical cards in canonical order. Input: candidates in the order the game handed them over.
/// Output: one representative per class (its first instance), the class sizes (null when every class has
/// one member) and the members of each class in input order. The class ORDER depends only on the keys.
///
/// Whose key: the registered <see cref="CanonicalCardClasses"/> provider of the driver (RunContext.CanonicalCardClasses;
/// contract C7 = obs O1). The observation encoder needs the classes in ITS content-key order (obs-schema-v1 5.2.3:
/// the encoded row + its printed values; the deduplicated CHOICE rows must be strictly ascending by that key, which
/// only the encoder can compute - it contains vocabulary ids). In production the provider is registered in every
/// mode; a pile-sourced prompt without one FAILS the episode (HandlerFault/card_select_no_class_provider) - there is
/// no silent fallback to another order. The local <see cref="CardClassKey"/> remains for tests (the overload without
/// a context) and as the hint that decides which class a pair query tries first.
/// </summary>
public static class CanonicalCandidates
{
    private sealed class Group
    {
        public string Key = "";
        public List<CardModel> Members = new List<CardModel>(2);
    }

    /// <summary>Calls of the provider on a whole candidate list / on a pair (membership test).</summary>
    public static long ProviderCalls;
    public static long ProviderPairCalls;

    public static void Build(RunContext ctx, IReadOnlyList<CardModel> raw, PileType pile, int seat, bool unpoweredPreviews, out List<CardModel> options, out List<int>? counts, out List<List<CardModel>>? members)
    {
        if (raw.Count == 0)
        {
            Build(raw, out options, out counts, out members);
            return;
        }
        CanonicalCardClasses? provider = ctx.CanonicalCardClasses;
        if (provider == null)
        {
            ctx.Fail(AbortReason.HandlerFault, "card_select_no_class_provider",
                "A pile-sourced CARD_SELECT prompt needs the canonical card-class provider (DriverRegistry.SetCanonicalCardClasses), and none is registered.");
        }
        BuildWithProvider(ctx, provider, raw, pile, seat, unpoweredPreviews, out options, out counts, out members);
    }

    /// <summary>The local key (<see cref="CardClassKey"/>): tests only, and the empty list.</summary>
    public static void Build(IReadOnlyList<CardModel> raw, out List<CardModel> options, out List<int>? counts, out List<List<CardModel>>? members)
    {
        List<Group> groups = new List<Group>(raw.Count);
        Dictionary<string, Group> byKey = new Dictionary<string, Group>(raw.Count, StringComparer.Ordinal);
        for (int i = 0; i < raw.Count; i++)
        {
            string key = CardClassKey.Of(raw[i]);
            if (!byKey.TryGetValue(key, out Group? g))
            {
                g = new Group { Key = key };
                byKey[key] = g;
                groups.Add(g);
            }
            g.Members.Add(raw[i]);
        }
        groups.Sort(static (a, b) => string.CompareOrdinal(a.Key, b.Key)); // keys are unique: the order is total
        options = new List<CardModel>(groups.Count);
        members = new List<List<CardModel>>(groups.Count);
        List<int> sizes = new List<int>(groups.Count);
        bool anyMerged = false;
        for (int i = 0; i < groups.Count; i++)
        {
            options.Add(groups[i].Members[0]);
            members.Add(groups[i].Members);
            sizes.Add(groups[i].Members.Count);
            anyMerged |= groups[i].Members.Count > 1;
        }
        counts = anyMerged ? sizes : null;
    }

    /// <summary>
    /// Classes and class order from the provider. The provider returns representatives and sizes only; which
    /// instance belongs to which class is found with the provider itself: two cards are in one class exactly when
    /// the provider, asked about the two of them, answers "one class of two". (The local key only decides which
    /// class is tried first.) Everything the provider says is checked; an inconsistent answer aborts the episode
    /// (HandlerFault/card_select_class_provider) instead of producing a frame whose options and instances disagree.
    /// </summary>
    private static void BuildWithProvider(RunContext ctx, CanonicalCardClasses provider, IReadOnlyList<CardModel> raw, PileType pile, int seat, bool unpowered,
        out List<CardModel> options, out List<int>? counts, out List<List<CardModel>>? members)
    {
        int n = raw.Count;
        List<CardModel> representatives = new List<CardModel>(n);
        List<int> sizes = new List<int>(n);
        ProviderCalls++;
        provider(ctx, raw, pile, seat, unpowered, representatives, sizes);
        int k = representatives.Count;
        if (k == 0 || sizes.Count != k)
        {
            ctx.Fail(AbortReason.HandlerFault, "card_select_class_provider", $"the class provider returned {k} classes and {sizes.Count} sizes for {n} candidates");
        }
        int total = 0;
        for (int j = 0; j < k; j++)
        {
            if (sizes[j] < 1)
            {
                ctx.Fail(AbortReason.HandlerFault, "card_select_class_provider", $"the class provider returned a class of size {sizes[j]}");
            }
            total += sizes[j];
        }
        if (total != n)
        {
            ctx.Fail(AbortReason.HandlerFault, "card_select_class_provider", $"the class sizes of the provider sum to {total} for {n} candidates");
        }
        int[] classOf = new int[n];
        Array.Fill(classOf, -1);
        int[] assigned = new int[k];
        for (int j = 0; j < k; j++)
        {
            int at = -1;
            for (int i = 0; i < n; i++)
            {
                if (ReferenceEquals(raw[i], representatives[j]))
                {
                    at = i;
                    break;
                }
            }
            if (at < 0 || classOf[at] >= 0)
            {
                ctx.Fail(AbortReason.HandlerFault, "card_select_class_provider", at < 0 ? "a class representative of the provider is not one of the candidates" : "the provider returned one card as the representative of two classes");
            }
            classOf[at] = j;
            assigned[j] = 1;
        }
        if (k < n)
        {
            string[] representativeKey = new string[k];
            for (int j = 0; j < k; j++)
            {
                representativeKey[j] = CardClassKey.Of(representatives[j]);
            }
            List<CardModel> pair = new List<CardModel>(2);
            List<CardModel> pairClasses = new List<CardModel>(2);
            List<int> pairCounts = new List<int>(2);
            for (int i = 0; i < n; i++)
            {
                if (classOf[i] >= 0)
                {
                    continue;
                }
                string key = CardClassKey.Of(raw[i]);
                int found = -1;
                for (int pass = 0; pass < 2 && found < 0; pass++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        if (assigned[j] >= sizes[j] || (representativeKey[j] == key) != (pass == 0))
                        {
                            continue;
                        }
                        pair.Clear();
                        pair.Add(representatives[j]);
                        pair.Add(raw[i]);
                        pairClasses.Clear();
                        pairCounts.Clear();
                        ProviderPairCalls++;
                        provider(ctx, pair, pile, seat, unpowered, pairClasses, pairCounts);
                        bool same = pairClasses.Count == 1 && pairCounts.Count == 1 && pairCounts[0] == 2;
                        bool different = pairClasses.Count == 2 && pairCounts.Count == 2 && pairCounts[0] == 1 && pairCounts[1] == 1;
                        if (same == different)
                        {
                            ctx.Fail(AbortReason.HandlerFault, "card_select_class_provider", $"the class provider answered {pairClasses.Count} classes for a pair of cards");
                        }
                        if (same)
                        {
                            found = j;
                            break;
                        }
                    }
                }
                if (found < 0)
                {
                    ctx.Fail(AbortReason.HandlerFault, "card_select_class_provider", "a candidate belongs to none of the provider's classes that still has room: " + raw[i].Id.Entry);
                }
                classOf[i] = found;
                assigned[found]++;
            }
        }
        members = new List<List<CardModel>>(k);
        for (int j = 0; j < k; j++)
        {
            if (assigned[j] != sizes[j])
            {
                ctx.Fail(AbortReason.HandlerFault, "card_select_class_provider", $"class {j}: {assigned[j]} instances found, the provider said {sizes[j]}");
            }
            members.Add(new List<CardModel>(sizes[j]));
        }
        for (int i = 0; i < n; i++)
        {
            members[classOf[i]].Add(raw[i]); // input order inside a class
        }
        options = new List<CardModel>(k);
        bool anyMerged = false;
        for (int j = 0; j < k; j++)
        {
            options.Add(members[j][0]);
            anyMerged |= sizes[j] > 1;
        }
        counts = anyMerged ? sizes : null;
    }

    /// <summary>The class keys in canonical order with their sizes (tests: must not depend on the pile order).</summary>
    public static List<string> Describe(IReadOnlyList<CardModel> raw)
    {
        Build(raw, out List<CardModel> options, out List<int>? counts, out _);
        List<string> r = new List<string>(options.Count);
        for (int i = 0; i < options.Count; i++)
        {
            r.Add((counts != null ? counts[i] : 1) + "x " + CardClassKey.Of(options[i]));
        }
        return r;
    }
}
