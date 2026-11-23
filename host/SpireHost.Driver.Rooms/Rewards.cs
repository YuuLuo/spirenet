using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Rewards;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

// =========================================================================================================
//  REWARD / CARD_REWARD
//
//  Game side (TestMode branch of RewardsSet.Offer, Rewards/RewardsSet.cs:123-166): GenerateWithoutOffering ->
//  RewardsSetSynchronizer.BeginRewardsSet (pushes the set on the synchronizer's per-player stack) -> for the local
//  player `await testSelector(this)` -> throw if the set is not complete and ThrowInTestIfRewardsNotTaken ->
//  `await task` (the set's completion). The real game shows NRewardsScreen instead (:162) and only awaits `task`.
//
//  Host side = what NRewardsScreen / NRewardButton / NCombatUi / NCardRewardSelectionScreen do:
//    * testSelector parks a RewardFrame on the decision stack (nested sets nest as frames);
//    * TAKE(i) = RewardsSetSynchronizer.SelectLocalReward(reward) (NRewardButton.cs:184-215), any order;
//    * PROCEED  = NRewardsScreen.OnProceedButtonPressed (:421-470) incl. the enable rule TryEnableProceedButton
//      (:494-504: not while DisallowSkipping and rewards are left; Hook.ShouldProceedToNextMapPoint);
//    * a fully taken non-terminal set closes itself (UpdateScreenState :366-391);
//    * CombatManager.CombatWon -> NCombatUi.OnCombatWon (:257-304);
//    * the card reward screen = a selector pushed on the per-player stack for that one reward
//      (CardReward.OnSelect reads only that stack, Rewards/CardReward.cs:185); GetSelectedCardReward is a
//      SYNCHRONOUS callback -> nested decision.
// =========================================================================================================

internal sealed class RewardsLifecycle : IRunLifecycle
{
    private enum Expect
    {
        None,
        SecondBoss,     // after "first of two bosses": the next room is the second boss point
        NextAct,        // after SetLocalPlayerReady (not last act): the next room is in act index + 1
        Architect,      // after SetLocalPlayerReady in the last act: the next room is TheArchitect (victory room)
    }

    private RunContext? _ctx;
    private Action<CombatRoom>? _combatWon;
    private readonly Func<RewardsSet, Task> _selector;
    private Expect _expect;
    private int _expectAct;
    private MapCoord _expectCoord;

    /// <summary>Reward frames currently on the decision stack (nesting depth).</summary>
    internal int Depth;

    public RewardsLifecycle()
    {
        _selector = OnRewardsOffered;
    }

    public string Name => RoomsNames.RewardsLifecycle;

    public void OnRunSetUp(RunContext ctx)
    {
        _ctx = ctx;
        Depth = 0;
        _expect = Expect.None;
        if (RewardsSet.testSelector != null)
        {
            throw new InvalidOperationException("RewardsSet.testSelector is already set (left over from a previous episode, or another module installs it).");
        }
        // RewardsSet.cs:140-147: the only way the TestMode branch lets a player choose rewards (DRIVER-API.md 7.8: static, removed in OnEpisodeEnd).
        RewardsSet.testSelector = _selector;
        _combatWon ??= OnCombatWon;
        ctx.CombatManager.CombatWon += _combatWon;
    }

    public void OnRoomEntered(RunContext ctx)
    {
        if (_expect == Expect.None)
        {
            return;
        }
        RunState run = ctx.Run;
        switch (_expect)
        {
            case Expect.SecondBoss:
                RoomsProbe.TreeCheck(run.CurrentMapCoord.HasValue && run.CurrentMapCoord.Value == _expectCoord && run.CurrentRoom?.RoomType == RoomType.Boss,
                    $"first-of-two-bosses: next room {run.CurrentRoom?.RoomType} at {run.CurrentMapCoord}, expected the second boss at {_expectCoord}");
                break;
            case Expect.NextAct:
                RoomsProbe.TreeCheck(run.CurrentActIndex == _expectAct + 1,
                    $"act change: next room in act {run.CurrentActIndex}, expected {_expectAct + 1}");
                break;
            case Expect.Architect:
                RoomsProbe.TreeCheck(run.CurrentRoom is EventRoom er && er.CanonicalEvent is TheArchitect && run.CurrentRoom.IsVictoryRoom,
                    $"last act boss: next room {RoomDescription(run.CurrentRoom)}, expected TheArchitect");
                break;
        }
        _expect = Expect.None;
    }

    public void OnEpisodeEnd(RunContext ctx)
    {
        if (ReferenceEquals(RewardsSet.testSelector, _selector))
        {
            RewardsSet.testSelector = null;
        }
        if (_combatWon != null)
        {
            ctx.CombatManager.CombatWon -= _combatWon;
        }
        _ctx = null;
        Depth = 0;
        _expect = Expect.None;
    }

    private static string RoomDescription(AbstractRoom? room) => room == null ? "null" : room.GetType().Name + ":" + room.RoomType;

    /// <summary>RewardsSet.testSelector: called by RewardsSet.Offer for the local player's sets (RewardsSet.cs:140-147); the game awaits the task.</summary>
    private Task OnRewardsOffered(RewardsSet set)
    {
        RunContext ctx = _ctx ?? throw new InvalidOperationException("A reward set was offered outside an episode.");
        // NRewardsScreen.ShowScreen(this, isTerminal: Room is CombatRoom, ...) (RewardsSet.cs:130, :162).
        bool terminal = set.Room is CombatRoom;
        RewardFrame frame = new RewardFrame(this, set, terminal);
        ctx.Push(frame);
        Depth++;
        RoomsProbe.RewardSetsOffered++;
        if (terminal)
        {
            RoomsProbe.TerminalSetsOffered++;
        }
        if (Depth > RoomsProbe.RewardFrameMaxDepth)
        {
            RoomsProbe.RewardFrameMaxDepth = Depth;
        }
        return frame.Done.Task;
    }

    /// <summary>
    /// Host stand-in for NCombatUi.OnCombatWon (Nodes.Combat/NCombatUi.cs:257-304). The UI reacts on a later frame
    /// (after the death animations; ShowRewards / ProceedWithoutRewards wait first), so the work is posted and
    /// waits for the engine to be quiescent (CombatManager.EndCombatInternal has finished).
    /// </summary>
    private void OnCombatWon(CombatRoom room)
    {
        RunContext? ctx = _ctx;
        if (ctx == null)
        {
            return;
        }
        RoomsProbe.CombatsWon++; // (EpisodeCounters.CombatsWon and the other public-history counters are maintained by the core)
        ctx.Post(async () =>
        {
            await ctx.Quiescence();
            if (room.Encounter.ShouldGiveRewards)
            {
                // NCombatUi.ShowRewards (:274-304) -> CombatRoom.OfferRoomEndRewards (Rooms/CombatRoom.cs:239-252).
                RoomsProbe.CombatWonWithRewards++;
                await room.OfferRoomEndRewards();
            }
            else
            {
                // NCombatUi.ProceedWithoutRewards (:268-273): ProceedFromTerminalRewardsScreenAsync, NOT the proceed tree.
                RoomsProbe.CombatWonWithoutRewards++;
                bool resumed = await RoomsFlow.ProceedFromTerminal(ctx, "combat without rewards");
                if (resumed)
                {
                    RoomsProbe.NoRewardsResume++;
                }
                else
                {
                    RoomsProbe.NoRewardsMap++;
                    // Multi-seat: every client's NCombatUi runs the same ProceedWithoutRewards for itself, without a click
                    // and without a message: each remote seat's map is open as well.
                    for (int s = 1; s < ctx.SeatCount; s++)
                    {
                        ctx.OpenMapFor(s, "combat without rewards (remote)");
                    }
                }
            }
        }, "after combat won: " + room.Encounter.Id.Entry);
    }

    /// <summary>
    /// NRewardsScreen.OnProceedButtonPressed for a TERMINAL screen (Nodes.Screens/NRewardsScreen.cs:421-464):
    ///   (debug) RunManager.debugAfterCombatRewardsOverride                    -> invoke it (:423-426)
    ///   flag : a second boss exists and the current point is the first boss  -> ProceedFromTerminalRewardsScreen (:429, :431-434)
    ///   flag2: (Boss room or victory room) and current point == (second boss ?? boss)
    ///                                                                         -> ActChangeSynchronizer.SetLocalPlayerReady (:430, :435-444)
    ///   else (skip or proceed; TestMode skips the FTUE)                       -> ProceedFromTerminalRewardsScreen (:445-463)
    /// ProceedFromTerminalRewardsScreen resumes the parent event of an event combat or opens the map (RoomsFlow).
    /// SetLocalPlayerReady -> VoteToMoveToNextActAction -> MoveToNextAct -> EnterNextAct: next act, or TheArchitect
    /// after the last act (ActChangeSynchronizer.cs:34-90, RunManager.cs:1162-1187).
    /// </summary>
    internal async Task TerminalProceed(RunContext ctx)
    {
        RunState run = ctx.Run;
        RunManager rm = ctx.RunManager;
        if (rm.debugAfterCombatRewardsOverride != null)
        {
            RoomsProbe.TreeDebugOverride++;
            rm.debugAfterCombatRewardsOverride.Invoke();
            return;
        }
        AbstractRoom room = run.CurrentRoom ?? throw new InvalidOperationException("Terminal proceed without a current room.");
        ActMap map = run.Map;
        MapPoint? current = run.CurrentMapPoint;
        bool firstOfTwoBosses = map.SecondBossMapPoint != null && current == map.BossMapPoint;
        bool lastBossOfAct = (room.RoomType == RoomType.Boss || room.IsVictoryRoom) && current == (map.SecondBossMapPoint ?? map.BossMapPoint);
        if (firstOfTwoBosses)
        {
            RoomsProbe.TreeFirstOfTwoBosses++;
            _expect = Expect.SecondBoss;
            _expectCoord = map.SecondBossMapPoint!.coord;
            await RoomsFlow.ProceedFromTerminal(ctx, "first boss of two");
        }
        else if (lastBossOfAct)
        {
            bool lastAct = run.CurrentActIndex >= run.Acts.Count - 1;
            if (lastAct)
            {
                RoomsProbe.TreeArchitect++;
                _expect = Expect.Architect;
            }
            else
            {
                RoomsProbe.TreeActChange++;
                _expect = Expect.NextAct;
                _expectAct = run.CurrentActIndex;
            }
            if (ctx.SeatCount > 1)
            {
                ctx.SetActReady(0); // the move waits for every seat's vote (ActChangeSynchronizer.cs:69)
            }
            rm.ActChangeSynchronizer.SetLocalPlayerReady();
        }
        else
        {
            bool resumed = await RoomsFlow.ProceedFromTerminal(ctx, "terminal rewards proceed");
            if (resumed)
            {
                RoomsProbe.TreeResumeEvent++;
            }
            else
            {
                RoomsProbe.TreeMap++;
            }
        }
    }
}

/// <summary>
/// One reward set waiting for the player (one NRewardsScreen). Pushed by testSelector, popped on PROCEED or when a
/// non-terminal set is fully taken.
/// </summary>
internal sealed class RewardFrame : IPendingDecision
{
    private readonly RewardsLifecycle _owner;
    private readonly RewardContext _context;
    private readonly Decision _decision = new Decision();
    private Task? _take;
    private string _takeWhat = "";
    private bool _finished;

    internal readonly RewardsSet Set;
    internal readonly bool Terminal;

    /// <summary>Completes the testSelector task: RewardsSet.Offer resumes (inline) when it is set.</summary>
    public readonly TaskCompletionSource Done = new TaskCompletionSource();

    public RewardFrame(RewardsLifecycle owner, RewardsSet set, bool terminal)
    {
        _owner = owner;
        Set = set;
        Terminal = terminal;
        _context = new RewardContext { Set = set, Terminal = terminal };
    }

    public string Name => RoomsNames.RewardFrame;

    public Decision? Build(RunContext ctx)
    {
        if (_take != null && !_take.IsCompleted)
        {
            // The frame is on top again but the reward it is taking has not finished and nothing else is asking:
            // the game is waiting for something the host does not know about.
            ctx.Stall("reward_take_incomplete", "Reward frame is on top while '" + _takeWhat + "' is still running. " + ctx.Diagnose());
        }
        RewardsSetSynchronizer rss = ctx.RunManager.RewardsSetSynchronizer;
        bool completed = rss.IsRewardsSetCompleted(Set);
        if (completed && !Set.AllRewardsSuccessfullySelected)
        {
            ctx.Fail(AbortReason.HandlerFault, "reward_set_skipped_externally", "Reward set " + Set + " was skipped by the game (room exit?) while its screen was still open. " + ctx.Diagnose());
        }
        if (completed && !Terminal)
        {
            // NRewardsScreen.UpdateScreenState (:366-391): every reward taken -> a non-terminal screen removes itself.
            RoomsProbe.RewardAutoClosed++;
            Finish(ctx);
            return null;
        }
        if (!completed)
        {
            AssertOnTop(ctx, rss);
        }
        Player player = Set.Player;
        RunState run = ctx.Run;
        int seat = ctx.SeatOf(player);
        Decision d = _decision.Reset(DecisionKind.Reward, this, actingSeat: seat, context: _context);
        List<Reward> rewards = Set.Rewards;
        bool anyUntaken = false;
        for (int i = 0; i < rewards.Count; i++)
        {
            Reward r = rewards[i];
            if (r.SuccessfullySelected)
            {
                continue; // NRewardsScreen.RewardCollectedFrom removed its button
            }
            anyUntaken = true;
            if (r is LinkedRewardSet || r.ParentRewardSet != null)
            {
                // v0.113: no code constructs a LinkedRewardSet (driver-spec-rooms.md 3.1); NLinkedRewardSet is not proxied.
                ctx.Fail(AbortReason.HandlerFault, "linked_reward_set", "A LinkedRewardSet appeared in " + Set + "; the rooms driver does not implement NLinkedRewardSet.");
            }
            if (!IsTakeEffective(r, player, run))
            {
                continue;
            }
            d.Add(Verb.ChooseOption, seat, GameRef.OfReward(r, i), flags: r is GoldReward ? ActionFlags.AutoGold : ActionFlags.None);
        }
        // TryEnableProceedButton (:494-504): the button is a "skip" while rewards are left; DisallowSkipping disables
        // the skip (not the final proceed); Hook.ShouldProceedToNextMapPoint gates both.
        bool skipDisallowed = Set.DisallowSkipping && anyUntaken;
        if (skipDisallowed)
        {
            RoomsProbe.ProceedDisallowed++;
        }
        if (!skipDisallowed && Hook.ShouldProceedToNextMapPoint(run))
        {
            d.Add(Verb.Proceed, seat);
        }
        return d;
    }

    /// <summary>
    /// A reward button the real UI lets the player click, minus the clicks that are guaranteed no-ops (the button
    /// just re-enables, NRewardButton.cs:184-215): a PotionReward with a full belt whose potion
    /// Hook.ShouldProcurePotion allows (TryToProcure -> TooFull -> OnSelect false, PotionReward.cs:76-93; with
    /// ShouldProcurePotion false it is "taken" without a potion and stays legal), a CardUpgradeReward without an
    /// upgradable card (FromDeckForUpgrade returns nothing, CardSelectCmd.cs:315-321) and a CardRemovalReward without a
    /// removable card (FromDeckForRemoval -> FromDeckGeneric returns nothing, CardSelectCmd.cs:476-498).
    /// </summary>
    internal static bool IsTakeEffective(Reward r, Player player, RunState run)
    {
        switch (r)
        {
            case PotionReward pr:
                if (!player.HasOpenPotionSlots && pr.Potion != null && Hook.ShouldProcurePotion(run, player.Creature.CombatState, pr.Potion, player))
                {
                    RoomsProbe.PotionRewardMasked++;
                    return false;
                }
                return true;
            case CardUpgradeReward:
                if (!AnyDeckCard(player, upgradable: true))
                {
                    RoomsProbe.NoOpRewardMasked++;
                    return false;
                }
                return true;
            case CardRemovalReward:
                if (!AnyDeckCard(player, upgradable: false))
                {
                    RoomsProbe.NoOpRewardMasked++;
                    return false;
                }
                return true;
            default:
                return true;
        }
    }

    internal static bool AnyDeckCard(Player player, bool upgradable)
    {
        IReadOnlyList<CardModel> deck = player.Deck.Cards;
        for (int i = 0; i < deck.Count; i++)
        {
            if (upgradable ? deck[i].IsUpgradable : deck[i].IsRemovable)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>SelectLocalReward / SkipLocalRewardsSet act on the synchronizer's stack top (RewardsSetSynchronizer.cs:291, :389-399); it must be this set.</summary>
    private void AssertOnTop(RunContext ctx, RewardsSetSynchronizer rss)
    {
        // Private state read through the publicized reference (driving only; DRIVER-API.md 2.2).
        var stack = rss._rewardStates[ctx.SeatOf(Set.Player)].rewardsStack;
        if (stack.Count == 0 || !ReferenceEquals(stack[stack.Count - 1].set, Set))
        {
            ctx.Fail(AbortReason.HandlerFault, "reward_set_not_on_top", "The reward frame's set " + Set + " is not on top of the RewardsSetSynchronizer stack (" + stack.Count + " sets). " + ctx.Diagnose());
        }
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        RewardsSetSynchronizer rss = ctx.RunManager.RewardsSetSynchronizer;
        if (a.Verb == Verb.ChooseOption)
        {
            Reward r = a.Source.Reward;
            if (a.Source.Index >= Set.Rewards.Count || !ReferenceEquals(Set.Rewards[a.Source.Index], r))
            {
                ctx.Fail(AbortReason.HandlerFault, "reward_moved", "Reward " + r + " is no longer at index " + a.Source.Index + ".");
            }
            RoomsProbe.RewardTakes++;
            _takeWhat = "take reward " + r.GetType().Name;
            _take = Take(ctx, rss, r);
            ctx.Track(_take, _takeWhat);
            return;
        }
        if (a.Verb != Verb.Proceed)
        {
            ctx.Fail(AbortReason.HandlerFault, "bad_verb", "Reward frame got " + a);
        }
        if (!Terminal)
        {
            // NRewardsScreen.cs:465-469: SkipLocalRewardsSet, then the screen is removed.
            if (!rss.IsRewardsSetCompleted(Set))
            {
                AssertOnTop(ctx, rss);
                rss.SkipLocalRewardsSet();
                RoomsProbe.NonTerminalSkips++;
            }
            Finish(ctx);
            return;
        }
        // Terminal: the UI does NOT skip the set; it stays on the synchronizer stack until the room is left
        // (RewardsSetSynchronizer.BeforeLeavingRoom, :446-466, called by RunManager.ExitCurrentRoom :1047-1052),
        // where OnSkipped runs. RewardsSet.Offer would throw in TestMode for an incomplete set (:155-158); the
        // game's own public switch turns that test-only assertion off so the set is left exactly as the UI leaves it.
        if (!rss.IsRewardsSetCompleted(Set))
        {
            Set.ThrowInTestIfRewardsNotTaken = false;
            RoomsProbe.TerminalLeftForRoomExit++;
        }
        Finish(ctx);
        ctx.Track(_owner.TerminalProceed(ctx), "terminal proceed");
    }

    private void Finish(RunContext ctx)
    {
        if (_finished)
        {
            return;
        }
        _finished = true;
        ctx.Pop(this);
        _owner.Depth--;
        Done.SetResult(); // RewardsSet.Offer resumes inline (then awaits the set's completion task)
    }

    private async Task Take(RunContext ctx, RewardsSetSynchronizer rss, Reward reward)
    {
        bool ok = reward is CardReward cardReward
            ? await TakeCardReward(ctx, rss, cardReward)
            : await rss.SelectLocalReward(reward);
        RoomsProbe.Bump(RoomsProbe.RewardTypesTaken, reward.GetType().Name + (ok ? "" : ":rejected"));
        if (!ok)
        {
            RoomsProbe.RewardTakeRejected++;
            // Only these may legitimately return false: closing/skipping the card reward screen, cancelling the
            // card selection of a removal / upgrade reward. Anything else means the host offered a no-op.
            if (!(reward is CardReward || reward is CardRemovalReward || reward is CardUpgradeReward))
            {
                ctx.Fail(AbortReason.ActionFault, "reward_take_rejected:" + reward.GetType().Name, "SelectLocalReward returned false for " + reward + " (the host offered a reward the game refused).");
            }
        }
    }

    /// <summary>
    /// CardReward.OnSelect only reads the per-player selector stack (Rewards/CardReward.cs:185), so the screen is
    /// stood in for by a selector pushed for this one reward. It pops itself as soon as the card selection ends
    /// (see <see cref="CardRewardSelector"/>); the finally block guarantees the stack is clean afterwards.
    /// </summary>
    private static async Task<bool> TakeCardReward(RunContext ctx, RewardsSetSynchronizer rss, CardReward reward)
    {
        CardRewardSelector selector = new CardRewardSelector(ctx, reward);
        selector.Scope = ctx.RunManager.CardSelectorStack.PushSelector(reward.Player, selector);
        try
        {
            return await rss.SelectLocalReward(reward);
        }
        finally
        {
            selector.Release();
            var stack = ctx.RunManager.CardSelectorStack._selectorStacks[ctx.SeatOf(reward.Player)];
            if (stack.Contains(selector))
            {
                ctx.Fail(AbortReason.HandlerFault, "card_reward_selector_leaked", "The card reward selector is still on the per-player selector stack (something was pushed on top of it and not removed).");
            }
        }
    }
}

/// <summary>
/// The per-player stack selector for ONE card reward (the host's NCardRewardSelectionScreen). Each
/// GetSelectedCardReward call (the first one and every call after a REROLL) is one nested CARD_REWARD decision.
/// </summary>
internal sealed class CardRewardSelector : ICardSelector
{
    private const string RerollId = "REROLL"; // CardRewardAlternative.Generate (Entities.CardRewardAlternatives/CardRewardAlternative.cs:49-56)

    private readonly RunContext _ctx;
    private readonly CardReward _reward;
    private readonly Decision _decision = new Decision();
    private readonly CardRewardContext _context = new CardRewardContext();
    private readonly List<CardRewardAlternative> _exposed = new List<CardRewardAlternative>(3);
    private bool _released;

    public IDisposable? Scope;

    public CardRewardSelector(RunContext ctx, CardReward reward)
    {
        _ctx = ctx;
        _reward = reward;
    }

    /// <summary>Pops the selector (StackedSelectorScope.Dispose pops only when it is the top, CardSelectorStack.cs:25-35).</summary>
    public void Release()
    {
        if (_released)
        {
            return;
        }
        _released = true;
        Scope?.Dispose();
    }

    /// <summary>Selection prompts raised while this selector is pushed keep their owner: the run's selector (DRIVER-API.md 7.5).</summary>
    public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        ICardSelector main = _ctx.CardSelector ?? throw new InvalidOperationException("RunContext.CardSelector is not set: no lifecycle installed the run's card selector.");
        RoomsProbe.CardRewardForwardedSelections++;
        return main.GetSelectedCards(options, minSelect, maxSelect);
    }

    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        RunContext ctx = _ctx;
        if (_released)
        {
            ctx.Fail(AbortReason.HandlerFault, "card_reward_selector_reused", "GetSelectedCardReward was called on a released card reward selector.");
        }
        // CardReward.OnSelect generates the alternatives once (CardReward.cs:167) and hands the same list to every
        // call, while the real screen is refreshed with freshly generated ones after a reroll (Populate ->
        // RefreshOptions, :139-141): REROLL is generated only while CanReroll (CardRewardAlternative.cs:49), and
        // Reroll() clears it (:301). So a REROLL that is no longer generated is not a button any more.
        _exposed.Clear();
        for (int k = 0; k < alternatives.Count; k++)
        {
            CardRewardAlternative alt = alternatives[k];
            if (alt.OptionId == RerollId && !_reward.CanReroll)
            {
                RoomsProbe.CardRewardRerollFiltered++;
                continue;
            }
            _exposed.Add(alt);
        }
        Player player = _reward.Player;
        int seat = ctx.SeatOf(player);
        _context.Reward = _reward;
        _context.Options = options;
        _context.Alternatives = _exposed;
        Decision d = _decision.Reset(DecisionKind.CardReward, null, actingSeat: seat, context: _context);
        // NCardRewardSelectionScreen: one holder per card (index in the options list, :195-215), one button per
        // alternative (:216-226). There is no separate close button: the Skip alternative carries the cancel hotkeys.
        for (int i = 0; i < options.Count; i++)
        {
            d.Add(Verb.PickOfferCard, seat, GameRef.OfCard(options[i].Card, i));
        }
        for (int k = 0; k < _exposed.Count; k++)
        {
            d.Add(Verb.ChooseOption, seat, GameRef.OfCardRewardAlt(_exposed[k], k));
            RoomsProbe.Bump(RoomsProbe.CardRewardAlternativesOffered, _exposed[k].OptionId);
        }
        if (d.Actions.Count == 0)
        {
            ctx.Stall("card_reward_no_option", "A card reward screen with no card and no alternative (the real UI would show an empty screen). " + ctx.Diagnose());
        }
        RoomsProbe.CardRewardDecisions++;
        int index = ctx.DecideNested(d);
        LegalAction a = d.Actions[index];
        bool endsSelection;
        CardRewardSelection result;
        if (a.Verb == Verb.PickOfferCard)
        {
            result = new CardRewardSelection { card = a.Source.Card };
            // CardReward.cs:215, evaluated by the game right after this call returns.
            endsSelection = !Hook.ShouldAllowSelectingMoreCardRewards(player.RunState, player, _reward);
        }
        else
        {
            CardRewardAlternative alt = a.Source.CardRewardAlt;
            result = new CardRewardSelection { alternative = alt };
            RoomsProbe.Bump(RoomsProbe.CardRewardAlternativesChosen, alt.OptionId);
            // CardReward.cs:227-230: EndSelectionAndDoNotCompleteReward (1) and EndSelectionAndCompleteReward (2) end the loop.
            endsSelection = alt.AfterSelected is PostAlternateCardRewardAction.EndSelectionAndDoNotCompleteReward or PostAlternateCardRewardAction.EndSelectionAndCompleteReward;
        }
        if (endsSelection)
        {
            // The game will not ask this selector again for this reward. Pop it now (it is the stack top: the game
            // just read it through GetSelector), so whatever OnSelect still does (CardPileCmd.Add hooks, the
            // alternative's OnSelect such as PaelsWing's relic) asks through the real local-selector path, as in the
            // real game where no stack selector exists.
            RoomsProbe.CardRewardEarlyRelease++;
            Release();
        }
        return result;
    }
}
