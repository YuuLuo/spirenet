using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game.Sync;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

// =========================================================================================================
//  REWARD / CARD_REWARD of REMOTE seats (phase H5; multi-seat runs only)
//
//  Game side. RewardsSet.Offer (Rewards/RewardsSet.cs:123-166) runs on the host for EVERY player: it generates the
//  set, begins it on that player's stack of the RewardsSetSynchronizer (:131) and - for a player that is not the
//  local one - simply awaits the set's completion (:165). The screen (NRewardsScreen, :162) exists only on the
//  owner's client; what that client does reaches the host as
//      RewardSelectedMessage{location, setId, rewardIndex}   NRewardButton -> RewardsSetSynchronizer.SelectLocalReward (:280-302)
//      RewardSetSkippedMessage{location, setId}              proceed of a NON-terminal screen -> SkipLocalRewardsSet (:304-314)
//      PlayerChoiceMessage{choiceId, index}                  the card reward screen, once per loop turn of
//                                                            CardReward.OnSelect (Rewards/CardReward.cs:172-205)
//      (nothing)                                             proceed of a TERMINAL screen: the set stays on the stack
//                                                            until the client leaves the room (BeforeLeavingRoom :446-466)
//  Host side. A remote seat's reward screen is an OVERLAY screen (IRemoteOverlay): the counterpart of seat 0's
//  RewardFrame, with the same click rules (RewardFrame.Build), asked right after the decision stack. The screens of
//  a seat form a stack (a reward that opens another set - Cauldron, CallingBell - puts a screen on top), mirrored
//  here from RewardsSetSynchronizer.BeginRewardsSet. Which sets get a screen is RewardsSet.Offer's own rule (:132).
// =========================================================================================================

internal sealed class RemoteRewardScreen
{
    public RewardsSet Set = null!;
    public Player Player = null!;
    public int Seat;
    public bool Terminal;
    public readonly RewardContext Context = new RewardContext();

    /// <summary>The take in flight (the task HandleRewardSelectedMessage started), and what is being taken.</summary>
    public Task? Take;
    public Reward? Taking;
    public string TakeWhat = "";
}

internal sealed class RemoteRewards : IRemoteOverlay, IRunLifecycle, IRemoteChoiceRouter
{
    /// <summary>The registered instance (one per session; the patch callbacks reach it through here).</summary>
    public static RemoteRewards? Instance;

    private RunContext? _ctx;
    private readonly List<RemoteRewardScreen>[] _screens =
    {
        new List<RemoteRewardScreen>(2), new List<RemoteRewardScreen>(2), new List<RemoteRewardScreen>(2), new List<RemoteRewardScreen>(2),
    };
    private readonly Decision _decision = new Decision();
    private readonly Dictionary<CardReward, IReadOnlyList<CardRewardAlternative>> _alternatives = new Dictionary<CardReward, IReadOnlyList<CardRewardAlternative>>();

    public string Name => RoomsNames.RemoteRewards;

    // ------------------------------------------------------------------------------------------------
    //  lifecycle
    // ------------------------------------------------------------------------------------------------

    public void OnRunSetUp(RunContext ctx)
    {
        _ctx = ctx.Mp != null ? ctx : null; // single seat: every callback below is inert
        Clear();
    }

    public void OnRoomEntered(RunContext ctx)
    {
        // A screen never survives its room's exit (the exit skips every open set, see OnBeforeLeavingRoom). A room that is
        // entered WITHOUT leaving the current one (an event combat) keeps the screens: their sets are still open.
    }

    public void OnEpisodeEnd(RunContext ctx)
    {
        _ctx = null;
        Clear();
    }

    private void Clear()
    {
        for (int s = 0; s < _screens.Length; s++)
        {
            _screens[s].Clear();
        }
        _alternatives.Clear();
    }

    /// <summary>The top reward screen of a seat, or null. (FoulPotion at the merchant: the inventory is covered by it.)</summary>
    public bool HasScreen(int seat) => seat > 0 && seat < _screens.Length && _screens[seat].Count > 0;

    public bool IsOpen(RunContext ctx, int seat) => HasScreen(seat);

    public bool AnyOpen(RunContext ctx)
    {
        for (int s = 1; s < ctx.SeatCount; s++)
        {
            if (_screens[s].Count > 0)
            {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------------------------------------
    //  observation points (RoomsPatches)
    // ------------------------------------------------------------------------------------------------

    /// <summary>
    /// RewardsSetSynchronizer.BeginRewardsSet just ran for a set (RewardsSet.Offer :131). For a remote player: the owner's
    /// client shows NRewardsScreen now, unless Offer returns without a screen (:132: no reward, not a combat room, empty
    /// sets not allowed).
    /// </summary>
    public void OnSetBegun(RewardsSet set)
    {
        RunContext? ctx = _ctx;
        if (ctx == null)
        {
            return;
        }
        int seat = ctx.SeatOf(set.Player);
        if (!ctx.IsRemoteSeat(seat))
        {
            return;
        }
        bool terminal = set.Room is CombatRoom;
        if (set.Rewards.Count <= 0 && !terminal && !set._allowEmptyRewards)
        {
            RoomsProbe.RemoteSetsWithoutScreen++;
            return;
        }
        RemoteRewardScreen screen = new RemoteRewardScreen { Set = set, Player = set.Player, Seat = seat, Terminal = terminal };
        screen.Context.Set = set;
        screen.Context.Terminal = terminal;
        _screens[seat].Add(screen);
        RoomsProbe.RemoteSetsOffered++;
        if (terminal)
        {
            RoomsProbe.RemoteTerminalSetsOffered++;
        }
        if (_screens[seat].Count > RoomsProbe.RemoteScreenMaxDepth)
        {
            RoomsProbe.RemoteScreenMaxDepth = _screens[seat].Count;
        }
    }

    /// <summary>
    /// CardRewardAlternative.Generate just produced the alternative buttons of a card reward of a remote player: at the
    /// start of CardReward.OnSelect (:167; the list the host indexes the client's answers with) and again in
    /// CardReward.Populate after a reroll (:139-141; the list the refreshed screen of the client shows). The latest
    /// list is what the client's screen displays.
    /// </summary>
    public void OnAlternativesGenerated(CardReward reward, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        RunContext? ctx = _ctx;
        if (ctx == null || !ctx.IsRemoteSeat(ctx.SeatOf(reward.Player)))
        {
            return;
        }
        _alternatives[reward] = alternatives;
    }

    /// <summary>
    /// RewardsSetSynchronizer.BeforeLeavingRoom just skipped the LOCAL player's open sets (RunManager.ExitCurrentRoom,
    /// Runs/RunManager.cs:1053, before the room is popped). Every client runs the same line for its own player at its own
    /// room exit and sends one RewardSetSkippedMessage per open set, top first (:453-463). Delivered here, inline: the
    /// host's handler skips the sender's stack TOP whatever the set id says (:336-354, :389-400), so a later delivery
    /// could hit a set that the next room began.
    /// </summary>
    public void OnBeforeLeavingRoom(RewardsSetSynchronizer rss)
    {
        RunContext? ctx = _ctx;
        if (ctx == null)
        {
            return;
        }
        for (int seat = 1; seat < ctx.SeatCount; seat++)
        {
            var stack = rss._rewardStates[seat].rewardsStack;
            VirtualClient client = ctx.Mp!.Client(seat);
            for (int guard = stack.Count; stack.Count > 0; guard--)
            {
                if (guard <= 0)
                {
                    ctx.Fail(AbortReason.HandlerFault, "remote_exit_skip_stuck", "Skipping the open reward sets of seat " + seat + " at the room exit does not empty its stack.");
                }
                RewardsSet top = stack[stack.Count - 1].set;
                RoomsProbe.RemoteExitSkips++;
                client.Send(new RewardSetSkippedMessage { location = client.Location, setId = top.Id });
            }
            if (_screens[seat].Count > 0)
            {
                RoomsProbe.RemoteScreensClosedByExit += _screens[seat].Count;
                _screens[seat].Clear();
            }
        }
        _alternatives.Clear();
    }

    // ------------------------------------------------------------------------------------------------
    //  the overlay screen
    // ------------------------------------------------------------------------------------------------

    public HandlerStep? Step(RunContext ctx, int seat)
    {
        List<RemoteRewardScreen> screens = _screens[seat];
        if (screens.Count == 0)
        {
            return null;
        }
        RemoteRewardScreen screen = screens[screens.Count - 1];
        RewardsSet set = screen.Set;
        if (screen.Take != null)
        {
            if (!screen.Take.IsCompleted)
            {
                // The screen is on top again but the reward it is taking has not finished and nothing else is asking.
                ctx.Stall("remote_reward_take_incomplete", "The reward screen of seat " + seat + " is on top while '" + screen.TakeWhat + "' is still running. " + ctx.Diagnose());
            }
            screen.Take = null;
        }
        RewardsSetSynchronizer rss = ctx.RunManager.RewardsSetSynchronizer;
        bool completed = rss.IsRewardsSetCompleted(set);
        if (completed && !set.AllRewardsSuccessfullySelected)
        {
            ctx.Fail(AbortReason.HandlerFault, "reward_set_skipped_externally", "Reward set " + set + " of seat " + seat + " was skipped while its screen was still open. " + ctx.Diagnose());
        }
        if (completed && !screen.Terminal)
        {
            // NRewardsScreen.UpdateScreenState (:366-391): every reward taken -> a non-terminal screen removes itself.
            RoomsProbe.RemoteAutoClosed++;
            screens.RemoveAt(screens.Count - 1);
            return HandlerStep.Progressed("remote reward screen closed itself");
        }
        if (!completed)
        {
            AssertOnTop(ctx, rss, screen);
        }
        Player player = screen.Player;
        RunState run = ctx.Run;
        Decision d = _decision.Reset(DecisionKind.Reward, this, actingSeat: seat, context: screen.Context);
        List<Reward> rewards = set.Rewards;
        bool anyUntaken = false;
        for (int i = 0; i < rewards.Count; i++)
        {
            Reward r = rewards[i];
            if (r.SuccessfullySelected)
            {
                continue;
            }
            anyUntaken = true;
            if (r is LinkedRewardSet || r.ParentRewardSet != null)
            {
                ctx.Fail(AbortReason.HandlerFault, "linked_reward_set", "A LinkedRewardSet appeared in " + set + "; the rooms driver does not implement NLinkedRewardSet.");
            }
            if (!RewardFrame.IsTakeEffective(r, player, run))
            {
                continue;
            }
            d.Add(Verb.ChooseOption, seat, GameRef.OfReward(r, i), flags: r is GoldReward ? ActionFlags.AutoGold : ActionFlags.None);
        }
        bool skipDisallowed = set.DisallowSkipping && anyUntaken;
        if (skipDisallowed)
        {
            RoomsProbe.ProceedDisallowed++;
        }
        if (!skipDisallowed && Hook.ShouldProceedToNextMapPoint(run))
        {
            d.Add(Verb.Proceed, seat);
        }
        return HandlerStep.Ask(d);
    }

    /// <summary>The client's messages act on the stack top of its player (RewardsSetSynchronizer.cs:356-371, :389-400); it must be this screen's set.</summary>
    private static void AssertOnTop(RunContext ctx, RewardsSetSynchronizer rss, RemoteRewardScreen screen)
    {
        var stack = rss._rewardStates[screen.Seat].rewardsStack;
        if (stack.Count == 0 || !ReferenceEquals(stack[stack.Count - 1].set, screen.Set))
        {
            ctx.Fail(AbortReason.HandlerFault, "reward_set_not_on_top", "The reward screen's set " + screen.Set + " of seat " + screen.Seat + " is not on top of the RewardsSetSynchronizer stack (" + stack.Count + " sets). " + ctx.Diagnose());
        }
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        int seat = a.Seat;
        List<RemoteRewardScreen> screens = _screens[seat];
        if (screens.Count == 0)
        {
            ctx.Fail(AbortReason.HandlerFault, "remote_reward_screen_gone", "Seat " + seat + " has no reward screen any more.");
        }
        RemoteRewardScreen screen = screens[screens.Count - 1];
        RewardsSet set = screen.Set;
        RewardsSetSynchronizer rss = ctx.RunManager.RewardsSetSynchronizer;
        VirtualClient client = ctx.Mp!.Client(seat);
        if (a.Verb == Verb.ChooseOption)
        {
            Reward r = a.Source.Reward;
            if (a.Source.Index >= set.Rewards.Count || !ReferenceEquals(set.Rewards[a.Source.Index], r))
            {
                ctx.Fail(AbortReason.HandlerFault, "reward_moved", "Reward " + r + " is no longer at index " + a.Source.Index + ".");
            }
            AssertOnTop(ctx, rss, screen);
            RoomsProbe.RemoteTakes++;
            screen.Taking = r;
            screen.TakeWhat = "seat " + seat + " takes reward " + r.GetType().Name;
            // NRewardButton -> SelectLocalReward on the client (:280-302): the message, then the same SelectRewardForPlayer.
            Task take = client.SendTracked(new RewardSelectedMessage { location = client.Location, setId = set.Id, rewardIndex = a.Source.Index });
            screen.Take = Observe(ctx, take, screen, r);
            ctx.Track(screen.Take, screen.TakeWhat, seat);
            return;
        }
        if (a.Verb != Verb.Proceed)
        {
            ctx.Fail(AbortReason.HandlerFault, "bad_verb", "Remote reward screen got " + a);
        }
        screens.RemoveAt(screens.Count - 1);
        if (!screen.Terminal)
        {
            // NRewardsScreen.cs:465-469: SkipLocalRewardsSet (the client's message), then the screen is removed.
            if (!rss.IsRewardsSetCompleted(set))
            {
                AssertOnTop(ctx, rss, screen);
                RoomsProbe.RemoteNonTerminalSkips++;
                client.Send(new RewardSetSkippedMessage { location = client.Location, setId = set.Id });
            }
            return;
        }
        if (!rss.IsRewardsSetCompleted(set))
        {
            RoomsProbe.RemoteTerminalLeftForRoomExit++;
        }
        TerminalProceed(ctx, seat, client);
    }

    /// <summary>The outcome of a take, checked when its task ends (the local path gets the same bool from SelectLocalReward).</summary>
    private static async Task Observe(RunContext ctx, Task take, RemoteRewardScreen screen, Reward reward)
    {
        try
        {
            await take;
        }
        finally
        {
            screen.Taking = null; // (set before the message is handled: the card reward's first wait happens inside the handler)
        }
        bool ok = reward.SuccessfullySelected;
        RoomsProbe.Bump(RoomsProbe.RewardTypesTaken, reward.GetType().Name + (ok ? "" : ":rejected") + ":remote");
        if (!ok)
        {
            RoomsProbe.RemoteTakeRejected++;
            // As for seat 0: only closing / skipping the card reward screen and cancelling the selection of a removal /
            // upgrade reward may legitimately end without the reward being taken.
            if (!(reward is CardReward || reward is CardRemovalReward || reward is CardUpgradeReward))
            {
                ctx.Fail(AbortReason.ActionFault, "reward_take_rejected:" + reward.GetType().Name, "Taking " + reward + " for a remote seat did not select it (the host offered a reward the game refused).");
            }
        }
    }

    /// <summary>
    /// NRewardsScreen.OnProceedButtonPressed of a TERMINAL screen on the remote player's client (:421-464), the tree of
    /// <see cref="RewardsLifecycle.TerminalProceed"/>:
    ///   first of two bosses      -> its map opens (only the second boss is travelable); nothing is sent
    ///   last boss of the act     -> ActChangeSynchronizer.SetLocalPlayerReady = VoteToMoveToNextActAction(player, act) (:34-42)
    ///   else                     -> RunManager.ProceedFromTerminalRewardsScreen: its map opens; nothing is sent.
    /// A terminal screen in a combat room that resumes its parent event (ShouldResumeParentEventAfterCombat) would make
    /// the client leave the combat room in its own instance; no encounter of v0.113 that resumes gives rewards
    /// (BattlewornDummy), so that state fails the episode instead of being guessed.
    /// </summary>
    private static void TerminalProceed(RunContext ctx, int seat, VirtualClient client)
    {
        RunState run = ctx.Run;
        AbstractRoom room = run.CurrentRoom ?? throw new InvalidOperationException("Terminal proceed without a current room.");
        ActMap map = run.Map;
        MapPoint? current = run.CurrentMapPoint;
        bool firstOfTwoBosses = map.SecondBossMapPoint != null && current == map.BossMapPoint;
        bool lastBossOfAct = (room.RoomType == RoomType.Boss || room.IsVictoryRoom) && current == (map.SecondBossMapPoint ?? map.BossMapPoint);
        if (firstOfTwoBosses)
        {
            RoomsProbe.RemoteTreeFirstOfTwoBosses++;
            ctx.OpenMapFor(seat, "first boss of two (remote)");
        }
        else if (lastBossOfAct)
        {
            RoomsProbe.RemoteTreeActVote++;
            ctx.SetActReady(seat);
            client.SendAction(new VoteToMoveToNextActAction(client.Player, run.CurrentActIndex));
        }
        else
        {
            if (run.CurrentRoomCount > 1 && room is CombatRoom { ShouldResumeParentEventAfterCombat: true })
            {
                ctx.Fail(AbortReason.HandlerFault, "remote_terminal_proceed_resume", "A remote seat proceeds from a terminal reward screen of a combat that resumes its parent event; no such encounter gives rewards in v0.113.");
            }
            RoomsProbe.RemoteTreeMap++;
            ctx.OpenMapFor(seat, "terminal rewards proceed (remote)");
        }
    }

    // ------------------------------------------------------------------------------------------------
    //  the card reward screen of a remote seat (a pending frame per WaitForRemoteChoice)
    // ------------------------------------------------------------------------------------------------

    /// <summary>
    /// IRemoteChoiceRouter: the game waits for a choice of a remote player whose reward screen is taking a CardReward -
    /// one turn of the loop of CardReward.OnSelect (:172-205). Asked after the routers of the prompts that can be raised
    /// INSIDE that loop (card selections, bundles), which claim their own waits first.
    /// </summary>
    public bool TryRoute(RunContext ctx, Player player, uint choiceId)
    {
        int seat = ctx.SeatOf(player);
        if (!ctx.IsRemoteSeat(seat))
        {
            return false;
        }
        List<RemoteRewardScreen> screens = _screens[seat];
        if (screens.Count == 0)
        {
            return false;
        }
        RemoteRewardScreen screen = screens[screens.Count - 1];
        if (screen.Taking is not CardReward reward)
        {
            return false; // no take of a card reward is in flight on this seat's top screen
        }
        if (!_alternatives.TryGetValue(reward, out IReadOnlyList<CardRewardAlternative>? alternatives))
        {
            ctx.Fail(AbortReason.HandlerFault, "remote_card_reward_without_alternatives", "The card reward of seat " + seat + " waits for a choice but its alternative buttons were never generated (rooms-remote-card-reward-alternatives).");
        }
        ctx.Push(new RemoteCardRewardFrame(reward, ctx.Mp!.Client(seat), choiceId, alternatives));
        RoomsProbe.RemoteCardRewardFrames++;
        return true;
    }
}

/// <summary>
/// One wait of CardReward.OnSelect for a remote player = the card reward screen of that player's client
/// (NCardRewardSelectionScreen): one holder per offered card, one button per alternative of the screen's CURRENT
/// alternative list. The answer is what the screen returns and the client syncs (CardReward.cs:181, :203-204;
/// NCardRewardSelectionScreen.cs:267, :277): the index of the card among the offered cards, or
/// (number of offered cards) + (position of the button on the screen).
/// </summary>
internal sealed class RemoteCardRewardFrame : IPendingDecision
{
    private readonly CardReward _reward;
    private readonly VirtualClient _client;
    private readonly uint _choiceId;
    private readonly IReadOnlyList<CardRewardAlternative> _alternatives;
    private readonly Decision _decision = new Decision();
    private readonly CardRewardContext _context = new CardRewardContext();

    public RemoteCardRewardFrame(CardReward reward, VirtualClient client, uint choiceId, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        _reward = reward;
        _client = client;
        _choiceId = choiceId;
        _alternatives = alternatives;
    }

    public string Name => RoomsNames.RemoteCardRewardFrame;

    public Decision? Build(RunContext ctx)
    {
        List<CardCreationResult> options = _reward._cards;
        int seat = _client.Seat;
        _context.Reward = _reward;
        _context.Options = options;
        _context.Alternatives = _alternatives;
        Decision d = _decision.Reset(DecisionKind.CardReward, this, actingSeat: seat, context: _context);
        for (int i = 0; i < options.Count; i++)
        {
            d.Add(Verb.PickOfferCard, seat, GameRef.OfCard(options[i].Card, i));
        }
        for (int k = 0; k < _alternatives.Count; k++)
        {
            d.Add(Verb.ChooseOption, seat, GameRef.OfCardRewardAlt(_alternatives[k], k));
            RoomsProbe.Bump(RoomsProbe.CardRewardAlternativesOffered, _alternatives[k].OptionId);
        }
        if (d.Actions.Count == 0)
        {
            ctx.Stall("card_reward_no_option", "A card reward screen of seat " + seat + " with no card and no alternative. " + ctx.Diagnose());
        }
        RoomsProbe.RemoteCardRewardDecisions++;
        return d;
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        int index;
        if (a.Verb == Verb.PickOfferCard)
        {
            index = a.Source.Index;
            if (index >= _reward._cards.Count || !ReferenceEquals(_reward._cards[index].Card, a.Source.Card))
            {
                ctx.Fail(AbortReason.HandlerFault, "card_reward_moved", "The offered card is no longer at index " + index + ".");
            }
        }
        else
        {
            index = _reward._cards.Count + a.Source.Index;
            RoomsProbe.Bump(RoomsProbe.CardRewardAlternativesChosen, a.Source.CardRewardAlt.OptionId);
        }
        ctx.Pop(this);
        _client.SendChoice(_choiceId, PlayerChoiceResult.FromIndex(index));
    }
}
