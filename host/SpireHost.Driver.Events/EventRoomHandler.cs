using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game.Sync;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rooms;

namespace SpireHost.Driver.Events;

/// <summary>
/// Room handler of RoomScreen.Event: the host's stand-in for NEventRoom (Nodes.Rooms/NEventRoom.cs) and, for the
/// FakeMerchant event, for NFakeMerchant (Nodes.Events.Custom/NFakeMerchant.cs).
///
/// EVENT decision (driver-spec-rooms.md 4.2, testmode-audit-nodes.md U-EV-1/2):
///   unfinished event  CHOOSE_OPTION(i) for every CurrentOptions[i] with !IsLocked (NEventOptionButton.cs:262,
///                     NEventRoom.cs:236) and !(DisableOnChosen && WasChosen) (EventOption.cs:124, a guard: such a
///                     click would do nothing). Single player: an option that WillKillPlayer stays legal (the UI only
///                     shows a red glow, NEventOptionButton.cs:266-288); with more than one seat the UI blocks it.
///   finished event    the UI-synthesized PROCEED (NEventRoom.cs:217-224) = RunManager.Proceed() = open the map.
/// Apply (NEventRoom.OptionButtonClicked, :233-250):
///   IsProceed option  option.Chosen() directly, not through the synchronizer (no history, no vote). The only two
///                     in v0.113: the Ancient "PROCEED" of AncientEventModel.cs:172 (OnChosen = NEventRoom.Proceed ->
///                     RunManager.Proceed(): the core opens the host map) and Trial DOUBLE_DOWN (-> CONFIRM frame).
///   other option      EventSynchronizer.ChooseLocalOption(i): non-shared -> ChooseOptionForEvent (RunSafely(Chosen())
///                     + history); shared -> the vote path (PlayerVotedForSharedOptionIndex, one vote in single player,
///                     host picks with the event_synchronizer RNG). Every option task it starts is tracked.
///
/// FakeMerchant (driver-spec-rooms.md 4.4; NFakeMerchant.cs): while !StartedFight the room is a SHOP over the event's
/// MerchantInventory (relic entries only): BUY(e) for IsStocked && EnoughGold (MerchantEntry.cs:30, what
/// OnTryPurchaseUnsynchronized checks), applied through MerchantSynchronizer.PurchaseLocalEntry (NMerchantSlot path,
/// purchase lock); PROCEED = NFakeMerchant.OnProceedButtonReleased -> RunManager.Proceed() = open the map (the map is
/// travel-enabled from the start, NFakeMerchant.cs:153). Throwing a FoulPotion at the merchant is a potion action
/// added by the out-of-combat potion extension; EventPatches makes it usable / effective headlessly.
///
/// MULTI-SEAT (phase H5). EventSynchronizer.BeginEvent makes one mutable event per player (:92-99); every player sees
/// and clicks its own instance (GetEventForPlayer).
///   non-shared event  a remote seat's click is its client's OptionIndexChosenMessage{Event, index, location}
///                     (ChooseLocalOption :202-211 -> host :172-188); its PROCEED opens its own map, nothing is sent.
///   shared event      every player votes for an option of the current page; when all votes are in the host draws one
///                     of them (event_synchronizer RNG) and runs that option for EVERY player (:117-156, :214-227).
///                     Seat 0 votes through ChooseLocalOption; a remote seat's vote is its client's
///                     VotedForSharedEventOptionMessage{optionIndex, pageIndex, location} (:190-201 -> :102-138).
///                     The page vote is a TEAM VOTE (StepKind.Vote; lead ruling Q5): asked only when no seat has
///                     anything else to do; with EpisodeOptions.TeamVotes it is asked once and the same option is cast
///                     for every other seat for which that button is clickable (a seat for which it is not is asked).
///                     A seat that has voted waits; no re-vote.
///   event combat      needs no input: the host runs the chosen option for every instance, so every instance reports
///                     ReadyToEnterCombat (EventCombatSynchronizer.cs:76-94).
/// The UI mirror (<see cref="MirrorDisplay"/>) is done for seat 0 only: on a real host the remote players' texts are
/// never displayed. FakeMerchant cannot occur with more than one seat (FakeMerchant.IsAllowed, single player only).
/// </summary>
internal sealed class EventRoomHandler : IRoomHandler
{
    private readonly Decision _decision = new Decision();
    private readonly EventContext _context = new EventContext();
    private readonly ShopContext _shop = new ShopContext { FakeMerchant = true };

    // ---- multi-seat, shared events: the team's vote for one page ----
    private RunContext? _teamOf;
    private int _teamRoomSerial = -1;
    private uint _teamPage;
    private int _teamOption = -1;

    public string Name => "events:event";

    public HandlerStep Step(RunContext ctx, int seat)
    {
        if (ctx.CurrentRoom is not EventRoom)
        {
            return HandlerStep.Unrecognised("Event screen but the current room is " + (ctx.CurrentRoom?.GetType().Name ?? "null"));
        }
        EventSynchronizer sync = ctx.RunManager.EventSynchronizer;
        Player owner = ctx.Seats[seat];
        EventModel ev = sync.GetEventForPlayer(owner);
        if (!ReferenceEquals(ev.Owner, owner))
        {
            throw new InvalidOperationException("Event " + ev.Id.Entry + " of seat " + seat + " has another owner (BeginEvent did not run).");
        }
        bool multi = ctx.SeatCount > 1;

        if (ev is FakeMerchant fm && !fm.StartedFight)
        {
            if (multi)
            {
                return HandlerStep.Unrecognised("FakeMerchant in a multi-seat run (the event is single-player only)");
            }
            return ShopStep(ctx, fm, seat);
        }

        if (seat == 0)
        {
            MirrorDisplay(ev, owner);
        }
        // Contract R1: the event screen as the UI shows it, in the context (the encoder reads nothing else).
        _context.BeginPage(ev);
        _context.SeatVotes.Clear();
        Decision d = _decision.Reset(DecisionKind.Event, this, actingSeat: seat, context: _context);
        if (ev.IsFinished)
        {
            _context.AddSynthesizedProceed();
            d.Add(Verb.Proceed, seat);
            return HandlerStep.Ask(d);
        }
        bool sharedVote = multi && ev.IsShared;
        if (sharedVote && sync.GetPlayerVote(owner).HasValue)
        {
            return HandlerStep.Waiting("voted on the shared event page");
        }
        IReadOnlyList<EventOption> options = ev.CurrentOptions;
        for (int i = 0; i < options.Count; i++)
        {
            EventOption o = options[i];
            bool willKill = o.WillKillPlayer?.Invoke(owner) ?? false;
            EventOptionInfo view = _context.AddOption(o, i, willKill);
            if (o.IsLocked)
            {
                continue;
            }
            if (o.WasChosen && o.DisableOnChosen)
            {
                continue; // Chosen() would be a no-op (EventOption.cs:124)
            }
            if (multi && willKill)
            {
                continue; // NEventOptionButton.OnRelease blocks it in multiplayer only (:266-284)
            }
            view.Legal = true;
            d.Add(Verb.ChooseOption, seat, GameRef.OfEventOption(o, i));
        }
        if (d.Actions.Count == 0)
        {
            return HandlerStep.Unrecognised("event " + ev.Id.Entry + " is not finished and has no clickable option (page " + (ev.Description?.LocEntryKey ?? "-") + ")");
        }
        if (!sharedVote)
        {
            return HandlerStep.Ask(d);
        }

        // Shared event: the page vote. The votes cast so far are public (EventSynchronizer.PlayerVoteChanged / GetPlayerVote).
        for (int s = 0; s < ctx.SeatCount; s++)
        {
            uint? vote = sync.GetPlayerVote(ctx.Seats[s]);
            _context.SeatVotes.Add(vote.HasValue ? (int)vote.Value : -1);
        }
        if (ctx.Options.TeamVotes && ReferenceEquals(_teamOf, ctx) && _teamRoomSerial == ctx.RoomSerial && _teamPage == sync._pageIndex && _teamOption >= 0)
        {
            for (int i = 0; i < d.Actions.Count; i++)
            {
                if (d.Actions[i].Verb == Verb.ChooseOption && d.Actions[i].Source.Index == _teamOption)
                {
                    // Lead ruling Q5: the team's vote was asked once; this seat's client casts the same vote now.
                    EventsStats.TeamVotesCast++;
                    CastSharedVote(ctx, sync, seat, _teamOption);
                    return HandlerStep.Progressed("team vote of seat " + seat + " on the shared event page");
                }
            }
            EventsStats.TeamVoteNotClickable++; // the team's option is not a button this seat can press: the seat is asked itself
        }
        return HandlerStep.Vote(d);
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        if (decision.Kind == DecisionKind.Shop)
        {
            ApplyShop(ctx, a);
            return;
        }
        int seat = a.Seat;
        VirtualClient? client = ctx.Mp != null && ctx.IsRemoteSeat(seat) ? ctx.Mp.Client(seat) : null;
        if (a.Verb == Verb.Proceed)
        {
            if (client != null)
            {
                // The client's synthesized PROCEED opens the map of its own instance; nothing reaches the host.
                ctx.OpenMapFor(seat, "event finished (remote)");
                return;
            }
            // The UI-synthesized PROCEED of a finished event (NEventRoom.cs:217-224): OnChosen = NEventRoom.Proceed =
            // RunManager.Proceed() (NEventRoom.cs:308-312). The game call is made; the core opens the host map on it.
            ctx.RunManager.Proceed();
            RequireMapOpen(ctx, "event finished (synthesized PROCEED)");
            return;
        }
        EventSynchronizer sync = ctx.RunManager.EventSynchronizer;
        EventModel ev = sync.GetEventForPlayer(ctx.Seats[seat]);
        EventOption option = a.Source.EventOption;
        int index = a.Source.Index;
        if (index >= ev.CurrentOptions.Count || !ReferenceEquals(ev.CurrentOptions[index], option))
        {
            ctx.Fail(AbortReason.HostBug, "event_option_moved", "Event option " + option.TextKey + " is no longer at index " + index + " of " + ev.Id.Entry);
        }
        if (option.IsProceed)
        {
            if (client != null)
            {
                // A client runs an IsProceed option in its own instance only (NEventRoom.cs:240-244: no synchronizer, no message).
                // Both such options of v0.113 are unreachable for a remote seat (Ancient PROCEED: never generated; Trial
                // DOUBLE_DOWN: WillKillPlayer, blocked with more than one player).
                ctx.Fail(AbortReason.HandlerFault, "remote_proceed_option", "Seat " + seat + " chose the IsProceed option " + option.TextKey + " of " + ev.Id.Entry + "; a client runs it without telling the host.");
            }
            // NEventRoom.cs:240-244: TaskHelper.RunSafely(option.Chosen()), no synchronizer.
            // An option whose OnChosen is NEventRoom.Proceed (the Ancient PROCEED of AncientEventModel.cs:172) reaches
            // RunManager.Proceed(); the core opens the host map on that call (run-proceed-opens-map, contract R3).
            Task t = option.Chosen();
            ctx.Track(t, "event proceed option " + option.TextKey, seat);
            return;
        }
        if (ctx.SeatCount > 1 && ev.IsShared)
        {
            if (ctx.Options.TeamVotes)
            {
                _teamOf = ctx;
                _teamRoomSerial = ctx.RoomSerial;
                _teamPage = sync._pageIndex;
                _teamOption = index;
            }
            CastSharedVote(ctx, sync, seat, index);
            return;
        }
        List<Task> pending = sync._pendingOptionTasks;
        int before = pending.Count;
        if (client != null)
        {
            EventsStats.RemoteOptions++;
            OptionIndexChosenMessage message = default;
            message.type = OptionIndexType.Event;
            message.optionIndex = (uint)index;
            message.location = client.Location;
            client.Send(message);
        }
        else
        {
            sync.ChooseLocalOption(index);
        }
        if (pending.Count != before + 1 && ctx.SeatCount > 1)
        {
            ctx.Fail(AbortReason.HandlerFault, "event_option_not_started", "Choosing option " + option.TextKey + " of " + ev.Id.Entry + " for seat " + seat + " started " + (pending.Count - before) + " option task(s).");
        }
        for (int i = before; i < pending.Count; i++)
        {
            ctx.Track(pending[i], "event option " + option.TextKey, seat);
        }
    }

    /// <summary>
    /// One seat's vote on the current page of a shared event. When it is the last vote the host draws the option and
    /// starts it for every player, in player order (EventSynchronizer.cs:140-156, :214-227): those option tasks are
    /// tracked, each for its own seat.
    /// </summary>
    private static void CastSharedVote(RunContext ctx, EventSynchronizer sync, int seat, int index)
    {
        List<Task> pending = sync._pendingOptionTasks;
        int before = pending.Count;
        EventsStats.SharedVotes++;
        ctx.SetSeatFlag(seat, SeatStatus.VotedSharedPage);
        if (ctx.IsRemoteSeat(seat))
        {
            VirtualClient client = ctx.Mp!.Client(seat);
            EventsStats.RemoteSharedVotes++;
            VotedForSharedEventOptionMessage message = default;
            message.optionIndex = (uint)index;
            message.pageIndex = sync._pageIndex;
            message.location = client.Location;
            client.Send(message);
        }
        else
        {
            sync.ChooseLocalOption(index);
        }
        int started = pending.Count - before;
        if (started == 0)
        {
            return; // not the last vote
        }
        if (started != ctx.SeatCount)
        {
            ctx.Fail(AbortReason.HandlerFault, "shared_option_tasks", "The shared event resolved a page and started " + started + " option task(s) for " + ctx.SeatCount + " players.");
        }
        EventsStats.SharedPagesResolved++;
        ctx.ClearSeatFlag(SeatStatus.VotedSharedPage); // the page is resolved: the next one starts without votes
        for (int i = 0; i < started; i++)
        {
            ctx.Track(pending[before + i], "shared event option (seat " + i + ")", i);
        }
    }

    // ------------------------------------------------------------------------------------------------
    //  FakeMerchant as a SHOP frame
    // ------------------------------------------------------------------------------------------------

    private HandlerStep ShopStep(RunContext ctx, FakeMerchant fm, int seat)
    {
        MerchantInventory inventory = fm.Inventory ?? throw new InvalidOperationException("FakeMerchant has no inventory (BeforeEventStarted did not run).");
        _shop.Inventory = inventory;
        _shop.FakeMerchant = true;
        Decision d = _decision.Reset(DecisionKind.Shop, this, actingSeat: seat, context: _shop);
        List<MerchantEntry> entries = inventory.AllEntries;
        for (int i = 0; i < entries.Count; i++)
        {
            MerchantEntry e = entries[i];
            if (e.IsStocked && e.EnoughGold)
            {
                d.Add(Verb.ChooseOption, seat, GameRef.OfShopEntry(e, i));
            }
        }
        d.Add(Verb.Proceed, seat);
        return HandlerStep.Ask(d);
    }

    private static void ApplyShop(RunContext ctx, LegalAction a)
    {
        if (a.Verb == Verb.Proceed)
        {
            // NFakeMerchant.OnProceedButtonReleased (NFakeMerchant.cs:247-250) = RunManager.Proceed(); the core opens the host map on it.
            ctx.RunManager.Proceed();
            RequireMapOpen(ctx, "leave fake merchant");
            return;
        }
        ctx.Track(Purchase(ctx, a.Source.ShopEntry), "fake merchant purchase " + a.Source.ShopEntry.GetType().Name);
    }

    /// <summary>Fail closed: after RunManager.Proceed() the host map must be open (the core hook ran).</summary>
    private static void RequireMapOpen(RunContext ctx, string why)
    {
        if (!ctx.MapOpen)
        {
            ctx.Fail(AbortReason.HostBug, "map_not_opened", "RunManager.Proceed() was called (" + why + ") but the host map is not open: run-proceed-opens-map did not run.");
        }
    }

    private static async Task Purchase(RunContext ctx, MerchantEntry entry)
    {
        bool ok = await ctx.RunManager.MerchantSynchronizer.PurchaseLocalEntry(entry);
        if (!ok)
        {
            // Offered only when IsStocked && EnoughGold: a refusal is a host bug, never trial and error (DRIVER-API.md 8.3).
            ctx.Fail(AbortReason.HostBug, "purchase_rejected", "FakeMerchant purchase of " + entry.GetType().Name + " was rejected.");
        }
    }

    // ------------------------------------------------------------------------------------------------
    //  UI mirror
    // ------------------------------------------------------------------------------------------------

    /// <summary>
    /// What the UI does to the displayed texts, unasked, every time it shows a page (idempotent):
    ///   NEventRoom.SetDescription (NEventRoom.cs:205-216): Character.AddDetailsTo(description), "IsMultiplayer",
    ///                                                      Event.DynamicVars.AddTo(description)
    ///   NEventOptionButton._Ready (NEventOptionButton.cs:181-182): Event.DynamicVars.AddTo(Option.Description / Title)
    /// The option-title variables matter beyond display: EventSynchronizer.SaveEventOptionToHistory copies
    /// option.HistoryName.Variables (HistoryName == Title by default) into the run history, so without this mirror the
    /// headless history would lack the event variables the real game records. Writes LocString variable maps only.
    /// </summary>
    private static void MirrorDisplay(EventModel ev, Player owner)
    {
        if (ev.Description != null && LocStringExists(ev.Description))
        {
            owner.Character.AddDetailsTo(ev.Description, ev.FoolPronounGuess);
            ev.Description.Add("IsMultiplayer", owner.RunState.Players.Count > 1);
            ev.DynamicVars.AddTo(ev.Description);
        }
        if (ev.IsFinished)
        {
            return;
        }
        IReadOnlyList<EventOption> options = ev.CurrentOptions;
        for (int i = 0; i < options.Count; i++)
        {
            ev.DynamicVars.AddTo(options[i].Description);
            ev.DynamicVars.AddTo(options[i].Title);
        }
    }

    private static bool LocStringExists(MegaCrit.Sts2.Core.Localization.LocString s)
    {
        return !s.IsEmpty && MegaCrit.Sts2.Core.Localization.LocString.Exists(s.LocTable, s.LocEntryKey);
    }
}

/// <summary>Multi-seat counters of the events module (module counters "events.mp.*"; all 0 in single-seat runs).</summary>
internal static class EventsStats
{
    public static long RemoteOptions;
    public static long SharedVotes;
    public static long RemoteSharedVotes;
    public static long SharedPagesResolved;
    public static long TeamVotesCast;
    public static long TeamVoteNotClickable;
    public static long RemoteBundleFrames;
    public static long RemoteCrystalFrames;
    public static long RemoteCrystalRewardsSent;
}
