using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

// =========================================================================================================
//  TREASURE (driver-spec-rooms.md 7; testmode-audit-nodes.md U-TR-1..4)
//
//  Game side: TreasureRoom.EnterInternal -> TreasureRoomRelicSynchronizer.BeginRelicPicking (relics fixed on entry,
//  hidden until the chest is open; 0 relics -> voting ended at once). Everything after that is UI:
//    NTreasureRoom.OpenChest (:225-268)        DoNormalRewards (gold) -> DoExtraRewardsIfNeeded (non-terminal sets)
//                                              -> relic collection InitializeRelics / AnimIn
//    NTreasureRoomRelicCollection (:154-166, :256-263)  empty chest -> CompleteWithNoRelics() after a 1 s tween
//    pick (:294-300)                           PickRelicLocally(i) -> PickRelicAction -> OnPicked -> AwardRelics
//                                              -> RelicsAwarded(results, isFinished)
//    OnRelicsAwarded / AnimateRelicAwards (:302-418)  results sorted by type; per result: RunSafely(RelicCmd.Obtain)
//                                              (not awaited) + MoveToFallback for every other player; isFinished ->
//                                              relic picking finished -> proceed button + travel enabled (:241-254)
//    skip = proceed button while "Skip" (:183-197)    SkipRelicLocally + (single player) ProceedFromTerminalRewardsScreen
//    proceed after the pick                    ProceedFromTerminalRewardsScreen
//  Opening is forced: there is no proceed button and no travel before it (:109, :285-296). Host pages (SubKind):
//    CHEST  OPEN_CHEST only (auto-advanced unless potion actions are added; the first decision point of the room is
//           therefore before the chest gold, as the oracle's room-boundary dump - replay request R5)
//    RELIC  CHOOSE_OPTION(relic) + SKIP                 (relics exposed only now)
//    RELIC  PROCEED only, after RelicsAwarded(isFinished) or the empty-chest completion: TreasureContext.Finished
//           (+ Empty, Awarded), Relics = []
//
//  MULTI-SEAT (phase H5). Every player has the same room on its own client and goes through the same pages; the
//  relics (one per player, TreasureRoomRelicSynchronizer.cs:79-91) are shared and voted for:
//    CHEST  per seat. Seat 0: NTreasureRoom.OpenChest as above (DoExtraRewardsIfNeeded offers the - empty - extra sets
//           of EVERY player, Rooms/TreasureRoom.cs:66-88: it runs once, at the host's own chest). A remote seat:
//           its client's DoLocalTreasureRoomRewards = TreasureChestOpenedMessage (OneOffSynchronizer.cs:51-59); the
//           host grants that player's gold from that player's own stream (:61-90).
//    RELIC  per seat, after its own chest: CHOOSE_OPTION(relic i) = PickRelicLocally(i) for seat 0, the client's
//           PickRelicAction(player, i) for a remote seat (:139-161). There is NO skip with more than one player: the
//           button reads "Skip" and enables only for one player (NTreasureRoom.cs:158, :241-245; otherwise it stays
//           disabled until the picking is finished, :246-254, :285-296). A seat that has voted waits; no re-vote. The
//           treasure vote is not a team vote (the seats want different relics): every seat is asked, in seat order,
//           and sees the votes cast so far (TreasureContext.SeatVotes; NTreasureRoomRelicCollection shows them).
//    RELIC  PROCEED only, per seat, once RelicsAwarded(isFinished) arrived: seat 0 = ProceedFromTerminalRewardsScreen,
//           a remote seat = its map opens (nothing is sent).
// =========================================================================================================

internal sealed class TreasureState
{
    /// <summary>Index = seat (a single-seat run uses [0] only).</summary>
    public readonly bool[] Opened = new bool[DriverApi.MaxSeats];
    public readonly bool[] PickRequested = new bool[DriverApi.MaxSeats];
    public readonly int[] Gold = new int[DriverApi.MaxSeats];

    /// <summary>The relic index a seat voted for (multi-seat), or -1.</summary>
    public readonly int[] Vote = { -1, -1, -1, -1 };

    public bool Skipped;
    public bool AwardFinished;
    public bool Empty;
    public readonly List<RelicModel> Awarded = new List<RelicModel>(2);
}

internal sealed class TreasureLifecycle : IRunLifecycle
{
    private RunContext? _ctx;
    private Action<List<RelicPickingResult>, bool>? _awarded;
    private TreasureRoomRelicSynchronizer? _sync;

    public string Name => RoomsNames.TreasureLifecycle;

    public void OnRunSetUp(RunContext ctx)
    {
        _ctx = ctx;
        _awarded ??= OnRelicsAwarded;
        _sync = ctx.RunManager.TreasureRoomRelicSynchronizer; // per run (RunManager.InitializeShared)
        _sync.RelicsAwarded += _awarded;
    }

    public void OnRoomEntered(RunContext ctx)
    {
    }

    public void OnEpisodeEnd(RunContext ctx)
    {
        if (_sync != null && _awarded != null)
        {
            _sync.RelicsAwarded -= _awarded;
        }
        _sync = null;
        _ctx = null;
    }

    /// <summary>
    /// Host stand-in for NTreasureRoomRelicCollection.OnRelicsAwarded -> AnimateRelicAwards (:302-418): the same game
    /// calls in the same order. The UI awards after its animations (frames later), so the work is posted and waits for
    /// quiescence. Each Obtain is started like the UI's RunSafely (not awaited before the next call) but tracked, so a
    /// fault aborts the episode instead of being logged and swallowed. Seat-agnostic: on a real host this UI code
    /// awards every player's relic (it is the only run-state mutation that lives in UI code).
    /// </summary>
    private void OnRelicsAwarded(List<RelicPickingResult> results, bool isFinished)
    {
        RunContext? ctx = _ctx;
        if (ctx == null)
        {
            return;
        }
        AbstractRoom? room = ctx.CurrentRoom;
        ctx.Post(async () =>
        {
            await ctx.Quiescence();
            TreasureState? state = room is TreasureRoom ? ctx.RoomData<TreasureState>(room) : null;
            // AnimateRelicAwards :331 (List.Sort on the same list, same comparison -> same order as the UI).
            results.Sort((RelicPickingResult r1, RelicPickingResult r2) => r1.type.CompareTo(r2.type));
            IReadOnlyList<Player> players = ctx.Run.Players;
            foreach (RelicPickingResult result in results)
            {
                RelicModel relic = result.relic.ToMutable();
                if (result.type != RelicPickingResultType.Skipped)
                {
                    ctx.Track(RelicCmd.Obtain(relic, result.player!), "treasure relic obtain " + relic.Id.Entry);
                    RoomsProbe.TreasureRelicsObtained++;
                    state?.Awarded.Add(relic);
                }
                for (int i = 0; i < players.Count; i++)
                {
                    if (players[i] != result.player)
                    {
                        players[i].RelicGrabBag.MoveToFallback(result.relic);
                        RoomsProbe.TreasureFallbackMoves++;
                    }
                }
            }
            if (isFinished && state != null)
            {
                state.AwardFinished = true;
            }
        }, "award treasure relics");
    }
}

internal sealed class TreasureHandler : IRoomHandler
{
    private readonly Decision _decision = new Decision();
    private readonly TreasureContext _context = new TreasureContext();

    public string Name => RoomsNames.TreasureHandler;

    public HandlerStep Step(RunContext ctx, int seat)
    {
        if (ctx.CurrentRoom is not TreasureRoom room)
        {
            return HandlerStep.Unrecognised("treasure handler outside a TreasureRoom");
        }
        TreasureState state = ctx.RoomData<TreasureState>(room);
        TreasureRoomRelicSynchronizer sync = ctx.RunManager.TreasureRoomRelicSynchronizer;
        bool multi = ctx.SeatCount > 1;
        _context.Room = room;
        _context.Finished = state.AwardFinished;
        _context.Empty = state.Empty;
        _context.Awarded = state.Awarded;
        _context.SeatVotes.Clear();
        if (!state.Opened[seat])
        {
            // CHEST page (obs-schema-v1 section 6; DECISIONS-H1 B.8 "the chest must be opened first"): the chest button is
            // the only room action of NTreasureRoom before opening - no proceed button and no travel (:109, :285-296) - so
            // OPEN_CHEST is the only action of this handler and the core auto-advances it unless a frame extension adds
            // something the real UI also allows here (out-of-combat potions on the top bar). The relics are fixed on entry
            // but hidden until the chest is open (driver-spec-rooms.md 7.2): nothing of them is exposed on this page.
            _context.Relics = Array.Empty<RelicModel>();
            _context.GoldGained = 0;
            Decision chest = _decision.Reset(DecisionKind.Treasure, this, actingSeat: seat, subKind: (int)TreasurePhase.Chest, context: _context);
            chest.Add(Verb.OpenChest, seat);
            return HandlerStep.Ask(chest);
        }
        if (state.Skipped)
        {
            return HandlerStep.Unrecognised("treasure relic skipped but the map is not open");
        }
        IReadOnlyList<RelicModel>? relics = sync.CurrentRelics;
        _context.GoldGained = state.Gold[seat];
        Decision d = _decision.Reset(DecisionKind.Treasure, this, actingSeat: seat, subKind: (int)TreasurePhase.Relic, context: _context);
        if (state.AwardFinished)
        {
            // Relic picking finished (or the chest was empty): only the proceed button (NTreasureRoom.cs:241-254).
            _context.Relics = Array.Empty<RelicModel>();
            d.Add(Verb.Proceed, seat);
            return HandlerStep.Ask(d);
        }
        if (state.PickRequested[seat])
        {
            // One seat: the vote resolves at once, so this state means the award never came. Several seats: the award
            // waits for every player's vote (TreasureRoomRelicSynchronizer.cs:191).
            return multi
                ? HandlerStep.Waiting("voted for a treasure relic")
                : HandlerStep.Unrecognised("a relic pick was requested but RelicsAwarded never finished the picking");
        }
        if (relics == null || relics.Count == 0)
        {
            return HandlerStep.Unrecognised("chest opened, relic picking neither active nor finished");
        }
        _context.Relics = relics;
        for (int i = 0; i < relics.Count; i++)
        {
            d.Add(Verb.ChooseOption, seat, GameRef.OfTreasureRelic(relics[i], i));
        }
        if (multi)
        {
            // No skip with more than one player (NTreasureRoom.cs:158, :241-245). The votes cast so far are public.
            for (int s = 0; s < ctx.SeatCount; s++)
            {
                _context.SeatVotes.Add(state.PickRequested[s] ? state.Vote[s] : -1);
            }
        }
        else
        {
            // Single player: the proceed button turns into "Skip" (enabled after 2.5 s, NTreasureRoom.cs:246-268).
            d.Add(Verb.Skip, seat);
        }
        return HandlerStep.Ask(d);
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        TreasureRoomRelicSynchronizer sync = ctx.RunManager.TreasureRoomRelicSynchronizer;
        TreasureState state = ctx.RoomData<TreasureState>();
        int seat = a.Seat;
        VirtualClient? client = ctx.Mp != null && ctx.IsRemoteSeat(seat) ? ctx.Mp.Client(seat) : null;
        switch (a.Verb)
        {
            case Verb.OpenChest:
                // NTreasureRoom.OnChestButtonReleased (:198-202): RunSafely(OpenChest()) - here tracked, so a fault aborts.
                if (state.Opened[seat] || ctx.CurrentRoom is not TreasureRoom room)
                {
                    ctx.Fail(AbortReason.HandlerFault, "chest_already_opened", "OPEN_CHEST applied twice or outside a TreasureRoom.");
                    return;
                }
                state.Opened[seat] = true;
                if (client != null)
                {
                    // The client's NTreasureRoom.OpenChest: DoNormalRewards = DoLocalTreasureRoomRewards sends this message.
                    RoomsProbe.RemoteChestsOpened++;
                    Task gold = client.SendTracked(new TreasureChestOpenedMessage { Location = client.Location });
                    ctx.Track(RemoteChestGold((Task<int>)gold, state, seat), "seat " + seat + " opens its chest", seat);
                }
                else
                {
                    ctx.Track(OpenChest(ctx, room, sync, state), "open chest", seat);
                }
                break;
            case Verb.ChooseOption:
                // NTreasureRoomRelicCollection.PickRelic (:294-300) -> PickRelicAction -> OnPicked -> AwardRelics -> RelicsAwarded.
                state.PickRequested[seat] = true;
                state.Vote[seat] = a.Source.Index;
                ctx.SetSeatFlag(seat, SeatStatus.VotedTreasure);
                RoomsProbe.TreasurePicks++;
                if (client != null)
                {
                    RoomsProbe.RemoteTreasurePicks++;
                    client.SendAction(new PickRelicAction(client.Player, a.Source.Index));
                }
                else
                {
                    sync.PickRelicLocally(a.Source.Index);
                }
                break;
            case Verb.Skip:
                // NTreasureRoom.OnProceedButtonReleased (:183-197), IsSkip: SkipRelicLocally; single player also proceeds.
                // (OnPicked sets _singleplayerSkipped; OnRoomExited ends the voting, so the next chest can start.)
                if (ctx.SeatCount != 1)
                {
                    ctx.Fail(AbortReason.HandlerFault, "treasure_skip_multi_seat", "SKIP of the treasure relic exists only for one player.");
                }
                state.Skipped = true;
                RoomsProbe.TreasureSkips++;
                sync.SkipRelicLocally();
                ctx.Track(RoomsFlow.ProceedFromTerminal(ctx, "treasure relic skipped"), "treasure skip proceed");
                break;
            case Verb.Proceed:
                if (client != null)
                {
                    ctx.OpenMapFor(seat, "leave treasure room (remote)");
                }
                else
                {
                    ctx.Track(RoomsFlow.ProceedFromTerminal(ctx, "leave treasure room"), "treasure proceed", seat);
                }
                break;
            default:
                ctx.Fail(AbortReason.HandlerFault, "bad_verb", "Treasure frame got " + a);
                break;
        }
    }

    /// <summary>NTreasureRoom.OpenChest (:225-245) up to the relic collection, plus the empty-chest completion of NTreasureRoomRelicCollection.AnimIn (:254-263).</summary>
    private static async Task OpenChest(RunContext ctx, TreasureRoom room, TreasureRoomRelicSynchronizer sync, TreasureState state)
    {
        RoomsProbe.ChestsOpened++;
        state.Gold[0] = await room.DoNormalRewards();
        RoomsProbe.ChestGold += state.Gold[0];
        await room.DoExtraRewardsIfNeeded();
        IReadOnlyList<RelicModel>? relics = sync.CurrentRelics;
        if (relics == null || relics.Count == 0)
        {
            // InitializeRelics: empty chest (SilverCrucible / ShouldGenerateTreasure false) -> AnimIn's tween callback
            // CompleteWithNoRelics() (raises RelicsAwarded(empty, isFinished: true)).
            state.Empty = true;
            RoomsProbe.ChestsEmpty++;
            sync.CompleteWithNoRelics();
        }
    }

    /// <summary>The gold the host granted to a remote seat for its chest (OneOffSynchronizer.DoTreasureRoomRewards): shown on that seat's screen.</summary>
    private static async Task RemoteChestGold(Task<int> gold, TreasureState state, int seat)
    {
        RoomsProbe.ChestsOpened++;
        state.Gold[seat] = await gold;
        RoomsProbe.ChestGold += state.Gold[seat];
    }
}
