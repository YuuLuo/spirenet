using System.Collections.Generic;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Runs;

namespace SpireHost.Driver;

/// <summary>
/// The MAP decision.
///
/// Legal next nodes = exactly what the real UI marks Travelable
/// (NMapScreen.RecalculateTravelability, Nodes.Screens.Map/NMapScreen.cs:623-670):
///   no visited coord in this act          -> { StartingMapPoint }
///   last visited == BossMapPoint and a SecondBossMapPoint exists -> { SecondBossMapPoint }
///   last visited is in the last grid row  -> { BossMapPoint }
///   otherwise                             -> MapTravel.GetTravelablePointsFrom(run, point)
///                                            = Hook.ShouldAllowFreeTravel ? every point of the next row : point.Children
/// and nothing is travelable unless Hook.ShouldProceedToNextMapPoint(run) (NMapScreen.SetTravelEnabled, :979-983).
///
/// Travel = the real vote path, identical for 1 and N seats:
///   MapSelectionSynchronizer.OnLocalPlayerVotedFor(coord) -> VoteForMapCoordAction -> (all votes in)
///   MoveToMapCoord (map_point_selection RNG picks among the votes) -> MoveToMapCoordAction -> RunManager.EnterMapCoord.
/// The room type of an unknown ('?') node is rolled by the game inside EnterMapPointInternal; the host never
/// looks at it beforehand.
///
/// MULTI-SEAT (phase H5; lead ruling Q5). Every seat votes (the move waits for every slot,
/// MapSelectionSynchronizer.cs:101), each for one of ITS travelable points (the same set for every seat: it is run
/// state), once, when its own map is open; no re-vote. The MAP decision of a seat is a team vote (StepKind.Vote): the
/// core asks it only when no seat has anything else to do, lowest seat first. Seat 0 votes through
/// OnLocalPlayerVotedFor; a remote seat's vote is its client's RequestEnqueueActionMessage carrying the
/// VoteForMapCoordAction that the client's own OnLocalPlayerVotedFor builds (source = current coord + act, vote =
/// coord + the map generation count; :52-67). With EpisodeOptions.TeamVotes the decision is asked once and the same
/// coord is then cast for every other seat, one seat per step. The run is expected at one of the voted coords once
/// the last vote is cast.
///
/// Out-of-combat potion use / discard on the MAP frame (DECISIONS-H1 B.7) is added by the rooms module's frame
/// extension rooms:potions, like on every other room-level frame.
/// </summary>
public sealed class MapHandler : IRoomHandler
{
    private readonly Decision _decision = new Decision();
    private readonly MapContext _context = new MapContext();

    // ---- multi-seat: which seats have cast their vote for the current location (the vote of a seat that was cast
    //      for an older map generation is void, MapSelectionSynchronizer.BeforeMapGenerated :143-165) ----
    private RunContext? _votesOf;
    private int _votesRoomSerial = -1;
    private readonly int[] _voteGeneration = { -1, -1, -1, -1 };
    private readonly MapCoord[] _voteCoord = new MapCoord[RunDriver.MaxSeats];
    private bool _teamVoteSet;
    private MapCoord _teamVote;
    private int _teamVoteGeneration;
    private readonly List<MapCoord> _expected = new List<MapCoord>(4);

    /// <summary>Team map votes that the host cast for a seat without asking (EpisodeOptions.TeamVotes); module counter driver.team_map_votes.</summary>
    public static long TeamVotesCast;

    /// <summary>Map votes of remote seats sent through their virtual clients; module counter driver.remote_map_votes.</summary>
    public static long RemoteVotesSent;

    public string Name => "map";

    public HandlerStep Step(RunContext ctx, int seat)
    {
        RunState run = ctx.Run;
        ActMap map = run.Map;
        List<MapPoint> travelable = _context.Travelable;
        travelable.Clear();
        _context.Map = map;
        _context.SeatVotes.Clear();
        bool multi = ctx.SeatCount > 1;
        MapSelectionSynchronizer sync = ctx.RunManager.MapSelectionSynchronizer;
        if (multi)
        {
            SyncVoteRecord(ctx);
            if (HasVoted(sync, seat))
            {
                return HandlerStep.Waiting("voted on the map");
            }
        }

        if (!Hook.ShouldProceedToNextMapPoint(run))
        {
            return HandlerStep.Unrecognised("Hook.ShouldProceedToNextMapPoint is false: travel is disabled and nothing else is asking for input");
        }

        IReadOnlyList<MapCoord> visited = run.VisitedMapCoords;
        if (visited.Count == 0)
        {
            travelable.Add(map.StartingMapPoint);
        }
        else
        {
            MapCoord last = visited[visited.Count - 1];
            if (map.SecondBossMapPoint != null && last == map.BossMapPoint.coord)
            {
                travelable.Add(map.SecondBossMapPoint);
            }
            else if (last.row == map.GetRowCount() - 1)
            {
                travelable.Add(map.BossMapPoint);
            }
            else
            {
                MapPoint? point = map.GetPoint(last);
                if (point == null)
                {
                    // The real UI logs an error and falls back to the starting point (map regenerated under the player).
                    travelable.Add(map.StartingMapPoint);
                }
                else
                {
                    foreach (MapPoint next in MapTravel.GetTravelablePointsFrom(run, point))
                    {
                        travelable.Add(next);
                    }
                }
            }
        }
        if (travelable.Count == 0)
        {
            return HandlerStep.Unrecognised("the map is open but no node is travelable (act boss already beaten?)");
        }
        // Canonical order (row, col): a HashSet's enumeration order must never reach the action list.
        travelable.Sort(static (a, b) => a.coord.row != b.coord.row ? a.coord.row.CompareTo(b.coord.row) : a.coord.col.CompareTo(b.coord.col));

        if (multi)
        {
            if (ctx.Options.TeamVotes && _teamVoteSet && _teamVoteGeneration == sync.MapGenerationCount)
            {
                // Lead ruling Q5: the team's vote was asked once; this seat's client casts the same vote now.
                bool legal = false;
                for (int i = 0; i < travelable.Count; i++)
                {
                    legal |= travelable[i].coord == _teamVote;
                }
                if (!legal)
                {
                    ctx.Fail(AbortReason.HandlerFault, "team_vote_not_travelable", "The team's map vote " + _teamVote + " is not travelable for seat " + seat + ".");
                }
                TeamVotesCast++;
                CastVote(ctx, sync, seat, _teamVote);
                return HandlerStep.Progressed("team map vote of seat " + seat);
            }
            for (int s = 0; s < ctx.SeatCount; s++)
            {
                _context.SeatVotes.Add(HasVoted(sync, s) ? map.GetPoint(_voteCoord[s]) : null);
            }
        }

        Decision d = _decision.Reset(DecisionKind.Map, this, actingSeat: seat, context: _context);
        for (int i = 0; i < travelable.Count; i++)
        {
            d.Add(Verb.Travel, seat, GameRef.OfMapPoint(travelable[i]));
        }
        return multi ? HandlerStep.Vote(d) : HandlerStep.Ask(d);
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction action = decision.Actions[actionIndex];
        MapCoord coord = action.Source.MapPoint.coord;
        if (ctx.SeatCount == 1)
        {
            ctx.CloseMap();
            ctx.Session.ExpectTravel(coord);
            // Requires an unpaused ActionExecutor (it is: non-combat rooms unpause on entry, combat end unpauses).
            // If it were paused the vote would sit in the queue and the driver would report a livelock, not hang.
            ctx.RunManager.MapSelectionSynchronizer.OnLocalPlayerVotedFor(coord);
            return;
        }
        MapSelectionSynchronizer sync = ctx.RunManager.MapSelectionSynchronizer;
        SyncVoteRecord(ctx);
        if (HasVoted(sync, action.Seat))
        {
            ctx.Fail(AbortReason.HandlerFault, "map_vote_twice", "Seat " + action.Seat + " already voted on the map for this location.");
        }
        if (ctx.Options.TeamVotes)
        {
            _teamVoteSet = true;
            _teamVote = coord;
            _teamVoteGeneration = sync.MapGenerationCount;
        }
        CastVote(ctx, sync, action.Seat, coord);
    }

    /// <summary>Per-episode / per-location reset of the vote record.</summary>
    private void SyncVoteRecord(RunContext ctx)
    {
        if (ReferenceEquals(_votesOf, ctx) && _votesRoomSerial == ctx.RoomSerial)
        {
            return;
        }
        _votesOf = ctx;
        _votesRoomSerial = ctx.RoomSerial;
        for (int s = 0; s < _voteGeneration.Length; s++)
        {
            _voteGeneration[s] = -1;
        }
        _teamVoteSet = false;
    }

    private bool HasVoted(MapSelectionSynchronizer sync, int seat) => _voteGeneration[seat] == sync.MapGenerationCount;

    private void CastVote(RunContext ctx, MapSelectionSynchronizer sync, int seat, MapCoord coord)
    {
        RunState run = ctx.Run;
        int generation = sync.MapGenerationCount;
        _voteGeneration[seat] = generation;
        _voteCoord[seat] = coord;
        ctx.SetSeatFlag(seat, SeatStatus.VotedMap);
        bool last = true;
        for (int s = 0; s < ctx.SeatCount; s++)
        {
            last &= HasVoted(sync, s);
        }
        if (last)
        {
            // All votes are in once this one executes: the host draws the destination among them (MapSelectionSynchronizer.cs:101-128).
            _expected.Clear();
            for (int s = 0; s < ctx.SeatCount; s++)
            {
                _expected.Add(_voteCoord[s]);
            }
            ctx.Session.ExpectTravelAnyOf(_expected);
        }
        if (seat == 0)
        {
            sync.OnLocalPlayerVotedFor(coord);
            return;
        }
        // The client's own OnLocalPlayerVotedFor (:52-67) builds exactly this action and requests it.
        MapLocation source = new MapLocation(run.CurrentMapCoord, run.CurrentActIndex);
        MapVote vote = default;
        vote.coord = coord;
        vote.mapGenerationCount = generation;
        RemoteVotesSent++;
        ctx.Mp!.Client(seat).SendAction(new VoteForMapCoordAction(ctx.Seats[seat], source, vote));
    }
}
