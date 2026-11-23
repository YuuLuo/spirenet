using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Unlocks;
using SpireHost.Boot;
using SpireHost.Engine;
using SpireHost.Sim;

namespace SpireHost.Driver;

/// <summary>
/// The run driver core: run start chain, the settle / classify / ask / apply loop, the decision stack, the
/// tracked operations, auto-advance, strict-mode checks, terminal detection and teardown.
///
/// DEFINITIONS (the contract; also in DRIVER-API.md section 5)
///
///   quiescent       The pump queue is empty, no recorded game-task fault, no strict-counter violation, every
///                   tracked operation that completed has been observed, nobody is awaiting
///                   RunContext.Quiescence(), and either no virtual timer is pending or a pending frame is on
///                   the decision stack (a paused executor polls on a virtual timer; it must not be allowed to
///                   spin while the game is waiting for an answer).
///
///   decision point  Quiescent, and exactly one of, in this priority order:
///                     T  run.IsGameOver                                  -> terminal (Win in the victory room, else Loss)
///                     P1 the decision stack is not empty                 -> the top frame builds the decision
///                     P2 CombatManager.IsInProgress                      -> the Combat room handler
///                     P3 a room-level screen (Map / Event / Shop / Rest / Treasure) with no blocking tracked
///                        operation pending                               -> that screen's room handler
///                   and the decision that comes out has at least one legal action.
///
///   stall           Quiescent and no decision point: unknown screen, no handler registered, the handler does
///                   not recognise the state, a blocking operation is still incomplete with an empty stack and
///                   no combat, a decision with zero legal actions, or a map move that did not arrive.
///                   -> the episode is Aborted(Stall, detail) with a diagnostics snapshot. No waiting, no retry.
///
///   livelock        The engine or the driver keeps running without becoming a decision point:
///                     pump_callbacks       more than MaxCallbacksPerSettle continuations in one settle
///                     virtual_time_budget  more than VirtualTimeBudget of virtual time in one settle
///                     handler_spin         more than MaxHandlerProgressSteps "Progressed" steps in a row
///                     auto_advance_spin    more than MaxAutoAdvanceRun auto-advanced decisions in a row
///                     decision_cap         EpisodeOptions.DecisionCap decision points in the episode
///                     combat_work_without_decision  more than LivelockGuard.CombatWorkLimit combat-history entries since the
///                                             last decision: one game call that does not return within the bound (stress
///                                             finding F1: an endless loop; also a finite chain longer than the bound)
///                     turns_without_decision  more than LivelockGuard.Limit side-turn starts since the last decision
///                                          point (the game's turn loop spinning inside one continuation)
///                   -> Aborted(Livelock, detail).
/// </summary>
internal sealed class RunDriver
{
    public const long MaxCallbacksPerSettle = 1_000_000;
    public const int MaxHandlerProgressSteps = 256;
    public const int MaxAutoAdvanceRun = 20_000;
    public static readonly TimeSpan VirtualTimeBudget = TimeSpan.FromSeconds(120);

    private enum Phase
    {
        Idle,
        Start,
        Settle,
        Build,
        Ask,
        Apply,
        Teardown,
    }

    private struct TrackedOp
    {
        public Task Task;
        public string What;
        public bool Blocking;

        /// <summary>The seat the operation belongs to (multi-seat: only that seat's room handler waits for it), or -1.</summary>
        public int Seat;
    }

    private readonly HostSession _session;
    private readonly EnginePump _pump;
    private readonly List<IPendingDecision> _stack = new List<IPendingDecision>(8);
    private readonly List<TrackedOp> _ops = new List<TrackedOp>(8);
    private readonly List<TaskCompletionSource> _quiescenceWaiters = new List<TaskCompletionSource>(4);
    private readonly List<TaskCompletionSource> _quiescenceBatch = new List<TaskCompletionSource>(4);
    private readonly Action _roomEntered;
    private readonly Action<CombatRoom> _combatWon;

    private RunContext? _ctx;
    private IPolicy _policy = null!;
    private IEpisodeObserver? _observer;
    private EpisodeOptions _opt = new EpisodeOptions();
    private EpisodeCounters _c = new EpisodeCounters();
    private Phase _phase;
    private string _phaseNote = "";
    private SigHasher _trace;
    private List<ulong>? _traceList;
    private List<int>? _actionLog;
    private readonly List<MapCoord> _pendingTravel = new List<MapCoord>(4);
    private StrictCounters _strictStart;
    private long[] _nativeStart = Array.Empty<long>();
    private int _handlerSpin;
    private int _autoRun;
    private string _lastNote = "";
    private string _lastDecision = "";
    private bool _observerStarted;
    private bool _roomBoundaryPending;
    private bool _combatWonSubscribed;
    private SortedDictionary<string, long> _moduleStart = new SortedDictionary<string, long>(StringComparer.Ordinal);

    /// <summary>Thrown inside the start chain to end the episode as Truncated (limits reached while a scenario Setup is still running).</summary>
    private sealed class EpisodeTruncatedException : Exception
    {
        public EpisodeTruncatedException(string by) : base(by)
        {
        }
    }

    public RunDriver(HostSession session)
    {
        _session = session;
        _pump = session.Pump;
        _roomEntered = OnRoomEntered;
        _combatWon = OnCombatWon;
    }

    public RunContext? Context => _ctx;

    public EpisodeCounters Counters => _c;

    // =====================================================================================================
    //  Episode
    // =====================================================================================================

    public EpisodeResult Run(EpisodeSpec spec, IPolicy policy, EpisodeOptions options, IEpisodeObserver? observer)
    {
        if (_ctx != null || _phase != Phase.Idle)
        {
            throw new InvalidOperationException("An episode is already running in this session.");
        }
        Stopwatch wall = Stopwatch.StartNew();
        EpisodeResult result = new EpisodeResult
        {
            RequestedAscension = spec.Ascension,
            Character = spec.Characters.Length > 0 ? spec.Characters[0].ToUpperInvariant() : "",
        };
        _policy = policy;
        _observer = observer;
        _opt = options;
        _c = result.Counters;
        _stack.Clear();
        _ops.Clear();
        _quiescenceWaiters.Clear();
        _pendingTravel.Clear();
        _handlerSpin = 0;
        _autoRun = 0;
        _observerStarted = false;
        _roomBoundaryPending = false;
        _lastNote = "";
        _lastDecision = "";
        _trace = SigHasher.Create();
        _traceList = options.RecordTrace ? new List<ulong>(1024) : null;
        _actionLog = options.RecordActions ? new List<int>(1024) : null;
        _strictStart = _session.ReadStrict();
        _nativeStart = NullGodot.SnapshotCounters();
        _moduleStart = _session.Registry.ReadModuleCounters();
        _session.Registry.Frozen = true;
        _pump.EpisodeRunning = true;
        try
        {
            try
            {
                _phase = Phase.Start;
                foreach (IEpisodeCheck check in _session.Registry.EpisodeChecks)
                {
                    _phaseNote = "OnEpisodeBegin:" + check.Name;
                    check.OnEpisodeBegin();     // step 2 of the episode lifecycle: before anything of this episode has happened (R-8)
                }
                _phaseNote = "";
                StartRun(spec, result);
                result.StartMs = wall.Elapsed.TotalMilliseconds;
                if (_observer != null)
                {
                    _observer.OnEpisodeStart(_ctx!);
                    _observerStarted = true;
                    _roomBoundaryPending = true; // the first room: reported at the first quiescent point of the main loop
                }
                MainLoop(result);
            }
            catch (EpisodeAbortException e)
            {
                result.Outcome = EpisodeOutcome.Aborted;
                result.Abort = MakeAbort(e.Reason, e.Detail, e.Message, e.InnerException);
            }
            catch (EpisodeTruncatedException e)
            {
                result.Outcome = EpisodeOutcome.Truncated;
                result.TruncatedBy = e.Message;
            }
            catch (PolicyStopException e)
            {
                result.Outcome = EpisodeOutcome.Truncated;
                result.TruncatedBy = "policy_stop: " + e.Message;
            }
            catch (Exception e)
            {
                // An exception that escaped game code or host code without being classified: attribute it to the phase.
                AbortReason reason = _phase switch
                {
                    Phase.Start => AbortReason.StartFailed,
                    Phase.Settle => AbortReason.GameTaskFault,
                    Phase.Build => AbortReason.HandlerFault,
                    Phase.Ask => AbortReason.PolicyFault,
                    Phase.Apply => AbortReason.ActionFault,
                    _ => AbortReason.HostBug,
                };
                result.Outcome = EpisodeOutcome.Aborted;
                result.Abort = MakeAbort(reason, _phase.ToString().ToLowerInvariant() + (_phaseNote.Length > 0 ? ":" + _phaseNote : ""), e.GetBaseException().Message, e);
            }
        }
        finally
        {
            if (_ctx != null && result.Outcome != EpisodeOutcome.Aborted)
            {
                // The last evaluation of the registered episode checks, while the run is still alive: a condition that
                // game code swallowed and that no later quiescent point saw must still fail the episode.
                try
                {
                    CheckFailure? failed = EvaluateChecks(final: true);
                    if (failed.HasValue)
                    {
                        result.Outcome = EpisodeOutcome.Aborted;
                        result.Abort = MakeAbort(failed.Value.Reason, failed.Value.Detail, failed.Value.Message, null);
                    }
                }
                catch (Exception e)
                {
                    result.Outcome = EpisodeOutcome.Aborted;
                    result.Abort = MakeAbort(AbortReason.HostBug, "episode_check_threw", e.GetBaseException().Message, e);
                }
            }
            if (_ctx != null && _observer != null && _observerStarted)
            {
                try
                {
                    _observer.OnTerminalState(_ctx, result);
                }
                catch (Exception e)
                {
                    result.TeardownFaults++;
                    result.TeardownFault ??= "observer.OnTerminalState: " + e.Message;
                }
            }
            Teardown(result);
            _session.Registry.Frozen = false;
            _pump.EpisodeRunning = false;
            _phase = Phase.Idle;
        }
        if (_opt.ComputeSignatures || _opt.RecordTrace)
        {
            result.TraceHash = _trace.Finish();
        }
        result.Trace = _traceList;
        result.ActionLog = _actionLog;
        result.WallMs = wall.Elapsed.TotalMilliseconds;
        try
        {
            _observer?.OnEpisodeEnd(result);
        }
        catch (Exception e)
        {
            HostLog.Error("observer.OnEpisodeEnd threw: " + e);
        }
        _observer = null;
        return result;
    }

    // =====================================================================================================
    //  Run start: the real chain (driver-spec-rooms.md section 1; DECISIONS-H1 D)
    //    NCharacterSelectScreen.InitializeSingleplayer -> StartRunLobby.BeginRunLocally (act selection RNG,
    //    ascension clamp) -> GeneratePlayers (UnlockState round trip) -> NGame.StartNewSingleplayerRun
    //    (acts.ToMutable, RunState.CreateForNewRun, SetUpNewSingleplayer) -> NGame.StartRun
    //    (FinalizeStartingRelics, Launch, EnterAct(0, doTransition: false)).
    // =====================================================================================================

    private void StartRun(EpisodeSpec spec, EpisodeResult result)
    {
        if (!string.IsNullOrEmpty(spec.SavedRunJson))
        {
            StartSavedRun(spec, result);
            return;
        }
        if (spec.Characters == null || spec.Characters.Length == 0)
        {
            throw new EpisodeAbortException(AbortReason.StartFailed, "bad_spec", "EpisodeSpec.Characters is empty.");
        }
        if (spec.Characters.Length > MaxSeats)
        {
            throw new EpisodeAbortException(AbortReason.StartFailed, "bad_spec", "EpisodeSpec.Characters has " + spec.Characters.Length + " seats; a run has 1 to " + MaxSeats + ".");
        }
        int seatCount = spec.Characters.Length;
        RunManager rm = RunManager.Instance;
        if (rm.IsInProgress)
        {
            // The previous teardown failed half-way. One forced attempt; if the run is still there the process must be restarted.
            try
            {
                rm.CleanUp(graceful: false);
                _pump.TryRunUntilQuiescent(MaxCallbacksPerSettle, out _);
                _pump.DiscardQueued();
            }
            catch (Exception)
            {
                // reported below
            }
            if (rm.IsInProgress)
            {
                throw new EpisodeAbortException(AbortReason.StartFailed, "run_in_progress", "RunManager still has a run in progress and cannot be cleaned up; restart the process.");
            }
        }
        if (!_pump.IsQueueEmpty)
        {
            throw new EpisodeAbortException(AbortReason.StartFailed, "pump_not_empty", "The pump queue is not empty before run start.");
        }
        _pump.Clock.Clear();
        HostFaults.Drain();
        SpireHost.Patches.LivelockGuard.Arm();

        // ---- seed (StartRunLobby.BeginRunForAllPlayersIfAllReady: canonicalised user seed, or SeedHelper.GetRandomSeed) ----
        string seed = SeedHelper.CanonicalizeSeed(spec.Seed ?? _session.NewRandomSeed());
        if (seed.Length == 0)
        {
            throw new EpisodeAbortException(AbortReason.StartFailed, "bad_seed", "Empty seed.");
        }
        result.Seed = seed;

        // ---- profile: restore the pinned veteran snapshot; WongoPoints derived from the run seed (D.2, D.3) ----
        // (EpisodeOptions.DebugWongoPoints >= 0: tool-only override for the real-client trace comparison, whose traces were recorded with 0 points)
        int wongo = _opt.DebugWongoPoints >= 0 ? _opt.DebugWongoPoints : VeteranProfile.WongoPointsFor(seed);
        long tRestore = Stopwatch.GetTimestamp();
        VeteranProfile.Restore(wongo);
        result.ProfileRestoreMs = (Stopwatch.GetTimestamp() - tRestore) * 1000.0 / Stopwatch.Frequency;
        result.WongoPoints = wongo;

        RunState run;
        Player[] players;
        HeadlessHostNetService? hostService = null;
        if (seatCount == 1)
        {
            // ---- StartRunLobby.BeginRunLocally ----
            CharacterModel? character = null;
            foreach (CharacterModel c in ModelDb.AllCharacters)
            {
                if (c.Id.Entry.Equals(spec.Characters[0], StringComparison.OrdinalIgnoreCase))
                {
                    character = c;
                }
            }
            if (character == null)
            {
                throw new EpisodeAbortException(AbortReason.StartFailed, "bad_spec", "Unknown character: " + spec.Characters[0]);
            }
            long tActs = Stopwatch.GetTimestamp();
            UnlockState unlock = VeteranProfile.CreateUnlockState();                       // AddLocalHostPlayer + GeneratePlayers round trip
            UnlockState lobbyUnlock = new UnlockState(new[] { unlock });                    // StartRunLobby.GetUnlockState()
            Rng actRng = new Rng(StringHelper.GetDeterministicHashCode(seed), "act_selection");
            List<ActModel> acts = ActModel.GetRandomList(actRng, lobbyUnlock, isMultiplayer: false).ToList();
            // list[0] = GetAct(Act1) ?? list[0]: the lobby's Act1 is "random" by default, so there is no override.
            int ascension = Math.Min(spec.Ascension, SaveManager.Instance.Progress.GetOrCreateCharacterStats(character.Id).MaxAscension);
            if (ascension < 0)
            {
                ascension = 0;
            }
            result.Ascension = ascension;
            result.Character = character.Id.Entry;

            result.ActSelectionMs = (Stopwatch.GetTimestamp() - tActs) * 1000.0 / Stopwatch.Frequency;

            // ---- StartRunLobby.GeneratePlayers / NGame.StartNewSingleplayerRun ----
            Player player = Player.CreateForNewRun(character, unlock, 1UL);
            if (_opt.DebugForcedHp > 0)
            {
                player.Creature.SetMaxHpInternal(_opt.DebugForcedHp);
                player.Creature.SetCurrentHpInternal(_opt.DebugForcedHp);
            }
            long tSetup = Stopwatch.GetTimestamp();
            run = RunState.CreateForNewRun(new[] { player }, acts.Select(a => a.ToMutable()).ToList(), Array.Empty<ModifierModel>(), GameMode.Standard, false, ascension, seed);
            // D.5 / fidelity R3: the value of the real predicate (RunManager.ShouldApplyTutorialModifications :644-663: Standard and not
            // custom), so that the real discovery-order branch runs exactly when the real client runs it (a no-op for the veteran profile).
            rm.ForceDiscoveryOrderModifications = run.GameMode == GameMode.Standard && !run.IsCustomGame;
            rm.SetUpNewSingleplayer(run, shouldSave: false);                                // includes InitializeNewRun + GenerateRooms
            result.RunSetUpMs = (Stopwatch.GetTimestamp() - tSetup) * 1000.0 / Stopwatch.Frequency;
            players = new[] { player };
        }
        else
        {
            run = SetUpMultiSeatRun(spec, result, rm, seed, out players, out hostService);
        }
        foreach (ActModel a in run.Acts)
        {
            result.Acts.Add(a.Id.Entry);
        }
        result.Characters.Clear();
        for (int s = 0; s < players.Length; s++)
        {
            result.Characters.Add(players[s].Character.Id.Entry);
        }

        _ctx = new RunContext(_session, this, spec, _opt, run, seed);
        if (hostService != null)
        {
            _ctx.Mp = new MultiSeatRun(_ctx, hostService);
            MultiSeatStats.Runs++;
        }
        DriverPatches.ResetIntentTracker();
        rm.RoomEntered += _roomEntered;
        CombatManager.Instance.CombatWon += _combatWon;
        _combatWonSubscribed = true;
        foreach (IRunLifecycle l in _session.Registry.Lifecycles)
        {
            _phaseNote = "OnRunSetUp:" + l.Name;
            l.OnRunSetUp(_ctx);
        }
        foreach (IEpisodeCheck check in _session.Registry.EpisodeChecks)
        {
            _phaseNote = "OnEpisodeStart:" + check.Name;
            check.OnEpisodeStart(_ctx);
        }
        _phaseNote = "";

        // ---- NGame.StartRun ----
        long tEnter = Stopwatch.GetTimestamp();
        DriveStartTask(rm.FinalizeStartingRelics(), "FinalizeStartingRelics");
        rm.Launch();
        if (_ctx.Mp != null)
        {
            AssertMultiSeatLaunched(_ctx, rm);
        }
        DriveStartTask(rm.EnterAct(0, doTransition: false), "EnterAct(0)");
        result.EnterActMs = (Stopwatch.GetTimestamp() - tEnter) * 1000.0 / Stopwatch.Frequency;
        for (int s = 0; s < players.Length; s++)
        {
            Player player = players[s];
            int forcedHp = players.Length == 1 ? _opt.DebugForcedHp : ForcedHpOf(s);
            if (forcedHp > 0 && player.Creature.CurrentHp < forcedHp)
            {
                // Neow sets HP to 0 and heals; A2+ heals only 80% of the missing HP. Test-only top-up.
                player.Creature.SetCurrentHpInternal(forcedHp);
            }
        }

        if (spec.Scenario?.Setup != null)
        {
            _phaseNote = "scenario:" + spec.Scenario.Name;
            DriveStartTask(spec.Scenario.Setup(_ctx), "scenario " + spec.Scenario.Name);
            _phaseNote = "";
        }
    }

    /// <summary>
    /// Restore a saved single-player <see cref="SerializableRun"/> snapshot (exact real-client Continue chain:
    /// <c>RunState.FromSerializable</c> -> <c>SetUpSavedSingleplayer</c> -> <c>Launch</c> -> <c>GenerateMap</c> ->
    /// <c>LoadIntoLatestMapCoord</c>), then populate public win-history counters from <c>MapPointHistory</c>.
    /// Used by <c>SpireHost --shadow-serve</c>.
    /// </summary>
    private void StartSavedRun(EpisodeSpec spec, EpisodeResult result)
    {
        RunManager rm = RunManager.Instance;
        if (rm.IsInProgress)
        {
            try
            {
                rm.CleanUp(graceful: false);
                _pump.TryRunUntilQuiescent(MaxCallbacksPerSettle, out _);
                _pump.DiscardQueued();
            }
            catch (Exception)
            {
                // reported below
            }
            if (rm.IsInProgress)
            {
                throw new EpisodeAbortException(AbortReason.StartFailed, "run_in_progress", "RunManager still has a run in progress and cannot be cleaned up; restart the process.");
            }
        }
        if (!_pump.IsQueueEmpty)
        {
            throw new EpisodeAbortException(AbortReason.StartFailed, "pump_not_empty", "The pump queue is not empty before saved run start.");
        }
        _pump.Clock.Clear();
        HostFaults.Drain();
        SpireHost.Patches.LivelockGuard.Arm();

        SerializableRun save = JsonSerializer.Deserialize(spec.SavedRunJson!, JsonSerializationUtility.GetTypeInfo<SerializableRun>())
            ?? throw new EpisodeAbortException(AbortReason.StartFailed, "bad_save", "SavedRunJson deserialized to null.");

        string seed = save.SerializableRng?.Seed ?? "";
        result.Seed = seed;
        int wongo = _opt.DebugWongoPoints >= 0 ? _opt.DebugWongoPoints : VeteranProfile.WongoPointsFor(seed);
        long tRestore = Stopwatch.GetTimestamp();
        VeteranProfile.Restore(wongo);
        result.ProfileRestoreMs = (Stopwatch.GetTimestamp() - tRestore) * 1000.0 / Stopwatch.Frequency;
        result.WongoPoints = wongo;

        long tSetup = Stopwatch.GetTimestamp();
        RunState run = RunState.FromSerializable(save);
        if (run.Players.Count != 1)
        {
            throw new EpisodeAbortException(AbortReason.StartFailed, "bad_save", $"Shadow save expects 1 player, got {run.Players.Count}.");
        }
        result.Ascension = run.AscensionLevel;
        result.Character = run.Players[0].Character.Id.Entry;
        rm.ForceDiscoveryOrderModifications = run.GameMode == GameMode.Standard && !run.IsCustomGame;
        DriveStartTask(rm.SetUpSavedSingleplayer(run, save), "SetUpSavedSingleplayer");
        rm.ShouldSave = false;
        result.RunSetUpMs = (Stopwatch.GetTimestamp() - tSetup) * 1000.0 / Stopwatch.Frequency;

        foreach (ActModel a in run.Acts)
        {
            result.Acts.Add(a.Id.Entry);
        }
        result.Characters.Clear();
        for (int s = 0; s < run.Players.Count; s++)
        {
            result.Characters.Add(run.Players[s].Character.Id.Entry);
        }

        _ctx = new RunContext(_session, this, spec, _opt, run, seed);
        DriverPatches.ResetIntentTracker();
        rm.RoomEntered += _roomEntered;
        CombatManager.Instance.CombatWon += _combatWon;
        _combatWonSubscribed = true;
        foreach (IRunLifecycle l in _session.Registry.Lifecycles)
        {
            _phaseNote = "OnRunSetUp:" + l.Name;
            l.OnRunSetUp(_ctx);
        }
        foreach (IEpisodeCheck check in _session.Registry.EpisodeChecks)
        {
            _phaseNote = "OnEpisodeStart:" + check.Name;
            check.OnEpisodeStart(_ctx);
        }
        _phaseNote = "";

        long tEnter = Stopwatch.GetTimestamp();
        rm.Launch();
        DriveStartTask(rm.GenerateMap(), "GenerateMap");
        AbstractRoom? preFinished = save.PreFinishedRoom != null ? AbstractRoom.FromSerializable(save.PreFinishedRoom, run) : null;
        DriveStartTask(rm.LoadIntoLatestMapCoord(preFinished), "LoadIntoLatestMapCoord");
        FillWinHistory(run, _c);
        result.EnterActMs = (Stopwatch.GetTimestamp() - tEnter) * 1000.0 / Stopwatch.Frequency;
    }

    private static void FillWinHistory(RunState run, EpisodeCounters c)
    {
        c.CombatsWon = 0;
        c.ElitesWon = 0;
        c.BossesWon = 0;
        c.LastWonRoomType = -1;
        c.LastWonTotalFloor = -1;
        bool inProgress = CombatManager.Instance.IsInProgress;
        int floor = 0;
        int lastFloor = -1;
        RoomType lastType = RoomType.Unassigned;
        List<(RoomType type, int floor)> won = new List<(RoomType, int)>();
        foreach (IReadOnlyList<MapPointHistoryEntry> act in run.MapPointHistory)
        {
            foreach (MapPointHistoryEntry e in act)
            {
                floor++;
                foreach (MapPointRoomHistoryEntry r in e.Rooms)
                {
                    if (r.RoomType == RoomType.Monster || r.RoomType == RoomType.Elite || r.RoomType == RoomType.Boss)
                    {
                        won.Add((r.RoomType, floor));
                    }
                }
            }
        }
        if (inProgress && won.Count > 0)
        {
            won.RemoveAt(won.Count - 1);
        }
        foreach ((RoomType type, int f) in won)
        {
            c.CombatsWon++;
            if (type == RoomType.Elite) c.ElitesWon++;
            else if (type == RoomType.Boss) c.BossesWon++;
            lastType = type;
            lastFloor = f;
        }
        if (won.Count > 0)
        {
            c.LastWonRoomType = (int)lastType;
            c.LastWonTotalFloor = lastFloor;
        }
    }

    // =====================================================================================================
    //  Run start, 2-4 seats (phase H5): the real multiplayer chain with the host's own Host-type net service
    //    StartRunLobby.BeginRunLocally (:492-528: act selection with isMultiplayer, union unlock state; buffering on)
    //    -> GeneratePlayers (:949-960: one Player per lobby member, its own unlock state, net id = lobby id)
    //    -> NGame.StartNewMultiplayerRun (NGame.cs:878-886: acts.ToMutable, RunState.CreateForNewRun, SetUpNewMultiplayer)
    //    -> NGame.StartRun (:903-914).
    //  SetUpNewMultiplayer needs a StartRunLobby, whose service must be a NetHostGameService with a transport
    //  (StartRunLobby.cs:130, :324, :335). RunManager.SetUpTest(state, service, ...) (RunManager.cs:355-374) is the same
    //  set-up for an arbitrary INetGameService; its three differences are closed here: it does not call GenerateRooms
    //  (called right after it, as SetUpNewMultiplayer does :291-292), it would disable the combat-state sync (kept on:
    //  disableCombatStateSync: false, as SetUpNewMultiplayer leaves it), and it creates its own PeerInputSynchronizer
    //  instead of the lobby's (mouse / hover flavour only).
    // =====================================================================================================

    /// <summary>The game's lobby size (NMultiplayerHostSubmenu: InitializeMultiplayerAsHost(netService, 4)).</summary>
    public const int MaxSeats = 4;

    private RunState SetUpMultiSeatRun(EpisodeSpec spec, EpisodeResult result, RunManager rm, string seed, out Player[] players, out HeadlessHostNetService hostService)
    {
        int seatCount = spec.Characters.Length;
        CharacterModel[] characters = new CharacterModel[seatCount];
        for (int s = 0; s < seatCount; s++)
        {
            CharacterModel? character = null;
            foreach (CharacterModel c in ModelDb.AllCharacters)
            {
                if (c.Id.Entry.Equals(spec.Characters[s], StringComparison.OrdinalIgnoreCase))
                {
                    character = c;
                }
            }
            // The same character on several seats is legal: the lobby's ChangeCharacter has no uniqueness check (StartRunLobby.cs:448-464).
            characters[s] = character ?? throw new EpisodeAbortException(AbortReason.StartFailed, "bad_spec", "Unknown character on seat " + s + ": " + spec.Characters[s]);
        }

        // ---- StartRunLobby.BeginRunLocally ----
        long tActs = Stopwatch.GetTimestamp();
        // One unlock state per lobby member (AddLocalHostPlayer / the join request carry new UnlockState(Progress); GeneratePlayers
        // round-trips it): every member has the pinned veteran profile.
        UnlockState[] unlocks = new UnlockState[seatCount];
        for (int s = 0; s < seatCount; s++)
        {
            unlocks[s] = VeteranProfile.CreateUnlockState();
        }
        UnlockState lobbyUnlock = new UnlockState(unlocks);                             // StartRunLobby.GetUnlockState() (:967-974): the union
        Rng actRng = new Rng(StringHelper.GetDeterministicHashCode(seed), "act_selection");
        List<ActModel> acts = ActModel.GetRandomList(actRng, lobbyUnlock, isMultiplayer: true).ToList();   // :494 NetService.Type.IsMultiplayer()
        // list[0] = GetAct(Act1) ?? list[0]: no override ("random"). Random / rotating characters would be drawn from actRng here
        // (:497-510); the host only accepts concrete characters, so nothing is drawn.
        // Ascension: the lobby's MaxAscension is the minimum of the members' maxStandardAscension (:299-312), each member's being its
        // Progress.MaxMultiplayerAscension (NCharacterSelectScreen.cs:273, JoinFlow.cs:174). No per-character clamp in multiplayer (:511).
        int ascension = Math.Min(spec.Ascension, SaveManager.Instance.Progress.MaxMultiplayerAscension);
        if (ascension < 0)
        {
            ascension = 0;
        }
        result.Ascension = ascension;
        result.Character = characters[0].Id.Entry;
        result.ActSelectionMs = (Stopwatch.GetTimestamp() - tActs) * 1000.0 / Stopwatch.Frequency;

        // ---- StartRunLobby.GeneratePlayers / NGame.StartNewMultiplayerRun ----
        players = new Player[seatCount];
        for (int s = 0; s < seatCount; s++)
        {
            // Net ids: the lobby uses the platform ids of its members; any distinct ids do, as long as seat order and net-id order
            // agree (DeterministicModelComparer.cs:62 orders identical models of different players by net id). The host takes the
            // ids of a real ENet session (MultiSeatRun.NetIdOfSeat): the host player 1, the k-th client 1000 + k.
            Player player = Player.CreateForNewRun(characters[s], unlocks[s], MultiSeatRun.NetIdOfSeat(s));
            int forcedHp = ForcedHpOf(s);
            if (forcedHp > 0)
            {
                player.Creature.SetMaxHpInternal(forcedHp);
                player.Creature.SetCurrentHpInternal(forcedHp);
            }
            players[s] = player;
        }
        long tSetup = Stopwatch.GetTimestamp();
        RunState run = RunState.CreateForNewRun(players, acts.Select(a => a.ToMutable()).ToList(), Array.Empty<ModifierModel>(), GameMode.Standard, false, ascension, seed);
        rm.ForceDiscoveryOrderModifications = run.GameMode == GameMode.Standard && !run.IsCustomGame;
        hostService = new HeadlessHostNetService(players[0].NetId);                     // a fresh service per run: RunManager.CleanUp does not dispose every synchronizer
        hostService.WireRoundTrip = _opt.WireRoundTrip;
        hostService.SetBufferMessages(true);                                            // StartRunLobby.BeginRunLocally (:526); RunManager.Launch releases (:597)
        rm.SetUpTest(run, hostService, disableCombatStateSync: false, shouldSave: false);
        rm.GenerateRooms();
        result.RunSetUpMs = (Stopwatch.GetTimestamp() - tSetup) * 1000.0 / Stopwatch.Frequency;

        // Fail closed: the set-up must be a real multiplayer set-up on our service.
        if (!ReferenceEquals(rm.NetService, hostService) || rm.IsSingleplayerOrFakeMultiplayer || rm.RunLobby == null)
        {
            throw new EpisodeAbortException(AbortReason.StartFailed, "multi_seat_setup", "RunManager.SetUpTest did not produce a Host-type multiplayer run on the host's net service.");
        }
        int lobbyMembers = 0;
        foreach (ulong id in rm.RunLobby.PlayerIds)
        {
            if (lobbyMembers >= seatCount || id != players[lobbyMembers].NetId)
            {
                throw new EpisodeAbortException(AbortReason.StartFailed, "multi_seat_setup", "The run lobby does not list the seats in order.");
            }
            lobbyMembers++;
        }
        if (lobbyMembers != seatCount || run.Players.Count != seatCount || rm.CombatStateSynchronizer.IsDisabled || run.Modifiers.Count != 0)
        {
            throw new EpisodeAbortException(AbortReason.StartFailed, "multi_seat_setup",
                $"Unexpected multiplayer set-up: {lobbyMembers} lobby members, {run.Players.Count} players, combat-state sync disabled = {rm.CombatStateSynchronizer.IsDisabled}, {run.Modifiers.Count} modifiers.");
        }
        return run;
    }

    /// <summary>TEST ONLY: the forced HP of a seat (EpisodeOptions.DebugForcedHpPerSeat, else DebugForcedHp); 0 = natural.</summary>
    private int ForcedHpOf(int seat)
    {
        int[]? perSeat = _opt.DebugForcedHpPerSeat;
        if (perSeat != null && seat < perSeat.Length && perSeat[seat] > 0)
        {
            return perSeat[seat];
        }
        return _opt.DebugForcedHp;
    }

    /// <summary>After RunManager.Launch (:594-601): the local player is seat 0, and the service released its buffered messages.</summary>
    private static void AssertMultiSeatLaunched(RunContext ctx, RunManager rm)
    {
        Player? me = MegaCrit.Sts2.Core.Context.LocalContext.GetMe(ctx.Run);
        if (!ReferenceEquals(me, ctx.Run.Players[0]) || rm.ActionQueueSynchronizer.CombatState != MegaCrit.Sts2.Core.Entities.Multiplayer.ActionSynchronizerCombatState.NotInCombat)
        {
            throw new EpisodeAbortException(AbortReason.StartFailed, "multi_seat_launch", "After Launch the local player is not seat 0, or the action synchronizer is not idle.");
        }
    }

    /// <summary>
    /// Runs a start-chain task to completion. Only pending frames may ask for input here (a starting relic that prompts;
    /// a scenario Setup that offers rewards or jumps into a combat). The episode limits apply here exactly as in the
    /// main loop (rooms R4 / R7): MaxPolicyDecisions, MaxCombatTurns -> Truncated; DecisionCap -> Livelock.
    /// </summary>
    private void DriveStartTask(Task task, string what)
    {
        _phaseNote = what;
        Track(task, what, blocking: true, seat: -1);
        while (true)
        {
            Settle();
            if (task.IsCompleted)
            {
                break;
            }
            string? limit = LimitReached(_ctx!.Run);
            if (limit != null)
            {
                throw new EpisodeTruncatedException(limit);
            }
            CheckDecisionCap();
            if (_stack.Count == 0)
            {
                // Multi-seat: a reward set that the start step opened for a remote seat (its overlay screen) may ask as well.
                if (_ctx.SeatCount > 1 && TryRemoteOverlays(_ctx, out Decision? overlay))
                {
                    if (overlay != null)
                    {
                        _handlerSpin = 0;
                        AskAndApply(overlay);
                    }
                    _phase = Phase.Start;
                    continue;
                }
                throw new EpisodeAbortException(AbortReason.StartFailed, "stall:" + what, "Run start step '" + what + "' did not complete and nothing is asking for input. " + Diagnose());
            }
            Decision? top = BuildTop();
            if (top != null)
            {
                AskAndApply(top);
            }
            _phase = Phase.Start;
        }
        _phaseNote = "";
    }

    // =====================================================================================================
    //  Main loop
    // =====================================================================================================

    private void MainLoop(EpisodeResult result)
    {
        RunContext ctx = _ctx!;
        RunState run = ctx.Run;
        Scenario? scenario = ctx.Spec.Scenario;
        while (true)
        {
            Settle();

            // "Room entered and settled" (replay R3): before the handler's first Step and before any pending frame is answered.
            if (_roomBoundaryPending && _observerStarted)
            {
                _roomBoundaryPending = false;
                _observer?.OnRoomBoundary(ctx);
            }

            // T: terminal
            if (run.IsGameOver)
            {
                result.Outcome = run.CurrentRoom?.IsVictoryRoom == true ? EpisodeOutcome.Win : EpisodeOutcome.Loss;
                return;
            }
            if (scenario?.StopWhen != null && scenario.StopWhen(ctx))
            {
                result.Outcome = EpisodeOutcome.Truncated;
                result.TruncatedBy = "scenario";
                return;
            }
            string? limit = LimitReached(run);
            if (limit != null)
            {
                result.Outcome = EpisodeOutcome.Truncated;
                result.TruncatedBy = limit;
                return;
            }
            CheckDecisionCap();

            Decision decision;
            if (_stack.Count > 0)
            {
                // P1: a request of the game is parked on the stack; the top frame is always answered first.
                Decision? top = BuildTop();
                if (top == null)
                {
                    continue;
                }
                decision = top;
            }
            else if (ctx.SeatCount > 1)
            {
                // Multi-seat (phase H5): remote overlays, then combat (union), then the seats' room chains, votes last.
                Decision? next = NextMultiSeatDecision(ctx, run);
                if (next == null)
                {
                    continue;
                }
                decision = next;
            }
            else
            {
                RoomScreen screen = Classify(run, ctx);
                if (screen != RoomScreen.Combat && HasBlockingOperation)
                {
                    throw new EpisodeAbortException(AbortReason.Stall, "pending_operation",
                        "Engine is quiescent, the decision stack is empty, but a tracked operation has not completed: " + DescribeOps() + ". " + Diagnose());
                }
                if (screen == RoomScreen.None)
                {
                    throw new EpisodeAbortException(AbortReason.Stall, "unknown_screen:" + (run.CurrentRoom?.GetType().Name ?? "null"), "No recognisable screen. " + Diagnose());
                }
                IRoomHandler? handler = _session.Registry.GetRoomHandler(screen);
                if (handler == null)
                {
                    throw new EpisodeAbortException(AbortReason.Stall, "no_handler:" + screen, "No room handler is registered for " + screen + ". " + Diagnose());
                }
                _phase = Phase.Build;
                _phaseNote = handler.Name;
                long tb = Stopwatch.GetTimestamp();
                int prof = HostProfile.On ? HostProfile.Enter(HostProfile.Build) : 0;
                HandlerStep step = handler.Step(ctx, 0);
                if (HostProfile.On)
                {
                    HostProfile.Enter(prof);
                }
                _c.TicksBuild += Stopwatch.GetTimestamp() - tb;
                _phaseNote = "";
                if (step.Kind == StepKind.Progressed)
                {
                    _c.HandlerProgressSteps++;
                    if (++_handlerSpin > MaxHandlerProgressSteps)
                    {
                        throw new EpisodeAbortException(AbortReason.Livelock, "handler_spin:" + handler.Name, $"Handler '{handler.Name}' reported progress {MaxHandlerProgressSteps} times in a row without a decision. " + Diagnose());
                    }
                    continue;
                }
                if (step.Kind == StepKind.None || step.Decision == null)
                {
                    // (A single seat that "waits for a teammate" has nobody to wait for: Waiting is a stall here.)
                    throw new EpisodeAbortException(AbortReason.Stall, "handler_unrecognised:" + handler.Name, "Handler '" + handler.Name + "' does not recognise the state: " + (step.Note ?? "-") + ". " + Diagnose());
                }
                decision = step.Decision;
                if (decision.Source == null)
                {
                    decision.Source = handler;
                }
            }
            _handlerSpin = 0;
            AskAndApply(decision);
        }
    }

    // =====================================================================================================
    //  Multi-seat linearisation (phase H5; DRIVER-API.md section 12)
    // =====================================================================================================

    /// <summary>
    /// The next decision of a run with 2-4 seats when no frame is parked on the decision stack (frames are P1 for
    /// every seat: a remote seat's prompt is a frame like seat 0's):
    ///   P1'  a remote seat's overlay screen (a reward set the game opened for it), lowest seat first - the counterpart
    ///        of seat 0's reward frames;
    ///   P2   combat: one decision, the union over the seats;
    ///   P3   the room chains, seat-major: the lowest seat that has a decision on its own room-level screen is asked.
    ///        A seat that waits for a teammate is skipped (StepKind.Waiting), and so is a seat with an incomplete
    ///        blocking operation of its own. A seat whose next decision is a team vote is skipped while any seat has
    ///        something else to do (lead ruling Q5: votes come last);
    ///   P4   the team votes (map, shared-event page), lowest seat first.
    /// Nothing to ask is the only out-of-combat stall of a multi-seat run: all_seats_waiting, with what every seat
    /// waits for. Returns null when a handler reported progress (settle and look again).
    /// </summary>
    private Decision? NextMultiSeatDecision(RunContext ctx, RunState run)
    {
        if (TryRemoteOverlays(ctx, out Decision? overlay))
        {
            return overlay;
        }

        if (CombatManager.Instance.IsInProgress)
        {
            IRoomHandler combat = _session.Registry.GetRoomHandler(RoomScreen.Combat)
                ?? throw new EpisodeAbortException(AbortReason.Stall, "no_handler:" + RoomScreen.Combat, "No room handler is registered for " + RoomScreen.Combat + ". " + Diagnose());
            HandlerStep step = RunStep(combat, ctx, -1);
            if (step.Kind == StepKind.Progressed)
            {
                CountProgress(combat.Name);
                return null;
            }
            if (step.Kind != StepKind.Decision || step.Decision == null)
            {
                throw new EpisodeAbortException(AbortReason.Stall, "handler_unrecognised:" + combat.Name, "Handler '" + combat.Name + "' does not recognise the state: " + (step.Note ?? "-") + ". " + Diagnose());
            }
            step.Decision.Source ??= combat;
            return step.Decision;
        }

        int seats = ctx.SeatCount;
        StringBuilder waits = new StringBuilder(96);
        int voteSeat = -1;
        IRoomHandler? voteHandler = null;
        for (int s = 0; s < seats; s++)
        {
            if (ctx.IsActReady(s))
            {
                waits.Append(" seat").Append(s).Append("=ready for the next act;");
                continue;
            }
            RoomScreen screen = ClassifySeat(run, ctx, s);
            if (screen == RoomScreen.None)
            {
                throw new EpisodeAbortException(AbortReason.Stall, "unknown_screen:" + (run.CurrentRoom?.GetType().Name ?? "null"), "No recognisable screen for seat " + s + ". " + Diagnose());
            }
            if (HasBlockingOperationFor(s))
            {
                waits.Append(" seat").Append(s).Append("=operation pending;");
                continue;
            }
            IRoomHandler handler = _session.Registry.GetRoomHandler(screen)
                ?? throw new EpisodeAbortException(AbortReason.Stall, "no_handler:" + screen, "No room handler is registered for " + screen + ". " + Diagnose());
            HandlerStep step = RunStep(handler, ctx, s);
            switch (step.Kind)
            {
                case StepKind.Progressed:
                    CountProgress(handler.Name);
                    return null;
                case StepKind.Decision:
                    if (step.Decision == null)
                    {
                        throw new EpisodeAbortException(AbortReason.HandlerFault, "step_without_decision:" + handler.Name, "Handler '" + handler.Name + "' returned a Decision step without a decision.");
                    }
                    step.Decision.Source ??= handler;
                    return step.Decision;
                case StepKind.Vote:
                    if (voteSeat < 0)
                    {
                        voteSeat = s;
                        voteHandler = handler;
                    }
                    waits.Append(" seat").Append(s).Append("=at a team vote;");
                    break;
                case StepKind.Waiting:
                    waits.Append(" seat").Append(s).Append('=').Append(step.Note ?? "waiting").Append(';');
                    break;
                default:
                    throw new EpisodeAbortException(AbortReason.Stall, "handler_unrecognised:" + handler.Name, "Handler '" + handler.Name + "' does not recognise the state of seat " + s + ": " + (step.Note ?? "-") + ". " + Diagnose());
            }
        }
        if (voteSeat >= 0)
        {
            // Every seat is at its vote or waits for one. The step is built again: handlers reuse their decision objects,
            // and a later seat's Step of the same handler has overwritten what the first pass returned for this seat.
            HandlerStep step = RunStep(voteHandler!, ctx, voteSeat);
            if (step.Kind == StepKind.Progressed)
            {
                CountProgress(voteHandler!.Name);
                return null;
            }
            if (step.Kind != StepKind.Vote || step.Decision == null)
            {
                throw new EpisodeAbortException(AbortReason.HandlerFault, "vote_step_changed:" + voteHandler!.Name, "Handler '" + voteHandler.Name + "' returned a team vote for seat " + voteSeat + " and then " + step.Kind + " for the same state.");
            }
            step.Decision.Source ??= voteHandler;
            return step.Decision;
        }
        throw new EpisodeAbortException(AbortReason.Stall, "all_seats_waiting",
            "Engine is quiescent, the decision stack is empty and no seat has a decision:" + waits + " operations: " + DescribeOps() + ". " + Diagnose());
    }

    /// <summary>The overlay screens of the remote seats (P1'). True = something happened: a decision to ask, or (null) progress.</summary>
    private bool TryRemoteOverlays(RunContext ctx, out Decision? decision)
    {
        decision = null;
        IRemoteOverlay? overlay = _session.Registry.RemoteOverlay;
        if (overlay == null)
        {
            return false;
        }
        for (int s = 1; s < ctx.SeatCount; s++)
        {
            _phase = Phase.Build;
            _phaseNote = overlay.Name;
            long tb = Stopwatch.GetTimestamp();
            int prof = HostProfile.On ? HostProfile.Enter(HostProfile.Build) : 0;
            HandlerStep? os = overlay.Step(ctx, s);
            if (HostProfile.On)
            {
                HostProfile.Enter(prof);
            }
            _c.TicksBuild += Stopwatch.GetTimestamp() - tb;
            _phaseNote = "";
            if (!os.HasValue)
            {
                continue;
            }
            HandlerStep step = os.Value;
            switch (step.Kind)
            {
                case StepKind.Progressed:
                    CountProgress(overlay.Name);
                    return true;
                case StepKind.Decision:
                    if (step.Decision == null)
                    {
                        throw new EpisodeAbortException(AbortReason.HandlerFault, "step_without_decision:" + overlay.Name, "The remote overlay returned a Decision step without a decision.");
                    }
                    step.Decision.Source ??= overlay;
                    decision = step.Decision;
                    return true;
                case StepKind.Waiting:
                    break;
                default:
                    throw new EpisodeAbortException(AbortReason.Stall, "handler_unrecognised:" + overlay.Name, "The overlay screen of seat " + s + " is in a state its owner does not recognise: " + (step.Note ?? "-") + ". " + Diagnose());
            }
        }
        return false;
    }

    private HandlerStep RunStep(IRoomHandler handler, RunContext ctx, int seat)
    {
        _phase = Phase.Build;
        _phaseNote = handler.Name;
        long tb = Stopwatch.GetTimestamp();
        int prof = HostProfile.On ? HostProfile.Enter(HostProfile.Build) : 0;
        HandlerStep step = handler.Step(ctx, seat);
        if (HostProfile.On)
        {
            HostProfile.Enter(prof);
        }
        _c.TicksBuild += Stopwatch.GetTimestamp() - tb;
        _phaseNote = "";
        return step;
    }

    private void CountProgress(string name)
    {
        _c.HandlerProgressSteps++;
        if (++_handlerSpin > MaxHandlerProgressSteps)
        {
            throw new EpisodeAbortException(AbortReason.Livelock, "handler_spin:" + name, $"Handler '{name}' reported progress {MaxHandlerProgressSteps} times in a row without a decision. " + Diagnose());
        }
    }

    /// <summary>
    /// The room-level screen of ONE seat (multi-seat; no combat in progress, no frame, no overlay): the map once the
    /// seat's map is open, else the screen of the current room. A seat inside a CombatRoom whose combat is over has no
    /// room-level screen: its reward screen was an overlay / frame, and its PROCEED opened its map or cast its act vote.
    /// </summary>
    internal static RoomScreen ClassifySeat(RunState run, RunContext ctx, int seat)
    {
        AbstractRoom? room = run.CurrentRoom;
        if (ctx.IsMapOpen(seat) || room is MapRoom)
        {
            return RoomScreen.Map;
        }
        return room switch
        {
            EventRoom => RoomScreen.Event,
            MerchantRoom => RoomScreen.Shop,
            RestSiteRoom => RoomScreen.Rest,
            TreasureRoom => RoomScreen.Treasure,
            _ => RoomScreen.None,
        };
    }

    /// <summary>The configured limits that end an episode as Truncated: "max_policy_decisions", "combat_turn_cap"; null = none reached.</summary>
    private string? LimitReached(RunState run)
    {
        if (_opt.MaxPolicyDecisions > 0 && _c.PolicyDecisions >= _opt.MaxPolicyDecisions)
        {
            return "max_policy_decisions";
        }
        if (_opt.MaxCombatTurns > 0 && CombatManager.Instance.IsInProgress)
        {
            int turn = 0;
            IReadOnlyList<Player> seats = run.Players;
            for (int i = 0; i < seats.Count; i++)
            {
                int t = seats[i].PlayerCombatState?.TurnNumber ?? 0;
                if (t > turn)
                {
                    turn = t;
                }
            }
            if (turn > _opt.MaxCombatTurns)
            {
                return "combat_turn_cap";
            }
        }
        return null;
    }

    private void CheckDecisionCap()
    {
        if (_c.DecisionPoints >= _opt.DecisionCap)
        {
            throw new EpisodeAbortException(AbortReason.Livelock, "decision_cap", $"Episode reached {_c.DecisionPoints} decision points. " + Diagnose());
        }
    }

    /// <summary>The central screen classifier (used only when the decision stack is empty).</summary>
    internal static RoomScreen Classify(RunState run, RunContext ctx)
    {
        if (CombatManager.Instance.IsInProgress)
        {
            return RoomScreen.Combat;
        }
        AbstractRoom? room = run.CurrentRoom;
        if (ctx.MapOpen || room is MapRoom)
        {
            return RoomScreen.Map;
        }
        return room switch
        {
            EventRoom => RoomScreen.Event,
            MerchantRoom => RoomScreen.Shop,
            RestSiteRoom => RoomScreen.Rest,
            TreasureRoom => RoomScreen.Treasure,
            _ => RoomScreen.None, // e.g. a CombatRoom whose combat is over but neither rewards nor the map were opened
        };
    }

    /// <summary>Asks the top frame for its decision. Null = the frame popped itself (its request became moot); classify again.</summary>
    private Decision? BuildTop()
    {
        IPendingDecision frame = _stack[_stack.Count - 1];
        _phase = Phase.Build;
        _phaseNote = frame.Name;
        long tb = Stopwatch.GetTimestamp();
        int prof = HostProfile.On ? HostProfile.Enter(HostProfile.Build) : 0;
        Decision? d = frame.Build(_ctx!);
        if (HostProfile.On)
        {
            HostProfile.Enter(prof);
        }
        _c.TicksBuild += Stopwatch.GetTimestamp() - tb;
        _phaseNote = "";
        if (d == null)
        {
            if (_stack.Contains(frame))
            {
                throw new EpisodeAbortException(AbortReason.Stall, "frame_built_nothing:" + frame.Name, "Pending frame '" + frame.Name + "' returned no decision but is still on the stack. " + Diagnose());
            }
            _c.HandlerProgressSteps++;
            if (++_handlerSpin > MaxHandlerProgressSteps)
            {
                throw new EpisodeAbortException(AbortReason.Livelock, "handler_spin:" + frame.Name, "Pending frames keep removing themselves without a decision. " + Diagnose());
            }
            return null;
        }
        if (d.Source == null)
        {
            d.Source = frame;
        }
        return d;
    }

    private void AskAndApply(Decision decision)
    {
        ExtendRoomFrame(decision);
        int index = Ask(decision, nested: false);
        IDecisionSource source = decision.Actions[index].Owner ?? decision.Source
            ?? throw new EpisodeAbortException(AbortReason.HandlerFault, "no_source", "Decision without a Source: " + decision);
        _phase = Phase.Apply;
        _phaseNote = source.Name;
        long ta = Stopwatch.GetTimestamp();
        int prof = HostProfile.On ? HostProfile.Enter(HostProfile.Apply) : 0;
        source.Apply(_ctx!, decision, index);
        if (HostProfile.On)
        {
            HostProfile.Enter(prof);
        }
        _c.TicksApply += Stopwatch.GetTimestamp() - ta;
        _phaseNote = "";
    }

    /// <summary>Room-level frames (DECISIONS-H1 B.7) are offered to the registered frame extensions before they are sealed.</summary>
    private void ExtendRoomFrame(Decision d)
    {
        IReadOnlyList<IRoomFrameExtension> extensions = _session.Registry.FrameExtensions;
        if (extensions.Count == 0 || d.Sealed)
        {
            return;
        }
        switch (d.Kind)
        {
            case DecisionKind.Map:
            case DecisionKind.Shop:
            case DecisionKind.Rest:
            case DecisionKind.Event:
            case DecisionKind.Reward:
            case DecisionKind.Treasure:
                break;
            default:
                return;
        }
        List<LegalAction> actions = d.Actions;
        for (int e = 0; e < extensions.Count; e++)
        {
            IRoomFrameExtension ext = extensions[e];
            int before = actions.Count;
            _phase = Phase.Build;
            _phaseNote = ext.Name;
            int prof = HostProfile.On ? HostProfile.Enter(HostProfile.Build) : 0;
            ext.Extend(_ctx!, d);
            if (HostProfile.On)
            {
                HostProfile.Enter(prof);
            }
            _phaseNote = "";
            for (int i = before; i < actions.Count; i++)
            {
                LegalAction a = actions[i];
                a.Owner = ext;
                actions[i] = a;
            }
        }
    }

    // =====================================================================================================
    //  Ask: the one path every decision takes (pump-driven and nested alike)
    // =====================================================================================================

    internal int AskNested(Decision decision)
    {
        if (_ctx == null)
        {
            throw new InvalidOperationException("No episode is running.");
        }
        if (!_pump.IsQueueEmpty)
        {
            _c.NestedWithQueuedWork++;
        }
        Phase saved = _phase;
        string savedNote = _phaseNote;
        int index = Ask(decision, nested: true);
        _phase = saved;
        _phaseNote = savedNote;
        return index;
    }

    /// <summary>True when Ask will answer the (sealed) decision itself, without the policy: the AutoGold / AutoSingle rules below.</summary>
    private bool IsAutoAdvanced(Decision d)
    {
        List<LegalAction> actions = d.Actions;
        if (_opt.AutoGold)
        {
            for (int i = 0; i < actions.Count; i++)
            {
                if ((actions[i].Flags & ActionFlags.AutoGold) != 0)
                {
                    return true;
                }
            }
        }
        return _opt.AutoSingle && actions.Count == 1;
    }

    private int Ask(Decision d, bool nested)
    {
        int prof = HostProfile.On ? HostProfile.Enter(HostProfile.AskOther) : 0;
        try
        {
            return AskCore(d, nested);
        }
        finally
        {
            if (HostProfile.On)
            {
                HostProfile.Enter(prof);
            }
        }
    }

    private int AskCore(Decision d, bool nested)
    {
        RunContext ctx = _ctx!;
        d.Seal(_opt.AllowAbandon);
        if (d.Actions.Count == 0)
        {
            throw new EpisodeAbortException(AbortReason.Stall, "no_legal_action:" + d.Kind, "Decision has no legal action: " + d + ". " + Diagnose());
        }
        if (nested && _opt.MaxPolicyDecisions > 0 && _c.PolicyDecisions >= _opt.MaxPolicyDecisions && !IsAutoAdvanced(d))
        {
            // Verifier finding R-1: the main loop and the start chain check the cap before they build a decision, but a decision
            // asked from inside an action (DecideNested: the card reward that a REWARD pick opens) came straight here, so a cap of
            // N could yield N + 1 policy decisions. The cap must yield exactly N (DRIVER-API.md 5.3): the policy is not asked and
            // the decision is not counted; the episode ends as Truncated exactly as in the main loop. The exception unwinds
            // through the game's task that asked (RethrowIfAbort restores it).
            throw new EpisodeTruncatedException("max_policy_decisions");
        }
        d.Sequence = _c.DecisionPoints++;
        d.Nested = nested;
        d.Episode = ctx;
        _c.DecisionsByKind[(int)d.Kind]++;
        for (int mask = d.VerbMask, v = 0; mask != 0; mask >>= 1, v++)
        {
            if ((mask & 1) != 0)
            {
                _c.VerbOffered[v]++;
            }
        }
        if (SpireHost.Patches.LivelockGuard.Tripped is string tripped)
        {
            // A livelock bound was passed inside game code. Its exception normally surfaces through the task that was running (the
            // fault checks of the settle that follows). If game code swallowed it and the engine came back to a decision point all
            // the same, the episode still ends here: fail closed, never a decision on a state the guard interrupted.
            throw new EpisodeAbortException(AbortReason.Livelock, tripped, "A livelock bound of the engine was passed (" + tripped + ") and the exception thrown at the site did not surface. " + Diagnose());
        }
        SpireHost.Patches.LivelockGuard.Reset();
        _lastDecision = d.Kind.ToString();
        long depth = EngineThread.StackDepthBytes();
        if (depth > _c.MaxStackBytes)
        {
            _c.MaxStackBytes = depth;
        }

        int index = -1;
        bool auto = false;
        List<LegalAction> actions = d.Actions;
        if (_opt.AutoGold)
        {
            for (int i = 0; i < actions.Count; i++)
            {
                if ((actions[i].Flags & ActionFlags.AutoGold) != 0)
                {
                    index = i;
                    auto = true;
                    _c.AutoGold++;
                    break;
                }
            }
        }
        if (!auto && _opt.AutoSingle && actions.Count == 1)
        {
            index = 0;
            auto = true;
            _c.AutoSingle[(int)d.Kind]++;
        }

        if (auto)
        {
            if (++_autoRun > MaxAutoAdvanceRun)
            {
                throw new EpisodeAbortException(AbortReason.Livelock, "auto_advance_spin:" + d.Kind, $"{MaxAutoAdvanceRun} auto-advanced decisions in a row. " + Diagnose());
            }
        }
        else
        {
            _autoRun = 0;
            bool sign = _opt.ComputeSignatures || _traceList != null;
            if (sign)
            {
                long ts = Stopwatch.GetTimestamp();
                ulong sig = RunSignature.Compute(ctx.Run) ^ d.ShapeHash();
                _trace.Add(sig);
                _traceList?.Add(sig);
                _c.TicksSignature += Stopwatch.GetTimestamp() - ts;
            }
            _phase = Phase.Ask;
            long tp = Stopwatch.GetTimestamp();
            index = _policy.Decide(d);
            _c.TicksPolicy += Stopwatch.GetTimestamp() - tp;
            if ((uint)index >= (uint)actions.Count)
            {
                throw new EpisodeAbortException(AbortReason.PolicyFault, "bad_index", $"Policy answered {index} for a decision with {actions.Count} actions ({d}).");
            }
            _c.PolicyDecisions++;
            _c.PolicyByKind[(int)d.Kind]++;
            if (nested)
            {
                _c.NestedDecisions++;
            }
            if (sign)
            {
                _trace.Add(index);
            }
            _actionLog?.Add(index);
        }
        _c.VerbChosen[(int)actions[index].Verb]++;
        _observer?.OnDecision(ctx, d, index, auto);
        return index;
    }

    // =====================================================================================================
    //  Settle: pump to quiescence (strict)
    // =====================================================================================================

    private void Settle()
    {
        long t0 = Stopwatch.GetTimestamp();
        int prof = HostProfile.On ? HostProfile.Enter(HostProfile.Settle) : 0;
        try
        {
            SettleCore();
        }
        finally
        {
            if (HostProfile.On)
            {
                HostProfile.Enter(prof);
            }
            _c.TicksSettle += Stopwatch.GetTimestamp() - t0;
        }
    }

    private void SettleCore()
    {
        _phase = Phase.Settle;
        VirtualClock clock = _pump.Clock;
        long executed = 0;
        int waiterRounds = 0;
        long virtualStart = clock.NowTicks;
        while (true)
        {
            if (!_pump.TryRunUntilQuiescent(MaxCallbacksPerSettle - executed, out long n))
            {
                _c.PumpCallbacks += executed + n;
                throw new EpisodeAbortException(AbortReason.Livelock, "pump_callbacks", $"More than {MaxCallbacksPerSettle} continuations in one settle. " + Diagnose());
            }
            executed += n;
            CheckFaults();
            ObserveOperations();
            if (_pump.QueueCount > 0)
            {
                continue;
            }
            if (clock.Pending > 0 && _stack.Count == 0)
            {
                if (clock.NowTicks - virtualStart > VirtualTimeBudget.Ticks)
                {
                    _c.PumpCallbacks += executed;
                    throw new EpisodeAbortException(AbortReason.Livelock, "virtual_time_budget",
                        $"A poll loop consumed {VirtualTimeBudget.TotalSeconds:0} virtual seconds without the engine becoming quiescent (paused executor with queued actions?). " + Diagnose());
                }
                clock.FireNext();
                _c.VirtualTimerFires++;
                continue;
            }
            if (_quiescenceWaiters.Count > 0)
            {
                // Quiescent: release host code that awaits RunContext.Quiescence(); it runs inline and may start
                // new engine work, so settle again afterwards.
                if (++waiterRounds > MaxHandlerProgressSteps)
                {
                    _c.PumpCallbacks += executed;
                    throw new EpisodeAbortException(AbortReason.Livelock, "quiescence_waiter_spin", "Host code keeps re-awaiting RunContext.Quiescence() without reaching a decision point. " + Diagnose());
                }
                _quiescenceBatch.Clear();
                _quiescenceBatch.AddRange(_quiescenceWaiters);
                _quiescenceWaiters.Clear();
                for (int i = 0; i < _quiescenceBatch.Count; i++)
                {
                    _quiescenceBatch[i].TrySetResult();
                }
                continue;
            }
            break;
        }
        _c.PumpCallbacks += executed;
        if (_stack.Count > _c.MaxPendingStackDepth)
        {
            _c.MaxPendingStackDepth = _stack.Count;
        }

        // Quiescent: the registered episode checks (fidelity R1). A failure ends the episode here, whatever game code did with
        // the exception that was thrown at the site.
        int profc = HostProfile.On ? HostProfile.Enter(HostProfile.Checks) : 0;
        CheckFailure? failed = EvaluateChecks(final: false);
        if (HostProfile.On)
        {
            HostProfile.Enter(profc);
        }
        if (failed.HasValue)
        {
            throw new EpisodeAbortException(failed.Value.Reason, failed.Value.Detail, failed.Value.Message);
        }

        if (_pendingTravel.Count > 0)
        {
            RunContext ctx = _ctx!;
            MapCoord? at = ctx.Run.CurrentMapCoord;
            bool arrived = at.HasValue && _pendingTravel.Contains(at.Value) && !RunManager.Instance.IsChangingRooms;
            // (multi-seat: a reward set that the room entry opened for a remote seat suspends the move like a frame does)
            bool suspended = _stack.Count > 0 || (ctx.SeatCount > 1 && _session.Registry.RemoteOverlay?.AnyOpen(ctx) == true);
            if (arrived || ctx.Run.IsGameOver || suspended)
            {
                // arrived; or the move is suspended on a request the game made while entering the room
                // (it is re-checked once the stack is empty again); or the run ended on the way.
                if (arrived || ctx.Run.IsGameOver)
                {
                    _pendingTravel.Clear();
                }
            }
            else
            {
                string want = string.Join(" / ", _pendingTravel);
                _pendingTravel.Clear();
                throw new EpisodeAbortException(AbortReason.Stall, "travel_incomplete", $"Voted for map coord {want} but the run is at {(at.HasValue ? at.Value.ToString() : "null")} (IsChangingRooms={RunManager.Instance.IsChangingRooms}). " + Diagnose());
            }
        }
    }

    private void CheckFaults()
    {
        if (HostFaults.Count > 0)
        {
            List<Exception> faults = HostFaults.Drain();
            RethrowIfAbort(faults[0]);
            throw new EpisodeAbortException(AbortReason.GameTaskFault, "unobserved_fault", "A fire-and-forget game task faulted: " + faults[0].GetBaseException().Message, faults[0]);
        }
        StrictCounters delta = _session.ReadStrict() - _strictStart;
        if (delta.TutorialRoomForced != 0 || delta.DiscoveryBossChanged != 0 || delta.DiscoveryRoomsChanged != 0)
        {
            throw new EpisodeAbortException(AbortReason.ProfileGateFired,
                delta.TutorialRoomForced != 0 ? "tutorial_room_type" : delta.DiscoveryBossChanged != 0 ? "discovery_order_boss" : "discovery_order_rooms",
                "A tutorial / discovery branch that must be a no-op for the veteran profile changed the run. The profile is wrong." +
                (delta.DiscoveryRoomsChanged != 0 ? " " + SpireHost.Patches.FidelityGatePatches.LastDiscoveryChange : ""));
        }
        if (_session.Options.Strict)
        {
            string? violation = delta.FirstViolation();
            if (violation != null)
            {
                string extra = violation == "foreign_post" ? " origins: " + string.Join(" | ", _pump.ForeignPostOrigins.Keys.Take(3)) : "";
                if (violation == "late_unobserved_fault")
                {
                    extra = " last: " + (HostFaults.LastLateUnobserved ?? "").Split('\n')[0];
                }
                if (violation == "deferred_call_dropped")
                {
                    extra = " dropped: " + (SpireHost.Patches.GodotApiStubs.LastDropped ?? "?");
                }
                if (violation == "network_attempt")
                {
                    extra = " blocked: " + SpireHost.Patches.NetworkGuard.LastAttempt;
                }
                throw new EpisodeAbortException(AbortReason.StrictViolation, violation, "Strict mode: " + violation + "." + extra);
            }
        }
    }

    /// <summary>Evaluates every registered IEpisodeCheck; returns the first failure.</summary>
    private CheckFailure? EvaluateChecks(bool final)
    {
        IReadOnlyList<IEpisodeCheck> checks = _session.Registry.EpisodeChecks;
        RunContext? ctx = _ctx;
        if (checks.Count == 0 || ctx == null)
        {
            return null;
        }
        long t0 = Stopwatch.GetTimestamp();
        try
        {
            for (int i = 0; i < checks.Count; i++)
            {
                _c.EpisodeChecks++;
                CheckFailure? f = checks[i].Check(ctx, final);
                if (f.HasValue)
                {
                    return f;
                }
            }
            return null;
        }
        finally
        {
            _c.TicksChecks += Stopwatch.GetTimestamp() - t0;
        }
    }

    // =====================================================================================================
    //  Decision stack
    // =====================================================================================================

    internal void Push(IPendingDecision frame)
    {
        if (_ctx == null)
        {
            throw new InvalidOperationException("No episode is running.");
        }
        _stack.Add(frame);
    }

    internal void Pop(IPendingDecision frame)
    {
        int i = _stack.LastIndexOf(frame);
        if (i < 0)
        {
            throw new InvalidOperationException("Frame '" + frame.Name + "' is not on the decision stack.");
        }
        _stack.RemoveAt(i);
    }

    internal int PendingCount => _stack.Count;

    internal IPendingDecision? TopPending => _stack.Count > 0 ? _stack[_stack.Count - 1] : null;

    // =====================================================================================================
    //  Tracked operations
    // =====================================================================================================

    internal void Track(Task task, string what, bool blocking, int seat)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }
        _c.TrackedOperations++;
        if (task.IsCompletedSuccessfully)
        {
            return;
        }
        _ops.Add(new TrackedOp { Task = task, What = what, Blocking = blocking, Seat = seat });
    }

    internal void PostWork(Func<Task> work, string what, bool blocking)
    {
        RunContext? owner = _ctx;
        _pump.Post(_ =>
        {
            if (!ReferenceEquals(owner, _ctx) || owner == null)
            {
                return; // the episode that posted this is gone
            }
            Task t;
            try
            {
                t = work();
            }
            catch (Exception e)
            {
                t = Task.FromException(e);
            }
            Track(t, what, blocking, -1);
        }, null);
    }

    internal Task Quiescence()
    {
        if (_ctx == null)
        {
            throw new InvalidOperationException("No episode is running.");
        }
        TaskCompletionSource tcs = new TaskCompletionSource();
        _quiescenceWaiters.Add(tcs);
        return tcs.Task;
    }

    internal bool HasBlockingOperation
    {
        get
        {
            for (int i = 0; i < _ops.Count; i++)
            {
                if (_ops[i].Blocking && !_ops[i].Task.IsCompleted)
                {
                    return true;
                }
            }
            return false;
        }
    }

    /// <summary>Multi-seat: an incomplete blocking operation of this seat, or of no particular seat.</summary>
    internal bool HasBlockingOperationFor(int seat)
    {
        for (int i = 0; i < _ops.Count; i++)
        {
            if (_ops[i].Blocking && !_ops[i].Task.IsCompleted && (_ops[i].Seat == seat || _ops[i].Seat < 0))
            {
                return true;
            }
        }
        return false;
    }

    private void ObserveOperations()
    {
        for (int i = _ops.Count - 1; i >= 0; i--)
        {
            TrackedOp op = _ops[i];
            if (!op.Task.IsCompleted)
            {
                continue;
            }
            _ops.RemoveAt(i);
            if (op.Task.IsFaulted)
            {
                Exception e = op.Task.Exception!.GetBaseException();
                RethrowIfAbort(e);
                throw new EpisodeAbortException(AbortReason.GameTaskFault, "operation:" + op.What, "Tracked operation '" + op.What + "' faulted: " + e.Message, e);
            }
        }
    }

    /// <summary>
    /// A handler that calls RunContext.Fail from inside a game callback has its EpisodeAbortException captured
    /// by the game's task; when that task surfaces as a fault the original reason is restored.
    /// </summary>
    private static void RethrowIfAbort(Exception e)
    {
        Exception root = e.GetBaseException();
        if (root is EpisodeAbortException abort)
        {
            throw new EpisodeAbortException(abort.Reason, abort.Detail, abort.Message, abort);
        }
        if (root is SpireHost.Patches.EngineLivelockException livelock)
        {
            throw new EpisodeAbortException(AbortReason.Livelock, livelock.Detail, root.Message, root);
        }
        if (root is PolicyStopException stop)
        {
            throw new PolicyStopException(stop.Message); // a policy stopped the episode from inside a nested decision
        }
        if (root is EpisodeTruncatedException truncated)
        {
            throw new EpisodeTruncatedException(truncated.Message); // an episode limit was reached at a nested decision (R-1)
        }
    }

    private string DescribeOps()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < _ops.Count; i++)
        {
            if (!_ops[i].Task.IsCompleted)
            {
                if (sb.Length > 0)
                {
                    sb.Append(", ");
                }
                sb.Append(_ops[i].What).Append(_ops[i].Seat >= 0 ? " [seat " + _ops[i].Seat + "]" : "").Append(_ops[i].Blocking ? "" : " (non-blocking)");
            }
        }
        return sb.Length == 0 ? "none" : sb.ToString();
    }

    // =====================================================================================================
    //  Travel bookkeeping (MapHandler)
    // =====================================================================================================

    internal void ExpectTravel(MapCoord coord)
    {
        _pendingTravel.Clear();
        _pendingTravel.Add(coord);
    }

    /// <summary>Multi-seat: the last vote is in; the run must arrive at one of the voted coords (the host draws among them).</summary>
    internal void ExpectTravelAnyOf(IReadOnlyList<MapCoord> coords)
    {
        _pendingTravel.Clear();
        for (int i = 0; i < coords.Count; i++)
        {
            if (!_pendingTravel.Contains(coords[i]))
            {
                _pendingTravel.Add(coords[i]);
            }
        }
    }

    private void OnRoomEntered()
    {
        RunContext? ctx = _ctx;
        if (ctx == null)
        {
            return;
        }
        ctx.OnRoomEntered();
        _roomBoundaryPending = true;
        foreach (IRunLifecycle l in _session.Registry.Lifecycles)
        {
            l.OnRoomEntered(ctx);
        }
    }

    /// <summary>CombatManager.CombatWon (Combat/CombatManager.cs:1074): public history counters (obs O3). Read-only.</summary>
    private void OnCombatWon(CombatRoom room)
    {
        RunContext? ctx = _ctx;
        if (ctx == null)
        {
            return;
        }
        _c.CombatsWon++;
        if (room.RoomType == RoomType.Elite)
        {
            _c.ElitesWon++;
        }
        else if (room.RoomType == RoomType.Boss)
        {
            _c.BossesWon++;
        }
        _c.LastWonRoomType = (int)room.RoomType;
        _c.LastWonTotalFloor = ctx.Run.TotalFloor;
    }

    internal void Note(string note)
    {
        _lastNote = note;
    }

    // =====================================================================================================
    //  Teardown
    // =====================================================================================================

    private void Teardown(EpisodeResult result)
    {
        _phase = Phase.Teardown;
        SpireHost.Patches.LivelockGuard.Disarm();
        RunManager rm = RunManager.Instance;
        RunContext? ctx = _ctx;
        try
        {
            if (ctx != null)
            {
                RunState run = ctx.Run;
                result.ActIndex = run.CurrentActIndex;
                result.ActFloor = run.ActFloor;
                result.TotalFloor = run.TotalFloor;
                result.FinalRoom = DescribeRoom(run.CurrentRoom);
                result.FinalHp = run.Players.Count > 0 ? run.Players[0].Creature.CurrentHp : 0;
                result.FinalHps.Clear();
                for (int s = 0; s < run.Players.Count; s++)
                {
                    result.FinalHps.Add(run.Players[s].Creature.CurrentHp);
                }
            }
        }
        catch (Exception e)
        {
            result.TeardownFaults++;
            result.TeardownFault ??= "result: " + e.Message;
        }

        if (ctx != null)
        {
            foreach (IRunLifecycle l in _session.Registry.Lifecycles)
            {
                try
                {
                    l.OnEpisodeEnd(ctx);
                }
                catch (Exception e)
                {
                    result.TeardownFaults++;
                    result.TeardownFault ??= "OnEpisodeEnd(" + l.Name + "): " + e.Message;
                }
            }
        }
        rm.RoomEntered -= _roomEntered;
        if (_combatWonSubscribed)
        {
            CombatManager.Instance.CombatWon -= _combatWon;
            _combatWonSubscribed = false;
        }
        DriverPatches.ResetIntentTracker();
        _stack.Clear();
        _ops.Clear();
        _quiescenceWaiters.Clear();
        _pendingTravel.Clear();
        ctx?.Mp?.Detach();
        _ctx = null;

        try
        {
            if (rm.IsInProgress)
            {
                INetGameService net = rm.NetService;
                rm.CleanUp();
                NetLoadingHandle.Release(net); // the static load-count dictionary otherwise keeps the net service of every run alive
                if (net is HeadlessHostNetService)
                {
                    // Lead ruling Q10 (ledger: mp-run-lobby-reset). RunManager never resets RunLobby (:141, set at :425, disposed at
                    // :1480-1484): the disposed lobby of a multi-seat run would be handed to the CombatStateSynchronizer and the
                    // RestSiteSynchronizer of the NEXT run, a single-seat one included (:421-430). A process that has never played a
                    // multiplayer run has null there; the host restores that.
                    rm.RunLobby = null;
                    MultiSeatStats.RunLobbyResets++;
                }
            }
            rm.ForceDiscoveryOrderModifications = false;
            // Let cancelled continuations (turn loop, parked selections) unwind before the next episode.
            _pump.TryRunUntilQuiescent(MaxCallbacksPerSettle, out _);
            _pump.Clock.Clear();
            _pump.DiscardQueued();
        }
        catch (Exception e)
        {
            result.TeardownFaults++;
            result.TeardownFault ??= "cleanup: " + e.GetBaseException().Message;
            try
            {
                _pump.Clock.Clear();
                _pump.DiscardQueued();
            }
            catch
            {
                // nothing more to do
            }
        }

        if (result.Outcome == EpisodeOutcome.Aborted)
        {
            // Flush "unobserved task exception" notifications of the broken episode now, so that they are
            // attributed to it and cannot fail a later, innocent episode.
            GC.Collect();
            if (!HostSession.NoFinalizerThread)
            {
                GC.WaitForPendingFinalizers();
            }
        }
        List<Exception> late = HostFaults.Drain();
        if (late.Count > 0 && result.Outcome != EpisodeOutcome.Aborted)
        {
            result.TeardownFaults += late.Count;
            result.TeardownFault ??= "fault during teardown: " + late[0].GetBaseException().Message;
        }

        // ---- end-of-episode strict accounting ----
        result.Strict = _session.ReadStrict() - _strictStart;
        foreach (KeyValuePair<string, long> kv in NullGodot.Delta(_nativeStart))
        {
            result.NativeCalls[kv.Key] = kv.Value;
            if (!_session.Options.NativeAllowList.Contains(kv.Key))
            {
                result.UnexpectedNativeCalls[kv.Key] = kv.Value;
            }
        }
        if (result.Outcome != EpisodeOutcome.Aborted && _session.Options.Strict)
        {
            string? violation = result.Strict.FirstViolation();
            if (violation == null && result.UnexpectedNativeCalls.Count > 0)
            {
                violation = "native_call:" + result.UnexpectedNativeCalls.Keys.First();
            }
            if (violation != null)
            {
                result.Outcome = EpisodeOutcome.Aborted;
                result.Abort = MakeAbort(AbortReason.StrictViolation, violation, "Strict mode: " + violation + " (detected at the end of the episode).", null);
            }
        }

        // ---- module counters of this episode (sorted "module.key"; "_max" keys are high-water marks) ----
        try
        {
            SortedDictionary<string, long> now = _session.Registry.ReadModuleCounters();
            SortedDictionary<string, long> into = result.ModuleCounters;
            foreach (KeyValuePair<string, long> kv in now)
            {
                if (kv.Key.EndsWith("_max", StringComparison.Ordinal))
                {
                    into[kv.Key] = kv.Value;
                }
                else
                {
                    into[kv.Key] = kv.Value - (_moduleStart.TryGetValue(kv.Key, out long before) ? before : 0);
                }
            }
            EpisodeCounters c = result.Counters;
            for (int k = 1; k < c.DecisionsByKind.Length; k++)
            {
                string kind = ((DecisionKind)k).ToString();
                into["driver.kind." + kind] = c.DecisionsByKind[k];
                into["driver.kind_policy." + kind] = c.PolicyByKind[k];
            }
            for (int v = 1; v < c.VerbChosen.Length; v++)
            {
                string verb = ((Verb)v).ToString();
                into["driver.verb_chosen." + verb] = c.VerbChosen[v];
                into["driver.verb_offered." + verb] = c.VerbOffered[v];
            }
            into["driver.episode_checks"] = c.EpisodeChecks;
            into["driver.elites_won"] = c.ElitesWon;
            into["driver.bosses_won"] = c.BossesWon;
        }
        catch (Exception e)
        {
            result.TeardownFaults++;
            result.TeardownFault ??= "module counters: " + e.Message;
        }
    }

    // =====================================================================================================
    //  Diagnostics
    // =====================================================================================================

    private AbortInfo MakeAbort(AbortReason reason, string detail, string message, Exception? inner)
    {
        AbortInfo info = new AbortInfo { Reason = reason, Detail = detail, Message = FirstLine(message) };
        Exception? root = inner?.GetBaseException();
        if (root != null)
        {
            info.ExceptionType = root.GetType().Name;
            info.Frames = FirstFrames(root, 6);
        }
        try
        {
            info.Diagnostics = Diagnose();
        }
        catch (Exception e)
        {
            info.Diagnostics = "diagnose failed: " + e.Message;
        }
        return info;
    }

    private static string FirstLine(string s)
    {
        int i = s.IndexOf('\n');
        return (i >= 0 ? s.Substring(0, i) : s).Trim();
    }

    /// <summary>First stack frames of an exception without file info; the first game frame ("MegaCrit.") is moved to the front.</summary>
    internal static List<string> FirstFrames(Exception root, int max)
    {
        List<string> all = new List<string>();
        foreach (string line in (root.StackTrace ?? "").Split('\n'))
        {
            string s = line.Trim();
            if (!s.StartsWith("at ", StringComparison.Ordinal))
            {
                continue;
            }
            s = s.Substring(3);
            int i = s.IndexOf(" in ", StringComparison.Ordinal);
            if (i > 0)
            {
                s = s.Substring(0, i);
            }
            if (s.StartsWith("System.Runtime.", StringComparison.Ordinal) || s.StartsWith("System.Threading.", StringComparison.Ordinal))
            {
                continue;
            }
            all.Add(s);
        }
        int game = all.FindIndex(f => f.StartsWith("MegaCrit.", StringComparison.Ordinal));
        List<string> r = new List<string>(max);
        if (game > 0)
        {
            r.Add(all[game]);
        }
        for (int i = 0; i < all.Count && r.Count < max; i++)
        {
            if (i != game || game == 0)
            {
                r.Add(all[i]);
            }
        }
        return r;
    }

    internal static string DescribeRoom(AbstractRoom? room)
    {
        if (room == null)
        {
            return "null";
        }
        return room.GetType().Name + ":" + room.RoomType + ":" + (room.ModelId?.Entry ?? "-");
    }

    internal string Diagnose()
    {
        StringBuilder sb = new StringBuilder(256);
        RunManager rm = RunManager.Instance;
        CombatManager cm = CombatManager.Instance;
        RunContext? ctx = _ctx;
        sb.Append("[phase=").Append(_phase);
        if (_phaseNote.Length > 0)
        {
            sb.Append('(').Append(_phaseNote).Append(')');
        }
        if (ctx != null)
        {
            RunState run = ctx.Run;
            sb.Append(" seed=").Append(ctx.Seed).Append(" act=").Append(run.CurrentActIndex).Append(" floor=").Append(run.ActFloor).Append(" totalFloor=").Append(run.TotalFloor);
            sb.Append(" room=").Append(DescribeRoom(run.CurrentRoom)).Append(" rooms=").Append(run.CurrentRoomCount);
            sb.Append(" mapOpen=").Append(ctx.MapOpen).Append(" gameOver=").Append(run.IsGameOver);
            Player p = run.Players[0];
            sb.Append(" hp=").Append(p.Creature.CurrentHp).Append('/').Append(p.Creature.MaxHp);
            for (int s = 1; s < run.Players.Count; s++)
            {
                Player q = run.Players[s];
                sb.Append(" hp").Append(s).Append('=').Append(q.Creature.CurrentHp).Append('/').Append(q.Creature.MaxHp);
                if (cm.IsInProgress)
                {
                    sb.Append(cm.IsPlayerReadyToEndTurn(q) ? "(ended)" : "");
                }
            }
            if (run.Players.Count > 1)
            {
                sb.Append(" seats{");
                for (int s = 0; s < run.Players.Count; s++)
                {
                    sb.Append(s).Append(':').Append(ctx.IsMapOpen(s) ? "map" : "room").Append(ctx.IsActReady(s) ? "+actReady" : "").Append(s + 1 < run.Players.Count ? " " : "");
                }
                sb.Append('}');
            }
            if (cm.IsInProgress || cm.IsStarting)
            {
                PlayerCombatState? pcs = p.PlayerCombatState;
                sb.Append(" combat{inProgress=").Append(cm.IsInProgress).Append(" starting=").Append(cm.IsStarting).Append(" phase=").Append(pcs?.Phase.ToString() ?? "-")
                  .Append(" turn=").Append(pcs?.TurnNumber ?? -1).Append(" sync=").Append(rm.ActionQueueSynchronizer?.CombatState.ToString() ?? "-")
                  .Append(" readyToEnd=").Append(cm.IsPlayerReadyToEndTurn(p)).Append('}');
            }
        }
        if (rm.IsInProgress)
        {
            MegaCrit.Sts2.Core.GameActions.ActionExecutor? ex = rm.ActionExecutor;
            sb.Append(" changingRooms=").Append(rm.IsChangingRooms);
            sb.Append(" executor{running=").Append(ex?.IsRunning).Append(" paused=").Append(ex?.IsPaused).Append(" current=").Append(ex?.CurrentlyRunningAction?.GetType().Name ?? "-").Append('}');
        }
        sb.Append(" stack=").Append(_stack.Count);
        if (_stack.Count > 0)
        {
            sb.Append('(').Append(_stack[_stack.Count - 1].Name).Append(')');
        }
        sb.Append(" ops=").Append(DescribeOps());
        sb.Append(" pumpQueue=").Append(_pump.QueueCount).Append(" virtualTimers=").Append(_pump.Clock.Pending);
        sb.Append(" decisions=").Append(_c.DecisionPoints).Append(" lastDecision=").Append(_lastDecision).Append(" lastNote=").Append(_lastNote).Append(']');
        return sb.ToString();
    }
}
