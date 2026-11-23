using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SpireHost.Driver;

/// <summary>
/// What to play. Seats: 1-4 characters. One seat = the game's single-player run (RunManager.SetUpNewSingleplayer);
/// two to four seats = a multiplayer run on the host's own Host-type net service, seat 0 local, the others virtual
/// clients (phase H5; the same character may sit on several seats).
/// </summary>
public sealed class EpisodeSpec
{
    /// <summary>
    /// 12-character game seed (canonicalised with SeedHelper.CanonicalizeSeed), or null = a fresh one from the
    /// game's SeedHelper.GetRandomSeed over the session's CSPRNG. Never reaches an observation.
    /// </summary>
    public string? Seed;

    /// <summary>Character id per seat ("IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT", "FOOL").</summary>
    public string[] Characters = { "IRONCLAD" };

    /// <summary>Requested ascension 0..10; clamped to the profile's MaxAscension exactly like the lobby does.</summary>
    public int Ascension;

    /// <summary>Optional test scenario (loadout + jump into a room); see <see cref="Scenario"/>.</summary>
    public Scenario? Scenario;

    /// <summary>
    /// Optional UTF-8 JSON of a game <c>SerializableRun</c> (used by <c>--shadow-serve</c> to restore a live room
    /// snapshot via <c>RunState.FromSerializable</c> + <c>SetUpSavedSingleplayer</c> + <c>LoadIntoLatestMapCoord</c>).
    /// </summary>
    public string? SavedRunJson;

    public string Character => Characters.Length > 0 ? Characters[0] : "";
}

/// <summary>Host-side rules of an episode. Training and evaluation use the same defaults (DECISIONS-H1 C, B.6).</summary>
public sealed class EpisodeOptions
{
    /// <summary>A decision with exactly one legal action is applied by the host without asking the policy.</summary>
    public bool AutoSingle = true;

    /// <summary>Gold rewards are taken by the host without asking the policy.</summary>
    public bool AutoGold = true;

    /// <summary>Keep actions flagged <see cref="ActionFlags.Abandon"/> in the action list.</summary>
    public bool AllowAbandon;

    /// <summary>Compute the run signature at every policy decision and fold it into the trace hash.</summary>
    public bool ComputeSignatures;

    /// <summary>Additionally keep the per-decision signatures in the result (implies ComputeSignatures).</summary>
    public bool RecordTrace;

    /// <summary>Keep the chosen action index of every policy decision in the result (cheap; enough to replay with ScriptedPolicy).</summary>
    public bool RecordActions;

    /// <summary>Stop (outcome Truncated) after this many policy decisions; 0 = unlimited.</summary>
    public long MaxPolicyDecisions;

    /// <summary>
    /// A combat that reaches this player turn number ends the episode as Truncated("combat_turn_cap"); 0 = no cap.
    /// Not a host fault: an endless combat is possible in the real game (and trivially with DebugForcedHp);
    /// the consumer decides how to score it. Real fights are far below the default.
    /// </summary>
    public int MaxCombatTurns = 500;

    /// <summary>Livelock guard: abort when the episode has had this many decision points (policy + auto).</summary>
    public long DecisionCap = 400_000;

    /// <summary>
    /// TEST ONLY (win-path fuzz, long-run coverage): force every seat's max and current HP at run start. 0 = off.
    /// A DEBUG option: in serve mode it is honoured only when the process was started with --allow-debug-options (ruling A4).
    /// </summary>
    public int DebugForcedHp;

    /// <summary>
    /// TEST ONLY, multi-seat: forced max and current HP per seat (index = seat; 0 = that seat keeps <see cref="DebugForcedHp"/>
    /// or its natural HP). For the death / revive / team-wipe tests. A DEBUG option like <see cref="DebugForcedHp"/>.
    /// </summary>
    public int[]? DebugForcedHpPerSeat;

    /// <summary>
    /// VERIFICATION MODE, multi-seat (lead ruling Q11; default off): every message of a virtual client is serialised and
    /// deserialised with the game's own packet serializer before the host handles it (HeadlessHostNetService.WireRoundTrip).
    /// An episode must be identical with the mode on and off.
    /// </summary>
    public bool WireRoundTrip;

    /// <summary>
    /// MULTI-SEAT (phase H5; lead ruling Q5): team votes. A vote whose result is one draw from the list of all seats'
    /// votes - the map vote, the page vote of a shared event - is asked ONCE, from the lowest seat that has not voted;
    /// the host then casts the same vote for every other seat, each as that seat's own client message at the point
    /// where that seat would have been asked. Off: every seat is asked (seat order). The treasure relic vote is never a
    /// team vote (the seats vote for different relics). On for --serve by default, off in the fuzz / determinism tools
    /// unless --team-votes is given (both settings must be exercised). No effect on a single-seat run.
    /// </summary>
    public bool TeamVotes;

    /// <summary>
    /// MULTI-SEAT (phase H5; lead ruling Q6): offer UNDO_END_TURN to a seat that has ended its turn while teammates
    /// have not (the real button). On in the fuzz / determinism tools (the default here); the protocol server turns it
    /// off unless the RESET asks for it: one controller never needs to take an END_TURN back, and END / UNDO ping-pong
    /// is a loop a policy can learn. Not offering it removes options only: every line played stays legal. No effect on
    /// a single-seat run (the button does not exist with one player).
    /// </summary>
    public bool OfferUndoEndTurn = true;

    /// <summary>
    /// TOOL ONLY (real-client run-trace comparison): the profile's WongoPoints for this episode instead of the value
    /// derived from the run seed (DECISIONS-H1 D.3). -1 = derive from the seed (production). The oracle's traces were
    /// recorded with a profile that has 0 points (realgame\mod-src\SpireOracle\Profile.cs); WelcomeToWongos reads the
    /// value (Models.Events/WelcomeToWongos.cs:122-136). Not reachable from the protocol: no RESET word carries it.
    /// </summary>
    public int DebugWongoPoints = -1;

    public EpisodeOptions Clone() => (EpisodeOptions)MemberwiseClone();
}

/// <summary>
/// Test scenario: after the real run start chain (the run sits in its first room), <see cref="Setup"/> runs
/// on the engine thread as a tracked operation - give relics / cards / potions with the game's own commands
/// and jump with RunManager.EnterRoomDebug (see ScenarioTools) - then the normal driver loop takes over.
/// The episode ends with outcome Truncated as soon as <see cref="StopWhen"/> holds at a quiescent point.
/// </summary>
public sealed class Scenario
{
    public string Name = "scenario";
    public Func<RunContext, Task>? Setup;
    public Func<RunContext, bool>? StopWhen;

    /// <summary>
    /// TEST ONLY, explicit: ids of invariants that this scenario deliberately violates in order to build its state
    /// (e.g. "rest_options_inert": the scenario sets the game's own test hook RestSiteOption.generateForTests to obtain
    /// a rest site without an enabled option). A check that supports exemptions asks <see cref="IsExempt"/>; everything
    /// else stays strict. A Scenario exists only in test code (a RESET of the protocol cannot carry one).
    /// </summary>
    public HashSet<string>? Exempt;

    public bool IsExempt(string invariantId) => Exempt != null && Exempt.Contains(invariantId);
}

public enum EpisodeOutcome
{
    /// <summary>run.IsGameOver in the victory room (TheArchitect).</summary>
    Win,

    /// <summary>run.IsGameOver anywhere else.</summary>
    Loss,

    /// <summary>The episode could not be completed faithfully. Always counted, never dropped.</summary>
    Aborted,

    /// <summary>Ended on request or on a configured limit (MaxPolicyDecisions, MaxCombatTurns, scripted policy exhausted, scenario StopWhen). Not a fault.</summary>
    Truncated,
}

public enum AbortReason
{
    None = 0,

    /// <summary>The run start chain threw or stalled.</summary>
    StartFailed,

    /// <summary>A fire-and-forget game task faulted (TaskHelper.UnobservedFault) or a tracked operation faulted.</summary>
    GameTaskFault,

    /// <summary>Applying an action threw synchronously (game exception inside IDecisionSource.Apply).</summary>
    ActionFault,

    /// <summary>A room handler / pending frame threw while building a decision.</summary>
    HandlerFault,

    /// <summary>The policy threw or answered with an index outside the action list.</summary>
    PolicyFault,

    /// <summary>Quiescent, not terminal, and nothing that can ask for input (see RunDriver for the exact predicate). Detail = sub-code.</summary>
    Stall,

    /// <summary>The engine or the driver keeps moving without reaching a decision point. Detail = which bound was hit.</summary>
    Livelock,

    /// <summary>A strict-mode counter moved (foreign-thread post, real-time wait, dropped deferred call, unexpected native call, late unobserved fault, off-thread continuation).</summary>
    StrictViolation,

    /// <summary>A tutorial / discovery branch that must be a no-op for the veteran profile did something.</summary>
    ProfileGateFired,

    /// <summary>Unexpected exception in host code.</summary>
    HostBug,
}

/// <summary>Thrown inside the driver to end the current episode as Aborted. Handlers use RunContext.Fail.</summary>
public sealed class EpisodeAbortException : Exception
{
    public AbortReason Reason { get; }
    public string Detail { get; }

    public EpisodeAbortException(AbortReason reason, string detail, string message, Exception? inner = null)
        : base(message, inner)
    {
        Reason = reason;
        Detail = detail;
    }
}

/// <summary>Thrown by a policy to end the episode as Truncated (e.g. a script ran out).</summary>
public sealed class PolicyStopException : Exception
{
    public PolicyStopException(string message) : base(message)
    {
    }
}

/// <summary>Per-episode counters. All plain fields; engine thread only.</summary>
public sealed class EpisodeCounters
{
    /// <summary>Decision points reached (policy + auto-advanced).</summary>
    public long DecisionPoints;

    /// <summary>Decisions answered by the policy.</summary>
    public long PolicyDecisions;

    /// <summary>Of those, asked from inside a synchronous game callback.</summary>
    public long NestedDecisions;

    /// <summary>Nested decisions that found callbacks still queued on the pump (expected 0 in single player).</summary>
    public long NestedWithQueuedWork;

    /// <summary>Forced single-action frames advanced by the host, per DecisionKind (index = (int)kind).</summary>
    public readonly long[] AutoSingle = new long[14];

    /// <summary>Gold rewards taken by the host.</summary>
    public long AutoGold;

    public long AutoSingleTotal
    {
        get
        {
            long n = 0;
            for (int i = 0; i < AutoSingle.Length; i++)
            {
                n += AutoSingle[i];
            }
            return n;
        }
    }

    /// <summary>Policy decisions per DecisionKind.</summary>
    public readonly long[] PolicyByKind = new long[14];

    /// <summary>Decision points per DecisionKind (policy + auto-advanced).</summary>
    public readonly long[] DecisionsByKind = new long[14];

    /// <summary>Chosen actions per Verb (policy and auto-advanced alike; index = (int)verb).</summary>
    public readonly long[] VerbChosen = new long[16];

    /// <summary>Decision points at which a Verb was offered at least once (index = (int)verb).</summary>
    public readonly long[] VerbOffered = new long[16];

    /// <summary>Stopwatch ticks spent in: pumping to quiescence / building decisions (legal-action enumeration) / applying actions (most game logic runs inline here) / inside the policy.</summary>
    public long TicksSettle;
    public long TicksBuild;
    public long TicksApply;
    public long TicksPolicy;
    public long TicksSignature;

    /// <summary>Time inside the registered IEpisodeCheck evaluations (they run at the end of every settle: part of TicksSettle).</summary>
    public long TicksChecks;

    public long HandlerProgressSteps;
    public long RoomsEntered;

    // ---- public history (obs request O3; maintained by the core from CombatManager.CombatWon) ----

    /// <summary>Combats won in this run so far.</summary>
    public long CombatsWon;

    /// <summary>Of those, in rooms of type Elite / Boss.</summary>
    public int ElitesWon;
    public int BossesWon;

    /// <summary>(int)RoomType of the room of the most recent won combat; -1 = none yet.</summary>
    public int LastWonRoomType = -1;

    /// <summary>RunState.TotalFloor at that moment; -1 = none yet.</summary>
    public int LastWonTotalFloor = -1;

    /// <summary>Episode checks evaluated (IEpisodeCheck.Check calls).</summary>
    public long EpisodeChecks;
    public long PumpCallbacks;
    public long VirtualTimerFires;
    public long MaxStackBytes;
    public long MaxPendingStackDepth;
    public long TrackedOperations;
}

/// <summary>Strict-mode counters (process-wide sources, reported as per-episode deltas).</summary>
public struct StrictCounters
{
    public long ForeignPosts;
    public long RealDelays;
    public long ForeignWaits;
    public long DeferredCallsDropped;
    public long LateUnobservedFaults;
    public long OffThreadContinuations;

    /// <summary>Outbound network entry points of the game that were reached (and blocked; Core NetworkGuard). The host performs no network IO.</summary>
    public long NetworkAttempts;

    /// <summary>
    /// SaveManager.SaveRun calls that run-save-completes-after-pump-turn could not be applied to (called without the engine
    /// pump or off the engine thread; the awaiter then continues inline, which is another schedule). Verifier finding R-13:
    /// this was an informational counter; it is a violation.
    /// </summary>
    public long RunSaveBoundaryNotApplied;
    public long TutorialRoomForced;
    public long DiscoveryBossChanged;

    /// <summary>ApplyDiscoveryOrderModifications changed an act's room set (fidelity R2). Like DiscoveryBossChanged: must stay 0, fatal (ProfileGateFired).</summary>
    public long DiscoveryRoomsChanged;
    public long GameErrorsLogged;
    public long DeferredCalls;
    public long VirtualDelays;
    public long InlineContinuations;
    public long InterceptedProgressWrites;

    public static StrictCounters operator -(StrictCounters a, StrictCounters b) => new StrictCounters
    {
        ForeignPosts = a.ForeignPosts - b.ForeignPosts,
        RealDelays = a.RealDelays - b.RealDelays,
        ForeignWaits = a.ForeignWaits - b.ForeignWaits,
        DeferredCallsDropped = a.DeferredCallsDropped - b.DeferredCallsDropped,
        LateUnobservedFaults = a.LateUnobservedFaults - b.LateUnobservedFaults,
        OffThreadContinuations = a.OffThreadContinuations - b.OffThreadContinuations,
        NetworkAttempts = a.NetworkAttempts - b.NetworkAttempts,
        RunSaveBoundaryNotApplied = a.RunSaveBoundaryNotApplied - b.RunSaveBoundaryNotApplied,
        TutorialRoomForced = a.TutorialRoomForced - b.TutorialRoomForced,
        DiscoveryBossChanged = a.DiscoveryBossChanged - b.DiscoveryBossChanged,
        DiscoveryRoomsChanged = a.DiscoveryRoomsChanged - b.DiscoveryRoomsChanged,
        GameErrorsLogged = a.GameErrorsLogged - b.GameErrorsLogged,
        DeferredCalls = a.DeferredCalls - b.DeferredCalls,
        VirtualDelays = a.VirtualDelays - b.VirtualDelays,
        InlineContinuations = a.InlineContinuations - b.InlineContinuations,
        InterceptedProgressWrites = a.InterceptedProgressWrites - b.InterceptedProgressWrites,
    };

    /// <summary>
    /// Sum over episodes (tool summaries). Verifier finding R-2: the fuzz tool summed the fields by hand and left
    /// DiscoveryRoomsChanged out, so its total was 0 by construction; the sum now lives next to the difference and the
    /// driver self-test checks by reflection that both cover every field.
    /// </summary>
    public static StrictCounters operator +(StrictCounters a, StrictCounters b) => new StrictCounters
    {
        ForeignPosts = a.ForeignPosts + b.ForeignPosts,
        RealDelays = a.RealDelays + b.RealDelays,
        ForeignWaits = a.ForeignWaits + b.ForeignWaits,
        DeferredCallsDropped = a.DeferredCallsDropped + b.DeferredCallsDropped,
        LateUnobservedFaults = a.LateUnobservedFaults + b.LateUnobservedFaults,
        OffThreadContinuations = a.OffThreadContinuations + b.OffThreadContinuations,
        NetworkAttempts = a.NetworkAttempts + b.NetworkAttempts,
        RunSaveBoundaryNotApplied = a.RunSaveBoundaryNotApplied + b.RunSaveBoundaryNotApplied,
        TutorialRoomForced = a.TutorialRoomForced + b.TutorialRoomForced,
        DiscoveryBossChanged = a.DiscoveryBossChanged + b.DiscoveryBossChanged,
        DiscoveryRoomsChanged = a.DiscoveryRoomsChanged + b.DiscoveryRoomsChanged,
        GameErrorsLogged = a.GameErrorsLogged + b.GameErrorsLogged,
        DeferredCalls = a.DeferredCalls + b.DeferredCalls,
        VirtualDelays = a.VirtualDelays + b.VirtualDelays,
        InlineContinuations = a.InlineContinuations + b.InlineContinuations,
        InterceptedProgressWrites = a.InterceptedProgressWrites + b.InterceptedProgressWrites,
    };

    /// <summary>The first strict violation in this delta, or null. (Informational counters are not violations.)</summary>
    public string? FirstViolation()
    {
        if (ForeignPosts != 0)
        {
            return "foreign_post";
        }
        if (RealDelays != 0 || ForeignWaits != 0)
        {
            return "real_time_wait";
        }
        if (DeferredCallsDropped != 0)
        {
            return "deferred_call_dropped";
        }
        if (LateUnobservedFaults != 0)
        {
            return "late_unobserved_fault";
        }
        if (OffThreadContinuations != 0)
        {
            return "off_thread_continuation";
        }
        if (NetworkAttempts != 0)
        {
            return "network_attempt";
        }
        if (RunSaveBoundaryNotApplied != 0)
        {
            return "run_save_boundary_skipped";
        }
        return null;
    }
}

public sealed class AbortInfo
{
    public AbortReason Reason;

    /// <summary>Sub-code, e.g. "no_handler:Event", "pending_operation", "virtual_time_budget", "foreign_post".</summary>
    public string Detail = "";
    public string Message = "";
    public string? ExceptionType;

    /// <summary>First stack frames of the root exception, game frames first ("MegaCrit...." without file info).</summary>
    public List<string> Frames = new List<string>();

    /// <summary>Engine / driver snapshot at the moment of the abort.</summary>
    public string Diagnostics = "";

    /// <summary>Key for grouping in fuzz summaries: reason + detail + first frame.</summary>
    public string Signature => Reason + "/" + Detail + (Frames.Count > 0 ? " @ " + Frames[0] : ExceptionType != null ? " " + ExceptionType : "");
}

public sealed class EpisodeResult
{
    public EpisodeOutcome Outcome;
    public AbortInfo? Abort;

    /// <summary>Truncated: why ("max_policy_decisions", "combat_turn_cap", "policy_stop: ...", "scenario").</summary>
    public string? TruncatedBy;

    public string Seed = "";

    /// <summary>Seat 0's character (the only seat of a single-seat run).</summary>
    public string Character = "";

    /// <summary>The character of every seat, in seat order (one entry in a single-seat run).</summary>
    public List<string> Characters = new List<string>();
    public int Ascension;

    /// <summary>Requested ascension before the profile clamp.</summary>
    public int RequestedAscension;
    public List<string> Acts = new List<string>();

    /// <summary>0-based act index and floors at the end of the episode.</summary>
    public int ActIndex;
    public int ActFloor;

    /// <summary>Map points entered over the whole run (RunState.TotalFloor): the "furthest floor".</summary>
    public int TotalFloor;
    public string FinalRoom = "";

    /// <summary>Seat 0's HP at the end of the episode.</summary>
    public int FinalHp;

    /// <summary>Every seat's HP at the end of the episode, in seat order.</summary>
    public List<int> FinalHps = new List<int>();
    public int WongoPoints;

    public EpisodeCounters Counters = new EpisodeCounters();
    public StrictCounters Strict;

    /// <summary>
    /// Per-module counters of this episode, "module.key" -> value (sorted): the difference of every registered
    /// IModuleCounters source between episode start and end, plus the core's own "driver.*" / "core.*" entries
    /// (decision kinds, verbs, scheduling boundaries). Keys ending in "_max" are high-water marks.
    /// </summary>
    public SortedDictionary<string, long> ModuleCounters = new SortedDictionary<string, long>(StringComparer.Ordinal);

    /// <summary>Native (Godot) entry points reached during the episode that are not on the allow-list: name -> calls.</summary>
    public Dictionary<string, long> UnexpectedNativeCalls = new Dictionary<string, long>();
    public Dictionary<string, long> NativeCalls = new Dictionary<string, long>();

    /// <summary>Faults raised while tearing the run down (after the outcome was decided).</summary>
    public int TeardownFaults;
    public string? TeardownFault;

    /// <summary>Rolling hash over (state signature, decision shape, chosen index) of every policy decision; 0 when signatures are off.</summary>
    public ulong TraceHash;

    /// <summary>Per policy decision: state signature ^ decision shape (EpisodeOptions.RecordTrace).</summary>
    public List<ulong>? Trace;

    /// <summary>Per policy decision: chosen action index (EpisodeOptions.RecordActions). Feed to ScriptedPolicy to replay.</summary>
    public List<int>? ActionLog;

    public double WallMs;

    /// <summary>Run start (profile restore .. EnterAct(0) settled), and two parts of it.</summary>
    public double StartMs;
    public double ProfileRestoreMs;
    public double ActSelectionMs;
    public double RunSetUpMs;
    public double EnterActMs;

    public bool Clean => Outcome != EpisodeOutcome.Aborted && TeardownFaults == 0;
}
