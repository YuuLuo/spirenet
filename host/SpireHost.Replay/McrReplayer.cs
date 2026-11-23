using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Boot;
using SpireHost.Engine;

namespace SpireHost.Replay;

/// <summary>One recorded combat: the .mcr read with the game's own reader + the oracle sidecar.</summary>
public sealed class McrFile
{
    public required string Path;
    public required CombatReplay Replay;
    public JsonNode? Sidecar;

    /// <summary>The collector's per-decision records (sidecar actions[] or a run trace's NNN-combat-fN.combat.json Actions[]).</summary>
    public JsonArray? Actions;

    /// <summary>Run-trace fights: the combat.json that belongs to this replay (null when the trace has none for it).</summary>
    public string? CombatJsonPath;

    /// <summary>
    /// A combat entered from a map node (run traces): the save was recorded in EnterMapPointInternal (RunManager.cs:780-783)
    /// before the room type / encounter were rolled, so the room is re-entered with LoadIntoLatestMapCoord exactly like
    /// NMultiplayerTest.RunReplay does. Corpus replays (console 'fight') carry the encounter + rng seed in the sidecar instead.
    /// </summary>
    public bool MapNode;
    public string Encounter = "";
    public ulong EncounterSeed;
    public string Name => System.IO.Path.GetFileNameWithoutExtension(Path);

    /// <summary>PacketReader.Reset(bytes); Read&lt;CombatReplay&gt;() (oracle-mod.md 5.1, NMultiplayerTest.LoadReplay).</summary>
    public static McrFile Load(string path)
    {
        byte[] data = File.ReadAllBytes(path);
        PacketReader reader = new PacketReader();
        reader.Reset(data);
        CombatReplay replay = reader.Read<CombatReplay>();
        McrFile f = new McrFile { Path = path, Replay = replay };
        string side = System.IO.Path.ChangeExtension(path, ".json");
        if (File.Exists(side))
        {
            f.Sidecar = JsonNode.Parse(File.ReadAllText(side));
            f.Encounter = f.Sidecar?["encounter"]?.GetValue<string>() ?? "";
            string? seed = f.Sidecar?["encounter_rng_seed"]?.GetValue<string>();
            if (seed != null)
            {
                f.EncounterSeed = ulong.Parse(seed);
            }
            f.Actions = f.Sidecar?["actions"] as JsonArray;
        }
        else
        {
            // run-trace fight: NNN-combat-fN.mcr has NNN-combat-fN.combat.json next to it; the fatal fight's replay is stored as
            // <end dump index + 1>-combat-final-loss.mcr while its combat.json keeps the fight's own tag (verifier finding F2:
            // the lookup by file name alone skipped the recorded decisions of every final-loss fight).
            string? combat = V2.V2Trace.SidecarOf(path);
            if (combat != null)
            {
                f.CombatJsonPath = combat;
                f.Actions = JsonNode.Parse(File.ReadAllText(combat))?["Actions"] as JsonArray;
            }
            f.MapNode = true;
        }
        return f;
    }
}

/// <summary>Knobs of one replay (tests flip these to prove the comparison detects things).</summary>
public sealed class ReplayOptions
{
    /// <summary>Wait for the collector's ready-for-input predicate before player-driven actions (oracle ReplayCheck.StrictTiming = true, what the real-client rerun used).</summary>
    public bool StrictTiming = true;

    /// <summary>Continue injecting after the first divergence (to count later mismatches); stops at the first injection failure.</summary>
    public bool ContinueAfterDivergence = true;

    /// <summary>Test only: alter the event with this index (swap the target of a play / replace by the next card) to prove detection.</summary>
    public int AlterEventIndex = -1;

    /// <summary>Record an extended host-side trace per checksum (intents + monster move ids + round number) for the determinism cross-check.</summary>
    public bool ExtendedTrace;

    /// <summary>Upper bound for continuations in one wait (livelock guard).</summary>
    public long MaxCallbacksPerWait = 2_000_000;

    public TimeSpan VirtualTimeBudget = TimeSpan.FromSeconds(120);

    /// <summary>Maximum diff lines kept for the first divergence.</summary>
    public int DiffLines = 60;
}

public sealed class Divergence
{
    public int ChecksumOrdinal;          // index in the recorded checksum list
    public uint ChecksumId;
    public string LocalContext = "";
    public string RecordedContext = "";
    public int EventIndex;               // index of the last event injected before the checksum was produced (-1 = before the first event)
    public string? Event;                // description of that event
    public int PlayerTurn;
    public uint Local;
    public uint Recorded;
    public List<string> Diff = new List<string>();
    public int DiffRemoved;
    public int DiffAdded;
    public List<string> DiffIds = new List<string>();
    public string? InvolvedCard;         // card of the last play event, or the action in the context
    public List<string> InvolvedPowers = new List<string>();
}

public sealed class ExtendedCheck
{
    public int PolicyActions;            // sidecar actions[] entries
    public int Compared;
    public int SigMismatches;
    public int OptionMismatches;
    public int NotADecisionPoint;        // the collector's IsDecision predicate was false where the event was injected
    public string? FirstSigMismatch;
    public string? FirstOptionMismatch;
    public int OptionMismatchesPotionsLockedInRecording;          // recorded count == host count without any potion option
    public int FirstMatchingPotionDecision = -1;                  // first action index where potions were held and the counts agreed
    public List<string> OptionMismatchList = new List<string>();   // "k:turn:host:recorded:potions:potionUseLocked" (first 64)
}

public sealed class McrReplayResult
{
    public string File = "";
    public int Index = -1;
    public string Character = "";
    public int Ascension;
    public string Encounter = "";
    public string RoomType = "";
    public string Act = "";
    public string RecordedResult = "";
    public string Block = "";
    public string Status = "";           // pass | diverged | incomplete | fault | stall | error
    public string? StatusDetail;
    public int Events;
    public int EventsInjected;
    public int ChecksumsRecorded;
    public int ChecksumsGenerated;
    public int Matched;
    public int Mismatched;
    public int MissingInReplay;          // generated ids that are not in the recording
    public Divergence? First;
    public string? Error;
    public string? ErrorType;

    /// <summary>The whole exception chain (outer exceptions first), truncated: the root alone can hide where game code was when it threw.</summary>
    public string? ErrorChain;
    public List<string>? Frames;
    public string? Diagnostics;
    public bool CombatEnded;
    public bool HostPlayerDead;
    public bool RecordedPlayerDead;

    /// <summary>Players of the recorded run (phase H5: a replay written by a client of a real 2-4 player session has several).</summary>
    public int Players = 1;

    /// <summary>Players that are dead when the replay ends (multi-player replays).</summary>
    public int PlayersDead;

    /// <summary>
    /// Multi-player replays: actions that finished while not in state Executing (a card play that cancelled itself because its
    /// card / target was no longer valid when its turn in the queue came). The real client logs these and goes on; so does the replay.
    /// </summary>
    public long ActionsFinishedNotExecuting;

    /// <summary>Multi-player replays: recorded inputs that reached the real host after the fight had ended (the queues cancel them).</summary>
    public int EventsAfterCombatEnd;
    public ExtendedCheck Extended = new ExtendedCheck();
    public long PostActionSubscriptions;

    /// <summary>Outbound network entry points the game reached during this replay (blocked by Core NetworkGuard). Must be 0.</summary>
    public long NetworkAttempts;
    public double Ms;
    public ulong TraceHash;              // fold of every generated (id, checksum) (+ extended trace when enabled)
    public List<uint>? Checksums;        // generated checksums in order (determinism cross-check)
    public List<ulong>? ExtendedTrace;
}

/// <summary>
/// Re-executes one real-game combat replay in the host under TestMode and compares the game's own
/// full-combat-state checksum after every action with the recorded one.
///
/// Call sequence = the real client's replay facility (NMultiplayerTest.RunReplay, NMultiplayerTest.cs:409-480)
/// as used by the oracle's rerun (realgame\mod-src\SpireOracle\ReplayCheck.cs, which reproduced 1488/1488
/// replays in the real client):
///   RunState.FromSerializable(save) -> RunManager.SetUpReplay(state, replay, netId) (NetReplayGameService)
///   -> CombatStateSynchronizer.IsDisabled = true -> Launch -> GenerateMap
///   -> FastForwardNextActionId / FastForwardHookId / LoadReplayChecksums / FastForwardChoiceIds / FastForwardRewardIds
///   -> EnterRoomDebug(Monster, Unassigned, encounter.ToMutable() with DebugSetRng(seed))   (what the console 'fight' did when recording)
///   -> per event: GameAction -> ActionQueueSet.EnqueueWithoutSynchronizing(action.ToGameAction(player))
///                 HookAction -> EnqueueWithoutSynchronizing(ActionQueueSynchronizer.GetHookActionForId(...))
///                 ResumeAction -> ActionQueueSet.ResumeActionWithoutSynchronizing(id)
///                 PlayerChoice -> PlayerChoiceSynchronizer.ReceiveReplayChoice(player, choiceId, result)
/// "Waiting a frame" is replaced by draining the pump to quiescence (and firing virtual timers) until the same
/// predicate the oracle used holds. Nothing answers a choice except ReceiveReplayChoice (the NetReplayGameService
/// makes CardSelectCmd wait for the "remote" choice, CardSelectCmd.cs:32-39); no selector is installed.
///
/// This is a combat-level tool like the H0 harness (DRIVER-API.md 9: "stays for ... the replay team"): it drives
/// the pump itself and does not run inside a RunDriver episode.
/// </summary>
public sealed class McrReplayer
{
    private readonly EnginePump _pump;

    public McrReplayer(EnginePump pump)
    {
        _pump = pump;
    }

    // per-replay state
    private McrFile _file = null!;
    private ReplayOptions _opt = null!;
    private McrReplayResult _r = null!;
    private Dictionary<uint, ReplayChecksumData> _recordedById = null!;
    private Dictionary<uint, int> _ordinalById = null!;
    private int _eventIndex;
    private string? _eventDesc;
    private string? _lastPlayedCard;
    private RunState? _run;
    private SigFold _fold;
    private NetFullCombatState? _firstLocalState;

    public McrReplayResult Run(McrFile file, ReplayOptions? options = null)
    {
        _file = file;
        _opt = options ?? new ReplayOptions();
        Stopwatch sw = Stopwatch.StartNew();
        CombatReplay replay = file.Replay;
        _r = new McrReplayResult
        {
            File = file.Name,
            Events = replay.events.Count,
            ChecksumsRecorded = replay.checksumData.Count,
            Checksums = new List<uint>(replay.checksumData.Count + 8),
            ExtendedTrace = _opt.ExtendedTrace ? new List<ulong>(replay.checksumData.Count + 8) : null,
        };
        FillMeta(file, _r);
        _recordedById = new Dictionary<uint, ReplayChecksumData>(replay.checksumData.Count);
        _ordinalById = new Dictionary<uint, int>(replay.checksumData.Count);
        for (int i = 0; i < replay.checksumData.Count; i++)
        {
            uint id = replay.checksumData[i].checksumData.id;
            _recordedById.TryAdd(id, replay.checksumData[i]);
            _ordinalById.TryAdd(id, i);
        }
        _eventIndex = -1;
        _eventDesc = null;
        _lastPlayedCard = null;
        _fold = SigFold.Create();
        _firstLocalState = null;

        RunManager rm = RunManager.Instance;
        long networkBefore = SpireHost.Patches.NetworkGuard.NetworkAttempts;
        ReplayInterventions.Scope? scope = null;
        ChecksumTracker? tracker = null;
        Action<NetChecksumData, string, NetFullCombatState>? onChecksum = null;
        long finishedBefore = ReplayInterventions.FinishedNotExecuting;
        _multi = false;
        try
        {
            Preflight(file);
            Prepare(rm);

            // ---- NMultiplayerTest.RunReplay ----
            RunState runState = RunState.FromSerializable(replay.serializableRun);
            _run = runState;
            _r.Players = runState.Players.Count;
            _multi = runState.Players.Count > 1;
            if (_multi)
            {
                // tool-only: an action that finishes in a state other than Executing is logged, not thrown, as in the real client
                ReplayInterventions.RegisterMultiPlayer();
                finishedBefore = ReplayInterventions.FinishedNotExecuting;
                ReplayInterventions.MultiPlayerReplayActive = true;
            }
            // (several players: the local player is Players[playerIndex] with the game's default ReplayPlayerIndex 0, NMultiplayerTest.cs:404, :417)
            ulong netId = runState.Players[0].NetId;          // the anonymised id of the save: keep it (oracle-mod.md 10.2)
            rm.SetUpReplay(runState, replay, netId);
            rm.CombatStateSynchronizer.IsDisabled = true;
            scope = new ReplayInterventions.Scope(rm);         // ChecksumTracker.IsEnabled = true + post-action checksum subscription
            tracker = rm.ChecksumTracker;
            onChecksum = OnChecksum;
            tracker.ChecksumGenerated += onChecksum;
            // (PreloadManager.LoadRunAssets / LoadActAssets / NRun.Create are presentation; not run headlessly)
            rm.Launch();
            Drive(rm.GenerateMap(), "GenerateMap");
            rm.ActionQueueSet.FastForwardNextActionId(replay.nextActionId);
            rm.ActionQueueSynchronizer.FastForwardHookId(replay.nextHookId);
            rm.ChecksumTracker.LoadReplayChecksums(replay.checksumData, replay.nextChecksumId);
            rm.PlayerChoiceSynchronizer.FastForwardChoiceIds(replay.choiceIds);
            rm.RewardsSetSynchronizer.FastForwardRewardIds(replay.rewardIds);

            // ---- enter the recorded encounter exactly like FightConsoleCmd.Process (FightConsoleCmd.cs:64-86) ----
            // "The fight has begun" is the game's own event (CombatManager.CombatBegan, Combat/CombatManager.cs:395), not the
            // IsInProgress flag at the moment the engine is looked at: a fight whose enemies all die in the start-of-combat sequence
            // is over again by then (v2 corpus: SILENT-A0-DEP200000021/048-combat-f46, 0 events, 1 checksum). Subscribed before
            // the room is entered: the entry can run through the whole start sequence before it returns.
            bool began = false;
            Action<CombatState> onBegan = _ => began = true;
            CombatManager.Instance.CombatBegan += onBegan;
            Task enter;
            try
            {
                if (file.MapNode)
                {
                    // NMultiplayerTest.cs:428 (map-node combat; the save has no PreFinishedRoom)
                    enter = rm.LoadIntoLatestMapCoord(AbstractRoom.FromSerializable(replay.serializableRun.PreFinishedRoom, runState));
                }
                else
                {
                    EncounterModel enc = ModelDb.GetById<EncounterModel>(new ModelId(ModelId.SlugifyCategory<EncounterModel>(), file.Encounter.ToUpperInvariant())).ToMutable();
                    enc.DebugSetRng(file.EncounterSeed);
                    enter = rm.EnterRoomDebug(RoomType.Monster, MapPointType.Unassigned, enc);
                }
                Wait(() => began || CombatManager.Instance.IsInProgress || enter.IsFaulted, "combat start");
            }
            catch (ReplayStepException e)
            {
                // (no exception filter here: this method is patched by the combat cross-check tool, and Harmony cannot re-emit a
                // method that contains a filter block)
                if (e.Status != "stall" || !file.MapNode || runState.CurrentRoom is not EventRoom eventRoom)
                {
                    throw;
                }
                // The fight belongs to an event. The game records a replay's initial state when the map point is entered
                // (RunManager.EnterMapPointInternal, Runs/RunManager.cs:780-783) and afterwards only what happens while a combat is in
                // progress (CombatReplayWriter.cs:81/114/130/148); the event options that lead to the fight are in no replay event. The
                // game's own replay facility (NMultiplayerTest.RunReplay, :427-441) enters the map point - the event's first page - and
                // injects the combat actions at once; there is no fight to inject them into. Such a file cannot be re-executed from
                // itself. It is reported as its own status, never as a pass; its recorded states are compared inside the run by
                // --compare-run-traces-v2, where the event is played up to the fight.
                throw new ReplayStepException("event_fight", $"the replay's save enters the event {eventRoom.ModelId?.Entry}; the fight is started by an event option that the replay file does not record", null);
            }
            finally
            {
                CombatManager.Instance.CombatBegan -= onBegan;
            }
            if (enter.IsFaulted)
            {
                throw new ReplayStepException("fault", "entering the combat room faulted", enter.Exception!.GetBaseException());
            }
            if (runState.CurrentRoom is CombatRoom croom)
            {
                _r.Encounter = croom.Encounter.Id.Entry;
                _r.RoomType = croom.RoomType.ToString();
                _r.Act = runState.Act.Id.Entry;
            }
            Wait(() => !rm.ActionExecutor.IsPaused, "executor unpaused");

            _altered = false;
            InjectAll(rm, runState, replay);
            if (_opt.AlterEventIndex >= 0 && !_altered)
            {
                // F4: an alteration that was asked for and never applied (index beyond the recording) must not look like a pass
                throw new ReplayStepException("error", $"--alter-event {_opt.AlterEventIndex}: the recording has {replay.events.Count} events; nothing was altered", null);
            }

            // let the last actions finish (the recording ends with the combat or the death)
            WaitQuiet(() => !CombatManager.Instance.IsInProgress);
            _r.CombatEnded = !CombatManager.Instance.IsInProgress;
            _r.HostPlayerDead = runState.Players[0].Creature.IsDead;
            _r.PlayersDead = runState.Players.Count(p => p.Creature.IsDead);
            Classify();
        }
        catch (ReplayStepException e)
        {
            _r.Status = _r.Mismatched > 0 ? "diverged" : e.Status;
            _r.StatusDetail = e.Status + ": " + e.Message;
            if (e.InnerException != null)
            {
                Attach(e.InnerException);
            }
            _r.Diagnostics = Diagnose();
        }
        catch (Exception e)
        {
            _r.Status = _r.Mismatched > 0 ? "diverged" : "error";
            _r.StatusDetail = "host_exception: " + e.GetBaseException().Message;
            Attach(e);
            _r.Diagnostics = Diagnose();
        }
        finally
        {
            if (tracker != null && onChecksum != null)
            {
                tracker.ChecksumGenerated -= onChecksum;
            }
            scope?.Dispose();
            Teardown(rm);
            _run = null;
            ReplayInterventions.MultiPlayerReplayActive = false;
            _r.ActionsFinishedNotExecuting = ReplayInterventions.FinishedNotExecuting - finishedBefore;
        }
        _r.PostActionSubscriptions = scope?.Subscriptions ?? 0;
        _r.NetworkAttempts = SpireHost.Patches.NetworkGuard.NetworkAttempts - networkBefore;
        if (_r.NetworkAttempts != 0)
        {
            // Fail-closed: game code reached an outbound HTTP entry point (blocked by Core NetworkGuard). Never a pass.
            _r.Status = "error";
            _r.StatusDetail = "network_attempt: " + SpireHost.Patches.NetworkGuard.LastAttempt + (_r.StatusDetail != null ? " | " + _r.StatusDetail : "");
        }
        _r.TraceHash = _fold.Value;
        _r.Ms = Math.Round(sw.Elapsed.TotalMilliseconds, 2);
        return _r;
    }

    private void Classify()
    {
        if (_r.Mismatched > 0)
        {
            _r.Status = "diverged";
        }
        else if (_r.MissingInReplay > 0 || _r.ChecksumsGenerated != _r.ChecksumsRecorded)
        {
            _r.Status = "incomplete";
            _r.StatusDetail = $"generated {_r.ChecksumsGenerated} of {_r.ChecksumsRecorded} recorded checksums, {_r.MissingInReplay} ids not in the recording";
            _r.Diagnostics = Diagnose();
        }
        else if (_r.Extended.SigMismatches > 0)
        {
            // Verifier finding F3: the collector's public signature covers what no checksum does (RoundNumber, hand order as the
            // player sees it); a replay whose checksums all match but whose signature differs somewhere is not a pass.
            _r.Status = "sig_mismatch";
            _r.StatusDetail = _r.Extended.FirstSigMismatch;
        }
        else
        {
            _r.Status = "pass";
        }
    }

    private static void Preflight(McrFile f)
    {
        CombatReplay r = f.Replay;
        if (r.modelIdHash != ModelIdSerializationCache.Hash)
        {
            throw new ReplayStepException("error", $"modelIdHash {r.modelIdHash} != host {ModelIdSerializationCache.Hash}", null);
        }
        if (r.version != "v0.113.0" || r.gitCommit != "16ac0fe3")
        {
            throw new ReplayStepException("error", $"replay from {r.version}/{r.gitCommit}, host is v0.113.0/16ac0fe3", null);
        }
        if (!f.MapNode && (string.IsNullOrEmpty(f.Encounter) || f.Sidecar == null))
        {
            throw new ReplayStepException("error", "no sidecar with encounter + encounter_rng_seed (the save was recorded inside EnterRoomDebug and has no room)", null);
        }
        if (r.serializableRun.Players.Count < 1 || r.serializableRun.Players.Count > 4)
        {
            throw new ReplayStepException("error", "the replay's save has " + r.serializableRun.Players.Count + " players", null);
        }
    }

    /// <summary>
    /// Phase H5: the replay holds several players (written by a client of a real 2-4 player session). The call sequence is
    /// the same - NMultiplayerTest.RunReplay is the game's multi-player replay facility, every event names its player -
    /// with two differences from the single-player tool: (1) the timing is the game's own (wait only for the player side
    /// before a play-phase action): the collector's "a human could act now" predicate wants an empty action queue, and
    /// with several players an action of ANOTHER player may be paused there for a choice that arrives later in the
    /// recording; (2) the sidecar comparison (the collector's signature and option count of its OWN decisions) is not run:
    /// the recorded checksums cover every player's state after every action of every player.
    /// </summary>
    private bool _multi;

    private void Prepare(RunManager rm)
    {
        if (rm.IsInProgress)
        {
            Teardown(rm);
        }
        if (!_pump.IsQueueEmpty)
        {
            _pump.TryRunUntilQuiescent(_opt.MaxCallbacksPerWait, out _);
            _pump.DiscardQueued();
        }
        _pump.Clock.Clear();
        HostFaults.Drain();
        // Same profile for every replay (the save carries everything the combat reads; the profile is meta).
        VeteranProfile.Restore(0);
    }

    private void Teardown(RunManager rm)
    {
        try
        {
            if (rm.IsInProgress)
            {
                INetGameService net = rm.NetService;
                rm.CleanUp();
                NetLoadingHandle.Release(net);
            }
            _pump.TryRunUntilQuiescent(_opt.MaxCallbacksPerWait, out _);
            _pump.Clock.Clear();
            _pump.DiscardQueued();
        }
        catch (Exception e)
        {
            _r.Error ??= "teardown: " + e.GetBaseException().Message;
            try
            {
                _pump.Clock.Clear();
                _pump.DiscardQueued();
            }
            catch (Exception)
            {
                // nothing more to do; the next Prepare forces a clean-up again
            }
        }
        HostFaults.Drain();
    }

    // =====================================================================================================
    //  Event injection (NMultiplayerTest.RunReplay:441-480 + ReplayCheck StrictTiming)
    // =====================================================================================================

    private void InjectAll(RunManager rm, RunState runState, CombatReplay replay)
    {
        int policyAction = 0;
        JsonArray? actions = _file.Actions;
        _r.Extended.PolicyActions = actions?.Count ?? 0;
        for (int i = 0; i < replay.events.Count; i++)
        {
            if (_r.Mismatched > 0 && !_opt.ContinueAfterDivergence)
            {
                throw new ReplayStepException("diverged", "stopped at the first divergence", null);
            }
            CombatReplayEvent ev = replay.events[i];
            if (i == _opt.AlterEventIndex)
            {
                ev = Alter(ev, runState, i);
            }
            CombatManager cm = CombatManager.Instance;
            switch (ev.eventType)
            {
                case CombatReplayEventType.GameAction:
                {
                    Wait(() => !cm.EndingPlayerTurnPhaseOne && !cm.EndingPlayerTurnPhaseTwo, "turn phase end", i);
                    Player player = runState.GetPlayer(ev.playerId!.Value) ?? throw new ReplayStepException("error", "unknown player " + ev.playerId, null);
                    bool isPolicy = IsPolicyAction(ev.action);
                    if (isPolicy && !CombatManager.Instance.IsInProgress)
                    {
                        if (!_multi)
                        {
                            throw new ReplayStepException("incomplete", $"combat ended before event {i} ({ev.action})", null);
                        }
                        // Several players: an input of ANOTHER player can reach the host after the fight has ended (it was sent
                        // before that player's client knew). The event is injected like every other one; the queues cancel combat
                        // actions after CombatEnded (ActionQueueSet.cs:121-134, :326-355). Counted.
                        _r.EventsAfterCombatEnd++;
                    }
                    GameAction action = ev.action!.ToGameAction(player);
                    if (action.ActionType == GameActionType.CombatPlayPhaseOnly)
                    {
                        if (_opt.StrictTiming && !_multi)
                        {
                            Wait(() => !CombatManager.Instance.IsInProgress || ReadyForPlayerAction(player), "player-ready for replay action", i);
                        }
                        else
                        {
                            Wait(() => CombatManager.Instance.DebugOnlyGetState()?.CurrentSide != CombatSide.Enemy, "player side", i);
                        }
                    }
                    if (isPolicy && !_multi)
                    {
                        CheckExtended(actions, policyAction, player, ev.action!);
                        policyAction++;
                    }
                    if (ev.action is NetPlayCardAction play)
                    {
                        _lastPlayedCard = play.modelId.Entry;
                    }
                    _eventIndex = i;
                    _eventDesc = "GameAction " + action;
                    rm.ActionQueueSet.EnqueueWithoutSynchronizing(action);
                    _r.EventsInjected++;
                    if (action is EndPlayerTurnAction || action is ReadyToBeginEnemyTurnAction)
                    {
                        Task finished = rm.ActionExecutor.FinishedExecutingActions();
                        Wait(() => finished.IsCompleted, "executor finished (end turn)", i);
                        if (finished.IsFaulted)
                        {
                            throw new ReplayStepException("fault", "ActionExecutor faulted", finished.Exception!.GetBaseException());
                        }
                    }
                    break;
                }
                case CombatReplayEventType.HookAction:
                    _eventIndex = i;
                    _eventDesc = $"HookAction hook={ev.hookId} type={ev.gameActionType}";
                    rm.ActionQueueSet.EnqueueWithoutSynchronizing(rm.ActionQueueSynchronizer.GetHookActionForId(ev.hookId!.Value, ev.playerId!.Value, ev.gameActionType!.Value));
                    _r.EventsInjected++;
                    break;
                case CombatReplayEventType.ResumeAction:
                    _eventIndex = i;
                    _eventDesc = $"ResumeAction id={ev.actionId}";
                    rm.ActionQueueSet.ResumeActionWithoutSynchronizing(ev.actionId!.Value);
                    _r.EventsInjected++;
                    break;
                case CombatReplayEventType.PlayerChoice:
                {
                    Player player = runState.GetPlayer(ev.playerId!.Value) ?? throw new ReplayStepException("error", "unknown player " + ev.playerId, null);
                    _eventIndex = i;
                    _eventDesc = $"PlayerChoice id={ev.choiceId} result={ev.playerChoiceResult}";
                    rm.PlayerChoiceSynchronizer.ReceiveReplayChoice(player, ev.choiceId!.Value, ev.playerChoiceResult!.Value);
                    _r.EventsInjected++;
                    break;
                }
                default:
                    throw new ReplayStepException("error", "unknown replay event type " + ev.eventType, null);
            }
            // Drain what the injection started (inline execution) before looking at the next event; faults surface here.
            WaitQuiet(() => false);
        }
    }

    private static bool IsPolicyAction(INetAction? a)
    {
        string n = a?.GetType().Name ?? "";
        return n == "NetPlayCardAction" || n == "NetUsePotionAction" || n == "NetDiscardPotionGameAction" || n == "NetEndPlayerTurnAction";
    }

    /// <summary>The collector's / rerun's "a human could act now" predicate (ReplayCheck.ReadyForPlayerAction).</summary>
    public static bool ReadyForPlayerAction(Player p)
    {
        CombatManager cm = CombatManager.Instance;
        RunManager rm = RunManager.Instance;
        CombatState? cs = cm.DebugOnlyGetState();
        return cs != null && cs.CurrentSide == CombatSide.Player && p.PlayerCombatState?.Phase == PlayerTurnPhase.Play
            && rm.ActionQueueSynchronizer.CombatState == ActionSynchronizerCombatState.PlayPhase
            && !rm.ActionExecutor.IsRunning && rm.ActionExecutor.CurrentlyRunningAction == null && rm.ActionQueueSet.IsEmpty
            && !cm.EndingPlayerTurnPhaseOne && !cm.EndingPlayerTurnPhaseTwo && !cm.IsPlayerReadyToEndTurn(p);
    }

    /// <summary>
    /// Test hook: change one recorded event so that the comparison must fail (determinism cross-check, task 4). Only a card
    /// play can be altered (it becomes an end turn). Verifier finding F4: asked to alter any other event the tool used to
    /// change nothing and report the unaltered replay as "pass" - a negative test could pass vacuously. Now that is an error.
    /// </summary>
    private CombatReplayEvent Alter(CombatReplayEvent ev, RunState runState, int index)
    {
        if (ev.eventType == CombatReplayEventType.GameAction && ev.action is NetPlayCardAction)
        {
            Player p = runState.GetPlayer(ev.playerId!.Value)!;
            ev.action = new NetEndPlayerTurnAction { turnNumber = p.PlayerCombatState?.TurnNumber ?? 1 };
            _r.StatusDetail = "altered: play -> end turn at event " + index;
            _altered = true;
            return ev;
        }
        throw new ReplayStepException("error", $"--alter-event {index}: that event is not a card play ({ev.eventType}{(ev.action != null ? " " + ev.action.GetType().Name : "")}); nothing can be altered there, so the replay proves nothing", null);
    }

    private bool _altered;

    // =====================================================================================================
    //  Checksums
    // =====================================================================================================

    private void OnChecksum(NetChecksumData d, string context, NetFullCombatState state)
    {
        _r.ChecksumsGenerated++;
        _r.Checksums?.Add(d.checksum);
        _fold.Add(((ulong)d.id << 32) | d.checksum);
        if (_r.ExtendedTrace != null && _run != null)
        {
            ulong ext = ExtendedState.Hash(_run);
            _r.ExtendedTrace.Add(ext);
            _fold.Add(ext);
        }
        if (!_recordedById.TryGetValue(d.id, out ReplayChecksumData rec))
        {
            _r.MissingInReplay++;
            if (_r.First == null)
            {
                _r.First = new Divergence
                {
                    ChecksumOrdinal = -1,
                    ChecksumId = d.id,
                    LocalContext = context,
                    RecordedContext = "(no checksum with this id in the recording)",
                    EventIndex = _eventIndex,
                    Event = _eventDesc,
                    Local = d.checksum,
                    InvolvedCard = _lastPlayedCard,
                };
            }
            return;
        }
        uint expect = RunManager.Instance.ChecksumTracker.GenerateChecksum(rec.fullState);
        if (expect == d.checksum)
        {
            _r.Matched++;
            return;
        }
        _r.Mismatched++;
        if (_r.First != null && _r.First.ChecksumOrdinal >= 0)
        {
            return;
        }
        StateDiff.Result diff = StateDiff.Diff(rec.fullState, state, _opt.DiffLines);
        Divergence div = new Divergence
        {
            ChecksumOrdinal = _ordinalById[d.id],
            ChecksumId = d.id,
            LocalContext = context,
            RecordedContext = rec.context,
            EventIndex = _eventIndex,
            Event = _eventDesc,
            PlayerTurn = _run?.Players[0].PlayerCombatState?.TurnNumber ?? 0,
            Local = d.checksum,
            Recorded = expect,
            Diff = diff.Lines,
            DiffRemoved = diff.Removed,
            DiffAdded = diff.Added,
            DiffIds = diff.Ids,
            InvolvedCard = _lastPlayedCard ?? ActionFromContext(rec.context),
        };
        foreach (string id in diff.Ids)
        {
            if (id.StartsWith("POWER.", StringComparison.Ordinal))
            {
                div.InvolvedPowers.Add(id);
            }
        }
        _r.First = div;
        _firstLocalState = state;
    }

    private static string? ActionFromContext(string? ctx)
    {
        if (ctx == null || !ctx.StartsWith("finished action execution ", StringComparison.Ordinal))
        {
            return null;
        }
        string s = ctx.Substring(26);
        int i = s.IndexOfAny(new[] { ' ', '(', ':', '[' });
        return i > 0 ? s.Substring(0, i) : s;
    }

    // =====================================================================================================
    //  Extended comparison: the collector's per-decision public signature and legal-option count
    //  (sidecar actions[]: Sig contains RoundNumber, which the checksum does not cover; Options = count of legal
    //  plays x targets + potion uses + discards + end turn as enumerated with the game's own predicates).
    // =====================================================================================================

    private void CheckExtended(JsonArray? actions, int k, Player me, INetAction net)
    {
        if (actions == null || k >= actions.Count)
        {
            return;
        }
        CombatState? cs = CombatManager.Instance.DebugOnlyGetState();
        if (cs == null)
        {
            return;
        }
        JsonNode rec = actions[k]!;
        _r.Extended.Compared++;
        if (!CollectorDecision(me))
        {
            _r.Extended.NotADecisionPoint++;
        }
        string sig = Signature(me, cs);
        string recSig = rec["Sig"]?.GetValue<string>() ?? "";
        if (sig != recSig)
        {
            _r.Extended.SigMismatches++;
            _r.Extended.FirstSigMismatch ??= $"action {k} ({net.GetType().Name}): host '{sig}' recorded '{recSig}'";
        }
        int options = CountOptions(me, cs);
        int recOptions = rec["Options"]?.GetValue<int>() ?? -1;
        if (options == recOptions && _r.Extended.FirstMatchingPotionDecision < 0 && me.PotionSlots.Any(p => p != null))
        {
            _r.Extended.FirstMatchingPotionDecision = k;
        }
        if (options != recOptions)
        {
            _r.Extended.OptionMismatches++;
            int withoutPotions = CountOptions(me, cs, includePotions: false);
            if (recOptions == withoutPotions)
            {
                // the recording offered no potion option at all here (collector: !baseOk); see the report (room-transition potion lock)
                _r.Extended.OptionMismatchesPotionsLockedInRecording++;
            }
            if (_r.Extended.OptionMismatchList.Count < 64)
            {
                int potions = me.PotionSlots.Count(p => p != null);
                _r.Extended.OptionMismatchList.Add($"{k}:{me.PlayerCombatState?.TurnNumber}:{options}:{recOptions}:{potions}:{withoutPotions}");
            }
            _r.Extended.FirstOptionMismatch ??= $"action {k} ({net.GetType().Name}) sig '{sig}': host {options} options, recorded {recOptions} " +
                $"(potionUseLocked={RunManager.Instance.IsPotionUseLocked} changingRooms={RunManager.Instance.IsChangingRooms} canUseOrRemovePotions={me.CanUseOrRemovePotions})";
        }
    }

    /// <summary>CombatDriver.IsDecision of the collector minus its UI-node terms (no prompt open, no card in play animation).</summary>
    private static bool CollectorDecision(Player me)
    {
        CombatManager cm = CombatManager.Instance;
        if (!cm.IsInProgress || cm.IsOverOrEnding)
        {
            return false;
        }
        if (!ReadyForPlayerAction(me) || !me.Creature.IsAlive)
        {
            return false;
        }
        if (cm.PlayersTakingExtraTurn.Count > 0 && !cm.PlayersTakingExtraTurn.Contains(me))
        {
            return false;
        }
        return !cm.IsExecutingCardOrPotionEffect(me) && !cm.PlayerActionsDisabled;
    }

    /// <summary>The collector's public-state signature (CombatDriver.Signature, verbatim format).</summary>
    public static string Signature(Player me, CombatState cs)
    {
        StringBuilder sb = new StringBuilder();
        PlayerCombatState? pcs = me.PlayerCombatState;
        sb.Append($"r{cs.RoundNumber} t{pcs?.TurnNumber} e{pcs?.Energy} hp{me.Creature.CurrentHp}/{me.Creature.MaxHp} b{me.Creature.Block}");
        sb.Append(" H[").Append(string.Join(",", PileType.Hand.GetPile(me).Cards.Select(c => c.Id.Entry + (c.IsUpgraded ? "+" : "")))).Append(']');
        sb.Append($" D{PileType.Draw.GetPile(me).Cards.Count} X{PileType.Discard.GetPile(me).Cards.Count} Ex{PileType.Exhaust.GetPile(me).Cards.Count}");
        sb.Append(" P[").Append(string.Join(",", me.PotionSlots.Select(p => p?.Id.Entry ?? "-"))).Append(']');
        sb.Append(" M[").Append(string.Join(",", cs.Enemies.Select(e => $"{e.Monster?.Id.Entry}:{e.CurrentHp}/{e.MaxHp}:{e.Block}"))).Append(']');
        return sb.ToString();
    }

    /// <summary>The collector's legal-option count (CombatDriver.Enumerate; game predicates only; no prompt is open at a decision point).</summary>
    public static int CountOptions(Player me, CombatState cs, bool includePotions = true)
    {
        int n = 0;
        foreach (CardModel c in PileType.Hand.GetPile(me).Cards.ToList())
        {
            if (!c.CanPlay(out _, out _))
            {
                continue;
            }
            switch (c.TargetType)
            {
                case TargetType.AnyEnemy:
                    foreach (Creature e in cs.HittableEnemies)
                    {
                        if (c.CanPlayTargeting(e))
                        {
                            n++;
                        }
                    }
                    break;
                case TargetType.AnyAlly:
                    foreach (Creature a in cs.PlayerCreatures.Where(p => p.IsPlayer && p.IsAlive && p != me.Creature && p.IsHittable))
                    {
                        if (c.CanPlayTargeting(a))
                        {
                            n++;
                        }
                    }
                    break;
                default:
                    if (c.CanPlayTargeting(null))
                    {
                        n++;
                    }
                    break;
            }
        }
        CombatManager cm = CombatManager.Instance;
        bool baseOk = me.Creature.IsAlive && me.CanUseOrRemovePotions && !RunManager.Instance.IsPotionUseLocked;
        bool canUseInCombatNow = cm.IsInProgress && cs.CurrentSide == me.Creature.Side && me.Creature.IsAlive && !cm.PlayerActionsDisabled;
        for (int i = 0; i < me.PotionSlots.Count; i++)
        {
            PotionModel? p = me.PotionSlots[i];
            if (p == null || p.IsQueued || !baseOk || !includePotions)
            {
                continue;
            }
            bool useOk = p.PassesCustomUsabilityCheck && (p.Usage == PotionUsage.AnyTime || (p.Usage == PotionUsage.CombatOnly && canUseInCombatNow));
            if (useOk)
            {
                TargetType tt = p.TargetType;
                if (tt == TargetType.AnyEnemy)
                {
                    n += cs.HittableEnemies.Count;
                }
                else if (tt == TargetType.TargetedNoCreature || p.CanThrowAtAlly())
                {
                    // not reachable in single-player combat (the collector offers nothing)
                }
                else
                {
                    n++;
                }
            }
            n++; // discard
        }
        return n + 1; // end turn
    }

    // =====================================================================================================
    //  Waiting = pumping to quiescence until a predicate holds
    // =====================================================================================================

    /// <summary>Drain the pump (and fire virtual timers) until <paramref name="pred"/> holds; a quiescent engine without the predicate is a stall.</summary>
    private void Wait(Func<bool> pred, string what, int eventIndex = -1)
    {
        if (!PumpUntil(pred, out string why))
        {
            throw new ReplayStepException(why == "fault" ? "fault" : "stall", $"waiting for '{what}' before event {eventIndex}: {why}", why == "fault" ? FirstFault() : null);
        }
    }

    /// <summary>Drain to full quiescence (no queue, no timers) or until <paramref name="stop"/>; a fault throws.</summary>
    private void WaitQuiet(Func<bool> stop)
    {
        if (!PumpUntil(stop, out string why) && why != "quiescent")
        {
            throw new ReplayStepException(why == "fault" ? "fault" : "stall", "draining: " + why, why == "fault" ? FirstFault() : null);
        }
    }

    private Exception? _pendingFault;

    private Exception? FirstFault()
    {
        Exception? e = _pendingFault;
        _pendingFault = null;
        return e;
    }

    /// <summary>
    /// EnginePump.DrainUntil (replay request R1) with the tool's fault rule: a fault recorded by the engine stops the
    /// wait before the predicate is looked at.
    /// </summary>
    private bool PumpUntil(Func<bool> pred, out string why)
    {
        bool fault = false;
        PumpStop stop = _pump.DrainUntil(() =>
        {
            if (HostFaults.Count > 0)
            {
                fault = true;
                return true;
            }
            return pred();
        }, _opt.MaxCallbacksPerWait, _opt.VirtualTimeBudget);
        if (fault)
        {
            List<Exception> f = HostFaults.Drain();
            _pendingFault = f[0];
            why = "fault";
            return false;
        }
        if (stop == PumpStop.Decision)
        {
            why = "ok";
            return true;
        }
        why = _pump.LastStuckReason;
        return false;
    }

    private void Drive(Task task, string what)
    {
        Wait(() => task.IsCompleted, what);
        if (task.IsFaulted)
        {
            throw new ReplayStepException("fault", what + " faulted", task.Exception!.GetBaseException());
        }
    }

    // =====================================================================================================
    //  Reporting helpers
    // =====================================================================================================

    private void Attach(Exception e)
    {
        Exception root = e.GetBaseException();
        _r.Error = root.GetType().Name + ": " + root.Message;
        _r.ErrorType = root.GetType().Name;
        string chain = e.ToString();
        _r.ErrorChain = chain.Length > 6000 ? chain.Substring(0, 6000) : chain;
        _r.Frames = FirstFrames(root, 8);
    }

    public static List<string> FirstFrames(Exception root, int max)
    {
        List<string> frames = new List<string>();
        StackTrace st = new StackTrace(root, false);
        List<string> all = new List<string>();
        for (int i = 0; i < st.FrameCount; i++)
        {
            System.Reflection.MethodBase? m = st.GetFrame(i)?.GetMethod();
            if (m == null)
            {
                continue;
            }
            string type = m.DeclaringType?.FullName ?? "?";
            if (type.StartsWith("System.Runtime.CompilerServices", StringComparison.Ordinal) || type.StartsWith("System.Threading.Tasks", StringComparison.Ordinal))
            {
                continue;
            }
            all.Add(type + "." + m.Name);
        }
        int game = all.FindIndex(s => s.StartsWith("MegaCrit", StringComparison.Ordinal));
        if (game > 0)
        {
            frames.Add(all[game]);
        }
        foreach (string s in all)
        {
            if (frames.Count >= max)
            {
                break;
            }
            if (game > 0 && s == all[game] && frames.Count == 1)
            {
                continue;
            }
            frames.Add(s);
        }
        return frames;
    }

    private string Diagnose()
    {
        try
        {
            RunManager rm = RunManager.Instance;
            CombatManager cm = CombatManager.Instance;
            Player? p = _run?.Players.FirstOrDefault();
            CombatState? cs = cm.DebugOnlyGetState();
            return $"event={_eventIndex} ({_eventDesc}) inProgress={cm.IsInProgress} side={cs?.CurrentSide} round={cs?.RoundNumber} phase={p?.PlayerCombatState?.Phase} turn={p?.PlayerCombatState?.TurnNumber} " +
                   $"sync={rm.ActionQueueSynchronizer?.CombatState} execRunning={rm.ActionExecutor?.IsRunning} execPaused={rm.ActionExecutor?.IsPaused} current={rm.ActionExecutor?.CurrentlyRunningAction} " +
                   $"queueEmpty={rm.ActionQueueSet?.IsEmpty} readyToEnd={(p != null && cm.IsInProgress ? cm.IsPlayerReadyToEndTurn(p) : false)} p1={cm.EndingPlayerTurnPhaseOne} p2={cm.EndingPlayerTurnPhaseTwo} " +
                   $"hp={p?.Creature.CurrentHp} timers={_pump.Clock.Pending} generated={_r.ChecksumsGenerated}/{_r.ChecksumsRecorded}";
        }
        catch (Exception e)
        {
            return "diagnose failed: " + e.Message;
        }
    }

    private static void FillMeta(McrFile f, McrReplayResult r)
    {
        JsonNode? s = f.Sidecar;
        if (s == null)
        {
            // run-trace combat: <trace dir>\NNN-combat-fN.mcr
            MegaCrit.Sts2.Core.Saves.Runs.SerializablePlayer? p = f.Replay.serializableRun.Players.FirstOrDefault();
            r.Character = p?.CharacterId?.Entry ?? "";
            r.Ascension = f.Replay.serializableRun.Ascension;
            r.Block = "run-trace";
            string parent = System.IO.Path.GetDirectoryName(f.Path) ?? "";
            string parentName = System.IO.Path.GetFileName(parent);
            r.File = parentName + "/" + f.Name;
            if (parentName.Length == 2 && parentName[0] == 'c' && char.IsDigit(parentName[1]))
            {
                // a client directory of a multi-client session: <session>\c<k>\NNN-combat-*.mcr (the lost fight is NNN-combat-final.mcr)
                r.Block = "mp-session";
                r.File = System.IO.Path.GetFileName(System.IO.Path.GetDirectoryName(parent)) + "/" + r.File;
            }
            r.RecordedResult = f.Name.Contains("final-loss", StringComparison.Ordinal) || (r.Block == "mp-session" && f.Name.EndsWith("-final", StringComparison.Ordinal)) ? "loss" : "win";
            r.RecordedPlayerDead = r.RecordedResult == "loss";
            return;
        }
        r.Index = s["index"]?.GetValue<int>() ?? -1;
        r.Character = s["character"]?.GetValue<string>() ?? "";
        r.Ascension = s["ascension"]?.GetValue<int>() ?? 0;
        r.Encounter = f.Encounter;
        r.RoomType = s["encounter_room_type"]?.GetValue<string>() ?? "";
        r.Act = s["encounter_act"]?.GetValue<string>() ?? "";
        r.RecordedResult = s["result"]?.GetValue<string>() ?? "";
        r.RecordedPlayerDead = s["player_dead"]?.GetValue<bool>() ?? false;
        r.Block = r.Index < 1000 ? "A" : r.Index < 2000 ? "B" : "C"; // worklist-600 (0..), worklist-B (1000..), worklist-C (2000..)
    }
}

public sealed class ReplayStepException : Exception
{
    public string Status { get; }

    public ReplayStepException(string status, string message, Exception? inner) : base(message, inner)
    {
        Status = status;
    }
}

/// <summary>FNV-1a style 64-bit fold (host-only trace hash).</summary>
public struct SigFold
{
    public ulong Value;

    public static SigFold Create() => new SigFold { Value = 14695981039346656037UL };

    public void Add(ulong x)
    {
        for (int i = 0; i < 8; i++)
        {
            Value ^= (x >> (i * 8)) & 0xFF;
            Value *= 1099511628211UL;
        }
    }
}
