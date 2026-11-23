using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using HarmonyLib;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Patches;

namespace SpireHost.Replay;

/// <summary>
/// The two interventions the .mcr oracle needs to see the game's own per-action checksums under TestMode.
/// They are installed ONLY by this team's replay tools (--replay-mcr, --selftest), never by the module's
/// Register (so they are not part of the default patch set), and only for the run the replay tool itself
/// sets up. Both are registered in PatchRegistry and in host\FIDELITY.replay.md.
///
///   replay-checksum-tracker-enabled   RunManager.InitializeShared sets ChecksumTracker.IsEnabled = !TestMode.IsOn
///                                     (RunManager.cs:385). The replay tool sets the public setter back to true
///                                     after SetUpReplay (what oracle-mod.md 10.3 prescribes). The tracker only reads
///                                     state (NetFullCombatState.FromRun) and counts ids; NextId is not game state.
///
///   replay-post-action-checksum       The real (interactive) branch of ActionExecutor.ExecuteActions subscribes
///                                     readyAction.JustBeforeFinished += JustBeforeActionFinished (ActionExecutor.cs:127)
///                                     and unsubscribes after the action (:158); the NonInteractive branch taken under
///                                     TestMode (:118-123) does not, so RunManager.SendPostActionChecksum (RunManager.cs:
///                                     403, 485-491) never runs. The tool subscribes the executor's own private handler
///                                     JustBeforeActionFinished to every action from the public ActionExecutor.BeforeActionExecuted
///                                     event (raised at :117, immediately before the real branch subscribes), removing a
///                                     previous subscription first so that an action resumed after a player choice is
///                                     subscribed once, exactly like the real branch. The handler is the game's: it checks
///                                     CurrentlyRunningAction == action and State == Finished and raises
///                                     JustBeforeActionFinishedExecuting -> SendPostActionChecksum. Read-only w.r.t. game state.
/// </summary>
public static class ReplayInterventions
{
    private static bool _registered;
    private static bool _registeredSync;

    public const string TrackerId = "replay-checksum-tracker-enabled";
    public const string PostActionId = "replay-post-action-checksum";
    public const string SyncDisabledId = "replay-combat-state-sync-disabled";

    /// <summary>
    /// Records the interventions in PatchRegistry (once per process; called by the replay tools only).
    /// <paramref name="replayRun"/> false: only the two read-only checksum interventions. The v2 run-trace comparison enables
    /// the tracker in the RunDriver episode it replays; it sets up no replay run, so CombatStateSynchronizer.IsDisabled is
    /// not touched and not recorded.
    /// </summary>
    public static void Register(bool replayRun = true)
    {
        if (!_registered)
        {
            _registered = true;
            PatchRegistry.Note(TrackerId, "MegaCrit.Sts2.Core.Multiplayer.Game.ChecksumTracker::IsEnabled (public setter)", "property write (replay tool only)", PatchCategory.Instrumentation,
                "RunManager.InitializeShared disables the checksum tracker under TestMode (RunManager.cs:385). The .mcr replay tool re-enables it for the replay run it sets up " +
                "(oracle-mod.md 10.3), and the v2 run-trace comparison for the episode it replays (the fights' recorded per-action states are compared inside the run). " +
                "Read-only: NetFullCombatState.FromRun + XxHash32; ChecksumTracker.NextId is not game state. Not in the default patch set.");
            PatchRegistry.Note(PostActionId, "MegaCrit.Sts2.Core.GameActions.ActionExecutor::BeforeActionExecuted (public event)", "event subscription (replay tool only)", PatchCategory.Instrumentation,
                "Restores the real branch's per-action subscription readyAction.JustBeforeFinished += ActionExecutor.JustBeforeActionFinished (ActionExecutor.cs:127/158) that the " +
                "NonInteractive branch (:118-123) omits, so RunManager.SendPostActionChecksum (RunManager.cs:485-491) produces the 'finished action execution <action>' checksums " +
                "recorded in every .mcr. The game's own handler runs (same guards). Not in the default patch set.");
        }
        if (replayRun && !_registeredSync)
        {
            _registeredSync = true;
            PatchRegistry.Note(SyncDisabledId, "MegaCrit.Sts2.Core.Multiplayer.Game.CombatStateSynchronizer::IsDisabled (public setter)", "property write (replay tool only)", PatchCategory.Plumbing,
                "The replay run is set up exactly like the real client's replay facility: RunManager.SetUpReplay (NetReplayGameService) + CombatStateSynchronizer.IsDisabled = true " +
                "(NMultiplayerTest.cs:417-419; the oracle's ReplayCheck.Rerun does the same). No peer exists to sync with. Not in the default patch set.");
        }
    }

    // =====================================================================================================
    //  Multi-player replays (phase H5): the recordings of real 2-4 player sessions
    // =====================================================================================================

    public const string MpFinishedStateId = "replay-mp-action-finished-state-as-production";
    private const string MonitorId = "fidelity-monitor-action-finished-bad-state";
    private static bool _registeredMp;
    private static MethodInfo? _monitorMethod;
    private static Func<bool>? _monitor;

    /// <summary>True while a multi-player replay is being re-executed (set by McrReplayer around the replay).</summary>
    public static bool MultiPlayerReplayActive;

    /// <summary>Actions that finished while not in state Executing during multi-player replays (what the real client logs and survives).</summary>
    public static long FinishedNotExecuting;

    /// <summary>
    /// Installed by the replay tool the first time it meets a replay with several players; never by the module's Register.
    ///
    /// GameAction.Execute (GameActions/GameAction.cs:91-106): an action that finishes while its state is not Executing
    /// throws under NonInteractiveMode; the real client logs an error and goes on. With one player the host never reaches
    /// it (the production monitor fidelity-monitor-action-finished-bad-state counts 0 over every corpus and fuzz) and keeps
    /// it fail-closed. In real multi-player sessions it is ordinary: PlayCardAction.ExecuteAction cancels ITSELF when its
    /// card is not playable or its target is not valid any more at the moment the action's turn in the queue comes
    /// (GameActions/PlayCardAction.cs:81-85) - two players act at the same time, one kills the other's target. The
    /// recorded session went on from there, so the replay must too: while a multi-player replay runs, the one call the
    /// production monitor put there evaluates to false (the real client's branch: Log.Error, no throw). Outside such a
    /// replay the call is the production monitor's, unchanged.
    /// </summary>
    public static void RegisterMultiPlayer()
    {
        if (_registeredMp)
        {
            return;
        }
        _registeredMp = true;
        Type? fidelity = null;
        foreach (Assembly a in AppDomain.CurrentDomain.GetAssemblies())
        {
            fidelity = a.GetType("SpireHost.Fidelity.Patches.FidelityPatches", false);
            if (fidelity != null)
            {
                break;
            }
        }
        _monitorMethod = fidelity?.GetMethod("ActionFinishedInBadState", BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException("replay-mcr (multi-player replay): the production monitor FidelityPatches.ActionFinishedInBadState is not loaded.");
        _monitor = (Func<bool>)Delegate.CreateDelegate(typeof(Func<bool>), _monitorMethod);
        MethodInfo exec = AccessTools.Method(typeof(GameAction), nameof(GameAction.Execute)) ?? throw new MissingMethodException("GameAction.Execute");
        PatchRegistry.DeclareSharedSite("The production monitor counts an action that finishes in a state other than Executing and keeps the NonInteractive throw; the replay tool's transpiler " +
                                        "replaces the monitor's call for the time a multi-player replay runs (the real client's branch).", MonitorId, MpFinishedStateId);
        PatchRegistry.Transpiler(MpFinishedStateId, AccessTools.AsyncMoveNext(exec), PatchRegistry.M(typeof(ReplayInterventions), nameof(FinishedStateTranspiler)), PatchCategory.Fidelity,
            "REPLAY TOOL ONLY, multi-player replays only (--replay-mcr on the recordings of real 2-4 player sessions). GameAction.Execute (GameActions/GameAction.cs:91-106): an action " +
            "that finishes while its state is not Executing throws under NonInteractiveMode and is only logged by the real client. In real multi-player sessions that is an ordinary race: " +
            "PlayCardAction.ExecuteAction cancels itself when the card is not playable or its target not valid any more when its turn in the queue comes (GameActions/PlayCardAction.cs:81-85). " +
            "Transpiler: the one call the production monitor fidelity-monitor-action-finished-bad-state put there is replaced (exactly 1) by a helper that returns false while a multi-player " +
            "replay is running (the real client's branch: Log.Error, no throw) and calls the monitor otherwise. Not in the default patch set; production stays fail-closed.");
        PatchRegistry.AssertOnePatchPerSite();
    }

    [HarmonyPriority(Priority.Last)]
    public static IEnumerable<CodeInstruction> FinishedStateTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        List<CodeInstruction> code = new List<CodeInstruction>(instructions);
        MethodInfo to = PatchRegistry.M(typeof(ReplayInterventions), nameof(FinishedInBadState));
        int hits = 0;
        foreach (CodeInstruction ci in code)
        {
            if ((ci.opcode == OpCodes.Call || ci.opcode == OpCodes.Callvirt) && ci.operand is MethodInfo m && m == _monitorMethod)
            {
                ci.opcode = OpCodes.Call;
                ci.operand = to;
                hits++;
            }
        }
        if (hits != 1)
        {
            throw new InvalidOperationException(MpFinishedStateId + ": expected exactly 1 call of the production monitor in GameAction.Execute, found " + hits + ".");
        }
        return code;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static bool FinishedInBadState()
    {
        if (!MultiPlayerReplayActive)
        {
            return _monitor!();
        }
        FinishedNotExecuting++;
        return false;
    }

    /// <summary>A per-run subscription; dispose after the replay (the executor is per run anyway).</summary>
    public sealed class Scope : IDisposable
    {
        private readonly ActionExecutor _executor;
        private readonly Action<GameAction> _before;
        private readonly Action<GameAction> _justBefore;

        public long Subscriptions;

        public Scope(RunManager rm, bool replayRun = true)
        {
            Register(replayRun);
            rm.ChecksumTracker.IsEnabled = true;
            _executor = rm.ActionExecutor;
            _justBefore = _executor.JustBeforeActionFinished; // the executor's own (private) handler, ActionExecutor.cs:177-187
            _before = OnBeforeActionExecuted;
            _executor.BeforeActionExecuted += _before;
        }

        private void OnBeforeActionExecuted(GameAction action)
        {
            action.JustBeforeFinished -= _justBefore;
            action.JustBeforeFinished += _justBefore;
            Subscriptions++;
        }

        public void Dispose()
        {
            _executor.BeforeActionExecuted -= _before;
        }
    }
}
