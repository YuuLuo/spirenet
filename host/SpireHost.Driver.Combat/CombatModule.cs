using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Patches;

namespace SpireHost.Driver.Combat;

/// <summary>
/// The combat module: the COMBAT room handler and the card-selection machinery of the run.
///
///   room:Combat                  <see cref="CombatHandler"/>
///   lifecycle combat:lifecycle   <see cref="CombatLifecycle"/>    installs the ICardSelector of the run (UseLocalSelector),
///                                                                 publishes ctx.CardSelector, audits submitted actions
///   module counters "combat"     <see cref="CombatCounters"/>
///   patches                      15 read-only CardSelectCmd.From* captures (every overload that asks an
///                                ICardSelector; NOT the bundle entry, which the events module owns).
///                                Ledger: host\FIDELITY.md
///
/// One configuration: the module has no modes and reads no environment variable. (The combat-end scheduling
/// boundary and the intent tracker moved to the core with the H1 integration: run-save-completes-after-pump-turn,
/// enemy-turn-intent-performed.)
/// </summary>
public sealed class CombatModule : IDriverModule
{
    public string Name => "combat";

    public void Register(DriverRegistry registry)
    {
        SelectionCapture.Install();
        PatchRegistry.Note("combat-local-card-selector", "MegaCrit.Sts2.Core.CardSelection.CardSelectorStack::UseLocalSelector", "game test hook (per run)", PatchCategory.Plumbing,
            "The run's ICardSelector is installed with CardSelectorStack.UseLocalSelector (CardSelectorStack.cs:180-188), the path on which CardSelectCmd runs ReserveChoiceId / SignalPlayerChoiceBegun / SignalPlayerChoiceEnded exactly as with the real screens (CardSelectCmd.cs:66-98); only the screen is replaced by a pending CARD_SELECT frame. The global _testSelector and UseSelector are never used (DECISIONS-H1 B.2).");

        PatchRegistry.Note("combat-action-enqueued-audit", "MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionQueueSet::ActionEnqueued", "event subscription (per run)", PatchCategory.Instrumentation,
            "Read-only: the combat driver watches ActionQueueSet.ActionEnqueued (ActionQueueSet.cs:71, 102) to verify that every action it submits reaches a queue, is not cancelled on arrival (ActionQueueSet.cs:105-131) and has finished at the next COMBAT decision. A violation aborts the episode (ActionFault). Unsubscribed at episode end.");
        PatchRegistry.Note("combat-ready-to-begin-enemy-turn-relay", "MegaCrit.Sts2.Core.GameActions.ReadyToBeginEnemyTurnAction", "client-by-proxy (multi-seat only, inactive with one seat)", PatchCategory.Plumbing,
            "MULTI-SEAT ONLY (phase H5; never runs with one seat): the host instance enqueues ReadyToBeginEnemyTurnAction for its local player only, at the end of the player turn's phase one (CombatManager.cs:1190-1193), while SetReadyToBeginEnemyTurn waits for every player, dead ones included (:797). Every real client reaches the same line of its own simulation and sends its own ready action (RequestEnqueueActionMessage{NetReadyToBeginEnemyTurnAction}). When the driver sees seat 0's action in ActionQueueSet.ActionEnqueued it posts, to the pump, 'the client of seat k sends its ReadyToBeginEnemyTurnAction' for every other seat in seat order (VirtualClient.SendAction; actionDuringEnemyTurn is lost on the wire, NetReadyToBeginEnemyTurnAction.cs:11-14, and no caller of v0.113 passes one). Self-check: --selftest-mp handshake (the enemy turn starts only after N ready actions).");

        CombatLifecycle lifecycle = new CombatLifecycle();
        registry.SetLifecycle(lifecycle);
        registry.SetRemoteChoiceRouter(new CardSelectRemoteRouter(lifecycle));
        registry.SetRoomHandler(RoomScreen.Combat, new CombatHandler(lifecycle), "combat");
        registry.SetModuleCounters(new CombatCounters());
    }
}

/// <summary>
/// The counters of the combat module (contract request C4): what was submitted, the "zero rejected / cancelled
/// actions" invariants, CARD_SELECT frames by kind / context / entry point, the class provider.
/// </summary>
public sealed class CombatCounters : IModuleCounters
{
    public string Module => "combat";

    public void Read(IDictionary<string, long> c)
    {
        c["decisions"] = CombatStats.Decisions;
        c["plays_submitted"] = CombatStats.PlaysSubmitted;
        c["potions_used"] = CombatStats.PotionsUsed;
        c["potions_discarded"] = CombatStats.PotionsDiscarded;
        c["end_turns"] = CombatStats.EndTurns;
        c["undo_end_turns"] = CombatStats.UndoEndTurns;
        // hard requirement: all of these stay 0 (each one also aborts the episode)
        c["violation.rejected"] = CombatStats.Rejected;
        c["violation.not_enqueued"] = CombatStats.NotEnqueued;
        c["violation.cancelled"] = CombatStats.Cancelled;
        c["violation.not_finished"] = CombatStats.NotFinished;
        c["unrecognised_states"] = CombatStats.UnrecognisedStates;

        c["select.frames"] = CardSelectStats.Frames;
        c["select.frames_merged"] = CardSelectStats.FramesMerged;
        c["select.instances_merged_away"] = CardSelectStats.ClassesMerged;
        c["select.picks"] = CardSelectStats.Picks;
        c["select.completed"] = CardSelectStats.Completed;
        c["select.completed_empty"] = CardSelectStats.CompletedEmpty;
        c["select.revalidated_empty"] = CardSelectStats.RevalidatedEmpty;
        c["select.auto_completed"] = CardSelectStats.AutoCompleted;
        c["select.softlock_avoided"] = CardSelectStats.SoftlockAvoided;
        c["select.only_cancel"] = CardSelectStats.OnlyCancel;
        c["select.frame_depth_max"] = CardSelectStats.MaxFrameDepth;
        for (int i = 1; i < CardSelectStats.FramesByKind.Length; i++)
        {
            c["select.kind." + (CardSelectKind)i] = CardSelectStats.FramesByKind[i];
        }
        for (int i = 0; i < CardSelectStats.FramesByContext.Length; i++)
        {
            c["select.context." + (ChoiceContextKind)i] = CardSelectStats.FramesByContext[i];
        }
        for (int i = 1; i < SelectionCapture.EntryCalls.Length; i++)
        {
            c["select.entry_calls." + (CardSelectEntry)i] = SelectionCapture.EntryCalls[i];
            c["select.entry_asked." + (CardSelectEntry)i] = SelectionCapture.EntryAsked[i];
        }
        c["select.class_provider_calls"] = CanonicalCandidates.ProviderCalls;
        c["select.class_provider_pair_calls"] = CanonicalCandidates.ProviderPairCalls;
        // the rebuild of a prompt's candidate list (remote seats), checked against the game at every local prompt
        c["select.rebuild.local_checks"] = RemoteCandidates.LocalChecks;
        c["violation.select_rebuild_mismatch"] = RemoteCandidates.LocalMismatches;
        c["select.encoding.local_checks"] = RemoteCandidates.EncodingChecks;
        c["violation.select_encoding_mismatch"] = RemoteCandidates.EncodingMismatches;
        c["select.remote_frames"] = CardSelectStats.RemoteFrames;
        c["select.remote_answers"] = CardSelectStats.RemoteAnswers;
    }
}

/// <summary>
/// Per-run hooks of the combat driver. OnRunSetUp runs after RunManager.SetUpNewSingleplayer (the per-run
/// CardSelectorStack and ActionQueueSet exist) and before FinalizeStartingRelics / Launch / EnterAct(0), so
/// the selector is in place before any game code can ask for a card (driver-spec-combat.md 8.3-11).
/// </summary>
public sealed class CombatLifecycle : IRunLifecycle
{
    public const string LifecycleName = "combat:lifecycle";

    private IDisposable? _selectorScope;
    private ActionQueueSet? _queues;
    private Action<GameAction>? _onEnqueued;
    private RunContext? _ctx;

    public string Name => LifecycleName;

    public SubmissionAudit Audit { get; } = new SubmissionAudit();

    /// <summary>
    /// The prompt owner of each seat (index = seat). The local seat's is the run's ICardSelector (UseLocalSelector); a
    /// remote seat's is never installed in the game: it parks the frames of that seat's remote prompts (HostCardSelector.OpenRemote).
    /// </summary>
    public HostCardSelector?[] Selectors { get; private set; } = Array.Empty<HostCardSelector?>();

    public void OnRunSetUp(RunContext ctx)
    {
        _ctx = ctx;
        SelectionCapture.ResetForRun();
        Audit.Reset();

        RunManager rm = ctx.RunManager;
        Selectors = new HostCardSelector?[ctx.SeatCount];
        int localSeat = ctx.SeatOf(ctx.LocalPlayer);
        HostCardSelector local = new HostCardSelector(ctx, localSeat);
        Selectors[localSeat] = local;
        // Seat 0 = the local player: the local-selector path. Seats 1..3 (phase H5) answer through the remote path
        // (the game's WaitForRemoteChoice, answered by the seat's PlayerChoiceMessage), never through a selector.
        _selectorScope = rm.CardSelectorStack.UseLocalSelector(local);
        ctx.CardSelector = local;
        for (int s = 0; s < Selectors.Length; s++)
        {
            Selectors[s] ??= new HostCardSelector(ctx, s);
        }

        _queues = rm.ActionQueueSet;
        _onEnqueued = OnActionEnqueued;
        _queues.ActionEnqueued += _onEnqueued;
    }

    public void OnRoomEntered(RunContext ctx)
    {
        // A new room = a new combat at most: the "last submitted action" of the previous combat is not checked
        // against the next combat's first decision.
        Audit.Reset();
    }

    public void OnEpisodeEnd(RunContext ctx)
    {
        try
        {
            _selectorScope?.Dispose();
        }
        finally
        {
            _selectorScope = null;
            if (_queues != null && _onEnqueued != null)
            {
                _queues.ActionEnqueued -= _onEnqueued;
            }
            _queues = null;
            _onEnqueued = null;
            for (int i = 0; i < Selectors.Length; i++)
            {
                Selectors[i]?.Reset();
            }
            Selectors = Array.Empty<HostCardSelector?>();
            if (ReferenceEquals(ctx.CardSelector, null) == false)
            {
                ctx.CardSelector = null;
            }
            Audit.Reset();
            SelectionCapture.ResetForRun();
            _ctx = null;
        }
    }

    /// <summary>
    /// ActionQueueSet.ActionEnqueued (ActionQueueSet.cs:71, 102). Two jobs:
    ///   1. the submission audit (single seat and multi seat);
    ///   2. MULTI-SEAT ONLY (phase H5, not reachable with one seat): the host instance enqueues
    ///      ReadyToBeginEnemyTurnAction for its local player only (CombatManager.cs:1190-1193) while
    ///      SetReadyToBeginEnemyTurn waits for every player (:797). In the real game each client sends its
    ///      own; the driver acts for the other seats and relays it - from the pump queue, never from inside
    ///      this event (driver-spec-combat.md 7-4). actionDuringEnemyTurn is null: no caller of
    ///      PlayerCmd.EndTurn passes one in v0.113.
    /// </summary>
    private void OnActionEnqueued(GameAction action)
    {
        Audit.OnEnqueued(action);
        RunContext? ctx = _ctx;
        if (ctx == null || ctx.SeatCount <= 1 || action is not ReadyToBeginEnemyTurnAction)
        {
            return;
        }
        Player local = ctx.LocalPlayer;
        if (action.OwnerId != local.NetId)
        {
            return;
        }
        MultiSeatRun? mp = ctx.Mp;
        if (mp == null)
        {
            return;
        }
        ctx.Post(() =>
        {
            if (!ctx.CombatManager.IsInProgress)
            {
                return Task.CompletedTask;
            }
            for (int s = 1; s < ctx.SeatCount; s++)
            {
                MultiSeatStats.ReadyRelays++;
                mp.Client(s).SendAction(new ReadyToBeginEnemyTurnAction(ctx.Seats[s]));
            }
            return Task.CompletedTask;
        }, "the other seats' clients send ReadyToBeginEnemyTurn", blocking: false);
    }
}
