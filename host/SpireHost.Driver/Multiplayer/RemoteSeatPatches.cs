using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using SpireHost.Patches;

namespace SpireHost.Driver;

/// <summary>
/// The observation points of the virtual clients (phase H5; installed by HostSession in every mode, inert unless the
/// running episode is a multi-seat episode - <see cref="RunContext.Mp"/> is null otherwise, and outside an episode).
///
/// A real client plays the same simulation in lockstep and sends a message at four points of the choice protocol.
/// The host's own simulation passes the same points for the remote player without sending anything (the game sends
/// only for the local owner, or for everybody when the test flag TestFlags.ShouldSendResumeForRemotePlayers is set,
/// which the host keeps off: FidelityInvariants "test_flags_inert"). Each patch below is a READ-ONLY prefix / postfix on
/// one of those points: no argument, result or game state is touched; it posts "the seat's client sends X" to the
/// engine pump, so the message arrives - like a real packet - after the host's own call has returned.
///
///   mp-remote-choice-wait         PlayerChoiceSynchronizer.WaitForRemoteChoice(player, choiceId) (:78-111): the host
///                                 starts to wait for the remote player's answer = that player's client is showing the
///                                 prompt. The registered <see cref="IRemoteChoiceRouter"/>s turn it into a pending frame
///                                 (they know which prompt it is); the answer is the client's PlayerChoiceMessage.
///   mp-remote-hook-enqueue        GenericHookGameAction.SetChoiceContext (:41-49), called by
///                                 HookPlayerChoiceContext.SignalPlayerChoiceBegun right before
///                                 "if (owner == local) RequestEnqueueHookAction" (:165-169): the remote owner's client
///                                 sends RequestEnqueueHookActionMessage{HookId, ActionType}.
///   mp-remote-resume-game-action  GameActionPlayerChoiceContext.SignalPlayerChoiceEnded (:63-70) and
///   mp-remote-resume-hook         HookPlayerChoiceContext.SignalPlayerChoiceEnded (:176-182): "if (owner == local)
///                                 RequestResumeActionAfterPlayerChoice": the remote owner's client sends
///                                 RequestResumeActionAfterPlayerChoiceMessage{action id}.
///   mp-sync-identity-check        Player.SyncWithSerializedPlayer (Player.cs:325-365): the rebuild of a remote player
///                                 from its client's data must not change what a run signature sees of that player.
///
/// The fifth client duty in combat, ReadyToBeginEnemyTurnAction, is relayed by the combat module
/// (combat-ready-to-begin-enemy-turn-relay) and every player action is sent by the handler that applies it.
/// </summary>
internal static class RemoteSeatPatches
{
    private static bool _installed;

    public const string ChoiceWaitId = "mp-remote-choice-wait";
    public const string HookEnqueueId = "mp-remote-hook-enqueue";
    public const string ResumeGameActionId = "mp-remote-resume-game-action";
    public const string ResumeHookId = "mp-remote-resume-hook";
    public const string SyncCheckId = "mp-sync-identity-check";

    public static void Install()
    {
        if (_installed)
        {
            return;
        }
        _installed = true;

        PatchRegistry.Prefix(ChoiceWaitId,
            AccessTools.Method(typeof(PlayerChoiceSynchronizer), nameof(PlayerChoiceSynchronizer.WaitForRemoteChoice), new[] { typeof(Player), typeof(uint) }),
            PatchRegistry.M(typeof(RemoteSeatPatches), nameof(WaitForRemoteChoicePrefix)), PatchCategory.Instrumentation,
            "MULTI-SEAT ONLY (phase H5; inert with one seat and outside an episode). Read-only: PlayerChoiceSynchronizer.WaitForRemoteChoice " +
            "(GameActions.Multiplayer/PlayerChoiceSynchronizer.cs:78-111) is where the host starts to wait for a remote player's answer to a prompt - the moment that " +
            "player's own client shows the prompt (the local branch of the same call site, e.g. Commands/CardSelectCmd.cs:70-93 vs :94-98). The prefix hands (player, " +
            "choice id) to the registered remote-choice routers, which park a pending frame for that seat; the frame's answer is sent as the client's PlayerChoiceMessage " +
            "(VirtualClient.SendChoice). A wait that no router recognises aborts the episode (HandlerFault/remote_choice_unrouted). Arguments, result and game state untouched.");

        PatchRegistry.Postfix(HookEnqueueId,
            AccessTools.Method(typeof(GenericHookGameAction), nameof(GenericHookGameAction.SetChoiceContext)),
            PatchRegistry.M(typeof(RemoteSeatPatches), nameof(HookSetChoiceContextPostfix)), PatchCategory.Instrumentation,
            "MULTI-SEAT ONLY (phase H5), observation point of a client duty. HookPlayerChoiceContext.SignalPlayerChoiceBegun (GameActions.Multiplayer/HookPlayerChoiceContext.cs:165-169) " +
            "calls GenericHookGameAction.SetChoiceContext and then enqueues the hook action only when its owner is the local player; for a remote owner the host waits " +
            "until that player's client - which runs the same line for its own player - sends RequestEnqueueHookActionMessage (ActionQueueSynchronizer.cs:170-181, " +
            "handled at :291-300). Postfix, remote owners only: posts 'the owner's client sends RequestEnqueueHookActionMessage{HookId, ActionType}' to the pump " +
            "(VirtualClient.SendHookEnqueue). The game's test flag that would make the host enqueue it itself (TestFlags.ShouldSendResumeForRemotePlayers) stays off. " +
            "Arguments, result and game state untouched by the patch itself.");

        PatchRegistry.Prefix(ResumeGameActionId,
            AccessTools.Method(typeof(GameActionPlayerChoiceContext), nameof(GameActionPlayerChoiceContext.SignalPlayerChoiceEnded)),
            PatchRegistry.M(typeof(RemoteSeatPatches), nameof(GameActionChoiceEndedPrefix)), PatchCategory.Instrumentation,
            "MULTI-SEAT ONLY (phase H5), observation point of a client duty. GameActionPlayerChoiceContext.SignalPlayerChoiceEnded (GameActions.Multiplayer/GameActionPlayerChoiceContext.cs:63-70) " +
            "requests the resume of the paused action only when its owner is the local player; a remote owner's action stays paused until that player's client sends " +
            "RequestResumeActionAfterPlayerChoiceMessage (ActionQueueSynchronizer.cs:189-201, handled at :313-321). Prefix, remote owners only: posts 'the owner's client " +
            "sends the resume request for action id N' to the pump (VirtualClient.SendResume; N = the action's id at this point, what the client's own call would send). " +
            "Arguments, result and game state untouched by the patch itself.");

        PatchRegistry.Prefix(ResumeHookId,
            AccessTools.Method(typeof(HookPlayerChoiceContext), nameof(HookPlayerChoiceContext.SignalPlayerChoiceEnded)),
            PatchRegistry.M(typeof(RemoteSeatPatches), nameof(HookChoiceEndedPrefix)), PatchCategory.Instrumentation,
            "MULTI-SEAT ONLY (phase H5), observation point of a client duty. HookPlayerChoiceContext.SignalPlayerChoiceEnded (GameActions.Multiplayer/HookPlayerChoiceContext.cs:176-182): as " +
            "mp-remote-resume-game-action, for the hook action of a remote owner.");

        PatchRegistry.Note(RunLobbyResetId, "MegaCrit.Sts2.Core.Runs.RunManager::RunLobby (private setter)", "property write (after the CleanUp of a multi-seat run)", PatchCategory.Plumbing,
            "MULTI-SEAT ONLY (phase H5; lead ruling Q10). RunManager.RunLobby is set when a multiplayer run is set up (Runs/RunManager.cs:425) and never reset: CleanUp only disposes it " +
            "(:1480-1484). InitializeRunLobby hands the field - stale and disposed - to the CombatStateSynchronizer and the RestSiteSynchronizer of the NEXT run, a single-player run " +
            "included (:421-430; they subscribe to its RemotePlayerDisconnected event). After RunManager.CleanUp of a run on the host's own net service the driver writes null, the value " +
            "of a process that never played a multiplayer run, so that a single-seat episode after a multi-seat episode starts from the state of a fresh process. " +
            "Self-check: --selftest-mp mixed (1-seat / 4-seat / 1-seat episodes in one process: the third reproduces the first's trace).");

        InstallTaskCaptures();

        PatchRegistry.PrefixPostfix(SyncCheckId,
            AccessTools.Method(typeof(Player), nameof(Player.SyncWithSerializedPlayer)),
            PatchRegistry.M(typeof(RemoteSeatPatches), nameof(SyncPrefix)), PatchRegistry.M(typeof(RemoteSeatPatches), nameof(SyncPostfix)), PatchCategory.Instrumentation,
            "MULTI-SEAT ONLY (phase H5). Read-only: Player.SyncWithSerializedPlayer (Entities.Players/Player.cs:325-365) rebuilds a remote player from the data its client " +
            "sent at a map-point entry (CombatStateSynchronizer.WaitForSync, Multiplayer/CombatStateSynchronizer.cs:136-147). In lockstep that data describes the state the " +
            "host already has, so the rebuild must not change what a run signature sees of the player (gold, HP, energy, deck in order, relics, potions, player RNG streams). " +
            "Prefix / postfix take that signature before and after; a difference counts mp.violation.sync_changed_state and aborts the episode " +
            "(StrictViolation/sync_changed_player_state). Arguments, result and game state untouched.");
    }

    private static RunContext? Episode => HostSession.Current?.CurrentEpisode;

    /// <summary>The registered non-Harmony intervention of the multi-seat teardown (RunDriver.Teardown; lead ruling Q10).</summary>
    public const string RunLobbyResetId = "mp-run-lobby-reset";

    // -----------------------------------------------------------------------------------------------------
    //  mp-remote-reward-take / -purchase / -rest-choice / -chest-gold: the task a room-message handler starts
    // -----------------------------------------------------------------------------------------------------

    public const string RewardTakeId = "mp-remote-reward-take";
    public const string PurchaseId = "mp-remote-purchase";
    public const string RestChoiceId = "mp-remote-rest-choice";
    public const string ChestGoldId = "mp-remote-chest-gold";

    private static void InstallTaskCaptures()
    {
        const string Common = " MULTI-SEAT ONLY (phase H5), read-only. The host's handler of this client message starts the player's work with TaskHelper.RunSafely and drops the task " +
            "(the local path of the same flow returns it to the UI code that awaits it). Postfix: while a room message of a virtual client is being handled " +
            "(VirtualClient.SendTracked) the returned task is noted, so that the room owner tracks it for the seat - completion, result and fault are observed exactly as for seat 0. " +
            "Outside such a delivery (every local path, single-seat runs) the postfix does nothing. Arguments, result and game state untouched.";

        PatchRegistry.Postfix(RewardTakeId,
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer), "SelectRewardForPlayer", new[] { typeof(Player), typeof(int) }),
            PatchRegistry.M(typeof(RemoteSeatPatches), nameof(NoteTask)), PatchCategory.Instrumentation,
            "RewardsSetSynchronizer.SelectRewardForPlayer(Player, int) (Multiplayer.Game/RewardsSetSynchronizer.cs:356-371), called by HandleRewardSelectedMessage (:332): a remote player takes " +
            "reward i of its stack-top set." + Common);

        PatchRegistry.Postfix(PurchaseId,
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Multiplayer.Game.MerchantSynchronizer), "PurchaseEntry", new[] { typeof(Player), typeof(uint) }),
            PatchRegistry.M(typeof(RemoteSeatPatches), nameof(NoteBoolTask)), PatchCategory.Instrumentation,
            "MerchantSynchronizer.PurchaseEntry(Player, uint) (Multiplayer.Game/MerchantSynchronizer.cs:112-117), called by HandleMerchantOptionPurchasedMessage (:109): a remote player's purchase; " +
            "the bool result (false = refused or cancelled) is otherwise lost." + Common);

        PatchRegistry.Postfix(RestChoiceId,
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer), "ChooseOption", new[] { typeof(Player), typeof(int) }),
            PatchRegistry.M(typeof(RemoteSeatPatches), nameof(NoteBoolTask)), PatchCategory.Instrumentation,
            "RestSiteSynchronizer.ChooseOption(Player, int) (Multiplayer.Game/RestSiteSynchronizer.cs:170-213), called by HandleRestSiteOptionChosenMessage (:115): a remote player's rest-site " +
            "option; the bool result (false = the option's own prompt was cancelled) is otherwise lost." + Common);

        PatchRegistry.Postfix(ChestGoldId,
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Multiplayer.Game.OneOffSynchronizer), "DoTreasureRoomRewards", new[] { typeof(Player) }),
            PatchRegistry.M(typeof(RemoteSeatPatches), nameof(NoteIntTask)), PatchCategory.Instrumentation,
            "OneOffSynchronizer.DoTreasureRoomRewards(Player) (Multiplayer.Game/OneOffSynchronizer.cs:71-90), called by HandleTreasureChestOpenedMessage (:68): a remote player's chest gold; " +
            "the amount (shown on that player's treasure screen) is otherwise lost." + Common);
    }

    public static void NoteTask(Task __result)
    {
        RemoteDelivery.Note(__result);
    }

    public static void NoteBoolTask(Task<bool> __result)
    {
        RemoteDelivery.Note(__result);
    }

    public static void NoteIntTask(Task<int> __result)
    {
        RemoteDelivery.Note(__result);
    }

    // -----------------------------------------------------------------------------------------------------
    //  mp-remote-choice-wait
    // -----------------------------------------------------------------------------------------------------

    public static void WaitForRemoteChoicePrefix(Player player, uint choiceId)
    {
        RunContext? ctx = Episode;
        if (ctx?.Mp == null)
        {
            return; // single seat (the game throws: PlayerChoiceSynchronizer.cs:81-84), or a replay tool outside an episode
        }
        MultiSeatStats.RemoteWaits++;
        IReadOnlyList<IRemoteChoiceRouter> routers = ctx.Session.Registry.RemoteChoiceRouters;
        for (int i = 0; i < routers.Count; i++)
        {
            if (routers[i].TryRoute(ctx, player, choiceId))
            {
                return;
            }
        }
        MultiSeatStats.RemoteWaitsUnrouted++;
        EpisodeAbortException e = new EpisodeAbortException(AbortReason.HandlerFault, "remote_choice_unrouted",
            "The game waits for a choice of remote player " + player.NetId + " (choice id " + choiceId + ") and no module recognises the prompt.");
        // The caller is game code whose task captures the exception (and may swallow it): hand the failure to the core as well.
        ctx.Post(() => Task.FromException(e), "remote choice without a router");
        throw e;
    }

    // -----------------------------------------------------------------------------------------------------
    //  mp-remote-hook-enqueue
    // -----------------------------------------------------------------------------------------------------

    public static void HookSetChoiceContextPostfix(GenericHookGameAction __instance)
    {
        RunContext? ctx = Episode;
        MultiSeatRun? mp = ctx?.Mp;
        if (mp == null)
        {
            return;
        }
        VirtualClient? client = mp.ClientOf(__instance.OwnerId);
        if (client == null)
        {
            return; // the local player's hook action: the game enqueues it itself (HookPlayerChoiceContext.cs:166-169)
        }
        uint hookId = __instance.HookId;
        GameActionType type = __instance.ActionType;
        MultiSeatStats.HookEnqueuesPosted++;
        ctx!.Post(() =>
        {
            client.SendHookEnqueue(hookId, type);
            return Task.CompletedTask;
        }, "seat " + client.Seat + " requests its hook action " + hookId, blocking: false);
    }

    // -----------------------------------------------------------------------------------------------------
    //  mp-remote-resume-*
    // -----------------------------------------------------------------------------------------------------

    public static void GameActionChoiceEndedPrefix(GameActionPlayerChoiceContext __instance)
    {
        PostResume(__instance.Action);
    }

    public static void HookChoiceEndedPrefix(HookPlayerChoiceContext __instance)
    {
        PostResume(__instance.GameAction);
    }

    private static void PostResume(GameAction? action)
    {
        RunContext? ctx = Episode;
        MultiSeatRun? mp = ctx?.Mp;
        if (mp == null || action == null)
        {
            return;
        }
        VirtualClient? client = mp.ClientOf(action.OwnerId);
        if (client == null)
        {
            return; // the local player's action: the game requests the resume itself
        }
        uint? id = action.Id;
        if (!id.HasValue)
        {
            EpisodeAbortException e = new EpisodeAbortException(AbortReason.HandlerFault, "remote_resume_without_id", "A player choice of remote player " + action.OwnerId + " ended on an action that was never enqueued: " + action);
            ctx!.Post(() => Task.FromException(e), "remote resume without an action id");
            throw e;
        }
        uint actionId = id.Value;
        MultiSeatStats.ResumesPosted++;
        ctx!.Post(() =>
        {
            client.SendResume(actionId);
            return Task.CompletedTask;
        }, "seat " + client.Seat + " requests the resume of its action " + actionId, blocking: false);
    }

    // -----------------------------------------------------------------------------------------------------
    //  mp-sync-identity-check
    // -----------------------------------------------------------------------------------------------------

    public static void SyncPrefix(Player __instance, out ulong __state)
    {
        __state = Episode?.Mp == null ? 0UL : MultiSeatRun.SeatSignature(__instance);
    }

    public static void SyncPostfix(Player __instance, ulong __state)
    {
        RunContext? ctx = Episode;
        if (ctx?.Mp == null)
        {
            return;
        }
        MultiSeatStats.SyncPlayersRebuilt++;
        ulong after = MultiSeatRun.SeatSignature(__instance);
        if (after == __state)
        {
            return;
        }
        MultiSeatStats.SyncChangedState++;
        EpisodeAbortException e = new EpisodeAbortException(AbortReason.StrictViolation, "sync_changed_player_state",
            "Player.SyncWithSerializedPlayer changed player " + __instance.NetId + " (seat " + ctx.SeatOf(__instance) + "): signature " + __state.ToString("x16") + " -> " + after.ToString("x16") +
            ". The host and that player's client would have diverged in the real game.");
        ctx.Post(() => Task.FromException(e), "combat-state sync changed a player");
    }
}
