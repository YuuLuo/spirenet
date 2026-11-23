using System;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace SpireHost.Driver;

/// <summary>
/// One remote seat of a multi-seat run (seats 1..3): everything the real client of that player sends to the host,
/// and nothing else. There is no second simulation: a client that plays in lockstep holds the same state as the host,
/// so the message a client would build from its own copy is built here from the host's objects
/// (notes\h5-plan\rooms-multiseat-duties.md summary 1).
///
/// Every method is "this client sends ...": the message goes through <see cref="HeadlessHostNetService.Receive{T}"/>
/// (wire round trip, then the host's registered handlers, sender = this seat's net id). Location-targeted messages
/// carry RunLocationTargetedMessageBuffer.CurrentLocation, which is what a client at the same place puts in
/// (ActionQueueSynchronizer.cs:150, :177, :198); RunState.RunLocation would be wrong during a room entry (it already
/// names the room that the buffer has not visited yet, RunManager.cs:1093 vs :1102).
///
/// What a client sends, and where the real client does it:
///   SendAction       RequestEnqueueActionMessage      every player action: ActionQueueSynchronizer.RequestEnqueue on a
///                                                     Client (:143-153); the host makes the SENDER the owner (:269-278, :333-341)
///   SendHookEnqueue  RequestEnqueueHookActionMessage  HookPlayerChoiceContext.SignalPlayerChoiceBegun for its own hook (:166-169)
///   SendChoice       PlayerChoiceMessage              PlayerChoiceSynchronizer.SyncLocalChoice in the local branch of a prompt (:63-76)
///   SendResume       RequestResumeActionAfterPlayerChoiceMessage   SignalPlayerChoiceEnded for its own action
///                                                     (GameActionPlayerChoiceContext.cs:63-70, HookPlayerChoiceContext.cs:176-182)
///   SendPlayerData   SyncPlayerDataMessage            CombatStateSynchronizer.StartSync (:109-112)
///   Send             any other client message (room flows: rewards, rest, shop, events, treasure, crystal sphere)
/// Never a host -> client message (ActionEnqueuedMessage ...: the host's handlers throw, ActionQueueSynchronizer.cs:282, :304, :325).
/// </summary>
public sealed class VirtualClient
{
    private readonly HeadlessHostNetService _net;
    private readonly RunManager _rm;

    internal VirtualClient(HeadlessHostNetService net, RunManager rm, Player player, int seat)
    {
        _net = net;
        _rm = rm;
        Player = player;
        Seat = seat;
        NetId = player.NetId;
    }

    public Player Player { get; }

    public int Seat { get; }

    public ulong NetId { get; }

    /// <summary>
    /// The client's own "this potion's use is requested" flags. PotionModel.EnqueueManualUse sets PotionModel.IsQueued on
    /// the object of the peer that requests the use (Models/PotionModel.cs:225; read only by the potion popup,
    /// Nodes.Potions/NPotionPopup.cs:146): on the real host a remote player's potion is never flagged, so the flag of the
    /// seat's client lives here. An entry is forgotten once the potion left its slot; a use that cancels itself aborts
    /// the episode (GameAction.cs:96-106 throws in TestMode), so there is no "cancelled, usable again" state to track.
    /// </summary>
    private readonly System.Collections.Generic.List<MegaCrit.Sts2.Core.Models.PotionModel> _queuedPotions = new System.Collections.Generic.List<MegaCrit.Sts2.Core.Models.PotionModel>(2);

    /// <summary>The client requested the use of this potion and it has not left the belt yet (the client's PotionModel.IsQueued).</summary>
    public bool IsPotionQueued(MegaCrit.Sts2.Core.Models.PotionModel potion)
    {
        for (int i = _queuedPotions.Count - 1; i >= 0; i--)
        {
            MegaCrit.Sts2.Core.Models.PotionModel p = _queuedPotions[i];
            if (p.HasBeenRemovedFromState || Player.GetPotionSlotIndex(p) < 0)
            {
                _queuedPotions.RemoveAt(i);
            }
            else if (ReferenceEquals(p, potion))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>PotionModel.EnqueueManualUse on the client (:215-226): flags the potion, completes the target, requests the use.</summary>
    public void SendPotionUse(MegaCrit.Sts2.Core.Models.PotionModel potion, MegaCrit.Sts2.Core.Entities.Creatures.Creature? target, bool isCombatInProgress)
    {
        if (target == null && potion.IsValidTarget(Player.Creature))
        {
            target = Player.Creature;
        }
        UsePotionAction action = new UsePotionAction(potion, target, isCombatInProgress);
        _queuedPotions.Add(potion);
        SendAction(action);
    }

    public long ActionsSent;
    public long HookEnqueuesSent;
    public long ChoicesSent;
    public long ResumesSent;
    public long PlayerDataSent;
    public long OtherSent;

    /// <summary>The location a client at the host's place writes into a location-targeted message.</summary>
    public RunLocation Location => _rm.RunLocationTargetedBuffer.CurrentLocation;

    /// <summary>
    /// The client's UI built this action for its own player and requested it. A client keeps a play-phase-only action
    /// that is requested outside the play phase to itself until its play phase (ActionQueueSynchronizer.cs:137-141);
    /// the host never offers such an action then, so reaching that branch here is a host bug.
    /// </summary>
    public void SendAction(GameAction action)
    {
        if (action.OwnerId != NetId)
        {
            throw new InvalidOperationException("Client of seat " + Seat + " cannot send " + action + ": it is owned by player " + action.OwnerId + ".");
        }
        if (action.ActionType == GameActionType.CombatPlayPhaseOnly && _rm.ActionQueueSynchronizer.CombatState == ActionSynchronizerCombatState.NotPlayPhase)
        {
            throw new InvalidOperationException("Client of seat " + Seat + " would defer " + action + " (not the play phase): the host offered an action outside the play phase.");
        }
        ActionsSent++;
        _net.Receive(new RequestEnqueueActionMessage { action = action.ToNetAction(), location = Location }, NetId);
    }

    public void SendHookEnqueue(uint hookActionId, GameActionType gameActionType)
    {
        HookEnqueuesSent++;
        _net.Receive(new RequestEnqueueHookActionMessage { hookActionId = hookActionId, gameActionType = gameActionType, location = Location }, NetId);
    }

    public void SendChoice(uint choiceId, PlayerChoiceResult result)
    {
        ChoicesSent++;
        _net.Receive(new PlayerChoiceMessage { choiceId = choiceId, result = result.ToNetData() }, NetId);
    }

    public void SendResume(uint actionId)
    {
        ResumesSent++;
        _net.Receive(new RequestResumeActionAfterPlayerChoiceMessage { actionId = actionId, location = Location }, NetId);
    }

    public void SendPlayerData(SerializablePlayer data)
    {
        PlayerDataSent++;
        _net.Receive(new SyncPlayerDataMessage { player = data }, NetId);
    }

    /// <summary>Any other message of this client (the room flows build it, with <see cref="Location"/> where it is location-targeted).</summary>
    public void Send<T>(T message) where T : INetMessage
    {
        OtherSent++;
        _net.Receive(message, NetId);
    }

    /// <summary>
    /// As <see cref="Send{T}"/>, for a room message whose host handler starts the player's work as a fire-and-forget
    /// task (TaskHelper.RunSafely: taking a reward, a purchase, a rest option, the chest gold). Returns that task - the
    /// one the handler started for THIS message - so that the room owner can track it for the seat: its completion,
    /// its result and its fault are then observed exactly like the task the local path returns to its caller.
    /// Fail closed: a handler that started no task, or more than one, aborts the episode (the capture point moved).
    /// </summary>
    public System.Threading.Tasks.Task SendTracked<T>(T message) where T : INetMessage
    {
        OtherSent++;
        System.Threading.Tasks.Task? task = RemoteDelivery.Capture(() => _net.Receive(message, NetId), out int captured);
        if (task == null || captured != 1)
        {
            throw new EpisodeAbortException(AbortReason.HandlerFault, "remote_task_not_captured:" + typeof(T).Name,
                "The host handled " + typeof(T).Name + " of seat " + Seat + " and started " + captured + " observed task(s); exactly one was expected (mp-remote-* capture points).");
        }
        return task;
    }
}

/// <summary>
/// The capture scope of <see cref="VirtualClient.SendTracked{T}"/>: while a room message of a virtual client is being
/// handled, the read-only postfixes of RemoteSeatPatches (mp-remote-reward-take, -purchase, -rest-choice, -chest-gold)
/// note the task their method returned. Outside a scope they do nothing (the local paths call the same methods).
/// </summary>
internal static class RemoteDelivery
{
    private static bool _active;
    private static System.Threading.Tasks.Task? _task;
    private static int _count;

    /// <summary>Tasks noted in capture scopes since process start (module counter mp.remote.tasks_captured).</summary>
    public static long Captured;

    public static System.Threading.Tasks.Task? Capture(Action deliver, out int captured)
    {
        if (_active)
        {
            throw new InvalidOperationException("A tracked delivery is already in progress (re-entrant SendTracked).");
        }
        _active = true;
        _task = null;
        _count = 0;
        try
        {
            deliver();
            captured = _count;
            return _task;
        }
        finally
        {
            _active = false;
            _task = null;
        }
    }

    public static void Note(System.Threading.Tasks.Task task)
    {
        if (!_active)
        {
            return;
        }
        _count++;
        _task = task;
        Captured++;
    }
}
