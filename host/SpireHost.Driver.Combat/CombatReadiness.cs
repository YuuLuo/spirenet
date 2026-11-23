using System.Collections.Generic;
using System.Text;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Runs;

namespace SpireHost.Driver.Combat;

/// <summary>
/// The "engine is waiting for combat input" predicate (driver-spec-combat.md 1.5, DECISIONS-H1 A.4, B.1).
/// There is no explicit signal in the game; the predicate is evaluated at a quiescent point with an empty
/// decision stack (a pending selection prompt always comes first: DRIVER-API.md 5.2 P1).
///
/// All of the spec's 11 conditions are here, split into the engine-wide ones and the per-seat ones, plus
/// four that make "what is offered will be accepted" checkable instead of assumed: the executor is not
/// paused, the seat's queue is neither paused nor cancelling, and the local "actions disabled" flag is clear.
/// Allocation-free: private state is read through the publicized reference (no LINQ, no list copies).
/// </summary>
public static class CombatReadiness
{
    /// <summary>Engine-wide conditions. On success <paramref name="state"/> is the live combat state.</summary>
    public static NotReady Engine(CombatManager cm, RunManager rm, out CombatState? state)
    {
        state = null;
        CombatTurnState? turn = cm._turnState;
        if (turn == null || !turn.IsInProgress)
        {
            return NotReady.NoCombat;
        }
        state = turn.State;
        if (cm.IsOverOrEnding)
        {
            return NotReady.CombatOverOrEnding;
        }
        if (state.CurrentSide != CombatSide.Player)
        {
            return NotReady.EnemySide;
        }
        if (turn.EndingPlayerTurnPhaseOne || turn.EndingPlayerTurnPhaseTwo)
        {
            return NotReady.EndingPlayerTurn;
        }
        if (rm.ActionQueueSynchronizer.CombatState != ActionSynchronizerCombatState.PlayPhase)
        {
            return NotReady.SynchronizerNotPlayPhase;
        }
        ActionExecutor executor = rm.ActionExecutor;
        if (executor.IsPaused)
        {
            return NotReady.ExecutorPaused;
        }
        if (executor.IsRunning || executor.CurrentlyRunningAction != null)
        {
            return NotReady.ExecutorRunning;
        }
        if (!rm.ActionQueueSet.IsEmpty)
        {
            return NotReady.QueueNotEmpty;
        }
        return NotReady.Ready;
    }

    /// <summary>Per-seat conditions for playing cards, using potions and ending the turn. Call only when <see cref="Engine"/> is Ready.</summary>
    public static NotReady Seat(CombatManager cm, RunManager rm, Player player, bool isLocalSeat)
    {
        PlayerCombatState? pcs = player.PlayerCombatState;
        if (pcs == null)
        {
            return NotReady.NoCombatState;
        }
        if (!player.Creature.IsAlive)
        {
            return NotReady.Dead;
        }
        CombatTurnState turn = cm._turnState!;
        List<Player> extra = turn.PlayersTakingExtraTurn;
        if (extra.Count > 0 && !extra.Contains(player))
        {
            return NotReady.NotInExtraTurn;
        }
        if (turn.PlayersReadyToEndTurn.Contains(player))
        {
            return NotReady.ReadyToEndTurn;
        }
        if (pcs.Phase != PlayerTurnPhase.Play)
        {
            return NotReady.PhaseNotPlay;
        }
        if (cm.IsExecutingCardOrPotionEffect(player))
        {
            return NotReady.ExecutingEffect;
        }
        if (isLocalSeat && cm.PlayerActionsDisabled)
        {
            return NotReady.ActionsDisabledLocally;
        }
        ActionQueueSet.ActionQueue? queue = FindQueue(rm.ActionQueueSet, player.NetId);
        if (queue == null)
        {
            return NotReady.QueueCancelling;
        }
        if (queue.isPaused)
        {
            return NotReady.QueuePaused;
        }
        if (queue.IsCancellingPlayCardActions || queue.isCancellingPlayerDrivenCombatActions || queue.isCancellingCombatActions)
        {
            return NotReady.QueueCancelling;
        }
        return NotReady.Ready;
    }

    /// <summary>
    /// UNDO_END_TURN (multi-seat model only; DECISIONS-H1 B.1): the seat has ended its turn, may take it back
    /// (CombatManager.CanUnEndTurn :749-773) and not everybody is ready yet (NEndTurnButton.cs:299-318).
    /// Call only when <see cref="Engine"/> is Ready.
    /// </summary>
    public static bool CanUndoEndTurn(CombatManager cm, Player player, int seatCount)
    {
        if (seatCount <= 1 || player.PlayerCombatState == null)
        {
            return false;
        }
        CombatTurnState turn = cm._turnState!;
        if (!turn.PlayersReadyToEndTurn.Contains(player) || turn.PlayersReadyToEndTurn.Count == turn.State.Players.Count)
        {
            return false;
        }
        return cm.CanUnEndTurn(player);
    }

    internal static ActionQueueSet.ActionQueue? FindQueue(ActionQueueSet set, ulong ownerId)
    {
        List<ActionQueueSet.ActionQueue> queues = set._actionQueues;
        for (int i = 0; i < queues.Count; i++)
        {
            if (queues[i].ownerId == ownerId)
            {
                return queues[i];
            }
        }
        return null;
    }

    /// <summary>Stall diagnostics (driver-spec-combat.md 1.5): what the core's one-line snapshot does not contain.</summary>
    public static string Describe(CombatManager cm, RunManager rm, IReadOnlyList<Player> seats)
    {
        StringBuilder sb = new StringBuilder(256);
        CombatTurnState? turn = cm._turnState;
        sb.Append("side=").Append(turn?.State.CurrentSide.ToString() ?? "-")
          .Append(" round=").Append(turn?.State.RoundNumber ?? -1)
          .Append(" ending=").Append(cm.IsEnding)
          .Append(" aboutToLose=").Append(cm.IsAboutToLose)
          .Append(" endingPhaseOne=").Append(cm.EndingPlayerTurnPhaseOne)
          .Append(" endingPhaseTwo=").Append(cm.EndingPlayerTurnPhaseTwo)
          .Append(" enemyTurnStarted=").Append(cm.IsEnemyTurnStarted)
          .Append(" sync=").Append(rm.ActionQueueSynchronizer.CombatState)
          .Append(" executor{paused=").Append(rm.ActionExecutor.IsPaused).Append(" running=").Append(rm.ActionExecutor.IsRunning)
          .Append(" current=").Append(rm.ActionExecutor.CurrentlyRunningAction?.ToString() ?? "-").Append('}')
          .Append(" queuesEmpty=").Append(rm.ActionQueueSet.IsEmpty);
        for (int s = 0; s < seats.Count; s++)
        {
            Player p = seats[s];
            PlayerCombatState? pcs = p.PlayerCombatState;
            sb.Append(" seat").Append(s).Append("{phase=").Append(pcs?.Phase.ToString() ?? "-")
              .Append(" alive=").Append(p.Creature.IsAlive)
              .Append(" readyToEnd=").Append(cm.IsPlayerReadyToEndTurn(p))
              .Append(" inEffect=").Append(cm.IsExecutingCardOrPotionEffect(p));
            ActionQueueSet.ActionQueue? q = FindQueue(rm.ActionQueueSet, p.NetId);
            if (q != null)
            {
                sb.Append(" queue{n=").Append(q.actions.Count).Append(" paused=").Append(q.isPaused)
                  .Append(" cancelPlay=").Append(q.IsCancellingPlayCardActions)
                  .Append(" cancelPlayerDriven=").Append(q.isCancellingPlayerDrivenCombatActions)
                  .Append(" cancelCombat=").Append(q.isCancellingCombatActions);
                if (q.actions.Count > 0)
                {
                    sb.Append(" head=").Append(q.actions[0].GetType().Name).Append(':').Append(q.actions[0].State);
                }
                sb.Append('}');
            }
            sb.Append('}');
        }
        return sb.ToString();
    }
}
