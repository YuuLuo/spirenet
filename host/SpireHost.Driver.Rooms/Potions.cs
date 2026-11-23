using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

// =========================================================================================================
//  Out-of-combat potions on room-level frames (DECISIONS-H1 B.7; driver-spec-rooms.md 8.1; DRIVER-API.md 6.6)
//
//  One IRoomFrameExtension for every non-nested MAP / SHOP / REST / EVENT / REWARD / TREASURE decision (the core
//  never calls it for nested selection frames). The rules are the real UI's:
//    popup can open      NPotionHolder.OpenPotionPopup (:237-245): holder usable (local belt), potion present,
//                        !RunState.IsGameOver, holder not disabled until the potion is removed (a use / discard
//                        already requested)
//    both buttons off    NPotionPopup._Ready (:146-150, :192-197) / RefreshButtons (:400-424): potion queued, owner
//                        dead, !Owner.CanUseOrRemovePotions, RunManager.IsPotionUseLocked
//    use button          CanEnableUseButton (:374-388): PassesCustomUsabilityCheck and Usage == AnyTime out of combat
//                        (CombatOnly needs a combat, Automatic never)
//    use = target        NPotionHolder.UsePotion (:351-372): AnyEnemy / TargetedNoCreature / CanThrowAtAlly ->
//                        targeting (out of combat only the merchant button exists, for TargetedNoCreature); else
//                        Self -> own creature, other -> null; PotionModel.EnqueueManualUse (:216-227)
//    discard             NPotionPopup.OnDiscardButtonPressed (:263-277): DiscardPotionGameAction(owner, slot, inCombat)
//  Plus the execution-time check of UsePotionAction.ExecuteAction (:89-114): out of combat the target creature is
//  the target player's (or the owner's unless TargetedNoCreature) and must pass PotionModel.IsValidTarget, else the
//  action cancels itself - which throws in TestMode (GameAction.cs:96-106). That predicate is evaluated here.
// =========================================================================================================

internal sealed class OutOfCombatPotions : IRoomFrameExtension
{
    private sealed class PotionUse
    {
        public PotionModel Potion = null!;
        public Creature? Target;
    }

    /// <summary>Per-episode state: potions whose discard was requested but has not executed yet (the holder stays disabled).</summary>
    internal sealed class PotionState : IRunLifecycle
    {
        public readonly HashSet<PotionModel> PendingDiscard = new HashSet<PotionModel>();

        public string Name => RoomsNames.PotionsLifecycle;

        public void OnRunSetUp(RunContext ctx) => PendingDiscard.Clear();

        public void OnRoomEntered(RunContext ctx)
        {
        }

        public void OnEpisodeEnd(RunContext ctx) => PendingDiscard.Clear();
    }

    public readonly PotionState State = new PotionState();
    private readonly List<PotionModel> _drop = new List<PotionModel>(4);

    public string Name => RoomsNames.PotionsExtension;

    public void Extend(RunContext ctx, Decision decision)
    {
        if (decision.Nested)
        {
            return; // DECISIONS-H1 B.7: never on nested selection frames (the core does not call us for them anyway)
        }
        if (CombatManager.Instance.IsInProgress)
        {
            return; // in-combat potions are the combat team's (CombatOnly usability, combat targets)
        }
        RunState run = ctx.Run;
        if (run.IsGameOver || ctx.RunManager.IsPotionUseLocked)
        {
            return;
        }
        ForgetExecutedDiscards();
        if (decision.ActingSeat >= 0)
        {
            ExtendSeat(ctx, decision, decision.ActingSeat);
        }
        else
        {
            for (int seat = 0; seat < ctx.SeatCount; seat++)
            {
                ExtendSeat(ctx, decision, seat);
            }
        }
    }

    private void ExtendSeat(RunContext ctx, Decision decision, int seat)
    {
        Player player = ctx.Seats[seat];
        if (player.Creature.IsDead || !player.CanUseOrRemovePotions)
        {
            return;
        }
        IReadOnlyList<PotionModel?> slots = player.PotionSlots;
        for (int i = 0; i < slots.Count; i++)
        {
            PotionModel? p = slots[i];
            if (p == null || ctx.IsPotionQueued(p) || p.HasBeenRemovedFromState || State.PendingDiscard.Contains(p))
            {
                continue;
            }
            decision.Add(Verb.DiscardPotion, seat, GameRef.OfPotionSlot(player, i), payload: p);
            if (TryUiUse(p, player, out Creature? target))
            {
                decision.Add(Verb.UsePotion, seat, GameRef.OfPotionSlot(player, i), payload: new PotionUse { Potion = p, Target = target });
                RoomsProbe.Bump(RoomsProbe.PotionUseOfferedByKind, decision.Kind.ToString());
            }
        }
    }

    /// <summary>The UI's use button + target for an out-of-combat potion; false when it cannot be used now.</summary>
    internal static bool TryUiUse(PotionModel p, Player owner, out Creature? target)
    {
        target = null;
        if (p.Usage != PotionUsage.AnyTime || !p.PassesCustomUsabilityCheck)
        {
            return false;
        }
        TargetType tt = p.TargetType;
        Creature self = owner.Creature;
        if (tt == TargetType.AnyEnemy || p.CanThrowAtAlly())
        {
            // Targeting mode needs an enemy / ally node; out of combat there is none (CanThrowAtAlly requires a combat).
            return false;
        }
        Creature? uiTarget = tt == TargetType.Self ? self : null; // TargetedNoCreature: the merchant button -> null
        // EnqueueManualUse: a null target becomes the owner when the owner is a valid target (PotionModel.cs:219-222).
        Creature? enqueued = uiTarget ?? (p.IsValidTarget(self) ? self : null);
        // UsePotionAction.ExecuteAction out of combat (:94-104): the target player's creature, else the owner unless TargetedNoCreature.
        Creature? atExecution = enqueued != null ? enqueued.Player!.Creature : (tt != TargetType.TargetedNoCreature ? self : null);
        if (!p.IsValidTarget(atExecution))
        {
            return false;
        }
        target = uiTarget;
        return true;
    }

    private void ForgetExecutedDiscards()
    {
        if (State.PendingDiscard.Count == 0)
        {
            return;
        }
        _drop.Clear();
        foreach (PotionModel p in State.PendingDiscard)
        {
            if (p.HasBeenRemovedFromState || p.Owner.GetPotionSlotIndex(p) < 0)
            {
                _drop.Add(p);
            }
        }
        for (int i = 0; i < _drop.Count; i++)
        {
            State.PendingDiscard.Remove(_drop[i]);
        }
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        Player owner = a.Source.PotionOwner;
        int slot = a.Source.Index;
        PotionModel? inSlot = slot < owner.PotionSlots.Count ? owner.PotionSlots[slot] : null;
        if (a.Verb == Verb.DiscardPotion)
        {
            PotionModel p = (PotionModel)a.Payload!;
            if (!ReferenceEquals(inSlot, p))
            {
                ctx.Fail(AbortReason.HandlerFault, "potion_moved", "Potion slot " + slot + " no longer holds " + p.Id.Entry + ".");
            }
            State.PendingDiscard.Add(p);
            RoomsProbe.PotionDiscards++;
            DiscardPotionGameAction discard = new DiscardPotionGameAction(owner, (uint)slot, CombatManager.Instance.IsInProgress);
            VirtualClient? discarder = ctx.Mp?.ClientOf(owner.NetId);
            if (discarder != null)
            {
                discarder.SendAction(discard); // a remote seat: its client requests the action
            }
            else
            {
                ctx.RunManager.ActionQueueSynchronizer.RequestEnqueue(discard);
            }
            return;
        }
        if (a.Verb == Verb.UsePotion)
        {
            PotionUse use = (PotionUse)a.Payload!;
            if (!ReferenceEquals(inSlot, use.Potion))
            {
                ctx.Fail(AbortReason.HandlerFault, "potion_moved", "Potion slot " + slot + " no longer holds " + use.Potion.Id.Entry + ".");
            }
            RoomsProbe.PotionUses++;
            RoomsProbe.Bump(RoomsProbe.PotionsUsed, use.Potion.Id.Entry);
            VirtualClient? user = ctx.Mp?.ClientOf(owner.NetId);
            if (user != null)
            {
                user.SendPotionUse(use.Potion, use.Target, CombatManager.Instance.IsInProgress); // PotionModel.EnqueueManualUse on the client
            }
            else
            {
                use.Potion.EnqueueManualUse(use.Target);
            }
            return;
        }
        ctx.Fail(AbortReason.HandlerFault, "bad_verb", "Potion extension got " + a);
    }
}
