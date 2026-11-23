using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Actions;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace SpireHost.Driver.Combat;

/// <summary>Process-wide counters of the combat driver (reported by the combat self-tests).</summary>
public static class CombatStats
{
    public static long Decisions;
    public static long PlaysSubmitted;
    public static long PotionsUsed;
    public static long PotionsDiscarded;
    public static long EndTurns;
    public static long UndoEndTurns;

    /// <summary>Hard requirement: all four stay 0 (each one also aborts the episode).</summary>
    public static long Rejected;       // card.TryManualPlay returned false
    public static long NotEnqueued;    // the submitted action never reached a queue (deferred or dropped)
    public static long Cancelled;      // the submitted action was cancelled when enqueued
    public static long NotFinished;    // the submitted action had not finished at the next combat decision

    public static long UnrecognisedStates;

    public static object Snapshot() => new
    {
        decisions = Decisions,
        plays_submitted = PlaysSubmitted,
        potions_used = PotionsUsed,
        potions_discarded = PotionsDiscarded,
        end_turns = EndTurns,
        undo_end_turns = UndoEndTurns,
        rejected = Rejected,
        not_enqueued = NotEnqueued,
        cancelled = Cancelled,
        not_finished = NotFinished,
        unrecognised_states = UnrecognisedStates,
    };

    public static long Violations => Rejected + NotEnqueued + Cancelled + NotFinished;
}

/// <summary>
/// COMBAT decisions (driver-spec-combat.md 1.5, 2, 3, 8.2; DECISIONS-H1 B.1).
///
/// Step (quiescent, decision stack empty, CombatManager.IsInProgress):
///   1. the previously submitted action must have finished (audit, see <see cref="CombatLifecycle"/>);
///   2. engine-wide readiness (<see cref="CombatReadiness.Engine"/>), else Unrecognised with the reason;
///   3. per seat, in seat order: if the seat is ready -> its card plays, potion uses / discards and END_TURN;
///      else, in the multi-seat model, UNDO_END_TURN when the game allows it;
///   4. no action at all -> Unrecognised (stall), listing every seat's reason.
///
/// Legality = the engine's own execution-time checks (PlayCardAction.cs:58-85: card in hand, CanPlay,
/// IsValidTarget; UsePotionAction.cs:88-114: slot filled, IsValidTarget of the completed target) plus the
/// rules only the UI enforces (targets must be hittable; AnyAlly = other living players; every other
/// target type passes null; a seat that ended its turn does nothing; Automatic potions cannot be used).
/// Nothing is tried and taken back: in TestMode an action that cancels itself throws (GameAction.cs:96-106).
///
/// Apply uses the entry points of the real UI: card.TryManualPlay(target) (NCardPlay.cs:172/188),
/// potion.EnqueueManualUse(target) (NPotionHolder.cs:369/450), DiscardPotionGameAction (NPotionPopup.cs:275),
/// EndPlayerTurnAction / UndoEndPlayerTurnAction (NEndTurnButton.cs:426-442). The game executes the action
/// inline up to its first real suspension, so "checked while quiescent, submitted immediately" is the same
/// state the execution-time checks see.
///
/// Seats. Everything is per seat (RunState.Players index). With one seat ActingSeat is 0; with several the
/// decision is the union of all seats that can act (ActingSeat -1, SeatMask from the core) and the policy
/// chooses which seat moves next: the game executes queue heads in id order (ActionQueueSet.cs:151-223), and
/// the host submits one action at a time.
///
/// MULTI-SEAT (phase H5). Seat 0 is the host instance's local player: its actions take the entry points above. A
/// remote seat's action is what that player's client sends: the client's UI builds the same action for its own player
/// and its ActionQueueSynchronizer.RequestEnqueue sends it as RequestEnqueueActionMessage (:143-153); the host makes
/// the sender the owner (:269-278, :333-341). So Apply builds the action with the game's own constructor - after the
/// same client-side checks (CardModel.CanPlayTargeting in TryManualPlay :1562-1570; the target completion of
/// PotionModel.EnqueueManualUse :215-226) - and hands it to the seat's VirtualClient. The host-side enumeration is the
/// client's UI for every seat: a seat that ended its turn only gets UNDO_END_TURN (the game itself would still accept
/// a card of a ready player: only the local UI's PlayerActionsDisabled stops it, PlayerCmd.cs:211-221).
/// </summary>
public sealed class CombatHandler : IRoomHandler
{
    private readonly Decision _decision = new Decision();
    private readonly CombatContext _context = new CombatContext();
    private readonly CombatLifecycle _lifecycle;

    public CombatHandler(CombatLifecycle lifecycle)
    {
        _lifecycle = lifecycle;
    }

    public string Name => "combat:combat";

    public HandlerStep Step(RunContext ctx, int seat)
    {
        CombatManager cm = ctx.CombatManager;
        RunManager rm = ctx.RunManager;
        _lifecycle.Audit.CheckFinished(ctx);

        IReadOnlyList<Player> seats = ctx.Seats;
        CombatContext info = _context;
        Decision d = _decision.Reset(DecisionKind.Combat, this, actingSeat: 0, context: _context);
        NotReady engine = Enumerate(cm, rm, seats, ctx.LocalPlayer, info, d, out int actingSeat, ctx);
        if (engine != NotReady.Ready)
        {
            CombatStats.UnrecognisedStates++;
            return HandlerStep.Unrecognised("combat in progress but the engine is not waiting for input: " + engine + "; " + CombatReadiness.Describe(cm, rm, seats));
        }
        if (d.Actions.Count == 0)
        {
            CombatStats.UnrecognisedStates++;
            return HandlerStep.Unrecognised("combat in progress but no seat can act: " + SeatReasons(info, seats.Count) + "; " + CombatReadiness.Describe(cm, rm, seats));
        }
        d.ActingSeat = actingSeat;
        CombatStats.Decisions++;
        return HandlerStep.Ask(d);
    }

    /// <summary>
    /// The enumeration itself: the engine-wide predicate, then every seat's predicate and legal actions,
    /// appended to <paramref name="d"/> (unsorted; the core sorts), and the COMBAT context <paramref name="info"/>
    /// filled with what was evaluated on the way (contract C1). It needs no RunContext - only the game's
    /// managers and the seat list - so that the same code can be evaluated outside an episode: the .mcr
    /// cross-check (Tests\McrCrossCheck.cs) calls it at every recorded decision of the real client's replays.
    /// Returns the engine-wide reason, <see cref="NotReady.Ready"/> when seats were examined; in that case
    /// <paramref name="actingSeat"/> is the only seat with actions, or -1 when none or several have some.
    /// </summary>
    public static NotReady Enumerate(CombatManager cm, RunManager rm, IReadOnlyList<Player> seats, Player local, CombatContext info, Decision d, out int actingSeat, RunContext? ctx = null)
    {
        actingSeat = -1;
        NotReady engine = CombatReadiness.Engine(cm, rm, out CombatState? state);
        if (engine != NotReady.Ready)
        {
            return engine;
        }
        if (state == null)
        {
            return NotReady.NoCombat;
        }
        info.State = state;
        List<Creature> allies = state._allies;
        List<Creature> enemies = state._enemies;
        info.Creatures.Clear();
        info.Hittable.Clear();
        info.AllyCount = allies.Count;
        for (int i = 0; i < allies.Count; i++)
        {
            info.Creatures.Add(allies[i]);
            info.Hittable.Add(0);
        }
        for (int i = 0; i < enemies.Count; i++)
        {
            info.Creatures.Add(enemies[i]);
            info.Hittable.Add(0);
        }

        int seatCount = seats.Count;
        int acting = 0;
        // CanPlay of every hand card with the combat's hook listeners enumerated once per enumeration (HookCache.Playability.cs);
        // the engine is quiescent and the enumeration writes no game state, so the listener set is constant until End.
        s_hooks.Begin(state);
        try
        {
            EnumerateSeats(cm, rm, seats, local, info, d, ctx, seatCount, ref acting, ref actingSeat);
        }
        finally
        {
            s_hooks.End();
        }
        if (acting != 1)
        {
            actingSeat = -1;
        }
        return NotReady.Ready;
    }

    /// <summary>Engine thread only (one enumeration at a time).</summary>
    private static readonly HookCache s_hooks = new HookCache();

    private static void EnumerateSeats(CombatManager cm, RunManager rm, IReadOnlyList<Player> seats, Player local, CombatContext info, Decision d, RunContext? ctx,
        int seatCount, ref int acting, ref int actingSeat)
    {
        for (int s = 0; s < seatCount; s++)
        {
            Player p = seats[s];
            CombatSeatInfo seat = info.SeatSlot(s, p);
            NotReady why = CombatReadiness.Seat(cm, rm, p, ReferenceEquals(p, local));
            seat.State = why;
            seat.ReadyToEndTurn = why == NotReady.ReadyToEndTurn;
            int before = d.Actions.Count;
            if (why == NotReady.Ready)
            {
                EnumerateSeat(rm, info, seat, d, ctx);
            }
            else if (why == NotReady.ReadyToEndTurn && CombatReadiness.CanUndoEndTurn(cm, p, seatCount) && (ctx == null || ctx.Options.OfferUndoEndTurn))
            {
                seat.CanUndoEndTurn = true;
                d.Add(Verb.UndoEndTurn, s);
            }
            if (d.Actions.Count > before)
            {
                acting++;
                actingSeat = s;
            }
        }
    }

    private static string SeatReasons(CombatContext info, int seatCount)
    {
        string r = "";
        for (int s = 0; s < seatCount; s++)
        {
            r += (s > 0 ? ", " : "") + "seat " + s + ": " + info.Seats[s].State;
        }
        return r;
    }

    // -----------------------------------------------------------------------------------------------------
    //  Enumeration for one seat that can act
    // -----------------------------------------------------------------------------------------------------

    private static void EnumerateSeat(RunManager rm, CombatContext info, CombatSeatInfo seat, Decision d, RunContext? ctx)
    {
        int s = seat.Seat;
        Player player = seat.Player;
        Creature self = player.Creature;
        List<Creature> creatures = info.Creatures;
        int allyCount = info.AllyCount;

        // ---- cards: hand order is the public ordinal ----
        IReadOnlyList<CardModel> hand = player.PlayerCombatState!.Hand.Cards;
        for (int i = 0; i < hand.Count; i++)
        {
            CardModel card = hand[i];
            bool playable = s_hooks.CanPlay(card, out UnplayableReason reason, out AbstractModel? preventer);
            seat.HandPlayable.Add(playable);
            seat.HandReason.Add(reason);
            seat.HandPreventer.Add(preventer);
            if (!playable)
            {
                continue;
            }
            GameRef source = GameRef.OfCard(card, i);
            switch (card.TargetType)
            {
                case TargetType.AnyEnemy:
                    // UI: only hittable enemies can be aimed at (NTargetManager.cs:300-313, NControllerCardPlay.cs:171-186).
                    for (int c = allyCount; c < creatures.Count; c++)
                    {
                        Creature target = creatures[c];
                        if (card.IsValidTarget(target) && IsHittable(info, c))
                        {
                            d.Add(Verb.PlayCard, s, source, GameRef.OfCreature(target, c));
                        }
                    }
                    break;
                case TargetType.AnyAlly:
                    // UI: another living player, never yourself, never a pet (NTargetManager.cs:434-441).
                    for (int c = 0; c < allyCount; c++)
                    {
                        Creature target = creatures[c];
                        if (target.IsPlayer && !ReferenceEquals(target, self) && card.IsValidTarget(target) && IsHittable(info, c))
                        {
                            d.Add(Verb.PlayCard, s, source, GameRef.OfCreature(target, c));
                        }
                    }
                    break;
                default:
                    // Every other target type is played with a null target (NCardPlay.cs:139-206); a creature would be rejected (CardModel.cs:1559).
                    if (card.IsValidTarget(null))
                    {
                        d.Add(Verb.PlayCard, s, source);
                    }
                    break;
            }
        }

        // ---- potions: NPotionPopup rules (Nodes.Potions/NPotionPopup.cs:146-197, 374-423) ----
        IReadOnlyList<PotionModel?> slots = player.PotionSlots;
        bool beltOpen = player.CanUseOrRemovePotions && !rm.IsPotionUseLocked; // owner alive: guaranteed by the seat predicate
        for (int i = 0; i < slots.Count; i++)
        {
            PotionModel? potion = slots[i];
            bool usable = false;
            bool discardable = false;
            // "Queued" = the owner's own popup state: the potion's flag for the local seat, the client's flag for a remote seat.
            if (potion != null && beltOpen && !(ctx != null ? ctx.IsPotionQueued(potion) : potion.IsQueued))
            {
                discardable = true;
                GameRef slot = GameRef.OfPotionSlot(player, i);
                d.Add(Verb.DiscardPotion, s, slot);
                PotionUsage usage = potion.Usage;
                // canUseInCombatNow (:398) is implied by the seat predicate; Automatic potions are never usable (:381).
                if ((usage == PotionUsage.CombatOnly || usage == PotionUsage.AnyTime) && potion.PassesCustomUsabilityCheck)
                {
                    int before = d.Actions.Count;
                    AddPotionUses(info, d, s, player, potion, slot);
                    usable = d.Actions.Count > before;
                }
            }
            seat.PotionUsable.Add(usable);
            seat.PotionDiscardable.Add(discardable);
        }

        d.Add(Verb.EndTurn, s);
    }

    /// <summary>
    /// Targets as the real UI offers them (Nodes.Potions/NPotionHolder.cs:351-372, 424-450; NTargetManager.cs:417-441),
    /// each checked with the execution-time predicate PotionModel.IsValidTarget (Models/PotionModel.cs:229-264).
    /// </summary>
    private static void AddPotionUses(CombatContext info, Decision d, int s, Player player, PotionModel potion, GameRef slot)
    {
        Creature self = player.Creature;
        List<Creature> creatures = info.Creatures;
        int allyCount = info.AllyCount;
        TargetType type = potion.TargetType;
        switch (type)
        {
            case TargetType.AnyEnemy:
                for (int c = allyCount; c < creatures.Count; c++)
                {
                    Creature target = creatures[c];
                    if (potion.IsValidTarget(target) && IsHittable(info, c))
                    {
                        d.Add(Verb.UsePotion, s, slot, GameRef.OfCreature(target, c));
                    }
                }
                break;
            case TargetType.AnyPlayer:
                if (potion.CanThrowAtAlly())
                {
                    // Several players, in combat: any living player, yourself included (NTargetManager.cs:428-433).
                    for (int c = 0; c < allyCount; c++)
                    {
                        Creature target = creatures[c];
                        if (target.IsPlayer && potion.IsValidTarget(target))
                        {
                            d.Add(Verb.UsePotion, s, slot, GameRef.OfCreature(target, c));
                        }
                    }
                }
                else if (potion.IsValidTarget(self))
                {
                    // One player: the UI passes null and EnqueueManualUse completes it to the owner (PotionModel.cs:220-223).
                    d.Add(Verb.UsePotion, s, slot, GameRef.OfCreature(self, IndexOf(creatures, self)));
                }
                break;
            case TargetType.Self:
                if (potion.IsValidTarget(self))
                {
                    d.Add(Verb.UsePotion, s, slot, GameRef.OfCreature(self, IndexOf(creatures, self)));
                }
                break;
            case TargetType.AnyAlly:
            case TargetType.TargetedNoCreature:
                // AnyAlly: no such potion in v0.113; the UI path would enqueue a use that cancels itself
                // (UsePotionAction.cs:100 completes the target to the owner, which AnyAlly rejects).
                // TargetedNoCreature: FoulPotion outside combat only (the merchant is the target).
                break;
            default:
                // AllEnemies and the other untargeted types: null (FoulPotion in combat, ExplosiveAmpoule ...).
                if (potion.IsValidTarget(null))
                {
                    d.Add(Verb.UsePotion, s, slot);
                }
                break;
        }
    }

    private static bool IsHittable(CombatContext info, int index)
    {
        byte v = info.Hittable[index];
        if (v == 0)
        {
            v = s_hooks.IsHittable(info.Creatures[index]) ? (byte)1 : (byte)2; // one hook dispatch per creature per decision (cached listeners)
            info.Hittable[index] = v;
        }
        return v == 1;
    }

    private static int IndexOf(List<Creature> creatures, Creature creature)
    {
        for (int i = 0; i < creatures.Count; i++)
        {
            if (ReferenceEquals(creatures[i], creature))
            {
                return i;
            }
        }
        return -1;
    }

    // -----------------------------------------------------------------------------------------------------
    //  Apply
    // -----------------------------------------------------------------------------------------------------

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        Player player = ctx.Seats[a.Seat];
        RunManager rm = ctx.RunManager;
        SubmissionAudit audit = _lifecycle.Audit;
        // MULTI-SEAT: a remote seat's action is sent by its client; null = the local seat (and every single-seat run).
        VirtualClient? client = ctx.Mp != null && ctx.IsRemoteSeat(a.Seat) ? ctx.Mp.Client(a.Seat) : null;
        switch (a.Verb)
        {
            case Verb.PlayCard:
            {
                CardModel card = a.Source.Card;
                Creature? target = a.Target.IsNone ? null : a.Target.Creature;
                audit.Expect(typeof(PlayCardAction), player, a);
                CombatStats.PlaysSubmitted++;
                // The client's TryManualPlay: CanPlayTargeting, then its request (OnEnqueuePlayVfx is the client's own presentation).
                bool accepted = client != null ? card.CanPlayTargeting(target) : card.TryManualPlay(target);
                if (!accepted)
                {
                    CombatStats.Rejected++;
                    ctx.Fail(AbortReason.ActionFault, "play_rejected", $"card.TryManualPlay refused {card.Id.Entry} (hand {a.Source.Index}) -> {(target == null ? "no target" : "creature " + a.Target.Index)}: the enumeration offered an illegal play.");
                }
                client?.SendAction(new PlayCardAction(card, target));
                break;
            }
            case Verb.UsePotion:
            {
                int slot = a.Source.Index;
                PotionModel? potion = player.GetPotionAtSlotIndex(slot);
                if (potion == null)
                {
                    CombatStats.Rejected++;
                    ctx.Fail(AbortReason.ActionFault, "potion_slot_empty", "USE_POTION on empty slot " + slot);
                }
                // The argument the UI passes: the aimed creature for targeted potions, the owner for Self, null otherwise.
                TargetType type = potion.TargetType;
                Creature? argument = type == TargetType.AnyEnemy || (type == TargetType.AnyPlayer && potion.CanThrowAtAlly())
                    ? a.Target.Creature
                    : type == TargetType.Self ? player.Creature : null;
                audit.Expect(typeof(UsePotionAction), player, a);
                CombatStats.PotionsUsed++;
                if (client != null)
                {
                    client.SendPotionUse(potion, argument, ctx.CombatManager.IsInProgress);
                }
                else
                {
                    potion.EnqueueManualUse(argument);
                }
                break;
            }
            case Verb.DiscardPotion:
                audit.Expect(typeof(DiscardPotionGameAction), player, a);
                CombatStats.PotionsDiscarded++;
                Submit(rm, client, new DiscardPotionGameAction(player, (uint)a.Source.Index, ctx.CombatManager.IsInProgress));
                break;
            case Verb.EndTurn:
                audit.Expect(typeof(EndPlayerTurnAction), player, a);
                CombatStats.EndTurns++;
                Submit(rm, client, new EndPlayerTurnAction(player, player.PlayerCombatState!.TurnNumber));
                break;
            case Verb.UndoEndTurn:
                audit.Expect(typeof(UndoEndPlayerTurnAction), player, a);
                CombatStats.UndoEndTurns++;
                Submit(rm, client, new UndoEndPlayerTurnAction(player, player.PlayerCombatState!.TurnNumber));
                break;
            default:
                ctx.Fail(AbortReason.HostBug, "combat_bad_verb", "Unexpected verb " + a.Verb + " on a COMBAT decision.");
                break;
        }
        audit.AfterSubmit(ctx);
    }

    /// <summary>The local seat requests the action on the host (the UI's call); a remote seat's client sends its request.</summary>
    private static void Submit(RunManager rm, VirtualClient? client, GameAction action)
    {
        if (client != null)
        {
            client.SendAction(action);
        }
        else
        {
            rm.ActionQueueSynchronizer.RequestEnqueue(action);
        }
    }
}

/// <summary>
/// "Zero rejected or cancelled actions" as a checked invariant instead of a hope. The lifecycle subscribes to
/// ActionQueueSet.ActionEnqueued; for every action the combat handler submits:
///   - it must reach a queue during Apply (RequestEnqueue defers CombatPlayPhaseOnly actions outside the play
///     phase, ActionQueueSynchronizer.cs:137-142: that would mean the ready predicate was wrong);
///   - it must not have been cancelled on arrival (ActionQueueSet.cs:105-131 cancels silently, without the
///     TestMode exception, when the queue is in a cancelling state);
///   - at the next COMBAT decision it must be Finished.
/// (An action that cancels itself while executing throws in TestMode and surfaces as a game-task fault.)
/// </summary>
public sealed class SubmissionAudit
{
    private System.Type? _expectType;
    private ulong _expectOwner;
    private bool _expecting;
    private GameAction? _seen;
    private GameAction? _last;
    private string _what = "";

    internal void Expect(System.Type actionType, Player owner, in LegalAction action)
    {
        _expectType = actionType;
        _expectOwner = owner.NetId;
        _expecting = true;
        _seen = null;
        _what = action.Verb.ToString();
    }

    /// <summary>ActionQueueSet.ActionEnqueued (raised before the queue decides whether to keep the action).</summary>
    internal void OnEnqueued(GameAction action)
    {
        if (_expecting && _seen == null && action.GetType() == _expectType && action.OwnerId == _expectOwner)
        {
            _seen = action;
        }
    }

    internal void AfterSubmit(RunContext ctx)
    {
        _expecting = false;
        GameAction? seen = _seen;
        _seen = null;
        if (seen == null)
        {
            CombatStats.NotEnqueued++;
            ctx.Fail(AbortReason.ActionFault, "action_not_enqueued", _what + ": the submitted " + _expectType?.Name + " did not reach an action queue (deferred or dropped): the ready-for-input predicate was wrong.");
        }
        if (seen.State == GameActionState.Canceled)
        {
            CombatStats.Cancelled++;
            ctx.Fail(AbortReason.ActionFault, "action_cancelled", _what + ": the submitted " + seen + " was cancelled when it was enqueued.");
        }
        _last = seen;
    }

    /// <summary>At the next COMBAT decision (quiescent, no pending prompt) the last submitted action must have run to completion.</summary>
    internal void CheckFinished(RunContext ctx)
    {
        GameAction? last = _last;
        if (last == null)
        {
            return;
        }
        _last = null;
        if (last.State != GameActionState.Finished)
        {
            if (last.State == GameActionState.Canceled)
            {
                CombatStats.Cancelled++;
            }
            else
            {
                CombatStats.NotFinished++;
            }
            ctx.Fail(AbortReason.ActionFault, "action_not_finished", _what + ": " + last + " is in state " + last.State + " at the next combat decision.");
        }
    }

    internal void Reset()
    {
        _expecting = false;
        _expectType = null;
        _seen = null;
        _last = null;
    }
}
