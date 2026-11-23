using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.GameActions;

namespace SpireHost.Driver.Combat.Tests;

/// <summary>
/// "turnend": a player choice raised inside an END-OF-TURN hook context, followed by the combat being won at
/// the win check right after that hook (CombatManager.EndPlayerTurnPhaseOneInternal :1273-1283).
///
/// This is the path of the abort the rooms team's forced-HP fuzz found
/// (GameTaskFault @ ActionQueueSet.PopAction, repro (F0LFP2L3EKLM, FOOL, A10, policy seed 7954, hp 99999)):
/// SimpletonPower auto-plays hand cards when the auto-post-play phase is entered
/// (Models.Powers/SimpletonPower.cs:17-39, cheapest first); one of them asks for a card (-> a
/// GenericHookGameAction that pauses and resumes); a later one kills the last enemy; the hook finishes and the
/// turn-end sequence wins the combat while that hook action is still being finished. Root cause and patch:
/// SpireHost.Patches.SchedulingPatches, run-save-completes-after-pump-turn (the Core's uniform run-save boundary).
///
/// Cases (each: one enemy, SimpletonPower 8, END_TURN at the first decision):
///   A  the game answers the choice itself  LockedIn+ (cost 0, played first): CardSelectCmd.FromCombatPile over
///                                           the draw pile filtered by the Shoot tag: no candidate, but the choice
///                                           is still signalled (CardSelectCmd.cs:245-249) - exactly the recorded abort;
///   B  a selection frame answers it        Prepared (cost 0): FromHandForDiscard with several candidates -> a
///                                           CARD_SELECT frame of this module raised by a hook at turn END.
/// Pass: no abort; the combat was won at the end of the first turn (one COMBAT decision); a hook action was
/// enqueued during that turn end; the run-save boundary of the combat end ran exactly once.
/// Negative control (suite "turnend-control"): the same cases with the boundary passed through
/// (SchedulingPatches.RunSaveBoundaryDisabledForTest, a test-only switch set and reset here; the patch set is
/// unchanged): every case must abort in ActionQueueSet.PopAction.
/// </summary>
public static class TurnEndSuite
{
    public static List<CaseResult> Run(HostSession session, string? only) => Run(session, only, negativeControl: false);

    /// <summary>The negative control: without the run-save boundary every case must abort in ActionQueueSet.PopAction.</summary>
    public static List<CaseResult> RunNegativeControl(HostSession session, string? only)
    {
        bool saved = SpireHost.Patches.SchedulingPatches.RunSaveBoundaryDisabledForTest;
        SpireHost.Patches.SchedulingPatches.RunSaveBoundaryDisabledForTest = true;
        try
        {
            return Run(session, only, negativeControl: true);
        }
        finally
        {
            SpireHost.Patches.SchedulingPatches.RunSaveBoundaryDisabledForTest = saved;
        }
    }

    private static List<CaseResult> Run(HostSession session, string? only, bool negativeControl)
    {
        List<CaseResult> results = new List<CaseResult>();
        (string name, string character, string[] hand, int enemyHp, bool expectFrame)[] cases =
        {
            ("A: Simpleton auto-plays LockedIn+ (the game answers the hook choice), then a Strike kills", "IRONCLAD", new[] { "LOCKED_IN+", "STRIKE_IRONCLAD" }, 1, false),
            ("B: Simpleton auto-plays Prepared (CARD_SELECT frame inside the turn-end hook), then a Strike kills", "SILENT", new[] { "PREPARED", "STRIKE_SILENT", "STRIKE_SILENT", "DEFEND_SILENT" }, 5, true),
        };
        ulong seed = 7000;
        foreach ((string name, string character, string[] hand, int enemyHp, bool expectFrame) in cases)
        {
            if (only != null && !name.Contains(only, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            // Several policy seeds: the order inside a cost group and the picks of case B are random.
            for (int k = 0; k < 4; k++)
            {
                results.Add(RunCase(session, name + " #" + k, character, hand, enemyHp, expectFrame, ++seed, negativeControl));
            }
        }
        return results;
    }

    private static CaseResult RunCase(HostSession session, string name, string character, string[] hand, int enemyHp, bool expectFrame, ulong seed, bool negativeControl)
    {
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(seed); // COMBAT: END_TURN; everything else random
        int hookActions = 0;
        long yieldsBefore = SpireHost.Patches.SchedulingPatches.RunSaveBoundariesCombatEnd;
        int turnAtWin = -1;
        bool wonSeen = false;
        bool popActionAbort = false;
        CaseResult result = CaseRunner.Run(session, negativeControl ? "turnend-control" : "turnend", name + (negativeControl ? " [boundary off: must abort in PopAction]" : ""), character, async ctx =>
        {
            await ScenarioTools.JumpToEncounter(ctx, "BYRDONIS_ELITE");
            await ctx.Quiescence();
            ctx.RunManager.ActionQueueSet.ActionEnqueued += action =>
            {
                if (action is GenericHookGameAction)
                {
                    hookActions++;
                }
            };
            ctx.CombatManager.CombatWon += OnWon;
            await ScenarioTools.ApplyPower(ctx, "SIMPLETON_POWER", 8, ctx.LocalPlayer.Creature);
            // A known hand: the dealt cards go to the discard pile (CardPileCmd.Add moves a card), then the case's cards.
            foreach (MegaCrit.Sts2.Core.Models.CardModel dealt in ctx.LocalPlayer.PlayerCombatState!.Hand.Cards.ToList())
            {
                await MegaCrit.Sts2.Core.Commands.CardPileCmd.Add(dealt, PileType.Discard);
            }
            foreach (string id in hand)
            {
                bool upgraded = id.EndsWith("+", StringComparison.Ordinal);
                await ScenarioTools.AddCard(ctx, upgraded ? id.Substring(0, id.Length - 1) : id, PileType.Hand, upgrades: upgraded ? 1 : 0);
            }
            foreach (Creature enemy in ScenarioTools.CombatState(ctx).Enemies.ToList())
            {
                enemy.SetCurrentHpInternal(enemyHp);
            }

            void OnWon(MegaCrit.Sts2.Core.Rooms.CombatRoom room)
            {
                wonSeen = true;
                turnAtWin = ctx.LocalPlayer.PlayerCombatState?.TurnNumber ?? -1;
                ctx.CombatManager.CombatWon -= OnWon;
            }
        }, policy, probe, stopWhen: ctx => ctx.MapOpen || probe.MaxTurn >= 3, verify: (r, p) =>
        {
            if (negativeControl)
            {
                return popActionAbort ? null : "the case did NOT abort in ActionQueueSet.PopAction without the run-save boundary: the regression suite would not detect a missing boundary";
            }
            if (!wonSeen)
            {
                return "the combat was not won (the scenario did not reach the win check after the turn-end hook)";
            }
            if (p.CombatDecisions != 1 || turnAtWin != 1)
            {
                return $"{p.CombatDecisions} COMBAT decisions, won in turn {turnAtWin}; expected the win at the end of the first turn";
            }
            if (hookActions < 1)
            {
                return "no GenericHookGameAction was enqueued: the turn-end hook raised no player choice";
            }
            long yields = SpireHost.Patches.SchedulingPatches.RunSaveBoundariesCombatEnd - yieldsBefore;
            if (yields != 1)
            {
                return $"the run-save boundary of the combat end ran {yields} times, expected 1";
            }
            bool frame = p.Selects.Any(s => s.Kind == CardSelectKind.HandDiscard && s.Context == ChoiceContextKind.Hook && s.OutsidePlayPhase);
            if (expectFrame && !frame)
            {
                return "no CARD_SELECT frame was raised by the turn-end hook; saw [" + string.Join(", ", p.Selects.Select(s => s.Kind + "/" + s.Context)) + "]";
            }
            return null;
        }, hp: 99999, maxTurns: 6, maxDecisions: 300,
        tolerateAbort: negativeControl ? (abort => popActionAbort = abort.Reason == AbortReason.GameTaskFault && abort.Frames.Count > 0 && abort.Frames[0].Contains("ActionQueueSet.PopAction", StringComparison.Ordinal)) : null);
        if (negativeControl && !popActionAbort && result.Passed)
        {
            // The verify callback did not run (the episode ended otherwise): still a failed control.
            result.Passed = false;
            result.Failure = "the case did NOT abort in ActionQueueSet.PopAction without the run-save boundary";
        }
        return result;
    }
}
