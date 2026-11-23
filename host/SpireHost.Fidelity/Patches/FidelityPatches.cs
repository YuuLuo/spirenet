using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Enchantments;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Unlocks;
using MegaCrit.Sts2.Core.ValueProps;
using SpireHost.Driver;
using SpireHost.Engine;
using SpireHost.Patches;

namespace SpireHost.Fidelity.Patches;

/// <summary>
/// The fidelity patch pack (host\FIDELITY.md has one row per registry id below). Installed once per
/// process from <see cref="FidelityModule.Register"/>, on the engine thread, after boot. Every patch is
/// fail-closed: a missing target or a transpiler that does not find exactly the expected number of sites throws
/// at install time and the process does not start.
///
///   Fidelity      real-game branch of a TestMode gate restored (outcome / rng / bookkeeping sites)
///   Scheduling    a real-time wait mapped onto the pump (TCS / virtual clock), same logical order
///   Instrumentation  read-only counters, and assertions that fail the episode on a path proven unreachable
///
/// Nothing here is Behaviour (the default set must contain none; --selftest checks it).
/// </summary>
public static class FidelityPatches
{
    public static bool Installed { get; private set; }

    // ---- self-check counters (engine thread) ----
    public static long BoulderCalls;
    public static long BoulderHits;
    public static long BoulderSkippedRemoved;
    public static List<string>? BoulderLastOrder;
    public static long NaturalLossOnEnded;
    public static long UnpauseWaits;
    public static long UnpauseWakeups;
    public static long PreviousTurnLoopWaits;
    public static long PreviousTurnLoopTimeouts;
    public static long ActionFinishedBadState;

    /// <summary>GodotSharp methods whose body never runs because a host prefix returns first (Core GodotApiStubs); used by the native scan.</summary>
    public static HashSet<string> StubbedGodotMethods() => new HashSet<string>(StringComparer.Ordinal)
    {
        "Godot.OS::GetCmdlineArgs", "Godot.OS::HasFeature", "Godot.Time::GetTicksMsec", "Godot.ResourceLoader::Load",
        "Godot.ResourceLoader::Exists", "Godot.FileAccess::FileExists", "Godot.Callable::CallDeferred",
    };

    public static void Install()
    {
        if (Installed)
        {
            return;
        }
        Installed = true;
        InstallFidelityGates();
        InstallRollingBoulder();
        InstallNaturalDeath();
        InstallScheduling();
        InstallAssertions();
        InstallMonitors();
        InstallMend();
        PatchRegistry.Note("fidelity-run-invariants", "SpireHost.Fidelity.FidelityInvariants (IRunLifecycle 'fidelity:invariants', OnRunSetUp)", "driver lifecycle (read-only)", PatchCategory.Instrumentation,
            "Asserts every run that the audit sites proven inert stay inert: TestMode on, AutoSlayerCheck() false, TestRngInjector / TestFlags / RestSiteOption.generateForTests / " +
            "CardSelectorStack._testSelector unset, Standard non-custom run with ForceDiscoveryOrderModifications (the value of the real predicate), no Deprived seat. " +
            "Violation = Aborted(StartFailed, fidelity_assert:<id>).");
        PatchRegistry.Note("fidelity-assert-backstop", "SpireHost.Fidelity.FidelityInvariants (IEpisodeCheck 'fidelity:invariants')", "driver episode check (read-only)", PatchCategory.Instrumentation,
            "At every quiescent point and once more before teardown: fails the episode if a fidelity assertion fired but game code swallowed the exception (fidelity_assert_swallowed), " +
            "if a static test hook of the game is no longer inert (same ids as fidelity-run-invariants; a test scenario can exempt one explicitly with Scenario.Exempt), and once per episode " +
            "that CardSelectorStack.LocalSelector is installed (fidelity_assert:local_selector_installed; CardSelectCmd's UI branch dereferences null UI singletons, CardSelectCmd.cs:139-147).");
    }

    // =====================================================================================================
    //  Fidelity: TestMode gates restored to the real branch
    // =====================================================================================================

    private static void InstallFidelityGates()
    {
        GatePatch.ForceGetter("fidelity-merchant-potion-price",
            AccessTools.Method(typeof(MerchantPotionEntry), nameof(MerchantPotionEntry.CalcCost)),
            GatePatch.TestModeIsOff, true, 1, PatchCategory.Fidelity,
            "MerchantPotionEntry.CalcCost (Entities.Merchant/MerchantPotionEntry.cs:62-65): the real game multiplies the base price (50/75/100 by rarity) by " +
            "PlayerRng.Shops.NextFloat(0.95, 1.05) and rounds; TestMode keeps the base price and draws nothing, which shifts the whole Shops stream after it " +
            "(card prices MerchantCardEntry.cs:117, restocks, random potions). Real branch restored.");

        GatePatch.ForceGetter("fidelity-calling-bell-rewards",
            AccessTools.Method(typeof(CallingBell), "GenerateRewards"),
            GatePatch.TestModeIsOn, false, 1, PatchCategory.Fidelity,
            "CallingBell.GenerateRewards (Models.Relics/CallingBell.cs:39-63): the real game offers RelicReward(Common), RelicReward(Uncommon), RelicReward(Rare) " +
            "(populated from the relic grab bag, RelicReward.cs:66-86); TestMode offers the fixed Anchor / GremlinHorn / MummifiedHand and leaves the grab bag untouched. Real branch restored.");

        GatePatch.ForceGetter("fidelity-cauldron-rewards",
            AccessTools.Method(typeof(Cauldron), "GenerateRewards"),
            GatePatch.TestModeIsOn, false, 1, PatchCategory.Fidelity,
            "Cauldron.GenerateRewards (Models.Relics/Cauldron.cs:40-53): the real game offers 5 PotionReward(owner) populated from PlayerRng.Rewards (PotionReward.cs:54-61); " +
            "TestMode offers the fixed Flex/Weak/Vulnerable/Strength/Dexterity and draws nothing (every later card / potion reward shifts). Real branch restored.");

        GatePatch.ForceGetter("fidelity-merchant-room-exit",
            AccessTools.Method(typeof(MerchantRoom), nameof(MerchantRoom.Exit)),
            GatePatch.TestModeIsOn, false, 1, PatchCategory.Fidelity,
            "MerchantRoom.Exit (Rooms/MerchantRoom.cs:68-76): the real game calls MerchantSynchronizer.OnMerchantExited (MerchantSynchronizer.cs:119-168: unbought cards / relics / potions " +
            "into the map-point history, then _inventories.Clear()); TestMode returns early (history incomplete, the stale inventory survives into the next OnMerchantEntered, which logs a warning). Real branch restored.");

        GatePatch.ForceGetter("fidelity-map-history-player-stats",
            AccessTools.Method(typeof(RunManager), "UpdatePlayerStatsInMapPointHistory"),
            GatePatch.TestModeIsOn, false, 1, PatchCategory.Fidelity,
            "RunManager.UpdatePlayerStatsInMapPointHistory (Runs/RunManager.cs:1590-1605): the real game writes CurrentGold / CurrentHp / MaxHp of every player into the current " +
            "map-point history entry before every map move (:763) and in OnEnded (:1506); TestMode returns early (fields stay 0). Bookkeeping only (read by serialisation / metrics). Real branch restored.");
    }

    // =====================================================================================================
    //  Fidelity: RollingBoulderPower per target in list order (DECISIONS-H1 E)
    // =====================================================================================================

    private static void InstallRollingBoulder()
    {
        MethodInfo m = AccessTools.Method(typeof(RollingBoulderPower), nameof(RollingBoulderPower.AfterPlayerTurnStart))
            ?? throw new MissingMethodException("RollingBoulderPower.AfterPlayerTurnStart");
        PatchRegistry.Transpiler("fidelity-rolling-boulder-per-target", AccessTools.AsyncMoveNext(m),
            PatchRegistry.M(typeof(FidelityPatches), nameof(BoulderTranspiler)), PatchCategory.Fidelity,
            "RollingBoulderPower.AfterPlayerTurnStart (Models.Powers/RollingBoulderPower.cs:36-52): in the real game the boulder VFX captures HittableEnemies.ToList() " +
            "(NRollingBoulderVfx.cs:122-126) and its HitCreature signal starts one DoDamage per creature as the boulder passes it (:204-224); a creature whose node is gone " +
            "(died earlier and was removed from combat) is skipped without damage. TestMode calls DoDamage once for all HittableEnemies (batch: all HP losses first, then the hooks, " +
            "then one Kill). The single DoDamage call of the TestMode branch is replaced (exactly 1) by a host helper that calls the game's own DoDamage once per target, sequentially, " +
            "in list order (ruled: DECISIONS-H1 E, no extra IsHittable filter), skipping only creatures no longer in combat. Flash / SetAmount are the original code.");
    }

    public static IEnumerable<CodeInstruction> BoulderTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        MethodInfo doDamage = AccessTools.Method(typeof(RollingBoulderPower), "DoDamage") ?? throw new MissingMethodException("RollingBoulderPower.DoDamage");
        MethodInfo helper = PatchRegistry.M(typeof(FidelityPatches), nameof(BoulderPerTarget));
        return ReplaceCalls(instructions, doDamage, helper, 1, "fidelity-rolling-boulder-per-target");
    }

    /// <summary>Same stack shape as the instance call RollingBoulderPower.DoDamage(ctx, targets).</summary>
    public static async Task<IEnumerable<DamageResult>> BoulderPerTarget(RollingBoulderPower power, MegaCrit.Sts2.Core.GameActions.Multiplayer.PlayerChoiceContext choiceContext, IEnumerable<Creature> targets)
    {
        List<Creature> list = targets.ToList();
        BoulderCalls++;
        List<string> order = new List<string>(list.Count);
        List<DamageResult> results = new List<DamageResult>();
        foreach (Creature c in list)
        {
            if (c.CombatState == null || !c.CombatState.Enemies.Contains(c))
            {
                BoulderSkippedRemoved++; // real game: GetCreatureNode(c) == null -> counted as hit, no damage (NRollingBoulderVfx.cs:208-212)
                continue;
            }
            BoulderHits++;
            order.Add(c.CombatId?.ToString() ?? "?");
            results.AddRange(await power.DoDamage(choiceContext, new[] { c }));
        }
        BoulderLastOrder = order;
        return results;
    }

    // =====================================================================================================
    //  Fidelity: natural death -> RunManager.OnEnded(false)
    // =====================================================================================================

    private static void InstallNaturalDeath()
    {
        MethodInfo kill = AccessTools.Method(typeof(MegaCrit.Sts2.Core.Commands.CreatureCmd), nameof(MegaCrit.Sts2.Core.Commands.CreatureCmd.Kill),
            new[] { typeof(IReadOnlyCollection<Creature>), typeof(MegaCrit.Sts2.Core.Models.CardModel), typeof(bool) })
            ?? throw new MissingMethodException("CreatureCmd.Kill(IReadOnlyCollection<Creature>, CardModel, bool)");
        PatchRegistry.Transpiler("fidelity-natural-death-on-ended", AccessTools.AsyncMoveNext(kill),
            PatchRegistry.M(typeof(FidelityPatches), nameof(NaturalDeathTranspiler)), PatchCategory.Fidelity,
            "CreatureCmd.Kill (Commands/CreatureCmd.cs:376-391): when every player is dead the real game calls LoseCombat, stops the music and, if !RunManager.HasRunEnded, " +
            "RunManager.OnEnded(isVictory: false) + the game-over screen; TestMode only calls LoseCombat, so HasRunEnded stays false and OnEnded's bookkeeping (map history stats, " +
            "TurnsTaken, CheckUpdateEnemyDiscoveryAfterLoss, RunManager.cs:1498-1578) never runs on a loss although WinRun runs it on a win. The single TestMode.get_IsOff of the method " +
            "is replaced (exactly 1) by a host helper that performs `if (!HasRunEnded) OnEnded(false)` and returns false, so the music / screen block stays skipped.");
    }

    public static IEnumerable<CodeInstruction> NaturalDeathTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        return ReplaceCalls(instructions, GatePatch.TestModeIsOff, PatchRegistry.M(typeof(FidelityPatches), nameof(NaturalLossGate)), 1, "fidelity-natural-death-on-ended");
    }

    /// <summary>Replaces TestMode.IsOff at CreatureCmd.cs:382. Returns false: the UI block is still skipped.</summary>
    public static bool NaturalLossGate()
    {
        RunManager rm = RunManager.Instance;
        if (!rm.HasRunEnded)
        {
            NaturalLossOnEnded++;
            rm.OnEnded(isVictory: false);
        }
        return false;
    }

    // =====================================================================================================
    //  Scheduling leftovers
    // =====================================================================================================

    private sealed class UnpauseBox
    {
        public TaskCompletionSource? Tcs;
    }

    private static readonly ConditionalWeakTable<ActionExecutor, UnpauseBox> _unpause = new ConditionalWeakTable<ActionExecutor, UnpauseBox>();

    private static void InstallScheduling()
    {
        MethodInfo wait = AccessTools.Method(typeof(ActionExecutor), "WaitForUnpause") ?? throw new MissingMethodException("ActionExecutor.WaitForUnpause");
        PatchRegistry.Transpiler("fidelity-executor-unpause-signal", AccessTools.AsyncMoveNext(wait),
            PatchRegistry.M(typeof(FidelityPatches), nameof(UnpauseTranspiler)), PatchCategory.Scheduling,
            "ActionExecutor.WaitForUnpause (GameActions/ActionExecutor.cs:205-222): TestMode polls `while (_isPaused) await Task.Delay(50)` (the real game awaits one process " +
            "frame per poll). DECISIONS-H1 A.5: the poll's Task.Delay is replaced (exactly 1) by a task that completes when ActionExecutor.Unpause is called on the same executor " +
            "(posted to the pump: RunContinuationsAsynchronously), so a paused executor creates no timer - real or virtual - and resumes on the next pump turn after Unpause; " +
            "the loop still re-checks _isPaused. Same logical order, no time. (The H0 patch virtual-clock-executor-unpause-poll on this method was removed: one patch per site.)");
        PatchRegistry.Postfix("fidelity-executor-unpause-wake", AccessTools.Method(typeof(ActionExecutor), nameof(ActionExecutor.Unpause)),
            PatchRegistry.M(typeof(FidelityPatches), nameof(UnpausePostfix)), PatchCategory.Scheduling,
            "ActionExecutor.Unpause (ActionExecutor.cs:71-75): completes the wait task of fidelity-executor-unpause-signal for this executor (if any). Reads / writes no game state.");

        MethodInfo prev = AccessTools.Method(typeof(CombatManager), "WaitForPreviousTurnLoopToFinish") ?? throw new MissingMethodException("CombatManager.WaitForPreviousTurnLoopToFinish");
        PatchRegistry.Transpiler("fidelity-turn-loop-wait-virtual-timeout", AccessTools.AsyncMoveNext(prev),
            PatchRegistry.M(typeof(FidelityPatches), nameof(TurnLoopWaitTranspiler)), PatchCategory.Scheduling,
            "CombatManager.WaitForPreviousTurnLoopToFinish (Combat/CombatManager.cs:343-370): `await previousTurnLoopTask.WaitAsync(10 s)` arms a real System.Threading timer " +
            "whose expiry would run on the thread pool. Replaced (exactly 1) by the same wait with the 10 s timeout on the pump's virtual clock (TimeoutException on expiry, the task's " +
            "own fault otherwise - the game's catch blocks are unchanged). Only reached when a combat starts while the previous combat's turn loop is still alive (never observed: counted).");
    }

    public static IEnumerable<CodeInstruction> UnpauseTranspiler(IEnumerable<CodeInstruction> instructions, MethodBase original)
    {
        List<CodeInstruction> code = new List<CodeInstruction>(instructions);
        MethodInfo taskDelay = AccessTools.Method(typeof(Task), nameof(Task.Delay), new[] { typeof(int) });
        FieldInfo self = AccessTools.Field(original.DeclaringType, "<>4__this") ?? throw new MissingFieldException("WaitForUnpause state machine <>4__this");
        MethodInfo helper = PatchRegistry.M(typeof(FidelityPatches), nameof(UnpauseSignal));
        int hits = 0;
        for (int i = 0; i < code.Count; i++)
        {
            CodeInstruction ci = code[i];
            if ((ci.opcode == OpCodes.Call || ci.opcode == OpCodes.Callvirt) && ci.operand is MethodInfo m && m == taskDelay)
            {
                CodeInstruction ld = new CodeInstruction(OpCodes.Ldarg_0);
                ld.labels.AddRange(ci.labels);
                ld.blocks.AddRange(ci.blocks);
                ci.labels.Clear();
                ci.blocks.Clear();
                code.Insert(i, ld);
                code.Insert(i + 1, new CodeInstruction(OpCodes.Ldfld, self));
                ci.opcode = OpCodes.Call;
                ci.operand = helper;
                i += 2;
                hits++;
            }
        }
        if (hits != 1)
        {
            throw new InvalidOperationException($"fidelity-executor-unpause-signal: expected exactly 1 Task.Delay(int) call in {original.DeclaringType?.FullName}::{original.Name}, found {hits}; the game assembly changed (or another patch already replaced the call).");
        }
        return code;
    }

    /// <summary>Replaces the poll delay: a task completed by the next Unpause of this executor.</summary>
    public static Task UnpauseSignal(int milliseconds, ActionExecutor executor)
    {
        UnpauseWaits++;
        if (!executor.IsPaused)
        {
            return Task.CompletedTask;
        }
        UnpauseBox box = _unpause.GetValue(executor, static _ => new UnpauseBox());
        box.Tcs ??= new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        return box.Tcs.Task;
    }

    public static void UnpausePostfix(ActionExecutor __instance)
    {
        if (_unpause.TryGetValue(__instance, out UnpauseBox? box) && box.Tcs != null)
        {
            TaskCompletionSource t = box.Tcs;
            box.Tcs = null;
            UnpauseWakeups++;
            t.TrySetResult();
        }
    }

    /// <summary>For the self-test: is a wait task pending for this executor?</summary>
    public static bool HasPendingUnpause(ActionExecutor executor) => _unpause.TryGetValue(executor, out UnpauseBox? b) && b.Tcs != null;

    public static IEnumerable<CodeInstruction> TurnLoopWaitTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        MethodInfo waitAsync = AccessTools.Method(typeof(Task), nameof(Task.WaitAsync), new[] { typeof(TimeSpan) });
        return ReplaceCalls(instructions, waitAsync, PatchRegistry.M(typeof(FidelityPatches), nameof(VirtualWaitAsync)), 1, "fidelity-turn-loop-wait-virtual-timeout");
    }

    /// <summary>Task.WaitAsync(timeout) on the virtual clock (same stack shape as the instance call).</summary>
    public static async Task VirtualWaitAsync(Task task, TimeSpan timeout)
    {
        PreviousTurnLoopWaits++;
        if (task.IsCompleted)
        {
            await task;
            return;
        }
        Task delay = SchedulingPatches.Delay(timeout);
        Task first = await Task.WhenAny(task, delay);
        if (first != task)
        {
            PreviousTurnLoopTimeouts++;
            throw new TimeoutException("previous turn loop still running after " + timeout.TotalSeconds + " virtual seconds");
        }
        await task;
    }

    // =====================================================================================================
    //  Assertions: proven-unreachable paths fail the episode if ever hit
    // =====================================================================================================

    private static void InstallAssertions()
    {
        MethodInfo timer = AccessTools.Method(typeof(CombatState), "GodotTimerTask") ?? throw new MissingMethodException("CombatState.GodotTimerTask");
        PatchRegistry.Prefix("fidelity-assert-creature-spawn-wait", timer, PatchRegistry.M(typeof(FidelityPatches), nameof(CreatureSpawnWaitPrefix)), PatchCategory.Instrumentation,
            "Assertion. CombatState.GodotTimerTask (Combat/CombatState.cs:656-665) is the 10 s timeout of GetCreatureAsync (:483-515), created only when a played card / potion targets a " +
            "combat id that has not been spawned yet (id >= _nextCreatureId). In the host every target comes from an existing creature, for every seat: a virtual client builds its action " +
            "from the host's own state at the decision point and the host executes it before it asks again, so a client is never ahead of the host (the case the wait exists for: a real " +
            "client that targets a creature the host has not spawned yet). So this is unreachable with one seat and with several; if it is ever hit the " +
            "episode fails (StrictViolation, fidelity_assert:creature_spawn_wait) instead of arming a timer: the method body never runs, so the real Task.Delay inside it cannot be reached " +
            "(the H0 patch virtual-clock-creature-spawn-timeout on this method was removed: one patch per site).");

        MethodInfo hover = AccessTools.Method(typeof(RestSiteSynchronizer), "TrySendHoverMessage") ?? throw new MissingMethodException("RestSiteSynchronizer.TrySendHoverMessage");
        PatchRegistry.Prefix("fidelity-assert-rest-hover-delay", hover, PatchRegistry.M(typeof(FidelityPatches), nameof(RestHoverPrefix)), PatchCategory.Instrumentation,
            "Assertion. RestSiteSynchronizer.TrySendHoverMessage (Multiplayer.Game/RestSiteSynchronizer.cs:293-307) schedules QueueHoverMessage (Task.Delay, :311) / " +
            "SendHoverMessageAfterSmallDelay (Task.Yield, :317). Only caller: LocalOptionHovered <- NRestSiteRoom.cs:370/376 (UI hover). The host never hovers: unreachable; a hit fails the episode.");

        MethodInfo sync = AccessTools.Method(typeof(PeerInputSynchronizer), "TrySendSyncMessage") ?? throw new MissingMethodException("PeerInputSynchronizer.TrySendSyncMessage");
        PatchRegistry.Prefix("fidelity-assert-peer-input-delay", sync, PatchRegistry.M(typeof(FidelityPatches), nameof(PeerInputPrefix)), PatchCategory.Instrumentation,
            "Assertion. PeerInputSynchronizer.TrySendSyncMessage (Multiplayer.Game.PeerInput/PeerInputSynchronizer.cs:180-194) schedules QueueSyncMessage (Task.Delay, :198) for " +
            "cursor / input sync of remote peers; only reached from UI input (NTargetManager / cursor). Unreachable in the host; a hit fails the episode.");

        foreach ((Type t, string member, bool setter) in new (Type, string, bool)[]
        {
            (typeof(SneckoOil), nameof(SneckoOil.TestEnergyCostOverride), true),
            (typeof(ConfusedPower), "TestEnergyCostOverride", true),
            (typeof(Slither), nameof(Slither.TestEnergyCostOverride), true),
            (typeof(SneckoEye), nameof(SneckoEye.SetTestEnergyCostOverride), false),
            (typeof(FakeSneckoEye), nameof(FakeSneckoEye.SetTestEnergyCostOverride), false),
        })
        {
            MethodBase target = setter ? (AccessTools.PropertySetter(t, member) ?? throw new MissingMethodException(t.Name + "." + member + " setter"))
                : (AccessTools.Method(t, member) ?? throw new MissingMethodException(t.Name + "." + member));
            PatchRegistry.Prefix("fidelity-guard-test-energy-cost-" + t.Name.ToLowerInvariant(), target, PatchRegistry.M(typeof(FidelityPatches), nameof(TestEnergyCostPrefix)), PatchCategory.Instrumentation,
                "Assertion. " + t.Name + "." + member + " is a test-only cost override (Slither.cs:14-27, SneckoOil.cs:29-40, ConfusedPower.cs:17-28, SneckoEye.cs:53-59, FakeSneckoEye.cs:42-48): " +
                "TestMode permits it and NextEnergyCost then returns the override instead of drawing Rng.CombatEnergyCosts. No production code sets it (the relays inside SneckoEye / " +
                "FakeSneckoEye only fire when their own field, default -1, was set). Any call fails the episode, so cost randomisation always consumes the real stream.");
        }
    }

    public static bool CreatureSpawnWaitPrefix()
    {
        FidelityAssert.Fail("creature_spawn_wait", AbortReason.StrictViolation,
            "CombatState.GetCreatureAsync waited for a creature id that does not exist yet (CombatState.cs:495-500): unreachable in single player.");
        return false;
    }

    public static bool RestHoverPrefix()
    {
        FidelityAssert.Fail("rest_hover_delay", AbortReason.StrictViolation, "RestSiteSynchronizer.TrySendHoverMessage reached (UI hover path, Task.Delay at RestSiteSynchronizer.cs:311).");
        return false;
    }

    public static bool PeerInputPrefix()
    {
        FidelityAssert.Fail("peer_input_delay", AbortReason.StrictViolation, "PeerInputSynchronizer.TrySendSyncMessage reached (UI input sync, Task.Delay at PeerInputSynchronizer.cs:198).");
        return false;
    }

    public static bool TestEnergyCostPrefix(MethodBase __originalMethod)
    {
        FidelityAssert.Fail("test_energy_cost_override", AbortReason.StrictViolation,
            __originalMethod.DeclaringType?.Name + "." + __originalMethod.Name + ": a test-only energy-cost override was set; cost randomisation must draw Rng.CombatEnergyCosts.");
        return false;
    }

    // =====================================================================================================
    //  Monitors (read-only)
    // =====================================================================================================

    private static void InstallMonitors()
    {
        MethodInfo exec = AccessTools.Method(typeof(GameAction), nameof(GameAction.Execute)) ?? throw new MissingMethodException("GameAction.Execute");
        PatchRegistry.Transpiler("fidelity-monitor-action-finished-bad-state", AccessTools.AsyncMoveNext(exec),
            PatchRegistry.M(typeof(FidelityPatches), nameof(ActionMonitorTranspiler)), PatchCategory.Instrumentation,
            "Read-only. GameAction.Execute (GameActions/GameAction.cs:96-106): an action that finishes while not in state Executing throws under NonInteractive (the real game logs + " +
            "Sentry and continues). Kept fail-closed (DECISIONS-H1 A.7): the single NonInteractiveMode.get_IsActive there is replaced (exactly 1) by a helper that counts the event and " +
            "returns the same value, so the fault still aborts the episode and the count is visible (--selftest / fuzz stderr summary). If it is ever non-zero the lead decides whether to flip it.");
    }

    public static IEnumerable<CodeInstruction> ActionMonitorTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        return ReplaceCalls(instructions, GatePatch.NonInteractiveIsActive, PatchRegistry.M(typeof(FidelityPatches), nameof(ActionFinishedInBadState)), 1, "fidelity-monitor-action-finished-bad-state");
    }

    public static bool ActionFinishedInBadState()
    {
        ActionFinishedBadState++;
        return NonInteractiveMode.IsActive;
    }

    // =====================================================================================================
    //  MEND (multi-seat, phase H5; lead ruling Q3)
    // =====================================================================================================

    /// <summary>MEND options chosen by the local seat that took the remote branch (ledger self-check).</summary>
    public static long MendLocalTargetings;

    private static void InstallMend()
    {
        MethodInfo onSelect = AccessTools.Method(typeof(MegaCrit.Sts2.Core.Entities.RestSite.MendRestSiteOption), nameof(MegaCrit.Sts2.Core.Entities.RestSite.MendRestSiteOption.OnSelect))
            ?? throw new MissingMethodException("MendRestSiteOption.OnSelect");
        PatchRegistry.Transpiler("fidelity-mend-local-targeting", AccessTools.AsyncMoveNext(onSelect),
            PatchRegistry.M(typeof(FidelityPatches), nameof(MendTranspiler)), PatchCategory.Fidelity,
            "MULTI-SEAT ONLY (the MEND option exists only with more than one player, Entities.RestSite/RestSiteOption.cs:56-59). MendRestSiteOption.OnSelect " +
            "(Entities.RestSite/MendRestSiteOption.cs:78-158) asks the owner for the player to heal. For the LOCAL owner that is UI code with no TestMode gate - " +
            "NRestSiteRoom.Instance.AnimateDescriptionDown(), GetButtonForOption, NTargetManager.StartTargeting / SelectionFinished (:84-131) - which dereferences the null " +
            "NRestSiteRoom.Instance headless; for any other owner it waits for that player's choice (PlayerChoiceSynchronizer.WaitForRemoteChoice(...).AsPlayerId(), :135-139). " +
            "Both branches end in the same lines (:141-157: heal the TARGET by the target's own rest amount, Hook.AfterRestSiteHeal). Transpiler: the one " +
            "LocalContext.IsMe(Owner) of the method (:82) evaluates to false, so the local owner's MEND takes the waiting branch too; the driver answers that wait with the " +
            "target the REST decision named (rooms:mend-target: PlayerChoiceSynchronizer.ReceiveReplayChoice for seat 0, the client's PlayerChoiceMessage for a remote seat). " +
            "WaitForRemoteChoice has no locality check (GameActions.Multiplayer/PlayerChoiceSynchronizer.cs:78-111). Difference to the local branch: " +
            "PlayerChoiceSynchronizer.SyncLocalChoice (:118) is not called for seat 0, i.e. no PlayerChoiceMessage broadcast (no receiver) and the PlayerChoiceReceived event fires " +
            "from the receive path instead - the same as for every prompt the host answers through its local selector. Cancelling the targeting (a no-op: OnSelect returns false) is not offered. " +
            "[transpiler: exactly 1 call of LocalContext.IsMe(Player) -> FidelityPatches.MendOwnerIsLocal(Player) = false]");
    }

    public static IEnumerable<CodeInstruction> MendTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        MethodInfo isMe = AccessTools.Method(typeof(MegaCrit.Sts2.Core.Context.LocalContext), nameof(MegaCrit.Sts2.Core.Context.LocalContext.IsMe), new[] { typeof(MegaCrit.Sts2.Core.Entities.Players.Player) })
            ?? throw new MissingMethodException("LocalContext.IsMe(Player)");
        return ReplaceCalls(instructions, isMe, PatchRegistry.M(typeof(FidelityPatches), nameof(MendOwnerIsLocal)), 1, "fidelity-mend-local-targeting");
    }

    /// <summary>Stand-in for LocalContext.IsMe(Owner) in MendRestSiteOption.OnSelect: the targeting branch is never taken.</summary>
    public static bool MendOwnerIsLocal(MegaCrit.Sts2.Core.Entities.Players.Player? owner)
    {
        if (MegaCrit.Sts2.Core.Context.LocalContext.IsMe(owner))
        {
            MendLocalTargetings++;
        }
        return false;
    }

    // =====================================================================================================

    /// <summary>Replaces exactly <paramref name="expected"/> calls of <paramref name="from"/> by a static call of <paramref name="to"/> (same stack shape); throws otherwise.</summary>
    public static IEnumerable<CodeInstruction> ReplaceCalls(IEnumerable<CodeInstruction> instructions, MethodInfo from, MethodInfo to, int expected, string id)
    {
        List<CodeInstruction> code = new List<CodeInstruction>(instructions);
        int hits = 0;
        foreach (CodeInstruction ci in code)
        {
            if ((ci.opcode == OpCodes.Call || ci.opcode == OpCodes.Callvirt) && ci.operand is MethodInfo m && m == from)
            {
                ci.opcode = OpCodes.Call;
                ci.operand = to;
                hits++;
            }
        }
        if (hits != expected)
        {
            throw new InvalidOperationException($"{id}: expected exactly {expected} call(s) of {from.DeclaringType?.Name}.{from.Name}, found {hits}; the game assembly changed.");
        }
        return code;
    }

    public static object Counters() => new
    {
        boulder_calls = BoulderCalls,
        boulder_hits = BoulderHits,
        boulder_skipped_removed = BoulderSkippedRemoved,
        natural_loss_on_ended = NaturalLossOnEnded,
        unpause_waits = UnpauseWaits,
        unpause_wakeups = UnpauseWakeups,
        previous_turn_loop_waits = PreviousTurnLoopWaits,
        previous_turn_loop_timeouts = PreviousTurnLoopTimeouts,
        action_finished_bad_state = ActionFinishedBadState,
        mend_local_targetings = MendLocalTargetings,
        asserts_fired = FidelityAssert.Fired,
        asserts_by_id = FidelityAssert.ById,
        core_tutorial_room_forced = FidelityGatePatches.TutorialRoomForced,
        core_discovery_boss_changed = FidelityGatePatches.DiscoveryBossChanged,
        core_discovery_rooms_changed = FidelityGatePatches.DiscoveryRoomsChanged,
        core_discovery_order_calls = FidelityGatePatches.DiscoveryOrderCalls,
    };
}
