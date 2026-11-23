using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Saves;
using SpireHost.Engine;

namespace SpireHost.Patches;

/// <summary>
/// The Core's scheduling patches: places where the headless host has to supply an asynchronous boundary, a
/// continuation thread or a clock that the real client gets from Godot's main loop, its thread pool or wall time.
/// None of them changes what the game computes; each one fails closed (exactly N call sites, or a missing
/// target throws at boot).
///
///   run-save-completes-after-pump-turn        SaveManager.SaveRun never completes synchronously for its awaiter
///   hook-action-started-inline-continuation   ContinueWith without a scheduler -> inline on the engine thread
///   virtual-clock-amalgamator-*               Task.Delay(300) -> the pump's virtual clock
///
/// (The two H0 patches virtual-clock-executor-unpause-poll and virtual-clock-creature-spawn-timeout were removed by
/// the H1 integration: the first is superseded by the fidelity pack's unpause signal, the second is pre-empted by
/// its assertion on the same method. One patch per site.)
/// </summary>
public static class SchedulingPatches
{
    public static long VirtualDelays;
    public static long RealDelays;

    /// <summary>Continuations of ContinueInline that did not run on the engine thread. Must stay 0 (strict-mode fault).</summary>
    public static long OffThreadContinuations;

    /// <summary>ContinueInline calls (hook-initiated player choices).</summary>
    public static long InlineContinuations;

    // ---- run-save boundary counters (engine thread) ----

    /// <summary>SaveManager.SaveRun calls whose completion was moved to the next pump turn, by call site.</summary>
    public static long RunSaveBoundariesCombatEnd;     // CombatManager.EndCombatInternal   (Combat/CombatManager.cs:1059: saveProgress false)
    public static long RunSaveBoundariesRoomEntry;     // RunManager.EnterMapPointInternal  (Runs/RunManager.cs:778: preFinishedRoom null)
    public static long RunSaveBoundariesEventRoom;     // EventRoom fire-and-forget         (Rooms/EventRoom.cs:132: nobody awaits it)

    /// <summary>Of those: the pump queue was NOT empty when the save was requested, i.e. something else runs before the caller resumes.</summary>
    public static long RunSaveQueuedWorkCombatEnd;
    public static long RunSaveQueuedWorkRoomEntry;
    public static long RunSaveQueuedWorkEventRoom;

    /// <summary>Calls left on the game's own schedule: no engine pump on the calling thread, or the test switch below. 0 in production.</summary>
    public static long RunSaveBoundariesSkipped;

    /// <summary>
    /// TEST ONLY. True = <c>run-save-completes-after-pump-turn</c> passes the game's task through unchanged (the patch
    /// stays installed: the patch SET is identical). Set and reset only by the negative control of the combat
    /// self-test suite "turnend", which proves that the suite's cases abort in ActionQueueSet.PopAction without the
    /// boundary. No command-line option, environment variable or protocol field can set it.
    /// </summary>
    public static bool RunSaveBoundaryDisabledForTest;

    public static void Install()
    {
        MethodInfo saveRun = AccessTools.Method(typeof(SaveManager), nameof(SaveManager.SaveRun), new[] { typeof(AbstractRoom), typeof(bool) })
            ?? throw new MissingMethodException("SaveManager.SaveRun(AbstractRoom, bool)");
        PatchRegistry.Postfix("run-save-completes-after-pump-turn", saveRun,
            PatchRegistry.M(typeof(SchedulingPatches), nameof(RunSavePostfix)), PatchCategory.Scheduling,
            "SaveManager.SaveRun (Saves/SaveManager.cs:570-597). Real client (Singleplayer / Host, ShouldSave): RunSaveManager.SaveRun serialises the run and awaits " +
            "ISaveStore.WriteFileAsync (Saves.Managers/RunSaveManager.cs:74-97), which writes through Stream.WriteAsync on a stream that only overrides the synchronous Write " +
            "(Saves/GodotFileIo.cs:137-161, Saves/FileAccessStream.cs), i.e. on the thread pool: the task is never complete when SaveRun returns, so EVERY awaiter resumes later, " +
            "from the main-thread context. Host: shouldSave false and a synchronous in-memory store make the task complete synchronously and the awaiter runs straight on. " +
            "Two awaiters exist in game logic: CombatManager.EndCombatInternal (Combat/CombatManager.cs:1059; its remainder reaches ActionQueueSet.CombatEnded(), :1077, which must not run " +
            "inside the finally block of a hook action's GameAction.Execute, GameActions/GameAction.cs:92-97 -> PopAction throws, GameActions.Multiplayer/ActionQueueSet.cs:525) and " +
            "RunManager.EnterMapPointInternal (Runs/RunManager.cs:776-779, before the room type roll and EnterRoom); EventRoom.cs:132 starts one without awaiting it. " +
            "Postfix (uniform rule, ruling E2): the returned task is replaced by one that awaits the game's own save and then yields once to the engine pump, so every awaiter resumes " +
            "from the pump queue, after everything queued before it - where the real client resumes it. Nothing is computed differently. Counted per call site, together with how often " +
            "other work was queued at that moment. Fails closed if the continuation is ever resumed off the engine thread.");

        MethodInfo getHook = AccessTools.Method(typeof(MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionQueueSynchronizer), "GetHookActionForId")
            ?? throw new MissingMethodException("ActionQueueSynchronizer.GetHookActionForId");
        PatchRegistry.Transpiler("hook-action-started-inline-continuation", getHook,
            PatchRegistry.M(typeof(SchedulingPatches), nameof(ReplaceContinueWithTranspiler)), PatchCategory.Scheduling,
            "ActionQueueSynchronizer.GetHookActionForId (ActionQueueSynchronizer.cs:343-353) registers `action.ExecutionStartedTask.ContinueWith(_ => HookActionStarted(action))` without a scheduler: " +
            "on the engine thread TaskScheduler.Current is Default, so HookActionStarted (-> _hookActions.Remove, :369-373) would run on a THREAD-POOL thread, racing the engine thread on a List. " +
            "The call is replaced (exactly 1) by the same ContinueWith with TaskContinuationOptions.ExecuteSynchronously: the continuation runs inline on the thread that starts the hook action, " +
            "i.e. the engine thread. In the real game the moment is nondeterministic; inline is one of its legal schedules. Off-thread executions are counted (must be 0).");

        MethodInfo delayInt = PatchRegistry.M(typeof(SchedulingPatches), nameof(ReplaceDelayIntTranspiler));
        Type amalgamator = typeof(MegaCrit.Sts2.Core.Models.Events.Amalgamator);
        foreach (string name in new[] { "CombineStrikes", "CombineDefends" })
        {
            MethodInfo m = AccessTools.Method(amalgamator, name) ?? throw new MissingMethodException("Amalgamator." + name);
            PatchRegistry.Transpiler("virtual-clock-amalgamator-" + name.ToLowerInvariant(), AccessTools.AsyncMoveNext(m), delayInt, PatchCategory.Scheduling,
                "Amalgamator." + name + " awaits a real Task.Delay(300) between two screen shakes (Amalgamator.cs:44-49 / :58-63; no TestMode gate). Redirected (exactly 1) to the pump's virtual clock. " +
                "SHARED SITE (ruling B3): the events module's events-amalgamator-" + (name == "CombineStrikes" ? "combine-strikes" : "combine-defends") +
                "-shake swaps the two unguarded NGame.Instance.ScreenShakeTrauma calls of the same method.");
        }
    }

    // -----------------------------------------------------------------------------------------------------
    //  run-save-completes-after-pump-turn
    // -----------------------------------------------------------------------------------------------------

    /// <summary>SaveRun calls the boundary could not be applied to (no engine pump, or not on the engine thread) outside the test switch. Strict: must be 0.</summary>
    public static long RunSaveBoundariesNotApplied;

    public static void RunSavePostfix(AbstractRoom? preFinishedRoom, bool saveProgress, ref Task __result)
    {
        EnginePump? pump = EnginePump.Current;
        if (RunSaveBoundaryDisabledForTest || pump == null || !pump.OnEngineThread)
        {
            // Without the engine pump Task.Yield would queue the awaiter's remainder to the thread pool. Not a host
            // configuration (every command runs on the engine thread of a HostSession); counted, never silent.
            RunSaveBoundariesSkipped++;
            if (!RunSaveBoundaryDisabledForTest)
            {
                // Verifier finding R-13: outside the one test switch this changes scheduling (the awaiter continues inline), so
                // it is a STRICT counter: the episode it happens in is aborted (run_save_boundary_skipped).
                System.Threading.Interlocked.Increment(ref RunSaveBoundariesNotApplied);
            }
            return;
        }
        bool queued = pump.QueueCount > 0;
        if (!saveProgress)
        {
            RunSaveBoundariesCombatEnd++;
            if (queued)
            {
                RunSaveQueuedWorkCombatEnd++;
            }
        }
        else if (preFinishedRoom == null)
        {
            RunSaveBoundariesRoomEntry++;
            if (queued)
            {
                RunSaveQueuedWorkRoomEntry++;
            }
        }
        else
        {
            RunSaveBoundariesEventRoom++;
            if (queued)
            {
                RunSaveQueuedWorkEventRoom++;
            }
        }
        __result = AfterPumpTurn(__result, pump);
    }

    /// <summary>The game's own save, then one turn of the engine pump - the stand-in for the real client's thread-pool file write.</summary>
    private static async Task AfterPumpTurn(Task save, EnginePump pump)
    {
        await save;
        await Task.Yield(); // SynchronizationContext.Current is the engine pump: the continuation is appended to its queue
        if (!pump.OnEngineThread)
        {
            // Fail closed: game code must never continue on a second thread. The fault surfaces as an aborted episode.
            throw new InvalidOperationException("run-save-completes-after-pump-turn: the awaiter of SaveManager.SaveRun was resumed on managed thread " +
                                                Environment.CurrentManagedThreadId + ", not on the engine thread " + pump.EngineThreadId + ".");
        }
    }

    // -----------------------------------------------------------------------------------------------------
    //  hook-action-started-inline-continuation
    // -----------------------------------------------------------------------------------------------------

    public static Task ContinueInline(Task antecedent, Action<Task> continuation)
    {
        InlineContinuations++;
        return antecedent.ContinueWith(t =>
        {
            EnginePump? pump = EnginePump.Current;
            if (pump != null && !pump.OnEngineThread)
            {
                System.Threading.Interlocked.Increment(ref OffThreadContinuations);
            }
            continuation(t);
        }, System.Threading.CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
    }

    public static IEnumerable<CodeInstruction> ReplaceContinueWithTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        return Replace(instructions,
            AccessTools.Method(typeof(Task), nameof(Task.ContinueWith), new[] { typeof(Action<Task>) }),
            AccessTools.Method(typeof(SchedulingPatches), nameof(ContinueInline)));
    }

    // -----------------------------------------------------------------------------------------------------
    //  virtual clock
    // -----------------------------------------------------------------------------------------------------

    public static Task Delay(int milliseconds)
    {
        return Delay(TimeSpan.FromMilliseconds(milliseconds));
    }

    /// <summary>A delay on the pump's virtual clock (engine thread). Off the engine thread it would be a real timer: counted as a strict violation.</summary>
    public static Task Delay(TimeSpan delay)
    {
        EnginePump? pump = EnginePump.Current;
        if (pump != null && pump.OnEngineThread)
        {
            VirtualDelays++;
            return pump.Clock.Delay(delay);
        }
        RealDelays++;
        return Task.Delay(delay);
    }

    public static IEnumerable<CodeInstruction> ReplaceDelayIntTranspiler(IEnumerable<CodeInstruction> instructions)
    {
        return Replace(instructions,
            AccessTools.Method(typeof(Task), nameof(Task.Delay), new[] { typeof(int) }),
            AccessTools.Method(typeof(SchedulingPatches), nameof(Delay), new[] { typeof(int) }));
    }

    private static IEnumerable<CodeInstruction> Replace(IEnumerable<CodeInstruction> instructions, MethodInfo from, MethodInfo to)
    {
        List<CodeInstruction> list = new List<CodeInstruction>(instructions);
        int hits = 0;
        foreach (CodeInstruction ci in list)
        {
            if ((ci.opcode == OpCodes.Call || ci.opcode == OpCodes.Callvirt) && ci.operand is MethodInfo m && m == from)
            {
                ci.opcode = OpCodes.Call;
                ci.operand = to;
                hits++;
            }
        }
        if (hits != 1)
        {
            throw new InvalidOperationException($"Expected exactly one {from.DeclaringType?.Name}.{from.Name} call, found {hits}; the game assembly changed.");
        }
        return list;
    }
}
