using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using HarmonyLib;

namespace SpireHost.Patches;

/// <summary>
/// Proof instrument for the pump: a Harmony prefix on hot gameplay methods records
/// Environment.CurrentManagedThreadId. Any call that is not on the engine thread is counted per method
/// (and per thread id). Read-only; installed only with --audit-threads.
/// </summary>
public static class ThreadAudit
{
    private static int _engineThreadId;
    private static long _onEngine;
    private static long _offEngine;
    private static readonly ConcurrentDictionary<string, long> _offByMethod = new ConcurrentDictionary<string, long>();
    private static readonly ConcurrentDictionary<int, long> _offByThread = new ConcurrentDictionary<int, long>();
    private static readonly ConcurrentDictionary<string, long> _callsByMethod = new ConcurrentDictionary<string, long>();

    public static bool Installed { get; private set; }

    /// <summary>Audited calls that ran on a thread other than the engine thread. Must be 0.</summary>
    public static long OffEngineCalls => Interlocked.Read(ref _offEngine);

    /// <summary>Audited calls on the engine thread (a clean audit with 0 calls proves nothing).</summary>
    public static long OnEngineCalls => _onEngine;

    public static void SetEngineThread(int managedThreadId)
    {
        _engineThreadId = managedThreadId;
    }

    public static void Install()
    {
        if (Installed)
        {
            return;
        }
        Installed = true;
        Assembly game = typeof(MegaCrit.Sts2.Core.Models.ModelDb).Assembly;
        MethodInfo prefix = PatchRegistry.M(typeof(ThreadAudit), nameof(Prefix));
        List<MethodBase> targets = new List<MethodBase>();

        // Synchronous entry points of the hot gameplay paths (async methods run their first segment on the caller
        // thread, so the async state machines' MoveNext are patched as well: that is where continuations land).
        AddAll(targets, typeof(MegaCrit.Sts2.Core.GameActions.GameAction), "Execute");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.GameActions.PlayCardAction), "ExecuteAction");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.GameActions.ActionExecutor), "ExecuteActions");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Models.CardModel), "OnPlayWrapper");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Commands.CreatureCmd), "Damage");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Commands.CreatureCmd), "GainBlock");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Commands.CardPileCmd), "Draw");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Commands.PowerCmd), "Apply");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Commands.PlayerCmd), "EndTurn");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Entities.Creatures.Creature), "TakeTurn");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Models.MonsterModel), "PerformMove");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Combat.CombatManager), "StartTurn");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Combat.CombatManager), "ExecuteEnemyTurn");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Combat.CombatManager), "EndPlayerTurnPhaseOneInternal");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Combat.CombatManager), "EndCombatInternal");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.Combat.CombatManager), "CheckWinCondition");
        // choice protocol (only reached with --selector local): GetHookActionForId uses Task.ContinueWith without a scheduler
        AddAll(targets, typeof(MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionQueueSynchronizer), "HookActionStarted");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionQueueSynchronizer), "RequestResumeActionAfterPlayerChoice");
        AddAll(targets, typeof(MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionQueueSet), "PauseActionForPlayerChoice");

        // Verifier finding R-5: every one of these prefixes goes through the registry (one id per patched method, family
        // `thread-audit:*` in the ledger), so the site table sees them and nothing is patched outside PatchRegistry.
        const string reason = "Read-only thread-id recorder used to prove that gameplay code only runs on the engine thread. Installed only with --audit-threads.";
        Dictionary<string, int> names = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (MethodBase m in targets.Distinct())
        {
            string name = (m.DeclaringType?.Name ?? "?") + "." + m.Name;
            int k = names[name] = names.GetValueOrDefault(name) + 1;
            string id = "thread-audit:" + name + (k > 1 ? "#" + k : "");
            PatchRegistry.Prefix(id, m, prefix, PatchCategory.Instrumentation, reason);
            MethodInfo? moveNext = m is MethodInfo mi ? SafeAsyncMoveNext(mi) : null;
            if (moveNext != null)
            {
                PatchRegistry.Prefix(id + ":MoveNext", moveNext, prefix, PatchCategory.Instrumentation, reason);
            }
        }
    }

    private static MethodInfo? SafeAsyncMoveNext(MethodInfo m)
    {
        try
        {
            return AccessTools.AsyncMoveNext(m);
        }
        catch
        {
            return null;
        }
    }

    private static void AddAll(List<MethodBase> into, Type type, string name)
    {
        int before = into.Count;
        foreach (MethodInfo m in type.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
        {
            if (m.Name == name && !m.IsAbstract && !m.IsGenericMethodDefinition)
            {
                into.Add(m);
            }
        }
        if (into.Count == before)
        {
            throw new MissingMethodException(type.FullName, name);
        }
    }

    public static void Prefix(MethodBase __originalMethod)
    {
        int tid = Environment.CurrentManagedThreadId;
        if (__originalMethod.DeclaringType == typeof(MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionQueueSynchronizer)
            || __originalMethod.DeclaringType == typeof(MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionQueueSet))
        {
            _callsByMethod.AddOrUpdate(__originalMethod.DeclaringType!.Name + "." + __originalMethod.Name, 1, (_, c) => c + 1);
        }
        if (tid == _engineThreadId)
        {
            _onEngine++;
            return;
        }
        Interlocked.Increment(ref _offEngine);
        string name = (__originalMethod.DeclaringType?.Name ?? "?") + "." + __originalMethod.Name;
        _offByMethod.AddOrUpdate(name, 1, (_, c) => c + 1);
        _offByThread.AddOrUpdate(tid, 1, (_, c) => c + 1);
    }

    public static object Report()
    {
        return new
        {
            installed = Installed,
            engine_thread_id = _engineThreadId,
            calls_on_engine_thread = _onEngine,
            calls_off_engine_thread = Interlocked.Read(ref _offEngine),
            off_thread_by_method = _offByMethod.OrderByDescending(kv => kv.Value).Take(20).ToDictionary(kv => kv.Key, kv => kv.Value),
            off_thread_distinct_threads = _offByThread.Count,
            calls_by_method_choice_protocol = _callsByMethod.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value),
        };
    }
}
