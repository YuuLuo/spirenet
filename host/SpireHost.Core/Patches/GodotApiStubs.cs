using System;
using System.Reflection;
using HarmonyLib;
using SpireHost.Engine;

namespace SpireHost.Patches;

/// <summary>
/// Managed replacements for the handful of *GodotSharp* (engine API, not game) static methods that game
/// code calls on TestMode paths. Without the engine their class initializers throw (method binds are
/// null), so the bodies are skipped entirely and a headless-correct value is returned.
/// These patch GodotSharp.dll, never sts2.dll.
/// </summary>
public static class GodotApiStubs
{
    public static void Install()
    {
        PatchRegistry.Prefix("godot-os-cmdline-args",
            AccessTools.Method(typeof(Godot.OS), nameof(Godot.OS.GetCmdlineArgs)),
            PatchRegistry.M(typeof(GodotApiStubs), nameof(EmptyStringArray)),
            PatchCategory.Plumbing,
            "Godot.OS.GetCmdlineArgs needs the engine. Returns an empty array: the host has no Godot command line (used by Logger..cctor, CommandLineHelper..cctor, TestMode.IsTestRunFromCmdline).");

        PatchRegistry.Prefix("godot-os-has-feature",
            AccessTools.Method(typeof(Godot.OS), nameof(Godot.OS.HasFeature)),
            PatchRegistry.M(typeof(GodotApiStubs), nameof(False)),
            PatchCategory.Plumbing,
            "Godot.OS.HasFeature needs the engine. Returns false (not an editor / demo / debug-feature build).");

        PatchRegistry.Prefix("godot-time-ticks-msec",
            AccessTools.Method(typeof(Godot.Time), nameof(Godot.Time.GetTicksMsec)),
            PatchRegistry.M(typeof(GodotApiStubs), nameof(TicksMsec)),
            PatchCategory.Plumbing,
            "Godot.Time.GetTicksMsec needs the engine. Returns the pump's virtual clock in ms. On the gameplay path it is only read by CardModel.OnPlayWrapper (CardModel.cs:1651/1755) to size a Cmd.Wait that is a no-op in TestMode.");

        MethodInfo? load = null;
        foreach (MethodInfo m in typeof(Godot.ResourceLoader).GetMethods(BindingFlags.Static | BindingFlags.Public))
        {
            ParameterInfo[] ps = m.GetParameters();
            if (m.Name == "Load" && !m.IsGenericMethodDefinition && m.ReturnType == typeof(Godot.Resource) && ps.Length == 3
                && ps[0].ParameterType == typeof(string) && ps[1].ParameterType == typeof(string) && ps[2].ParameterType == typeof(Godot.ResourceLoader.CacheMode))
            {
                load = m;
            }
        }
        PatchRegistry.Prefix("godot-resource-loader-load", load,
            PatchRegistry.M(typeof(GodotApiStubs), nameof(NullResource)),
            PatchCategory.Plumbing,
            "Godot.ResourceLoader.Load needs the engine. Returns null: no texture/scene/audio resource exists headlessly. " +
            "Reached in TestMode only by presentation code (e.g. DecimillipedeSegment.ReattachMove swapping a phobia-mode texture, which then finds no creature node and does nothing).");

        PatchRegistry.Prefix("godot-resource-loader-exists",
            AccessTools.Method(typeof(Godot.ResourceLoader), nameof(Godot.ResourceLoader.Exists), new[] { typeof(string), typeof(string) }),
            PatchRegistry.M(typeof(GodotApiStubs), nameof(False)),
            PatchCategory.Plumbing,
            "Godot.ResourceLoader.Exists needs the engine. Returns false: no res:// asset exists headlessly. First reached by icon lookups while building hover tips " +
            "(EnchantmentModel.IconPath <- HoverTipFactory <- RelicModel.HoverTipsExcludingRelic <- EventOption.FromRelic in Neow.GenerateInitialOptions). Hover tips are UI data only.");

        PatchRegistry.Prefix("godot-file-access-file-exists",
            AccessTools.Method(typeof(Godot.FileAccess), nameof(Godot.FileAccess.FileExists), new[] { typeof(string) }),
            PatchRegistry.M(typeof(GodotApiStubs), nameof(False)),
            PatchCategory.Plumbing,
            "Godot.FileAccess.FileExists needs the engine (its class initializer throws NativeMethodBindNotFoundException headlessly). Returns false: no res:// file exists. " +
            "Reached by AtlasManager.BuildAtlasData (AtlasManager.cs:70-77, which then logs a warning and returns null) while game logic builds hover tips, e.g. " +
            "StunIntent.GetStaticHoverTip <- Whistle.ExtraHoverTips <- TanxsWhistle <- EventOption.FromRelic in Tanx.GenerateInitialOptions. Hover-tip icons are UI data only. Found by the H1 run fuzz.");

        PatchRegistry.Prefix("godot-callable-call-deferred",
            AccessTools.Method(typeof(Godot.Callable), nameof(Godot.Callable.CallDeferred)),
            PatchRegistry.M(typeof(GodotApiStubs), nameof(CallDeferredPrefix)),
            PatchCategory.Scheduling,
            "Godot.Callable.CallDeferred means: run on the main thread at the next idle point. The engine is absent, so a managed delegate without parameters is posted to the " +
            "engine pump queue instead (FIFO, engine thread), which gives the same ordering guarantee: an Action, or - phase H5 - a Func<T> whose result Godot discards " +
            "(Callable.From(() => expr) binds to From<T>(Func<T>) when the lambda has a value). Used by LocManager (locale callbacks, Action) and by GameAction.Cancel " +
            "(GameActions/GameAction.cs:157: `() => _completionSource.TrySetCanceled()`, a Func<bool> that completes CompletionTask as cancelled; reached when a combat ends " +
            "while combat actions of other players are still queued, ActionQueueSet.CombatEnded :327-356 - never seen with one seat, where it would have been counted as dropped). " +
            "Any other callable (a node method: no node exists headlessly) is counted as dropped and fails the episode in strict mode. Without this the deferred call would silently never run.");
    }

    public static long DeferredCalls;
    public static long DeferredCallsDropped;

    /// <summary>Deferred calls of a parameterless delegate that returns a value (Func&lt;T&gt;), part of <see cref="DeferredCalls"/>.</summary>
    public static long DeferredFuncCalls;

    /// <summary>Diagnostics: the delegate / target and the first game frames of the last dropped deferred call.</summary>
    public static string? LastDropped;

    public static bool NullResource(ref Godot.Resource? __result)
    {
        __result = null;
        return false;
    }

    public static bool CallDeferredPrefix(ref Godot.Callable __instance)
    {
        if (__instance.Delegate is Action action)
        {
            DeferredCalls++;
            EnginePump? pump = EnginePump.Current;
            if (pump != null)
            {
                pump.Post(static a => ((Action)a!)(), action);
            }
            else
            {
                action(); // no pump (comparison mode / boot): there is no later idle point, run now
            }
        }
        else if (__instance.Delegate is Delegate func && func.Method.GetParameters().Length == 0)
        {
            DeferredCalls++;
            DeferredFuncCalls++;
            EnginePump? pump = EnginePump.Current;
            if (pump != null)
            {
                pump.Post(static d => ((Delegate)d!).DynamicInvoke(), func);
            }
            else
            {
                func.DynamicInvoke(); // no pump (comparison mode / boot): there is no later idle point, run now
            }
        }
        else
        {
            DeferredCallsDropped++; // node-method callables: no node exists headlessly
            try
            {
                System.Text.StringBuilder sb = new System.Text.StringBuilder(256);
                sb.Append(__instance.Delegate?.Method.DeclaringType?.FullName ?? "(no delegate)").Append("::").Append(__instance.Delegate?.Method.Name ?? "-");
                int frames = 0;
                foreach (string line in Environment.StackTrace.Split('\n'))
                {
                    string f = line.Trim();
                    if (f.StartsWith("at MegaCrit.", StringComparison.Ordinal) && frames < 5)
                    {
                        int cut = f.IndexOf(" in ", StringComparison.Ordinal);
                        sb.Append(" <- ").Append(cut > 0 ? f.Substring(3, cut - 3) : f.Substring(3));
                        frames++;
                    }
                }
                LastDropped = sb.ToString();
            }
            catch (Exception)
            {
                // diagnostics only
            }
        }
        return false;
    }

    public static bool EmptyStringArray(ref string[] __result)
    {
        __result = Array.Empty<string>();
        return false;
    }

    public static bool False(ref bool __result)
    {
        __result = false;
        return false;
    }

    public static bool TicksMsec(ref ulong __result)
    {
        __result = EnginePump.Current?.Clock.NowMs ?? 0UL;
        return false;
    }
}
