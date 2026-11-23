using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.TestSupport;
using SpireHost.Patches;

namespace SpireHost.Boot;

/// <summary>
/// The boot has ONE configuration (H1 integration: one production patch set, defined in one place). The only
/// knob left is diagnostic: how much of the game's log is copied to stderr.
/// </summary>
public sealed class BootOptions
{
    /// <summary>Game log level routed to stderr: "off" (errors only), "warn", "info", "debug". Observational only.</summary>
    public string GameLog = "off";
}

public sealed class BootReport
{
    public double TotalMs;
    public List<KeyValuePair<string, double>> Stages = new List<KeyValuePair<string, double>>();
    public long[] NativeCountersAfterBoot = Array.Empty<long>();
}

/// <summary>
/// Boots the unmodified sts2.dll headlessly, following the game's own OneTimeInitialization order
/// (ExecuteVeryEarly / ExecuteEssential) with TestMode on and the Godot-facing steps replaced.
/// </summary>
public static class GameBoot
{
    private static bool _done;

    public static BootReport Report { get; private set; } = new BootReport();

    public static BootReport Run(BootOptions options)
    {
        if (_done)
        {
            return Report;
        }
        _done = true;
        BootReport report = new BootReport();
        Stopwatch total = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();
        void Stage(string name)
        {
            report.Stages.Add(new KeyValuePair<string, double>(name, sw.Elapsed.TotalMilliseconds));
            sw.Restart();
        }

        TaskScheduler.UnobservedTaskException += (_, e) =>
        {
            // Raised on the finalizer thread at an arbitrary later GC. Faults that already went through
            // TaskHelper.UnobservedFault were attributed to their episode; anything else is counted as "late".
            HostFaults.RecordLate(e.Exception.GetBaseException());
            e.SetObserved();
        };

        NullGodot.Install();
        Stage("null_godot");

        GodotApiStubs.Install();
        HeadlessPatches.InstallEarly();
        NetworkGuard.Install();                 // the host performs no network IO (telemetry upload / time server): fail-closed, see NetworkGuard
        Stage("harmony_patches");

        // --- OneTimeInitialization.ExecuteVeryEarly (TestMode branch) ---
        TestMode.TurnOnInternal();
        PatchRegistry.Note("test-mode", "MegaCrit.Sts2.Core.TestSupport.TestMode::TurnOnInternal", "game API call", PatchCategory.Plumbing,
            "The game's own headless switch: node creation, VFX, audio and Cmd.Wait are skipped by the game itself. Rule-affecting TestMode branches are tracked in FIDELITY.md section B.");
        SetModManagerSkipped();
        LogBootstrap.Install(options.GameLog);
        Stage("testmode_log");

        SaveBootstrap.Install();
        Stage("save_manager");

        // --- OneTimeInitialization.ExecuteEssential (minus atlas / resource loader) ---
        LocalizationBootstrap.Install();
        Stage("localization");
        AssemblyInfo.Init();
        ModelDb.Init();
        ModelIdSerializationCache.Init();
        ModelDb.InitIds();
        MessageTypes.Initialize();
        ActionTypes.Initialize();
        Stage("model_db");

        AssertRealGameAssembly();
        VeteranProfile.Install();
        Stage("profile");

        HeadlessPatches.InstallLate();
        SchedulingPatches.Install();
        FidelityGatePatches.Install();
        LivelockGuard.Install();
        Stage("late_patches");

        report.TotalMs = total.Elapsed.TotalMilliseconds;
        report.NativeCountersAfterBoot = NullGodot.SnapshotCounters();
        Report = report;
        return report;
    }

    /// <summary>
    /// The publicized reference (host\publicized\sts2.dll) is for the compiler only. If it ever got loaded
    /// instead of the SHA-pinned host\game\sts2.dll the fidelity claim would be void, so refuse to continue.
    /// </summary>
    private static void AssertRealGameAssembly()
    {
        string loaded = Path.GetFullPath(typeof(ModelDb).Assembly.Location);
        string expected = Path.GetFullPath(Path.Combine(GameAssemblies.GameDir, "sts2.dll"));
        if (!string.Equals(loaded, expected, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("sts2 was loaded from " + loaded + ", expected the pinned " + expected);
        }
    }

    private static void SetModManagerSkipped()
    {
        // ModManager.Initialize (file IO + Godot) is never run; the state the game itself uses for "mods
        // skipped" is set so that ModManager.IsRunningModded()/GetLoadedMods() take their vanilla branch.
        PropertyInfo state = typeof(ModManager).GetProperty("State", BindingFlags.Static | BindingFlags.Public)
            ?? throw new MissingMemberException("ModManager.State");
        MethodInfo setter = state.GetSetMethod(nonPublic: true) ?? throw new MissingMethodException("ModManager.State setter");
        setter.Invoke(null, new object[] { Enum.Parse(state.PropertyType, "Skipped") });
        PatchRegistry.Note("mod-manager-skipped", "MegaCrit.Sts2.Core.Modding.ModManager::State", "private setter call", PatchCategory.Plumbing,
            "No mod loading in the host; same state the game sets when mods are skipped.");
    }
}

/// <summary>Host diagnostics on stderr. stdout is reserved for results / protocol.</summary>
public static class HostLog
{
    public static bool Verbose;

    public static void Error(string s) => Console.Error.WriteLine("[host/error] " + s);

    public static void Info(string s)
    {
        if (Verbose)
        {
            Console.Error.WriteLine("[host] " + s);
        }
    }
}

/// <summary>Faults raised by game tasks that nobody awaited (TaskHelper.RunSafely / unobserved). Never dropped silently.</summary>
public static class HostFaults
{
    private static readonly List<Exception> _faults = new List<Exception>();
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Exception, object> _seen = new System.Runtime.CompilerServices.ConditionalWeakTable<Exception, object>();
    private static int _count;

    /// <summary>Unobserved task exceptions that never passed through TaskHelper.RunSafely (cannot be attributed to an episode).</summary>
    public static int LateUnobserved;
    public static string? LastLateUnobserved;

    public static int Count => System.Threading.Volatile.Read(ref _count);

    public static void Record(Exception e)
    {
        lock (_faults)
        {
            _seen.TryAdd(e.GetBaseException(), _faults);
            _faults.Add(e);
            _count = _faults.Count;
        }
    }

    public static void RecordLate(Exception e)
    {
        lock (_faults)
        {
            if (_seen.TryGetValue(e, out _))
            {
                return;
            }
            _seen.TryAdd(e, _faults);
            LateUnobserved++;
            LastLateUnobserved = e.ToString();
        }
        HostLog.Error("LATE_UNOBSERVED_TASK_EXCEPTION " + e);
    }

    public static List<Exception> Drain()
    {
        lock (_faults)
        {
            List<Exception> r = new List<Exception>(_faults);
            _faults.Clear();
            _count = 0;
            return r;
        }
    }
}
