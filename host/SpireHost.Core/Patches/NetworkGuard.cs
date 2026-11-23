using System;
using System.Reflection;
using System.Threading.Tasks;
using HarmonyLib;

namespace SpireHost.Patches;

/// <summary>
/// The host performs NO network IO. sts2.dll contains two outbound HTTP clients outside UI nodes (a third one,
/// NSendFeedbackScreen, is a UI node that is never instantiated headlessly). The first is reachable from game logic
/// (see below); the second only through the Daily-run flow, which the host never starts - it is blocked all the same.
/// One more piece of game code touches a network library before this guard exists: the Sentry.Godot module initializer
/// of sts2.dll (SentryAutoInit) runs when the assembly is first touched. It returns at once because the host process
/// has no native Sentry export (it prints "Sentry GDExtension not loaded in this process; skipping" to stderr on every
/// start); SentryService.Initialize is only called by a Godot node and never runs (verifier finding R-11).
///
///   MetricUtilities.PutRequest      (Runs.Metrics/MetricUtilities.cs:303-326)  HttpClient.PutAsync to
///                                   https://sts2-metric-upload.megacrit.com/record_{data,achievement,epoch,settings}/
///   TimeServer.RequestTime          (Daily/TimeServer.cs:21-63)                 HttpClient.GetAsync https://time.megacrit.com
///
/// Why this is not hypothetical: the host's release-info-file patch (HeadlessPatches) hands the game its real
/// release_info.json, so MetricUtilities.ShouldUploadMetrics (:59-87) no longer stops at "this is a debug build",
/// and PrefsSave.UploadData defaults to true. RunManager.OnEnded only uploads inside `if (ShouldSave)`
/// (Runs/RunManager.cs:1530-1544); episodes are set up with shouldSave: false, but RunManager.SetUpReplay
/// hard-codes shouldSave: true (:343), so with fidelity-natural-death-on-ended a replayed death of a normal run
/// (no debug command, >= 5 map points) reached the upload. Found by the H1 integration baseline.
///
/// Three layers, all fail-closed:
///   1. metrics-upload-disabled       ShouldUploadMetrics -> false: the game's own "user has upload data unset"
///                                    branch (:66-70); nothing is built, no locale override, nothing is sent.
///   2. metrics-put-request-blocked   PutRequest never runs; a call is counted. NetworkAttempts must stay 0:
///   3. time-server-request-blocked   it is a strict counter (episode aborted) and fails a replay.
/// </summary>
public static class NetworkGuard
{
    /// <summary>Calls of a blocked network entry point. Must stay 0 (strict counter; any value above 0 is a host fault).</summary>
    public static long NetworkAttempts;

    /// <summary>Last blocked attempt ("PutRequest record_data ..."), for diagnostics. Never contains the payload.</summary>
    public static string? LastAttempt;

    /// <summary>ShouldUploadMetrics calls answered with false (informational: how often the game asked).</summary>
    public static long MetricsUploadChecks;

    public static bool Installed { get; private set; }

    public static void Install()
    {
        if (Installed)
        {
            return;
        }
        Installed = true;
        Assembly game = typeof(MegaCrit.Sts2.Core.Runs.RunManager).Assembly;
        Type metrics = game.GetType("MegaCrit.Sts2.Core.Runs.Metrics.MetricUtilities", throwOnError: true)!;

        PatchRegistry.Prefix("metrics-upload-disabled",
            AccessTools.Method(metrics, "ShouldUploadMetrics", new[] { typeof(bool) }),
            PatchRegistry.M(typeof(NetworkGuard), nameof(ShouldUploadPrefix)),
            PatchCategory.Plumbing,
            "MetricUtilities.ShouldUploadMetrics (Runs.Metrics/MetricUtilities.cs:59-87) returns false: the host never uploads telemetry. Same result as the game's own " +
            "`!PrefsSave.UploadData` branch (:66-70), so UploadRunMetricsInternal / UploadAchievementMetric / UploadEpochMetric / UploadSettingsMetric return before building anything " +
            "(no LocManager.StartOverridingLanguageAsEnglish, no HTTP). Without it the host's own release-info-file patch makes the first check (`ReleaseInfo == null`) pass. " +
            "No game state, no RNG. Calls are counted (informational).");

        PatchRegistry.Prefix("metrics-put-request-blocked",
            AccessTools.Method(metrics, "PutRequest", new[] { typeof(string), typeof(string), typeof(string) }),
            PatchRegistry.M(typeof(NetworkGuard), nameof(PutRequestPrefix)),
            PatchCategory.Plumbing,
            "MetricUtilities.PutRequest (Runs.Metrics/MetricUtilities.cs:303-326) is the only sender of the four metric endpoints (HttpClient.PutAsync). The body never runs: the prefix returns " +
            "Task.CompletedTask and counts the call. Backstop of metrics-upload-disabled: the counter is a strict counter (network_attempt aborts the episode; a replay is failed), so it must stay 0.");

        Type timeServer = game.GetType("MegaCrit.Sts2.Core.Daily.TimeServer", throwOnError: true)!;
        MethodInfo request = AccessTools.Method(timeServer, "RequestTime", new[] { typeof(string) })
            ?? throw new MissingMethodException("TimeServer.RequestTime(string)");
        Type result = request.ReturnType.GetGenericArguments()[0]; // Nullable<TimeServerResult>
        PatchRegistry.Prefix("time-server-request-blocked", request,
            PatchRegistry.M(typeof(NetworkGuard), nameof(TimeRequestPrefix)).MakeGenericMethod(result),
            PatchCategory.Plumbing,
            "TimeServer.RequestTime (Daily/TimeServer.cs:21-63) GETs https://time.megacrit.com for daily runs (DailyRunUtility.cs:107-129; never reached: the host starts Standard runs, " +
            "DailyTime is null). The body never runs: the prefix returns a completed task with a null result (the game's own 'gave up' value) and counts the call (strict counter, must stay 0).");
    }

    public static bool ShouldUploadPrefix(ref bool __result)
    {
        MetricsUploadChecks++;
        __result = false;
        return false;
    }

    public static bool PutRequestPrefix(string url, string context, ref Task __result)
    {
        System.Threading.Interlocked.Increment(ref NetworkAttempts);
        LastAttempt = "MetricUtilities.PutRequest " + Tail(url) + " (" + context + ")";
        __result = Task.CompletedTask;
        return false;
    }

    public static bool TimeRequestPrefix<T>(string url, ref Task<T> __result)
    {
        System.Threading.Interlocked.Increment(ref NetworkAttempts);
        LastAttempt = "TimeServer.RequestTime " + Tail(url);
        __result = Task.FromResult(default(T)!);
        return false;
    }

    /// <summary>Last path segment of a URL (enough to tell which endpoint; the host name is in the ledger).</summary>
    private static string Tail(string url)
    {
        string s = url.TrimEnd('/');
        int i = s.LastIndexOf('/');
        return i >= 0 ? s.Substring(i + 1) : s;
    }
}
