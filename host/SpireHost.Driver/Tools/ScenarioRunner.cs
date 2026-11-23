using System;
using System.Threading.Tasks;

namespace SpireHost.Driver.Tools;

/// <summary>
/// Runs one isolated scenario episode: real run start, then <c>setup</c> (loadout + debug jump, see
/// ScenarioTools), then the normal driver loop with whatever handlers are registered, until
/// <c>stopWhen</c> holds (default: the room was left for the map), the run ends, or the decision budget is
/// spent. This is the building block of every team's handler tests (DRIVER-API.md section 9).
/// </summary>
public static class ScenarioRunner
{
    public static EpisodeResult Run(
        HostSession session,
        string name,
        Func<RunContext, Task> setup,
        string character = "IRONCLAD",
        int ascension = 0,
        string? seed = null,
        IPolicy? policy = null,
        Func<RunContext, bool>? stopWhen = null,
        EpisodeOptions? options = null,
        IEpisodeObserver? observer = null,
        string[]? exempt = null)
    {
        EpisodeOptions o = options ?? new EpisodeOptions();
        if (o.MaxPolicyDecisions == 0)
        {
            o.MaxPolicyDecisions = 20_000; // a scenario that does not stop is a failed test, not a hang
        }
        EpisodeSpec spec = new EpisodeSpec
        {
            Seed = seed ?? HostSession.SeedFromLabel("SCENARIO:" + name),
            Characters = new[] { character },
            Ascension = ascension,
            Scenario = new Scenario
            {
                Name = name,
                Setup = setup,
                StopWhen = stopWhen ?? ScenarioTools.RoomFinished,
                Exempt = exempt == null ? null : new System.Collections.Generic.HashSet<string>(exempt, StringComparer.Ordinal),
            },
        };
        return session.RunEpisode(spec, policy ?? new RandomLegalPolicy(1), o, observer);
    }

    /// <summary>A scenario passed when it stopped on its stop condition (or ended in a real terminal state) without an abort.</summary>
    public static bool Passed(EpisodeResult r)
    {
        return r.Outcome != EpisodeOutcome.Aborted && r.TeardownFaults == 0 && (r.Outcome != EpisodeOutcome.Truncated || r.TruncatedBy == "scenario");
    }

    public static object Summarize(string name, EpisodeResult r)
    {
        return new
        {
            scenario = name,
            passed = Passed(r),
            outcome = r.Outcome.ToString(),
            truncated_by = r.TruncatedBy,
            abort = r.Abort?.Signature,
            abort_message = r.Abort?.Message,
            abort_frames = r.Abort?.Frames,
            diagnostics = r.Abort?.Diagnostics,
            seed = r.Seed,
            character = r.Character,
            final_room = r.FinalRoom,
            final_hp = r.FinalHp,
            policy_decisions = r.Counters.PolicyDecisions,
            decision_points = r.Counters.DecisionPoints,
            auto_single = r.Counters.AutoSingleTotal,
            auto_gold = r.Counters.AutoGold,
            nested = r.Counters.NestedDecisions,
            strict_violation = r.Strict.FirstViolation(),
            unexpected_native_calls = r.UnexpectedNativeCalls,
            wall_ms = Math.Round(r.WallMs, 1),
        };
    }
}
