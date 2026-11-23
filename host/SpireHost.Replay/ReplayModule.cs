namespace SpireHost.Replay;

/// <summary>
/// The real-client oracle tools (--replay-mcr, --compare-run-traces, replay --replay-determinism, replay --selftest;
/// <see cref="ReplayCommands"/>). They are verification commands of the executable, NOT a driver module: nothing here is
/// registered in the driver registry and nothing is part of the production patch set. Every tool runs on the full
/// production configuration (ruling B4 of the H1 integration) plus only its own read-only instrumentation, which is
/// registered per command and listed in the tool-only section of host\FIDELITY.md:
///   * replay-checksum-tracker-enabled / replay-post-action-checksum / replay-combat-state-sync-disabled
///     (only inside the replay run the tool sets up; ReplayInterventions),
///   * replay-coverage:* (--replay-mcr --coverage; CoverageProbe).
/// There are no prototype fixes any more: what they prototyped is in the production fidelity pack.
/// </summary>
public static class ReplayModule
{
    public const string Name = "replay";
}
