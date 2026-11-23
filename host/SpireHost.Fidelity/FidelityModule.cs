using System;
using SpireHost.Driver;

namespace SpireHost.Fidelity;

/// <summary>
/// The fidelity patch pack as one IDriverModule (DRIVER-API.md section 12 "fidelity").
///
/// Register (engine thread, after boot, before the first episode):
///   1. installs every patch of <see cref="Patches.FidelityPatches"/> through PatchRegistry / GatePatch
///      (fail-closed; one row per id in host\FIDELITY.md);
///   2. registers "fidelity:invariants" as a lifecycle (per-run assertions over the game's static test hooks, the
///      discovery-order flag and the run mode) and as an episode check (driver slot IEpisodeCheck: the swallowed-
///      assertion backstop and the static-hook invariants, at every quiescent point and before teardown);
///   3. registers the module counters "fidelity.*".
/// The constructor must not touch a game type (the module is created before the resolver can load sts2).
/// </summary>
public sealed class FidelityModule : IDriverModule
{
    public string Name => "fidelity";

    public void Register(DriverRegistry registry)
    {
        Patches.FidelityPatches.Install();
        FidelityInvariants inv = new FidelityInvariants();
        registry.SetLifecycle(inv);
        registry.SetEpisodeCheck(inv);
        registry.SetModuleCounters(new FidelityCounters());
    }
}
