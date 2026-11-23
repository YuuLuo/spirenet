using System.Collections.Generic;
using SpireHost.Boot;
using SpireHost.Patches;

namespace SpireHost.Driver;

/// <summary>
/// The counters of the Core / Driver patches as a module-counter source ("core.*"). Registered by HostSession in
/// every mode. (The per-episode decision statistics "driver.*" are written by the run driver itself.)
/// </summary>
internal sealed class CoreCounters : IModuleCounters
{
    public string Module => "core";

    public void Read(IDictionary<string, long> c)
    {
        // run-save-completes-after-pump-turn (ruling E2; E1: how often other work was queued at the room-entry save)
        c["run_save_boundary.combat_end"] = SchedulingPatches.RunSaveBoundariesCombatEnd;
        c["run_save_boundary.room_entry"] = SchedulingPatches.RunSaveBoundariesRoomEntry;
        c["run_save_boundary.event_room"] = SchedulingPatches.RunSaveBoundariesEventRoom;
        c["run_save_boundary.queued_work.combat_end"] = SchedulingPatches.RunSaveQueuedWorkCombatEnd;
        c["run_save_boundary.queued_work.room_entry"] = SchedulingPatches.RunSaveQueuedWorkRoomEntry;
        c["run_save_boundary.queued_work.event_room"] = SchedulingPatches.RunSaveQueuedWorkEventRoom;
        c["run_save_boundary.skipped"] = SchedulingPatches.RunSaveBoundariesSkipped;
        // hook-action-started-inline-continuation / virtual clock
        c["inline_hook_continuations"] = SchedulingPatches.InlineContinuations;
        c["virtual_delays"] = SchedulingPatches.VirtualDelays;
        c["deferred_calls"] = GodotApiStubs.DeferredCalls;
        // progress-save-intercept / network guard
        c["intercepted_progress_writes"] = VeteranProfile.InterceptedProgressWrites;
        c["metrics_upload_checks"] = NetworkGuard.MetricsUploadChecks;
        c["network_attempts"] = System.Threading.Interlocked.Read(ref NetworkGuard.NetworkAttempts);
        // fidelity gates of the core
        c["discovery_order_calls"] = FidelityGatePatches.DiscoveryOrderCalls;
        // driver patches
        c["take_turn_calls"] = DriverPatches.TakeTurnCalls;
        c["proceed_opened_map"] = DriverPatches.ProceedOpenedMap;
        c["proceed_ignored"] = DriverPatches.ProceedIgnored;
        c["game_errors_logged"] = HeadlessPatches.GameErrors;
    }
}
