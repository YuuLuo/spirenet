using System;
using System.Collections;
using System.Reflection;
using System.Text;
using HarmonyLib;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace SpireHost.Patches;

/// <summary>
/// DECISIONS-H1 D.5: the tutorial / discovery-order branches of run start are switched to their REAL-game
/// branch even though they are no-ops for the veteran profile, so that a wrong profile shows up as a visible
/// fault instead of being masked by TestMode. Each one comes with a "did it fire?" counter that the run
/// driver checks at every quiescent point (non-zero = episode aborted with reason profile_gate_fired).
///
///   ActModel.GetRandomList               (ActModel.cs:465)    TestMode.IsOff -> true
///   RunManager.TryGetRoomTypeForTutorial (RunManager.cs:883)  TestMode.IsOn  -> false
///   RunManager.ShouldApplyTutorialModifications (:644-663): no patch - the driver sets the game's own
///       public ForceDiscoveryOrderModifications to the value of the real predicate before SetUpNewSingleplayer.
///
/// The remaining behaviour-affecting TestMode gates belong to the fidelity module (SpireHost.Fidelity).
/// </summary>
public static class FidelityGatePatches
{
    /// <summary>TryGetRoomTypeForTutorial returned true (would force the first '?' to be an event): must stay 0.</summary>
    public static long TutorialRoomForced;

    /// <summary>ApplyDiscoveryOrderModifications changed an act's boss: must stay 0 for the veteran profile.</summary>
    public static long DiscoveryBossChanged;

    /// <summary>
    /// ApplyDiscoveryOrderModifications changed the act's ROOM SET (events, normal / elite encounters in order, boss,
    /// second boss, ancient): must stay 0 for the veteran profile. (Fidelity request R2: the boss alone does not show the
    /// first-run reordering of Overgrowth, Models.Acts/Overgrowth.cs:110-127.)
    /// </summary>
    public static long DiscoveryRoomsChanged;

    /// <summary>What changed at the last fired room-set check (diagnostics).</summary>
    public static string? LastDiscoveryChange;

    /// <summary>ApplyDiscoveryOrderModifications calls (proves the real branch runs: one per act per run).</summary>
    public static long DiscoveryOrderCalls;

    public static bool Installed { get; private set; }

    private static FieldInfo _rooms = null!;
    private static Func<object, object?>[] _lists = Array.Empty<Func<object, object?>>();
    private static Func<object, object?>[] _singles = Array.Empty<Func<object, object?>>();

    public static void Install()
    {
        if (Installed)
        {
            return;
        }
        Installed = true;

        GatePatch.ForceGetter("fidelity-act-selection-discovery-gate",
            AccessTools.Method(typeof(ActModel), nameof(ActModel.GetRandomList)),
            GatePatch.TestModeIsOff, true, 1, PatchCategory.Fidelity,
            "ActModel.GetRandomList (ActModel.cs:465): the real game forces a non-default act that is unlocked but not yet in Progress.DiscoveredActs (no RNG draw); TestMode skips that. " +
            "Real branch restored. No-op for the veteran profile (every act discovered; asserted at boot by VeteranProfile.AssertInvariants).");

        MethodInfo tutorial = AccessTools.Method(typeof(RunManager), "TryGetRoomTypeForTutorial")
            ?? throw new MissingMethodException("RunManager.TryGetRoomTypeForTutorial");
        GatePatch.ForceGetter("fidelity-tutorial-room-type-gate", tutorial,
            GatePatch.TestModeIsOn, false, 1, PatchCategory.Fidelity,
            "RunManager.TryGetRoomTypeForTutorial (RunManager.cs:880-907): in the real game the first Unassigned map point of a profile with Progress.NumberOfRuns == 0 becomes an Event; TestMode never does that. " +
            "Real branch restored. No-op for the veteran profile (NumberOfRuns > 0). SHARED SITE (ruling B3): fidelity-tutorial-room-type-noop-check counts any 'true' result of the same method.");
        PatchRegistry.Postfix("fidelity-tutorial-room-type-noop-check", tutorial,
            PatchRegistry.M(typeof(FidelityGatePatches), nameof(TutorialPostfix)), PatchCategory.Instrumentation,
            "Read-only: counts TryGetRoomTypeForTutorial results that are true. The run driver aborts the episode (profile_gate_fired) if the counter moved. " +
            "SHARED SITE (ruling B3): the gate fidelity-tutorial-room-type-gate is a transpiler on the same method; this postfix only reads its result.");
        PatchRegistry.DeclareSharedSite("RunManager.TryGetRoomTypeForTutorial: the real-branch gate (transpiler) and the read-only counter of its result (postfix) that proves the gate is a no-op for the veteran profile.",
            "fidelity-tutorial-room-type-gate", "fidelity-tutorial-room-type-noop-check");

        MethodInfo discovery = AccessTools.Method(typeof(ActModel), nameof(ActModel.ApplyDiscoveryOrderModifications))
            ?? throw new MissingMethodException("ActModel.ApplyDiscoveryOrderModifications");
        BindRoomSet();
        PatchRegistry.PrefixPostfix("fidelity-discovery-order-noop-check", discovery,
            PatchRegistry.M(typeof(FidelityGatePatches), nameof(DiscoveryPrefix)),
            PatchRegistry.M(typeof(FidelityGatePatches), nameof(DiscoveryPostfix)), PatchCategory.Instrumentation,
            "Read-only: fingerprints the act's boss AND its whole room set (events, normal / elite encounters in order, boss, second boss, ancient) before and after " +
            "ActModel.ApplyDiscoveryOrderModifications (ActModel.cs:332-343 -> ApplyActDiscoveryOrderModifications, e.g. Overgrowth.cs:110-127: fixed first-run order of 7 normal encounters, " +
            "2 events, 2 elites when NumberOfRuns == 0), which the driver enables with the game's own RunManager.ForceDiscoveryOrderModifications (RunManager.cs:644-663). " +
            "A change means the profile is not the veteran profile; counted (DiscoveryBossChanged / DiscoveryRoomsChanged) and fatal for the episode (profile_gate_fired).");
        PatchRegistry.Note("fidelity-force-discovery-order", "MegaCrit.Sts2.Core.Runs.RunManager::ForceDiscoveryOrderModifications (public setter)", "game API call", PatchCategory.Fidelity,
            "Set before SetUpNewSingleplayer to `GameMode == Standard && !IsCustomGame`, the value of the real predicate in RunManager.ShouldApplyTutorialModifications (:644-663), so that the real-game path " +
            "(ApplyDiscoveryOrderModifications) runs exactly when the real client runs it. No-op for the veteran profile: every boss in every BossDiscoveryOrder has been seen and NumberOfRuns > 0.");
    }

    public static void TutorialPostfix(bool __result)
    {
        if (__result)
        {
            TutorialRoomForced++;
        }
    }

    public sealed class DiscoveryState
    {
        public ModelId? Boss;
        public string Rooms = "";
    }

    public static void DiscoveryPrefix(ActModel __instance, out DiscoveryState __state)
    {
        DiscoveryOrderCalls++;
        __state = new DiscoveryState { Boss = __instance.BossEncounter?.Id, Rooms = RoomsFingerprint(__instance) };
    }

    public static void DiscoveryPostfix(ActModel __instance, DiscoveryState __state)
    {
        if (__instance.BossEncounter?.Id != __state.Boss)
        {
            DiscoveryBossChanged++;
        }
        string after = RoomsFingerprint(__instance);
        if (after != __state.Rooms)
        {
            DiscoveryRoomsChanged++;
            LastDiscoveryChange = __instance.Id.Entry + ": before " + __state.Rooms + " after " + after;
        }
    }

    // -----------------------------------------------------------------------------------------------------
    //  Room-set fingerprint. The members are not public and this assembly is compiled against the real
    //  sts2.dll, so they are bound once by reflection (fail-closed: a missing member throws at boot).
    // -----------------------------------------------------------------------------------------------------

    private static void BindRoomSet()
    {
        _rooms = AccessTools.Field(typeof(ActModel), "_rooms") ?? throw new MissingFieldException("ActModel._rooms");
        Type set = _rooms.FieldType;
        _lists = new[] { Getter(set, "events"), Getter(set, "normalEncounters"), Getter(set, "eliteEncounters") };
        _singles = new[] { Getter(set, "_boss"), Getter(set, "SecondBoss"), Getter(set, "_ancient") };
    }

    private static Func<object, object?> Getter(Type type, string name)
    {
        FieldInfo? f = AccessTools.Field(type, name);
        if (f != null)
        {
            return o => f.GetValue(o);
        }
        PropertyInfo? p = AccessTools.Property(type, name);
        if (p != null && p.GetMethod != null)
        {
            return o => p.GetValue(o);
        }
        throw new MissingMemberException(type.FullName, name);
    }

    /// <summary>"E:..|N:..|L:..|B:..|S:..|A:.." over the act's RoomSet; "-" before the rooms were generated.</summary>
    public static string RoomsFingerprint(ActModel act)
    {
        object? rooms = _rooms.GetValue(act);
        if (rooms == null)
        {
            return "-";
        }
        StringBuilder sb = new StringBuilder(256);
        string[] tags = { "E:", "|N:", "|L:" };
        for (int i = 0; i < _lists.Length; i++)
        {
            sb.Append(tags[i]);
            if (_lists[i](rooms) is IEnumerable list)
            {
                foreach (object? m in list)
                {
                    sb.Append((m as AbstractModel)?.Id.Entry).Append(',');
                }
            }
        }
        sb.Append("|B:").Append((_singles[0](rooms) as AbstractModel)?.Id.Entry);
        sb.Append("|S:").Append((_singles[1](rooms) as AbstractModel)?.Id.Entry);
        sb.Append("|A:").Append((_singles[2](rooms) as AbstractModel)?.Id.Entry);
        return sb.ToString();
    }
}
