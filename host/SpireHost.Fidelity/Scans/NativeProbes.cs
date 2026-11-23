using System;
using System.Collections.Generic;
using SpireHost.Boot;

namespace SpireHost.Fidelity.Scans;

/// <summary>
/// Ground truth for the static native scan: calls the GodotSharp entry points that gameplay code can reach
/// (found by StaticScans (b)) exactly as the game calls them, and records which NullGodot thunks moved. The
/// static closure over-approximates (it follows every call inside GodotSharp); these probes say what really
/// happens. Only GodotSharp is touched - no game state.
/// </summary>
internal static class NativeProbes
{
    public static object Run(HashSet<string> allow)
    {
        List<object> rows = new List<object>();
        void Probe(string name, string usedBy, Action a)
        {
            long[] before = NullGodot.SnapshotCounters();
            string? error = null;
            try
            {
                a();
            }
            catch (Exception e)
            {
                error = e.GetType().Name + ": " + e.Message;
            }
            SortedDictionary<string, long> delta = NullGodot.Delta(before);
            List<string> unexpected = new List<string>();
            foreach (string k in delta.Keys)
            {
                if (!allow.Contains(k))
                {
                    unexpected.Add(k);
                }
            }
            rows.Add(new { probe = name, used_by = usedBy, natives = delta, outside_allow_list = unexpected, error });
        }

        Probe("Callable.From(Action)", "GameAction.Cancel (GameAction.cs:157), CreatureCmd.cs:754, LocManager.cs:323", () => { Godot.Callable c = Godot.Callable.From(() => { }); _ = c.Delegate; });
        Probe("Callable.From(Action).CallDeferred() [host stub]", "GameAction.Cancel", () => { Godot.Callable.From(() => { }).CallDeferred(); });
        Probe("StringExtensions.Capitalize", "ActModel.DefeatedAllEnemiesAchievement / CharacterModel.RunWonAchievement (achievement ids, OnEnded with ShouldSave only)", () => { _ = Godot.StringExtensions.Capitalize("overgrowth_act"); });
        Probe("StringExtensions.ToSnakeCase", "CrystalSphereItem.TexturePath (UI)", () => { _ = Godot.StringExtensions.ToSnakeCase("CrystalSphereItem"); });
        Probe("GodotObject.IsInstanceValid(null)", "AssetCache / GodotTreeExtensions.QueueFreeSafely / FlavorSynchronizer", () => { _ = Godot.GodotObject.IsInstanceValid(null); });
        Probe("StringName(string) implicit", "lazily initialised StringName statics of game types", () => { Godot.StringName s = "fidelity_probe_name"; _ = s; });
        Probe("NodePath(string) implicit", "tween property paths (CardPileCmd, only with a live node)", () => { Godot.NodePath p = "position"; _ = p; });
        Probe("Mathf.Round / Vector2 math", "pure managed math used by gameplay (MerchantPotionEntry.CalcCost uses Mathf.Round)", () => { _ = Godot.Mathf.Round(52.4f); _ = new Godot.Vector2(1, 2).Length(); });
        Probe("Color(string)", "VFX colours in card / potion code (only with a live node)", () => { _ = new Godot.Color("6ec46f"); });
        GC.Collect();
        GC.WaitForPendingFinalizers();
        Probe("finalizers after the probes", "Dispose of Callable / StringName / NodePath created above", () => { GC.Collect(); GC.WaitForPendingFinalizers(); });
        return rows;
    }
}
