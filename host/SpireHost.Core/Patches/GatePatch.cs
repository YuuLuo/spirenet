using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using HarmonyLib;

namespace SpireHost.Patches;

/// <summary>
/// Fail-closed helper for the commonest fidelity patch: "this method contains exactly N calls of a static
/// bool getter (TestMode.IsOn / TestMode.IsOff / NonInteractiveMode.IsActive ...); make those calls yield a
/// constant so that the real-game branch runs". The transpiler replaces the call instruction with
/// ldc.i4.0 / ldc.i4.1 and throws at install time unless it finds exactly the expected number of hits
/// (DECISIONS-H1 E: transpilers are fail-closed).
///
/// For an async method pass the method itself with <c>async: true</c>; the state machine's MoveNext is patched.
/// </summary>
public static class GatePatch
{
    private sealed record Spec(string Id, MethodInfo Getter, bool Value, int ExpectedHits);

    private static readonly Dictionary<MethodBase, List<Spec>> _specs = new Dictionary<MethodBase, List<Spec>>();

    public static MethodInfo TestModeIsOn => AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.TestSupport.TestMode), "IsOn") ?? throw new MissingMethodException("TestMode.get_IsOn");

    public static MethodInfo TestModeIsOff => AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.TestSupport.TestMode), "IsOff") ?? throw new MissingMethodException("TestMode.get_IsOff");

    public static MethodInfo NonInteractiveIsActive => AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Helpers.NonInteractiveMode), "IsActive") ?? throw new MissingMethodException("NonInteractiveMode.get_IsActive");

    /// <summary>
    /// Registers and installs the patch. <paramref name="value"/> is what the getter call evaluates to after
    /// patching (e.g. TestMode.IsOn -> false, TestMode.IsOff -> true to take the real branch).
    /// </summary>
    public static void ForceGetter(string id, MethodBase? method, MethodInfo getter, bool value, int expectedHits, PatchCategory category, string reason, bool async = false)
    {
        if (method == null)
        {
            throw new MissingMethodException("Patch target for '" + id + "' was not found; the game assembly changed.");
        }
        MethodBase target = async ? (AccessTools.AsyncMoveNext(method) ?? throw new MissingMethodException("async MoveNext of " + method.Name + " for '" + id + "'")) : method;
        MethodInfo transpiler = PatchRegistry.M(typeof(GatePatch), nameof(Transpile));
        bool first = !_specs.TryGetValue(target, out List<Spec>? list);
        if (first)
        {
            list = new List<Spec>();
            _specs[target] = list;
        }
        list!.Add(new Spec(id, getter, value, expectedHits));
        if (!first)
        {
            PatchRegistry.Harmony.Unpatch(target, transpiler); // re-run the one transpiler with all specs of this method
        }
        try
        {
            PatchRegistry.Transpiler(id, target, transpiler, category, reason + $" [transpiler: exactly {expectedHits} call(s) of {getter.DeclaringType?.Name}.{getter.Name} -> {(value ? "true" : "false")}]");
        }
        catch
        {
            list.RemoveAt(list.Count - 1);
            throw;
        }
    }

    public static IEnumerable<CodeInstruction> Transpile(IEnumerable<CodeInstruction> instructions, MethodBase original)
    {
        List<CodeInstruction> code = new List<CodeInstruction>(instructions);
        if (!_specs.TryGetValue(original, out List<Spec>? specs))
        {
            throw new InvalidOperationException("GatePatch: no spec registered for " + original.DeclaringType?.FullName + "::" + original.Name);
        }
        foreach (Spec spec in specs)
        {
            int hits = 0;
            foreach (CodeInstruction ci in code)
            {
                if ((ci.opcode == OpCodes.Call || ci.opcode == OpCodes.Callvirt) && ci.operand is MethodInfo m && m == spec.Getter)
                {
                    ci.opcode = spec.Value ? OpCodes.Ldc_I4_1 : OpCodes.Ldc_I4_0;
                    ci.operand = null;
                    hits++;
                }
            }
            if (hits != spec.ExpectedHits)
            {
                throw new InvalidOperationException($"GatePatch '{spec.Id}': expected exactly {spec.ExpectedHits} call(s) of {spec.Getter.DeclaringType?.Name}.{spec.Getter.Name} in {original.DeclaringType?.FullName}::{original.Name}, found {hits}; the game assembly changed.");
            }
        }
        return code;
    }
}
