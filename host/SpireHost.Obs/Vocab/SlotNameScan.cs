using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Obs.Vocab;

/// <summary>
/// Export-time only: the creature slot names of v0.113 (vocab "slot_name", schema 12.1).
///
/// EncounterModel.Slots lists only the fixed layouts; many encounters name their slots in GenerateMonsters
/// (<c>(ModelDb.Monster&lt;BowlbugRock&gt;().ToMutable(), "odd")</c>, Models.Encounters/BowlbugsWeak.cs:31) and powers /
/// monsters spawn into named slots (<c>CombatState.CreateCreature(..., "fat")</c>, Models.Powers/SurprisePower.cs:24).
/// This scan decodes the IL of every method of sts2 and collects a string literal that is IMMEDIATELY consumed as
///   * the string half of a <c>ValueTuple&lt;MonsterModel-or-subclass, string&gt;</c> (newobj .ctor), or
///   * the last argument (parameter "slot" / "slotName") of CombatState.CreateCreature / CreatureCmd.Add / the Creature constructor,
/// which is exactly how the game names a slot (the literal is the Creature.SlotName), plus static string / string[] fields
/// of encounter types whose name contains "slot". Static game knowledge only.
/// </summary>
public static class SlotNameScan
{
    private static readonly Dictionary<short, OperandType> Operands = BuildOperands();

    private static Dictionary<short, OperandType> BuildOperands()
    {
        Dictionary<short, OperandType> d = new Dictionary<short, OperandType>();
        foreach (FieldInfo f in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (f.GetValue(null) is OpCode op)
            {
                d[op.Value] = op.OperandType;
            }
        }
        return d;
    }

    public static SortedSet<string> Scan(out int methods)
    {
        SortedSet<string> names = new SortedSet<string>(StringComparer.Ordinal);
        methods = 0;
        Assembly asm = typeof(EncounterModel).Assembly;
        Type[] types;
        try
        {
            types = asm.GetTypes();
        }
        catch (ReflectionTypeLoadException e)
        {
            types = Array.FindAll(e.Types, t => t != null)!;
        }
        const BindingFlags all = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
        foreach (Type t in types)
        {
            foreach (MethodBase m in Methods(t, all))
            {
                byte[]? il;
                try
                {
                    il = m.GetMethodBody()?.GetILAsByteArray();
                }
                catch (Exception)
                {
                    continue;
                }
                if (il == null)
                {
                    continue;
                }
                methods++;
                ScanMethod(m, il, names);
            }
        }
        // slot names kept in a static table of an encounter (private static readonly string[] _slotNames = { "first",
        // "middle", "last" }, Models.Encounters/BowlbugsNormal.cs:29) or a const (GremlinMercNormal.sneakySlot)
        foreach (Type t in types)
        {
            if (!typeof(EncounterModel).IsAssignableFrom(t))
            {
                continue;
            }
            foreach (FieldInfo f in t.GetFields(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (f.Name.IndexOf("slot", StringComparison.OrdinalIgnoreCase) < 0)
                {
                    continue;
                }
                object? v = f.IsLiteral ? f.GetRawConstantValue() : f.GetValue(null);
                if (v is string one && one.Length > 0)
                {
                    names.Add(one);
                }
                else if (v is string[] many)
                {
                    foreach (string x in many)
                    {
                        if (!string.IsNullOrEmpty(x))
                        {
                            names.Add(x);
                        }
                    }
                }
            }
        }
        return names;
    }

    private static IEnumerable<MethodBase> Methods(Type t, BindingFlags flags)
    {
        foreach (MethodInfo m in t.GetMethods(flags))
        {
            yield return m;
        }
        foreach (ConstructorInfo c in t.GetConstructors(flags))
        {
            yield return c;
        }
    }

    private static void ScanMethod(MethodBase m, byte[] il, SortedSet<string> names)
    {
        Module mod = m.Module;
        string? lastString = null; // the operand of the previous instruction if it was ldstr
        int i = 0;
        while (i < il.Length)
        {
            short code = il[i++];
            if (code == 0xFE)
            {
                code = unchecked((short)(0xFE00 | il[i++]));
            }
            OperandType ot = Operands.TryGetValue(code, out OperandType o) ? o : OperandType.InlineNone;
            string? thisString = null;
            switch (ot)
            {
                case OperandType.InlineString:
                {
                    int token = BitConverter.ToInt32(il, i);
                    i += 4;
                    try
                    {
                        thisString = mod.ResolveString(token);
                    }
                    catch (ArgumentException)
                    {
                    }
                    break;
                }
                case OperandType.InlineMethod:
                {
                    int token = BitConverter.ToInt32(il, i);
                    i += 4;
                    if (lastString != null && (code == OpCodes.Newobj.Value || code == OpCodes.Call.Value || code == OpCodes.Callvirt.Value))
                    {
                        MethodBase? target = null;
                        try
                        {
                            target = mod.ResolveMethod(token, m.DeclaringType?.IsGenericType == true ? m.DeclaringType.GetGenericArguments() : null,
                                m.IsGenericMethod ? m.GetGenericArguments() : null);
                        }
                        catch (Exception)
                        {
                        }
                        if (target != null && IsSlotConsumer(target))
                        {
                            names.Add(lastString);
                        }
                    }
                    break;
                }
                case OperandType.InlineNone:
                    break;
                case OperandType.ShortInlineBrTarget:
                case OperandType.ShortInlineI:
                case OperandType.ShortInlineVar:
                    i += 1;
                    break;
                case OperandType.InlineVar:
                    i += 2;
                    break;
                case OperandType.InlineI8:
                case OperandType.InlineR:
                    i += 8;
                    break;
                case OperandType.InlineSwitch:
                {
                    int n = BitConverter.ToInt32(il, i);
                    i += 4 + 4 * n;
                    break;
                }
                default:
                    i += 4;
                    break;
            }
            lastString = thisString;
        }
    }

    private static bool IsSlotConsumer(MethodBase target)
    {
        // CombatState.CreateCreature(..., string? slotName), CreatureCmd.Add[<T>](..., string? slotName), new Creature(monster, side, slotName)
        ParameterInfo[] ps = target.GetParameters();
        if (ps.Length > 0 && ps[^1].ParameterType == typeof(string) && (ps[^1].Name == "slotName" || ps[^1].Name == "slot"))
        {
            string ns = target.DeclaringType?.Namespace ?? "";
            return !ns.Contains(".Bindings", StringComparison.Ordinal) && !ns.Contains(".Nodes", StringComparison.Ordinal);
        }
        if (target is ConstructorInfo && target.DeclaringType is { IsGenericType: true } dt && dt.GetGenericTypeDefinition() == typeof(ValueTuple<,>))
        {
            Type[] args = dt.GetGenericArguments();
            return typeof(MonsterModel).IsAssignableFrom(args[0]) && args[1] == typeof(string);
        }
        return false;
    }
}
