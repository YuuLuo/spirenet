using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using SpireHost.Boot;

namespace SpireHost.Fidelity.Scans;

/// <summary>
/// The three static scans of the fidelity pack (task item 3), run over the loaded, unmodified sts2.dll and
/// GodotSharp.dll with System.Reflection (read-only):
///
///   (a) UI-singleton dereferences: every TestMode-reachable instruction in gameplay-reachable code that
///       consumes the value of a static getter / static field whose type is a Godot node (NGame.Instance,
///       NRun.Instance, NCombatRoom.Instance ...), classified by how the value is consumed: dereferenced
///       (instance call / field access on it = NullReferenceException headlessly), null-conditional (?.),
///       null-checked, stored, passed as an argument, returned.
///   (b) GodotSharp natives: every gameplay-reachable call into GodotSharp whose callee (transitively inside
///       GodotSharp, including the callee type's static constructor) reaches a NativeFuncs entry that is not on
///       HostOptions.NativeAllowList and is not short-circuited by a host stub. Value-returning natives yield a
///       silent 0 headlessly (NullGodot thunks), object-returning ones null.
///   (c) Real-time / thread-pool scheduling: Task.Delay / Run / Yield / ContinueWith / WaitAsync / timers /
///       threads / clocks reached from gameplay code.
///
/// "Gameplay-reachable": forward closure over the call graph (call / callvirt / newobj / ldftn at
/// TestMode-reachable instructions) from roots = non-UI methods that are virtual, unreferenced, or static
/// constructors; UI code (Nodes, AutoSlay, DevConsole, addons, GodotPlugins, Bindings, Mocks, and every type
/// deriving from GodotObject) is never a root and is entered only through static members. "TestMode-reachable":
/// see <see cref="Cfg"/>; branches that the host flips to the real-game branch are evaluated with the flipped value.
/// </summary>
internal static class StaticScans
{
    private const BindingFlags All = BindingFlags.DeclaredOnly | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;

    private sealed class MethodInfoEx
    {
        public MethodBase M = null!;
        public List<Ins>? Code;
        public Cfg? Cfg;
        public bool Ui;
        public bool Root;
        public bool Reachable;
        public string Owner = "";
        public List<int> RefCount = new List<int>();
    }

    private static readonly Dictionary<(string type, string method, string getter), bool> Flipped = new Dictionary<(string, string, string), bool>
    {
        // Core (stage 0, FIDELITY.md E1/E3)
        { ("MegaCrit.Sts2.Core.Models.ActModel", "GetRandomList", "get_IsOff"), true },
        { ("MegaCrit.Sts2.Core.Runs.RunManager", "TryGetRoomTypeForTutorial", "get_IsOn"), false },
        // this pack
        { ("MegaCrit.Sts2.Core.Entities.Merchant.MerchantPotionEntry", "CalcCost", "get_IsOff"), true },
        { ("MegaCrit.Sts2.Core.Models.Relics.CallingBell", "GenerateRewards", "get_IsOn"), false },
        { ("MegaCrit.Sts2.Core.Models.Relics.Cauldron", "GenerateRewards", "get_IsOn"), false },
        { ("MegaCrit.Sts2.Core.Rooms.MerchantRoom", "Exit", "get_IsOn"), false },
        { ("MegaCrit.Sts2.Core.Runs.RunManager", "UpdatePlayerStatsInMapPointHistory", "get_IsOn"), false },
    };

    public static bool IsUiNamespace(string? ns)
    {
        if (ns == null)
        {
            return false;
        }
        return ns.StartsWith("MegaCrit.Sts2.Core.Nodes", StringComparison.OrdinalIgnoreCase)
            || ns.StartsWith("MegaCrit.Sts2.Core.AutoSlay", StringComparison.Ordinal)
            || ns.StartsWith("MegaCrit.Sts2.Core.DevConsole", StringComparison.Ordinal)
            || ns.StartsWith("MegaCrit.Sts2.Core.Bindings", StringComparison.Ordinal)
            || ns.StartsWith("MegaCrit.Sts2.addons", StringComparison.Ordinal)
            || ns.StartsWith("GodotPlugins", StringComparison.Ordinal)
            || ns.StartsWith("RiderTestRunner", StringComparison.Ordinal)
            || ns.EndsWith(".Mocks", StringComparison.Ordinal) || ns.EndsWith(".Mock", StringComparison.Ordinal);
    }

    /// <summary>Namespaces whose public / virtual members the host or the game's hook dispatch can call during a run.</summary>
    public static bool IsGameplayNamespace(string? ns)
    {
        if (ns == null || IsUiNamespace(ns))
        {
            return false;
        }
        string[] gameplay =
        {
            "MegaCrit.Sts2.Core.Combat", "MegaCrit.Sts2.Core.Commands", "MegaCrit.Sts2.Core.Entities", "MegaCrit.Sts2.Core.Events",
            "MegaCrit.Sts2.Core.Factories", "MegaCrit.Sts2.Core.GameActions", "MegaCrit.Sts2.Core.Hooks", "MegaCrit.Sts2.Core.HoverTips",
            "MegaCrit.Sts2.Core.Map", "MegaCrit.Sts2.Core.Models", "MegaCrit.Sts2.Core.MonsterMoves", "MegaCrit.Sts2.Core.Multiplayer.Game",
            "MegaCrit.Sts2.Core.Odds", "MegaCrit.Sts2.Core.Random", "MegaCrit.Sts2.Core.Rewards", "MegaCrit.Sts2.Core.Rooms",
            "MegaCrit.Sts2.Core.Runs", "MegaCrit.Sts2.Core.Unlocks", "MegaCrit.Sts2.Core.ValueProps", "MegaCrit.Sts2.Core.CardSelection",
            "MegaCrit.Sts2.Core.Localization", "MegaCrit.Sts2.Core.Helpers", "MegaCrit.Sts2.Core.Context",
        };
        foreach (string g in gameplay)
        {
            if (ns == g || ns.StartsWith(g + ".", StringComparison.Ordinal))
            {
                return true;
            }
        }
        return false;
    }

    private static Type Outer(Type t)
    {
        while (t.DeclaringType != null)
        {
            t = t.DeclaringType;
        }
        return t;
    }

    private static bool IsGodotObjectType(Type? t)
    {
        try
        {
            return t != null && typeof(Godot.GodotObject).IsAssignableFrom(t);
        }
        catch
        {
            return false;
        }
    }

    private static string Name(MethodBase m) => (m.DeclaringType?.FullName ?? "?") + "::" + m.Name;

    /// <summary>Kickoff method name of a compiler-generated state machine / closure method (for the flipped-gate lookup and reports).</summary>
    private static (Type outer, string method) Logical(MethodBase m)
    {
        Type? t = m.DeclaringType;
        string name = m.Name;
        while (t != null && t.Name.StartsWith("<", StringComparison.Ordinal))
        {
            int close = t.Name.IndexOf('>');
            if (close > 1 && (name == "MoveNext" || name.StartsWith("<", StringComparison.Ordinal) || name.StartsWith("b__", StringComparison.Ordinal)))
            {
                string inner = t.Name.Substring(1, close - 1);
                if (!inner.StartsWith("c__DisplayClass", StringComparison.Ordinal) && inner.Length > 0)
                {
                    name = inner;
                }
            }
            t = t.DeclaringType;
        }
        if (name.StartsWith("<", StringComparison.Ordinal))
        {
            int close = name.IndexOf('>');
            if (close > 1)
            {
                name = name.Substring(1, close - 1);
            }
        }
        return (t ?? m.DeclaringType!, name);
    }

    private static bool? ConstantOf(MethodBase method, MethodBase getter)
    {
        string? decl = getter.DeclaringType?.FullName;
        bool? v = null;
        if (decl == "MegaCrit.Sts2.Core.TestSupport.TestMode")
        {
            v = getter.Name == "get_IsOn" ? true : getter.Name == "get_IsOff" ? false : null;
        }
        else if (decl == "MegaCrit.Sts2.Core.Helpers.NonInteractiveMode" && getter.Name == "get_IsActive")
        {
            v = true;
        }
        if (v == null)
        {
            return null;
        }
        (Type outer, string logical) = Logical(method);
        Type? dt = method.DeclaringType;
        // the flipped table is keyed by the declaring (non-generated) type of the patched method
        Type? owner = dt;
        while (owner != null && owner.Name.StartsWith("<", StringComparison.Ordinal))
        {
            owner = owner.DeclaringType;
        }
        if (owner != null && Flipped.TryGetValue((owner.FullName ?? "", logical, getter.Name), out bool flipped))
        {
            return flipped;
        }
        _ = outer;
        return v;
    }

    private static IEnumerable<Type> SafeTypes(Assembly a)
    {
        try
        {
            return a.GetTypes();
        }
        catch (ReflectionTypeLoadException e)
        {
            return e.Types.Where(t => t != null)!;
        }
    }

    private static IEnumerable<MethodBase> MethodsOf(Type t)
    {
        MethodInfo[] ms;
        ConstructorInfo[] cs;
        try
        {
            ms = t.GetMethods(All);
            cs = t.GetConstructors(All);
        }
        catch
        {
            yield break;
        }
        foreach (MethodInfo m in ms)
        {
            yield return m;
        }
        foreach (ConstructorInfo c in cs)
        {
            yield return c;
        }
    }

    // =====================================================================================================

    public static object Run(string sourceRoot, HashSet<string> nativeAllowList, HashSet<string> stubbedGodotMethods)
    {
        Stopwatch sw = Stopwatch.StartNew();
        Assembly sts2 = typeof(MegaCrit.Sts2.Core.Models.ModelDb).Assembly;
        Assembly godot = typeof(Godot.GodotObject).Assembly;

        // ---- index sts2 ----
        Dictionary<MethodBase, MethodInfoEx> index = new Dictionary<MethodBase, MethodInfoEx>();
        foreach (Type t in SafeTypes(sts2))
        {
            Type outer = Outer(t);
            bool uiType = IsUiNamespace(outer.Namespace) || IsGodotObjectType(t) || IsGodotObjectType(outer);
            foreach (MethodBase m in MethodsOf(t))
            {
                MethodInfoEx x = new MethodInfoEx { M = m, Ui = uiType };
                x.Code = IlReader.Read(m);
                if (x.Code != null)
                {
                    x.Cfg = new Cfg(m, x.Code, ConstantOf);
                }
                index[m] = x;
            }
        }
        // state machine kickoff edges: kickoff -> MoveNext
        Dictionary<MethodBase, List<MethodBase>> extraEdges = new Dictionary<MethodBase, List<MethodBase>>();
        foreach (MethodInfoEx x in index.Values)
        {
            StateMachineAttribute? sm = x.M.GetCustomAttribute<StateMachineAttribute>();
            if (sm?.StateMachineType != null)
            {
                MethodInfo? moveNext = sm.StateMachineType.GetMethod("MoveNext", All);
                if (moveNext != null)
                {
                    if (!extraEdges.TryGetValue(x.M, out List<MethodBase>? l))
                    {
                        l = new List<MethodBase>();
                        extraEdges[x.M] = l;
                    }
                    l.Add(moveNext);
                }
            }
        }
        // references
        Dictionary<MethodBase, int> refCount = new Dictionary<MethodBase, int>();
        foreach (MethodInfoEx x in index.Values)
        {
            if (x.Code == null)
            {
                continue;
            }
            foreach (Ins i in x.Code)
            {
                if (i.Callee is MethodBase c && index.ContainsKey(c))
                {
                    refCount[c] = refCount.GetValueOrDefault(c) + 1;
                }
            }
        }
        foreach (List<MethodBase> l in extraEdges.Values)
        {
            foreach (MethodBase c in l)
            {
                refCount[c] = refCount.GetValueOrDefault(c) + 1;
            }
        }
        // roots + forward closure
        Queue<MethodInfoEx> q = new Queue<MethodInfoEx>();
        foreach (MethodInfoEx x in index.Values)
        {
            if (x.Ui || x.M.DeclaringType!.Name.StartsWith("<", StringComparison.Ordinal) || x.M.Name.StartsWith("<", StringComparison.Ordinal))
            {
                continue; // UI code is never a root; compiler-generated state machines / closures are entered from their owner
            }
            if (!IsGameplayNamespace(Outer(x.M.DeclaringType!).Namespace))
            {
                continue; // platform / transport / saves / debug / achievements ...: entered only from gameplay code
            }
            bool isVirtual = x.M is MethodInfo mi && mi.IsVirtual;
            bool isCctor = x.M is ConstructorInfo ci && ci.IsStatic;
            bool isPublic = x.M.IsPublic && (x.M.DeclaringType!.IsPublic || x.M.DeclaringType.IsNestedPublic);
            if (isVirtual || isCctor || isPublic)
            {
                x.Root = true;
                x.Reachable = true;
                q.Enqueue(x);
            }
        }
        while (q.Count > 0)
        {
            MethodInfoEx x = q.Dequeue();
            IEnumerable<MethodBase> callees = Enumerable.Empty<MethodBase>();
            if (x.Code != null)
            {
                List<MethodBase> cs = new List<MethodBase>();
                for (int k = 0; k < x.Code.Count; k++)
                {
                    if (x.Cfg!.Reachable[k] && x.Code[k].Callee is MethodBase c)
                    {
                        cs.Add(c);
                    }
                }
                callees = cs;
            }
            if (extraEdges.TryGetValue(x.M, out List<MethodBase>? more))
            {
                callees = callees.Concat(more);
            }
            foreach (MethodBase c in callees)
            {
                MethodBase key = c;
                if (c is MethodInfo cmi && cmi.IsGenericMethod && !cmi.IsGenericMethodDefinition)
                {
                    key = cmi.GetGenericMethodDefinition();
                }
                if (!index.TryGetValue(key, out MethodInfoEx? y))
                {
                    // members of generic type instantiations: map to the definition
                    if (c.DeclaringType != null && c.DeclaringType.IsGenericType && !c.DeclaringType.IsGenericTypeDefinition)
                    {
                        try
                        {
                            MethodBase? def = MethodBase.GetMethodFromHandle(c.MethodHandle, c.DeclaringType.GetGenericTypeDefinition().TypeHandle);
                            if (def != null)
                            {
                                index.TryGetValue(def, out y);
                            }
                        }
                        catch
                        {
                        }
                    }
                    if (y == null)
                    {
                        continue;
                    }
                }
                if (y.Reachable)
                {
                    continue;
                }
                if (y.Ui && !y.M.IsStatic && !(y.M is ConstructorInfo))
                {
                    continue; // a UI instance method needs a live node: never runs headlessly (a call on a null node is caught as a dereference at the call site)
                }
                y.Reachable = true;
                q.Enqueue(y);
            }
        }

        int reachableMethods = index.Values.Count(x => x.Reachable);

        // ---- (a) UI singletons ----
        List<Dictionary<string, object?>> derefs = new List<Dictionary<string, object?>>();
        SortedDictionary<string, int> derefByClass = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (MethodInfoEx x in index.Values)
        {
            if (!x.Reachable || x.Code == null)
            {
                continue;
            }
            for (int k = 0; k < x.Code.Count; k++)
            {
                if (!x.Cfg!.Reachable[k])
                {
                    continue;
                }
                Ins ins = x.Code[k];
                string? singleton = null;
                if ((ins.Op == OpCodes.Call) && ins.Callee is MethodInfo g && g.IsStatic && g.GetParameters().Length == 0 && g.Name.StartsWith("get_", StringComparison.Ordinal)
                    && g.DeclaringType != null && g.DeclaringType.Assembly == sts2 && IsGodotObjectType(g.ReturnType))
                {
                    singleton = g.DeclaringType.Name + "." + g.Name.Substring(4);
                }
                else if (ins.Op == OpCodes.Ldsfld && ins.Field is FieldInfo f && f.DeclaringType?.Assembly == sts2 && IsGodotObjectType(f.FieldType))
                {
                    singleton = f.DeclaringType!.Name + "." + f.Name;
                }
                if (singleton == null)
                {
                    continue;
                }
                (string use, string detail) = ClassifyUse(x.Code, k);
                derefByClass[use] = derefByClass.GetValueOrDefault(use) + 1;
                (Type outerT, string logical) = Logical(x.M);
                derefs.Add(new Dictionary<string, object?>
                {
                    ["use"] = use,
                    ["singleton"] = singleton,
                    ["detail"] = detail,
                    ["type"] = outerT.FullName,
                    ["method"] = logical,
                    ["il"] = $"{Name(x.M)}+IL_{ins.Offset:X4}",
                    ["ui_code"] = x.Ui,
                    ["source"] = SourceLines(sourceRoot, outerT, singleton),
                });
            }
        }

        // ---- (b) GodotSharp natives ----
        Dictionary<MethodBase, HashSet<string>> nativeOf = NativeClosure(godot, stubbedGodotMethods);
        List<Dictionary<string, object?>> natives = new List<Dictionary<string, object?>>();
        SortedDictionary<string, int> nativeCallees = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (MethodInfoEx x in index.Values)
        {
            if (!x.Reachable || x.Code == null)
            {
                continue;
            }
            for (int k = 0; k < x.Code.Count; k++)
            {
                if (!x.Cfg!.Reachable[k] || x.Code[k].Callee is not MethodBase c || c.DeclaringType?.Assembly != godot)
                {
                    continue;
                }
                if (stubbedGodotMethods.Contains((c.DeclaringType?.FullName ?? "") + "::" + c.Name))
                {
                    continue; // a host prefix returns before the body (and before the type initializer is touched)
                }
                HashSet<string> set = NativesOf(c, nativeOf);
                set.ExceptWith(nativeAllowList);
                if (set.Count == 0)
                {
                    continue;
                }
                string callee = (c.DeclaringType?.FullName ?? "?") + "::" + c.Name;
                nativeCallees[callee] = nativeCallees.GetValueOrDefault(callee) + 1;
                (Type outerT, string logical) = Logical(x.M);
                natives.Add(new Dictionary<string, object?>
                {
                    ["callee"] = callee,
                    ["returns"] = c is MethodInfo cm ? cm.ReturnType.Name : "ctor",
                    ["natives"] = set.OrderBy(s => s, StringComparer.Ordinal).Take(6).ToList(),
                    ["type"] = outerT.FullName,
                    ["method"] = logical,
                    ["il"] = $"{Name(x.M)}+IL_{x.Code[k].Offset:X4}",
                    ["ui_code"] = x.Ui,
                });
            }
        }

        // ---- (c) scheduling ----
        List<Dictionary<string, object?>> scheduling = new List<Dictionary<string, object?>>();
        foreach (MethodInfoEx x in index.Values)
        {
            if (!x.Reachable || x.Code == null)
            {
                continue;
            }
            for (int k = 0; k < x.Code.Count; k++)
            {
                if (!x.Cfg!.Reachable[k] || x.Code[k].Callee is not MethodBase c)
                {
                    continue;
                }
                string? kind = SchedulingKind(c, x.Code, k);
                if (kind == null)
                {
                    continue;
                }
                (Type outerT, string logical) = Logical(x.M);
                scheduling.Add(new Dictionary<string, object?>
                {
                    ["kind"] = kind,
                    ["callee"] = (c.DeclaringType?.FullName ?? "?") + "::" + c.Name,
                    ["type"] = outerT.FullName,
                    ["method"] = logical,
                    ["il"] = $"{Name(x.M)}+IL_{x.Code[k].Offset:X4}",
                    ["ui_code"] = x.Ui,
                });
            }
        }

        return new
        {
            kind = "fidelity-static-scan",
            wall_ms = sw.ElapsedMilliseconds,
            methods = index.Count,
            reachable_methods = reachableMethods,
            roots = index.Values.Count(x => x.Root),
            ui_singletons = new
            {
                by_use = derefByClass,
                sites = derefs.OrderBy(d => (string)d["use"]!).ThenBy(d => (string)d["type"]!).ThenBy(d => (string)d["method"]!).ToList(),
            },
            godot_natives = new
            {
                by_callee = nativeCallees,
                sites = natives.OrderBy(d => (string)d["callee"]!).ThenBy(d => (string)d["type"]!).ToList(),
            },
            scheduling = scheduling.OrderBy(d => (string)d["kind"]!).ThenBy(d => (string)d["type"]!).ToList(),
        };
    }

    /// <summary>Debug aid (--scan-il --type T --method M): the decoded IL of a method and of its state machine / closures, with TestMode reachability.</summary>
    public static string DumpIl(string typeName, string method)
    {
        Assembly sts2 = typeof(MegaCrit.Sts2.Core.Models.ModelDb).Assembly;
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        foreach (Type t in SafeTypes(sts2))
        {
            Type outer = Outer(t);
            if (outer.FullName != typeName && outer.Name != typeName)
            {
                continue;
            }
            foreach (MethodBase m in MethodsOf(t))
            {
                (Type _, string logical) = Logical(m);
                if (logical != method && m.Name != method)
                {
                    continue;
                }
                List<Ins>? code = IlReader.Read(m);
                if (code == null)
                {
                    continue;
                }
                Cfg cfg = new Cfg(m, code, ConstantOf);
                sb.AppendLine("== " + Name(m));
                foreach (Ins i in code)
                {
                    sb.Append(cfg.Reachable[i.Index] ? "  R " : "  . ").AppendLine(i.ToString());
                }
            }
        }
        return sb.ToString();
    }

    // -----------------------------------------------------------------------------------------------------
    //  (a) how is the value pushed at code[k] consumed?
    // -----------------------------------------------------------------------------------------------------

    private static (string use, string detail) ClassifyUse(List<Ins> code, int k)
    {
        // stack model: true = our value (or a copy of it)
        List<bool> stack = new List<bool> { true };
        for (int j = k + 1; j < code.Count && j < k + 64; j++)
        {
            Ins ins = code[j];
            OpCode op = ins.Op;
            if (op == OpCodes.Dup)
            {
                if (stack.Count > 0)
                {
                    stack.Add(stack[^1]);
                }
                else
                {
                    stack.Add(false);
                }
                continue;
            }
            int pops = Pops(ins, out bool hasThis);
            int pushes = Pushes(ins);
            if (pops < 0)
            {
                return ("complex", ins.Op.Name);
            }
            int ours = -1; // position among the popped values, 0 = deepest
            for (int p = 0; p < pops; p++)
            {
                int si = stack.Count - pops + p;
                if (si >= 0 && stack[si])
                {
                    ours = p;
                    break;
                }
            }
            if (ours >= 0)
            {
                if (op == OpCodes.Brtrue || op == OpCodes.Brtrue_S || op == OpCodes.Brfalse || op == OpCodes.Brfalse_S)
                {
                    bool dupBefore = code[j - 1].Op == OpCodes.Dup;
                    return (dupBefore ? "null_conditional" : "null_checked", op.Name);
                }
                if (op == OpCodes.Ceq || op == OpCodes.Cgt_Un || op == OpCodes.Clt_Un || op == OpCodes.Beq || op == OpCodes.Beq_S || op == OpCodes.Bne_Un || op == OpCodes.Bne_Un_S)
                {
                    return ("null_checked", op.Name);
                }
                if (op == OpCodes.Ldfld || op == OpCodes.Ldflda || (op == OpCodes.Stfld && ours == 0))
                {
                    return ("deref", op.Name + " " + ins.Field?.Name);
                }
                if ((op == OpCodes.Call || op == OpCodes.Callvirt) && ins.Callee is MethodBase c)
                {
                    if (hasThis && ours == 0)
                    {
                        return ("deref", c.DeclaringType?.Name + "." + c.Name);
                    }
                    if (c.Name is "IsInstanceValid" or "op_Equality" or "op_Inequality" or "ReferenceEquals")
                    {
                        return ("null_checked", c.Name);
                    }
                    return ("argument", c.DeclaringType?.Name + "." + c.Name);
                }
                if (op == OpCodes.Newobj)
                {
                    return ("argument", "newobj " + ins.Callee?.DeclaringType?.Name);
                }
                if (op == OpCodes.Isinst || op == OpCodes.Castclass)
                {
                    stack.RemoveAt(stack.Count - 1);
                    stack.Add(true);
                    continue;
                }
                if (op == OpCodes.Pop)
                {
                    return ("discarded", "pop");
                }
                if (op == OpCodes.Ret)
                {
                    return ("returned", "ret");
                }
                if (op.Name.StartsWith("stloc", StringComparison.Ordinal) || op.Name.StartsWith("starg", StringComparison.Ordinal) || op == OpCodes.Stfld || op == OpCodes.Stsfld || op.Name.StartsWith("stelem", StringComparison.Ordinal))
                {
                    // stored: look for an immediate null test of the reloaded value
                    if (op.Name.StartsWith("stloc", StringComparison.Ordinal) && j + 2 < code.Count && code[j + 1].Op.Name.StartsWith("ldloc", StringComparison.Ordinal)
                        && (code[j + 2].Op == OpCodes.Brfalse || code[j + 2].Op == OpCodes.Brfalse_S || code[j + 2].Op == OpCodes.Brtrue || code[j + 2].Op == OpCodes.Brtrue_S))
                    {
                        return ("null_checked", "stloc+test");
                    }
                    return ("stored", op.Name);
                }
                return ("complex", op.Name);
            }
            if (op.FlowControl == FlowControl.Branch || op.FlowControl == FlowControl.Cond_Branch || op.FlowControl == FlowControl.Return || op.FlowControl == FlowControl.Throw)
            {
                return ("complex", "flow " + op.Name);
            }
            for (int p = 0; p < pops && stack.Count > 0; p++)
            {
                stack.RemoveAt(stack.Count - 1);
            }
            for (int p = 0; p < pushes; p++)
            {
                stack.Add(false);
            }
        }
        return ("complex", "window");
    }

    private static int Pops(Ins ins, out bool hasThis)
    {
        hasThis = false;
        OpCode op = ins.Op;
        if (op == OpCodes.Call || op == OpCodes.Callvirt || op == OpCodes.Newobj)
        {
            if (ins.Callee is not MethodBase m)
            {
                return -1;
            }
            int n = m.GetParameters().Length;
            if (op != OpCodes.Newobj && !m.IsStatic)
            {
                hasThis = true;
                n++;
            }
            return n;
        }
        if (op == OpCodes.Ret)
        {
            return 1;
        }
        return op.StackBehaviourPop switch
        {
            StackBehaviour.Pop0 => 0,
            StackBehaviour.Pop1 or StackBehaviour.Popi or StackBehaviour.Popref => 1,
            StackBehaviour.Pop1_pop1 or StackBehaviour.Popi_pop1 or StackBehaviour.Popi_popi or StackBehaviour.Popi_popi8 or StackBehaviour.Popi_popr4
                or StackBehaviour.Popi_popr8 or StackBehaviour.Popref_pop1 or StackBehaviour.Popref_popi => 2,
            StackBehaviour.Popi_popi_popi or StackBehaviour.Popref_popi_popi or StackBehaviour.Popref_popi_popi8 or StackBehaviour.Popref_popi_popr4
                or StackBehaviour.Popref_popi_popr8 or StackBehaviour.Popref_popi_popref or StackBehaviour.Popref_popi_pop1 => 3,
            _ => -1,
        };
    }

    private static int Pushes(Ins ins)
    {
        OpCode op = ins.Op;
        if (op == OpCodes.Call || op == OpCodes.Callvirt)
        {
            return ins.Callee is MethodInfo m && m.ReturnType != typeof(void) ? 1 : 0;
        }
        if (op == OpCodes.Newobj)
        {
            return 1;
        }
        return op.StackBehaviourPush switch
        {
            StackBehaviour.Push0 => 0,
            StackBehaviour.Push1_push1 => 2,
            _ => 1,
        };
    }

    private static List<string> SourceLines(string root, Type outer, string singleton)
    {
        List<string> r = new List<string>();
        try
        {
            string path = Path.Combine(root, outer.Namespace ?? "", outer.Name.Split('`')[0] + ".cs");
            if (!File.Exists(path))
            {
                return r;
            }
            string needle = singleton.Replace(".", ".") + ".";
            string[] lines = File.ReadAllLines(path);
            for (int i = 0; i < lines.Length; i++)
            {
                if (lines[i].Contains(needle, StringComparison.Ordinal) || lines[i].Contains(singleton + "?", StringComparison.Ordinal) || lines[i].Contains(singleton + " ", StringComparison.Ordinal) || lines[i].Contains(singleton + ")", StringComparison.Ordinal))
                {
                    r.Add($"{outer.Namespace}/{Path.GetFileName(path)}:{i + 1}: {lines[i].Trim()}");
                }
            }
        }
        catch
        {
        }
        return r;
    }

    // -----------------------------------------------------------------------------------------------------
    //  (b) native closure inside GodotSharp
    // -----------------------------------------------------------------------------------------------------

    private static Dictionary<MethodBase, HashSet<string>> NativeClosure(Assembly godot, HashSet<string> stubbed)
    {
        Dictionary<MethodBase, HashSet<string>> direct = new Dictionary<MethodBase, HashSet<string>>();
        Dictionary<MethodBase, List<MethodBase>> calls = new Dictionary<MethodBase, List<MethodBase>>();
        Type? nativeFuncs = godot.GetType("Godot.NativeInterop.NativeFuncs");
        foreach (Type t in SafeTypes(godot))
        {
            foreach (MethodBase m in MethodsOf(t))
            {
                HashSet<string> own = new HashSet<string>(StringComparer.Ordinal);
                if (t == nativeFuncs && m.Name.StartsWith("godotsharp_", StringComparison.Ordinal))
                {
                    own.Add(m.Name);
                }
                direct[m] = own;
                if (stubbed.Contains((t.FullName ?? "") + "::" + m.Name))
                {
                    calls[m] = new List<MethodBase>(); // body never runs (host prefix returns first)
                    continue;
                }
                List<Ins>? code = IlReader.Read(m);
                List<MethodBase> cs = new List<MethodBase>();
                if (code != null)
                {
                    foreach (Ins i in code)
                    {
                        if (i.Callee is MethodBase c && c.DeclaringType?.Assembly == godot)
                        {
                            cs.Add(Norm(c));
                        }
                    }
                }
                calls[m] = cs;
            }
        }
        // fixpoint
        Dictionary<MethodBase, HashSet<string>> closure = direct.ToDictionary(kv => kv.Key, kv => new HashSet<string>(kv.Value, StringComparer.Ordinal));
        bool changed = true;
        int passes = 0;
        while (changed && passes < 64)
        {
            changed = false;
            passes++;
            foreach (KeyValuePair<MethodBase, List<MethodBase>> kv in calls)
            {
                HashSet<string> mine = closure[kv.Key];
                foreach (MethodBase c in kv.Value)
                {
                    if (closure.TryGetValue(c, out HashSet<string>? theirs) && theirs.Count > 0)
                    {
                        int before = mine.Count;
                        mine.UnionWith(theirs);
                        changed |= mine.Count != before;
                    }
                }
            }
        }
        return closure;
    }

    private static MethodBase Norm(MethodBase c)
    {
        if (c is MethodInfo mi && mi.IsGenericMethod && !mi.IsGenericMethodDefinition)
        {
            return mi.GetGenericMethodDefinition();
        }
        if (c.DeclaringType != null && c.DeclaringType.IsGenericType && !c.DeclaringType.IsGenericTypeDefinition)
        {
            try
            {
                MethodBase? def = MethodBase.GetMethodFromHandle(c.MethodHandle, c.DeclaringType.GetGenericTypeDefinition().TypeHandle);
                if (def != null)
                {
                    return def;
                }
            }
            catch
            {
            }
        }
        return c;
    }

    private static HashSet<string> NativesOf(MethodBase c, Dictionary<MethodBase, HashSet<string>> closure)
    {
        HashSet<string> r = new HashSet<string>(StringComparer.Ordinal);
        if (closure.TryGetValue(Norm(c), out HashSet<string>? s))
        {
            r.UnionWith(s);
        }
        ConstructorInfo? cctor = c.DeclaringType?.TypeInitializer;
        if (cctor != null && closure.TryGetValue(cctor, out HashSet<string>? s2))
        {
            r.UnionWith(s2);
        }
        return r;
    }

    // -----------------------------------------------------------------------------------------------------
    //  (c) scheduling
    // -----------------------------------------------------------------------------------------------------

    private static string? SchedulingKind(MethodBase c, List<Ins> code, int k)
    {
        string t = c.DeclaringType?.FullName ?? "";
        string n = c.Name;
        if (t.StartsWith("System.Threading.Tasks.Task", StringComparison.Ordinal) && !t.StartsWith("System.Threading.Tasks.TaskCompletionSource", StringComparison.Ordinal))
        {
            if (n == "Delay")
            {
                return "Task.Delay";
            }
            if (n == "Run" || n == "StartNew")
            {
                return "Task.Run";
            }
            if (n == "Yield")
            {
                return "Task.Yield";
            }
            if (n == "ContinueWith")
            {
                return "ContinueWith";
            }
            if (n == "WaitAsync")
            {
                return "WaitAsync";
            }
            if (n == "ConfigureAwait")
            {
                return "ConfigureAwait";
            }
            if (n is "Wait" or "WaitAll" or "WaitAny" or "get_Result" or "RunSynchronously")
            {
                return "blocking:" + n;
            }
            if (t == "System.Threading.Tasks.TaskScheduler")
            {
                return "TaskScheduler." + n;
            }
        }
        if (t.StartsWith("System.Threading.Tasks.TaskCompletionSource", StringComparison.Ordinal) && c is ConstructorInfo ctor && ctor.GetParameters().Any(p => p.ParameterType.Name.Contains("Options", StringComparison.Ordinal)))
        {
            return "TCS(options)";
        }
        if (t == "System.Threading.Timer" || t == "System.Timers.Timer" || t == "System.Threading.PeriodicTimer")
        {
            return "Timer";
        }
        if (t == "System.Threading.Thread" && (n == ".ctor" || n == "Start" || n == "Sleep"))
        {
            return "Thread." + n;
        }
        if (t == "System.Threading.ThreadPool")
        {
            return "ThreadPool." + n;
        }
        if (t == "System.Threading.CancellationTokenSource" && (n == "CancelAfter" || (n == ".ctor" && c.GetParameters().Length == 1)))
        {
            return "CTS.timeout";
        }
        if (t == "System.Threading.SemaphoreSlim" && n == "WaitAsync" && c.GetParameters().Length > 0)
        {
            return "SemaphoreSlim.WaitAsync(timeout)";
        }
        if (t == "System.Threading.SynchronizationContext")
        {
            return "SynchronizationContext." + n;
        }
        if (t == "System.Diagnostics.Stopwatch")
        {
            return "Stopwatch." + n;
        }
        if ((t == "System.DateTime" || t == "System.DateTimeOffset") && (n == "get_Now" || n == "get_UtcNow"))
        {
            return "clock:" + t.Substring(7) + "." + n.Substring(4);
        }
        if (t == "System.Environment" && n.StartsWith("get_TickCount", StringComparison.Ordinal))
        {
            return "clock:Environment.TickCount";
        }
        if (t.StartsWith("Godot.", StringComparison.Ordinal) && (n == "CreateTimer" || n == "ToSignal" || n == "GetTicksMsec" || n == "GetTicksUsec" || n == "GetMainLoop" || n == "CallDeferred"))
        {
            return "godot:" + c.DeclaringType!.Name + "." + n;
        }
        if (t == "MegaCrit.Sts2.Core.Helpers.TaskHelper" && n == "RunSafely")
        {
            return null; // fire-and-forget on the pump: not a scheduling leftover (DESIGN 3.1); counted elsewhere
        }
        return null;
    }
}
