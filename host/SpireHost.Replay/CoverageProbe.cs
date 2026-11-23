using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text.Json;
using HarmonyLib;
using SpireHost.Patches;

namespace SpireHost.Replay;

/// <summary>
/// Opt-in (--coverage) READ-ONLY instrumentation for the oracle run: which TestMode / NonInteractive gates
/// listed by the two non-UI audits (notes\testmode-audit-core.json, testmode-audit-models-commands.json; Node
/// sites are skipped - no node exists headlessly) were EVALUATED while the corpus was replayed. In every listed
/// method (async: its state machine's MoveNext) each call of TestMode.IsOn / TestMode.IsOff /
/// NonInteractiveMode.IsActive / NonInteractiveMode.AutoSlayerCheck is replaced by a wrapper that counts and
/// returns the same value (semantics unchanged). A replay that evaluated a gate and still matched every checksum
/// is evidence that the TestMode branch is equivalent there (within the checksum's coverage); a gate that no
/// replay evaluated is not covered by this oracle. Counts are per method (several sites in one method share it).
/// PatchRegistry category Instrumentation, ids replay-coverage:&lt;site&gt;. Never part of the default patch set.
/// </summary>
public static class CoverageProbe
{
    public sealed class Site
    {
        public string Key = "";            // File.cs:line
        public string Method = "";
        public string Classification = "";
        public string Audit = "";
        public List<MethodBase> Targets = new List<MethodBase>();
        public string? Unresolved;
        public long Calls;
        public int Replays;               // replays during which a gate of the site's method was evaluated
        public List<string> ReplayFiles = new List<string>();
        public int ReplaysNotPass;
    }

    private static readonly List<Site> _sites = new List<Site>();
    private static readonly Dictionary<MethodBase, List<int>> _byMethod = new Dictionary<MethodBase, List<int>>();
    private static readonly Dictionary<MethodBase, int> _slotByMethod = new Dictionary<MethodBase, int>();
    private static readonly Dictionary<MethodBase, int> _gateHits = new Dictionary<MethodBase, int>();
    private static long[] _slotCounts = new long[1024];
    private static long[] _counts = Array.Empty<long>();
    private static long[] _before = Array.Empty<long>();

    public static bool Installed { get; private set; }

    public static IReadOnlyList<Site> Sites => _sites;

    public static void Install(string notesDir)
    {
        if (Installed)
        {
            return;
        }
        Installed = true;
        foreach (string audit in new[] { "testmode-audit-core.json", "testmode-audit-models-commands.json" })
        {
            string path = Path.Combine(notesDir, audit);
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(path));
            foreach (JsonElement e in doc.RootElement.EnumerateArray())
            {
                string file = e.GetProperty("file").GetString() ?? "";
                if (file.Contains(".Nodes", StringComparison.Ordinal))
                {
                    continue;
                }
                Site s = new Site
                {
                    Key = file.Replace("MegaCrit.Sts2.Core.", "") + ":" + e.GetProperty("line").GetInt32(),
                    Method = e.GetProperty("method").GetString() ?? "",
                    Classification = e.GetProperty("classification").GetString() ?? "",
                    Audit = audit,
                };
                Resolve(s, file);
                _sites.Add(s);
            }
        }
        _counts = new long[_sites.Count];
        _before = new long[_sites.Count];
        MethodInfo transpiler = PatchRegistry.M(typeof(CoverageProbe), nameof(CountGates));
        for (int i = 0; i < _sites.Count; i++)
        {
            foreach (MethodBase declared in _sites[i].Targets)
            {
                MethodBase m = declared;
                if (declared is MethodInfo mi && declared.GetCustomAttribute<AsyncStateMachineAttribute>() != null)
                {
                    m = AccessTools.AsyncMoveNext(mi) ?? declared;
                }
                if (_byMethod.TryGetValue(m, out List<int>? list))
                {
                    list.Add(i);
                    continue;
                }
                list = new List<int> { i };
                _byMethod[m] = list;
                _slotByMethod[m] = _slotByMethod.Count;
                try
                {
                    PatchRegistry.Transpiler("replay-coverage:" + _sites[i].Key + ":" + declared.Name, m, transpiler, PatchCategory.Instrumentation,
                        "Read-only gate-evaluation counter for the .mcr oracle coverage report (replay tool, --coverage only): every TestMode.IsOn/IsOff / " +
                        "NonInteractiveMode.IsActive/AutoSlayerCheck call in " + _sites[i].Key + " " + _sites[i].Method + " is replaced by a wrapper that counts and returns the same value.");
                }
                catch (Exception ex)
                {
                    _sites[i].Unresolved = "not instrumented: " + ex.GetBaseException().Message;
                    _byMethod.Remove(m);
                    continue;
                }
            }
        }
        // a site is instrumented when at least one of its methods (overloads) contains a gate call
        foreach (Site s in _sites)
        {
            if (s.Unresolved != null || s.Targets.Count == 0)
            {
                continue;
            }
            int hits = 0;
            foreach (KeyValuePair<MethodBase, List<int>> kv in _byMethod)
            {
                if (kv.Value.Contains(_sites.IndexOf(s)) && _gateHits.TryGetValue(kv.Key, out int h))
                {
                    hits += h;
                }
            }
            if (hits == 0)
            {
                s.Unresolved = "no gate call in the method body (inside a lambda / local function, or already replaced by a default patch)";
            }
        }
    }

    private static void Resolve(Site s, string file)
    {
        // file "MegaCrit.Sts2.Core.Commands/CardCmd.cs"; method "CardCmd.Transform(...)" / "X..cctor" / "X.P (setter)" / "X.M (note)"
        string ns = file.Substring(0, file.IndexOf('/'));
        string spec = s.Method;
        string note = "";
        int paren = spec.IndexOf(" (", StringComparison.Ordinal);
        if (paren > 0)
        {
            note = spec.Substring(paren + 2).TrimEnd(')');
            spec = spec.Substring(0, paren);
        }
        int args = spec.IndexOf('(');
        if (args > 0)
        {
            spec = spec.Substring(0, args);
        }
        int dot = spec.IndexOf('.');
        if (dot <= 0)
        {
            s.Unresolved = "cannot parse method";
            return;
        }
        string typeName = spec.Substring(0, dot);
        string member = spec.Substring(dot + 1);
        Type? t = typeof(MegaCrit.Sts2.Core.Models.ModelDb).Assembly.GetType(ns + "." + typeName);
        if (t == null)
        {
            s.Unresolved = "type not found: " + ns + "." + typeName;
            return;
        }
        if (member.StartsWith(".cctor", StringComparison.Ordinal) || note.Contains("static ctor"))
        {
            s.Unresolved = "static constructor (runs once per process, before any replay)";
            return;
        }
        if (note == "setter" || note == "getter")
        {
            PropertyInfo? p = AccessTools.Property(t, member);
            MethodBase? acc = note == "setter" ? p?.GetSetMethod(true) : p?.GetGetMethod(true);
            if (acc != null)
            {
                s.Targets.Add(acc);
            }
        }
        else
        {
            const BindingFlags all = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
            foreach (MethodInfo m in t.GetMethods(all))
            {
                if (m.Name == member && !m.IsAbstract && !m.ContainsGenericParameters)
                {
                    s.Targets.Add(m);
                }
            }
            if (s.Targets.Count == 0 && AccessTools.Property(t, member)?.GetGetMethod(true) is MethodInfo g)
            {
                s.Targets.Add(g);
            }
        }
        if (s.Targets.Count == 0)
        {
            s.Unresolved = "member not found: " + member;
        }
    }

    private static readonly MethodInfo IsOn = AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.TestSupport.TestMode), "IsOn")!;
    private static readonly MethodInfo IsOff = AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.TestSupport.TestMode), "IsOff")!;
    private static readonly MethodInfo IsActive = AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Helpers.NonInteractiveMode), "IsActive")!;
    private static readonly MethodInfo AutoSlayer = AccessTools.Method(typeof(MegaCrit.Sts2.Core.Helpers.NonInteractiveMode), "AutoSlayerCheck")!;
    private static readonly MethodInfo CountedIsOnM = AccessTools.Method(typeof(CoverageProbe), nameof(CountedIsOn));
    private static readonly MethodInfo CountedIsOffM = AccessTools.Method(typeof(CoverageProbe), nameof(CountedIsOff));
    private static readonly MethodInfo CountedIsActiveM = AccessTools.Method(typeof(CoverageProbe), nameof(CountedIsActive));
    private static readonly MethodInfo CountedAutoSlayerM = AccessTools.Method(typeof(CoverageProbe), nameof(CountedAutoSlayer));

    public static IEnumerable<CodeInstruction> CountGates(IEnumerable<CodeInstruction> instructions, MethodBase __originalMethod)
    {
        List<CodeInstruction> list = new List<CodeInstruction>(instructions);
        int slot = _slotByMethod.TryGetValue(__originalMethod, out int sl) ? sl : -1;
        int hits = 0;
        for (int k = 0; k < list.Count; k++)
        {
            CodeInstruction ci = list[k];
            if ((ci.opcode == OpCodes.Call || ci.opcode == OpCodes.Callvirt) && ci.operand is MethodInfo mi)
            {
                MethodInfo? wrapper = mi == IsOn ? CountedIsOnM : mi == IsOff ? CountedIsOffM : mi == IsActive ? CountedIsActiveM : mi == AutoSlayer ? CountedAutoSlayerM : null;
                if (wrapper == null)
                {
                    continue;
                }
                CodeInstruction load = new CodeInstruction(OpCodes.Ldc_I4, slot);
                load.labels.AddRange(ci.labels);
                load.blocks.AddRange(ci.blocks);
                ci.labels.Clear();
                ci.blocks.Clear();
                ci.opcode = OpCodes.Call;
                ci.operand = wrapper;
                list.Insert(k, load);
                k++;
                hits++;
            }
        }
        _gateHits[__originalMethod] = hits; // 0 = the gate is in a lambda / local function or was already replaced by another patch
        return list;
    }

    private static void Hit(int slot)
    {
        if ((uint)slot < (uint)_slotCounts.Length)
        {
            _slotCounts[slot]++;
        }
    }

    public static bool CountedIsOn(int slot)
    {
        Hit(slot);
        return MegaCrit.Sts2.Core.TestSupport.TestMode.IsOn;
    }

    public static bool CountedIsOff(int slot)
    {
        Hit(slot);
        return MegaCrit.Sts2.Core.TestSupport.TestMode.IsOff;
    }

    public static bool CountedIsActive(int slot)
    {
        Hit(slot);
        return MegaCrit.Sts2.Core.Helpers.NonInteractiveMode.IsActive;
    }

    public static bool CountedAutoSlayer(int slot)
    {
        Hit(slot);
        return MegaCrit.Sts2.Core.Helpers.NonInteractiveMode.AutoSlayerCheck();
    }

    private static void Collect()
    {
        Array.Clear(_counts);
        foreach (KeyValuePair<MethodBase, List<int>> kv in _byMethod)
        {
            long c = _slotCounts[_slotByMethod[kv.Key]];
            foreach (int site in kv.Value)
            {
                _counts[site] += c;
            }
        }
    }

    public static void BeginReplay()
    {
        if (!Installed)
        {
            return;
        }
        Collect();
        Array.Copy(_counts, _before, _counts.Length);
    }

    public static void EndReplay(string file, bool pass)
    {
        if (!Installed)
        {
            return;
        }
        Collect();
        for (int i = 0; i < _sites.Count; i++)
        {
            if (_counts[i] - _before[i] <= 0)
            {
                continue;
            }
            Site s = _sites[i];
            s.Replays++;
            if (s.ReplayFiles.Count < 8)
            {
                s.ReplayFiles.Add(file);
            }
            if (!pass)
            {
                s.ReplaysNotPass++;
            }
        }
    }

    public static object Report()
    {
        Collect();
        for (int i = 0; i < _sites.Count; i++)
        {
            _sites[i].Calls = _counts[i];
        }
        return new
        {
            sites = _sites.Count,
            instrumented = _sites.Count(s => s.Unresolved == null),
            reached = _sites.Count(s => s.Replays > 0),
            by_site = _sites.OrderByDescending(s => s.Replays).ThenBy(s => s.Key, StringComparer.Ordinal).Select(s => new
            {
                site = s.Key,
                method = s.Method,
                classification = s.Classification,
                audit = s.Audit,
                gate_evaluations = s.Calls,
                replays = s.Replays,
                replays_not_pass = s.ReplaysNotPass,
                example_replays = s.ReplayFiles,
                unresolved = s.Unresolved,
            }).ToList(),
        };
    }
}
