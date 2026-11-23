using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using HarmonyLib;

namespace SpireHost.Patches;

public enum PatchCategory
{
    /// <summary>Replaces a Godot-native / UI / platform / IO call that cannot run headlessly. Must not change game state or RNG.</summary>
    Plumbing,

    /// <summary>Removes wall-clock dependence without changing the logical order of events (virtual clock).</summary>
    Scheduling,

    /// <summary>Read-only instrumentation (thread audit, counters, captures). Never alters an argument, a result or game state.</summary>
    Instrumentation,

    /// <summary>Changes what the game computes. The production set must have none of these (checked by --list-patches and the gates).</summary>
    Behaviour,

    /// <summary>
    /// Restores the REAL-game branch of a TestMode / NonInteractive / UI-node gate (the host must run with
    /// TestMode on, so without the patch the test branch would run). Every entry needs: site (file:line), what
    /// the real branch does, how it is patched, and a self-check (host\FIDELITY.md).
    /// </summary>
    Fidelity,
}

public sealed record PatchRecord(string Id, string Target, string Kind, PatchCategory Category, string Reason);

/// <summary>Several registry ids that deliberately touch one method (ruling B3 of the H1 integration): declared, with the reason.</summary>
public sealed record SharedSite(string Site, IReadOnlyList<string> Ids, string Reason);

/// <summary>
/// Every Harmony patch the host installs goes through this registry so that <c>--list-patches</c> and
/// host\FIDELITY.md can be checked against what is really applied. Fail-closed:
///   * a missing target throws at installation;
///   * an id can be registered only once;
///   * ONE PATCH PER SITE: a method (an async method and its state machine count as one site) may carry the
///     patches of one registry id only, unless the ids that share it were declared with
///     <see cref="DeclareSharedSite"/> (checked by <see cref="AssertOnePatchPerSite"/> before the first episode).
/// </summary>
public static class PatchRegistry
{
    private static readonly List<PatchRecord> _records = new List<PatchRecord>();
    private static readonly HashSet<string> _ids = new HashSet<string>(StringComparer.Ordinal);
    private static readonly Dictionary<object, List<string>> _sites = new Dictionary<object, List<string>>();
    private static readonly Dictionary<object, string> _siteNames = new Dictionary<object, string>();
    private static readonly List<(string[] Ids, string Reason)> _declared = new List<(string[], string)>();

    /// <summary>
    /// The one Harmony instance. Not public: a patch is installed through the methods below, which record it (verifier
    /// finding R-5: the thread audit used this instance directly; nothing outside this assembly can any more).
    /// </summary>
    internal static Harmony Harmony { get; } = new Harmony("spirehost");

    public static IReadOnlyList<PatchRecord> Records => _records;

    public static void Prefix(string id, MethodBase? target, MethodInfo prefix, PatchCategory category, string reason)
    {
        Require(id, target);
        Harmony.Patch(target, prefix: new HarmonyMethod(prefix));
        Add(new PatchRecord(id, Describe(target!), "prefix", category, reason), target);
    }

    public static void Postfix(string id, MethodBase? target, MethodInfo postfix, PatchCategory category, string reason)
    {
        Require(id, target);
        Harmony.Patch(target, postfix: new HarmonyMethod(postfix));
        Add(new PatchRecord(id, Describe(target!), "postfix", category, reason), target);
    }

    public static void PrefixPostfix(string id, MethodBase? target, MethodInfo prefix, MethodInfo postfix, PatchCategory category, string reason)
    {
        Require(id, target);
        Harmony.Patch(target, prefix: new HarmonyMethod(prefix), postfix: new HarmonyMethod(postfix));
        Add(new PatchRecord(id, Describe(target!), "prefix+postfix", category, reason), target);
    }

    public static void Transpiler(string id, MethodBase? target, MethodInfo transpiler, PatchCategory category, string reason)
    {
        Require(id, target);
        Harmony.Patch(target, transpiler: new HarmonyMethod(transpiler));
        Add(new PatchRecord(id, Describe(target!), "transpiler", category, reason), target);
    }

    public static void Finalizer(string id, MethodBase? target, MethodInfo finalizer, PatchCategory category, string reason)
    {
        Require(id, target);
        Harmony.Patch(target, finalizer: new HarmonyMethod(finalizer));
        Add(new PatchRecord(id, Describe(target!), "finalizer", category, reason), target);
    }

    /// <summary>Registers a non-Harmony intervention (a field write, a game-provided test hook) so it is listed too.</summary>
    public static void Note(string id, string target, string kind, PatchCategory category, string reason)
    {
        Add(new PatchRecord(id, target, kind, category, reason), null);
    }

    /// <summary>
    /// Declares that these ids must touch the same method (each of them names the others in its ledger row).
    /// Called by whoever installs the later patch, before <see cref="AssertOnePatchPerSite"/>.
    /// </summary>
    public static void DeclareSharedSite(string reason, params string[] ids)
    {
        if (ids.Length < 2)
        {
            throw new ArgumentException("A shared site has at least two ids.");
        }
        string[] sorted = ids.OrderBy(x => x, StringComparer.Ordinal).ToArray();
        _declared.Add((sorted, reason));
    }

    /// <summary>Sites that carry the patches of more than one registry id, with the declaration that allows it (Reason empty = undeclared).</summary>
    public static List<SharedSite> SharedSites()
    {
        List<SharedSite> r = new List<SharedSite>();
        foreach (KeyValuePair<object, List<string>> kv in _sites)
        {
            if (kv.Value.Count < 2)
            {
                continue;
            }
            string[] ids = kv.Value.OrderBy(x => x, StringComparer.Ordinal).ToArray();
            string reason = "";
            foreach ((string[] declared, string why) in _declared)
            {
                if (declared.SequenceEqual(ids, StringComparer.Ordinal))
                {
                    reason = why;
                }
            }
            r.Add(new SharedSite(_siteNames[kv.Key], ids, reason));
        }
        r.Sort((a, b) => string.CompareOrdinal(a.Site, b.Site));
        return r;
    }

    /// <summary>Throws when a method carries the patches of several ids that were not declared together (ruling B3), or when a declaration names an unknown id or matches no site.</summary>
    public static void AssertOnePatchPerSite()
    {
        List<SharedSite> shared = SharedSites();
        foreach (SharedSite s in shared)
        {
            if (s.Reason.Length == 0)
            {
                throw new InvalidOperationException("One patch per site: " + s.Site + " is patched by " + string.Join(" + ", s.Ids) +
                                                    " and the combination is not declared (PatchRegistry.DeclareSharedSite). Remove the superseded patch or declare why both are needed.");
            }
        }
        foreach ((string[] declared, string _) in _declared)
        {
            foreach (string id in declared)
            {
                if (!_ids.Contains(id))
                {
                    throw new InvalidOperationException("Shared-site declaration names the unknown patch id '" + id + "'.");
                }
            }
            if (!shared.Any(s => s.Ids.SequenceEqual(declared, StringComparer.Ordinal)))
            {
                throw new InvalidOperationException("Shared-site declaration " + string.Join(" + ", declared) + " matches no site: these ids do not patch one method.");
            }
        }
    }

    private static void Add(PatchRecord record, MethodBase? target)
    {
        if (!_ids.Add(record.Id))
        {
            throw new InvalidOperationException("Patch id '" + record.Id + "' is registered twice.");
        }
        _records.Add(record);
        if (target == null)
        {
            return;
        }
        object key = SiteKey(target);
        if (!_sites.TryGetValue(key, out List<string>? ids))
        {
            ids = new List<string>(1);
            _sites[key] = ids;
            _siteNames[key] = SiteName(target);
        }
        ids.Add(record.Id);
    }

    /// <summary>An async / iterator method and the MoveNext of its state machine are one site.</summary>
    private static object SiteKey(MethodBase m)
    {
        Type? dt = m.DeclaringType;
        if (m.Name == "MoveNext" && dt != null && dt.IsNested && dt.Name.StartsWith("<", StringComparison.Ordinal))
        {
            return dt;
        }
        StateMachineAttribute? sm = m.GetCustomAttribute<StateMachineAttribute>(inherit: false);
        return sm != null ? sm.StateMachineType : m;
    }

    private static string SiteName(MethodBase m)
    {
        Type? dt = m.DeclaringType;
        if (m.Name == "MoveNext" && dt != null && dt.IsNested && dt.Name.StartsWith("<", StringComparison.Ordinal))
        {
            int end = dt.Name.IndexOf('>');
            return (dt.DeclaringType?.FullName ?? "?") + "::" + (end > 1 ? dt.Name.Substring(1, end - 1) : dt.Name);
        }
        return Describe(m);
    }

    private static void Require(string id, MethodBase? target)
    {
        if (target == null)
        {
            throw new MissingMethodException("Patch target for '" + id + "' was not found; the game assembly changed.");
        }
        if (_ids.Contains(id))
        {
            throw new InvalidOperationException("Patch id '" + id + "' is registered twice.");
        }
    }

    private static string Describe(MethodBase m)
    {
        return (m.DeclaringType?.FullName ?? "?") + "::" + m.Name;
    }

    public static MethodInfo M(Type type, string name)
    {
        return type.GetMethod(name, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new MissingMethodException(type.FullName, name);
    }
}
