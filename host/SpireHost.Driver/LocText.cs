using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Localization.DynamicVars;

namespace SpireHost.Driver;

/// <summary>
/// "Which numbers does a localised text print?" - the one implementation shared by the modules that fill decision
/// contexts (event pages / options) and by the observation encoder (obs-schema-v1 5.3.3: a numeric variable is on
/// the wire only when its name appears as a placeholder in the template). Static game knowledge: the raw
/// templates come from the game's own LocManager tables; no text is ever formatted and nothing is written into
/// a LocString. Engine thread only (the cache is not synchronised).
/// </summary>
public static class LocText
{
    /// <summary>Context (non-numeric / formatting) placeholders, CardModel.cs:1236-1248, AbstractIntent.cs:74 and friends.</summary>
    public static readonly HashSet<string> ContextNames = new HashSet<string>(StringComparer.Ordinal)
    {
        "InCombat", "IfUpgraded", "energyPrefix", "singleStarIcon", "TargetType", "GainsBlock", "IsOstyAlive", "OnTable",
        "IsTargeting", "IsMultiplayer", "Character", "ExtraText",
    };

    private static readonly Regex Placeholder = new Regex(@"\{([A-Za-z_][A-Za-z0-9_]*)", RegexOptions.CultureInvariant | RegexOptions.Compiled);
    private static readonly string[] Empty = Array.Empty<string>();
    private static readonly Dictionary<string, string[]> _cache = new Dictionary<string, string[]>(StringComparer.Ordinal);

    private static LocTable? TryTable(string table)
    {
        LocManager? loc = LocManager.Instance;
        if (loc == null)
        {
            return null;
        }
        // LocManager.GetTable throws for a missing table; read its dictionary instead (publicized reference)
        return loc._tables.TryGetValue(table, out LocTable? t) ? t : null;
    }

    public static bool Exists(string table, string key)
    {
        LocTable? t = TryTable(table);
        return t != null && t.HasEntry(key);
    }

    /// <summary>The raw template text of a localisation entry, or null when the table / entry does not exist.</summary>
    public static string? Raw(string table, string key)
    {
        LocTable? t = TryTable(table);
        if (t == null || !t.HasEntry(key))
        {
            return null;
        }
        return t.GetRawText(key);
    }

    /// <summary>All placeholder names of a raw template (context names removed), in order of first appearance.</summary>
    public static string[] Extract(string raw)
    {
        List<string> names = new List<string>();
        foreach (Match m in Placeholder.Matches(raw))
        {
            string n = m.Groups[1].Value;
            if (!ContextNames.Contains(n) && !names.Contains(n))
            {
                names.Add(n);
            }
        }
        return names.Count == 0 ? Empty : names.ToArray();
    }

    /// <summary>Placeholder names of one template, cached per (table, key); empty when the entry is missing.</summary>
    public static string[] Placeholders(string table, string key)
    {
        string ck = table + "/" + key;
        if (_cache.TryGetValue(ck, out string[]? v))
        {
            return v;
        }
        string? raw = Raw(table, key);
        v = raw == null ? Empty : Extract(raw);
        _cache[ck] = v;
        return v;
    }

    /// <summary>Placeholder names of a LocString's own template (event option title / description, rest option description ...).</summary>
    public static string[] Placeholders(LocString? s)
    {
        if (s == null || s.IsEmpty)
        {
            return Empty;
        }
        return Placeholders(s.LocTable, s.LocEntryKey);
    }

    public static bool IsNumeric(DynamicVar dv) => dv is not BoolVar && dv is not StringVar;

    /// <summary>
    /// The numbers a LocString-rendered text prints: every placeholder of the text's template that resolves to a number -
    /// a DynamicVar of <paramref name="vars"/> (what the UI adds with DynamicVars.AddTo, NEventOptionButton.cs:181-184) or
    /// a numeric variable of the LocString itself. Appends (name, shown value) pairs in template order; a name that is
    /// already in <paramref name="into"/> is kept (first occurrence wins).
    /// </summary>
    public static void PrintedVars(DynamicVarSet? vars, LocString? s, List<KeyValuePair<string, int>> into)
    {
        if (s == null || s.IsEmpty)
        {
            return;
        }
        string[] names = Placeholders(s);
        for (int i = 0; i < names.Length; i++)
        {
            string n = names[i];
            bool dup = false;
            for (int k = 0; k < into.Count; k++)
            {
                if (into[k].Key == n)
                {
                    dup = true;
                    break;
                }
            }
            if (dup)
            {
                continue;
            }
            if (vars != null && vars.TryGetValue(n, out DynamicVar? dv) && IsNumeric(dv))
            {
                into.Add(new KeyValuePair<string, int>(n, (int)dv.PreviewValue));
                continue;
            }
            if (s.Variables.TryGetValue(n, out object? o))
            {
                int? val = o switch
                {
                    DynamicVar d when IsNumeric(d) => (int)d.PreviewValue,
                    int x => x,
                    long x => (int)x,
                    decimal x => (int)x,
                    double x => (int)x,
                    float x => (int)x,
                    _ => null,
                };
                if (val.HasValue)
                {
                    into.Add(new KeyValuePair<string, int>(n, val.Value));
                }
            }
        }
    }
}
