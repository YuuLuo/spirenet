using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Obs.Vocab;

/// <summary>
/// "Printed variable" rule of obs-schema-v1 5.3.3: a numeric variable is only put on the wire when its name
/// appears as a placeholder in the entity's localisation template (static game knowledge, read from the
/// game's own LocManager tables with LocTable.GetRawText). The template parser and its per-(table, key) cache
/// are the driver's (SpireHost.Driver.LocText): the modules that fill decision contexts (event options) use the
/// same one, so that what a context says is printed and what the encoder puts on the wire cannot differ.
/// </summary>
public sealed class Templates
{
    /// <summary>Context (non-numeric / formatting) placeholders: the one list of the driver (SpireHost.Driver.LocText).</summary>
    public static HashSet<string> ContextNames => SpireHost.Driver.LocText.ContextNames;

    private static readonly string[] Empty = Array.Empty<string>();

    private readonly Dictionary<ModelId, string[]> _byModel = new Dictionary<ModelId, string[]>(ReferenceEqualityComparer.Instance);

    public static bool Exists(string table, string key) => SpireHost.Driver.LocText.Exists(table, key);

    public static string? Raw(string table, string key) => SpireHost.Driver.LocText.Raw(table, key);

    /// <summary>All placeholder names of one template (context names removed), in order of first appearance; empty if missing.</summary>
    public string[] Of(string table, string key) => SpireHost.Driver.LocText.Placeholders(table, key);

    public static string[] Extract(string raw) => SpireHost.Driver.LocText.Extract(raw);

    private string[] Union(params string[][] parts)
    {
        List<string> all = new List<string>();
        foreach (string[] p in parts)
        {
            foreach (string s in p)
            {
                if (!all.Contains(s))
                {
                    all.Add(s);
                }
            }
        }
        return all.Count == 0 ? Empty : all.ToArray();
    }

    /// <summary>Printed vars of a model (card / relic / potion / power / enchantment description templates, schema 5.3.3).</summary>
    public string[] PrintedVars(AbstractModel model)
    {
        ModelId id = model.Id;
        if (_byModel.TryGetValue(id, out string[]? v))
        {
            return v;
        }
        string e = id.Entry;
        v = model switch
        {
            CardModel => Of("cards", e + ".description"),
            RelicModel => Of("relics", e + ".description"),
            PotionModel => Of("potions", e + ".description"),
            PowerModel pw => PowerPrinted(pw, Exists("powers", e + ".smartDescription") ? Of("powers", e + ".smartDescription") : Of("powers", e + ".description")),
            EnchantmentModel => Without(Union(Of("enchantments", e + ".extraCardText"), Of("enchantments", e + ".description")), "Amount"),
            AfflictionModel => Without(Union(Of("afflictions", e + ".extraCardText"), Of("afflictions", e + ".description")), "Amount"),
            OrbModel => Exists("orbs", e + ".smartDescription") ? Of("orbs", e + ".smartDescription") : Of("orbs", e + ".description"),
            _ => Empty,
        };
        _byModel[id] = v;
        return v;
    }

    public const string PowerAmount = "Amount";

    /// <summary>
    /// A power's printed variables. Its <c>{Amount}</c> placeholder is the power's Amount. When the icon's number already
    /// is the Amount (StackType Counter and DisplayAmount not overridden: POWERS.amount) the placeholder is redundant and
    /// dropped (schema 5.5 "Amount 不重复输出"); otherwise - DisplayAmount overridden (SlothPower shows the cards played,
    /// its tooltip "You cannot play more than {Amount} cards" the limit; 16 such powers in v0.113) or a non-Counter
    /// power - the tooltip is the only place the Amount is shown, so it is kept and the encoder writes it as a VARS row
    /// named "Amount" (errata E-34).
    /// </summary>
    private static string[] PowerPrinted(PowerModel pw, string[] names)
    {
        if (Array.IndexOf(names, PowerAmount) < 0)
        {
            return names;
        }
        bool iconShowsAmount = pw.StackType == MegaCrit.Sts2.Core.Entities.Powers.PowerStackType.Counter && !OverridesDisplayAmount(pw.GetType());
        return iconShowsAmount ? Without(names, PowerAmount) : names;
    }

    private static readonly Dictionary<Type, bool> s_displayOverride = new Dictionary<Type, bool>();

    public static bool OverridesDisplayAmount(Type t)
    {
        if (!s_displayOverride.TryGetValue(t, out bool v))
        {
            System.Reflection.PropertyInfo? p = t.GetProperty(nameof(PowerModel.DisplayAmount), System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public);
            // (MethodInfo equality includes the reflected type, so compare declaring types only)
            v = p?.GetMethod == null || p.GetMethod.DeclaringType != typeof(PowerModel);
            s_displayOverride[t] = v;
        }
        return v;
    }

    private static string[] Without(string[] names, string drop)
    {
        int i = Array.IndexOf(names, drop);
        if (i < 0)
        {
            return names;
        }
        List<string> l = new List<string>(names);
        l.RemoveAt(i);
        return l.Count == 0 ? Empty : l.ToArray();
    }

    /// <summary>Placeholders of a LocString's own template (event option title/description, rest option description ...).</summary>
    public string[] Of(LocString? s)
    {
        if (s == null || s.IsEmpty)
        {
            return Empty;
        }
        return Of(s.LocTable, s.LocEntryKey);
    }
}
