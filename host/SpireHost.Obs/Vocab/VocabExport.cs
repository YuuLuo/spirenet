using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Entities.Ascension;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Enchantments;
using MegaCrit.Sts2.Core.Entities.FoolArrows;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.Powers;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Combat;

namespace SpireHost.Obs.Vocab;

/// <summary>
/// vocab.json / catalog.json export from the game's model database and localisation tables
/// (obs-schema-v1 section 12, decisions in schema-errata-v1 section V).
///
/// Ordering: per category, 0 = &lt;PAD&gt;, 1 = &lt;UNK&gt;, then (first export) model / string categories in
/// StringComparer.Ordinal order of their names and enum categories in ascending enum value order.
/// Merge (game update): old entries keep their ids, new names are appended in ordinal order, vanished names are
/// marked retired (never deleted, never reused); vocab_version increments only when something changed.
/// </summary>
public static class VocabExport
{
    /// <summary>vocab category -> ModelId.Category of the game (ModelDb.GetCategory: slug of the direct AbstractModel subclass).</summary>
    public static readonly (string Vocab, string Game)[] ModelCategories =
    {
        ("card", "CARD"), ("relic", "RELIC"), ("potion", "POTION"), ("power", "POWER"), ("monster", "MONSTER"),
        ("encounter", "ENCOUNTER"), ("event", "EVENT"), ("enchantment", "ENCHANTMENT"), ("affliction", "AFFLICTION"),
        ("orb", "ORB"), ("act", "ACT"), ("character", "CHARACTER"), ("modifier", "MODIFIER"), ("card_pool", "CARD_POOL"),
        ("relic_pool", "RELIC_POOL"), ("potion_pool", "POTION_POOL"),
    };

    public static readonly (string Name, Type Type)[] EnumCategories =
    {
        ("CardType", typeof(CardType)), ("CardRarity", typeof(CardRarity)), ("TargetType", typeof(TargetType)),
        ("CardKeyword", typeof(CardKeyword)), ("CardTag", typeof(CardTag)), ("CardCostColor", typeof(CardCostColor)),
        ("PileType", typeof(PileType)), ("UnplayableReason", typeof(UnplayableReason)), ("OrbEvokeType", typeof(OrbEvokeType)),
        ("CardMultiplayerConstraint", typeof(CardMultiplayerConstraint)), ("IntentType", typeof(IntentType)),
        ("PowerType", typeof(PowerType)), ("PowerStackType", typeof(PowerStackType)), ("PowerInstanceType", typeof(PowerInstanceType)),
        ("RelicRarity", typeof(RelicRarity)), ("RelicStatus", typeof(RelicStatus)), ("PotionRarity", typeof(PotionRarity)),
        ("PotionUsage", typeof(PotionUsage)), ("EnchantmentStatus", typeof(EnchantmentStatus)), ("HpDisplay", typeof(HpDisplay)),
        ("CombatSide", typeof(CombatSide)), ("PlayerTurnPhase", typeof(PlayerTurnPhase)), ("MapPointType", typeof(MapPointType)),
        ("MapPointState", typeof(MapPointState)), ("RoomType", typeof(RoomType)), ("RewardType", typeof(RewardType)),
        ("GameMode", typeof(GameMode)), ("AscensionLevel", typeof(AscensionLevel)), ("ArrowStatus", typeof(ArrowStatus)),
        ("CrystalSphereItemType", typeof(CrystalSphereItemType)), ("CrystalSphereToolType", typeof(CrystalSphereMinigame.CrystalSphereToolType)),
        ("StaticHoverTip", typeof(StaticHoverTip)),
    };

    public const string RelicOptionKey = "RELIC_OPTION";
    public const string ProceedOptionKey = "PROCEED";
    public const string DialogueSuffix = ".dialogue";

    private static readonly Regex DialogueKey = new Regex(@"^(\w+)\.dialogue\.\d+$", RegexOptions.CultureInvariant);

    /// <summary>
    /// The vocab name of an event option (errata V-6): its TextKey when that is in the vocabulary; otherwise the structural
    /// key - "&lt;EVENT&gt;.dialogue" for dialogue buttons, RELIC_OPTION for an option offering a relic (EventOption.FromRelic,
    /// DollRoom's relic choices whose TextKey is the relic's title text, Models.Events/DollRoom.cs:137); else the TextKey (UNK).
    /// </summary>
    public static string EventOptionName(VocabSet v, string textKey, bool offersRelic)
    {
        VocabSet.Category c = v["event_option_key"];
        if (c.Index.ContainsKey(textKey))
        {
            return textKey;
        }
        Match m = DialogueKey.Match(textKey);
        if (m.Success && c.Index.ContainsKey(m.Groups[1].Value + DialogueSuffix))
        {
            return m.Groups[1].Value + DialogueSuffix;
        }
        return offersRelic ? RelicOptionKey : textKey;
    }

    public static readonly string[] ShopSections = { "CARD_REMOVAL", "CHARACTER_CARD", "COLORLESS_CARD", "POTION", "RELIC" };

    /// <summary>
    /// The names of category confirm_key: the two button keys NO / YES of CONFIRM options, EVERY ConfirmContext.Key a
    /// module uses - taken from the driver's registry of them (SpireHost.Driver.ConfirmKeys.All; obs request O7), so a
    /// new key cannot be forgotten here - and "trial_double_down", the documentation example of the stage-0 contract
    /// that vocabulary v1 / v2 already contain (kept so that the merge retires nothing).
    /// </summary>
    public static string[] ConfirmKeys
    {
        get
        {
            List<string> keys = new List<string> { SpireHost.Driver.ConfirmKeys.No, SpireHost.Driver.ConfirmKeys.Yes, "trial_double_down" };
            foreach (string k in SpireHost.Driver.ConfirmKeys.All)
            {
                if (!keys.Contains(k))
                {
                    keys.Add(k);
                }
            }
            return keys.ToArray();
        }
    }

    /// <summary>
    /// The fixed headers of the two selection screens whose entry point has no CardSelectorPrefs (CardSelectContext.PromptTable /
    /// PromptKey, contract C2): NChooseACardSelectionScreen.cs:164 and CardSelectCmd.cs:641. Names of category prompt_key.
    /// </summary>
    public static readonly string[] FixedPromptKeys = { "gameplay_ui/CHOOSE_CARD_HEADER", "gameplay_ui/CHOOSE_CARD_UPGRADE_HEADER" };

    /// <summary>
    /// Event page descriptions (GLOBAL event_page_key = EventModel.Description.LocEntryKey) that the game sets from keys of
    /// the "events" table OUTSIDE the pattern "&lt;EVENT&gt;.pages.&lt;PAGE&gt;.description" which the category is enumerated with.
    /// Before the H1 integration these four encoded as UNK (found by the UNK-aware frame audit on the all-events sweep):
    ///   GENERIC.youAreDead.description                                  Models/EventModel.cs:213 (the owner is dead)
    ///   THE_LANTERN_KEY.pages.DONE.options.RETURN_THE_KEY.description   Models.Events/TheLanternKey.cs:36 (an option's text as the page)
    ///   TRIAL.trialFormat                                               Models.Events/Trial.cs:110 (wrapper around the trial's story;
    ///                                                                   which trial it is shows in the option keys)
    ///   TRIAL.trialResult                                               Models.Events/Trial.cs:202 (wrapper around the verdict)
    /// Each must exist in the game's localisation (the export fails otherwise: the game changed).
    /// Not listed: Neow's "NEOW.EVENT.description" (Models.Events/Neow.cs:47) - only for runs with modifiers (custom runs,
    /// which the host does not start) and absent from the localisation tables.
    /// </summary>
    public static readonly string[] ExtraEventPageKeys =
    {
        "GENERIC.youAreDead.description", "THE_LANTERN_KEY.pages.DONE.options.RETURN_THE_KEY.description", "TRIAL.trialFormat", "TRIAL.trialResult",
    };

    private static readonly Regex EventOptionKey = new Regex(@"^(\w+)\.pages\.(\w+)\.options\.(\w+)\.title$", RegexOptions.CultureInvariant);
    private static readonly Regex EventPageKey = new Regex(@"^(\w+)\.pages\.(\w+)\.description$", RegexOptions.CultureInvariant);

    public static bool IsMock(AbstractModel m)
    {
        Type t = m.GetType();
        return m.IsMock || (t.Namespace?.Contains(".Mocks", StringComparison.Ordinal) ?? false) || t.Name.StartsWith("Mock", StringComparison.Ordinal);
    }

    /// <summary>Current names per category (no PAD/UNK), each list in its canonical first-export order.</summary>
    public static SortedDictionary<string, List<string>> CurrentNames(out JObj varMeta)
    {
        SortedDictionary<string, List<string>> cats = new SortedDictionary<string, List<string>>(StringComparer.Ordinal);

        // ---- model categories ----
        Dictionary<string, SortedSet<string>> models = new Dictionary<string, SortedSet<string>>(StringComparer.Ordinal);
        foreach ((string v, string g) in ModelCategories)
        {
            models[g] = new SortedSet<string>(StringComparer.Ordinal);
        }
        foreach (AbstractModel m in ModelDb.All)
        {
            if (models.TryGetValue(m.Id.Category, out SortedSet<string>? set) && !IsMock(m))
            {
                set.Add(m.Id.Entry);
            }
        }
        foreach ((string v, string g) in ModelCategories)
        {
            cats[v] = models[g].ToList();
        }

        // ---- enum categories ----
        foreach ((string name, Type type) in EnumCategories)
        {
            cats["enum:" + name] = EnumNamesByValue(type);
        }

        // ---- string categories ----
        cats["var_name"] = VarNames(out varMeta);
        SortedSet<string> optionKeys = new SortedSet<string>(StringComparer.Ordinal);
        SortedSet<string> pageKeys = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string table in new[] { "events", "ancients" })
        {
            foreach (string key in TableKeys(table))
            {
                Match mo = EventOptionKey.Match(key);
                if (mo.Success)
                {
                    optionKeys.Add(key.Substring(0, key.Length - ".title".Length));
                    continue;
                }
                if (EventPageKey.IsMatch(key) && !key.Contains(".options.", StringComparison.Ordinal))
                {
                    pageKeys.Add(key);
                }
            }
        }
        // Relic options (EventModel.RelicOption -> EventOption.FromRelic, Models/EventModel.cs:490) carry a text key
        // "<EVENT>.pages.<PAGE>.options.<RELIC>" that has no localisation entry: they are encoded as RELIC_OPTION and the
        // relic itself is the option's ref (errata V-6). Ancients' pages have no loc entries either: add the structural ones.
        optionKeys.Add(RelicOptionKey);
        // structural keys of options whose TextKey is not a loc entry (errata V-6): AncientEventModel's / TheArchitect's
        // "PROCEED" (Models/AncientEventModel.cs:172, Models.Events/TheArchitect.cs:341), the Architect's dialogue buttons
        // "<EVENT>.dialogue.<line>" (TheArchitect.cs:336, line index dropped) and its empty page "PROCEED.description" (:38)
        optionKeys.Add(ProceedOptionKey);
        pageKeys.Add("PROCEED.description");
        List<string> eventTable = TableKeys("events");
        foreach (string extra in ExtraEventPageKeys)
        {
            if (!eventTable.Contains(extra))
            {
                throw new InvalidOperationException("vocabulary export: the event page key '" + extra + "' is not in the game's localisation table 'events' any more");
            }
            pageKeys.Add(extra);
        }
        foreach (AbstractModel m in ModelsOf("EVENT"))
        {
            if (m is AncientEventModel)
            {
                pageKeys.Add(m.Id.Entry + ".pages.INITIAL.description");
                pageKeys.Add(m.Id.Entry + ".pages.DONE.description");
            }
            if (m is AncientEventModel || m.Id.Entry == "THE_ARCHITECT")
            {
                optionKeys.Add(m.Id.Entry + DialogueSuffix);
            }
        }
        cats["event_option_key"] = optionKeys.ToList();
        cats["event_page"] = pageKeys.ToList();
        cats["rest_option"] = PrefixedIds("rest_site_ui", "OPTION_", ".name");
        cats["card_reward_alt"] = PrefixedIds("card_reward_ui", "OPTION_", ".name");
        SortedSet<string> prompts = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string key in TableKeys("card_selection"))
        {
            prompts.Add("card_selection/" + key);
        }
        foreach (string table in AllTables())
        {
            foreach (string key in TableKeys(table))
            {
                if (key.EndsWith(".selectionScreenPrompt", StringComparison.Ordinal) || key.EndsWith(".discardSelectionPrompt", StringComparison.Ordinal) || key.EndsWith(".selectionPrompt", StringComparison.Ordinal))
                {
                    prompts.Add(table + "/" + key);
                }
            }
        }
        foreach (string fixedKey in FixedPromptKeys)
        {
            prompts.Add(fixedKey);
        }
        cats["prompt_key"] = prompts.ToList();
        cats["confirm_key"] = ConfirmKeys.OrderBy(x => x, StringComparer.Ordinal).ToList();
        // slot names: the fixed layouts (EncounterModel.Slots) + every literal the game hands to a creature as its slot
        SortedSet<string> slots = SlotNameScan.Scan(out _);
        foreach (EncounterModel e in ModelDb.AllEncounters)
        {
            foreach (string s in e.Slots)
            {
                if (!string.IsNullOrEmpty(s))
                {
                    slots.Add(s);
                }
            }
        }
        cats["slot_name"] = slots.ToList();
        cats["shop_section"] = ShopSections.OrderBy(x => x, StringComparer.Ordinal).ToList();
        return cats;
    }

    public static List<string> EnumNamesByValue(Type type)
    {
        List<(long, string)> vals = new List<(long, string)>();
        foreach (object v in Enum.GetValues(type))
        {
            vals.Add((Convert.ToInt64(v), Enum.GetName(type, v)!));
        }
        vals.Sort((a, b) => a.Item1 != b.Item1 ? a.Item1.CompareTo(b.Item1) : string.CompareOrdinal(a.Item2, b.Item2));
        List<string> names = new List<string>();
        foreach ((long _, string n) in vals)
        {
            if (!names.Contains(n))
            {
                names.Add(n);
            }
        }
        return names;
    }

    private static IEnumerable<string> AllTables()
    {
        LocManager loc = LocManager.Instance;
        return loc._tables.Keys.OrderBy(x => x, StringComparer.Ordinal).ToList();
    }

    private static List<string> TableKeys(string table)
    {
        LocManager loc = LocManager.Instance;
        if (!loc._tables.TryGetValue(table, out LocTable? t))
        {
            return new List<string>();
        }
        return t.Keys.OrderBy(x => x, StringComparer.Ordinal).ToList();
    }

    private static List<string> PrefixedIds(string table, string prefix, string suffix)
    {
        SortedSet<string> ids = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string k in TableKeys(table))
        {
            if (k.StartsWith(prefix, StringComparison.Ordinal) && k.EndsWith(suffix, StringComparison.Ordinal))
            {
                ids.Add(k.Substring(prefix.Length, k.Length - prefix.Length - suffix.Length));
            }
        }
        return ids.ToList();
    }

    private static IEnumerable<AbstractModel> ModelsOf(string gameCategory)
    {
        return ModelDb.All.Where(m => m.Id.Category == gameCategory && !IsMock(m)).OrderBy(m => m.Id.Entry, StringComparer.Ordinal);
    }

    /// <summary>DynamicVarSet of a canonical model (null for categories without one).</summary>
    public static DynamicVarSet? VarsOf(AbstractModel m) => m switch
    {
        CardModel c => c.DynamicVars,
        RelicModel r => r.DynamicVars,
        PotionModel p => p.DynamicVars,
        PowerModel w => w.DynamicVars,
        EnchantmentModel e => e.DynamicVars,
        EventModel ev => ev.DynamicVars,
        _ => null,
    };

    private static List<string> VarNames(out JObj meta)
    {
        SortedSet<string> names = new SortedSet<string>(StringComparer.Ordinal);
        meta = new JObj();
        foreach (string cat in new[] { "CARD", "RELIC", "POTION", "POWER", "ENCHANTMENT", "EVENT" })
        {
            foreach (AbstractModel m in ModelsOf(cat))
            {
                DynamicVarSet? set = VarsOf(m);
                if (set == null)
                {
                    continue;
                }
                foreach (string key in set.Keys.OrderBy(x => x, StringComparer.Ordinal))
                {
                    if (Templates.ContextNames.Contains(key))
                    {
                        continue;
                    }
                    names.Add(key);
                    if (!meta.ContainsKey(key))
                    {
                        meta[key] = VarMeta(set[key]);
                    }
                }
            }
        }
        // a power's {Amount} (not a DynamicVar; printed by tooltips whose icon shows something else, errata E-34)
        names.Add(Templates.PowerAmount);
        if (!meta.ContainsKey(Templates.PowerAmount))
        {
            meta[Templates.PowerAmount] = new JObj { ["var_class"] = "PowerModel.Amount" };
        }
        foreach (string table in new[] { "events", "ancients", "rest_site_ui", "card_reward_ui", "gameplay_ui" })
        {
            if (!LocManager.Instance._tables.TryGetValue(table, out LocTable? t))
            {
                continue;
            }
            foreach (string key in TableKeys(table))
            {
                foreach (string n in Templates.Extract(t.GetRawText(key)))
                {
                    names.Add(n);
                }
            }
        }
        return names.ToList();
    }

    public static JObj VarMeta(DynamicVar v)
    {
        Type t = v.GetType();
        JObj o = new JObj();
        string cls = t.Name;
        int tick = cls.IndexOf('`');
        if (tick >= 0)
        {
            cls = cls.Substring(0, tick);
        }
        o["var_class"] = cls;
        if (t.IsGenericType && t.GetGenericArguments().Length == 1 && typeof(PowerModel).IsAssignableFrom(t.GetGenericArguments()[0]))
        {
            o["power"] = ModelDb.GetId(t.GetGenericArguments()[0]).Entry;
        }
        return o;
    }

    // ---------------------------------------------------------------------------------------------- vocab build / merge

    public static VocabSet BuildFresh(string sts2Sha)
    {
        SortedDictionary<string, List<string>> cur = CurrentNames(out JObj varMeta);
        VocabSet v = new VocabSet { Sts2Sha256 = sts2Sha, VocabVersion = 1 };
        foreach (KeyValuePair<string, List<string>> kv in cur)
        {
            VocabSet.Category c = v.GetOrAdd(kv.Key);
            foreach (string n in kv.Value)
            {
                c.Add(n);
            }
        }
        v["var_name"].Meta = FilterMeta(varMeta, v["var_name"]);
        v.Seal();
        return v;
    }

    private static JObj FilterMeta(JObj meta, VocabSet.Category cat)
    {
        JObj o = new JObj();
        foreach (KeyValuePair<string, object?> kv in meta)
        {
            if (cat.Index.ContainsKey(kv.Key))
            {
                o[kv.Key] = kv.Value;
            }
        }
        return o;
    }

    /// <summary>Append-only merge of the current game names into an existing vocabulary (schema 12.2).</summary>
    public static VocabSet Merge(VocabSet old, string sts2Sha, out List<string> report)
    {
        report = new List<string>();
        SortedDictionary<string, List<string>> cur = CurrentNames(out JObj varMeta);
        VocabSet v = VocabSet.FromJson(old.Raw ?? old.ToJson());
        v.Raw = null;
        bool changed = false;
        foreach (KeyValuePair<string, List<string>> kv in cur)
        {
            bool existed = v.Has(kv.Key);
            VocabSet.Category c = v.GetOrAdd(kv.Key);
            if (!existed)
            {
                changed = true;
                report.Add("new category " + kv.Key);
            }
            HashSet<string> now = new HashSet<string>(kv.Value, StringComparer.Ordinal);
            foreach (string n in kv.Value.Where(n => !c.Index.ContainsKey(n)).OrderBy(n => n, StringComparer.Ordinal))
            {
                c.Add(n);
                changed = true;
                report.Add($"{kv.Key}: + {n}");
            }
            for (int i = 2; i < c.Entries.Count; i++)
            {
                string n = c.Entries[i];
                if (!now.Contains(n) && c.Retired.Add(n))
                {
                    changed = true;
                    report.Add($"{kv.Key}: retired {n}");
                }
            }
        }
        if (v.Has("var_name"))
        {
            JObj meta = v["var_name"].Meta ?? new JObj();
            foreach (KeyValuePair<string, object?> kv in FilterMeta(varMeta, v["var_name"]))
            {
                if (!meta.ContainsKey(kv.Key))
                {
                    meta[kv.Key] = kv.Value;
                    changed = true;
                }
            }
            v["var_name"].Meta = meta;
        }
        if (changed)
        {
            v.VocabVersion = old.VocabVersion + 1;
        }
        v.Sts2Sha256 = sts2Sha;
        v.GameVersion = SpireHost.Boot.GameAssemblies.PinnedVersion;
        v.Seal();
        return v;
    }

    public static void Write(string dir, VocabSet vocab, JObj catalog)
    {
        Directory.CreateDirectory(dir);
        JObj vj = vocab.ToJson();
        File.WriteAllText(Path.Combine(dir, "vocab.json"), CanonicalJson.Pretty(vj), new System.Text.UTF8Encoding(false));
        File.WriteAllText(Path.Combine(dir, "catalog.json"), CanonicalJson.Pretty(catalog), new System.Text.UTF8Encoding(false));
    }
}
