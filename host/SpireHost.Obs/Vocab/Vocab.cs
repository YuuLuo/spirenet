using System;
using System.Collections.Generic;
using System.IO;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Obs.Vocab;

/// <summary>
/// Runtime view of vocab.json (schema 12): per category, id 0 = &lt;PAD&gt;, 1 = &lt;UNK&gt;, append-only.
/// Lookups on the encoder's hot path go through per-category dictionaries (strings) and, for models, a
/// reference-equality cache keyed by the model's <see cref="ModelId"/> instance (mutable copies share their
/// canonical model's ModelId object), so steady-state lookups cost one pointer-hash probe.
/// Names that are not in the vocabulary map to UNK (1) and are reported once on stderr (schema 12.2).
/// </summary>
public sealed class VocabSet
{
    public const string Pad = "<PAD>";
    public const string Unk = "<UNK>";
    public const int PadId = 0;
    public const int UnkId = 1;

    public sealed class Category
    {
        public readonly string Name;
        public readonly List<string> Entries = new List<string>();
        public readonly SortedSet<string> Retired = new SortedSet<string>(StringComparer.Ordinal);
        public readonly Dictionary<string, int> Index = new Dictionary<string, int>(StringComparer.Ordinal);
        public JObj? Meta;

        internal readonly Dictionary<ModelId, int> ByModelId = new Dictionary<ModelId, int>(ReferenceEqualityComparer.Instance);

        public Category(string name)
        {
            Name = name;
        }

        public int Count => Entries.Count;

        public void Add(string entry)
        {
            if (Index.ContainsKey(entry))
            {
                throw new InvalidOperationException($"vocab {Name}: duplicate entry {entry}");
            }
            Index[entry] = Entries.Count;
            Entries.Add(entry);
        }
    }

    private readonly Dictionary<string, Category> _cats = new Dictionary<string, Category>(StringComparer.Ordinal);
    private readonly HashSet<string> _reportedUnknown = new HashSet<string>(StringComparer.Ordinal);

    public string GameVersion = SpireHost.Boot.GameAssemblies.PinnedVersion;
    public string Sts2Sha256 = "";
    public int VocabVersion = 1;

    /// <summary>Canonical hash (header VOCAB_HASH); set by <see cref="Seal"/>.</summary>
    public int Hash { get; private set; }

    public long UnknownLookups { get; private set; }

    /// <summary>The unknown lookups ("category/name") since the list was last cleared (bounded; read by the frame audit of the self-tests).</summary>
    public readonly List<string> UnknownLog = new List<string>();

    public IEnumerable<Category> Categories => _cats.Values;

    public bool Has(string cat) => _cats.ContainsKey(cat);

    public Category this[string cat] => _cats[cat];

    public Category GetOrAdd(string cat)
    {
        if (!_cats.TryGetValue(cat, out Category? c))
        {
            c = new Category(cat);
            c.Add(Pad);
            c.Add(Unk);
            _cats[cat] = c;
        }
        return c;
    }

    public int Id(string cat, string? name)
    {
        if (name == null)
        {
            return PadId;
        }
        Category c = _cats[cat];
        if (c.Index.TryGetValue(name, out int id))
        {
            return id;
        }
        ReportUnknown(cat, name);
        return UnkId;
    }

    /// <summary>Vocab id of a model (by <c>Id.Entry</c>); null model = PAD.</summary>
    public int Model(string cat, AbstractModel? model)
    {
        if (model == null)
        {
            return PadId;
        }
        return ModelIdOf(cat, model.Id);
    }

    public int ModelIdOf(string cat, ModelId? id)
    {
        if (id is null)
        {
            return PadId;
        }
        Category c = _cats[cat];
        if (c.ByModelId.TryGetValue(id, out int v))
        {
            return v;
        }
        if (!c.Index.TryGetValue(id.Entry, out v))
        {
            ReportUnknown(cat, id.Entry);
            v = UnkId;
        }
        c.ByModelId[id] = v;
        return v;
    }

    private void ReportUnknown(string cat, string name)
    {
        UnknownLookups++;
        if (UnknownLog.Count < 64)
        {
            UnknownLog.Add(cat + "/" + name);
        }
        if (_reportedUnknown.Add(cat + "/" + name))
        {
            Console.Error.WriteLine($"[obs] vocab: '{name}' is not in category '{cat}' -> UNK (re-export the vocabulary)");
        }
    }

    // ---------------------------------------------------------------------------------------------- JSON

    public JObj ToJson()
    {
        JObj cats = new JObj();
        foreach (Category c in _cats.Values)
        {
            JObj o = new JObj();
            List<object?> entries = new List<object?>(c.Entries.Count);
            foreach (string e in c.Entries)
            {
                entries.Add(e);
            }
            o["entries"] = entries;
            List<object?> retired = new List<object?>();
            foreach (string r in c.Retired)
            {
                retired.Add(r);
            }
            o["retired"] = retired;
            if (c.Meta != null)
            {
                o["meta"] = c.Meta;
            }
            cats[c.Name] = o;
        }
        return new JObj
        {
            ["schema"] = 1L,
            ["game_version"] = GameVersion,
            ["sts2_sha256"] = Sts2Sha256,
            ["vocab_version"] = (long)VocabVersion,
            ["categories"] = cats,
        };
    }

    /// <summary>The parsed file this vocabulary was loaded from (hashed as is), or null for a freshly built one.</summary>
    public JObj? Raw;

    public void Seal()
    {
        Hash = CanonicalJson.Hash(Raw ?? ToJson());
    }

    public static VocabSet FromJson(object? root)
    {
        if (root is not JObj o || o.GetValueOrDefault("categories") is not JObj cats)
        {
            throw new InvalidDataException("vocab.json: missing 'categories'");
        }
        VocabSet v = new VocabSet
        {
            GameVersion = o.GetValueOrDefault("game_version") as string ?? "",
            Sts2Sha256 = o.GetValueOrDefault("sts2_sha256") as string ?? "",
            VocabVersion = (int)(o.GetValueOrDefault("vocab_version") as long? ?? 0),
        };
        foreach (KeyValuePair<string, object?> kv in cats)
        {
            JObj c = (JObj)kv.Value!;
            Category cat = new Category(kv.Key);
            foreach (object? e in (List<object?>)c["entries"]!)
            {
                cat.Add((string)e!);
            }
            if (cat.Entries.Count < 2 || cat.Entries[0] != Pad || cat.Entries[1] != Unk)
            {
                throw new InvalidDataException($"vocab category {kv.Key}: entries must start with <PAD>, <UNK>");
            }
            if (c.GetValueOrDefault("retired") is List<object?> retired)
            {
                foreach (object? r in retired)
                {
                    cat.Retired.Add((string)r!);
                }
            }
            cat.Meta = c.GetValueOrDefault("meta") as JObj;
            v._cats[kv.Key] = cat;
        }
        v.Raw = o;
        v.Seal();
        return v;
    }
}
