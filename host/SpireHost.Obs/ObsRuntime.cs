using System;
using System.Collections.Generic;
using System.IO;
using SpireHost.Boot;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs;

/// <summary>Raised when the shipped vocabulary cannot be used. Startup error in every mode (H1 integration, ruling B1).</summary>
public sealed class VocabularyException : Exception
{
    public VocabularyException(string message) : base(message)
    {
    }
}

/// <summary>Loads the vocabulary / catalog used by the encoder and builds encoders.</summary>
public static class ObsRuntime
{
    public sealed class Loaded
    {
        public VocabSet Vocab = null!;
        public int CatalogHash;
        public string Source = "";
        public string Dir = "";
        public Templates Templates = new Templates();
        public int MissingNames;
        public List<string> MissingExamples = new List<string>();
    }

    /// <summary>The vocabulary directory of this process: --vocab-dir, else the "vocab" folder next to the executable.</summary>
    public static string DefaultDir(string? dir) => Path.GetFullPath(dir ?? Path.Combine(AppContext.BaseDirectory, "vocab"));

    /// <summary>
    /// Loads vocab.json + catalog.json from <paramref name="dir"/> (default: the "vocab" folder next to the executable,
    /// which the build / make-dist copies from host\vocab). FAIL-CLOSED - a <see cref="VocabularyException"/> for:
    ///   * a missing vocab.json or catalog.json (there is no in-memory fallback vocabulary);
    ///   * a vocabulary exported from another game build (its sts2_sha256 is not the loaded sts2.dll), unless
    ///     <paramref name="allowMissingNames"/> (--export-vocab re-exports it for the loaded build);
    ///   * a catalog that belongs to another vocabulary (catalog.vocab_hash != VOCAB_HASH);
    ///   * names of the running game that the vocabulary does not know (they would encode as UNK), unless
    ///     <paramref name="allowMissingNames"/> - which only --export-vocab passes, the command that repairs exactly that.
    /// </summary>
    public static Loaded Load(string? dir, bool allowMissingNames = false)
    {
        string d = DefaultDir(dir);
        string vp = Path.Combine(d, "vocab.json");
        string cp = Path.Combine(d, "catalog.json");
        if (!File.Exists(vp))
        {
            throw new VocabularyException("vocabulary not found: " + vp + " (the host ships it in <exe dir>\\vocab; use --vocab-dir to point elsewhere)");
        }
        // --export-vocab (allowMissingNames) also starts without a catalog: it is the command that writes one
        bool noCatalog = allowMissingNames && !File.Exists(cp);
        if (!File.Exists(cp) && !noCatalog)
        {
            throw new VocabularyException("catalog not found: " + cp + " (generate it from your game with --export-vocab, see README.md)");
        }
        Loaded l = new Loaded { Dir = d, Source = vp };
        l.Vocab = VocabSet.FromJson(CanonicalJson.ParseFile(vp));
        object? catalog = noCatalog ? null : CanonicalJson.ParseFile(cp);
        l.CatalogHash = noCatalog ? 0 : CanonicalJson.Hash(catalog);
        if (!allowMissingNames && !string.IsNullOrEmpty(l.Vocab.Sts2Sha256) && !string.Equals(l.Vocab.Sts2Sha256, GameAssemblies.Sts2Sha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new VocabularyException("vocabulary " + vp + " was exported from sts2.dll " + l.Vocab.Sts2Sha256 + ", the loaded game is " + GameAssemblies.Sts2Sha256);
        }
        if (catalog is JObj root && root.TryGetValue("vocab_hash", out object? vh) && vh != null)
        {
            long recorded = Convert.ToInt64(vh);
            if (unchecked((int)recorded) != l.Vocab.Hash)
            {
                throw new VocabularyException($"catalog {cp} belongs to another vocabulary: its vocab_hash is 0x{unchecked((uint)recorded):X8}, vocab.json hashes to 0x{unchecked((uint)l.Vocab.Hash):X8}");
            }
        }
        else if (!noCatalog)
        {
            throw new VocabularyException("catalog " + cp + " has no vocab_hash");
        }
        // names of the running game (models, enums, localisation keys, the keys the modules use) that the file does not know
        SortedDictionary<string, List<string>> current = VocabExport.CurrentNames(out _);
        foreach (KeyValuePair<string, List<string>> kv in current)
        {
            if (!l.Vocab.Has(kv.Key))
            {
                l.MissingNames += kv.Value.Count;
                if (l.MissingExamples.Count < 8)
                {
                    l.MissingExamples.Add("category " + kv.Key);
                }
                continue;
            }
            foreach (string n in kv.Value)
            {
                if (!l.Vocab[kv.Key].Index.ContainsKey(n))
                {
                    l.MissingNames++;
                    if (l.MissingExamples.Count < 8)
                    {
                        l.MissingExamples.Add(kv.Key + ":" + n);
                    }
                }
            }
        }
        if (l.MissingNames > 0 && !allowMissingNames)
        {
            throw new VocabularyException($"vocabulary {vp} lacks {l.MissingNames} name(s) the running game uses (they would encode as UNK): " +
                                          string.Join(", ", l.MissingExamples) + " - re-export with --export-vocab DIR --merge " + vp);
        }
        return l;
    }

    public static ObsEncoder NewEncoder(Loaded l)
    {
        return new ObsEncoder(l.Vocab, l.Templates) { CatalogHash = l.CatalogHash };
    }
}
