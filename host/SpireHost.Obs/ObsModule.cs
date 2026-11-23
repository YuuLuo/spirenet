using System;
using System.Collections.Generic;
using SpireHost.Driver;
using SpireHost.Obs.Encoding;

namespace SpireHost.Obs;

/// <summary>
/// The observation module. Registered in EVERY mode (serve, fuzz, determinism, replay, the oracle tools, every
/// self-test): ruling B1 of the H1 integration.
///
/// Register (engine thread, after boot, before the first episode):
///   1. loads the shipped vocabulary + catalog (ObsRuntime.Load: missing / mismatching / incomplete = startup error);
///   2. registers the CANONICAL CARD-CLASS PROVIDER of the run driver (combat C7 = obs O1): the content key of the
///      encoder (obs-schema-v1 5.2.3) orders the candidates of every pile-sourced CARD_SELECT prompt, in every mode,
///      so that a run is the same game whether or not frames are encoded;
///   3. records the vocabulary identity in the registry (action logs and traces carry VOCAB_HASH);
///   4. registers the module counters "obs.*".
/// It installs no Harmony patch and owns no room handler. The encoder reads decision contexts and the white-listed
/// public game API only: no reference to another module (ruling D6).
///
/// It also provides the encoder (Encoding\ObsEncoder*.cs), the protocol server (--serve), vocab / catalog export
/// (--export-vocab) and the verification commands (obs --selftest, --leak-test, --golden, --bench-encode, --sample-frames).
/// </summary>
public sealed class ObsModule : IDriverModule
{
    private readonly string? _vocabDir;
    private readonly bool _allowMissingNames;

    /// <param name="vocabDir">--vocab-dir, or null for &lt;exe dir&gt;\vocab.</param>
    /// <param name="allowMissingNames">Only for --export-vocab (the command that adds the missing names).</param>
    public ObsModule(string? vocabDir = null, bool allowMissingNames = false)
    {
        _vocabDir = vocabDir;
        _allowMissingNames = allowMissingNames;
    }

    public string Name => "obs";

    /// <summary>The vocabulary loaded at registration (one per process).</summary>
    public static ObsRuntime.Loaded Current { get; private set; } = null!;

    /// <summary>The encoder behind the canonical card-class provider (a private scratch encoder; no frame state of other encoders is touched).</summary>
    public static ObsEncoder ClassEncoder { get; private set; } = null!;

    public const string ProviderName = "obs:content-key";

    public void Register(DriverRegistry registry)
    {
        ObsRuntime.Loaded l = ObsRuntime.Load(_vocabDir, _allowMissingNames);
        Current = l;
        ClassEncoder = ObsRuntime.NewEncoder(l);
        registry.SetCanonicalCardClasses(ClassEncoder.CanonicalClasses, ProviderName);
        registry.SetVocabulary(unchecked((uint)l.Vocab.Hash), unchecked((uint)l.CatalogHash), l.Source);
        registry.SetModuleCounters(new ObsCounters());
    }
}

/// <summary>"obs.*": how often the class provider ran and what the vocabulary could not name.</summary>
internal sealed class ObsCounters : IModuleCounters
{
    public string Module => "obs";

    public void Read(IDictionary<string, long> c)
    {
        c["canonical_class_calls"] = ObsEncoder.CanonicalClassCalls;
        c["canonical_class_cards"] = ObsEncoder.CanonicalClassCards;
        c["vocab_unk_lookups"] = ObsModule.Current.Vocab.UnknownLookups;
        c["frames_encoded"] = ObsEncoder.FramesEncodedTotal;
    }
}
