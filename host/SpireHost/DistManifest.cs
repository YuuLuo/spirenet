using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using SpireHost.Boot;
using SpireHost.Driver;
using SpireHost.Patches;

namespace SpireHost;

/// <summary>
/// dist.json: the identity of an immutable deployment directory (H1 integration, ruling A5).
///
/// host\make-dist copies the build output, the vocabulary and the three documents into
/// host\dist\&lt;yyyymmdd-HHmmss&gt;-&lt;label&gt;\ and then runs "SpireHost --write-dist-manifest --label L --stamp S" FROM that
/// directory; the executable describes itself (it is the only one that knows its patch set and vocabulary) and
/// writes dist.json next to itself - once: an existing dist.json is never overwritten.
///
/// From then on EVERY start of an executable that has a dist.json next to it verifies the directory before it runs a
/// command (<see cref="VerifyAtStartup"/>): every shipped file still has the recorded SHA256 (no file changed, added or
/// removed), the game assembly is the recorded one, and the live patch set, vocabulary and contract version are the
/// recorded ones. A difference is a startup error (exit 3): a dist directory is never modified after creation.
///
/// The game itself is not part of a dist: the executable finds it by walking up from its own directory to the first
/// "game\sts2.dll" (host\dist\&lt;stamp&gt;\ -> host\game), or through --game-dir / SPIREHOST_GAME_DIR; the pinned SHA256 of
/// sts2.dll is checked either way.
/// </summary>
public static class DistManifest
{
    public const string FileName = "dist.json";

    public static string Dir => Path.GetFullPath(AppContext.BaseDirectory);

    public static string ManifestPath => Path.Combine(Dir, FileName);

    /// <summary>The apphost the build produces: SpireHost.exe on Windows, SpireHost on Linux.</summary>
    private static string ExeName => OperatingSystem.IsWindows() ? "SpireHost.exe" : "SpireHost";

    public static bool Present => File.Exists(ManifestPath);

    /// <summary>SHA256 of every file under the executable's directory except dist.json, keyed by relative path with '/'.</summary>
    public static SortedDictionary<string, string> HashFiles()
    {
        SortedDictionary<string, string> r = new SortedDictionary<string, string>(StringComparer.Ordinal);
        string root = Dir;
        foreach (string f in Directory.GetFiles(root, "*", SearchOption.AllDirectories))
        {
            string rel = Path.GetRelativePath(root, f).Replace('\\', '/');
            if (rel == FileName)
            {
                continue;
            }
            using FileStream fs = new FileStream(f, FileMode.Open, FileAccess.Read, FileShare.Read);
            r[rel] = Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
        }
        return r;
    }

    private static SortedDictionary<string, int> ByCategory(IReadOnlyList<PatchRecord> records)
    {
        SortedDictionary<string, int> by = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (PatchCategory c in Enum.GetValues<PatchCategory>())
        {
            by[c.ToString()] = records.Count(r => r.Category == c);
        }
        return by;
    }

    /// <summary>--write-dist-manifest --label L --stamp S (called by make-dist from inside the new dist directory).</summary>
    public static int Write(HostSession session, Args a)
    {
        string? label = a.Get("label");
        string? stamp = a.Get("stamp");
        if (string.IsNullOrEmpty(label) || string.IsNullOrEmpty(stamp))
        {
            Console.Error.WriteLine("--write-dist-manifest needs --label L --stamp yyyymmdd-HHmmss (it is called by host\\make-dist)");
            return 1;
        }
        if (Present)
        {
            Console.Error.WriteLine("fatal: " + ManifestPath + " exists: a dist directory is never modified after creation.");
            return 1;
        }
        // A dist ships its own ledger and documents; a build whose ledger does not describe its patch set is not shippable.
        string ledgerPath = Path.Combine(Dir, "FIDELITY.md");
        PatchLedger.Result ledger = PatchLedger.Compare(ledgerPath, session.ProductionPatches, session.ToolPatches, PatchRegistry.SharedSites());
        if (!ledger.Ok)
        {
            Console.Error.WriteLine("fatal: the ledger " + ledgerPath + " does not match the patch registry; no manifest written.\n" + HostJson.Serialize(ledger));
            return 4;
        }
        foreach (string doc in new[] { "DRIVER-API.md", "README.md", "vocab/vocab.json", "vocab/catalog.json" })
        {
            if (!File.Exists(Path.Combine(Dir, doc)))
            {
                Console.Error.WriteLine("fatal: " + doc + " is missing from " + Dir + "; no manifest written.");
                return 4;
            }
        }
        SpireHost.Obs.ObsRuntime.Loaded vocab = SpireHost.Obs.ObsModule.Current;
        if (!Path.GetFullPath(vocab.Dir).StartsWith(Dir, StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine("fatal: the vocabulary was loaded from " + vocab.Dir + ", not from the dist directory (do not pass --vocab-dir here).");
            return 1;
        }
        SortedDictionary<string, string> files = HashFiles();
        object manifest = new
        {
            kind = "spirehost-dist",
            label,
            stamp,
            created_utc = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
            driver_api_version = DriverApi.Version,
            exe = ExeName,
            command = $"{ExeName} --<command> [options]   |   {ExeName} <area> --<command> [options]   (see README.md)",
            game = new
            {
                sts2_sha256 = GameAssemblies.Sts2Sha256,
                pinned_sts2_sha256 = GameAssemblies.PinnedSts2Sha256,
                how_the_exe_finds_it = "--game-dir DIR, else the environment variable SPIREHOST_GAME_DIR, else the first directory 'game' containing sts2.dll found by walking up from the " +
                                       "executable's directory (host\\dist\\<stamp>-<label>\\ -> host\\game). The SHA256 of sts2.dll must equal the pinned value in every case.",
                game_dir_when_created = GameAssemblies.GameDir,
            },
            modules = session.Registry.Modules,
            canonical_card_classes = session.Registry.CanonicalCardClassesName,
            vocab = new
            {
                vocab_hash = session.Registry.VocabHash,
                vocab_hash_hex = "0x" + session.Registry.VocabHash.ToString("X8"),
                vocab_version = vocab.Vocab.VocabVersion,
                catalog_hash = session.Registry.CatalogHash,
                catalog_hash_hex = "0x" + session.Registry.CatalogHash.ToString("X8"),
                files = new[] { "vocab/vocab.json", "vocab/catalog.json" },
            },
            patches = new
            {
                count = session.ProductionPatches.Count,
                by_category = ByCategory(session.ProductionPatches),
                behaviour = session.ProductionPatches.Count(p => p.Category == PatchCategory.Behaviour),
                hash = session.ProductionPatchHash,
                hash_of = "sha256 over the lines 'id|target|kind|category' sorted by id (PatchLedger.Hash)",
                shared_sites = PatchRegistry.SharedSites().Select(s => new { site = s.Site, ids = s.Ids, reason = s.Reason }).ToList(),
                list = session.ProductionPatches.OrderBy(p => p.Id, StringComparer.Ordinal).Select(p => new { id = p.Id, category = p.Category.ToString(), kind = p.Kind, target = p.Target }).ToList(),
            },
            ledger = new { file = "FIDELITY.md", sha256 = ledger.Sha256, matches_registry = ledger.Ok, production_rows = ledger.ProductionRows, tool_only_rows = ledger.ToolRows },
            files_sha256 = files,
        };
        File.WriteAllText(ManifestPath, HostJson.Serialize(manifest));
        HostJson.Emit(new
        {
            kind = "write-dist-manifest",
            ok = true,
            dist_dir = Dir,
            manifest = ManifestPath,
            label,
            stamp,
            files = files.Count,
            patches = session.ProductionPatches.Count,
            patch_hash = session.ProductionPatchHash,
            vocab_hash_hex = "0x" + session.Registry.VocabHash.ToString("X8"),
            catalog_hash_hex = "0x" + session.Registry.CatalogHash.ToString("X8"),
        }, a.Get("out"));
        return 0;
    }

    /// <summary>What differs between dist.json and the directory / the live session; empty = verified.</summary>
    public static List<string> Differences(HostSession session, out JsonNode? manifest)
    {
        List<string> diffs = new List<string>();
        manifest = JsonNode.Parse(File.ReadAllText(ManifestPath));
        if (manifest == null)
        {
            diffs.Add("dist.json is empty");
            return diffs;
        }
        void Same(string what, string? recorded, string live)
        {
            if (!string.Equals(recorded, live, StringComparison.Ordinal))
            {
                diffs.Add(what + ": dist.json has '" + recorded + "', this process has '" + live + "'");
            }
        }
        Same("driver_api_version", manifest["driver_api_version"]?.GetValue<string>(), DriverApi.Version);
        Same("game.sts2_sha256", manifest["game"]?["sts2_sha256"]?.GetValue<string>(), GameAssemblies.Sts2Sha256);
        Same("patches.hash", manifest["patches"]?["hash"]?.GetValue<string>(), session.ProductionPatchHash);
        Same("vocab.vocab_hash", manifest["vocab"]?["vocab_hash"]?.GetValue<uint>().ToString(), session.Registry.VocabHash.ToString());
        Same("vocab.catalog_hash", manifest["vocab"]?["catalog_hash"]?.GetValue<uint>().ToString(), session.Registry.CatalogHash.ToString());
        Same("modules", string.Join(",", manifest["modules"]?.AsArray().Select(m => m!.GetValue<string>()) ?? Array.Empty<string>()), string.Join(",", session.Registry.Modules));
        SortedDictionary<string, string> live = HashFiles();
        JsonObject? recordedFiles = manifest["files_sha256"] as JsonObject;
        if (recordedFiles == null)
        {
            diffs.Add("dist.json has no files_sha256");
            return diffs;
        }
        foreach (KeyValuePair<string, JsonNode?> kv in recordedFiles)
        {
            if (!live.TryGetValue(kv.Key, out string? sha))
            {
                diffs.Add("file removed: " + kv.Key);
            }
            else if (!string.Equals(sha, kv.Value?.GetValue<string>(), StringComparison.Ordinal))
            {
                diffs.Add("file changed: " + kv.Key);
            }
        }
        foreach (string f in live.Keys)
        {
            if (!recordedFiles.ContainsKey(f))
            {
                diffs.Add("file added: " + f);
            }
        }
        return diffs;
    }

    /// <summary>Every start from a dist directory: verify it, or throw (startup error). No dist.json next to the executable = a development build: nothing to verify.</summary>
    public static void VerifyAtStartup(HostSession session)
    {
        if (!Present)
        {
            return;
        }
        List<string> diffs = Differences(session, out _);
        if (diffs.Count > 0)
        {
            throw new InvalidOperationException("this dist directory was modified after creation (" + ManifestPath + "): " + string.Join("; ", diffs.Take(12)) +
                                                (diffs.Count > 12 ? "; ... " + (diffs.Count - 12) + " more" : ""));
        }
    }

    /// <summary>--verify-dist: the same verification as a command, with a JSON result (exit 0 verified, 4 not a dist).</summary>
    public static int VerifyCommand(HostSession session, Args a)
    {
        if (!Present)
        {
            HostJson.Emit(new { kind = "verify-dist", ok = false, dist_dir = Dir, problem = "no dist.json next to the executable: this is a development build, not a dist directory" }, a.Get("out"));
            return 4;
        }
        List<string> diffs = Differences(session, out JsonNode? manifest);
        HostJson.Emit(new
        {
            kind = "verify-dist",
            ok = diffs.Count == 0,
            dist_dir = Dir,
            label = manifest?["label"]?.GetValue<string>(),
            stamp = manifest?["stamp"]?.GetValue<string>(),
            driver_api_version = DriverApi.Version,
            sts2_sha256 = GameAssemblies.Sts2Sha256,
            game_dir = GameAssemblies.GameDir,
            files_verified = (manifest?["files_sha256"] as JsonObject)?.Count ?? 0,
            patches = session.ProductionPatches.Count,
            patch_hash = session.ProductionPatchHash,
            vocab_hash_hex = "0x" + session.Registry.VocabHash.ToString("X8"),
            catalog_hash_hex = "0x" + session.Registry.CatalogHash.ToString("X8"),
            differences = diffs,
        }, a.Get("out"));
        return diffs.Count == 0 ? 0 : 4;
    }
}
