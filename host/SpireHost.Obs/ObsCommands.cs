using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SpireHost.Boot;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Protocol;
using SpireHost.Obs.Tests;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs;

/// <summary>
/// Commands of the observation module. "SpireHost --serve" and "SpireHost --export-vocab" are top-level commands of
/// the executable; the rest is reached as "SpireHost obs --&lt;command&gt;". Every command runs on the production
/// configuration and uses the ONE vocabulary that ObsModule loaded at boot (ObsModule.Current).
/// </summary>
public static class ObsCommands
{
    public const string Usage =
        "       --serve [--pipe NAME] [--max-decisions N] [--turn-cap N] [--decision-cap N] [--verbose]\n" +
        "               [--allow-debug-options [--hp N]]            (debug options: see host/README.md, test-only switches)\n" +
        "       --export-vocab DIR --merge OLD_VOCAB_JSON            (append-only; DIR must not be the shipped vocabulary directory)\n" +
        "obs:   --selftest [--runs N] [--quick] [--cases a,b]\n" +
        "       --leak-test [--runs N] [--every K] [--hp N] [--label L] [--policy-seed P]\n" +
        "       --golden DIR [--runs N] [--hp N] [--characters A,B] [--seed-base S]   (frames for the Python checker)\n" +
        "       --sample-frames DIR [--per-kind N]                   (at least N frames of every DecisionKind + index.json)\n" +
        "       --bench-encode [--runs N] [--hp N] [--seats N | --lineup A+B] [--out FILE]";

    public static int Run(HostSession session, Args a)
    {
        switch (a.Command)
        {
            case "serve":
                return Serve(session, a);
            case "export-vocab":
                return ExportVocab(session, a);
            case "golden":
                return Golden(session, a);
            case "sample-frames":
                return SampleFrames.Run(session, a);
            case "bench-encode":
                return BenchEncode(session, a);
            case "selftest":
                return SelfTest.Run(session, a);
            case "leak-test":
                return LeakTests.Run(session, a);
            default:
                Console.Error.WriteLine(Usage);
                return 1;
        }
    }

    // -----------------------------------------------------------------------------------------------------
    //  --serve
    // -----------------------------------------------------------------------------------------------------

    /// <summary>
    /// Every option --serve accepts. Anything else on the command line is a startup error (ruling A4): a test-only
    /// switch, a mistyped option or an option of another command cannot silently change what a training server does.
    /// </summary>
    public static readonly string[] ServeOptionNames =
    {
        "serve", "pipe", "max-decisions", "turn-cap", "decision-cap", "verbose", "allow-debug-options", "hp",
        // process-level options of the executable
        "game-dir", "vocab-dir", "game-log", "patch-report",
    };

    /// <summary>What is wrong with a --serve command line, or null (ruling A4). Touches no game type: callable before boot.</summary>
    public static string? ServeArgumentError(Args a) => ServeArgumentError(a, Array.Empty<string>());

    /// <summary>
    /// The same check for a server that also accepts <paramref name="extraOptions"/> (the search build's --serve --search,
    /// SpireHost.Search). The production allow-list <see cref="ServeOptionNames"/> is not changed by it.
    /// </summary>
    public static string? ServeArgumentError(Args a, IReadOnlyCollection<string> extraOptions)
    {
        List<string> unknown = a.Keys.Where(k => Array.IndexOf(ServeOptionNames, k) < 0 && !extraOptions.Contains(k)).OrderBy(k => k, StringComparer.Ordinal).ToList();
        if (unknown.Count > 0)
        {
            return "--serve does not accept: --" + string.Join(", --", unknown);
        }
        // R-3: the two switches of a server take no value; a stray token after one is refused, a boolean literal is honoured
        // ("--allow-debug-options false" = debug options off).
        string? stray = a.SwitchValueError("allow-debug-options") ?? a.SwitchValueError("verbose");
        if (stray != null)
        {
            return stray;
        }
        if (a.Get("hp") != null && !a.Has("allow-debug-options"))
        {
            return "--hp is a debug option; a server only honours it when started with --allow-debug-options (see host/README.md)";
        }
        return null;
    }

    private static int Serve(HostSession session, Args a) => Serve(session, a, null, Array.Empty<string>());

    /// <summary>
    /// --serve; <paramref name="extension"/> != null only for the search build's --serve --search (SpireHost.Search): it is
    /// attached to the protocol server before the first command is read.
    /// </summary>
    public static int Serve(HostSession session, Args a, Func<ProtocolServer, IServeExtension>? extension, IReadOnlyCollection<string> extraOptions)
    {
        string? bad = ServeArgumentError(a, extraOptions);
        if (bad != null)
        {
            Console.Error.WriteLine("fatal: " + bad + "\n" + Usage);
            return 1;
        }
        bool allowDebug = a.Has("allow-debug-options");
        ObsRuntime.Loaded l = ObsModule.Current;
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        ServeOptions opt = new ServeOptions
        {
            MaxPolicyDecisions = a.GetLong("max-decisions", 100_000),
            MaxCombatTurns = a.GetInt("turn-cap", 500),
            DecisionCap = a.GetLong("decision-cap", 400_000),
            AllowDebugOptions = allowDebug,
            DebugForcedHp = allowDebug ? a.GetInt("hp", 0) : 0,
            Verbose = a.Has("verbose"),
        };
        IMessageTransport t;
        string? pipe = a.Get("pipe");
        try
        {
            t = pipe != null ? StreamTransport.NamedPipe(pipe) : StreamTransport.Stdio();
        }
        catch (Exception e) when (e is IOException || e is TimeoutException || e is UnauthorizedAccessException || e is ArgumentException)
        {
            Console.Error.WriteLine("fatal: cannot open transport: " + e.Message);
            return 3;
        }
        Console.Error.WriteLine($"[obs] serving on {t.Name}; vocab {l.Source} hash 0x{l.Vocab.Hash:X8}, catalog hash 0x{l.CatalogHash:X8}" +
                                (allowDebug ? "; DEBUG OPTIONS ALLOWED (--allow-debug-options" + (opt.DebugForcedHp != 0 ? ", forced HP " + opt.DebugForcedHp : "") + ")" : ""));
        ProtocolServer server = new ProtocolServer(session, t, enc, opt);
        if (extension != null)
        {
            server.Extension = extension(server);
        }
        Stopwatch wall = Stopwatch.StartNew();
        TimeSpan cpu0 = Process.GetCurrentProcess().TotalProcessorTime;
        int code;
        using (t)
        {
            code = server.Serve();
        }
        double cpu = (Process.GetCurrentProcess().TotalProcessorTime - cpu0).TotalSeconds;
        EncoderStats s = enc.Stats;
        Console.Error.WriteLine($"[obs] serve done: exit {code}, {server.Episodes} episodes ({server.Wins} win, {server.Losses} loss, {server.Truncations} truncated, {server.Aborts} aborted, {server.Abandoned} abandoned), " +
            $"{server.Decisions} decisions, {server.Errors} errors, debug episodes {server.DebugEpisodes}, debug RESETs refused {server.DebugResetsRefused}, " +
            $"{wall.Elapsed.TotalSeconds:F1} s wall, {cpu:F1} s cpu; encoder {s.Frames} frames, mean {(s.Frames == 0 ? 0 : s.TotalUs / s.Frames):F1} us, " +
            $"order violations {s.ActionOrderViolations}, choice order {s.ChoiceOrderViolations}, unmapped refs {s.UnmappedActionRefs}, overflows {s.Overflows}, UNK lookups {l.Vocab.UnknownLookups}" +
            (server.ExactCompared > 0 ? $"; exact check: {server.ExactCompared} frames compared, {server.ExactMismatches} mismatches (first: {server.FirstMismatch ?? "-"}), {server.SideEffects} side effects; driver CanPlay {SpireHost.Driver.HookCache.CanPlayCompared} compared, {SpireHost.Driver.HookCache.CanPlayMismatches} mismatches; IsHittable {SpireHost.Driver.HookCache.HittableCompared} compared, {SpireHost.Driver.HookCache.HittableMismatches} mismatches; card classes {ObsEncoder.ClassesCompared} compared, {ObsEncoder.ClassesMismatches} mismatches" : ""));
        return code;
    }

    // -----------------------------------------------------------------------------------------------------
    //  --export-vocab DIR --merge OLD
    // -----------------------------------------------------------------------------------------------------

    private static int ExportVocab(HostSession session, Args a)
    {
        string? dir = a.Get("export-vocab");
        if (string.IsNullOrEmpty(dir))
        {
            Console.Error.WriteLine("--export-vocab needs an output directory");
            return 1;
        }
        // A released vocabulary is never overwritten in place: the output goes to a new directory and is then installed
        // (host\\vocab) by hand / by the release procedure in host\\README.md.
        string full = Path.GetFullPath(dir);
        if (string.Equals(full.TrimEnd('\\', '/'), ObsModule.Current.Dir.TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine("--export-vocab: " + full + " is the vocabulary directory this process loaded; export into another directory.");
            return 1;
        }
        string sha = GameAssemblies.Sts2Sha256;
        List<string> report = new List<string>();
        VocabSet vocab;
        string? merge = a.Get("merge");
        if (merge != null)
        {
            VocabSet old = VocabSet.FromJson(CanonicalJson.ParseFile(merge));
            vocab = VocabExport.Merge(old, sha, out report);
        }
        else if (a.Has("fresh"))
        {
            // ids are NOT compatible with any released vocabulary: only for a new schema generation
            vocab = VocabExport.BuildFresh(sha);
        }
        else
        {
            Console.Error.WriteLine("--export-vocab needs --merge OLD_VOCAB_JSON (the vocabulary is append-only); --fresh starts a new, incompatible id space.");
            return 1;
        }
        CatalogExport cat = new CatalogExport(vocab);
        foreach (int asc in new[] { 0, 10 })
        {
            bool done = false;
            EpisodeResult r = ScenarioRunner.Run(session, "catalog-a" + asc, ctx =>
            {
                cat.CollectMonsterHp(ctx, asc >= 8);
                if (asc == 0)
                {
                    cat.CollectCards(ctx);
                }
                done = true;
                return Task.CompletedTask;
            }, character: "IRONCLAD", ascension: asc, stopWhen: _ => done);
            if (!ScenarioRunner.Passed(r))
            {
                Console.Error.WriteLine("catalog episode failed: " + r.Abort?.Signature + " " + r.Abort?.Message);
                return 4;
            }
        }
        JObj catalog = cat.Build();
        VocabExport.Write(dir, vocab, catalog);
        int catalogHash = CanonicalJson.Hash(catalog);
        HostJson.Emit(new
        {
            kind = "obs-export-vocab",
            dir = full,
            merged_from = merge == null ? null : Path.GetFullPath(merge),
            vocab_hash = vocab.Hash,
            vocab_hash_hex = "0x" + vocab.Hash.ToString("X8"),
            catalog_hash = catalogHash,
            catalog_hash_hex = "0x" + catalogHash.ToString("X8"),
            vocab_version = vocab.VocabVersion,
            categories = vocab.Categories.ToDictionary(c => c.Name, c => c.Count),
            merge_report = report,
            catalog_problems = cat.Problems,
        }, a.Get("out"));
        return 0;
    }

    // -----------------------------------------------------------------------------------------------------
    //  --golden DIR: whole random runs, every decision encoded, frames written for the Python checker
    // -----------------------------------------------------------------------------------------------------

    private static int Golden(HostSession session, Args a)
    {
        string dir = a.Get("golden") ?? "golden";
        Directory.CreateDirectory(dir);
        ObsRuntime.Loaded l = ObsModule.Current;
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = EncoderOptions.FromResetFlags(a.GetInt("reset-flags", Schema.ResetFlag.Default), 0, -1);
        int runs = a.GetInt("runs", 6);
        string[] chars = (a.Get("characters") ?? string.Join(",", DriverCommands.AllCharacters)).Split(',', StringSplitOptions.RemoveEmptyEntries);
        string path = Path.Combine(dir, a.Get("name") ?? "frames.bin");
        List<object> episodes = new List<object>();
        FrameDump dump;
        using (FileStream fs = File.Create(path))
        {
            dump = new FrameDump(enc, fs);
            for (int i = 0; i < runs; i++)
            {
                // one seat: chars[i % n] (as before); --seats N / --lineup A+B: the tools' lineup rule (phase H5 golden frames)
                string[] lineup = DriverCommands.LineupFor(a, chars, i);
                string ch = string.Join("+", lineup);
                EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel(a.Get("seed-base", "GOLDEN") + ":" + i), Characters = lineup, Ascension = a.GetInt("ascension", 0) };
                EpisodeOptions o = DriverCommands.EpisodeOptionsFrom(a);
                EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)(1000 + i)), o, dump);
                episodes.Add(new { seed = r.Seed, character = ch, outcome = r.Outcome.ToString(), abort = r.Abort?.Signature, floor = r.TotalFloor, decisions = r.Counters.DecisionPoints });
            }
        }
        // synthetic CONFIRM / BUNDLE / CRYSTAL frames in controlled states -> special.bin
        bool specialOk = true;
        object? special = null;
        if (a.GetBool("special", true))
        {
            special = SpecialFrames.Run(session, l, Path.Combine(dir, "special.bin"), out specialOk);
        }
        HostJson.Emit(new
        {
            kind = "obs-golden",
            file = Path.GetFullPath(path),
            frames = dump.FramesWritten,
            encode_failures = dump.EncodeFailures,
            first_failure = dump.FirstFailure,
            vocab_hash = l.Vocab.Hash,
            catalog_hash = l.CatalogHash,
            stats = Stats(enc.Stats, l.Vocab),
            by_kind = dump.ByKind.ToDictionary(kv => kv.Key, kv => kv.Value.Summary()),
            episodes,
            special_file = special == null ? null : Path.GetFullPath(Path.Combine(dir, "special.bin")),
            special,
        }, a.Get("out"));
        return enc.Stats.ActionOrderViolations == 0 && enc.Stats.UnmappedActionRefs == 0 && enc.Stats.ChoiceOrderViolations == 0 && specialOk ? 0 : 4;
    }

    public static Dictionary<string, double> Sections(ObsEncoder enc)
    {
        Dictionary<string, double> d = new Dictionary<string, double>();
        long frames = Math.Max(1, enc.Stats.Frames);
        for (int i = 0; i < enc.SectionTicks.Length; i++)
        {
            d[ObsEncoder.SectionNames[i]] = Math.Round(enc.SectionTicks[i] * 1e6 / Stopwatch.Frequency / frames, 2);
        }
        return d;
    }

    public static object Stats(EncoderStats s, VocabSet v) => new
    {
        frames = s.Frames,
        mean_us = s.Frames == 0 ? 0 : Math.Round(s.TotalUs / s.Frames, 2),
        max_us = s.MaxUs,
        action_order_violations = s.ActionOrderViolations,
        choice_order_violations = s.ChoiceOrderViolations,
        unmapped_action_refs = s.UnmappedActionRefs,
        overflows = s.Overflows,
        upgrade_previews = s.UpgradePreviews,
        target_preview_rows = s.TargetPreviewRows,
        unk_lookups = v.UnknownLookups,
    };

    // -----------------------------------------------------------------------------------------------------
    //  --bench-encode: encoder cost and frame sizes by kind on whole runs (no file I/O)
    // -----------------------------------------------------------------------------------------------------

    private static int BenchEncode(HostSession session, Args a)
    {
        ObsRuntime.Loaded l = ObsModule.Current;
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = EncoderOptions.FromResetFlags(a.GetInt("reset-flags", Schema.ResetFlag.Default), 0, -1);
        enc.Options.SectionTiming = a.GetBool("sections", true);
        enc.Options.ExactGameCalls = a.GetBool("exact", false);
        int runs = a.GetInt("runs", 24);
        FrameDump dump = new FrameDump(enc, null) { EncodeAuto = a.GetBool("auto", false) };
        List<Dictionary<string, double>> probes = new List<Dictionary<string, double>>();
        if (a.Has("probe"))
        {
            dump.AfterEncode = (ctx, d) =>
            {
                if (d.Kind == DecisionKind.Combat && probes.Count < 40 && ctx.Counters.PolicyDecisions % 7 == 3)
                {
                    probes.Add(CostProbe.Measure(ctx, 20));
                }
                return true;
            };
        }
        Stopwatch wall = Stopwatch.StartNew();
        TimeSpan cpu0 = Process.GetCurrentProcess().TotalProcessorTime;
        long decisions = 0;
        List<string> aborts = new List<string>();
        Dictionary<string, int> outcomes = new Dictionary<string, int>();
        for (int i = 0; i < runs; i++)
        {
            string ch = DriverCommands.AllCharacters[i % 6];
            // --seats N / --lineup A+B (H5): the encoder cost of multi-seat frames; without them one seat as before
            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel(a.Get("seed-base", "BENCH") + ":" + i), Characters = DriverCommands.LineupFor(a, DriverCommands.AllCharacters, i) };
            EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)(77 + i)), DriverCommands.EpisodeOptionsFrom(a), dump);
            decisions += r.Counters.PolicyDecisions;
            outcomes[r.Outcome.ToString()] = outcomes.GetValueOrDefault(r.Outcome.ToString()) + 1;
            if (r.Abort != null)
            {
                aborts.Add(r.Abort.Signature + " | " + r.Abort.Message + " | seed " + r.Seed + " " + ch);
            }
        }
        double cpu = (Process.GetCurrentProcess().TotalProcessorTime - cpu0).TotalSeconds;
        HostJson.Emit(new
        {
            kind = "obs-bench-encode",
            aborts,
            runs,
            hp = a.GetInt("hp", 0),
            reset_flags = a.GetInt("reset-flags", Schema.ResetFlag.Default),
            outcomes,
            policy_decisions = decisions,
            wall_s = Math.Round(wall.Elapsed.TotalSeconds, 2),
            cpu_s = Math.Round(cpu, 2),
            policy_decisions_per_cpu_second = cpu <= 0 ? 0 : Math.Round(decisions / cpu, 1),
            all = dump.All.Summary(),
            by_kind = dump.ByKind.ToDictionary(kv => kv.Key, kv => kv.Value.Summary()),
            stats = Stats(enc.Stats, l.Vocab),
            section_us_per_frame = Sections(enc),
            getter_us_per_call = probes.Count == 0 ? null : probes[0].Keys.ToDictionary(k => k, k => Math.Round(probes.Average(p => p.GetValueOrDefault(k)), 3)),
        }, a.Get("out"));
        return 0;
    }
}
