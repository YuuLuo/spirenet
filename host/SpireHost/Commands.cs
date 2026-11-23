using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SpireHost.Boot;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Patches;

namespace SpireHost;

/// <summary>
/// Command dispatch of the one executable (H1 integration, ruling A3).
///
///     SpireHost --&lt;command&gt; [options]             top-level commands
///     SpireHost &lt;area&gt; --&lt;command&gt; [options]      area = combat | rooms | events | fidelity | obs | replay
///
/// EVERY command runs inside a HostSession created from <see cref="ProductionHost.Options"/>: one engine thread, the
/// production module set, the production patch set, the vocabulary loaded at boot. Before a command starts the
/// executable checks the configuration (ProductionHost.Verify) and, when it runs from a dist directory, the directory
/// itself (DistManifest.VerifyAtStartup). A failure there is a startup error: exit 3, nothing runs.
///
/// Process-level options (any command except where noted):
///   --game-dir DIR        the game directory (default: SPIREHOST_GAME_DIR, else the first "game" directory above the exe)
///   --vocab-dir DIR       the vocabulary directory (default: &lt;exe dir&gt;\vocab)
///   --game-log off|warn|info|debug   the game's own log (default off)
///   --verbose             host diagnostics on stderr
///   --patch-report FILE   after the command: the production patch list + hash, the tool instrumentation the command
///                         added and the registry == ledger verdict, as JSON (gate G1)
///   --audit-threads [--audit-out FILE]   TEST INSTRUMENT (not accepted by --serve): record the thread of every audited
///                         gameplay call; exit 6 if any ran off the engine thread
///
/// Exit codes: 0 ok; 1 usage; 2 unexpected exception; 3 startup error (game directory, vocabulary, dist verification,
/// configuration check); 4 a self-test / check failed; 5 an oracle comparison did not match; 6 thread audit not clean.
/// </summary>
public static class Commands
{
    public static readonly string[] Areas = { "combat", "rooms", "events", "fidelity", "obs", "replay" };

    public const string Usage =
        "usage: SpireHost --<command> [options]            SpireHost <area> --<command> [options]\n" +
        "run:   --serve ... | --fuzz-runs ... | --determinism ... | --replay-run ...\n" +
        "search (Linux): --serve --search ... (notes/search/protocol.md) | --search-fork-test --mode fidelity|bench ...\n" +
        "tools: --export-vocab DIR --merge OLD | --replay-mcr ... | --compare-run-traces ... | --compare-run-traces-v2 ... | --selftest-driver | --driver-info\n" +
        "       --list-patches | --check-ledger | --boot | --list-encounters | --verify-dist | --write-dist-manifest --label L --stamp S\n" +
        "areas: combat | rooms | events | fidelity | obs | replay   (each: --selftest and its own tools; an area without a command prints its usage)\n" +
        "any:   [--game-dir DIR] [--vocab-dir DIR] [--game-log off|warn|info|debug] [--verbose] [--patch-report FILE] [--audit-threads [--audit-out FILE]]\n" +
        "Full reference: host/README.md";

    public static int Run(string? area, Args a)
    {
        HostLog.Verbose = a.Has("verbose");
        if (area != null && Array.IndexOf(Areas, area) < 0)
        {
            Console.Error.WriteLine("unknown command area '" + area + "'\n" + Usage);
            return 1;
        }
        if (a.Command.Length == 0 || a.Command == "help")
        {
            Console.Error.WriteLine(Usage);
            return 1;
        }
        if (area == null && a.Command == "serve")
        {
            // Ruling A4: a server's command line is checked against an allow-list BEFORE anything is booted.
            // The search build's --serve --search (SpireHost.Search) has its own check; the production one is unchanged.
            string? bad = IsSearchServe(a) ? SpireHost.Search.SearchCommands.ServeArgumentError(a) : SpireHost.Obs.ObsCommands.ServeArgumentError(a);
            if (bad != null)
            {
                Console.Error.WriteLine("fatal: " + bad + "\n" + SpireHost.Obs.ObsCommands.Usage);
                return 1;
            }
            string? env = IsSearchServe(a) ? SpireHost.Search.SearchCommands.RuntimeError() : null;
            if (env != null)
            {
                Console.Error.WriteLine("fatal: --search: " + env);
                return 3;
            }
        }
        bool started = false;
        try
        {
            return HostSession.Run(ProductionHost.Options(area, a), session =>
            {
                ProductionHost.Verify(session);
                DistManifest.VerifyAtStartup(session);
                started = true;
                return RunCommand(session, area, a);
            });
        }
        catch (Exception e) when (!started)
        {
            Exception b = e.GetBaseException();
            Console.Error.WriteLine("SPIREHOST_FATAL startup: " + b.GetType().Name + ": " + b.Message);
            if (HostLog.Verbose)
            {
                Console.Error.WriteLine(e);
            }
            return 3;
        }
    }

    /// <summary>--serve --search: the search build's server (notes/search/protocol.md).</summary>
    private static bool IsSearchServe(Args a) => a.Keys.Any(k => k.Equals("search", StringComparison.OrdinalIgnoreCase));

    private static int RunCommand(HostSession session, string? area, Args a)
    {
        bool audit = a.Has("audit-threads");
        if (audit)
        {
            // TEST INSTRUMENT (read-only; ledger: thread-audit, tool-only). Never reachable in --serve (allow-list).
            ThreadAudit.Install();
            ThreadAudit.SetEngineThread(Environment.CurrentManagedThreadId);
        }
        if (area != null)
        {
            // TEST SEAM: the self-test probes of the modules audit every decision with the real encoder
            // (SpireHost.Driver.Tools.FrameAudit). Plugged in for the area commands only - never for a top-level command.
            SpireHost.Obs.Tests.ObsFrameAudit.Install();
        }
        int code = Dispatch(session, area, a);
        if (a.Get("patch-report") is string reportPath)
        {
            WritePatchReport(session, area, a, code, reportPath);
        }
        if (audit)
        {
            bool clean = ThreadAudit.OffEngineCalls == 0 && ThreadAudit.OnEngineCalls > 0;
            Console.Error.WriteLine($"[thread-audit] {(clean ? "clean" : "NOT CLEAN")}: {ThreadAudit.OnEngineCalls} audited calls on the engine thread, {ThreadAudit.OffEngineCalls} off it");
            if (a.Get("audit-out") is string auditPath)
            {
                string full = Path.GetFullPath(auditPath);
                Directory.CreateDirectory(Path.GetDirectoryName(full)!);
                File.WriteAllText(full, HostJson.Serialize(new { kind = "thread-audit", ok = clean, area, command = a.Command, exit_code = code, report = ThreadAudit.Report() }));
            }
            if (!clean && code == 0)
            {
                code = 6;
            }
        }
        return code;
    }

    private static int Dispatch(HostSession session, string? area, Args a)
    {
        switch (area)
        {
            case "combat":
                return SpireHost.Driver.Combat.CombatCommands.Run(session, a);
            case "rooms":
                return SpireHost.Driver.Rooms.RoomsCommands.Run(session, a);
            case "events":
                return SpireHost.Driver.Events.EventsCommands.Run(session, a);
            case "fidelity":
                return SpireHost.Fidelity.FidelityCommands.Run(session, a);
            case "obs":
                if (a.Command is "serve" or "export-vocab")
                {
                    Console.Error.WriteLine("--" + a.Command + " is a top-level command: SpireHost --" + a.Command + " ...");
                    return 1;
                }
                return SpireHost.Obs.ObsCommands.Run(session, a);
            case "replay":
                return SpireHost.Replay.ReplayCommands.Run(session, a);
        }
        if (DriverCommands.TryRun(session, a, out int code))
        {
            return code;
        }
        switch (a.Command)
        {
            case "serve" when IsSearchServe(a):
                return SpireHost.Search.SearchCommands.Serve(session, a);
            case "shadow-serve":
                return SpireHost.Search.ShadowServe.Run(session, a);
            case "shadow-selftest":
                return SpireHost.Search.ShadowServe.SelfTest(session, a);
            case "search-fork-test":
                return SpireHost.Search.Tests.ForkTests.Run(session, a);
            case "serve":
            case "export-vocab":
                return SpireHost.Obs.ObsCommands.Run(session, a);
            case "replay-mcr":
            case "compare-run-traces":
            case "compare-run-traces-v2":
                return SpireHost.Replay.ReplayCommands.Run(session, a);
            case "boot":
                return Boot(session, a);
            case "list-patches":
                return ListPatches(session, a);
            case "check-ledger":
                return CheckLedger(session, a);
            case "list-encounters":
                return ListEncounters(a);
            case "verify-dist":
                return DistManifest.VerifyCommand(session, a);
            case "write-dist-manifest":
                return DistManifest.Write(session, a);
            default:
                Console.Error.WriteLine("unknown command --" + a.Command + "\n" + Usage);
                return 1;
        }
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

    private static object Row(PatchRecord p) => new { id = p.Id, category = p.Category.ToString(), kind = p.Kind, target = p.Target, reason = p.Reason };

    private static object SharedSites() => PatchRegistry.SharedSites().Select(s => new { site = s.Site, ids = s.Ids, reason = s.Reason }).ToList();

    private static int Boot(HostSession session, Args a)
    {
        BootReport r = session.Boot;
        HostJson.Emit(new
        {
            kind = "boot",
            driver_api_version = DriverApi.Version,
            dist = DistManifest.Present ? DistManifest.Dir : null,
            sts2_sha256 = GameAssemblies.Sts2Sha256,
            game_dir = GameAssemblies.GameDir,
            boot_ms = Math.Round(r.TotalMs, 1),
            stages = r.Stages,
            modules = session.Registry.Modules,
            patches = session.ProductionPatches.Count,
            patches_by_category = ByCategory(session.ProductionPatches),
            patch_hash = session.ProductionPatchHash,
            vocab = session.Registry.VocabSource,
            vocab_hash_hex = "0x" + session.Registry.VocabHash.ToString("X8"),
            catalog_hash_hex = "0x" + session.Registry.CatalogHash.ToString("X8"),
            native_calls_during_boot = NullGodot.Delta(null),
            game_errors = HeadlessPatches.GameErrors,
            last_game_error = HeadlessPatches.LastGameError,
            loc_tables = LocalizationBootstrap.Tables,
            loc_entries = LocalizationBootstrap.Entries,
            profile = VeteranProfile.Installed ? VeteranProfile.Describe() : null,
            save_store = SaveBootstrap.Store == null ? null : new { files = SaveBootstrap.Store.FileCount, writes = SaveBootstrap.Store.Writes },
        }, a.Get("out"));
        return 0;
    }

    private static int ListEncounters(Args a)
    {
        var acts = MegaCrit.Sts2.Core.Models.ModelDb.Acts;
        List<object> rows = new List<object>();
        foreach (var e in MegaCrit.Sts2.Core.Models.ModelDb.AllEncounters)
        {
            string act = "";
            foreach (var act0 in acts)
            {
                foreach (var ae in act0.AllEncounters)
                {
                    if (ae.Id == e.Id)
                    {
                        act = act0.Id.Entry;
                    }
                }
            }
            rows.Add(new { id = e.Id.Entry, room = e.RoomType.ToString(), weak = e.IsWeak, act });
        }
        HostJson.Emit(rows, a.Get("out"));
        return 0;
    }

    /// <summary>
    /// --list-patches: the PRODUCTION patch set (what every mode runs on), by category, with its hash, the declared
    /// shared sites (ruling B3) and the registry == ledger verdict. Exit 4 if the set contains a Behaviour patch or the
    /// ledger does not match.
    /// </summary>
    private static int ListPatches(HostSession session, Args a)
    {
        PatchLedger.Result ledger = PatchLedger.Compare(a.Get("ledger"), session.ProductionPatches, session.ToolPatches, PatchRegistry.SharedSites());
        int behaviour = session.ProductionPatches.Count(p => p.Category == PatchCategory.Behaviour);
        List<string> toolRows = File.Exists(ledger.Path) ? PatchLedger.Parse(ledger.Path).Where(r => r.ToolOnly).Select(r => r.Id).ToList() : new List<string>();
        HostJson.Emit(new
        {
            kind = "list-patches",
            driver_api_version = DriverApi.Version,
            modules = session.Registry.Modules,
            patches = session.ProductionPatches.Count,
            by_category = ByCategory(session.ProductionPatches),
            behaviour,
            patch_hash = session.ProductionPatchHash,
            shared_sites = SharedSites(),
            ledger_ok = ledger.Ok,
            ledger,
            tool_only_rows_in_ledger = toolRows,
            records = session.ProductionPatches.Select(Row).ToList(),
        }, a.Get("out"));
        return behaviour == 0 && ledger.Ok ? 0 : 4;
    }

    /// <summary>--check-ledger [--ledger FILE]: registry == ledger (every production id exactly one row and vice versa, category and kind equal, shared sites cross-referenced).</summary>
    private static int CheckLedger(HostSession session, Args a)
    {
        PatchLedger.Result ledger = PatchLedger.Compare(a.Get("ledger"), session.ProductionPatches, session.ToolPatches, PatchRegistry.SharedSites());
        HostJson.Emit(new { kind = "check-ledger", ok = ledger.Ok, patch_hash = session.ProductionPatchHash, ledger }, a.Get("out"));
        return ledger.Ok ? 0 : 4;
    }

    /// <summary>
    /// --patch-report FILE (any command): what this process really ran on. The gate compares these files across serve,
    /// fuzz, determinism, replay-run, replay-mcr and compare-run-traces: the production ids and their hash must be equal,
    /// the tool instrumentation must be exactly what the ledger lists for that command.
    /// </summary>
    private static void WritePatchReport(HostSession session, string? area, Args a, int exitCode, string path)
    {
        PatchLedger.Result ledger = PatchLedger.Compare(a.Get("ledger"), session.ProductionPatches, session.ToolPatches, PatchRegistry.SharedSites());
        string full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(full, HostJson.Serialize(new
        {
            kind = "patch-report",
            area,
            command = a.Command,
            exit_code = exitCode,
            driver_api_version = DriverApi.Version,
            dist = DistManifest.Present ? DistManifest.Dir : null,
            modules = session.Registry.Modules,
            registry = session.Registry.Describe(),
            canonical_card_classes = session.Registry.CanonicalCardClassesName,
            vocab_hash = session.Registry.VocabHash,
            catalog_hash = session.Registry.CatalogHash,
            production_patch_count = session.ProductionPatches.Count,
            production_patch_hash = session.ProductionPatchHash,
            by_category = ByCategory(session.ProductionPatches),
            behaviour = session.ProductionPatches.Count(p => p.Category == PatchCategory.Behaviour),
            production_patches = session.ProductionPatches.Select(p => p.Id).OrderBy(x => x, StringComparer.Ordinal).ToList(),
            tool_patches = session.ToolPatches.Select(p => p.Id).OrderBy(x => x, StringComparer.Ordinal).ToList(),
            tool_patch_records = session.ToolPatches.Select(Row).ToList(),
            shared_sites = SharedSites(),
            ledger_ok = ledger.Ok,
            ledger,
            episodes_run = session.EpisodesRun,
        }));
    }
}
