using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SpireHost.Boot;
using SpireHost.Driver;

namespace SpireHost.Replay;

/// <summary>
/// --replay-mcr  [--file F.mcr | --dir DIR] [--files a,b,..] [--start I] [--count N] [--stride S --offset K]
///               [--run-traces | --runs-v2 natural,deep | --runs-mp [corpus|deep|SESSION] [--mp-clients 0|all]]
///               [--loose-timing] [--stop-at-divergence] [--extended-trace] [--keep-checksums] [--diff-lines N]
///               [--out FILE]
/// Replays real-game .mcr combats in the host (McrReplayer) and prints one JSON document: totals, per-status
/// counts, aggregates by character / ascension / encounter / room type / block / first-divergence card, power
/// and context, and one record per replay (first diverging checksum, field-level diff, extended checks).
/// Exit 0 = every replay passed, 5 = at least one did not.
/// </summary>
public static class McrCommands
{
    /// <summary>oracle\replays next to the host directory (OraclePaths).</summary>
    public static string DefaultCorpus => OraclePaths.Replays;

    public static List<string> SelectFiles(Args a)
    {
        List<string> files = new List<string>();
        if (a.Get("file") is string one)
        {
            files.Add(Path.GetFullPath(one));
            return files;
        }
        string dir = a.Get("dir", DefaultCorpus);
        if (a.Has("runs-mp"))
        {
            // Phase H5: the fights of the real multi-client sessions (oracle\runs-mp\<session>\c<k>\NNN-combat-*.mcr, notes\oracle-mp.md 3.5).
            //   --runs-mp [corpus | <set directory: deep, prelim> | <session>]   corpus (default) = every session directly under the root
            //   --mp-clients 0 | 0,2 | all                                       whose replay files (default: the host client's, c0)
            string root = a.Get("dir", OraclePaths.RunsMp);
            string set = a.Get("runs-mp") ?? "corpus";
            string[] notSessions = { "dev", "failed", "prelim", "deep" };
            List<string> sessions;
            if (set == "corpus" || set.Length == 0)
            {
                sessions = Directory.GetDirectories(root).Where(d => !notSessions.Contains(Path.GetFileName(d), StringComparer.OrdinalIgnoreCase)).ToList();
            }
            else if (Directory.Exists(Path.Combine(root, set, "c0")))
            {
                sessions = new List<string> { Path.Combine(root, set) };
            }
            else
            {
                sessions = Directory.GetDirectories(Path.Combine(root, set)).ToList();
            }
            string clients = a.Get("mp-clients", "0");
            string[] wanted = clients.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            foreach (string session in sessions.OrderBy(s => s, StringComparer.Ordinal))
            {
                foreach (string client in Directory.GetDirectories(session, "c?").OrderBy(s => s, StringComparer.Ordinal))
                {
                    if (clients == "all" || wanted.Contains(Path.GetFileName(client).Substring(1)))
                    {
                        files.AddRange(Directory.GetFiles(client, "*.mcr").OrderBy(f => f, StringComparer.Ordinal));
                    }
                }
            }
            int strideMp = Math.Max(1, a.GetInt("stride", 1));
            int offsetMp = a.GetInt("offset", 0);
            return files.Skip(a.GetInt("start", 0)).Take(a.GetInt("count", int.MaxValue)).Where((_, i) => i % strideMp == offsetMp).ToList();
        }
        if (a.Has("run-traces") || a.Has("runs-v2"))
        {
            // the map-node combats of the golden run traces (oracle\runs\*\NNN-combat-*.mcr), or - with --runs-v2 [natural|deep|natural,deep] -
            // of the v2 traces (oracle\runs-v2\<set>\*\NNN-combat-*.mcr): the fights of real runs through all three acts
            if (a.Has("runs-v2"))
            {
                string root = a.Get("dir", V2.RunTraceCompareV2.DefaultRoot);
                foreach (string set in (a.Get("runs-v2") ?? "natural,deep").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                {
                    files.AddRange(Directory.GetFiles(Path.Combine(root, set), "*.mcr", SearchOption.AllDirectories));
                }
                files = files.OrderBy(f => f, StringComparer.Ordinal).ToList();
            }
            else
            {
                dir = a.Get("dir", RunTraceCompare.DefaultDir);
                files.AddRange(Directory.GetFiles(dir, "*.mcr", SearchOption.AllDirectories).OrderBy(f => f, StringComparer.Ordinal));
            }
            int stride0 = Math.Max(1, a.GetInt("stride", 1));
            int offset0 = a.GetInt("offset", 0);
            return files.Skip(a.GetInt("start", 0)).Take(a.GetInt("count", int.MaxValue)).Where((_, i) => i % stride0 == offset0).ToList();
        }
        if (a.Get("files") is string list)
        {
            foreach (string s in list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                string name = s.EndsWith(".mcr", StringComparison.OrdinalIgnoreCase) ? s : s + ".mcr";
                files.Add(Path.Combine(dir, name));
            }
            return files;
        }
        files.AddRange(Directory.GetFiles(dir, "*.mcr").Where(f => !f.EndsWith(".rerun.mcr", StringComparison.OrdinalIgnoreCase)).OrderBy(f => f, StringComparer.Ordinal));
        int start = a.GetInt("start", 0);
        int count = a.GetInt("count", int.MaxValue);
        files = files.Skip(start).Take(count).ToList();
        int stride = Math.Max(1, a.GetInt("stride", 1));
        int offset = a.GetInt("offset", 0);
        if (stride > 1)
        {
            files = files.Where((_, i) => i % stride == offset).ToList();
        }
        return files;
    }

    public static ReplayOptions OptionsFrom(Args a)
    {
        return new ReplayOptions
        {
            StrictTiming = !a.Has("loose-timing"),
            ContinueAfterDivergence = !a.Has("stop-at-divergence"),
            ExtendedTrace = a.Has("extended-trace"),
            DiffLines = a.GetInt("diff-lines", 60),
            AlterEventIndex = a.GetInt("alter-event", -1),   // test only: replace that recorded card play by an end turn
        };
    }

    public static int ReplayMcr(HostSession session, Args a)
    {
        List<string> files = SelectFiles(a);
        if (files.Count == 0)
        {
            Console.Error.WriteLine("replay-mcr: no .mcr files selected");
            return 1;
        }
        if (a.Has("metrics-gate-report"))
        {
            return MetricsGateReport(files, a);
        }
        ReplayOptions opt = OptionsFrom(a);
        bool keepChecksums = a.Has("keep-checksums");
        bool coverage = a.Has("coverage");
        if (coverage)
        {
            CoverageProbe.Install(a.Get("notes-dir", OraclePaths.Notes)); // tool-only, read-only (ledger: replay-coverage:*)
        }
        McrReplayer replayer = new McrReplayer(session.Pump);
        List<McrReplayResult> results = new List<McrReplayResult>(files.Count);
        Stopwatch wall = Stopwatch.StartNew();
        List<string> emptyRecordings = new List<string>();
        List<McrReplayResult> eventFights = new List<McrReplayResult>();
        foreach (string f in files)
        {
            McrReplayResult r;
            McrFile file;
            try
            {
                file = McrFile.Load(f);
            }
            catch (Exception e)
            {
                results.Add(new McrReplayResult { File = Path.GetFileNameWithoutExtension(f), Status = "error", Error = "load: " + e.GetBaseException().Message });
                continue;
            }
            if (file.Replay.events.Count == 0 && file.Replay.checksumData.Count == 0)
            {
                // Not a recorded fight: the file holds no event and no checksum (the real client writes its replay buffer when a
                // run is left, RunManager.CleanUp; a run abandoned outside combat leaves an empty one - validation report P4).
                // There is nothing to re-execute and nothing to compare: it is listed, never counted as a replay or as a pass.
                emptyRecordings.Add(Path.GetFileName(Path.GetDirectoryName(f)) + "/" + Path.GetFileName(f));
                continue;
            }
            CoverageProbe.BeginReplay();
            try
            {
                r = replayer.Run(file, opt);
            }
            catch (Exception e)
            {
                r = new McrReplayResult { File = Path.GetFileNameWithoutExtension(f), Status = "error", Error = "replay: " + e.GetBaseException().Message };
            }
            CoverageProbe.EndReplay(r.File, r.Status == "pass");
            if (!keepChecksums)
            {
                r.Checksums = null;
                r.ExtendedTrace = null;
            }
            if (r.Status == "event_fight")
            {
                // A fight started by an event option: the replay file cannot be re-executed from itself (see McrReplayer.Run). Listed
                // with what it holds, never counted as a replay or as a pass.
                eventFights.Add(r);
                if (HostLog.Verbose || a.Has("progress"))
                {
                    Console.Error.WriteLine($"[replay] {r.File}: event fight, not startable from the replay file - {r.StatusDetail}");
                }
                continue;
            }
            results.Add(r);
            if (HostLog.Verbose || a.Has("progress"))
            {
                Console.Error.WriteLine($"[replay] {r.File} {r.Character} A{r.Ascension} {r.Encounter}: {r.Status} {r.Matched}/{r.ChecksumsRecorded} ({r.Ms:0} ms)" +
                                        (r.First != null ? $" first@{r.First.ChecksumOrdinal} '{r.First.LocalContext}'" : "") + (r.StatusDetail != null ? " - " + r.StatusDetail : ""));
            }
        }
        object summary = Summarize(results, wall.Elapsed.TotalSeconds, a.Has("no-records"), coverage ? CoverageProbe.Report() : null, session, emptyRecordings, eventFights);
        HostJson.Emit(summary, a.Get("out"));
        // (a replay with a signature mismatch has status sig_mismatch - verifier finding F3; a run in which nothing was replayed is not a pass)
        return results.Count > 0 && results.All(r => r.Status == "pass") ? 0 : 5;
    }

    /// <summary>
    /// --replay-mcr --metrics-gate-report [--file | --dir | --run-traces ...]: a STATIC report over the recorded saves; no
    /// replay is run and nothing is sent anywhere. The game's end-of-run telemetry (Runs.Metrics/MetricUtilities.cs:89-137)
    /// reaches its HTTP request only for a run that is not a custom game (:111), did not use a debug command (:116), is a
    /// Standard / Dread / Blitz run (:121) and has at least 5 map-point history entries (3 in Blitz; :126-137) - and only
    /// from RunManager.OnEnded, i.e. when the replayed combat ends the run. The report says for how many of the selected
    /// recordings the save passes those gates, which bounds what a host WITHOUT the network guard could have sent.
    /// (The host blocks the upload unconditionally: metrics-upload-disabled, metrics-put-request-blocked.)
    /// </summary>
    private static int MetricsGateReport(List<string> files, Args a)
    {
        int loaded = 0, failed = 0, debugCommand = 0, customGame = 0, otherMode = 0, shortHistory = 0, passes = 0;
        List<string> passing = new List<string>();
        SortedDictionary<int, int> historyHistogram = new SortedDictionary<int, int>();
        foreach (string f in files)
        {
            McrFile file;
            try
            {
                file = McrFile.Load(f);
            }
            catch (Exception)
            {
                failed++;
                continue;
            }
            loaded++;
            MegaCrit.Sts2.Core.Saves.SerializableRun run = file.Replay.serializableRun;
            int entries = run.MapPointHistory.Sum(l => l.Count);
            historyHistogram[Math.Min(entries, 20)] = historyHistogram.GetValueOrDefault(Math.Min(entries, 20)) + 1;
            bool mode = run.GameMode == MegaCrit.Sts2.Core.Runs.GameMode.Standard || run.GameMode == MegaCrit.Sts2.Core.Runs.GameMode.Dread || run.GameMode == MegaCrit.Sts2.Core.Runs.GameMode.Blitz;
            bool enough = run.GameMode == MegaCrit.Sts2.Core.Runs.GameMode.Blitz ? entries >= 3 : entries >= 5;
            if (run.IsCustomGame)
            {
                customGame++;
            }
            if (run.ExtraFields.UsedDebugCommand)
            {
                debugCommand++;
            }
            if (!mode)
            {
                otherMode++;
            }
            if (!enough)
            {
                shortHistory++;
            }
            if (!run.IsCustomGame && !run.ExtraFields.UsedDebugCommand && mode && enough)
            {
                passes++;
                if (passing.Count < 2000)
                {
                    passing.Add(file.Name);
                }
            }
        }
        HostJson.Emit(new
        {
            command = "replay-mcr --metrics-gate-report",
            note = "static: no replay was run, nothing was sent; the map-point history is the SAVED one (the replayed combat's own entry may add to it)",
            files = files.Count,
            loaded,
            load_failures = failed,
            saves_with_used_debug_command = debugCommand,
            saves_of_custom_games = customGame,
            saves_of_other_game_modes = otherMode,
            saves_with_fewer_history_entries_than_the_upload_needs = shortHistory,
            saves_that_pass_every_gate_before_the_http_request = passes,
            map_point_history_entries_histogram_capped_at_20 = historyHistogram,
            passing_examples = passing.Take(12).ToList(),
        }, a.Get("out"));
        return 0;
    }

    public static object Summarize(List<McrReplayResult> results, double seconds, bool noRecords = false, object? coverage = null, HostSession? session = null, List<string>? emptyRecordings = null,
        List<McrReplayResult>? eventFights = null)
    {
        eventFights ??= new List<McrReplayResult>();
        session ??= HostSession.Current;
        Dictionary<string, int> byStatus = results.GroupBy(r => r.Status).OrderBy(g => g.Key).ToDictionary(g => g.Key, g => g.Count());
        object Agg(Func<McrReplayResult, string> key) => results.GroupBy(key).OrderBy(g => g.Key, StringComparer.Ordinal).ToDictionary(g => g.Key, g => new
        {
            replays = g.Count(),
            pass = g.Count(r => r.Status == "pass"),
            not_pass = g.Count(r => r.Status != "pass"),
            statuses = g.GroupBy(r => r.Status).ToDictionary(x => x.Key, x => x.Count()),
        });
        List<McrReplayResult> bad = results.Where(r => r.Status != "pass").ToList();
        object AggBad(Func<McrReplayResult, IEnumerable<string>> keys) => bad.SelectMany(r => keys(r).Select(k => (k, r)))
            .GroupBy(t => t.k).OrderByDescending(g => g.Count()).ThenBy(g => g.Key, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => new { replays = g.Count(), files = g.Select(t => t.r.File).Take(12).ToList() });
        int checksums = results.Sum(r => r.ChecksumsRecorded);
        return new
        {
            command = "replay-mcr",
            // the configuration the replays ran on: the production set + this tool's read-only instrumentation (ruling B4)
            vocab_hash = session?.Registry.VocabHash,
            production_patches = session?.ProductionPatches.Count,
            production_patch_hash = session?.ProductionPatchHash,
            tool_patches = session?.ToolPatches.Select(p => p.Id).Where(id => !id.StartsWith("replay-coverage:", StringComparison.Ordinal)).ToList(),
            replays = results.Count,
            // files without any event or checksum: not recorded fights; listed, never counted as replays (see ReplayMcr)
            empty_recordings = emptyRecordings ?? new List<string>(),
            // fights started by an event option: the game's replay format cannot re-execute them from the file alone (the save is the map
            // point's entry, the event options are not replay events - McrReplayer.Run); listed, never counted as replays or passes.
            // Their recorded states are compared inside the run by --compare-run-traces-v2.
            event_fights_not_startable_from_the_replay_file = new
            {
                count = eventFights.Count,
                events = eventFights.Sum(r => r.Events),
                checksums_recorded = eventFights.Sum(r => r.ChecksumsRecorded),
                by_event = eventFights.GroupBy(r => EventOf(r.StatusDetail)).OrderBy(g => g.Key, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.Count()),
                files = eventFights.Select(r => r.File).ToList(),
            },
            policy_actions_in_sidecars = results.Sum(r => r.Extended.PolicyActions),
            replays_without_sidecar_actions = results.Where(r => r.Extended.PolicyActions == 0).Select(r => r.File).Take(20).ToList(),
            pass = results.Count(r => r.Status == "pass"),
            by_status = byStatus,
            checksums_recorded = checksums,
            checksums_generated = results.Sum(r => r.ChecksumsGenerated),
            checksums_matched = results.Sum(r => r.Matched),
            checksums_mismatched = results.Sum(r => r.Mismatched),
            events = results.Sum(r => r.Events),
            events_injected = results.Sum(r => r.EventsInjected),
            // multi-player replays only (0 for single-player recordings): see McrReplayResult
            multi_player = new
            {
                replays = results.Count(r => r.Players > 1),
                pass = results.Count(r => r.Players > 1 && r.Status == "pass"),
                checksums_recorded = results.Where(r => r.Players > 1).Sum(r => r.ChecksumsRecorded),
                checksums_matched = results.Where(r => r.Players > 1).Sum(r => r.Matched),
                actions_finished_not_executing = results.Sum(r => r.ActionsFinishedNotExecuting),
                replays_with_an_action_finished_not_executing = results.Count(r => r.ActionsFinishedNotExecuting > 0),
                events_after_combat_end = results.Sum(r => r.EventsAfterCombatEnd),
                replays_with_events_after_combat_end = results.Count(r => r.EventsAfterCombatEnd > 0),
                replays_with_a_dead_player = results.Count(r => r.Players > 1 && r.PlayersDead > 0),
                replays_with_every_player_dead = results.Count(r => r.Players > 1 && r.PlayersDead == r.Players),
            },
            extended = new
            {
                compared = results.Sum(r => r.Extended.Compared),
                sig_mismatches = results.Sum(r => r.Extended.SigMismatches),
                replays_with_sig_mismatch = results.Count(r => r.Extended.SigMismatches > 0),
                option_mismatches = results.Sum(r => r.Extended.OptionMismatches),
                replays_with_option_mismatch = results.Count(r => r.Extended.OptionMismatches > 0),
                option_mismatches_potions_locked_in_recording = results.Sum(r => r.Extended.OptionMismatchesPotionsLockedInRecording),
                not_a_decision_point = results.Sum(r => r.Extended.NotADecisionPoint),
            },
            seconds = Math.Round(seconds, 2),
            ms_per_replay = results.Count > 0 ? Math.Round(results.Average(r => r.Ms), 2) : 0,
            by_players = Agg(r => r.Players + " player" + (r.Players == 1 ? "" : "s")),
            by_character = Agg(r => r.Character),
            by_ascension = Agg(r => "A" + r.Ascension),
            by_room_type = Agg(r => r.RoomType),
            by_block = Agg(r => r.Block),
            by_act = Agg(r => r.Act),
            by_encounter = Agg(r => r.Encounter),
            first_divergence_by_context = AggBad(r => new[] { ContextKind(r.First?.RecordedContext ?? r.First?.LocalContext ?? r.StatusDetail ?? r.Status) }),
            first_divergence_by_card = AggBad(r => new[] { r.First?.InvolvedCard ?? "-" }),
            first_divergence_by_power = AggBad(r => r.First?.InvolvedPowers.Count > 0 ? r.First.InvolvedPowers : new List<string> { "-" }),
            first_divergence_by_diff_id = AggBad(r => r.First?.DiffIds.Count > 0 ? r.First.DiffIds : new List<string> { "-" }),
            failure_by_frame = AggBad(r => new[] { r.Frames != null && r.Frames.Count > 0 ? r.Frames[0] : "-" }),
            coverage,
            network_attempts_blocked = SpireHost.Patches.NetworkGuard.NetworkAttempts,
            last_network_attempt = SpireHost.Patches.NetworkGuard.LastAttempt,
            metrics_upload_checks = SpireHost.Patches.NetworkGuard.MetricsUploadChecks,
            records = noRecords ? (object)bad : results,
        };
    }

    private static string EventOf(string? detail)
    {
        const string key = "enters the event ";
        int i = detail?.IndexOf(key, StringComparison.Ordinal) ?? -1;
        if (i < 0)
        {
            return "?";
        }
        string s = detail!.Substring(i + key.Length);
        int j = s.IndexOf(';');
        return j > 0 ? s.Substring(0, j) : s;
    }

    public static string ContextKind(string ctx)
    {
        if (ctx.StartsWith("finished action execution ", StringComparison.Ordinal))
        {
            string s = ctx.Substring(26);
            int i = s.IndexOfAny(new[] { ' ', '(', ':', '[' });
            return "finished action execution <" + (i > 0 ? s.Substring(0, i) : s) + ">";
        }
        int colon = ctx.IndexOf(':');
        return colon > 0 && colon < 40 ? ctx.Substring(0, colon) : ctx;
    }
}
