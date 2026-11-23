using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SpireHost.Driver;
using SpireHost.Driver.Tools;

namespace SpireHost.Replay;

/// <summary>
/// --selftest [--sample N] [--skip-run-level]: the replay team's own gate.
///   1. N corpus replays (default 16, spread over blocks / characters) must all pass (every checksum equal).
///   2. Determinism: 4 of them twice -> identical checksums and extended traces; one altered event each -> detected.
///   3. A harness negative control: comparing a replay against the checksums of a DIFFERENT replay of the same
///      encounter must fail (proves the comparison is not vacuous).
///   4. Run-level replay self-check (DriverCommands --replay-run --selfcheck 2) with this module registered.
///   5. Run-trace comparison on 6 golden traces (one per character): each must match to its end.
///   6. Ledger: everything the tools registered is listed as tool-only instrumentation in host\FIDELITY.md.
///   7. Regression tests of the verifiers' findings on the replay tools (parity report F1 - F4) and of the v2 tools:
///      a changed end-of-run dump is a difference (F1); a final-loss fight's recorded decisions are compared (F2); a
///      signature mismatch is not a pass (F3); --alter-event on a non-play is an error (F4); a fight that an event starts
///      is reported as such, a fight that ends before any decision replays; two v2 traces match to their end with every
///      recorded combat state equal, and a changed offer, choice, dump and combat signature are each detected.
/// Exit 0 = all green, 4 = a check failed.
/// </summary>
public static class ReplaySelfTest
{
    public static int Run(HostSession session, Args a)
    {
        List<object> checks = new List<object>();
        bool ok = true;
        void Check(string name, bool pass, object detail)
        {
            checks.Add(new { name, pass, detail });
            ok &= pass;
        }

        // 1. corpus sample
        List<string> sample = ReplayDeterminism.DefaultSample(a.GetInt("sample", 16));
        McrReplayer replayer = new McrReplayer(session.Pump);
        List<McrReplayResult> results = sample.Select(f => replayer.Run(McrFile.Load(f))).ToList();
        Check("corpus_sample_pass", results.All(r => r.Status == "pass") && results.Count > 0, new
        {
            replays = results.Count,
            pass = results.Count(r => r.Status == "pass"),
            checksums = results.Sum(r => r.ChecksumsRecorded),
            matched = results.Sum(r => r.Matched),
            failing = results.Where(r => r.Status != "pass").Select(r => r.File + ":" + r.Status + ":" + r.StatusDetail).ToList(),
        });

        // 2. determinism + alteration
        List<ReplayDeterminism.Row> rows = ReplayDeterminism.Check(session, sample.Take(4).ToList());
        Check("same_replay_twice_identical", rows.All(r => r.ChecksumsEqual && r.ExtendedEqual), rows.Select(r => new { r.File, r.Checksums, r.ChecksumsEqual, r.ExtendedEqual }).ToList());
        Check("altered_event_detected", rows.All(r => r.AlteredEvent < 0 || r.AlterationDetected) && rows.Any(r => r.AlteredEvent >= 0),
            rows.Select(r => new { r.File, r.AlteredEvent, r.AlteredStatus, r.AlteredFirstOrdinal, r.AlteredFirstEventIndex, r.AlteredContext }).ToList());

        // 3. negative control: checksums of another replay of the same encounter
        Check("negative_control_foreign_checksums", NegativeControl(session, out string why), why);

        // 4. run-level replay self-check (the module is registered; the replay tools register nothing)
        if (!a.Has("skip-run-level"))
        {
            int code = DriverCommands.ReplayRun(session, Args.Parse(new[] { "--replay-run", "--selfcheck", a.Get("selfcheck", "2"), "--hp", a.Get("hp", "0") }));
            Check("run_level_replay_selfcheck", code == 0, new { exit_code = code });
        }

        // 5. run-trace comparison: one golden trace per character (120 traces = 6 characters x 2 ascensions x 10 seeds,
        //    ordered by name: stride 20), each to its end
        {
            List<RunTraceCompare.TraceResult> tr = RunTraceCompare.CompareAll(session, RunTraceCompare.SelectTraces(Args.Parse(new[] { "--compare-run-traces", "--stride", "20", "--offset", "3" })), RunTraceCompare.DefaultOptions());
            Check("run_traces_match_to_their_end", tr.Count == 6 && tr.All(t => t.StoppedBy == "trace_end" && t.StartMatches && t.Error == null),
                tr.Select(t => new { t.Trace, t.StartMatches, t.TraceDumps, t.RoomsMatched, t.RoomsMatchedExplained, t.ChoicesMatched, t.TraceChoices, t.StoppedBy, t.StoppedAt, t.Error }).ToList());
        }

        // 6. ledger: the production set is unchanged by the tools and everything they registered (after the session was
        //    created) has a row in the tool-only section of host\FIDELITY.md
        {
            SpireHost.Patches.PatchLedger.Result lr = SpireHost.Patches.PatchLedger.Compare(a.Get("ledger"), session.ProductionPatches, session.ToolPatches, SpireHost.Patches.PatchRegistry.SharedSites());
            List<string> tools = session.ToolPatches.Select(p => p.Id).ToList();
            bool onlyOurs = tools.All(id => id.StartsWith("replay-", StringComparison.Ordinal));
            Check("ledger_lists_tool_instrumentation", lr.Ok && onlyOurs && tools.Count >= 3 && session.ProductionPatchHash == SpireHost.Patches.PatchLedger.Hash(session.ProductionPatches),
                new { ledger = lr.Path, lr.Ok, tool_ids = tools, lr.MissingRows, lr.StaleRows, lr.WrongSection, lr.FieldMismatches, lr.DuplicateRows, lr.SharedSiteProblems, lr.Problems });
        }

        // 7. regression tests (verifier findings F1 - F4; the v2 tools). Skipped only by --skip-regressions (never by a gate).
        if (!a.Has("skip-regressions"))
        {
            Regressions(session, Check);
        }

        HostJson.Emit(new { command = "replay selftest", ok, checks }, a.Get("out"));
        return ok ? 0 : 4;
    }

    private static void Regressions(HostSession session, Action<string, bool, object> check)
    {
        McrReplayer replayer = new McrReplayer(session.Pump);

        // ---- F1: the v1 comparison's tolerance was "any HP-only difference in any dump taken in a combat room" ----
        {
            string? dir = Directory.GetDirectories(OraclePaths.Runs).OrderBy(d => d, StringComparer.Ordinal)
                .FirstOrDefault(d => Directory.GetFiles(d, "*-end-death.json").Length == 1);
            string change = "";
            List<RunTraceCompare.TraceResult> tr = dir == null ? new List<RunTraceCompare.TraceResult>() : RunTraceCompare.CompareAll(session, new List<string> { dir }, RunTraceCompare.DefaultOptions(), t =>
            {
                System.Text.Json.Nodes.JsonNode end = t.Dumps[t.Dumps.Count - 1].Dump;
                string hp = end["hp"]!.GetValue<string>();
                string doctored = (int.Parse(hp.Split('/')[0]) + 1) + "/" + hp.Split('/')[1];
                end["hp"] = doctored;
                change = "end dump hp " + hp + " -> " + doctored;
            });
            check("f1_changed_end_of_run_dump_is_a_difference", tr.Count == 1 && tr[0].StoppedBy == "difference:structure" && (tr[0].StoppedAt ?? "").Contains("dump.hp", StringComparison.Ordinal)
                && tr[0].RoomsMatchedExplained == 0, new { trace = dir == null ? null : Path.GetFileName(dir), change, stopped_by = tr.FirstOrDefault()?.StoppedBy, stopped_at = tr.FirstOrDefault()?.StoppedAt });
        }

        // ---- F2 / F3 / F4 on a fatal fight of the v1 run traces ----
        {
            string? final = Directory.GetDirectories(OraclePaths.Runs).OrderBy(d => d, StringComparer.Ordinal).SelectMany(d => Directory.GetFiles(d, "*-combat-final-loss.mcr")).FirstOrDefault();
            McrReplayResult? plain = final == null ? null : replayer.Run(McrFile.Load(final));
            check("f2_final_loss_fight_compares_its_recorded_decisions", plain != null && plain.Status == "pass" && plain.Extended.PolicyActions > 0 && plain.Extended.Compared == plain.Extended.PolicyActions
                && plain.Extended.SigMismatches == 0, new { file = plain?.File, status = plain?.Status, recorded_decisions = plain?.Extended.PolicyActions, compared = plain?.Extended.Compared });

            McrReplayResult? sig = null;
            if (final != null)
            {
                McrFile f = McrFile.Load(final);
                if (f.Actions != null && f.Actions.Count > 1)
                {
                    f.Actions[1]!["Sig"] = "tampered " + f.Actions[1]!["Sig"]!.GetValue<string>();
                    sig = replayer.Run(f);
                }
            }
            check("f3_signature_mismatch_is_not_a_pass", sig != null && sig.Status == "sig_mismatch" && sig.Mismatched == 0 && sig.Extended.SigMismatches == 1,
                new { file = sig?.File, status = sig?.Status, checksums_mismatched = sig?.Mismatched, sig_mismatches = sig?.Extended.SigMismatches });

            McrReplayResult? nonPlay = null, play = null, beyond = null;
            int nonPlayIndex = -1, playIndex = -1;
            if (final != null)
            {
                McrFile f = McrFile.Load(final);
                for (int i = 0; i < f.Replay.events.Count; i++)
                {
                    bool isPlay = f.Replay.events[i].action is MegaCrit.Sts2.Core.GameActions.NetPlayCardAction;
                    if (isPlay && playIndex < 0)
                    {
                        playIndex = i;
                    }
                    if (!isPlay && nonPlayIndex < 0)
                    {
                        nonPlayIndex = i;
                    }
                }
                nonPlay = replayer.Run(McrFile.Load(final), new ReplayOptions { AlterEventIndex = nonPlayIndex, ContinueAfterDivergence = false });
                play = replayer.Run(McrFile.Load(final), new ReplayOptions { AlterEventIndex = playIndex, ContinueAfterDivergence = false });
                beyond = replayer.Run(McrFile.Load(final), new ReplayOptions { AlterEventIndex = f.Replay.events.Count + 5, ContinueAfterDivergence = false });
            }
            check("f4_alter_event_on_a_non_play_is_an_error", nonPlay != null && nonPlayIndex >= 0 && playIndex >= 0 && nonPlay.Status == "error" && beyond!.Status == "error" && play!.Status != "pass" && play.Status != "error",
                new { non_play_event = nonPlayIndex, non_play_status = nonPlay?.Status, non_play_detail = nonPlay?.StatusDetail, play_event = playIndex, play_status = play?.Status, beyond_the_recording = beyond?.Status });
        }

        // ---- v2 corpus: a fight started by an event option; a fight that ends before any decision ----
        string v2 = V2.RunTraceCompareV2.DefaultRoot;
        {
            string eventFight = Path.Combine(v2, "deep", "DEFECT-A0-DEP200000009", "048-combat-f46.mcr");
            string silent = Path.Combine(v2, "deep", "SILENT-A0-DEP200000021", "048-combat-f46.mcr");
            McrReplayResult? e = File.Exists(eventFight) ? replayer.Run(McrFile.Load(eventFight)) : null;
            McrReplayResult? s = File.Exists(silent) ? replayer.Run(McrFile.Load(silent)) : null;
            check("event_fight_is_reported_and_never_a_pass", e != null && e.Status == "event_fight" && e.EventsInjected == 0, new { file = e?.File, status = e?.Status, detail = e?.StatusDetail });
            check("fight_that_ends_before_any_decision_replays", s != null && s.Status == "pass" && s.Events == 0 && s.ChecksumsRecorded == 1 && s.Matched == 1,
                new { file = s?.File, status = s?.Status, events = s?.Events, checksums = s?.ChecksumsRecorded, matched = s?.Matched });
        }

        // ---- v2 run-trace comparison: two traces to their end; four in-memory alterations, each must be found where it was made ----
        {
            List<string> dirs = new List<string>();
            foreach (string set in new[] { "natural", "deep" })
            {
                string root = Path.Combine(v2, set);
                if (Directory.Exists(root))
                {
                    dirs.AddRange(Directory.GetDirectories(root).Where(V2.V2Trace.IsTraceDir).OrderBy(d => d, StringComparer.Ordinal).Take(1));
                }
            }
            V2.RunTraceCompareV2.Options opt = new V2.RunTraceCompareV2.Options();
            List<V2.RunTraceCompareV2.TraceResult> plain = V2.RunTraceCompareV2.CompareAll(session, dirs, opt);
            check("v2_traces_match_to_their_end_with_every_recorded_combat_state_equal",
                plain.Count == 2 && plain.All(t => t.MatchedToEnd && t.RecordsConsumed == t.Records && t.FightsChecksumsCompared == t.FightsWithReplayFile && t.FightsChecksumsEqual == t.FightsWithReplayFile
                    && t.ChecksumsMatched == t.ChecksumsRecorded && t.ChecksumsRecorded > 0),
                plain.Select(t => new { t.Set, t.Trace, t.MatchedToEnd, t.StoppedBy, t.Records, t.RecordsConsumed, t.DumpsCompared, t.Dumps, t.CombatsReplayed, t.Combats, t.FightsChecksumsEqual, t.ChecksumsRecorded, t.ChecksumsMatched, t.DivergenceCounts, t.Error }).ToList());

            List<object> rows = new List<object>();
            bool allFound = dirs.Count == 2;
            void Alteration(string name, string wantClassPrefix, Action<V2.V2Trace> doctor)
            {
                string what = "";
                List<V2.RunTraceCompareV2.TraceResult> r = V2.RunTraceCompareV2.CompareAll(session, dirs.Take(1).ToList(), opt, false, t => { doctor(t); what = t.Name; });
                string first = r.Count == 1 && r[0].Divergences.Count > 0 ? r[0].Divergences[0].Class : "";
                bool found = r.Count == 1 && !r[0].MatchedToEnd && first.StartsWith(wantClassPrefix, StringComparison.Ordinal);
                allFound &= found;
                rows.Add(new { alteration = name, trace = what, expected_class = wantClassPrefix + "*", first_divergence = first, where = r.Count == 1 && r[0].Divergences.Count > 0 ? r[0].Divergences[0].Where : null, found });
            }
            Alteration("a gold reward's amount + 1", "content.rewards", t =>
            {
                System.Text.Json.Nodes.JsonObject rec = t.Records.First(x => x["kind"]!.GetValue<string>() == "rewards" && x["offer"]!.AsArray().Any(o => o!["type"]!.GetValue<string>() == "GoldReward"));
                System.Text.Json.Nodes.JsonNode item = rec["offer"]!.AsArray().First(o => o!["type"]!.GetValue<string>() == "GoldReward")!;
                item["gold"] = item["gold"]!.GetValue<int>() + 1;
            });
            Alteration("a map choice replaced by another offered node", "dump.structure:dump.coord", t =>
            {
                System.Text.Json.Nodes.JsonObject rec = t.Records.First(x => x["kind"]!.GetValue<string>() == "map" && x["offer"]!.AsArray().Count > 1);
                System.Text.Json.Nodes.JsonObject chosen = rec["chosen"]!.AsObject();
                System.Text.Json.Nodes.JsonNode other = rec["offer"]!.AsArray().First(o => o!["col"]!.GetValue<int>() != chosen["col"]!.GetValue<int>())!;
                chosen["col"] = other["col"]!.GetValue<int>();
                chosen["row"] = other["row"]!.GetValue<int>();
                chosen["type"] = other["type"]!.GetValue<string>();
            });
            Alteration("the gold of the second room dump + 1", "dump.structure:dump.gold", t =>
            {
                t.Dumps[1].Dump["gold"] = t.Dumps[1].Dump["gold"]!.GetValue<int>() + 1;
            });
            Alteration("the recorded signature of the third action of the first fight", "combat.signature", t =>
            {
                System.Text.Json.Nodes.JsonNode action = t.Combats.First(c => c.Actions.Count > 3).Actions[2]!;
                action["Sig"] = action["Sig"]!.GetValue<string>().Replace(" e", " e9");
            });
            check("v2_changed_offer_choice_dump_and_combat_signature_are_each_detected", allFound && rows.Count == 4, rows);
        }
    }

    /// <summary>Replays file A with the recorded checksums of file B (same encounter, different combat): must not pass.</summary>
    private static bool NegativeControl(HostSession session, out string why)
    {
        Dictionary<string, List<string>> byEncounter = new Dictionary<string, List<string>>();
        foreach (string f in Directory.GetFiles(McrCommands.DefaultCorpus, "*.json").OrderBy(f => f, StringComparer.Ordinal).Take(400))
        {
            string enc = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(f))?["encounter"]?.GetValue<string>() ?? "";
            if (!byEncounter.TryGetValue(enc, out List<string>? l))
            {
                byEncounter[enc] = l = new List<string>();
            }
            l.Add(Path.ChangeExtension(f, ".mcr"));
        }
        List<string>? pair = byEncounter.Values.FirstOrDefault(l => l.Count >= 2);
        if (pair == null)
        {
            why = "no encounter with two replays in the first 400 files";
            return false;
        }
        McrFile a = McrFile.Load(pair[0]);
        McrFile b = McrFile.Load(pair[1]);
        a.Replay.checksumData = b.Replay.checksumData;
        a.Replay.nextChecksumId = b.Replay.nextChecksumId;
        McrReplayResult r = new McrReplayer(session.Pump).Run(a, new ReplayOptions { ContinueAfterDivergence = false });
        why = $"{a.Name} replayed against the checksums of {b.Name}: status {r.Status}, first divergence at checksum {r.First?.ChecksumOrdinal}";
        return r.Status != "pass";
    }
}
