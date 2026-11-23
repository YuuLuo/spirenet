using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using SpireHost.Driver;

namespace SpireHost.Replay.V2;

/// <summary>
/// --compare-run-traces-v2 [--dir D | --set natural|deep | --root R] [--traces A,B | --start I --count N] [--stride S --offset K]
///                         [--stop-at-first] [--no-save] [--no-known-masks] [--strict-sampling] [--no-mcr-identity] [--no-explained]
///                         [--no-checksums] [--wongo seed|N] [--max-divergences N] [--max-diffs N] [--dump-host DIR] [--no-records]
///                         [--progress] [--out FILE]
///
/// The host against the real client's run traces of format v2 (oracle\runs-v2, notes\oracle-deep.md): an auto-play mod
/// in the real v0.113.0 client wrote, BEFORE every action outside combat, a decision record with the complete offer
/// read from the UI nodes and their enabled flags; per fight a combat.json (actions, selection prompts and answers)
/// and the game's own combat replay; and at every room boundary a dump + the game's own run-save JSON.
///
/// For each trace the host plays ONE RunDriver episode on the production configuration - the trace's seed, character
/// and ascension, the host's veteran profile with WongoPoints 0 (the trace profile's value; tool-only switch
/// EpisodeOptions.DebugWongoPoints), and for deep traces the forced-HP fixture exactly as the trace header declares it
/// (EpisodeOptions.DebugForcedHp) - with AutoSingle and AutoGold off and AllowAbandon on, so that the host asks wherever
/// the mod decided. The policy replays the recorded choices one to one through the production handlers, and at EVERY
/// recorded decision compares what the real UI offered with what the host offers:
///
///   legal-action oracle   every item the real UI showed as enabled <-> exactly one legal host action, and back
///   public content        ids, upgrades, enchantments, prices, sale flags, gold amounts, text keys, flags, cells
///   state                 the record's floor / act / room / coord / HP / gold against the host at that decision
///   room boundaries       the tracer's dump and the game's own RunManager.ToSave JSON (ignore list: the v1 tool's plus
///                         the fields the validation report proves irreproducible in the real client itself)
///   combat inside the run the recorded actions and prompt answers through the production combat and selection
///                         handlers; per action the collector's public signature and legal-option count; and, for every
///                         fight with a replay file, the game's own per-action full-combat-state checksums (the states the
///                         real client recorded) against the host's states at the same points of the run
///
/// Where the two representations legitimately differ, the difference is a NAMED rule in the code (V2Classes) with the
/// UI source that proves it; every application is counted ("explained") and reported. Nothing is ignored wholesale.
/// Anything else is a divergence: recorded with its context, counted by class, and - when the run can no longer be
/// replayed faithfully (state or choice mismatch) - the end of that trace's comparison.
/// Exit 0 = every trace matched to its end with zero unexplained divergences, 5 otherwise.
/// </summary>
public static class RunTraceCompareV2
{
    public static string DefaultRoot => Path.Combine(Path.GetDirectoryName(OraclePaths.Runs)!, "runs-v2");

    public sealed class Options
    {
        public bool CompareSave = true;
        public bool StopAtFirst;
        public int MaxDivergences = 10;
        public int MaxDiffs = 12;
        public string? DumpDir;

        /// <summary>--strict-sampling: no combat-entry sampling rule (a combat-room dump must equal the host's first decision point).</summary>
        public bool StrictSampling;

        /// <summary>--no-mcr-identity: identify recorded combat actions / prompt answers from combat.json only (test switch).</summary>
        public bool UseMcr = true;

        /// <summary>WongoPoints of the comparison profile: 0 = the trace profile's value; -1 = derive from the seed (the production rule; test switch --wongo seed).</summary>
        public int WongoPoints;

        /// <summary>Save fields whose difference is explained by the profile recipes (counted, not a stop). --no-known-masks clears it.</summary>
        public HashSet<string> KnownSaveDiffs = new HashSet<string>(StringComparer.Ordinal)
        {
            "discovered_cards", "discovered_relics", "discovered_potions", "discovered_enemies", "discovered_epochs",
        };

        /// <summary>
        /// --no-explained (test switch): the named rules that excuse a recorded item, a recorded click or the early end of a trace
        /// are not applied; each place where one would apply then shows as a divergence. The rules that only count how UI items and
        /// steps map to host actions are unaffected (V2Replayer.Rule / Explain).
        /// </summary>
        public bool ApplyExplainedRules = true;

        /// <summary>
        /// Compare, for every fight that has a replay file, the game's own per-action full-combat-state checksums recorded by the real
        /// client with the states the host is in at the same points of the run (--no-checksums switches it off: test switch, used to
        /// prove that the read-only tracker does not change the episode).
        /// </summary>
        public bool CompareChecksums = true;
    }

    public sealed class Divergence
    {
        public string Class = "";
        public bool Hard;
        public int Seq = -1;
        public string Where = "";
        public string Detail = "";
        public string? Trace;
        public string? Host;
    }

    public sealed class TraceResult
    {
        public string Trace = "";
        public string Set = "";
        public string Character = "";
        public int Ascension;
        public string Seed = "";
        public string TraceEnd = "";
        public string ModVersion = "";
        public int ForcedHp;

        public int Records;
        public int RecordsConsumed;
        public int DecisionsCompared;            // records whose offer / state was compared with a host decision
        public int DecisionsMatched;             // ... without an unexplained difference
        public int ActionsApplied;               // recorded choices applied through the production handlers
        public int OfferItemsCompared;           // UI items / host actions put side by side
        public int ContentFieldsCompared;
        public SortedDictionary<string, int> RecordsByKind = new SortedDictionary<string, int>(StringComparer.Ordinal);
        public SortedDictionary<string, int> ComparedByKind = new SortedDictionary<string, int>(StringComparer.Ordinal);

        public int Dumps;
        public int DumpsCompared;
        public int DumpsMatched;                 // structure and save identical to the host's first decision point of the room
        public int DumpsMatchedBySampling;       // identical once the combat-entry sampling rule is applied
        public int SavesCompared;

        public int Combats;
        public int CombatsReplayed;              // fights replayed to their recorded end
        public int CombatActions;                // recorded actions in the trace
        public int CombatActionsReplayed;
        public int CombatActionsByMcrIdentity;
        public int CombatSigCompared;
        public int CombatSigMatched;
        public int CombatOptionCountsCompared;
        public int CombatOptionCountsEqual;
        public int CombatPrompts;
        public int CombatPromptsReplayed;

        public int FightsWithReplayFile;         // fights of the trace that have a .mcr
        public int FightsChecksumsCompared;      // ... whose recorded per-action states were compared with the host's inside the run
        public int FightsChecksumsEqual;         // ... with every state equal
        public int ChecksumsRecorded;            // recorded states in the compared fights
        public int ChecksumsCompared;
        public int ChecksumsMatched;

        public string StoppedBy = "";
        public string? StoppedAt;
        public bool MatchedToEnd;
        public int HostFloorReached;
        public string HostOutcome = "";
        public string? HostAbort;
        public string? Error;
        public double Ms;

        /// <summary>Unexplained differences by class.</summary>
        public SortedDictionary<string, int> DivergenceCounts = new SortedDictionary<string, int>(StringComparer.Ordinal);

        /// <summary>The first divergences with full context.</summary>
        public List<Divergence> Divergences = new List<Divergence>();

        /// <summary>Named, justified rules that were applied, by class (V2Classes).</summary>
        public SortedDictionary<string, int> Explained = new SortedDictionary<string, int>(StringComparer.Ordinal);

        /// <summary>Informational counters of the comparison itself (how identities were established, what was checked how often). Never a verdict.</summary>
        public SortedDictionary<string, int> Info = new SortedDictionary<string, int>(StringComparer.Ordinal);
    }

    public static List<string> SelectTraces(Args a)
    {
        List<string> all = new List<string>();
        if (a.Get("dir") is string dir)
        {
            all.AddRange(Directory.GetDirectories(dir).Where(V2Trace.IsTraceDir));
        }
        else
        {
            string root = a.Get("root", DefaultRoot);
            string sets = a.Get("set", "natural,deep");
            foreach (string s in sets.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                all.AddRange(Directory.GetDirectories(Path.Combine(root, s)).Where(V2Trace.IsTraceDir));
            }
        }
        all = all.OrderBy(d => Path.GetFileName(Path.GetDirectoryName(d)) + "/" + Path.GetFileName(d), StringComparer.Ordinal).ToList();
        if (a.Get("traces") is string list)
        {
            HashSet<string> want = new HashSet<string>(list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries), StringComparer.OrdinalIgnoreCase);
            return all.Where(d => want.Contains(Path.GetFileName(d))).ToList();
        }
        List<string> sel = all.Skip(a.GetInt("start", 0)).Take(a.GetInt("count", int.MaxValue)).ToList();
        int stride = Math.Max(1, a.GetInt("stride", 1));
        int offset = a.GetInt("offset", 0);
        return stride > 1 ? sel.Where((_, i) => i % stride == offset).ToList() : sel;
    }

    public static Options OptionsFrom(Args a)
    {
        Options opt = new Options
        {
            CompareSave = !a.Has("no-save"),
            StopAtFirst = a.Has("stop-at-first"),
            MaxDivergences = a.GetInt("max-divergences", 10),
            MaxDiffs = a.GetInt("max-diffs", 12),
            DumpDir = a.Get("dump-host"),
            StrictSampling = a.Has("strict-sampling"),
            UseMcr = !a.Has("no-mcr-identity"),
            ApplyExplainedRules = !a.Has("no-explained"),
            CompareChecksums = !a.Has("no-checksums"),
        };
        if (a.Has("no-known-masks"))
        {
            opt.KnownSaveDiffs.Clear();
        }
        string wongo = a.Get("wongo", "0");
        opt.WongoPoints = wongo.Equals("seed", StringComparison.OrdinalIgnoreCase) ? -1 : int.Parse(wongo);
        return opt;
    }

    public static int Run(HostSession session, Args a)
    {
        Options opt = OptionsFrom(a);
        List<string> dirs = SelectTraces(a);
        if (dirs.Count == 0)
        {
            Console.Error.WriteLine("compare-run-traces-v2: no trace selected");
            return 1;
        }
        Stopwatch sw = Stopwatch.StartNew();
        List<TraceResult> results = CompareAll(session, dirs, opt, a.Has("progress"));
        bool all = results.Count > 0 && results.All(r => r.MatchedToEnd);
        HostJson.Emit(Summarize(session, results, opt, sw.Elapsed.TotalSeconds, a.Has("no-records")), a.Get("out"));
        return all ? 0 : 5;
    }

    /// <param name="doctor">Self-tests only: changes the loaded trace in memory (a negative control); never set by a command.</param>
    public static List<TraceResult> CompareAll(HostSession session, List<string> traceDirs, Options opt, bool progress = false, Action<V2Trace>? doctor = null)
    {
        V2Sampling.Register();
        if (opt.CompareChecksums)
        {
            ReplayInterventions.Register(replayRun: false);
        }
        List<TraceResult> results = new List<TraceResult>();
        foreach (string dir in traceDirs)
        {
            Stopwatch sw = Stopwatch.StartNew();
            TraceResult r = new TraceResult { Trace = Path.GetFileName(dir) };
            try
            {
                V2Trace t = V2Trace.Load(dir);
                doctor?.Invoke(t);
                r.Set = t.Set;
                r.Character = t.Character;
                r.Ascension = t.Ascension;
                r.Seed = t.Seed;
                r.TraceEnd = t.End;
                r.ModVersion = t.ModVersion;
                r.ForcedHp = t.ForcedHp;
                r.Records = t.Records.Count;
                r.Dumps = t.Dumps.Count;
                r.Combats = t.Combats.Count;
                r.CombatActions = t.Combats.Sum(c => c.Actions.Count);
                r.CombatPrompts = t.Combats.Sum(c => c.Prompts.Count);
                r.FightsWithReplayFile = t.Combats.Count(c => c.McrPath != null);
                foreach (JsonObject rec in t.Records)
                {
                    string k = rec["kind"]!.GetValue<string>();
                    r.RecordsByKind[k] = r.RecordsByKind.GetValueOrDefault(k) + 1;
                }
                V2Replayer replayer = new V2Replayer(t, r, opt);
                EpisodeOptions eo = new EpisodeOptions
                {
                    AutoSingle = false,                 // the host asks wherever the mod decided
                    AutoGold = false,                   // random-v2 skips gold with p = 0.15 (oracle-deep.md section 9)
                    AllowAbandon = true,                // Trial double-down "yes" is a recorded choice
                    DebugForcedHp = t.ForcedHp,         // the deep fixture exactly as the header declares it; 0 for natural traces
                    DebugWongoPoints = opt.WongoPoints, // the trace profile has 0 points
                    MaxCombatTurns = 0,                 // the trace's own cap (150 player turns) ends its fights
                };
                EpisodeResult er = session.RunEpisode(new EpisodeSpec { Seed = t.Seed, Characters = new[] { t.Character }, Ascension = t.Ascension }, replayer, eo, replayer);
                r.HostOutcome = er.Outcome.ToString();
                r.HostFloorReached = er.TotalFloor;
                if (er.Abort != null)
                {
                    r.HostAbort = er.Abort.Signature + " " + er.Abort.Message + " | " + er.Abort.Diagnostics;
                    replayer.HostAborted(er);
                }
                if (r.StoppedBy.Length == 0)
                {
                    r.StoppedBy = er.Outcome == EpisodeOutcome.Aborted ? "host_abort" : "host_end:" + er.Outcome + (er.TruncatedBy != null ? ":" + er.TruncatedBy : "");
                }
                r.MatchedToEnd = r.StoppedBy.StartsWith("trace_end", StringComparison.Ordinal) && r.DivergenceCounts.Count == 0 && r.Error == null && r.HostAbort == null;
            }
            catch (Exception e)
            {
                r.Error = e.GetBaseException().GetType().Name + ": " + e.GetBaseException().Message;
                r.StoppedBy = "error";
            }
            r.Ms = Math.Round(sw.Elapsed.TotalMilliseconds, 1);
            results.Add(r);
            if (progress)
            {
                Console.Error.WriteLine($"[v2] {r.Set}/{r.Trace}: {(r.MatchedToEnd ? "match" : "DIFF")} stopped_by={r.StoppedBy} records={r.RecordsConsumed}/{r.Records} dumps={r.DumpsMatched + r.DumpsMatchedBySampling}/{r.Dumps} " +
                                        $"combats={r.CombatsReplayed}/{r.Combats} div={string.Join(",", r.DivergenceCounts.Select(kv => kv.Key + ":" + kv.Value))} ({r.Ms:0} ms)");
            }
        }
        return results;
    }

    public static object Summarize(HostSession session, List<TraceResult> results, Options opt, double seconds, bool noRecords)
    {
        SortedDictionary<string, int> Sum(Func<TraceResult, SortedDictionary<string, int>> f)
        {
            SortedDictionary<string, int> t = new SortedDictionary<string, int>(StringComparer.Ordinal);
            foreach (TraceResult r in results)
            {
                foreach (KeyValuePair<string, int> kv in f(r))
                {
                    t[kv.Key] = t.GetValueOrDefault(kv.Key) + kv.Value;
                }
            }
            return t;
        }
        Dictionary<string, object> classes = new Dictionary<string, object>();
        foreach (IGrouping<string, (string cls, TraceResult r, int n)> g in results.SelectMany(r => r.DivergenceCounts.Select(kv => (cls: kv.Key, r, n: kv.Value))).GroupBy(x => x.cls).OrderByDescending(g => g.Sum(x => x.n)))
        {
            classes[g.Key] = new
            {
                count = g.Sum(x => x.n),
                traces = g.Count(),
                first_divergence_of = results.Count(r => r.Divergences.Count > 0 && r.Divergences[0].Class == g.Key),
                examples = g.Take(6).Select(x => x.r.Set + "/" + x.r.Trace).ToList(),
            };
        }
        Dictionary<string, object> explained = new Dictionary<string, object>();
        foreach (IGrouping<string, (string cls, TraceResult r, int n)> g in results.SelectMany(r => r.Explained.Select(kv => (cls: kv.Key, r, n: kv.Value))).GroupBy(x => x.cls).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            explained[g.Key] = new { count = g.Sum(x => x.n), traces = g.Count(), kind = V2Classes.KindOf(g.Key), rule = V2Classes.RuleOf(g.Key) };
        }
        List<TraceResult> bad = results.Where(r => !r.MatchedToEnd).ToList();
        return new
        {
            command = "compare-run-traces-v2",
            vocab_hash = session.Registry.VocabHash,
            production_patches = session.ProductionPatches.Count,
            production_patch_hash = session.ProductionPatchHash,
            tool_patches = session.ToolPatches.Select(p => p.Id).ToList(),
            comparison = new
            {
                auto_single = false,
                auto_gold = false,
                allow_abandon = true,
                wongo_points = opt.WongoPoints,
                compare_save = opt.CompareSave,
                known_save_diffs_masked = opt.KnownSaveDiffs,
                combat_entry_sampling_rule = !opt.StrictSampling,
                mcr_identity = opt.UseMcr,
                explained_rules_applied = opt.ApplyExplainedRules,
                in_run_checksums = opt.CompareChecksums,
                stop_at_first = opt.StopAtFirst,
            },
            traces = results.Count,
            by_set = results.GroupBy(r => r.Set).OrderBy(g => g.Key).ToDictionary(g => g.Key, g => new { traces = g.Count(), matched_to_end = g.Count(r => r.MatchedToEnd) }),
            matched_to_end = results.Count(r => r.MatchedToEnd),
            not_matched = bad.Count,
            stopped_by = results.GroupBy(r => r.StoppedBy).OrderByDescending(g => g.Count()).ToDictionary(g => g.Key, g => g.Count()),
            records_in_traces = results.Sum(r => r.Records),
            records_consumed = results.Sum(r => r.RecordsConsumed),
            decisions_compared = results.Sum(r => r.DecisionsCompared),
            decisions_matched = results.Sum(r => r.DecisionsMatched),
            actions_applied = results.Sum(r => r.ActionsApplied),
            offer_items_compared = results.Sum(r => r.OfferItemsCompared),
            content_fields_compared = results.Sum(r => r.ContentFieldsCompared),
            records_by_kind = Sum(r => r.RecordsByKind),
            compared_by_kind = Sum(r => r.ComparedByKind),
            dumps_in_traces = results.Sum(r => r.Dumps),
            dumps_compared = results.Sum(r => r.DumpsCompared),
            dumps_matched = results.Sum(r => r.DumpsMatched),
            dumps_matched_by_sampling_rule = results.Sum(r => r.DumpsMatchedBySampling),
            saves_compared = results.Sum(r => r.SavesCompared),
            combats_in_traces = results.Sum(r => r.Combats),
            combats_replayed_to_their_end = results.Sum(r => r.CombatsReplayed),
            combat_actions_in_traces = results.Sum(r => r.CombatActions),
            combat_actions_replayed = results.Sum(r => r.CombatActionsReplayed),
            combat_actions_identified_by_mcr = results.Sum(r => r.CombatActionsByMcrIdentity),
            combat_signatures_compared = results.Sum(r => r.CombatSigCompared),
            combat_signatures_matched = results.Sum(r => r.CombatSigMatched),
            combat_option_counts_compared = results.Sum(r => r.CombatOptionCountsCompared),
            combat_option_counts_equal = results.Sum(r => r.CombatOptionCountsEqual),
            combat_prompts_in_traces = results.Sum(r => r.CombatPrompts),
            combat_prompts_replayed = results.Sum(r => r.CombatPromptsReplayed),
            fights_with_replay_file = results.Sum(r => r.FightsWithReplayFile),
            fights_checksums_compared = results.Sum(r => r.FightsChecksumsCompared),
            fights_checksums_equal = results.Sum(r => r.FightsChecksumsEqual),
            checksums_recorded = results.Sum(r => r.ChecksumsRecorded),
            checksums_compared = results.Sum(r => r.ChecksumsCompared),
            checksums_matched = results.Sum(r => r.ChecksumsMatched),
            divergence_classes = classes,
            explained_classes = explained,
            info = Sum(r => r.Info),
            seconds = Math.Round(seconds, 2),
            results = noRecords ? bad : results,
        };
    }
}
