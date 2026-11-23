using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SpireHost;
using SpireHost.Driver;
using SpireHost.Driver.Combat.Tests;
using SpireHost.Driver.Tools;

namespace SpireHost.Driver.Combat;

public static class CombatCommands
{
    private static readonly string[] DefaultSuites = { "sources", "entries", "callers", "turnend", "turnend-control", "potions", "costs", "vetoes", "drawperm", "allcards" };

    /// <summary>
    /// --selftest [--suite sources,entries,callers,turnend,turnend-control,potions,costs,vetoes,drawperm,allcards,sweep] [--only ID] [--verbose] [--out FILE]
    ///            sweep only:   [--hp N] [--variant K] [--encounter E] [--character C] [--shard I --shards N] [--turn-cap T]
    ///            callers only: [--seeds N] (policy seeds per caller, default 4)
    /// --crosscheck-mcr [--dir D | --file F | --files a,b | --run-traces] [--start I --count N] [--stride S --offset K] [--out FILE]
    ///                            the handler's legal actions against the real client's recorded decisions, through the
    ///                            replay tool (Tests\McrCrossCheck.cs); also this module's --replay-mcr regression
    /// --list-ids                 cards / potions / relics / powers / encounters as the game enumerates them
    /// --list-patches             the combat module's PatchRegistry entries (rows of host\FIDELITY.md)
    /// Reached as "SpireHost combat --<command>". Like every command they run on the production configuration.
    /// Every case is one scenario episode: real run start, loadout through the game's own commands, a debug
    /// jump into the encounter, then the registered handlers with a probing observer (Tests\CombatProbe.cs).
    /// Exit 4 if a case fails or the "zero rejected / cancelled actions" counters moved.
    /// </summary>
    public static int Run(HostSession session, Args a)
    {
        if (a.Command == "list-ids")
        {
            HostJson.Emit(CombatCatalog.Describe(), a.Get("out"));
            return 0;
        }
        if (a.Command == "list-patches")
        {
            // The combat module's registry entries. Their ledger rows are in the ONE merged ledger host\FIDELITY.md (section 1.7), checked against the
            // registry by --check-ledger; host\FIDELITY.combat.md is the pre-integration fragment, kept for history (verifier finding R-10).
            List<SpireHost.Patches.PatchRecord> mine = SpireHost.Patches.PatchRegistry.Records.Where(r => r.Id.StartsWith("combat-", StringComparison.Ordinal)).ToList();
            HostJson.Emit(new
            {
                kind = "combat-patches",
                count = mine.Count,
                by_category = mine.GroupBy(r => r.Category.ToString()).ToDictionary(g => g.Key, g => g.Count()),
                behaviour_patches = mine.Count(r => r.Category == SpireHost.Patches.PatchCategory.Behaviour),
                patches = mine.Select(r => new { id = r.Id, target = r.Target, kind = r.Kind, category = r.Category.ToString(), reason = r.Reason }).ToList(),
            }, a.Get("out"));
            return 0;
        }
        if (a.Command == "crosscheck-mcr")
        {
            return McrCrossCheck.Run(session, a);
        }
        if (a.Command != "selftest")
        {
            Console.Error.WriteLine("combat: --selftest [--suite sources,entries,callers,turnend,turnend-control,potions,costs,vetoes,drawperm,allcards,sweep] [--only ID] [--hp N] [--shard I --shards N] | --crosscheck-mcr [--run-traces] | --list-ids | --list-patches");
            return 1;
        }
        string[] suites = a.Get("suite") is string s ? s.Split(',', StringSplitOptions.RemoveEmptyEntries) : DefaultSuites;
        string? only = a.Get("only");
        bool verbose = a.Has("verbose");
        Stopwatch wall = Stopwatch.StartNew();
        List<object> suiteReports = new List<object>();
        List<object> failures = new List<object>();
        bool ok = true;
        long totalCases = 0;
        long totalPassed = 0;
        long totalLegality = 0;
        long totalCombatDecisions = 0;
        long totalSelectDecisions = 0;
        foreach (string suite in suites)
        {
            Stopwatch sw = Stopwatch.StartNew();
            List<CaseResult> results;
            object? extra = null;
            switch (suite)
            {
                case "sources":
                    results = Suites.Sources(session, only);
                    break;
                case "entries":
                    results = EntrySuite.Run(session, only);
                    break;
                case "callers":
                    results = CallerSuite.Run(session, only, a.GetInt("seeds", 4), out extra);
                    break;
                case "turnend":
                    results = TurnEndSuite.Run(session, only);
                    break;
                case "turnend-control":
                    // negative control of "turnend": the run-save boundary passed through -> every case must abort in PopAction
                    results = TurnEndSuite.RunNegativeControl(session, only);
                    break;
                case "potions":
                    results = Suites.Potions(session, only);
                    break;
                case "costs":
                    results = Suites.Costs(session, only);
                    break;
                case "vetoes":
                    results = Suites.Vetoes(session, only);
                    break;
                case "drawperm":
                    results = Suites.DrawPermutation(session);
                    break;
                case "allcards":
                    results = Suites.AllCards(session, only, out extra);
                    break;
                case "sweep":
                    results = Suites.Sweep(session, a.GetInt("hp", 0), a.Get("encounter"), a.Get("character"), a.GetInt("shard", 0), a.GetInt("shards", 1), a.GetInt("turn-cap", 60), a.GetInt("variant", 0), out extra);
                    break;
                default:
                    Console.Error.WriteLine("combat: unknown suite '" + suite + "'");
                    return 1;
            }
            int passed = results.Count(r => r.Passed);
            totalCases += results.Count;
            totalPassed += passed;
            totalLegality += results.Sum(r => r.LegalityChecks);
            totalCombatDecisions += results.Sum(r => r.CombatDecisions);
            totalSelectDecisions += results.Sum(r => r.SelectDecisions);
            ok &= passed == results.Count;
            foreach (CaseResult r in results)
            {
                if (!r.Passed)
                {
                    failures.Add(r.ToJson(verbose: true));
                }
            }
            suiteReports.Add(new
            {
                suite,
                cases = results.Count,
                passed,
                failed = results.Count - passed,
                seconds = Math.Round(sw.Elapsed.TotalSeconds, 1),
                outcomes = results.GroupBy(r => r.Outcome == "Truncated" ? "Truncated:" + r.TruncatedBy : r.Outcome).ToDictionary(g => g.Key, g => g.Count()),
                combat_decisions = results.Sum(r => r.CombatDecisions),
                legality_checks = results.Sum(r => r.LegalityChecks),
                select_decisions = results.Sum(r => r.SelectDecisions),
                plays = results.Sum(r => r.Plays),
                summary = extra,
                results = verbose ? results.Select(r => r.ToJson(true)).ToList() : null,
                notes = suite == "sweep" && !verbose ? null : results.Where(r => r.Note != null && r.Passed && suite != "sweep").Select(r => r.Name + ": " + r.Note).Take(200).ToList(),
            });
        }
        if (CombatStats.Violations != 0)
        {
            ok = false;
        }
        Dictionary<string, object> entries = new Dictionary<string, object>();
        for (int i = 1; i < SelectionCapture.EntryCalls.Length; i++)
        {
            entries[((CardSelectEntry)i).ToString()] = new { calls = SelectionCapture.EntryCalls[i], selector_asked = SelectionCapture.EntryAsked[i] };
        }
        HostJson.Emit(new
        {
            kind = "combat-selftest",
            ok,
            combat_handler = session.Registry.GetRoomHandler(RoomScreen.Combat)?.Name,
            card_selector = session.Registry.FindLifecycle<CombatLifecycle>()?.Name,
            suites = suiteReports,
            totals = new
            {
                cases = totalCases,
                passed = totalPassed,
                failed = totalCases - totalPassed,
                combat_decisions = totalCombatDecisions,
                legality_checks = totalLegality,
                select_decisions = totalSelectDecisions,
                wall_seconds = Math.Round(wall.Elapsed.TotalSeconds, 1),
            },
            combat = CombatStats.Snapshot(),
            card_select = CardSelectStats.Snapshot(),
            from_entry_points = entries,
            entry_points_patched = SelectionCapture.PatchCount,
            class_provider = session.Registry.CanonicalCardClassesName,
            class_provider_calls = CanonicalCandidates.ProviderCalls,
            class_provider_pair_calls = CanonicalCandidates.ProviderPairCalls,
            frame_audit = FrameAudit.Snapshot(),
            run_save_boundaries_combat_end = SpireHost.Patches.SchedulingPatches.RunSaveBoundariesCombatEnd,
            run_save_boundaries_skipped = SpireHost.Patches.SchedulingPatches.RunSaveBoundariesSkipped,
            modules = session.Registry.Modules,
            failures,
        }, a.Get("out"));
        return ok ? 0 : 4;
    }
}
