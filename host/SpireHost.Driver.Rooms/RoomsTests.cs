using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SpireHost;
using SpireHost.Driver;
using SpireHost.Boot;
using SpireHost.Driver.Tools;

namespace SpireHost.Driver.Rooms;

internal static class RoomsTests
{
    public static int SelfTest(HostSession session, Args a)
    {
        string chars = a.Get("characters", "IRONCLAD");
        string[] characters = chars.Equals("all", StringComparison.OrdinalIgnoreCase) ? DriverCommands.AllCharacters : chars.Split(',', StringSplitOptions.RemoveEmptyEntries);
        string? group = a.Get("group");
        string? only = a.Get("scenario");
        int seeds = a.GetInt("policy-seeds", 1);
        List<RoomScenario> battery = RoomScenarios.Battery();
        List<object> results = new List<object>();
        int passed = 0, failed = 0;
        Stopwatch sw = Stopwatch.StartNew();
        foreach (string character in characters)
        {
            foreach (RoomScenario sc in battery)
            {
                if (group != null && sc.Group != group)
                {
                    continue;
                }
                if (only != null && !sc.Name.Contains(only, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                for (int k = 0; k < seeds; k++)
                {
                    HostLog.Info("rooms selftest: " + character + ": " + sc.Name + " (policy seed " + (1 + k) + ")");
                    object res = RoomScenarios.RunOne(session, sc, character.ToUpperInvariant(), (ulong)(1 + k), out bool ok);
                    results.Add(res);
                    if (ok)
                    {
                        passed++;
                    }
                    else
                    {
                        failed++;
                        HostLog.Error("rooms selftest FAILED: " + character + ": " + sc.Name + " (policy seed " + (1 + k) + ")");
                    }
                }
            }
        }
        HostJson.Emit(new
        {
            kind = "rooms-selftest",
            ok = failed == 0,
            passed,
            failed,
            wall_s = Math.Round(sw.Elapsed.TotalSeconds, 1),
            registry = session.Registry.Describe(),
            results,
        }, a.Get("out"));
        return failed == 0 ? 0 : 4;
    }

    /// <summary>
    /// Debug: replays one fuzz repro tuple (seed, character, ascension, policy seed, hp) with RandomLegalPolicy and prints the
    /// last N decisions (kind, chosen action, room, pending stack, executor) plus the result. Same policy as --fuzz-runs.
    /// </summary>
    public static int TraceRun(HostSession session, Args a)
    {
        int tail = a.GetInt("tail", 40);
        EpisodeSpec spec = new EpisodeSpec { Seed = a.Get("seed"), Characters = new[] { a.Get("character", "IRONCLAD") }, Ascension = a.GetInt("ascension", 0) };
        EpisodeOptions o = DriverCommands.EpisodeOptionsFrom(a);
        o.DebugForcedHp = a.GetInt("hp", 0);
        TailObserver obs = new TailObserver(tail);
        EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy(a.GetULong("policy-seed", 1)), o, obs);
        HostJson.Emit(new
        {
            kind = "rooms-trace-run",
            seed = r.Seed,
            outcome = r.Outcome.ToString(),
            abort = r.Abort?.Signature,
            message = r.Abort?.Message,
            frames = r.Abort?.Frames,
            diagnostics = r.Abort?.Diagnostics,
            policy_decisions = r.Counters.PolicyDecisions,
            last = obs.Lines.ToArray(),
        }, a.Get("out"));
        return 0;
    }

    private sealed class TailObserver : IEpisodeObserver
    {
        private readonly int _n;
        public readonly Queue<string> Lines = new Queue<string>();

        public TailObserver(int n) => _n = n;

        public void OnEpisodeStart(RunContext ctx)
        {
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }

        public void OnDecision(RunContext ctx, Decision d, int chosen, bool auto)
        {
            LegalAction act = d.Actions[chosen];
            string src = act.Source.Kind + ":" + act.Source.Index + (act.Source.Object is MegaCrit.Sts2.Core.Models.AbstractModel m ? "(" + m.Id.Entry + ")" : act.Source.Object is MegaCrit.Sts2.Core.Entities.Players.Player p && act.Source.Kind == RefKind.PotionSlot ? "(" + p.PotionSlots[act.Source.Index]?.Id.Entry + ")" : "");
            Lines.Enqueue($"#{d.Sequence} {d.Kind}/{d.SubKind}{(auto ? "*" : "")}{(d.Nested ? " nested" : "")} n={d.Actions.Count} -> {act.Verb} {src} tgt={act.Target.Kind}:{act.Target.Index} owner={(act.Owner ?? d.Source)?.Name} | room={ctx.CurrentRoom?.GetType().Name} map={ctx.MapOpen} stack={ctx.PendingCount}");
            while (Lines.Count > _n)
            {
                Lines.Dequeue();
            }
        }
    }

    /// <summary>Whole random runs with the proceed-tree counters aggregated (the probe is process-wide and only counts).</summary>
    public static int TreeRuns(HostSession session, Args a)
    {
        int runs = a.GetInt("runs", 24);
        string[] characters = a.Get("characters", string.Join(",", DriverCommands.AllCharacters)).Split(',', StringSplitOptions.RemoveEmptyEntries);
        int[] ascensions = a.Get("ascensions", "0,10").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        int hp = a.GetInt("hp", 99999);
        string seedBase = a.Get("seed-base", "ROOMSTREE");
        ulong policySeed = a.GetULong("policy-seed", 7);
        RoomsProbe.Reset();
        SortedDictionary<string, int> outcomes = new SortedDictionary<string, int>(StringComparer.Ordinal);
        List<object> aborts = new List<object>();
        int wins = 0, maxFloor = 0;
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < runs; i++)
        {
            string character = characters[i % characters.Length];
            int asc = ascensions[(i / characters.Length) % ascensions.Length];
            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel(seedBase + ":" + i), Characters = new[] { character }, Ascension = asc };
            EpisodeOptions o = DriverCommands.EpisodeOptionsFrom(a);
            o.DebugForcedHp = hp;
            EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy(policySeed + (ulong)i), o);
            string key = r.Outcome + (r.TruncatedBy != null ? ":" + r.TruncatedBy : "");
            outcomes[key] = outcomes.GetValueOrDefault(key) + 1;
            wins += r.Outcome == EpisodeOutcome.Win ? 1 : 0;
            maxFloor = Math.Max(maxFloor, r.TotalFloor);
            if (r.Outcome == EpisodeOutcome.Aborted && aborts.Count < 50)
            {
                aborts.Add(new { index = i, seed = r.Seed, character, ascension = asc, signature = r.Abort?.Signature, message = r.Abort?.Message, frames = r.Abort?.Frames });
            }
        }
        bool ok = RoomsProbe.TreeChecksFailed == 0;
        HostJson.Emit(new
        {
            kind = "rooms-tree-runs",
            ok,
            runs,
            hp,
            ascensions,
            wins,
            max_total_floor = maxFloor,
            outcomes,
            aborts,
            wall_s = Math.Round(sw.Elapsed.TotalSeconds, 1),
            probe = RoomsProbe.Snapshot(),
        }, a.Get("out"));
        return ok ? 0 : 4;
    }
}
