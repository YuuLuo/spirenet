using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using SpireHost.Driver;

namespace SpireHost.Replay;

/// <summary>
/// --compare-run-traces [--dir DIR] [--traces A,B | --start I --count N] [--no-save] [--max-diffs N] [--out FILE]
///
/// Run-start and room-boundary parity against the oracle's golden run traces (oracle\runs\&lt;CHAR&gt;-A&lt;asc&gt;-&lt;SEED&gt;,
/// recorded in the real client with the scripted policy of realgame\mod-src\SpireOracle\RunTracer.cs; notes\oracle-mod.md 9).
/// For each trace the host plays a normal RunDriver episode on the PRODUCTION configuration (veteran profile, the
/// trace's seed / character / ascension, the production module and patch set; no prototype fix, no test switch) with
/// a policy that replays the trace's documented scripted policy where the host decision maps one-to-one, and an
/// observer that, at every room boundary (IEpisodeObserver.OnRoomBoundary: the first quiescent point after run start
/// and after every RunManager.RoomEntered, before anything in the room is answered - the host analogue of "RoomEntered
/// and the room transition finished", where the trace dumps), rebuilds the trace's public dump (RunTracer.Dump: act list,
/// bosses, room, coord, HP, gold, deck, relics, potions, the act map with edges, event options, shop inventory)
/// and the game's own RunManager.ToSave(null) JSON and compares both with the trace's NNN-*.json / NNN-*.save.json.
/// It also compares the choice log (summary.json choices) and the card reward offers in order.
///
/// Mapping of the trace's scripted policy onto host decisions (realgame\mod-src\SpireOracle\RunTracer.cs header, Screens.cs):
///   * selection screens take "the first k cards in UI order": the host's CARD_SELECT options are canonical classes
///     (DRIVER-API 6.4), so the policy asks the frame's HOST-ONLY view (ICardSelectHostView on Decision.Source) for the
///     candidates in the order the game handed them to the screen, takes the first one that is not picked yet and
///     selects the option (class) that contains it. A verification tool may read that order; a policy never sees it.
///   * treasure: OPEN_CHEST when it is offered (the CHEST page is asked when potion actions are present and
///     auto-advanced otherwise; both log the trace's "treasure open"), then the first relic, then PROCEED.
///   * potions are never used or discarded outside the trace policy's rules (the scripted policy never does).
/// The comparison stops at the first difference or the first decision that cannot be mapped, and reports how far
/// each trace matched and what blocked it. Exit 0 = every trace matched to its end, 5 otherwise.
/// </summary>
public static class RunTraceCompare
{
    /// <summary>oracle\runs next to the host directory (OraclePaths).</summary>
    public static string DefaultDir => OraclePaths.Runs;

    public sealed class Options
    {
        public bool CompareSave = true;
        public int MaxDiffs = 12;
        public int MaxPlaysPerTurn = 40;   // runs config max_actions_per_turn
        public string? DumpDir;            // --dump-host DIR: write the host's dump + save JSON next to each compared trace dump
        /// <summary>
        /// The combat-entry sampling rule (--strict-hp-timing / --strict-sampling disables it). The tracer dumps a combat room at a
        /// frame boundary inside the turn-1 start sequence; the host's room boundary is its first decision point, after it. A "room"
        /// dump of a combat room may therefore differ from the host's first decision point - but only
        /// if the WHOLE dump and save equal ONE of the states the host itself went through in that sequence (sampled at
        /// CombatManager.CombatBegan and after every CombatHistory entry). Verifier finding F1: the earlier rule accepted ANY
        /// HP-only difference, of any sign and size, in ANY dump taken in a combat room - including the end-of-run dump.
        /// </summary>
        public bool TolerateCombatStartHp = true;

        /// <summary>
        /// Save fields whose difference is EXPLAINED and reported (counted per trace) but does not stop the comparison.
        /// players[].discovered_*: the run's "newly discovered" bookkeeping (Player.Discovered*, Player.cs:142-211) depends on
        /// what the PROFILE has already seen; the trace profile (realgame\mod-src\SpireOracle\Profile.cs) marks no card / relic /
        /// potion / event / enemy as seen, the host's veteran profile (SpireHost.Core VeteranProfile.Build) marks all of them.
        /// Profile recipe difference (class c), no gameplay effect.
        /// </summary>
        public HashSet<string> KnownSaveDiffs = new HashSet<string>(StringComparer.Ordinal)
        {
            "discovered_cards", "discovered_relics", "discovered_potions", "discovered_enemies", "discovered_epochs",
        };
    }

    public static Options DefaultOptions() => new Options();

    public sealed class Diff
    {
        public int Dump;
        public string Kind = "";
        public string Path = "";
        public string? Trace;
        public string? Host;
    }

    public sealed class TraceResult
    {
        public string Trace = "";
        public string Character = "";
        public int Ascension;
        public string Seed = "";
        public string TraceEnd = "";
        public int TraceDumps;
        public int TraceChoices;
        public int TraceCardRewardOffers;
        public bool StartCompared;
        public bool StartMatches;
        public bool StartStructureMatches;
        public bool StartSaveMatches;
        public int DumpsCompared;
        public int RoomsMatched;               // dumps (incl. start) whose structure AND save matched, in order before the first difference
        public int StructureMatched;           // dumps whose public structure matched (before the first difference)
        public int ChoicesCompared;
        public int ChoicesMatched;
        public int CardRewardOffersCompared;
        public int CardRewardOffersMatched;
        public int HostFloorReached;
        public string StoppedBy = "";
        public string? StoppedAt;
        public List<Diff> FirstDiffs = new List<Diff>();
        public Dictionary<string, int> KnownDiffsMasked = new Dictionary<string, int>();
        public int CardSelectsMapped;          // CARD_SELECT picks mapped through the frame's host-only raw order
        public int ChestPagesAsked;            // CHEST pages the policy answered with OPEN_CHEST (asked because potion actions were present)
        public int RoomsMatchedExplained;     // dumps that differ only by an explained oracle-timing difference (see TolerateCombatStartHp)
        public List<Diff> ExplainedDiffs = new List<Diff>();
        public string? FirstChoiceDiff;
        public string? FirstOfferDiff;
        public string HostOutcome = "";
        public string? HostAbort;
        public string? Error;
        public double Ms;
    }

    public static List<string> SelectTraces(Args a)
    {
        string dir = a.Get("dir", DefaultDir);
        List<string> all = Directory.GetDirectories(dir).Where(d => File.Exists(Path.Combine(d, "summary.json"))).OrderBy(d => d, StringComparer.Ordinal).ToList();
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

    public static int Run(HostSession session, Args a)
    {
        Options opt = new Options { CompareSave = !a.Has("no-save"), MaxDiffs = a.GetInt("max-diffs", 12), DumpDir = a.Get("dump-host") };
        if (a.Has("no-known-masks"))
        {
            opt.KnownSaveDiffs.Clear();
        }
        opt.TolerateCombatStartHp = !a.Has("strict-hp-timing") && !a.Has("strict-sampling");
        Stopwatch sw = Stopwatch.StartNew();
        List<TraceResult> results = CompareAll(session, SelectTraces(a), opt);
        bool all = results.Count > 0 && results.All(r => r.StoppedBy == "trace_end");
        HostJson.Emit(new
        {
            command = "compare-run-traces",
            // the configuration the traces were compared on: the production set and nothing else (ruling D2 / B4)
            vocab_hash = session.Registry.VocabHash,
            production_patches = session.ProductionPatches.Count,
            production_patch_hash = session.ProductionPatchHash,
            tool_patches = session.ToolPatches.Select(p => p.Id).ToList(),
            known_save_diffs_masked = opt.KnownSaveDiffs,
            tolerate_combat_start_hp_timing = opt.TolerateCombatStartHp,   // (the name is kept for the gate scripts) = the combat-entry sampling rule
            combat_entry_sampling_rule = opt.TolerateCombatStartHp,
            traces = results.Count,
            start_compared = results.Count(r => r.StartCompared),
            start_matches = results.Count(r => r.StartMatches),
            start_structure_matches = results.Count(r => r.StartStructureMatches),
            start_save_matches = results.Count(r => r.StartSaveMatches),
            matched_to_trace_end = results.Count(r => r.StoppedBy == "trace_end"),
            dumps_in_traces = results.Sum(r => r.TraceDumps),
            dumps_matched = results.Sum(r => r.RoomsMatched),
            dumps_matched_explained = results.Sum(r => r.RoomsMatchedExplained),
            card_selects_mapped = results.Sum(r => r.CardSelectsMapped),
            chest_pages_asked = results.Sum(r => r.ChestPagesAsked),
            structure_matched = results.Sum(r => r.StructureMatched),
            choices_in_traces = results.Sum(r => r.TraceChoices),
            choices_matched = results.Sum(r => r.ChoicesMatched),
            card_reward_offers_in_traces = results.Sum(r => r.TraceCardRewardOffers),
            card_reward_offers_matched = results.Sum(r => r.CardRewardOffersMatched),
            stopped_by = results.GroupBy(r => StopKind(r.StoppedBy)).OrderByDescending(g => g.Count()).ToDictionary(g => g.Key, g => g.Count()),
            first_diff_paths = results.Where(r => r.FirstDiffs.Count > 0).GroupBy(r => r.FirstDiffs[0].Kind + ":" + NormalizePath(r.FirstDiffs[0].Path)).OrderByDescending(g => g.Count())
                .ToDictionary(g => g.Key, g => g.Select(r => r.Trace).ToList()),
            seconds = Math.Round(sw.Elapsed.TotalSeconds, 2),
            results,
        }, a.Get("out"));
        return all ? 0 : 5;
    }

    private static string StopKind(string s)
    {
        int c = s.IndexOf(':');
        return c > 0 ? s.Substring(0, c) : s;
    }

    private static string NormalizePath(string p)
    {
        // collapse list indices so that the same field groups together
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        bool inIndex = false;
        foreach (char ch in p)
        {
            if (ch == '[')
            {
                inIndex = true;
                sb.Append("[]");
                continue;
            }
            if (ch == ']')
            {
                inIndex = false;
                continue;
            }
            if (!inIndex)
            {
                sb.Append(ch);
            }
        }
        return sb.ToString();
    }

    /// <param name="doctor">Self-tests only: changes the loaded trace in memory (a negative control); never set by a command.</param>
    public static List<TraceResult> CompareAll(HostSession session, List<string> traceDirs, Options opt, Action<Trace>? doctor = null)
    {
        if (opt.TolerateCombatStartHp)
        {
            V2.V2Sampling.Register();   // the read-only event subscription of the sampling rule (ledger: replay-combat-start-samples)
        }
        List<TraceResult> results = new List<TraceResult>();
        foreach (string dir in traceDirs)
        {
            Stopwatch sw = Stopwatch.StartNew();
            TraceResult r = new TraceResult { Trace = Path.GetFileName(dir) };
            try
            {
                Trace t = Trace.Load(dir);
                doctor?.Invoke(t);
                r.Character = t.Character;
                r.Ascension = t.Ascension;
                r.Seed = t.Seed;
                r.TraceEnd = t.End;
                r.TraceDumps = t.Dumps.Count;
                r.TraceChoices = t.Choices.Count;
                r.TraceCardRewardOffers = t.Offers.Count;
                TracePolicy policy = new TracePolicy(t, r, opt);
                EpisodeResult er = session.RunEpisode(new EpisodeSpec { Seed = t.Seed, Characters = new[] { t.Character }, Ascension = t.Ascension }, policy,
                    new EpisodeOptions(), policy);
                r.HostOutcome = er.Outcome.ToString();
                r.HostFloorReached = er.TotalFloor;
                if (er.Abort != null)
                {
                    r.HostAbort = er.Abort.Signature + " " + er.Abort.Message;
                }
                if (r.StoppedBy.Length == 0)
                {
                    r.StoppedBy = er.Outcome == EpisodeOutcome.Aborted ? "host_abort" : "host_end:" + er.Outcome;
                }
            }
            catch (Exception e)
            {
                r.Error = e.GetBaseException().GetType().Name + ": " + e.GetBaseException().Message;
                r.StoppedBy = "error";
            }
            r.Ms = Math.Round(sw.Elapsed.TotalMilliseconds, 1);
            results.Add(r);
        }
        return results;
    }

    // =====================================================================================================
    //  The trace on disk
    // =====================================================================================================

    public sealed class Trace
    {
        public string Dir = "";
        public string Character = "";
        public int Ascension;
        public string Seed = "";
        public string End = "";
        public List<(string Kind, JsonNode Dump, JsonNode? Save)> Dumps = new List<(string, JsonNode, JsonNode?)>();
        public List<string> Choices = new List<string>();
        public List<string> Offers = new List<string>();

        public static Trace Load(string dir)
        {
            JsonNode summary = JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "summary.json")))!;
            Trace t = new Trace
            {
                Dir = dir,
                Character = summary["character"]!.GetValue<string>(),
                Ascension = summary["ascension"]!.GetValue<int>(),
                Seed = summary["seed"]!.GetValue<string>(),
                End = summary["end"]?.GetValue<string>() ?? "",
            };
            foreach (JsonNode? c in summary["choices"]!.AsArray())
            {
                t.Choices.Add(c!.GetValue<string>());
            }
            foreach (JsonNode? c in summary["card_reward_offers"]!.AsArray())
            {
                t.Offers.Add(c!.GetValue<string>());
            }
            foreach (string f in Directory.GetFiles(dir, "*.json").Where(f => !f.EndsWith(".save.json", StringComparison.Ordinal) && !f.EndsWith(".combat.json", StringComparison.Ordinal)
                         && !f.EndsWith("summary.json", StringComparison.Ordinal)).OrderBy(f => f, StringComparer.Ordinal))
            {
                JsonNode d = JsonNode.Parse(File.ReadAllText(f))!;
                string save = Path.Combine(dir, Path.GetFileNameWithoutExtension(f) + ".save.json");
                t.Dumps.Add((d["kind"]!.GetValue<string>(), d, File.Exists(save) ? JsonNode.Parse(File.ReadAllText(save)) : null));
            }
            return t;
        }
    }

    // =====================================================================================================
    //  Policy + observer
    // =====================================================================================================

    private sealed class TracePolicy : IPolicy, IEpisodeObserver
    {
        private readonly Trace _t;
        private readonly TraceResult _r;
        private readonly Options _opt;
        private int _nextDump;
        private bool _stop;
        private string _stopWhy = "";
        private int _choiceIndex;
        private int _offerIndex;
        private int _playsThisTurn;
        private int _turn = -1;
        private readonly List<string> _hostOffers = new List<string>();
        private int _lastMapActDumped = -1;

        // the host's own states of a combat room's turn-1 start sequence (see Options.TolerateCombatStartHp)
        private readonly List<(JsonObject Dump, string Save)> _samples = new List<(JsonObject, string)>();
        private AbstractRoom? _sampleRoom;
        private Action<CombatState>? _onCombatBegan;
        private Action? _onHistoryChanged;
        private const int MaxSamples = 400;

        public TracePolicy(Trace t, TraceResult r, Options opt)
        {
            _t = t;
            _r = r;
            _opt = opt;
        }

        private void Stop(string why, string? at = null)
        {
            if (_stop)
            {
                return;
            }
            _stop = true;
            _stopWhy = why;
            _r.StoppedBy = why;
            _r.StoppedAt = at;
        }

        // ---------------------------------------------------------------- choices ---

        public int Decide(Decision d)
        {
            if (_stop)
            {
                throw new PolicyStopException(_stopWhy);
            }
            RunContext ctx = d.Episode;
            int floor = ctx.Run.TotalFloor;
            switch (d.Kind)
            {
                case DecisionKind.Event:
                    return Event(d, ctx, floor);
                case DecisionKind.Map:
                {
                    int best = -1;
                    for (int i = 0; i < d.Actions.Count; i++)
                    {
                        if (d.Actions[i].Verb == Verb.Travel && (best < 0 || d.Actions[i].Source.Index < d.Actions[best].Source.Index))
                        {
                            best = i;
                        }
                    }
                    if (best < 0)
                    {
                        return Unmappable(d, "map frame without a travel action");
                    }
                    return best;
                }
                case DecisionKind.Combat:
                    return Combat(d, ctx);
                case DecisionKind.CardSelect:
                    return CardSelect(d, ctx);
                case DecisionKind.Reward:
                {
                    int take = First(d, Verb.ChooseOption);
                    return take >= 0 ? take : Require(d, Verb.Proceed, "reward frame without proceed");
                }
                case DecisionKind.CardReward:
                    return Require(d, Verb.PickOfferCard, "card reward without a card");
                case DecisionKind.Rest:
                {
                    int opt = First(d, Verb.ChooseOption);
                    return opt >= 0 ? opt : Require(d, Verb.Proceed, "rest frame without option or proceed");
                }
                case DecisionKind.Shop:
                    return Require(d, Verb.Proceed, "shop frame without proceed");
                case DecisionKind.Treasure:
                {
                    // RunTracer.StepRoom: click the chest, take the (first) relic, Proceed. The CHEST page reaches the policy
                    // only when the rooms potion extension added actions to it (otherwise OPEN_CHEST is auto-advanced).
                    int open = First(d, Verb.OpenChest);
                    if (open >= 0)
                    {
                        _r.ChestPagesAsked++;
                        return open;
                    }
                    int relic = First(d, Verb.ChooseOption);
                    return relic >= 0 ? relic : Require(d, Verb.Proceed, "treasure frame without chest, relic or proceed");
                }
                default:
                    return Unmappable(d, "decision kind " + d.Kind + " has no counterpart in the trace policy");
            }
        }

        private int Event(Decision d, RunContext ctx, int floor)
        {
            EventModel? ev = (d.Context as EventContext)?.Event;
            string id = ev?.Id.Entry ?? "?";
            // first unlocked option in CurrentOptions order whose WillKillPlayer is false; if all would kill, the first unlocked
            int pick = -1, firstAny = -1;
            for (int i = 0; i < d.Actions.Count; i++)
            {
                LegalAction a = d.Actions[i];
                if (a.Verb != Verb.ChooseOption || a.Source.Kind != RefKind.EventOption)
                {
                    continue;
                }
                if (firstAny < 0 || a.Source.Index < d.Actions[firstAny].Source.Index)
                {
                    firstAny = i;
                }
                EventOption o = a.Source.EventOption;
                bool kills = o.WillKillPlayer != null && o.WillKillPlayer(ctx.LocalPlayer);
                if (!kills && (pick < 0 || a.Source.Index < d.Actions[pick].Source.Index))
                {
                    pick = i;
                }
            }
            if (pick < 0)
            {
                pick = firstAny;
            }
            if (pick >= 0)
            {
                return pick;
            }
            int proceed = First(d, Verb.Proceed);
            return proceed >= 0 ? proceed : Unmappable(d, "event frame without an option or proceed");
        }

        private int Combat(Decision d, RunContext ctx)
        {
            Player me = ctx.LocalPlayer;
            int turn = me.PlayerCombatState?.TurnNumber ?? 0;
            if (turn != _turn)
            {
                _turn = turn;
                _playsThisTurn = 0;
            }
            int end = First(d, Verb.EndTurn);
            if (_playsThisTurn >= _opt.MaxPlaysPerTurn)
            {
                return end >= 0 ? end : Unmappable(d, "play cap reached but no end turn");
            }
            // first playable card in hand order, first legal target in CombatState.Enemies order (hittable)
            CombatState? cs = CombatManager.Instance.DebugOnlyGetState();
            int best = -1;
            for (int i = 0; i < d.Actions.Count; i++)
            {
                LegalAction a = d.Actions[i];
                if (a.Verb != Verb.PlayCard)
                {
                    continue;
                }
                if (!a.Target.IsNone && cs != null && a.Target.Creature is Creature c && c.Side != me.Creature.Side && !cs.HittableEnemies.Contains(c))
                {
                    continue;
                }
                if (best < 0)
                {
                    best = i;
                    continue;
                }
                LegalAction b = d.Actions[best];
                if (a.Source.Index < b.Source.Index || (a.Source.Index == b.Source.Index && EnemyOrder(cs, a.Target) < EnemyOrder(cs, b.Target)))
                {
                    best = i;
                }
            }
            return best >= 0 ? best : end >= 0 ? end : Unmappable(d, "combat frame without play or end turn");
        }

        private static int EnemyOrder(CombatState? cs, GameRef target)
        {
            if (target.IsNone || cs == null || target.Creature is not Creature c)
            {
                return -1;
            }
            for (int i = 0; i < cs.Enemies.Count; i++)
            {
                if (ReferenceEquals(cs.Enemies[i], c))
                {
                    return i;
                }
            }
            return 1000 + target.Index;
        }

        private int CardSelect(Decision d, RunContext ctx)
        {
            if (d.Context is not CardSelectContext c)
            {
                return Unmappable(d, "card selection without context");
            }
            // FirstChoice (Screens.cs AnswerHand / AnswerGrid / AnswerChooseACard): hand selections take the first k = min
            // cards in hand order; grids k = max(min, min(1, max)) cards in the order the grid displays them.
            bool hand = c.InCombat && c.Options.Count > 0 && c.Options.All(o => o.Pile?.Type == PileType.Hand);
            int k = hand ? c.MinSelect : Math.Max(c.MinSelect, Math.Min(1, c.MaxSelect));
            if (c.Picked.Count < k)
            {
                // "UI order": the order the game handed the candidates to the screen (pile-sourced prompts: the host offers
                // canonical classes instead, so the first displayed card is looked up through the frame's host-only view);
                // a simple grid with CardSelectorPrefs.Comparison displays the sorted list, which is the option order.
                int ordinal = -1;
                if (!c.SortedByComparison && d.Source is ICardSelectHostView view)
                {
                    foreach (CardModel raw in view.RawOptions)
                    {
                        bool picked = false;
                        foreach (CardModel done in c.Picked)
                        {
                            picked |= ReferenceEquals(done, raw);
                        }
                        if (!picked)
                        {
                            ordinal = view.OrdinalOf(raw);
                            break;
                        }
                    }
                    if (ordinal < 0)
                    {
                        return Unmappable(d, "card selection: the first displayed card is not among the options");
                    }
                    _r.CardSelectsMapped++;
                }
                int best = -1;
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    if (d.Actions[i].Verb != Verb.SelectCard)
                    {
                        continue;
                    }
                    if (ordinal >= 0 ? d.Actions[i].Source.Index == ordinal : best < 0 || d.Actions[i].Source.Index < d.Actions[best].Source.Index)
                    {
                        best = i;
                    }
                }
                if (best >= 0)
                {
                    return best;
                }
                if (ordinal >= 0)
                {
                    return Unmappable(d, "card selection: no SELECT_CARD action for the first displayed card");
                }
            }
            int confirm = First(d, Verb.ConfirmSelection);
            if (confirm >= 0)
            {
                return confirm;
            }
            int skip = First(d, Verb.Skip);
            if (skip >= 0)
            {
                return skip;
            }
            int sel = First(d, Verb.SelectCard);
            return sel >= 0 ? sel : Unmappable(d, "card selection without select / confirm / skip");
        }

        private static int First(Decision d, Verb v)
        {
            int best = -1;
            for (int i = 0; i < d.Actions.Count; i++)
            {
                if (d.Actions[i].Verb == v && (best < 0 || d.Actions[i].Source.Index < d.Actions[best].Source.Index))
                {
                    best = i;
                }
            }
            return best;
        }

        private int Require(Decision d, Verb v, string why)
        {
            int i = First(d, v);
            return i >= 0 ? i : Unmappable(d, why);
        }

        private int Unmappable(Decision d, string why)
        {
            Stop("unmappable:" + why, $"decision {d.Sequence} kind {d.Kind} floor {d.Episode.Run.TotalFloor}");
            throw new PolicyStopException("unmappable: " + why);
        }

        /// <summary>The trace's choice log entry for the action taken (RunTracer: events, travel, rest, shop leave, treasure open).</summary>
        private void LogChoice(RunContext ctx, Decision d, int chosen)
        {
            LegalAction a = d.Actions[chosen];
            int floor = ctx.Run.TotalFloor;
            switch (d.Kind)
            {
                case DecisionKind.Event:
                {
                    string id = (d.Context as EventContext)?.Event?.Id.Entry ?? "?";
                    if (a.Verb == Verb.ChooseOption && a.Source.Kind == RefKind.EventOption)
                    {
                        EventOption o = a.Source.EventOption;
                        Choice($"f{floor} event {id} option {o.TextKey} proceed={(o.IsProceed ? "True" : "False")}");
                    }
                    else if (a.Verb == Verb.Proceed)
                    {
                        // the real UI shows a finished event's PROCEED as an option button (TextKey PROCEED, IsProceed)
                        Choice($"f{floor} event {id} option PROCEED proceed=True");
                    }
                    break;
                }
                case DecisionKind.Map:
                    if (a.Verb == Verb.Travel)
                    {
                        MapPoint p = a.Source.MapPoint;
                        Choice($"f{floor} travel col{p.coord.col} row{p.coord.row} {p.PointType}");
                    }
                    break;
                case DecisionKind.Rest:
                    if (a.Verb == Verb.ChooseOption)
                    {
                        Choice($"f{floor} rest {a.Source.RestOption.OptionId}");
                    }
                    break;
                case DecisionKind.Shop:
                    if (a.Verb == Verb.Proceed)
                    {
                        Choice($"f{floor} shop leave");
                    }
                    break;
                case DecisionKind.Treasure:
                    if (a.Verb == Verb.OpenChest)
                    {
                        // the trace logs "treasure open" when it clicks the chest: OPEN_CHEST, asked or auto-advanced
                        Choice($"f{floor} treasure open");
                    }
                    break;
                case DecisionKind.CardReward:
                    if (d.Context is CardRewardContext crc)
                    {
                        Offer(string.Join(",", crc.Options.Select(o => o.Card.Id.Entry + (o.Card.IsUpgraded ? "+" : ""))));
                    }
                    break;
                case DecisionKind.Combat:
                    if (a.Verb == Verb.PlayCard)
                    {
                        _playsThisTurn++;
                    }
                    break;
            }
        }

        private void Choice(string s)
        {
            if (_choiceIndex >= _t.Choices.Count)
            {
                Stop("trace_end", "host made more choices than the trace (" + s + ")");
                return;
            }
            string want = _t.Choices[_choiceIndex++];
            _r.ChoicesCompared++;
            if (want == s)
            {
                _r.ChoicesMatched++;
                return;
            }
            _r.FirstChoiceDiff ??= $"choice {_choiceIndex - 1}: trace '{want}' host '{s}'";
            Stop("difference:choice", _r.FirstChoiceDiff);
        }

        private void Offer(string s)
        {
            _hostOffers.Add(s);
            if (_offerIndex >= _t.Offers.Count)
            {
                return;
            }
            string want = _t.Offers[_offerIndex++];
            _r.CardRewardOffersCompared++;
            if (want == s)
            {
                _r.CardRewardOffersMatched++;
                return;
            }
            _r.FirstOfferDiff ??= $"card reward {_offerIndex - 1}: trace '{want}' host '{s}'";
            Stop("difference:card_reward_offer", _r.FirstOfferDiff);
        }

        // ---------------------------------------------------------------- dumps ---

        public void OnEpisodeStart(RunContext ctx)
        {
            if (!_opt.TolerateCombatStartHp || _onCombatBegan != null)
            {
                return;
            }
            // read-only: the tracer's dump and the game's own save at the moments the real client's tracer may be dumping
            _onCombatBegan = _ =>
            {
                if (_stop)
                {
                    return;
                }
                _samples.Clear();
                _sampleRoom = ctx.Run.CurrentRoom;
                _samples.Add((Dump(ctx, "room", peek: true), SaveJson()));
            };
            _onHistoryChanged = () =>
            {
                if (_stop || _sampleRoom == null || !ReferenceEquals(_sampleRoom, ctx.Run.CurrentRoom) || _samples.Count >= MaxSamples)
                {
                    return;
                }
                _samples.Add((Dump(ctx, "room", peek: true), SaveJson()));
            };
            ctx.CombatManager.CombatBegan += _onCombatBegan;
            ctx.CombatManager.History.Changed += _onHistoryChanged;
        }

        private void Unsubscribe(RunContext ctx)
        {
            if (_onCombatBegan != null)
            {
                ctx.CombatManager.CombatBegan -= _onCombatBegan;
                _onCombatBegan = null;
            }
            if (_onHistoryChanged != null)
            {
                ctx.CombatManager.History.Changed -= _onHistoryChanged;
                _onHistoryChanged = null;
            }
            _samples.Clear();
            _sampleRoom = null;
        }

        private static string SaveJson() => JsonSerializer.Serialize(RunManager.Instance.ToSave(null), JsonSerializationUtility.GetTypeInfo<SerializableRun>());

        /// <summary>
        /// Run start and every room entry, settled, before anything in the room is answered (replay request R3): where the
        /// trace dumps ("start", then "room" at every RunManager.RoomEntered once the room transition finished).
        /// </summary>
        public void OnRoomBoundary(RunContext ctx)
        {
            if (_stop)
            {
                return;
            }
            CompareDump(ctx, _nextDump == 0 ? "start" : "room");
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
        {
            if (_stop)
            {
                return;
            }
            LogChoice(ctx, decision, chosen);
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
            Unsubscribe(ctx);
            if (_stop)
            {
                return;
            }
            if (_nextDump < _t.Dumps.Count && _t.Dumps[_nextDump].Kind.StartsWith("end", StringComparison.Ordinal))
            {
                // (the end-of-run bookkeeping of a death - RunManager.OnEnded - is the production fidelity pack's:
                // fidelity-natural-death-on-ended; this tool calls nothing)
                CompareDump(ctx, _t.Dumps[_nextDump].Kind);
            }
            if (!_stop)
            {
                Stop(_nextDump >= _t.Dumps.Count ? "trace_end" : "host_end_early", $"host terminal {provisional.Outcome} at dump {_nextDump}/{_t.Dumps.Count}");
            }
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }

        private void CompareDump(RunContext ctx, string kind)
        {
            if (_nextDump >= _t.Dumps.Count)
            {
                Stop("trace_end", "host entered more rooms than the trace has dumps");
                return;
            }
            (string traceKind, JsonNode traceDump, JsonNode? traceSave) = _t.Dumps[_nextDump];
            if (traceKind.StartsWith("end", StringComparison.Ordinal) && !kind.StartsWith("end", StringComparison.Ordinal))
            {
                Stop("difference:host_continues", $"trace ended ({traceKind}) at dump {_nextDump} but the host entered another room");
                return;
            }
            int index = _nextDump++;
            _r.DumpsCompared++;
            // the samples of this combat room's turn-1 start (taken since CombatBegan); sampling ends here, at the first decision point
            List<(JsonObject Dump, string Save)> samples = new List<(JsonObject, string)>(_samples);
            bool sampled = _opt.TolerateCombatStartHp && kind == "room" && samples.Count > 0 && samples.Count < MaxSamples && ctx.Run.CurrentRoom is CombatRoom && ReferenceEquals(_sampleRoom, ctx.Run.CurrentRoom);
            _samples.Clear();
            _sampleRoom = null;
            JsonObject host = Dump(ctx, kind);
            List<Diff> diffs = new List<Diff>();
            void Structure(JsonObject hostDump, List<Diff> into)
            {
                foreach (KeyValuePair<string, JsonNode?> kv in traceDump.AsObject())
                {
                    if (kv.Key is "index" or "kind" or "choices_so_far" or "card_reward_offers_so_far")
                    {
                        continue;
                    }
                    CompareNodes(kv.Value, hostDump[kv.Key], "dump." + kv.Key, index, "structure", into);
                }
            }
            Structure(host, diffs);
            bool structureOk = diffs.Count == 0;
            bool saveOk = true;
            string hostSaveJson = SaveJson();
            if (_opt.DumpDir != null)
            {
                string d = Path.Combine(_opt.DumpDir, Path.GetFileName(_t.Dir));
                Directory.CreateDirectory(d);
                File.WriteAllText(Path.Combine(d, $"{index:D3}-{kind}.host.json"), host.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
                File.WriteAllText(Path.Combine(d, $"{index:D3}-{kind}.host.save.json"), hostSaveJson);
            }
            if (_opt.CompareSave && traceSave != null)
            {
                JsonNode? hostSave = JsonNode.Parse(hostSaveJson);
                int before = diffs.Count;
                CompareNodes(traceSave, hostSave, "save", index, "save", diffs);
                saveOk = diffs.Count == before;
            }
            if (index == 0)
            {
                _r.StartCompared = true;
                _r.StartStructureMatches = structureOk;
                _r.StartSaveMatches = saveOk;
                _r.StartMatches = structureOk && saveOk;
            }
            if (structureOk)
            {
                _r.StructureMatched++;
            }
            if (structureOk && saveOk)
            {
                _r.RoomsMatched++;
                return;
            }
            if (sampled)
            {
                // Oracle dump timing (class b): the trace dumps a combat room while the real client is still inside the turn-1
                // start sequence (CombatManager.StartCombatInternal: Cmd.CustomScaledWait(0.5f, 1f) before StartTurn,
                // CombatManager.cs:407-409, and the waits inside the start hooks), e.g. BEFORE BloodVial.AfterPlayerTurnStartLate
                // (+2 HP); the host's first decision point is after it. The tracer's dump and save are one snapshot of one moment:
                // ONE of the host's own sampled states of that sequence must equal the WHOLE dump and save. Anything else is a difference.
                JsonNode? traceSaveNode = _opt.CompareSave ? traceSave : null;
                for (int i = 0; i < samples.Count; i++)
                {
                    List<Diff> early = new List<Diff>();
                    Structure(samples[i].Dump, early);
                    if (traceSaveNode != null)
                    {
                        CompareNodes(traceSaveNode, JsonNode.Parse(samples[i].Save), "save", index, "save", early);
                    }
                    if (early.Count == 0)
                    {
                        _r.RoomsMatchedExplained++;
                        _r.KnownDiffsMasked["combat_entry_sampled_before_turn_1_start_hooks"] = _r.KnownDiffsMasked.TryGetValue("combat_entry_sampled_before_turn_1_start_hooks", out int n) ? n + 1 : 1;
                        _r.ExplainedDiffs.AddRange(diffs.Take(4));
                        return;
                    }
                }
            }
            _r.FirstDiffs.AddRange(diffs.Take(_opt.MaxDiffs));
            Diff first = diffs[0];
            Stop("difference:" + first.Kind, $"dump {index} ({traceKind}) {first.Path}: trace {first.Trace} host {first.Host}");
        }

        /// <summary>
        /// RunTracer.Dump, field by field, from the host's run state. <paramref name="peek"/>: a sample of the turn-1 start
        /// sequence - the act map is always included and the "map written for this act" state is not touched.
        /// </summary>
        private JsonObject Dump(RunContext ctx, string kind, bool peek = false)
        {
            RunState rs = ctx.Run;
            Player me = ctx.LocalPlayer;
            static JsonArray Arr(IEnumerable<string?> xs) => new JsonArray(xs.Select(x => (JsonNode?)JsonValue.Create(x)).ToArray());
            JsonObject o = new JsonObject
            {
                ["total_floor"] = rs.TotalFloor,
                ["act_index"] = rs.CurrentActIndex,
                ["act"] = rs.Act.Id.Entry,
                ["act_floor"] = rs.ActFloor,
                ["acts"] = Arr(rs.Acts.Select(a => a.Id.Entry)),
                ["bosses"] = Arr(rs.Acts.Select(a => $"{a.BossEncounter?.Id.Entry}|{a.SecondBossEncounter?.Id.Entry}")),
                ["room_type"] = rs.CurrentRoom?.RoomType.ToString(),
                ["room_model"] = rs.CurrentRoom?.ModelId?.Entry,
                ["coord"] = rs.CurrentMapCoord is MapCoord c ? $"{c.col},{c.row}" : null,
                ["hp"] = $"{me.Creature.CurrentHp}/{me.Creature.MaxHp}",
                ["gold"] = me.Gold,
                ["deck"] = Arr(me.Deck.Cards.Select(x => x.Id.Entry + (x.IsUpgraded ? "+" : ""))),
                ["relics"] = Arr(me.Relics.Select(x => x.Id.Entry)),
                ["potions"] = Arr(me.PotionSlots.Select(x => x?.Id.Entry ?? "-")),
            };
            if ((peek || rs.CurrentActIndex != _lastMapActDumped) && rs.Map != null)
            {
                if (!peek)
                {
                    _lastMapActDumped = rs.CurrentActIndex;
                }
                o["map"] = new JsonArray(rs.Map.GetAllMapPoints().OrderBy(p => p.coord.row).ThenBy(p => p.coord.col).Select(p => (JsonNode)new JsonObject
                {
                    ["col"] = p.coord.col,
                    ["row"] = p.coord.row,
                    ["type"] = p.PointType.ToString(),
                    ["children"] = Arr(p.Children.OrderBy(ch => ch.coord.col).Select(ch => $"{ch.coord.col},{ch.coord.row}")),
                }).ToArray());
                o["map_boss"] = $"{rs.Map.BossMapPoint?.coord.col},{rs.Map.BossMapPoint?.coord.row}";
                o["map_start"] = $"{rs.Map.StartingMapPoint?.coord.col},{rs.Map.StartingMapPoint?.coord.row}";
            }
            if (rs.CurrentRoom is EventRoom er)
            {
                o["event"] = er.CanonicalEvent?.Id.Entry;
                // the trace lists the option buttons of NEventRoom (all CurrentOptions of the local event)
                EventModel? local = RunManager.Instance.EventSynchronizer.GetLocalEvent();
                if (local != null)
                {
                    o["event_options_visible"] = Arr(local.CurrentOptions.Select(x => x.TextKey));
                }
            }
            if (rs.CurrentRoom is MerchantRoom)
            {
                MerchantInventory inv = RunManager.Instance.MerchantSynchronizer.GetLocalInventory();
                o["shop"] = new JsonArray(inv.AllEntries.Select(e => (JsonNode)new JsonObject
                {
                    ["type"] = e.GetType().Name,
                    ["cost"] = e.IsStocked ? e.Cost : -1,
                    ["item"] = e switch
                    {
                        MerchantCardEntry ce => ce.CreationResult?.Card?.Id.Entry + ((ce.CreationResult?.Card?.IsUpgraded ?? false) ? "+" : ""),
                        MerchantRelicEntry re => re.Model?.Id.Entry,
                        MerchantPotionEntry pe => pe.Model?.Id.Entry,
                        _ => null,
                    },
                }).ToArray());
            }
            return o;
        }

        private static readonly HashSet<string> IgnoredSaveFields = new HashSet<string>(StringComparer.Ordinal)
        {
            "save_time", "start_time", "run_time", "win_time", "map_drawings",
        };

        private void CompareNodes(JsonNode? a, JsonNode? b, string path, int dump, string kind, List<Diff> diffs)
        {
            if (diffs.Count >= _opt.MaxDiffs * 4)
            {
                return;
            }
            if (a is JsonObject oa && b is JsonObject ob)
            {
                foreach (KeyValuePair<string, JsonNode?> kv in oa)
                {
                    if (kind == "save" && IgnoredSaveFields.Contains(kv.Key))
                    {
                        continue;
                    }
                    if (kind == "save" && _opt.KnownSaveDiffs.Contains(kv.Key))
                    {
                        if ((kv.Value?.ToJsonString() ?? "null") != (ob[kv.Key]?.ToJsonString() ?? "null"))
                        {
                            _r.KnownDiffsMasked[kv.Key] = _r.KnownDiffsMasked.TryGetValue(kv.Key, out int n) ? n + 1 : 1;
                        }
                        continue;
                    }
                    CompareNodes(kv.Value, ob[kv.Key], path + "." + kv.Key, dump, kind, diffs);
                }
                foreach (KeyValuePair<string, JsonNode?> kv in ob)
                {
                    if (!oa.ContainsKey(kv.Key) && !(kind == "save" && (IgnoredSaveFields.Contains(kv.Key) || _opt.KnownSaveDiffs.Contains(kv.Key))))
                    {
                        diffs.Add(new Diff { Dump = dump, Kind = kind, Path = path + "." + kv.Key, Trace = "(absent)", Host = Short(kv.Value) });
                    }
                }
                return;
            }
            if (a is JsonArray la && b is JsonArray lb)
            {
                int n = Math.Max(la.Count, lb.Count);
                for (int i = 0; i < n; i++)
                {
                    if (i >= la.Count || i >= lb.Count)
                    {
                        diffs.Add(new Diff { Dump = dump, Kind = kind, Path = path + "[" + i + "]", Trace = i < la.Count ? Short(la[i]) : "(absent)", Host = i < lb.Count ? Short(lb[i]) : "(absent)" });
                        return;
                    }
                    CompareNodes(la[i], lb[i], path + "[" + i + "]", dump, kind, diffs);
                }
                return;
            }
            string sa = a?.ToJsonString() ?? "null";
            string sb = b?.ToJsonString() ?? "null";
            if (sa != sb)
            {
                diffs.Add(new Diff { Dump = dump, Kind = kind, Path = path, Trace = Short(a), Host = Short(b) });
            }
        }

        private static string Short(JsonNode? n)
        {
            string s = n?.ToJsonString() ?? "null";
            return s.Length > 200 ? s.Substring(0, 200) + "..." : s;
        }
    }
}
