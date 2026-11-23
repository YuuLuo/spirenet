using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace SpireHost.Driver.Tools;

/// <summary>
/// --selftest-mp, the cases of milestone M2 ("a whole run with 2-4 seats"; notes\h5-plan.md section 2, M2): one
/// scenario per room flow that a remote seat goes through, each on a real multi-seat run start, played over the
/// driver's ordinary decisions. Every case states what it checks in its own summary.
///
///   rest      MEND by the local seat (fidelity-mend-local-targeting) and by a remote seat, the heal lands on the
///             target; a remote seat that leaves with options left (MiniatureTent) skips at the room exit
///   treasure  every seat opens its own chest and votes; no SKIP with several seats; two seats contest one relic;
///             every seat ends with exactly one relic of the chest
///   shop      per-seat inventories, remote purchases and a remote card removal (remote prompt); LordsParasol of a
///             remote owner (no yield) and of the local owner (yield)
///   crystal   the Crystal Sphere minigame of remote seats on the host's own minigame object, then the client's
///             CrystalSphereRewardsMessage and the reward screen
///   bundle    the bundle prompt (Scroll Boxes) of the local seat and of a remote seat
///   shared    every shared event that multiplayer allows, page votes per seat and as team votes, event fights
///   isolation a purchase of one seat leaves every other seat's private state and player RNG streams untouched
///   runs      whole runs with forced HP (act changes through the act vote of every seat, double boss, Architect)
/// </summary>
internal static class MultiSeatRoomTests
{
    public static readonly string[] Cases = { "rest", "treasure", "shop", "crystal", "bundle", "shared", "isolation", "runs" };

    /// <summary>A policy with a preference: <see cref="Choose"/> returns an action index or -1 (then uniformly random).</summary>
    private sealed class GuidedPolicy : IPolicy
    {
        private readonly RandomLegalPolicy _random;

        public Func<Decision, int>? Choose;

        public GuidedPolicy(ulong seed)
        {
            _random = new RandomLegalPolicy(seed);
        }

        public int Decide(Decision decision)
        {
            int i = Choose?.Invoke(decision) ?? -1;
            return i >= 0 && i < decision.Actions.Count ? i : _random.Decide(decision);
        }
    }

    /// <summary>Records what the test asserts about decisions; checks queued in <see cref="After"/> run at the next decision (the state has settled).</summary>
    private sealed class Audit : IEpisodeObserver
    {
        public readonly List<string> Problems = new List<string>();
        public readonly List<Action<RunContext>> After = new List<Action<RunContext>>();
        public Action<RunContext, Decision, int, bool>? Decision;
        public Action<RunContext>? Terminal;
        public readonly long[] ByKind = new long[16];
        public readonly long[,] ByKindSeat = new long[16, 5];

        public void OnEpisodeStart(RunContext ctx)
        {
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
        {
            Flush(ctx);
            ByKind[(int)decision.Kind]++;
            ByKindSeat[(int)decision.Kind, decision.ActingSeat < 0 ? 4 : decision.ActingSeat]++;
            if (decision.Kind != DecisionKind.Combat)
            {
                if (decision.ActingSeat < 0 || decision.ActingSeat >= ctx.SeatCount)
                {
                    Problems.Add(decision.Kind + " decision without one acting seat (" + decision.ActingSeat + ")");
                }
                foreach (LegalAction a in decision.Actions)
                {
                    if (a.Seat != decision.ActingSeat)
                    {
                        Problems.Add(decision.Kind + " decision of seat " + decision.ActingSeat + " carries an action of seat " + a.Seat);
                        break;
                    }
                }
            }
            Decision?.Invoke(ctx, decision, chosen, auto);
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
            Flush(ctx);
            Terminal?.Invoke(ctx);
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }

        private void Flush(RunContext ctx)
        {
            if (After.Count == 0)
            {
                return;
            }
            Action<RunContext>[] run = After.ToArray();
            After.Clear();
            foreach (Action<RunContext> a in run)
            {
                a(ctx);
            }
        }
    }

    private static bool AllMapsOpen(RunContext ctx)
    {
        for (int s = 0; s < ctx.SeatCount; s++)
        {
            if (!ctx.IsMapOpen(s))
            {
                return false;
            }
        }
        return true;
    }

    private static long Counter(EpisodeResult r, string key) => r.ModuleCounters.TryGetValue(key, out long v) ? v : 0;

    private static string Describe(EpisodeResult r)
    {
        return r.Outcome == EpisodeOutcome.Aborted ? "Aborted(" + r.Abort!.Signature + " | " + r.Abort.Message + ")"
            : r.Outcome == EpisodeOutcome.Truncated ? "Truncated(" + r.TruncatedBy + ")"
            : r.Outcome.ToString();
    }

    private static void Clean(EpisodeResult r, List<string> problems, bool expectScenarioStop = true)
    {
        if (r.Outcome == EpisodeOutcome.Aborted || (expectScenarioStop && !(r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy == "scenario")))
        {
            problems.Add(Describe(r));
        }
        foreach (KeyValuePair<string, long> kv in r.ModuleCounters)
        {
            if (kv.Value != 0 && kv.Key.Contains("violation", StringComparison.Ordinal))
            {
                problems.Add(kv.Key + " = " + kv.Value);
            }
        }
        string? strict = r.Strict.FirstViolation();
        if (strict != null)
        {
            problems.Add("strict: " + strict);
        }
        if (r.TeardownFaults != 0)
        {
            problems.Add("teardown: " + r.TeardownFault);
        }
    }

    private static string[] Lineup(int seats, int index)
    {
        string[] all = DriverCommands.AllCharacters;
        string[] r = new string[seats];
        for (int k = 0; k < seats; k++)
        {
            r[k] = all[(index + k) % all.Length];
        }
        return r;
    }

    private static int IndexOf(Decision d, Func<LegalAction, bool> match)
    {
        for (int i = 0; i < d.Actions.Count; i++)
        {
            if (match(d.Actions[i]))
            {
                return i;
            }
        }
        return -1;
    }

    // =====================================================================================================

    public static void Run(HostSession session, Args a, HashSet<string> cases, int[] seatCounts, bool wire, Dictionary<string, object> results, ref bool all)
    {
        if (cases.Contains("rest"))
        {
            all &= Rest(session, seatCounts, wire, results);
        }
        if (cases.Contains("treasure"))
        {
            all &= Treasure(session, seatCounts, wire, results);
        }
        if (cases.Contains("shop"))
        {
            all &= Shop(session, seatCounts, wire, results);
        }
        if (cases.Contains("crystal"))
        {
            all &= Crystal(session, seatCounts, wire, results);
        }
        if (cases.Contains("bundle"))
        {
            all &= Bundle(session, seatCounts, wire, results);
        }
        if (cases.Contains("shared"))
        {
            all &= Shared(session, seatCounts, wire, results);
        }
        if (cases.Contains("isolation"))
        {
            all &= Isolation(session, seatCounts, wire, results);
        }
        if (cases.Contains("runs"))
        {
            all &= Runs(session, seatCounts, wire, a.GetInt("episodes", 6), results);
        }
        if (cases.Contains("trace"))
        {
            all &= Trace(session, a, results);
        }
    }

    // ------------------------------------------------------------------------------------------------ trace (diagnosis)

    /// <summary>
    /// Not a test: replays one tuple of a fuzz report (--seed, --lineup, --ascension, --policy-seed, --hp, --team-votes,
    /// --wire) with a random legal policy and lists the last --last (default 60) decisions before the episode ended.
    /// </summary>
    private static bool Trace(HostSession session, Args a, Dictionary<string, object> results)
    {
        int keep = a.GetInt("last", 60);
        Queue<string> lines = new Queue<string>();
        Audit audit = new Audit();
        audit.Decision = (ctx, d, chosen, auto) =>
        {
            LegalAction c = d.Actions[chosen];
            string src = c.Source.Kind == RefKind.Card ? c.Source.Card.Id.Entry : c.Source.Kind == RefKind.None ? "-" : c.Source.Kind + "#" + c.Source.Index;
            string extra = "";
            if (d.Context is CardSelectContext sel)
            {
                extra = " select{" + sel.Entry + " " + sel.ContextKind + " src=" + (sel.SourceModel?.Id.Entry ?? "-") + " min=" + sel.MinSelect + " max=" + sel.MaxSelect + " options=" + sel.Options.Count + " inCombat=" + sel.InCombat + " outsidePlay=" + sel.OutsidePlayPhase + "}";
            }
            lines.Enqueue("#" + d.Sequence + " " + d.Kind + "/" + d.SubKind + (auto ? "*" : "") + " seat=" + d.ActingSeat + " n=" + d.Actions.Count + " -> " + c.Verb + "(seat " + c.Seat + ", " + src + (c.Target.IsNone ? "" : " -> #" + c.Target.Index) + (c.Aux != 0 ? ", aux " + c.Aux : "") + ")" + extra
                + " | room=" + (ctx.CurrentRoom?.GetType().Name ?? "-") + " combat=" + ctx.CombatManager.IsInProgress + " stack=" + ctx.PendingCount + " floor=" + ctx.Run.TotalFloor);
            while (lines.Count > keep)
            {
                lines.Dequeue();
            }
        };
        string[] lineup = a.Get("lineup", "IRONCLAD+SILENT").Split('+', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        EpisodeOptions options = DriverCommands.EpisodeOptionsFrom(a);
        options.ComputeSignatures = true;
        EpisodeResult r = session.RunEpisode(new EpisodeSpec { Seed = a.Get("seed"), Characters = lineup, Ascension = a.GetInt("ascension", 0) }, new RandomLegalPolicy((ulong)a.GetLong("policy-seed", 1)), options, audit);
        results["trace"] = new
        {
            ok = r.Outcome != EpisodeOutcome.Aborted,
            seed = r.Seed,
            lineup = string.Join("+", r.Characters),
            outcome = Describe(r),
            diagnostics = r.Abort?.Diagnostics,
            floor = r.TotalFloor,
            policy_decisions = r.Counters.PolicyDecisions,
            trace_hash = r.TraceHash.ToString("x16"),
            last_decisions = lines.ToArray(),
        };
        return r.Outcome != EpisodeOutcome.Aborted;
    }

    // ------------------------------------------------------------------------------------------------ rest

    private static bool Rest(HostSession session, int[] seatCounts, bool wire, Dictionary<string, object> results)
    {
        List<object> rows = new List<object>();
        bool ok = true;
        long localMends = 0, remoteMends = 0, exitSkips = 0;
        foreach (int n in seatCounts)
        {
            foreach (bool teamVotes in new[] { false, true })
            {
                List<string> problems = new List<string>();
                Audit audit = new Audit();
                bool sawRest = false;
                int tent = n - 1;
                int mends = 0;
                Scenario scenario = new Scenario
                {
                    Name = "mp-rest",
                    Setup = async ctx =>
                    {
                        for (int s = 0; s < ctx.SeatCount; s++)
                        {
                            ScenarioTools.SetHp(ctx, Math.Max(1, ctx.Seats[s].Creature.MaxHp / 3), null, s);
                        }
                        await ScenarioTools.GiveRelic(ctx, "MINIATURE_TENT", tent);
                        await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite);
                    },
                    StopWhen = ctx =>
                    {
                        if (ctx.CurrentRoom is RestSiteRoom)
                        {
                            sawRest = true;
                            return false;
                        }
                        return sawRest;
                    },
                };
                GuidedPolicy policy = new GuidedPolicy(11UL + (ulong)n);
                policy.Choose = d =>
                {
                    if (d.Kind != DecisionKind.Rest)
                    {
                        return -1;
                    }
                    RestContext rc = (RestContext)d.Context!;
                    int seat = d.ActingSeat;
                    if (rc.Successes > 0)
                    {
                        return IndexOf(d, x => x.Verb == Verb.Proceed);
                    }
                    if (seat == 0)
                    {
                        return IndexOf(d, x => x.Verb == Verb.ChooseOption && x.Source.RestOption is MendRestSiteOption && x.Aux == 1);
                    }
                    if (seat == 1)
                    {
                        return IndexOf(d, x => x.Verb == Verb.ChooseOption && x.Source.RestOption is MendRestSiteOption && x.Aux == 0);
                    }
                    return IndexOf(d, x => x.Verb == Verb.ChooseOption && x.Source.RestOption is HealRestSiteOption);
                };
                audit.Decision = (ctx, d, chosen, auto) =>
                {
                    if (d.Kind != DecisionKind.Rest)
                    {
                        return;
                    }
                    int seat = d.ActingSeat;
                    RestContext rc = (RestContext)d.Context!;
                    if (!ReferenceEquals(rc.Options, ctx.RunManager.RestSiteSynchronizer.GetOptionsForPlayer(ctx.Seats[seat])))
                    {
                        problems.Add("the REST frame of seat " + seat + " does not show that seat's options");
                    }
                    int mendRows = 0;
                    foreach (LegalAction x in d.Actions)
                    {
                        if (x.Verb == Verb.ChooseOption && x.Source.RestOption is MendRestSiteOption)
                        {
                            mendRows++;
                            if (x.Aux == seat || x.Aux < 0 || x.Aux >= ctx.SeatCount)
                            {
                                problems.Add("MEND of seat " + seat + " offers target seat " + x.Aux);
                            }
                        }
                    }
                    bool hasMend = rc.Options.Any(o => o is MendRestSiteOption && o.IsEnabled);
                    if (hasMend && mendRows != ctx.SeatCount - 1)
                    {
                        problems.Add("MEND of seat " + seat + " has " + mendRows + " target rows, expected " + (ctx.SeatCount - 1));
                    }
                    LegalAction c = d.Actions[chosen];
                    if (c.Verb == Verb.ChooseOption && c.Source.RestOption is MendRestSiteOption)
                    {
                        mends++;
                        Player target = ctx.Seats[c.Aux];
                        int before = target.Creature.CurrentHp;
                        int missing = target.Creature.MaxHp - before;
                        decimal amount = MendRestSiteOption.GetHealAmount(target);
                        int[] others = Enumerable.Range(0, ctx.SeatCount).Where(s => s != c.Aux).Select(s => ctx.Seats[s].Creature.CurrentHp).ToArray();
                        audit.After.Add(ctx2 =>
                        {
                            int gained = target.Creature.CurrentHp - before;
                            if (gained <= 0 || gained > missing || (gained != (int)amount && gained != missing))
                            {
                                problems.Add("MEND of seat " + seat + " on seat " + c.Aux + ": HP +" + gained + ", the target's rest amount is " + amount + " (missing " + missing + ")");
                            }
                            int k = 0;
                            for (int s = 0; s < ctx2.SeatCount; s++)
                            {
                                if (s != c.Aux && ctx2.Seats[s].Creature.CurrentHp != others[k++])
                                {
                                    problems.Add("MEND of seat " + seat + " changed the HP of seat " + s);
                                }
                            }
                        });
                    }
                };
                EpisodeResult r = session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("MPREST:" + n + ":" + teamVotes), Characters = Lineup(n, n), Scenario = scenario }, policy,
                    new EpisodeOptions { WireRoundTrip = wire, TeamVotes = teamVotes, MaxPolicyDecisions = 400 }, audit);
                Clean(r, problems);
                problems.AddRange(audit.Problems);
                long local = Counter(r, "fidelity.mend_local_targetings"), answered = Counter(r, "rooms.rest.mend_targets_answered"), skips = Counter(r, "rooms.remote.rest_exit_skips");
                if (local != 1) problems.Add("fidelity.mend_local_targetings = " + local + ", expected 1 (seat 0's MEND)");
                if (mends != Math.Min(n, 2) || answered != mends) problems.Add(mends + " MEND choices, " + answered + " targets answered");
                if (skips != 1) problems.Add("rooms.remote.rest_exit_skips = " + skips + ", expected 1 (the MiniatureTent seat leaves with options left)");
                if (Counter(r, "rooms.remote.rest_choices") != n - 1) problems.Add("rooms.remote.rest_choices = " + Counter(r, "rooms.remote.rest_choices"));
                long mapAsked = audit.ByKind[(int)DecisionKind.Map], teamCast = Counter(r, "mp.team_votes.map");
                if (teamVotes ? (mapAsked != 1 || teamCast != n - 1) : (mapAsked != n || teamCast != 0)) problems.Add("map votes asked " + mapAsked + ", team votes cast " + teamCast + " (team votes " + teamVotes + ")");
                localMends += local;
                remoteMends += answered - local;
                exitSkips += skips;
                ok &= problems.Count == 0;
                rows.Add(new { seats = n, team_votes = teamVotes, seed = r.Seed, outcome = Describe(r), mend_choices = mends, map_decisions = mapAsked, team_map_votes_cast = teamCast, passed = problems.Count == 0, problems });
            }
        }
        results["rest"] = new
        {
            ok,
            what = "rest site with 2-4 seats: every seat's REST frame shows its own options; MEND has one row per other seat (aux = target seat), the local seat's MEND goes through the waiting branch " +
                   "(fidelity-mend-local-targeting) and heals the target by the target's own rest amount, a remote seat's MEND likewise; the MiniatureTent seat leaves with options left and its client " +
                   "skips at the room exit (rooms-remote-rest-exit-skip); the map vote is asked once with team votes and once per seat without",
            local_mends = localMends,
            remote_mends = remoteMends,
            rest_exit_skips = exitSkips,
            rows,
        };
        return ok;
    }

    // ------------------------------------------------------------------------------------------------ treasure

    private static bool Treasure(HostSession session, int[] seatCounts, bool wire, Dictionary<string, object> results)
    {
        List<object> rows = new List<object>();
        bool ok = true;
        foreach (int n in seatCounts)
        {
            foreach (int ascension in new[] { 0, 10 })
            {
                List<string> problems = new List<string>();
                Audit audit = new Audit();
                int[] relicsBefore = new int[n], goldBefore = new int[n];
                bool[] chest = new bool[n], voted = new bool[n];
                bool captured = false;
                Scenario scenario = new Scenario
                {
                    Name = "mp-treasure",
                    Setup = ctx => ScenarioTools.JumpToRoom(ctx, RoomType.Treasure),
                    StopWhen = AllMapsOpen,
                };
                GuidedPolicy policy = new GuidedPolicy(23UL + (ulong)n);
                policy.Choose = d =>
                {
                    if (d.Kind == DecisionKind.Treasure && d.SubKind == (int)TreasurePhase.Relic && d.ActingSeat <= 1)
                    {
                        return IndexOf(d, x => x.Verb == Verb.ChooseOption && x.Source.Index == 0); // seats 0 and 1 contest relic 0
                    }
                    return -1;
                };
                audit.Decision = (ctx, d, chosen, auto) =>
                {
                    if (!captured)
                    {
                        captured = true;
                        for (int s = 0; s < n; s++)
                        {
                            relicsBefore[s] = ctx.Seats[s].Relics.Count;
                            goldBefore[s] = ctx.Seats[s].Gold;
                        }
                    }
                    if (d.Kind != DecisionKind.Treasure)
                    {
                        return;
                    }
                    int seat = d.ActingSeat;
                    TreasureContext tc = (TreasureContext)d.Context!;
                    LegalAction c = d.Actions[chosen];
                    if (d.Actions.Any(x => x.Verb == Verb.Skip))
                    {
                        problems.Add("SKIP offered on a TREASURE frame of seat " + seat + " (there is no skip with several players)");
                    }
                    if (d.SubKind == (int)TreasurePhase.Chest)
                    {
                        if (tc.Relics.Count != 0) problems.Add("relics shown to seat " + seat + " before it opened its chest");
                        if (chest[seat]) problems.Add("seat " + seat + " is asked to open its chest twice");
                        chest[seat] = c.Verb == Verb.OpenChest;
                    }
                    else if (c.Verb == Verb.ChooseOption)
                    {
                        if (!chest[seat]) problems.Add("seat " + seat + " votes before opening its chest");
                        if (voted[seat]) problems.Add("seat " + seat + " votes twice");
                        voted[seat] = true;
                        if (tc.Relics.Count != n) problems.Add("the chest shows " + tc.Relics.Count + " relics to seat " + seat + ", expected one per player");
                        if (tc.SeatVotes.Count != n) problems.Add("TreasureContext.SeatVotes has " + tc.SeatVotes.Count + " entries");
                        for (int s = 0; s < tc.SeatVotes.Count; s++)
                        {
                            if ((tc.SeatVotes[s] >= 0) != (voted[s] && s != seat)) problems.Add("seat " + seat + " sees the vote of seat " + s + " as " + tc.SeatVotes[s]);
                        }
                    }
                    else if (c.Verb == Verb.Proceed && !tc.Finished)
                    {
                        problems.Add("PROCEED of seat " + seat + " before the relic picking finished");
                    }
                };
                audit.Terminal = ctx =>
                {
                    for (int s = 0; s < n; s++)
                    {
                        if (ctx.Seats[s].Relics.Count != relicsBefore[s] + 1) problems.Add("seat " + s + " has " + (ctx.Seats[s].Relics.Count - relicsBefore[s]) + " new relics, expected exactly 1");
                        if (ctx.Seats[s].Gold <= goldBefore[s]) problems.Add("seat " + s + " got no chest gold");
                        if (!chest[s] || !voted[s]) problems.Add("seat " + s + ": chest opened " + chest[s] + ", voted " + voted[s]);
                    }
                };
                EpisodeResult r = session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("MPTREASURE:" + n + ":" + ascension), Characters = Lineup(n, n + ascension), Ascension = ascension, Scenario = scenario }, policy,
                    new EpisodeOptions { WireRoundTrip = wire, MaxPolicyDecisions = 400 }, audit);
                Clean(r, problems);
                problems.AddRange(audit.Problems);
                if (Counter(r, "rooms.remote.chests_opened") != n - 1) problems.Add("rooms.remote.chests_opened = " + Counter(r, "rooms.remote.chests_opened"));
                if (Counter(r, "rooms.remote.treasure_picks") != n - 1) problems.Add("rooms.remote.treasure_picks = " + Counter(r, "rooms.remote.treasure_picks"));
                if (Counter(r, "rooms.treasure.relics_obtained") != n) problems.Add("rooms.treasure.relics_obtained = " + Counter(r, "rooms.treasure.relics_obtained"));
                if (Counter(r, "rooms.treasure.skips") != 0) problems.Add("rooms.treasure.skips = " + Counter(r, "rooms.treasure.skips"));
                ok &= problems.Count == 0;
                rows.Add(new { seats = n, ascension, seed = r.Seed, outcome = Describe(r), relics_obtained = Counter(r, "rooms.treasure.relics_obtained"), fallback_moves = Counter(r, "rooms.treasure.fallback_moves"), passed = problems.Count == 0, problems });
            }
        }
        results["treasure"] = new
        {
            ok,
            what = "treasure room with 2-4 seats: every seat opens its own chest (gold from its own stream) before it sees and votes for a relic; no SKIP; seats 0 and 1 vote for the same relic " +
                   "(the game's contest), the others at random; the votes cast so far are in the context; after the award (Obtain + MoveToFallback for all results) every seat has exactly one new relic; " +
                   "PROCEED only after the picking finished",
            rows,
        };
        return ok;
    }

    // ------------------------------------------------------------------------------------------------ shop

    private static bool Shop(HostSession session, int[] seatCounts, bool wire, Dictionary<string, object> results)
    {
        List<object> rows = new List<object>();
        bool ok = true;
        foreach (int n in seatCounts)
        {
            foreach (int parasolSeat in new[] { -1, 0, 1 })
            {
                List<string> problems = new List<string>();
                Audit audit = new Audit();
                bool[] removalTried = new bool[n];
                int[] buys = new int[n], deckBefore = new int[n];
                bool captured = false;
                Scenario scenario = new Scenario
                {
                    Name = "mp-shop",
                    Setup = async ctx =>
                    {
                        for (int s = 0; s < ctx.SeatCount; s++)
                        {
                            ScenarioTools.SetGold(ctx, 2000, s);
                        }
                        if (parasolSeat >= 0)
                        {
                            await ScenarioTools.GiveRelic(ctx, "LORDS_PARASOL", parasolSeat);
                        }
                        for (int s = 0; s < ctx.SeatCount; s++)
                        {
                            deckBefore[s] = ctx.Seats[s].Deck.Cards.Count;
                        }
                        await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
                    },
                    StopWhen = AllMapsOpen,
                };
                GuidedPolicy policy = new GuidedPolicy(37UL + (ulong)n);
                policy.Choose = d =>
                {
                    if (d.Kind != DecisionKind.Shop)
                    {
                        return -1;
                    }
                    int seat = d.ActingSeat;
                    if (!removalTried[seat])
                    {
                        int removal = IndexOf(d, x => x.Verb == Verb.ChooseOption && x.Source.ShopEntry is MerchantCardRemovalEntry);
                        if (removal >= 0)
                        {
                            removalTried[seat] = true;
                            return removal;
                        }
                    }
                    if (buys[seat] < 3)
                    {
                        int buy = IndexOf(d, x => x.Verb == Verb.ChooseOption && !(x.Source.ShopEntry is MerchantCardRemovalEntry));
                        if (buy >= 0)
                        {
                            buys[seat]++;
                            return buy;
                        }
                    }
                    return IndexOf(d, x => x.Verb == Verb.Proceed);
                };
                audit.Decision = (ctx, d, chosen, auto) =>
                {
                    if (d.Kind == DecisionKind.Shop)
                    {
                        captured = true;
                        ShopContext sc = (ShopContext)d.Context!;
                        if (!ReferenceEquals(sc.Inventory.Player, ctx.Seats[d.ActingSeat]) || !ReferenceEquals(sc.Inventory, ctx.RunManager.MerchantSynchronizer.GetInventoryFor(ctx.Seats[d.ActingSeat])))
                        {
                            problems.Add("the SHOP frame of seat " + d.ActingSeat + " shows another player's inventory");
                        }
                        foreach (LegalAction x in d.Actions)
                        {
                            if (x.Verb == Verb.ChooseOption && !ReferenceEquals(x.Source.ShopEntry.Player, ctx.Seats[d.ActingSeat]))
                            {
                                problems.Add("the SHOP frame of seat " + d.ActingSeat + " offers an entry of another player");
                            }
                        }
                    }
                };
                audit.Terminal = ctx =>
                {
                    if (parasolSeat >= 0 && ctx.CurrentRoom is MerchantRoom)
                    {
                        MerchantInventory inv = ctx.RunManager.MerchantSynchronizer.GetInventoryFor(ctx.Seats[parasolSeat]);
                        int stocked = inv.CharacterCardEntries.Count(e => e.IsStocked) + inv.ColorlessCardEntries.Count(e => e.IsStocked) + inv.RelicEntries.Count(e => e.IsStocked);
                        if (stocked != 0) problems.Add("LordsParasol of seat " + parasolSeat + " left " + stocked + " card / relic entries in stock");
                    }
                };
                EpisodeResult r = session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("MPSHOP:" + n + ":" + parasolSeat), Characters = Lineup(n, n + parasolSeat + 1), Scenario = scenario }, policy,
                    new EpisodeOptions { WireRoundTrip = wire, MaxPolicyDecisions = 600 }, audit);
                Clean(r, problems);
                problems.AddRange(audit.Problems);
                if (!captured) problems.Add("no SHOP decision was asked");
                long remoteBuys = Counter(r, "rooms.remote.shop_purchases"), remoteFrames = Counter(r, "combat.select.remote_frames"), yields = Counter(r, "rooms.shop.parasol_yields"), remoteOwner = Counter(r, "rooms.shop.parasol_remote_owner");
                if (remoteBuys == 0 && !(n == 2 && parasolSeat == 1)) problems.Add("no remote purchase"); // (2 seats, the remote seat owns the parasol: it bought everything already)
                if (remoteFrames == 0) problems.Add("no remote card-removal prompt (combat.select.remote_frames = 0)");
                if (parasolSeat == 0 && (yields != 1 || remoteOwner != 0)) problems.Add("local LordsParasol: yields " + yields + ", remote-owner calls " + remoteOwner);
                if (parasolSeat == 1 && (yields != 0 || remoteOwner != 1)) problems.Add("remote LordsParasol: yields " + yields + ", remote-owner calls " + remoteOwner);
                if (parasolSeat < 0 && (yields != 0 || remoteOwner != 0)) problems.Add("no LordsParasol: yields " + yields + ", remote-owner calls " + remoteOwner);
                ok &= problems.Count == 0;
                rows.Add(new { seats = n, parasol_seat = parasolSeat, seed = r.Seed, outcome = Describe(r), purchases = Counter(r, "rooms.shop.purchases"), remote_purchases = remoteBuys, remote_prompt_frames = remoteFrames, passed = problems.Count == 0, problems });
            }
        }
        results["shop"] = new
        {
            ok,
            what = "merchant room with 2-4 seats: every seat's SHOP frame shows its own inventory; remote seats buy through MerchantEntryPurchasedMessage (the result of the purchase task is observed, " +
                   "mp-remote-purchase), a remote card removal asks through the remote prompt path; LordsParasol of the local seat yields once (rooms-lords-parasol-ui-yield) and of a remote seat " +
                   "runs without the yield, and buys every card and relic of its owner's inventory",
            rows,
        };
        return ok;
    }

    // ------------------------------------------------------------------------------------------------ crystal sphere

    private static bool Crystal(HostSession session, int[] seatCounts, bool wire, Dictionary<string, object> results)
    {
        List<object> rows = new List<object>();
        bool ok = true;
        long frames = 0, sent = 0, screens = 0;
        foreach (int n in seatCounts)
        {
            for (int i = 0; i < 3; i++)
            {
                List<string> problems = new List<string>();
                Audit audit = new Audit();
                Scenario scenario = new Scenario
                {
                    Name = "mp-crystal",
                    Setup = async ctx =>
                    {
                        for (int s = 0; s < ctx.SeatCount; s++)
                        {
                            ScenarioTools.SetGold(ctx, 500, s);
                        }
                        await ScenarioTools.JumpToEvent(ctx, "CRYSTAL_SPHERE");
                    },
                    StopWhen = AllMapsOpen,
                };
                audit.Decision = (ctx, d, chosen, auto) =>
                {
                    if (d.Kind == DecisionKind.Crystal)
                    {
                        CrystalContext cc = (CrystalContext)d.Context!;
                        if (!ReferenceEquals(cc.Minigame.Owner, ctx.Seats[d.ActingSeat])) problems.Add("the CRYSTAL frame of seat " + d.ActingSeat + " is over another player's minigame");
                        foreach (LegalAction x in d.Actions)
                        {
                            if (x.Verb != Verb.CrystalClick || !cc.Cells[x.Source.Index].Hidden)
                            {
                                problems.Add("the CRYSTAL frame of seat " + d.ActingSeat + " offers something else than a click on a hidden cell");
                                break;
                            }
                        }
                    }
                };
                EpisodeResult r = session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("MPCRYSTAL:" + n + ":" + i), Characters = Lineup(n, n + i), Scenario = scenario }, new RandomLegalPolicy(41UL + (ulong)(n * 7 + i)),
                    new EpisodeOptions { WireRoundTrip = wire, MaxPolicyDecisions = 800 }, audit);
                Clean(r, problems);
                problems.AddRange(audit.Problems);
                long f = Counter(r, "events.mp.remote_crystal_frames"), s2 = Counter(r, "events.mp.remote_crystal_rewards_sent");
                if (f != n - 1 || s2 != n - 1) problems.Add("remote crystal frames " + f + ", reward messages " + s2 + ", expected " + (n - 1) + " each");
                if (audit.ByKind[(int)DecisionKind.Crystal] == 0) problems.Add("no CRYSTAL decision");
                for (int s = 0; s < n; s++)
                {
                    if (audit.ByKindSeat[(int)DecisionKind.Crystal, s] == 0) problems.Add("seat " + s + " never clicked");
                }
                frames += f;
                sent += s2;
                screens += Counter(r, "rooms.remote.reward_screens");
                ok &= problems.Count == 0;
                rows.Add(new { seats = n, seed = r.Seed, outcome = Describe(r), crystal_decisions = audit.ByKind[(int)DecisionKind.Crystal], remote_reward_screens = Counter(r, "rooms.remote.reward_screens"), passed = problems.Count == 0, problems });
            }
        }
        results["crystal"] = new
        {
            ok,
            what = "Crystal Sphere with 2-4 seats: every seat plays its own minigame - the local seat on the screen stand-in, a remote seat on the host's minigame object of that player " +
                   "(events-crystal-sphere-remote), hidden cells only; after its last click the seat's client sends CrystalSphereRewardsMessage and the host offers that seat's rewards as a reward screen",
            remote_crystal_frames = frames,
            reward_messages = sent,
            remote_reward_screens = screens,
            rows,
        };
        return ok;
    }

    // ------------------------------------------------------------------------------------------------ bundle

    private static bool Bundle(HostSession session, int[] seatCounts, bool wire, Dictionary<string, object> results)
    {
        List<object> rows = new List<object>();
        bool ok = true;
        foreach (int n in seatCounts)
        {
            List<string> problems = new List<string>();
            Audit audit = new Audit();
            int[] before = new int[n], after = new int[n];
            Scenario scenario = new Scenario
            {
                Name = "mp-bundle",
                Setup = async ctx =>
                {
                    for (int s = 0; s < n; s++)
                    {
                        before[s] = ctx.Seats[s].Deck.Cards.Count;
                    }
                    await ScenarioTools.GiveRelic(ctx, "SCROLL_BOXES", 0);
                    await ScenarioTools.GiveRelic(ctx, "SCROLL_BOXES", n - 1);
                    for (int s = 0; s < n; s++)
                    {
                        after[s] = ctx.Seats[s].Deck.Cards.Count;
                    }
                },
                StopWhen = _ => true,
            };
            GuidedPolicy policy = new GuidedPolicy(5UL);
            policy.Choose = d => d.Kind == DecisionKind.Bundle ? IndexOf(d, x => x.Verb == Verb.ChooseOption) : -1;
            int[] bundleSize = new int[n];
            audit.Decision = (ctx, d, chosen, auto) =>
            {
                if (d.Kind == DecisionKind.Bundle)
                {
                    BundleContext bc = (BundleContext)d.Context!;
                    if (!ReferenceEquals(bc.Player, ctx.Seats[d.ActingSeat])) problems.Add("BUNDLE frame of seat " + d.ActingSeat + " names another player");
                    bundleSize[d.ActingSeat] = bc.Bundles[d.Actions[chosen].Source.Index].Count;
                }
            };
            EpisodeResult r = session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("MPBUNDLE:" + n), Characters = Lineup(n, n), Scenario = scenario }, policy, new EpisodeOptions { WireRoundTrip = wire }, audit);
            Clean(r, problems);
            problems.AddRange(audit.Problems);
            if (Counter(r, "events.mp.remote_bundle_frames") != 1) problems.Add("events.mp.remote_bundle_frames = " + Counter(r, "events.mp.remote_bundle_frames"));
            if (audit.ByKindSeat[(int)DecisionKind.Bundle, 0] != 1 || audit.ByKindSeat[(int)DecisionKind.Bundle, n - 1] != 1) problems.Add("BUNDLE decisions: seat 0 " + audit.ByKindSeat[(int)DecisionKind.Bundle, 0] + ", seat " + (n - 1) + " " + audit.ByKindSeat[(int)DecisionKind.Bundle, n - 1]);
            foreach (int s in new[] { 0, n - 1 })
            {
                if (bundleSize[s] == 0 || after[s] - before[s] != bundleSize[s]) problems.Add("seat " + s + ": deck +" + (after[s] - before[s]) + " cards, the chosen bundle has " + bundleSize[s]);
            }
            ok &= problems.Count == 0;
            rows.Add(new { seats = n, seed = r.Seed, outcome = Describe(r), passed = problems.Count == 0, problems });
        }
        results["bundle"] = new
        {
            ok,
            what = "Scroll Boxes picked up by the local seat and by the last (remote) seat: the remote seat's bundle prompt is a BUNDLE frame of that seat (events-bundle-remote-capture), answered " +
                   "with the client's choice message, and the chosen bundle's cards are in that seat's deck",
            rows,
        };
        return ok;
    }

    // ------------------------------------------------------------------------------------------------ shared events

    private static bool Shared(HostSession session, int[] seatCounts, bool wire, Dictionary<string, object> results)
    {
        List<object> rows = new List<object>();
        bool ok = true;
        long fights = 0, pages = 0, cast = 0;
        List<EventModel> shared = ModelDb.AllEvents.Where(e => e.IsShared).OrderBy(e => e.Id.Entry, StringComparer.Ordinal).ToList();
        HashSet<string> excluded = new HashSet<string>(StringComparer.Ordinal) { "FAKE_MERCHANT", "WAR_HISTORIAN_REPY" }; // single-player only / never allowed
        foreach (EventModel ev in shared)
        {
            if (excluded.Contains(ev.Id.Entry))
            {
                continue;
            }
            foreach (int n in seatCounts)
            {
                foreach (bool teamVotes in new[] { false, true })
                {
                    for (int i = 0; i < 2; i++)
                    {
                        List<string> problems = new List<string>();
                        Audit audit = new Audit();
                        bool sawEvent = false;
                        Scenario scenario = new Scenario
                        {
                            Name = "mp-shared",
                            Setup = async ctx =>
                            {
                                for (int s = 0; s < ctx.SeatCount; s++)
                                {
                                    ScenarioTools.SetGold(ctx, 300, s);
                                }
                                await ScenarioTools.JumpToEvent(ctx, ev);
                            },
                            StopWhen = ctx =>
                            {
                                sawEvent |= ctx.CurrentRoom is EventRoom;
                                return sawEvent && AllMapsOpen(ctx);
                            },
                        };
                        audit.Decision = (ctx, d, chosen, auto) =>
                        {
                            if (d.Kind == DecisionKind.Event)
                            {
                                EventContext ec = (EventContext)d.Context!;
                                if (!ReferenceEquals(ec.Event, ctx.RunManager.EventSynchronizer.GetEventForPlayer(ctx.Seats[d.ActingSeat]))) problems.Add("EVENT frame of seat " + d.ActingSeat + " shows another player's event instance");
                                if (!ec.Finished && ec.SeatVotes.Count != ctx.SeatCount) problems.Add("EventContext.SeatVotes has " + ec.SeatVotes.Count + " entries on a shared page");
                            }
                        };
                        EpisodeResult r = session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("MPSHARED:" + ev.Id.Entry + ":" + n + ":" + teamVotes + ":" + i), Characters = Lineup(n, n + i), Scenario = scenario },
                            new RandomLegalPolicy(53UL + (ulong)(n * 11 + i)), new EpisodeOptions { WireRoundTrip = wire, TeamVotes = teamVotes, DebugForcedHp = 99999, MaxPolicyDecisions = 6000 }, audit);
                        Clean(r, problems);
                        problems.AddRange(audit.Problems);
                        long votes = Counter(r, "events.mp.shared_votes"), resolved = Counter(r, "events.mp.shared_pages_resolved"), team = Counter(r, "events.mp.team_votes_cast"), notClickable = Counter(r, "events.mp.team_vote_not_clickable");
                        if (resolved < 1) problems.Add("no page of the shared event was resolved");
                        if (votes != resolved * n) problems.Add(votes + " votes for " + resolved + " resolved pages of " + n + " seats");
                        if (teamVotes && team + notClickable != resolved * (n - 1)) problems.Add("team votes cast " + team + " (+" + notClickable + " asked instead) for " + resolved + " pages");
                        if (!teamVotes && team != 0) problems.Add("team votes cast without the option");
                        fights += Counter(r, "driver.kind.Combat") > 0 ? 1 : 0;
                        pages += resolved;
                        cast += team;
                        ok &= problems.Count == 0;
                        rows.Add(new { @event = ev.Id.Entry, seats = n, team_votes = teamVotes, seed = r.Seed, outcome = Describe(r), pages = resolved, votes, team_votes_cast = team, fight = Counter(r, "driver.kind.Combat") > 0, passed = problems.Count == 0, problems });
                    }
                }
            }
        }
        results["shared"] = new
        {
            ok,
            what = "every shared event that multiplayer allows, 2-4 seats, forced HP: every seat votes on every page (seat 0 through ChooseLocalOption, remote seats with VotedForSharedEventOptionMessage), " +
                   "the host draws the option and runs it for every instance; with team votes the page is asked once and cast for the other seats; event fights start without any remote input and " +
                   "the run reaches the map of every seat",
            events = shared.Where(e => !excluded.Contains(e.Id.Entry)).Select(e => e.Id.Entry).ToList(),
            pages_resolved = pages,
            team_votes_cast = cast,
            episodes_with_a_fight = fights,
            rows,
        };
        return ok;
    }

    // ------------------------------------------------------------------------------------------------ per-seat isolation

    /// <summary>
    /// Plan M2-4: the out-of-combat streams are per player (PlayerRngSet, seeded hash(seed) + slot). Two episodes with the
    /// same seed: in A the last seat buys from its shop before it leaves, in B it leaves at once; everybody else leaves
    /// at once in both. At the end the other seats' signatures (gold, HP, deck, relics, potions, the four player RNG
    /// streams) must be equal in A and B, and the buyer's must differ.
    /// </summary>
    private static bool Isolation(HostSession session, int[] seatCounts, bool wire, Dictionary<string, object> results)
    {
        List<object> rows = new List<object>();
        bool ok = true;
        foreach (int n in seatCounts)
        {
            List<string> problems = new List<string>();
            ulong[][] sig = new ulong[2][];
            int[] purchases = new int[2];
            string seed = HostSession.SeedFromLabel("MPISOLATION:" + n);
            for (int variant = 0; variant < 2; variant++)
            {
                int v = variant;
                int buyer = n - 1;
                bool bought = false;
                Audit audit = new Audit();
                Scenario scenario = new Scenario
                {
                    Name = "mp-isolation",
                    Setup = async ctx =>
                    {
                        for (int s = 0; s < ctx.SeatCount; s++)
                        {
                            ScenarioTools.SetGold(ctx, 2000, s);
                        }
                        await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
                    },
                    StopWhen = AllMapsOpen,
                };
                GuidedPolicy policy = new GuidedPolicy(7UL);
                policy.Choose = d =>
                {
                    if (d.Kind != DecisionKind.Shop)
                    {
                        return 0; // any prompt of the purchase: the first legal action, the same in both variants
                    }
                    if (v == 0 && d.ActingSeat == buyer && !bought)
                    {
                        int buy = IndexOf(d, x => x.Verb == Verb.ChooseOption && !(x.Source.ShopEntry is MerchantCardRemovalEntry));
                        if (buy >= 0)
                        {
                            bought = true;
                            return buy;
                        }
                    }
                    return IndexOf(d, x => x.Verb == Verb.Proceed);
                };
                audit.Terminal = ctx =>
                {
                    sig[v] = new ulong[n];
                    for (int s = 0; s < n; s++)
                    {
                        sig[v][s] = MultiSeatRun.SeatSignature(ctx.Seats[s]);
                    }
                };
                EpisodeResult r = session.RunEpisode(new EpisodeSpec { Seed = seed, Characters = Lineup(n, n), Scenario = scenario }, policy, new EpisodeOptions { WireRoundTrip = wire, MaxPolicyDecisions = 200 }, audit);
                Clean(r, problems);
                problems.AddRange(audit.Problems);
                purchases[v] = (int)Counter(r, "rooms.shop.purchases");
            }
            if (purchases[0] != 1 || purchases[1] != 0) problems.Add("purchases: " + purchases[0] + " in A, " + purchases[1] + " in B (expected 1 and 0)");
            if (sig[0] == null || sig[1] == null)
            {
                problems.Add("no terminal state");
            }
            else
            {
                for (int s = 0; s < n - 1; s++)
                {
                    if (sig[0][s] != sig[1][s]) problems.Add("seat " + s + " differs between the two episodes although only seat " + (n - 1) + " acted differently");
                }
                if (sig[0][n - 1] == sig[1][n - 1]) problems.Add("the buyer's state is the same with and without the purchase");
            }
            ok &= problems.Count == 0;
            rows.Add(new { seats = n, seed, passed = problems.Count == 0, problems });
        }
        results["isolation"] = new
        {
            ok,
            what = "two shop episodes with the same seed that differ only in one purchase of the last seat: every other seat's gold, deck, relics, potions and player RNG streams are identical " +
                   "at the end, the buyer's differ (per-player out-of-combat streams; a remote seat's purchase reaches only its own inventory and player)",
            rows,
        };
        return ok;
    }

    // ------------------------------------------------------------------------------------------------ whole runs

    private static bool Runs(HostSession session, int[] seatCounts, bool wire, int episodes, Dictionary<string, object> results)
    {
        List<object> rows = new List<object>();
        bool ok = true;
        SortedDictionary<string, long> sums = new SortedDictionary<string, long>(StringComparer.Ordinal);
        long wins = 0, losses = 0, truncated = 0, ran = 0;
        foreach (int n in seatCounts)
        {
            for (int i = 0; i < episodes; i++)
            {
                List<string> problems = new List<string>();
                bool teamVotes = i % 2 == 1;
                int hp = i % 3 == 2 ? 0 : 99999;
                EpisodeResult r = session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("MPRUN:" + n + ":" + i), Characters = Lineup(n, i), Ascension = i % 2 == 0 ? 10 : 0 },
                    new RandomLegalPolicy(61UL + (ulong)(n * 13 + i)), new EpisodeOptions { WireRoundTrip = wire, TeamVotes = teamVotes, DebugForcedHp = hp, ComputeSignatures = true });
                Clean(r, problems, expectScenarioStop: false);
                if (r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy != "combat_turn_cap") problems.Add(Describe(r));
                ran++;
                wins += r.Outcome == EpisodeOutcome.Win ? 1 : 0;
                losses += r.Outcome == EpisodeOutcome.Loss ? 1 : 0;
                truncated += r.Outcome == EpisodeOutcome.Truncated ? 1 : 0;
                foreach (KeyValuePair<string, long> kv in r.ModuleCounters)
                {
                    if (kv.Key.StartsWith("rooms.remote.", StringComparison.Ordinal) || kv.Key.StartsWith("events.mp.", StringComparison.Ordinal) || kv.Key.StartsWith("mp.", StringComparison.Ordinal)
                        || kv.Key.StartsWith("rooms.tree.", StringComparison.Ordinal) || kv.Key.StartsWith("rooms.rest.mend", StringComparison.Ordinal) || kv.Key == "fidelity.mend_local_targetings")
                    {
                        sums[kv.Key] = kv.Key.EndsWith("_max", StringComparison.Ordinal) ? Math.Max(sums.GetValueOrDefault(kv.Key), kv.Value) : sums.GetValueOrDefault(kv.Key) + kv.Value;
                    }
                }
                ok &= problems.Count == 0;
                rows.Add(new { seats = n, lineup = string.Join("+", r.Characters), ascension = r.Ascension, hp, team_votes = teamVotes, seed = r.Seed, outcome = Describe(r), floor = r.TotalFloor, policy_decisions = r.Counters.PolicyDecisions, passed = problems.Count == 0, problems });
            }
        }
        List<string> coverage = new List<string>();
        foreach (string key in new[] { "rooms.remote.tree.act_vote", "rooms.tree.act_change", "rooms.remote.reward_exit_skips", "rooms.remote.card_reward_frames", "rooms.remote.shop_purchases",
                     "rooms.remote.rest_choices", "rooms.remote.chests_opened", "events.mp.remote_options", "events.mp.shared_votes", "mp.remote.map_votes", "mp.team_votes.map" })
        {
            if (sums.GetValueOrDefault(key) == 0)
            {
                coverage.Add(key + " = 0");
            }
        }
        ok &= coverage.Count == 0;
        results["runs"] = new
        {
            ok,
            what = "whole runs with 2-4 seats and a random legal policy (A0 / A10, forced and natural HP, with and without team votes): no abort, no violation counter; the coverage counters " +
                   "listed in not_covered must all be non-zero over the set",
            episodes = ran,
            won = wins,
            lost = losses,
            truncated,
            not_covered = coverage,
            counters = sums,
            rows,
        };
        return ok;
    }
}
