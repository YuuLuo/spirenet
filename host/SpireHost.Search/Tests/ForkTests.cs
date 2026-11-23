using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Rngs;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Obs;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Tests;
using SpireHost.Sim;

namespace SpireHost.Search.Tests;

/// <summary>
/// TEST TOOL (search build; reads and writes hidden state on purpose). The fork study's fidelity / leak experiment
/// (notes/search/fork-study.md section 2; alt/search/fork ForkStudy.cs) re-run through the search module: children are made
/// by <see cref="ForkPrimitive.Fork"/> (with its child hygiene) and re-randomised by <see cref="Resampler"/>.
///
///   SpireHost --search-fork-test --mode fidelity --roles R1,R2,... [--pile-order canonical|stable] [--make-pair]
///             --lineups A+B+C+D,... [--room any|boss|elite] [--fight-index N] [--at N | --at-max N] [--fork-kind Combat|CardSelect]
///             [--encounter ID[@act]] [--hp N --turn-cap N] --runs N [--first-run K] [--seed-base S] --out FILE
///
/// At one decision of the chosen fight the parent forks one child per role; every process then plays with the SAME policy
/// seed to the end of the fight and records, at every decision, RunSignature, the game's combat checksum, DeepSignature,
/// the TEAM frame hash and the decision shape. Roles:
///   exact       no change (H1 fidelity: must equal the parent)
///   nested      no change; forks a grandchild at its 3rd record and compares with it (H1)
///   rerand1/2   re-randomised with seeds 1001 / 2002
///   hiddenmove  every monster given another move of its state machine (the serialised combat state does not see it)
///   leakB       re-randomised with seed 3003
///   leakA       first destroys the hidden truth (every draw pile reversed, every Rng object replaced), then seed 3003
///   leakC       first swaps every pair of same-Id draw-pile cards whose states differ (Strike / Strike+), keeps every other
///               position, replaces every Rng object, then seed 3003
/// T1 of the leak test: leakA == leakB and leakC == leakB on the four hashes to the fight's end.
/// --pile-order stable = the game's StableShuffle in the resampler (positive control: must FAIL on same-Id pairs).
/// --make-pair = before forking, the parent upgrades one card of a same-Id pair in a draw pile (constructs the pair case).
/// </summary>
public static class ForkTests
{
    private sealed class Rec
    {
        public long Dp;
        public int Kind;
        public int NActions;
        public int Chosen;
        public bool Auto;
        public ulong RunSig;
        public uint Checksum;
        public ulong Deep;
        public ulong Frame;
    }

    private sealed class Child
    {
        public int Pid;
        public int Fd;
        public string Role = "";
        public double ForkMs;
        public long T0;
    }

    private sealed class State
    {
        public HostSession Session = null!;
        public Args A = null!;
        public ObsEncoder Enc = null!;
        public PileOrder Order;
        public bool IsChild;
        public string Role = "parent";
        public int WriteFd = -1;
        public IPolicy Policy = null!;
        public List<Rec>? Recs;
        public bool RecsClosed;
        public CombatRoom? ForkRoom;
        public long ForkT0;
        public long DecisionsAtFork;
        public long PolicyAtFork;
        public StrictCounters StrictAtFork;
        public int RoundAtFork;
        public Dictionary<string, object?> ChildInfo = new Dictionary<string, object?>();
        public Dictionary<string, object?>? Terminal;
        public bool FightOver;

        public string RoomFilter = "any";
        public int FightIndex = 1;
        public int At = 1;
        public int ChildTimeoutMs = 600_000;
        public string ForkKind = "any";
        public int NestedAt = 3;
        public Child? NestedChild;
        public string Lineup = "";
        public int FightsSeen;
        public CombatRoom? CurrentFight;
        public int FightDecisions;
        public bool Forked;
        public bool InTargetFight;

        public byte[]? Frame0;
        public byte[]? Net0;
        public readonly List<Child> Children = new List<Child>();
        public readonly List<object> ForkPoints = new List<object>();
    }

    private static State S = null!;

    public static int Run(HostSession session, Args a)
    {
        string? err = ForkPrimitive.RuntimeError();
        if (err != null)
        {
            Console.Error.WriteLine("search-fork-test: " + err);
            return 1;
        }
        if (a.Get("mode", "fidelity") != "fidelity")
        {
            Console.Error.WriteLine("search-fork-test: --mode fidelity is the only mode");
            return 1;
        }
        S = new State
        {
            Session = session,
            A = a,
            Enc = ObsRuntime.NewEncoder(ObsModule.Current),
            Order = a.Get("pile-order", "canonical") switch
            {
                "canonical" => PileOrder.Canonical,
                "stable" => PileOrder.GameStableShuffle,
                string x => throw new ArgumentException("--pile-order " + x),
            },
            RoomFilter = a.Get("room", "any").ToLowerInvariant(),
            FightIndex = a.GetInt("fight-index", 1),
            At = a.GetInt("at", 1),
            ForkKind = a.Get("fork-kind", "any"),
            NestedAt = a.GetInt("nested-at", 3),
            ChildTimeoutMs = a.GetInt("child-timeout-ms", 600_000),
        };
        SearchCommands.RegisterGameWarmUps();
        ForkPrimitive.InitRoot();
        // children write JSON: build System.Text.Json's metadata for every shape once here
        JsonSerializer.SerializeToUtf8Bytes(new Dictionary<string, object?>
        {
            ["i"] = 1, ["l"] = 1L, ["d"] = 1.0, ["b"] = true, ["s"] = "x", ["n"] = null,
            ["m"] = new Dictionary<string, long> { ["x"] = 1 }, ["ls"] = new List<string> { "x" },
            ["r"] = new List<object[]> { new object[] { 1L, 1, 1, 1, false, "1", 1u, "1", "1" } },
            ["c"] = Compare(new List<Rec>(), new List<Rec>()),
        });
        int runs = a.GetInt("runs", 1);
        int firstRun = a.GetInt("first-run", 0);
        string[] lineups = a.Get("lineups", "DEFECT+DEFECT+IRONCLAD+IRONCLAD").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int atMax = a.GetInt("at-max", 0);
        string seedBase = a.Get("seed-base", "FORK");
        int ascension = a.GetInt("ascension", 10);
        EpisodeOptions options = DriverCommands.EpisodeOptionsFrom(a);
        options.TeamVotes = a.GetBool("team-votes", true);
        options.OfferUndoEndTurn = a.GetBool("undo", false);
        string? encounter = a.Get("encounter");
        List<object> episodes = new List<object>();
        Stopwatch wall = Stopwatch.StartNew();
        for (int i = firstRun; i < firstRun + runs; i++)
        {
            string seed = HostSession.SeedFromLabel(seedBase + ":" + i);
            ulong policySeed = a.GetULong("policy-seed", 1) * 7919UL + (ulong)i;
            EpisodeSpec spec = new EpisodeSpec
            {
                Seed = seed,
                Characters = lineups[i % lineups.Length].Split('+', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
                Ascension = ascension,
                Scenario = encounter == null ? null : JumpScenario(encounter),
            };
            ResetRunState();
            S.Lineup = lineups[i % lineups.Length];
            if (atMax > 0)
            {
                S.At = 1 + (int)((ulong)(i * 7919 + 13) % (ulong)atMax);
            }
            S.Policy = new RandomLegalPolicy(policySeed);
            Stopwatch ep = Stopwatch.StartNew();
            EpisodeResult r = session.RunEpisode(spec, new TestPolicy(), options, new TestObserver());
            if (S.IsChild)
            {
                ChildFinish(null, r);
            }
            episodes.Add(new
            {
                run = i,
                seed,
                lineup = S.Lineup,
                at = S.At,
                outcome = r.Outcome.ToString(),
                truncated_by = r.TruncatedBy,
                abort = r.Abort?.Signature,
                abort_message = r.Abort?.Message,
                floor = r.TotalFloor,
                policy_decisions = r.Counters.PolicyDecisions,
                decision_points = r.Counters.DecisionPoints,
                ms = Math.Round(ep.Elapsed.TotalMilliseconds, 1),
                fork_points = S.ForkPoints.ToList(),
            });
            Console.Error.WriteLine("[search-fork-test] run " + i + " done in " + ep.Elapsed.TotalSeconds.ToString("0.0") + " s");
        }
        ForkPrimitive.ReapNonBlocking();
        HostJson.Emit(new
        {
            kind = "search-fork-test",
            mode = "fidelity",
            pile_order = S.Order.ToString(),
            make_pair = a.Has("make-pair"),
            roles = a.Get("roles", "exact"),
            ascension,
            encounter,
            room = S.RoomFilter,
            fight_index = S.FightIndex,
            pid = Libc.getpid(),
            env = new
            {
                wx = Environment.GetEnvironmentVariable("DOTNET_EnableWriteXorExecute"),
                tc = Environment.GetEnvironmentVariable("DOTNET_TieredCompilation"),
                gen0 = Environment.GetEnvironmentVariable("DOTNET_GCgen0size"),
            },
            reaped = ForkPrimitive.Reaped,
            abnormal_exits_reaped = ForkPrimitive.AbnormalExits,
            seconds = Math.Round(wall.Elapsed.TotalSeconds, 2),
            episodes,
        }, a.Get("out"));
        return 0;
    }

    private static void ResetRunState()
    {
        S.Recs = null;
        S.RecsClosed = false;
        S.ForkRoom = null;
        S.FightsSeen = 0;
        S.CurrentFight = null;
        S.FightDecisions = 0;
        S.Forked = false;
        S.InTargetFight = false;
        S.FightOver = false;
        S.Children.Clear();
        S.ForkPoints.Clear();
        S.Terminal = null;
    }

    private static Scenario JumpScenario(string encounterSpec)
    {
        int at = encounterSpec.IndexOf('@');
        string enc = at < 0 ? encounterSpec : encounterSpec.Substring(0, at);
        int act = at < 0 ? 0 : int.Parse(encounterSpec.Substring(at + 1));
        return new Scenario
        {
            Name = "search-fork-test-jump",
            Setup = async ctx =>
            {
                if (act > 0)
                {
                    await ScenarioTools.EnterAct(ctx, act, doTransition: false);
                }
                await ScenarioTools.JumpToEncounter(ctx, enc);
            },
        };
    }

    // =========================================================================================================
    //  Policy and observer
    // =========================================================================================================

    private static bool InCombat(RunContext ctx, out CombatRoom? room)
    {
        room = ctx.Run.CurrentRoom as CombatRoom;
        return room != null && ctx.CombatManager.IsInProgress;
    }

    private static bool RoomMatches(CombatRoom room)
    {
        return S.RoomFilter switch
        {
            "boss" => room.RoomType == RoomType.Boss,
            "elite" => room.RoomType == RoomType.Elite,
            "monster" => room.RoomType == RoomType.Monster,
            _ => true,
        };
    }

    private sealed class TestPolicy : IPolicy
    {
        public int Decide(Decision d)
        {
            RunContext ctx = d.Episode;
            bool inFight = InCombat(ctx, out CombatRoom? room);
            if (S.IsChild)
            {
                if (!inFight || !ReferenceEquals(room, S.ForkRoom))
                {
                    ChildFinish(ctx, null);
                }
                if (S.Role == "nested" && S.NestedChild == null && S.Recs != null && S.Recs.Count == S.NestedAt)
                {
                    ForkGrandchild();
                }
                return S.Policy.Decide(d);
            }
            if (S.ForkRoom != null && (!inFight || !ReferenceEquals(room, S.ForkRoom)))
            {
                // parent after its fork point: end the episode at the end of the fork fight
                ParentFightOver(ctx);
                throw new PolicyStopException("fork fight over");
            }
            if (inFight)
            {
                if (!ReferenceEquals(room, S.CurrentFight))
                {
                    S.CurrentFight = room;
                    S.FightDecisions = 0;
                    S.InTargetFight = RoomMatches(room!) && ++S.FightsSeen == S.FightIndex;
                }
                S.FightDecisions++;
                if (S.InTargetFight && !S.Forked && S.FightDecisions >= S.At && (S.ForkKind == "any" ? S.FightDecisions == S.At : d.Kind.ToString() == S.ForkKind))
                {
                    S.Forked = true;
                    return ForkHere(ctx, d, room!);
                }
            }
            return S.Policy.Decide(d);
        }
    }

    private sealed class TestObserver : IEpisodeObserver
    {
        public void OnEpisodeStart(RunContext ctx)
        {
        }

        public void OnRoomBoundary(RunContext ctx)
        {
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
        {
            if (S.Recs != null && !S.RecsClosed)
            {
                S.Recs.Add(Record(ctx, decision, chosen, auto));
                if (!S.IsChild && (!InCombat(ctx, out CombatRoom? pr) || !ReferenceEquals(pr, S.ForkRoom)))
                {
                    // the parent records exactly as far as a child does: up to and including the first decision after the fight
                    S.RecsClosed = true;
                }
            }
            if (S.IsChild && (!InCombat(ctx, out CombatRoom? room) || !ReferenceEquals(room, S.ForkRoom)))
            {
                ChildFinish(ctx, null);
            }
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
            S.Terminal = FightSummary(ctx);
            S.Terminal["outcome"] = provisional.Outcome.ToString();
            S.Terminal["truncated_by"] = provisional.TruncatedBy;
            S.Terminal["abort"] = provisional.Abort?.Signature;
            S.Terminal["abort_message"] = provisional.Abort?.Message;
            if (!S.IsChild && S.ForkRoom != null && !S.FightOver)
            {
                ParentFightOver(ctx);
            }
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }
    }

    private static Rec Record(RunContext ctx, Decision d, int chosen, bool auto)
    {
        Rec r = new Rec
        {
            Dp = ctx.Counters.DecisionPoints,
            Kind = (int)d.Kind,
            NActions = d.Actions.Count,
            Chosen = chosen,
            Auto = auto,
            RunSig = RunSignature.Compute(ctx.Run),
            Deep = DeepSignature.Compute(ctx.Run),
            Frame = FrameHash(EncodeFrame(d)),
        };
        if (ctx.CombatManager.IsInProgress)
        {
            r.Checksum = RunManager.Instance.ChecksumTracker.GenerateChecksum(NetFullCombatState.FromRun(ctx.Run, null));
        }
        return r;
    }

    private static byte[] EncodeFrame(Decision d)
    {
        S.Enc.EncodeDecision(d, 1, 0, 0);
        return S.Enc.Payload.ToArray();
    }

    private static readonly int[] HeaderWordsIgnored = { SpireHost.Obs.Schema.Hd.EpisodeId, SpireHost.Obs.Schema.Hd.StepId, SpireHost.Obs.Schema.Hd.AutoSteps, SpireHost.Obs.Schema.Hd.HostUs };

    private static ulong FrameHash(byte[] f)
    {
        SigHasher h = SigHasher.Create();
        for (int i = 0; i + 3 < f.Length; i += 4)
        {
            int w = i / 4;
            if (w < SpireHost.Obs.Schema.SchemaV1.HeaderLen && Array.IndexOf(HeaderWordsIgnored, w) >= 0)
            {
                continue;
            }
            h.Add(BitConverter.ToUInt32(f, i));
        }
        h.Add(f.Length);
        return h.Finish();
    }

    private static byte[] NetStateBytes(RunState run)
    {
        NetFullCombatState st = NetFullCombatState.FromRun(run, null);
        PacketWriter w = new PacketWriter();
        w.Write(st);
        w.ZeroByteRemainder();
        return w.Buffer.AsSpan(0, w.BytePosition).ToArray();
    }

    private static Dictionary<string, object?> FightSummary(RunContext ctx)
    {
        Dictionary<string, object?> o = new Dictionary<string, object?>();
        CombatRoom? room = S.ForkRoom ?? ctx.Run.CurrentRoom as CombatRoom;
        o["combat_in_progress"] = ctx.CombatManager.IsInProgress;
        o["game_over"] = ctx.Run.IsGameOver;
        o["team_hp"] = ctx.Run.Players.Sum(p => p.Creature.CurrentHp);
        o["seats_alive"] = ctx.Run.Players.Count(p => p.Creature.IsAlive);
        if (room != null)
        {
            CombatState cs = room.CombatState;
            o["round"] = cs.RoundNumber;
            o["enemy_hp"] = cs._enemies.Where(e => e.IsAlive).Sum(e => e.CurrentHp);
            o["enemies_alive"] = cs._enemies.Count(e => e.IsAlive);
        }
        return o;
    }

    // =========================================================================================================
    //  Fork
    // =========================================================================================================

    private static int ForkHere(RunContext ctx, Decision d, CombatRoom room)
    {
        Dictionary<string, object?> fp = new Dictionary<string, object?>
        {
            ["encounter"] = room.Encounter?.Id.Entry,
            ["room_type"] = room.RoomType.ToString(),
            ["total_floor"] = ctx.Run.TotalFloor,
            ["fight_decision"] = S.FightDecisions,
            ["decision_kind"] = d.Kind.ToString(),
            ["n_actions"] = d.Actions.Count,
            ["round"] = room.CombatState.RoundNumber,
        };
        if (S.A.Get("make-pair") is string pairKind)
        {
            fp["pair_made"] = MakePair(ctx, pairKind);
        }
        fp["draw_pile"] = DrawPileStats(ctx);
        S.Frame0 = EncodeFrame(d);
        S.Net0 = NetStateBytes(ctx.Run);
        S.ForkPoints.Add(fp);
        S.ForkRoom = room;
        ulong postSeed = S.A.GetULong("post-seed", 4242);
        string[] roles = S.A.Get("roles", "exact,rerand1,rerand2,hiddenmove").Split(',');
        foreach (string role in roles)
        {
            int[] fds = new int[2];
            if (Libc.pipe(fds) != 0)
            {
                throw new InvalidOperationException("pipe() failed: errno " + Libc.Errno);
            }
            long t0 = Stopwatch.GetTimestamp();
            int pid = ForkPrimitive.Fork(fds[1]);
            if (pid == 0)
            {
                S.IsChild = true;
                S.Role = role;
                S.ForkT0 = Stopwatch.GetTimestamp();
                S.WriteFd = fds[1];
                S.Children.Clear();
                S.ChildInfo = new Dictionary<string, object?> { ["role"] = role, ["pid"] = Libc.getpid() };
                S.DecisionsAtFork = ctx.Counters.DecisionPoints;
                S.PolicyAtFork = ctx.Counters.PolicyDecisions;
                S.StrictAtFork = S.Session.ReadStrict();
                S.RoundAtFork = room.CombatState.RoundNumber;
                ApplyRole(ctx, d, role);
                S.Policy = new RandomLegalPolicy(postSeed);
                S.Recs = new List<Rec>();
                return S.Policy.Decide(d);
            }
            Libc.close(fds[1]);
            S.Children.Add(new Child { Pid = pid, Fd = fds[0], Role = role, ForkMs = Ms(t0, Stopwatch.GetTimestamp()), T0 = t0 });
        }
        fp["children"] = roles;
        S.Policy = new RandomLegalPolicy(postSeed);
        S.Recs = new List<Rec>();
        return S.Policy.Decide(d);
    }

    private static void ApplyRole(RunContext c, Decision d, string role)
    {
        if (role.StartsWith("rerand", StringComparison.Ordinal))
        {
            Resampler.Info ri = Resampler.Rerandomise(c, S.Enc, role == "rerand1" ? 1001UL : role == "rerand2" ? 2002UL : 7007UL, S.Order);
            S.ChildInfo["canonical_ties"] = ri.CanonicalTies;
            S.ChildInfo["frame_equal_after_rerand"] = SearchExtension.SameFrame(S.Frame0, EncodeFrame(d));
            S.ChildInfo["net_state_equal_after_rerand"] = NetStateBytes(c.Run).AsSpan().SequenceEqual(S.Net0);
        }
        else if (role == "leakA" || role == "leakB" || role == "leakC")
        {
            if (role == "leakA")
            {
                S.ChildInfo["perturbed"] = PerturbAll(c);
            }
            else if (role == "leakC")
            {
                S.ChildInfo["perturbed"] = SwapDifferingPairs(c);
            }
            Resampler.Info ri = Resampler.Rerandomise(c, S.Enc, 3003UL, S.Order);
            S.ChildInfo["canonical_ties"] = ri.CanonicalTies;
            S.ChildInfo["frame_equal_after_rerand"] = SearchExtension.SameFrame(S.Frame0, EncodeFrame(d));
        }
        else if (role == "hiddenmove")
        {
            S.ChildInfo["moves_changed"] = ChangeNextMoves(c);
            S.ChildInfo["net_state_equal_after_change"] = NetStateBytes(c.Run).AsSpan().SequenceEqual(S.Net0);
            S.ChildInfo["frame_equal_after_change"] = SearchExtension.SameFrame(S.Frame0, EncodeFrame(d));
        }
        else if (role != "exact" && role != "nested")
        {
            throw new ArgumentException("unknown role " + role);
        }
    }

    private static List<string> ChangeNextMoves(RunContext ctx)
    {
        List<string> changed = new List<string>();
        CombatState cs = ((CombatRoom)ctx.Run.CurrentRoom!).CombatState;
        foreach (Creature e in cs._enemies)
        {
            MonsterModel? m = e.Monster;
            if (m == null || !e.IsAlive || m._moveStateMachine == null)
            {
                continue;
            }
            foreach (MonsterState st in m._moveStateMachine.States.Values)
            {
                if (st is MoveState mv && mv.Id != m.NextMove.Id)
                {
                    string from = m.NextMove.Id;
                    m.SetMoveImmediate(mv, forceTransition: true);
                    changed.Add(m.Id.Entry + ":" + from + "->" + mv.Id);
                    break;
                }
            }
        }
        return changed;
    }

    private static void ReplaceEveryRng(RunContext ctx, System.Random g)
    {
        foreach (Player p in ctx.Run.Players)
        {
            foreach (PlayerRngType t in p.PlayerRng._rngs.Keys.ToList())
            {
                p.PlayerRng._rngs[t] = new Rng((ulong)g.NextInt64());
            }
        }
        foreach (RunRngType t in ctx.Run.Rng._rngs.Keys.ToList())
        {
            ctx.Run.Rng._rngs[t] = new Rng((ulong)g.NextInt64());
        }
        CombatState cs = ((CombatRoom)ctx.Run.CurrentRoom!).CombatState;
        foreach (Creature e in cs._enemies)
        {
            if (e.Monster != null)
            {
                e.Monster._rng = new Rng((ulong)g.NextInt64());
            }
        }
        if (cs.Encounter != null)
        {
            cs.Encounter._rng = new Rng((ulong)g.NextInt64());
        }
    }

    /// <summary>leakA: reverse every draw pile, replace every Rng object of the run, the seats, the monsters and the encounter.</summary>
    private static object PerturbAll(RunContext ctx)
    {
        int differingPairsReordered = 0;
        foreach (Player p in ctx.Run.Players)
        {
            List<CardModel>? cards = p.PlayerCombatState?.DrawPile._cards;
            if (cards == null)
            {
                continue;
            }
            differingPairsReordered += DifferingSameIdPairs(cards).Count;
            cards.Reverse();
        }
        ReplaceEveryRng(ctx, new System.Random(77));
        return new { piles_reversed = true, same_id_differing_pairs = differingPairsReordered };
    }

    /// <summary>leakC: only the relative order of same-Id cards whose states differ changes (each such pair swapped); every Rng replaced.</summary>
    private static object SwapDifferingPairs(RunContext ctx)
    {
        int swapped = 0;
        foreach (Player p in ctx.Run.Players)
        {
            List<CardModel>? cards = p.PlayerCombatState?.DrawPile._cards;
            if (cards == null)
            {
                continue;
            }
            HashSet<int> used = new HashSet<int>();
            foreach ((int i, int j) in DifferingSameIdPairs(cards))
            {
                if (used.Contains(i) || used.Contains(j))
                {
                    continue;
                }
                used.Add(i);
                used.Add(j);
                (cards[i], cards[j]) = (cards[j], cards[i]);
                swapped++;
            }
        }
        ReplaceEveryRng(ctx, new System.Random(78));
        return new { pairs_swapped = swapped };
    }

    /// <summary>Position pairs (i &lt; j) of cards with the same model Id whose card states differ (state bytes or variables).</summary>
    private static List<(int, int)> DifferingSameIdPairs(List<CardModel> cards)
    {
        List<(int, int)> pairs = new List<(int, int)>();
        for (int i = 0; i < cards.Count; i++)
        {
            for (int j = i + 1; j < cards.Count; j++)
            {
                if (cards[i].Id == cards[j].Id && !SameState(cards[i], cards[j]))
                {
                    pairs.Add((i, j));
                }
            }
        }
        return pairs;
    }

    private static bool SameState(CardModel a, CardModel b)
    {
        return Resampler.StateBytes(a).AsSpan().SequenceEqual(Resampler.StateBytes(b)) && Resampler.VarBytes(a).AsSpan().SequenceEqual(Resampler.VarBytes(b));
    }

    /// <summary>
    /// --make-pair upgrade|floor (parent, at the fork point): in the first seat whose draw pile holds two cards with the same Id
    /// and the same state, make them differ: "upgrade" upgrades the one nearer the top with the game's own routine
    /// (UpgradeInternal + FinalizeUpgradeInternal; the game's comparer orders such a pair), "floor" changes its
    /// FloorAddedToDeck (a difference CardModel.CompareTo does not see: the game's StableShuffle keeps the true order).
    /// </summary>
    private static object MakePair(RunContext ctx, string kind)
    {
        if (kind != "upgrade" && kind != "floor")
        {
            throw new ArgumentException("--make-pair upgrade|floor, not " + kind);
        }
        IReadOnlyList<Player> players = ctx.Run.Players;
        for (int s = 0; s < players.Count; s++)
        {
            List<CardModel>? cards = players[s].PlayerCombatState?.DrawPile._cards;
            if (cards == null)
            {
                continue;
            }
            for (int i = 0; i < cards.Count; i++)
            {
                for (int j = i + 1; j < cards.Count; j++)
                {
                    if (cards[i].Id == cards[j].Id && (kind == "floor" || cards[i].IsUpgradable) && SameState(cards[i], cards[j]))
                    {
                        if (kind == "upgrade")
                        {
                            cards[i].UpgradeInternal();
                            cards[i].FinalizeUpgradeInternal();
                        }
                        else
                        {
                            cards[i].FloorAddedToDeck = (cards[i].FloorAddedToDeck ?? 0) + 100;
                        }
                        return new { kind, seat = s, card = cards[i].Id.Entry, pos_changed = i, pos_twin = j, pile = cards.Count };
                    }
                }
            }
        }
        return false;
    }

    private static object DrawPileStats(RunContext ctx)
    {
        int cards = 0, upgraded = 0, enchanted = 0, pairs = 0, gameTies = 0;
        List<string> examples = new List<string>();
        List<string> gameTieExamples = new List<string>();
        foreach (Player p in ctx.Run.Players)
        {
            List<CardModel>? c = p.PlayerCombatState?.DrawPile._cards;
            if (c == null)
            {
                continue;
            }
            cards += c.Count;
            upgraded += c.Count(x => x.CurrentUpgradeLevel > 0);
            enchanted += c.Count(x => x.Enchantment != null);
            List<(int, int)> dp = DifferingSameIdPairs(c);
            pairs += dp.Count;
            foreach ((int i, int j) in dp)
            {
                // the pairs the game's own StableShuffle cannot order (CardModel.CompareTo == 0): their relative order stays the true one
                if (c[i].CompareTo(c[j]) == 0)
                {
                    gameTies++;
                    if (gameTieExamples.Count < 4)
                    {
                        gameTieExamples.Add(DescribeDifference(c[i], c[j]));
                    }
                }
            }
            foreach ((int i, int j) in dp.Take(Math.Max(0, 4 - examples.Count)))
            {
                CardModel a = c[i], b = c[j];
                examples.Add(a.Id.Entry + " upg " + a.CurrentUpgradeLevel + "/" + b.CurrentUpgradeLevel + " ench " + (a.Enchantment?.Id.Entry ?? "-") + "/" + (b.Enchantment?.Id.Entry ?? "-") +
                             " state " + (Resampler.StateBytes(a).AsSpan().SequenceEqual(Resampler.StateBytes(b)) ? "same" : "differs") +
                             " vars " + (Resampler.VarBytes(a).AsSpan().SequenceEqual(Resampler.VarBytes(b)) ? "same" : "differ"));
            }
        }
        return new { cards, upgraded, enchanted, same_id_differing_pairs = pairs, game_comparer_ties_that_differ = gameTies, examples, game_tie_examples = gameTieExamples };
    }

    private static string DescribeDifference(CardModel a, CardModel b)
    {
        List<string> d = new List<string>();
        if (a.FloorAddedToDeck != b.FloorAddedToDeck)
        {
            d.Add("floor_added " + a.FloorAddedToDeck + "/" + b.FloorAddedToDeck);
        }
        if ((a.Enchantment?.Amount ?? 0) != (b.Enchantment?.Amount ?? 0) || a.Enchantment?.Status != b.Enchantment?.Status)
        {
            d.Add("enchantment " + a.Enchantment?.Id.Entry + " amount " + a.Enchantment?.Amount + "/" + b.Enchantment?.Amount + " status " + a.Enchantment?.Status + "/" + b.Enchantment?.Status);
        }
        if ((a.Affliction?.Amount ?? 0) != (b.Affliction?.Amount ?? 0))
        {
            d.Add("affliction amount " + a.Affliction?.Amount + "/" + b.Affliction?.Amount);
        }
        string ka = string.Join(",", a.Keywords.OrderBy(k => k)), kb = string.Join(",", b.Keywords.OrderBy(k => k));
        if (ka != kb)
        {
            d.Add("keywords " + ka + "/" + kb);
        }
        if (!Resampler.VarBytes(a).AsSpan().SequenceEqual(Resampler.VarBytes(b)))
        {
            d.Add("dynamic vars (enchanted / preview values)");
        }
        if (d.Count == 0)
        {
            d.Add("saved properties");
        }
        return a.Id.Entry + ": " + string.Join("; ", d);
    }

    private static void ParentFightOver(RunContext ctx)
    {
        S.FightOver = true;
        Dictionary<string, object?> parent = FightSummary(ctx);
        List<Rec> mine = S.Recs ?? new List<Rec>();
        parent["records"] = mine.Count;
        Dictionary<string, List<Rec>> byRole = new Dictionary<string, List<Rec>>();
        List<Dictionary<string, object?>> kids = new List<Dictionary<string, object?>>();
        foreach (Child c in S.Children)
        {
            Dictionary<string, object?> o = Collect(c, S.ChildTimeoutMs);
            List<Rec> recs = o.TryGetValue("recs", out object? rv) && rv is JsonElement je ? Decode(je) : new List<Rec>();
            o.Remove("recs");
            byRole[c.Role] = recs;
            o["records"] = recs.Count;
            o["vs_parent"] = Compare(mine, recs);
            kids.Add(o);
        }
        if (byRole.TryGetValue("rerand1", out List<Rec>? r1) && byRole.TryGetValue("rerand2", out List<Rec>? r2))
        {
            parent["rerand1_vs_rerand2"] = Compare(r1, r2);
        }
        if (byRole.TryGetValue("leakB", out List<Rec>? lb))
        {
            if (byRole.TryGetValue("leakA", out List<Rec>? la))
            {
                parent["leakA_vs_leakB"] = Compare(la, lb);
            }
            if (byRole.TryGetValue("leakC", out List<Rec>? lc))
            {
                parent["leakC_vs_leakB"] = Compare(lc, lb);
            }
        }
        parent["children"] = kids;
        S.ForkPoints.Add(new Dictionary<string, object?> { ["fidelity"] = parent });
        S.Children.Clear();
    }

    private static unsafe byte[]? ReadAll(int fd, int timeoutMs)
    {
        MemoryStream ms = new MemoryStream();
        byte[] buf = new byte[1 << 16];
        while (true)
        {
            Libc.PollFd p = new Libc.PollFd { fd = fd, events = 1 };
            int pr = Libc.poll(ref p, 1, timeoutMs);
            if (pr == 0)
            {
                return null;
            }
            if (pr < 0)
            {
                if (Libc.Errno == Libc.EINTR)
                {
                    continue;
                }
                throw new InvalidOperationException("poll failed: errno " + Libc.Errno);
            }
            nint n;
            fixed (byte* b = buf)
            {
                n = Libc.read(fd, b, buf.Length);
            }
            if (n < 0)
            {
                if (Libc.Errno == Libc.EINTR)
                {
                    continue;
                }
                throw new InvalidOperationException("read failed: errno " + Libc.Errno);
            }
            if (n == 0)
            {
                return ms.ToArray();
            }
            ms.Write(buf, 0, (int)n);
        }
    }

    private static Dictionary<string, object?> Collect(Child c, int timeoutMs)
    {
        byte[]? b = ReadAll(c.Fd, timeoutMs);
        if (b == null)
        {
            Libc.kill(c.Pid, 9);
        }
        int w;
        int status;
        while ((w = Libc.waitpid(c.Pid, out status, 0)) < 0 && Libc.Errno == Libc.EINTR)
        {
        }
        if (w != c.Pid)
        {
            throw new InvalidOperationException("waitpid(" + c.Pid + ") returned " + w + " errno " + Libc.Errno);
        }
        Libc.close(c.Fd);
        Dictionary<string, object?> o = b != null && b.Length > 0
            ? JsonSerializer.Deserialize<Dictionary<string, object?>>(b)!
            : new Dictionary<string, object?> { ["role"] = c.Role, ["missing"] = true };
        o["hang_killed"] = b == null;
        o["exit_status"] = status;
        o["fork_ms"] = Math.Round(c.ForkMs, 3);
        o["wall_ms_fork_to_exit"] = Math.Round(Ms(c.T0, Stopwatch.GetTimestamp()), 2);
        return o;
    }

    private static object Compare(List<Rec> a, List<Rec> b)
    {
        int n = Math.Min(a.Count, b.Count);
        int firstRun = -1, firstChk = -1, firstDeep = -1, firstFrame = -1, firstChoice = -1;
        for (int i = 0; i < n; i++)
        {
            if (firstRun < 0 && a[i].RunSig != b[i].RunSig) firstRun = i;
            if (firstChk < 0 && a[i].Checksum != b[i].Checksum) firstChk = i;
            if (firstDeep < 0 && a[i].Deep != b[i].Deep) firstDeep = i;
            if (firstFrame < 0 && a[i].Frame != b[i].Frame) firstFrame = i;
            if (firstChoice < 0 && (a[i].NActions != b[i].NActions || a[i].Chosen != b[i].Chosen || a[i].Kind != b[i].Kind)) firstChoice = i;
        }
        bool identical = a.Count == b.Count && firstRun < 0 && firstChk < 0 && firstDeep < 0 && firstFrame < 0 && firstChoice < 0;
        return new
        {
            identical,
            len_a = a.Count,
            len_b = b.Count,
            first_runsig_diff = firstRun,
            first_checksum_diff = firstChk,
            first_deep_diff = firstDeep,
            first_frame_diff = firstFrame,
            first_choice_diff = firstChoice,
        };
    }

    private static List<Rec> Decode(JsonElement e)
    {
        List<Rec> l = new List<Rec>();
        foreach (JsonElement x in e.EnumerateArray())
        {
            l.Add(new Rec
            {
                Dp = x[0].GetInt64(),
                Kind = x[1].GetInt32(),
                NActions = x[2].GetInt32(),
                Chosen = x[3].GetInt32(),
                Auto = x[4].GetBoolean(),
                RunSig = ulong.Parse(x[5].GetString()!),
                Checksum = x[6].GetUInt32(),
                Deep = ulong.Parse(x[7].GetString()!),
                Frame = ulong.Parse(x[8].GetString()!),
            });
        }
        return l;
    }

    /// <summary>A child forks its own child (nested worlds): both continue with the same policy state.</summary>
    private static void ForkGrandchild()
    {
        int[] fds = new int[2];
        if (Libc.pipe(fds) != 0)
        {
            throw new InvalidOperationException("pipe() failed: errno " + Libc.Errno);
        }
        long t0 = Stopwatch.GetTimestamp();
        int pid = ForkPrimitive.Fork(fds[1]);
        if (pid == 0)
        {
            S.Role = "nested-gc";
            S.WriteFd = fds[1];
            S.NestedChild = null;
            S.ChildInfo = new Dictionary<string, object?> { ["role"] = "nested-gc", ["pid"] = Libc.getpid() };
            return;
        }
        Libc.close(fds[1]);
        S.NestedChild = new Child { Pid = pid, Fd = fds[0], Role = "nested-gc", ForkMs = Ms(t0, Stopwatch.GetTimestamp()), T0 = t0 };
    }

    private static void ChildFinish(RunContext? ctx, EpisodeResult? r)
    {
        Dictionary<string, object?> o = S.ChildInfo;
        o["playout_ms"] = Math.Round(Ms(S.ForkT0, Stopwatch.GetTimestamp()), 3);
        if (ctx != null)
        {
            foreach (KeyValuePair<string, object?> kv in FightSummary(ctx))
            {
                o[kv.Key] = kv.Value;
            }
            o["decision_points"] = ctx.Counters.DecisionPoints - S.DecisionsAtFork;
            o["end"] = "fight_over";
        }
        else
        {
            if (S.Terminal != null)
            {
                foreach (KeyValuePair<string, object?> kv in S.Terminal)
                {
                    o[kv.Key] = kv.Value;
                }
            }
            o["decision_points"] = r!.Counters.DecisionPoints - S.DecisionsAtFork;
            o["end"] = "episode_end";
        }
        StrictCounters st = S.Session.ReadStrict() - S.StrictAtFork;
        o["strict_violation"] = st.FirstViolation();
        o["game_errors_logged"] = st.GameErrorsLogged;
        o["depth"] = ForkPrimitive.Depth;
        if (S.Recs != null)
        {
            o["recs"] = S.Recs.Select(x => new object[] { x.Dp, x.Kind, x.NActions, x.Chosen, x.Auto, x.RunSig.ToString(), x.Checksum, x.Deep.ToString(), x.Frame.ToString() }).ToList();
        }
        if (S.NestedChild != null)
        {
            Dictionary<string, object?> g = Collect(S.NestedChild, S.ChildTimeoutMs);
            List<Rec> grec = g.TryGetValue("recs", out object? gv) && gv is JsonElement gje ? Decode(gje) : new List<Rec>();
            o["nested_vs_child"] = Compare(S.Recs!, grec);
            o["nested_exit_status"] = g["exit_status"];
            o["nested_hang_killed"] = g["hang_killed"];
            o["nested_strict_violation"] = g.TryGetValue("strict_violation", out object? gs) ? gs?.ToString() : "missing";
        }
        int e = Libc.WriteAll(S.WriteFd, JsonSerializer.SerializeToUtf8Bytes(o));
        if (e != 0)
        {
            ForkPrimitive.ChildFatal(12, "result pipe write failed: errno " + e);
        }
        Libc.close(S.WriteFd);
        Libc._exit(0);
    }

    internal static double Ms(long t0, long t1) => Math.Round((t1 - t0) * 1000.0 / Stopwatch.Frequency, 4);
}
