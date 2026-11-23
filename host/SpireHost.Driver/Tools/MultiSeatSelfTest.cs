using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Unlocks;
using MegaCrit.Sts2.Core.ValueProps;
using SpireHost.Boot;
using SpireHost.Patches;
using SpireHost.Sim;

namespace SpireHost.Driver.Tools;

/// <summary>
/// --selftest-mp: the multi-seat self-test of milestone M1 ("2-4 seats in one fight"; notes\h5-plan.md section 2).
/// Every case runs a real multi-seat run start (RunManager.SetUpTest on the host's Host-type net service) and then
/// jumps into a fixed encounter with the game's debug facility (ScenarioTools.JumpToEncounter), exactly like the
/// single-seat module tests; the fight is played by a policy over the driver's ordinary decisions.
///
///   --cases start,fight,idle,death,wipe,prompts,scaling,crossn,determinism,mixed   (M1)
///           rest,treasure,shop,crystal,bundle,shared,runs                            (M2: MultiSeatRoomTests; default: all of both)
///   --seats 2,3,4         seat counts for the per-N cases (default 2,3,4)
///   --episodes N          episodes per (N, encounter, HP mode) cell of fight / determinism (default 6)
///   --encounters A,B      override the encounter list of fight / scaling
///   --shard I --shards K  run only the cells with index % K == I (parallel gate runs)
///   --wire                the wire round-trip verification mode for every episode
///   --verbose
///
/// What each case checks is stated in its own summary; "ok" in the result is the conjunction of all checks.
/// The read-only probes the test installs (ledger section 2, tool-only: mp-test-*) record what the game's own scaling
/// hooks returned; the expected values are computed here from the decompiled formulas with the seat count and act the
/// TEST chose, never from the game's own state.
/// </summary>
public static class MultiSeatSelfTest
{
    public const string DefaultEncounters = "NIBBITS_NORMAL,CULTISTS_NORMAL,CUBEX_CONSTRUCT_NORMAL,INKLETS_NORMAL,BYRDONIS_ELITE,PHANTASMAL_GARDENERS_ELITE,CHOMPERS_NORMAL@1,LOUSE_PROGENITOR_NORMAL@1,KAISER_CRAB_BOSS@1,FROG_KNIGHT_NORMAL@2,QUEEN_BOSS@2";

    // =====================================================================================================
    //  Read-only scaling probes (tool-only instrumentation)
    // =====================================================================================================

    private struct PowerSample
    {
        public string Power;
        public decimal Amount;
        public decimal Result;
    }

    private struct BlockSample
    {
        public bool EnemyTarget;
        public bool PoweredMove;
        public decimal Result;
    }

    private static bool _probesInstalled;
    private static readonly List<PowerSample> _powerSamples = new List<PowerSample>();
    private static readonly List<BlockSample> _blockSamples = new List<BlockSample>();

    private static void InstallProbes()
    {
        if (_probesInstalled)
        {
            return;
        }
        _probesInstalled = true;
        Type[] sig = { typeof(ICombatState), typeof(Creature), typeof(decimal), typeof(Creature), typeof(CardModel) };
        foreach (Type t in new[] { typeof(PowerModel), typeof(ArtifactPower), typeof(PlatingPower), typeof(SkittishPower), typeof(SlipperyPower), typeof(BufferPower) })
        {
            PatchRegistry.Postfix("mp-test-scale-power:" + t.Name, AccessTools.DeclaredMethod(t, nameof(PowerModel.GetScaledAmountForMultiplayer), sig),
                PatchRegistry.M(typeof(MultiSeatSelfTest), nameof(PowerScalePostfix)), PatchCategory.Instrumentation,
                "TOOL ONLY (--selftest-mp). Read-only: records (power type, amount in, amount out) of " + t.Name + ".GetScaledAmountForMultiplayer (the multiplayer scaling of an enemy power, " +
                "Commands/PowerCmd.cs:88-91); the self-test compares them with the decompiled formula for the seat count and act it chose. Arguments and result untouched.");
        }
        PatchRegistry.Postfix("mp-test-scale-block", AccessTools.Method(typeof(MultiplayerScalingModel), nameof(MultiplayerScalingModel.ModifyBlockMultiplicative)),
            PatchRegistry.M(typeof(MultiSeatSelfTest), nameof(BlockScalePostfix)), PatchCategory.Instrumentation,
            "TOOL ONLY (--selftest-mp). Read-only: records (target is a primary / secondary enemy, powered card or monster-move block, multiplier out) of " +
            "MultiplayerScalingModel.ModifyBlockMultiplicative (Models/MultiplayerScalingModel.cs:11-27); the self-test compares the multiplier with the decompiled formula. " +
            "Arguments and result untouched.");
    }

    public static void PowerScalePostfix(PowerModel __instance, decimal amount, decimal __result)
    {
        _powerSamples.Add(new PowerSample { Power = __instance.GetType().Name, Amount = amount, Result = __result });
    }

    public static void BlockScalePostfix(Creature target, ValueProp props, decimal __result)
    {
        _blockSamples.Add(new BlockSample
        {
            EnemyTarget = target == null || target.IsPrimaryEnemy || target.IsSecondaryEnemy,
            PoweredMove = props.IsPoweredCardOrMonsterMoveBlock(),
            Result = __result,
        });
    }

    /// <summary>Models/MultiplayerScalingModel.cs:29-46.</summary>
    private static decimal Scaling(int actIndex, bool boss) => actIndex == 0 ? 1.1m : actIndex == 1 ? 1.2m : boss ? 1.3m : 1.2m;

    /// <summary>The decompiled formula of each scaled power (PowerModel.cs:426-429 and the four overrides); null = a type the test does not know.</summary>
    private static decimal? ExpectedPower(string type, decimal amount, int n, decimal s)
    {
        switch (type)
        {
            case nameof(ArtifactPower): return amount + n - 1m;                       // ArtifactPower.cs:43-46
            case nameof(PlatingPower): return ((n - 1) * 2 + 1) * amount;             // PlatingPower.cs:79-82
            case nameof(SkittishPower): return amount * (1m + (n - 1) * 0.5m);        // SkittishPower.cs:80-83
            case nameof(SlipperyPower): return amount * n;                            // SlipperyPower.cs:41-44
            case nameof(BufferPower): return null;                                    // never reached: BufferPower does not set ShouldScaleInMultiplayer
            default: return amount * n * s;                                            // PowerModel.cs:426-429
        }
    }

    private static readonly string[] ScaledPowers =
    {
        "ArtifactPower", "CurlUpPower", "FlutterPower", "HardenedShellPower", "PlatingPower", "PlowPower", "RampartPower", "ReattachPower", "RegenPower", "ShriekPower", "SkittishPower", "SlipperyPower",
    };

    // =====================================================================================================
    //  The combat audit: an observer that re-checks what the driver offered, with code of its own
    // =====================================================================================================

    private sealed class CombatAudit : IEpisodeObserver
    {
        public readonly List<string> Errors = new List<string>();
        public long CombatDecisions;
        public long UnionDecisions;
        public long ActionsChecked;
        public long Handshakes;
        public long ReadyActionsExecuted;
        public long SelectDecisions;
        public long RemoteSelectDecisions;
        public long DeadSeatChecks;
        public long UndoOffered;
        public long UndoChosen;
        public long AllyTargetsOffered;
        public readonly long[] ChosenBySeat = new long[4];
        public readonly bool[] SeatDied = new bool[4];
        public readonly List<(string Id, int MaxHp, int? Unscaled)> FirstEnemies = new List<(string, int, int?)>();
        public readonly List<string> Seat0FirstHand = new List<string>();
        public bool SawCombat;
        public int FirstDecisionActIndex = -1;
        public bool Boss;
        public int EnemyTurns;

        private RunContext? _ctx;
        private Action<CombatState>? _switch;
        private Action<GameAction>? _executed;
        private readonly HashSet<ulong> _readyOwners = new HashSet<ulong>();
        private int _readyThisTurn;

        private void Error(string what)
        {
            if (Errors.Count < 20)
            {
                Errors.Add(what);
            }
        }

        public void OnEpisodeStart(RunContext ctx)
        {
            _ctx = ctx;
            _switch = OnAboutToSwitch;
            _executed = OnActionExecuted;
            ctx.CombatManager.AboutToSwitchToEnemyTurn += _switch;
            ctx.RunManager.ActionExecutor.BeforeActionExecuted += _executed; // the last ready action starts the enemy turn from inside its own execution
        }

        private void Unsubscribe()
        {
            RunContext? ctx = _ctx;
            if (ctx == null)
            {
                return;
            }
            if (_switch != null)
            {
                ctx.CombatManager.AboutToSwitchToEnemyTurn -= _switch;
            }
            if (_executed != null && ctx.RunManager.IsInProgress)
            {
                ctx.RunManager.ActionExecutor.BeforeActionExecuted -= _executed;
            }
            _ctx = null;
        }

        private void OnActionExecuted(GameAction action)
        {
            if (action is ReadyToBeginEnemyTurnAction)
            {
                ReadyActionsExecuted++;
                _readyThisTurn++;
                _readyOwners.Add(action.OwnerId);
            }
        }

        /// <summary>CombatManager.AfterAllPlayersReadyToBeginEnemyTurn (:1430-1453): the enemy turn is about to start.</summary>
        private void OnAboutToSwitch(CombatState state)
        {
            RunContext? ctx = _ctx;
            if (ctx == null)
            {
                return;
            }
            Handshakes++;
            EnemyTurns++;
            int n = state.Players.Count;
            CombatTurnState? turn = ctx.CombatManager._turnState;
            if (turn == null)
            {
                Error("handshake: no turn state");
                return;
            }
            if (_readyThisTurn != n || _readyOwners.Count != n)
            {
                Error($"handshake: {_readyThisTurn} ReadyToBeginEnemyTurn actions of {_readyOwners.Count} owners executed before the enemy turn, expected {n} of {n}");
            }
            if (turn.PlayersReadyToBeginEnemyTurn.Count != n)
            {
                Error($"handshake: {turn.PlayersReadyToBeginEnemyTurn.Count} of {n} players ready to begin the enemy turn");
            }
            // An extra turn is taken by some players only; the others were set ready at its start (CombatManager.cs:585-596).
            if (turn.PlayersReadyToEndTurn.Count != n)
            {
                Error($"handshake: {turn.PlayersReadyToEndTurn.Count} of {n} players had ended their turn when the enemy turn started");
            }
            _readyThisTurn = 0;
            _readyOwners.Clear();
        }

        public void OnDecision(RunContext ctx, Decision d, int chosen, bool auto)
        {
            CombatManager cm = ctx.CombatManager;
            LegalAction pick = d.Actions[chosen];
            if (pick.Seat >= 0 && pick.Seat < 4)
            {
                ChosenBySeat[pick.Seat]++;
            }
            if (d.Kind == DecisionKind.CardSelect)
            {
                SelectDecisions++;
                if (ctx.IsRemoteSeat(d.ActingSeat))
                {
                    RemoteSelectDecisions++;
                }
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    if (d.Actions[i].Seat != d.ActingSeat)
                    {
                        Error("card select: an action of seat " + d.Actions[i].Seat + " on the prompt of seat " + d.ActingSeat);
                    }
                }
            }
            if (!cm.IsInProgress)
            {
                return;
            }
            CombatState? state = cm.DebugOnlyGetState();
            CombatTurnState? turn = cm._turnState;
            if (state == null || turn == null)
            {
                return;
            }
            IReadOnlyList<Player> seats = ctx.Seats;
            for (int s = 0; s < seats.Count; s++)
            {
                Player p = seats[s];
                if (p.Creature.IsAlive)
                {
                    continue;
                }
                SeatDied[s] = true;
                DeadSeatChecks++;
                PlayerCombatState? pcs = p.PlayerCombatState;
                bool somebodyAlive = seats.Any(q => q.Creature.IsAlive);
                // CombatManager.HandlePlayerDeath (:957-970): with a teammate alive, the dead player's cards leave the combat
                // (unless a card of its own is still resolving) and its energy and stars are set to 0.
                if (pcs != null && somebodyAlive && !cm.IsExecutingCardOrPotionEffect(p))
                {
                    int cards = pcs.Hand.Cards.Count + pcs.DrawPile.Cards.Count + pcs.DiscardPile.Cards.Count + pcs.ExhaustPile.Cards.Count + pcs.PlayPile.Cards.Count;
                    if (cards != 0 || pcs.Energy != 0)
                    {
                        Error($"dead seat {s}: {cards} cards in its combat piles, energy {pcs.Energy}");
                    }
                }
            }
            if (d.Kind != DecisionKind.Combat)
            {
                return;
            }
            CombatDecisions++;
            if (d.ActingSeat < 0)
            {
                UnionDecisions++;
            }
            if (!SawCombat)
            {
                SawCombat = true;
                FirstDecisionActIndex = ctx.Run.CurrentActIndex;
                Boss = state.Encounter?.RoomType == RoomType.Boss;
                foreach (Creature e in state._enemies)
                {
                    FirstEnemies.Add((e.Monster?.Id.Entry ?? "?", e.MaxHp, e.MonsterMaxHpBeforeModification));
                }
                PlayerCombatState? pcs0 = seats[0].PlayerCombatState;
                if (pcs0 != null)
                {
                    foreach (CardModel c in pcs0.Hand.Cards)
                    {
                        Seat0FirstHand.Add(c.Id.Entry + "+" + c.CurrentUpgradeLevel);
                    }
                }
            }
            bool allReady = turn.PlayersReadyToEndTurn.Count == state.Players.Count;
            for (int i = 0; i < d.Actions.Count; i++)
            {
                LegalAction a = d.Actions[i];
                ActionsChecked++;
                if (a.Seat < 0 || a.Seat >= seats.Count)
                {
                    Error("combat: action with seat " + a.Seat);
                    continue;
                }
                Player p = seats[a.Seat];
                bool ended = turn.PlayersReadyToEndTurn.Contains(p);
                if (!p.Creature.IsAlive)
                {
                    Error($"combat: {a.Verb} offered to dead seat {a.Seat}");
                }
                if (a.Verb == Verb.UndoEndTurn)
                {
                    UndoOffered++;
                    // NEndTurnButton.cs:299-318 / CombatManager.CanUnEndTurn (:751-773)
                    if (!ended || allReady || !cm.CanUnEndTurn(p))
                    {
                        Error($"combat: UNDO_END_TURN offered to seat {a.Seat} (ended {ended}, everybody ready {allReady}, can un-end {cm.CanUnEndTurn(p)})");
                    }
                    continue;
                }
                // A seat that ended its turn has its hand, belt and end-turn button disabled (PlayerCmd.cs:215-218, the UI only).
                if (ended)
                {
                    Error($"combat: {a.Verb} offered to seat {a.Seat}, which has ended its turn");
                }
                if (p.PlayerCombatState == null || p.PlayerCombatState.Phase != PlayerTurnPhase.Play)
                {
                    Error($"combat: {a.Verb} offered to seat {a.Seat} outside its play phase");
                }
                if (a.Verb == Verb.PlayCard)
                {
                    CardModel card = a.Source.Card;
                    if (!ReferenceEquals(card.Owner, p))
                    {
                        Error($"combat: seat {a.Seat} is offered a card of player {card.Owner.NetId}");
                    }
                    if (card.TargetType == TargetType.AnyAlly)
                    {
                        AllyTargetsOffered++;
                        Creature? t = a.Target.IsNone ? null : a.Target.Creature;
                        // NTargetManager.cs:434-441: another living player, never yourself, never a pet.
                        if (t == null || ReferenceEquals(t, p.Creature) || !t.IsPlayer || !t.IsAlive)
                        {
                            Error($"combat: AnyAlly card {card.Id.Entry} of seat {a.Seat} offered with target {(t == null ? "none" : ReferenceEquals(t, p.Creature) ? "itself" : t.IsPlayer ? "a dead player" : "a pet")}");
                        }
                    }
                    else if (card.TargetType == TargetType.AnyEnemy)
                    {
                        Creature? t = a.Target.IsNone ? null : a.Target.Creature;
                        if (t == null || t.Side == p.Creature.Side || !t.IsAlive)
                        {
                            Error($"combat: AnyEnemy card {card.Id.Entry} of seat {a.Seat} offered with a bad target");
                        }
                    }
                    else if (!a.Target.IsNone)
                    {
                        Error($"combat: untargeted card {card.Id.Entry} offered with a target");
                    }
                }
                else if (a.Verb == Verb.UsePotion || a.Verb == Verb.DiscardPotion)
                {
                    if (!ReferenceEquals(a.Source.PotionOwner, p))
                    {
                        Error($"combat: seat {a.Seat} is offered a potion slot of another player");
                    }
                    PotionModel? potion = p.GetPotionAtSlotIndex(a.Source.Index);
                    if (potion == null || ctx.IsPotionQueued(potion))
                    {
                        Error($"combat: {a.Verb} offered on an empty or queued potion slot of seat {a.Seat}");
                    }
                }
            }
            if (pick.Verb == Verb.UndoEndTurn)
            {
                UndoChosen++;
            }
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
            Unsubscribe();
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }
    }

    /// <summary>Seat 0 plays at random; every other seat only ends its turn and answers a prompt with its first option.</summary>
    private sealed class IdleTeammatesPolicy : IPolicy
    {
        private readonly PolicyRng _rng;
        private readonly List<int> _own = new List<int>(32);

        public IdleTeammatesPolicy(ulong seed)
        {
            _rng = new PolicyRng(seed);
        }

        public int Decide(Decision d)
        {
            if (d.Kind != DecisionKind.Combat)
            {
                return d.ActingSeat > 0 ? 0 : _rng.NextInt(d.Actions.Count);
            }
            for (int i = 0; i < d.Actions.Count; i++)
            {
                if (d.Actions[i].Seat > 0 && d.Actions[i].Verb == Verb.EndTurn)
                {
                    return i;
                }
            }
            _own.Clear();
            for (int i = 0; i < d.Actions.Count; i++)
            {
                if (d.Actions[i].Seat == 0 && d.Actions[i].Verb != Verb.UndoEndTurn)
                {
                    _own.Add(i);
                }
            }
            if (_own.Count == 0)
            {
                // only UNDO actions are left: seat 0 ended its turn and a teammate has not (cannot happen: teammates end first)
                return 0;
            }
            return _own[_rng.NextInt(_own.Count)];
        }
    }

    /// <summary>Plays the listed cards whenever one is playable (preferring a remote target), answers prompts at random, otherwise ends the turn.</summary>
    private sealed class PromptCardsPolicy : IPolicy
    {
        private readonly PolicyRng _rng;
        private readonly HashSet<string> _cards;
        private readonly List<int> _picks = new List<int>(16);
        public long PromptCardPlays;

        public PromptCardsPolicy(ulong seed, IEnumerable<string> cards)
        {
            _rng = new PolicyRng(seed);
            _cards = new HashSet<string>(cards, StringComparer.Ordinal);
        }

        public int Decide(Decision d)
        {
            if (d.Kind != DecisionKind.Combat)
            {
                return _rng.NextInt(d.Actions.Count);
            }
            _picks.Clear();
            for (int i = 0; i < d.Actions.Count; i++)
            {
                LegalAction a = d.Actions[i];
                if (a.Verb == Verb.PlayCard && _cards.Contains(a.Source.Card.Id.Entry))
                {
                    _picks.Add(i);
                }
            }
            if (_picks.Count > 0)
            {
                PromptCardPlays++;
                return _picks[_rng.NextInt(_picks.Count)];
            }
            for (int i = 0; i < d.Actions.Count; i++)
            {
                if (d.Actions[i].Verb == Verb.EndTurn)
                {
                    return i;
                }
            }
            return _rng.NextInt(d.Actions.Count);
        }
    }

    // =====================================================================================================
    //  Episodes
    // =====================================================================================================

    private static string[] Lineup(int seats, int index)
    {
        string[] all = DriverCommands.AllCharacters;
        string[] r = new string[seats];
        for (int k = 0; k < seats; k++)
        {
            // index 0 of every seat count repeats a character on purpose (seats 0 and 1 of an even index play the same one)
            r[k] = all[(index + (index % 2 == 0 && k > 0 ? k - 1 : k)) % all.Length];
        }
        return r;
    }

    private static (string Id, int Act) ParseEncounter(string spec)
    {
        int at = spec.IndexOf('@');
        return at < 0 ? (spec, 0) : (spec.Substring(0, at), int.Parse(spec.Substring(at + 1)));
    }

    /// <summary>Scenario set-up: (act change,) jump into the encounter. The episode stops when that combat is over (won), or the run ends.</summary>
    private static Scenario FightScenario(string name, string encounter, int act, Func<RunContext, Task>? afterJump = null, Func<RunContext, bool>? stopEarly = null)
    {
        bool sawCombat = false;
        return new Scenario
        {
            Name = name,
            Setup = async ctx =>
            {
                if (act > 0)
                {
                    await ScenarioTools.EnterAct(ctx, act, doTransition: false);
                }
                await ScenarioTools.JumpToEncounter(ctx, encounter);
                if (afterJump != null)
                {
                    await ctx.Quiescence();
                    await afterJump(ctx);
                }
            },
            StopWhen = ctx =>
            {
                if (ctx.CombatManager.IsInProgress)
                {
                    sawCombat = true;
                    return stopEarly != null && stopEarly(ctx);
                }
                return sawCombat;
            },
        };
    }

    private static string Describe(EpisodeResult r)
    {
        return r.Outcome == EpisodeOutcome.Aborted ? "Aborted(" + r.Abort!.Signature + " | " + r.Abort.Message + ")"
            : r.Outcome == EpisodeOutcome.Truncated ? "Truncated(" + r.TruncatedBy + ")"
            : r.Outcome.ToString();
    }

    private static bool CleanCounters(EpisodeResult r, List<string> problems)
    {
        bool ok = true;
        foreach (KeyValuePair<string, long> kv in r.ModuleCounters)
        {
            if (kv.Value != 0 && kv.Key.Contains("violation", StringComparison.Ordinal))
            {
                problems.Add(kv.Key + " = " + kv.Value);
                ok = false;
            }
        }
        string? strict = r.Strict.FirstViolation();
        if (strict != null)
        {
            problems.Add("strict: " + strict);
            ok = false;
        }
        if (r.TeardownFaults != 0)
        {
            problems.Add("teardown: " + r.TeardownFault);
            ok = false;
        }
        return ok;
    }

    private static long Counter(EpisodeResult r, string key) => r.ModuleCounters.TryGetValue(key, out long v) ? v : 0;

    // =====================================================================================================
    //  The command
    // =====================================================================================================

    public static int Run(HostSession session, Args a)
    {
        LogBootstrap.ShowErrors = a.Has("verbose");
        bool verbose = a.Has("verbose");
        HashSet<string> cases = new HashSet<string>(a.Get("cases", "start,fight,idle,death,wipe,prompts,scaling,crossn,determinism,mixed," + string.Join(",", MultiSeatRoomTests.Cases))
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries), StringComparer.Ordinal);
        int[] seatCounts = a.Get("seats", "2,3,4").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        int episodes = a.GetInt("episodes", 6);
        int shard = a.GetInt("shard", 0);
        int shards = Math.Max(1, a.GetInt("shards", 1));
        bool wire = a.GetBool("wire", false);
        (string Id, int Act)[] encounters = a.Get("encounters", DefaultEncounters).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(ParseEncounter).ToArray();
        InstallProbes();

        Dictionary<string, object> results = new Dictionary<string, object>(StringComparer.Ordinal);
        bool all = true;
        int cell = 0;
        bool Mine() => cell++ % shards == shard;

        // ------------------------------------------------------------------------------------------------ start
        if (cases.Contains("start"))
        {
            List<object> rows = new List<object>();
            bool ok = true;
            foreach (int n in seatCounts)
            {
                for (int i = 0; i < 2; i++)
                {
                    string[] lineup = Lineup(n, i);
                    List<string> problems = new List<string>();
                    long syncBefore = MultiSeatStats.SyncRounds, rebuiltBefore = MultiSeatStats.SyncPlayersRebuilt;
                    string seed = HostSession.SeedFromLabel("MPSTART:" + n + ":" + i);
                    EpisodeSpec spec = new EpisodeSpec
                    {
                        Seed = seed,
                        Characters = lineup,
                        Ascension = i == 0 ? 0 : 10,
                        Scenario = new Scenario
                        {
                            Name = "mp-start",
                            Setup = ctx =>
                            {
                                RunManager rm = ctx.RunManager;
                                RunState run = ctx.Run;
                                if (ctx.Mp == null || !ReferenceEquals(rm.NetService, ctx.Mp.Net) || rm.NetService.Type != NetGameType.Host) problems.Add("the run is not on the host's Host-type service");
                                if (rm.IsSingleplayerOrFakeMultiplayer) problems.Add("IsSingleplayerOrFakeMultiplayer");
                                if (run.Players.Count != n) problems.Add("players " + run.Players.Count);
                                for (int s = 0; s < run.Players.Count; s++)
                                {
                                    if (run.Players[s].NetId != MultiSeatRun.NetIdOfSeat(s)) problems.Add("seat " + s + " net id " + run.Players[s].NetId);
                                    if (!run.Players[s].Character.Id.Entry.Equals(lineup[s], StringComparison.OrdinalIgnoreCase)) problems.Add("seat " + s + " plays " + run.Players[s].Character.Id.Entry);
                                }
                                if (LocalContext.NetId != 1UL || !ReferenceEquals(LocalContext.GetMe(run), run.Players[0])) problems.Add("the local player is not seat 0");
                                if (rm.RunLobby == null || !rm.RunLobby.PlayerIds.SequenceEqual(run.Players.Select(p => p.NetId))) problems.Add("run lobby ids");
                                if (rm.CombatStateSynchronizer.IsDisabled) problems.Add("combat-state sync disabled");
                                if (run.Modifiers.Count != 0) problems.Add("modifiers");
                                if (run.AscensionLevel != (i == 0 ? 0 : 10)) problems.Add("ascension " + run.AscensionLevel);
                                if (ctx.Mp != null && ctx.Mp.Clients.Count(c => c != null) != n - 1) problems.Add("virtual clients");
                                // the act list of a multiplayer lobby: the same draw with isMultiplayer: true (StartRunLobby.cs:492-494)
                                UnlockState[] unlocks = Enumerable.Range(0, n).Select(_ => VeteranProfile.CreateUnlockState()).ToArray();
                                List<string> acts = ActModel.GetRandomList(new Rng(MegaCrit.Sts2.Core.Helpers.StringHelper.GetDeterministicHashCode(ctx.Seed), "act_selection"), new UnlockState(unlocks), isMultiplayer: true).Select(x => x.Id.Entry).ToList();
                                if (!acts.SequenceEqual(run.Acts.Select(x => x.Id.Entry))) problems.Add("act list " + string.Join(",", run.Acts.Select(x => x.Id.Entry)) + " != " + string.Join(",", acts));
                                if (!(run.CurrentRoom is EventRoom)) problems.Add("first room " + run.CurrentRoom?.GetType().Name);
                                return Task.CompletedTask;
                            },
                            StopWhen = _ => true,
                        },
                    };
                    EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy(1), new EpisodeOptions { WireRoundTrip = wire });
                    if (!(r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy == "scenario")) problems.Add(Describe(r));
                    if (MultiSeatStats.SyncRounds - syncBefore < 1 || MultiSeatStats.SyncPlayersRebuilt - rebuiltBefore != (MultiSeatStats.SyncRounds - syncBefore) * (n - 1)) problems.Add("combat-state sync rounds " + (MultiSeatStats.SyncRounds - syncBefore) + ", players rebuilt " + (MultiSeatStats.SyncPlayersRebuilt - rebuiltBefore));
                    CleanCounters(r, problems);
                    if (r.Characters.Count != n) problems.Add("result characters " + r.Characters.Count);
                    ok &= problems.Count == 0;
                    rows.Add(new { seats = n, lineup = string.Join("+", lineup), ascension = r.Ascension, seed, passed = problems.Count == 0, problems });
                }
            }
            results["start"] = new { ok, what = "run creation with 2-4 seats (repeated characters): Host-type service, lobby, local seat, act list with isMultiplayer, combat-state sync round at the first room", rows };
            all &= ok;
        }

        // ------------------------------------------------------------------------------------------------ fight / idle
        foreach (string mode in new[] { "fight", "idle" })
        {
            if (!cases.Contains(mode))
            {
                continue;
            }
            List<object> failures = new List<object>();
            SortedDictionary<string, long[]> byCell = new SortedDictionary<string, long[]>(StringComparer.Ordinal);
            long ran = 0, won = 0, lost = 0, capped = 0, aborted = 0, decisions = 0, union = 0, handshakes = 0, undoOffered = 0, undoChosen = 0, checkedActions = 0, remoteSelects = 0, allyTargets = 0, deadChecks = 0;
            long[] chosenBySeat = new long[4];
            SortedDictionary<string, long> counters = new SortedDictionary<string, long>(StringComparer.Ordinal);
            double ms = 0;
            long engineTicks = 0, decisionPoints = 0;
            foreach (int n in seatCounts)
            {
                foreach ((string enc, int act) in encounters)
                {
                    foreach (int hp in new[] { 0, 99999 })
                    {
                        if (mode == "idle" && hp == 0)
                        {
                            continue; // an idle team at natural HP mostly dies: the point of the case is that the fight still ends
                        }
                        if (!Mine())
                        {
                            continue;
                        }
                        for (int e = 0; e < episodes; e++)
                        {
                            string label = $"MP{mode.ToUpperInvariant()}:{n}:{enc}:{hp}:{e}";
                            CombatAudit audit = new CombatAudit();
                            EpisodeSpec spec = new EpisodeSpec
                            {
                                Seed = HostSession.SeedFromLabel(label),
                                Characters = Lineup(n, e),
                                Ascension = e % 2 == 0 ? 0 : 10,
                                Scenario = FightScenario("mp-" + mode, enc, act),
                            };
                            IPolicy policy = mode == "idle" ? new IdleTeammatesPolicy((ulong)(1000 + e)) : new RandomLegalPolicy((ulong)(7919 * n + e));
                            EpisodeOptions options = new EpisodeOptions { DebugForcedHp = hp, MaxPolicyDecisions = 60_000, MaxCombatTurns = 300, WireRoundTrip = wire };
                            EpisodeResult r = session.RunEpisode(spec, policy, options, audit);
                            ran++;
                            ms += r.WallMs;
                            engineTicks += r.Counters.TicksApply + r.Counters.TicksSettle;
                            decisionPoints += r.Counters.DecisionPoints;
                            DriverCommands.AddModuleCounters(counters, r.ModuleCounters);
                            List<string> problems = new List<string>(audit.Errors);
                            CleanCounters(r, problems);
                            bool endedWon = r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy == "scenario";
                            bool endedLost = r.Outcome == EpisodeOutcome.Loss;
                            bool turnCap = r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy == "combat_turn_cap";
                            if (r.Outcome == EpisodeOutcome.Aborted)
                            {
                                aborted++;
                                problems.Add(Describe(r) + " " + r.Abort!.Diagnostics);
                            }
                            else if (endedWon)
                            {
                                won++;
                                // CombatManager.EndCombatInternal (:1031-1034): every player is alive again after a won fight.
                                for (int s = 0; s < r.FinalHps.Count; s++)
                                {
                                    if (r.FinalHps[s] < 1) problems.Add($"seat {s} has {r.FinalHps[s]} HP after a won fight");
                                }
                            }
                            else if (endedLost)
                            {
                                lost++;
                                if (r.FinalHps.Any(h => h > 0)) problems.Add("Loss with a living seat: " + string.Join("/", r.FinalHps));
                            }
                            else if (turnCap)
                            {
                                capped++;
                            }
                            else
                            {
                                problems.Add("unexpected end: " + Describe(r));
                            }
                            if (!audit.SawCombat && r.Outcome != EpisodeOutcome.Aborted) problems.Add("no combat decision was reached");
                            // enemy HP against the decompiled formula (CombatState.cs:403-405, Creature.cs:335-342, :424-431, :648-655)
                            decimal sc = Scaling(act, audit.Boss);
                            foreach ((string id, int maxHp, int? unscaled) in audit.FirstEnemies)
                            {
                                if (!unscaled.HasValue)
                                {
                                    problems.Add("enemy " + id + " has no unscaled HP value");
                                    continue;
                                }
                                int expected = Math.Min((int)(unscaled.Value * (decimal)n * sc), 999999999);
                                if (maxHp != expected) problems.Add($"enemy {id}: MaxHp {maxHp}, expected trunc({unscaled.Value} x {n} x {sc}) = {expected}");
                            }
                            if (audit.FirstDecisionActIndex >= 0 && audit.FirstDecisionActIndex != act) problems.Add("the fight ran in act index " + audit.FirstDecisionActIndex + ", the test asked for " + act);
                            decisions += audit.CombatDecisions;
                            union += audit.UnionDecisions;
                            handshakes += audit.Handshakes;
                            undoOffered += audit.UndoOffered;
                            undoChosen += audit.UndoChosen;
                            checkedActions += audit.ActionsChecked;
                            remoteSelects += audit.RemoteSelectDecisions;
                            allyTargets += audit.AllyTargetsOffered;
                            deadChecks += audit.DeadSeatChecks;
                            for (int s = 0; s < 4; s++) chosenBySeat[s] += audit.ChosenBySeat[s];
                            string key = n + " seats/" + enc + "/hp " + hp;
                            if (!byCell.TryGetValue(key, out long[]? c4)) byCell[key] = c4 = new long[4];
                            c4[endedWon ? 0 : endedLost ? 1 : turnCap ? 2 : 3]++;
                            if (problems.Count > 0 && failures.Count < 30)
                            {
                                failures.Add(new { seats = n, encounter = enc, act, hp, episode = e, seed = r.Seed, lineup = string.Join("+", r.Characters), outcome = Describe(r), problems = problems.Take(8).ToList() });
                            }
                            if (problems.Count > 0) all = false;
                            if (verbose) Console.Error.WriteLine($"[mp {mode}] {key} #{e}: {Describe(r)} decisions {r.Counters.PolicyDecisions} hp {string.Join("/", r.FinalHps)} {r.WallMs:0} ms {(problems.Count > 0 ? "PROBLEMS " + problems[0] : "")}");
                        }
                    }
                }
            }
            // the scaling hooks, against the decompiled formulas - evaluated per case run (samples of all cells of this mode)
            results[mode] = new
            {
                ok = failures.Count == 0,
                what = mode == "fight"
                    ? "a fixed encounter played to its end by a uniform random legal policy over all seats (UNDO on): UI rules re-checked on every COMBAT decision, the ready handshake before every enemy turn, dead seats, enemy HP against the formula, zero violation counters"
                    : "seat 0 random, the other seats only end their turn (forced HP): the fight still ends, the handshake holds",
                episodes = ran,
                won,
                lost,
                turn_cap = capped,
                aborted,
                combat_decisions = decisions,
                union_decisions = union,
                actions_checked = checkedActions,
                enemy_turn_handshakes = handshakes,
                undo_offered = undoOffered,
                undo_chosen = undoChosen,
                any_ally_targets_offered = allyTargets,
                dead_seat_checks = deadChecks,
                remote_card_select_decisions = remoteSelects,
                chosen_by_seat = chosenBySeat,
                by_cell = byCell.ToDictionary(kv => kv.Key, kv => new { won = kv.Value[0], lost = kv.Value[1], turn_cap = kv.Value[2], other = kv.Value[3] }),
                mean_ms = ran == 0 ? 0 : Math.Round(ms / ran, 1),
                engine_microseconds_per_decision_point = decisionPoints == 0 ? 0 : Math.Round(engineTicks * 1e6 / System.Diagnostics.Stopwatch.Frequency / decisionPoints, 1),
                module_counters = counters.Where(kv => kv.Key.StartsWith("mp.", StringComparison.Ordinal) || kv.Key.Contains("remote", StringComparison.Ordinal) || kv.Key.Contains("violation", StringComparison.Ordinal) || kv.Key.Contains("rebuild", StringComparison.Ordinal) || kv.Key.Contains("encoding", StringComparison.Ordinal)).ToDictionary(kv => kv.Key, kv => kv.Value),
                failures,
            };
            all &= failures.Count == 0;
        }

        // ------------------------------------------------------------------------------------------------ death / wipe
        if (cases.Contains("death"))
        {
            List<object> rows = new List<object>();
            bool ok = true;
            foreach (int n in seatCounts)
            {
                for (int e = 0; e < Math.Max(2, episodes / 2); e++)
                {
                    int victim = 1 + e % (n - 1);                   // a remote seat dies
                    if (e % 3 == 2) victim = 0;                     // ... or the local one
                    int[] hpPerSeat = Enumerable.Repeat(99999, n).ToArray();
                    hpPerSeat[victim] = 1;
                    string[] lineup = Lineup(n, e + 1);
                    lineup[victim] = e % 2 == 0 ? "SILENT" : "DEFECT"; // no heal after combat: the revive leaves exactly 1 HP
                    CombatAudit audit = new CombatAudit();
                    EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel($"MPDEATH:{n}:{e}"), Characters = lineup, Scenario = FightScenario("mp-death", "CULTISTS_NORMAL", 0) };
                    EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)(31 + e)), new EpisodeOptions { DebugForcedHpPerSeat = hpPerSeat, MaxPolicyDecisions = 60_000, MaxCombatTurns = 300, WireRoundTrip = wire }, audit);
                    List<string> problems = new List<string>(audit.Errors);
                    CleanCounters(r, problems);
                    if (!(r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy == "scenario")) problems.Add("the fight was not won: " + Describe(r));
                    if (!audit.SeatDied[victim]) problems.Add("seat " + victim + " (1 HP) never died");
                    else if (r.FinalHps.Count == n && r.FinalHps[victim] != 1) problems.Add($"seat {victim} died and has {r.FinalHps[victim]} HP after the won fight, expected 1 (Player.ReviveBeforeCombatEnd)");
                    for (int s = 0; s < n && s < r.FinalHps.Count; s++)
                    {
                        if (s != victim && audit.SeatDied[s]) problems.Add("seat " + s + " (99999 HP) died");
                    }
                    if (audit.DeadSeatChecks == 0) problems.Add("no decision was made while a seat was dead");
                    ok &= problems.Count == 0;
                    rows.Add(new { seats = n, victim, lineup = string.Join("+", lineup), seed = r.Seed, outcome = Describe(r), final_hp = r.FinalHps, dead_seat_checks = audit.DeadSeatChecks, enemy_turns = audit.EnemyTurns, passed = problems.Count == 0, problems });
                }
            }
            results["death"] = new { ok, what = "one seat (remote or local) starts the fight with 1 HP and dies: its cards leave the combat, energy 0, no action is offered to it, the handshake still counts it; after the won fight it has exactly 1 HP", rows };
            all &= ok;
        }
        if (cases.Contains("wipe"))
        {
            List<object> rows = new List<object>();
            bool ok = true;
            foreach (int n in seatCounts)
            {
                CombatAudit audit = new CombatAudit();
                EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel("MPWIPE:" + n), Characters = Lineup(n, 1), Scenario = FightScenario("mp-wipe", "CULTISTS_NORMAL", 0) };
                EpisodeResult r = session.RunEpisode(spec, new IdleTeammatesPolicy(5), new EpisodeOptions { DebugForcedHpPerSeat = Enumerable.Repeat(1, n).ToArray(), MaxPolicyDecisions = 60_000, WireRoundTrip = wire }, audit);
                List<string> problems = new List<string>(audit.Errors);
                CleanCounters(r, problems);
                if (r.Outcome != EpisodeOutcome.Loss) problems.Add("expected Loss (team wipe), got " + Describe(r));
                if (r.FinalHps.Count != n || r.FinalHps.Any(h => h != 0)) problems.Add("final HP " + string.Join("/", r.FinalHps));
                ok &= problems.Count == 0;
                rows.Add(new { seats = n, seed = r.Seed, outcome = Describe(r), final_hp = r.FinalHps, enemy_turns = audit.EnemyTurns, passed = problems.Count == 0, problems });
            }
            results["wipe"] = new { ok, what = "every seat has 1 HP: all die, RunState.IsGameOver, outcome Loss (a single death is not a loss: case 'death')", rows };
            all &= ok;
        }

        // ------------------------------------------------------------------------------------------------ prompts
        if (cases.Contains("prompts"))
        {
            string[] promptCards = { "TUTOR", "WISH", "SURVIVOR", "DISCOVERY", "HEADBUTT", "ARMAMENTS" }; // the hand holds 10 cards: the first five always fit
            List<object> rows = new List<object>();
            bool ok = true;
            long remoteFrames = 0, remoteAnswers = 0, hookEnqueues = 0, resumes = 0, localFrames = 0, allyTargets = 0;
            foreach (int n in seatCounts)
            {
                for (int e = 0; e < Math.Max(2, episodes / 2); e++)
                {
                    CombatAudit audit = new CombatAudit();
                    PromptCardsPolicy policy = new PromptCardsPolicy((ulong)(77 + e), promptCards);
                    int turnsToPlay = 4;
                    async Task GiveCards(RunContext ctx)
                    {
                        for (int s = 0; s < ctx.SeatCount; s++)
                        {
                            await ScenarioTools.SetResources(ctx, 20, 0, s);
                            foreach (string card in promptCards)
                            {
                                if (ctx.Seats[s].PlayerCombatState!.Hand.Cards.Count < 10)
                                {
                                    await ScenarioTools.AddCard(ctx, card, PileType.Hand, seat: s);
                                }
                            }
                        }
                    }
                    EpisodeSpec spec = new EpisodeSpec
                    {
                        Seed = HostSession.SeedFromLabel($"MPPROMPT:{n}:{e}"),
                        Characters = Lineup(n, e),
                        Scenario = FightScenario("mp-prompts", "CULTISTS_NORMAL", 0, GiveCards, ctx => ctx.Seats.Max(p => p.PlayerCombatState?.TurnNumber ?? 0) > turnsToPlay),
                    };
                    EpisodeResult r = session.RunEpisode(spec, policy, new EpisodeOptions { DebugForcedHp = 99999, MaxPolicyDecisions = 20_000, WireRoundTrip = wire }, audit);
                    List<string> problems = new List<string>(audit.Errors);
                    CleanCounters(r, problems);
                    if (r.Outcome == EpisodeOutcome.Aborted || (r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy != "scenario")) problems.Add(Describe(r) + " " + (r.Abort?.Diagnostics ?? ""));
                    if (policy.PromptCardPlays == 0) problems.Add("no prompt card was played");
                    long rf = Counter(r, "combat.select.remote_frames"), ra = Counter(r, "combat.select.remote_answers");
                    if (rf != ra) problems.Add($"{rf} remote prompt frames, {ra} remote answers");
                    remoteFrames += rf;
                    remoteAnswers += ra;
                    hookEnqueues += Counter(r, "mp.remote.hook_enqueues");
                    resumes += Counter(r, "mp.remote.resumes");
                    localFrames += Counter(r, "combat.select.frames") - rf;
                    allyTargets += audit.AllyTargetsOffered;
                    ok &= problems.Count == 0;
                    rows.Add(new { seats = n, seed = r.Seed, outcome = Describe(r), prompt_card_plays = policy.PromptCardPlays, card_select_decisions = audit.SelectDecisions, remote_card_select_decisions = audit.RemoteSelectDecisions, remote_frames = rf, hook_enqueues = Counter(r, "mp.remote.hook_enqueues"), resumes = Counter(r, "mp.remote.resumes"), passed = problems.Count == 0, problems });
                }
            }
            if (remoteFrames == 0 || remoteAnswers == 0 || resumes == 0 || hookEnqueues == 0 || allyTargets == 0)
            {
                ok = false;
            }
            results["prompts"] = new
            {
                ok,
                what = "every seat holds hand-select, pile-select, choose-a-card and teammate-choice (TUTOR: the targeted ally chooses) cards and plays them: remote prompts become CARD_SELECT frames of that seat, answered as the seat's PlayerChoiceMessage; hook enqueues and resumes are sent by the owner's client",
                remote_frames = remoteFrames,
                remote_answers = remoteAnswers,
                remote_hook_enqueues = hookEnqueues,
                remote_resumes = resumes,
                local_frames = localFrames,
                any_ally_targets_offered = allyTargets,
                rows,
            };
            all &= ok;
        }

        // ------------------------------------------------------------------------------------------------ scaling
        if (cases.Contains("scaling"))
        {
            List<object> rows = new List<object>();
            List<string> problems = new List<string>();
            // static: which powers scale (Commands/PowerCmd.cs:88; the 12 of notes\h5-plan\rooms-multiseat-duties.md 2.1)
            List<string> scaled = ModelDb.AllPowers.Where(p => p.ShouldScaleInMultiplayer).Select(p => p.GetType().Name).Where(x => !x.StartsWith("Mock", StringComparison.Ordinal)).OrderBy(x => x, StringComparer.Ordinal).ToList();
            if (!scaled.SequenceEqual(ScaledPowers)) problems.Add("powers with ShouldScaleInMultiplayer: " + string.Join(",", scaled));
            long powerCalls = 0, blockCalls = 0, blockScaled = 0, hpChecks = 0;
            SortedDictionary<string, long> powerByType = new SortedDictionary<string, long>(StringComparer.Ordinal);
            foreach (int n in seatCounts)
            {
                foreach ((string enc, int act) in encounters)
                {
                    if (!Mine())
                    {
                        continue;
                    }
                    _powerSamples.Clear();
                    _blockSamples.Clear();
                    CombatAudit audit = new CombatAudit();
                    EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel($"MPSCALE:{n}:{enc}"), Characters = Lineup(n, 1), Scenario = FightScenario("mp-scaling", enc, act) };
                    EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)(n * 13)), new EpisodeOptions { DebugForcedHp = 99999, MaxPolicyDecisions = 40_000, MaxCombatTurns = 200, WireRoundTrip = wire }, audit);
                    List<string> local = new List<string>(audit.Errors);
                    CleanCounters(r, local);
                    if (r.Outcome == EpisodeOutcome.Aborted) local.Add(Describe(r));
                    decimal sc = Scaling(act, audit.Boss);
                    foreach ((string id, int maxHp, int? unscaled) in audit.FirstEnemies)
                    {
                        hpChecks++;
                        int expected = unscaled.HasValue ? Math.Min((int)(unscaled.Value * (decimal)n * sc), 999999999) : -1;
                        if (maxHp != expected) local.Add($"enemy {id}: MaxHp {maxHp}, expected trunc({unscaled} x {n} x {sc}) = {expected}");
                    }
                    foreach (PowerSample ps in _powerSamples)
                    {
                        powerCalls++;
                        powerByType[ps.Power] = powerByType.GetValueOrDefault(ps.Power) + 1;
                        decimal? expected = ExpectedPower(ps.Power, ps.Amount, n, sc);
                        if (!expected.HasValue || expected.Value != ps.Result) local.Add($"{ps.Power}: {ps.Amount} -> {ps.Result}, expected {expected?.ToString() ?? "no call"} (N {n}, scaling {sc})");
                    }
                    foreach (BlockSample bs in _blockSamples)
                    {
                        blockCalls++;
                        decimal expected = !bs.EnemyTarget || !bs.PoweredMove ? 1m : n <= 2 ? n : n * sc;
                        if (bs.EnemyTarget && bs.PoweredMove) blockScaled++;
                        if (expected != bs.Result) local.Add($"block multiplier {bs.Result}, expected {expected} (enemy target {bs.EnemyTarget}, powered move {bs.PoweredMove}, N {n}, scaling {sc})");
                    }
                    if (local.Count > 0) problems.Add($"{n} seats {enc}@{act}: " + string.Join(" | ", local.Take(4)));
                    rows.Add(new { seats = n, encounter = enc, act, boss = audit.Boss, scaling = sc, outcome = Describe(r), enemies = audit.FirstEnemies.Select(x => new { id = x.Id, max_hp = x.MaxHp, unscaled = x.Unscaled }).ToList(), power_calls = _powerSamples.Count, block_calls = _blockSamples.Count, passed = local.Count == 0 });
                }
            }
            _powerSamples.Clear();
            _blockSamples.Clear();
            results["scaling"] = new
            {
                ok = problems.Count == 0,
                what = "enemy HP, enemy block multiplier and scaled enemy powers of 2 / 3 / 4-seat fights in acts 1-3 (incl. bosses) against the decompiled formulas, with N and the act chosen by the test",
                hp_checks = hpChecks,
                power_scale_calls = powerCalls,
                power_scale_calls_by_type = powerByType,
                block_multiplier_calls = blockCalls,
                block_multiplier_calls_scaled = blockScaled,
                powers_with_should_scale = scaled,
                problems,
                rows,
            };
            all &= problems.Count == 0;
        }

        // ------------------------------------------------------------------------------------------------ cross-N
        if (cases.Contains("crossn"))
        {
            List<object> rows = new List<object>();
            bool ok = true;
            for (int e = 0; e < 3; e++)
            {
                string seed = HostSession.SeedFromLabel("MPCROSS:" + e);
                string character = DriverCommands.AllCharacters[e % DriverCommands.AllCharacters.Length];
                CombatAudit one = new CombatAudit();
                EpisodeResult r1 = session.RunEpisode(new EpisodeSpec { Seed = seed, Characters = new[] { character }, Scenario = FightScenario("mp-cross-1", "NIBBITS_NORMAL", 0, null, _ => one.SawCombat) }, new FirstActionPolicy(), new EpisodeOptions { MaxPolicyDecisions = 50 }, one);
                foreach (int n in seatCounts)
                {
                    string[] lineup = Lineup(n, e);
                    lineup[0] = character;
                    CombatAudit many = new CombatAudit();
                    EpisodeResult rn = session.RunEpisode(new EpisodeSpec { Seed = seed, Characters = lineup, Scenario = FightScenario("mp-cross-n", "NIBBITS_NORMAL", 0, null, _ => many.SawCombat) }, new FirstActionPolicy(), new EpisodeOptions { MaxPolicyDecisions = 50, WireRoundTrip = wire }, many);
                    // The audit records the enemies and seat 0's hand at the first COMBAT decision; the episode stops right after it.
                    List<string> notes = new List<string>();
                    bool sameUnscaled = one.FirstEnemies.Select(x => x.Unscaled).SequenceEqual(many.FirstEnemies.Select(x => x.Unscaled));
                    bool scaledOk = one.FirstEnemies.Count == many.FirstEnemies.Count && one.FirstEnemies.Zip(many.FirstEnemies, (x, y) => y.MaxHp == Math.Min((int)(x.MaxHp * (decimal)n * 1.1m), 999999999)).All(v => v);
                    bool sameHand = one.Seat0FirstHand.SequenceEqual(many.Seat0FirstHand);
                    if (one.FirstEnemies.Count == 0 || many.FirstEnemies.Count == 0) notes.Add("no first combat decision: " + Describe(r1) + " / " + Describe(rn));
                    rows.Add(new
                    {
                        seed,
                        character,
                        seats = n,
                        enemies_1 = one.FirstEnemies.Select(x => x.Id + ":" + x.MaxHp).ToList(),
                        enemies_n = many.FirstEnemies.Select(x => x.Id + ":" + x.MaxHp + " (unscaled " + x.Unscaled + ")").ToList(),
                        same_unscaled_hp_roll = sameUnscaled,
                        scaled_hp_is_trunc_of_1_seat_hp_x_n_x_1_1 = scaledOk,
                        seat0_first_hand_1 = one.Seat0FirstHand,
                        seat0_first_hand_n = many.Seat0FirstHand,
                        seat0_first_hand_equal = sameHand,
                        notes,
                    });
                    ok &= notes.Count == 0 && rn.Outcome != EpisodeOutcome.Aborted && r1.Outcome != EpisodeOutcome.Aborted;
                }
            }
            results["crossn"] = new
            {
                ok,
                what = "INFORMATIVE (h5-plan M1-7): same seed and seat-0 character, 1 seat vs N seats, same encounter from the first room. Reported: whether the unscaled enemy HP roll and seat 0's first hand agree. 'ok' only requires that both runs reach their first combat decision.",
                rows,
            };
            all &= ok;
        }

        // ------------------------------------------------------------------------------------------------ determinism
        if (cases.Contains("determinism"))
        {
            List<string> hashes = new List<string>();
            List<object> mismatches = new List<object>();
            long compared = 0, signatures = 0, differsOnSeed = 0, differsOnPolicy = 0, controls = 0;
            foreach (int n in seatCounts)
            {
                foreach ((string enc, int act) in encounters.Take(4))
                {
                    if (!Mine())
                    {
                        continue;
                    }
                    for (int e = 0; e < Math.Max(2, episodes / 2); e++)
                    {
                        EpisodeResult Play(string seedLabel, ulong policySeed)
                        {
                            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel(seedLabel), Characters = Lineup(n, e), Ascension = e % 2 == 0 ? 0 : 10, Scenario = FightScenario("mp-det", enc, act) };
                            return session.RunEpisode(spec, new RandomLegalPolicy(policySeed), new EpisodeOptions { DebugForcedHp = e % 2 == 0 ? 0 : 99999, MaxPolicyDecisions = 60_000, MaxCombatTurns = 300, ComputeSignatures = true, RecordTrace = true, WireRoundTrip = wire });
                        }
                        string label = $"MPDET:{n}:{enc}:{e}";
                        ulong ps = (ulong)(97 * n + e);
                        EpisodeResult ra = Play(label, ps);
                        EpisodeResult rb = Play(label, ps);
                        compared++;
                        signatures += ra.Trace?.Count ?? 0;
                        hashes.Add(ra.TraceHash.ToString("x16"));
                        bool same = ra.TraceHash == rb.TraceHash && ra.Outcome == rb.Outcome && ra.FinalHps.SequenceEqual(rb.FinalHps) && ra.Counters.DecisionPoints == rb.Counters.DecisionPoints
                                    && ra.Trace != null && rb.Trace != null && ra.Trace.SequenceEqual(rb.Trace) && ra.Outcome != EpisodeOutcome.Aborted;
                        if (!same && mismatches.Count < 10)
                        {
                            mismatches.Add(new { seats = n, encounter = enc, episode = e, a = Describe(ra), b = Describe(rb), hash_a = ra.TraceHash.ToString("x16"), hash_b = rb.TraceHash.ToString("x16"), len_a = ra.Trace?.Count, len_b = rb.Trace?.Count });
                        }
                        if (e == 0)
                        {
                            controls++;
                            if (Play(label + ":X", ps).TraceHash != ra.TraceHash) differsOnSeed++;
                            if (Play(label, ps ^ 0x5DEECE66DUL).TraceHash != ra.TraceHash) differsOnPolicy++;
                        }
                    }
                }
            }
            bool ok = mismatches.Count == 0 && compared > 0 && differsOnSeed == controls && differsOnPolicy == controls;
            results["determinism"] = new
            {
                ok,
                what = "every multi-seat fight twice in one process: identical per-decision signature traces (RunSignature of all players + decision shape + chosen index); controls with another game seed / policy seed must differ. trace_hashes are compared across processes by the gate script.",
                fights_compared = compared,
                signatures_compared = signatures,
                mismatches,
                controls,
                differs_on_other_game_seed = differsOnSeed,
                differs_on_other_policy_seed = differsOnPolicy,
                trace_hashes = hashes,
            };
            all &= ok;
        }

        // ------------------------------------------------------------------------------------------------ mixed
        if (cases.Contains("mixed"))
        {
            EpisodeResult Single(string label)
            {
                return session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel(label), Characters = new[] { "IRONCLAD" } }, new RandomLegalPolicy(11),
                    new EpisodeOptions { MaxPolicyDecisions = 400, ComputeSignatures = true, RecordTrace = true });
            }
            long resetsBefore = MultiSeatStats.RunLobbyResets;
            EpisodeResult first = Single("MPMIXED:1");
            EpisodeResult multi = session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("MPMIXED:4"), Characters = Lineup(4, 1), Scenario = FightScenario("mp-mixed", "NIBBITS_NORMAL", 0) },
                new RandomLegalPolicy(3), new EpisodeOptions { DebugForcedHp = 99999, MaxPolicyDecisions = 30_000, WireRoundTrip = wire });
            bool lobbyNull = RunManager.Instance.RunLobby == null;
            EpisodeResult third = Single("MPMIXED:1");
            bool ok = first.Outcome != EpisodeOutcome.Aborted && multi.Outcome != EpisodeOutcome.Aborted && third.Outcome != EpisodeOutcome.Aborted
                      && first.TraceHash == third.TraceHash && first.Trace != null && third.Trace != null && first.Trace.SequenceEqual(third.Trace)
                      && lobbyNull && MultiSeatStats.RunLobbyResets - resetsBefore == 1;
            results["mixed"] = new
            {
                ok,
                what = "a 1-seat episode, a 4-seat episode and the same 1-seat episode again in one process: the third reproduces the first's trace; RunManager.RunLobby is null again after the multi-seat run (mp-run-lobby-reset)",
                first = Describe(first),
                multi = Describe(multi),
                third = Describe(third),
                first_trace_hash = first.TraceHash.ToString("x16"),
                third_trace_hash = third.TraceHash.ToString("x16"),
                signatures = first.Trace?.Count ?? 0,
                run_lobby_null_after_multi_seat_run = lobbyNull,
                run_lobby_resets = MultiSeatStats.RunLobbyResets - resetsBefore,
            };
            all &= ok;
        }

        // ------------------------------------------------------------------------------------------------ M2: the room flows
        MultiSeatRoomTests.Run(session, a, cases, seatCounts, wire, results, ref all);

        HostJson.Emit(new
        {
            kind = "selftest-mp",
            ok = all,
            cases = cases.OrderBy(x => x, StringComparer.Ordinal).ToList(),
            seats = seatCounts,
            episodes,
            shard,
            shards,
            wire_round_trip = wire,
            encounters = encounters.Select(x => x.Id + "@" + x.Act).ToList(),
            vocab_hash = session.Registry.VocabHash,
            tool_patches = session.ToolPatches.Select(p => p.Id).ToList(),
            results,
        }, a.Get("out"));
        return all ? 0 : 4;
    }
}
