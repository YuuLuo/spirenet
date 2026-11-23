using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;
using SpireHost.Driver.Tools;

namespace SpireHost.Driver.Rooms;

/// <summary>One isolated room test: a real run start, a loadout + debug jump, a policy, a stop condition and a check.</summary>
internal sealed class RoomScenario
{
    public string Name = "";
    public string Group = "";
    public Func<RunContext, Task> Setup = _ => Task.CompletedTask;
    public Func<RunContext, bool>? StopWhen;
    public Func<ulong, PreferPolicy> Policy = seed => new PreferPolicy(seed);
    public int Hp;
    public int Ascension;
    public long MaxPolicyDecisions = 20_000;

    /// <summary>
    /// Fresh game seeds to try (name + "#k") when an attempt fails for a reason outside the rooms driver - used
    /// by the boss tests: some act-3 bosses (TEST_SUBJECT_BOSS) kill even a 99,999 HP player, which is a correct Loss.
    /// Every attempt is reported.
    /// </summary>
    public int Attempts = 1;

    /// <summary>Extra checks: returns null when they hold, else the failure reason. Called after the episode.</summary>
    public Func<EpisodeResult, RoomsObserver, string?>? Check;

    /// <summary>Per-decision hook installed on the observer.</summary>
    public Action<RunContext, Decision, int, bool>? OnEach;

    /// <summary>The scenario is EXPECTED to abort with this detail (soft-lock reporting).</summary>
    public string? ExpectAbortDetail;

    /// <summary>TEST ONLY, explicit: invariants this scenario deliberately violates to build its state (Scenario.Exempt).</summary>
    public string[]? Exempt;

    public Action? Cleanup;
}

/// <summary>The rooms scenario battery (DRIVER-API.md section 9 building blocks). TEST ONLY.</summary>
internal static class RoomScenarios
{
    // ------------------------------------------------------------------------------------------------
    //  Building blocks
    // ------------------------------------------------------------------------------------------------

    private static Player Me(RunContext ctx) => ctx.LocalPlayer;

    private static Task Relic<T>(RunContext ctx) where T : RelicModel => RelicCmd.Obtain(ModelDb.Relic<T>().ToMutable(), Me(ctx));

    private static Task Potion<T>(RunContext ctx) where T : PotionModel => PotionCmd.TryToProcure(ModelDb.Potion<T>().ToMutable(), Me(ctx));

    private static async Task FillBelt(RunContext ctx)
    {
        Player p = Me(ctx);
        int guard = 0;
        while (p.HasOpenPotionSlots && guard++ < 10)
        {
            await PotionCmd.TryToProcure(ModelDb.Potion<FirePotion>().ToMutable(), p);
        }
    }

    private static Task Offer(RunContext ctx, params Reward[] rewards) => RewardsCmd.OfferCustom(Me(ctx), rewards.ToList());

    private static CardReward NewCardReward(RunContext ctx) => new CardReward(CardCreationOptions.ForRoom(Me(ctx), RoomType.Monster), 3, Me(ctx));

    /// <summary>Travels to the act's boss point the way the game does once a vote resolved (test-only shortcut: no map walk).</summary>
    private static Task EnterBoss(RunContext ctx) => ScenarioTools.EnterBoss(ctx);

    private static bool Always(RunContext ctx) => true;

    private static long Count(SortedDictionary<string, long> d, string key) => d.GetValueOrDefault(key);

    private static string? Expect(params (bool ok, string what)[] checks)
    {
        foreach ((bool ok, string what) in checks)
        {
            if (!ok)
            {
                return what;
            }
        }
        return null;
    }

    private static string? TreeClean() => RoomsProbe.TreeChecksFailed == 0 ? null : "tree check failed: " + RoomsProbe.FirstTreeCheckFailure;

    // ------------------------------------------------------------------------------------------------
    //  The battery
    // ------------------------------------------------------------------------------------------------

    public static List<RoomScenario> Battery()
    {
        List<RoomScenario> s = new List<RoomScenario>();

        // ================= REWARDS =================
        s.Add(new RoomScenario
        {
            Group = "reward", Name = "reward:all-types",
            Setup = async ctx =>
            {
                Player p = Me(ctx);
                CardModel special = ctx.Run.CreateCard(ModelDb.Card<MegaCrit.Sts2.Core.Models.Cards.Anger>(), p);
                await Offer(ctx, new GoldReward(10, 20, p), new PotionReward(p), new RelicReward(p), new SpecialCardReward(special, p),
                    NewCardReward(ctx), new CardRemovalReward(p), new CardUpgradeReward(p));
            },
            StopWhen = Always,
            Policy = seed => new PreferPolicy(seed).Prefer(PreferPolicy.TakeAnyReward).Prefer(PreferPolicy.PickCard),
            Check = (r, o) => Expect(
                (RoomsProbe.RewardSetsOffered >= 1, "no reward set offered"),
                (Count(RoomsProbe.RewardTypesTaken, "GoldReward") == 1, "gold not taken"),
                (r.Counters.AutoGold >= 1, "gold not auto-taken"),
                (Count(RoomsProbe.RewardTypesTaken, "PotionReward") == 1, "potion not taken"),
                (Count(RoomsProbe.RewardTypesTaken, "RelicReward") == 1, "relic not taken"),
                (Count(RoomsProbe.RewardTypesTaken, "SpecialCardReward") == 1, "special card not taken"),
                (Count(RoomsProbe.RewardTypesTaken, "CardReward") == 1, "card reward not picked"),
                (RoomsProbe.CardRewardDecisions >= 1 && r.Counters.NestedDecisions >= 1, "no nested CARD_REWARD decision"),
                (Count(RoomsProbe.RewardTypesTaken, "CardRemovalReward") + Count(RoomsProbe.RewardTypesTaken, "CardRemovalReward:rejected") >= 1, "removal not taken"),
                (Count(RoomsProbe.RewardTypesTaken, "CardUpgradeReward") + Count(RoomsProbe.RewardTypesTaken, "CardUpgradeReward:rejected") >= 1, "upgrade not taken"),
                (RoomsProbe.RewardAutoClosed == 1, "fully taken non-terminal set did not close itself")),
        });

        s.Add(new RoomScenario
        {
            Group = "reward", Name = "reward:skip-non-terminal",
            Setup = ctx => Offer(ctx, new RelicReward(Me(ctx)), NewCardReward(ctx)),
            StopWhen = Always,
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Reward, Verb.Proceed)),
            Check = (r, o) => Expect((RoomsProbe.NonTerminalSkips == 1, "PROCEED did not SkipLocalRewardsSet"), (RoomsProbe.RewardTakes == 0, "a reward was taken")),
        });

        s.Add(new RoomScenario
        {
            Group = "reward", Name = "reward:potion-full-belt",
            Setup = async ctx =>
            {
                await FillBelt(ctx);
                await Offer(ctx, new PotionReward(Me(ctx)));
            },
            StopWhen = Always,
            Policy = seed => new PreferPolicy(seed).Prefer(PreferPolicy.TakeAnyReward).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Reward, Verb.DiscardPotion)),
            Check = (r, o) => Expect(
                (RoomsProbe.PotionRewardMasked >= 1, "potion reward was offered with a full belt"),
                (RoomsProbe.PotionDiscards >= 1, "no discard on the REWARD frame"),
                (Count(RoomsProbe.RewardTypesTaken, "PotionReward") == 1, "potion reward not taken after the discard")),
        });

        s.Add(new RoomScenario
        {
            Group = "reward", Name = "reward:sozu-full-belt",
            Setup = async ctx =>
            {
                await FillBelt(ctx);
                await Relic<Sozu>(ctx);
                await Offer(ctx, new PotionReward(Me(ctx)));
            },
            StopWhen = Always,
            Policy = seed => new PreferPolicy(seed).Prefer(PreferPolicy.TakeAnyReward),
            Check = (r, o) => Expect(
                (RoomsProbe.PotionRewardMasked == 0, "Sozu: ShouldProcurePotion is false, the reward must stay clickable"),
                (Count(RoomsProbe.RewardTypesTaken, "PotionReward") == 1, "Sozu: the potion reward is 'taken' without a potion")),
        });

        s.Add(new RoomScenario
        {
            Group = "reward", Name = "reward:neows-bones-disallow-skipping",
            Setup = ctx => Relic<NeowsBones>(ctx),
            StopWhen = Always,
            Check = (r, o) => Expect((RoomsProbe.ProceedDisallowed >= 1, "DisallowSkipping never applied"), (RoomsProbe.RewardTakes >= 2, "both relic rewards must be taken")),
        });

        s.Add(new RoomScenario
        {
            Group = "reward", Name = "reward:nested-orrery",
            Setup = ctx => Offer(ctx, new RelicReward(ModelDb.Relic<Orrery>().ToMutable(), Me(ctx)), new GoldReward(5, 6, Me(ctx))),
            StopWhen = Always,
            Policy = seed => new PreferPolicy(seed).Prefer(PreferPolicy.IsReward<RelicReward>).Prefer(PreferPolicy.TakeAnyReward),
            Check = (r, o) => Expect((RoomsProbe.RewardFrameMaxDepth >= 2, "no nested reward set"), (RoomsProbe.RewardSetsOffered >= 2, "Orrery offered no set")),
        });

        s.Add(new RoomScenario
        {
            Group = "card-reward", Name = "card-reward:driftwood-reroll",
            Setup = async ctx =>
            {
                await Relic<Driftwood>(ctx);
                await Offer(ctx, NewCardReward(ctx));
            },
            StopWhen = Always,
            Policy = seed => new PreferPolicy(seed).Prefer(PreferPolicy.TakeAnyReward).Prefer((d, a) => PreferPolicy.Alt(d, a, "REROLL")).Prefer(PreferPolicy.PickCard),
            Check = (r, o) => Expect(
                (Count(RoomsProbe.CardRewardAlternativesChosen, "REROLL") == 1, "REROLL not chosen exactly once"),
                (RoomsProbe.CardRewardDecisions >= 2, "no second CARD_REWARD call after the reroll"),
                (RoomsProbe.CardRewardRerollFiltered >= 1, "the stale REROLL was offered again after rerolling"),
                (Count(RoomsProbe.RewardTypesTaken, "CardReward") == 1, "no card picked after the reroll")),
        });

        s.Add(new RoomScenario
        {
            Group = "card-reward", Name = "card-reward:paels-wing-sacrifice",
            Setup = async ctx =>
            {
                await Relic<PaelsWing>(ctx);
                await Offer(ctx, NewCardReward(ctx), NewCardReward(ctx));
            },
            StopWhen = Always,
            Policy = seed => new PreferPolicy(seed).Prefer(PreferPolicy.TakeAnyReward).Prefer((d, a) => PreferPolicy.Alt(d, a, "SACRIFICE")),
            Check = (r, o) => Expect((Count(RoomsProbe.CardRewardAlternativesChosen, "SACRIFICE") == 2, "SACRIFICE not chosen twice"), (RoomsProbe.RewardAutoClosed == 1, "sacrificed rewards must complete the set")),
        });

        s.Add(new RoomScenario
        {
            Group = "card-reward", Name = "card-reward:skip-keeps-reward",
            Setup = ctx => Offer(ctx, NewCardReward(ctx)),
            StopWhen = Always,
            // Take + skip twice (the reward must stay), then leave the screen.
            Policy = seed => new PreferPolicy(seed)
                .Prefer((d, a) => Count(RoomsProbe.CardRewardAlternativesChosen, "Skip") < 2 && PreferPolicy.TakeAnyReward(d, a))
                .Prefer((d, a) => PreferPolicy.Alt(d, a, "Skip"))
                .Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Reward, Verb.Proceed)),
            Check = (r, o) => Expect((Count(RoomsProbe.CardRewardAlternativesChosen, "Skip") == 2, "skip did not keep the reward (it was not offered again)"),
                (Count(RoomsProbe.RewardTypesTaken, "CardReward:rejected") == 2, "skipped card reward must return false"), (RoomsProbe.NonTerminalSkips == 1, "set not skipped at PROCEED")),
        });

        s.Add(new RoomScenario
        {
            Group = "card-reward", Name = "card-reward:egg-relics",
            Setup = async ctx =>
            {
                await Relic<FrozenEgg>(ctx);
                await Relic<MoltenEgg>(ctx);
                await Relic<ToxicEgg>(ctx);
                await Offer(ctx, NewCardReward(ctx), NewCardReward(ctx));
            },
            StopWhen = Always,
            Policy = seed => new PreferPolicy(seed).Prefer(PreferPolicy.TakeAnyReward).Prefer(PreferPolicy.PickCard),
        });

        // ================= COMBAT WON + PROCEED TREE =================
        s.Add(new RoomScenario
        {
            Group = "tree", Name = "tree:monster->map",
            Setup = ctx => ScenarioTools.JumpToEncounter(ctx, "NIBBITS_WEAK"),
            Hp = 99999,
            Policy = seed => new PreferPolicy(seed, weakenEnemies: true),
            Check = (r, o) => Expect((RoomsProbe.CombatWonWithRewards == 1, "combat won did not offer rewards"), (RoomsProbe.TreeMap == 1, "terminal proceed did not open the map"), (RoomsProbe.TerminalSetsOffered == 1, "no terminal set")),
        });

        s.Add(new RoomScenario
        {
            Group = "tree", Name = "tree:elite-rewards->map",
            Setup = ctx => ScenarioTools.JumpToEncounter(ctx, "BYRDONIS_ELITE"),
            Hp = 99999,
            Policy = seed => new PreferPolicy(seed, weakenEnemies: true).Prefer(PreferPolicy.TakeAnyReward).Prefer(PreferPolicy.PickCard),
            Check = (r, o) => Expect((Count(RoomsProbe.RewardTypesTaken, "RelicReward") >= 1, "elite relic reward not taken"), (RoomsProbe.TreeMap == 1, "no map after the elite")),
        });

        s.Add(new RoomScenario
        {
            Group = "tree", Name = "tree:terminal-left-for-room-exit",
            Setup = ctx => ScenarioTools.JumpToEncounter(ctx, "NIBBITS_WEAK"),
            Hp = 99999,
            Policy = seed => new PreferPolicy(seed, weakenEnemies: true).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Reward, Verb.Proceed)),
            // Continue past the map: the next room entry must skip the left-over terminal set (BeforeLeavingRoom).
            StopWhen = ctx => ctx.RoomSerial >= 3,
            Check = (r, o) => Expect((RoomsProbe.TerminalLeftForRoomExit == 1, "terminal set not left for the room exit"), (RoomsProbe.TreeMap == 1, "no map")),
        });

        s.Add(new RoomScenario
        {
            Group = "tree", Name = "tree:act1-boss->act2",
            Attempts = 6,
            Setup = EnterBoss,
            Hp = 99999,
            Policy = seed => new PreferPolicy(seed, weakenEnemies: true),
            StopWhen = ctx => ctx.Run.CurrentActIndex >= 1,
            Check = (r, o) => Expect((RoomsProbe.TreeActChange == 1, "act boss proceed did not change act"), (RoomsProbe.TreeChecksPassed >= 1, "act change not verified"), (TreeClean() == null, TreeClean() ?? "")),
        });

        s.Add(new RoomScenario
        {
            Group = "tree", Name = "tree:last-act-boss->architect (A0)",
            Attempts = 6,
            Setup = async ctx =>
            {
                await ScenarioTools.EnterAct(ctx, ctx.Run.Acts.Count - 1);
                await EnterBoss(ctx);
            },
            Hp = 99999,
            Policy = seed => new PreferPolicy(seed, weakenEnemies: true),
            StopWhen = _ => false,
            Check = (r, o) => Expect((r.Outcome == EpisodeOutcome.Win, "not a win: " + r.Outcome), (RoomsProbe.TreeArchitect == 1, "last boss did not go to TheArchitect"), (RoomsProbe.TreeFirstOfTwoBosses == 0, "A0 has no second boss"), (TreeClean() == null, TreeClean() ?? "")),
        });

        s.Add(new RoomScenario
        {
            Group = "tree", Name = "tree:A10 double boss->architect",
            Attempts = 6,
            Ascension = 10,
            Setup = async ctx =>
            {
                await ScenarioTools.EnterAct(ctx, ctx.Run.Acts.Count - 1);
                await EnterBoss(ctx);
            },
            Hp = 99999,
            Policy = seed => new PreferPolicy(seed, weakenEnemies: true),
            StopWhen = _ => false,
            Check = (r, o) => Expect((r.Outcome == EpisodeOutcome.Win, "not a win: " + r.Outcome), (RoomsProbe.TreeFirstOfTwoBosses == 1, "first boss of two not taken"), (RoomsProbe.TreeArchitect == 1, "second boss did not go to TheArchitect"), (RoomsProbe.TreeChecksPassed >= 2, "branches not verified"), (TreeClean() == null, TreeClean() ?? "")),
        });

        s.Add(new RoomScenario
        {
            Group = "tree", Name = "tree:event-combat-no-rewards-resume (BattlewornDummy)",
            Setup = ctx => ctx.RunManager.EnterRoomDebug(RoomType.Event, MapPointType.Unassigned, ModelDb.Event<BattlewornDummy>(), showTransition: false),
            Hp = 99999,
            Policy = seed => new PreferPolicy(seed, weakenEnemies: true),
            Check = (r, o) => Expect((RoomsProbe.CombatWonWithoutRewards == 1, "no rewards-less combat"), (RoomsProbe.NoRewardsResume == 1, "the parent event was not resumed"), (TreeClean() == null, TreeClean() ?? "")),
        });

        s.Add(new RoomScenario
        {
            Group = "tree", Name = "tree:event-combat-rewards-no-resume (TheLanternKey)",
            Setup = ctx => ctx.RunManager.EnterRoomDebug(RoomType.Event, MapPointType.Unassigned, ModelDb.Event<TheLanternKey>(), showTransition: false),
            Hp = 99999,
            Policy = seed => new PreferPolicy(seed, weakenEnemies: true)
                .Prefer((d, a) => d.Kind == DecisionKind.Event && a.Verb == Verb.ChooseOption && (a.Source.EventOption.TextKey.Contains("KEEP_THE_KEY") || a.Source.EventOption.TextKey.Contains("FIGHT")))
                .Prefer(PreferPolicy.TakeAnyReward),
            Check = (r, o) => Expect((RoomsProbe.CombatWonWithRewards == 1, "event combat gave no rewards"), (RoomsProbe.TreeMap == 1, "event combat with ShouldResume=false must open the map"), (RoomsProbe.TreeResumeEvent == 0, "must not resume")),
        });

        // ================= TREASURE =================
        s.Add(new RoomScenario
        {
            Group = "treasure", Name = "treasure:pick",
            Setup = ctx => ScenarioTools.JumpToRoom(ctx, RoomType.Treasure),
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Treasure, Verb.ChooseOption)),
            Check = (r, o) => Expect((RoomsProbe.ChestsOpened == 1, "chest not opened"), (RoomsProbe.TreasurePicks == 1 && RoomsProbe.TreasureRelicsObtained == 1, "relic not obtained"), (o.TreasureProceedFrames >= 1, "no proceed frame after the pick"), (RoomsProbe.ChestGold > 0, "no chest gold")),
        });

        s.Add(new RoomScenario
        {
            Group = "treasure", Name = "treasure:skip",
            Setup = ctx => ScenarioTools.JumpToRoom(ctx, RoomType.Treasure),
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Treasure, Verb.Skip)),
            // Go on to the next room: the next chest must start its relic picking (the "pick or skip" pitfall).
            StopWhen = ctx => ctx.RoomSerial >= 4,
            Check = (r, o) => Expect((RoomsProbe.TreasureSkips == 1, "relic not skipped"), (RoomsProbe.TreasureRelicsObtained == 0, "skip obtained a relic")),
        });

        s.Add(new RoomScenario
        {
            Group = "treasure", Name = "treasure:silver-crucible-empty",
            Setup = async ctx =>
            {
                await Relic<SilverCrucible>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Treasure);
            },
            Check = (r, o) => Expect((RoomsProbe.ChestsEmpty == 1, "empty chest not completed with CompleteWithNoRelics"), (o.TreasureProceedFrames >= 1, "no proceed frame for the empty chest")),
        });

        s.Add(new RoomScenario
        {
            Group = "treasure", Name = "treasure:chest-page-auto",
            Setup = ctx => ScenarioTools.JumpToRoom(ctx, RoomType.Treasure),
            Check = (r, o) => Expect(
                (o.ChestFrames == 1 && o.ChestFramesAuto == 1, $"CHEST page: {o.ChestFrames} frames, {o.ChestFramesAuto} auto (expected exactly one auto-advanced OPEN_CHEST)"),
                (r.Counters.AutoSingle[(int)DecisionKind.Treasure] >= 1, "OPEN_CHEST not counted as an auto-single TREASURE decision"),
                (RoomsProbe.ChestsOpened == 1, "chest not opened")),
        });

        s.Add(new RoomScenario
        {
            Group = "treasure", Name = "treasure:potion-before-opening",
            Setup = async ctx =>
            {
                ScenarioTools.SetHp(ctx, 10);
                await Potion<FruitJuice>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Treasure);
            },
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => d.Kind == DecisionKind.Treasure && a.Verb == Verb.UsePotion).Prefer((d, a) => a.Verb is Verb.OpenChest or Verb.ChooseOption or Verb.Proceed),
            OnEach = (ctx, d, i, auto) =>
            {
                if (d.Kind == DecisionKind.Treasure && d.SubKind == (int)TreasurePhase.Chest && d.Actions[i].Verb == Verb.UsePotion)
                {
                    PotionBeforeChest = RoomsProbe.ChestsOpened == 0;
                }
            },
            Check = (r, o) => Expect(
                (PotionBeforeChest, "the potion was not used on the CHEST page before the chest was opened"),
                (Count(RoomsProbe.PotionsUsed, "FRUIT_JUICE") == 1, "FruitJuice not used"),
                (o.ChestFrames == 2 && o.ChestFramesAuto == 1, $"CHEST page: {o.ChestFrames} frames / {o.ChestFramesAuto} auto (expected: policy frame with the potion, then auto OPEN_CHEST)"),
                (RoomsProbe.TreasurePicks == 1, "relic not picked after opening")),
        });

        // ================= REST =================
        s.Add(new RoomScenario
        {
            Group = "rest", Name = "rest:basic",
            Setup = async ctx =>
            {
                ScenarioTools.SetHp(ctx, 10);
                await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite);
            },
            Check = (r, o) => Expect((RoomsProbe.RestSuccesses >= 1, "no successful rest option"), (RoomsProbe.RestLeaves == 1, "rest site not left")),
        });

        s.Add(new RoomScenario
        {
            Group = "rest", Name = "rest:miniature-tent-multi-choice",
            Setup = async ctx =>
            {
                ScenarioTools.SetHp(ctx, 10);
                await Relic<MiniatureTent>(ctx);
                await Relic<Girya>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite);
            },
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Rest, Verb.ChooseOption)).Prefer((d, a) => d.Kind == DecisionKind.CardSelect && a.Verb == Verb.SelectCard),
            Check = (r, o) => Expect((RoomsProbe.RestSuccesses >= 2, "MiniatureTent: only one option was chosen"), (o.RestChooseAfterSuccess >= 1, "no CHOOSE offered after a success")),
        });

        s.Add(new RoomScenario
        {
            Group = "rest", Name = "rest:relic-options",
            Setup = async ctx =>
            {
                ScenarioTools.SetHp(ctx, 10);
                await Relic<Shovel>(ctx);
                await Relic<MeatCleaver>(ctx);
                await Relic<PumpkinCandle>(ctx);
                await Relic<DreamCatcher>(ctx);
                await Relic<TinyMailbox>(ctx);
                await Relic<MiniatureTent>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite);
            },
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Rest, Verb.ChooseOption)).Prefer(PreferPolicy.TakeAnyReward),
            Check = (r, o) => Expect((RoomsProbe.RestSuccesses >= 3, "relic options not chosen")),
        });

        s.Add(new RoomScenario
        {
            Group = "rest", Name = "rest:soft-lock-reported",
            Setup = async ctx =>
            {
                Player p = Me(ctx);
                await CardPileCmd.RemoveFromDeck(p.Deck.Cards.ToList(), showPreview: false);
                // TEST ONLY: the game's own test hook (RestSiteOption.cs:41-45) yields a site whose only option is disabled.
                // The fidelity invariant rest_options_inert forbids exactly this in production; this one scenario is
                // exempted explicitly (ruling D3) and resets the hook in Cleanup.
                RestSiteOption.generateForTests = pl => new List<RestSiteOption> { new SmithRestSiteOption(pl) };
                await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite);
            },
            Exempt = new[] { "rest_options_inert" },
            Cleanup = () => RestSiteOption.generateForTests = null,
            ExpectAbortDetail = "rest_site_soft_lock",
        });

        // ================= SHOP =================
        s.Add(new RoomScenario
        {
            Group = "shop", Name = "shop:buy-anything",
            Setup = async ctx =>
            {
                ScenarioTools.SetGold(ctx, 999);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
            },
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Shop, Verb.ChooseOption)).Prefer(PreferPolicy.TakeAnyReward),
            Check = (r, o) => Expect((RoomsProbe.ShopPurchases >= 5, "few purchases with 999 gold"), (RoomsProbe.ShopLeaves == 1, "shop not left")),
        });

        s.Add(new RoomScenario
        {
            Group = "shop", Name = "shop:card-removal",
            Setup = async ctx =>
            {
                ScenarioTools.SetGold(ctx, 300);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
            },
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => d.Kind == DecisionKind.Shop && a.Verb == Verb.ChooseOption && a.Source.Object is MerchantCardRemovalEntry).Prefer((d, a) => d.Kind == DecisionKind.CardSelect && a.Verb == Verb.SelectCard),
            Check = (r, o) => Expect((Count(RoomsProbe.ShopEntryTypesBought, "MerchantCardRemovalEntry") == 1, "card removal not bought")),
        });

        s.Add(new RoomScenario
        {
            Group = "shop", Name = "shop:lords-parasol",
            Setup = async ctx =>
            {
                await Relic<LordsParasol>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
            },
            OnEach = (ctx, d, i, auto) =>
            {
                if (d.Kind == DecisionKind.Shop && d.Context is ShopContext sc && !ParasolChecked)
                {
                    ParasolChecked = true;
                    ParasolLeftover = sc.Inventory.CharacterCardEntries.Count(e => e.IsStocked) + sc.Inventory.ColorlessCardEntries.Count(e => e.IsStocked) + sc.Inventory.RelicEntries.Count(e => e.IsStocked)
                        + (sc.Inventory.CardRemovalEntry is { IsStocked: true } ? 1 : 0);
                    LordsParasol? lp = ctx.LocalPlayer.Relics.OfType<LordsParasol>().FirstOrDefault();
                    ParasolTaskDone = lp?.LastPurchaseEverythingTask?.IsCompleted == true;
                }
            },
            Check = (r, o) => Expect((RoomsProbe.ParasolYields == 1, "the LordsParasol UI-yield patch did not run"), (ParasolChecked, "no SHOP frame"), (ParasolTaskDone, "SHOP frame asked before LastPurchaseEverythingTask completed"), (ParasolLeftover == 0, ParasolLeftover + " entries still stocked at the first SHOP frame")),
        });

        s.Add(new RoomScenario
        {
            Group = "shop", Name = "shop:foul-potion",
            Setup = async ctx =>
            {
                ScenarioTools.SetGold(ctx, 0);
                await Potion<FoulPotion>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
            },
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => d.Kind == DecisionKind.Shop && PreferPolicy.UsePotion<FoulPotion>(d, a)).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Shop, Verb.Proceed)),
            OnEach = (ctx, d, i, auto) =>
            {
                if (d.Kind == DecisionKind.Shop)
                {
                    if (FoulGoldBefore < 0 && d.Actions[i].Verb == Verb.UsePotion)
                    {
                        FoulGoldBefore = ctx.LocalPlayer.Gold;
                    }
                    else if (FoulGoldBefore >= 0 && FoulGoldAfter < 0)
                    {
                        FoulGoldAfter = ctx.LocalPlayer.Gold;
                    }
                }
            },
            Check = (r, o) => Expect((RoomsProbe.FoulPotionMerchantOverrides >= 1, "usability postfix never fired"), (Count(RoomsProbe.PotionsUsed, "FOUL_POTION") == 1, "FoulPotion not used"), (FoulGoldAfter - FoulGoldBefore == 100, $"gold {FoulGoldBefore} -> {FoulGoldAfter}, expected +100")),
        });

        s.Add(new RoomScenario
        {
            Group = "shop", Name = "shop:foul-potion-not-after-leave",
            Setup = async ctx =>
            {
                ScenarioTools.SetGold(ctx, 0);
                await Potion<FoulPotion>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
            },
            // Leave at once; on the MAP frame (still in the MerchantRoom) the merchant button is covered: USE must not be offered.
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Shop, Verb.Proceed)).Prefer((d, a) => a.Verb == Verb.Travel),
            StopWhen = ctx => ctx.RoomSerial >= 3,
            OnEach = (ctx, d, i, auto) =>
            {
                if (d.Kind == DecisionKind.Shop && d.Actions.Exists(x => PreferPolicy.UsePotion<FoulPotion>(d, x)))
                {
                    FoulOfferedOnShop = true;
                }
                if (d.Kind == DecisionKind.Map && ctx.CurrentRoom is MerchantRoom)
                {
                    FoulMapFramesInShop++;
                    if (d.Actions.Exists(x => PreferPolicy.UsePotion<FoulPotion>(d, x)))
                    {
                        FoulOfferedOnMap = true;
                    }
                }
            },
            Check = (r, o) => Expect(
                (FoulOfferedOnShop, "FoulPotion USE not offered on the SHOP frame"),
                (FoulMapFramesInShop >= 1, "no MAP frame while still in the merchant room"),
                (!FoulOfferedOnMap, "FoulPotion USE offered on the MAP frame after LEAVE"),
                (RoomsProbe.FoulPotionMerchantKeptFalse >= 1, "the postfix never kept the original false")),
        });

        {
            // The FakeMerchant SHOP frame + FakeMerchant FoulPotion patches of the events module with THIS module's potion
            // extension (ruling D4: the two FoulPotion postfixes compose; one test per room).
            s.Add(new RoomScenario
            {
                Group = "integration", Name = "integration:fake-merchant-foul-potion",
                Setup = async ctx =>
                {
                    ScenarioTools.SetGold(ctx, 0);
                    await Potion<FoulPotion>(ctx);
                    await ctx.RunManager.EnterRoomDebug(RoomType.Event, MapPointType.Unassigned, ModelDb.Event<FakeMerchant>(), showTransition: false);
                },
                Hp = 99999,
                Policy = seed => new PreferPolicy(seed, weakenEnemies: true).Prefer((d, a) => d.Kind == DecisionKind.Shop && PreferPolicy.UsePotion<FoulPotion>(d, a)).Prefer(PreferPolicy.TakeAnyReward),
                OnEach = (ctx, d, i, auto) =>
                {
                    if (d.Kind == DecisionKind.Shop && d.Context is ShopContext { FakeMerchant: true } && d.Actions[i].Verb == Verb.UsePotion)
                    {
                        FakeMerchantThrow = d.Actions[i].Target.Kind == RefKind.None && d.Actions[i].Owner is OutOfCombatPotions;
                    }
                },
                Check = (r, o) => Expect(
                    (FakeMerchantThrow, "FoulPotion not thrown (no target) by the rooms extension on the FakeMerchant SHOP frame"),
                    (Count(RoomsProbe.PotionsUsed, "FOUL_POTION") == 1, "FoulPotion not used"),
                    (RoomsProbe.CombatsWon >= 1, "the FakeMerchant fight did not happen / was not won"),
                    (RoomsProbe.FoulPotionMerchantOverrides == 0, "the rooms (MerchantRoom) postfix fired in the FakeMerchant event room")),
            });
        }

        s.Add(new RoomScenario
        {
            Group = "shop", Name = "shop:price-relics",
            Setup = async ctx =>
            {
                ScenarioTools.SetGold(ctx, 600);
                await Relic<MembershipCard>(ctx);
                await Relic<TheCourier>(ctx);
                await Relic<MawBank>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
            },
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => PreferPolicy.Verb_(d, a, DecisionKind.Shop, Verb.ChooseOption)).Prefer(PreferPolicy.TakeAnyReward),
            Check = (r, o) => Expect((RoomsProbe.ShopPurchases >= 3, "few purchases")),
        });

        // ================= POTIONS ON ROOM FRAMES =================
        s.Add(new RoomScenario
        {
            Group = "potions", Name = "potions:any-time-at-rest",
            Setup = async ctx =>
            {
                ScenarioTools.SetHp(ctx, 10);
                await Potion<FruitJuice>(ctx);
                await Potion<BloodPotion>(ctx);
                await Potion<FirePotion>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite);
            },
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => a.Verb == Verb.UsePotion),
            Check = (r, o) => Expect((Count(RoomsProbe.PotionsUsed, "FRUIT_JUICE") == 1 && Count(RoomsProbe.PotionsUsed, "BLOOD_POTION") == 1, "AnyTime potions not used"), (Count(RoomsProbe.PotionsUsed, "FIRE_POTION") == 0, "combat potion used out of combat")),
        });

        s.Add(new RoomScenario
        {
            Group = "potions", Name = "potions:entropic-brew-on-map",
            Setup = async ctx =>
            {
                await Potion<EntropicBrew>(ctx);
                await Potion<Ambergris>(ctx);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Treasure);
            },
            Policy = seed => new PreferPolicy(seed).Prefer((d, a) => a.Verb == Verb.UsePotion),
            StopWhen = ctx => ctx.RoomSerial >= 3,
            Check = (r, o) => Expect((RoomsProbe.PotionUses >= 2, "potions not used on room frames")),
        });

        return s;
    }

    // per-scenario scratch for the OnEach hooks (reset by Reset())
    internal static bool ParasolChecked;
    internal static int ParasolLeftover = -1;
    internal static bool ParasolTaskDone;
    internal static int FoulGoldBefore = -1;
    internal static int FoulGoldAfter = -1;
    internal static bool PotionBeforeChest;
    internal static bool FoulOfferedOnShop;
    internal static bool FoulOfferedOnMap;
    internal static int FoulMapFramesInShop;
    internal static bool FakeMerchantThrow;

    internal static void ResetScratch()
    {
        PotionBeforeChest = false;
        FoulOfferedOnShop = false;
        FoulOfferedOnMap = false;
        FoulMapFramesInShop = 0;
        FakeMerchantThrow = false;
        ParasolChecked = false;
        ParasolLeftover = -1;
        ParasolTaskDone = false;
        FoulGoldBefore = -1;
        FoulGoldAfter = -1;
    }

    // ------------------------------------------------------------------------------------------------
    //  Runner
    // ------------------------------------------------------------------------------------------------

    public static object RunOne(HostSession session, RoomScenario sc, string character, ulong policySeed, out bool passed)
    {
        List<object> attempts = new List<object>();
        passed = false;
        for (int k = 0; k < Math.Max(1, sc.Attempts) && !passed; k++)
        {
            attempts.Add(RunAttempt(session, sc, character, policySeed, k, out passed));
        }
        return attempts.Count == 1 ? attempts[0] : new { scenario = character + ": " + sc.Name, group = sc.Group, passed, attempts };
    }

    private static object RunAttempt(HostSession session, RoomScenario sc, string character, ulong policySeed, int attempt, out bool passed)
    {
        RoomsProbe.Reset();
        ResetScratch();
        RoomsObserver obs = new RoomsObserver { OnEach = sc.OnEach };
        EpisodeOptions options = new EpisodeOptions { DebugForcedHp = sc.Hp, MaxPolicyDecisions = sc.MaxPolicyDecisions };
        string name = character + ": " + sc.Name;
        string seed = HostSession.SeedFromLabel("SCENARIO:" + name + (attempt > 0 ? "#" + attempt : ""));
        EpisodeResult r;
        try
        {
            r = ScenarioRunner.Run(session, name, sc.Setup, character, ascension: sc.Ascension, seed: seed, policy: sc.Policy(policySeed), stopWhen: sc.StopWhen, options: options, observer: obs, exempt: sc.Exempt);
        }
        finally
        {
            sc.Cleanup?.Invoke();
        }
        string? failure = null;
        if (sc.ExpectAbortDetail != null)
        {
            if (r.Outcome != EpisodeOutcome.Aborted || r.Abort?.Detail != sc.ExpectAbortDetail)
            {
                failure = "expected Aborted/" + sc.ExpectAbortDetail + ", got " + r.Outcome + "/" + r.Abort?.Detail;
            }
        }
        else if (!ScenarioRunner.Passed(r))
        {
            failure = "episode: " + r.Outcome + " " + (r.TruncatedBy ?? r.Abort?.Signature);
        }
        else if (obs.Violations.Count > 0)
        {
            failure = "contract: " + string.Join(" | ", obs.Violations);
        }
        else if (sc.Check != null)
        {
            failure = sc.Check(r, obs);
        }
        passed = failure == null;
        return new
        {
            scenario = name,
            group = sc.Group,
            passed,
            failure,
            attempt,
            outcome = r.Outcome.ToString(),
            truncated_by = r.TruncatedBy,
            abort = r.Abort?.Signature,
            abort_message = r.Abort?.Message,
            abort_frames = r.Abort?.Frames,
            diagnostics = failure != null ? r.Abort?.Diagnostics : null,
            seed = r.Seed,
            policy_seed = policySeed,
            final_room = r.FinalRoom,
            final_hp = r.FinalHp,
            floor = r.TotalFloor,
            policy_decisions = r.Counters.PolicyDecisions,
            nested = r.Counters.NestedDecisions,
            auto_gold = r.Counters.AutoGold,
            kinds = obs.Kinds,
            potion_actions = obs.PotionActionsByKind,
            chest_frames = obs.ChestFrames,
            probe = RoomsProbe.Snapshot(),
            wall_ms = Math.Round(r.WallMs, 1),
        };
    }
}
