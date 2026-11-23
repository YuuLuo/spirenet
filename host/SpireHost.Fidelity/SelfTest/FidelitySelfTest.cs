using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Unlocks;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Fidelity.Patches;
using SpireHost.Patches;

namespace SpireHost.Fidelity.SelfTest;

/// <summary>
/// --selftest: the machine-checkable invariant test of the fidelity pack (task item 5).
///
///   static    0 Behaviour patches in the production set; every Fidelity-category patch of this pack / the Core gates
///             has a named self-check below that ran and passed, and every Fidelity-category patch of another module
///             names its self-check in the ledger; host\FIDELITY.md == the registry (one row per id);
///             the audit-site table covers every non-presentation site of the three audits.
///   scenarios each Fidelity patch exercised on its real branch through the run driver (real run start,
///             ScenarioTools debug jumps, the registered handlers): shop potion prices vary and draw the Shops
///             stream; CallingBell offers random Common / Uncommon / Rare relics and pulls the grab bag; Cauldron
///             offers random potions and draws PlayerRng.Rewards; RollingBoulder hits per target in list order;
///             a natural loss runs OnEnded(false) and fills the map-history stats; leaving a shop runs
///             OnMerchantExited; act selection / tutorial room take the real branch when the profile asks for it.
///   scheduling the Unpause TCS and the virtual turn-loop timeout, directly.
///   asserts   each "unreachable" assertion fires when its path is forced (negative tests), and the session is
///             clean afterwards.
///   fuzz      N natural-HP random runs (default 12): the veteran-profile gate counters (Core and this pack) stay 0,
///             no fidelity assertion fires, no abort's first frame is in a method this pack patches.
/// </summary>
public static class FidelitySelfTest
{
    private sealed class Check
    {
        public string Name = "";
        public bool Pass;
        public string Expected = "";
        public object? Observed;
        public string[] Covers = Array.Empty<string>();
    }

    /// <summary>Picks the first legal action whose verb is in the preference list (in order), else uniform random.</summary>
    private sealed class PreferPolicy : IPolicy
    {
        private readonly Verb[] _prefer;
        private readonly RandomLegalPolicy _fallback;

        public PreferPolicy(ulong seed, params Verb[] prefer)
        {
            _prefer = prefer;
            _fallback = new RandomLegalPolicy(seed);
        }

        public int Decide(Decision d)
        {
            foreach (Verb v in _prefer)
            {
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    if (d.Actions[i].Verb == v)
                    {
                        return i;
                    }
                }
            }
            return _fallback.Decide(d);
        }
    }

    private sealed class Observer : IEpisodeObserver
    {
        public Action<RunContext, Decision>? OnDecisionHook;
        public Action<RunContext, EpisodeResult>? OnTerminal;

        public void OnEpisodeStart(RunContext ctx)
        {
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto) => OnDecisionHook?.Invoke(ctx, decision);

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional) => OnTerminal?.Invoke(ctx, provisional);

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }
    }

    public static int Run(HostSession session, Args a)
    {
        List<Check> checks = new List<Check>();
        void Add(string name, bool pass, string expected, object? observed, params string[] covers)
        {
            checks.Add(new Check { Name = name, Pass = pass, Expected = expected, Observed = observed, Covers = covers });
        }

        StaticChecks(Add);
        ShopPotionPrices(session, Add);
        CallingBellAndCauldron(session, Add);
        RollingBoulder(session, Add);
        NaturalLoss(session, Add);
        MerchantExit(session, Add);
        MendLocalTargeting(session, Add);
        ActSelectionAndTutorial(Add);
        Scheduling(session, Add);
        Assertions(session, Add);
        Fuzz(session, a.GetInt("runs", 12), Add);
        CoverageCheck(checks, Add);

        bool ok = checks.All(c => c.Pass);
        HostJson.Emit(new
        {
            kind = "fidelity-selftest",
            ok,
            passed = checks.Count(c => c.Pass),
            total = checks.Count,
            patches = PatchRegistry.Records.Count,
            by_category = PatchRegistry.Records.GroupBy(r => r.Category.ToString()).ToDictionary(g => g.Key, g => g.Count()),
            counters = FidelityPatches.Counters(),
            checks = checks.Select(c => new { name = c.Name, pass = c.Pass, expected = c.Expected, observed = c.Observed, covers = c.Covers }),
        }, a.Get("out"));
        return ok ? 0 : 4;
    }

    // =====================================================================================================

    /// <summary>Fidelity-category registry id -> the check(s) that exercise its real branch.</summary>
    private static readonly Dictionary<string, string> FidelityCheckOf = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["fidelity-merchant-potion-price"] = "shop_potion_prices",
        ["fidelity-calling-bell-rewards"] = "calling_bell_rewards",
        ["fidelity-cauldron-rewards"] = "cauldron_rewards",
        ["fidelity-rolling-boulder-per-target"] = "rolling_boulder_per_target",
        ["fidelity-natural-death-on-ended"] = "natural_loss_bookkeeping",
        ["fidelity-map-history-player-stats"] = "natural_loss_bookkeeping",
        ["fidelity-merchant-room-exit"] = "merchant_exit_bookkeeping",
        ["fidelity-act-selection-discovery-gate"] = "act_selection_real_branch",
        ["fidelity-tutorial-room-type-gate"] = "tutorial_room_real_branch",
        ["fidelity-force-discovery-order"] = "veteran_gates_noop_fuzz",
        ["fidelity-mend-local-targeting"] = "mend_local_targeting",
    };

    private static void StaticChecks(Action<string, bool, string, object?, string[]> add)
    {
        List<string> behaviour = PatchRegistry.Records.Where(r => r.Category == PatchCategory.Behaviour).Select(r => r.Id).ToList();
        add("no_behaviour_patches", behaviour.Count == 0, "0 Behaviour patches in the default set", behaviour, Array.Empty<string>());

        // Fidelity-category ids of THIS pack and of the Core gates: each maps to a scenario check of this self-test.
        // The integrated production set also holds Fidelity-category ids of the rooms and events modules; their real-branch
        // checks are scenarios of those modules' own self-tests, named in the self-check column of their ledger rows.
        List<string> fidelity = PatchRegistry.Records.Where(r => r.Category == PatchCategory.Fidelity).Select(r => r.Id).ToList();
        List<string> mineFidelity = fidelity.Where(id => id.StartsWith("fidelity-", StringComparison.Ordinal)).ToList();
        List<string> unchecked_ = mineFidelity.Where(id => !FidelityCheckOf.ContainsKey(id)).ToList();
        add("every_fidelity_patch_has_a_check", unchecked_.Count == 0 && mineFidelity.Count > 0, "every Fidelity-category id of the fidelity pack / Core gates maps to a scenario check of this self-test",
            new { fidelity = mineFidelity, without_check = unchecked_ }, Array.Empty<string>());

        // the whole ledger against the whole registry (one merged ledger: host\FIDELITY.md)
        HostSession host = HostSession.Current ?? throw new InvalidOperationException("no session");
        PatchLedger.Result ledger = PatchLedger.Compare(null, host.ProductionPatches, host.ToolPatches, PatchRegistry.SharedSites());
        add("ledger_matches_registry", ledger.Ok,
            "host\\FIDELITY.md has exactly one row per registry id (same category, kind, target and registered text), no stale row, and the rows of a shared site name each other", ledger, Array.Empty<string>());

        // Verifier finding R-4: the check itself must see a row whose target, text, kind or category is not the registry's.
        // A copy of the ledger is doctored in memory, one field of one production row at a time.
        {
            string[] original = File.Exists(ledger.Path) ? File.ReadAllLines(ledger.Path) : Array.Empty<string>();
            PatchRecord victim = host.ProductionPatches.First(p => p.Id == "fidelity-map-history-player-stats");
            int at = Array.FindIndex(original, l => l.TrimStart().StartsWith("| `" + victim.Id + "` |", StringComparison.Ordinal));
            Dictionary<string, bool> seen = new Dictionary<string, bool>();
            bool control = original.Length > 0 && PatchLedger.CompareLines(original, host.ProductionPatches, host.ToolPatches, PatchRegistry.SharedSites()).Ok;
            if (at >= 0)
            {
                string[] cells = original[at].Split('|');
                void Doctor(string name, int cell, Func<string, string> change)
                {
                    string[] c2 = (string[])cells.Clone();
                    c2[cell] = change(c2[cell]);
                    string[] lines = (string[])original.Clone();
                    lines[at] = string.Join("|", c2);
                    PatchLedger.Result dr = PatchLedger.CompareLines(lines, host.ProductionPatches, host.ToolPatches, PatchRegistry.SharedSites());
                    seen[name] = !dr.Ok && dr.FieldMismatches.Count == 1 && dr.FieldMismatches[0].StartsWith(victim.Id + ":", StringComparison.Ordinal);
                }
                Doctor("target", 4, s => s.Replace("::", "::Moved"));
                Doctor("text", 5, s => " something else entirely ");
                Doctor("kind", 3, s => " postfix-doctored ");
                Doctor("category", 2, s => " Plumbing ");
            }
            add("ledger_check_sees_a_changed_target_text_kind_category", control && at >= 0 && seen.Count == 4 && seen.Values.All(v => v),
                "the unchanged ledger passes; with the target, the registered text, the kind or the category of one production row changed (in memory) the check reports exactly that row",
                new { control_passes = control, row_found = at >= 0, detected = seen }, Array.Empty<string>());
        }

        // Fidelity-category ids of the other modules: the ledger row must name a self-check (the module's self-test scenario).
        List<PatchLedger.Row> rows = File.Exists(ledger.Path) ? PatchLedger.Parse(ledger.Path) : new List<PatchLedger.Row>();
        List<string> others = fidelity.Where(id => !id.StartsWith("fidelity-", StringComparison.Ordinal)).ToList();
        List<string> withoutSelfCheck = new List<string>();
        foreach (string id in others)
        {
            PatchLedger.Row? row = rows.FirstOrDefault(r => r.Id == id);
            string[] cells = row == null ? Array.Empty<string>() : row.Text.Split('|');
            // | `id` | category | kind | target | what / why | self-check |
            string selfCheck = cells.Length > 6 ? cells[6].Trim() : "";
            if (selfCheck.Length < 8 || !selfCheck.Contains("--selftest", StringComparison.Ordinal))
            {
                withoutSelfCheck.Add(id);
            }
        }
        add("other_modules_fidelity_patches_name_a_selfcheck", withoutSelfCheck.Count == 0,
            "every Fidelity-category id of another module has a ledger row whose self-check cell names that module's self-test",
            new { ids = others, without_self_check = withoutSelfCheck }, Array.Empty<string>());

        SiteTable.Result st = SiteTable.Build(SiteTable.DefaultNotesDir());
        add("audit_site_table_complete", st.Ok, "every non-presentation audit site -> patch | invariant | team | no-op, every patch id installed",
            new { st.Total, st.NonPresentation, by_handler = st.Rows.GroupBy(r => r.HandledBy).ToDictionary(g => g.Key, g => g.Count()), st.Problems }, Array.Empty<string>());
    }

    // =====================================================================================================

    private static int RngCounter(Rng rng) => rng._counter;

    private static void ShopPotionPrices(HostSession session, Action<string, bool, string, object?, string[]> add)
    {
        int entries = 0, varied = 0, outOfBand = 0, directDraws = -1, directTestBranchWouldBe = 0;
        List<string> samples = new List<string>();
        bool allPassed = true;
        for (int s = 0; s < 6; s++)
        {
            int seedIndex = s;
            EpisodeResult r = ScenarioRunner.Run(session, "fidelity:shop-potion-price:" + s, async ctx =>
            {
                await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
                MerchantInventory inv = ctx.RunManager.MerchantSynchronizer.GetLocalInventory();
                foreach (MerchantPotionEntry e in inv.PotionEntries)
                {
                    if (e.Model == null)
                    {
                        continue;
                    }
                    int b = e.Model.Rarity switch { MegaCrit.Sts2.Core.Entities.Potions.PotionRarity.Rare => 100, MegaCrit.Sts2.Core.Entities.Potions.PotionRarity.Uncommon => 75, _ => 50 };
                    entries++;
                    if (e._cost != b)
                    {
                        varied++;
                    }
                    if (e._cost < (int)Math.Round(b * 0.95) || e._cost > (int)Math.Round(b * 1.05))
                    {
                        outOfBand++;
                    }
                    if (samples.Count < 8)
                    {
                        samples.Add(e.Model.Id.Entry + ":" + e._cost + "/" + b);
                    }
                }
                if (seedIndex == 0)
                {
                    Player p = ctx.LocalPlayer;
                    int before = RngCounter(p.PlayerRng.Shops);
                    _ = new MerchantPotionEntry(ScenarioTools.Potion("FLEX_POTION").ToMutable(), p);
                    directDraws = RngCounter(p.PlayerRng.Shops) - before;
                }
            }, stopWhen: _ => true);
            allPassed &= ScenarioRunner.Passed(r);
        }
        add("shop_potion_prices", allPassed && entries > 0 && varied > 0 && outOfBand == 0 && directDraws == 1,
            "shop potion prices = round(base x Shops.NextFloat(0.95, 1.05)): some differ from the base, none outside the band; one CalcCost draws exactly 1 from PlayerRng.Shops (TestMode branch: 0)",
            new { scenarios_passed = allPassed, entries, varied, out_of_band = outOfBand, shops_draws_per_calc_cost = directDraws, test_branch_draws = directTestBranchWouldBe, samples }, new[] { "fidelity-merchant-potion-price" });
    }

    private static int GrabBagCount(Player p)
    {
        int n = 0;
        foreach (List<RelicModel> l in p.RelicGrabBag._deques.Values)
        {
            n += l.Count;
        }
        return n;
    }

    private static void CallingBellAndCauldron(HostSession session, Action<string, bool, string, object?, string[]> add)
    {
        // CallingBell
        List<string> bellSets = new List<string>();
        bool bellOk = true;
        bool bellRun = true;
        for (int s = 0; s < 3; s++)
        {
            int bagBefore = -1;
            RewardsSet? seen = null;
            int bagAtOffer = -1;
            Observer obs = new Observer
            {
                OnDecisionHook = (ctx, d) =>
                {
                    if (seen == null && bagBefore >= 0 && d.Context is RewardContext rc)
                    {
                        seen = rc.Set;
                        bagAtOffer = GrabBagCount(ctx.LocalPlayer);
                    }
                },
            };
            EpisodeResult r = ScenarioRunner.Run(session, "fidelity:calling-bell:" + s, async ctx =>
            {
                bagBefore = GrabBagCount(ctx.LocalPlayer);
                await ScenarioTools.GiveRelic(ctx, "CALLING_BELL");
            }, seed: HostSession.SeedFromLabel("FIDELITY:BELL:" + s), policy: new PreferPolicy((ulong)s + 1, Verb.ChooseOption, Verb.Proceed), stopWhen: _ => true, observer: obs);
            bellRun &= ScenarioRunner.Passed(r);
            if (seen == null)
            {
                bellOk = false;
                bellSets.Add("no reward set offered");
                continue;
            }
            List<RelicReward> rr = seen.Rewards.OfType<RelicReward>().ToList();
            string desc = string.Join(",", rr.Select(x => x.Relic?.Id.Entry + "(" + x.Relic?.Rarity + ")"));
            bellSets.Add(desc + " bag " + bagBefore + "->" + bagAtOffer);
            bool rarities = rr.Count == 3 && rr[0].Relic?.Rarity == MegaCrit.Sts2.Core.Entities.Relics.RelicRarity.Common
                && rr[1].Relic?.Rarity == MegaCrit.Sts2.Core.Entities.Relics.RelicRarity.Uncommon && rr[2].Relic?.Rarity == MegaCrit.Sts2.Core.Entities.Relics.RelicRarity.Rare;
            bool notFixed = !(rr.Count == 3 && rr[0].Relic?.Id.Entry == "ANCHOR" && rr[1].Relic?.Id.Entry == "GREMLIN_HORN" && rr[2].Relic?.Id.Entry == "MUMMIFIED_HAND");
            bellOk &= rarities && notFixed && bagBefore - bagAtOffer >= 3;
        }
        bool bellRandom = bellSets.Select(x => x.Split(' ')[0]).Distinct().Count() > 1;
        add("calling_bell_rewards", bellRun && bellOk && bellRandom,
            "CallingBell offers RelicReward(Common, Uncommon, Rare) pulled from the grab bag (bag shrinks by >= 3 at the offer), different across seeds; never the fixed Anchor / GremlinHorn / MummifiedHand",
            new { scenarios_passed = bellRun, sets = bellSets }, new[] { "fidelity-calling-bell-rewards" });

        // Cauldron
        List<string> cauldronSets = new List<string>();
        bool cOk = true;
        bool cRun = true;
        HashSet<string> fixedList = new HashSet<string> { "FLEX_POTION", "WEAK_POTION", "VULNERABLE_POTION", "STRENGTH_POTION", "DEXTERITY_POTION" };
        for (int s = 0; s < 3; s++)
        {
            int before = -1;
            int atOffer = -1;
            RewardsSet? seen = null;
            Observer obs = new Observer
            {
                OnDecisionHook = (ctx, d) =>
                {
                    if (seen == null && before >= 0 && d.Context is RewardContext rc)
                    {
                        seen = rc.Set;
                        atOffer = RngCounter(ctx.LocalPlayer.PlayerRng.Rewards);
                    }
                },
            };
            EpisodeResult r = ScenarioRunner.Run(session, "fidelity:cauldron:" + s, async ctx =>
            {
                before = RngCounter(ctx.LocalPlayer.PlayerRng.Rewards);
                await ScenarioTools.GiveRelic(ctx, "CAULDRON");
            }, seed: HostSession.SeedFromLabel("FIDELITY:CAULDRON:" + s), policy: new PreferPolicy((ulong)s + 1, Verb.Proceed), stopWhen: _ => true, observer: obs);
            cRun &= ScenarioRunner.Passed(r);
            if (seen == null)
            {
                cOk = false;
                cauldronSets.Add("no reward set offered");
                continue;
            }
            List<PotionReward> pr = seen.Rewards.OfType<PotionReward>().ToList();
            List<string> ids = pr.Select(x => x.Potion?.Id.Entry ?? "?").ToList();
            cauldronSets.Add(string.Join(",", ids) + " rewards-rng +" + (atOffer - before));
            cOk &= pr.Count == 5 && atOffer - before >= 5 && !ids.All(fixedList.Contains);
        }
        bool cRandom = cauldronSets.Select(x => x.Split(' ')[0]).Distinct().Count() > 1;
        add("cauldron_rewards", cRun && cOk && cRandom,
            "Cauldron offers 5 PotionReward(owner) populated from PlayerRng.Rewards (counter +>= 5 at the offer), different across seeds; not the fixed test list",
            new { scenarios_passed = cRun, sets = cauldronSets }, new[] { "fidelity-cauldron-rewards" });
    }

    private static void RollingBoulder(HostSession session, Action<string, bool, string, object?, string[]> add)
    {
        long calls0 = FidelityPatches.BoulderCalls, hits0 = FidelityPatches.BoulderHits;
        List<string> orderAtStart = new List<string>();
        int targetsAtTrigger = -1;
        EpisodeResult r = ScenarioRunner.Run(session, "fidelity:rolling-boulder", async ctx =>
        {
            await ScenarioTools.JumpToEncounter(ctx, "SLIMES_NORMAL");
            CombatState cs = (CombatState)ctx.LocalPlayer.Creature.CombatState!;
            await PowerCmd.Apply<RollingBoulderPower>(new ThrowingPlayerChoiceContext(), ctx.LocalPlayer.Creature, 5m, ctx.LocalPlayer.Creature, null);
            foreach (var c in cs.HittableEnemies)
            {
                orderAtStart.Add(c.CombatId?.ToString() ?? "?");
            }
        }, character: "IRONCLAD", policy: new PreferPolicy(3, Verb.EndTurn), options: new EpisodeOptions { DebugForcedHp = 999 },
        stopWhen: ctx =>
        {
            if (FidelityPatches.BoulderCalls > calls0 && targetsAtTrigger < 0)
            {
                targetsAtTrigger = FidelityPatches.BoulderLastOrder?.Count ?? -1;
            }
            return FidelityPatches.BoulderCalls > calls0 || ctx.MapOpen;
        });
        long calls = FidelityPatches.BoulderCalls - calls0, hits = FidelityPatches.BoulderHits - hits0;
        List<string> order = FidelityPatches.BoulderLastOrder ?? new List<string>();
        bool listOrder = order.Count > 0 && order.SequenceEqual(orderAtStart.Where(order.Contains));
        add("rolling_boulder_per_target", ScenarioRunner.Passed(r) && calls == 1 && hits >= 2 && hits == order.Count && listOrder,
            "the boulder fires once at the next player turn start and calls the game's DoDamage once per hittable enemy (>= 2), sequentially, in HittableEnemies list order",
            new { passed = ScenarioRunner.Passed(r), abort = r.Abort?.Signature, boulder_calls = calls, damage_calls = hits, order, hittable_at_setup = orderAtStart }, new[] { "fidelity-rolling-boulder-per-target" });
    }

    private static void NaturalLoss(HostSession session, Action<string, bool, string, object?, string[]> add)
    {
        object? observed = null;
        bool ok = false;
        for (int attempt = 0; attempt < 6 && !ok; attempt++)
        {
            long onEnded0 = FidelityPatches.NaturalLossOnEnded;
            bool hasRunEnded = false;
            int entries = 0, entriesWithStats = 0;
            Observer obs = new Observer
            {
                OnTerminal = (ctx, res) =>
                {
                    hasRunEnded = ctx.RunManager.HasRunEnded;
                    foreach (IReadOnlyList<MapPointHistoryEntry> act in ctx.Run.MapPointHistory)
                    {
                        foreach (MapPointHistoryEntry e in act)
                        {
                            entries++;
                            PlayerMapPointHistoryEntry pe = e.GetEntry(ctx.LocalPlayer.NetId);
                            if (pe != null && pe.MaxHp > 0)
                            {
                                entriesWithStats++;
                            }
                        }
                    }
                },
            };
            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel("FIDELITY:LOSS:" + attempt), Characters = new[] { "SILENT" } };
            EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)attempt + 11), new EpisodeOptions(), obs);
            long onEnded = FidelityPatches.NaturalLossOnEnded - onEnded0;
            observed = new { attempt, outcome = r.Outcome.ToString(), abort = r.Abort?.Signature, has_run_ended = hasRunEnded, on_ended_calls = onEnded, map_points = entries, map_points_with_stats = entriesWithStats, floor = r.TotalFloor };
            if (r.Outcome == EpisodeOutcome.Loss)
            {
                ok = hasRunEnded && onEnded == 1 && entries > 0 && entriesWithStats == entries;
                break;
            }
        }
        add("natural_loss_bookkeeping", ok,
            "a natural loss calls RunManager.OnEnded(false) exactly once (HasRunEnded = true at the terminal state) and every map-point history entry carries the player's HP / max HP / gold (UpdatePlayerStatsInMapPointHistory before every move and in OnEnded)",
            observed, new[] { "fidelity-natural-death-on-ended", "fidelity-map-history-player-stats" });
    }

    private static void MerchantExit(HostSession session, Action<string, bool, string, object?, string[]> add)
    {
        bool visited = false;
        int inventoriesAfter = -1;
        int unboughtLogged = -1;
        int historyBefore = -1;
        EpisodeResult r = ScenarioRunner.Run(session, "fidelity:merchant-exit", async ctx =>
        {
            await ScenarioTools.JumpToRoom(ctx, RoomType.Shop);
            visited = true;
            historyBefore = HistoryChoices(ctx);
        }, policy: new PreferPolicy(5, Verb.Proceed, Verb.Travel), options: new EpisodeOptions { DebugForcedHp = 999 },
        stopWhen: ctx =>
        {
            if (visited && ctx.CurrentRoom is not MerchantRoom)
            {
                inventoriesAfter = ctx.RunManager.MerchantSynchronizer._inventories.Count;
                unboughtLogged = HistoryChoices(ctx); // OnMerchantExited writes the unbought entries into the entry that was current at the exit
                return true;
            }
            return false;
        });
        add("merchant_exit_bookkeeping", ScenarioRunner.Passed(r) && inventoriesAfter == 0 && unboughtLogged > historyBefore,
            "leaving a shop runs MerchantSynchronizer.OnMerchantExited: the unbought cards / relics / potions are written to the map-point history and the inventories are cleared",
            new { passed = ScenarioRunner.Passed(r), abort = r.Abort?.Signature, inventories_after_exit = inventoriesAfter, history_choices_before = historyBefore, history_choices_after = unboughtLogged }, new[] { "fidelity-merchant-room-exit" });
    }

    /// <summary>Seat 0 takes MEND on seat 1 at its first REST decision, then the first legal action of the preference list, else uniform random.</summary>
    private sealed class MendPolicy : IPolicy
    {
        private readonly RandomLegalPolicy _fallback = new RandomLegalPolicy(3);
        public bool ChoseMend;

        public int Decide(Decision d)
        {
            if (!ChoseMend && d.Kind == DecisionKind.Rest && d.ActingSeat == 0)
            {
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    LegalAction a = d.Actions[i];
                    if (a.Verb == Verb.ChooseOption && a.Source.RestOption is MendRestSiteOption && a.Aux == 1)
                    {
                        ChoseMend = true;
                        return i;
                    }
                }
            }
            return _fallback.Decide(d);
        }
    }

    /// <summary>
    /// Phase H5 (multi-seat only): the local owner's MEND. The game's branch for the local owner is UI targeting code; the
    /// transpiler sends every owner through the waiting branch, which the driver answers with the target the controller
    /// chose. Real branch = a 2-seat run at a rest site, seat 0 mends seat 1.
    /// </summary>
    private static void MendLocalTargeting(HostSession session, Action<string, bool, string, object?, string[]> add)
    {
        long before = FidelityPatches.MendLocalTargetings;
        MendPolicy policy = new MendPolicy();
        int stage = 0, targetBefore = -1, targetAfter = -1, ownerBefore = -1, ownerAfter = -1, missing = -1, amount = -1, mendRows = -1;
        bool sawRest = false;
        Observer obs = new Observer
        {
            OnDecisionHook = (ctx, d) =>
            {
                if (stage == 0 && policy.ChoseMend)
                {
                    // the policy has just chosen MEND at this decision; the action is not applied yet
                    Player target = ctx.Seats[1];
                    targetBefore = target.Creature.CurrentHp;
                    ownerBefore = ctx.Seats[0].Creature.CurrentHp;
                    missing = target.Creature.MaxHp - targetBefore;
                    amount = (int)MendRestSiteOption.GetHealAmount(target);
                    mendRows = d.Actions.Count(x => x.Verb == Verb.ChooseOption && x.Source.RestOption is MendRestSiteOption);
                    stage = 1;
                }
                else if (stage == 1)
                {
                    // the next decision point: the option has finished
                    targetAfter = ctx.Seats[1].Creature.CurrentHp;
                    ownerAfter = ctx.Seats[0].Creature.CurrentHp;
                    stage = 2;
                }
            },
        };
        EpisodeSpec spec = new EpisodeSpec
        {
            Seed = HostSession.SeedFromLabel("FIDELITY:MEND"),
            Characters = new[] { "IRONCLAD", "SILENT" },
            Scenario = new Scenario
            {
                Name = "fidelity:mend-local-targeting",
                Setup = async ctx =>
                {
                    for (int s = 0; s < ctx.SeatCount; s++)
                    {
                        ScenarioTools.SetHp(ctx, Math.Max(1, ctx.Seats[s].Creature.MaxHp / 3), null, s);
                    }
                    await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite);
                    sawRest = ctx.CurrentRoom is RestSiteRoom;
                },
                StopWhen = _ => stage == 2,
            },
        };
        EpisodeResult r = session.RunEpisode(spec, policy, new EpisodeOptions { MaxPolicyDecisions = 200 }, obs);
        long calls = FidelityPatches.MendLocalTargetings - before;
        int gained = targetAfter - targetBefore;
        bool ok = ScenarioRunner.Passed(r) && sawRest && stage == 2 && calls == 1 && mendRows == 1 && gained > 0 && gained == Math.Min(amount, missing) && ownerAfter == ownerBefore;
        add("mend_local_targeting", ok,
            "2 seats at a rest site: seat 0's REST frame has one MEND row (target seat 1); choosing it runs MendRestSiteOption.OnSelect through the waiting branch exactly once " +
            "(LocalContext.IsMe replaced, counter + 1), the target's HP rises by the target's own rest amount (capped at its missing HP) and the owner's HP does not change",
            new { passed = ScenarioRunner.Passed(r), outcome = r.Outcome.ToString(), truncated_by = r.TruncatedBy, abort = r.Abort?.Signature, saw_rest_site = sawRest, stage, waiting_branch_calls = calls, mend_rows = mendRows,
                  target_hp_before = targetBefore, target_hp_after = targetAfter, target_missing_hp = missing, target_rest_amount = amount, owner_hp_before = ownerBefore, owner_hp_after = ownerAfter },
            new[] { "fidelity-mend-local-targeting" });
    }

    private static int HistoryChoices(RunContext ctx)
    {
        int n = 0;
        foreach (IReadOnlyList<MapPointHistoryEntry> act in ctx.Run.MapPointHistory)
        {
            foreach (MapPointHistoryEntry e in act)
            {
                PlayerMapPointHistoryEntry? pe = e.GetEntry(ctx.LocalPlayer.NetId);
                if (pe != null)
                {
                    n += pe.CardChoices.Count + pe.RelicChoices.Count + pe.PotionChoices.Count;
                }
            }
        }
        return n;
    }

    private static void ActSelectionAndTutorial(Action<string, bool, string, object?, string[]> add)
    {
        // E1 act selection: the real branch forces an unlocked, non-default act that is not discovered (no RNG draw).
        ProgressState progress = SaveManager.Instance.Progress;
        UnlockState unlocks = SaveManager.Instance.GenerateUnlockStateFromProgress();
        ActModel? target = null;
        foreach (IReadOnlyList<ActModel> slot in ModelDb.ActsByIndex)
        {
            foreach (ActModel act in slot)
            {
                if (!act.IsDefault && act.IsUnlocked(unlocks))
                {
                    target = act;
                    break;
                }
            }
            if (target != null)
            {
                break;
            }
        }
        List<string> withAll = new List<string>(), withHidden = new List<string>();
        bool forced = false;
        if (target != null)
        {
            for (int s = 0; s < 8; s++)
            {
                withAll.Add(ActModel.GetRandomList(new Rng((ulong)(s + 1), "act_selection"), unlocks, false).First(x => ModelDb.ActsByIndex.Any(sl => sl.Contains(x) && sl.Contains(target))).Id.Entry);
            }
            progress._discoveredActs.Remove(target.Id);
            try
            {
                forced = true;
                for (int s = 0; s < 8; s++)
                {
                    string got = ActModel.GetRandomList(new Rng((ulong)(s + 1), "act_selection"), unlocks, false).First(x => ModelDb.ActsByIndex.Any(sl => sl.Contains(x) && sl.Contains(target))).Id.Entry;
                    withHidden.Add(got);
                    forced &= got == target.Id.Entry;
                }
            }
            finally
            {
                progress.MarkActAsSeen(target.Id);
            }
        }
        add("act_selection_real_branch", target != null && forced && withAll.Distinct().Count() > 1 && progress.DiscoveredActs.Contains(target!.Id),
            "ActModel.GetRandomList takes the real branch: with every act discovered the slot is random across seeds; with one undiscovered act it is forced for every seed (restored afterwards)",
            new { act = target?.Id.Entry, all_discovered = withAll, one_undiscovered = withHidden }, new[] { "fidelity-act-selection-discovery-gate" });

        // E3 tutorial room: with a fresh profile (NumberOfRuns == 0) the real branch turns the first '?' into an Event.
        long forcedBefore = FidelityGatePatches.TutorialRoomForced;
        bool veteran = RunManager.Instance.TryGetRoomTypeForTutorial(MapPointType.Unassigned, out RoomType vetType);
        ProgressState saved = SaveManager.Instance.Progress;
        bool fresh;
        RoomType freshType;
        try
        {
            SaveManager.Instance.Progress = ProgressState.CreateDefault();
            fresh = RunManager.Instance.TryGetRoomTypeForTutorial(MapPointType.Unassigned, out freshType);
        }
        finally
        {
            SaveManager.Instance.Progress = saved;
        }
        long counted = FidelityGatePatches.TutorialRoomForced - forcedBefore;
        add("tutorial_room_real_branch", !veteran && fresh && freshType == RoomType.Event && counted == 1,
            "RunManager.TryGetRoomTypeForTutorial takes the real branch: false for the veteran profile, Event for a fresh profile (the Core no-op counter sees it once; outside any episode)",
            new { veteran, veteran_type = vetType.ToString(), fresh_profile = fresh, fresh_type = freshType.ToString(), counter_delta = counted }, new[] { "fidelity-tutorial-room-type-gate" });
    }

    private static void Scheduling(HostSession session, Action<string, bool, string, object?, string[]> add)
    {
        object? unpause = null;
        bool unpauseOk = false;
        object? turnLoop = null;
        bool turnLoopOk = false;
        EpisodeResult r = ScenarioRunner.Run(session, "fidelity:scheduling", async ctx =>
        {
            ActionExecutor ex = ctx.RunManager.ActionExecutor;
            bool wasPaused = ex.IsPaused;
            long wakes0 = FidelityPatches.UnpauseWakeups;
            int timers0 = ctx.Pump.Clock.Pending;
            ex.Pause();
            Task w = ex.WaitForUnpause();
            bool pendingAfterWait = !w.IsCompleted && FidelityPatches.HasPendingUnpause(ex);
            int timersWhilePaused = ctx.Pump.Clock.Pending - timers0;
            ex.Unpause();
            await w;
            if (wasPaused)
            {
                ex.Pause();
            }
            unpauseOk = pendingAfterWait && timersWhilePaused == 0 && FidelityPatches.UnpauseWakeups - wakes0 == 1 && w.IsCompletedSuccessfully;
            unpause = new { pending_while_paused = pendingAfterWait, virtual_timers_created = timersWhilePaused, wakeups = FidelityPatches.UnpauseWakeups - wakes0, completed = w.IsCompletedSuccessfully };

            CombatManager cm = ctx.CombatManager;
            long waits0 = FidelityPatches.PreviousTurnLoopWaits, timeouts0 = FidelityPatches.PreviousTurnLoopTimeouts;
            TaskCompletionSource prev = new TaskCompletionSource();
            int t0 = ctx.Pump.Clock.Pending;
            Task w1 = cm.WaitForPreviousTurnLoopToFinish(prev.Task, null);
            int virtualTimer = ctx.Pump.Clock.Pending - t0;
            bool waiting = !w1.IsCompleted;
            prev.SetResult();
            await w1;
            ulong vt0 = ctx.Pump.Clock.NowMs;
            Task w2 = cm.WaitForPreviousTurnLoopToFinish(new TaskCompletionSource().Task, null);
            await w2; // completes when the virtual 10 s timeout fires (the game catches the TimeoutException and logs it)
            ulong advanced = ctx.Pump.Clock.NowMs - vt0;
            turnLoopOk = waiting && virtualTimer == 1 && FidelityPatches.PreviousTurnLoopWaits - waits0 == 2 && FidelityPatches.PreviousTurnLoopTimeouts - timeouts0 == 1 && advanced >= 10_000;
            turnLoop = new { waiting_on_live_loop = waiting, virtual_timers_created = virtualTimer, waits = FidelityPatches.PreviousTurnLoopWaits - waits0, timeouts = FidelityPatches.PreviousTurnLoopTimeouts - timeouts0, virtual_ms_advanced = advanced };
        }, stopWhen: _ => true);
        add("executor_unpause_tcs", ScenarioRunner.Passed(r) && unpauseOk,
            "a paused ActionExecutor waits on a task (no timer, virtual or real) that the next Unpause completes",
            new { passed = ScenarioRunner.Passed(r), abort = r.Abort?.Signature, unpause }, new[] { "fidelity-executor-unpause-signal", "fidelity-executor-unpause-wake" });
        add("turn_loop_wait_virtual_timeout", ScenarioRunner.Passed(r) && turnLoopOk,
            "WaitForPreviousTurnLoopToFinish waits for a live previous turn loop on one virtual timer (no real timer); a loop that never ends times out after 10 virtual seconds at ~0 wall time",
            new { passed = ScenarioRunner.Passed(r), abort = r.Abort?.Signature, wall_ms = Math.Round(r.WallMs, 1), turn_loop = turnLoop }, new[] { "fidelity-turn-loop-wait-virtual-timeout" });
    }

    private static void Assertions(HostSession session, Action<string, bool, string, object?, string[]> add)
    {
        void Negative(string name, string id, string patch, Func<RunContext, Task> setup)
        {
            long fired0 = FidelityAssert.Fired;
            EpisodeResult r = ScenarioRunner.Run(session, "fidelity:assert:" + name, setup, stopWhen: _ => true);
            bool aborted = r.Outcome == EpisodeOutcome.Aborted && r.Abort!.Detail == "fidelity_assert:" + id;
            add("assert_" + name, aborted && FidelityAssert.Fired - fired0 == 1,
                "forcing the path fails the episode with fidelity_assert:" + id,
                new { observed = r.Outcome == EpisodeOutcome.Aborted ? r.Abort!.Reason + "/" + r.Abort.Detail : r.Outcome.ToString(), r.Abort?.Message }, new[] { patch });
        }

        Negative("creature_spawn_wait", "creature_spawn_wait", "fidelity-assert-creature-spawn-wait", async ctx =>
        {
            await ScenarioTools.JumpToEncounter(ctx, "NIBBITS_WEAK");
            CombatState cs = (CombatState)ctx.LocalPlayer.Creature.CombatState!;
            await cs.GetCreatureAsync(999_999u, 10.0);
        });
        Negative("test_energy_cost", "test_energy_cost_override", "fidelity-guard-test-energy-cost-sneckooil", ctx =>
        {
            SneckoOil oil = (SneckoOil)ScenarioTools.Potion("SNECKO_OIL").ToMutable();
            oil.TestEnergyCostOverride = 1;
            return Task.CompletedTask;
        });
        Negative("rest_hover_delay", "rest_hover_delay", "fidelity-assert-rest-hover-delay", ctx =>
        {
            ctx.RunManager.RestSiteSynchronizer.TrySendHoverMessage();
            return Task.CompletedTask;
        });

        // Verifier finding F7: the run-level switches are checked at every quiescent point, not only when the run is set up. The
        // AutoSlayer check is switched on in the middle of an episode (it has no effect on game code while TestMode is on:
        // NonInteractiveMode.IsActive consults it only when TestMode is off) and the episode must end at the next quiescent point.
        {
            Func<bool> original = NonInteractiveMode.AutoSlayerCheck;
            long fired0 = FidelityAssert.Fired;
            EpisodeResult r;
            try
            {
                r = ScenarioRunner.Run(session, "fidelity:assert:autoslayer_mid_episode", ctx =>
                {
                    NonInteractiveMode.AutoSlayerCheck = () => true;
                    return Task.CompletedTask;
                }, stopWhen: _ => true);
            }
            finally
            {
                NonInteractiveMode.AutoSlayerCheck = original;
            }
            add("run_switches_are_checked_at_quiescent_points", r.Outcome == EpisodeOutcome.Aborted && r.Abort!.Detail == "fidelity_assert:autoslayer_off" && FidelityAssert.Fired - fired0 == 1,
                "AutoSlayerCheck switched on after the run was set up fails the episode with fidelity_assert:autoslayer_off at the next quiescent point",
                new { observed = r.Outcome == EpisodeOutcome.Aborted ? r.Abort!.Reason + "/" + r.Abort.Detail : r.Outcome.ToString(), r.Abort?.Message }, new[] { "fidelity-run-invariants" });
        }

        // Verifier finding R-8: an assertion that fires while the run is being set up and whose exception is swallowed (here: counted
        // without a throw, from a lifecycle that runs inside the set-up) must fail the episode at its first quiescent point.
        {
            SetupAssertProbe probe = new SetupAssertProbe();
            session.Registry.SetLifecycle(probe);
            EpisodeResult r;
            try
            {
                r = session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("FIDELITY:SETUP-ASSERT"), Characters = new[] { "DEFECT" } }, new RandomLegalPolicy(9), new EpisodeOptions { MaxPolicyDecisions = 40 });
            }
            finally
            {
                session.Registry.RemoveLifecycle(probe.Name);
            }
            add("assert_swallowed_during_run_setup_fails_the_episode", probe.Fired == 1 && r.Outcome == EpisodeOutcome.Aborted && r.Abort!.Detail == "fidelity_assert_swallowed" && r.Counters.PolicyDecisions == 0,
                "an assertion counted (not thrown) inside OnRunSetUp aborts the episode with fidelity_assert_swallowed before the first policy decision",
                new { observed = r.Outcome == EpisodeOutcome.Aborted ? r.Abort!.Reason + "/" + r.Abort.Detail : r.Outcome.ToString(), policy_decisions = r.Counters.PolicyDecisions, r.Abort?.Message }, new[] { "fidelity-assert-backstop" });
        }

        // the session is clean afterwards: the same seed twice gives the same trace
        EpisodeSpec Spec() => new EpisodeSpec { Seed = HostSession.SeedFromLabel("FIDELITY:CLEAN"), Characters = new[] { "DEFECT" } };
        EpisodeOptions opt = new EpisodeOptions { ComputeSignatures = true, MaxPolicyDecisions = 400 };
        EpisodeResult a1 = session.RunEpisode(Spec(), new RandomLegalPolicy(9), opt);
        EpisodeResult a2 = session.RunEpisode(Spec(), new RandomLegalPolicy(9), new EpisodeOptions { ComputeSignatures = true, MaxPolicyDecisions = 400 });
        add("clean_after_asserts", a1.Outcome != EpisodeOutcome.Aborted && a1.TraceHash == a2.TraceHash && a1.Outcome == a2.Outcome,
            "after the negative tests the same seed + policy seed twice gives identical traces and no abort",
            new { outcome = a1.Outcome.ToString(), abort = a1.Abort?.Signature, same_trace = a1.TraceHash == a2.TraceHash }, Array.Empty<string>());
    }

    /// <summary>A lifecycle that "fires and swallows" one fidelity assertion while the run is being set up (R-8 regression test).</summary>
    private sealed class SetupAssertProbe : IRunLifecycle
    {
        public int Fired;

        public string Name => "selftest:setup-assert";

        public void OnRunSetUp(RunContext ctx)
        {
            FidelityAssert.Count("selftest_assert_during_run_setup");
            FidelityAssert.Last = "selftest_assert_during_run_setup: raised by the self-test inside OnRunSetUp";
            Fired++;
        }

        public void OnRoomEntered(RunContext ctx)
        {
        }

        public void OnEpisodeEnd(RunContext ctx)
        {
        }
    }

    private static readonly string[] PatchedMethods =
    {
        "MerchantPotionEntry.CalcCost", "CallingBell.GenerateRewards", "Cauldron.GenerateRewards", "MerchantRoom.Exit", "RunManager.UpdatePlayerStatsInMapPointHistory",
        "RollingBoulderPower", "CreatureCmd.Kill", "ActionExecutor.WaitForUnpause", "CombatManager.WaitForPreviousTurnLoopToFinish", "GameAction.Execute",
        "ActModel.ApplyDiscoveryOrderModifications", "FidelityPatches", "FidelityInvariants",
    };

    private static void Fuzz(HostSession session, int runs, Action<string, bool, string, object?, string[]> add)
    {
        long tut0 = FidelityGatePatches.TutorialRoomForced, boss0 = FidelityGatePatches.DiscoveryBossChanged, calls0 = FidelityGatePatches.DiscoveryOrderCalls;
        long roomsChanged0 = FidelityGatePatches.DiscoveryRoomsChanged, asserts0 = FidelityAssert.Fired, bad0 = FidelityPatches.ActionFinishedBadState;
        SortedDictionary<string, int> aborts = new SortedDictionary<string, int>(StringComparer.Ordinal);
        List<string> mineAborts = new List<string>();
        int losses = 0, wins = 0, aborted = 0;
        for (int i = 0; i < runs; i++)
        {
            string character = DriverCommands.AllCharacters[i % DriverCommands.AllCharacters.Length];
            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel("FIDELITY:FUZZ:" + i), Characters = new[] { character }, Ascension = i % 2 == 0 ? 0 : 10 };
            EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)(1000 + i)), new EpisodeOptions());
            if (r.Outcome == EpisodeOutcome.Loss)
            {
                losses++;
            }
            else if (r.Outcome == EpisodeOutcome.Win)
            {
                wins++;
            }
            else if (r.Outcome == EpisodeOutcome.Aborted)
            {
                aborted++;
                string sig = r.Abort!.Signature;
                aborts[sig] = aborts.GetValueOrDefault(sig) + 1;
                string frames = string.Join(" ", r.Abort.Frames);
                string first = r.Abort.Frames.Count > 0 ? r.Abort.Frames[0] : "";
                if (r.Abort.Detail.StartsWith("fidelity_assert", StringComparison.Ordinal) || PatchedMethods.Any(m => first.Contains(m, StringComparison.Ordinal)))
                {
                    mineAborts.Add(sig + " | " + frames);
                }
            }
        }
        long tut = FidelityGatePatches.TutorialRoomForced - tut0, boss = FidelityGatePatches.DiscoveryBossChanged - boss0, calls = FidelityGatePatches.DiscoveryOrderCalls - calls0;
        long roomsChanged = FidelityGatePatches.DiscoveryRoomsChanged - roomsChanged0;
        add("veteran_gates_noop_fuzz", runs == 0 || (tut == 0 && boss == 0 && roomsChanged == 0 && calls >= runs),
            "across the fuzz the real tutorial / discovery branches run (ApplyDiscoveryOrderModifications >= 1 per run; boss and whole room set fingerprinted at every call) and change nothing: tutorial room forced 0, boss changed 0, room set changed 0",
            new { runs, discovery_order_calls = calls, tutorial_room_forced = tut, discovery_boss_changed = boss, discovery_rooms_changed = roomsChanged },
            new[] { "fidelity-force-discovery-order", "fidelity-discovery-order-noop-check", "fidelity-tutorial-room-type-noop-check" });
        add("fuzz_no_abort_in_fidelity_area", mineAborts.Count == 0 && FidelityAssert.Fired == asserts0 && FidelityPatches.ActionFinishedBadState == bad0,
            "no fuzz abort whose detail is a fidelity assertion or whose first game frame is a method this pack patches; GameAction.cs:102 monitor stays 0",
            new { runs, losses, wins, aborted, aborts_by_signature = aborts, mine = mineAborts, asserts_fired = FidelityAssert.Fired - asserts0, action_finished_bad_state = FidelityPatches.ActionFinishedBadState - bad0 },
            new[] { "fidelity-monitor-action-finished-bad-state" });
    }

    private static void CoverageCheck(List<Check> checks, Action<string, bool, string, object?, string[]> add)
    {
        HashSet<string> ran = new HashSet<string>(checks.Where(c => c.Pass).Select(c => c.Name), StringComparer.Ordinal);
        List<string> missing = FidelityCheckOf.Where(kv => PatchRegistry.Records.Any(r => r.Id == kv.Key) && !ran.Contains(kv.Value)).Select(kv => kv.Key + " -> " + kv.Value).ToList();
        add("fidelity_patches_exercised", missing.Count == 0, "every Fidelity-category patch's check ran and passed", new { missing }, Array.Empty<string>());
    }
}
