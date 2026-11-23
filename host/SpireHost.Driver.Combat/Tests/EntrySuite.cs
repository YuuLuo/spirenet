using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Enchantments;

namespace SpireHost.Driver.Combat.Tests;

/// <summary>
/// "entries": every public CardSelectCmd.From* overload that asks an ICardSelector (15 overloads, 12 names) is
/// called DIRECTLY - with the arguments game code passes, from inside a scenario - once per rule that the real
/// screen of that entry point enforces: forced pick / skip button, exact count, range with manual confirmation,
/// "any number", cancelable deck screens, fewer candidates than the minimum, a caller-supplied filter, the
/// comparison of the simple grid, and the cases the game answers itself (no decision at all).
///
/// The "sources" suite reaches the entry points through the game objects that call them; this suite pins the
/// rule matrix down independently of which cards / relics happen to exist:
///   - the frame's published bounds, verbs and exits (probe checks + the expectations below);
///   - what CardSelectCmd returns to its caller = exactly the instances the frame handed over, in pick order;
///   - prompts that the game does not ask (n &lt;= MinSelect without manual confirmation) produce no frame.
/// The choice context is BlockingPlayerChoiceContext (what events, relic pick-ups and KnowledgeDemon use): it
/// neither pauses an action nor creates a hook action, so the call can be made from the scenario task.
/// </summary>
public static class EntrySuite
{
    private sealed class Plan
    {
        public readonly List<int> Picks = new List<int>();   // option ordinals, in pick order
        public Verb Finish = Verb.Invalid;                    // ConfirmSelection / Skip / Cancel; Invalid = completes with the last pick
    }

    private sealed class Expect
    {
        public bool NoPrompt;                                 // the game answers itself: no CARD_SELECT frame
        public CardSelectKind Kind;
        public CardSelectEntry Entry;
        public int Min;
        public int Max;
        public bool Cancelable;
        public bool CanSkip;
        public bool Manual;
        public string FirstVerbs = "";
        public string End = "";                               // Auto / ConfirmSelection / Skip / Cancel
        public int ResultCount;
        public int Options = -1;                              // -1 = not asserted
        public int Instances = -1;
        public bool Merged;
        public PileType? Pile;
        public string Prompt = "";                            // "table/key"; empty = not asserted
        public Func<RunContext, SelectRecord?, IReadOnlyList<CardModel>, string?>? Check;
    }

    private sealed class Case
    {
        public string Name = "";
        public bool InCombat;
        public Func<RunContext, Task<IReadOnlyList<CardModel>>> Call = null!;
        public Plan Plan = new Plan();
        public Expect Expect = new Expect();
    }

    private static IReadOnlyList<CardModel> One(CardModel? card) => card == null ? Array.Empty<CardModel>() : new[] { card };

    private static async Task<IReadOnlyList<CardModel>> Many(Task<IEnumerable<CardModel>> task) => (await task).ToList();

    private static Plan P(Verb finish, params int[] picks)
    {
        Plan p = new Plan { Finish = finish };
        p.Picks.AddRange(picks);
        return p;
    }

    /// <summary>Fresh, unowned-by-any-pile cards as a card factory would hand them to a "choose" screen.</summary>
    private static List<CardModel> Generated(RunContext ctx, params string[] ids)
    {
        Player player = ctx.LocalPlayer;
        List<CardModel> cards = new List<CardModel>();
        foreach (string id in ids)
        {
            CardModel canonical = ScenarioTools.Card(id);
            cards.Add(ctx.CombatManager.IsInProgress ? ScenarioTools.CombatState(ctx).CreateCard(canonical, player) : ctx.Run.CreateCard(canonical, player));
        }
        return cards;
    }

    private static PlayerChoiceContext Blocking() => new BlockingPlayerChoiceContext();

    private static List<Case> Cases()
    {
        List<Case> cases = new List<Case>();
        List<CardModel> given = new List<CardModel>();        // what the case handed to the entry point (for Check)
        List<CardModel> snapshot = new List<CardModel>();     // pile / hand as it was when the entry point was called

        // ------------------------------------------------------------------ FromChooseACardScreen (:48)
        cases.Add(new Case
        {
            Name = "FromChooseACardScreen forced (canSkip false, 3 cards)",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "ARMAMENTS", "TRUE_GRIT");
                return One(await CardSelectCmd.FromChooseACardScreen(Blocking(), given, ctx.LocalPlayer));
            },
            Plan = P(Verb.Invalid, 2),
            Expect = new Expect
            {
                Kind = CardSelectKind.ChooseACard, Entry = CardSelectEntry.ChooseACardScreen, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1,
                Options = 3, Instances = 3, Prompt = "gameplay_ui/CHOOSE_CARD_HEADER",
                Check = (ctx, rec, result) => ReferenceEquals(result[0], given[2]) ? null : "the third offered card was picked but another instance came back",
            },
        });
        cases.Add(new Case
        {
            Name = "FromChooseACardScreen canSkip -> SKIP",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "ARMAMENTS", "TRUE_GRIT");
                return One(await CardSelectCmd.FromChooseACardScreen(Blocking(), given, ctx.LocalPlayer, canSkip: true));
            },
            Plan = P(Verb.Skip),
            Expect = new Expect { Kind = CardSelectKind.ChooseACard, Entry = CardSelectEntry.ChooseACardScreen, Min = 0, Max = 1, CanSkip = true, FirstVerbs = "SelectCard+Skip", End = "Skip", ResultCount = 0, Options = 3 },
        });
        cases.Add(new Case
        {
            Name = "FromChooseACardScreen canSkip -> pick",
            InCombat = true,
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "ARMAMENTS");
                return One(await CardSelectCmd.FromChooseACardScreen(Blocking(), given, ctx.LocalPlayer, canSkip: true));
            },
            Plan = P(Verb.Invalid, 0),
            Expect = new Expect
            {
                Kind = CardSelectKind.ChooseACard, Entry = CardSelectEntry.ChooseACardScreen, Min = 0, Max = 1, CanSkip = true, FirstVerbs = "SelectCard+Skip", End = "Auto", ResultCount = 1, Options = 2,
                Check = (ctx, rec, result) => ReferenceEquals(result[0], given[0]) ? null : "wrong instance",
            },
        });
        cases.Add(new Case
        {
            Name = "FromChooseACardScreen one card, forced (single action, auto-advanced)",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT");
                return One(await CardSelectCmd.FromChooseACardScreen(Blocking(), given, ctx.LocalPlayer));
            },
            Plan = P(Verb.Invalid, 0),
            Expect = new Expect { Kind = CardSelectKind.ChooseACard, Entry = CardSelectEntry.ChooseACardScreen, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Options = 1 },
        });

        // ------------------------------------------------------------------ FromSimpleGrid (:165)
        cases.Add(new Case
        {
            Name = "FromSimpleGrid exactly 2 of 4 (pick order kept)",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "ARMAMENTS", "TRUE_GRIT", "BURNING_PACT");
                return await Many(CardSelectCmd.FromSimpleGrid(Blocking(), given, ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, 2)));
            },
            Plan = P(Verb.Invalid, 3, 1),
            Expect = new Expect
            {
                Kind = CardSelectKind.SimpleGrid, Entry = CardSelectEntry.SimpleGrid, Min = 2, Max = 2, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 2, Options = 4, Instances = 4,
                Prompt = "card_selection/TO_TRANSFORM",
                Check = (ctx, rec, result) => ReferenceEquals(result[0], given[3]) && ReferenceEquals(result[1], given[1]) ? null : "the result is not [4th, 2nd] in pick order",
            },
        });
        cases.Add(new Case
        {
            Name = "FromSimpleGrid 0..2 -> one pick, CONFIRM",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "ARMAMENTS", "TRUE_GRIT");
                return await Many(CardSelectCmd.FromSimpleGrid(Blocking(), given, ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, 0, 2)));
            },
            Plan = P(Verb.ConfirmSelection, 1),
            Expect = new Expect { Kind = CardSelectKind.SimpleGrid, Entry = CardSelectEntry.SimpleGrid, Min = 0, Max = 2, CanSkip = true, Manual = true, FirstVerbs = "SelectCard+Skip", End = "ConfirmSelection", ResultCount = 1 },
        });
        cases.Add(new Case
        {
            Name = "FromSimpleGrid 0..2 -> SKIP",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "ARMAMENTS", "TRUE_GRIT");
                return await Many(CardSelectCmd.FromSimpleGrid(Blocking(), given, ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, 0, 2)));
            },
            Plan = P(Verb.Skip),
            Expect = new Expect { Kind = CardSelectKind.SimpleGrid, Entry = CardSelectEntry.SimpleGrid, Min = 0, Max = 2, CanSkip = true, Manual = true, FirstVerbs = "SelectCard+Skip", End = "Skip", ResultCount = 0 },
        });
        cases.Add(new Case
        {
            Name = "FromSimpleGrid n <= MinSelect without confirmation: the game answers itself",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "ARMAMENTS", "TRUE_GRIT");
                return await Many(CardSelectCmd.FromSimpleGrid(Blocking(), given, ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, 3)));
            },
            Expect = new Expect { NoPrompt = true, ResultCount = 3 },
        });
        cases.Add(new Case
        {
            Name = "FromSimpleGrid with CardSelectorPrefs.Comparison (displayed order = sorted)",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "TRUE_GRIT", "ARMAMENTS", "BURNING_PACT");
                CardSelectorPrefs prefs = new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, 1) { Comparison = (a, b) => string.CompareOrdinal(a.Id.Entry, b.Id.Entry) };
                return await Many(CardSelectCmd.FromSimpleGrid(Blocking(), given, ctx.LocalPlayer, prefs));
            },
            Plan = P(Verb.Invalid, 0),
            Expect = new Expect
            {
                Kind = CardSelectKind.SimpleGrid, Entry = CardSelectEntry.SimpleGrid, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Options = 4,
                Check = (ctx, rec, result) => result[0].Id.Entry == "ARMAMENTS" ? null : "option 0 of the sorted grid is " + result[0].Id.Entry + ", expected ARMAMENTS",
            },
        });
        // No caller in v0.113 produces this (0 occurrences in sweeps and fuzz): manual confirmation, exactly 3,
        // only 2 candidates. The game does not answer itself (CardSelectCmd.cs:185 needs !RequireManualConfirmation),
        // the real grid can never enable its confirm button (NSimpleCardSelectScreen.cs:118-124) and has no close
        // button: it would soft-lock. The frame takes what there is, counts it and logs an error line
        // (driver-spec-combat.md 4.4-2); this case pins that rule down.
        long softlocksBefore = 0;
        cases.Add(new Case
        {
            Name = "FromSimpleGrid manual, exactly 3 of only 2 candidates: the real screen would soft-lock, the frame takes both and counts it",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "ARMAMENTS");
                softlocksBefore = CardSelectStats.SoftlockAvoided;
                return await Many(CardSelectCmd.FromSimpleGrid(Blocking(), given, ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, 3) { RequireManualConfirmation = true }));
            },
            Plan = P(Verb.ConfirmSelection, 1, 0),
            Expect = new Expect
            {
                Kind = CardSelectKind.SimpleGrid, Entry = CardSelectEntry.SimpleGrid, Min = 2, Max = 2, Manual = true, FirstVerbs = "SelectCard", End = "ConfirmSelection", ResultCount = 2, Options = 2, Instances = 2,
                Check = (ctx, rec, result) => CardSelectStats.SoftlockAvoided != softlocksBefore + 1 ? "CardSelectStats.SoftlockAvoided did not count the prompt"
                    : ReferenceEquals(result[0], given[1]) && ReferenceEquals(result[1], given[0]) ? null : "the result is not [2nd, 1st] in pick order",
            },
        });

        // ------------------------------------------------------------------ FromSimpleGridForRewards (:105)
        cases.Add(new Case
        {
            Name = "FromSimpleGridForRewards 1 of 3",
            Call = async ctx =>
            {
                given = Generated(ctx, "HEADBUTT", "ARMAMENTS", "TRUE_GRIT");
                List<CardCreationResult> results = given.Select(c => new CardCreationResult(c)).ToList();
                return await Many(CardSelectCmd.FromSimpleGridForRewards(Blocking(), results, ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, 1)));
            },
            Plan = P(Verb.Invalid, 1),
            Expect = new Expect
            {
                Kind = CardSelectKind.SimpleGridRewards, Entry = CardSelectEntry.SimpleGridForRewards, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Options = 3,
                Check = (ctx, rec, result) => ReferenceEquals(result[0], given[1]) ? null : "wrong instance",
            },
        });

        // ------------------------------------------------------------------ FromCombatPile (:223 and :228)
        cases.Add(new Case
        {
            Name = "FromCombatPile(discard) exactly 1: classes, first instance of the class",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                List<string> starters = CombatTestTools.StarterIds(ctx);
                await ScenarioTools.AddCard(ctx, starters[0], PileType.Discard);
                await ScenarioTools.AddCard(ctx, "HEADBUTT", PileType.Discard, upgrades: 1);
                await ScenarioTools.AddCard(ctx, starters[0], PileType.Discard);
                await ScenarioTools.AddCard(ctx, starters[1], PileType.Discard);
                snapshot = PileType.Discard.GetPile(p).Cards.ToList();
                return await Many(CardSelectCmd.FromCombatPile(Blocking(), PileType.Discard.GetPile(p), p, new CardSelectorPrefs(CardSelectorPrefs.ExhaustSelectionPrompt, 1)));
            },
            Plan = P(Verb.Invalid, 0),
            Expect = new Expect
            {
                Kind = CardSelectKind.CombatPile, Entry = CardSelectEntry.CombatPile, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Options = 3, Instances = 4, Merged = true,
                Pile = PileType.Discard, Prompt = "card_selection/TO_EXHAUST",
                Check = (ctx, rec, result) =>
                {
                    string key = CardClassKey.Of(result[0]);
                    return ReferenceEquals(snapshot.First(c => CardClassKey.Of(c) == key), result[0]) ? null : "the picked instance is not the first of its class in pile order";
                },
            },
        });
        cases.Add(new Case
        {
            Name = "FromCombatPile(draw, filter) exactly 2 from one class: two different instances",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                for (int i = 0; i < 3; i++)
                {
                    await ScenarioTools.AddCard(ctx, "HEADBUTT", PileType.Draw);
                }
                await ScenarioTools.AddCard(ctx, "ARMAMENTS", PileType.Draw);
                snapshot = PileType.Draw.GetPile(p).Cards.ToList();
                return await Many(CardSelectCmd.FromCombatPile(Blocking(), PileType.Draw.GetPile(p), p, new CardSelectorPrefs(CardSelectorPrefs.ExhaustSelectionPrompt, 2),
                    c => c.Id.Entry == "HEADBUTT" || c.Id.Entry == "ARMAMENTS"));
            },
            // Class order = key order: rarity, then id. Both picks go to the class that has 3 instances.
            Plan = P(Verb.Invalid, -1, -1),
            Expect = new Expect
            {
                Kind = CardSelectKind.CombatPile, Entry = CardSelectEntry.CombatPileFiltered, Min = 2, Max = 2, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 2, Options = 2, Instances = 4, Merged = true,
                Pile = PileType.Draw,
                Check = (ctx, rec, result) =>
                {
                    if (result[0].Id.Entry != "HEADBUTT" || result[1].Id.Entry != "HEADBUTT" || ReferenceEquals(result[0], result[1]))
                    {
                        return "expected two different HEADBUTT instances";
                    }
                    List<CardModel> inPile = snapshot.Where(c => c.Id.Entry == "HEADBUTT").ToList();
                    return ReferenceEquals(result[0], inPile[0]) && ReferenceEquals(result[1], inPile[1]) ? null : "the instances are not the first two of the class in pile order";
                },
            },
        });
        cases.Add(new Case
        {
            Name = "FromCombatPile(exhaust) 3..5 with only 2 candidates: effective [2, 2] (min(Min, n))",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                await ScenarioTools.AddCard(ctx, "HEADBUTT", PileType.Exhaust);
                await ScenarioTools.AddCard(ctx, "ARMAMENTS", PileType.Exhaust);
                return await Many(CardSelectCmd.FromCombatPile(Blocking(), PileType.Exhaust.GetPile(p), p, new CardSelectorPrefs(CardSelectorPrefs.ExhaustSelectionPrompt, 3, 5)));
            },
            Plan = P(Verb.ConfirmSelection, 0, 1),
            Expect = new Expect { Kind = CardSelectKind.CombatPile, Entry = CardSelectEntry.CombatPile, Min = 2, Max = 2, Manual = true, FirstVerbs = "SelectCard", End = "ConfirmSelection", ResultCount = 2, Options = 2, Instances = 2, Pile = PileType.Exhaust },
        });
        cases.Add(new Case
        {
            Name = "FromCombatPile(discard) 0..2 -> SKIP",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                await ScenarioTools.AddCard(ctx, "HEADBUTT", PileType.Discard);
                await ScenarioTools.AddCard(ctx, "ARMAMENTS", PileType.Discard);
                await ScenarioTools.AddCard(ctx, "TRUE_GRIT", PileType.Discard);
                return await Many(CardSelectCmd.FromCombatPile(Blocking(), PileType.Discard.GetPile(p), p, new CardSelectorPrefs(CardSelectorPrefs.ExhaustSelectionPrompt, 0, 2)));
            },
            Plan = P(Verb.Skip),
            Expect = new Expect { Kind = CardSelectKind.CombatPile, Entry = CardSelectEntry.CombatPile, Min = 0, Max = 2, CanSkip = true, Manual = true, FirstVerbs = "SelectCard+Skip", End = "Skip", ResultCount = 0, Options = 3, Pile = PileType.Discard },
        });
        cases.Add(new Case
        {
            Name = "FromCombatPile n <= MinSelect without confirmation: the game answers itself",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                await ScenarioTools.AddCard(ctx, "HEADBUTT", PileType.Exhaust);
                return await Many(CardSelectCmd.FromCombatPile(Blocking(), PileType.Exhaust.GetPile(p), p, new CardSelectorPrefs(CardSelectorPrefs.ExhaustSelectionPrompt, 1)));
            },
            Expect = new Expect { NoPrompt = true, ResultCount = 1 },
        });

        // ------------------------------------------------------------------ FromHand (:536) / FromHandForDiscard (:593) / FromHandForUpgrade (:608)
        cases.Add(new Case
        {
            Name = "FromHand exactly 1 (hand order, hand ordinal)",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                snapshot = p.PlayerCombatState!.Hand.Cards.ToList();
                return await Many(CardSelectCmd.FromHand(Blocking(), p, new CardSelectorPrefs(CardSelectorPrefs.ExhaustSelectionPrompt, 1), null, snapshot[0]));
            },
            Plan = P(Verb.Invalid, 1),
            Expect = new Expect
            {
                Kind = CardSelectKind.Hand, Entry = CardSelectEntry.Hand, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Pile = PileType.Hand,
                Check = (ctx, rec, result) => rec!.Options == snapshot.Count && ReferenceEquals(result[0], snapshot[1]) ? null : "option 1 is not the second hand card",
            },
        });
        cases.Add(new Case
        {
            Name = "FromHand any number -> two picks in reverse hand order, CONFIRM",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                snapshot = p.PlayerCombatState!.Hand.Cards.ToList();
                return await Many(CardSelectCmd.FromHand(Blocking(), p, new CardSelectorPrefs(CardSelectorPrefs.ExhaustSelectionPrompt, 0, 999999999), null, snapshot[0]));
            },
            Plan = P(Verb.ConfirmSelection, 2, 0),
            Expect = new Expect
            {
                Kind = CardSelectKind.Hand, Entry = CardSelectEntry.Hand, Min = 0, CanSkip = true, Manual = true, FirstVerbs = "SelectCard+Skip", End = "ConfirmSelection", ResultCount = 2, Pile = PileType.Hand,
                Check = (ctx, rec, result) =>
                    rec!.Max == snapshot.Count && ReferenceEquals(result[0], snapshot[2]) && ReferenceEquals(result[1], snapshot[0]) ? null : "expected max = hand size and the result [3rd, 1st hand card]",
            },
        });
        cases.Add(new Case
        {
            Name = "FromHand with a filter: candidates = the matching hand cards, in hand order",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                await ScenarioTools.AddCard(ctx, "HEADBUTT", PileType.Hand);
                await ScenarioTools.AddCard(ctx, "ARMAMENTS", PileType.Hand);
                await ScenarioTools.AddCard(ctx, "HEADBUTT", PileType.Hand);
                snapshot = p.PlayerCombatState!.Hand.Cards.Where(c => c.Id.Entry == "HEADBUTT" || c.Id.Entry == "ARMAMENTS").ToList();
                return await Many(CardSelectCmd.FromHand(Blocking(), p, new CardSelectorPrefs(CardSelectorPrefs.ExhaustSelectionPrompt, 1),
                    c => c.Id.Entry == "HEADBUTT" || c.Id.Entry == "ARMAMENTS", snapshot[0]));
            },
            Plan = P(Verb.Invalid, 2),
            Expect = new Expect
            {
                Kind = CardSelectKind.Hand, Entry = CardSelectEntry.Hand, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Options = 3, Instances = 3, Pile = PileType.Hand,
                Check = (ctx, rec, result) => ReferenceEquals(result[0], snapshot[2]) ? null : "option 2 is not the third matching hand card (identical hand cards must stay separate options)",
            },
        });
        cases.Add(new Case
        {
            Name = "FromHandForDiscard exactly 2",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                snapshot = p.PlayerCombatState!.Hand.Cards.ToList();
                return await Many(CardSelectCmd.FromHandForDiscard(Blocking(), p, new CardSelectorPrefs(CardSelectorPrefs.DiscardSelectionPrompt, 2), null, snapshot[0]));
            },
            Plan = P(Verb.Invalid, 4, 3),
            Expect = new Expect
            {
                Kind = CardSelectKind.HandDiscard, Entry = CardSelectEntry.HandForDiscard, Min = 2, Max = 2, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 2, Pile = PileType.Hand, Prompt = "card_selection/TO_DISCARD",
                Check = (ctx, rec, result) => ReferenceEquals(result[0], snapshot[4]) && ReferenceEquals(result[1], snapshot[3]) ? null : "the result is not [5th, 4th hand card]",
            },
        });
        cases.Add(new Case
        {
            Name = "FromHandForUpgrade (2+ upgradable cards): forced pick of 1",
            InCombat = true,
            Call = async ctx =>
            {
                Player p = ctx.LocalPlayer;
                snapshot = p.PlayerCombatState!.Hand.Cards.Where(c => c.IsUpgradable).ToList();
                return One(await CardSelectCmd.FromHandForUpgrade(Blocking(), p, snapshot[0]));
            },
            Plan = P(Verb.Invalid, 1),
            Expect = new Expect
            {
                Kind = CardSelectKind.HandUpgrade, Entry = CardSelectEntry.HandForUpgrade, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Pile = PileType.Hand, Prompt = "gameplay_ui/CHOOSE_CARD_UPGRADE_HEADER",
                Check = (ctx, rec, result) => ReferenceEquals(result[0], snapshot[1]) ? null : "option 1 is not the second upgradable hand card",
            },
        });

        // ------------------------------------------------------------------ deck screens
        cases.Add(new Case
        {
            Name = "FromDeckForUpgrade cancelable + manual (Smith) -> CANCEL",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForUpgrade(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.UpgradeSelectionPrompt, 1) { Cancelable = true, RequireManualConfirmation = true })),
            Plan = P(Verb.Cancel),
            Expect = new Expect { Kind = CardSelectKind.DeckUpgrade, Entry = CardSelectEntry.DeckForUpgrade, Min = 1, Max = 1, Cancelable = true, Manual = true, FirstVerbs = "SelectCard+Cancel", End = "Cancel", ResultCount = 0, Merged = true, Pile = PileType.Deck, Prompt = "card_selection/TO_UPGRADE" },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForUpgrade cancelable + manual (Smith) -> pick, CONFIRM",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForUpgrade(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.UpgradeSelectionPrompt, 1) { Cancelable = true, RequireManualConfirmation = true })),
            Plan = P(Verb.ConfirmSelection, 0),
            Expect = new Expect
            {
                Kind = CardSelectKind.DeckUpgrade, Entry = CardSelectEntry.DeckForUpgrade, Min = 1, Max = 1, Cancelable = true, Manual = true, FirstVerbs = "SelectCard+Cancel", End = "ConfirmSelection", ResultCount = 1, Merged = true, Pile = PileType.Deck,
                Check = (ctx, rec, result) => ReferenceEquals(result[0].Pile, ctx.LocalPlayer.Deck) && result[0].IsUpgradable ? null : "the result is not an upgradable deck card",
            },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForUpgrade not cancelable, no confirmation: completes with the pick",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForUpgrade(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.UpgradeSelectionPrompt, 1))),
            Plan = P(Verb.Invalid, 1),
            Expect = new Expect { Kind = CardSelectKind.DeckUpgrade, Entry = CardSelectEntry.DeckForUpgrade, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Merged = true, Pile = PileType.Deck },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForTransformation 1..2 -> one pick, CONFIRM",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForTransformation(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, 1, 2))),
            Plan = P(Verb.ConfirmSelection, 0),
            Expect = new Expect { Kind = CardSelectKind.DeckTransform, Entry = CardSelectEntry.DeckForTransformation, Min = 1, Max = 2, Manual = true, FirstVerbs = "SelectCard", End = "ConfirmSelection", ResultCount = 1, Merged = true, Pile = PileType.Deck },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForTransformation 0..2 not cancelable -> SKIP",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForTransformation(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.TransformSelectionPrompt, 0, 2))),
            Plan = P(Verb.Skip),
            Expect = new Expect { Kind = CardSelectKind.DeckTransform, Entry = CardSelectEntry.DeckForTransformation, Min = 0, Max = 2, CanSkip = true, Manual = true, FirstVerbs = "SelectCard+Skip", End = "Skip", ResultCount = 0, Merged = true, Pile = PileType.Deck },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForEnchantment(player) exactly 1",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForEnchantment(ctx.LocalPlayer, ModelDb.Enchantment<Sharp>(), 2, new CardSelectorPrefs(CardSelectorPrefs.EnchantSelectionPrompt, 1))),
            Plan = P(Verb.Invalid, 0),
            Expect = new Expect
            {
                Kind = CardSelectKind.DeckEnchant, Entry = CardSelectEntry.DeckForEnchantment, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Merged = true, Pile = PileType.Deck, Prompt = "card_selection/TO_ENCHANT",
                Check = (ctx, rec, result) => result[0].Type == CardType.Attack ? null : "Sharp was offered for a non-attack: " + result[0].Id.Entry,
            },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForEnchantment(player, filter) exactly 1",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForEnchantment(ctx.LocalPlayer, ModelDb.Enchantment<Sharp>(), 2, c => c != null && c.Id.Entry.StartsWith("STRIKE", StringComparison.Ordinal),
                new CardSelectorPrefs(CardSelectorPrefs.EnchantSelectionPrompt, 1))),
            Plan = P(Verb.Invalid, 0),
            Expect = new Expect
            {
                Kind = CardSelectKind.DeckEnchant, Entry = CardSelectEntry.DeckForEnchantmentFiltered, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Options = 1, Merged = true, Pile = PileType.Deck,
                Check = (ctx, rec, result) => result[0].Id.Entry.StartsWith("STRIKE", StringComparison.Ordinal) ? null : "the filter was not applied: " + result[0].Id.Entry,
            },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForEnchantment(cards) 1..2 cancelable -> two picks, CONFIRM",
            Call = async ctx =>
            {
                Sharp sharp = ModelDb.Enchantment<Sharp>();
                given = ctx.LocalPlayer.Deck.Cards.Where(c => sharp.CanEnchant(c)).ToList();
                return await Many(CardSelectCmd.FromDeckForEnchantment(given, sharp, 2, new CardSelectorPrefs(CardSelectorPrefs.EnchantSelectionPrompt, 1, 2) { Cancelable = true }));
            },
            Plan = P(Verb.ConfirmSelection, -1, -1),
            Expect = new Expect
            {
                Kind = CardSelectKind.DeckEnchant, Entry = CardSelectEntry.DeckForEnchantmentCards, Min = 1, Max = 2, Cancelable = true, Manual = true, FirstVerbs = "SelectCard+Cancel", End = "ConfirmSelection", ResultCount = 2, Merged = true, Pile = PileType.Deck,
                Check = (ctx, rec, result) => !ReferenceEquals(result[0], result[1]) && CardClassKey.Of(result[0]) == CardClassKey.Of(result[1]) ? null : "two picks of one class must be two different instances of that class",
            },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForRemoval 2, cancelable + manual (Cook) -> two picks, CONFIRM",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForRemoval(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.RemoveSelectionPrompt, 2) { Cancelable = true, RequireManualConfirmation = true })),
            Plan = P(Verb.ConfirmSelection, 0, 1),
            Expect = new Expect { Kind = CardSelectKind.DeckRemoval, Entry = CardSelectEntry.DeckForRemoval, Min = 2, Max = 2, Cancelable = true, Manual = true, FirstVerbs = "SelectCard+Cancel", End = "ConfirmSelection", ResultCount = 2, Merged = true, Pile = PileType.Deck, Prompt = "card_selection/TO_REMOVE" },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForRemoval 2, cancelable + manual, one removable card: CANCEL is the only exit",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForRemoval(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.RemoveSelectionPrompt, 2) { Cancelable = true, RequireManualConfirmation = true },
                c => c.Id.Entry.StartsWith("BASH", StringComparison.Ordinal))),
            Plan = P(Verb.Cancel),
            Expect = new Expect { Kind = CardSelectKind.DeckRemoval, Entry = CardSelectEntry.DeckForRemoval, Min = 0, Max = 0, Cancelable = true, Manual = true, FirstVerbs = "Cancel", End = "Cancel", ResultCount = 0, Options = 1, Instances = 1, Pile = PileType.Deck },
        });
        cases.Add(new Case
        {
            Name = "FromDeckForRemoval 1 with a filter, no confirmation",
            Call = async ctx => await Many(CardSelectCmd.FromDeckForRemoval(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.RemoveSelectionPrompt, 1), c => c.Type == CardType.Skill)),
            Plan = P(Verb.Invalid, 0),
            Expect = new Expect
            {
                Kind = CardSelectKind.DeckRemoval, Entry = CardSelectEntry.DeckForRemoval, Min = 1, Max = 1, FirstVerbs = "SelectCard", End = "Auto", ResultCount = 1, Merged = true, Pile = PileType.Deck,
                Check = (ctx, rec, result) => result[0].Type == CardType.Skill ? null : "the filter was not applied",
            },
        });
        cases.Add(new Case
        {
            Name = "FromDeckGeneric 0..3 with a sorting order -> SKIP",
            Call = async ctx => await Many(CardSelectCmd.FromDeckGeneric(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.RemoveSelectionPrompt, 0, 3), null, c => -c.Id.Entry.Length)),
            Plan = P(Verb.Skip),
            Expect = new Expect { Kind = CardSelectKind.DeckGeneric, Entry = CardSelectEntry.DeckGeneric, Min = 0, Max = 3, CanSkip = true, Manual = true, FirstVerbs = "SelectCard+Skip", End = "Skip", ResultCount = 0, Merged = true, Pile = PileType.Deck },
        });
        cases.Add(new Case
        {
            Name = "FromDeckGeneric 1..3 -> three picks (two of one class), then CONFIRM is the only action",
            Call = async ctx => await Many(CardSelectCmd.FromDeckGeneric(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.RemoveSelectionPrompt, 1, 3))),
            Plan = P(Verb.ConfirmSelection, -1, -1, 0),
            Expect = new Expect { Kind = CardSelectKind.DeckGeneric, Entry = CardSelectEntry.DeckGeneric, Min = 1, Max = 3, Manual = true, FirstVerbs = "SelectCard", End = "ConfirmSelection", ResultCount = 3, Merged = true, Pile = PileType.Deck },
        });
        cases.Add(new Case
        {
            Name = "FromDeckGeneric n <= MinSelect without confirmation: the game answers itself",
            Call = async ctx => await Many(CardSelectCmd.FromDeckGeneric(ctx.LocalPlayer, new CardSelectorPrefs(CardSelectorPrefs.RemoveSelectionPrompt, 1), c => c.Id.Entry.StartsWith("BASH", StringComparison.Ordinal))),
            Expect = new Expect { NoPrompt = true, ResultCount = 1 },
        });
        return cases;
    }

    public static List<CaseResult> Run(HostSession session, string? only)
    {
        List<CaseResult> results = new List<CaseResult>();
        ulong seed = 4000;
        foreach (Case c in Cases())
        {
            seed++;
            if (only != null && !c.Name.Contains(only, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            results.Add(RunCase(session, c, seed));
        }
        return results;
    }

    private static CaseResult RunCase(HostSession session, Case c, ulong seed)
    {
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(seed);
        Plan plan = c.Plan;
        Expect x = c.Expect;
        IReadOnlyList<CardModel>? result = null;
        string? planError = null;
        RunContext? context = null;
        policy.Prefer = d =>
        {
            if (d.Kind != DecisionKind.CardSelect || d.Context is not CardSelectContext sc)
            {
                return -1;
            }
            int k = sc.Picked.Count;
            if (k < plan.Picks.Count)
            {
                int want = plan.Picks[k];
                if (want < 0)
                {
                    // "the class with the most instances" (class ordinals depend on key order)
                    want = 0;
                    for (int i = 1; i < sc.Options.Count; i++)
                    {
                        if (sc.OptionCounts != null && sc.OptionCounts[i] > sc.OptionCounts[want])
                        {
                            want = i;
                        }
                    }
                }
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    if (d.Actions[i].Verb == Verb.SelectCard && d.Actions[i].Source.Index == want)
                    {
                        return i;
                    }
                }
                planError ??= $"pick {k}: SELECT_CARD of option {want} is not offered";
                return -1;
            }
            if (plan.Finish != Verb.Invalid)
            {
                int i = policy.FirstVerb(d, plan.Finish);
                if (i < 0)
                {
                    planError ??= $"after {k} picks {plan.Finish} is not offered ({string.Join("+", d.Actions.Select(a => a.Verb).Distinct())})";
                }
                return i;
            }
            planError ??= $"the prompt is still open after {k} picks and the plan has no exit";
            return -1;
        };
        return CaseRunner.Run(session, "entries", c.Name, "IRONCLAD", async ctx =>
        {
            context = ctx;
            if (c.InCombat)
            {
                await ScenarioTools.JumpToEncounter(ctx, CaseRunner.DefaultEncounter);
                await ctx.Quiescence();
            }
            result = await c.Call(ctx);
        }, policy, probe, stopWhen: ctx => true, verify: (r, p) =>
        {
            if (result == null)
            {
                return "the entry point did not return";
            }
            if (planError != null)
            {
                return planError;
            }
            SelectRecord? rec = p.Selects.LastOrDefault();
            if (x.NoPrompt)
            {
                if (p.Selects.Count != 0)
                {
                    return "a CARD_SELECT frame was raised for a prompt the game answers itself";
                }
                return result.Count == x.ResultCount ? null : $"the game returned {result.Count} cards, expected {x.ResultCount}";
            }
            if (rec == null || p.Selects.Count != 1)
            {
                return $"{p.Selects.Count} prompts were raised, expected exactly 1";
            }
            if (rec.Kind != x.Kind || rec.Entry != x.Entry)
            {
                return $"kind {rec.Kind} entry {rec.Entry}; expected {x.Kind} / {x.Entry}";
            }
            // The deck entry points take no PlayerChoiceContext at all (CardSelectCmd.cs:315-534).
            bool deck = x.Kind == CardSelectKind.DeckUpgrade || x.Kind == CardSelectKind.DeckTransform || x.Kind == CardSelectKind.DeckEnchant
                || x.Kind == CardSelectKind.DeckRemoval || x.Kind == CardSelectKind.DeckGeneric;
            ChoiceContextKind expectedContext = deck ? ChoiceContextKind.None : ChoiceContextKind.Blocking;
            if (rec.Context != expectedContext)
            {
                return "context " + rec.Context + ", expected " + expectedContext;
            }
            if (rec.Min != x.Min || (x.Max != 0 || x.FirstVerbs == "Cancel" ? rec.Max != x.Max : false))
            {
                return $"bounds [{rec.Min}, {rec.Max}]; expected [{x.Min}, {x.Max}]";
            }
            if (rec.Cancelable != x.Cancelable || rec.CanSkip != x.CanSkip || rec.Manual != x.Manual)
            {
                return $"cancelable {rec.Cancelable} canSkip {rec.CanSkip} manual {rec.Manual}; expected {x.Cancelable} / {x.CanSkip} / {x.Manual}";
            }
            if (rec.FirstVerbs != x.FirstVerbs)
            {
                return $"verbs of the first decision '{rec.FirstVerbs}', expected '{x.FirstVerbs}'";
            }
            if (rec.End != x.End)
            {
                return $"the prompt ended with '{rec.End}', expected '{x.End}'";
            }
            if ((x.Options >= 0 && rec.Options != x.Options) || (x.Instances >= 0 && rec.Instances != x.Instances))
            {
                return $"options {rec.Options} instances {rec.Instances}; expected {x.Options} / {x.Instances}";
            }
            if (x.Merged && !CardSelectFrame.MergesCandidates(((CardSelectFrame)rec.Frame).Prompt))
            {
                return "the candidates of this entry point were expected to be merged into classes";
            }
            if (x.Pile != rec.Pile)
            {
                return $"pile {rec.Pile}, expected {x.Pile}";
            }
            if (x.Prompt.Length > 0 && rec.Prompt != x.Prompt)
            {
                return $"prompt key '{rec.Prompt}', expected '{x.Prompt}'";
            }
            if (result.Count != x.ResultCount)
            {
                return $"CardSelectCmd returned {result.Count} cards, expected {x.ResultCount}";
            }
            if (result.Count != rec.PickedCards.Count)
            {
                return $"CardSelectCmd returned {result.Count} cards, the frame handed over {rec.PickedCards.Count}";
            }
            for (int i = 0; i < result.Count; i++)
            {
                if (!ReferenceEquals(result[i], rec.PickedCards[i]))
                {
                    return "CardSelectCmd did not return the picked instances in pick order (position " + i + ")";
                }
            }
            return x.Check?.Invoke(context!, rec, result);
        }, hp: 99999, maxTurns: 4, maxDecisions: 200);
    }
}
