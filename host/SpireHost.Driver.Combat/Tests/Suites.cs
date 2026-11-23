using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Driver.Combat.Tests;

/// <summary>
/// The combat team's test suites (DRIVER-API.md section 9: real run start + debug facilities).
///   sources    one scenario per game object that calls CardSelectCmd.From* (cards, relics, powers, KnowledgeDemon)
///   potions    every potion: use (all usage / target types) and discard
///   costs      X-cost and star-cost cards, without and with resources
///   vetoes     the 8 Hook.ShouldPlay vetoers
///   allcards   every card of the game is put into a hand and played if the engine allows it
///   drawperm   permuting the draw pile changes neither the shape nor the order of a draw-pile prompt
///   sweep      all encounters x 6 characters with random loadouts
/// </summary>
public static class Suites
{
    // ------------------------------------------------------------------------------------------------
    //  sources: every caller of CardSelectCmd.From* (grep of Models.* in ref\sts2-v0113-src)
    // ------------------------------------------------------------------------------------------------

    private static readonly (string id, CardSelectKind kind)[] CardSources =
    {
        ("ABUNDANCE", CardSelectKind.ChooseACard), ("DISCOVERY", CardSelectKind.ChooseACard), ("QUASAR", CardSelectKind.ChooseACard), ("SPLASH", CardSelectKind.ChooseACard),
        ("ARROW_STAB", CardSelectKind.CombatPile), ("CHARGE", CardSelectKind.CombatPile), ("CLEANSE", CardSelectKind.CombatPile), ("COSMIC_INDIFFERENCE", CardSelectKind.CombatPile),
        ("DREDGE", CardSelectKind.CombatPile), ("GRAVEBLAST", CardSelectKind.CombatPile), ("HEADBUTT", CardSelectKind.CombatPile), ("HOLOGRAM", CardSelectKind.CombatPile),
        ("KI_CHANT", CardSelectKind.CombatPile), ("LOCKED_IN", CardSelectKind.CombatPile), ("NEOWS_FURY", CardSelectKind.CombatPile), ("RECALL", CardSelectKind.CombatPile),
        ("SEANCE", CardSelectKind.CombatPile), ("SECRET_TECHNIQUE", CardSelectKind.CombatPile), ("SECRET_WEAPON", CardSelectKind.CombatPile), ("SEEKER_STRIKE", CardSelectKind.CombatPile),
        ("TUTOR", CardSelectKind.CombatPile), ("WISH", CardSelectKind.CombatPile),
        ("ARMAMENTS", CardSelectKind.HandUpgrade),
        ("BEGONE", CardSelectKind.Hand), ("BRAND", CardSelectKind.Hand), ("BURNING_PACT", CardSelectKind.Hand), ("DECISIONS_DECISIONS", CardSelectKind.Hand),
        ("DUAL_WIELD", CardSelectKind.Hand), ("GLIMMER", CardSelectKind.Hand), ("GUARDS", CardSelectKind.Hand), ("HAND_TRICK", CardSelectKind.Hand),
        ("HEIRLOOM_HAMMER", CardSelectKind.Hand), ("NIGHTMARE", CardSelectKind.Hand), ("PHOTON_CUT", CardSelectKind.Hand), ("PURITY", CardSelectKind.Hand),
        ("SCAVENGE", CardSelectKind.Hand), ("SCULPTING_STRIKE", CardSelectKind.Hand), ("SNAP", CardSelectKind.Hand), ("THINKING_AHEAD", CardSelectKind.Hand),
        ("TRANSFIGURE", CardSelectKind.Hand), ("TRUE_GRIT", CardSelectKind.Hand),
        ("ACROBATICS", CardSelectKind.HandDiscard), ("DAGGER_THROW", CardSelectKind.HandDiscard), ("HIDDEN_DAGGERS", CardSelectKind.HandDiscard),
        ("PREPARED", CardSelectKind.HandDiscard), ("SURVIVOR", CardSelectKind.HandDiscard), ("THE_HERMIT", CardSelectKind.HandDiscard),
    };

    private static readonly (string id, CardSelectKind kind, bool inCombat)[] RelicSources =
    {
        ("TOOLBOX", CardSelectKind.ChooseACard, true), ("HEFTY_TABLET", CardSelectKind.ChooseACard, false), ("LEAD_PAPERWEIGHT", CardSelectKind.ChooseACard, false),
        // MASSIVE_SCROLL is a multiplayer-only relic (its card factory call throws with one player).
        ("POMANDER", CardSelectKind.DeckUpgrade, false), ("YUMMY_COOKIE", CardSelectKind.DeckUpgrade, false),
        ("ELECTRIC_SHRYMP", CardSelectKind.DeckEnchant, false), ("GNARLED_HAMMER", CardSelectKind.DeckEnchant, false), ("KIFUDA", CardSelectKind.DeckEnchant, false),
        ("PAELS_GROWTH", CardSelectKind.DeckEnchant, false), ("PUNCH_DAGGER", CardSelectKind.DeckEnchant, false), ("ROYAL_STAMP", CardSelectKind.DeckEnchant, false),
        ("TRI_BOOMERANG", CardSelectKind.DeckEnchant, false),
        ("CHOICES_PARADOX", CardSelectKind.SimpleGrid, true),
        ("BIIIG_HUG", CardSelectKind.DeckRemoval, false), ("EMPTY_CAGE", CardSelectKind.DeckRemoval, false), ("PAELS_TOOTH", CardSelectKind.DeckRemoval, false),
        ("PRECARIOUS_SHEARS", CardSelectKind.DeckRemoval, false), ("PRECISE_SCISSORS", CardSelectKind.DeckRemoval, false), ("PRESERVED_FOG", CardSelectKind.DeckRemoval, false),
        ("ASTROLABE", CardSelectKind.DeckTransform, false), ("CLAWS", CardSelectKind.DeckTransform, false), ("NEW_LEAF", CardSelectKind.DeckTransform, false),
        ("TOASTY_MITTENS", CardSelectKind.Hand, true), ("GAMBLING_CHIP", CardSelectKind.HandDiscard, true),
        ("DOLLYS_MIRROR", CardSelectKind.DeckGeneric, false), ("SEA_GLASS", CardSelectKind.SimpleGridRewards, false),
    };

    private static readonly (string id, CardSelectKind kind)[] PowerSources =
    {
        ("TOOLS_OF_THE_TRADE_POWER", CardSelectKind.HandDiscard), ("TYRANNY_POWER", CardSelectKind.Hand), ("ENTROPY_POWER", CardSelectKind.Hand),
        ("FOREGONE_CONCLUSION_POWER", CardSelectKind.CombatPile), ("STRATAGEM_POWER", CardSelectKind.CombatPile),
    };

    public static List<CaseResult> Sources(HostSession session, string? only)
    {
        List<CaseResult> results = new List<CaseResult>();
        ulong seed = 100;
        foreach ((string id, CardSelectKind kind) in CardSources)
        {
            if (only != null && only != id)
            {
                continue;
            }
            seed++;
            if (!CombatCatalog.HasCard(id))
            {
                results.Add(new CaseResult { Suite = "sources", Name = "card " + id, Failure = "unknown card id" });
                continue;
            }
            results.Add(CardSource(session, id, kind, seed));
        }
        foreach ((string id, CardSelectKind kind, bool inCombat) in RelicSources)
        {
            if (only != null && only != id)
            {
                continue;
            }
            seed++;
            if (!CombatCatalog.HasRelic(id))
            {
                results.Add(new CaseResult { Suite = "sources", Name = "relic " + id, Failure = "unknown relic id" });
                continue;
            }
            results.Add(RelicSource(session, id, kind, inCombat, seed));
        }
        foreach ((string id, CardSelectKind kind) in PowerSources)
        {
            if (only != null && only != id)
            {
                continue;
            }
            seed++;
            results.Add(PowerSource(session, id, kind, seed));
        }
        if (only == null || only == "KNOWLEDGE_DEMON")
        {
            results.Add(KnowledgeDemon(session));
        }
        if (only == null || only == "SCROLL_BOXES")
        {
            results.Add(ScrollBoxes(session));
        }
        return results;
    }

    private static CaseResult CardSource(HostSession session, string id, CardSelectKind kind, ulong seed)
    {
        CardModel canonical = ScenarioTools.Card(id);
        string character = CombatTestTools.CharacterFor(canonical);
        CombatProbe probe = new CombatProbe();
        HashSet<string> ids = new HashSet<string> { id };
        bool played = false;
        bool done = false;
        string? unplayable = null;
        string? orderError = null;
        PreferPolicy policy = new PreferPolicy(seed);
        policy.Prefer = d => played ? -1 : policy.PlayOf(d, ids);
        probe.OnEach = (ctx, d, chosen, auto) =>
        {
            if (d.Kind != DecisionKind.Combat)
            {
                return;
            }
            if (played)
            {
                done = true;
                // The order the selector returns is the order the game uses. DREDGE moves the picked cards to the
                // hand in that order (visible in the public hand order); GLIMMER / PHOTON_CUT put the picked card
                // on top of the draw pile (CardPilePosition.Top = index 0, Commands/CardPileCmd.cs:416-419).
                if (EffectChecked.Contains(id) && orderError == null)
                {
                    SelectRecord? r = probe.Selects.LastOrDefault(s => s.Source == id);
                    if (r != null && r.PickedCards.Count > 0)
                    {
                        PlayerCombatState pcs = ctx.LocalPlayer.PlayerCombatState!;
                        int n = r.PickedCards.Count;
                        bool same;
                        if (id == "DREDGE")
                        {
                            IReadOnlyList<CardModel> hand = pcs.Hand.Cards;
                            same = hand.Count >= n;
                            for (int i = 0; same && i < n; i++)
                            {
                                same = ReferenceEquals(hand[hand.Count - n + i], r.PickedCards[i]);
                            }
                            orderError = same ? "" : "hand tail does not equal the picked cards in pick order";
                        }
                        else
                        {
                            IReadOnlyList<CardModel> draw = pcs.DrawPile.Cards;
                            same = draw.Count >= n;
                            for (int i = 0; same && i < n; i++)
                            {
                                same = ReferenceEquals(draw[n - 1 - i], r.PickedCards[i]); // each pick is inserted at index 0
                            }
                            orderError = same ? "" : "top of the draw pile does not hold the picked cards (last pick on top)";
                        }
                    }
                }
                return;
            }
            if (d.Actions[chosen].Verb == Verb.PlayCard && d.Actions[chosen].Source.Card.Id.Entry == id)
            {
                played = true;
                return;
            }
            // Not offered: why?
            foreach (CardModel c in ctx.LocalPlayer.PlayerCombatState!.Hand.Cards)
            {
                if (c.Id.Entry == id && !c.CanPlay(out UnplayableReason reason, out AbstractModel? preventer))
                {
                    unplayable = reason + (preventer != null ? " by " + preventer.Id.Entry : "");
                }
            }
        };
        return CaseRunner.Run(session, "sources", "card " + id, character, async ctx =>
        {
            await ScenarioTools.JumpToEncounter(ctx, CaseRunner.DefaultEncounter);
            await ctx.Quiescence();
            await ScenarioTools.SetResources(ctx, 20, 20);
            List<string> starters = CombatTestTools.StarterIds(ctx);
            // Discard pile: two identical cards and two different ones (class merging + a real choice).
            await ScenarioTools.AddCard(ctx, starters[0], PileType.Discard);
            await ScenarioTools.AddCard(ctx, starters[0], PileType.Discard);
            await ScenarioTools.AddCard(ctx, starters[Math.Min(1, starters.Count - 1)], PileType.Discard);
            await ScenarioTools.AddCard(ctx, starters[starters.Count - 1], PileType.Discard, upgrades: 1);
            await ScenarioTools.AddCard(ctx, id, PileType.Hand, upgrades: PromptNeedsUpgrade.Contains(id) ? 1 : 0);
        }, policy, probe, stopWhen: ctx => done || ctx.MapOpen || (probe.MaxTurn >= 2 && !played), verify: (r, p) =>
        {
            if (canonical.TargetType == TargetType.AnyAlly)
            {
                // Single player: AnyAlly cards can never be played (CardModel.cs:1522-1525).
                return !played && unplayable != null && unplayable.Contains("NoLivingAllies") ? null : "AnyAlly card with one player: played " + played + ", reason " + unplayable;
            }
            if (!played)
            {
                return "the card was never playable: " + (unplayable ?? "not in hand / no legal target");
            }
            if (!string.IsNullOrEmpty(orderError))
            {
                return orderError;
            }
            if (EffectChecked.Contains(id) && orderError == null)
            {
                return "the pick-order / effect check did not run";
            }
            SelectRecord? rec = p.Selects.FirstOrDefault(s => s.Source == id);
            if (rec == null)
            {
                return OptionalPrompt.Contains(id) ? null : "no selection prompt was raised";
            }
            if (rec.Kind != kind)
            {
                return $"prompt kind {rec.Kind}, expected {kind}";
            }
            if (rec.Context != ChoiceContextKind.GameAction || !rec.InCombat || rec.OutsidePlayPhase)
            {
                return $"prompt context {rec.Context}, inCombat {rec.InCombat}, outsidePlayPhase {rec.OutsidePlayPhase}; expected a card-play prompt";
            }
            return null;
        });
    }

    /// <summary>Cards whose selection result is checked against the game state afterwards.</summary>
    private static readonly HashSet<string> EffectChecked = new HashSet<string> { "DREDGE", "GLIMMER", "PHOTON_CUT" };

    /// <summary>Cards that ask only when upgraded (TrueGrit exhausts a random card otherwise).</summary>
    private static readonly HashSet<string> PromptNeedsUpgrade = new HashSet<string> { "TRUE_GRIT" };

    /// <summary>Cards whose prompt depends on state the generic scenario does not guarantee (the game answers itself).</summary>
    private static readonly HashSet<string> OptionalPrompt = new HashSet<string>
    {
        "ARROW_STAB",        // discard pile filtered by the Shoot tag: empty in the generic scenario
        "CLEANSE", "CHARGE", "LOCKED_IN", "SEEKER_STRIKE", "KI_CHANT", "COSMIC_INDIFFERENCE", "GRAVEBLAST", "NEOWS_FURY", "RECALL",
        "SCAVENGE", "HEIRLOOM_HAMMER", "TRANSFIGURE", "SCULPTING_STRIKE", "SNAP", "BEGONE", "HAND_TRICK", "THINKING_AHEAD", "DECISIONS_DECISIONS",
        "SECRET_TECHNIQUE", "SECRET_WEAPON", "QUASAR", "SPLASH", "THE_HERMIT", "HIDDEN_DAGGERS", "DAGGER_THROW", "BRAND", "NIGHTMARE", "DUAL_WIELD",
    };

    private static CaseResult RelicSource(HostSession session, string id, CardSelectKind kind, bool inCombat, ulong seed)
    {
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(seed);
        return CaseRunner.Run(session, "sources", "relic " + id, "IRONCLAD", async ctx =>
        {
            // A few upgradable / enchantable / removable cards beyond the starter deck.
            await ScenarioTools.GiveCard(ctx, "HEADBUTT");
            await ScenarioTools.GiveCard(ctx, "HEADBUTT");
            await ScenarioTools.GiveCard(ctx, "ARMAMENTS", upgrades: 1);
            await ScenarioTools.GiveRelic(ctx, id);
            await ScenarioTools.JumpToEncounter(ctx, CaseRunner.DefaultEncounter);
        }, policy, probe, stopWhen: ctx => probe.MaxTurn >= 3 || ctx.MapOpen, verify: (r, p) =>
        {
            SelectRecord? rec = p.Selects.FirstOrDefault(s => s.Kind == kind && s.InCombat == inCombat);
            if (rec == null)
            {
                return $"no {kind} prompt (inCombat {inCombat}) was raised; saw [{string.Join(", ", p.Selects.Select(s => s.Kind + "/" + s.Source))}]";
            }
            if (inCombat && rec.Source != id)
            {
                return "prompt source is " + rec.Source;
            }
            if (id == "TOOLBOX" && (rec.CanSkip || rec.Cancelable || rec.Min != 1 || !rec.OutsidePlayPhase || rec.Context != ChoiceContextKind.Hook))
            {
                return $"Toolbox: canSkip {rec.CanSkip} min {rec.Min} outsidePlayPhase {rec.OutsidePlayPhase} context {rec.Context}; expected a forced pick raised by a hook before the play phase";
            }
            if ((id == "GAMBLING_CHIP" || id == "CHOICES_PARADOX" || id == "TOASTY_MITTENS") && (!rec.OutsidePlayPhase || rec.Context != ChoiceContextKind.Hook))
            {
                return $"{id}: outsidePlayPhase {rec.OutsidePlayPhase} context {rec.Context}; expected a start-of-turn hook prompt";
            }
            if (id == "GAMBLING_CHIP" && (rec.Min != 0 || !rec.CanSkip))
            {
                return $"GamblingChip: min {rec.Min} canSkip {rec.CanSkip}; expected 'any number'";
            }
            return null;
        });
    }

    private static CaseResult PowerSource(HostSession session, string id, CardSelectKind kind, ulong seed)
    {
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(seed);
        return CaseRunner.Run(session, "sources", "power " + id, "SILENT", async ctx =>
        {
            await ScenarioTools.JumpToEncounter(ctx, CaseRunner.DefaultEncounter);
            await ctx.Quiescence();
            await ScenarioTools.ApplyPower(ctx, id, 1, ctx.LocalPlayer.Creature);
        }, policy, probe, stopWhen: ctx => probe.MaxTurn >= 7 || ctx.MapOpen || probe.Selects.Any(s => s.Kind == kind && s.End.Length > 0), verify: (r, p) =>
        {
            SelectRecord? rec = p.Selects.FirstOrDefault(s => s.Kind == kind);
            if (rec == null)
            {
                return $"no {kind} prompt was raised in {p.MaxTurn} turns";
            }
            if (rec.Context != ChoiceContextKind.Hook || !rec.InCombat)
            {
                return $"context {rec.Context}; expected a hook prompt";
            }
            if (!rec.OutsidePlayPhase)
            {
                return "the hook prompt was asked inside the play phase";
            }
            if (kind == CardSelectKind.CombatPile && rec.Pile != PileType.Draw)
            {
                return "pile " + rec.Pile;
            }
            return null;
        });
    }

    private static CaseResult KnowledgeDemon(HostSession session)
    {
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(7);
        return CaseRunner.Run(session, "sources", "monster KNOWLEDGE_DEMON", "IRONCLAD", async ctx =>
        {
            await ScenarioTools.JumpToEncounter(ctx, "KNOWLEDGE_DEMON_BOSS");
        }, policy, probe, stopWhen: ctx => probe.MaxTurn >= 12 || ctx.MapOpen || probe.Selects.Any(s => s.DuringEnemyTurn && s.End.Length > 0), verify: (r, p) =>
        {
            SelectRecord? rec = p.Selects.FirstOrDefault(s => s.DuringEnemyTurn);
            if (rec == null)
            {
                return $"no enemy-turn prompt in {p.MaxTurn} turns";
            }
            if (rec.Kind != CardSelectKind.ChooseACard || rec.Context != ChoiceContextKind.Blocking || rec.CanSkip || rec.Min != 1 || rec.Max != 1)
            {
                return $"kind {rec.Kind} context {rec.Context} canSkip {rec.CanSkip} [{rec.Min}, {rec.Max}]; expected a forced ChooseACard on a blocking context";
            }
            if (!rec.IntentsHiddenSeen)
            {
                return "RunContext.IntentPerformed did not report the acting enemy's intent as performed";
            }
            return null;
        });
    }

    /// <summary>
    /// CardSelectCmd.FromChooseABundleScreen belongs to the events team (TestMode gate patch + BUNDLE frame); a
    /// second patch on it from this module would break their fail-closed "exactly one" check at startup. This
    /// case proves the combat module leaves the entry alone - no registry record, no Harmony patch method of this
    /// assembly on the stub or on its state machine - and that obtaining SCROLL_BOXES (the only caller,
    /// Models.Relics/ScrollBoxes.cs:40) still runs to completion with this module registered: the game's own
    /// branch answers (TestMode: bundles[0], CardSelectCmd.cs:670-673, in this console; the events team's frame
    /// after integration), and no CARD_SELECT frame of this module appears for it.
    /// </summary>
    private static CaseResult ScrollBoxes(HostSession session)
    {
        CombatProbe probe = new CombatProbe();
        int deckBefore = -1;
        int deckAfter = -1;
        int selectsDuringObtain = -1;
        return CaseRunner.Run(session, "sources", "relic SCROLL_BOXES (bundle entry is not ours)", "IRONCLAD", async ctx =>
        {
            deckBefore = ctx.LocalPlayer.Deck.Cards.Count;
            await ScenarioTools.GiveRelic(ctx, "SCROLL_BOXES");
            deckAfter = ctx.LocalPlayer.Deck.Cards.Count;
            selectsDuringObtain = probe.Selects.Count;
            await ScenarioTools.JumpToEncounter(ctx, CaseRunner.DefaultEncounter);
        }, new PreferPolicy(3), probe, stopWhen: ctx => probe.MaxTurn >= 2 || ctx.MapOpen, verify: (r, p) =>
        {
            System.Reflection.MethodInfo? stub = HarmonyLib.AccessTools.Method(typeof(MegaCrit.Sts2.Core.Commands.CardSelectCmd), nameof(MegaCrit.Sts2.Core.Commands.CardSelectCmd.FromChooseABundleScreen));
            if (stub == null)
            {
                return "CardSelectCmd.FromChooseABundleScreen not found";
            }
            System.Reflection.MethodInfo? moveNext = (stub.GetCustomAttributes(typeof(System.Runtime.CompilerServices.AsyncStateMachineAttribute), false).FirstOrDefault() as System.Runtime.CompilerServices.AsyncStateMachineAttribute)
                ?.StateMachineType.GetMethod("MoveNext", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public);
            System.Reflection.Assembly mine = typeof(SelectionCapture).Assembly;
            foreach (System.Reflection.MethodBase? target in new System.Reflection.MethodBase?[] { stub, moveNext })
            {
                if (target == null)
                {
                    continue;
                }
                HarmonyLib.Patches? info = HarmonyLib.Harmony.GetPatchInfo(target);
                if (info == null)
                {
                    continue;
                }
                foreach (HarmonyLib.Patch patch in info.Prefixes.Concat(info.Postfixes).Concat(info.Transpilers).Concat(info.Finalizers))
                {
                    if (patch.PatchMethod.DeclaringType?.Assembly == mine)
                    {
                        return "the combat module has a Harmony patch on " + target.DeclaringType?.Name + "." + target.Name + ": " + patch.PatchMethod.Name;
                    }
                }
            }
            foreach (SpireHost.Patches.PatchRecord record in SpireHost.Patches.PatchRegistry.Records)
            {
                if (record.Id.StartsWith("combat-", StringComparison.Ordinal) && record.Target.Contains("Bundle", StringComparison.OrdinalIgnoreCase))
                {
                    return "combat registry record on the bundle entry: " + record.Id;
                }
            }
            if (selectsDuringObtain != 0)
            {
                return "a CARD_SELECT frame of this module appeared while SCROLL_BOXES was obtained";
            }
            return deckAfter > deckBefore ? null : $"SCROLL_BOXES added no cards (deck {deckBefore} -> {deckAfter})";
        });
    }

    // ------------------------------------------------------------------------------------------------
    //  potions
    // ------------------------------------------------------------------------------------------------

    public static List<CaseResult> Potions(HostSession session, string? only)
    {
        List<CaseResult> results = new List<CaseResult>();
        ulong seed = 500;
        foreach (PotionModel potion in CombatCatalog.Potions())
        {
            string id = potion.Id.Entry;
            if (only != null && only != id)
            {
                continue;
            }
            foreach (string character in new[] { "IRONCLAD", "DEFECT" })
            {
                results.Add(PotionUse(session, potion, character, ++seed));
            }
            results.Add(PotionDiscard(session, potion, ++seed));
        }
        return results;
    }

    private static CaseResult PotionUse(HostSession session, PotionModel canonical, string character, ulong seed)
    {
        string id = canonical.Id.Entry;
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(seed);
        bool used = false;
        bool done = false;
        int slot = -1;
        string? shape = null;
        string? after = null;
        int offeredUses = -1;
        bool discardOffered = false;
        PotionModel? usedPotion = null;
        policy.Prefer = d => used || d.Kind != DecisionKind.Combat ? -1 : policy.RandomVerb(d, Verb.UsePotion);
        probe.OnEach = (ctx, d, chosen, auto) =>
        {
            if (d.Kind != DecisionKind.Combat)
            {
                return;
            }
            Player p = ctx.LocalPlayer;
            if (used)
            {
                if (!done)
                {
                    done = true;
                    after = !ReferenceEquals(p.PotionSlots[slot], usedPotion) ? "" : "the potion is still in its slot after the use resolved";
                }
                return;
            }
            if (offeredUses < 0)
            {
                // First combat decision: what is offered for the potion?
                CombatState state = ((CombatContext)d.Context!).State;
                int hittable = state.HittableEnemies.Count;
                Creature self = p.Creature;
                offeredUses = 0;
                List<int> targets = new List<int>();
                foreach (LegalAction a in d.Actions)
                {
                    if (a.Verb == Verb.UsePotion)
                    {
                        offeredUses++;
                        targets.Add(a.Target.Index);
                        slot = a.Source.Index;
                    }
                    if (a.Verb == Verb.DiscardPotion)
                    {
                        discardOffered = true;
                    }
                }
                PotionModel live = p.PotionSlots.First(x => x != null)!;
                usedPotion = live;
                TargetType type = live.TargetType;
                int selfIndex = state.Creatures.ToList().IndexOf(self);
                if (live.Usage == PotionUsage.Automatic)
                {
                    shape = offeredUses == 0 ? "" : "an Automatic potion is offered for use";
                }
                else if (type == TargetType.AnyEnemy)
                {
                    shape = offeredUses == hittable && targets.All(t => t >= state.Allies.Count) ? "" : $"AnyEnemy potion: {offeredUses} uses for {hittable} hittable enemies";
                }
                else if (type == TargetType.AnyPlayer || type == TargetType.Self)
                {
                    shape = offeredUses == 1 && targets[0] == selfIndex ? "" : $"{type} potion: {offeredUses} uses, targets [{string.Join(",", targets)}], self is {selfIndex}";
                }
                else
                {
                    shape = offeredUses == 1 && targets[0] == -1 ? "" : $"{type} potion: {offeredUses} uses, targets [{string.Join(",", targets)}]; expected one untargeted use";
                }
            }
            if (d.Actions[chosen].Verb == Verb.UsePotion)
            {
                used = true;
            }
            else if (offeredUses == 0)
            {
                done = true; // nothing to use (Automatic)
            }
        };
        return CaseRunner.Run(session, "potions", "use " + id + " as " + character, character, async ctx =>
        {
            await ScenarioTools.GivePotion(ctx, id);
            await ScenarioTools.JumpToEncounter(ctx, CaseRunner.DefaultEncounter);
            await ctx.Quiescence();
            if (ctx.CombatManager.IsInProgress)
            {
                await ScenarioTools.SetResources(ctx, 3, 3);
                List<string> starters = CombatTestTools.StarterIds(ctx);
                await ScenarioTools.AddCard(ctx, starters[0], PileType.Discard);
                await ScenarioTools.AddCard(ctx, starters[Math.Min(1, starters.Count - 1)], PileType.Discard);
            }
        }, policy, probe, stopWhen: ctx => done || ctx.MapOpen, verify: (r, p) =>
        {
            if (shape == null)
            {
                return "no combat decision was reached";
            }
            if (shape.Length > 0)
            {
                return shape;
            }
            if (!discardOffered)
            {
                return "DISCARD_POTION was not offered";
            }
            if (canonical.Usage == PotionUsage.Automatic)
            {
                return null;
            }
            if (!used)
            {
                return "the potion was not used";
            }
            if (!string.IsNullOrEmpty(after))
            {
                return after;
            }
            return null;
        }, hp: 99999, maxTurns: 6, maxDecisions: 300);
    }

    private static CaseResult PotionDiscard(HostSession session, PotionModel canonical, ulong seed)
    {
        string id = canonical.Id.Entry;
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(seed);
        bool discarded = false;
        bool done = false;
        string? after = null;
        int slot = -1;
        policy.Prefer = d => discarded || d.Kind != DecisionKind.Combat ? -1 : policy.FirstVerb(d, Verb.DiscardPotion);
        probe.OnEach = (ctx, d, chosen, auto) =>
        {
            if (d.Kind != DecisionKind.Combat)
            {
                return;
            }
            if (discarded)
            {
                if (!done)
                {
                    done = true;
                    after = ctx.LocalPlayer.PotionSlots[slot] == null ? "" : "the potion is still in its slot after the discard resolved";
                }
                return;
            }
            if (d.Actions[chosen].Verb == Verb.DiscardPotion)
            {
                discarded = true;
                slot = d.Actions[chosen].Source.Index;
            }
        };
        return CaseRunner.Run(session, "potions", "discard " + id, "SILENT", async ctx =>
        {
            await ScenarioTools.GivePotion(ctx, id);
            await ScenarioTools.JumpToEncounter(ctx, CaseRunner.DefaultEncounter);
        }, policy, probe, stopWhen: ctx => done || ctx.MapOpen, verify: (r, p) =>
        {
            if (!discarded)
            {
                return "DISCARD_POTION was never offered";
            }
            return string.IsNullOrEmpty(after) ? null : after;
        }, hp: 99999, maxTurns: 4, maxDecisions: 200);
    }

    // ------------------------------------------------------------------------------------------------
    //  costs: X-cost and star-cost cards
    // ------------------------------------------------------------------------------------------------

    public static List<CaseResult> Costs(HostSession session, string? only)
    {
        List<CaseResult> results = new List<CaseResult>();
        ulong seed = 900;
        foreach (CardModel card in CombatCatalog.Cards())
        {
            bool starCard = card.CanonicalStarCost > 0 || card.HasStarCostX;
            if (!card.EnergyCost.CostsX && !starCard)
            {
                continue;
            }
            if (only != null && only != card.Id.Entry)
            {
                continue;
            }
            results.Add(CostCase(session, card, resources: false, ++seed));
            results.Add(CostCase(session, card, resources: true, ++seed));
        }
        return results;
    }

    private static CaseResult CostCase(HostSession session, CardModel canonical, bool resources, ulong seed)
    {
        string id = canonical.Id.Entry;
        string character = CombatTestTools.CharacterFor(canonical, "REGENT");
        bool costsX = canonical.EnergyCost.CostsX;
        bool starX = canonical.HasStarCostX;
        int starCost = Math.Max(0, canonical.CanonicalStarCost);
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(seed);
        HashSet<string> ids = new HashSet<string> { id };
        string? finding = null;
        bool played = false;
        bool done = false;
        int energyBefore = -1;
        int starsBefore = -1;
        int starCostShown = -1;
        CardModel? playedCard = null;
        policy.Prefer = d => played ? -1 : policy.PlayOf(d, ids);
        probe.OnEach = (ctx, d, chosen, auto) =>
        {
            if (d.Kind != DecisionKind.Combat)
            {
                return;
            }
            PlayerCombatState pcs = ctx.LocalPlayer.PlayerCombatState!;
            if (played)
            {
                if (!done)
                {
                    done = true;
                    // Engine facts observed through the driver (what SpendResources recorded on the card,
                    // Models/CardModel.cs:1593-1609): an X-cost card took all the energy, a star cost was paid.
                    if (playedCard != null)
                    {
                        if (costsX && playedCard.EnergyCost.CapturedXValue != energyBefore)
                        {
                            finding = $"X-cost card captured X = {playedCard.EnergyCost.CapturedXValue} with {energyBefore} energy";
                        }
                        else if (starX && playedCard.LastStarsSpent != starsBefore)
                        {
                            finding = $"star-X card spent {playedCard.LastStarsSpent} of {starsBefore} stars";
                        }
                        else if (!starX && starCost > 0 && playedCard.LastStarsSpent != starCostShown)
                        {
                            finding = $"star-cost card spent {playedCard.LastStarsSpent} stars, cost shown {starCostShown}";
                        }
                    }
                }
                return;
            }
            CardModel? card = pcs.Hand.Cards.FirstOrDefault(c => c.Id.Entry == id);
            if (card == null)
            {
                finding ??= "the card is not in hand";
                done = true;
                return;
            }
            bool can = card.CanPlay(out UnplayableReason reason, out _);
            bool offered = d.Actions.Any(a => a.Verb == Verb.PlayCard && ReferenceEquals(a.Source.Card, card));
            bool noTarget = can && !offered; // a targeted card without a legal target
            if (!resources)
            {
                // 0 energy, 0 stars.
                done = true;
                bool expectResourceBlock = (!costsX && canonical.EnergyCost.Canonical > 0) || (!starX && starCost > 0);
                if (!starX && starCost > 0 && (reason & UnplayableReason.StarCostTooHigh) == 0)
                {
                    finding = "star-cost card without stars: reason " + reason + " lacks StarCostTooHigh";
                }
                else if (expectResourceBlock && offered)
                {
                    finding = "offered without the resources to pay for it";
                }
                else if (!expectResourceBlock && !offered && !noTarget && (reason & (UnplayableReason.EnergyCostTooHigh | UnplayableReason.StarCostTooHigh)) != 0)
                {
                    finding = "X-cost card blocked by resources at 0 energy / 0 stars: " + reason;
                }
                return;
            }
            if (d.Actions[chosen].Verb == Verb.PlayCard && ReferenceEquals(d.Actions[chosen].Source.Card, card))
            {
                played = true;
                energyBefore = pcs.Energy;
                starsBefore = pcs.Stars;
                starCostShown = starX ? -1 : card.GetStarCostWithModifiers();
                playedCard = card;
            }
            else if (!offered)
            {
                done = true;
                if ((reason & (UnplayableReason.EnergyCostTooHigh | UnplayableReason.StarCostTooHigh)) != 0)
                {
                    finding = "blocked by resources with 9 energy / 9 stars: " + reason;
                }
                else
                {
                    finding = "note:not offered (" + (noTarget ? "no legal target" : reason.ToString()) + ")";
                }
            }
        };
        return CaseRunner.Run(session, "costs", id + (resources ? " with resources" : " without resources"), character, async ctx =>
        {
            await ScenarioTools.JumpToEncounter(ctx, CaseRunner.DefaultEncounter);
            await ctx.Quiescence();
            await ScenarioTools.SetResources(ctx, resources ? 9 : 0, resources ? 9 : 0);
            await ScenarioTools.AddCard(ctx, id, PileType.Hand);
        }, policy, probe, stopWhen: ctx => done || ctx.MapOpen, verify: (r, p) =>
        {
            if (finding != null && !finding.StartsWith("note:", StringComparison.Ordinal))
            {
                return finding;
            }
            if (resources && !played && finding == null)
            {
                return "the card was not played";
            }
            return null;
        }, maxTurns: 4, maxDecisions: 200);
    }

    // ------------------------------------------------------------------------------------------------
    //  vetoes: the Hook.ShouldPlay vetoers (grep "override bool ShouldPlay(": 8 models)
    // ------------------------------------------------------------------------------------------------

    public static List<CaseResult> Vetoes(HostSession session, string? only)
    {
        (string name, string preventer, Func<RunContext, Task> arm)[] cases =
        {
            ("NORMALITY (card in hand)", "NORMALITY", ctx => ScenarioTools.AddCard(ctx, "NORMALITY", PileType.Hand)),
            ("ENTHRALLED (card in hand)", "ENTHRALLED", ctx => ScenarioTools.AddCard(ctx, "ENTHRALLED", PileType.Hand)),
            ("VELVET_CHOKER (relic)", "VELVET_CHOKER", ctx => Task.CompletedTask),
            ("SLOTH_POWER", "SLOTH_POWER", ctx => ScenarioTools.ApplyPower(ctx, "SLOTH_POWER", 2, ctx.LocalPlayer.Creature)),
            ("ENCORE_POWER", "ENCORE_POWER", ctx => ScenarioTools.ApplyPower(ctx, "ENCORE_POWER", 2, ctx.LocalPlayer.Creature)),
            ("CHAINS_OF_BINDING_POWER + Bound", "CHAINS_OF_BINDING_POWER", ctx => ArmAffliction(ctx, "CHAINS_OF_BINDING_POWER", "BOUND")),
            ("RINGING_POWER + Ringing", "RINGING_POWER", ctx => ArmAffliction(ctx, "RINGING_POWER", "RINGING")),
            ("SMOGGY_POWER + Smog", "SMOGGY_POWER", ctx => ArmAffliction(ctx, "SMOGGY_POWER", "SMOG")),
        };
        List<CaseResult> results = new List<CaseResult>();
        ulong seed = 1300;
        foreach ((string name, string preventer, Func<RunContext, Task> arm) in cases)
        {
            if (only != null && only != preventer)
            {
                continue;
            }
            foreach (string character in new[] { "IRONCLAD", "SILENT" })
            {
                CombatProbe probe = new CombatProbe();
                PreferPolicy policy = new PreferPolicy(++seed);
                // Play cards as long as any is legal, so that per-turn limits are reached.
                policy.Prefer = d => d.Kind == DecisionKind.Combat ? policy.RandomVerb(d, Verb.PlayCard) : -1;
                results.Add(CaseRunner.Run(session, "vetoes", name + " as " + character, character, async ctx =>
                {
                    if (preventer == "VELVET_CHOKER")
                    {
                        await ScenarioTools.GiveRelic(ctx, "VELVET_CHOKER");
                    }
                    await ScenarioTools.JumpToEncounter(ctx, "BYRDONIS_ELITE");
                    await ctx.Quiescence();
                    await ScenarioTools.SetResources(ctx, 40, 0);
                    List<string> starters = CombatTestTools.StarterIds(ctx);
                    await arm(ctx);
                    for (int i = 0; i < 2; i++)
                    {
                        await ScenarioTools.AddCard(ctx, starters[i % 2], PileType.Hand);
                    }
                    if (preventer.EndsWith("_POWER", StringComparison.Ordinal) && preventer != "SLOTH_POWER" && preventer != "ENCORE_POWER")
                    {
                        await arm(ctx); // afflict the cards that were just added as well (the power stacks)
                    }
                }, policy, probe, stopWhen: ctx => probe.MaxTurn >= 4 || ctx.MapOpen, verify: (r, p) =>
                {
                    long blocked = p.VetoBy.GetValueOrDefault(preventer);
                    if (blocked == 0)
                    {
                        return $"no hand card was ever vetoed by {preventer} (vetoes seen: [{string.Join(", ", p.VetoBy.Select(kv => kv.Key + ":" + kv.Value))}], plays {p.Plays})";
                    }
                    return null;
                }, maxTurns: 8, maxDecisions: 400));
                results[results.Count - 1].Note = "vetoed hand cards seen: " + probe.VetoBy.GetValueOrDefault(preventer);
            }
        }
        return results;
    }

    private static async Task ArmAffliction(RunContext ctx, string power, string affliction)
    {
        await ScenarioTools.ApplyPower(ctx, power, 1, ctx.LocalPlayer.Creature);
        foreach (CardModel card in ctx.LocalPlayer.PlayerCombatState!.Hand.Cards.ToList())
        {
            if (card.Affliction == null)
            {
                await ScenarioTools.Afflict(ctx, affliction, 1, card);
            }
        }
    }

    // ------------------------------------------------------------------------------------------------
    //  allcards: every card of the game goes through a hand once
    // ------------------------------------------------------------------------------------------------

    public static List<CaseResult> AllCards(HostSession session, string? only, out object summary)
    {
        List<CaseResult> results = new List<CaseResult>();
        Dictionary<string, List<CardModel>> byCharacter = new Dictionary<string, List<CardModel>>();
        foreach (CardModel card in CombatCatalog.Cards())
        {
            if (only != null && only != card.Id.Entry)
            {
                continue;
            }
            string character = CombatTestTools.CharacterFor(card, "");
            if (character.Length == 0)
            {
                // colorless / curse / status / event cards: spread over the characters
                character = CombatCatalog.Characters[Math.Abs(StableHash(card.Id.Entry)) % CombatCatalog.Characters.Length];
            }
            if (!byCharacter.TryGetValue(character, out List<CardModel>? list))
            {
                byCharacter[character] = list = new List<CardModel>();
            }
            list.Add(card);
        }
        long played = 0;
        long neverLegal = 0;
        Dictionary<string, long> reasons = new Dictionary<string, long>();
        List<string> notPlayed = new List<string>();
        ulong seed = 2000;
        foreach (string character in CombatCatalog.Characters)
        {
            if (!byCharacter.TryGetValue(character, out List<CardModel>? cards))
            {
                continue;
            }
            // Pass 1: three cards per episode. Pass 2: every card that was not played gets an episode of its own
            // (the hand was full, or an earlier card of the chunk ended the fight).
            List<string> retry = new List<string>();
            for (int at = 0; at < cards.Count; at += 3)
            {
                List<string> ids = cards.Skip(at).Take(3).Select(c => c.Id.Entry).Where(id => !AllCardsExcluded.Contains(id)).ToList();
                if (ids.Count == 0)
                {
                    continue;
                }
                results.Add(AllCardsEpisode(session, character, ids, ++seed, 30, out HashSet<string> remaining, out _));
                played += ids.Count - remaining.Count;
                retry.AddRange(remaining);
            }
            foreach (string id in retry)
            {
                results.Add(AllCardsEpisode(session, character, new List<string> { id }, ++seed, 12, out HashSet<string> remaining, out Dictionary<string, string> why));
                if (remaining.Count == 0)
                {
                    played++;
                    continue;
                }
                neverLegal++;
                string reason = why.GetValueOrDefault(id, "not seen in hand");
                string bucket = reason.Split(':')[0];
                reasons[bucket] = reasons.GetValueOrDefault(bucket) + 1;
                notPlayed.Add(id + " (" + reason + ")");
            }
        }
        summary = new { cards = played + neverLegal, played, never_legal = neverLegal, never_legal_by_reason = reasons, never_legal_cards = notPlayed };
        return results;
    }

    /// <summary>
    /// MAD_SCIENCE exists only as the configured product of TinkerTime (its type and rider are set there);
    /// a bare copy throws in its own OnPlay (Models.Cards/MadScience.cs:192-204). Not a card a run can hold.
    /// </summary>
    public static readonly HashSet<string> AllCardsExcluded = new HashSet<string> { "MAD_SCIENCE" };

    private static CaseResult AllCardsEpisode(HostSession session, string character, List<string> ids, ulong seed, int energy, out HashSet<string> remainingOut, out Dictionary<string, string> whyOut)
    {
        HashSet<string> remaining = new HashSet<string>(ids);
        Dictionary<string, string> why = new Dictionary<string, string>();
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(seed);
        policy.Prefer = d => policy.PlayOf(d, remaining);
        probe.OnEach = (ctx, d, chosen, auto) =>
        {
            if (d.Kind != DecisionKind.Combat)
            {
                return;
            }
            LegalAction a = d.Actions[chosen];
            if (a.Verb == Verb.PlayCard)
            {
                remaining.Remove(a.Source.Card.Id.Entry);
                return;
            }
            foreach (CardModel c in ctx.LocalPlayer.PlayerCombatState!.Hand.Cards)
            {
                if (remaining.Contains(c.Id.Entry))
                {
                    why[c.Id.Entry] = c.CanPlay(out UnplayableReason reason, out AbstractModel? preventer)
                        ? "NoLegalTarget"
                        : reason + (preventer != null ? ":" + preventer.Id.Entry : "");
                }
            }
        };
        CaseResult result = CaseRunner.Run(session, "allcards", character + " " + string.Join(",", ids), character, async ctx =>
        {
            await ScenarioTools.JumpToEncounter(ctx, "BYRDONIS_ELITE");
            await ctx.Quiescence();
            await ScenarioTools.SetResources(ctx, energy, 30);
            List<string> starters = CombatTestTools.StarterIds(ctx);
            await ScenarioTools.AddCard(ctx, starters[0], PileType.Discard);
            await ScenarioTools.AddCard(ctx, starters[Math.Min(1, starters.Count - 1)], PileType.Discard);
            foreach (string id in ids)
            {
                await ScenarioTools.AddCard(ctx, id, PileType.Hand);
            }
        }, policy, probe, stopWhen: ctx => remaining.Count == 0 && probe.CombatDecisions > 0 || probe.MaxTurn >= 3 || ctx.MapOpen, maxTurns: 6, maxDecisions: 400);
        remainingOut = remaining;
        whyOut = why;
        return result;
    }

    private static int StableHash(string s)
    {
        unchecked
        {
            int h = 17;
            foreach (char c in s)
            {
                h = h * 31 + c;
            }
            return h;
        }
    }

    // ------------------------------------------------------------------------------------------------
    //  drawperm: permuting the draw pile must not change a draw-pile prompt (DECISIONS-H1 B.4)
    // ------------------------------------------------------------------------------------------------

    private sealed class DrawPromptCapture
    {
        public ulong Shape;
        public List<string> Classes = new List<string>();     // "count x key" in option order
        public List<string> LocalClasses = new List<string>(); // the same for the local key (CardClassKey order)
        public List<string> RawIds = new List<string>();      // host-only: the engine's order, by test tag
        public List<string> PileOrder = new List<string>();   // host-only: draw pile order, by test tag
        public string ChosenTag = "";
        public string ExpectedTag = "";
        public int Options;
        public int Instances;
        public bool Seen;
    }

    private enum ClassOrder
    {
        Unchecked,      // the production provider (the observation encoder's content key): invariance only; the encoder checks its own order
        Local,          // CardClassKey order (test provider)
        ReversedLocal,  // reversed CardClassKey order (test provider)
    }

    public static List<CaseResult> DrawPermutation(HostSession session)
    {
        List<CaseResult> results = new List<CaseResult>();
        DriverRegistry registry = session.Registry;
        CanonicalCardClasses configured = registry.CanonicalCardClasses
            ?? throw new InvalidOperationException("drawperm: no canonical card-class provider is registered (the production set always registers the encoder's).");
        string configuredName = registry.CanonicalCardClassesName;
        // 1. the production provider (the encoder's content key, registered in every mode)
        DrawPermutationSet(session, results, "", ClassOrder.Unchecked);
        bool paused = SpireHost.Driver.Tools.FrameAudit.Paused;
        try
        {
            // TEST PROVIDERS, registered between episodes and removed in the finally block. The encoder would rightly
            // report their orders as not its own: its frame audit is paused for these cases.
            SpireHost.Driver.Tools.FrameAudit.Paused = true;
            // 2. the local CardClassKey as the provider: the classes must come out in ascending key order
            registry.SetCanonicalCardClasses(LocalTestProvider, "test:local-key");
            CombatProbe.LocalKeyOrder = true;
            DrawPermutationSet(session, results, " [test provider: local key]", ClassOrder.Local);
            CombatProbe.LocalKeyOrder = false;
            // 3. the local classes in REVERSED key order, each represented by its LAST instance
            registry.SetCanonicalCardClasses(ReversedTestProvider, "test:reversed-local-key");
            DrawPermutationSet(session, results, " [test provider: reversed]", ClassOrder.ReversedLocal);
            // 4. a provider whose answer is inconsistent must abort the episode instead of producing a frame
            registry.SetCanonicalCardClasses(BrokenTestProvider, "test:broken");
            results.Add(DrawPermutationRun(session, "SECRET_WEAPON", "IRONCLAD", "as dealt [test provider: an inconsistent provider must abort]", l => l, new DrawPromptCapture(), expectProviderAbort: true));
        }
        finally
        {
            CombatProbe.LocalKeyOrder = false;
            registry.SetCanonicalCardClasses(configured, configuredName);
            SpireHost.Driver.Tools.FrameAudit.Paused = paused;
        }
        return results;
    }

    /// <summary>TEST provider: the local CardClassKey classes in ascending key order, represented by their first instance.</summary>
    private static void LocalTestProvider(RunContext ctx, IReadOnlyList<CardModel> cards, PileType pile, int seat, bool unpoweredPreviews, List<CardModel> classes, List<int> counts)
    {
        classes.Clear();
        counts.Clear();
        CanonicalCandidates.Build(cards, out List<CardModel> options, out _, out List<List<CardModel>>? members);
        for (int i = 0; i < options.Count; i++)
        {
            classes.Add(options[i]);
            counts.Add(members![i].Count);
        }
    }

    /// <summary>TEST provider: the local classes in reversed key order, represented by their last instance (the frame must not depend on which instance represents a class).</summary>
    private static void ReversedTestProvider(RunContext ctx, IReadOnlyList<CardModel> cards, PileType pile, int seat, bool unpoweredPreviews, List<CardModel> classes, List<int> counts)
    {
        classes.Clear();
        counts.Clear();
        CanonicalCandidates.Build(cards, out List<CardModel> options, out _, out List<List<CardModel>>? members);
        for (int i = options.Count - 1; i >= 0; i--)
        {
            List<CardModel> m = members![i];
            classes.Add(m[m.Count - 1]);
            counts.Add(m.Count);
        }
    }

    /// <summary>TEST provider: class sizes that do not add up.</summary>
    private static void BrokenTestProvider(RunContext ctx, IReadOnlyList<CardModel> cards, PileType pile, int seat, bool unpoweredPreviews, List<CardModel> classes, List<int> counts)
    {
        ReversedTestProvider(ctx, cards, pile, seat, unpoweredPreviews, classes, counts);
        if (counts.Count > 0)
        {
            counts[0]++;
        }
    }

    private static void DrawPermutationSet(HostSession session, List<CaseResult> results, string label, ClassOrder order)
    {
        // The same run and the same public history; only the order of the draw pile differs.
        (string name, Func<List<CardModel>, List<CardModel>> permute)[] variants =
        {
            ("as dealt", l => l),
            ("reversed", l => { l.Reverse(); return l; }),
            ("rotated", l => { List<CardModel> r = l.Skip(3).Concat(l.Take(3)).ToList(); return r; }),
            ("shuffled-a", l => Shuffle(l, 11)),
            ("shuffled-b", l => Shuffle(l, 12)),
            ("shuffled-c", l => Shuffle(l, 13)),
        };
        foreach ((string source, string character) in new[] { ("SECRET_WEAPON", "IRONCLAD"), ("SEANCE", "NECROBINDER"), ("WISH", "REGENT") })
        {
            if (!CombatCatalog.HasCard(source))
            {
                results.Add(new CaseResult { Suite = "drawperm", Name = source + label, Failure = "unknown card id" });
                continue;
            }
            List<DrawPromptCapture> captures = new List<DrawPromptCapture>();
            List<CaseResult> runs = new List<CaseResult>();
            foreach ((string name, Func<List<CardModel>, List<CardModel>> permute) in variants)
            {
                DrawPromptCapture capture = new DrawPromptCapture();
                captures.Add(capture);
                runs.Add(DrawPermutationRun(session, source, character, name + label, permute, capture));
            }
            CaseResult summary = new CaseResult { Suite = "drawperm", Name = source + ": " + variants.Length + " draw-pile orders" + label, Character = character };
            string? failure = runs.FirstOrDefault(r => !r.Passed)?.Failure;
            DrawPromptCapture first = captures[0];
            bool anyRawDiffers = false;
            bool anyPileDiffers = false;
            for (int i = 1; failure == null && i < captures.Count; i++)
            {
                DrawPromptCapture c = captures[i];
                if (c.Shape != first.Shape)
                {
                    failure = $"decision shape differs between '{variants[0].name}' and '{variants[i].name}'";
                }
                else if (!c.Classes.SequenceEqual(first.Classes))
                {
                    failure = $"candidate classes / order differ between '{variants[0].name}' and '{variants[i].name}'";
                }
                anyRawDiffers |= !c.RawIds.SequenceEqual(first.RawIds);
                anyPileDiffers |= !c.PileOrder.SequenceEqual(first.PileOrder);
            }
            if (failure == null && !anyPileDiffers)
            {
                failure = "positive control failed: the draw pile order did not change between the variants";
            }
            if (failure == null && !anyRawDiffers)
            {
                failure = "positive control failed: the engine's own candidate order did not change (no identical cards in the pile?)";
            }
            if (failure == null && first.Options >= first.Instances)
            {
                failure = "nothing was merged: the scenario has no identical draw-pile cards";
            }
            if (failure == null && order == ClassOrder.Local && !first.Classes.SequenceEqual(first.LocalClasses))
            {
                failure = "the classes are not in the order of the local key";
            }
            if (failure == null && order == ClassOrder.ReversedLocal && !first.Classes.SequenceEqual(Enumerable.Reverse(first.LocalClasses)))
            {
                failure = "the frame does not use the classes / the order of the registered class provider";
            }
            summary.Failure = failure;
            summary.Passed = failure == null;
            summary.Note = $"options {first.Options}, instances {first.Instances}, engine order changed {anyRawDiffers}, chosen instance = first of its class in pile order in all variants";
            summary.Selects = new List<object> { new { classes = first.Classes } };
            results.AddRange(runs);
            results.Add(summary);
        }
    }

    private static List<CardModel> Shuffle(List<CardModel> list, ulong seed)
    {
        SpireHost.Sim.PolicyRng rng = new SpireHost.Sim.PolicyRng(seed);
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.NextInt(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
        return list;
    }

    private static CaseResult DrawPermutationRun(HostSession session, string source, string character, string variant, Func<List<CardModel>, List<CardModel>> permute, DrawPromptCapture capture, bool expectProviderAbort = false)
    {
        CombatProbe probe = new CombatProbe();
        PreferPolicy policy = new PreferPolicy(42);
        HashSet<string> ids = new HashSet<string> { source };
        Dictionary<CardModel, string> tags = new Dictionary<CardModel, string>(ReferenceEqualityComparer.Instance);
        bool played = false;
        bool done = false;
        string? finding = null;
        // Always the first SELECT (class 0): the same public choice in every variant.
        policy.Prefer = d => d.Kind == DecisionKind.CardSelect ? policy.FirstVerb(d, Verb.SelectCard) : played ? -1 : policy.PlayOf(d, ids);
        probe.OnEach = (ctx, d, chosen, auto) =>
        {
            if (d.Kind == DecisionKind.Combat)
            {
                if (played)
                {
                    done = true;
                }
                else if (d.Actions[chosen].Verb == Verb.PlayCard && d.Actions[chosen].Source.Card.Id.Entry == source)
                {
                    played = true;
                }
                return;
            }
            if (d.Kind != DecisionKind.CardSelect || capture.Seen || d.Source is not CardSelectFrame frame || frame.Context.Pile != PileType.Draw)
            {
                return;
            }
            capture.Seen = true;
            capture.Shape = d.ShapeHash();
            CardSelectContext c = frame.Context;
            capture.Options = c.Options.Count;
            capture.Instances = c.InstanceCount;
            for (int i = 0; i < c.Options.Count; i++)
            {
                capture.Classes.Add((c.OptionCounts != null ? c.OptionCounts[i] : 1) + " x " + CardClassKey.Of(c.Options[i]));
            }
            CanonicalCandidates.Build(frame.RawOptions, out List<CardModel> localOptions, out List<int>? localCounts, out _);
            for (int i = 0; i < localOptions.Count; i++)
            {
                capture.LocalClasses.Add((localCounts != null ? localCounts[i] : 1) + " x " + CardClassKey.Of(localOptions[i]));
            }
            foreach (CardModel raw in frame.RawOptions)
            {
                capture.RawIds.Add(tags.GetValueOrDefault(raw, "?"));
            }
            IReadOnlyList<CardModel> pile = ctx.LocalPlayer.PlayerCombatState!.DrawPile.Cards;
            foreach (CardModel card in pile)
            {
                capture.PileOrder.Add(tags.GetValueOrDefault(card, "?"));
            }
            // The chosen class maps to its first instance in the order the game handed over = first in pile order.
            LegalAction a = d.Actions[chosen];
            if (a.Verb == Verb.SelectCard)
            {
                string key = CardClassKey.Of(a.Source.Card);
                CardModel? firstInPile = pile.FirstOrDefault(x => CardClassKey.Of(x) == key);
                capture.ChosenTag = tags.GetValueOrDefault(a.Source.Card, "?");
                capture.ExpectedTag = firstInPile != null ? tags.GetValueOrDefault(firstInPile, "?") : "-";
                if (!ReferenceEquals(firstInPile, a.Source.Card))
                {
                    finding = "the representative of the chosen class is not its first instance in pile order";
                }
            }
        };
        return CaseRunner.Run(session, "drawperm", source + " / " + variant, character, async ctx =>
        {
            await ScenarioTools.JumpToEncounter(ctx, CaseRunner.DefaultEncounter);
            await ctx.Quiescence();
            await ScenarioTools.SetResources(ctx, 20, 20);
            List<string> starters = CombatTestTools.StarterIds(ctx);
            // Identical copies, an upgraded copy (a different class) and distinct cards.
            for (int i = 0; i < 3; i++)
            {
                await ScenarioTools.AddCard(ctx, starters[0], PileType.Draw);
            }
            await ScenarioTools.AddCard(ctx, starters[0], PileType.Draw, upgrades: 1);
            await ScenarioTools.AddCard(ctx, starters[Math.Min(1, starters.Count - 1)], PileType.Draw);
            await ScenarioTools.AddCard(ctx, starters[Math.Min(1, starters.Count - 1)], PileType.Draw);
            await ScenarioTools.AddCard(ctx, "HEADBUTT", PileType.Draw);
            await ScenarioTools.AddCard(ctx, source, PileType.Hand);
            // TEST ONLY: reorder the hidden draw pile in place (same multiset, same public state).
            List<CardModel> pile = ctx.LocalPlayer.PlayerCombatState!.DrawPile._cards;
            for (int i = 0; i < pile.Count; i++)
            {
                tags[pile[i]] = pile[i].Id.Entry + "+" + pile[i].CurrentUpgradeLevel + "#" + i; // tag = position as dealt
            }
            List<CardModel> reordered = permute(new List<CardModel>(pile));
            pile.Clear();
            pile.AddRange(reordered);
        }, policy, probe, stopWhen: ctx => done || ctx.MapOpen, verify: (r, p) =>
        {
            if (expectProviderAbort)
            {
                bool aborted = r.Outcome == EpisodeOutcome.Aborted && r.Abort != null && r.Abort.Signature.Contains("card_select_class_provider", StringComparison.Ordinal);
                return aborted ? null : "an inconsistent class provider did not abort the episode (outcome " + r.Outcome + ", abort " + (r.Abort?.Signature ?? "-") + ")";
            }
            if (!capture.Seen)
            {
                return "no draw-pile prompt was raised";
            }
            return finding;
        }, maxTurns: 4, maxDecisions: 200, seedLabel: "drawperm:" + source,
            tolerateAbort: expectProviderAbort ? info => info.Signature.Contains("card_select_class_provider", StringComparison.Ordinal) : null);
    }

    // ------------------------------------------------------------------------------------------------
    //  sweep: all encounters x 6 characters, random loadouts
    // ------------------------------------------------------------------------------------------------

    public static List<CaseResult> Sweep(HostSession session, int hp, string? onlyEncounter, string? onlyCharacter, int shard, int shards, int turnCap, int variant, out object summary)
    {
        List<CaseResult> results = new List<CaseResult>();
        List<string> encounters = CombatCatalog.Encounters();
        List<CardModel> allCards = CombatCatalog.Cards().Where(c => !AllCardsExcluded.Contains(c.Id.Entry)).ToList();
        List<RelicModel> allRelics = CombatCatalog.Relics().Where(r => !SweepExcludedRelics.Contains(r.Id.Entry)).ToList();
        List<PotionModel> allPotions = CombatCatalog.Potions();
        Dictionary<string, long> outcomes = new Dictionary<string, long>();
        long index = 0;
        foreach (string encounter in encounters)
        {
            if (onlyEncounter != null && onlyEncounter != encounter)
            {
                continue;
            }
            foreach (string character in CombatCatalog.Characters)
            {
                if (onlyCharacter != null && onlyCharacter != character)
                {
                    continue;
                }
                long i = index++;
                if (shards > 1 && i % shards != shard)
                {
                    continue;
                }
                ulong seed = (ulong)(StableHash(encounter + "/" + character + (variant != 0 ? "#" + variant : "")) & 0x7FFFFFFF) + (ulong)hp;
                SpireHost.Sim.PolicyRng rng = new SpireHost.Sim.PolicyRng(seed * 7919 + 1);
                List<(string id, int upgrades)> cards = new List<(string, int)>();
                for (int k = 0; k < 8; k++)
                {
                    cards.Add((allCards[rng.NextInt(allCards.Count)].Id.Entry, rng.NextInt(2)));
                }
                List<string> relics = new List<string>();
                for (int k = 0; k < 3; k++)
                {
                    relics.Add(allRelics[rng.NextInt(allRelics.Count)].Id.Entry);
                }
                List<string> potions = new List<string>();
                for (int k = 0; k < 2; k++)
                {
                    potions.Add(allPotions[rng.NextInt(allPotions.Count)].Id.Entry);
                }
                CombatProbe probe = new CombatProbe();
                PreferPolicy policy = new PreferPolicy(seed, randomCombat: true);
                string loadout = "cards " + string.Join(",", cards.Select(c => c.id + (c.upgrades > 0 ? "+" : ""))) + " relics " + string.Join(",", relics.Distinct()) + " potions " + string.Join(",", potions);
                CaseResult result = CaseRunner.Run(session, "sweep", encounter + " / " + character + (hp > 0 ? " hp" + hp : " natural") + (variant != 0 ? " v" + variant : ""), character, async ctx =>
                {
                    foreach ((string id, int upgrades) in cards)
                    {
                        await ScenarioTools.GiveCard(ctx, id, upgrades);
                    }
                    foreach (string relic in relics.Distinct())
                    {
                        // The game's own gate for "may this relic be offered in this run" (RelicModel.IsAllowed :401;
                        // e.g. MassiveScroll and other multiplayer-only relics with one player).
                        if (ScenarioTools.Relic(relic).IsAllowed(ctx.Run))
                        {
                            await ScenarioTools.GiveRelic(ctx, relic);
                        }
                    }
                    foreach (string potion in potions)
                    {
                        await ScenarioTools.GivePotion(ctx, potion);
                    }
                    await ScenarioTools.JumpToEncounter(ctx, encounter);
                }, policy, probe, verify: null, hp: hp, maxTurns: turnCap, maxDecisions: 6000);
                result.Note = loadout;
                string key = result.Outcome == "Truncated" ? "Truncated:" + result.TruncatedBy : result.Outcome;
                outcomes[key] = outcomes.GetValueOrDefault(key) + 1;
                results.Add(result);
            }
        }
        summary = new { hp, variant, episodes = results.Count, outcomes };
        return results;
    }

    /// <summary>
    /// Relics left out of the random loadouts because obtaining them fails OUTSIDE the combat / selection flow
    /// (sites owned by other teams; each one is listed in notes\host-h1-combat.md).
    /// </summary>
    public static readonly HashSet<string> SweepExcludedRelics = new HashSet<string>();
}
