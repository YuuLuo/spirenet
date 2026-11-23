using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Boot;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

// =========================================================================================================
//  TEST ONLY: observers and policies used by the rooms self-tests (never registered in a module).
// =========================================================================================================

/// <summary>
/// Looks at every decision exactly as the encoder will (after the central rules) and checks the rooms contract:
/// context type per kind, REWARD / CARD_REWARD ordinals point at the live objects of the context, gold is flagged
/// AutoGold, PROCEED respects DisallowSkipping, CARD_REWARD is nested, potion verbs only on non-nested room frames
/// and USE only for AnyTime potions out of combat.
/// </summary>
internal sealed class RoomsObserver : IEpisodeObserver
{
    public readonly List<string> Violations = new List<string>();
    public readonly SortedDictionary<string, long> Kinds = new SortedDictionary<string, long>(StringComparer.Ordinal);
    public readonly SortedDictionary<string, long> PolicyKinds = new SortedDictionary<string, long>(StringComparer.Ordinal);
    public long PotionActionsOffered;
    private long _seen;
    public long RestChooseAfterSuccess;
    public long TreasureProceedFrames;
    public long ChestFrames;
    public long ChestFramesAuto;
    public long ShopFrames;
    public readonly SortedDictionary<string, long> PotionActionsByKind = new SortedDictionary<string, long>(StringComparer.Ordinal);

    /// <summary>Scenario-specific hook, called for every decision after the generic checks.</summary>
    public Action<RunContext, Decision, int, bool>? OnEach;

    public void OnEpisodeStart(RunContext ctx)
    {
    }

    public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
    {
    }

    public void OnEpisodeEnd(EpisodeResult result)
    {
    }

    private void Violation(string s)
    {
        if (Violations.Count < 20)
        {
            Violations.Add(s);
        }
    }

    public void OnDecision(RunContext ctx, Decision d, int chosen, bool auto)
    {
        // the observation encoder's own consistency checks on every decision (test seam of the driver)
        string? audit = SpireHost.Driver.Tools.FrameAudit.Run(d);
        if (audit != null)
        {
            Violation("frame audit: " + audit);
        }
        string kind = d.Kind.ToString();
        Kinds[kind] = Kinds.GetValueOrDefault(kind) + 1;
        if (++_seen % 5000 == 0)
        {
            HostLog.Info("rooms observer: " + _seen + " decisions; " + ctx.Diagnose()); // --verbose only
        }
        if (!auto)
        {
            PolicyKinds[kind] = PolicyKinds.GetValueOrDefault(kind) + 1;
        }
        bool roomFrame = d.Kind is DecisionKind.Map or DecisionKind.Shop or DecisionKind.Rest or DecisionKind.Event or DecisionKind.Reward or DecisionKind.Treasure;
        List<LegalAction> actions = d.Actions;
        for (int i = 0; i < actions.Count; i++)
        {
            LegalAction a = actions[i];
            if (a.Verb is Verb.UsePotion or Verb.DiscardPotion && d.Kind != DecisionKind.Combat)
            {
                PotionActionsOffered++;
                string pk = kind + ":" + a.Verb;
                PotionActionsByKind[pk] = PotionActionsByKind.GetValueOrDefault(pk) + 1;
                if (!roomFrame || d.Nested)
                {
                    Violation($"potion action on {d.Kind} (nested={d.Nested})");
                }
                if (a.Verb == Verb.UsePotion)
                {
                    PotionModel? p = a.Source.PotionOwner.PotionSlots[a.Source.Index];
                    if (p == null || p.Usage != PotionUsage.AnyTime)
                    {
                        Violation($"USE offered out of combat for {p?.Id.Entry ?? "empty slot"} (usage {p?.Usage})");
                    }
                }
            }
        }
        switch (d.Kind)
        {
            case DecisionKind.Reward:
                CheckReward(d);
                break;
            case DecisionKind.CardReward:
                CheckCardReward(d);
                break;
            case DecisionKind.Shop:
                ShopFrames++;
                if (d.Context is not ShopContext { Inventory: not null })
                {
                    Violation("SHOP without ShopContext");
                }
                break;
            case DecisionKind.Rest:
                if (d.Context is not RestContext rc)
                {
                    Violation("REST without RestContext");
                }
                else if (rc.Successes > 0 && actions.Exists(x => x.Verb == Verb.ChooseOption))
                {
                    RestChooseAfterSuccess++;
                }
                break;
            case DecisionKind.Treasure:
                if (d.Context is not TreasureContext tc || tc.Room == null)
                {
                    Violation("TREASURE without TreasureContext");
                }
                else if (d.SubKind == (int)TreasurePhase.Chest)
                {
                    ChestFrames++;
                    if (auto)
                    {
                        ChestFramesAuto++;
                    }
                    if (tc.Relics.Count > 0 || tc.GoldGained != 0)
                    {
                        Violation("CHEST page exposes relics / gold before the chest is opened");
                    }
                    if (!actions.Exists(x => x.Verb == Verb.OpenChest) || actions.Exists(x => x.Verb is not (Verb.OpenChest or Verb.UsePotion or Verb.DiscardPotion)))
                    {
                        Violation("CHEST page must offer OPEN_CHEST and nothing but potion actions besides");
                    }
                }
                else if (actions.Exists(x => x.Verb == Verb.Proceed))
                {
                    TreasureProceedFrames++;
                    if (actions.Exists(x => x.Verb is Verb.ChooseOption or Verb.Skip or Verb.OpenChest))
                    {
                        Violation("TREASURE proceed page offers a relic / skip / open action");
                    }
                }
                else if (actions.Exists(x => x.Verb == Verb.OpenChest))
                {
                    Violation("OPEN_CHEST offered on the RELIC page");
                }
                break;
        }
        OnEach?.Invoke(ctx, d, chosen, auto);
    }

    private void CheckReward(Decision d)
    {
        if (d.Context is not RewardContext rc || rc.Set == null)
        {
            Violation("REWARD without RewardContext");
            return;
        }
        if (d.Nested)
        {
            Violation("REWARD asked as a nested decision");
        }
        bool anyUntaken = false;
        foreach (Reward r in rc.Set.Rewards)
        {
            anyUntaken |= !r.SuccessfullySelected;
        }
        foreach (LegalAction a in d.Actions)
        {
            if (a.Verb == Verb.ChooseOption)
            {
                if (a.Source.Kind != RefKind.Reward || a.Source.Index >= rc.Set.Rewards.Count || !ReferenceEquals(rc.Set.Rewards[a.Source.Index], a.Source.Object))
                {
                    Violation("REWARD action ordinal does not index RewardsSet.Rewards");
                    continue;
                }
                Reward r = a.Source.Reward;
                if (r.SuccessfullySelected)
                {
                    Violation("REWARD offers an already taken reward");
                }
                if ((r is GoldReward) != ((a.Flags & ActionFlags.AutoGold) != 0))
                {
                    Violation("AutoGold flag mismatch on " + r.GetType().Name);
                }
            }
            else if (a.Verb == Verb.Proceed && rc.Set.DisallowSkipping && anyUntaken)
            {
                Violation("PROCEED offered on a DisallowSkipping set with untaken rewards");
            }
        }
    }

    private void CheckCardReward(Decision d)
    {
        if (d.Context is not CardRewardContext cc || cc.Reward == null)
        {
            Violation("CARD_REWARD without CardRewardContext");
            return;
        }
        if (!d.Nested)
        {
            Violation("CARD_REWARD not nested");
        }
        foreach (LegalAction a in d.Actions)
        {
            if (a.Verb == Verb.PickOfferCard)
            {
                if (a.Source.Index >= cc.Options.Count || !ReferenceEquals(cc.Options[a.Source.Index].Card, a.Source.Object))
                {
                    Violation("PICK_OFFER_CARD ordinal does not index CardRewardContext.Options");
                }
            }
            else if (a.Verb == Verb.ChooseOption)
            {
                if (a.Source.Kind != RefKind.CardRewardAlt || a.Source.Index >= cc.Alternatives.Count || !ReferenceEquals(cc.Alternatives[a.Source.Index], a.Source.Object))
                {
                    Violation("card reward alternative ordinal does not index CardRewardContext.Alternatives");
                }
            }
            else
            {
                Violation("unexpected verb on CARD_REWARD: " + a.Verb);
            }
        }
    }
}

/// <summary>
/// TEST ONLY. Picks uniformly among the actions matching the first preference that matches anything, else
/// uniformly among all (SplitMix64, independent of every game RNG). Optionally sets every living enemy to 1 HP at the
/// first COMBAT decision of each combat room (so that scripted room tests reach the reward screens quickly).
/// </summary>
internal sealed class PreferPolicy : IPolicy
{
    private readonly List<Func<Decision, LegalAction, bool>> _prefs = new List<Func<Decision, LegalAction, bool>>();
    private readonly List<int> _match = new List<int>(16);
    private ulong _state;
    private object? _weakenedRoom;

    public bool WeakenEnemies;

    public PreferPolicy(ulong seed, bool weakenEnemies = false)
    {
        _state = seed ^ 0xA5A5_5A5A_1234_5678UL;
        WeakenEnemies = weakenEnemies;
    }

    public PreferPolicy Prefer(Func<Decision, LegalAction, bool> predicate)
    {
        _prefs.Add(predicate);
        return this;
    }

    private int Next(int n)
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (int)(z % (ulong)n);
    }

    public int Decide(Decision d)
    {
        // At every COMBAT decision, not only the first of a room: bosses that revive with fresh HP (TestSubject's
        // RespawnMove -> Revive -> SetMaxHp, Models.Monsters/TestSubject.cs:264-346) otherwise drag a forced-HP fight on
        // for tens of seconds and may still win it.
        if (WeakenEnemies && d.Kind == DecisionKind.Combat && d.Episode.CurrentRoom is CombatRoom room)
        {
            _weakenedRoom = room;
            foreach (Creature c in room.CombatState.Enemies)
            {
                if (c.IsAlive && c.CurrentHp > 1)
                {
                    c.SetCurrentHpInternal(1); // TEST ONLY: shortens the fight; the room flow under test starts after it
                }
            }
        }
        foreach (Func<Decision, LegalAction, bool> pred in _prefs)
        {
            _match.Clear();
            for (int i = 0; i < d.Actions.Count; i++)
            {
                if (pred(d, d.Actions[i]))
                {
                    _match.Add(i);
                }
            }
            if (_match.Count > 0)
            {
                return _match[Next(_match.Count)];
            }
        }
        return Next(d.Actions.Count);
    }

    // ---- common predicates ----
    public static bool IsReward<T>(Decision d, LegalAction a) where T : Reward => d.Kind == DecisionKind.Reward && a.Verb == Verb.ChooseOption && a.Source.Object is T;

    public static bool TakeAnyReward(Decision d, LegalAction a) => d.Kind == DecisionKind.Reward && a.Verb == Verb.ChooseOption;

    public static bool Alt(Decision d, LegalAction a, string optionId) => d.Kind == DecisionKind.CardReward && a.Verb == Verb.ChooseOption && a.Source.CardRewardAlt.OptionId == optionId;

    public static bool PickCard(Decision d, LegalAction a) => d.Kind == DecisionKind.CardReward && a.Verb == Verb.PickOfferCard;

    public static bool Verb_(Decision d, LegalAction a, DecisionKind kind, Verb verb) => d.Kind == kind && a.Verb == verb;

    public static bool UsePotion<T>(Decision d, LegalAction a) where T : PotionModel
    {
        return a.Verb == Verb.UsePotion && a.Source.PotionOwner.PotionSlots[a.Source.Index] is T;
    }

    public static Player Me(Decision d) => d.Episode.LocalPlayer;
}
