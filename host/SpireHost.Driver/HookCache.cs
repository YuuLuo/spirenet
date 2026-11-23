using System;
using System.Collections.Generic;
using System.Reflection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers.Models;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Driver;

/// <summary>
/// Encoder cost: per-frame cached hook-listener dispatch for four read-only display values of a card.
///
/// Why: every game hook call enumerates CombatState.IterateHookListeners(), which builds a fresh list of every power,
/// relic, potion, orb, card (of every pile), enchantment and affliction in the combat and filters it with
/// CombatState.Contains (Combat/CombatState.cs:25-200). The encoder evaluates the printed energy cost, star cost,
/// keywords and the two cost colours for every card of every pile, i.e. 3-7 such enumerations per card and frame:
/// measured 60-70% of the encoder's time before this class existed (notes/host-h1-obs.md 4.3).
///
/// What: at the start of a frame the listener enumeration is materialised ONCE, in the game's own order, and split into
/// the sub-sequences of listeners whose runtime type overrides the hook method in question (reflection once per type,
/// cached for the process). The methods below are line-by-line transcriptions of the game methods named on each (same
/// early returns, same order of listeners, same decimal arithmetic, same Math.Max); a listener that does not override
/// the method runs AbstractModel's identity implementation (AbstractModel.cs:878-893), so leaving it out changes nothing.
/// The engine is quiescent while the encoder runs and the encoder does not change the listener set (the upgrade-preview
/// clone is in no pile, so it is never a listener), hence the cached sequence equals the one every call would build.
///
/// Proof: with <see cref="Exact"/> = true every method calls the game method instead; the --selftest fuzz encodes every
/// frame both ways and requires identical bytes (case "fuzz", counter exact_mismatch). A card whose CombatState is not the
/// frame's combat state (none in v0.113) takes the exact path.
///
/// No game state is written here (TryModify* implementations only compute; the scratch keyword set is the encoder's own).
/// </summary>
internal sealed partial class HookCache
{
    [Flags]
    private enum Ov
    {
        None = 0,
        Energy = 1,
        EnergyLate = 2,
        Star = 4,
        Keywords = 8,
        // HookCache.Playability.cs
        ShouldPlay = 8192,
        // HookCache.Previews.cs
        DamageAdditive = 16,
        DamageMultiplicative = 32,
        DamageMultiplicativeLate = 64,
        DamageCap = 128,
        BlockAdditive = 256,
        BlockMultiplicative = 512,
        PowerGivenAdditive = 1024,
        PowerGivenMultiplicative = 2048,
        SummonAmount = 4096,
        MaxEnergy = 16384,
        AllowHitting = 32768,
        PayExcessWithStars = 65536,
    }

    private static readonly Dictionary<Type, Ov> s_overrides = new Dictionary<Type, Ov>();
    private static readonly Type[] s_costSig = { typeof(CardModel), typeof(decimal), typeof(decimal).MakeByRefType() };
    private static readonly Type[] s_kwSig = { typeof(CardModel), typeof(ISet<CardKeyword>) };

    private ICombatState? _state;
    private bool _skipCombatEffects;
    private readonly List<AbstractModel> _energy = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _energyLate = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _star = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _keywords = new List<AbstractModel>(8);
    private readonly HashSet<CardKeyword> _kwScratch = new HashSet<CardKeyword>();

    /// <summary>Call the game methods themselves (reference mode of the equivalence test).</summary>
    public bool Exact;

    public long Listeners;

    public void Begin(ICombatState? state)
    {
        End();
        if (state == null || Exact)
        {
            return;
        }
        _state = state;
        // Hook.IterateCombatHookListeners (Hooks/Hook.cs:178 + its iterator): nothing when the combat is ending
        _skipCombatEffects = CombatManager.Instance.ShouldSkipCombatEffect;
        foreach (AbstractModel m in state.IterateHookListeners())
        {
            Listeners++;
            Ov o = OverridesOf(m.GetType());
            if (o == Ov.None)
            {
                continue;
            }
            if ((o & Ov.Energy) != 0)
            {
                _energy.Add(m);
            }
            if ((o & Ov.EnergyLate) != 0)
            {
                _energyLate.Add(m);
            }
            if ((o & Ov.Star) != 0)
            {
                _star.Add(m);
            }
            if ((o & Ov.Keywords) != 0)
            {
                _keywords.Add(m);
            }
            AddCombatPreviewListener(m, o);
            if ((o & Ov.ShouldPlay) != 0)
            {
                _shouldPlay.Add(m);
            }
        }
    }

    public void End()
    {
        _state = null;
        _energy.Clear();
        _energyLate.Clear();
        _star.Clear();
        _keywords.Clear();
        EndPreviews();
        _shouldPlay.Clear();
    }

    private static Ov OverridesOf(Type t)
    {
        if (s_overrides.TryGetValue(t, out Ov o))
        {
            return o;
        }
        o = Ov.None;
        if (Overrides(t, nameof(AbstractModel.TryModifyEnergyCostInCombat), s_costSig))
        {
            o |= Ov.Energy;
        }
        if (Overrides(t, nameof(AbstractModel.TryModifyEnergyCostInCombatLate), s_costSig))
        {
            o |= Ov.EnergyLate;
        }
        if (Overrides(t, nameof(AbstractModel.TryModifyStarCost), s_costSig))
        {
            o |= Ov.Star;
        }
        if (Overrides(t, nameof(AbstractModel.TryModifyKeywordsInCombat), s_kwSig))
        {
            o |= Ov.Keywords;
        }
        o |= PreviewOverridesOf(t);
        if (Overrides(t, nameof(AbstractModel.ShouldPlay), s_shouldPlaySig))
        {
            o |= Ov.ShouldPlay;
        }
        s_overrides[t] = o;
        return o;
    }

    private static bool Overrides(Type t, string name, Type[] sig)
    {
        MethodInfo? m = t.GetMethod(name, BindingFlags.Instance | BindingFlags.Public, null, sig, null);
        // a method that cannot be resolved is treated as overridden (the listener is called: always exact)
        return m == null || m.DeclaringType != typeof(AbstractModel);
    }

    private bool Cached(ICombatState? state) => state != null && ReferenceEquals(state, _state);

    // ---------------------------------------------------------------------------------------------- energy cost

    /// <summary>= card.EnergyCost.GetWithModifiers(CostModifiers.All) (Entities.Cards/CardEnergyCost.cs:56-83).</summary>
    public int EnergyCost(CardModel card)
    {
        CardEnergyCost ec = card.EnergyCost;
        if (Exact)
        {
            return ec.GetWithModifiers(CostModifiers.All);
        }
        int num = ec._base;
        if (card.IsCanonical)
        {
            return num;
        }
        if (ec._base < 0)
        {
            return num;
        }
        if (ec.CostsX)
        {
            return num;
        }
        List<LocalCostModifier> local = ec._localModifiers;
        for (int i = 0; i < local.Count; i++)
        {
            num = local[i].Modify(num);
        }
        ICombatState? cs = card.CombatState;
        if (cs != null)
        {
            num = (int)ModifyEnergyCostInCombat(cs, card, num);
        }
        return Math.Max(0, num);
    }

    /// <summary>= Hook.ModifyEnergyCostInCombat (Hooks/Hook.cs:1428-1444).</summary>
    private decimal ModifyEnergyCostInCombat(ICombatState cs, CardModel card, decimal originalCost)
    {
        if (!Cached(cs))
        {
            return Hook.ModifyEnergyCostInCombat(cs, card, originalCost);
        }
        if (originalCost < 0m)
        {
            return originalCost;
        }
        decimal modifiedCost = originalCost;
        if (_skipCombatEffects)
        {
            return modifiedCost;
        }
        for (int i = 0; i < _energy.Count; i++)
        {
            _energy[i].TryModifyEnergyCostInCombat(card, modifiedCost, out modifiedCost);
        }
        for (int i = 0; i < _energyLate.Count; i++)
        {
            _energyLate[i].TryModifyEnergyCostInCombatLate(card, modifiedCost, out modifiedCost);
        }
        return modifiedCost;
    }

    // ---------------------------------------------------------------------------------------------- star cost

    /// <summary>= card.GetStarCostWithModifiers() (Models/CardModel.cs:1382-1394).</summary>
    public int StarCost(CardModel card)
    {
        if (Exact)
        {
            return card.GetStarCostWithModifiers();
        }
        if (card.HasStarCostX)
        {
            return card.Owner.PlayerCombatState?.Stars ?? 0;
        }
        CardPile? pile = card.Pile;
        ICombatState? cs;
        if (pile != null && pile.IsCombatPile && (cs = card.CombatState) != null)
        {
            return (int)ModifyStarCost(cs, card, card.CurrentStarCost);
        }
        return card.CurrentStarCost;
    }

    /// <summary>= Hook.ModifyStarCost (Hooks/Hook.cs:1820-1832).</summary>
    private decimal ModifyStarCost(ICombatState cs, CardModel card, decimal originalCost)
    {
        if (!Cached(cs))
        {
            return Hook.ModifyStarCost(cs, card, originalCost);
        }
        if (originalCost < 0m)
        {
            return originalCost;
        }
        decimal modifiedCost = originalCost;
        if (_skipCombatEffects)
        {
            return modifiedCost;
        }
        for (int i = 0; i < _star.Count; i++)
        {
            _star[i].TryModifyStarCost(card, modifiedCost, out modifiedCost);
        }
        return modifiedCost;
    }

    // ---------------------------------------------------------------------------------------------- keywords

    /// <summary>
    /// Bits of card.Keywords = GetKeywordsWithSources(KeywordSources.All) (Models/CardModel.cs:1003-1031, hook
    /// Hook.ModifyKeywordsInCombat:1446-1452 over CombatState.IterateHookListeners, no skip check), with the display
    /// semantics of CardModel.GetDescriptionForPile: Retain = ShouldRetainThisTurn, Sly = IsSlyThisTurn (:546-581).
    /// </summary>
    public int KeywordBits(CardModel card)
    {
        IReadOnlySet<CardKeyword> kw = Keywords(card);
        int bits = 0;
        for (int k = 1; k <= 7; k++)
        {
            CardKeyword key = (CardKeyword)k;
            bool on = kw.Contains(key) || (key == CardKeyword.Retain && card.HasSingleTurnRetain) || (key == CardKeyword.Sly && card.HasSingleTurnSly);
            if (on)
            {
                bits |= 1 << (k - 1);
            }
        }
        return bits;
    }

    /// <summary>
    /// = card.Keywords = GetKeywordsWithSources(KeywordSources.All) (Models/CardModel.cs:1003-1031). The returned set is the
    /// cache's scratch set when the hook ran here: valid until the next call.
    /// </summary>
    public IReadOnlySet<CardKeyword> Keywords(CardModel card)
    {
        if (Exact || card.IsCanonical)
        {
            return card.Keywords;
        }
        ICombatState? cs = card.CombatState;
        if (cs == null)
        {
            return card.LocalKeywords;
        }
        if (!Cached(cs))
        {
            return card.Keywords;
        }
        HashSet<CardKeyword> set = _kwScratch;
        set.Clear();
        foreach (CardKeyword k in card.LocalKeywords)
        {
            set.Add(k);
        }
        for (int i = 0; i < _keywords.Count; i++)
        {
            _keywords[i].TryModifyKeywordsInCombat(card, set);
        }
        return set;
    }

    // ---------------------------------------------------------------------------------------------- cost colours (hand)

    /// <summary>
    /// = CardCostHelper.GetEnergyCostColor(card, state) (Helpers.Models/CardCostHelper.cs:9-31) given the card's
    /// CanPlay result (the game method evaluates CanPlay itself; it is a pure function of the quiescent state).
    /// </summary>
    public CardCostColor EnergyColor(CardModel card, ICombatState? state, bool canPlay, UnplayableReason reason)
    {
        if (Exact || !Cached(state))
        {
            return CardCostHelper.GetEnergyCostColor(card, state);
        }
        if (!canPlay && reason.HasFlag(UnplayableReason.EnergyCostTooHigh))
        {
            return CardCostColor.InsufficientResources;
        }
        CardEnergyCost ec = card.EnergyCost;
        if (ec.CostsX)
        {
            return CardCostColor.Unmodified;
        }
        // TryModifyEnergyCostWithHooks (:86-99): state.IterateHookListeners(), no skip check, flag = any listener reported
        decimal cost = ec.GetWithModifiers(CostModifiers.None);
        bool flag = false;
        for (int i = 0; i < _energy.Count; i++)
        {
            flag |= _energy[i].TryModifyEnergyCostInCombat(card, cost, out cost);
        }
        for (int i = 0; i < _energyLate.Count; i++)
        {
            flag |= _energyLate[i].TryModifyEnergyCostInCombatLate(card, cost, out cost);
        }
        if (flag)
        {
            return ColorForHookModifiedCost(cost, ec.GetWithModifiers(CostModifiers.None));
        }
        if (ec.HasLocalModifiers)
        {
            return ColorForLocalCost(ec.GetWithModifiers(CostModifiers.Local), ec.GetWithModifiers(CostModifiers.None));
        }
        return CardCostColor.Unmodified;
    }

    /// <summary>= CardCostHelper.GetStarCostColor(card, state) (:33-55), given the card's CanPlay result.</summary>
    public CardCostColor StarColor(CardModel card, ICombatState? state, bool canPlay, UnplayableReason reason)
    {
        if (Exact || !Cached(state))
        {
            return CardCostHelper.GetStarCostColor(card, state);
        }
        if (!canPlay && reason.HasFlag(UnplayableReason.StarCostTooHigh))
        {
            return CardCostColor.InsufficientResources;
        }
        if (card.HasStarCostX)
        {
            return CardCostColor.Unmodified;
        }
        // TryModifyStarCostWithHooks (:101-110)
        decimal cost = card.BaseStarCost;
        bool flag = false;
        for (int i = 0; i < _star.Count; i++)
        {
            flag |= _star[i].TryModifyStarCost(card, cost, out cost);
        }
        if (flag)
        {
            return ColorForHookModifiedCost(cost, card.BaseStarCost);
        }
        if (card.TemporaryStarCost != null)
        {
            return ColorForLocalCost(card.TemporaryStarCost.Cost, card.BaseStarCost);
        }
        return CardCostColor.Unmodified;
    }

    // CardCostHelper.GetColorForLocalCost / GetColorForHookModifiedCost (:57-81)
    private static CardCostColor ColorForLocalCost(int localCost, int baseCost)
    {
        if (localCost > baseCost)
        {
            return CardCostColor.Increased;
        }
        if (localCost < baseCost)
        {
            return CardCostColor.Decreased;
        }
        return CardCostColor.Unmodified;
    }

    private static CardCostColor ColorForHookModifiedCost(decimal hookModifiedCost, int baseCost)
    {
        if (hookModifiedCost > baseCost)
        {
            return CardCostColor.Increased;
        }
        if (hookModifiedCost < baseCost)
        {
            return CardCostColor.Decreased;
        }
        return CardCostColor.Unmodified;
    }
}
