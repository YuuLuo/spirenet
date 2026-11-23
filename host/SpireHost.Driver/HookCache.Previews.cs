using System;
using System.Collections.Generic;
using System.Reflection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.ValueProps;

namespace SpireHost.Driver;

/// <summary>
/// Encoder cost, second part: the printed numbers of hand cards (CardModel.UpdateDynamicVarPreview with global hooks, once
/// untargeted and once per preview target) and the per-seat intent damage (AttackIntent.GetSingleDamage). Every damage
/// preview is Hook.ModifyDamage, which enumerates RunState.IterateHookListeners(combat) - every deck card of every
/// player, then CombatState.IterateHookListeners() (every power, relic, potion, orb, combat card ...) - four times; block
/// and power previews enumerate the combat listeners twice. Measured (notes/speed/host.md): the hand rows were 66 % of the
/// encoder and 39 % of the host's CPU in 4-seat games.
///
/// Same method as the first part (HookCache.cs): the run-level enumeration is materialised once per frame (lazily, in the
/// game's order) and split into the listeners whose runtime type overrides the hook method in question; the combat-level
/// one is the first part's. A listener that does not override runs AbstractModel's identity implementation
/// (ModifyDamageAdditive 0, ModifyDamageMultiplicative / Late 1, ModifyDamageCap decimal.MaxValue, ModifyBlockAdditive 0,
/// ModifyBlockMultiplicative 1, ModifyPowerAmountGivenAdditive 0, ...Multiplicative 1, ModifySummonAmount amount:
/// Models/AbstractModel.cs:653-819); x + 0m and x * 1m keep value and scale, so leaving it out changes nothing. The methods
/// below are line-by-line transcriptions of the game methods named on each; every overriding listener is called with the
/// same arguments in the same order as in the game, so even a listener with side effects behaves the same. The modifier
/// lists the game methods also return are not built: every caller transcribed here discards them.
///
/// Dispatch is on the exact runtime type of each DynamicVar: the classes transcribed here are DamageVar, BlockVar,
/// OstyDamageVar, CalculatedDamageVar, CalculatedBlockVar, SummonVar and PowerVar&lt;T&gt;; every other variable (and any
/// subclass) runs its own UpdateCardPreview. Exact mode (EncoderOptions.ExactGameCalls) calls the game methods instead; the
/// --selftest fuzz and the speed benchmark compare both (identical bytes).
/// </summary>
internal sealed partial class HookCache
{
    private static readonly Type[] s_dmgSig = { typeof(Creature), typeof(decimal), typeof(ValueProp), typeof(Creature), typeof(CardModel), typeof(CardPlay) };
    private static readonly Type[] s_dmgCapSig = { typeof(Creature), typeof(ValueProp), typeof(Creature), typeof(CardModel), typeof(CardPlay) };
    private static readonly Type[] s_blkSig = { typeof(Creature), typeof(decimal), typeof(ValueProp), typeof(CardModel), typeof(CardPlay) };
    private static readonly Type[] s_pwrSig = { typeof(PowerModel), typeof(Creature), typeof(decimal), typeof(Creature), typeof(CardModel) };
    private static readonly Type[] s_summonSig = { typeof(Player), typeof(decimal), typeof(AbstractModel) };
    private static readonly Type[] s_maxEnergySig = { typeof(Player), typeof(decimal) };
    private static readonly Type[] s_creatureSig = { typeof(Creature) };
    private static readonly Type[] s_playerSig = { typeof(Player) };

    // combat level (CombatState.IterateHookListeners, filled by Begin; empty use when _skipCombatEffects as in the game)
    private readonly List<AbstractModel> _blockAdd = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _blockMul = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _powerAdd = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _powerMul = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _summon = new List<AbstractModel>(4);
    private readonly List<AbstractModel> _maxEnergy = new List<AbstractModel>(4);
    private readonly List<AbstractModel> _allowHitting = new List<AbstractModel>(4);
    private readonly List<AbstractModel> _payWithStars = new List<AbstractModel>(4);
    private readonly Dictionary<Type, PowerModel> _canonicalPowerFrame = new Dictionary<Type, PowerModel>();

    // run level (RunState.IterateHookListeners(_state)), built on first use in a frame
    private IRunState? _runState;
    private readonly List<AbstractModel> _dmgAdd = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _dmgMul = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _dmgMulLate = new List<AbstractModel>(8);
    private readonly List<AbstractModel> _dmgCap = new List<AbstractModel>(8);

    private readonly List<DynamicVar> _varScratch = new List<DynamicVar>(8);
    private static readonly Dictionary<Type, Func<PowerModel>> s_canonicalPower = new Dictionary<Type, Func<PowerModel>>();

    public long RunListeners;

    private static Ov PreviewOverridesOf(Type t)
    {
        Ov o = Ov.None;
        if (Overrides(t, nameof(AbstractModel.ModifyDamageAdditive), s_dmgSig))
        {
            o |= Ov.DamageAdditive;
        }
        if (Overrides(t, nameof(AbstractModel.ModifyDamageMultiplicative), s_dmgSig))
        {
            o |= Ov.DamageMultiplicative;
        }
        if (Overrides(t, nameof(AbstractModel.ModifyDamageMultiplicativeLate), s_dmgSig))
        {
            o |= Ov.DamageMultiplicativeLate;
        }
        if (Overrides(t, nameof(AbstractModel.ModifyDamageCap), s_dmgCapSig))
        {
            o |= Ov.DamageCap;
        }
        if (Overrides(t, nameof(AbstractModel.ModifyBlockAdditive), s_blkSig))
        {
            o |= Ov.BlockAdditive;
        }
        if (Overrides(t, nameof(AbstractModel.ModifyBlockMultiplicative), s_blkSig))
        {
            o |= Ov.BlockMultiplicative;
        }
        if (Overrides(t, nameof(AbstractModel.ModifyPowerAmountGivenAdditive), s_pwrSig))
        {
            o |= Ov.PowerGivenAdditive;
        }
        if (Overrides(t, nameof(AbstractModel.ModifyPowerAmountGivenMultiplicative), s_pwrSig))
        {
            o |= Ov.PowerGivenMultiplicative;
        }
        if (Overrides(t, nameof(AbstractModel.ModifySummonAmount), s_summonSig))
        {
            o |= Ov.SummonAmount;
        }
        if (Overrides(t, nameof(AbstractModel.ModifyMaxEnergy), s_maxEnergySig))
        {
            o |= Ov.MaxEnergy;
        }
        if (Overrides(t, nameof(AbstractModel.ShouldAllowHitting), s_creatureSig))
        {
            o |= Ov.AllowHitting;
        }
        if (Overrides(t, nameof(AbstractModel.ShouldPayExcessEnergyCostWithStars), s_playerSig))
        {
            o |= Ov.PayExcessWithStars;
        }
        return o;
    }

    private void AddCombatPreviewListener(AbstractModel m, Ov o)
    {
        if ((o & Ov.BlockAdditive) != 0)
        {
            _blockAdd.Add(m);
        }
        if ((o & Ov.BlockMultiplicative) != 0)
        {
            _blockMul.Add(m);
        }
        if ((o & Ov.PowerGivenAdditive) != 0)
        {
            _powerAdd.Add(m);
        }
        if ((o & Ov.PowerGivenMultiplicative) != 0)
        {
            _powerMul.Add(m);
        }
        if ((o & Ov.SummonAmount) != 0)
        {
            _summon.Add(m);
        }
        if ((o & Ov.MaxEnergy) != 0)
        {
            _maxEnergy.Add(m);
        }
        if ((o & Ov.AllowHitting) != 0)
        {
            _allowHitting.Add(m);
        }
        if ((o & Ov.PayExcessWithStars) != 0)
        {
            _payWithStars.Add(m);
        }
    }

    private void EndPreviews()
    {
        _blockAdd.Clear();
        _blockMul.Clear();
        _powerAdd.Clear();
        _powerMul.Clear();
        _summon.Clear();
        _maxEnergy.Clear();
        _allowHitting.Clear();
        _payWithStars.Clear();
        _canonicalPowerFrame.Clear();
        _runState = null;
        _dmgAdd.Clear();
        _dmgMul.Clear();
        _dmgMulLate.Clear();
        _dmgCap.Clear();
    }

    /// <summary>True when the run-level lists hold RunState.IterateHookListeners(<paramref name="combatState"/>) of <paramref name="runState"/>.</summary>
    private bool RunCached(IRunState? runState, ICombatState? combatState)
    {
        if (runState == null || !Cached(combatState))
        {
            return false;
        }
        if (_runState == null)
        {
            _runState = runState;
            foreach (AbstractModel m in runState.IterateHookListeners(combatState))
            {
                RunListeners++;
                Ov o = OverridesOf(m.GetType());
                if ((o & Ov.DamageAdditive) != 0)
                {
                    _dmgAdd.Add(m);
                }
                if ((o & Ov.DamageMultiplicative) != 0)
                {
                    _dmgMul.Add(m);
                }
                if ((o & Ov.DamageMultiplicativeLate) != 0)
                {
                    _dmgMulLate.Add(m);
                }
                if ((o & Ov.DamageCap) != 0)
                {
                    _dmgCap.Add(m);
                }
            }
        }
        return ReferenceEquals(runState, _runState);
    }

    // ---------------------------------------------------------------------------------------------- Hook.ModifyDamage

    /// <summary>= Hook.ModifyDamage(...) with its modifier list discarded (Hooks/Hook.cs:1343-1426).</summary>
    public decimal ModifyDamage(IRunState runState, ICombatState? combatState, Creature? target, Creature? dealer, decimal damage, ValueProp props,
        CardModel? cardSource, CardPlay? cardPlay, ModifyDamageHookType modifyDamageHookType, CardPreviewMode previewMode)
    {
        if (Exact || !RunCached(runState, combatState))
        {
            return Hook.ModifyDamage(runState, combatState, target, dealer, damage, props, cardSource, cardPlay, modifyDamageHookType, previewMode, out _);
        }
        decimal num = damage;
        if (cardSource != null && cardSource.Enchantment != null)
        {
            if (modifyDamageHookType.HasFlag(ModifyDamageHookType.Additive))
            {
                num += cardSource.Enchantment.EnchantDamageAdditive(num, props);
            }
            if (modifyDamageHookType.HasFlag(ModifyDamageHookType.Multiplicative))
            {
                num *= cardSource.Enchantment.EnchantDamageMultiplicative(num, props);
            }
        }
        bool multi = false;
        if (target == null && previewMode == CardPreviewMode.MultiCreatureTargeting && cardSource != null)
        {
            TargetType targetType = cardSource.TargetType;
            if ((uint)(targetType - 3) <= 1u) // AllEnemies, RandomEnemy
            {
                CardPile? pile = cardSource.Pile;
                multi = pile != null && (pile.Type == PileType.Hand || pile.Type == PileType.Play);
            }
        }
        bool uniform = false;
        if (multi)
        {
            bool same = true;
            decimal? first = null;
            foreach (Creature item in combatState?.HittableEnemies ?? Array.Empty<Creature>())
            {
                decimal v = ModifyDamageInternal(item, dealer, num, props, cardSource, cardPlay, modifyDamageHookType);
                if (!first.HasValue)
                {
                    first = v;
                }
                else if ((int)v != (int)first.Value)
                {
                    same = false;
                    break;
                }
            }
            if (first.HasValue && same)
            {
                uniform = true;
                num = first.Value;
            }
        }
        if (!multi || !uniform)
        {
            num = ModifyDamageInternal(target, dealer, num, props, cardSource, cardPlay, modifyDamageHookType);
        }
        return Math.Max(0m, num);
    }

    /// <summary>= Hook.ModifyDamageInternal over the cached run-level listeners (Hooks/Hook.cs:2238-2294).</summary>
    private decimal ModifyDamageInternal(Creature? target, Creature? dealer, decimal damage, ValueProp props, CardModel? cardSource, CardPlay? cardPlay, ModifyDamageHookType modifyDamageHookType)
    {
        decimal num = damage;
        if (modifyDamageHookType.HasFlag(ModifyDamageHookType.Additive))
        {
            for (int i = 0; i < _dmgAdd.Count; i++)
            {
                num += _dmgAdd[i].ModifyDamageAdditive(target, num, props, dealer, cardSource, cardPlay);
            }
        }
        if (modifyDamageHookType.HasFlag(ModifyDamageHookType.Multiplicative))
        {
            for (int i = 0; i < _dmgMul.Count; i++)
            {
                num *= _dmgMul[i].ModifyDamageMultiplicative(target, num, props, dealer, cardSource, cardPlay);
            }
            for (int i = 0; i < _dmgMulLate.Count; i++)
            {
                num *= _dmgMulLate[i].ModifyDamageMultiplicativeLate(target, num, props, dealer, cardSource, cardPlay);
            }
        }
        if (modifyDamageHookType.HasFlag(ModifyDamageHookType.Cap))
        {
            decimal cap = decimal.MaxValue;
            for (int i = 0; i < _dmgCap.Count; i++)
            {
                decimal c = _dmgCap[i].ModifyDamageCap(target, props, dealer, cardSource, cardPlay);
                if (c < cap)
                {
                    cap = c;
                    if (num > c)
                    {
                        num = c;
                    }
                }
            }
        }
        return num;
    }

    // ---------------------------------------------------------------------------------------------- combat-level hooks

    /// <summary>= Hook.ModifyBlock(...) with its modifier list discarded (Hooks/Hook.cs:1194-1224).</summary>
    public decimal ModifyBlock(ICombatState combatState, Creature target, decimal block, ValueProp props, CardModel? cardSource, CardPlay? cardPlay)
    {
        if (Exact || !Cached(combatState))
        {
            return Hook.ModifyBlock(combatState, target, block, props, cardSource, cardPlay, out _);
        }
        decimal num = block;
        if (cardSource != null && cardSource.Enchantment != null)
        {
            EnchantmentModel enchantment = cardSource.Enchantment;
            num += enchantment.EnchantBlockAdditive(num);
            num *= enchantment.EnchantBlockMultiplicative(num);
        }
        if (!_skipCombatEffects)
        {
            for (int i = 0; i < _blockAdd.Count; i++)
            {
                num += _blockAdd[i].ModifyBlockAdditive(target, num, props, cardSource, cardPlay);
            }
            for (int i = 0; i < _blockMul.Count; i++)
            {
                num *= _blockMul[i].ModifyBlockMultiplicative(target, num, props, cardSource, cardPlay);
            }
        }
        return Math.Max(0m, num);
    }

    /// <summary>= Hook.ModifyPowerAmountGiven(...) with its modifier list discarded (Hooks/Hook.cs:1692-1716).</summary>
    public decimal ModifyPowerAmountGiven(ICombatState combatState, PowerModel power, Creature giver, decimal amount, Creature? target, CardModel? cardSource)
    {
        if (Exact || !Cached(combatState))
        {
            return Hook.ModifyPowerAmountGiven(combatState, power, giver, amount, target, cardSource, out _);
        }
        decimal num = amount;
        if (!_skipCombatEffects)
        {
            for (int i = 0; i < _powerAdd.Count; i++)
            {
                num += _powerAdd[i].ModifyPowerAmountGivenAdditive(power, giver, num, target, cardSource);
            }
            for (int i = 0; i < _powerMul.Count; i++)
            {
                num *= _powerMul[i].ModifyPowerAmountGivenMultiplicative(power, giver, num, target, cardSource);
            }
        }
        return num;
    }

    /// <summary>= Hook.ModifySummonAmount (Hooks/Hook.cs:1834-1842).</summary>
    public decimal ModifySummonAmount(ICombatState combatState, Player summoner, decimal amount, AbstractModel? source)
    {
        if (Exact || !Cached(combatState))
        {
            return Hook.ModifySummonAmount(combatState, summoner, amount, source);
        }
        decimal num = amount;
        if (!_skipCombatEffects)
        {
            for (int i = 0; i < _summon.Count; i++)
            {
                num = _summon[i].ModifySummonAmount(summoner, num, source);
            }
        }
        return num;
    }

    /// <summary>= pcs.MaxEnergy = (int)Hook.ModifyMaxEnergy(player.Creature.CombatState, player, player.MaxEnergy) (Entities.Players/PlayerCombatState.cs:97; Hooks/Hook.cs:1602-1610).</summary>
    public int MaxEnergy(PlayerCombatState pcs)
    {
        Player player = pcs._player;
        ICombatState? combatState = player.Creature.CombatState;
        if (Exact || !Cached(combatState))
        {
            return pcs.MaxEnergy;
        }
        decimal num = player.MaxEnergy;
        if (!_skipCombatEffects)
        {
            for (int i = 0; i < _maxEnergy.Count; i++)
            {
                num = _maxEnergy[i].ModifyMaxEnergy(player, num);
            }
        }
        return (int)num;
    }

    /// <summary>= creature.IsHittable (Entities.Creatures/Creature.cs:242-256; Hook.ShouldAllowHitting, Hooks/Hook.cs:1922-1932; AbstractModel.ShouldAllowHitting returns true).</summary>
    public bool IsHittable(Creature creature)
    {
        bool r = IsHittableCore(creature);
        if (ExactCheck && !Exact)
        {
            HittableCompared++;
            if (r != creature.IsHittable)
            {
                HittableMismatches++;
            }
        }
        return r;
    }

    /// <summary>DIAGNOSTICS (SPIREHOST_EXACT_CHECK=1): every cached IsHittable (driver targets and encoder rows) compared with the game's.</summary>
    public static long HittableCompared, HittableMismatches;

    private bool IsHittableCore(Creature creature)
    {
        if (Exact || !Cached(creature.CombatState))
        {
            return creature.IsHittable;
        }
        if (creature.IsDead)
        {
            return false;
        }
        if (!_skipCombatEffects)
        {
            for (int i = 0; i < _allowHitting.Count; i++)
            {
                if (!_allowHitting[i].ShouldAllowHitting(creature))
                {
                    return false;
                }
            }
        }
        return true;
    }

    /// <summary>= Hook.ShouldPayExcessEnergyCostWithStars(combatState, player) (Hooks/Hook.cs:2094-2104; AbstractModel's returns false).</summary>
    public bool ShouldPayExcessEnergyCostWithStars(ICombatState combatState, Player player)
    {
        if (Exact || !Cached(combatState))
        {
            return Hook.ShouldPayExcessEnergyCostWithStars(combatState, player);
        }
        if (!_skipCombatEffects)
        {
            for (int i = 0; i < _payWithStars.Count; i++)
            {
                if (_payWithStars[i].ShouldPayExcessEnergyCostWithStars(player))
                {
                    return true;
                }
            }
        }
        return false;
    }

    /// <summary>= poison.CalculateTotalDamageNextTurn() (Models.Powers/PoisonPower.cs:36-47).</summary>
    public int PoisonDamageNextTurn(PoisonPower poison)
    {
        if (Exact)
        {
            return poison.CalculateTotalDamageNextTurn();
        }
        decimal num = default(decimal);
        int num2 = Math.Min(poison.Amount, poison.TriggerCount);
        for (int i = 0; i < num2; i++)
        {
            decimal damage = poison.Amount - i;
            damage = ModifyDamage(poison.Owner.CombatState.RunState, poison.Owner.CombatState, poison.Owner, null, damage, ValueProp.Unblockable | ValueProp.Unpowered, null, null, ModifyDamageHookType.All, CardPreviewMode.None);
            num += damage;
        }
        return (int)num;
    }

    // ---------------------------------------------------------------------------------------------- card previews

    /// <summary>= card.UpdateDynamicVarPreview(previewMode, target, set) (Models/CardModel.cs:1301-1330).</summary>
    public void UpdateDynamicVarPreview(CardModel card, CardPreviewMode previewMode, Creature? target, DynamicVarSet set)
    {
        if (Exact)
        {
            card.UpdateDynamicVarPreview(previewMode, target, set);
            return;
        }
        if (card.RunState == null && card.CombatState == null)
        {
            return;
        }
        bool runGlobalHooks = false;
        if (card.CombatState != null)
        {
            PileType? pile = card.Pile?.Type;
            runGlobalHooks = pile == PileType.Hand || pile == PileType.Play || card.UpgradePreviewType == CardUpgradePreviewType.Combat;
        }
        // the game iterates a snapshot (dynamicVarSet.Values.ToList())
        List<DynamicVar> vars = _varScratch;
        vars.Clear();
        vars.AddRange(set.Values);
        for (int i = 0; i < vars.Count; i++)
        {
            UpdateCardPreview(vars[i], card, previewMode, target, runGlobalHooks);
        }
        vars.Clear();
    }

    private void UpdateCardPreview(DynamicVar dv, CardModel card, CardPreviewMode previewMode, Creature? target, bool runGlobalHooks)
    {
        Type t = dv.GetType();
        if (t == typeof(DamageVar))
        {
            DamageVarPreview((DamageVar)dv, card, previewMode, target, runGlobalHooks);
        }
        else if (t == typeof(BlockVar))
        {
            BlockVarPreview((BlockVar)dv, card, runGlobalHooks);
        }
        else if (t == typeof(OstyDamageVar))
        {
            OstyDamageVarPreview((OstyDamageVar)dv, card, previewMode, target, runGlobalHooks);
        }
        else if (t == typeof(CalculatedDamageVar))
        {
            CalculatedDamageVarPreview((CalculatedDamageVar)dv, card, previewMode, target, runGlobalHooks);
        }
        else if (t == typeof(CalculatedBlockVar))
        {
            CalculatedBlockVarPreview((CalculatedBlockVar)dv, card, target, runGlobalHooks);
        }
        else if (t == typeof(SummonVar))
        {
            // SummonVar.UpdateCardPreview (Localization.DynamicVars/SummonVar.cs)
            if (runGlobalHooks)
            {
                dv.PreviewValue = ModifySummonAmount(card.CombatState!, card.Owner, dv.BaseValue, card);
            }
        }
        else if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(PowerVar<>))
        {
            // PowerVar<T>.UpdateCardPreview (Localization.DynamicVars/PowerVar.cs:21-27)
            if (runGlobalHooks)
            {
                dv.PreviewValue = ModifyPowerAmountGiven(card.CombatState!, CanonicalPower(t), card.Owner.Creature, dv.BaseValue, target, card);
            }
        }
        else
        {
            dv.UpdateCardPreview(card, previewMode, target, runGlobalHooks);
        }
    }

    /// <summary>
    /// ModelDb.Power&lt;T&gt;() of a PowerVar&lt;T&gt; type: one delegate per closed type (created once), called once per frame
    /// and type (a dictionary look-up of the canonical model whose key, ModelDb.GetId, is rebuilt with a regex at every
    /// call; the content of ModelDb does not change while the encoder runs).
    /// </summary>
    private PowerModel CanonicalPower(Type powerVarType)
    {
        if (_canonicalPowerFrame.TryGetValue(powerVarType, out PowerModel? power))
        {
            return power;
        }
        if (!s_canonicalPower.TryGetValue(powerVarType, out Func<PowerModel>? f))
        {
            Type t = powerVarType.GetGenericArguments()[0];
            MethodInfo m = typeof(ModelDb).GetMethod(nameof(ModelDb.Power), BindingFlags.Public | BindingFlags.Static, Type.EmptyTypes)!.MakeGenericMethod(t);
            f = m.CreateDelegate<Func<PowerModel>>();
            s_canonicalPower[powerVarType] = f;
        }
        power = f();
        _canonicalPowerFrame[powerVarType] = power;
        return power;
    }

    /// <summary>DamageVar.UpdateCardPreview (Localization.DynamicVars/DamageVar.cs:28-45).</summary>
    private void DamageVarPreview(DamageVar dv, CardModel card, CardPreviewMode previewMode, Creature? target, bool runGlobalHooks)
    {
        decimal num = dv.BaseValue;
        EnchantmentModel? enchantment = card.Enchantment;
        if (enchantment != null)
        {
            num += enchantment.EnchantDamageAdditive(num, dv.Props);
            num *= enchantment.EnchantDamageMultiplicative(num, dv.Props);
            if (!card.IsEnchantmentPreview)
            {
                dv.EnchantedValue = num;
            }
        }
        if (runGlobalHooks)
        {
            num = ModifyDamage(card.Owner.RunState, card.CombatState, target, card.Owner.Creature, dv.BaseValue, dv.Props, card, null, ModifyDamageHookType.All, previewMode);
        }
        dv.PreviewValue = num;
    }

    /// <summary>OstyDamageVar.UpdateCardPreview (Localization.DynamicVars/OstyDamageVar.cs).</summary>
    private void OstyDamageVarPreview(OstyDamageVar dv, CardModel card, CardPreviewMode previewMode, Creature? target, bool runGlobalHooks)
    {
        decimal num = dv.BaseValue;
        EnchantmentModel? enchantment = card.Enchantment;
        if (enchantment != null)
        {
            num += enchantment.EnchantDamageAdditive(num, dv.Props);
            num *= enchantment.EnchantDamageMultiplicative(num, dv.Props);
            if (!card.IsEnchantmentPreview)
            {
                dv.EnchantedValue = num;
            }
        }
        if (runGlobalHooks)
        {
            ICombatState? combatState = card.CombatState ?? card.Owner.Creature.CombatState;
            num = ModifyDamage(card.Owner.RunState, combatState, target, card.Owner.Osty, dv.BaseValue, dv.Props, card, null, ModifyDamageHookType.All, previewMode);
        }
        dv.PreviewValue = num;
    }

    /// <summary>BlockVar.UpdateCardPreview (Localization.DynamicVars/BlockVar.cs).</summary>
    private void BlockVarPreview(BlockVar dv, CardModel card, bool runGlobalHooks)
    {
        decimal num = dv.BaseValue;
        EnchantmentModel? enchantment = card.Enchantment;
        if (enchantment != null)
        {
            num += enchantment.EnchantBlockAdditive(num);
            num *= enchantment.EnchantBlockMultiplicative(num);
            if (!card.IsEnchantmentPreview)
            {
                dv.EnchantedValue = num;
            }
        }
        if (runGlobalHooks)
        {
            num = ModifyBlock(card.CombatState!, card.Owner.Creature, dv.BaseValue, dv.Props, card, null);
        }
        dv.PreviewValue = num;
    }

    /// <summary>CalculatedDamageVar.UpdateCardPreview (Localization.DynamicVars/CalculatedDamageVar.cs).</summary>
    private void CalculatedDamageVarPreview(CalculatedDamageVar dv, CardModel card, CardPreviewMode previewMode, Creature? target, bool runGlobalHooks)
    {
        EnchantmentModel? enchantment = card.Enchantment;
        if (enchantment != null)
        {
            decimal baseValue = dv.GetBaseVar().BaseValue;
            baseValue += enchantment.EnchantDamageAdditive(baseValue, dv.Props);
            baseValue *= enchantment.EnchantDamageMultiplicative(baseValue, dv.Props);
            baseValue = Math.Max(baseValue, 0m);
            if (card.IsEnchantmentPreview)
            {
                dv.PreviewValue = baseValue;
            }
            else
            {
                dv.EnchantedValue = baseValue;
            }
        }
        decimal num = dv.Calculate(target);
        if (runGlobalHooks)
        {
            ICombatState? combatState = card.CombatState ?? card.Owner.Creature.CombatState;
            dv.PreviewValue = ModifyDamage(card.Owner.RunState, combatState, target, dv.IsFromOsty ? card.Owner.Osty : card.Owner.Creature, num, dv.Props, card, null, ModifyDamageHookType.All, previewMode);
        }
        else if (!card.IsEnchantmentPreview)
        {
            if (enchantment != null)
            {
                num += enchantment.EnchantDamageAdditive(num, dv.Props);
                num *= enchantment.EnchantDamageMultiplicative(num, dv.Props);
            }
            dv.PreviewValue = num;
        }
        dv.PreviewValue = Math.Max(dv.PreviewValue, 0m);
    }

    /// <summary>CalculatedBlockVar.UpdateCardPreview (Localization.DynamicVars/CalculatedBlockVar.cs).</summary>
    private void CalculatedBlockVarPreview(CalculatedBlockVar dv, CardModel card, Creature? target, bool runGlobalHooks)
    {
        EnchantmentModel? enchantment = card.Enchantment;
        if (enchantment != null)
        {
            decimal baseValue = dv.GetBaseVar().BaseValue;
            baseValue += enchantment.EnchantBlockAdditive(baseValue);
            baseValue *= enchantment.EnchantBlockMultiplicative(baseValue);
            if (card.IsEnchantmentPreview)
            {
                dv.PreviewValue = baseValue;
            }
            else
            {
                dv.EnchantedValue = baseValue;
            }
        }
        decimal num = dv.Calculate(target);
        if (runGlobalHooks)
        {
            dv.PreviewValue = ModifyBlock(card.CombatState!, card.Owner.Creature, dv.Calculate(target), dv.Props, card, null);
        }
        else if (!card.IsEnchantmentPreview)
        {
            if (enchantment != null)
            {
                num += enchantment.EnchantBlockAdditive(num);
                num *= enchantment.EnchantBlockMultiplicative(num);
            }
            dv.PreviewValue = num;
        }
    }

    // ---------------------------------------------------------------------------------------------- intents

    /// <summary>= attack.GetSingleDamage(targets, owner) (MonsterMoves.Intents/AttackIntent.cs:84-93); DamageCalc is called twice as there.</summary>
    public int SingleDamage(AttackIntent attack, IEnumerable<Creature> targets, Creature owner)
    {
        if (Exact)
        {
            return attack.GetSingleDamage(targets, owner);
        }
        decimal num = attack.DamageCalc!();
        Player? me = LocalContext.GetMe(owner.CombatState);
        if (me != null)
        {
            num = ModifyDamage(me.RunState, me.Creature.CombatState, me.Creature, owner, attack.DamageCalc!(), ValueProp.Move, null, null, ModifyDamageHookType.All, CardPreviewMode.None);
        }
        return Math.Max(0, (int)num);
    }
}
