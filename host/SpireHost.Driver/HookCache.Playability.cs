using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Driver;

/// <summary>
/// Third part: CardModel.CanPlay for the COMBAT handler's enumeration (every hand card of every seat that can act, at every
/// COMBAT decision). CanPlay = the Unplayable keyword (Hook.ModifyKeywordsInCombat over CombatState.IterateHookListeners),
/// the energy and star costs with their hooks (PlayerCombatState.HasEnoughResourcesFor), and Hook.ShouldPlay over the
/// combat hook listeners: three to five listener enumerations per card. Same method as the first part: transcriptions over
/// the listeners that override the hook (AbstractModel.ShouldPlay returns true: Models/AbstractModel.cs:1022-1025, so a
/// listener that does not override it never vetoes and never becomes the preventer).
/// </summary>
internal sealed partial class HookCache
{
    private static readonly Type[] s_shouldPlaySig = { typeof(CardModel), typeof(AutoPlayType) };
    private readonly List<AbstractModel> _shouldPlay = new List<AbstractModel>(8);

    /// <summary>DIAGNOSTICS (SPIREHOST_EXACT_CHECK=1): every cached CanPlay is compared with the game's; mismatches counted here.</summary>
    public static readonly bool ExactCheck = Environment.GetEnvironmentVariable("SPIREHOST_EXACT_CHECK") == "1";
    public static long CanPlayCompared, CanPlayMismatches;

    /// <summary>= card.CanPlay(out reason, out preventer) (Models/CardModel.cs:1505-1535).</summary>
    public bool CanPlay(CardModel card, out UnplayableReason reason, out AbstractModel? preventer)
    {
        bool r = CanPlayCore(card, out reason, out preventer);
        if (ExactCheck && !Exact)
        {
            bool g = card.CanPlay(out UnplayableReason gr, out AbstractModel? gp);
            CanPlayCompared++;
            if (g != r || gr != reason || !ReferenceEquals(gp, preventer))
            {
                CanPlayMismatches++;
            }
        }
        return r;
    }

    private bool CanPlayCore(CardModel card, out UnplayableReason reason, out AbstractModel? preventer)
    {
        ICombatState? combatState = card.CombatState ?? card._owner?.Creature.CombatState;
        if (Exact || combatState == null || card.Owner.PlayerCombatState == null || !Cached(combatState) || !Cached(card.CombatState))
        {
            return card.CanPlay(out reason, out preventer);
        }
        reason = UnplayableReason.None;
        if (Keywords(card).Contains(CardKeyword.Unplayable))
        {
            reason |= UnplayableReason.HasUnplayableKeyword;
        }
        if (!HasEnoughResourcesFor(card.Owner.PlayerCombatState, card, out UnplayableReason reason2))
        {
            reason |= reason2;
        }
        if (card.TargetType == TargetType.AnyAlly && combatState.PlayerCreatures.Count((Creature c) => c.IsAlive) <= 1)
        {
            reason |= UnplayableReason.NoLivingAllies;
        }
        if (!ShouldPlay(card, out preventer))
        {
            reason |= UnplayableReason.BlockedByHook;
        }
        if (!card.IsPlayable)
        {
            reason |= UnplayableReason.BlockedByCardLogic;
        }
        return reason == UnplayableReason.None;
    }

    /// <summary>= pcs.HasEnoughResourcesFor(card, out reason) (Entities.Players/PlayerCombatState.cs:199-218).</summary>
    private bool HasEnoughResourcesFor(PlayerCombatState pcs, CardModel card, out UnplayableReason reason)
    {
        int num = Math.Max(0, EnergyCost(card));
        int num2 = Math.Max(0, StarCost(card));
        if (num > pcs.Energy && card.CombatState != null && ShouldPayExcessEnergyCostWithStars(card.CombatState, pcs._player))
        {
            num2 += (num - pcs.Energy) * 2;
            num = pcs.Energy;
        }
        reason = UnplayableReason.None;
        if (num > pcs.Energy)
        {
            reason |= UnplayableReason.EnergyCostTooHigh;
        }
        if (num2 > pcs.Stars)
        {
            reason |= UnplayableReason.StarCostTooHigh;
        }
        return reason == UnplayableReason.None;
    }

    /// <summary>= Hook.ShouldPlay(combatState, card, out preventer, AutoPlayType.None) (Hooks/Hook.cs:2106-2118) on the cached state.</summary>
    private bool ShouldPlay(CardModel card, out AbstractModel? preventer)
    {
        if (!_skipCombatEffects)
        {
            for (int i = 0; i < _shouldPlay.Count; i++)
            {
                if (!_shouldPlay[i].ShouldPlay(card, AutoPlayType.None))
                {
                    preventer = _shouldPlay[i];
                    return false;
                }
            }
        }
        preventer = null;
        return true;
    }
}
