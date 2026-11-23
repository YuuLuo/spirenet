using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Sim;

namespace SpireHost.Obs.Tests;

/// <summary>
/// TEST ONLY - reads hidden state on purpose. "Encoding has no side effects" (schema 14.4, DECISIONS-H1 F.8): a
/// signature over everything the three writing encoder steps could touch, taken before and after encoding:
///   RunSignature.Compute (seed, every RNG stream, map, deck order, combat signature), the game's own combat checksum
///   (ChecksumTracker.GenerateChecksum(NetFullCombatState.FromRun): creatures, powers, piles, RNG, choice ids),
///   every DynamicVar of every card in every pile + its enchantment (Base / Enchanted / Preview / WasJustUpgraded),
///   UpgradePreviewType, the card scopes' card counts (upgrade-preview clones must be removed), LocalContext.NetId,
///   each creature's powers (amount) and NextMove identity, monster intents count.
/// </summary>
public static class DeepSignature
{
    public static ulong Compute(RunState run)
    {
        ulong h = 1469598103934665603UL;
        void Mix(long v)
        {
            h ^= unchecked((ulong)v);
            h *= 1099511628211UL;
            h ^= h >> 31;
        }
        void MixD(decimal d)
        {
            int[] bits = decimal.GetBits(d);
            Mix(bits[0]);
            Mix(bits[1]);
            Mix(bits[2]);
            Mix(bits[3]);
        }
        void Vars(DynamicVarSet set)
        {
            foreach (DynamicVar dv in set._vars.Values)
            {
                Mix(dv.Name.GetHashCode(StringComparison.Ordinal));
                MixD(dv.BaseValue);
                MixD(dv.EnchantedValue);
                MixD(dv.PreviewValue);
                Mix(dv.WasJustUpgraded ? 1 : 0);
            }
        }
        void Card(CardModel c)
        {
            Mix(c.Id.Entry.GetHashCode(StringComparison.Ordinal));
            Mix((int)c.UpgradePreviewType);
            Mix(c.CurrentUpgradeLevel);
            if (c._dynamicVars != null)
            {
                Vars(c._dynamicVars);
            }
            if (c.Enchantment?._dynamicVars != null)
            {
                Vars(c.Enchantment._dynamicVars);
            }
        }
        Mix((long)RunSignature.Compute(run));
        Mix((long)(LocalContext.NetId ?? ulong.MaxValue));
        Mix(run._allCards.Count);
        foreach (Player p in run.Players)
        {
            foreach (CardModel c in p.Deck.Cards)
            {
                Card(c);
            }
            PlayerCombatState? pcs = p.PlayerCombatState;
            if (pcs != null)
            {
                foreach (var pile in pcs.AllPiles)
                {
                    Mix(pile.Cards.Count);
                    foreach (CardModel c in pile.Cards)
                    {
                        Card(c);
                    }
                }
            }
        }
        if (CombatManager.Instance.IsInProgress && run.CurrentRoom is CombatRoom room)
        {
            CombatState cs = room.CombatState;
            Mix(cs._allCards.Count);
            foreach (Creature c in cs._allies)
            {
                Creature(c);
            }
            foreach (Creature c in cs._enemies)
            {
                Creature(c);
            }
            Mix(RunManager.Instance.ChecksumTracker.GenerateChecksum(NetFullCombatState.FromRun(run, null)));
        }
        return h;

        void Creature(Creature c)
        {
            Mix(c.CurrentHp);
            Mix(c.Block);
            foreach (PowerModel pw in c.Powers)
            {
                Mix(pw.Id.Entry.GetHashCode(StringComparison.Ordinal));
                Mix(pw.Amount);
            }
            if (c.Monster != null)
            {
                Mix(System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(c.Monster.NextMove));
                Mix(c.Monster.NextMove.Intents?.Count ?? -1);
            }
        }
    }
}
