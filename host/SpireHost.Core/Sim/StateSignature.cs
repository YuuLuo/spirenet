using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace SpireHost.Sim;

/// <summary>64-bit order-sensitive hash (FNV-1a over 64-bit words with a final avalanche).</summary>
public struct SigHasher
{
    private ulong _h;

    public static SigHasher Create() => new SigHasher { _h = 14695981039346656037UL };

    public void Add(ulong v)
    {
        _h ^= v;
        _h *= 1099511628211UL;
        _h ^= _h >> 29;
    }

    public void Add(long v) => Add(unchecked((ulong)v));

    public void Add(int v) => Add(unchecked((ulong)(long)v));

    public void Add(bool v) => Add(v ? 1UL : 2UL);

    public void Add(string? s)
    {
        if (s == null)
        {
            Add(0xFFFFUL);
            return;
        }
        ulong h = 1469598103934665603UL;
        for (int i = 0; i < s.Length; i++)
        {
            h ^= s[i];
            h *= 1099511628211UL;
        }
        Add(h);
        Add(s.Length);
    }

    public void Add(ModelId? id) => Add(id?.Entry);

    public ulong Finish()
    {
        ulong x = _h;
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdUL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53UL;
        x ^= x >> 33;
        return x;
    }
}

/// <summary>
/// Full-information state signature used ONLY by the host's own determinism / replay checks. It reads
/// hidden state (pile order, RNG counters and RNG internal state, monster next move) and must never be
/// part of an observation.
/// </summary>
public static class StateSignature
{
    public static ulong Compute(RunState run, CombatState state, CombatManager cm)
    {
        SigHasher h = SigHasher.Create();
        h.Add(state.RoundNumber);
        h.Add((int)state.CurrentSide);
        h.Add(cm.IsInProgress);

        IReadOnlyList<Creature> creatures = state.Creatures;
        h.Add(creatures.Count);
        for (int i = 0; i < creatures.Count; i++)
        {
            Creature c = creatures[i];
            h.Add(c.ModelId);
            h.Add((long)(c.CombatId ?? uint.MaxValue));
            h.Add((int)c.Side);
            h.Add(c.CurrentHp);
            h.Add(c.MaxHp);
            h.Add(c.Block);
            IReadOnlyList<PowerModel> powers = c.Powers;
            h.Add(powers.Count);
            for (int p = 0; p < powers.Count; p++)
            {
                h.Add(powers[p].Id);
                h.Add(powers[p].Amount);
            }
            if (c.Monster != null)
            {
                h.Add(c.Monster.NextMove?.Id);
            }
        }

        IReadOnlyList<Player> players = run.Players;
        h.Add(players.Count);
        for (int i = 0; i < players.Count; i++)
        {
            Player pl = players[i];
            h.Add(pl.Gold);
            h.Add(pl.Relics.Count);
            PlayerCombatState? pcs = pl.PlayerCombatState;
            if (pcs == null)
            {
                h.Add(-1);
                continue;
            }
            h.Add(pcs.TurnNumber);
            h.Add((int)pcs.Phase);
            h.Add(pcs.Energy);
            h.Add(pcs.Stars);
            AddPile(ref h, pcs.Hand);
            AddPile(ref h, pcs.DrawPile);
            AddPile(ref h, pcs.DiscardPile);
            AddPile(ref h, pcs.ExhaustPile);
            AddPile(ref h, pcs.PlayPile);
            IReadOnlyList<OrbModel> orbs = pcs.OrbQueue.Orbs;
            h.Add(pcs.OrbQueue.Capacity);
            h.Add(orbs.Count);
            for (int o = 0; o < orbs.Count; o++)
            {
                h.Add(orbs[o].Id);
            }
            AddRng(ref h, pl.PlayerRng.Rewards);
            AddRng(ref h, pl.PlayerRng.Shops);
            AddRng(ref h, pl.PlayerRng.Transformations);
            AddRng(ref h, pl.PlayerRng.NonCombatGeneration);
        }

        h.Add(run.Rng.StringSeed);
        for (int i = 0; i < RunRngTypes.Length; i++)
        {
            AddRng(ref h, run.Rng.GetRng(RunRngTypes[i]));
        }
        return h.Finish();
    }

    private static void AddPile(ref SigHasher h, CardPile pile)
    {
        IReadOnlyList<CardModel> cards = pile.Cards;
        h.Add(0x50494C45L + (int)pile.Type);
        h.Add(cards.Count);
        for (int i = 0; i < cards.Count; i++)
        {
            CardModel card = cards[i];
            h.Add(card.Id);
            h.Add(card.CurrentUpgradeLevel);
            h.Add(card.EnergyCost.Canonical);
            h.Add(card.EnergyCost.HasLocalModifiers);
            h.Add(card.Enchantment?.Id);
            h.Add(card.Affliction?.Id);
        }
    }

    private static readonly MegaCrit.Sts2.Core.Entities.Rngs.RunRngType[] RunRngTypes = System.Enum.GetValues<MegaCrit.Sts2.Core.Entities.Rngs.RunRngType>();

    /// <summary>Counter and full generator state of one RNG stream.</summary>
    private static void AddRng(ref SigHasher h, Rng rng)
    {
        SerializableRng s = rng.ToSerializable();
        h.Add(s.counter);
        h.Add(s.state0);
        h.Add(s.state1);
        h.Add(s.state2);
        h.Add(s.state3);
    }

    /// <summary>Draw counters of the run RNG streams, for reporting (name -> counter).</summary>
    public static Dictionary<string, int> RngCounters(RunState run)
    {
        Dictionary<string, int> d = new Dictionary<string, int>();
        for (int i = 0; i < RunRngTypes.Length; i++)
        {
            d[RunRngTypes[i].ToString()] = run.Rng.GetRng(RunRngTypes[i]).ToSerializable().counter;
        }
        return d;
    }
}
