using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Ascension;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Entities.Rngs;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;
using SpireHost.Obs.Encoding;

namespace SpireHost.Search;

/// <summary>How a draw pile is ordered before the re-draw.</summary>
public enum PileOrder
{
    /// <summary>The search resampler: a TOTAL canonical order on card states (<see cref="Resampler.SortCanonical"/>).</summary>
    Canonical = 0,

    /// <summary>
    /// TEST ONLY (leak-test positive control): the game's own ListExtensions.StableShuffle = List.Sort by
    /// AbstractModel.CompareTo (model Id only; List.Sort is not stable) + shuffle. Same-Id cards that differ (Strike /
    /// Strike+) keep a relative order that comes from the true pile: a leak.
    /// </summary>
    GameStableShuffle = 1,
}

/// <summary>
/// Re-randomisation of a fork child (plan step H3; fork study section 3): every hidden quantity of the running fight is
/// replaced by a fresh sample drawn from the search seed alone, through the game's own objects and routines.
///   * every seat's DRAW PILE ORDER: the pile is put into a canonical order that is a function of the multiset of card
///     states only (<see cref="SortCanonical"/>), then shuffled with the game's UnstableShuffle and a fresh Rng. The
///     content (multiset) is public and kept. Discard and exhaust piles keep their order (the game shows them unsorted).
///     NOT handled: draw-pile positions that are public (a card put on top / bottom by a displayed effect, PerfectFit):
///     the uniform re-draw forgets them (fair, pessimistic);
///   * all 12 RUN streams (RunRngSet.MockRng), every seat's 4 PLAYER streams, every monster's own Rng and the encounter's Rng.
/// Kept from the truth: each monster's displayed next move and its move-state machine (public history), and every hidden
/// but deterministic field (audit H5).
/// The search seed comes from the client (master search seed + public counters); nothing of the game is mixed in.
/// </summary>
public static class Resampler
{
    public sealed class Info
    {
        public ulong Seed;
        public int CardsRedrawn;

        /// <summary>Adjacent draw-pile cards equal under the canonical order (indistinguishable states; their relative order does not matter).</summary>
        public int CanonicalTies;
    }

    private static readonly FieldInfo s_runRngSeedField =
        typeof(RunRngSet).GetField("<Seed>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!;
    private static readonly FieldInfo s_playerRngSeedField =
        typeof(PlayerRngSet).GetField("<Seed>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!;
    private static readonly PropertyInfo s_runActsProp =
        typeof(RunState).GetProperty(nameof(RunState.Acts), BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public)!;
    private static readonly FieldInfo s_cardRarityRngField =
        FindInstanceField(typeof(MegaCrit.Sts2.Core.Odds.CardRarityOdds), "_rng");
    private static readonly FieldInfo s_potionRewardRngField =
        FindInstanceField(typeof(MegaCrit.Sts2.Core.Odds.PotionRewardOdds), "_rng");
    private static readonly FieldInfo s_unknownMapPointRngField =
        FindInstanceField(typeof(MegaCrit.Sts2.Core.Odds.UnknownMapPointOdds), "_rng");

    private static FieldInfo FindInstanceField(Type type, string name)
    {
        for (Type? cur = type; cur != null; cur = cur.BaseType)
        {
            FieldInfo? f = cur.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.DeclaredOnly);
            if (f != null)
            {
                return f;
            }
        }
        throw new MissingFieldException(type.FullName, name);
    }

    /// <summary>Re-seed an existing Rng instance IN-PLACE so cached references (AbstractOdds._rng, etc.) are also resampled.</summary>
    internal static void ReseedInPlace(Rng target, ulong seed)
    {
        target.LoadFromSerializable(new Rng(seed).ToSerializable());
    }

    internal static void SetRunRngSeed(RunRngSet rngSet, ulong seed) => s_runRngSeedField.SetValue(rngSet, seed);

    internal static void SetPlayerRngSeed(PlayerRngSet rngSet, ulong seed) => s_playerRngSeedField.SetValue(rngSet, seed);

    public static Info Rerandomise(
        RunContext ctx,
        ObsEncoder keyEncoder,
        ulong seed,
        PileOrder order,
        bool resampleMacroState = false)
    {
        Info info = new Info { Seed = seed };
        Rng r = new Rng(seed);
        IReadOnlyList<Player> players = ctx.Run.Players;
        for (int s = 0; s < players.Count; s++)
        {
            Player p = players[s];
            PlayerCombatState? pcs = p.PlayerCombatState;
            if (pcs != null)
            {
                List<CardModel> cards = pcs.DrawPile._cards;
                if (order == PileOrder.Canonical)
                {
                    info.CanonicalTies += SortCanonical(ctx, keyEncoder, cards, s);
                    cards.UnstableShuffle(new Rng(r.NextUnsignedLong()));
                }
                else
                {
                    cards.StableShuffle(new Rng(r.NextUnsignedLong()));
                }
                info.CardsRedrawn += cards.Count;
            }
            foreach (PlayerRngType t in Enum.GetValues<PlayerRngType>())
            {
                ulong pSeed = r.NextUnsignedLong();
                Rng target = p.PlayerRng._rngs[t];
                ReseedInPlace(target, pSeed);
                if (t == PlayerRngType.Rewards)
                {
                    // Restore exact STS2 aliasing invariant (CardRarity._rng & PotionReward._rng === PlayerRng._rngs[Rewards])
                    s_cardRarityRngField.SetValue(p.PlayerOdds.CardRarity, target);
                    s_potionRewardRngField.SetValue(p.PlayerOdds.PotionReward, target);
                }
            }
        }
        foreach (RunRngType t in Enum.GetValues<RunRngType>())
        {
            ulong rSeed = r.NextUnsignedLong();
            Rng target = ctx.Run.Rng._rngs[t];
            ReseedInPlace(target, rSeed);
            if (t == RunRngType.UnknownMapPoint)
            {
                // Restore exact STS2 aliasing invariant (UnknownMapPoint._rng === RunRng._rngs[UnknownMapPoint])
                s_unknownMapPointRngField.SetValue(ctx.Run.Odds.UnknownMapPoint, target);
            }
        }
        if (ctx.Run.CurrentRoom is CombatRoom combatRoom && ctx.CombatManager.IsInProgress)
        {
            CombatState cs = combatRoom.CombatState;
            foreach (Creature e in cs._enemies)
            {
                if (e.Monster != null)
                {
                    e.Monster.Rng = new Rng(r.NextUnsignedLong());
                }
            }
            if (cs.Encounter != null)
            {
                cs.Encounter._rng = new Rng(r.NextUnsignedLong());
            }
        }
        if (resampleMacroState)
        {
            RerandomiseMacroState(ctx, r);
        }
        return info;
    }

    /// <summary>
    /// Erases all non-combat hidden state (RunRngSet.Seed, PlayerRngSet.Seed, synchronizer RNGs,
    /// RelicGrabBags, unvisited RoomSet suffixes, future Acts, and active EventModel.Rng) while strictly
    /// preserving public state (current deck, relics, potions, HP, gold, current Act map, and current Act's
    /// public Boss / SecondBoss / Ancient).
    /// </summary>
    private static void RerandomiseMacroState(RunContext ctx, Rng r)
    {
        RunState run = ctx.Run;

        // 1. Replace get-only Seed backing fields read by EncounterModel, EventModel, StandardActMap, Rng(Player, Id)
        SetRunRngSeed(run.Rng, r.NextUnsignedLong());
        foreach (Player p in run.Players)
        {
            SetPlayerRngSeed(p.PlayerRng, r.NextUnsignedLong());
            ReshuffleRelicGrabBag(p.RelicGrabBag, r);
        }
        ReshuffleRelicGrabBag(run.SharedRelicGrabBag, r);

        // 2. Re-seed multiplayer synchronizer RNGs in-place (unconditional draws keep r aligned)
        ulong evSyncSeed = r.NextUnsignedLong();
        ulong mapSyncSeed = r.NextUnsignedLong();
        if (RunManager.Instance?.EventSynchronizer?._multiplayerOptionSelectionRng != null)
        {
            ReseedInPlace(RunManager.Instance.EventSynchronizer._multiplayerOptionSelectionRng, evSyncSeed);
        }
        if (RunManager.Instance?.MapSelectionSynchronizer?._multiplayerMapPointSelection != null)
        {
            ReseedInPlace(RunManager.Instance.MapSelectionSynchronizer._multiplayerMapPointSelection, mapSyncSeed);
        }

        // 3. Active EventRoom mutable event RNG (if inside an EventRoom)
        if (run.CurrentRoom is EventRoom er && er.LocalMutableEvent?.Rng != null)
        {
            ReseedInPlace(er.LocalMutableEvent.Rng, r.NextUnsignedLong());
        }

        // 4. Current Act: keep PUBLIC Boss, SecondBoss, Ancient, and visited prefixes; regenerate unvisited suffixes
        bool isMultiplayer = run.Players.Count > 1;
        bool isBlitz = run.GameMode == GameMode.Blitz;
        ResampleCurrentActRoomSuffixes(run, run.Acts[run.CurrentActIndex], r, isMultiplayer, isBlitz);

        // 5. Future Acts (i > CurrentActIndex): re-roll hidden Act choices, SharedAncients subsets, RoomSets, and A10 SecondBoss
        if (run.CurrentActIndex + 1 < run.Acts.Count)
        {
            List<ActModel> acts = run.Acts.ToList();
            List<AncientEventModel> sharedPool = run.UnlockState.SharedAncients
                .Where(a => !run.VisitedEventIds.Contains(a.Id) && a.Id != run.Act.Ancient.Id)
                .OrderBy(a => a.Id.Entry, StringComparer.Ordinal)
                .ToList();
            sharedPool.UnstableShuffle(new Rng(r.NextUnsignedLong()));

            for (int i = run.CurrentActIndex + 1; i < acts.Count; i++)
            {
                List<ActModel> choices = ModelDb.ActsByIndex[i]
                    .Where(a => a.IsUnlocked(run.UnlockState))
                    .OrderBy(a => a.Id.Entry, StringComparer.Ordinal)
                    .ToList();
                ActModel chosen = r.NextItem(choices)!.ToMutable();
                int takeCount = r.NextInt(sharedPool.Count + 1);
                List<AncientEventModel> subset = sharedPool.Take(takeCount).ToList();
                sharedPool.RemoveRange(0, takeCount);
                chosen.SetSharedAncientSubset(subset);
                chosen.GenerateRooms(new Rng(r.NextUnsignedLong()), run.UnlockState, isMultiplayer, isBlitz);
                if (i == acts.Count - 1 && run.AscensionLevel >= (int)AscensionLevel.DoubleBoss)
                {
                    List<EncounterModel> secondChoices = chosen.AllBossEncounters
                        .Where(e => e.Id != chosen.BossEncounter.Id)
                        .OrderBy(e => e.Id.Entry, StringComparer.Ordinal)
                        .ToList();
                    chosen.SetSecondBossEncounter(r.NextItem(secondChoices));
                }
                acts[i] = chosen;
            }
            s_runActsProp.SetValue(run, acts);
        }
    }

    private static void ReshuffleRelicGrabBag(RelicGrabBag bag, Rng r)
    {
        if (!bag.IsPopulated)
        {
            return;
        }
        foreach (RelicRarity rarity in bag._deques.Keys.OrderBy(k => (int)k))
        {
            List<RelicModel> deque = bag._deques[rarity];
            deque.Sort((a, b) => StringComparer.Ordinal.Compare(a.Id.Entry, b.Id.Entry));
            deque.UnstableShuffle(new Rng(r.NextUnsignedLong()));
        }
        if (bag._mpFallbackDequeue.Count > 1)
        {
            bag._mpFallbackDequeue.Sort((a, b) => StringComparer.Ordinal.Compare(a.Id.Entry, b.Id.Entry));
            bag._mpFallbackDequeue.UnstableShuffle(new Rng(r.NextUnsignedLong()));
        }
        if (bag._originalRelics != null && bag._originalRelics.Count > 1)
        {
            bag._originalRelics.Sort((a, b) => StringComparer.Ordinal.Compare(a.Id.Entry, b.Id.Entry));
        }
    }

    private static void ResampleCurrentActRoomSuffixes(RunState run, ActModel act, Rng r, bool isMultiplayer, bool isBlitz)
    {
        RoomSet cur = act.GetRoomsDebug();
        ActModel freshAct = act.CanonicalInstance.ToMutable();
        freshAct.GenerateRooms(new Rng(r.NextUnsignedLong()), run.UnlockState, isMultiplayer, isBlitz);
        RoomSet fresh = freshAct.GetRoomsDebug();

        // Events: rebuild from unvisited events in fresh order and reset eventsVisited=0 so old skipped indices cannot leak
        List<EventModel> remEvents = fresh.events.Where(e => !run.VisitedEventIds.Contains(e.Id)).ToList();
        if (remEvents.Count == 0)
        {
            remEvents = fresh.events.ToList();
        }
        cur.events.Clear();
        cur.events.AddRange(remEvents);
        cur.eventsVisited = 0;

        // Normal & Elite encounters: preserve visited prefix [0..visited-1] (public history & last-encounter no-repeat tag),
        // and replace every unvisited suffix slot from fresh!
        SpliceEncounterSuffix(cur.normalEncounters, cur.normalEncountersVisited, fresh.normalEncounters);
        SpliceEncounterSuffix(cur.eliteEncounters, cur.eliteEncountersVisited, fresh.eliteEncounters);
    }

    private static void SpliceEncounterSuffix(List<EncounterModel> cur, int visited, List<EncounterModel> fresh)
    {
        int keep = Math.Min(visited, cur.Count);
        EncounterModel? lastSeen = keep > 0 ? cur[keep - 1] : null;
        cur.RemoveRange(keep, cur.Count - keep);
        for (int i = keep; i < fresh.Count; i++)
        {
            EncounterModel cand = fresh[i];
            if (cur.Count == keep && lastSeen != null && (cand == lastSeen || cand.SharesTagsWith(lastSeen)))
            {
                int swap = fresh.FindIndex(i + 1, e => e != lastSeen && !e.SharesTagsWith(lastSeen));
                if (swap >= 0)
                {
                    (fresh[i], fresh[swap]) = (fresh[swap], fresh[i]);
                    cand = fresh[i];
                }
            }
            cur.Add(cand);
        }
    }

    private sealed class Keyed
    {
        public CardModel Card = null!;
        public int[] Enc = null!;
        public byte[] State = null!;
        public byte[] Vars = null!;
    }

    private static readonly List<int[]> s_keys = new List<int[]>(128);

    /// <summary>
    /// Puts <paramref name="cards"/> into the canonical order, a TOTAL order on card states: (1) the encoder's multiset
    /// content key of the card's CARDS row (schema 5.2.3: id, upgrade, costs, keywords, enchantment, affliction, printed
    /// VARS ... = what the team sees of the card), then for ties (2) the game's own serialised card state
    /// (NetFullCombatState.CardState: SerializableCard with upgrade level, enchantment and saved properties, affliction,
    /// modified energy cost, keywords), then (3) every DynamicVar of the card and of its enchantment (name, base,
    /// enchanted and preview value, just-upgraded). Cards equal on all three are indistinguishable states; the result as a
    /// sequence of states depends only on the multiset of states, never on the input order. Returns the number of such ties.
    /// </summary>
    public static int SortCanonical(RunContext ctx, ObsEncoder keyEncoder, List<CardModel> cards, int seat)
    {
        int n = cards.Count;
        if (n < 2)
        {
            return 0;
        }
        keyEncoder.DrawPileKeys(ctx, cards, seat, s_keys);
        if (s_keys.Count != n)
        {
            throw new InvalidOperationException("DrawPileKeys returned " + s_keys.Count + " keys for " + n + " cards");
        }
        Keyed[] k = new Keyed[n];
        for (int i = 0; i < n; i++)
        {
            k[i] = new Keyed { Card = cards[i], Enc = s_keys[i], State = StateBytes(cards[i]), Vars = VarBytes(cards[i]) };
        }
        // insertion sort (piles are small): stable, so the result is deterministic even for indistinguishable cards
        for (int i = 1; i < n; i++)
        {
            Keyed x = k[i];
            int j = i - 1;
            while (j >= 0 && Compare(k[j], x) > 0)
            {
                k[j + 1] = k[j];
                j--;
            }
            k[j + 1] = x;
        }
        int ties = 0;
        for (int i = 0; i < n; i++)
        {
            cards[i] = k[i].Card;
            if (i > 0 && Compare(k[i - 1], k[i]) == 0)
            {
                ties++;
            }
        }
        return ties;
    }

    private static int Compare(Keyed a, Keyed b)
    {
        int c = CompareInts(a.Enc, b.Enc);
        if (c != 0)
        {
            return c;
        }
        c = CompareBytes(a.State, b.State);
        return c != 0 ? c : CompareBytes(a.Vars, b.Vars);
    }

    /// <summary>Lexicographic; a proper prefix is smaller.</summary>
    private static int CompareInts(int[] a, int[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            int c = a[i].CompareTo(b[i]);
            if (c != 0)
            {
                return c;
            }
        }
        return a.Length.CompareTo(b.Length);
    }

    private static int CompareBytes(byte[] a, byte[] b)
    {
        int c = a.AsSpan().SequenceCompareTo(b);
        return c < 0 ? -1 : c > 0 ? 1 : 0;
    }

    /// <summary>The game's serialisation of the card as the full-combat checksum sees it (NetFullCombatState.CardState).</summary>
    public static byte[] StateBytes(CardModel card)
    {
        PacketWriter w = new PacketWriter();
        w.Write(NetFullCombatState.CardState.From(card));
        w.ZeroByteRemainder();
        return w.Buffer.AsSpan(0, w.BytePosition).ToArray();
    }

    public static byte[] VarBytes(CardModel card)
    {
        List<byte> b = new List<byte>(64);
        AddVars(b, 1, card._dynamicVars);
        AddVars(b, 2, card.Enchantment?._dynamicVars);
        return b.ToArray();
    }

    private static readonly List<string> s_names = new List<string>(16);

    private static void AddVars(List<byte> b, byte tag, DynamicVarSet? set)
    {
        if (set == null)
        {
            return;
        }
        s_names.Clear();
        s_names.AddRange(set._vars.Keys);
        s_names.Sort(StringComparer.Ordinal);
        foreach (string name in s_names)
        {
            DynamicVar dv = set._vars[name];
            b.Add(tag);
            b.AddRange(Encoding.UTF8.GetBytes(name));
            b.Add(0);
            AddDecimal(b, dv.BaseValue);
            AddDecimal(b, dv.EnchantedValue);
            AddDecimal(b, dv.PreviewValue);
            b.Add(dv.WasJustUpgraded ? (byte)1 : (byte)0);
        }
    }

    private static void AddDecimal(List<byte> b, decimal d)
    {
        // the raw bits (value and scale) are state of the card: any total order on them is canonical
        foreach (int part in decimal.GetBits(d))
        {
            b.Add((byte)part);
            b.Add((byte)(part >> 8));
            b.Add((byte)(part >> 16));
            b.Add((byte)(part >> 24));
        }
    }
}
