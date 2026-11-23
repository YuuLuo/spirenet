using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace SpireHost.Sim;

/// <summary>
/// Full-information signature of a whole run, used ONLY by the host's determinism / replay checks.
/// It reads hidden state (seed, every RNG stream's counter and generator state, deck order, map contents)
/// and must never be part of an observation.
///
/// Covered: seed, all run RNG streams, act / floor / visited coordinates, the room stack (type + model id),
/// the current act's map (every point's coordinate and type) and boss ids, per player: gold, HP, max HP,
/// deck in order (id, upgrade, enchantment, affliction), relics in order, potion slots, the 4 player RNG
/// streams; plus the complete combat signature (<see cref="StateSignature"/>) while a combat is in progress.
/// Not covered: wall-clock fields, process-level ids (action / hook / choice ids, CombatId sequence), private
/// fields of monsters / powers / relics (a divergence there shows up as soon as it touches RNG, HP or piles).
/// </summary>
public static class RunSignature
{
    private static readonly MegaCrit.Sts2.Core.Entities.Rngs.RunRngType[] RunRngTypes = System.Enum.GetValues<MegaCrit.Sts2.Core.Entities.Rngs.RunRngType>();

    public static ulong Compute(RunState run)
    {
        SigHasher h = SigHasher.Create();
        h.Add(run.Rng.StringSeed);
        h.Add(run.AscensionLevel);
        h.Add(run.CurrentActIndex);
        h.Add(run.ActFloor);
        h.Add(run.Act.Id);
        h.Add(run.Act.BossEncounter?.Id);
        h.Add(run.Act.SecondBossEncounter?.Id);

        IReadOnlyList<MapCoord> visited = run.VisitedMapCoords;
        h.Add(visited.Count);
        for (int i = 0; i < visited.Count; i++)
        {
            h.Add(visited[i].col);
            h.Add(visited[i].row);
        }

        IReadOnlyList<IReadOnlyList<MegaCrit.Sts2.Core.Runs.History.MapPointHistoryEntry>> history = run.MapPointHistory;
        int floors = 0;
        for (int i = 0; i < history.Count; i++)
        {
            floors += history[i].Count;
        }
        h.Add(floors);

        ActMap map = run.Map;
        int points = 0;
        foreach (MapPoint p in map.GetAllMapPoints())
        {
            h.Add(p.coord.col);
            h.Add(p.coord.row);
            h.Add((int)p.PointType);
            h.Add(p.Children.Count);
            points++;
        }
        h.Add(points);

        h.Add(run.CurrentRoomCount);
        AbstractRoom? room = run.CurrentRoom;
        if (room != null)
        {
            h.Add((int)room.RoomType);
            h.Add(room.ModelId);
            h.Add(room.GetType().Name);
        }
        AbstractRoom? baseRoom = run.BaseRoom;
        if (baseRoom != null && !ReferenceEquals(baseRoom, room))
        {
            h.Add((int)baseRoom.RoomType);
            h.Add(baseRoom.ModelId);
        }

        IReadOnlyList<Player> players = run.Players;
        h.Add(players.Count);
        for (int i = 0; i < players.Count; i++)
        {
            Player pl = players[i];
            h.Add(pl.Gold);
            h.Add(pl.Creature.CurrentHp);
            h.Add(pl.Creature.MaxHp);
            h.Add(pl.MaxEnergy);
            IReadOnlyList<CardModel> deck = pl.Deck.Cards;
            h.Add(deck.Count);
            for (int c = 0; c < deck.Count; c++)
            {
                CardModel card = deck[c];
                h.Add(card.Id);
                h.Add(card.CurrentUpgradeLevel);
                h.Add(card.Enchantment?.Id);
                h.Add(card.Affliction?.Id);
            }
            IReadOnlyList<RelicModel> relics = pl.Relics;
            h.Add(relics.Count);
            for (int r = 0; r < relics.Count; r++)
            {
                h.Add(relics[r].Id);
            }
            IReadOnlyList<PotionModel?> potions = pl.PotionSlots;
            h.Add(potions.Count);
            for (int s = 0; s < potions.Count; s++)
            {
                h.Add(potions[s]?.Id);
            }
            AddRng(ref h, pl.PlayerRng.Rewards);
            AddRng(ref h, pl.PlayerRng.Shops);
            AddRng(ref h, pl.PlayerRng.Transformations);
            AddRng(ref h, pl.PlayerRng.NonCombatGeneration);
        }

        for (int i = 0; i < RunRngTypes.Length; i++)
        {
            AddRng(ref h, run.Rng.GetRng(RunRngTypes[i]));
        }

        CombatManager cm = CombatManager.Instance;
        if (cm.IsInProgress && room is CombatRoom combatRoom)
        {
            h.Add(StateSignature.Compute(run, combatRoom.CombatState, cm));
        }
        else
        {
            h.Add(0x4E4F434F4D424154L);
        }
        return h.Finish();
    }

    private static void AddRng(ref SigHasher h, Rng rng)
    {
        SerializableRng s = rng.ToSerializable();
        h.Add(s.counter);
        h.Add(s.state0);
        h.Add(s.state1);
        h.Add(s.state2);
        h.Add(s.state3);
    }
}
