using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.FoolArrows;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Encoding;

public sealed partial class ObsEncoder
{
    // =====================================================================================================
    //  RELICS / POTIONS owned (schema 5.7, 5.8), ORBS / ARROWS (5.9)
    // =====================================================================================================

    private void EncodeOwnedRelicsAndPotions()
    {
        IReadOnlyList<Player> players = _run.Players;
        for (int s = 0; s < _nSeats; s++)
        {
            IReadOnlyList<RelicModel> relics = players[s].Relics;
            for (int i = 0; i < relics.Count; i++)
            {
                AddRelicRow(relics[i], s, ItemZone.Owned, i);
            }
        }
        for (int s = 0; s < _nSeats; s++)
        {
            Player p = players[s];
            IReadOnlyList<PotionModel?> slots = p.PotionSlots;
            for (int i = 0; i < slots.Count; i++)
            {
                PotionModel? pm = slots[i];
                if (pm == null)
                {
                    continue; // empty slots are implied: potion_slots - rows (schema 5.8)
                }
                int r = AddPotionRow(pm, s, ItemZone.Owned, i);
                if (s < 4 && i < 16)
                {
                    _potionRow[s * 16 + i] = r;
                }
                int flags = _w.Get(T.Potions, r, PotionC.Flags);
                if (HasPotionAction(Verb.UsePotion, s, i))
                {
                    flags |= PotionFlag.Usable;
                }
                if (p.CanUseOrRemovePotions)
                {
                    flags |= PotionFlag.Discardable;
                }
                _w.Set(T.Potions, r, PotionC.Flags, flags);
            }
        }
    }

    private bool HasPotionAction(Verb verb, int seat, int slot)
    {
        if (_d == null)
        {
            return false;
        }
        List<LegalAction> actions = _d.Actions;
        for (int i = 0; i < actions.Count; i++)
        {
            LegalAction a = actions[i];
            if (a.Verb == verb && a.Seat == seat && a.Source.Kind == RefKind.PotionSlot && a.Source.Index == slot)
            {
                return true;
            }
        }
        return false;
    }

    private int AddRelicRow(RelicModel relic, int seat, int zone, int pos)
    {
        int r = _w.AddRow(T.Relics);
        Span<int> row = _w.Row(T.Relics, r);
        row[RelicC.Seat] = seat;
        row[RelicC.Zone] = zone;
        row[RelicC.Pos] = pos;
        row[RelicC.RelicId] = _v.Model("relic", relic);
        // NRelicInventoryHolder shows DisplayAmount only when ShowCounter && RunManager.IsInProgress (Nodes.Relics/NRelicInventoryHolder.cs:164-175)
        row[RelicC.Counter] = zone == ItemZone.Owned && relic.ShowCounter && RunManager.Instance.IsInProgress ? relic.DisplayAmount : -1;
        row[RelicC.Status] = Enum(Ev.RelicStatus, (int)relic.Status);
        int f = 0;
        if (relic.IsUsedUp)
        {
            f |= RelicFlag.UsedUp;
        }
        if (relic.IsWax)
        {
            f |= RelicFlag.Wax;
        }
        if (relic.IsMelted)
        {
            f |= RelicFlag.Melted;
        }
        row[RelicC.Flags] = f;
        AddModelVars(T.Relics, r, relic, relic.DynamicVars, 0);
        return r;
    }

    private int AddPotionRow(PotionModel potion, int seat, int zone, int slot)
    {
        int r = _w.AddRow(T.Potions);
        Span<int> row = _w.Row(T.Potions, r);
        row[PotionC.Seat] = seat;
        row[PotionC.Zone] = zone;
        row[PotionC.Slot] = slot;
        row[PotionC.PotionId] = _v.Model("potion", potion);
        row[PotionC.Flags] = _ctx.IsPotionQueued(potion) ? PotionFlag.Queued : 0;
        AddModelVars(T.Potions, r, potion, potion.DynamicVars, 0);
        return r;
    }

    private void EncodeOrbsAndArrows()
    {
        IReadOnlyList<Player> players = _run.Players;
        for (int s = 0; s < _nSeats; s++)
        {
            PlayerCombatState? pcs = players[s].PlayerCombatState;
            if (pcs == null)
            {
                continue;
            }
            IReadOnlyList<OrbModel> orbs = pcs.OrbQueue.Orbs;
            for (int i = 0; i < orbs.Count; i++)
            {
                int r = _w.AddRow(T.Orbs);
                Span<int> row = _w.Row(T.Orbs, r);
                row[OrbC.Seat] = s;
                row[OrbC.Pos] = i;
                row[OrbC.OrbId] = _v.Model("orb", orbs[i]);
                row[OrbC.Passive] = (int)orbs[i].PassiveVal; // NOrb labels (Nodes.Orbs/NOrb.cs:185-202)
                row[OrbC.Evoke] = (int)orbs[i].EvokeVal;
            }
        }
        for (int s = 0; s < _nSeats; s++)
        {
            PlayerCombatState? pcs = players[s].PlayerCombatState;
            if (pcs == null)
            {
                continue;
            }
            int slot = 0;
            foreach (ArrowSlot a in pcs._arrows)
            {
                int r = _w.AddRow(T.Arrows);
                Span<int> row = _w.Row(T.Arrows, r);
                row[ArrowC.Seat] = s;
                row[ArrowC.Slot] = slot++;
                row[ArrowC.Status] = Enum(Ev.ArrowStatus, (int)a.Status);
            }
        }
    }
}
