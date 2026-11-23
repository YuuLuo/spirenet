using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using SpireHost.Driver;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Encoding;

public sealed partial class ObsEncoder
{
    // =====================================================================================================
    //  ACTIONS (schema 5.13): one row per legal action, IN THE DRIVER'S CANONICAL ORDER (DRIVER-API.md 6.4) -
    //  row i == action index i. src/tgt are rows of the entity tables written above.
    // =====================================================================================================

    private void EncodeActions(Decision d)
    {
        List<LegalAction> actions = d.Actions;
        int prevRow = -1;
        for (int i = 0; i < actions.Count; i++)
        {
            LegalAction a = actions[i];
            int r = _w.AddRow(T.Actions);
            Span<int> row = _w.Row(T.Actions, r);
            row[ActionC.Verb] = (int)a.Verb;
            row[ActionC.Seat] = a.Seat;
            (int table, int srcRow) = SourceRow(a);
            row[ActionC.SrcTable] = table;
            row[ActionC.SrcRow] = srcRow;
            row[ActionC.TgtRow] = TargetRow(a);
            row[ActionC.Aux] = a.Verb == Verb.CrystalClick ? Enum(Ev.CrystalSphereToolType, a.Aux) : a.Aux;
            row[ActionC.Flags] = 0;
            if (prevRow >= 0 && CompareActionRows(prevRow, r) >= 0)
            {
                Stats.ActionOrderViolations++;
            }
            prevRow = r;
        }
    }

    private int CompareActionRows(int a, int b)
    {
        Span<int> x = _w.Row(T.Actions, a), y = _w.Row(T.Actions, b);
        for (int c = 0; c < 6; c++)
        {
            int k = x[c].CompareTo(y[c]);
            if (k != 0)
            {
                return k;
            }
        }
        return 0;
    }

    private (int table, int row) SourceRow(in LegalAction a)
    {
        GameRef s = a.Source;
        switch (s.Kind)
        {
            case RefKind.None:
                return (-1, -1);
            case RefKind.Card:
                if (a.Verb == Verb.SelectCard)
                {
                    return Mapped(T.Cards, s.Index < _choiceRow.Length && s.Index >= 0 && _d?.Context is CardSelectContext ? _choiceRow[s.Index] : -1);
                }
                if (a.Verb == Verb.PickOfferCard)
                {
                    return Mapped(T.Cards, s.Index >= 0 && s.Index < _offerCardRow.Length && _d?.Context is CardRewardContext ? _offerCardRow[s.Index] : -1);
                }
                return Mapped(T.Cards, s.Object is CardModel c && _cardRow.TryGetValue(c, out int cr) ? cr : -1);
            case RefKind.PotionSlot:
            {
                int seat = SeatOf(s.Object as MegaCrit.Sts2.Core.Entities.Players.Player);
                return Mapped(T.Potions, seat >= 0 && seat < 4 && s.Index >= 0 && s.Index < 16 ? _potionRow[seat * 16 + s.Index] : -1);
            }
            case RefKind.MapPoint:
                return Mapped(T.MapNodes, s.Object is MapPoint mp && _mapRow.TryGetValue(mp, out int mr) ? mr : -1);
            case RefKind.CrystalCell:
                return Mapped(T.Cells, s.Index >= 0 && s.Index < _cellRow.Length && _w.Count(T.Cells) > 0 ? _cellRow[s.Index] : -1);
            case RefKind.Creature:
                return Mapped(T.Creatures, CreatureRowOf(s.Object as Creature));
            default:
                // option-like kinds: EventOption, RestOption, Reward, ShopEntry, CardRewardAlt, TreasureRelic, Bundle, ConfirmChoice
                return Mapped(T.Options, (int)s.Kind == _optRefKind && s.Index >= 0 && s.Index < _optRow.Length ? _optRow[s.Index] : -1);
        }
    }

    private (int, int) Mapped(int table, int row)
    {
        if (row < 0)
        {
            Stats.UnmappedActionRefs++;
            return (-1, -1);
        }
        return (table, row);
    }

    private int TargetRow(in LegalAction a)
    {
        if (a.Target.Kind != RefKind.Creature)
        {
            return -1;
        }
        int r = CreatureRowOf(a.Target.Object as Creature);
        if (r < 0)
        {
            Stats.UnmappedActionRefs++;
        }
        return r;
    }
}
