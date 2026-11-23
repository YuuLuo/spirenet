using System;
using System.Collections.Generic;
using System.Diagnostics;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using SpireHost.Driver;

namespace SpireHost.Obs.Tests;

/// <summary>
/// Measures the cost of the individual game getters the encoder relies on, on real combat states (diagnostic for the
/// encoder-cost section of the report): per call, averaged over every draw-pile card of sampled combat decisions.
/// </summary>
public static class CostProbe
{
    public static Dictionary<string, double> Measure(RunContext ctx, int repeat)
    {
        Dictionary<string, double> us = new Dictionary<string, double>();
        Player p = ctx.LocalPlayer;
        PlayerCombatState? pcs = p.PlayerCombatState;
        if (pcs == null)
        {
            return us;
        }
        List<CardModel> cards = new List<CardModel>(pcs.DrawPile.Cards);
        if (cards.Count == 0)
        {
            return us;
        }
        void Time(string name, Action<CardModel> f)
        {
            long t0 = Stopwatch.GetTimestamp();
            for (int r = 0; r < repeat; r++)
            {
                foreach (CardModel c in cards)
                {
                    f(c);
                }
            }
            us[name] = Math.Round((Stopwatch.GetTimestamp() - t0) * 1e6 / Stopwatch.Frequency / (repeat * cards.Count), 3);
        }
        int sink = 0;
        Time("Pile", c => sink += c.Pile == null ? 0 : 1);
        Time("CombatState", c => sink += c.CombatState == null ? 0 : 1);
        Time("Keywords", c => sink += c.Keywords.Count);
        Time("EnergyCost.GetWithModifiers(All)", c => sink += c.EnergyCost.GetWithModifiers(CostModifiers.All));
        Time("GetStarCostWithModifiers", c => sink += c.GetStarCostWithModifiers());
        Time("UpdateDynamicVarPreview", c =>
        {
            c.DynamicVars.ClearPreview();
            c.UpdateDynamicVarPreview(CardPreviewMode.Normal, null, c.DynamicVars);
        });
        Time("GetEnchantedReplayCount", c => sink += c.GetEnchantedReplayCount());
        us["draw_pile_cards"] = cards.Count;
        us["sink"] = sink & 1;
        return us;
    }
}
