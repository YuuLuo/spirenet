using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Models;
using SpireHost.Driver;

namespace SpireHost.Obs.Tests.Canary;

/// <summary>
/// NEVER CALLED. Code that reads hidden information on purpose, so that the static encoder audit can prove that its rules
/// fire (verifier finding R-7: three host-only members had no rule and one rule could no longer match anything). The
/// audit scans this namespace and must report every read below; if it reports fewer, the audit - not the encoder - is broken.
/// </summary>
public static class AuditCanary
{
    /// <summary>1: the item under a crystal cell; 2: the live minigame behind the crystal view; 3: the frame object; 4: the raw candidates of a selection frame; 5: the draw pile.</summary>
    public static int ReadsHiddenState(CrystalSphereCell cell, CrystalContext crystal, Decision decision, Player player)
    {
        int n = 0;
        if (cell.Item != null)
        {
            n++;
        }
        if (crystal.Minigame != null)
        {
            n++;
        }
        if (decision.Source is ICardSelectHostView view)
        {
            IReadOnlyList<CardModel> raw = view.RawOptions;
            n += raw.Count;
        }
        CardPile? draw = player.PlayerCombatState?.DrawPile;
        return n + (draw == null ? 0 : 1);
    }
}
