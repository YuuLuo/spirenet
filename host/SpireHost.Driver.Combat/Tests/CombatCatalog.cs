using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Driver.Combat.Tests;

/// <summary>Id lists for the combat test suites, read from ModelDb (nothing is hard-coded that the game can enumerate).</summary>
public static class CombatCatalog
{
    public static readonly string[] Characters = { "IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT", "FOOL" };

    public static List<string> Encounters()
    {
        List<string> r = ModelDb.AllEncounters.Select(e => e.Id.Entry).Distinct().ToList();
        r.Sort(StringComparer.Ordinal);
        return r;
    }

    public static List<CardModel> Cards()
    {
        List<CardModel> r = ModelDb.AllCards.ToList();
        r.Sort((a, b) => string.CompareOrdinal(a.Id.Entry, b.Id.Entry));
        return r;
    }

    public static List<PotionModel> Potions()
    {
        List<PotionModel> r = ModelDb.AllPotions.ToList();
        r.Sort((a, b) => string.CompareOrdinal(a.Id.Entry, b.Id.Entry));
        return r;
    }

    public static List<RelicModel> Relics()
    {
        List<RelicModel> r = ModelDb.AllRelics.ToList();
        r.Sort((a, b) => string.CompareOrdinal(a.Id.Entry, b.Id.Entry));
        return r;
    }

    public static bool HasCard(string entry) => ModelDb.AllCards.Any(c => c.Id.Entry == entry);

    public static bool HasRelic(string entry) => ModelDb.AllRelics.Any(c => c.Id.Entry == entry);

    public static bool HasPotion(string entry) => ModelDb.AllPotions.Any(c => c.Id.Entry == entry);

    public static object Describe()
    {
        return new
        {
            encounters = Encounters(),
            cards = Cards().Select(c => new
            {
                id = c.Id.Entry,
                type = c.Type.ToString(),
                rarity = c.Rarity.ToString(),
                target = c.TargetType.ToString(),
                cost = c.EnergyCost.Canonical,
                x = c.EnergyCost.CostsX,
                star = c.CanonicalStarCost,
                star_x = c.HasStarCostX,
                unplayable = c.CanonicalKeywords.Contains(CardKeyword.Unplayable),
            }).ToList(),
            potions = Potions().Select(p => new { id = p.Id.Entry, usage = p.Usage.ToString(), target = p.TargetType.ToString(), rarity = p.Rarity.ToString() }).ToList(),
            relics = Relics().Select(r => r.Id.Entry).ToList(),
            powers = ModelDb.AllPowers.Select(p => p.Id.Entry).OrderBy(s => s, StringComparer.Ordinal).ToList(),
        };
    }
}
