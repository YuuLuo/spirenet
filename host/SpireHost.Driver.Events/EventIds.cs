using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Driver;

namespace SpireHost.Driver.Events;

/// <summary>Kept apart from the command dispatch so that no game type is touched before the session exists.</summary>
internal static class EventIds
{
    /// <summary>ModelDb.AllEvents (sorted) - the set the stage-0 sweep used.</summary>
    public static List<string> All()
    {
        List<string> ids = new List<string>();
        foreach (EventModel e in ModelDb.AllEvents)
        {
            ids.Add(e.Id.Entry);
        }
        ids.Sort(StringComparer.Ordinal);
        return ids;
    }

    /// <summary>
    /// Every concrete EventModel type of the game (73 in v0.113: ModelDb.AllEvents, the Ancients, TheArchitect and two
    /// Deprecated* placeholders), as canonical models, sorted by id. <paramref name="includeDeprecated"/>: the two
    /// placeholders only stand in for events of old saves (SaveUtil.EventOrDeprecated) and are never entered by a run.
    /// </summary>
    public static List<EventModel> Every(bool includeDeprecated = false)
    {
        MethodInfo generic = typeof(ModelDb).GetMethod(nameof(ModelDb.Event), BindingFlags.Public | BindingFlags.Static)
            ?? throw new MissingMethodException("ModelDb.Event<T>");
        List<EventModel> list = new List<EventModel>();
        foreach (Type t in typeof(EventModel).Assembly.GetTypes())
        {
            if (t.IsAbstract || !typeof(EventModel).IsAssignableFrom(t) || t.Namespace != typeof(Neow).Namespace)
            {
                continue;
            }
            if (!includeDeprecated && (t == typeof(DeprecatedEvent) || t == typeof(DeprecatedAncientEvent)))
            {
                continue;
            }
            list.Add((EventModel)generic.MakeGenericMethod(t).Invoke(null, null)!);
        }
        list.Sort((a, b) => string.CompareOrdinal(a.Id.Entry, b.Id.Entry));
        return list;
    }

    public static int CountTypes() => Every(includeDeprecated: true).Count;

    public static Task Jump(RunContext ctx, string eventEntry)
    {
        ScenarioTools.SetGold(ctx, 500);
        return ScenarioTools.JumpToEvent(ctx, eventEntry);
    }

    /// <summary>Deterministic test potions: the first <paramref name="n"/> common, non-automatic, non-Foul potions by id.</summary>
    public static List<PotionModel> TestPotions(int n)
    {
        return ModelDb.AllPotions.Where(p => p.Rarity == PotionRarity.Common && p.Usage != PotionUsage.Automatic && p is not FoulPotion)
            .OrderBy(p => p.Id.Entry, StringComparer.Ordinal).Take(n).ToList();
    }

    /// <summary>Deterministic tradable test relics (RanwidTheElder / RelicTrader need tradable relics): common, by id.</summary>
    public static List<RelicModel> TestRelics(int n)
    {
        return ModelDb.AllRelics.Where(r => r.Rarity == RelicRarity.Common && r.IsTradable)
            .OrderBy(r => r.Id.Entry, StringComparer.Ordinal).Take(n).ToList();
    }
}
