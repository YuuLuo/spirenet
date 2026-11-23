using System;
using System.Collections.Generic;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs.Encoding;

/// <summary>Enum value -> vocab id tables (vocab "enum:&lt;Name&gt;" categories), built once per vocabulary.</summary>
public static class Ev
{
    public const string GameMode = "enum:GameMode";
    public const string RoomType = "enum:RoomType";
    public const string CombatSide = "enum:CombatSide";
    public const string PlayerTurnPhase = "enum:PlayerTurnPhase";
    public const string CardCostColor = "enum:CardCostColor";
    public const string EnchantmentStatus = "enum:EnchantmentStatus";
    public const string PileType = "enum:PileType";
    public const string HpDisplay = "enum:HpDisplay";
    public const string PowerStackType = "enum:PowerStackType";
    public const string PowerType = "enum:PowerType";
    public const string IntentType = "enum:IntentType";
    public const string RelicStatus = "enum:RelicStatus";
    public const string ArrowStatus = "enum:ArrowStatus";
    public const string MapPointType = "enum:MapPointType";
    public const string RewardType = "enum:RewardType";
    public const string CrystalSphereItemType = "enum:CrystalSphereItemType";
    public const string CrystalSphereToolType = "enum:CrystalSphereToolType";
}

public sealed partial class ObsEncoder
{
    private readonly Dictionary<string, int[]> _enumIds = new Dictionary<string, int[]>(StringComparer.Ordinal);

    /// <summary>vocab id of an enum value (by its name; enums here are dense 0..n-1). Unknown value -> UNK.</summary>
    private int Enum(string cat, int value)
    {
        if (!_enumIds.TryGetValue(cat, out int[]? ids))
        {
            ids = BuildEnumIds(cat);
            _enumIds[cat] = ids;
        }
        return (uint)value < (uint)ids.Length ? ids[value] : VocabSet.UnkId;
    }

    private int[] BuildEnumIds(string cat)
    {
        string name = cat.Substring("enum:".Length);
        Type? type = null;
        foreach ((string n, Type t) in VocabExport.EnumCategories)
        {
            if (n == name)
            {
                type = t;
            }
        }
        if (type == null)
        {
            throw new InvalidOperationException("no enum type for " + cat);
        }
        int max = 0;
        foreach (object v in System.Enum.GetValues(type))
        {
            max = Math.Max(max, Convert.ToInt32(v));
        }
        int[] ids = new int[max + 1];
        Array.Fill(ids, VocabSet.UnkId);
        foreach (object v in System.Enum.GetValues(type))
        {
            int i = Convert.ToInt32(v);
            if (i >= 0)
            {
                ids[i] = _v.Id(cat, System.Enum.GetName(type, v));
            }
        }
        return ids;
    }
}
