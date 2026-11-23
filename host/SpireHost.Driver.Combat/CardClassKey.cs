using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Driver.Combat;

/// <summary>
/// "Identical card" for the purpose of merging pile candidates into classes (DECISIONS-H1 B.4,
/// DRIVER-API.md 6.4). The key is a pure function of what the real UI shows on the card face - never of the
/// card's position in a pile, its creation order or any id assigned in hidden order - so both the classes and
/// the order of the classes are independent of the hidden pile order.
///
/// It contains every field CardModel.CompareTo looks at (Models/CardModel.cs:2018-2071: id, upgrade level,
/// enchantment id, affliction id, resolved energy cost, base replay count, DynamicVar base values) plus what
/// that comparison ignores but the card face shows (obs-schema-v1 5.2.4): rarity, enchantment amount / status,
/// affliction amount, energy cost with modifiers, star cost, the displayed keyword set (Retain / Sly "this
/// turn" included) and the enchanted replay count. Two cards with equal keys therefore compare equal for the
/// game's own stable sort too, i.e. "first instance of the class in the engine's list" is "first in pile order".
///
/// Only called when a selection prompt over a pile is built (rare), never on the combat hot path.
/// </summary>
public static class CardClassKey
{
    [ThreadStatic]
    private static StringBuilder? _sb;

    [ThreadStatic]
    private static List<string>? _names;

    public static string Of(CardModel card)
    {
        StringBuilder sb = _sb ??= new StringBuilder(160);
        sb.Clear();
        CultureInfo inv = CultureInfo.InvariantCulture;
        // Rarity first, then id: the classes come out in (roughly) the order the real pile screen uses.
        sb.Append(((int)card.Rarity).ToString("D2", inv)).Append('|');
        sb.Append(card.Id.Category).Append('.').Append(card.Id.Entry).Append('|');
        sb.Append(card.CurrentUpgradeLevel.ToString(inv)).Append('|');

        EnchantmentModel? enchantment = card.Enchantment;
        if (enchantment != null)
        {
            sb.Append(enchantment.Id.Entry).Append(':').Append(enchantment.Amount.ToString(inv)).Append(':').Append(((int)enchantment.Status).ToString(inv));
        }
        sb.Append('|');
        AfflictionModel? affliction = card.Affliction;
        if (affliction != null)
        {
            sb.Append(affliction.Id.Entry).Append(':').Append(affliction.Amount.ToString(inv));
        }
        sb.Append('|');

        CardEnergyCost cost = card.EnergyCost;
        if (cost.CostsX)
        {
            sb.Append('X');
        }
        else
        {
            sb.Append(cost.GetWithModifiers(CostModifiers.All).ToString(inv));
        }
        sb.Append('/').Append(cost.GetResolved().ToString(inv)).Append('|');
        if (card.HasStarCostX)
        {
            sb.Append('X');
        }
        else
        {
            sb.Append(card.GetStarCostWithModifiers().ToString(inv));
        }
        sb.Append('|');
        sb.Append(card.BaseReplayCount.ToString(inv)).Append('/').Append(card.GetEnchantedReplayCount().ToString(inv)).Append('|');

        int keywordBits = 0;
        foreach (CardKeyword k in card.Keywords)
        {
            keywordBits |= 1 << (int)k;
        }
        if (card.ShouldRetainThisTurn)
        {
            keywordBits |= 1 << (int)CardKeyword.Retain;
        }
        if (card.IsSlyThisTurn)
        {
            keywordBits |= 1 << (int)CardKeyword.Sly;
        }
        sb.Append(keywordBits.ToString(inv)).Append('|');

        DynamicVarSet vars = card.DynamicVars;
        List<string> names = _names ??= new List<string>(8);
        names.Clear();
        foreach (string name in vars.Keys)
        {
            names.Add(name);
        }
        names.Sort(StringComparer.Ordinal);
        for (int i = 0; i < names.Count; i++)
        {
            DynamicVar v = vars[names[i]];
            sb.Append(names[i]).Append('=').Append(v.BaseValue.ToString(inv)).Append(';');
        }
        return sb.ToString();
    }
}
