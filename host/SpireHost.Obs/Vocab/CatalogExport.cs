using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.CardPools;
using SpireHost.Driver;

namespace SpireHost.Obs.Vocab;

/// <summary>
/// catalog.json (obs-schema-v1 12.4, structure fixed by schema-errata-v1 V-3):
///   {"schema":1, "game_version", "vocab_hash", "vocab_version", "categories": {cat: {"&lt;vocab id&gt;": entry}}}
/// card entries are lists indexed by upgrade level 0..MaxUpgradeLevel; field names are the ones py/spire/vocab.py
/// DEFAULT_FEATURES reads; categorical values are vocab entry NAMES (strings) of the named category.
/// Built inside a live run (one at ascension 0 and one at ascension 10, for the ascension-dependent monster HP):
/// upgrade levels are computed on a real owned copy (Run.CreateCard, UpgradeInternal, FinalizeUpgradeInternal).
/// </summary>
public sealed class CatalogExport
{
    private readonly VocabSet _vocab;
    private readonly Templates _templates = new Templates();
    private readonly Dictionary<string, (int min, int max)> _hpA0 = new Dictionary<string, (int, int)>(StringComparer.Ordinal);
    private readonly Dictionary<string, (int min, int max)> _hpA8 = new Dictionary<string, (int, int)>(StringComparer.Ordinal);
    private readonly Dictionary<string, List<object?>> _cards = new Dictionary<string, List<object?>>(StringComparer.Ordinal);
    public readonly List<string> Problems = new List<string>();

    /// <summary>Catalog upgrade levels 0..3 (py tokenize STATIC_LEVELS = 4); higher levels repeat the last entry on the Python side.</summary>
    public const int CatalogMaxLevel = 3;

    public CatalogExport(VocabSet vocab)
    {
        _vocab = vocab;
    }

    // ---------------------------------------------------------------------------------- inside-run collection

    /// <summary>Monster HP under the current run's ascension (AscensionHelper reads RunManager).</summary>
    public void CollectMonsterHp(RunContext ctx, bool a8)
    {
        Dictionary<string, (int, int)> dst = a8 ? _hpA8 : _hpA0;
        foreach (AbstractModel m in ModelDb.All.Where(x => x.Id.Category == "MONSTER" && !VocabExport.IsMock(x)))
        {
            MonsterModel mm = (MonsterModel)m;
            dst[m.Id.Entry] = (mm.MinInitialHp, mm.MaxInitialHp);
        }
    }

    /// <summary>Per-level card records, computed on owned mutable copies (removed from the run again).</summary>
    public void CollectCards(RunContext ctx)
    {
        Player owner = ctx.LocalPlayer;
        foreach (AbstractModel m in ModelDb.All.Where(x => x.Id.Category == "CARD" && !VocabExport.IsMock(x)).OrderBy(x => x.Id.Entry, StringComparer.Ordinal))
        {
            CardModel canonical = (CardModel)m;
            List<object?> levels = new List<object?>();
            levels.Add(CardRecord(canonical, canonical, 0));
            int max = Math.Min(canonical.MaxUpgradeLevel, CatalogMaxLevel); // DINGY_DAGGER: 999999999 levels
            if (max > 0)
            {
                CardModel copy = ctx.Run.CreateCard(canonical, owner);
                try
                {
                    for (int lv = 1; lv <= max && copy.IsUpgradable; lv++)
                    {
                        copy.UpgradeInternal();
                        copy.FinalizeUpgradeInternal();
                        levels.Add(CardRecord(canonical, copy, lv));
                    }
                }
                catch (Exception e) when (e is InvalidOperationException || e is NullReferenceException || e is KeyNotFoundException)
                {
                    // export tool only: a card whose upgrade needs a combat context keeps the levels computed so far
                    Problems.Add($"card {m.Id.Entry}: upgrade level {levels.Count} not computable outside combat ({e.GetType().Name}: {e.Message})");
                }
                finally
                {
                    ctx.Run.RemoveCard(copy);
                }
            }
            _cards[m.Id.Entry] = levels;
        }
    }

    private JObj CardRecord(CardModel canonical, CardModel c, int level)
    {
        JObj o = new JObj
        {
            ["type"] = c.Type.ToString(),
            ["rarity"] = c.Rarity.ToString(),
            ["pool"] = SafePool(canonical),
            ["character"] = PoolCharacter(SafePoolModel(canonical)),
            ["target_type"] = c.TargetType.ToString(),
            ["energy_cost"] = (long)c.EnergyCost.GetWithModifiers(CostModifiers.None),
            ["costs_x"] = c.EnergyCost.CostsX,
            ["star_cost"] = (long)c.CurrentStarCost,
            ["star_x"] = c.HasStarCostX,
            ["keywords"] = Names(c.GetKeywordsWithSources(KeywordSources.Local).OrderBy(k => (int)k).Select(k => k.ToString())),
            ["tags"] = Names(c.Tags.OrderBy(k => (int)k).Select(k => k.ToString())),
            ["multiplayer_constraint"] = c.MultiplayerConstraint.ToString(),
            ["max_upgrade_level"] = (long)c.MaxUpgradeLevel,
            ["orb_evoke_type"] = c.OrbEvokeType.ToString(),
            ["gains_block"] = c.GainsBlock,
            ["can_be_generated_in_combat"] = c.CanBeGeneratedInCombat,
            ["is_basic_strike_or_defend"] = c.IsBasicStrikeOrDefend,
            ["has_turn_end_in_hand_effect"] = c.HasTurnEndInHandEffect,
            ["level"] = (long)level,
        };
        string[] printed = _templates.PrintedVars(canonical);
        o["printed_vars"] = Names(printed);
        o["vars"] = VarsRecord(c.DynamicVars, printed);
        if (level == 0)
        {
            o["text"] = Templates.Raw("cards", canonical.Id.Entry + ".description");
        }
        return o;
    }

    private static List<object?> Names(IEnumerable<string> names)
    {
        List<object?> l = new List<object?>();
        foreach (string n in names)
        {
            l.Add(n);
        }
        return l;
    }

    private static JObj VarsRecord(DynamicVarSet? set, string[] printed)
    {
        JObj o = new JObj();
        if (set == null)
        {
            return o;
        }
        foreach (string k in set.Keys.OrderBy(x => x, StringComparer.Ordinal))
        {
            DynamicVar v = set[k];
            if (v is BoolVar || v is StringVar)
            {
                continue;
            }
            decimal b = v.BaseValue;
            object val = decimal.Truncate(b) == b ? (object)(long)b : (double)b;
            o[k] = new JObj { ["base"] = val, ["printed"] = Array.IndexOf(printed, k) >= 0, ["var_class"] = VocabExport.VarMeta(v)["var_class"] };
        }
        return o;
    }

    private static CardPoolModel? SafePoolModel(CardModel c)
    {
        foreach (CardPoolModel p in ModelDb.AllCardPools)
        {
            if (p.AllCardIds.Contains(c.Id))
            {
                return p;
            }
        }
        return null;
    }

    private static string? SafePool(CardModel c) => SafePoolModel(c)?.Id.Entry;

    private static string? PoolCharacter(AbstractModel? pool)
    {
        if (pool == null)
        {
            return null;
        }
        foreach (CharacterModel ch in ModelDb.AllCharacters)
        {
            if (ReferenceEquals(ch.CardPool, pool) || ReferenceEquals(ch.RelicPool, pool) || ReferenceEquals(ch.PotionPool, pool))
            {
                return ch.Id.Entry;
            }
        }
        return null;
    }

    // ---------------------------------------------------------------------------------- assembly

    public JObj Build()
    {
        JObj cats = new JObj();
        cats["card"] = ByVocab("card", entry => _cards.TryGetValue(entry, out List<object?>? l) ? l : null);
        cats["relic"] = ByModel("relic", "RELIC", m => Relic((RelicModel)m));
        cats["potion"] = ByModel("potion", "POTION", m => Potion((PotionModel)m));
        cats["power"] = ByModel("power", "POWER", m => Power((PowerModel)m));
        cats["monster"] = ByModel("monster", "MONSTER", m => Monster((MonsterModel)m));
        cats["encounter"] = ByModel("encounter", "ENCOUNTER", m => Encounter((EncounterModel)m));
        cats["event"] = ByModel("event", "EVENT", m => Event((EventModel)m));
        cats["enchantment"] = ByModel("enchantment", "ENCHANTMENT", m => Enchantment((EnchantmentModel)m));
        cats["affliction"] = ByModel("affliction", "AFFLICTION", m => Affliction((AfflictionModel)m));
        cats["act"] = ByModel("act", "ACT", m => Act((ActModel)m));
        cats["character"] = ByModel("character", "CHARACTER", m => Character((CharacterModel)m));
        cats["orb"] = ByModel("orb", "ORB", m => new JObj { ["printed_vars"] = Names(_templates.PrintedVars(m)) });
        JObj varCat = new JObj();
        VocabSet.Category vn = _vocab["var_name"];
        for (int i = 2; i < vn.Count; i++)
        {
            if (vn.Meta != null && vn.Meta.TryGetValue(vn.Entries[i], out object? meta) && meta is JObj mo)
            {
                varCat[i.ToString()] = mo;
            }
        }
        cats["var_name"] = varCat;
        return new JObj
        {
            ["schema"] = 1L,
            ["game_version"] = _vocab.GameVersion,
            ["vocab_hash"] = (long)_vocab.Hash,
            ["vocab_version"] = (long)_vocab.VocabVersion,
            ["categories"] = cats,
        };
    }

    private JObj ByVocab(string cat, Func<string, object?> make)
    {
        JObj o = new JObj();
        VocabSet.Category c = _vocab[cat];
        for (int i = 2; i < c.Count; i++)
        {
            if (c.Retired.Contains(c.Entries[i]))
            {
                continue;
            }
            object? e = make(c.Entries[i]);
            if (e != null)
            {
                o[i.ToString()] = e;
            }
        }
        return o;
    }

    private JObj ByModel(string cat, string gameCat, Func<AbstractModel, object?> make)
    {
        Dictionary<string, AbstractModel> byEntry = new Dictionary<string, AbstractModel>(StringComparer.Ordinal);
        foreach (AbstractModel m in ModelDb.All)
        {
            if (m.Id.Category == gameCat && !VocabExport.IsMock(m))
            {
                byEntry[m.Id.Entry] = m;
            }
        }
        return ByVocab(cat, entry => byEntry.TryGetValue(entry, out AbstractModel? m) ? make(m) : null);
    }

    private JObj Relic(RelicModel r)
    {
        string? pool = null;
        foreach (var p in ModelDb.AllRelicPools)
        {
            if (p.AllRelicIds.Contains(r.Id))
            {
                pool = p.Id.Entry;
                break;
            }
        }
        string[] printed = _templates.PrintedVars(r);
        JObj o = new JObj
        {
            ["rarity"] = r.Rarity.ToString(),
            ["pool"] = pool,
            ["character"] = pool == null ? null : PoolCharacter(ModelDb.AllRelicPools.First(p => p.Id.Entry == pool)),
            ["is_stackable"] = r.IsStackable,
            ["adds_pet"] = r.AddsPet,
            ["spawns_pets"] = r.SpawnsPets,
            ["has_upon_pickup_effect"] = r.HasUponPickupEffect,
            ["is_allowed_in_shops"] = r.IsAllowedInShops,
            ["printed_vars"] = Names(printed),
            ["vars"] = VarsRecord(r.DynamicVars, printed),
            ["text"] = Templates.Raw("relics", r.Id.Entry + ".description"),
        };
        try
        {
            o["merchant_cost"] = (long)r.MerchantCost;
        }
        catch (Exception e) when (e is ArgumentOutOfRangeException || e is InvalidOperationException || e is System.Runtime.CompilerServices.SwitchExpressionException)
        {
            o["merchant_cost"] = null; // rarities without a shop price (starter / event / ancient relics)
        }
        return o;
    }

    private JObj Potion(PotionModel p)
    {
        string? pool = null;
        foreach (var pp in ModelDb.AllPotionPools)
        {
            if (pp.AllPotionIds.Contains(p.Id))
            {
                pool = pp.Id.Entry;
                break;
            }
        }
        string[] printed = _templates.PrintedVars(p);
        return new JObj
        {
            ["rarity"] = p.Rarity.ToString(),
            ["usage"] = p.Usage.ToString(),
            ["target_type"] = p.TargetType.ToString(),
            ["pool"] = pool,
            ["can_be_generated_in_combat"] = p.CanBeGeneratedInCombat,
            ["printed_vars"] = Names(printed),
            ["vars"] = VarsRecord(p.DynamicVars, printed),
        };
    }

    private JObj Power(PowerModel w)
    {
        string[] printed = _templates.PrintedVars(w);
        return new JObj
        {
            ["type"] = w.Type.ToString(),
            ["stack_type"] = w.StackType.ToString(),
            ["instance_type"] = w.InstanceType.ToString(),
            ["allow_negative"] = w.AllowNegative,
            ["should_scale_in_multiplayer"] = w.ShouldScaleInMultiplayer,
            ["printed_vars"] = Names(printed),
            ["vars"] = VarsRecord(w.DynamicVars, printed),
        };
    }

    private JObj Monster(MonsterModel m)
    {
        JObj o = new JObj();
        if (_hpA0.TryGetValue(m.Id.Entry, out (int min, int max) a0))
        {
            o["hp_min"] = (long)a0.min;
            o["hp_max"] = (long)a0.max;
        }
        if (_hpA8.TryGetValue(m.Id.Entry, out (int min, int max) a8))
        {
            o["hp_min_a8"] = (long)a8.min;
            o["hp_max_a8"] = (long)a8.max;
        }
        o["hp_sentinel"] = Math.Max(a0.max, a8.max) >= 9999;
        o["acts"] = Names(ModelDb.Acts.Where(a => a.AllMonsters.Contains(m)).Select(a => a.Id.Entry).OrderBy(x => x, StringComparer.Ordinal));
        o["encounters"] = Names(ModelDb.AllEncounters.Where(e => e.AllPossibleMonsters.Contains(m)).Select(e => e.Id.Entry).OrderBy(x => x, StringComparer.Ordinal));
        return o;
    }

    private JObj Encounter(EncounterModel e)
    {
        return new JObj
        {
            ["room_type"] = e.RoomType.ToString(),
            ["is_weak"] = e.IsWeak,
            ["acts"] = Names(ModelDb.Acts.Where(a => a.AllEncounters.Contains(e)).Select(a => a.Id.Entry).OrderBy(x => x, StringComparer.Ordinal)),
            ["monsters"] = Names(e.AllPossibleMonsters.Select(x => x.Id.Entry).Distinct().OrderBy(x => x, StringComparer.Ordinal)),
            ["slots"] = Names(e.Slots.Where(s => !string.IsNullOrEmpty(s))),
        };
    }

    private JObj Event(EventModel ev)
    {
        string prefix = ev.Id.Entry + ".pages.";
        string table = ev.LocTable;
        List<string> keys = LocKeys(table).Where(k => k.StartsWith(prefix, StringComparison.Ordinal)).ToList();
        return new JObj
        {
            ["is_shared"] = ev.IsShared,
            ["is_ancient"] = ev is AncientEventModel,
            ["layout_type"] = ev.LayoutType.ToString(),
            ["loc_table"] = table,
            ["acts"] = Names(ModelDb.Acts.Where(a => a.AllEvents.Contains(ev) || a.AllAncients.Contains(ev)).Select(a => a.Id.Entry).OrderBy(x => x, StringComparer.Ordinal)),
            ["option_keys"] = Names(keys.Where(k => k.Contains(".options.", StringComparison.Ordinal) && k.EndsWith(".title", StringComparison.Ordinal)).Select(k => k.Substring(0, k.Length - 6))),
            ["page_keys"] = Names(keys.Where(k => !k.Contains(".options.", StringComparison.Ordinal) && k.EndsWith(".description", StringComparison.Ordinal))),
        };
    }

    private static IEnumerable<string> LocKeys(string table)
    {
        return MegaCrit.Sts2.Core.Localization.LocManager.Instance._tables.TryGetValue(table, out var t)
            ? t.Keys.OrderBy(x => x, StringComparer.Ordinal)
            : Enumerable.Empty<string>();
    }

    private JObj Enchantment(EnchantmentModel e)
    {
        string[] printed = _templates.PrintedVars(e);
        return new JObj
        {
            ["show_amount"] = e.ShowAmount,
            ["is_stackable"] = e.IsStackable,
            ["has_extra_card_text"] = e.HasExtraCardText,
            ["printed_vars"] = Names(printed),
            ["vars"] = VarsRecord(e.DynamicVars, printed),
        };
    }

    private JObj Affliction(AfflictionModel a)
    {
        return new JObj
        {
            ["can_afflict_unplayable"] = a.CanAfflictUnplayableCards,
            ["is_stackable"] = a.IsStackable,
            ["has_extra_card_text"] = a.HasExtraCardText,
            ["printed_vars"] = Names(_templates.PrintedVars(a)),
        };
    }

    private static JObj Act(ActModel a)
    {
        return new JObj
        {
            ["index"] = (long)a.Index,
            ["boss_encounters"] = Names(a.AllBossEncounters.Select(e => e.Id.Entry).OrderBy(x => x, StringComparer.Ordinal)),
            ["elite_encounters"] = Names(a.AllEliteEncounters.Select(e => e.Id.Entry).OrderBy(x => x, StringComparer.Ordinal)),
            ["ancients"] = Names(a.AllAncients.Select(e => e.Id.Entry).OrderBy(x => x, StringComparer.Ordinal)),
        };
    }

    private static JObj Character(CharacterModel c)
    {
        return new JObj
        {
            ["starting_hp"] = (long)c.StartingHp,
            ["starting_gold"] = (long)c.StartingGold,
            ["max_energy"] = (long)c.MaxEnergy,
            ["base_orb_slots"] = (long)c.BaseOrbSlotCount,
            ["starting_deck"] = Names(c.StartingDeck.Select(x => x.Id.Entry)),
            ["starting_relics"] = Names(c.StartingRelics.Select(x => x.Id.Entry)),
            ["pools"] = new JObj { ["card"] = c.CardPool.Id.Entry, ["relic"] = c.RelicPool.Id.Entry, ["potion"] = c.PotionPool.Id.Entry },
        };
    }
}
