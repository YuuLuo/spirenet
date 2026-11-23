using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using SpireHost.Driver;
using SpireHost.Obs.Schema;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs.Encoding;

public sealed partial class ObsEncoder
{
    // =====================================================================================================
    //  Kind-specific tables: OPTIONS / REFS / OFFER entities / CELLS + GLOBAL 24-42 (schema 4, 5.11, 5.12, 6)
    // =====================================================================================================

    private void EncodeKindSpecific(DecisionKind kind)
    {
        if (_d == null)
        {
            return; // TERMINAL: final state only
        }
        object? context = _d.Context;
        int seat = _d.ActingSeat >= 0 ? _d.ActingSeat : 0;
        int[] g = _w.Global;
        switch (kind)
        {
            case DecisionKind.CardSelect when context is CardSelectContext sel:
                SelectionGlobals(sel);
                break;
            case DecisionKind.Event when context is EventContext ev:
                EventOptions(ev, seat);
                break;
            case DecisionKind.Rest when context is RestContext rc:
                RestOptions(rc, seat);
                break;
            case DecisionKind.Reward when context is RewardContext rw:
                RewardOptions(rw, seat);
                break;
            case DecisionKind.CardReward when context is CardRewardContext cr:
                CardRewardOptions(cr, seat);
                break;
            case DecisionKind.Shop when context is ShopContext sc:
                ShopOptions(sc, seat);
                break;
            case DecisionKind.Treasure:
                g[Gl.TreasurePhase] = _d.SubKind;
                if (context is TreasureContext tc && _d.SubKind == (int)TreasurePhase.Relic)
                {
                    TreasureOptions(tc, seat);
                }
                break;
            case DecisionKind.Bundle when context is BundleContext bc:
                BundleOptions(bc, seat);
                break;
            case DecisionKind.Confirm:
                ConfirmOptions(context as ConfirmContext, seat);
                break;
            case DecisionKind.Crystal when context is CrystalContext cc:
                CrystalCells(cc);
                break;
        }
    }

    private void SelectionGlobals(CardSelectContext sel)
    {
        int[] g = _w.Global;
        // GLOBAL 24 prompt_key: the header the real screen shows (contract C2: CardSelectorPrefs.Prompt for the 13 entry points
        // that have prefs, the fixed header of the screen for FromChooseACardScreen / FromHandForUpgrade). A context whose
        // builder did not fill it (synthetic test prompts) falls back to the prefs' prompt.
        string table = sel.PromptTable, key = sel.PromptKey;
        if (key.Length == 0)
        {
            LocString? prompt = sel.Prefs?.Prompt;
            if (prompt != null && !prompt.IsEmpty)
            {
                table = prompt.LocTable;
                key = prompt.LocEntryKey;
            }
        }
        g[Gl.PromptKey] = key.Length == 0 ? 0 : _v.Id("prompt_key", table + "/" + key);
        (int cat, int id) = ModelRef(sel.SourceModel);
        g[Gl.PromptSourceCat] = cat;
        g[Gl.PromptSourceId] = id;
        g[Gl.SelMin] = sel.MinSelect;
        g[Gl.SelMax] = sel.MaxSelect;
        g[Gl.SelCount] = sel.Picked.Count;
        int f = 0;
        if (sel.CanSkip)
        {
            f |= SelFlag.CanSkip;
        }
        if (sel.Cancelable)
        {
            f |= SelFlag.Cancelable;
        }
        if (sel.RequireManualConfirmation)
        {
            f |= SelFlag.ManualConfirm;
        }
        if (sel.Prefs?.UnpoweredPreviews == true)
        {
            f |= SelFlag.UnpoweredPreviews;
        }
        if (sel.Prefs?.PretendCardsCanBePlayed == true)
        {
            f |= SelFlag.PretendPlayable;
        }
        if (_hasUpgradePreview)
        {
            f |= SelFlag.UpgradePreview;
        }
        g[Gl.SelFlags] = f;
    }

    /// <summary>REFS category (errata E-1 numbering) + vocab id of a model.</summary>
    private (int cat, int id) ModelRef(AbstractModel? m)
    {
        return m switch
        {
            null => (RefCategory.Unknown, 0),
            CardModel => (RefCategory.Card, _v.Model("card", m)),
            RelicModel => (RefCategory.Relic, _v.Model("relic", m)),
            PotionModel => (RefCategory.Potion, _v.Model("potion", m)),
            PowerModel => (RefCategory.Power, _v.Model("power", m)),
            EnchantmentModel => (RefCategory.Enchantment, _v.Model("enchantment", m)),
            AfflictionModel => (RefCategory.Affliction, _v.Model("affliction", m)),
            OrbModel => (RefCategory.Orb, _v.Model("orb", m)),
            MonsterModel => (RefCategory.Monster, _v.Model("monster", m)),
            EventModel => (RefCategory.Event, _v.Model("event", m)),
            _ => (RefCategory.Unknown, 0),
        };
    }

    private int AddOption(int kind, int seat, int pos, int keyId)
    {
        int r = _w.AddRow(T.Options);
        Span<int> row = _w.Row(T.Options, r);
        row[OptionC.Kind] = kind;
        row[OptionC.Seat] = seat;
        row[OptionC.Pos] = pos;
        row[OptionC.KeyId] = keyId;
        row[OptionC.Price] = -1;
        row[OptionC.Amount] = -1;
        row[OptionC.RefTable] = -1;
        row[OptionC.RefRow] = -1;
        row[OptionC.Group] = -1;
        row[OptionC.RefOff] = -1;
        row[OptionC.RefCnt] = 0;
        return r;
    }

    private void MapOption(RefKind kind, int index, int row)
    {
        _optRefKind = (int)kind;
        if (index >= _optRow.Length)
        {
            int[] n = new int[Math.Max(index + 1, _optRow.Length * 2)];
            Array.Fill(n, -1);
            Array.Copy(_optRow, n, _optRow.Length);
            _optRow = n;
        }
        _optRow[index] = row;
    }

    private void ResetOptionMap()
    {
        Array.Fill(_optRow, -1);
    }

    private int OfferCard(CardModel card, int seat, int pos, int group)
    {
        return EncodeCardRow(card, Zone.Offer, seat, pos, 1, CardPreviewMode.Normal, unpowered: false, linkRow: -1, aux: group, selCount: 0);
    }

    private void SetRef(int optRow, int table, int row)
    {
        _w.Set(T.Options, optRow, OptionC.RefTable, table);
        _w.Set(T.Options, optRow, OptionC.RefRow, row);
    }

    private int AddRefRow(int optRow, int category, int vocabId, int entTable, int entRow)
    {
        int r = _w.AddRow(T.Refs);
        Span<int> row = _w.Row(T.Refs, r);
        row[RefC.OwnerTable] = T.Options;
        row[RefC.OwnerRow] = optRow;
        row[RefC.Category] = category;
        row[RefC.VocabId] = vocabId;
        row[RefC.EntTable] = entTable;
        row[RefC.EntRow] = entRow;
        return r;
    }

    private void CloseRefs(int optRow, int firstRef)
    {
        int n = _w.Count(T.Refs) - firstRef;
        if (n > 0)
        {
            _w.Set(T.Options, optRow, OptionC.RefOff, firstRef);
            _w.Set(T.Options, optRow, OptionC.RefCnt, n);
        }
    }

    private readonly List<KeyValuePair<string, int>> _locVars = new List<KeyValuePair<string, int>>(8);

    /// <summary>OPTIONS VARS rows from (name, shown) pairs: base = enchanted = shown (errata E-27), flags bit1 (LocString-rendered).</summary>
    private void AddOptionVars(int optRow, List<KeyValuePair<string, int>> vars)
    {
        for (int i = 0; i < vars.Count; i++)
        {
            int id = _v.Id("var_name", vars[i].Key);
            if (_w.HasVar(T.Options, optRow, -1, id))
            {
                continue;
            }
            int v = vars[i].Value;
            _w.AddVar(T.Options, optRow, -1, id, v, v, v, VarFlag.LocString);
        }
    }

    // ---------------------------------------------------------------------------------------------- EVENT

    /// <summary>
    /// EVENT (schema 5.11, GLOBAL 31-33) from the EventContext the events module filled (contract R1): the screen exactly as
    /// the UI shows it. Nothing else is read (no side table; EventOptionInfo.Option only for the vocabulary name rule).
    /// </summary>
    private void EventOptions(EventContext ec, int seat)
    {
        EventModel ev = ec.Event;
        int[] g = _w.Global;
        g[Gl.EventId] = _v.Model("event", ev);
        g[Gl.EventPageKey] = ec.PageKey == null ? 0 : _v.Id("event_page", ec.PageKey);
        g[Gl.EventFlags] = (ec.IsShared ? 1 : 0) | (ec.IsAncient ? 2 : 0) | (ec.Finished ? 4 : 0);
        if (ec.Finished)
        {
            return; // the UI-synthesized PROCEED has no OPTIONS row (its action has no source)
        }
        ResetOptionMap();
        List<EventOptionInfo> buttons = ec.Options;
        for (int k = 0; k < buttons.Count; k++)
        {
            EventOptionInfo b = buttons[k];
            if (b.Synthesized || b.Index < 0)
            {
                continue;
            }
            int i = b.Index;
            // TextKeys that are not loc entries map to structural keys (relic choices -> RELIC_OPTION, dialogue -> <EVENT>.dialogue; errata V-6)
            bool offersRelic = b.Relic != null;
            for (int m = 0; !offersRelic && m < b.Models.Count; m++)
            {
                offersRelic = b.Models[m] is RelicModel;
            }
            string key = VocabExport.EventOptionName(_v, b.TextKey, offersRelic);
            int r = AddOption(OptionKind.EventOption, seat, i, _v.Id("event_option_key", key));
            MapOption(RefKind.EventOption, i, r);
            int f = 0;
            if (b.IsLocked)
            {
                f |= OptionFlag.Locked;
            }
            if (b.WillKillPlayer)
            {
                f |= OptionFlag.Lethal; // NEventOptionButton kill warning (Nodes.Events/NEventOptionButton.cs:214,266,338)
            }
            if (b.IsProceed)
            {
                f |= OptionFlag.IsProceed;
            }
            if (b.WasChosen)
            {
                f |= OptionFlag.WasChosen;
            }
            for (int s = 0; s < ec.SeatVotes.Count; s++)
            {
                if (ec.SeatVotes[s] == i)
                {
                    f |= OptionFlag.VoteSeat0 << s; // erratum H5: the votes on a shared event page are public
                }
            }
            _w.Set(T.Options, r, OptionC.Flags, f);
            AddOptionVars(r, b.Vars);
            int firstRef = _w.Count(T.Refs);
            // REFS order (errata E-10): the option's relic icon, then card hover tips, then model hover tips (each in hover-tip order)
            if (b.Relic != null)
            {
                int rr = AddRelicRow(b.Relic, seat, ItemZone.Offer, i);
                SetRef(r, T.Relics, rr);
                AddRefRow(r, RefCategory.Relic, _v.Model("relic", b.Relic), T.Relics, rr);
            }
            for (int c = 0; c < b.Cards.Count; c++)
            {
                int cr = OfferCard(b.Cards[c], seat, i, -1);
                AddRefRow(r, RefCategory.Card, _v.Model("card", b.Cards[c]), T.Cards, cr);
            }
            for (int m = 0; m < b.Models.Count; m++)
            {
                ModelRefRow(r, b.Models[m], seat, i);
            }
            CloseRefs(r, firstRef);
        }
    }

    /// <summary>A REFS row for a hover-tip model: category + vocab id; relics and potions also get an OFFER row (schema 5.11).</summary>
    private void ModelRefRow(int optRow, AbstractModel m, int seat, int pos)
    {
        (int cat, int id) = ModelRef(m);
        if (cat == RefCategory.Unknown)
        {
            return;
        }
        if (m is RelicModel rm)
        {
            int rr = AddRelicRow(rm, seat, ItemZone.Offer, pos);
            AddRefRow(optRow, cat, id, T.Relics, rr);
        }
        else if (m is PotionModel pm)
        {
            int pr = AddPotionRow(pm, seat, ItemZone.Offer, pos);
            AddRefRow(optRow, cat, id, T.Potions, pr);
        }
        else
        {
            AddRefRow(optRow, cat, id, -1, -1);
        }
    }

    // ---------------------------------------------------------------------------------------------- REST

    private void RestOptions(RestContext rc, int seat)
    {
        ResetOptionMap();
        IReadOnlyList<RestSiteOption> options = rc.Options;
        bool any = false;
        for (int i = 0; i < options.Count; i++)
        {
            RestSiteOption o = options[i];
            int r = AddOption(OptionKind.RestOption, seat, i, _v.Id("rest_option", o.OptionId));
            MapOption(RefKind.RestOption, i, r);
            if (!o.IsEnabled)
            {
                _w.Set(T.Options, r, OptionC.Flags, OptionFlag.Locked);
            }
            else
            {
                any = true;
            }
            _locVars.Clear();
            LocText.PrintedVars(null, o.Description, _locVars);
            AddOptionVars(r, _locVars);
        }
        _w.Global[Gl.RestFlags] = (any ? 1 : 0) | (rc.Successes > 0 ? 2 : 0);
    }

    // ---------------------------------------------------------------------------------------------- REWARD

    private void RewardOptions(RewardContext rc, int seat)
    {
        ResetOptionMap();
        RewardsSet set = rc.Set;
        _w.Global[Gl.RewardFlags] = set.DisallowSkipping ? 1 : 0;
        List<Reward> rewards = set.Rewards;
        Player player = set.Player;
        for (int i = 0; i < rewards.Count; i++)
        {
            Reward rw = rewards[i];
            if (rw.SuccessfullySelected)
            {
                continue; // taken rewards disappear from the screen
            }
            int r = AddOption(OptionKind.RewardItem, seat, i, Enum(Ev.RewardType, (int)rw.RewardType));
            MapOption(RefKind.Reward, i, r);
            switch (rw)
            {
                case GoldReward gr:
                    _w.Set(T.Options, r, OptionC.Amount, gr.Amount); // NRewardButton (Nodes.Rewards/NRewardButton.cs:180)
                    break;
                case PotionReward pr when pr.Potion != null:
                    SetRef(r, T.Potions, AddPotionRow(pr.Potion, seat, ItemZone.Offer, i));
                    if (!player.HasOpenPotionSlots)
                    {
                        _w.Set(T.Options, r, OptionC.Flags, OptionFlag.Locked);
                    }
                    break;
                case RelicReward rr when rr.Relic != null:
                    SetRef(r, T.Relics, AddRelicRow(rr.Relic, seat, ItemZone.Offer, i));
                    break;
                case SpecialCardReward sr when sr._card != null:
                    SetRef(r, T.Cards, OfferCard(sr._card, seat, i, -1));
                    break;
                // CardReward: the three cards are NOT shown before the reward is opened (DECISIONS-H1 F.4)
            }
        }
    }

    // ---------------------------------------------------------------------------------------------- CARD_REWARD

    private void CardRewardOptions(CardRewardContext cr, int seat)
    {
        ResetOptionMap();
        CardReward reward = cr.Reward;
        _w.Global[Gl.CardRewardFlags] = (reward.CanSkip ? 1 : 0) | (reward.CanReroll ? 2 : 0);
        IReadOnlyList<CardCreationResult> options = cr.Options;
        if (_offerCardRow.Length < options.Count)
        {
            _offerCardRow = new int[Math.Max(options.Count, _offerCardRow.Length * 2)];
        }
        for (int i = 0; i < options.Count; i++)
        {
            _offerCardRow[i] = OfferCard(options[i].Card, seat, i, -1);
        }
        IReadOnlyList<CardRewardAlternative> alts = cr.Alternatives;
        for (int k = 0; k < alts.Count; k++)
        {
            int r = AddOption(OptionKind.CardRewardAlt, seat, k, _v.Id("card_reward_alt", alts[k].OptionId.ToUpperInvariant()));
            MapOption(RefKind.CardRewardAlt, k, r);
        }
    }

    // ---------------------------------------------------------------------------------------------- SHOP

    private void ShopOptions(ShopContext sc, int seat)
    {
        ResetOptionMap();
        _w.Global[Gl.ShopFlags] = sc.FakeMerchant ? 1 : 0;
        MerchantInventory inv = sc.Inventory;
        List<MerchantEntry> entries = inv.AllEntries;
        for (int i = 0; i < entries.Count; i++)
        {
            MerchantEntry e = entries[i];
            string section = e switch
            {
                MerchantCardEntry ce when Contains(inv.ColorlessCardEntries, ce) => "COLORLESS_CARD",
                MerchantCardEntry => "CHARACTER_CARD",
                MerchantRelicEntry => "RELIC",
                MerchantPotionEntry => "POTION",
                MerchantCardRemovalEntry => "CARD_REMOVAL",
                // an entry type this encoder does not know: not a vocabulary name, so the lookup below returns UNK (the same id 1 as
                // before) AND counts it (obs verifier finding F5: the literal id bypassed the UNK-lookup alarm)
                _ => "UNRECOGNISED:" + e.GetType().Name,
            };
            int r = AddOption(OptionKind.ShopEntry, seat, i, _v.Id("shop_section", section));
            MapOption(RefKind.ShopEntry, i, r);
            int f = 0;
            if (e.IsStocked)
            {
                f |= OptionFlag.Stocked;
            }
            if (e.EnoughGold)
            {
                f |= OptionFlag.Affordable;
            }
            if (e is MerchantCardEntry mc && mc.IsOnSale)
            {
                f |= OptionFlag.OnSale;
            }
            _w.Set(T.Options, r, OptionC.Flags, f);
            _w.Set(T.Options, r, OptionC.Price, e.IsStocked ? e.Cost : -1);
            switch (e)
            {
                case MerchantCardEntry ce when ce.CreationResult != null:
                    SetRef(r, T.Cards, OfferCard(ce.CreationResult.Card, seat, i, -1));
                    break;
                case MerchantRelicEntry re when re.Model != null:
                    SetRef(r, T.Relics, AddRelicRow(re.Model, seat, ItemZone.Offer, i));
                    break;
                case MerchantPotionEntry pe when pe.Model != null:
                    SetRef(r, T.Potions, AddPotionRow(pe.Model, seat, ItemZone.Offer, i));
                    break;
            }
        }
    }

    private static bool Contains<TItem>(IReadOnlyList<TItem> list, TItem item) where TItem : class
    {
        for (int i = 0; i < list.Count; i++)
        {
            if (ReferenceEquals(list[i], item))
            {
                return true;
            }
        }
        return false;
    }

    // ---------------------------------------------------------------------------------------------- TREASURE / BUNDLE / CONFIRM

    private void TreasureOptions(TreasureContext tc, int seat)
    {
        ResetOptionMap();
        IReadOnlyList<RelicModel> relics = tc.Relics;
        for (int i = 0; i < relics.Count; i++)
        {
            int r = AddOption(OptionKind.TreasureRelic, seat, i, 0);
            MapOption(RefKind.TreasureRelic, i, r);
            SetRef(r, T.Relics, AddRelicRow(relics[i], seat, ItemZone.Offer, i));
            int f = 0;
            for (int s = 0; s < tc.SeatVotes.Count; s++)
            {
                if (tc.SeatVotes[s] == i)
                {
                    f |= OptionFlag.VoteSeat0 << s; // erratum H5: the treasure votes are public (NTreasureRoomRelicCollection)
                }
            }
            if (f != 0)
            {
                _w.Set(T.Options, r, OptionC.Flags, f);
            }
        }
    }

    private void BundleOptions(BundleContext bc, int seat)
    {
        ResetOptionMap();
        _w.Global[Gl.BundleFlags] = bc.CanSkip ? 1 : 0;
        IReadOnlyList<IReadOnlyList<CardModel>> bundles = bc.Bundles;
        // cards first (zone OFFER, aux = bundle index), then one option per bundle referencing its cards
        int[] firstCard = new int[bundles.Count];
        for (int b = 0; b < bundles.Count; b++)
        {
            firstCard[b] = _w.Count(T.Cards);
            for (int k = 0; k < bundles[b].Count; k++)
            {
                OfferCard(bundles[b][k], seat, k, b);
            }
        }
        for (int b = 0; b < bundles.Count; b++)
        {
            int r = AddOption(OptionKind.Bundle, seat, b, 0);
            MapOption(RefKind.Bundle, b, r);
            _w.Set(T.Options, r, OptionC.Group, b);
            int firstRef = _w.Count(T.Refs);
            for (int k = 0; k < bundles[b].Count; k++)
            {
                AddRefRow(r, RefCategory.Card, _v.Model("card", bundles[b][k]), T.Cards, firstCard[b] + k);
            }
            CloseRefs(r, firstRef);
        }
    }

    private void ConfirmOptions(ConfirmContext? cc, int seat)
    {
        ResetOptionMap();
        _w.Global[Gl.ConfirmKey] = cc == null || cc.Key.Length == 0 ? 0 : _v.Id("confirm_key", cc.Key);
        int no = AddOption(OptionKind.ConfirmChoice, seat, 0, _v.Id("confirm_key", "NO"));
        MapOption(RefKind.ConfirmChoice, 0, no);
        int yes = AddOption(OptionKind.ConfirmChoice, seat, 1, _v.Id("confirm_key", "YES"));
        MapOption(RefKind.ConfirmChoice, 1, yes);
    }

    // ---------------------------------------------------------------------------------------------- CRYSTAL

    /// <summary>
    /// CRYSTAL (schema 5.12, GLOBAL 41-42) from the PUBLIC view of the context (contract R2): hidden cells carry no item
    /// information. CrystalContext.Minigame (host-only: the hidden item layout) is never read.
    /// </summary>
    private void CrystalCells(CrystalContext cc)
    {
        int[] g = _w.Global;
        g[Gl.CrystalDivinations] = cc.Divinations;
        g[Gl.CrystalTool] = Enum(Ev.CrystalSphereToolType, (int)cc.Tool);
        List<CrystalCellView> cells = cc.Cells;
        int n = cells.Count;
        if (_cellRow.Length < n)
        {
            _cellRow = new int[n];
        }
        for (int i = 0; i < n; i++)
        {
            CrystalCellView c = cells[i];
            int r = _w.AddRow(T.Cells);
            _cellRow[i] = r; // list index = y * width + x = the CrystalCell ordinal
            Span<int> row = _w.Row(T.Cells, r);
            row[CellC.X] = c.X;
            row[CellC.Y] = c.Y;
            row[CellC.Hidden] = c.Hidden ? 1 : 0;
            if (c.Hidden || c.ItemType == -2)
            {
                // fog (Nodes.Events.Custom.CrystalSphere/NCrystalSphereCell.cs:68,92,98): nothing of the item
                row[CellC.ItemType] = c.Hidden ? -1 : VocabSet.UnkId;
                row[CellC.ItemGood] = -1;
                row[CellC.ItemIdx] = -1;
                continue;
            }
            row[CellC.ItemType] = c.ItemType < 0 ? 0 : Enum(Ev.CrystalSphereItemType, c.ItemType);
            row[CellC.ItemGood] = c.ItemType < 0 ? -1 : c.ItemGood;
            row[CellC.ItemIdx] = c.ItemType < 0 ? -1 : c.ItemIndex;
        }
    }
}
