using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;
using SpireHost.Engine;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Encoding;

public sealed partial class ObsEncoder
{
    /// <summary>CARDS columns that are part of the multiset content key (schema 5.2.3: all but pos/count/var_off/var_cnt/link_row/sel_count).</summary>
    private static readonly int[] KeyCols =
    {
        CardC.Zone, CardC.Seat, CardC.CardId, CardC.UpgradeLevel, CardC.EnergyCost, CardC.CostFlags, CardC.EnergyCostColor,
        CardC.StarCost, CardC.StarCostColor, CardC.KeywordBits, CardC.EnchantId, CardC.EnchantAmount, CardC.EnchantStatus,
        CardC.AfflictionId, CardC.AfflictionAmount, CardC.ReplayCount, CardC.HandFlags, CardC.UnplayableReason, CardC.Aux,
    };

    private readonly List<Creature> _targets = new List<Creature>(8);
    private decimal[] _savedVals = new decimal[64];
    private int[] _keyBuf = new int[4096];
    private int[] _keyOff = new int[256];
    private int[] _keyLen = new int[256];
    private int[] _order = new int[256];
    private int[] _newRel = new int[256];
    private int[] _permScratch = new int[256 * 25];
    private readonly int[] _varIdx = new int[64];

    // =====================================================================================================
    //  CARDS (schema 5.2): zones in ascending zone order, seats ascending inside a zone
    // =====================================================================================================

    private void EncodeCards(DecisionKind kind)
    {
        IReadOnlyList<Player> players = _run.Players;
        if (_inCombat)
        {
            for (int s = 0; s < _nSeats; s++)
            {
                PlayerCombatState? pcs = players[s].PlayerCombatState;
                if (pcs == null || !Options.SeesPrivate(s))
                {
                    continue;
                }
                IReadOnlyList<CardModel> hand = pcs.Hand.Cards;
                _handKnown = KnownPlayability(s, players[s], hand.Count);
                for (int i = 0; i < hand.Count; i++)
                {
                    _handIndex = i;
                    _cardRow[hand[i]] = EncodeCardRow(hand[i], Zone.Hand, s, i, 1, CardPreviewMode.Normal, unpowered: false, linkRow: -1, aux: -1, selCount: 0);
                }
                _handKnown = null;
            }
            if (HostProfile.On)
            {
                HostProfile.EncLap(9);
            }
            EncodePileZone(Zone.Draw, PileType.Draw);
            if (HostProfile.On)
            {
                HostProfile.EncLap(10);
            }
            EncodePileZone(Zone.Discard, PileType.Discard);
            if (HostProfile.On)
            {
                HostProfile.EncLap(11);
            }
            EncodePileZone(Zone.Exhaust, PileType.Exhaust);
            if (HostProfile.On)
            {
                HostProfile.EncLap(12);
            }
            for (int s = 0; s < _nSeats; s++)
            {
                PlayerCombatState? pcs = players[s].PlayerCombatState;
                if (pcs == null || !Options.SeesPrivate(s))
                {
                    continue;
                }
                IReadOnlyList<CardModel> play = pcs.PlayPile.Cards;
                for (int i = 0; i < play.Count; i++)
                {
                    _cardRow[play[i]] = EncodeCardRow(play[i], Zone.Play, s, i, 1, CardPreviewMode.Normal, unpowered: false, linkRow: -1, aux: -1, selCount: 0);
                }
            }
            if (HostProfile.On)
            {
                HostProfile.EncLap(13);
            }
        }
        if (!_inCombat || Options.DeckInCombat)
        {
            // DECK is public for every seat (NMultiplayerPlayerExpandedState.cs:211-217)
            for (int s = 0; s < _nSeats; s++)
            {
                EncodeMultiset(players[s].Deck.Cards, Zone.Deck, s);
            }
            if (HostProfile.On)
            {
                HostProfile.EncLap(14);
            }
        }
        if (kind == DecisionKind.CardSelect && _d?.Context is CardSelectContext sel)
        {
            EncodeChoice(sel);
            if (HostProfile.On)
            {
                HostProfile.EncLap(15);
            }
        }
    }

    /// <summary>
    /// The CanPlay results of a seat's hand that the COMBAT handler evaluated for this very decision (CombatContext, contract
    /// C1: CombatSeatInfo.HandPlayable / HandReason per hand position, filled for the seats that can act). The engine is
    /// quiescent between the handler's Step and the encoder, and CanPlay is a pure function of that state, so the values are
    /// the ones a second call would return. Null = not evaluated (another decision kind, a seat that cannot act): call CanPlay.
    /// </summary>
    private CombatSeatInfo? KnownPlayability(int seat, Player player, int handCount)
    {
        if (Options.ExactGameCalls || _d?.Context is not CombatContext cc || seat >= cc.Seats.Count)
        {
            return null;
        }
        CombatSeatInfo info = cc.Seats[seat];
        return ReferenceEquals(info.Player, player) && info.HandPlayable.Count == handCount ? info : null;
    }

    private CombatSeatInfo? _handKnown;
    private int _handIndex;

    private void EncodePileZone(int zone, PileType pile)
    {
        for (int s = 0; s < _nSeats; s++)
        {
            PlayerCombatState? pcs = _run.Players[s].PlayerCombatState;
            if (pcs == null || !Options.SeesPrivate(s))
            {
                continue;
            }
            CardPile p = pile switch
            {
                PileType.Draw => pcs.DrawPile,
                PileType.Discard => pcs.DiscardPile,
                _ => pcs.ExhaustPile,
            };
            // PublicMultiset: the draw pile's order is hidden information; only its content (as a multiset) is public
            EncodeMultiset(p.Cards, zone, s);
        }
    }

    /// <summary>
    /// The one place that turns a pile into rows (schema 5.2.3 "PublicMultiset.Build"): every instance is encoded,
    /// then the rows are grouped by their full content key (row columns + VARS) and written in ascending key order
    /// with count = group size. The output depends only on the multiset, never on the pile's internal order.
    /// </summary>
    private void EncodeMultiset(IReadOnlyList<CardModel> cards, int zone, int seat)
    {
        int n = cards.Count;
        if (n == 0)
        {
            return;
        }
        int first = _w.Count(T.Cards);
        for (int i = 0; i < n; i++)
        {
            EncodeCardRow(cards[i], zone, seat, -1, 1, CardPreviewMode.Normal, unpowered: false, linkRow: -1, aux: -1, selCount: 0);
        }
        Canonicalize(first, n);
        for (int i = 0; i < n; i++)
        {
            _cardRow[cards[i]] = first + _newRelOfInstance[i];
        }
    }

    private int[] _newRelOfInstance = new int[256];

    /// <summary>Sorts rows [first, first+n) of CARDS by content key, merges equal keys (count), fixes VARS owners. Returns the group count.</summary>
    private int Canonicalize(int first, int n)
    {
        EnsureKeyScratch(n);
        int pos = 0;
        for (int i = 0; i < n; i++)
        {
            _keyOff[i] = pos;
            pos = WriteKey(first + i, pos);
            _keyLen[i] = pos - _keyOff[i];
            _order[i] = i;
        }
        // insertion sort (n is small; stable, deterministic)
        for (int i = 1; i < n; i++)
        {
            int x = _order[i];
            int j = i - 1;
            while (j >= 0 && CompareKeys(_order[j], x) > 0)
            {
                _order[j + 1] = _order[j];
                j--;
            }
            _order[j + 1] = x;
        }
        int groups = 0;
        for (int k = 0; k < n; k++)
        {
            int inst = _order[k];
            if (k > 0 && CompareKeys(_order[k - 1], inst) == 0)
            {
                _newRelOfInstance[inst] = groups - 1;
                _newRel[inst] = -1; // merged: its VARS rows are dropped
                _counts[groups - 1]++;
            }
            else
            {
                _newRelOfInstance[inst] = groups;
                _newRel[inst] = groups;
                _repOf[groups] = inst;
                _counts[groups] = 1;
                groups++;
            }
        }
        // move the representatives into place
        int cols = SchemaV1.NCols[T.Cards];
        Span<int> scratch = _permScratch.AsSpan(0, n * cols);
        for (int i = 0; i < n; i++)
        {
            _w.Row(T.Cards, first + i).CopyTo(scratch.Slice(i * cols, cols));
        }
        for (int g = 0; g < groups; g++)
        {
            Span<int> dst = _w.Row(T.Cards, first + g);
            scratch.Slice(_repOf[g] * cols, cols).CopyTo(dst);
            dst[CardC.Count] = _counts[g];
        }
        _w.Truncate(T.Cards, first + groups);
        _w.RemapVars(T.Cards, first, n, _newRel.AsSpan(0, n));
        return groups;
    }

    private int[] _repOf = new int[256];
    private int[] _counts = new int[256];

    private void EnsureKeyScratch(int n)
    {
        if (_keyOff.Length < n)
        {
            int m = Math.Max(n, _keyOff.Length * 2);
            _keyOff = new int[m];
            _keyLen = new int[m];
            _order = new int[m];
            _newRel = new int[m];
            _newRelOfInstance = new int[m];
            _repOf = new int[m];
            _counts = new int[m];
            _permScratch = new int[m * SchemaV1.NCols[T.Cards]];
        }
    }

    private int WriteKey(int row, int pos)
    {
        int need = pos + KeyCols.Length + 6 * 64;
        if (_keyBuf.Length < need)
        {
            Array.Resize(ref _keyBuf, Math.Max(need, _keyBuf.Length * 2));
        }
        Span<int> r = _w.Row(T.Cards, row);
        for (int c = 0; c < KeyCols.Length; c++)
        {
            _keyBuf[pos++] = r[KeyCols[c]];
        }
        int nv = _w.CollectVarsOf(T.Cards, row, _varIdx);
        // VARS of the instance sorted by (target_row, var_id), each contributing (var_id, target_row, shown, base, enchanted, flags)
        for (int i = 1; i < nv; i++)
        {
            int x = _varIdx[i];
            int j = i - 1;
            while (j >= 0 && VarOrder(_varIdx[j], x) > 0)
            {
                _varIdx[j + 1] = _varIdx[j];
                j--;
            }
            _varIdx[j + 1] = x;
        }
        for (int i = 0; i < nv; i++)
        {
            ReadOnlySpan<int> v = _w.StagedVar(_varIdx[i]);
            _keyBuf[pos++] = v[VarC.VarId];
            _keyBuf[pos++] = v[VarC.TargetRow];
            _keyBuf[pos++] = v[VarC.Shown];
            _keyBuf[pos++] = v[VarC.Base];
            _keyBuf[pos++] = v[VarC.Enchanted];
            _keyBuf[pos++] = v[VarC.Flags];
        }
        return pos;
    }

    private int VarOrder(int a, int b)
    {
        ReadOnlySpan<int> x = _w.StagedVar(a), y = _w.StagedVar(b);
        int c = x[VarC.TargetRow].CompareTo(y[VarC.TargetRow]);
        return c != 0 ? c : x[VarC.VarId].CompareTo(y[VarC.VarId]);
    }

    /// <summary>Lexicographic int32 comparison, a proper prefix is smaller (= Python tuple order).</summary>
    private int CompareKeys(int a, int b)
    {
        int la = _keyLen[a], lb = _keyLen[b];
        int oa = _keyOff[a], ob = _keyOff[b];
        int n = Math.Min(la, lb);
        for (int i = 0; i < n; i++)
        {
            int c = _keyBuf[oa + i].CompareTo(_keyBuf[ob + i]);
            if (c != 0)
            {
                return c;
            }
        }
        return la.CompareTo(lb);
    }

    // =====================================================================================================
    //  CHOICE (schema 5.2.4) and PREVIEW rows
    // =====================================================================================================

    private void EncodeChoice(CardSelectContext sel)
    {
        IReadOnlyList<CardModel> options = sel.Options;
        int seat = SeatOf(sel.Player);
        if (seat < 0)
        {
            seat = _d!.ActingSeat >= 0 ? _d.ActingSeat : 0;
        }
        if (_choiceRow.Length < options.Count)
        {
            _choiceRow = new int[Math.Max(options.Count, _choiceRow.Length * 2)];
        }
        bool fromPile = sel.OptionCounts != null || sel.Pile is PileType.Draw or PileType.Discard or PileType.Exhaust or PileType.Deck;
        bool unpowered = sel.Prefs?.UnpoweredPreviews ?? false;
        int firstDedup = -1, lastDedup = -1;
        for (int i = 0; i < options.Count; i++)
        {
            CardModel c = options[i];
            int link = _cardRow.TryGetValue(c, out int lr) ? lr : -1;
            int linkZone = link >= 0 ? _w.Get(T.Cards, link, CardC.Zone) : Zone.None;
            int pos;
            int count = 1;
            if (fromPile)
            {
                pos = -1;
                count = sel.OptionCounts != null && i < sel.OptionCounts.Count ? sel.OptionCounts[i] : 1;
            }
            else if (linkZone == Zone.Hand)
            {
                pos = _w.Get(T.Cards, link, CardC.Pos);
            }
            else
            {
                pos = i; // generated / offered candidates keep the game's order (schema 5.2.4)
            }
            PileType? pileType = sel.Pile ?? ZonePile(linkZone);
            int aux = pileType.HasValue && pileType.Value != PileType.None ? Enum(Ev.PileType, (int)pileType.Value) : 0;
            // sel_count (schema 7.2, CARDS col 24): picks per OPTION as the frame counts them. For a merged class the picked
            // instances are other objects than Options[i], so identity with Picked would undercount (contract C2).
            int picked = i < sel.OptionPicked.Length ? sel.OptionPicked[i] : CountPickedByIdentity(sel, c);
            int r = EncodeCardRow(c, Zone.Choice, seat, pos, count, CardPreviewMode.Normal, unpowered, link, aux, picked);
            _choiceRow[i] = r;
            if (pos == -1)
            {
                if (firstDedup < 0)
                {
                    firstDedup = r;
                }
                lastDedup = r;
            }
        }
        if (firstDedup >= 0)
        {
            CheckAscending(firstDedup, lastDedup);
        }
        bool upgrade = sel.Kind is CardSelectKind.HandUpgrade or CardSelectKind.DeckUpgrade;
        if (upgrade && Options.UpgradePreviews)
        {
            for (int i = 0; i < options.Count; i++)
            {
                if (options[i].IsUpgradable)
                {
                    EncodeUpgradePreview(options[i], _choiceRow[i], seat);
                }
            }
        }
    }

    /// <summary>For decisions whose builder did not fill CardSelectContext.OptionPicked (synthetic test prompts without merged classes).</summary>
    private static int CountPickedByIdentity(CardSelectContext sel, CardModel option)
    {
        int picked = 0;
        for (int k = 0; k < sel.Picked.Count; k++)
        {
            if (ReferenceEquals(sel.Picked[k], option))
            {
                picked++;
            }
        }
        return picked;
    }

    private static PileType? ZonePile(int zone) => zone switch
    {
        Zone.Hand => PileType.Hand,
        Zone.Draw => PileType.Draw,
        Zone.Discard => PileType.Discard,
        Zone.Exhaust => PileType.Exhaust,
        Zone.Play => PileType.Play,
        Zone.Deck => PileType.Deck,
        _ => null,
    };

    /// <summary>Deduplicated CHOICE rows must be in canonical content-key order (they come from the CARD_SELECT owner's class order).</summary>
    private void CheckAscending(int firstRow, int lastRow)
    {
        int n = lastRow - firstRow + 1;
        if (n < 2)
        {
            return;
        }
        EnsureKeyScratch(n);
        int pos = 0;
        for (int i = 0; i < n; i++)
        {
            _keyOff[i] = pos;
            pos = WriteKey(firstRow + i, pos);
            _keyLen[i] = pos - _keyOff[i];
        }
        for (int i = 1; i < n; i++)
        {
            if (CompareKeys(i - 1, i) >= 0)
            {
                Stats.ChoiceOrderViolations++;
                return;
            }
        }
    }

    /// <summary>
    /// UPGRADE preview (schema 5.2.4; Nodes.Cards/NUpgradePreview.cs:93-99): CardScope.CloneCard(card) ->
    /// UpgradeInternal() -> UpgradePreviewType = Deck/Combat -> preview vars in CardPreviewMode.Upgrade; the clone is
    /// removed from the card scope again (CombatState/RunState.RemoveCard) before the encoder returns.
    /// </summary>
    private void EncodeUpgradePreview(CardModel card, int linkRow, int seat)
    {
        ICardScope? scope = card.CardScope;
        if (scope == null)
        {
            return;
        }
        bool combat = card.Pile?.IsCombatPile ?? false;
        CardModel clone = scope.CloneCard(card);
        try
        {
            clone.UpgradeInternal();
            clone.UpgradePreviewType = combat ? CardUpgradePreviewType.Combat : CardUpgradePreviewType.Deck;
            EncodeCardRow(clone, Zone.Preview, seat, -1, 1, CardPreviewMode.Upgrade, unpowered: false, linkRow, aux: -1, selCount: 0);
            _hasUpgradePreview = true;
            Stats.UpgradePreviews++;
        }
        finally
        {
            scope.RemoveCard(clone);
        }
    }

    // =====================================================================================================
    //  One CARDS row + its VARS (schema 5.2.2, 5.3.2)
    // =====================================================================================================

    private int EncodeCardRow(CardModel c, int zone, int seat, int pos, int count, CardPreviewMode mode, bool unpowered, int linkRow, int aux, int selCount)
    {
        int r = _w.AddRow(T.Cards);
        Span<int> row = _w.Row(T.Cards, r);
        row[CardC.Zone] = zone;
        row[CardC.Seat] = seat;
        row[CardC.Pos] = pos;
        row[CardC.Count] = count;
        row[CardC.CardId] = _v.Model("card", c);
        row[CardC.UpgradeLevel] = c.CurrentUpgradeLevel;
        CardEnergyCost ec = c.EnergyCost;
        // NCard.UpdateEnergyCostVisuals shows "X" or GetWithModifiers(All) (Nodes.Cards/NCard.cs:680-690); HookCache = the same
        // computation with the frame's hook listeners enumerated once (exactness tested against the game call)
        row[CardC.EnergyCost] = ec.CostsX ? 0 : _hooks.EnergyCost(c);
        int starCost = c.HasStarCostX ? 0 : _hooks.StarCost(c);
        int cf = 0;
        if (ec.CostsX)
        {
            cf |= CostFlag.CostsX;
        }
        if (ec.WasJustUpgraded)
        {
            cf |= CostFlag.CostJustUpgraded;
        }
        if (c.HasStarCostX)
        {
            cf |= CostFlag.StarX;
        }
        if (c.WasStarCostJustUpgraded)
        {
            cf |= CostFlag.StarJustUpgraded;
        }
        if (c.HasStarCostX || starCost >= 0)
        {
            cf |= CostFlag.StarVisible; // NCard.UpdateStarCostVisuals (NCard.cs:744-752)
        }
        row[CardC.CostFlags] = cf;
        row[CardC.StarCost] = c.HasStarCostX ? 0 : (starCost >= 0 ? starCost : -1);
        bool hand = zone == Zone.Hand;
        if (hand)
        {
            ICombatState? cs = c.CombatState;
            bool canPlay;
            UnplayableReason reason;
            if (_handKnown != null)
            {
                canPlay = _handKnown.HandPlayable[_handIndex];
                reason = _handKnown.HandReason[_handIndex];
            }
            else
            {
                canPlay = c.CanPlay(out reason, out AbstractModel? _);
            }
            // CardCostHelper.GetEnergyCostColor / GetStarCostColor (NCard.cs:724-728, :754) given this CanPlay result
            row[CardC.EnergyCostColor] = Enum(Ev.CardCostColor, (int)_hooks.EnergyColor(c, cs, canPlay, reason));
            row[CardC.StarCostColor] = Enum(Ev.CardCostColor, (int)_hooks.StarColor(c, cs, canPlay, reason));
            int hf = 0;
            if (canPlay)
            {
                hf |= HandFlag.CanPlay;
            }
            if (c.ShouldGlowGold)
            {
                hf |= HandFlag.GlowGold;
            }
            if (c.ShouldGlowRed)
            {
                hf |= HandFlag.GlowRed;
            }
            if (!canPlay && !reason.HasResourceCostReason())
            {
                hf |= HandFlag.UnplayableIcon;
            }
            row[CardC.HandFlags] = hf;
            row[CardC.UnplayableReason] = (int)reason;
        }
        row[CardC.KeywordBits] = _hooks.KeywordBits(c);
        EnchantmentModel? en = c.Enchantment;
        if (en != null)
        {
            row[CardC.EnchantId] = _v.Model("enchantment", en);
            row[CardC.EnchantAmount] = en.ShowAmount ? en.DisplayAmount : -1;
            row[CardC.EnchantStatus] = Enum(Ev.EnchantmentStatus, (int)en.Status);
        }
        else
        {
            row[CardC.EnchantAmount] = -1;
        }
        AfflictionModel? af = c.Affliction;
        if (af != null)
        {
            row[CardC.AfflictionId] = _v.Model("affliction", af);
            row[CardC.AfflictionAmount] = af.Amount;
        }
        row[CardC.ReplayCount] = c.GetEnchantedReplayCount();
        row[CardC.LinkRow] = linkRow;
        row[CardC.Aux] = zone == Zone.Choice || zone == Zone.Offer ? aux : -1;
        row[CardC.SelCount] = selCount;
        CardVars(c, r, hand, unpowered, mode);
        return r;
    }

    /// <summary>
    /// The card's printed numbers exactly as NCard.UpdateVisuals computes them (Nodes.Cards/NCard.cs:562-605):
    /// ClearPreview, UpdateDynamicVarPreview(mode, target = null) for the card's and the enchantment's DynamicVars
    /// (CardModel.UpdateDynamicVarPreview decides from the card's pile whether global hooks run). Per-target rows
    /// for hand cards (schema 5.3.2-3). Every PreviewValue / EnchantedValue written here is restored afterwards.
    /// </summary>
    private void CardVars(CardModel c, int row, bool hand, bool unpowered, CardPreviewMode mode)
    {
        DynamicVarSet set = c.DynamicVars;
        EnchantmentModel? en = c.Enchantment;
        DynamicVarSet? eset = en?.DynamicVars;
        if (set.Count == 0 && (eset == null || eset.Count == 0))
        {
            return;
        }
        int saved = Save(set, 0);
        if (eset != null)
        {
            Save(eset, saved);
        }
        try
        {
            Preview(c, set, eset, null, unpowered, mode);
            string[] printed = _tpl.PrintedVars(c);
            for (int i = 0; i < printed.Length; i++)
            {
                if (set.TryGetValue(printed[i], out DynamicVar? dv) && IsNumeric(dv))
                {
                    _w.AddVar(T.Cards, row, -1, _v.Id("var_name", printed[i]), (int)dv.PreviewValue, (int)dv.BaseValue, (int)dv.EnchantedValue, dv.WasJustUpgraded ? VarFlag.JustUpgraded : 0);
                }
            }
            if (en != null && eset != null && eset.Count > 0)
            {
                string[] ep = _tpl.PrintedVars(en);
                for (int i = 0; i < ep.Length; i++)
                {
                    if (eset.TryGetValue(ep[i], out DynamicVar? dv) && IsNumeric(dv))
                    {
                        int id = _v.Id("var_name", ep[i]);
                        if (!_w.HasVar(T.Cards, row, -1, id)) // a name the card prints itself wins (errata E-18)
                        {
                            _w.AddVar(T.Cards, row, -1, id, (int)dv.PreviewValue, (int)dv.BaseValue, (int)dv.EnchantedValue, VarFlag.Enchantment | (dv.WasJustUpgraded ? VarFlag.JustUpgraded : 0));
                        }
                    }
                }
            }
            if (hand && Options.TargetPreviews && !unpowered && printed.Length > 0 && _inCombat && AnyTargetSensitive(set, printed))
            {
                TargetPreviews(c, row, set, eset, printed, mode);
            }
        }
        finally
        {
            Restore(set, 0);
            if (eset != null)
            {
                Restore(eset, saved);
            }
        }
    }

    private static bool IsNumeric(DynamicVar dv) => LocText.IsNumeric(dv);

    private static readonly Dictionary<Type, bool> s_targetSensitive = new Dictionary<Type, bool>();

    /// <summary>
    /// True if a printed variable of the card can change with the preview target. Only these DynamicVar classes read the
    /// target in UpdateCardPreview (Localization.DynamicVars/*.cs, v0.113): DamageVar (Hook.ModifyDamage(..., target, ...)),
    /// OstyDamageVar (same), PowerVar&lt;T&gt; (Hook.ModifyPowerAmountGiven(..., target, ...)) and every CalculatedVar
    /// (Calculate(target)); no card overrides UpdateDynamicVarPreview and no DynamicVar subclass exists outside that
    /// namespace. If none of the printed variables is one of them, every targeted value equals the untargeted one and
    /// schema 5.3.2-3 writes no per-target row - the loop is skipped. (Per-target rows exist for the card's own printed
    /// variables only, never for the enchantment's, so its variables do not matter here.)
    /// </summary>
    private static bool AnyTargetSensitive(DynamicVarSet set, string[] printed)
    {
        for (int i = 0; i < printed.Length; i++)
        {
            if (set.TryGetValue(printed[i], out DynamicVar? dv) && TargetSensitive(dv.GetType()))
            {
                return true;
            }
        }
        return false;
    }

    private static bool TargetSensitive(Type t)
    {
        if (s_targetSensitive.TryGetValue(t, out bool v))
        {
            return v;
        }
        v = typeof(DamageVar).IsAssignableFrom(t) || typeof(CalculatedVar).IsAssignableFrom(t) || typeof(OstyDamageVar).IsAssignableFrom(t);
        for (Type? b = t; b != null && !v; b = b.BaseType)
        {
            v = b.IsGenericType && b.GetGenericTypeDefinition() == typeof(PowerVar<>);
        }
        // anything not known to be target-free is treated as sensitive (only costs time)
        if (!v && t.Namespace != typeof(DynamicVar).Namespace)
        {
            v = true;
        }
        s_targetSensitive[t] = v;
        return v;
    }

    private void Preview(CardModel c, DynamicVarSet set, DynamicVarSet? eset, Creature? target, bool unpowered, CardPreviewMode mode)
    {
        // c.UpdateDynamicVarPreview(mode, target, set) with the frame's hook listeners enumerated once (HookCache.Previews.cs)
        set.ClearPreview();
        if (!unpowered)
        {
            _hooks.UpdateDynamicVarPreview(c, mode, target, set);
        }
        if (eset != null)
        {
            eset.ClearPreview();
            if (!unpowered)
            {
                _hooks.UpdateDynamicVarPreview(c, mode, target, eset);
            }
        }
    }

    /// <summary>
    /// Per-target values of a hand card (schema 5.3.2-3): for every target the UI lets the card aim at (the decision's own
    /// PLAY_CARD targets; AllEnemies / RandomEnemy: CombatState.HittableEnemies), a VARS row with target_row is written for
    /// each printed variable whose value differs from the untargeted one.
    /// </summary>
    private void TargetPreviews(CardModel c, int row, DynamicVarSet set, DynamicVarSet? eset, string[] printed, CardPreviewMode mode)
    {
        _targets.Clear();
        TargetType tt = c.TargetType;
        if (tt is TargetType.AllEnemies or TargetType.RandomEnemy)
        {
            // CombatState.HittableEnemies, read from the CREATURES rows already written (hittable = Creature.IsHittable)
            List<Creature> enemies = _cs!._enemies;
            for (int i = 0; i < enemies.Count; i++)
            {
                int er = CreatureRowOf(enemies[i]);
                if (er >= 0 && _w.Get(T.Creatures, er, CreatureC.Hittable) == 1 && _w.Get(T.Creatures, er, CreatureC.Kind) == CreatureKind.Enemy)
                {
                    _targets.Add(enemies[i]);
                }
            }
        }
        else if (_d != null)
        {
            List<LegalAction> actions = _d.Actions;
            for (int i = 0; i < actions.Count; i++)
            {
                LegalAction a = actions[i];
                if (a.Verb == Verb.PlayCard && ReferenceEquals(a.Source.Object, c) && a.Target.Kind == RefKind.Creature && a.Target.Object is Creature t && !_targets.Contains(t))
                {
                    _targets.Add(t);
                }
            }
        }
        if (_targets.Count == 0)
        {
            return;
        }
        // untargeted shown values are the VARS rows just written for this card
        int nv = _w.CollectVarsOf(T.Cards, row, _varIdx);
        for (int ti = 0; ti < _targets.Count; ti++)
        {
            Creature t = _targets[ti];
            int trow = CreatureRowOf(t);
            if (trow < 0)
            {
                continue;
            }
            Preview(c, set, eset, t, unpowered: false, mode);
            for (int i = 0; i < printed.Length; i++)
            {
                if (!set.TryGetValue(printed[i], out DynamicVar? dv) || !IsNumeric(dv))
                {
                    continue;
                }
                int id = _v.Id("var_name", printed[i]);
                int shown = (int)dv.PreviewValue;
                int untargeted = int.MinValue;
                for (int k = 0; k < nv; k++)
                {
                    ReadOnlySpan<int> sv = _w.StagedVar(_varIdx[k]);
                    if (sv[VarC.VarId] == id && sv[VarC.TargetRow] == -1)
                    {
                        untargeted = sv[VarC.Shown];
                        break;
                    }
                }
                if (untargeted != int.MinValue && shown != untargeted)
                {
                    _w.AddVar(T.Cards, row, trow, id, shown, (int)dv.BaseValue, (int)dv.EnchantedValue, dv.WasJustUpgraded ? VarFlag.JustUpgraded : 0);
                    _hasTargetPreview = true;
                    Stats.TargetPreviewRows++;
                }
            }
        }
    }

    private int Save(DynamicVarSet set, int at)
    {
        int need = at + set.Count * 2;
        if (_savedVals.Length < need)
        {
            Array.Resize(ref _savedVals, Math.Max(need, _savedVals.Length * 2));
        }
        foreach (DynamicVar dv in set._vars.Values)
        {
            _savedVals[at++] = dv.EnchantedValue;
            _savedVals[at++] = dv.PreviewValue;
        }
        return at;
    }

    private void Restore(DynamicVarSet set, int at)
    {
        foreach (DynamicVar dv in set._vars.Values)
        {
            dv.EnchantedValue = _savedVals[at++];
            dv.PreviewValue = _savedVals[at++];
        }
    }
}
