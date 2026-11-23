using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using SpireHost.Driver;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Encoding;

public sealed partial class ObsEncoder
{
    private ObsEncoder? _keyEncoder;

    /// <summary>Calls / cards of <see cref="CanonicalClasses"/> in this process (module counters obs.canonical_class_*).</summary>
    public static long CanonicalClassCalls;
    public static long CanonicalClassCards;

    /// <summary>
    /// Canonical classes of selection candidates that come from a pile (DECISIONS-H1 B.4, DRIVER-API.md 6.4): cards whose
    /// CHOICE rows would encode identically (schema 5.2.3 content key: row columns + printed VARS) form one class; classes
    /// are returned in ascending key order with their sizes; the representative is the class's first instance in that
    /// order. This is the order the CARD_SELECT owner must give CardSelectContext.Options / OptionCounts so that the
    /// deduplicated CHOICE rows are canonical. This is the driver's canonical card-class provider (signature
    /// SpireHost.Driver.CanonicalCardClasses; registered by ObsModule in every mode: combat C7 = obs O1).
    /// <paramref name="unpoweredPreviews"/> = CardSelectorPrefs.UnpoweredPreviews of the prompt: the CHOICE rows are then
    /// encoded without the preview pass, and so is the key. Uses a private scratch encoder; no frame state is touched.
    /// </summary>
    public void CanonicalClasses(RunContext ctx, IReadOnlyList<CardModel> cards, PileType pile, int seat, bool unpoweredPreviews, List<CardModel> classes, List<int> counts)
    {
        classes.Clear();
        counts.Clear();
        if (cards.Count == 0)
        {
            return;
        }
        CanonicalClassCalls++;
        CanonicalClassCards += cards.Count;
        _keyEncoder ??= new ObsEncoder(_v, _tpl) { Options = new EncoderOptions { TargetPreviews = false, UpgradePreviews = false, ExactGameCalls = Options.ExactGameCalls } };
        _keyEncoder.KeyClasses(ctx, cards, pile, seat, unpoweredPreviews, classes, counts);
        if (HookCache.ExactCheck && !Options.ExactGameCalls)
        {
            // DIAGNOSTICS (SPIREHOST_EXACT_CHECK=1): the same classes from an exact-mode key encoder (the game's own methods)
            _exactKeyEncoder ??= new ObsEncoder(_v, _tpl) { Options = new EncoderOptions { TargetPreviews = false, UpgradePreviews = false, ExactGameCalls = true } };
            _exactClasses.Clear(); // KeyClasses appends
            _exactCounts.Clear();
            _exactKeyEncoder.KeyClasses(ctx, cards, pile, seat, unpoweredPreviews, _exactClasses, _exactCounts);
            ClassesCompared++;
            bool same = _exactClasses.Count == classes.Count;
            for (int i = 0; same && i < classes.Count; i++)
            {
                same = ReferenceEquals(_exactClasses[i], classes[i]) && _exactCounts[i] == counts[i];
            }
            if (!same)
            {
                ClassesMismatches++;
            }
        }
    }

    private ObsEncoder? _exactKeyEncoder;
    private readonly List<CardModel> _exactClasses = new List<CardModel>();
    private readonly List<int> _exactCounts = new List<int>();

    /// <summary>DIAGNOSTICS (SPIREHOST_EXACT_CHECK=1): canonical-class calls compared with an exact-mode key encoder, and mismatches.</summary>
    public static long ClassesCompared, ClassesMismatches;

    /// <summary>
    /// The multiset content key (schema 5.2.3: row columns + printed VARS) of every card of a seat's DRAW pile, as the
    /// encoder computes it for the pile's CARDS rows: keys[i] = key of cards[i]. Used by the search resampler
    /// (SpireHost.Search) to order a pile canonically before re-drawing it. Uses the private scratch encoder; no frame
    /// state is touched.
    /// </summary>
    public void DrawPileKeys(RunContext ctx, IReadOnlyList<CardModel> cards, int seat, List<int[]> keys)
    {
        keys.Clear();
        if (cards.Count == 0)
        {
            return;
        }
        _keyEncoder ??= new ObsEncoder(_v, _tpl) { Options = new EncoderOptions { TargetPreviews = false, UpgradePreviews = false, ExactGameCalls = Options.ExactGameCalls } };
        _keyEncoder.RowKeys(ctx, cards, seat, keys);
    }

    private void RowKeys(RunContext ctx, IReadOnlyList<CardModel> cards, int seat, List<int[]> keys)
    {
        Begin(ctx, null);
        int n = cards.Count;
        for (int i = 0; i < n; i++)
        {
            EncodeCardRow(cards[i], Zone.Draw, seat, -1, 1, CardPreviewMode.Normal, unpowered: false, linkRow: -1, aux: -1, selCount: 0);
        }
        EnsureKeyScratch(n);
        for (int i = 0; i < n; i++)
        {
            int end = WriteKey(i, 0);
            keys.Add(_keyBuf.AsSpan(0, end).ToArray());
        }
        _w.Clear();
        _hooks.End();
        _ctx = null!;
        _run = null!;
        _cs = null;
    }

    private void KeyClasses(RunContext ctx, IReadOnlyList<CardModel> cards, PileType pile, int seat, bool unpowered, List<CardModel> classes, List<int> counts)
    {
        Begin(ctx, null);
        int aux = pile == PileType.None ? 0 : Enum(Ev.PileType, (int)pile);
        int n = cards.Count;
        for (int i = 0; i < n; i++)
        {
            EncodeCardRow(cards[i], Zone.Choice, seat, -1, 1, CardPreviewMode.Normal, unpowered, linkRow: -1, aux, selCount: 0);
        }
        int groups = Canonicalize(0, n);
        // _repOf[g] = instance index of the class representative (first instance in key order), _counts[g] = class size
        for (int g = 0; g < groups; g++)
        {
            classes.Add(cards[_repOf[g]]);
            counts.Add(_counts[g]);
        }
        _w.Clear();
        _hooks.End();
        _ctx = null!;
        _run = null!;
        _cs = null;
    }
}
