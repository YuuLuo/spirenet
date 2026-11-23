using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using SpireHost.Driver;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Tests;

/// <summary>
/// Self-test observer: at every decision (policy and auto) it
///   1. takes DeepSignature, encodes, validates the frame with FrameCheck, takes DeepSignature again (side effects),
///   2. encodes a second time and compares the bytes (determinism of encoding; HOST_US ignored),
///   3. optionally (every k-th decision) encodes synthetic CARD_SELECT decisions over the live state - deck upgrade with
///      previews, hand upgrade in combat, a deduplicated draw-pile prompt - so that these prompt shapes are checked on
///      many more states than the real prompts of a random run reach, with the same checks.
/// </summary>
public sealed class EncodeChecker : IEpisodeObserver
{
    private readonly ObsEncoder _enc;
    public readonly SortedDictionary<string, int> Problems = new SortedDictionary<string, int>(StringComparer.Ordinal);
    public readonly List<string> Examples = new List<string>();
    public readonly SortedDictionary<string, long> FramesByKind = new SortedDictionary<string, long>(StringComparer.Ordinal);
    public long Frames;
    public long SideEffects;
    public long NonDeterministic;
    public long Synthetic;
    public int SyntheticEvery = 0;

    /// <summary>Optional reference encoder (EncoderOptions.ExactGameCalls): every frame must be byte-identical to it.</summary>
    public ObsEncoder? Reference;
    public long ExactMismatches;
    public long ExactCompared;

    /// <summary>Optional extra inspection of every real decision (after the checks).</summary>
    public Action<RunContext, Decision>? Also;
    private int _episode;
    private long _n;

    public EncodeChecker(ObsEncoder enc)
    {
        _enc = enc;
    }

    private void Problem(string what, string example)
    {
        Problems[what] = Problems.GetValueOrDefault(what) + 1;
        if (Examples.Count < 20)
        {
            Examples.Add(what + ": " + example);
        }
    }

    public void OnEpisodeStart(RunContext ctx)
    {
        _episode++;
    }

    public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
    {
        Check(ctx, decision, decision.Kind.ToString());
        Also?.Invoke(ctx, decision);
        _n++;
        if (SyntheticEvery > 0 && _n % SyntheticEvery == 0)
        {
            Synthetics(ctx, decision);
        }
    }

    private void Check(RunContext ctx, Decision? d, string label)
    {
        ulong s0 = DeepSignature.Compute(ctx.Run);
        if (d != null)
        {
            _enc.EncodeDecision(d, _episode, (int)ctx.Counters.PolicyDecisions, 0);
        }
        else
        {
            _enc.EncodeTerminal(ctx, _episode, (int)ctx.Counters.PolicyDecisions, 0);
        }
        byte[] b1 = _enc.Payload.ToArray();
        ulong s1 = DeepSignature.Compute(ctx.Run);
        Frames++;
        FramesByKind[label] = FramesByKind.GetValueOrDefault(label) + 1;
        if (s0 != s1)
        {
            SideEffects++;
            Problem("side effect (" + label + ")", "episode " + _episode + " decision " + ctx.Counters.DecisionPoints);
        }
        foreach (string p in FrameCheck.Problems(b1, out _))
        {
            Problem("frame: " + p, label + " episode " + _episode);
        }
        if (d != null && ScriptTransport.Word(b1, Hd.NActions) != d.Actions.Count)
        {
            Problem("ACTIONS rows != decision actions", label);
        }
        if (Reference != null)
        {
            Reference.Options = _enc.Options.Clone();
            Reference.Options.ExactGameCalls = true;
            if (d != null)
            {
                Reference.EncodeDecision(d, _episode, (int)ctx.Counters.PolicyDecisions, 0);
            }
            else
            {
                Reference.EncodeTerminal(ctx, _episode, (int)ctx.Counters.PolicyDecisions, 0);
            }
            ExactCompared++;
            if (!SameIgnoringHostUs(b1, Reference.Payload.ToArray()))
            {
                ExactMismatches++;
                Problem("hook cache differs from the exact game calls (" + label + ")", "episode " + _episode + " decision " + ctx.Counters.DecisionPoints);
            }
        }
        if (d != null)
        {
            _enc.EncodeDecision(d, _episode, (int)ctx.Counters.PolicyDecisions, 0);
        }
        else
        {
            _enc.EncodeTerminal(ctx, _episode, (int)ctx.Counters.PolicyDecisions, 0);
        }
        byte[] b2 = _enc.Payload.ToArray();
        if (!SameIgnoringHostUs(b1, b2))
        {
            NonDeterministic++;
            Problem("encoding twice differs (" + label + ")", "episode " + _episode);
        }
        if (DeepSignature.Compute(ctx.Run) != s1)
        {
            SideEffects++;
            Problem("side effect on second encode (" + label + ")", "episode " + _episode);
        }
    }

    public static bool SameIgnoringHostUs(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }
        for (int i = 0; i < a.Length; i++)
        {
            if (i / 4 == Hd.HostUs)
            {
                continue;
            }
            if (a[i] != b[i])
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>Synthetic prompts over the live (quiescent) state; the game is not asked anything, only the encoder runs.</summary>
    private void Synthetics(RunContext ctx, Decision real)
    {
        Player p = ctx.LocalPlayer;
        // 1. deck upgrade prompt with previews (rest site SMITH / upgrade events): RunState.CloneCard path out of combat,
        //    CombatState.CloneCard path in combat (deck cards are not combat cards: Deck scope)
        //    Deck candidates are a pile: the CARD_SELECT owner merges them into canonical classes (schema 5.2.4, contract request obs-1)
        List<CardModel> deckClasses = new List<CardModel>();
        List<int> deckCounts = new List<int>();
        _enc.CanonicalClasses(ctx, p.Deck.Cards, PileType.Deck, 0, false, deckClasses, deckCounts);
        SyntheticSelect(ctx, CardSelectKind.DeckUpgrade, deckClasses, deckCounts, PileType.Deck, "synthetic:DeckUpgrade");
        PlayerCombatState? pcs = p.PlayerCombatState;
        if (ctx.CombatManager.IsInProgress && pcs != null)
        {
            // 2. hand upgrade prompt (Armaments-like): combat-scope clones of hand cards
            SyntheticSelect(ctx, CardSelectKind.HandUpgrade, new List<CardModel>(pcs.Hand.Cards), null, null, "synthetic:HandUpgrade");
            // 3. draw-pile prompt, deduplicated into classes ordered by the encoder's own canonical key
            //    (what DRIVER-API 6.4 asks of the CARD_SELECT owner; contract request obs-1)
            List<CardModel> classes = new List<CardModel>();
            List<int> counts = new List<int>();
            _enc.CanonicalClasses(ctx, pcs.DrawPile.Cards, PileType.Draw, 0, false, classes, counts);
            SyntheticSelect(ctx, CardSelectKind.CombatPile, classes, counts, PileType.Draw, "synthetic:CombatPile");
        }
    }

    private void SyntheticSelect(RunContext ctx, CardSelectKind kind, List<CardModel> options, List<int>? counts, PileType? pile, string label)
    {
        if (options.Count == 0)
        {
            return;
        }
        CardSelectContext sc = new CardSelectContext
        {
            Kind = kind,
            Player = ctx.LocalPlayer,
            Options = options,
            OptionCounts = counts,
            MinSelect = 1,
            MaxSelect = 1,
            Pile = pile,
            InCombat = ctx.CombatManager.IsInProgress,
        };
        Decision d = new Decision(DecisionKind.CardSelect, null, 0, (int)kind, sc) { Episode = ctx };
        for (int i = 0; i < options.Count; i++)
        {
            d.Add(Verb.SelectCard, 0, GameRef.OfCard(options[i], i));
        }
        Synthetic++;
        Check(ctx, d, label);
    }

    public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
    {
        if (provisional.Outcome == EpisodeOutcome.Aborted)
        {
            return; // the state of an aborted episode may be inconsistent; the server sends a bare TERMINAL frame then
        }
        Check(ctx, null, "Terminal");
    }

    public void OnEpisodeEnd(EpisodeResult result)
    {
    }
}
