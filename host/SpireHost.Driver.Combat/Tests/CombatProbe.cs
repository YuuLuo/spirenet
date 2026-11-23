using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Driver.Combat.Tests;

/// <summary>What the probe saw of one selection prompt (one CardSelectFrame).</summary>
public sealed class SelectRecord
{
    public object Frame = null!;
    public CardSelectKind Kind;
    public CardSelectEntry Entry;
    public ChoiceContextKind Context;
    public string? Source;
    public PileType? Pile;
    public bool InCombat;
    public bool OutsidePlayPhase;
    public bool DuringEnemyTurn;
    public int Min;
    public int Max;
    public int RawMin;
    public int RawMax;
    public int Options;
    public int Instances;
    public bool Merged;
    public bool Cancelable;
    public bool CanSkip;
    public bool Manual;
    public string FirstVerbs = "";
    public string Prompt = "";
    public int Decisions;
    public int Picks;
    public string End = "";           // Auto / ConfirmSelection / Skip / Cancel / "" (still open)
    public List<string> PickedIds = new List<string>();
    public List<CardModel> PickedCards = new List<CardModel>();
    public bool IntentsHiddenSeen;    // an enemy's intent was reported as performed while this prompt was open

    public object ToJson() => new
    {
        kind = Kind.ToString(), entry = Entry.ToString(), context = Context.ToString(), source = Source, pile = Pile?.ToString(),
        in_combat = InCombat, outside_play_phase = OutsidePlayPhase, during_enemy_turn = DuringEnemyTurn,
        min = Min, max = Max, raw_min = RawMin, raw_max = RawMax, options = Options, instances = Instances, merged = Merged,
        cancelable = Cancelable, can_skip = CanSkip, manual = Manual, first_verbs = FirstVerbs, prompt = Prompt, decisions = Decisions, picks = Picks, end = End,
    };
}

/// <summary>
/// Test observer. Looks at every decision the driver produces (auto-advanced ones included) and
///   - COMBAT: recomputes the legal action set with the straightforward enumeration of driver-spec-combat.md
///     8.2 (the game's own LINQ properties: HittableEnemies, PlayerCreatures ...) and compares it with what the
///     handler offered (a differential test of the allocation-free enumeration), and checks that every offered
///     play passes the engine's own gate CardModel.CanPlayTargeting;
///   - CARD_SELECT: checks the structure of the frame (ordinals, class sizes, key order, verbs against the
///     published bounds) and records one <see cref="SelectRecord"/> per prompt.
/// It only reads the game (the same queries the UI makes every frame).
/// </summary>
public sealed class CombatProbe : IEpisodeObserver
{
    public readonly List<string> Errors = new List<string>();
    public readonly List<SelectRecord> Selects = new List<SelectRecord>();
    public readonly Dictionary<string, long> VetoBy = new Dictionary<string, long>();
    public readonly Dictionary<string, long> PlayedCards = new Dictionary<string, long>();

    public long CombatDecisions;
    public long LegalityChecks;
    public long SelectDecisions;
    public long Plays;
    public long PotionUses;
    public long PotionDiscards;
    public long EndTurns;
    public long UnplayableSeen;
    public int MaxTurn;
    public bool CheckLegality = true;

    /// <summary>TEST: set by "drawperm" while it has registered the local CardClassKey as the class provider.</summary>
    public static bool LocalKeyOrder;

    /// <summary>Suite-specific hook, called after the built-in checks.</summary>
    public Action<RunContext, Decision, int, bool>? OnEach;

    private readonly HashSet<(int verb, int seat, int src, int tgt)> _expected = new HashSet<(int, int, int, int)>();
    private readonly HashSet<(int verb, int seat, int src, int tgt)> _offered = new HashSet<(int, int, int, int)>();
    private SelectRecord? _open;

    public void OnEpisodeStart(RunContext ctx)
    {
    }

    public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
    {
    }

    public void OnEpisodeEnd(EpisodeResult result)
    {
    }

    private void Error(string message)
    {
        if (Errors.Count < 20)
        {
            Errors.Add(message);
        }
    }

    public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
    {
        try
        {
            if (decision.Kind == DecisionKind.Combat)
            {
                Combat(ctx, decision, chosen);
            }
            else if (decision.Kind == DecisionKind.CardSelect)
            {
                Select(ctx, decision, chosen);
            }
            if (decision.Kind == DecisionKind.Combat || decision.Kind == DecisionKind.CardSelect)
            {
                // The observation encoder's own consistency checks on this frame (plugged in by the executable).
                string? problem = SpireHost.Driver.Tools.FrameAudit.Run(decision);
                if (problem != null)
                {
                    Error(problem);
                }
            }
            OnEach?.Invoke(ctx, decision, chosen, auto);
        }
        catch (Exception e)
        {
            Error("probe threw: " + e.GetType().Name + ": " + e.Message);
        }
    }

    // -----------------------------------------------------------------------------------------------------
    //  COMBAT
    // -----------------------------------------------------------------------------------------------------

    private void Combat(RunContext ctx, Decision d, int chosen)
    {
        CombatDecisions++;
        if (d.Context is not CombatContext info || info.State == null)
        {
            Error("COMBAT decision without a filled CombatContext");
            return;
        }
        CombatState state = info.State;
        IReadOnlyList<Creature> creatures = state.Creatures;
        if (creatures.Count != info.Creatures.Count || !creatures.SequenceEqual(info.Creatures))
        {
            Error("CombatContext.Creatures differs from CombatState.Creatures");
        }
        if (ctx.SeatCount == 1 && (d.ActingSeat != 0 || d.SeatMask != 1))
        {
            Error($"single seat: ActingSeat {d.ActingSeat} SeatMask {d.SeatMask}");
        }
        LegalAction a = d.Actions[chosen];
        switch (a.Verb)
        {
            case Verb.PlayCard:
                Plays++;
                string id = a.Source.Card.Id.Entry;
                PlayedCards[id] = PlayedCards.GetValueOrDefault(id) + 1;
                break;
            case Verb.UsePotion:
                PotionUses++;
                break;
            case Verb.DiscardPotion:
                PotionDiscards++;
                break;
            case Verb.EndTurn:
                EndTurns++;
                break;
        }
        for (int s = 0; s < ctx.SeatCount; s++)
        {
            int turn = ctx.Seats[s].PlayerCombatState?.TurnNumber ?? 0;
            if (turn > MaxTurn)
            {
                MaxTurn = turn;
            }
        }
        if (!CheckLegality)
        {
            return;
        }
        LegalityChecks++;
        _expected.Clear();
        _offered.Clear();
        CombatManager cm = ctx.CombatManager;
        for (int s = 0; s < ctx.SeatCount; s++)
        {
            Player p = ctx.Seats[s];
            PlayerCombatState? pcs = p.PlayerCombatState;
            // Reference version of the seat predicate, from public API only.
            bool ready = cm.IsInProgress && !cm.IsOverOrEnding && state.CurrentSide == CombatSide.Player && pcs != null && pcs.Phase == PlayerTurnPhase.Play
                && ctx.RunManager.ActionQueueSynchronizer.CombatState == MegaCrit.Sts2.Core.Entities.Multiplayer.ActionSynchronizerCombatState.PlayPhase
                && !ctx.RunManager.ActionExecutor.IsRunning && ctx.RunManager.ActionExecutor.CurrentlyRunningAction == null && ctx.RunManager.ActionQueueSet.IsEmpty
                && !cm.IsPlayerReadyToEndTurn(p) && p.Creature.IsAlive
                && (cm.PlayersTakingExtraTurn.Count == 0 || cm.PlayersTakingExtraTurn.Contains(p))
                && !cm.IsExecutingCardOrPotionEffect(p) && !cm.EndingPlayerTurnPhaseOne && !cm.EndingPlayerTurnPhaseTwo;
            if (ready != (info.Seats[s].State == NotReady.Ready))
            {
                Error($"seat {s}: reference predicate says ready={ready}, handler says {info.Seats[s].State}");
            }
            if (!ready)
            {
                continue;
            }
            IReadOnlyList<CardModel> hand = pcs!.Hand.Cards;
            for (int i = 0; i < hand.Count; i++)
            {
                CardModel c = hand[i];
                bool can = c.CanPlay(out UnplayableReason reason, out AbstractModel? preventer);
                if (i < info.Seats[s].HandPlayable.Count && info.Seats[s].HandPlayable[i] != can)
                {
                    Error($"HandPlayable[{i}] mismatch for {c.Id.Entry}");
                }
                if (!can)
                {
                    UnplayableSeen++;
                    if ((reason & UnplayableReason.BlockedByHook) != 0 && preventer != null)
                    {
                        string who = preventer.Id.Entry;
                        VetoBy[who] = VetoBy.GetValueOrDefault(who) + 1;
                    }
                    continue;
                }
                switch (c.TargetType)
                {
                    case TargetType.AnyEnemy:
                        foreach (Creature e in state.HittableEnemies)
                        {
                            if (c.IsValidTarget(e))
                            {
                                _expected.Add(((int)Verb.PlayCard, s, i, IndexOf(creatures, e)));
                            }
                        }
                        break;
                    case TargetType.AnyAlly:
                        foreach (Creature t in state.PlayerCreatures)
                        {
                            if (t != p.Creature && t.IsHittable && c.IsValidTarget(t))
                            {
                                _expected.Add(((int)Verb.PlayCard, s, i, IndexOf(creatures, t)));
                            }
                        }
                        break;
                    default:
                        if (c.IsValidTarget(null))
                        {
                            _expected.Add(((int)Verb.PlayCard, s, i, -1));
                        }
                        break;
                }
            }
            bool beltOpen = p.CanUseOrRemovePotions && !ctx.RunManager.IsPotionUseLocked && p.Creature.IsAlive;
            for (int i = 0; i < p.PotionSlots.Count; i++)
            {
                PotionModel? po = p.PotionSlots[i];
                if (po == null || po.IsQueued || !beltOpen)
                {
                    continue;
                }
                _expected.Add(((int)Verb.DiscardPotion, s, i, -1));
                if ((po.Usage == PotionUsage.CombatOnly || po.Usage == PotionUsage.AnyTime) && po.PassesCustomUsabilityCheck)
                {
                    foreach (Creature? t in ReferencePotionTargets(po, p, state))
                    {
                        // Execution-time rule (UsePotionAction.cs:97-105): a missing target of a single-target potion becomes the owner.
                        Creature? completed = t ?? (po.IsValidTarget(p.Creature) ? p.Creature : null);
                        if (po.IsValidTarget(completed))
                        {
                            _expected.Add(((int)Verb.UsePotion, s, i, completed == null ? -1 : IndexOf(creatures, completed)));
                        }
                    }
                }
            }
            _expected.Add(((int)Verb.EndTurn, s, -1, -1));
        }
        for (int i = 0; i < d.Actions.Count; i++)
        {
            LegalAction x = d.Actions[i];
            if (x.Verb == Verb.UndoEndTurn)
            {
                continue;
            }
            _offered.Add(((int)x.Verb, x.Seat, x.Source.Index, x.Target.Index));
            if (x.Verb == Verb.PlayCard)
            {
                CardModel card = x.Source.Card;
                Creature? target = x.Target.IsNone ? null : x.Target.Creature;
                if (!card.CanPlayTargeting(target))
                {
                    Error($"offered play fails CanPlayTargeting: {card.Id.Entry} -> {x.Target}");
                }
                if (card.Pile?.Type != PileType.Hand)
                {
                    Error($"offered play of a card that is not in hand: {card.Id.Entry}");
                }
                if (target != null && (!target.CombatId.HasValue || !ReferenceEquals(creatures[x.Target.Index], target)))
                {
                    Error($"play target ordinal / combat id wrong: {card.Id.Entry} -> {x.Target}");
                }
            }
        }
        if (!_expected.SetEquals(_offered))
        {
            List<string> missing = _expected.Except(_offered).Select(Describe).Take(4).ToList();
            List<string> extra = _offered.Except(_expected).Select(Describe).Take(4).ToList();
            Error($"legal action set differs from the reference enumeration: missing [{string.Join(", ", missing)}] extra [{string.Join(", ", extra)}]");
        }
    }

    private static string Describe((int verb, int seat, int src, int tgt) k) => $"{(Verb)k.verb}(seat {k.seat}, src {k.src}, tgt {k.tgt})";

    /// <summary>driver-spec-combat.md 3.2: the targets the real UI offers for a potion.</summary>
    private static IEnumerable<Creature?> ReferencePotionTargets(PotionModel po, Player p, CombatState state)
    {
        switch (po.TargetType)
        {
            case TargetType.AnyEnemy:
                return state.HittableEnemies.Cast<Creature?>();
            case TargetType.AnyPlayer:
                if (po.CanThrowAtAlly())
                {
                    return state.PlayerCreatures.Where(c => c.IsAlive).Cast<Creature?>();
                }
                return new Creature?[] { p.Creature };
            case TargetType.Self:
                return new Creature?[] { p.Creature };
            case TargetType.AnyAlly:
            case TargetType.TargetedNoCreature:
                return Array.Empty<Creature?>();
            default:
                return new Creature?[] { null };
        }
    }

    private static int IndexOf(IReadOnlyList<Creature> list, Creature c)
    {
        for (int i = 0; i < list.Count; i++)
        {
            if (ReferenceEquals(list[i], c))
            {
                return i;
            }
        }
        return -1;
    }

    // -----------------------------------------------------------------------------------------------------
    //  CARD_SELECT
    // -----------------------------------------------------------------------------------------------------

    private void Select(RunContext ctx, Decision d, int chosen)
    {
        SelectDecisions++;
        if (d.Context is not CardSelectContext c || d.Source is not CardSelectFrame frame)
        {
            Error("CARD_SELECT decision without CardSelectContext / CardSelectFrame");
            return;
        }
        CardSelectContext extra = c; // contract C2: the fields the local shim CardSelectExtra used to carry
        SelectRecord? r = _open != null && ReferenceEquals(_open.Frame, frame) ? _open : null;
        if (r == null)
        {
            r = new SelectRecord
            {
                Frame = frame,
                Kind = c.Kind,
                Entry = extra.Entry,
                Context = extra.ContextKind,
                Source = c.SourceModel?.Id.Entry,
                Pile = c.Pile,
                InCombat = c.InCombat,
                OutsidePlayPhase = extra.OutsidePlayPhase,
                DuringEnemyTurn = extra.DuringEnemyTurn,
                Min = c.MinSelect,
                Max = c.MaxSelect,
                RawMin = extra.RawMin,
                RawMax = extra.RawMax,
                Options = c.Options.Count,
                Instances = extra.InstanceCount,
                Merged = extra.Merged,
                Cancelable = c.Cancelable,
                CanSkip = c.CanSkip,
                Manual = c.RequireManualConfirmation,
                FirstVerbs = string.Join("+", d.Actions.Select(x => x.Verb).Distinct()),
                Prompt = extra.PromptTable + "/" + extra.PromptKey,
            };
            Selects.Add(r);
            _open = r;
            if (c.Picked.Count != 0)
            {
                Error("first decision of a prompt already has picks");
            }
            if (d.SubKind != (int)c.Kind)
            {
                Error("SubKind != CardSelectContext.Kind");
            }
            if (c.Kind == CardSelectKind.Unknown)
            {
                Error("CARD_SELECT of kind Unknown (no captured entry point)");
            }
            int total = 0;
            for (int i = 0; i < c.Options.Count; i++)
            {
                total += c.OptionCounts != null ? c.OptionCounts[i] : 1;
            }
            if (total != extra.InstanceCount || frame.RawOptions.Count != extra.InstanceCount)
            {
                Error($"class sizes sum to {total}, the game handed over {frame.RawOptions.Count} cards");
            }
            if (CardSelectFrame.MergesCandidates(frame.Prompt))
            {
                // The class order is the registered provider's (production: the encoder's content key, checked by the
                // encoder itself through the frame audit, and by "drawperm"). Only while a test has registered the
                // LOCAL key as the provider are the classes in ascending CardClassKey order.
                string? previous = null;
                for (int i = 0; LocalKeyOrder && i < c.Options.Count; i++)
                {
                    string key = CardClassKey.Of(c.Options[i]);
                    if (previous != null && string.CompareOrdinal(previous, key) >= 0)
                    {
                        Error("merged candidates are not in strictly ascending key order");
                    }
                    previous = key;
                }
                // Whatever the key: every option is the first instance of its class in the order the game handed
                // over, and no instance is offered twice.
                HashSet<CardModel> seenInstances = new HashSet<CardModel>(ReferenceEqualityComparer.Instance);
                for (int i = 0; i < c.Options.Count; i++)
                {
                    int size = c.OptionCounts != null ? c.OptionCounts[i] : 1;
                    if (!ReferenceEquals(frame.InstanceOf(i, 0), c.Options[i]))
                    {
                        Error("option " + i + " is not the first instance of its class");
                    }
                    int last = -1;
                    for (int m = 0; m < size; m++)
                    {
                        CardModel instance = frame.InstanceOf(i, m);
                        int at = -1;
                        for (int q = 0; q < frame.RawOptions.Count; q++)
                        {
                            if (ReferenceEquals(frame.RawOptions[q], instance))
                            {
                                at = q;
                                break;
                            }
                        }
                        if (at < 0 || !seenInstances.Add(instance))
                        {
                            Error("class " + i + " contains a card that is not a candidate, or one that is in two classes");
                        }
                        if (at <= last)
                        {
                            Error("the instances of class " + i + " are not in the order the game handed them over");
                        }
                        last = at;
                    }
                }
            }
            else if (extra.SortedByComparison)
            {
                // The displayed order of a simple grid with CardSelectorPrefs.Comparison (NSimpleCardSelectScreen.Create).
                List<CardModel> shown = frame.RawOptions.ToList();
                shown.Sort(frame.Prompt.Prefs.Comparison!);
                if (!c.Options.SequenceEqual(shown))
                {
                    Error("candidates of a grid with a comparison are not in the displayed (sorted) order");
                }
            }
            else if (!c.Options.SequenceEqual(frame.RawOptions))
            {
                Error("unmerged candidates are not in the order the game handed them over");
            }
            if (extra.PromptKey.Length == 0 || extra.PromptTable.Length == 0)
            {
                Error("CARD_SELECT without a prompt key");
            }
            if (c.Player != ctx.Seats[d.ActingSeat])
            {
                Error("CardSelectContext.Player is not the acting seat");
            }
        }
        r.Decisions++;
        if (ctx.CombatManager.IsInProgress)
        {
            CombatState? state = ctx.CombatManager.DebugOnlyGetState();
            if (state != null)
            {
                foreach (Creature e in state.Enemies)
                {
                    if (ctx.IntentPerformed(e))
                    {
                        r.IntentsHiddenSeen = true;
                    }
                }
            }
        }

        // ---- verbs against the published bounds ----
        int k = c.Picked.Count;
        bool hasSelect = false;
        bool hasConfirm = false;
        bool hasSkip = false;
        bool hasCancel = false;
        for (int i = 0; i < d.Actions.Count; i++)
        {
            LegalAction a = d.Actions[i];
            switch (a.Verb)
            {
                case Verb.SelectCard:
                    hasSelect = true;
                    if ((uint)a.Source.Index >= (uint)c.Options.Count || !ReferenceEquals(a.Source.Card, c.Options[a.Source.Index]))
                    {
                        Error("SELECT_CARD ordinal does not index CardSelectContext.Options");
                    }
                    break;
                case Verb.ConfirmSelection:
                    hasConfirm = true;
                    break;
                case Verb.Skip:
                    hasSkip = true;
                    break;
                case Verb.Cancel:
                    hasCancel = true;
                    break;
                default:
                    Error("unexpected verb on CARD_SELECT: " + a.Verb);
                    break;
            }
        }
        if (hasSelect != (k < c.MaxSelect))
        {
            Error($"SELECT offered={hasSelect} with {k} picked, max {c.MaxSelect}");
        }
        if (hasCancel != c.Cancelable)
        {
            Error("CANCEL offered != Cancelable");
        }
        if (hasSkip != (k == 0 && c.CanSkip))
        {
            Error($"SKIP offered={hasSkip} with {k} picked, CanSkip {c.CanSkip}");
        }
        if (hasConfirm && (k < c.MinSelect || k > c.MaxSelect || k == 0 || !c.RequireManualConfirmation))
        {
            Error($"CONFIRM offered with {k} picked, bounds [{c.MinSelect}, {c.MaxSelect}], manual {c.RequireManualConfirmation}");
        }
        if (c.Kind == CardSelectKind.ChooseACard && !frame.Prompt.CanSkip && (hasSkip || hasCancel || hasConfirm))
        {
            Error("ChooseACard without canSkip offers a way out");
        }
        if (extra.PickedOrdinals.Count != k)
        {
            Error("PickedOrdinals out of step with Picked");
        }

        LegalAction ch = d.Actions[chosen];
        if (ch.Verb == Verb.SelectCard)
        {
            r.Picks++;
            r.PickedIds.Add(ch.Source.Card.Id.Entry);
            if (k + 1 >= c.MaxSelect && !c.RequireManualConfirmation)
            {
                r.End = "Auto";
                _open = null;
            }
        }
        else
        {
            r.End = ch.Verb.ToString();
            _open = null;
        }
        if (r.End.Length > 0)
        {
            // What will be handed to the game (pick order), for suites that check the effect afterwards.
            r.PickedCards.AddRange(c.Picked);
            if (ch.Verb == Verb.SelectCard)
            {
                // The last pick is applied after this callback: resolve it the way the frame will.
                int ordinal = ch.Source.Index;
                r.PickedCards.Add(frame.InstanceOf(ordinal, extra.OptionPicked[ordinal]));
            }
            else if (ch.Verb != Verb.ConfirmSelection)
            {
                r.PickedCards.Clear();
            }
        }
    }
}
