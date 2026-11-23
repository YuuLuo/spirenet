using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using SpireHost.Driver.Tools;
using SpireHost.Sim;

namespace SpireHost.Driver.Combat.Tests;

/// <summary>
/// Test data helpers. (The mid-combat scenario tools AddCard / SetResources / ApplyPower / Afflict moved to
/// SpireHost.Driver.ScenarioTools with the H1 integration, contract request C8.)
/// </summary>
public static class CombatTestTools
{
    /// <summary>The character whose card pool contains the card, or the fallback (colorless / curses / statuses).</summary>
    public static string CharacterFor(CardModel canonical, string fallback = "IRONCLAD")
    {
        foreach (CharacterModel c in ModelDb.AllCharacters)
        {
            if (c.CardPool.AllCards.Any(x => x.Id == canonical.Id))
            {
                return c.Id.Entry;
            }
        }
        return fallback;
    }

    /// <summary>Distinct canonical starter cards of the seat's character (fillers for hand / pile prompts).</summary>
    public static List<string> StarterIds(RunContext ctx, int seat = 0)
    {
        return ctx.Seats[seat].Character.StartingDeck.Select(c => c.Id.Entry).Distinct().ToList();
    }
}

/// <summary>
/// Test policy: <see cref="Prefer"/> picks an action index (or -1); otherwise COMBAT decisions end the turn
/// (so the fight lasts) or, with <see cref="RandomCombat"/>, play randomly; every other decision is random.
/// </summary>
public sealed class PreferPolicy : IPolicy
{
    private PolicyRng _rng;

    public Func<Decision, int>? Prefer;
    public bool RandomCombat;

    public PreferPolicy(ulong seed, Func<Decision, int>? prefer = null, bool randomCombat = false)
    {
        _rng = new PolicyRng(seed);
        Prefer = prefer;
        RandomCombat = randomCombat;
    }

    public int Decide(Decision decision)
    {
        if (Prefer != null)
        {
            int i = Prefer(decision);
            if (i >= 0 && i < decision.Actions.Count)
            {
                return i;
            }
        }
        if (decision.Kind == DecisionKind.Combat && !RandomCombat)
        {
            for (int i = 0; i < decision.Actions.Count; i++)
            {
                if (decision.Actions[i].Verb == Verb.EndTurn)
                {
                    return i;
                }
            }
        }
        return _rng.NextInt(decision.Actions.Count);
    }

    public int Random(int n) => _rng.NextInt(n);

    /// <summary>Index of the first PLAY_CARD of a card with one of these ids (random target), or -1.</summary>
    public int PlayOf(Decision d, ISet<string> ids)
    {
        if (d.Kind != DecisionKind.Combat)
        {
            return -1;
        }
        int first = -1;
        int count = 0;
        CardModel? card = null;
        for (int i = 0; i < d.Actions.Count; i++)
        {
            LegalAction a = d.Actions[i];
            if (a.Verb != Verb.PlayCard || !ids.Contains(a.Source.Card.Id.Entry))
            {
                continue;
            }
            if (card == null)
            {
                card = a.Source.Card;
                first = i;
            }
            if (ReferenceEquals(a.Source.Card, card))
            {
                count++;
            }
        }
        return first < 0 ? -1 : first + _rng.NextInt(count); // actions of one card are contiguous in canonical order
    }

    public int FirstVerb(Decision d, Verb verb)
    {
        for (int i = 0; i < d.Actions.Count; i++)
        {
            if (d.Actions[i].Verb == verb)
            {
                return i;
            }
        }
        return -1;
    }

    public int RandomVerb(Decision d, Verb verb)
    {
        int n = 0;
        for (int i = 0; i < d.Actions.Count; i++)
        {
            if (d.Actions[i].Verb == verb)
            {
                n++;
            }
        }
        if (n == 0)
        {
            return -1;
        }
        int pick = _rng.NextInt(n);
        for (int i = 0; i < d.Actions.Count; i++)
        {
            if (d.Actions[i].Verb == verb && pick-- == 0)
            {
                return i;
            }
        }
        return -1;
    }
}

public sealed class CaseResult
{
    public string Suite = "";
    public string Name = "";
    public bool Passed;
    public string? Failure;
    public string Outcome = "";
    public string? TruncatedBy;
    public string? Abort;
    public string? AbortMessage;
    public List<string>? AbortFrames;
    public string? Diagnostics;
    public string Seed = "";
    public string Character = "";
    public long PolicyDecisions;
    public long CombatDecisions;
    public long SelectDecisions;
    public long LegalityChecks;
    public long Plays;
    public double WallMs;
    public string? Note;
    public List<object>? Selects;

    public object ToJson(bool verbose) => new
    {
        suite = Suite, name = Name, passed = Passed, failure = Failure, outcome = Outcome, truncated_by = TruncatedBy,
        abort = Abort, abort_message = AbortMessage, abort_frames = AbortFrames, diagnostics = verbose || !Passed ? Diagnostics : null,
        seed = Seed, character = Character, policy_decisions = PolicyDecisions, combat_decisions = CombatDecisions, select_decisions = SelectDecisions,
        legality_checks = LegalityChecks, plays = Plays, wall_ms = Math.Round(WallMs, 1), note = Note, selects = verbose || !Passed ? Selects : null,
    };
}

/// <summary>Runs one test case = one scenario episode with a probe, and turns it into a <see cref="CaseResult"/>.</summary>
public static class CaseRunner
{
    public const string DefaultEncounter = "CULTISTS_NORMAL";

    public static CaseResult Run(
        HostSession session,
        string suite,
        string name,
        string character,
        Func<RunContext, Task> setup,
        PreferPolicy policy,
        CombatProbe probe,
        Func<RunContext, bool>? stopWhen = null,
        Func<EpisodeResult, CombatProbe, string?>? verify = null,
        int hp = 99999,
        int maxTurns = 30,
        long maxDecisions = 1500,
        bool lossIsFailure = false,
        string? seedLabel = null,
        Func<AbortInfo, bool>? tolerateAbort = null)
    {
        EpisodeOptions options = new EpisodeOptions { DebugForcedHp = hp, MaxCombatTurns = maxTurns, MaxPolicyDecisions = maxDecisions };
        EpisodeResult r = ScenarioRunner.Run(session, suite + ":" + name, setup, character, seed: seedLabel != null ? HostSession.SeedFromLabel(seedLabel) : null,
            policy: policy, stopWhen: stopWhen, options: options, observer: probe);
        CaseResult c = new CaseResult
        {
            Suite = suite,
            Name = name,
            Outcome = r.Outcome.ToString(),
            TruncatedBy = r.TruncatedBy,
            Abort = r.Abort?.Signature,
            AbortMessage = r.Abort?.Message,
            AbortFrames = r.Abort?.Frames,
            Diagnostics = r.Abort?.Diagnostics,
            Seed = r.Seed,
            Character = r.Character,
            PolicyDecisions = r.Counters.PolicyDecisions,
            CombatDecisions = probe.CombatDecisions,
            SelectDecisions = probe.SelectDecisions,
            LegalityChecks = probe.LegalityChecks,
            Plays = probe.Plays,
            WallMs = r.WallMs,
            Selects = probe.Selects.Select(s => s.ToJson()).ToList(),
        };
        string? failure = null;
        bool tolerated = r.Outcome == EpisodeOutcome.Aborted && r.Abort != null && tolerateAbort != null && tolerateAbort(r.Abort);
        if (r.Outcome == EpisodeOutcome.Aborted && !tolerated)
        {
            failure = "aborted: " + r.Abort?.Signature + " | " + r.Abort?.Message;
        }
        else if (r.TeardownFaults != 0 && !tolerated)
        {
            failure = "teardown fault: " + r.TeardownFault;
        }
        else if (probe.Errors.Count > 0)
        {
            failure = "probe: " + string.Join(" || ", probe.Errors.Take(3));
        }
        else if (lossIsFailure && r.Outcome == EpisodeOutcome.Loss)
        {
            failure = "the run was lost";
        }
        else if (verify != null)
        {
            failure = verify(r, probe);
        }
        c.Failure = failure;
        c.Passed = failure == null;
        return c;
    }
}
