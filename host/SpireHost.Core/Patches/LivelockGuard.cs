using System;
using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;

namespace SpireHost.Patches;

/// <summary>
/// Thrown by <see cref="LivelockGuard"/> from inside game code. The run driver turns it into Aborted(Livelock, <see cref="Detail"/>):
/// "turns_without_decision" (the turn loop spins) or "combat_work_without_decision" (one game call does not return within
/// the bound: a loop that never ends, or a chain too long to play out).
/// </summary>
public sealed class EngineLivelockException : Exception
{
    public string Detail { get; }

    public EngineLivelockException(string message) : this("turns_without_decision", message)
    {
    }

    public EngineLivelockException(string detail, string message) : base(message)
    {
        Detail = detail;
    }
}

/// <summary>
/// The pump bounds the number of queued continuations per settle (1e6), but the game's turn loop
/// (CombatManager.StartTurn -> enemy turn -> StartTurn -> ...) returns to the pump only once per round
/// (the Task.Yield at CombatManager.cs:1442); everything else completes synchronously. If no seat is able to
/// act and nobody dies - e.g. a player that is auto-readied every turn - that loop spins for minutes before
/// the pump bound is reached. In the real game this is an animated soft-lock; headlessly it looks like a hang.
///
/// The guard is a read-only prefix on CombatManager.StartTurn that counts turn starts since the last
/// decision point and throws once the count passes <see cref="Limit"/>. It never fires in a healthy episode
/// (a decision point resets the counter), does not touch game state and consumes no RNG.
///
/// SECOND BOUND (stress verifier finding F1). A loop INSIDE ONE GAME CALL passes neither a decision point, nor a turn
/// start, nor the pump: Pillage.OnPlay draws "while the drawn card is an Attack and the hand is not full"
/// (Models.Cards/Pillage.cs), Hellraiser auto-plays every drawn Strike, so the hand never fills, the discard pile is
/// reshuffled for ever, and against an enemy the Strikes cannot hurt nothing ends it. No TestMode branch is involved:
/// it is the game's own loop (in the real client an animated soft-lock). Headlessly the process hung with its memory
/// growing by about 320 MB per minute (every auto-play adds combat history) and no bound of the host could fire.
/// Every step of such a loop that does anything records a combat-history entry (CombatHistory.Add raises Changed for
/// every card drawn / played / discarded, every hit, block, power, ...; Combat.History/CombatHistory.cs:123-130). The
/// guard subscribes to that public event, counts entries since the last decision point and throws once the count
/// passes <see cref="CombatWorkLimit"/> - and at every further entry until the driver ends the episode, so that game code
/// which catches the exception cannot go on. Read-only.
///
/// Measured with the chain diagnostic of bench\v2-parity\probe: the F1 loop is a cycle of five Strikes and one Twin
/// Strike, each auto-played for 0 damage; it reaches the limit after 29.6 s (BelowNormal priority, machine shared with a
/// training run).
///
/// It is a BOUND, not a rule of the game: a chain that does end, but only after more than the limit, is cut off as
/// well. Such chains exist. Voltaic channels one Lightning per Lightning channeled in this combat
/// (Models.Cards/Voltaic.cs), so every play of Voltaic doubles the work (the card exhausts unless it is upgraded). Measured on the gate
/// fuzz (h1-v2parity): natural HP, 6,000 runs: at most 32 entries between two decision points; forced HP 99999, 2,400
/// runs: at most 710 in three of the four processes; in the fourth, ONE episode (DEFECT A10) played Voltaic six times
/// in rounds 103 to 142 of one fight for 12,399 / 24,795 / 49,587 / 99,171 / 198,339 / 396,675 entries (the last one:
/// 17 s). Its next play would have had at least 793,347 entries, the one after that at least 1,586,691: cut off. The fuzz tools report the
/// largest count (max_combat_history_entries_between_decision_points) and how many decision points passed 10,000 and
/// 100,000. An abort carries the tuple that reproduces the run; what the chain consists of is shown by the chain
/// diagnostic (bench\v2-parity\probe, --vp-chains) on that tuple, not by the abort's signature, which names the game
/// method that added the last entry (CreatureCmd.Damage for the F1 loop; half of a Voltaic chain's entries are damage
/// entries too, the other half orb-channel entries).
/// </summary>
public static class LivelockGuard
{
    /// <summary>Side-turn starts (player or enemy) allowed between two decision points.</summary>
    public static int Limit = 4000;

    /// <summary>
    /// Only the run driver arms the guard (for the duration of an episode) and resets the counter at every
    /// decision point. Unarmed (H0 combat harness, benchmarks) the prefix only tests this flag.
    /// </summary>
    public static bool Armed;

    public static int TurnStartsSinceDecision;

    public static long Fired;

    /// <summary>
    /// Combat-history entries allowed between two decision points. Measured (BelowNormal priority, machine shared with a
    /// training run): the loop of finding F1 reaches it 30 s after its last decision point (gate G7: 33 s for the whole run);
    /// the longest chain that ended in the gate fuzz had 396,675 entries and took 17 s.
    /// </summary>
    public static int CombatWorkLimit = 1_000_000;

    public static int CombatWorkSinceDecision;

    /// <summary>The largest number of combat-history entries between two decision points seen by this process (information).</summary>
    public static int MaxCombatWorkSeen;

    /// <summary>Decision points of this process that were preceded by more than 10,000 / 100,000 combat-history entries (information).</summary>
    public static long CombatWorkAbove10K;
    public static long CombatWorkAbove100K;

    public static long CombatWorkFired;

    /// <summary>Set when a bound was passed in the current episode ("turns_without_decision" / "combat_work_without_decision"); the driver's fault check reads it too.</summary>
    public static string? Tripped;

    public static bool Installed { get; private set; }

    public static void Install()
    {
        if (Installed)
        {
            return;
        }
        Installed = true;
        MethodInfo startTurn = AccessTools.Method(typeof(CombatManager), "StartTurn") ?? throw new MissingMethodException("CombatManager.StartTurn");
        PatchRegistry.Prefix("livelock-guard-turn-loop", startTurn, PatchRegistry.M(typeof(LivelockGuard), nameof(StartTurnPrefix)), PatchCategory.Instrumentation,
            "Read-only counter on CombatManager.StartTurn (CombatManager.cs:470): side-turn starts since the last decision point. Above LivelockGuard.Limit (4000) the prefix throws " +
            "EngineLivelockException, which ends the episode as Aborted(Livelock, turns_without_decision). The turn loop returns to the pump only once per round " +
            "(the Task.Yield at CombatManager.cs:1442), so the pump's own bound (1e6 continuations per settle) would let a spinning loop run for minutes. " +
            "Armed only by the run driver; never fires in a healthy episode; no game state or RNG is touched.");
        CombatManager.Instance.History.Changed += OnCombatWork;
        PatchRegistry.Note("livelock-guard-combat-work", "MegaCrit.Sts2.Core.Combat.History.CombatHistory::Changed (public event)", "event subscription", PatchCategory.Instrumentation,
            "Read-only counter of combat-history entries (CombatHistory.Add raises Changed once per entry, Combat.History/CombatHistory.cs:123-130) since the last decision point. Above " +
            "LivelockGuard.CombatWorkLimit (1,000,000) the handler throws EngineLivelockException at that entry and at every later one, which ends the episode as " +
            "Aborted(Livelock, combat_work_without_decision). Bounds a loop inside ONE game call, which passes no decision point, no turn start and no pump turn: Pillage.OnPlay " +
            "(draw while the drawn card is an Attack and the hand is not full) with Hellraiser auto-playing every drawn Strike never ends against an enemy the Strikes cannot hurt " +
            "(stress verification F1: the host hung, memory growing without bound). The game's own loop, no TestMode branch; in the real client an animated soft-lock. " +
            "A bound, not a rule: a chain that would end after more than the limit is cut off too, and such chains exist - every play of Voltaic doubles the Lightning " +
            "channeled in the combat (Models.Cards/Voltaic.cs). Gate fuzz: at most 32 entries between two decision points at natural HP (6,000 runs); 396,675 in one Voltaic play of " +
            "2,400 forced-HP runs (the sixth play of one fight, in its round 142), whose play after the next would have passed the limit. Armed only by the run driver; no game state or RNG is touched.");
    }

    private static void OnCombatWork()
    {
        if (!Armed)
        {
            return;
        }
        int n = ++CombatWorkSinceDecision;
        if (n > CombatWorkLimit)
        {
            if (Tripped == null)
            {
                CombatWorkFired++;
                Tripped = "combat_work_without_decision";
            }
            throw new EngineLivelockException("combat_work_without_decision",
                $"More than {CombatWorkLimit} combat-history entries without a decision point: one game call does not return (a loop that never ends, or a chain of cards drawn / played, orbs channeled, hits recorded that is too long to play out). The episode is cut off.");
        }
    }

    /// <summary>Called by the driver at every decision point.</summary>
    public static void Reset()
    {
        TurnStartsSinceDecision = 0;
        if (CombatWorkSinceDecision > MaxCombatWorkSeen)
        {
            MaxCombatWorkSeen = CombatWorkSinceDecision;
        }
        if (CombatWorkSinceDecision > 10_000)
        {
            CombatWorkAbove10K++;
            if (CombatWorkSinceDecision > 100_000)
            {
                CombatWorkAbove100K++;
            }
        }
        CombatWorkSinceDecision = 0;
    }

    public static void Arm()
    {
        TurnStartsSinceDecision = 0;
        CombatWorkSinceDecision = 0;
        Tripped = null;
        Armed = true;
    }

    public static void Disarm()
    {
        Armed = false;
    }

    public static void StartTurnPrefix()
    {
        if (Armed && ++TurnStartsSinceDecision > Limit)
        {
            TurnStartsSinceDecision = 0;
            Fired++;
            Tripped ??= "turns_without_decision";
            throw new EngineLivelockException($"More than {Limit} side-turn starts without a decision point: the combat turn loop is spinning with no seat able to act.");
        }
    }
}
