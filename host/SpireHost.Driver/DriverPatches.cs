using System.Collections.Generic;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Patches;

namespace SpireHost.Driver;

/// <summary>
/// The two patches the driver core owns itself (installed by HostSession in every mode; ledger host\FIDELITY.md):
///
///   enemy-turn-intent-performed   which enemies have begun their move in the current enemy turn (combat C3)
///   run-proceed-opens-map         RunManager.Proceed() opens the host's map (events R3)
///
/// Both are read-only for the game: no argument, result or game state is changed.
/// </summary>
internal static class DriverPatches
{
    private static bool _installed;

    public static void Install()
    {
        if (_installed)
        {
            return;
        }
        _installed = true;
        PatchRegistry.Prefix("enemy-turn-intent-performed", AccessTools.Method(typeof(Creature), nameof(Creature.TakeTurn)),
            PatchRegistry.M(typeof(DriverPatches), nameof(TakeTurnPrefix)), PatchCategory.Instrumentation,
            "Read-only: remembers which enemies have started their move in the current enemy turn (Creature.TakeTurn, Entities.Creatures/Creature.cs:614, called from " +
            "CombatManager.ExecuteEnemyTurn :1146 right after the UI hides that enemy's intent, :1142-1147 -> Nodes.Combat/NCreature.cs:481-496). The game keeps no logical flag " +
            "(CombatHistory.MonsterPerformedMove is written AFTER the move, Models/MonsterModel.cs:543-545). Read through RunContext.IntentPerformed by the encoder to show a prompt " +
            "raised inside the enemy turn (KnowledgeDemon) the way the UI shows it: without the intents of the enemies that already acted (DECISIONS-H1 F.6). " +
            "No argument, result or state is changed.");

        PatchRegistry.Postfix("run-proceed-opens-map", AccessTools.Method(typeof(RunManager), nameof(RunManager.Proceed)),
            PatchRegistry.M(typeof(DriverPatches), nameof(ProceedPostfix)), PatchCategory.Plumbing,
            "RunManager.Proceed (Runs/RunManager.cs:1289-1296) only talks to the UI: unless it advances a Blitz node it calls NMapScreen.Instance?.SetTravelEnabled(true) and " +
            "NMapScreen.Instance?.Open(), a no-op without the node. Postfix: in a running episode whose run is not in Blitz mode the host's stand-in for 'travel enabled, map open' " +
            "is set (RunContext.OpenMap, idempotent; reset at the next room entry). Callers: the room owners' LEAVE / PROCEED actions call the game method exactly as the UI nodes do " +
            "(NMerchantRoom.cs:183, NRestSiteRoom.cs:477, NFakeMerchant.cs:249, NEventRoom.cs:310, NCrystalSphereScreen.cs:330) and ProceedFromTerminalRewardsScreenAsync (:1266-1287). " +
            "The game's own state is not touched.");
    }

    // -----------------------------------------------------------------------------------------------------
    //  enemy-turn-intent-performed
    // -----------------------------------------------------------------------------------------------------

    private static readonly List<Creature> _acted = new List<Creature>(8);
    private static CombatState? _state;
    private static int _round = -1;

    /// <summary>Creature.TakeTurn calls seen since process start (module counter driver.take_turn_calls).</summary>
    public static long TakeTurnCalls;

    public static void TakeTurnPrefix(Creature __instance)
    {
        TakeTurnCalls++;
        if (__instance.CombatState is not CombatState state)
        {
            return;
        }
        if (!ReferenceEquals(state, _state) || state.RoundNumber != _round)
        {
            _acted.Clear();
            _state = state;
            _round = state.RoundNumber;
        }
        _acted.Add(__instance);
    }

    /// <summary>True while the enemy side's turn of this round is running and this enemy has begun its move (its intent is no longer displayed).</summary>
    public static bool IntentPerformed(Creature creature)
    {
        if (_state == null || !ReferenceEquals(creature.CombatState, _state) || _state.CurrentSide != CombatSide.Enemy || _state.RoundNumber != _round)
        {
            return false;
        }
        for (int i = 0; i < _acted.Count; i++)
        {
            if (ReferenceEquals(_acted[i], creature))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Run set-up and episode end: nothing of a previous run may be associated with a new one.</summary>
    public static void ResetIntentTracker()
    {
        _acted.Clear();
        _state = null;
        _round = -1;
    }

    // -----------------------------------------------------------------------------------------------------
    //  run-proceed-opens-map
    // -----------------------------------------------------------------------------------------------------

    /// <summary>RunManager.Proceed calls that opened the host map / were ignored (no episode, Blitz).</summary>
    public static long ProceedOpenedMap;
    public static long ProceedIgnored;

    public static void ProceedPostfix()
    {
        RunContext? ctx = HostSession.Current?.CurrentEpisode;
        if (ctx == null || ctx.Run.GameMode == GameMode.Blitz)
        {
            ProceedIgnored++;
            return;
        }
        ProceedOpenedMap++;
        ctx.OpenMap("RunManager.Proceed");
    }
}
