using System;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Runs;

namespace SpireHost.Replay;

/// <summary>
/// HOST-ONLY extension of the checksum for the determinism cross-check (task 4): state the game's
/// NetFullCombatState does not cover (game-state-fork.md 1.2) - RoundNumber, CurrentSide, and per monster the
/// next move id, its intent types, the state machine's StateLog length and current state id. This reads hidden
/// information (monster state machines) and therefore must never reach a policy or an observation; it is only
/// folded into the replay tool's own trace hash (DRIVER-API.md 8.4).
/// </summary>
public static class ExtendedState
{
    public static ulong Hash(RunState run)
    {
        SigFold f = SigFold.Create();
        CombatState? cs = CombatManager.Instance.DebugOnlyGetState();
        if (cs == null)
        {
            return f.Value;
        }
        f.Add((ulong)cs.RoundNumber);
        f.Add((ulong)cs.CurrentSide);
        foreach (Creature c in cs.Enemies)
        {
            f.Add((ulong)c.CurrentHp);
            MonsterModel? m = c.Monster;
            if (m == null)
            {
                continue;
            }
            MoveState next = m.NextMove;
            f.Add(Str(next?.StateId));
            if (next?.Intents != null)
            {
                foreach (AbstractIntent i in next.Intents)
                {
                    f.Add((ulong)i.IntentType);
                }
            }
            MonsterMoveStateMachine? sm = m.MoveStateMachine;
            if (sm != null)
            {
                f.Add((ulong)sm.StateLog.Count);
                f.Add(Str(sm._currentState?.Id));
                f.Add(sm._performedFirstMove ? 1UL : 0UL);
            }
        }
        return f.Value;
    }

    private static ulong Str(string? s)
    {
        ulong h = 1469598103934665603UL;
        if (s != null)
        {
            foreach (char ch in s)
            {
                h = (h ^ ch) * 1099511628211UL;
            }
        }
        return h;
    }
}
