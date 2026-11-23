using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Encoding;

/// <summary>
/// FORECAST (notes/forecast/contract.md; RESET flag FORECAST): the moves of the next two enemy turns after the shown one.
///
/// Owner ruling 2026-11-17: fixed rule knowledge (move patterns with their numbers) may be an input of the policy, hidden
/// state (pile order, seeds, RNG state, anything not yet rolled) may not. The walk below is
/// MonsterMoveStateMachine.FindNextMoveState (MonsterMoves.MonsterMoveStateMachine/MonsterMoveStateMachine.cs:57-82) with the
/// RNG draw of RandomBranchState.GetNextState replaced by the enumeration of its positive-weight branches and the log append
/// replaced by a private copy. It reads the machine, never writes it: no GetNextState / RollMove / SetCurrentState /
/// ForceCurrentState / OnEnterState / OnExitState and no Rng member is called (EncoderAudit allows the state machine
/// namespace only in the methods of this file). The game methods it calls are getters: MoveState.FollowUpState /
/// FollowUpStateId / MustPerformOnceBeforeTransitioning / Intents, ConditionalBranch.Evaluate (the monster's own
/// conditions on public state and its move history, contract 1.3), StateWeight.GetWeight, and the intent values INTENTS
/// reads (<see cref="IntentColumns"/>). Numbers are "now": the candidate move's intents evaluated at this instant.
/// </summary>
public sealed partial class ObsEncoder
{
    private struct ForecastCand
    {
        public MoveState Move;
        public double P;
        public bool Via;     // reached through a branch state
        public bool Stayed;  // the machine does not transition (nothing is logged)
    }

    private const int ForecastMaxDepth = 16;
    private readonly List<ForecastCand> _forecast1 = new List<ForecastCand>();
    private readonly List<ForecastCand> _forecast2 = new List<ForecastCand>();
    private readonly List<ForecastCand> _forecastStep = new List<ForecastCand>();
    private readonly List<MonsterState> _forecastLog = new List<MonsterState>();

    /// <summary>The FORECAST block of an enemy whose intents are encoded (called right after EncodeIntents): AHEAD 1 and 2.</summary>
    private void EncodeForecast(Creature enemy, int ownerRow)
    {
        Stats.ForecastBlocks++;
        MonsterModel monster = enemy.Monster!;
        MonsterMoveStateMachine? m = monster.MoveStateMachine;
        if (m == null)
        {
            Stats.ForecastUnknownNoMachine++;
            ForecastUnknownRow(ownerRow, 1, -1);
            ForecastUnknownRow(ownerRow, 2, -1);
            return;
        }
        MoveState s0 = monster.NextMove;
        if (!ReferenceEquals(m._currentState, s0))
        {
            throw new InvalidOperationException($"forecast: {monster.Id.Entry} NextMove {s0.Id} is not the machine's current state {m._currentState.Id}");
        }
        int shownOrd = ForecastMoveOrd(m, s0);
        IReadOnlyList<AbstractIntent> shown = s0.Intents;
        for (int i = 0; i < shown.Count; i++)
        {
            if (shown[i].IntentType == IntentType.Hidden)
            {
                Stats.ForecastUnknownHidden++;
                ForecastUnknownRow(ownerRow, 1, shownOrd);
                ForecastUnknownRow(ownerRow, 2, shownOrd);
                return;
            }
        }
        // S0 is performed before the next roll unless the monster was spawned in this enemy turn (Creature.TakeTurn,
        // Entities.Creatures/Creature.cs:614-623); OnMovePerformed then sets _performedFirstMove.
        bool perf0 = !(monster.SpawnedThisTurn && _cs!.CurrentSide == CombatSide.Enemy);
        bool first0 = m._performedFirstMove || perf0;

        _forecastLog.Clear();
        _forecastLog.AddRange(m.StateLog);
        _forecast1.Clear();
        if (!ForecastNext(m, s0, _forecastLog, perf0, first0, _forecast1))
        {
            Stats.ForecastUnknownNoFollowUp += 2;
            ForecastUnknownRow(ownerRow, 1, shownOrd);
            ForecastUnknownRow(ownerRow, 2, shownOrd);
            return;
        }
        ForecastCheckSum(monster, _forecast1);
        if (_forecast1.Count > SchemaX1.ForecastMaxCand)
        {
            Stats.ForecastUnknownTooMany += 2;
            ForecastUnknownRow(ownerRow, 1, shownOrd);
            ForecastUnknownRow(ownerRow, 2, shownOrd);
            return;
        }

        // t+2: the marginal over distinct moves; the log of each t+1 path is L0 + [m1] (nothing when the machine stayed)
        _forecast2.Clear();
        bool known2 = true;
        for (int c = 0; c < _forecast1.Count && known2; c++)
        {
            ForecastCand c1 = _forecast1[c];
            _forecastLog.Clear();
            _forecastLog.AddRange(m.StateLog);
            if (!c1.Stayed)
            {
                _forecastLog.Add(c1.Move);
            }
            _forecastStep.Clear();
            known2 = ForecastNext(m, c1.Move, _forecastLog, performed: true, performedFirst: true, _forecastStep);
            for (int k = 0; k < _forecastStep.Count; k++)
            {
                ForecastCand c2 = _forecastStep[k];
                ForecastAdd(_forecast2, c2.Move, c1.P * c2.P, c1.Via || c2.Via, stayed: false);
            }
        }
        if (known2)
        {
            ForecastCheckSum(monster, _forecast2);
        }

        ulong? savedNetId = LocalContext.NetId;
        try
        {
            ForecastSlot(enemy, m, s0, ownerRow, 1, _forecast1, shownOrd);
            if (!known2)
            {
                Stats.ForecastUnknownNoFollowUp++;
                ForecastUnknownRow(ownerRow, 2, shownOrd);
            }
            else if (_forecast2.Count > SchemaX1.ForecastMaxCand)
            {
                Stats.ForecastUnknownTooMany++;
                ForecastUnknownRow(ownerRow, 2, shownOrd);
            }
            else
            {
                ForecastSlot(enemy, m, s0, ownerRow, 2, _forecast2, shownOrd);
            }
        }
        finally
        {
            LocalContext.NetId = savedNetId;
        }
    }

    /// <summary>
    /// The candidates of the next roll after move <paramref name="s"/> was (or was not) performed: FindNextMoveState's
    /// stay test, then MoveState.GetNextState (MonsterMoves.MonsterMoveStateMachine/MoveState.cs:72-75) and the walk.
    /// False when the move has no follow-up state (GasBomb's EXPLODE_MOVE: the monster kills itself performing it, the game
    /// never rolls after it; GetNextState would throw): UNKNOWN reason U4 (notes/forecast/deviations.md).
    /// </summary>
    private bool ForecastNext(MonsterMoveStateMachine m, MoveState s, List<MonsterState> log, bool performed, bool performedFirst, List<ForecastCand> into)
    {
        bool canLeave = !s.MustPerformOnceBeforeTransitioning || performed;
        if (!canLeave || !performedFirst)
        {
            ForecastAdd(into, s, 1.0, via: false, stayed: true);
            return true;
        }
        string? id = s.FollowUpState?.Id ?? s.FollowUpStateId;
        if (id == null)
        {
            return false;
        }
        ForecastWalk(m, id, log, 1.0, via: false, 0, into);
        return true;
    }

    /// <summary>One step of FindNextMoveState's do-loop: a move ends the chain, a branch state is evaluated without drawing.</summary>
    private void ForecastWalk(MonsterMoveStateMachine m, string id, List<MonsterState> log, double p, bool via, int depth, List<ForecastCand> into)
    {
        if (depth >= ForecastMaxDepth)
        {
            throw new InvalidOperationException("forecast: branch chain longer than " + ForecastMaxDepth + " at " + id);
        }
        if (!string.IsNullOrEmpty(id) && !m.States.ContainsKey(id))
        {
            throw new InvalidOperationException("forecast: no valid state found: " + id);
        }
        MonsterState x = string.IsNullOrEmpty(id) ? m._initialState : m.States[id];
        switch (x)
        {
            case MoveState move:
                ForecastAdd(into, move, p, via, stayed: false);
                return;
            case ConditionalBranchState cond:
                // ConditionalBranchState.GetNextState: the first branch whose condition holds now
                foreach (ConditionalBranchState.ConditionalBranch b in cond.States)
                {
                    if (b.Evaluate() > 0f)
                    {
                        ForecastWalk(m, b.id, log, p, true, depth + 1, into);
                        return;
                    }
                }
                throw new InvalidOperationException("forecast: no true branch in " + cond.Id);
            case RandomBranchState rnd:
            {
                // RandomBranchState.GetNextState draws NextFloat(sum) and takes the first branch that brings it to <= 0: every
                // positive weight is a candidate with probability w / sum; a zero sum draws 0 and takes branch 0
                List<RandomBranchState.StateWeight> branches = rnd.States;
                double sum = 0;
                for (int i = 0; i < branches.Count; i++)
                {
                    sum += ForecastWeight(m, branches[i], log);
                }
                if (sum == 0)
                {
                    ForecastWalk(m, branches[0].stateId, log, p, true, depth + 1, into);
                    return;
                }
                for (int i = 0; i < branches.Count; i++)
                {
                    float w = ForecastWeight(m, branches[i], log);
                    if (w > 0f)
                    {
                        ForecastWalk(m, branches[i].stateId, log, p * w / sum, true, depth + 1, into);
                    }
                }
                return;
            }
            default:
                throw new InvalidOperationException("forecast: state " + x.Id + " of type " + x.GetType().Name + " is no move / conditional / random state");
        }
    }

    /// <summary>
    /// RandomBranchState.GetStateWeight (MonsterMoves.MonsterMoveStateMachine/RandomBranchState.cs:153-187) line by line with
    /// <paramref name="log"/> in place of the machine's StateLog.
    /// </summary>
    private static float ForecastWeight(MonsterMoveStateMachine m, RandomBranchState.StateWeight sw, List<MonsterState> log)
    {
        float num = 1f;
        if (sw.repeatType.Equals(MoveRepeatType.UseOnlyOnce))
        {
            MonsterState item = m.States[sw.stateId];
            if (log.Contains(item))
            {
                num = 0f;
            }
        }
        else if (!sw.repeatType.Equals(MoveRepeatType.CanRepeatForever))
        {
            float num2 = sw.repeatType.Equals(MoveRepeatType.CannotRepeat) ? 1 : sw.maxTimes;
            num = (float)log.Count < num2 ? 1 : 0;
            int num3 = 0;
            while ((float)log.Count >= num2 && (float)num3 < num2 && log.Count - num3 > 0)
            {
                MonsterState state = m.States[sw.stateId];
                if (log[log.Count - 1 - num3] != state)
                {
                    num = 1f;
                    break;
                }
                num3++;
            }
        }
        if (sw.cooldown > 0)
        {
            // the last `cooldown` moves of the log
            int seen = 0;
            for (int i = log.Count - 1; i >= 0 && seen < sw.cooldown; i--)
            {
                if (!log[i].IsMove)
                {
                    continue;
                }
                if (log[i].Id == sw.stateId)
                {
                    return 0f;
                }
                seen++;
            }
        }
        return num * sw.GetWeight();
    }

    /// <summary>Adds a candidate, merging by MoveState reference (order of first appearance).</summary>
    private static void ForecastAdd(List<ForecastCand> into, MoveState move, double p, bool via, bool stayed)
    {
        for (int i = 0; i < into.Count; i++)
        {
            ForecastCand c = into[i];
            if (ReferenceEquals(c.Move, move))
            {
                c.P += p;
                c.Via |= via;
                c.Stayed |= stayed;
                into[i] = c;
                return;
            }
        }
        into.Add(new ForecastCand { Move = move, P = p, Via = via, Stayed = stayed });
    }

    private static void ForecastCheckSum(MonsterModel monster, List<ForecastCand> slot)
    {
        double sum = 0;
        for (int i = 0; i < slot.Count; i++)
        {
            sum += slot[i].P;
        }
        if (slot.Count == 0 || Math.Abs(sum - 1.0) > 1e-9)
        {
            throw new InvalidOperationException($"forecast: {monster.Id.Entry}: {slot.Count} candidates with probability sum {sum:R}");
        }
    }

    /// <summary>The rows of one slot: one per intent of each candidate (a placeholder for a move without intents).</summary>
    private void ForecastSlot(Creature enemy, MonsterMoveStateMachine m, MoveState s0, int ownerRow, int ahead, List<ForecastCand> slot, int shownOrd)
    {
        int n = slot.Count;
        int cls = n > 1 ? ForecastClass.Random : slot[0].Via ? ForecastClass.Conditional : ForecastClass.Fixed;
        for (int c = 0; c < n; c++)
        {
            MoveState move = slot[c].Move;
            int prob = n == 1 ? SchemaX1.ProbOne : (int)Math.Round(slot[c].P * SchemaX1.ProbOne, MidpointRounding.ToEven);
            int moveOrd = ForecastMoveOrd(m, move);
            int same = ReferenceEquals(move, s0) ? ForecastFlag.SameAsShown : 0;
            IReadOnlyList<AbstractIntent> intents = move.Intents;
            int k = 0;
            do
            {
                Span<int> row = _w.AddForecastRow();
                if (intents.Count == 0)
                {
                    row[ForecastC.OwnerRow] = ownerRow;
                    for (int s = 0; s < 4; s++)
                    {
                        row[ForecastC.DmgSeat0 + s] = -1;
                    }
                    row[ForecastC.Flags] = ForecastFlag.NoIntents | same;
                }
                else
                {
                    IntentColumns(row, ownerRow, k, intents[k], enemy);
                    row[ForecastC.Flags] |= same;
                }
                row[ForecastC.Ahead] = ahead;
                row[ForecastC.Cand] = c;
                row[ForecastC.NCand] = n;
                row[ForecastC.Class] = cls;
                row[ForecastC.Prob] = prob;
                row[ForecastC.MoveOrd] = moveOrd;
                row[ForecastC.ShownOrd] = shownOrd;
                k++;
            }
            while (k < intents.Count);
        }
    }

    private void ForecastUnknownRow(int ownerRow, int ahead, int shownOrd)
    {
        Span<int> row = _w.AddForecastRow();
        row[ForecastC.OwnerRow] = ownerRow;
        for (int s = 0; s < 4; s++)
        {
            row[ForecastC.DmgSeat0 + s] = -1;
        }
        row[ForecastC.Flags] = ForecastFlag.NoIntents;
        row[ForecastC.Ahead] = ahead;
        row[ForecastC.Class] = ForecastClass.Unknown;
        row[ForecastC.MoveOrd] = -1;
        row[ForecastC.ShownOrd] = shownOrd;
    }

    /// <summary>Position of the move's Id in the machine's States enumeration order if that entry is this move, else -1 (ad-hoc STUNNED).</summary>
    private static int ForecastMoveOrd(MonsterMoveStateMachine m, MoveState move)
    {
        int i = 0;
        foreach (KeyValuePair<string, MonsterState> e in m.States)
        {
            if (e.Key == move.Id)
            {
                return ReferenceEquals(e.Value, move) ? i : -1;
            }
            i++;
        }
        return -1;
    }
}
