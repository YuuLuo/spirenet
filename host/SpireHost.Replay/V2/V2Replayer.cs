using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;

namespace SpireHost.Replay.V2;

/// <summary>
/// Policy + observer of one v2 trace comparison (see <see cref="RunTraceCompareV2"/>). Split over several files:
///   V2Replayer.cs            the record cursor, state checks, divergence bookkeeping, observer callbacks
///   V2Replayer.Rooms.cs      event / map / rewards / card reward / rest / shop / treasure / belt / crystal / confirm / bundle
///   V2Replayer.Selection.cs  selection screens (outside and inside combat)
///   V2Replayer.Combat.cs     combat actions and prompts from combat.json (+ the fight's .mcr for instance identity)
///   V2Replayer.Checksums.cs  the game's own per-action full-combat-state checksums of every fight, inside the run
///   V2Replayer.Dumps.cs      room-boundary dumps and the game's own run-save JSON
/// </summary>
public sealed partial class V2Replayer : IPolicy, IEpisodeObserver
{
    private readonly V2Trace _t;
    private readonly RunTraceCompareV2.TraceResult _r;
    private readonly RunTraceCompareV2.Options _opt;

    private int _rec;                    // next unconsumed record of decisions.jsonl
    private bool _stop;
    private string _stopWhy = "";
    private bool _recordDiverged;        // an unexplained difference was found in the decision being compared
    private RunContext? _ctx;

    public V2Replayer(V2Trace t, RunTraceCompareV2.TraceResult r, RunTraceCompareV2.Options opt)
    {
        _t = t;
        _r = r;
        _opt = opt;
        if (t.StraySelectionResults > 0)
        {
            Explain(V2Classes.SelectionStrayResult, t.StraySelectionResults);
        }
    }

    // =====================================================================================================
    //  Bookkeeping
    // =====================================================================================================

    private void Stop(string why, string? at = null)
    {
        if (_stop)
        {
            return;
        }
        _stop = true;
        _stopWhy = why;
        _r.StoppedBy = why;
        _r.StoppedAt = at;
    }

    private void Explain(string cls, int n = 1)
    {
        _r.Explained[cls] = _r.Explained.GetValueOrDefault(cls) + n;
    }

    private void Info(string key, int n = 1)
    {
        _r.Info[key] = _r.Info.GetValueOrDefault(key) + n;
    }

    /// <summary>
    /// True when the named rule may be applied (always, except under the test switch --no-explained). For the rules that excuse a
    /// recorded item, a recorded click or the early end of a trace: without the rule the caller reports a divergence. The rules
    /// that only describe how the UI's items and steps map to host actions (identical pile cards are one option, a preview
    /// confirmation is no host step, ...) are counted with <see cref="Explain"/> and cannot be switched off: the host has no
    /// action that an unmapped comparison could be made against.
    /// </summary>
    private bool Rule(string cls, int n = 1)
    {
        if (!_opt.ApplyExplainedRules)
        {
            return false;
        }
        Explain(cls, n);
        return true;
    }

    /// <summary>
    /// An unexplained difference. Soft = the recorded choice can still be applied and the state is still the recorded
    /// one (an offer / content difference): counted, the replay goes on. Hard = the replay cannot continue faithfully.
    /// </summary>
    private void Diverge(string cls, bool hard, string where, string detail, string? trace = null, string? host = null, int seq = -1)
    {
        _recordDiverged = true;
        _r.DivergenceCounts[cls] = _r.DivergenceCounts.GetValueOrDefault(cls) + 1;
        if (_r.Divergences.Count < _opt.MaxDivergences)
        {
            _r.Divergences.Add(new RunTraceCompareV2.Divergence { Class = cls, Hard = hard, Seq = seq, Where = where, Detail = detail, Trace = Short(trace), Host = Short(host) });
        }
        if (hard || _opt.StopAtFirst)
        {
            Stop("divergence:" + cls, where + ": " + detail);
        }
    }

    private static string? Short(string? s, int max = 1500) => s == null ? null : s.Length > max ? s.Substring(0, max) + "..." : s;

    private static string Js(JsonNode? n) => n?.ToJsonString() ?? "null";

    private static string? S(JsonNode? n) => n == null ? null : n.GetValueKind() == System.Text.Json.JsonValueKind.String ? n.GetValue<string>() : n.ToJsonString();

    private static int I(JsonNode? n, int fallback = -1) => n == null ? fallback : n.GetValue<int>();

    private static bool B(JsonNode? n, bool fallback = false)
    {
        if (n == null)
        {
            return fallback;
        }
        System.Text.Json.JsonValueKind k = n.GetValueKind();
        return k == System.Text.Json.JsonValueKind.True || (k != System.Text.Json.JsonValueKind.False && fallback);
    }

    private static string Kind(JsonObject rec) => rec["kind"]!.GetValue<string>();

    private string Where(JsonObject rec) => $"record {I(rec["seq"])} ({Kind(rec)}) floor {I(rec["floor"])} {S(rec["room_type"])} {S(rec["room_model"])}";

    private static string Describe(Decision d)
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.Append(d.Kind).Append(d.SubKind != 0 ? "/" + d.SubKind : "").Append(" [");
        for (int i = 0; i < d.Actions.Count && i < 60; i++)
        {
            LegalAction a = d.Actions[i];
            sb.Append(i > 0 ? ", " : "").Append(a.Verb);
            if (!a.Source.IsNone)
            {
                sb.Append('(').Append(a.Source.Kind).Append('#').Append(a.Source.Index);
                if (!a.Target.IsNone)
                {
                    sb.Append("->").Append(a.Target.Index);
                }
                if (a.Aux != 0)
                {
                    sb.Append(",aux").Append(a.Aux);
                }
                sb.Append(')');
            }
        }
        if (d.Actions.Count > 60)
        {
            sb.Append(", ... ").Append(d.Actions.Count).Append(" actions");
        }
        return sb.Append(']').ToString();
    }

    private static int Find(Decision d, Verb verb, RefKind kind = RefKind.None, int index = -1, int aux = int.MinValue)
    {
        for (int i = 0; i < d.Actions.Count; i++)
        {
            LegalAction a = d.Actions[i];
            if (a.Verb != verb)
            {
                continue;
            }
            if (kind != RefKind.None && (a.Source.Kind != kind || a.Source.Index != index))
            {
                continue;
            }
            if (kind == RefKind.None && !a.Source.IsNone && verb != Verb.UsePotion && verb != Verb.DiscardPotion)
            {
                continue;
            }
            if (aux != int.MinValue && a.Aux != aux)
            {
                continue;
            }
            return i;
        }
        return -1;
    }

    private static bool Has(Decision d, Verb verb, RefKind kind = RefKind.None, int index = -1) => Find(d, verb, kind, index) >= 0;

    private static bool IsPotionVerb(Verb v) => v == Verb.UsePotion || v == Verb.DiscardPotion;

    // =====================================================================================================
    //  IPolicy
    // =====================================================================================================

    public int Decide(Decision d)
    {
        if (_stop)
        {
            throw new PolicyStopException(_stopWhy);
        }
        RunContext ctx = d.Episode;
        _ctx = ctx;
        int index;
        try
        {
            index = DecideCore(d, ctx);
        }
        catch (PolicyStopException)
        {
            throw;
        }
        catch (Exception e)
        {
            // a bug of the comparison tool itself must never look like a host divergence or a pass
            _r.Error = "tool exception at decision " + d.Sequence + " (" + d.Kind + "): " + e.GetType().Name + ": " + e.Message + " | " + (e.StackTrace ?? "").Split('\n').FirstOrDefault()?.Trim();
            Stop("tool_error", _r.Error);
            throw new PolicyStopException("tool_error");
        }
        if (_stop || index < 0)
        {
            if (!_stop)
            {
                Stop("tool_error", "no action chosen at decision " + d.Sequence + " (" + d.Kind + ")");
            }
            throw new PolicyStopException(_stopWhy);
        }
        return index;
    }

    private Decision? _d;                // the decision being compared (context of the divergence records)

    private int DecideCore(Decision d, RunContext ctx)
    {
        _d = d;
        bool inCombat = ctx.CombatManager.IsInProgress;
        // A selection that was being answered: its per-pick state is compared when the next decision shows what the frame did.
        ResolveSelectionStep(d);
        if (_stop)
        {
            return -1;
        }
        if (_sel != null)
        {
            return SelectionStep(d);     // the same frame: the next recorded pick, or the confirmation it asks for
        }
        if (inCombat && (d.Kind == DecisionKind.Combat || d.Kind == DecisionKind.CardSelect))
        {
            return CombatDecision(d, ctx);
        }
        if (_combat != null)
        {
            // the fight is over and the run goes on: it must be the recorded win, with everything consumed
            CloseCombat(ctx, hostWon: true);
            if (_stop)
            {
                return -1;
            }
        }
        if (!inCombat && !ReconcileSilentFights(ctx, hostFightInProgress: false))
        {
            return -1;
        }
        if (!inCombat && _dumpDeferred)
        {
            // a room dump the tracer wrote after the room's fight (see CompareDump): the fight is over now
            _dumpDeferred = false;
            _dumpDeferredNow = true;
            CompareDump(ctx, "room");
            if (_stop)
            {
                return -1;
            }
        }
        return RoomDecision(d, ctx);
    }

    // =====================================================================================================
    //  The record cursor (decisions.jsonl)
    // =====================================================================================================

    /// <summary>
    /// Skips the records that describe what already happened (selection_result, crystal_result): they are compared with the
    /// record they belong to. <paramref name="attachedOnly"/>: only those the collector wrote right after the record they
    /// describe (a stray selection_result of a hand prompt that ended with its fight - validation report P2 - was written
    /// later, after the fight, and is passed over only when the next record is needed).
    /// </summary>
    private void SkipAfterTheFactRecords(bool attachedOnly = false)
    {
        while (_rec < _t.Records.Count && IsAfterTheFact(_t.Records[_rec], attachedOnly))
        {
            _rec++;
            _r.RecordsConsumed++;
        }
    }

    private static bool IsAfterTheFact(JsonObject rec, bool attachedOnly)
    {
        string k = Kind(rec);
        return (k == "selection_result" || k == "crystal_result") && (!attachedOnly || rec.ContainsKey("_attached"));
    }

    /// <summary>The next record that asks for an action, or null when the trace has none left.</summary>
    private JsonObject? PeekRecord()
    {
        SkipAfterTheFactRecords();
        return _rec < _t.Records.Count ? _t.Records[_rec] : null;
    }

    private void Consume(JsonObject rec, bool compared)
    {
        if (!ReferenceEquals(_t.Records[_rec], rec))
        {
            throw new InvalidOperationException("record cursor out of step");
        }
        _rec++;
        _r.RecordsConsumed++;
        if (compared)
        {
            _r.DecisionsCompared++;
            string k = Kind(rec);
            _r.ComparedByKind[k] = _r.ComparedByKind.GetValueOrDefault(k) + 1;
            if (!_recordDiverged)
            {
                _r.DecisionsMatched++;
            }
        }
    }

    /// <summary>
    /// The state fields every record carries (RunTracer2.Record: floor, act, room, coord, HP, gold at the moment of the
    /// decision, before its action) against the host at this decision. A difference means the run has diverged: hard.
    /// </summary>
    private bool CheckContext(JsonObject rec, RunContext ctx)
    {
        RunState rs = ctx.Run;
        Player me = ctx.LocalPlayer;
        string where = Where(rec);
        int seq = I(rec["seq"]);
        bool ok = true;
        void Field(string name, string? trace, string? host)
        {
            _r.ContentFieldsCompared++;
            if (trace != host && ok)
            {
                ok = false;
                Diverge("state." + name, true, where, $"the record's {name} differs from the host's at this decision", trace, host, seq);
            }
        }
        Field("floor", S(rec["floor"]), rs.TotalFloor.ToString());
        Field("act", S(rec["act"]), rs.CurrentActIndex.ToString());
        Field("act_id", S(rec["act_id"]), rs.Act?.Id.Entry);
        Field("act_floor", S(rec["act_floor"]), rs.ActFloor.ToString());
        Field("room_type", S(rec["room_type"]), rs.CurrentRoom?.RoomType.ToString());
        Field("room_model", S(rec["room_model"]), rs.CurrentRoom?.ModelId?.Entry);
        Field("room_count", S(rec["room_count"]), rs.CurrentRoomCount.ToString());
        Field("coord", S(rec["coord"]), rs.CurrentMapCoord is MapCoord c ? $"{c.col},{c.row}" : null);
        Field("hp", S(rec["hp"]), $"{me.Creature.CurrentHp}/{me.Creature.MaxHp}");
        Field("gold", S(rec["gold"]), me.Gold.ToString());
        return ok;
    }

    // =====================================================================================================
    //  IEpisodeObserver
    // =====================================================================================================

    public void OnEpisodeStart(RunContext ctx)
    {
        _ctx = ctx;
        SubscribeSampling(ctx);
        SubscribeChecksums(ctx);
    }

    public void OnRoomBoundary(RunContext ctx)
    {
        if (_stop)
        {
            return;
        }
        _ctx = ctx;
        try
        {
            CompareDump(ctx, _nextDump == 0 ? "start" : "room");
        }
        catch (Exception e)
        {
            _r.Error = "tool exception at room boundary " + _nextDump + ": " + e.GetType().Name + ": " + e.Message + " | " + (e.StackTrace ?? "").Split('\n').FirstOrDefault()?.Trim();
            Stop("tool_error", _r.Error);
        }
    }

    public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
    {
    }

    public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
    {
        UnsubscribeSampling(ctx);
        UnsubscribeChecksums();
        if (_stop)
        {
            return;
        }
        _ctx = ctx;
        try
        {
            if (provisional.Outcome == EpisodeOutcome.Win || provisional.Outcome == EpisodeOutcome.Loss)
            {
                TerminalState(ctx, provisional);
            }
        }
        catch (Exception e)
        {
            _r.Error = "tool exception at the terminal state: " + e.GetType().Name + ": " + e.Message + " | " + (e.StackTrace ?? "").Split('\n').FirstOrDefault()?.Trim();
            Stop("tool_error", _r.Error);
        }
    }

    public void OnEpisodeEnd(EpisodeResult result)
    {
    }

    /// <summary>The host aborted the episode: that is a divergence of its own (the real client played this trace through).</summary>
    public void HostAborted(EpisodeResult er)
    {
        if (_stop && !_stopWhy.StartsWith("trace_end", StringComparison.Ordinal))
        {
            return;
        }
        _stop = false;
        string sig = er.Abort?.Signature ?? "?";
        Diverge("host_abort", true, $"record cursor {_rec}/{_t.Records.Count}, dump {_nextDump}/{_t.Dumps.Count}", sig + " " + er.Abort?.Message, null, er.Abort?.Diagnostics);
    }

    /// <summary>
    /// The run is over in the host (run.IsGameOver). The trace must be at its end too: every record consumed, the fatal
    /// fight (if any) consumed, and the end dump - taken by the tracer after the game-over screen appeared - equal.
    /// </summary>
    private void TerminalState(RunContext ctx, EpisodeResult provisional)
    {
        string hostEnd = provisional.Outcome == EpisodeOutcome.Win ? "win" : ctx.RunManager.IsAbandoned ? "abandoned" : "death";
        if (_combat != null)
        {
            CloseCombat(ctx, hostWon: false);
            if (_stop)
            {
                return;
            }
        }
        if (!ReconcileSilentFights(ctx, hostFightInProgress: false))
        {
            return;
        }
        if (_dumpDeferred)
        {
            Diverge("dump.deferred_dump_never_reached", true, "terminal", "the run ended in the host before the trace's dump of the last fight's room could be compared", _t.Dumps[_nextDump].File, hostEnd);
            return;
        }
        SkipAfterTheFactRecords();
        if (_rec < _t.Records.Count)
        {
            JsonObject next = _t.Records[_rec];
            Diverge("end.host_run_over_but_records_remain", true, Where(next), $"the host's run is over ({hostEnd}) while the trace has {_t.Records.Count - _rec} more records", Js(next["chosen"]), hostEnd, I(next["seq"]));
            return;
        }
        if (_combatIndex < _t.Combats.Count)
        {
            Diverge("end.host_run_over_but_fights_remain", true, "terminal", $"the host's run is over ({hostEnd}) while the trace has {_t.Combats.Count - _combatIndex} more fights", _t.Combats[_combatIndex].File, hostEnd);
            return;
        }
        if (hostEnd != _t.End)
        {
            Diverge("end.kind", true, "terminal", "the host's run ended differently from the trace", _t.End, hostEnd);
            return;
        }
        if (_nextDump < _t.Dumps.Count)
        {
            CompareDump(ctx, "end-" + hostEnd);
        }
        if (_stop)
        {
            return;
        }
        if (_nextDump != _t.Dumps.Count)
        {
            Diverge("end.dumps_remain", true, "terminal", $"the trace has {_t.Dumps.Count - _nextDump} more dumps after the host's end", _t.Dumps[_nextDump].File, null);
            return;
        }
        Stop("trace_end", $"host terminal {provisional.Outcome}; trace end {_t.End}");
    }
}
