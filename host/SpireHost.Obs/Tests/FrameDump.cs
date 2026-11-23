using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SpireHost.Driver;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Protocol;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Tests;

/// <summary>Per-kind frame statistics (size and encoder cost).</summary>
public sealed class KindStats
{
    public long Frames;
    public long Bytes;
    public long MaxBytes;
    public double Us;
    public long MaxUs;
    public readonly List<int> Sizes = new List<int>();
    public readonly List<double> Costs = new List<double>();

    public void Add(int bytes, double us)
    {
        Frames++;
        Bytes += bytes;
        MaxBytes = Math.Max(MaxBytes, bytes);
        Us += us;
        MaxUs = Math.Max(MaxUs, (long)us);
        if (Sizes.Count < 200_000)
        {
            Sizes.Add(bytes);
            Costs.Add(us);
        }
    }

    public object Summary()
    {
        int[] s = Sizes.ToArray();
        Array.Sort(s);
        double[] c = Costs.ToArray();
        Array.Sort(c);
        return new
        {
            frames = Frames,
            mean_bytes = Frames == 0 ? 0 : Math.Round((double)Bytes / Frames, 1),
            p50_bytes = Pct(s, 0.50),
            p99_bytes = Pct(s, 0.99),
            max_bytes = MaxBytes,
            mean_us = Frames == 0 ? 0 : Math.Round(Us / Frames, 2),
            p50_us = Math.Round(PctD(c, 0.50), 2),
            p99_us = Math.Round(PctD(c, 0.99), 2),
            max_us = MaxUs,
        };
    }

    private static int Pct(int[] a, double q) => a.Length == 0 ? 0 : a[Math.Min(a.Length - 1, (int)(q * a.Length))];

    private static double PctD(double[] a, double q) => a.Length == 0 ? 0 : a[Math.Min(a.Length - 1, (int)(q * a.Length))];
}

/// <summary>
/// An observer that encodes every decision the driver reaches (policy-asked and auto-advanced alike, after the
/// canonical sort, before the action is applied) and the terminal state; optionally writes the frames to a file
/// ([u32 len][payload] records, readable with spire.proto.read_message). Used by --golden, --bench-encode and the
/// side-effect / determinism checks.
/// </summary>
public sealed class FrameDump : IEpisodeObserver
{
    private readonly ObsEncoder _enc;
    private readonly Stream? _out;
    private int _episode;
    private long _lastAuto;

    public readonly SortedDictionary<string, KindStats> ByKind = new SortedDictionary<string, KindStats>(StringComparer.Ordinal);
    public readonly KindStats All = new KindStats();
    public long FramesWritten;
    public long EncodeFailures;
    public string? FirstFailure;
    public bool EncodeAuto = true;
    public Func<RunContext, Decision, bool>? AfterEncode;

    public FrameDump(ObsEncoder encoder, Stream? output)
    {
        _enc = encoder;
        _out = output;
    }

    public void OnEpisodeStart(RunContext ctx)
    {
        _episode++;
        _lastAuto = 0;
    }

    public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
    {
        if (auto && !EncodeAuto)
        {
            return;
        }
        long a = ctx.Counters.AutoSingleTotal + ctx.Counters.AutoGold;
        long t0 = Stopwatch.GetTimestamp();
        _enc.EncodeDecision(decision, _episode, (int)ctx.Counters.PolicyDecisions, (int)(a - _lastAuto));
        double us = (Stopwatch.GetTimestamp() - t0) * 1e6 / Stopwatch.Frequency;
        _lastAuto = a;
        Record(decision.Kind.ToString(), us);
        AfterEncode?.Invoke(ctx, decision);
    }

    private void Record(string kind, double us)
    {
        int bytes = _enc.Writer.PayloadLength;
        if (!ByKind.TryGetValue(kind, out KindStats? k))
        {
            k = new KindStats();
            ByKind[kind] = k;
        }
        k.Add(bytes, us);
        All.Add(bytes, us);
        if (_out != null)
        {
            Span<byte> head = stackalloc byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(head, bytes);
            _out.Write(head);
            _out.Write(_enc.Writer.PayloadBuffer, 0, bytes);
            FramesWritten++;
        }
    }

    private bool _terminalEncoded;

    public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
    {
        long a = ctx.Counters.AutoSingleTotal + ctx.Counters.AutoGold;
        _terminalEncoded = false;
        try
        {
            long t0 = Stopwatch.GetTimestamp();
            _enc.EncodeTerminal(ctx, _episode, (int)ctx.Counters.PolicyDecisions, (int)(a - _lastAuto));
            _terminalUs = (Stopwatch.GetTimestamp() - t0) * 1e6 / Stopwatch.Frequency;
            _terminalEncoded = true;
        }
        catch (Exception e)
        {
            EncodeFailures++;
            FirstFailure ??= "terminal: " + e;
        }
    }

    private double _terminalUs;

    public void OnEpisodeEnd(EpisodeResult result)
    {
        if (!_terminalEncoded)
        {
            _enc.EncodeBareTerminal(_episode, (int)result.Counters.PolicyDecisions, 0, 1);
        }
        (int res, int reason) = ProtocolServer.Outcome(result);
        _enc.Writer.PatchHeader(Hd.Result, res);
        _enc.Writer.PatchGlobal(Gl.TerminalReasonIdx, reason);
        _enc.Writer.PatchGlobal(Gl.AbortReasonIdx, result.Outcome == EpisodeOutcome.Aborted ? (int)(result.Abort?.Reason ?? AbortReason.HostBug) : 0);
        _enc.Writer.PatchGlobal(Gl.PolicyDecisionsIdx, (int)result.Counters.PolicyDecisions);
        Record("Terminal", _terminalUs);
    }
}
