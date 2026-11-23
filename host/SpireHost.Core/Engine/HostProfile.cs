using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace SpireHost.Engine;

/// <summary>
/// DIAGNOSTICS ONLY (notes/speed/host.md): per-phase CPU time of the engine thread, allocations, GC and JIT, written as one
/// JSON line per finished episode (cumulative since process start) to the file named by SPIREHOST_PROFILE. Off when the
/// variable is not set: every probe is behind the static readonly <see cref="On"/>, which the JIT folds to a constant.
/// Reads clocks and runtime counters only; never touches game state or frame bytes.
/// </summary>
public static class HostProfile
{
    public static readonly string? FilePath = Environment.GetEnvironmentVariable("SPIREHOST_PROFILE");
    public static readonly bool On = !string.IsNullOrEmpty(FilePath);

    // engine-thread phases (exclusive: see Enter)
    public const int Settle = 0, Build = 1, Apply = 2, Encode = 3, Write = 4, ReadWait = 5, Checks = 6, AskOther = 7, LoopOther = 8;
    public const int NPhases = 9;
    public static readonly string[] PhaseNames = { "settle", "build", "apply", "encode", "write", "read_wait", "checks", "ask_other", "loop_other" };

    public static readonly long[] CpuNs = new long[NPhases];
    public static readonly long[] WallTicks = new long[NPhases];
    public static readonly long[] Alloc = new long[NPhases];
    public static readonly long[] Calls = new long[NPhases];

    // encoder sections (CPU ns), the names are the encoder's
    public static readonly long[] EncCpuNs = new long[32];
    public static string[] EncNames = Array.Empty<string>();

    // per decision kind: encode CPU of the frame of that kind; apply+settle+build CPU after an answer to that kind
    public static readonly long[] KindFrames = new long[16];
    public static readonly long[] KindEncodeNs = new long[16];
    public static readonly long[] KindAfterNs = new long[16];
    public static int LastAnsweredKind = -1;

    [StructLayout(LayoutKind.Sequential)]
    private struct Timespec
    {
        public long Sec;
        public long Nsec;
    }

    [DllImport("libc.so.6", EntryPoint = "clock_gettime")]
    private static extern int clock_gettime(int clk, out Timespec ts);

    private const int ClockThreadCputimeId = 3;

    public static long ThreadCpuNs()
    {
        if (clock_gettime(ClockThreadCputimeId, out Timespec ts) != 0)
        {
            throw new InvalidOperationException("clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
        }
        return ts.Sec * 1_000_000_000L + ts.Nsec;
    }

    // exclusive accounting on the engine thread: the time since the last switch goes to the phase that was current
    private static int s_cur = LoopOther;
    private static long s_cpu, s_wall, s_alloc;
    private static bool s_started;

    /// <summary>Makes <paramref name="phase"/> current; returns the phase that was current (pass it back to restore).</summary>
    public static int Enter(int phase)
    {
        long cpu = ThreadCpuNs(), wall = Stopwatch.GetTimestamp(), alloc = GC.GetAllocatedBytesForCurrentThread();
        if (s_started)
        {
            CpuNs[s_cur] += cpu - s_cpu;
            WallTicks[s_cur] += wall - s_wall;
            Alloc[s_cur] += alloc - s_alloc;
        }
        s_started = true;
        s_cpu = cpu;
        s_wall = wall;
        s_alloc = alloc;
        int prev = s_cur;
        s_cur = phase;
        Calls[phase]++;
        return prev;
    }

    /// <summary>CPU ns of the engine thread now, as of the last <see cref="Enter"/> (for the per-kind sums).</summary>
    public static long LastCpu => s_cpu;

    private static long s_afterStart;

    /// <summary>ProtocolServer.Decide: called right after Enter(Encode) and before the frame of kind <paramref name="kind"/> is encoded.</summary>
    public static void FrameStart(int kind)
    {
        if (LastAnsweredKind >= 0)
        {
            KindAfterNs[LastAnsweredKind] += s_cpu - s_afterStart;
        }
        LastAnsweredKind = -1;
        KindFrames[kind]++;
    }

    public static void FrameEncoded(int kind, long encodeStartCpu)
    {
        KindEncodeNs[kind] += s_cpu - encodeStartCpu;
    }

    /// <summary>The policy's answer to a frame of kind <paramref name="kind"/> was read (called right after Enter of the next phase).</summary>
    public static void Answered(int kind)
    {
        LastAnsweredKind = kind;
        s_afterStart = s_cpu;
    }

    private static long s_encLap;

    public static void EncLapStart()
    {
        s_encLap = ThreadCpuNs();
    }

    public static void EncLap(int section)
    {
        long now = ThreadCpuNs();
        EncCpuNs[section] += now - s_encLap;
        s_encLap = now;
    }

    private static int s_episodes;

    public static void EpisodeEnd(long decisions)
    {
        s_episodes++;
        StringBuilder b = new StringBuilder(2048);
        b.Append("{\"episodes\":").Append(s_episodes).Append(",\"decisions_total\":").Append(decisions);
        using (Process p = Process.GetCurrentProcess())
        {
            b.Append(",\"process_cpu_ns\":").Append(p.TotalProcessorTime.Ticks * 100);
            b.Append(",\"threads\":").Append(p.Threads.Count);
        }
        b.Append(",\"engine_thread_cpu_ns\":").Append(ThreadCpuNs());
        b.Append(",\"gc_pause_ns\":").Append(GC.GetTotalPauseDuration().Ticks * 100);
        b.Append(",\"gc0\":").Append(GC.CollectionCount(0)).Append(",\"gc1\":").Append(GC.CollectionCount(1)).Append(",\"gc2\":").Append(GC.CollectionCount(2));
        b.Append(",\"alloc_bytes\":").Append(GC.GetTotalAllocatedBytes(false));
        b.Append(",\"engine_alloc_bytes\":").Append(GC.GetAllocatedBytesForCurrentThread());
        b.Append(",\"heap_bytes\":").Append(GC.GetTotalMemory(false));
        b.Append(",\"jit_ns\":").Append(System.Runtime.JitInfo.GetCompilationTime(false).Ticks * 100);
        b.Append(",\"jit_methods\":").Append(System.Runtime.JitInfo.GetCompiledMethodCount(false));
        b.Append(",\"phases\":{");
        for (int i = 0; i < NPhases; i++)
        {
            b.Append(i == 0 ? "" : ",").Append('"').Append(PhaseNames[i]).Append("\":[").Append(CpuNs[i]).Append(',')
                .Append((long)(WallTicks[i] * (1e9 / Stopwatch.Frequency))).Append(',').Append(Alloc[i]).Append(',').Append(Calls[i]).Append(']');
        }
        b.Append("},\"encoder\":{");
        for (int i = 0; i < EncNames.Length; i++)
        {
            b.Append(i == 0 ? "" : ",").Append('"').Append(EncNames[i]).Append("\":").Append(EncCpuNs[i]);
        }
        b.Append("},\"kinds\":{");
        bool first = true;
        for (int k = 0; k < KindFrames.Length; k++)
        {
            if (KindFrames[k] == 0 && KindAfterNs[k] == 0)
            {
                continue;
            }
            b.Append(first ? "" : ",").Append('"').Append(k).Append("\":[").Append(KindFrames[k]).Append(',').Append(KindEncodeNs[k]).Append(',').Append(KindAfterNs[k]).Append(']');
            first = false;
        }
        b.Append("}}\n");
        File.AppendAllText(FilePath!, b.ToString());
    }
}
