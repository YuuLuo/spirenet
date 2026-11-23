using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;

namespace SpireHost.Engine;

public enum PumpStop
{
    /// <summary>The predicate supplied by the driver became true while the engine was quiescent.</summary>
    Decision,

    /// <summary>Quiescent, predicate false, no virtual timer pending and nothing arrived from another thread within the timeout.</summary>
    Stuck,
}

/// <summary>
/// Single-threaded deterministic engine pump.
///
/// The real game runs every gameplay continuation on Godot's main thread through
/// GodotSynchronizationContext (a queue drained once per frame). This class is the headless equivalent: it
/// is installed as SynchronizationContext.Current on the one engine thread, so every <c>await</c> in game
/// code (the game never uses ConfigureAwait(false)) captures it and every continuation that cannot run
/// inline is queued here and executed on the engine thread in FIFO order.
///
/// "Quiescent" = the queue is empty. Because all game continuations go through this queue, a quiescent
/// engine can only move again when (a) the driver injects input, (b) a virtual timer fires, or (c) another
/// thread posts (real timers / thread pool) - (c) is counted and attributed, and must stay at zero on the
/// gameplay path.
/// </summary>
public sealed class EnginePump : SynchronizationContext
{
    private readonly Queue<WorkItem> _local = new Queue<WorkItem>(256);
    private readonly ConcurrentQueue<WorkItem> _foreign = new ConcurrentQueue<WorkItem>();
    private readonly SemaphoreSlim _foreignSignal = new SemaphoreSlim(0);
    private readonly int _engineThreadId;

    public new static EnginePump? Current { get; private set; }

    public VirtualClock Clock { get; } = new VirtualClock();

    // ---- statistics (engine thread only, except ForeignPosts) ----
    public long LocalPosts;
    public long ForeignPosts;
    public long Executed;
    public long VirtualTimerFires;

    /// <summary>Times the pump had to block in real time waiting for another thread. Must be 0 on the gameplay path.</summary>
    public long ForeignWaits;
    public long MaxQueueDepth;
    public readonly ConcurrentDictionary<string, int> ForeignPostOrigins = new ConcurrentDictionary<string, int>();

    /// <summary>Diagnostic: record who posts continuations from the engine thread (i.e. which awaits do not complete inline).</summary>
    public bool TraceLocalPosts;
    public readonly ConcurrentDictionary<string, int> LocalPostOrigins = new ConcurrentDictionary<string, int>();

    /// <summary>When true, the first stack frames of every post from a non-engine thread are recorded.</summary>
    public bool TraceForeignPosts = true;

    /// <summary>
    /// Strict mode (H1 run driver, DECISIONS-H1 A.2): a post from any thread other than the engine thread is a
    /// fault. It is counted, attributed (ForeignPostOrigins) and REJECTED: the callback is never executed, so no
    /// game code can run at a wall-clock-dependent point. The driver aborts the episode when it sees the counter
    /// move. Non-strict (H0 harness, pump-off comparisons): foreign posts are queued and executed.
    /// </summary>
    public bool Strict;

    /// <summary>Foreign posts that were rejected in strict mode.</summary>
    public long ForeignPostsRejected;

    /// <summary>Callbacks currently queued on the engine thread (engine thread only).</summary>
    public int QueueCount => _local.Count;

    /// <summary>True when nothing is queued (local queue empty and no accepted foreign post waiting).</summary>
    public bool IsQueueEmpty => _local.Count == 0 && _foreign.IsEmpty;

    public int EngineThreadId => _engineThreadId;

    public bool OnEngineThread => Environment.CurrentManagedThreadId == _engineThreadId;

    private readonly struct WorkItem
    {
        public readonly SendOrPostCallback Callback;
        public readonly object? State;

        public WorkItem(SendOrPostCallback callback, object? state)
        {
            Callback = callback;
            State = state;
        }
    }

    private EnginePump()
    {
        _engineThreadId = Environment.CurrentManagedThreadId;
    }

    /// <summary>Makes the calling thread the engine thread.</summary>
    public static EnginePump InstallOnCurrentThread()
    {
        if (Current != null)
        {
            throw new InvalidOperationException("EnginePump is already installed.");
        }
        EnginePump pump = new EnginePump();
        SetSynchronizationContext(pump);
        Current = pump;
        return pump;
    }

    public static void Uninstall()
    {
        SetSynchronizationContext(null);
        Current = null;
    }

    public override SynchronizationContext CreateCopy() => this;

    public override void Post(SendOrPostCallback d, object? state)
    {
        if (Environment.CurrentManagedThreadId == _engineThreadId)
        {
            _local.Enqueue(new WorkItem(d, state));
            LocalPosts++;
            if (TraceLocalPosts)
            {
                string origin = DescribeOrigin(new StackTrace(1, false));
                LocalPostOrigins.AddOrUpdate(origin, 1, (_, c) => c + 1);
            }
            if (_local.Count > MaxQueueDepth)
            {
                MaxQueueDepth = _local.Count;
            }
            return;
        }
        Interlocked.Increment(ref ForeignPosts);
        if (TraceForeignPosts)
        {
            string origin = DescribeOrigin(new StackTrace(1, false));
            ForeignPostOrigins.AddOrUpdate(origin, 1, (_, c) => c + 1);
        }
        if (Strict)
        {
            Interlocked.Increment(ref ForeignPostsRejected);
            return;
        }
        _foreign.Enqueue(new WorkItem(d, state));
        _foreignSignal.Release();
    }

    public override void Send(SendOrPostCallback d, object? state)
    {
        if (Environment.CurrentManagedThreadId == _engineThreadId)
        {
            d(state);
            return;
        }
        using ManualResetEventSlim done = new ManualResetEventSlim(false);
        Exception? error = null;
        Post(_ =>
        {
            try
            {
                d(state);
            }
            catch (Exception e)
            {
                error = e;
            }
            finally
            {
                done.Set();
            }
        }, null);
        done.Wait();
        if (error != null)
        {
            throw new InvalidOperationException("EnginePump.Send callback failed", error);
        }
    }

    /// <summary>Runs queued continuations until the queue is empty. Returns the number executed.</summary>
    public int RunUntilQuiescent()
    {
        AssertEngineThread();
        int n = 0;
        while (true)
        {
            while (_foreign.TryDequeue(out WorkItem f))
            {
                _foreignSignal.Wait(0);
                _local.Enqueue(f);
            }
            if (!_local.TryDequeue(out WorkItem item))
            {
                break;
            }
            item.Callback(item.State);
            n++;
        }
        Executed += n;
        return n;
    }

    /// <summary>
    /// Bounded variant for the strict driver: runs queued continuations until the queue is empty or
    /// <paramref name="maxCallbacks"/> were executed. Returns false when the bound was hit (livelock guard).
    /// </summary>
    public bool TryRunUntilQuiescent(long maxCallbacks, out long executed)
    {
        AssertEngineThread();
        long n = 0;
        bool drained = true;
        while (true)
        {
            while (_foreign.TryDequeue(out WorkItem f))
            {
                _foreignSignal.Wait(0);
                _local.Enqueue(f);
            }
            if (!_local.TryDequeue(out WorkItem item))
            {
                break;
            }
            if (n >= maxCallbacks)
            {
                _local.Enqueue(item); // keep it; the episode is going to be aborted and cleaned up
                drained = false;
                break;
            }
            item.Callback(item.State);
            n++;
        }
        Executed += n;
        executed = n;
        return drained;
    }

    /// <summary>Drops everything that is still queued (after an aborted episode was cleaned up). Returns the number dropped.</summary>
    public int DiscardQueued()
    {
        AssertEngineThread();
        int n = _local.Count;
        _local.Clear();
        while (_foreign.TryDequeue(out _))
        {
            _foreignSignal.Wait(0);
            n++;
        }
        return n;
    }

    /// <summary>
    /// Why the last PumpUntil / DrainUntil returned Stuck. PumpUntil: "no_progress_timeout" or "virtual_time_budget";
    /// DrainUntil: "quiescent", "virtual_time_budget" or "pump_callbacks".
    /// </summary>
    public string? LastStuckReason { get; private set; }

    /// <summary>
    /// Set by the run driver for the duration of an episode. While it is set <see cref="DrainUntil"/> throws: only the
    /// driver core pumps during an episode (DRIVER-API.md section 3; verifier finding R-10: the rule was only written down).
    /// </summary>
    public bool EpisodeRunning;

    /// <summary>
    /// Upper bound on virtual time that one PumpUntil call may consume. A poll loop that can never succeed
    /// (e.g. ActionExecutor.WaitForUnpause while nothing will ever unpause the executor) would otherwise spin
    /// through virtual timers forever; with real timers the same situation is a silent hang.
    /// </summary>
    public TimeSpan VirtualTimeBudget = TimeSpan.FromSeconds(120);

    /// <summary>
    /// Pump until quiescent, then: stop if <paramref name="atDecisionPoint"/> holds; otherwise fire the next
    /// virtual timer (bounded by <see cref="VirtualTimeBudget"/>); otherwise wait (real time, bounded) for a
    /// post from another thread; otherwise Stuck.
    /// </summary>
    public PumpStop PumpUntil(Func<bool> atDecisionPoint, int stuckTimeoutMs = 5000)
    {
        AssertEngineThread();
        long virtualStart = Clock.NowTicks;
        while (true)
        {
            RunUntilQuiescent();
            if (atDecisionPoint())
            {
                return PumpStop.Decision;
            }
            if (Clock.Pending > 0)
            {
                if (Clock.NowTicks - virtualStart > VirtualTimeBudget.Ticks)
                {
                    LastStuckReason = "virtual_time_budget";
                    return PumpStop.Stuck;
                }
                Clock.FireNext();
                VirtualTimerFires++;
                continue;
            }
            // Nothing runnable on the engine thread. The only thing that can still move the game is a post from
            // another thread (a real timer or a thread-pool completion); wait for one, bounded.
            ForeignWaits++;
            if (!_foreignSignal.Wait(stuckTimeoutMs) && _foreign.IsEmpty)
            {
                LastStuckReason = "no_progress_timeout";
                return PumpStop.Stuck;
            }
        }
    }

    /// <summary>
    /// For verification tools that drive the engine themselves OUTSIDE a RunDriver episode (the .mcr replay harness;
    /// DRIVER-API.md section 3). Runs queued continuations to quiescence; then returns Decision if
    /// <paramref name="predicate"/> holds, else fires the next virtual timer and repeats. Unlike
    /// <see cref="PumpUntil"/> it NEVER waits in real time for another thread: a quiescent engine without the
    /// predicate is Stuck. <see cref="LastStuckReason"/>: "quiescent" (nothing left to run), "virtual_time_budget",
    /// "pump_callbacks" (more than <paramref name="maxCallbacks"/> continuations: a livelock).
    /// </summary>
    public PumpStop DrainUntil(Func<bool> predicate, long maxCallbacks, TimeSpan virtualBudget)
    {
        AssertEngineThread();
        if (EpisodeRunning)
        {
            throw new InvalidOperationException("EnginePump.DrainUntil was called while a RunDriver episode is running: only the driver core pumps during an episode.");
        }
        long start = Clock.NowTicks;
        long budget = maxCallbacks;
        while (true)
        {
            if (!TryRunUntilQuiescent(budget, out long n))
            {
                LastStuckReason = "pump_callbacks";
                return PumpStop.Stuck;
            }
            budget -= n;
            if (predicate())
            {
                return PumpStop.Decision;
            }
            if (Clock.Pending > 0)
            {
                if (Clock.NowTicks - start > virtualBudget.Ticks)
                {
                    LastStuckReason = "virtual_time_budget";
                    return PumpStop.Stuck;
                }
                Clock.FireNext();
                VirtualTimerFires++;
                continue;
            }
            LastStuckReason = "quiescent";
            return PumpStop.Stuck;
        }
    }

    private void AssertEngineThread()
    {
        if (Environment.CurrentManagedThreadId != _engineThreadId)
        {
            throw new InvalidOperationException("EnginePump may only be driven from the engine thread.");
        }
    }

    private static string DescribeOrigin(StackTrace st)
    {
        // First few frames that are not Task/await infrastructure: enough to tell a Timer from a Task.Run.
        List<string> parts = new List<string>(4);
        for (int i = 0; i < st.FrameCount && parts.Count < 6; i++)
        {
            System.Reflection.MethodBase? m = st.GetFrame(i)?.GetMethod();
            if (m == null)
            {
                continue;
            }
            string type = m.DeclaringType?.FullName ?? "?";
            if (type.StartsWith("SpireHost.Engine.EnginePump", StringComparison.Ordinal)
                || type.StartsWith("System.Runtime.CompilerServices", StringComparison.Ordinal)
                || type.StartsWith("System.Threading.Tasks.Task", StringComparison.Ordinal)
                || type.StartsWith("System.Threading.Tasks.AwaitTaskContinuation", StringComparison.Ordinal)
                || type.StartsWith("System.Threading.Tasks.SynchronizationContextAwaitTaskContinuation", StringComparison.Ordinal)
                || type.StartsWith("System.Threading.ExecutionContext", StringComparison.Ordinal))
            {
                continue;
            }
            parts.Add(type + "." + m.Name);
        }
        return string.Join(" <- ", parts);
    }
}
