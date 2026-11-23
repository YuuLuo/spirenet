using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SpireHost.Engine;

/// <summary>
/// Virtual time for the engine thread. The game has exactly two <c>Task.Delay</c> sites on the gameplay
/// path in TestMode (ActionExecutor.WaitForUnpause: 50 ms poll; CombatState.GodotTimerTask: creature-spawn
/// timeout). With real timers they cost wall-clock time and complete on a timer thread. Routed here they
/// become entries in a queue ordered by (due time, creation order); the pump fires the earliest one only
/// when the engine is quiescent and not at a decision point, so their relative order is exactly the order
/// real timers would have fired in, with zero wall-clock cost and no second thread.
/// </summary>
public sealed class VirtualClock
{
    private readonly List<Entry> _timers = new List<Entry>();
    private long _seq;

    public long NowTicks { get; private set; }

    public ulong NowMs => (ulong)(NowTicks / TimeSpan.TicksPerMillisecond);

    public int Pending => _timers.Count;

    public long Created;

    private readonly struct Entry
    {
        public readonly long Due;
        public readonly long Seq;
        public readonly TaskCompletionSource Source;

        public Entry(long due, long seq, TaskCompletionSource source)
        {
            Due = due;
            Seq = seq;
            Source = source;
        }
    }

    public Task Delay(TimeSpan delay)
    {
        if (delay <= TimeSpan.Zero)
        {
            return Task.CompletedTask;
        }
        TaskCompletionSource tcs = new TaskCompletionSource();
        _timers.Add(new Entry(NowTicks + delay.Ticks, _seq++, tcs));
        Created++;
        return tcs.Task;
    }

    /// <summary>Advances virtual time to the earliest pending timer and completes it. False if none.</summary>
    public bool FireNext()
    {
        if (_timers.Count == 0)
        {
            return false;
        }
        int best = 0;
        for (int i = 1; i < _timers.Count; i++)
        {
            Entry e = _timers[i];
            Entry b = _timers[best];
            if (e.Due < b.Due || (e.Due == b.Due && e.Seq < b.Seq))
            {
                best = i;
            }
        }
        Entry fire = _timers[best];
        _timers.RemoveAt(best);
        if (fire.Due > NowTicks)
        {
            NowTicks = fire.Due;
        }
        fire.Source.TrySetResult();
        return true;
    }

    /// <summary>Drops timers that nobody is waiting on any more (between episodes).</summary>
    public void Clear()
    {
        _timers.Clear();
    }
}
