using System;
using System.Runtime.ExceptionServices;
using System.Threading;

namespace SpireHost.Engine;

/// <summary>
/// The dedicated engine thread (DECISIONS-H1 A.1). All game logic runs on one thread with a very large
/// stack: the pump runs continuations inline, so one settled action can be a deep call chain, and the .NET
/// runtime switches an await continuation from "inline" to "posted" when the stack runs low
/// (RuntimeHelpers.TryEnsureSufficientExecutionStack) - which would change the order of continuations.
/// A 256 MB reservation (committed lazily by the OS) keeps that branch out of reach.
/// </summary>
public static unsafe class EngineThread
{
    public const int DefaultStackBytes = 256 * 1024 * 1024;

    [ThreadStatic]
    private static byte* _stackBase;

    /// <summary>Runs <paramref name="body"/> on a new big-stack thread and blocks until it returns. Exceptions are rethrown on the caller.</summary>
    public static T Run<T>(Func<T> body, int stackBytes = DefaultStackBytes, string name = "spire-engine")
    {
        T result = default!;
        ExceptionDispatchInfo? error = null;
        Thread t = new Thread(() =>
        {
            byte marker = 0;
            _stackBase = &marker;
            try
            {
                result = body();
            }
            catch (Exception e)
            {
                error = ExceptionDispatchInfo.Capture(e);
            }
        }, stackBytes)
        {
            Name = name,
            IsBackground = false,
        };
        t.Start();
        t.Join();
        error?.Throw();
        return result;
    }

    /// <summary>Marks the calling thread as an engine thread for stack-depth accounting (used when a caller brings its own thread).</summary>
    public static void MarkStackBase()
    {
        byte marker = 0;
        _stackBase = &marker;
    }

    /// <summary>Bytes of stack in use on the calling engine thread (0 when the thread was not started through this class).</summary>
    public static long StackDepthBytes()
    {
        if (_stackBase == null)
        {
            return 0;
        }
        byte here = 0;
        return (long)(_stackBase - &here);
    }
}
