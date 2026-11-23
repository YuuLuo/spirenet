using System;
using System.Runtime.CompilerServices;
using SpireHost.Boot;

namespace SpireHost;

/// <summary>
/// Thin entry point of THE executable. Nothing here may reference a game type: the assembly resolver is
/// installed first and every command lives behind a NoInlining call so the JIT does not touch sts2.dll before that.
///
///     SpireHost --&lt;command&gt; [options]
///     SpireHost &lt;area&gt; --&lt;command&gt; [options]      area = combat | rooms | events | fidelity | obs | replay
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        // An optional leading word (no dashes) selects a command area; everything after it is "--key value".
        string? area = null;
        if (args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal))
        {
            area = args[0].ToLowerInvariant();
            args = args.AsSpan(1).ToArray();
        }
        Args a = Args.Parse(args);
        // stdout carries results / protocol only. Anything the game or its dependencies print with
        // Console.Write (e.g. the Sentry module initializer) is diverted to stderr.
        HostIo.Out = Console.Out;
        Console.SetOut(Console.Error);
        try
        {
            GameAssemblies.Install(a.Get("game-dir"));
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("SPIREHOST_FATAL " + e.Message);
            return 3;
        }
        try
        {
            return Dispatch(area, a);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("SPIREHOST_FATAL " + e);
            return 2;
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int Dispatch(string? area, Args a)
    {
        return Commands.Run(area, a);
    }
}
