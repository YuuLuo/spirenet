using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;
using SpireHost.Obs;

namespace SpireHost.Search;

/// <summary>
/// Entry points of the search build:
///   SpireHost --serve --search [--allow-debug-options] ...   the serve protocol plus FORK (notes/search/protocol.md)
///   SpireHost --search-fork-test --mode fidelity|bench ...  TEST TOOL: fork fidelity / leak test / bench (Tests/ForkTests.cs)
/// A server without --search never reaches this class.
/// </summary>
public static class SearchCommands
{
    /// <summary>What --serve --search accepts on top of the production allow-list (ObsCommands.ServeOptionNames, unchanged).</summary>
    public static readonly string[] ExtraServeOptions = { "search" };

    /// <summary>The pre-boot check of a --serve --search command line (usage errors), or null.</summary>
    public static string? ServeArgumentError(Args a)
    {
        string? stray = a.SwitchValueError("search");
        if (stray != null)
        {
            return stray;
        }
        if (!a.Has("search"))
        {
            return "--search takes no value (--search false is not accepted: leave the switch out)";
        }
        return ObsCommands.ServeArgumentError(a, ExtraServeOptions);
    }

    /// <summary>The runtime condition of a --serve --search process (Linux, W^X off, tiered compilation off), or null. Exit 3 when not met.</summary>
    public static string? RuntimeError() => ForkPrimitive.RuntimeError();

    public static int Serve(HostSession session, Args a)
    {
        string? bad = ServeArgumentError(a);
        if (bad != null)
        {
            Console.Error.WriteLine("fatal: " + bad);
            return 1;
        }
        string? env = RuntimeError();
        if (env != null)
        {
            Console.Error.WriteLine("fatal: --search: " + env);
            return 3;
        }
        RegisterGameWarmUps();
        ForkPrimitive.InitRoot();
        bool allowDebug = a.Has("allow-debug-options");
        SearchExtension ext = new SearchExtension(allowDebug, a.Has("verbose"));
        Console.Error.WriteLine("[search] --search: FORK enabled (protocol v" + SearchWire.Version + ", root pid " + ForkPrimitive.RootPid + ", mode EXACT " + (allowDebug ? "allowed (debug)" : "refused") + ")");
        int code = ObsCommands.Serve(session, a, _ => ext, ExtraServeOptions);
        ForkPrimitive.ReapNonBlocking();
        Console.Error.WriteLine("[search] serve done: " + ext.ForkCommands + " FORK commands, " + ext.Forks + " children forked, " + ForkPrimitive.Reaped + " reaped, " + ForkPrimitive.AbnormalExits + " abnormal exits, " + ForkPrimitive.SleepWaits + " pre-fork yields");
        return code;
    }

    /// <summary>Game methods that only a fork child calls (the resampler), compiled once in the root.</summary>
    public static void RegisterGameWarmUps()
    {
        ForkPrimitive.RegisterWarmUp(() =>
        {
            Prepare(typeof(RunRngSet).GetMethod(nameof(RunRngSet.MockRng))!);
            Prepare(typeof(NetFullCombatState.CardState).GetMethod(nameof(NetFullCombatState.CardState.From))!);
            Prepare(typeof(NetFullCombatState.CardState).GetMethod(nameof(NetFullCombatState.CardState.Serialize))!);
            Prepare(typeof(PacketWriter).GetMethod(nameof(PacketWriter.ZeroByteRemainder))!);
            MethodInfo write = typeof(PacketWriter).GetMethods().Single(m => m.Name == nameof(PacketWriter.Write) && m.IsGenericMethodDefinition && m.GetParameters().Length == 1);
            RuntimeHelpers.PrepareMethod(write.MethodHandle, new[] { typeof(NetFullCombatState.CardState).TypeHandle });
            foreach (string name in new[] { nameof(ListExtensions.UnstableShuffle), nameof(ListExtensions.StableShuffle) })
            {
                MethodInfo g = typeof(ListExtensions).GetMethod(name)!;
                RuntimeHelpers.PrepareMethod(g.MethodHandle, new[] { typeof(CardModel).TypeHandle });
            }
            Rng r = new Rng(1UL);
            r.NextUnsignedLong();
            new List<CardModel>().UnstableShuffle(r);
        });
    }

    private static void Prepare(MethodInfo m)
    {
        RuntimeHelpers.PrepareMethod(m.MethodHandle);
    }
}
