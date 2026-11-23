using System;
using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Logging;
using SpireHost.Boot;

namespace SpireHost.Patches;

/// <summary>
/// Headless plumbing patches on *game* code. Each one replaces a presentation / IO leaf that cannot run
/// without the engine; none may change game state or RNG consumption. Keep this list as short as possible:
/// most Godot contact is already neutralised by NullGodot + TestMode.
/// </summary>
public static class HeadlessPatches
{
    /// <summary>Patches that must exist before any game type initializer runs.</summary>
    public static void InstallEarly()
    {
        PatchRegistry.Prefix("log-printer",
            AccessTools.Method(typeof(ConsoleLogPrinter), nameof(ConsoleLogPrinter.Print)),
            PatchRegistry.M(typeof(HeadlessPatches), nameof(PrintPrefix)),
            PatchCategory.Plumbing,
            "ConsoleLogPrinter.Print writes through GD.Print/GD.PrintErr (engine). Redirected to the host's stderr sink; errors are always kept and counted.");

        Type nullPlatform = typeof(MegaCrit.Sts2.Core.Platform.Null.NullPlatformUtilStrategy);
        _nullPlatformLocalPlayerId = nullPlatform.GetField("<LocalPlayerId>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingFieldException("NullPlatformUtilStrategy.LocalPlayerId backing field");
        PatchRegistry.Prefix("null-platform-ctor",
            nullPlatform.GetConstructor(Type.EmptyTypes),
            PatchRegistry.M(typeof(HeadlessPatches), nameof(NullPlatformCtorPrefix)),
            PatchCategory.Plumbing,
            "NullPlatformUtilStrategy..ctor probes an optional mp_names.json through GodotFileIo/DirAccess (engine) and a clientId command-line override. " +
            "The body is skipped and LocalPlayerId is set to its declared default 1 (the same value the original field initializer assigns); no nickname file.");

        InstallReleaseInfo();
    }

    private static FieldInfo? _nullPlatformLocalPlayerId;
    private static object? _releaseInfo;

    private static void InstallReleaseInfo()
    {
        Type manager = typeof(MegaCrit.Sts2.Core.Debug.ReleaseInfoManager);
        MethodInfo load = manager.GetMethod("LoadConfig", BindingFlags.Instance | BindingFlags.NonPublic, null, Type.EmptyTypes, null)
            ?? throw new MissingMethodException("ReleaseInfoManager.LoadConfig");
        string json = System.IO.File.ReadAllText(System.IO.Path.Combine(GameAssemblies.GameDir, "release_info.json"));
        using (System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(json))
        {
            if (doc.RootElement.GetProperty("version").GetString() != GameAssemblies.PinnedVersion || doc.RootElement.GetProperty("commit").GetString() != GameAssemblies.PinnedCommit)
            {
                throw new InvalidOperationException("release_info.json does not describe the pinned " + GameAssemblies.PinnedVersion + " build.");
            }
        }
        _releaseInfo = System.Text.Json.JsonSerializer.Deserialize(json, load.ReturnType)
            ?? throw new InvalidOperationException("release_info.json failed to deserialize.");
        PatchRegistry.Prefix("release-info-file",
            load,
            PatchRegistry.M(typeof(HeadlessPatches), nameof(ReleaseInfoPrefix)).MakeGenericMethod(load.ReturnType),
            PatchCategory.Plumbing,
            "ReleaseInfoManager.LoadConfig locates release_info.json through OS.GetExecutablePath / Godot FileAccess (engine). " +
            "Returns the game's own release_info.json (v0.113.1, d09d48e5) read with System.IO from the game directory.");
    }

    public static bool ReleaseInfoPrefix<T>(ref T __result)
    {
        __result = (T)_releaseInfo!;
        return false;
    }

    public static bool NullPlatformCtorPrefix(object __instance)
    {
        _nullPlatformLocalPlayerId!.SetValue(__instance, 1UL);
        return false;
    }

    /// <summary>
    /// Monster presentation code that dereferences a UI singleton without the TestMode / null guard the rest of
    /// the game uses. Found by fuzzing all 85 encounters x 6 characters to a win (see notes/host-spike.md).
    /// </summary>
    public static void InstallLate()
    {
        foreach (Type t in new[] { typeof(MegaCrit.Sts2.Core.Models.Monsters.Crusher), typeof(MegaCrit.Sts2.Core.Models.Monsters.Rocket) })
        {
            PatchRegistry.Prefix("kaiser-" + t.Name.ToLowerInvariant() + "-before-death",
                AccessTools.DeclaredMethod(t, "BeforeDeath", new[] { typeof(MegaCrit.Sts2.Core.Entities.Creatures.Creature) }),
                PatchRegistry.M(typeof(HeadlessPatches), nameof(CompletedTaskPrefix)),
                PatchCategory.Plumbing,
                t.Name + ".BeforeDeath is audio/animation/music only (NAudioManager.Instance.PlayOneShot, background arm/body death animation, " +
                "NRunMusicController parameter) and dereferences NAudioManager.Instance, which is null headlessly. Skipped; returns Task.CompletedTask as the original does.");
        }
        PatchRegistry.Finalizer("soul-nexus-after-death",
            AccessTools.DeclaredMethod(typeof(MegaCrit.Sts2.Core.Models.Monsters.SoulNexus), "AfterDeath", new[] { typeof(MegaCrit.Sts2.Core.Entities.Creatures.Creature) }),
            PatchRegistry.M(typeof(HeadlessPatches), nameof(SwallowNullReference)),
            PatchCategory.Plumbing,
            "SoulNexus.AfterDeath = `Creature.Died -= AfterDeath;` then a spine animation on NCombatRoom.Instance (null headlessly; SoulNexus.cs:49-53). " +
            "The original body runs (the unsubscribe happens); the NullReferenceException of the animation line is swallowed.");
    }

    public static bool CompletedTaskPrefix(ref System.Threading.Tasks.Task __result)
    {
        __result = System.Threading.Tasks.Task.CompletedTask;
        return false;
    }

    public static Exception? SwallowNullReference(Exception? __exception)
    {
        return __exception is NullReferenceException ? null : __exception;
    }

    public static long GameErrors;
    public static string? LastGameError;

    public static bool PrintPrefix(LogLevel logLevel, string text)
    {
        if (logLevel == LogLevel.Error)
        {
            GameErrors++;
            LastGameError = text;
            if (LogBootstrap.ShowErrors)
            {
                Console.Error.WriteLine("[game/ERROR] " + text);
            }
        }
        else if (logLevel >= LogBootstrap.MinLevel)
        {
            Console.Error.WriteLine("[game/" + logLevel + "] " + text);
        }
        return false;
    }
}
