using System;
using System.IO;
using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;

namespace SpireHost.Boot;

/// <summary>
/// Locates the game directory, pins sts2.dll by SHA256 and resolves the game's managed dependencies at
/// runtime. Nothing in this file may touch a type from sts2.dll / GodotSharp.dll / 0Harmony.dll: it runs
/// before the resolver is installed.
/// </summary>
public static class GameAssemblies
{
    /// <summary>SHA256 of the unmodified sts2.dll v0.113.1 (commit d09d48e5).</summary>
    public const string PinnedSts2Sha256 = "62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3";

    /// <summary>release_info.json of the pinned build (version and commit).</summary>
    public const string PinnedVersion = "v0.113.1";

    public const string PinnedCommit = "d09d48e5";

    /// <summary>Assemblies that must never be loaded into this process (third-party transport).</summary>
    private static readonly string[] Forbidden = { "sim", "sts2probe", "sim-ironclad-reference", "FairEngineHost" };

    public static string GameDir { get; private set; } = "";

    public static string Sts2Sha256 { get; private set; } = "";

    public static void Install(string? explicitGameDir)
    {
        GameDir = Locate(explicitGameDir);
        string sts2 = Path.Combine(GameDir, "sts2.dll");
        using (FileStream fs = File.OpenRead(sts2))
        {
            Sts2Sha256 = Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
        }
        if (Sts2Sha256 != PinnedSts2Sha256)
        {
            throw new InvalidOperationException(
                $"sts2.dll at {sts2} has SHA256 {Sts2Sha256}, expected {PinnedSts2Sha256}. Refusing to run a modified or different game assembly.");
        }
        AssemblyLoadContext.Default.Resolving += Resolve;
    }

    private static Assembly? Resolve(AssemblyLoadContext context, AssemblyName name)
    {
        string simple = name.Name ?? "";
        foreach (string f in Forbidden)
        {
            if (simple.Equals(f, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("SpireHost must never load " + simple + ".dll");
            }
        }
        string path = Path.Combine(GameDir, simple + ".dll");
        return File.Exists(path) ? context.LoadFromAssemblyPath(path) : null;
    }

    private static string Locate(string? explicitGameDir)
    {
        string? dir = explicitGameDir ?? Environment.GetEnvironmentVariable("SPIREHOST_GAME_DIR");
        if (!string.IsNullOrEmpty(dir))
        {
            dir = Path.GetFullPath(dir);
            if (!File.Exists(Path.Combine(dir, "sts2.dll")))
            {
                throw new FileNotFoundException("sts2.dll not found in game dir " + dir);
            }
            return dir;
        }
        // Default layout: <host>\SpireHost\bin\<cfg>\net9.0\SpireHost.dll  and  <host>\game\sts2.dll
        DirectoryInfo? d = new DirectoryInfo(AppContext.BaseDirectory);
        while (d != null)
        {
            string candidate = Path.Combine(d.FullName, "game");
            if (File.Exists(Path.Combine(candidate, "sts2.dll")))
            {
                return candidate;
            }
            d = d.Parent;
        }
        throw new DirectoryNotFoundException("Could not locate the game directory (pass --game-dir or set SPIREHOST_GAME_DIR).");
    }
}
