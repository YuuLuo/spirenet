using System.IO;

namespace SpireHost.Replay;

/// <summary>
/// Default locations of the real-client ground truth, relative to the game directory the host runs on
/// (&lt;root&gt;\host\game -> &lt;root&gt;\oracle, &lt;root&gt;\notes). Every command takes an explicit --dir / --notes-dir.
/// </summary>
public static class OraclePaths
{
    private static string Root => Path.GetFullPath(Path.Combine(SpireHost.Boot.GameAssemblies.GameDir, "..", ".."));

    /// <summary>The real-client combat replay corpus (*.mcr + sidecar *.json).</summary>
    public static string Replays => Path.Combine(Root, "oracle", "replays");

    /// <summary>The golden run traces (one directory per run).</summary>
    public static string Runs => Path.Combine(Root, "oracle", "runs");

    /// <summary>The real multi-client sessions (one directory per session, c0..c3 per client; notes\oracle-mp.md).</summary>
    public static string RunsMp => Path.Combine(Root, "oracle", "runs-mp");

    /// <summary>notes\testmode-audit-*.json (the gate site lists the coverage probe instruments).</summary>
    public static string Notes => Path.Combine(Root, "notes");
}
