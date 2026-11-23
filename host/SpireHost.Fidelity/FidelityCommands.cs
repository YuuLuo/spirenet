using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SpireHost;
using SpireHost.Driver;
using SpireHost.Patches;

namespace SpireHost.Fidelity;

/// <summary>
/// Commands of the fidelity module, reached as "SpireHost fidelity --<command>" (production configuration):
///   --selftest [--runs N] [--brief]     the machine-checkable invariant test (FidelitySelfTest)
///   --scan [--out FILE]                 the static scans (UI singletons, Godot natives, scheduling) over the loaded game
///   --list-patches                      every registry record (what host\FIDELITY.md must match)
///   --site-table [--out FILE]           the audit-site table (every non-presentation TestMode site -> handled by)
/// </summary>
public static class FidelityCommands
{
    public static int Run(HostSession session, Args a)
    {
        switch (a.Command)
        {
            case "selftest":
                return SelfTest.FidelitySelfTest.Run(session, a);
            case "scan":
                return Scan(session, a);
            case "site-table":
                return SiteTableCommand(a);
            case "scan-il":
                HostIo.Out.Write(Scans.StaticScans.DumpIl(a.Get("type", ""), a.Get("method", "")));
                return 0;
            case "list-patches":
                HostJson.Emit(new
                {
                    kind = "fidelity-list-patches",
                    patches = PatchRegistry.Records.Count,
                    by_category = PatchRegistry.Records.GroupBy(r => r.Category.ToString()).ToDictionary(g => g.Key, g => g.Count()),
                    records = PatchRegistry.Records,
                }, a.Get("out"));
                return 0;
            default:
                Console.Error.WriteLine("fidelity: --selftest [--runs N] | --scan [--out F] | --site-table | --list-patches");
                return 1;
        }
    }

    private static int Scan(HostSession session, Args a)
    {
        string root = a.Get("source-root", Path.GetFullPath(Path.Combine(Boot.GameAssemblies.GameDir, "..", "..", "ref", "sts2-v0113-src")));
        object result = Scans.StaticScans.Run(root, session.Options.NativeAllowList, Patches.FidelityPatches.StubbedGodotMethods());
        HostJson.Emit(new { scan = result, native_probes = Scans.NativeProbes.Run(session.Options.NativeAllowList) }, a.Get("out"));
        return 0;
    }

    private static int SiteTableCommand(Args a)
    {
        SiteTable.Result r = SiteTable.Build(SiteTable.DefaultNotesDir());
        HostJson.Emit(new { kind = "fidelity-site-table", ok = r.Ok, r.Total, r.NonPresentation, r.Problems, rows = r.Rows }, a.Get("out"));
        return r.Ok ? 0 : 4;
    }
}
