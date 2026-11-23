using System;
using SpireHost;
using SpireHost.Driver;
using SpireHost.Driver.Tools;

namespace SpireHost.Replay;

/// <summary>
/// The oracle commands. "SpireHost --replay-mcr", "--compare-run-traces" and "--compare-run-traces-v2" are top-level
/// commands of the executable; the rest is reached as "SpireHost replay --&lt;command&gt;":
///   --replay-mcr ...            real-game .mcr combat replays re-executed in the host, per-action checksum comparison
///                               (the corpus, the v1 run traces, or with --runs-v2 the fights of the v2 run traces)
///   --compare-run-traces ...    run start / room-boundary parity against oracle\runs golden traces (format v1)
///   --compare-run-traces-v2 ... every recorded decision of the v2 run traces (oracle\runs-v2): legal actions, offers,
///                               state, room dumps and saves, fights inside the run with the recorded combat states
///   --replay-determinism ...    same replay twice -> identical states; one altered event -> detected
///   --selftest [--sample N]     all of the above on a small fixed sample + the run-level replay self-check
/// All of them run on the production module and patch set (ruling B4) plus their own read-only instrumentation.
/// </summary>
public static class ReplayCommands
{
    public const string Usage =
        "       --replay-mcr [--file F | --dir D | --run-traces | --runs-v2 [natural|deep|natural,deep]] [--files a,b] [--start I --count N]\n" +
        "                    [--stride S --offset K] [--loose-timing] [--stop-at-divergence] [--extended-trace] [--coverage] [--no-records] [--out FILE]\n" +
        "       --compare-run-traces [--dir D] [--traces a,b | --start I --count N] [--no-save] [--max-diffs N] [--dump-host DIR] [--out FILE]\n" +
        "       --compare-run-traces-v2 [--dir D | --set natural|deep | --root R] [--traces a,b | --start I --count N] [--stride S --offset K]\n" +
        "                    [--no-save] [--max-divergences N] [--max-diffs N] [--dump-host DIR] [--no-records] [--progress] [--out FILE]\n" +
        "                    test switches (host/README.md): --stop-at-first --no-known-masks --strict-sampling --no-mcr-identity --no-explained\n" +
        "                    --no-checksums --wongo seed|N\n" +
        "replay: --replay-determinism [--files a,b] [--out FILE]\n" +
        "       --selftest [--sample N] [--skip-run-level] [--out FILE]";

    public static int Run(HostSession session, Args a)
    {
        switch (a.Command)
        {
            case "replay-mcr":
                return McrCommands.ReplayMcr(session, a);
            case "compare-run-traces":
                return RunTraceCompare.Run(session, a);
            case "compare-run-traces-v2":
                return V2.RunTraceCompareV2.Run(session, a);
            case "replay-determinism":
                return ReplayDeterminism.Run(session, a);
            case "selftest":
                return ReplaySelfTest.Run(session, a);
            default:
                Console.Error.WriteLine(Usage);
                return 1;
        }
    }
}
