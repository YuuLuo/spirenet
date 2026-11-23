using System;
using SpireHost;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

/// <summary>
/// Team commands of the rooms console (no game type here; the implementations live in RoomsTests):
///   --selftest [--characters X,Y | all] [--group reward|card-reward|tree|treasure|rest|shop|potions] [--scenario substr]
///              [--policy-seeds N] [--out FILE]
///       the scenario battery: real run start, loadout + debug jump, scripted / random policy, contract observer,
///       per-scenario checks of the rooms probe counters.
///   --tree-runs [--runs N] [--characters ..] [--ascensions 0,10] [--hp N] [--seed-base S] [--policy-seed P] [--out FILE]
///       whole random runs (default forced HP 99999); aggregates the proceed-tree branch counters and their
///       post-condition checks (next act / second boss / TheArchitect / parent event) over all runs.
///   --trace-run --seed S --character C --ascension A --policy-seed P [--hp N] [--tail N]
///       debug: one fuzz repro tuple with RandomLegalPolicy; prints the last N decisions and the abort.
/// Plus the run-level commands of every team console (--fuzz-runs, --replay-run, --determinism --runs N,
/// --selftest-driver, --driver-info).
/// </summary>
public static class RoomsCommands
{
    public static int Run(HostSession session, Args a)
    {
        switch (a.Command)
        {
            case "selftest":
                return RoomsTests.SelfTest(session, a);
            case "tree-runs":
                return RoomsTests.TreeRuns(session, a);
            case "trace-run":
                return RoomsTests.TraceRun(session, a);
            default:
                Console.Error.WriteLine("rooms: --selftest [--characters X,Y|all] [--group G] [--scenario S] [--policy-seeds N] | --tree-runs [--runs N] [--hp N] [--ascensions 0,10] | --trace-run ...");
                return 1;
        }
    }
}
