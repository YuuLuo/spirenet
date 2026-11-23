using System;
using System.Collections.Generic;
using System.Linq;
using SpireHost.Driver;
using SpireHost.Patches;

namespace SpireHost;

/// <summary>
/// THE production configuration of the host, defined in exactly one place (H1 integration, rulings A3 / B1).
///
///   production module set = Fidelity + Combat + Rooms + Events + Obs      (registration order below)
///   production patch set  = the Core boot patches + the driver core's two patches + whatever those five modules register
///
/// Every command of the executable - serve, fuzz, determinism, replay-run, the oracle tools (replay-mcr,
/// compare-run-traces, the combat cross-check) and every team's self-tests - creates its HostSession from
/// <see cref="Options"/>, so they all run on the same handlers, the same patches, the same canonical card-class
/// provider and the same vocabulary. There is no other way to build a session in this executable and no command-line
/// switch, environment variable or RESET field that changes the set.
///
/// Nothing in <see cref="Options"/> may touch a game type: the modules are created before the game boots (their
/// constructors must not touch the game either; IDriverModule.Register runs later, on the engine thread).
/// </summary>
public static class ProductionHost
{
    /// <summary>The names of the production modules, in registration order.</summary>
    public static readonly string[] ModuleNames = { "fidelity", "combat", "rooms", "events", "obs" };

    /// <summary>
    /// Registration order matters in two places only: OnRunSetUp runs in this order (fidelity's per-run
    /// assertions first, then the combat lifecycle that installs the run's card selector, then the room / event
    /// lifecycles), and frame extensions run in this order (the rooms potion extension).
    /// </summary>
    public static HostOptions Options(string? area, Args a)
    {
        HostOptions o = new HostOptions();
        o.Boot.GameLog = a.Get("game-log", "off");
        o.Strict = true;
        o.Modules.Add(new SpireHost.Fidelity.FidelityModule());
        o.Modules.Add(new SpireHost.Driver.Combat.CombatModule());
        o.Modules.Add(new SpireHost.Driver.Rooms.RoomsModule());
        o.Modules.Add(new SpireHost.Driver.Events.EventsModule());
        // The vocabulary is loaded at boot in every mode; a missing, mismatching or incomplete vocabulary is a startup
        // error. The one exception is the command that repairs an incomplete vocabulary: --export-vocab.
        bool export = area == null && a.Command == "export-vocab";
        o.Modules.Add(new SpireHost.Obs.ObsModule(a.Get("vocab-dir"), allowMissingNames: export));
        return o;
    }

    /// <summary>
    /// Fail-closed startup check, run for every command before it starts (rulings B1, B2, hard rule "zero Behaviour
    /// patches"): the five modules in order, a handler for every room screen (there is no fallback handler), the
    /// canonical card-class provider, a loaded vocabulary, no Behaviour patch. Throws with everything that is wrong.
    /// </summary>
    public static void Verify(HostSession session)
    {
        List<string> problems = new List<string>();
        if (!session.Registry.Modules.SequenceEqual(ModuleNames))
        {
            problems.Add("modules are [" + string.Join(", ", session.Registry.Modules) + "], expected [" + string.Join(", ", ModuleNames) + "]");
        }
        List<RoomScreen> unowned = session.Registry.ScreensWithoutHandler();
        if (unowned.Count > 0)
        {
            problems.Add("room screens without a handler: " + string.Join(", ", unowned));
        }
        if (session.Registry.CanonicalCardClasses == null)
        {
            problems.Add("no canonical card-class provider is registered");
        }
        if (session.Registry.VocabSource.Length == 0)
        {
            problems.Add("no vocabulary is loaded");
        }
        List<string> behaviour = session.ProductionPatches.Where(p => p.Category == PatchCategory.Behaviour).Select(p => p.Id).ToList();
        if (behaviour.Count > 0)
        {
            problems.Add("Behaviour patches in the production set: " + string.Join(", ", behaviour));
        }
        if (!session.Options.Strict)
        {
            problems.Add("strict mode is off");
        }
        if (problems.Count > 0)
        {
            throw new InvalidOperationException("production configuration check failed: " + string.Join("; ", problems));
        }
    }
}
