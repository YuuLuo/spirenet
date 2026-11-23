using System;
using System.Collections.Generic;

namespace SpireHost.Driver;

/// <summary>
/// Which room-level screen the run is on. Computed centrally by the driver core at a quiescent point when
/// no pending frame is on the decision stack (RunDriver.Classify); a room handler is registered per value.
/// </summary>
public enum RoomScreen
{
    /// <summary>Not recognised: stall.</summary>
    None = 0,

    /// <summary>The map is open (RunContext.MapOpen) or the current room is a MapRoom.</summary>
    Map,

    /// <summary>CombatManager.Instance.IsInProgress (wherever the combat came from: map, event, debug jump).</summary>
    Combat,

    /// <summary>Current room is an EventRoom (Ancients, Neow, TheArchitect, FakeMerchant and Crystal Sphere included) and no combat is in progress.</summary>
    Event,

    /// <summary>Current room is a MerchantRoom.</summary>
    Shop,

    /// <summary>Current room is a RestSiteRoom.</summary>
    Rest,

    /// <summary>Current room is a TreasureRoom.</summary>
    Treasure,
}

public enum StepKind
{
    /// <summary>The handler does not recognise the state: the episode stalls (aborted, with diagnostics).</summary>
    None = 0,

    /// <summary>Here is the decision to ask.</summary>
    Decision,

    /// <summary>
    /// The handler did host-side work that the real UI does without asking the player (open the chest, start
    /// a tracked operation); the core settles and classifies again. Bounded: too many in a row is a livelock.
    /// </summary>
    Progressed,

    /// <summary>
    /// MULTI-SEAT (phase H5). The seat has nothing to decide until a teammate has acted (it voted and the vote is not
    /// resolved, it is ready for the next act, the treasure award waits for the other votes). The core looks at the
    /// next seat; when no seat has a decision the episode stalls (all_seats_waiting). In a single-seat run this is a
    /// stall like <see cref="None"/>: nobody else can act.
    /// </summary>
    Waiting,

    /// <summary>
    /// MULTI-SEAT (phase H5). Here is the decision, and it is the seat's TEAM VOTE - a vote whose result is one draw
    /// from the list of all seats' votes (the map vote, the page vote of a shared event). The core asks it only when
    /// no seat has any other decision left (lead ruling Q5: votes come last), lowest seat first.
    /// </summary>
    Vote,
}

public readonly struct HandlerStep
{
    public readonly StepKind Kind;
    public readonly Decision? Decision;

    /// <summary>For None: why (goes into the stall diagnostics).</summary>
    public readonly string? Note;

    private HandlerStep(StepKind kind, Decision? decision, string? note)
    {
        Kind = kind;
        Decision = decision;
        Note = note;
    }

    public static HandlerStep Ask(Decision decision) => new HandlerStep(StepKind.Decision, decision, null);

    public static HandlerStep Progressed(string? note = null) => new HandlerStep(StepKind.Progressed, null, note);

    public static HandlerStep Unrecognised(string why) => new HandlerStep(StepKind.None, null, why);

    /// <summary>MULTI-SEAT: this seat waits for a teammate (<paramref name="why"/> goes into the all_seats_waiting diagnostics).</summary>
    public static HandlerStep Waiting(string why) => new HandlerStep(StepKind.Waiting, null, why);

    /// <summary>MULTI-SEAT: this seat's team vote (asked after every other kind of decision of every seat).</summary>
    public static HandlerStep Vote(Decision decision) => new HandlerStep(StepKind.Vote, decision, null);
}

/// <summary>
/// Room-level driver for one <see cref="RoomScreen"/>. <see cref="Step"/> is called on the engine thread
/// when: the engine is quiescent, the run is not over, the decision stack is empty, no blocking tracked
/// operation is pending (Combat is exempt from that last condition) and the screen classifies to this
/// handler. It must enumerate legal actions with the game's own predicates plus the UI-only rules - never
/// by trial and error - and must not read hidden information to order or filter anything.
///
/// <paramref name="seat"/> (phase H5) = the seat whose screen is asked for: always 0 in a single-seat run. In a
/// multi-seat run the core walks the seats in order (DRIVER-API.md section 12) and calls Step once per seat that has
/// no blocking operation of its own; the handler answers for THAT seat only - its own event instance, inventory, rest
/// options, chest - and applies a remote seat's action as the message of its virtual client (RunContext.Mp.Client(seat)).
/// RoomScreen.Combat is the exception: one decision for all seats (the union), called with seat -1.
/// </summary>
public interface IRoomHandler : IDecisionSource
{
    HandlerStep Step(RunContext ctx, int seat);
}

/// <summary>
/// MULTI-SEAT (phase H5). The overlay screens of REMOTE seats: screens the game opens on top of whatever room screen
/// the player is on - the reward sets (RewardsSet.Offer shows NRewardsScreen only on the owner's client; the host
/// merely waits for the set's completion). For seat 0 the same thing is a pending frame on the decision stack; a
/// remote seat's screen is asked right after the stack, before combat and before any room-level screen, lowest seat
/// first. One implementation is registered (the rooms module).
/// </summary>
public interface IRemoteOverlay : IDecisionSource
{
    /// <summary>The seat's top overlay screen: a decision, a progress step, or Waiting; null = the seat has no overlay screen.</summary>
    HandlerStep? Step(RunContext ctx, int seat);

    /// <summary>True while any remote seat has an overlay screen open (the core then does not expect a room change to complete).</summary>
    bool AnyOpen(RunContext ctx);

    /// <summary>True while this seat has an overlay screen open (RunContext.SeatScreen).</summary>
    bool IsOpen(RunContext ctx, int seat);
}

/// <summary>
/// A decision the game asked for through a callback (card selection, reward set, bundle, crystal sphere,
/// confirmation ...), parked on the decision stack. The frame on top of the stack is always answered first.
/// The object that pushes a frame also pops it (RunContext.Pop) when the game's request has been answered.
/// </summary>
public interface IPendingDecision : IDecisionSource
{
    /// <summary>
    /// Builds the decision for the frame's current state. Called at a quiescent point while the frame is on top.
    /// May return null ONLY after the frame removed itself with RunContext.Pop (the request became moot, e.g.
    /// the game withdrew it); the core then settles and classifies again.
    /// </summary>
    Decision? Build(RunContext ctx);
}

/// <summary>
/// Per-run hooks of a sub-driver: install selectors / static test hooks / event subscriptions when the run
/// has been set up, and remove every one of them when the episode ends (static hooks such as
/// RewardsSet.testSelector outlive the run otherwise).
/// </summary>
public interface IRunLifecycle
{
    /// <summary>Stable owner name ("temp:rewards", "rooms:rewards", "combat:selector" ...): registering the same name replaces.</summary>
    string Name { get; }

    /// <summary>
    /// After RunManager.SetUpNewSingleplayer (all synchronizers and the per-run CardSelectorStack exist),
    /// before FinalizeStartingRelics / Launch / EnterAct(0).
    /// </summary>
    void OnRunSetUp(RunContext ctx);

    /// <summary>RunManager.RoomEntered fired (map moves, event combats, act changes, debug jumps).</summary>
    void OnRoomEntered(RunContext ctx);

    /// <summary>Before RunManager.CleanUp, for every outcome including aborts. Must not throw.</summary>
    void OnEpisodeEnd(RunContext ctx);
}

/// <summary>
/// Adds actions to ROOM-LEVEL frames that it does not own. The core calls <see cref="Extend"/> for every
/// non-nested decision of kind Map / Shop / Rest / Event / Reward / Treasure after its source built it and
/// before it is sealed (DECISIONS-H1 B.7: out-of-combat potion use / discard on room-level frames, never on
/// nested selection frames). Actions appended inside Extend get <see cref="LegalAction.Owner"/> = this
/// extension, and the extension's Apply is called when one of them is chosen. The extension must apply the
/// same legality rules as the real UI and must not remove or reorder the source's actions.
/// </summary>
public interface IRoomFrameExtension : IDecisionSource
{
    void Extend(RunContext ctx, Decision decision);
}


/// <summary>The entry point of a module: registers its handlers / lifecycles / extensions / checks / counters.</summary>
public interface IDriverModule
{
    string Name { get; }

    void Register(DriverRegistry registry);
}

public sealed record RegistryEntry(string Slot, string Name, string Owner);

/// <summary>
/// The extension point of the run driver. One room handler per <see cref="RoomScreen"/> (registering again
/// replaces), any number of named lifecycles, frame extensions, episode checks and counter sources, one
/// canonical-card-class provider and the identity of the loaded vocabulary. The MAP handler is registered by
/// HostSession; everything else comes from the modules (the production set is defined in one place: the main
/// executable, ProductionHost). There are no fallback handlers: a screen without a registered handler stalls
/// the episode (no_handler:&lt;screen&gt;), a prompt without an owner fails it.
/// </summary>
public sealed class DriverRegistry
{
    private readonly IRoomHandler?[] _handlers = new IRoomHandler?[Enum.GetValues<RoomScreen>().Length];
    private readonly string[] _handlerOwners = new string[Enum.GetValues<RoomScreen>().Length];
    private readonly List<IRunLifecycle> _lifecycles = new List<IRunLifecycle>();
    private readonly List<string> _modules = new List<string>();
    private readonly List<IRoomFrameExtension> _extensions = new List<IRoomFrameExtension>();
    private readonly List<IEpisodeCheck> _checks = new List<IEpisodeCheck>();
    private readonly List<IModuleCounters> _counters = new List<IModuleCounters>();
    private readonly List<IRemoteChoiceRouter> _routers = new List<IRemoteChoiceRouter>();
    private IRemoteOverlay? _overlay;

    /// <summary>Set while an episode is running: registrations are frozen.</summary>
    internal bool Frozen;

    public void SetRoomHandler(RoomScreen screen, IRoomHandler handler, string owner)
    {
        AssertMutable();
        if (screen == RoomScreen.None)
        {
            throw new ArgumentException("RoomScreen.None cannot have a handler.");
        }
        _handlers[(int)screen] = handler ?? throw new ArgumentNullException(nameof(handler));
        _handlerOwners[(int)screen] = owner;
    }

    public IRoomHandler? GetRoomHandler(RoomScreen screen) => _handlers[(int)screen];

    /// <summary>Removes the handler of a screen (reaching that screen then stalls with no_handler). Returns false if there was none.</summary>
    public bool RemoveRoomHandler(RoomScreen screen)
    {
        AssertMutable();
        bool had = _handlers[(int)screen] != null;
        _handlers[(int)screen] = null;
        return had;
    }

    /// <summary>Adds a lifecycle, replacing an existing one with the same Name. Order of OnRunSetUp = registration order.</summary>
    public void SetLifecycle(IRunLifecycle lifecycle)
    {
        AssertMutable();
        for (int i = 0; i < _lifecycles.Count; i++)
        {
            if (_lifecycles[i].Name == lifecycle.Name)
            {
                _lifecycles[i] = lifecycle;
                return;
            }
        }
        _lifecycles.Add(lifecycle);
    }

    /// <summary>Removes the lifecycle with this name. Returns false if there was none.</summary>
    public bool RemoveLifecycle(string name)
    {
        AssertMutable();
        for (int i = 0; i < _lifecycles.Count; i++)
        {
            if (_lifecycles[i].Name == name)
            {
                _lifecycles.RemoveAt(i);
                return true;
            }
        }
        return false;
    }

    public T? FindLifecycle<T>() where T : class, IRunLifecycle
    {
        foreach (IRunLifecycle l in _lifecycles)
        {
            if (l is T t)
            {
                return t;
            }
        }
        return null;
    }

    public IReadOnlyList<IRunLifecycle> Lifecycles => _lifecycles;

    /// <summary>Adds a room-frame extension, replacing one with the same Name. Extensions run in registration order.</summary>
    public void SetFrameExtension(IRoomFrameExtension extension)
    {
        AssertMutable();
        for (int i = 0; i < _extensions.Count; i++)
        {
            if (_extensions[i].Name == extension.Name)
            {
                _extensions[i] = extension;
                return;
            }
        }
        _extensions.Add(extension);
    }

    public bool RemoveFrameExtension(string name)
    {
        AssertMutable();
        return _extensions.RemoveAll(e => e.Name == name) > 0;
    }

    public IReadOnlyList<IRoomFrameExtension> FrameExtensions => _extensions;

    /// <summary>Adds an episode check (evaluated at every quiescent point and before teardown), replacing one with the same Name.</summary>
    public void SetEpisodeCheck(IEpisodeCheck check)
    {
        AssertMutable();
        for (int i = 0; i < _checks.Count; i++)
        {
            if (_checks[i].Name == check.Name)
            {
                _checks[i] = check;
                return;
            }
        }
        _checks.Add(check);
    }

    public bool RemoveEpisodeCheck(string name)
    {
        AssertMutable();
        return _checks.RemoveAll(c => c.Name == name) > 0;
    }

    public IReadOnlyList<IEpisodeCheck> EpisodeChecks => _checks;

    /// <summary>Adds a counter source, replacing the one of the same module.</summary>
    public void SetModuleCounters(IModuleCounters counters)
    {
        AssertMutable();
        for (int i = 0; i < _counters.Count; i++)
        {
            if (_counters[i].Module == counters.Module)
            {
                _counters[i] = counters;
                return;
            }
        }
        _counters.Add(counters);
    }

    public IReadOnlyList<IModuleCounters> ModuleCounterSources => _counters;

    /// <summary>MULTI-SEAT (phase H5): adds a router for prompts of remote seats, replacing one with the same Name. Routers are asked in registration order.</summary>
    public void SetRemoteChoiceRouter(IRemoteChoiceRouter router)
    {
        AssertMutable();
        for (int i = 0; i < _routers.Count; i++)
        {
            if (_routers[i].Name == router.Name)
            {
                _routers[i] = router;
                return;
            }
        }
        _routers.Add(router);
    }

    public bool RemoveRemoteChoiceRouter(string name)
    {
        AssertMutable();
        return _routers.RemoveAll(r => r.Name == name) > 0;
    }

    public IReadOnlyList<IRemoteChoiceRouter> RemoteChoiceRouters => _routers;

    /// <summary>MULTI-SEAT (phase H5): the owner of the remote seats' overlay screens (registering again replaces).</summary>
    public void SetRemoteOverlay(IRemoteOverlay overlay)
    {
        AssertMutable();
        _overlay = overlay ?? throw new ArgumentNullException(nameof(overlay));
    }

    public IRemoteOverlay? RemoteOverlay => _overlay;

    /// <summary>Cumulative values of every registered counter source as "module.key" (sorted).</summary>
    public SortedDictionary<string, long> ReadModuleCounters()
    {
        SortedDictionary<string, long> all = new SortedDictionary<string, long>(StringComparer.Ordinal);
        Dictionary<string, long> one = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (IModuleCounters c in _counters)
        {
            one.Clear();
            c.Read(one);
            foreach (KeyValuePair<string, long> kv in one)
            {
                all[c.Module + "." + kv.Key] = kv.Value;
            }
        }
        return all;
    }

    // ------------------------------------------------------------------------------------------------
    //  Canonical card classes (combat C7 = obs O1) and the vocabulary identity (ruling B1)
    // ------------------------------------------------------------------------------------------------

    /// <summary>The registered provider, or null. Production: the content key of the observation encoder, in every mode.</summary>
    public CanonicalCardClasses? CanonicalCardClasses { get; private set; }

    /// <summary>Who registered it ("obs:content-key").</summary>
    public string CanonicalCardClassesName { get; private set; } = "";

    public void SetCanonicalCardClasses(CanonicalCardClasses provider, string name)
    {
        AssertMutable();
        CanonicalCardClasses = provider ?? throw new ArgumentNullException(nameof(provider));
        CanonicalCardClassesName = name;
    }

    /// <summary>VOCAB_HASH / CATALOG_HASH of the vocabulary loaded at boot (0 = none registered) and where it came from. Recorded in action logs and traces.</summary>
    public uint VocabHash { get; private set; }

    public uint CatalogHash { get; private set; }

    public string VocabSource { get; private set; } = "";

    public void SetVocabulary(uint vocabHash, uint catalogHash, string source)
    {
        AssertMutable();
        VocabHash = vocabHash;
        CatalogHash = catalogHash;
        VocabSource = source;
    }

    // ------------------------------------------------------------------------------------------------

    public void Register(IDriverModule module)
    {
        AssertMutable();
        module.Register(this);
        _modules.Add(module.Name);
    }

    public IReadOnlyList<string> Modules => _modules;

    /// <summary>What is registered (reports and gates).</summary>
    public List<RegistryEntry> Describe()
    {
        List<RegistryEntry> r = new List<RegistryEntry>();
        foreach (RoomScreen s in Enum.GetValues<RoomScreen>())
        {
            IRoomHandler? h = _handlers[(int)s];
            if (h != null)
            {
                r.Add(new RegistryEntry("room:" + s, h.Name, _handlerOwners[(int)s]));
            }
        }
        foreach (IRunLifecycle l in _lifecycles)
        {
            r.Add(new RegistryEntry("lifecycle", l.Name, l.GetType().Name));
        }
        foreach (IRoomFrameExtension e in _extensions)
        {
            r.Add(new RegistryEntry("frame-extension", e.Name, e.GetType().Name));
        }
        foreach (IEpisodeCheck c in _checks)
        {
            r.Add(new RegistryEntry("episode-check", c.Name, c.GetType().Name));
        }
        foreach (IModuleCounters c in _counters)
        {
            r.Add(new RegistryEntry("module-counters", c.Module, c.GetType().Name));
        }
        foreach (IRemoteChoiceRouter router in _routers)
        {
            r.Add(new RegistryEntry("remote-choice-router", router.Name, router.GetType().Name));
        }
        if (_overlay != null)
        {
            r.Add(new RegistryEntry("remote-overlay", _overlay.Name, _overlay.GetType().Name));
        }
        if (CanonicalCardClasses != null)
        {
            r.Add(new RegistryEntry("canonical-card-classes", CanonicalCardClassesName, "provider"));
        }
        return r;
    }

    /// <summary>Room screens without a handler (a production host must have none: checked at startup by the executable).</summary>
    public List<RoomScreen> ScreensWithoutHandler()
    {
        List<RoomScreen> missing = new List<RoomScreen>();
        foreach (RoomScreen s in Enum.GetValues<RoomScreen>())
        {
            if (s != RoomScreen.None && _handlers[(int)s] == null)
            {
                missing.Add(s);
            }
        }
        return missing;
    }

    private void AssertMutable()
    {
        if (Frozen)
        {
            throw new InvalidOperationException("DriverRegistry cannot be changed while an episode is running.");
        }
    }
}
