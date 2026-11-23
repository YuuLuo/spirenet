using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;
using SpireHost.Boot;
using SpireHost.Engine;
using SpireHost.Patches;

namespace SpireHost.Driver;

public sealed class HostOptions
{
    public BootOptions Boot = new BootOptions();

    /// <summary>Stack reservation of the engine thread.</summary>
    public int StackBytes = EngineThread.DefaultStackBytes;

    /// <summary>
    /// Strict mode (default): a foreign-thread post, a real-time wait, a dropped deferred call, an unexpected
    /// native call, an off-thread continuation or a late unobserved fault marks the episode as Aborted
    /// (StrictViolation). Off: the same counters are only reported.
    /// </summary>
    public bool Strict = true;

    /// <summary>
    /// The modules to register, in order. The production set is defined in exactly one place (the main executable,
    /// ProductionHost); there are no built-in fallback handlers: a room screen without a handler stalls the episode.
    /// </summary>
    public List<IDriverModule> Modules = new List<IDriverModule>();

    /// <summary>Godot native entry points that may be reached after boot without failing the episode.</summary>
    public HashSet<string> NativeAllowList = new HashSet<string>(StringComparer.Ordinal)
    {
        // Lazily initialised StringName statics of game types (value-type handles; inert headlessly).
        "godotsharp_string_name_new_from_string",
        "godotsharp_string_new_with_utf16_chars",
    };
}

/// <summary>
/// One game instance = one process = one HostSession. The session owns the dedicated big-stack ENGINE
/// THREAD, boots the game on it, installs the single-threaded pump in strict mode and runs episodes.
///
/// Threading contract: everything in this class except <see cref="Run"/> is called on the engine thread.
/// Game code, handlers, policies and observers all run there; no other thread may touch the game or the pump.
/// </summary>
public sealed class HostSession
{
    private readonly RunDriver _driver;
    private readonly RandomNumberGenerator _csprng = RandomNumberGenerator.Create();

    public static HostSession? Current { get; private set; }

    /// <summary>
    /// True in a fork child of the search host (SpireHost.Search sets it right after fork()): the process has only the
    /// thread that forked, so there is no finalizer thread and GC.WaitForPendingFinalizers would block forever.
    /// </summary>
    public static bool NoFinalizerThread;

    public HostOptions Options { get; }

    public EnginePump Pump { get; }

    public DriverRegistry Registry { get; } = new DriverRegistry();

    public BootReport Boot { get; }

    public long EpisodesRun { get; private set; }

    /// <summary>The context of the episode that is running right now, or null.</summary>
    public RunContext? CurrentEpisode => _driver.Context;

    /// <summary>
    /// The PRODUCTION patch set: every registry record present when the session was created. It is the same in every
    /// mode of the executable (gate G1); whatever a verification command registers afterwards is its own read-only
    /// tool instrumentation (<see cref="ToolPatches"/>, ruling B4).
    /// </summary>
    public IReadOnlyList<PatchRecord> ProductionPatches { get; }

    /// <summary>PatchLedger.Hash of <see cref="ProductionPatches"/> (recorded by dist.json, logs and tool outputs).</summary>
    public string ProductionPatchHash { get; }

    /// <summary>Registry records added after the session was created (tool-only instrumentation of the running command).</summary>
    public IReadOnlyList<PatchRecord> ToolPatches
    {
        get
        {
            List<PatchRecord> extra = new List<PatchRecord>();
            for (int i = ProductionPatches.Count; i < PatchRegistry.Records.Count; i++)
            {
                extra.Add(PatchRegistry.Records[i]);
            }
            return extra;
        }
    }

    private HostSession(HostOptions options)
    {
        Options = options;
        Boot = GameBoot.Run(options.Boot);
        if (!VeteranProfile.Installed)
        {
            throw new InvalidOperationException("The run driver requires the veteran profile.");
        }
        Pump = EnginePump.Current ?? EnginePump.InstallOnCurrentThread();
        Pump.Strict = options.Strict;
        _driver = new RunDriver(this);
        DriverPatches.Install();
        RemoteSeatPatches.Install();
        Registry.SetRoomHandler(RoomScreen.Map, new MapHandler(), "core");
        Registry.SetModuleCounters(new CoreCounters());
        Registry.SetModuleCounters(new MultiSeatCounters());
        foreach (IDriverModule module in options.Modules)
        {
            Registry.Register(module);
        }
        // Ruling B3: one patch per site, unless the ids that share a method were declared together. Fail closed at startup.
        PatchRegistry.AssertOnePatchPerSite();
        List<PatchRecord> production = new List<PatchRecord>(PatchRegistry.Records);
        ProductionPatches = production;
        ProductionPatchHash = PatchLedger.Hash(production);
    }

    /// <summary>
    /// Creates the engine thread, boots the game on it, runs <paramref name="body"/> there and returns its
    /// result. The calling thread only waits. This is the normal way to host anything (commands, the future
    /// protocol server, tests).
    /// </summary>
    public static int Run(HostOptions options, Func<HostSession, int> body)
    {
        return EngineThread.Run(() =>
        {
            HostSession session = CreateOnCurrentThread(options);
            try
            {
                return body(session);
            }
            finally
            {
                session.Close();
            }
        }, options.StackBytes);
    }

    /// <summary>For callers that already are on a big-stack engine thread (EngineThread.Run). One session per process.</summary>
    public static HostSession CreateOnCurrentThread(HostOptions options)
    {
        if (Current != null)
        {
            throw new InvalidOperationException("A HostSession already exists in this process (the game's managers are process-wide singletons).");
        }
        HostSession s = new HostSession(options);
        Current = s;
        return s;
    }

    public void Close()
    {
        if (ReferenceEquals(Current, this))
        {
            Current = null;
        }
    }

    /// <summary>
    /// Plays one episode to its end and returns the result. Blocks the engine thread; the policy is called
    /// on this same thread. Never throws for anything that happens inside the episode: every failure is an
    /// Aborted result with a reason (an episode is never dropped).
    /// </summary>
    public EpisodeResult RunEpisode(EpisodeSpec spec, IPolicy policy, EpisodeOptions? options = null, IEpisodeObserver? observer = null)
    {
        if (!Pump.OnEngineThread)
        {
            throw new InvalidOperationException("RunEpisode must be called on the engine thread (use HostSession.Run).");
        }
        EpisodesRun++;
        return _driver.Run(spec, policy, options ?? new EpisodeOptions(), observer);
    }

    /// <summary>A fresh 12-character seed from the game's own SeedHelper over a cryptographic 64-bit seed (never Rng.Chaotic: presentation code shares it).</summary>
    public string NewRandomSeed()
    {
        Span<byte> b = stackalloc byte[8];
        _csprng.GetBytes(b);
        return SeedHelper.GetRandomSeed(new Rng(BitConverter.ToUInt64(b)), 12);
    }

    /// <summary>A reproducible seed derived from a label (fuzz / determinism tools): SeedHelper.GetRandomSeed over a hash of the label.</summary>
    public static string SeedFromLabel(string label)
    {
        return SeedHelper.GetRandomSeed(new Rng(StringHelper.GetDeterministicHashCode(label)), 12);
    }

    internal void ExpectTravel(MegaCrit.Sts2.Core.Map.MapCoord coord) => _driver.ExpectTravel(coord);

    /// <summary>MULTI-SEAT: the last map vote was cast; the host draws the destination among the votes (MapSelectionSynchronizer.cs:123).</summary>
    internal void ExpectTravelAnyOf(System.Collections.Generic.IReadOnlyList<MegaCrit.Sts2.Core.Map.MapCoord> coords) => _driver.ExpectTravelAnyOf(coords);

    /// <summary>Current values of every strict-mode counter source.</summary>
    public StrictCounters ReadStrict()
    {
        return new StrictCounters
        {
            ForeignPosts = System.Threading.Interlocked.Read(ref Pump.ForeignPosts),
            ForeignWaits = Pump.ForeignWaits,
            RealDelays = SchedulingPatches.RealDelays,
            VirtualDelays = SchedulingPatches.VirtualDelays,
            DeferredCalls = GodotApiStubs.DeferredCalls,
            DeferredCallsDropped = GodotApiStubs.DeferredCallsDropped,
            LateUnobservedFaults = HostFaults.LateUnobserved,
            OffThreadContinuations = System.Threading.Interlocked.Read(ref SchedulingPatches.OffThreadContinuations),
            NetworkAttempts = System.Threading.Interlocked.Read(ref NetworkGuard.NetworkAttempts),
            RunSaveBoundaryNotApplied = System.Threading.Interlocked.Read(ref SchedulingPatches.RunSaveBoundariesNotApplied),
            InlineContinuations = SchedulingPatches.InlineContinuations,
            TutorialRoomForced = FidelityGatePatches.TutorialRoomForced,
            DiscoveryBossChanged = FidelityGatePatches.DiscoveryBossChanged,
            DiscoveryRoomsChanged = FidelityGatePatches.DiscoveryRoomsChanged,
            GameErrorsLogged = HeadlessPatches.GameErrors,
            InterceptedProgressWrites = VeteranProfile.InterceptedProgressWrites,
        };
    }
}
