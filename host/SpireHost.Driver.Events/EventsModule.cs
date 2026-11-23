using SpireHost.Driver;

namespace SpireHost.Driver.Events;

/// <summary>
/// The events module: the EVENT room handler and the event-specific frames:
///
///     room handler  RoomScreen.Event  -> EventRoomHandler ("events:event"): EVENT decisions for every event
///                                        (Ancients / Neow / TheArchitect included) and the FakeMerchant SHOP frame
///     lifecycle     "events:frames"   -> per-episode reset of the patch-side state (bundle stand-in, counters)
///     patches       EventPatches       -> Crystal Sphere (CRYSTAL frame), Trial double-down (CONFIRM frame),
///                                        Scroll Boxes bundle gate + screen (BUNDLE frame), FakeMerchant FoulPotion,
///                                        the presentation null dereferences inside event models
///
///     counters      "events"          -> one counter per patch id / presentation stub that fired
///
/// Every patch is registered through PatchRegistry and listed in host\FIDELITY.md. One configuration: the module has
/// no test extensions (the FakeMerchant FoulPotion throw is offered by the rooms module's potion extension).
/// The constructor must not touch a game type (it runs before the assembly resolver exists).
/// </summary>
public sealed class EventsModule : IDriverModule
{
    public string Name => "events";

    public void Register(DriverRegistry registry)
    {
        EventPatches.InstallOnce();
        registry.SetRoomHandler(RoomScreen.Event, new EventRoomHandler(), owner: "events");
        registry.SetRemoteChoiceRouter(new EventPatches.BundleRemoteRouter()); // multi-seat (phase H5): a remote seat's bundle prompt
        registry.SetLifecycle(new EventsLifecycle());
        registry.SetModuleCounters(new EventsCounters());
    }
}

/// <summary>The counters of the events module: one per patch id / presentation stub that fired ("events.patch.&lt;id&gt;").</summary>
internal sealed class EventsCounters : IModuleCounters
{
    public string Module => "events";

    public void Read(System.Collections.Generic.IDictionary<string, long> c)
    {
        foreach (System.Collections.Generic.KeyValuePair<string, long> kv in EventPatches.Counts)
        {
            c["patch." + kv.Key] = kv.Value;
        }
        // multi-seat (phase H5)
        c["mp.remote_options"] = EventsStats.RemoteOptions;
        c["mp.shared_votes"] = EventsStats.SharedVotes;
        c["mp.remote_shared_votes"] = EventsStats.RemoteSharedVotes;
        c["mp.shared_pages_resolved"] = EventsStats.SharedPagesResolved;
        c["mp.team_votes_cast"] = EventsStats.TeamVotesCast;
        c["mp.team_vote_not_clickable"] = EventsStats.TeamVoteNotClickable;
        c["mp.remote_bundle_frames"] = EventsStats.RemoteBundleFrames;
        c["mp.remote_crystal_frames"] = EventsStats.RemoteCrystalFrames;
        c["mp.remote_crystal_rewards_sent"] = EventsStats.RemoteCrystalRewardsSent;
    }
}

/// <summary>Per-episode housekeeping of the patch-side state. Installs nothing on the game.</summary>
internal sealed class EventsLifecycle : IRunLifecycle
{
    public string Name => "events:frames";

    public void OnRunSetUp(RunContext ctx)
    {
        EventPatches.BeginEpisode();
    }

    public void OnRoomEntered(RunContext ctx)
    {
    }

    public void OnEpisodeEnd(RunContext ctx)
    {
        EventPatches.EndEpisode();
    }
}
