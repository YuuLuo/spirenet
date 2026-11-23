using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

/// <summary>Names of everything the rooms module registers (DRIVER-API.md section 10: "&lt;team&gt;:&lt;thing&gt;").</summary>
public static class RoomsNames
{
    public const string RewardsLifecycle = "rooms:rewards";
    public const string TreasureLifecycle = "rooms:treasure";
    public const string PotionsLifecycle = "rooms:potions-state";
    public const string PotionsExtension = "rooms:potions";
    public const string RewardFrame = "rooms:reward";
    public const string TreasureHandler = "rooms:treasure";
    public const string RestHandler = "rooms:rest";
    public const string ShopHandler = "rooms:shop";
    public const string RemoteRewards = "rooms:remote-reward";
    public const string RemoteCardRewardFrame = "rooms:card-reward-remote";
    public const string MendRouter = "rooms:mend-target";
}

internal static class RoomsRegistration
{
    public static void Register(DriverRegistry registry)
    {
        RoomsPatches.InstallOnce();

        registry.SetLifecycle(new RewardsLifecycle());
        TreasureLifecycle treasure = new TreasureLifecycle();
        registry.SetLifecycle(treasure);
        registry.SetRoomHandler(RoomScreen.Treasure, new TreasureHandler(), "rooms");
        MendRouter mend = new MendRouter();
        registry.SetRoomHandler(RoomScreen.Rest, new RestHandler(mend), "rooms");
        registry.SetRoomHandler(RoomScreen.Shop, new ShopHandler(), "rooms");

        // Multi-seat (phase H5): the reward screens of the remote seats (overlay + per-episode state + the card reward
        // waits) and the MEND target. Router order: MEND before the card reward (each claims only its own waits).
        RemoteRewards remote = new RemoteRewards();
        RemoteRewards.Instance = remote;
        registry.SetLifecycle(remote);
        registry.SetRemoteOverlay(remote);
        registry.SetRemoteChoiceRouter(mend);
        registry.SetRemoteChoiceRouter(remote);

        OutOfCombatPotions potions = new OutOfCombatPotions();
        registry.SetLifecycle(potions.State);
        registry.SetFrameExtension(potions);
        registry.SetModuleCounters(new RoomsCounters());
    }
}
