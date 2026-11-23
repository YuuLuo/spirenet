using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

/// <summary>
/// The rooms module: REWARD / CARD_REWARD (RewardsSet.testSelector, combat-won trigger, terminal proceed
/// tree), TREASURE, REST, SHOP, the out-of-combat potion frame extension (every room-level frame, the FakeMerchant
/// SHOP frame of the events module included) and the module counters "rooms.*".
/// The class itself must not touch a game type (it is created before the game boots); everything that does lives
/// in <see cref="RoomsRegistration"/>.
/// </summary>
public sealed class RoomsModule : IDriverModule
{
    public string Name => "rooms";

    public void Register(DriverRegistry registry)
    {
        RoomsRegistration.Register(registry);
    }
}
