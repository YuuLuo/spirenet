namespace SpireHost.Driver;

/// <summary>The version of the driver contract (host\DRIVER-API.md, section "Changing this contract") this build implements.</summary>
public static class DriverApi
{
    public const string Version = "h5-4p";

    /// <summary>The game's lobby size: a run has 1 to 4 seats (phase H5).</summary>
    public const int MaxSeats = 4;
}
