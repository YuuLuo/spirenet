using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Encoding;

/// <summary>Per-episode encoder switches (RESET.flags / view, schema 7.1 as amended by schema-errata-v1).</summary>
public sealed class EncoderOptions
{
    public bool MapInCombat;
    public bool DeckInCombat;
    public bool UpgradePreviews = true;
    public bool TargetPreviews = true;

    /// <summary>RESET flag FORECAST: the FORECAST trailer table (notes/forecast/contract.md).</summary>
    public bool Forecast;

    /// <summary>RESET flag CALC: the TURN / FOE / ROUTE trailer tables (notes/calc/contract.md).</summary>
    public bool Calc;

    /// <summary>0 TEAM, 1 SEAT (schema 1, 8).</summary>
    public int ViewMode;

    /// <summary>SEAT view: the seat; TEAM: -1.</summary>
    public int ViewSeat = -1;

    public static EncoderOptions FromResetFlags(int flags, int viewMode, int viewSeat)
    {
        return new EncoderOptions
        {
            MapInCombat = (flags & ResetFlag.MapInCombat) != 0,
            DeckInCombat = (flags & ResetFlag.DeckInCombat) != 0,
            UpgradePreviews = (flags & ResetFlag.UpgradePreviews) != 0,
            TargetPreviews = (flags & ResetFlag.TargetPreviews) != 0,
            Forecast = (flags & ResetFlag.Forecast) != 0,
            Calc = (flags & ResetFlag.Calc) != 0,
            ViewMode = viewMode,
            ViewSeat = viewMode == 0 ? -1 : viewSeat,
        };
    }

    /// <summary>
    /// Reference mode: evaluate every card display value with the game's own methods instead of the per-frame hook cache
    /// (<see cref="HookCache"/>). Identical bytes by construction; the self-test encodes every frame both ways.
    /// </summary>
    public bool ExactGameCalls;

    /// <summary>Diagnostics: accumulate per-section Stopwatch time (ObsEncoder.SectionTicks); costs about 1 us per frame.</summary>
    public bool SectionTiming;

    public bool SeesPrivate(int seat) => ViewMode == 0 || seat == ViewSeat;

    public EncoderOptions Clone() => (EncoderOptions)MemberwiseClone();
}

/// <summary>Encoder health counters (reported by the self-tests and the server on stderr at exit).</summary>
public sealed class EncoderStats
{
    public long Frames;
    public long Terminal;
    public long ActionOrderViolations;
    public long ChoiceOrderViolations;
    public long UnmappedActionRefs;
    public long Overflows;
    public long UpgradePreviews;
    public long TargetPreviewRows;

    /// <summary>
    /// FORECAST blocks written, and the UNKNOWN slots by reason (contract 1.3 U1-U3; U4 = a move without follow-up state,
    /// notes/forecast/deviations.md). U1 / U2 count blocks (both slots), U3 / U4 count slots.
    /// </summary>
    public long ForecastBlocks;
    public long ForecastUnknownNoMachine;
    public long ForecastUnknownHidden;
    public long ForecastUnknownTooMany;
    public long ForecastUnknownNoFollowUp;

    /// <summary>Calc tables (notes/calc/contract.md): frames with a TURN row, FOE rows, ROUTE rows, knapsack maxima, CAPPED maxima.</summary>
    public long CalcTurnRows;
    public long CalcFoeRows;
    public long CalcRouteRows;
    public long CalcMaxima;
    public long CalcCapped;
    public double TotalUs;
    public long MaxUs;
}
