using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using SpireHost.Driver;

namespace SpireHost.Fidelity.Patches;

/// <summary>
/// "This path is unreachable in the host; if it is ever hit, fail the episode." Every assertion throws an
/// <see cref="EpisodeAbortException"/> (detail <c>fidelity_assert:&lt;id&gt;</c>) at the site and counts it. The
/// driver restores the abort reason when the exception surfaces through a game task (RunDriver.RethrowIfAbort).
/// Because a game method could in principle swallow the exception (e.g. CombatManager.cs:369-371 catches every
/// exception), the counter is also checked by <see cref="FidelityInvariants"/> as a driver episode check: at every
/// quiescent point and once more before teardown (so a swallowed assertion followed directly by the terminal state
/// still fails the episode). Counters are module counters ("fidelity.assert.&lt;id&gt;").
/// </summary>
public static class FidelityAssert
{
    public static long Fired;

    public static readonly SortedDictionary<string, long> ById = new SortedDictionary<string, long>(System.StringComparer.Ordinal);

    public static string? Last;

    [DoesNotReturn]
    public static void Fail(string id, AbortReason reason, string message)
    {
        Count(id);
        Last = id + ": " + message;
        throw new EpisodeAbortException(reason, "fidelity_assert:" + id, message);
    }

    /// <summary>Counts an assertion that is reported by returning a CheckFailure instead of throwing.</summary>
    public static void Count(string id)
    {
        Fired++;
        ById[id] = ById.GetValueOrDefault(id) + 1;
    }
}
