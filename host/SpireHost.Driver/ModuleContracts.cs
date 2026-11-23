using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;

namespace SpireHost.Driver;

// ---------------------------------------------------------------------------------------------------------
//  Registry slots that modules fill besides handlers / lifecycles / frame extensions (H1 integration:
//  combat C4 + C7, rooms R3, fidelity R1, obs O1).
// ---------------------------------------------------------------------------------------------------------

/// <summary>
/// "Canonical classes of identical cards" (combat C7 = obs O1). Given the candidates of a pile-sourced CARD_SELECT
/// prompt in the order the game handed them over, returns one representative per class of cards that encode
/// identically - an element of <paramref name="cards"/> - in canonical (content-key) order, and the class sizes.
/// The result must be a pure function of the PUBLIC faces of the cards (never of the pile order).
/// <paramref name="unpoweredPreviews"/> = CardSelectorPrefs.UnpoweredPreviews of the prompt: the CHOICE rows are then
/// encoded without the preview pass, and the class key must be computed the same way.
/// Registered once by the observation module (the key contains vocabulary ids, so only the encoder can compute it)
/// and reached by the CARD_SELECT owner through <see cref="RunContext.CanonicalCardClasses"/>.
/// </summary>
public delegate void CanonicalCardClasses(RunContext ctx, IReadOnlyList<CardModel> cards, PileType pile, int seat, bool unpoweredPreviews,
                                          List<CardModel> classes, List<int> counts);

/// <summary>
/// MULTI-SEAT (phase H5). The owner of a kind of prompt for REMOTE seats. The game asks a remote player by waiting for
/// that player's client (PlayerChoiceSynchronizer.WaitForRemoteChoice(player, choiceId)); the driver core observes
/// every such wait (mp-remote-choice-wait) and offers it to the registered routers in registration order.
/// A router that recognises the prompt - from what it captured about the game call that is waiting - parks a pending
/// frame for that seat (RunContext.Push) and returns true; the frame answers with
/// <c>ctx.Mp.Client(seat).SendChoice(choiceId, result)</c>, encoded exactly as the local branch of the same call site
/// encodes it for PlayerChoiceSynchronizer.SyncLocalChoice. A router must return false for a wait that is not its own;
/// a wait nobody routes aborts the episode. <see cref="TryRoute"/> runs INSIDE the game call: it must not start game
/// work and must not throw for "not mine".
/// </summary>
public interface IRemoteChoiceRouter
{
    /// <summary>Stable name ("combat:card-select-remote").</summary>
    string Name { get; }

    bool TryRoute(RunContext ctx, MegaCrit.Sts2.Core.Entities.Players.Player player, uint choiceId);
}

/// <summary>What an <see cref="IEpisodeCheck"/> found. The episode ends Aborted(Reason, Detail).</summary>
public readonly struct CheckFailure
{
    public readonly AbortReason Reason;
    public readonly string Detail;
    public readonly string Message;

    public CheckFailure(AbortReason reason, string detail, string message)
    {
        Reason = reason;
        Detail = detail;
        Message = message;
    }
}

/// <summary>
/// A read-only check that a module registers and the driver core evaluates at EVERY quiescent point and once more
/// when the episode ends, before teardown (fidelity R1). For conditions that must fail the episode even when the
/// exception thrown at the site was swallowed by game code, or when the episode ends without another decision.
/// Must not change game state, must not pump, and must be cheap (it runs once per settle).
/// </summary>
public interface IEpisodeCheck
{
    /// <summary>Stable name ("fidelity:assert-backstop").</summary>
    string Name { get; }

    /// <summary>
    /// Called on the engine thread at the very beginning of an episode (DRIVER-API.md 4.3 step 2): before the profile is
    /// restored and before the run is created, right after the module counters were snapshotted. The place for a baseline
    /// of process-wide counters: whatever happens from here on - the run set-up included - belongs to this episode.
    /// (Verifier finding R-8: the only hook used to be OnEpisodeStart, after the set-up, so a condition raised and
    /// swallowed by game code while the run was being set up was absorbed into the baseline.) Default: no-op.
    /// </summary>
    void OnEpisodeBegin()
    {
    }

    /// <summary>Called on the engine thread after the run was set up (the RunContext exists), before FinalizeStartingRelics / Launch / EnterAct(0).</summary>
    void OnEpisodeStart(RunContext ctx);

    /// <summary>
    /// null = fine. <paramref name="final"/> is true for the last evaluation of the episode (terminal state reached or
    /// the episode is being ended; the run is still alive).
    /// </summary>
    CheckFailure? Check(RunContext ctx, bool final);
}

/// <summary>
/// A module's counters (combat C4 = rooms R3 = fidelity R1). <see cref="Read"/> reports CUMULATIVE process-wide
/// values; the driver core reads them when an episode starts and when it ends and stores the difference in
/// <see cref="EpisodeResult.ModuleCounters"/> under "<see cref="Module"/>.&lt;key&gt;" (sorted, summed by
/// --fuzz-runs / --determinism). A key ending in "_max" is a high-water mark: the episode value is the current
/// value and sums take the maximum. Counters never influence a decision.
/// </summary>
public interface IModuleCounters
{
    /// <summary>Key prefix: "combat", "rooms", "events", "fidelity", "obs" ...</summary>
    string Module { get; }

    /// <summary>Adds / overwrites every counter of this module (keys without the prefix, lower_snake_case, dots for groups).</summary>
    void Read(IDictionary<string, long> into);
}
