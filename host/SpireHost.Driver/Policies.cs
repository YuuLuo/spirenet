using System;
using System.Collections.Generic;
using SpireHost.Sim;

namespace SpireHost.Driver;

/// <summary>
/// A policy answers decisions. <see cref="Decide"/> is called on the ENGINE thread and may block for as
/// long as it likes (read a frame from a pipe, run a network): while it blocks nothing else moves, because
/// the engine thread is the only thread that runs game code (DECISIONS-H1 A.1).
/// It must not call into the game, touch the pump, or keep the Decision after returning.
/// </summary>
public interface IPolicy
{
    /// <summary>Returns the index of the chosen action in <c>decision.Actions</c> (canonical order). Throw PolicyStopException to end the episode as Truncated.</summary>
    int Decide(Decision decision);
}

/// <summary>Optional episode callbacks (protocol writers, loggers, test probes). Engine thread.</summary>
public interface IEpisodeObserver
{
    void OnEpisodeStart(RunContext ctx);

    /// <summary>
    /// "Room entered and settled" (replay request R3): called at the first quiescent point after RunManager.RoomEntered,
    /// before the first Step of the room handler and before any pending frame of that room is answered - the point
    /// where the real UI has finished the room transition and the player could act. Not called for rooms entered while
    /// the start chain runs before OnEpisodeStart (the first room is reported right after OnEpisodeStart). Default: no-op.
    /// </summary>
    void OnRoomBoundary(RunContext ctx)
    {
    }

    /// <summary>After the choice was made and before it is applied. <paramref name="auto"/>: advanced by the host (forced frame / gold).</summary>
    void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto);

    /// <summary>
    /// The episode's last state, while the run is still alive (before teardown): the place to encode a
    /// terminal observation. Called for every outcome once the run was started. <paramref name="provisional"/>
    /// holds the outcome decided so far; end-of-episode strict checks may still turn it into Aborted.
    /// </summary>
    void OnTerminalState(RunContext ctx, EpisodeResult provisional);

    /// <summary>After teardown: the final result (also for episodes whose start failed).</summary>
    void OnEpisodeEnd(EpisodeResult result);
}

/// <summary>Uniformly random legal action. SplitMix64 seeded by the caller; independent of every game RNG stream.</summary>
public sealed class RandomLegalPolicy : IPolicy
{
    private PolicyRng _rng;

    public ulong Seed { get; private set; }

    public RandomLegalPolicy(ulong seed)
    {
        Seed = seed;
        _rng = new PolicyRng(seed);
    }

    public void Reseed(ulong seed)
    {
        Seed = seed;
        _rng = new PolicyRng(seed);
    }

    public int Decide(Decision decision) => _rng.NextInt(decision.Actions.Count);
}

/// <summary>
/// Plays back a recorded list of action indices (EpisodeResult.ActionLog), one per policy decision. When the
/// script is exhausted: continues with <see cref="Fallback"/> if set, otherwise ends the episode as Truncated.
/// An index outside the action list is a PolicyFault abort (the run diverged from the recording).
/// </summary>
public sealed class ScriptedPolicy : IPolicy
{
    private readonly IReadOnlyList<int> _script;

    public int Position { get; private set; }

    public IPolicy? Fallback;

    public ScriptedPolicy(IReadOnlyList<int> script, IPolicy? fallback = null)
    {
        _script = script;
        Fallback = fallback;
    }

    public int Decide(Decision decision)
    {
        if (Position >= _script.Count)
        {
            if (Fallback != null)
            {
                return Fallback.Decide(decision);
            }
            throw new PolicyStopException("script exhausted after " + _script.Count + " decisions");
        }
        return _script[Position++];
    }
}

/// <summary>Always the first legal action in canonical order (END_TURN in combat). Cheap smoke-test policy.</summary>
public sealed class FirstActionPolicy : IPolicy
{
    public int Decide(Decision decision) => 0;
}
