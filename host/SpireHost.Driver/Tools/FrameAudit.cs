using System;

namespace SpireHost.Driver.Tools;

/// <summary>
/// TEST SEAM. The observation encoder's own consistency checks on a decision, reachable from the test probes of
/// every module without a reference to the observation project (ruling D6 keeps the projects apart; the
/// executable, which references both, plugs the encoder in when it runs a self-test command).
/// Nothing in the production path (serve, fuzz, determinism, replay) calls it.
/// </summary>
public static class FrameAudit
{
    /// <summary>
    /// Encodes the decision with a private encoder and returns what its checks found (canonical CHOICE order,
    /// ACTIONS order, unmapped action references, table overflow, an exception), or null. Null delegate = no audit.
    /// </summary>
    public static Func<Decision, string?>? Check;

    public static long Frames;
    public static long Problems;
    public static string? FirstProblem;

    /// <summary>Set while a test runs with a deliberately different class provider: the encoder's order check does not apply there.</summary>
    public static bool Paused;

    /// <summary>Runs the audit (if one is plugged in); returns its finding.</summary>
    public static string? Run(Decision decision)
    {
        Func<Decision, string?>? check = Check;
        if (check == null || Paused)
        {
            return null;
        }
        Frames++;
        string? problem;
        try
        {
            problem = check(decision);
        }
        catch (Exception e)
        {
            problem = "frame audit threw on a " + decision.Kind + " frame: " + e.GetBaseException().GetType().Name + ": " + e.GetBaseException().Message;
        }
        if (problem != null)
        {
            Problems++;
            FirstProblem ??= problem;
        }
        return problem;
    }

    public static object Snapshot() => new { plugged_in = Check != null, frames = Frames, problems = Problems, first_problem = FirstProblem };
}
