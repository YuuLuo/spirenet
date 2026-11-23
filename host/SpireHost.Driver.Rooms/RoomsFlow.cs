using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

/// <summary>Game calls that several room screens of the real UI share.</summary>
internal static class RoomsFlow
{
    /// <summary>
    /// <c>RunManager.ProceedFromTerminalRewardsScreen(Async)</c> as the UI calls it (terminal reward screen,
    /// combat without rewards, treasure room PROCEED / SKIP). The game method (RunManager.cs:1266-1287) either
    /// resumes the parent event (<c>CurrentRoomCount &gt; 1</c> and the combat room has
    /// <c>ShouldResumeParentEventAfterCombat</c>) or calls <c>Proceed()</c>, which only opens the map through
    /// <c>NMapScreen.Instance?</c> (RunManager.cs:1289-1296) - a no-op headlessly; the core's run-proceed-opens-map
    /// sets the host's stand-in (RunContext.MapOpen) on that call. The async version is used (the void one wraps it
    /// in RunSafely, which would swallow a fault). Returns true when the parent event was resumed.
    /// </summary>
    public static async Task<bool> ProceedFromTerminal(RunContext ctx, string why)
    {
        RunState run = ctx.Run;
        // Same predicate as RunManager.ProceedFromTerminalRewardsScreenAsync, evaluated before the call changes the room stack.
        bool resumes = run.CurrentRoomCount > 1 && run.CurrentRoom is CombatRoom { ShouldResumeParentEventAfterCombat: true };
        await ctx.RunManager.ProceedFromTerminalRewardsScreenAsync();
        if (resumes)
        {
            RoomsProbe.TreeCheck(run.CurrentRoom is EventRoom, "resume: the room after ResumePreviousRoom is " + (run.CurrentRoom?.GetType().Name ?? "null") + ", expected the parent EventRoom");
        }
        else
        {
            RequireMapOpen(ctx, why);
        }
        return resumes;
    }

    /// <summary>
    /// A room's LEAVE / PROCEED button whose handler is <c>RunManager.Instance.Proceed()</c> (rest site
    /// NRestSiteRoom.cs:475-478, shop NMerchantRoom.cs:179-185). The call itself is made (it is a no-op in Standard
    /// mode without NMapScreen); the core's run-proceed-opens-map sets the host's stand-in on it.
    /// </summary>
    public static void Proceed(RunContext ctx, string why)
    {
        ctx.RunManager.Proceed();
        RequireMapOpen(ctx, why);
    }

    /// <summary>Fail closed: after RunManager.Proceed() the host map must be open (the core hook ran).</summary>
    private static void RequireMapOpen(RunContext ctx, string why)
    {
        if (!ctx.MapOpen)
        {
            ctx.Fail(AbortReason.HostBug, "map_not_opened", "RunManager.Proceed() was called (" + why + ") but the host map is not open: run-proceed-opens-map did not run.");
        }
    }
}
