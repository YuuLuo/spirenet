using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game.Sync;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

// =========================================================================================================
//  REST (driver-spec-rooms.md 6; testmode-audit-nodes.md U-RS-1)
//
//  Options: RestSiteSynchronizer.GetLocalOptions() (RestSiteOption.Generate + Hook.ModifyRestSiteOptions, built on
//  room entry). A button is clickable only when IsEnabled (NRestSiteButton.Create :145-151, OnRelease :172-178).
//  Choosing = RestSiteSynchronizer.ChooseLocalOption(index into the CURRENT list) (NRestSiteButton.SelectOption
//  :180-209); false (e.g. Smith's selection cancelled) leaves the option. On success the synchronizer clears the
//  remaining options unless Hook.ShouldDisableRemainingRestSiteOptions is false (MiniatureTent), which only
//  removes the chosen one (RestSiteSynchronizer.cs:181-212).
//  Proceed: disabled in _Ready (NRestSiteRoom.cs:221); enabled after the first success (AfterSelectingOptionAsync
//  -> ShowProceedButton, :441-472) or when no option is left (OnActiveScreenUpdated, :547-557). Its handler is
//  RunManager.Proceed() (:475-478).
//
//  MULTI-SEAT (phase H5). BeginRestSite builds the options of EVERY player (RestSiteSynchronizer.cs:90-103); each
//  player sees and clicks its own (GetOptionsForPlayer). A remote seat's click is its client's
//  OptionIndexChosenMessage{RestSite, index, location} (ChooseLocalOption :159-168 -> host :105-117); its PROCEED
//  opens its own map and sends nothing; a remote seat that leaves with options left sends RestSiteSkippedMessage at
//  the room exit (rooms-remote-rest-exit-skip).
//  MEND (more than one player only, RestSiteOption.cs:56-59): the button starts a targeting of another player
//  (MendRestSiteOption.OnSelect, Entities.RestSite/MendRestSiteOption.cs:78-158; every player except the owner,
//  :204-207) and heals the TARGET by the target's own rest amount. Offered as one action per target
//  (CHOOSE_OPTION(MEND), aux = target seat): button and target are one decision; a cancelled targeting is a no-op and
//  is not offered. The target is the answer to the choice the option waits for - PlayerChoiceResult.FromPlayerId -
//  which the driver gives itself (<see cref="MendRouter"/>): a remote owner's client sends it as its
//  PlayerChoiceMessage; seat 0's own MEND takes the same wait (fidelity-mend-local-targeting: the local branch is UI
//  code that cannot run headless) and receives it through PlayerChoiceSynchronizer.ReceiveReplayChoice.
// =========================================================================================================

internal sealed class RestState
{
    /// <summary>Index = seat (a single-seat run uses [0] only).</summary>
    public readonly int[] Successes = new int[DriverApi.MaxSeats];
    public readonly int[] Choices = new int[DriverApi.MaxSeats];
}

internal sealed class RestHandler : IRoomHandler
{
    private readonly Decision _decision = new Decision();
    private readonly RestContext _context = new RestContext();
    private readonly MendRouter _mend;

    public RestHandler(MendRouter mend)
    {
        _mend = mend;
    }

    public string Name => RoomsNames.RestHandler;

    public HandlerStep Step(RunContext ctx, int seat)
    {
        if (ctx.CurrentRoom is not RestSiteRoom room)
        {
            return HandlerStep.Unrecognised("rest handler outside a RestSiteRoom");
        }
        RestState state = ctx.RoomData<RestState>(room);
        Player player = ctx.Seats[seat];
        IReadOnlyList<RestSiteOption> options = ctx.RunManager.RestSiteSynchronizer.GetOptionsForPlayer(player);
        _context.Options = options;
        _context.Successes = state.Successes[seat];
        Decision d = _decision.Reset(DecisionKind.Rest, this, actingSeat: seat, context: _context);
        for (int i = 0; i < options.Count; i++)
        {
            if (!options[i].IsEnabled)
            {
                continue;
            }
            if (options[i] is MendRestSiteOption)
            {
                // One action per target: every player except the owner (MendRestSiteOption.AllowHoveringNode, :204-207).
                for (int t = 0; t < ctx.SeatCount; t++)
                {
                    if (t != seat)
                    {
                        d.Add(Verb.ChooseOption, seat, GameRef.OfRestOption(options[i], i), aux: t);
                    }
                }
                continue;
            }
            d.Add(Verb.ChooseOption, seat, GameRef.OfRestOption(options[i], i));
        }
        if (state.Successes[seat] > 0 || options.Count == 0)
        {
            d.Add(Verb.Proceed, seat);
        }
        if (d.Actions.Count == 0)
        {
            // Every option disabled and nothing chosen yet: the real UI has no clickable button and the proceed
            // button stays disabled - a soft lock. Reported as a stall, never resolved by an invented exit
            // (driver-spec-rooms.md 6.2). Checked before the potion extension adds anything (a potion cannot unlock it).
            ctx.Stall("rest_site_soft_lock", "Rest site: " + options.Count + " option(s), none enabled, no successful choice yet. " + ctx.Diagnose());
        }
        return HandlerStep.Ask(d);
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        int seat = a.Seat;
        VirtualClient? client = ctx.Mp != null && ctx.IsRemoteSeat(seat) ? ctx.Mp.Client(seat) : null;
        if (a.Verb == Verb.Proceed)
        {
            RoomsProbe.RestLeaves++;
            if (client != null)
            {
                ctx.OpenMapFor(seat, "leave rest site (remote)");
            }
            else
            {
                RoomsFlow.Proceed(ctx, "leave rest site");
            }
            return;
        }
        RestState state = ctx.RoomData<RestState>();
        RestSiteSynchronizer sync = ctx.RunManager.RestSiteSynchronizer;
        Player player = ctx.Seats[seat];
        IReadOnlyList<RestSiteOption> options = sync.GetOptionsForPlayer(player);
        RestSiteOption option = a.Source.RestOption;
        if (a.Source.Index >= options.Count || !ReferenceEquals(options[a.Source.Index], option))
        {
            ctx.Fail(AbortReason.HandlerFault, "rest_option_moved", "Rest option " + option.OptionId + " is no longer at index " + a.Source.Index + ".");
        }
        RoomsProbe.RestChoices++;
        if (state.Successes[seat] > 0)
        {
            RoomsProbe.RestChoicesAfterSuccess++;
        }
        if (option is MendRestSiteOption)
        {
            // The target of this click; MendRestSiteOption.OnSelect asks for it right away (its first statement reserves the choice id).
            _mend.Expect(ctx, seat, ctx.Seats[a.Aux]);
            RoomsProbe.MendChoices++;
        }
        state.Choices[seat]++;
        if (client != null)
        {
            RoomsProbe.RemoteRestChoices++;
            OptionIndexChosenMessage message = default;
            message.type = OptionIndexType.RestSite;
            message.optionIndex = (uint)a.Source.Index;
            message.location = client.Location;
            Task chosen = client.SendTracked(message);
            ctx.Track(Observe(ctx, (Task<bool>)chosen, state, seat), "seat " + seat + " rest option " + option.OptionId, seat);
        }
        else
        {
            ctx.Track(Observe(ctx, sync.ChooseLocalOption(a.Source.Index), state, seat), "rest option " + option.OptionId, seat);
        }
        _mend.AssertConsumed(ctx, seat);
    }

    private static async Task Observe(RunContext ctx, Task<bool> chosen, RestState state, int seat)
    {
        if (await chosen)
        {
            state.Successes[seat]++;
            RoomsProbe.RestSuccesses++;
            ctx.SetSeatFlag(seat, SeatStatus.RestDone);
        }
        else
        {
            RoomsProbe.RestFailures++;
        }
    }
}

/// <summary>
/// The MEND target (IRemoteChoiceRouter). MendRestSiteOption.OnSelect reserves a choice id and waits for
/// PlayerChoiceResult.AsPlayerId (:80, :135). The REST decision already named the target (one action per target), so
/// the wait is answered at once - posted, so that it arrives after the game's call has returned, like a packet:
/// a remote owner's client sends PlayerChoiceMessage{choiceId, FromPlayerId(target)} (what its local branch syncs,
/// :118); for seat 0 (which reaches the same wait through fidelity-mend-local-targeting) the result is handed to
/// PlayerChoiceSynchronizer.ReceiveReplayChoice, the synchronizer's public entry that does what its message handler does.
/// </summary>
internal sealed class MendRouter : IRemoteChoiceRouter
{
    private RunContext? _of;
    private readonly Player?[] _target = new Player?[DriverApi.MaxSeats];

    public string Name => RoomsNames.MendRouter;

    public void Expect(RunContext ctx, int seat, Player target)
    {
        if (!ReferenceEquals(_of, ctx))
        {
            _of = ctx;
            System.Array.Clear(_target);
        }
        _target[seat] = target;
    }

    /// <summary>The option's wait must have consumed the target inside the call that chose the option (it is the option's first statement).</summary>
    public void AssertConsumed(RunContext ctx, int seat)
    {
        if (ReferenceEquals(_of, ctx) && _target[seat] != null)
        {
            _target[seat] = null;
            ctx.Fail(AbortReason.HandlerFault, "mend_target_not_asked", "MEND was chosen for seat " + seat + " but the option did not ask for its target.");
        }
    }

    public bool TryRoute(RunContext ctx, Player player, uint choiceId)
    {
        if (!ReferenceEquals(_of, ctx))
        {
            return false;
        }
        int seat = ctx.SeatOf(player);
        Player? target = _target[seat];
        if (target == null)
        {
            return false;
        }
        _target[seat] = null;
        PlayerChoiceResult result = PlayerChoiceResult.FromPlayerId(target.NetId);
        RoomsProbe.MendTargetsAnswered++;
        if (ctx.IsRemoteSeat(seat))
        {
            VirtualClient client = ctx.Mp!.Client(seat);
            ctx.Post(() =>
            {
                client.SendChoice(choiceId, result);
                return Task.CompletedTask;
            }, "seat " + seat + " sends its MEND target", blocking: false);
        }
        else
        {
            ctx.Post(() =>
            {
                ctx.RunManager.PlayerChoiceSynchronizer.ReceiveReplayChoice(player, choiceId, result.ToNetData());
                return Task.CompletedTask;
            }, "seat 0's MEND target", blocking: false);
        }
        return true;
    }
}
