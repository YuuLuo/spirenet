using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;

namespace SpireHost.Driver.Events;

// =========================================================================================================
//  CRYSTAL: the host's NCrystalSphereScreen (Nodes.Events.Custom.CrystalSphere/NCrystalSphereScreen.cs)
// =========================================================================================================

/// <summary>
/// Pushed by the NCrystalSphereScreen.ShowScreen prefix (CrystalSphereMinigame.PlayMinigame, :135-143, awaits the
/// minigame's completion source right after it). One frame for the whole minigame; each Apply is one click.
///
/// Legal actions = what the screen lets the player do while DivinationCount > 0 (NCrystalSphereScreen.OnCellClicked,
/// :283-295): pick a tool (the two divination buttons, SetBigDivination / SetSmallDivination -> SetTool, :234-246)
/// and click a cell; only HIDDEN cells receive the mouse / focus (NCrystalSphereCell._Ready / EntityClicked:
/// MouseFilter = IsHidden ? Stop : Ignore, FocusMode = IsHidden ? All : None, :68-69, :98-99). The tool choice is
/// folded into the click: CRYSTAL_CLICK(cell, aux = tool) for every hidden cell x {Small, Big}. Hovering
/// (SetHoveredCell / IsHighlighted) is presentation only and is not reproduced.
/// Apply: SetTool(tool) when it differs from the current tool (the button press), then CellClicked(cell) - exactly
/// the two game calls the UI makes. The click that spends the last divination completes the minigame's
/// TaskCompletionSource; PlayMinigame then continues inline with CompleteMinigame ->
/// OneOffSynchronizer.DoLocalCrystalSphereRewards -> RewardsCmd.OfferCustom (a non-terminal REWARD frame) and the
/// event finishes (CrystalSphere.cs:68-81). The frame pops itself before that last click.
/// </summary>
internal sealed class CrystalFrame : IPendingDecision
{
    private readonly CrystalSphereMinigame _minigame;
    private readonly Decision _decision = new Decision();
    private readonly CrystalContext _context;
    private readonly VirtualClient? _remote;

    public CrystalFrame(CrystalSphereMinigame minigame)
    {
        _minigame = minigame;
        _context = new CrystalContext { Minigame = minigame };
    }

    /// <summary>
    /// MULTI-SEAT (phase H5; lead ruling Q4 (a)): the minigame of a REMOTE owner. On the host PlayMinigame returns at once
    /// for a player that is not the local one (CrystalSphereMinigame.cs:135-143) - that player's client plays it - but the
    /// host's minigame object of that player exists and has the client's layout (same constructor, same event Rng). The
    /// seat's clicks are made on that object (SetTool / CellClicked, game code); after the last one the client's
    /// CompleteMinigame is reproduced by its one message, CrystalSphereRewardsMessage{revealed items}
    /// (OneOffSynchronizer.DoLocalCrystalSphereRewards, :113-127), which makes the host offer that player's rewards.
    /// Known difference (ledger: events-crystal-sphere-remote): a revealed curse is added by the client path
    /// (CardPileCmd.AddCurseToDeck in CrystalSphereCurse.RevealItem) instead of arriving as the client's
    /// RewardObtainedMessage; the card and the hooks are the same, the CardChoiceHistoryEntry of the host path is not written.
    /// </summary>
    public CrystalFrame(CrystalSphereMinigame minigame, VirtualClient remote)
        : this(minigame)
    {
        _remote = remote;
    }

    public string Name => "events:crystal";

    public CrystalSphereMinigame Minigame => _minigame;

    public Decision? Build(RunContext ctx)
    {
        CrystalSphereMinigame m = _minigame;
        if (m.DivinationCount <= 0)
        {
            ctx.Fail(AbortReason.HostBug, "crystal_frame_finished", "CRYSTAL frame is on the stack but the minigame has no divination left.");
        }
        _context.FillPublicView(); // contract R2: the encoder reads this view, never CrystalContext.Minigame
        int seat = ctx.SeatOf(m.Owner);
        Decision d = _decision.Reset(DecisionKind.Crystal, this, actingSeat: seat, context: _context);
        CrystalSphereCell[,] cells = m.cells;
        int w = cells.GetLength(0);
        int h = cells.GetLength(1);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                CrystalSphereCell c = cells[x, y];
                if (!c.IsHidden)
                {
                    continue;
                }
                GameRef cell = GameRef.OfCrystalCell(c, y * w + x);
                d.Add(Verb.CrystalClick, seat, cell, aux: (int)CrystalSphereMinigame.CrystalSphereToolType.Small);
                d.Add(Verb.CrystalClick, seat, cell, aux: (int)CrystalSphereMinigame.CrystalSphereToolType.Big);
            }
        }
        return d;
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        CrystalSphereMinigame m = _minigame;
        CrystalSphereCell cell = (CrystalSphereCell)a.Source.Object!;
        CrystalSphereMinigame.CrystalSphereToolType tool = (CrystalSphereMinigame.CrystalSphereToolType)a.Aux;
        if (!cell.IsHidden)
        {
            ctx.Fail(AbortReason.HostBug, "crystal_cell_not_hidden", "Crystal cell " + cell.X + "," + cell.Y + " is not hidden.");
        }
        if (m.CrystalSphereTool != tool)
        {
            m.SetTool(tool);
        }
        bool last = m.DivinationCount == 1;
        if (last)
        {
            // This click completes the minigame (CellClicked: DivinationCount-- first, SetResult at the end): the
            // request is answered, the rewards that follow are a new frame on top.
            ctx.Pop(this);
        }
        if (_remote != null)
        {
            ctx.Track(RemoteClick(m, cell, last, _remote), "crystal sphere click of seat " + _remote.Seat + " " + tool + " " + cell.X + "," + cell.Y, _remote.Seat);
            return;
        }
        ctx.Track(m.CellClicked(cell), "crystal sphere click " + tool + " " + cell.X + "," + cell.Y);
    }

    private static async Task RemoteClick(CrystalSphereMinigame m, CrystalSphereCell cell, bool last, VirtualClient client)
    {
        await m.CellClicked(cell);
        if (!last)
        {
            return;
        }
        // The client's CompleteMinigame (:308-313) -> DoLocalCrystalSphereRewards: the revealed items, in reveal order.
        MegaCrit.Sts2.Core.Multiplayer.Messages.Game.CrystalSphereRewardsMessage message = new MegaCrit.Sts2.Core.Multiplayer.Messages.Game.CrystalSphereRewardsMessage { Location = client.Location };
        foreach (CrystalSphereItem item in m._revealed)
        {
            message.rewards.Add(item.ToSerializable());
        }
        EventsStats.RemoteCrystalRewardsSent++;
        client.Send(message);
    }
}

// =========================================================================================================
//  CONFIRM: Trial "DOUBLE_DOWN" (Models.Events/Trial.cs:116-132) -> NAbandonRunConfirmPopup
// =========================================================================================================

/// <summary>
/// Pushed by the Trial.DoubleDown prefix in place of NModalContainer.Instance.Add(NAbandonRunConfirmPopup.Create(null)).
/// The popup's buttons (NAbandonRunConfirmPopup.cs:98-111): YES -> RunManager.Instance.Abandon() (mainMenu is null),
/// NO -> close. YES gives up the run and is flagged ActionFlags.Abandon (removed by the core unless
/// EpisodeOptions.AllowAbandon, DECISIONS-H1 B.6); with the default options only NO remains and is auto-advanced.
/// </summary>
internal sealed class TrialConfirmFrame : IPendingDecision
{
    /// <summary>Registered in the driver's ConfirmKeys list (obs O7): the vocabulary category confirm_key is built from it.</summary>
    public const string Key = ConfirmKeys.AbandonRun;

    private readonly Trial _trial;
    private readonly Decision _decision = new Decision();
    private readonly ConfirmContext _context;

    public TrialConfirmFrame(Trial trial)
    {
        _trial = trial;
        _context = new ConfirmContext { Key = Key, SourceModel = trial };
    }

    public string Name => "events:trial-confirm";

    public Decision? Build(RunContext ctx)
    {
        int seat = _trial.Owner != null ? ctx.SeatOf(_trial.Owner) : 0;
        Decision d = _decision.Reset(DecisionKind.Confirm, this, actingSeat: seat, context: _context);
        d.Add(Verb.ChooseOption, seat, GameRef.OfConfirm(false));
        d.Add(Verb.ChooseOption, seat, GameRef.OfConfirm(true), flags: ActionFlags.Abandon);
        return d;
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        ctx.Pop(this);
        if (a.Source.Index == 1)
        {
            if (ctx.SeatCount != 1)
            {
                // RunManager.Abandon on a Host-type service casts the service to the game's own NetHostGameService
                // (RunLobby.AbandonRun, Multiplayer.Game.Lobby/RunLobby.cs:152). Unreachable: DOUBLE_DOWN is a WillKillPlayer
                // option, which the UI blocks with more than one player (NEventOptionButton.cs:266-284).
                ctx.Fail(AbortReason.HandlerFault, "abandon_multi_seat", "The abandon confirmation of Trial was answered YES in a multi-seat run.");
            }
            EventPatches.Count("events-trial-double-down-confirm:yes");
            ctx.RunManager.Abandon(); // RunSafely(AbandonInternal()): kills every player -> run.IsGameOver (Loss)
        }
        else
        {
            EventPatches.Count("events-trial-double-down-confirm:no");
        }
    }
}

// =========================================================================================================
//  BUNDLE: NChooseABundleSelectionScreen (Commands/CardSelectCmd.cs:657-698, Scroll Boxes)
// =========================================================================================================

/// <summary>
/// Pushed by the stand-in for NChooseABundleSelectionScreen.ShowScreen inside the real-game branch of
/// CardSelectCmd.FromChooseABundleScreen; the game awaits <see cref="Completion"/> through the stand-in for
/// CardsSelected(). Actions: CHOOSE_OPTION(bundle i) for every bundle; SKIP when canSkip (the screen's skip button
/// completes with an empty list, NChooseABundleSelectionScreen.cs:274-279). The game then runs SyncLocalChoice
/// (choice id reserved before the gate) and LogChoice exactly as with the real screen.
/// </summary>
internal sealed class BundleFrame : IPendingDecision
{
    private readonly IReadOnlyList<IReadOnlyList<CardModel>> _bundles;
    private readonly Decision _decision = new Decision();
    private readonly BundleContext _context;
    private readonly VirtualClient? _remote;
    private readonly uint _remoteChoiceId;

    public readonly TaskCompletionSource<IEnumerable<IReadOnlyList<CardModel>>> Completion = new TaskCompletionSource<IEnumerable<IReadOnlyList<CardModel>>>();

    public BundleFrame(Player player, IReadOnlyList<IReadOnlyList<CardModel>> bundles, bool canSkip)
    {
        _bundles = bundles;
        _context = new BundleContext { Player = player, Bundles = bundles, CanSkip = canSkip };
    }

    /// <summary>
    /// MULTI-SEAT (phase H5): the bundle screen of a REMOTE player. The game waits for that player's choice
    /// (CardSelectCmd.cs:681-693: WaitForRemoteChoice(...).AsIndex(), a negative index = nothing); the client's local
    /// branch syncs FromIndex(index of the chosen bundle), -1 for the skip button (:677-679).
    /// </summary>
    public BundleFrame(Player player, IReadOnlyList<IReadOnlyList<CardModel>> bundles, bool canSkip, VirtualClient remote, uint choiceId)
        : this(player, bundles, canSkip)
    {
        _remote = remote;
        _remoteChoiceId = choiceId;
    }

    public string Name => "events:bundle";

    public Decision? Build(RunContext ctx)
    {
        int seat = ctx.SeatOf(_context.Player);
        Decision d = _decision.Reset(DecisionKind.Bundle, this, actingSeat: seat, context: _context);
        for (int i = 0; i < _bundles.Count; i++)
        {
            d.Add(Verb.ChooseOption, seat, GameRef.OfBundle(_bundles[i], i));
        }
        if (_context.CanSkip)
        {
            d.Add(Verb.Skip, seat);
        }
        return d;
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        ctx.Pop(this);
        if (_remote != null)
        {
            _remote.SendChoice(_remoteChoiceId, MegaCrit.Sts2.Core.GameActions.PlayerChoiceResult.FromIndex(a.Verb == Verb.Skip ? -1 : a.Source.Index));
            return;
        }
        if (a.Verb == Verb.Skip)
        {
            Completion.SetResult(Array.Empty<IReadOnlyList<CardModel>>());
        }
        else
        {
            Completion.SetResult(new[] { _bundles[a.Source.Index] });
        }
    }
}
