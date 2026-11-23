using System;
using System.Collections.Generic;
using System.Diagnostics;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;
using SpireHost.Engine;
using SpireHost.Obs.Schema;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs.Encoding;

/// <summary>
/// obs-schema-v1 encoder: <see cref="Decision"/> (+ <c>decision.Episode</c>) -> frame bytes.
///
/// Inputs: the Decision, its typed context (Decision.Context, filled completely by the module that built the
/// decision) and the white-listed public game API - nothing else of the builder (no side table, no reference to
/// another module; ruling D6). HOST-ONLY context members (CrystalContext.Minigame) are never read.
///
/// Fair-play boundary: only the white-listed public API of schema 2.2 is read (see the per-table comments);
/// the run seed, RNG state, draw-pile order, unrevealed rewards and the run signature are never touched. Monster state
/// machines are read only by the move forecast (RESET flag FORECAST, ObsEncoder.Forecast.cs; owner ruling 2026-11-17:
/// fixed move patterns may be an input, hidden state may not): it reads them, never writes them, and touches no RNG. The calc tables (RESET flag CALC, ObsEncoder.Calc.cs) are
/// computed from the staged rows plus the getters TargetType / HasTurnEndInHandEffect of hand cards. Every collection is emitted either in its public UI order (hand, relic bar, potion belt, power
/// icons, creature slots, option buttons) or in the canonical multiset order of schema 5.2.3.
///
/// Row order and the driver's canonical action order (DRIVER-API.md 6.4): CREATURES rows are emitted in
/// CombatState.Creatures order (allies, then enemies) so that a creature's row == its GameRef ordinal; HAND rows
/// in hand order; CHOICE / OFFER rows in candidate-ordinal order; POTIONS in (seat, slot) order; MAP_NODES in
/// (row, col) order with the boss nodes last; OPTIONS in option-index order. Hence the ACTIONS rows, written in
/// the driver's order, are also ascending in (verb, seat, src_table, src_row, tgt_row, aux) - checked per frame
/// (<see cref="EncoderStats.ActionOrderViolations"/>).
///
/// Side effects: the encoder calls three game methods that write state (schema 2.1-4, DECISIONS-H1 F.8):
/// CardModel.UpdateDynamicVarPreview (PreviewValue / EnchantedValue of the card's DynamicVars), the upgrade
/// preview clone (CardScope.CloneCard + UpgradeInternal, removed again with RemoveCard), and LocalContext.NetId
/// (per-seat views). All three are restored before the encoder returns: the DynamicVar values are saved and
/// written back, the clone is removed, NetId is reset. The --selftest "side effects" case checks this.
/// </summary>
public sealed partial class ObsEncoder
{
    private readonly VocabSet _v;
    private readonly Templates _tpl;
    private readonly FrameWriter _w = new FrameWriter();

    public EncoderOptions Options = new EncoderOptions();
    public readonly EncoderStats Stats = new EncoderStats();

    /// <summary>Frames encoded by all encoders of this process (module counter obs.frames_encoded).</summary>
    public static long FramesEncodedTotal;
    public int CatalogHash;

    // ---- per-frame state ----
    private RunContext _ctx = null!;
    private RunState _run = null!;
    private Decision? _d;
    private CombatState? _cs;
    private bool _inCombat;
    private int _nSeats;
    private bool _hasUpgradePreview;
    private bool _hasTargetPreview;

    private readonly Dictionary<Creature, int> _creatureRow = new Dictionary<Creature, int>(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<CardModel, int> _cardRow = new Dictionary<CardModel, int>(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<MapPoint, int> _mapRow = new Dictionary<MapPoint, int>(ReferenceEqualityComparer.Instance);
    private readonly int[] _seatCreatureRow = new int[4];
    private readonly int[] _potionRow = new int[4 * 16];
    private int[] _optRow = new int[64];
    private int _optRefKind;
    private int[] _choiceRow = new int[64];
    private int[] _offerCardRow = new int[64];
    private int[] _cellRow = new int[128];

    private readonly HookCache _hooks = new HookCache();

    public ObsEncoder(VocabSet vocab, Templates templates)
    {
        _v = vocab;
        _tpl = templates;
    }

    public FrameWriter Writer => _w;

    public VocabSet Vocab => _v;

    /// <summary>The bytes of the last encoded frame (valid until the next Encode*).</summary>
    public ReadOnlySpan<byte> Payload => _w.Payload;

    // =====================================================================================================
    //  Entry points
    // =====================================================================================================

    /// <summary>Encodes an OBS frame for a decision point (engine thread; the engine is quiescent).</summary>
    public void EncodeDecision(Decision d, int episodeId, int stepId, int autoSteps)
    {
        long t0 = Stopwatch.GetTimestamp();
        Begin(d.Episode, d);
        EncodeBody(d.Kind, d.SubKind, d.ActingSeat);
        EncodeActions(d);
        Lap(7);
        Finish(MsgType.Obs, episodeId, stepId, autoSteps, Result.InProgress, t0);
    }

    /// <summary>Encodes the TERMINAL frame (final observation, no actions). RESULT and the terminal GLOBAL fields are patched by the caller.</summary>
    public void EncodeTerminal(RunContext ctx, int episodeId, int stepId, int autoSteps)
    {
        long t0 = Stopwatch.GetTimestamp();
        Begin(ctx, null);
        EncodeBody(DecisionKind.Terminal, 0, -1);
        Finish(MsgType.Terminal, episodeId, stepId, autoSteps, Result.Loss, t0);
        Stats.Terminal++;
    }

    /// <summary>A TERMINAL frame without any game state (the episode never started, or its final state could not be read).</summary>
    public void EncodeBareTerminal(int episodeId, int stepId, int autoSteps, int nSeats)
    {
        _w.Clear();
        _w.Global[Gl.Kind] = (int)DecisionKind.Terminal;
        _w.Global[Gl.NSeats] = nSeats;
        _w.Global[Gl.ViewMode] = Options.ViewMode;
        _w.Global[Gl.ViewSeat] = Options.ViewSeat;
        _w.Global[Gl.ActingSeat] = -1;
        _w.Global[Gl.LastWonTotalFloor] = -1;
        SetHeader(MsgType.Terminal, episodeId, stepId, autoSteps, Result.Aborted);
        _w.Header[Hd.Flags] = FrameFlag.Terminal | (Options.ViewMode == 0 ? FrameFlag.TeamView : 0);
        _w.Build(Options.Forecast, Options.Calc);
    }

    /// <summary>ERROR frame (schema 7.3, errata-v1 E-3): empty tables, global[0] = code.</summary>
    public void EncodeError(int code, int episodeId, int stepId)
    {
        _w.Clear();
        _w.Global[Gl.ErrorCode] = code;
        SetHeader(MsgType.Error, episodeId, stepId, 0, Result.InProgress);
        _w.Build(forecast: false, calc: false);
    }

    private void SetHeader(int msg, int episodeId, int stepId, int autoSteps, int result)
    {
        _w.Header[Hd.MsgType] = msg;
        _w.Header[Hd.VocabHash] = _v.Hash;
        _w.Header[Hd.CatalogHash] = CatalogHash;
        _w.Header[Hd.EpisodeId] = episodeId;
        _w.Header[Hd.StepId] = stepId;
        _w.Header[Hd.Result] = result;
        _w.Header[Hd.AutoSteps] = autoSteps;
    }

    private void Begin(RunContext ctx, Decision? d)
    {
        _w.Clear();
        _ctx = ctx;
        _run = ctx.Run;
        _d = d;
        _nSeats = _run.Players.Count;
        CombatManager cm = CombatManager.Instance;
        _inCombat = cm.IsInProgress && _run.CurrentRoom is CombatRoom;
        _cs = _inCombat ? ((CombatRoom)_run.CurrentRoom!).CombatState : null;
        _hasUpgradePreview = false;
        _hasTargetPreview = false;
        _hooks.Exact = Options.ExactGameCalls;
        _hooks.Begin(_cs);
        _creatureRow.Clear();
        _cardRow.Clear();
        _mapRow.Clear();
        Array.Fill(_seatCreatureRow, -1);
        Array.Fill(_potionRow, -1);
        _optRefKind = -1;
    }

    private void Finish(int msg, int episodeId, int stepId, int autoSteps, int result, long t0)
    {
        if (Options.Calc)
        {
            // after every v1 table (ACTIONS included) and the forecast are staged (notes/calc/contract.md 2.4)
            if (HostProfile.On)
            {
                HostProfile.EncLap(msg == MsgType.Obs ? 7 : 6);
            }
            EncodeCalc();
        }
        SetHeader(msg, episodeId, stepId, autoSteps, result);
        int flags = 0;
        if (msg == MsgType.Terminal)
        {
            flags |= FrameFlag.Terminal;
        }
        if (_w.Count(T.MapNodes) > 0)
        {
            flags |= FrameFlag.HasMap;
        }
        if (_inCombat && (Options.DeckInCombat))
        {
            flags |= FrameFlag.DeckInCombat;
        }
        if (Options.ViewMode == 0)
        {
            flags |= FrameFlag.TeamView;
        }
        if (_hasUpgradePreview)
        {
            flags |= FrameFlag.UpgradePreview;
        }
        if (_hasTargetPreview)
        {
            flags |= FrameFlag.TargetPreview;
        }
        _w.Header[Hd.Flags] = flags;
        long us = (Stopwatch.GetTimestamp() - t0) * 1_000_000 / Stopwatch.Frequency;
        _w.Header[Hd.HostUs] = (int)Math.Min(int.MaxValue, us);
        if (Options.SectionTiming)
        {
            _tick = Stopwatch.GetTimestamp();
        }
        _w.Build(Options.Forecast, Options.Calc);
        Lap(8);
        if (_w.Overflowed)
        {
            Stats.Overflows++;
        }
        long total = (Stopwatch.GetTimestamp() - t0) * 1_000_000 / Stopwatch.Frequency;
        Stats.Frames++;
        FramesEncodedTotal++;
        Stats.TotalUs += total;
        if (total > Stats.MaxUs)
        {
            Stats.MaxUs = total;
        }
        _hooks.End();
        _ctx = null!;
        _run = null!;
        _d = null;
        _cs = null;
    }

    // =====================================================================================================
    //  Body: every table except ACTIONS
    // =====================================================================================================

    /// <summary>Cumulative Stopwatch ticks per encoder section (global, seats, creatures, cards, items, map, kind-specific, actions, build).</summary>
    public readonly long[] SectionTicks = new long[9];

    public static readonly string[] SectionNames = { "global", "seats", "creatures", "cards", "items", "map", "kind", "actions", "build" };

    /// <summary>HostProfile sections: SectionNames + the zones of CARDS (cards = the rest of EncodeCards).</summary>
    private static readonly string[] ProfileSectionNames =
        { "global", "seats", "creatures", "cards", "items", "map", "kind", "actions", "build", "cards.hand", "cards.draw", "cards.discard", "cards.exhaust", "cards.play", "cards.deck", "cards.choice", "forecast", "calc", "calc.route" };

    /// <summary>HostProfile section of <see cref="EncodeForecast"/> (split out of "creatures").</summary>
    private const int ProfileForecast = 16;

    /// <summary>HostProfile sections of <see cref="EncodeCalc"/> (RESET flag CALC): TURN + FOE, and ROUTE.</summary>
    private const int ProfileCalc = 17, ProfileCalcRoute = 18;

    private long _tick;

    private void Lap(int section)
    {
        if (HostProfile.On)
        {
            HostProfile.EncLap(section);
        }
        if (!Options.SectionTiming)
        {
            return;
        }
        long now = Stopwatch.GetTimestamp();
        SectionTicks[section] += now - _tick;
        _tick = now;
    }

    private void EncodeBody(DecisionKind kind, int subKind, int actingSeat)
    {
        if (HostProfile.On)
        {
            HostProfile.EncNames = ProfileSectionNames;
            HostProfile.EncLapStart();
        }
        if (Options.SectionTiming)
        {
            _tick = Stopwatch.GetTimestamp();
        }
        EncodeGlobal(kind, subKind, actingSeat);
        Lap(0);
        EncodeSeats();
        Lap(1);
        if (_inCombat)
        {
            EncodeCreatures();
        }
        Lap(2);
        EncodeCards(kind);
        Lap(3);
        EncodeOwnedRelicsAndPotions();
        if (_inCombat)
        {
            EncodeOrbsAndArrows();
        }
        Lap(4);
        if (!_inCombat || Options.MapInCombat || kind == DecisionKind.Map)
        {
            EncodeMap(kind);
        }
        Lap(5);
        EncodeKindSpecific(kind);
        Lap(6);
    }

    private int SeatOf(Player? p)
    {
        if (p == null)
        {
            return -1;
        }
        IReadOnlyList<Player> players = _run.Players;
        for (int i = 0; i < players.Count; i++)
        {
            if (ReferenceEquals(players[i], p))
            {
                return i;
            }
        }
        return -1;
    }

    private void EncodeGlobal(DecisionKind kind, int subKind, int actingSeat)
    {
        int[] g = _w.Global;
        RunState run = _run;
        g[Gl.Kind] = (int)kind;
        g[Gl.KindSub] = kind == DecisionKind.CardSelect || kind == DecisionKind.Treasure ? subKind : 0;
        g[Gl.NSeats] = _nSeats;
        g[Gl.ViewMode] = Options.ViewMode;
        g[Gl.ViewSeat] = Options.ViewSeat;
        g[Gl.ActingSeat] = actingSeat;
        g[Gl.Ascension] = run.AscensionLevel;
        g[Gl.GameMode] = Enum(Ev.GameMode, (int)run.GameMode);
        g[Gl.ActIndex] = run.CurrentActIndex;
        ActModel act = run.Act;
        g[Gl.ActId] = _v.Model("act", act);
        g[Gl.ActFloor] = run.ActFloor;
        g[Gl.TotalFloor] = run.TotalFloor;
        ActMap map = run.Map;
        g[Gl.MapRows] = MapRows(map);
        MapCoord? cur = run.CurrentMapCoord;
        g[Gl.CurRow] = cur.HasValue ? cur.Value.row : -1;
        g[Gl.CurCol] = cur.HasValue ? cur.Value.col : -1;
        AbstractRoom? room = run.CurrentRoom;
        g[Gl.RoomType] = room == null ? 0 : Enum(Ev.RoomType, (int)room.RoomType);
        g[Gl.BossEncounter] = _v.Model("encounter", act.BossEncounter);
        g[Gl.SecondBossEncounter] = _v.Model("encounter", act.SecondBossEncounter);
        // The Ancient is shown on the map's starting node only when that node is an Ancient (NMapScreen.cs:527-529).
        g[Gl.AncientEvent] = map.StartingMapPoint != null && map.StartingMapPoint.PointType == MapPointType.Ancient ? _v.Model("event", act.Ancient) : 0;
        g[Gl.InCombat] = _inCombat ? 1 : 0;
        if (_inCombat)
        {
            CombatState cs = _cs!;
            g[Gl.CombatRound] = cs.RoundNumber;
            g[Gl.CombatSide] = Enum(Ev.CombatSide, (int)cs.CurrentSide);
            int phaseSeat = actingSeat >= 0 ? actingSeat : 0;
            PlayerCombatState? pcs = phaseSeat < _nSeats ? run.Players[phaseSeat].PlayerCombatState : null;
            g[Gl.PlayerPhase] = pcs == null ? 0 : Enum(Ev.PlayerTurnPhase, (int)pcs.Phase);
        }
        int alive = 0;
        for (int s = 0; s < _nSeats; s++)
        {
            if (run.Players[s].Creature.IsAlive)
            {
                alive++;
            }
        }
        g[Gl.PlayersAlive] = alive;
        // Public history (errata E-23, GLOBAL 44-48): maintained by the driver core from CombatManager.CombatWon (obs request O3).
        EpisodeCounters h = _ctx.Counters;
        g[Gl.CombatsWon] = (int)h.CombatsWon;
        g[Gl.ElitesWon] = h.ElitesWon;
        g[Gl.BossesWon] = h.BossesWon;
        g[Gl.LastWonRoomType] = h.LastWonRoomType < 0 ? 0 : Enum(Ev.RoomType, h.LastWonRoomType);
        g[Gl.LastWonTotalFloor] = h.LastWonTotalFloor;
        if (_nSeats > 1)
        {
            // Erratum H5: what every player sees of the teammates' progress (the screen type shown on the teammate panel,
            // the votes). The seat whose own screen is being asked has the frame's kind. Single-seat frames keep 0 here.
            for (int s = 0; s < _nSeats; s++)
            {
                g[Gl.SeatScreen0 + s] = _d != null && s == actingSeat ? (int)kind : (int)_ctx.SeatScreen(s);
                g[Gl.SeatStatus0 + s] = _ctx.SeatStatusBits(s);
            }
        }
    }

    private static int MapRows(ActMap map)
    {
        int max = -1;
        foreach (MapPoint p in map.GetAllMapPoints())
        {
            if (p.coord.row > max)
            {
                max = p.coord.row;
            }
        }
        if (map.BossMapPoint != null && map.BossMapPoint.coord.row > max)
        {
            max = map.BossMapPoint.coord.row;
        }
        if (map.SecondBossMapPoint != null && map.SecondBossMapPoint.coord.row > max)
        {
            max = map.SecondBossMapPoint.coord.row;
        }
        return max + 1;
    }

    // =====================================================================================================
    //  SEATS (schema 5.1)
    // =====================================================================================================

    private void EncodeSeats()
    {
        CombatManager cm = CombatManager.Instance;
        if (_inCombat)
        {
            CountHistory();
        }
        for (int s = 0; s < _nSeats; s++)
        {
            Player p = _run.Players[s];
            Creature c = p.Creature;
            int r = _w.AddRow(T.Seats);
            Span<int> row = _w.Row(T.Seats, r);
            bool priv = Options.SeesPrivate(s);
            row[SeatC.Seat] = s;
            row[SeatC.CharacterId] = _v.Model("character", p.Character);
            row[SeatC.CreatureRow] = -1; // filled by EncodeCreatures
            row[SeatC.PrivateVisible] = priv ? 1 : 0;
            row[SeatC.Alive] = c.IsAlive ? 1 : 0;
            row[SeatC.Hp] = c.CurrentHp;
            row[SeatC.MaxHp] = c.MaxHp;
            row[SeatC.Gold] = p.Gold;
            row[SeatC.PotionSlots] = p.MaxPotionCount;
            row[SeatC.DeckCount] = p.Deck.Cards.Count;
            PlayerCombatState? pcs = _inCombat ? p.PlayerCombatState : null;
            if (pcs == null)
            {
                row[SeatC.Energy] = -1;
                row[SeatC.MaxEnergy] = p.MaxEnergy;
                continue;
            }
            row[SeatC.Block] = c.Block;
            row[SeatC.Energy] = pcs.Energy;
            row[SeatC.MaxEnergy] = _hooks.MaxEnergy(pcs);
            row[SeatC.Stars] = pcs.Stars;
            row[SeatC.OrbCapacity] = pcs.OrbQueue.Capacity;
            row[SeatC.HandCount] = pcs.Hand.Cards.Count;
            row[SeatC.DrawCount] = priv ? pcs.DrawPile.Cards.Count : -1;
            row[SeatC.DiscardCount] = priv ? pcs.DiscardPile.Cards.Count : -1;
            row[SeatC.ExhaustCount] = priv ? pcs.ExhaustPile.Cards.Count : -1;
            row[SeatC.ReadyToEndTurn] = cm.IsPlayerReadyToEndTurn(p) ? 1 : 0;
            row[SeatC.TurnNumber] = pcs.TurnNumber;
            int slots = 0, available = 0;
            foreach (var a in pcs._arrows)
            {
                slots++;
                if (a.IsAvailable)
                {
                    available++;
                }
            }
            row[SeatC.ArrowSlots] = slots;
            row[SeatC.ArrowsAvailable] = available;
            row[SeatC.MelodyBits] = MelodyBits(p);
            HistoryCounters(s, row);
        }
    }
}
