using System;
using System.Buffers.Binary;
using SpireHost.Driver;
using SpireHost.Engine;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Schema;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs.Protocol;

/// <summary>A decoded Python -> host command (obs-schema-v1 7.1).</summary>
public struct Command
{
    public int Type; // CmdType.*, 0 = malformed
    public string? Error;

    // RESET
    public int NSeats, ViewMode, ViewSeat, Ascension, Flags, MaxDecisions;

    /// <summary>RESET word 15: reserved, 0 in production (errata P-2). Non-zero = a DEBUG option (forced HP), see ServeOptions.AllowDebugOptions.</summary>
    public int DebugWord;
    public int[] Characters;
    public bool SeedPresent;
    public ulong Seed;

    // STEP
    public int StepId, ActionIndex;

    private static int R(ReadOnlySpan<byte> p, int word) => BinaryPrimitives.ReadInt32LittleEndian(p.Slice(word * 4));

    public static Command Parse(ReadOnlySpan<byte> p)
    {
        Command c = default;
        c.Characters = Array.Empty<int>();
        if (p.Length < 8 || p.Length % 4 != 0)
        {
            c.Error = $"command of {p.Length} bytes";
            return c;
        }
        int magic = BinaryPrimitives.ReadInt32LittleEndian(p);
        int type = BinaryPrimitives.ReadInt32LittleEndian(p.Slice(4));
        if (magic != SchemaV1.CommandMagic)
        {
            c.Error = $"bad command magic 0x{magic:X8}";
            return c;
        }
        switch (type)
        {
            case CmdType.Reset when p.Length == 64:
                c.NSeats = R(p, 2);
                c.ViewMode = R(p, 3);
                c.ViewSeat = R(p, 4);
                c.Ascension = R(p, 5);
                if (c.NSeats < 1 || c.NSeats > 4)
                {
                    c.Error = "RESET with n_seats=" + c.NSeats;
                    return c;
                }
                c.Characters = new int[c.NSeats];
                for (int s = 0; s < c.NSeats; s++)
                {
                    c.Characters[s] = R(p, 6 + s);
                }
                c.SeedPresent = R(p, 10) != 0;
                c.Seed = (uint)R(p, 11) | ((ulong)(uint)R(p, 12) << 32);
                c.Flags = R(p, 13);
                c.MaxDecisions = R(p, 14);
                c.DebugWord = R(p, 15);
                if (c.ViewMode == 0 ? c.ViewSeat != -1 : c.ViewMode != 1 || c.ViewSeat < 0 || c.ViewSeat >= c.NSeats)
                {
                    c.Error = $"RESET with view_mode={c.ViewMode} view_seat={c.ViewSeat}";
                    return c;
                }
                c.Type = CmdType.Reset;
                return c;
            case CmdType.Step when p.Length == 16:
                c.StepId = R(p, 2);
                c.ActionIndex = R(p, 3);
                c.Type = CmdType.Step;
                return c;
            case CmdType.Quit when p.Length == 8:
                c.Type = CmdType.Quit;
                return c;
            default:
                c.Error = $"unknown command type {type} / length {p.Length}";
                return c;
        }
    }
}

/// <summary>
/// Opt-in extension of the serve protocol: the search build's FORK command and fork children (SpireHost.Search,
/// notes/search/protocol.md). A production server has none (<see cref="ProtocolServer.Extension"/> is null) and behaves
/// exactly as without this interface.
/// </summary>
public interface IServeExtension
{
    /// <summary>
    /// A message of a command type the protocol does not define arrived while an OBS is pending. Returns -1 = not this
    /// extension's (the server answers ERROR 3 + resend, as without an extension), 0 = handled (the server reads the next
    /// command, nothing is resent), &gt; 0 = an ERROR code (the server sends ERROR + the same OBS again).
    /// </summary>
    int OnPendingCommand(ProtocolServer server, ReadOnlySpan<byte> message, Decision decision);

    /// <summary>The OBS of a policy decision is encoded (<see cref="ProtocolServer.PendingObs"/>) and about to be sent.</summary>
    void BeforeObs(ProtocolServer server, Decision decision);

    void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto);

    /// <summary>The episode is over and the run is still alive (before teardown).</summary>
    void OnTerminalState(RunContext ctx, EpisodeResult provisional);

    /// <summary>The episode's final frame (<see cref="ObsEncoder.Payload"/>) is about to be sent / was sent.</summary>
    void BeforeFinalFrame(ProtocolServer server, EpisodeResult result);

    void AfterFinalFrame(ProtocolServer server);

    /// <summary>Inside an episode the command stream ended (QUIT, end of stream) or broke.</summary>
    void OnStreamEnd(ProtocolServer server, bool broken, string why);

    /// <summary>True: a RESET while an OBS is pending is answered with ERROR 3 + resend instead of abandoning the episode.</summary>
    bool RefusesReset { get; }
}

/// <summary>Host-side defaults of the protocol server (command line of --serve).</summary>
public sealed class ServeOptions
{
    /// <summary>Per-episode policy-decision cap -> TERMINAL(LOSS, terminal_reason = DECISION_CAP). RESET word 14 overrides when &gt; 0.</summary>
    public long MaxPolicyDecisions = 100_000;
    public int MaxCombatTurns = 500;
    public long DecisionCap = 400_000;

    /// <summary>
    /// The process was started with --allow-debug-options (ruling A4). Only then are debug options honoured:
    ///   * <see cref="DebugForcedHp"/> (command line --hp N: every episode), and
    ///   * RESET word 15 != 0 (forced HP of that episode; overrides the command line).
    /// Without the flag --hp is a startup error and a RESET whose word 15 is not 0 is answered with ERROR 3 (bad
    /// message) and starts no episode: a production server can never be put into a debug configuration by a client.
    /// </summary>
    public bool AllowDebugOptions;

    /// <summary>DEBUG (needs AllowDebugOptions): forced HP like --fuzz-runs --hp, for every episode of this process.</summary>
    public int DebugForcedHp;
    public bool Verbose;
}

/// <summary>
/// The binary protocol server (--serve): an <see cref="IPolicy"/> + <see cref="IEpisodeObserver"/> that runs on the
/// engine thread. Per episode: RESET -> EpisodeSpec / EpisodeOptions -> HostSession.RunEpisode; every policy decision is
/// encoded, written, and answered by a blocking STEP read inside <see cref="Decide"/>; the final state is encoded in
/// OnTerminalState (run still alive) and sent with the final result in OnEpisodeEnd.
///
/// Protocol rulings (notes/schema-errata-v1.md section P):
///  * EPISODE_ID starts at 1 and increments per accepted RESET. STEP_ID = Counters.PolicyDecisions (0-based, every OBS
///    frame +1, auto-advanced decisions excluded, a re-sent frame keeps its STEP_ID). TERMINAL STEP_ID = number of
///    policy decisions of the episode (= last OBS STEP_ID + 1).
///  * Bad STEP (wrong step_id / index outside [0, N_ACTIONS) / malformed message while an OBS is pending): ERROR
///    (code 2 / 1 / 3) followed by the SAME OBS bytes again. A STEP outside an episode or a malformed RESET: ERROR, no
///    resend. A RESET while an OBS is pending abandons the episode (no TERMINAL for it) and starts the new one. QUIT or
///    end of stream at a message boundary: the episode is abandoned and the process exits 0. End of stream inside a
///    message or a length prefix above 64 KiB: the stream cannot be resynchronised -> exit 1.
/// </summary>
public sealed class ProtocolServer : IPolicy, IEpisodeObserver
{
    private readonly HostSession _session;
    private IMessageTransport _t;
    private readonly ObsEncoder _enc;
    private readonly ServeOptions _opt;
    private readonly byte[] _errorBuf = new byte[SchemaV1.DataOffsetWords * 4];

    private int _episodeId;
    private int _stepId;
    private int _autoSteps;
    private long _lastAuto;
    private Command? _pending;
    private bool _quit;
    private bool _abandoned;
    private bool _terminalEncoded;
    private RunContext? _ctx;
    private int _nSeats = 1;

    public long Episodes, Decisions, Errors, Abandoned, Wins, Losses, Aborts, Truncations, DebugResetsRefused, DebugEpisodes;

    /// <summary>
    /// DIAGNOSTICS ONLY (notes/speed/host.md): SPIREHOST_EXACT_CHECK=1 encodes every OBS frame a second time with a reference
    /// encoder in EncoderOptions.ExactGameCalls mode (the game's own methods instead of the hook caches) and compares the
    /// bytes (HOST_US aside), and compares DeepSignature of the run before and after both encodings (side effects). The
    /// frame that is sent is always the production encoder's.
    /// </summary>
    private static readonly bool s_exactCheck = Environment.GetEnvironmentVariable("SPIREHOST_EXACT_CHECK") == "1";
    private ObsEncoder? _ref;
    public long ExactCompared, ExactMismatches, SideEffects;
    public string? FirstMismatch;

    /// <summary>The search build's protocol extension (--serve --search); null in every other server.</summary>
    public IServeExtension? Extension;

    public ObsEncoder Encoder => _enc;

    public int EpisodeId => _episodeId;

    public int StepId => _stepId;

    /// <summary>The bytes of the OBS frame that is pending (the last one encoded; valid until the next encode).</summary>
    public ReadOnlySpan<byte> PendingObs => _enc.Payload;

    /// <summary>Encodes the pending decision again (same EPISODE_ID / STEP_ID / AUTO_STEPS): a fork child after re-randomisation.</summary>
    public void EncodePendingAgain(Decision decision)
    {
        _enc.EncodeDecision(decision, _episodeId, _stepId, _autoSteps);
    }

    /// <summary>A fork child continues the episode on its own socket.</summary>
    public void SwitchTransport(IMessageTransport transport)
    {
        _t = transport;
    }

    /// <summary>Writes one message as it is (FORKED, HELLO, FIGHT_END, a pending OBS) on the current transport.</summary>
    public void SendMessage(ReadOnlySpan<byte> payload)
    {
        Send(payload);
    }

    public ProtocolServer(HostSession session, IMessageTransport transport, ObsEncoder encoder, ServeOptions options)
    {
        _session = session;
        _t = transport;
        _enc = encoder;
        _opt = options;
    }

    /// <summary>The serve loop; returns the process exit code.</summary>
    public int Serve()
    {
        while (!_quit)
        {
            Command cmd;
            if (_pending.HasValue)
            {
                cmd = _pending.Value;
                _pending = null;
            }
            else
            {
                ReadOnlyMemory<byte>? msg;
                int prof = HostProfile.On ? HostProfile.Enter(HostProfile.ReadWait) : 0;
                try
                {
                    msg = _t.Read();
                }
                catch (TransportBroken e)
                {
                    Console.Error.WriteLine("[obs] serve: " + e.Message + " - exiting");
                    return 1;
                }
                if (HostProfile.On)
                {
                    HostProfile.Enter(prof);
                }
                if (msg == null)
                {
                    return 0; // peer closed the stream
                }
                cmd = Command.Parse(msg.Value.Span);
            }
            switch (cmd.Type)
            {
                case CmdType.Quit:
                    return 0;
                case CmdType.Reset:
                    if (!RunEpisode(cmd))
                    {
                        return 1; // the byte stream broke inside the episode (cannot be resynchronised)
                    }
                    break;
                case CmdType.Step:
                    SendError(ErrorCode.BadMessage, _episodeId, cmd.StepId); // no episode is waiting for an action
                    break;
                default:
                    if (_opt.Verbose)
                    {
                        Console.Error.WriteLine("[obs] serve: bad message: " + cmd.Error);
                    }
                    SendError(ErrorCode.BadMessage, _episodeId, 0);
                    break;
            }
        }
        return 0;
    }

    /// <summary>Plays one episode; false if the transport broke.</summary>
    private bool RunEpisode(Command reset)
    {
        if (!TryBuildSpec(reset, out EpisodeSpec? spec, out string? why))
        {
            Console.Error.WriteLine("[obs] serve: RESET rejected: " + why);
            SendError(ErrorCode.BadMessage, _episodeId, 0);
            return true;
        }
        _episodeId++;
        Episodes++;
        _stepId = 0;
        _lastAuto = 0;
        _abandoned = false;
        _terminalEncoded = false;
        _nSeats = reset.NSeats;
        _enc.Options = EncoderOptions.FromResetFlags(reset.Flags, reset.ViewMode, reset.ViewSeat);
        int forcedHp = reset.DebugWord != 0 ? reset.DebugWord : _opt.DebugForcedHp; // both only reachable with AllowDebugOptions
        if (forcedHp != 0)
        {
            DebugEpisodes++;
        }
        EpisodeOptions o = new EpisodeOptions
        {
            AutoSingle = (reset.Flags & ResetFlag.AutoSingle) != 0,
            AutoGold = true,
            AllowAbandon = false,
            MaxPolicyDecisions = reset.MaxDecisions > 0 ? reset.MaxDecisions : _opt.MaxPolicyDecisions,
            MaxCombatTurns = _opt.MaxCombatTurns,
            DecisionCap = _opt.DecisionCap,
            DebugForcedHp = forcedHp,
            // Multi-seat (erratum H5; no effect with one seat): team votes unless the RESET asks for every seat's vote,
            // UNDO_END_TURN only when the RESET asks for it.
            TeamVotes = (reset.Flags & ResetFlag.AskEverySeat) == 0,
            OfferUndoEndTurn = (reset.Flags & ResetFlag.OfferUndoEndTurn) != 0,
        };
        _session.RunEpisode(spec!, this, o, this);
        return !_brokenTransport;
    }

    private bool _brokenTransport;

    private bool TryBuildSpec(Command c, out EpisodeSpec? spec, out string? why)
    {
        spec = null;
        // Ruling A4: a debug option in a RESET is honoured only when the process was started with --allow-debug-options.
        if (c.DebugWord != 0 && !_opt.AllowDebugOptions)
        {
            DebugResetsRefused++;
            why = "RESET word 15 (debug option) is " + c.DebugWord + " but the server was not started with --allow-debug-options";
            return false;
        }
        if (c.DebugWord < 0 || c.DebugWord > 9_999_999)
        {
            why = "RESET word 15 (debug forced HP) out of range: " + c.DebugWord;
            return false;
        }
        // Undefined flag bits are refused (bits 0-5 are defined, schema 7.1 / errata E-2, P-11; bits 6-7 only for a multi-seat
        // RESET, erratum H5; bit 8 FORECAST for 1-4 seats, notes/forecast/contract.md 2.1): a client that believes in an
        // option this host does not have must hear about it.
        // Bit 9 CALC (notes/calc/contract.md 2.1) for 1-4 seats, only together with bit 4 TARGET_PREVIEWS.
        int definedFlags = (c.NSeats > 1 ? 0xFF : 0x3F) | ResetFlag.Forecast | ResetFlag.Calc;
        if ((c.Flags & ~definedFlags) != 0)
        {
            why = "RESET flags 0x" + c.Flags.ToString("X") + " contain undefined bits";
            return false;
        }
        if ((c.Flags & ResetFlag.Calc) != 0 && (c.Flags & ResetFlag.TargetPreviews) == 0)
        {
            why = "RESET flags 0x" + c.Flags.ToString("X") + ": CALC (bit 9) needs TARGET_PREVIEWS (bit 4)";
            return false;
        }
        if (c.NSeats > 1 && c.ViewMode != 0)
        {
            // Phase H5: 2-4 seats are served to ONE controller in TEAM view. The SEAT view (one stream per seat) is not built.
            why = "n_seats=" + c.NSeats + " with view_mode=" + c.ViewMode + " (multi-seat runs are served in TEAM view only)";
            return false;
        }
        string[] chars = new string[c.NSeats];
        VocabSet.Category cat = _enc.Vocab["character"];
        for (int s = 0; s < c.NSeats; s++)
        {
            int id = c.Characters[s];
            if (id < 2 || id >= cat.Count || cat.Retired.Contains(cat.Entries[id]))
            {
                why = $"character vocab id {id} of seat {s} is not a character";
                return false;
            }
            chars[s] = cat.Entries[id];
        }
        string? seed = null;
        if (c.SeedPresent)
        {
            if (!SeedCodec.TryDecode(c.Seed, out seed))
            {
                why = "seed word " + c.Seed + " is not a 12-character game seed (base-34, < 34^12)";
                return false;
            }
        }
        if (c.Ascension < 0 || c.Ascension > 20)
        {
            why = "ascension " + c.Ascension;
            return false;
        }
        spec = new EpisodeSpec { Seed = seed, Characters = chars, Ascension = c.Ascension };
        why = null;
        return true;
    }

    // ---------------------------------------------------------------------------------------------- IPolicy

    public int Decide(Decision decision)
    {
        RunContext ctx = decision.Episode;
        _stepId = (int)ctx.Counters.PolicyDecisions;
        long auto = ctx.Counters.AutoSingleTotal + ctx.Counters.AutoGold;
        int autoSteps = (int)(auto - _lastAuto);
        _lastAuto = auto;
        _autoSteps = autoSteps;
        int kind = (int)decision.Kind;
        int prof = 0;
        long encodeStart = 0;
        if (HostProfile.On)
        {
            prof = HostProfile.Enter(HostProfile.Encode);
            HostProfile.FrameStart(kind);
            encodeStart = HostProfile.LastCpu;
        }
        ulong sig0 = s_exactCheck ? SpireHost.Obs.Tests.DeepSignature.Compute(ctx.Run) : 0;
        _enc.EncodeDecision(decision, _episodeId, _stepId, autoSteps);
        if (s_exactCheck)
        {
            ExactCheck(decision, autoSteps, sig0);
        }
        Decisions++;
        if (HostProfile.On)
        {
            HostProfile.Enter(HostProfile.Write);
            HostProfile.FrameEncoded(kind, encodeStart);
        }
        Extension?.BeforeObs(this, decision);
        Send(_enc.Payload);
        if (HostProfile.On)
        {
            HostProfile.Enter(HostProfile.ReadWait);
        }
        int n = decision.Actions.Count;
        while (true)
        {
            ReadOnlyMemory<byte>? msg;
            try
            {
                msg = _t.Read();
            }
            catch (TransportBroken e)
            {
                Console.Error.WriteLine("[obs] serve: " + e.Message);
                Extension?.OnStreamEnd(this, true, e.Message);
                _brokenTransport = true;
                _quit = true;
                _abandoned = true;
                throw new PolicyStopException("transport broken");
            }
            if (msg == null)
            {
                Extension?.OnStreamEnd(this, false, "end of command stream");
                _quit = true;
                _abandoned = true;
                throw new PolicyStopException("end of command stream");
            }
            Command cmd = Command.Parse(msg.Value.Span);
            switch (cmd.Type)
            {
                case CmdType.Step:
                    if (cmd.StepId != _stepId)
                    {
                        Resend(ErrorCode.StaleStep);
                        continue;
                    }
                    if ((uint)cmd.ActionIndex >= (uint)n)
                    {
                        Resend(ErrorCode.BadIndex);
                        continue;
                    }
                    if (HostProfile.On)
                    {
                        HostProfile.Enter(prof);
                        HostProfile.Answered(kind);
                    }
                    return cmd.ActionIndex;
                case CmdType.Reset when Extension != null && Extension.RefusesReset:
                    Resend(ErrorCode.BadMessage);
                    continue;
                case CmdType.Reset:
                    _pending = cmd;
                    _abandoned = true;
                    Abandoned++;
                    throw new PolicyStopException("abandoned by RESET");
                case CmdType.Quit:
                    Extension?.OnStreamEnd(this, false, "QUIT");
                    _quit = true;
                    _abandoned = true;
                    throw new PolicyStopException("QUIT");
                default:
                    if (Extension != null)
                    {
                        int r = Extension.OnPendingCommand(this, msg.Value.Span, decision);
                        if (r == 0)
                        {
                            continue;
                        }
                        if (r > 0)
                        {
                            Resend(r);
                            continue;
                        }
                    }
                    if (_opt.Verbose)
                    {
                        Console.Error.WriteLine("[obs] serve: bad message during step: " + cmd.Error);
                    }
                    Resend(ErrorCode.BadMessage);
                    continue;
            }
        }
    }

    private void ExactCheck(Decision decision, int autoSteps, ulong sig0)
    {
        RunContext ctx = decision.Episode;
        ulong sig1 = SpireHost.Obs.Tests.DeepSignature.Compute(ctx.Run);
        _ref ??= ObsRuntime.NewEncoder(ObsModule.Current);
        _ref.Options = _enc.Options.Clone();
        _ref.Options.ExactGameCalls = true;
        _ref.EncodeDecision(decision, _episodeId, _stepId, autoSteps);
        ulong sig2 = SpireHost.Obs.Tests.DeepSignature.Compute(ctx.Run);
        ExactCompared++;
        if (sig0 != sig1 || sig1 != sig2)
        {
            SideEffects++;
        }
        if (!SpireHost.Obs.Tests.EncodeChecker.SameIgnoringHostUs(_enc.Payload.ToArray(), _ref.Payload.ToArray()))
        {
            ExactMismatches++;
            FirstMismatch ??= $"episode {_episodeId} step {_stepId} kind {decision.Kind}";
        }
    }

    private void Resend(int code)
    {
        Errors++;
        SendError(code, _episodeId, _stepId);
        Send(_enc.Payload); // the OBS buffer is untouched by error frames
    }

    private void Send(ReadOnlySpan<byte> payload)
    {
        try
        {
            _t.Write(payload);
        }
        catch (TransportBroken e)
        {
            Console.Error.WriteLine("[obs] serve: " + e.Message);
            Extension?.OnStreamEnd(this, true, e.Message);
            _brokenTransport = true;
            _quit = true;
            _abandoned = true;
            throw new PolicyStopException("transport broken");
        }
    }

    /// <summary>ERROR frame (errata P-3): header + GLOBAL + directory of 16 empty tables, global[0] = code, RESULT 0, FLAGS 0.</summary>
    private void SendError(int code, int episodeId, int stepId)
    {
        ErrorFrame.Write(_errorBuf, code, episodeId, stepId, _enc.Vocab.Hash, _enc.CatalogHash);
        try
        {
            _t.Write(_errorBuf);
        }
        catch (TransportBroken e)
        {
            Console.Error.WriteLine("[obs] serve: " + e.Message);
            _brokenTransport = true;
            _quit = true;
        }
    }

    // ---------------------------------------------------------------------------------------------- IEpisodeObserver

    public void OnEpisodeStart(RunContext ctx)
    {
        _ctx = ctx; // (the public-history counters of GLOBAL 44-48 are the driver core's: EpisodeCounters, obs request O3)
    }

    public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
    {
        Extension?.OnDecision(ctx, decision, chosen, auto);
    }

    public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
    {
        Extension?.OnTerminalState(ctx, provisional);
        if (_abandoned)
        {
            return;
        }
        long auto = ctx.Counters.AutoSingleTotal + ctx.Counters.AutoGold;
        int prof = 0;
        if (HostProfile.On)
        {
            prof = HostProfile.Enter(HostProfile.Encode);
            HostProfile.FrameStart((int)DecisionKind.Terminal);
        }
        try
        {
            _enc.EncodeTerminal(ctx, _episodeId, (int)ctx.Counters.PolicyDecisions, (int)(auto - _lastAuto));
            _terminalEncoded = true;
        }
        catch (Exception e)
        {
            // The episode is over; its last state may be inconsistent (an abort in the middle of an action). Fall back to
            // a bare TERMINAL frame (sent in OnEpisodeEnd) and report - never silently.
            _terminalEncoded = false;
            Console.Error.WriteLine("[obs] serve: final state not encodable (" + e.GetType().Name + ": " + e.Message + "); sending a bare TERMINAL frame");
        }
        if (HostProfile.On)
        {
            HostProfile.Enter(prof);
        }
    }

    public void OnEpisodeEnd(EpisodeResult result)
    {
        _ctx = null;
        if (_abandoned || _brokenTransport)
        {
            return;
        }
        int steps = (int)result.Counters.PolicyDecisions;
        if (!_terminalEncoded)
        {
            long auto = result.Counters.AutoSingleTotal + result.Counters.AutoGold;
            _enc.EncodeBareTerminal(_episodeId, steps, (int)(auto - _lastAuto), _nSeats);
        }
        (int res, int reason) = Outcome(result);
        switch (res)
        {
            case Result.Win:
                Wins++;
                break;
            case Result.Aborted:
                Aborts++;
                break;
            default:
                if (reason == TerminalReason.Death)
                {
                    Losses++;
                }
                else
                {
                    Truncations++;
                }
                break;
        }
        FrameWriter w = _enc.Writer;
        w.PatchHeader(Hd.Result, res);
        w.PatchHeader(Hd.StepId, steps);
        w.PatchGlobal(Gl.TerminalReasonIdx, reason);
        w.PatchGlobal(Gl.AbortReasonIdx, result.Outcome == EpisodeOutcome.Aborted ? (int)(result.Abort?.Reason ?? AbortReason.HostBug) : 0);
        w.PatchGlobal(Gl.PolicyDecisionsIdx, steps);
        if (result.Outcome == EpisodeOutcome.Aborted)
        {
            // diagnostics only (stderr is never fed to a model): the host-side tuple that reproduces the episode
            Console.Error.WriteLine($"[obs] serve: episode {_episodeId} aborted: {result.Abort?.Signature} - {result.Abort?.Message} | seed {result.Seed} {result.Character} A{result.Ascension} step {steps}");
        }
        Extension?.BeforeFinalFrame(this, result);
        int prof = HostProfile.On ? HostProfile.Enter(HostProfile.Write) : 0;
        try
        {
            _t.Write(w.Payload);
        }
        catch (TransportBroken e)
        {
            Console.Error.WriteLine("[obs] serve: " + e.Message);
            Extension?.OnStreamEnd(this, true, e.Message);
            _brokenTransport = true;
            _quit = true;
        }
        if (HostProfile.On)
        {
            HostProfile.Enter(prof);
            HostProfile.EpisodeEnd(Decisions);
        }
        Extension?.AfterFinalFrame(this);
    }

    /// <summary>RESULT + GLOBAL[TERMINAL_REASON] of an episode result (errata P-5).</summary>
    public static (int result, int reason) Outcome(EpisodeResult r)
    {
        switch (r.Outcome)
        {
            case EpisodeOutcome.Win:
                return (Result.Win, TerminalReason.Win);
            case EpisodeOutcome.Loss:
                return (Result.Loss, TerminalReason.Death);
            case EpisodeOutcome.Truncated:
                return (Result.Loss, r.TruncatedBy switch
                {
                    "max_policy_decisions" => TerminalReason.DecisionCap,
                    "combat_turn_cap" => TerminalReason.CombatTurnCap,
                    _ => TerminalReason.OtherTruncation,
                });
            default:
                return (Result.Aborted, TerminalReason.Aborted);
        }
    }
}

/// <summary>The 640-byte ERROR frame (header + GLOBAL + directory; all 16 tables empty).</summary>
public static class ErrorFrame
{
    private static void W(Span<byte> buf, int word, int v) => BinaryPrimitives.WriteInt32LittleEndian(buf.Slice(word * 4), v);

    public static void Write(Span<byte> buf, int code, int episodeId, int stepId, int vocabHash, int catalogHash)
    {
        buf.Clear();
        W(buf, Hd.Magic, SchemaV1.FrameMagic);
        W(buf, Hd.SchemaVersion, SchemaV1.Version);
        W(buf, Hd.MsgType, MsgType.Error);
        W(buf, Hd.VocabHash, vocabHash);
        W(buf, Hd.CatalogHash, catalogHash);
        W(buf, Hd.EpisodeId, episodeId);
        W(buf, Hd.StepId, stepId);
        W(buf, Hd.FrameBytes, SchemaV1.DataOffsetWords * 4);
        W(buf, Hd.NTables, SchemaV1.NTables);
        W(buf, Hd.GlobalLen, SchemaV1.GlobalLen);
        W(buf, SchemaV1.GlobalOffsetWords + Gl.ErrorCode, code);
        for (int t = 0; t < SchemaV1.NTables; t++)
        {
            int d = SchemaV1.DirOffsetWords + t * SchemaV1.DirStride;
            W(buf, d, t);
            W(buf, d + 1, 0);
            W(buf, d + 2, SchemaV1.NCols[t]);
            W(buf, d + 3, SchemaV1.DataOffsetWords * 4);
            W(buf, d + 4, unchecked((int)SchemaV1.FnvOffsetBasis));
        }
    }
}

/// <summary>RESET seed word &lt;-&gt; 12-character game seed (errata P-4: the Python seed_to_wire base-34 reading, SeedHelper alphabet).</summary>
public static class SeedCodec
{
    public const string Alphabet = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"; // SeedHelper._characters (Helpers/SeedHelper.cs:8)
    public const int Length = 12;

    public static bool TryDecode(ulong x, out string? seed)
    {
        seed = null;
        ulong limit = 1;
        for (int i = 0; i < Length; i++)
        {
            limit *= (ulong)Alphabet.Length;
        }
        if (x >= limit)
        {
            return false;
        }
        char[] c = new char[Length];
        for (int i = Length - 1; i >= 0; i--)
        {
            c[i] = Alphabet[(int)(x % (ulong)Alphabet.Length)];
            x /= (ulong)Alphabet.Length;
        }
        seed = new string(c);
        return true;
    }

    public static ulong Encode(string seed)
    {
        string s = seed.ToUpperInvariant().Replace('O', '0').Replace('I', '1').Trim();
        if (s.Length != Length)
        {
            throw new ArgumentException("not a 12-character seed: " + seed);
        }
        ulong x = 0;
        foreach (char ch in s)
        {
            int i = Alphabet.IndexOf(ch);
            if (i < 0)
            {
                throw new ArgumentException("not a game seed character: " + ch);
            }
            x = x * (ulong)Alphabet.Length + (ulong)i;
        }
        return x;
    }
}
