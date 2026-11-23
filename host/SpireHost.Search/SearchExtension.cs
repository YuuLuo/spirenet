using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Driver;
using SpireHost.Obs.Protocol;
using SpireHost.Obs.Schema;

namespace SpireHost.Search;

/// <summary>Message constants of the search protocol (notes/search/protocol.md).</summary>
public static class SearchWire
{
    public const int CmdFork = 4;
    public const int MsgForked = 4, MsgFightEnd = 5;
    public const int ModeResample = 0, ModeExact = 1;

    /// <summary>
    /// FORK word 7 (flags). Game: the child plays on to the end of the game (no FIGHT_END; its last message is the episode's
    /// TERMINAL frame). DieWithParent: the child gets SIGKILL when the process that forked it dies (PR_SET_PDEATHSIG).
    /// </summary>
    public const int FlagGame = 1, FlagDieWithParent = 2, KnownFlags = FlagGame | FlagDieWithParent;
    public const int ErrForkRefused = 4;
    public const uint HelloMagic = 0x4B465053; // "SPFK"
    public const int Version = 1;
    public const int MaxChildren = 256;
    public const int OutcomeWon = 1, OutcomeWiped = 2, OutcomeTruncated = 3, OutcomeAborted = 4;
    public const int FightEndSeatWords = 10, FightEndSeatOffset = 28;

    /// <summary>Child exit codes besides 0.</summary>
    public const int ExitConnect = 10, ExitFrameChanged = 11, ExitBrokenSocket = 12, ExitHygiene = 13, ExitInvariant = 14, ExitForkFailed = 15;
}

/// <summary>A parsed FORK command (protocol.md 2.1).</summary>
public sealed class ForkCommand
{
    public int StepId, N, Mode, Tag, Flags;
    public ulong[] Seeds = Array.Empty<ulong>();
    public byte[] Path = Array.Empty<byte>();

    private static int Word(ReadOnlySpan<byte> p, int w) => BinaryPrimitives.ReadInt32LittleEndian(p.Slice(w * 4));

    /// <summary>null = not a FORK; otherwise the command, or <paramref name="error"/> set when it is malformed.</summary>
    public static ForkCommand? Parse(ReadOnlySpan<byte> p, out string? error)
    {
        error = null;
        if (p.Length < 8 || BinaryPrimitives.ReadInt32LittleEndian(p) != SchemaV1.CommandMagic || BinaryPrimitives.ReadInt32LittleEndian(p.Slice(4)) != SearchWire.CmdFork)
        {
            return null;
        }
        ForkCommand c = new ForkCommand();
        if (p.Length < 32)
        {
            error = "FORK of " + p.Length + " bytes";
            return c;
        }
        c.StepId = Word(p, 2);
        c.N = Word(p, 3);
        c.Mode = Word(p, 4);
        c.Tag = Word(p, 5);
        int pathBytes = Word(p, 6);
        if (c.N < 1 || c.N > SearchWire.MaxChildren)
        {
            error = "FORK n=" + c.N;
            return c;
        }
        if (c.Mode != SearchWire.ModeResample && c.Mode != SearchWire.ModeExact)
        {
            error = "FORK mode=" + c.Mode;
            return c;
        }
        if (pathBytes < 1 || pathBytes > 107)
        {
            error = "FORK path_bytes=" + pathBytes;
            return c;
        }
        c.Flags = Word(p, 7);
        if ((c.Flags & ~SearchWire.KnownFlags) != 0)
        {
            error = "FORK flags = " + c.Flags;
            return c;
        }
        int expect = 32 + 8 * c.N + 4 * ((pathBytes + 3) / 4);
        if (p.Length != expect)
        {
            error = "FORK of " + p.Length + " bytes, expected " + expect;
            return c;
        }
        c.Seeds = new ulong[c.N];
        for (int i = 0; i < c.N; i++)
        {
            c.Seeds[i] = BinaryPrimitives.ReadUInt64LittleEndian(p.Slice(32 + 8 * i));
        }
        c.Path = p.Slice(32 + 8 * c.N, pathBytes).ToArray();
        if (Array.IndexOf(c.Path, (byte)0) >= 0)
        {
            error = "FORK socket path contains a NUL byte";
        }
        return c;
    }
}

/// <summary>
/// The search build's serve extension (--serve --search; plan step H2, notes/search/protocol.md): the FORK command in the
/// process that runs the real game, and the whole life of a fork child (re-randomisation, socket, HELLO, serving the rest
/// of the fight, FIGHT_END, _exit; with FORK flag Game: serving the rest of the game, its TERMINAL frame, _exit). One
/// instance per process; a child inherits it at fork() and turns into child mode.
/// </summary>
public sealed class SearchExtension : IServeExtension
{
    private readonly bool _allowExact;
    private readonly bool _verbose;

    // ---- child state (set in the child right after fork) ----
    private bool _isChild;
    private int _tag, _index, _mode;
    private bool _game;
    private CombatRoom? _forkRoom;
    private int _roundAtFork;
    private long _policyAtFork, _dpAtFork;
    private int[] _hpFork = Array.Empty<int>(), _maxHpFork = Array.Empty<int>(), _potionsFork = Array.Empty<int>(), _goldFork = Array.Empty<int>(), _potionsUsed = Array.Empty<int>();
    private int[]? _terminalWords;

    public long Forks, ForkCommands;

    public SearchExtension(bool allowExact, bool verbose)
    {
        _allowExact = allowExact;
        _verbose = verbose;
    }

    public bool RefusesReset => _isChild;

    private static bool InFight(RunContext ctx, out CombatRoom? room)
    {
        room = ctx.Run.CurrentRoom as CombatRoom;
        return room != null && ctx.CombatManager.IsInProgress;
    }

    /// <summary>
    /// Fair-play gate deciding whether FORK is permitted at <paramref name="decision"/>.
    /// Combat decisions allow fight-mode or game-mode forks (subject to fork-room scoping in fight-mode children).
    /// Non-combat decisions require <see cref="SearchWire.FlagGame"/> and strictly exclude <see cref="DecisionKind.Crystal"/>
    /// (hidden fogged grid) and <see cref="DecisionKind.Reward"/> (unopened pre-populated CardReward._cards).
    /// </summary>
    public static bool CanForkAt(
        Decision decision,
        int flags,
        bool isChild,
        bool isGameChild,
        CombatRoom? forkRoom,
        out CombatRoom? room)
    {
        bool wantGame = (flags & SearchWire.FlagGame) != 0;
        if (InFight(decision.Episode, out room))
        {
            return !(isChild && !isGameChild && !ReferenceEquals(room, forkRoom));
        }
        if (!wantGame || (isChild && !isGameChild))
        {
            return false;
        }
        return decision.Kind != DecisionKind.Crystal && decision.Kind != DecisionKind.Reward;
    }

    // ---------------------------------------------------------------------------------------------- FORK

    public int OnPendingCommand(ProtocolServer server, ReadOnlySpan<byte> message, Decision decision)
    {
        ForkPrimitive.ReapNonBlocking();
        ForkCommand? c = ForkCommand.Parse(message, out string? error);
        if (c == null)
        {
            return -1;
        }
        if (error != null)
        {
            Console.Error.WriteLine("[search] FORK refused: " + error);
            return ErrorCode.BadMessage;
        }
        if (c.StepId != server.StepId)
        {
            return ErrorCode.StaleStep;
        }
        if (c.Mode == SearchWire.ModeExact && !_allowExact)
        {
            Console.Error.WriteLine("[search] FORK refused: mode EXACT needs --allow-debug-options");
            return ErrorCode.BadMessage;
        }
        if (!CanForkAt(decision, c.Flags, _isChild, _game, _forkRoom, out CombatRoom? room))
        {
            if (_verbose)
            {
                Console.Error.WriteLine("[search] FORK refused: kind=" + decision.Kind + " flags=" + c.Flags);
            }
            return SearchWire.ErrForkRefused;
        }
        ForkCommands++;
        byte[] parentObs = server.PendingObs.ToArray();
        int forkerPid = Libc.getpid();
        int[] pids = new int[c.N];
        long t0 = Stopwatch.GetTimestamp();
        for (int i = 0; i < c.N; i++)
        {
            int pid = ForkPrimitive.ForkOrExit(-1);
            if (pid == 0)
            {
                ChildStart(server, decision, room, c, i, forkerPid, parentObs);
                return 0;
            }
            pids[i] = pid;
            Forks++;
        }
        long us = (Stopwatch.GetTimestamp() - t0) * 1_000_000 / Stopwatch.Frequency;
        int[] body = new int[4 + c.N];
        body[0] = c.Tag;
        body[1] = c.N;
        body[2] = c.Mode;
        body[3] = (int)ForkPrimitive.AbnormalExits;
        Array.Copy(pids, 0, body, 4, c.N);
        server.SendMessage(Message(server, SearchWire.MsgForked, server.StepId, (int)Math.Min(int.MaxValue, us), body));
        return 0;
    }

    /// <summary>A small frame-shaped message: the 16-word frame header (N_TABLES 0, GLOBAL_LEN 0) + body words.</summary>
    private static byte[] Message(ProtocolServer server, int msgType, int stepId, int hostUs, int[] body)
    {
        byte[] b = new byte[(SchemaV1.HeaderLen + body.Length) * 4];
        void W(int word, int v) => BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(word * 4), v);
        W(Hd.Magic, SchemaV1.FrameMagic);
        W(Hd.SchemaVersion, SchemaV1.Version);
        W(Hd.MsgType, msgType);
        W(Hd.VocabHash, server.Encoder.Vocab.Hash);
        W(Hd.CatalogHash, server.Encoder.CatalogHash);
        W(Hd.EpisodeId, server.EpisodeId);
        W(Hd.StepId, stepId);
        W(Hd.FrameBytes, b.Length);
        W(Hd.HostUs, hostUs);
        for (int i = 0; i < body.Length; i++)
        {
            W(SchemaV1.HeaderLen + i, body[i]);
        }
        return b;
    }

    // ---------------------------------------------------------------------------------------------- the child

    private void ChildStart(ProtocolServer server, Decision decision, CombatRoom? room, ForkCommand c, int index, int forkerPid, byte[] parentObs)
    {
        RunContext ctx = decision.Episode;
        if ((c.Flags & SearchWire.FlagDieWithParent) != 0)
        {
            // fires when the forking THREAD dies (the engine thread, i.e. with the process); a parent that died before the
            // prctl is no longer our parent
            if (Libc.prctl(Libc.PR_SET_PDEATHSIG, Libc.SIGKILL, 0, 0, 0) != 0)
            {
                ForkPrimitive.ChildFatal(SearchWire.ExitHygiene, "prctl(PR_SET_PDEATHSIG) failed: errno " + Libc.Errno);
            }
            if (Libc.getppid() != forkerPid)
            {
                Libc._exit(0);
            }
        }
        _isChild = true;
        _tag = c.Tag;
        _index = index;
        _mode = c.Mode;
        _game = (c.Flags & SearchWire.FlagGame) != 0;
        _forkRoom = room;
        _roundAtFork = room != null ? room.CombatState.RoundNumber : 0;
        _policyAtFork = ctx.Counters.PolicyDecisions;
        _dpAtFork = ctx.Counters.DecisionPoints;
        IReadOnlyList<Player> players = ctx.Run.Players;
        int n = players.Count;
        _hpFork = new int[n];
        _maxHpFork = new int[n];
        _potionsFork = new int[n];
        _goldFork = new int[n];
        _potionsUsed = new int[n];
        for (int s = 0; s < n; s++)
        {
            Player p = players[s];
            _hpFork[s] = p.Creature.CurrentHp;
            _maxHpFork[s] = p.Creature.MaxHp;
            _potionsFork[s] = p.Potions.Count();
            _goldFork[s] = p.Gold;
        }
        ulong seed = c.Seeds[index];
        if (c.Mode == SearchWire.ModeResample)
        {
            Resampler.Rerandomise(ctx, server.Encoder, seed, PileOrder.Canonical, resampleMacroState: _game);
        }
        server.EncodePendingAgain(decision);
        if (!SameFrame(server.PendingObs, parentObs))
        {
            ForkPrimitive.ChildFatal(SearchWire.ExitFrameChanged, "the frame of the fork decision changed after the re-randomisation (mode " + c.Mode + "): the resampler touched public state");
        }
        SocketTransport? t = SocketTransport.Connect(c.Path, out int errno);
        if (t == null)
        {
            ForkPrimitive.ChildFatal(SearchWire.ExitConnect, "cannot connect to " + System.Text.Encoding.UTF8.GetString(c.Path) + ": errno " + errno);
        }
        server.SwitchTransport(t!);
        byte[] hello = new byte[64];
        Span<byte> h = hello;
        BinaryPrimitives.WriteUInt32LittleEndian(h, SearchWire.HelloMagic);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(4), SearchWire.Version);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(8), Libc.getpid());
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(12), forkerPid);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(16), c.Tag);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(20), index);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(24), c.Mode);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(28), ForkPrimitive.Depth);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(32), server.EpisodeId);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(36), server.StepId);
        BinaryPrimitives.WriteUInt64LittleEndian(h.Slice(40), seed);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(48), ForkPrimitive.RootPid);
        BinaryPrimitives.WriteInt32LittleEndian(h.Slice(52), c.Flags);
        server.SendMessage(hello);
        server.SendMessage(server.PendingObs);
    }

    /// <summary>Byte-identical, HOST_US (header word 14) aside.</summary>
    public static bool SameFrame(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        if (a.Length != b.Length || a.Length < SchemaV1.HeaderLen * 4)
        {
            return false;
        }
        int us = Hd.HostUs * 4;
        return a.Slice(0, us).SequenceEqual(b.Slice(0, us)) && a.Slice(us + 4).SequenceEqual(b.Slice(us + 4));
    }

    public void BeforeObs(ProtocolServer server, Decision decision)
    {
        if (!_isChild || _game)
        {
            return;
        }
        RunContext ctx = decision.Episode;
        if (InFight(ctx, out CombatRoom? room) && ReferenceEquals(room, _forkRoom))
        {
            return;
        }
        // the first policy decision after the forked fight: FIGHT_END, that frame, exit
        int[] words = FightEndWords(ctx, SearchWire.OutcomeWon, terminal: false, ctx.Counters.PolicyDecisions, ctx.Counters.DecisionPoints);
        server.SendMessage(Message(server, SearchWire.MsgFightEnd, server.StepId, 0, words));
        server.SendMessage(server.PendingObs);
        Libc._exit(0);
    }

    private int[] FightEndWords(RunContext ctx, int outcome, bool terminal, long policyNow, long dpNow)
    {
        IReadOnlyList<Player> players = ctx.Run.Players;
        int n = players.Count;
        int[] w = new int[SearchWire.FightEndSeatOffset - SchemaV1.HeaderLen + SearchWire.FightEndSeatWords * n];
        CombatState cs = _forkRoom!.CombatState;
        int enemiesAlive = 0, enemyHp = 0;
        foreach (Creature e in cs._enemies)
        {
            if (e.IsAlive)
            {
                enemiesAlive++;
                enemyHp += e.CurrentHp;
            }
        }
        w[0] = _tag;
        w[1] = _index;
        w[2] = outcome;
        w[3] = n;
        w[4] = terminal ? 1 : 0;
        w[5] = _roundAtFork;
        w[6] = cs.RoundNumber;
        w[7] = (int)(policyNow - _policyAtFork);
        w[8] = (int)(dpNow - _dpAtFork);
        w[9] = enemiesAlive;
        w[10] = enemyHp;
        w[11] = ForkPrimitive.Depth;
        for (int s = 0; s < n; s++)
        {
            Player p = players[s];
            int o = SearchWire.FightEndSeatOffset - SchemaV1.HeaderLen + SearchWire.FightEndSeatWords * s;
            w[o] = p.Creature.CurrentHp;
            w[o + 1] = p.Creature.MaxHp;
            w[o + 2] = p.Creature.IsAlive ? 1 : 0;
            w[o + 3] = _hpFork[s];
            w[o + 4] = _maxHpFork[s];
            w[o + 5] = _potionsFork[s];
            w[o + 6] = p.Potions.Count();
            w[o + 7] = _potionsUsed[s];
            w[o + 8] = _goldFork[s];
            w[o + 9] = p.Gold;
        }
        return w;
    }

    public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
    {
        if (_isChild && decision.Actions[chosen].Verb == Verb.UsePotion)
        {
            _potionsUsed[decision.Actions[chosen].Seat]++;
        }
    }

    public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
    {
        if (!_isChild || _game)
        {
            return;
        }
        // the run is still alive here (teardown comes before the final frame is sent): read the end state now; the
        // outcome is the final one only in OnEpisodeEnd (episode checks may still turn it into an abort)
        _terminalWords = FightEndWords(ctx, 0, terminal: true, ctx.Counters.PolicyDecisions, ctx.Counters.DecisionPoints);
    }

    public void BeforeFinalFrame(ProtocolServer server, EpisodeResult result)
    {
        if (!_isChild || _game)
        {
            return;
        }
        if (_terminalWords == null)
        {
            ForkPrimitive.ChildFatal(SearchWire.ExitInvariant, "episode ended without a terminal state (" + result.Outcome + " " + result.TruncatedBy + ")");
        }
        _terminalWords![2] = result.Outcome switch
        {
            EpisodeOutcome.Win => SearchWire.OutcomeWon,
            EpisodeOutcome.Loss => SearchWire.OutcomeWiped,
            EpisodeOutcome.Truncated => SearchWire.OutcomeTruncated,
            _ => SearchWire.OutcomeAborted,
        };
        int step = BinaryPrimitives.ReadInt32LittleEndian(server.Encoder.Payload.Slice(Hd.StepId * 4));
        server.SendMessage(Message(server, SearchWire.MsgFightEnd, step, 0, _terminalWords));
    }

    public void AfterFinalFrame(ProtocolServer server)
    {
        if (_isChild)
        {
            Libc._exit(0);
        }
    }

    public void OnStreamEnd(ProtocolServer server, bool broken, string why)
    {
        if (!_isChild)
        {
            return;
        }
        if (broken)
        {
            ForkPrimitive.ChildFatal(SearchWire.ExitBrokenSocket, "socket broken: " + why);
        }
        Libc._exit(0);
    }
}
