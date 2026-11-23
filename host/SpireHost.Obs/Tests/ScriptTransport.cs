using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using SpireHost.Obs.Protocol;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Tests;

/// <summary>
/// In-process transport for protocol tests: every frame the server writes is kept; every Read asks a client callback
/// (which sees all frames written so far) for the next command. Runs on the engine thread like the real server.
/// </summary>
public sealed class ScriptTransport : IMessageTransport
{
    private readonly Func<ScriptTransport, byte[]?> _client;
    public readonly List<byte[]> Written = new List<byte[]>();
    public int Reads;

    /// <summary>Safety net: a scripted client that keeps answering after the scenario went wrong would loop forever.</summary>
    public int MaxReads = 50_000;

    public bool Overrun;

    public ScriptTransport(Func<ScriptTransport, byte[]?> client)
    {
        _client = client;
    }

    public string Name => "script";

    public byte[]? Last => Written.Count == 0 ? null : Written[^1];

    public ReadOnlyMemory<byte>? Read()
    {
        Reads++;
        if (Reads > MaxReads)
        {
            if (!Overrun)
            {
                Console.Error.WriteLine("[selftest] script transport overrun at: " + Environment.StackTrace);
            }
            Overrun = true;
            return null;
        }
        byte[]? next = _client(this);
        if (next == null)
        {
            // NOTE: "next == null ? null : new ReadOnlyMemory<byte>(next)" would type as ReadOnlyMemory<byte> (null converts
            // through the byte[] operator) and hand the server an EMPTY message instead of end-of-stream.
            return null;
        }
        return new ReadOnlyMemory<byte>(next);
    }

    public void Write(ReadOnlySpan<byte> payload)
    {
        if (Written.Count >= 4096)
        {
            Written.RemoveRange(0, 2048); // bounded history (the scenarios look at the last few frames)
        }
        Written.Add(payload.ToArray());
    }

    public void Dispose()
    {
    }

    // ---- command builders (obs-schema-v1 7.1) ----

    /// <param name="debugWord">RESET word 15: 0 in production (errata P-2); non-zero = the DEBUG forced HP of the episode (ruling A4).</param>
    public static byte[] Reset(int characterVocabId, int ascension = 0, ulong? seed = null, int flags = ResetFlag.Default, int maxDecisions = 0, int nSeats = 1, int viewMode = 0, int viewSeat = -1,
        int debugWord = 0)
    {
        byte[] b = new byte[64];
        int[] w = { SchemaV1.CommandMagic, CmdType.Reset, nSeats, viewMode, viewSeat, ascension, characterVocabId, 0, 0, 0,
            seed.HasValue ? 1 : 0, (int)(uint)(seed ?? 0), (int)(uint)((seed ?? 0) >> 32), flags, maxDecisions, debugWord };
        for (int i = 0; i < 16; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(i * 4), w[i]);
        }
        return b;
    }

    public static byte[] Step(int stepId, int index)
    {
        byte[] b = new byte[16];
        BinaryPrimitives.WriteInt32LittleEndian(b, SchemaV1.CommandMagic);
        BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(4), CmdType.Step);
        BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(8), stepId);
        BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(12), index);
        return b;
    }

    public static byte[] Quit()
    {
        byte[] b = new byte[8];
        BinaryPrimitives.WriteInt32LittleEndian(b, SchemaV1.CommandMagic);
        BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(4), CmdType.Quit);
        return b;
    }

    public static int Word(byte[] frame, int index) => BinaryPrimitives.ReadInt32LittleEndian(frame.AsSpan(index * 4));

    public static int Msg(byte[] f) => Word(f, Hd.MsgType);

    public static int StepOf(byte[] f) => Word(f, Hd.StepId);

    public static int EpisodeOf(byte[] f) => Word(f, Hd.EpisodeId);

    public static int NActions(byte[] f) => Word(f, Hd.NActions);

    public static int GlobalOf(byte[] f, int i) => Word(f, SchemaV1.GlobalOffsetWords + i);
}
