using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Pipes;

namespace SpireHost.Obs.Protocol;

/// <summary>A message channel carrying <c>[u32 LE length][payload]</c> in both directions (obs-schema-v1 3.1).</summary>
public interface IMessageTransport : IDisposable
{
    string Name { get; }

    /// <summary>
    /// Blocks until one complete message arrived and returns its payload (valid until the next call), or null at a
    /// clean end of stream (peer closed at a message boundary). Throws <see cref="TransportBroken"/> if the stream ends
    /// inside a message or announces a length beyond the limit (the byte stream cannot be resynchronised).
    /// </summary>
    ReadOnlyMemory<byte>? Read();

    /// <summary>Writes one message (length prefix + payload) with a single Write call and flushes.</summary>
    void Write(ReadOnlySpan<byte> payload);
}

public sealed class TransportBroken : Exception
{
    public TransportBroken(string message, Exception? inner = null) : base(message, inner)
    {
    }
}

/// <summary>Length-prefixed messages over a pair of streams (stdin/stdout, or one duplex named pipe).</summary>
public sealed class StreamTransport : IMessageTransport
{
    /// <summary>Commands are tiny (RESET = 64 bytes); anything above this is garbage on the wire.</summary>
    public const int MaxCommandBytes = 1 << 16;

    private readonly Stream _in;
    private readonly Stream _out;
    private readonly bool _owns;
    private byte[] _rbuf = new byte[256];
    private byte[] _wbuf = new byte[64 * 1024];

    public StreamTransport(string name, Stream input, Stream output, bool owns)
    {
        Name = name;
        _in = input;
        _out = output;
        _owns = owns;
    }

    public string Name { get; }

    public long BytesIn { get; private set; }

    public long BytesOut { get; private set; }

    public ReadOnlyMemory<byte>? Read()
    {
        Span<byte> head = stackalloc byte[4];
        int got = ReadFully(head, eofOk: true);
        if (got == 0)
        {
            return null;
        }
        uint n = BinaryPrimitives.ReadUInt32LittleEndian(head);
        if (n > MaxCommandBytes)
        {
            throw new TransportBroken($"message length {n} exceeds the command limit {MaxCommandBytes}");
        }
        if (_rbuf.Length < n)
        {
            _rbuf = new byte[Math.Max((int)n, _rbuf.Length * 2)];
        }
        ReadFully(new Span<byte>(_rbuf, 0, (int)n), eofOk: false);
        BytesIn += 4 + n;
        return new ReadOnlyMemory<byte>(_rbuf, 0, (int)n);
    }

    private int ReadFully(Span<byte> dst, bool eofOk)
    {
        int got = 0;
        while (got < dst.Length)
        {
            int k;
            try
            {
                k = _in.Read(dst.Slice(got));
            }
            catch (IOException e)
            {
                // broken pipe: the peer is gone (transport-spec 5: exit as on EOF)
                if (got == 0 && eofOk)
                {
                    return 0;
                }
                throw new TransportBroken("read failed: " + e.Message, e);
            }
            if (k <= 0)
            {
                if (got == 0 && eofOk)
                {
                    return 0;
                }
                throw new TransportBroken($"end of stream after {got} of {dst.Length} bytes");
            }
            got += k;
        }
        return got;
    }

    public void Write(ReadOnlySpan<byte> payload)
    {
        int total = payload.Length + 4;
        if (_wbuf.Length < total)
        {
            _wbuf = new byte[Math.Max(total, _wbuf.Length * 2)];
        }
        BinaryPrimitives.WriteUInt32LittleEndian(_wbuf, (uint)payload.Length);
        payload.CopyTo(new Span<byte>(_wbuf, 4, payload.Length));
        try
        {
            _out.Write(_wbuf, 0, total);
            _out.Flush();
        }
        catch (IOException e)
        {
            throw new TransportBroken("write failed: " + e.Message, e);
        }
        BytesOut += total;
    }

    public void Dispose()
    {
        if (_owns)
        {
            _in.Dispose();
            if (!ReferenceEquals(_in, _out))
            {
                _out.Dispose();
            }
        }
    }

    // ---------------------------------------------------------------------------------------------- factories

    /// <summary>stdin / stdout (the process's raw handles; Console.Out has been redirected to stderr by the executable's entry point).</summary>
    public static StreamTransport Stdio()
    {
        return new StreamTransport("stdio", Console.OpenStandardInput(), Console.OpenStandardOutput(), owns: true);
    }

    public const uint HelloMagic = 0x4C485053; // "SPHL"
    public const int TransportVersion = 1;

    /// <summary>
    /// notes/transport-spec.md section 3: open the actor's named pipe as a client and send HELLO
    /// (magic, transport version 1, Environment.ProcessId, 0) as the first message.
    /// </summary>
    public static StreamTransport NamedPipe(string fullName, int connectTimeoutMs = 30_000)
    {
        const string prefix = @"\\.\pipe\";
        string name = fullName.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) ? fullName.Substring(prefix.Length) : fullName;
        NamedPipeClientStream pipe = new NamedPipeClientStream(".", name, PipeDirection.InOut, PipeOptions.None);
        pipe.Connect(connectTimeoutMs);
        StreamTransport t = new StreamTransport("pipe", pipe, pipe, owns: true);
        Span<byte> hello = stackalloc byte[16];
        BinaryPrimitives.WriteUInt32LittleEndian(hello, HelloMagic);
        BinaryPrimitives.WriteInt32LittleEndian(hello.Slice(4), TransportVersion);
        BinaryPrimitives.WriteInt32LittleEndian(hello.Slice(8), Environment.ProcessId);
        BinaryPrimitives.WriteInt32LittleEndian(hello.Slice(12), 0);
        t.Write(hello);
        return t;
    }
}
