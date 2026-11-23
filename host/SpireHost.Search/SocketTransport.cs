using System;
using System.Buffers.Binary;
using SpireHost.Obs.Protocol;

namespace SpireHost.Search;

/// <summary>
/// A fork child's message channel: the serve framing ([u32 length][payload], obs-schema-v1 3.1) over a connected
/// AF_UNIX stream socket, with blocking libc calls only (a fork child has one thread: no thread pool, no async I/O).
/// </summary>
public sealed unsafe class SocketTransport : IMessageTransport
{
    private const int MSG_NOSIGNAL = 0x4000;
    private readonly int _fd;
    private byte[] _rbuf = new byte[256];
    private byte[] _wbuf = new byte[64 * 1024];

    private SocketTransport(int fd)
    {
        _fd = fd;
    }

    public string Name => "unix-socket";

    /// <summary>Connects to <paramref name="pathUtf8"/>; null with <paramref name="errno"/> set when socket() or connect() fails.</summary>
    public static SocketTransport? Connect(ReadOnlySpan<byte> pathUtf8, out int errno)
    {
        int fd = Libc.socket(Libc.AF_UNIX, Libc.SOCK_STREAM, 0);
        if (fd < 0)
        {
            errno = Libc.Errno;
            return null;
        }
        byte[] addr = Libc.UnixAddress(pathUtf8);
        while (Libc.connect(fd, addr, addr.Length) != 0)
        {
            errno = Libc.Errno;
            if (errno != Libc.EINTR)
            {
                Libc.close(fd);
                return null;
            }
        }
        errno = 0;
        return new SocketTransport(fd);
    }

    public ReadOnlyMemory<byte>? Read()
    {
        Span<byte> head = stackalloc byte[4];
        int got = Libc.ReadFully(_fd, head);
        if (got < 0)
        {
            throw new TransportBroken("socket read failed: errno " + -got);
        }
        if (got == 0)
        {
            return null;
        }
        if (got < 4)
        {
            throw new TransportBroken("end of stream after " + got + " of 4 length bytes");
        }
        uint n = BinaryPrimitives.ReadUInt32LittleEndian(head);
        if (n > StreamTransport.MaxCommandBytes)
        {
            throw new TransportBroken($"message length {n} exceeds the command limit {StreamTransport.MaxCommandBytes}");
        }
        if (_rbuf.Length < n)
        {
            _rbuf = new byte[Math.Max((int)n, _rbuf.Length * 2)];
        }
        int body = Libc.ReadFully(_fd, new Span<byte>(_rbuf, 0, (int)n));
        if (body != n)
        {
            throw new TransportBroken(body < 0 ? "socket read failed: errno " + -body : "end of stream after " + body + " of " + n + " bytes");
        }
        return new ReadOnlyMemory<byte>(_rbuf, 0, (int)n);
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
        fixed (byte* p = _wbuf)
        {
            int off = 0;
            while (off < total)
            {
                // MSG_NOSIGNAL: a peer that closed gives EPIPE here, not SIGPIPE
                nint k = Libc.send(_fd, p + off, total - off, MSG_NOSIGNAL);
                if (k < 0)
                {
                    int e = Libc.Errno;
                    if (e == Libc.EINTR)
                    {
                        continue;
                    }
                    throw new TransportBroken("socket write failed: errno " + e);
                }
                off += (int)k;
            }
        }
    }

    public void Dispose()
    {
        Libc.close(_fd);
    }
}
