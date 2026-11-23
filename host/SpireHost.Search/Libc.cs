using System;
using System.Runtime.InteropServices;
using System.Text;

namespace SpireHost.Search;

/// <summary>libc entry points of the fork primitive and the fork child's socket (Linux only).</summary>
public static unsafe class Libc
{
    public const int AF_UNIX = 1, SOCK_STREAM = 1, O_RDONLY = 0, O_RDWR = 2, WNOHANG = 1, ENOENT = 2, ESRCH = 3, EINTR = 4, ECHILD = 10, PR_SET_PDEATHSIG = 1, PR_SET_CHILD_SUBREAPER = 36, SIGKILL = 9;

    [DllImport("libc", SetLastError = true)] public static extern int fork();
    [DllImport("libc")] public static extern void _exit(int code);
    [DllImport("libc", SetLastError = true)] public static extern int waitpid(int pid, out int status, int options);
    [DllImport("libc", SetLastError = true)] public static extern int pipe(int[] fds);
    [DllImport("libc", SetLastError = true)] public static extern nint write(int fd, byte* buf, nint n);
    [DllImport("libc", SetLastError = true)] public static extern nint read(int fd, byte* buf, nint n);
    [DllImport("libc", SetLastError = true)] public static extern int close(int fd);
    [DllImport("libc", SetLastError = true)] public static extern int dup2(int oldfd, int newfd);
    [DllImport("libc", SetLastError = true, EntryPoint = "open")] public static extern int open_(byte[] path, int flags);
    [DllImport("libc", SetLastError = true)] public static extern int kill(int pid, int sig);
    [DllImport("libc")] public static extern int getpid();
    [DllImport("libc")] public static extern int getppid();
    [DllImport("libc")] public static extern int gettid();
    [DllImport("libc")] public static extern int sched_yield();
    [DllImport("libc", SetLastError = true)] public static extern int socket(int domain, int type, int protocol);
    [DllImport("libc", SetLastError = true)] public static extern int connect(int fd, byte[] addr, int addrLen);
    [DllImport("libc", SetLastError = true)] public static extern nint send(int fd, byte* buf, nint n, int flags);
    [DllImport("libc", SetLastError = true)] public static extern nint readlink(byte[] path, byte[] buf, nint size);
    [DllImport("libc", SetLastError = true)] public static extern int prctl(int option, ulong arg2, ulong arg3, ulong arg4, ulong arg5);

    [StructLayout(LayoutKind.Sequential)]
    public struct PollFd
    {
        public int fd;
        public short events;
        public short revents;
    }

    [DllImport("libc", SetLastError = true)] public static extern int poll(ref PollFd fds, ulong n, int timeoutMs);

    public static int Errno => Marshal.GetLastPInvokeError();

    public static byte[] CPath(string path)
    {
        byte[] b = new byte[Encoding.UTF8.GetByteCount(path) + 1];
        Encoding.UTF8.GetBytes(path, 0, path.Length, b, 0);
        return b;
    }

    /// <summary>Writes every byte or returns the errno of the failed write (EINTR is repeated: a signal is not an error).</summary>
    public static unsafe int WriteAll(int fd, ReadOnlySpan<byte> b)
    {
        fixed (byte* p = b)
        {
            int off = 0;
            while (off < b.Length)
            {
                nint n = write(fd, p + off, b.Length - off);
                if (n < 0)
                {
                    int e = Errno;
                    if (e == EINTR)
                    {
                        continue;
                    }
                    return e;
                }
                off += (int)n;
            }
        }
        return 0;
    }

    /// <summary>Reads exactly dst.Length bytes: the count read (less than dst.Length = end of stream), or -errno.</summary>
    public static unsafe int ReadFully(int fd, Span<byte> dst)
    {
        fixed (byte* p = dst)
        {
            int got = 0;
            while (got < dst.Length)
            {
                nint n = read(fd, p + got, dst.Length - got);
                if (n < 0)
                {
                    int e = Errno;
                    if (e == EINTR)
                    {
                        continue;
                    }
                    return -e;
                }
                if (n == 0)
                {
                    return got;
                }
                got += (int)n;
            }
            return got;
        }
    }

    /// <summary>sockaddr_un for a filesystem path (sun_family = AF_UNIX, sun_path NUL-terminated, 108 bytes).</summary>
    public static byte[] UnixAddress(ReadOnlySpan<byte> pathUtf8)
    {
        if (pathUtf8.Length < 1 || pathUtf8.Length > 107)
        {
            throw new ArgumentException("unix socket path of " + pathUtf8.Length + " bytes (1..107)");
        }
        byte[] a = new byte[2 + 108];
        a[0] = AF_UNIX & 0xFF;
        a[1] = 0;
        pathUtf8.CopyTo(a.AsSpan(2));
        return a;
    }
}
