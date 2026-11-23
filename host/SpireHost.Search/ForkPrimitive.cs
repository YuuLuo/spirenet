using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;

namespace SpireHost.Search;

/// <summary>
/// The OS-fork primitive (plan step H1; fork study section 1.4). <see cref="Fork"/> is called on the engine thread at a
/// decision point; the child is a byte-exact copy of the whole game and continues from the same call. Linux only, and only
/// with the JIT's W^X double mapping off and tiered compilation off (<see cref="RuntimeError"/>).
///
/// Child hygiene, done before <see cref="Fork"/> returns 0 in the child:
///   * fds 0 and 1 are re-opened on /dev/null: a child never reads its parent's commands or writes into its frame stream;
///   * every other inherited pipe or socket fd above 2 (except the one the caller keeps) is replaced by /dev/null with
///     dup2: the child no longer holds its parent's or its siblings' streams, so they see EOF when their owners exit.
///     The fd NUMBERS stay occupied on purpose: the runtime keeps some of them cached (e.g. the signal-handling pipe), and
///     a closed number would be reused by the child's socket, so a late write of the runtime would land in the protocol;
///   * fd 2 (stderr) is kept.
/// </summary>
public static class ForkPrimitive
{
    /// <summary>0 in the process that runs the real game; a child of depth d forks children of depth d + 1.</summary>
    public static int Depth;

    /// <summary>pid of the depth-0 process (the real game's server).</summary>
    public static int RootPid;

    /// <summary>This process's children (any depth re-parented to it included) reaped with a non-zero status or a signal other than SIGKILL.</summary>
    public static long AbnormalExits;

    public static long Reaped;

    /// <summary>What forbids forking in this process, or null (Linux, W^X off, tiered compilation off).</summary>
    public static string? RuntimeError()
    {
        if (!OperatingSystem.IsLinux())
        {
            return "fork is Linux-only (this is " + System.Runtime.InteropServices.RuntimeInformation.OSDescription + ")";
        }
        List<string> bad = new List<string>();
        if (Environment.GetEnvironmentVariable("DOTNET_EnableWriteXorExecute") != "0")
        {
            bad.Add("DOTNET_EnableWriteXorExecute=0 is required (with W^X the JIT's code pages are a MAP_SHARED double mapping: parent and child write into the same pages and both die)");
        }
        if (Environment.GetEnvironmentVariable("DOTNET_TieredCompilation") != "0")
        {
            bad.Add("DOTNET_TieredCompilation=0 is required (with tiered compilation a child can block forever on a runtime lock held by a thread that does not exist in it)");
        }
        return bad.Count == 0 ? null : string.Join("; ", bad);
    }

    /// <summary>
    /// Once per forking server (depth 0), before the first fork: child subreaper (orphaned grandchildren are re-parented to
    /// this process and reaped by it) and the JIT warm-up of everything only children run (with tiered compilation off
    /// every method is compiled once; compiled in a child, the code dies with the child).
    /// </summary>
    public static void InitRoot()
    {
        string? err = RuntimeError();
        if (err != null)
        {
            throw new InvalidOperationException(err);
        }
        RootPid = Libc.getpid();
        Depth = 0;
        if (Libc.prctl(Libc.PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0)
        {
            throw new InvalidOperationException("prctl(PR_SET_CHILD_SUBREAPER) failed: errno " + Libc.Errno);
        }
        WarmUp();
    }

    /// <summary>
    /// fork(). Parent: returns the child's pid. Child: applies the hygiene above (keeping <paramref name="keepFd"/>, -1 =
    /// none), increments <see cref="Depth"/> and returns 0. A failing fork() throws (the caller treats it as fatal).
    /// </summary>
    public static int Fork(int keepFd)
    {
        int pid = ForkCore(keepFd);
        if (pid < 0)
        {
            throw new InvalidOperationException("fork() failed: errno " + -pid);
        }
        return pid;
    }

    /// <summary>
    /// <see cref="Fork"/> for the serve protocol: a failing fork() (EAGAIN / ENOMEM) is fatal for the process
    /// (protocol.md 2.3): message on stderr, exit 1 (a fork child: _exit(15)). No partial FORKED is ever sent.
    /// </summary>
    public static int ForkOrExit(int keepFd)
    {
        int pid = ForkCore(keepFd);
        if (pid < 0)
        {
            if (Depth > 0)
            {
                ChildFatal(15, "fork() failed: errno " + -pid);
            }
            Console.Error.WriteLine("[search] fatal: fork() failed: errno " + -pid + " - exiting");
            Console.Error.Flush();
            Environment.Exit(1);
        }
        return pid;
    }

    private static int ForkCore(int keepFd)
    {
        WaitForSleepingThreads();
        int pid = Libc.fork();
        if (pid < 0)
        {
            return -Libc.Errno;
        }
        if (pid == 0)
        {
            ChildHygiene(keepFd);
            SpireHost.Driver.HostSession.NoFinalizerThread = true;
            Depth++;
            AbnormalExits = 0;
            Reaped = 0;
            return 0;
        }
        return pid;
    }

    private static readonly byte[] StatBuf = new byte[1024];
    private static readonly List<byte[]> s_threadStats = new List<byte[]>();

    /// <summary>
    /// Returns when every other thread of this process is asleep (state S in /proc/self/task/TID/stat), so that none of
    /// them is inside one of the runtime's locks when fork() copies the process: a child inherits such a lock taken by a
    /// thread it does not have and blocks on it for ever. Measured without this wait (notes/gym/host.md, about 1 child in
    /// 11,000): the PAL handle-manager lock in the SuspendEE of a child's GC and the class-init lock of a static-base
    /// helper; the active thread in a serving parent is the finalizer thread. The check loop does not allocate, so it
    /// cannot start a GC that wakes the finalizer thread again. A fork child has no other thread: nothing to wait for.
    /// </summary>
    private static void WaitForSleepingThreads()
    {
        int self = Libc.gettid();
        s_threadStats.Clear();
        foreach (string task in Directory.GetFileSystemEntries("/proc/self/task"))
        {
            if (int.Parse(Path.GetFileName(task)) != self)
            {
                s_threadStats.Add(Libc.CPath(task + "/stat"));
            }
        }
        while (!AllAsleep())
        {
            Libc.sched_yield();
            SleepWaits++;
        }
    }

    /// <summary>Spins of <see cref="WaitForSleepingThreads"/> (some thread was not asleep).</summary>
    public static long SleepWaits;

    private static bool AllAsleep()
    {
        foreach (byte[] path in s_threadStats)
        {
            int fd = Libc.open_(path, Libc.O_RDONLY);
            if (fd < 0)
            {
                int e = Libc.Errno;
                if (e != Libc.ENOENT && e != Libc.ESRCH)
                {
                    throw new InvalidOperationException("open(/proc/self/task/.../stat) failed: errno " + e);
                }
                continue; // the thread has exited since the listing
            }
            int n = Libc.ReadFully(fd, StatBuf);
            Libc.close(fd);
            if (n < 0)
            {
                if (-n != Libc.ESRCH)
                {
                    throw new InvalidOperationException("read(/proc/self/task/.../stat) failed: errno " + -n);
                }
                continue; // the thread exited between open and read
            }
            // "tid (comm) S ...": comm may contain ") ", the state follows the LAST ')'
            int paren = StatBuf.AsSpan(0, n).LastIndexOf((byte)')');
            if (StatBuf[paren + 2] != (byte)'S')
            {
                return false;
            }
        }
        return true;
    }

    private static readonly byte[] DevNull = Libc.CPath("/dev/null");
    private static readonly byte[] LinkBuf = new byte[256];

    private static void ChildHygiene(int keepFd)
    {
        int dn = Libc.open_(DevNull, Libc.O_RDWR);
        if (dn < 0)
        {
            ChildFatal(13, "open(/dev/null) failed: errno " + Libc.Errno);
        }
        if (Libc.dup2(dn, 0) < 0 || Libc.dup2(dn, 1) < 0)
        {
            ChildFatal(13, "dup2(/dev/null, 0/1) failed: errno " + Libc.Errno);
        }
        foreach (int fd in InheritedStreams(keepFd, dn))
        {
            if (Libc.dup2(dn, fd) < 0)
            {
                ChildFatal(13, "dup2(/dev/null, " + fd + ") failed: errno " + Libc.Errno);
            }
        }
        Libc.close(dn);
    }

    /// <summary>The pipe and socket fds above 2 of this process, except <paramref name="keep"/> and <paramref name="devNull"/>.</summary>
    private static List<int> InheritedStreams(int keep, int devNull)
    {
        List<int> fds = new List<int>();
        foreach (string entry in Directory.GetFileSystemEntries("/proc/self/fd"))
        {
            int fd = int.Parse(Path.GetFileName(entry));
            if (fd <= 2 || fd == keep || fd == devNull)
            {
                continue;
            }
            nint n = Libc.readlink(Libc.CPath(entry), LinkBuf, LinkBuf.Length);
            if (n < 0)
            {
                // the directory fd of the enumeration itself is listed and already closed again
                if (Libc.Errno != Libc.ENOENT)
                {
                    throw new InvalidOperationException("readlink(" + entry + ") failed: errno " + Libc.Errno);
                }
                continue;
            }
            string target = Encoding.ASCII.GetString(LinkBuf, 0, (int)n);
            if (target.StartsWith("pipe:", StringComparison.Ordinal) || target.StartsWith("socket:", StringComparison.Ordinal))
            {
                fds.Add(fd);
            }
        }
        return fds;
    }

    /// <summary>A fork child that cannot continue: message on stderr, _exit(code) (no runtime shutdown in a forked child).</summary>
    public static void ChildFatal(int code, string message)
    {
        Console.Error.WriteLine("[search] fork child " + Libc.getpid() + " (depth " + Depth + "): " + message + " - _exit(" + code + ")");
        Console.Error.Flush();
        Libc._exit(code);
    }

    /// <summary>Reaps every child that has exited (never blocks); counts and reports abnormal exits on stderr.</summary>
    public static void ReapNonBlocking()
    {
        while (true)
        {
            int pid = Libc.waitpid(-1, out int status, Libc.WNOHANG);
            if (pid == 0)
            {
                return;
            }
            if (pid < 0)
            {
                int e = Libc.Errno;
                if (e == Libc.ECHILD)
                {
                    return;
                }
                if (e == Libc.EINTR)
                {
                    continue;
                }
                throw new InvalidOperationException("waitpid failed: errno " + e);
            }
            Reaped++;
            if (IsAbnormal(status))
            {
                AbnormalExits++;
                Console.Error.WriteLine("[search] child " + pid + " exited " + DescribeStatus(status));
            }
        }
    }

    /// <summary>Normal: exit status 0, or killed by SIGKILL (the client abandons a world that way).</summary>
    public static bool IsAbnormal(int status)
    {
        int sig = status & 0x7F;
        if (sig == 0)
        {
            return ((status >> 8) & 0xFF) != 0;
        }
        return sig != 9;
    }

    public static string DescribeStatus(int status)
    {
        int sig = status & 0x7F;
        return sig == 0 ? "with code " + ((status >> 8) & 0xFF) : "by signal " + sig;
    }

    // ---------------------------------------------------------------------------------------------- JIT warm-up

    private static readonly List<Action> s_warmUps = new List<Action>();

    /// <summary>Registers child-only game calls to be compiled in the root (e.g. generic instantiations); run by <see cref="InitRoot"/>.</summary>
    public static void RegisterWarmUp(Action a)
    {
        s_warmUps.Add(a);
    }

    private static void WarmUp()
    {
        // every method of this assembly with a body (child-side code included)
        foreach (Type t in typeof(ForkPrimitive).Assembly.GetTypes())
        {
            if (t.ContainsGenericParameters)
            {
                continue;
            }
            foreach (MethodInfo m in t.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            {
                if (m.IsAbstract || m.ContainsGenericParameters || m.GetMethodBody() == null)
                {
                    continue;
                }
                RuntimeHelpers.PrepareMethod(m.MethodHandle);
            }
            foreach (ConstructorInfo c in t.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
            {
                if (c.GetMethodBody() != null)
                {
                    RuntimeHelpers.PrepareMethod(c.MethodHandle);
                }
            }
        }
        // the P/Invoke stubs a child calls, by harmless calls here
        int dn = Libc.open_(DevNull, Libc.O_RDWR);
        if (dn < 0)
        {
            throw new InvalidOperationException("open(/dev/null) failed: errno " + Libc.Errno);
        }
        Libc.dup2(dn, dn);
        Span<byte> one = stackalloc byte[1];
        Libc.WriteAll(dn, one);
        Libc.ReadFully(dn, one);
        int s = Libc.socket(Libc.AF_UNIX, Libc.SOCK_STREAM, 0);
        if (s < 0)
        {
            throw new InvalidOperationException("socket(AF_UNIX) failed: errno " + Libc.Errno);
        }
        byte[] addr = Libc.UnixAddress(Encoding.UTF8.GetBytes("/proc/spirehost-warmup-no-such-socket"));
        Libc.connect(s, addr, addr.Length);
        Libc.close(s);
        Libc.close(dn);
        InheritedStreams(-1, -1);
        Libc.getppid();
        ReapNonBlocking();
        foreach (Action a in s_warmUps)
        {
            a();
        }
    }
}
