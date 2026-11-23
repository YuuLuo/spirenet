"""Run one benchmark command with a hard timeout, a memory guard and guaranteed cleanup (Windows).

    python -B scripts/bench_guard.py --timeout 300 [--min-commit-gb 24] [--abort-commit-gb 8] -- <command ...>

Why (notes/py-throughput-1-hotpath.md section 5): a benchmark with 192-256 fake hosts once hung for
five hours and left hundreds of host processes behind. Every fake host committed ~760 MB (OpenBLAS
thread buffers, now disabled in ``fake_spirehost.py``), the machine ran out of commit, a host's stderr
thread failed to start with ``MemoryError`` and ``threading.Thread.start`` waited forever for it.

The guard:

* puts itself into a fresh kill-on-close job object before starting the command, so the command and
  every descendant (the venv launcher, the real interpreter, every host and the hosts' launchers) are
  job members - nested jobs on Windows 8+; breakaway is not allowed;
* refuses to start while the machine's available commit is below ``--min-commit-gb`` (waits up to
  ``--wait-commit-s`` for it, then exits 75), and aborts the run when it drops below
  ``--abort-commit-gb`` while running (exit 76) - before the machine runs out and processes start
  failing in arbitrary places;
* kills everything in the job on timeout (exit 124), on Ctrl-C, on its own exit and after a normal
  exit of the command (stragglers), then checks that the job is empty.

The command's own exit code is returned otherwise. One status line per event goes to stderr.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import runpy
import subprocess
import sys
import threading
import time
from ctypes import wintypes

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectBasicProcessIdList = 3
JobObjectExtendedLimitInformation = 9
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class _Basic(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]


class _Extended(ctypes.Structure):
    _fields_ = [("Basic", _Basic), ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True) if sys.platform == "win32" else None  # type: ignore[attr-defined]
if _k32 is not None:
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                               ctypes.c_void_p]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.GetCurrentProcess.restype = wintypes.HANDLE


def say(msg: str) -> None:
    print(f"[bench_guard {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def commit_available_gb() -> float:
    """Machine-wide commit still available (GlobalMemoryStatusEx.ullAvailPageFile), GiB."""
    assert _k32 is not None
    st = _MemStatus()
    st.dwLength = ctypes.sizeof(st)
    _k32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPageFile / 2**30


def make_job(process_memory_gb: float = 0.0, below_normal: bool = False) -> int:
    assert _k32 is not None
    job = _k32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = _Extended()
    info.Basic.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE  # no BREAKAWAY_OK / SILENT_BREAKAWAY_OK
    if process_memory_gb:
        info.Basic.LimitFlags |= 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = int(process_memory_gb * 2**30)
    if below_normal:
        info.Basic.LimitFlags |= 0x20  # JOB_OBJECT_LIMIT_PRIORITY_CLASS (also constrains descendants)
        info.Basic.PriorityClass = 0x4000  # BELOW_NORMAL_PRIORITY_CLASS
    if not _k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info),
                                        ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not _k32.AssignProcessToJobObject(job, _k32.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(job)


def job_pids(job: int) -> list[int]:
    assert _k32 is not None
    n = 4096

    class _List(ctypes.Structure):
        _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD), ("NumberOfProcessIdsInList", wintypes.DWORD),
                    ("ProcessIdList", ctypes.c_size_t * n)]
    lst = _List()
    if not _k32.QueryInformationJobObject(job, JobObjectBasicProcessIdList, ctypes.byref(lst), ctypes.sizeof(lst),
                                          None):
        return []
    return [int(lst.ProcessIdList[i]) for i in range(lst.NumberOfProcessIdsInList)]


def kill_job_members(job: int) -> int:
    """Terminate every job member except this process; returns how many were still alive."""
    assert _k32 is not None
    me, killed = os.getpid(), 0
    for _ in range(5):  # members may spawn while we kill: repeat until the job is empty
        pids = [p for p in job_pids(job) if p != me]
        if not pids:
            break
        for pid in pids:
            h = _k32.OpenProcess(PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if h:
                if _k32.TerminateProcess(h, 137):
                    killed += 1
                _k32.CloseHandle(h)
        time.sleep(0.2)
    return killed


def run_in_process(job: int, cmd: list[str], timeout: float, min_physical_gb: float,
                   abort_commit_gb: float) -> None:
    """Run a Python script without another launcher/worker pair (budgeted real-host probes)."""
    assert _k32 is not None
    done = threading.Event()
    started = time.monotonic()

    def monitor() -> None:
        assert _k32 is not None
        mem = _MemStatus()
        mem.dwLength = ctypes.sizeof(mem)
        while not done.wait(1):
            assert _k32.GlobalMemoryStatusEx(ctypes.byref(mem))
            expired = time.monotonic() - started >= timeout
            low = mem.ullAvailPhys / 2**30 < min_physical_gb or mem.ullAvailPageFile / 2**30 < abort_commit_gb
            if expired or low:
                say("in-process TIMEOUT" if expired else "in-process memory reserve exhausted")
                kill_job_members(job)
                os._exit(124 if expired else 76)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    sys.argv = cmd
    sys.path.insert(0, os.path.dirname(os.path.abspath(cmd[0])))
    try:
        if cmd[0] == "-m":
            assert len(cmd) >= 2, "-m requires a module name"
            sys.argv = cmd[1:]
            runpy.run_module(cmd[1], run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(cmd[0], run_name="__main__")
    finally:
        done.set()
        watcher.join()
        killed = kill_job_members(job)
        left = [p for p in job_pids(job) if p != os.getpid()]
        say(f"in-process finished after {time.monotonic() - started:.1f}s; killed {killed}; members left: {len(left)}")
        assert not left


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=float, required=True, help="hard wall-clock limit of the command, seconds")
    ap.add_argument("--min-commit-gb", type=float, default=24.0, help="available commit needed to start")
    ap.add_argument("--wait-commit-s", type=float, default=120.0, help="how long to wait for --min-commit-gb")
    ap.add_argument("--abort-commit-gb", type=float, default=8.0, help="abort the run below this available commit")
    ap.add_argument("--min-physical-gb", type=float, default=0.0, help="minimum free physical memory throughout")
    ap.add_argument("--process-memory-gb", type=float, default=0.0, help="per-process job memory limit")
    ap.add_argument("--below-normal", action="store_true", help="constrain the job to BelowNormal priority")
    ap.add_argument("--in-process", action="store_true", help="run a Python script in this guarded interpreter")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    if _k32 is None:
        say("not Windows: running without job object / memory guard")
        return subprocess.run(cmd, timeout=a.timeout, check=False).returncode

    t_wait = time.monotonic() + a.wait_commit_s
    while (free := commit_available_gb()) < a.min_commit_gb:
        if time.monotonic() > t_wait:
            say(f"REFUSED: available commit {free:.1f} GiB < {a.min_commit_gb} GiB")
            return 75
        time.sleep(5)
    st = _MemStatus()
    st.dwLength = ctypes.sizeof(st)
    assert _k32.GlobalMemoryStatusEx(ctypes.byref(st))
    assert st.ullAvailPhys / 2**30 >= a.min_physical_gb, "insufficient free physical memory"
    job = make_job(a.process_memory_gb, a.below_normal)
    say(f"start (timeout {a.timeout:.0f}s, commit available {free:.1f} GiB): {' '.join(cmd)}")
    if a.in_process:
        run_in_process(job, cmd, a.timeout, a.min_physical_gb, a.abort_commit_gb)
        return 0
    t0 = time.monotonic()
    rc: int | None = None
    reason = "exit"
    min_free = free
    proc = subprocess.Popen(cmd)
    try:
        while True:
            try:
                rc = proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                pass
            free = commit_available_gb()
            min_free = min(min_free, free)
            assert _k32.GlobalMemoryStatusEx(ctypes.byref(st))
            if st.ullAvailPhys / 2**30 < a.min_physical_gb:
                reason, rc = "ABORTED: free physical memory below required reserve", 76
                break
            if free < a.abort_commit_gb:
                reason, rc = f"ABORTED: available commit {free:.1f} GiB < {a.abort_commit_gb} GiB", 76
                break
            if time.monotonic() - t0 > a.timeout:
                reason, rc = f"TIMEOUT after {a.timeout:.0f}s", 124
                break
    except KeyboardInterrupt:
        reason, rc = "INTERRUPTED", 130
    finally:
        killed = kill_job_members(job)
        left = [p for p in job_pids(job) if p != os.getpid()]
    say(f"{reason}: rc={rc} after {time.monotonic() - t0:.0f}s; min available commit {min_free:.1f} GiB; "
        f"killed {killed} leftover process(es); job members left: {len(left)}")
    return 1 if rc is None else rc


if __name__ == "__main__":
    sys.exit(main())
