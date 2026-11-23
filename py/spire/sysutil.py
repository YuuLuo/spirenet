"""Process placement on this Windows machine: priority class and CPU affinity (P-cores / E-cores).

The Ultra 9 285K has 8 performance cores and 16 efficiency cores (no SMT); the single-threaded Python
loops of the actor and the learner run measurably faster on P-cores, the host processes are many and
parallel. :func:`apply_placement` is called by the training processes on themselves (``actor.cpus`` /
``learner.cpus`` / ``*.priority`` in the training config) and by the host wrapper on each host process.

``cpus`` syntax: ``""`` (leave the OS default), ``"all"`` (every logical processor), ``"p"`` (all performance
cores), ``"e"`` (all efficiency cores), or a comma-separated list of logical processor numbers / ranges
(``"0,1,10-13"``). Everything is a no-op on other platforms or when the machine has no hybrid core classes
(``"p"`` = ``"e"`` = all cores).

Child processes inherit the affinity mask of their creator: a process restricted to the P-cores (an actor)
must give its host processes an explicit mask (:func:`affinity_restricted`; ``spire.host`` does).
"""
from __future__ import annotations

import ctypes
import gc
import os
import struct
import sys
import time
from functools import lru_cache

PRIORITY_CLASSES = {
    "idle": 0x40, "below_normal": 0x4000, "normal": 0x20, "above_normal": 0x8000, "high": 0x80,
}


@lru_cache(maxsize=1)
def core_classes() -> dict[int, int]:
    """Logical processor -> efficiency class (higher = faster; all 0 on non-hybrid machines)."""
    if sys.platform != "win32":
        return {}
    k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    n = ctypes.c_uint32(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))  # RelationProcessorCore
    buf = ctypes.create_string_buffer(n.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
        return {}
    out: dict[int, int] = {}
    off = 0
    while off < n.value:
        _rel, size = struct.unpack_from("<II", buf, off)
        eff = buf.raw[off + 9]
        (mask,) = struct.unpack_from("<Q", buf, off + 8 + 24)  # PROCESSOR_RELATIONSHIP.GroupMask[0].Mask
        for i in range(64):
            if mask >> i & 1:
                out[i] = eff
        off += size
    return out


def parse_cpus(spec: str) -> list[int]:
    """Logical processors named by ``spec`` (module docstring); [] = no restriction."""
    spec = spec.strip().lower()
    if not spec:
        return []
    classes = core_classes()
    if spec == "all":
        return sorted(classes) if classes else []
    if spec in ("p", "e"):
        if not classes:
            return []
        top = max(classes.values())
        want = [c for c, e in sorted(classes.items()) if (e == top) == (spec == "p")]
        return want or sorted(classes)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def apply_placement(priority: str = "", cpus: str = "", handle: int | None = None) -> dict[str, object]:
    """Set the priority class and affinity of the process ``handle`` (None = this process).
    Returns what was applied (for the run log); unknown names raise ``ValueError``."""
    applied: dict[str, object] = {}
    if sys.platform != "win32":
        return applied
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    h = ctypes.c_void_p(k32.GetCurrentProcess() if handle is None else handle)
    if priority:
        cls = PRIORITY_CLASSES.get(priority.lower())
        if cls is None:
            raise ValueError(f"unknown priority {priority!r} (one of {sorted(PRIORITY_CLASSES)})")
        if k32.SetPriorityClass(h, cls):
            applied["priority"] = priority.lower()
    cl = parse_cpus(cpus)
    if cl:
        mask = 0
        for c in cl:
            mask |= 1 << c
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if k32.SetProcessAffinityMask(h, mask):
            applied["cpus"] = cl
    return applied


def affinity_restricted() -> bool:
    """True if this process may not run on every logical processor (its children would inherit that)."""
    if sys.platform != "win32":
        return False
    k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    proc, system = ctypes.c_size_t(0), ctypes.c_size_t(0)
    if not k32.GetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.byref(proc),
                                      ctypes.byref(system)):
        return False
    return proc.value != system.value


GPU_PRIORITY_CLASSES = {"idle": 0, "below_normal": 1, "normal": 2, "above_normal": 3, "high": 4}
"""D3DKMT_SCHEDULINGPRIORITYCLASS of the Windows GPU scheduler (WDDM time-slices the GPU between processes;
REALTIME is deliberately not offered: it can starve the desktop compositor)."""


def set_gpu_priority(name: str) -> str:
    """Set this process's GPU scheduling priority class (WDDM); returns the class now in effect ("" if not
    applicable). Unknown names raise ``ValueError``."""
    if not name or sys.platform != "win32":
        return ""
    cls = GPU_PRIORITY_CLASSES.get(name.lower())
    if cls is None:
        raise ValueError(f"unknown GPU priority {name!r} (one of {sorted(GPU_PRIORITY_CLASSES)})")
    try:
        gdi = ctypes.WinDLL("gdi32")  # type: ignore[attr-defined]
        k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        f = gdi.D3DKMTSetProcessSchedulingPriorityClass
        f.argtypes = [ctypes.c_void_p, ctypes.c_int]
        f.restype = ctypes.c_long
        if f(ctypes.c_void_p(k32.GetCurrentProcess()), cls) != 0:
            return ""
    except (OSError, AttributeError):
        return ""
    return name.lower()


def cuda_blocking_sync(device_index: int = 0) -> bool:
    """Make this process's CUDA synchronisations block (cudaDeviceScheduleBlockingSync) instead of
    spin-waiting. Must run before the process creates its CUDA context. The learner process waits for the
    GPU for most of an update (it queues a whole minibatch ahead): spinning there burns one to two cores the
    host processes need (measured, py-stage2-2). Returns True if the flag was set.

    Linux: the flag is set on the primary context of ``device_index`` (the device the process will use) through
    the CUDA runtime torch loaded; Windows: on the default device."""
    if sys.platform == "linux":
        import torch  # loads the CUDA runtime library torch was built with
        assert torch.version.cuda is not None, "CPU-only torch build"
        cudart = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
        assert int(cudart.cudaSetDevice(ctypes.c_int(device_index))) == 0
        return int(cudart.cudaSetDeviceFlags(ctypes.c_uint(0x04))) == 0  # cudaDeviceScheduleBlockingSync
    if sys.platform != "win32":
        return False
    try:
        import torch  # the runtime DLL shipped with torch (the one the context will be created with)
        lib = __import__("pathlib").Path(torch.__file__).parent / "lib"
        dlls = sorted(lib.glob("cudart64_*.dll"))
        if not dlls:
            return False
        cudart = ctypes.WinDLL(str(dlls[-1]))  # type: ignore[attr-defined]
        return int(cudart.cudaSetDeviceFlags(ctypes.c_uint(0x04))) == 0  # cudaDeviceScheduleBlockingSync
    except OSError:
        return False


GC_POLICIES = ("off", "freeze")
GC_THRESHOLD = (50_000, 20, 100)
"""Collector thresholds of policy ``freeze`` (CPython's default: 700, 10, 10)."""


class GcPolicy:
    """The cyclic garbage collector of a training process (processes topology: the learner and every actor), chosen
    by the environment variable ``SPIRE_GC_POLICY`` (notes/next/round9-1119.md A2, G1):

    * ``off``: nothing is called (CPython's defaults).
    * ``freeze`` (default): at process start ``gc.set_threshold(*GC_THRESHOLD)`` and a ``gc.callbacks`` counter
      (collections and seconds of generations 1 and 2, cumulative); after warm-up (:meth:`settle`) ONE
      ``gc.collect(); gc.freeze()``: everything alive then (compiled models, captured graphs, tokenizer tables) leaves
      the collector's generations for good (reference counting still frees it), so a full collection no longer walks
      it. No periodic re-freeze: objects frozen while alive that later die in a cycle would never be reclaimed.

    A full collection holds the GIL for its whole duration in whichever thread crosses the threshold; with millions
    of long-lived tracked objects it stalled the learner for ~2 s every ~9 s (notes/next/round9-1119.md)."""

    def __init__(self, policy: str) -> None:
        assert policy in GC_POLICIES, f"SPIRE_GC_POLICY {policy!r} (one of {GC_POLICIES})"
        self.policy = policy
        self.pending = policy == "freeze"  # a settle (collect + freeze) is still due
        self.frozen = False
        self.n = [0, 0, 0]
        self.s = [0.0, 0.0, 0.0]
        self._t0 = 0.0
        if policy == "freeze":
            gc.set_threshold(*GC_THRESHOLD)
            gc.callbacks.append(self._callback)

    def _callback(self, phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            self._t0 = time.perf_counter()
            return
        g = info["generation"]
        self.n[g] += 1
        self.s[g] += time.perf_counter() - self._t0

    def settle(self) -> None:
        """The end of warm-up: one full collection, then freeze (once per process; no-op with policy off)."""
        if self.pending:
            self.pending = False
            gc.collect()
            gc.freeze()
            self.frozen = True

    def counters(self) -> dict[str, float | int]:
        """Collections and seconds of generations 1 and 2 since the process started (the settle's own included)."""
        return {"gen1_n": self.n[1], "gen1_s": round(self.s[1], 4), "gen2_n": self.n[2], "gen2_s": round(self.s[2], 4)}

    def fields(self) -> dict[str, object]:
        """The learner row's ``gc`` field."""
        return {"policy": self.policy, "frozen": self.frozen, **self.counters()}


_GC_POLICY: GcPolicy | None = None


def gc_policy() -> GcPolicy:
    """This process's :class:`GcPolicy`, installed at the first call (``SPIRE_GC_POLICY``, default ``freeze``)."""
    global _GC_POLICY
    if _GC_POLICY is None:
        _GC_POLICY = GcPolicy(os.environ.get("SPIRE_GC_POLICY", "freeze"))
    return _GC_POLICY
