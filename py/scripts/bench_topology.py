"""End-to-end topology benchmark: one ``spire.train`` run (collection + learning in steady state) measured
from outside. Run it under ``bench_guard.py`` (it reads the guard's job object for per-process CPU):

    python -B scripts/bench_guard.py --timeout 900 -- python -B scripts/bench_topology.py \
        --config ../bench/py-stage2-2/topo_real.toml --set ppo.epochs=1 --set actor.max_batch=64 \
        --warmup 60 --seconds 120 --tag procs-e1

* ``--config`` = a training config (TOML / JSON); ``--set a.b=value`` overrides (value parsed as JSON,
  else a string). The run gets its own directory under ``runs/bench-topo/``; ``duration_s`` is set to
  warmup + measurement (+ a few seconds), so the supervisor stops by itself.
* Refuses to start when the available commit is below ``--min-commit-gb`` or (real hosts) when more than
  ``--max-real-hosts`` (64) host processes would run.
* Measurement window = the last ``--seconds`` before the stop: decisions/s from the ``train`` records
  (actor reports) inside the window; learner from the ``learner`` records (samples/s while training,
  consumed samples/s, update period, epochs, staleness in versions and seconds); inference latency
  p50 / p99 (actor ``infer_ms``) and batch sizes; GPU utilisation / power (``gpu`` records); CPU cores per
  role from per-process CPU times of the job members (supervisor / learner / actors identified by the pids
  the run reports, host processes = everything else that burns CPU) and the background load of other
  programs (machine busy cores - this job); memory (private commit / working set) per role at the end.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
from ctypes import wintypes
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from bench_real_host import memory_status, process_memory  # noqa: E402
from spire.metrics import read_jsonl  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]


class _PidList(ctypes.Structure):
    _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD), ("NumberOfProcessIdsInList", wintypes.DWORD),
                ("ProcessIdList", ctypes.c_size_t * 4096)]


class _PE32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def job_pids() -> list[int]:
    """PIDs of this process's descendants (every process of the benchmark; the venv launcher lets its
    children break away from its job, so job enumeration would miss them)."""
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    parent: dict[int, int] = {}
    try:
        e = _PE32()
        e.dwSize = ctypes.sizeof(e)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            parent[int(e.th32ProcessID)] = int(e.th32ParentProcessID)
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    me = os.getpid()
    out: set[int] = set()
    frontier = {me}
    while frontier:
        nxt = {p for p, pp in parent.items() if pp in frontier and p not in out and p != me}
        out |= nxt
        frontier = nxt
    return sorted(out)


def proc_info(pid: int) -> tuple[str, float] | None:
    """(image name, CPU seconds user + kernel) of ``pid``."""
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return None
    try:
        c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
        if not k32.GetProcessTimes(wintypes.HANDLE(h), ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt),
                                   ctypes.byref(ut)):
            return None
        buf = ctypes.create_unicode_buffer(1024)
        n = wintypes.DWORD(1024)
        name = ""
        if k32.QueryFullProcessImageNameW(wintypes.HANDLE(h), 0, buf, ctypes.byref(n)):
            name = Path(buf.value).name.lower()

        def ft(x: wintypes.FILETIME) -> float:
            return ((x.dwHighDateTime << 32) | x.dwLowDateTime) / 1e7
        return name, ft(kt) + ft(ut)
    finally:
        k32.CloseHandle(wintypes.HANDLE(h))


def system_busy() -> float:
    idle, kern, user = (ctypes.c_uint64() for _ in range(3))
    k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    return (kern.value + user.value - idle.value) / 1e7


def snapshot() -> dict[str, Any]:
    procs = {}
    for pid in job_pids():
        pi = proc_info(pid)
        if pi is not None:
            procs[pid] = pi
    return {"wall": time.perf_counter(), "busy": system_busy(), "procs": procs}


def set_path(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parse_value(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def pct(v: list[float], qs: tuple[int, ...] = (50, 90, 99)) -> dict[str, float]:
    if not v:
        return {}
    a = np.asarray(v, np.float64)
    out = {f"p{q}": round(float(np.percentile(a, q)), 3) for q in qs}
    out["mean"] = round(float(a.mean()), 3)
    out["n"] = int(a.size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value override (JSON value)")
    ap.add_argument("--warmup", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--tag", default="topo")
    ap.add_argument("--idle-seconds", type=float, default=3.0)
    ap.add_argument("--min-commit-gb", type=float, default=30.0)
    ap.add_argument("--max-real-hosts", type=int, default=64)
    ap.add_argument("--out", default=str(REPO / "bench" / "py-stage2-2" / "topology.jsonl"))
    ap.add_argument("--keep-run", action="store_true", help="keep the run directory (default: keep metrics only)")
    a = ap.parse_args()
    p = Path(a.config)
    raw = tomllib.loads(p.read_text("utf-8")) if p.suffix == ".toml" else json.loads(p.read_text("utf-8"))
    for s in a.set:
        k, v = s.split("=", 1)
        set_path(raw, k, parse_value(v))
    stamp = time.strftime("%m%d-%H%M%S")
    raw["run_name"] = f"{a.tag}-{stamp}"
    raw["run_root"] = str(REPO / "runs" / "bench-topo")
    raw["duration_s"] = a.warmup + a.seconds + 3.0
    raw.setdefault("gpu_interval_s", 2.0)
    raw.setdefault("metrics_interval_s", 10.0)
    hosts = raw.get("hosts", {})
    n_hosts = int(raw.get("actor", {}).get("actors", 1)) * int(hosts.get("envs_per_actor", 16))
    if hosts.get("real") and n_hosts > a.max_real_hosts:
        print(f"refusing: {n_hosts} real hosts > {a.max_real_hosts}", file=sys.stderr)
        return 2
    mem0 = memory_status()
    if mem0["avail_commit_gb"] < a.min_commit_gb:
        print(f"refusing: available commit {mem0['avail_commit_gb']} GiB < {a.min_commit_gb}", file=sys.stderr)
        return 75
    run_dir = Path(raw["run_root"]) / raw["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "bench_config.json"
    cfg_path.write_text(json.dumps(raw, indent=1), encoding="utf-8")

    s0 = snapshot()
    time.sleep(a.idle_seconds)
    s1 = snapshot()
    idle_bg = (s1["busy"] - s0["busy"]) / (s1["wall"] - s0["wall"])

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    t_launch = time.monotonic()
    proc = subprocess.Popen([sys.executable, "-B", "-m", "spire.train", "--config", str(cfg_path)],
                            cwd=str(REPO / "py"), env=env, stdout=subprocess.DEVNULL,
                            stderr=open(run_dir / "train.stderr.log", "wb"))  # noqa: SIM115
    snaps: list[dict[str, Any]] = []
    mem: dict[str, Any] = {}
    stop = threading.Event()

    def sampler() -> None:
        t_meas = t_launch + a.warmup + 2.0
        t_mem = t_meas + a.seconds * 0.8
        while not stop.wait(0.5):
            now = time.monotonic()
            if now >= t_meas and len(snaps) == 0:
                snaps.append(snapshot())
            if now >= t_mem and not mem:
                mem["procs"] = {pid: process_memory(pid) for pid in job_pids()}
                mem["machine"] = memory_status()
            if now >= t_meas + a.seconds and len(snaps) == 1:
                snaps.append(snapshot())
                return
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        rc = proc.wait(timeout=a.warmup + a.seconds + 240)
    except subprocess.TimeoutExpired:
        (run_dir / "STOP").write_text("bench timeout", encoding="utf-8")
        rc = proc.wait(timeout=120)
    stop.set()
    th.join(5)
    recs = list(read_jsonl(run_dir / "metrics.jsonl"))
    res = summarize(recs, a, raw, snaps, mem, idle_bg, mem0)
    res["rc"] = rc
    res["run_dir"] = str(run_dir)
    line = json.dumps(res)
    print(line, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if not a.keep_run:  # weights / checkpoints are large and useless after a benchmark
        for sub in ("weights", "ckpt"):
            d = run_dir / sub
            if d.exists():
                for x in d.iterdir():
                    try:
                        x.unlink()
                    except OSError:
                        pass
    return 0 if rc == 0 else 1


def summarize(recs: list[dict[str, Any]], a: argparse.Namespace, raw: dict[str, Any], snaps: list[dict[str, Any]],
              mem: dict[str, Any], idle_bg: float, mem0: dict[str, float]) -> dict[str, Any]:
    events = [r for r in recs if r["kind"] == "event"]
    stop_ev = next((e for e in events if e.get("what") == "stop"), None)
    t_end = stop_ev["elapsed"] if stop_ev else max((r["elapsed"] for r in recs), default=0.0)
    t0 = t_end - a.seconds
    train = [r for r in recs if r["kind"] == "train" and r["elapsed"] >= t0 and r.get("decisions_per_s")]
    learn = [r for r in recs if r["kind"] == "learner" and r["elapsed"] >= t0]
    gpu = [r["gpus"][0] for r in recs if r["kind"] == "gpu" and r["elapsed"] >= t0 and r.get("gpus")]
    all_train = [r for r in recs if r["kind"] == "train"]
    w_train = [r for r in all_train if r["elapsed"] >= t0]
    dec = 0.0
    if len(w_train) >= 2:
        dec = (w_train[-1]["decisions"] - w_train[0]["decisions"]) / max(1e-9, w_train[-1]["elapsed"] - w_train[0]["elapsed"])
    pids: dict[str, set[int]] = {"supervisor": set(), "learner": set(), "actor": set()}
    for e in events:
        if e.get("what") == "start" and e.get("pid"):
            pids["supervisor"].add(int(e["pid"]))
        if e.get("what") == "learner_ready" and e.get("pid"):
            pids["learner"].add(int(e["pid"]))
    for r in recs:
        if r["kind"] == "train":
            for p in r.get("actor_pids", []) or []:
                pids["actor"].add(int(p))
    cores: dict[str, float] = {}
    if len(snaps) == 2:
        s0, s1 = snaps
        dt = s1["wall"] - s0["wall"]
        role_cpu: dict[str, float] = {"supervisor": 0.0, "learner": 0.0, "actor": 0.0, "hosts": 0.0, "other": 0.0}
        for pid, (name, cpu1) in s1["procs"].items():
            cpu0 = s0["procs"].get(pid, (name, 0.0))[1]
            d = cpu1 - cpu0
            role = next((k for k, v in pids.items() if pid in v), None)
            if role is None:
                role = "hosts" if name in ("dotnet.exe", "spirehost.exe", "spirehost.obs.exe") or \
                    (name == "python.exe" and not raw.get("hosts", {}).get("real")) else "other"
            role_cpu[role] += d
        cores = {k: round(v / dt, 2) for k, v in role_cpu.items()}
        busy = (s1["busy"] - s0["busy"]) / dt
        ours = sum(role_cpu.values()) / dt
        cores["machine_busy"] = round(busy, 2)
        cores["job_total"] = round(ours, 2)
        cores["background"] = round(max(0.0, busy - ours), 2)
    cores["background_before"] = round(idle_bg, 2)
    memo: dict[str, Any] = {}
    if mem.get("procs"):
        role_mem: dict[str, list[tuple[float, float]]] = {}
        for pid, m in mem["procs"].items():
            if m is None:
                continue
            role = next((k for k, v in pids.items() if pid in v), "hosts_or_other")
            role_mem.setdefault(role, []).append(m)
        memo = {k: {"private_gb": round(sum(x[0] for x in v) / 1024, 2), "ws_gb": round(sum(x[1] for x in v) / 1024, 2),
                    "n": len(v)} for k, v in role_mem.items()}
        memo["machine_after"] = mem.get("machine")
    memo["machine_before"] = mem0

    def tv(key: str) -> list[float]:
        return [r[key] for r in train if r.get(key) is not None]
    lr = learn
    lag_s = [x for r in lr for x in (r.get("lag_s") or {}).get("samples", [])]
    out: dict[str, Any] = {
        "tag": a.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "overrides": a.set,
        "topology": raw.get("topology", "processes"), "actors": raw.get("actor", {}).get("actors", 1),
        "envs_per_actor": raw.get("hosts", {}).get("envs_per_actor"), "real": bool(raw.get("hosts", {}).get("real")),
        "preset": raw.get("model", {}).get("preset", "2.5m"), "epochs": raw.get("ppo", {}).get("epochs", 2),
        "iteration_samples": raw.get("learner", {}).get("iteration_samples", 65536),
        "minibatch": raw.get("ppo", {}).get("minibatch", 4096),
        "window_s": round(t_end - t0, 1),
        "decisions_per_s": round(dec, 0),
        "decisions_per_s_reports": pct(tv("decisions_per_s")),
        "games_per_hour": pct(tv("games_per_hour")),
        "infer_ms_p50": pct(tv("infer_ms_p50")), "infer_ms_p99": pct(tv("infer_ms_p99")),
        "actor_latency_ms_p50": pct(tv("actor_latency_ms_p50")), "actor_latency_ms_p99": pct(tv("actor_latency_ms_p99")),
        "host_latency_ms_p50": pct(tv("host_latency_ms_p50")), "batch_mean": pct(tv("batch_mean")),
        "stage_us": train[-1]["stage_us"] if train else None,
        "throttled_frac": pct(tv("throttled_frac")),
        "learner": {
            "updates": len(lr),
            "update_period_s": round((lr[-1]["elapsed"] - lr[0]["elapsed"]) / max(1, len(lr) - 1), 2) if len(lr) > 1 else None,
            "samples_per_s_training": pct([r["samples_per_s"] for r in lr if r.get("samples_per_s")]),
            "consumed_samples_per_s": round(sum(r["samples"] for r in lr[1:]) / max(1e-9, lr[-1]["elapsed"] - lr[0]["elapsed"]), 0)
            if len(lr) > 1 else None,
            "time_train_s": pct([r["time_train"] for r in lr if r.get("time_train") is not None]),
            "wait_s": pct([r["wait_s"] for r in lr if r.get("wait_s") is not None]),
            "staleness_mean": pct([r["staleness_mean"] for r in lr if r.get("staleness_mean") is not None]),
            "staleness_max": max((r.get("staleness_max", 0) for r in lr), default=None),
            "segment_staleness_hist": _merge_hist([r.get("segment_staleness_hist", {}) for r in lr]),
            "lag_s": _merge_lag([r.get("lag_s") for r in lr]),
            "clipfrac_epoch0": pct([r["clipfrac_epoch0"] for r in lr if r.get("clipfrac_epoch0") is not None]),
            "approx_kl": pct([r["approx_kl"] for r in lr if r.get("approx_kl") is not None]),
            "ring_backlog": pct([r["ring_samples"] for r in lr if r.get("ring_samples") is not None]),
            "dropped_samples": _ring_drops(recs),
            "cuda_mem_mb": max((r.get("cuda_mem_mb", 0) for r in lr), default=None),
        },
        "gpu": {"util": pct([g["util"] for g in gpu]), "power_w": pct([g["power_w"] for g in gpu]),
                "mem_used_mb": pct([g["mem_used_mb"] for g in gpu])},
        "cores": cores, "memory": memo,
        "counts": train[-1]["counts"] if train else None,
        "stop": stop_ev,
    }
    del lag_s
    return out


def _merge_hist(hs: list[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for h in hs:
        for k, v in (h or {}).items():
            out[str(k)] = out.get(str(k), 0) + int(v)
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _merge_lag(ls: list[Any]) -> dict[str, float] | None:
    vals = [x for x in ls if x]
    if not vals:
        return None
    keys = ("p50", "p90", "p99", "mean", "max")
    return {k: round(float(np.mean([v[k] for v in vals if v.get(k) is not None])), 3) for k in keys
            if any(v.get(k) is not None for v in vals)}


def _ring_drops(recs: list[dict[str, Any]]) -> dict[str, int]:
    last = [r for r in recs if r["kind"] == "learner"]
    if not last:
        return {}
    ring = last[-1].get("ring") or {}
    return {k: v for k, v in ring.items() if "drop" in k or "evict" in k}


if __name__ == "__main__":
    sys.exit(main())
