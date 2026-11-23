"""Actor throughput / per-decision Python overhead against fake hosts.

Hosts serve pooled frames (``FakeHostConfig.frame_pool``: the host side costs microseconds),
either in-process (``--hosts inproc``: zero latency, host work runs inside the actor thread and
is part of the ``send`` stage) or as real subprocesses (``--hosts proc``) over stdin/stdout
(``--transport stdio``) or named pipes (``--transport pipe``, notes/transport-spec.md), with an
optional ``--latency-ms`` per reply (busy-waited below 2 ms). The actor runs in train mode
(tokenize + rollout recording + segment sink) with the uniform policy (no GPU) or SpireNet on CUDA
graphs. Steady state only: the first ``--warmup`` seconds (pool generation, graph captures) are
excluded. ``--envs`` may list several host counts (one run each, same process).

Every result line also reports the machine load during the measured window: total busy cores
(GetSystemTimes), the cores used by this benchmark (actor process + every host process, via the
hosts' job object) and the difference, the background load of other programs.

    python -B scripts/bench_actor.py --hosts proc --transport pipe --envs 32,64,96,128 --policy net
    python -B scripts/bench_actor.py --hosts proc --transport pipe --envs 64 --latency-ms 0.2
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from spire import host as host_mod  # noqa: E402
from spire.actor import (  # noqa: E402
    STAGES,
    Actor,
    ActorConfig,
    EpisodeDist,
    EpisodeSampler,
    RunTotals,
    TorchPolicy,
    UniformPolicy,
)
from spire.host import Host  # noqa: E402
from spire.ipc import ShmRing  # noqa: E402
from spire.model import PRESETS, SpireNet  # noqa: E402
from spire.rewards import RoomIds  # noqa: E402
from spire.rollout import RolloutConfig, RolloutStore, Segment  # noqa: E402
from spire.synth import FakeHost, FakeHostConfig  # noqa: E402
from spire.tokenize import Tokenizer  # noqa: E402
from spire.train import HostSection, make_hosts  # noqa: E402
from spire.vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- machine load (Windows)

class _Basic(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", ctypes.c_uint32), ("TotalProcesses", ctypes.c_uint32),
                ("ActiveProcesses", ctypes.c_uint32), ("TotalTerminatedProcesses", ctypes.c_uint32)]


def cpu_snapshot() -> dict[str, float]:
    """Seconds of CPU: machine busy (all cores), this process, the host job (all host processes)."""
    if sys.platform == "linux":  # /proc: machine busy jiffies, and the CPU time of every child (the hosts)
        tick = os.sysconf("SC_CLK_TCK")
        cpu = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        kids = [c for t in Path("/proc/self/task").iterdir() for c in (t / "children").read_text().split()]
        job = 0
        for pid in kids:
            stat = Path(f"/proc/{pid}/stat").read_text()
            job += sum(int(x) for x in stat.rsplit(")", 1)[1].split()[11:13])
        return {"wall": time.perf_counter(), "busy": (sum(cpu[:8]) - cpu[3] - cpu[4]) / tick,
                "self": time.process_time(), "job": job / tick}
    if sys.platform != "win32":
        return {"wall": time.perf_counter(), "busy": 0.0, "self": time.process_time(), "job": 0.0}
    k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    idle, kern, user = (ctypes.c_uint64() for _ in range(3))
    k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    job = 0.0
    if host_mod._JOB:
        info = _Basic()
        k32.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
                                                  ctypes.c_void_p]
        if k32.QueryInformationJobObject(ctypes.c_void_p(host_mod._JOB), 1, ctypes.byref(info),
                                         ctypes.sizeof(info), None):
            job = (info.TotalUserTime + info.TotalKernelTime) / 1e7
    return {"wall": time.perf_counter(), "busy": (kern.value + user.value - idle.value) / 1e7,
            "self": time.process_time(), "job": job}


def load_between(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    dt = max(b["wall"] - a["wall"], 1e-9)
    busy = (b["busy"] - a["busy"]) / dt
    ours = (b["self"] - a["self"] + b["job"] - a["job"]) / dt
    return {"machine_busy_cores": round(busy, 2), "bench_cores": round(ours, 2),
            "actor_cores": round((b["self"] - a["self"]) / dt, 2), "hosts_cores": round((b["job"] - a["job"]) / dt, 2),
            "background_cores": round(max(0.0, busy - ours), 2)}


def idle_load(seconds: float = 2.0) -> dict[str, float]:
    a = cpu_snapshot()
    time.sleep(seconds)
    return load_between(a, cpu_snapshot())


# --------------------------------------------------------------------------- one run

def run_one(a: argparse.Namespace, envs: int, vocab: Vocab, tok: Tokenizer, net: SpireNet | None) -> dict[str, Any]:
    fake = FakeHostConfig(frame_pool=16, mean_decisions=300, hashes=False, abort_prob=0.0, size=a.size)
    pre = idle_load(a.idle_seconds)
    if a.hosts == "inproc":
        hosts: list[Host] = [FakeHost(vocab, seed=i, config=fake) for i in range(envs)]
    else:
        hosts = make_hosts(HostSection(fake=fake, fake_latency_ms=a.latency_ms, transport=a.transport), envs,
                           seed=1, log_dir=REPO / "runs" / "host-logs" / "bench", name="bench")
    policy: UniformPolicy | TorchPolicy = UniformPolicy(0) if net is None else TorchPolicy(net, tok)
    ring = consumer = owner = None
    if a.shm:
        import multiprocessing as mp
        owner, spec = ShmRing.create(256 << 20)
        rx, tx = mp.Pipe(duplex=False)
        ring, consumer = ShmRing.attach(spec, tx), ShmRing.attach(spec, rx)

        def sink(seg: Segment) -> None:
            assert ring is not None and consumer is not None
            ring.put(seg)
            consumer.get(rx.recv())
    else:
        def sink(seg: Segment) -> None:
            pass

    store = RolloutStore(sink, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=256))
    reports: list[dict[str, Any]] = []
    snaps: list[dict[str, float]] = []
    acc = [0.0]

    def on_report(rep: dict[str, Any]) -> None:
        acc[0] += rep["seconds"]
        if acc[0] >= a.warmup:
            if not snaps:
                snaps.append(cpu_snapshot())
            else:
                reports.append(rep)

    actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=((a.lineup, 1.0),)), vocab, 0),
                  ActorConfig(max_batch=a.max_batch, max_wait_ms=a.max_wait_ms, max_inflight=a.max_inflight,
                              min_group=a.min_group, report_interval=1.0),
                  store=store, on_report=on_report)
    t0 = time.monotonic()
    try:
        actor.run(max_seconds=a.warmup + a.seconds + 1.0)
        end = cpu_snapshot()
    finally:
        for h in hosts:
            h.close()
        if isinstance(policy, TorchPolicy):
            policy.close()
    elapsed = time.monotonic() - t0
    reports = [r for r in reports if r["seconds"] > 0.5]
    tot = RunTotals()
    for r in reports:
        tot.add(r)
    us = tot.us_per_decision()
    bm = [r["batch_mean"] for r in reports if r["batch_mean"]]
    lat = [r["infer_ms"][0] for r in reports if r["infer_ms"][0] is not None]

    def med(key: str, i: int) -> float | None:
        v = sorted(r[key][i] for r in reports if r[key][i] is not None)
        return round(v[len(v) // 2], 3) if v else None
    res = {
        "hosts": a.hosts, "transport": a.transport if a.hosts == "proc" else None, "envs": envs,
        "latency_ms": a.latency_ms, "policy": a.policy, "preset": a.preset if net is not None else None,
        "size": a.size, "lineup": a.lineup, "max_batch": a.max_batch, "max_wait_ms": a.max_wait_ms,
        "max_inflight": a.max_inflight, "min_group": a.min_group, "shm": a.shm,
        "decisions_per_s": round(tot.decisions / max(tot.seconds, 1e-9), 1),
        "us_per_decision": {k: round(v, 2) for k, v in us.items() if k in STAGES or k == "python"},
        "batch_mean": round(sum(bm) / len(bm), 1) if bm else None,
        "infer_ms_p50": round(sum(lat) / len(lat), 3) if lat else None,
        "host_latency_ms_p50_p99": [med("host_latency_ms", 0), med("host_latency_ms", 1)],
        "actor_latency_ms_p50_p99": [med("actor_latency_ms", 0), med("actor_latency_ms", 1)],
        "measured_s": round(tot.seconds, 1), "decisions": tot.decisions, "elapsed": round(elapsed, 1),
        "load_before": pre, "load": load_between(snaps[0], end) if snaps else None,
        "transfer_us_per_step": round(ring.stats.encode_ns / 1e3 / max(1, store.stats.steps_released), 2)
        if ring is not None else None,
        "graph_captures": getattr(getattr(policy, "runner", None), "captures", None),
        "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if ring is not None and consumer is not None and owner is not None:
        ring.close()
        consumer.close()
        owner.close()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", choices=("inproc", "proc"), default="proc")
    ap.add_argument("--transport", choices=("stdio", "pipe"), default="pipe")
    ap.add_argument("--envs", default="32", help="host count, or a comma-separated list (one run each)")
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--policy", choices=("uniform", "net"), default="net")
    ap.add_argument("--preset", default="2.5m")
    ap.add_argument("--size", choices=("typical", "extreme"), default="typical")
    ap.add_argument("--lineup", default="IRONCLAD")
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--max-wait-ms", type=float, default=1.0)
    ap.add_argument("--max-inflight", type=int, default=2)
    ap.add_argument("--min-group", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=12.0)
    ap.add_argument("--idle-seconds", type=float, default=2.0, help="background-load sample before each run")
    ap.add_argument("--priority", action="store_true", help="actor ABOVE_NORMAL priority class")
    ap.add_argument("--shm", action="store_true", help="send segments through a ShmRing (consumer drains inline)")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(REPO / "bench" / "actor_bench.jsonl"))
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    if a.priority and sys.platform == "win32":
        k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000)
    vocab = Vocab(make_synthetic_vocab())
    tok = Tokenizer(vocab)
    net = None
    if a.policy == "net":
        catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
        net = SpireNet(tok.layout, PRESETS[a.preset], catalog).cuda()
    for envs in (int(x) for x in a.envs.split(",")):
        res = run_one(a, envs, vocab, tok, net)
        res["tag"] = a.tag
        print(json.dumps(res), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
