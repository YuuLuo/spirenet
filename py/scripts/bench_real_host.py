"""Actor throughput on the REAL SpireHost: spire.actor driving N real host processes over named pipes.

    python -B scripts/bench_guard.py --timeout 900 -- python -B scripts/bench_real_host.py --envs 16,32 --policy uniform
    python -B scripts/bench_real_host.py --envs 32 --policy net --host-args "--hp 99999 --turn-cap 100" --tag long

The hosts come from a real-host launch spec (``--spec``, default ``py/configs/real_host.toml``); the vocab /
catalog of the spec build the tokenizer and (``--policy net``) an untrained SpireNet (``--preset``, CUDA graphs).
Train mode: tokenize + rollout recording + segment sink, exactly as in training. After ``--warmup`` seconds
(host boot, JIT, graph captures) the actor reports of ``--seconds`` are accumulated. Per run one JSON line:

* decisions/s, games/hour (episodes that reached a TERMINAL frame within the window, x3600 / window), Python
  us/decision by stage (``spire.actor.STAGES``), batch sizes, inference latency;
* host side: HOST_US (the host's own frame-encode time, header word 14) and the round-trip latency STEP sent
  -> frame processed (actor report percentiles), per decision kind where sampled;
* frames: bytes and token counts per decision kind (every ``--sample-every``-th frame of every host is kept and
  tokenized after the run), act / floor reached, results;
* machine: busy cores (GetSystemTimes), cores used by the actor and by the host processes (job object), the
  background load of other programs, GPU utilisation / memory (nvidia-smi every second), private commit and
  working set of every host process and of the actor, available commit before / after.

A run refuses to start more than 64 hosts or when the available commit is below ``--min-commit-gb``.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import json
import os
import shlex
import struct
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench_actor import cpu_snapshot, idle_load, load_between  # noqa: E402
from spire.actor import STAGES, Actor, ActorConfig, EpisodeDist, EpisodeSampler, RunTotals, TorchPolicy, UniformPolicy  # noqa: E402,E501
from spire.host import Host, HostProcess, real_host_spec  # noqa: E402
from spire.ipc import pack_segment  # noqa: E402
from spire.metrics import EpisodeRecord, query_gpu  # noqa: E402
from spire.model import PRESETS, SpireNet  # noqa: E402
from spire.proto import parse_frame  # noqa: E402
from spire.rewards import RoomIds  # noqa: E402
from spire.rollout import RolloutConfig, RolloutStore, Segment  # noqa: E402
from spire.schema import G, Kind  # noqa: E402
from spire.tokenize import Tokenizer, TokenizerConfig  # noqa: E402
from spire.train import HostSection, make_hosts  # noqa: E402
from spire.vocab import Catalog, Vocab  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MAX_REAL_HOSTS = 64
_HDR = struct.Struct("<16i")
_KIND_OFF, _ACT_OFF, _FLOOR_OFF = 64 + 4 * int(G.KIND), 64 + 4 * int(G.ACT_INDEX), 64 + 4 * int(G.TOTAL_FLOOR)
_I = struct.Struct("<i")


# --------------------------------------------------------------------------- memory (Windows)

class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


def memory_status() -> dict[str, float]:
    st = _MemStatus()
    st.dwLength = ctypes.sizeof(st)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
    return {"avail_phys_gb": round(st.ullAvailPhys / 2**30, 1), "avail_commit_gb": round(st.ullAvailPageFile / 2**30, 1),
            "commit_limit_gb": round(st.ullTotalPageFile / 2**30, 1)}


def process_memory(pid: int | None) -> tuple[float, float] | None:
    """(private commit MB, working set MB) of ``pid``."""
    if pid is None:
        return None
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi  # type: ignore[attr-defined]
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
    if not h:
        return None
    try:
        c = _PMC()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(wintypes.HANDLE(h), ctypes.byref(c), c.cb):
            return None
        return c.PrivateUsage / 2**20, c.WorkingSetSize / 2**20
    finally:
        k32.CloseHandle(wintypes.HANDLE(h))


# --------------------------------------------------------------------------- frame statistics


class FrameTap:
    """Wraps ``host.receive``: per frame (kind, bytes, HOST_US, act, floor) into flat lists, and every
    ``sample_every``-th payload kept for offline tokenization. Recording starts with :meth:`arm`."""

    def __init__(self, sample_every: int) -> None:
        self.sample_every = sample_every
        self.armed = False
        self.rows: list[tuple[int, int, int, int, int]] = []
        self.samples: list[bytes] = []
        self._n = 0

    def arm(self) -> None:
        self.armed = True

    def wrap(self, host: Host) -> None:
        inner = host.receive

        def receive(payload: bytes | bytearray) -> Any:
            f = inner(payload)
            if self.armed and f.msg_type != 3:
                self.rows.append((_I.unpack_from(payload, _KIND_OFF)[0], len(payload), _I.unpack_from(payload, 56)[0],
                                  _I.unpack_from(payload, _ACT_OFF)[0], _I.unpack_from(payload, _FLOOR_OFF)[0]))
                self._n += 1
                if self._n % self.sample_every == 0:
                    self.samples.append(bytes(payload))
            return f
        host.receive = receive  # type: ignore[method-assign]


def pct(v: Sequence[float] | np.ndarray, qs: tuple[int, ...] = (50, 90, 99)) -> dict[str, float]:
    a = np.asarray(v, dtype=np.float64)
    if not a.size:
        return {}
    out = {f"p{q}": round(float(np.percentile(a, q)), 1) for q in qs}
    out["mean"] = round(float(a.mean()), 1)
    out["max"] = round(float(a.max()), 1)
    out["n"] = int(a.size)
    return out


def frame_report(tap: FrameTap, tok: Tokenizer) -> dict[str, Any]:
    rows = np.asarray(tap.rows, np.int64).reshape(-1, 5)
    by_kind: dict[str, Any] = {}
    for k in sorted(set(rows[:, 0].tolist())):
        sel = rows[rows[:, 0] == k]
        by_kind[Kind(k).name] = {"frames": int(len(sel)), "bytes": pct(sel[:, 1]), "host_us": pct(sel[:, 2])}
    obs = rows[rows[:, 0] != int(Kind.TERMINAL)]
    tokens: dict[str, list[int]] = collections.defaultdict(list)
    tok_act: dict[int, list[int]] = collections.defaultdict(list)
    rowsn: dict[str, list[int]] = collections.defaultdict(list)
    for p in tap.samples:
        f = parse_frame(p)
        if f.msg_type != 1:
            continue
        s = tok.encode(f)
        name = Kind(f.kind).name
        tokens[name].append(s.n_tokens)
        tokens["ALL"].append(s.n_tokens)
        tok_act[int(f.gvals[G.ACT_INDEX])].append(s.n_tokens)
        rowsn[name].append(int(sum(r for r in f.rows if r > 0)))
    for k, v in tokens.items():
        if k in by_kind:
            by_kind[k]["tokens"] = pct(v, (50, 90, 99, 100))
            by_kind[k]["table_rows"] = pct(rowsn[k])
    return {
        "frames": int(len(rows)), "obs_frames": int(len(obs)),
        "bytes_all_obs": pct(obs[:, 1]) if len(obs) else {}, "host_us_all_obs": pct(obs[:, 2]) if len(obs) else {},
        "by_kind": by_kind, "tokens_all": pct(tokens.get("ALL", []), (50, 90, 95, 99, 100)),
        "tokens_by_act": {str(a): pct(v, (50, 90, 99, 100)) for a, v in sorted(tok_act.items())},
        "act_of_obs_frames": {str(a): int(n) for a, n in zip(*np.unique(obs[:, 3], return_counts=True))} if len(obs) else {},
        "token_samples": tokens.get("ALL", []),
    }


# --------------------------------------------------------------------------- one run


def run_one(a: argparse.Namespace, envs: int, vocab: Vocab, catalog: Catalog, tok: Tokenizer,
            net: SpireNet | None) -> dict[str, Any]:
    if envs > MAX_REAL_HOSTS:
        raise SystemExit(f"refusing to start {envs} real hosts (limit {MAX_REAL_HOSTS})")
    mem0 = memory_status()
    if mem0["avail_commit_gb"] < a.min_commit_gb:
        raise SystemExit(f"available commit {mem0['avail_commit_gb']} GiB < {a.min_commit_gb} GiB: not starting")
    pre = idle_load(a.idle_seconds)
    sec = HostSection(real=True, real_config=a.spec or "", extra_args=tuple(shlex.split(a.host_args)), transport="pipe",
                      start_timeout=180.0, reset_timeout=60.0, step_timeout=60.0)
    log_dir = REPO / "runs" / "host-logs" / "bench-real"
    t_boot = time.monotonic()
    hosts = make_hosts(sec, envs, seed=1, log_dir=log_dir, name=f"br{envs}",
                       expected_hashes=(vocab.hash, catalog.hash))
    tap = FrameTap(a.sample_every)
    for h in hosts:
        tap.wrap(h)
    policy: UniformPolicy | TorchPolicy = UniformPolicy(0) if net is None else TorchPolicy(net, tok)
    segs = [0]
    dump = open(a.dump, "ab") if a.dump else None  # noqa: SIM115 - closed below
    dumped = [0]

    def sink(seg: Segment) -> None:
        segs[0] += len(seg)
        if dump is not None and dumped[0] < a.dump_samples:  # real segments for learner benchmarks
            blob = pack_segment(seg)
            dump.write(struct.pack("<Q", len(blob)) + blob)
            dumped[0] += len(seg)

    store = RolloutStore(sink, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=256))
    reports: list[dict[str, Any]] = []
    snaps: list[dict[str, float]] = []
    episodes: list[EpisodeRecord] = []
    gpu: list[dict[str, float]] = []
    gpu_stop = threading.Event()
    acc = [0.0]
    t_measure = [0.0]
    mem_samples: dict[str, Any] = {}

    def gpu_loop() -> None:
        while not gpu_stop.wait(1.0):
            if t_measure[0]:
                rows = query_gpu()
                if rows:
                    gpu.append(rows[0])

    def on_report(rep: dict[str, Any]) -> None:
        acc[0] += rep["seconds"]
        if acc[0] >= a.warmup:
            if not snaps:
                snaps.append(cpu_snapshot())
                t_measure[0] = time.monotonic()
                tap.arm()
            else:
                reports.append(rep)
                if not mem_samples and acc[0] >= a.warmup + a.seconds / 2:
                    pm = [process_memory(h.pid) for h in hosts if isinstance(h, HostProcess)]
                    ok = [m for m in pm if m is not None]
                    me = process_memory(os.getpid())
                    mem_samples.update({
                        "host_private_mb": pct([m[0] for m in ok]), "host_ws_mb": pct([m[1] for m in ok]),
                        "hosts_private_total_gb": round(sum(m[0] for m in ok) / 1024, 2),
                        "hosts_ws_total_gb": round(sum(m[1] for m in ok) / 1024, 2),
                        "actor_private_mb": round(me[0], 0) if me else None, "actor_ws_mb": round(me[1], 0) if me else None,
                        "machine": memory_status()})

    def on_episode(rec: EpisodeRecord) -> None:
        if t_measure[0]:
            episodes.append(rec)

    gt = threading.Thread(target=gpu_loop, daemon=True)
    gt.start()
    sampler = EpisodeSampler(EpisodeDist(lineups=tuple((n, float(w)) for n, w in a.lineups)), vocab, 0)
    actor = Actor(hosts, policy, tok, sampler,
                  ActorConfig(max_batch=a.max_batch, max_wait_ms=a.max_wait_ms, min_group=a.min_group,
                              report_interval=1.0, max_episode_decisions=a.max_episode_decisions),
                  store=store, on_report=on_report, on_episode=on_episode)
    t0 = time.monotonic()
    prof = None
    if a.profile:
        import cProfile
        prof = cProfile.Profile()
    try:
        if prof is not None:
            prof.enable()
        actor.run(max_seconds=a.warmup + a.seconds + 1.0)
        if prof is not None:
            prof.disable()
            prof.dump_stats(a.profile)
        end = cpu_snapshot()
        t_end = time.monotonic()
    finally:
        gpu_stop.set()
        if dump is not None:
            dump.close()
        for h in hosts:
            h.close()
        if isinstance(policy, TorchPolicy):
            policy.close()
    window = t_end - t_measure[0] if t_measure[0] else 0.0
    reports = [r for r in reports if r["seconds"] > 0.5]
    tot = RunTotals()
    for r in reports:
        tot.add(r)
    us = tot.us_per_decision()

    def med(key: str, i: int) -> float | None:
        v = sorted(r[key][i] for r in reports if r.get(key) and r[key][i] is not None)
        return round(v[len(v) // 2], 3) if v else None
    bm = [r["batch_mean"] for r in reports if r["batch_mean"]]
    ended = [e for e in episodes if e.result != "shutdown"]
    results = collections.Counter(e.result for e in ended)
    games = sum(n for k, n in results.items() if k in ("win", "loss", "decision_cap", "turn_cap", "truncated", "aborted"))
    fr = frame_report(tap, tok)
    token_samples = fr.pop("token_samples")
    host_tot: collections.Counter[str] = collections.Counter()
    for r in reports:
        host_tot.update(r["host"])
    res = {
        "tag": a.tag, "envs": envs, "policy": a.policy, "preset": a.preset if net is not None else None,
        "host_args": a.host_args, "max_episode_decisions": a.max_episode_decisions, "lineups": a.lineups,
        "max_batch": a.max_batch, "max_wait_ms": a.max_wait_ms, "min_group": a.min_group,
        "token_buckets": list(tok.config.token_buckets),
        "decisions_per_s": round(tot.decisions / max(tot.seconds, 1e-9), 1),
        "games_per_hour": round(games / max(window, 1e-9) * 3600, 0), "games_in_window": games,
        "results": dict(results), "decisions_per_game": pct([e.decisions for e in ended]),
        "floor_reached": pct([e.floor for e in ended]), "act_reached": dict(collections.Counter(e.act for e in ended)),
        "us_per_decision": {k: round(v, 2) for k, v in us.items() if k in STAGES or k == "python"},
        "batch_mean": round(sum(bm) / len(bm), 1) if bm else None,
        "infer_ms_p50_p99": [med("infer_ms", 0), med("infer_ms", 1)],
        "host_roundtrip_ms_p50_p99": [med("host_latency_ms", 0), med("host_latency_ms", 1)],
        "actor_latency_ms_p50_p99": [med("actor_latency_ms", 0), med("actor_latency_ms", 1)],
        "host_counters": {k: v for k, v in host_tot.items() if v},
        "frames": fr, "measured_s": round(tot.seconds, 1), "window_s": round(window, 1),
        "decisions": tot.decisions, "elapsed": round(time.monotonic() - t0, 1),
        "load_before": pre, "load": load_between(snaps[0], end) if snaps else None,
        "gpu": {"util": pct([g["util"] for g in gpu]), "mem_used_mb": pct([g["mem_used_mb"] for g in gpu]),
                "power_w": pct([g["power_w"] for g in gpu])} if gpu else None,
        "memory": mem_samples, "memory_before": mem0, "memory_after": memory_status(),
        "graph_captures": getattr(getattr(policy, "runner", None), "captures", None),
        "rollout_steps_released": segs[0], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "boot_s": round(t0 - t_boot, 1),
    }
    if a.tokens_out:
        with open(a.tokens_out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tag": a.tag, "envs": envs, "policy": a.policy, "host_args": a.host_args,
                                "tokens": token_samples}) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="16", help="host count(s), comma-separated (<= 64 each)")
    ap.add_argument("--policy", choices=("uniform", "net"), default="uniform")
    ap.add_argument("--preset", default="2.5m")
    ap.add_argument("--spec", default=None, help="real-host launch spec TOML")
    ap.add_argument("--host-args", default="", help='extra host arguments, e.g. "--hp 99999 --turn-cap 100"')
    ap.add_argument("--max-episode-decisions", type=int, default=0)
    ap.add_argument("--lineups", default="IRONCLAD:0.7,SILENT:0.3")
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--max-wait-ms", type=float, default=1.0)
    ap.add_argument("--min-group", type=int, default=16)
    ap.add_argument("--buckets", default="", help="token bucket edges, comma-separated (default: TokenizerConfig)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--idle-seconds", type=float, default=3.0)
    ap.add_argument("--sample-every", type=int, default=25, help="keep every N-th frame for token statistics")
    ap.add_argument("--min-commit-gb", type=float, default=30.0)
    ap.add_argument("--profile", default="", help="cProfile the actor run into this file (diagnostics only)")
    ap.add_argument("--dump", default="", help="append released segments (u64 length + spire.ipc blob) to this file")
    ap.add_argument("--dump-samples", type=int, default=262144, help="stop dumping after this many samples")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(REPO / "bench" / "py-stage2-1" / "real_bench.jsonl"))
    ap.add_argument("--tokens-out", default=str(REPO / "bench" / "py-stage2-1" / "real_tokens.jsonl"))
    a = ap.parse_args()
    a.lineups = [(x.split(":")[0], float(x.split(":")[1])) for x in a.lineups.split(",")]
    torch.set_num_threads(1)
    spec = real_host_spec(a.spec)
    vocab = Vocab.load(spec.vocab)
    catalog = Catalog.load(spec.catalog, vocab)
    tok = Tokenizer(vocab, TokenizerConfig(token_buckets=tuple(int(x) for x in a.buckets.split(",")))
                    if a.buckets else TokenizerConfig())
    net = None
    if a.policy == "net":
        torch.manual_seed(0)
        net = SpireNet(tok.layout, PRESETS[a.preset], catalog).cuda()
    for envs in (int(x) for x in a.envs.split(",")):
        res = run_one(a, envs, vocab, catalog, tok, net)
        line = json.dumps(res)
        print(line, flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
