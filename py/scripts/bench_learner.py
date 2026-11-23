"""Learner (PPO update) throughput on REAL rollout segments, standalone or next to other GPU work.

    python -B scripts/bench_guard.py --timeout 900 -- python -B scripts/bench_learner.py \
        --segments ../bench/py-stage2-2/segs_natural.bin --epochs 1 --reps 4

The segments come from ``bench_real_host.py --dump`` (released training segments of real-host play:
u64 length + :func:`spire.ipc.pack_segment` blob each). The learner is the production one
(:class:`spire.learner.Learner` + :class:`spire.model.SpireNet` with the real vocab / catalog of the
launch spec), so the numbers include collation, host->device copies, bf16 forward / backward, fused AdamW,
EMA and the diagnostics. Per update one JSON line: samples/s (samples x epochs / train seconds), decisions/s
the learner could absorb (= samples / train seconds), time in ``Tokenizer.collate`` (CPU), process CPU
cores, GPU utilisation / power (nvidia-smi once per second), peak CUDA memory.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from spire.host import real_host_spec  # noqa: E402
from spire.ipc import decode_segment  # noqa: E402
from spire.learner import Learner, PPOConfig  # noqa: E402
from spire.metrics import query_gpu  # noqa: E402
from spire.model import PRESETS, SpireNet  # noqa: E402
from spire.rollout import Segment  # noqa: E402
from spire.tokenize import Tokenizer, TokenizerConfig  # noqa: E402
from spire.vocab import Catalog, Vocab  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
_U64 = struct.Struct("<Q")
RECOMMENDED_BUCKETS = (20, 28, 40, 64, 80, 96, 128, 192, 256, 384)


def load_segments(path: str, max_samples: int) -> list[Segment]:
    data = Path(path).read_bytes()
    out: list[Segment] = []
    off = n = 0
    while off < len(data) and n < max_samples:
        (k,) = _U64.unpack_from(data, off)
        off += 8
        seg = decode_segment(bytes(data[off:off + k]))
        off += k
        out.append(seg)
        n += len(seg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--spec", default=str(REPO / "bench" / "py-stage2-1" / "real_host_frozen.toml"))
    ap.add_argument("--preset", default="2.5m")
    ap.add_argument("--iteration", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--micro-tokens", type=int, default=65536)
    ap.add_argument("--micro-batch", type=int, default=4096)
    ap.add_argument("--buckets", default="recommended", help="'recommended', 'default' or comma-separated edges")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fast", choices=("on", "off"), default="on", help="spire.trainstep fast path")
    ap.add_argument("--graphs", choices=("on", "off"), default="on", help="CUDA graphs in the fast path")
    ap.add_argument("--compile", default="", help="probe: torch.compile mode for the fast eager path's model "
                    "(needs Triton, e.g. the venv-compile-probe interpreter); implies --graphs off")
    ap.add_argument("--out", default=str(REPO / "bench" / "py-stage2-2" / "learner_bench.jsonl"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    spec = real_host_spec(a.spec)
    vocab = Vocab.load(spec.vocab)
    catalog = Catalog.load(spec.catalog, vocab)
    buckets = RECOMMENDED_BUCKETS if a.buckets == "recommended" else TokenizerConfig().token_buckets \
        if a.buckets == "default" else tuple(int(x) for x in a.buckets.split(","))
    tok = Tokenizer(vocab, TokenizerConfig(token_buckets=buckets))
    segs = load_segments(a.segments, a.iteration * 2)
    n_avail = sum(len(s) for s in segs)
    torch.manual_seed(0)
    net = SpireNet(tok.layout, PRESETS[a.preset], catalog)
    cfg = PPOConfig(epochs=a.epochs, minibatch=a.minibatch, micro_batch=a.micro_batch, micro_tokens=a.micro_tokens)
    learner = Learner(net, tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                      fast=a.fast == "on", graphs=a.graphs == "on")
    if a.compile:
        assert learner.stepper is not None and a.graphs == "off", "--compile needs --fast on --graphs off"
        learner.stepper.model = torch.compile(learner.model, mode=None if a.compile == "default" else a.compile,
                                              dynamic=True)  # type: ignore[assignment]
    collate_ns = [0]
    inner = tok.collate

    def timed_collate(samples: Any, **kw: Any) -> Any:
        t = time.perf_counter_ns()
        out = inner(samples, **kw)
        collate_ns[0] += time.perf_counter_ns() - t
        return out
    tok.collate = timed_collate  # type: ignore[method-assign]

    # iterations: consecutive windows of >= a.iteration samples (wrapping around the dump)
    def iteration(i: int) -> list[Segment]:
        out: list[Segment] = []
        got = 0
        j = (i * 97) % len(segs)
        while got < a.iteration:
            out.append(segs[j % len(segs)])
            got += len(segs[j % len(segs)])
            j += 1
        return out

    gpu: list[dict[str, float]] = []
    stop = threading.Event()
    measuring = threading.Event()

    def gpu_loop() -> None:
        while not stop.wait(1.0):
            if measuring.is_set():
                rows = query_gpu()
                if rows:
                    gpu.append(rows[0])
    th = threading.Thread(target=gpu_loop, daemon=True)
    th.start()
    ntok = np.array([s.n_tokens for seg in segs for s in seg.samples])
    padded = np.array([tok.config.bucket(int(t)) for t in ntok])
    for r in range(a.warmup + a.reps):
        segs_i = iteration(r)
        if r == a.warmup:
            measuring.set()
            torch.cuda.reset_peak_memory_stats()
        collate_ns[0] = 0
        c0, w0 = time.process_time(), time.perf_counter()
        m = learner.update(segs_i)
        c1, w1 = time.process_time(), time.perf_counter()
        if r < a.warmup:
            continue
        rec = {
            "tag": a.tag, "segments_file": Path(a.segments).name, "preset": a.preset, "epochs": a.epochs,
            "iteration": m["samples"], "minibatch": a.minibatch, "micro_tokens": a.micro_tokens,
            "micro_batch": a.micro_batch, "buckets": list(buckets), "fast": a.fast, "graphs": a.graphs, "compile": a.compile,
            "stepper": m.get("stepper"),
            "samples_per_s": round(m["samples_per_s"], 0),
            "decisions_per_s": round(m["samples"] / max(m["time_train"], 1e-9), 0),
            "time_train_s": round(m["time_train"], 3), "time_prep_s": round(m["time_prep"], 3),
            "update_wall_s": round(w1 - w0, 3), "collate_s": round(collate_ns[0] / 1e9, 3),
            "cpu_cores": round((c1 - c0) / max(w1 - w0, 1e-9), 2), "micro_batches": m.get("micro_batches"),
            "tokens_mean": round(float(ntok.mean()), 1), "padded_over_real": round(float(padded.sum() / ntok.sum()), 3),
            "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 0),
            "gpu_util": round(float(np.mean([g["util"] for g in gpu])), 1) if gpu else None,
            "gpu_power_w": round(float(np.mean([g["power_w"] for g in gpu])), 0) if gpu else None,
            "loss_pi": m["loss_pi"], "nonfinite_skipped": m["nonfinite_skipped"], "samples_available": n_avail,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        gpu.clear()
        line = json.dumps(rec)
        print(line, flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    stop.set()


if __name__ == "__main__":
    main()
