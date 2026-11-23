"""CPU-only H5 shape audit; tensor-byte estimates are not CUDA allocator measurements."""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from ctypes import wintypes
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .actor import group_ranges
from .h5eval import distribution
from .learner import PPOConfig
from .model import GraphRunner, ModelConfig, PRESETS, SpireNet, packed_layout
from .proto import frame_problems, parse_frame
from .schema import Kind, RESET_FLAGS_DEFAULT, ViewMode
from .synth import FrameGen, SizeMode
from .tokenize import BatchShape, Sample, Tokenizer, TokenizerConfig
from .trainstep import batch_cap, bucket_micros, pad_batch, round_dim
from .vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab

MIXED_BUCKETS = (32, 48, 64, 80, 112, 160, 192, 224, 256, 288, 320, 384, 448, 640)
RUN_BUCKETS = (20, 28, 40, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384)
LIVE_BUCKETS = (20, 28, 40, 64, 80, 96, 128, 192, 256, 384)


def windows_process_memory() -> dict[str, int]:
    """Measured bytes of this benchmark process, never of another process."""
    assert sys.platform == "win32"

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in ("peak_rss", "rss", "peak_paged", "paged",
                      "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return {"rss_bytes": int(counters.rss), "peak_rss_bytes": int(counters.peak_rss),
            "private_commit_bytes": int(counters.private)}


def workload(vocab: Vocab, count: int, seed: int) -> list[Sample]:
    assert count >= 4
    rng = np.random.default_rng(seed)
    gen, tok = FrameGen(vocab), Tokenizer(vocab)
    samples = []
    for index in range(count):
        n = 1 if index % 2 == 0 else 4
        size: SizeMode = "typical" if n == 1 else "h5"
        ctx = gen.random_ctx(rng, n, view=ViewMode.TEAM, size=size)
        ctx.reset_flags = int(RESET_FLAGS_DEFAULT)  # no optional map/deck in combat
        frame = parse_frame(gen.frame(rng, Kind.COMBAT, ctx, size=size, hashes=False))
        assert not frame_problems(frame), frame_problems(frame)
        samples.append(tok.encode(frame))
    return samples


def shape_report(samples: Sequence[Sample], tok: Tokenizer, *, max_batch: int = 32,
                 micro_batch: int = 256, micro_tokens: int = 32768,
                 model: ModelConfig = PRESETS["4m"]) -> dict[str, Any]:
    order = sorted(samples, key=lambda sample: tok.config.bucket(sample.n_tokens))
    buckets = np.array([tok.config.bucket(s.n_tokens) for s in order], np.int64)
    graphs: dict[tuple[int, int], BatchShape] = {}
    groups = group_ranges(buckets, min_group=1, max_batch=max_batch)
    padded_tokens = padded_actions = 0
    max_input = max_attention = 0
    for lo, hi in groups:
        shape = GraphRunner.pad_shape(tok.shape_of(order[lo:hi]))
        key = (shape.batch, shape.tokens)
        if key in graphs:
            shape = BatchShape(*(max(a, b) for a, b in zip(graphs[key].astuple(), shape.astuple())))
        graphs[key] = shape
        padded_tokens += shape.batch * shape.tokens
        padded_actions += shape.batch * shape.actions
        max_input = max(max_input, packed_layout(shape, tok.width)[1])
        max_attention = max(max_attention, shape.batch * model.n_heads * shape.tokens**2 * 2)
    cfg = PPOConfig(micro_batch=micro_batch, micro_tokens=micro_tokens)
    micros = bucket_micros(np.arange(len(samples)), buckets, cfg)
    micro_rows = []
    for bucket, indices in micros:
        count = len(indices)
        padded = pad_batch(count, batch_cap(cfg, bucket))
        assert count <= batch_cap(cfg, bucket)
        micro_rows.append({"tokens": bucket, "samples": count, "padded_samples": padded,
                           "padded_tokens": padded * bucket})
    largest = max(samples, key=lambda sample: sample.n_tokens)
    bucket = tok.config.bucket(largest.n_tokens)
    capacity: dict[str, Any] = {}
    for name, count in (("actor_full", max_batch), ("learner_full", batch_cap(cfg, bucket))):
        natural = tok.shape_of([largest] * count)
        shape = (GraphRunner.pad_shape(natural) if name == "actor_full" else BatchShape(
            pad_batch(count, count), natural.tokens, round_dim(natural.actions, 8),
            round_dim(natural.kids, 64), round_dim(natural.vars, 64), round_dim(natural.edges, 64),
            round_dim(natural.rows, 256)))
        capacity[name] = {"samples": count, "padded_samples": shape.batch,
                          "tokens": shape.tokens, "actions": shape.actions,
                          "packed_input_bytes": packed_layout(shape, tok.width)[1],
                          "one_attention_scores_bf16_bytes": shape.batch * model.n_heads * shape.tokens**2 * 2,
                          "hidden_plus_ff_per_layer_bf16_bytes": shape.batch * shape.tokens * (model.d_model + model.d_ff) * 2}
    n_tokens = sum(s.n_tokens for s in samples)
    n_actions = sum(s.n_actions for s in samples)
    return {"buckets": list(tok.config.token_buckets), "frames": len(samples), "max_batch": max_batch,
            "natural_tokens": n_tokens, "natural_actions": n_actions,
            "token_padding_overhead": padded_tokens / n_tokens - 1,
            "action_padding_overhead": padded_actions / n_actions - 1,
            "graph_keys": len(graphs), "graph_cache_limit": 64,
            "graph_shapes": [asdict(shape) for shape in graphs.values()],
            "max_graph_input_bytes_exact": max_input,
            "sum_graph_input_bytes_exact": sum(packed_layout(s, tok.width)[1] for s in graphs.values()),
            "max_one_attention_scores_bf16_bytes_estimate": max_attention,
            "learner_micro_batch": micro_batch, "learner_micro_tokens": micro_tokens,
            "full_batches_estimate": capacity,
            "learner_micros": micro_rows,
            "learner_full_bucket_caps": {str(t): batch_cap(cfg, int(t)) for t in np.unique(buckets)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=256)
    ap.add_argument("--micro-tokens", type=int, default=32768)
    ap.add_argument("--preset", choices=tuple(PRESETS), default="4m")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    memory_before = windows_process_memory() if sys.platform == "win32" else None
    vocab = Vocab(make_synthetic_vocab())
    samples = workload(vocab, a.frames, a.seed)
    model = replace(PRESETS[a.preset], seat_symmetry=True)
    report: dict[str, Any] = {"device": "cpu", "cuda_executed": False, "model": asdict(model),
                              "workload": "50% typical 1-seat combat, 50% busy synthetic H5 combat",
                              "frame_sizes": {str(n): {
                                  "tokens": distribution([s.n_tokens for s in samples if s.cond[1] == n]),
                                  "actions": distribution([s.n_actions for s in samples if s.cond[1] == n]),
                                  "stored_sample_bytes": distribution([s.nbytes for s in samples if s.cond[1] == n])}
                                  for n in (1, 4)}}
    for name, config in (("current", TokenizerConfig()),
                         ("live", TokenizerConfig(token_buckets=LIVE_BUCKETS)),
                         ("candidate", TokenizerConfig(token_buckets=MIXED_BUCKETS)),
                         ("run_config", TokenizerConfig(token_buckets=RUN_BUCKETS))):
        report[name] = shape_report(samples, Tokenizer(vocab, config), max_batch=a.max_batch,
                                    micro_batch=a.micro_batch, micro_tokens=a.micro_tokens, model=model)
    tok = Tokenizer(vocab)
    catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
    net = SpireNet(tok.layout, model, catalog).cpu()
    largest = max(samples, key=lambda sample: sample.n_tokens)
    batch = tok.collate([largest])
    t0 = time.perf_counter()
    out = net(batch)
    assert torch.isfinite(out["logits"][batch["act_mask"]]).all()
    (out["value"].sum() + out["logits"][batch["act_mask"]].mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    report["cpu_forward_backward"] = {"batch": 1, "tokens": largest.n_tokens,
                                       "actions": largest.n_actions, "seconds": time.perf_counter() - t0,
                                       "parameters": net.n_params,
                                       "fp32_parameter_gradient_adam_ema_bytes_estimate": net.n_params * 20}
    report["measured_cpu_process_memory"] = {
        "before": memory_before,
        "after": windows_process_memory() if sys.platform == "win32" else None}
    report["limits"] = ("Graph shapes and packed tensor bytes calculated on CPU; no CUDA graph capture, "
                         "allocator peak, throughput or OOM threshold measured. Attention estimate is one "
                         "dense B*heads*T*T bf16 tensor, not FlashAttention allocation. Learner padding can "
                         "round a microbatch above the nominal token cap; inspect padded_tokens.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
