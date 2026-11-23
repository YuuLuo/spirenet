"""Per-frame cost of the actor's CPU hot-path components, old vs new, by batch size and frame size.

Single thread, no hosts, no GPU: the same synthetic frames go through

* ``parse``   - ``proto.parse_frame`` (header + directory validation)
* ``facts``   - ``rewards.frame_facts``
* ``encode``  - frozen per-frame reference (``tokenize_ref``) vs the numpy batch encoder vs the C
  kernel (``spire._native``); ``+samples`` adds the per-frame ``Sample`` views the rollout store needs
* ``collate`` - ``Tokenizer.collate`` of reference samples (+ the 10 per-tensor copies into pinned
  staging the old policy made) vs ``collate_encoded_into`` straight into packed staging (numpy / C)

Frame sets: ``train-mix`` (single-player decision mix of schema 11: 72% COMBAT ...), ``combat-4p``
(4-seat combat), ``extreme`` (every kind at the generator's near-limit sizes: a stand-in for large
late-game frames). Best of ``--reps`` timings. Results: one JSON line per (set, B) in ``--out``.

    python -B scripts/bench_hotpath.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from spire.model import packed_layout, packed_views  # noqa: E402
from spire.proto import Frame, parse_frame  # noqa: E402
from spire.rewards import frame_facts  # noqa: E402
from spire.schema import DECISION_KINDS, Kind  # noqa: E402
from spire.synth import FrameGen, random_frame  # noqa: E402
from spire.tokenize import BatchShape, Tokenizer  # noqa: E402
from spire.tokenize_ref import encode_reference  # noqa: E402
from spire.vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MIX = [Kind.COMBAT] * 72 + [Kind.CARD_SELECT] * 6 + [Kind.MAP] * 6 + [Kind.CARD_REWARD] * 4 + [Kind.REWARD] * 4 + \
      [Kind.EVENT] * 3 + [Kind.SHOP] * 2 + [Kind.REST] * 3


def frame_set(gen: FrameGen, name: str, n: int, seed: int = 0) -> list[bytes]:
    rng = np.random.default_rng(seed)
    if name == "train-mix":
        return [random_frame(gen, rng, MIX[i % len(MIX)], n_seats=1, hashes=False) for i in range(n)]
    if name == "combat-4p":
        return [random_frame(gen, rng, Kind.COMBAT, n_seats=4, hashes=False) for _ in range(n)]
    kinds = list(DECISION_KINDS)
    return [random_frame(gen, rng, kinds[i % len(kinds)], n_seats=1 + i % 4, size="extreme", hashes=False)
            for i in range(n)]


def best(fn: Callable[[], Any], reps: int) -> float:
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def pad(s: BatchShape) -> BatchShape:  # the CUDA-graph shape the policy would use (GraphRunner.pad_shape)
    p2 = lambda n, lo: max(lo, 1 << max(0, n - 1).bit_length())  # noqa: E731
    return BatchShape(p2(s.batch, 8), s.tokens, p2(s.actions, 8), p2(s.kids, 64), p2(s.vars, 64), p2(s.edges, 64),
                      p2(s.rows, 256))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="train-mix,combat-4p,extreme")
    ap.add_argument("--batches", default="1,8,32,64,128")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--out", default=str(REPO / "bench" / "hotpath_bench.jsonl"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    vocab = Vocab(make_synthetic_vocab())
    gen = FrameGen(vocab, Catalog(make_synthetic_catalog(vocab, seed=1), vocab).hash)
    tok_c, tok_np = Tokenizer(vocab), Tokenizer(vocab, native=False)
    out = []
    for name in a.sets.split(","):
        payloads = frame_set(gen, name, 256)
        frames = [parse_frame(p) for p in payloads]
        refs = [encode_reference(tok_c, f) for f in frames]
        rows = float(np.mean([sum(s.counts) for s in refs]))
        kb = float(np.mean([len(p) for p in payloads])) / 1024
        for B in (int(x) for x in a.batches.split(",")):
            ps, fs, rs = payloads[:B], frames[:B], refs[:B]
            r: dict[str, Any] = {"set": name, "B": B, "rows_per_frame": round(rows, 1), "frame_kib": round(kb, 2),
                                 "native": tok_c.native, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            us = lambda t: round(t / B * 1e6, 2)  # noqa: E731
            r["parse"] = us(best(lambda: [parse_frame(p) for p in ps], a.reps))
            r["facts"] = us(best(lambda: [frame_facts(parse_frame(p)) for p in ps], a.reps)) - r["parse"]
            r["encode_reference"] = us(best(lambda: [encode_reference(tok_c, f) for f in fs], max(5, a.reps // 4)))
            r["encode_numpy_batch"] = us(best(lambda: tok_np.encode_batch(fs, sort=True), a.reps))
            r["encode_native"] = us(best(lambda: tok_c.encode_batch(fs, sort=True), a.reps))
            r["encode_native_samples"] = us(best(lambda: tok_c.encode_batch(fs, sort=True).samples(), a.reps))
            # collation of the whole batch as one group, into the policy's (padded) graph shape
            shape = pad(tok_c.shape_of(rs))
            stage = {k: torch.empty_like(v, pin_memory=True) for k, v in tok_c.collate(rs, shape=shape).items()}

            def old_collate() -> None:
                for k, v in tok_c.collate(rs, shape=shape).items():
                    stage[k].copy_(v)

            r["collate_reference"] = us(best(old_collate, a.reps))
            enc = tok_c.encode_batch(fs)
            layout, size = packed_layout(shape, tok_c.width)
            buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            dst = {k: v.numpy() for k, v in packed_views(buf, layout).items()}
            r["collate_numpy"] = us(best(lambda: tok_np.collate_encoded_into(enc, 0, B, shape, dst), a.reps))
            r["collate_native"] = us(best(lambda: tok_c.collate_encoded_into(enc, 0, B, shape, dst), a.reps))
            r["old_total"] = round(r["encode_reference"] + r["collate_reference"], 2)
            r["new_total"] = round(r["encode_native_samples"] + r["collate_native"], 2)
            out.append(r)
            print(json.dumps(r), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
