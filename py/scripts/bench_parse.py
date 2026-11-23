"""Measure frame parse cost (us/frame) on synthetic frames.

    python -B scripts/bench_parse.py [--frames 400] [--reps 20]

Cases: typical single-seat COMBAT (map/deck excluded, the DECISIONS-H1 F.2 default), typical
single-seat COMBAT with map+deck, typical 4-seat TEAM COMBAT, extreme 4-seat COMBAT, typical MAP.
Columns: parse (directory checks + 16 views), parse+verify_hashes (pure-Python FNV-1a), and
frame_problems (the schema 14.5 deep check meant for ~1% sampling).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spire.proto import frame_problems, parse_frame  # noqa: E402
from spire.schema import RESET_FLAGS_DEFAULT, Kind, ResetFlag, ViewMode  # noqa: E402
from spire.synth import FrameGen  # noqa: E402
from spire.vocab import Vocab, make_synthetic_vocab  # noqa: E402


def frames(gen: FrameGen, n: int, kind: int, seats: int, size: str, flags: int, seed: int) -> list[bytes]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ctx = gen.random_ctx(rng, seats, view=ViewMode.TEAM, size=size)  # type: ignore[arg-type]
        ctx.reset_flags = flags
        out.append(gen.frame(rng, kind, ctx, size=size))  # type: ignore[arg-type]
    return out


def per_frame_us(fn, payloads: list[bytes], reps: int) -> float:  # type: ignore[no-untyped-def]
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        for p in payloads:
            fn(p)
        best = min(best, (time.perf_counter() - t) / len(payloads))
    return best * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    gen = FrameGen(Vocab(make_synthetic_vocab()))
    full = int(RESET_FLAGS_DEFAULT | ResetFlag.MAP_IN_COMBAT | ResetFlag.DECK_IN_COMBAT)
    cases = [
        ("1-seat COMBAT typical (no map/deck)", Kind.COMBAT, 1, "typical", int(RESET_FLAGS_DEFAULT)),
        ("1-seat COMBAT typical (+map+deck)", Kind.COMBAT, 1, "typical", full),
        ("4-seat COMBAT typical TEAM", Kind.COMBAT, 4, "typical", int(RESET_FLAGS_DEFAULT)),
        ("4-seat COMBAT extreme TEAM", Kind.COMBAT, 4, "extreme", int(RESET_FLAGS_DEFAULT)),
        ("1-seat MAP typical", Kind.MAP, 1, "typical", int(RESET_FLAGS_DEFAULT)),
    ]
    print(f"{'case':38s} {'KiB':>6s} {'parse':>8s} {'+hash':>9s} {'deepchk':>9s}   (us/frame, best of {a.reps})")
    for i, (name, kind, seats, size, flags) in enumerate(cases):
        n = a.frames if size == "typical" else max(40, a.frames // 8)
        ps = frames(gen, n, kind, seats, size, flags, seed=i)
        kib = np.mean([len(p) for p in ps]) / 1024
        t_parse = per_frame_us(parse_frame, ps, a.reps)
        t_hash = per_frame_us(lambda p: parse_frame(p, verify_hashes=True), ps, max(2, a.reps // 5))
        parsed = [parse_frame(p) for p in ps]
        t_deep = per_frame_us(frame_problems, parsed, max(2, a.reps // 5))  # type: ignore[arg-type]
        print(f"{name:38s} {kib:6.1f} {t_parse:8.1f} {t_hash:9.1f} {t_deep:9.1f}")


if __name__ == "__main__":
    main()
