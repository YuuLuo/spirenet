"""Overhead benchmark of the PPO core: rewards, rollout bookkeeping, GAE, learner update.

    ..\\venv\\Scripts\\python.exe -B scripts\\bench_ppo.py [--device cuda] [--samples 65536]

The learner part uses a feature-vector MLP in place of SpireNet, so it measures the update
machinery (indexing, collate, micro-batching, AMP, AdamW, EMA, diagnostics) plus a known
model cost, not the real transformer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spire.learner import Learner, PPOConfig  # noqa: E402
from spire.proto import parse_frame  # noqa: E402
from spire.rewards import RoomIds, episode_targets, frame_facts  # noqa: E402
from spire.rollout import (  # noqa: E402
    EpisodeMeta, GAEConfig, RolloutConfig, RolloutStore, Segment, SegmentEnd, build_batch, gae,
)
from spire.schema import Kind  # noqa: E402
from spire.synth import FrameGen, random_frame  # noqa: E402
from spire.vocab import Vocab, make_synthetic_vocab  # noqa: E402


def best_of(fn: Callable[[], Any], reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


class VecTok:
    def __init__(self, a_max: int) -> None:
        self.a_max = a_max

    def collate(self, samples: Sequence[tuple[np.ndarray, int]]) -> dict[str, Tensor]:
        return {"x": torch.from_numpy(np.stack([s[0] for s in samples])),
                "n": torch.tensor([s[1] for s in samples])}


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int, a_max: int, layers: int) -> None:
        super().__init__()
        mods: list[nn.Module] = [nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.GELU()]
        self.trunk = nn.Sequential(*mods)
        self.out = nn.Linear(hidden, a_max + 4)
        self.a_max = a_max

    def forward(self, b: dict[str, Tensor]) -> dict[str, Tensor]:
        y = self.out(self.trunk(b["x"]))
        legal = torch.arange(self.a_max, device=y.device)[None] < b["n"][:, None]
        v, pw, fl, hp = y[:, self.a_max:].unbind(-1)
        return {"logits": y[:, :self.a_max].masked_fill(~legal, float("-inf")), "value": v, "pwin_logit": pw,
                "floors_pred": fl, "hp_pred": hp}


def synthetic_segments(n_samples: int, seg_len: int, d_in: int, a_max: int, rng: np.random.Generator) -> list[Segment]:
    segs = []
    for i in range(n_samples // seg_len):
        n_legal = rng.integers(1, a_max + 1, seg_len)
        s = Segment(EpisodeMeta(i % 40, i, (2,), 0), 0,
                    [(rng.normal(size=d_in).astype(np.float32), int(k)) for k in n_legal],
                    (rng.random(seg_len) * n_legal).astype(np.int64), np.full(seg_len, -1.0, np.float32),
                    rng.normal(size=seg_len).astype(np.float32), rng.integers(1, 13, seg_len).astype(np.int8),
                    n_legal.astype(np.int16), np.zeros(seg_len, np.int64), end=SegmentEnd(i % 3),
                    bootstrap_value=0.5)
        s.rewards = (rng.normal(size=seg_len) * 0.01).astype(np.float32)
        cid = np.where(rng.random(seg_len) < 0.7, i, -1).astype(np.int32)
        s.combat_id = cid
        s.win, s.win_mask = np.ones(seg_len, np.float32), np.ones(seg_len, bool)
        s.floors_left, s.hp_post, s.hp_mask = np.full(seg_len, 10.0, np.float32), np.full(seg_len, 0.5, np.float32), cid >= 0
        segs.append(s)
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples", type=int, default=65536)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    vocab = Vocab(make_synthetic_vocab())
    rooms = RoomIds.from_vocab(vocab)
    gen = FrameGen(vocab)

    frames = [parse_frame(random_frame(gen, rng, Kind.COMBAT, hashes=False)) for _ in range(200)]
    t = best_of(lambda: [frame_facts(f) for f in frames], 20) / len(frames)
    print(f"frame_facts (1-seat COMBAT typical): {t * 1e6:.1f} us/frame")
    facts = [frame_facts(f) for f in frames]

    sink: list[Segment] = []
    store = RolloutStore(sink.append, rooms, config=RolloutConfig(segment_len=256))
    n_steps = 20000

    def drive() -> None:
        for env in range(40):
            store.begin_episode(env, (2,))
        for i in range(n_steps):
            store.add_step(i % 40, facts[i % len(facts)], None, 0, -1.0, 0.0, 0)
        for env in range(40):
            store.end_episode(env, facts[0]._replace(msg_type=2, result=1))

    t = best_of(drive, 5) / n_steps
    print(f"RolloutStore.add_step + release (40 envs, 500-step episodes): {t * 1e6:.2f} us/step")
    ep = [facts[i % len(facts)] for i in range(1000)] + [facts[0]._replace(msg_type=2, result=1)]
    t = best_of(lambda: episode_targets(ep, rooms), 20)
    print(f"episode_targets (1000-decision episode): {t * 1e3:.2f} ms")

    d_in, a_max = 64, 32
    segs = synthetic_segments(args.samples, 256, d_in, a_max, rng)
    for mm in (False, True):
        t = best_of(lambda: gae(segs, GAEConfig(macro_micro=mm)), 5)
        print(f"gae ({len(segs)} x 256 = {args.samples} steps, macro_micro={mm}): {t * 1e3:.1f} ms")
    t = best_of(lambda: build_batch(segs, GAEConfig(), 1), 5)
    print(f"build_batch ({args.samples} steps): {t * 1e3:.1f} ms")

    torch.manual_seed(0)
    model = MLP(d_in, args.hidden, a_max, args.layers)
    n_params = sum(p.numel() for p in model.parameters())
    learner = Learner(model, VecTok(a_max), PPOConfig(micro_batch=4096), device=args.device,
                      size_fn=lambda s: s[1])
    learner.update(segs[:32])  # warm-up (CUDA init, kernels)
    for _ in range(2):
        m = learner.update(segs)
        print(f"learner.update on {args.device} (MLP {n_params / 1e6:.1f}M params, {args.samples} samples, "
              f"2 epochs x minibatch 4096, bf16={learner.autocast}): prep {m['time_prep'] * 1e3:.0f} ms, "
              f"train {m['time_train']:.2f} s, {m['samples_per_s']:.0f} sample-epochs/s")


if __name__ == "__main__":
    main()
