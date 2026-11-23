"""Helper of tests/test_advice.py (not a test): one CPU PPO update of fixed synthetic segments that carry an old-format
advice record, with the ``spire`` on PYTHONPATH and the advice rules of argv[1] (JSON, old format or []); prints its
metrics and a hash of the parameters as one JSON line. Uses only APIs of the advice commit (b89108a), so that tree and
a later one run the same code path."""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import torch

from ppo_stubs import StubNet, StubSample, StubTokenizer
import spire
from spire.learner import Learner, PPOConfig
from spire.rollout import EpisodeMeta, Segment, SegmentEnd
from spire.schema import Kind


def segments(seed: int = 0, n_segments: int = 12, length: int = 40) -> list[Segment]:
    """REST / COMBAT / EVENT steps; at REST steps a candidate row (70%) with HP fractions on the bucket edges."""
    rng = np.random.default_rng(seed)
    tok = StubTokenizer()
    out = []
    for i in range(n_segments):
        kinds = rng.choice(np.array([Kind.REST, Kind.COMBAT, Kind.EVENT], np.int8), length)
        n_legal = rng.integers(1, 5, length).astype(np.int16)
        actions = (rng.random(length) * n_legal).astype(np.int64)
        row = np.where((kinds == Kind.REST) & (rng.random(length) < 0.7), (rng.random(length) * n_legal), -1)
        row = row.astype(np.int16)
        pick = (row >= 0) & (rng.random(length) < 0.4)
        actions[pick] = row[pick]
        hp = np.where(row >= 0, rng.choice(np.array([0.2, 0.4999, 0.5, 0.7, 0.75, 0.9, 1.0], np.float32), length), 0)
        x = rng.standard_normal((length, tok.n_features)).astype(np.float32)
        out.append(Segment(
            EpisodeMeta(i % 3, i, (2, 2, 2, 2), 9), 0, [StubSample(x[t], int(n_legal[t])) for t in range(length)],
            actions, np.log(1.0 / n_legal).astype(np.float32), rng.standard_normal(length).astype(np.float32) * 0.1,
            kinds, n_legal, np.zeros(length, np.int64), end=SegmentEnd.TERMINAL if i % 2 else SegmentEnd.CUT,
            bootstrap_value=0.3, rewards=(rng.random(length) * 0.02).astype(np.float32),
            combat_id=np.where(kinds == Kind.COMBAT, i, -1).astype(np.int32), win=np.full(length, i % 2, np.float32),
            win_mask=np.ones(length, bool), floors_left=np.full(length, 10.0, np.float32),
            hp_post=np.full(length, 0.5, np.float32), hp_mask=kinds == Kind.COMBAT,
            advice_row=row, advice_hp=hp.astype(np.float32)))
    return out


def main() -> None:
    advice = tuple(tuple(r) for r in json.loads(sys.argv[1]))
    torch.manual_seed(5)
    tok = StubTokenizer()
    cfg = PPOConfig(epochs=2, minibatch=128, micro_batch=64, lr=3e-3, ent_coef=0.01, advice=advice, seed=0)
    learner = Learner(StubNet(tok.n_features), tok, cfg, device="cpu", size_fn=StubTokenizer.size)
    segs = segments()
    if not advice:  # the plain path: segments without a record
        for s in segs:
            s.advice_row = s.advice_hp = None
    m = learner.update(segs)
    h = hashlib.sha256()
    for k, v in learner.model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    metrics = {k: v for k, v in m.items() if not k.startswith("time_") and k not in ("samples_per_s", "lag_s")}
    print(json.dumps({"tree": spire.__file__, "samples": m["samples"],
                      "metrics": json.loads(json.dumps(metrics, default=str)), "params": h.hexdigest()}))


if __name__ == "__main__":
    main()
