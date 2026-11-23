"""Helper of tests/test_advice.py (not a test): one CPU PPO update of a fixed synthetic batch with the ``spire`` on
PYTHONPATH and no advice rule; prints its metrics and a hash of the parameters as one JSON line. Uses only APIs that
exist before the advice patch, so the reference tree and the patched tree run the same code path."""
from __future__ import annotations

import hashlib
import importlib.util
import json

import torch

from ppo_stubs import CorridorEnv, StubNet, StubTokenizer, collect
import spire
from spire.learner import Learner, PPOConfig
from spire.rewards import RewardConfig, RoomIds
from spire.rollout import GAEConfig, RolloutConfig, RolloutStore, SegmentRing
from spire.vocab import Vocab, make_synthetic_vocab


def main() -> None:
    torch.manual_seed(5)
    vocab = Vocab(make_synthetic_vocab())
    tok = StubTokenizer()
    cfg = PPOConfig(epochs=2, minibatch=128, micro_batch=64, lr=3e-3, ent_coef=0.01, gae=GAEConfig(gamma=0.99, lam=0.95),
                    seed=0)
    learner = Learner(StubNet(tok.n_features), tok, cfg, device="cpu", size_fn=StubTokenizer.size)
    ring = SegmentRing()
    store = RolloutStore(ring.put, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16),
                         rewards=RewardConfig(gamma=0.99))
    envs = [CorridorEnv(vocab, 100 + i) for i in range(16)]
    frames: list = [None] * len(envs)
    gen = torch.Generator().manual_seed(7)
    while ring.n_samples < 512:
        collect(envs, frames, tok, learner.model, store, 0, 8, gen)
    segs = ring.take(512)
    assert segs is not None
    m = learner.update(segs)
    h = hashlib.sha256()
    for k, v in learner.model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    metrics = {k: v for k, v in m.items() if not k.startswith("time_") and k not in ("samples_per_s", "lag_s")}
    has_advice = importlib.util.find_spec("spire.advice") is not None
    print(json.dumps({"tree": spire.__file__, "advice_module": has_advice, "samples": m["samples"],
                      "metrics": json.loads(json.dumps(metrics, default=str)), "params": h.hexdigest()}))


if __name__ == "__main__":
    main()
