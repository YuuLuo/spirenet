"""Off = identical on the CUDA paths (notes/solo/kickstart-spec.md 5.1); PYTHONPATH = the tree (py-stage-m or m +
kickstart). Fake-host segments of a fixed seed, then one fixed-seed PPO update of a small model through the eager CUDA
path, the captured path and the captured + compiled path (``PATHS`` comma list, default all three). Prints one JSON
line per path: sha256 of the updated parameters and of the EMA, and the loss diagnostics. Run it twice per tree
(device determinism) and once per tree: the lines must be equal."""
from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab, make_synthetic_vocab

import torch._dynamo as dynamo
dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
torch.manual_seed(0)
torch.set_num_threads(1)
vocab = Vocab(make_synthetic_vocab())
tok = Tokenizer(vocab, TokenizerConfig(token_buckets=(20, 28, 40, 64, 80, 96, 128, 192, 256, 384), cycle_ban=True))
segs: list[Segment] = []
store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=64))
hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False)) for i in range(12)]
Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0), ActorConfig(max_batch=32),
      store=store).run(max_decisions=3000)
cfg = PPOConfig(epochs=2, minibatch=512, micro_batch=256, micro_tokens=4096, lr=1e-4, seed=3, kl_lr_target=0.01)
base = SpireNet(tok.layout, ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32))


def sha(ts: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


for name in os.environ.get("PATHS", "fast,graphs,compiled").split(","):
    torch.manual_seed(1)
    lr = Learner(copy.deepcopy(base), tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                 fast=True, graphs=name != "fast", compile_model=name == "compiled")
    m = lr.update(segs)
    print(json.dumps({"path": name, "params": sha(list(lr.model.parameters())), "ema": sha(list(lr.ema.parameters())),
                      **{k: round(float(m[k]), 9) for k in ("loss_pi", "loss_v", "approx_kl", "entropy", "grad_norm")},
                      "keys": sorted(m)}, sort_keys=True), flush=True)
