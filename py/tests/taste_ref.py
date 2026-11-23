"""Reference run for tests/test_taste.py::test_off_is_the_reference_tree (PYTHONPATH = the tree under test).

A fixed-seed CPU actor (TorchPolicy, fake hosts, exploration mixture, habit advice) plays 2,400 decisions; one eager
learner update follows. Prints one JSON line: hashes of every recorded action / log-prob / value and of the updated
parameters, so two trees can be compared bit for bit. TASTE_P0=1 (taste tree only): ppo.taste with p_episode 0."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

import spire.learner as L
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore
from spire.schema import Kind
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.vocab import Vocab, make_synthetic_vocab

torch.manual_seed(0)
torch.set_num_threads(1)
vocab = Vocab(make_synthetic_vocab())
tok = Tokenizer(vocab)
net = SpireNet(tok.layout, ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32))
expl = ((int(Kind.CARD_REWARD), 0.1), (int(Kind.COMBAT), 0.05))
extra = {}
p0 = bool(os.environ.get("TASTE_P0"))
if p0:
    from spire.taste import TasteConfig
    extra["taste"] = TasteConfig("unused.json", 0.0, 0.7)
cfg = PPOConfig(minibatch=512, micro_batch=256, epochs=1, decoupled_clip_anchor=True, exploration_by_kind=expl,
                **extra)
learner = Learner(net, tok, cfg, device="cpu", fast=False)
policy = TorchPolicy(learner.model, tok, seed=11, exploration_by_kind=expl)
segs = []
store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
         for i in range(4)]
actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0), ("IRONCLAD+SILENT", 1.0))),
                                                 vocab, 0), ActorConfig(), store=store)
actor.run(max_decisions=2400)
h = hashlib.sha256()
for s in segs:
    for a in (s.actions, s.logp, s.values, s.kinds):
        h.update(np.ascontiguousarray(a).tobytes())
m = learner.update(segs)
hp = hashlib.sha256()
for v in learner.model.state_dict().values():
    hp.update(v.detach().cpu().numpy().tobytes())
print(json.dumps({"decisions": int(sum(len(s) for s in segs)), "steps": h.hexdigest(), "params": hp.hexdigest(),
                  "loss_pi": m["loss_pi"], "kl": m["behavior_kl_before_update"],
                  "taste_module": hasattr(L, "TasteConfig"), "p0": p0}))
