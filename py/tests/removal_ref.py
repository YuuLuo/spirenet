"""Reference run for tests/test_removal.py::test_off_is_the_reference_tree (PYTHONPATH = the tree under test).

A fixed-seed CPU actor (TorchPolicy, production tokenizer flags cycle_ban + public_history + commit_select, fake
hosts with shop / event removal prompts, exploration mixture) plays 2,400 decisions; one eager learner update
follows. Prints one JSON line: hashes of every recorded action / log-prob / value / legal count and of the updated
parameters. REMOVAL_MODE: base (a tree without the rule), off (rule keys left at their defaults), on (rule on,
REMOVAL_TABLE = a table for the synthetic vocabulary)."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore
from spire.schema import CardSelectSub, G, Kind
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab, make_synthetic_vocab

mode = os.environ["REMOVAL_MODE"]
assert mode in ("base", "off", "on")
torch.manual_seed(0)
torch.set_num_threads(1)
vocab = Vocab(make_synthetic_vocab())
flags = {"cycle_ban": True, "public_history": True, "commit_select": True}
if mode == "on":
    flags.update(removal_rule=True, removal_ratings=os.environ["REMOVAL_TABLE"])
tok = Tokenizer(vocab, TokenizerConfig(**flags))
net = SpireNet(tok.layout, ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32))
expl = ((int(Kind.CARD_REWARD), 0.1), (int(Kind.CARD_SELECT), 0.1), (int(Kind.COMBAT), 0.05))
cfg = PPOConfig(minibatch=512, micro_batch=256, epochs=1, decoupled_clip_anchor=True, exploration_by_kind=expl)
learner = Learner(net, tok, cfg, device="cpu", fast=False)
policy = TorchPolicy(learner.model, tok, seed=11, exploration_by_kind=expl)
segs = []
store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
         for i in range(4)]
removal = []


def on_decision(env: int, f, a: int) -> None:
    g = f.gvals
    if g[G.KIND] == Kind.CARD_SELECT and g[G.KIND_SUB] == CardSelectSub.DECK_REMOVAL and not g[G.IN_COMBAT]:
        removal.append(f.n_actions)


actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0), ("IRONCLAD", 1.0))),
                                                 vocab, 0), ActorConfig(), store=store, on_decision=on_decision)
actor.run(max_decisions=2400)
h = hashlib.sha256()
one = []
for s in segs:
    for a in (s.actions, s.logp, s.values, s.kinds, s.n_legal):
        h.update(np.ascontiguousarray(a).tobytes())
    for i in range(len(s)):
        if s.samples[i].history[-6] < 0:
            one.append(int(s.n_legal[i]) == 1 and abs(float(s.logp[i])) < 1e-6)
m = learner.update(segs)
hp = hashlib.sha256()
for v in learner.model.state_dict().values():
    hp.update(v.detach().cpu().numpy().tobytes())
print(json.dumps({"mode": mode, "decisions": int(sum(len(s) for s in segs)), "steps": h.hexdigest(),
                  "params": hp.hexdigest(), "loss_pi": m["loss_pi"], "kl": m["behavior_kl_before_update"],
                  "removal_frames": len(removal), "rule_steps": len(one),
                  "rule_one_row": bool(one) and all(one) if mode == "on" else None}))
