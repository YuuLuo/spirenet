"""Identity run of the temperature tree against its base (notes/next/temperature-spec.md 4.1); PYTHONPATH = the tree.

tests/identity_ref.py with FEATURES empty, except that here (in the temperature tree) the store records the beh_temp
column and the policy carries ppo.combat_temperature = 1.0. Prints the same JSON line as identity_ref.py (plus
"beh_temp_ones"); run identity_ref.py (FEATURES="") in the base tree: the common keys must be equal."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore
from spire.schema import RESET_FLAGS_DEFAULT, Kind
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab, make_synthetic_vocab

torch.manual_seed(0)
torch.set_num_threads(1)
vocab = Vocab(make_synthetic_vocab())
tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, public_history=True, commit_select=True))
net = SpireNet(tok.layout, ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32))
expl = ((int(Kind.CARD_REWARD), 0.1), (int(Kind.CARD_SELECT), 0.1), (int(Kind.COMBAT), 0.05))
cfg = PPOConfig(minibatch=512, micro_batch=256, epochs=1, exploration_by_kind=expl, combat_temperature=1.0)
learner = Learner(net, tok, cfg, device="cpu", fast=False)
policy = TorchPolicy(learner.model, tok, seed=11, exploration_by_kind=expl, combat_temperature=cfg.combat_temperature)
segs: list = []
store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32), beh_temp=True)
hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
         for i in range(4)]
actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0), ("IRONCLAD", 1.0))),
                                                 vocab, 0), ActorConfig(reset_flags=int(RESET_FLAGS_DEFAULT)), store=store)
actor.run(max_decisions=2400)
h = hashlib.sha256()
for s in segs:
    for a in (s.actions, s.logp, s.values, s.kinds, s.n_legal):
        h.update(np.ascontiguousarray(a).tobytes())
ones = bool(all((s.beh_temp == 1).all() for s in segs))
m = learner.update(segs)
hp = hashlib.sha256()
for v in learner.model.state_dict().values():
    hp.update(v.detach().cpu().numpy().tobytes())
print(json.dumps({"features": [], "decisions": int(sum(len(s) for s in segs)), "steps": h.hexdigest(),
                  "params": hp.hexdigest(), "loss_pi": m["loss_pi"], "taste_steps": None, "beh_temp_ones": ones,
                  "combat_steps": actor.counters.combat_steps}))
