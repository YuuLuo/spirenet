"""Identity run of py-stage-m against its source trees (notes/solo/removal-spec.md section 8); PYTHONPATH = the tree.

FEATURES = comma list of forecast / taste / removal (empty = all off). A fixed-seed CPU actor (TorchPolicy, fake
hosts, production tokenizer flags cycle_ban + public_history + commit_select, exploration mixture) plays 2,400
decisions with the features on; one eager learner update follows. Prints one JSON line: hashes of every recorded
action / log-prob / value / kind / legal count (and taste terms) and of the updated parameters. Run the same
FEATURES in py-stage-h3 (none), py-stage-j (forecast), py-stage-k (taste) and py-stage-m: the lines must be equal."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

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

features = {f for f in os.environ["FEATURES"].split(",") if f}
assert features <= {"forecast", "taste", "removal"}, features
torch.manual_seed(0)
torch.set_num_threads(1)
vocab = Vocab(make_synthetic_vocab())
tmp = Path(tempfile.mkdtemp())
flags: dict = {"cycle_ban": True, "public_history": True, "commit_select": True}
reset_flags = int(RESET_FLAGS_DEFAULT)
if "forecast" in features:
    flags["forecast"] = True
    reset_flags |= 256
if "removal" in features:
    names = vocab["card"].entries[2:]
    chars = ("DEFECT", "FOOL", "IRONCLAD", "NECROBINDER", "REGENT", "SILENT")
    table = {"version": 1, "vocab_hash": vocab.hash, "game_version": vocab.game_version,
             "ratings": {n: 1 + i % 5 for i, n in enumerate(names)}, "curses": list(names[10:13]),
             "strike": {c: [names[100 + 2 * i]] for i, c in enumerate(chars)},
             "defend": {c: [names[101 + 2 * i]] for i, c in enumerate(chars)}, "dropped": []}
    (tmp / "removal.json").write_text(json.dumps(table), encoding="utf-8")
    flags.update(removal_rule=True, removal_ratings=str(tmp / "removal.json"))
tok = Tokenizer(vocab, TokenizerConfig(**flags))
net = SpireNet(tok.layout, ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32))
expl = ((int(Kind.CARD_REWARD), 0.1), (int(Kind.CARD_SELECT), 0.1), (int(Kind.COMBAT), 0.05))
ppo: dict = {}
actor_extra: dict = {}
store_extra: dict = {}
if "taste" in features:
    from spire.taste import Taste, TasteConfig
    names = vocab["card"].entries
    fams = {c: {"families": {"r0": {"cards": {names[i]: 1 + i % 3 for i in range(2, len(names)) if i % 3 == 0}},
                             "r1": {"cards": {names[i]: 2 for i in range(2, len(names)) if i % 3 == 1}}}}
            for c in ("IRONCLAD", "SILENT")}
    (tmp / "taste.json").write_text(json.dumps({"version": 1, "characters": fams}), encoding="utf-8")
    tcfg = TasteConfig(str(tmp / "taste.json"), 0.5, 0.7, p_thin=0.5, thin_deck=0)
    ppo["taste"] = tcfg
    actor_extra["taste"] = Taste(tcfg, vocab, (0, 1, 2, 3, 4))
    store_extra["taste"] = True
cfg = PPOConfig(minibatch=512, micro_batch=256, epochs=1, exploration_by_kind=expl, **ppo)
learner = Learner(net, tok, cfg, device="cpu", fast=False)
policy = TorchPolicy(learner.model, tok, seed=11, exploration_by_kind=expl)
segs: list = []
store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32), **store_extra)
hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
         for i in range(4)]
actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0), ("IRONCLAD", 1.0))),
                                                 vocab, 0), ActorConfig(reset_flags=reset_flags), store=store,
              **actor_extra)
actor.run(max_decisions=2400)
h = hashlib.sha256()
for s in segs:
    for a in (s.actions, s.logp, s.values, s.kinds, s.n_legal):
        h.update(np.ascontiguousarray(a).tobytes())
    if "taste" in features:
        h.update(np.ascontiguousarray(s.t_keep).tobytes())
        h.update(np.ascontiguousarray(s.t_add).tobytes())
m = learner.update(segs)
hp = hashlib.sha256()
for v in learner.model.state_dict().values():
    hp.update(v.detach().cpu().numpy().tobytes())
print(json.dumps({"features": sorted(features), "decisions": int(sum(len(s) for s in segs)), "steps": h.hexdigest(),
                  "params": hp.hexdigest(), "loss_pi": m["loss_pi"],
                  "taste_steps": int(sum(int((s.t_keep < 1).sum()) for s in segs)) if "taste" in features else None}))
