"""Identity run of tree p2 against the feature trees (notes/next/p2.md B); PYTHONPATH = the tree. Not a test.

FEATURE = one of
  temp      tests/identity_ref.py's run (FEATURES empty) with ppo.combat_temperature = TEMP (default 0.5) on the policy
            and the learner, the store recording beh_temp; run in nt (alt/next/temp/py) and p2
  kick1     the same run, the learner kickstarted (phase 1) toward a fixed-seed teacher checkpoint; run in nk2 and p2
  kick0     phase 0: the teacher acts (a plain TorchPolicy of the teacher network = the phase-0 actor, test_kickstart
            v2.3) and the student distils (one epoch); run in nk2 and p2
  practice  test_practice.run_actor (fake fork servers, Linux) with practice (and SCOUTS=1: scouts) on, then one PPO
            update of its segments (macro_micro); run in np (alt/practice/py) and p2; only parent_* are deterministic
Prints one JSON line of hashes; the lines of the two trees must be equal. CPU, one thread."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import sys
import tempfile
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import GAEConfig, RolloutConfig, RolloutStore
from spire.schema import Kind
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab, make_synthetic_vocab

feature = os.environ["FEATURE"]
assert feature in ("temp", "kick1", "kick0", "practice"), feature
torch.manual_seed(0)
torch.set_num_threads(1)
vocab = Vocab(make_synthetic_vocab())
tmp = Path(tempfile.mkdtemp())
expl = ((int(Kind.CARD_REWARD), 0.1), (int(Kind.CARD_SELECT), 0.1), (int(Kind.COMBAT), 0.05))
SMALL = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32)
WIDE = ModelConfig(d_model=48, n_layers=2, n_heads=4, d_ff=96, d_id=16, d_cond=16, d_static=32, init_std=0.3)


def seg_hash(segs: list, extra: tuple[str, ...] = ()) -> str:
    h = hashlib.sha256()
    for s in segs:
        for name in ("actions", "logp", "values", "kinds", "n_legal", *extra):
            h.update(np.ascontiguousarray(getattr(s, name)).tobytes())
    return h.hexdigest()


def param_hash(m: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for v in m.state_dict().values():
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(policy: TorchPolicy, tok: Tokenizer, store_extra: dict) -> list:
    segs: list = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32), **store_extra)
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
             for i in range(4)]
    Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0), ("IRONCLAD", 1.0))), vocab, 0),
          ActorConfig(), store=store).run(max_decisions=2400)
    return segs


out: dict = {"feature": feature}
if feature in ("temp", "kick1", "kick0"):
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, public_history=True, commit_select=True))
    net = SpireNet(tok.layout, SMALL)
    if feature == "temp":
        T = float(os.environ.get("TEMP", "0.5"))
        cfg = PPOConfig(minibatch=512, micro_batch=256, epochs=1, exploration_by_kind=expl, combat_temperature=T)
        learner = Learner(net, tok, cfg, device="cpu", fast=False)
        policy = TorchPolicy(learner.model, tok, seed=11, exploration_by_kind=expl, combat_temperature=T)
        segs = run(policy, tok, {"beh_temp": True})
        out["steps"] = seg_hash(segs, ("beh_temp",))
        out["tempered_steps"] = int(sum(int((s.beh_temp != 1).sum()) for s in segs))
    else:
        from spire.kickstart import KickstartConfig, load_teacher
        torch.manual_seed(5)
        tnet = SpireNet(tok.layout, WIDE)
        saver = Learner(copy.deepcopy(tnet), tok, PPOConfig(), device="cpu")
        tk = json.loads(json.dumps(asdict(tok.config)))
        saver.save(tmp / "teacher.pt", extra={"model_config": asdict(WIDE), "tokenizer": tk, "vocab": "synthetic",
                                              "catalog": "none", "vocab_hash": tok.layout.vocab_hash,
                                              "catalog_hash": 0, "run": "teacher"})
        teacher, sha = load_teacher(tmp / "teacher.pt", tok, None, SMALL)
        phase = 0 if feature == "kick0" else 1
        ks = KickstartConfig(str(tmp / "teacher.pt"), 1.0, 0.5, 10, phase=phase)
        cfg = PPOConfig(minibatch=512, micro_batch=256, epochs=1, exploration_by_kind=expl, kickstart=ks)
        torch.manual_seed(0)
        learner = Learner(net, tok, cfg, device="cpu", fast=False, teacher=teacher, teacher_sha256=sha)
        acting = load_teacher(tmp / "teacher.pt", tok, None, SMALL)[0] if phase == 0 else learner.model
        policy = TorchPolicy(acting, tok, seed=11, exploration_by_kind=expl)
        segs = run(policy, tok, {})
        out["steps"] = seg_hash(segs)
    out["decisions"] = int(sum(len(s) for s in segs))
    m = learner.update(segs)
    out["params"] = param_hash(learner.model)
    out["loss_pi"] = m["loss_pi"]
    if "kickstart" in m:
        out["kickstart"] = {k: v for k, v in m["kickstart"].items() if not isinstance(v, dict)}
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_practice as tp
    from spire.gym import PracticeConfig, Priors
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True, cycle_ban=True, commit_select=True))
    scouts = os.environ.get("SCOUTS", "0") == "1"
    r = tp.run_actor(vocab, tok, tmp, practice=PracticeConfig(actors=1, scouts=scouts,
                                                              priors=Priors(0.25, 0.35, 0.55, 0.9)), episodes=4)
    assert not r.errs and r.total("fork_failures") == r.total("lost") == 0
    keep = lambda s: {f.name: getattr(s, f.name) for f in fields(s) if f.name not in ("t_first", "t_last")}  # noqa: E731
    segs = sorted(r.segs, key=lambda s: (s.meta.episode_uid, s.start))
    # the parent games are deterministic (practice T5); the children's batches depend on fork timing, so their
    # segments / records / the update differ from run to run in any tree (recorded, not compared)
    out["parent_hash"] = hashlib.sha256(pickle.dumps(r.parent(), protocol=4)).hexdigest()
    out["parent_records"] = hashlib.sha256(json.dumps([{k: v for k, v in x.as_dict().items() if k != "seconds"}
                                                       for x in r.recs if not x.gym], sort_keys=True).encode()).hexdigest()
    out["parent_decisions"] = len(r.parent())
    out["segments"] = len(segs)
    out["children"] = len(r.kids)
    out["seg_hash"] = hashlib.sha256(pickle.dumps([keep(s) for s in segs], protocol=4)).hexdigest()
    out["decisions_hash"] = hashlib.sha256(pickle.dumps(sorted(r.decisions), protocol=4)).hexdigest()
    out["records"] = hashlib.sha256(json.dumps(sorted(json.dumps({k: v for k, v in x.as_dict().items()
                                                                   if k != "seconds"}, sort_keys=True)
                                                       for x in r.recs)).encode()).hexdigest()
    torch.manual_seed(0)
    cfg = PPOConfig(minibatch=512, micro_batch=256, epochs=1, exploration_by_kind=tp.EXPLORATION,
                    gae=GAEConfig(gamma=1.0, lam=0.99, macro_micro=True, lam_macro=0.99, lam_micro=0.99,
                                  stitch_segments=True))
    learner = Learner(SpireNet(tok.layout, tp.TINY), tok, cfg, device="cpu", fast=False)
    m = learner.update(segs)
    out["params"] = param_hash(learner.model)
    out["loss_pi"] = m["loss_pi"]
print(json.dumps(out, sort_keys=True))
