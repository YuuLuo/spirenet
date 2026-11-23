"""Helper of tests/test_practice.py T3 (not a test): a deterministic train-mode actor run with the ``spire`` on
PYTHONPATH - in-process fake hosts (ERROR frames and aborts included), the tiny model on CPU with an exploration
mixture, cycle ban + commit_select + the removal rule, public history, an advice rule - and one PPO update of its
segments (macro_micro + stitched GAE, the solo configuration); prints hashes of the segments, the episode records,
the actor counters and the updated parameters as one JSON line.

    python -B tests/practice_off_ref.py none|off <removal-ratings.json>

``none``: the Actor without a ``practice`` argument (the only call tree n has); ``off``:
``practice=PracticeConfig()`` (actors = 0). Uses only APIs of tree n otherwise, so both trees run the same path."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import fields

import torch

import spire
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import GAEConfig, RolloutConfig, RolloutStore, Segment
from spire.schema import Kind
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab, make_synthetic_vocab

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
ADVICE = (("REST", "SMITH", 0.5, 0.05),)
EXPLORATION = ((int(Kind.COMBAT), 0.05), (int(Kind.MAP), 0.1))


TEMP_COUNTERS = ("combat_steps", "combat_nonargmax", "noncombat_steps", "noncombat_nonargmax")
TEMP_METRICS = ("beh_temp_mean", "entropy_beh_combat", "entropy_base_combat", "combat_policy_samples")
R07_COUNTERS = ("censored_win_steps", "censored_loss_steps", "win_steps", "loss_steps")  # p2-fixes: not in tree n


def seg_bytes(s: Segment) -> bytes:
    """Everything of a segment except its wall-clock stamps (and, in tree p2, the temperature's ``beh_temp`` field
    when it is None: a field tree n does not have; in the critic tree its four critic fields when None)."""
    return pickle.dumps({f.name: getattr(s, f.name) for f in fields(s) if f.name not in ("t_first", "t_last")
                         and not (f.name == "beh_temp" and s.beh_temp is None)
                         and not (f.name in ("values_c", "crit", "bootstrap_c", "rewards_c")
                                  and getattr(s, f.name) is None)}, protocol=4)


def main(mode: str, ratings: str) -> None:
    torch.set_num_threads(1)  # the CPU update is bit-reproducible only single-threaded (scatter / index_add sums)
    torch.manual_seed(3)
    vocab = Vocab(make_synthetic_vocab())
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True, cycle_ban=True, commit_select=True,
                                           removal_rule=True, removal_ratings=ratings))
    net = SpireNet(tok.layout, TINY)
    policy = TorchPolicy(net, tok, use_graphs=False, seed=4, exploration_by_kind=EXPLORATION)
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32), advice=True)
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=40, hashes=False, error_prob=0.002,
                                                           abort_prob=0.05)) for i in range(6)]
    recs: list[dict] = []
    kw = {}
    if mode == "off":
        from spire.gym import PracticeConfig
        kw = {"practice": PracticeConfig(), "gym_seed": 9}
    actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(), vocab, 0), ActorConfig(max_batch=8, min_group=2),
                  store=store, on_episode=lambda r: recs.append(r.as_dict()), advice=ADVICE, **kw)
    counters = actor.run(max_decisions=1500)
    h_seg = hashlib.sha256()
    for s in segs:
        h_seg.update(seg_bytes(s))
    h_rec = hashlib.sha256(json.dumps([{k: v for k, v in r.items() if k != "seconds"} for r in recs],
                                      sort_keys=True).encode())
    cfg = PPOConfig(epochs=1, minibatch=256, micro_batch=128, lr=1e-3, seed=0, advice=ADVICE,
                    exploration_by_kind=EXPLORATION,
                    gae=GAEConfig(gamma=1.0, lam=0.99, macro_micro=True, lam_macro=0.99, lam_micro=0.99,
                                  stitch_segments=True))
    torch.manual_seed(5)
    learner = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu")
    m = learner.update(segs)
    h_par = hashlib.sha256()
    for k, v in learner.model.state_dict().items():
        h_par.update(k.encode())
        h_par.update(v.detach().contiguous().numpy().tobytes())
    # tree p2: the combat temperature's counters and learner metrics are not in tree n (notes/next/p2.md)
    metrics = {k: v for k, v in m.items() if not k.startswith("time_") and k not in ("samples_per_s", "lag_s")
               and k not in TEMP_METRICS}
    print(json.dumps({"tree": spire.__file__, "mode": mode, "segments": len(segs), "steps": sum(map(len, segs)),
                      "records": len(recs), "seg_hash": h_seg.hexdigest(), "rec_hash": h_rec.hexdigest(),
                      "counters": {k: v for k, v in counters.as_dict().items() if k not in TEMP_COUNTERS}, "rollout": {k: v for k, v in store.stats.as_dict().items() if k not in R07_COUNTERS},
                      "removal_steps": sum(r.get("removal", {}).get("steps", 0) for r in recs),
                      "metrics": json.loads(json.dumps(metrics, default=str)), "params": h_par.hexdigest()}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
