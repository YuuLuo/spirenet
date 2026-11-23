"""Helper of tests/test_dual.py D4 (not a test): in the tree on PYTHONPATH (this tree or the base tree p7b), a learner of
the test's v2a-free recipe loads CKPT, prints the digest of the loaded state, runs one update on the pickled segments,
saves OUT and prints the digest after the update (one JSON line). With the argument "dual" (this tree only) the learner
has a replica on cuda:1 and the line says whether the replica equals the loaded model.

    PYTHONPATH=<tree> DUAL_TABLE=<removal table> python -B dual_ref.py load CKPT SEGS.pkl OUT [dual]"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import sys

import torch

from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab, make_synthetic_vocab

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
BUCKETS = (20, 28, 40, 64, 80, 96, 128, 192, 256, 384)
V2A = dict(forecast=True, calc=True, cycle_ban=True, commit_select=True, map_tokens="no_combat", route_summary=True,
           second_boss_table=True)


def digest(lr: Learner) -> str:
    h = hashlib.sha256()
    for sd in (lr.model.state_dict(), lr.ema.state_dict()):
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    st = lr.opt.state_dict()["state"]
    for i in sorted(st):
        for f in sorted(st[i]):
            t = st[i][f]
            h.update(f"{i}/{f}".encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(t) else repr(t).encode())
    h.update(json.dumps([lr.version, lr.updates, lr.samples_seen, lr.ema_steps]).encode())
    return h.hexdigest()


def main() -> None:
    mode, ckpt, seg_path, out = sys.argv[1:5]
    dual = sys.argv[5:] == ["dual"]
    assert mode == "load"
    vocab = Vocab(make_synthetic_vocab())
    tok = Tokenizer(vocab, TokenizerConfig(token_buckets=BUCKETS, removal_rule=True, removal_ratings=os.environ["DUAL_TABLE"],
                                           rules_mode="free", **V2A))  # type: ignore[arg-type]
    cfg = PPOConfig(epochs=2, minibatch=512, micro_batch=256, micro_tokens=4096, lr=1e-4, seed=3, kl_lr_target=0.01,
                    weight_decay=0.01)
    torch.manual_seed(0)
    kw = {"replica_device": "cuda:1"} if dual else {}
    lr = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cuda:0", fast=True, graphs=True, **kw)
    lr.load(ckpt)
    row: dict = {"tree_dual": importlib.util.find_spec("spire.replica") is not None, "loaded": digest(lr)}
    if dual:
        rep = lr.replica
        assert rep is not None
        row["replica_equal"] = all(torch.equal(a.detach().cpu(), b.detach().cpu())
                                   for a, b in zip(lr.model.parameters(), rep.model.parameters()))
    with open(seg_path, "rb") as fh:
        segs = pickle.load(fh)
    lr.update(segs)
    lr.save(out)
    row["next"] = digest(lr)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
