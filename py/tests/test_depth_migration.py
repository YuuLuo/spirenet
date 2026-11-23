"""Depth migration (notes/solo/deepen-spec.md 2-3): appended blocks are the identity, Adam history is kept."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spire.depth_migration import (equivalence, migrate_depth_optimizer, migrate_depth_weights, new_block_names,
                                   new_block_state)
from spire.learner import Learner, PPOConfig, _param_groups
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.schema import DECISION_KINDS, Kind
from spire.synth import FrameGen
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Catalog, Vocab

SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16, seat_symmetry=True)
FLAGS = {"coop_features": True, "mend_target": True, "cycle_ban": True, "commit_select": True}


def _samples(tok: Tokenizer, gen: FrameGen, n: int) -> list:
    rng = np.random.default_rng(31)
    out = []
    for i in range(n):
        ctx = gen.random_ctx(rng, int(rng.integers(1, 5)))
        out.append(tok.encode(parse_frame(gen.frame(rng, int(DECISION_KINDS[i % len(DECISION_KINDS)])
                                                    if i % 2 else int(Kind.COMBAT), ctx))))
    return out


def _pair(vocab: Vocab, catalog: Catalog, layers: int) -> tuple[Tokenizer, SpireNet, SpireNet]:
    tok = Tokenizer(vocab, TokenizerConfig(**FLAGS))
    torch.manual_seed(0)
    old = SpireNet(tok.layout, SMALL, catalog).eval()
    deep = SpireNet(tok.layout, replace(SMALL, n_layers=layers), catalog).eval()
    deep.load_state_dict(migrate_depth_weights(old.state_dict(), deep, new_block_state(deep, SMALL.n_layers)))
    return tok, old, deep


def test_identity_on_synthetic_frames(vocab: Vocab, catalog: Catalog, gen: FrameGen) -> None:
    tok, old, deep = _pair(vocab, catalog, 4)
    samples = _samples(tok, gen, 80)
    r = equivalence(samples, tok, old, deep)
    assert r["bit_identical"] == len(samples) and r["argmax_agree"] == r["decisions"] > 0
    assert all(v == 0.0 for v in r["max_abs_diff"].values())
    with torch.no_grad():  # batched too
        x, y = old(tok.collate(samples)), deep(tok.collate(samples))
    for k in OUTPUT_KEYS:
        assert torch.equal(x[k], y[k]), k


def test_new_blocks(vocab: Vocab, catalog: Catalog, gen: FrameGen) -> None:
    tok, old, deep = _pair(vocab, catalog, 4)
    sd = deep.state_dict()
    for name in new_block_names(deep, 2):
        module, kind = name.split(".")[2:]
        if module in ("proj", "ff2") or kind == "bias":
            assert torch.count_nonzero(sd[name]) == 0, name
        else:
            assert abs(float(sd[name].std()) - SMALL.init_std) < 0.3 * SMALL.init_std, name
    assert torch.equal(sd["blocks.2.qkv.weight"], new_block_state(deep, 2)["blocks.2.qkv.weight"])  # deterministic
    assert not torch.equal(sd["blocks.2.qkv.weight"], sd["blocks.3.qkv.weight"])
    # the new blocks learn: gradients reach qkv / ff1 / film through the zero output projections
    deep.train()
    out = deep(tok.collate(_samples(tok, gen, 8)))
    (out["logits"][torch.isfinite(out["logits"])].sum() + out["value"].sum()).backward()
    assert deep.blocks[3].proj.weight.grad.abs().sum() > 0 and deep.blocks[3].ff2.weight.grad.abs().sum() > 0


def test_refuses_no_change(vocab: Vocab, catalog: Catalog) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(**FLAGS))
    net = SpireNet(tok.layout, SMALL, catalog)
    with pytest.raises(AssertionError, match="must add blocks"):
        new_block_state(net, SMALL.n_layers)


def test_checkpoint_migration_preserves_adam(vocab: Vocab, catalog: Catalog, gen: FrameGen, tmp_path: Path) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(**FLAGS))
    old = Learner(SpireNet(tok.layout, SMALL, catalog), tok, PPOConfig(), device="cpu", fast=False)
    for p in old.model.parameters():
        p.grad = torch.ones_like(p)
    old.opt.step()
    before = {n: {k: v.clone() for k, v in old.opt.state[p].items()} for n, p in old.model.named_parameters()}
    src = tmp_path / "before.pt"
    old.save(src)
    ck = torch.load(src, map_location="cpu", weights_only=True)
    deep_cfg = replace(SMALL, n_layers=4)
    net = SpireNet(tok.layout, deep_cfg, catalog)
    new = new_block_state(net, 2)
    migrated = {k: migrate_depth_weights(ck[k], net, new) for k in ("model", "ema")}
    opt = torch.optim.AdamW(_param_groups(net, 0.0))
    optimizer = migrate_depth_optimizer(ck["optimizer"], net, 2, opt)
    dst = tmp_path / "after.pt"
    torch.save({**ck, **migrated, "optimizer": optimizer}, dst)
    deep = Learner(SpireNet(tok.layout, deep_cfg, catalog), tok, PPOConfig(), device="cpu", fast=False)
    with pytest.raises(RuntimeError):
        deep.load(src)
    deep.load(dst)
    added = set(new_block_names(deep.model, 2))
    for name, p in deep.model.named_parameters():
        st = deep.opt.state[p]
        if name in added:
            assert all(torch.count_nonzero(v) == 0 for v in st.values()), name
        else:
            for k, v in st.items():
                assert torch.equal(v, before[name][k]), (name, k)
    assert deep.version == old.version and deep.samples_seen == old.samples_seen
    for k in ("model", "ema"):
        assert torch.equal(migrated[k]["blocks.2.qkv.weight"], new["blocks.2.qkv.weight"])
    for p in deep.model.parameters():
        p.grad = torch.ones_like(p)
    deep.opt.step()
    opt2 = torch.optim.AdamW(_param_groups(deep.model, 0.0))
    with pytest.raises(AssertionError):
        migrate_depth_optimizer(deep.opt.state_dict(), deep.model, 2, opt2)
