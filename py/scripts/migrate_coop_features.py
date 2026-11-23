"""Explicit CPU Stage B migration: zero new inputs/MEND gate, preserve online/EMA/Adam."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import torch

from spire.coop_migration import migrate_coop_features_optimizer, migrate_coop_features_weights
from spire.learner import _param_groups
from spire.model import ModelConfig, SpireNet
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import build
from spire.vocab import Catalog, Vocab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--proof-output", type=Path)
    args = ap.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1", "migration is CPU-only"
    assert not args.output.exists(), "output already exists; migration never overwrites a checkpoint"
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert ck["format"] == 1
    extra = dict(ck["extra"])
    vocab = Vocab.load(args.vocab)
    assert vocab.hash == extra["vocab_hash"], "source vocabulary must match exactly"
    catalog = Catalog.load(args.catalog, vocab)
    assert catalog.hash == extra["catalog_hash"], "source catalog must match exactly"
    before = build(TokenizerConfig, extra["tokenizer"])
    assert not before.coop_features and not before.mend_target, "already migrated"
    after = replace(before, coop_features=True, mend_target=True)
    tok = Tokenizer(vocab, after)
    model_config = build(ModelConfig, extra["model_config"])
    net = SpireNet(tok.layout, model_config, catalog).eval()
    migrated = {key: migrate_coop_features_weights(ck[key], net, tok.layout) for key in ("model", "ema")}
    opt = torch.optim.AdamW(_param_groups(net, float(ck["config"]["weight_decay"])))
    optimizer = migrate_coop_features_optimizer(ck["optimizer"], net, tok.layout, opt)
    opt.load_state_dict(optimizer)
    proof = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "device": "cpu", "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
        "input_width_before_after": [model_config.d_id + tok.layout.legacy_n_wide, net.in_proj.in_features],
        "additional_parameters": net.config.d_model * (tok.layout.n_wide - tok.layout.legacy_n_wide + 16) + 1,
        "optimizer": "Existing steps/moments/group settings exact; added columns/moments zero; new parameters step zero",
        "mend": "New target-seat relation; legacy aux retained; target scalar gate initially zero",
        "tokenizer_before": asdict(before), "tokenizer_after": asdict(after),
    }
    extra["tokenizer"] = asdict(after)
    extra["vocab"], extra["catalog"] = str(args.vocab.resolve()), str(args.catalog.resolve())
    extra["coop_features_migration"] = proof
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, **migrated, "optimizer": optimizer, "extra": extra}, args.output)
    if args.proof_output is not None:
        args.proof_output.parent.mkdir(parents=True, exist_ok=True)
        args.proof_output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **proof}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
