"""Explicit CPU checkpoint migration, optionally proving SP equality against the frozen pre-H5 model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spire.learner import _param_groups
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet, model_config_dict
from spire.proto import parse_frame
from spire.schema import DECISION_KINDS, ViewMode
from spire.seat_migration import (migrate_seat_symmetry_optimizer, migrate_seat_symmetry_weights,
                                  seat_column_groups)
from spire.synth import FrameGen
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import build
from spire.vocab import Catalog, Vocab


def _baseline(path: Path, tok: Tokenizer, config: ModelConfig, catalog: Catalog) -> Any:
    name = "spire._model_before_h5"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    data = model_config_dict(config)
    del data["seat_symmetry"]
    return module.SpireNet(tok.layout, module.ModelConfig(**data), catalog).eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True, help="explicit stable vocab; source checkpoint paths are not read")
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--source-view-mode", choices=("team", "seat"), default="team")
    ap.add_argument("--reset-optimizer", action="store_true", help="discard all Adam moments and steps")
    ap.add_argument("--baseline-model", type=Path, help="frozen original model.py for live-checkpoint bit proof")
    ap.add_argument("--proof-output", type=Path)
    args = ap.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1", "this migration/proof tool is CPU-only"
    assert not args.output.exists(), "output already exists; migration never overwrites a checkpoint"
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert ck["format"] == 1
    extra = dict(ck["extra"])
    vocab = Vocab.load(args.vocab)
    assert vocab.hash == extra["vocab_hash"], "source vocabulary must match exactly"
    catalog = Catalog.load(args.catalog, vocab)
    assert catalog.hash == extra["catalog_hash"], "source catalog must match exactly"
    tok = Tokenizer(vocab, build(TokenizerConfig, extra["tokenizer"]))
    config = build(ModelConfig, extra["model_config"])
    assert not config.seat_symmetry, "already migrated"
    migrated_config = replace(config, seat_symmetry=True)
    net = SpireNet(tok.layout, migrated_config, catalog).eval()
    source_view = ViewMode.TEAM if args.source_view_mode == "team" else ViewMode.SEAT
    migrated = {key: migrate_seat_symmetry_weights(ck[key], net, tok.layout, source_view_mode=source_view)
                for key in ("model", "ema")}
    proof: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()), "source_view_mode": args.source_view_mode,
        "device": "cpu", "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
        "input_width_before_after": [net.in_proj.in_features, net.in_proj.in_features],
        "additional_parameters": net.config.d_model * 4,
        "column_groups": seat_column_groups(tok.layout, config.d_id),
        "optimizer": ("all Adam state reset; saved parameter-group hyperparameters retained" if args.reset_optimizer
                      else "Adam moments copied with weights; steps and saved parameter-group hyperparameters retained"),
        "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    }
    if args.baseline_model is not None:
        baseline = _baseline(args.baseline_model, tok, config, catalog)
        current = SpireNet(tok.layout, config, catalog).eval()
        rng = np.random.default_rng(20261115)
        gen = FrameGen(vocab, catalog.hash)
        # All decision kinds, two separate contexts each; same batch/ordering for every comparison.
        samples = []
        for kind in DECISION_KINDS:
            for _ in range(2):
                ctx = gen.random_ctx(rng, 1, view=source_view)
                ctx.view_seat = 0 if source_view == ViewMode.SEAT else -1
                samples.append(tok.encode(parse_frame(gen.frame(rng, kind, ctx))))
        batches = [tok.collate(samples[i:i + 4]) for i in range(0, len(samples), 4)]
        comparisons = {}
        with torch.inference_mode():
            for key in ("model", "ema"):
                baseline.load_state_dict(ck[key], strict=True)
                current.load_state_dict(ck[key], strict=True)
                net.load_state_dict(migrated[key], strict=True)
                checked = 0
                for batch in batches:
                    before, changed, after = baseline(batch), current(batch), net(batch)
                    for output in OUTPUT_KEYS:
                        assert torch.equal(before[output].view(torch.int32), changed[output].view(torch.int32)), \
                            f"{key}/{output}: code change differs"
                        assert torch.equal(before[output].view(torch.int32), after[output].view(torch.int32)), \
                            f"{key}/{output}: migration differs"
                        checked += before[output].numel()
                chars = torch.arange(1, net.cond_char.num_embeddings)
                c = net.cond_char(chars)
                one = torch.zeros((len(chars), 4), dtype=torch.long)
                one[:, 0] = chars
                for encoded in (one, chars[:, None].repeat(1, 4)):
                    assert torch.equal(net.character_condition(encoded).view(torch.int32), c.view(torch.int32))
                comparisons[key] = {"frames": len(samples), "kinds": len(DECISION_KINDS),
                                    "output_values_compared": checked, "all_bits_identical": True,
                                    "one_and_four_identical_character_means_bit_identical": True}
        proof["comparisons"] = comparisons
        proof["baseline_sha256"] = hashlib.sha256(args.baseline_model.read_bytes()).hexdigest()
    opt = torch.optim.AdamW(_param_groups(net, float(ck["config"]["weight_decay"])))
    optimizer = migrate_seat_symmetry_optimizer(ck["optimizer"], net, tok.layout, opt,
                                                source_view_mode=source_view, reset=args.reset_optimizer)
    opt.load_state_dict(optimizer)
    extra["model_config"] = model_config_dict(migrated_config)
    extra["vocab"], extra["catalog"] = str(args.vocab.resolve()), str(args.catalog.resolve())
    extra["seat_symmetry_migration"] = proof
    ck = {**ck, **migrated, "optimizer": optimizer, "extra": extra}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.output)
    if args.proof_output is not None:
        args.proof_output.parent.mkdir(parents=True, exist_ok=True)
        args.proof_output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **proof}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
