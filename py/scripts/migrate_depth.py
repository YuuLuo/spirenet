"""Explicit CPU depth migration (notes/solo/deepen-spec.md 2): append identity blocks to the main token trunk,
preserve online / EMA / Adam / counters, and prove on real frames that the outputs are unchanged.

    CUDA_VISIBLE_DEVICES=-1 python -B scripts/migrate_depth.py IN.pt OUT.pt --layers 8 --frames FRAMES.bin \\
        [--max-frames 1000] [--vocab vocab.json --catalog catalog.json]

``--frames``: real frames (u32 length + payload each, as written by scripts/forecast_check.py record, i.e. flag-on
frames with the FORECAST trailer); for a checkpoint without the forecast input each frame's flag-off prefix
(``synth.strip_forecast``) is used. CPU fp32, one frame at a time, online and EMA weights; every output must agree
within 1e-5. The proof is written to OUT.pt.proof.json and into the checkpoint's extra["depth_migration"].
``--vocab`` / ``--catalog`` default to the checkpoint's own paths (hashes asserted either way).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

import torch

from spire.depth_migration import (DEPTH_INIT_SEED, equivalence, migrate_depth_optimizer, migrate_depth_weights,
                                   new_block_names, new_block_state)
from spire.learner import _param_groups
from spire.model import ModelConfig, SpireNet, model_config_dict
from spire.proto import parse_frame
from spire.synth import strip_forecast
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import build
from spire.vocab import Catalog, Vocab

TOLERANCE = 1e-5


def read_frames(path: Path) -> Iterator[bytes]:
    data = path.read_bytes()
    pos = 0
    while pos < len(data):
        n = int.from_bytes(data[pos : pos + 4], "little")
        yield data[pos + 4 : pos + 4 + n]
        pos += 4 + n
    assert pos == len(data), "truncated frames file"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=1000)
    ap.add_argument("--vocab", type=Path)
    ap.add_argument("--catalog", type=Path)
    args = ap.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1", "migration is CPU-only"
    assert not args.output.exists(), "output already exists; migration never overwrites a checkpoint"
    assert args.max_frames >= 1000, "the proof needs at least 1,000 real frames"
    torch.set_num_threads(4)
    source = args.checkpoint.read_bytes()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert ck["format"] == 1
    extra = dict(ck["extra"])
    before = build(ModelConfig, extra["model_config"])
    trunk = {int(k.split(".")[1]) for k in ck["model"] if k.startswith("blocks.")}
    assert trunk == set(range(before.n_layers)), "checkpoint blocks differ from its model config"
    assert args.layers > before.n_layers, f"--layers {args.layers} adds no block to the {before.n_layers}-block trunk"
    after = replace(before, n_layers=args.layers)
    vocab_path = args.vocab if args.vocab is not None else Path(extra["vocab"])
    catalog_path = args.catalog if args.catalog is not None else Path(extra["catalog"])
    vocab = Vocab.load(vocab_path)
    assert vocab.hash == extra["vocab_hash"], "source vocabulary must match exactly"
    catalog = Catalog.load(catalog_path, vocab)
    assert catalog.hash == extra["catalog_hash"], "source catalog must match exactly"
    tok_cfg = build(TokenizerConfig, extra["tokenizer"])
    tok = Tokenizer(vocab, tok_cfg)
    net = SpireNet(tok.layout, after, catalog).eval()
    new = new_block_state(net, before.n_layers)
    migrated = {key: migrate_depth_weights(ck[key], net, new) for key in ("model", "ema")}
    opt = torch.optim.AdamW(_param_groups(net, float(ck["config"]["weight_decay"])))
    optimizer = migrate_depth_optimizer(ck["optimizer"], net, before.n_layers, opt)
    opt.load_state_dict(optimizer)
    frames = list(read_frames(args.frames))[: args.max_frames]
    assert len(frames) == args.max_frames, f"{args.frames}: only {len(frames)} frames"
    samples = [tok.encode(parse_frame(p if tok_cfg.forecast else strip_forecast(p))) for p in frames]
    checks: dict[str, Any] = {}
    n_before = 0
    for key in ("model", "ema"):
        old = SpireNet(tok.layout, before, catalog).eval()
        old.load_state_dict(ck[key])
        n_before = old.n_params
        deep = SpireNet(tok.layout, after, catalog).eval()
        deep.load_state_dict(migrated[key])
        checks[key] = equivalence(samples, tok, old, deep)
        worst = max(checks[key]["max_abs_diff"].values())
        assert worst < TOLERANCE, f"{key}: outputs differ by {worst}"
        assert checks[key]["argmax_agree"] == checks[key]["decisions"], key
    proof = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "version": int(ck["counters"]["version"]), "samples_seen": int(ck["counters"]["samples_seen"]),
        "device": "cpu", "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
        "n_layers": [before.n_layers, after.n_layers],
        "parameters": [n_before, net.n_params], "additional_parameters": net.n_params - n_before,
        "new_parameters": new_block_names(net, before.n_layers), "init_seed": DEPTH_INIT_SEED,
        "init": "appended blocks: proj / ff2 weights and biases 0; qkv / ff1 / film weights N(0, init_std), biases 0; "
                "same values online and EMA",
        "optimizer": "existing steps / moments / group settings exact (remapped by position in group); "
                     "appended blocks zero moments, step 0",
        "frames": str(args.frames.resolve()), "frames_used": len(frames),
        "frame_form": "as recorded" if tok_cfg.forecast else "strip_forecast (flag-off prefix)",
        "tolerance": TOLERANCE,
        "equivalence_cpu_fp32_b1": checks,
        "model_config_before": model_config_dict(before), "model_config_after": model_config_dict(after),
    }
    extra["model_config"] = model_config_dict(after)
    extra["depth_migration"] = proof
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, **migrated, "optimizer": optimizer, "extra": extra}, args.output)
    proof_path = args.output.with_name(args.output.name + ".proof.json")
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "proof": str(proof_path),
                      **{k: v for k, v in proof.items() if k != "new_parameters"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
