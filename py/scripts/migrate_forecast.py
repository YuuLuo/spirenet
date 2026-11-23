"""Explicit CPU forecast migration (notes/forecast/contract.md 4): add the zero-gated FORECAST input, preserve
online/EMA/Adam, and prove on frames that the outputs are unchanged.

    CUDA_VISIBLE_DEVICES=-1 python -B scripts/migrate_forecast.py --checkpoint SRC.pt --output DST.pt \\
        --vocab .../vocab.json --catalog .../catalog.json [--frames FRAMES.bin] [--proof-output proof.json]

``--frames``: flag-on frames (u32 length + payload each, as written by scripts/forecast_check.py record). The
source network sees each frame's v1 prefix (``synth.strip_forecast``: the host's flag-off frame of the same
state), the migrated one the full frame; CPU fp32, one frame at a time, online and EMA weights. Without
``--frames`` the proof uses synthetic frames of the real vocabulary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from spire.forecast_migration import migrate_forecast_optimizer, migrate_forecast_weights
from spire.learner import _param_groups
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.schema import DECISION_KINDS, Kind, reset_flags_for
from spire.synth import FrameGen, strip_forecast
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import build
from spire.vocab import Catalog, Vocab


def read_frames(path: Path) -> Iterator[bytes]:
    data = path.read_bytes()
    pos = 0
    while pos < len(data):
        n = int.from_bytes(data[pos : pos + 4], "little")
        yield data[pos + 4 : pos + 4 + n]
        pos += 4 + n


def synthetic_frames(vocab: Vocab, catalog: Catalog, n: int) -> list[bytes]:
    gen = FrameGen(vocab, catalog)
    rng = np.random.default_rng(0x46435354)
    out = []
    for i in range(n):
        ctx = gen.random_ctx(rng, 4)
        ctx.reset_flags = reset_flags_for(True, ctx.reset_flags)
        out.append(gen.frame(rng, int(DECISION_KINDS[i % len(DECISION_KINDS)]) if i % 2 else int(Kind.COMBAT), ctx))
    return out


@torch.no_grad()
def equivalence(frames: list[bytes], tok_off: Tokenizer, tok_on: Tokenizer, off: SpireNet, on: SpireNet
                ) -> dict[str, Any]:
    """Max |difference| per output, bit-identical share and greedy-argmax agreement (CPU fp32, B = 1)."""
    worst = {k: 0.0 for k in OUTPUT_KEYS}
    exact = same_argmax = n_dec = same_shape = exact_same_shape = 0
    for p in frames:
        a, b = tok_off.encode(parse_frame(strip_forecast(p))), tok_on.encode(parse_frame(p))
        x, y = off(tok_off.collate([a])), on(tok_on.collate([b]))
        ok = True
        for k in OUTPUT_KEYS:
            fin = torch.isfinite(x[k])
            assert torch.equal(fin, torch.isfinite(y[k])), k
            d = float((x[k][fin] - y[k][fin]).abs().max()) if fin.any() else 0.0
            worst[k] = max(worst[k], d)
            ok &= torch.equal(x[k], y[k])
        exact += ok
        shape = tok_off.shape_of([a]).kids == tok_on.shape_of([b]).kids
        same_shape += shape
        exact_same_shape += ok and shape
        if a.n_actions:
            n_dec += 1
            same_argmax += int(x["logits"][0, : a.n_actions].argmax() == y["logits"][0, : b.n_actions].argmax())
    return {"frames": len(frames), "decisions": n_dec, "bit_identical": exact,
            "same_padded_kids": same_shape, "bit_identical_same_padded_kids": exact_same_shape,
            "max_abs_diff": worst, "argmax_agree": same_argmax}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--frames", type=Path)
    ap.add_argument("--max-frames", type=int, default=1000)
    ap.add_argument("--proof-output", type=Path)
    args = ap.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1", "migration is CPU-only"
    assert not args.output.exists(), "output already exists; migration never overwrites a checkpoint"
    torch.set_num_threads(4)
    source = args.checkpoint.read_bytes()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert ck["format"] == 1
    extra = dict(ck["extra"])
    vocab = Vocab.load(args.vocab)
    assert vocab.hash == extra["vocab_hash"], "source vocabulary must match exactly"
    catalog = Catalog.load(args.catalog, vocab)
    assert catalog.hash == extra["catalog_hash"], "source catalog must match exactly"
    before = build(TokenizerConfig, extra["tokenizer"])
    assert not before.forecast, "already migrated"
    after = replace(before, forecast=True)
    tok_off, tok_on = Tokenizer(vocab, before), Tokenizer(vocab, after)
    model_config = build(ModelConfig, extra["model_config"])
    net = SpireNet(tok_on.layout, model_config, catalog).eval()
    migrated = {key: migrate_forecast_weights(ck[key], net, tok_on.layout) for key in ("model", "ema")}
    opt = torch.optim.AdamW(_param_groups(net, float(ck["config"]["weight_decay"])))
    optimizer = migrate_forecast_optimizer(ck["optimizer"], net, tok_on.layout, opt)
    opt.load_state_dict(optimizer)
    frames = (list(read_frames(args.frames)) if args.frames is not None
              else synthetic_frames(vocab, catalog, args.max_frames))[: args.max_frames]
    checks: dict[str, Any] = {}
    for key in ("model", "ema"):
        off = SpireNet(tok_off.layout, model_config, catalog).eval()
        off.load_state_dict(ck[key])
        on = SpireNet(tok_on.layout, model_config, catalog).eval()
        on.load_state_dict(migrated[key])
        checks[key] = equivalence(frames, tok_off, tok_on, off, on)
    d = model_config.d_model
    proof = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "device": "cpu", "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
        "additional_parameters": 12 * d + d + 1,
        "optimizer": "existing steps/moments/group settings exact (remapped by name); type_emb moments + zero row; "
                     "forecast_in / forecast_gate zero moments, step 0",
        "frames": str(args.frames) if args.frames is not None else f"synthetic x{len(frames)}",
        "equivalence_cpu_fp32_b1": checks,
        "tokenizer_before": asdict(before), "tokenizer_after": asdict(after),
    }
    extra["tokenizer"] = asdict(after)
    extra["vocab"], extra["catalog"] = str(args.vocab.resolve()), str(args.catalog.resolve())
    extra["forecast_migration"] = proof
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, **migrated, "optimizer": optimizer, "extra": extra}, args.output)
    if args.proof_output is not None:
        args.proof_output.parent.mkdir(parents=True, exist_ok=True)
        args.proof_output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **proof}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
