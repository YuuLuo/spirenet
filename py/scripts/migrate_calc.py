"""Explicit CPU calc migration (notes/calc/contract.md 4, gate PG3): add the zero-gated TURN / FOE / ROUTE input,
preserve online / EMA / Adam / counters, and prove on real frames that the outputs are unchanged.

    CUDA_VISIBLE_DEVICES=-1 python -B scripts/migrate_calc.py --checkpoint SRC.pt --output DST.pt \\
        --vocab .../vocab.json --catalog .../catalog.json --frames FRAMES.bin [--max-frames 1000]

``--frames``: real flag-on frames with the calc trailers (u32 length + payload each); with or without the FORECAST
trailer. The source network sees each frame with neither calc trailer (and the forecast trailer iff the source
has the forecast input: ``synth.keep_trailers``, the host's frame of the same state without bit 9), the migrated one
the frame with the calc trailers. CPU fp32, 4 threads, one frame at a time, online and EMA weights. A hook on
``phi_kid`` asserts every pooled kid row finite, so with ``calc_gate`` = 0 each calc kid adds exactly +-0. The source
may have 4 or 8 layers and the forecast on or off. The proof goes to DST.pt.proof.json and extra["calc_migration"].
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

from spire.calc_migration import ADDED, migrate_calc_optimizer, migrate_calc_weights
from spire.learner import _param_groups
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet, calc_type_rows
from spire.proto import parse_frame
from spire.schema import Kind
from spire.synth import keep_trailers
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import build
from spire.vocab import Catalog, Vocab

TOLERANCE = 1e-6


def read_frames(path: Path) -> Iterator[bytes]:
    data = path.read_bytes()
    pos = 0
    while pos < len(data):
        n = int.from_bytes(data[pos : pos + 4], "little")
        yield data[pos + 4 : pos + 4 + n]
        pos += 4 + n
    assert pos == len(data), "truncated frames file"


@torch.no_grad()
def equivalence(pairs: list[tuple[bytes, bytes]], tok_off: Tokenizer, tok_on: Tokenizer, off: SpireNet,
                on: SpireNet) -> dict[str, Any]:
    """Max |difference| per output, bit-identical share and greedy-argmax agreement (CPU fp32, B = 1)."""
    finite_kids = []
    h = on.phi_kid.register_forward_hook(lambda m, i, o: finite_kids.append(bool(torch.isfinite(o).all())))
    assert on.calc_gate is not None and not bool(on.calc_gate.any()), "calc_gate must be exactly 0"
    worst = {k: 0.0 for k in OUTPUT_KEYS}
    exact = same_argmax = n_dec = same_shape = exact_same_shape = 0
    for p_off, p_on in pairs:
        a, b = tok_off.encode(parse_frame(p_off)), tok_on.encode(parse_frame(p_on))
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
    h.remove()
    assert all(finite_kids) and len(finite_kids) == len(pairs), "a pooled kid row is not finite"
    return {"frames": len(pairs), "decisions": n_dec, "bit_identical": exact,
            "same_padded_kids": same_shape, "bit_identical_same_padded_kids": exact_same_shape,
            "max_abs_diff": worst, "argmax_agree": same_argmax, "gated_calc_contribution": "exactly 0 (gate 0, kids finite)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=1000)
    args = ap.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1", "migration is CPU-only"
    assert not args.output.exists(), "output already exists; migration never overwrites a checkpoint"
    assert args.max_frames >= 1000, "the proof needs at least 1,000 real frames"
    torch.set_num_threads(4)  # with dozens of threads MKL's sgemm bits depend on the row count
    source = args.checkpoint.read_bytes()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert ck["format"] == 1
    extra = dict(ck["extra"])
    vocab = Vocab.load(args.vocab)
    assert vocab.hash == extra["vocab_hash"], "source vocabulary must match exactly"
    catalog = Catalog.load(args.catalog, vocab)
    assert catalog.hash == extra["catalog_hash"], "source catalog must match exactly"
    before = build(TokenizerConfig, extra["tokenizer"])
    assert not before.calc, "already migrated"
    after = replace(before, calc=True)
    tok_off, tok_on = Tokenizer(vocab, before), Tokenizer(vocab, after)
    model_config = build(ModelConfig, extra["model_config"])
    net = SpireNet(tok_on.layout, model_config, catalog).eval()
    migrated = {key: migrate_calc_weights(ck[key], net, tok_on.layout) for key in ("model", "ema")}
    opt = torch.optim.AdamW(_param_groups(net, float(ck["config"]["weight_decay"])))
    optimizer = migrate_calc_optimizer(ck["optimizer"], net, tok_on.layout, opt)
    opt.load_state_dict(optimizer)
    old_opt = ck["optimizer"]
    frames = [p for p in read_frames(args.frames) if not parse_frame(p).is_error][: args.max_frames]
    assert len(frames) == args.max_frames, f"{args.frames}: only {len(frames)} frames"
    parsed = [parse_frame(p) for p in frames]
    n_combat = sum(f.kind == Kind.COMBAT for f in parsed)
    n_map = sum(f.route is not None and f.route.shape[0] > 0 for f in parsed)
    assert n_combat >= 500 and n_map >= 50, f"frame mix: {n_combat} combat, {n_map} with ROUTE rows"
    fc = before.forecast
    pairs = [(keep_trailers(p, forecast=fc, calc=False), keep_trailers(p, forecast=fc, calc=True)) for p in frames]
    checks: dict[str, Any] = {}
    n_before = 0
    for key in ("model", "ema"):
        off = SpireNet(tok_off.layout, model_config, catalog).eval()
        off.load_state_dict(ck[key])
        n_before = off.n_params
        on = SpireNet(tok_on.layout, model_config, catalog).eval()
        on.load_state_dict(migrated[key])
        checks[key] = equivalence(pairs, tok_off, tok_on, off, on)
        worst = max(checks[key]["max_abs_diff"].values())
        assert worst <= TOLERANCE, f"{key}: outputs differ by {worst}"
        assert checks[key]["argmax_agree"] == checks[key]["decisions"], key
    # old optimizer entries bytewise equal (type_emb: its old rows); state ids enumerate the live groups in order
    pname = {id(q): n for n, q in net.named_parameters()}
    idx_name = dict(enumerate(pname[id(q)] for g in opt.param_groups for q in g["params"]))
    old_rows = calc_type_rows(before.forecast)
    n_entries = 0
    for g_old, g_new in zip(old_opt["param_groups"], optimizer["param_groups"]):
        kept = [j for j in g_new["params"] if idx_name[j] not in ADDED]
        assert len(kept) == len(g_old["params"])
        for i_old, j in zip(g_old["params"], kept):
            if i_old not in old_opt["state"]:
                continue
            for k, v in old_opt["state"][i_old].items():
                w = optimizer["state"][j][k]
                w = w[:old_rows] if idx_name[j] == "type_emb.weight" and k != "step" else w
                assert torch.equal(w, v), (idx_name[j], k)
                n_entries += 1
    proof = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "version": int(ck["counters"]["version"]), "samples_seen": int(ck["counters"]["samples_seen"]),
        "device": "cpu", "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
        "n_layers": model_config.n_layers, "forecast": before.forecast,
        "parameters": [n_before, net.n_params], "additional_parameters": net.n_params - n_before,
        "optimizer": "existing steps / moments / group settings exact (remapped by name; checked: "
                     f"{n_entries} tensors equal); type_emb moments + zero rows; calc_in / calc_gate zero moments, "
                     "step 0",
        "frames": str(args.frames.resolve()), "frames_used": len(frames), "combat_frames": n_combat,
        "frames_with_route_rows": n_map, "tolerance": TOLERANCE,
        "equivalence_cpu_fp32_b1": checks,
        "tokenizer_before": asdict(before), "tokenizer_after": asdict(after),
    }
    extra["tokenizer"] = asdict(after)
    extra["vocab"], extra["catalog"] = str(args.vocab.resolve()), str(args.catalog.resolve())
    extra["calc_migration"] = proof
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, **migrated, "optimizer": optimizer, "extra": extra}, args.output)
    proof_path = args.output.with_name(args.output.name + ".proof.json")
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "proof": str(proof_path), **proof}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
