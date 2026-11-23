"""Python-side forecast checks on real frames (notes/forecast/py.md): play greedy 4-seat games on a forecast dist
with the FORECAST flag, and on every decision frame

* tokenize with the native, the numpy and the reference tokenizer (forecast=True): bit-identical Samples;
* run the source checkpoint on the frame's v1 prefix (``synth.strip_forecast``) and its forecast migration on the
  full frame (CPU fp32, B = 1): max |difference| per output, bit-identical share, greedy-argmax agreement;
* count FORECAST rows / classes.

Actions are sampled from the source network's policy (greedy play without the cycle ban loops on reward
screens); ``--max-decisions`` caps a game. Frames are written to ``--out`` (u32 length + payload).

    CUDA_VISIBLE_DEVICES=-1 python -B scripts/forecast_check.py --checkpoint cr-new.pt --dist DIST --frames 1000 \\
        --out frames.bin --report report.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spire.forecast_migration import migrate_forecast_weights
from spire.host import REPO, HostConfig, HostProcess
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet
from spire.proto import ResetCmd, parse_frame
from spire.schema import ForecastClass, ForecastCol, Kind, reset_flags_for
from spire.synth import strip_forecast
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.tokenize_ref import encode_reference
from spire.train import build
from spire.vocab import Catalog, Vocab

_FIELDS = ("types", "raw", "rel", "edges", "cond")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dist", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=46435354)
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--character", default="NECROBINDER")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-decisions", type=int, default=500)
    args = ap.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert not args.out.exists()
    torch.set_num_threads(args.threads)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    extra = ck["extra"]
    vocab = Vocab.load(args.dist / "vocab" / "vocab.json")
    catalog = Catalog.load(args.dist / "vocab" / "catalog.json", vocab)
    assert vocab.hash == extra["vocab_hash"] and catalog.hash == extra["catalog_hash"]
    before = build(TokenizerConfig, extra["tokenizer"])
    assert not before.forecast
    after = replace(before, forecast=True)
    tok_off, tok_on = Tokenizer(vocab, before), Tokenizer(vocab, after)
    tok_np = Tokenizer(vocab, after, native=False)
    assert tok_on.native and not tok_np.native
    mc = build(ModelConfig, extra["model_config"])
    off = SpireNet(tok_off.layout, mc, catalog).eval()
    off.load_state_dict(ck["model"])
    on = SpireNet(tok_on.layout, mc, catalog).eval()
    on.load_state_dict(migrate_forecast_weights(ck["model"], on, tok_on.layout))
    env = {**os.environ, "SPIREHOST_GAME_DIR": str(REPO / "host" / "game"), "DOTNET_ROOT": "/opt/dotnet",
           "LC_ALL": "C.UTF-8"}
    host = HostProcess(HostConfig(command=("/opt/dotnet/dotnet", str(args.dist / "SpireHost.dll"), "--serve"),
                                  env=env, transport="stdio", max_restarts=0, start_timeout=180, step_timeout=60,
                                  log_path=str(args.out.with_suffix(".host.log")), verify_hashes=True))
    flags = reset_flags_for(True)
    chars = (vocab.id("character", args.character),) * 4
    rng = np.random.default_rng(args.seed)
    worst = {k: 0.0 for k in OUTPUT_KEYS}
    n = n_dec = exact = argmax_same = tok_mismatch = same_kids = exact_same_kids = games = 0
    combat = fc_rows = 0
    classes: Counter[str] = Counter()
    ncand: Counter[int] = Counter()
    kinds: Counter[str] = Counter()
    kid_off = kid_on = 0
    t0 = time.monotonic()
    t_net = 0.0
    with args.out.open("wb") as fo:
        try:
            while n < args.frames:
                games += 1
                frame = host.reset_with(ResetCmd(chars, args.ascension, seed=int(rng.integers(0, 34**12)),
                                                 flags=flags, max_decisions=args.max_decisions))
                while frame.is_obs and n < args.frames:
                    p = bytes(frame.payload)
                    fo.write(len(p).to_bytes(4, "little") + p)
                    fc = frame.forecast
                    assert fc is not None
                    n += 1
                    kinds[Kind(frame.kind).name] += 1
                    if frame.kind == Kind.COMBAT:
                        combat += 1
                        fc_rows += fc.shape[0]
                    slot_key = fc[:, [ForecastCol.OWNER_ROW, ForecastCol.AHEAD]]
                    idx = np.unique(slot_key, axis=0, return_index=True)[1] if len(fc) else []
                    for i in idx:  # one row per (owner, AHEAD) slot
                        classes[f"t+{fc[i, ForecastCol.AHEAD]} {ForecastClass(fc[i, ForecastCol.CLASS]).name}"] += 1
                        ncand[int(fc[i, ForecastCol.N_CAND])] += 1
                    full = parse_frame(p)
                    a = tok_on.encode(full)
                    b, c = tok_np.encode(full), encode_reference(tok_on, full)
                    ok = all(np.array_equal(getattr(a, k), getattr(x, k)) for x in (b, c) for k in _FIELDS)
                    tok_mismatch += not (ok and a.counts == b.counts == c.counts)
                    s = tok_off.encode(parse_frame(strip_forecast(p)))
                    kid_off += s.counts[1]
                    kid_on += a.counts[1]
                    t1 = time.monotonic()
                    with torch.no_grad():
                        x, y = off(tok_off.collate([s])), on(tok_on.collate([a]))
                    t_net += time.monotonic() - t1
                    bit = True
                    for k in OUTPUT_KEYS:
                        fin = torch.isfinite(x[k])
                        assert torch.equal(fin, torch.isfinite(y[k])), k
                        if fin.any():
                            worst[k] = max(worst[k], float((x[k][fin] - y[k][fin]).abs().max()))
                        bit &= torch.equal(x[k], y[k])
                    exact += bit
                    shape = tok_off.shape_of([s]).kids == tok_on.shape_of([a]).kids
                    same_kids += shape
                    exact_same_kids += bit and shape
                    na = s.n_actions
                    n_dec += 1
                    argmax_same += int(x["logits"][0, :na].argmax()) == int(y["logits"][0, :na].argmax())
                    prob = torch.softmax(x["logits"][0, :na].double(), 0).numpy()
                    frame = host.step(int(rng.choice(na, p=prob / prob.sum())))
        finally:
            host.close()
    rep: dict[str, Any] = {
        "checkpoint": str(args.checkpoint), "dist": str(args.dist), "games": games, "frames": n,
        "combat_frames": combat, "kinds": dict(kinds), "decisions": n_dec,
        "tokenizer_native_numpy_reference_mismatch_frames": tok_mismatch,
        "forecast_rows_per_combat_frame": fc_rows / max(combat, 1),
        "slots_by_class": dict(classes), "slots_by_n_cand": {str(k): v for k, v in sorted(ncand.items())},
        "kid_rows_per_frame_off_on": [kid_off / max(n, 1), kid_on / max(n, 1)],
        "bit_identical_frames": exact, "same_padded_kids": same_kids, "bit_identical_same_padded_kids": exact_same_kids,
        "max_abs_diff": worst, "argmax_agree": argmax_same,
        "seconds": time.monotonic() - t0, "net_seconds_both": t_net,
    }
    args.report.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
