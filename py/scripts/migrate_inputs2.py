"""Explicit CPU inputs-v2a migration (notes/inputs2/v2a.md 3, gate GM): add the second-boss table (a copy of the
encounter rows) and the zero-gated route summary AHEAD, switch a finished kickstart off, preserve online / EMA / Adam /
counters, and prove on real frames that the outputs are unchanged.

    CUDA_VISIBLE_DEVICES=-1 python -B scripts/migrate_inputs2.py --checkpoint SRC.pt --output DST.pt \\
        --vocab .../vocab.json --catalog .../catalog.json --frames F795.bin [--frames-off F793.bin] [--max-frames N]

``--frames``: real frames of a host with RESET bit 1 and the calc trailers (flags 795, or 539 without the forecast; u32
length + payload each). The migrated network sees them; the source network sees the frame of the same game moment
without bit 1: ``--frames-off`` (frames recorded at the same moments with bit 1 off, e.g. the same command stream
replayed with flags 793; aligned one to one) or else ``synth.strip_map_in_combat`` of each frame. The FORECAST trailer
is kept iff the source has the forecast input (``synth.keep_trailers``). CPU fp32, 4 threads, B = 1, online and EMA
weights; outputs compared: policy logits (every row; the legal rows are the finite ones), value and the three
auxiliary heads, bit for bit. A hook on ``phi_kid`` asserts every pooled kid row finite, so with ``ahead_gate`` = 0 the
AHEAD kid adds exactly +-0. The proof goes to DST.pt.proof.json and extra["inputs2_migration"].
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

from spire.inputs2_migration import ADDED, KEYS, check_source, migrate_inputs2_optimizer, migrate_inputs2_weights
from spire.kickstart import KickstartConfig, teacher_active
from spire.learner import _param_groups
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.schema import G, Kind, MapNodeFlag, Table
from spire.synth import keep_trailers, strip_map_in_combat
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
    assert pos == len(data), f"{path}: truncated frames file"


@torch.no_grad()
def equivalence(pairs: list[tuple[bytes, bytes]], tok_off: Tokenizer, tok_on: Tokenizer, off: SpireNet,
                on: SpireNet) -> dict[str, Any]:
    """Bit-identical share, max |difference| per output and greedy agreement (CPU fp32, B = 1)."""
    finite_kids: list[bool] = []
    h = on.phi_kid.register_forward_hook(lambda m, i, o: finite_kids.append(bool(torch.isfinite(o).all())))
    assert on.ahead_gate is not None and float(on.ahead_gate) == 0.0, "ahead_gate must be exactly 0"
    worst = {k: 0.0 for k in OUTPUT_KEYS}
    exact = same_argmax = n_dec = 0
    first_diff: list[int] = []
    for j, (p_off, p_on) in enumerate(pairs):
        a, b = tok_off.encode(parse_frame(p_off)), tok_on.encode(parse_frame(p_on))
        assert a.n_actions == b.n_actions and a.n_tokens == b.n_tokens and b.counts[1] == a.counts[1] + 1, \
            f"pair {j}: the samples differ by more than the AHEAD row: {a} {b}"
        x, y = off(tok_off.collate([a])), on(tok_on.collate([b]))
        ok = True
        for k in OUTPUT_KEYS:
            fin = torch.isfinite(x[k])
            assert torch.equal(fin, torch.isfinite(y[k])), (j, k)
            d = float((x[k][fin] - y[k][fin]).abs().max()) if fin.any() else 0.0
            worst[k] = max(worst[k], d)
            ok &= torch.equal(x[k], y[k])
        exact += ok
        if not ok and len(first_diff) < 10:
            first_diff.append(j)
        if a.n_actions:
            n_dec += 1
            same_argmax += int(x["logits"][0, : a.n_actions].argmax() == y["logits"][0, : b.n_actions].argmax())
    h.remove()
    assert all(finite_kids) and len(finite_kids) == len(pairs), "a pooled kid row is not finite"
    return {"frames": len(pairs), "decisions": n_dec, "bit_identical": exact, "max_abs_diff": worst,
            "argmax_agree": same_argmax, "first_differing_pairs": first_diff,
            "gated_ahead_contribution": "exactly 0 (gate 0, kids finite)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True, help="frames with RESET bit 1 (the migrated network's)")
    ap.add_argument("--frames-off", type=Path, help="the same moments without bit 1 (default: strip_map_in_combat)")
    ap.add_argument("--max-frames", type=int, default=2000)
    args = ap.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1", "migration is CPU-only"
    assert not args.output.exists(), "output already exists; migration never overwrites a checkpoint"
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
    check_source(before)
    after = replace(before, map_tokens="no_combat", route_summary=True, second_boss_table=True)
    ppo = dict(ck["config"])
    kick = KickstartConfig(**ppo["kickstart"]) if "kickstart" in ppo else KickstartConfig()
    kc = ck["counters"].get("kickstart")
    if kc is not None:  # v2a.md 1, 3: only after the teacher was dropped (release + fade finished)
        assert not teacher_active(kick, int(kc["u"]), kc["release_update"],
                                  None if kc.get("refade") is None else tuple(kc["refade"])), \
            f"the kickstart teacher is still active at update {kc['u']}: migrate after its fade"
    tok_off, tok_on = Tokenizer(vocab, before), Tokenizer(vocab, after)
    model_config = build(ModelConfig, extra["model_config"])
    net = SpireNet(tok_on.layout, model_config, catalog).eval()
    migrated = {key: migrate_inputs2_weights(ck[key], net, tok_off.layout, tok_on.layout) for key in ("model", "ema")}
    opt = torch.optim.AdamW(_param_groups(net, float(ppo["weight_decay"])))
    optimizer = migrate_inputs2_optimizer(ck["optimizer"], net, tok_off.layout, tok_on.layout, opt)
    opt.load_state_dict(optimizer)
    on_frames = [p for p in read_frames(args.frames) if not parse_frame(p).is_error]
    if args.frames_off is not None:
        off_frames = [p for p in read_frames(args.frames_off) if not parse_frame(p).is_error]
        assert len(off_frames) == len(on_frames), f"{len(off_frames)} frames without bit 1, {len(on_frames)} with"
    else:
        off_frames = [strip_map_in_combat(p) for p in on_frames]
    pairs_all = list(zip(off_frames, on_frames))[: args.max_frames]
    fc = before.forecast
    pairs = [(keep_trailers(a, forecast=fc, calc=True), keep_trailers(b, forecast=fc, calc=True)) for a, b in pairs_all]
    parsed = [parse_frame(b) for _, b in pairs]
    mix = {"combat": sum(int(f.glob[G.IN_COMBAT]) == 1 for f in parsed), "map": sum(f.kind == Kind.MAP for f in parsed),
           "acts": [sum(int(f.glob[G.ACT_INDEX]) == k for f in parsed) for k in range(3)],
           "second_boss_node": sum(bool(len(f.table(Table.MAP_NODES)) and (
               f.table(Table.MAP_NODES)[:, 5] & int(MapNodeFlag.SECOND_BOSS)).any()) for f in parsed)}
    checks: dict[str, Any] = {}
    n_before = 0
    for key in ("model", "ema"):
        off = SpireNet(tok_off.layout, model_config, catalog).eval()
        off.load_state_dict(ck[key])
        n_before = off.n_params
        on = SpireNet(tok_on.layout, model_config, catalog).eval()
        on.load_state_dict(migrated[key])
        checks[key] = equivalence(pairs, tok_off, tok_on, off, on)
        assert checks[key]["bit_identical"] == len(pairs), f"{key}: {checks[key]}"
    # old optimizer entries bytewise equal (ids / type_emb: their old rows); state ids enumerate the live groups
    old_opt = ck["optimizer"]
    pname = {id(q): n for n, q in net.named_parameters()}
    idx_name = dict(enumerate(pname[id(q)] for g in opt.param_groups for q in g["params"]))
    n_learned, n_types = tok_off.layout.n_learned, tok_off.layout.n_types
    n_entries = 0
    for g_old, g_new in zip(old_opt["param_groups"], optimizer["param_groups"]):
        kept = [j for j in g_new["params"] if idx_name[j] not in ADDED]
        assert len(kept) == len(g_old["params"])
        for i_old, j in zip(g_old["params"], kept):
            if i_old not in old_opt["state"]:
                continue
            for k, v in old_opt["state"][i_old].items():
                w = optimizer["state"][j][k]
                if k != "step" and idx_name[j] in ("ids.weight", "type_emb.weight"):
                    w = w[: n_learned if idx_name[j] == "ids.weight" else n_types]
                assert torch.equal(w, v), (idx_name[j], k)
                n_entries += 1
    proof = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "version": int(ck["counters"]["version"]), "samples_seen": int(ck["counters"]["samples_seen"]),
        "device": "cpu", "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
        "n_layers": model_config.n_layers, "d_model": model_config.d_model, "forecast": before.forecast,
        "parameters": [n_before, net.n_params], "additional_parameters": net.n_params - n_before,
        "optimizer": "existing steps / moments / group settings exact (remapped by name; checked: "
                     f"{n_entries} tensors equal); ids moments + the copied encounter rows' moments; type_emb moments + "
                     "a zero row; ahead_in / ahead_gate zero moments, step 0",
        "kickstart_before": asdict(kick), "kickstart_after": asdict(KickstartConfig()),
        "frames": str(args.frames.resolve()),
        "frames_off": str(args.frames_off.resolve()) if args.frames_off else "synth.strip_map_in_combat",
        "frames_used": len(pairs), "frame_mix": mix,
        "equivalence_cpu_fp32_b1": checks,
        "tokenizer_before": asdict(before), "tokenizer_after": asdict(after),
    }
    extra["tokenizer"] = asdict(after)
    assert [extra["tokenizer"][k] for k in KEYS] == ["no_combat", True, True]
    extra["vocab"], extra["catalog"] = str(args.vocab.resolve()), str(args.catalog.resolve())
    extra["inputs2_migration"] = proof
    ppo["kickstart"] = asdict(KickstartConfig())  # v2a.md 3: ppo.kickstart off (the resumed run's config has it off)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, **migrated, "config": ppo, "optimizer": optimizer, "extra": extra}, args.output)
    proof_path = args.output.with_name(args.output.name + ".proof.json")
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "proof": str(proof_path), **proof}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
