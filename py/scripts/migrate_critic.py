"""Explicit CPU critic migration (notes/critic/contract.md 5, gate PG2): add the training-only critic tower, keep
online / EMA / Adam / counters / RNG, and prove on real frames that the network's outputs are unchanged.

    CUDA_VISIBLE_DEVICES=-1 python -B scripts/migrate_critic.py --checkpoint SRC.pt --output DST.pt \\
        --vocab .../vocab.json --catalog .../catalog.json --frames FRAMES.bin [--max-frames 1000] [--critic-lr 3e-4]

``--frames``: real frames of the source's tokenizer configuration (u32 length + payload each), with or without the
HDRAW trailer (a frame with it is seen through ``synth.strip_hidden`` by ``forward``; its CRIT feeds the tower). CPU
fp32, 4 threads, B = 1, online and EMA weights: ``forward`` of the source and of the migrated network, and the old
outputs of the migrated ``forward_critic``, must be bit-identical (max |diff| 0, arg-max equal). The source may have
any depth and any rules mode. The proof goes to DST.pt.proof.json and extra["critic_migration"].
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

from spire.critic import CriticConfig, empty_crit, optimizer_groups, side
from spire.critic_migration import PREFIX, migrate_critic_optimizer, migrate_critic_weights
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet, model_config_dict
from spire.proto import parse_frame
from spire.schema import Kind
from spire.synth import strip_hidden
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
    assert pos == len(data), "truncated frames file"


@torch.no_grad()
def equivalence(frames: list[bytes], tok: Tokenizer, off: SpireNet, on: SpireNet, draw_k: int) -> dict[str, Any]:
    """Per output: max |difference| of the migrated ``forward`` and ``forward_critic`` against the source's ``forward``
    (CPU fp32, B = 1); frames bit-identical in all five outputs; greedy arg-max agreement; tower outputs finite."""
    worst = {k: 0.0 for k in OUTPUT_KEYS}
    worst_c = {k: 0.0 for k in OUTPUT_KEYS}
    exact = exact_c = same_argmax = n_dec = finite_vc = 0
    for p in frames:
        f = parse_frame(p)
        plain = parse_frame(strip_hidden(p)) if f.hd_rows >= 0 else f
        enc = tok.encode_batch((plain,))
        s = enc.sample(0)
        batch = tok.collate([s])
        crit = side(f, int(enc.card_base[0]), s.n_tokens, draw_k) if f.hd_rows >= 0 else empty_crit(1, draw_k)[0]
        x, y = off(batch), on(batch)
        z = on.forward_critic(batch, torch.from_numpy(crit[None]))
        ok = ok_c = True
        for k in OUTPUT_KEYS:
            fin = torch.isfinite(x[k])
            assert torch.equal(fin, torch.isfinite(y[k])) and torch.equal(fin, torch.isfinite(z[k])), k
            if fin.any():
                worst[k] = max(worst[k], float((x[k][fin] - y[k][fin]).abs().max()))
                worst_c[k] = max(worst_c[k], float((x[k][fin] - z[k][fin]).abs().max()))
            ok &= torch.equal(x[k], y[k])
            ok_c &= torch.equal(x[k], z[k])
        exact += ok
        exact_c += ok_c
        finite_vc += bool(torch.isfinite(z["vc"]).all())
        if s.n_actions:
            n_dec += 1
            same_argmax += int(x["logits"][0, : s.n_actions].argmax() == y["logits"][0, : s.n_actions].argmax()
                               == z["logits"][0, : s.n_actions].argmax())
    return {"frames": len(frames), "decisions": n_dec, "bit_identical_forward": exact,
            "bit_identical_forward_critic": exact_c, "max_abs_diff_forward": worst,
            "max_abs_diff_forward_critic": worst_c, "argmax_agree": same_argmax, "vc_finite": finite_vc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=1000)
    ap.add_argument("--critic-lr", type=float, default=3e-4, help="ppo.critic.lr of the tower's optimizer groups")
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
    tcfg = build(TokenizerConfig, extra["tokenizer"])
    tok = Tokenizer(vocab, tcfg)
    before: ModelConfig = build(ModelConfig, extra["model_config"])
    assert before.critic is None, "already migrated"
    after = replace(before, critic=CriticConfig())
    net = SpireNet(tok.layout, after, catalog).eval()
    migrated = {key: migrate_critic_weights(ck[key], net) for key in ("model", "ema")}
    opt = torch.optim.AdamW(optimizer_groups(net, float(ck["config"]["weight_decay"]), args.critic_lr))
    optimizer = migrate_critic_optimizer(ck["optimizer"], net, opt)
    opt.load_state_dict(optimizer)
    old_opt = ck["optimizer"]
    frames = [p for p in read_frames(args.frames) if not parse_frame(p).is_error][: args.max_frames]
    assert len(frames) == args.max_frames, f"{args.frames}: only {len(frames)} frames"
    n_combat = sum(parse_frame(p).kind == Kind.COMBAT for p in frames)
    n_hidden = sum(parse_frame(p).hd_rows > 0 for p in frames)
    assert n_combat >= 300, f"frame mix: {n_combat} combat frames"
    draw_k = after.critic.draw_k  # type: ignore[union-attr]
    checks: dict[str, Any] = {}
    n_before = 0
    for key in ("model", "ema"):
        off = SpireNet(tok.layout, before, catalog).eval()
        off.load_state_dict(ck[key])
        n_before = off.n_params
        on = SpireNet(tok.layout, after, catalog).eval()
        on.load_state_dict(migrated[key])
        checks[key] = equivalence(frames, tok, off, on, draw_k)
        c = checks[key]
        assert c["bit_identical_forward"] == c["bit_identical_forward_critic"] == len(frames), (key, c)
        assert max(c["max_abs_diff_forward"].values()) == 0 == max(c["max_abs_diff_forward_critic"].values()), key
        assert c["argmax_agree"] == c["decisions"] and c["vc_finite"] == len(frames), key
        for name, v in ck[key].items():  # every old key bytewise equal in the output
            assert torch.equal(migrated[key][name], v), (key, name)
    # old optimizer entries bytewise equal; the tower's zero, step 0
    n_entries = n_zero = 0
    for i, entry in old_opt["state"].items():
        for k, v in entry.items():
            assert torch.equal(optimizer["state"][i][k], v), (i, k)
            n_entries += 1
    for g in optimizer["param_groups"][len(old_opt["param_groups"]):]:
        assert g["critic"] and g["lr"] == args.critic_lr
        for i in g["params"]:
            assert all(not bool(v.any()) for v in optimizer["state"][i].values())
            n_zero += 1
    for g_old, g_new in zip(old_opt["param_groups"], optimizer["param_groups"]):
        assert {k: v for k, v in g_old.items() if k != "params"} == {k: v for k, v in g_new.items() if k != "params"}
    n_tower = sum(v.numel() for k, v in migrated["model"].items() if k.startswith(PREFIX))
    proof = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "version": int(ck["counters"]["version"]), "samples_seen": int(ck["counters"]["samples_seen"]),
        "device": "cpu", "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
        "n_layers": before.n_layers, "d_model": before.d_model, "rules_mode": tcfg.rules_mode,
        "parameters": [n_before, net.n_params], "tower_parameters": n_tower,
        "optimizer": f"saved groups and entries exact ({n_entries} tensors equal); tower: "
                     f"{len(optimizer['param_groups']) - len(old_opt['param_groups'])} own groups (lr "
                     f"{args.critic_lr}), {n_zero} parameters with zero moments, step 0",
        "frames": str(args.frames.resolve()), "frames_used": len(frames), "combat_frames": n_combat,
        "frames_with_hdraw_rows": n_hidden,
        "equivalence_cpu_fp32_b1": checks,
        "model_config_before": model_config_dict(before), "model_config_after": model_config_dict(after),
        "tokenizer": asdict(tcfg),
    }
    extra["model_config"] = model_config_dict(after)
    extra["vocab"], extra["catalog"] = str(args.vocab.resolve()), str(args.catalog.resolve())
    extra["critic_migration"] = proof
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, **migrated, "optimizer": optimizer, "extra": extra}, args.output)
    proof_path = args.output.with_name(args.output.name + ".proof.json")
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "proof": str(proof_path), **proof}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
