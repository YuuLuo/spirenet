"""PG1 / FP1 / FP2 of the critic contract (notes/critic/contract.md 6) on recorded flag-on frames of a real host
(``--training-hidden``, RESET bit 10): the same checks as tests/test_critic_wire.py and tests/test_critic_fairplay.py.

    CUDA_VISIBLE_DEVICES=-1 python -B scripts/critic_gates.py --checkpoint CRITIC.pt --vocab V --catalog C \\
        --removal-ratings R --frames A.frames [B.frames ...] --out gates.json [--fp2-frames 1000]

``--frames``: length-prefixed payloads with a ``<name>.cmd`` beside each (the command stream: RESET / STEP), which
gives the action taken after each OBS frame. ``--checkpoint``: a network with ``model.critic`` (scripts/
migrate_critic.py) whose tokenizer configuration the frames carry; its tower is perturbed (seeded) for FP2's
"the tower reads the rows" count, the trunk is the checkpoint's. PG1: every frame parses with ``verify_hashes``, the
HDRAW invariants hold, ``strip_hidden`` = ``keep_trailers`` without HDRAW, card_base native = numpy, CRIT = the
independent reference. FP1: Samples (native, numpy, tokenize_ref), the collated batch, the cycle-ban / removal-rule
state and the public history record bit-identical for each frame and its stripped twin along the recorded actions.
FP2 (CPU fp32, B = 1, the first ``--fp2-frames``): logits / value of forward(stripped) = forward_critic with the true,
permuted and empty CRIT; sampled actions equal; vc differs where the permutation changes the token sequence.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from spire import tokenize_ref  # noqa: E402
from spire.critic import empty_crit, side, truncated  # noqa: E402
from spire.cycle import CycleBan  # noqa: E402
from spire.history import PublicHistory  # noqa: E402
from spire.model import ModelConfig, SpireNet  # noqa: E402
from spire.proto import ResetCmd, StepCmd, decode_command, hdraw_problems, parse_frame  # noqa: E402
from spire.removal import load_rule  # noqa: E402
from spire.schema import G, Table  # noqa: E402
from spire.synth import keep_trailers, strip_hidden  # noqa: E402
from spire.tokenize import Ent, Tokenizer, TokenizerConfig  # noqa: E402
from spire.train import build  # noqa: E402
from spire.vocab import Catalog, Vocab  # noqa: E402

SAMPLE_ARRAYS = ("types", "raw", "rel", "edges", "cond", "history")


def read_messages(path: Path) -> list[bytes]:
    data = path.read_bytes()
    out, pos = [], 0
    while pos < len(data):
        n = int.from_bytes(data[pos : pos + 4], "little")
        out.append(data[pos + 4 : pos + 4 + n])
        pos += 4 + n
    assert pos == len(data), f"{path}: truncated"
    return out


def episodes_of(frames_path: Path) -> list[tuple[list[bytes], list[int]]]:
    """(frames, action after each frame; -1 after a TERMINAL) per episode, from the frames and the command stream."""
    frames = read_messages(frames_path)
    cmds = [decode_command(c) for c in read_messages(frames_path.with_suffix(".cmd"))]
    cmds = [c for c in cmds if isinstance(c, (ResetCmd, StepCmd))]
    assert isinstance(cmds[0], ResetCmd)
    eps: list[tuple[list[bytes], list[int]]] = []
    fi = 0
    for i, c in enumerate(cmds):
        if isinstance(c, ResetCmd):
            eps.append(([], []))
        nxt = cmds[i + 1] if i + 1 < len(cmds) else None
        p = frames[fi]
        fi += 1
        f = parse_frame(p)
        eps[-1][0].append(p)
        if isinstance(nxt, StepCmd):
            assert f.is_obs and nxt.step_id == f.step_id, (frames_path, fi)
            eps[-1][1].append(int(nxt.action_index))
        else:
            assert f.is_terminal, (frames_path, fi, f)
            eps[-1][1].append(-1)
    assert fi == len(frames), (frames_path, fi, len(frames))
    return eps


def crit_reference(f, s, k: int) -> np.ndarray:
    card_tokens = [i for i in range(s.n_tokens) if int(s.types[i]) == int(Ent.CARD)]
    out = [int(f.gvals[G.KIND]), 0] + [-1] * k
    j = 0
    for seat, pos, owner in sorted(map(tuple, f.hdraw.tolist())):
        if pos >= k or j == k:
            continue
        out[2 + j] = card_tokens[owner]
        j += 1
    out[1] = j
    return np.array(out, np.int16)


def same_sample(a, b) -> bool:
    return a.counts == b.counts and all(np.array_equal(getattr(a, k), getattr(b, k)) for k in SAMPLE_ARRAYS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--removal-ratings", type=Path, required=True)
    ap.add_argument("--frames", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fp2-frames", type=int, default=1000)
    args = ap.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    extra = ck["extra"]
    vocab = Vocab.load(args.vocab)
    assert vocab.hash == extra["vocab_hash"]
    catalog = Catalog.load(args.catalog, vocab)
    tcfg = dataclasses.replace(build(TokenizerConfig, extra["tokenizer"]), removal_ratings=str(args.removal_ratings))
    mcfg = build(ModelConfig, extra["model_config"])
    assert mcfg.critic is not None, "--checkpoint must have model.critic"
    k = mcfg.critic.draw_k
    toks = {}
    for native in (True, False):
        on, off = Tokenizer(vocab, tcfg, native=native), Tokenizer(vocab, tcfg, native=native)
        on.allow_hidden = True
        assert on.native == native
        toks[native] = (on, off)
    rule = load_rule(vocab, args.removal_ratings) if tcfg.removal_rule else None
    on, off = toks[True]
    t0 = time.time()
    pg1 = {"frames": 0, "fight_frames": 0, "hidden_rows": 0, "max_rows": 0, "truncated_rows": 0, "crit_equal": 0,
           "invariants_ok": 0, "strip_equal": 0, "card_base_equal": 0}
    fp1 = {"frames": 0, "decisions": 0, "samples_equal": 0, "collated_equal": 0, "trackers_equal": 0,
           "tracker_restarts": 0, "impl_agree": 0}
    per_file = {}
    all_steps = []
    for path in args.frames:
        eps = episodes_of(path)
        n0 = pg1["frames"]
        for frames, acts in eps:
            hist = [PublicHistory(on.width), PublicHistory(on.width)]
            cyc = [CycleBan(vocab, commit_select=tcfg.commit_select, removal=rule, mode=tcfg.rules_mode)
                   for _ in range(2)]
            for i, (p, a) in enumerate(zip(frames, acts)):
                all_steps.append((i == 0, p, a))
                f = parse_frame(p, verify_hashes=True)
                assert f.hdraw is not None
                g = parse_frame(strip_hidden(p), verify_hashes=True)
                pg1["frames"] += 1
                pg1["fight_frames"] += int(f.gvals[G.IN_COMBAT] == 1)
                pg1["hidden_rows"] += f.hd_rows
                pg1["max_rows"] = max(pg1["max_rows"], f.hd_rows)
                pg1["invariants_ok"] += hdraw_problems(f) == []
                fc, cc = f.forecast is not None, f.cc_rows is not None
                pg1["strip_equal"] += (keep_trailers(p, forecast=fc, calc=cc) == strip_hidden(p)
                                       and keep_trailers(p, forecast=fc, calc=cc, hidden=True) == p)
                enc = [toks[True][0].encode_batch((f,)), toks[True][1].encode_batch((g,)),
                       toks[False][0].encode_batch((f,)), toks[False][1].encode_batch((g,))]
                pg1["card_base_equal"] += all(int(e.card_base[0]) == 2 + f.table(Table.SEATS).shape[0] for e in enc)
                s = [e.sample(0) for e in enc]
                c = side(f, int(enc[0].card_base[0]), int(enc[0].counts[0, 0]), k)
                pg1["crit_equal"] += np.array_equal(c, crit_reference(f, s[0], k))
                pg1["truncated_rows"] += truncated(f, c)
                refs = [tokenize_ref.encode_reference(on, f), tokenize_ref.encode_reference(off, g)]
                # fair play: per implementation, the flag-on frame and its stripped twin give the same Sample (before the trackers write the
                # ban slots of the native pair)
                ok_s = same_sample(s[0], s[1]) and same_sample(s[2], s[3]) and same_sample(refs[0], refs[1])
                fp1["impl_agree"] += same_sample(s[0], s[2]) and same_sample(s[0], refs[0])  # reported, not FP1
                rows = [h.observe(x) for h, x in zip(hist, (f, g))]
                st = [cy.observe(x, y) for cy, x, y in zip(cyc, (f, g), s[:2])]
                ok_t = (np.array_equal(rows[0], rows[1]) and st[0] == st[1] and cyc[0].active_bans == cyc[1].active_bans
                        and cyc[0].rule_row == cyc[1].rule_row)
                b0, b1 = on.collate([s[0]]), off.collate([s[1]])
                ok_b = b0.keys() == b1.keys() and all(torch.equal(b0[kk], b1[kk]) for kk in b0)
                if a >= 0:
                    fp1["decisions"] += 1
                    for cy, x in zip(cyc, (f, g)):
                        if a in cy.active_bans:  # the recording policy ignored the ban: both twins restart the tracker
                            cy.clear()
                            fp1["tracker_restarts"] += 1
                        else:
                            cy.record(x, a)
                    rec = [h.record(x, y, a) for h, x, y in zip(hist, (f, g), s[:2])]
                    ok_t &= rec[0] == rec[1]
                fp1["frames"] += 1
                fp1["samples_equal"] += ok_s
                fp1["collated_equal"] += ok_b
                fp1["trackers_equal"] += ok_t
        per_file[path.name] = pg1["frames"] - n0
    pg1["seconds"] = fp1["seconds"] = round(time.time() - t0, 1)
    # FP2 on the checkpoint's network (trunk as trained, tower perturbed with a fixed seed)
    net = SpireNet(on.layout, mcfg, catalog).eval()
    net.load_state_dict(ck["ema"])
    with torch.no_grad():
        gen = torch.Generator().manual_seed(17)
        for q in net.critic.parameters():  # type: ignore[union-attr]
            q.add_(torch.randn(q.shape, generator=gen) * 0.05)
    rng = np.random.default_rng(5)
    fp2 = {"frames": 0, "fight_frames": 0, "exact": 0, "sampled_equal": 0, "perm_changes_sequence": 0,
           "vc_differs_where_changed": 0}
    t1 = time.time()
    hist0 = PublicHistory(on.width)
    cyc0 = CycleBan(vocab, commit_select=tcfg.commit_select, removal=rule, mode=tcfg.rules_mode)
    with torch.no_grad():
        for first, p, a in all_steps[: args.fp2_frames]:
            if first:
                hist0 = PublicHistory(on.width)
                cyc0 = CycleBan(vocab, commit_select=tcfg.commit_select, removal=rule, mode=tcfg.rules_mode)
            f, g = parse_frame(p), parse_frame(strip_hidden(p))
            e_on, e_off = on.encode_batch((f,)), off.encode_batch((g,))
            s_on, s_off = e_on.sample(0), e_off.sample(0)
            if tcfg.public_history:
                s_on.history[:] = s_off.history[:] = hist0.observe(g)
            if tcfg.cycle_ban:
                cyc0.observe(g, s_off)
                s_on.history[:] = s_off.history
            true = side(f, int(e_on.card_base[0]), int(e_on.counts[0, 0]), k)
            perm = true.copy()
            n = int(true[1])
            perm[2 : 2 + n] = true[2 : 2 + n][rng.permutation(n)]
            none = empty_crit(1, k)[0]
            none[0] = true[0]
            b_on, b_off = on.collate([s_on]), off.collate([s_off])
            ref = net(b_off)
            outs = [net.forward_critic(b_on, torch.from_numpy(c[None])) for c in (true, perm, none)]
            fp2["exact"] += all(torch.equal(ref[kk], o[kk]) for o in outs for kk in ref)
            fp2["frames"] += 1
            fight = f.gvals[G.IN_COMBAT] == 1
            fp2["fight_frames"] += int(fight)
            if s_off.n_actions:
                acts = set()
                for o in [ref, *outs]:
                    gg = torch.Generator().manual_seed(1234)
                    acts.add(int(torch.multinomial(torch.softmax(o["logits"][0, : s_off.n_actions], 0), 1,
                                                   generator=gg)))
                fp2["sampled_equal"] += len(acts) == 1
            else:
                fp2["sampled_equal"] += 1
            if a >= 0:
                if tcfg.cycle_ban and a in cyc0.active_bans:
                    cyc0.clear()
                elif tcfg.cycle_ban:
                    cyc0.record(g, a)
                if tcfg.public_history:
                    hist0.record(g, s_off, a)
            if fight and not np.array_equal(true, perm):
                fp2["perm_changes_sequence"] += 1
                fp2["vc_differs_where_changed"] += int(not torch.equal(outs[0]["vc"], outs[1]["vc"]))
    fp2["seconds"] = round(time.time() - t1, 1)
    res = {"checkpoint": str(args.checkpoint), "tokenizer": dataclasses.asdict(tcfg), "frames_per_file": per_file,
           "PG1": pg1, "FP1": fp1, "FP2_cpu_fp32_b1": fp2}
    n = pg1["frames"]
    res["PG1_pass"] = (pg1["crit_equal"] == pg1["invariants_ok"] == pg1["strip_equal"] == pg1["card_base_equal"] == n)
    res["FP1_pass"] = fp1["samples_equal"] == fp1["collated_equal"] == fp1["trackers_equal"] == fp1["frames"] == n
    res["FP2_pass"] = (fp2["exact"] == fp2["sampled_equal"] == fp2["frames"]
                       and fp2["vc_differs_where_changed"] >= 0.9 * fp2["perm_changes_sequence"])
    args.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return 0 if res["PG1_pass"] and res["FP1_pass"] and res["FP2_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
