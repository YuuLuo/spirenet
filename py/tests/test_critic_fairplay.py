"""Fair play of the critic tower (notes/critic/contract.md 6, FP1-FP4; build-v1.md 4): the draw order reaches the
tower and nothing else. Synthetic flag-on frames (tests/critic_data.py); the real frames of the host engineer go
through the same functions (scripts/critic_gates.py)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

import critic_data as cd
from spire import tokenize_ref
from spire.critic import CriticConfig, empty_crit, side, training_reset_flags
from spire.cycle import CycleBan
from spire.history import PublicHistory
from spire.model import OUTPUT_KEYS, GraphRunner, ModelConfig, SpireNet, packed_layout, packed_views
from spire.proto import Frame, parse_frame
from spire.schema import G, ResetFlag, reset_flags_for
from spire.synth import FakeHost, FakeHostConfig, strip_hidden
from spire.tokenize import Sample, Tokenizer

SMALL = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32,
                    critic=CriticConfig())
SAMPLE_ARRAYS = ("types", "raw", "rel", "edges", "cond", "history")
ROOT = Path(__file__).resolve().parents[1]


def _same_sample(a: Sample, b: Sample) -> bool:
    return a.counts == b.counts and all(np.array_equal(getattr(a, k), getattr(b, k)) for k in SAMPLE_ARRAYS)


# =========================================================================== FP1 policy inputs


def fp1_check(episodes: list[list[bytes]], actions: list[list[int]], config: str, rules: str | None) -> dict:
    """Every Sample array (native, numpy, tokenize_ref), the collated batch (legal mask included), the cycle-ban /
    removal-rule state and the public history record are bit-identical for each flag-on frame and its stripped twin;
    the actor's per-decision sequence (observe, record) runs on both streams."""
    on_n, off_n = cd.tokenizers(config, rules, native=True)
    on_p, off_p = cd.tokenizers(config, rules, native=False)
    assert on_n.native and not on_p.native
    out = {"frames": 0, "fight_frames": 0, "hidden_rows": 0, "decisions": 0}
    for frames, acts in zip(episodes, actions):
        hist = [PublicHistory(on_n.width), PublicHistory(on_n.width)]
        cyc = [CycleBan(cd.vocab(), commit_select=True, removal=cd.rule(), mode=rules or "mask") for _ in range(2)]
        for p, a in zip(frames, acts):
            f_on, f_off = parse_frame(p), parse_frame(strip_hidden(p))
            assert f_on.hdraw is not None and f_off.hdraw is None
            out["frames"] += 1
            out["fight_frames"] += int(f_on.gvals[G.IN_COMBAT] == 1)
            out["hidden_rows"] += f_on.hd_rows
            enc = [on_n.encode_batch((f_on,)), off_n.encode_batch((f_off,)), on_p.encode_batch((f_on,)),
                   off_p.encode_batch((f_off,))]
            assert all(np.array_equal(e.card_base, enc[0].card_base) for e in enc)
            s = [e.sample(0) for e in enc]
            refs = [tokenize_ref.encode_reference(on_n, f_on), tokenize_ref.encode_reference(off_n, f_off)]
            assert all(_same_sample(s[0], x) for x in s[1:] + refs), f"frame {out['frames']}: samples differ"
            rows = [h.observe(f) for h, f in zip(hist, (f_on, f_off))]
            assert np.array_equal(rows[0], rows[1])
            st = [c.observe(f, x) for c, f, x in zip(cyc, (f_on, f_off), s[:2])]
            assert st[0] == st[1] and cyc[0].active_bans == cyc[1].active_bans and cyc[0].rule_row == cyc[1].rule_row
            assert np.array_equal(s[0].history, s[1].history)
            b0, b1 = on_n.collate([s[0]]), off_n.collate([s[1]])
            assert b0.keys() == b1.keys() and all(torch.equal(b0[k], b1[k]) for k in b0), "collated batch differs"
            if a >= 0:
                out["decisions"] += 1
                for c, f in zip(cyc, (f_on, f_off)):
                    c.record(f, a)
                rec = [h.record(f, x, a) for h, f, x in zip(hist, (f_on, f_off), s[:2])]
                assert rec[0] == rec[1]
    return out


@pytest.mark.parametrize("config", cd.CONFIGS)
def test_fp1_policy_inputs(config: str) -> None:
    st = cd.stream(config)
    r = fp1_check(st.episodes, st.actions, config, "teach")
    assert r["frames"] >= cd.N_FRAMES and r["fight_frames"] >= cd.N_FIGHT and r["hidden_rows"] > 0, r


def test_fp1_strict_count_refuses_the_table() -> None:
    """Every tool's tokenizer (allow_hidden False) refuses a flag-on frame; the training tokenizer refuses a frame
    without the table (the count is strict both ways: native, numpy and the reference)."""
    p = cd.stream("calc").frames[1]
    for native in (True, False):
        on, off = cd.tokenizers("calc", native=native)
        with pytest.raises(AssertionError, match="N_TABLES"):
            off.encode_batch((parse_frame(p),))
        with pytest.raises(AssertionError, match="N_TABLES"):
            on.encode_batch((parse_frame(strip_hidden(p)),))
    on, off = cd.tokenizers("calc")
    with pytest.raises(AssertionError, match="HDRAW trailer iff"):
        tokenize_ref.encode_reference(off, parse_frame(p))


# =========================================================================== FP2 policy outputs


def _permuted(crit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = crit.copy()
    n = int(crit[1])
    out[2 : 2 + n] = crit[2 : 2 + n][rng.permutation(n)]
    return out


def _crits(frames: list[Frame], enc, draw_k: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    true = np.stack([side(f, int(enc.card_base[i]), int(enc.counts[i, 0]), draw_k) for i, f in enumerate(frames)])
    perm = np.stack([_permuted(c, rng) for c in true])
    none = empty_crit(len(frames), draw_k)
    none[:, 0] = true[:, 0]
    return true, perm, none


def _net(tok: Tokenizer, seed: int = 3) -> SpireNet:
    torch.manual_seed(seed)
    net = SpireNet(tok.layout, SMALL).eval()
    assert net.critic is not None
    with torch.no_grad():  # a "trained" tower: every weight non-zero (the head's last layer is 0 at birth)
        g = torch.Generator().manual_seed(seed)
        for p in net.critic.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return net


def fp2_cpu(episodes: list[list[bytes]], actions: list[list[int]], config: str, rules: str | None,
            seed: int = 11) -> dict:
    """CPU fp32, B = 1: logits and value of (a) forward on the stripped frame, (b) forward_critic with the true CRIT,
    (c) with CRIT's rows permuted, (d) with CRIT = no rows: bit-identical; sampled actions under a fixed seed equal;
    vc of (b) and (c) differ where the permutation changes the token sequence."""
    on, off = cd.tokenizers(config, rules)
    net = _net(on)
    rng = np.random.default_rng(seed)
    k = SMALL.critic.draw_k  # type: ignore[union-attr]
    out = {"frames": 0, "fight_frames": 0, "exact": 0, "sampled_equal": 0, "perm_changes_sequence": 0,
           "vc_differs_where_changed": 0}
    steps = [(i == 0, p, a) for e, x in zip(episodes, actions) for i, (p, a) in enumerate(zip(e, x))]
    hist = PublicHistory(on.width)
    cyc = CycleBan(cd.vocab(), commit_select=True, removal=cd.rule(), mode=rules or "mask")
    with torch.no_grad():
        for first, p, a in steps:
            if first:  # a new episode: fresh trackers, as the actor's env after a RESET
                hist = PublicHistory(on.width)
                cyc = CycleBan(cd.vocab(), commit_select=True, removal=cd.rule(), mode=rules or "mask")
            f_on, f_off = parse_frame(p), parse_frame(strip_hidden(p))
            enc_on, enc_off = on.encode_batch((f_on,)), off.encode_batch((f_off,))
            s_on, s_off = enc_on.sample(0), enc_off.sample(0)
            if on.config.public_history:
                s_on.history[:] = s_off.history[:] = hist.observe(f_off)
            if on.config.cycle_ban:
                cyc.observe(f_off, s_off)
                s_on.history[:] = s_off.history
            true, perm, none = _crits([f_on], enc_on, k, rng)
            b_on, b_off = on.collate([s_on]), off.collate([s_off])
            ref = net(b_off)
            outs = [net.forward_critic(b_on, torch.from_numpy(c)) for c in (true, perm, none)]
            ok = all(torch.equal(ref[key], o[key]) for o in outs for key in ref)
            out["exact"] += ok
            out["frames"] += 1
            fight = f_on.gvals[G.IN_COMBAT] == 1
            out["fight_frames"] += int(fight)
            if s_off.n_actions:
                acts = []
                for o in [ref, *outs]:
                    gen = torch.Generator().manual_seed(1234)
                    pr = torch.softmax(o["logits"][0, : s_off.n_actions], 0)
                    acts.append(int(torch.multinomial(pr, 1, generator=gen)))
                out["sampled_equal"] += len(set(acts)) == 1
                if on.config.cycle_ban:
                    cyc.record(f_off, a)
                if on.config.public_history:
                    hist.record(f_off, s_off, a)
            else:
                out["sampled_equal"] += 1
            if fight and not np.array_equal(true, perm):
                out["perm_changes_sequence"] += 1
                out["vc_differs_where_changed"] += int(not torch.equal(outs[0]["vc"], outs[1]["vc"]))
    return out


FP2_CASES = (("calc", None), ("calc", "mask"), ("calc", "teach"), ("calc", "free"), ("v2a", "mask"))


@pytest.mark.parametrize("config,rules", FP2_CASES)
def test_fp2_policy_outputs_cpu(config: str, rules: str | None) -> None:
    st = cd.stream(config)
    r = fp2_cpu(st.episodes, st.actions, config, rules)
    assert r["exact"] == r["frames"] >= cd.N_FRAMES and r["sampled_equal"] == r["frames"], r
    assert r["fight_frames"] >= cd.N_FIGHT
    assert r["vc_differs_where_changed"] >= 0.9 * r["perm_changes_sequence"] > 0, r


def fp2_gpu(frames: list[bytes], config: str, rules: str | None, batch: int = 64, seed: int = 5,
            compiled: bool = False) -> dict:
    """The actor's graph path on the GPU in bf16 at B = 64 (packed, the same graph, three CRIT inputs) against the
    plain graph on the stripped frames: logits and value bit-identical; sampled actions equal. ``compiled`` (tree p7):
    both graphs replay torch.compile'd entries, as TorchPolicy(compile_model=True) builds them; ``exact_crit`` counts
    the frames whose logits and value are bit-identical across the three CRIT inputs of the same graph."""
    on, off = cd.tokenizers(config, rules)
    net = _net(on).cuda().eval()
    k = SMALL.critic.draw_k  # type: ignore[union-attr]
    if compiled:
        r_crit = GraphRunner(net, on, critic=True, critic_forward=torch.compile(net.forward_critic, dynamic=True))
        r_plain = GraphRunner(torch.compile(net, dynamic=True).eval(), off)  # type: ignore[arg-type]
    else:
        r_crit, r_plain = GraphRunner(net, on, critic=True), GraphRunner(net, off)
    rng = np.random.default_rng(seed)
    out = {"batches": 0, "frames": 0, "exact": 0, "exact_crit": 0, "sampled_equal": 0, "max_abs_vs_plain": 0.0}
    for lo in range(0, len(frames) - batch + 1, batch):
        ps = frames[lo : lo + batch]
        f_on = [parse_frame(p) for p in ps]
        f_off = [parse_frame(strip_hidden(p)) for p in ps]
        e_on, e_off = on.encode_batch(f_on), off.encode_batch(f_off)
        crits = _crits(f_on, e_on, k, rng)
        nat = on.shape_encoded(e_on, 0, batch)
        shape = r_crit.pad_shape(nat)
        assert shape == r_plain.pad_shape(off.shape_encoded(e_off, 0, batch))

        def staged(tok: Tokenizer, enc, draw_k: int, crit: np.ndarray | None) -> torch.Tensor:
            layout, size = packed_layout(shape, tok.width, draw_k)
            buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            views = {kk: v.numpy() for kk, v in packed_views(buf, layout).items()}
            tok.collate_encoded_into(enc, 0, batch, shape, {kk: v for kk, v in views.items() if kk != "crit"},
                                     nat=nat)
            if crit is not None:
                views["crit"][:] = empty_crit(shape.batch, draw_k)
                views["crit"][:batch] = crit
            return buf

        plain = r_plain.run_packed(staged(off, e_off, 0, None), shape, batch).clone()
        res = [r_crit.run_packed(staged(on, e_on, k, c), shape, batch).clone() for c in crits]
        A = shape.actions
        for i in range(batch):
            ok = all(torch.equal(plain[i, : A + 1], x[i, : A + 1]) for x in res)
            out["exact"] += ok
            out["exact_crit"] += all(torch.equal(res[0][i, : A + 1], x[i, : A + 1]) for x in res[1:])
            fin = torch.isfinite(plain[i, : A + 1])
            assert all(torch.equal(fin, torch.isfinite(x[i, : A + 1])) for x in res)
            out["max_abs_vs_plain"] = max(out["max_abs_vs_plain"], max(
                float((plain[i, : A + 1][fin] - x[i, : A + 1][fin]).abs().max()) for x in res))
            n = f_off[i].n_actions
            if n:
                acts = set()
                for x in [plain, *res]:
                    gen = torch.Generator(device="cuda").manual_seed(99 + i)
                    acts.add(int(torch.multinomial(torch.softmax(x[i, :n], 0), 1, generator=gen)))
                out["sampled_equal"] += len(acts) == 1
            else:
                out["sampled_equal"] += 1
        out["batches"] += 1
        out["frames"] += batch
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("config,rules", FP2_CASES)
def test_fp2_policy_outputs_gpu_graph(config: str, rules: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under deterministic algorithms: the trunk's atomic ``index_add`` (pooling) makes two replays of the SAME graph on
    the same input differ in a few rows otherwise (measured: 3 of 8,832 rows, |d| 1.2e-4), whatever CRIT holds."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    st = cd.stream(config)
    torch.use_deterministic_algorithms(True)
    try:
        r = fp2_gpu(st.frames, config, rules)
    finally:
        torch.use_deterministic_algorithms(False)
    assert r["exact"] == r["frames"] == r["sampled_equal"] and r["batches"] >= 40, r


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("config,rules", FP2_CASES)
def test_fp2_policy_outputs_gpu_compiled_graph(config: str, rules: str | None, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture) -> None:
    """Tree p7, FP2 on the COMPILED actor graph path, under deterministic algorithms: the same compiled critic graph
    with the three CRIT inputs gives bit-identical logits and value and the same sampled actions, and these equal the
    plain compiled graph's on the stripped frames bit for bit (measured: 2,944 of 2,944 frames per case)."""
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    st = cd.stream(config)
    torch.use_deterministic_algorithms(True)
    try:
        r = fp2_gpu(st.frames, config, rules, compiled=True)
    finally:
        torch.use_deterministic_algorithms(False)
    with capsys.disabled():
        print(f"\nFP2-COMPILED {config} {rules}: {r}")
    assert r["exact_crit"] == r["exact"] == r["frames"] == r["sampled_equal"] and r["batches"] >= 40, r
    assert r["max_abs_vs_plain"] == 0.0, r


# =========================================================================== FP3 nobody else can ask (Python side)


def test_fp3_reset_flags_for_cannot_produce_hidden() -> None:
    with pytest.raises(AssertionError, match="training-only"):
        reset_flags_for(True, base=int(ResetFlag.HIDDEN) | 25, calc=True)
    assert not reset_flags_for(True, calc=True, map_in_combat=True) & int(ResetFlag.HIDDEN)
    assert training_reset_flags(793) == 793 | 1024 and training_reset_flags(795) == 795 | 1024
    with pytest.raises(AssertionError):
        training_reset_flags(793 | 1024)


def test_fp3_host_refuses_hidden_without_training_hidden() -> None:
    v = cd.vocab()
    chars = [v.id("character", v["character"].entries[2])]
    with pytest.raises(AssertionError, match="training_hidden"):
        FakeHost(v, 0, FakeHostConfig()).reset(1, chars, 0, flags=cd.flags_of("calc"))
    f = FakeHost(v, 0, FakeHostConfig(), training_hidden=True).reset(1, chars, 0, flags=cd.flags_of("calc"))
    assert f.hdraw is not None
    f = FakeHost(v, 0, FakeHostConfig()).reset(1, chars, 0, flags=reset_flags_for(True, calc=True))
    assert f.hdraw is None


def test_fp3_source_scan() -> None:
    """``ResetFlag.HIDDEN``, ``training_reset_flags`` and ``training_hidden`` occur only in schema.py, critic.py,
    host.py, train.py, the fake host (synth.py: its pass-through constructor argument) and the tests."""
    allowed = {"spire/schema.py", "spire/critic.py", "spire/host.py", "spire/train.py", "spire/synth.py"}
    pat = re.compile(r"ResetFlag\.HIDDEN|training_reset_flags|training_hidden")
    hits = set()
    for p in sorted((ROOT / "spire").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if pat.search(p.read_text(encoding="utf-8")):
            hits.add(rel)
    assert hits <= allowed, sorted(hits - allowed)
    assert {"spire/schema.py", "spire/critic.py", "spire/host.py"} <= hits


# =========================================================================== FP4 isolation


def test_fp4_isolation() -> None:
    """Backward of the tower's loss alone: no trunk parameter gets a gradient. Backward of the policy, entropy, public
    value and aux losses alone: no ``critic.*`` parameter gets a gradient."""
    on, _ = cd.tokenizers("calc", "teach")
    torch.manual_seed(0)
    net = SpireNet(on.layout, SMALL)
    net.train()
    frames = [parse_frame(p) for p in cd.stream("calc").frames[:96]]
    enc = on.encode_batch(frames)
    batch = on.collate(enc.samples())
    crit = torch.from_numpy(np.stack([side(f, int(enc.card_base[i]), int(enc.counts[i, 0]), 32)
                                      for i, f in enumerate(frames)]))
    tower = {id(p) for p in net.critic.parameters()}  # type: ignore[union-attr]

    def grads(loss: torch.Tensor) -> tuple[set[str], set[str]]:
        net.zero_grad(set_to_none=True)
        loss.backward()
        got = {n for n, p in net.named_parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0)}
        crit_names = {n for n, p in net.named_parameters() if id(p) in tower}
        return got & crit_names, got - crit_names

    o = net.forward_critic(batch, crit)
    ret = torch.rand(o["vc"].shape[0], 3)
    l_crit = (torch.nn.functional.binary_cross_entropy_with_logits(o["win_logit_c"], ret[:, 0])
              + 0.5 * ((o["vc"][:, 1] - ret[:, 1]) ** 2).mean() + 0.5 * ((o["vc"][:, 2] - ret[:, 2]) ** 2).mean())
    c, t = grads(l_crit)
    assert c and not t, sorted(t)[:5]
    o = net.forward_critic(batch, crit)
    lg = o["logits"]
    fin = torch.isfinite(lg)
    logp = torch.log_softmax(lg.masked_fill(~fin, -1e9), 1)
    l_pub = (-(logp * fin).sum() + (logp.exp() * logp).masked_fill(~fin, 0).sum() + (o["value"] ** 2).mean()
             + o["pwin_logit"].mean() + o["floors_pred"].mean() + o["hp_pred"].mean())
    l_pub = l_pub + o["teach_removal"].sum()  # the teach term of rules mode "teach"
    c, t = grads(l_pub)
    assert t and not c, sorted(c)[:5]
