"""Move forecast (notes/forecast/contract.md): trailer parsing, tokenizers, zero-gated model input, migration."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spire import tokenize as T
from spire.forecast_migration import migrate_forecast_optimizer, migrate_forecast_weights
from spire.learner import Learner, PPOConfig, _param_groups
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.schema import (DECISION_KINDS, FORECAST_N_COLS, ForecastClass, ForecastCol, H, Kind, MsgType, Result,
                          ResetFlag, SchemaError, reset_flags_for)
from spire.synth import FrameGen, strip_forecast
from spire.tokenize import Ent, Tokenizer, TokenizerConfig
from spire.tokenize_ref import encode_reference
from spire.vocab import Catalog, Vocab

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16)


def _frames(gen: FrameGen, rng: np.random.Generator, n: int, kinds=(Kind.COMBAT,), size: str = "typical") -> list[bytes]:
    out = []
    for i in range(n):
        ctx = gen.random_ctx(rng, int(rng.integers(1, 5)), size=size)  # type: ignore[arg-type]
        ctx.reset_flags = reset_flags_for(True, ctx.reset_flags)
        ctx.enemies = tuple(int(m) for m in gen.rids(rng, "monster", int(rng.integers(1, 6))))
        out.append(gen.frame(rng, int(kinds[i % len(kinds)]), ctx, size=size))  # type: ignore[arg-type]
    return out


def test_reset_flags_for() -> None:
    base = int(ResetFlag.AUTO_SINGLE | ResetFlag.TARGET_PREVIEWS)
    assert reset_flags_for(True, base) == base | 256 and reset_flags_for(False, base | 256) == base


def test_trailer_parse_hash_strip(gen: FrameGen) -> None:
    rng = np.random.default_rng(3)
    for p in _frames(gen, rng, 40, DECISION_KINDS):
        f = parse_frame(p, verify_hashes=True)
        fc = f.forecast
        assert fc is not None and fc.shape[1] == FORECAST_N_COLS
        off = parse_frame(strip_forecast(p), verify_hashes=True)
        assert off.forecast is None
        x0 = len(strip_forecast(p))
        head = bytes(p[:x0])  # [0, X0) equal except header words 7 (FRAME_BYTES) and 8 (N_TABLES)
        assert head[:28] == strip_forecast(p)[:28] and head[36:] == strip_forecast(p)[36:]
        assert len(p) == x0 + 20 + 68 * fc.shape[0]
        for t in range(16):  # every v1 table identical
            assert np.array_equal(f.tables[t], off.tables[t])
        if fc.shape[0]:
            bad = bytearray(p)
            bad[-4] ^= 1
            with pytest.raises(SchemaError):
                parse_frame(bad, verify_hashes=True)
            bad = bytearray(p)
            bad[x0 + 4 : x0 + 8] = (fc.shape[0] + 1).to_bytes(4, "little")
            with pytest.raises(SchemaError):
                parse_frame(bad)


def test_synthetic_coverage(gen: FrameGen) -> None:
    """The generator covers every class, 0-3 candidates, both slots and 1-5 enemies (what the equality tests use)."""
    rng = np.random.default_rng(5)
    rows = np.concatenate([parse_frame(p).forecast for p in _frames(gen, rng, 200)])
    assert set(rows[:, ForecastCol.CLASS].tolist()) == {int(c) for c in ForecastClass}
    assert set(rows[:, ForecastCol.N_CAND].tolist()) == {0, 1, 2, 3}
    assert set(rows[:, ForecastCol.AHEAD].tolist()) == {1, 2}
    owners = [len(set(parse_frame(p).forecast[:, 0].tolist())) for p in _frames(gen, rng, 60)]
    assert set(owners) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("size", ["typical", "extreme"])
def test_native_numpy_reference_identical(vocab: Vocab, gen: FrameGen, size: str) -> None:
    rng = np.random.default_rng(11)
    cfgs = [TokenizerConfig(forecast=True), TokenizerConfig(forecast=True, coop_features=True, mend_target=True)]
    frames = [parse_frame(p) for p in _frames(gen, rng, 120, DECISION_KINDS, size)]
    for cfg in cfgs:
        toks = [Tokenizer(vocab, cfg, native=False)] + ([Tokenizer(vocab, cfg)] if T._native is not None else [])
        encs = [t.encode_batch(frames) for t in toks]
        for e in encs[1:]:
            for k in ("types", "raw", "rel", "edges", "cond", "counts", "row0", "edge0", "act0"):
                assert np.array_equal(getattr(encs[0], k), getattr(e, k)), k
        for i, f in enumerate(frames):
            ref, got = encode_reference(toks[0], f), encs[0].sample(i)
            for k in ("types", "raw", "rel", "edges", "cond"):
                assert np.array_equal(getattr(ref, k), getattr(got, k)), (i, k)
            assert ref.counts == got.counts


def test_forecast_kids_only_addition(vocab: Vocab, gen: FrameGen) -> None:
    """forecast=True minus its FORECAST kid rows equals forecast=False on the stripped frame (rel shifted back)."""
    rng = np.random.default_rng(12)
    on, off = Tokenizer(vocab, TokenizerConfig(forecast=True)), Tokenizer(vocab, TokenizerConfig())
    for p in _frames(gen, rng, 60, DECISION_KINDS):
        a, b = on.encode(parse_frame(p)), off.encode(parse_frame(strip_forecast(p)))
        fc = a.types == Ent.FORECAST
        n_fc = int(fc.sum())
        assert n_fc == parse_frame(p).forecast.shape[0]
        assert a.counts == (b.counts[0], b.counts[1] + n_fc, b.counts[2], b.counts[3])
        first = int(np.flatnonzero(fc)[0]) if n_fc else 0
        rel = a.rel[~fc].astype(np.int64)
        rel = np.where(rel >= first + n_fc, rel - n_fc, rel) if n_fc else rel
        assert np.array_equal(a.types[~fc], b.types) and np.array_equal(a.raw[~fc], b.raw)
        assert np.array_equal(rel, b.rel) and np.array_equal(a.edges, b.edges)
        if n_fc:  # kids of their creature token
            cre = a.rel[fc, 0]
            assert (a.types[cre] == Ent.CREATURE).all() and (a.rel[fc, 1] == -1).all()


def test_strict_table_count(vocab: Vocab, gen: FrameGen) -> None:
    rng = np.random.default_rng(13)
    p = _frames(gen, rng, 1)[0]
    with pytest.raises(AssertionError):
        Tokenizer(vocab, TokenizerConfig()).encode(parse_frame(p))
    with pytest.raises(AssertionError):
        Tokenizer(vocab, TokenizerConfig(forecast=True)).encode(parse_frame(strip_forecast(p)))
    with pytest.raises(AssertionError):
        Tokenizer(vocab, TokenizerConfig(forecast=True, pool_children=False))


def test_terminal_and_noncombat_have_empty_trailer(gen: FrameGen) -> None:
    rng = np.random.default_rng(14)
    ctx = gen.random_ctx(rng, 4)
    ctx.reset_flags = reset_flags_for(True, ctx.reset_flags)
    f = parse_frame(gen.terminal(rng, ctx, Result.WIN))
    assert f.msg_type == MsgType.TERMINAL and f.forecast is not None and f.forecast.shape == (0, 17)
    f = parse_frame(gen.frame(rng, Kind.MAP, ctx))
    assert f.forecast is not None and f.forecast.shape[0] == 0
    assert int(f.words[H.N_TABLES]) == 17


def _pair(vocab: Vocab, catalog: Catalog | None, **flags: bool) -> tuple[Tokenizer, Tokenizer, SpireNet, SpireNet]:
    old_tok = Tokenizer(vocab, TokenizerConfig(**flags))
    new_tok = Tokenizer(vocab, TokenizerConfig(forecast=True, **flags))
    config = replace(TINY, seat_symmetry=True)
    torch.manual_seed(0)
    old = SpireNet(old_tok.layout, config, catalog).eval()
    new = SpireNet(new_tok.layout, config, catalog).eval()
    new.load_state_dict(migrate_forecast_weights(old.state_dict(), new, new_tok.layout))
    return old_tok, new_tok, old, new


@pytest.mark.parametrize("flags", [{}, {"coop_features": True, "mend_target": True, "public_history": True}])
def test_migrated_outputs_equal(vocab: Vocab, catalog: Catalog, gen: FrameGen, flags: dict[str, bool]) -> None:
    """CPU fp32, B = 1: bit-identical when the padded kid count is unchanged, else equal to float rounding."""
    old_tok, new_tok, old, new = _pair(vocab, catalog, **flags)
    rng = np.random.default_rng(21)
    same = 0
    frames = _frames(gen, rng, 60, DECISION_KINDS)
    for p in frames:
        a, b = old_tok.encode(parse_frame(strip_forecast(p))), new_tok.encode(parse_frame(p))
        with torch.no_grad():
            x, y = old(old_tok.collate([a])), new(new_tok.collate([b]))
        exact = all(torch.equal(x[k], y[k]) for k in OUTPUT_KEYS)
        same += exact
        for k in OUTPUT_KEYS:
            finite = torch.isfinite(x[k])
            assert torch.equal(finite, torch.isfinite(y[k]))
            assert (x[k][finite] - y[k][finite]).abs().max() <= 1e-5, k
        if old_tok.shape_of([a]).kids == new_tok.shape_of([b]).kids:
            assert exact
    assert same >= len(frames) // 2


def test_gate_contribution_is_zero_and_learnable(vocab: Vocab, catalog: Catalog, gen: FrameGen) -> None:
    _, tok, _, net = _pair(vocab, catalog)
    rng = np.random.default_rng(22)
    samples = [tok.encode(parse_frame(p)) for p in _frames(gen, rng, 8)]
    assert sum(int((s.types == Ent.FORECAST).sum()) for s in samples) > 0
    batch = tok.collate(samples)
    stripped = []
    for s in samples:  # same samples with every FORECAST row removed must give the same outputs (up to rounding)
        keep = s.types != Ent.FORECAST
        n_fc = int((~keep).sum())
        first = int(np.flatnonzero(~keep)[0]) if n_fc else 0
        rel = s.rel[keep].astype(np.int64)
        rel = np.where(rel >= first + n_fc, rel - n_fc, rel) if n_fc else rel
        stripped.append(T.Sample(s.types[keep], s.raw[keep], rel.astype(np.int16), s.edges, s.cond,
                                 (s.counts[0], s.counts[1] - n_fc, s.counts[2], s.counts[3]), s.history))
    out = net(batch)
    with torch.no_grad():
        ref = net(tok.collate(stripped))
    for k in OUTPUT_KEYS:
        f = torch.isfinite(ref[k])
        assert (out[k].detach()[f] - ref[k][f]).abs().max() <= 1e-5
    (out["logits"][torch.isfinite(out["logits"])].sum() + out["value"].sum()).backward()
    assert net.forecast_gate is not None and net.forecast_gate.grad is not None and net.forecast_gate.grad.abs() > 0
    with torch.no_grad():
        net.forecast_gate.fill_(1.0)
        moved = net(batch)
    assert not torch.equal(moved["value"], out["value"].detach())


def test_param_groups(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(forecast=True))
    net = SpireNet(tok.layout, TINY)
    decay, other = _param_groups(net, 0.01)
    assert decay["params"][-1] is net.forecast_in.weight  # registered last
    assert any(p is net.forecast_gate for p in other["params"])  # 0-dim: no decay


@pytest.mark.parametrize("amsgrad", [False, True])
def test_checkpoint_migration_preserves_adam(vocab: Vocab, tmp_path: Path, amsgrad: bool) -> None:
    old_tok = Tokenizer(vocab, TokenizerConfig(coop_features=True, mend_target=True))
    new_tok = Tokenizer(vocab, replace(old_tok.config, forecast=True))
    old = Learner(SpireNet(old_tok.layout, TINY), old_tok, PPOConfig(), device="cpu", fast=False)
    for group in old.opt.param_groups:
        group["amsgrad"] = amsgrad
    for p in old.model.parameters():
        p.grad = torch.ones_like(p)
    old.opt.step()
    before = {n: {k: v.clone() for k, v in old.opt.state[p].items()} for n, p in old.model.named_parameters()}
    src = tmp_path / "before.pt"
    old.save(src)
    new = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    with pytest.raises(RuntimeError):
        new.load(src)
    new.load(src, migrate_forecast=True)
    for name, p in new.model.named_parameters():
        st = new.opt.state[p]
        if name in ("forecast_in.weight", "forecast_gate"):
            assert all(torch.count_nonzero(v) == 0 for v in st.values()), name
            continue
        for k, v in st.items():
            if name == "type_emb.weight" and k != "step":
                assert torch.equal(v[:-1], before[name][k]) and torch.count_nonzero(v[-1]) == 0
            else:
                assert torch.equal(v, before[name][k]), (name, k)
    sd = new.model.state_dict()
    assert float(sd["forecast_gate"]) == 0.0
    assert torch.equal(sd["type_emb.weight"][Ent.FORECAST], sd["type_emb.weight"][Ent.INTENT])
    tgt = tmp_path / "after.pt"
    new.save(tgt)
    resumed = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    resumed.load(tgt)
    for p in resumed.model.parameters():
        p.grad = torch.ones_like(p)
    resumed.opt.step()
    with pytest.raises(AssertionError, match="pre-forecast"):
        new.load(tgt, migrate_forecast=True)
    opt = torch.optim.AdamW(_param_groups(new.model, 0.0))
    with pytest.raises(AssertionError):
        migrate_forecast_optimizer(new.opt.state_dict(), new.model, new_tok.layout, opt)
