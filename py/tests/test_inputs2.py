"""Inputs v2a (notes/inputs2/contract.md, notes/inputs2/v2a.md): the map_tokens / route_summary / second_boss_table
keys of the three tokenizers (DROP rule, assertion A1, the AHEAD row, the encounter#second table), the model's zero-gated
AHEAD input, the exact migration and the run-config asserts, on synthetic frames."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spire import tokenize as T
from spire.ahead_ref import AheadRules, ahead_reference, route_reference
from spire.inputs2_migration import migrate_inputs2_weights
from spire.learner import Learner, PPOConfig, _param_groups
from spire.model import AHEAD_INIT_SEED, OUTPUT_KEYS, ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.schema import (CALC_N_COLS, CALC_TABLES, DECISION_KINDS, DIR_STRIDE, GLOBAL_LEN, HEADER_LEN, N_COLS,
                          N_TABLES, ActionCol, G, H, Kind, MapNodeCol, MapNodeFlag, MapNodeState, OptionCol, RefCol,
                          ResetFlag, Table, reset_flags_for)
from spire.synth import FrameBuilder, FrameGen, strip_map_in_combat
from spire.tokenize import AHEAD_SLOTS, Ent, Tokenizer, TokenizerConfig
from spire.tokenize_ref import encode_reference
from spire.train import TrainConfig, build, check_forecast_flags
from spire.vocab import Catalog, Vocab

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16)
ALL3 = dict(map_tokens="no_combat", route_summary=True, second_boss_table=True)
G5 = [{}, {"map_tokens": "no_combat"}, ALL3, {"second_boss_table": True}]  # the four configs of gate G5 (v2a.md 4)


def _frames(gen: FrameGen, rng: np.random.Generator, n: int, kinds=(Kind.COMBAT,), *, size: str = "typical",
            forecast: bool = False, bit1: bool = True, seats: int = 4) -> list[bytes]:
    out = []
    for i in range(n):
        ctx = gen.random_ctx(rng, int(rng.integers(1, seats + 1)), size=size)  # type: ignore[arg-type]
        ctx.reset_flags = reset_flags_for(forecast, ctx.reset_flags, calc=True, map_in_combat=bit1)
        ctx.enemies = tuple(int(m) for m in gen.rids(rng, "monster", int(rng.integers(1, 6))))
        out.append(gen.frame(rng, int(kinds[i % len(kinds)]), ctx, size=size))  # type: ignore[arg-type]
    return out


def _kinds() -> tuple[int, ...]:
    return (*DECISION_KINDS, Kind.COMBAT, Kind.COMBAT, Kind.CARD_SELECT)


def _toks(vocab: Vocab, cfg: TokenizerConfig) -> list[Tokenizer]:
    assert T._native is not None, "build the native tokenizer: python -B native/build.py"
    toks = [Tokenizer(vocab, cfg, native=False), Tokenizer(vocab, cfg)]
    assert toks[1].native and not toks[0].native
    return toks


def _same(a: T.Sample, b: T.Sample) -> bool:
    return all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("types", "raw", "rel", "edges", "cond")) \
        and a.counts == b.counts


def _rules(tmp_path: Path, vocab: Vocab) -> AheadRules:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"categories": {"enum:MapPointType": {"entries": list(vocab["enum:MapPointType"].entries)}}}))
    return AheadRules(path)


def _with_route(p: bytes, route: np.ndarray) -> bytes:
    """Frame ``p`` with its ROUTE trailer table replaced by ``route`` (the chain re-addressed)."""
    f = parse_frame(p)
    w = f.words
    x0 = HEADER_LEN + GLOBAL_LEN + N_TABLES * DIR_STRIDE + int(sum(a.size for a in f.tables))
    parts = [w[:x0].copy()]
    pos = x0
    trailers = [] if f.forecast is None else [(16, f.forecast)]
    trailers += [(tid, route if tid == CALC_TABLES[2] else f.calc_table(k)) for k, tid in enumerate(CALC_TABLES)]
    for tid, a in trailers:
        a = np.asarray(a, "<i4").reshape(-1, CALC_N_COLS[tid - 17] if tid >= 17 else 17)
        parts.append(np.array([tid, a.shape[0], a.shape[1], (pos + DIR_STRIDE) * 4, 0], "<i4"))
        parts.append(a.ravel())
        pos += DIR_STRIDE + a.size
    out = np.concatenate(parts).astype("<i4")
    out[H.FRAME_BYTES] = out.nbytes
    return out.tobytes()


# --------------------------------------------------------------------------- config

def test_config_validation() -> None:
    TokenizerConfig(map_tokens="no_combat")
    TokenizerConfig(second_boss_table=True)
    TokenizerConfig(calc=True, **ALL3)
    with pytest.raises(AssertionError):
        TokenizerConfig(map_tokens="decision")  # not built (v2a.md 1)
    with pytest.raises(AssertionError):
        TokenizerConfig(map_tokens="none")
    for bad in ({"route_summary": True, "map_tokens": "no_combat"}, {"route_summary": True, "calc": True},
                {"route_summary": True, "calc": True, "map_tokens": "no_combat", "pool_children": False}):
        with pytest.raises(AssertionError):
            TokenizerConfig(**bad)


def test_reset_flags_for() -> None:
    base = int(ResetFlag.AUTO_SINGLE | ResetFlag.UPGRADE_PREVIEWS | ResetFlag.TARGET_PREVIEWS)
    assert base == 25
    assert reset_flags_for(True, base, calc=True, map_in_combat=True) == 795
    assert reset_flags_for(False, base, calc=True, map_in_combat=True) == 539
    assert reset_flags_for(True, 795, calc=True) == 793 and reset_flags_for(True, base, calc=True) == 793


def test_run_flag_asserts() -> None:
    def cfg(flags: int, **tok: object) -> TrainConfig:
        return build(TrainConfig, {"run_name": "t", "actor": {"reset_flags": flags}, "tokenizer": {"calc": True, "forecast": True, **tok}})
    check_forecast_flags(cfg(795, **ALL3))
    check_forecast_flags(cfg(793))
    check_forecast_flags(cfg(793, map_tokens="no_combat", second_boss_table=True))
    for flags, tok in ((793, ALL3), (795, {}), (795, {"map_tokens": "no_combat"})):
        with pytest.raises(AssertionError):
            check_forecast_flags(cfg(flags, **tok))


# --------------------------------------------------------------------------- layout

def test_second_boss_layout(vocab: Vocab) -> None:
    off = T.TokenLayout(vocab, calc=True)
    on = T.TokenLayout(vocab, calc=True, second_boss_table=True)
    rows = vocab.embedding_rows("encounter")
    assert on.regions["encounter#second"] == (off.n_learned, rows)
    assert {k: v for k, v in on.regions.items() if k != "encounter#second"} == off.regions
    assert on.n_learned == off.n_learned + rows and on.n_rows == off.n_rows + rows
    assert all(on.static[c][0] == off.static[c][0] + rows and on.static[c][1:] == off.static[c][1:] for c in off.static)
    assert [i.table for i in on.specs[Ent.GLOBAL].ids if i.src == G.SECOND_BOSS_ENCOUNTER] == ["encounter#second"]
    assert [i.table for i in off.specs[Ent.GLOBAL].ids if i.src == G.SECOND_BOSS_ENCOUNTER] == ["encounter"]
    assert on.width == off.width and on.n_wide == off.n_wide and on.n_types == off.n_types == 24


def test_ahead_layout(vocab: Vocab) -> None:
    calc = T.TokenLayout(vocab, forecast=True, calc=True)
    ah = T.TokenLayout(vocab, forecast=True, calc=True, route_summary=True)
    assert ah.ahead_cols == (calc.calc_cols[1], calc.calc_cols[1] + 35) == (calc.n_wide, ah.n_wide)
    assert ah.width == calc.width >= AHEAD_SLOTS and ah.n_types == 25 and calc.n_types == 24
    assert ah.slot_cols[Ent.AHEAD].tolist() == list(range(30)) and ah.regions == calc.regions
    assert T.history_width(ah.width) == T.history_width(calc.width)


# --------------------------------------------------------------------------- tokenizers (G5, G2b, G3a, G4)

@pytest.mark.parametrize("size", ["typical", "extreme"])
@pytest.mark.parametrize("forecast", [False, True])
def test_native_numpy_reference_identical(vocab: Vocab, gen: FrameGen, size: str, forecast: bool) -> None:
    """G5: native == numpy == tokenize_ref for the four configs, on frames with RESET bit 1 (and, for the configs
    without the route summary, also without it); sorted batches included."""
    rng = np.random.default_rng(31)
    with_bit = [parse_frame(p) for p in _frames(gen, rng, 120, _kinds(), size=size, forecast=forecast)]
    without = [parse_frame(p) for p in _frames(gen, rng, 60, _kinds(), size=size, forecast=forecast, bit1=False)]
    for keys in G5:
        cfg = TokenizerConfig(forecast=forecast, calc=True, **keys)  # type: ignore[arg-type]
        for frames in (with_bit,) if cfg.route_summary else (with_bit, without):
            toks = _toks(vocab, cfg)
            for sort in (False, True):
                encs = [t.encode_batch(frames, sort=sort) for t in toks]
                for k in ("types", "raw", "rel", "edges", "cond", "counts", "row0", "edge0", "act0", "order"):
                    a, b = getattr(encs[0], k), getattr(encs[1], k)
                    assert (a is None and b is None) or np.array_equal(a, b), (keys, sort, k)
            enc = encs[0]
            for i, f in enumerate(frames):
                j = int(enc.order[i]) if enc.order is not None else i
                assert _same(encode_reference(toks[0], frames[j]), enc.sample(i)), (keys, i)


def test_g5_counts(vocab: Vocab, gen: FrameGen) -> None:
    """Per frame (contract G5 with the teacher = "no_combat", the student = all three keys of v2a): tokens equal,
    kids(all three) = kids(no_combat) + 1; DROP frames have no MAP_NODE token, no edge, no ROUTE kid."""
    rng = np.random.default_rng(32)
    frames = [parse_frame(p) for p in _frames(gen, rng, 150, _kinds(), forecast=True)]
    t_nc = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True, map_tokens="no_combat"))
    t_all = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True, **ALL3))  # type: ignore[arg-type]
    t_frame = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True))
    n_drop = 0
    for f in frames:
        a, b, c = t_nc.encode(f), t_all.encode(f), t_frame.encode(f)
        drop = int(f.glob[G.IN_COMBAT]) == 1 and f.kind != Kind.MAP
        n_drop += drop
        assert a.counts[0] == b.counts[0] and b.counts[1] == a.counts[1] + 1 and a.counts[2:] == b.counts[2:]
        assert (b.types == int(Ent.AHEAD)).sum() == 1 and b.rel[b.types == int(Ent.AHEAD)].tolist() == [[0, -1, -1]]
        n_nodes = f.table(Table.MAP_NODES).shape[0]
        n_route = f.calc_table(2).shape[0]
        assert c.counts[0] == a.counts[0] + drop * n_nodes and c.counts[1] == a.counts[1] + drop * n_route
        if drop:
            assert not (a.types == int(Ent.MAP_NODE)).any() and a.edges.shape[0] == 0
            assert not (a.types == int(Ent.ROUTE)).any()
        else:
            assert _same(a, c)
    assert n_drop >= 30


@pytest.mark.parametrize("forecast", [False, True])
def test_drop_rule_equals_frame_without_bit1(vocab: Vocab, gen: FrameGen, forecast: bool) -> None:
    """G2b on synthetic frames: Tokenizer(T).encode(f_793) == Tokenizer(T + "no_combat").encode(f_795), with f_793 =
    the frame of the same moment without bit 1 (synth.strip_map_in_combat), in all three tokenizers."""
    rng = np.random.default_rng(33)
    raw = _frames(gen, rng, 150, _kinds(), forecast=forecast)
    on = [parse_frame(p) for p in raw]
    off = [parse_frame(strip_map_in_combat(p), verify_hashes=True) for p in raw]
    assert sum(int(f.glob[G.IN_COMBAT]) == 1 and f.table(Table.MAP_NODES).shape[0] > 0 for f in on) >= 30
    assert all(f.table(Table.MAP_NODES).shape[0] == 0 for f in off if int(f.glob[G.IN_COMBAT]) == 1)
    for extra in ({}, {"coop_features": True, "mend_target": True}):
        cfg = TokenizerConfig(forecast=forecast, calc=True, **extra)  # type: ignore[arg-type]
        for t_off, t_on in zip(_toks(vocab, cfg), _toks(vocab, replace(cfg, map_tokens="no_combat"))):
            for i, (a, b) in enumerate(zip(t_off.encode_batch(off).samples(), t_on.encode_batch(on).samples())):
                assert _same(a, b), i
            for i in range(0, len(on), 7):
                assert _same(encode_reference(t_off, off[i]), encode_reference(t_on, on[i])), i


def _combat_frame_with(gen: FrameGen, rng: np.random.Generator, kind: int) -> bytes:
    """A frame of ``kind`` with RESET bit 1 and at least one ACTIONS, OPTIONS and REFS row, made a DROP frame by
    setting GLOBAL[IN_COMBAT] = 1."""
    for _ in range(200):
        p = _frames(gen, rng, 1, (kind,), seats=1)[0]
        f = parse_frame(p)
        if all(f.table(t).shape[0] for t in (Table.ACTIONS, Table.OPTIONS, Table.REFS, Table.MAP_NODES)):
            w = np.frombuffer(p, "<i4").copy()
            w[HEADER_LEN + int(G.IN_COMBAT)] = 1
            return w.tobytes()
    raise AssertionError("no frame with ACTIONS, OPTIONS and REFS rows")


@pytest.mark.parametrize("clause", ["actions", "options", "refs"])
def test_a1_raises(vocab: Vocab, gen: FrameGen, clause: str) -> None:
    """G3a: assertion A1 raises in the numpy path, the native path and the reference, for each of its clauses."""
    rng = np.random.default_rng({"actions": 41, "options": 42, "refs": 43}[clause])
    p = _combat_frame_with(gen, rng, Kind.EVENT)
    cfg = TokenizerConfig(calc=True, map_tokens="no_combat")
    for t in _toks(vocab, cfg):  # the valid DROP frame encodes
        t.encode(parse_frame(p))
    encode_reference(Tokenizer(vocab, cfg, native=False), parse_frame(p))
    f = parse_frame(p)
    t, c = {"actions": (Table.ACTIONS, ActionCol.SRC_TABLE), "options": (Table.OPTIONS, OptionCol.REF_TABLE),
            "refs": (Table.REFS, RefCol.ENT_TABLE)}[clause]
    w = np.frombuffer(p, "<i4").copy()
    row0 = int(f.words[HEADER_LEN + GLOBAL_LEN + int(t) * DIR_STRIDE + 3]) // 4
    k = f.table(t).shape[0] - 1
    w[row0 + k * N_COLS[t] + int(c)] = int(Table.MAP_NODES)
    bad = parse_frame(w.tobytes())
    for tok in _toks(vocab, cfg):
        with pytest.raises(AssertionError, match=f"A1: frame step {int(bad.words[H.STEP_ID])}"):
            tok.encode(bad)
        with pytest.raises(AssertionError, match="A1"):
            tok.encode_batch([parse_frame(p), bad], sort=True)
    with pytest.raises(AssertionError, match=f"row {k} has"):
        encode_reference(Tokenizer(vocab, cfg, native=False), bad)
    for tok in _toks(vocab, TokenizerConfig(calc=True)):  # "frame": no DROP, no A1
        tok.encode(bad)
    m = np.frombuffer(w.tobytes(), "<i4").copy()  # a MAP decision is never a DROP frame
    m[HEADER_LEN + int(G.KIND)] = int(Kind.MAP)
    for tok in _toks(vocab, cfg):
        tok.encode(parse_frame(m.tobytes()))


def test_route_summary_needs_map_in_fights(vocab: Vocab, gen: FrameGen) -> None:
    rng = np.random.default_rng(44)
    p = _frames(gen, rng, 1, (Kind.COMBAT,), bit1=False)[0]
    cfg = TokenizerConfig(calc=True, **ALL3)  # type: ignore[arg-type]
    for t in _toks(vocab, cfg):
        with pytest.raises(AssertionError, match="no MAP_NODES rows"):
            t.encode(parse_frame(p))
    with pytest.raises(AssertionError, match="no MAP_NODES rows"):
        encode_reference(Tokenizer(vocab, cfg, native=False), parse_frame(p))


def test_ahead_hand_built(vocab: Vocab, tmp_path: Path) -> None:
    """The 30 slots and the 35 decoded features of a hand-built fight frame (contract 2.2 / 2.3, v2a.md 2)."""
    t = {n: vocab.id("enum:MapPointType", n) for n in ("Monster", "Elite", "RestSite", "Shop", "Unknown", "Boss")}
    types = ["Monster", "Elite", "RestSite", "Shop", "Unknown", "Boss", "Boss"]
    b = FrameBuilder()
    b.glob[G.IN_COMBAT], b.glob[G.KIND] = 1, int(Kind.COMBAT)
    nodes = []
    for i, name in enumerate(types):
        r = b.new_rows(int(Table.MAP_NODES), 1)[0]
        r[[MapNodeCol.ROW, MapNodeCol.COL, MapNodeCol.POINT_TYPE]] = (0, 0, 1, 1, 2, 3, 4)[i], i, t[name]
        r[MapNodeCol.STATE] = MapNodeState.REACHABLE if i < 2 else MapNodeState.NORMAL
        r[MapNodeCol.FLAGS] = {1: MapNodeFlag.QUEST, 5: MapNodeFlag.BOSS, 6: MapNodeFlag.SECOND_BOSS}.get(i, 0)
        nodes.append(r)
    b.extend(Table.MAP_NODES, nodes)
    b.extend(Table.MAP_EDGES, [(0, 2), (0, 3), (1, 3), (2, 4), (3, 4)])
    route = np.array([[0, 3, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1], [1, 3, -1, 1, -1, -1, 1, 1, 0, 1, 1, 1]])
    b.calc = (np.zeros((0, 11)), np.zeros((0, 5)), route)
    f = parse_frame(b.build())
    rules = _rules(tmp_path, vocab)
    assert route_reference(f, rules).tolist() == route.tolist()
    want = [2, 1, 1, 0, 0, 0, 0, 0, 3, 1, -1, 1, 1, 0, -1, 1, -1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
    assert ahead_reference(f, rules) == want
    cfg = TokenizerConfig(calc=True, **ALL3)  # type: ignore[arg-type]
    for tok in _toks(vocab, cfg):
        s = tok.encode(f)
        row = int(np.flatnonzero(s.types == int(Ent.AHEAD))[0])
        assert s.raw[row, :30].tolist() == want and not s.raw[row, 30:].any() and s.rel[row].tolist() == [0, -1, -1]
        assert not (s.types == int(Ent.MAP_NODE)).any() and not (s.types == int(Ent.ROUTE)).any()  # DROP frame
    s = encode_reference(Tokenizer(vocab, cfg, native=False), f)
    assert s.raw[s.types == int(Ent.AHEAD)][0, :30].tolist() == want
    L = Tokenizer(vocab, cfg).layout
    lo, hi = L.ahead_cols
    raw = torch.from_numpy(s.raw[s.types == int(Ent.AHEAD)].copy())
    _, w = T.FeatureDecoder(L)(torch.tensor([int(Ent.AHEAD)]), raw)
    assert torch.count_nonzero(w[0, :lo]) == 0
    feat = [0.5, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 3 / 16, 1 / 8, 0, 0, 1, 1 / 8, 0, 1 / 8, 0, 0, 0, 0.25, 0, 0, 0.25,
            1 / 8, 1 / 8, 0, 0.25, 0.5, 0.5, 0, 0.25, 1 / 8, 1 / 8, 0.5]
    assert np.allclose(w[0, lo:hi].numpy(), feat, atol=1e-7)
    empty = FrameBuilder()  # no ROUTE rows: N = 0, NEXT 0, slots 8-28 = -1
    empty.calc = (np.zeros((0, 11)), np.zeros((0, 5)), np.zeros((0, 12)))
    for tok in _toks(vocab, cfg):
        s = tok.encode(parse_frame(empty.build()))
        assert s.raw[s.types == int(Ent.AHEAD)][0, :30].tolist() == [0] * 8 + [-1] * 21 + [0]


def test_ahead_against_brute_force(vocab: Vocab, gen: FrameGen, tmp_path: Path) -> None:
    """G4 on synthetic frames: with the frames' ROUTE rows set to the brute-force ROUTE of their map, the AHEAD row of
    all three tokenizers equals spire.ahead_ref (which recomputes the ROUTE rows itself)."""
    rules = _rules(tmp_path, vocab)
    rng = np.random.default_rng(45)
    raw = _frames(gen, rng, 200, _kinds(), forecast=True)
    frames = [parse_frame(_with_route(p, route_reference(parse_frame(p), rules))) for p in raw]
    want = [ahead_reference(f, rules) for f in frames]
    assert {w[0] for w in want} >= {1, 2, 3} and sum(w[0] >= 3 for w in want) >= 5 and sum(w[29] > 0 for w in want)
    cfg = TokenizerConfig(forecast=True, calc=True, **ALL3)  # type: ignore[arg-type]
    for tok in _toks(vocab, cfg):
        enc = tok.encode_batch(frames)
        for i in range(len(frames)):
            s = enc.sample(i)
            assert s.raw[s.types == int(Ent.AHEAD)][0, :30].tolist() == want[i], i
    ref = Tokenizer(vocab, cfg, native=False)
    for i in range(0, len(frames), 5):
        s = encode_reference(ref, frames[i])
        assert s.raw[s.types == int(Ent.AHEAD)][0, :30].tolist() == want[i], i


# --------------------------------------------------------------------------- model

def _pair(vocab: Vocab, catalog: Catalog | None, forecast: bool = True, **flags: bool):
    old_tok = Tokenizer(vocab, TokenizerConfig(forecast=forecast, calc=True, **flags))
    new_tok = Tokenizer(vocab, replace(old_tok.config, **ALL3))  # type: ignore[arg-type]
    config = replace(TINY, seat_symmetry=True)
    torch.manual_seed(0)
    old = SpireNet(old_tok.layout, config, catalog).eval()
    new = SpireNet(new_tok.layout, config, catalog).eval()
    new.load_state_dict(migrate_inputs2_weights(old.state_dict(), new, old_tok.layout, new_tok.layout))
    return old_tok, new_tok, old, new


def test_off_model_unchanged(vocab: Vocab, catalog: Catalog) -> None:
    """A model with every key off draws nothing new (same keys, same initialisation as with map_tokens only)."""
    a_tok = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True))
    b_tok = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True, map_tokens="no_combat"))
    torch.manual_seed(5)
    a = SpireNet(a_tok.layout, TINY, catalog).state_dict()
    torch.manual_seed(5)
    b = SpireNet(b_tok.layout, TINY, catalog).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert "ahead_in.weight" not in a and "ahead_gate" not in a


def test_fresh_net_equals_migration(vocab: Vocab, catalog: Catalog) -> None:
    _, new_tok, old, new = _pair(vocab, catalog)
    fresh = SpireNet(new_tok.layout, replace(TINY, seat_symmetry=True), catalog)
    assert fresh.ahead_in is not None and new.ahead_in is not None
    assert torch.equal(fresh.ahead_in.weight, new.ahead_in.weight)
    assert torch.equal(fresh.type_emb.weight[24], new.type_emb.weight[24])
    g = torch.Generator().manual_seed(AHEAD_INIT_SEED)
    assert torch.equal(new.type_emb.weight[24], torch.empty(32).normal_(0.0, TINY.init_std, generator=g))
    assert float(fresh.ahead_gate.detach()) == 0.0 and new.ahead_in.weight.shape == (32, 35)
    sd, osd = new.state_dict(), old.state_dict()
    off, rows = new_tok.layout.regions["encounter"]
    n = osd["ids.weight"].shape[0]
    assert torch.equal(sd["ids.weight"][:n], osd["ids.weight"]) and torch.equal(sd["ids.weight"][n:],
                                                                                 osd["ids.weight"][off : off + rows])
    assert torch.equal(sd["type_emb.weight"][:24], osd["type_emb.weight"])
    for k, v in osd.items():
        if k not in ("ids.weight", "type_emb.weight"):
            assert torch.equal(sd[k], v), k


@pytest.mark.parametrize("flags", [{}, {"coop_features": True, "mend_target": True, "public_history": True}])
def test_migrated_outputs_equal(vocab: Vocab, catalog: Catalog, gen: FrameGen, flags: dict[str, bool]) -> None:
    """GM on synthetic frames: the migrated network on frames with bit 1 = the source network on the same moments
    without it; CPU fp32, B = 1, at most 4 threads; bit-identical whenever the padded kid count is unchanged."""
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 4))
    try:
        old_tok, new_tok, old, new = _pair(vocab, catalog, **flags)
        rng = np.random.default_rng(51)
        same = n_same_shape = 0
        raw = _frames(gen, rng, 80, _kinds(), forecast=True)
        for p in raw:
            a, b = old_tok.encode(parse_frame(strip_map_in_combat(p))), new_tok.encode(parse_frame(p))
            with torch.no_grad():
                x, y = old(old_tok.collate([a])), new(new_tok.collate([b]))
            exact = all(torch.equal(x[k], y[k]) for k in OUTPUT_KEYS)
            same += exact
            for k in OUTPUT_KEYS:
                fin = torch.isfinite(x[k])
                assert torch.equal(fin, torch.isfinite(y[k]))
                assert (x[k][fin] - y[k][fin]).abs().max() <= 1e-5, k
            if old_tok.shape_of([a]).kids == new_tok.shape_of([b]).kids:
                n_same_shape += 1
                assert exact
        assert n_same_shape >= len(raw) // 2 and same >= n_same_shape
    finally:
        torch.set_num_threads(threads)


def test_gate_zero_and_learnable(vocab: Vocab, catalog: Catalog, gen: FrameGen) -> None:
    _, tok, _, net = _pair(vocab, catalog)
    rng = np.random.default_rng(52)
    samples = [tok.encode(parse_frame(p)) for p in _frames(gen, rng, 16, _kinds(), forecast=True)]
    batch = tok.collate(samples)
    stripped = []
    for s in samples:  # the same samples without their AHEAD row give the same outputs (up to rounding)
        keep = s.types != int(Ent.AHEAD)
        first = int(np.flatnonzero(~keep)[0])
        rel = s.rel[keep].astype(np.int64)
        rel = np.where(rel > first, rel - 1, rel)
        stripped.append(T.Sample(s.types[keep], s.raw[keep], rel.astype(np.int16), s.edges, s.cond,
                                 (s.counts[0], s.counts[1] - 1, s.counts[2], s.counts[3]), s.history))
    out = net(batch)
    with torch.no_grad():
        ref = net(tok.collate(stripped))
    for k in OUTPUT_KEYS:
        f = torch.isfinite(ref[k])
        assert (out[k].detach()[f] - ref[k][f]).abs().max() <= 1e-5
    (out["logits"][torch.isfinite(out["logits"])].sum() + out["value"].sum()).backward()
    assert net.ahead_gate is not None and net.ahead_gate.grad is not None and float(net.ahead_gate.grad.abs()) > 0
    with torch.no_grad():
        net.ahead_gate.fill_(1.0)
        moved = net(batch)
    assert not torch.equal(moved["value"], out["value"].detach())


def test_param_groups(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True, **ALL3))  # type: ignore[arg-type]
    net = SpireNet(tok.layout, TINY)
    decay, other = _param_groups(net, 0.01)
    assert decay["params"][-1] is net.ahead_in.weight  # registered last
    assert any(p is net.ahead_gate for p in other["params"])  # 0-dim: no decay


def test_second_boss_swap(vocab: Vocab, catalog: Catalog, gen: FrameGen) -> None:
    """G7: with second_boss_table, swapping the two GLOBAL boss ids changes the decoded id rows and the value of a
    randomly initialised network on every frame with a SECOND_BOSS node; without it the outputs are bit-identical
    under the swap (the defect, documented)."""
    rng = np.random.default_rng(53)
    frames = []
    while len(frames) < 200:
        ctx = gen.random_ctx(rng, 1)
        ctx.ascension = 10
        ctx.map = gen.make_map(rng, "typical", 10)
        ctx.path = []
        gen.walk(rng, ctx, int(rng.integers(0, ctx.map.n_map_rows - 2)))
        ctx.reset_flags = reset_flags_for(True, ctx.reset_flags, calc=True, map_in_combat=True)
        p = gen.frame(rng, int(rng.choice([Kind.COMBAT, Kind.MAP, Kind.REWARD])), ctx)
        w = np.frombuffer(p, "<i4")
        boss, second = int(w[HEADER_LEN + G.BOSS_ENCOUNTER]), int(w[HEADER_LEN + G.SECOND_BOSS_ENCOUNTER])
        if second and second != boss:
            frames.append(p)
    for on in (True, False):
        tok = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True, **(ALL3 if on else {})))  # type: ignore
        torch.manual_seed(7)
        net = SpireNet(tok.layout, TINY, catalog).eval()
        for p in frames:
            f = parse_frame(p)
            assert (f.table(Table.MAP_NODES)[:, MapNodeCol.FLAGS] & int(MapNodeFlag.SECOND_BOSS)).any()
            w = np.frombuffer(p, "<i4").copy()
            w[HEADER_LEN + G.BOSS_ENCOUNTER], w[HEADER_LEN + G.SECOND_BOSS_ENCOUNTER] = (
                w[HEADER_LEN + G.SECOND_BOSS_ENCOUNTER], w[HEADER_LEN + G.BOSS_ENCOUNTER])
            a, b = tok.encode(f), tok.encode(parse_frame(w.tobytes()))
            ba, bb = tok.collate([a]), tok.collate([b])
            ia, _ = net.decoder(ba["types"], ba["raw"])
            ib, _ = net.decoder(bb["types"], bb["raw"])
            with torch.no_grad():
                va, vb = net(ba)["value"], net(bb)["value"]
            if on:
                assert not torch.equal(ia[0], ib[0]) and not torch.equal(va, vb)
            else:
                # the same ids in other slots: equal up to the order of the float sum of the id embeddings
                assert torch.equal(ia[0].sort().values, ib[0].sort().values)
                assert float((va - vb).abs().max()) <= 1e-6


# --------------------------------------------------------------------------- checkpoint migration (learner)

def test_checkpoint_migration_preserves_adam(vocab: Vocab, tmp_path: Path) -> None:
    old_tok = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True))
    new_tok = Tokenizer(vocab, replace(old_tok.config, **ALL3))  # type: ignore[arg-type]
    old = Learner(SpireNet(old_tok.layout, TINY), old_tok, PPOConfig(), device="cpu", fast=False)
    for p in old.model.parameters():
        p.grad = torch.rand_like(p)
    old.opt.step()
    before = {n: {k: v.clone() for k, v in old.opt.state[p].items()} for n, p in old.model.named_parameters()}
    src = tmp_path / "before.pt"
    old.save(src)
    new = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    with pytest.raises(RuntimeError):
        new.load(src)
    new.load(src, migrate_inputs2=True)
    off, rows = new_tok.layout.regions["encounter"]
    n_old = old_tok.layout.n_learned
    for name, p in new.model.named_parameters():
        st = new.opt.state[p]
        if name in ("ahead_in.weight", "ahead_gate"):
            assert all(torch.count_nonzero(v) == 0 for v in st.values()), name
            continue
        for k, v in st.items():
            if name == "type_emb.weight" and k != "step":
                assert torch.equal(v[:24], before[name][k]) and torch.count_nonzero(v[24:]) == 0
            elif name == "ids.weight" and k != "step":
                assert torch.equal(v[:n_old], before[name][k]) and torch.equal(v[n_old:], before[name][k][off:off + rows])
            else:
                assert torch.equal(v, before[name][k]), (name, k)
    assert float(new.model.state_dict()["ahead_gate"]) == 0.0
    tgt = tmp_path / "after.pt"
    new.save(tgt)
    resumed = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    resumed.load(tgt)
    for p in resumed.model.parameters():
        p.grad = torch.ones_like(p)
    resumed.opt.step()
    with pytest.raises(AssertionError, match="pre-inputs2"):
        new.load(tgt, migrate_inputs2=True)


def test_run_config_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """scripts/migrate_inputs2_config.py: the three keys, reset_flags 793 -> 795, kickstart off; refuses a second run."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("mi2c", Path(__file__).parents[1] / "scripts" / "migrate_inputs2_config.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    src = tmp_path / "config.json"
    src.write_text(json.dumps({"run_name": "x", "actor": {"reset_flags": 793},
                               "tokenizer": {"forecast": True, "calc": True},
                               "ppo": {"kickstart": {"teacher": "/t.pt", "phase": 1}}, "migrate_calc": False}))
    out = tmp_path / "config.inputs2.json"
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(src), "--output", str(out)])
    assert mod.main() == 0
    cfg = build(TrainConfig, json.loads(out.read_text()))
    assert (cfg.tokenizer.map_tokens, cfg.tokenizer.route_summary, cfg.tokenizer.second_boss_table) == ("no_combat", True,
                                                                                                         True)
    assert cfg.actor.reset_flags == 795 and not cfg.ppo.kickstart.on
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(out), "--output", str(tmp_path / "again.json")])
    with pytest.raises(AssertionError):
        mod.main()


def test_actor_learner_with_route_summary(vocab: Vocab) -> None:
    """A CPU actor on fake hosts with RESET flags 795 and the three keys on, then one learner update: the AHEAD rows
    reach the learner, the row carries ahead_gate (0) and the gate gets a gradient."""
    from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
    from spire.rewards import RoomIds
    from spire.rollout import RolloutConfig, RolloutStore
    from spire.synth import FakeHost, FakeHostConfig
    torch.manual_seed(0)
    tok = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True, cycle_ban=True, **ALL3))  # type: ignore[arg-type]
    learner = Learner(SpireNet(tok.layout, TINY), tok, PPOConfig(minibatch=256, micro_batch=128, epochs=1),
                      device="cpu", fast=False)
    policy = TorchPolicy(learner.model, tok, seed=3)
    segs: list = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
             for i in range(2)]
    flags = reset_flags_for(True, calc=True, map_in_combat=True)
    assert flags == 795
    actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0),)), vocab, 0),
                  ActorConfig(reset_flags=flags), store=store)
    actor.run(max_decisions=400)
    blk = T.SampleBlock.concat([s.samples for s in segs])
    assert int((blk.types == int(Ent.AHEAD)).sum()) == len(blk)
    grads: list[float] = []
    assert learner.model.ahead_gate is not None
    learner.model.ahead_gate.register_hook(lambda g: grads.append(float(g.abs())))
    m = learner.update(segs)
    assert isinstance(m["ahead_gate"], float)  # the learner row carries the gate
    assert grads and grads[0] > 0.0, grads
