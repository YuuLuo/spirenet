"""Calc features (notes/calc/contract.md): trailer chain parsing, tokenizers, zero-gated model input, migration,
and the brute-force reference on hand-built frames."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spire import tokenize as T
from spire.calc_migration import migrate_calc_optimizer, migrate_calc_weights
from spire.calc_ref import CalcRules, reference_tables
from spire.forecast_migration import migrate_forecast_weights
from spire.learner import Learner, PPOConfig, _param_groups
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.schema import (CALC_N_COLS, DECISION_KINDS, CardCol, CostFlag, CreatureCol, CreatureKind, G, H, HandFlag,
                          IntentCol, ItemZone, Kind, MapNodeCol, MapNodeFlag, MapNodeState, MsgType, PowerCol, RelicCol,
                          ResetFlag, Result, SchemaError, SeatCol, Table, VarCol, Zone, reset_flags_for)
from spire.synth import FrameBuilder, FrameGen, strip_calc, strip_forecast
from spire.tokenize import CALC_ENTS, Ent, Tokenizer, TokenizerConfig
from spire.tokenize_ref import encode_reference
from spire.vocab import Catalog, Vocab

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16)
COMBOS = [(False, True), (True, True)]  # (forecast, calc)


def _frames(gen: FrameGen, rng: np.random.Generator, n: int, kinds=(Kind.COMBAT,), size: str = "typical",
            forecast: bool = False) -> list[bytes]:
    out = []
    for i in range(n):
        ctx = gen.random_ctx(rng, int(rng.integers(1, 5)), size=size)  # type: ignore[arg-type]
        ctx.reset_flags = reset_flags_for(forecast, ctx.reset_flags, calc=True)
        ctx.enemies = tuple(int(m) for m in gen.rids(rng, "monster", int(rng.integers(1, 6))))
        out.append(gen.frame(rng, int(kinds[i % len(kinds)]), ctx, size=size))  # type: ignore[arg-type]
    return out


def test_reset_flags_for() -> None:
    base = int(ResetFlag.AUTO_SINGLE | ResetFlag.TARGET_PREVIEWS)
    assert reset_flags_for(False, base, calc=True) == base | 512
    assert reset_flags_for(True, base, calc=True) == base | 768
    assert reset_flags_for(True, base | 512) == base | 256
    with pytest.raises(AssertionError):
        reset_flags_for(False, int(ResetFlag.AUTO_SINGLE), calc=True)


@pytest.mark.parametrize("forecast", [False, True])
def test_trailer_parse_hash_strip(gen: FrameGen, forecast: bool) -> None:
    rng = np.random.default_rng(3)
    for p in _frames(gen, rng, 40, (*DECISION_KINDS, Kind.MAP), forecast=forecast):
        f = parse_frame(p, verify_hashes=True)
        assert f.cc_rows is not None and (f.forecast is not None) == forecast
        assert int(f.words[H.N_TABLES]) == 16 + forecast + 3
        off_b = strip_calc(p)
        off = parse_frame(off_b, verify_hashes=True)
        assert off.cc_rows is None and off.turn is None and (off.forecast is not None) == forecast
        x1 = len(off_b)  # [0, X1) equal except header words 7 (FRAME_BYTES) and 8 (N_TABLES)
        assert p[:28] == off_b[:28] and p[36:x1] == off_b[36:]
        assert len(p) == x1 + 60 + sum(4 * n * c for n, c in zip(f.cc_rows, CALC_N_COLS))
        for t in range(16):
            assert np.array_equal(f.tables[t], off.tables[t])
        if forecast:
            assert np.array_equal(f.forecast, off.forecast)
            assert parse_frame(strip_forecast(off_b)).forecast is None
        for k, a in enumerate((f.turn, f.foe, f.route)):
            assert a is not None and a.shape[1] == CALC_N_COLS[k]
        if f.route.shape[0]:
            bad = bytearray(p)
            bad[-4] ^= 1
            with pytest.raises(SchemaError):
                parse_frame(bad, verify_hashes=True)
        bad = bytearray(p)
        bad[x1 + 4 : x1 + 8] = (f.cc_rows[0] + 1).to_bytes(4, "little")  # TURN row count
        with pytest.raises(SchemaError):
            parse_frame(bad)
        bad = bytearray(p)
        bad[x1 : x1 + 4] = (18).to_bytes(4, "little")  # wrong table id
        with pytest.raises(SchemaError):
            parse_frame(bad)


def test_synthetic_coverage(gen: FrameGen) -> None:
    rng = np.random.default_rng(5)
    fr = [parse_frame(p) for p in _frames(gen, rng, 200, (Kind.COMBAT, Kind.MAP))]
    assert sum(f.turn.shape[0] for f in fr) >= 25 and sum(f.foe.shape[0] for f in fr) >= 50
    assert sum(f.route.shape[0] for f in fr) >= 50
    assert {f.foe.shape[0] for f in fr} >= {0, 1, 2, 3}


@pytest.mark.parametrize("size", ["typical", "extreme"])
@pytest.mark.parametrize("forecast", [False, True])
def test_native_numpy_reference_identical(vocab: Vocab, gen: FrameGen, size: str, forecast: bool) -> None:
    rng = np.random.default_rng(11)
    cfgs = [TokenizerConfig(forecast=forecast, calc=True),
            TokenizerConfig(forecast=forecast, calc=True, coop_features=True, mend_target=True)]
    frames = [parse_frame(p) for p in _frames(gen, rng, 120, (*DECISION_KINDS, Kind.MAP), size, forecast)]
    for cfg in cfgs:
        toks = [Tokenizer(vocab, cfg, native=False)] + ([Tokenizer(vocab, cfg)] if T._native is not None else [])
        assert T._native is not None and toks[-1].native
        encs = [t.encode_batch(frames) for t in toks]
        for e in encs[1:]:
            for k in ("types", "raw", "rel", "edges", "cond", "counts", "row0", "edge0", "act0"):
                assert np.array_equal(getattr(encs[0], k), getattr(e, k)), k
        for i, f in enumerate(frames):
            ref, got = encode_reference(toks[0], f), encs[0].sample(i)
            for k in ("types", "raw", "rel", "edges", "cond"):
                assert np.array_equal(getattr(ref, k), getattr(got, k)), (i, k)
            assert ref.counts == got.counts


@pytest.mark.parametrize("forecast", [False, True])
def test_calc_kids_only_addition(vocab: Vocab, gen: FrameGen, forecast: bool) -> None:
    """calc=True minus its calc kid rows equals calc=False on strip_calc(frame) (rel shifted back); owners right."""
    rng = np.random.default_rng(12)
    on = Tokenizer(vocab, TokenizerConfig(forecast=forecast, calc=True))
    off = Tokenizer(vocab, TokenizerConfig(forecast=forecast))
    for p in _frames(gen, rng, 60, (*DECISION_KINDS, Kind.MAP), forecast=forecast):
        f = parse_frame(p)
        a, b = on.encode(f), off.encode(parse_frame(strip_calc(p)))
        cc = np.isin(a.types, [int(e) for e in CALC_ENTS])
        n_cc = int(cc.sum())
        assert n_cc == sum(f.cc_rows)
        assert a.counts == (b.counts[0], b.counts[1] + n_cc, b.counts[2], b.counts[3])
        first = int(np.flatnonzero(cc)[0]) if n_cc else 0
        assert not n_cc or cc[first : first + n_cc].all()  # one contiguous block
        rel = a.rel[~cc].astype(np.int64)
        rel = np.where(rel >= first + n_cc, rel - n_cc, rel) if n_cc else rel
        assert np.array_equal(a.types[~cc], b.types) and np.array_equal(a.raw[~cc], b.raw)
        assert np.array_equal(rel, b.rel) and np.array_equal(a.edges, b.edges)
        for e, owner, tab in ((Ent.TURN, Ent.CREATURE, f.turn), (Ent.FOE, Ent.CREATURE, f.foe),
                              (Ent.ROUTE, Ent.MAP_NODE, f.route)):
            m = a.types == e
            assert int(m.sum()) == tab.shape[0]
            if m.any():
                assert (a.types[a.rel[m, 0]] == owner).all() and (a.rel[m, 1] == -1).all()
                first_owner = int(np.flatnonzero(a.types == owner)[0])
                assert np.array_equal(a.rel[m, 0] - first_owner, tab[:, 0])


def test_feature_values(vocab: Vocab) -> None:
    """Decoded numeric features of hand-built TURN / FOE / ROUTE rows (contract 3.2)."""
    tok = Tokenizer(vocab, TokenizerConfig(calc=True))
    L = tok.layout
    lo, hi = L.calc_cols
    assert hi - lo == 40 and L.n_wide == hi and L.n_types == 24
    dec = T.FeatureDecoder(L)

    def wide(e: Ent, row: list[int]) -> np.ndarray:
        raw = np.zeros((1, tok.width), np.int16)
        cols = L.slot_cols[e]
        raw[0, : len(cols)] = np.array(row)[cols]
        _, w = dec(torch.tensor([int(e)]), torch.from_numpy(raw))
        assert torch.count_nonzero(w[0, :lo]) == 0  # no legacy column
        return w[0, lo:hi].numpy()

    sl = lambda x: float(np.sign(x) * np.log1p(abs(x)))  # noqa: E731
    w = wide(Ent.TURN, [0, 40, 5, 30, 4, 2, 1, 24, 10, 13, 0b10101])
    want = [sl(24), sl(13), sl(30), sl(5), sl(10), sl(4), sl(2), sl(1), 24 / 40, 13 / 40, sl(25), sl(11), 1, 0, 1, 0, 1]
    assert np.allclose(w[:17], want, atol=1e-6) and not w[17:].any()
    w = wide(Ent.FOE, [3, -1, 12, 7, 0b00100])
    assert np.allclose(w[17:27], [0, 1, sl(12), sl(7), 0, 0, 0, 1, 0, 0], atol=1e-6)
    w = wide(Ent.FOE, [3, 20, 12, 7, 0b00001])
    assert np.allclose(w[17:27], [sl(20), 0, sl(12), sl(7), sl(8), 1, 0, 0, 0, 0], atol=1e-6)
    w = wide(Ent.ROUTE, [5, 8, -1, 2, -1, -1, 1, 4, 0, 1, 2, 3])
    assert np.allclose(w[27:], [0.5, 0, 1, 0.25, 0, 0, 0, 0.25, 0.5, 0, 0.5, 0.5, 0.375], atol=1e-6)


def test_strict_table_count(vocab: Vocab, gen: FrameGen) -> None:
    rng = np.random.default_rng(13)
    p = _frames(gen, rng, 1)[0]
    for cfg in (TokenizerConfig(), TokenizerConfig(forecast=True)):
        for native in (False, True):
            with pytest.raises(AssertionError):
                Tokenizer(vocab, cfg, native=native).encode(parse_frame(p))
    for native in (False, True):
        with pytest.raises(AssertionError):
            Tokenizer(vocab, TokenizerConfig(calc=True), native=native).encode(parse_frame(strip_calc(p)))
        with pytest.raises(AssertionError):
            Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True), native=native).encode(parse_frame(p))
    with pytest.raises(AssertionError):
        Tokenizer(vocab, TokenizerConfig(calc=True, pool_children=False))


def test_terminal_has_empty_tables(gen: FrameGen) -> None:
    rng = np.random.default_rng(14)
    ctx = gen.random_ctx(rng, 4)
    ctx.reset_flags = reset_flags_for(True, ctx.reset_flags, calc=True)
    f = parse_frame(gen.terminal(rng, ctx, Result.WIN))
    assert f.msg_type == MsgType.TERMINAL and f.cc_rows is not None and f.turn.shape == (0, 11)
    assert int(f.words[H.N_TABLES]) == 20


def _pair(vocab: Vocab, catalog: Catalog | None, forecast: bool = False, **flags: bool):
    old_tok = Tokenizer(vocab, TokenizerConfig(forecast=forecast, **flags))
    new_tok = Tokenizer(vocab, TokenizerConfig(forecast=forecast, calc=True, **flags))
    config = replace(TINY, seat_symmetry=True)
    torch.manual_seed(0)
    old = SpireNet(old_tok.layout, config, catalog).eval()
    new = SpireNet(new_tok.layout, config, catalog).eval()
    new.load_state_dict(migrate_calc_weights(old.state_dict(), new, new_tok.layout))
    return old_tok, new_tok, old, new


@pytest.mark.parametrize("forecast,flags", [(False, {}), (True, {}),
                                            (True, {"coop_features": True, "mend_target": True,
                                                    "public_history": True})])
def test_migrated_outputs_equal(vocab: Vocab, catalog: Catalog, gen: FrameGen, forecast: bool,
                                flags: dict[str, bool]) -> None:
    """CPU fp32, B = 1: bit-identical when the padded kid count is unchanged, else equal to float rounding. At most 4
    intra-op threads: with dozens, MKL's sgemm splits rows by count, so a row's bits depend on the number of rows."""
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 4))
    try:
        _migrated_outputs_equal(vocab, catalog, gen, forecast, flags)
    finally:
        torch.set_num_threads(threads)


def _migrated_outputs_equal(vocab: Vocab, catalog: Catalog, gen: FrameGen, forecast: bool,
                            flags: dict[str, bool]) -> None:
    old_tok, new_tok, old, new = _pair(vocab, catalog, forecast, **flags)
    rng = np.random.default_rng(21)
    same = 0
    frames = _frames(gen, rng, 60, (*DECISION_KINDS, Kind.MAP), forecast=forecast)
    for p in frames:
        a, b = old_tok.encode(parse_frame(strip_calc(p))), new_tok.encode(parse_frame(p))
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


def test_fresh_calc_net_equals_migration(vocab: Vocab, catalog: Catalog) -> None:
    """A calc net draws calc_in and the new type rows from the fixed generator, exactly as the migration does."""
    _, new_tok, _, new = _pair(vocab, catalog, True)
    fresh = SpireNet(new_tok.layout, replace(TINY, seat_symmetry=True), catalog)
    assert torch.equal(fresh.calc_in.weight, new.calc_in.weight)
    assert torch.equal(fresh.type_emb.weight[21:], new.type_emb.weight[21:])
    assert float(fresh.calc_gate.detach().abs().sum()) == 0.0


def test_gate_contribution_is_zero_and_learnable(vocab: Vocab, catalog: Catalog, gen: FrameGen) -> None:
    _, tok, _, net = _pair(vocab, catalog, True)
    rng = np.random.default_rng(22)
    samples = [tok.encode(parse_frame(p)) for p in _frames(gen, rng, 12, (Kind.COMBAT, Kind.MAP), forecast=True)]
    for e in CALC_ENTS:
        assert sum(int((s.types == e).sum()) for s in samples) > 0, e
    batch = tok.collate(samples)
    stripped = []
    cc_ids = [int(e) for e in CALC_ENTS]
    for s in samples:  # same samples with every calc row removed must give the same outputs (up to rounding)
        keep = ~np.isin(s.types, cc_ids)
        n_cc = int((~keep).sum())
        first = int(np.flatnonzero(~keep)[0]) if n_cc else 0
        rel = s.rel[keep].astype(np.int64)
        rel = np.where(rel >= first + n_cc, rel - n_cc, rel) if n_cc else rel
        stripped.append(T.Sample(s.types[keep], s.raw[keep], rel.astype(np.int16), s.edges, s.cond,
                                 (s.counts[0], s.counts[1] - n_cc, s.counts[2], s.counts[3]), s.history))
    out = net(batch)
    with torch.no_grad():
        ref = net(tok.collate(stripped))
    for k in OUTPUT_KEYS:
        f = torch.isfinite(ref[k])
        assert (out[k].detach()[f] - ref[k][f]).abs().max() <= 1e-5
    (out["logits"][torch.isfinite(out["logits"])].sum() + out["value"].sum()).backward()
    assert net.calc_gate is not None and net.calc_gate.grad is not None
    assert (net.calc_gate.grad.abs() > 0).all(), net.calc_gate.grad
    for k in range(3):
        with torch.no_grad():
            net.calc_gate.zero_()
            net.calc_gate[k] = 1.0
            moved = net(batch)
        assert not torch.equal(moved["value"], out["value"].detach()), k


def test_param_groups(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(forecast=True, calc=True))
    net = SpireNet(tok.layout, TINY)
    decay, other = _param_groups(net, 0.01)
    assert decay["params"][-1] is net.calc_in.weight  # registered last
    assert any(p is net.calc_gate for p in other["params"])  # 1-dim: no decay


@pytest.mark.parametrize("forecast", [False, True])
def test_checkpoint_migration_preserves_adam(vocab: Vocab, tmp_path: Path, forecast: bool) -> None:
    old_tok = Tokenizer(vocab, TokenizerConfig(coop_features=True, mend_target=True, forecast=forecast))
    new_tok = Tokenizer(vocab, replace(old_tok.config, calc=True))
    old = Learner(SpireNet(old_tok.layout, TINY), old_tok, PPOConfig(), device="cpu", fast=False)
    for p in old.model.parameters():
        p.grad = torch.ones_like(p)
    old.opt.step()
    before = {n: {k: v.clone() for k, v in old.opt.state[p].items()} for n, p in old.model.named_parameters()}
    src = tmp_path / "before.pt"
    old.save(src)
    new = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    with pytest.raises(RuntimeError):
        new.load(src)
    new.load(src, migrate_calc=True)
    n_old = 21 if forecast else 20
    for name, p in new.model.named_parameters():
        st = new.opt.state[p]
        if name in ("calc_in.weight", "calc_gate"):
            assert all(torch.count_nonzero(v) == 0 for v in st.values()), name
            continue
        for k, v in st.items():
            if name == "type_emb.weight" and k != "step":
                assert torch.equal(v[:n_old], before[name][k]) and torch.count_nonzero(v[n_old:]) == 0
            else:
                assert torch.equal(v, before[name][k]), (name, k)
    sd, osd = new.model.state_dict(), old.model.state_dict()
    assert float(sd["calc_gate"].abs().sum()) == 0.0 and sd["calc_gate"].shape == (3,)
    assert torch.equal(sd["type_emb.weight"][:n_old], osd["type_emb.weight"])
    for k, v in osd.items():
        if k != "type_emb.weight":
            assert torch.equal(sd[k], v), k
    tgt = tmp_path / "after.pt"
    new.save(tgt)
    resumed = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    resumed.load(tgt)
    for p in resumed.model.parameters():
        p.grad = torch.ones_like(p)
    resumed.opt.step()
    with pytest.raises(AssertionError, match="pre-calc"):
        new.load(tgt, migrate_calc=True)
    opt = torch.optim.AdamW(_param_groups(new.model, 0.0))
    with pytest.raises(AssertionError):
        migrate_calc_optimizer(new.opt.state_dict(), new.model, new_tok.layout, opt)
    if not forecast:  # forecast first, then calc
        fc_tok = Tokenizer(vocab, replace(new_tok.config, forecast=True))
        with pytest.raises(AssertionError):
            migrate_forecast_weights(new.model.state_dict(), SpireNet(fc_tok.layout, TINY), fc_tok.layout)


# --------------------------------------------------------------------------- reference on hand-built frames

_POWERS = ["PLATING_POWER", "CONSTRICT_POWER", "BUFFER_POWER", "INTANGIBLE_POWER", "HARDENED_SHELL_POWER",
           "SLIPPERY_POWER", "SKITTISH_POWER", "STRENGTH_POWER"]
_RELICS = ["ORICHALCUM", "TUNGSTEN_ROD", "BEATING_REMNANT", "ANCHOR"]
_VARS = ["Damage", "CalculatedDamage", "HitCount", "CalculatedHits", "Block", "CalculatedBlock", "HpLoss"]
_CARDS = {  # name: (target_type, has_turn_end_in_hand_effect)
    "STRIKE": ("AnyEnemy", False), "DEFEND": ("Self", False), "CLEAVE": ("AllEnemies", False),
    "WHIRL": ("AllEnemies", False), "BURN": ("None", True), "DECAY": ("None", True), "SHIELD_ALLY": ("AnyAlly", False),
    "BOLT": ("RandomEnemy", False)}


def _rules(tmp_path: Path) -> tuple[CalcRules, dict[str, dict[str, int]]]:
    pad = ["<PAD>", "<UNK>"]
    cats = {"power": pad + _POWERS, "relic": pad + _RELICS, "var_name": pad + _VARS, "card": pad + list(_CARDS),
            "enum:IntentType": pad + ["Attack", "Buff", "Hidden", "Unknown"],
            "enum:CombatSide": pad + ["None", "Player", "Enemy"],
            "enum:MapPointType": pad + ["Unknown", "Shop", "Treasure", "RestSite", "Monster", "Elite", "Boss"]}
    (tmp_path / "vocab.json").write_text(json.dumps({"categories": {k: {"entries": v} for k, v in cats.items()}}))
    card = {str(i + 2): [{"target_type": tt, "has_turn_end_in_hand_effect": te}]
            for i, (tt, te) in enumerate(_CARDS.values())}
    relic = {str(cats["relic"].index("ORICHALCUM")): {"vars": {"Block": {"base": 6}}}}
    (tmp_path / "catalog.json").write_text(json.dumps({"categories": {"card": card, "relic": relic}}))
    ids = {k: {n: i for i, n in enumerate(v)} for k, v in cats.items()}
    return CalcRules(tmp_path / "vocab.json", tmp_path / "catalog.json"), ids


def _row(b: FrameBuilder, t: Table, **cols: int) -> np.ndarray:
    r = b.new_rows(int(t), 1)[0]
    for k, v in cols.items():
        r[int(k) if k.isdigit() else int(_COLS[t][k])] = v
    return r


_COLS = {Table.SEATS: SeatCol, Table.CREATURES: CreatureCol, Table.INTENTS: IntentCol, Table.POWERS: PowerCol,
         Table.RELICS: RelicCol, Table.CARDS: CardCol, Table.VARS: VarCol, Table.MAP_NODES: MapNodeCol}


def _combat(ids: dict[str, dict[str, int]], hand: list[tuple], *, energy: int = 3, block: int = 0,
            enemies: tuple = ((30, 5, 1),), intents: tuple = (), powers: tuple = (), relics: tuple = (),
            side: str = "Player", hp: int = 50) -> bytes:
    """One single-seat combat frame. hand rows: (card, cost, cost flags, {var: shown}, {(var, tgt): shown});
    enemies: (hp, block, hittable); intents: (owner, type, dmg, hits); powers: (owner, name, amount)."""
    b = FrameBuilder()
    b.glob[G.IN_COMBAT], b.glob[G.ACTING_SEAT], b.glob[G.N_SEATS] = 1, 0, 1
    b.glob[G.COMBAT_SIDE] = ids["enum:CombatSide"][side]
    b.extend(Table.SEATS, [_row(b, Table.SEATS, SEAT=0, CREATURE_ROW=0, PRIVATE_VISIBLE=1, ALIVE=1, ENERGY=energy,
                                STARS=0, HP=hp)])
    cre = [_row(b, Table.CREATURES, KIND=CreatureKind.PLAYER, SEAT=0, HP=hp, BLOCK=block, ALIVE=1, HITTABLE=1)]
    cre += [_row(b, Table.CREATURES, KIND=CreatureKind.ENEMY, SEAT=-1, HP=h, BLOCK=k, ALIVE=1, HITTABLE=x)
            for h, k, x in enemies]
    b.extend(Table.CREATURES, cre)
    for owner in sorted({o for o, *_ in intents}):
        b.attach(Table.INTENTS, owner, [
            _row(b, Table.INTENTS, IDX=i, INTENT_TYPE=ids["enum:IntentType"][ty], HITS=h, DMG_SEAT0=d, DMG_SEAT1=-1,
                 DMG_SEAT2=-1, DMG_SEAT3=-1) for i, (o, ty, d, h) in enumerate(x for x in intents if x[0] == owner)])
    for owner in sorted({o for o, *_ in powers}):
        b.attach(Table.POWERS, owner, [_row(b, Table.POWERS, POWER_ID=ids["power"][n], AMOUNT=a, APPLIER_ROW=-1)
                                       for o, n, a in powers if o == owner])
    if relics:
        b.extend(Table.RELICS, [_row(b, Table.RELICS, SEAT=0, ZONE=ItemZone.OWNED, POS=i, RELIC_ID=ids["relic"][n])
                                for i, n in enumerate(relics)])
    for pos, (name, cost, flags, un, tg) in enumerate(hand):
        b.extend(Table.CARDS, [_row(b, Table.CARDS, ZONE=Zone.HAND, SEAT=0, POS=pos, COUNT=1, CARD_ID=ids["card"][name],
                                    ENERGY_COST=cost, COST_FLAGS=flags, STAR_COST=-1, HAND_FLAGS=HandFlag.CAN_PLAY)])
        rows = [_row(b, Table.VARS, OWNER_TABLE=Table.CARDS, OWNER_ROW=pos, TARGET_ROW=t, VAR_ID=ids["var_name"][v],
                     SHOWN=x, BASE=x, ENCHANTED=x, FLAGS=0)
                for (v, t), x in [((v, -1), x) for v, x in un.items()] + list(tg.items())]
        b.attach_vars(Table.CARDS, pos, rows)
    return b.build()


def test_reference_combat(tmp_path: Path) -> None:
    R, ids = _rules(tmp_path)
    X = int(CostFlag.COSTS_X)
    hand = [("STRIKE", 1, 0, {"Damage": 6}, {("Damage", 1): 9}),   # 9 on the vulnerable enemy 1
            ("STRIKE", 1, 0, {"Damage": 6}, {}),
            ("DEFEND", 1, 0, {"Block": 5}, {}),
            ("CLEAVE", 2, 0, {"Damage": 8}, {}),
            ("WHIRL", 0, X, {"Damage": 5}, {}),                      # X card: 5 x energy left
            ("BURN", 0, 0, {"Damage": 2}, {}),                       # turn-end damage, no target
            ("DECAY", 0, 0, {"HpLoss": 2}, {}),
            ("SHIELD_ALLY", 0, 0, {"Block": 7}, {})]                 # block on an ally: 0
    p = _combat(ids, hand, energy=3, block=0, enemies=((20, 4, 1), (12, 0, 1)),
                intents=((1, "Attack", 6, 2), (2, "Attack", 9, 1), (2, "Buff", -1, 0)),
                powers=((0, "PLATING_POWER", 3), (2, "SLIPPERY_POWER", 1)), relics=("ORICHALCUM",))
    turn, foe, route = reference_tables(parse_frame(p), R)
    # incoming 12 + 9 = 21; end block plating 3 + orichalcum 6 (block 0); end dmg Burn 2; end loss Decay 2
    # block reachable: DEFEND 5 (1 energy); a_best = max(0 + 9, 0 + 3 + 5) = 9
    assert turn.tolist() == [[0, 50, 0, 21, 9, 2, 2, 23 - 9 + 2, 5, 23 - 9 + 2, 0]]
    # enemy 1 (row 1, 24 to kill): STRIKE 9 + STRIKE 6 + WHIRL 5*1 (3 energy: two strikes, X gets 1) = 20, or
    # CLEAVE 8 + WHIRL 5*1 = 13, or STRIKE 9 + WHIRL 5*2 = 19, or WHIRL 15 ... best 20
    assert foe.tolist()[0] == [1, 24, 20, 12, 0]
    # enemy 2 (12, slippery: GUARDED): 6 + 6 + 5 = 17 >= 12 but KILL 0
    assert foe.tolist()[1] == [2, 12, 17, 9, 2]
    assert route.shape == (0, 12)
    # side Enemy: no turn-end terms; a hidden intent; an unhittable enemy; Orichalcum off with block
    p = _combat(ids, hand[:3], energy=1, block=4, enemies=((-1, 0, 0),), intents=((1, "Hidden", -1, 0),),
                side="Enemy", relics=("ORICHALCUM", "TUNGSTEN_ROD"))
    turn, foe, _ = reference_tables(parse_frame(p), R)
    assert turn.tolist() == [[0, 50, 4, 0, 0, 0, 0, 0, 5, 0, 0b01100]]
    assert foe.tolist() == [[1, -1, 0, 0, 0b10100]]


def test_reference_cap_and_random_enemy(tmp_path: Path) -> None:
    R, ids = _rules(tmp_path)
    hand = [("DEFEND", 1, 0, {"Block": 1 + i}, {}) for i in range(13)] + [("BOLT", 1, 0, {"Damage": 4}, {})]
    p = _combat(ids, hand, energy=2, enemies=((10, 0, 1), (10, 0, 0)))
    turn, foe, _ = reference_tables(parse_frame(p), R)
    assert turn[0, 8] == 13 + 12 and turn[0, 10] & 16  # the two best of the 12 kept; CAPPED
    assert foe[:, 2].tolist() == [4, 0]  # RandomEnemy aims: exactly one hittable enemy


def test_reference_route(tmp_path: Path) -> None:
    R, ids = _rules(tmp_path)
    t = ids["enum:MapPointType"]
    #   row 0: 0 Monster (REACHABLE), 1 Elite (REACHABLE)
    #   row 1: 2 RestSite, 3 Shop;  row 2: 4 Unknown (no out-edge -> BOSS);  5 BOSS, 6 SECOND_BOSS
    types = ["Monster", "Elite", "RestSite", "Shop", "Unknown", "Boss", "Boss"]
    b = FrameBuilder()
    b.extend(Table.MAP_NODES, [_row(b, Table.MAP_NODES, ROW=(0, 0, 1, 1, 2, 3, 4)[i], COL=i, POINT_TYPE=t[name],
                                    STATE=MapNodeState.REACHABLE if i < 2 else MapNodeState.NORMAL,
                                    FLAGS={5: MapNodeFlag.BOSS, 6: MapNodeFlag.SECOND_BOSS}.get(i, 0))
                               for i, name in enumerate(types)])
    b.extend(Table.MAP_EDGES, [(0, 2), (0, 3), (1, 3), (2, 4), (3, 4)])
    _, _, route = reference_tables(parse_frame(b.build()), R)
    # node 0: paths 0-2-4-5-6, 0-3-4-5-6; node 1: 1-3-4-5-6
    assert route.tolist() == [[0, 3, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                              [1, 3, -1, 1, -1, -1, 1, 1, 0, 1, 1, 1]]
