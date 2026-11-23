"""Package P1 of the critic contract (notes/critic/contract.md 1.2-1.3, 2.1; gate PG1 on synthetic frames): the HDRAW
trailer on the wire, ``Frame.hdraw``, ``strip_hidden``, the tokenizer's ``allow_hidden`` / ``card_base`` and CRIT
against an independent reference."""
from __future__ import annotations

import numpy as np
import pytest

import critic_data as cd
from spire.critic import empty_crit, side, truncated
from spire.proto import fnv1a32, hdraw_problems, parse_frame, read_message
from spire.schema import (DIR_STRIDE, HDRAW_MAX_BYTES, HDRAW_TABLE, MAX_FRAME_BYTES_CALC, MAX_FRAME_BYTES_HIDDEN, G, H,
                          CardCol, FrameFlag, Kind, ResetFlag, SchemaError, Table, Zone, n_tables_for, reset_flags_for)
from spire.synth import FrameBuilder, FrameGen, keep_trailers, strip_hidden
from spire.tokenize import Ent, Tokenizer, TokenizerConfig, _frame_bytes

K = 32


def crit_reference(payload: bytes, tok: Tokenizer, k: int = K) -> np.ndarray:
    """CRIT written from contract 2.1 without card_base: the token of CARDS row r is the r-th CARD-type token of the
    Sample; HDRAW rows walked in (SEAT, POS) order, POS < K, at most K."""
    f = parse_frame(payload)
    s = tok.encode(f)
    card_tokens = [i for i in range(s.n_tokens) if int(s.types[i]) == int(Ent.CARD)]
    out = [int(f.gvals[G.KIND]), 0] + [-1] * k
    rows = sorted(map(tuple, f.hdraw.tolist())) if f.hdraw is not None else []
    j = 0
    for seat, pos, owner in rows:
        if pos >= k or j == k:
            continue
        out[2 + j] = card_tokens[owner]
        j += 1
    out[1] = j
    return np.array(out, np.int16)


@pytest.mark.parametrize("config", cd.CONFIGS)
def test_pg1_synthetic(config: str) -> None:
    """Every flag-on frame parses with verify_hashes, satisfies the HDRAW invariants (HG2's list), strips to the
    flag-off frame (= keep_trailers without HDRAW), and its CRIT equals the reference; card_base is the same from the
    native and the numpy tokenizer and equals the first CARD token."""
    on_n, _ = cd.tokenizers(config, native=True)
    on_p, _ = cd.tokenizers(config, native=False)
    frames = cd.stream(config).frames
    n = rows = fights = trunc = 0
    for p in frames:
        f = parse_frame(p, verify_hashes=True)
        assert f.hdraw is not None and hdraw_problems(f) == [], hdraw_problems(f)
        assert keep_trailers(p, forecast=True, calc=True) == strip_hidden(p)
        assert keep_trailers(p, forecast=True, calc=True, hidden=True) == p
        q = strip_hidden(p)
        xh = f.hd_entry * 4
        a, b = np.frombuffer(p[:xh], "<i4").copy(), np.frombuffer(q, "<i4").copy()
        assert a[H.N_TABLES] == b[H.N_TABLES] + 1 and b[H.FRAME_BYTES] == xh
        a[H.N_TABLES], a[H.FRAME_BYTES] = b[H.N_TABLES], b[H.FRAME_BYTES]
        assert np.array_equal(a, b)
        e_n, e_p = on_n.encode_batch((f,)), on_p.encode_batch((f,))
        assert e_n.card_base[0] == e_p.card_base[0] == 2 + f.table(Table.SEATS).shape[0]
        c = side(f, int(e_n.card_base[0]), int(e_n.counts[0, 0]), K)
        assert np.array_equal(c, crit_reference(p, on_n)), (c, crit_reference(p, on_n))
        n += 1
        rows += f.hd_rows
        fights += int(f.gvals[G.IN_COMBAT] == 1)
        trunc += truncated(f, c)
        if f.gvals[G.IN_COMBAT] != 1:
            assert f.hd_rows == 0
    assert n >= cd.N_FRAMES and fights >= cd.N_FIGHT and rows > 0


def test_pg1_batch_card_base_sorted() -> None:
    """card_base follows ``EncodedBatch.order`` when the batch is sorted (native and numpy)."""
    frames = [parse_frame(p) for p in cd.stream("calc").frames[:200]]
    for native in (True, False):
        on, _ = cd.tokenizers("calc", native=native)
        enc = on.encode_batch(frames, sort=True)
        order = range(len(frames)) if enc.order is None else enc.order.tolist()
        want = [2 + frames[i].table(Table.SEATS).shape[0] for i in order]
        assert enc.card_base.dtype == np.int32 and enc.card_base.tolist() == want


def _builder_frame(gen: FrameGen, flags: int, n_seats: int, seed: int, kind: int = Kind.COMBAT) -> bytes:
    rng = np.random.default_rng(seed)
    ctx = gen.random_ctx(rng, n_seats, size="typical")
    ctx.reset_flags = int(flags)
    return gen.frame(rng, kind, ctx)


@pytest.mark.parametrize("fc,calc", [(False, False), (True, False), (False, True), (True, True)])
def test_trailer_chain_by_table_ids(gen: FrameGen, fc: bool, calc: bool) -> None:
    """The header count is ambiguous (17 = forecast or hidden, 20 = forecast + calc or calc + hidden): the parser
    reads the chain's table ids; the tokenizer of the same trailers accepts the frame with allow_hidden."""
    base = reset_flags_for(fc, calc=calc)
    for seed in range(20):
        p = _builder_frame(gen, base | int(ResetFlag.HIDDEN), 1, seed)
        f = parse_frame(p, verify_hashes=True)
        assert f.hdraw is not None and (f.forecast is not None) == fc and (f.cc_rows is not None) == calc
        assert int(f.words[H.N_TABLES]) == n_tables_for(fc, calc, True)
        q = strip_hidden(p)
        g = parse_frame(q, verify_hashes=True)
        assert g.hdraw is None and (g.forecast is not None) == fc and (g.cc_rows is not None) == calc
        assert q == _builder_frame(gen, base, 1, seed)  # the flag-off frame of the same draw
        tok = Tokenizer(gen.vocab, TokenizerConfig(forecast=fc, calc=calc))
        tok.allow_hidden = True
        assert np.array_equal(tok.encode(f).raw, Tokenizer(gen.vocab, TokenizerConfig(forecast=fc, calc=calc))
                              .encode(g).raw)


def test_multi_seat_and_truncation(gen: FrameGen) -> None:
    """Seats concatenated in seat order (seat 0 first), at most K rows, POS >= K dropped and counted."""
    flags = reset_flags_for(True, calc=True) | int(ResetFlag.HIDDEN)
    tok = Tokenizer(gen.vocab, TokenizerConfig(forecast=True, calc=True))
    tok.allow_hidden = True
    seen_two = 0
    for seed in range(40):
        p = _builder_frame(gen, flags, 2, seed)
        f = parse_frame(p, verify_hashes=True)
        assert hdraw_problems(f) == []
        seen_two += len(set(f.hdraw[:, 0].tolist())) == 2
        enc = tok.encode_batch((f,))
        for k in (4, 32):
            c = side(f, int(enc.card_base[0]), int(enc.counts[0, 0]), k)
            assert np.array_equal(c, crit_reference(p, tok, k))
            assert truncated(f, c) == f.hd_rows - int(c[1]) and int(c[1]) <= k
    assert seen_two >= 5


def test_big_pile_is_truncated_at_k() -> None:
    b = FrameBuilder()
    b.add(Table.SEATS, seat=0, creature_row=-1, draw_count=40, private_visible=1)
    inst = np.tile(np.array(FrameBuilder().new_rows(Table.CARDS, 1)[0]), (40, 1))
    inst[:, CardCol.ZONE], inst[:, CardCol.SEAT], inst[:, CardCol.COUNT] = int(Zone.DRAW), 0, 1
    inst[:, CardCol.CARD_ID] = np.arange(40) % 5 + 2
    groups = b.add_multiset(inst, [None] * 40)
    owner = np.repeat(np.arange(groups.start, groups.stop), b.rows(Table.CARDS)[:, CardCol.COUNT])
    b.hdraw = np.stack([np.zeros(40, np.int64), np.arange(40), np.random.default_rng(0).permutation(owner)], 1)
    b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.IN_COMBAT] = int(Kind.COMBAT), 1, 1
    b.add(Table.ACTIONS, verb=1, seat=0)
    f = parse_frame(b.build(), verify_hashes=True)
    assert hdraw_problems(f) == [] and f.hd_rows == 40
    c = side(f, 3, 3 + 5, K)
    assert int(c[1]) == K and truncated(f, c) == 8
    assert (c[2:] - 3 == f.hdraw[:K, 2]).all()


def test_side_without_table_and_bad_rows(gen: FrameGen) -> None:
    p = _builder_frame(gen, reset_flags_for(True, calc=True), 1, 3)
    f = parse_frame(p)
    c = side(f, 3, 10, K)
    assert c[0] == f.gvals[G.KIND] and c[1] == 0 and (c[2:] == -1).all()
    assert np.array_equal(empty_crit(2, K)[0], np.array([0, 0] + [-1] * K, np.int16))
    p = cd.stream("calc").frames[2]
    f = parse_frame(p)
    tok, _ = cd.tokenizers("calc")
    enc = tok.encode_batch((f,))
    if f.hd_rows:
        with pytest.raises(AssertionError, match="outside the sample"):
            side(f, int(enc.card_base[0]), 1, K)


def test_hdraw_problems_detects_broken_tables() -> None:
    p = next(q for q in cd.stream("calc").frames if parse_frame(q).hd_rows >= 3)
    f = parse_frame(p)
    w = np.frombuffer(p, "<i4").copy()
    o = f.hd_off
    w[o + 1], w[o + 4] = w[o + 4], w[o + 1]  # POS order broken
    w[f.hd_entry + 4] = np.int32(np.uint32(fnv1a32(w[o : o + 3 * f.hd_rows].tobytes())).astype(np.int32))
    g = parse_frame(w.tobytes(), verify_hashes=True)
    assert any("POS" in s or "ascending" in s for s in hdraw_problems(g))
    w2 = np.frombuffer(p, "<i4").copy()
    w2[f.hd_off + 2] = -5
    with pytest.raises(SchemaError):
        parse_frame(w2.tobytes(), verify_hashes=True)
    assert hdraw_problems(parse_frame(w2.tobytes())) == ["HDRAW: OWNER_ROW outside CARDS"]


def test_elided_rebuild_keeps_the_trailer() -> None:
    """``_frame_bytes`` of an elided frame re-appends HDRAW after the calc tables (byte-equal to the payload)."""
    for p in cd.stream("v2a").frames[:50]:
        f = parse_frame(p)
        f.flags |= int(FrameFlag.ELIDED)
        assert bytes(_frame_bytes(f)) == p


def test_limits_and_read_message() -> None:
    import io
    assert HDRAW_MAX_BYTES == 12308 and MAX_FRAME_BYTES_HIDDEN == MAX_FRAME_BYTES_CALC + 12308
    big = b"\x00" * (MAX_FRAME_BYTES_CALC + 100)
    assert read_message(io.BytesIO(len(big).to_bytes(4, "little") + big)) == big
    assert HDRAW_TABLE == 20 and DIR_STRIDE == 5


def test_tokenizer_allow_hidden_is_runtime_only() -> None:
    cfg = cd.tok_config("calc")
    tok = Tokenizer(cd.vocab(), cfg)
    assert not tok.allow_hidden and "allow_hidden" not in {f for f in cfg.__dataclass_fields__}
    tok.allow_hidden = True
    assert tok._n_hdr == 21
    tok.allow_hidden = False
    assert tok._n_hdr == 20
