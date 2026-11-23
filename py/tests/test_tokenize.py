from __future__ import annotations

import numpy as np
import pytest
import torch

from spire.proto import parse_frame
from spire.schema import (
    DECISION_KINDS,
    ActionCol,
    CardCol,
    CreatureKind,
    G,
    HandFlag,
    Kind,
    MsgType,
    OptionKind,
    SeatCol,
    Table,
    VarCol,
    Verb,
    Zone,
)
from spire.synth import FrameBuilder, FrameGen, error_frame, random_frame
from spire.tokenize import (
    SPECS,
    BatchShape,
    Ent,
    FeatureDecoder,
    Sample,
    Tokenizer,
    TokenizerConfig,
    symlog,
)
from spire.vocab import Vocab

TOKEN_ENTS = {Ent.GLOBAL, Ent.DECISION, Ent.SEAT, Ent.CARD, Ent.CREATURE, Ent.RELIC, Ent.POTION, Ent.ORB,
              Ent.ARROW, Ent.MAP_NODE, Ent.OPTION, Ent.CELL}
ENT_OF_TABLE = {Table.SEATS: Ent.SEAT, Table.CARDS: Ent.CARD, Table.CREATURES: Ent.CREATURE, Table.RELICS: Ent.RELIC,
                Table.POTIONS: Ent.POTION, Table.ORBS: Ent.ORB, Table.ARROWS: Ent.ARROW,
                Table.MAP_NODES: Ent.MAP_NODE, Table.OPTIONS: Ent.OPTION, Table.CELLS: Ent.CELL,
                Table.POWERS: Ent.POWER, Table.INTENTS: Ent.INTENT, Table.REFS: Ent.REF}


def all_frames(gen: FrameGen, seed: int = 0) -> list[bytes]:
    rng = np.random.default_rng(seed)
    out = [random_frame(gen, rng, k, n_seats=s) for k in DECISION_KINDS for s in (1, 2, 4)]
    out += [random_frame(gen, rng, k, n_seats=4, size="extreme") for k in (Kind.COMBAT, Kind.SHOP, Kind.CRYSTAL)]
    return out


@pytest.fixture(scope="module")
def tok(vocab: Vocab) -> Tokenizer:
    return Tokenizer(vocab)


@pytest.fixture(scope="module")
def tok_tokens(vocab: Vocab) -> Tokenizer:
    return Tokenizer(vocab, TokenizerConfig(pool_children=False))


def rows(f, t: Table) -> int:  # type: ignore[no-untyped-def]
    return int(f.table(t).shape[0])


@pytest.mark.parametrize("pooled", [True, False])
def test_encode_structure_every_kind(gen: FrameGen, vocab: Vocab, pooled: bool) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(pool_children=pooled))
    for payload in all_frames(gen):
        f = parse_frame(payload)
        s = tok.encode(f)
        n_tok, n_kid, n_var, n_act = s.counts
        N = sum(s.counts)
        assert s.types.shape == (N,) and s.raw.shape == (N, tok.width) and s.rel.shape == (N, 3)
        assert s.raw.dtype == np.int16 and s.rel.dtype == np.int16
        ty = s.types.astype(int)
        assert ty[0] == Ent.GLOBAL and ty[1] == Ent.DECISION
        base_tok = 2 + sum(rows(f, t) for t in (Table.SEATS, Table.CARDS, Table.CREATURES, Table.RELICS,
                                                  Table.POTIONS, Table.ORBS, Table.ARROWS, Table.MAP_NODES,
                                                  Table.OPTIONS, Table.CELLS))
        children = rows(f, Table.POWERS) + rows(f, Table.INTENTS) + rows(f, Table.REFS)
        assert n_act == f.n_actions
        assert s.edges.shape == (2 * rows(f, Table.MAP_EDGES), 3)
        if pooled:
            assert n_tok == base_tok and n_kid == children and n_var == rows(f, Table.VARS)
            assert set(ty[:n_tok]) <= {int(e) for e in TOKEN_ENTS}
        else:
            assert n_kid == 0 and n_tok + n_var == base_tok + children + rows(f, Table.VARS)
        assert (ty[n_tok + n_kid : n_tok + n_kid + n_var] == Ent.VAR).all()
        assert (ty[N - n_act :] == Ent.ACTION).all()
        rel = s.rel.astype(int)
        assert ((rel >= -1) & (rel < N)).all()
        # every relation of a token / kid points at a token
        assert (rel[: n_tok + n_kid, :2][rel[: n_tok + n_kid, :2] >= 0] < n_tok).all()
        # actions: source entity type matches the source table, target is a creature, actor a seat
        acts = f.actions
        ar = rel[N - n_act :]
        for i in range(n_act):
            st = int(acts[i, ActionCol.SRC_TABLE])
            if st >= 0:
                assert ty[ar[i, 0]] == ENT_OF_TABLE[Table(st)]
            else:
                assert ar[i, 0] == -1
            assert (ar[i, 1] == -1) == (acts[i, ActionCol.TGT_ROW] == -1)
            if ar[i, 1] >= 0:
                assert ty[ar[i, 1]] == Ent.CREATURE
            assert ty[ar[i, 2]] == Ent.SEAT
        # edges link map-node tokens in both directions
        if s.edges.shape[0]:
            e = s.edges.astype(int)
            assert (ty[e[:, :2]] == Ent.MAP_NODE).all()
            E = e.shape[0] // 2
            assert (e[:E, 0] == e[E:, 1]).all() and (e[:E, 2] == 0).all() and (e[E:, 2] == 1).all()
        assert s.cond.tolist()[:2] == [int(f.glob[G.ASCENSION]), int(f.glob[G.N_SEATS])]


def test_relations_and_var_routing(gen: FrameGen, tok: Tokenizer) -> None:
    rng = np.random.default_rng(3)
    seen_targeted = 0
    for _ in range(30):
        f = parse_frame(random_frame(gen, rng, Kind.COMBAT, n_seats=2))
        s = tok.encode(f)
        n_tok, n_kid, n_var, n_act = s.counts
        rel, ty = s.rel.astype(int), s.types.astype(int)
        b_seat, b_card = 2, 2 + rows(f, Table.SEATS)
        b_cre = b_card + rows(f, Table.CARDS)
        seats = f.table(Table.SEATS)
        for k in range(seats.shape[0]):
            cr = int(seats[k, SeatCol.CREATURE_ROW])
            assert rel[b_seat + k, 1] == b_cre + cr
        cards = f.table(Table.CARDS)
        lr = cards[:, CardCol.LINK_ROW]
        assert (rel[b_card : b_card + len(cards), 1] == np.where(lr >= 0, b_card + lr, -1)).all()
        vars_ = f.table(Table.VARS)
        vrel = rel[n_tok + n_kid : n_tok + n_kid + n_var, 0]
        acts = f.actions
        a0 = n_tok + n_kid + n_var
        for i in range(len(vars_)):
            own = int(vars_[i, VarCol.OWNER_ROW])
            tgt = int(vars_[i, VarCol.TARGET_ROW])
            match = np.flatnonzero((acts[:, ActionCol.SRC_TABLE] == vars_[i, VarCol.OWNER_TABLE])
                                   & (acts[:, ActionCol.SRC_ROW] == own) & (acts[:, ActionCol.TGT_ROW] == tgt))
            if tgt >= 0 and len(match):
                seen_targeted += 1
                assert vrel[i] == a0 + match[0]
                assert acts[match[0], ActionCol.VERB] == Verb.PLAY_CARD
            elif vars_[i, VarCol.OWNER_TABLE] == Table.CARDS:
                assert vrel[i] == b_card + own
            elif vars_[i, VarCol.OWNER_TABLE] == Table.POWERS:
                assert ty[vrel[i]] == Ent.POWER and n_tok <= vrel[i] < n_tok + n_kid
    assert seen_targeted > 0


def test_rejects_error_and_elided_frames(tok: Tokenizer, gen: FrameGen) -> None:
    with pytest.raises(ValueError):
        tok.encode(parse_frame(error_frame(1)))
    f = parse_frame(random_frame(gen, np.random.default_rng(0), Kind.MAP))
    f.tables[Table.RELICS] = None
    with pytest.raises(ValueError):
        tok.encode(f)


def test_terminal_frame_encodes_without_actions(gen: FrameGen, tok: Tokenizer) -> None:
    rng = np.random.default_rng(1)
    ctx = gen.random_ctx(rng, 1)
    s = tok.encode(parse_frame(gen.terminal(rng, ctx, 1)))
    assert s.n_actions == 0
    b = tok.collate([s])
    assert b["act_mask"].shape[1] == tok.config.action_align and not b["act_mask"].any()


def test_collate_layout(gen: FrameGen, tok: Tokenizer) -> None:
    samples = [tok.encode(parse_frame(p)) for p in all_frames(gen, seed=5)]
    shape = tok.shape_of(samples)
    b = tok.collate(samples)
    B, T = b["tok_mask"].shape
    A = b["act_mask"].shape[1]
    Nk, Nv = b["kid_mask"].shape[0], b["var_mask"].shape[0]
    assert (B, T, A, Nk, Nv) == (shape.batch, shape.tokens, shape.actions, shape.kids, shape.vars)
    N = B * T + Nk + Nv + B * A
    dest = b["dest"].numpy()
    M = sum(sum(s.counts) for s in samples)
    assert dest.shape == (M,) and len(np.unique(dest)) == M and dest.max() < N
    # every row lands in its group's region, tokens of sample i in row block i
    off = 0
    for i, s in enumerate(samples):
        t, k, v, a = s.counts
        d = dest[off : off + t + k + v + a]
        assert (d[:t] == i * T + np.arange(t)).all()
        assert ((d[t : t + k] >= B * T) & (d[t : t + k] < B * T + Nk)).all()
        assert ((d[t + k : t + k + v] >= B * T + Nk) & (d[t + k : t + k + v] < B * T + Nk + Nv)).all()
        assert (d[t + k + v :] == B * T + Nk + Nv + i * A + np.arange(a)).all()
        # relations translate to the destination of the referenced row
        r = s.rel.astype(int)
        g = b["rel"].numpy()[off : off + len(d)]
        assert (g[r < 0] == N).all()
        assert (g[r >= 0] == d[r[r >= 0]]).all()
        assert b["tok_mask"][i].sum() == t and b["act_mask"][i].sum() == a
        off += len(d)
    assert (b["raw"].numpy() == np.concatenate([s.raw for s in samples])).all()


def test_collate_fixed_shape_pads_to_sink(gen: FrameGen, tok: Tokenizer) -> None:
    samples = [tok.encode(parse_frame(p)) for p in all_frames(gen, seed=6)[:5]]
    nat = tok.shape_of(samples)
    big = BatchShape(nat.batch + 3, nat.tokens + 64, nat.actions + 8, nat.kids + 64, nat.vars + 64,
                     nat.edges + 64, nat.rows + 100)
    b = tok.collate(samples, shape=big)
    N = big.batch * big.tokens + big.kids + big.vars + big.batch * big.actions
    assert b["raw"].shape[0] == big.rows and b["edges"].shape[0] == big.edges
    assert (b["dest"][nat.rows :] == N + 1).all() and (b["types"][nat.rows :] == 0).all()
    assert (b["rel"][nat.rows :] == N).all()
    assert b["tok_mask"][nat.batch :, 0].all() and not b["tok_mask"][nat.batch :, 1:].any()
    assert not b["act_mask"][nat.batch :].any()
    with pytest.raises(ValueError):
        tok.collate(samples, shape=BatchShape(1, 48, 8, 64, 64, 64, 10))


def test_bucket_batches(gen: FrameGen, tok: Tokenizer) -> None:
    samples = [tok.encode(parse_frame(p)) for p in all_frames(gen, seed=7)] * 3
    groups = tok.bucket_batches(samples, max_tokens=1000, max_batch=7)
    assert sorted(i for g in groups for i in g) == list(range(len(samples)))
    for g in groups:
        bk = {tok.config.bucket(samples[i].n_tokens) for i in g}
        assert len(bk) == 1
        assert len(g) <= 7 and (len(g) == 1 or len(g) * bk.pop() <= 1000)


def _hand_frame(vocab: Vocab) -> bytes:
    """A small hand-built COMBAT frame with known values."""
    b = FrameBuilder()
    b.add(Table.SEATS, seat=0, character_id=vocab.id("character", "SILENT"), creature_row=0, private_visible=1,
          alive=1, hp=30, max_hp=60, block=5, gold=100, potion_slots=3, energy=-1, max_energy=3, hand_count=1,
          draw_count=-1)
    b.add(Table.CREATURES, kind=CreatureKind.PLAYER, seat=0, pos=0, model_id=vocab.id("character", "SILENT"),
          hp=30, max_hp=60, alive=1, hittable=1)
    enemy = b.add(Table.CREATURES, kind=CreatureKind.ENEMY, seat=-1, pos=0, model_id=5, hp=-1, max_hp=-1, alive=1,
                  hittable=1, poison_next_turn=0)
    card = b.add(Table.CARDS, zone=Zone.HAND, seat=0, pos=0, count=1, card_id=7, upgrade_level=1, energy_cost=2,
                 hand_flags=HandFlag.CAN_PLAY | HandFlag.GLOW_RED, star_cost=-1, enchant_amount=-1, aux=-1)
    var = b.new_rows(Table.VARS, 2)
    var[:, VarCol.VAR_ID] = 3
    var[:, VarCol.SHOWN], var[:, VarCol.BASE], var[:, VarCol.ENCHANTED] = (6, 9), 6, 6
    var[:, VarCol.TARGET_ROW] = (-1, enemy)
    b.attach_vars(Table.CARDS, card, var)
    opt = b.add(Table.OPTIONS, kind=OptionKind.REST_OPTION, seat=0, pos=0, key_id=vocab.id("rest_option", "HEAL"),
                price=-1, amount=-1, ref_table=-1, ref_row=-1, group=-1)
    assert opt == 0
    b.add(Table.ACTIONS, verb=Verb.PLAY_CARD, seat=0, src_table=Table.CARDS, src_row=card, tgt_row=enemy)
    b.add(Table.ACTIONS, verb=Verb.END_TURN, seat=0)
    b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.ASCENSION], b.glob[G.CUR_ROW] = Kind.COMBAT, 1, 7, -1
    b.msg_type = MsgType.OBS
    return b.build()


def test_feature_values_and_id_regions(vocab: Vocab, tok: Tokenizer) -> None:
    f = parse_frame(_hand_frame(vocab))
    s = tok.encode(f)
    L = tok.layout
    ids, wide = FeatureDecoder(L)(torch.from_numpy(s.types.astype(np.int64)), torch.from_numpy(s.raw))
    ty = s.types.astype(int)
    seat = int(np.flatnonzero(ty == Ent.SEAT)[0])
    names = [n for n in SPECS[Ent.SEAT].nums]
    o = L.num_offsets[Ent.SEAT]
    feat = {j: float(wide[seat, o + j]) for j in range(len(names))}
    assert feat[2] == pytest.approx(0.5)  # hp / max_hp
    assert feat[3] == pytest.approx(float(np.log1p(30)))  # symlog hp
    assert feat[6] == pytest.approx(float(np.log1p(10)))  # symlog gold / 10
    assert feat[8] == 0.0 and feat[9] == 1.0  # energy -1 -> value 0 + missing bit
    assert feat[17] == 1.0  # draw_count missing
    # only the seat's own block is non-zero
    other = np.ones(L.n_wide, bool)
    other[o : o + len(names)] = False
    assert not wide[seat, torch.from_numpy(other)].any()
    # ids land inside the region of their table
    off, rows_ = L.regions["character"]
    assert off <= int(ids[seat, 1]) < off + rows_
    assert int(ids[seat, 0]) == L.regions["p:seat"][0] + 1  # seat 0 -> +1
    cre = np.flatnonzero(ty == Ent.CREATURE)
    model_slot = [i.src for i in SPECS[Ent.CREATURE].ids].index("model")
    assert L.regions["character"][0] <= int(ids[cre[0], model_slot]) < sum(L.regions["character"])
    assert int(ids[cre[1], model_slot]) == L.regions["monster"][0] + 5
    o_c = L.num_offsets[Ent.CREATURE]
    assert float(wide[cre[1], o_c + 0]) == 0.0 and float(wide[cre[1], o_c + 2]) == 1.0  # hp -1: ratio 0, missing
    opt = int(np.flatnonzero(ty == Ent.OPTION)[0])
    key_slot = [i.src for i in SPECS[Ent.OPTION].ids].index("key")
    assert int(ids[opt, key_slot]) == L.regions["rest_option"][0] + vocab.id("rest_option", "HEAL")
    card = int(np.flatnonzero(ty == Ent.CARD)[0])
    static_slot = [i.src for i in SPECS[Ent.CARD].ids].index("static")
    assert int(ids[card, static_slot]) == L.static["card"][0] + 7 * L.static["card"][2] + 1
    o_k = L.num_offsets[Ent.CARD]
    hand_bits = [j for j, n in enumerate(SPECS[Ent.CARD].nums) if n.src == CardCol.HAND_FLAGS]
    assert [float(wide[card, o_k + j]) for j in hand_bits] == [1.0, 0.0, 1.0, 0.0]
    eq = [j for j, n in enumerate(SPECS[Ent.CARD].nums) if n.src == CardCol.ENERGY_COST and n.mode == 3]
    assert [float(wide[card, o_k + j]) for j in eq] == [0, 0, 1, 0, 0, 0]
    # untargeted var pooled into the card, targeted var routed to the PLAY_CARD action
    n_tok, n_kid, n_var, n_act = s.counts
    vr = s.rel[n_tok + n_kid : n_tok + n_kid + n_var, 0].astype(int)
    play = int(np.flatnonzero(f.actions[:, ActionCol.VERB] == Verb.PLAY_CARD)[0])
    assert sorted(vr.tolist()) == [card, n_tok + n_kid + n_var + play]
    vrow = n_tok + n_kid + int(np.flatnonzero(vr != card)[0])
    o_v = L.num_offsets[Ent.VAR]
    assert float(wide[vrow, o_v + 0]) == pytest.approx(float(np.log1p(9)))
    assert float(wide[vrow, o_v + 1]) == pytest.approx(float(np.log1p(3)))  # shown - enchanted
    assert s.cond.tolist() == [7, 1, vocab.id("character", "SILENT"), 0, 0, 0]


def test_symlog_and_int16_saturation(vocab: Vocab, tok: Tokenizer) -> None:
    x = torch.tensor([-1e6, -3.0, 0.0, 3.0, 1e6])
    assert torch.allclose(symlog(x), torch.tensor([-8.0, -float(np.log1p(3)), 0.0, float(np.log1p(3)), 8.0]))
    b = FrameBuilder()
    b.add(Table.SEATS, seat=0, gold=1_000_000, hp=10, max_hp=10, creature_row=-1)
    b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=0)
    b.glob[G.KIND], b.glob[G.N_SEATS] = Kind.REWARD, 1
    s = tok.encode(parse_frame(b.build()))
    slot = tok.layout.slot(Ent.SEAT, SeatCol.GOLD)
    assert int(s.raw[2, slot]) == 32767


def test_sample_is_independent_of_payload(gen: FrameGen, tok: Tokenizer) -> None:
    payload = bytearray(random_frame(gen, np.random.default_rng(9), Kind.COMBAT))
    s = tok.encode(parse_frame(payload))
    before = (s.raw.copy(), s.rel.copy())
    payload[640:] = bytes(len(payload) - 640)
    assert isinstance(s, Sample)
    assert (s.raw == before[0]).all() and (s.rel == before[1]).all()
