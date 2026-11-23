"""Reference implementation of the per-frame tokenizer (``Tokenizer.encode`` as reviewed in
py-review, 2026-11-14).

The production :meth:`spire.tokenize.Tokenizer.encode_batch` tokenizes a whole inference batch in
one vectorised pass. This module keeps the original one-frame-at-a-time algorithm so that
tests (``tests/test_tokenize_batch.py``) and benchmarks (``scripts/bench_hotpath.py``) can prove
that the batch path produces bit-identical :class:`~spire.tokenize.Sample` records and
:data:`~spire.tokenize.Batch` tensors. Do not optimise this file: its value is that it is the
simple independent implementation. Opt-in stage-B features are mirrored explicitly.
"""
from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from .proto import Frame
from .schema import (
    N_TABLES,
    ActionCol,
    CardCol,
    CreatureCol,
    CreatureKind,
    FoeCol,
    ForecastCol,
    G,
    H,
    IntentCol,
    Kind,
    MapNodeCol,
    MapNodeFlag,
    OptionCol,
    OptionKind,
    PotionCol,
    PowerCol,
    RefCol,
    RelicCol,
    RouteCol,
    SeatCol,
    Table,
    TurnCol,
    VarCol,
    Zone,
)
from .tokenize import AHEAD_PAIRS, AHEAD_SLOTS, AHEAD_TYPES, Ent, Sample, Tokenizer

I32 = NDArray[np.int32]
I64 = NDArray[np.int64]

_I16_MIN, _I16_MAX = -(1 << 15), (1 << 15) - 1
_NO_EDGES = np.zeros((0, 3), np.int16)
(_SEATS, _CARDS, _VARS, _CREATURES, _POWERS, _INTENTS, _RELICS, _POTIONS, _ORBS, _ARROWS, _NODES, _EDGES,
 _OPTIONS, _REFS, _CELLS, _ACTIONS) = (int(t) for t in Table)
_A_SEAT, _A_SRC_T, _A_SRC_R, _A_TGT = (int(c) for c in (ActionCol.SEAT, ActionCol.SRC_TABLE, ActionCol.SRC_ROW,
                                                          ActionCol.TGT_ROW))
_V_OWNER_T, _V_OWNER_R, _V_TGT = int(VarCol.OWNER_TABLE), int(VarCol.OWNER_ROW), int(VarCol.TARGET_ROW)
_K_ZONE, _K_ID, _K_UP, _K_LINK, _K_AUX = (int(c) for c in (CardCol.ZONE, CardCol.CARD_ID, CardCol.UPGRADE_LEVEL,
                                                             CardCol.LINK_ROW, CardCol.AUX))
_CR_KIND, _CR_MODEL, _CR_HP, _CR_POISON = (int(c) for c in (CreatureCol.KIND, CreatureCol.MODEL_ID, CreatureCol.HP,
                                                             CreatureCol.POISON_NEXT_TURN))
_P_ID, _P_OWNER, _P_APPLIER = int(PowerCol.POWER_ID), int(PowerCol.OWNER_ROW), int(PowerCol.APPLIER_ROW)
_O_KIND, _O_KEY, _O_REF_T, _O_REF_R = (int(c) for c in (OptionCol.KIND, OptionCol.KEY_ID, OptionCol.REF_TABLE,
                                                         OptionCol.REF_ROW))
_ZONE_CHOICE, _ZONE_OFFER, _PLAYER = int(Zone.CHOICE), int(Zone.OFFER), int(CreatureKind.PLAYER)


def _resolve(off: I64, rows: I64, disc: NDArray[np.integer], ids: NDArray[np.integer]) -> I64:
    o = off.take(disc, mode="clip")
    r = rows.take(disc, mode="clip")
    return np.where((r > 0) & (ids > 0), o + ids, 0)


def _ref(base: I64, table: NDArray[np.integer], row: NDArray[np.integer]) -> I64:
    b = base.take(table, mode="clip")
    return np.where((table >= 0) & (b >= 0), b + row, -1)


def _match_vars(vars_: I32, acts: I32) -> I64:
    out = np.full(vars_.shape[0], -1, np.int64)
    tgt = vars_[:, _V_TGT]
    targeted = tgt >= 0
    if not acts.shape[0] or not targeted.any():
        return out
    a_tgt = acts[:, _A_TGT]
    a_ok = (acts[:, _A_SRC_T] >= 0) & (a_tgt >= 0)
    a_key = np.where(a_ok, (acts[:, _A_SRC_T].astype(np.int64) * 4096 + acts[:, _A_SRC_R]) * 64 + a_tgt, -1)
    v_key = (vars_[:, _V_OWNER_T].astype(np.int64) * 4096 + vars_[:, _V_OWNER_R]) * 64 + tgt
    order = np.argsort(a_key, kind="stable")
    sk = a_key[order]
    pos = np.minimum(np.searchsorted(sk, v_key), len(sk) - 1)
    hit = targeted & (sk[pos] == v_key)
    out[hit] = order[pos[hit]]
    return out


def _ahead_reference(tok: Tokenizer, route: I32, nodes: I32) -> I32:
    """The 30 AHEAD slots (notes/inputs2/contract.md 2.2, v2a.md 2) of a frame's ROUTE rows, one row at a time."""
    types = [tok.vocab.id("enum:MapPointType", name) for name in AHEAD_TYPES]
    rows = [[int(x) for x in r] for r in route]
    node = [nodes[r[RouteCol.OWNER_ROW]] for r in rows]
    ty = [int(x[MapNodeCol.POINT_TYPE]) for x in node]
    fl = [int(x[MapNodeCol.FLAGS]) for x in node]

    def minv(c: int) -> int:
        v = [r[c] for r in rows if r[c] >= 0]
        return min(v) if v else -1

    def maxv(c: int) -> int:
        v = [r[c] for r in rows]
        return -1 if not v or -1 in v else max(v)

    out = [len(rows)] + [ty.count(t) for t in types]
    out.append(sum(1 for x in fl if x & (MapNodeFlag.BOSS | MapNodeFlag.SECOND_BOSS)))
    out.append(minv(RouteCol.ROWS_TO_BOSS))
    for c in AHEAD_PAIRS:
        out += [minv(c), maxv(c)]
    out.append(sum(1 for x in fl if x & MapNodeFlag.QUEST))
    assert len(out) == AHEAD_SLOTS
    return np.array([out], np.int32)


def encode_reference(tok: Tokenizer, frame: Frame) -> Sample:
    """The original ``Tokenizer.encode`` (one frame, ~50 small numpy calls)."""
    if not (frame.is_obs or frame.is_terminal):
        raise ValueError("only OBS / TERMINAL frames can be tokenized")
    if any(a is None for a in frame.tables):
        raise ValueError("frame has elided tables; call resolve_elided(prev) first")
    fc = frame.forecast
    assert (fc is not None) == tok.config.forecast, "FORECAST trailer iff tokenizer.forecast"
    assert (frame.cc_rows is not None) == tok.config.calc, "calc trailers iff tokenizer.calc"
    assert (frame.hd_rows >= 0) == tok.allow_hidden, "HDRAW trailer iff tokenizer.allow_hidden"
    T = list(cast(list[I32], frame.tables))
    glob = frame.glob
    L = tok.layout
    calc_tabs = [cast(I32, frame.calc_table(k)) for k in range(3)] if tok.config.calc else []
    ahead = None
    if tok.config.route_summary:  # from the ROUTE rows before the DROP rule (notes/inputs2/contract.md 2.1)
        assert int(glob[G.IN_COMBAT]) != 1 or T[_NODES].shape[0] > 0, \
            "tokenizer.route_summary: a frame with IN_COMBAT = 1 has no MAP_NODES rows (hosts need RESET bit 1)"
        ahead = _ahead_reference(tok, calc_tabs[2], T[_NODES])
    if tok.config.map_tokens == "no_combat" and int(glob[G.IN_COMBAT]) == 1 and int(glob[G.KIND]) != Kind.MAP:
        # DROP: the three map tables have 0 rows; A1 first (notes/inputs2/contract.md 1)
        step = int(frame.words[H.STEP_ID])
        for t, c, what in ((_ACTIONS, _A_SRC_T, "ACTIONS row {} has SRC_TABLE"),
                           (_OPTIONS, _O_REF_T, "OPTIONS row {} has REF_TABLE"),
                           (_REFS, int(RefCol.ENT_TABLE), "REFS row {} has ENT_TABLE")):
            for i, row in enumerate(T[t]):
                if int(row[c]) == _NODES:
                    raise AssertionError(f"A1: frame step {step} (IN_COMBAT = 1, KIND != MAP: map tables dropped): "
                                         f"{what.format(i)} = MAP_NODES")
        T[_NODES], T[_EDGES] = T[_NODES][:0], T[_EDGES][:0]
        if calc_tabs:
            calc_tabs[2] = calc_tabs[2][:0]
    cols = [L.slot_cols[e] for e in Ent]
    pooled = tok.config.pool_children
    d = L.derived_slot
    d_kind_sub, d_prompt = d[Ent.GLOBAL]["kind_sub"], d[Ent.DECISION]["prompt_source"]
    d_acts = d[Ent.SEAT]["acts"]
    d_card = (d[Ent.CARD]["static"], d[Ent.CARD]["pile"], d[Ent.CARD]["group"])
    d_cre = (d[Ent.CREATURE]["model"], d[Ent.CREATURE]["static"], d[Ent.CREATURE]["poison_lethal"])
    d_relic, d_potion = d[Ent.RELIC]["static"], d[Ent.POTION]["static"]
    d_drow, d_key = d[Ent.MAP_NODE]["drow"], d[Ent.OPTION]["key"]
    d_power = (d[Ent.POWER]["static"], d[Ent.POWER]["applier_kind"])
    d_ref = d[Ent.REF]["vocab"]
    st = {cat: off for cat, (off, _, _) in L.static.items()}
    card_levels = L.static["card"][2]
    ref_c = (int(RefCol.CATEGORY), int(RefCol.VOCAB_ID), int(RefCol.OWNER_TABLE), int(RefCol.OWNER_ROW),
             int(RefCol.ENT_TABLE), int(RefCol.ENT_ROW))
    tok_ents = [int(e) for e in (Ent.SEAT, Ent.CARD, Ent.CREATURE, Ent.RELIC, Ent.POTION, Ent.ORB,
                                 Ent.ARROW, Ent.MAP_NODE, Ent.OPTION, Ent.CELL)]
    tok_tabs = [_SEATS, _CARDS, _CREATURES, _RELICS, _POTIONS, _ORBS, _ARROWS, _NODES, _OPTIONS, _CELLS]

    seats, cards, vars_, cre, pw, it = T[0], T[1], T[2], T[3], T[4], T[5]
    edges, opts, refs, acts = T[_EDGES], T[_OPTIONS], T[_REFS], T[_ACTIONS]
    cnt = [a.shape[0] for a in T]
    n_var, n_act = cnt[_VARS], cnt[_ACTIONS]

    base = [-1] * N_TABLES
    seg_e, seg_n = [int(Ent.GLOBAL), int(Ent.DECISION)], [1, 1]
    seg_a: list[tuple[int, I32, I64]] = []
    n = 2
    for e, t in zip(tok_ents, tok_tabs):
        base[t] = n
        seg_e.append(e)
        seg_n.append(cnt[t])
        seg_a.append((n, T[t], cols[e]))
        n += cnt[t]
    child = ((int(Ent.POWER), _POWERS), (int(Ent.INTENT), _INTENTS), (int(Ent.REF), _REFS))
    var_act = _match_vars(vars_, acts) if n_var else np.zeros(0, np.int64)
    tok_var: NDArray[np.bool_] | None = None
    if not pooled:
        for e, t in child:
            base[t] = n
            seg_e.append(e)
            seg_n.append(cnt[t])
            seg_a.append((n, T[t], cols[e]))
            n += cnt[t]
        tok_var = var_act < 0
        v_tok = vars_[tok_var]
        base[_VARS] = n
        seg_e.append(int(Ent.VAR))
        seg_n.append(v_tok.shape[0])
        seg_a.append((n, v_tok, cols[Ent.VAR]))
        n += v_tok.shape[0]
    n_tok = n
    if pooled:
        for e, t in child:
            base[t] = n
            seg_e.append(e)
            seg_n.append(cnt[t])
            seg_a.append((n, T[t], cols[e]))
            n += cnt[t]
    fc_base = n
    if fc is not None:  # FORECAST kids after REFS
        seg_e.append(int(Ent.FORECAST))
        seg_n.append(fc.shape[0])
        seg_a.append((n, fc, cols[Ent.FORECAST]))
        n += fc.shape[0]
    calc_kids: list[tuple[int, I32, int, int]] = []  # (first row, table, owner column, owner base)
    if tok.config.calc:  # TURN, FOE, ROUTE kids after FORECAST (notes/calc/contract.md 3.2)
        for k, (e, oc, ot) in enumerate(((Ent.TURN, TurnCol.OWNER_ROW, _CREATURES),
                                         (Ent.FOE, FoeCol.OWNER_ROW, _CREATURES),
                                         (Ent.ROUTE, RouteCol.OWNER_ROW, _NODES))):
            a = calc_tabs[k]
            seg_e.append(int(e))
            seg_n.append(a.shape[0])
            seg_a.append((n, a, cols[e]))
            calc_kids.append((n, a, int(oc), ot))
            n += a.shape[0]
    ahead_row = n
    if ahead is not None:  # the AHEAD row after ROUTE, a kid of the GLOBAL token (contract 2.3)
        seg_e.append(int(Ent.AHEAD))
        seg_n.append(1)
        seg_a.append((n, ahead, cols[Ent.AHEAD]))
        n += 1
    n_kid = n - n_tok
    v_grp = vars_ if tok_var is None else vars_[~tok_var]
    var_base = n
    seg_e += [int(Ent.VAR), int(Ent.ACTION)]
    seg_n += [v_grp.shape[0], n_act]
    seg_a += [(n, v_grp, cols[Ent.VAR]), (n + v_grp.shape[0], acts, cols[Ent.ACTION])]
    act_base = n + v_grp.shape[0]
    n = act_base + n_act
    bt = np.array(base, np.int64)

    raw = np.zeros((n, tok.width), np.int32)
    gc, dc = cols[Ent.GLOBAL], cols[Ent.DECISION]
    raw[0, : len(gc)] = glob[gc]
    raw[1, : len(dc)] = glob[dc]
    for r0, a, c in seg_a:
        k = a.shape[0]
        if k and len(c):
            raw[r0 : r0 + k, : len(c)] = a[:, c]
    types = np.repeat(np.array(seg_e, np.int8), seg_n)
    rel = np.full((n, 3), -1, np.int32)

    g_kind, g_sub = int(glob[G.KIND]), int(glob[G.KIND_SUB])
    raw[0, d_kind_sub] = g_kind * 16 + min(max(g_sub, 0), 15)
    cat, sid = int(glob[G.PROMPT_SOURCE_CAT]), int(glob[G.PROMPT_SOURCE_ID])
    if 0 < cat < len(L.ref_off) and L.ref_rows[cat] and sid > 0:
        raw[1, d_prompt] = int(L.ref_off[cat]) + sid
    ns, bs, bcr = cnt[_SEATS], base[_SEATS], base[_CREATURES]
    if ns:
        if n_act:
            raw[bs : bs + ns, d_acts] = np.bincount(acts[:, _A_SEAT], minlength=4)[:ns] > 0
        crow = seats[:, int(SeatCol.CREATURE_ROW)]
        rel[bs : bs + ns, 1] = np.where(crow >= 0, bcr + crow, -1)
        if tok.config.coop_features:
            seat_ids = seats[:, int(SeatCol.SEAT)]
            raw[bs : bs + ns, d_acts] |= ((glob[int(G.SEAT_SCREEN0) + seat_ids] << 1) |
                                         (glob[int(G.SEAT_STATUS0) + seat_ids] << 5))
    if cnt[_CARDS]:
        bc = base[_CARDS]
        sl = slice(bc, bc + cnt[_CARDS])
        ds, dp, dg = d_card
        lv = card_levels
        raw[sl, ds] = st["card"] + cards[:, _K_ID] * lv + np.minimum(cards[:, _K_UP], lv - 1)
        zone, aux = cards[:, _K_ZONE], cards[:, _K_AUX]
        raw[sl, dp] = np.where(zone == _ZONE_CHOICE, aux, 0)
        raw[sl, dg] = np.where(zone == _ZONE_OFFER, aux, -1)
        lr = cards[:, _K_LINK]
        rel[sl, 1] = np.where(lr >= 0, bc + lr, -1)
    if cnt[_CREATURES]:
        sl = slice(bcr, bcr + cnt[_CREATURES])
        dm, dms, dl = d_cre
        kind, mid = cre[:, _CR_KIND], cre[:, _CR_MODEL]
        raw[sl, dm] = _resolve(L.creature_off, L.creature_rows, kind, mid)
        raw[sl, dms] = np.where(kind == _PLAYER, 0, st["monster"] + mid)
        hp, poison = cre[:, _CR_HP], cre[:, _CR_POISON]
        raw[sl, dl] = (poison > 0) & (poison >= hp) & (hp > 0)
    if cnt[_RELICS]:
        b0 = base[_RELICS]
        raw[b0 : b0 + cnt[_RELICS], d_relic] = st["relic"] + T[_RELICS][:, int(RelicCol.RELIC_ID)]
    if cnt[_POTIONS]:
        b0 = base[_POTIONS]
        raw[b0 : b0 + cnt[_POTIONS], d_potion] = st["potion"] + T[_POTIONS][:, int(PotionCol.POTION_ID)]
    if cnt[_NODES]:
        b0 = base[_NODES]
        raw[b0 : b0 + cnt[_NODES], d_drow] = T[_NODES][:, int(MapNodeCol.ROW)] - int(glob[G.CUR_ROW])
    if cnt[_OPTIONS]:
        b0 = base[_OPTIONS]
        sl = slice(b0, b0 + cnt[_OPTIONS])
        raw[sl, d_key] = _resolve(L.opt_key_off, L.opt_key_rows, opts[:, _O_KIND], opts[:, _O_KEY])
        rel[sl, 1] = _ref(bt, opts[:, _O_REF_T], opts[:, _O_REF_R])
        if tok.config.coop_features:
            raw[sl, d[Ent.OPTION]["vote_count"]] = [
                ((int(flags) >> 7) & 15).bit_count() for flags in opts[:, int(OptionCol.FLAGS)]]
    if cnt[_POWERS]:
        b0 = base[_POWERS]
        sl = slice(b0, b0 + cnt[_POWERS])
        dps, dak = d_power
        raw[sl, dps] = st["power"] + pw[:, _P_ID]
        app = pw[:, _P_APPLIER]
        has = app >= 0
        raw[sl, dak] = np.where(has, cre[:, _CR_KIND].take(app, mode="clip"), 0)
        rel[sl, 0] = bcr + pw[:, _P_OWNER]
        rel[sl, 1] = np.where(has, bcr + app, -1)
    if cnt[_INTENTS]:
        b0 = base[_INTENTS]
        rel[b0 : b0 + cnt[_INTENTS], 0] = bcr + it[:, int(IntentCol.OWNER_ROW)]
    if fc is not None and fc.shape[0]:
        rel[fc_base : fc_base + fc.shape[0], 0] = bcr + fc[:, int(ForecastCol.OWNER_ROW)]
    for r0, a, oc, ot in calc_kids:
        rel[r0 : r0 + a.shape[0], 0] = base[ot] + a[:, oc]
    if ahead is not None:
        rel[ahead_row, 0] = 0
    if cnt[_REFS]:
        b0 = base[_REFS]
        sl = slice(b0, b0 + cnt[_REFS])
        c_cat, c_voc, c_ot, c_or, c_et, c_er = ref_c
        raw[sl, d_ref] = _resolve(L.ref_off, L.ref_rows, refs[:, c_cat], refs[:, c_voc])
        rel[sl, 0] = _ref(bt, refs[:, c_ot], refs[:, c_or])
        rel[sl, 1] = _ref(bt, refs[:, c_et], refs[:, c_er])
    if n_var:
        own = _ref(bt, vars_[:, _V_OWNER_T], vars_[:, _V_OWNER_R])
        dst = np.where(var_act >= 0, act_base + var_act, own)
        if tok_var is None:
            rel[var_base : var_base + n_var, 0] = dst
        else:
            bv = base[_VARS]
            rel[bv : bv + int(tok_var.sum()), 0] = own[tok_var]
            rel[var_base:act_base, 0] = dst[~tok_var]
    if n_act:
        sl = slice(act_base, n)
        rel[sl, 0] = _ref(bt, acts[:, _A_SRC_T], acts[:, _A_SRC_R])
        tg = acts[:, _A_TGT]
        rel[sl, 1] = np.where(tg >= 0, bcr + tg, -1)
        seat = acts[:, _A_SEAT]
        rel[sl, 2] = np.where(seat < ns, bs + seat, -1)
        if tok.config.mend_target:
            for action_index, action in enumerate(acts):
                if action[_A_SRC_T] != _OPTIONS:
                    continue
                source = int(action[_A_SRC_R])
                assert 0 <= source < cnt[_OPTIONS]
                option = opts[source]
                if option[_O_KIND] == OptionKind.REST_OPTION and option[_O_KEY] == tok._mend_key:
                    target_seat = int(action[ActionCol.AUX])
                    assert 0 <= target_seat < ns and target_seat != int(action[_A_SEAT])
                    rel[act_base + action_index, 1] = bs + target_seat
    ne = cnt[_EDGES]
    if ne:
        e2 = edges + base[_NODES]
        er = np.empty((2 * ne, 3), np.int16)
        er[:ne, :2], er[ne:, 0], er[ne:, 1] = e2, e2[:, 1], e2[:, 0]
        er[:ne, 2], er[ne:, 2] = 0, 1
    else:
        er = _NO_EDGES

    cond = np.zeros(6, np.int16)
    cond[0], cond[1] = glob[G.ASCENSION], glob[G.N_SEATS]
    k = min(ns, 4)
    cond[2 : 2 + k] = seats[:k, int(SeatCol.CHARACTER_ID)]
    np.minimum(raw, _I16_MAX, out=raw)
    np.maximum(raw, _I16_MIN, out=raw)
    return Sample(types, raw.astype(np.int16), rel.astype(np.int16), er, cond,
                  (n_tok, n_kid, act_base - var_base, n_act))
