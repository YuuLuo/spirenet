"""Brute-force reference of the calc tables TURN / FOE / ROUTE (notes/calc/contract.md 1, gate PG2).

Recomputes the three trailer tables of a frame from its v1 tables, ``vocab.json`` and ``catalog.json`` alone,
written from the contract and independent of the host's method:

* 1.2, 1.3, 1.5, 1.6: the formulas, one row at a time;
* 1.4: ``itertools.combinations`` over ALL subsets of the paying candidates (after the cap rule), no dynamic
  programming;
* 1.7: explicit enumeration of every path from a node by depth-first search (no memo); minima and reachable sets
  are taken over the enumerated paths.

The vocab and catalog are read as plain JSON here (not through spire.vocab), names resolved once per file.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from .proto import Frame
from .schema import (
    CALC_MAX_ITEMS,
    CALC_N_COLS,
    CardCol,
    CostFlag,
    CreatureCol,
    CreatureKind,
    FoeFlag,
    G,
    HandFlag,
    IntentCol,
    ItemZone,
    MapEdgeCol,
    MapNodeCol,
    MapNodeFlag,
    MapNodeState,
    PowerCol,
    RelicCol,
    SeatCol,
    Table,
    TurnFlag,
    VarCol,
    VarFlag,
    Zone,
)

I64 = NDArray[np.int64]


class CalcRules:
    """Vocab ids and catalog constants named by notes/calc/contract.md 1 (resolved once; a missing name raises)."""

    def __init__(self, vocab_path: Path, catalog_path: Path) -> None:
        v = json.loads(Path(vocab_path).read_text(encoding="utf-8"))["categories"]
        c = json.loads(Path(catalog_path).read_text(encoding="utf-8"))["categories"]

        def vid(cat: str, name: str) -> int:
            return list(v[cat]["entries"]).index(name)

        self.intent_unknown = {vid("enum:IntentType", "Hidden"), vid("enum:IntentType", "Unknown")}
        self.side_player = vid("enum:CombatSide", "Player")
        self.plating = vid("power", "PLATING_POWER")
        self.constrict = vid("power", "CONSTRICT_POWER")
        self.reduced_powers = {vid("power", "BUFFER_POWER"), vid("power", "INTANGIBLE_POWER")}
        self.guarded_powers = {vid("power", n) for n in
                               ("HARDENED_SHELL_POWER", "SLIPPERY_POWER", "BUFFER_POWER", "SKITTISH_POWER")}
        self.orichalcum = vid("relic", "ORICHALCUM")
        self.reduced_relics = {vid("relic", "TUNGSTEN_ROD"), vid("relic", "BEATING_REMNANT")}
        self.orichalcum_block = int(c["relic"][str(self.orichalcum)]["vars"]["Block"]["base"])
        self.var = {n: vid("var_name", n) for n in ("Damage", "CalculatedDamage", "HitCount", "CalculatedHits",
                                                     "Block", "CalculatedBlock", "HpLoss")}
        pt = "enum:MapPointType"
        self.rest, self.shop, self.elite = vid(pt, "RestSite"), vid(pt, "Shop"), vid(pt, "Elite")
        self.monster, self.unknown = vid(pt, "Monster"), vid(pt, "Unknown")
        self.cards: dict[int, list[dict[str, Any]]] = {int(k): lv for k, lv in c["card"].items()}

    def card_level(self, card_id: int, upgrade: int) -> dict[str, Any]:
        lv = self.cards[card_id]
        return lv[min(upgrade, len(lv) - 1)]


def _rows(f: Frame, t: Table) -> NDArray[np.int32]:
    return f.table(t)


def reference_tables(f: Frame, rules: CalcRules) -> tuple[I64, I64, I64]:
    """(TURN, FOE, ROUTE) of frame ``f`` as the contract defines them (int64 arrays, table column order)."""
    turn, foe = _combat(f, rules)
    return turn, foe, _route(f, rules)


# --------------------------------------------------------------------------- combat (1.1-1.6)


def _combat(f: Frame, R: CalcRules) -> tuple[I64, I64]:
    g = f.glob
    seats, cre, cards = _rows(f, Table.SEATS), _rows(f, Table.CREATURES), _rows(f, Table.CARDS)
    vars_, pw, it, rel = _rows(f, Table.VARS), _rows(f, Table.POWERS), _rows(f, Table.INTENTS), _rows(f, Table.RELICS)
    s = int(g[G.ACTING_SEAT])
    empty = (np.zeros((0, CALC_N_COLS[0]), np.int64), np.zeros((0, CALC_N_COLS[1]), np.int64))
    if not (int(g[G.IN_COMBAT]) == 1 and 0 <= s < len(seats)):
        return empty
    P = int(seats[s, SeatCol.CREATURE_ROW])
    if not (P >= 0 and int(cre[P, CreatureCol.ALIVE]) == 1 and int(seats[s, SeatCol.PRIVATE_VISIBLE]) == 1):
        return empty
    foes = [e for e in range(len(cre))
            if int(cre[e, CreatureCol.KIND]) == CreatureKind.ENEMY and int(cre[e, CreatureCol.ALIVE]) == 1]

    def powers_of(owner: int) -> list[tuple[int, int]]:
        return [(int(r[PowerCol.POWER_ID]), int(r[PowerCol.AMOUNT])) for r in pw if int(r[PowerCol.OWNER_ROW]) == owner]

    def power_amount(owner: int, pid: int) -> int:
        a = [amt for p, amt in powers_of(owner) if p == pid]
        assert len(a) <= 1, f"two POWERS rows {pid} on creature {owner}"
        return a[0] if a else 0

    def owns_relic(rid: int) -> bool:
        return any(int(r[RelicCol.SEAT]) == s and int(r[RelicCol.ZONE]) == ItemZone.OWNED
                   and int(r[RelicCol.RELIC_ID]) == rid for r in rel)

    # 1.2 incoming
    dmg_col = int(IntentCol.DMG_SEAT0) + s
    inc_from: dict[int, int] = {}
    unknown = False
    for e in foes:
        tot = 0
        for r in it:
            if int(r[IntentCol.OWNER_ROW]) != e:
                continue
            if int(r[IntentCol.INTENT_TYPE]) in R.intent_unknown:
                unknown = True
            d = int(r[dmg_col])
            if d >= 0:
                tot += d * int(r[IntentCol.HITS])
        inc_from[e] = tot
    incoming = sum(inc_from.values())

    # card variables (untargeted rows: TARGET_ROW -1; never ENCHANTMENT rows)
    def card_vars(row: int) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
        un: dict[int, int] = {}
        tg: dict[tuple[int, int], int] = {}
        for v in vars_:
            if (int(v[VarCol.OWNER_TABLE]) != Table.CARDS or int(v[VarCol.OWNER_ROW]) != row
                    or int(v[VarCol.FLAGS]) & VarFlag.ENCHANTMENT):
                continue
            k, t = int(v[VarCol.VAR_ID]), int(v[VarCol.TARGET_ROW])
            if t < 0:
                assert k not in un, f"two untargeted VARS rows {k} of card row {row}"
                un[k] = int(v[VarCol.SHOWN])
            else:
                tg[(k, t)] = int(v[VarCol.SHOWN])
        return un, tg

    hand = [c for c in range(len(cards)) if int(cards[c, CardCol.ZONE]) == Zone.HAND and int(cards[c, CardCol.SEAT]) == s]
    cv = {c: card_vars(c) for c in hand}

    # 1.3 turn end
    player_side = int(g[G.COMBAT_SIDE]) == R.side_player
    block_now = int(cre[P, CreatureCol.BLOCK])
    plating = power_amount(P, R.plating) if player_side else 0
    ori = R.orichalcum_block if player_side and owns_relic(R.orichalcum) and block_now == 0 else 0
    end_block = plating + ori
    end_dmg = end_loss = 0
    if player_side:
        end_dmg = power_amount(P, R.constrict)
        for c in hand:
            lvl = R.card_level(int(cards[c, CardCol.CARD_ID]), int(cards[c, CardCol.UPGRADE_LEVEL]))
            if not lvl["has_turn_end_in_hand_effect"]:
                continue
            un = cv[c][0]
            end_dmg += un.get(R.var["Damage"], 0)
            end_loss += un.get(R.var["HpLoss"], 0)

    # 1.4 candidates
    E = int(seats[s, SeatCol.ENERGY])
    assert E >= 0, f"seat {s} energy {E}"
    ST = max(int(seats[s, SeatCol.STARS]), 0)
    hittable = {e: int(cre[e, CreatureCol.HITTABLE]) == 1 for e in foes}
    n_hittable = sum(hittable.values())
    H: list[dict[str, Any]] = []
    for c in hand:
        cf = int(cards[c, CardCol.COST_FLAGS])
        if not int(cards[c, CardCol.HAND_FLAGS]) & HandFlag.CAN_PLAY or cf & CostFlag.STAR_X:
            continue
        is_x = bool(cf & CostFlag.COSTS_X)
        cost = 0 if is_x else int(cards[c, CardCol.ENERGY_COST])
        assert cost >= 0, f"playable card row {c} with energy cost {cost}"
        tt = R.card_level(int(cards[c, CardCol.CARD_ID]), int(cards[c, CardCol.UPGRADE_LEVEL]))["target_type"]
        H.append({"row": c, "pos": int(cards[c, CardCol.POS]), "cost": cost,
                  "star": max(int(cards[c, CardCol.STAR_COST]), 0), "x": is_x, "tt": tt})

    def var(c: int, name: str, e: int = -1) -> int | None:
        un, tg = cv[c]
        k = R.var[name]
        if k not in un:
            return None
        return max(tg.get((k, e), un[k]) if e >= 0 else un[k], 0)

    def first(*vals: int | None, default: int) -> int:
        for x in vals:
            if x is not None:
                return x
        return default

    def block(h: dict[str, Any]) -> int:
        if h["x"] or h["tt"] in ("AnyAlly", "AllAllies", "AnyPlayer"):
            return 0
        return first(var(h["row"], "CalculatedBlock"), var(h["row"], "Block"), default=0)

    def per(h: dict[str, Any], e: int) -> int:
        return first(var(h["row"], "CalculatedDamage", e), var(h["row"], "Damage", e), default=0)

    def hits_var(h: dict[str, Any]) -> int | None:
        a = var(h["row"], "CalculatedHits")
        return a if a is not None else var(h["row"], "HitCount")

    def aims(h: dict[str, Any], e: int) -> bool:
        return hittable[e] and (h["tt"] in ("AnyEnemy", "AllEnemies") or (h["tt"] == "RandomEnemy" and n_hittable == 1))

    def best(v: Any, xval: Any, xkey: Any) -> tuple[int, bool]:
        """Brute-force maximum of 1.4 for value function ``v`` (``xval(h, energy_left)`` for an X card, ``xkey(h)``
        its per-hit value; ``xkey`` None: X cards never qualify)."""
        x_ok = xkey is not None
        free = [h for h in H if v(h) > 0 and h["cost"] == 0 and h["star"] == 0 and not h["x"]]
        fid = {id(h) for h in free}
        paying = [h for h in H if (v(h) > 0 or (x_ok and h["x"] and xkey(h) > 0)) and id(h) not in fid]
        capped = len(paying) > CALC_MAX_ITEMS
        if capped:
            paying = sorted(paying, key=lambda h: (-(xkey(h) * E if h["x"] else v(h)), h["pos"]))[:CALC_MAX_ITEMS]
        top = 0
        for k in range(len(paying) + 1):
            for S in combinations(paying, k):
                if sum(h["cost"] for h in S) > E or sum(h["star"] for h in S) > ST or sum(h["x"] for h in S) > 1:
                    continue
                left = E - sum(h["cost"] for h in S)
                tot = sum(v(h) for h in S if not h["x"]) + sum(xval(h, left) for h in S if h["x"])
                top = max(top, tot)
        return sum(v(h) for h in free) + top, capped

    block_reach, block_capped = best(block, None, None)

    foe_rows: list[list[int]] = []
    for e in foes:
        def dmg(h: dict[str, Any], e: int = e) -> int:
            hv = hits_var(h)
            return per(h, e) * (hv if hv is not None else 1) if aims(h, e) else 0

        def xval(h: dict[str, Any], left: int, e: int = e) -> int:
            if not aims(h, e):
                return 0
            hv = hits_var(h)
            return per(h, e) * (hv if hv is not None else left)

        reach, capped = best(dmg, xval, lambda h, e=e: per(h, e)) if hittable[e] else (0, False)
        hp, blk = int(cre[e, CreatureCol.HP]), int(cre[e, CreatureCol.BLOCK])
        to_kill = hp + blk if hp >= 0 else -1
        guarded = any(p in R.guarded_powers for p, _ in powers_of(e))
        kill = to_kill > 0 and reach >= to_kill and not guarded
        fl = (int(FoeFlag.KILL) * kill | int(FoeFlag.GUARDED) * guarded | int(FoeFlag.HP_HIDDEN) * (hp < 0)
              | int(FoeFlag.CAPPED) * capped | int(FoeFlag.UNHITTABLE) * (not hittable[e]))
        foe_rows.append([e, to_kill, reach, inc_from[e], fl])

    # 1.5 TURN row
    hp = int(cre[P, CreatureCol.HP])
    assert hp >= 0
    T = end_dmg + incoming
    a_now = block_now + end_block
    a_best = max(a_now, block_now + plating + block_reach)
    loss_now = max(0, T - a_now) + end_loss
    loss_min = max(0, T - a_best) + end_loss
    reduced = (any(p in R.reduced_powers for p, _ in powers_of(P)) or any(owns_relic(r) for r in R.reduced_relics)
               or any(int(r[CreatureCol.KIND]) == CreatureKind.PET and int(r[CreatureCol.SEAT]) == s
                      and int(r[CreatureCol.ALIVE]) == 1 for r in cre))
    fl = (int(TurnFlag.LETHAL_NOW) * (loss_now >= hp) | int(TurnFlag.LETHAL_UNAVOIDABLE) * (loss_min >= hp)
          | int(TurnFlag.UNKNOWN_INCOMING) * unknown | int(TurnFlag.REDUCED) * reduced
          | int(TurnFlag.CAPPED) * block_capped)
    turn = [[P, hp, block_now, incoming, end_block, end_dmg, end_loss, loss_now, block_reach, loss_min, fl]]
    return np.array(turn, np.int64), np.array(foe_rows, np.int64).reshape(-1, CALC_N_COLS[1])


# --------------------------------------------------------------------------- route (1.7)


def _route(f: Frame, R: CalcRules) -> I64:
    nodes, edges = _rows(f, Table.MAP_NODES), _rows(f, Table.MAP_EDGES)
    if not len(nodes):
        return np.zeros((0, CALC_N_COLS[2]), np.int64)
    n = len(nodes)
    flags = [int(r[MapNodeCol.FLAGS]) for r in nodes]
    bosses = [i for i in range(n) if flags[i] & MapNodeFlag.BOSS]
    assert len(bosses) == 1, f"{len(bosses)} BOSS nodes"
    boss = bosses[0]
    second = [i for i in range(n) if flags[i] & MapNodeFlag.SECOND_BOSS]
    assert len(second) <= 1
    succ: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges[:, [MapEdgeCol.FROM_NODE_ROW, MapEdgeCol.TO_NODE_ROW]]:
        if int(b) not in succ[int(a)]:
            succ[int(a)].append(int(b))
    for m in range(n):
        if not succ[m] and m != boss and m not in second:
            succ[m].append(boss)
    if second and second[0] not in succ[boss]:
        succ[boss].append(second[0])
    special = {boss, *second}
    ptype = [int(r[MapNodeCol.POINT_TYPE]) for r in nodes]

    def paths(start: int) -> Iterator[list[int]]:
        stack: list[tuple[int, list[int]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            assert len(path) <= n, "map graph has a cycle"
            if not succ[node]:
                yield path
                continue
            for nx in succ[node]:
                stack.append((nx, path + [nx]))

    out: list[list[int]] = []
    for row in range(n):
        if int(nodes[row, MapNodeCol.STATE]) != MapNodeState.REACHABLE:
            continue
        to_boss = rest_d = shop_d = e_rest = f_rest = e_boss = f_boss = None
        seen: set[int] = set()
        # deviations.md P5: the SECOND_BOSS node, reachable once the first boss is CURRENT, is its own target
        target = row if row in second else boss
        for p in paths(row):
            assert target in p, f"BOSS not reachable from node {row}"
            seen.update(p)
            ib = p.index(target)
            to_boss = ib if to_boss is None else min(to_boss, ib)
            pre = p[:ib]
            eb = sum(ptype[i] == R.elite for i in pre)
            fb = sum(ptype[i] in (R.elite, R.monster) for i in pre)
            e_boss = eb if e_boss is None else min(e_boss, eb)
            f_boss = fb if f_boss is None else min(f_boss, fb)
            rests = [k for k, i in enumerate(p) if ptype[i] == R.rest]
            if rests:
                k = rests[0]
                rest_d = k if rest_d is None else min(rest_d, k)
                er = sum(ptype[i] == R.elite for i in p[:k])
                fr = sum(ptype[i] in (R.elite, R.monster) for i in p[:k])
                e_rest = er if e_rest is None else min(e_rest, er)
                f_rest = fr if f_rest is None else min(f_rest, fr)
            shops = [k for k, i in enumerate(p) if ptype[i] == R.shop]
            if shops:
                shop_d = shops[0] if shop_d is None else min(shop_d, shops[0])
        cnt = seen - special
        out.append([row, int(to_boss), -1 if rest_d is None else rest_d, -1 if shop_d is None else shop_d,
                    -1 if e_rest is None else e_rest, -1 if f_rest is None else f_rest, int(e_boss), int(f_boss),
                    sum(ptype[i] == R.rest for i in cnt), sum(ptype[i] == R.shop for i in cnt),
                    sum(ptype[i] == R.elite for i in cnt), sum(ptype[i] == R.unknown for i in cnt)])
    return np.array(out, np.int64).reshape(-1, CALC_N_COLS[2])
