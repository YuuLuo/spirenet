"""Brute-force reference of the route summary AHEAD (notes/inputs2/contract.md 2.2, v2a.md 2; gate G4).

Written from the contract, independent of the tokenizers: the ROUTE rows of a frame are recomputed from MAP_NODES,
MAP_EDGES and STATE alone by the ROUTE reference of calc gate PG2 (``spire.calc_ref``: every path from each REACHABLE
node enumerated by depth-first search without memo, the SECOND_BOSS node its own target, deviation D1), then the 30
slots of section 2.2 (+ slot 29 NEXT_QUEST, v2a.md 2) are taken over those rows, one row at a time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .calc_ref import CalcRules, _route
from .proto import Frame
from .schema import MapNodeCol, MapNodeFlag, RouteCol, Table

SLOTS = 30
NAMES = ("N", "NEXT_MONSTER", "NEXT_ELITE", "NEXT_REST", "NEXT_SHOP", "NEXT_TREASURE", "NEXT_UNKNOWN", "NEXT_BOSS",
         "BOSS_ROWS_MIN", "REST_DIST_MIN", "REST_DIST_MAX", "SHOP_DIST_MIN", "SHOP_DIST_MAX", "ELITES_TO_REST_MIN",
         "ELITES_TO_REST_MAX", "FIGHTS_TO_REST_MIN", "FIGHTS_TO_REST_MAX", "ELITES_TO_BOSS_MIN", "ELITES_TO_BOSS_MAX",
         "FIGHTS_TO_BOSS_MIN", "FIGHTS_TO_BOSS_MAX", "N_REST_MIN", "N_REST_MAX", "N_SHOP_MIN", "N_SHOP_MAX",
         "N_ELITE_MIN", "N_ELITE_MAX", "N_UNKNOWN_MIN", "N_UNKNOWN_MAX", "NEXT_QUEST")
assert len(NAMES) == SLOTS


class AheadRules(CalcRules):
    """The MapPointType ids the ROUTE reference and section 2.2 name (from vocab.json; a missing name raises)."""

    def __init__(self, vocab_path: Path) -> None:  # noqa: D107 - only the map names of CalcRules are needed
        v = json.loads(Path(vocab_path).read_text(encoding="utf-8"))["categories"]
        entries = list(v["enum:MapPointType"]["entries"])
        self.rest, self.shop, self.elite = (entries.index(n) for n in ("RestSite", "Shop", "Elite"))
        self.monster, self.unknown, self.treasure = (entries.index(n) for n in ("Monster", "Unknown", "Treasure"))


def route_reference(f: Frame, rules: AheadRules) -> NDArray[np.int64]:
    """The ROUTE rows of frame ``f`` recomputed from its map tables (calc contract 1.7 with D1)."""
    return _route(f, rules)


def ahead_reference(f: Frame, rules: AheadRules) -> list[int]:
    """The 30 AHEAD slots of frame ``f`` (contract 2.2 + v2a.md 2) over the brute-force ROUTE rows."""
    route = route_reference(f, rules)
    nodes = cast(NDArray[np.int32], f.table(Table.MAP_NODES))
    rows = [[int(x) for x in r] for r in route]
    ty = [int(nodes[r[RouteCol.OWNER_ROW], MapNodeCol.POINT_TYPE]) for r in rows]
    fl = [int(nodes[r[RouteCol.OWNER_ROW], MapNodeCol.FLAGS]) for r in rows]
    n = len(rows)

    def mn(col: int) -> int:  # MINV: min over the rows with a value >= 0; -1 if there is none
        best = -1
        for r in rows:
            if r[col] >= 0 and (best == -1 or r[col] < best):
                best = r[col]
        return best

    def mx(col: int) -> int:  # MAXV: -1 if there is no row or some row has -1, else the max
        if n == 0 or any(r[col] == -1 for r in rows):
            return -1
        best = rows[0][col]
        for r in rows:
            best = max(best, r[col])
        return best

    out = [n,
           sum(t == rules.monster for t in ty), sum(t == rules.elite for t in ty), sum(t == rules.rest for t in ty),
           sum(t == rules.shop for t in ty), sum(t == rules.treasure for t in ty), sum(t == rules.unknown for t in ty),
           sum(1 for x in fl if x & MapNodeFlag.BOSS or x & MapNodeFlag.SECOND_BOSS),
           mn(RouteCol.ROWS_TO_BOSS)]
    for col in (RouteCol.REST_DIST, RouteCol.SHOP_DIST, RouteCol.ELITES_TO_REST, RouteCol.FIGHTS_TO_REST,
                RouteCol.ELITES_TO_BOSS, RouteCol.FIGHTS_TO_BOSS, RouteCol.N_REST, RouteCol.N_SHOP, RouteCol.N_ELITE,
                RouteCol.N_UNKNOWN):
        out += [mn(col), mx(col)]
    out.append(sum(1 for x in fl if x & MapNodeFlag.QUEST))
    assert len(out) == SLOTS
    return out
