"""The removal rule (notes/solo/removal-spec.md): the TARGET of an out-of-combat card removal is fixed rule knowledge.

At every CARD_SELECT frame with KIND_SUB = DECK_REMOVAL outside combat the legal set is reduced to ONE row:

1. while SEL_COUNT < SEL_MAX and a SELECT_CARD row exists: the select row of the first offered card in the order
   a. a curse (several: lowest card id first);
   b. a starter Strike of the acting seat's character (not upgraded before upgraded);
   c. a starter Defend of that character (not upgraded before upgraded);
   d. the lowest rating of the table (ties: not upgraded first, then the lowest card id);
   remaining ties (identical id and upgrade level, e.g. an enchanted copy): the lower action row;
2. otherwise the CONFIRM_SELECTION row.

Only the prompt's own rows are candidates (cards it does not offer are never targets); no policy score enters.
The table (``removal-ratings.json``, built by :func:`build_table` / scripts/build_removal_ratings.py) holds the
ratings and the card classes read from the catalog; every card the prompt offers must have a rating.
The mask itself lives in :class:`spire.cycle.CycleBan` (``removal_rule``), next to ``commit_select``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .proto import Frame
from .schema import ActionCol, CardCol, CardSelectSub, G, Kind, SeatCol, Table, Verb
from .vocab import Catalog, Vocab

TABLE_VERSION = 1
CLASSES = ("curse", "strike", "defend", "other")  # rule order a-d; index = class rank
_CURSE, _STRIKE, _DEFEND, _OTHER = range(4)


def is_removal_prompt(frame: Frame) -> bool:
    g = frame.gvals
    return g[G.KIND] == Kind.CARD_SELECT and g[G.KIND_SUB] == CardSelectSub.DECK_REMOVAL and not g[G.IN_COMBAT]


def read_tsv(texts: Iterable[tuple[str, str]]) -> dict[str, int]:
    """(source name, TSV text) pairs -> card name -> rating; a card rated twice must have one rating."""
    out: dict[str, int] = {}
    for src, text in texts:
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, rating = line.split("\t")[:2]
            assert out.get(name, int(rating)) == int(rating), f"{src}: {name} rated {rating} and {out[name]}"
            out[name] = int(rating)
    return out


def build_table(vocab: Vocab, catalog: Catalog, ratings: Mapping[str, int]) -> dict[str, Any]:
    """The table object: every vocab card that can be in a deck (a catalog entry whose type is not Status) is rated.

    Ratings of names the vocabulary does not have are dropped and listed under ``dropped``."""
    names = vocab["card"].entries
    cards = catalog.entries["card"]
    deckable = [names[i] for i, levels in sorted(cards.items()) if levels[0]["type"] != "Status"]
    missing = [n for n in deckable if n not in ratings]
    assert not missing, f"cards without a removal rating: {missing}"
    curses = [names[i] for i, levels in sorted(cards.items()) if levels[0]["type"] == "Curse"]
    strike: dict[str, list[str]] = {}
    defend: dict[str, list[str]] = {}
    for i, levels in sorted(cards.items()):
        e = levels[0]
        if e["is_basic_strike_or_defend"]:
            assert ("Strike" in e["tags"]) != ("Defend" in e["tags"]), (names[i], e["tags"])
            assert e["character"] in vocab["character"].index, (names[i], e["character"])
            (strike if "Strike" in e["tags"] else defend).setdefault(e["character"], []).append(names[i])
    assert strike and sorted(strike) == sorted(defend), (strike, defend)
    index = vocab["card"].index
    return {"version": TABLE_VERSION, "vocab_hash": vocab.hash, "game_version": vocab.game_version,
            "ratings": {n: int(r) for n, r in sorted(ratings.items(), key=lambda kv: index.get(kv[0], 1 << 30))
                        if n in index},
            "curses": curses, "strike": strike, "defend": defend,
            "dropped": sorted(n for n in ratings if n not in index)}


class RemovalRule:
    """The rule's choice for one removal prompt (:meth:`choose`); built from the table object."""

    def __init__(self, vocab: Vocab, table: Mapping[str, Any]) -> None:
        assert table["version"] == TABLE_VERSION, table["version"]
        assert table["vocab_hash"] == vocab.hash, "removal table built for another vocabulary"
        index = vocab["card"].index
        n = vocab.size("card")

        def cid(name: str) -> int:
            assert name in index, f"removal table card {name} not in the vocabulary"
            return index[name]

        self.rating = np.full(n, -1, np.int64)
        for name, r in table["ratings"].items():
            self.rating[cid(name)] = int(r)
        base = np.full(n, _OTHER, np.int64)
        base[[cid(c) for c in table["curses"]]] = _CURSE
        # character vocab id -> card id -> class rank (starter cards of other characters stay "other")
        self.classes: dict[int, np.ndarray] = {}
        for char in table["strike"]:
            cls = base.copy()
            cls[[cid(c) for c in table["strike"][char]]] = _STRIKE
            cls[[cid(c) for c in table["defend"][char]]] = _DEFEND
            self.classes[vocab.id("character", char)] = cls
        self.names = vocab["card"].entries
        self.char_names = vocab["character"].entries

    def choose(self, frame: Frame) -> tuple[int, str | None]:
        """(the one legal row, the class name of the selected card or None for the CONFIRM row)."""
        assert is_removal_prompt(frame)
        g = frame.gvals
        acts = frame.actions
        verb = acts[:, ActionCol.VERB]
        sel = np.flatnonzero(verb == Verb.SELECT_CARD)
        if g[G.SEL_COUNT] < g[G.SEL_MAX] and len(sel):
            assert (acts[sel, ActionCol.SRC_TABLE] == Table.CARDS).all(), "SELECT_CARD rows name CARDS rows"
            seats = frame.table(Table.SEATS)
            seat = seats[seats[:, SeatCol.SEAT] == g[G.ACTING_SEAT]]
            assert len(seat) == 1, "the acting seat has one SEATS row"
            char = int(seat[0, SeatCol.CHARACTER_ID])
            assert char in self.classes, f"no starter cards for character {self.char_names[char]}"
            cards = frame.table(Table.CARDS)[acts[sel, ActionCol.SRC_ROW]]
            ids = cards[:, CardCol.CARD_ID].astype(np.int64)
            rating = self.rating[ids]
            assert (rating >= 0).all(), f"offered cards without a rating: {[self.names[i] for i in ids[rating < 0]]}"
            cls = self.classes[char][ids]
            upgraded = np.where(cls == _CURSE, 0, cards[:, CardCol.UPGRADE_LEVEL])
            by_rating = np.where(cls == _OTHER, rating, 0)
            first = int(np.lexsort((sel, ids, upgraded, by_rating, cls))[0])
            return int(sel[first]), CLASSES[int(cls[first])]
        confirm = np.flatnonzero(verb == Verb.CONFIRM_SELECTION)
        assert len(confirm) == 1, (f"removal prompt: no row for the rule (SEL_COUNT {g[G.SEL_COUNT]}, "
                                   f"SEL_MAX {g[G.SEL_MAX]}, verbs {verb.tolist()})")
        return int(confirm[0]), None


def load_rule(vocab: Vocab, path: str | Path) -> RemovalRule:
    return RemovalRule(vocab, json.loads(Path(path).read_text(encoding="utf-8")))
