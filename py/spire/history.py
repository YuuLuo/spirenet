"""Bounded public decision history, maintained by each actor environment.

No frame reconstruction or host state: the five descriptors are copied from the
previous stored Sample. Reward positions are the public, original set indices
(not transient option row indices). No unrevealed card identities are retained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from struct import Struct

import numpy as np
from numpy.typing import NDArray

from .proto import Frame
from .schema import N_COLS, WORD, ActionCol, G, Kind, OptionCol, OptionKind, Table, Verb
from .tokenize import HISTORY_ROWS, HISTORY_STATS, Sample, history_width

Offer = tuple[int, int, int]  # seat, original position, public reward type
_ACTION = Struct("<" + "i" * N_COLS[Table.ACTIONS])


def _options(frame: Frame) -> NDArray[np.int32]:
    # Elided tables have already been resolved by Host. The ordinary path avoids
    # constructing all sixteen table views merely to read one reward menu.
    n, offset = frame.rows[Table.OPTIONS], frame.offsets[Table.OPTIONS]
    if n < 0:
        return frame.table(Table.OPTIONS)
    return frame.words[offset:offset + n * N_COLS[Table.OPTIONS]].reshape(n, N_COLS[Table.OPTIONS])


@dataclass
class _RewardSet:
    signature: frozenset[Offer]
    opened: set[Offer] = field(default_factory=set)
    declined: set[Offer] = field(default_factory=set)
    last_opened: int = 0
    last_declined: int = 0


def _offer(row: NDArray[np.int32]) -> Offer:
    return int(row[OptionCol.SEAT]), int(row[OptionCol.POS]), int(row[OptionCol.KEY_ID])


class PublicHistory:
    """One room of history; ``observe`` precedes inference, ``record`` follows it.

    A skipped host decision has no frame/action payload. ``auto_steps`` records
    that gap; previous decision always means previous *observed* policy decision.
    Sets nested through reward choices are matched by visible remaining options.
    Indistinguishable nested sets cannot be identified without additional public
    observations; no hidden set identifier is inferred.
    """

    def __init__(self, width: int) -> None:
        self.width = width
        self.clear()

    def clear(self) -> None:
        self.data = np.zeros(history_width(self.width), np.int16)
        self.stats = self.data[-HISTORY_STATS:]
        self.types = self.data[:HISTORY_ROWS]
        self.raw = self.data[HISTORY_ROWS:-HISTORY_STATS].reshape(HISTORY_ROWS, self.width)
        self.room: tuple[int, ...] | None = None
        self.observed: tuple[int, int] | None = None
        self.sets: list[_RewardSet] = []
        self.pending: Offer | None = None
        self.card_offer: Offer | None = None

    def fork(self) -> "PublicHistory":
        """An independent tracker in this one's state between two decisions (a fork child of the game, boss gym)."""
        other = PublicHistory(self.width)
        other.data[:] = self.data
        other.room = self.room
        other.observed = self.observed
        other.sets = [_RewardSet(s.signature, set(s.opened), set(s.declined), s.last_opened, s.last_declined)
                      for s in self.sets]
        other.pending = self.pending
        other.card_offer = self.card_offer
        return other

    def observe(self, frame: Frame) -> NDArray[np.int16]:
        g = frame.gvals
        room = (frame.episode_id, g[G.ACT_INDEX], g[G.TOTAL_FLOOR], g[G.CUR_ROW],
                g[G.CUR_COL], g[G.ROOM_TYPE])
        if room != self.room:
            self.clear()
            self.room = room
        self.observed = (frame.episode_id, frame.step_id)
        kind = g[G.KIND]
        if kind == Kind.REWARD:
            signature = frozenset(_offer(r) for r in _options(frame)
                                  if r[OptionCol.KIND] == OptionKind.REWARD_ITEM)
            # Taken rewards disappear without renumbering. Search from the inner
            # set outwards so returning from a distinct nested set restores it.
            match = next((i for i in range(len(self.sets) - 1, -1, -1)
                          if signature <= self.sets[i].signature), None)
            if match is None:
                self.sets.append(_RewardSet(signature))
            else:
                del self.sets[match + 1:]
            # Ordinary SKIP is a CARD_REWARD_ALT option in real frames (some
            # clients also use Verb.SKIP). Returning with the same reward still
            # available is the public evidence of declining either spelling.
            if self.card_offer is not None and self.card_offer in signature:
                self.sets[-1].declined.add(self.card_offer)
                self.sets[-1].last_declined = self.card_offer[1] + 1
            self.card_offer = None
        elif kind == Kind.CARD_REWARD and self.pending is not None:
            assert self.sets
            self.card_offer = self.pending
            self.sets[-1].opened.add(self.pending)
            self.sets[-1].last_opened = self.pending[1] + 1
        elif kind not in (Kind.CARD_REWARD, Kind.CARD_SELECT, Kind.CONFIRM, Kind.BUNDLE):
            self.sets.clear()
            self.card_offer = None
        self.pending = None
        stats = self.stats
        assert 0 <= frame.auto_steps <= 32767
        stats[2] = frame.auto_steps
        stats[15:19] = 0
        if self.sets:
            s = self.sets[-1]
            stats[15:19] = len(s.opened), len(s.declined), s.last_opened, s.last_declined
        return self.data.copy()

    def record(self, frame: Frame, sample: Sample, action: int) -> bool:
        assert self.observed == (frame.episode_id, frame.step_id)
        if frame.rows[Table.ACTIONS] < 0:
            row = tuple(int(v) for v in frame.table(Table.ACTIONS)[action])
        else:
            row = _ACTION.unpack_from(frame.payload, frame.offsets[Table.ACTIONS] * WORD + action * _ACTION.size)
        kind, sub = frame.gvals[G.KIND], frame.gvals[G.KIND_SUB]
        verb = int(row[ActionCol.VERB])
        reopened = False
        if kind == Kind.REWARD and verb == Verb.CHOOSE_OPTION:
            assert row[ActionCol.SRC_TABLE] == Table.OPTIONS
            option = _options(frame)[row[ActionCol.SRC_ROW]]
            if option[OptionCol.KIND] == OptionKind.REWARD_ITEM:
                self.pending = _offer(option)
                assert self.sets
                reopened = self.pending in self.sets[-1].declined
        elif kind == Kind.REWARD and verb == Verb.PROCEED:
            self.sets.pop()
            self.card_offer = None

        # Copy semantic descriptors before the current frame and its row indices
        # disappear. Null pointers become PAD, never a row from the next frame.
        a = sum(sample.counts[:3]) + action
        src, target, seat = (int(x) for x in sample.rel[a])
        related = int(sample.rel[src, 1]) if src >= 0 else -1
        types, raw = self.types, self.raw
        for i, index in enumerate((a, src, related, target, seat)):
            if index < 0:
                types[i] = 0
                raw[i] = 0
            else:
                types[i] = sample.types[index]
                raw[i] = sample.raw[index]
        stats = self.stats
        stats[0:2] = kind, sub
        assert 1 <= kind <= 12
        count = int(stats[2 + kind]) + 1
        assert count <= 32767, "room history exceeds supported decision count"
        stats[2 + kind] = count
        assert all(-32768 <= v <= 32767 for v in row)
        stats[19:26] = row
        return reopened
