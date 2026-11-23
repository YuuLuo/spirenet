"""Synthetic frames and a fake environment.

* :class:`FrameBuilder` - programmatic encoder of valid v1 frames (directory, offsets,
  FNV-1a hashes, CSR slices, canonical VARS / ACTIONS order, multiset grouping). Also meant
  as a readable reference encoder for the C# side.
* :class:`FrameGen` - random valid frames for all 12 decision kinds, 1-4 seats, TEAM/SEAT
  views, ``"typical"`` sizes (schema 11 statistics) or ``"extreme"`` sizes (near the game /
  v1 limits), with consistent cross-table references.
* :class:`FakeGame` - bytes-level fake SpireHost (command -> response payloads) producing
  plausible episodes; :class:`FakeHost` - the in-process :class:`~spire.host.Host` over it.
  ``scripts/fake_spirehost.py`` serves the same engine over pipes.
"""
from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Generator, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .host import Host, HostStateError, HostTimeout
from .proto import (
    Command,
    ProtocolError,
    QuitCmd,
    ResetCmd,
    StepCmd,
    card_content_key,
    decode_command,
    fnv1a32,
    parse_frame,
)
from .schema import (
    ACTION_SORT_COLS,
    CALC_MAX_ROWS,
    CALC_N_COLS,
    CALC_TABLES,
    COLUMN_INDEX,
    DIR_STRIDE,
    DYNAMIC_POINTERS,
    FORECAST_MAX_CAND,
    FORECAST_MAX_ROWS,
    FORECAST_N_COLS,
    FORECAST_PROB_ONE,
    FORECAST_TABLE,
    FRAME_MAGIC,
    HDRAW_MAX_ROWS,
    HDRAW_N_COLS,
    HDRAW_TABLE,
    G_ERROR_CODE,
    GLOBAL_LEN,
    GLOBAL_OFFSET,
    HEADER_LEN,
    MAX_ROWS,
    N_COLS,
    N_TABLES,
    N_TABLES_FORECAST,
    POINTERS,
    RESET_FLAGS_DEFAULT,
    SCHEMA_VERSION,
    SLICES,
    WORD,
    SliceSpec,
    ActionCol,
    ArrowCol,
    BundleFlag,
    CardCol,
    CardRewardFlag,
    CardSelectSub,
    CellCol,
    CostFlag,
    CreatureCol,
    CreatureKind,
    ErrorCode,
    EventFlag,
    ForecastClass,
    ForecastCol,
    ForecastFlag,
    FrameFlag,
    G,
    H,
    HandFlag,
    IntentCol,
    IntentFlag,
    ItemZone,
    Kind,
    MapEdgeCol,
    MapNodeCol,
    MapNodeFlag,
    MapNodeState,
    MsgType,
    OptionCol,
    OptionFlag,
    OptionKind,
    OrbCol,
    PotionCol,
    PotionFlag,
    PowerCol,
    RefCategory,
    RefCol,
    RelicCol,
    RelicFlag,
    ResetFlag,
    RestFlag,
    Result,
    RewardFlag,
    SeatCol,
    SelFlag,
    Table,
    TerminalReason,
    TreasurePhase,
    VarCol,
    VarFlag,
    Verb,
    ViewMode,
    Zone,
)
from .vocab import Vocab, make_synthetic_vocab

I64 = NDArray[np.int64]
SizeMode = Literal["typical", "extreme", "h5"]
_I32_MIN, _I32_MAX = -(1 << 31), (1 << 31) - 1
_I32 = struct.Struct("<i")


# =========================================================================== builder


def _default_rows() -> list[I64]:
    rows = [np.zeros(c, dtype=np.int64) for c in N_COLS]
    for (t, col) in POINTERS:
        rows[t][col] = -1
    for t, tcol, rcol in DYNAMIC_POINTERS:
        rows[t][tcol] = rows[t][rcol] = -1
    for s in SLICES:
        rows[s.owner][s.off_col] = -1
    return rows


DEFAULT_ROWS: tuple[I64, ...] = tuple(_default_rows())
"""Per-table default row: 0 everywhere, -1 for row pointers, pointer table ids and slice offsets."""

_SLICE_BY_CHILD: dict[Table, SliceSpec] = {s.child: s for s in SLICES if s.child != Table.VARS}
_VAR_SLICE_BY_OWNER: dict[Table, SliceSpec] = {s.owner: s for s in SLICES if s.child == Table.VARS}


def _to_i32_signed(h: int) -> int:
    return h - (1 << 32) if h >= 1 << 31 else h


class FrameBuilder:
    """Build one frame. Rows are int64 internally and range-checked to int32 on ``build``.

    * plain tables: :meth:`add` / :meth:`extend`; edit with :meth:`rows` (a mutable array);
    * POWERS / INTENTS / REFS: :meth:`attach` to their owner row, in owner-row order;
    * VARS: :meth:`attach_vars` in any order - grouped by (owner_table, owner_row) and sorted
      by (target_row, var_id) at build, owners' ``var_off/var_cnt`` filled in;
    * canonical multisets (DRAW/DISCARD/EXHAUST/DECK, CHOICE from piles): :meth:`add_multiset`;
    * ACTIONS are put in canonical order at build (nothing may point at action rows).
    """

    def __init__(self, msg_type: int = MsgType.OBS) -> None:
        self.msg_type = int(msg_type)
        self.vocab_hash = 0
        self.catalog_hash = 0
        self.episode_id = 0
        self.step_id = 0
        self.flags = 0
        self.result = 0
        self.auto_steps = 0
        self.host_us = 0
        self.glob = np.zeros(GLOBAL_LEN, dtype=np.int64)
        self._blocks: list[list[I64]] = [[] for _ in range(N_TABLES)]
        self._n = [0] * N_TABLES
        self._vars: list[tuple[int, int, I64]] = []
        self._last_owner: dict[int, int] = {}
        self.forecast: I64 | None = None  # (n, 17) FORECAST rows: the frame gets the trailer (N_TABLES 17)
        self.calc: tuple[I64, I64, I64] | None = None  # TURN / FOE / ROUTE rows: the calc trailers (N_TABLES + 3)
        self.hdraw: I64 | None = None  # (n, 3) HDRAW rows: the hidden trailer after the calc tables (N_TABLES + 1)

    def n_rows(self, t: int) -> int:
        return self._n[t]

    def rows(self, t: int) -> I64:
        """The table's rows as one mutable (n, cols) array."""
        blocks = self._blocks[t]
        if len(blocks) != 1:
            merged = np.concatenate(blocks) if blocks else np.zeros((0, N_COLS[t]), dtype=np.int64)
            self._blocks[t] = blocks = [merged]
        return blocks[0]

    def new_rows(self, t: int, n: int) -> I64:
        """``n`` default rows of table ``t`` (not yet added)."""
        return np.tile(DEFAULT_ROWS[t], (n, 1))

    def add(self, t: int, **cols: int) -> int:
        row = DEFAULT_ROWS[t].copy()
        names = _COL_NAMES[t]
        for k, v in cols.items():
            row[names[k]] = v
        return self.extend(t, row[None, :]).start

    def extend(self, t: int, rows: ArrayLike) -> range:
        a = np.asarray(rows, dtype=np.int64)
        if a.ndim != 2 or a.shape[1] != N_COLS[t]:
            raise ValueError(f"{Table(t).name}: rows must have shape (n, {N_COLS[t]}), got {a.shape}")
        if t == Table.VARS or t in _SLICE_BY_CHILD:
            raise ValueError(f"{Table(t).name} rows must be added through attach/attach_vars")
        start = self._n[t]
        if len(a):
            self._blocks[t].append(a.copy())
            self._n[t] += len(a)
        return range(start, self._n[t])

    def attach(self, child: int, owner_row: int, rows: ArrayLike) -> range:
        """Append POWERS/INTENTS (owner CREATURES) or REFS (owner OPTIONS) rows of one owner."""
        spec = _SLICE_BY_CHILD[Table(child)]
        a = np.array(rows, dtype=np.int64, ndmin=2)
        if not len(a):
            return range(self._n[child], self._n[child])
        if not 0 <= owner_row < self._n[spec.owner]:
            raise ValueError(f"{Table(child).name}: owner row {owner_row} does not exist")
        if owner_row <= self._last_owner.get(child, -1):
            raise ValueError(f"{Table(child).name}: attach in owner-row order, once per owner")
        self._last_owner[child] = owner_row
        a[:, spec.child_owner_row_col] = owner_row
        if spec.child_owner_table_col is not None:
            a[:, spec.child_owner_table_col] = spec.owner
        start = self._n[child]
        self._blocks[child].append(a)
        self._n[child] += len(a)
        own = self.rows(spec.owner)
        own[owner_row, spec.off_col] = start
        own[owner_row, spec.cnt_col] = len(a)
        return range(start, self._n[child])

    def attach_vars(self, owner_table: int, owner_row: int, rows: ArrayLike | None) -> None:
        if rows is None or not np.size(rows):
            return
        if owner_table not in _VAR_SLICE_BY_OWNER:
            raise ValueError(f"{Table(owner_table).name} rows cannot own VARS")
        self._vars.append((int(owner_table), int(owner_row), np.array(rows, dtype=np.int64, ndmin=2)))

    def pending_vars(self, owner_table: int, owner_row: int) -> I64 | None:
        """VARS rows attached so far to one owner (unsorted), or ``None``."""
        mine = [a for t, r, a in self._vars if t == owner_table and r == owner_row]
        return np.concatenate(mine) if mine else None

    def add_multiset(self, instances: ArrayLike, instance_vars: Sequence[I64 | None]) -> range:
        """Group identical CARDS instances (schema 5.2.3) and add the groups in canonical order.

        ``instances``: (k, 25) rows, one per card instance; ``instance_vars[i]``: its VARS rows.
        Each group row gets ``count`` = group size and ``pos = -1``.
        """
        inst = np.asarray(instances, dtype=np.int64)
        groups: dict[tuple[int, ...], list[int]] = {}
        for i in range(len(inst)):
            groups.setdefault(card_content_key(inst[i], instance_vars[i]), []).append(i)
        start = self._n[Table.CARDS]
        for key in sorted(groups):
            members = groups[key]
            row = inst[members[0]].copy()
            row[CardCol.COUNT] = len(members)
            row[CardCol.POS] = -1
            row[CardCol.VAR_OFF], row[CardCol.VAR_CNT] = -1, 0
            r = self.extend(Table.CARDS, row[None, :]).start
            self.attach_vars(Table.CARDS, r, instance_vars[members[0]])
        return range(start, self._n[Table.CARDS])

    # ------------------------------------------------------------------ build
    def _resolve_vars(self) -> I64:
        for s in _VAR_SLICE_BY_OWNER.values():
            own = self.rows(s.owner)
            own[:, s.off_col], own[:, s.cnt_col] = -1, 0
        if not self._vars:
            return np.zeros((0, N_COLS[Table.VARS]), dtype=np.int64)
        merged: dict[tuple[int, int], list[I64]] = {}
        for t, r, a in self._vars:
            if not 0 <= r < self._n[t]:
                raise ValueError(f"VARS owner {Table(t).name}[{r}] does not exist")
            merged.setdefault((t, r), []).append(a)
        out: list[I64] = []
        n = 0
        for (t, r) in sorted(merged):
            a = np.concatenate(merged[(t, r)])
            a = a[np.lexsort((a[:, VarCol.VAR_ID], a[:, VarCol.TARGET_ROW]))]
            k = a[:, [VarCol.TARGET_ROW, VarCol.VAR_ID]]
            if len(a) > 1 and not (np.diff(k[:, 0] * (1 << 32) + k[:, 1]) > 0).all():
                raise ValueError(f"duplicate (target_row, var_id) in VARS of {Table(t).name}[{r}]")
            a[:, VarCol.OWNER_TABLE], a[:, VarCol.OWNER_ROW] = t, r
            spec = _VAR_SLICE_BY_OWNER[Table(t)]
            own = self.rows(t)
            own[r, spec.off_col], own[r, spec.cnt_col] = n, len(a)
            n += len(a)
            out.append(a)
        return np.concatenate(out)

    def build(self, *, hashes: bool = True) -> bytes:
        """Encode the frame payload (without the length prefix).

        With ``hashes=False`` the directory hash of non-empty tables is 0 (fast path for
        bulk synthetic data; such frames fail ``verify_hashes``).
        """
        tables = [self.rows(t) for t in range(N_TABLES)]
        tables[Table.VARS] = self._resolve_vars()
        act = tables[Table.ACTIONS]
        if len(act):
            order = np.lexsort(tuple(act[:, c] for c in reversed(ACTION_SORT_COLS)))
            act = act[order]
            keys = act[:, list(ACTION_SORT_COLS)]
            if (np.diff(keys, axis=0) == 0).all(axis=1).any():
                raise ValueError("duplicate ACTIONS rows")
            tables[Table.ACTIONS] = act
        for t, a in enumerate(tables):
            if len(a) > MAX_ROWS[t]:
                raise ValueError(f"{Table(t).name}: {len(a)} rows exceed the v1 cap {MAX_ROWS[t]}")
            if a.size and (a.min() < _I32_MIN or a.max() > _I32_MAX):
                raise ValueError(f"{Table(t).name}: value outside int32")
        data_words = sum(a.size for a in tables)
        head_words = HEADER_LEN + GLOBAL_LEN + N_TABLES * DIR_STRIDE
        out = np.zeros(head_words + data_words, dtype="<i4")
        pos = head_words
        directory = out[HEADER_LEN + GLOBAL_LEN : head_words].reshape(N_TABLES, DIR_STRIDE)
        for t, a in enumerate(tables):
            out[pos : pos + a.size] = a.ravel()
            raw = out[pos : pos + a.size].tobytes()
            h = fnv1a32(raw) if (hashes or not raw) else 0
            directory[t] = (t, len(a), N_COLS[t], pos * WORD, _to_i32_signed(h))
            pos += a.size
        if self.forecast is not None:  # trailer at X0 = the end of the v1 data (notes/forecast/contract.md 2.2)
            fc = np.asarray(self.forecast, np.int64).reshape(-1, FORECAST_N_COLS)
            if len(fc) > FORECAST_MAX_ROWS or (fc.size and (fc.min() < _I32_MIN or fc.max() > _I32_MAX)):
                raise ValueError("FORECAST rows exceed the cap or int32")
            data = fc.astype("<i4").tobytes()
            h = fnv1a32(data) if (hashes or not data) else 0
            entry = np.array([FORECAST_TABLE, len(fc), FORECAST_N_COLS, (pos + DIR_STRIDE) * WORD, _to_i32_signed(h)],
                             "<i4")
            out = np.concatenate([out, entry, fc.astype("<i4").ravel()])
            pos += DIR_STRIDE + fc.size
        if self.calc is not None:  # TURN, FOE, ROUTE after [FORECAST] (notes/calc/contract.md 2.2)
            for tid, nc, cap, rows in zip(CALC_TABLES, CALC_N_COLS, CALC_MAX_ROWS, self.calc):
                a = np.asarray(rows, np.int64).reshape(-1, nc)
                if len(a) > cap or (a.size and (a.min() < _I32_MIN or a.max() > _I32_MAX)):
                    raise ValueError(f"calc table {tid}: rows exceed the cap or int32")
                data = a.astype("<i4").tobytes()
                h = fnv1a32(data) if (hashes or not data) else 0
                entry = np.array([tid, len(a), nc, (pos + DIR_STRIDE) * WORD, _to_i32_signed(h)], "<i4")
                out = np.concatenate([out, entry, a.astype("<i4").ravel()])
                pos += DIR_STRIDE + a.size
        if self.hdraw is not None:  # HDRAW closes the chain (notes/critic/contract.md 1.3)
            a = np.asarray(self.hdraw, np.int64).reshape(-1, HDRAW_N_COLS)
            if len(a) > HDRAW_MAX_ROWS or (a.size and (a.min() < _I32_MIN or a.max() > _I32_MAX)):
                raise ValueError("HDRAW rows exceed the cap or int32")
            data = a.astype("<i4").tobytes()
            h = fnv1a32(data) if (hashes or not data) else 0
            entry = np.array([HDRAW_TABLE, len(a), HDRAW_N_COLS, (pos + DIR_STRIDE) * WORD, _to_i32_signed(h)], "<i4")
            out = np.concatenate([out, entry, a.astype("<i4").ravel()])
            pos += DIR_STRIDE + a.size
        if self.glob.min() < _I32_MIN or self.glob.max() > _I32_MAX:
            raise ValueError("global value outside int32")
        out[HEADER_LEN : HEADER_LEN + GLOBAL_LEN] = self.glob
        head = out[:HEADER_LEN]
        head[H.MAGIC] = FRAME_MAGIC
        head[H.SCHEMA_VERSION] = SCHEMA_VERSION
        head[H.MSG_TYPE] = self.msg_type
        head[H.VOCAB_HASH] = self.vocab_hash
        head[H.CATALOG_HASH] = self.catalog_hash
        head[H.EPISODE_ID] = self.episode_id
        head[H.STEP_ID] = self.step_id
        head[H.FRAME_BYTES] = out.nbytes
        head[H.N_TABLES] = ((N_TABLES if self.forecast is None else N_TABLES_FORECAST) + 3 * (self.calc is not None)
                            + (self.hdraw is not None))
        head[H.GLOBAL_LEN] = GLOBAL_LEN
        head[H.FLAGS] = self.flags
        head[H.RESULT] = self.result
        head[H.N_ACTIONS] = len(tables[Table.ACTIONS])
        head[H.AUTO_STEPS] = self.auto_steps
        head[H.HOST_US] = self.host_us
        return out.tobytes()


_COL_NAMES = [COLUMN_INDEX[Table(t)] for t in range(N_TABLES)]


def strip_forecast(payload: bytes | bytearray | memoryview) -> bytes:
    """The flag-off frame of a frame with the FORECAST trailer: bytes [0, X0) with header FRAME_BYTES = X0 and
    N_TABLES = 16 (contract 2.2: the host's flag-off frame of the same state is exactly this)."""
    w = np.frombuffer(bytes(payload), "<i4").copy()
    assert w[H.N_TABLES] == N_TABLES_FORECAST and parse_frame(payload).hd_rows < 0
    d = w[HEADER_LEN + GLOBAL_LEN : HEADER_LEN + GLOBAL_LEN + N_TABLES * DIR_STRIDE].reshape(N_TABLES, DIR_STRIDE)
    x0 = HEADER_LEN + GLOBAL_LEN + N_TABLES * DIR_STRIDE + int((np.maximum(d[:, 1], 0) * np.array(N_COLS)).sum())
    out = w[:x0].copy()
    out[H.FRAME_BYTES], out[H.N_TABLES] = x0 * WORD, N_TABLES
    return out.tobytes()


def strip_calc(payload: bytes | bytearray | memoryview) -> bytes:
    """The calc-flag-off frame of a frame with the calc trailers: bytes [0, X1) (X1 = the end of the v1 data and the
    FORECAST trailer, if any) with header FRAME_BYTES = X1 and N_TABLES - 3 (notes/calc/contract.md 2.2)."""
    f = parse_frame(payload)
    assert f.cc_off is not None, "frame without the calc trailers"
    assert f.hd_rows < 0, "strip_hidden first"
    x1 = f.cc_off[0] - DIR_STRIDE  # the TURN entry starts at X1
    w = np.frombuffer(bytes(payload), "<i4")
    out = w[:x1].copy()
    out[H.FRAME_BYTES], out[H.N_TABLES] = x1 * WORD, int(w[H.N_TABLES]) - 3
    return out.tobytes()


def strip_hidden(payload: bytes | bytearray | memoryview) -> bytes:
    """The frame of a host without RESET bit 10 (notes/critic/contract.md 1.3): bytes [0, XH) (XH = the HDRAW entry)
    with header FRAME_BYTES = XH and N_TABLES - 1; every other byte is the host's flag-off frame of the same state."""
    f = parse_frame(payload)
    assert f.hd_entry >= 0, "frame without the HDRAW trailer"
    w = np.frombuffer(bytes(payload), "<i4")
    out = w[: f.hd_entry].copy()
    out[H.FRAME_BYTES], out[H.N_TABLES] = f.hd_entry * WORD, int(w[H.N_TABLES]) - 1
    return out.tobytes()


def keep_trailers(payload: bytes | bytearray | memoryview, *, forecast: bool, calc: bool,
                  hidden: bool = False) -> bytes:
    """The frame with only the named trailers (each must be present): the v1 data, then [FORECAST], then [TURN FOE
    ROUTE], then [HDRAW], entries re-addressed; header FRAME_BYTES / N_TABLES set. The host's frame of the same state
    with those RESET flags is exactly this (notes/calc/contract.md 2.2, notes/critic/contract.md 1.3)."""
    f = parse_frame(payload)
    assert not f.elided, "resolve elided tables first"
    w = f.words
    x0 = HEADER_LEN + GLOBAL_LEN + N_TABLES * DIR_STRIDE + int(sum(a.size for a in f.tables))
    parts = [w[:x0].copy()]
    pos = x0
    trailers: list[tuple[int, int, np.ndarray, int]] = []
    if forecast:
        assert f.forecast is not None
        trailers.append((FORECAST_TABLE, FORECAST_N_COLS, f.forecast, f.fc_hash))
    if calc:
        assert f.cc_rows is not None and f.cc_hash is not None
        trailers += [(tid, nc, f.calc_table(k), f.cc_hash[k]) for k, (tid, nc) in enumerate(zip(CALC_TABLES, CALC_N_COLS))]
    if hidden:
        assert f.hdraw is not None
        trailers.append((HDRAW_TABLE, HDRAW_N_COLS, f.hdraw, f.hd_hash))
    for tid, nc, a, h in trailers:
        parts.append(np.array([tid, a.shape[0], nc, (pos + DIR_STRIDE) * WORD, h], "<i4"))
        parts.append(np.asarray(a, "<i4").ravel())
        pos += DIR_STRIDE + a.size
    out = np.concatenate(parts).astype("<i4")
    out[H.FRAME_BYTES] = out.nbytes
    out[H.N_TABLES] = N_TABLES + int(forecast) + 3 * int(calc) + int(hidden)
    return out.tobytes()


def strip_map_in_combat(payload: bytes | bytearray | memoryview) -> bytes:
    """The frame of the same game moment from a host without RESET bit 1 (MAP_IN_COMBAT; notes/inputs2/contract.md 3):
    a frame with GLOBAL[IN_COMBAT] = 1 and KIND != MAP gets 0 MAP_NODES / MAP_EDGES rows, 0 ROUTE rows (if it carries
    the calc trailers) and header flag HAS_MAP clear, the directory and FRAME_BYTES re-addressed and re-hashed; any
    other frame is returned unchanged."""
    f = parse_frame(payload)
    assert not f.elided, "resolve elided tables first"
    g = f.glob
    if not (int(g[G.IN_COMBAT]) == 1 and int(g[G.KIND]) != Kind.MAP):
        return bytes(payload)
    w = f.words
    head = w[: HEADER_LEN + GLOBAL_LEN + N_TABLES * DIR_STRIDE].copy()
    d = head[HEADER_LEN + GLOBAL_LEN :].reshape(N_TABLES, DIR_STRIDE)
    parts = [head]
    pos = head.size
    for t, a in enumerate(f.tables):
        assert a is not None
        a = a[:0] if t in (Table.MAP_NODES, Table.MAP_EDGES) else a
        data = np.ascontiguousarray(a, "<i4").ravel()
        d[t, 1], d[t, 3], d[t, 4] = a.shape[0], pos * WORD, _to_i32_signed(fnv1a32(data.tobytes()))
        parts.append(data)
        pos += data.size
    trailers: list[tuple[int, int, np.ndarray]] = []
    if f.forecast is not None:
        trailers.append((FORECAST_TABLE, FORECAST_N_COLS, f.forecast))
    if f.cc_rows is not None:
        trailers += [(tid, nc, f.calc_table(k)[:0] if tid == CALC_TABLES[2] else f.calc_table(k))
                     for k, (tid, nc) in enumerate(zip(CALC_TABLES, CALC_N_COLS))]
    if f.hdraw is not None:  # CARDS rows keep their indices: HDRAW is unchanged
        trailers.append((HDRAW_TABLE, HDRAW_N_COLS, f.hdraw))
    for tid, nc, a in trailers:
        data = np.ascontiguousarray(a, "<i4").ravel()
        parts.append(np.array([tid, a.shape[0], nc, (pos + DIR_STRIDE) * WORD,
                                _to_i32_signed(fnv1a32(data.tobytes()))], "<i4"))
        parts.append(data)
        pos += DIR_STRIDE + data.size
    out = np.concatenate(parts).astype("<i4")
    out[H.FRAME_BYTES] = out.nbytes
    out[H.FLAGS] &= ~int(FrameFlag.HAS_MAP)
    return out.tobytes()


def error_frame(code: int, *, episode_id: int = 0, step_id: int = 0, vocab_hash: int = 0,
                catalog_hash: int = 0) -> bytes:
    """An ERROR frame (schema 7.3): empty tables, ``global[0]`` = error code."""
    b = FrameBuilder(MsgType.ERROR)
    b.glob[G_ERROR_CODE] = code
    b.episode_id, b.step_id, b.vocab_hash, b.catalog_hash = episode_id, step_id, vocab_hash, catalog_hash
    return b.build()


# =========================================================================== random frames

# (mean, max) for "typical" sizes from schema 11; value used for "extreme".
_TYPICAL: dict[str, tuple[float, int]] = {
    "hand": (3.85, 10), "draw": (5.06, 17), "discard": (3.83, 16), "exhaust": (0.48, 8), "deck": (9.9, 22),
    "extra_enemies": (0.48, 5), "powers": (0.89, 6), "relics": (3.72, 9), "choice": (8.0, 25),
    "event_opts": (2.55, 4), "reward_items": (2.0, 4), "potions": (1.6, 5), "intents": (0.2, 2),
}
_EXTREME: dict[str, int] = {
    "hand": 10, "draw": 40, "discard": 30, "exhaust": 20, "deck": 45, "extra_enemies": 5, "powers": 6,
    "relics": 30, "choice": 60, "event_opts": 4, "reward_items": 6, "potions": 5, "intents": 2,
}
# H5 capacity workload: 4 private piles, six targets and ten playable cards per seat.
# Deliberately busy, not a claim about the real host's size distribution.
_H5 = dict(_EXTREME, draw=12, discard=8, exhaust=3, deck=20, relics=8, powers=2, intents=1)
_VARS_K = np.array([0, 1, 2, 3, 4, 11])
_VARS_P_TYPICAL = np.array([0.143, 0.540, 0.285, 0.024, 0.006, 0.002])
_VARS_P_EXTREME = np.array([0.0, 0.2, 0.4, 0.25, 0.13, 0.02])

COMBAT_SELECT_SUBS = frozenset({CardSelectSub.COMBAT_PILE, CardSelectSub.HAND, CardSelectSub.HAND_DISCARD,
                                CardSelectSub.HAND_UPGRADE, CardSelectSub.CHOOSE_A_CARD})
_HAND_SUBS = frozenset({CardSelectSub.HAND, CardSelectSub.HAND_DISCARD, CardSelectSub.HAND_UPGRADE})
_DECK_SUBS = frozenset({CardSelectSub.DECK_UPGRADE, CardSelectSub.DECK_TRANSFORM, CardSelectSub.DECK_ENCHANT,
                        CardSelectSub.DECK_REMOVAL, CardSelectSub.DECK_GENERIC})
_UPGRADE_SUBS = frozenset({CardSelectSub.HAND_UPGRADE, CardSelectSub.DECK_UPGRADE})
_ROOM_LEVEL = frozenset({Kind.MAP, Kind.SHOP, Kind.REST, Kind.EVENT, Kind.REWARD, Kind.TREASURE})
_DEFAULT_REASON = {int(Result.WIN): int(TerminalReason.WIN), int(Result.LOSS): int(TerminalReason.DEATH),
                   int(Result.ABORTED): int(TerminalReason.ABORTED)}
FAKE_ABORT_REASON = 2  # GLOBAL[ABORT_REASON] of fake aborted episodes (some SpireHost.Driver.AbortReason)
_TGT_NONE, _TGT_ENEMY, _TGT_ALLY = 0, 1, 2


@dataclass
class MapLayout:
    nodes: I64  # (n, 6) MAP_NODES rows with state = NORMAL
    edges: I64  # (m, 2)
    children: list[list[int]]
    start: int  # node row of the starting point
    n_map_rows: int


@dataclass
class RunCtx:
    """Run-level context a frame is generated in (persistent across a FakeGame episode)."""

    characters: tuple[int, ...]
    ascension: int = 0
    view_mode: int = ViewMode.TEAM
    view_seat: int = -1
    reset_flags: int = int(RESET_FLAGS_DEFAULT)
    episode_id: int = 1
    step_id: int = 0
    act_index: int = 0
    act_id: int = 2
    total_floor: int = 0
    hp: list[int] = field(default_factory=list)
    max_hp: list[int] = field(default_factory=list)
    gold: list[int] = field(default_factory=list)
    combat_round: int = 0
    enemies: tuple[int, ...] = ()  # monster vocab ids of the current combat
    map: MapLayout | None = None
    path: list[int] = field(default_factory=list)  # visited node rows, last = current
    room: str = ""  # enum:RoomType name of the current combat ("Monster" / "Elite" / "Boss"); "" = by kind
    # public win history, GLOBAL 44..48 (errata-v1 section 5)
    combats_won: int = 0
    elites_won: int = 0
    bosses_won: int = 0
    last_won_room: str = ""  # enum:RoomType name of the last won combat ("" = none yet)
    last_won_floor: int = -1
    # Fake H5 arbitration. None preserves the historical random-frame / single-seat path.
    acting_seat: int | None = None
    ready_seats: set[int] = field(default_factory=set)
    room_done: set[int] = field(default_factory=set)
    rest_done: set[int] = field(default_factory=set)
    treasure_votes: dict[int, int] = field(default_factory=dict)
    shared_vote: bool = False

    @property
    def n_seats(self) -> int:
        return len(self.characters)

    @property
    def act_floor(self) -> int:
        return 0 if self.map is None or not self.path else int(self.map.nodes[self.path[-1], MapNodeCol.ROW])


class FrameGen:
    """Random valid frames. Deterministic given the ``rng`` passed to each call."""

    def __init__(self, vocab: Vocab, catalog_hash: int = 0) -> None:
        self.vocab = vocab
        self.catalog_hash = catalog_hash
        v = vocab
        self.size = {c: v.size(c) for c in v.categories}
        self.characters = [v.id("character", n) for n in v["character"].entries[2:]]
        self.char_defect = v.id("character", "DEFECT")
        self.char_fool = v.id("character", "FOOL")
        self.char_regent = v.id("character", "REGENT")
        self.char_necro = v.id("character", "NECROBINDER")
        self.attack_intents = frozenset({v.id("enum:IntentType", "Attack"), v.id("enum:IntentType", "DeathBlow")})
        self.status_intent = v.id("enum:IntentType", "Status")
        self.pile = {n: v.id("enum:PileType", n) for n in ("Draw", "Hand", "Discard", "Deck")}
        self.tools = (v.id("enum:CrystalSphereToolType", "Small"), v.id("enum:CrystalSphereToolType", "Big"))
        self.reward = {n: v.id("enum:RewardType", n) for n in v["enum:RewardType"].entries[2:]}
        self.shop = {n: v.id("shop_section", n) for n in v["shop_section"].entries[2:]}
        self.point = {n: v.id("enum:MapPointType", n) for n in v["enum:MapPointType"].entries[2:]}
        self.room = {n: v.id("enum:RoomType", n) for n in v["enum:RoomType"].entries[2:]}
        self.rest = {n: v.id("rest_option", n) for n in v["rest_option"].entries[2:]}
        self.confirm = (v.id("confirm_key", "CONFIRM_YES"), v.id("confirm_key", "CONFIRM_NO"))
        self.alt = {n: v.id("card_reward_alt", n) for n in v["card_reward_alt"].entries[2:]}

    def rid(self, rng: np.random.Generator, cat: str) -> int:
        """A random non-special vocab id of ``cat``."""
        hi = self.size[cat]
        return int(rng.integers(2, hi)) if hi > 2 else 1

    def rids(self, rng: np.random.Generator, cat: str, n: int) -> I64:
        """``n`` random non-special vocab ids of ``cat``."""
        hi = self.size[cat]
        return rng.integers(2, hi, n) if hi > 2 else np.ones(n, dtype=np.int64)

    # ------------------------------------------------------------------ contexts and maps
    def random_ctx(self, rng: np.random.Generator, n_seats: int = 1, *, view: int | None = None,
                   size: SizeMode = "typical") -> RunCtx:
        chars = tuple(int(c) for c in rng.choice(self.characters, n_seats))
        if view is None:
            view = ViewMode.SEAT if n_seats > 1 and rng.random() < 0.25 else ViewMode.TEAM
        asc = int(rng.integers(0, 11))
        max_hp = [int(x) for x in rng.integers(60, 90, n_seats)]
        ctx = RunCtx(
            characters=chars, ascension=asc, view_mode=view,
            view_seat=int(rng.integers(0, n_seats)) if view == ViewMode.SEAT else -1,
            reset_flags=int(RESET_FLAGS_DEFAULT | (ResetFlag.MAP_IN_COMBAT | ResetFlag.DECK_IN_COMBAT
                                                    if rng.random() < 0.3 else 0)),
            episode_id=int(rng.integers(1, 1000)), step_id=int(rng.integers(0, 800)),
            act_index=int(rng.integers(0, 3)), max_hp=max_hp,
            hp=[int(rng.integers(1, m + 1)) for m in max_hp],
            gold=[int(x) for x in rng.integers(0, 400 if size == "typical" else 2000, n_seats)],
            combat_round=int(rng.integers(1, 6)),
        )
        ctx.act_id = self.rid(rng, "act")
        ctx.map = self.make_map(rng, size, asc)
        depth = int(rng.integers(0, ctx.map.n_map_rows - 2))
        self.walk(rng, ctx, depth)
        ctx.total_floor = ctx.act_index * 17 + ctx.act_floor
        ctx.room = str(rng.choice(["Monster", "Monster", "Monster", "Elite", "Boss"]))
        if ctx.total_floor > 1 and rng.random() < 0.9:  # some combats won already
            ctx.bosses_won = ctx.act_index
            ctx.elites_won = int(rng.integers(0, 2 + 2 * ctx.act_index))
            ctx.combats_won = ctx.bosses_won + ctx.elites_won + int(rng.integers(1, max(2, ctx.total_floor // 2)))
            ctx.last_won_room = str(rng.choice(["Monster", "Elite", "Boss"] if ctx.act_index else ["Monster", "Elite"]))
            ctx.last_won_floor = int(rng.integers(1, ctx.total_floor))
        return ctx

    def walk(self, rng: np.random.Generator, ctx: RunCtx, steps: int) -> None:
        """Advance ``ctx.path`` by ``steps`` random travels."""
        assert ctx.map is not None
        for _ in range(steps):
            nxt = self.reachable(ctx)
            if not nxt:
                return
            ctx.path.append(int(rng.choice(nxt)))

    def reachable(self, ctx: RunCtx) -> list[int]:
        assert ctx.map is not None
        return [ctx.map.start] if not ctx.path else ctx.map.children[ctx.path[-1]]

    def make_map(self, rng: np.random.Generator, size: SizeMode, ascension: int) -> MapLayout:
        n_rows = 16  # regular rows 1..14 + start row 0, boss row 15
        per_row = [1] + [int(x) for x in rng.integers(3 if size == "typical" else 5, 6, n_rows - 2)]
        types = [self.point[n] for n in ("Monster", "Monster", "Monster", "Unknown", "Unknown", "Elite",
                                         "RestSite", "Shop", "Treasure")]
        rows: list[tuple[int, int, int, int, int, int]] = []
        by_row: list[list[int]] = []
        for r, k in enumerate(per_row):
            cols = [3] if r == 0 else sorted(int(c) for c in rng.choice(7, k, replace=False))
            ids = []
            for c in cols:
                if r == 0:
                    pt, fl = self.point["Ancient"], int(MapNodeFlag.START)
                else:
                    pt = self.point["Monster"] if r == 1 else (self.point["RestSite"] if r == n_rows - 2
                                                               else int(rng.choice(types)))
                    fl = int(MapNodeFlag.QUEST) if rng.random() < 0.02 else 0
                ids.append(len(rows))
                rows.append((r, c, pt, MapNodeState.NORMAL, 0, fl))
            by_row.append(ids)
        boss = len(rows)
        rows.append((n_rows - 1, 3, self.point["Boss"], MapNodeState.NORMAL, 0, int(MapNodeFlag.BOSS)))
        second = -1
        if ascension >= 10:
            second = len(rows)
            rows.append((n_rows, 3, self.point["Boss"], MapNodeState.NORMAL, 0, int(MapNodeFlag.SECOND_BOSS)))
        nodes = np.array(rows, dtype=np.int64)
        edges: set[tuple[int, int]] = set()
        cols = nodes[:, MapNodeCol.COL].tolist()
        for r in range(len(by_row) - 1):
            cur, nxt = by_row[r], by_row[r + 1]
            fanout = rng.integers(1, 3, len(cur)).tolist()
            for a, k in zip(cur, fanout):
                for b_ in sorted(nxt, key=lambda x: abs(cols[x] - cols[a]))[:k]:
                    edges.add((a, b_))
            reached = {b_ for (_, b_) in edges}
            for b_ in nxt:
                if b_ not in reached:
                    edges.add((min(cur, key=lambda x: abs(cols[x] - cols[b_])), b_))
        for a in by_row[-1]:
            edges.add((a, boss))
        if second >= 0:
            edges.add((boss, second))
        e = np.array(sorted(edges), dtype=np.int64)
        children: list[list[int]] = [[] for _ in rows]
        for a, b_ in e.tolist():
            children[a].append(b_)
        unknown = nodes[:, MapNodeCol.POINT_TYPE] == self.point["Unknown"]
        nodes[unknown, MapNodeCol.RESOLVED_ROOM] = rng.choice(
            [self.room["Monster"], self.room["Event"], self.room["Shop"], self.room["Treasure"]], int(unknown.sum()))
        return MapLayout(nodes, e, children, by_row[0][0], n_rows)

    # ------------------------------------------------------------------ public API
    def frame(self, rng: np.random.Generator, kind: int, ctx: RunCtx | None = None, *, n_seats: int = 1,
              kind_sub: int | None = None, size: SizeMode = "typical", auto_steps: int = 0,
              hashes: bool = True) -> bytes:
        """A random OBS frame of ``kind`` (one of the 12 decision kinds)."""
        if not Kind.MAP <= kind <= Kind.CRYSTAL:
            raise ValueError(f"not a decision kind: {kind}")
        ctx = ctx or self.random_ctx(rng, n_seats, size=size)
        if kind_sub is None:
            if kind == Kind.CARD_SELECT:
                kind_sub = int(rng.choice([int(s) for s in CardSelectSub]))
            elif kind == Kind.TREASURE:
                kind_sub = int(rng.choice([TreasurePhase.CHEST, TreasurePhase.RELIC]))
            else:
                kind_sub = 0
        for _ in range(100):
            d = _Draft(self, rng, ctx, Kind(kind), kind_sub, size, MsgType.OBS)
            d.b.auto_steps = auto_steps
            try:
                return d.run().build(hashes=hashes)
            except _Retry:
                continue
        raise RuntimeError(f"could not generate kind {Kind(kind).name} sub {kind_sub}")

    def terminal(self, rng: np.random.Generator, ctx: RunCtx, result: int, *, auto_steps: int = 0,
                 hashes: bool = True, reason: int | None = None, abort_reason: int = 0) -> bytes:
        """A TERMINAL frame as the real host writes it (errata-v1 P-5): the final observation without
        actions or combat tables, STEP_ID = ``ctx.step_id`` (the episode's policy decisions),
        GLOBAL[TERMINAL_REASON] = ``reason`` (default: WIN / DEATH / ABORTED by ``result``),
        GLOBAL[ABORT_REASON] = ``abort_reason`` (aborted episodes), GLOBAL[POLICY_DECISIONS] = STEP_ID."""
        d = _Draft(self, rng, ctx, Kind.TERMINAL, 0, "typical", MsgType.TERMINAL)
        d.b.result = result
        d.b.auto_steps = auto_steps
        b = d.run()
        if reason is None:
            reason = _DEFAULT_REASON.get(result, int(TerminalReason.NONE))
        b.glob[G.TERMINAL_REASON] = reason
        b.glob[G.ABORT_REASON] = abort_reason if result == Result.ABORTED else 0
        b.glob[G.POLICY_DECISIONS] = ctx.step_id
        return b.build(hashes=hashes)

    def error(self, ctx: RunCtx, code: int) -> bytes:
        return error_frame(code, episode_id=ctx.episode_id, step_id=ctx.step_id,
                           vocab_hash=self.vocab.hash, catalog_hash=self.catalog_hash)


class _Draft:
    """State of one frame under construction."""

    def __init__(self, gen: FrameGen, rng: np.random.Generator, ctx: RunCtx, kind: Kind, sub: int,
                 size: SizeMode, msg: int) -> None:
        self.g, self.rng, self.ctx, self.kind, self.sub, self.size = gen, rng, ctx, kind, sub, size
        self.b = FrameBuilder(msg)
        self.n = ctx.n_seats
        self.in_combat = kind == Kind.COMBAT or (kind == Kind.CARD_SELECT and sub in COMBAT_SELECT_SUBS)
        team = ctx.view_mode == ViewMode.TEAM
        self.private = [team or s == ctx.view_seat for s in range(self.n)]
        self.view_seats = [s for s in range(self.n) if self.private[s]]
        alive = [s for s in range(self.n) if ctx.hp[s] > 0] or [0]
        if kind == Kind.TERMINAL:
            self.acting: list[int] = []
        elif ctx.acting_seat is not None:
            self.acting = ([s for s in alive if s not in ctx.ready_seats]
                           if ctx.acting_seat == -1 else [ctx.acting_seat])
            assert self.acting
        elif not team:
            self.acting = [ctx.view_seat]
        elif self.in_combat or kind == Kind.MAP:
            self.acting = alive
        else:
            self.acting = [int(rng.choice(alive))]
        self.player_rows = [-1] * self.n
        self.enemy_rows: list[int] = []
        self.hittable: list[int] = []
        self.hand: list[list[tuple[int, int]]] = [[] for _ in range(self.n)]  # (row, target kind)
        self.draw_groups: list[range] = [range(0) for _ in range(self.n)]
        self.deck_groups: list[range] = [range(0) for _ in range(self.n)]
        self.potions: list[tuple[int, int, int, int]] = []  # (row, seat, flags, target kind)
        self.open_slots = [0] * self.n
        self.actions: list[tuple[int, int, int, int, int, int]] = []
        self.has_target_preview = False
        self.has_upgrade_preview = False

    # ------------------------------------------------------------------ helpers
    def count(self, name: str, lo: int = 0) -> int:
        if self.size == "h5":
            return max(lo, _H5[name])
        if self.size == "extreme":
            return max(lo, _EXTREME[name])
        mean, hi = _TYPICAL[name]
        return int(min(hi, max(lo, self.rng.poisson(mean))))

    def act(self, verb: int, seat: int, src_table: int = -1, src_row: int = -1, tgt: int = -1, aux: int = 0) -> None:
        self.actions.append((verb, seat, src_table, src_row, tgt, aux))

    def vars_rows(self, k: int) -> I64:
        """VARS rows for one owner with ``k`` distinct var ids (owner columns set at build)."""
        return self.vars_blocks(np.array([k]))[0]

    def vars_blocks(self, ks: NDArray[np.integer]) -> list[I64]:
        """One VARS block per owner, ``ks[i]`` rows each, generated in one vectorised pass."""
        r = self.rng
        total = int(ks.sum())
        a = self.b.new_rows(Table.VARS, total)
        m = self.g.size["var_name"] - 2
        start = np.repeat(r.integers(0, m, len(ks)), ks)
        j = np.arange(total) - np.repeat(np.cumsum(ks) - ks, ks)
        a[:, VarCol.VAR_ID] = 2 + (start + j) % m  # distinct within an owner while k <= m
        base = r.integers(1, 25, total)
        ench = base + (r.random(total) < 0.05) * r.integers(1, 4, total)
        a[:, VarCol.BASE], a[:, VarCol.ENCHANTED], a[:, VarCol.SHOWN] = base, ench, ench
        a[:, VarCol.FLAGS] = np.where(r.random(total) < 0.05, int(VarFlag.JUST_UPGRADED), 0)
        return np.split(a, np.cumsum(ks)[:-1])

    def card_vars(self, n: int) -> list[I64]:
        p = _VARS_P_TYPICAL if self.size == "typical" else _VARS_P_EXTREME
        return self.vars_blocks(self.rng.choice(_VARS_K, n, p=p))

    def cards(self, n: int, zone: int, seat: int) -> I64:
        r, g = self.rng, self.g
        a = self.b.new_rows(Table.CARDS, n)
        a[:, CardCol.ZONE], a[:, CardCol.SEAT], a[:, CardCol.POS], a[:, CardCol.COUNT] = zone, seat, -1, 1
        a[:, CardCol.CARD_ID] = g.rids(r, "card", n)
        up = r.random(n) < 0.25
        a[:, CardCol.UPGRADE_LEVEL] = up
        x = r.random(n) < 0.03
        a[:, CardCol.ENERGY_COST] = np.where(x, 0, r.choice([0, 1, 1, 1, 2, 2, 3], n))
        regent = self.ctx.characters[seat] == g.char_regent
        star = regent & (r.random(n) < 0.2)
        a[:, CardCol.STAR_COST] = np.where(star, r.integers(0, 4, n), -1)
        a[:, CardCol.COST_FLAGS] = (x * CostFlag.COSTS_X | (up & (r.random(n) < 0.1)) * CostFlag.COST_JUST_UPGRADED
                                    | star * CostFlag.STAR_VISIBLE)
        a[:, CardCol.KEYWORD_BITS] = (r.random((n, 7)) < 0.06) @ (1 << np.arange(7))
        ench = r.random(n) < 0.05
        a[:, CardCol.ENCHANT_ID] = np.where(ench, g.rids(r, "enchantment", n), 0)
        a[:, CardCol.ENCHANT_AMOUNT] = np.where(ench & (r.random(n) < 0.5), r.integers(1, 4, n), -1)
        a[:, CardCol.ENCHANT_STATUS] = np.where(ench, g.rids(r, "enum:EnchantmentStatus", n), 0)
        aff = r.random(n) < 0.03
        a[:, CardCol.AFFLICTION_ID] = np.where(aff, g.rids(r, "affliction", n), 0)
        a[:, CardCol.AFFLICTION_AMOUNT] = np.where(aff, r.integers(0, 3, n), 0)
        a[:, CardCol.REPLAY_COUNT] = r.random(n) < 0.02
        if zone == Zone.HAND:
            a[:, CardCol.POS] = np.arange(n)
            a[:, CardCol.ENERGY_COST_COLOR] = g.rids(r, "enum:CardCostColor", n)
            a[:, CardCol.STAR_COST_COLOR] = np.where(star, g.rids(r, "enum:CardCostColor", n), 0)
            can = r.random(n) < (0.8 if self.size == "typical" else 1.0)
            a[:, CardCol.HAND_FLAGS] = (can * HandFlag.CAN_PLAY | (can & (r.random(n) < 0.1)) * HandFlag.GLOW_GOLD
                                        | (~can & (r.random(n) < 0.5)) * HandFlag.UNPLAYABLE_ICON)
            a[:, CardCol.UNPLAYABLE_REASON] = np.where(can, 0, 1 << r.integers(0, 6, n))
        a[:, CardCol.AUX] = -1
        return a

    def multiset(self, zone: int, seat: int, groups: int, dup: float, **set_cols: int) -> range:
        if groups <= 0:
            return range(self.b.n_rows(Table.CARDS), self.b.n_rows(Table.CARDS))
        tpl = self.cards(groups, zone, seat)
        for k, v in set_cols.items():
            tpl[:, _COL_NAMES[Table.CARDS][k]] = v
        tvars = self.card_vars(groups)
        reps = 1 + self.rng.poisson(dup, groups)
        inst = np.repeat(tpl, reps, axis=0)
        ivars = [tvars[i] for i in np.repeat(np.arange(groups), reps)]
        return self.b.add_multiset(inst, ivars)

    def zone_count(self, rows: range) -> int:
        return int(self.b.rows(Table.CARDS)[rows.start : rows.stop, CardCol.COUNT].sum()) if len(rows) else 0

    # ------------------------------------------------------------------ sections
    def run(self) -> FrameBuilder:
        self.seats()
        if self.in_combat:
            self.creatures()
        self.combat_cards()
        with_deck = not self.in_combat or bool(self.ctx.reset_flags & ResetFlag.DECK_IN_COMBAT)
        if with_deck:
            for s in range(self.n):
                self.deck_groups[s] = self.multiset(Zone.DECK, s, self.count("deck", 4), 0.6)
                self.b.rows(Table.SEATS)[s, SeatCol.DECK_COUNT] = self.zone_count(self.deck_groups[s])
        else:
            self.b.rows(Table.SEATS)[:, SeatCol.DECK_COUNT] = self.rng.integers(10, 30, self.n)
        self.relics_owned()
        self.potions_owned()
        with_map = not self.in_combat or bool(self.ctx.reset_flags & ResetFlag.MAP_IN_COMBAT)
        if with_map:
            self.map_tables()
        getattr(self, f"k_{self.kind.name.lower()}")()
        if self.kind in _ROOM_LEVEL:
            self.room_potion_actions()
        self.finish(with_map, with_deck)
        if self.ctx.reset_flags & ResetFlag.FORECAST:
            self.b.forecast = self.forecast_rows()
        if self.ctx.reset_flags & ResetFlag.CALC:
            self.b.calc = self.calc_rows()
        if self.ctx.reset_flags & ResetFlag.HIDDEN:
            self.b.hdraw = self.hdraw_rows()
        return self.b

    def hdraw_rows(self) -> I64:
        """HDRAW rows (notes/critic/contract.md 1.3): every card instance of every visible seat's draw pile in a
        random draw order, (SEAT, POS, OWNER_ROW = its multiset DRAW row). Drawn last, so every other table and
        trailer is the flag-off frame's."""
        cards = self.b.rows(Table.CARDS)
        out: list[I64] = []
        for s in range(self.n):
            rows = self.draw_groups[s]
            if not len(rows):
                continue
            owner = np.repeat(np.arange(rows.start, rows.stop), cards[rows.start : rows.stop, CardCol.COUNT])
            owner = self.rng.permutation(owner)
            out.append(np.stack([np.full(len(owner), s), np.arange(len(owner)), owner], 1))
        return np.concatenate(out).astype(np.int64) if out else np.zeros((0, HDRAW_N_COLS), np.int64)

    def calc_rows(self) -> tuple[I64, I64, I64]:
        """TURN / FOE / ROUTE rows (notes/calc/contract.md 1.1 row rules, 2.3 ranges) with random values; drawn
        after everything else, so the v1 tables and FORECAST do not depend on the flag."""
        r, b = self.rng, self.b
        gl, seats, crea = b.glob, b.rows(Table.SEATS), b.rows(Table.CREATURES)
        nodes = b.rows(Table.MAP_NODES)
        s = int(gl[G.ACTING_SEAT])
        turn: list[list[int]] = []
        foe: list[list[int]] = []
        p = int(seats[s, SeatCol.CREATURE_ROW]) if 0 <= s < len(seats) else -1
        if (gl[G.IN_COMBAT] == 1 and p >= 0 and crea[p, CreatureCol.ALIVE] == 1
                and seats[s, SeatCol.PRIVATE_VISIBLE] == 1):
            hp, blk = max(int(crea[p, CreatureCol.HP]), 1), int(crea[p, CreatureCol.BLOCK])
            inc, eb, ed, el = (int(r.integers(0, 60)), int(r.integers(0, 8)), int(r.integers(0, 6)),
                               int(r.integers(0, 4)))
            now = max(0, ed + inc - blk - eb) + el
            reach = int(r.integers(0, 40))
            mn = min(now, max(0, ed + inc - blk - reach) + el)
            fl = int(now >= hp) | int(mn >= hp) << 1 | int(r.random() < 0.1) << 2 | int(r.random() < 0.1) << 3 | \
                int(r.random() < 0.05) << 4
            turn.append([p, hp, blk, inc, eb, ed, el, now, reach, mn, fl])
            for e in range(len(crea)):
                if crea[e, CreatureCol.KIND] != CreatureKind.ENEMY or crea[e, CreatureCol.ALIVE] != 1:
                    continue
                hidden = int(crea[e, CreatureCol.HP]) < 0
                tk = -1 if hidden else int(crea[e, CreatureCol.HP]) + int(crea[e, CreatureCol.BLOCK])
                dmg = int(r.integers(0, 50))
                guarded = r.random() < 0.1
                kill = tk > 0 and dmg >= tk and not guarded
                foe.append([e, tk, dmg, int(r.integers(0, 30)),
                            int(kill) | int(guarded) << 1 | int(hidden) << 2 | int(r.random() < 0.05) << 3 |
                            int(crea[e, CreatureCol.HITTABLE] == 0) << 4])
        route: list[list[int]] = []
        for n in range(len(nodes)):
            if nodes[n, MapNodeCol.STATE] != MapNodeState.REACHABLE:
                continue
            rest = int(r.integers(-1, 8))
            route.append([n, int(r.integers(1, 17)), rest, int(r.integers(-1, 10)),
                          -1 if rest < 0 else int(r.integers(0, 3)), -1 if rest < 0 else int(r.integers(0, 6)),
                          int(r.integers(0, 5)), int(r.integers(0, 12)), *(int(v) for v in r.integers(0, 9, 4))])
        return (np.array(turn, np.int64).reshape(-1, CALC_N_COLS[0]), np.array(foe, np.int64).reshape(-1, CALC_N_COLS[1]),
                np.array(route, np.int64).reshape(-1, CALC_N_COLS[2]))

    def forecast_rows(self) -> I64:
        """FORECAST rows (contract 2.3) for every enemy with INTENTS: two slots each, all classes, 0 (UNKNOWN) to 3
        candidates, moves with 0-2 intents. Drawn after the rest of the frame, so the v1 tables do not depend on
        the flag."""
        r, g, b = self.rng, self.g, self.b
        crea = b.rows(Table.CREATURES)
        out: list[list[int]] = []
        for c in range(b.n_rows(Table.CREATURES)):
            if not crea[c, CreatureCol.INTENT_CNT]:
                continue
            n_states = int(r.integers(1, 9))
            shown = int(r.integers(-1, n_states))  # -1: an ad-hoc state (stun)
            cls1 = int(r.choice(4, p=[0.55, 0.15, 0.22, 0.08]))
            for ahead in (1, 2):
                cls = cls1 if cls1 == ForecastClass.UNKNOWN else int(r.choice(4, p=[0.55, 0.15, 0.25, 0.05]))
                if cls == ForecastClass.UNKNOWN:
                    out.append([c, 0, 0, 0, 0, -1, -1, -1, -1, int(ForecastFlag.NO_INTENTS), ahead, 0, 0, cls, 0, -1,
                                shown])
                    continue
                k = int(r.integers(2, 4)) if cls == ForecastClass.RANDOM else 1
                assert k <= FORECAST_MAX_CAND
                w = r.random(k) + 0.05
                probs = [int(np.round(p * FORECAST_PROB_ONE)) for p in w / w.sum()]
                moves = r.choice(n_states, k, replace=False) if k <= n_states else r.integers(0, n_states, k)
                for cand in range(k):
                    move = int(moves[cand])
                    same = int(ForecastFlag.SAME_AS_SHOWN) * (move == shown)
                    base = [ahead, cand, k, cls, probs[cand], move, shown]
                    n_int = int(r.choice(3, p=[0.15, 0.65, 0.2]))
                    if not n_int:
                        out.append([c, 0, 0, 0, 0, -1, -1, -1, -1, int(ForecastFlag.NO_INTENTS) | same, *base])
                    for i in range(n_int):
                        t = int(g.rid(r, "enum:IntentType"))
                        attack = t in g.attack_intents or r.random() < 0.4
                        t = next(iter(g.attack_intents)) if attack else t
                        dmg = [int(r.integers(0, 35)) if attack and s in self.view_seats else -1 for s in range(4)]
                        out.append([c, i, t, int(r.choice([1, 1, 2, 3])) if attack else 0,
                                    int(r.integers(1, 4)) if t == g.status_intent else 0, *dmg,
                                    (int(ForecastFlag.HAS_TIP) if r.random() < 0.95 else 0) | same, *base])
        return np.array(out, np.int64).reshape(-1, FORECAST_N_COLS)[:FORECAST_MAX_ROWS]

    def seats(self) -> None:
        r, ctx, n, g = self.rng, self.ctx, self.n, self.g
        a = self.b.new_rows(Table.SEATS, n)
        chars = np.array(ctx.characters)
        a[:, SeatCol.SEAT], a[:, SeatCol.CHARACTER_ID] = np.arange(n), chars
        a[:, SeatCol.PRIVATE_VISIBLE] = self.private
        a[:, SeatCol.HP], a[:, SeatCol.MAX_HP] = ctx.hp, ctx.max_hp
        a[:, SeatCol.ALIVE] = np.array(ctx.hp) > 0
        a[:, SeatCol.GOLD] = ctx.gold
        a[:, SeatCol.POTION_SLOTS] = 3 - (ctx.ascension >= 4) + (r.random(n) < 0.1)
        a[:, SeatCol.MAX_ENERGY] = 3 + (r.random(n) < 0.15)
        a[:, SeatCol.ORB_CAPACITY] = np.where(chars == g.char_defect, 3, 0) if self.size != "extreme" else 10
        if self.in_combat:
            a[:, SeatCol.BLOCK] = r.integers(0, 15, n) * (r.random(n) < 0.4)
            a[:, SeatCol.ENERGY] = r.integers(0, a[:, SeatCol.MAX_ENERGY] + 1)
            a[:, SeatCol.STARS] = np.where(chars == g.char_regent, r.integers(0, 6, n), 0)
            a[:, SeatCol.READY_TO_END_TURN] = (r.random(n) < 0.3) if n > 1 and self.size == "typical" else 0
            if n > 1 and a[self.acting, SeatCol.READY_TO_END_TURN].all():
                a[self.acting[0], SeatCol.READY_TO_END_TURN] = 0
            if ctx.acting_seat is not None:
                a[:, SeatCol.READY_TO_END_TURN] = [s in ctx.ready_seats for s in range(n)]
            a[:, SeatCol.TURN_NUMBER] = ctx.combat_round
            fool = chars == g.char_fool
            a[:, SeatCol.ARROW_SLOTS] = np.where(fool, r.integers(2, 5, n), 0) if self.size != "extreme" else 4
            a[:, SeatCol.ARROWS_AVAILABLE] = r.integers(0, a[:, SeatCol.ARROW_SLOTS] + 1)
            a[:, SeatCol.MELODY_BITS] = np.where(fool, r.integers(0, 8, n), 0)
            hist = r.integers(0, 4, (n, 8))
            a[:, SeatCol.CARDS_PLAYED_TURN : SeatCol.HP_LOSS_EVENTS_COMBAT + 1] = hist
        else:
            a[:, SeatCol.ENERGY] = -1
        priv = np.array(self.private)
        for c in (SeatCol.DRAW_COUNT, SeatCol.DISCARD_COUNT, SeatCol.EXHAUST_COUNT):
            a[:, c] = np.where(priv, 0, -1) if self.in_combat else 0
        self.b.extend(Table.SEATS, a)

    def creatures(self) -> None:
        r, g, ctx, b = self.rng, self.g, self.ctx, self.b
        seats = b.rows(Table.SEATS)
        ally = 0
        for s in range(self.n):
            row = b.add(Table.CREATURES, kind=CreatureKind.PLAYER, seat=s, pos=ally, model_id=ctx.characters[s],
                        hp=ctx.hp[s], max_hp=ctx.max_hp[s], hp_display=g.vocab.id("enum:HpDisplay", "Normal"),
                        block=int(seats[s, SeatCol.BLOCK]), alive=int(ctx.hp[s] > 0), hittable=int(ctx.hp[s] > 0))
            ally += 1
            self.player_rows[s] = row
            seats[s, SeatCol.CREATURE_ROW] = row
            n_pets = (int(ctx.characters[s] == g.char_necro) or int(r.random() < 0.05)) if self.size != "extreme" else 2
            for _ in range(n_pets):
                sentinel = r.random() < 0.3
                mhp = int(r.integers(5, 40))
                b.add(Table.CREATURES, kind=CreatureKind.PET, seat=s, pos=ally, model_id=g.rid(r, "monster"),
                      hp=-1 if sentinel else int(r.integers(1, mhp + 1)), max_hp=-1 if sentinel else mhp,
                      hp_display=g.rid(r, "enum:HpDisplay"), alive=1, hittable=0, slot_id=0)
                ally += 1
        n_enemies = len(ctx.enemies) or 1 + self.count("extra_enemies")
        models = ctx.enemies or tuple(int(m) for m in g.rids(r, "monster", n_enemies))
        for i, m in enumerate(models):
            mhp = int(r.integers(10, 300))
            alive = r.random() > 0.1 or i == 0
            row = b.add(Table.CREATURES, kind=CreatureKind.ENEMY, seat=-1, pos=i, model_id=m,
                        hp=int(r.integers(1, mhp + 1)) if alive else 0, max_hp=mhp,
                        hp_display=g.vocab.id("enum:HpDisplay", "Normal"), block=int(r.integers(0, 20) * (r.random() < 0.3)),
                        alive=int(alive), hittable=int(alive and (r.random() < 0.95 or i == 0)),
                        slot_id=g.rid(r, "slot_name"), poison_next_turn=int(r.integers(0, 12) * (r.random() < 0.2)))
            self.enemy_rows.append(row)
        crea = b.rows(Table.CREATURES)
        self.hittable = [e for e in self.enemy_rows if crea[e, CreatureCol.HITTABLE]]
        n_crea = b.n_rows(Table.CREATURES)
        all_mask = (1 << self.n) - 1
        for c in range(n_crea):
            k = self.count("powers")
            if k:
                p = b.new_rows(Table.POWERS, k)
                p[:, PowerCol.POWER_ID] = g.rids(r, "power", k)
                counter = r.random(k) < 0.7
                p[:, PowerCol.STACK_TYPE] = np.where(counter, g.vocab.id("enum:PowerStackType", "Counter"),
                                                     g.vocab.id("enum:PowerStackType", "Single"))
                p[:, PowerCol.AMOUNT] = np.where(counter, r.integers(-3, 12, k), 0)
                p[:, PowerCol.TYPE_NOW] = g.rids(r, "enum:PowerType", k)
                p[:, PowerCol.APPLIER_ROW] = np.where(r.random(k) < 0.4, r.integers(0, n_crea, k), -1)
                masks = np.where(r.random(k) < 0.9, all_mask, 1 << r.integers(0, self.n, k))
                p[:, PowerCol.VISIBLE_MASK] = masks
                p[:, PowerCol.ORDER] = np.arange(k)
                vis = (masks & sum(1 << s for s in self.view_seats)) != 0
                p = p[vis]
                p[:, PowerCol.ORDER] = np.arange(len(p))
                rows = b.attach(Table.POWERS, c, p)
                for pr in rows:
                    if r.random() < 0.3:
                        b.attach_vars(Table.POWERS, pr, self.vars_rows(1))
            if crea[c, CreatureCol.KIND] == CreatureKind.ENEMY and crea[c, CreatureCol.ALIVE]:
                k = 1 + self.count("intents")
                it = b.new_rows(Table.INTENTS, k)
                it[:, IntentCol.IDX] = np.arange(k)
                types = g.rids(r, "enum:IntentType", k)
                attack = np.isin(types, list(g.attack_intents)) | (r.random(k) < 0.4)
                types = np.where(attack, next(iter(g.attack_intents)), types)
                it[:, IntentCol.INTENT_TYPE] = types
                it[:, IntentCol.HITS] = np.where(attack, r.choice([1, 1, 1, 2, 3, 4], k), 0)
                it[:, IntentCol.CARD_COUNT] = np.where(types == g.status_intent, r.integers(1, 4, k), 0)
                dmg = np.full((k, 4), -1, dtype=np.int64)
                for s in self.view_seats:
                    dmg[:, s] = np.where(attack, r.integers(0, 35, k), -1)
                it[:, IntentCol.DMG_SEAT0 : IntentCol.DMG_SEAT3 + 1] = dmg
                it[:, IntentCol.FLAGS] = np.where(r.random(k) < 0.95, int(IntentFlag.HAS_TIP), 0)
                b.attach(Table.INTENTS, c, it)
        cap = seats[:, SeatCol.ORB_CAPACITY]
        for s in range(self.n):
            k = int(cap[s]) if self.size == "extreme" else int(r.integers(0, cap[s] + 1))
            if k:
                o = b.new_rows(Table.ORBS, k)
                o[:, OrbCol.SEAT], o[:, OrbCol.POS], o[:, OrbCol.ORB_ID] = s, np.arange(k), g.rids(r, "orb", k)
                o[:, OrbCol.PASSIVE], o[:, OrbCol.EVOKE] = r.integers(0, 10, k), r.integers(0, 25, k)
                b.extend(Table.ORBS, o)
            k = int(seats[s, SeatCol.ARROW_SLOTS])
            if k:
                ar = b.new_rows(Table.ARROWS, k)
                ar[:, ArrowCol.SEAT], ar[:, ArrowCol.SLOT], ar[:, ArrowCol.STATUS] = s, np.arange(k), g.rids(r, "enum:ArrowStatus", k)
                b.extend(Table.ARROWS, ar)

    def combat_cards(self) -> None:
        if not self.in_combat:
            return
        r, b = self.rng, self.b
        seats = b.rows(Table.SEATS)
        priv = [s for s in range(self.n) if self.private[s]]
        for s in range(self.n):  # HAND rows of every private seat first (zone-major order)
            k = self.count("hand")
            seats[s, SeatCol.HAND_COUNT] = k
            if s not in priv or not k:
                continue
            rows = b.extend(Table.CARDS, self.cards(k, Zone.HAND, s))
            vs = self.card_vars(k)
            for i, row in enumerate(rows):
                tk = (int(r.choice([_TGT_NONE, _TGT_ENEMY, _TGT_ENEMY, _TGT_ALLY if self.n > 1 else _TGT_NONE]))
                      if self.size == "typical" else _TGT_ENEMY)
                v = vs[i]
                if (tk == _TGT_ENEMY and len(v) and self.ctx.reset_flags & ResetFlag.TARGET_PREVIEWS):
                    extra = []
                    for e in self.hittable:
                        if r.random() < 0.5:
                            t = v[:1].copy()
                            t[:, VarCol.TARGET_ROW] = e
                            t[:, VarCol.SHOWN] += int(r.integers(1, 6))
                            extra.append(t)
                    if extra:
                        self.has_target_preview = True
                        v = np.concatenate([v, *extra])
                b.attach_vars(Table.CARDS, row, v)
                self.hand[s].append((row, tk))
        for zone, name, col, dup in ((Zone.DRAW, "draw", SeatCol.DRAW_COUNT, 0.4),
                                     (Zone.DISCARD, "discard", SeatCol.DISCARD_COUNT, 0.4),
                                     (Zone.EXHAUST, "exhaust", SeatCol.EXHAUST_COUNT, 0.2)):
            for s in priv:
                rows = self.multiset(zone, s, self.count(name), dup)
                seats[s, col] = self.zone_count(rows)
                if zone == Zone.DRAW:
                    self.draw_groups[s] = rows

    def relics_owned(self) -> None:
        r, g, b = self.rng, self.g, self.b
        for s in range(self.n):
            k = self.count("relics", 1)
            a = b.new_rows(Table.RELICS, k)
            a[:, RelicCol.SEAT], a[:, RelicCol.ZONE], a[:, RelicCol.POS] = s, ItemZone.OWNED, np.arange(k)
            a[:, RelicCol.RELIC_ID] = g.rids(r, "relic", k)
            a[:, RelicCol.COUNTER] = np.where(r.random(k) < 0.25, r.integers(0, 10, k), -1)
            a[:, RelicCol.STATUS] = g.rids(r, "enum:RelicStatus", k)
            a[:, RelicCol.FLAGS] = (r.random(k) < 0.05) * RelicFlag.USED_UP | (r.random(k) < 0.02) * RelicFlag.WAX
            for row in b.extend(Table.RELICS, a):
                if r.random() < 0.6:
                    b.attach_vars(Table.RELICS, row, self.vars_rows(int(r.integers(1, 3))))

    def potions_owned(self) -> None:
        r, g, b = self.rng, self.g, self.b
        slots = b.rows(Table.SEATS)[:, SeatCol.POTION_SLOTS]
        usable_kind = self.kind == Kind.COMBAT or self.kind in _ROOM_LEVEL
        p_use = (0.7 if self.kind == Kind.COMBAT else 0.3) if self.size == "typical" else 1.0
        for s in range(self.n):
            k = min(int(slots[s]), self.count("potions"))
            self.open_slots[s] = int(slots[s]) - k
            if not k:
                continue
            a = b.new_rows(Table.POTIONS, k)
            a[:, PotionCol.SEAT], a[:, PotionCol.ZONE] = s, ItemZone.OWNED
            a[:, PotionCol.SLOT] = np.sort(r.choice(int(slots[s]), k, replace=False))
            a[:, PotionCol.POTION_ID] = g.rids(r, "potion", k)
            acting = s in self.acting and usable_kind
            usable = acting & (r.random(k) < p_use)
            disc = r.random(k) < 0.95
            a[:, PotionCol.FLAGS] = usable * PotionFlag.USABLE | disc * PotionFlag.DISCARDABLE
            for i, row in enumerate(b.extend(Table.POTIONS, a)):
                if r.random() < 0.5:
                    b.attach_vars(Table.POTIONS, row, self.vars_rows(1))
                tk = int(r.choice([_TGT_NONE, _TGT_ENEMY, _TGT_ALLY])) if self.kind == Kind.COMBAT else _TGT_NONE
                self.potions.append((row, s, int(a[i, PotionCol.FLAGS]), tk))

    def map_tables(self) -> None:
        ctx, b = self.ctx, self.b
        if ctx.map is None:
            ctx.map = self.g.make_map(self.rng, self.size, ctx.ascension)
        m = ctx.map
        nodes = m.nodes.copy()
        for nr in ctx.path[:-1]:
            nodes[nr, MapNodeCol.STATE] = MapNodeState.VISITED
        if ctx.path:
            nodes[ctx.path[-1], MapNodeCol.STATE] = MapNodeState.CURRENT
        visited = np.zeros(len(nodes), dtype=bool)
        visited[ctx.path] = True
        nodes[~visited, MapNodeCol.RESOLVED_ROOM] = 0
        for nr in self.g.reachable(ctx):
            nodes[nr, MapNodeCol.STATE] = MapNodeState.REACHABLE
        b.extend(Table.MAP_NODES, nodes)
        b.extend(Table.MAP_EDGES, m.edges)

    # ------------------------------------------------------------------ kinds
    def k_map(self) -> None:
        b, r = self.b, self.rng
        reach = self.g.reachable(self.ctx)
        voters = list(self.acting)
        if self.n > 1 and len(voters) > 1:
            voted = [s for s in voters[1:] if r.random() < 0.4]
            nodes = b.rows(Table.MAP_NODES)
            for s in voted:
                nodes[int(r.choice(reach)), MapNodeCol.FLAGS] |= MapNodeFlag.VOTE_SEAT0 << s
            voters = [s for s in voters if s not in voted]
        for s in voters:
            for nr in reach:
                self.act(Verb.TRAVEL, s, Table.MAP_NODES, nr)

    def k_combat(self) -> None:
        seats = self.b.rows(Table.SEATS)
        players = [p for p in self.player_rows if p >= 0]
        for s in self.acting:
            if seats[s, SeatCol.READY_TO_END_TURN]:
                self.act(Verb.UNDO_END_TURN, s)
                continue
            cards = self.b.rows(Table.CARDS)
            for row, tk in self.hand[s]:
                if not cards[row, CardCol.HAND_FLAGS] & HandFlag.CAN_PLAY:
                    continue
                for t in self.targets(s, tk, players):
                    self.act(Verb.PLAY_CARD, s, Table.CARDS, row, t)
            self.act(Verb.END_TURN, s)
        for row, s, flags, tk in self.potions:
            if s not in self.acting:
                continue
            if flags & PotionFlag.USABLE:
                for t in self.targets(s, tk, players):
                    self.act(Verb.USE_POTION, s, Table.POTIONS, row, t)
            if flags & PotionFlag.DISCARDABLE:
                self.act(Verb.DISCARD_POTION, s, Table.POTIONS, row)

    def targets(self, seat: int, tk: int, players: list[int]) -> list[int]:
        if tk == _TGT_ENEMY:
            return self.hittable
        if tk == _TGT_ALLY:
            return [p for p in players if p != self.player_rows[seat]] or [-1]
        return [-1]

    def room_potion_actions(self) -> None:
        for row, s, flags, _ in self.potions:
            if s not in self.acting:
                continue
            if flags & PotionFlag.USABLE:
                self.act(Verb.USE_POTION, s, Table.POTIONS, row)
            if flags & PotionFlag.DISCARDABLE:
                self.act(Verb.DISCARD_POTION, s, Table.POTIONS, row)

    def k_card_select(self) -> None:
        r, g, b, sub = self.rng, self.g, self.b, self.sub
        s = self.acting[0]
        want = self.count("choice", 1)
        if sub in _HAND_SUBS:
            src = [row for row, _ in self.hand[s]]
        elif sub == CardSelectSub.COMBAT_PILE:
            src = list(self.draw_groups[s])
        elif sub in _DECK_SUBS:
            src = list(self.deck_groups[s])
        else:
            src = None
        if src is not None and not src:
            raise _Retry  # this draw has no cards in the prompt's source pile
        cards = b.rows(Table.CARDS)
        if src is not None:
            pick = sorted(int(x) for x in r.choice(src, min(want, len(src)), replace=False))
            a = cards[pick].copy()
            a[:, CardCol.ZONE], a[:, CardCol.LINK_ROW] = Zone.CHOICE, pick
            a[:, [CardCol.ENERGY_COST_COLOR, CardCol.STAR_COST_COLOR, CardCol.HAND_FLAGS,
                  CardCol.UNPLAYABLE_REASON]] = 0
            pile = "Hand" if sub in _HAND_SUBS else ("Draw" if sub == CardSelectSub.COMBAT_PILE else "Deck")
            a[:, CardCol.AUX] = g.pile[pile]
            if sub not in _HAND_SUBS:
                a[:, CardCol.POS] = -1
            src_vars: list[I64 | None] = [self.untargeted_vars(p) for p in pick]
        else:
            a = self.cards(want, Zone.CHOICE, s)
            a[:, CardCol.POS], a[:, CardCol.AUX] = np.arange(want), 0
            src_vars = list(self.card_vars(want))
        a[:, CardCol.VAR_OFF], a[:, CardCol.VAR_CNT], a[:, CardCol.SEL_COUNT] = -1, 0, 0
        choice = b.extend(Table.CARDS, a)
        for row, v in zip(choice, src_vars):
            b.attach_vars(Table.CARDS, row, v)
        multi = r.random() < 0.3
        sel_min, sel_max = (int(r.integers(0, 2)), int(r.integers(2, 4))) if multi else (1, 1)
        manual = sel_min != sel_max
        flags = (manual * SelFlag.MANUAL_CONFIRM | (r.random() < 0.2) * SelFlag.CANCELABLE
                 | (sub == CardSelectSub.CHOOSE_A_CARD and r.random() < 0.5) * SelFlag.CAN_SKIP)
        cards = b.rows(Table.CARDS)
        sel = 0
        if manual:
            for row in choice:
                if sel < sel_max - 1 and r.random() < 0.3:
                    cards[row, CardCol.SEL_COUNT] = 1
                    sel += 1
        if sub in _UPGRADE_SUBS and self.ctx.reset_flags & ResetFlag.UPGRADE_PREVIEWS:
            flags |= SelFlag.UPGRADE_PREVIEW
            self.has_upgrade_preview = True
            pv = cards[list(choice)].copy()
            pv[:, CardCol.ZONE], pv[:, CardCol.POS], pv[:, CardCol.COUNT] = Zone.PREVIEW, -1, 1
            pv[:, CardCol.UPGRADE_LEVEL] += 1
            pv[:, CardCol.LINK_ROW], pv[:, CardCol.AUX], pv[:, CardCol.SEL_COUNT] = list(choice), -1, 0
            pv[:, CardCol.VAR_OFF], pv[:, CardCol.VAR_CNT] = -1, 0
            for row, v in zip(b.extend(Table.CARDS, pv), src_vars):
                if v is not None and len(v):
                    u = v.copy()
                    u[:, VarCol.SHOWN] += 3
                    u[:, VarCol.BASE] += 3
                    u[:, VarCol.ENCHANTED] += 3
                    u[:, VarCol.FLAGS] |= VarFlag.JUST_UPGRADED
                    b.attach_vars(Table.CARDS, row, u)
        gl = b.glob
        gl[G.PROMPT_KEY] = g.rid(r, "prompt_key")
        if r.random() < 0.7:
            gl[G.PROMPT_SOURCE_CAT], gl[G.PROMPT_SOURCE_ID] = RefCategory.CARD, g.rid(r, "card")
        gl[G.SEL_MIN], gl[G.SEL_MAX], gl[G.SEL_COUNT], gl[G.SEL_FLAGS] = sel_min, sel_max, sel, flags
        if sel < sel_max:
            for row in choice:
                if cards[row, CardCol.COUNT] > cards[row, CardCol.SEL_COUNT]:
                    self.act(Verb.SELECT_CARD, s, Table.CARDS, row)
        if manual and sel >= sel_min:
            self.act(Verb.CONFIRM_SELECTION, s)
        if flags & SelFlag.CANCELABLE:
            self.act(Verb.CANCEL, s)
        if flags & SelFlag.CAN_SKIP:
            self.act(Verb.SKIP, s)

    def untargeted_vars(self, card_row: int) -> I64 | None:
        v = self.b.pending_vars(Table.CARDS, card_row)
        return None if v is None else v[v[:, VarCol.TARGET_ROW] == -1].copy()

    def offer_card(self, seat: int, pos: int, group: int = -1) -> int:
        a = self.cards(1, Zone.OFFER, seat)
        a[0, CardCol.POS], a[0, CardCol.AUX] = pos, group
        row = self.b.extend(Table.CARDS, a).start
        self.b.attach_vars(Table.CARDS, row, self.card_vars(1)[0])
        return row

    def offer_relic(self, seat: int, pos: int) -> int:
        r = self.rng
        row = self.b.add(Table.RELICS, seat=seat, zone=ItemZone.OFFER, pos=pos, relic_id=self.g.rid(r, "relic"),
                         counter=-1, status=self.g.vocab.id("enum:RelicStatus", "Normal"))
        if r.random() < 0.6:
            self.b.attach_vars(Table.RELICS, row, self.vars_rows(1))
        return row

    def offer_potion(self, seat: int, pos: int) -> int:
        row = self.b.add(Table.POTIONS, seat=seat, zone=ItemZone.OFFER, slot=pos, potion_id=self.g.rid(self.rng, "potion"))
        if self.rng.random() < 0.5:
            self.b.attach_vars(Table.POTIONS, row, self.vars_rows(1))
        return row

    def option(self, kind: int, seat: int, pos: int, *, refs: Sequence[tuple[int, int, int, int]] = (),
               vars_k: int = 0, **cols: int) -> int:
        cols.setdefault("price", -1)
        cols.setdefault("amount", -1)
        cols.setdefault("group", -1)
        row = self.b.add(Table.OPTIONS, kind=kind, seat=seat, pos=pos, **cols)
        if vars_k:
            v = self.vars_rows(vars_k)
            v[:, VarCol.FLAGS] |= VarFlag.LOCSTRING
            self.b.attach_vars(Table.OPTIONS, row, v)
        if refs:
            ref = self.b.new_rows(Table.REFS, len(refs))
            ref[:, [RefCol.CATEGORY, RefCol.VOCAB_ID, RefCol.ENT_TABLE, RefCol.ENT_ROW]] = refs
            self.b.attach(Table.REFS, row, ref)
        return row

    def k_reward(self) -> None:
        r, g = self.rng, self.g
        s = self.acting[0]
        no_skip = r.random() < 0.1
        if self.ctx.acting_seat is not None:
            # H5 fake fights expose terminal reward screens, which always permit leaving.
            no_skip = False
        self.b.glob[G.REWARD_FLAGS] = no_skip * RewardFlag.NO_SKIP
        names = ["Gold", "Potion", "Relic", "Card", "Card", "SpecialCard", "RemoveCard", "UpgradeCard", "Linked"]
        pos, group = 0, 0
        for _ in range(self.count("reward_items", 1)):
            name = str(r.choice(names))
            items = [str(r.choice(["Relic", "Potion", "Gold"])) for _ in range(2)] if name == "Linked" else [name]
            grp = group if name == "Linked" else -1
            group += name == "Linked"
            for it in items:
                key = g.reward[it]
                ref_t, ref_r, amount, vk = -1, -1, -1, 0
                if it == "Gold":
                    amount, vk = int(r.integers(10, 120)), int(r.random() < 0.5)
                elif it == "Potion":
                    ref_t, ref_r = Table.POTIONS, self.offer_potion(s, pos)
                elif it == "Relic":
                    ref_t, ref_r = Table.RELICS, self.offer_relic(s, pos)
                elif it == "SpecialCard":
                    ref_t, ref_r = Table.CARDS, self.offer_card(s, pos)
                opt = self.option(OptionKind.REWARD_ITEM, s, pos, key_id=key, amount=amount, ref_table=ref_t,
                                  ref_row=ref_r, group=grp, vars_k=vk)
                if it != "Potion" or self.open_slots[s] > 0:
                    self.act(Verb.CHOOSE_OPTION, s, Table.OPTIONS, opt)
                pos += 1
        if not no_skip or not self.actions:
            self.act(Verb.PROCEED, s)

    def k_card_reward(self) -> None:
        r, g = self.rng, self.g
        s = self.acting[0]
        can_reroll = r.random() < 0.2
        self.b.glob[G.CARD_REWARD_FLAGS] = CardRewardFlag.CAN_SKIP | can_reroll * CardRewardFlag.CAN_REROLL
        for i in range(3 if self.size == "typical" else 5):
            self.act(Verb.PICK_OFFER_CARD, s, Table.CARDS, self.offer_card(s, i, 0))
        alts = ["SKIP"] + (["REROLL"] if can_reroll else [])
        for i, name in enumerate(alts):
            self.act(Verb.CHOOSE_OPTION, s, Table.OPTIONS,
                     self.option(OptionKind.CARD_REWARD_ALT, s, i, key_id=g.alt[name]))

    def k_event(self) -> None:
        r, g = self.rng, self.g
        s = self.acting[0]
        gl = self.b.glob
        ancient = r.random() < 0.1
        gl[G.EVENT_ID], gl[G.EVENT_PAGE_KEY] = g.rid(r, "event"), g.rid(r, "event_page")
        gl[G.EVENT_FLAGS] = (self.n > 1 and r.random() < 0.3) * EventFlag.SHARED | ancient * EventFlag.ANCIENT
        if self.ctx.acting_seat is not None:
            gl[G.EVENT_FLAGS] = self.ctx.shared_vote * EventFlag.SHARED | ancient * EventFlag.ANCIENT
        k = self.count("event_opts", 1)
        unlocked = 0
        for i in range(k):
            refs: list[tuple[int, int, int, int]] = []
            ref_t = ref_r = -1
            if ancient and r.random() < 0.7:
                ref_t, ref_r = Table.RELICS, self.offer_relic(s, i)
                refs.append((RefCategory.RELIC, int(self.b.rows(Table.RELICS)[ref_r, RelicCol.RELIC_ID]), ref_t, ref_r))
            if r.random() < 0.3:
                cr = self.offer_card(s, i)
                refs.append((RefCategory.CARD, int(self.b.rows(Table.CARDS)[cr, CardCol.CARD_ID]), Table.CARDS, cr))
            if r.random() < 0.1:
                refs.append((RefCategory.POWER, g.rid(r, "power"), -1, -1))
            locked = r.random() < 0.1 and (i < k - 1 or unlocked)
            flags = (locked * OptionFlag.LOCKED | (r.random() < 0.05) * OptionFlag.LETHAL
                     | (k == 1 and r.random() < 0.5) * OptionFlag.IS_PROCEED)
            opt = self.option(OptionKind.EVENT_OPTION, s, i, key_id=g.rid(r, "event_option_key"), flags=flags,
                              ref_table=ref_t, ref_row=ref_r, refs=refs, vars_k=int(r.choice([0, 0, 1, 2])))
            if not locked:
                unlocked += 1
                self.act(Verb.CHOOSE_OPTION, s, Table.OPTIONS, opt)

    def k_shop(self) -> None:
        r, g = self.rng, self.g
        s = self.acting[0]
        gold = self.ctx.gold[s]
        self.b.glob[G.SHOP_FLAGS] = (r.random() < 0.05) * 1
        layout = [("CHARACTER_CARD", 5), ("COLORLESS_CARD", 2), ("RELIC", 3), ("POTION", 3), ("CARD_REMOVAL", 1)]
        pos = 0
        sale = int(r.integers(0, 5))
        for section, k in layout:
            for j in range(k):
                ref_t, ref_r = -1, -1
                if section.endswith("_CARD"):
                    ref_t, ref_r = Table.CARDS, self.offer_card(s, pos)
                    price = int(r.integers(45, 160))
                elif section == "RELIC":
                    ref_t, ref_r, price = Table.RELICS, self.offer_relic(s, pos), int(r.integers(140, 300))
                elif section == "POTION":
                    ref_t, ref_r, price = Table.POTIONS, self.offer_potion(s, pos), int(r.integers(48, 110))
                else:
                    price = 75 + 25 * int(r.integers(0, 4))
                on_sale = section == "CHARACTER_CARD" and j == sale and r.random() < 0.5
                stocked = r.random() < 0.9
                flags = (on_sale * OptionFlag.ON_SALE | stocked * OptionFlag.STOCKED
                         | (gold >= price) * OptionFlag.AFFORDABLE)
                opt = self.option(OptionKind.SHOP_ENTRY, s, pos, key_id=g.shop[section], flags=flags, price=price,
                                  ref_table=ref_t, ref_row=ref_r)
                if stocked and gold >= price and (section != "POTION" or self.open_slots[s] > 0):
                    self.act(Verb.CHOOSE_OPTION, s, Table.OPTIONS, opt)
                pos += 1
        self.act(Verb.PROCEED, s)

    def k_rest(self) -> None:
        r, g = self.rng, self.g
        s = self.acting[0]
        names = ["HEAL", "SMITH"] + [str(x) for x in r.choice(list(g.rest), int(r.integers(0, 3)), replace=False)
                                     if x not in ("HEAL", "SMITH") and (x != "MEND" or self.n > 1)]
        chosen = r.random() < 0.3
        if self.ctx.acting_seat is not None:
            chosen = s in self.ctx.rest_done
        enabled = 0
        for i, name in enumerate(names):
            locked = r.random() < 0.15
            opt = self.option(OptionKind.REST_OPTION, s, i, key_id=g.rest[name], flags=locked * OptionFlag.LOCKED,
                              vars_k=int(name == "HEAL"))
            if not locked:
                enabled += 1
                if name == "MEND" and self.n > 1:
                    for target in range(self.n):
                        if target != s:
                            self.act(Verb.CHOOSE_OPTION, s, Table.OPTIONS, opt, aux=target)
                else:
                    self.act(Verb.CHOOSE_OPTION, s, Table.OPTIONS, opt)
        chosen = chosen or not enabled
        self.b.glob[G.REST_FLAGS] = (enabled > 0) * RestFlag.HAS_OPTIONS | chosen * RestFlag.CHOSEN
        if chosen:
            self.act(Verb.PROCEED, s)

    def k_treasure(self) -> None:
        s = self.acting[0]
        self.b.glob[G.TREASURE_PHASE] = self.sub
        if self.sub == TreasurePhase.CHEST:
            self.act(Verb.OPEN_CHEST, s)
            self.act(Verb.PROCEED, s)
            return
        for i in range(max(1, self.n)):
            rr = self.offer_relic(s, i)
            votes = sum(1 << (7 + seat) for seat, choice in self.ctx.treasure_votes.items() if choice == i)
            opt = self.option(OptionKind.TREASURE_RELIC, s, i, ref_table=Table.RELICS, ref_row=rr,
                              flags=votes)
            self.act(Verb.CHOOSE_OPTION, s, Table.OPTIONS, opt)
        if self.n == 1 or self.ctx.acting_seat is None:
            self.act(Verb.SKIP, s)

    def k_confirm(self) -> None:
        s = self.acting[0]
        self.b.glob[G.CONFIRM_KEY] = self.g.vocab.id("confirm_key", "TRIAL_DOUBLE_DOWN")
        for i, key in enumerate(self.g.confirm):
            self.act(Verb.CHOOSE_OPTION, s, Table.OPTIONS, self.option(OptionKind.CONFIRM_CHOICE, s, i, key_id=key))

    def k_bundle(self) -> None:
        r = self.rng
        s = self.acting[0]
        can_skip = r.random() < 0.5
        self.b.glob[G.BUNDLE_FLAGS] = can_skip * BundleFlag.CAN_SKIP
        nb = int(r.integers(2, 4))
        bundles = [[self.offer_card(s, j, bi) for j in range(3)] for bi in range(nb)]
        cards = self.b.rows(Table.CARDS)
        for bi, rows in enumerate(bundles):
            refs = [(RefCategory.CARD, int(cards[cr, CardCol.CARD_ID]), Table.CARDS, cr) for cr in rows]
            self.act(Verb.CHOOSE_OPTION, s, Table.OPTIONS,
                     self.option(OptionKind.BUNDLE, s, bi, group=bi, refs=refs))
        if can_skip:
            self.act(Verb.SKIP, s)

    def k_crystal(self) -> None:
        r, g = self.rng, self.g
        s = self.acting[0]
        c = self.b.new_rows(Table.CELLS, 121)
        yy, xx = np.divmod(np.arange(121), 11)
        c[:, CellCol.X], c[:, CellCol.Y] = xx, yy
        hidden = np.ones(121, dtype=bool) if self.size == "extreme" else r.random(121) < r.uniform(0.5, 0.95)
        hidden[int(r.integers(0, 121))] = True
        c[:, CellCol.HIDDEN] = hidden
        n_items = int(r.integers(0, 6))
        item_of = np.where(r.random(121) < 0.15, r.integers(0, max(1, n_items), 121), -1) if n_items else np.full(121, -1)
        types = g.rids(r, "enum:CrystalSphereItemType", max(1, n_items))
        show = ~hidden & (item_of >= 0)
        c[:, CellCol.ITEM_TYPE] = np.where(hidden, -1, np.where(show, types[np.maximum(item_of, 0)], 0))
        c[:, CellCol.ITEM_GOOD] = np.where(show, (item_of % 2 == 0), -1)
        c[:, CellCol.ITEM_IDX] = np.where(show, item_of, -1)
        rows = self.b.extend(Table.CELLS, c)
        gl = self.b.glob
        gl[G.CRYSTAL_DIVINATIONS], gl[G.CRYSTAL_TOOL] = int(r.integers(0, 4)), self.g.tools[0]
        for cell in np.flatnonzero(hidden).tolist():
            for tool in g.tools:
                self.act(Verb.CRYSTAL_CLICK, s, Table.CELLS, rows.start + cell, -1, tool)

    def k_terminal(self) -> None:
        return

    # ------------------------------------------------------------------ globals / flags
    def finish(self, with_map: bool, with_deck: bool) -> None:
        r, g, ctx, b = self.rng, self.g, self.ctx, self.b
        if self.actions:
            b.extend(Table.ACTIONS, np.array([(*a, 0) for a in self.actions], dtype=np.int64))
        gl = b.glob
        gl[G.KIND], gl[G.KIND_SUB], gl[G.N_SEATS] = self.kind, self.sub, self.n
        gl[G.VIEW_MODE], gl[G.VIEW_SEAT] = ctx.view_mode, ctx.view_seat
        gl[G.ACTING_SEAT] = self.acting[0] if len(self.acting) == 1 else -1
        gl[G.ASCENSION], gl[G.GAME_MODE] = ctx.ascension, g.vocab.id("enum:GameMode", "Standard")
        gl[G.ACT_INDEX], gl[G.ACT_ID] = ctx.act_index, ctx.act_id
        gl[G.ACT_FLOOR], gl[G.TOTAL_FLOOR] = ctx.act_floor, ctx.total_floor
        m = ctx.map
        if m is not None:
            gl[G.MAP_ROWS] = m.n_map_rows
            cur = ctx.path[-1] if ctx.path else -1
            gl[G.CUR_ROW] = m.nodes[cur, MapNodeCol.ROW] if cur >= 0 else -1
            gl[G.CUR_COL] = m.nodes[cur, MapNodeCol.COL] if cur >= 0 else -1
        room = {Kind.COMBAT: "Monster", Kind.EVENT: "Event", Kind.SHOP: "Shop", Kind.REST: "RestSite",
                Kind.TREASURE: "Treasure", Kind.MAP: "Map"}.get(self.kind, "Event")
        if self.in_combat:  # combat room = the encounter's room type; event-started fights are Monster (E-22)
            room = ctx.room or "Monster"
        elif self.kind == Kind.TERMINAL and ctx.room:
            room = ctx.room
        gl[G.ROOM_TYPE] = g.room.get(room, 0)
        rr = np.random.default_rng((ctx.episode_id, ctx.act_index))  # stable per act
        gl[G.BOSS_ENCOUNTER] = g.rid(rr, "encounter")
        gl[G.SECOND_BOSS_ENCOUNTER] = g.rid(rr, "encounter") if ctx.ascension >= 10 else 0
        gl[G.ANCIENT_EVENT] = g.rid(rr, "event")
        gl[G.IN_COMBAT] = self.in_combat
        if self.in_combat:
            gl[G.COMBAT_ROUND] = ctx.combat_round
            gl[G.COMBAT_SIDE] = g.vocab.id("enum:CombatSide", "Player")
            gl[G.PLAYER_PHASE] = g.vocab.id("enum:PlayerTurnPhase", "Main")
        gl[G.PLAYERS_ALIVE] = sum(h > 0 for h in ctx.hp)
        if ctx.acting_seat is not None:
            for s in range(self.n):
                gl[52 + s] = int(self.kind) if s in self.acting else 0
                gl[56 + s] = (int(s in ctx.room_done) | (int(s in ctx.treasure_votes) << 4)
                              | (int(s in ctx.rest_done) << 5))
        gl[G.COMBATS_WON], gl[G.ELITES_WON], gl[G.BOSSES_WON] = ctx.combats_won, ctx.elites_won, ctx.bosses_won
        gl[G.LAST_WON_ROOM_TYPE] = g.room.get(ctx.last_won_room, 0) if ctx.last_won_room else 0
        gl[G.LAST_WON_TOTAL_FLOOR] = ctx.last_won_floor
        b.vocab_hash, b.catalog_hash = g.vocab.hash, g.catalog_hash
        b.episode_id, b.step_id = ctx.episode_id, ctx.step_id
        b.flags = int(
            (self.kind == Kind.TERMINAL) * FrameFlag.TERMINAL
            | with_map * FrameFlag.HAS_MAP
            | (self.in_combat and with_deck) * FrameFlag.DECK_IN_COMBAT
            | (ctx.view_mode == ViewMode.TEAM) * FrameFlag.TEAM_VIEW
            | self.has_upgrade_preview * FrameFlag.UPGRADE_PREVIEW
            | self.has_target_preview * FrameFlag.TARGET_PREVIEW
        )
        b.host_us = int(r.integers(30, 250))


class _Retry(Exception):
    """Internal: the random draw could not satisfy a kind's preconditions; draw again."""


def random_frame(gen: FrameGen, rng: np.random.Generator, kind: int, *, n_seats: int = 1,
                 size: SizeMode = "typical", kind_sub: int | None = None, hashes: bool = True,
                 view: int | None = None) -> bytes:
    """A random OBS frame in a fresh random context."""
    ctx = gen.random_ctx(rng, n_seats, view=view, size=size)
    return gen.frame(rng, kind, ctx, kind_sub=kind_sub, size=size, hashes=hashes)


# =========================================================================== fake environment


@dataclass(frozen=True)
class FakeHostConfig:
    """Episode model of :class:`FakeGame`.

    Episode length (decisions) ~ lognormal(log(mean_decisions), length_sigma) clipped to
    [min_decisions, max_decisions]; the episode ends with WIN (prob ``win_prob``) or LOSS,
    or with ABORTED at a uniform random point (prob ``abort_prob`` per episode).
    ``error_prob``: per STEP, answer ERROR(BAD_MESSAGE) + re-send instead (fault injection).

    ``learnable``: the outcome depends on the policy instead of ``win_prob``. Every END_TURN
    chosen while a PLAY_CARD action was legal costs every seat ``end_turn_penalty`` HP; the
    team loses when no seat has HP left and wins if it survives the planned length (an
    observable, context-dependent rule for end-to-end learning tests).

    ``frame_pool`` > 0: frames are not generated per step but drawn from a pool of that many
    frames per (lineup, ascension, view, kind, sub), generated on first use, with the header
    (episode / step id, auto steps) patched - the host side then costs microseconds per step
    (actor overhead benchmarks). Pooled frames do not track floor / HP / map / win progress.

    Episode ends as the real host reports them (errata-v1 P-5 / P-8, section 5): WIN, DEATH,
    ABORTED (GLOBAL[ABORT_REASON] = :data:`FAKE_ABORT_REASON`), DECISION_CAP when the RESET's
    ``max_decisions`` (or ``decision_cap``, the host's ``--max-decisions`` default; 0 = none) is
    reached first, and COMBAT_TURN_CAP with probability ``turn_cap_prob`` per episode. Every frame
    carries the public win counters GLOBAL 44..48; TERMINAL frames carry 49..51.
    """

    win_prob: float = 0.5
    mean_decisions: float = 300.0
    length_sigma: float = 0.6
    min_decisions: int = 3
    max_decisions: int = 4000
    auto_step_prob: float = 0.15
    max_auto_steps: int = 3
    abort_prob: float = 0.01
    error_prob: float = 0.0
    size: SizeMode = "typical"
    hashes: bool = True
    learnable: bool = False
    end_turn_penalty: int = 8
    frame_pool: int = 0
    decision_cap: int = 0
    turn_cap_prob: float = 0.0


@dataclass
class _Episode:
    seed: int
    ctx: RunCtx
    length: int
    result: int
    reason: int
    sched: Generator[tuple[int, int], int, None]
    rng: np.random.Generator
    decisions: int = 0
    last_obs: bytes = b""
    last_actions: I64 | None = None
    kind: tuple[int, int] = (0, 0)
    done: bool = False


class FakeGame:
    """A fake SpireHost engine at the payload level: ``handle(command) -> [payloads]``.

    Deterministic: an explicit RESET seed fixes the episode plan and, together with the
    actions taken, every frame byte (HOST_US included). Unseeded RESETs draw from ``seed``.
    """

    def __init__(self, vocab: Vocab | None = None, seed: int = 0, config: FakeHostConfig = FakeHostConfig(),
                 catalog_hash: int = 0) -> None:
        self.vocab = vocab or Vocab(make_synthetic_vocab())
        self.gen = FrameGen(self.vocab, catalog_hash)
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.episode_id = 0
        self.quit = False
        self.ep: _Episode | None = None
        self._pool: dict[tuple[object, ...], list[tuple[bytes, I64]]] = {}

    def handle(self, cmd: Command) -> list[bytes]:
        if isinstance(cmd, QuitCmd):
            self.quit = True
            return []
        if isinstance(cmd, ResetCmd):
            return [self._reset(cmd)]
        return self._step(cmd)

    def _error(self, code: int, resend: bool) -> list[bytes]:
        ep = self.ep
        if ep is None:
            return [error_frame(code, vocab_hash=self.vocab.hash, catalog_hash=self.gen.catalog_hash)]
        out = [self.gen.error(ep.ctx, code)]
        return out + [ep.last_obs] if resend else out

    def _reset(self, cmd: ResetCmd) -> bytes:
        seed = int(self.rng.integers(0, 1 << 63)) if cmd.seed is None else cmd.seed
        self.episode_id += 1
        cfg = self.cfg
        plan = np.random.default_rng((seed, 0))
        length = int(np.clip(round(plan.lognormal(math.log(cfg.mean_decisions), cfg.length_sigma)),
                             cfg.min_decisions, cfg.max_decisions))
        result = Result.WIN if plan.random() < cfg.win_prob or cfg.learnable else Result.LOSS
        reason = TerminalReason.WIN if result == Result.WIN else TerminalReason.DEATH
        if plan.random() < cfg.abort_prob:
            length, result, reason = int(plan.integers(1, length + 1)), Result.ABORTED, TerminalReason.ABORTED
        elif cfg.turn_cap_prob and plan.random() < cfg.turn_cap_prob:
            result, reason = Result.LOSS, TerminalReason.COMBAT_TURN_CAP
        cap = cmd.max_decisions or cfg.decision_cap
        if cap and length > cap and result != Result.ABORTED:
            length, result, reason = cap, Result.LOSS, TerminalReason.DECISION_CAP
        gen = self.gen
        max_hp = [int(x) for x in plan.integers(65, 90, cmd.n_seats)]
        ctx = RunCtx(characters=cmd.characters, ascension=cmd.ascension, view_mode=cmd.view_mode,
                     view_seat=cmd.view_seat, reset_flags=cmd.flags, episode_id=self.episode_id,
                     act_id=gen.rid(plan, "act"), max_hp=max_hp, hp=list(max_hp),
                     gold=[99] * cmd.n_seats)
        ctx.map = gen.make_map(plan, cfg.size, cmd.ascension)
        # The real H5 server only accepts TEAM. Keep the fake's existing SEAT-view
        # protocol-testing mode on its historical schedule.
        schedule = _schedule_h5 if cmd.n_seats > 1 and cmd.view_mode == ViewMode.TEAM else _schedule
        ep = _Episode(seed, ctx, length, int(result), int(reason), schedule(plan, ctx, gen, cfg.size), plan)
        self.ep = ep
        ep.kind = next(ep.sched)
        return self._obs(-1)

    def _obs(self, action: int) -> bytes:
        ep = self.ep
        assert ep is not None
        cfg = self.cfg
        kind, sub = ep.kind
        if cfg.frame_pool:
            r = ep.rng
            auto = int(r.integers(1, cfg.max_auto_steps + 1)) if r.random() < cfg.auto_step_prob else 0
            context = (ep.ctx.acting_seat, tuple(sorted(ep.ctx.ready_seats)), tuple(sorted(ep.ctx.room_done)),
                       tuple(sorted(ep.ctx.rest_done)),
                       tuple(sorted(ep.ctx.treasure_votes.items())), ep.ctx.shared_vote)
            pool = self._pooled((kind, sub, *context), lambda rng: self.gen.frame(
                rng, kind, ep.ctx, kind_sub=sub, size=cfg.size, hashes=cfg.hashes))
            tpl, ep.last_actions = pool[int(r.integers(len(pool)))]
            payload = self._patch(tpl, ep.ctx, auto)
        else:
            rng = np.random.default_rng((ep.seed, 1, ep.ctx.step_id, action + 1))
            auto = int(rng.integers(1, cfg.max_auto_steps + 1)) if rng.random() < cfg.auto_step_prob else 0
            payload = self.gen.frame(rng, kind, ep.ctx, kind_sub=sub, size=cfg.size, auto_steps=auto,
                                     hashes=cfg.hashes)
            ep.last_actions = np.array(parse_frame(payload).actions, dtype=np.int64)
        ep.last_obs = payload
        return payload

    def _terminal(self, ep: _Episode) -> bytes:
        cfg = self.cfg
        abort = FAKE_ABORT_REASON if ep.result == Result.ABORTED else 0
        if not cfg.frame_pool:
            return self.gen.terminal(np.random.default_rng((ep.seed, 2)), ep.ctx, ep.result, hashes=cfg.hashes,
                                     reason=ep.reason, abort_reason=abort)
        pool = self._pooled(("terminal", ep.result, ep.reason),
                            lambda rng: self.gen.terminal(rng, ep.ctx, ep.result, hashes=cfg.hashes,
                                                          reason=ep.reason, abort_reason=abort))
        buf = bytearray(self._patch(pool[0][0], ep.ctx, 0))
        _I32.pack_into(buf, GLOBAL_OFFSET + G.POLICY_DECISIONS * WORD, ep.ctx.step_id)  # = STEP_ID (P-5)
        return bytes(buf)

    def _pooled(self, what: tuple[object, ...], make: Callable[[np.random.Generator], bytes]
                ) -> list[tuple[bytes, I64]]:
        ep = self.ep
        assert ep is not None
        c = ep.ctx
        key = (c.characters, c.ascension, c.view_mode, c.view_seat, c.reset_flags, *what)
        pool = self._pool.get(key)
        if pool is None:
            rng = np.random.default_rng((self.seed, 3, len(self._pool)))
            pool = []
            for _ in range(1 if what[0] == "terminal" else self.cfg.frame_pool):
                payload = make(rng)
                pool.append((payload, np.array(parse_frame(payload).actions, dtype=np.int64)))
            self._pool[key] = pool
        return pool

    @staticmethod
    def _patch(tpl: bytes, ctx: RunCtx, auto: int) -> bytes:
        buf = bytearray(tpl)
        _I32.pack_into(buf, H.EPISODE_ID * WORD, ctx.episode_id)
        _I32.pack_into(buf, H.STEP_ID * WORD, ctx.step_id)
        _I32.pack_into(buf, H.AUTO_STEPS * WORD, auto)
        return bytes(buf)

    def _step(self, cmd: StepCmd) -> list[bytes]:
        ep = self.ep
        if ep is None or ep.done:
            return self._error(ErrorCode.BAD_MESSAGE, resend=False)
        if cmd.step_id != ep.ctx.step_id:
            return self._error(ErrorCode.STALE_STEP, resend=True)
        assert ep.last_actions is not None
        if not 0 <= cmd.action_index < len(ep.last_actions):
            return self._error(ErrorCode.BAD_INDEX, resend=True)
        if ep.rng.random() < self.cfg.error_prob:
            return self._error(ErrorCode.BAD_MESSAGE, resend=True)
        acts = ep.last_actions
        act = acts[cmd.action_index]
        verb = int(act[ActionCol.VERB])
        ctx = ep.ctx
        cfg = self.cfg
        if ctx.acting_seat is not None and verb == Verb.END_TURN:
            ctx.ready_seats.add(int(act[ActionCol.SEAT]))
        if ctx.acting_seat is not None and ep.kind[0] == Kind.REST and verb == Verb.CHOOSE_OPTION:
            ctx.rest_done.add(int(act[ActionCol.SEAT]))
        if ctx.acting_seat is not None and ep.kind == (Kind.TREASURE, TreasurePhase.RELIC):
            if verb == Verb.CHOOSE_OPTION:
                assert ep.last_obs
                options = parse_frame(ep.last_obs).table(Table.OPTIONS)
                ctx.treasure_votes[int(act[ActionCol.SEAT])] = int(options[int(act[ActionCol.SRC_ROW]), OptionCol.POS])
        if verb == Verb.TRAVEL:
            if cfg.frame_pool:  # pooled MAP frames do not show this episode's map
                assert ctx.map is not None
                ctx.path.append(int(ep.rng.choice(self.gen.reachable(ctx) or [ctx.map.start])))
            else:
                ctx.path.append(int(act[ActionCol.SRC_ROW]))
            ctx.total_floor += 1
        ep.decisions += 1
        ctx.step_id += 1
        if (cfg.learnable and ep.result != Result.ABORTED and verb == Verb.END_TURN
                and bool((acts[:, ActionCol.VERB] == Verb.PLAY_CARD).any())):
            ctx.hp = [h - cfg.end_turn_penalty for h in ctx.hp]
            if max(ctx.hp) <= 0:
                ep.decisions, ep.result, ep.reason = ep.length, Result.LOSS, TerminalReason.DEATH
        if ep.decisions >= ep.length:
            ep.done = True
            if ep.reason == TerminalReason.DEATH:
                ctx.hp = [0] * ctx.n_seats
            elif ep.result == Result.WIN:  # the final boss fight was won
                _won(ctx, "Boss")
                ctx.hp = [0] * ctx.n_seats  # real host WIN TERMINAL reads HP in the victory room
            return [self._terminal(ep)]
        ep.kind = ep.sched.send(verb)
        return [self._obs(cmd.action_index)]


def _won(ctx: RunCtx, room: str) -> None:
    """Public win history after a won combat in a ``room`` room (GLOBAL 44..48)."""
    ctx.combats_won += 1
    ctx.elites_won += room == "Elite"
    ctx.bosses_won += room == "Boss"
    ctx.last_won_room, ctx.last_won_floor = room, ctx.total_floor


def _fight(rng: np.random.Generator, ctx: RunCtx, gen: FrameGen, room: str
           ) -> Generator[tuple[int, int], int, None]:
    """One combat in a ``room`` room ("Monster" / "Elite" / "Boss"), won when it returns. The win
    counters change on the first frame after the fight, as on the real host."""
    rounds = int(rng.integers(2, 5)) + 2 * (room != "Monster")
    ctx.enemies = tuple(int(m) for m in gen.rids(rng, "monster", int(rng.integers(1, 4))))
    ctx.room = room
    for rnd in range(1, rounds + 1):
        ctx.combat_round = rnd
        while True:
            if rng.random() < 0.08:
                yield Kind.CARD_SELECT, int(rng.choice([int(s) for s in COMBAT_SELECT_SUBS]))
            if (yield (Kind.COMBAT, 0)) == Verb.END_TURN:
                break
        ctx.hp = [max(1, h - int(rng.integers(0, 8))) for h in ctx.hp]
    ctx.enemies, ctx.combat_round = (), 0
    _won(ctx, room)


def _schedule(rng: np.random.Generator, ctx: RunCtx, gen: FrameGen, size: SizeMode
              ) -> Generator[tuple[int, int], int, None]:
    """Decision-kind sequence of a fake run; receives the verb chosen at each decision."""
    point_names = {v: k for k, v in gen.point.items()}
    while True:
        assert ctx.map is not None
        while (yield (Kind.MAP, 0)) != Verb.TRAVEL:
            pass
        node = ctx.path[-1]
        pt = point_names.get(int(ctx.map.nodes[node, MapNodeCol.POINT_TYPE]), "Monster")
        if pt == "Unknown":
            pt = str(rng.choice(["Monster", "Event", "Event", "Shop", "Treasure"]))
        if pt in ("Monster", "Elite", "Boss"):
            yield from _fight(rng, ctx, gen, pt)
            ctx.gold = [g + int(rng.integers(10, 40)) for g in ctx.gold]
            if rng.random() < 0.8:
                yield Kind.CARD_REWARD, 0
            for _ in range(4):
                if (yield (Kind.REWARD, 0)) == Verb.PROCEED:
                    break
            if pt == "Boss":
                ctx.act_index = min(2, ctx.act_index + 1)
                ctx.act_id = gen.rid(rng, "act")
                ctx.map = gen.make_map(rng, size, ctx.ascension)
                ctx.path = []
        elif pt in ("Event", "Ancient"):
            for _ in range(int(rng.integers(1, 4))):
                yield Kind.EVENT, 0
            if pt == "Event" and rng.random() < 0.1:  # an event-started fight: a Monster room fight (E-22)
                yield from _fight(rng, ctx, gen, "Monster")
                yield Kind.EVENT, 0
            x = rng.random()  # follow-up screens, over-represented relative to the game for coverage
            if x < 0.1:
                yield Kind.CONFIRM, 0
            elif x < 0.25:
                for _ in range(int(rng.integers(2, 6))):
                    yield Kind.CRYSTAL, 0
            elif x < 0.37:
                yield Kind.BUNDLE, 0
            elif x < 0.5:
                yield Kind.CARD_SELECT, int(rng.choice([int(s) for s in _DECK_SUBS] + [1, 2]))
        elif pt == "Shop":
            for _ in range(5):
                v = yield Kind.SHOP, 0
                if v == Verb.PROCEED:
                    break
                if v == Verb.CHOOSE_OPTION and rng.random() < 0.15:
                    yield Kind.CARD_SELECT, int(CardSelectSub.DECK_REMOVAL)
        elif pt == "RestSite":
            for _ in range(3):
                v = yield Kind.REST, 0
                if v == Verb.PROCEED:
                    break
                if v == Verb.CHOOSE_OPTION and rng.random() < 0.5:
                    yield Kind.CARD_SELECT, int(CardSelectSub.DECK_UPGRADE)
        else:  # Treasure
            if (yield (Kind.TREASURE, int(TreasurePhase.CHEST))) == Verb.OPEN_CHEST:
                yield Kind.TREASURE, int(TreasurePhase.RELIC)


def _fight_h5(rng: np.random.Generator, ctx: RunCtx, gen: FrameGen, room: str
              ) -> Generator[tuple[int, int], int, None]:
    """Union combat; each seat ends once, while a prompt owns exactly one seat."""
    ctx.enemies = tuple(int(m) for m in gen.rids(rng, "monster", int(rng.integers(1, 4))))
    ctx.room = room
    for rnd in range(1, int(rng.integers(2, 5)) + 2 * (room != "Monster") + 1):
        ctx.combat_round = rnd
        ctx.ready_seats.clear()
        while len(ctx.ready_seats) < ctx.n_seats:
            if rng.random() < 0.08:
                ctx.acting_seat = int(rng.choice([s for s in range(ctx.n_seats) if s not in ctx.ready_seats]))
                yield Kind.CARD_SELECT, int(rng.choice([int(s) for s in COMBAT_SELECT_SUBS]))
            ctx.acting_seat = -1
            yield Kind.COMBAT, 0
        ctx.hp = [max(1, h - int(rng.integers(0, 8))) for h in ctx.hp]
    ctx.enemies, ctx.combat_round = (), 0
    ctx.ready_seats.clear()
    _won(ctx, room)


def _schedule_h5(rng: np.random.Generator, ctx: RunCtx, gen: FrameGen, size: SizeMode
                 ) -> Generator[tuple[int, int], int, None]:
    """H5 ordering contract, not game simulation: seat-major chains then one team vote.

    Treasure allocation remains per seat; combat offers a union, with single-seat prompts.
    The historical single-seat schedule is kept separate to preserve its RNG stream.
    """
    point_names = {v: k for k, v in gen.point.items()}
    while True:
        assert ctx.map is not None
        ctx.acting_seat = 0
        while (yield (Kind.MAP, 0)) != Verb.TRAVEL:
            pass
        ctx.room_done.clear()
        ctx.rest_done.clear()
        ctx.treasure_votes.clear()
        pt = point_names[int(ctx.map.nodes[ctx.path[-1], MapNodeCol.POINT_TYPE])]
        if pt == "Unknown":
            pt = str(rng.choice(["Monster", "Event", "Shop", "Treasure"]))
        combat = pt in ("Monster", "Elite", "Boss")
        if combat:
            yield from _fight_h5(rng, ctx, gen, pt)
            ctx.gold = [g + int(rng.integers(10, 40)) for g in ctx.gold]
        shared = pt == "Event" and rng.random() < 0.3
        for seat in range(ctx.n_seats):
            ctx.acting_seat = seat
            if combat:
                while True:
                    verb = yield Kind.REWARD, 0
                    if verb == Verb.PROCEED:
                        break
                    if verb == Verb.CHOOSE_OPTION:
                        yield Kind.CARD_REWARD, 0
            elif pt in ("Event", "Ancient") and not shared:
                for _ in range(int(rng.integers(1, 3))):
                    while (yield (Kind.EVENT, 0)) != Verb.CHOOSE_OPTION:
                        pass
                if rng.random() < 0.3:
                    yield Kind.CARD_SELECT, int(CardSelectSub.DECK_REMOVAL)
            elif pt == "Shop":
                while (yield (Kind.SHOP, 0)) != Verb.PROCEED:
                    pass
            elif pt == "RestSite":
                while True:
                    verb = yield Kind.REST, 0
                    if verb == Verb.PROCEED:
                        break
                    if verb == Verb.CHOOSE_OPTION:
                        yield Kind.CARD_SELECT, int(CardSelectSub.DECK_UPGRADE)
            elif pt == "Treasure":
                while (yield (Kind.TREASURE, int(TreasurePhase.CHEST))) != Verb.OPEN_CHEST:
                    pass
                while (yield (Kind.TREASURE, int(TreasurePhase.RELIC))) != Verb.CHOOSE_OPTION:
                    pass
            ctx.room_done.add(seat)
        if shared:
            ctx.acting_seat, ctx.shared_vote = 0, True
            while (yield (Kind.EVENT, 0)) != Verb.CHOOSE_OPTION:
                pass
            ctx.shared_vote = False
        if pt == "Boss":
            ctx.act_index = min(2, ctx.act_index + 1)
            ctx.act_id = gen.rid(rng, "act")
            ctx.map = gen.make_map(rng, size, ctx.ascension)
            ctx.path = []


class FakeHost(Host):
    """In-process :class:`~spire.host.Host` backed by :class:`FakeGame`.

    Messages go through the real encoders/decoders, so the full protocol path is exercised.
    """

    def __init__(self, vocab: Vocab | None = None, seed: int = 0, config: FakeHostConfig = FakeHostConfig(), *,
                 catalog_hash: int = 0, verify_hashes: bool = False,
                 expected_hashes: tuple[int, int] | None = None, training_hidden: bool = False) -> None:
        super().__init__(reset_timeout=0.0, step_timeout=0.0, verify_hashes=verify_hashes,
                         expected_hashes=expected_hashes, training_hidden=training_hidden)
        self.game = FakeGame(vocab, seed, config, catalog_hash)
        self._out: deque[bytes] = deque()
        self._closed = False

    def _send(self, message: bytes) -> None:
        if self._closed:
            raise HostStateError("FakeHost is closed")
        if len(message) < 4 or int.from_bytes(message[:4], "little") != len(message) - 4:
            raise ProtocolError("bad length prefix")
        replies = self.game.handle(decode_command(message[4:]))
        sink = self._sink
        if sink is None:
            self._out.extend(replies)
        else:
            for r in replies:
                sink(self.generation, r)

    def _recv(self, timeout: float) -> bytes:
        if not self._out:
            raise HostTimeout("FakeHost: no reply pending")
        return self._out.popleft()

    def close(self) -> None:
        if not self._closed:
            self.game.handle(QuitCmd())
            self._closed = True
