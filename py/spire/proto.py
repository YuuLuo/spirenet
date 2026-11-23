"""SpireHost wire protocol: message framing, zero-copy frame parsing, control messages.

Wire format (schema 3.1): every message in both directions is ``[u32 LE length][payload]``.
Host -> Python payloads are frames (OBS / TERMINAL / ERROR); Python -> host payloads are
RESET / STEP / QUIT commands (schema 7.1).

``parse_frame`` returns a :class:`Frame` whose header, global vector, directory and 16
tables are read-only ``np.int32`` views into the received buffer (no copies). A frame of an
episode reset with ``ResetFlag.FORECAST`` carries N_TABLES = 17: the extension table FORECAST
after the v1 data (``Frame.forecast``; notes/forecast/contract.md 2.2); with ``ResetFlag.CALC`` three more
trailer tables TURN / FOE / ROUTE follow (``Frame.turn`` / ``foe`` / ``route``; N_TABLES 16 + [forecast] + 3 * [calc];
notes/calc/contract.md 2.2); with RESET bit 10 (training only) the HDRAW table closes the chain
(``Frame.hdraw``; N_TABLES 16 + [forecast] + 3 * [calc] + [hidden]; notes/critic/contract.md 1.3). The trailer
kinds are read from the chain itself (table ids), since the header count alone is ambiguous (17 = forecast or
hidden, 20 = forecast + calc or calc + hidden).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from itertools import accumulate
from operator import le, mul
from typing import IO, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from .schema import (
    ACTION_SORT_COLS,
    CARD_KEY_COLS,
    CMD_MAGIC,
    DATA_OFFSET,
    DIR_OFFSET,
    DIR_STRIDE,
    DYNAMIC_POINTERS,
    FNV_OFFSET_BASIS,
    FNV_PRIME,
    FORECAST_MAX_ROWS,
    FORECAST_N_COLS,
    FORECAST_TABLE,
    FRAME_MAGIC,
    G_ERROR_CODE,
    GLOBAL_LEN,
    GLOBAL_OFFSET,
    HEADER_LEN,
    MAX_FRAME_BYTES_FORECAST,
    MAX_ROWS,
    MULTISET_ZONES,
    N_COLS,
    N_TABLES,
    N_TABLES_FORECAST,
    POINTERS,
    RESET_FLAGS_DEFAULT,
    SCHEMA_VERSION,
    SLICES,
    VAR_KEY_COLS,
    WORD,
    CALC_MAX_ROWS,
    CALC_N_COLS,
    CALC_TABLES,
    MAX_FRAME_BYTES_CALC,
    HDRAW_MAX_ROWS,
    HDRAW_N_COLS,
    HDRAW_TABLE,
    MAX_FRAME_BYTES_HIDDEN,
    HDrawCol,
    ActionCol,
    CardCol,
    CellCol,
    CmdType,
    FrameFlag,
    G,
    H,
    MapEdgeCol,
    MapNodeCol,
    MapNodeFlag,
    MsgType,
    Result,
    SchemaError,
    SeatCol,
    Table,
    VarCol,
    Verb,
    ViewMode,
    Zone,
    validate_directory,
)

I32 = NDArray[np.int32]

_LEN = struct.Struct("<I")
_DIR_WORD0 = DIR_OFFSET // WORD
_DIR_WORD1 = _DIR_WORD0 + N_TABLES * DIR_STRIDE
_GLOBAL_WORD0 = GLOBAL_OFFSET // WORD
_U32 = 0xFFFFFFFF
# plain ints: ``int & IntFlag`` dispatches to IntFlag.__rand__ (enum machinery, ~3 us per test)
_F_TERMINAL, _F_OVERFLOW = int(FrameFlag.TERMINAL), int(FrameFlag.OVERFLOW)
_TABLE_IDS = tuple(range(N_TABLES))
_N_COLS_T = tuple(N_COLS)
_MAX_ROWS_T = tuple(int(m) for m in MAX_ROWS)
_ROW_BYTES = tuple(c * WORD for c in N_COLS)
_ACTIONS_T = int(Table.ACTIONS)
# header + directory words in one call (the GLOBAL vector in between is skipped, unpacked lazily)
_PREFIX = struct.Struct(f"<{HEADER_LEN}i{GLOBAL_LEN * WORD}x{N_TABLES * DIR_STRIDE}i")
_D0, _D1 = HEADER_LEN, HEADER_LEN + N_TABLES * DIR_STRIDE  # directory inside the unpacked tuple
_GLOBAL_S = struct.Struct(f"<{GLOBAL_LEN}i")
_FC_DIR = struct.Struct(f"<{DIR_STRIDE}i")  # the FORECAST directory entry at X0 (end of the v1 data)
_FC_ROW_BYTES = FORECAST_N_COLS * WORD
_CC_ROW_BYTES = tuple(c * WORD for c in CALC_N_COLS)
_HD_ROW_BYTES = HDRAW_N_COLS * WORD
# header N_TABLES -> number of trailer entries (16 + [forecast] + 3 * [calc] + [hidden]: 0..5)
_N_TAB_KIND = {N_TABLES + k: k for k in range(6)}
(_H_MAGIC, _H_VERSION, _H_N_TABLES, _H_GLOBAL_LEN, _H_FRAME_BYTES, _H_MSG_TYPE, _H_FLAGS, _H_RESULT, _H_N_ACTIONS,
 _H_EPISODE_ID, _H_STEP_ID, _H_AUTO_STEPS, _H_VOCAB_HASH, _H_CATALOG_HASH) = (
    int(h) for h in (H.MAGIC, H.SCHEMA_VERSION, H.N_TABLES, H.GLOBAL_LEN, H.FRAME_BYTES, H.MSG_TYPE, H.FLAGS, H.RESULT,
                     H.N_ACTIONS, H.EPISODE_ID, H.STEP_ID, H.AUTO_STEPS, H.VOCAB_HASH, H.CATALOG_HASH))
_MSG_OBS, _MSG_TERMINAL, _MSG_ERROR = int(MsgType.OBS), int(MsgType.TERMINAL), int(MsgType.ERROR)
_RES_IN_PROGRESS, _RES_WIN, _RES_ABORTED = int(Result.IN_PROGRESS), int(Result.WIN), int(Result.ABORTED)
_G_KIND, _G_KIND_SUB, _G_N_SEATS = int(G.KIND), int(G.KIND_SUB), int(G.N_SEATS)


class ProtocolError(RuntimeError):
    """The byte stream violates message framing (bad length, EOF inside a message ...)."""


class HashMismatch(SchemaError):
    """A table's FNV-1a content hash does not match the directory."""


# --------------------------------------------------------------------------- hashing


def fnv1a32(data: bytes | bytearray | memoryview) -> int:
    """32-bit FNV-1a (offset basis 2166136261, prime 16777619) as an unsigned int."""
    h = FNV_OFFSET_BASIS
    for b in bytes(data):
        h = ((h ^ b) * FNV_PRIME) & _U32
    return h


# --------------------------------------------------------------------------- frames


class Frame:
    """A parsed host frame. All arrays are read-only int32 views into ``payload``.

    ``fc_rows`` / ``fc_off``: FORECAST row count and first data word (both -1 when the frame has no
    FORECAST trailer); ``forecast`` is the (n, 17) view or None, ``fc_hash`` its directory hash.
    ``cc_rows`` / ``cc_off`` / ``cc_hash``: the same for the calc trailers TURN / FOE / ROUTE (3-tuples, or None
    when the frame has no calc trailer); views ``turn`` / ``foe`` / ``route``.
    ``hd_rows`` / ``hd_off`` / ``hd_hash``: the HDRAW trailer (-1 without it; training only, notes/critic/contract.md
    1.3); ``hdraw`` its (n, 3) view or None; ``hd_entry`` = XH, the word offset of its directory entry (-1 without).

    ``tables[t]`` is an ``(n_rows, N_COLS[t])`` array, or ``None`` for a table elided
    under ``ResetFlag.ELIDE_UNCHANGED`` until :meth:`resolve_elided` fills it in. The numpy views
    (``words``, ``glob``, ``header``, ``directory``, ``tables``) are built on first access: the
    actor's hot path only needs the scalars unpacked at parse time - ``gvals`` (the GLOBAL vector
    as a tuple), ``rows`` / ``offsets`` (row count and first word of each table, ``-1`` rows for an
    elided table) - and the batch tokenizer reads the tables straight from ``payload``.
    """

    __slots__ = (
        "payload", "_words", "_glob", "_gvals", "rows", "offsets", "_tables",
        "msg_type", "episode_id", "step_id", "flags", "result", "n_actions", "auto_steps",
        "vocab_hash", "catalog_hash", "fc_rows", "fc_off", "fc_hash", "cc_rows", "cc_off", "cc_hash",
        "hd_rows", "hd_off", "hd_hash", "hd_entry",
    )

    payload: bytes | bytearray | memoryview
    _words: I32 | None
    _glob: I32 | None
    _gvals: tuple[int, ...] | None
    rows: tuple[int, ...]  # directory n_rows per table (-1: elided in this payload)
    offsets: tuple[int, ...]  # directory byte_offset // 4 per table
    _tables: list[I32 | None] | None
    msg_type: int
    episode_id: int
    step_id: int
    flags: int
    result: int
    n_actions: int
    auto_steps: int
    vocab_hash: int
    catalog_hash: int
    fc_rows: int
    fc_off: int
    fc_hash: int
    cc_rows: tuple[int, int, int] | None
    cc_off: tuple[int, int, int] | None
    cc_hash: tuple[int, int, int] | None
    hd_rows: int
    hd_off: int
    hd_hash: int
    hd_entry: int

    @property
    def words(self) -> I32:
        w = self._words
        if w is None:
            w = np.frombuffer(self.payload, dtype="<i4")
            if w.flags.writeable:
                w.flags.writeable = False
            self._words = w
        return w

    @property
    def gvals(self) -> tuple[int, ...]:
        """The GLOBAL vector as a tuple of ints."""
        g = self._gvals
        if g is None:
            g = self._gvals = _GLOBAL_S.unpack_from(self.payload, GLOBAL_OFFSET)
        return g

    @property
    def glob(self) -> I32:
        g = self._glob
        if g is None:
            g = self._glob = self.words[_GLOBAL_WORD0 : _GLOBAL_WORD0 + GLOBAL_LEN]
        return g

    @property
    def header(self) -> I32:
        return self.words[:HEADER_LEN]

    @property
    def directory(self) -> I32:
        return self.words[_DIR_WORD0:_DIR_WORD1].reshape(N_TABLES, DIR_STRIDE)

    @property
    def tables(self) -> list[I32 | None]:
        t = self._tables
        if t is None:
            w, rows, offs = self.words, self.rows, self.offsets
            t = self._tables = [None if n < 0 else w[o : o + n * c].reshape(n, c)
                                for n, o, c in zip(rows, offs, N_COLS)]
        return t

    @tables.setter
    def tables(self, value: list[I32 | None]) -> None:
        self._tables = value

    @property
    def forecast(self) -> I32 | None:
        """The FORECAST table (n, 17) of a frame with the trailer, else None."""
        if self.fc_rows < 0:
            return None
        o = self.fc_off
        return self.words[o : o + self.fc_rows * FORECAST_N_COLS].reshape(self.fc_rows, FORECAST_N_COLS)

    def calc_table(self, k: int) -> I32 | None:
        """Calc trailer ``k`` (0 TURN, 1 FOE, 2 ROUTE) as an (n, cols) view, or None without the calc trailers."""
        if self.cc_rows is None:
            return None
        assert self.cc_off is not None
        n, o, c = self.cc_rows[k], self.cc_off[k], CALC_N_COLS[k]
        return self.words[o : o + n * c].reshape(n, c)

    @property
    def turn(self) -> I32 | None:
        return self.calc_table(0)

    @property
    def foe(self) -> I32 | None:
        return self.calc_table(1)

    @property
    def route(self) -> I32 | None:
        return self.calc_table(2)

    @property
    def hdraw(self) -> I32 | None:
        """The HDRAW table (n, 3) = (SEAT, POS, OWNER_ROW) of a frame with the hidden trailer, else None."""
        if self.hd_rows < 0:
            return None
        o = self.hd_off
        return self.words[o : o + self.hd_rows * HDRAW_N_COLS].reshape(self.hd_rows, HDRAW_N_COLS)

    @property
    def kind(self) -> int:
        return self.gvals[_G_KIND]

    @property
    def kind_sub(self) -> int:
        return self.gvals[_G_KIND_SUB]

    @property
    def n_seats(self) -> int:
        return self.gvals[_G_N_SEATS]

    @property
    def is_obs(self) -> bool:
        return self.msg_type == _MSG_OBS

    @property
    def is_terminal(self) -> bool:
        return self.msg_type == _MSG_TERMINAL

    @property
    def is_error(self) -> bool:
        return self.msg_type == _MSG_ERROR

    @property
    def error_code(self) -> int:
        """Error code of an ERROR frame (global[0], schema 7.3)."""
        if not self.is_error:
            raise ValueError("not an ERROR frame")
        return self.gvals[G_ERROR_CODE]

    @property
    def overflow(self) -> bool:
        return bool(self.flags & _F_OVERFLOW)

    @property
    def elided(self) -> tuple[int, ...]:
        return tuple(t for t, a in enumerate(self.tables) if a is None)

    def table(self, t: int) -> I32:
        a = self.tables[t]
        if a is None:
            raise ValueError(f"table {Table(t).name} is elided; call resolve_elided(prev) first")
        return a

    @property
    def actions(self) -> I32:
        return self.table(Table.ACTIONS)

    def table_hash(self, t: int) -> int:
        """Directory content hash of table ``t`` (unsigned)."""
        return int(self.directory[t, 4]) & _U32

    def table_bytes(self, t: int) -> memoryview:
        """Raw bytes of table ``t`` inside the payload (zero-copy)."""
        off = int(self.directory[t, 3])
        n = int(self.directory[t, 1]) * N_COLS[t] * WORD
        return memoryview(self.payload)[off : off + n]

    def resolve_elided(self, prev: "Frame") -> None:
        """Replace elided tables with ``prev``'s tables after checking the hashes agree."""
        for t in self.elided:
            src = prev.tables[t]
            if src is None:
                raise SchemaError(f"table {Table(t).name} elided in both frames")
            if prev.table_hash(t) != self.table_hash(t):
                raise HashMismatch(f"elided table {Table(t).name}: hash differs from previous frame")
            self.tables[t] = src

    def __repr__(self) -> str:
        rows = ",".join("-" if a is None else str(a.shape[0]) for a in self.tables)
        return (f"Frame(msg={self.msg_type} kind={self.gvals[0]} ep={self.episode_id} "
                f"step={self.step_id} result={self.result} rows=[{rows}])")


def parse_frame(payload: bytes | bytearray | memoryview, *, verify_hashes: bool = False) -> Frame:
    """Parse one frame payload (without the length prefix) into a zero-copy :class:`Frame`.

    Always validates magic, schema version, header sizes, message type/flags/result
    consistency and the table directory (``schema.validate_directory``). With
    ``verify_hashes`` every table's FNV-1a content hash is recomputed (pure Python,
    roughly 0.1 us per byte; meant for sampled checks and tests).
    """
    n = payload.nbytes if type(payload) is memoryview else len(payload)
    if n < (_DIR_WORD1 * WORD) or n % WORD:
        raise SchemaError(f"frame of {n} bytes is too short or not 4-byte aligned")
    t = _PREFIX.unpack_from(payload, 0)  # header | directory, one C call
    if t[_H_MAGIC] != FRAME_MAGIC:
        raise SchemaError(f"bad frame magic 0x{t[_H_MAGIC] & _U32:08X}")
    if t[_H_VERSION] != SCHEMA_VERSION:
        raise SchemaError(f"schema version {t[_H_VERSION]} != {SCHEMA_VERSION}")
    n_tab = t[_H_N_TABLES]
    kind = _N_TAB_KIND.get(n_tab)
    if kind is None or t[_H_GLOBAL_LEN] != GLOBAL_LEN:
        raise SchemaError(f"header n_tables={n_tab} global_len={t[_H_GLOBAL_LEN]}")
    if t[_H_FRAME_BYTES] != n:
        raise SchemaError(f"header frame_bytes {t[_H_FRAME_BYTES]} != payload length {n}")
    msg = t[_H_MSG_TYPE]
    flags = t[_H_FLAGS]
    result = t[_H_RESULT]
    if msg == _MSG_OBS:
        if flags & _F_TERMINAL or result != _RES_IN_PROGRESS:
            raise SchemaError("OBS frame with terminal flag or a result")
    elif msg == _MSG_TERMINAL:
        if not flags & _F_TERMINAL or not _RES_WIN <= result <= _RES_ABORTED:
            raise SchemaError(f"TERMINAL frame with flags=0x{flags:x} result={result}")
    elif msg != _MSG_ERROR:
        raise SchemaError(f"unknown msg_type {msg}")

    rows = t[_D0 + 1 : _D1 : DIR_STRIDE]
    offs = t[_D0 + 3 : _D1 : DIR_STRIDE]
    end = n  # end of the v1 data: the frame end, or X0 before the trailer chain
    fc_rows = fc_off = fc_hash = -1
    cc_rows = cc_off = cc_hash = None
    hd_rows = hd_off = hd_hash = hd_entry = -1
    if n_tab != N_TABLES:
        end = DATA_OFFSET + sum([r * b for r, b in zip(rows, _ROW_BYTES) if r > 0])  # X0 (elided tables count 0)
        x = end
        if n < x + _FC_DIR.size:
            raise SchemaError(f"trailer directory entry at {x} beyond the frame ({n} bytes)")
        has_fc = _FC_DIR.unpack_from(payload, x)[0] == FORECAST_TABLE  # the chain names its first table
        rest = kind - has_fc
        if rest not in (0, 1, 3, 4):
            raise SchemaError(f"header n_tables={n_tab}: no trailer chain with {kind} entries (forecast={has_fc})")
        if has_fc:
            tid, fc_rows, fc_cols, off, fc_hash = _FC_DIR.unpack_from(payload, x)
            if (tid != FORECAST_TABLE or fc_cols != FORECAST_N_COLS or not 0 <= fc_rows <= FORECAST_MAX_ROWS
                    or off != x + _FC_DIR.size or n < off + fc_rows * _FC_ROW_BYTES):
                raise SchemaError(f"FORECAST entry ({tid}, {fc_rows}, {fc_cols}, {off}) at X0={x}, frame {n} bytes")
            fc_off = off >> 2
            x = off + fc_rows * _FC_ROW_BYTES
        if rest >= 3:  # TURN, FOE, ROUTE in table-id order (notes/calc/contract.md 2.2)
            cr, co, ch = [], [], []
            for k, (want, nc, cap) in enumerate(zip(CALC_TABLES, CALC_N_COLS, CALC_MAX_ROWS)):
                if n < x + _FC_DIR.size:
                    raise SchemaError(f"calc directory entry {want} at {x} beyond the frame ({n} bytes)")
                tid, nr, ncols, off, h = _FC_DIR.unpack_from(payload, x)
                if (tid != want or ncols != nc or not 0 <= nr <= cap or off != x + _FC_DIR.size
                        or n < off + nr * _CC_ROW_BYTES[k]):
                    raise SchemaError(f"calc entry ({tid}, {nr}, {ncols}, {off}) at {x}, expected table {want}, "
                                      f"frame {n} bytes")
                cr.append(nr)
                co.append(off >> 2)
                ch.append(h)
                x = off + nr * _CC_ROW_BYTES[k]
            cc_rows, cc_off, cc_hash = (cr[0], cr[1], cr[2]), (co[0], co[1], co[2]), (ch[0], ch[1], ch[2])
        if rest in (1, 4):  # HDRAW closes the chain (notes/critic/contract.md 1.3)
            if n < x + _FC_DIR.size:
                raise SchemaError(f"HDRAW directory entry at {x} beyond the frame ({n} bytes)")
            tid, hd_rows, hd_cols, off, hd_hash = _FC_DIR.unpack_from(payload, x)
            if (tid != HDRAW_TABLE or hd_cols != HDRAW_N_COLS or not 0 <= hd_rows <= HDRAW_MAX_ROWS
                    or off != x + _FC_DIR.size or n < off + hd_rows * _HD_ROW_BYTES):
                raise SchemaError(f"HDRAW entry ({tid}, {hd_rows}, {hd_cols}, {off}) at {x}, frame {n} bytes")
            hd_entry = x >> 2
            hd_off = off >> 2
            x = off + hd_rows * _HD_ROW_BYTES
        if x != n:
            raise SchemaError(f"trailer chain ends at {x}, frame has {n} bytes")
    # Fast path of schema.validate_directory (identical acceptance for frames without elided
    # tables, checked tuple-wise at C speed); anything else goes through the full validator, which
    # raises the precise error or accepts the elided layout.
    if not (t[_D0:_D1:DIR_STRIDE] == _TABLE_IDS
            and t[_D0 + 2 : _D1 : DIR_STRIDE] == _N_COLS_T
            and min(rows) >= 0 and all(map(le, rows, _MAX_ROWS_T))
            and tuple(accumulate(map(mul, rows, _ROW_BYTES), initial=DATA_OFFSET)) == offs + (end,)):
        validate_directory(t[_D0:_D1], end, flags)
    n_act = rows[_ACTIONS_T]
    if n_act >= 0 and t[_H_N_ACTIONS] != n_act:
        raise SchemaError(f"header n_actions {t[_H_N_ACTIONS]} != ACTIONS rows {n_act}")

    f = Frame()
    f.payload = payload
    f._words = f._glob = f._gvals = f._tables = None
    f.rows = rows
    f.offsets = tuple([o >> 2 for o in offs])
    f.msg_type = msg
    f.episode_id = t[_H_EPISODE_ID]
    f.step_id = t[_H_STEP_ID]
    f.flags = flags
    f.result = result
    f.n_actions = t[_H_N_ACTIONS]
    f.auto_steps = t[_H_AUTO_STEPS]
    f.vocab_hash = t[_H_VOCAB_HASH]
    f.catalog_hash = t[_H_CATALOG_HASH]
    f.fc_rows, f.fc_off, f.fc_hash = fc_rows, fc_off, fc_hash
    f.cc_rows, f.cc_off, f.cc_hash = cc_rows, cc_off, cc_hash
    f.hd_rows, f.hd_off, f.hd_hash, f.hd_entry = hd_rows, hd_off, hd_hash, hd_entry
    if verify_hashes:
        verify_table_hashes(f)
    return f


def verify_table_hashes(f: Frame) -> None:
    """Recompute every present table's FNV-1a hash; raise :class:`HashMismatch` on mismatch."""
    for t in range(N_TABLES):
        if f.tables[t] is None:
            continue
        got = fnv1a32(f.table_bytes(t))
        if got != f.table_hash(t):
            raise HashMismatch(f"table {Table(t).name}: fnv1a {got:#010x} != directory {f.table_hash(t):#010x}")
    if f.fc_rows >= 0:
        o = f.fc_off * WORD
        got = fnv1a32(memoryview(f.payload)[o : o + f.fc_rows * _FC_ROW_BYTES])
        if got != f.fc_hash & _U32:
            raise HashMismatch(f"table FORECAST: fnv1a {got:#010x} != directory {f.fc_hash & _U32:#010x}")
    if f.cc_rows is not None:
        assert f.cc_off is not None and f.cc_hash is not None
        for k, tid in enumerate(CALC_TABLES):
            o = f.cc_off[k] * WORD
            got = fnv1a32(memoryview(f.payload)[o : o + f.cc_rows[k] * _CC_ROW_BYTES[k]])
            if got != f.cc_hash[k] & _U32:
                raise HashMismatch(f"calc table {tid}: fnv1a {got:#010x} != directory {f.cc_hash[k] & _U32:#010x}")
    if f.hd_rows >= 0:
        o = f.hd_off * WORD
        got = fnv1a32(memoryview(f.payload)[o : o + f.hd_rows * _HD_ROW_BYTES])
        if got != f.hd_hash & _U32:
            raise HashMismatch(f"table HDRAW: fnv1a {got:#010x} != directory {f.hd_hash & _U32:#010x}")


# --------------------------------------------------------------------------- message framing


def read_message(stream: IO[bytes], max_bytes: int = MAX_FRAME_BYTES_HIDDEN) -> bytearray | None:
    """Read exactly one length-prefixed message from a binary stream.

    Returns the payload, or ``None`` on a clean EOF at a message boundary. Raises
    :class:`ProtocolError` on EOF inside a message or a length above ``max_bytes``.
    Works on unbuffered pipes (short reads are retried).
    """
    head = _read_exact(stream, _LEN.size, eof_ok=True)
    if head is None:
        return None
    (n,) = _LEN.unpack(head)
    if n > max_bytes:
        raise ProtocolError(f"message length {n} exceeds limit {max_bytes}")
    body = _read_exact(stream, n, eof_ok=False)
    assert body is not None
    return body


def _read_exact(stream: IO[bytes], n: int, *, eof_ok: bool) -> bytearray | None:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = stream.readinto(view[got:])  # type: ignore[attr-defined]
        if not k:
            if got == 0 and eof_ok:
                return None
            raise ProtocolError(f"EOF after {got} of {n} bytes")
        got += k
    return buf


def frame_message(payload: bytes | bytearray | memoryview) -> bytes:
    """Prefix ``payload`` with its u32 LE length."""
    return _LEN.pack(len(payload)) + bytes(payload)


def write_message(stream: IO[bytes], message: bytes) -> None:
    """Write an already length-prefixed message completely (handles short writes)."""
    view = memoryview(message)
    while view:
        k = stream.write(view)
        if k is None:  # non-blocking raw stream with nothing written
            continue
        view = view[k:]
    stream.flush()


# --------------------------------------------------------------------------- control messages

# Payload layouts (schema 7.1); seed_lo / seed_hi are packed as uint32 (same bytes as int32).
_RESET = struct.Struct("<11i2I3i")
_STEP = struct.Struct("<4i")
_QUIT = struct.Struct("<2i")
_STEP_MSG = struct.Struct("<I4i")  # length prefix + STEP payload in one pack
_CMD_STEP = int(CmdType.STEP)
_INT32_MIN, _INT32_MAX = -(1 << 31), (1 << 31) - 1


@dataclass(frozen=True)
class ResetCmd:
    """RESET (schema 7.1). ``characters`` are ``character`` vocab ids, one per seat (1-4).

    ``max_decisions`` (word 14, errata-v1 P-2): this episode's policy-decision cap; 0 = the host's
    ``--max-decisions`` default. Reaching it ends the episode with TERMINAL(LOSS, DECISION_CAP).
    """

    characters: tuple[int, ...]
    ascension: int = 0
    seed: int | None = None
    view_mode: int = ViewMode.TEAM
    view_seat: int = -1
    flags: int = int(RESET_FLAGS_DEFAULT)
    max_decisions: int = 0

    def __post_init__(self) -> None:
        if not 1 <= len(self.characters) <= 4:
            raise ValueError(f"need 1-4 seats, got {len(self.characters)}")
        if any(c < 0 for c in self.characters):
            raise ValueError("character vocab ids must be >= 0")
        if self.ascension < 0:
            raise ValueError("ascension must be >= 0")
        if self.seed is not None and not 0 <= self.seed < (1 << 64):
            raise ValueError("seed must be a 64-bit unsigned integer")
        if not 0 <= self.max_decisions <= _INT32_MAX:
            raise ValueError("max_decisions must be in [0, 2**31)")
        if self.view_mode == ViewMode.TEAM:
            if self.view_seat != -1:
                raise ValueError("TEAM view requires view_seat == -1")
        elif self.view_mode == ViewMode.SEAT:
            if not 0 <= self.view_seat < len(self.characters):
                raise ValueError(f"SEAT view requires 0 <= view_seat < {len(self.characters)}")
        else:
            raise ValueError(f"unknown view_mode {self.view_mode}")

    @property
    def n_seats(self) -> int:
        return len(self.characters)


@dataclass(frozen=True)
class StepCmd:
    step_id: int
    action_index: int


@dataclass(frozen=True)
class QuitCmd:
    pass


Command = Union[ResetCmd, StepCmd, QuitCmd]


def encode_reset(cmd: ResetCmd) -> bytes:
    """Complete wire message (length prefix included) for RESET."""
    chars = list(cmd.characters) + [0] * (4 - len(cmd.characters))
    seed = cmd.seed or 0
    return _LEN.pack(_RESET.size) + _RESET.pack(
        CMD_MAGIC, CmdType.RESET, cmd.n_seats, cmd.view_mode, cmd.view_seat,
        cmd.ascension, *chars, int(cmd.seed is not None), seed & _U32, seed >> 32, cmd.flags, cmd.max_decisions, 0,
    )


def encode_step(step_id: int, action_index: int) -> bytes:
    """Complete wire message (length prefix included) for STEP."""
    if not (_INT32_MIN <= step_id <= _INT32_MAX and _INT32_MIN <= action_index <= _INT32_MAX):
        raise ValueError("step_id / action_index out of int32 range")
    return _STEP_MSG.pack(_STEP.size, CMD_MAGIC, _CMD_STEP, step_id, action_index)


def encode_quit() -> bytes:
    """Complete wire message (length prefix included) for QUIT."""
    return _LEN.pack(_QUIT.size) + _QUIT.pack(CMD_MAGIC, CmdType.QUIT)


def decode_command(payload: bytes | bytearray | memoryview) -> Command:
    """Decode a Python -> host command payload (host side / fake hosts)."""
    n = len(payload)
    if n < 8 or n % 4:
        raise ProtocolError(f"command of {n} bytes")
    magic, ctype = struct.unpack_from("<Ii", payload)
    if magic != CMD_MAGIC:
        raise ProtocolError(f"bad command magic 0x{magic:08X}")
    if ctype == CmdType.RESET and n == _RESET.size:
        v = _RESET.unpack_from(payload)
        n_seats, view_mode, view_seat, asc = v[2:6]
        if not 1 <= n_seats <= 4:
            raise ProtocolError(f"RESET with n_seats={n_seats}")
        present, lo, hi, flags, max_dec = v[10:15]
        try:
            return ResetCmd(
                characters=tuple(v[6 : 6 + n_seats]), ascension=asc,
                seed=(lo | (hi << 32)) if present else None,
                view_mode=view_mode, view_seat=view_seat, flags=flags, max_decisions=max_dec,
            )
        except ValueError as e:
            raise ProtocolError(f"invalid RESET: {e}") from e
    if ctype == CmdType.STEP and n == _STEP.size:
        _, _, step_id, idx = _STEP.unpack_from(payload)
        return StepCmd(step_id, idx)
    if ctype == CmdType.QUIT and n == _QUIT.size:
        return QuitCmd()
    raise ProtocolError(f"unknown command type {ctype} / length {n}")


# --------------------------------------------------------------------------- semantic checks (schema 14.5)


def card_content_key(card_row: Sequence[int], var_rows: NDArray[np.integer] | None) -> tuple[int, ...]:
    """Multiset content key of a CARDS row (schema 5.2.3).

    Row columns minus pos/count/var_off/var_cnt/link_row/sel_count, followed by the
    instance's VARS rows sorted by (target_row, var_id), each contributing
    (var_id, target_row, shown, base, enchanted, flags). Compared as int32 tuples.
    """
    key = [int(card_row[c]) for c in CARD_KEY_COLS]
    if var_rows is not None and len(var_rows):
        v = np.asarray(var_rows)
        order = np.lexsort((v[:, VarCol.VAR_ID], v[:, VarCol.TARGET_ROW]))
        key.extend(v[order][:, VAR_KEY_COLS].ravel().tolist())
    return tuple(key)


def strictly_ascending(keys: NDArray[np.integer]) -> NDArray[np.bool_]:
    """For each consecutive row pair of ``keys`` (n, k): True iff row i < row i+1 lexicographically."""
    if keys.shape[0] < 2:
        return np.ones(0, dtype=bool)
    d = np.diff(keys.astype(np.int64), axis=0)
    nz = d != 0
    first = nz.argmax(axis=1)
    return nz.any(axis=1) & (d[np.arange(d.shape[0]), first] > 0)


def frame_problems(f: Frame) -> list[str]:
    """Deep consistency checks of a frame (schema 14.5); returns a list of problems.

    Pointer and CSR-slice bounds, slice ownership (every VARS/POWERS/INTENTS/REFS row
    claimed by exactly one owner slice that names it), canonical orders (VARS, ACTIONS,
    MAP_EDGES, MAP_NODES, CELLS, multiset zones) and SEATS/ACTIONS counts.
    """
    if f.elided:
        return [f"elided tables {f.elided} must be resolved before checking"]
    T: list[I32] = [f.table(t) for t in range(N_TABLES)]
    n = np.array([a.shape[0] for a in T], dtype=np.int64)
    out: list[str] = []

    for (t, col), tgt in POINTERS.items():
        v = T[t][:, col]
        if v.size and ((v < -1) | (v >= n[tgt])).any():
            out.append(f"{Table(t).name}.{col}: pointer outside [-1, {n[tgt]}) of {Table(tgt).name}")
    for t, tcol, rcol in DYNAMIC_POINTERS:
        a = T[t]
        if not a.shape[0]:
            continue
        tt, rr = a[:, tcol], a[:, rcol]
        if ((tt < -1) | (tt >= N_TABLES)).any():
            out.append(f"{Table(t).name}.{tcol}: table id out of range")
            continue
        has = tt >= 0
        lim = n[np.where(has, tt, 0)]
        if (has & ((rr < 0) | (rr >= lim))).any():
            out.append(f"{Table(t).name}.{rcol}: row outside referenced table")

    claimed = {s.child: np.zeros(n[s.child], dtype=np.int64) for s in SLICES}
    for s in SLICES:
        own, child = T[s.owner], T[s.child]
        off, cnt = own[:, s.off_col].astype(np.int64), own[:, s.cnt_col].astype(np.int64)
        name = f"{Table(s.owner).name}->{Table(s.child).name}"
        if (cnt < 0).any():
            out.append(f"{name}: negative count")
            continue
        zero = cnt == 0
        if (off[zero] != -1).any():
            out.append(f"{name}: cnt=0 with off != -1")
        nz = ~zero
        o, c = off[nz], cnt[nz]
        if ((o < 0) | (o + c > n[s.child])).any():
            out.append(f"{name}: slice out of bounds")
            continue
        total = int(c.sum())
        if not total:
            continue
        rows = np.arange(total) - np.repeat(np.cumsum(c) - c, c) + np.repeat(o, c)
        owners = np.repeat(np.flatnonzero(nz), c)
        if (child[rows, s.child_owner_row_col] != owners).any():
            out.append(f"{name}: child rows do not name their owner row")
        if s.child_owner_table_col is not None and (child[rows, s.child_owner_table_col] != s.owner).any():
            out.append(f"{name}: child rows do not name their owner table")
        np.add.at(claimed[s.child], rows, 1)
    for child_t, cl in claimed.items():
        if (cl != 1).any():
            out.append(f"{child_t.name}: {(cl == 0).sum()} unclaimed, {(cl > 1).sum()} multiply claimed rows")

    v = T[Table.VARS]
    if not strictly_ascending(v[:, [VarCol.OWNER_TABLE, VarCol.OWNER_ROW, VarCol.TARGET_ROW, VarCol.VAR_ID]]).all():
        out.append("VARS not strictly ascending by (owner_table, owner_row, target_row, var_id)")
    a = T[Table.ACTIONS]
    if not strictly_ascending(a[:, list(ACTION_SORT_COLS)]).all():
        out.append("ACTIONS not in canonical strictly ascending order")
    if a.shape[0] and ((a[:, ActionCol.VERB] <= Verb.INVALID) | (a[:, ActionCol.VERB] > max(Verb))).any():
        out.append("ACTIONS: invalid verb")
    n_seats = f.n_seats if f.is_obs or f.is_terminal else 0
    if a.shape[0] and ((a[:, ActionCol.SEAT] < 0) | (a[:, ActionCol.SEAT] >= n_seats)).any():
        out.append("ACTIONS: seat out of range")
    if f.is_obs and not a.shape[0]:
        out.append("OBS frame without legal actions")
    if f.is_terminal and a.shape[0]:
        out.append("TERMINAL frame with actions")
    e = T[Table.MAP_EDGES]
    if not strictly_ascending(e[:, [MapEdgeCol.FROM_NODE_ROW, MapEdgeCol.TO_NODE_ROW]]).all():
        out.append("MAP_EDGES not strictly ascending")
    m = T[Table.MAP_NODES]
    boss = (m[:, MapNodeCol.FLAGS] & (MapNodeFlag.BOSS | MapNodeFlag.SECOND_BOSS)) != 0
    n_boss = int(boss.sum())
    if n_boss and not boss[m.shape[0] - n_boss :].all():
        out.append("MAP_NODES: boss nodes not last")
    regular = m[~boss]
    if not strictly_ascending(regular[:, [MapNodeCol.ROW, MapNodeCol.COL]]).all():
        out.append("MAP_NODES not ascending by (row, col)")
    cells = T[Table.CELLS]
    if cells.shape[0] and not strictly_ascending(cells[:, [CellCol.Y, CellCol.X]]).all():
        out.append("CELLS not ascending by (y, x)")
    seats = T[Table.SEATS]
    if seats.shape[0]:
        if seats.shape[0] != n_seats:
            out.append(f"SEATS has {seats.shape[0]} rows, global n_seats={n_seats}")
        if (seats[:, SeatCol.SEAT] != np.arange(seats.shape[0])).any():
            out.append("SEATS not one row per seat in seat order")
    out.extend(_multiset_problems(T[Table.CARDS], v))
    if f.hd_rows >= 0:
        out.extend(hdraw_problems(f))
    return out


def hdraw_problems(f: Frame) -> list[str]:
    """The invariants of the HDRAW table (notes/critic/contract.md 1.3): rows ordered by (SEAT, POS); per seat
    POS = 0..n-1 without gaps and n = SEATS.DRAW_COUNT; every OWNER_ROW a CARDS row with ZONE = DRAW of that seat,
    and every DRAW row r of the seat named exactly CARDS[r].COUNT times; only seats whose DRAW rows the frame has."""
    hd = f.hdraw
    assert hd is not None, "frame without the HDRAW trailer"
    cards, seats = f.table(Table.CARDS), f.table(Table.SEATS)
    out: list[str] = []
    s, p, r = (hd[:, c].astype(np.int64) for c in (HDrawCol.SEAT, HDrawCol.POS, HDrawCol.OWNER_ROW))
    if hd.shape[0] > 1 and not strictly_ascending(np.stack([s, p], 1)).all():
        out.append("HDRAW not strictly ascending by (SEAT, POS)")
    if ((s < 0) | (s >= seats.shape[0])).any():
        return out + ["HDRAW: seat outside SEATS"]
    if ((r < 0) | (r >= cards.shape[0])).any():
        return out + ["HDRAW: OWNER_ROW outside CARDS"]
    if (cards[r, CardCol.ZONE] != Zone.DRAW).any() or (cards[r, CardCol.SEAT] != s).any():
        out.append("HDRAW: OWNER_ROW not a DRAW row of the seat")
    draw = cards[:, CardCol.ZONE] == Zone.DRAW
    for seat in range(seats.shape[0]):
        mine = s == seat
        rows_here = np.flatnonzero(draw & (cards[:, CardCol.SEAT] == seat))
        n = int(mine.sum())
        if not n and not rows_here.size:
            continue
        if not rows_here.size:
            out.append(f"HDRAW: seat {seat} has rows but no DRAW rows in CARDS")
            continue
        if (p[mine] != np.arange(n)).any():
            out.append(f"HDRAW: seat {seat} POS not 0..{n - 1}")
        if n != int(seats[seat, SeatCol.DRAW_COUNT]):
            out.append(f"HDRAW: seat {seat} has {n} rows, DRAW_COUNT {int(seats[seat, SeatCol.DRAW_COUNT])}")
        named = np.bincount(r[mine], minlength=cards.shape[0])
        if (named[rows_here] != cards[rows_here, CardCol.COUNT]).any() or named.sum() != named[rows_here].sum():
            out.append(f"HDRAW: seat {seat} OWNER_ROW multiplicities differ from CARDS.COUNT")
    return out


def _multiset_problems(cards: I32, vars_: I32) -> list[str]:
    zone, pos = cards[:, CardCol.ZONE], cards[:, CardCol.POS]
    deduped = np.isin(zone, [int(z) for z in MULTISET_ZONES]) | ((zone == Zone.CHOICE) & (pos == -1))
    out: list[str] = []
    if not deduped.any():
        return out
    if (cards[deduped, CardCol.COUNT] < 1).any():
        out.append("CARDS: multiset group with count < 1")
    if (pos[deduped] != -1).any():
        out.append("CARDS: multiset group with pos != -1")
    last: dict[tuple[int, int], tuple[int, ...]] = {}
    for r in np.flatnonzero(deduped).tolist():
        row = cards[r]
        o, k = int(row[CardCol.VAR_OFF]), int(row[CardCol.VAR_CNT])
        key = card_content_key(row, vars_[o : o + k] if k > 0 else None)
        g = (int(row[CardCol.ZONE]), int(row[CardCol.SEAT]))
        prev = last.get(g)
        if prev is not None and not prev < key:
            out.append(f"CARDS row {r}: multiset zone {Zone(g[0]).name} seat {g[1]} not strictly ascending")
        last[g] = key
    return out


def check_frame(f: Frame) -> None:
    """Raise :class:`SchemaError` listing every problem found by :func:`frame_problems`."""
    problems = frame_problems(f)
    if problems:
        raise SchemaError("; ".join(problems))
