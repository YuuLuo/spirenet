from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from spire import proto as P
from spire.schema import (
    DATA_OFFSET,
    DECISION_KINDS,
    N_COLS,
    N_TABLES,
    ActionCol,
    CardCol,
    FrameFlag,
    G,
    H,
    Kind,
    MsgType,
    Result,
    SchemaError,
    Table,
    VarCol,
    Verb,
    ViewMode,
    Zone,
)
from spire.synth import FrameBuilder, FrameGen, error_frame, random_frame


def reencode(f: P.Frame) -> bytes:
    parts = [f.header, f.glob, f.directory.ravel(), *[f.table(t).ravel() for t in range(N_TABLES)]]
    return np.concatenate(parts).astype("<i4").tobytes()


def words_of(payload: bytes | bytearray | memoryview) -> np.ndarray:
    return np.frombuffer(bytearray(payload), dtype="<i4")


# ---------------------------------------------------------------- hashing


def test_fnv1a_reference_vectors() -> None:
    assert P.fnv1a32(b"") == 0x811C9DC5
    assert P.fnv1a32(b"a") == 0xE40C292C
    assert P.fnv1a32(b"foobar") == 0xBF9CF968


# ---------------------------------------------------------------- builder -> bytes -> parse


def small_combat() -> bytes:
    b = FrameBuilder()
    b.glob[G.KIND], b.glob[G.N_SEATS] = Kind.COMBAT, 1
    b.episode_id, b.step_id, b.vocab_hash, b.catalog_hash = 7, 3, -12345, 999
    b.add(Table.SEATS, seat=0, character_id=4, hp=50, max_hp=80)
    p = b.add(Table.CREATURES, kind=1, seat=0, model_id=4, hp=50, max_hp=80, alive=1, hittable=1)
    e = b.add(Table.CREATURES, kind=3, seat=-1, model_id=9, hp=20, max_hp=30, alive=1, hittable=1)
    b.rows(Table.SEATS)[0, 2] = p
    b.attach(Table.POWERS, p, [[0, 11, 2, 2, 2, -1, 1, -1, 0, 0]])
    b.attach(Table.INTENTS, e, [[0, 0, 3, 2, 0, 6, -1, -1, -1, 1]])
    h0 = b.add(Table.CARDS, zone=Zone.HAND, seat=0, pos=0, count=1, card_id=17, aux=-1)
    h1 = b.add(Table.CARDS, zone=Zone.HAND, seat=0, pos=1, count=1, card_id=18, aux=-1)
    # deliberately out of order: (target_row, var_id) sorting happens at build
    b.attach_vars(Table.CARDS, h0, [[0, 0, e, 5, 9, 6, 6, 0], [0, 0, -1, 5, 6, 6, 6, 0], [0, 0, -1, 3, 1, 1, 1, 0]])
    b.attach_vars(Table.CARDS, h1, [[0, 0, -1, 4, 8, 8, 8, 1]])
    b.attach_vars(Table.POWERS, 0, [[0, 0, -1, 2, 2, 2, 2, 0]])
    b.add(Table.ACTIONS, verb=Verb.PLAY_CARD, seat=0, src_table=Table.CARDS, src_row=h1, tgt_row=-1)
    b.add(Table.ACTIONS, verb=Verb.END_TURN, seat=0)
    b.add(Table.ACTIONS, verb=Verb.PLAY_CARD, seat=0, src_table=Table.CARDS, src_row=h0, tgt_row=e)
    return b.build()


def test_builder_parse_roundtrip_small() -> None:
    payload = small_combat()
    f = P.parse_frame(payload, verify_hashes=True)
    assert (f.msg_type, f.episode_id, f.step_id, f.vocab_hash, f.catalog_hash) == (MsgType.OBS, 7, 3, -12345, 999)
    assert f.kind == Kind.COMBAT and f.n_actions == 3 and len(payload) == f.header[H.FRAME_BYTES]
    assert [f.table(t).shape for t in range(N_TABLES)][:6] == [(1, 32), (2, 25), (5, 8), (2, 16), (1, 10), (1, 10)]
    v = f.table(Table.VARS)
    # VARS grouped by owner (CARDS 0, CARDS 1, POWERS 0) and sorted by (target_row, var_id)
    assert v[:, [VarCol.OWNER_TABLE, VarCol.OWNER_ROW, VarCol.TARGET_ROW, VarCol.VAR_ID]].tolist() == [
        [1, 0, -1, 3], [1, 0, -1, 5], [1, 0, 1, 5], [1, 1, -1, 4], [4, 0, -1, 2]]
    c = f.table(Table.CARDS)
    assert c[:, [CardCol.VAR_OFF, CardCol.VAR_CNT]].tolist() == [[0, 3], [3, 1]]
    a = f.actions
    assert a[:, ActionCol.VERB].tolist() == [Verb.END_TURN, Verb.PLAY_CARD, Verb.PLAY_CARD]
    assert a[1:, ActionCol.SRC_ROW].tolist() == [0, 1]
    assert P.frame_problems(f) == []
    assert reencode(f) == payload


@pytest.mark.parametrize("kind", DECISION_KINDS, ids=lambda k: Kind(k).name)
@pytest.mark.parametrize("n_seats", [1, 2, 4])
def test_roundtrip_every_kind(gen: FrameGen, kind: int, n_seats: int) -> None:
    rng = np.random.default_rng(1000 * kind + n_seats)
    for size in ("typical", "extreme"):
        for _ in range(3):
            payload = random_frame(gen, rng, kind, n_seats=n_seats, size=size)  # type: ignore[arg-type]
            f = P.parse_frame(payload, verify_hashes=True)
            assert f.kind == kind and f.n_seats == n_seats and f.is_obs
            assert P.frame_problems(f) == []
            assert reencode(f) == payload


def test_frame_views_are_zero_copy_and_read_only(gen: FrameGen) -> None:
    payload = random_frame(gen, np.random.default_rng(5), Kind.COMBAT)
    f = P.parse_frame(bytearray(payload))
    for t in range(N_TABLES):
        a = f.table(t)
        assert a.dtype == np.int32 and a.shape[1] == N_COLS[t]
        assert np.shares_memory(a, f.words) or a.size == 0
        with pytest.raises(ValueError):
            a[...] = 0
    assert np.shares_memory(f.words, np.frombuffer(f.payload, dtype=np.uint8))


def test_terminal_and_error_frames(gen: FrameGen) -> None:
    rng = np.random.default_rng(9)
    ctx = gen.random_ctx(rng, 2)
    t = P.parse_frame(gen.terminal(rng, ctx, Result.WIN), verify_hashes=True)
    assert t.is_terminal and t.result == Result.WIN and t.flags & FrameFlag.TERMINAL and t.n_actions == 0
    assert t.kind == Kind.TERMINAL and P.frame_problems(t) == []
    e = P.parse_frame(error_frame(2, episode_id=4, step_id=11), verify_hashes=True)
    assert e.is_error and e.error_code == 2 and (e.episode_id, e.step_id) == (4, 11)
    with pytest.raises(ValueError):
        _ = t.error_code


# ---------------------------------------------------------------- malformed input


def _mutated(payload: bytes, idx: int, value: int) -> bytes:
    w = words_of(payload)
    w[idx] = value
    return w.tobytes()


@pytest.mark.parametrize(
    "fn, msg",
    [
        (lambda p: p[:600], "too short"),
        (lambda p: p + b"\0\0", "aligned"),
        (lambda p: _mutated(p, H.MAGIC, 0x12345678), "magic"),
        (lambda p: _mutated(p, H.SCHEMA_VERSION, 2), "schema version"),
        (lambda p: _mutated(p, H.N_TABLES, 15), "n_tables"),
        (lambda p: _mutated(p, H.FRAME_BYTES, len(p) + 4), "frame_bytes"),
        (lambda p: _mutated(p, H.MSG_TYPE, 9), "msg_type"),
        (lambda p: _mutated(p, H.RESULT, Result.WIN), "OBS frame"),
        (lambda p: _mutated(p, H.MSG_TYPE, MsgType.TERMINAL), "TERMINAL frame"),
        (lambda p: _mutated(p, H.N_ACTIONS, 99), "n_actions"),
        (lambda p: _mutated(p, 80 + 5 * Table.CARDS + 2, 24), "n_cols"),
        (lambda p: _mutated(p, 80 + 5 * Table.CARDS + 1, 1), "byte_offset|ends at"),
        (lambda p: _mutated(p, 80 + 5 * Table.VARS + 3, DATA_OFFSET), "byte_offset"),
        (lambda p: _mutated(p, 80 + 5 * Table.SEATS + 1, 9), "n_rows"),
        (lambda p: p + b"\0\0\0\0", "frame_bytes"),
    ],
)
def test_malformed_frames_rejected(fn, msg: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SchemaError, match=msg):
        P.parse_frame(fn(small_combat()))


def test_hash_verification_is_switchable() -> None:
    w = words_of(small_combat())
    off = int(w[80 + 5 * Table.CARDS + 3]) // 4
    w[off + CardCol.CARD_ID] += 1
    bad = w.tobytes()
    P.parse_frame(bad)  # hashes not checked by default
    with pytest.raises(P.HashMismatch, match="CARDS"):
        P.parse_frame(bad, verify_hashes=True)


def test_frame_problems_detects_semantic_corruption() -> None:
    base = small_combat()
    w = words_of(base)
    d = w[80:160].reshape(16, 5)

    def cell(t: int, row: int, col: int) -> int:
        return int(d[t, 3]) // 4 + row * N_COLS[t] + col

    cases = {
        "pointer": (cell(Table.CARDS, 0, CardCol.LINK_ROW), 5),
        "slice out of bounds": (cell(Table.CARDS, 1, CardCol.VAR_CNT), 9),
        "do not name their owner": (cell(Table.VARS, 0, VarCol.OWNER_ROW), 1),
        "VARS not strictly ascending": (cell(Table.VARS, 0, VarCol.VAR_ID), 7),
        "ACTIONS not in canonical": (cell(Table.ACTIONS, 0, ActionCol.VERB), Verb.PROCEED),
    }
    for needle, (idx, value) in cases.items():
        x = w.copy()
        x[idx] = value
        problems = P.frame_problems(P.parse_frame(x.tobytes()))
        assert any(needle in p for p in problems), (needle, problems)


def test_multiset_order_violation_detected(gen: FrameGen) -> None:
    rng = np.random.default_rng(3)
    while True:
        f = P.parse_frame(random_frame(gen, rng, Kind.MAP))
        deck = np.flatnonzero(f.table(Table.CARDS)[:, CardCol.ZONE] == Zone.DECK)
        if len(deck) >= 2:
            break
    w = words_of(f.payload)
    off = int(f.directory[Table.CARDS, 3]) // 4
    a, b = (off + int(r) * N_COLS[Table.CARDS] for r in deck[:2])
    ra, rb = w[a : a + 25].copy(), w[b : b + 25].copy()
    # swap the two groups' rows including their VARS slices -> still consistent, but out of order
    w[a : a + 25], w[b : b + 25] = rb, ra
    v_off = int(f.directory[Table.VARS, 3]) // 4
    for r_new, src in ((deck[0], rb), (deck[1], ra)):
        o, k = int(src[CardCol.VAR_OFF]), int(src[CardCol.VAR_CNT])
        for j in range(k):
            w[v_off + (o + j) * 8 + VarCol.OWNER_ROW] = r_new
    problems = P.frame_problems(P.parse_frame(w.tobytes()))
    assert any("multiset" in p for p in problems), problems


def test_elided_tables_resolve_from_previous(gen: FrameGen) -> None:
    prev_payload = random_frame(gen, np.random.default_rng(4), Kind.MAP)
    prev = P.parse_frame(prev_payload)
    # Re-pack prev with MAP_NODES elided, as the host does under ELIDE_UNCHANGED.
    t_el = Table.MAP_NODES
    d = prev.directory.copy()
    data = [prev.table(t).ravel() for t in range(N_TABLES)]
    data[t_el] = np.zeros(0, dtype=np.int32)
    off = DATA_OFFSET
    for t in range(N_TABLES):
        if t == t_el:
            d[t, 1], d[t, 3] = -1, 0
            continue
        d[t, 3] = off
        off += data[t].size * 4
    h = prev.header.copy()
    h[H.FLAGS] |= FrameFlag.ELIDED
    h[H.FRAME_BYTES] = off
    payload = np.concatenate([h, prev.glob, d.ravel(), *data]).astype("<i4").tobytes()
    f = P.parse_frame(payload, verify_hashes=True)
    assert f.elided == (t_el,)
    assert P.frame_problems(f)[0].startswith("elided")
    f.resolve_elided(prev)
    assert np.array_equal(f.table(t_el), prev.table(t_el)) and P.frame_problems(f) == []


# ---------------------------------------------------------------- framing


class Trickle(io.BytesIO):
    """A raw stream that returns at most ``k`` bytes per read (like a pipe)."""

    def __init__(self, data: bytes, k: int) -> None:
        super().__init__(data)
        self.k = k

    def readinto(self, b) -> int:  # type: ignore[no-untyped-def,override]
        chunk = self.read(min(self.k, len(b)))
        b[: len(chunk)] = chunk
        return len(chunk)


def test_read_message_handles_short_reads_and_eof() -> None:
    p1, p2 = small_combat(), error_frame(1)
    stream = Trickle(P.frame_message(p1) + P.frame_message(p2), k=7)
    assert bytes(P.read_message(stream)) == p1  # type: ignore[arg-type]
    assert bytes(P.read_message(stream)) == p2  # type: ignore[arg-type]
    assert P.read_message(stream) is None


def test_read_message_rejects_truncation_and_oversize() -> None:
    msg = P.frame_message(small_combat())
    with pytest.raises(P.ProtocolError, match="EOF"):
        P.read_message(io.BytesIO(msg[:-3]))
    with pytest.raises(P.ProtocolError, match="EOF"):
        P.read_message(io.BytesIO(msg[:2]))
    with pytest.raises(P.ProtocolError, match="exceeds"):
        P.read_message(io.BytesIO(struct.pack("<I", 1 << 30) + b"x"))


def test_write_message_roundtrip() -> None:
    out = io.BytesIO()
    P.write_message(out, P.encode_step(3, 4))
    out.seek(0)
    assert P.decode_command(P.read_message(out)) == P.StepCmd(3, 4)  # type: ignore[arg-type]


# ---------------------------------------------------------------- control messages


def test_command_golden_bytes() -> None:
    assert P.encode_step(5, 2) == struct.pack("<I4i", 16, 0x4D435053, 2, 5, 2)
    assert P.encode_quit() == struct.pack("<I2i", 8, 0x4D435053, 3)
    seed = 0xFEDCBA9876543210
    msg = P.encode_reset(P.ResetCmd((4, 6), ascension=10, seed=seed, view_mode=ViewMode.SEAT, view_seat=1, flags=0x1D))
    w = struct.unpack("<I16I", msg)
    assert w[0] == 64
    assert list(w[1:]) == [0x4D435053, 1, 2, 1, 1, 10, 4, 6, 0, 0, 1, 0x76543210, 0xFEDCBA98, 0x1D, 0, 0]
    w0 = struct.unpack("<I16i", P.encode_reset(P.ResetCmd((3,))))
    assert w0[3:6] == (1, 0, -1) and w0[11:14] == (0, 0, 0)


@pytest.mark.parametrize(
    "cmd",
    [
        P.ResetCmd((5,)),
        P.ResetCmd((1, 2, 3, 4), ascension=7, seed=(1 << 64) - 1, flags=0x3F),
        P.ResetCmd((2, 2), seed=0, view_mode=ViewMode.SEAT, view_seat=0),
        P.StepCmd(0, 0),
        P.StepCmd(123456, -1),
        P.QuitCmd(),
    ],
)
def test_command_roundtrip(cmd: P.Command) -> None:
    enc = {P.ResetCmd: lambda c: P.encode_reset(c), P.StepCmd: lambda c: P.encode_step(c.step_id, c.action_index),
           P.QuitCmd: lambda c: P.encode_quit()}[type(cmd)](cmd)
    assert int.from_bytes(enc[:4], "little") == len(enc) - 4
    assert P.decode_command(enc[4:]) == cmd


def test_command_validation() -> None:
    with pytest.raises(ValueError):
        P.ResetCmd(())
    with pytest.raises(ValueError):
        P.ResetCmd((1,) * 5)
    with pytest.raises(ValueError):
        P.ResetCmd((1,), seed=1 << 64)
    with pytest.raises(ValueError):
        P.ResetCmd((1, 2), view_mode=ViewMode.SEAT, view_seat=2)
    with pytest.raises(ValueError):
        P.ResetCmd((1,), view_seat=0)
    with pytest.raises(ValueError):
        P.encode_step(1 << 31, 0)
    with pytest.raises(P.ProtocolError, match="magic"):
        P.decode_command(struct.pack("<2i", 1, 3))
    with pytest.raises(P.ProtocolError, match="unknown"):
        P.decode_command(struct.pack("<Ii", 0x4D435053, 9))
    with pytest.raises(P.ProtocolError, match="n_seats"):
        P.decode_command(P.encode_reset(P.ResetCmd((1,)))[4:12] + struct.pack("<14i", 0, *([0] * 13)))
