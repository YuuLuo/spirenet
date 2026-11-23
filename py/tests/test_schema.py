from __future__ import annotations

import pytest

from spire import schema as S


def test_layout_constants() -> None:
    assert (S.GLOBAL_OFFSET, S.DIR_OFFSET, S.DATA_OFFSET) == (64, 320, 640)
    assert S.N_COLS == (32, 25, 8, 16, 10, 10, 9, 7, 5, 3, 6, 2, 14, 6, 6, 7)
    assert S.FRAME_MAGIC.to_bytes(4, "little") == b"SPOB"
    assert S.CMD_MAGIC.to_bytes(4, "little") == b"SPCM"
    assert S.FNV_EMPTY_I32 & 0xFFFFFFFF == 2166136261
    assert S.MAX_FRAME_BYTES == 640 + 4 * sum(r * c for r, c in zip(S.MAX_ROWS, S.N_COLS))


def test_column_maps_cover_every_table() -> None:
    for t in S.Table:
        cols = S.COLUMN_INDEX[t]
        assert sorted(cols.values()) == list(range(S.N_COLS[t]))
    assert S.COLUMN_INDEX[S.Table.CARDS]["sel_count"] == 24
    assert S.COLUMN_INDEX[S.Table.INTENTS]["dmg_seat3"] == 8
    assert S.COLUMN_INDEX[S.Table.SEATS]["hp_loss_events_combat"] == 31
    assert S.COLUMN_INDEX[S.Table.OPTIONS]["ref_cnt"] == 13


def test_appendix_a_values() -> None:
    assert [k.value for k in S.Kind] == list(range(14))
    assert len(S.DECISION_KINDS) == 12
    assert S.Verb.CRYSTAL_CLICK == 15 and S.Verb.END_TURN == 1
    assert S.Zone.PREVIEW == 9 and S.CardSelectSub.CHOOSE_A_CARD == 12
    assert S.OptionKind.CONFIRM_CHOICE == 8 and S.CreatureKind.ENEMY == 3
    assert S.G.PLAYERS_ALIVE == 43 and S.H.RESERVED == 15
    assert int(S.FrameFlag.TARGET_PREVIEW) == 1 << 7
    assert int(S.ResetFlag.ELIDE_UNCHANGED) == 1 << 5


def _good_directory(rows: list[int]) -> tuple[list[int], int]:
    d: list[int] = []
    off = S.DATA_OFFSET
    for t, n in enumerate(rows):
        d += [t, n, S.N_COLS[t], off, 0]
        off += n * S.N_COLS[t] * 4
    return d, off


def test_validate_directory_accepts_packed_tables() -> None:
    rows = [1, 3, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4]
    d, end = _good_directory(rows)
    S.validate_directory(d, end, 0)


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda d: d.__setitem__(5 * 2 + 0, 7), "table_id"),
        (lambda d: d.__setitem__(5 * 1 + 2, 24), "n_cols"),
        (lambda d: d.__setitem__(5 * 3 + 3, d[5 * 3 + 3] + 4), "byte_offset"),
        (lambda d: d.__setitem__(5 * 0 + 1, 5), "n_rows"),
        (lambda d: d.__setitem__(5 * 4 + 1, -1), "ELIDED"),
    ],
)
def test_validate_directory_rejects(mutate, msg: str) -> None:  # type: ignore[no-untyped-def]
    d, end = _good_directory([1, 3, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4])
    mutate(d)
    with pytest.raises(S.SchemaError, match=msg):
        S.validate_directory(d, end, 0)


def test_validate_directory_elided_and_length() -> None:
    d, end = _good_directory([1, 3, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4])
    with pytest.raises(S.SchemaError, match="ends at"):
        S.validate_directory(d, end + 4, 0)
    # Elide CARDS: it consumes no bytes, later offsets shift down.
    rows = [1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4]
    d2, end2 = _good_directory(rows)
    d2[5 * 1 + 1], d2[5 * 1 + 3] = -1, 0
    S.validate_directory(d2, end2, int(S.FrameFlag.ELIDED))
