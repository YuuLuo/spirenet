from __future__ import annotations

import numpy as np
import pytest
import torch

from spire.proto import Frame, parse_frame
from spire.schema import DECISION_KINDS, ActionCol, G, Kind, OptionCol, OptionFlag, OptionKind, Table, Verb
from spire.synth import FrameBuilder, FrameGen, random_frame
from spire.tokenize import SPECS, Ent, FeatureDecoder, Tokenizer, TokenizerConfig
from spire.tokenize_ref import encode_reference
from spire.vocab import Vocab


def rest_frame(vocab: Vocab, *, votes: int = 0b1011) -> Frame:
    b = FrameBuilder()
    b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.ACTING_SEAT] = Kind.REST, 4, 1
    b.glob[G.SEAT_SCREEN0:G.SEAT_SCREEN3 + 1] = [Kind.MAP, Kind.REST, Kind.REWARD, Kind.EVENT]
    b.glob[G.SEAT_STATUS0:G.SEAT_STATUS3 + 1] = [3, 32, 16, 12]
    for seat in range(4):
        b.add(Table.SEATS, seat=seat, hp=10 + seat, max_hp=80)
    for pos, name in enumerate(("HEAL", "SMITH", "MEND")):
        source = b.add(Table.OPTIONS, kind=OptionKind.REST_OPTION, seat=1, pos=pos,
                       key_id=vocab["rest_option"].index[name], flags=votes << 7)
        for target in ((0, 2, 3) if name == "MEND" else (0,)):
            b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=1, src_table=Table.OPTIONS,
                  src_row=source, aux=target)
    b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=1)
    return parse_frame(b.build())


@pytest.mark.parametrize("native", [False, True])
def test_mend_only_adds_target_relation_and_keeps_aux_for_gate(vocab: Vocab, native: bool) -> None:
    legacy = Tokenizer(vocab, native=native)
    tok = Tokenizer(vocab, TokenizerConfig(mend_target=True), native=native)
    if native:
        assert tok.native, "build native/spire_native.c before the stage-B native proof"
    f = rest_frame(vocab)
    before, after = legacy.encode(f), tok.encode(f)
    for field in ("types", "raw", "edges", "cond", "history"):
        assert np.array_equal(getattr(before, field), getattr(after, field)), field
    expected = before.rel.copy()
    first_action = sum(before.counts[:3])
    for row, action in enumerate(f.actions):
        if action[ActionCol.SRC_ROW] == 2 and action[ActionCol.SRC_TABLE] == Table.OPTIONS:
            expected[first_action + row, 1] = 2 + action[ActionCol.AUX]
    assert np.array_equal(after.rel, expected)
    assert np.array_equal(after.rel, encode_reference(tok, f).rel)


@pytest.mark.parametrize("pooled", [False, True])
def test_stage_b_paths_match_every_kind(gen: FrameGen, vocab: Vocab, pooled: bool) -> None:
    config = TokenizerConfig(pool_children=pooled, mend_target=True, coop_features=True)
    numpy_tok, native_tok = Tokenizer(vocab, config, native=False), Tokenizer(vocab, config)
    if pooled:
        assert native_tok.native
    rng = np.random.default_rng(342)
    frames = [parse_frame(random_frame(gen, rng, kind, n_seats=seats))
              for kind in DECISION_KINDS for seats in (1, 2, 3, 4)]
    frames.append(rest_frame(vocab))
    for frame in frames:
        ref = encode_reference(numpy_tok, frame)
        for tok in (numpy_tok, native_tok):
            actual = tok.encode(frame)
            assert actual.counts == ref.counts
            for name in ("types", "raw", "rel", "edges", "cond"):
                assert np.array_equal(getattr(actual, name), getattr(ref, name)), (frame.kind, name)
    encoded = native_tok.encode_batch(frames, sort=False)
    for i, frame in enumerate(frames):
        expected, actual = encode_reference(numpy_tok, frame), encoded.sample(i)
        for name in ("types", "raw", "rel", "edges", "cond"):
            assert np.array_equal(getattr(actual, name), getattr(expected, name)), (i, name)


def test_status_screen_and_symmetric_votes(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(coop_features=True))
    f = rest_frame(vocab)
    sample = tok.encode(f)
    layout = tok.layout
    seats = sample.types == Ent.SEAT
    opts = sample.types == Ent.OPTION
    packed = sample.raw[seats, layout.slot(Ent.SEAT, "acts")]
    assert ((packed >> 1) & 15).tolist() == [1, 8, 4, 6]
    assert ((packed >> 5) & 63).tolist() == [3, 32, 16, 12]
    assert sample.raw[opts, layout.slot(Ent.OPTION, "vote_count")].tolist() == [3, 3, 3]
    decoder = FeatureDecoder(layout)
    decoded = decoder(torch.from_numpy(sample.types.astype(np.int64)), torch.from_numpy(sample.raw))
    wide = decoded[1]
    cols = layout.num_columns[Ent.SEAT][len(SPECS[Ent.SEAT].nums):]
    assert wide[seats][:, cols].tolist() == [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0]]
    vote_col = layout.num_columns[Ent.OPTION][-1]
    assert wide[opts, vote_col].tolist() == [0.75, 0.75, 0.75]
    # Three votes are three votes under every permutation of the four seat labels.
    for bits in (0b0111, 0b1011, 0b1101, 0b1110):
        changed = tok.encode(rest_frame(vocab, votes=bits))
        assert changed.raw[opts, layout.slot(Ent.OPTION, "vote_count")].tolist() == [3, 3, 3]


def test_layout_extensions_keep_legacy_id_and_numeric_indices(vocab: Vocab) -> None:
    old, new = Tokenizer(vocab).layout, Tokenizer(vocab, TokenizerConfig(coop_features=True)).layout
    assert old.regions == new.regions and old.static == new.static
    assert old.n_rows == new.n_rows and old.n_id_slots == new.n_id_slots
    assert old.width == new.width
    assert old.n_wide == new.legacy_n_wide and new.n_wide == old.n_wide + 7
    for entity, spec in SPECS.items():
        assert old.num_columns[entity] == new.num_columns[entity][:len(spec.nums)]
        for column in old.slot_cols[entity]:
            assert old.slot(entity, int(column)) == new.slot(entity, int(column))
        for derived in spec.derived:
            assert old.slot(entity, derived) == new.slot(entity, derived)


@pytest.mark.parametrize("seats", [1, 2, 3, 4])
def test_synthetic_mend_has_only_other_seat_targets(gen: FrameGen, vocab: Vocab, seats: int) -> None:
    seen = 0
    for seed in range(128):
        frame = parse_frame(random_frame(gen, np.random.default_rng(seed), Kind.REST, n_seats=seats))
        for row, option in enumerate(frame.table(Table.OPTIONS)):
            if option[OptionCol.KIND] != OptionKind.REST_OPTION or option[OptionCol.KEY_ID] != vocab.id("rest_option", "MEND"):
                continue
            assert seats > 1, "MEND does not exist in a single-seat host episode (H5-6)"
            actions = frame.actions
            selected = ((actions[:, ActionCol.VERB] == Verb.CHOOSE_OPTION) &
                        (actions[:, ActionCol.SRC_TABLE] == Table.OPTIONS) & (actions[:, ActionCol.SRC_ROW] == row))
            targets = actions[selected, ActionCol.AUX].tolist()
            if option[OptionCol.FLAGS] & OptionFlag.LOCKED:
                assert targets == []
            else:
                assert targets == [seat for seat in range(seats) if seat != option[OptionCol.SEAT]]
                seen += 1
    assert seen == 0 if seats == 1 else seen > 0


@pytest.mark.parametrize("bad", ["self_target", "outside_target", "outside_source", "empty_source"])
def test_mend_invalid_references_fail_in_all_paths(vocab: Vocab, bad: str) -> None:
    b = FrameBuilder()
    b.glob[G.KIND], b.glob[G.N_SEATS] = Kind.REST, 1
    b.add(Table.SEATS, seat=0)
    if bad != "empty_source":
        b.add(Table.OPTIONS, kind=OptionKind.REST_OPTION, seat=0,
              key_id=vocab.id("rest_option", "HEAL" if bad == "outside_source" else "MEND"))
    b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS,
          src_row=1 if bad == "outside_source" else 0, aux=4 if bad == "outside_target" else 0)
    frame = parse_frame(b.build())
    for native in (False, True):
        tok = Tokenizer(vocab, TokenizerConfig(mend_target=True), native=native)
        with pytest.raises(AssertionError):
            tok.encode(frame)
        with pytest.raises(AssertionError):
            encode_reference(tok, frame)


def test_single_seat_nonzero_extension_fields_are_not_silently_dropped(vocab: Vocab) -> None:
    # H5 writes zero here in real SP frames. The encoder still preserves reference
    # semantics for supplied nonzero data instead of masking it by n_seats.
    b = FrameBuilder()
    b.glob[G.KIND], b.glob[G.N_SEATS] = Kind.REST, 1
    b.glob[G.SEAT_SCREEN0], b.glob[G.SEAT_STATUS0] = Kind.REST, 32
    b.add(Table.SEATS, seat=0)
    b.add(Table.OPTIONS, kind=OptionKind.REST_OPTION, seat=0, flags=2 << 7,
          key_id=vocab.id("rest_option", "HEAL"))
    b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=0)
    frame = parse_frame(b.build())
    for native in (False, True):
        tok = Tokenizer(vocab, TokenizerConfig(coop_features=True, mend_target=True), native=native)
        sample = tok.encode(frame)
        assert np.array_equal(sample.raw, encode_reference(tok, frame).raw)
        assert sample.raw[2, tok.layout.slot(Ent.SEAT, "acts")] == 1 | (Kind.REST << 1) | (32 << 5)
        assert sample.raw[3, tok.layout.slot(Ent.OPTION, "vote_count")] == 1
