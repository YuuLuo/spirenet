"""Batch tokenizer (``Tokenizer.encode_batch``) and encoded collation (``collate_encoded``) are
bit-identical to the frozen per-frame reference (``spire.tokenize_ref``) and the sample collation."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pytest
import torch

from spire import ipc
from spire.actor import ActorConfig, group_ranges
from spire.proto import Frame, parse_frame
from spire.rollout import EpisodeMeta, Segment
from spire.schema import (
    DATA_OFFSET,
    DECISION_KINDS,
    GLOBAL_OFFSET,
    N_TABLES,
    ActionCol,
    CardCol,
    CreatureCol,
    FrameFlag,
    G,
    H,
    Kind,
    MapEdgeCol,
    MapNodeCol,
    OptionCol,
    PowerCol,
    RefCol,
    SeatCol,
    Table,
    VarCol,
)
from spire.synth import FrameGen, random_frame
from spire.tokenize import BatchShape, EncodedBatch, Sample, SampleBlock, Tokenizer, TokenizerConfig
from spire.tokenize_ref import encode_reference
from spire.vocab import Vocab

ARRAYS = ("types", "raw", "rel", "edges", "cond")


def assert_same_sample(a: Sample, b: Sample, ctx: object = "") -> None:
    for k in ARRAYS:
        x, y = getattr(a, k), getattr(b, k)
        assert x.dtype == y.dtype and x.shape == y.shape, (ctx, k, x.dtype, y.dtype, x.shape, y.shape)
        assert np.array_equal(x, y), (ctx, k)
    assert a.counts == b.counts and all(type(c) is int for c in a.counts), (ctx, a.counts, b.counts)


def assert_same_batch(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], ctx: object = "") -> None:
    assert a.keys() == b.keys(), ctx
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, (ctx, k, a[k].shape, b[k].shape)
        assert torch.equal(a[k], b[k]), (ctx, k)


def corpus(gen: FrameGen, seed: int = 0, reps: int = 1) -> list[bytes]:
    """Every decision kind x 1-4 seats x typical / extreme sizes, TEAM and SEAT views, + TERMINAL frames."""
    rng = np.random.default_rng(seed)
    out: list[bytes] = []
    for _ in range(reps):
        for k in DECISION_KINDS:
            for s in (1, 2, 3, 4):
                for size in ("typical", "extreme"):
                    out.append(random_frame(gen, rng, k, n_seats=s, size=size))  # type: ignore[arg-type]
        for s in (1, 2, 3, 4):
            ctx = gen.random_ctx(rng, s)
            out.append(gen.terminal(rng, ctx, int(rng.integers(1, 4))))
    return out


@pytest.fixture(scope="module")
def frames(gen: FrameGen) -> list[Frame]:
    return [parse_frame(p) for p in corpus(gen, 0, reps=2)]


@pytest.fixture(scope="module", params=[(True, True), (True, False), (False, True), (False, False)],
                ids=["pooled-native", "pooled-numpy", "tokens-native-collate", "tokens-numpy"])
def tok(vocab: Vocab, request: Any) -> Tokenizer:
    """native: C encoder (pooled layout only) + C collation; numpy: the vectorised fallback."""
    pooled, native = request.param
    t = Tokenizer(vocab, TokenizerConfig(pool_children=pooled), native=native)
    if native and not t._native_collate:
        pytest.skip("spire._native is not built (python -B native/build.py)")
    assert t.native == (native and pooled) and t._native_collate == native
    return t


def test_single_frame_encode_matches_reference(tok: Tokenizer, frames: list[Frame]) -> None:
    for i, f in enumerate(frames):
        assert_same_sample(tok.encode(f), encode_reference(tok, f), i)


@pytest.mark.parametrize("sort", [False, True])
def test_batches_match_reference(tok: Tokenizer, frames: list[Frame], sort: bool) -> None:
    refs = [encode_reference(tok, f) for f in frames]
    rng = np.random.default_rng(1)
    for B in (1, 2, 7, 32, 64, 129, len(frames)):
        idx = rng.choice(len(frames), size=B, replace=B > len(frames))
        enc = tok.encode_batch([frames[i] for i in idx], sort=sort)
        assert len(enc) == B
        order = np.arange(B) if enc.order is None else enc.order
        if sort:  # stable sort by padded token bucket
            b = [tok.config.bucket(refs[idx[o]].n_tokens) for o in order]
            assert b == sorted(b)
            keys = [(tok.config.bucket(refs[idx[o]].n_tokens), int(o)) for o in order]
            assert keys == sorted(keys)
        else:
            assert enc.order is None
        for j, s in enumerate(enc.samples()):
            assert_same_sample(s, refs[idx[order[j]]], (B, j))
        for j, s in enumerate(enc.samples(copy=True)):
            assert_same_sample(s, refs[idx[order[j]]], (B, j, "copy"))
            assert s.raw.base is None or s.raw.base is not enc.raw


def test_sample_views_do_not_alias_the_payload(tok: Tokenizer, gen: FrameGen) -> None:
    payload = bytearray(random_frame(gen, np.random.default_rng(9), Kind.COMBAT))
    s = tok.encode_batch([parse_frame(payload)]).sample(0)
    before = [getattr(s, k).copy() for k in ARRAYS]
    payload[DATA_OFFSET:] = bytes(len(payload) - DATA_OFFSET)
    assert all(np.array_equal(b, getattr(s, k)) for b, k in zip(before, ARRAYS))


# --------------------------------------------------------------------------- fuzzed frames

# column -> value generator; ranges keep the *reference* free of exceptions (it raises on a few
# out-of-range Python-int assignments and on negative action seats), everything else is arbitrary.
I32 = (-(1 << 31), (1 << 31) - 1)
FUZZ: dict[int, list[tuple[int, Callable[[np.random.Generator, int], np.ndarray]]]] = {
    Table.SEATS: [(int(SeatCol.HP), lambda r, n: r.integers(*I32, n)), (int(SeatCol.GOLD), lambda r, n: r.integers(*I32, n)),
                  (int(SeatCol.CREATURE_ROW), lambda r, n: r.integers(-3, 70000, n)),
                  (int(SeatCol.CHARACTER_ID), lambda r, n: r.integers(*I32, n))],
    Table.CARDS: [(int(CardCol.ZONE), lambda r, n: r.integers(-2, 12, n)), (int(CardCol.AUX), lambda r, n: r.integers(*I32, n)),
                  (int(CardCol.CARD_ID), lambda r, n: r.integers(*I32, n)),
                  (int(CardCol.UPGRADE_LEVEL), lambda r, n: r.integers(-5, 9, n)),
                  (int(CardCol.LINK_ROW), lambda r, n: r.integers(-2, 40000, n)),
                  (int(CardCol.COUNT), lambda r, n: r.integers(*I32, n))],
    Table.CREATURES: [(int(CreatureCol.KIND), lambda r, n: r.integers(-3, 9, n)),
                      (int(CreatureCol.MODEL_ID), lambda r, n: r.integers(*I32, n)),
                      (int(CreatureCol.HP), lambda r, n: r.integers(-100, 100000, n)),
                      (int(CreatureCol.POISON_NEXT_TURN), lambda r, n: r.integers(-100, 100000, n))],
    Table.POWERS: [(int(PowerCol.APPLIER_ROW), lambda r, n: r.integers(-3, 9, n)),
                   (int(PowerCol.OWNER_ROW), lambda r, n: r.integers(-3, 9, n)),
                   (int(PowerCol.POWER_ID), lambda r, n: r.integers(*I32, n))],
    Table.MAP_NODES: [(int(MapNodeCol.ROW), lambda r, n: r.integers(*I32, n))],
    Table.MAP_EDGES: [(int(MapEdgeCol.FROM_NODE_ROW), lambda r, n: r.integers(-5, 70000, n)),
                      (int(MapEdgeCol.TO_NODE_ROW), lambda r, n: r.integers(-5, 70000, n))],
    Table.OPTIONS: [(int(OptionCol.KIND), lambda r, n: r.integers(-3, 12, n)),
                    (int(OptionCol.KEY_ID), lambda r, n: r.integers(*I32, n)),
                    (int(OptionCol.REF_TABLE), lambda r, n: r.integers(-3, 20, n)),
                    (int(OptionCol.REF_ROW), lambda r, n: r.integers(-3, 30, n))],
    Table.REFS: [(int(RefCol.CATEGORY), lambda r, n: r.integers(-3, 14, n)),
                 (int(RefCol.VOCAB_ID), lambda r, n: r.integers(-3, 1 << 20, n)),
                 (int(RefCol.ENT_TABLE), lambda r, n: r.integers(-3, 20, n)),
                 (int(RefCol.OWNER_TABLE), lambda r, n: r.integers(-3, 20, n))],
    Table.VARS: [(int(VarCol.TARGET_ROW), lambda r, n: r.integers(-2, 4, n)),
                 (int(VarCol.OWNER_ROW), lambda r, n: r.integers(-1, 6, n)),
                 (int(VarCol.SHOWN), lambda r, n: r.integers(*I32, n))],
    Table.ACTIONS: [(int(ActionCol.SEAT), lambda r, n: r.integers(0, 6, n)),
                    (int(ActionCol.SRC_TABLE), lambda r, n: r.integers(-2, 17, n)),
                    (int(ActionCol.SRC_ROW), lambda r, n: r.integers(-2, 8, n)),
                    (int(ActionCol.TGT_ROW), lambda r, n: r.integers(-2, 4, n))],
}
GLOBAL_FUZZ = [(int(G.KIND), (-5, 1 << 20)), (int(G.KIND_SUB), (-40, 40)), (int(G.PROMPT_SOURCE_CAT), (-3, 14)),
               (int(G.PROMPT_SOURCE_ID), (-5, 1 << 29)), (int(G.CUR_ROW), I32), (int(G.ASCENSION), (-32768, 32768)),
               (int(G.N_SEATS), (0, 7))]


def fuzz(payload: bytes, rng: np.random.Generator) -> Frame:
    f = parse_frame(payload)
    w = f.words.copy()
    g0 = GLOBAL_OFFSET // 4
    for col, (lo, hi) in GLOBAL_FUZZ:
        if rng.random() < 0.5:
            w[g0 + col] = rng.integers(lo, hi)
    for t, cols in FUZZ.items():
        n = f.rows[t]
        if n <= 0 or (t == Table.POWERS and f.rows[Table.CREATURES] == 0):
            continue
        o, nc = f.offsets[t], f.tables[t].shape[1]  # type: ignore[union-attr]
        for c, gen_vals in cols:
            if rng.random() < 0.6:
                w[o + c : o + n * nc : nc] = gen_vals(rng, n)
        if t == Table.ACTIONS and n > 1 and rng.random() < 0.5:  # duplicate (src, tgt) match keys
            i = int(rng.integers(0, n - 1))
            w[o + (i + 1) * nc : o + (i + 2) * nc] = w[o + i * nc : o + (i + 1) * nc]
    return parse_frame(w.tobytes())


def test_fuzzed_frames_match_reference(tok: Tokenizer, gen: FrameGen) -> None:
    """Columns overwritten with arbitrary values (saturation, int16 wrap, clipped lookups, unmatched or
    duplicated VARS -> ACTIONS keys, garbage references) - batch and reference still agree exactly."""
    rng = np.random.default_rng(7)
    payloads = corpus(gen, 3)
    frames = [fuzz(p, rng) for p in payloads for _ in range(3)]
    refs = [encode_reference(tok, f) for f in frames]
    enc = tok.encode_batch(frames)
    for j, s in enumerate(enc.samples()):
        assert_same_sample(s, refs[j], j)
    sat = np.concatenate([s.raw.ravel() for s in refs])
    assert (sat == 32767).any() and (sat == -32768).any()  # saturation was exercised


def test_resolved_elided_frame_matches_reference(tok: Tokenizer, gen: FrameGen) -> None:
    prev = parse_frame(random_frame(gen, np.random.default_rng(4), Kind.MAP))
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
    f = parse_frame(np.concatenate([h, prev.glob, d.ravel(), *data]).astype("<i4").tobytes())
    with pytest.raises(ValueError):
        tok.encode_batch([f])
    f.resolve_elided(prev)
    enc = tok.encode_batch([prev, f, prev])
    for s in enc.samples():
        assert_same_sample(s, encode_reference(tok, f))


# --------------------------------------------------------------------------- collation and grouping


def padded(s: BatchShape) -> BatchShape:  # what the CUDA-graph policy uses (all sizes grown)
    p2 = lambda n, lo: max(lo, 1 << max(0, n - 1).bit_length())  # noqa: E731
    return BatchShape(p2(s.batch, 8), s.tokens, p2(s.actions, 8), p2(s.kids, 64), p2(s.vars, 64),
                      p2(s.edges, 64), p2(s.rows, 256))


def test_collate_encoded_matches_sample_collate(tok: Tokenizer, frames: list[Frame]) -> None:
    rng = np.random.default_rng(2)
    reused: dict[BatchShape, dict[str, np.ndarray]] = {}
    for trial in range(12):
        B = int(rng.integers(1, 100))
        enc = tok.encode_batch([frames[i] for i in rng.choice(len(frames), size=B)], sort=True)
        samples = enc.samples()
        for lo, hi in group_ranges(tok.bucket_of(enc.n_tokens), int(rng.integers(1, 24)), int(rng.integers(4, 64))):
            nat = tok.shape_of(samples[lo:hi])
            assert tok.shape_encoded(enc, lo, hi) == nat
            assert_same_batch(tok.collate_encoded(enc, lo, hi), tok.collate(samples[lo:hi]), (trial, lo, hi))
            big = padded(nat)
            want = tok.collate(samples[lo:hi], shape=big)
            assert_same_batch(tok.collate_encoded(enc, lo, hi, shape=big), want, (trial, lo, hi, "padded"))
            # into recycled (dirty) buffers of a fixed shape
            key = BatchShape(big.batch, big.tokens, 64, 2048, 2048, 2048, 8192)
            if key.fits(nat):
                buf = reused.setdefault(key, {k: np.full(v.shape, 77, v.numpy().dtype) if v.dtype != torch.bool
                                              else np.ones(v.shape, np.bool_)
                                              for k, v in tok.collate(samples[lo:hi], shape=key).items()})
                tok.collate_encoded_into(enc, lo, hi, key, buf)
                assert_same_batch({k: torch.from_numpy(v) for k, v in buf.items()},
                                  tok.collate(samples[lo:hi], shape=key), (trial, lo, hi, "into"))
        with pytest.raises(ValueError):
            tok.collate_encoded(enc, 0, len(enc), shape=BatchShape(1, 32, 8, 64, 64, 64, 256)) if len(enc) > 1 \
                else tok.collate_encoded(enc, 0, 0)


def test_group_ranges_reproduce_unsorted_grouping(vocab: Vocab, frames: list[Frame]) -> None:
    """Sorted contiguous groups == the reference grouping of the unsorted frames (same members, same
    order, same group order), for many batch / min_group / max_batch combinations."""
    from spire.actor import Actor

    tok = Tokenizer(vocab)
    rng = np.random.default_rng(5)
    for _ in range(60):
        B = int(rng.integers(1, 90))
        idx = rng.choice(len(frames), size=B)
        fr = [frames[i] for i in idx]
        cfg = ActorConfig(min_group=int(rng.integers(1, 40)), max_batch=int(rng.integers(1, 50)))
        fake = Actor.__new__(Actor)
        fake.tok, fake.cfg = tok, cfg
        old = Actor._groups(fake, [tok.encode(f) for f in fr])
        enc = tok.encode_batch(fr, sort=True)
        order = np.arange(B) if enc.order is None else enc.order
        new = [order[lo:hi].tolist() for lo, hi in group_ranges(tok.bucket_of(enc.n_tokens), cfg.min_group,
                                                                cfg.max_batch)]
        assert new == old


def test_action_column(tok: Tokenizer, frames: list[Frame]) -> None:
    obs = [f for f in frames if f.is_obs]
    enc = tok.encode_batch(obs, sort=True)
    order = np.arange(len(obs)) if enc.order is None else enc.order
    rng = np.random.default_rng(0)
    rows = np.array([rng.integers(0, obs[o].n_actions) for o in order])
    for col in range(int(ActionCol.VERB), int(ActionCol.TGT_ROW) + 1):
        want = [int(obs[o].actions[r, col]) for o, r in zip(order, rows)]
        assert enc.action_column(col, rows).tolist() == want
    assert enc.action_column(0, np.full(len(obs), 10_000)).tolist() == [0] * len(obs)  # out of range -> 0


# --------------------------------------------------------------------------- sample blocks


def test_sample_block_and_segment_blob(tok: Tokenizer, frames: list[Frame]) -> None:
    enc = tok.encode_batch(frames[:40], sort=True)
    views = enc.samples()
    blk = SampleBlock.from_samples(views)
    assert len(blk) == len(views)
    for a, b in zip(blk, views):
        assert_same_sample(a, b)
    assert_same_sample(blk[-1], views[-1])
    assert [s.counts for s in blk[3:7]] == [s.counts for s in views[3:7]]

    def seg(samples: Sequence[Any], boot: Sample | None) -> Segment:
        n = len(samples)
        return Segment(meta=EpisodeMeta(1, 2, (3,), 0), start=0, samples=samples,
                       actions=np.arange(n, dtype=np.int64), logp=np.zeros(n, np.float32),
                       values=np.ones(n, np.float32), kinds=np.zeros(n, np.int8), n_legal=np.ones(n, np.int16),
                       policy_version=np.zeros(n, np.int64), bootstrap_sample=boot)

    for boot in (None, views[5]):
        blob_list = ipc.pack_segment(seg(list(views), boot))
        blob_blk = ipc.pack_segment(seg(blk, boot))
        assert blob_list == blob_blk
        back = ipc.decode_segment(blob_blk)
        for x, y in zip(back.samples, views):
            assert_same_sample(x, y)


def test_sample_block_native_concat_matches_numpy(tok: Tokenizer, frames: list[Frame],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """SampleBlock.from_samples with the C concat kernel == the np.concatenate path, for views of several
    batches (as in a rollout segment), copies, frames without edges, and a non-contiguous input (falls back)."""
    from spire import tokenize as tkz
    if getattr(tkz._native, "concat_into", None) is None:
        pytest.skip("spire._native is not built (python -B native/build.py)")
    rng = np.random.default_rng(3)
    groups = [tok.encode_batch([frames[int(i)] for i in rng.choice(len(frames), 23)], sort=True).samples()
              for _ in range(6)]
    mixed = [g[int(rng.integers(len(g)))] for g in groups for _ in range(5)] + groups[0][:3]
    strided = Sample(groups[1][0].types, groups[1][0].raw[:, ::-1], groups[1][0].rel, groups[1][0].edges,
                     groups[1][0].cond, groups[1][0].counts)
    for samples in (mixed, [s.copy() if hasattr(s, "copy") else s for s in mixed[:7]], groups[2], mixed + [strided]):
        fast = SampleBlock.from_samples(samples)
        with monkeypatch.context() as m:
            m.setattr(tkz._native, "concat_into", None, raising=False)
            slow = SampleBlock.from_samples(samples)
        for k in ARRAYS:
            x, y = getattr(fast, k), getattr(slow, k)
            assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y), k
        assert np.array_equal(fast.counts, slow.counts) and fast._edges == slow._edges
        for a, b in zip(fast, samples):
            assert_same_sample(a, b)
