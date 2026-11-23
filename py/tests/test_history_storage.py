"""Frozen actor history survives every sample/IPC/collation path, including bootstraps."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spire.ipc import decode_segment, pack_segment
from spire.model import packed_layout, packed_views
from spire.proto import parse_frame
from spire.rollout import EpisodeMeta, Segment
from spire.schema import Kind
from spire.synth import FrameGen, random_frame
from spire.tokenize import SampleBlock, Tokenizer, TokenizerConfig
from spire.vocab import Vocab


@pytest.mark.parametrize("native", [False, True])
def test_history_all_storage_and_collation_paths(gen: FrameGen, vocab: Vocab, native: bool) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True), native=native)
    rng = np.random.default_rng(31)
    frames = [parse_frame(random_frame(gen, rng, k)) for k in (Kind.COMBAT, Kind.REWARD, Kind.SHOP)]
    enc = tok.encode_batch(frames, sort=True)
    enc.history[:] = np.arange(enc.history.size, dtype=np.int16).reshape(enc.history.shape)
    samples = enc.samples(copy=True)
    for i, sample in enumerate(samples):
        assert np.array_equal(sample.history, enc.sample(i).history)
        assert not np.shares_memory(sample.history, enc.history)
    blk = SampleBlock.from_samples(samples)
    joined = SampleBlock.concat([blk, blk])
    assert np.array_equal(joined[4].history, samples[1].history)
    z = np.zeros(3)
    seg = Segment(EpisodeMeta(0, 1, (2,), 0), 0, blk, z.astype(np.int64), z.astype(np.float32),
                  z.astype(np.float32), z.astype(np.int8), np.ones(3, np.int16), z.astype(np.int64),
                  bootstrap_sample=samples[2])
    restored = decode_segment(pack_segment(seg))
    assert restored.bootstrap_sample is not None
    assert np.array_equal(restored.bootstrap_sample.history, samples[2].history)
    for a, b in zip(samples, restored.samples):
        assert np.array_equal(a.history, b.history)
    expected = tok.collate(samples)
    encoded = tok.collate_encoded(enc)
    for key in expected:
        assert torch.equal(expected[key], encoded[key]), key
    idx = np.array([2, 0, 1, 1], np.int64)
    shape = tok.shape_block(joined, idx)
    layout, nbytes = packed_layout(shape, tok.width)
    packed = packed_views(torch.empty(nbytes, dtype=torch.uint8), layout)
    tok.collate_block_into(joined, idx, shape, {k: v.numpy() for k, v in packed.items()})
    eager = tok.collate([joined[int(i)] for i in idx], shape=shape)
    for key in eager:
        assert torch.equal(eager[key], packed[key]), key
