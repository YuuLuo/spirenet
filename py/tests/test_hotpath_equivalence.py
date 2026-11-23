"""The rewritten actor hot path decides exactly like the old one.

Old path: per-frame reference encoder (spire.tokenize_ref) -> reference bucket grouping of the
unsorted samples (Actor._groups) -> per-sample collate -> TorchPolicy.launch.
New path: Tokenizer.encode_batch(sort=True) -> group_ranges -> TorchPolicy.launch_encoded (collated
into packed pinned staging; one host->device copy per replay).

Same network, same policy seed: identical actions, behaviour log-probs, values and entropies
(CPU eager: bit-identical; CUDA graphs: identical batch tensors, logits within the bf16 tolerance of
the existing graph-vs-eager test and identical sampled actions).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, PolicyOut, TorchPolicy, group_ranges
from spire.model import ModelConfig, SpireNet
from spire.proto import Frame, parse_frame
from spire.schema import DECISION_KINDS, Kind
from spire.synth import FrameGen, random_frame
from spire.tokenize import Tokenizer
from spire.tokenize_ref import encode_reference
from spire.vocab import Catalog, Vocab

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)


def batch_frames(gen: FrameGen, seed: int, n: int) -> list[Frame]:
    rng = np.random.default_rng(seed)
    kinds = [Kind.COMBAT] * 8 + list(DECISION_KINDS)
    return [parse_frame(random_frame(gen, rng, kinds[int(rng.integers(len(kinds)))], n_seats=int(rng.integers(1, 5))))
            for _ in range(n)]


def old_path(pol: TorchPolicy, tok: Tokenizer, frames: list[Frame], cfg: ActorConfig) -> tuple[np.ndarray, ...]:
    samples = [encode_reference(tok, f) for f in frames]
    fake = Actor.__new__(Actor)
    fake.tok, fake.cfg = tok, cfg
    n = len(frames)
    out: list[np.ndarray] = [np.zeros(n, np.int64), np.zeros(n, np.float32), np.zeros(n, np.float32),
                             np.zeros(n, np.float32)]
    n_act = np.array([f.n_actions for f in frames])
    for g in Actor._groups(fake, samples):
        o: PolicyOut = pol.finish(pol.launch([samples[i] for i in g], n_act[g]))
        for dst, src in zip(out, (o.actions, o.logp, o.value, o.entropy)):
            dst[np.asarray(g)] = src
    return tuple(out)


def new_path(pol: TorchPolicy, tok: Tokenizer, frames: list[Frame], cfg: ActorConfig) -> tuple[np.ndarray, ...]:
    enc = tok.encode_batch(frames, sort=True)
    order = np.arange(len(frames)) if enc.order is None else enc.order
    n = len(frames)
    out: list[np.ndarray] = [np.zeros(n, np.int64), np.zeros(n, np.float32), np.zeros(n, np.float32),
                             np.zeros(n, np.float32)]
    n_act = np.array([frames[i].n_actions for i in order])
    for lo, hi in group_ranges(tok.bucket_of(enc.n_tokens), cfg.min_group, cfg.max_batch):
        o = pol.finish(pol.launch_encoded(enc, lo, hi, n_act[lo:hi]))
        for dst, src in zip(out, (o.actions, o.logp, o.value, o.entropy)):
            dst[order[lo:hi]] = src
    return tuple(out)


def run_both(net: SpireNet, tok: Tokenizer, gen: FrameGen, use_graphs: bool) -> list[tuple[Any, Any]]:
    cfg = ActorConfig(max_batch=24, min_group=6)
    old = TorchPolicy(net, tok, use_graphs=use_graphs, seed=123)
    new = TorchPolicy(net, tok, use_graphs=use_graphs, seed=123)
    res = []
    for b, n in enumerate((1, 5, 40, 77, 120)):
        frames = batch_frames(gen, b, n)
        res.append((old_path(old, tok, frames, cfg), new_path(new, tok, frames, cfg)))
    return res


@pytest.mark.parametrize("native", [True, False], ids=["native", "numpy"])
def test_cpu_decisions_bit_identical(vocab: Vocab, catalog: Catalog, gen: FrameGen, native: bool) -> None:
    torch.manual_seed(0)
    tok = Tokenizer(vocab, native=native)
    net = SpireNet(tok.layout, TINY, catalog)
    for old, new in run_both(net, tok, gen, use_graphs=False):
        for a, b in zip(old, new):
            assert a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_decisions_match(vocab: Vocab, catalog: Catalog, gen: FrameGen) -> None:
    torch.manual_seed(0)
    tok = Tokenizer(vocab)
    net = SpireNet(tok.layout, TINY, catalog).cuda()
    for old, new in run_both(net, tok, gen, use_graphs=True):
        a_old, lp_old, v_old, e_old = old
        a_new, lp_new, v_new, e_new = new
        assert np.array_equal(a_old, a_new)  # same Gumbel noise, same logits up to bf16 noise
        np.testing.assert_allclose(lp_new, lp_old, atol=5e-3, rtol=0)
        np.testing.assert_allclose(v_new, v_old, atol=5e-3, rtol=0)
        np.testing.assert_allclose(e_new, e_old, atol=5e-3, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_pool_survives_replacing_its_only_graph(vocab: Vocab, catalog: Catalog, gen: FrameGen) -> None:
    """Regression: dropping the only graph of a runner's private pool, then capturing again, used to
    trip a CUDACachingAllocator internal assert (TorchPolicy replaces a graph whenever its shape grows)."""
    from spire.model import GraphRunner

    tok = Tokenizer(vocab)
    net = SpireNet(tok.layout, TINY, catalog).cuda().eval()
    runner = GraphRunner(net, tok)
    for n in (3, 9, 30):
        frames = batch_frames(gen, n, n)
        enc = tok.encode_batch(frames)
        shape = runner.shape_for_encoded(enc)
        runner.graphs.clear()  # the only graph goes away
        out = runner.run(tok.collate_encoded(enc, shape=shape), shape, n)
        assert out["logits"].shape[0] == n
