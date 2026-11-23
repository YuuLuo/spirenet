"""alt/speed e2e: the patch's switches ON (learner compile_model + bag_atomic, TorchPolicy compile_model) against the
production path, on the tiny model and fake-host segments of tests/test_learner_fast.py."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy, UniformPolicy
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def tok(vocab: Vocab) -> Tokenizer:
    return Tokenizer(vocab, TokenizerConfig(token_buckets=(20, 28, 40, 64, 80, 96, 128, 192, 256, 384)))


@pytest.fixture(scope="module")
def segments(vocab: Vocab, tok: Tokenizer) -> list[Segment]:
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=64))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False)) for i in range(12)]
    actor = Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
                  ActorConfig(max_batch=32), store=store)
    actor.run(max_decisions=3000)
    return segs


def _params(lr: Learner) -> list[torch.Tensor]:
    return [p.detach().float().cpu().clone() for p in lr.model.parameters()]


@cuda
def test_compiled_atomic_learner_matches_the_graph_path(vocab: Vocab, tok: Tokenizer, segments: list[Segment]) -> None:
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    cfg = PPOConfig(epochs=2, minibatch=512, micro_batch=256, micro_tokens=4096, lr=1e-4, seed=3, kl_lr_target=.01)
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    ls = {}
    for name, on in (("graphs", False), ("compiled", True)):
        net = copy.deepcopy(base)
        net.bag_atomic = on
        ls[name] = Learner(net, tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens), fast=True,
                           graphs=True, compile_model=on)
    p0 = _params(ls["graphs"])
    ms = {k: lr.update(segments) for k, lr in ls.items()}
    assert ms["compiled"]["stepper"]["captures"] == ms["graphs"]["stepper"]["captures"] > 0
    d_ref = [a - b for a, b in zip(_params(ls["graphs"]), p0)]
    d = [a - b for a, b in zip(_params(ls["compiled"]), p0)]
    for x, r in zip(d, d_ref):
        assert (x - r).abs().max() <= 2.5 * cfg.lr * cfg.epochs * 2
    assert sum(float((x - r).abs().sum()) for x, r in zip(d, d_ref)) / sum(float(r.abs().sum()) for r in d_ref) < 0.05
    for key in ("loss_pi", "loss_v", "loss_win", "loss_hp", "entropy", "approx_kl", "grad_norm", "kl_lr_scale_mean"):
        assert ms["compiled"][key] == pytest.approx(ms["graphs"][key], rel=0.05, abs=2e-3), key
    for key in ("samples", "minibatches", "nonfinite_skipped", "illegal_chosen"):
        assert ms["compiled"][key] == ms["graphs"][key], key
    # checkpoints hold the plain module's keys (no torch.compile wrapper prefix)
    st = ls["compiled"].state()
    assert set(st["model"]) == set(ls["graphs"].state()["model"])
    assert not any("_orig_mod" in k for k in st["model"])
    # later updates: new captures only for new shapes, as on the graph path
    for _ in range(2):
        for lr in ls.values():
            lr.update(segments)
    assert ls["compiled"].stepper.captures == ls["graphs"].stepper.captures


@cuda
def test_compiled_policy_matches_the_graph_policy(vocab: Vocab, tok: Tokenizer, segments: list[Segment]) -> None:
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    torch.manual_seed(1)
    net = SpireNet(tok.layout, TINY).cuda()
    pols = {k: TorchPolicy(copy.deepcopy(net), tok, mode="greedy", compile_model=k == "compiled")
            for k in ("graphs", "compiled")}
    samples = [s for seg in segments for s in seg.samples][:600]
    rng = np.random.default_rng(0)
    lo = 0
    while lo < len(samples):
        hi = min(len(samples), lo + int(rng.integers(1, 64)))
        outs = {}
        for k, p in pols.items():
            h = p.launch(samples[lo:hi], np.zeros(hi - lo, np.int64))
            h.event.synchronize()
            outs[k] = h.out.numpy().copy()
            p.finish(h)
        a, b = outs["graphs"], outs["compiled"]
        assert np.array_equal(np.isfinite(a), np.isfinite(b))
        f = np.isfinite(a)
        assert np.abs(a[f] - b[f]).max() <= 0.07
        lo = hi
    assert pols["compiled"].runner.captures == pols["graphs"].runner.captures
