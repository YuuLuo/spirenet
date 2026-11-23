"""The learner's fast path (spire.trainstep): C collation from sample blocks, sync-free micro-steps, CUDA graphs.

* collation of an arbitrary sample subset of a concatenated block is bit-identical to Tokenizer.collate
  (C kernel and fallback), with padding shapes;
* decoded IPC segments carry SampleBlocks (the fast path's input) and their decision timestamps;
* one PPO update through the generic eager path, the fast eager path and the fast CUDA-graph path, from the same
  weights on the same segments, lands on the same parameters (bf16 / summation-order tolerance) with the same
  diagnostics; non-finite minibatches are skipped on the device; KL early stop works; graphs stop being
  captured once the shapes are known.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.ipc import decode_segment, pack_segment
from spire.learner import Learner, PPOConfig, policy_lag_seconds
from spire.model import BATCH_FIELDS, ModelConfig, SpireNet, batch_field_shapes, state_to_cpu, PinnedState
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import BatchShape, SampleBlock, Tokenizer, TokenizerConfig
from spire.trainstep import bucket_micros, pad_batch, round_dim
from spire.vocab import Vocab

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def tok(vocab: Vocab) -> Tokenizer:
    return Tokenizer(vocab, TokenizerConfig(token_buckets=(20, 28, 40, 64, 80, 96, 128, 192, 256, 384)))


@pytest.fixture(scope="module")
def segments(vocab: Vocab, tok: Tokenizer) -> list[Segment]:
    """Real training segments of in-process fake hosts (samples compacted into SampleBlocks by RolloutStore)."""
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=64))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False)) for i in range(12)]
    actor = Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
                  ActorConfig(max_batch=32), store=store)
    actor.run(max_decisions=3000)
    assert sum(len(s) for s in segs) > 1500
    return segs


def _empty(shape: BatchShape, width: int) -> dict[str, np.ndarray]:
    shapes = batch_field_shapes(shape, width)
    return {k: np.full(shapes[k], 77, torch.empty((), dtype=dt).numpy().dtype) for k, dt in BATCH_FIELDS}


@pytest.mark.parametrize("native", [True, False])
def test_block_collation_matches_collate(vocab: Vocab, segments: list[Segment], native: bool) -> None:
    tk = Tokenizer(vocab, TokenizerConfig(token_buckets=(20, 28, 40, 64, 80, 96, 128, 192, 256, 384)), native=native)
    blocks = [s.samples for s in segments]
    assert all(isinstance(b, SampleBlock) for b in blocks)
    blk = SampleBlock.concat(blocks)  # type: ignore[arg-type]
    flat = [x for b in blocks for x in b]
    assert len(blk) == len(flat)
    for i in (0, 7, len(flat) - 1):  # concat keeps every sample
        a, b = blk[i], flat[i]
        assert a.counts == b.counts and all(np.array_equal(getattr(a, f), getattr(b, f))
                                            for f in ("types", "raw", "rel", "edges", "cond"))
    rng = np.random.default_rng(1)
    for trial in range(25):
        idx = rng.choice(len(blk), size=int(rng.integers(1, 120)), replace=False)
        nat = tk.shape_block(blk, idx)
        assert nat == tk.shape_of([blk[int(i)] for i in idx])
        shape = nat if trial % 3 == 0 else BatchShape(nat.batch + int(rng.integers(0, 9)), nat.tokens + 8,
                                                      nat.actions + 8, nat.kids + 64, nat.vars + 64, nat.edges + 64,
                                                      nat.rows + 300)
        ref = tk.collate([blk[int(i)] for i in idx], shape=shape)
        out = _empty(shape, tk.width)
        tk.collate_block_into(blk, idx, shape, out)
        for k, v in ref.items():
            assert np.array_equal(v.numpy(), out[k]), (trial, k)


def test_decoded_segments_are_blocks_with_times(segments: list[Segment]) -> None:
    seg = segments[3]
    assert seg.t_first > 0 and seg.t_last >= seg.t_first
    d = decode_segment(pack_segment(seg))
    assert isinstance(d.samples, SampleBlock) and len(d.samples) == len(seg)
    assert (d.t_first, d.t_last) == (seg.t_first, seg.t_last)
    for a, b in zip(d.samples, seg.samples):
        assert a.counts == b.counts and np.array_equal(a.raw, b.raw) and np.array_equal(a.rel, b.rel)
    if seg.bootstrap_sample is not None:
        assert d.bootstrap_sample is not None and np.array_equal(d.bootstrap_sample.raw, seg.bootstrap_sample.raw)
    lag = policy_lag_seconds(segments, time.time())
    assert lag is not None and 0 <= lag["p50"] <= lag["max"]


def test_micro_batching_and_padding_rules() -> None:
    cfg = PPOConfig(micro_batch=1000, micro_tokens=4096)
    buckets = np.array([20] * 700 + [64] * 90 + [384] * 3)
    mb = np.random.default_rng(0).permutation(len(buckets))
    micros = bucket_micros(mb, buckets, cfg)
    assert sorted(np.concatenate([i for _, i in micros]).tolist()) == sorted(mb.tolist())  # a partition
    assert [t for t, _ in micros] == sorted((t for t, _ in micros), reverse=True)
    for t, idx in micros:
        assert (buckets[idx] == t).all() and len(idx) <= min(1000, 4096 // t)
    by20 = [len(i) for t, i in micros if t == 20]
    assert len(by20) == 4 and max(by20) - min(by20) <= 1  # 700 over cap 204: 4 equal chunks
    for b in (1, 8, 9, 15, 17, 100, 129, 700, 3000):
        p = pad_batch(b, 10_000)
        assert p >= b and (p == 8 or p < 1.125 * b)
    assert round_dim(5, 8) == 8 and round_dim(1000, 64) == 1024 and round_dim(1100, 64) == 1280


def _learners(vocab: Vocab, tok: Tokenizer, cfg: PPOConfig) -> dict[str, Learner]:
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    out = {}
    for name, fast, graphs in (("generic", False, False), ("fast", True, False), ("graphs", True, True)):
        out[name] = Learner(copy.deepcopy(base), tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                            fast=fast, graphs=graphs)
    return out


def _params(lr: Learner) -> list[torch.Tensor]:
    return [p.detach().float().cpu().clone() for p in lr.model.parameters()]


@cuda
@pytest.mark.parametrize("kl_lr_target", [None, .01])
def test_fast_and_graph_updates_match_the_generic_path(vocab: Vocab, tok: Tokenizer, segments: list[Segment],
                                                       kl_lr_target: float | None) -> None:
    cfg = PPOConfig(epochs=2, minibatch=512, micro_batch=256, micro_tokens=4096, lr=1e-4, seed=3,
                    kl_lr_target=kl_lr_target)
    ls = _learners(vocab, tok, cfg)
    p0 = _params(ls["generic"])
    ms = {k: lr.update(segments) for k, lr in ls.items()}
    assert not ms["generic"]["fast_path"] and ms["fast"]["fast_path"] and ms["graphs"]["fast_path"]
    assert ms["graphs"]["stepper"]["captures"] > 0 and ms["fast"]["stepper"]["captures"] == 0
    deltas = {k: [a - b for a, b in zip(_params(lr), p0)] for k, lr in ls.items()}
    for k in ("fast", "graphs"):
        for d_ref, d in zip(deltas["generic"], deltas[k]):
            # Adam's first steps move every weight by ~lr; bf16 / summation order may flip near-zero gradients
            assert (d - d_ref).abs().max() <= 2.5 * cfg.lr * cfg.epochs * 2, k
        num = sum(float((d - r).abs().sum()) for d, r in zip(deltas[k], deltas["generic"]))
        den = sum(float(r.abs().sum()) for r in deltas["generic"])
        assert num / den < 0.05, (k, num / den)
        for key in ("loss_pi", "loss_v", "loss_win", "loss_hp", "entropy", "approx_kl", "clipfrac_epoch0",
                    "ratio_dev_epoch0", "grad_norm", "first_minibatch_kl", "kl_lr_scale_mean", "kl_lr_scale_min"):
            assert ms[k][key] == pytest.approx(ms["generic"][key], rel=0.05, abs=2e-3), (k, key)
        for key in ("samples", "minibatches", "nonfinite_skipped", "illegal_chosen", "staleness_mean",
                    "explained_variance"):
            assert ms[k][key] == ms["generic"][key], (k, key)
        assert set(ms[k]["entropy_by_kind"]) == set(ms["generic"]["entropy_by_kind"])
    for metrics in ms.values():
        assert 0 < metrics["kl_lr_scale_min"] <= metrics["kl_lr_scale_mean"] <= 1
        if kl_lr_target is None:
            assert metrics["kl_lr_scale_mean"] == metrics["kl_lr_scale_min"] == 1
        elif metrics["approx_kl"] > kl_lr_target:
            # A weighted mean above target proves at least one minibatch required damping.
            assert metrics["kl_lr_scale_min"] < 1 and metrics["kl_lr_scale_mean"] < 1
    # fast eager vs fast graphs: same micro-batches and shapes, only the launch mechanism differs
    for a, b in zip(deltas["fast"], deltas["graphs"]):
        assert (a - b).abs().max() <= 2 * cfg.lr
    assert ls["graphs"].ema_steps == ls["generic"].ema_steps == ms["generic"]["minibatches"] * cfg.epochs
    for a, b in zip(ls["graphs"].ema.parameters(), ls["generic"].ema.parameters()):
        assert (a - b).abs().max() <= 2.5 * cfg.lr * cfg.epochs * 2
    # later updates reuse the graphs: new captures only for (batch, bucket) sizes not seen yet, ever fewer
    st = ls["graphs"].stepper
    assert st is not None
    new = [st.captures]
    for _ in range(3):
        c = st.captures
        ls["graphs"].update(segments)
        new.append(st.captures - c)
    assert new[-1] < new[0] and sum(new[1:]) < 2 * new[0], new


class _NaNValue(torch.nn.Module):
    def __init__(self, inner: SpireNet) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, batch: dict) -> dict:
        out = self.inner(batch)
        out["value"] = out["value"] * float("nan")
        return out


@cuda
@pytest.mark.parametrize("graphs", [False, True])
def test_fast_path_skips_nonfinite_minibatches_on_the_device(vocab: Vocab, tok: Tokenizer, segments: list[Segment],
                                                            graphs: bool) -> None:
    torch.manual_seed(0)
    cfg = PPOConfig(epochs=1, minibatch=512, micro_batch=256, micro_tokens=4096, seed=1)
    lr = Learner(_NaNValue(SpireNet(tok.layout, TINY)), tok, cfg, device="cuda", fast=True, graphs=graphs)
    before = _params(lr)
    ema0 = [p.detach().clone() for p in lr.ema.parameters()]
    m = lr.update(segments)
    assert m["fast_path"] and m["nonfinite_skipped"] == m["minibatches"] > 0
    assert all(torch.equal(a, b) for a, b in zip(before, _params(lr)))
    assert all(torch.equal(a, b) for a, b in zip(ema0, lr.ema.parameters()))
    grads = [p.grad for p in lr.model.parameters() if p.requires_grad]
    assert all(g is not None and float(g.abs().sum()) == 0.0 for g in grads)  # re-zeroed in place after every step


@cuda
def test_fast_path_kl_early_stop(vocab: Vocab, tok: Tokenizer, segments: list[Segment]) -> None:
    torch.manual_seed(0)
    cfg = PPOConfig(epochs=3, minibatch=256, micro_batch=256, micro_tokens=4096, target_kl=1e-12, lr=1e-3, seed=2)
    lr = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cuda", fast=True, graphs=True)
    m = lr.update(segments)
    assert m["fast_path"] and m["early_stop"] and m["epochs_done"] < 3.0


@cuda
def test_weight_transfer_helpers(vocab: Vocab, tok: Tokenizer) -> None:
    torch.manual_seed(0)
    a = SpireNet(tok.layout, TINY).cuda()
    b = SpireNet(tok.layout, TINY).cuda()
    sd = state_to_cpu(a)
    assert list(sd) == list(a.state_dict()) and all(v.device.type == "cpu" for v in sd.values())
    assert all(torch.equal(sd[k], v.cpu()) for k, v in a.state_dict().items())
    ptrs = [p.data_ptr() for p in b.parameters()]
    PinnedState(sd).load_into(b)
    torch.cuda.synchronize()
    assert [p.data_ptr() for p in b.parameters()] == ptrs  # in place: captured graphs stay valid
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    with pytest.raises(KeyError):
        PinnedState({k: v for i, (k, v) in enumerate(sd.items()) if i}).load_into(b)
