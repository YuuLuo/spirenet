"""KL damping acts before Adam, retains data, and uses the captured body's KL statistic."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from ppo_stubs import StubSample, StubTokenizer
from spire.learner import Learner, PPOConfig, _Accum
from spire.rollout import EpisodeMeta, GAEConfig, Segment, SegmentEnd, build_batch
from spire.schema import Kind
from spire.tokenize import BatchShape, SampleBlock
from spire.train import _learner_fields
from spire.trainstep import DEN_COLS, MicroStepper, _Res


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([math.log(9.0), 0.0]))
        self.value = nn.Parameter(torch.tensor(0.25))

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        n = len(batch["x"])
        v = self.value.expand(n)
        return {"logits": self.logits.expand(n, 2), "value": v, "pwin_logit": v,
                "floors_pred": v, "hp_pred": v}


def _cfg(*, kl_lr_target: float | None = None, target_kl: float | None = None,
         decoupled_clip_anchor: bool = False, micro_batch: int = 1) -> PPOConfig:
    return PPOConfig(epochs=1, minibatch=2, micro_batch=micro_batch, adv_norm=False, ent_coef=0.01,
                    vf_coef=0.5, aux_win=0.0, aux_floors=0.0, aux_hp=0.0, bf16=False,
                    gae=GAEConfig(gamma=1.0, lam=0.99), kl_lr_target=kl_lr_target,
                    target_kl=target_kl, decoupled_clip_anchor=decoupled_clip_anchor)


def _segments(tok: StubTokenizer, behavior_p: float = 0.1) -> list[Segment]:
    return [Segment(EpisodeMeta(0, uid, (2,), 0), 0,
                    [StubSample(np.zeros(tok.n_features, np.float32), 2)],
                    np.array([0], np.int64), np.array([math.log(behavior_p)], np.float32),
                    np.array([0.0], np.float32), np.array([Kind.REST], np.int8), np.array([2], np.int16),
                    np.array([0], np.int64), end=SegmentEnd.TERMINAL,
                    rewards=np.array([-1.0], np.float32), combat_id=np.array([-1], np.int32),
                    win=np.array([0.0], np.float32), win_mask=np.array([True]),
                    floors_left=np.array([0.0], np.float32), hp_post=np.array([0.0], np.float32),
                    hp_mask=np.array([False])) for uid in range(8)]


@pytest.mark.parametrize("target", [0.0, -0.1, math.inf, math.nan])
def test_invalid_target_fails(target: float) -> None:
    with pytest.raises(AssertionError, match="finite and positive"):
        PPOConfig(kl_lr_target=target)


@pytest.mark.parametrize("kl, expected", [(0.0, 1.0), (.005, 1.0), (.01, 1.0), (.04, .25), (1., .01)])
def test_threshold_and_inverse_scale(kl: float, expected: float) -> None:
    learner = Learner(_Net(), StubTokenizer(), _cfg(kl_lr_target=.01), device="cpu")
    acc = _Accum(torch.device("cpu"))
    learner._minibatch_lr(torch.tensor(kl), acc)
    assert float(learner.opt.param_groups[0]["lr"]) == pytest.approx(learner.cfg.lr * expected)
    result = acc.result()
    assert result["kl_lr_scale_mean"] == pytest.approx(expected)
    assert result["kl_lr_scale_min"] == pytest.approx(expected)
    assert result["first_minibatch_kl"] == pytest.approx(kl)
    acc.reset()
    assert acc.result()["kl_lr_scale_min"] == 1.0


def test_brake_scales_adam_displacement_including_existing_momentum() -> None:
    tok = StubTokenizer()
    plain = Learner(_Net(), tok, _cfg(), device="cpu")
    for _ in range(3):
        for p in plain.model.parameters():
            p.grad = torch.ones_like(p)
        plain.opt.step()
    braked = Learner(_Net(), tok, _cfg(kl_lr_target=.01), device="cpu")
    braked.model.load_state_dict(plain.model.state_dict())
    braked.opt.load_state_dict(copy.deepcopy(plain.opt.state_dict()))
    before = [p.detach().clone() for p in plain.model.parameters()]
    for learner in (plain, braked):
        for p in learner.model.parameters():
            p.grad = torch.zeros_like(p)  # displacement comes entirely from existing Adam momentum
        learner._minibatch_lr(torch.tensor(.04), _Accum(torch.device("cpu")))
        learner.opt.step()
    for start, p, q in zip(before, plain.model.parameters(), braked.model.parameters()):
        assert torch.count_nonzero(p - start) > 0
        torch.testing.assert_close(q - start, .25 * (p - start), rtol=2e-3, atol=1e-7)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(plain.opt.state[p][key], braked.opt.state[q][key], rtol=0, atol=0)


def test_brake_never_releases_within_an_iteration() -> None:
    learner = Learner(_Net(), StubTokenizer(), _cfg(kl_lr_target=.01), device="cpu")
    acc = _Accum(torch.device("cpu"))
    for kl, expected in ((.04, .25), (.001, .25), (.08, .125), (0., .125)):
        learner._minibatch_lr(torch.tensor(kl), acc)
        acc.minibatches += 1
        assert float(learner.opt.param_groups[0]["lr"]) == pytest.approx(expected * learner.cfg.lr)
        assert acc.result()["kl_lr_scale_min"] == pytest.approx(expected)
    assert acc.result()["kl_lr_scale_mean"] == pytest.approx(.1875)
    assert acc.result()["first_minibatch_kl"] == pytest.approx(.04)
    acc.reset()  # the captured path reuses and resets its accumulator each iteration
    learner._minibatch_lr(torch.tensor(.001), acc)
    assert float(learner.opt.param_groups[0]["lr"]) == pytest.approx(learner.cfg.lr)
    assert acc.result()["kl_lr_scale_mean"] == acc.result()["kl_lr_scale_min"] == 1.


def test_nan_kl_keeps_the_retained_scale_and_infinite_kl_stops_the_iteration() -> None:
    # a NaN KL comes with a non-finite gradient (that step is skipped by the optimizer); the minibatches
    # after it must still get a finite rate. An infinite KL is "infinitely far": rate 0 from then on.
    learner = Learner(_Net(), StubTokenizer(), _cfg(kl_lr_target=.01), device="cpu")
    acc = _Accum(torch.device("cpu"))
    for kl, expected in ((.02, .5), (math.nan, .5), (.001, .5), (math.inf, 0.), (.001, 0.)):
        learner._minibatch_lr(torch.tensor(kl), acc)
        acc.minibatches += 1
        rate = float(learner.opt.param_groups[0]["lr"])
        assert math.isfinite(rate) and rate == pytest.approx(expected * learner.cfg.lr)
    assert acc.result()["kl_lr_scale_min"] == 0.


def test_new_iteration_resets_damping_for_fresh_samples() -> None:
    tok = StubTokenizer()
    net = _Net()
    learner = Learner(net, tok, _cfg(kl_lr_target=.01), device="cpu")
    stale = learner.update(_segments(tok))
    assert stale["kl_lr_scale_min"] < .01
    with torch.no_grad():
        current_probability = float(net.logits.softmax(-1)[0])
    fresh = learner.update(_segments(tok, behavior_p=current_probability))
    assert fresh["kl_lr_scale_mean"] == fresh["kl_lr_scale_min"] == 1.
    assert fresh["epochs_done"] == 1. and not fresh["early_stop"]


def test_default_is_exact_and_all_samples_are_retained() -> None:
    tok = StubTokenizer()
    default = Learner(_Net(), tok, _cfg(), device="cpu")
    explicit = Learner(_Net(), tok, _cfg(kl_lr_target=None), device="cpu")
    braked = Learner(_Net(), tok, _cfg(kl_lr_target=.01), device="cpu")
    segs = _segments(tok)
    a, b, c = (learner.update(segs) for learner in (default, explicit, braked))
    for p, q in zip(default.model.parameters(), explicit.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert a["approx_kl"] == b["approx_kl"] and a["loss_pi"] == b["loss_pi"]
    assert a["kl_lr_scale_mean"] == a["kl_lr_scale_min"] == 1.0
    assert isinstance(default.opt.param_groups[0]["lr"], float)
    assert c["epochs_done"] == 1 and not c["early_stop"]
    assert c["samples"] == c["micro_batches"] == 8
    assert c["kl_lr_scale_mean"] < .01 and c["kl_lr_scale_min"] < .01
    assert c["first_minibatch_kl"] == pytest.approx(8 - math.log(9), rel=1e-6)
    logged = _learner_fields(c)
    for key in ("first_minibatch_kl", "kl_lr_scale_mean", "kl_lr_scale_min"):
        assert logged[key] == c[key]


def test_existing_early_stop_remains_independent() -> None:
    tok = StubTokenizer()
    learner = Learner(_Net(), tok, _cfg(kl_lr_target=.01, target_kl=.03), device="cpu")
    result = learner.update(_segments(tok))
    assert result["early_stop"] and result["epochs_done"] == .25
    assert result["micro_batches"] == 2 and result["kl_lr_scale_mean"] < .01


def test_calm_iteration_keeps_full_learning_rate() -> None:
    tok = StubTokenizer()
    plain = Learner(_Net(), tok, _cfg(), device="cpu")
    braked = Learner(_Net(), tok, _cfg(kl_lr_target=.01), device="cpu")
    segs = _segments(tok, behavior_p=.9)
    a, b = plain.update(segs), braked.update(segs)
    assert b["kl_lr_scale_mean"] == b["kl_lr_scale_min"] == 1.0
    assert a["epochs_done"] == b["epochs_done"] == 1.0
    for p, q in zip(plain.model.parameters(), braked.model.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("anchored", [False, True])
def test_captured_body_and_eager_use_same_pre_step_kl_and_adam_scale(anchored: bool) -> None:
    """Execute the exact captured loss body with CPU tensors; CUDA capture itself is not exercised."""
    tok = StubTokenizer()
    cfg = _cfg(kl_lr_target=.01, decoupled_clip_anchor=anchored, micro_batch=2)
    eager = Learner(_Net(), tok, cfg, device="cpu")
    body = Learner(_Net(), tok, cfg, device="cpu")
    rb = build_batch(_segments(tok)[:2], cfg.gae, 0)
    reference = np.full(2, math.log(.6), np.float32) if anchored else None
    acc = _Accum(torch.device("cpu"))
    eager._minibatch(rb.samples, np.arange(2), None, eager._tensors(rb, reference), acc, 0)

    pf, pi, _ = body._columns(rb, reference)
    st = MicroStepper.__new__(MicroStepper)
    st.model, st.fwd, st.cfg, st.acc, st.autocast = body.model, body.model, cfg, _Accum(torch.device("cpu")), False
    st._exploration = st._uniform_kl = None
    st.den = torch.tensor([2, 2, 2, 1, 2, 1, 1], dtype=torch.float32)
    assert len(st.den) == len(DEN_COLS)
    st.mb_kl = torch.zeros(())
    res = _Res(BatchShape(2, 1, 2, 0, 0, 0, 0), torch.empty(0), dict(tok.collate(rb.samples)),
               torch.from_numpy(pf), torch.from_numpy(pi))
    st._body(res)
    torch.nn.utils.clip_grad_norm_(body.model.parameters(), cfg.grad_clip)
    body._minibatch_lr(st.mb_kl / 2, st.acc)
    body.opt.step()
    for p, q in zip(eager.model.parameters(), body.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    for key in ("first_minibatch_kl", "kl_lr_scale_mean", "kl_lr_scale_min"):
        assert acc.result()[key] == pytest.approx(st.acc.result()[key])


@pytest.mark.parametrize("target", [None, .01])
def test_fast_minibatch_orchestration_resets_kl_and_never_reads_it_to_host(
        monkeypatch: pytest.MonkeyPatch, target: float | None) -> None:
    """Use real orchestration/body/Adam on CPU; replace CUDA staging and fused found_inf handling only."""
    tok = StubTokenizer()
    learner = Learner(_Net(), tok, _cfg(kl_lr_target=target), device="cpu")
    rb = build_batch(_segments(tok)[:2], learner.cfg.gae, 0)
    pf, pi, masks = learner._columns(rb)
    st = MicroStepper.__new__(MicroStepper)
    st.model, st.fwd, st.cfg, st.acc, st.autocast = learner.model, learner.model, learner.cfg, _Accum(torch.device("cpu")), False
    st._exploration = st._uniform_kl = None
    st.den = torch.ones(len(DEN_COLS))
    st.mb_kl = torch.tensor(1234.)  # must be reset by _minibatch_fast before the first micro-step

    def set_den(pol: float, win: float, floors: float, hp: float, n_all: float, v_scale: float,
                epoch0: bool) -> None:
        st.den.copy_(torch.tensor([pol, win, floors, hp, n_all, v_scale, float(epoch0)]))

    def run(blk: SampleBlock, idx: np.ndarray, rows: np.ndarray, ints: np.ndarray, bucket: int) -> None:
        samples = [rb.samples[int(i)] for i in idx]
        res = _Res(BatchShape(len(idx), 1, 2, 0, 0, 0, 0), torch.empty(0), dict(tok.collate(samples)),
                   torch.from_numpy(rows), torch.from_numpy(ints))
        st._body(res)

    monkeypatch.setattr(st, "set_den", set_den)
    monkeypatch.setattr(st, "run", run)
    learner.stepper = st
    original_step = learner.opt.step

    def cpu_fused_step() -> None:
        found = getattr(learner.opt, "found_inf")
        assert torch.equal(found, torch.zeros(()))
        if target is not None:
            assert isinstance(learner.opt.param_groups[0]["lr"], Tensor)
        setattr(learner.opt, "found_inf", None)  # CPU Adam lacks fused found_inf, already checked above
        try:
            original_step()
        finally:
            setattr(learner.opt, "found_inf", found)

    monkeypatch.setattr(learner.opt, "step", cpu_fused_step)
    original_float = Tensor.__float__

    def no_kl_read(value: Tensor) -> float:
        assert value is not st.mb_kl, "fast path read minibatch KL back to the host"
        return original_float(value)

    monkeypatch.setattr(Tensor, "__float__", no_kl_read)
    for _ in range(2):
        result = learner._minibatch_fast(SampleBlock.__new__(SampleBlock), np.arange(2),
                                         np.ones(2, np.int64), pf, pi, masks, st.acc, 0)
        assert result is None
        assert torch.allclose(st.mb_kl, 2 * torch.tensor(8 - math.log(9)), atol=.02)
    diagnostics = st.acc.result()
    assert diagnostics["first_minibatch_kl"] == pytest.approx(8 - math.log(9), rel=1e-6)
    assert st.acc.minibatches == 2 and st.acc.samples == 4
    assert diagnostics["kl_lr_scale_min"] == (1.0 if target is None else pytest.approx(.01 / (8 - math.log(9))))


def test_old_checkpoint_uses_disabled_default_and_new_config_requires_opt_in(tmp_path: Path) -> None:
    tok = StubTokenizer()
    source = Learner(_Net(), tok, _cfg(), device="cpu")
    state = source.state()
    del state["config"]["kl_lr_target"]
    path = tmp_path / "old.pt"
    torch.save(state, path)
    default = Learner(_Net(), tok, _cfg(), device="cpu")
    default.load(path)
    braked = Learner(_Net(), tok, _cfg(kl_lr_target=.01), device="cpu")
    with pytest.raises(ValueError, match="config differs"):
        braked.load(path)
    braked.load(path, allow_config_change=True)
    result = braked.update(_segments(tok))
    assert result["kl_lr_scale_min"] < 1
    braked.save(path)
    restored = Learner(_Net(), tok, braked.cfg, device="cpu")
    restored.load(path)
    braked.update(_segments(tok))
    restored.update(_segments(tok))
    for p, q in zip(braked.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
