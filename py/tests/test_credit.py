"""Credit targets and asynchronous PPO anchors, using analytic one-state examples."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from ppo_stubs import StubSample, StubTokenizer
from spire.learner import Learner, PPOConfig
from spire.rollout import EpisodeMeta, GAEConfig, Segment, SegmentEnd, build_batch
from spire.schema import Kind
from spire.trainstep import PF_COLS


class ConstantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([math.log(9.0), 0.0]))
        self.value = nn.Parameter(torch.tensor(0.25))

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        n = batch["x"].shape[0]
        v = self.value.expand(n)
        return {"logits": self.logits.expand(n, 2), "value": v, "pwin_logit": v,
                "floors_pred": v, "hp_pred": v}


def one_step(tok: StubTokenizer) -> Segment:
    sample = StubSample(np.zeros(tok.n_features, np.float32), 2)
    return Segment(EpisodeMeta(0, 0, (2,), 0), 0, [sample], np.array([0], np.int64),
                   np.array([math.log(0.1)], np.float32), np.array([0.0], np.float32),
                   np.array([Kind.REST], np.int8), np.array([2], np.int16), np.array([0], np.int64),
                   end=SegmentEnd.TERMINAL, rewards=np.array([1.0], np.float32),
                   combat_id=np.array([-1], np.int32), win=np.array([1.0], np.float32),
                   win_mask=np.array([True]), floors_left=np.array([0.0], np.float32),
                   hp_post=np.array([0.0], np.float32), hp_mask=np.array([False]))


def config(*, decoupled_clip_anchor: bool = False, recompute_values: bool = False,
           exploration_by_kind: tuple[tuple[int, float], ...] = ()) -> PPOConfig:
    cfg = PPOConfig(epochs=1, minibatch=1, micro_batch=1, adv_norm=False, ent_coef=0.0,
                    vf_coef=0.0, aux_win=0.0, aux_floors=0.0, aux_hp=0.0, bf16=False,
                    gae=GAEConfig(gamma=1.0, lam=0.99))
    return replace(cfg, decoupled_clip_anchor=decoupled_clip_anchor, recompute_values=recompute_values,
                   exploration_by_kind=exploration_by_kind)


@pytest.mark.parametrize("anchored", [False, True])
def test_fresh_anchor_removes_stale_clipping_without_discarding_importance_ratio(anchored: bool) -> None:
    tok = StubTokenizer()
    net = ConstantNet()
    learner = Learner(net, tok, config(decoupled_clip_anchor=anchored), device="cpu")
    before = net.logits.detach().clone()
    result = learner.update([one_step(tok)])
    # Behavior chose a with probability .1; the frozen learner starts at .9. A=1.
    # Old clipping kills its improving gradient at ratio 9. Fresh clipping starts
    # at ratio 1 relative to the anchor but correctly retains IS weight .9/.1=9.
    assert result["loss_pi"] == pytest.approx(-9.0 if anchored else -1.2, rel=1e-6)
    assert result["clipfrac_epoch0"] == (0.0 if anchored else 1.0)
    if anchored:
        assert net.logits[0] > before[0] and net.logits[1] < before[1]
        assert result["behavior_kl_before_update"] == pytest.approx(8 - math.log(9), rel=1e-6)
        assert result["behavior_importance_p99"] == pytest.approx(9, rel=1e-6)
        assert result["behavior_importance_max"] == pytest.approx(9, rel=1e-6)
    else:
        torch.testing.assert_close(net.logits, before, rtol=0, atol=0)
        assert "behavior_kl_before_update" not in result
        assert "behavior_importance_p99" not in result and "behavior_importance_max" not in result


def test_preupdate_importance_tail_is_reported_without_clipping() -> None:
    tok = StubTokenizer()
    learner = Learner(ConstantNet(), tok, config(decoupled_clip_anchor=True), device="cpu")
    stale = one_step(tok)
    current = replace(stale, meta=replace(stale.meta, episode_uid=1),
                      logp=np.array([math.log(0.9)], np.float32))
    result = learner.update([stale, current])
    # These are frozen before either optimizer step: behavior IS weights [9, 1].
    assert result["behavior_importance_p99"] == pytest.approx(8.92, rel=1e-6)
    assert result["behavior_importance_max"] == pytest.approx(9, rel=1e-6)


@pytest.mark.parametrize("refresh", [False, True])
def test_value_refresh_changes_baseline_and_bootstrap_only_when_enabled(refresh: bool) -> None:
    tok = StubTokenizer()
    learner = Learner(ConstantNet(), tok, config(recompute_values=refresh), device="cpu")
    terminal = one_step(tok)
    seg = replace(terminal, end=SegmentEnd.CUT, bootstrap_sample=terminal.samples[0], bootstrap_value=10.0)
    result = learner.update([seg])
    assert result["value_mean"] == pytest.approx(0.25 if refresh else 0.0)
    assert result["return_mean"] == pytest.approx(1.25 if refresh else 11.0)


def test_value_refresh_requires_real_nonterminal_bootstrap_sample() -> None:
    tok = StubTokenizer()
    learner = Learner(ConstantNet(), tok, config(recompute_values=True), device="cpu")
    seg = replace(one_step(tok), end=SegmentEnd.CUT)
    with pytest.raises(AssertionError, match="bootstrap samples"):
        learner.update([seg])


def test_anchor_uses_same_exploration_mixture_as_training_and_keeps_old_denominator() -> None:
    tok = StubTokenizer()
    learner = Learner(ConstantNet(), tok,
                      config(decoupled_clip_anchor=True, exploration_by_kind=((int(Kind.REST), 0.2),)),
                      device="cpu")
    seg = one_step(tok)
    values, boot, ref = learner._recompute([seg])
    assert ref is not None
    np.testing.assert_allclose(ref, [math.log(0.8 * 0.9 + 0.2 * 0.5)], rtol=1e-6)
    rb = build_batch([seg], learner.cfg.gae, 0, values, boot)
    cols, _, _ = learner._columns(rb, ref)
    tens = learner._tensors(rb, ref)
    assert cols[0, PF_COLS.index("logp_b")] == seg.logp[0]
    assert cols[0, PF_COLS.index("logp_ref")] == ref[0]
    torch.testing.assert_close(tens.logp_b, torch.from_numpy(seg.logp))
    torch.testing.assert_close(tens.logp_ref, torch.from_numpy(ref))
    # With the feature disabled, the old reference is exactly the behavior logp.
    old_cols, _, _ = learner._columns(rb)
    np.testing.assert_array_equal(old_cols[:, PF_COLS.index("logp_ref")], rb.logp)


@pytest.mark.parametrize("epsilon", [0.0, 0.2])
def test_entropy_diagnostics_separate_base_policy_from_exploration(epsilon: float) -> None:
    tok = StubTokenizer()
    learner = Learner(ConstantNet(), tok,
                      config(exploration_by_kind=((int(Kind.REST), epsilon),) if epsilon else ()), device="cpu")
    result = learner.update([one_step(tok)])
    q = (1 - epsilon) * 0.9 + epsilon / 2
    base_entropy = -0.9 * math.log(0.9) - 0.1 * math.log(0.1)
    behavior_entropy = -q * math.log(q) - (1 - q) * math.log(1 - q)
    rest = result["entropy_by_kind"]["REST"]
    assert rest["base_entropy"] == pytest.approx(base_entropy, rel=1e-6)
    assert rest["entropy"] == pytest.approx(behavior_entropy, rel=1e-6)
    assert rest["base_entropy_norm"] == pytest.approx(base_entropy / math.log(2), rel=1e-6)
    assert result["entropy_base"] == pytest.approx(base_entropy, rel=1e-6)


def test_legacy_checkpoint_accepts_new_default_off_keys_and_requires_opt_in_to_changes(tmp_path: Path) -> None:
    tok = StubTokenizer()
    source = Learner(ConstantNet(), tok, config(), device="cpu")
    source.update([one_step(tok)])
    state = source.state()
    for key in ("exploration_by_kind", "uniform_kl_by_kind", "decoupled_clip_anchor"):
        del state["config"][key]
    del state["config"]["gae"]["stitch_segments"]
    path = tmp_path / "legacy.pt"
    torch.save(state, path)
    resumed = Learner(ConstantNet(), tok, config(), device="cpu")
    resumed.load(path)
    assert resumed.version == source.version and resumed.samples_seen == source.samples_seen
    for actual, expected in zip(resumed.model.parameters(), source.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed = Learner(ConstantNet(), tok, config(decoupled_clip_anchor=True), device="cpu")
    with pytest.raises(ValueError, match="checkpoint config differs"):
        changed.load(path)
    changed.load(path, allow_config_change=True)
    assert changed.cfg.decoupled_clip_anchor and changed.version == source.version


@pytest.mark.parametrize("section", ["ppo", "gae"])
def test_checkpoint_unknown_configuration_key_is_never_silently_ignored(tmp_path: Path, section: str) -> None:
    tok = StubTokenizer()
    learner = Learner(ConstantNet(), tok, config(), device="cpu")
    state = learner.state()
    target = state["config"] if section == "ppo" else state["config"]["gae"]
    target["unknown_credit_option"] = True
    path = tmp_path / "unknown.pt"
    torch.save(state, path)
    with pytest.raises(TypeError, match="unknown_credit_option"):
        learner.load(path, allow_config_change=True)


@pytest.mark.parametrize("stitch", [False, True])
def test_learner_reports_storage_cuts_and_the_ones_the_stitched_trace_crossed(stitch: bool) -> None:
    """py-adopt-algo: ``gae_joined_cuts`` is the live signal that stitching acts (0 when it is off). Of three CUT
    fragments, one continues into its adjacent successor; one is followed by another episode; one has a gap."""
    tok = StubTokenizer()
    base = one_step(tok)

    def frag(uid: int, start: int, end: SegmentEnd) -> Segment:
        return replace(base, meta=EpisodeMeta(0, uid, (2,), 0), start=start, end=end, bootstrap_value=0.5)

    segs = [frag(0, 0, SegmentEnd.CUT), frag(0, 1, SegmentEnd.CUT), frag(1, 0, SegmentEnd.CUT),
            frag(1, 2, SegmentEnd.TERMINAL)]
    cfg = replace(config(), gae=GAEConfig(gamma=1.0, lam=0.99, stitch_segments=stitch))
    m = Learner(ConstantNet(), tok, cfg, device="cpu").update(segs)
    assert m["gae_cut_segments"] == 3
    assert m["gae_joined_cuts"] == (1 if stitch else 0)
