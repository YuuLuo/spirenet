"""Kickstarting (spire.kickstart, notes/solo/kickstart-spec.md v1, section 5): each invariant is a test.

1. OFF = IDENTICAL: no ``ppo.kickstart`` section -> default, no teacher, no counters, no row record; the update with
   the section absent equals tree m's (identity_ref.py / kickstart_off_ref.py across the two trees; here: the PPO
   config and checkpoint layout are unchanged and a teacher-less learner loads an old checkpoint).
2. SELF-TEACHER: teacher = a checkpoint of the student's own current weights -> kick and its gradient are 0
   (max abs < 1e-6): the kickstarted update equals the plain one, on fake-host frames and (SPIRE_KICK_REAL =
   "<run config.json>:<segments.pt>") on real frames.
3. FORMULA: a synthetic minibatch with 1, 2 and 5 legal rows and known logits gives the hand-computed kick; illegal
   rows contribute nothing, the single-row step contributes 0.
4. SCHEDULE: coefficients at u = 0, updates / 2, updates, updates + 1; no teacher forward from u = updates on; the
   schedule position and the teacher's sha256 survive a checkpoint.
5. PATHS: the captured / compiled training step and the eager one give the same kick (CUDA).
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.kickstart import KickstartConfig, coefficients, input_config, kick_loss, kick_terms, load_teacher
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
WIDE = ModelConfig(d_model=96, n_layers=3, n_heads=4, d_ff=192, d_id=16, d_cond=16, d_static=32)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CFG = PPOConfig(epochs=2, minibatch=512, micro_batch=256, micro_tokens=4096, lr=1e-4, seed=3)


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
    assert sum(len(s) for s in segs) > 1500
    return segs


def save_teacher(path: Path, net: SpireNet, tok: Tokenizer, catalog_hash: int = 0, **tok_changes: object) -> Path:
    """A learner checkpoint of ``net`` (online = EMA) with the extra of spire.train.checkpoint_extra."""
    lr = Learner(copy.deepcopy(net), tok, PPOConfig(), device="cpu")
    tk = json.loads(json.dumps(asdict(replace(tok.config, **tok_changes))))  # type: ignore[arg-type]
    lr.save(path, extra={"model_config": asdict(net.config), "tokenizer": tk, "vocab": "synthetic",
                         "catalog": "none", "vocab_hash": tok.layout.vocab_hash, "catalog_hash": catalog_hash,
                         "run": "teacher"})
    return path


def kick_cfg(path: Path, **kw: object) -> PPOConfig:
    return replace(CFG, kickstart=KickstartConfig(teacher=str(path), **kw))  # type: ignore[arg-type]


def params(m: nn.Module) -> list[torch.Tensor]:
    return [p.detach().float().cpu().clone() for p in m.parameters()]


# --------------------------------------------------------------------------- 1. off = identical
def test_off_by_default_changes_nothing(tok: Tokenizer, segments: list[Segment], tmp_path: Path) -> None:
    assert not PPOConfig().kickstart.on and PPOConfig().kickstart == KickstartConfig()
    torch.manual_seed(0)
    lr = Learner(SpireNet(tok.layout, TINY), tok, CFG, device="cpu", size_fn=lambda s: tok.config.bucket(s.n_tokens))
    assert lr.teacher is None
    m = lr.update(segments)
    assert "kickstart" not in m
    st = lr.state()
    assert "kickstart" not in st["counters"]
    # a checkpoint of tree m (no kickstart key in its config) loads into a teacher-less learner and back
    old = dict(st)
    old["config"] = {k: v for k, v in st["config"].items() if k != "kickstart"}
    torch.save(old, tmp_path / "m.pt")
    lr2 = Learner(SpireNet(tok.layout, TINY), tok, CFG, device="cpu")
    lr2.load(tmp_path / "m.pt")
    assert lr2.updates == 1 and lr2.kick_u == 0


def test_teacher_must_be_fed_like_the_student(tok: Tokenizer, catalog, tmp_path: Path) -> None:
    torch.manual_seed(1)
    net = SpireNet(tok.layout, TINY)
    with pytest.raises(AssertionError, match="fed differently"):
        load_teacher(save_teacher(tmp_path / "fc.pt", net, tok, forecast=True), tok, None, TINY)
    with pytest.raises(AssertionError, match="fed differently"):
        load_teacher(save_teacher(tmp_path / "cat.pt", net, tok, catalog_hash=catalog.hash), tok, None, TINY)
    with pytest.raises(AssertionError, match="fed differently"):  # seat symmetry
        load_teacher(save_teacher(tmp_path / "ok.pt", net, tok), tok, None, replace(TINY, seat_symmetry=True))
    # the removal table's path is not an input (the teacher is fed the student's samples and their masks)
    a = input_config({**asdict(tok.config), "cycle_ban": True, "removal_rule": True, "removal_ratings": "/a"}, 1, 2, False)
    b = input_config({**asdict(tok.config), "cycle_ban": True, "removal_rule": True, "removal_ratings": "/b"}, 1, 2, False)
    c = input_config({**asdict(tok.config), "cycle_ban": True}, 1, 2, False)
    assert a == b != c
    t, sha = load_teacher(tmp_path / "ok.pt", tok, None, TINY)  # a different size is fine
    assert not t.training and not any(p.requires_grad for p in t.parameters()) and len(sha) == 64
    with pytest.raises(AssertionError, match="a teacher iff"):
        Learner(SpireNet(tok.layout, TINY), tok, kick_cfg(tmp_path / "ok.pt"), device="cpu")


# --------------------------------------------------------------------------- 2. self-teacher
def _self_teacher_pair(tok: Tokenizer, cfg: PPOConfig, mcfg: ModelConfig, tmp_path: Path, device: str, catalog=None,
                       phase: int = 1, **kw: object) -> tuple[Learner, Learner]:
    # one epoch, one minibatch: the invariant holds at the student's current weights, i.e. up to its first step
    cfg = replace(cfg, epochs=1, minibatch=1 << 24)
    torch.manual_seed(0)
    net = SpireNet(tok.layout, mcfg, catalog)
    path = save_teacher(tmp_path / "self.pt", net, tok, 0 if catalog is None else catalog.hash)
    teacher, sha = load_teacher(path, tok, catalog, mcfg)
    size = lambda s: tok.config.bucket(s.n_tokens)  # noqa: E731
    plain = Learner(copy.deepcopy(net), tok, cfg, device=device, size_fn=size, **kw)  # type: ignore[arg-type]
    kick = Learner(copy.deepcopy(net), tok, replace(cfg, kickstart=KickstartConfig(str(path), 1.0, 0.5, 10, phase=phase)),
                   device=device, size_fn=size, teacher=teacher, teacher_sha256=sha, **kw)  # type: ignore[arg-type]
    return plain, kick


def _assert_self_teacher(plain: Learner, kick: Learner, segments: list[Segment], tol_params: float) -> None:
    p0 = params(plain.model)
    mp, mk = plain.update(segments), kick.update(segments)
    k = mk["kickstart"]
    assert k["u"] == 0 and k["pi_coef"] == 1.0 and k["v_coef"] == 0.5 and k["teacher_evaluated"]
    assert abs(k["kick_kl"]) < 1e-6 and k["kick_v"] < 1e-6 and k["kick_agree"] == 1.0, k
    for a, b, c in zip(params(plain.model), params(kick.model), p0):
        assert (a - b).abs().max() <= tol_params, float((a - b).abs().max())
    assert mp["loss_pi"] == pytest.approx(mk["loss_pi"], abs=1e-6)


def _grad_of_kick(lr: Learner, segments: list[Segment]) -> tuple[float, float]:
    """max |kick| and max |d kick / d params| of the student on the first micro-batches (eager, fp32)."""
    samples = [x for s in segments for x in s.samples][:512]
    worst_k = worst_g = 0.0
    for i in range(0, len(samples), 128):
        lr.model.zero_grad(set_to_none=True)
        out = lr._forward(lr.model, samples[i:i + 128], lr.teacher)
        legal = torch.isfinite(out["logits"])
        pmf = (legal.sum(-1) > 1).float()
        acc = torch.zeros(5, device=lr.device)
        loss = kick_loss(out, pmf, torch.ones_like(pmf), max(1.0, float(pmf.sum())), float(len(pmf)), 1.0, 1.0, acc)
        loss.backward()
        worst_k = max(worst_k, abs(float(loss.detach())))
        worst_g = max(worst_g, max(float(p.grad.abs().max()) for p in lr.model.parameters() if p.grad is not None))
    return worst_k, worst_g


def test_self_teacher_kick_and_gradient_are_zero(tok: Tokenizer, segments: list[Segment], tmp_path: Path) -> None:
    plain, kick = _self_teacher_pair(tok, CFG, TINY, tmp_path, "cpu")
    k, g = _grad_of_kick(kick, segments)
    assert k < 1e-6 and g < 1e-6, (k, g)
    _assert_self_teacher(plain, kick, segments, 1e-2 * CFG.lr)


def _real_frames() -> tuple[str, str]:
    spec = os.environ.get("SPIRE_KICK_REAL", "")
    if not spec:
        pytest.skip("SPIRE_KICK_REAL=<run config.json>:<segments.pt> not set")
    cfg, segs = spec.split(":")
    return cfg, segs


def test_self_teacher_on_real_frames(tmp_path: Path) -> None:
    cfg_path, segs_path = _real_frames()
    from spire.train import load_config, load_vocab
    cfg = load_config(cfg_path)
    vocab, catalog = load_vocab(cfg)
    tk = Tokenizer(vocab, cfg.tokenizer)
    segs = [replace(s, policy_version=s.policy_version * 0)  # recorded by an old checkpoint; learner version 0
            for s in torch.load(segs_path, map_location="cpu", weights_only=False)["segments"][:60]]
    assert sum(len(s) for s in segs) > 2000
    ppo = replace(cfg.ppo, epochs=1, minibatch=1024, micro_batch=256, micro_tokens=16384)
    plain, kick = _self_teacher_pair(tk, ppo, cfg.model.config(), tmp_path, "cpu", catalog)
    k, g = _grad_of_kick(kick, segs)
    assert k < 1e-6 and g < 1e-6, (k, g)
    _assert_self_teacher(plain, kick, segs, 1e-2 * ppo.lr)


@cuda
@pytest.mark.parametrize("graphs", [False, True])
def test_self_teacher_fast_path(tok: Tokenizer, segments: list[Segment], tmp_path: Path, graphs: bool) -> None:
    plain, kick = _self_teacher_pair(tok, CFG, TINY, tmp_path, "cuda", fast=True, graphs=graphs)
    _assert_self_teacher(plain, kick, segments, 1e-2 * CFG.lr)


# --------------------------------------------------------------------------- 3. formula
def test_formula_on_a_synthetic_minibatch() -> None:
    inf = float("-inf")
    s = torch.tensor([[0.3, inf, inf, inf, inf, inf],        # 1 legal row
                      [1.0, -0.5, inf, inf, inf, inf],       # 2
                      [0.2, 0.1, -1.0, 2.0, 0.5, inf]])      # 5
    t = torch.tensor([[-4.0, 9.0, 9.0, 9.0, 9.0, 9.0],       # illegal rows of the student: anything
                      [0.0, 0.7, 50.0, 50.0, 50.0, 50.0],
                      [1.0, -1.0, 0.0, 0.3, 0.2, 77.0]])
    vs, vt = torch.tensor([0.5, -1.0, 2.0]), torch.tensor([0.0, 1.0, 2.5])

    def kl(a: list[float], b: list[float]) -> float:
        pa, pb = np.exp(a) / np.exp(a).sum(), np.exp(b) / np.exp(b).sum()
        return float((pa * (np.log(pa) - np.log(pb))).sum())

    want = [0.0, kl([0.0, 0.7], [1.0, -0.5]), kl([1.0, -1.0, 0.0, 0.3, 0.2], [0.2, 0.1, -1.0, 2.0, 0.5])]
    kl_t, dv2, agree = kick_terms(s, vs, t, vt)
    assert kl_t[0].item() == 0.0
    assert kl_t.tolist() == pytest.approx(want, abs=1e-6)
    assert dv2.tolist() == pytest.approx([0.25, 4.0, 0.25])
    assert agree.tolist() == [True, False, False]
    pmf = torch.tensor([0.0, 1.0, 1.0])  # the PPO policy-step mask (n_legal > 1)
    acc = torch.zeros(5)
    loss = kick_loss({"logits": s, "value": vs, "teacher_logits": t, "teacher_value": vt}, pmf, torch.ones(3),
                     2.0, 3.0, 1.5, 0.4, acc)
    hand = 1.5 * (want[1] + want[2]) / 2.0 + 0.4 * 0.5 * (0.25 + 4.0 + 0.25) / 3.0
    assert float(loss) == pytest.approx(hand, abs=1e-6)
    assert acc.tolist() == pytest.approx([want[1] + want[2], 4.5, 0.0, 2.0, 3.0], abs=1e-6)
    # with the single-row step weighted too, the loss is unchanged (it contributes exactly 0 KL)
    acc2 = torch.zeros(5)
    loss1 = kick_loss({"logits": s, "value": vs, "teacher_logits": t, "teacher_value": vt}, torch.ones(3),
                      torch.zeros(3), 2.0, 3.0, 1.5, 0.4, acc2)
    assert float(loss1) == pytest.approx(1.5 * (want[1] + want[2]) / 2.0, abs=1e-6)


# --------------------------------------------------------------------------- 4. schedule
class _Counting(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, batch):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.inner(batch)


def test_schedule_and_checkpoint_counters(tok: Tokenizer, segments: list[Segment], tmp_path: Path) -> None:
    ks = KickstartConfig("x", 1.0, 0.5, 4000)
    assert coefficients(ks, 0) == (1.0, 0.5)
    assert coefficients(ks, 2000) == (0.5, 0.25)
    assert coefficients(ks, 4000) == (0.0, 0.0) and coefficients(ks, 4001) == (0.0, 0.0)
    torch.manual_seed(2)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, WIDE), tok)
    teacher, sha = load_teacher(path, tok, None, TINY)
    counting = _Counting(teacher)
    cfg = kick_cfg(path, pi_coef=1.0, v_coef=0.5, updates=2)
    torch.manual_seed(0)
    lr = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                 teacher=counting, teacher_sha256=sha)
    rows = []
    calls = []
    for _ in range(2):
        rows.append(lr.update(segments)["kickstart"])
        calls.append(counting.calls)
    assert rows[0]["pi_coef"] == 1.0 and rows[1]["pi_coef"] == 0.5 and rows[1]["v_coef"] == 0.25
    assert 0 < calls[0] < calls[1] and rows[0]["kick_kl"] > 0 and rows[0]["kick_v"] > 0
    lr.save(tmp_path / "s.pt")
    st = torch.load(tmp_path / "s.pt", weights_only=True)
    assert st["counters"]["kickstart"] == {"u": 2, "teacher": str(path), "teacher_sha256": sha, "phase": 1,
                                           "release_update": None}
    assert not any(k.startswith("inner") for k in st["model"])  # the teacher is not saved
    # resume: the schedule continues at u = 2 = updates -> no teacher forward, coefficients 0
    counting2 = _Counting(load_teacher(path, tok, None, TINY)[0])
    lr2 = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                  teacher=counting2, teacher_sha256=sha)
    lr2.load(tmp_path / "s.pt")
    assert lr2.kick_u == 2
    for u in (2, 3):
        r = lr2.update(segments)["kickstart"]
        assert r == {"u": u, "phase": 1, "pi_coef": 0.0, "v_coef": 0.0, "teacher_evaluated": False, "release_update": None}
    assert counting2.calls == 0 and lr2.teacher is None
    # a different teacher file is refused; switching off before the end is refused, after it allowed
    lr3 = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu", teacher=counting2, teacher_sha256="0" * 64)
    with pytest.raises(AssertionError, match="teacher changed"):
        lr3.load(tmp_path / "s.pt")
    off = Learner(SpireNet(tok.layout, TINY), tok, CFG, device="cpu")
    off.load(tmp_path / "s.pt", allow_config_change=True)  # u = 2 >= updates: over
    long_cfg = kick_cfg(path, updates=10)
    lr4 = Learner(SpireNet(tok.layout, TINY), tok, long_cfg, device="cpu", teacher=counting2, teacher_sha256=sha)
    lr4.save(tmp_path / "s4.pt")
    with pytest.raises(AssertionError, match="switched off"):
        Learner(SpireNet(tok.layout, TINY), tok, CFG, device="cpu").load(tmp_path / "s4.pt", allow_config_change=True)
    with pytest.raises(AssertionError, match="no kickstart counters"):
        Learner(SpireNet(tok.layout, TINY), tok, long_cfg, device="cpu", teacher=counting2,
                teacher_sha256=sha).load(_strip_kick(tmp_path / "s4.pt"))


def _strip_kick(path: Path) -> Path:
    st = torch.load(path, weights_only=True)
    del st["counters"]["kickstart"]
    out = path.with_name(path.stem + "-nokick.pt")
    torch.save(st, out)
    return out


# --------------------------------------------------------------------------- 5. paths
@cuda
def test_paths_give_the_same_kick(tok: Tokenizer, segments: list[Segment], tmp_path: Path) -> None:
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    torch.manual_seed(5)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, replace(WIDE, init_std=0.3)), tok)  # a sharp policy
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    cfg = kick_cfg(path, pi_coef=1.0, v_coef=0.5, updates=100)
    ls = {}
    for name, fast, graphs, comp in (("generic", False, False, False), ("fast", True, False, False),
                                     ("graphs", True, True, False), ("compiled", True, True, True)):
        teacher, sha = load_teacher(path, tok, None, TINY)
        ls[name] = Learner(copy.deepcopy(base), tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                           fast=fast, graphs=graphs, compile_model=comp, teacher=teacher, teacher_sha256=sha)
    p0 = params(ls["generic"].model)
    ms = {k: lr.update(segments) for k, lr in ls.items()}
    ref = ms["generic"]["kickstart"]
    assert ref["kick_kl"] > 0.01 and ref["kick_v"] > 0
    deltas = {k: [a - b for a, b in zip(params(lr.model), p0)] for k, lr in ls.items()}
    for k in ("fast", "graphs", "compiled"):
        got = ms[k]["kickstart"]
        assert ms[k]["fast_path"]
        # the tolerance of tests/test_learner_fast.py for the other losses
        for key, tol in (("kick_kl", 1e-4), ("kick_v", 1e-4), ("kick_agree", 0.02)):
            assert got[key] == pytest.approx(ref[key], rel=0.05, abs=tol), (k, key, got[key], ref[key])
        assert (got["pi_coef"], got["v_coef"]) == (ref["pi_coef"], ref["v_coef"])
        for key in ("loss_pi", "loss_v", "approx_kl"):
            assert ms[k][key] == pytest.approx(ms["generic"][key], rel=0.05, abs=2e-3), (k, key)
        num = sum(float((d - r).abs().sum()) for d, r in zip(deltas[k], deltas["generic"]))
        den = sum(float(r.abs().sum()) for r in deltas["generic"])
        assert num / den < 0.05, (k, num / den)
    assert ms["graphs"]["stepper"]["captures"] > 0
    # the schedule's end drops the teacher and the graphs captured with it; the next update recaptures without it
    lr = ls["graphs"]
    lr.kick_u = cfg.kickstart.updates
    c0 = lr.stepper.captures  # type: ignore[union-attr]
    m = lr.update(segments)
    assert m["kickstart"]["teacher_evaluated"] is False and lr.stepper.teacher is None  # type: ignore[union-attr]
    assert lr.stepper.captures > c0  # type: ignore[union-attr]
    assert math.isfinite(m["loss_pi"])


# =========================================================================== v2 (notes/solo/kickstart-spec-v2.md 4)
P0 = replace(CFG, epochs=1)


def _size(tok: Tokenizer):  # type: ignore[no-untyped-def]
    return lambda s: tok.config.bucket(s.n_tokens)


# --------------------------------------------------------------------------- v2.1 phase 0, student = teacher
def _assert_phase0_self(kick: Learner, segments: list[Segment]) -> None:
    m = kick.update(segments)
    k = m["kickstart"]
    assert k["phase"] == 0 and k["u"] == 0 and k["teacher_evaluated"] and k["release_update"] is None
    assert abs(k["kick_kl"]) < 1e-6 and k["kick_v"] < 1e-6 and k["kick_agree"] == 1.0, k
    assert m["loss_pi"] == 0.0 and m["approx_kl"] == 0.0 and m["explained_variance"] == k["ev_student"]


def test_phase0_self_teacher(tok: Tokenizer, segments: list[Segment], tmp_path: Path) -> None:
    _, kick = _self_teacher_pair(tok, CFG, TINY, tmp_path, "cpu", phase=0)
    _assert_phase0_self(kick, segments)


def test_phase0_self_teacher_on_real_frames(tmp_path: Path) -> None:
    cfg_path, segs_path = _real_frames()
    from spire.train import load_config, load_vocab
    cfg = load_config(cfg_path)
    vocab, catalog = load_vocab(cfg)
    tk = Tokenizer(vocab, cfg.tokenizer)
    segs = [replace(s, policy_version=s.policy_version * 0)
            for s in torch.load(segs_path, map_location="cpu", weights_only=False)["segments"][:60]]
    assert sum(len(s) for s in segs) > 2000
    ppo = replace(cfg.ppo, epochs=1, minibatch=1024, micro_batch=256, micro_tokens=16384)
    _, kick = _self_teacher_pair(tk, ppo, cfg.model.config(), tmp_path, "cpu", catalog, phase=0)
    _assert_phase0_self(kick, segs)


# --------------------------------------------------------------------------- v2.2 phase 0 loss
def _phase0_hand_loss(model: nn.Module, teacher: nn.Module, tok: Tokenizer, segments: list[Segment],
                      cfg: PPOConfig) -> float:
    """pi_coef KL(pi_T' || pi_S) + vf_coef value loss + v_coef 0.5 (V_S - V_T)^2 + the auxiliary losses of the whole
    (single) minibatch, eagerly in fp32 on the CPU."""
    import torch.nn.functional as F
    from spire.kickstart import smooth_table
    from spire.rollout import build_batch
    rb = build_batch(segments, cfg.gae, 0)
    model, teacher = model.cpu(), teacher.cpu()
    batch = tok.collate(rb.samples)
    with torch.no_grad():
        s, t = model.eval()(batch), teacher(batch)
    model.train()
    legal = torch.isfinite(s["logits"])
    a = torch.as_tensor(rb.actions, dtype=torch.int64)
    pmf = (torch.as_tensor(rb.n_legal > 1) & legal.gather(1, a[:, None]).squeeze(1)).float()
    eps = smooth_table(cfg.kickstart, "cpu")
    kl, dv2, _ = kick_terms(s["logits"], s["value"], t["logits"], t["value"],
                            None if eps is None else eps[torch.as_tensor(rb.kinds, dtype=torch.int64)])
    n = len(rb)
    ret = torch.as_tensor(rb.returns)
    wm, hm = torch.as_tensor(rb.win_mask).float(), torch.as_tensor(rb.hp_mask).float()
    ks = cfg.kickstart
    t1 = ks.pi_coef * float((kl * pmf).sum()) / max(1.0, float(pmf.sum()))
    t2 = cfg.vf_coef * 0.5 * float((s["value"] - ret).square().sum()) / n
    t3 = ks.v_coef * 0.5 * float(dv2.sum()) / n
    win = float((F.binary_cross_entropy_with_logits(s["pwin_logit"], torch.as_tensor(rb.win), reduction="none")
                 * wm).sum()) / max(1.0, float(wm.sum()))
    fl = float(((s["floors_pred"] - torch.as_tensor(rb.floors_left / cfg.floors_scale)).square() * wm).sum()) \
        / max(1.0, float(wm.sum()))
    hp = float(((s["hp_pred"] - torch.as_tensor(rb.hp_post)).square() * hm).sum()) / max(1.0, float(hm.sum()))
    assert t1 > 1e-3 and t3 > 1e-4
    return t1 + t2 + t3 + cfg.aux_win * win + cfg.aux_floors * fl + cfg.aux_hp * hp


def _phase0_cfg(path: Path, kinds: np.ndarray) -> PPOConfig:
    # PPO terms that would show if they were on: a large entropy bonus, a uniform barrier, a KL brake and an early stop
    # that every minibatch would trigger; smoothing on the most frequent decision kind
    k = int(np.bincount(kinds.astype(np.int64)).argmax())
    return replace(P0, minibatch=1 << 24, ent_coef=0.5, uniform_kl_by_kind=((k, 0.5),), target_kl=1e-9,
                   kl_lr_target=1e-9, kickstart=KickstartConfig(str(path), 1.0, 0.5, 10, phase=0, smooth=((k, 0.2),)))


def _assert_phase0_row(m: dict, cfg: PPOConfig) -> None:
    assert m["loss_pi"] == 0.0 and m["loss_ent_bonus"] == 0.0 and m["loss_uniform_ce"] == 0.0
    assert m["approx_kl"] == 0.0 and m["clipfrac"] == 0.0 and m["kl_lr_scale_min"] == 1.0
    assert not m["early_stop"] and m["epochs_done"] == 1.0 and m["lr"] == cfg.lr


def test_phase0_loss_is_the_four_terms(tok: Tokenizer, segments: list[Segment], tmp_path: Path) -> None:
    torch.manual_seed(5)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, replace(WIDE, init_std=0.3)), tok)
    cfg = _phase0_cfg(path, np.concatenate([s.kinds for s in segments]))
    torch.manual_seed(0)
    teacher, sha = load_teacher(path, tok, None, TINY)
    lr = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu", size_fn=_size(tok), teacher=teacher,
                 teacher_sha256=sha)
    hand = _phase0_hand_loss(copy.deepcopy(lr.model), teacher, tok, segments, cfg)
    m = lr.update(segments)
    _assert_phase0_row(m, cfg)
    assert m["kickstart"]["loss_total"] == pytest.approx(hand, rel=1e-5)


@cuda
@pytest.mark.parametrize("graphs", [False, True])
def test_phase0_loss_fast_paths(tok: Tokenizer, segments: list[Segment], tmp_path: Path, graphs: bool) -> None:
    torch.manual_seed(5)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, replace(WIDE, init_std=0.3)), tok)
    cfg = _phase0_cfg(path, np.concatenate([s.kinds for s in segments]))
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    teacher, sha = load_teacher(path, tok, None, TINY)
    hand = _phase0_hand_loss(copy.deepcopy(base), copy.deepcopy(teacher), tok, segments, cfg)
    lr = Learner(copy.deepcopy(base), tok, cfg, device="cuda", size_fn=_size(tok), fast=True, graphs=graphs,
                 teacher=teacher, teacher_sha256=sha)
    m = lr.update(segments)
    assert m["fast_path"] and (m["stepper"]["captures"] > 0) == graphs
    _assert_phase0_row(m, cfg)
    assert m["kickstart"]["loss_total"] == pytest.approx(hand, rel=0.02)  # bf16 autocast


# --------------------------------------------------------------------------- v2.3 phase 0 acting
def test_phase0_actor_acts_with_the_teacher(vocab: Vocab, tmp_path: Path) -> None:
    from spire.learner import FileWeightPublisher
    from spire.train import TrainConfig, acting_policy, build
    torch.manual_seed(7)
    common = {"run_root": str(tmp_path), "vocab": "synthetic", "catalog": "none", "seed": 3,
              "tokenizer": {"token_buckets": [20, 28, 40, 64, 80, 96, 128, 192, 256, 384]},
              "actor": {"use_graphs": False, "device": "cpu"}, "rollout": {"segment_len": 64}}
    explore = [[2, 0.05], [5, 0.1]]
    t_cfg = build(TrainConfig, {**common, "run_name": "teacher", "model": {"override": asdict(WIDE)},
                                "ppo": {"exploration_by_kind": explore}})
    tk = Tokenizer(vocab, t_cfg.tokenizer)
    tpath = save_teacher(tmp_path / "teacher.pt", SpireNet(tk.layout, WIDE), tk)
    s_cfg = build(TrainConfig, {**common, "run_name": "student", "model": {"override": asdict(TINY)},
                                "ppo": {"exploration_by_kind": explore, "epochs": 1,
                                        "kickstart": {"teacher": str(tpath), "phase": 0}}})
    ema = torch.load(tpath, weights_only=True)["ema"]
    FileWeightPublisher(t_cfg.run_dir / "weights").publish(ema, 5)  # the teacher's own run acting with these weights
    FileWeightPublisher(s_cfg.run_dir / "weights").publish(SpireNet(tk.layout, TINY).state_dict(), 5)  # the student's
    runs = {}
    for name, cfg in (("plain", t_cfg), ("phase0", s_cfg)):
        policy = acting_policy(cfg, tk, None, torch.device("cpu"), 1, 0)
        assert policy.frozen == (name == "phase0") and policy.wait_for_weights(10.0) and policy.version == 5
        segs: list[Segment] = []
        store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=cfg.rollout)
        hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False)) for i in range(8)]
        Actor(hosts, policy, tk, EpisodeSampler(EpisodeDist(), vocab, 0), ActorConfig(max_batch=32),
              store=store).run(max_decisions=1500)
        policy.close()
        assert all(torch.equal(policy.net.state_dict()[k], v) for k, v in ema.items())  # never replaced
        assert policy.weight_loads == (1 if name == "plain" else 0)
        runs[name] = segs
    a, b = runs["plain"], runs["phase0"]
    assert len(a) == len(b) and sum(len(s) for s in a) > 500
    for x, y in zip(a, b):
        for f in ("actions", "logp", "values", "kinds", "policy_version"):
            assert np.array_equal(getattr(x, f), getattr(y, f)), f


# --------------------------------------------------------------------------- v2.4 smoothing
def test_smoothed_teacher() -> None:
    inf = float("-inf")
    s = torch.tensor([[0.3, inf, inf, inf, inf, inf],
                      [1.0, -0.5, inf, inf, inf, inf],
                      [0.2, 0.1, -1.0, 2.0, 0.5, inf],
                      [0.2, 0.1, -1.0, 2.0, 0.5, inf]])
    t = torch.tensor([[-4.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                      [0.0, 0.7, 50.0, 50.0, 50.0, 50.0],
                      [1.0, -1.0, 0.0, 30.3, 0.2, 77.0],     # a near one-hot teacher
                      [1.0, -1.0, 0.0, 30.3, 0.2, 77.0]])
    vs, vt = torch.zeros(4), torch.zeros(4)
    eps = torch.tensor([0.1, 0.1, 0.05, 0.0])  # the last step: an unlisted kind

    def hand(tl: list[float], sl: list[float], e: float) -> float:
        pt = np.exp(np.array(tl) - max(tl))
        pt = (1 - e) * pt / pt.sum() + e / len(tl)
        ps = np.exp(np.array(sl) - max(sl))
        ps = ps / ps.sum()
        return float((pt * (np.log(pt) - np.log(ps))).sum())

    want = [0.0, hand([0.0, 0.7], [1.0, -0.5], 0.1), hand([1.0, -1.0, 0.0, 30.3, 0.2], [0.2, 0.1, -1.0, 2.0, 0.5], 0.05),
            hand([1.0, -1.0, 0.0, 30.3, 0.2], [0.2, 0.1, -1.0, 2.0, 0.5], 0.0)]
    kl, _, agree = kick_terms(s, vs, t, vt, eps)
    assert kl[0].item() == 0.0
    assert kl.tolist() == pytest.approx(want, rel=1e-5, abs=1e-6)
    assert kl[2] != kl[3]  # the listed step is smoothed, the unlisted one is not
    # eps 0 everywhere = v1, bit for bit
    v1 = kick_terms(s, vs, t, vt)
    v0 = kick_terms(s, vs, t, vt, torch.zeros(4))
    assert all(torch.equal(x, y) for x, y in zip(v1, v0))
    # the configured table: listed kinds get their eps, the others 0
    from spire.kickstart import smooth_table
    from spire.schema import Kind
    tab = smooth_table(KickstartConfig("x", smooth=((int(Kind.EVENT), 0.05),)), "cpu")
    assert tab is not None and tab[int(Kind.EVENT)] == pytest.approx(0.05) and float(tab.sum()) == pytest.approx(0.05)
    assert smooth_table(KickstartConfig("x"), "cpu") is None
    with pytest.raises(AssertionError):
        KickstartConfig("x", smooth=((int(Kind.EVENT), 1.5),))


# --------------------------------------------------------------------------- v2.5 release
def test_release_by_file(tok: Tokenizer, segments: list[Segment], tmp_path: Path) -> None:
    from spire.kickstart import fade_factor
    ks = KickstartConfig("x", 1.0, 0.5, 4000, phase=1, fade_updates=4)
    assert [fade_factor(ks, u, None) for u in (0, 10, 10 ** 6)] == [1.0, 1.0, 1.0]
    assert [fade_factor(ks, u, 7) for u in (7, 8, 9, 10, 11)] == [0.75, 0.5, 0.25, 0.0, 0.0]
    assert fade_factor(replace(ks, phase=0), 10 ** 6, None) == 1.0
    segs = segments[:12]
    torch.manual_seed(2)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, WIDE), tok)
    fade = 5
    cfg = replace(CFG, epochs=1, kickstart=KickstartConfig(str(path), 1.0, 0.5, 3, phase=1, fade_updates=fade))
    release = tmp_path / "KICK_RELEASE"

    def learner() -> tuple[Learner, _Counting]:
        counting = _Counting(load_teacher(path, tok, None, TINY)[0])
        return Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu", size_fn=_size(tok), teacher=counting,
                       teacher_sha256=load_teacher(path, tok, None, TINY)[1], release_file=release), counting

    torch.manual_seed(0)
    lr, counting = learner()
    for u in range(50):  # no file: constant (past v1's `updates` = 3: fade_updates replaces the count schedule)
        r = lr.update(segs)["kickstart"]
        assert (r["u"], r["pi_coef"], r["v_coef"], r["teacher_evaluated"], r["release_update"]) == \
            (u, 1.0, 0.5, True, None)
    release.write_text("")
    coefs = [lr.update(segs)["kickstart"] for _ in range(3)]
    assert [c["release_update"] for c in coefs] == [50] * 3
    assert [c["pi_coef"] for c in coefs] == pytest.approx([0.8, 0.6, 0.4])
    assert [c["v_coef"] for c in coefs] == pytest.approx([0.4, 0.3, 0.2])
    release.unlink()  # read once: the fade continues without the file
    lr.save(tmp_path / "mid.pt")
    assert torch.load(tmp_path / "mid.pt", weights_only=True)["counters"]["kickstart"]["release_update"] == 50
    lr2, counting2 = learner()
    lr2.load(tmp_path / "mid.pt")
    r = lr2.update(segs)["kickstart"]
    assert (r["u"], r["pi_coef"], r["release_update"], r["teacher_evaluated"]) == (53, pytest.approx(0.2), 50, True)
    calls = counting2.calls
    assert calls > 0
    for u in (54, 55):  # the fade_updates-th update of the fade has 0: the teacher is gone
        r = lr2.update(segs)["kickstart"]
        assert r == {"u": u, "phase": 1, "pi_coef": 0.0, "v_coef": 0.0, "teacher_evaluated": False,
                     "release_update": 50}
    assert counting2.calls == calls and lr2.teacher is None
    # switching the kickstart off: refused during the fade, allowed after it
    off = Learner(SpireNet(tok.layout, TINY), tok, CFG, device="cpu")
    with pytest.raises(AssertionError, match="switched off"):
        off.load(tmp_path / "mid.pt", allow_config_change=True)
    lr2.save(tmp_path / "end.pt")
    off.load(tmp_path / "end.pt", allow_config_change=True)


# --------------------------------------------------------------------------- v2.6 phase switch
def _state_sha(lr: Learner) -> dict[str, str]:
    import hashlib
    import io

    def h(obj: object) -> str:
        b = io.BytesIO()
        torch.save(obj, b)
        return hashlib.sha256(b.getvalue()).hexdigest()
    return {"model": h(lr.model.state_dict()), "ema": h(lr.ema.state_dict()), "opt": h(lr.opt.state_dict()),
            "counters": h([lr.version, lr.updates, lr.samples_seen, lr.ema_steps, lr.kick_u, lr.kick_release])}


def test_phase_switch(tok: Tokenizer, segments: list[Segment], tmp_path: Path) -> None:
    torch.manual_seed(2)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, WIDE), tok)
    ks0 = KickstartConfig(str(path), 1.0, 0.5, 4000, phase=0, fade_updates=1500, smooth=((2, 0.05),))
    cfg0 = replace(P0, kickstart=ks0)
    cfg1 = replace(cfg0, kickstart=replace(ks0, phase=1))
    teacher, sha = load_teacher(path, tok, None, TINY)
    torch.manual_seed(0)
    lr0 = Learner(SpireNet(tok.layout, TINY), tok, cfg0, device="cpu", size_fn=_size(tok), teacher=teacher,
                  teacher_sha256=sha, release_file=tmp_path / "KICK_RELEASE")
    for _ in range(2):
        m = lr0.update(segments)
        assert m["kickstart"]["phase"] == 0 and m["loss_pi"] == 0.0
    lr0.save(tmp_path / "p0.pt")
    before = _state_sha(lr0)
    lr1 = Learner(SpireNet(tok.layout, TINY), tok, cfg1, device="cpu", size_fn=_size(tok), teacher=teacher,
                  teacher_sha256=sha, release_file=tmp_path / "KICK_RELEASE")
    with pytest.raises(ValueError, match="config differs"):
        lr1.load(tmp_path / "p0.pt")
    lr1.load(tmp_path / "p0.pt", allow_config_change=True)
    assert _state_sha(lr1) == before and lr1.kick_u == 2 and lr1.updates == 2
    m = lr1.update(segments)
    k = m["kickstart"]
    assert (k["phase"], k["u"], k["pi_coef"], k["v_coef"], k["teacher_evaluated"]) == (1, 2, 1.0, 0.5, True)
    assert m["loss_pi"] != 0.0 and m["approx_kl"] > 0.0 and "ev_student" not in k
    # back to phase 0 is refused
    lr1.save(tmp_path / "p1.pt")
    lr0b = Learner(SpireNet(tok.layout, TINY), tok, cfg0, device="cpu", teacher=teacher, teacher_sha256=sha)
    with pytest.raises(AssertionError, match="phase 1 -> 0 refused"):
        lr0b.load(tmp_path / "p1.pt", allow_config_change=True)
    with pytest.raises(AssertionError, match="one epoch"):
        replace(cfg0, epochs=2)
