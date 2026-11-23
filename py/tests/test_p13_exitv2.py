"""Verification suite for py-stage-p13.1 (ExIt v2: recency/legacy prioritization, margin-capped Focal & soft MPO
targets, legal_mask enforcement, momentum-free Adam-preconditioned step, macro ExIt records, online pwin calibrator)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.exit_buffer import ExitBuffer, ExitRecord, load_exit_shard, save_exit_shard
from spire.exit_teacher import (
    MACRO_ELIGIBLE_KINDS,
    ExitTeacherConfig,
    OnlinePwinCalibrator,
    _macro_step_cap,
    _score_macro_horizon_frame,
    format_ab_pair_event,
    is_macro_v3_horizon_reached,
    passes_macro_emission_filter,
    passes_macro_v3_emission_filter,
    select_macro_candidates,
    should_expand_macro_v3_stage,
    should_trigger_fight,
    should_trigger_macro,
)
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import Kind
from spire.synth import FakeHost, FakeHostConfig, FrameGen, random_frame
from spire.tokenize import Sample, Tokenizer, TokenizerConfig
from spire.train import TrainConfig, _learner_fields, build, config_dict
from spire.vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab

P13V2_PY = Path(__file__).resolve().parents[1]
P13_PY = P13V2_PY.parents[1] / "p13" / "py"
TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def tok(vocab: Vocab) -> Tokenizer:
    return Tokenizer(vocab, TokenizerConfig(token_buckets=(20, 28, 40, 64, 80, 96, 128, 192, 256, 384)))


@pytest.fixture(scope="module")
def fake_segments(vocab: Vocab, tok: Tokenizer) -> list[Segment]:
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=64))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=45, hashes=False)) for i in range(6)]
    Actor(
        hosts,
        UniformPolicy(0),
        tok,
        EpisodeSampler(EpisodeDist(), vocab, 0),
        ActorConfig(max_batch=16),
        store=store,
    ).run(max_decisions=1000)
    assert sum(len(s) for s in segs) >= 500
    return segs


def _make_sample(vocab: Vocab, tok: Tokenizer, kind: Kind = Kind.COMBAT, seed: int = 77, min_act: int = 4) -> tuple[Sample, int]:
    gen = FrameGen(vocab)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        raw = random_frame(gen, rng, kind, n_seats=1)
        fr = parse_frame(raw)
        if fr.n_actions >= min_act:
            s = tok.encode_batch([fr], sort=True).samples()[0]
            return s, fr.n_actions
    raise AssertionError(f"could not generate {kind.name} frame with >= {min_act} actions")


def _rec(
    sample: Sample,
    target_action: int = 1,
    *,
    gain: float = 0.12,
    teacher_version: int = 100,
    legal_mask: np.ndarray | None = None,
    cand_rows: tuple[int, ...] = (),
    cand_scores: tuple[float, ...] = (),
    cand_p_old: tuple[float, ...] = (),
    stage: str = "A3-BB",
) -> ExitRecord:
    n_act = sample.n_actions
    lm = np.ones(n_act, dtype=np.bool_) if legal_mask is None else legal_mask
    cr = cand_rows if cand_rows else (0, target_action)
    return ExitRecord(
        sample=sample,
        target_action=target_action,
        legal_mask=lm,
        cand_rows=cr,
        greedy_action=0,
        gain=gain,
        gain_s2=gain,
        t_stat=2.5,
        dv_after=0.08,
        p_old=1e-4,
        p_group_old=1e-4,
        teacher_version=teacher_version,
        stage=stage,
        verb="PLAY",
        char="SILENT",
        ascension=10,
        ts=time.time(),
        cand_scores=cand_scores,
        cand_p_old=cand_p_old,
    )


# =========================================================================== 1. DEFAULTS = BIT-IDENTICAL TO P13


_PARITY_PROBE = r"""
import hashlib, json, tempfile
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.exit_buffer import ExitRecord, save_exit_shard
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore
from spire.schema import Kind
from spire.synth import FakeHost, FakeHostConfig, FrameGen, random_frame
from spire.tokenize import Tokenizer
from spire.train import TrainConfig, build, config_dict
from spire.vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

vocab = Vocab(make_synthetic_vocab())
catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
tok = Tokenizer(vocab)
mcfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)

segs = []
store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=40, hashes=False)) for i in range(4)]
Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
      ActorConfig(max_batch=16), store=store).run(max_decisions=450)

gen = FrameGen(vocab)
rng = np.random.default_rng(9)
for _ in range(50):
    fr = parse_frame(random_frame(gen, rng, Kind.COMBAT, n_seats=1))
    if fr.n_actions >= 4:
        s0 = tok.encode_batch([fr], sort=True).samples()[0]
        break

with tempfile.TemporaryDirectory() as td:
    edir = Path(td) / "shards"
    recs = [ExitRecord(s0, 1, np.ones(s0.n_actions, dtype=np.bool_), (0, 1), 0, 0.15, 0.12, 2.5, 0.1,
                       1e-3, 1e-3, 0, "A3-BB", "PLAY", "SILENT", 10, 100.0) for _ in range(16)]
    save_exit_shard(edir, "w0", 1, recs, vocab_hash=vocab.hash)
    torch.manual_seed(17)
    net = SpireNet(tok.layout, mcfg, catalog)
    ppo = PPOConfig(epochs=1, minibatch=128, micro_batch=64, lr=1e-4, seed=5, bf16=False,
                    exit_coef=0.05, exit_dir=str(edir), exit_batch_size=8, exit_min_samples=8)
    lr = Learner(net, tok, ppo, device="cpu", fast=False)
    m = lr.update(segs)
    for k in ("time_prep", "time_train", "samples_per_s", "lag_s", "exit_step_ms"):
        m.pop(k, None)
    st = lr.state()
    pd = ppo.to_dict()
    pd["exit_dir"] = "<tmp>"
    out = {
        "cfg_off": config_dict(build(TrainConfig, {"run_name": "x", "run_root": "/tmp/r"})),
        "ppo_dict": pd,
        "metrics_keys": sorted(m.keys()),
        "exit_loss": m["exit_loss"],
        "exit_step_kl": m["exit_step_kl"],
        "model_hash": sha(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in st["model"].values())),
        "ema_hash": sha(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in st["ema"].values())),
    }
print(json.dumps(out, sort_keys=True))
"""


def test_defaults_are_bit_identical_to_p13() -> None:
    """With all p13.1 knobs at defaults, config_dict, PPOConfig.to_dict, metrics keys, and updated weights
    (with exit_coef > 0!) match py-stage-p13 bit-for-bit."""
    if not P13_PY.is_dir():
        pytest.skip(f"reference p13 missing: {P13_PY}")
    res = []
    for tree in (P13_PY, P13V2_PY):
        env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"}
        r = subprocess.run([sys.executable, "-B", "-c", _PARITY_PROBE], env=env, capture_output=True, text=True,
                           timeout=120, cwd=str(tree))
        assert r.returncode == 0, f"{tree} failed:\n{r.stderr}"
        res.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert res[1] == res[0]


# =========================================================================== 2. IMMORTAL G145 FIX & RECENCY PRIORITIZATION


def test_legacy_bank_prioritization_and_auto_retirement(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    s, _ = _make_sample(vocab, tok, Kind.COMBAT, seed=21)
    edir = tmp_path / "merged"
    # Simulate live S35 at update=3100 with 1,000 legacy 4M G145 records (teacher_version=213783!)
    legacy_recs = [_rec(s, target_action=1, gain=0.14, teacher_version=213783) for _ in range(1000)]
    save_exit_shard(edir, "g145", 1, legacy_recs, vocab_hash=vocab.hash)
    # Plus 50 fresh on-policy 19M records (teacher_version=3100)
    live_recs = [_rec(s, target_action=2, gain=0.14, teacher_version=3100) for _ in range(50)]
    save_exit_shard(edir, "s35w0", 1, live_recs, vocab_hash=vocab.hash)

    # 1. Old p13 uniform mode: legacy shards are immortal and drown live 20:1 (~4.8% live share)
    buf_old = ExitBuffer(edir, capacity=8192, max_age_updates=2000, expected_vocab_hash=vocab.hash)
    buf_old.poll_and_evict(current_update=3100)
    assert len(buf_old) == 1050
    buf_old.poll_and_evict(current_update=10000)  # +6900 updates later: live evicted, 1000 G145 STILL IMMORTAL!
    assert len(buf_old) == 1000

    # 2. p13.1 v2 mode: legacy_weight=0.10 (10x live priority), retire_legacy_at=100
    buf_v2 = ExitBuffer(
        edir, capacity=8192, max_age_updates=2000, expected_vocab_hash=vocab.hash,
        recency_halflife=500, legacy_weight=0.10, retire_legacy_at=100,
    )
    buf_v2.poll_and_evict(current_update=3100)
    assert buf_v2.counts() == (50, 1000)
    rng = np.random.default_rng(42)
    live_hits = 0
    total = 0
    for _ in range(40):
        recs, flags = buf_v2.sample_with_flags(64, rng, current_update=3100)
        live_hits += sum(not f for f in flags)
        total += len(flags)
    live_share = live_hits / total
    # With n_live=50, retire_legacy_at=100 -> taper=0.5, eff_leg=0.05 -> live mass 50 vs legacy mass 50 -> ~50%!
    assert live_share > 0.35, f"expected strong live prioritization (>35%), got {live_share:.3f}"

    # 3. Once 50 more live 19M records arrive (n_live=100 >= retire_legacy_at), ALL 1,000 legacy records retire!
    save_exit_shard(edir, "s35w0", 2, [_rec(s, target_action=2, gain=0.14, teacher_version=3150) for _ in range(50)],
                    vocab_hash=vocab.hash)
    ing, ev, cor = buf_v2.poll_and_evict(current_update=3150)
    assert cor == 0 and ing == 50 and ev == 1000
    assert buf_v2.counts() == (100, 0)

    # 4. Also verify legacy shards age from ingest_update (never immortal even if retire_legacy_at=0)
    buf_age = ExitBuffer(edir, capacity=8192, max_age_updates=500, expected_vocab_hash=vocab.hash, legacy_weight=0.2)
    buf_age.poll_and_evict(current_update=3100)
    assert buf_age.counts() == (100, 1000)
    buf_age.poll_and_evict(current_update=3700)  # +600 updates > max_age_updates=500 -> ALL evicted!
    assert buf_age.counts() == (0, 0)


# =========================================================================== 3. MARGIN CAP, SOFT MPO, LEGAL MASK & PRECOND


@pytest.mark.parametrize("use_cuda", [False, pytest.param(True, marks=cuda)])
def test_precond_step_preserves_ppo_adam_and_cuda_graphs_and_margin_cap(
    vocab: Vocab, tok: Tokenizer, fake_segments: list[Segment], tmp_path: Path, use_cuda: bool
) -> None:
    device = "cuda" if use_cuda else "cpu"
    catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
    s, n_act = _make_sample(vocab, tok, Kind.COMBAT, seed=33, min_act=5)
    target_row = n_act - 1
    banned_row = 0

    torch.manual_seed(11)
    net = SpireNet(tok.layout, TINY, catalog).to(device)
    batch_1 = {k: v.to(device) for k, v in tok.collate([s]).items()}

    # Collapse target_row < 1e-3 so margin-capped ExIt must lift it cleanly
    rival = 1
    opt_pre = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _ in range(30):
        opt_pre.zero_grad(set_to_none=True)
        lg = net(batch_1)["logits"]
        lp = torch.log_softmax(lg.masked_fill(~torch.isfinite(lg), -1e30), dim=-1)
        (-lp[0, rival]).backward()
        opt_pre.step()
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net = SpireNet(tok.layout, TINY, catalog).to(device)
    net.load_state_dict(sd)

    lm = np.ones(n_act, dtype=np.bool_)
    lm[banned_row] = False
    edir = tmp_path / f"exit_v2_{device}"
    save_exit_shard(
        edir, "w0", 1,
        [_rec(s, target_action=target_row, gain=0.10, teacher_version=0, legal_mask=lm) for _ in range(16)],
        vocab_hash=vocab.hash,
    )
    # p_cap(gain=0.10) = min(0.80, 0.55 + 0.20) = 0.75
    cfg = PPOConfig(
        epochs=1, minibatch=256, micro_batch=128, lr=1e-3, seed=3, bf16=False,
        exit_coef=2.5, exit_dir=str(edir), exit_batch_size=16, exit_min_samples=8,
        exit_max_kl_guard=0.5, exit_target_cap=0.80, exit_precond=True, exit_apply_legal_mask=True,
    )
    lr = Learner(net, tok, cfg, device=device, size_fn=lambda x: tok.config.bucket(x.n_tokens),
                 fast=use_cuda, graphs=use_cuda)
    if use_cuda:
        ptrs = [p.grad.data_ptr() for p in lr._params if p.grad is not None]

    # Run 1 PPO update first to populate PPO's Adam exp_avg / exp_avg_sq, then verify _exit_step NEVER mutates them!
    m0 = lr.update(fake_segments)
    assert m0["exit_loss"] is not None and math.isfinite(m0["exit_loss"])
    assert "exit_active_frac" in m0 and m0["exit_active_frac"] > 0.0
    for k in ("exit_live_n", "exit_legacy_n", "exit_live_share", "exit_live_top1_acc", "exit_active_frac"):
        assert k in _learner_fields(m0)

    # Snapshot PPO Adam state before calling _exit_step directly
    snap_m = {id(p): st["exp_avg"].detach().clone() for p, st in lr.opt.state.items() if "exp_avg" in st}
    snap_v = {id(p): st["exp_avg_sq"].detach().clone() for p, st in lr.opt.state.items() if "exp_avg_sq" in st}
    snap_s = {id(p): st["step"].detach().clone() for p, st in lr.opt.state.items() if "step" in st}
    assert snap_m and snap_v

    for _ in range(40):
        em = lr._exit_step(lr=1e-3)
        if em["exit_active_frac"] == 0.0:
            break

    # 1. PPO Adam states MUST remain 100% bit-identical across momentum-free preconditioned _exit_step calls!
    for p, st in lr.opt.state.items():
        if id(p) in snap_m:
            assert torch.equal(st["exp_avg"], snap_m[id(p)])
            assert torch.equal(st["exp_avg_sq"], snap_v[id(p)])
            assert torch.equal(st["step"], snap_s[id(p)])
    # 2. Target probability reached p_cap (~0.75) and stopped updating (exit_grad_norm == 0.0, zero memorization!)
    assert em["exit_active_frac"] == 0.0 and em["exit_grad_norm"] == 0.0 and em["exit_step_kl"] == 0.0
    assert em["exit_target_prob_mean"] >= 0.74
    # 3. Persistent CUDA graph .grad pointers untouched
    if use_cuda:
        assert [p.grad.data_ptr() for p in lr._params if p.grad is not None] == ptrs


def test_soft_mpo_targets_and_macro_records_and_online_calibrator(
    vocab: Vocab, tok: Tokenizer, tmp_path: Path
) -> None:
    # 1. Verify non-combat macro records (CARD_REWARD, SHOP, MAP, REST) + soft K-candidate MPO targets
    macro_recs: list[ExitRecord] = []
    for kind, sd in ((Kind.CARD_REWARD, 1), (Kind.SHOP, 2), (Kind.MAP, 3), (Kind.REST, 4)):
        s, na = _make_sample(vocab, tok, kind, seed=sd, min_act=2)
        cands = (0, 1) if na == 2 else (0, 1, 2)
        scores = (0.70, 0.85) if na == 2 else (0.70, 0.85, 0.72)
        p_old = tuple(1.0 / len(cands) for _ in cands)
        macro_recs.append(_rec(s, target_action=1, gain=0.15, teacher_version=0,
                               cand_rows=cands, cand_scores=scores, cand_p_old=p_old, stage=f"MACRO-{kind.name}"))
    edir = tmp_path / "macro_shards"
    p = save_exit_shard(edir, "macro0", 1, macro_recs, vocab_hash=vocab.hash)
    loaded = load_exit_shard(p, expected_vocab_hash=vocab.hash)
    assert len(loaded) == 4 and all(len(r.cand_scores) == len(r.cand_rows) for r in loaded)

    catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
    net = SpireNet(tok.layout, TINY, catalog)
    cfg = PPOConfig(
        epochs=1, minibatch=64, micro_batch=32, lr=1e-3, seed=9, bf16=False,
        exit_coef=0.5, exit_dir=str(edir), exit_batch_size=4, exit_min_samples=4,
        exit_soft_beta=15.0, exit_precond=True, exit_apply_legal_mask=True,
    )
    lr = Learner(net, tok, cfg, device="cpu", fast=False)
    em = lr._exit_step(lr=1e-3)
    assert em["exit_buffer_n"] == 4 and em["exit_loss"] is not None and math.isfinite(em["exit_loss"])
    assert em["exit_step_kl"] is not None and em["exit_step_kl"] > 0.0

    # 2. OnlinePwinCalibrator removes systematic -20 pp Act-1 downward bias
    cal = OnlinePwinCalibrator(window=200, min_samples=25, l2=0.5)
    rng = np.random.default_rng(123)
    # True greedy win prob is sigmoid(logit + 1.1) (~82% at raw logit=0.4 where raw sigmoid is ~60%)
    for _ in range(150):
        raw_logit = float(rng.normal(0.4, 0.8))
        true_p = 1.0 / (1.0 + math.exp(-(raw_logit + 1.1)))
        cal.observe(0, raw_logit, bool(rng.random() < true_p))
    raw_at_04 = 1.0 / (1.0 + math.exp(-0.4))  # 0.5987 (~22 pp too low)
    cal_at_04 = cal.predict(0, 0.4)           # should be near sigmoid(1.5) = 0.8176
    assert abs(cal_at_04 - 0.8176) < 0.06 and cal_at_04 > raw_at_04 + 0.14

    # 3. Emergency hallway trigger works when enabled and stays off by default
    assert not should_trigger_fight(1, "Monster", 0.30, ExitTeacherConfig())[0]
    assert should_trigger_fight(1, "Monster", 0.30, ExitTeacherConfig(emergency_hallway=True))[0]
    assert not should_trigger_fight(1, "Monster", 0.60, ExitTeacherConfig(emergency_hallway=True))[0]


def test_exit_teacher_macro_defaults_triggers_and_fair_play() -> None:
    """--enable-macro defaults False (bit-identical Combat-only workers), excludes CRYSTAL/REWARD,
    and uses tuned MAP pacing (greedy_gp < 0.70, m1=6, m2=6, horizon=4)."""
    from spire.schema import EventFlag, Verb

    cfg_off = ExitTeacherConfig()
    assert cfg_off.enable_macro is False
    assert (cfg_off.macro_m1, cfg_off.macro_m2, cfg_off.macro_horizon_floors) == (6, 6, 4)
    assert cfg_off.macro_map_uncertain == pytest.approx(0.70)
    assert Kind.CRYSTAL not in MACRO_ELIGIBLE_KINDS
    assert Kind.REWARD not in MACRO_ELIGIBLE_KINDS

    # Off by default: never triggers macro
    assert not should_trigger_macro(
        kind=Kind.CARD_REWARD, kind_sub=0, room_name="Boss", event_flags=0, gold=100,
        n_unique=4, greedy_gp=0.40, cur_pwin=0.60, macro_decs_episode=0, shop_decs_visit=0, cfg=cfg_off,
    )[0]

    cfg_on = ExitTeacherConfig(enable_macro=True)
    # CRYSTAL & REWARD strictly refused even with enable_macro=True
    for bad_kind in (Kind.CRYSTAL, Kind.REWARD):
        assert not should_trigger_macro(
            kind=bad_kind, kind_sub=0, room_name="Event", event_flags=int(EventFlag.ANCIENT), gold=300,
            n_unique=4, greedy_gp=0.30, cur_pwin=0.50, macro_decs_episode=0, shop_decs_visit=0, cfg=cfg_on,
        )[0]

    # MAP disabled by default (myopic +4-floor horizon); opt-in requires greedy_gp < 0.70
    assert cfg_on.macro_enable_map is False
    assert not should_trigger_macro(
        kind=Kind.MAP, kind_sub=0, room_name="Monster", event_flags=0, gold=100,
        n_unique=3, greedy_gp=0.50, cur_pwin=0.70, macro_decs_episode=0, shop_decs_visit=0, cfg=cfg_on,
    )[0]
    assert should_trigger_macro(
        kind=Kind.MAP, kind_sub=0, room_name="Monster", event_flags=0, gold=100,
        n_unique=3, greedy_gp=0.65, cur_pwin=0.70, macro_decs_episode=0, shop_decs_visit=0,
        cfg=ExitTeacherConfig(enable_macro=True, macro_enable_map=True),
    ) == (True, "map_fork")

    # Ancient relic always triggers (below skip_pwin)
    assert should_trigger_macro(
        kind=Kind.EVENT, kind_sub=0, room_name="Event", event_flags=int(EventFlag.ANCIENT), gold=99,
        n_unique=3, greedy_gp=0.98, cur_pwin=0.70, macro_decs_episode=0, shop_decs_visit=0, cfg=cfg_on,
    ) == (True, "ancient_relic")

    # Candidate reservation works cleanly across ALL 8 macro screen kinds (SHOP, CARD_REWARD, REST, EVENT, MAP, CARD_SELECT, BUNDLE, TREASURE)
    reps_cr = [
        {"row": 0, "verb": Verb.PICK_OFFER_CARD, "verb_name": "PICK_OFFER_CARD", "prob": 0.50, "group_prob": 0.50, "logit": 2.0, "is_greedy": True},
        {"row": 1, "verb": Verb.PICK_OFFER_CARD, "verb_name": "PICK_OFFER_CARD", "prob": 0.30, "group_prob": 0.30, "logit": 1.2, "is_greedy": False},
        {"row": 2, "verb": Verb.PICK_OFFER_CARD, "verb_name": "PICK_OFFER_CARD", "prob": 0.19, "group_prob": 0.19, "logit": 0.6, "is_greedy": False},
        {"row": 3, "verb": Verb.SKIP, "verb_name": "SKIP", "prob": 0.01, "group_prob": 0.01, "logit": -2.0, "is_greedy": False},
    ]
    assert [c["row"] for c in select_macro_candidates(reps_cr, Kind.CARD_REWARD, max_k=3)] == [0, 3, 1]

    reps_shop = [
        {"row": 0, "verb": Verb.CHOOSE_OPTION, "verb_name": "CHOOSE_OPTION", "prob": 0.55, "group_prob": 0.55, "logit": 2.1, "is_greedy": True},
        {"row": 1, "verb": Verb.CHOOSE_OPTION, "verb_name": "CHOOSE_OPTION", "prob": 0.30, "group_prob": 0.30, "logit": 1.0, "is_greedy": False},
        {"row": 2, "verb": Verb.CHOOSE_OPTION, "verb_name": "CHOOSE_OPTION", "prob": 0.14, "group_prob": 0.14, "logit": 0.2, "is_greedy": False},
        {"row": 3, "verb": Verb.PROCEED, "verb_name": "PROCEED", "prob": 0.01, "group_prob": 0.01, "logit": -1.5, "is_greedy": False},
    ]
    assert [c["row"] for c in select_macro_candidates(reps_shop, Kind.SHOP, max_k=3)] == [0, 3, 1]

    reps_sel = [
        {"row": 0, "verb": Verb.SELECT_CARD, "verb_name": "SELECT_CARD", "prob": 0.60, "group_prob": 0.60, "logit": 2.0, "is_greedy": True},
        {"row": 1, "verb": Verb.SELECT_CARD, "verb_name": "SELECT_CARD", "prob": 0.38, "group_prob": 0.38, "logit": 1.0, "is_greedy": False},
        {"row": 2, "verb": Verb.CANCEL, "verb_name": "CANCEL", "prob": 0.02, "group_prob": 0.02, "logit": -1.0, "is_greedy": False},
    ]
    assert [c["row"] for c in select_macro_candidates(reps_sel, Kind.CARD_SELECT, max_k=2)] == [0, 2]

    for k_other in (Kind.REST, Kind.EVENT, Kind.MAP, Kind.BUNDLE, Kind.TREASURE):
        out = select_macro_candidates(reps_cr, k_other, max_k=3)
        assert out[0]["row"] == 0 and len(out) == 3

    # Macro emission gate: unconditional dv_after > 0 required (including Ancient Boss Relics!) + DISCARD_POTION banned
    assert passes_macro_emission_filter(is_ancient=False, gain_all=0.05, t_all=2.5, gain_s2=0.03, dv_after=0.01, cfg=cfg_on)
    assert not passes_macro_emission_filter(is_ancient=False, gain_all=0.05, t_all=2.5, gain_s2=0.03, dv_after=-0.01, cfg=cfg_on)
    assert not passes_macro_emission_filter(is_ancient=True, gain_all=0.19, t_all=4.3, gain_s2=0.22, dv_after=-0.13, cfg=cfg_on)
    assert passes_macro_emission_filter(is_ancient=True, gain_all=0.09, t_all=3.1, gain_s2=0.10, dv_after=0.06, cfg=cfg_on)
    reps_dp = [
        {"row": 0, "verb": Verb.CHOOSE_OPTION, "verb_name": "CHOOSE_OPTION", "prob": 0.90, "group_prob": 0.90, "logit": 2.0, "is_greedy": True},
        {"row": 1, "verb": Verb.DISCARD_POTION, "verb_name": "DISCARD_POTION", "prob": 0.10, "group_prob": 0.10, "logit": 0.0, "is_greedy": False},
    ]
    assert [c["row"] for c in select_macro_candidates(reps_dp, Kind.REST, max_k=4)] == [0]


def test_recv_exact_raises_eof_on_closed_child_socket() -> None:
    """When a forked Rust/C# child exits mid-rollout (e.g., _exit(14)), _recv_exact MUST raise EOFError
    so eval_candidates_on_seeds / eval_macro_candidates_on_seeds convert it to _ChildAbortedError."""
    from spire.exit_teacher import _recv_exact

    class _ClosedSock:
        def recv(self, n: int) -> bytes:
            return b""

    with pytest.raises(EOFError):
        _recv_exact(_ClosedSock(), 4)  # type: ignore[arg-type]


@pytest.mark.parametrize("use_cuda", [False, pytest.param(True, marks=cuda)])
def test_torch_policy_include_pwin_matches_eager(vocab: Vocab, tok: Tokenizer, use_cuda: bool) -> None:
    """TorchPolicy(include_pwin=True) returns pwin_logit & value matching eager net(batch) on CPU and CUDA graphs."""
    from spire.actor import TorchPolicy
    from spire.tokenize import batch_to

    device = "cuda" if use_cuda else "cpu"
    catalog = Catalog(make_synthetic_catalog(vocab, seed=7), vocab)
    torch.manual_seed(17)
    net = SpireNet(tok.layout, TINY, catalog).to(device).eval()
    gen = FrameGen(vocab)
    frames = [parse_frame(random_frame(gen, np.random.default_rng(100 + i), Kind.COMBAT, n_seats=1)) for i in range(4)]
    enc = tok.encode_batch(frames, sort=True)
    n_act = np.fromiter((frames[i].n_actions for i in (enc.order.tolist() if enc.order is not None else range(4))), np.int64, 4)

    with torch.inference_mode():
        b = batch_to(tok.collate_encoded(enc, 0, 4), torch.device(device))
        if use_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ref = net(b)
        else:
            ref = net(b)
    ref_v = ref["value"].float().cpu().numpy()
    ref_p = ref["pwin_logit"].float().cpu().numpy()

    pol_off = TorchPolicy(net, tok, mode="greedy", use_graphs=use_cuda, include_pwin=False)
    out_off = pol_off.finish(pol_off.launch_encoded(enc, 0, 4, n_act))
    assert out_off.pwin_logit is None
    pol_off.close()

    pol_on = TorchPolicy(net, tok, mode="greedy", use_graphs=use_cuda, include_pwin=True)
    out_on = pol_on.finish(pol_on.launch_encoded(enc, 0, 4, n_act))
    assert out_on.pwin_logit is not None
    tol = 2e-2 if use_cuda else 1e-5
    np.testing.assert_allclose(out_on.value, ref_v, atol=tol, rtol=tol)
    np.testing.assert_allclose(out_on.pwin_logit, ref_p, atol=tol, rtol=tol)
    np.testing.assert_allclose(out_on.logits, out_off.logits, atol=1e-5, rtol=1e-5)
    pol_on.close()


def test_macro_blindspot_probe_trigger_bounds() -> None:
    """Opt-in CARD_REWARD blind-spot probe triggers iff enabled + has_blindspot_card + probe_u < prob,
    while preserving default=0.0 behavior and macro_skip_pwin / episode cap guardrails."""
    cfg_off = ExitTeacherConfig(enable_macro=True)
    assert cfg_off.macro_blindspot_prob == 0.0
    assert should_trigger_macro(
        kind=Kind.CARD_REWARD, kind_sub=0, room_name="Monster", event_flags=0, gold=100,
        n_unique=4, greedy_gp=0.98, cur_pwin=0.85, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_off, has_blindspot_card=True, probe_u=0.01,
    ) == (False, "")

    cfg_bs = ExitTeacherConfig(enable_macro=True, macro_blindspot_prob=0.15)
    assert should_trigger_macro(
        kind=Kind.CARD_REWARD, kind_sub=0, room_name="Monster", event_flags=0, gold=100,
        n_unique=4, greedy_gp=0.98, cur_pwin=0.85, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_bs, has_blindspot_card=True, probe_u=0.10,
    ) == (True, "card_reward_blindspot")
    assert should_trigger_macro(
        kind=Kind.CARD_REWARD, kind_sub=0, room_name="Monster", event_flags=0, gold=100,
        n_unique=4, greedy_gp=0.98, cur_pwin=0.85, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_bs, has_blindspot_card=True, probe_u=0.20,
    ) == (False, "")
    assert should_trigger_macro(
        kind=Kind.CARD_REWARD, kind_sub=0, room_name="Monster", event_flags=0, gold=100,
        n_unique=4, greedy_gp=0.98, cur_pwin=0.85, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_bs, has_blindspot_card=False, probe_u=0.01,
    ) == (False, "")
    # Still respects macro_skip_pwin (0.95)
    assert should_trigger_macro(
        kind=Kind.CARD_REWARD, kind_sub=0, room_name="Monster", event_flags=0, gold=100,
        n_unique=4, greedy_gp=0.98, cur_pwin=0.96, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_bs, has_blindspot_card=True, probe_u=0.01,
    ) == (False, "")
    assert cfg_bs.combat_max_child_steps == 1200


def test_native_batch_unpack_frames() -> None:
    import struct
    from spire.native_batch import unpack_frames

    p1, p2 = b"hello", b"world123"
    wire = struct.pack("<I", len(p1)) + p1 + struct.pack("<I", len(p2)) + p2
    assert unpack_frames(wire) == (p1, p2)
    assert unpack_frames(b"") == ()
    with pytest.raises(ValueError):
        unpack_frames(b"\x05\x00\x00\x00abc")


# =========================================================================== 9. MACRO TEACHER v3 & PAIRED A/B LIFT


def test_macro_v3_defaults_and_triggers() -> None:
    """Macro v3 defaults OFF (100% backward-compatible); when ON, triggers every Ancient EVENT
    and every Act-3 REST/CARD_REWARD/funded SHOP while preserving Acts 1-2 gates and REWARD/CRYSTAL bans."""
    from spire.schema import EventFlag, Verb

    cfg_def = ExitTeacherConfig()
    assert cfg_def.macro_v3 is False
    assert cfg_def.enable_macro is False
    assert cfg_def.enable_combat_search is True
    assert cfg_def.macro_v3_max_k == 4
    assert cfg_def.macro_v3_world_stages == (16, 32, 64)
    assert cfg_def.macro_v3_min_worlds == 48
    assert cfg_def.macro_v3_min_gain == pytest.approx(0.020)
    assert cfg_def.macro_v3_min_t == pytest.approx(2.50)
    assert cfg_def.macro_v3_t_lo == pytest.approx(1.0)
    assert cfg_def.macro_v3_t_hi == pytest.approx(3.0)
    assert cfg_def.macro_v3_max_child_steps == 2500
    assert cfg_def.macro_v3_max_per_episode == 40

    # v2 behavior unchanged when macro_v3=False
    cfg_v2 = ExitTeacherConfig(enable_macro=True)
    assert should_trigger_macro(
        kind=Kind.REST, kind_sub=0, room_name="RestSite", event_flags=0, gold=100,
        n_unique=2, greedy_gp=0.99, cur_pwin=0.80, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_v2, act_idx=2,
    ) == (False, "")
    assert should_trigger_macro(
        kind=Kind.EVENT, kind_sub=0, room_name="Event", event_flags=int(EventFlag.ANCIENT), gold=100,
        n_unique=3, greedy_gp=0.99, cur_pwin=0.98, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_v2, act_idx=2,
    ) == (False, "")

    # v3 enabled standalone (even without enable_macro=True)
    cfg_v3 = ExitTeacherConfig(macro_v3=True)

    # Every Ancient / Relic EVENT triggers in Acts 1, 2, 3 even at high confidence & pwin
    for a_idx in (0, 1, 2):
        assert should_trigger_macro(
            kind=Kind.EVENT, kind_sub=0, room_name="Event", event_flags=int(EventFlag.ANCIENT), gold=0,
            n_unique=3, greedy_gp=0.99, cur_pwin=0.99, macro_decs_episode=0, shop_decs_visit=0,
            cfg=cfg_v3, act_idx=a_idx,
        ) == (True, "ancient_relic")

    # Non-Ancient EVENT keeps uncertainty & skip_pwin gates
    assert should_trigger_macro(
        kind=Kind.EVENT, kind_sub=0, room_name="Event", event_flags=0, gold=100,
        n_unique=3, greedy_gp=0.95, cur_pwin=0.80, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_v3, act_idx=2,
    ) == (False, "")
    assert should_trigger_macro(
        kind=Kind.EVENT, kind_sub=0, room_name="Event", event_flags=0, gold=100,
        n_unique=3, greedy_gp=0.70, cur_pwin=0.80, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_v3, act_idx=2,
    ) == (True, "event_uncertain")

    # Act 3 (act_idx=2): EVERY REST, CARD_REWARD, and funded SHOP triggers even at greedy_gp=0.99, cur_pwin=0.99
    assert should_trigger_macro(
        kind=Kind.REST, kind_sub=0, room_name="RestSite", event_flags=0, gold=0,
        n_unique=2, greedy_gp=0.99, cur_pwin=0.99, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_v3, act_idx=2,
    ) == (True, "act3_rest")
    assert should_trigger_macro(
        kind=Kind.CARD_REWARD, kind_sub=0, room_name="Monster", event_flags=0, gold=0,
        n_unique=4, greedy_gp=0.99, cur_pwin=0.99, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_v3, act_idx=2,
    ) == (True, "act3_card_reward")
    assert should_trigger_macro(
        kind=Kind.SHOP, kind_sub=0, room_name="Shop", event_flags=0, gold=120,
        n_unique=5, greedy_gp=0.99, cur_pwin=0.99, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_v3, act_idx=2,
    ) == (True, "act3_shop")
    # Act-3 SHOP still skips broke shops and respects per-visit cap
    assert should_trigger_macro(
        kind=Kind.SHOP, kind_sub=0, room_name="Shop", event_flags=0, gold=50,
        n_unique=5, greedy_gp=0.50, cur_pwin=0.50, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_v3, act_idx=2,
    ) == (False, "")
    assert should_trigger_macro(
        kind=Kind.SHOP, kind_sub=0, room_name="Shop", event_flags=0, gold=300,
        n_unique=5, greedy_gp=0.50, cur_pwin=0.50, macro_decs_episode=0, shop_decs_visit=2,
        cfg=cfg_v3, act_idx=2,
    ) == (False, "")

    # Acts 1-2 keep existing uncertain / Boss / Elite / skip_pwin gates
    for a_idx in (0, 1):
        assert should_trigger_macro(
            kind=Kind.REST, kind_sub=0, room_name="RestSite", event_flags=0, gold=100,
            n_unique=2, greedy_gp=0.99, cur_pwin=0.80, macro_decs_episode=0, shop_decs_visit=0,
            cfg=cfg_v3, act_idx=a_idx,
        ) == (False, "")
        assert should_trigger_macro(
            kind=Kind.CARD_REWARD, kind_sub=0, room_name="Monster", event_flags=0, gold=100,
            n_unique=4, greedy_gp=0.95, cur_pwin=0.85, macro_decs_episode=0, shop_decs_visit=0,
            cfg=cfg_v3, act_idx=a_idx,
        ) == (False, "")
        assert should_trigger_macro(
            kind=Kind.CARD_REWARD, kind_sub=0, room_name="Elite", event_flags=0, gold=100,
            n_unique=4, greedy_gp=0.95, cur_pwin=0.85, macro_decs_episode=0, shop_decs_visit=0,
            cfg=cfg_v3, act_idx=a_idx,
        ) == (True, "card_reward_elite")
        assert should_trigger_macro(
            kind=Kind.SHOP, kind_sub=0, room_name="Shop", event_flags=0, gold=200,
            n_unique=5, greedy_gp=0.95, cur_pwin=0.88, macro_decs_episode=0, shop_decs_visit=0,
            cfg=cfg_v3, act_idx=a_idx,
        ) == (False, "")

    # REWARD & CRYSTAL strictly refused even in Act 3; episode cap & n_unique<=1 respected
    for banned in (Kind.REWARD, Kind.CRYSTAL):
        assert banned not in MACRO_ELIGIBLE_KINDS
        assert should_trigger_macro(
            kind=banned, kind_sub=0, room_name="Boss", event_flags=int(EventFlag.ANCIENT), gold=500,
            n_unique=4, greedy_gp=0.20, cur_pwin=0.40, macro_decs_episode=0, shop_decs_visit=0,
            cfg=cfg_v3, act_idx=2,
        ) == (False, "")
    assert should_trigger_macro(
        kind=Kind.REST, kind_sub=0, room_name="RestSite", event_flags=0, gold=100,
        n_unique=2, greedy_gp=0.50, cur_pwin=0.50, macro_decs_episode=39, shop_decs_visit=0,
        cfg=cfg_v3, act_idx=2,
    )[0] is True
    assert should_trigger_macro(
        kind=Kind.REST, kind_sub=0, room_name="RestSite", event_flags=0, gold=100,
        n_unique=2, greedy_gp=0.50, cur_pwin=0.50, macro_decs_episode=40, shop_decs_visit=0,
        cfg=cfg_v3, act_idx=2,
    ) == (False, "")
    assert should_trigger_macro(
        kind=Kind.REST, kind_sub=0, room_name="RestSite", event_flags=0, gold=100,
        n_unique=1, greedy_gp=1.0, cur_pwin=0.50, macro_decs_episode=0, shop_decs_visit=0,
        cfg=cfg_v3, act_idx=2,
    ) == (False, "")

    # Candidate cap <= 4 & DISCARD_POTION excluded
    reps = [
        {"row": 0, "verb": Verb.PICK_OFFER_CARD, "verb_name": "PICK_OFFER_CARD", "prob": 0.50, "group_prob": 0.50, "logit": 1.5, "is_greedy": True},
        {"row": 1, "verb": Verb.DISCARD_POTION, "verb_name": "DISCARD_POTION", "prob": 0.25, "group_prob": 0.25, "logit": 1.0, "is_greedy": False},
        {"row": 2, "verb": Verb.PICK_OFFER_CARD, "verb_name": "PICK_OFFER_CARD", "prob": 0.12, "group_prob": 0.12, "logit": 0.5, "is_greedy": False},
        {"row": 3, "verb": Verb.PICK_OFFER_CARD, "verb_name": "PICK_OFFER_CARD", "prob": 0.08, "group_prob": 0.08, "logit": 0.2, "is_greedy": False},
        {"row": 4, "verb": Verb.SKIP, "verb_name": "SKIP", "prob": 0.03, "group_prob": 0.03, "logit": -0.5, "is_greedy": False},
        {"row": 5, "verb": Verb.PICK_OFFER_CARD, "verb_name": "PICK_OFFER_CARD", "prob": 0.02, "group_prob": 0.02, "logit": -1.0, "is_greedy": False},
    ]
    cands = select_macro_candidates(reps, Kind.CARD_REWARD, cfg_v3.macro_v3_max_k)
    assert len(cands) == 4
    assert [c["row"] for c in cands] == [0, 4, 2, 3]
    assert all(c["verb"] != Verb.DISCARD_POTION for c in cands)


def test_macro_v3_deep_horizon_and_scoring() -> None:
    """Verify Macro v3 Act-3 A10 Boss-B door horizon, Act-3 A9 terminal rollout, Acts 1-2 next-act door,
    raw-vs-calibrated scoring split, and 2500-step safety backstop."""
    from types import SimpleNamespace
    from spire.schema import G

    n_gvals = max(int(x) for x in G) + 1

    def _mk_fr(*, act: int, bw: int, in_combat: int, floor: int = 10, is_obs: bool = True) -> object:
        gv = np.zeros(n_gvals, dtype=np.int64)
        gv[G.ACT_INDEX] = act
        gv[G.BOSSES_WON] = bw
        gv[G.IN_COMBAT] = in_combat
        gv[G.TOTAL_FLOOR] = floor
        return SimpleNamespace(is_obs=is_obs, gvals=gv)

    cfg_v3 = ExitTeacherConfig(macro_v3=True)
    cfg_v2 = ExitTeacherConfig(enable_macro=True)
    assert _macro_step_cap(cfg_v3) == 2500
    assert _macro_step_cap(cfg_v2) == 1440

    # Act 3 A10: stops ONLY at Boss B's exact door (act=2, bw>=3, in_combat=1)
    assert is_macro_v3_horizon_reached(_mk_fr(act=2, bw=2, in_combat=1, floor=48), start_act=2, ascension=10) is False  # Boss A combat
    assert is_macro_v3_horizon_reached(_mk_fr(act=2, bw=3, in_combat=0, floor=48), start_act=2, ascension=10) is False  # Post-Boss-A REWARD/MAP
    assert is_macro_v3_horizon_reached(_mk_fr(act=2, bw=3, in_combat=1, floor=49), start_act=2, ascension=10) is True   # Boss B Turn-1 door!
    assert is_macro_v3_horizon_reached(_mk_fr(act=2, bw=2, in_combat=1, floor=48, is_obs=False), start_act=2, ascension=10) is True

    # Act 3 A9: rolls to terminal / _is_final_boss_win_obs (returns False on obs)
    assert is_macro_v3_horizon_reached(_mk_fr(act=2, bw=2, in_combat=1, floor=48), start_act=2, ascension=9) is False
    assert is_macro_v3_horizon_reached(_mk_fr(act=2, bw=3, in_combat=1, floor=49), start_act=2, ascension=9) is False

    # Acts 1-2: stop at first non-combat decision of next Act (NOT mid-combat!)
    assert is_macro_v3_horizon_reached(_mk_fr(act=0, bw=0, in_combat=0, floor=10), start_act=0, ascension=10) is False
    assert is_macro_v3_horizon_reached(_mk_fr(act=1, bw=1, in_combat=1, floor=18), start_act=0, ascension=10) is False
    assert is_macro_v3_horizon_reached(_mk_fr(act=1, bw=1, in_combat=0, floor=17), start_act=0, ascension=10) is True
    assert is_macro_v3_horizon_reached(_mk_fr(act=1, bw=1, in_combat=0, floor=30), start_act=1, ascension=10) is False
    assert is_macro_v3_horizon_reached(_mk_fr(act=2, bw=2, in_combat=1, floor=35), start_act=1, ascension=10) is False
    assert is_macro_v3_horizon_reached(_mk_fr(act=2, bw=2, in_combat=0, floor=34), start_act=1, ascension=10) is True

    # Scoring split: raw pwin at Act-3 A10 Boss-B door (sharp Brier 0.03), calibrated pwin on Acts 1-2 next-act door
    cal = OnlinePwinCalibrator(window=100, min_samples=10, l2=1.0)
    for i in range(40):
        # Train Act-2 calibrator with strong positive shift so predict(2, 0.0) >> 0.5
        cal.observe(2, -0.2 + 0.01 * i, True if i >= 4 else False)
    cal_pred_at_zero = cal.predict(2, 0.0)
    assert cal_pred_at_zero > 0.70

    vh = {"pwin": 0.50, "pwin_logit": 0.0, "value": 0.45}
    fr_boss_b = _mk_fr(act=2, bw=3, in_combat=1, floor=49)
    fr_act3_door = _mk_fr(act=2, bw=2, in_combat=0, floor=34)
    # Act-3 A10 Boss-B door uses RAW pwin even with calibrator present
    assert _score_macro_horizon_frame(
        fr_boss_b, vh, start_act=2, ascension=10, cfg=cfg_v3, calibrator=cal
    ) == pytest.approx(0.50)
    # Act-2 -> Act-3 transition uses CALIBRATED pwin when calibrator present
    assert _score_macro_horizon_frame(
        fr_act3_door, vh, start_act=1, ascension=10, cfg=cfg_v3, calibrator=cal
    ) == pytest.approx(cal_pred_at_zero)
    assert _score_macro_horizon_frame(
        fr_act3_door, vh, start_act=1, ascension=10, cfg=cfg_v3, calibrator=None
    ) == pytest.approx(0.50)


def test_macro_v3_adaptive_schedule_and_emission_filter() -> None:
    """Verify adaptive 16 -> 32 -> 64 CRN expansion logic and strict >= 48 worlds / +2pp / t>=2.5 / dv>0 gate."""
    cfg = ExitTeacherConfig(macro_v3=True)

    # Clear losers & noise stop immediately at 16
    assert should_expand_macro_v3_stage(mean_d=-0.05, t_stat=-2.2, n_worlds=16, max_worlds=64, cfg=cfg) is False
    assert should_expand_macro_v3_stage(mean_d=0.0, t_stat=0.0, n_worlds=16, max_worlds=64, cfg=cfg) is False
    assert should_expand_macro_v3_stage(mean_d=0.015, t_stat=0.95, n_worlds=16, max_worlds=64, cfg=cfg) is False
    assert should_expand_macro_v3_stage(mean_d=0.015, t_stat=1.0, n_worlds=16, max_worlds=64, cfg=cfg) is False
    assert should_expand_macro_v3_stage(mean_d=float("nan"), t_stat=2.0, n_worlds=16, max_worlds=64, cfg=cfg) is False

    # Undecided band (1.0 < |t| < 3.0) expands 16 -> 32 -> 64
    assert should_expand_macro_v3_stage(mean_d=0.018, t_stat=1.8, n_worlds=16, max_worlds=64, cfg=cfg) is True
    assert should_expand_macro_v3_stage(mean_d=0.022, t_stat=2.3, n_worlds=32, max_worlds=64, cfg=cfg) is True
    assert should_expand_macro_v3_stage(mean_d=0.022, t_stat=2.3, n_worlds=64, max_worlds=64, cfg=cfg) is False

    # Huge switch (+8pp, t=4.5 or +inf) still confirms up to >= 48 worlds before emitting!
    assert should_expand_macro_v3_stage(mean_d=0.08, t_stat=4.5, n_worlds=16, max_worlds=64, cfg=cfg) is True
    assert should_expand_macro_v3_stage(mean_d=0.08, t_stat=math.inf, n_worlds=32, max_worlds=64, cfg=cfg) is True
    assert should_expand_macro_v3_stage(mean_d=0.08, t_stat=4.5, n_worlds=48, max_worlds=64, cfg=cfg) is False

    # Sharply small effect (< +2pp, t >= 3.0) stops early without wasting 32/64 worlds
    assert should_expand_macro_v3_stage(mean_d=0.012, t_stat=3.6, n_worlds=16, max_worlds=64, cfg=cfg) is False

    # Strict emission gate: ALL FOUR required (n_worlds >= 48, gain >= 0.02, t >= 2.5, dv_after > 0.0)
    assert passes_macro_v3_emission_filter(n_worlds=48, gain_all=0.020, t_all=2.50, dv_after=0.001, cfg=cfg) is True
    assert passes_macro_v3_emission_filter(n_worlds=64, gain_all=0.045, t_all=3.20, dv_after=0.020, cfg=cfg) is True
    assert passes_macro_v3_emission_filter(n_worlds=32, gain_all=0.080, t_all=5.00, dv_after=0.050, cfg=cfg) is False
    assert passes_macro_v3_emission_filter(n_worlds=64, gain_all=0.0199, t_all=3.50, dv_after=0.020, cfg=cfg) is False
    assert passes_macro_v3_emission_filter(n_worlds=64, gain_all=0.030, t_all=2.49, dv_after=0.020, cfg=cfg) is False
    assert passes_macro_v3_emission_filter(n_worlds=64, gain_all=0.050, t_all=4.00, dv_after=0.0, cfg=cfg) is False
    assert passes_macro_v3_emission_filter(n_worlds=64, gain_all=0.050, t_all=4.00, dv_after=-0.01, cfg=cfg) is False


def test_ab_paired_lift_event_and_cli_flags() -> None:
    """Verify exact paired A/B lift McNemar telemetry math and CLI flag registration."""
    # Zero-discordant tie
    ev0 = format_ab_pair_event(
        worker_id="w0", pair_idx=1, run_seed="SEED00000001", version=123,
        off_stats={"won": True, "floor": 49, "b3a": True, "wall_s": 0.9},
        on_stats={"won": True, "floor": 49, "b3a": True, "emitted": 2, "macro_emitted": 1, "wall_s": 14.2},
        cum_off=1, cum_on=1, cum_gained=0, cum_lost=0,
    )
    assert ev0["event"] == "ab_pair"
    assert ev0["diff"] == 0
    assert ev0["cum_lift_pp"] == 0.0
    assert ev0["cum_z"] == 0.0
    assert ev0["b3a_off"] is True and ev0["b3a_on"] is True
    assert ev0["emitted"] == 2 and ev0["macro_emitted"] == 1

    # 100 pairs: 8 gained, 2 lost -> +6.0 pp, z = 6 / sqrt(10) = 1.897
    ev1 = format_ab_pair_event(
        worker_id="w0", pair_idx=100, run_seed="SEED00000100", version=125,
        off_stats={"won": False, "floor": 49, "b3a": True, "wall_s": 1.1},
        on_stats={"won": True, "floor": 49, "b3a": True, "emitted": 3, "macro_emitted": 2, "wall_s": 18.5},
        cum_off=80, cum_on=86, cum_gained=8, cum_lost=2,
    )
    assert ev1["diff"] == 1
    assert ev1["cum_lift_pp"] == pytest.approx(6.0, abs=1e-3)
    assert ev1["cum_z"] == pytest.approx(6.0 / math.sqrt(10.0), abs=1e-3)
    assert ev1["wall_s_off"] == 1.1 and ev1["wall_s_on"] == 18.5

    # CLI --help includes all Macro v3 & paired A/B lift flags
    env = dict(os.environ, PYTHONPATH=str(P13V2_PY))
    r = subprocess.run(
        [sys.executable, "-B", "-m", "spire.exit_teacher", "--help"],
        capture_output=True, text=True, check=True, env=env,
    )
    for flag in (
        "--macro-v3",
        "--macro-v3-max-k",
        "--macro-v3-min-worlds",
        "--macro-v3-min-gain",
        "--macro-v3-min-t",
        "--macro-v3-max-child-steps",
        "--no-combat-search",
        "--ab-paired-lift",
        "--no-shards",
    ):
        assert flag in r.stdout, f"missing CLI flag {flag} in --help"
