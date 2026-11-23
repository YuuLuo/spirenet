"""Verification suite for py-stage-p12 (all 5 features + OFF = bit-identical parity with py-stage-p11)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy, UniformPolicy
from spire.gym import PracticeConfig, fork_group_id
from spire.hlgauss_migration import migrate_checkpoint, migrate_hlgauss_optimizer, migrate_hlgauss_weights
from spire.ipc import decode_segment, pack_segment
from spire.learner import Learner, PPOConfig
from spire.model import (
    OUTPUT_KEYS,
    PRESETS,
    GraphRunner,
    ModelConfig,
    SpireNet,
    export_weights,
    hl_gauss_probs,
    model_config_dict,
)
from spire.proto import parse_frame
from spire.rewards import Facts, RewardConfig, RoomIds, episode_targets
from spire.rollout import (
    EpisodeMeta,
    GAEConfig,
    RolloutConfig,
    RolloutStore,
    Segment,
    SegmentEnd,
    SegmentRing,
    apply_rloo,
    build_batch,
    gae,
)
from spire.schema import DECISION_KINDS, Kind, MsgType, Result
from spire.synth import FakeHost, FakeHostConfig, FrameGen, random_frame
from spire.tokenize import Sample, SampleBlock, Tokenizer, TokenizerConfig, batch_to
from spire.train import (
    LearnerSection,
    ModelSection,
    Preparer,
    TrainConfig,
    build,
    check_confident,
    check_practice,
    checkpoint_extra,
    config_dict,
    load_config,
)
from spire.vocab import Catalog, Vocab

P12_PY = Path(__file__).resolve().parents[1]
P11_PY = P12_PY.parents[1] / "subsamp-cc" / "py"
TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def tok(vocab: Vocab) -> Tokenizer:
    return Tokenizer(vocab, TokenizerConfig(token_buckets=(20, 28, 40, 64, 80, 96, 128, 192, 256, 384)))


@pytest.fixture(scope="module")
def frames(gen: FrameGen) -> list[bytes]:
    rng = np.random.default_rng(123)
    return [random_frame(gen, rng, k, n_seats=1) for k in DECISION_KINDS for _ in range(2)]


@pytest.fixture(scope="module")
def fake_segments(vocab: Vocab, tok: Tokenizer) -> list[Segment]:
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=64), record_prog=True)
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False)) for i in range(12)]
    Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
          ActorConfig(max_batch=32), store=store).run(max_decisions=3000)
    assert sum(len(s) for s in segs) > 1500
    return segs


# =========================================================================== 0. OFF = BIT-IDENTICAL TO P11


_PARITY_PROBE = r"""
import hashlib, json
import numpy as np
import torch
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.ipc import pack_segment, decode_segment
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, PRESETS, SpireNet, model_config_dict
from spire.rewards import RoomIds
from spire.rollout import GAEConfig, RolloutConfig, RolloutStore, gae
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.train import TrainConfig, build, config_dict
from spire.vocab import Vocab, make_synthetic_catalog, make_synthetic_vocab

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

vocab = Vocab(make_synthetic_vocab())
catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab) if False else None
from spire.vocab import Catalog
catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
tok = Tokenizer(vocab)

cfg_off = build(TrainConfig, {"run_name": "parity", "run_root": "/tmp/parity_runs"})
cfg_prac = build(TrainConfig, {"run_name": "parity-p", "run_root": "/tmp/parity_runs", "actor": {"actors": 2},
                               "ppo": {"gae": {"macro_micro": True}}, "practice": {"actors": 1}})

torch.manual_seed(42)
mcfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
net = SpireNet(tok.layout, mcfg, catalog)
init_sd_keys = list(net.state_dict().keys())
init_sd_hash = sha(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in net.state_dict().values()))

segs = []
store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=40, hashes=False)) for i in range(4)]
Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
      ActorConfig(max_batch=16), store=store).run(max_decisions=600)
# zero timestamps so IPC blob hashes are deterministic across runs
for s in segs:
    s.t_first = 100.0
    s.t_last = 101.0

blobs_hash = sha(b"".join(pack_segment(s) for s in segs))
decoded = [decode_segment(pack_segment(s)) for s in segs]

batch = tok.collate(list(decoded[0].samples)[:8])
net.eval()
with torch.no_grad():
    fwd = net(batch)
fwd_keys = sorted(fwd.keys())
fwd_hash = sha(b"".join(fwd[k].detach().cpu().contiguous().numpy().tobytes() for k in fwd_keys))

gae_cfg = GAEConfig(gamma=0.999, lam=0.95, macro_micro=True, lam_macro=0.95, lam_micro=0.95, stitch_segments=True)
adv, ret = gae(decoded, gae_cfg)
gae_hash = sha(adv.tobytes() + ret.tobytes())

torch.manual_seed(7)
net2 = SpireNet(tok.layout, mcfg, catalog)
ppo_cfg = PPOConfig(epochs=2, minibatch=128, micro_batch=64, lr=1e-4, seed=11, bf16=False, gae=gae_cfg)
lr = Learner(net2, tok, ppo_cfg, device="cpu", fast=False)
metrics = lr.update(decoded)
for k in ("time_prep", "time_train", "samples_per_s", "lag_s"):
    metrics.pop(k, None)
st = lr.state()
st_model_hash = sha(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in st["model"].values()))
st_ema_hash = sha(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in st["ema"].values()))

out = {
    "config_off": config_dict(cfg_off),
    "config_prac": config_dict(cfg_prac),
    "model_config_default": model_config_dict(ModelConfig()),
    "model_config_4m": model_config_dict(PRESETS["4m"]),
    "ppo_to_dict": ppo_cfg.to_dict(),
    "init_sd_keys": init_sd_keys,
    "init_sd_hash": init_sd_hash,
    "blobs_hash": blobs_hash,
    "fwd_keys": fwd_keys,
    "fwd_hash": fwd_hash,
    "gae_hash": gae_hash,
    "metrics": metrics,
    "ckpt_top_keys": sorted(st.keys()),
    "ckpt_counter_keys": sorted(st["counters"].keys()),
    "ckpt_model_keys": list(st["model"].keys()),
    "st_model_hash": st_model_hash,
    "st_ema_hash": st_ema_hash,
}
print(json.dumps(out, sort_keys=True))
"""


def test_off_is_bit_identical_to_p11() -> None:
    """With all p12 options at defaults, config_dict, model_config_dict, PPOConfig.to_dict, state_dict,
    forward outputs, IPC blobs, GAE outputs, and 1 full PPO update match py-stage-p11 bit-for-bit."""
    if not P11_PY.is_dir():
        pytest.skip(f"reference p11 tree missing: {P11_PY}")
    results = []
    for tree in (P11_PY, P12_PY):
        env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"}
        r = subprocess.run([sys.executable, "-B", "-c", _PARITY_PROBE], env=env, capture_output=True, text=True,
                           timeout=120, cwd=str(tree))
        assert r.returncode == 0, f"{tree} failed:\n{r.stderr}"
        results.append(json.loads(r.stdout.strip().splitlines()[-1]))
    p11, p12 = results
    assert p11.keys() == p12.keys()
    for k in p11:
        assert p12[k] == p11[k], f"OFF parity mismatch on {k}:\np11={p11[k]}\np12={p12[k]}"


# =========================================================================== 1. MACRO-PROTECTED SUBSAMPLING


def test_confident_exempt_kinds_config_and_selection(vocab: Vocab, tmp_path: Path) -> None:
    off = build(TrainConfig, {"run_name": "x"})
    assert off.learner.confident_exempt_kinds == ()
    assert "confident_exempt_kinds" not in config_dict(off)["learner"]

    on = build(TrainConfig, {"run_name": "x", "learner": {
        "confident_keep": 0.2, "confident_tau": 0.9, "confident_exempt_kinds": [1, 5, 6, 7, 8]}})
    check_confident(on)
    d = config_dict(on)
    assert d["learner"]["confident_exempt_kinds"] == [1, 5, 6, 7, 8]
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert load_config(p).learner.confident_exempt_kinds == (1, 5, 6, 7, 8)

    with pytest.raises(AssertionError, match="confident_exempt_kinds"):
        check_confident(build(TrainConfig, {"run_name": "x", "learner": {"confident_exempt_kinds": [99]}}))

    # Selection unit check: 100% of exempt kinds kept, non-exempt confident dropped at (1 - keep)
    exempt_kinds = (int(Kind.MAP), int(Kind.CARD_REWARD), int(Kind.EVENT), int(Kind.SHOP), int(Kind.REST))
    n = 100_000
    rng = np.random.default_rng(0)
    # 50% confident COMBAT, 25% confident exempt macro kinds, 25% non-confident COMBAT
    kinds = np.full(n, int(Kind.COMBAT), np.int8)
    kinds[50_000:75_000] = rng.choice(exempt_kinds, size=25_000)
    probs = np.empty(n, np.float64)
    probs[:75_000] = rng.uniform(0.96, 1.0, 75_000)
    probs[75_000:] = rng.uniform(0.1, 0.8, 25_000)

    class _MockRB:
        def __init__(self) -> None:
            self.logp = np.log(probs).astype(np.float32)
            self.kinds = kinds

        def __len__(self) -> int:
            return n

    tok_simple = Tokenizer(vocab)
    lr = Learner(SpireNet(tok_simple.layout, TINY), tok_simple, PPOConfig(), device="cpu", fast=False,
                 confident_keep=0.15, confident_tau=0.95, confident_seed=7, confident_exempt_kinds=exempt_kinds)
    rows, conf = lr._kept_rows(_MockRB())  # type: ignore[arg-type]
    assert conf is not None and int(conf.sum()) == 75_000
    kept = np.zeros(n, bool)
    kept[rows] = True
    assert kept[50_000:75_000].all(), "every exempted macro row must be kept"
    assert kept[75_000:].all(), "every non-confident row must be kept"
    combat_conf_rate = kept[:50_000].mean()
    assert abs(combat_conf_rate - 0.15) < 4 * math.sqrt(0.15 * 0.85 / 50_000), combat_conf_rate


@cuda
@pytest.mark.parametrize("mode", ["eager", "fast", "graphs"])
def test_confident_exempt_kinds_learner_modes(tok: Tokenizer, fake_segments: list[Segment], mode: str) -> None:
    exempt = (int(Kind.MAP), int(Kind.CARD_REWARD), int(Kind.EVENT), int(Kind.SHOP), int(Kind.REST))
    # Force high behaviour logp so many rows (both combat and macro) are confident
    segs = [copy.copy(s) for s in fake_segments]
    for s in segs:
        s.logp = np.full(len(s), math.log(0.98), np.float32)
    rb = build_batch(segs, GAEConfig(), 0, with_samples=False)
    exempt_mask = np.isin(rb.kinds, exempt)
    assert exempt_mask.any() and (~exempt_mask).any()

    torch.manual_seed(0)
    net = SpireNet(tok.layout, TINY)
    cfg = PPOConfig(epochs=1, minibatch=512, micro_batch=256, micro_tokens=4096, seed=3)
    fast, graphs = (mode != "eager"), (mode == "graphs")
    lr = Learner(net, tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                 fast=fast, graphs=graphs, confident_keep=0.2, confident_tau=0.95, confident_seed=9,
                 confident_exempt_kinds=exempt)
    rows, conf = lr._kept_rows(rb)
    assert conf is not None
    kept_set = set(rows.tolist())
    assert set(np.flatnonzero(exempt_mask).tolist()) <= kept_set
    m = lr.update(segs)
    assert m["conf_exempt_kept"] == int((conf & exempt_mask).sum()) > 0
    assert m["kept"] == len(rows) and m["conf_dropped"] == len(rb) - len(rows) > 0
    assert m["conf_dropped"] <= m["conf_taken"] - m["conf_exempt_kept"]
    assert np.isfinite([m["loss_pi"], m["loss_v"], m["grad_norm"]]).all()


# =========================================================================== 2. PRACTICE RLOO ADVANTAGE


def _make_rloo_seg(env_id: int, uid: int, fork_group: int, rewards: list[float], values: list[float],
                   combat_ids: list[int], *, start: int = 0, fork_start: int = 0,
                   end: SegmentEnd = SegmentEnd.TERMINAL, boot: float = 0.0,
                   sample: Sample | None = None) -> Segment:
    n = len(rewards)
    samples = SampleBlock.from_samples([sample] * n) if sample is not None else [None] * n
    return Segment(
        meta=EpisodeMeta(env_id, uid, (2,), 10, -1, fork_group, fork_start),
        start=start, samples=samples,
        actions=np.zeros(n, np.int64), logp=np.full(n, -0.5, np.float32),
        values=np.asarray(values, np.float32), kinds=np.full(n, int(Kind.COMBAT), np.int8),
        n_legal=np.full(n, 3, np.int16), policy_version=np.zeros(n, np.int64),
        end=end, bootstrap_value=boot,
        rewards=np.asarray(rewards, np.float32), combat_id=np.asarray(combat_ids, np.int32),
        win=np.zeros(n, np.float32), win_mask=np.zeros(n, bool),
        floors_left=np.zeros(n, np.float32), hp_post=np.zeros(n, np.float32), hp_mask=np.zeros(n, bool),
    )


def test_rloo_config_and_group_id(tmp_path: Path) -> None:
    assert PracticeConfig().rloo_alpha == 0.0 and not PracticeConfig().rloo_by_pair
    with pytest.raises(AssertionError, match="rloo_alpha"):
        PracticeConfig(actors=1, rloo_alpha=1.5)
    with pytest.raises(AssertionError, match="rloo_by_pair"):
        PracticeConfig(actors=1, rloo_alpha=0.5, rloo_by_pair=True, paired_seeds=False)
    with pytest.raises(AssertionError, match="practice.actors > 0"):
        check_practice(build(TrainConfig, {"run_name": "x", "practice": {"actors": 0, "rloo_alpha": 0.5}}))

    cfg = build(TrainConfig, {"run_name": "x", "actor": {"actors": 2},
                              "ppo": {"gae": {"macro_micro": True}},
                              "practice": {"actors": 1, "paired_seeds": True, "rloo_alpha": 0.5, "rloo_by_pair": True}})
    check_practice(cfg)
    d = config_dict(cfg)
    assert d["practice"]["rloo_alpha"] == 0.5 and d["practice"]["rloo_by_pair"] is True
    assert build(TrainConfig, d).practice == cfg.practice

    # fork_group_id properties
    g0 = fork_group_id(12345, tag=1, pair=0)
    g1 = fork_group_id(12345, tag=1, pair=1)
    g2 = fork_group_id(12345, tag=2, pair=0)
    g_other_actor = fork_group_id(99999, tag=1, pair=0)
    assert all(0 < g < (1 << 63) for g in (g0, g1, g2, g_other_actor))
    assert len({g0, g1, g2, g_other_actor}) == 4
    assert fork_group_id(12345, tag=1, pair=0) == g0


def test_rloo_exact_math_2_and_4_siblings_and_game_child(tok: Tokenizer, frames: list[bytes]) -> None:
    gae_cfg = GAEConfig(gamma=0.99, lam=0.95, macro_micro=True, lam_macro=0.95, lam_micro=0.9, stitch_segments=True)
    decay = gae_cfg.gamma * gae_cfg.lam_micro

    # --- Case A: 2 siblings (m = 2), fork_start = 10, alpha = 0.5
    s1 = _make_rloo_seg(100, 1, 777, [0.1, 0.2, 0.7], [0.4, 0.5, 0.6], [2, 2, 2], start=10, fork_start=10)
    s2 = _make_rloo_seg(101, 2, 777, [0.0, -0.1], [0.4, 0.2], [2, 2], start=10, fork_start=10)
    # Singleton group (m = 1): must remain untouched
    s_single = _make_rloo_seg(102, 3, 888, [0.5, 0.5], [0.3, 0.4], [0, 0], start=0, fork_start=0)
    # Plain episode (fork_group = 0): must remain untouched
    s_plain = _make_rloo_seg(0, 4, 0, [0.2, 0.3], [0.1, 0.2], [0, 0], start=0, fork_start=0)

    rb0 = build_batch([s1, s2, s_single, s_plain], gae_cfg, 0, with_samples=False)
    adv_before = rb0.advantages.copy()
    ret_before = rb0.returns.copy()

    # Hand-compute door returns G_1, G_2 and shifts
    G1 = float(adv_before[0]) + float(rb0.values[0])
    G2 = float(adv_before[3]) + float(rb0.values[3])
    alpha = 0.5
    d1 = alpha * (float(rb0.values[0]) - G2)
    d2 = alpha * (float(rb0.values[3]) - G1)

    ng, nc, sstd = apply_rloo(rb0, gae_cfg, alpha)
    assert (ng, nc) == (1, 2)
    assert sstd == pytest.approx(float(np.std([d1, d2])), rel=1e-6, abs=1e-7)
    # Value head targets (returns) MUST be untouched bit-for-bit!
    assert np.array_equal(rb0.returns, ret_before)
    # Sibling 1 (steps 0, 1, 2 -> t = 0, 1, 2)
    for t in range(3):
        assert rb0.advantages[t] == pytest.approx(adv_before[t] + d1 * (decay ** t), abs=1e-6)
    # Sibling 2 (steps 3, 4 -> t = 0, 1)
    for t in range(2):
        assert rb0.advantages[3 + t] == pytest.approx(adv_before[3 + t] + d2 * (decay ** t), abs=1e-6)
    # Singleton and plain untouched
    assert np.array_equal(rb0.advantages[5:], adv_before[5:])

    # --- Case B: 4 siblings (m = 4), alpha = 1.0, plus GAME child continuing into macro + fight 2
    # Sibling 0 is a GAME child: fight 0 (2 steps, cid=1), macro (1 step, cid=-1), fight 1 (2 steps, cid=2)
    g_segs = [
        _make_rloo_seg(200, 10, 999, [0.1, 0.9, 0.0, 0.2, 0.3], [0.5, 0.6, 0.7, 0.8, 0.9], [1, 1, -1, 2, 2],
                       start=5, fork_start=5),
        _make_rloo_seg(201, 11, 999, [0.0, 0.2], [0.5, 0.3], [1, 1], start=5, fork_start=5),
        _make_rloo_seg(202, 12, 999, [0.3, 0.3], [0.5, 0.4], [1, 1], start=5, fork_start=5),
        _make_rloo_seg(203, 13, 999, [-0.2, 0.0], [0.5, 0.1], [1, 1], start=5, fork_start=5),
    ]
    rb4 = build_batch(g_segs, gae_cfg, 0, with_samples=False)
    adv4_0 = rb4.advantages.copy()
    ret4_0 = rb4.returns.copy()
    door_pos = [0, 5, 7, 9]
    Gs = [float(adv4_0[p]) + float(rb4.values[p]) for p in door_pos]
    ng4, nc4, _ = apply_rloo(rb4, gae_cfg, 1.0)
    assert (ng4, nc4) == (1, 4)
    assert np.array_equal(rb4.returns, ret4_0)
    # At alpha=1.0 with identical door V_0=0.5, shifted door advantages match exact G_i - G_{-i} and sum to 0
    for i, p in enumerate(door_pos):
        g_loo = (sum(Gs) - Gs[i]) / 3.0
        assert rb4.advantages[p] == pytest.approx(Gs[i] - g_loo, abs=1e-6)
    assert sum(float(rb4.advantages[p]) for p in door_pos) == pytest.approx(0.0, abs=1e-5)
    # GAME child's post-fight macro (idx 2) and second fight (idx 3, 4) MUST NOT be shifted!
    assert np.array_equal(rb4.advantages[2:5], adv4_0[2:5])
    # GAME child's forked fight step t=1 (idx 1) WAS shifted by delta_0 * decay
    delta0 = 0.5 - (sum(Gs) - Gs[0]) / 3.0
    assert rb4.advantages[1] == pytest.approx(adv4_0[1] + delta0 * decay, abs=1e-6)

    # --- Case C: IPC round-trip of fork_group & fork_start, and Learner metrics
    sample = tok.encode(parse_frame(frames[0]))
    seg_a = _make_rloo_seg(300, 20, 55555, [0.1, 0.8], [0.4, 0.5], [1, 1], start=7, fork_start=7, sample=sample)
    seg_b = _make_rloo_seg(301, 21, 55555, [0.0, -0.1], [0.4, 0.1], [1, 1], start=7, fork_start=7, sample=sample)
    dec_a = decode_segment(pack_segment(seg_a))
    dec_b = decode_segment(pack_segment(seg_b))
    assert dec_a.meta.fork_group == 55555 and dec_a.meta.fork_start == 7
    assert dec_b.meta.fork_group == 55555 and dec_b.meta.fork_start == 7

    lr = Learner(SpireNet(tok.layout, TINY), tok, PPOConfig(epochs=1, minibatch=8, micro_batch=8, gae=gae_cfg),
                 device="cpu", fast=False, rloo_alpha=0.5)
    m = lr.update([dec_a, dec_b])
    assert m["rloo_groups"] == 1 and m["rloo_children"] == 2 and m["rloo_shift_std"] > 0.0


def test_rloo_fork_episode_prefix_and_missing_door_segment(vocab: Vocab) -> None:
    """RolloutStore.fork_episode sets fork_group and fork_start=len(facts); a child whose door segment was
    in an earlier batch is not falsely treated as a door step."""
    rooms = RoomIds.from_vocab(vocab)
    segs: list[Segment] = []
    store = RolloutStore(segs.append, rooms, config=RolloutConfig(segment_len=3), rewards=RewardConfig(gamma=1.0))
    f_map = Facts(int(MsgType.OBS), int(Result.IN_PROGRESS), int(Kind.MAP), 1, 0, False, 1.0, 2, False)
    f_fight = Facts(int(MsgType.OBS), int(Result.IN_PROGRESS), int(Kind.COMBAT), 2, 0, True, 0.9, 3, False)
    f_term = Facts(int(MsgType.TERMINAL), int(Result.LOSS), int(Kind.TERMINAL), 2, 0, True, 0.0, 0, False)
    store.begin_episode(0, (2,), 0)
    store.add_step(0, f_map, None, 0, -0.5, 0.2, 0)
    store.add_step(0, f_map, None, 0, -0.5, 0.2, 0)
    pt = store.fork_point(0)
    for k in range(2):
        meta = store.fork_episode(pt, 10 + k, game=False, fork_group=1234567)
        assert meta.fork_group == 1234567 and meta.fork_start == 2
        for _ in range(5):
            store.add_step(10 + k, f_fight, None, 0, -0.5, 0.3, 0)
        store.end_episode(10 + k, f_term)
    store.abort_episode(0, "shutdown")
    # Each child produced 2 segments: start=2 (len 3) and start=5 (len 2)
    assert [s.start for s in segs] == [2, 5, 2, 5]
    # Take ONLY the 2nd segments (start=5): neither has fork_step == 0 -> 0 RLOO groups!
    rb_tail = build_batch([segs[1], segs[3]], GAEConfig(gamma=1.0, macro_micro=True), 0, with_samples=False)
    assert (rb_tail.fork_step >= 3).all()
    assert apply_rloo(rb_tail, GAEConfig(gamma=1.0, macro_micro=True), 1.0) == (0, 0, 0.0)


# =========================================================================== 3. SMOOTH SHAPING DECAY


def test_shaping_fade_schedule_ipc_and_resume(vocab: Vocab, tok: Tokenizer, fake_segments: list[Segment],
                                              tmp_path: Path) -> None:
    # Config validation & omission when off
    off_ppo = PPOConfig()
    assert "shaping_end_scale" not in off_ppo.to_dict() and "shaping_fade_updates" not in off_ppo.to_dict()
    with pytest.raises(AssertionError, match="shaping_fade_updates > 0"):
        PPOConfig(shaping_end_scale=0.1, shaping_fade_updates=0)
    with pytest.raises(AssertionError, match="shaping_end_scale"):
        PPOConfig(shaping_end_scale=1.5, shaping_fade_updates=10)

    # Verify rewards_prog populated and round-trips through IPC; omitted when record_prog=False
    assert all(s.rewards_prog is not None for s in fake_segments)
    assert any(float(np.abs(s.rewards_prog).sum()) > 0 for s in fake_segments)  # type: ignore[arg-type]
    dec = decode_segment(pack_segment(fake_segments[0]))
    assert dec.rewards_prog is not None and np.array_equal(dec.rewards_prog, fake_segments[0].rewards_prog)

    seg_no_prog = copy.copy(fake_segments[0])
    seg_no_prog.rewards_prog = None
    assert decode_segment(pack_segment(seg_no_prog)).rewards_prog is None

    # Exact linear schedule over 6 updates: fade=4, end_scale=0.2 -> [1.0, 0.8, 0.6, 0.4, 0.2, 0.2]
    gae_cfg = GAEConfig(gamma=0.999, lam=0.95)
    cfg = PPOConfig(epochs=1, minibatch=512, micro_batch=256, seed=0, gae=gae_cfg,
                    shaping_end_scale=0.2, shaping_fade_updates=4)
    torch.manual_seed(0)
    lr = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu", fast=False)
    want_scales = [1.0, 0.8, 0.6, 0.4, 0.2, 0.2]
    raw_r = np.concatenate([s.rewards for s in fake_segments]).astype(np.float64)
    prog_r = np.concatenate([s.rewards_prog for s in fake_segments]).astype(np.float64)  # type: ignore[misc]
    for step_idx, want in enumerate(want_scales):
        prep = lr.prepare(fake_segments, lr.version)
        assert prep.shaping_scale == pytest.approx(want, abs=1e-12)
        expected_r = (raw_r - (1.0 - want) * prog_r).astype(np.float32)
        assert np.allclose(prep.rb.rewards, expected_r, atol=1e-6)
        _, exp_ret = gae(fake_segments, gae_cfg, rewards=expected_r)
        assert np.allclose(prep.rb.returns, exp_ret, atol=1e-6)
        m = lr.update_prepared(prep)
        assert m["shaping_scale"] == pytest.approx(want, abs=1e-12)
        assert lr.shaping_start_update == 0

    # Resume test 1: old checkpoint (no shaping) at updates=3 resumed into shaping_fade_updates=4, end_scale=0.0
    base_cfg = PPOConfig(epochs=1, minibatch=512, micro_batch=256, seed=0, gae=gae_cfg)
    lr_old = Learner(SpireNet(tok.layout, TINY), tok, base_cfg, device="cpu", fast=False)
    for _ in range(3):
        lr_old.update(fake_segments)
    old_ckpt = tmp_path / "old.pt"
    lr_old.save(old_ckpt)

    fade_cfg = PPOConfig(epochs=1, minibatch=512, micro_batch=256, seed=0, gae=gae_cfg,
                         shaping_end_scale=0.0, shaping_fade_updates=4)
    lr_res = Learner(SpireNet(tok.layout, TINY), tok, fade_cfg, device="cpu", fast=False)
    lr_res.load(old_ckpt, allow_config_change=True)
    assert lr_res.updates == 3 and lr_res.shaping_start_update is None
    assert lr_res.update(fake_segments)["shaping_scale"] == pytest.approx(1.0)
    assert lr_res.shaping_start_update == 3
    assert lr_res.update(fake_segments)["shaping_scale"] == pytest.approx(0.75)

    # Resume test 2: save mid-fade (updates=5, start=3) and reload -> continues at 0.50, 0.25, 0.0
    mid_ckpt = tmp_path / "mid.pt"
    lr_res.save(mid_ckpt)
    lr_res2 = Learner(SpireNet(tok.layout, TINY), tok, fade_cfg, device="cpu", fast=False)
    lr_res2.load(mid_ckpt)
    assert lr_res2.updates == 5 and lr_res2.shaping_start_update == 3
    for want in (0.5, 0.25, 0.0, 0.0):
        assert lr_res2.update(fake_segments)["shaping_scale"] == pytest.approx(want)


@cuda
def test_shaping_fade_pipeline_prepare_matches_sequential(tok: Tokenizer, fake_segments: list[Segment]) -> None:
    gae_cfg = GAEConfig(gamma=0.999, lam=0.95)
    cfg = PPOConfig(epochs=1, minibatch=512, micro_batch=256, micro_tokens=4096, seed=5, gae=gae_cfg,
                    shaping_end_scale=0.0, shaping_fade_updates=3)
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    seq = Learner(copy.deepcopy(base), tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                  fast=True, graphs=True)
    pip = Learner(copy.deepcopy(base), tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                  fast=True, graphs=True)
    ring = SegmentRing(1 << 20)
    total_samples = sum(len(s) for s in fake_segments)
    for _ in range(4):
        for s in fake_segments:
            ring.put(s)
    prep = Preparer(pip, ring, total_samples, max_staleness=None)
    try:
        seq_scales, pip_scales = [], []
        for _ in range(4):
            seq_scales.append(seq.update(fake_segments)["shaping_scale"])
            got = prep.get(timeout=30.0)
            assert got is not None
            pip_scales.append(pip.update_prepared(got.prepared)["shaping_scale"])
            prep.done()
    finally:
        prep.close()
    assert seq_scales == pip_scales == [pytest.approx(x) for x in (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)]
    # Compare parameter deltas (bf16 + CUDA atomic scatter order causes tiny ~1e-4 rounding differences across runs)
    p0 = [p.detach().float().cpu() for p in base.parameters()]
    d_seq = [p.detach().float().cpu() - b for p, b in zip(seq.model.parameters(), p0)]
    d_pip = [p.detach().float().cpu() - b for p, b in zip(pip.model.parameters(), p0)]
    num = sum(float((a - b).abs().sum()) for a, b in zip(d_seq, d_pip))
    den = sum(float(a.abs().sum()) for a in d_seq)
    assert num / max(den, 1e-12) < 0.02, num / max(den, 1e-12)


# =========================================================================== 4. HL-GAUSS DISTRIBUTIONAL VALUE HEAD


def test_hlgauss_probs_and_forward_and_migration(tok: Tokenizer, catalog: Catalog, frames: list[bytes],
                                                 tmp_path: Path) -> None:
    # Config omission when off, presence when on
    assert "value_bins" not in model_config_dict(TINY)
    hl_cfg = replace(TINY, value_bins=51, value_vmin=-0.5, value_vmax=2.5, value_sigma=0.05)
    d = model_config_dict(hl_cfg)
    assert (d["value_bins"], d["value_vmin"], d["value_vmax"], d["value_sigma"]) == (51, -0.5, 2.5, 0.05)

    # ModelSection config round-trip
    tc = build(TrainConfig, {"run_name": "hl", "model": {"preset": "4m", "value_bins": 51, "value_sigma": 0.06}})
    assert tc.model.config().value_bins == 51 and tc.model.config().value_sigma == 0.06
    assert build(TrainConfig, config_dict(tc)).model.config() == tc.model.config()

    # Exact HL-Gauss probability properties
    edges = torch.linspace(-0.5, 2.5, 52)
    centers = 0.5 * (edges[:-1] + edges[1:])
    targets = torch.tensor([-2.0, -0.5, 0.0, 0.75, 1.5, 2.5, 10.0])
    p = hl_gauss_probs(targets, edges, -0.5, 2.5, 0.05)
    assert p.shape == (7, 51) and (p >= 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(7), atol=1e-6, rtol=0)
    exp_v = (p * centers).sum(-1)
    for idx in (2, 3, 4):  # interior targets: expectation matches target closely
        assert abs(float(exp_v[idx]) - float(targets[idx])) < 1e-4

    # Forward & gradient check
    samples = [tok.encode(parse_frame(f)) for f in frames]
    batch = tok.collate(samples)
    torch.manual_seed(0)
    net = SpireNet(tok.layout, hl_cfg, catalog)
    assert net.value_head is None and net.val_dist is not None
    assert set(export_weights(net)) == {n for n, _ in net.named_parameters()}  # buffers non-persistent
    out = net(batch)
    assert out["value"].shape == (len(samples),) and out["value_logits"].shape == (len(samples), 51)
    torch.testing.assert_close(out["value"], (F.softmax(out["value_logits"], -1) * net.val_centers).sum(-1), atol=1e-6, rtol=0)

    # Loss gradient reaches val_dist and trunk
    tgt = torch.linspace(-0.2, 1.8, len(samples))
    p_tgt = hl_gauss_probs(tgt, net.val_edges, hl_cfg.value_vmin, hl_cfg.value_vmax, hl_cfg.value_sigma)
    loss = -(p_tgt * F.log_softmax(out["value_logits"], -1)).sum(-1).mean()
    loss.backward()
    for name, param in net.val_dist.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name
        assert torch.isfinite(param.grad).all(), name

    # Migration accuracy: train scalar value_head slightly so values span [-0.2, 1.8], then migrate
    torch.manual_seed(3)
    scalar_net = SpireNet(tok.layout, TINY, catalog)
    with torch.no_grad():
        scalar_net.value_head[1][2].weight.mul_(5.0)  # widen output range across [-0.2, 1.6]
        scalar_net.value_head[1][2].bias.fill_(0.6)
    scalar_net.eval()
    with torch.no_grad():
        ref_out = scalar_net(batch)
    # Verify non-trivial spread of scalar values
    assert float(ref_out["value"].min()) > -0.35 and float(ref_out["value"].max()) < 2.35

    hl_net = SpireNet(tok.layout, hl_cfg, catalog).eval()
    migrated_sd = migrate_hlgauss_weights(scalar_net.state_dict(), hl_net)
    hl_net.load_state_dict(migrated_sd)
    with torch.no_grad():
        mig_out = hl_net(batch)
    # Non-value outputs MUST be bit-identical
    for k in ("logits", "pwin_logit", "floors_pred", "hp_pred"):
        assert torch.equal(mig_out[k], ref_out[k]), k
    # Migrated expected value matches old scalar value within < 0.01!
    max_val_diff = float((mig_out["value"] - ref_out["value"]).abs().max())
    assert max_val_diff < 0.01, f"migrated HL-Gauss value diff {max_val_diff} >= 0.01"

    # Full checkpoint & optimizer migration test
    ppo_cfg = PPOConfig(epochs=1, minibatch=8, micro_batch=8, lr=1e-3, seed=0)
    lr_scalar = Learner(copy.deepcopy(scalar_net), tok, ppo_cfg, device="cpu", fast=False)
    segs = [_make_rloo_seg(0, 1, 0, [0.1] * 8, [0.5] * 8, [0] * 8, sample=samples[0])]
    lr_scalar.update(segs)
    ckpt_in = tmp_path / "scalar.pt"
    ckpt_out = tmp_path / "hl.pt"
    tc_scalar = build(TrainConfig, {"run_name": "m", "model": {"override": TINY}})
    lr_scalar.save(ckpt_in, extra=checkpoint_extra(tc_scalar, tok.vocab, catalog))

    migrate_checkpoint(ckpt_in, ckpt_out, value_bins=51, value_vmin=-0.5, value_vmax=2.5, value_sigma=0.05)
    ck_old = torch.load(ckpt_in, map_location="cpu", weights_only=True)
    ck_new = torch.load(ckpt_out, map_location="cpu", weights_only=True)
    assert ck_new["extra"]["model_config"]["value_bins"] == 51
    # Optimizer: non-val_dist states preserved bit-for-bit, val_dist states zeroed at step 0
    lr_hl = Learner(SpireNet(tok.layout, hl_cfg, catalog), tok, ppo_cfg, device="cpu", fast=False)
    lr_hl.load(ckpt_out)
    names_by_id = {id(p): n for n, p in lr_hl.model.named_parameters()}
    opt_sd = lr_hl.opt.state_dict()
    val_checked = other_checked = 0
    for old_g, new_g, live_g in zip(ck_old["optimizer"]["param_groups"], opt_sd["param_groups"], lr_hl.opt.param_groups):
        for oi, ni, param in zip(old_g["params"], new_g["params"], live_g["params"]):
            nm = names_by_id[id(param)]
            if nm.startswith("val_dist."):
                assert float(opt_sd["state"][ni]["step"]) == 0.0
                assert float(opt_sd["state"][ni]["exp_avg"].abs().sum()) == 0.0
                assert float(opt_sd["state"][ni]["exp_avg_sq"].abs().sum()) == 0.0
                val_checked += 1
            elif oi in ck_old["optimizer"]["state"]:
                assert torch.equal(opt_sd["state"][ni]["step"], ck_old["optimizer"]["state"][oi]["step"])
                assert torch.equal(opt_sd["state"][ni]["exp_avg"], ck_old["optimizer"]["state"][oi]["exp_avg"])
                other_checked += 1
    assert val_checked == 4 and other_checked > 20

    # Also test Learner.load(..., migrate_hlgauss=True) directly
    lr_hl2 = Learner(SpireNet(tok.layout, hl_cfg, catalog), tok, ppo_cfg, device="cpu", fast=False)
    lr_hl2.load(ckpt_in, migrate_hlgauss=True)
    for k in lr_hl.model.state_dict():
        assert torch.equal(lr_hl.model.state_dict()[k], lr_hl2.model.state_dict()[k]), k


@cuda
def test_hlgauss_cuda_graphs_and_compile_and_actor_runner(tok: Tokenizer, fake_segments: list[Segment]) -> None:
    import importlib.util
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)

    hl_cfg = replace(TINY, value_bins=51, value_vmin=-0.5, value_vmax=2.5, value_sigma=0.05)
    cfg = PPOConfig(epochs=2, minibatch=512, micro_batch=256, micro_tokens=4096, lr=1e-4, seed=2)
    torch.manual_seed(0)
    base = SpireNet(tok.layout, hl_cfg)

    has_triton = importlib.util.find_spec("triton") is not None
    ls = {
        "eager": Learner(copy.deepcopy(base), tok, cfg, device="cuda",
                         size_fn=lambda s: tok.config.bucket(s.n_tokens), fast=False, graphs=False),
        "graphs": Learner(copy.deepcopy(base), tok, cfg, device="cuda",
                          size_fn=lambda s: tok.config.bucket(s.n_tokens), fast=True, graphs=True),
    }
    if has_triton:
        ls["compiled"] = Learner(copy.deepcopy(base), tok, cfg, device="cuda",
                                 size_fn=lambda s: tok.config.bucket(s.n_tokens), fast=True, graphs=True,
                                 compile_model=True)
    ms = {k: lr.update(fake_segments) for k, lr in ls.items()}
    assert ms["graphs"]["stepper"]["captures"] > 0
    for k in ("graphs", *(("compiled",) if has_triton else ())):
        assert ms[k]["nonfinite_skipped"] == 0
        for metric in ("loss_pi", "loss_v", "entropy", "approx_kl", "grad_norm"):
            assert ms[k][metric] == pytest.approx(ms["eager"][metric], rel=0.05, abs=5e-3), (k, metric)
    # Verify torch.compile (fullgraph=True, dynamic=True) on forward + HL-Gauss loss with zero graph breaks
    samples = [s for seg in fake_segments[:2] for s in seg.samples][:16]
    batch = batch_to(tok.collate(samples), "cuda")
    tgt = torch.linspace(-0.2, 1.8, len(samples), device="cuda")
    net_c = copy.deepcopy(base).cuda().train()
    backend = "inductor" if has_triton else "aot_eager"
    compiled_fn = torch.compile(
        lambda b, y: -(hl_gauss_probs(y, net_c.val_edges, hl_cfg.value_vmin, hl_cfg.value_vmax, hl_cfg.value_sigma)
                       * F.log_softmax(net_c(b)["value_logits"], dim=-1)).sum(-1).mean(),
        dynamic=True, fullgraph=True, backend=backend,
    )
    loss_c = compiled_fn(batch, tgt)
    loss_c.backward()
    assert torch.isfinite(loss_c) and all(torch.isfinite(p.grad).all() for p in net_c.val_dist.parameters())  # type: ignore[union-attr]

    # Replay graphs on a 2nd update
    c_before = ls["graphs"].stepper.captures  # type: ignore[union-attr]
    m2 = ls["graphs"].update(fake_segments)
    assert m2["nonfinite_skipped"] == 0 and np.isfinite(m2["loss_v"])
    assert ls["graphs"].stepper.captures >= c_before  # type: ignore[union-attr]

    # Actor GraphRunner & TorchPolicy with HL-Gauss
    net_eval = copy.deepcopy(ls["graphs"].model).eval()
    pol_g = TorchPolicy(copy.deepcopy(net_eval), tok, mode="greedy", use_graphs=True)
    pol_e = TorchPolicy(copy.deepcopy(net_eval), tok, mode="greedy", use_graphs=False)
    n_act = np.array([s.n_actions for s in samples], np.int64)
    og = pol_g.finish(pol_g.launch(samples, n_act))
    oe = pol_e.finish(pol_e.launch(samples, n_act))
    assert np.allclose(og.value, oe.value, atol=5e-3)
    assert np.array_equal(og.actions, oe.actions)


# =========================================================================== 5. CHARACTER-ROUTED FFN EXPERTS


def test_char_experts_zero_init_parity_and_gradient_isolation(tok: Tokenizer, catalog: Catalog,
                                                              frames: list[bytes]) -> None:
    # Config & presets
    assert "char_experts" not in model_config_dict(TINY)
    moe_cfg = replace(TINY, char_experts=8, char_expert_ff=64)
    d = model_config_dict(moe_cfg)
    assert d["char_experts"] == 8 and d["char_expert_ff"] == 64
    with pytest.raises(AssertionError, match="char_experts and char_expert_ff"):
        replace(TINY, char_experts=8, char_expert_ff=0)

    # Check 19m and 19m-moe presets build cleanly
    n19 = SpireNet(tok.layout, PRESETS["19m"], catalog).n_params
    n19m = SpireNet(tok.layout, PRESETS["19m-moe"], catalog).n_params
    assert n19 > 14e6 and n19m > n19, (n19, n19m)

    # Zero-init exact identity parity against non-expert model
    samples = [tok.encode(parse_frame(f)) for f in frames]
    batch = tok.collate(samples)
    torch.manual_seed(0)
    plain = SpireNet(tok.layout, TINY, catalog).eval()
    moe = SpireNet(tok.layout, moe_cfg, catalog).eval()
    # Copy every shared weight from plain into moe (keeping moe's zero-init exp_w2 / exp_b2)
    sd = moe.state_dict()
    for k, v in plain.state_dict().items():
        sd[k] = v.clone()
    moe.load_state_dict(sd)
    with torch.no_grad():
        o_plain = plain(batch)
        o_moe = moe(batch)
    for k in OUTPUT_KEYS:
        assert torch.equal(o_plain[k], o_moe[k]), f"zero-init MoE differs from plain on {k}"

    # Routing check: perturbing expert 2 affects ONLY samples with char_id == 2
    b_mix = copy.deepcopy(batch)
    b_mix["cond"] = b_mix["cond"].clone()
    b_mix["cond"][:, 2] = torch.tensor([2 if i % 2 == 0 else 3 for i in range(len(samples))], dtype=torch.int64)
    with torch.no_grad():
        before_mix = moe(b_mix)
        for blk in moe.blocks:
            blk.exp_w2[2].normal_(0.0, 0.1)
            blk.exp_b2[2].normal_(0.0, 0.1)
        after_mix = moe(b_mix)
    even = slice(0, None, 2)
    odd = slice(1, None, 2)
    assert not torch.allclose(before_mix["value"][even], after_mix["value"][even])
    for k in OUTPUT_KEYS:
        assert torch.equal(before_mix[k][odd], after_mix[k][odd]), f"expert 2 leaked into char 3 on {k}"

    # Gradient isolation: updating character A (char_id=2) NEVER touches character B's (char_id=3) expert weights!
    moe.train()
    opt = torch.optim.AdamW(moe.parameters(), lr=1e-3, weight_decay=0.0)
    b_char_a = copy.deepcopy(batch)
    b_char_a["cond"] = b_char_a["cond"].clone()
    b_char_a["cond"][:, 2] = 2

    exp_before = [{name: p.detach().clone() for name, p in blk.named_parameters() if name.startswith("exp_")}
                  for blk in moe.blocks]
    # Run 2 steps so exp_w2 becomes non-zero on step 1 AND exp_w1 gets gradients on step 2
    for step in range(2):
        opt.zero_grad()
        out = moe(b_char_a)
        loss = out["value"].pow(2).mean() + out["pwin_logit"].pow(2).mean()
        loss.backward()
        for blk in moe.blocks:
            for name in ("exp_w1", "exp_b1", "exp_w2", "exp_b2"):
                g = getattr(blk, name).grad
                assert g is not None
                # Experts != 2 MUST have exact zero gradient!
                for e in range(8):
                    if e != 2:
                        assert float(g[e].abs().sum()) == 0.0, (step, name, e)
            assert float(blk.exp_w2.grad[2].abs().sum()) > 0.0
            if step == 1:
                assert float(blk.exp_w1.grad[2].abs().sum()) > 0.0
        opt.step()

    for blk, prev in zip(moe.blocks, exp_before):
        for name in ("exp_w1", "exp_b1", "exp_w2", "exp_b2"):
            cur = getattr(blk, name).detach()
            assert not torch.equal(cur[2], prev[name][2]), f"active expert 2 did not update ({name})"
            for e in range(8):
                if e != 2:
                    assert torch.equal(cur[e], prev[name][e]), f"inactive expert {e} was mutated ({name})"


@cuda
def test_char_experts_and_combined_hlgauss_cuda_graphs(tok: Tokenizer, fake_segments: list[Segment]) -> None:
    import importlib.util
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    has_triton = importlib.util.find_spec("triton") is not None
    backend = "inductor" if has_triton else "aot_eager"

    # Test both char_experts alone AND combined char_experts + HL-Gauss on CUDA graphs + torch.compile
    for mcfg in (replace(TINY, char_experts=8, char_expert_ff=64),
                 replace(TINY, char_experts=8, char_expert_ff=64, value_bins=51)):
        cfg = PPOConfig(epochs=2, minibatch=512, micro_batch=256, micro_tokens=4096, lr=1e-4, seed=4)
        torch.manual_seed(0)
        base = SpireNet(tok.layout, mcfg)
        eager = Learner(copy.deepcopy(base), tok, cfg, device="cuda",
                        size_fn=lambda s: tok.config.bucket(s.n_tokens), fast=False, graphs=False)
        graphs = Learner(copy.deepcopy(base), tok, cfg, device="cuda",
                         size_fn=lambda s: tok.config.bucket(s.n_tokens), fast=True, graphs=True)
        me, mg = eager.update(fake_segments), graphs.update(fake_segments)
        assert mg["stepper"]["captures"] > 0 and mg["nonfinite_skipped"] == 0
        for key in ("loss_pi", "loss_v", "entropy", "approx_kl", "grad_norm"):
            assert mg[key] == pytest.approx(me[key], rel=0.05, abs=5e-3), (mcfg, key)
        if has_triton:
            compiled = Learner(copy.deepcopy(base), tok, cfg, device="cuda",
                               size_fn=lambda s: tok.config.bucket(s.n_tokens), fast=True, graphs=True,
                               compile_model=True)
            mc = compiled.update(fake_segments)
            assert mc["stepper"]["captures"] > 0 and mc["nonfinite_skipped"] == 0
            for key in ("loss_pi", "loss_v", "entropy", "approx_kl", "grad_norm"):
                assert mc[key] == pytest.approx(me[key], rel=0.05, abs=5e-3), (mcfg, "compiled", key)

        # Verify torch.compile(fullgraph=True, dynamic=True) forward+backward has zero graph breaks
        samples = [s for seg in fake_segments[:2] for s in seg.samples][:16]
        batch = batch_to(tok.collate(samples), "cuda")
        net_c = copy.deepcopy(base).cuda().train()
        c_fwd = torch.compile(net_c, dynamic=True, fullgraph=True, backend=backend)
        o_c = c_fwd(batch)
        o_c["value"].pow(2).mean().backward()
        assert torch.isfinite(o_c["value"]).all()

        # Actor GraphRunner replay with char_experts (+ HL-Gauss)
        net_eval = copy.deepcopy(graphs.model).eval()
        runner = GraphRunner(net_eval, tok)
        g_out = runner(samples)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            e_out = net_eval(batch_to(tok.collate(samples), "cuda"))
        a = max(s.n_actions for s in samples)
        torch.testing.assert_close(g_out["logits"][:, :a], e_out["logits"][:, :a], atol=5e-3, rtol=1e-2)
        torch.testing.assert_close(g_out["value"], e_out["value"], atol=5e-3, rtol=1e-2)
