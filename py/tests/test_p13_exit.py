"""Verification suite for py-stage-p13 (Async ExIt Search Teacher + Bounded ExIt Aux Loss)."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.cycle import CycleBan
from spire.evalpanel import load_policy_net
from spire.exit_buffer import (
    SHARD_FORMAT_VERSION,
    ExitBuffer,
    ExitRecord,
    load_exit_shard,
    save_exit_shard,
)
from spire.exit_teacher import (
    ExitTeacherConfig,
    LiveWeightPoller,
    make_scout_seed,
    passes_emission_filter,
    select_candidates,
    should_trigger_decision,
    should_trigger_fight,
)
from spire.gym import SeedStream
from spire.learner import FileWeightPublisher, Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import Kind, Verb
from spire.synth import FakeHost, FakeHostConfig, FrameGen, random_frame
from spire.tokenize import Sample, Tokenizer, TokenizerConfig
from spire.train import TrainConfig, _learner_fields, build, checkpoint_extra, config_dict, load_config
from spire.vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab

P13_PY = Path(__file__).resolve().parents[1]
P12_PY = P13_PY.parents[1] / "p12" / "py"
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
    ).run(max_decisions=1200)
    assert sum(len(s) for s in segs) >= 600
    return segs


def _make_combat_sample(vocab: Vocab, tok: Tokenizer, seed: int = 77) -> tuple[Sample, int]:
    gen = FrameGen(vocab)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        raw = random_frame(gen, rng, Kind.COMBAT, n_seats=1)
        fr = parse_frame(raw)
        if fr.n_actions >= 5:
            enc = tok.encode_batch([fr], sort=True)
            s = enc.samples()[0]
            return s, fr.n_actions
    raise AssertionError("could not generate combat frame with >= 5 actions")


def _make_record(
    sample: Sample,
    target_action: int = 1,
    *,
    gain: float = 0.12,
    teacher_version: int = 0,
    p_old: float = 1e-5,
    legal_mask: np.ndarray | None = None,
) -> ExitRecord:
    n_act = sample.n_actions
    lm = np.ones(n_act, dtype=np.bool_) if legal_mask is None else legal_mask
    return ExitRecord(
        sample=sample,
        target_action=target_action,
        legal_mask=lm,
        cand_rows=(0, target_action),
        greedy_action=0,
        gain=gain,
        gain_s2=gain,
        t_stat=2.8,
        dv_after=0.10,
        p_old=p_old,
        p_group_old=p_old,
        teacher_version=teacher_version,
        stage="A3-BB",
        verb="END_TURN",
        char="SILENT",
        ascension=10,
        ts=time.time(),
    )


# =========================================================================== 1. OFF = BIT-IDENTICAL TO P12


_PARITY_PROBE = r"""
import hashlib, json
import numpy as np
import torch
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet, model_config_dict
from spire.rewards import RoomIds
from spire.rollout import GAEConfig, RolloutConfig, RolloutStore
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.train import TrainConfig, build, config_dict
from spire.vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

vocab = Vocab(make_synthetic_vocab())
catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
tok = Tokenizer(vocab)

cfg_off = build(TrainConfig, {"run_name": "p13-off", "run_root": "/tmp/p13_parity"})
mcfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)

segs = []
store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=40, hashes=False)) for i in range(4)]
Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
      ActorConfig(max_batch=16), store=store).run(max_decisions=500)

torch.manual_seed(13)
net = SpireNet(tok.layout, mcfg, catalog)
init_sd_keys = list(net.state_dict().keys())
init_sd_hash = sha(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in net.state_dict().values()))
gae_cfg = GAEConfig(gamma=0.999, lam=0.95, macro_micro=True, lam_macro=0.95, lam_micro=0.95, stitch_segments=True)
ppo_cfg = PPOConfig(epochs=2, minibatch=128, micro_batch=64, lr=1e-4, seed=11, bf16=False, gae=gae_cfg)
lr = Learner(net, tok, ppo_cfg, device="cpu", fast=False)
metrics = lr.update(segs)
for k in ("time_prep", "time_train", "samples_per_s", "lag_s"):
    metrics.pop(k, None)
st = lr.state()
out = {
    "config_off": config_dict(cfg_off),
    "ppo_to_dict": ppo_cfg.to_dict(),
    "model_config_default": model_config_dict(ModelConfig()),
    "init_sd_keys": init_sd_keys,
    "init_sd_hash": init_sd_hash,
    "metrics": metrics,
    "ckpt_top_keys": sorted(st.keys()),
    "ckpt_counter_keys": sorted(st["counters"].keys()),
    "ckpt_model_keys": list(st["model"].keys()),
    "st_model_hash": sha(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in st["model"].values())),
    "st_ema_hash": sha(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for v in st["ema"].values())),
}
print(json.dumps(out, sort_keys=True))
"""


def test_off_is_bit_identical_to_p12() -> None:
    """With exit_coef=0.0 (default OFF), config_dict, PPOConfig.to_dict, state_dict, metrics,
    and trained model/EMA weights match alt/p12/py bit-for-bit."""
    if not P12_PY.is_dir():
        pytest.skip(f"reference p12 tree missing: {P12_PY}")
    results = []
    for tree in (P12_PY, P13_PY):
        env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"}
        r = subprocess.run(
            [sys.executable, "-B", "-c", _PARITY_PROBE],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(tree),
        )
        assert r.returncode == 0, f"{tree} failed:\n{r.stderr}"
        results.append(json.loads(r.stdout.strip().splitlines()[-1]))
    p12, p13 = results
    assert p12.keys() == p13.keys()
    for k in p12:
        assert p13[k] == p12[k], f"OFF parity mismatch on {k}:\np12={p12[k]}\np13={p13[k]}"


# =========================================================================== 2. CONFIG VALIDATION & TELEMETRY


def test_exit_config_validation_and_roundtrip(tmp_path: Path) -> None:
    off = build(TrainConfig, {"run_name": "x"})
    assert off.ppo.exit_coef == 0.0 and off.ppo.exit_dir == ""
    d_off = config_dict(off)
    for k in list(d_off["ppo"]):
        assert not k.startswith("exit_"), f"default OFF leaked key {k}"

    with pytest.raises(AssertionError, match="exit_dir"):
        PPOConfig(exit_coef=0.05)
    with pytest.raises(AssertionError, match="exit_coef"):
        PPOConfig(exit_coef=-0.01, exit_dir="x")
    with pytest.raises(AssertionError, match="exit_weight"):
        PPOConfig(exit_coef=0.05, exit_dir="x", exit_weight_min=3.0, exit_weight_max=1.0)
    with pytest.raises(AssertionError, match="exit_max_kl_guard"):
        PPOConfig(exit_coef=0.05, exit_dir="x", exit_max_kl_guard=0.0)

    on = build(
        TrainConfig,
        {
            "run_name": "x",
            "ppo": {
                "exit_coef": 0.04,
                "exit_dir": "exit_shards",
                "exit_batch_size": 32,
                "exit_min_samples": 16,
                "exit_buffer_size": 2048,
                "exit_max_age_updates": 1500,
                "exit_min_gain": 0.03,
                "exit_gain_scale": 0.10,
                "exit_weight_min": 0.5,
                "exit_weight_max": 2.5,
                "exit_max_kl_guard": 0.03,
            },
        },
    )
    d_on = config_dict(on)
    assert d_on["ppo"]["exit_coef"] == 0.04
    assert d_on["ppo"]["exit_dir"] == "exit_shards"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(d_on), encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.ppo == on.ppo

    # Verify _learner_fields passes all required ExIt telemetry keys into metrics.jsonl
    sample_m = {
        "exit_loss": 0.1,
        "exit_buffer_n": 100,
        "exit_ingested": 10,
        "exit_evicted": 2,
        "exit_target_prob_mean": 0.25,
        "exit_top1_acc": 0.5,
        "exit_step_ms": 1.2,
    }
    kept = _learner_fields(sample_m)
    assert kept == sample_m


# =========================================================================== 3. SHARD ATOMICITY, EVICTION & RESILIENCE


def test_exit_shard_atomicity_pruning_and_resilience(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    s, n_act = _make_combat_sample(vocab, tok)
    exit_dir = tmp_path / "shards"

    # Missing dir handled gracefully
    buf = ExitBuffer(exit_dir, capacity=5, max_age_updates=100, min_gain=0.025, expected_vocab_hash=vocab.hash)
    assert buf.poll_and_evict(current_update=10) == (0, 0, 0)
    assert len(buf) == 0

    # Atomic write + exact round-trip under weights_only=True
    rec0 = _make_record(s, target_action=2, gain=0.15, teacher_version=10)
    p1 = save_exit_shard(exit_dir, "w0", 1, [rec0], max_shards=3, vocab_hash=vocab.hash)
    assert p1.exists()
    assert not list(exit_dir.glob("*.tmp")), "temp file left behind after atomic replace"
    loaded = load_exit_shard(p1, expected_vocab_hash=vocab.hash)
    assert len(loaded) == 1
    r = loaded[0]
    assert r.target_action == 2 and r.gain == pytest.approx(0.15) and r.teacher_version == 10
    assert np.array_equal(r.sample.types, s.types)
    assert np.array_equal(r.sample.raw, s.raw)
    assert np.array_equal(r.sample.rel, s.rel)
    assert np.array_equal(r.sample.edges, s.edges)
    assert np.array_equal(r.sample.cond, s.cond)
    assert np.array_equal(r.sample.history, s.history)
    assert r.sample.counts == s.counts

    # Pruning beyond max_shards=3
    for seq in range(2, 6):
        time.sleep(0.015)
        save_exit_shard(exit_dir, "w0", seq, [_make_record(s, target_action=1, gain=0.10, teacher_version=10 + seq)],
                        max_shards=3, vocab_hash=vocab.hash)
    remaining = sorted(p.name for p in exit_dir.glob("shard-*.pt"))
    assert len(remaining) == 3
    assert remaining == ["shard-w0-000003.pt", "shard-w0-000004.pt", "shard-w0-000005.pt"]

    # Corrupt shard + leftover .tmp file + wrong vocab hash + low gain are handled resiliently
    (exit_dir / "shard-corrupt-000001.pt").write_bytes(b"not a valid torch file")
    (exit_dir / ".shard-partial.tmp").write_bytes(b"partial")
    bad_vocab = exit_dir / "shard-badv-000001.pt"
    torch.save({"format": SHARD_FORMAT_VERSION, "vocab_hash": vocab.hash + 1, "records": []}, bad_vocab)
    save_exit_shard(
        exit_dir,
        "w1",
        1,
        [
            _make_record(s, target_action=1, gain=0.01, teacher_version=15),  # below min_gain=0.025
            _make_record(s, target_action=2, gain=0.20, teacher_version=15),  # kept
            _make_record(s, target_action=3, gain=0.20, teacher_version=15),  # kept
            _make_record(s, target_action=4, gain=0.20, teacher_version=15),  # kept
        ],
        max_shards=20,
        vocab_hash=vocab.hash,
    )
    ing, ev, corr = buf.poll_and_evict(current_update=20)
    # 3 from w0 (seq 3,4,5) + 3 valid from w1 = 6 ingested -> capacity=5 evicts 1 oldest!
    assert corr == 2  # corrupt + bad_vocab
    assert ing == 6 and ev == 1 and len(buf) == 5

    # Age eviction: advance current_update past max_age_updates=100 for teacher_version <= 14
    ing2, ev2, corr2 = buf.poll_and_evict(current_update=115)
    assert ing2 == 0 and corr2 == 0 and ev2 == 1 and len(buf) == 4  # seq 4 (v=14) evicted, v=15 kept

    # Advance past v=15 -> all remaining 4 evicted
    ing3, ev3, corr3 = buf.poll_and_evict(current_update=116)
    assert ing3 == 0 and ev3 == 4 and len(buf) == 0


# =========================================================================== 4. TEACHER TRIGGERS, FILTERS & WEIGHT POLLER


def test_teacher_triggers_filters_candidates_and_poller(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    cfg = ExitTeacherConfig()

    # Fight triggers: high-yield stages only
    assert should_trigger_fight(2, "Boss", 0.99, cfg)[0] is True
    assert should_trigger_fight(1, "Boss", 0.70, cfg)[0] is True
    assert should_trigger_fight(1, "Boss", 0.80, cfg)[0] is False
    assert should_trigger_fight(0, "Elite", 0.50, cfg)[0] is True
    assert should_trigger_fight(0, "Elite", 0.60, cfg)[0] is False
    assert should_trigger_fight(2, "Monster", 0.10, cfg)[0] is False

    # Per-turn pacing: <= 3 searched decs per turn across a 20-turn BB fight (fixes seed[1] front-loaded cap!)
    searched_by_turn = []
    fight_total = 0
    for turn in range(1, 21):
        t_cnt = 0
        chk_sel = chk_pot = chk_end = False
        for dec in range(10):
            ok, reason = should_trigger_decision(
                is_boss3=True,
                cur_pwin=0.55,
                greedy_gp=0.50,
                is_turn_start=(dec == 0),
                is_card_select=(dec == 1),
                has_potion=True,
                has_end_turn=(dec >= 2),
                greedy_is_end=False,
                turn_decs_searched=t_cnt,
                fight_decs_searched=fight_total,
                checked_card_select_turn=chk_sel,
                checked_potion_turn=chk_pot,
                checked_end_turn_turn=chk_end,
                cfg=cfg,
            )
            if ok:
                t_cnt += 1
                fight_total += 1
                if dec == 1:
                    chk_sel = True
        searched_by_turn.append(t_cnt)
    assert searched_by_turn == [3] * 20 and fight_total == 60
    # Skip when cur_pwin >= 0.94
    assert should_trigger_decision(
        is_boss3=True, cur_pwin=0.95, greedy_gp=0.30, is_turn_start=True, is_card_select=False,
        has_potion=True, has_end_turn=True, greedy_is_end=False, turn_decs_searched=0,
        fight_decs_searched=0, checked_card_select_turn=False, checked_potion_turn=False,
        checked_end_turn_turn=False, cfg=cfg,
    )[0] is False

    # High-precision emission filters
    assert passes_emission_filter(is_boss3=True, gain_all=0.03, t_all=1.9, gain_s2=0.02, dv_after=0.01, cfg=cfg)
    assert not passes_emission_filter(is_boss3=True, gain_all=0.02, t_all=2.5, gain_s2=0.02, dv_after=0.01, cfg=cfg)
    assert not passes_emission_filter(is_boss3=True, gain_all=0.10, t_all=2.5, gain_s2=0.05, dv_after=-0.01, cfg=cfg)
    assert not passes_emission_filter(is_boss3=False, gain_all=0.035, t_all=2.5, gain_s2=0.03, dv_after=0.02, cfg=cfg)
    assert passes_emission_filter(is_boss3=False, gain_all=0.05, t_all=2.1, gain_s2=0.025, dv_after=0.02, cfg=cfg)

    # Candidate selection K <= 4 retains [greedy, top-1, END_TURN, best_potion]
    reps = [
        {"row": 0, "verb": Verb.PLAY_CARD, "verb_name": "PLAY_CARD", "prob": 0.70, "group_prob": 0.70, "logit": 2.0, "is_greedy": True},
        {"row": 1, "verb": Verb.PLAY_CARD, "verb_name": "PLAY_CARD", "prob": 0.20, "group_prob": 0.20, "logit": 0.8, "is_greedy": False},
        {"row": 2, "verb": Verb.PLAY_CARD, "verb_name": "PLAY_CARD", "prob": 0.09, "group_prob": 0.09, "logit": 0.0, "is_greedy": False},
        {"row": 3, "verb": Verb.END_TURN, "verb_name": "END_TURN", "prob": 1e-6, "group_prob": 1e-6, "logit": -11.0, "is_greedy": False},
        {"row": 4, "verb": Verb.USE_POTION, "verb_name": "USE_POTION", "prob": 1e-7, "group_prob": 1e-7, "logit": -13.0, "is_greedy": False},
    ]
    cands = select_candidates(reps, max_k=4)
    assert [c["row"] for c in cands] == [0, 1, 3, 4]

    # Private counter scout seeds
    ss = SeedStream(12345)
    seeds = [make_scout_seed(ss) for _ in range(20)]
    assert len(set(seeds)) == 20 and all(len(x) == 12 and x.isalnum() for x in seeds)

    # LiveWeightPoller updates in-place (preserving parameter data_ptr for CUDA graphs)
    net = SpireNet(tok.layout, TINY)
    ptrs_before = {k: v.data_ptr() for k, v in net.named_parameters()}
    wdir = tmp_path / "weights"
    pub = FileWeightPublisher(wdir)
    target_sd = {k: v.clone() + 0.25 if v.is_floating_point() else v.clone() for k, v in net.state_dict().items()}
    pub.publish(target_sd, version=5)
    poller = LiveWeightPoller(ckpt_path=tmp_path / "ckpt" / "latest.pt", weights_dir=wdir, poll_s=1.0, have_version=0)
    assert poller.maybe_refresh(net, force=True) is True
    assert poller.version == 5
    assert {k: v.data_ptr() for k, v in net.named_parameters()} == ptrs_before
    for k, p in net.named_parameters():
        assert torch.allclose(p, target_sd[k])
    assert poller.maybe_refresh(net, force=True) is False  # same version -> no-op


# =========================================================================== 5. REAL SPIRENET LIFT, LEGAL MASKS & CHECKPOINT COMPAT


@pytest.mark.parametrize("use_cuda", [False, pytest.param(True, marks=cuda)])
def test_exit_loss_lifts_collapsed_action_respects_masks_and_checkpoint_compat(
    vocab: Vocab, tok: Tokenizer, fake_segments: list[Segment], tmp_path: Path, use_cuda: bool
) -> None:
    """Real SpireNet forward/backward test:
      - Lifts a collapsed p(c*) < 1e-4 action smoothly above 0.5 with zero NaNs and bounded grad norm.
      - Respects cycle-ban legal masks (illegal actions remain -inf / prob 0).
      - Works cleanly on both CPU eager and CUDA fast+graphs (preserving persistent .grad buffers).
      - Preserves 100% state_dict & checkpoint compatibility with p12 / py-p4n.
    """
    device = "cuda" if use_cuda else "cpu"
    tok_cb = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, token_buckets=tok.config.token_buckets))
    # Build segments matching tok_cb (cycle_ban=True)
    segs_cb: list[Segment] = []
    store_cb = RolloutStore(segs_cb.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=64))
    hosts_cb = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=45, hashes=False)) for i in range(6)]
    Actor(
        hosts_cb,
        UniformPolicy(0),
        tok_cb,
        EpisodeSampler(EpisodeDist(), vocab, 0),
        ActorConfig(max_batch=16),
        store=store_cb,
    ).run(max_decisions=1200)

    catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
    torch.manual_seed(42)
    net = SpireNet(tok_cb.layout, TINY, catalog).to(device)
    init_keys = list(net.state_dict().keys())

    # Build a combat sample with 1 cycle-banned illegal action (row 0 banned via history) and >= 4 legal actions
    s, n_act = _make_combat_sample(vocab, tok_cb, seed=99)
    banned_row = 0
    # Mark row 0 cycle-banned in the sample's cycle-ban tail slots (action_idx + 1)
    s_masked = Sample(
        s.types.copy(), s.raw.copy(), s.rel.copy(), s.edges.copy(), s.cond.copy(), s.counts, s.history.copy()
    )
    s_masked.history[-6] = banned_row + 1  # CYCLE_BAN_SLOTS = 6

    # Pick a legal target action c* and verify we can drive its initial probability < 1e-4
    target_row = n_act - 1
    assert target_row != banned_row
    batch_1 = {k: v.to(device) for k, v in tok_cb.collate([s_masked]).items()}
    net.eval()
    with torch.no_grad():
        logits0 = net(batch_1)["logits"][0]
        assert not torch.isfinite(logits0[banned_row]), "cycle-banned row must be -inf"
        assert torch.isfinite(logits0[target_row])

    # Create a collapsed initial policy on this state where p(target_row) < 1e-4 by pushing a rival legal action
    rival = 1 if target_row != 1 else 2
    opt_pre = torch.optim.Adam(net.parameters(), lr=5e-3)
    net.train()
    for _ in range(40):
        opt_pre.zero_grad(set_to_none=True)
        lg = net(batch_1)["logits"]
        lp = torch.log_softmax(lg.masked_fill(~torch.isfinite(lg), -1e30), dim=-1)
        loss_pre = -lp[0, rival]
        loss_pre.backward()
        opt_pre.step()
        del lg, lp, loss_pre
    opt_pre.zero_grad(set_to_none=True)
    del opt_pre
    # Rebuild a fresh SpireNet from the collapsed state_dict (mirrors real checkpoint load, zero pre-existing autograd nodes)
    collapsed_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net = SpireNet(tok_cb.layout, TINY, catalog).to(device)
    net.load_state_dict(collapsed_sd)
    net.eval()
    with torch.no_grad():
        logits_col = net(batch_1)["logits"][0]
        probs_col = torch.softmax(logits_col.masked_fill(~torch.isfinite(logits_col), -1e30), dim=-1)
        p_init = float(probs_col[target_row])
        assert p_init < 1e-4, f"expected collapsed initial prob < 1e-4, got {p_init}"
        assert not torch.isfinite(logits_col[banned_row])

    # Write teacher shard containing this collapsed state with target_action = target_row
    exit_dir = tmp_path / f"exit_{device}"
    lm = np.ones(n_act, dtype=np.bool_)
    lm[banned_row] = False
    recs = [_make_record(s_masked, target_action=target_row, gain=0.25, teacher_version=0, p_old=p_init, legal_mask=lm)
            for _ in range(16)]
    save_exit_shard(exit_dir, "w0", 1, recs, vocab_hash=vocab.hash)

    ppo_cfg = PPOConfig(
        epochs=1,
        minibatch=512,
        micro_batch=256,
        lr=1.5e-3,
        seed=7,
        bf16=False,
        grad_clip=1.0,
        exit_coef=0.25,
        exit_dir=str(exit_dir),
        exit_batch_size=16,
        exit_min_samples=8,
        exit_buffer_size=256,
        exit_max_age_updates=500,
        exit_min_gain=0.025,
        exit_gain_scale=0.08,
        exit_weight_min=0.5,
        exit_weight_max=3.0,
        exit_max_kl_guard=0.5,  # allow brisk lifting in this fast 15-update unit test
    )
    lr = Learner(
        net,
        tok_cb,
        ppo_cfg,
        device=device,
        size_fn=lambda x: tok_cb.config.bucket(x.n_tokens),
        fast=use_cuda,
        graphs=use_cuda,
    )
    if use_cuda:
        grad_ptrs_before = [p.grad.data_ptr() for p in lr._params if p.grad is not None]
        assert len(grad_ptrs_before) == len(lr._params)

    probs_trace: list[float] = [p_init]
    for step in range(25):
        m = lr.update(segs_cb)
        assert m["exit_buffer_n"] == 16
        assert m["exit_loss"] is not None and math.isfinite(m["exit_loss"])
        assert m["exit_grad_norm"] is not None and math.isfinite(m["exit_grad_norm"])
        assert m["exit_step_kl"] is not None and math.isfinite(m["exit_step_kl"]) and m["exit_step_kl"] >= 0.0
        assert m["exit_step_ms"] >= 0.0
        with torch.no_grad():
            lr.model.eval()
            lg = lr.model(batch_1)["logits"][0]
            assert not torch.isfinite(lg[banned_row]), "cycle-banned action must remain illegal (-inf)"
            p_now = float(torch.softmax(lg.masked_fill(~torch.isfinite(lg), -1e30), dim=-1)[target_row])
            probs_trace.append(p_now)
        if p_now > 0.60:
            break

    assert probs_trace[-1] > 0.50, f"L_exit failed to lift p(c*) from {p_init} above 0.5: trace={probs_trace}"
    if use_cuda:
        # Persistent gradient buffer addresses must remain unchanged for CUDA graphs!
        assert [p.grad.data_ptr() for p in lr._params if p.grad is not None] == grad_ptrs_before

    # Checkpoint compatibility with p12 / py-p4n
    ckpt_path = tmp_path / f"p13_{device}.pt"
    tcfg = build(
        TrainConfig,
        {"run_name": "compat", "tokenizer": {"cycle_ban": True}, "model": {"override": TINY}, "ppo": ppo_cfg.to_dict()},
    )
    extra = checkpoint_extra(tcfg, vocab, catalog)
    lr.save(ckpt_path, extra=extra)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert list(ck["model"].keys()) == init_keys
    assert list(ck["ema"].keys()) == init_keys

    # Load p13 checkpoint into a plain OFF (p12-compatible) learner
    net_off = SpireNet(tok_cb.layout, TINY, catalog)
    lr_off = Learner(net_off, tok_cb, PPOConfig(epochs=1, minibatch=512, micro_batch=256, bf16=False), device="cpu", fast=False)
    lr_off.load(ckpt_path, allow_config_change=True)
    assert lr_off.version == lr.version
    # Also verify load_policy_net loads EMA cleanly (exact eval / py-p4n path)
    eval_net, eval_tok, eval_v, _ = load_policy_net(ckpt_path, device="cpu", weights="ema")
    assert eval_v == lr.version and list(eval_net.state_dict().keys()) == init_keys

    # Reverse direction: save plain OFF checkpoint and load into ExIt-enabled p13 learner
    off_ckpt = tmp_path / f"p12_off_{device}.pt"
    lr_off.save(off_ckpt, extra=extra)
    net_re = SpireNet(tok_cb.layout, TINY, catalog)
    lr_re = Learner(net_re, tok_cb, ppo_cfg, device="cpu", fast=False)
    lr_re.load(off_ckpt, allow_config_change=True)
    assert lr_re.version == lr.version


# =========================================================================== 6. GRACEFUL MISSING/STALE DIR & KL GUARD


def test_exit_learner_graceful_missing_stale_and_kl_guard(
    vocab: Vocab, tok: Tokenizer, fake_segments: list[Segment], tmp_path: Path
) -> None:
    missing_dir = tmp_path / "does_not_exist_yet"
    cfg = PPOConfig(
        epochs=1,
        minibatch=256,
        micro_batch=128,
        lr=1e-3,
        seed=5,
        bf16=False,
        exit_coef=0.1,
        exit_dir=str(missing_dir),
        exit_batch_size=8,
        exit_min_samples=4,
        exit_max_age_updates=50,
        exit_max_kl_guard=1e-5,  # tiny guard to verify adaptive damping triggers cleanly
    )
    net = SpireNet(tok.layout, TINY)
    lr = Learner(net, tok, cfg, device="cpu", fast=False)

    # 1. Missing directory -> skips cleanly with zero error
    m0 = lr.update(fake_segments)
    assert m0["exit_loss"] is None and m0["exit_buffer_n"] == 0 and m0["exit_ingested"] == 0

    # 2. Only stale shards -> evicted immediately, skips cleanly
    s, _ = _make_combat_sample(vocab, tok, seed=12)
    lr.updates = 200
    save_exit_shard(missing_dir, "w0", 1, [_make_record(s, target_action=1, gain=0.2, teacher_version=10) for _ in range(8)],
                    vocab_hash=vocab.hash)
    m1 = lr.update(fake_segments)
    assert m1["exit_evicted"] == 8 and m1["exit_buffer_n"] == 0 and m1["exit_loss"] is None

    # 3. Fresh shards with tiny exit_max_kl_guard -> steps cleanly and damps exit_lr_scale < 1.0
    save_exit_shard(missing_dir, "w0", 2, [_make_record(s, target_action=2, gain=0.2, teacher_version=200) for _ in range(8)],
                    vocab_hash=vocab.hash)
    m2 = lr.update(fake_segments)
    assert m2["exit_buffer_n"] == 8 and m2["exit_ingested"] == 8
    assert m2["exit_loss"] is not None and math.isfinite(m2["exit_loss"])
    assert m2["exit_step_kl"] is not None and m2["exit_step_kl"] > 1e-5
    assert m2["exit_lr_scale"] < 1.0
