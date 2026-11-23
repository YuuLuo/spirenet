"""Multi-replica learner tests (K = 1 + N cards, e.g. 4+4 on 8xRTX 5090).

Covers:
1. Config parsing, normalization, JSON/TOML round-trips, and backward compatibility
   (``""``, ``"cuda:1"``, ``["cuda:1", "cuda:2", "cuda:3"]``, comma strings).
2. ``check_dual`` refusals for duplicate cards, invalid device strings, and collisions with
   ``learner.device``, ``actor.device(s)``, or ``eval.device``.
3. LPT load-balancing in ``Replica.assign`` for K=2 (bit-identical to legacy dual) and K=4.
4. Simulated K=4 multi-replica end-to-end CUDA tests on a single GPU (real CUDA graphs,
   ``enqueue_all``, deterministic fixed-order ``combine``, event-ordered ``sync``, rollback,
   ExIt v2 + ``exit_precond`` sync, non-finite skip on worker 2, and cross-GPU-count CUDA RNG
   checkpoint resume).
5. Real 4-GPU hardware parity tests (``@four_cards``; skipped when ``device_count() < 4``).
"""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.exit_buffer import ExitRecord, save_exit_shard
from spire.learner import Learner, PPOConfig, _Accum
from spire.model import ModelConfig, SpireNet
from spire.replica import Replica, _ReplicaWorker, normalize_replica_devices
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import reset_flags_for
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import LearnerSection, TrainConfig, build, check_dual, config_dict, load_config
from spire.vocab import Vocab

from test_dual import (
    BUCKETS,
    TINY,
    V2A,
    _bit_equal,
    _cfg,
    _flat,
    _Grads,
    _metrics,
    _raw,
    _record_lr,
    _rel,
    _state,
    _sum_rel,
    _tok,
    COUNTS,
    det,  # noqa: F401
    table_path,  # noqa: F401
)

need_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
four_cards = pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs four CUDA devices")


@pytest.fixture(autouse=True)
def _collect_multi() -> Any:
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =========================================================================== 1. Config parsing & validation


def test_normalize_replica_devices() -> None:
    assert normalize_replica_devices(None) == ()
    assert normalize_replica_devices("") == ()
    assert normalize_replica_devices("   ") == ()
    assert normalize_replica_devices([]) == ()
    assert normalize_replica_devices(()) == ()
    assert normalize_replica_devices("cuda:1") == ("cuda:1",)
    assert normalize_replica_devices(torch.device("cuda:2")) == ("cuda:2",)
    assert normalize_replica_devices(["cuda:1"]) == ("cuda:1",)
    assert normalize_replica_devices(["cuda:1", "cuda:2", "cuda:3"]) == ("cuda:1", "cuda:2", "cuda:3")
    assert normalize_replica_devices(("cuda:1", torch.device("cuda:2"))) == ("cuda:1", "cuda:2")
    assert normalize_replica_devices("cuda:1,cuda:2,cuda:3") == ("cuda:1", "cuda:2", "cuda:3")
    assert normalize_replica_devices(" cuda:1 , cuda:2 , cuda:3 ") == ("cuda:1", "cuda:2", "cuda:3")

    for bad in ("cuda:1,", ",cuda:1", "cuda:1,,cuda:2", [""], ["cuda:1", ""], ["cuda:1,cuda:2"], 123, [1]):
        with pytest.raises(AssertionError):
            normalize_replica_devices(bad)  # type: ignore[arg-type]


def test_config_roundtrip_single_multi_and_off(tmp_path: Path) -> None:
    # Off ("" or []) -> omitted from config_dict
    for empty_val in ("", []):
        off = build(TrainConfig, {"run_name": "x", "learner": {"replica_device": empty_val}})
        assert off.learner.replica_device == ""
        assert "replica_device" not in config_dict(off)["learner"]
        check_dual(off)

    # Single string -> stays string "cuda:1" (backward compatible with older configs)
    single = build(TrainConfig, _raw(learner={"replica_device": "cuda:1"}))
    assert single.learner.replica_device == "cuda:1"
    check_dual(single)
    assert config_dict(single)["learner"]["replica_device"] == "cuda:1"

    # 1-element list -> canonicalizes to single string "cuda:1"
    single_list = build(TrainConfig, _raw(learner={"replica_device": ["cuda:1"]}))
    assert single_list.learner.replica_device == "cuda:1"
    assert config_dict(single_list)["learner"]["replica_device"] == "cuda:1"

    # 4+4 production JSON config: 4 learner GPUs (0..3) + 4 actor GPUs (4..7) + eval on CPU
    raw44 = {
        "run_name": "s35-4plus4",
        "learner": {
            "device": "cuda:0",
            "replica_device": ["cuda:1", "cuda:2", "cuda:3"],
            "cuda_graphs": True,
            "compile": True,
            "pipeline_prepare": True,
            "acting_weights": "ema",
            "confident_keep": 0.2,
            "confident_exempt_kinds": [1, 5, 6, 7, 8],
        },
        "actor": {"actors": 32, "devices": ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]},
        "eval": {"device": "cpu"},
        "ppo": {"exit_coef": 0.06, "exit_dir": "exit_shards", "exit_precond": True},
    }
    cfg = build(TrainConfig, raw44)
    check_dual(cfg)
    assert cfg.learner.replica_device == ("cuda:1", "cuda:2", "cuda:3")
    # Hashable frozen dataclass!
    assert isinstance(hash(cfg.learner), int)
    d = config_dict(cfg)
    assert d["learner"]["replica_device"] == ["cuda:1", "cuda:2", "cuda:3"]
    p_json = tmp_path / "config.json"
    p_json.write_text(json.dumps(d), encoding="utf-8")
    reloaded = load_config(p_json)
    assert reloaded.learner.replica_device == ("cuda:1", "cuda:2", "cuda:3")

    # Comma-separated string + TOML roundtrip
    p_toml = tmp_path / "run.toml"
    p_toml.write_text(
        'run_name = "toml-44"\n'
        '[learner]\n'
        'device = "cuda:0"\n'
        'replica_device = ["cuda:1", "cuda:2", "cuda:3"]\n'
        '[actor]\n'
        'devices = ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]\n'
        '[eval]\n'
        'device = "cpu"\n',
        encoding="utf-8",
    )
    assert load_config(p_toml).learner.replica_device == ("cuda:1", "cuda:2", "cuda:3")
    csv_cfg = build(TrainConfig, {**raw44, "learner": {**raw44["learner"], "replica_device": "cuda:1,cuda:2,cuda:3"}})
    assert csv_cfg.learner.replica_device == ("cuda:1", "cuda:2", "cuda:3")
    check_dual(csv_cfg)


MULTI_REFUSED = [
    ({"learner": {"replica_device": ["cuda:1", "cuda:1"]}}, "duplicate CUDA device"),
    ({"learner": {"replica_device": ["cuda:1", "cuda:2", "cuda:1"]}}, "duplicate CUDA device"),
    ({"learner": {"replica_device": ["cuda:1", "cuda"]}}, "a CUDA device cuda:N"),
    ({"learner": {"replica_device": ["cuda:1", "cpu"]}}, "a CUDA device cuda:N"),
    ({"learner": {"replica_device": ["cuda:1", "bogus:2"]}}, "a CUDA device cuda:N"),
    ({"learner": {"device": "cuda:2", "replica_device": ["cuda:1", "cuda:2", "cuda:3"]}}, "is the learner's device"),
    ({"learner": {"device": "cuda", "replica_device": ["cuda:1", "cuda:0"]}}, "is the learner's device"),
    ({"actor": {"devices": ["cuda:4", "cuda:3"]}, "learner": {"replica_device": ["cuda:1", "cuda:2", "cuda:3"]}},
     "is an actor's device"),
    ({"actor": {"device": "cuda:2", "devices": []}, "learner": {"replica_device": ["cuda:1", "cuda:2", "cuda:3"]}},
     "is an actor's device"),
    ({"eval": {"device": "cuda:3"}, "learner": {"replica_device": ["cuda:1", "cuda:2", "cuda:3"]}},
     "is the evaluator's device"),
    ({"learner": {"replica_device": ["cuda:1", "cuda:2", "cuda:3"], "fast": False}}, "learner.fast false"),
    ({"learner": {"replica_device": ["cuda:1", "cuda:2", "cuda:3"], "cuda_graphs": False}},
     "learner.cuda_graphs false"),
    ({"topology": "threads", "learner": {"replica_device": ["cuda:1", "cuda:2", "cuda:3"]}}, "topology 'threads'"),
    ({"ppo": {"kickstart": {"teacher": "t.pt"}}, "learner": {"replica_device": ["cuda:1", "cuda:2", "cuda:3"]}},
     "an active kickstart"),
]


@pytest.mark.parametrize("over,message", MULTI_REFUSED, ids=[f"{i}-{m}" for i, (_, m) in enumerate(MULTI_REFUSED)])
def test_multi_replica_refusals(over: dict[str, Any], message: str) -> None:
    base: dict[str, Any] = {
        "run_name": "x",
        "learner": {"device": "cuda:0", "replica_device": ["cuda:1", "cuda:2", "cuda:3"]},
        "actor": {"devices": ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]},
        "eval": {"device": "cpu"},
    }
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    cfg = build(TrainConfig, base)
    with pytest.raises(AssertionError, match=message.replace(".", r"\.").replace("(", r"\(")):
        check_dual(cfg)


# =========================================================================== 2. LPT assign() unit tests


def _stub_replica(n_cards: int, cfg: PPOConfig | None = None) -> Replica:
    rep = Replica.__new__(Replica)
    rep._cfg = cfg or PPOConfig(micro_batch=256, micro_tokens=4096)
    rep.n_cards = n_cards
    rep.micro = [0] * n_cards
    rep.load = [0] * n_cards
    rep.split = "balance"
    return rep


def test_assign_k2_matches_legacy_and_k4_balances() -> None:
    cfg = PPOConfig(micro_batch=256, micro_tokens=4096)
    # 16 micro-batches with descending bucket sizes (as bucket_micros emits)
    micros = [
        (256, np.arange(16)), (256, np.arange(16)), (192, np.arange(20)), (192, np.arange(20)),
        (128, np.arange(32)), (128, np.arange(32)), (128, np.arange(30)), (96, np.arange(40)),
        (96, np.arange(40)), (80, np.arange(48)), (80, np.arange(48)), (64, np.arange(64)),
        (64, np.arange(64)), (40, np.arange(96)), (40, np.arange(96)), (28, np.arange(120)),
    ]
    # K=2: ties to 0, exact match to old (load[1] < load[0])
    r2 = _stub_replica(2, cfg)
    s2 = r2.assign(micros)
    assert s2[0] == 0 and s2[1] == 1
    assert sum(r2.micro) == len(micros) and all(c > 0 for c in r2.micro)
    assert abs(r2.load[0] / sum(r2.load) - 0.5) < 0.05

    # K=4: LPT balances across all 4 cards within ~5% of 0.25 each
    r4 = _stub_replica(4, cfg)
    s4 = r4.assign(micros)
    assert s4[:4] == [0, 1, 2, 3]
    assert sum(r4.micro) == len(micros) and all(c == 4 for c in r4.micro)
    tot = sum(r4.load)
    shares = [ld / tot for ld in r4.load]
    assert all(abs(sh - 0.25) < 0.05 for sh in shares), shares

    # record=False leaves micro & load untouched
    prev_m, prev_l = r4.micro.copy(), r4.load.copy()
    s4_nr = r4.assign(micros, record=False)
    assert s4_nr == s4 and r4.micro == prev_m and r4.load == prev_l

    # split switches: "a" -> all 0; "b" -> balanced over 1..3 (0 idle); int -> single card
    r4.split = "a"
    assert r4.assign(micros, record=False) == [0] * len(micros)
    r4.split = "b"
    sb = r4.assign(micros, record=False)
    assert all(x in (1, 2, 3) for x in sb) and {1, 2, 3}.issubset(sb)
    r4.split = 2
    assert r4.assign(micros, record=False) == [2] * len(micros)
    r4.split = 4
    with pytest.raises(AssertionError):
        r4.assign(micros, record=False)


# =========================================================================== 3. Simulated K=4 CUDA parity (runs on 1 GPU!)


def _attach_sim_replicas(lr: Learner, n_replicas: int, *, graphs: bool = True, compile_model: bool = False) -> Replica:
    """Attach N real `_ReplicaWorker` instances on `lr.device` (same physical GPU) to exercise every Replica &
    Learner multi-replica code path (CUDA graphs, enqueue_all, combine, sync, rollback, ExIt sync) on a 1-GPU box."""
    assert lr.stepper is not None
    dev = lr.device
    model = lr.model
    params_a = [p for p in model.parameters() if p.requires_grad]
    names_a = [n for n, p in model.named_parameters() if p.requires_grad]
    sizes = [p.numel() for p in params_a]
    offs = [0]
    for n in sizes:
        offs.append(offs[-1] + -(-n // 128) * 128)
    total = offs[-1]
    rep = Replica.__new__(Replica)
    rep.a_device = dev
    rep.params_a = params_a
    rep.send = torch.zeros(total, device=dev)
    rep.send_views = [rep.send[o:o + n].view_as(p) for p, o, n in zip(params_a, offs, sizes)]
    rep._cfg = lr.cfg
    rules_on = lr.rules_mode != "mask"
    rep.workers = [
        _ReplicaWorker(
            idx, model, params_a, names_a, offs, sizes, total, dev,
            lr.tokenizer, lr.cfg, dev, _Accum(dev),  # type: ignore[arg-type]
            graphs=graphs, autocast=lr.autocast, compile_model=compile_model, rules_on=rules_on,
        )
        for idx in range(n_replicas)
    ]
    rep.devices = tuple(w.device for w in rep.workers)
    rep.n_cards = 1 + n_replicas
    rep.micro = [0] * rep.n_cards
    rep.load = [0] * rep.n_cards
    rep.split = "balance"
    lr.replica = rep
    lr._grads_a = [p.grad for p in params_a]  # type: ignore[misc]
    return rep


def _all_replicas_equal_a(lr: Learner) -> None:
    rep = lr.replica
    assert rep is not None
    for idx, w in enumerate(rep.workers):
        for (na, pa), (nb, pb) in zip(lr.model.named_parameters(), w.model.named_parameters()):
            assert na == nb and torch.equal(pa.detach().cpu(), pb.detach().cpu()), (idx, na)
        for (na, ba), (nb, bb) in zip(lr.model.named_buffers(), w.model.named_buffers()):
            assert na == nb and torch.equal(ba.cpu(), bb.cpu()), (idx, na)
        assert float(w.gflat.abs().sum()) == 0.0, idx
    assert all(p.grad is not None and float(p.grad.abs().sum()) == 0.0 for p in rep.params_a)


def _sim_segments(vocab: Vocab, tok: Tokenizer) -> list[Segment]:
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=64))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
             for i in range(12)]
    acfg = ActorConfig(max_batch=32, reset_flags=reset_flags_for(True, calc=True, map_in_combat=True))
    Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0),)), vocab, 0), acfg,
          store=store).run(max_decisions=3000)
    return segs


@need_cuda
@pytest.mark.parametrize("bf16", [False, True])
def test_simulated_k4_parity_determinism_and_row(vocab: Vocab, table_path: str, det: Any, bf16: bool) -> None:  # noqa: F811
    tok = _tok(vocab, "v2a-free", table_path)
    segs = _sim_segments(vocab, tok)
    cfg = _cfg(bf16=bf16, minibatch=512, micro_batch=128, micro_tokens=2048)
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    single = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True)
    quad1 = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True)
    quad2 = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True)
    _attach_sim_replicas(quad1, 3)
    _attach_sim_replicas(quad2, 3)
    try:
        for k in range(3):
            ms = single.update(segs)
            m1 = quad1.update(segs)
            m2 = quad2.update(segs)
            _all_replicas_equal_a(quad1)
            assert _metrics(m1) == _metrics(m2), k  # deterministic across repeated K=4 runs
            d = m1["dual"]
            assert len(d["devices"]) == 3 and len(d["shares"]) == 4 and len(d["micros"]) == 4
            assert len(d["steppers"]) == 3
            assert all(c > 0 for c in d["micros"]), d["micros"]
            assert sum(d["micros"]) == m1["micro_batches"]
            assert d["micro_a"] == d["micros"][0] and d["micro_b"] == sum(d["micros"][1:])
            assert abs(sum(d["shares"]) - 1.0) < 1e-9
            for key in COUNTS:
                assert ms[key] == m1[key], (k, key)
            worst, name = _sum_rel(ms, m1)
            if not bf16:
                assert worst <= 1e-4, (k, name, worst)
        _bit_equal(_state(quad1), _state(quad2))
        p_rel = float((_flat(quad1) - _flat(single)).norm() / max(float((_flat(single)).norm()), 1e-12))
        assert p_rel <= (1e-5 if not bf16 else 2e-3), p_rel
    finally:
        assert quad1.replica is not None and quad2.replica is not None
        quad1.replica.close()
        quad2.replica.close()


@need_cuda
def test_simulated_k4_all_on_a_and_nonfinite_on_worker2(vocab: Vocab, table_path: str, det: Any) -> None:  # noqa: F811
    tok = _tok(vocab, "v2a-free", table_path)
    segs = _sim_segments(vocab, tok)
    cfg = _cfg(bf16=False, minibatch=512, micro_batch=128, micro_tokens=2048)
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    single = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True)
    on_a = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True)
    poison = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True)
    rep_a = _attach_sim_replicas(on_a, 3)
    rep_p = _attach_sim_replicas(poison, 3)
    try:
        # All on A: bit-identical to single learner, all 3 replicas stay synced
        rep_a.split = "a"
        for _ in range(2):
            ms, ma = single.update(segs), on_a.update(segs)
            assert _metrics(ms) == _metrics(ma)
            assert ma["dual"]["micro_b"] == 0 and ma["dual"]["share_a"] == 1.0
            _all_replicas_equal_a(on_a)
        _bit_equal(_state(single), _state(on_a))

        # Non-finite gradient injected on worker 2 (4th card) skips the step and syncs all 3 replicas cleanly
        orig_combine = rep_p.combine
        calls = [-1]

        def poisoned_combine(grads_a: list[torch.Tensor], kl: torch.Tensor) -> torch.Tensor:
            calls[0] += 1
            if calls[0] == 1:
                rep_p.workers[2].gflat[0] = float("nan")
            return orig_combine(grads_a, kl)

        rep_p.combine = poisoned_combine  # type: ignore[method-assign]
        mp = poison.update(segs)
        assert mp["nonfinite_skipped"] == 1
        _all_replicas_equal_a(poison)
        mp2 = poison.update(segs)
        assert mp2["nonfinite_skipped"] == 0
        _all_replicas_equal_a(poison)
    finally:
        rep_a.close()
        rep_p.close()


@need_cuda
@pytest.mark.parametrize("exit_precond", [False, True])
def test_simulated_k4_exit_v2_and_cross_gpu_rng_resume(
    vocab: Vocab, table_path: str, tmp_path: Path, exit_precond: bool,  # noqa: F811
) -> None:
    tok = _tok(vocab, "v2a-free", table_path)
    segs = _sim_segments(vocab, tok)
    exit_dir = tmp_path / "exit"
    exit_dir.mkdir()
    # Build synthetic ExIt shard from the real SampleBlock samples
    rows: list[ExitRecord] = []
    for seg in segs:
        blk = seg.samples
        for i in range(len(blk)):
            s = blk[i]
            n_act = s.n_actions
            if n_act < 2:
                continue
            rows.append(ExitRecord(
                sample=s,
                target_action=1,
                legal_mask=np.ones(n_act, dtype=np.bool_),
                cand_rows=(0, 1),
                greedy_action=0,
                gain=0.15,
                gain_s2=0.15,
                t_stat=2.5,
                dv_after=0.10,
                p_old=0.1,
                p_group_old=0.1,
                teacher_version=0,
                stage="A3-BB",
                verb="PLAY",
                cand_scores=(0.55, 0.70),
                cand_p_old=(0.9, 0.1),
            ))
            if len(rows) >= 32:
                break
        if len(rows) >= 32:
            break
    assert len(rows) >= 16
    save_exit_shard(exit_dir, "s35w0", 1, rows, vocab_hash=int(tok.vocab.hash))
    cfg = _cfg(
        bf16=True, minibatch=512, micro_batch=128, micro_tokens=2048,
        exit_coef=0.06, exit_dir=str(exit_dir), exit_batch_size=16, exit_min_samples=16,
        exit_precond=exit_precond, exit_soft_beta=8.0, exit_apply_legal_mask=True,
    )
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    lr = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True)
    rep = _attach_sim_replicas(lr, 3)
    try:
        for _ in range(2):
            m = lr.update(segs)
            assert m["exit_loss"] is not None and np.isfinite(m["exit_loss"])
            _all_replicas_equal_a(lr)  # ExIt step synced all 3 replicas without race!
        # Save checkpoint and forge CUDA RNG state lengths (4-GPU ckpt and 8-GPU ckpt -> load on current machine)
        ckpt_path = tmp_path / "ckpt.pt"
        lr.save(ckpt_path)
        for forged_n in (4, 8):
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cur_state = ck["rng"]["cuda"][0]
            ck["rng"]["cuda"] = [cur_state.clone() for _ in range(forged_n)]
            forged = tmp_path / f"ckpt-{forged_n}gpu.pt"
            torch.save(ck, forged)
            fresh = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True)
            rep_f = _attach_sim_replicas(fresh, 3)
            try:
                fresh.load(forged)
                _all_replicas_equal_a(fresh)
                _bit_equal(_state(fresh), _state(lr))
                m_after = fresh.update(segs)
                assert m_after["exit_loss"] is not None
                _all_replicas_equal_a(fresh)
            finally:
                rep_f.close()
    finally:
        rep.close()


@need_cuda
def test_simulated_k4_target_kl_early_stop(vocab: Vocab, table_path: str, det: Any) -> None:  # noqa: F811
    tok = _tok(vocab, "v2a-free", table_path)
    segs = _sim_segments(vocab, tok)
    probe_cfg = _cfg(bf16=False, lr=3e-3, kl_lr_target=None, minibatch=512, micro_batch=128, micro_tokens=2048)
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    probe = Learner(copy.deepcopy(base), tok, probe_cfg, device="cuda:0", fast=True, graphs=True)
    rp = _record_lr(probe)
    probe.update(segs)
    kls = [kl for kl, _ in rp]
    target = 0.5 * (kls[0] + max(kls[1:]))
    assert any(kl > target for kl in kls[1:]) and kls[0] <= target

    stop_cfg = _cfg(bf16=False, lr=3e-3, kl_lr_target=None, target_kl=target, minibatch=512, micro_batch=128,
                    micro_tokens=2048)
    single = Learner(copy.deepcopy(base), tok, stop_cfg, device="cuda:0", fast=True, graphs=True)
    quad = Learner(copy.deepcopy(base), tok, stop_cfg, device="cuda:0", fast=True, graphs=True)
    rep = _attach_sim_replicas(quad, 3)
    try:
        ms, mq = single.update(segs), quad.update(segs)
        assert ms["early_stop"] and mq["early_stop"] and ms["epochs_done"] == mq["epochs_done"] < 2.0
        _all_replicas_equal_a(quad)
    finally:
        rep.close()


# =========================================================================== 4. Real 4-GPU hardware parity tests


@four_cards
@pytest.mark.parametrize("bf16", [False, True])
def test_4gpu_d3_parity_determinism_and_save_load(
    vocab: Vocab, table_path: str, det: Any, tmp_path: Path, bf16: bool, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    tok = _tok(vocab, "v2a-free", table_path)
    segs = _sim_segments(vocab, tok)
    cfg = _cfg(bf16=bf16, minibatch=512, micro_batch=128, micro_tokens=2048)
    grads = _Grads(monkeypatch)
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    rdevs = ["cuda:1", "cuda:2", "cuda:3"]
    single = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True)
    quad1 = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True, replica_device=rdevs)
    quad2 = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True, replica_device=rdevs)
    gs, g1, g2 = grads.watch(single), grads.watch(quad1), grads.watch(quad2)
    try:
        for k in range(3):
            ms, m1, m2 = single.update(segs), quad1.update(segs), quad2.update(segs)
            _all_replicas_equal_a(quad1)
            assert _metrics(m1) == _metrics(m2), k
            assert all(c > 0 for c in m1["dual"]["micros"]), m1["dual"]["micros"]
            for key in COUNTS:
                assert ms[key] == m1[key], (k, key)
        _bit_equal(_state(quad1), _state(quad2))
        assert all(torch.equal(a, b) for a, b in zip(g1, g2))
        g_rel = _rel(g1, gs)
        if not bf16:
            assert g_rel <= 1e-5, g_rel
        # Save 2-GPU checkpoint -> load into 4-GPU learner
        dual = Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True, replica_device="cuda:1")
        try:
            dual.update(segs)
            p = tmp_path / "from-2gpu.pt"
            dual.save(p)
            quad1.load(p)
            _all_replicas_equal_a(quad1)
            _bit_equal(_state(quad1), _state(dual))
            quad1.update(segs)
            _all_replicas_equal_a(quad1)
        finally:
            assert dual.replica is not None
            dual.replica.close()
    finally:
        assert quad1.replica is not None and quad2.replica is not None
        quad1.replica.close()
        quad2.replica.close()
