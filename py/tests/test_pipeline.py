"""learner.pipeline_prepare (notes/next/pipeline-spec.md): the split of Learner.update, the pipelined learner loop.

* the refusals (one message each) and config.json without the key when off;
* the split inside the tree: update(segments) == update_prepared(prepare(segments, version)), CPU fp32 eager, K = 5
  (the cross-tree check against the base tree is tools/t2_cross.py);
* T3: a ring pre-filled with N = 8 iterations, no staleness filter, no new data: the sequential and the pipelined
  learner loop give the same batches (segment ids per update), samples_seen, parameters, Adam state, EMA, counters
  and non-timing metrics;
* T4: max_staleness on, a scripted ring (arrivals before every take): the pipelined loop evicts exactly the segments
  the sequential loop evicts (the cv rule), and no update trips the staleness asserts;
* T5: stop with a prepared, unconsumed batch (clean end, final checkpoint, the dropped samples counted); an exception
  in prepare is the learner's failure (learner_error, crash checkpoint, FAILED.json through the supervisor), not a
  hang; SPIRE_TEST_LEARNER_FAIL_AT keeps working with the option on."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.learner import Learner, PPOConfig
from spire.metrics import read_jsonl
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment, SegmentRing
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import (Supervisor, TrainConfig, _learner_loop, build, check_pipeline, config_dict, learner_main)
from spire.vocab import Vocab

from test_train import small_config

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
N_ITER = 8
ITER = 512
TINY = {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "d_id": 8, "d_cond": 8, "d_static": 16}
TIMING = ("time_prep", "time_train", "samples_per_s", "lag_s", "wait_s", "prep_wait_s", "handoff_wait_s",
          "cuda_mem_mb", "pipeline", "ring", "ring_samples", "transfer", "t", "elapsed")


# =========================================================================== refusals, config


REFUSED = [
    ({"ppo": {"recompute_values": True}}, "ppo.recompute_values"),
    ({"ppo": {"decoupled_clip_anchor": True}}, "ppo.decoupled_clip_anchor"),
    ({"ppo": {"normalize_returns": True}}, "ppo.normalize_returns"),
    ({"ppo": {"kickstart": {"teacher": "t.pt"}}}, "an active kickstart"),
    ({"ppo": {"taste": {"file": "t.json", "p_episode": 0.1, "eps": 0.2}}}, "ppo.taste.on"),
    ({"ppo": {"advice": [["REST", "SMITH", 0.5, 0.05]]}}, "ppo.advice"),
    ({"learner": {"fast": False}}, "learner.fast false"),
    ({"learner": {"device": "cpu"}}, "learner.device cpu"),
    ({"topology": "threads"}, "topology 'threads'"),
]


def _raw(**over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {"run_name": "x", "learner": {"pipeline_prepare": True}}
    for k, v in over.items():
        raw[k] = {**raw.get(k, {}), **v} if isinstance(v, dict) else v
    return raw


@pytest.mark.parametrize("over,message", REFUSED, ids=[m for _, m in REFUSED])
def test_refusals(over: dict[str, Any], message: str) -> None:
    cfg = build(TrainConfig, _raw(**over))
    with pytest.raises(AssertionError, match=message.replace(".", r"\.").replace("(", r"\(")):
        check_pipeline(cfg)
    off = build(TrainConfig, {**_raw(**over), "learner": {**_raw(**over)["learner"], "pipeline_prepare": False}})
    check_pipeline(off)  # off: nothing is refused


def test_production_shape_is_accepted_and_config_json() -> None:
    cfg = build(TrainConfig, _raw(learner={"cuda_graphs": True, "compile": True}, ppo={"combat_temperature": 1.0}))
    check_pipeline(cfg)
    assert config_dict(cfg)["learner"]["pipeline_prepare"] is True
    off = build(TrainConfig, {"run_name": "x"})
    assert "pipeline_prepare" not in config_dict(off)["learner"]  # off: the config.json of tree p5
    assert build(TrainConfig, config_dict(off)).learner.pipeline_prepare is False


# =========================================================================== the split inside the tree


def _segments(vocab: Vocab, tok: Tokenizer, decisions: int, seg_len: int = 64) -> list[Segment]:
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=seg_len))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
             for i in range(12)]
    actor = Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0), ActorConfig(max_batch=32),
                  store=store)
    actor.run(max_decisions=decisions)
    return segs


def _state(lr: Learner) -> dict[str, Any]:
    return {"model": lr.model.state_dict(), "ema": lr.ema.state_dict(), "opt": lr.opt.state_dict(),
            "counters": (lr.version, lr.updates, lr.samples_seen, lr.ema_steps),
            "rng": json.dumps(lr.rng.bit_generator.state, sort_keys=True)}


def _same_state(a: dict[str, Any], b: dict[str, Any]) -> None:
    for k in ("model", "ema"):
        assert a[k].keys() == b[k].keys()
        for name in a[k]:
            assert torch.equal(a[k][name], b[k][name]), (k, name)
    sa, sb = a["opt"]["state"], b["opt"]["state"]
    assert sa.keys() == sb.keys()
    for i in sa:
        for f in sa[i]:
            assert torch.equal(sa[i][f], sb[i][f]), ("adam", i, f)
    assert a["counters"] == b["counters"] and a["rng"] == b["rng"]


def _metrics(m: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps({k: v for k, v in m.items() if k not in TIMING}))


@pytest.fixture()
def one_thread() -> Any:
    """The CPU update is bit-reproducible only single-threaded (scatter / index_add sums); other test modules of the
    suite leave the process at 2-4 threads."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_update_is_prepare_then_update_prepared(vocab: Vocab, one_thread: Any) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True, cycle_ban=True, commit_select=True))
    segs = _segments(vocab, tok, 1500, 32)
    cfg = PPOConfig(epochs=2, minibatch=256, micro_batch=128, lr=1e-3, seed=0, kl_lr_target=0.002, weight_decay=0.01)
    torch.manual_seed(5)
    net = SpireNet(tok.layout, ModelConfig(**TINY))
    a, b = (Learner(copy.deepcopy(net), tok, cfg, device="cpu", fast=False) for _ in range(2))
    for _ in range(5):
        ma = a.update(segs)
        mb = b.update_prepared(b.prepare(segs, b.version))
        assert _metrics(ma) == _metrics(mb)
    _same_state(_state(a), _state(b))
    with pytest.raises(AssertionError, match="prepared for version 4 consumed at version 5"):
        b.update_prepared(b.prepare(segs, 4))


# =========================================================================== the loops (T3, T4, T5)


class _Q:
    """msg_q of the learner loop in this process; ``hook`` sees every message."""

    def __init__(self, hook: Any = None) -> None:
        self.items: list[tuple[str, int, dict[str, Any]]] = []
        self.hook = hook

    def put(self, item: tuple[str, int, dict[str, Any]]) -> None:
        self.items.append(item)
        if self.hook is not None:
            self.hook(item)

    def rows(self, kind: str) -> list[dict[str, Any]]:
        return [p for k, _, p in self.items if k == kind]


def _sid(s: Segment) -> tuple[int, int, int]:
    return (s.meta.env_id, s.meta.episode_uid, s.start)


class _Ring(SegmentRing):
    """A SegmentRing that records the segment ids of every batch it hands out and the stale drops before it; with
    ``script``, ``script[j]`` arrives (put) at the first take call after j batches were handed out."""

    def __init__(self, script: list[list[Segment]] | None = None) -> None:
        super().__init__(1 << 30)
        self.script = script
        self.batches: list[list[tuple[int, int, int]]] = []
        self.versions: list[int | None] = []
        self.stale: list[int] = []
        self._arrived = 0

    def take(self, min_samples: int, **kw: Any) -> list[Segment] | None:
        if self.script is not None and self._arrived == len(self.batches) < len(self.script):
            for s in self.script[self._arrived]:
                self.put(s)
            self._arrived += 1
        out = super().take(min_samples, **kw)
        if out is not None:
            self.batches.append([_sid(s) for s in out])
            self.versions.append(kw.get("current_version"))
            self.stale.append(self.stats.stale_dropped_samples)
        return out


def _loop_cfg(tmp_path: Path, name: str, pipeline: bool, **learner: Any) -> dict[str, Any]:
    return {"run_name": name, "run_root": str(tmp_path), "seed": 3, "catalog": "none", "max_updates": N_ITER,
            "model": {"override": TINY},
            "ppo": {"minibatch": 256, "micro_batch": 128, "epochs": 2, "lr": 1e-3, "kl_lr_target": 0.002},
            "learner": {"device": "cuda", "iteration_samples": ITER, "max_staleness": None, "blocking_sync": False,
                        "checkpoint_interval_s": 3600.0, "pipeline_prepare": pipeline, **learner}}


def _run_loop(cfg_d: dict[str, Any], ring: SegmentRing, q: _Q, stop: threading.Event | None = None,
              timeout: float = 300.0) -> dict[str, Any]:
    state: dict[str, Any] = {}
    err: list[BaseException] = []

    def target() -> None:
        try:
            _learner_loop(cfg_d, ring, q, stop or threading.Event(), False, None, None, state)
        except BaseException as e:  # noqa: BLE001 - re-raised below
            err.append(e)

    th = threading.Thread(target=target)
    th.start()
    th.join(timeout)
    assert not th.is_alive(), "the learner loop hangs"
    if err:
        raise err[0]
    assert not [t for t in threading.enumerate() if t.name == "pipeline-preparer"], "the preparer thread survived"
    return state


@pytest.fixture()
def deterministic(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Deterministic CUDA kernels: without them two runs of the same sequential loop already differ (the atomic
    gradient sums of the embedding tables, tools/t3_probe.py); with them both loops must be bit-identical."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="module")
def loop_segments(vocab: Vocab) -> list[Segment]:
    segs = _segments(vocab, Tokenizer(vocab, TokenizerConfig()), 6000)
    assert sum(map(len, segs)) >= (N_ITER + 2) * ITER
    return segs


def _ckpt(run: Path) -> dict[str, Any]:
    ck = torch.load(run / "ckpt" / "latest.pt", map_location="cpu", weights_only=True)
    return {"model": ck["model"], "ema": ck["ema"], "opt": ck["optimizer"],
            "counters": tuple(ck["counters"][k] for k in ("version", "updates", "samples_seen", "ema_steps")),
            "rng": json.dumps(ck["rng"]["numpy"], sort_keys=True)}


def _batch_hash(ids: list[tuple[int, int, int]]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


@cuda
def test_t3_pipelined_loop_equals_the_sequential_loop(tmp_path: Path, loop_segments: list[Segment],
                                                      deterministic: Any) -> None:
    n_fill = 0
    fill: list[Segment] = []
    for s in loop_segments:  # N iterations and less than one more: no batch is left for a ninth take
        if n_fill >= N_ITER * ITER + ITER // 2:
            break
        fill.append(s)
        n_fill += len(s)
    out = {}
    for pipeline in (False, True):
        ring = _Ring()
        for s in fill:
            ring.put(s)
        q = _Q()
        name = "pipe" if pipeline else "seq"
        _run_loop(_loop_cfg(tmp_path, name, pipeline), ring, q)
        rows = q.rows("learner")
        assert len(rows) == N_ITER and all(r.get("pipeline", False) == pipeline for r in rows)
        out[name] = (ring, rows, q.rows("checkpoint"), _ckpt(tmp_path / name))
    (rs, ms, cs, ks), (rp, mp_, cp, kp) = out["seq"], out["pipe"]
    assert [_batch_hash(b) for b in rs.batches] == [_batch_hash(b) for b in rp.batches] and len(rs.batches) == N_ITER
    assert [r["samples_seen"] for r in ms] == [r["samples_seen"] for r in mp_]
    for a, b in zip(ms, mp_):
        assert _metrics(a) == _metrics(b)
    _same_state(ks, kp)
    assert cs[-1]["reason"] == cp[-1]["reason"] == "final" and cp[-1]["pipeline_dropped_samples"] == 0
    for r in mp_:
        assert r["prep_wait_s"] >= 0 and r["handoff_wait_s"] >= 0 and r["time_prep"] > 0


def _scripted(segs: list[Segment], n: int, staleness: int) -> list[list[Segment]]:
    """Arrivals before take j: one iteration of segments acted with version j - 1 (the newest the actors can hold while
    update j - 1 is in flight), and segments of versions j - 1 - staleness (evicted at cv = j, kept at cv = j - 1) and
    j - 2 - staleness (evicted either way)."""
    it = iter(segs)
    out = []
    for j in range(n):
        batch: list[Segment] = []
        for v, samples in ((max(0, j - 1), ITER), (j - 1 - staleness, ITER // 4), (j - 2 - staleness, ITER // 4)):
            if v < 0:
                continue
            got = 0
            while got < samples:
                s = next(it)
                batch.append(dataclasses.replace(s, policy_version=np.full(len(s), v, np.int64)))
                got += len(s)
        out.append(batch)
    return out


@cuda
def test_t4_staleness_eviction_follows_the_cv_rule(tmp_path: Path, loop_segments: list[Segment], vocab: Vocab,
                                                   deterministic: Any) -> None:
    S = 2
    segs = loop_segments + _segments(vocab, Tokenizer(vocab, TokenizerConfig()), 9000)
    script = _scripted(segs, N_ITER, S)
    out = {}
    for pipeline in (False, True):
        ring = _Ring(script)
        q = _Q()
        name = "pipe" if pipeline else "seq"
        _run_loop(_loop_cfg(tmp_path, name, pipeline, max_staleness=S), ring, q)
        rows = q.rows("learner")
        assert len(rows) == N_ITER
        out[name] = (ring, rows)
    (rs, ms), (rp, mp_) = out["seq"], out["pipe"]
    assert rs.versions == rp.versions == list(range(N_ITER))  # the version of every take: cv = the consume version
    assert rs.stale == rp.stale and rs.stale[-1] > 0, (rs.stale, rp.stale)
    assert rs.batches == rp.batches
    for a, b in zip(ms, mp_):
        assert (a["staleness_mean"], a["staleness_max"], a["segment_staleness_hist"]) == \
            (b["staleness_mean"], b["staleness_max"], b["segment_staleness_hist"])
        assert _metrics(a) == _metrics(b)


@cuda
def test_t5_stop_with_a_prepared_batch(tmp_path: Path, loop_segments: list[Segment]) -> None:
    ring = _Ring()
    for s in loop_segments:
        ring.put(s)
    stop = threading.Event()

    def hook(item: tuple[str, int, dict[str, Any]]) -> None:
        if item[0] == "learner" and item[2]["updates"] == 3:
            stop.set()

    q = _Q(hook)
    cfg_d = _loop_cfg(tmp_path, "stop", True)
    cfg_d["max_updates"] = None
    _run_loop(cfg_d, ring, q, stop)
    final = q.rows("checkpoint")[-1]
    assert final["reason"] == "final" and final["updates"] == 3 and final["pipeline"] is True
    ck = torch.load(tmp_path / "stop" / "ckpt" / "latest.pt", map_location="cpu", weights_only=True)
    assert ck["counters"]["updates"] == 3
    # the batch taken for update 4 was prepared (or being prepared) and never consumed
    assert final["pipeline_dropped_samples"] == ring.stats.taken_samples - final["samples_seen"] > 0
    assert len(ring.batches) == 4


@cuda
def test_t5_exception_in_prepare_is_the_learners_failure(tmp_path: Path, loop_segments: list[Segment],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    calls = [0]
    real = Learner.prepare

    def failing(self: Learner, *a: Any, **k: Any) -> Any:
        calls[0] += 1
        if calls[0] == 3:
            raise ValueError("prepare broke")
        return real(self, *a, **k)

    monkeypatch.setattr(Learner, "prepare", failing)
    ring = _Ring()
    for s in loop_segments:
        ring.put(s)
    q = _Q()
    cfg_d = _loop_cfg(tmp_path, "fail", True)
    cfg_d["max_updates"] = None
    err: list[BaseException] = []

    def target() -> None:
        try:
            learner_main(cfg_d, ring, q, threading.Event(), False)
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    th = threading.Thread(target=target)
    t0 = time.monotonic()
    th.start()
    th.join(300)
    assert not th.is_alive() and time.monotonic() - t0 < 300, "a hang"
    assert len(err) == 1 and isinstance(err[0], ValueError) and str(err[0]) == "prepare broke"
    (e,) = q.rows("learner_error")
    assert "prepare broke" in e["error"] and "Traceback" in e["traceback"] and e["updates"] == 2
    assert Path(e["crash_checkpoint"]).name == "crash-u000002.pt" and Path(e["crash_checkpoint"]).exists()
    assert [r["updates"] for r in q.rows("learner")] == [1, 2]


def _pipe_run(tmp_path: Path, name: str, **over: Any) -> TrainConfig:
    cfg = small_config(tmp_path, name, **over)
    raw = json.loads(json.dumps(config_dict(cfg)))
    raw["learner"].update(device="cuda", pipeline_prepare=True)
    return build(TrainConfig, raw)


@cuda
@pytest.mark.parametrize("env,message", [("SPIRE_TEST_PREPARE_FAIL_AT", "injected preparer failure"),
                                         ("SPIRE_TEST_LEARNER_FAIL_AT", "injected learner failure")])
def test_t5_failed_json_through_the_supervisor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: str,
                                               message: str) -> None:
    monkeypatch.setenv(env, "2")
    cfg = _pipe_run(tmp_path, "dead", max_updates=10, duration_s=240)
    t0 = time.monotonic()
    assert Supervisor(cfg).run() == 1
    assert time.monotonic() - t0 < 200
    run = cfg.run_dir
    failed = json.loads((run / "FAILED.json").read_text(encoding="utf-8"))
    assert "learner died" in failed["reason"] and message in failed["learner_error"]
    assert "Traceback" in failed["traceback"]
    crash = Path(failed["crash_checkpoint"])
    assert crash.exists() and crash.name == "crash-u000002.pt"
    rows = [r for r in read_jsonl(run / "metrics.jsonl") if r["kind"] == "learner"]
    assert [r["version"] for r in rows] == [1, 2] and all(r["pipeline"] for r in rows)


@cuda
def test_pipelined_run_end_to_end_and_resume(tmp_path: Path) -> None:
    """Fake hosts, processes topology, the option on: rows carry the pipeline fields; the final checkpoint row counts
    the dropped samples; a plain resume continues (config.json has the key)."""
    cfg = _pipe_run(tmp_path, "e2e", max_updates=4)
    assert Supervisor(cfg).run() == 0
    run = cfg.run_dir
    recs = list(read_jsonl(run / "metrics.jsonl"))
    rows = [r for r in recs if r["kind"] == "learner"]
    assert [r["version"] for r in rows] == [1, 2, 3, 4]
    assert all(r["pipeline"] and r["prep_wait_s"] >= 0 and r["handoff_wait_s"] >= 0 for r in rows)
    final = [r for r in recs if r["kind"] == "checkpoint"][-1]
    assert final["reason"] == "final" and final["pipeline_dropped_samples"] >= 0
    assert json.loads((run / "config.json").read_text(encoding="utf-8"))["learner"]["pipeline_prepare"] is True
    cfg2 = _pipe_run(tmp_path, "e2e", max_updates=6)
    assert Supervisor(cfg2, resume=True).run() == 0
    rows = [r for r in read_jsonl(run / "metrics.jsonl") if r["kind"] == "learner"]
    assert [r["version"] for r in rows] == [1, 2, 3, 4, 5, 6]
