"""learner.confident_keep / confident_tau (notes/next/round21-1121.md): the learner trains on a share ``keep`` of the
samples it is already sure about (behaviour probability of the taken action >= tau).

* the selection: its rate, the same rows for the same run seed and update index (resume, both learner cards), other
  rows for another update or seed; every non-confident row kept;
* the row fields and that the minibatches hold exactly the kept rows, each once per epoch;
* the three refusals (kickstart, recompute_values, the critic), the ranges, config.json without the keys at their
  defaults;
* keep 1.0 is the learner without the option: 3 updates on test_learner's Harness, rows and checkpoints equal;
* keep 0.2 end to end on fake hosts in the threads topology;
* the dual learner (two cards) with keep 0.2: all on A is the single learner bit for bit, the split keeps the counts."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from spire.learner import Learner
from spire.rollout import build_batch
from spire.metrics import read_jsonl
from spire.train import Supervisor, TrainConfig, build, check_confident, config_dict, load_config
from spire.vocab import Vocab

from test_dual import TINY, _bit_equal, _cfg, _learners, _metrics, _segments, _state, _tok, det, table_path  # noqa: F401
from test_dual import two_cards
from test_learner import FAST, Harness, strip_times
from test_train import small_config

CONF = ("conf_keep", "conf_tau", "conf_taken", "conf_dropped", "kept", "conf_share_by_kind")


class _Rows:
    """The part of a RolloutBatch the selection reads."""

    def __init__(self, logp: np.ndarray) -> None:
        self.logp = logp.astype(np.float32)

    def __len__(self) -> int:
        return len(self.logp)


def _logp(n: int, seed: int = 0) -> np.ndarray:
    """Half the rows confident at tau .95 (p in [.96, 1]), half not (p in [.05, .9])."""
    g = np.random.default_rng(seed)
    p = np.where(np.arange(n) % 2 == 0, g.uniform(0.96, 1.0, n), g.uniform(0.05, 0.9, n))
    return np.log(p)


def _learner(vocab: Vocab, **kw: Any) -> Learner:
    return Harness(vocab, seed=5, n_envs=2, **kw).learner


# =========================================================================== the selection


def test_selection_rate_and_unconfident_rows(vocab: Vocab) -> None:
    lr = _learner(vocab, confident_keep=0.2, confident_tau=0.95, confident_seed=11)
    rb = _Rows(_logp(200_000))
    rows, conf = lr._kept_rows(rb)
    assert conf is not None and conf.sum() == 100_000 and (conf == (np.arange(200_000) % 2 == 0)).all()
    kept = np.zeros(200_000, bool)
    kept[rows] = True
    assert kept[~conf].all()  # every non-confident row is kept
    rate = kept[conf].mean()
    assert abs(rate - 0.2) < 4 * np.sqrt(0.2 * 0.8 / 100_000), rate
    assert (np.diff(rows) > 0).all()  # each row once


def test_selection_is_a_function_of_seed_and_update(vocab: Vocab) -> None:
    rb = _Rows(_logp(10_000))
    a = _learner(vocab, confident_keep=0.3, confident_seed=11)
    b = _learner(vocab, confident_keep=0.3, confident_seed=11, cfg=FAST)
    other_seed = _learner(vocab, confident_keep=0.3, confident_seed=12)
    r0 = a._kept_rows(rb)[0]
    assert np.array_equal(r0, a._kept_rows(rb)[0])  # the same update: the same rows
    assert np.array_equal(r0, b._kept_rows(rb)[0])  # another learner of the run (a resume, the other card)
    assert not np.array_equal(r0, other_seed._kept_rows(rb)[0])
    a.updates = 1
    r1 = a._kept_rows(rb)[0]
    assert not np.array_equal(r0, r1)
    b.updates = 1
    assert np.array_equal(r1, b._kept_rows(rb)[0])


def test_selection_ignores_the_ppo_config(vocab: Vocab) -> None:
    """The rows depend on the run seed, the update index, the behaviour log-probabilities, keep and tau only (not on
    ppo.seed, the minibatch size or the learner's own generator): one selection for every device split."""
    rb = _Rows(_logp(10_000, seed=3))
    a = _learner(vocab, confident_keep=0.2, confident_seed=7)
    b = _learner(vocab, cfg=replace(FAST, seed=99, minibatch=512), confident_keep=0.2, confident_seed=7)
    b.rng.random(100)
    assert np.array_equal(a._kept_rows(rb)[0], b._kept_rows(rb)[0])


def test_keep_one_is_every_row(vocab: Vocab) -> None:
    rows, conf = _learner(vocab, confident_keep=1.0, confident_tau=0.5)._kept_rows(_Rows(_logp(1000)))
    assert conf is None and np.array_equal(rows, np.arange(1000))


# =========================================================================== the learner


def test_rows_and_minibatches_hold_the_kept_rows(vocab: Vocab, monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(vocab, seed=3, n_envs=16, confident_keep=0.2, confident_tau=0.4, confident_seed=21)
    lr = h.learner
    seen: list[np.ndarray] = []
    orig = Learner._minibatch

    def rec(self: Learner, samples: Any, mb: np.ndarray, *a: Any, **k: Any) -> Any:
        seen.append(np.array(mb))
        return orig(self, samples, mb, *a, **k)

    monkeypatch.setattr(Learner, "_minibatch", rec)
    taken = 0
    for u in range(3):
        segs = h.segments(1024)
        rb = build_batch(segs, lr.cfg.gae, lr.version, with_samples=False)
        want, conf = lr._kept_rows(rb)
        assert conf is not None
        seen.clear()
        m = lr.update(segs)
        n = len(rb)
        taken += n
        assert m["samples"] == n and m["samples_seen"] == taken == lr.samples_seen
        assert m["conf_keep"] == 0.2 and m["conf_tau"] == 0.4
        assert m["conf_taken"] == int((np.exp(rb.logp.astype(np.float64)) >= 0.4).sum()) == int(conf.sum()) > 0
        assert m["kept"] == len(want) and m["conf_dropped"] == n - len(want) > 0
        assert m["minibatches"] == max(1, round(len(want) / lr.cfg.minibatch))
        epochs = len(seen) // m["minibatches"]
        assert epochs * m["minibatches"] == len(seen) and epochs == round(m["epochs_done"])
        for e in range(epochs):  # every epoch: the kept rows, each once
            got = np.sort(np.concatenate(seen[e * m["minibatches"]:(e + 1) * m["minibatches"]]))
            assert np.array_equal(got, want), (u, e)
        kinds = {k for k in m["sample_counts_by_kind"]}
        assert set(m["conf_share_by_kind"]) == kinds  # over the taken rows, as the per-kind counts
        assert sum(m["sample_counts_by_kind"].values()) == n
        assert all(0.0 <= v <= 1.0 for v in m["conf_share_by_kind"].values())
        assert np.isfinite([m["loss_pi"], m["loss_v"], m["approx_kl"], m["entropy"]]).all()


def test_keep_one_is_the_learner_without_the_option(vocab: Vocab, tmp_path: Path) -> None:
    base = Harness(vocab, seed=7, n_envs=8)
    off = Harness(vocab, seed=7, n_envs=8, confident_keep=1.0, confident_tau=0.5, confident_seed=1234)
    for _ in range(3):
        mb, mo = strip_times(base.step(512)), strip_times(off.step(512))
        assert mb == mo and not set(CONF) & set(mo)
    base.learner.save(tmp_path / "base.pt")
    off.learner.save(tmp_path / "off.pt")
    a = torch.load(tmp_path / "base.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "off.pt", map_location="cpu", weights_only=False)
    _same(a, b)


def _same(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif torch.is_tensor(a):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    else:
        assert a == b or (a != a and b != b), path


def test_learner_refuses_out_of_range_and_the_three(vocab: Vocab) -> None:
    for kw, msg in (({"confident_keep": 0.0}, "confident_keep 0.0"), ({"confident_keep": 1.5}, "confident_keep 1.5"),
                    ({"confident_tau": 0.0}, "confident_tau 0.0")):
        with pytest.raises(AssertionError, match=msg):
            _learner(vocab, **kw)
    with pytest.raises(AssertionError, match="confident_keep < 1 refuses"):
        _learner(vocab, cfg=replace(FAST, recompute_values=True), confident_keep=0.5)
    _learner(vocab, cfg=replace(FAST, recompute_values=True), confident_keep=1.0)


# =========================================================================== config


REFUSED = [
    ({"ppo": {"kickstart": {"teacher": "t.pt"}}}, "an active kickstart"),
    ({"ppo": {"recompute_values": True}}, "ppo.recompute_values"),
    ({"model": {"critic": {}}, "ppo": {"critic": {}}}, "the critic (ppo.critic / model.critic)"),
]


def _raw(keep: float = 0.2, **over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {"run_name": "x", "learner": {"confident_keep": keep}}
    for k, v in over.items():
        raw[k] = {**raw.get(k, {}), **v}
    return raw


def _esc(s: str) -> str:
    return s.replace(".", r"\.").replace("(", r"\(").replace(")", r"\)")


@pytest.mark.parametrize("over,message", REFUSED, ids=[m for _, m in REFUSED])
def test_refusals(over: dict[str, Any], message: str) -> None:
    with pytest.raises(AssertionError, match="learner.confident_keep < 1 refuses " + _esc(message)):
        check_confident(build(TrainConfig, _raw(**over)))
    check_confident(build(TrainConfig, _raw(1.0, **over)))  # keep 1: nothing is refused


@pytest.mark.parametrize("learner,message", [({"confident_keep": 0.0}, "learner.confident_keep 0.0"),
                                             ({"confident_keep": 1.2}, "learner.confident_keep 1.2"),
                                             ({"confident_tau": 0.0}, "learner.confident_tau 0.0"),
                                             ({"confident_tau": 1.5}, "learner.confident_tau 1.5")])
def test_ranges(learner: dict[str, Any], message: str) -> None:
    with pytest.raises(AssertionError, match=_esc(message)):
        check_confident(build(TrainConfig, {"run_name": "x", "learner": learner}))


def test_config_json_and_load_config(tmp_path: Path) -> None:
    off = build(TrainConfig, {"run_name": "x"})
    assert (off.learner.confident_keep, off.learner.confident_tau) == (1.0, 0.95)
    assert not {"confident_keep", "confident_tau"} & set(config_dict(off)["learner"])  # the config.json of tree p10
    on = build(TrainConfig, _raw(learner={"confident_tau": 0.9}))
    d = config_dict(on)
    assert d["learner"]["confident_keep"] == 0.2 and d["learner"]["confident_tau"] == 0.9
    p = tmp_path / "config.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    got = load_config(p)
    assert (got.learner.confident_keep, got.learner.confident_tau) == (0.2, 0.9)
    p.write_text(json.dumps(_raw(ppo={"recompute_values": True})), encoding="utf-8")
    with pytest.raises(AssertionError, match=r"confident_keep < 1 refuses ppo\.recompute_values"):
        load_config(p)  # load_config runs check_confident


def test_production_shape_is_accepted() -> None:
    from test_dual import _raw as dual_raw
    raw = dual_raw(learner={"cuda_graphs": True, "compile": True, "pipeline_prepare": True, "acting_weights": "ema",
                            "confident_keep": 0.2, "confident_tau": 0.95, "iteration_samples": 86016},
                   actor={"device": "cuda", "devices": ["cuda:2", "cuda:3"]}, eval={"device": "cuda:3"},
                   ppo={"target_kl": 0.03, "kl_lr_target": 0.01})
    raw["learner"]["device"] = "cuda:0"
    check_confident(build(TrainConfig, raw))


# =========================================================================== threads topology end to end


def test_threads_end_to_end_keep_02(tmp_path: Path) -> None:
    d = config_dict(small_config(tmp_path, "conf-thr", topology="threads", max_updates=4))
    d["learner"]["confident_keep"] = 0.2
    d["learner"]["confident_tau"] = 0.5
    cfg = build(TrainConfig, d)
    assert Supervisor(cfg).run() == 0
    run = cfg.run_dir
    rows = [r for r in read_jsonl(run / "metrics.jsonl") if r["kind"] == "learner"]
    assert [r["version"] for r in rows] == [1, 2, 3, 4]
    for r in rows:
        assert all(k in r for k in CONF), sorted(r)
        assert r["conf_keep"] == 0.2 and r["conf_tau"] == 0.5
        assert r["kept"] + r["conf_dropped"] == r["samples"] >= 384
        assert 0 <= r["conf_dropped"] <= r["conf_taken"] <= r["samples"]
    assert sum(r["conf_taken"] for r in rows) > 0 and sum(r["conf_dropped"] for r in rows) > 0
    assert rows[-1]["samples_seen"] == sum(r["samples"] for r in rows)
    on_disk = json.loads((run / "config.json").read_text(encoding="utf-8"))["learner"]
    assert on_disk["confident_keep"] == 0.2 and on_disk["confident_tau"] == 0.5


# =========================================================================== dual learner (two cards)


@two_cards
def test_dual_keep_02(vocab: Vocab, table_path: str, det: Any) -> None:  # noqa: F811
    variant = "v2a-free"
    tok = _tok(vocab, variant, table_path)
    segs = _segments(vocab, variant, tok)
    cfg = _cfg()

    def make(replica: bool, n: int = 1) -> list[Learner]:
        out = _learners(tok, cfg, replica=replica, n=n)
        for lr in out:
            lr.confident_keep, lr.confident_tau, lr.confident_seed = 0.2, 0.3, 17
        return out

    single, = make(False)
    on_a, split = make(True, 2)
    assert on_a.replica is not None and split.replica is not None
    on_a.replica.split = "a"
    for k in range(3):
        ms, ma, md = single.update(segs), on_a.update(segs), split.update(segs)
        assert ms["conf_dropped"] > 0 and ms["kept"] < ms["samples"]
        assert _metrics(ms) == _metrics(ma), k  # all on A: the single learner bit for bit
        for key in CONF + ("samples", "minibatches", "micro_batches", "epochs_done"):
            assert ms[key] == md[key], (k, key)
        assert md["dual"]["micro_a"] > 0 and md["dual"]["micro_b"] > 0
    _bit_equal(_state(single), _state(on_a))
