"""py-stage2-2 training features: back-pressure, live sampling weights, resume seed streams, dead learner,
periodic panel evaluation, act-1 strata."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy, validate_distribution
from spire.evalpanel import summarize_strata
from spire.metrics import MetricsWriter, read_jsonl
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.train import Beat, SamplingWeights, Supervisor, TrainConfig, backlog_limits, build, make_throttle
from spire.vocab import Vocab

from test_train import small_config


def _kinds(run: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in read_jsonl(run / "metrics.jsonl"):
        out.setdefault(r["kind"], []).append(r)
    return out


# =========================================================================== back-pressure


def test_throttle_hysteresis() -> None:
    cfg = build(TrainConfig, {"run_name": "x", "learner": {"iteration_samples": 100, "backpressure": "throttle",
                                                           "backlog_high_iterations": 2.0,
                                                           "backlog_low_iterations": 1.0}})
    assert backlog_limits(cfg) == (200, 100)
    cell = Beat(0)
    th = make_throttle(cfg, cell)
    assert th is not None
    seq = []
    for b in (0, 150, 199, 200, 250, 150, 101, 100, 50, 199, 200):
        cell.value = b
        seq.append(th())
    assert seq == [False, False, False, True, True, True, True, False, False, False, True]
    for mode in ("drop_oldest", "none"):
        assert make_throttle(build(TrainConfig, {"run_name": "x", "learner": {"backpressure": mode}}), cell) is None


def test_actor_throttle_holds_frames_and_books_the_time(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=40, hashes=False)) for i in range(4)]
    t0 = time.monotonic()
    decisions_while: list[int] = []
    reports: list[dict[str, Any]] = []

    actor: Actor

    def throttled() -> bool:
        on = time.monotonic() - t0 < 0.6
        if on:
            decisions_while.append(actor.counters.decisions)
        return on
    actor = Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
                  ActorConfig(report_interval=0.25), store=store, throttle=throttled, on_report=reports.append)
    actor.run(max_seconds=1.2)
    assert decisions_while and max(decisions_while) == 0  # nothing was decided while throttled
    assert actor.counters.decisions > 50  # resumed afterwards
    assert sum(r["throttled_s"] for r in reports) == pytest.approx(0.6, abs=0.2)
    assert sum(len(s) for s in segs) + store.stats.steps_dropped + sum(
        t.n_pending for t in store._tracks.values()) == store.stats.steps_added  # accounting intact


# =========================================================================== sampling weights


def test_validate_distribution(vocab: Vocab) -> None:
    base = EpisodeDist(lineups=(("IRONCLAD", 1.0),), seeds="csprng")
    d = validate_distribution({"lineups": [["SILENT", 2], ["IRONCLAD+SILENT", 1]]}, vocab, base)
    assert d.lineups == (("SILENT", 2.0), ("IRONCLAD+SILENT", 1.0)) and d.ascensions == base.ascensions
    assert d.seeds == "csprng"
    bad = [({"lineups": []}, "non-empty"), ({"lineups": [["NOBODY", 1]]}, "unknown"),
           ({"lineups": [["IRONCLAD", -1]]}, ">= 0"), ({"lineups": [["IRONCLAD", 0]]}, "all weights are zero"),
           ({"lineups": [["IRONCLAD", float("nan")]]}, "finite"), ({"ascensions": [[25, 1]]}, "0..20"),
           ({"lineups": [["IRONCLAD", 1], ["IRONCLAD", 2]]}, "duplicate"), ({"weights": {}}, "unknown keys"),
           ({"lineups": [["IRONCLAD+IRONCLAD+IRONCLAD+IRONCLAD+IRONCLAD", 1]]}, "1-4"), ([1, 2], "object")]
    for data, msg in bad:
        with pytest.raises(ValueError, match=msg):
            validate_distribution(data, vocab, base)  # type: ignore[arg-type]


def test_sampling_weights_file_round_trip(vocab: Vocab, tmp_path: Path) -> None:
    cfg = build(TrainConfig, {"run_name": "s", "run_root": str(tmp_path), "sampling_poll_s": 0.0,
                              "episodes": {"lineups": [["IRONCLAD", 0.7], ["SILENT", 0.3]]}})
    run = cfg.run_dir
    run.mkdir(parents=True)
    m = MetricsWriter(run / "metrics.jsonl")
    sw = SamplingWeights(cfg, run, m)
    tmpl = json.loads((run / "sampling.json").read_text(encoding="utf-8"))
    assert tmpl["lineups"] == [["IRONCLAD", 0.7], ["SILENT", 0.3]] and sw.active["version"] == 1
    sampler = EpisodeSampler(cfg.episodes, vocab, 0, weights_file=run / "sampling.active.json", poll_s=0.0)
    assert sampler.weights_version == 1
    time.sleep(0.05)
    (run / "sampling.json").write_text('{"lineups": [["SILENT", 1.0], ["NOBODY", 1.0]]}', encoding="utf-8")
    assert not sw.poll() and sw.active["version"] == 1  # rejected, still the old weights
    time.sleep(0.05)
    (run / "sampling.json").write_text('{"lineups": [["SILENT", 1.0]], "ascensions": [[0, 1], [1, 1]]}',
                                       encoding="utf-8")
    assert sw.poll() and sw.active["version"] == 2
    jobs = [sampler.next_job(0) for _ in range(200)]
    assert sampler.weights_version == 2 and {j.lineup for j in jobs} == {"SILENT"}
    assert {j.ascension for j in jobs} == {0, 1}
    m.close()
    ev = [r for r in read_jsonl(run / "metrics.jsonl") if r.get("what") == "sampling"]
    assert [e["applied"] for e in ev] == [True, False, True] and "NOBODY" in ev[1]["error"]


# =========================================================================== resume / dead learner (processes)


def test_resumed_runs_use_new_seed_streams(tmp_path: Path) -> None:
    cfg = small_config(tmp_path, "seeds", max_updates=2)
    assert Supervisor(cfg).run() == 0
    run = cfg.run_dir
    first = {e["seed"] for e in read_jsonl(run / "episodes.jsonl")}
    assert json.loads((run / "config.json").read_text(encoding="utf-8"))["run_start"] == 1
    assert Supervisor(small_config(tmp_path, "seeds", max_updates=4), resume=True).run() == 0
    eps = list(read_jsonl(run / "episodes.jsonl"))
    second = {e["seed"] for e in eps} - first
    assert len(first) > 20 and len(second) > 20
    assert json.loads((run / "train_state.json").read_text(encoding="utf-8"))["starts"] == 2
    # the first episodes of the second start are not a replay of the first start's (py-review 5.6)
    later = [e["seed"] for e in eps[len(first):]][:20]
    assert not set(later) & first


def test_dead_learner_stops_the_run_and_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPIRE_TEST_LEARNER_FAIL_AT", "2")
    cfg = small_config(tmp_path, "dead", max_updates=10, duration_s=240)
    sup = Supervisor(cfg)
    t0 = time.monotonic()
    assert sup.run() == 1
    assert time.monotonic() - t0 < 200
    run = cfg.run_dir
    failed = json.loads((run / "FAILED.json").read_text(encoding="utf-8"))
    assert "learner died" in failed["reason"] and "injected learner failure" in failed["learner_error"]
    assert "Traceback" in failed["traceback"] and failed["resume_command"].endswith("--resume")
    crash = Path(failed["crash_checkpoint"])
    assert crash.exists() and crash.name == "crash-u000002.pt"
    k = _kinds(run)
    assert any(e.get("what") == "fatal" for e in k["event"])
    assert any(e.get("what") == "learner_error" for e in k["event"])
    state = json.loads((run / "train_state.json").read_text(encoding="utf-8"))
    assert state["updates"] == 2
    monkeypatch.delenv("SPIRE_TEST_LEARNER_FAIL_AT")
    cfg2 = small_config(tmp_path, "dead", max_updates=4, resume_from=str(crash))
    assert Supervisor(cfg2, resume=True).run() == 0
    assert not (run / "FAILED.json").exists()
    versions = [r["version"] for r in _kinds(run)["learner"]]
    assert versions == [1, 2, 3, 4]  # continued from the crash checkpoint's version 2
    saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert saved["resume_from"] == ""  # a later plain --resume continues from ckpt/latest.pt


# =========================================================================== periodic evaluation


def test_periodic_eval_plays_the_panel_and_never_feeds_training(tmp_path: Path) -> None:
    panel = tmp_path / "panels" / "unit-eval.json"
    cfg = small_config(tmp_path, "evalrun", max_updates=None, duration_s=50,
                       learner={"device": "cpu", "iteration_samples": 384, "micro_tokens": None,
                                "checkpoint_interval_s": 2.0},
                       eval={"enabled": True, "panel": str(panel), "panel_seeds": 6, "interval_s": 1.0,
                             "first_after_s": 1.0, "envs": 2, "max_fraction": 100.0,
                             "results_dir": str(tmp_path / "results"), "priority": ""})
    assert Supervisor(cfg).run() == 0
    run = cfg.run_dir
    k = _kinds(run)
    evals = k.get("eval", [])
    assert evals, [e for e in k["event"] if e.get("what") == "eval"]
    e0 = evals[0]
    assert e0["episodes"] == 6 * 2 and e0["summary"]["all"]["n"] == 12 and e0["decisions"] > 0
    assert Path(e0["path"]).exists() and Path(e0["path"]).parent == tmp_path / "results" / "unit-eval"
    seeds = set(json.loads(panel.read_text(encoding="utf-8"))["seeds"])
    train_seeds = {e["seed"] for e in read_jsonl(run / "episodes.jsonl")}
    assert train_seeds and not seeds & train_seeds  # panel seeds excluded, eval episodes not in training
    last = k["train"][-1]
    assert last["eval_decisions"] >= e0["decisions"] and 0 < last["eval_fraction"]


def test_act1_strata() -> None:
    recs = [{"lineup": "IRONCLAD", "ascension": 0, "result": r, "floor": 5, "act1": a}
            for r, a in [("win", "OVERGROWTH"), ("loss", "OVERGROWTH"), ("win", "UNDERDOCKS"), ("win", "UNDERDOCKS"),
                         ("loss", "")]]
    s = summarize_strata(recs)
    assert s["IRONCLAD|A0|OVERGROWTH"]["n"] == 2 and s["IRONCLAD|A0|UNDERDOCKS"]["win_rate"] == 1.0
    assert s["IRONCLAD|A0|?"]["n"] == 1
    st = s["IRONCLAD|A0|stratified"]
    assert st["n"] == 5 and st["win_rate"] == pytest.approx(0.6) and sum(st["act1_share"].values()) == pytest.approx(1)


def test_episode_records_carry_act1(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    recs: list[Any] = []
    store = RolloutStore(lambda s: None, RoomIds.from_vocab(vocab))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=30, hashes=False)) for i in range(3)]
    Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0), ActorConfig(), store=store,
          on_episode=recs.append).run(max_decisions=600)
    done = [r for r in recs if r.result in ("win", "loss")]
    assert done and all(r.act1 in vocab["act"].entries for r in done)


