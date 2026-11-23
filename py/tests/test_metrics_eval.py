"""metrics.py statistics / files / aggregators and evalpanel.py panels, tests and evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from spire.actor import UniformPolicy, seed_to_wire
from spire.evalpanel import (
    SPRT,
    Panel,
    evaluate,
    load_episodes,
    load_panel,
    make_panel,
    mcnemar_exact,
    paired_compare,
    save_panel,
    summarize,
)
from spire.metrics import (
    ActionStats,
    EpisodeRecord,
    JsonlWriter,
    MetricsWriter,
    OutcomeTracker,
    _z,
    merge_action_stats,
    read_jsonl,
    wilson_interval,
    wilson_lower,
)
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.vocab import Vocab


# --------------------------------------------------------------------------- statistics


def test_normal_quantile() -> None:
    for q, z in ((0.975, 1.959963985), (0.95, 1.644853627), (0.5, 0.0), (0.001, -3.090232306), (0.99, 2.326347874)):
        assert abs(_z(q) - z) < 1e-6


def test_wilson_matches_design_review_thresholds() -> None:
    # design-review-rl 4.1: wins needed for a one-sided 95% Wilson lower bound above the target
    for n, target, need in ((300, 0.90, 279), (300, 0.80, 252), (300, 0.50, 165), (500, 0.90, 462),
                            (500, 0.80, 415), (500, 0.50, 269), (300, 0.95, 292)):
        assert wilson_lower(need, n) > target >= wilson_lower(need - 1, n), (n, target)
    lo, hi = wilson_interval(83, 256)  # legacy fast agent, Silent A0
    assert abs(lo - 0.270) < 0.002 and abs(hi - 0.384) < 0.002
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(128, 256)
    assert abs((hi - lo) / 2 - 0.061) < 0.001  # +-6.1pp at N=256, p=0.5


def test_mcnemar_and_sprt() -> None:
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(5, 5) == 1.0 and mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 30) == pytest.approx(2 * sum(math.comb(40, i) for i in range(11)) / 2**40)
    rng = np.random.default_rng(0)
    decisions, ns = [], []
    for _ in range(200):
        t = SPRT(0.90, 0.95)
        t.run(rng.random(2000) < 0.95)
        decisions.append(t.decision)
        ns.append(t.n)
    assert decisions.count("H1") >= 185 and 80 < np.mean(ns) < 260  # design-review: E[N|H1] ~ 159
    t = SPRT(0.90, 0.95)
    assert t.run(rng.random(2000) < 0.88) == "H0"
    with pytest.raises(ValueError):
        SPRT(0.9, 0.8)


# --------------------------------------------------------------------------- files and aggregators


def test_jsonl_writer_and_torn_lines(tmp_path: Path) -> None:
    m = MetricsWriter(tmp_path / "m.jsonl", elapsed_offset=100.0)
    rec = m.emit("train", x=float("nan"), arr=np.arange(3), v=np.float32(1.5), nested={"inf": float("inf")})
    assert rec["elapsed"] >= 100.0
    m.close()
    with open(tmp_path / "m.jsonl", "a", encoding="utf-8") as f:
        f.write('{"kind": "torn", "x": ')
    rows = list(read_jsonl(tmp_path / "m.jsonl"))
    assert len(rows) == 1 and rows[0]["x"] is None and rows[0]["arr"] == [0, 1, 2] and rows[0]["nested"]["inf"] is None
    assert list(read_jsonl(tmp_path / "missing.jsonl")) == []


def test_outcome_tracker_and_action_stats() -> None:
    t = OutcomeTracker(window=10)
    for i in range(30):
        t.add(EpisodeRecord("IRONCLAD", 0, "win" if i % 3 == 0 else "loss", floor=10 + i % 5,
                            enemies=("M1", "M2") if i % 3 else ()))
    t.add(EpisodeRecord("IRONCLAD", 0, "lost"))
    t.add(EpisodeRecord("SILENT+SILENT", 10, "win", floor=51))
    s = t.summary()
    o = s["IRONCLAD|A0"]
    assert o["n"] == 10 and o["total_finished"] == 30 and o["total_wins"] == 10 and o["failures"] == {"lost": 1}
    assert o["wins"] == sum(1 for i in range(20, 30) if i % 3 == 0)
    assert sum(o["death_floor"].values()) == 10 - o["wins"] and o["top_killers"][0][0] == "M1+M2"
    assert s["SILENT+SILENT|A10"]["win_rate"] == 1.0
    a = ActionStats()
    a.add(np.array([2, 2, 1]), np.array([3, 1, 6]), np.array([1.0, 2.0, 0.5]))
    m = merge_action_stats([a.take(), a.take()])
    assert m["action_freq"]["COMBAT"]["PLAY_CARD"] == 0.5 and m["entropy_by_kind"]["COMBAT"] == 1.5
    assert m["action_freq"]["MAP"]["TRAVEL"] == 1.0


# --------------------------------------------------------------------------- panels and evaluation


def test_panel_generation_is_frozen(tmp_path: Path) -> None:
    p = make_panel("t", 300, ["IRONCLAD", "SILENT"], [0, 10], rng_seed=1, exclude=["000000000000"])
    assert len(set(p.seeds)) == 300 and "000000000000" not in p.seeds
    assert all(seed_to_wire(s) >= 0 for s in p.seeds) and len(p.configs) == 4
    path = save_panel(p, tmp_path)
    assert load_panel(path).seeds == p.seeds and load_panel("t", tmp_path) == load_panel(path)
    assert save_panel(p, tmp_path) == path  # identical: fine
    with pytest.raises(FileExistsError):
        save_panel(make_panel("t", 300, ["IRONCLAD"], [0], rng_seed=2), tmp_path)
    q = make_panel("u", 50, ["IRONCLAD"], [0], exclude=p.seeds)  # CSPRNG
    assert not set(q.seeds) & set(p.seeds) and q.generator["source"] == "csprng"


def _records(results: list[str]) -> list[dict]:
    return [EpisodeRecord("IRONCLAD", 0, r, seed=f"{i:012d}").as_dict() for i, r in enumerate(results)]


def test_summary_counts_failures_and_paired_compare() -> None:
    a = _records(["win"] * 6 + ["loss"] * 3 + ["lost"])
    s = summarize(a)
    assert s["all"]["n"] == 10 and s["all"]["wins"] == 6 and s["all"]["win_rate"] == 0.6
    assert s["IRONCLAD|A0"]["failures"] == {"lost": 1}
    b = _records(["win", "loss", "win", "win", "loss", "loss", "win", "win", "loss", "win"])
    c = paired_compare(a, b)["all"]
    assert (c["pairs"], c["both"], c["a_only"], c["b_only"], c["neither"]) == (10, 3, 3, 3, 1)
    assert c["p_mcnemar"] == 1.0 and c["diff"] == 0.0 and c["a_rate"] == 0.6


def test_evaluate_on_fake_hosts_is_reproducible(vocab: Vocab, tmp_path: Path) -> None:
    panel = Panel("fakepanel", tuple(make_panel("x", 12, ["IRONCLAD"], [0], rng_seed=3).seeds),
                  ("IRONCLAD", "SILENT+DEFECT"), (0, 10))
    tok = Tokenizer(vocab)
    cfg = FakeHostConfig(mean_decisions=30, hashes=False)

    def run(tag: str, seed: int) -> dict:
        hosts = [FakeHost(vocab, seed=10 + i, config=cfg) for i in range(5)]
        res, path = evaluate(panel, UniformPolicy(seed), tok, hosts, tag=tag, policy_info={"kind": "uniform"},
                             out_dir=tmp_path)
        assert path.exists() and (tmp_path / "fakepanel" / f"{tag}.episodes.jsonl").exists()
        assert len(res.episodes) == 48 and res.summary["all"]["n"] == 48
        assert set(res.summary) == {"IRONCLAD|A0", "IRONCLAD|A10", "DEFECT+SILENT|A0", "DEFECT+SILENT|A10", "all"}
        return json.loads(path.read_text(encoding="utf-8"))

    r1, r2 = run("a", 0), run("b", 1)
    # the fake outcome is a function of the seed only: identical results, McNemar p = 1
    key = lambda e: (e["seed"], e["lineup"], e["ascension"])  # noqa: E731
    assert {key(e): e["result"] for e in r1["episodes"]} == {key(e): e["result"] for e in r2["episodes"]}
    cmp = paired_compare(load_episodes(tmp_path / "fakepanel" / "a.json"),
                         load_episodes(tmp_path / "fakepanel" / "b.episodes.jsonl"))
    assert cmp["all"]["pairs"] == 48 and cmp["all"]["a_only"] == cmp["all"]["b_only"] == 0


def test_jsonl_writer_thread_safe(tmp_path: Path) -> None:
    import threading
    w = JsonlWriter(tmp_path / "e.jsonl")
    def burst(k: int) -> None:
        for i in range(500):
            w.write({"k": k, "i": i})

    ts = [threading.Thread(target=burst, args=(k,)) for k in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    w.close()
    assert len(list(read_jsonl(tmp_path / "e.jsonl"))) == 2000
