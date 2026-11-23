"""train.py: config parsing, both topologies end to end (fake host processes), watchdog, resume; dashboard."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from spire.dashboard import build_html, write_dashboard
from spire.learner import PPOConfig
from spire.metrics import MetricsWriter, read_jsonl
from spire.train import Supervisor, TrainConfig, build, load_config


def small_config(tmp_path: Path, name: str, **over: Any) -> TrainConfig:
    raw: dict[str, Any] = {
        "run_name": name, "run_root": str(tmp_path), "topology": "processes", "max_updates": 3,
        "metrics_interval_s": 1.0, "gpu_interval_s": 3600.0, "heartbeat_timeout_s": 60.0,
        "model": {"override": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64, "d_id": 8, "d_cond": 8,
                               "d_static": 16}},
        "catalog": "none",
        "ppo": {"minibatch": 256, "micro_batch": 256, "epochs": 1, "lr": 1e-3},
        "rollout": {"segment_len": 32},
        "episodes": {"lineups": [["IRONCLAD", 1.0], ["SILENT", 1.0]]},
        "actor": {"actors": 1, "device": "cpu", "report_interval": 0.5, "max_batch": 8},
        "hosts": {"envs_per_actor": 4, "fake": {"learnable": True, "mean_decisions": 40, "hashes": False}},
        "learner": {"device": "cpu", "iteration_samples": 384, "micro_tokens": None, "checkpoint_interval_s": 3600},
        "shm_mb": 16,
    }
    raw.update(over)
    return build(TrainConfig, raw)


def test_config_parsing(tmp_path: Path) -> None:
    p = tmp_path / "c.toml"
    p.write_text('run_name = "x"\nseed = 3\n[ppo]\nlr = 1e-3\nbetas = [0.9, 0.99]\nent_coef_by_kind = [[2, 0.003]]\n'
                 '[ppo.gae]\nlam = 0.95\n[episodes]\nlineups = [["IRONCLAD", 1], ["SILENT+SILENT", 2]]\n'
                 '[hosts]\ncommand = ["a.exe", "--serve"]\n[hosts.fake]\nlearnable = true\n', encoding="utf-8")
    c = load_config(p, duration_s=5)
    assert c.run_name == "x" and c.seed == 3 and c.duration_s == 5
    assert isinstance(c.ppo, PPOConfig) and c.ppo.betas == (0.9, 0.99) and c.ppo.gae.lam == 0.95
    assert c.ppo.ent_coef_by_kind == ((2, 0.003),) and c.episodes.lineups[1] == ("SILENT+SILENT", 2.0)
    assert c.hosts.command == ("a.exe", "--serve") and c.hosts.fake.learnable
    p.write_text('run_name = "x"\n[ppo]\nlearning_rate = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(p)
    js = tmp_path / "c.json"
    js.write_text(json.dumps({"run_name": "y", "topology": "threads"}), encoding="utf-8")
    assert load_config(js).topology == "threads"


def _kinds(run: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in read_jsonl(run / "metrics.jsonl"):
        out.setdefault(r["kind"], []).append(r)
    return out


def test_processes_topology_watchdog_and_resume(tmp_path: Path) -> None:
    cfg = small_config(tmp_path, "proc")
    sup = Supervisor(cfg)
    code: list[int] = []
    th = threading.Thread(target=lambda: code.append(sup.run()))
    th.start()
    try:
        deadline = time.monotonic() + 180
        while not (sup.slots and sup.slots[0].proc is not None and sup.agg.totals["decisions"] > 100):
            assert time.monotonic() < deadline and th.is_alive()
            time.sleep(0.2)
        first_pid = sup.slots[0].proc.pid
        sup.slots[0].proc.kill()  # the watchdog must restart the actor
        th.join(300)
    finally:
        if th.is_alive():
            sup.request_stop("test timeout")
            th.join(120)
    assert code == [0] and sup.stop_reason == "max_updates"
    run = cfg.run_dir
    k = _kinds(run)
    assert any(e.get("what") == "actor_restart" for e in k["event"])
    assert sup.slots[0].proc.pid != first_pid and sup.agg.totals["actor_restarts"] == 1
    assert len(k["learner"]) == 3 and k["learner"][-1]["version"] == 3
    last = k["train"][-1]
    assert last["decisions"] > 0 and last["stage_us"]["encode"] > 0 and last["outcomes"]
    assert last["counts"]["actor_lost_episodes"] == 4 and "transfer_us_per_step" in last
    assert (run / "ckpt" / "latest.pt").exists() and (run / "weights" / "LATEST").exists()
    state = json.loads((run / "train_state.json").read_text(encoding="utf-8"))
    assert state["version"] == 3 and state["decisions"] == last["decisions"]
    eps = list(read_jsonl(run / "episodes.jsonl"))
    assert eps and all("seed" in e and "result" in e for e in eps)

    # resume: versions and the wall-clock axis continue
    cfg2 = small_config(tmp_path, "proc", max_updates=5)
    assert Supervisor(cfg2, resume=True).run() == 0
    k2 = _kinds(run)
    assert [r["version"] for r in k2["learner"]] == [1, 2, 3, 4, 5]
    starts = [i for i, e in enumerate(k2["event"]) if e["what"] == "start"]
    assert len(starts) == 2 and k2["event"][starts[1]]["elapsed"] >= state["elapsed"] - 1
    assert k2["train"][-1]["decisions"] > state["decisions"]

    html = build_html(run)
    assert "训练看板" in html and "训练胜率 vs 决策数" in html and "目标 95%" in html
    assert "http://" not in html and "https://" not in html and "<link" not in html
    out = write_dashboard(run)
    assert out.exists() and out.stat().st_size > 10_000


def test_threads_topology(tmp_path: Path) -> None:
    cfg = small_config(tmp_path, "thr", topology="threads", max_updates=2)
    assert Supervisor(cfg).run() == 0
    k = _kinds(cfg.run_dir)
    assert [r["version"] for r in k["learner"]] == [1, 2] and k["train"][-1]["decisions"] > 0


def test_dashboard_escapes_untrusted_names(tmp_path: Path) -> None:
    run = tmp_path / "r"
    m = MetricsWriter(run / "metrics.jsonl")
    bad = '<script>alert(1)</script>|A0'
    for i in range(5):
        m.emit("train", decisions=1000 * i, games=10 * i, decisions_per_s=100.0, games_per_hour=5.0, version=i,
               updates=i, stage_us={"encode": 70.0, "python": 120.0}, counts={"lost": 1},
               outcomes={bad: {"n": 10, "wins": 3, "win_rate": 0.3, "lo": 0.1, "hi": 0.6, "mean_floor": 12.0,
                               "death_floor": {"5": 3}, "death_room": {}, "top_killers": [["<b>x</b>", 2]],
                               "total_finished": 10, "total_wins": 3, "failures": {}}})
    m.close()
    html = build_html(run, tmp_path / "no-results")
    assert "<script>alert" not in html and "&lt;script&gt;" in html and "<b>x</b>" not in html
