"""Review fixes on tree p2 (notes/review/v3-python.md R07, R09b, R10e; notes/next/p2-fixes.md): the censoring counters
of the forced release, the train row's decisions by origin, practice.k_max = 0 (the cost run) and the phase-0 +
recompute_values assert. The practice kit's placement (R09a) is tested by the kit (alt/p2/kit/test_practice_cfg.sh)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import spire.actor
from ppo_stubs import room_ids
from spire.gym import PracticeConfig
from spire.kickstart import KickstartConfig
from spire.learner import PPOConfig
from spire.metrics import ActionStats, JsonlWriter, MetricsWriter, read_jsonl
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import Result
from spire.train import Aggregator, Supervisor, TrainConfig, build
from spire.vocab import Vocab
from test_practice import HARD, KID_BASE, SERVER, _orphans, linux, ratings, run_actor, tok  # noqa: F401
from test_rollout import episode_facts, feed
from test_train import small_config

SEG, LIMIT = 4, 8  # at LIMIT + 1 steps two sealed segments (8 steps) are force-released


# =========================================================================== R07: censoring counters


@pytest.mark.parametrize("result", [Result.WIN, Result.LOSS])
@pytest.mark.parametrize("n", [LIMIT, LIMIT + 1, 3 * LIMIT + 3])
def test_censoring_counters(vocab: Vocab, result: int, n: int) -> None:
    """A won / lost episode just below the pending limit (nothing censored), just above it (the 8 force-released
    steps) and well above it: censored_*_steps = the steps shipped with win_mask 0, *_steps = the episode's steps;
    the other outcome's counters stay 0."""
    rng = np.random.default_rng(n)
    out: list[Segment] = []
    store = RolloutStore(out.append, RoomIds.from_vocab(vocab),
                         config=RolloutConfig(segment_len=SEG, max_pending_steps=LIMIT))
    facts = episode_facts(room_ids(vocab), n, result, rng)
    feed(store, 0, facts, np.zeros(n))
    store.end_episode(0, facts[-1])
    expect = {LIMIT: 0, LIMIT + 1: 8, 3 * LIMIT + 3: 24}[n]
    masked = sum(int((~s.win_mask).sum()) for s in out)
    assert sum(len(s) for s in out) == n and masked == expect
    assert (store.stats.forced_releases > 0) == (expect > 0)
    win, loss = (expect, n), (0, 0)
    if result == Result.LOSS:
        win, loss = loss, win
    st = store.stats
    assert (st.censored_win_steps, st.win_steps, st.censored_loss_steps, st.loss_steps) == (*win, *loss)


# =========================================================================== R10e


def test_phase0_refuses_recompute_values() -> None:
    ks0 = KickstartConfig("x", phase=0)
    with pytest.raises(AssertionError, match="recompute_values = false"):
        PPOConfig(epochs=1, kickstart=ks0, recompute_values=True)
    PPOConfig(epochs=1, kickstart=ks0)
    PPOConfig(kickstart=KickstartConfig("x", phase=1), recompute_values=True)
    PPOConfig(recompute_values=True)


# =========================================================================== R09b: decisions by origin


def test_decisions_by_origin(tmp_path: Path) -> None:
    """plain + child + scout = the interval's decisions (decisions_per_s keeps its meaning: all of them);
    decisions_per_s_trained = (plain + child) / span; without practice everything is plain."""
    for name, practice, child, scout in (("p", {"practice": {"actors": 1, "scouts": True}}, 1000, 700),
                                         ("n", {}, 0, 0)):
        cfg = build(TrainConfig, {"run_name": name, "run_root": str(tmp_path), "actor": {"actors": 2},
                                  "ppo": {"gae": {"macro_micro": True}}, **practice})
        cfg.run_dir.mkdir(parents=True)
        agg = Aggregator(cfg, MetricsWriter(cfg.run_dir / "metrics.jsonl"), JsonlWriter(cfg.run_dir / "e.jsonl"),
                         JsonlWriter(cfg.run_dir / "err.jsonl"), {},
                         *([JsonlWriter(cfg.run_dir / "pe.jsonl"), JsonlWriter(cfg.run_dir / "se.jsonl")] if practice
                           else []))
        common = _report()
        agg.handle("actor", 0, {**common, "decisions": 3000, "rollout": {}})
        rep1 = {**common, "decisions": 2000, "rollout": {}}
        if practice:
            rep1["practice"] = _practice_report(child, scout)
        agg.handle("actor", 1, rep1)
        agg.emit_train(10.0)
        assert agg.metrics is not None
        agg.metrics.close()
        row = [r for r in read_jsonl(cfg.run_dir / "metrics.jsonl") if r["kind"] == "train"][-1]
        assert row["decisions_by_origin"] == {"plain": 5000 - child - scout, "child": child, "scout": scout}
        assert sum(row["decisions_by_origin"].values()) == 5000 and row["decisions_per_s"] == 500.0
        assert row["decisions_per_s_trained"] == (5000 - scout) / 10.0
        if practice:
            assert row["practice_decision_share"] == child / (5000 - scout)
            assert row["practice"]["decisions"] == child and row["practice"]["scout_decisions"] == scout


def _report() -> dict:
    return {"seconds": 10.0, "stage_ns": {"wait": 0},
            "counters": {"card_offer_reopens": 0, "cycle_ban_decisions": 0, "cycle_ban_all_banned": 0,
                         "cycle_ban_capacity_hits": 0, "combat_steps": 0, "combat_nonargmax": 0, "noncombat_steps": 0,
                         "noncombat_nonargmax": 0},
            "host": {}, "actions": ActionStats().take(), "actor_latency_ms": [None, None],
            "host_latency_ms": [None, None], "infer_ms": [None, None], "batch_mean": None, "batch_max": None,
            "version": 0, "envs_alive": 4, "envs": 4, "throttled_s": 0.0}


def _practice_report(child: int, scout: int) -> dict:
    return {"forks": 1, "children": 3, "fork_failures": 0, "capped": 0, "trimmed": 0, "started": 3, "lost": 0,
            "decisions": child, "scout_decisions": scout, "alive": 0, "g": 1.0, "q": None, "rollout": {},
            "scout_rollout": {}, "host_us": 0, "fork_ms": [None, None], "arrive_ms": [None, None], "classes": {},
            "classes_asc": {}, "top": []}


# =========================================================================== R09b: k_max = 0 (the cost run)


def test_k_max_zero_is_allowed() -> None:
    assert PracticeConfig(actors=1, k_max=0).k_max == 0
    with pytest.raises(AssertionError):
        PracticeConfig(actors=1, k_max=-1)


@linux
def test_k_max_zero_sends_no_fork(vocab: Vocab, tok, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """With priors that give k = 8 at every fight, k_max = 0: no FORK is ever encoded, no child, nothing shipped;
    the scouts play their games."""
    sent: list[int] = []
    enc = spire.actor.encode_fork
    monkeypatch.setattr(spire.actor, "encode_fork", lambda *a: sent.append(1) or enc(*a))
    r = run_actor(vocab, tok, tmp_path, practice=PracticeConfig(actors=1, scouts=True, k_max=0, priors=HARD),
                  episodes=3)
    assert sent == [] and r.total("forks") == 0 and r.total("decisions") == 0 and not r.kids and not r.segs
    assert len([x for x in r.recs if x.scout]) >= 3 and r.total("scout_decisions") > 200
    assert r.store.gym_stats.episodes == 0
    assert _orphans(tmp_path) == []


@linux
def test_training_run_k_max_zero(tmp_path: Path) -> None:
    """Supervisor + learner + 2 actors (slot 1 practice with scouts, k_max = 0), forced releases on: the train rows
    carry the censoring counters (R07) and the decisions by origin; practice share 0, no fork, scouts played."""
    cfg = small_config(tmp_path, "k0", max_updates=3,
                       actor={"actors": 2, "device": "cpu", "report_interval": 0.5, "max_batch": 8},
                       hosts={"command": [sys.executable, "-B", SERVER, "--mean-decisions", "60", "--marker",
                                          tmp_path.name], "envs_per_actor": 4},
                       episodes={"lineups": [["IRONCLAD", 1.0]], "ascensions": [[0, 1.0]]},
                       rollout={"segment_len": 16, "max_pending_steps": 40},
                       ppo={"minibatch": 256, "micro_batch": 256, "epochs": 1, "lr": 1e-3,
                            "gae": {"macro_micro": True, "stitch_segments": True}},
                       tokenizer={"cycle_ban": True, "commit_select": True},
                       practice={"actors": 1, "scouts": True, "k_max": 0, "socket_dir": str(tmp_path / "s"),
                                 "priors": {"Monster": 0.9, "Elite": 0.9, "Boss": 0.9, "Boss2": 0.9}})
    assert Supervisor(cfg).run() == 0
    rows = [r for r in read_jsonl(cfg.run_dir / "metrics.jsonl") if r["kind"] == "train"]
    keys = ("censored_win_steps", "censored_loss_steps", "win_steps", "loss_steps")
    assert rows and all(set(keys) <= set(r["rollout"]) for r in rows if r["rollout"])
    tot = {k: sum(r["rollout"].get(k, 0) for r in rows) for k in keys}
    assert tot["win_steps"] + tot["loss_steps"] > 0 and tot["censored_win_steps"] + tot["censored_loss_steps"] > 0
    assert all(r["decisions_by_origin"]["child"] == 0 for r in rows)
    assert sum(r["decisions_by_origin"]["scout"] for r in rows) > 0
    shares = [r["practice_decision_share"] for r in rows if r["practice_decision_share"] is not None]
    assert shares and all(s == 0.0 for s in shares)
    last = rows[-1]["practice"]["totals"]
    assert last["forks"] == 0 and last["children"] == 0 and last["fork_failures"] == 0
    assert list(read_jsonl(cfg.run_dir / "scout_episodes.jsonl"))
    assert list(read_jsonl(cfg.run_dir / "practice_episodes.jsonl")) == []
    assert _orphans(tmp_path) == []

