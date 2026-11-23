"""Practice v1.2 (notes/gym/practice-spec.md "Version 1.2"): GAME children, test T9 of section E (T10 runs on the
real fork host: scripts/practice_game_real.py). Fake fork server = scripts/fake_gym_server.py (Linux: real
``os.fork``; flag GAME: the child plays to the episode's TERMINAL)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import spire.actor as actor_mod
from spire import gym
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler
from spire.gym import ASC_FIELDS, CLASS_FIELDS, GAME_CLASS_FIELDS, PracticeConfig, Priors
from spire.metrics import ActionStats, EpisodeRecord, JsonlWriter, MetricsWriter, read_jsonl
from spire.model import SpireNet
from spire.proto import parse_frame
from spire.rewards import Facts, RewardConfig, RoomIds, episode_targets, frame_facts
from spire.rollout import EpisodeMeta, RolloutConfig, RolloutStore, Segment, gae
from spire.schema import G
from spire.tokenize import Tokenizer
from spire.train import Aggregator, TrainConfig, build, check_practice, config_dict, load_config
from spire.vocab import Vocab
from test_practice import (EXPLORATION, KID_BASE, SOLO_GAE, TINY, Run, _game, _orphans, _Policy, _Store,  # noqa: F401
                           fights, gym_host, linux, ratings, tok)  # ratings, tok: the module-scoped fixtures

PY = Path(__file__).resolve().parents[1]
GAME_CLASSES = ("1|Monster", "2|Monster", "3|Monster")  # the fake games' most frequent fights
PRIORS = Priors(0.25, 0.35, 0.55, 0.9)


# =========================================================================== configuration, flags, taste assert


def test_game_classes_config_and_flags(tmp_path: Path) -> None:
    """T9: the config key, the flags by class, the start-up taste assert."""
    assert PracticeConfig().game_classes == ()
    cfg = PracticeConfig(actors=1, game_classes=("3|Boss",))
    assert cfg.fork_flags("3|Boss") == gym.FLAG_GAME | gym.FLAG_DIE_WITH_PARENT == gym.GAME_FLAGS == 3
    assert cfg.fork_flags("3|Boss|2") == cfg.fork_flags("2|Boss") == cfg.fork_flags("3|Elite") == 2
    assert PracticeConfig(actors=1).fork_flags("3|Boss") == gym.PRACTICE_FLAGS == 2
    PracticeConfig(game_classes=("3|Boss|2", "1|Monster"))
    for bad in ("3", "Boss", "3|Boss|1", "x|Boss", "0|Boss", "3|", "3|Boss|2|x"):
        with pytest.raises(AssertionError, match="not a class"):
            PracticeConfig(game_classes=(bad,))
    with pytest.raises(AssertionError):
        PracticeConfig(game_classes=("3|Boss", "3|Boss"))
    p = tmp_path / "c.toml"
    p.write_text('run_name = "x"\n[actor]\nactors = 4\n[ppo.gae]\nmacro_micro = true\n[practice]\nactors = 2\n'
                 'game_classes = ["3|Boss"]\n', encoding="utf-8")
    c = load_config(p)
    assert c.practice.game_classes == ("3|Boss",)
    assert config_dict(c)["practice"]["game_classes"] == ["3|Boss"]
    assert build(TrainConfig, config_dict(c)).practice == c.practice
    d = config_dict(c)  # the config.json of a v1.1 run (no game_classes key) is version 1.1
    del d["practice"]["game_classes"]
    assert build(TrainConfig, d).practice.game_classes == ()
    # taste: refused with GAME children, allowed without
    base = {"run_name": "x", "actor": {"actors": 2},
            "ppo": {"gae": {"macro_micro": True}, "taste": {"p_episode": 0.1, "file": "t.json", "eps": 0.1}}}
    check_practice(build(TrainConfig, {**base, "practice": {"actors": 1}}))
    with pytest.raises(AssertionError, match="without taste"):
        check_practice(build(TrainConfig, {**base, "practice": {"actors": 1, "game_classes": ["3|Boss"]}}))


# =========================================================================== a GAME child's numbers = the full run's


def _flat(segs: list[Segment]) -> dict[str, np.ndarray]:
    segs = sorted(segs, key=lambda s: s.start)
    adv, ret = gae(segs, SOLO_GAE)
    cat = lambda k: np.concatenate([getattr(s, k) for s in segs])  # noqa: E731
    return {"start": np.asarray([segs[0].start]), "rewards": cat("rewards"), "combat_id": cat("combat_id"),
            "adv": adv, "ret": ret, "hp_post": cat("hp_post"), "hp_mask": cat("hp_mask"), "win": cat("win"),
            "win_mask": cat("win_mask"), "floors_left": cat("floors_left"), "actions": cat("actions")}


def test_game_child_numbers_equal_the_full_run(vocab: Vocab) -> None:
    """T9: a GAME child forked (from a scout's prefix) at the first decision of any fight that takes the full run's
    decisions to its TERMINAL (a death game and a won game) gets the full run's rewards, combat ids, advantages,
    returns, HP targets and win / floors labels on its steps, bit for bit, with segment lengths 256 and 7; a
    fight-only child of the same fight keeps the version-1.1 masked labels; store.stats sees only the full run."""
    rooms = RoomIds.from_vocab(vocab)
    games = {}
    for seed in range(400):
        facts, term = _game(vocab, seed)
        if term.result in (1, 2) and not term.truncated and term.result not in games and len(fights(facts)) >= 4:
            games[term.result] = facts, term
        if len(games) == 2:
            break
    assert set(games) == {1, 2}
    checked = 0
    for facts, term in games.values():
        n = len(facts)
        rng = np.random.default_rng(n)
        values = rng.normal(size=n).astype(np.float32)
        logp = -rng.random(n).astype(np.float32)
        actions = rng.integers(0, 5, n)
        fs = fights(facts)
        for seg_len in (256, 7):
            segs: list[Segment] = []
            store = RolloutStore(segs.append, rooms, config=RolloutConfig(segment_len=seg_len),
                                 rewards=RewardConfig(gamma=1.0))
            store.begin_episode(0, (2,), 10, 5)
            for t in range(n):
                store.add_step(0, facts[t], 100 + t, int(actions[t]), float(logp[t]), float(values[t]), 1)
            store.end_episode(0, term)
            full = _flat([s for s in segs if s.meta.env_id == 0])
            assert full["win_mask"].all()
            stats0 = store.stats.as_dict()
            for a, b in fs:
                for game in (True, False):
                    segs.clear()
                    store.begin_episode(1, (2,), 10, 5, scout=True)
                    for t in range(a):
                        store.add_scout_step(1, facts[t])
                    store.fork_episode(store.fork_point(1), 2, game=game)
                    store.abort_episode(1, "shutdown")
                    last = n if game else b + 1
                    for t in range(a, last):
                        store.add_step(2, facts[t], 100 + t, int(actions[t]), float(logp[t]), float(values[t]), 1)
                    if last < n:
                        store.truncate_episode(2, facts[last], 0.5, 100 + last)
                    else:
                        store.end_episode(2, term)
                    kid = _flat([s for s in segs if s.meta.env_id == 2])
                    assert kid["start"][0] == a
                    keys = ["rewards", "combat_id", "hp_post", "hp_mask", "actions"]
                    if game:
                        keys += ["adv", "ret", "win", "win_mask", "floors_left"]
                        assert kid["win_mask"].all() and len(kid["rewards"]) == n - a
                    else:
                        assert not kid["win_mask"].any() and not kid["floors_left"].any()
                    for key in keys:
                        assert np.array_equal(kid[key], full[key][a:last]), (key, a, b, seg_len, game)
                    checked += 1
            assert store.stats.as_dict() == stats0
            gs = store.gym_stats
            assert gs.episodes == 2 * len(fs) and gs.truncated == sum(b + 1 < n for _, b in fs)
            assert gs.wins + gs.losses == len(fs) + sum(b + 1 == n for _, b in fs)
    assert checked >= 2 * 2 * 2 * 4


# =========================================================================== the actor with a fake fork server


class _GameStore(_Store):
    """test_practice._Store + which child episode (uid) is a GAME child."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.game: dict[int, bool] = {}

    def fork_episode(self, point: tuple[EpisodeMeta, list[Facts]], env_id: int, *, game: bool) -> EpisodeMeta:
        meta = super().fork_episode(point, env_id, game=game)
        self.game[meta.episode_uid] = game
        return meta


def run_game_actor(vocab: Vocab, tok: Tokenizer, tmp_path: Path, practice: PracticeConfig, *, episodes: int = 6,
                   ascension: int = 10, children: int = 12) -> Run:
    """test_practice.run_actor (one fake fork server) with a chosen ascension: classes_asc rows exist at A9 / A10."""
    torch.manual_seed(0)
    policy = _Policy(SpireNet(tok.layout, TINY), tok, mode="sample", use_graphs=False, seed=1,
                     exploration_by_kind=EXPLORATION)
    segs: list[Segment] = []
    store = _GameStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16))
    recs: list[EpisodeRecord] = []
    errs: list[dict] = []
    reps: list[dict] = []
    decisions: list[tuple[int, bytes, int]] = []
    hosts = [gym_host(tmp_path, 0)]
    practice = replace(practice, socket_dir=str(tmp_path / "s"))
    try:
        actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(ascensions=((ascension, 1.0),)), vocab, 0),
                      ActorConfig(max_batch=16, min_group=1, report_interval=0.2, gym_env_id_base=KID_BASE),
                      store=store, on_episode=recs.append, on_error=errs.append, on_report=reps.append,
                      on_decision=lambda e, f, a: decisions.append((e, bytes(f.payload), a)), practice=practice,
                      practice_q=0.5, practice_children=children, gym_seed=5)
        actor.run(lambda: sum(not r.gym for r in recs) >= episodes, max_seconds=300.0)
    finally:
        for h in hosts:
            h.close()
    return Run(actor, store, segs, recs, errs, reps, decisions, policy, {})


def _sum(reps: list[dict], key: str) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for rep in reps:
        for name, c in rep["practice"][key].items():
            out.setdefault(name, Counter()).update(c)
    return out


@linux
def test_game_children(vocab: Vocab, tok: Tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T9 (fake fork server): forks at a listed class send flags 3, the others 2; a GAME child plays past its fight
    and ends at its TERMINAL without FIGHT_END; its rewards, combat ids, actions, win and floors labels = an
    independent recomputation from the frames (parent prefix + child + TERMINAL), unmasked; the children of the
    unlisted classes end as in version 1.1; the parent games are the same as without children; the new counters
    add up; no process is left."""
    sent: list[int] = []
    enc = actor_mod.encode_fork

    def encode_fork(*a: Any) -> bytes:
        sent.append(a[-1])
        return enc(*a)

    monkeypatch.setattr(actor_mod, "encode_fork", encode_fork)
    off = run_game_actor(vocab, tok, tmp_path / "off", PracticeConfig(actors=1, priors=PRIORS, k_max=0), episodes=10)
    on = run_game_actor(vocab, tok, tmp_path / "on", PracticeConfig(actors=1, priors=PRIORS,
                                                                    game_classes=GAME_CLASSES), episodes=10)
    assert not off.kids and len(on.parent()) > 200
    assert off.parent() == on.parent()  # the parents' frames and actions: the same as with no child at all
    assert not on.errs and on.total("fork_failures") == on.total("lost") == 0
    kids = on.kids
    game = [r for r in kids if r.practice["mode"] == "game"]
    fight = [r for r in kids if r.practice["mode"] == "fight"]
    assert all((r.practice["class"] in GAME_CLASSES) == (r.practice["mode"] == "game") for r in kids)
    assert len(game) >= 10 and len(fight) >= 10, (len(game), len(fight))
    # flags by class: one FORK per answered fork, flags 3 exactly for the listed classes
    classes = _sum(on.reps, "classes")
    assert Counter(sent) == {3: sum(c["forks"] for n, c in classes.items() if n in GAME_CLASSES),
                             2: sum(c["forks"] for n, c in classes.items() if n not in GAME_CLASSES)}
    res = Counter(r.practice["result"] for r in game)
    assert set(res) <= {"win", "loss", "cap", "shutdown"} and res["win"] + res["loss"] >= 5, res
    assert all(r.practice["fight_end"] is None and r.practice["rounds"] is None for r in game)
    assert set(Counter(r.practice["result"] for r in fight)) <= {"fight_won", "loss", "win", "cap", "shutdown"}
    # every child's steps, from the frames
    rooms = RoomIds.from_vocab(vocab)
    parent_frames, parent_facts = {}, {}
    for p, _ in on.parent():
        f = parse_frame(p)
        parent_frames[f.episode_id, f.step_id] = p
        parent_facts.setdefault(f.episode_id, []).append(frame_facts(f))
    segs_by_uid: dict[int, list[Segment]] = {}
    for s in on.segs:
        segs_by_uid.setdefault(s.meta.episode_uid, []).append(s)
    checked = past = fight_checked = 0
    for uid in on.kid_uids:
        steps = on.steps(uid)
        first = parse_frame(steps[0][0])
        assert gym.same_obs(steps[0][0], parent_frames[first.episode_id, first.step_id])
        floor = first.glob[G.TOTAL_FLOOR]
        inside = [parse_frame(p).glob[G.IN_COMBAT] == 1 and parse_frame(p).glob[G.TOTAL_FLOOR] == floor
                  for p, _ in steps]
        if uid not in on.store.ends:
            continue
        end = on.store.ends[uid]
        ss = sorted(segs_by_uid[uid], key=lambda s: s.start)
        assert ss[0].start == first.step_id and all(a.start + len(a) == b.start for a, b in zip(ss, ss[1:]))
        facts = [frame_facts(parse_frame(p)) for p, _ in steps]
        ref = episode_targets([*parent_facts[first.episode_id][:first.step_id], *facts, end], rooms)
        cat = lambda k: np.concatenate([getattr(s, k) for s in ss])  # noqa: E731
        sl = slice(first.step_id, first.step_id + len(steps))
        assert np.array_equal(cat("rewards"), ref.rewards[sl]) and np.array_equal(cat("combat_id"), ref.combat_id[sl])
        assert np.array_equal(cat("actions"), np.asarray([a for _, a in steps]))
        if on.store.game[uid]:
            assert end.terminal and ss[-1].end == 1 and inside[0], "a GAME child ends at its TERMINAL"
            assert inside == sorted(inside, reverse=True), "a GAME child went back into its forked fight"
            past += not all(inside)
            assert cat("win_mask").all() == (ref.win is not None) and cat("win_mask").any() == (ref.win is not None)
            if ref.win is not None:
                assert ref.floors_left is not None
                assert np.array_equal(cat("win"), np.full(len(steps), ref.win, np.float32))
                assert np.array_equal(cat("floors_left"), ref.floors_left[sl])
            checked += 1
        else:
            assert all(inside) and not cat("win_mask").any() and ss[-1].end == (1 if end.terminal else 2)
            assert end.terminal or not (end.in_combat and end.total_floor == floor)
            fight_checked += 1
    assert checked >= 5 and past >= 5 and fight_checked >= 10, (checked, past, fight_checked)
    # the new counters add up: records = class totals = classes_asc (A10)
    fin = [r for r in game if r.practice["result"] in ("win", "loss", "cap")]
    assert sum(c["game_children"] for c in classes.values()) == len(game)
    assert sum(c["game_finished"] for c in classes.values()) == len(fin)
    assert sum(c["game_wins"] for c in classes.values()) == res["win"]
    assert sum(c["game_deaths_first"] + c["game_deaths_later"] for c in classes.values()) == res["loss"] + res["cap"]
    assert sum(c["game_deaths_later"] for c in classes.values()) > 0
    assert sum(c["game_decisions"] for c in classes.values()) == sum(r.decisions for r in game)
    assert sum(c["decisions"] for c in classes.values()) == sum(r.decisions for r in kids) == on.total("decisions")
    assert sum(c["finished"] for c in classes.values()) == sum(
        r.practice["result"] in ("fight_won", "win", "loss", "cap") for r in kids)
    asc = _sum(on.reps, "classes_asc")
    assert asc and all(k.endswith("|A10") for k in asc)
    for name, c in classes.items():
        a = asc.get(f"{name}|A10", Counter())
        for k in ("parent_fights", "parent_deaths", "children", "fight_wins", "deaths"):
            assert a[k] == c[k], (name, k)
    assert sum(a["game_wins"] for a in asc.values()) == sum(r.practice["result"] == "win" for r in kids)
    assert on.store.gym_stats.steps_added == sum(r.decisions for r in kids)
    assert on.actor.gym is not None and sum(on.actor.gym.alive) == 0
    assert _orphans(tmp_path) == []


@linux
def test_game_classes_unreached(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    """T9: with a GAME class listed that no fight has, every child is a version-1.1 child (fight mode, FIGHT_END +
    frame), and the classes_asc table stays empty below A9."""
    r = run_game_actor(vocab, tok, tmp_path, PracticeConfig(actors=1, priors=PRIORS, game_classes=("9|Boss",)),
                       ascension=0, episodes=4)
    assert r.kids and all(k.practice["mode"] == "fight" for k in r.kids)
    assert all(k.practice["result"] in ("fight_won", "loss", "win", "cap", "shutdown") for k in r.kids)
    assert not any(r.store.game.values())  # type: ignore[attr-defined]
    assert all(not rep["practice"]["classes_asc"] for rep in r.reps)
    assert _orphans(tmp_path) == []


# =========================================================================== metrics


def test_game_rows(tmp_path: Path) -> None:
    """T9 (rows): the new class counters and the classes_asc table are summed into the train row; a v1.1
    train_state.json (class totals without the game counters) resumes; GAME children stay out of the games."""
    cfg = build(TrainConfig, {"run_name": "m", "run_root": str(tmp_path), "actor": {"actors": 2},
                              "ppo": {"gae": {"macro_micro": True}},
                              "practice": {"actors": 1, "scouts": True, "game_classes": ["3|Boss"]}})
    cfg.run_dir.mkdir(parents=True)
    old = {k: 1 for k in CLASS_FIELDS if k not in GAME_CLASS_FIELDS}
    totals = {"practice": {"classes": {"3|Boss": dict(old)}, "results": {}, "scout_results": {}, "forks": 1,
                           "children": 1, "fork_failures": 0, "capped": 0, "trimmed": 0, "lost": 0,
                           "decisions": 1, "scout_decisions": 1}}
    agg = Aggregator(cfg, MetricsWriter(cfg.run_dir / "metrics.jsonl"), JsonlWriter(cfg.run_dir / "episodes.jsonl"),
                     JsonlWriter(cfg.run_dir / "errors.jsonl"), totals,
                     JsonlWriter(cfg.run_dir / "practice_episodes.jsonl"),
                     JsonlWriter(cfg.run_dir / "scout_episodes.jsonl"))
    assert agg.practice["classes"]["3|Boss"] == {**old, **dict.fromkeys(GAME_CLASS_FIELDS, 0)}
    with pytest.raises(AssertionError, match="class fields"):
        Aggregator(cfg, agg.metrics, agg.episodes, agg.errors,
                   {"practice": {"classes": {"3|Boss": {"forks": 1}}}}, agg.practice_episodes, agg.scout_episodes)
    base = EpisodeRecord(lineup="SILENT", ascension=10, result="win", floor=50, act=2, room="Boss",
                         in_combat=False, enemies=(), decisions=900, auto_steps=0, seed="", seed_id="", env=0,
                         version_first=0, version_last=0, seconds=1.0, detail="")
    agg.handle("episode", 0, base.as_dict())
    for res in ("win", "loss"):
        agg.handle("episode", 1, replace(base, result=res, gym="3|Boss|A", practice={
            "class": "3|Boss", "mode": "game", "result": res}).as_dict())
    cls = dict.fromkeys(CLASS_FIELDS, 0)
    cls.update(forks=1, children=2, finished=2, fight_wins=2, decisions=1800, game_children=2, game_finished=2,
               game_wins=1, game_deaths_later=1, game_decisions=1800)
    row_asc = dict(zip(ASC_FIELDS, (3, 1, 2, 2, 1, 0)))
    common = {"seconds": 10.0, "stage_ns": {"wait": 0},
              "counters": {"card_offer_reopens": 0, "cycle_ban_decisions": 0, "cycle_ban_all_banned": 0,
                           "cycle_ban_capacity_hits": 0, "combat_steps": 0, "combat_nonargmax": 0, "noncombat_steps": 0,
                           "noncombat_nonargmax": 0},
              "host": {}, "actions": ActionStats().take(), "actor_latency_ms": [None, None],
              "host_latency_ms": [None, None], "infer_ms": [None, None], "batch_mean": None, "batch_max": None,
              "version": 0, "envs_alive": 4, "envs": 4, "throttled_s": 0.0}
    agg.handle("actor", 0, {**common, "decisions": 3000, "rollout": {"wins": 1}})
    for _ in range(2):
        agg.handle("actor", 1, {**common, "decisions": 2000, "rollout": {}, "practice": {
            "forks": 1, "children": 2, "fork_failures": 0, "capped": 0, "trimmed": 0, "started": 2, "lost": 0,
            "decisions": 1800, "scout_decisions": 200, "alive": 2, "g": 1.0, "q": 0.5, "rollout": {},
            "scout_rollout": {}, "host_us": 0, "fork_ms": [None, None], "arrive_ms": [None, None],
            "classes": {"3|Boss": cls}, "classes_asc": {"3|Boss|A10": row_asc}, "top": []}})
    agg.emit_train(10.0)
    for w in (agg.metrics, agg.episodes, agg.errors, agg.practice_episodes, agg.scout_episodes):
        assert w is not None
        w.close()
    row = [r for r in read_jsonl(cfg.run_dir / "metrics.jsonl") if r["kind"] == "train"][-1]
    assert row["games"] == 1 and row["episodes_ended"] == 1
    c = row["practice"]["classes"]["3|Boss"]
    assert c["game_children"] == 4 and c["game_wins"] == 2 and c["game_decisions"] == 3600 and c["forks"] == 3
    assert row["practice"]["classes_asc"] == {"3|Boss|A10": {k: 2 * v for k, v in row_asc.items()}}
    kids = list(read_jsonl(cfg.run_dir / "practice_episodes.jsonl"))
    assert [(r["practice"]["mode"], r["practice"]["result"]) for r in kids] == [("game", "win"), ("game", "loss")]
    assert len(list(read_jsonl(cfg.run_dir / "episodes.jsonl"))) == 1


# =========================================================================== game_classes = [] is version 1.1


@linux
def test_game_classes_empty_is_v11() -> None:
    """T9 (T3 family): SPIRE_V11_TREE = the version-1.1 tree (alt/practice12/base). tests/p2_feature_ref.py's
    practice run (fake fork servers, scouts off and on) in that tree and in this one: the deterministic part (the
    parent games: frames, actions, records) is identical. The children's batches depend on fork timing in every tree
    (notes/next/p2.md B), so their hashes are recorded, not compared."""
    base = os.environ.get("SPIRE_V11_TREE")
    if not base:
        pytest.skip("SPIRE_V11_TREE (the version-1.1 tree) not set")
    for scouts in ("0", "1"):
        out = []
        for tree in (Path(base), PY):
            env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1",
                   "FEATURE": "practice", "SCOUTS": scouts}
            r = subprocess.run([sys.executable, "-B", str(tree / "tests" / "p2_feature_ref.py")], env=env,
                               capture_output=True, text=True, timeout=900, cwd=str(tree))
            assert r.returncode == 0, r.stderr[-3000:]
            out.append(json.loads(r.stdout.strip().splitlines()[-1]))
        for k in ("parent_hash", "parent_records", "parent_decisions"):
            assert out[0][k] == out[1][k], (scouts, k)
        assert out[0]["children"] > 0 and out[1]["children"] > 0
