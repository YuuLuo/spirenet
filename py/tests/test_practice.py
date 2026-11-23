"""Practice fights (spire.gym, notes/gym/practice-spec.md v1 + v1.1), tests T1-T5, T7, T8 of spec 10 (T6 runs on the
real fork host: scripts/practice_real.py). Fake fork server = scripts/fake_gym_server.py (Linux: real ``os.fork``,
AF_UNIX, fight mode: FIGHT_END + one frame)."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from spire import gym
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
from spire.cycle import CycleBan
from spire.gym import Controller, Difficulty, PracticeConfig, Priors, SeedStream, encounter_class, key_class
from spire.history import PublicHistory
from spire.host import HostConfig, HostProcess
from spire.metrics import ActionStats, EpisodeRecord, JsonlWriter, MetricsWriter, read_jsonl
from spire.model import ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.removal import load_rule
from spire.rewards import Facts, RewardConfig, RoomIds, episode_targets, frame_facts
from spire.rollout import GAEConfig, RolloutConfig, RolloutStats, RolloutStore, Segment, gae
from spire.schema import G, Kind
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import (Aggregator, Supervisor, TrainConfig, actor_layout, build, check_practice, config_dict,
                         load_config)
from spire.vocab import Vocab
from test_removal import synthetic_table
from test_train import small_config

PY = Path(__file__).resolve().parents[1]
SERVER = str(PY / "scripts" / "fake_gym_server.py")
OFF_REF = str(Path(__file__).with_name("practice_off_ref.py"))
TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
EXPLORATION = ((int(Kind.COMBAT), 0.05), (int(Kind.MAP), 0.1))
KID_BASE = 1000
SOLO_GAE = GAEConfig(gamma=1.0, lam=0.99, macro_micro=True, lam_macro=0.99, lam_micro=0.99, stitch_segments=True)
linux = pytest.mark.skipif(sys.platform != "linux", reason="practice forks (Linux)")


@pytest.fixture(scope="module")
def ratings(tmp_path_factory: pytest.TempPathFactory, vocab: Vocab) -> str:
    p = tmp_path_factory.mktemp("removal") / "removal-ratings.json"
    p.write_text(json.dumps(synthetic_table(vocab)), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def tok(vocab: Vocab, ratings: str) -> Tokenizer:
    return Tokenizer(vocab, TokenizerConfig(public_history=True, cycle_ban=True, commit_select=True,
                                            removal_rule=True, removal_ratings=ratings))


# =========================================================================== T1: a child's numbers = the full run's


def _game(vocab: Vocab, seed: int) -> tuple[list[Facts], Facts]:
    """Facts of the decisions of one fake game (random legal actions) and its TERMINAL facts."""
    host = FakeHost(vocab, seed=seed, config=FakeHostConfig(mean_decisions=500, min_decisions=300, hashes=False,
                                                            abort_prob=0.0))
    rng = np.random.default_rng(seed)
    f = host.reset(seed + 1, (2,))
    out = []
    while not f.is_terminal:
        out.append(frame_facts(f))
        f = host.step(int(rng.integers(f.n_actions)))
    return out, frame_facts(f)


def fights(facts: list[Facts]) -> list[tuple[int, int]]:
    """(a, b): first and last decision of every fight (spec 1)."""
    out, a = [], None
    for t, fa in enumerate(facts):
        inside = a is not None and fa.in_combat and fa.total_floor == facts[a].total_floor
        if a is not None and not inside:
            out.append((a, t - 1))
            a = None
        if fa.in_combat and a is None:
            a = t
    if a is not None:
        out.append((a, len(facts) - 1))
    return out


def _flat(segs: list[Segment], cfg: GAEConfig) -> dict[str, np.ndarray]:
    segs = sorted(segs, key=lambda s: s.start)
    adv, ret = gae(segs, cfg)
    cat = lambda k: np.concatenate([getattr(s, k) for s in segs])  # noqa: E731
    return {"start": np.asarray([segs[0].start]), "rewards": cat("rewards"), "combat_id": cat("combat_id"),
            "adv": adv, "ret": ret, "hp_post": cat("hp_post"), "hp_mask": cat("hp_mask"), "win_mask": cat("win_mask"),
            "floors_left": cat("floors_left"), "actions": cat("actions")}


def test_fight_child_numbers_equal_the_full_run(vocab: Vocab) -> None:
    """T1 (+ T8's scout part): for every fight a..b of a recorded game, a child forked at a that takes the same
    decisions and ends at b + 1 (truncated with V(b + 1)) or at the TERMINAL gets the full run's rewards, combat ids,
    advantages, value targets and HP targets bit for bit; its win / floors targets are masked; store.stats sees only
    the full run. The same with a scout parent (facts only)."""
    rooms = RoomIds.from_vocab(vocab)
    game = None
    for seed in range(200):  # a game with an elite, a boss, a death inside a fight and a fight longer than 7
        facts, term = _game(vocab, seed)
        fs = fights(facts)
        types = {facts[a].room_type for a, _ in fs}
        if (rooms.elite in types and rooms.boss in types and term.result == 2 and fs[-1][1] == len(facts) - 1
                and max(b - a for a, b in fs) >= 8):
            game = facts, term
            break
    assert game is not None
    facts, term = game
    n = len(facts)
    rng = np.random.default_rng(7)
    values = rng.normal(size=n + 1).astype(np.float32)
    logp = -rng.random(n).astype(np.float32)
    actions = rng.integers(0, 5, n)
    checked = 0
    for seg_len in (256, 7):
        for scout_parent in (False, True):
            segs: list[Segment] = []
            store = RolloutStore(segs.append, rooms, config=RolloutConfig(segment_len=seg_len),
                                 rewards=RewardConfig(gamma=1.0))
            store.begin_episode(0, (2,), 10, 5)
            for t in range(n):
                store.add_step(0, facts[t], 100 + t, int(actions[t]), float(logp[t]), float(values[t]), 1)
            store.end_episode(0, term)
            full = _flat([s for s in segs if s.meta.env_id == 0], SOLO_GAE)
            stats0 = store.stats.as_dict()
            for k, (a, b) in enumerate(fights(facts)):
                segs.clear()
                store.begin_episode(1, (2,), 10, 5, scout=scout_parent)
                for t in range(a):
                    if scout_parent:
                        store.add_scout_step(1, facts[t])
                    else:
                        store.add_step(1, facts[t], 100 + t, int(actions[t]), float(logp[t]), float(values[t]), 1)
                store.fork_episode(store.fork_point(1), 2 + k, game=False)
                store.abort_episode(1, "shutdown")
                for t in range(a, b + 1):
                    store.add_step(2 + k, facts[t], 100 + t, int(actions[t]), float(logp[t]), float(values[t]), 1)
                if b + 1 < n:
                    store.truncate_episode(2 + k, facts[b + 1], float(values[b + 1]), 100 + b + 1)
                else:
                    store.end_episode(2 + k, term)
                kid = _flat([s for s in segs if s.meta.env_id == 2 + k], SOLO_GAE)
                assert kid["start"][0] == a
                for key in ("rewards", "combat_id", "adv", "ret", "hp_post", "hp_mask", "actions"):
                    assert np.array_equal(kid[key], full[key][a:b + 1]), (key, a, b, seg_len)
                assert not kid["win_mask"].any() and not kid["floors_left"].any()
                assert facts[a].in_combat and (kid["combat_id"] >= 0).all()
                assert all(s.end != 0 or s.bootstrap_sample is not None for s in segs)
                checked += 1
            fs = fights(facts)
            prefix = sum(a for a, _ in fs)
            if not scout_parent:  # only the prefix episodes (opened, then aborted) moved store.stats
                assert store.stats.as_dict() == {**stats0, "episodes": stats0["episodes"] + len(fs),
                                                 "shutdown": len(fs), "steps_added": stats0["steps_added"] + prefix,
                                                 "steps_dropped": stats0["steps_dropped"] + prefix}
            else:
                assert store.stats.as_dict() == stats0 and store.scout_stats.shutdown == len(fs)
                assert store.scout_stats.steps_added == prefix
            gs = store.gym_stats
            assert gs.episodes == len(fs) and gs.truncated + gs.losses == len(fs)
            assert gs.steps_added == gs.steps_released == sum(b - a + 1 for a, b in fs)
    assert checked >= 4 * 8
    # start-up: practice needs the per-fight GAE
    base = {"run_name": "x", "practice": {"actors": 1}, "actor": {"actors": 2}}
    with pytest.raises(AssertionError, match="macro_micro"):
        check_practice(build(TrainConfig, base))
    check_practice(build(TrainConfig, {**base, "ppo": {"gae": {"macro_micro": True}}}))


# =========================================================================== T5: table, copies, gain, trackers


def test_copies_and_state_copies(vocab: Vocab, tok: Tokenizer, ratings: str) -> None:
    """T5 (unit part): the table update and the shrinkage against hand values, mean k = x, start = priors / g = 1,
    the controller, the tracker copies with the removal rule on."""
    cfg = PracticeConfig(actors=1)
    t = Difficulty(cfg)
    cls = encounter_class(2, "Elite", False)
    assert cls == "2|Elite" and encounter_class(3, "Boss", True) == "3|Boss|2" and key_class("3|Boss|A+B|2") == "3|Boss|2"
    assert t.s("2|Elite|X", cls, 10) == 0.35 and t.s("1|Monster|Y", "1|Monster", 0) == 0.10
    assert t.s("3|Boss|Z|2", "3|Boss|2", 10) == 0.90 and t.s("1|Boss|Z", "1|Boss", 10) == 0.55
    assert t.s("1|Event|Z", "1|Event", 10) == 0.10
    t.update("2|Elite|X", cls, 10, True, 0.5)
    t.update("2|Elite|X", cls, 10, False, 0.3)
    n, d, h = t.table["2|Elite|X", 10]
    assert (n, d, h) == (2.0, 0.5, 0.4)
    assert t.s("2|Elite|X", cls, 10) == (2 * (0.5 + 0.5 * 0.4) + 20 * 0.35) / 22
    for i in range(200):  # after 100 fights an exponential mean with weight 1/100
        t.update("1|Monster|Y", "1|Monster", 0, i % 4 == 0, 0.1)
    assert t.table["1|Monster|Y", 0][0] == 200
    e = [0.0, 0.0, 0.0]
    for i in range(200):
        e[0] += 1
        w = 1 / min(e[0], 100)
        e[1] += w * (float(i % 4 == 0) - e[1])
        e[2] += w * (0.1 - e[2])
    assert t.table["1|Monster|Y", 0] == e
    assert t.x(0.35, 1.0) == pytest.approx(3.0, abs=1e-12) and t.x(0.03, 2.0) == 0.0
    rng = np.random.default_rng(0)
    for x in (0.4, 1.5, 3.0, 5.0, 7.7):
        ks = [t.copies(x, float(u)) for u in rng.random(100_000)]
        assert abs(np.mean(ks) - x) < 5 * 0.5 / math.sqrt(100_000), x  # 5 standard errors of a Bernoulli
        assert set(ks) <= {math.floor(x), math.floor(x) + 1}
    assert t.copies(12.3, 0.0) == 8
    # controller: start g = 1; a synthetic actor whose child decisions/s = 30 x g, parent decisions/s = 50
    ctl = Controller(0.5, 0.25, 8.0, 0.0)
    assert ctl.g == 1.0
    child = total = 0
    gs = []
    for sec in range(1, 3 * 3600):
        c = 30.0 * ctl.g
        child += int(c)
        total += int(c) + 50
        ctl.tick(float(sec), child, total)
        gs.append(ctl.g)
    assert gs[298] == 1.0 and gs[300] != 1.0 and all(0.25 <= g <= 8.0 for g in gs)
    assert ctl.q is not None and abs(ctl.q - 0.5) < 0.03
    # tracker copies at 6 cut points with the removal rule on
    rule = load_rule(vocab, ratings)
    host = FakeHost(vocab, seed=3, config=FakeHostConfig(mean_decisions=300, min_decisions=300, hashes=False,
                                                         abort_prob=0.0))
    rng = np.random.default_rng(1)
    traj = []
    f = host.reset(11, (2,))
    while len(traj) < 1500:
        if f.is_terminal:
            f = host.reset(None, (2,))
            continue
        traj.append((f, int(rng.integers(f.n_actions))))
        f = host.step(traj[-1][1])
    rule_steps = 0
    for cut in (0, 1, 157, 650, 1033, 1499):
        cyc, hist = CycleBan(vocab, commit_select=True, removal=rule), PublicHistory(tok.width)
        copies: tuple[CycleBan, PublicHistory] | None = None
        out: list[list[Any]] = [[], []]
        for t_, (f, a) in enumerate(traj):
            if t_ == cut:
                copies = (cyc.fork(), hist.fork())
            pairs = [(cyc, hist)] + ([copies] if copies is not None else [])
            for k, (c, h) in enumerate(pairs):
                s = tok.encode(f)
                hv = h.observe(f)
                st = c.observe(f, s)
                bans = c.active_bans
                b = a if a not in bans else next(i for i in range(f.n_actions) if i not in bans)
                h.record(f, s, b)
                c.record(f, b)
                if t_ >= cut:
                    out[k].append((hv.tobytes(), st, bans, s.history.tobytes(), c.rule_row, c.rule_class))
                    rule_steps += k == 0 and st.removal_rule
        assert out[0] == out[1] and len(out[0]) == 1500 - cut
    assert rule_steps > 0


# =========================================================================== the actor with a fake fork server


class _Store(RolloutStore):
    """Logs which episode (uid) every recorded step belongs to and the TERMINAL / truncation facts."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.uids: list[int] = []
        self.ends: dict[int, Facts] = {}

    def add_step(self, env_id: int, *a: Any, **k: Any) -> bool:
        self.uids.append(self._tracks[env_id].meta.episode_uid)
        return super().add_step(env_id, *a, **k)

    def add_scout_step(self, env_id: int, facts: Facts) -> bool:
        self.uids.append(self._tracks[env_id].meta.episode_uid)
        return super().add_scout_step(env_id, facts)

    def end_episode(self, env_id: int, terminal: Facts, **k: Any) -> Any:
        self.ends[self._tracks[env_id].meta.episode_uid] = terminal
        return super().end_episode(env_id, terminal, **k)

    def truncate_episode(self, env_id: int, next_facts: Facts, *a: Any, **k: Any) -> None:
        self.ends[self._tracks[env_id].meta.episode_uid] = next_facts
        super().truncate_episode(env_id, next_facts, *a, **k)


class _Policy(TorchPolicy):
    """Logs per finished group: greedy rows, whether an exploration mixture was applied, scores, actions."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.log: list[tuple[Any, bool, np.ndarray, np.ndarray]] = []

    def finish(self, handle: Any) -> Any:
        out = super().finish(handle)
        self.log.append((handle.greedy_rows, handle.exploration is not None, out.logits, out.actions))
        return out


class Run:
    def __init__(self, actor: Actor, store: _Store, segs: list[Segment], recs: list[EpisodeRecord], errs: list[dict],
                 reps: list[dict], decisions: list[tuple[int, bytes, int]], policy: _Policy, extra: dict) -> None:
        self.actor, self.store, self.segs, self.recs, self.errs, self.reps = actor, store, segs, recs, errs, reps
        self.policy, self.extra = policy, extra
        assert len(store.uids) == len(decisions)
        self.decisions = [(e, u, p, a) for u, (e, p, a) in zip(store.uids, decisions)]

    def parent(self) -> list[tuple[bytes, int]]:
        return [(p, a) for e, _, p, a in self.decisions if e < KID_BASE]

    def steps(self, uid: int) -> list[tuple[bytes, int]]:
        return [(p, a) for _, u, p, a in self.decisions if u == uid]

    @property
    def kids(self) -> list[EpisodeRecord]:
        return [r for r in self.recs if r.gym]

    @property
    def kid_uids(self) -> list[int]:
        return sorted({u for e, u, _, _ in self.decisions if e >= KID_BASE})

    def total(self, key: str) -> int:
        return sum(r["practice"][key] for r in self.reps)


def gym_host(tmp_path: Path, i: int, *args: str, step_timeout: float = 10.0) -> HostProcess:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return HostProcess(HostConfig(command=(sys.executable, "-B", SERVER, "--seed", str(i), "--mean-decisions", "80",
                                           "--marker", tmp_path.name, *args),
                                  start_timeout=30.0, reset_timeout=10.0, step_timeout=step_timeout,
                                  log_path=str(tmp_path / f"h{i}.log"), name=f"g{i}"))


def run_actor(vocab: Vocab, tok: Tokenizer, tmp_path: Path, *, practice: PracticeConfig | None = None,
              episodes: int = 6, n_hosts: int = 1, args: tuple[str, ...] = (), mode: int = gym.MODE_RESAMPLE,
              children: int = 8, max_seconds: float = 300.0, on_loop: Any = None) -> Run:
    """Train-mode actor over ``n_hosts`` fake fork servers until ``episodes`` parent games per host ended."""
    torch.manual_seed(0)
    policy = _Policy(SpireNet(tok.layout, TINY), tok, mode="sample", use_graphs=False, seed=1,
                     exploration_by_kind=EXPLORATION)
    segs: list[Segment] = []
    store = _Store(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16))
    recs: list[EpisodeRecord] = []
    errs: list[dict] = []
    reps: list[dict] = []
    decisions: list[tuple[int, bytes, int]] = []
    extra: dict[str, Any] = {"max_alive": 0, "max_per_host": 0}
    hosts = [gym_host(tmp_path, i, *args) for i in range(n_hosts)]
    kw: dict[str, Any] = {}
    if practice is not None:
        practice = replace(practice, socket_dir=str(tmp_path / "s"))
        kw = {"practice": practice, "practice_q": 0.5, "practice_children": children, "gym_seed": 5}
    try:
        actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(ascensions=((0, 1.0),)), vocab, 0),
                      ActorConfig(max_batch=16, min_group=1, report_interval=0.2, gym_env_id_base=KID_BASE),
                      store=store, on_episode=recs.append, on_error=errs.append, on_report=reps.append,
                      on_decision=lambda e, f, a: decisions.append((e, bytes(f.payload), a)), **kw)
        if actor.gym is not None:
            actor.gym.mode = mode
        if mode == gym.MODE_EXACT:  # EXACT + greedy (test only): a child replays its parent's game
            policy.mode = "greedy"

        def stop() -> bool:
            if actor.gym is not None:
                extra["max_alive"] = max(extra["max_alive"], sum(actor.gym.alive))
                extra["max_per_host"] = max(extra["max_per_host"], max(actor.gym.alive))
            if on_loop is not None:
                on_loop(actor, extra)
            return sum(not r.gym for r in recs) >= episodes * n_hosts

        actor.run(stop, max_seconds=max_seconds)
    finally:
        for h in hosts:
            h.close()
    return Run(actor, store, segs, recs, errs, reps, decisions, policy, extra)


def _orphans(tmp_path: Path, wait: float = 5.0) -> list[str]:
    """Processes with this test's marker (fake servers, their fork children) still there after up to ``wait`` s."""
    end = time.monotonic() + wait
    while True:
        out = subprocess.run(["ps", "-eo", "pid,ppid,stat,args"], capture_output=True, text=True).stdout
        left = [ln for ln in out.splitlines() if f"--marker {tmp_path.name}" in ln and ln.split()[2][0] != "Z"]
        if not left or time.monotonic() > end:
            return left
        time.sleep(0.1)


def _running(pid: int) -> bool:
    """The process exists and is not a zombie (whoever its parent is now)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            s = fh.read().decode()
    except FileNotFoundError:
        return False
    return s[s.rindex(")") + 2] != "Z"


HARD = Priors(Monster=0.9, Elite=0.9, Boss=0.9, Boss2=0.9)  # x = 8.5: k = 8 at every fight before trimming


@linux
def test_parent_trajectory_and_children(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    """T5 (actor part) + T2 (fake): the parent games see the same frames and sample the same actions with and without
    practice; every child acts only inside its forked fight, ends by FIGHT_END + one frame (TERMINAL or the
    post-fight OBS outside the fight), is truncated there with that frame's value, and its slot and process are
    gone after it; its steps = an independent recomputation from the frames."""
    off = run_actor(vocab, tok, tmp_path / "off")
    on = run_actor(vocab, tok, tmp_path / "on", practice=PracticeConfig(actors=1, priors=Priors(0.25, 0.35, 0.55, 0.9)))
    assert not off.kids and off.parent() == on.parent() and len(on.parent()) > 200
    strip = lambda rs: [{k: v for k, v in r.as_dict().items() if k != "seconds"} for r in rs if not r.gym]  # noqa: E731
    assert strip(off.recs) == strip(on.recs)
    assert on.actor.counters.episodes_ended == off.actor.counters.episodes_ended and on.store.stats == off.store.stats
    kids = on.kids
    res = Counter(r.practice["result"] for r in kids)
    assert res["fight_won"] > 10 and set(res) <= {"fight_won", "loss", "win", "cap", "shutdown"}, res
    assert on.total("forks") >= 10 and len(kids) == on.total("children") and not on.errs
    assert on.total("fork_failures") == on.total("lost") == 0
    rooms = RoomIds.from_vocab(vocab)
    parent_frames, parent_facts = {}, {}
    for p, _ in on.parent():
        f = parse_frame(p)
        parent_frames[f.episode_id, f.step_id] = p
        parent_facts.setdefault(f.episode_id, []).append(frame_facts(f))
    segs_by_uid: dict[int, list[Segment]] = {}
    for s in on.segs:
        segs_by_uid.setdefault(s.meta.episode_uid, []).append(s)
    checked = 0
    for uid in on.kid_uids:
        steps = on.steps(uid)
        first = parse_frame(steps[0][0])
        assert gym.same_obs(steps[0][0], parent_frames[first.episode_id, first.step_id])
        floor = first.glob[G.TOTAL_FLOOR]
        assert all(parse_frame(p).glob[G.IN_COMBAT] == 1 and parse_frame(p).glob[G.TOTAL_FLOOR] == floor
                   for p, _ in steps)
        if uid not in on.store.ends:
            continue
        end = on.store.ends[uid]
        assert end.terminal or not (end.in_combat and end.total_floor == floor)
        ss = sorted(segs_by_uid[uid], key=lambda s: s.start)
        assert ss[0].start == first.step_id and all(a.start + len(a) == b.start for a, b in zip(ss, ss[1:]))
        assert ss[-1].end == (1 if end.terminal else 2)
        facts = [frame_facts(parse_frame(p)) for p, _ in steps]
        ref = episode_targets([*parent_facts[first.episode_id][:first.step_id], *facts, end], rooms)
        cat = lambda k: np.concatenate([getattr(s, k) for s in ss])  # noqa: E731
        sl = slice(first.step_id, first.step_id + len(steps))
        assert np.array_equal(cat("rewards"), ref.rewards[sl]) and np.array_equal(cat("combat_id"), ref.combat_id[sl])
        assert np.array_equal(cat("actions"), np.asarray([a for _, a in steps]))
        assert not cat("win_mask").any()
        checked += 1
    assert checked >= 20
    g = on.store.gym_stats
    assert g.steps_added == sum(r.decisions for r in kids) == on.total("decisions")
    assert g.truncated == res["fight_won"] and g.losses + g.host_truncated == res["loss"] + res["cap"]
    assert on.actor.gym is not None and len(on.actor.gym.free) == 8 and sum(on.actor.gym.alive) == 0
    # a parent game that ends before its children does not end them
    end_order = {(r.env, r.seed): i for i, r in enumerate(on.recs) if not r.gym}
    late = [r for i, r in enumerate(on.recs) if r.gym and r.practice["result"] != "shutdown"
            and end_order.get((r.practice["parent_env"], r.seed), 10**9) < i]
    assert late, "no child outlived its parent's game in this run"
    assert _orphans(tmp_path) == []


@linux
def test_exact_children_replay_the_parent(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    """EXACT fork (debug server; tests only), greedy children in the parent's game: up to the first differing
    action a child's frames equal its parent's, and a fight_won child's post-fight frame is the parent's frame."""
    on = run_actor(vocab, tok, tmp_path, practice=PracticeConfig(actors=1, priors=Priors(0.2, 0.2, 0.2, 0.2)),
                   args=("--allow-debug-options",), mode=gym.MODE_EXACT, episodes=4)
    by_ep: dict[int, dict[int, bytes]] = {}
    for p, _ in on.parent():
        f = parse_frame(p)
        by_ep.setdefault(f.episode_id, {})[f.step_id] = p
    compared = post = 0
    for uid in on.kid_uids:
        steps = on.steps(uid)
        first = parse_frame(steps[0][0])
        par = by_ep[first.episode_id]
        pa = {parse_frame(p).step_id: a for p, a in on.parent() if parse_frame(p).episode_id == first.episode_id}
        for p, a in steps:
            sid = parse_frame(p).step_id
            assert gym.same_obs(p, par[sid])
            if a != pa[sid]:
                break
        else:
            end = on.store.ends.get(uid)
            nxt = first.step_id + len(steps)
            if end is not None and not end.terminal and nxt in par:
                assert end == frame_facts(parse_frame(par[nxt]))
                post += 1
        compared += 1
    assert compared >= 8 and post >= 1


@linux
def test_caps(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    """T4: 3 hosts, priors forcing k = 8, 6 child slots, per_parent 4: never more than 6 alive nor 4 per host;
    capped and trimmed counted; the child records = the sum of the trimmed k."""
    r = run_actor(vocab, tok, tmp_path, practice=PracticeConfig(actors=1, priors=HARD, per_parent=4), n_hosts=3,
                  children=6, episodes=3)
    assert r.extra["max_alive"] <= 6 and r.extra["max_per_host"] <= 4 and r.extra["max_alive"] >= 4
    assert r.total("capped") > 0 and r.total("trimmed") > 0
    assert len(r.kids) == r.total("children") and r.total("fork_failures") == 0
    assert all(rep["practice"]["alive"] <= 6 for rep in r.reps)
    assert _orphans(tmp_path) == []


@linux
def test_children_end_with_their_fight_and_die_with_their_host(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    """T2 (fake): kill -9 of a parent's host with live children: they are gone within 1 s (DIE_WITH_PARENT) and
    counted lost; the host is restarted and the run goes on."""
    seen: dict[str, Any] = {}

    def on_loop(actor: Actor, extra: dict) -> None:
        if seen or actor.gym is None or actor.gym.alive[0] < 2:
            return
        kids = [k for k in actor.gym.kids if k.kid is not None and k.kid.parent_idx == 0 and k.fork is None]
        if len(kids) < 2:
            return
        seen["pids"] = [k.host.pid for k in kids]
        seen["host"] = actor.envs[0].host.pid
        os.kill(seen["host"], signal.SIGKILL)
        t0 = time.monotonic()
        while any(_running(p) for p in seen["pids"]):
            assert time.monotonic() - t0 < 1.0, "children outlived their host by 1 s"
            time.sleep(0.01)
        seen["gone_s"] = time.monotonic() - t0

    r = run_actor(vocab, tok, tmp_path, practice=PracticeConfig(actors=1, priors=HARD), n_hosts=1, episodes=4,
                  on_loop=on_loop)
    assert seen and seen["gone_s"] < 1.0
    lost = [k for k in r.kids if k.practice["result"] == "lost"]
    assert len(lost) >= len(seen["pids"]) and r.total("lost") == len(lost)
    assert _orphans(tmp_path) == []


KILL_SCRIPT = r"""
import sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/tests")
from pathlib import Path
from spire.vocab import Vocab, make_synthetic_vocab
from spire.tokenize import Tokenizer, TokenizerConfig
import test_practice as t
vocab = Vocab(make_synthetic_vocab())
tok = Tokenizer(vocab, TokenizerConfig(public_history=True, cycle_ban=True, commit_select=True))
hosts = [t.gym_host(Path(sys.argv[2]), i) for i in range(2)]
store = t._Store(lambda s: None, t.RoomIds.from_vocab(vocab))
actor = t.Actor(hosts, t.TorchPolicy(t.SpireNet(tok.layout, t.TINY), tok, use_graphs=False), tok,
                t.EpisodeSampler(t.EpisodeDist(ascensions=((0, 1.0),)), vocab, 0),
                t.ActorConfig(gym_env_id_base=t.KID_BASE), store=store,
                practice=t.PracticeConfig(actors=1, priors=t.HARD, socket_dir=sys.argv[2] + "/s"), practice_q=0.5,
                practice_children=12, gym_seed=3)
said = []
def stop():
    if not said and sum(k.kid is not None and k.fork is None for k in actor.gym.kids) >= 4:
        said.append(1)
        print("ALIVE", flush=True)
    return False
actor.run(stop)
"""


@linux
def test_no_orphans_after_actor_kill(tmp_path: Path) -> None:
    """T2: kill -9 of the actor process with live children: no host and no child left."""
    p = subprocess.Popen([sys.executable, "-B", "-c", KILL_SCRIPT, str(PY), str(tmp_path)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert p.stdout is not None and p.stderr is not None
    line = p.stdout.readline()
    assert line.strip() == "ALIVE", (line, p.stderr.read() if p.poll() is not None else "")
    assert len(_orphans(tmp_path, wait=0.0)) >= 2 + 4
    os.kill(p.pid, signal.SIGKILL)
    p.wait(timeout=10)
    assert _orphans(tmp_path, wait=10.0) == []


@linux
def test_scouts(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    """T8 (actor part): a scout takes the arg-max on every combat step and samples without the exploration mixture
    elsewhere; no scout segment is shipped, store.stats / episodes_ended stay those of no trained game; the
    children are trained with the training behaviour."""
    r = run_actor(vocab, tok, tmp_path, practice=PracticeConfig(actors=1, scouts=True), episodes=5)
    scout_groups = [x for x in r.policy.log if x[0] is not None]
    kid_groups = [x for x in r.policy.log if x[0] is None]
    assert scout_groups and kid_groups and all(not expl for _, expl, _, _ in scout_groups)
    assert all(expl for _, expl, _, _ in kid_groups)
    greedy_rows = 0
    for rows, _, logits, actions in scout_groups:
        for i in np.flatnonzero(rows):
            assert actions[i] == int(np.argmax(np.where(np.isfinite(logits[i]), logits[i], -np.inf)))
            greedy_rows += 1
    assert greedy_rows > 100 and any((~rows).any() for rows, *_ in scout_groups)
    assert all(s.meta.env_id >= KID_BASE for s in r.segs) and r.segs
    assert r.store.stats == RolloutStats() and r.actor.counters.episodes_ended == 0
    scouts = [x for x in r.recs if x.scout]
    assert len(scouts) >= 5 and all(not x.gym for x in scouts) and r.store.scout_stats.episodes >= 5
    assert r.total("scout_decisions") == r.store.scout_stats.steps_added > 0
    assert _orphans(tmp_path) == []


# =========================================================================== T3: off = tree n


def _run_ref(tree: Path, mode: str, ratings: str) -> dict:
    env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"}
    out = subprocess.run([sys.executable, "-B", OFF_REF, mode, ratings], env=env, capture_output=True, text=True,
                         timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_practice_off_is_bit_identical_to_tree_n(ratings: str) -> None:
    """T3: SPIRE_BASE_TREE = tree n (alt/calc/py, py-stage-n). Without the section and with actors = 0: the same
    segments, records, counters and parameters after one update."""
    base = os.environ.get("SPIRE_BASE_TREE")
    if not base:
        pytest.skip("SPIRE_BASE_TREE (tree n) not set")
    ref, none, off = _run_ref(Path(base), "none", ratings), _run_ref(PY, "none", ratings), _run_ref(PY, "off", ratings)
    assert ref["tree"] != none["tree"] == off["tree"]
    for k in ("segments", "steps", "records", "seg_hash", "rec_hash", "counters", "rollout", "metrics", "params",
              "removal_steps"):
        assert ref[k] == none[k] == off[k], k
    assert ref["steps"] >= 1000 and ref["counters"]["errors_recorded"] > 0 and ref["rollout"]["aborted"] > 0
    assert ref["removal_steps"] > 0


def test_plain_slots_of_a_practice_run(tmp_path: Path) -> None:
    """T3 second part: in a run with practice.actors = 1 of 2, slot 0's host section, host count, env ids, seeds and
    practice state (none) equal those of the practice-free run; slot 1 is the practice actor; config.json has no
    practice key when off."""
    free = small_config(tmp_path, "free", actor={"actors": 2, "device": "cpu"}, hosts={"envs_per_actor": 60})
    prac = small_config(tmp_path, "prac", actor={"actors": 2, "device": "cpu"}, hosts={"envs_per_actor": 60},
                        ppo={"gae": {"macro_micro": True}}, practice={"actors": 1})
    a, b = actor_layout(free, 0), actor_layout(prac, 0)
    assert a == b and a.practice is None and a.n_hosts == 60 and a.hosts == free.hosts
    c = actor_layout(prac, 1)
    assert c.practice == prac.practice and c.q_a == 0.5 and c.n_hosts == 30 and c.children == 48
    assert "practice" not in config_dict(free) and config_dict(prac)["practice"]["actors"] == 1
    assert build(TrainConfig, config_dict(prac)).practice == prac.practice
    scout = small_config(tmp_path, "sc", actor={"actors": 8, "device": "cpu"}, hosts={"envs_per_actor": 60},
                         ppo={"gae": {"macro_micro": True}}, practice={"actors": 3, "scouts": True, "q_a": 0.5})
    lay = [actor_layout(scout, s) for s in range(8)]
    assert [x.practice is not None for x in lay] == [False] * 5 + [True] * 3
    assert {x.n_hosts for x in lay[5:]} == {30} and {x.children for x in lay[5:]} == {48}
    p = tmp_path / "c.toml"
    p.write_text('run_name = "x"\n[actor]\nactors = 4\n[ppo.gae]\nmacro_micro = true\n[practice]\nactors = 2\n'
                 'priors = { Monster = 0.10, Elite = 0.35, Boss = 0.55, Boss2 = 0.90 }\n', encoding="utf-8")
    assert load_config(p).practice == PracticeConfig(actors=2)


# =========================================================================== T7: rows


def test_practice_rows(tmp_path: Path) -> None:
    """T7 (unit): children only in practice_episodes.jsonl and the practice totals, scouts only in
    scout_episodes.jsonl; games / episodes_ended / outcomes / taste stay those of the trained games; class and top
    fields filled."""
    cfg = build(TrainConfig, {"run_name": "m", "run_root": str(tmp_path), "actor": {"actors": 2},
                              "ppo": {"gae": {"macro_micro": True}}, "practice": {"actors": 1, "scouts": True}})
    cfg.run_dir.mkdir(parents=True)
    agg = Aggregator(cfg, MetricsWriter(cfg.run_dir / "metrics.jsonl"), JsonlWriter(cfg.run_dir / "episodes.jsonl"),
                     JsonlWriter(cfg.run_dir / "errors.jsonl"), {}, JsonlWriter(cfg.run_dir / "practice_episodes.jsonl"),
                     JsonlWriter(cfg.run_dir / "scout_episodes.jsonl"))
    base = EpisodeRecord(lineup="IRONCLAD", ascension=10, result="win", floor=50, act=2, room="Boss",
                         in_combat=False, enemies=(), decisions=2000, auto_steps=0, seed="", seed_id="", env=0,
                         version_first=0, version_last=0, seconds=1.0, detail="")
    agg.handle("episode", 0, base.as_dict())
    agg.handle("episode", 0, replace(base, result="loss").as_dict())
    agg.handle("episode", 1, replace(base, result="win", scout=True).as_dict())
    for res, dec in (("fight_won", 30), ("loss", 20), ("lost", 0)):
        agg.handle("episode", 1, replace(base, result=res, gym="2|Elite|X", decisions=dec,
                                         practice={"class": "2|Elite", "result": res}).as_dict())
    cls = {k: 0 for k in gym.CLASS_FIELDS}
    cls.update(forks=1, children=3, finished=2, fight_wins=1, deaths=1, decisions=50, hp_lost=0.6,
               parent_fights=4, parent_deaths=1, parent_hp_lost=1.2, forked_w=3, forked_deaths_w=1, forked_hp_lost_w=0.9)
    common = {"seconds": 10.0, "stage_ns": {"wait": 0},
              "counters": {"card_offer_reopens": 0, "cycle_ban_decisions": 0, "cycle_ban_all_banned": 0,
                           "cycle_ban_capacity_hits": 0, "combat_steps": 0, "combat_nonargmax": 0, "noncombat_steps": 0,
                           "noncombat_nonargmax": 0},
              "host": {}, "actions": ActionStats().take(), "actor_latency_ms": [None, None],
              "host_latency_ms": [None, None], "infer_ms": [None, None], "batch_mean": None, "batch_max": None,
              "version": 0, "envs_alive": 4, "envs": 4, "throttled_s": 0.0}
    agg.handle("actor", 0, {**common, "decisions": 3000, "rollout": {"wins": 1, "losses": 1}})
    agg.handle("actor", 1, {**common, "decisions": 2000, "rollout": {}, "practice": {
        "forks": 1, "children": 3, "fork_failures": 0, "capped": 2, "trimmed": 1, "started": 3, "lost": 1,
        "decisions": 1000, "scout_decisions": 1000, "alive": 2, "g": 1.25, "q": 0.5,
        "rollout": {"steps_released": 50}, "scout_rollout": {"steps_added": 1000}, "host_us": 3000,
        "fork_ms": [30.0, 40.0], "arrive_ms": [35.0, 45.0], "classes": {"2|Elite": cls}, "classes_asc": {},
        "top": [{"key": "2|Elite|X", "ascension": 10, "n": 4, "d": 0.25, "h": 0.3, "s": 0.4, "children": 3,
                 "fight_wins": 1, "deaths": 1, "hp_lost": 0.6}],
        "table": [{"key": "2|Elite|X", "ascension": 10, "class": "2|Elite", "n": 4, "d": 0.25, "h": 0.3, "s": 0.4,
                   "kids": [3, 1, 1, 0.6]}]}})
    agg.emit_train(10.0)
    for w in (agg.metrics, agg.episodes, agg.errors, agg.practice_episodes, agg.scout_episodes):
        assert w is not None
        w.close()
    row = [r for r in read_jsonl(cfg.run_dir / "metrics.jsonl") if r["kind"] == "train"][-1]
    assert row["games"] == 2 and row["episodes_ended"] == 2 and list(row["outcomes"]) == ["IRONCLAD|A10"]
    assert (row["outcomes"]["IRONCLAD|A10"]["n"], row["outcomes"]["IRONCLAD|A10"]["wins"]) == (2, 1)
    assert row["practice_decision_share"] == 0.25 and row["practice_forks_per_hour"] == 360
    assert (row["practice_capped"], row["practice_trimmed"], row["practice_lost"]) == (2, 1, 1)
    pr = row["practice"]
    assert pr["g"] == {"1": 1.25} and pr["alive"] == 2 and pr["k_mean"] == 3.0
    c = pr["classes"]["2|Elite"]
    assert c["death_rate"] == 0.5 and c["forked_death_rate"] == 1 / 3 and c["parent_death_rate"] == 0.25
    assert pr["top"][0]["key"] == "2|Elite|X" and pr["top"][0]["d"] == 0.25
    assert pr["results"] == {"fight_won": 1, "loss": 1, "lost": 1} and pr["scout_results"] == {"win": 1}
    assert len(list(read_jsonl(cfg.run_dir / "episodes.jsonl"))) == 2
    assert [r["practice"]["result"] for r in read_jsonl(cfg.run_dir / "practice_episodes.jsonl")] == \
        ["fight_won", "loss", "lost"]
    assert [r["scout"] for r in read_jsonl(cfg.run_dir / "scout_episodes.jsonl")] == [True]
    assert json.loads((cfg.run_dir / "practice_table.json").read_text())["1"][0]["kids"] == [3, 1, 1, 0.6]
    assert "gym" not in base.as_dict() and "scout" not in base.as_dict() and "practice" not in base.as_dict()


@linux
@pytest.mark.parametrize("scouts", [False, True])
def test_training_run_with_practice(tmp_path: Path, scouts: bool) -> None:
    """T7: Supervisor + learner + 2 actors (slot 1 practice) on fake fork hosts, 3 updates: children's samples are
    trained on, their records go to practice_episodes.jsonl only; games = slot 0's (and, without scouts, slot 1's
    parents') games; the train rows carry the practice fields; no process is left."""
    cfg = small_config(tmp_path, f"pr-{scouts}", max_updates=3,
                       actor={"actors": 2, "device": "cpu", "report_interval": 0.5, "max_batch": 8},
                       hosts={"command": [sys.executable, "-B", SERVER, "--mean-decisions", "60", "--marker",
                                          tmp_path.name], "envs_per_actor": 4},
                       episodes={"lineups": [["IRONCLAD", 1.0]], "ascensions": [[0, 1.0]]},
                       ppo={"minibatch": 256, "micro_batch": 256, "epochs": 1, "lr": 1e-3,
                            "gae": {"macro_micro": True, "stitch_segments": True}},
                       tokenizer={"cycle_ban": True, "commit_select": True},
                       practice={"actors": 1, "scouts": scouts, "socket_dir": str(tmp_path / "s"),
                                 "priors": {"Monster": 0.4, "Elite": 0.5, "Boss": 0.6, "Boss2": 0.9}})
    assert Supervisor(cfg).run() == 0
    rows = [r for r in read_jsonl(cfg.run_dir / "metrics.jsonl") if r["kind"] == "train"]
    learner = [r for r in read_jsonl(cfg.run_dir / "metrics.jsonl") if r["kind"] == "learner"]
    assert [r["version"] for r in learner] == [1, 2, 3]
    last = rows[-1]
    assert last["practice"]["totals"]["children"] > 0 and any(r["practice_decision_share"] for r in rows)
    assert last["practice"]["totals"]["fork_failures"] == 0 and last["practice"]["classes"]
    kids = list(read_jsonl(cfg.run_dir / "practice_episodes.jsonl"))
    full = list(read_jsonl(cfg.run_dir / "episodes.jsonl"))
    assert kids and all(r["gym"] and r["actor"] == 1 for r in kids) and all("gym" not in r for r in full)
    assert last["games"] == sum(r["result"] in ("win", "loss") for r in full)
    if scouts:
        sc = list(read_jsonl(cfg.run_dir / "scout_episodes.jsonl"))
        assert sc and all(r["actor"] == 1 for r in sc) and all(r["actor"] == 0 for r in full)
    else:
        assert {r["actor"] for r in full} == {0, 1}
    assert json.loads((cfg.run_dir / "config.json").read_text(encoding="utf-8"))["practice"]["actors"] == 1
    assert (cfg.run_dir / "practice_table.json").exists()
    assert _orphans(tmp_path) == []
