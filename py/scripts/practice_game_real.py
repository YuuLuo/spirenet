"""T10 of notes/gym/practice-spec.md (version 1.2): GAME children on the real fork host of the calc dist, trained
policy, SILENT, scouts (as T6, scripts/practice_real.py).

    python -B scripts/practice_game_real.py gate  --ckpt C --run-config R --host-config T --removal-ratings J --out DIR
    python -B scripts/practice_game_real.py exact ...   (EXACT + greedy, host with --allow-debug-options: tests only)
    python -B scripts/practice_game_real.py cost  ...   (the production fork rule, for --seconds; then drained)

``gate`` / ``exact``: only fights of ``--only`` (default 3|Boss) fork (a test device of this script: the difficulty of
every other class reads 0), every fork of ``--game-classes`` (default 3|Boss) has GAME children; until ``--forks``
FORKs were answered, then no new fork, and the run goes on until every child has ended. Checks (``summary.json``):
first frame == the parent's frame at the fork (trailers included); every GAME child ends by its TERMINAL (no
FIGHT_END: the pool and the actor assert it), lost = 0; how many reach the second act-3 boss (a decision of class
3|Boss|2), wins, deaths in the forked fight / the second boss / elsewhere; the independent recomputation (rewards,
combat ids, kinds, actions, and the win / floors labels of every unmasked segment == episode_targets(the parent's
frames before the fork + the child's frames + its TERMINAL), from the ``on_decision`` frames). ``exact``: a child's
frames == its parent's up to the first differing action.
``cost``: every fight forks by the measured difficulty (spec 3), 3|Boss children are GAME children, the controller
and the caps of spec 3.4 as configured: decisions and wall time per GAME child (fork to TERMINAL), the child slots
they hold on average (sampled every loop pass), capped / trimmed, the children's share of the decisions.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from practice_real import KID_BASE, alive, masked  # noqa: E402
from spire import gym  # noqa: E402
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy  # noqa: E402
from spire.evalpanel import load_policy_net  # noqa: E402
from spire.gym import Difficulty, PracticeConfig  # noqa: E402
from spire.host import HostConfig, HostProcess, real_host_spec  # noqa: E402
from spire.metrics import EpisodeRecord  # noqa: E402
from spire.proto import Frame, frame_problems  # noqa: E402
from spire.rewards import Facts, RewardConfig, RoomIds, episode_targets, frame_facts  # noqa: E402
from spire.rollout import EpisodeMeta, RolloutConfig, RolloutStore, Segment  # noqa: E402
from spire.schema import G  # noqa: E402
from spire.tokenize import Tokenizer  # noqa: E402
from spire.vocab import Catalog  # noqa: E402


class Store(RolloutStore):
    """Logs the episode uid of every recorded step (scouts' included), the parent uid and mode of every child and
    every end frame's facts."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.uids: list[int] = []
        self.parent_of: dict[int, int] = {}
        self.game: dict[int, bool] = {}
        self.ends: dict[int, Facts] = {}

    def add_step(self, env_id: int, *a: Any, **k: Any) -> bool:
        self.uids.append(self._tracks[env_id].meta.episode_uid)
        return super().add_step(env_id, *a, **k)

    def add_scout_step(self, env_id: int, facts: Facts) -> bool:
        self.uids.append(self._tracks[env_id].meta.episode_uid)
        return super().add_scout_step(env_id, facts)

    def fork_episode(self, point: tuple[EpisodeMeta, list[Facts]], env_id: int, *, game: bool) -> EpisodeMeta:
        meta = super().fork_episode(point, env_id, game=game)
        self.parent_of[meta.episode_uid] = point[0].episode_uid
        self.game[meta.episode_uid] = game
        return meta

    def end_episode(self, env_id: int, terminal: Facts, **k: Any) -> Any:
        self.ends[self._tracks[env_id].meta.episode_uid] = terminal
        return super().end_episode(env_id, terminal, **k)

    def truncate_episode(self, env_id: int, next_facts: Facts, *a: Any, **k: Any) -> None:
        self.ends[self._tracks[env_id].meta.episode_uid] = next_facts
        super().truncate_episode(env_id, next_facts, *a, **k)


class Probe:
    def __init__(self, keep: bool) -> None:
        self.keep = keep  # keep every decision's frame summary and every segment (gate / exact)
        self.n = 0
        # uid -> (masked payload, facts, step, episode, action, IN_COMBAT, TOTAL_FLOOR, ROOM_TYPE, BOSSES_WON, ACT_INDEX)
        self.frames: dict[int, list[tuple[Any, ...]]] = {}
        self.pending: list[tuple[int, Frame, int]] = []
        self.kid_checked = 0
        self.kid_problems: list[str] = []
        self.segs: dict[int, list[dict[str, Any]]] = {}
        self.reports: list[dict[str, Any]] = []
        self.forks = 0
        self.kid_pids: set[int] = set()

    def on_report(self, rep: dict[str, Any]) -> None:
        self.reports.append(rep)
        self.forks += rep["practice"]["forks"]

    def on_decision(self, env_id: int, f: Frame, a: int) -> None:
        if self.keep:
            self.pending.append((env_id, f, a))

    def flush(self, store: Store) -> None:
        if not self.keep:
            return
        uids = store.uids[self.n:self.n + len(self.pending)]
        assert len(uids) == len(self.pending)
        for (env_id, f, a), uid in zip(self.pending, uids):
            fr = self.frames.setdefault(uid, [])
            g = f.glob
            fr.append((masked(f.payload), frame_facts(f), f.step_id, f.episode_id, a, int(g[G.IN_COMBAT]),
                       int(g[G.TOTAL_FLOOR]), int(g[G.ROOM_TYPE]), int(g[G.BOSSES_WON]), int(g[G.ACT_INDEX])))
            if env_id >= KID_BASE and (len(fr) == 1 or len(fr) % 10 == 0):
                self.kid_checked += 1
                self.kid_problems += frame_problems(f)
        self.n += len(self.pending)
        self.pending = []

    def sink(self, s: Segment) -> None:
        if self.keep:
            self.segs.setdefault(s.meta.episode_uid, []).append(
                {"start": s.start, "n": len(s), "rewards": s.rewards.copy(), "actions": s.actions.copy(),
                 "kinds": s.kinds.copy(), "combat_id": s.combat_id.copy(), "end": int(s.end),
                 "win_mask": s.win_mask.copy(), "win": s.win.copy(), "floors_left": s.floors_left.copy()})


class Only(Difficulty):
    """Test device of this script: the difficulty of every class not in ``only`` reads 0 (no copies there)."""

    def __init__(self, cfg: PracticeConfig, only: tuple[str, ...]) -> None:
        super().__init__(cfg)
        self.only = only

    def s(self, key: str, cls: str, asc: int) -> float:
        return super().s(key, cls, asc) if cls in self.only else 0.0


def build(a: argparse.Namespace, probe: Probe, recs: list[EpisodeRecord], errs: list[dict[str, Any]]
          ) -> tuple[Actor, list[HostProcess], Store]:
    run = json.loads(Path(a.run_config).read_text(encoding="utf-8"))
    exact = a.mode == "exact"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    net, tok, version, extra = load_policy_net(a.ckpt, device=device, weights="online")
    tc = run["tokenizer"]
    tok = Tokenizer(tok.vocab, dataclasses.replace(tok.config, commit_select=tc["commit_select"],
                                                   removal_rule=tc["removal_rule"], removal_ratings=a.removal_ratings))
    assert tok.config.calc and tok.config.forecast, "T10 runs a calc + forecast checkpoint"
    catalog = Catalog.load(extra["catalog"], tok.vocab)
    ppo = run["ppo"]
    policy = TorchPolicy(net, tok, mode="sample", use_graphs=device == "cuda", seed=a.seed, version=version,
                         exploration_by_kind=() if exact else tuple((int(k), float(e)) for k, e in
                                                                     ppo["exploration_by_kind"]),
                         exploration_choice_uniform=not exact and ppo["exploration_choice_uniform"])
    assert not ppo["advice"] and not ppo["taste"]["p_episode"], "the S-run configs have no advice / taste"
    store = Store(probe.sink, RoomIds.from_vocab(tok.vocab), config=RolloutConfig(**run["rollout"]),
                  rewards=RewardConfig(**{**run["rewards"], "gamma": ppo["gae"]["gamma"]}))
    spec = real_host_spec(a.host_config)
    assert "--search" in spec.command
    hs = run["hosts"]
    hosts = [HostProcess(HostConfig(command=spec.command_with(*(("--allow-debug-options",) if exact else ())),
                                    env=spec.process_env(), start_timeout=hs["start_timeout"],
                                    reset_timeout=hs["reset_timeout"], step_timeout=hs["step_timeout"],
                                    log_path=f"{a.out}/host-logs/h{i:02d}.log", name=f"h{i:02d}",
                                    expected_hashes=(tok.vocab.hash, catalog.hash)))
             for i in range(a.envs)]
    ac = run["actor"]
    cfg = ActorConfig(mode="train", max_batch=ac["max_batch"], max_wait_ms=ac["max_wait_ms"], min_group=a.min_group,
                      max_inflight=ac["max_inflight"], reset_flags=ac["reset_flags"], report_interval=5.0,
                      max_episode_decisions=ac["max_episode_decisions"], pile_cap=ac["pile_cap"],
                      gym_env_id_base=KID_BASE)
    pcfg = PracticeConfig(actors=1, scouts=not exact, socket_dir=f"/tmp/spire-gym-p12-{os.getpid()}", k_max=a.k_max,
                          per_parent=a.per_parent, max_children=a.max_children, game_classes=tuple(a.game_classes))
    assert len(a.weights) == len(a.ascensions)
    dist = EpisodeDist(lineups=(("SILENT", 1.0),), ascensions=tuple(zip(a.ascensions, a.weights)))
    actor = Actor(hosts, policy, tok, EpisodeSampler(dist, tok.vocab, (a.seed, 1)), cfg, store=store,
                  on_episode=recs.append, on_error=errs.append, on_report=probe.on_report,
                  on_decision=probe.on_decision, catalog=catalog, practice=pcfg, practice_q=0.5,
                  practice_children=a.max_children, gym_seed=a.seed * 7 + 1)
    assert actor.gym is not None
    if a.mode != "cost":
        actor.gym.table = Only(pcfg, tuple(a.only))
    if exact:
        actor.gym.mode = gym.MODE_EXACT
        policy.mode = "greedy"
    return actor, hosts, store


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gate", "exact", "cost"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--host-config", required=True)
    ap.add_argument("--removal-ratings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--max-children", type=int, default=48)
    ap.add_argument("--per-parent", type=int, default=12)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--min-group", type=int, default=16)
    ap.add_argument("--forks", type=int, default=120)
    ap.add_argument("--seconds", type=float, default=900.0, help="cost: forking time before the drain")
    ap.add_argument("--ascensions", type=int, nargs="+", default=[10])
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0])
    ap.add_argument("--only", nargs="+", default=["3|Boss"])
    ap.add_argument("--game-classes", nargs="+", default=["3|Boss"])
    ap.add_argument("--max-seconds", type=float, default=5400.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu-gb", type=float, default=3.0)
    a = ap.parse_args()
    Path(a.out, "host-logs").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    probe = Probe(keep=a.mode != "cost")
    recs: list[EpisodeRecord] = []
    errs: list[dict[str, Any]] = []
    actor, hosts, store = build(a, probe, recs, errs)
    g = actor.gym
    assert g is not None
    adopt = actor._adopt

    def _adopt(env: Any, fk: Any, f: Frame) -> None:
        probe.kid_pids.add(env.host.pid)
        adopt(env, fk, f)

    actor._adopt = _adopt  # type: ignore[method-assign]
    host_pids = [h.pid for h in hosts]
    t0 = time.monotonic()
    occ = {"t": t0, "game": 0, "fight": 0, "int_game": 0.0, "int_fight": 0.0, "max_game": 0, "max_all": 0,
           "draining": False, "t_drain": 0.0, "t_start": t0}

    def stop() -> bool:
        probe.flush(store)
        now = time.monotonic()
        if not occ["draining"]:  # time-weighted child slots held (FORK to reap), up to the drain
            dt = now - occ["t"]
            occ["int_game"] += occ["game"] * dt
            occ["int_fight"] += occ["fight"] * dt
            occ["t"] = now
            held = [k.kid.game for k in g.kids if k.kid is not None]
            occ["game"], occ["fight"] = sum(held), len(held) - sum(held)
            occ["max_game"] = max(occ["max_game"], occ["game"])
            occ["max_all"] = max(occ["max_all"], len(held))
            done = now - t0 >= a.seconds if a.mode == "cost" else probe.forks + g.counters.forks >= a.forks
            if done:
                occ["draining"], occ["t_drain"] = True, now
                g.cfg = dataclasses.replace(g.cfg, ascensions=(-1,))  # no new fork from here on
        return occ["draining"] and all(k.kid is None for k in g.kids)

    actor.run(stop, max_seconds=a.max_seconds)
    probe.flush(store)
    seconds = time.monotonic() - t0
    for h in hosts:
        h.close()
    time.sleep(3.0)
    left = [p for p in [*host_pids, *probe.kid_pids] if alive(p)]
    out = analyse(a, actor, store, probe, recs, errs, seconds, left, occ)
    Path(a.out, "summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    with open(Path(a.out, "kids.jsonl"), "w", encoding="utf-8") as fh:
        for r in recs:
            if r.gym:
                fh.write(json.dumps({"decisions": r.decisions, "seconds": r.seconds, "ascension": r.ascension,
                                     "result": r.result, "floor": r.floor, "bosses_won": r.bosses_won,
                                     **r.practice}) + "\n")
    print(json.dumps(out, default=str))
    return 0


def _cat(segs: list[dict[str, Any]], k: str) -> np.ndarray:
    return np.concatenate([s[k] for s in sorted(segs, key=lambda s: s["start"])])


def _q(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    v = np.asarray(xs, float)
    return {"n": len(xs), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def analyse(a: argparse.Namespace, actor: Actor, store: Store, probe: Probe, recs: list[EpisodeRecord],
            errs: list[dict[str, Any]], seconds: float, left: list[int], occ: dict[str, Any]) -> dict[str, Any]:
    reps = probe.reports
    pr = [r["practice"] for r in reps]
    kids = [r for r in recs if r.gym]
    rooms = RoomIds.from_vocab(actor.vocab)
    boss = rooms.boss
    c: Counter[str] = Counter()
    bad: list[dict[str, Any]] = []
    exact = a.mode == "exact"
    for uid, puid in store.parent_of.items():
        kf, pf = probe.frames.get(uid, []), probe.frames.get(puid, [])
        if not probe.keep:
            break
        game = store.game[uid]
        mode = "game" if game else "fight"
        if not kf:
            c[f"{mode}_no_decision"] += 1
            continue
        s0 = kf[0][2]
        pidx = {x[2]: i for i, x in enumerate(pf)}
        assert s0 in pidx, "the parent's fork frame is not among its decisions"
        p0 = pidx[s0]
        c[f"{mode}_children"] += 1
        c[f"{mode}_first_frame_equal"] += kf[0][0] == pf[p0][0] and kf[0][3] == pf[p0][3]
        floor = kf[0][6]
        inside = [x[5] == 1 and x[6] == floor for x in kf]
        second = [x[5] == 1 and x[7] == boss and x[8] > x[9] for x in kf]  # a decision of the act's second boss
        end = store.ends.get(uid)
        if end is None:
            c[f"{mode}_no_end"] += 1
            continue
        c[f"{mode}_ended"] += 1
        c[f"{mode}_end_terminal"] += end.terminal
        if game:
            c["game_acted_past_fight"] += not all(inside)
            c["game_never_back_into_fight"] += inside == sorted(inside, reverse=True)
            c["game_reached_second_boss"] += any(second)
            if end.terminal and end.result == 1:
                c["game_win"] += 1
            elif end.terminal:
                where = "first" if inside[-1] else "second" if second[-1] else "other"
                c[f"game_loss_{where}"] += 1
                c["game_loss_truncated"] += end.truncated
        segs = probe.segs.get(uid, [])
        if not segs:
            c[f"{mode}_no_segment"] += 1
            continue
        ref = episode_targets([*(x[1] for x in pf[:p0]), *(x[1] for x in kf), end], rooms, store.reward_cfg)
        sl = slice(s0, s0 + len(kf))
        mask = _cat(segs, "win_mask")
        ok = {"steps": [x[2] for x in kf] == list(range(s0, s0 + len(kf))),
              "segments": sorted(segs, key=lambda s: s["start"])[0]["start"] == s0,
              "rewards": np.array_equal(_cat(segs, "rewards"), ref.rewards[sl]),
              "combat_id": np.array_equal(_cat(segs, "combat_id"), ref.combat_id[sl]),
              "kinds": np.array_equal(_cat(segs, "kinds"), np.asarray([x[1].kind for x in kf], np.int8)),
              "actions": np.array_equal(_cat(segs, "actions"), np.asarray([x[4] for x in kf]))}
        if game:  # the true labels on every unmasked step (masked: only force-released segments, the full run's rule)
            ok["win"] = ref.win is None or bool(np.all(_cat(segs, "win")[mask] == ref.win))
            ok["floors_left"] = ref.floors_left is None or np.array_equal(_cat(segs, "floors_left")[mask],
                                                                          ref.floors_left[sl][mask])
            ok["last_segment_labelled"] = ref.win is None or bool(sorted(segs, key=lambda s: s["start"])[-1]
                                                                  ["win_mask"].all())
            c["game_masked_steps"] += int((~mask).sum())
            c["game_steps"] += len(mask)
        else:
            ok["win_masked"] = not mask.any()
        c[f"{mode}_recomputed"] += 1
        for k, v in ok.items():
            c[f"{mode}_{k}_ok"] += bool(v)
        if not all(ok.values()) and len(bad) < 10:
            bad.append({"uid": uid, "mode": mode, **{k: bool(v) for k, v in ok.items()}})
        if exact:
            n = min(len(kf), len(pf) - p0)
            first_diff = next((t for t in range(n) if kf[t][4] != pf[p0 + t][4]), n)
            c[f"{mode}_exact_prefix_frames_equal"] += all(kf[t][0] == pf[p0 + t][0]
                                                          for t in range(min(first_diff + 1, n)))
            c[f"{mode}_exact_replayed_to_end"] += first_diff == len(kf) == len(pf) - p0
            c[f"{mode}_exact_prefix_decisions"] += first_diff
    game_recs = [r for r in kids if r.practice["mode"] == "game"]
    fin = [r for r in game_recs if r.practice["result"] in ("win", "loss", "cap")]
    span = occ["t_drain"] - occ["t_start"] if occ["draining"] else seconds
    classes: dict[str, Counter] = {}
    asc: dict[str, Counter] = {}
    for x in pr:
        for name, v in x["classes"].items():
            classes.setdefault(name, Counter()).update(v)
        for name, v in x["classes_asc"].items():
            asc.setdefault(name, Counter()).update(v)
    child_dec = sum(x["decisions"] for x in pr)
    scout_dec = sum(x["scout_decisions"] for x in pr)
    return {
        "mode": a.mode, "envs": a.envs, "max_children": a.max_children, "per_parent": a.per_parent,
        "seconds": round(seconds, 1), "forking_seconds": round(span, 1),
        "decisions_per_s": sum(r["decisions"] for r in reps) / seconds,
        "forks": sum(x["forks"] for x in pr), "children": sum(x["children"] for x in pr),
        "fork_failures": sum(x["fork_failures"] for x in pr), "lost": sum(x["lost"] for x in pr),
        "capped": sum(x["capped"] for x in pr), "trimmed": sum(x["trimmed"] for x in pr),
        "child_decisions": child_dec, "scout_decisions": scout_dec,
        "game_decisions": sum(v["game_decisions"] for v in classes.values()),
        "child_share_of_actor_decisions": child_dec / max(1, child_dec + scout_dec),
        "g_last": pr[-1]["g"] if pr else None, "q_last": pr[-1]["q"] if pr else None,
        "fork_ms_p50": float(np.median([x["fork_ms"][0] for x in pr if x["fork_ms"][0] is not None] or [np.nan])),
        "arrive_ms_p50": float(np.median([x["arrive_ms"][0] for x in pr if x["arrive_ms"][0] is not None] or [np.nan])),
        "kid_results": dict(Counter(f"{r.practice['mode']}:{r.practice['result']}" for r in kids)),
        "game_child_decisions": _q([r.decisions for r in fin]),
        "game_child_seconds": _q([r.seconds for r in fin]),
        "slots_held_mean": {"game": occ["int_game"] / span, "fight": occ["int_fight"] / span},
        "slots_held_max": {"game": occ["max_game"], "all": occ["max_all"]},
        "classes": {k: dict(v) for k, v in sorted(classes.items())}, "classes_asc": {k: dict(v) for k, v in
                                                                                     sorted(asc.items())},
        "checks": dict(c), "bad": bad,
        "kid_frames_checked": probe.kid_checked, "kid_frame_problems": probe.kid_problems[:10],
        "errors": Counter(e.get("kind", "error") for e in errs),
        "lost_details": [e["detail"][:200] for e in errs if e.get("kind") == "lost" and "gym" in e][:20],
        "fork_failure_details": [e["detail"][:200] for e in errs if e.get("kind") == "gym_fork"][:20],
        "left_processes": left,
        "parents": dict(Counter(r.result for r in recs if not r.gym)),
        "parents_by_asc": dict(Counter(f"A{r.ascension}:{r.result}" for r in recs if not r.gym)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
