"""T6 of notes/gym/practice-spec.md: practice fights on the real fork host of the tree-n dist with a trained policy.

    python -B scripts/practice_real.py resample --ckpt C --run-config R --host-config T --out DIR [--forks 300]
    python -B scripts/practice_real.py exact    ...  (EXACT + greedy, host with --allow-debug-options: tests only)

A train-mode practice actor (the run config's actor / tokenizer / exploration settings; ``--scouts``: v1.1 parents)
over ``--envs`` ``SpireHost --serve --search`` processes, SILENT at ``--ascensions``, until ``--forks`` FORKs were
answered; then no new fork, and the run goes on until every child has ended. Checks (``summary.json``): every child's
first frame == the parent's frame at the fork, trailers included (the actor asserts it; counted here from the
``on_decision`` frames); every child ends by FIGHT_END + one frame, lost children = 0; every post-fight OBS outside
the fight; every acted child frame inside it; the independent recomputation (rewards, combat ids, kinds of the
child's segments == episode_targets(the parent's frames before the fork + the child's frames + its end frame), all
from the ``on_decision`` frames). ``exact``: a child's frames == its parent's up to the first differing action, and
a fight_won child's post-fight frame == the parent's frame at that step.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
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

from spire import gym  # noqa: E402
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy  # noqa: E402
from spire.evalpanel import load_policy_net  # noqa: E402
from spire.host import HostConfig, HostProcess, real_host_spec  # noqa: E402
from spire.metrics import EpisodeRecord  # noqa: E402
from spire.proto import Frame, frame_problems  # noqa: E402
from spire.rewards import Facts, RewardConfig, RoomIds, episode_targets, frame_facts  # noqa: E402
from spire.rollout import EpisodeMeta, RolloutConfig, RolloutStore, Segment  # noqa: E402
from spire.schema import G, H  # noqa: E402
from spire.tokenize import Tokenizer  # noqa: E402
from spire.vocab import Catalog  # noqa: E402

KID_BASE = 1 << 20


def masked(payload: bytes | bytearray | memoryview) -> bytes:
    b = bytearray(payload)
    b[H.HOST_US * 4:H.HOST_US * 4 + 4] = b"\0\0\0\0"
    return hashlib.sha1(b).digest()


class Store(RolloutStore):
    """Logs the episode uid of every recorded step (scouts' included: they line up with ``on_decision``), the
    parent uid of every child and every end frame's facts (TERMINAL or post-fight)."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.uids: list[int] = []
        self.parent_of: dict[int, int] = {}
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
        return meta

    def end_episode(self, env_id: int, terminal: Facts, **k: Any) -> Any:
        self.ends[self._tracks[env_id].meta.episode_uid] = terminal
        return super().end_episode(env_id, terminal, **k)

    def truncate_episode(self, env_id: int, next_facts: Facts, *a: Any, **k: Any) -> None:
        self.ends[self._tracks[env_id].meta.episode_uid] = next_facts
        super().truncate_episode(env_id, next_facts, *a, **k)


class Probe:
    def __init__(self) -> None:
        self.n = 0
        self.frames: dict[int, list[tuple[bytes, Facts, int, int, int, int, int]]] = {}  # uid -> masked, facts,
        # step, episode, action, IN_COMBAT, TOTAL_FLOOR
        self.pending: list[tuple[int, Frame, int]] = []
        self.kid_checked = 0
        self.kid_problems: list[str] = []
        self.segs: dict[int, list[dict[str, Any]]] = {}
        self.reports: list[dict[str, Any]] = []
        self.forks = 0
        self.kid_pids: set[int] = set()
        self.post: dict[int, bytes] = {}  # child env id -> masked post-fight frame (latest)

    def on_report(self, rep: dict[str, Any]) -> None:
        self.reports.append(rep)
        self.forks += rep["practice"]["forks"]

    def on_decision(self, env_id: int, f: Frame, a: int) -> None:
        self.pending.append((env_id, f, a))

    def flush(self, store: Store) -> None:
        uids = store.uids[self.n:self.n + len(self.pending)]
        assert len(uids) == len(self.pending)
        for (env_id, f, a), uid in zip(self.pending, uids):
            fr = self.frames.setdefault(uid, [])
            fr.append((masked(f.payload), frame_facts(f), f.step_id, f.episode_id, a, f.glob[G.IN_COMBAT],
                       f.glob[G.TOTAL_FLOOR]))
            if env_id >= KID_BASE and (len(fr) == 1 or len(fr) % 10 == 0):
                self.kid_checked += 1
                self.kid_problems += frame_problems(f)
        self.n += len(self.pending)
        self.pending = []

    def sink(self, s: Segment) -> None:
        self.segs.setdefault(s.meta.episode_uid, []).append(
            {"start": s.start, "n": len(s), "rewards": s.rewards.copy(), "actions": s.actions.copy(),
             "kinds": s.kinds.copy(), "combat_id": s.combat_id.copy(), "end": int(s.end),
             "win_mask": bool(s.win_mask.any())})


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
    assert tok.config.calc and tok.config.forecast, "T6 runs a calc + forecast checkpoint"
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
    pcfg = gym.PracticeConfig(actors=1, scouts=a.scouts and not exact, socket_dir=f"/tmp/spire-gym-{os.getpid()}",
                              k_max=a.k_max)
    dist = EpisodeDist(lineups=(("SILENT", 1.0),), ascensions=tuple((x, 1.0) for x in a.ascensions))
    actor = Actor(hosts, policy, tok, EpisodeSampler(dist, tok.vocab, (a.seed, 1)), cfg, store=store,
                  on_episode=recs.append, on_error=errs.append, on_report=probe.on_report,
                  on_decision=probe.on_decision, catalog=catalog, practice=pcfg, practice_q=0.5,
                  practice_children=a.max_children, gym_seed=a.seed * 7 + 1)
    if exact:
        assert actor.gym is not None
        actor.gym.mode = gym.MODE_EXACT
        policy.mode = "greedy"
    return actor, hosts, store


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            s = fh.read().decode()
    except FileNotFoundError:
        return False
    return s[s.rindex(")") + 2:].split()[0] != "Z"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("resample", "exact"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--host-config", required=True)
    ap.add_argument("--removal-ratings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--max-children", type=int, default=24)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--min-group", type=int, default=16)
    ap.add_argument("--forks", type=int, default=300)
    ap.add_argument("--ascensions", type=int, nargs="+", default=[10])
    ap.add_argument("--scouts", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu-gb", type=float, default=3.0)
    a = ap.parse_args()
    Path(a.out, "host-logs").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    probe = Probe()
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
    state = {"draining": False}
    t0 = time.monotonic()

    def stop() -> bool:
        probe.flush(store)
        if not state["draining"] and probe.forks + g.counters.forks >= a.forks:
            state["draining"] = True
            g.cfg = dataclasses.replace(g.cfg, ascensions=(-1,))  # no new fork from here on
        return state["draining"] and all(k.kid is None for k in g.kids)

    actor.run(stop, max_seconds=a.max_seconds)
    probe.flush(store)
    seconds = time.monotonic() - t0
    for h in hosts:
        h.close()
    time.sleep(3.0)
    left = [p for p in [*host_pids, *probe.kid_pids] if alive(p)]
    out = analyse(a, actor, store, probe, recs, errs, seconds, left)
    Path(a.out, "summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, default=str))
    return 0


def _cat(segs: list[dict[str, Any]], k: str) -> np.ndarray:
    return np.concatenate([s[k] for s in sorted(segs, key=lambda s: s["start"])])


def analyse(a: argparse.Namespace, actor: Actor, store: Store, probe: Probe, recs: list[EpisodeRecord],
            errs: list[dict[str, Any]], seconds: float, left: list[int]) -> dict[str, Any]:
    reps = probe.reports
    pr = [r["practice"] for r in reps]
    kids = [r for r in recs if r.gym]
    rooms = RoomIds.from_vocab(actor.vocab)
    c: Counter[str] = Counter()
    bad: list[dict[str, Any]] = []
    exact = a.mode == "exact"
    for uid, puid in store.parent_of.items():
        kf, pf = probe.frames.get(uid, []), probe.frames.get(puid, [])
        if not kf:
            c["no_decision"] += 1
            continue
        s0 = kf[0][2]
        pidx = {x[2]: i for i, x in enumerate(pf)}
        assert s0 in pidx, "the parent's fork frame is not among its decisions"
        p0 = pidx[s0]
        c["children"] += 1
        c["first_frame_equal"] += kf[0][0] == pf[p0][0] and kf[0][3] == pf[p0][3]
        floor = kf[0][6]
        c["acted_inside_fight"] += all(x[5] == 1 and x[6] == floor for x in kf)
        end = store.ends.get(uid)
        if end is None:
            c["no_end"] += 1
            continue
        c["ended"] += 1
        c["end_terminal"] += end.terminal
        if not end.terminal:
            c["post_fight_outside"] += not (end.in_combat and end.total_floor == floor)
        segs = probe.segs.get(uid, [])
        if not segs:
            c["no_segment"] += 1
            continue
        ref = episode_targets([*(x[1] for x in pf[:p0]), *(x[1] for x in kf), end], rooms, store.reward_cfg)
        sl = slice(s0, s0 + len(kf))
        ok = {"steps": [x[2] for x in kf] == list(range(s0, s0 + len(kf))),
              "segments": sorted(segs, key=lambda s: s["start"])[0]["start"] == s0,
              "rewards": np.array_equal(_cat(segs, "rewards"), ref.rewards[sl]),
              "combat_id": np.array_equal(_cat(segs, "combat_id"), ref.combat_id[sl]),
              "kinds": np.array_equal(_cat(segs, "kinds"), np.asarray([x[1].kind for x in kf], np.int8)),
              "actions": np.array_equal(_cat(segs, "actions"), np.asarray([x[4] for x in kf])),
              "win_masked": not any(s["win_mask"] for s in segs)}
        c["recomputed"] += 1
        for k, v in ok.items():
            c[f"{k}_ok"] += bool(v)
        if not all(ok.values()) and len(bad) < 10:
            bad.append({"uid": uid, **{k: bool(v) for k, v in ok.items()}})
        if exact:
            n = min(len(kf), len(pf) - p0)
            first_diff = next((t for t in range(n) if kf[t][4] != pf[p0 + t][4]), n)
            c["exact_prefix_frames_equal"] += all(kf[t][0] == pf[p0 + t][0] for t in range(min(first_diff + 1, n)))
            if first_diff == len(kf) and not end.terminal and p0 + len(kf) < len(pf):
                c["exact_fight_replayed"] += 1
                c["exact_post_frame_equal"] += end == pf[p0 + len(kf)][1]
            elif first_diff == len(kf) and end.terminal:
                c["exact_fight_replayed_to_terminal"] += 1
    kid_res = Counter(r.practice["result"] for r in kids)
    by_class = Counter(f"{r.practice['class']}:{r.practice['result']}" for r in kids)
    return {
        "mode": a.mode, "envs": a.envs, "scouts": a.scouts, "seconds": round(seconds, 1),
        "decisions_per_s": sum(r["decisions"] for r in reps) / seconds,
        "forks": sum(x["forks"] for x in pr), "children": sum(x["children"] for x in pr),
        "fork_failures": sum(x["fork_failures"] for x in pr), "lost": sum(x["lost"] for x in pr),
        "capped": sum(x["capped"] for x in pr), "trimmed": sum(x["trimmed"] for x in pr),
        "child_decisions": sum(x["decisions"] for x in pr), "scout_decisions": sum(x["scout_decisions"] for x in pr),
        "fork_ms_p50": float(np.median([x["fork_ms"][0] for x in pr if x["fork_ms"][0] is not None] or [np.nan])),
        "arrive_ms_p50": float(np.median([x["arrive_ms"][0] for x in pr if x["arrive_ms"][0] is not None] or [np.nan])),
        "kid_results": dict(kid_res), "by_class": dict(by_class), "checks": dict(c), "bad": bad,
        "kid_frames_checked": probe.kid_checked, "kid_frame_problems": probe.kid_problems[:10],
        "errors": Counter(e.get("kind", "error") for e in errs),
        "lost_details": [e["detail"][:200] for e in errs if e.get("kind") == "lost" and "gym" in e][:20],
        "fork_failure_details": [e["detail"][:200] for e in errs if e.get("kind") == "gym_fork"][:20],
        "left_processes": left, "g_last": pr[-1]["g"] if pr else None,
        "parents": dict(Counter(r.result for r in recs if not r.gym)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
