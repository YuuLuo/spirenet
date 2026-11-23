"""Actor pile cap (ActorConfig.pile_cap): a decision frame whose seat holds more combat cards than the cap ends the
episode like the host's decision cap (terminal reward, record "pile_cap", host RESET); 0 = off changes nothing."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np
import pytest

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy, seat_pile_totals
from spire.metrics import TRUNCATIONS, EpisodeRecord, OutcomeTracker
from spire.proto import Frame, parse_frame
from spire.rewards import RewardConfig, RoomIds, episode_targets, frame_facts
from spire.rollout import RolloutConfig, RolloutStore, Segment, SegmentEnd
from spire.schema import CardCol, G, Kind, MsgType, Result, SeatCol, Table, TerminalReason, ViewMode, Zone
from spire.synth import FakeHost, FakeHostConfig, FrameGen, random_frame
from spire.tokenize import Tokenizer
from spire.train import ActorSection, TrainConfig, build
from spire.vocab import Vocab

TEAM4 = "IRONCLAD+IRONCLAD+SILENT+SILENT"


Capped = tuple[EpisodeRecord, Frame, list[tuple[Frame, int]]]  # record, frame over the cap, decided (frame, action)


def run(vocab: Vocab, lineup: str, decisions: int, **cfg: Any
        ) -> tuple[Actor, list[FakeHost], list[Segment], list[EpisodeRecord], list[Capped]]:
    """Train-mode actor on 3 fake hosts with fixed seeds; every episode the pile cap ended."""
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=40, hashes=False, abort_prob=0.0))
             for i in range(3)]
    segs: list[Segment] = []
    recs: list[EpisodeRecord] = []
    decided: dict[int, list[tuple[Frame, int]]] = defaultdict(list)
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16))
    sampler = EpisodeSampler(EpisodeDist(lineups=((lineup, 1.0),)), vocab, 0)
    actor = Actor(hosts, UniformPolicy(0), Tokenizer(vocab), sampler, ActorConfig(**cfg), store=store,
                  on_episode=recs.append, on_decision=lambda env, f, a: decided[env].append((f, a)))
    capped: list[Capped] = []
    inner = actor._pile_cap

    def spy(env: Any, f: Frame) -> bool:
        n = len(recs)
        hit = inner(env, f)
        if hit:
            assert len(recs) == n + 1
            capped.append((recs[-1], f, [x for x in decided[env.env_id] if x[0].episode_id == f.episode_id]))
        return hit
    actor._pile_cap = spy  # type: ignore[method-assign]
    actor.run(max_decisions=decisions)
    cap = int(cfg.get("pile_cap", 0))
    if cap:  # no decided combat frame was over the cap
        assert all(not f.glob[G.IN_COMBAT] or seat_pile_totals(f).max() <= cap for fs in decided.values() for f, _ in fs)
    return actor, hosts, segs, recs, capped


def test_seat_pile_totals(gen: FrameGen) -> None:
    rng = np.random.default_rng(3)
    for view, n in ((ViewMode.TEAM, 4), (ViewMode.SEAT, 3), (ViewMode.TEAM, 1)):
        f = parse_frame(random_frame(gen, rng, Kind.COMBAT, n_seats=n, view=view))
        seats, cards = f.table(Table.SEATS), f.table(Table.CARDS)
        tot = seat_pile_totals(f)
        for s in range(n):
            if not seats[s, SeatCol.PRIVATE_VISIBLE]:
                assert tot[s] == -1
                continue
            play = cards[(cards[:, CardCol.ZONE] == Zone.PLAY) & (cards[:, CardCol.SEAT] == s), CardCol.COUNT].sum()
            assert tot[s] == sum(int(seats[s, c]) for c in (SeatCol.HAND_COUNT, SeatCol.DRAW_COUNT,
                                                          SeatCol.DISCARD_COUNT, SeatCol.EXHAUST_COUNT)) + play
        assert (tot >= 0).sum() == seats[:, SeatCol.PRIVATE_VISIBLE].sum()


def check_capped(vocab: Vocab, cap: int, actor: Actor, hosts: list[FakeHost], segs: list[Segment],
                 recs: list[EpisodeRecord], capped: list[Capped]) -> None:
    """Every pile_cap record: ended at the first frame over the cap, stored with the decision cap's rewards."""
    done = [r for r in recs if r.result == "pile_cap"]
    assert done and [c[0] for c in capped] == done and len(done) == actor.counters.pile_caps
    assert sum(h.stats.abandoned for h in hosts) == len(done)  # the open episode was replaced by a RESET
    assert actor.counters.episodes_started == len(recs) + sum(e.job is not None for e in actor.envs)
    rewards, rooms = RewardConfig(), RoomIds.from_vocab(vocab)
    for r, f, steps in capped:
        tot = seat_pile_totals(f)
        seat = int(tot.argmax())
        assert f.glob[G.IN_COMBAT] and tot[seat] > cap and r.in_combat
        assert r.detail == f"seat {seat}: {int(tot[seat])} cards in its combat piles > pile_cap {cap} at step {f.step_id}"
        assert r.decisions == len(steps) == f.step_id and [g.step_id for g, _ in steps] == list(range(f.step_id))
        assert r.act == f.gvals[G.ACT_INDEX] and r.enemies and r.version_first >= 0
        assert r.terminal_reason == 0 and r.abort_reason == 0 and r.combats_won == f.gvals[G.COMBATS_WON] >= 0
        assert (r.elites_won, r.bosses_won, r.floor) == (f.gvals[G.ELITES_WON], f.gvals[G.BOSSES_WON],
                                                        f.gvals[G.TOTAL_FLOOR])
        term = frame_facts(f)._replace(msg_type=int(MsgType.TERMINAL), result=int(Result.LOSS),
                                       terminal_reason=int(TerminalReason.DECISION_CAP))
        want = episode_targets([*(frame_facts(g) for g, _ in steps), term], rooms, rewards)
        assert want.outcome.finished and want.components["terminal"][-1] == pytest.approx(rewards.death)
        mine = [s for s in segs if s.meta.env_id == r.env]
        eps = defaultdict(list)
        for s in mine:
            eps[s.meta.episode_uid].append(s)
        acts = np.asarray([a for _, a in steps], np.int64)
        got = [v for v in eps.values() if np.array_equal(np.concatenate([s.actions for s in v]), acts)]
        assert len(got) == 1, "the capped episode is stored once"
        ep = got[0]
        assert np.allclose(np.concatenate([s.rewards for s in ep]), want.rewards)  # the decision cap's rewards
        assert ep[-1].end == SegmentEnd.TERMINAL and ep[-1].bootstrap_value == 0.0  # no bootstrap
        assert all(s.win_mask.all() and not s.win.any() for s in ep)  # a failure (win target 0)


def test_pile_cap_ends_episode_like_decision_cap(vocab: Vocab) -> None:
    actor, hosts, segs, recs, capped = run(vocab, "IRONCLAD", 1500, pile_cap=25)
    check_capped(vocab, 25, actor, hosts, segs, recs, capped)
    assert "pile_cap" in TRUNCATIONS
    tr = OutcomeTracker()
    for r in recs:
        tr.add(r)
    s = tr.summary()["IRONCLAD|A0"]
    assert s["failures"]["pile_cap"] == len(capped)  # counted separately, inside the win-rate denominator
    assert s["n"] == sum(r.result in ("win", "loss", *TRUNCATIONS) for r in recs)
    assert actor.store is not None and actor.store.stats.host_truncated == len(capped)  # the decision-cap path


def test_pile_cap_one_seat_ends_the_team_episode(vocab: Vocab) -> None:
    actor, hosts, segs, recs, capped = run(vocab, TEAM4, 1500, pile_cap=30)
    check_capped(vocab, 30, actor, hosts, segs, recs, capped)
    assert all(r.lineup == TEAM4 and len(r.final_hp) == 4 for r, _, _ in capped)
    assert 1 in [(seat_pile_totals(f) > 30).sum() for _, f, _ in capped]  # one seat over the cap is enough


def digest(segs: list[Segment], recs: list[EpisodeRecord]) -> tuple[str, str]:
    h = hashlib.sha256()
    for s in segs:
        h.update(repr((s.meta, s.start, s.end, s.bootstrap_value)).encode())
        for a in (s.actions, s.logp, s.values, s.kinds, s.n_legal, s.policy_version, s.rewards, s.combat_id, s.win,
                  s.win_mask, s.floors_left, s.hp_post, s.hp_mask):
            h.update(np.ascontiguousarray(a).tobytes())
        for x in s.samples:
            for a in (x.types, x.raw, x.rel, x.edges, x.cond, x.history):
                h.update(np.ascontiguousarray(a).tobytes())
            h.update(repr(x.counts).encode())
    r = hashlib.sha256(repr([{k: v for k, v in x.as_dict().items() if k != "seconds"} for x in recs]).encode())
    return h.hexdigest(), r.hexdigest()


@pytest.mark.parametrize("lineup", ["IRONCLAD", TEAM4])
def test_pile_cap_off_is_identical(vocab: Vocab, lineup: str) -> None:
    off = run(vocab, lineup, 600)
    zero = run(vocab, lineup, 600, pile_cap=0)
    high = run(vocab, lineup, 600, pile_cap=10**6)
    assert off[0].counters.pile_caps == high[0].counters.pile_caps == 0 and not off[4] and not high[4]
    assert digest(off[2], off[3]) == digest(zero[2], zero[3]) == digest(high[2], high[3])
    assert len(off[3]) > 3 and len(off[2]) > 20


def test_pile_cap_config() -> None:
    cfg = build(TrainConfig, {"run_name": "pile-cap", "actor": {"pile_cap": 600}})
    assert cfg.actor.pile_cap == 600 and ActorSection().pile_cap == 0 and ActorConfig().pile_cap == 0
    with pytest.raises(AssertionError):
        ActorConfig(pile_cap=-1)
