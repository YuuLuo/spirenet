"""Actor loop: seeds, episode sources, accounting, fault handling, policies (CPU + CUDA graphs)."""
from __future__ import annotations

import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from spire.actor import (
    SEED_ALPHABET,
    Actor,
    ActorConfig,
    EpisodeDist,
    EpisodeJob,
    EpisodeSampler,
    JobList,
    Policy,
    RunTotals,
    TorchPolicy,
    UniformPolicy,
    canonical_seed,
    parse_lineup,
    random_seed,
    seed_to_wire,
    wire_to_seed,
)
from spire.host import HostConfig, HostProcess
from spire.learner import FileWeightPublisher, FileWeightSource
from spire.metrics import EpisodeRecord
from spire.model import ModelConfig, SpireNet, export_weights
from spire.proto import parse_frame
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import H, ActionCol, FrameFlag, Kind, Verb
from spire.synth import FakeGame, FakeHost, FakeHostConfig, FrameGen, random_frame
from spire.tokenize import Tokenizer
from spire.vocab import Vocab

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "fake_spirehost.py")
TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)


@pytest.fixture(scope="module")
def tok(vocab: Vocab) -> Tokenizer:
    return Tokenizer(vocab)


def fake_hosts(vocab: Vocab, n: int, **cfg: object) -> list[FakeHost]:
    opts: dict[str, object] = dict(mean_decisions=40, hashes=False)
    opts.update(cfg)
    return [FakeHost(vocab, seed=i, config=FakeHostConfig(**opts)) for i in range(n)]  # type: ignore[arg-type]


def train_actor(vocab: Vocab, tok: Tokenizer, hosts: list, policy: Policy | None = None, **cfg: object
                ) -> tuple[Actor, RolloutStore, list[Segment], list[EpisodeRecord], list[dict]]:
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
    recs: list[EpisodeRecord] = []
    errs: list[dict] = []
    actor = Actor(hosts, policy or UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(), vocab, 0),
                  ActorConfig(**cfg), store=store, on_episode=recs.append, on_error=errs.append)  # type: ignore[arg-type]
    return actor, store, segs, recs, errs


# --------------------------------------------------------------------------- seeds / sources


def test_seed_codec() -> None:
    assert seed_to_wire("000000000000") == 0
    assert seed_to_wire("ZZZZZZZZZZZZ") == 34**12 - 1 < 2**62
    assert canonical_seed(" abcio0 ") == "ABC100"
    assert seed_to_wire("o00000000001") == 1  # O -> 0 like SeedHelper.CanonicalizeSeed
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = random_seed(rng)
        assert len(s) == 12 and set(s) <= set(SEED_ALPHABET)
        assert wire_to_seed(seed_to_wire(s)) == s
    assert len({random_seed() for _ in range(50)}) == 50  # CSPRNG
    for bad in ("SHORT", "0000000000000", "00000000000_"):
        with pytest.raises(ValueError):
            seed_to_wire(bad)


def test_sampler_distribution_and_exclusion(vocab: Vocab) -> None:
    dist = EpisodeDist(lineups=(("IRONCLAD", 3.0), ("SILENT+SILENT", 1.0)), ascensions=((0, 1.0), (10, 1.0)))
    a = EpisodeSampler(dist, vocab, seed=7)
    jobs = [a.next_job(0) for _ in range(4000)]
    c = Counter(j.lineup for j in jobs)
    assert abs(c["IRONCLAD"] / 4000 - 0.75) < 0.03
    assert {j.ascension for j in jobs} == {0, 10}
    assert all(len(j.characters) == (2 if j.lineup == "SILENT+SILENT" else 1) for j in jobs)
    b = EpisodeSampler(dist, vocab, seed=7)
    assert [b.next_job(0) for _ in range(50)] == jobs[:50]
    excl = frozenset(j.seed for j in jobs[:100] if j.seed)
    c2 = EpisodeSampler(dist, vocab, seed=7, exclude=excl)
    assert not {c2.next_job(0).seed for _ in range(200)} & excl
    host_seeded = EpisodeSampler(EpisodeDist(seeds="host"), vocab)
    assert host_seeded.next_job(0).seed is None
    with pytest.raises(ValueError):
        parse_lineup("IRONCLAD+NOBODY", vocab)


# --------------------------------------------------------------------------- train mode accounting


def test_train_accounting_is_conserved(vocab: Vocab, tok: Tokenizer) -> None:
    hosts = fake_hosts(vocab, 6, abort_prob=0.2)
    actor, store, segs, recs, _ = train_actor(vocab, tok, hosts, max_batch=4, min_group=2)
    reports: list[dict] = []
    actor.on_report = reports.append
    actor.run(max_decisions=3000)
    st = store.stats
    assert st.steps_added == actor.counters.decisions >= 3000
    assert st.steps_added == st.steps_released + st.steps_dropped  # nothing pending after shutdown
    assert sum(len(s) for s in segs) == st.steps_released
    assert len(recs) == actor.counters.episodes_started == actor.counters.episodes_ended
    res = Counter(r.result for r in recs)
    assert res["shutdown"] == 6 and res["aborted"] > 0 and res["win"] + res["loss"] > 10
    assert res["win"] + res["loss"] == st.wins + st.losses and res["aborted"] == st.aborted
    assert sum(r.decisions for r in recs) == actor.counters.decisions
    for s in segs:  # contiguous, legal, stamped
        assert (s.n_legal > 0).all() and (s.actions < s.n_legal).all() and (s.policy_version == 0).all()
    finished = [r for r in recs if r.result in ("win", "loss")]
    assert all(r.seed and r.lineup in ("IRONCLAD", "SILENT") for r in finished)
    tot = RunTotals()
    for r in reports:
        tot.add(r)
    assert tot.decisions == actor.counters.decisions
    us = tot.us_per_decision()
    assert us["encode"] > 0 and us["python"] > 0


def test_error_frames_are_recorded_and_episode_discarded(vocab: Vocab, tok: Tokenizer) -> None:
    hosts = fake_hosts(vocab, 4, error_prob=0.03, abort_prob=0.0)
    actor, store, segs, recs, errs = train_actor(vocab, tok, hosts)
    actor.run(max_decisions=1500)
    res = Counter(r.result for r in recs)
    assert res["error"] == len(errs) == store.stats.lost > 0
    for e in errs:  # reproduction record
        assert e["code"] > 0 and e["seed"] and isinstance(e["actions"], list)
    assert sum(h.stats.errors for h in hosts) == len(errs)
    assert store.stats.steps_added == store.stats.steps_released + store.stats.steps_dropped


def test_overflow_episode_dropped(vocab: Vocab, tok: Tokenizer) -> None:
    class OverflowGame(FakeGame):
        def _obs(self, action: int) -> bytes:
            payload = super()._obs(action)
            assert self.ep is not None
            if self.ep.ctx.step_id == 3:  # mark the 4th decision frame of every episode as overflowed
                buf = bytearray(payload)
                flags = struct.unpack_from("<i", buf, H.FLAGS * 4)[0] | int(FrameFlag.OVERFLOW)
                struct.pack_into("<i", buf, H.FLAGS * 4, flags)
                payload = self.ep.last_obs = bytes(buf)
            return payload

    hosts = fake_hosts(vocab, 2, abort_prob=0.0)
    for i, h in enumerate(hosts):
        h.game = OverflowGame(vocab, i, FakeHostConfig(mean_decisions=40, hashes=False, abort_prob=0.0))
    actor, store, segs, recs, _ = train_actor(vocab, tok, hosts)
    actor.run(max_decisions=200)
    over = [r for r in recs if r.result == "overflow"]
    assert over and store.stats.overflow == len(over) == sum(h.stats.overflow_episodes for h in hosts)
    assert all(r.decisions == 3 for r in over)
    assert not segs  # every episode overflowed before finishing: nothing reaches training


def test_host_crash_and_hang_over_pipes(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    def proc(*args: str, **cfg: object) -> HostProcess:
        opts: dict[str, object] = dict(start_timeout=30.0, reset_timeout=5.0, step_timeout=1.0,
                                       log_path=str(tmp_path / f"h{len(args)}.log"), name="t", max_restarts=20)
        opts.update(cfg)
        return HostProcess(HostConfig(command=(sys.executable, "-B", SCRIPT, "--serve", "--no-hashes",
                                               "--mean-decisions", "30", *args), **opts))  # type: ignore[arg-type]

    # host 0 crashes every 25 steps and has a restart budget of 3: it is retired after the 4th crash
    hosts = [proc("--crash-after", "25", max_restarts=3), proc("--hang-after", "40", "--seed", "1"),
             proc("--seed", "2")]
    try:
        actor, store, _, recs, _ = train_actor(vocab, tok, hosts)
        actor.run(max_seconds=12.0)
        lost = [r for r in recs if r.result == "lost"]
        details = " ".join(r.detail for r in lost)
        assert "HostCrashed" in details and "HostTimeout" in details
        assert hosts[0].stats.crashes >= 1 and hosts[1].stats.timeouts >= 1
        assert hosts[0].stats.restarts == 3 and hosts[1].stats.restarts >= 1
        assert actor.counters.hosts_retired == 1 and actor.envs[0].dead and not actor.envs[1].dead
        assert "host retired" in details
        assert store.stats.lost == len(lost)
        assert sum(r.result in ("win", "loss") for r in recs) > 3  # the healthy host keeps playing
    finally:
        for h in hosts:
            h.close()


def test_eval_mode_plays_each_job_once(vocab: Vocab, tok: Tokenizer) -> None:
    jobs = [EpisodeJob(parse_lineup(lu, vocab), lu, a, random_seed(np.random.default_rng(i)), str(i))
            for i, (lu, a) in enumerate([("IRONCLAD", 0), ("SILENT", 0), ("DEFECT+DEFECT", 5)] * 7)]
    recs: list[EpisodeRecord] = []
    actor = Actor(fake_hosts(vocab, 5), UniformPolicy(1), tok, JobList(jobs), ActorConfig(mode="eval"),
                  on_episode=recs.append)
    actor.run()
    assert sorted(r.seed_id for r in recs) == sorted(j.seed_id for j in jobs)
    assert {(r.seed, r.lineup, r.ascension) for r in recs} == {(j.seed, j.lineup, j.ascension) for j in jobs}
    with pytest.raises(ValueError):
        Actor(fake_hosts(vocab, 1), UniformPolicy(), tok, JobList(jobs), ActorConfig(mode="train"))


def test_learnable_fake_rewards_the_right_rule(vocab: Vocab) -> None:
    def play(good: bool) -> float:
        h = FakeHost(vocab, seed=3, config=FakeHostConfig(learnable=True, mean_decisions=150, frame_pool=8,
                                                          abort_prob=0.0, hashes=False))
        rng = np.random.default_rng(0)
        wins = 0
        for _ in range(40):
            f = h.reset(None, (4,), 0)
            while not f.is_terminal:
                v = f.actions[:, ActionCol.VERB]
                ok = np.flatnonzero(~((v == Verb.END_TURN) & (v == Verb.PLAY_CARD).any())) if good \
                    else np.arange(f.n_actions)
                f = h.step(int(rng.choice(ok)))
            wins += f.result == 1
        return wins / 40

    assert play(True) == 1.0 and play(False) < 0.5


# --------------------------------------------------------------------------- policies


def test_torch_policy_cpu_and_weight_refresh(vocab: Vocab, tok: Tokenizer, tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = SpireNet(tok.layout, TINY)
    pub = FileWeightPublisher(tmp_path)
    pub.publish(export_weights(net), 0)
    net2 = SpireNet(tok.layout, TINY)
    pol = TorchPolicy(net2, tok, weights=FileWeightSource(tmp_path), poll_interval=0.05, seed=1)
    try:
        assert pol.wait_for_weights(10.0) and pol.version == 0
        assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), net2.state_dict().values()))
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.01)
        pub.publish(export_weights(net), 3)
        assert pol.wait_for_weights(10.0) and pol.version == 3
    finally:
        pol.close()
    hosts = fake_hosts(vocab, 4)
    actor, store, segs, recs, _ = train_actor(vocab, tok, hosts, pol)
    actor.run(max_decisions=400)
    assert store.stats.steps_added >= 400
    for s in segs:
        assert (s.logp <= 0).all() and np.isfinite(s.values).all() and (s.policy_version == 3).all()


def test_greedy_and_sampling_math(vocab: Vocab, tok: Tokenizer) -> None:
    pol = TorchPolicy(SpireNet(tok.layout, TINY), tok, mode="sample", seed=0)
    logits = np.array([[0.0, np.log(3.0), -np.inf, -np.inf]] * 20000, np.float32)
    a, logp, ent = pol._sample(logits)
    assert set(a.tolist()) == {0, 1} and abs((a == 1).mean() - 0.75) < 0.02
    assert np.allclose(logp[a == 1], np.log(0.75), atol=1e-5) and np.allclose(logp[a == 0], np.log(0.25), atol=1e-5)
    assert np.allclose(ent, -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)), atol=1e-5)
    g = TorchPolicy(SpireNet(tok.layout, TINY), tok, mode="greedy")
    assert (g._sample(logits[:3])[0] == 1).all()
    hot = TorchPolicy(SpireNet(tok.layout, TINY), tok, temperature=2.0, seed=0)
    a2, lp2, _ = hot._sample(logits)
    assert abs((a2 == 1).mean() - np.sqrt(3) / (1 + np.sqrt(3))) < 0.02  # p ~ exp(z / T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_policy_matches_eager_and_bounds_captures(vocab: Vocab, tok: Tokenizer, gen: FrameGen) -> None:
    torch.manual_seed(0)
    net = SpireNet(tok.layout, TINY).cuda()
    graph = TorchPolicy(net, tok, mode="greedy")
    eager = TorchPolicy(net, tok, mode="greedy", use_graphs=False)
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = int(rng.integers(1, 20))
        samples = [tok.encode(parse_frame(random_frame(gen, rng, Kind(int(rng.choice([1, 2, 3, 4]))),
                                                       hashes=False))) for _ in range(n)]
        n_act = np.array([s.n_actions for s in samples])
        h1, h2 = graph.launch(samples, n_act), eager.launch(samples, n_act)
        assert graph.done(h1) in (True, False)
        o1, o2 = graph.finish(h1), eager.finish(h2)
        assert np.allclose(o1.value, o2.value, atol=0.05)
        assert ((o1.actions == o2.actions) | (np.abs(o1.logp - o2.logp) < 0.05)).all()
        assert (o1.actions < n_act).all()
    assert graph.runner is not None and graph.runner.captures <= 2 * len(graph._caps) + 8
