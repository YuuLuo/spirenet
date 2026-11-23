"""Tree p2 (kickstart v2 + combat temperature + practice fights): the interactions the lead defined (notes/next/p2.md).
Calc / forecast inputs with all three: nothing new (children inherit the trailers; practice T6), not tested here."""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import test_practice as tp
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
from spire.gym import PracticeConfig, Priors
from spire.ipc import decode_segment, pack_segment
from spire.kickstart import KickstartConfig, load_teacher
from spire.learner import FileWeightPublisher, Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import GAEConfig, RolloutConfig, RolloutStore, Segment
from spire.schema import Kind
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import TrainConfig, acting_policy, build, check_practice
from spire.vocab import Vocab
from test_kickstart import save_teacher

T = 0.5
TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32)
WIDE = ModelConfig(d_model=48, n_layers=2, n_heads=4, d_ff=96, d_id=16, d_cond=16, d_static=32, init_std=0.3)
EXPL = ((int(Kind.CARD_REWARD), 0.1), (int(Kind.CARD_SELECT), 0.1), (int(Kind.COMBAT), 0.05))
COMBAT = (int(Kind.COMBAT), int(Kind.CARD_SELECT))
GAE = GAEConfig(gamma=1.0, lam=0.99, macro_micro=True, lam_macro=0.99, lam_micro=0.99, stitch_segments=True)


def practice_run(vocab: Vocab, tok: Tokenizer, net: SpireNet, tmp_path: Path, *, scouts: bool, temperature: float,
                 episodes: int = 4) -> tuple[Actor, list[Segment], tp._Policy, tp._Store]:
    """test_practice.run_actor with a combat temperature on the policy and the beh_temp column in the store."""
    policy = tp._Policy(net, tok, mode="sample", use_graphs=False, seed=1, exploration_by_kind=tp.EXPLORATION,
                        combat_temperature=temperature)
    segs: list[Segment] = []
    store = tp._Store(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16), beh_temp=True)
    recs: list[Any] = []
    errs: list[dict] = []
    hosts = [tp.gym_host(tmp_path, 0)]
    practice = PracticeConfig(actors=1, scouts=scouts, priors=Priors(0.25, 0.35, 0.55, 0.9),
                              socket_dir=str(tmp_path / "s"))
    try:
        actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(ascensions=((0, 1.0),)), vocab, 0),
                      ActorConfig(max_batch=16, min_group=1, report_interval=0.2, gym_env_id_base=tp.KID_BASE),
                      store=store, on_episode=recs.append, on_error=errs.append, practice=practice, practice_q=0.5,
                      practice_children=8, gym_seed=5)
        actor.run(lambda: sum(not r.gym for r in recs) >= episodes, max_seconds=300.0)
    finally:
        for h in hosts:
            h.close()
    assert not errs and sum(bool(r.gym) for r in recs) > 5
    assert tp._orphans(tmp_path) == []
    return actor, segs, policy, store


# ----------------------------------------------------------------------- temperature x practice
@tp.linux
def test_temperature_x_practice(vocab: Vocab, tmp_path: Path) -> None:
    """A child plays the training behaviour with the step's combat temperature: its stored beh_temp is T on its combat
    steps and 1 elsewhere, and the learner (T) reproduces its stored log q (first-minibatch KL 0). A scout takes the
    arg-max of the untempered scores on its combat steps and ships no segment; the non-arg-max counters see no scout."""
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True, cycle_ban=True, commit_select=True))
    torch.manual_seed(0)
    net = SpireNet(tok.layout, tp.TINY)
    actor, segs, policy, store = practice_run(vocab, tok, copy.deepcopy(net), tmp_path, scouts=True, temperature=T)
    assert segs and all(s.meta.env_id >= tp.KID_BASE for s in segs)  # children only: scouts train nothing
    tempered = 0
    for s in segs:
        c = np.isin(s.kinds, COMBAT)
        assert (s.beh_temp[c] == np.float32(T)).all() and (s.beh_temp[~c] == 1).all()
        tempered += int(c.sum())
    assert tempered > 100
    scout = [x for x in policy.log if x[0] is not None]
    rows = 0
    for greedy, expl, logits, actions in scout:
        assert not expl
        for i in np.flatnonzero(greedy):
            assert actions[i] == int(np.argmax(np.where(np.isfinite(logits[i]), logits[i], -np.inf)))
            rows += 1
    assert rows > 100
    c = actor.counters
    assert c.combat_steps + c.noncombat_steps == store.gym_stats.steps_added and c.combat_nonargmax > 0
    segs = [decode_segment(pack_segment(s)) for s in segs]
    n = sum(len(s) for s in segs)
    cfg = PPOConfig(minibatch=4 * n, micro_batch=256, epochs=1, exploration_by_kind=tp.EXPLORATION,
                    combat_temperature=T, gae=GAE)
    lr = Learner(copy.deepcopy(net), tok, cfg, device="cpu", fast=False)
    m = lr.update(segs)
    assert lr._tempered and m["first_minibatch_kl"] < 1e-10 and m["beh_temp_mean"] < 1.0, m


# ----------------------------------------------------------------------- temperature x kickstart
def tempered_segments(vocab: Vocab, tok: Tokenizer, policy: TorchPolicy, n: int = 1500) -> list[Segment]:
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32), beh_temp=True)
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
             for i in range(4)]
    Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0), ("IRONCLAD", 1.0))), vocab, 0),
          ActorConfig(), store=store).run(max_decisions=n)
    return [decode_segment(pack_segment(s)) for s in segs]


def test_temperature_x_kickstart_phase1(vocab: Vocab, tmp_path: Path) -> None:
    """The kick KL is on the untempered base policies: on tempered segments it equals the kick of the same segments
    learnt without the temperature (same weights, one minibatch), while the PPO loss differs."""
    tok = Tokenizer(vocab)
    torch.manual_seed(5)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, WIDE), tok)
    torch.manual_seed(0)
    net = SpireNet(tok.layout, TINY)
    pol = TorchPolicy(copy.deepcopy(net), tok, seed=7, exploration_by_kind=EXPL, combat_temperature=T)
    segs = tempered_segments(vocab, tok, pol)
    n = sum(len(s) for s in segs)
    rows = {}
    for t in (T, 1.0):
        teacher, sha = load_teacher(path, tok, None, TINY)
        cfg = PPOConfig(minibatch=4 * n, micro_batch=256, epochs=1, exploration_by_kind=EXPL, combat_temperature=t,
                        kickstart=KickstartConfig(str(path), 1.0, 0.5, 10))
        lr = Learner(copy.deepcopy(net), tok, cfg, device="cpu", fast=False, teacher=teacher, teacher_sha256=sha)
        m = lr.update(segs if t != 1.0 else [dataclasses.replace(s, beh_temp=None) for s in segs])
        assert lr._tempered == (t != 1.0)
        rows[t] = m
    a, b = rows[T]["kickstart"], rows[1.0]["kickstart"]
    assert a["kick_kl"] > 0 and (a["kick_kl"], a["kick_v"], a["kick_agree"]) == (b["kick_kl"], b["kick_v"], b["kick_agree"])
    assert rows[T]["first_minibatch_kl"] < 1e-10 < rows[1.0]["first_minibatch_kl"]


def test_temperature_x_kickstart_phase0(vocab: Vocab, tmp_path: Path) -> None:
    """Phase 0: the acting teacher samples with the run's ppo.combat_temperature and the steps store beh_temp; the
    phase-0 update on them uses no ratio (PPO terms 0)."""
    common = {"run_root": str(tmp_path), "vocab": "synthetic", "catalog": "none", "seed": 3,
              "tokenizer": {"token_buckets": [20, 28, 40, 64, 80, 96, 128, 192, 256, 384]},
              "actor": {"use_graphs": False, "device": "cpu"}, "rollout": {"segment_len": 32}}
    t_cfg = build(TrainConfig, {**common, "run_name": "teacher", "model": {"override": asdict(WIDE)}})
    tk = Tokenizer(vocab, t_cfg.tokenizer)
    torch.manual_seed(5)
    tpath = save_teacher(tmp_path / "teacher.pt", SpireNet(tk.layout, WIDE), tk)
    s_cfg = build(TrainConfig, {**common, "run_name": "student", "model": {"override": asdict(TINY)},
                                "ppo": {"exploration_by_kind": [list(x) for x in EXPL], "epochs": 1,
                                        "combat_temperature": T, "kickstart": {"teacher": str(tpath), "phase": 0}}})
    FileWeightPublisher(s_cfg.run_dir / "weights").publish(SpireNet(tk.layout, TINY).state_dict(), 0)
    policy = acting_policy(s_cfg, tk, None, torch.device("cpu"), 1, 0)
    assert policy.frozen and policy.combat_temperature == T and policy.wait_for_weights(10.0)
    segs = tempered_segments(vocab, tk, policy)
    policy.close()
    temps = np.concatenate([s.beh_temp for s in segs])
    assert (temps == np.float32(T)).sum() > 200 and (temps == 1).sum() > 200
    teacher, sha = load_teacher(tpath, tk, None, TINY)
    torch.manual_seed(0)
    lr = Learner(SpireNet(tk.layout, TINY), tk, s_cfg.ppo, device="cpu", fast=False, teacher=teacher,
                 teacher_sha256=sha)
    m = lr.update(segs)
    assert lr.distil and lr._tempered and m["beh_temp_mean"] == pytest.approx(float(temps.mean()))
    assert m["loss_pi"] == 0.0 and m["approx_kl"] == 0.0 and m["clipfrac"] == 0.0 and not m["early_stop"]
    assert m["kickstart"]["phase"] == 0 and m["kickstart"]["kick_kl"] > 0


# ----------------------------------------------------------------------- practice x kickstart
def _cfg(tmp_path: Path, phase: int) -> TrainConfig:
    return build(TrainConfig, {"run_name": f"k{phase}", "run_root": str(tmp_path), "actor": {"actors": 2},
                               "ppo": {"gae": {"macro_micro": True}, "epochs": 1,
                                       "kickstart": {"teacher": str(tmp_path / "t.pt"), "phase": phase}},
                               "practice": {"actors": 1}})


def test_practice_refused_in_kickstart_phase0(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="practice is refused in kickstart phase 0"):
        check_practice(_cfg(tmp_path, 0))
    check_practice(_cfg(tmp_path, 1))


class _CountingRows(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.rows = 0

    def forward(self, batch):  # type: ignore[no-untyped-def]
        out = self.inner(batch)
        self.rows += int(out["logits"].shape[0])
        return out


@tp.linux
def test_practice_x_kickstart_phase1(vocab: Vocab, tmp_path: Path) -> None:
    """Phase 1: children are ordinary student-acted trained episodes and the learner's teacher forward covers every
    sample of their segments like any segment's."""
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True, cycle_ban=True, commit_select=True))
    torch.manual_seed(5)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, WIDE), tok)
    torch.manual_seed(0)
    net = SpireNet(tok.layout, tp.TINY)
    _, segs, _, _ = practice_run(vocab, tok, copy.deepcopy(net), tmp_path / "run", scouts=False, temperature=1.0)
    kids = [s for s in segs if s.meta.env_id >= tp.KID_BASE]
    assert kids and len(kids) < len(segs)
    teacher, sha = load_teacher(path, tok, None, tp.TINY)
    counting = _CountingRows(teacher)
    n = sum(len(s) for s in segs)
    cfg = PPOConfig(minibatch=4 * n, micro_batch=256, epochs=1, exploration_by_kind=tp.EXPLORATION, gae=GAE,
                    kickstart=KickstartConfig(str(path), 1.0, 0.5, 10))
    lr = Learner(copy.deepcopy(net), tok, cfg, device="cpu", fast=False, teacher=counting, teacher_sha256=sha)
    m = lr.update(segs)
    assert counting.rows == n and m["kickstart"]["teacher_evaluated"] and m["kickstart"]["kick_kl"] > 0
