"""Pass A: long-episode diagnostics, per-kind entropy, and the exact second-run config."""
from __future__ import annotations

import json
import math
import tomllib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from ppo_stubs import StubNet, StubTokenizer, obs_frame, terminal_frame
from spire.actor import Actor, ActorConfig, EpisodeJob, JobList, UniformPolicy
from spire.learner import Learner, PPOConfig, _Accum
from spire.metrics import EpisodeRecord, JsonlWriter, MetricsWriter, read_jsonl, win_decision_percentiles
from spire.proto import ResetCmd, StepCmd
from spire.rewards import RoomIds, frame_facts
from spire.rollout import RolloutStore, Segment, build_batch
from spire.schema import DECISION_KINDS, ActionCol, Kind, Result, Verb
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.train import REPO, Aggregator, TrainConfig, config_dict, load_config
from spire.trainstep import MicroStepper, _Res
from spire.vocab import Vocab


def test_long_episode_counts_actions_and_kinds_and_resets(vocab: Vocab) -> None:
    host = FakeHost(vocab, config=FakeHostConfig(min_decisions=6000, max_decisions=6000,
                                                abort_prob=0, frame_pool=1, hashes=False))
    verbs: list[Counter[str]] = []
    kinds: list[Counter[str]] = []
    caps: list[int] = []
    handle = host.game.handle

    def tap(cmd: Any) -> list[bytes]:
        if isinstance(cmd, ResetCmd):
            caps.append(cmd.max_decisions)
            verbs.append(Counter())
            kinds.append(Counter())
        if isinstance(cmd, StepCmd):
            ep = host.game.ep
            assert ep is not None and ep.last_actions is not None
            verbs[-1][Verb(int(ep.last_actions[cmd.action_index, ActionCol.VERB])).name] += 1
            kinds[-1][Kind(ep.kind[0]).name] += 1
        return handle(cmd)

    host.game.handle = tap  # type: ignore[method-assign]
    records: list[EpisodeRecord] = []
    jobs = [EpisodeJob((vocab.id("character", "IRONCLAD"),), "IRONCLAD", 0, "000000000001", "one"),
            EpisodeJob((vocab.id("character", "IRONCLAD"),), "IRONCLAD", 0, "000000000002", "two")]
    actor = Actor([host], UniformPolicy(7), Tokenizer(vocab), JobList(jobs),
                  ActorConfig(mode="eval", max_episode_decisions=5001), on_episode=records.append)
    actor.run()
    assert caps == [5001, 5001]
    assert len(records) == 2
    for rec, expected_verbs, expected_kinds in zip(records, verbs, kinds):
        assert rec.result == "decision_cap" and rec.decisions == 5001
        assert rec.action_verbs == expected_verbs and rec.decision_kinds == expected_kinds
        assert sum(rec.action_verbs.values()) == sum(rec.decision_kinds.values()) == 5001


def test_episode_length_metrics_survive_resume(tmp_path: Path) -> None:
    cfg = TrainConfig(run_name="lengths", run_root=str(tmp_path))
    metrics = MetricsWriter(cfg.run_dir / "metrics.jsonl")
    episodes = JsonlWriter(cfg.run_dir / "episodes.jsonl")
    errors = JsonlWriter(cfg.run_dir / "errors.jsonl")
    try:
        agg = Aggregator(cfg, metrics, episodes, errors, {})
        agg.emit_train(1)
        for length, result in [(20, "win"), (5000, "win"), (6000, "win"), (30000, "decision_cap"), (10, "lost")]:
            agg.handle("episode", 0, EpisodeRecord("IRONCLAD", 0, result, decisions=length).as_dict())
        agg.write_state()
        state = json.loads((cfg.run_dir / "train_state.json").read_text(encoding="utf-8"))
        resumed = Aggregator(cfg, metrics, episodes, errors, state)
        resumed.handle("episode", 0, EpisodeRecord("IRONCLAD", 0, "win", decisions=8000).as_dict())
        resumed.emit_train(1)
    finally:
        metrics.close()
        episodes.close()
        errors.close()
    rows = list(read_jsonl(cfg.run_dir / "metrics.jsonl"))
    assert rows[0]["long_episode_decision_share"] is None
    assert rows[0]["win_episode_decisions"] == {"p50": None, "p99": None, "max": None}
    assert rows[-1]["long_episode_decision_share"] == pytest.approx(44000 / 49030)
    assert rows[-1]["win_episode_decisions"] == {"p50": 5500.0, "p99": 7940.0, "max": 8000}
    assert len(list(read_jsonl(cfg.run_dir / "episodes.jsonl"))) == 6
    expanded = [20] * 3 + [5001] * 7 + [7000] * 2
    quantiles = win_decision_percentiles(dict(Counter(str(x) for x in expanded)))
    assert [quantiles["p50"], quantiles["p99"]] == pytest.approx(np.percentile(expanded, [50, 99]))


@pytest.mark.parametrize("path", ["generic", "fast_body"])
def test_per_kind_entropy_reaches_loss_and_gradients(vocab: Vocab, path: str) -> None:
    cfg = PPOConfig(epochs=1, lr=0, bf16=False, adv_norm=False, grad_clip=1000,
                    ent_coef=0.01, ent_coef_by_kind=((Kind.REST, .03), (Kind.CARD_REWARD, .03), (Kind.SHOP, .03)),
                    vf_coef=0, aux_win=0, aux_floors=0, aux_hp=0)
    tok = StubTokenizer()
    net = StubNet(tok.n_features)
    bias = np.array([-1.0, .4, .7, 1.0], np.float32)
    with torch.no_grad():
        net.pi.bias.copy_(torch.from_numpy(bias))
    learner = Learner(net, tok, cfg, device="cpu")  # type: ignore[arg-type]
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab))
    for i, kind in enumerate((*DECISION_KINDS, Kind.CARD_REWARD)):
        n_actions = 1 if i == len(DECISION_KINDS) else 2 + i % 3
        frame = obs_frame(kind, 1, 0, False, [80], n_actions=n_actions)
        store.begin_episode(i, (2,), 0)
        store.add_step(i, frame_facts(frame), tok.encode(frame), 0, 0, 0, 0)
        store.end_episode(i, frame_facts(terminal_frame(Result.LOSS, 1, 0, [0])))
    rb = build_batch(segs, cfg.gae, 0)
    rb.advantages.fill(0)
    expected_bonus = 0.0
    expected_grad = np.zeros(4)
    n_pol = int((rb.n_legal > 1).sum())
    for kind, n in zip(rb.kinds, rb.n_legal):
        if n == 1:
            continue
        coef = .03 if kind in (Kind.REST, Kind.CARD_REWARD, Kind.SHOP) else .01
        weights = [math.exp(float(x)) for x in bias[:n]]
        probabilities = [x / sum(weights) for x in weights]
        entropy = -sum(p * math.log(p) for p in probabilities)
        expected_bonus += coef * entropy / n_pol
        expected_grad[:n] += [coef * p * (math.log(p) + entropy) / n_pol for p in probabilities]
    acc = _Accum(torch.device("cpu"))
    if path == "generic":
        learner._minibatch(list(rb.samples), np.arange(len(rb)), None, learner._tensors(rb), acc, 0)
    else:
        pf, pi, _ = learner._columns(rb)
        st = object.__new__(MicroStepper)
        st.model, st.fwd, st.cfg, st.acc, st.autocast = net, net, cfg, acc, False
        st._exploration = st._uniform_kl = None
        st.den = torch.tensor([n_pol, len(rb), len(rb), 1, len(rb), 1, 1], dtype=torch.float32)
        st.mb_kl = torch.zeros(())
        st._body(cast(_Res, SimpleNamespace(views=tok.collate(list(rb.samples)),
                                          pf=torch.from_numpy(pf), pi=torch.from_numpy(pi))))
    assert float(acc.loss["ent_bonus"]) == pytest.approx(expected_bonus, abs=1e-8)
    assert net.pi.bias.grad is not None
    np.testing.assert_allclose(net.pi.bias.grad.numpy(), expected_grad, atol=1e-8)


