"""Bounded evaluation, requested devices, and partial-round reporting without wall-clock races."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from spire import evalpanel, sysutil
from spire.actor import TorchPolicy, UniformPolicy
from spire.dashboard import build_html
from spire.evalpanel import evaluate, make_panel
from spire.metrics import read_jsonl
from spire.model import PRESETS, SpireNet
from spire.proto import ResetCmd
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.vocab import Vocab


class DecisionClock:
    """Advance only when the test host receives a decision; no machine-load dependence."""

    elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed


class TimedHost(FakeHost):
    def __init__(self, vocab: Vocab, clock: DecisionClock, length: int) -> None:
        super().__init__(vocab, config=FakeHostConfig(min_decisions=length, max_decisions=length,
                                                    mean_decisions=length, abort_prob=0.0, hashes=False))
        self.clock = clock
        self.reset_caps: list[int] = []

    def send_reset(self, cmd: ResetCmd) -> None:
        self.reset_caps.append(cmd.max_decisions)
        super().send_reset(cmd)

    def send_step(self, action_index: int, *, step_id: int | None = None) -> None:
        super().send_step(action_index, step_id=step_id)
        self.clock.elapsed += 1.0


def test_eval_forwards_the_cap_on_every_reset(vocab: Vocab, tmp_path: Path) -> None:
    panel = make_panel("cap", 3, ["IRONCLAD"], [0], rng_seed=1)
    host = TimedHost(vocab, DecisionClock(), length=20)
    try:
        res, _ = evaluate(panel, UniformPolicy(0), Tokenizer(vocab), [host], tag="cap", policy_info={},
                          max_episode_decisions=2, out_dir=tmp_path)
    finally:
        host.close()
    assert host.reset_caps == [2, 2, 2]
    assert [e["result"] for e in res.episodes] == ["decision_cap"] * 3
    assert [e["decisions"] for e in res.episodes] == [2] * 3
    assert res.decisions == 6 and res.summary["all"]["wins"] == 0
    assert not res.partial and res.stop_reason == "complete"
    assert res.completed_episodes == res.expected_episodes == 3


@pytest.mark.parametrize(("budget", "completed"), [(2.0, 0), (5.0, 1)])
def test_eval_deadline_reports_only_completed_games(vocab: Vocab, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch,
                                                   budget: float, completed: int) -> None:
    clock = DecisionClock()
    monkeypatch.setattr(evalpanel, "time", clock)
    panel = make_panel("partial", 3, ["IRONCLAD"], [0], rng_seed=2)
    host = TimedHost(vocab, clock, length=3)
    try:
        res, path = evaluate(panel, UniformPolicy(0), Tokenizer(vocab), [host], tag="timed", policy_info={},
                             round_budget_s=budget, out_dir=tmp_path)
    finally:
        host.close()
    assert res.partial and res.stop_reason == "round_budget"
    assert res.completed_episodes == res.summary["all"]["n"] == len(res.episodes) == completed
    assert res.expected_episodes == 3
    assert res.decisions == int(budget)  # unfinished games still consume the evaluator's budget
    logged = list(read_jsonl(path.with_suffix(".episodes.jsonl")))
    assert len(logged) == completed + 1 and logged[-1]["result"] == "shutdown"
    assert all(e["result"] != "shutdown" for e in res.episodes)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["partial"] and saved["completed_episodes"] == completed and saved["expected_episodes"] == 3
    dashboard = build_html(tmp_path / "run", tmp_path)
    assert f"partial {completed}/3 (round_budget)" in dashboard
    assert "每轮把整个面板打一遍" not in dashboard


def test_eval_stop_also_reports_a_partial_round(vocab: Vocab, tmp_path: Path) -> None:
    panel = make_panel("stop", 2, ["IRONCLAD"], [0], rng_seed=3)
    host = TimedHost(vocab, DecisionClock(), length=3)
    try:
        res, path = evaluate(panel, UniformPolicy(0), Tokenizer(vocab), [host], tag="stop", policy_info={},
                             stop=lambda: True, out_dir=tmp_path)
    finally:
        host.close()
    assert path.exists() and res.partial and res.stop_reason == "shutdown"
    assert res.completed_episodes == 0 and res.expected_episodes == 2


def test_checkpoint_loader_keeps_requested_cuda_and_ema(vocab: Vocab, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    # Capture placement without requiring CUDA: unavailable CUDA must never select CPU silently.
    class Net:
        def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            self.loaded = state

        def to(self, device: torch.device) -> Net:
            self.device = device
            return self

        def eval(self) -> Net:
            return self

    net = Net()
    monkeypatch.setattr(evalpanel, "SpireNet", lambda *args: net)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ckpt = tmp_path / "weights.pt"
    torch.save({"ema": {"weight": torch.tensor(3.0)}, "model": {"weight": torch.tensor(-2.0)},
                "counters": {"version": 7}, "extra": {"vocab": "synthetic", "vocab_hash": vocab.hash,
                    "catalog": "none", "tokenizer": {}, "model_config": {}}}, ckpt)
    _, _, version, _ = evalpanel.load_policy_net(ckpt, device="cuda")
    assert net.device == torch.device("cuda") and version == 7
    assert net.loaded["weight"].item() == 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_evaluator_loads_ema_and_completes_capped_panel(vocab: Vocab, tmp_path: Path) -> None:
    tok = Tokenizer(vocab)
    model_config = PRESETS["2.5m"]
    model = SpireNet(tok.layout, model_config, None)
    online = model.state_dict()
    ema = {name: value.clone() for name, value in online.items()}
    changed = next(iter(dict(model.named_parameters())))
    ema[changed].add_(0.01)
    checkpoint = tmp_path / "cuda-eval.pt"
    torch.save({"model": online, "ema": ema, "counters": {"version": 9},
                "extra": {"vocab": "synthetic", "vocab_hash": vocab.hash, "catalog": "none",
                          "tokenizer": asdict(tok.config), "model_config": asdict(model_config)}}, checkpoint)
    loaded, loaded_tok, version, _ = evalpanel.load_policy_net(checkpoint, device="cuda")
    assert next(loaded.parameters()).device.type == "cuda" and version == 9
    torch.testing.assert_close(loaded.state_dict()[changed].cpu(), ema[changed], rtol=0, atol=0)
    assert not torch.equal(ema[changed], online[changed])
    policy = TorchPolicy(loaded, loaded_tok, mode="greedy", version=version, use_graphs=True)
    host = TimedHost(vocab, DecisionClock(), length=5)
    panel = make_panel("cuda", 2, ["IRONCLAD"], [0], rng_seed=7)
    try:
        res, _ = evaluate(panel, policy, loaded_tok, [host], tag="ema-greedy", out_dir=tmp_path,
                          policy_info={"weights": "ema", "mode": "greedy"},
                          max_batch=1, max_episode_decisions=2)
    finally:
        policy.close()
        host.close()
    assert host.reset_caps == [2, 2]
    assert [e["result"] for e in res.episodes] == ["decision_cap", "decision_cap"]
    assert [e["decisions"] for e in res.episodes] == [2, 2]
    assert res.decisions == 4 and res.completed_episodes == res.expected_episodes == 2
    assert not res.partial and res.stop_reason == "complete"


def test_periodic_eval_routes_cuda_limits_and_reports_partial_rounds(vocab: Vocab, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    clock = DecisionClock()
    monkeypatch.setattr(evalpanel, "time", clock)
    monkeypatch.setattr(sysutil, "apply_placement", lambda *args: {})
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    panel = make_panel("periodic", 4, ["IRONCLAD"], [0], rng_seed=4)
    monkeypatch.setattr(evalpanel, "ensure_eval_panel", lambda cfg: panel)
    tok = Tokenizer(vocab)
    hosts: list[TimedHost] = []
    policies: list[ClosingPolicy] = []
    placements: list[dict[str, Any]] = []
    policy_args: list[dict[str, Any]] = []

    class ClosingPolicy(UniformPolicy):
        closed = False

        def close(self) -> None:
            self.closed = True

    def load(*args: Any, **kwargs: Any) -> tuple[Any, Tokenizer, int, dict[str, Any]]:
        placements.append(kwargs)
        return object(), tok, len(placements), {}

    def policy(*args: Any, **kwargs: Any) -> ClosingPolicy:
        policy_args.append(kwargs)
        p = ClosingPolicy(0)
        policies.append(p)
        return p

    def make_hosts(*args: Any, **kwargs: Any) -> list[TimedHost]:
        h = TimedHost(vocab, clock, length=20)
        hosts.append(h)
        return [h]

    messages: list[tuple[str, int, dict[str, Any]]] = []
    stop = SimpleNamespace(is_set=lambda: len(messages) == 2, wait=lambda timeout: False)
    monkeypatch.setattr(evalpanel, "load_policy_net", load)
    monkeypatch.setattr(evalpanel, "TorchPolicy", policy)
    monkeypatch.setattr(evalpanel, "make_hosts", make_hosts)
    run = tmp_path / "run"
    (run / "ckpt").mkdir(parents=True)
    (run / "ckpt" / "latest.pt").touch()
    evalpanel.eval_main({"run_name": "run", "run_root": str(tmp_path), "catalog": "none",
                        "eval": {"enabled": True, "device": "cuda", "priority": "normal",
                                 "envs": 1, "first_after_s": 0.0, "interval_s": 0.0,
                                 "max_episode_decisions": 2, "round_budget_s": 5.0,
                                 "results_dir": str(tmp_path / "results")}},
                       SimpleNamespace(put=messages.append), stop, SimpleNamespace(value=10000))
    assert placements == [{"device": "cuda", "weights": "ema"}] * 2
    assert len(hosts) == len(policies) == 2
    assert all(h._closed for h in hosts) and all(p.closed for p in policies)
    assert all(h.reset_caps == [2, 2, 2] for h in hosts)
    assert [p["mode"] for p in policy_args] == ["greedy", "greedy"]
    assert all(p["use_graphs"] for p in policy_args)
    for kind, slot, record in messages:
        assert kind == "eval" and slot == 0
        assert record["partial"] and record["stop_reason"] == "round_budget"
        assert record["completed_episodes"] == record["episodes"] == record["summary"]["all"]["n"] == 2
        assert record["expected_episodes"] == 4 and record["decisions"] == 5
    assert messages[-1][2]["eval_decisions_total"] == 10
