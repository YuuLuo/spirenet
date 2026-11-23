"""Cycle exclusions are frozen environment state, including exact PPO legal counts."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Literal, Sequence

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeJob, JobList, PolicyOut, TorchPolicy, UniformPolicy
from spire.distributions import legal_uniform_log_probs, uniform_cross_entropy
from spire.evalpanel import Panel, evaluate
from spire.ipc import decode_segment, pack_segment
from spire.learner import Learner, PPOConfig
from spire.metrics import JsonlWriter, MetricsWriter, read_jsonl
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet, packed_layout, packed_views
from spire.proto import Command, QuitCmd, ResetCmd, StepCmd, parse_frame
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import G, FrameFlag, Kind, MsgType, OptionKind, Result, Table, TerminalReason, Verb, Zone
from spire.synth import FakeGame, FakeHost, FrameBuilder
from spire.tokenize import Batch, EncodedBatch, Sample, SampleBlock, Tokenizer, TokenizerConfig
from spire.train import Aggregator, TrainConfig, build, config_dict
from spire.vocab import Vocab


TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16)


class _MenuGame(FakeGame):
    """Two offers: each can be inspected and declined; PROCEED or a pick ends play."""

    def __init__(self, vocab: Vocab) -> None:
        super().__init__(vocab)
        self.step = 0
        self.kind = Kind.REWARD
        self.offer = 0
        self.frames: list[bytes] = []

    def handle(self, cmd: Command) -> list[bytes]:
        if isinstance(cmd, QuitCmd):
            return []
        if isinstance(cmd, ResetCmd):
            self.episode_id += 1
            self.step, self.kind, self.offer = 0, Kind.REWARD, 0
        else:
            assert isinstance(cmd, StepCmd)
            assert cmd.step_id == self.step
            self.step += 1
            if self.kind == Kind.REWARD:
                assert 0 <= cmd.action_index < 3
                self.offer = cmd.action_index
                self.kind = Kind.TERMINAL if cmd.action_index == 2 else Kind.CARD_REWARD
            else:
                assert self.kind == Kind.CARD_REWARD and 0 <= cmd.action_index < 2
                self.kind = Kind.REWARD if cmd.action_index == 0 else Kind.TERMINAL
            if self.step == 9:
                self.kind = Kind.TERMINAL
        b = FrameBuilder(MsgType.TERMINAL if self.kind == Kind.TERMINAL else MsgType.OBS)
        b.episode_id, b.step_id = self.episode_id, self.step
        b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.TOTAL_FLOOR] = self.kind, 1, 1
        b.glob[G.ROOM_TYPE] = self.vocab.id("enum:RoomType", "Monster")
        b.add(Table.SEATS, seat=0, hp=80, max_hp=80, alive=1, creature_row=-1,
              character_id=self.vocab.id("character", "IRONCLAD"))
        if self.kind == Kind.REWARD:
            for pos in range(2):
                b.add(Table.OPTIONS, kind=OptionKind.REWARD_ITEM, seat=0, pos=pos,
                      key_id=self.vocab.id("enum:RewardType", "Card"))
                b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=pos)
            b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=0)
        elif self.kind == Kind.CARD_REWARD:
            b.add(Table.OPTIONS, kind=OptionKind.CARD_REWARD_ALT, seat=0, pos=0,
                  key_id=self.vocab.id("card_reward_alt", "SKIP"))
            b.add(Table.CARDS, seat=0, zone=Zone.OFFER, pos=0, card_id=2 + self.offer, count=1)
            b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=0)
            b.add(Table.ACTIONS, verb=Verb.PICK_OFFER_CARD, seat=0, src_table=Table.CARDS, src_row=0)
        else:
            b.flags, b.result = FrameFlag.TERMINAL, Result.LOSS
            b.glob[G.TERMINAL_REASON] = TerminalReason.DECISION_CAP if self.step == 9 else TerminalReason.DEATH
            b.glob[G.POLICY_DECISIONS] = self.step
        frame = b.build(hashes=False)
        self.frames.append(frame)
        return [frame]


def _host(vocab: Vocab) -> FakeHost:
    host = FakeHost(vocab)
    host.game = _MenuGame(vocab)
    return host


class _CapturePolicy:
    """Delegate actual inference and sampling, exposing the ordinary sample launch path."""

    needs_samples = True
    version = 0

    def __init__(self, policy: TorchPolicy) -> None:
        self.policy = policy
        self.seen: list[Sample] = []
        self.logits: list[np.ndarray] = []
        self.outputs: list[PolicyOut] = []

    def _capture(self, samples: Sequence[Sample], handle: Any) -> Any:
        self.seen.extend(SampleBlock.from_samples(samples))
        assert isinstance(handle.out, np.ndarray)  # this suite deliberately runs CPU only
        self.logits.extend(handle.out[:, :-1].copy())
        return handle

    def launch(self, samples: Sequence[Sample], n_actions: np.ndarray) -> Any:
        return self._capture(samples, self.policy.launch(samples, n_actions))

    def done(self, handle: Any) -> bool:
        return self.policy.done(handle)

    def finish(self, handle: Any) -> PolicyOut:
        out = self.policy.finish(handle)
        self.outputs.append(out)
        return out

    def refresh(self) -> bool:
        return self.policy.refresh()


class _EncodedCapturePolicy(_CapturePolicy):
    def launch_encoded(self, enc: EncodedBatch, lo: int, hi: int, n_actions: np.ndarray) -> Any:
        return self._capture(enc.samples()[lo:hi], self.policy.launch_encoded(enc, lo, hi, n_actions))


def _policy(tok: Tokenizer, *, encoded: bool, mode: Literal["sample", "greedy"] = "greedy",
            epsilon: float = 0.0) -> _CapturePolicy:
    torch.manual_seed(17)
    net = SpireNet(tok.layout, TINY)
    if mode == "greedy":
        # Equal legal logits make the initial deterministic choice the first row.
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
    pairs = ((int(Kind.REWARD), epsilon), (int(Kind.CARD_REWARD), epsilon)) if epsilon else ()
    policy = TorchPolicy(net, tok, mode=mode, seed=1, use_graphs=False, exploration_by_kind=pairs)
    return _EncodedCapturePolicy(policy) if encoded else _CapturePolicy(policy)


def _run(tok: Tokenizer, policy: _CapturePolicy
         ) -> tuple[Actor, list[Segment], list[bytes], list[dict[str, Any]]]:
    host = _host(tok.vocab)
    segments: list[Segment] = []
    reports: list[dict[str, Any]] = []
    store = RolloutStore(segments.append, RoomIds.from_vocab(tok.vocab), config=RolloutConfig(segment_len=2))
    actor = Actor([host], policy, tok,
                  JobList([EpisodeJob((tok.vocab.id("character", "IRONCLAD"),), "IRONCLAD")]),
                  ActorConfig(mode="train"), store=store, on_report=reports.append)
    actor.run()
    host.close()
    policy.policy.close()
    assert isinstance(host.game, _MenuGame)
    return actor, segments, host.game.frames, reports


def _assert_sample_equal(a: Sample, b: Sample) -> None:
    assert a.counts == b.counts
    for name in ("types", "raw", "rel", "edges", "cond", "history"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


@pytest.mark.parametrize("public_history", [False, True])
@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("epsilon", [0.0, 0.2])
def test_actor_stored_probability_and_learner_accounting(vocab: Vocab, public_history: bool,
                                                        encoded: bool, epsilon: float) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=public_history, cycle_ban=True, action_align=1))
    policy = _policy(tok, encoded=encoded, mode="sample", epsilon=epsilon)
    actor, segments, _, _ = _run(tok, policy)
    assert 3 <= actor.counters.decisions <= 5
    assert actor.counters.cycle_ban_decisions >= 1
    assert actor.counters.card_offer_reopens == 0
    restored = [decode_segment(pack_segment(s)) for s in segments]
    samples = [s for segment in restored for s in segment.samples]
    actions = np.concatenate([segment.actions for segment in restored])
    stored_logp = np.concatenate([segment.logp for segment in restored])
    legal_counts = np.concatenate([segment.n_legal for segment in restored])
    for saved, seen in zip(samples, policy.seen):
        _assert_sample_equal(saved, seen)
    assert any(s.history[-6:].any() for s in samples)
    if not public_history:
        assert all(not s.history[:-6].any() for s in samples)
    with torch.no_grad():
        # Each individual launch reproduces the actor's tensor shape and execution order.
        for i, sample in enumerate(samples):
            logits = policy.policy.net(tok.collate([sample]))["logits"]
            assert np.array_equal(logits.numpy()[0], policy.logits[i])
            legal = torch.isfinite(logits)
            assert int(legal.sum()) == legal_counts[i]
            assert legal[0, actions[i]]
            eps = torch.tensor([epsilon]) if epsilon else None
            logq = legal_uniform_log_probs(logits, eps)
            assert stored_logp[i] == pytest.approx(float(logq[0, actions[i]]), abs=1e-6)
            q = torch.where(legal, logq.exp(), 0.0)
            base = logits.softmax(-1)
            expected = (1 - epsilon) * base + epsilon * legal / legal.sum(-1, keepdim=True)
            torch.testing.assert_close(q, expected, rtol=1e-6, atol=1e-7)
            entropy = -(q * logq).sum()
            actor_entropy = policy.outputs[i].entropy
            assert actor_entropy is not None
            assert float(actor_entropy[0]) == pytest.approx(float(entropy), abs=1e-6)
            barrier = uniform_cross_entropy(logits)
            expected_barrier = -logits.log_softmax(-1)[legal].mean()
            torch.testing.assert_close(barrier[0], expected_barrier)
    # A cut's bootstrap carries the same exclusions as the next stored decision.
    for segment in restored[:-1]:
        assert segment.bootstrap_sample is not None
        _assert_sample_equal(segment.bootstrap_sample, samples[segment.start + len(segment)])


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("public_history", [False, True])
def test_every_collate_path_preserves_cycle_legality(vocab: Vocab, native: bool, public_history: bool) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=public_history, cycle_ban=True), native=native)
    if native:
        assert tok._native_collate, "native collation must really run in the native test arm"
    policy = _policy(tok, encoded=True)
    actor, segments, payloads, _ = _run(tok, policy)
    assert actor.counters.decisions == 5
    samples = [s for segment in segments for s in decode_segment(pack_segment(segment)).samples]
    enc = tok.encode_batch([parse_frame(payload) for payload in payloads[:-1]])
    enc.history[:] = np.stack([sample.history for sample in samples])
    block = SampleBlock.concat([SampleBlock.from_samples(samples[:2]), SampleBlock.from_samples(samples[2:])])
    indices = np.array([4, 0, 2, 2, 3], np.int64)
    indexed_samples = [block[int(i)] for i in indices]
    nat = tok.shape_of(samples)
    shape = replace(nat, batch=nat.batch + 2, actions=nat.actions + 8, rows=nat.rows + 16)
    eager = tok.collate(samples, shape=shape)
    encoded = tok.collate_encoded(enc, shape=shape)
    layout, size = packed_layout(shape, tok.width)
    packed = packed_views(torch.full((size,), 255, dtype=torch.uint8), layout)
    tok.collate_encoded_into(enc, 0, len(samples), shape, {k: v.numpy() for k, v in packed.items()})
    for batch in (encoded, packed):
        for key in eager:
            assert torch.equal(eager[key], batch[key]), key
    indexed_nat = tok.shape_block(block, indices)
    indexed_shape = replace(indexed_nat, batch=len(indices) + 2, actions=indexed_nat.actions + 8,
                            rows=indexed_nat.rows + 16)
    layout, size = packed_layout(indexed_shape, tok.width)
    indexed = packed_views(torch.full((size,), 255, dtype=torch.uint8), layout)
    tok.collate_block_into(block, indices, indexed_shape, {k: v.numpy() for k, v in indexed.items()})
    reference = tok.collate(indexed_samples, shape=indexed_shape)
    for key in reference:
        assert torch.equal(reference[key], indexed[key]), key
    with torch.no_grad():
        for batch, ordered in ((eager, samples), (encoded, samples), (packed, samples),
                               (indexed, indexed_samples)):
            logits = policy.policy.net(batch)["logits"]
            expected = batch["act_mask"].clone()
            for i, sample in enumerate(ordered):
                for action_plus_one in sample.history[-6:]:
                    if action_plus_one:
                        expected[i, int(action_plus_one) - 1] = False
            assert torch.equal(torch.isfinite(logits), expected)
            assert not batch["history"][len(ordered):].any()  # stale graph-buffer filler cleared
            assert not torch.isfinite(logits[len(ordered):]).any()
            assert torch.isfinite(policy.policy.net(batch)["value"]).all()


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("mode", ["sample", "greedy"])
def test_evaluator_uses_the_training_cycle_mask(vocab: Vocab, tmp_path: Path, encoded: bool,
                                               mode: Literal["sample", "greedy"]) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, action_align=1))
    training = _policy(tok, encoded=encoded, mode=mode)
    actor, _, _, _ = _run(tok, training)
    evaluation = _policy(tok, encoded=encoded, mode=mode)
    host = _host(vocab)
    panel = Panel("cycles", ("000000000001",), ("IRONCLAD",), (0,))
    result, _ = evaluate(panel, evaluation, tok, [host], tag="cycle-test", policy_info={}, out_dir=tmp_path)
    host.close()
    evaluation.policy.close()
    assert result.completed_episodes == 1
    assert len(evaluation.seen) == actor.counters.decisions < 9
    for a, b in zip(training.seen, evaluation.seen):
        _assert_sample_equal(a, b)
    assert all(np.array_equal(a, b) for a, b in zip(training.logits, evaluation.logits))
    assert np.array_equal(np.concatenate([o.actions for o in training.outputs]),
                          np.concatenate([o.actions for o in evaluation.outputs]))


@pytest.mark.parametrize("public_history", [False, True])
def test_disabled_switch_preserves_parameters_outputs_and_actor_bits(vocab: Vocab, public_history: bool) -> None:
    default = Tokenizer(vocab, TokenizerConfig(public_history=public_history, action_align=1))
    explicit = Tokenizer(vocab, TokenizerConfig(public_history=public_history, cycle_ban=False, action_align=1))
    policies = [_policy(tok, encoded=True) for tok in (default, explicit)]
    for key, value in policies[0].policy.net.state_dict().items():
        assert torch.equal(value, policies[1].policy.net.state_dict()[key]), key
    runs = [_run(tok, policy) for tok, policy in zip((default, explicit), policies)]
    assert [run[0].counters.decisions for run in runs] == [9, 9]
    assert [run[0].counters.cycle_ban_decisions for run in runs] == [0, 0]
    for a, b in zip(policies[0].seen, policies[1].seen):
        _assert_sample_equal(a, b)
        assert not a.history[-6:].any()
    for output_a, output_b in zip(policies[0].outputs, policies[1].outputs):
        for field in ("actions", "logp", "value", "entropy"):
            assert np.array_equal(getattr(output_a, field), getattr(output_b, field)), field
    batch = default.collate(policies[0].seen)
    torch.manual_seed(71)
    default_net = SpireNet(default.layout, TINY).eval()
    torch.manual_seed(71)
    explicit_net = SpireNet(explicit.layout, TINY).eval()
    if public_history:
        assert default_net.history_proj is not None
        with torch.no_grad():
            default_net.history_proj.weight.normal_(0, .01)
        explicit_net.load_state_dict(default_net.state_dict())
    before: Batch = {k: v.clone() for k, v in batch.items()}
    if not public_history:
        del before["history"]  # legacy history-off model contract
        batch["history"][:, -6:] = 1  # disabled cycle flag cannot remove a legal row
    with torch.no_grad():
        original = default_net(before)
        disabled = explicit_net(batch)
    for key in OUTPUT_KEYS:
        assert torch.equal(original[key].view(torch.int32), disabled[key].view(torch.int32)), key


def test_pre_history_checkpoint_loads_with_cycle_ban_without_migration(vocab: Vocab, tmp_path: Path) -> None:
    assert not TokenizerConfig().cycle_ban
    config = TrainConfig(run_name="cycle-config", tokenizer=TokenizerConfig(cycle_ban=True))
    assert build(TrainConfig, config_dict(config)) == config
    old_tok, new_tok = Tokenizer(vocab), Tokenizer(vocab, TokenizerConfig(cycle_ban=True))
    old = Learner(SpireNet(old_tok.layout, TINY), old_tok, PPOConfig(), device="cpu", fast=False)
    for i, parameter in enumerate(old.model.parameters()):
        parameter.grad = torch.full_like(parameter, (i + 1) * .001)
    old.opt.step()
    old.opt.zero_grad(set_to_none=True)
    old._ema_step()
    old.version, old.updates, old.samples_seen = 7, 3, 100
    path = tmp_path / "pre-history.pt"
    old.save(path)
    new = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    new.load(path)  # no migration flag; this switch adds no weights or optimizer slots
    assert (new.version, new.updates, new.samples_seen) == (7, 3, 100)
    assert set(old.model.state_dict()) == set(new.model.state_dict())
    for old_model, new_model in ((old.model, new.model), (old.ema, new.ema)):
        for key, value in old_model.state_dict().items():
            assert torch.equal(value, new_model.state_dict()[key]), key
    before, after = old.opt.state_dict(), new.opt.state_dict()
    assert before["param_groups"] == after["param_groups"]
    assert set(before["state"]) == set(after["state"])
    for parameter_id, state in before["state"].items():
        assert set(state) == set(after["state"][parameter_id])
        for key, value in state.items():
            assert torch.equal(value, after["state"][parameter_id][key]), key


def test_cycle_metrics_reach_train_record_and_keep_reopens(vocab: Vocab, tmp_path: Path) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True))
    actor, _, _, reports = _run(tok, _policy(tok, encoded=True))
    assert actor.counters.cycle_ban_decisions == 2
    # Tracker tests exercise these bounds; here verify nonzero counters survive
    # the actual actor reporting and supervisor aggregation boundaries.
    actor.counters.cycle_ban_all_banned += 3
    actor.counters.cycle_ban_capacity_hits += 4
    reports.append(actor._report(time.monotonic(), emit=False))
    metrics = MetricsWriter(tmp_path / "metrics.jsonl")
    episodes = JsonlWriter(tmp_path / "episodes.jsonl")
    errors = JsonlWriter(tmp_path / "errors.jsonl")
    agg = Aggregator(TrainConfig(run_name="cycle-metrics"), metrics, episodes, errors, {})
    for report in reports:
        agg.handle("actor", 0, report)
    agg.emit_train(1.0)
    metrics.close()
    episodes.close()
    errors.close()
    row = list(read_jsonl(tmp_path / "metrics.jsonl"))[-1]
    assert row["cycle_ban_decisions"] == 2 and row["cycle_ban_share"] == .4
    assert row["cycle_ban_all_banned"] == 3 and row["cycle_ban_capacity_hits"] == 4
    assert row["card_offer_reopens"] == 0 and row["card_offer_reopen_share"] == 0


def test_uniform_policy_uses_effective_legal_rows_even_without_sample_requirement(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True))
    host = _host(vocab)
    policy = UniformPolicy(seed=3)
    assert not policy.needs_samples
    actor = Actor([host], policy, tok,
                  JobList([EpisodeJob((vocab.id("character", "IRONCLAD"),), "IRONCLAD")]),
                  ActorConfig(mode="eval"))
    actor.run()
    host.close()
    assert actor.counters.decisions == 3
    assert actor.counters.cycle_ban_decisions == 1 and actor.counters.card_offer_reopens == 0
    assert isinstance(host.game, _MenuGame)
    sample = tok.encode(parse_frame(host.game.frames[0]))
    sample.history[-6] = 1
    result = UniformPolicy(seed=3).launch([sample] * 128, np.full(128, 3, np.int64))
    assert set(result.actions) == {1, 2}
    np.testing.assert_allclose(result.logp, -np.log(2), rtol=0, atol=1e-7)
    assert result.entropy is not None
    np.testing.assert_allclose(result.entropy, np.log(2), rtol=0, atol=1e-7)


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("public_history", [False, True])
def test_bucket_sort_keeps_each_environment_cycle_mask(vocab: Vocab, encoded: bool,
                                                      public_history: bool) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, public_history=public_history))
    policy = _policy(tok, encoded=encoded)
    hosts = [_host(vocab), _host(vocab)]
    actor = Actor(hosts, policy, tok, JobList([]), ActorConfig(mode="eval", min_group=1))
    for env, cards in zip(actor.envs, (80, 1)):
        b = FrameBuilder()
        b.episode_id = 1
        b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.TOTAL_FLOOR] = Kind.REWARD, 1, 1
        b.glob[G.ROOM_TYPE] = vocab.id("enum:RoomType", "Monster")
        b.add(Table.SEATS, seat=0, hp=80, max_hp=80, alive=1, creature_row=-1)
        for pos in range(cards):
            b.add(Table.CARDS, seat=0, zone=Zone.DECK, pos=pos, card_id=2, count=1)
        for pos in range(2):
            b.add(Table.OPTIONS, kind=OptionKind.REWARD_ITEM, seat=0, pos=pos,
                  key_id=vocab.id("enum:RewardType", "Card"))
            b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=pos)
        b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=0)
        previous = parse_frame(b.build(hashes=False))
        sample = tok.encode(previous)
        assert (env.history is not None) == public_history  # the tracker exists only when the model reads it
        if env.history is not None:
            sample.history[:] = env.history.observe(previous)
        assert env.cycle is not None
        env.cycle.observe(previous, sample)
        if env.history is not None:
            env.history.record(previous, sample, env.idx)
        env.cycle.record(previous, env.idx)
        b.step_id = 1
        env.frame = parse_frame(b.build(hashes=False))
    actor.ready = list(actor.envs)
    actor._launch()
    batch = actor.inflight[0]
    assert [env.idx for env in batch.envs] == [1, 0]
    assert len(policy.seen) == 2
    for env, sample, logits in zip(batch.envs, policy.seen, policy.logits):
        assert np.array_equal(sample.history[-6:], [env.idx + 1, 0, 0, 0, 0, 0])
        assert not np.isfinite(logits[env.idx])
        assert np.isfinite(logits[:3]).sum() == 2
        if not public_history:
            assert not sample.history[:-6].any()
    for host in hosts:
        host.close()
    policy.policy.close()
