"""Public-history snapshots survive actor inference, rollout transport and evaluation."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeJob, JobList, PolicyOut, UniformPolicy
from spire.evalpanel import Panel, evaluate
from spire.history import PublicHistory
from spire.metrics import JsonlWriter, MetricsWriter, read_jsonl
from spire.proto import Command, Frame, QuitCmd, ResetCmd, StepCmd, parse_frame
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import (
    DATA_OFFSET, GLOBAL_OFFSET, N_TABLES, G, H, ActionCol, FrameFlag, Kind, MsgType, OptionCol, OptionKind,
    Result, Table, TerminalReason, Verb, Zone,
)
from spire.synth import FakeGame, FakeHost, FrameBuilder, FrameGen, random_frame
from spire.tokenize import HISTORY_ROWS, HISTORY_STATS, EncodedBatch, Sample, Tokenizer, TokenizerConfig
from spire.train import Aggregator, TrainConfig
from spire.vocab import Vocab


def reward_trace(vocab: Vocab) -> list[bytes]:
    """A replay fixture: the same visible reward remains after each real SKIP alternative."""
    out = []
    for step, kind in enumerate((Kind.REWARD, Kind.CARD_REWARD, Kind.REWARD,
                                 Kind.CARD_REWARD, Kind.REWARD, Kind.TERMINAL)):
        b = FrameBuilder(MsgType.TERMINAL if kind == Kind.TERMINAL else MsgType.OBS)
        b.step_id = step
        b.glob[G.KIND], b.glob[G.N_SEATS] = kind, 1
        b.glob[G.TOTAL_FLOOR] = 1
        b.glob[G.ROOM_TYPE] = vocab.id("enum:RoomType", "Monster")
        b.add(Table.SEATS, seat=0, hp=80, max_hp=80, alive=1,
              character_id=vocab.id("character", "IRONCLAD"), creature_row=-1)
        if kind == Kind.REWARD:
            b.add(Table.OPTIONS, kind=OptionKind.REWARD_ITEM, seat=0, pos=0,
                  key_id=vocab.id("enum:RewardType", "Card"))
            b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=0)
            b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=0)
        elif kind == Kind.CARD_REWARD:
            b.add(Table.OPTIONS, kind=OptionKind.CARD_REWARD_ALT, seat=0, pos=0,
                  key_id=vocab.id("card_reward_alt", "SKIP"))
            b.add(Table.CARDS, seat=0, zone=Zone.OFFER, pos=0, card_id=2, count=1)
            b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=0)
            b.add(Table.ACTIONS, verb=Verb.PICK_OFFER_CARD, seat=0, src_table=Table.CARDS, src_row=0)
        else:
            b.flags = FrameFlag.TERMINAL
            b.result = Result.LOSS
            b.glob[G.TERMINAL_REASON] = TerminalReason.DEATH
            b.glob[G.POLICY_DECISIONS] = step
        out.append(b.build(hashes=False))
    return out


class TraceGame(FakeGame):
    def __init__(self, vocab: Vocab) -> None:
        super().__init__(vocab)
        self.frames = reward_trace(vocab)
        self.episode = 0
        self.index = 0

    def handle(self, cmd: Command) -> list[bytes]:
        if isinstance(cmd, QuitCmd):
            return []
        if isinstance(cmd, ResetCmd):
            self.episode += 1
            self.index = 0
        else:
            assert isinstance(cmd, StepCmd)
            assert cmd.action_index == 0 and cmd.step_id == self.index
            self.index += 1
        data = bytearray(self.frames[self.index])
        struct.pack_into("<i", data, H.EPISODE_ID * 4, self.episode)
        return [bytes(data)]


class CapturePolicy(UniformPolicy):
    """Replay action zero so feature agreement is conditioned on the same action history."""
    needs_samples = True

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[Sample] = []

    def launch(self, samples: Sequence[Sample], n_actions: np.ndarray) -> PolicyOut:
        self.seen.extend(samples)
        n = len(samples)
        return PolicyOut(np.zeros(n, np.int64), np.zeros(n, np.float32), np.zeros(n, np.float32),
                         np.zeros(n, np.float32))


class EncodedCapturePolicy(CapturePolicy):
    def launch_encoded(self, enc: EncodedBatch, lo: int, hi: int, n_actions: np.ndarray) -> PolicyOut:
        return self.launch(enc.samples()[lo:hi], n_actions)


def trace_host(vocab: Vocab) -> FakeHost:
    host = FakeHost(vocab)
    host.game = TraceGame(vocab)
    return host


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_actor_stored_sample_learner_and_evaluator_agree(vocab: Vocab, tmp_path: Path,
                                                       encoded: bool, enabled: bool) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=enabled))
    policy = EncodedCapturePolicy() if encoded else CapturePolicy()
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=3))
    jobs = [EpisodeJob((vocab.id("character", "IRONCLAD"),), "IRONCLAD", seed="000000000001") for _ in range(2)]
    host = trace_host(vocab)
    actor = Actor([host], policy, tok, JobList(jobs), ActorConfig(mode="train"), store=store)
    actor.run()
    host.close()
    # the tracker (and with it the re-open counter) runs only when the model reads the history
    assert actor.counters.decisions == 10 and actor.counters.card_offer_reopens == (4 if enabled else 0)
    stored = [s for segment in segs for s in segment.samples]
    assert len(stored) == len(policy.seen) == 10
    for actual, expected in zip(stored, policy.seen):
        assert np.array_equal(actual.history, expected.history)
    if enabled:
        assert not np.array_equal(stored[0].history, stored[2].history)
        assert np.array_equal(stored[0].history, stored[5].history)  # episode reset
    else:
        assert all(not s.history.any() for s in stored)
    # The learner collates the compacted snapshots; it never rebuilds history.
    assert torch.equal(tok.collate(stored)["history"], tok.collate(policy.seen)["history"])

    evaluator = EncodedCapturePolicy() if encoded else CapturePolicy()
    eval_host = trace_host(vocab)
    panel = Panel("history", ("000000000001", "000000000002"), ("IRONCLAD",), (0,))
    result, _ = evaluate(panel, evaluator, tok, [eval_host], tag="trace", policy_info={}, out_dir=tmp_path)
    eval_host.close()
    assert result.completed_episodes == 2
    for training, evaluation in zip(policy.seen, evaluator.seen):
        assert np.array_equal(training.history, evaluation.history)


def test_history_metrics_reach_train_records(vocab: Vocab, tmp_path: Path) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True))
    policy = CapturePolicy()
    host = trace_host(vocab)
    reports: list[dict[str, Any]] = []
    actor = Actor([host], policy, tok,
                  JobList([EpisodeJob((vocab.id("character", "IRONCLAD"),), "IRONCLAD")]),
                  ActorConfig(mode="eval"), on_report=reports.append)
    actor.run()
    host.close()
    metrics = MetricsWriter(tmp_path / "metrics.jsonl")
    episodes = JsonlWriter(tmp_path / "episodes.jsonl")
    errors = JsonlWriter(tmp_path / "errors.jsonl")
    agg = Aggregator(TrainConfig(run_name="history-test"), metrics, episodes, errors, {})
    for report in reports:
        agg.handle("actor", 0, report)
    agg.handle("learner", 0, {"version": 7, "updates": 3, "samples_seen": 5,
                              "sample_counts_by_kind": {"REWARD": 3, "CARD_REWARD": 2},
                              "sample_share_by_kind": {"REWARD": .6, "CARD_REWARD": .4}})
    agg.emit_train(1.0)
    metrics.close()
    episodes.close()
    errors.close()
    row = list(read_jsonl(tmp_path / "metrics.jsonl"))[-1]
    assert row["card_offer_reopens"] == 2 and row["card_offer_reopen_share"] == .4
    assert row["sample_share_by_kind"] == {"REWARD": .6, "CARD_REWARD": .4}
    assert row["sample_counts_by_kind"] == {"REWARD": 3, "CARD_REWARD": 2}
    assert row["sample_kind_version"] == 7


def test_actor_keeps_history_with_environment_after_bucket_sort(vocab: Vocab, gen: FrameGen) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True))
    policy = EncodedCapturePolicy()
    hosts = [trace_host(vocab), trace_host(vocab)]
    actor = Actor(hosts, policy, tok, JobList([]), ActorConfig(mode="eval", min_group=1))
    frames = [parse_frame(random_frame(gen, np.random.default_rng(9), Kind.CRYSTAL, hashes=False)),
              parse_frame(reward_trace(vocab)[0])]
    expected = []
    for env, frame in zip(actor.envs, frames):
        assert env.history is not None
        take(env.history, tok, stamp(frame, step=0))
        current = stamp(frame, step=1)
        expected.append(env.history.observe(current))
        env.frame = current
    actor.ready = list(actor.envs)
    actor._launch()
    batch = actor.inflight[0]
    assert [e.idx for e in batch.envs] == [1, 0]
    for env, sample in zip(batch.envs, batch.samples):
        assert np.array_equal(sample.history, expected[env.idx])
    for host in hosts:
        host.close()


def stamp(frame: Frame, *, step: int, episode: int = 1, floor: int = 1, auto: int = 0) -> Frame:
    data = bytearray(frame.payload)
    for col, value in ((H.STEP_ID, step), (H.EPISODE_ID, episode), (H.AUTO_STEPS, auto)):
        struct.pack_into("<i", data, col * 4, value)
    struct.pack_into("<i", data, GLOBAL_OFFSET + G.TOTAL_FLOOR * 4, floor)
    return parse_frame(bytes(data))


def take(history: PublicHistory, tok: Tokenizer, frame: Frame, action: int = 0) -> tuple[np.ndarray, bool]:
    seen = history.observe(frame)
    sample = tok.encode(frame)
    sample.history = seen
    return seen, history.record(frame, sample, action)


def test_public_history_room_episode_and_auto_boundaries(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True))
    history = PublicHistory(tok.width)
    frames = [stamp(parse_frame(f), step=i) for i, f in enumerate(reward_trace(vocab)[:3])]
    first, _ = take(history, tok, frames[0])
    take(history, tok, frames[1])
    rejected = history.observe(frames[2])
    assert np.array_equal(rejected[-HISTORY_STATS:][15:19], [1, 1, 1, 1])
    assert not first.any()  # observe returned an owned snapshot
    assert not history.observe(stamp(frames[0], step=3, floor=2)).any()
    take(history, tok, stamp(frames[0], step=4, floor=2))
    assert not history.observe(stamp(frames[0], step=0, episode=2, floor=2)).any()
    take(history, tok, stamp(frames[0], step=1, episode=2, floor=2))
    skipped = history.observe(stamp(frames[0], step=2, episode=2, floor=2, auto=3))
    stats = skipped[-HISTORY_STATS:]
    assert stats[0] == Kind.REWARD and stats[2] == 3
    assert stats[2 + Kind.REWARD] == 1  # omitted forced actions are not invented
    history.clear()
    assert not history.observe(stamp(frames[0], step=0, episode=3)).any()


def reward_options(vocab: Vocab, positions: Sequence[int], *, step: int) -> Frame:
    b = FrameBuilder()
    b.episode_id, b.step_id = 1, step
    b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.TOTAL_FLOOR] = Kind.REWARD, 1, 1
    b.glob[G.ROOM_TYPE] = vocab.id("enum:RoomType", "Monster")
    b.add(Table.SEATS, seat=0, hp=80, max_hp=80, alive=1, creature_row=-1)
    for i, pos in enumerate(positions):
        b.add(Table.OPTIONS, kind=OptionKind.REWARD_ITEM, seat=0, pos=pos,
              key_id=vocab.id("enum:RewardType", "Card"))
        b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=i)
    b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=0)
    return parse_frame(b.build(hashes=False))


def test_declined_offers_use_original_position_and_survive_other_decisions(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True))
    history = PublicHistory(tok.width)
    card = parse_frame(reward_trace(vocab)[1])
    assert not take(history, tok, reward_options(vocab, [7, 12], step=0))[1]
    take(history, tok, stamp(card, step=1))
    before_other, reopened = take(history, tok, reward_options(vocab, [7, 12], step=2), 1)
    assert not reopened
    assert np.array_equal(before_other[-HISTORY_STATS:][15:19], [1, 1, 8, 8])
    take(history, tok, stamp(card, step=3))
    both, reopened = take(history, tok, reward_options(vocab, [7, 12], step=4), 0)
    assert reopened and np.array_equal(both[-HISTORY_STATS:][15:19], [2, 2, 13, 13])
    # Taking another reward can remove preceding rows. Position 12 remains the same offer.
    take(history, tok, stamp(card, step=5))
    shifted, reopened = take(history, tok, reward_options(vocab, [12], step=6), 0)
    assert reopened and shifted[-HISTORY_STATS:][16] == 2
    # Explicitly proceeding ends this reward set even if a later set looks identical.
    take(history, tok, reward_options(vocab, [12], step=7), 1)
    fresh = history.observe(reward_options(vocab, [12], step=8))
    assert not fresh[-HISTORY_STATS:][15:19].any()
    assert fresh[-HISTORY_STATS:][2 + Kind.REWARD] == 5


@pytest.mark.parametrize("elided", [(Table.OPTIONS,), (Table.ACTIONS,), (Table.OPTIONS, Table.ACTIONS)])
def test_resolved_elided_reward_history_matches_full(vocab: Vocab, elided: tuple[Table, ...]) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True))
    full_history, elided_history = PublicHistory(tok.width), PublicHistory(tok.width)
    reopen_counts = [0, 0]
    for step, payload in enumerate(reward_trace(vocab)[:5]):
        full = stamp(parse_frame(payload), step=step)
        directory = full.directory.copy()
        data = [full.table(t).ravel() for t in range(N_TABLES)]
        offset = DATA_OFFSET
        for table in range(N_TABLES):
            if table in elided:
                directory[table, 1], directory[table, 3] = -1, 0
                data[table] = np.zeros(0, np.int32)
            else:
                directory[table, 3] = offset
                offset += data[table].nbytes
        header = full.header.copy()
        header[H.FLAGS] |= FrameFlag.ELIDED
        header[H.FRAME_BYTES] = offset
        resolved = parse_frame(np.concatenate([header, full.glob, directory.ravel(), *data]).astype("<i4").tobytes())
        resolved.resolve_elided(full)
        want, first = take(full_history, tok, full)
        actual, second = take(elided_history, tok, resolved)
        assert np.array_equal(actual, want)
        assert first == second
        reopen_counts[0] += first
        reopen_counts[1] += second
    assert reopen_counts == [2, 2]
    assert np.array_equal(full_history.data, elided_history.data)


def test_nested_reward_proceed_restores_declined_parent_offer(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True))
    history = PublicHistory(tok.width)
    card = parse_frame(reward_trace(vocab)[1])
    parent = reward_options(vocab, [7, 12], step=0)
    words = parent.words.copy()
    options = words[parent.offsets[Table.OPTIONS]:].reshape(-1)
    options[parent.table(Table.OPTIONS).shape[1] + OptionCol.KEY_ID] = vocab.id("enum:RewardType", "Relic")
    parent = parse_frame(words.tobytes())
    take(history, tok, parent, 0)
    take(history, tok, stamp(card, step=1))
    take(history, tok, stamp(parent, step=2), 1)  # a relic opens a distinct nested reward set
    take(history, tok, reward_options(vocab, [99], step=3))
    take(history, tok, stamp(card, step=4))
    inner, _ = take(history, tok, reward_options(vocab, [99], step=5), 1)  # PROCEED
    assert np.array_equal(inner[-HISTORY_STATS:][15:19], [1, 1, 100, 100])
    restored, reopened = take(history, tok, reward_options(vocab, [7], step=6))
    assert reopened
    assert np.array_equal(restored[-HISTORY_STATS:][15:19], [1, 1, 8, 8])


@pytest.mark.parametrize("kind,verb,table", [
    (Kind.SHOP, Verb.CHOOSE_OPTION, Table.OPTIONS),
    (Kind.REST, Verb.CHOOSE_OPTION, Table.OPTIONS),
    (Kind.COMBAT, Verb.USE_POTION, Table.POTIONS),
    (Kind.REWARD, Verb.DISCARD_POTION, Table.POTIONS),
])
def test_previous_action_freezes_semantic_source_and_cancel_return(vocab: Vocab, kind: Kind,
                                                                 verb: Verb, table: Table) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True))
    history = PublicHistory(tok.width)
    b = FrameBuilder()
    b.episode_id = 1
    b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.TOTAL_FLOOR] = kind, 1, 1
    b.glob[G.ROOM_TYPE] = vocab.id("enum:RoomType", "Event")
    b.add(Table.SEATS, seat=0, hp=80, max_hp=80, alive=1, creature_row=-1)
    if table == Table.OPTIONS:
        option_kind = OptionKind.SHOP_ENTRY if kind == Kind.SHOP else OptionKind.REST_OPTION
        key = vocab.id("shop_section", "CARD_REMOVAL") if kind == Kind.SHOP else vocab.id("rest_option", "SMITH")
        b.add(Table.OPTIONS, kind=option_kind, seat=0, pos=9, key_id=key, price=75)
    else:
        b.add(Table.POTIONS, seat=0, slot=1, potion_id=2)
    b.add(Table.ACTIONS, verb=verb, seat=0, src_table=table, src_row=0)
    parent = parse_frame(b.build(hashes=False))
    initial, _ = take(history, tok, parent)
    sample = tok.encode(parent)
    action = sum(sample.counts[:3])
    source = int(sample.rel[action, 0])
    b.step_id = 1
    b.glob[G.KIND] = Kind.CARD_SELECT
    b.rows(Table.ACTIONS)[0, ActionCol.VERB] = Verb.CANCEL
    cancel = parse_frame(b.build(hashes=False))
    opened = history.observe(cancel)
    raw = opened[HISTORY_ROWS:-HISTORY_STATS].reshape(HISTORY_ROWS, tok.width)
    assert opened[1] == sample.types[source] and np.array_equal(raw[1], sample.raw[source])
    assert np.array_equal(opened[-HISTORY_STATS:][19:26], parent.actions[0])
    take(history, tok, cancel)
    returned = history.observe(stamp(parent, step=2))
    assert not np.array_equal(initial, returned)
    assert returned[-HISTORY_STATS:][0] == Kind.CARD_SELECT
    assert returned[-HISTORY_STATS:][2 + Kind.CARD_SELECT] == 1
