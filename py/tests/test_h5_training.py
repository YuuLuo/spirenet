"""A migrated policy collects mixed seat counts and completes an actual CPU PPO update."""
from __future__ import annotations

from dataclasses import replace

import torch

from spire.actor import Actor, ActorConfig, EpisodeJob, JobList, TorchPolicy, parse_lineup
from spire.learner import Learner, PPOConfig
from spire.metrics import EpisodeRecord
from spire.model import ModelConfig, SpireNet
from spire.rewards import RewardConfig, RoomIds
from spire.rollout import GAEConfig, RolloutConfig, RolloutStore, Segment, build_batch
from spire.seat_migration import migrate_seat_symmetry_weights
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab


def test_migrated_mixed_rollout_completes_ppo_update(vocab: Vocab) -> None:
    torch.manual_seed(47)
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True))
    config = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16)
    old = SpireNet(tok.layout, config)
    model = SpireNet(tok.layout, replace(config, seat_symmetry=True))
    model.load_state_dict(migrate_seat_symmetry_weights(old.state_dict(), model, tok.layout))
    gae = GAEConfig(gamma=1.0, macro_micro=True, stitch_segments=True)
    ppo = PPOConfig(epochs=1, minibatch=64, micro_batch=8, micro_tokens=2048, bf16=False, gae=gae,
                    exploration_by_kind=((1, 0.05), (5, 0.1), (7, 0.1), (8, 0.15)),
                    target_kl=0.03, kl_lr_target=0.01)
    learner = Learner(model, tok, ppo, device="cpu", fast=False,
                      size_fn=lambda sample: tok.config.bucket(sample.n_tokens))
    policy = TorchPolicy(model.eval(), tok, use_graphs=False, seed=47,
                         exploration_by_kind=ppo.exploration_by_kind)
    segments: list[Segment] = []
    records: list[EpisodeRecord] = []
    store = RolloutStore(segments.append, RoomIds.from_vocab(vocab), rewards=RewardConfig(gamma=1.0),
                         config=RolloutConfig(segment_len=8))
    lineups = ("DEFECT", "DEFECT+DEFECT+IRONCLAD+IRONCLAD")
    jobs = [EpisodeJob(parse_lineup(lineup, vocab), lineup, seed="000000000017") for lineup in lineups]
    host = FakeHost(vocab, config=FakeHostConfig(mean_decisions=32, min_decisions=32, max_decisions=32,
                                                length_sigma=0, abort_prob=0, hashes=False))
    try:
        actor = Actor([host], policy, tok, JobList(jobs), ActorConfig(max_batch=2),
                      store=store, on_episode=records.append)
        actor.run(max_seconds=30)
    finally:
        policy.close()
        host.close()
    assert len(records) == 2 and {record.result for record in records} <= {"win", "loss"}
    assert {len(s.meta.characters) for s in segments} == {1, 4}
    assert sum(map(len, segments)) == sum(r.decisions for r in records) == 64
    batch = build_batch(segments, gae, 0)
    assert set(batch.n_seats) == {1, 4}
    value_weight = next(model.value_head.parameters())
    before = value_weight.detach().clone()
    metrics = learner.update(segments)
    assert learner.updates == 1 and learner.samples_seen == 64
    assert metrics["nonfinite_skipped"] == metrics["illegal_chosen"] == 0
    assert metrics["gae_joined_cuts"] > 0
    assert not torch.equal(before, value_weight)
