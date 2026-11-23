"""CPU measurement of the seat migration's first PPO updates on frozen actor data.

The real data comes from scripts/kl_probe.py collect: four full-episode training
groups and one disjoint held-out group. Every comparison uses the same minibatch
permutation and production Learner.update. Fake H5 data is an optional gradient
exercise, not a game-quality measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .actor import Actor, ActorConfig, EpisodeDist, EpisodeJob, EpisodeSampler, TorchPolicy
from .distributions import exploration_table, legal_uniform_log_probs
from .host import real_host_spec
from .learner import Learner, PPOConfig, _Accum, lr_at
from .metrics import EpisodeRecord
from .model import ModelConfig, SpireNet
from .rewards import RoomIds
from .rollout import RolloutStore, Segment, SegmentEnd
from .seat_migration import migrate_seat_symmetry_weights, seat_column_groups
from .synth import FakeHost, FakeHostConfig
from .tokenize import Sample, Tokenizer, TokenizerConfig
from .train import TrainConfig, build
from .vocab import Catalog, Vocab


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parameter_vector(model: SpireNet) -> Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def categorical_kl(before: Tensor, after: Tensor) -> Tensor:
    """Full forward KL per frame, retaining exactly the original legal support."""
    assert before.shape == after.shape
    legal = torch.isfinite(before)
    assert torch.equal(legal, torch.isfinite(after))
    assert bool(legal.any(dim=-1).all())
    # Double precision prevents cancellation when warm changes are very small.
    left = legal_uniform_log_probs(before.double())
    right = legal_uniform_log_probs(after.double())
    return (torch.where(legal, left.exp(), 0.0) * (left - right)).sum(dim=-1)


@torch.inference_mode()
def predictions(model: SpireNet, tok: Tokenizer, samples: Sequence[Sample], micro: int) -> list[dict[str, Tensor]]:
    was_training = model.training
    model.eval()
    try:
        return [{name: value.detach().clone() for name, value in model(tok.collate(samples[i:i + micro])).items()
                 if name in ("logits", "value")}
                for i in range(0, len(samples), micro)]
    finally:
        model.train(was_training)


def prediction_change(before: Sequence[dict[str, Tensor]], after: Sequence[dict[str, Tensor]],
                      kinds: np.ndarray, ppo: PPOConfig) -> dict[str, float | int]:
    assert len(before) == len(after)
    kl, delta, behavior, policy = [], [], [], []
    eps = torch.from_numpy(exploration_table(ppo.exploration_by_kind))
    pos = 0
    for left, right in zip(before, after):
        logits = left["logits"]
        n = len(logits)
        legal = torch.isfinite(logits)
        kl.append(categorical_kl(logits, right["logits"]))
        delta.append((left["value"].double() - right["value"].double()).abs())
        e = eps[torch.from_numpy(kinds[pos:pos + n].astype(np.int64))].double()
        logp = legal_uniform_log_probs(logits.double(), e)
        logq = legal_uniform_log_probs(right["logits"].double(), e)
        behavior.append((torch.where(legal, logp.exp(), 0.0) * (logp - logq)).sum(-1))
        policy.append(legal.sum(-1) > 1)
        pos += n
    assert pos == len(kinds)
    k, dv, b, pm = torch.cat(kl), torch.cat(delta), torch.cat(behavior), torch.cat(policy)
    assert bool(pm.any())
    return {"frames": pos, "policy_frames": int(pm.sum()), "mean_kl": float(k.mean()),
            "policy_frames_mean_kl": float(k[pm].mean()), "behavior_mean_kl": float(b.mean()),
            "mean_abs_value_change": float(dv.mean()), "max_kl": float(k.max())}


def copied_gradient_norms(model: SpireNet, tok: Tokenizer) -> dict[str, float]:
    """Clipped gradient norms, before Adam, on exactly the migrated destinations."""
    ids, cond, proj = model.ids.weight.grad, model.cond_seats.weight.grad, model.in_proj.weight.grad
    assert ids is not None and cond is not None and proj is not None
    norms = {}
    for region, rows in (("p:seat", (2, 3, 4)), ("p:n_seats", (2, 3, 4)), ("p:view_mode", (1,))):
        offset, _ = tok.layout.regions[region]
        norms[region] = float(ids[[offset + row for row in rows]].double().norm())
    norms["cond_seats.weight"] = float(cond[2:5].double().norm())
    for name, cols in seat_column_groups(tok.layout, model.config.d_id).items():
        if name != "intent_missing":
            norms[name] = float(proj[:, list(cols[1:])].double().norm())
    if model.multiseat_intent_missing is not None:
        grad = model.multiseat_intent_missing.grad
        assert grad is not None, "the multiplayer projection must remain connected to the loss"
        norms["multiseat_intent_missing"] = float(grad.double().norm())
    return norms


class SwitchLearner(Learner):
    """Observe the production LR brake without changing its calculation or update."""

    trace: list[dict[str, Any]]
    measured_model: SpireNet
    measured_tok: Tokenizer

    def _minibatch_lr(self, kl: Tensor, acc: _Accum) -> None:
        super()._minibatch_lr(kl, acc)
        self.trace.append({"minibatch": acc.minibatches, "pre_step_kl": float(kl.detach()),
                           "lr_scale": float(self.opt.param_groups[0]["lr"]) / lr_at(self.cfg, self.updates),
                           "copied_gradient_norms": copied_gradient_norms(self.measured_model, self.measured_tok)})


def make_learner(checkpoint: Path, ck: dict[str, Any], tok: Tokenizer, catalog: Catalog,
                 ppo: PPOConfig, variant: str) -> SwitchLearner:
    config: ModelConfig = build(ModelConfig, ck["extra"]["model_config"])
    assert not config.seat_symmetry and config.id_dropout == 0
    net = SpireNet(tok.layout, replace(config, seat_symmetry=variant != "baseline"), catalog)
    learner = SwitchLearner(net, tok, ppo, device="cpu", fast=False, graphs=False,
                            size_fn=lambda sample: sample.n_tokens)
    learner.load(checkpoint, allow_config_change=True, migrate_seat_symmetry=variant != "baseline",
                 reset_seat_symmetry_optimizer=variant == "reset")
    # Saved CUDA kernel selection is irrelevant to Adam's numerical state. All
    # variants retain the saved LR/betas/epsilon/weight decay but use CPU kernels.
    for group in learner.opt.param_groups:
        group["fused"] = False
        group["capturable"] = False
    learner.trace = []
    learner.measured_model, learner.measured_tok = net, tok
    learner.rng = np.random.default_rng(83141)
    torch.manual_seed(83141)
    return learner


def run_variant(args: argparse.Namespace, ck: dict[str, Any], tok: Tokenizer, catalog: Catalog,
                cfg: TrainConfig, groups: list[list[Segment]], heldout: list[Segment],
                variant: str, mode: str, anchor: list[dict[str, Tensor]],
                ema_anchor: list[dict[str, Tensor]]) -> dict[str, Any]:
    ppo = replace(cfg.ppo, micro_batch=args.micro, micro_tokens=None, minibatch=round(sum(map(len, groups[0])) / 16))
    learner = make_learner(args.checkpoint, ck, tok, catalog, ppo, variant)
    net = learner.measured_model
    samples = [sample for segment in heldout for sample in segment.samples]
    kinds = np.concatenate([segment.kinds for segment in heldout])
    before = predictions(net, tok, samples, args.micro)
    assert len(before) == len(anchor)
    for left, right in zip(anchor, before):
        for key in ("logits", "value"):
            assert torch.equal(left[key].view(torch.int32), right[key].view(torch.int32)), \
                f"{mode}/{variant}: single-player switch outputs differ"
    initial = parameter_vector(net)
    first_step: list[float] = []
    first_effect: list[dict[str, float | int]] = []

    def after_step(optimizer: torch.optim.Optimizer, hook_args: tuple[Any, ...], hook_kwargs: dict[str, Any]) -> None:
        if not first_step:
            first_step.append(float((parameter_vector(net) - initial).double().norm()))
            first_effect.append(prediction_change(before, predictions(net, tok, samples, args.micro), kinds, cfg.ppo))

    hook = learner.opt.register_step_post_hook(after_step)
    report: dict[str, Any] = {"variant": variant, "mode": mode, "switch_outputs_bit_identical": True,
                              "parameters": initial.numel(), "iterations": []}
    t0 = time.perf_counter()
    try:
        for i, segments in enumerate(groups):
            # Full episodes make group sizes unequal. Keep production's 16
            # minibatches per iteration, scaling only the samples/minibatch.
            learner.cfg = replace(ppo, minibatch=round(sum(map(len, segments)) / 16))
            learner.trace = []
            start = parameter_vector(net)
            metrics = learner.update(segments)
            assert metrics["nonfinite_skipped"] == metrics["illegal_chosen"] == 0
            assert first_step
            item: dict[str, Any] = {"iteration": i + 1, "samples": sum(map(len, segments)),
                                    "configured_minibatch": learner.cfg.minibatch,
                                    "first_optimizer_step_norm": first_step[0],
                                    "first_optimizer_step_heldout": first_effect[0],
                                    "iteration_parameter_step_norm": float((parameter_vector(net) - start).double().norm()),
                                    "parameter_displacement_from_switch": float((parameter_vector(net) - initial).double().norm()),
                                    "first_minibatch_kl": metrics["first_minibatch_kl"],
                                    "kl_lr_scale_min": metrics["kl_lr_scale_min"],
                                    "kl_lr_scale_mean": metrics["kl_lr_scale_mean"],
                                    "early_stop": metrics["early_stop"], "epochs_done": metrics["epochs_done"],
                                    "optimizer_steps": len(learner.trace), "trace": learner.trace,
                                    "learner_seconds": metrics["time_train"]}
            if i in (0, len(groups) - 1):
                item["heldout"] = prediction_change(before, predictions(net, tok, samples, args.micro), kinds, cfg.ppo)
                assert isinstance(learner.ema, SpireNet)
                item["heldout_ema"] = prediction_change(
                    ema_anchor, predictions(learner.ema, tok, samples, args.micro), kinds, cfg.ppo)
            report["iterations"].append(item)
            report["elapsed_seconds"] = time.perf_counter() - t0
            write_json(args.output / f"{mode}-{variant}.json", report)
            print(json.dumps({"mode": mode, "variant": variant, **{k: v for k, v in item.items() if k != "trace"}}), flush=True)
    finally:
        hook.remove()
    return report


class FakeBudgetSource:
    def __init__(self, sampler: EpisodeSampler, groups: list[list[Segment]]) -> None:
        self.sampler, self.groups = sampler, groups

    @property
    def exhausted(self) -> bool:
        return len(self.groups) == 4

    def next_job(self, env_id: int) -> EpisodeJob | None:
        return None if self.exhausted else self.sampler.next_job(env_id)


def collect_fake(args: argparse.Namespace, ck: dict[str, Any], tok: Tokenizer, catalog: Catalog,
                 cfg: TrainConfig, real_groups: list[list[Segment]], ppo_overrides: dict[str, Any]) -> None:
    assert not args.fake_data.exists(), "fake collection output already exists"
    model_config: ModelConfig = build(ModelConfig, ck["extra"]["model_config"])
    net = SpireNet(tok.layout, replace(model_config, seat_symmetry=True), catalog)
    net.load_state_dict(migrate_seat_symmetry_weights(ck["model"], net, tok.layout), strict=True)
    policy = TorchPolicy(net, tok, use_graphs=False, seed=args.seed,
                         version=int(ck["counters"]["version"]), exploration_by_kind=cfg.ppo.exploration_by_kind)
    segments: list[Segment] = []
    groups: list[list[Segment]] = []
    records: list[dict[str, Any]] = []
    group_start = 0
    targets = [round(sum(map(len, group)) / 3) for group in real_groups]
    dist = EpisodeDist(lineups=(("DEFECT+DEFECT+IRONCLAD+IRONCLAD", 1.0),), ascensions=((0, 1.0),))
    source = FakeBudgetSource(EpisodeSampler(dist, tok.vocab, seed=args.seed), groups)
    store = RolloutStore(segments.append, RoomIds.from_vocab(tok.vocab), config=cfg.rollout,
                         rewards=replace(cfg.rewards, gamma=cfg.ppo.gae.gamma))

    def on_episode(record: EpisodeRecord) -> None:
        nonlocal group_start
        assert record.result in ("win", "loss")
        records.append(record.as_dict())
        pending = segments[group_start:]
        assert pending and pending[-1].end == SegmentEnd.TERMINAL
        if sum(map(len, pending)) >= targets[len(groups)]:
            groups.append(pending)
            group_start = len(segments)

    def on_error(record: dict[str, Any]) -> None:
        raise AssertionError(record)

    host = FakeHost(tok.vocab, args.seed,
                    FakeHostConfig(mean_decisions=96, min_decisions=96, max_decisions=96, length_sigma=0,
                                   abort_prob=0, hashes=False, frame_pool=0), catalog_hash=catalog.hash)
    try:
        actor = Actor([host], policy, tok, source,
                      ActorConfig(max_batch=1, min_group=1, max_wait_ms=0, env_id_base=100,
                                  reset_flags=cfg.actor.reset_flags),
                      store=store, on_episode=on_episode, on_error=on_error)
        counters = actor.run()
        assert counters.protocol_errors == counters.hosts_retired == 0
        assert store.stats.steps_dropped == 0 and len(groups) == 4
        args.fake_data.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"groups": groups, "episodes": records, "seed": args.seed}, args.fake_data)
        write_json(args.output / "fake-collection.json", {"batch_samples": [sum(map(len, g)) for g in groups],
                   "episodes": records, "actor_counters": counters.as_dict(), "rollout_stats": store.stats.as_dict(),
                   "checkpoint_ppo_overrides": ppo_overrides,
                   "description": "in-process production Actor on H5 FakeHost schedule; no game-quality meaning"})
    finally:
        policy.close()
        host.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host-config", type=Path, default=Path("configs/real_host.toml"))
    parser.add_argument("--data", type=Path, required=True, help="data.pt from kl_probe.py collect")
    parser.add_argument("--collection-json", type=Path, required=True)
    parser.add_argument("--heldout-data", type=Path, help="extra independent collection; all five groups join heldout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "mixed"), default="single")
    parser.add_argument("--variants", nargs="+", choices=("baseline", "preserve", "reset"))
    parser.add_argument("--micro", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6271916)
    parser.add_argument("--prepare-fake", action="store_true")
    parser.add_argument("--fake-data", type=Path)
    args = parser.parse_args(argv)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg: TrainConfig = build(TrainConfig, json.loads(args.config.read_text(encoding="utf-8")))
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # Match train.learner_main's effective config before making the explicitly
    # recorded CPU batch-size changes below.
    source_ppo = replace(cfg.ppo, micro_tokens=cfg.learner.micro_tokens) if cfg.learner.micro_tokens else cfg.ppo
    saved_ppo: PPOConfig = build(PPOConfig, ck["config"])
    # The checkpoint predates the active KL brake. Apply the supplied run's brake
    # explicitly; every other source training option must agree exactly.
    assert source_ppo == replace(saved_ppo, kl_lr_target=source_ppo.kl_lr_target), \
        "live config and source checkpoint PPO differ beyond the explicit KL-brake override"
    ppo_overrides = {"kl_lr_target": {"checkpoint": saved_ppo.kl_lr_target, "study": source_ppo.kl_lr_target}}
    print(json.dumps({"stage": "configuration", "checkpoint_ppo_overrides": ppo_overrides}), flush=True)
    cfg = replace(cfg, ppo=source_ppo)
    spec = real_host_spec(args.host_config)
    vocab = Vocab.load(spec.vocab)
    catalog = Catalog.load(spec.catalog, vocab)
    assert vocab.hash == ck["extra"]["vocab_hash"] and catalog.hash == ck["extra"]["catalog_hash"]
    tok = Tokenizer(vocab, build(TokenizerConfig, ck["extra"]["tokenizer"]))
    collection = json.loads(args.collection_json.read_text(encoding="utf-8"))
    assert collection["checkpoint_sha256"] == hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    # This is the trusted artifact just written by this workspace's Actor probe;
    # the dataclass/SampleBlock payload intentionally requires pickle loading.
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    groups: list[list[Segment]] = data["groups"]
    assert len(groups) == 5
    assert all(len(segment.meta.characters) == 1 for group in groups for segment in group)
    episodes = [{(s.meta.env_id, s.meta.episode_uid) for s in group} for group in groups]
    assert sum(map(len, episodes)) == len(set.union(*episodes)), "train/heldout episodes overlap"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_fake:
        assert args.fake_data is not None
        collect_fake(args, ck, tok, catalog, cfg, groups[:4], ppo_overrides)
        return 0
    heldout = list(groups[4])
    heldout_extra = None
    if args.heldout_data is not None:
        heldout_extra = torch.load(args.heldout_data, map_location="cpu", weights_only=False)
        assert {record["seed"] for record in data["episodes"]}.isdisjoint(
            record["seed"] for record in heldout_extra["episodes"]), "train/heldout seeds overlap"
        heldout.extend(segment for group in heldout_extra["groups"] for segment in group)
    assert all(len(segment.meta.characters) == 1 for segment in heldout)
    assert all(bool((segment.policy_version == ck["counters"]["version"]).all()) for segment in heldout)
    train = groups[:4]
    if args.mode == "mixed":
        assert args.fake_data is not None
        fake = torch.load(args.fake_data, map_location="cpu", weights_only=False)["groups"]
        assert len(fake) == 4
        assert all(len(s.meta.characters) == 4 and s.meta.env_id == 100 for g in fake for s in g)
        train = [real + four for real, four in zip(train, fake)]
    variants = args.variants if args.variants is not None else (
        ["baseline", "preserve", "reset"] if args.mode == "single" else ["preserve", "reset"])
    assert args.mode != "mixed" or "baseline" not in variants, "mixed state comparison uses identical migrated weights"
    base = SpireNet(tok.layout, build(ModelConfig, ck["extra"]["model_config"]), catalog)
    base.load_state_dict(ck["model"], strict=True)
    heldout_samples = [sample for segment in heldout for sample in segment.samples]
    anchor = predictions(base, tok, heldout_samples, args.micro)
    base.load_state_dict(ck["ema"], strict=True)
    ema_anchor = predictions(base, tok, heldout_samples, args.micro)
    del base
    report: dict[str, Any] = {"checkpoint_sha256": collection["checkpoint_sha256"],
                              "collection": collection, "mode": args.mode, "device": "cpu", "threads": 4,
                              "micro_batch": args.micro, "source_ppo": asdict(cfg.ppo),
                              "checkpoint_ppo_overrides": ppo_overrides,
                              "heldout_frames": len(heldout_samples),
                              "heldout_extra_episodes": None if heldout_extra is None else heldout_extra["episodes"],
                              "minibatches_per_iteration": 16,
                              "note": "four disjoint frozen rollout groups; actor lag rises from 0 to 3 updates; fp32 CPU",
                              "variants": {}}
    for variant in variants:
        report["variants"][variant] = run_variant(args, ck, tok, catalog, cfg, train, heldout, variant,
                                                   args.mode, anchor, ema_anchor)
        write_json(args.output / f"{args.mode}-summary.json", report)
    return 0
