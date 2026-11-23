"""Real-frame halves of invariants 2 and 6 of notes/next/temperature-spec.md (quiet combat sampling).

    PYTHONPATH=. python -B scripts/temp_check.py --config <run config.json> --checkpoint <learner ckpt> \
        --envs 32 --decisions 12000 --out <result.json>

For T in (1.0, 0.5): a sampling TorchPolicy on the GPU (CUDA graphs, bf16, the run's exploration / choice-uniform
reference) with the checkpoint's online weights drives ``--envs`` real hosts of the run's launch spec for
``--decisions`` decisions (train mode, a RolloutStore that records beh_temp).

* Invariant 6 (predicate): at every decision the actor asserts that the policy's combat flag (from the sample's global
  token) equals the flag of the frame's facts; here every decision frame is also judged by noise_eval's predicate
  (alt/solo/noise/noise_eval.py line 77: IN_COMBAT and kind COMBAT / CARD_SELECT, read from the frame's GLOBAL values)
  and compared with the facts flag, and the stored beh_temp of every step equals T exactly where that predicate holds.
* Invariant 2 (consistency): with the unchanged weights, the learner's log q_new(a) of every stored step - its own
  functions (tempered_log_probs with the stored beh_temp, the exploration table, the choice-uniform reference) on the
  learner's collated batch under its autocast - against the stored log q(a), split combat / non-combat; and one
  learner update (fast path, CUDA graphs, lr = 0, one minibatch = the whole batch) whose first-minibatch KL, ratio
  deviation and clip fraction are the learner's own numbers at unchanged weights. T = 1.0 gives the tolerance the two
  already have; T = 0.5 must be within it.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from spire.actor import Actor, ActorConfig, EpisodeSampler, TorchPolicy
from spire.distributions import batch_choice_keys, choice_uniform_probs, exploration_table
from spire.learner import Learner
from spire.proto import Frame
from spire.rewards import RoomIds, frame_facts
from spire.rollout import RolloutStore, Segment
from spire.schema import ActionCol, G, Kind
from spire.temperature import combat_steps, tempered_log_probs
from spire.tokenize import Ent, Tokenizer
from spire.train import build_net, load_config, load_vocab, make_hosts


def noise_eval_combat(f: Frame) -> bool:
    """alt/solo/noise/noise_eval.py line 77, verbatim: combat = IN_COMBAT and kind COMBAT / CARD_SELECT."""
    g = f.gvals
    kind = Kind(int(g[G.KIND]))
    return bool(g[G.IN_COMBAT]) and kind in (Kind.COMBAT, Kind.CARD_SELECT)


def play(cfg, tok: Tokenizer, vocab, catalog, net: torch.nn.Module, t: float, envs: int, decisions: int, seed: int
         ) -> tuple[list[Segment], dict]:
    a = cfg.actor
    pol = TorchPolicy(net, tok, mode="sample", use_graphs=True, seed=seed,
                      exploration_by_kind=cfg.ppo.exploration_by_kind,
                      exploration_choice_uniform=cfg.ppo.exploration_choice_uniform, combat_temperature=t)
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=cfg.rollout,
                         rewards=dataclasses.replace(cfg.rewards, gamma=cfg.ppo.gae.gamma), beh_temp=True)
    log_dir = Path(cfg.run_root) / cfg.run_name / "host-logs"
    hosts = make_hosts(cfg.hosts, envs, seed=seed, log_dir=log_dir, name=f"tc{int(t * 10)}",
                       expected_hashes=(vocab.hash, catalog.hash))
    stats = {"frames": 0, "noise_combat": 0, "facts_mismatch": 0}

    def on_decision(env_id: int, f: Frame, action: int) -> None:
        fa = frame_facts(f)
        mine = bool(combat_steps(np.asarray([fa.kind]), np.asarray([fa.in_combat]))[0])
        theirs = noise_eval_combat(f)
        stats["frames"] += 1
        stats["noise_combat"] += theirs
        stats["facts_mismatch"] += mine != theirs

    sampler = EpisodeSampler(cfg.episodes, vocab, seed=(seed, 0, 0, 0))
    acfg = ActorConfig(mode="train", max_batch=a.max_batch, max_wait_ms=a.max_wait_ms, min_group=a.min_group,
                       max_inflight=a.max_inflight, reset_flags=a.reset_flags,
                       max_episode_decisions=a.max_episode_decisions, pile_cap=a.pile_cap)
    actor = Actor(hosts, pol, tok, sampler, acfg, store=store, on_decision=on_decision)
    t0 = time.monotonic()
    try:
        actor.run(max_decisions=decisions)
    finally:
        pol.close()
        for h in hosts:
            h.close()
    c = actor.counters
    stats.update(seconds=round(time.monotonic() - t0, 1), decisions=c.decisions, actor_combat_steps=c.combat_steps,
                 combat_nonargmax=c.combat_nonargmax / max(1, c.combat_steps),
                 noncombat_nonargmax=c.noncombat_nonargmax / max(1, c.noncombat_steps))
    return segs, stats


@torch.no_grad()
def per_step(learner: Learner, tok: Tokenizer, segs: list[Segment]) -> dict:
    """|learner log q_new(a) - stored log q(a)| per step with the learner's functions and autocast."""
    samples = [x for s in segs for x in s.samples]
    actions = np.concatenate([s.actions for s in segs])
    logp_b = np.concatenate([s.logp for s in segs])
    kinds = np.concatenate([s.kinds for s in segs]).astype(np.int64)
    temps = np.concatenate([s.beh_temp for s in segs])
    n_legal = np.concatenate([s.n_legal for s in segs])
    combat = np.concatenate([(s.combat_id >= 0) & np.isin(s.kinds, (Kind.COMBAT, Kind.CARD_SELECT)) for s in segs])
    cfg = learner.cfg
    expl = torch.as_tensor(exploration_table(cfg.exploration_by_kind), device=learner.device)
    slot = tok.layout.slot(Ent.ACTION, ActionCol.VERB)
    out = np.empty(len(samples), np.float64)
    learner.model.eval()
    for lo in range(0, len(samples), 256):
        hi = min(len(samples), lo + 256)
        batch = {k: v.to(learner.device) for k, v in tok.collate(samples[lo:hi]).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=learner.autocast):
            logits = learner.model(batch)["logits"].float()
        legal = torch.isfinite(logits)
        ref = choice_uniform_probs(legal, batch_choice_keys(batch, slot)) if cfg.exploration_choice_uniform else None
        k = torch.as_tensor(kinds[lo:hi], device=learner.device)
        lq = tempered_log_probs(logits, torch.as_tensor(temps[lo:hi], device=learner.device), expl[k], ref)
        a = torch.as_tensor(actions[lo:hi], device=learner.device)
        out[lo:hi] = lq.gather(1, a[:, None]).squeeze(1).double().cpu().numpy()
    learner.model.train()
    d = np.abs(out - logp_b)
    pol = n_legal > 1
    res = {}
    for name, m in (("combat", combat & pol), ("noncombat", ~combat & pol)):
        res[name] = {"steps": int(m.sum()), "max_abs": float(d[m].max()), "mean_abs": float(d[m].mean()),
                     "p99_abs": float(np.percentile(d[m], 99)),
                     "kl": float(np.mean(np.expm1(out[m] - logp_b[m]) - (out[m] - logp_b[m])))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--decisions", type=int, default=12000)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    vocab, catalog = load_vocab(cfg)
    tok = Tokenizer(vocab, cfg.tokenizer)
    result: dict = {"config": args.config, "checkpoint": args.checkpoint, "envs": args.envs}
    for i, t in enumerate((1.0, args.temperature)):
        ppo = dataclasses.replace(cfg.ppo, lr=0.0, kl_lr_target=None, target_kl=None, combat_temperature=t,
                                  epochs=1, minibatch=1 << 30)
        torch.manual_seed(0)
        learner = Learner(build_net(cfg, tok, catalog), tok, ppo, device="cuda", fast=True, graphs=True)
        learner.load(args.checkpoint, allow_config_change=True)
        net = copy.deepcopy(learner.model).eval()
        segs, stats = play(cfg, tok, vocab, catalog, net, t, args.envs, args.decisions, 20261180 + i)
        temps = np.concatenate([s.beh_temp for s in segs])
        combat = np.concatenate([(s.combat_id >= 0) & np.isin(s.kinds, (Kind.COMBAT, Kind.CARD_SELECT)) for s in segs])
        assert stats["facts_mismatch"] == 0, stats
        assert (temps[combat] == np.float32(t)).all() and (temps[~combat] == 1).all()
        steps = per_step(learner, tok, segs)
        m = learner.update(segs)
        result[f"T={t}"] = {"actor": stats, "stored_steps": int(len(temps)), "stored_combat_steps": int(combat.sum()),
                            "per_step": steps,
                            "learner": {k: m[k] for k in ("first_minibatch_kl", "approx_kl", "ratio_dev", "clipfrac",
                                                          "minibatches", "beh_temp_mean", "entropy_beh_combat",
                                                          "entropy_base_combat", "fast_path")}}
        print(json.dumps({f"T={t}": result[f"T={t}"]}), flush=True)
        del learner, net
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
