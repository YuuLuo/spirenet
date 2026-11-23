# Training notes

[中文](TRAINING.zh-CN.md)

This file describes how the released models were trained: the system, the lineage of each model and
the settings of the final runs. The run configurations themselves are in `py/configs/runs/`.

## System

```
           ┌──────────────── actor process (×16–32) ────────────────┐
 SpireHost │ 50–60 host processes ── frames ──► tokenizer ──► batched│
 (C#)      │      ▲                              GPU inference (EMA)│
           │      └──────────── actions ◄───────────────────────────┘
           └─────────────── segments (256 steps) ──► shared-memory ring
                                                         │
                                         learner (PPO, 1–4 GPUs) ──► weights ──► actors
                                                         ▲
                       ExIt teacher workers (search) ──► label shards
```

- **Host.** One SpireHost process plays one game at a time and answers RESET / STEP requests over stdio
  with binary frames. `actor.reset_flags = 795` selects what a frame contains: `AUTO_SINGLE` (decisions
  with one legal action are taken automatically), `MAP_IN_COMBAT`, `UPGRADE_PREVIEWS`,
  `TARGET_PREVIEWS`, `FORECAST` and `CALC`. The `HIDDEN` flag (draw order) exists only for experiments
  with a privileged critic and is refused unless the host is started with `--training-hidden`; no
  released model used it.
- **Actors.** Each actor process owns 50–60 hosts, tokenizes their frames and runs CUDA-graphed bf16
  inference on the EMA weights in batches. Exploration is mixed into the behaviour policy at strategic
  decision kinds (`ppo.exploration_by_kind`: kind 1 map, 5 card reward, 6 event, 7 shop, 8 rest) and
  the learner corrects for it exactly.
- **Learner.** PPO, one epoch over each 77,824-sample iteration, minibatch 8,192, bf16 autocast,
  `torch.compile` and CUDA graphs. `learner.replica_device` adds data-parallel replicas (one for the 4M
  lines, three for the FOOL 19M line). Samples older than `max_staleness` updates are dropped. Decisions
  where the policy is already confident (`p > confident_tau`) are subsampled to `confident_keep`, except
  for the strategic kinds in `confident_exempt_kinds`.
- **Throughput.** On 4× RTX 5090 (2 learner + 2 actor GPUs, ~180 cores): 55–70 k samples/s for the 4M
  network, ~32 k samples/s for the 19M network. 4 learner + 4 actor GPUs gave ~1.7× for 19M.

## Objective

- Reward per episode: +0.01 per floor, +0.02 per elite, +0.1 per boss, +1 for a win, −0.1 for a death,
  plus an HP term (`hp_coef` 0.1). Discount 0.9998 per decision for the reward, γ = 1 in GAE.
- GAE with λ = 0.99, computed separately for combat and out-of-combat decisions (`gae.macro_micro`) and
  stitched across segment boundaries.
- Auxiliary heads on the shared trunk: win probability (`aux_win`), floors reached and HP lost
  (`aux_floors`, `aux_hp`).
- Entropy bonus with larger coefficients at the strategic kinds (`ent_coef_by_kind`), a KL-driven
  learning-rate brake (`kl_lr_target`) and a hard KL stop (`target_kl`).

## Techniques used by the released lines

- **Ascension curriculum** (`py/scripts/solo_curriculum.py`): every 20 minutes the highest ascension
  with at least 200 recent games and a sampled win rate of at least 15 % becomes a\*; the next games are
  drawn 30 % at a\*+1, 20 % at a\*+2, 20 % at a\*, 10 % at a\*−1, 10 % at A10 and 10 % below. The final
  A10 lines train on A10 only.
- **Kickstart distillation** (`ppo.kickstart`, `spire/kickstart.py`): a trained 4M teacher is distilled
  into a freshly initialised 19M student, first with the student imitating the teacher's policy and value
  while the teacher acts (phase 0), then with the student acting and the teacher term faded out
  (phase 1). This took the SILENT 19M network to the 4M teacher's level in about 0.2 B samples.
- **ExIt** (`spire/exit_teacher.py`, `spire/exit_buffer.py`): teacher workers play their own games, fork
  the game at chosen decisions and evaluate each candidate action over many worlds in which the hidden
  information (draw order, future random rolls) is re-sampled, never read. Where the best candidate beats
  the network's choice by a margin, the candidates and their search scores are written to a label bank.
  The learner adds a cross-entropy term toward a Boltzmann target over those candidates
  (`ppo.exit_coef`, `exit_soft_beta`), weighted by recency and capped once the policy agrees. On paired
  seeds, combat search beat the raw FOOL network by 2.6 points and the SILENT network by about 0;
  out-of-combat search gave no measurable gain and was switched off.

## Lineage of the released models

All lines descend from one network trained from scratch with PPO on all six characters at A0 (one to four
seats), then specialised per character with the ascension curriculum.

| Release | Line | Path |
|---|---|---|
| `silent-19m` (v58071) | SILENT | 4M curriculum line → `S25` 4M PPO at A10 (14.5 B samples; 810 / 1,000 = 81.0 % on a separate sealed panel) → `S34` kickstart into 19M → `S35` PPO + ExIt (4.53 B samples on the 19M network) |
| `fool-19m` (v40096) | FOOL | `S31` 4M (15.3 B samples) → `S37` kickstart into 19M → `S38` PPO + ExIt with a 4-GPU learner (3.13 B samples on the 19M network) |
| `defect-4m` (v231079) | DEFECT | 4M curriculum line → `S39` PPO at A10 (16.4 B samples in total) |
| `ironclad-4m` (v216143) | IRONCLAD | 4M curriculum line → `S36` PPO + ExIt at A10 (15.2 B samples in total) |

## Final settings

| | S35 silent-19m | S38 fool-19m | S39 defect-4m | S36 ironclad-4m | S25 silent-4m |
|---|---|---|---|---|---|
| model | 19m | 19m | 4m | 4m | 4m |
| learning rate | 2.2e-5 | 2.2e-5 | 1.4e-4 | 9e-5 | 7.5e-5 |
| minibatch | 8,192 | 8,192 | 8,192 | 8,192 | 4,096 |
| entropy (base / strategic kinds) | 0.0035 / 0.02 | 0.0035 / 0.02 | 0.006 / 0.03 | 0.004 / 0.015 | 0.01 / 0.03 |
| EMA decay | 0.997 | 0.997 | 0.998 | 0.998 | 0.998 |
| weight decay | 1e-4 | 1e-4 | 0 | 0 | 0 |
| aux win weight | 0.25 | 0.25 | 0.1 | 0.1 | 0.1 |
| ExIt coefficient | 0.05 | 0.04 | – | 0.08 | – |
| actors × hosts | 16 × 60 | 32 × 50 | 24 × 60 | 24 × 60 | 25 × 60 |
| learner GPUs | 2 | 4 | 2 | 2 | 2 |
| host launch spec | linux_lowmem | linux_lowmem | linux | linux | linux |

## Evaluation practice

- Report greedy play of the EMA weights on a fixed seed panel with the Wilson interval. 300 games carry a
  standard error of about 2.3 points and 1,000 games about 1.3 points; differences smaller than that are
  noise.
- Compare two models or two settings on the same seeds and count the games each one wins that the other
  loses (`games.jsonl` has one line per seed).
- Greedy play is not bit-for-bit repeatable on a GPU: bf16 near-ties flip a few percent of games between
  runs of the same weights. Treat a single reading as one sample.
- Never tune on a panel you report as held out. The 1,000-seed panels in `eval/panels/` were used to
  monitor training and are reported as such.
