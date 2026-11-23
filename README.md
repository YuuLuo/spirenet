# SpireNet

[中文](README.zh-CN.md)

SpireNet is a reinforcement-learning agent for *Slay the Spire 2*. It plays Ascension 10, including the
act-3 double boss, from the information a human player can see. The policy is an entity transformer
trained with asynchronous PPO on a headless host that runs the unmodified game code.

This repository contains the host, the complete training and evaluation code, the configurations of the
released training lines and the seed panels used to measure them. Weights are attached to the GitHub
release.

> SpireNet is an independent project. It is not affiliated with or endorsed by Mega Crit. The repository
> contains no game files, game code or game text. Running it requires your own copy of the game.

## Results

Raw network, greedy, no search at play time. Ascension 10, 1,000 fixed seeds per character
(`eval/panels/<CHARACTER>-a10-dev1000.json`), one game per seed.

| Character | Model | Parameters | Training samples | Wins / 1,000 | Win rate (95 % CI) |
|---|---|---:|---:|---:|---|
| Silent | `silent-19m` | 18.2 M | 4.53 B | 854 | **85.4 %** (83.1–87.5) |
| Defect | `defect-4m` | 4.2 M | 16.40 B | 780 | **78.0 %** (75.3–80.5) |
| Ironclad | `ironclad-4m` | 4.2 M | 15.23 B | 741 | **74.1 %** (71.3–76.7) |
| Fool | `fool-19m` | 18.2 M | 3.13 B | 715 | **71.5 %** (68.6–74.2) |

- The win rates were measured on Windows with the code and host in this repository (`spire.habit_eval`,
  see [Evaluate](#evaluate)).
- The 95 % interval is the Wilson interval. A 1,000-game reading has a standard error of about 1.3 points.
  Greedy play is not bit-for-bit repeatable on a GPU (near-ties in bf16), so a rerun of the same weights on
  the same seeds moves by about one point.
- These seed panels were used to monitor training. No game from them was ever used as training data.
- Training samples are decisions the learner consumed for this network (the 19M networks count from their
  distillation start; the 4M networks from scratch).
- Necrobinder and Regent have not been trained to a comparable level and are not released.

## How it works

**Host.** `host/` is SpireHost, a .NET 9 program that loads the game's own `sts2.dll` and drives a run
headless and deterministically. Its Harmony patches replace only the engine around the rules (no
rendering, a virtual clock, an in-memory profile, input injection); none changes game behaviour, and each
one is listed with its purpose in `host/FIDELITY.md`.
For each decision the host sends a compact binary frame of tables (cards, creatures, powers, intents,
relics, potions, map, legal actions, ...). Each training actor runs 60 host processes.

**Information.** The acting policy sees only what the screen shows. Draw-pile order, seeds, RNG state and
unrevealed rooms are never sent. Rule knowledge a player can learn is allowed: the host adds a forecast of
each enemy's next moves computed from the enemy's public move pattern (for random patterns, the set of
possible moves, never the roll) and turn summaries (`FORECAST` / `CALC` tables).

**Model.** Every table row becomes a token: learned id embeddings, small MLPs over static catalog features,
symlog-scaled numbers, and pooling of attached rows (powers onto their owner, edges onto map nodes). A
pre-norm transformer without positional encoding processes the tokens. All layer norms are conditioned on
character, ascension and seat count. A pointer head scores every legal action from its source, target and
global tokens. Value, win probability, floors reached and HP are auxiliary heads on the same trunk.
Presets: `4m` (d 224, 4 layers) and `19m` (d 384, 8 layers).

**Training.** Actors run batched, CUDA-graphed bf16 inference on EMA weights; the learner runs PPO on
one or more GPUs. Features of the released lines:

- dense reward (floor +0.01, elite +0.02, boss +0.1, win +1, death −0.1, HP term), GAE with separate
  λ for combat and out-of-combat decisions;
- exploration mixed into the behaviour policy at strategic decision kinds (map, card reward, shop, rest,
  event), with exact importance accounting in the learner;
- an ascension curriculum (`py/scripts/solo_curriculum.py`) that raises the ascension when the sampled
  win rate at the current one reaches 15 %;
- **kickstart distillation** to grow a trained 4M network into the 19M network;
- **ExIt**: search workers replay positions with re-sampled hidden information (draw order and future
  rolls are re-drawn, never read) and add policy targets where search beats the network by a margin;

More detail, including every hyperparameter and the lineage of each released model, is in
[docs/TRAINING.md](docs/TRAINING.md).

## Repository layout

```
host/                 SpireHost (C#, .NET 9): engine driver, observation encoder, protocol, fork server
host/tools/           prepare_game_dir.py, Publicizer
host/vocab/vocab.json the id space of the released models (names only; catalog.json is generated locally)
py/spire/             the Python package: tokenizer, model, actor, learner, PPO, ExIt, evaluation
py/configs/           host launch specs (hosts/), released run configs (runs/), removal-ratings.json
py/scripts/           curriculum, checkpoint slimming, benchmarks, migrations
py/tests/             unit and integration tests
eval/panels/          the 1,000-seed panels used above
docs/                 training notes
```

## Getting started

### Requirements

- Windows 10/11 x64 or Linux x64 (WSL2 works; the build and evaluation were checked on Ubuntu 22.04 under
  WSL2). Training was done on Linux, evaluation works on both. Search and
  ExIt workers need Linux (the fork server uses `fork(2)`).
- .NET 9 SDK.
- Python 3.11+, PyTorch 2.7+ with CUDA, NumPy. An NVIDIA GPU (one is enough for evaluation).
- *Slay the Spire 2* **v0.113.1** (`sts2.dll` SHA256
  `62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3`). The host refuses any other build.
  If your installation has updated, Steam's depot download can fetch the old build.

### 1–3. Build the host from your game

`host/README.md` ("Build") describes every step and what it checks. From the repository root:

```
# Windows (PowerShell)
python host\tools\prepare_game_dir.py "<install>"
powershell -ExecutionPolicy Bypass -File host\make-publicized.ps1
dotnet build host\SpireHost\SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false
host\out\bin\SpireHost\release\SpireHost.exe --export-vocab host\out\vocab-export --merge host\vocab\vocab.json
copy host\out\vocab-export\catalog.json host\vocab\
dotnet build host\SpireHost\SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false

# Linux
python3 host/tools/prepare_game_dir.py "<install>"
bash host/make-publicized.sh
dotnet build host/SpireHost/SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false
host/out/bin/SpireHost/release/SpireHost --export-vocab host/out/vocab-export --merge host/vocab/vocab.json
cp host/out/vocab-export/catalog.json host/vocab/
dotnet build host/SpireHost/SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false
```

`<install>` is the game directory that contains `SlayTheSpire2.pck` (Steam: Manage → Browse local files).

1. `prepare_game_dir.py` copies the game's managed assemblies and `release_info.json` into `host/game/`
   and assembles the English localization tables from the game's `.pck` into the bundle the host reads.
   It checks the `sts2.dll` and bundle hashes before writing anything.
2. `make-publicized` makes a compile-time reference of `sts2.dll`; it is never loaded at run time.
3. `--export-vocab` writes `catalog.json` (static card, relic and monster data, including card text)
   from your game. It must print vocab hash `0xB824481A`, catalog hash `0x6D9889A2` and an empty
   `merge_report`; the released weights refuse other hashes. The second build copies the catalog next to
   the executable.

The launch specs in `py/configs/` start the build in `host/out/bin/SpireHost/release/`. A rebuild replaces
those files, so for long training runs make an immutable dist (`make-dist`, see `host/README.md`) and point
the launch spec at it.

### 4. Python environment

```
python -m venv .venv
.venv/bin/pip install -r py/requirements.txt     # Windows: .venv\Scripts\pip install -r py\requirements.txt
python py/native/build.py                         # optional C kernels (gcc on Linux, MSVC on Windows)
```

Install the PyTorch build that matches your CUDA version first if pip would pick a CPU build. All
commands run from `py/` (`python -m spire.<module>`); nothing needs to be installed as a package. The C
kernels only make tokenization faster; without them the NumPy path gives identical results.

### 5. Download the weights

Download the `*-ema.pt` files from the release into `checkpoints/` and check them against
`models/SHA256SUMS`. They contain only the EMA weights used for play (`ema`), the model and tokenizer
configuration (`extra`), the training counters and the PPO configuration.

### Evaluate

From `py/` (Windows uses `configs/real_host.toml`, Linux `configs/hosts/linux.toml`):

```
python -B -m spire.habit_eval --checkpoint ../checkpoints/silent-19m-ema.pt \
  --host-config configs/real_host.toml --panel ../eval/panels/SILENT-a10-dev1000.json \
  --habit baseline --commit-select on --removal-rule configs/removal-ratings.json --rules-mode free \
  --envs 16 --gamelog-games 0 --out ../eval/results/silent-19m
```

`summary.json` has the win rate with its interval, floors and failures; `games.jsonl` has one line per
game. 1,000 games take about 15 minutes with 16 environments on a 24-core CPU.

### Train

From `py/`:

```
python -B -m spire.train --config configs/runs/S25-silent-4m-ppo.json
python -B scripts/solo_curriculum.py S25-silent-v2a      # in a second shell: the ascension curriculum
```

- Adjust `actor.devices`, `actor.actors`, `hosts.envs_per_actor`, `learner.device` and
  `learner.replica_device` to your machine. The released configs were run on 4× RTX 5090 with
  ~180 CPU cores and ~300 GB RAM (about 30–70 k samples per second).
- `configs/runs/` holds the final configuration of each released line. They describe how the models were
  trained; most of them continued from earlier checkpoints (lineage in [docs/TRAINING.md](docs/TRAINING.md)).
  `S25-silent-4m-ppo.json` is plain PPO and the simplest starting point; set `episodes.lineup_ascensions`
  to A0 and let the curriculum raise it.
- ExIt (`ppo.exit_coef` > 0) needs teacher workers (`python -m spire.exit_teacher`, Linux) writing into
  `ppo.exit_dir`. With `exit_coef` 0 the learner is plain PPO.
- Run `python -m spire.dashboard --run ../runs/<name>` for an HTML dashboard of a run.

### Tests

```
cd py && python -m pytest -q
```

Tests that need a built host, a generated catalog or the native engine skip when those are absent.

## Not included

- Game files, decompiled code and game text.
- The native fast engine (a Rust re-implementation of the game rules used for faster search); it is
  derived from the game's code and therefore not published. Everything here runs on the original game
  code through SpireHost.
- Full training checkpoints with optimizer state, and the lab notebook of the experiments (code comments
  and host documents sometimes cite its `notes/...` files).

## License

MIT, see [LICENSE](LICENSE). *Slay the Spire 2* and its content belong to Mega Crit.
