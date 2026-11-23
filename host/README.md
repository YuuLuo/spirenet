# SpireHost

Clean-room headless C# host that drives the **unmodified** `sts2.dll` v0.113.1 directly, in the game's TestMode, on
one engine thread. One executable contains everything: the run driver, the five production modules (fidelity, combat,
rooms, events, obs), the binary protocol server and every verification tool.

Since phase H5 one process can also play a **co-op run with 2-4 seats** for one centralized controller (section
"Multi-seat runs" below). A single-seat run is unchanged: the same code path, the same frames, the same traces.

The host references only the game's own assemblies; `SpireHost.Core\Boot\GameAssemblies.cs` refuses a fixed list
of third-party assemblies by name. No network IO: the game's telemetry upload and
time-server request are blocked at their entry points and counted as a strict violation.

| Document | Content |
|---|---|
| `..\notes\DESIGN.md`, `..\notes\DECISIONS-H1.md` | design and the lead's rulings |
| **`DRIVER-API.md`** | the contract (version `h5-4p`): execution model, decision model and contexts, extension points, definitions, tests, commands; section 12 = multi-seat runs |
| `SpireHost.Obs\SCHEMA-ERRATA-H5.md`, `..\notes\host-h5-4p.md`, `..\notes\host-h5-design.md` | phase H5: the observation / protocol erratum for 2-4 seats, the report (what is verified how, what is not), the design as built |
| **`FIDELITY.md`** | the ONE ledger: every patch / intervention, why, and its self-check; generated from and checked against the patch registry |
| `..\notes\host-h1-integration.md` | the integration report: rulings, E1 / E2 analysis, bugs found, gate results, cost table, open items, Python follow-ups |
| `..\notes\obs-schema-v1.md`, `..\notes\schema-errata-v1.md` | the observation schema and protocol (frozen) |
| `FIDELITY.<team>.md`, `contract-requests\<team>.md`, `..\notes\host-h1-<team>.md` | history of the six parallel teams (fragments are marked merged; every contract request carries its status) |

## Layout

```
host\
  SpireHost.sln
  Directory.Build.props / .targets   one build mode, one output root (host\out)
  game\                 game binaries (sts2.dll SHA-pinned), pinned localization bundle, release_info.json
                        (made by tools\prepare_game_dir.py from your install; not in the repository)
  publicized\sts2.dll   compile-time-only reference (all members public); regenerate: make-publicized.sh|.ps1
  vocab\                the released vocabulary: vocab.json + catalog.json (copied next to the executable by the build;
                        catalog.json is made by --export-vocab, not in the repository)
  tools\                Publicizer\ (writes publicized\sts2.dll), prepare_game_dir.py (writes game\)
  out\                  ALL build output: out\bin\<Project>\release, out\obj\<Project>\release
  dist\                 immutable deployment directories: dist\<yyyymmdd-HHmmss>-<label>\   (make-dist)
  snapshot\             the stage-0 snapshot (history; nothing references it)
  SpireHost.Core\       Boot, Engine (pump, virtual clock, engine thread), Patches (registry, ledger check, plumbing,
                        scheduling, network guard, fidelity gates, livelock guard, thread audit), signatures
  SpireHost.Driver\     HostSession, RunDriver, RunContext, decision model + contexts, registry, MapHandler,
                        ScenarioTools, Tools\ (fuzz / replay / determinism, driver self-test, FrameAudit)
  SpireHost.Driver.Combat\  SpireHost.Driver.Rooms\  SpireHost.Driver.Events\  SpireHost.Fidelity\  SpireHost.Obs\
  SpireHost.Replay\     module / tool libraries (tests of each module live in its own assembly)
  SpireHost\            THE executable: Program, Commands, ProductionHost (the one production configuration), DistManifest
  make-dist.sh|.ps1     build + dist directory
..\bench\h1-integration\   run_gates.sh, summary.json, tools\, frames\ (sample frames), the integration's results
```

## Build

The repository contains no game file and no game text. Three things are made from your own copy of the game, once:

| Made by | What | Content |
|---|---|---|
| `tools\prepare_game_dir.py` | `host\game\` | the game's managed assemblies (`sts2.dll` and the 16 it needs), `release_info.json`, and `fair-localization-eng.json`: the English localisation tables, assembled from the game's own `res://localization/eng/*.json` inside `SlayTheSpire2.pck` |
| `make-publicized.ps1` / `.sh` | `host\publicized\sts2.dll` | compile-time reference only (every member public); never loaded at run time |
| `SpireHost --export-vocab` | `host\vocab\catalog.json` | static game data the encoder reads (card / relic / monster numbers and texts) |

`host\vocab\vocab.json` (the name -> id table the trained weights use) is part of the repository.

Requirements: Slay the Spire 2 **v0.113.1** - the host refuses any other build
(`SHA256(sts2.dll) = 62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3`); the .NET 9 SDK; Python 3.10
or newer (standard library only). `<install>` below is the game directory that contains `SlayTheSpire2.pck`,
`release_info.json` and `data_sts2_<platform>\` (Steam: right-click the game, Manage, Browse local files).

Windows (PowerShell, from the repository root):

```
python host\tools\prepare_game_dir.py "<install>"
powershell -ExecutionPolicy Bypass -File host\make-publicized.ps1
dotnet build host\SpireHost\SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false
host\out\bin\SpireHost\release\SpireHost.exe --export-vocab host\out\vocab-export --merge host\vocab\vocab.json
copy host\out\vocab-export\catalog.json host\vocab\
powershell -ExecutionPolicy Bypass -File host\make-dist.ps1 -Label mybuild
host\dist\<stamp>-mybuild\SpireHost.exe --verify-dist
```

Linux (bash, from the repository root):

```
python3 host/tools/prepare_game_dir.py "<install>"
bash host/make-publicized.sh
dotnet build host/SpireHost/SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false
host/out/bin/SpireHost/release/SpireHost --export-vocab host/out/vocab-export --merge host/vocab/vocab.json
cp host/out/vocab-export/catalog.json host/vocab/
bash host/make-dist.sh mybuild
host/dist/<stamp>-mybuild/SpireHost --verify-dist
```

What each step checks:

* `prepare_game_dir.py` checks the SHA256 of `sts2.dll` and of the localisation bundle it builds before it writes
  anything, and refuses a non-empty `host\game`. The bundle is pinned in `SpireHost.Core\Boot\Bootstraps.cs`
  (`LocalizationBootstrap.AcceptedBundles`: this v0.113.1 bundle, and the v0.113.0 bundle the host was developed
  with; they differ only in four timeline texts of the table `epochs`).
* `--export-vocab` must print `"vocab_hash_hex": "0xB824481A"`, `"catalog_hash_hex": "0x6D9889A2"` and an empty
  `merge_report`: the exported `vocab.json` is then byte-identical to `host\vocab\vocab.json`, and the catalog is the
  one the released weights were trained with. Other values mean other game files.
* `make-dist` builds again (now with the catalog), checks `registry == ledger` and verifies the new dist; the dist
  must report patch hash `b8da5999699f69bc9ef3f3727d75d453824940b35e221f28f5448989f7e8f821`.

Tested from a fresh clone with the Steam build on Windows 10 and on Ubuntu 22.04 (WSL2, .NET SDK 9.0.321): both
produce the hashes above and a dist that verifies.

Output of the build: `host\out\bin\SpireHost\release\SpireHost.exe` with every module assembly and `vocab\`.
Nothing is written into a project-local `bin\` or `obj\`. The game binaries are referenced with `Private=false`, stay
in `host\game` and are resolved at run time. At startup the host verifies the SHA256 of `sts2.dll` (exit code 3
otherwise) and, after boot, that `sts2` was really loaded from the game directory (never from `host\publicized`).

Do not run tests or training from `host\out`: a rebuild replaces those files. Make a dist.

## Deploy: make a dist

```
bash host/make-dist.sh <label>                       # or:  powershell -File host\make-dist.ps1 -Label <label>
-> host\dist\<yyyymmdd-HHmmss>-<label>\
     SpireHost.exe  SpireHost*.dll  *.pdb  SpireHost.deps.json  SpireHost.runtimeconfig.json
     vocab\vocab.json  vocab\catalog.json
     FIDELITY.md  DRIVER-API.md  README.md
     dist.json
```

`make-dist` builds, checks `registry == ledger`, copies the build output and the three documents, lets the copied
executable write `dist.json` (`--write-dist-manifest`), sets every file read-only and verifies the result
(`--verify-dist`). `dist.json`: label, stamp, DRIVER-API version, SHA256 of `sts2.dll` and of every shipped file,
the production patch list with its hash, shared sites, vocabulary and catalog hashes, the ledger's hash, and how the
executable finds the game.

**A dist directory is never modified.** Every start of an executable that has a `dist.json` next to it verifies the
directory first: a changed, added or removed file, another game assembly, or a live patch set / vocabulary / contract
version that differs from the manifest is a startup error (exit 3). Training and verification run only from dist
directories, so later rebuilds never disturb running processes. A new build is a new dist.

How the executable finds the game from a dist: `--game-dir DIR`, else the environment variable
`SPIREHOST_GAME_DIR`, else the first directory `game` containing `sts2.dll` found by walking up from the executable's
own directory (`host\dist\<stamp>-<label>\` -> `host\game`). The SHA256 pin applies in every case.

## Run

```
<dist>\SpireHost.exe --<command> [options]            top-level commands
<dist>\SpireHost.exe <area> --<command> [options]     area = combat | rooms | events | fidelity | obs | replay
```

stdout carries only the JSON result of the command (or protocol frames); everything else (game log, diagnostics,
anything a dependency prints) goes to stderr. Every command boots the game on a dedicated 256 MB-stack engine thread
and runs on **the same production configuration**: the five modules in order, the production patch set, the
vocabulary loaded from `<exe dir>\vocab` (`host\SpireHost\ProductionHost.cs`; checked at every start). Almost every
command takes `--out FILE` to also write its JSON to a file.

Process-level options, valid with any command: `--game-dir DIR`, `--vocab-dir DIR`, `--game-log off|warn|info|debug`,
`--verbose`, `--patch-report FILE` (after the command: the production patch list + hash, the tool instrumentation the
command added, and the registry == ledger verdict). Test instrument, any command except `--serve`:
`--audit-threads [--audit-out FILE]`.

### Training: the protocol server

```
SpireHost --serve                         frames on stdout, commands on stdin (binary, length-prefixed)
SpireHost --serve --pipe NAME             connects as a client to the named pipe \\.\pipe\NAME (the Python side creates it)
          [--max-decisions N]             per-episode policy-decision cap -> TERMINAL(LOSS, DECISION_CAP); default 100000; RESET word 14 overrides
          [--turn-cap N]                  combat turn cap -> TERMINAL(LOSS, COMBAT_TURN_CAP); default 500
          [--decision-cap N]              livelock bound on decision points; default 400000
          [--verbose]
```

Protocol: `..\notes\obs-schema-v1.md` section 7 as amended by `..\notes\schema-errata-v1.md` section P. The server's
command line is checked against an allow-list **before anything boots**: an option that is not one of those above, the
test-only `--allow-debug-options [--hp N]` (below) or a process-level option (`--game-dir`, `--vocab-dir`, `--game-log`,
`--patch-report`) is refused with exit code 1, so a mistyped or test-only switch cannot silently change what a training
server does. A switch takes no value: `--allow-debug-options false` (or `0` / `off` / `no`) leaves the debug options
off, any other value after a switch is refused. A RESET with `n_seats` outside 1-4, a multi-seat RESET in SEAT view,
an unknown character id, a bad seed word, a flags word with an undefined bit (bits 0-5 are defined; bits 6-7 only for
a RESET with 2-4 seats), or - on a server started without `--allow-debug-options` - a non-zero word 15 is answered
with ERROR 3 and starts no episode.

### Multi-seat runs (2-4 seats, phase H5)

A RESET with `n_seats` 2, 3 or 4, `view_mode` 0 (TEAM) and one character id per seat starts a real multiplayer run of
the game inside this one process: seat 0 is the host player, seats 1..3 are remote players whose inputs the host
injects as the messages their real clients send (`DRIVER-API.md` section 12, `FIDELITY.md` section 11). There is no
network and no second process. The controller is asked ONE decision at a time and sees the union of the four players'
own screens (TEAM view); format and order: `SpireHost.Obs\SCHEMA-ERRATA-H5.md`. In short:

* pending prompts first, then a remote seat's reward screen, then - in combat - one frame with every seat's legal
  actions (`acting_seat` -1), outside combat each seat's room screen in seat order (`acting_seat` = the seat), and the
  team votes (map, shared-event page) last;
* RESET flags bit 6 `ASK_EVERY_SEAT`: off (default) = a team vote is asked once and cast for every seat; on = every
  seat is asked. Bit 7 `OFFER_UNDO_END_TURN`: off (default) = the undo button is never offered. Both bits exist only
  for a multi-seat RESET;
* characters may repeat; ascension is clamped to the profile's multiplayer ascension (10);
* the debug forced HP (RESET word 15 / `--hp`, only with `--allow-debug-options`) applies to every seat.

The run-level tools take the same runs: `--seats N` (seat k of run i plays `characters[(i + k) % n]`) or
`--lineup A+B+C`, with `--team-votes` (default off in the tools), `--undo false` (default on in the tools) and the
verification mode `--wire`. Cost per decision and per run for 2 / 3 / 4 seats: `..\notes\host-h5-4p.md`.

### Run-level tools

| Command | What it does |
|---|---|
| `--fuzz-runs [--runs N \| --seconds T] [--characters A,B] [--ascensions 0,10] [--seed-base S \| --seed S] [--policy-seed P] [--start-index I] [--hp N] [--turn-cap N] [--max-decisions N] [--signatures] [--record-dir DIR]` | N whole runs with the uniform random legal policy. JSON: outcomes, per character x ascension, furthest-floor histogram, aborts grouped by signature (first stack frames, diagnostics, repro tuples), auto-advance counts, strict counters, module counters, the coverage table (how often each DecisionKind and Verb occurred), time split, decisions per second (wall and CPU). Exit 4 if any run aborted. |
| `--determinism [--runs N] [--characters ..] [--ascensions ..] [--hp N] [--controls false]` | each run twice in one process: identical per-decision signature traces and module counters required; two negative controls (another seed, another policy seed) must differ. `trace_hashes` are compared across processes by the gate script. Exit 5 on mismatch or abort. |
| `--replay-run --record FILE --seed S --character C [--ascension A] [--policy-seed P] [--hp N]` | play one random run and write its episode log (action indices, per-decision signatures, vocabulary hash) |
| `--replay-run --log FILE` \| `--seed S --character C --ascension A --actions i,j,k` \| `--selfcheck N [--hp N]` | replay with the scripted policy: every signature, the outcome and the final floor must be reproduced; a log file that does not name its vocabulary hash and game build, or names another one, is refused (the inline `--actions` form is an action list, not a recording: nothing to compare). Exit 5 on mismatch. |
| `--selftest-mp [--cases a,b] [--seats 2,3,4] [--episodes N] [--shard I --shards K] [--wire]` | the multi-seat self-test. Fights: run creation, the end-turn / enemy-turn handshake, the UI legality rules of every offered action, death / revive / team wipe, remote prompts, enemy scaling against the decompiled formulas, determinism, a 1-seat / 4-seat / 1-seat process. Rooms: rest (MEND), treasure, shop (LordsParasol), crystal sphere, bundle, every shared event, per-seat isolation, whole runs. `--cases trace --seed S --lineup A+B --policy-seed P [--hp N] [--team-votes]` replays one fuzz tuple and lists its last decisions (diagnosis) |
| `--selftest-driver` | the 41 contract cases: stalls, livelocks (incl. the two engine bounds: turns / combat work without a decision), strict violations (incl. the network guard and a run save without its boundary), fault attribution, auto-advance, abandon filter, nested decisions, the decision cap for every cap value, the event context of a locked option, replay-log acceptance, profile restore, clean after abort. Not every sub-code has a case: `unknown_screen`, `frame_built_nothing`, `travel_incomplete`, `quiescence_waiter_spin`, `deferred_call_dropped`, `off_thread_continuation` and `late_unobserved_fault` have none |
| `--driver-info` | registry, modules, canonical card-class provider, vocabulary hashes, profile, patch counts, shared patch sites |
| `--boot`, `--list-encounters` | boot report (stage timings, native-call counters, profile) / all encounters with room type and act |
| `--list-patches`, `--check-ledger [--ledger FILE]` | the production patch set: counts by category (0 Behaviour), hash, shared sites, every record; `registry == ledger`. Exit 4 on a Behaviour patch or a mismatch. |
| `--verify-dist` | re-verifies the dist directory the executable runs from (exit 4 for a development build) |
| `--write-dist-manifest --label L --stamp S` | used by make-dist only; refuses to overwrite a manifest |

### Real-client oracle tools

| Command | What it does |
|---|---|
| `--replay-mcr [--file F \| --dir D \| --run-traces \| --runs-v2 [natural\|deep\|natural,deep] \| --runs-mp [corpus\|deep\|SESSION] [--mp-clients 0\|all]] [--files a,b] [--start I --count N] [--stride S --offset K] [--loose-timing] [--stop-at-divergence] [--extended-trace] [--coverage] [--no-records] [--progress]` | re-executes combats recorded in the real client (`..\oracle\replays`, 1,488; `--run-traces`: the 621 map-node combats of `..\oracle\runs`; `--runs-v2`: the 6,372 replay files of `..\oracle\runs-v2`) and compares the game's own full-combat-state checksum after every action, and the collector's public signature and legal-option count before every recorded decision. Exit 5 unless every replay passes. Listed and never counted as replays or passes: a file without any event or checksum (`empty_recordings`), and a fight that an event option starts (`event_fights_not_startable_from_the_replay_file`: the game's replay format does not record the event options, so such a file cannot be re-executed from itself; `--compare-run-traces-v2` compares its recorded states inside the run). `--coverage`: which TestMode gates the replays evaluated (tool-only instrumentation). `--metrics-gate-report`: static report on the recorded saves (no replay). `--runs-mp` (phase H5): the fights of the real 2-4 player sessions (`..\oracle\runs-mp\<session>\c<k>\*.mcr`, `..\notes\oracle-mp.md`; default: the host client's recordings, `--mp-clients all`: every client's). A replay with several players is re-executed like the game's own multi-player replay facility does it (`NMultiplayerTest.RunReplay`): the game's timing instead of the collector's ready-for-input predicate, no sidecar comparison; every recorded checksum covers every player's state. Output block `multi_player`: actions that finished while not Executing (a card play that cancelled itself because another player's action had made it invalid - the real client logs these and goes on, so does the replay: ledger `replay-mp-action-finished-state-as-production`, tool-only) and recorded inputs that arrived after the fight had ended. |
| `--compare-run-traces [--dir D] [--traces a,b \| --start I --count N] [--stride S --offset K] [--no-save] [--max-diffs N] [--dump-host DIR] [--strict-hp-timing] [--no-known-masks]` | plays the 120 golden run traces (format v1) with their scripted policy and compares, at run start and at every room boundary, the public dump and the game's own run save JSON, the choice log and the card reward offers. Exit 5 unless every trace matches to its end. |
| `--compare-run-traces-v2 [--dir D \| --set natural\|deep \| --root R] [--traces a,b \| --start I --count N] [--stride S --offset K] [--no-save] [--max-divergences N] [--max-diffs N] [--dump-host DIR] [--no-records] [--progress]` | the 494 run traces of format v2 (`..\oracle\runs-v2`: 240 natural, 254 with the forced-HP fixture, up to the end of act 3). Per trace one episode on the production configuration (WongoPoints 0 and the fixture exactly as the trace declares it; AutoSingle and AutoGold off so that the host asks wherever the real client decided). At EVERY recorded decision: every UI item recorded as enabled <-> exactly one legal host action, both ways; the public content of the offer; the record's state fields; then the recorded choice is applied through the production handler. Every room dump and the game's own run save; every fight inside the run (recorded actions and prompt answers through the production combat handlers, per action the collector's signature and option count, and - for every fight with a replay file - the real client's recorded per-action full combat states against the host's). Representation differences are named rules with their UI source (counted in `explained_classes`); anything else is a divergence. Exit 5 unless every trace matches to its end with none. `..\notes\host-v2-parity.md` |
| `combat --crosscheck-mcr [--run-traces] [same selection options]` | the combat handler's legal-action enumeration and selection rules against every decision the real client recorded |

They run on the production patch set plus only their own read-only instrumentation (ledger section 2).

### Vocabulary

The released vocabulary is `host\vocab\vocab.json` + `catalog.json` (vocabulary version 3; hashes in `dist.json`,
`--driver-info` and every frame header). It is loaded at boot in every mode; a missing file, a vocabulary of another
game build, a catalog of another vocabulary, or a game name the vocabulary lacks is a startup error.

```
SpireHost --export-vocab NEWDIR --merge host\vocab\vocab.json
```

Append-only: existing ids keep their values, new names are appended, vanished names are marked retired. The command
refuses to write into the directory it loaded; install the result into `host\vocab` by hand, rebuild, make a new
dist, and give the new hashes to the Python side. `--fresh` instead of `--merge` starts a
new, incompatible id space (new schema generations only).

### Module self-tests and tools

All run on the production configuration; each prints one JSON document and exits 4 on failure.

| Command | What it does |
|---|---|
| `combat --selftest [--suite sources,entries,callers,turnend,turnend-control,potions,costs,vetoes,drawperm,allcards] [--seeds N] [--only ID]` | the combat module's suites (reference run: `--seeds 8`). `turnend-control` is the negative control of `turnend`: with the run-save boundary passed through, every case must abort in `ActionQueueSet.PopAction`. |
| `combat --selftest --suite sweep [--hp N] [--shard I --shards N] [--variant V]` | every encounter x every character (natural HP, and `--hp 99999`): 0 rejected / not enqueued / cancelled actions, 0 unrecognised states |
| `rooms --selftest [--characters X,Y\|all] [--group G] [--scenario S] [--policy-seeds N]` | reward, card reward, proceed tree, treasure, rest, shop and potion scenarios (reference run: `--characters all`) |
| `rooms --tree-runs [--runs N] [--hp N] [--ascensions 0,10]` | whole runs that check the terminal proceed tree's post-conditions at every room entry |
| `events --selftest [--events A,B\|all] [--seeds N] [--extra-rounds N] [--hp N] [--skip-cases] [--skip-sweep] [--brief]` | the event cases and the all-events sweep (every event x state variants x seeds until every legal option was chosen) |
| `fidelity --selftest [--runs N]` | every fidelity patch's real-branch check, the assertions' negative tests, the ledger checks, the audit site table, a fuzz (reference run: `--runs 60`) |
| `fidelity --site-table`, `fidelity --scan` | every non-presentation TestMode audit site -> patch / invariant / module / no-op; static scans for unguarded UI singletons and natives |
| `obs --selftest [--runs N] [--quick] [--cases vocab,keys,audit,serve_args,fuzz,scenarios,protocol,special,real_special,enemy_turn,sel_count,leak]` | encoder, protocol and leak tests on the real handlers. `audit`: static IL audit of the encoder AND of the driver's public views against the hidden-information blacklist, with a canary |
| `obs --leak-test [--runs N] [--every K] [--hp N] [--label L] [--policy-seed P] [--seats N]` | the hidden-information perturbation suite at scale (`--label` / `--policy-seed`: another sample of runs; `--hp 0`: natural HP; default forced HP 99999). `--seats 2..4`: the TEAM-view suite on multi-seat scenarios and runs (every seat's piles / RNG / unopened rewards, the vote RNGs, other seats' closed screens; controls on teammates' hands, gold, status, votes) |
| `obs --selftest --cases frames_mp,protocol_mp,leak_team` | opt-in multi-seat cases (not in the default list): every frame of 2 / 3 / 4-seat runs against the format + erratum H5; the protocol with `n_seats` 2-4; the TEAM leak suite |
| `obs --golden DIR [--runs N] [--hp N] [--characters A,B] [--seed-base S] [--seats N \| --lineup A+B] [--team-votes] [--undo false]` | whole runs with every decision encoded, written as frames for the Python checker (+ synthetic special frames) |
| `obs --sample-frames DIR [--per-kind N] [--runs R]` | at least N frames of EVERY DecisionKind (and TERMINAL) from real runs: `<Kind>.bin` + `index.json` |
| `obs --bench-encode [--runs N] [--hp N]` | encoder cost and frame sizes by kind |
| `replay --selftest [--sample N]`, `replay --replay-determinism [--files a,b]` | the oracle tools' own checks (corpus sample, same replay twice, altered event detected, negative control, run-level replay, six v1 traces to their end, ledger, and the regression tests of the tools: a changed end dump, the final-loss fights' decisions, a signature mismatch, `--alter-event` on a non-play, an event fight, two v2 traces with four in-memory alterations) |
| `combat --list-ids`, `combat --list-patches`, `fidelity --list-patches`, `events --list-event-patches` | inventories: the combat catalog ids; the registry entries of one module (the ONE ledger is `FIDELITY.md`; the per-team fragments `FIDELITY.<team>.md` are history) |
| `rooms --trace-run ...`, `fidelity --scan-il --type T --method M` | development aids: one whole run with the rooms probe's trace; the IL of one game method as the static scans see it |

## Test-only switches - the complete list

None of these can be active in a production server by accident: `--serve` accepts only its allow-list, and there is no
environment variable or protocol field that changes the module or patch set.

| Switch | Where it is honoured | Effect |
|---|---|---|
| `--allow-debug-options` | `--serve` | the explicit flag that makes the server honour debug options. Without it: `--hp` is a startup error; a RESET with a non-zero word 15 is answered with ERROR 3 and starts no episode (counted, reported in the server's exit line). |
| `--hp N` | `--fuzz-runs`, `--determinism`, `--replay-run`, the module tools; `--serve` only with `--allow-debug-options` | forced max and current HP (`EpisodeOptions.DebugForcedHp`): lets a random policy get through whole runs |
| `--hp-seats a,b,c` | the run-level tools, `--selftest-mp` | forced HP per seat (`EpisodeOptions.DebugForcedHpPerSeat`; 0 = that seat keeps `--hp` / natural HP): the death, revive and team-wipe tests |
| `--seats N`, `--lineup A+B+C` | `--fuzz-runs`, `--determinism`, `--replay-run`, `obs --golden`, `obs --leak-test` | a run with 2-4 seats. Not a test switch of the server: there the seat count is a RESET word |
| `--team-votes`, `--undo false` | the run-level tools, `obs --golden` | the two multi-seat episode options (`EpisodeOptions.TeamVotes`, `OfferUndoEndTurn`). The tools default to "every seat is asked, UNDO offered" so that both paths are exercised; the server takes them from RESET flags bits 6 / 7 (default: team votes on, UNDO off) |
| `--wire` | the run-level tools, `--selftest-mp` | verification mode (`EpisodeOptions.WireRoundTrip`): every message of a virtual client is serialised and deserialised with the game's own packet code before the host handles it. An episode must be identical with it on and off. Not reachable from the protocol |
| `--selftest-mp` scaling probes | `--selftest-mp` (case `scaling`) | read-only postfixes on the game's multiplayer scaling hooks (ledger: `mp-test-scale-power:*`, `mp-test-scale-block`, tool-only) |
| RESET word 15 (non-zero) | `--serve --allow-debug-options` | forced HP for that one episode (errata P-2 reserves the word as 0; a production client sends 0) |
| `--audit-threads [--audit-out FILE]` | any command except `--serve` | installs the thread audit (ledger: `thread-audit:*`, tool-only, one registry id per audited method); exit 6 unless every audited gameplay call ran on the engine thread |
| `--coverage` | `--replay-mcr` | installs the gate-evaluation counters (ledger: `replay-coverage:*`, tool-only) |
| the replay tool's three interventions | `--replay-mcr`, `replay --replay-determinism`, `replay --selftest`, `combat --crosscheck-mcr`; the first two also `--compare-run-traces-v2` | checksum tracker on, post-action checksum subscription, combat-state sync off - only inside the replay run the tool sets up; the v2 comparison switches the (read-only) tracker on for the episode it replays (ledger section 2) |
| the replay tool's multi-player intervention | `--replay-mcr` on a replay with several players (`--runs-mp`) | `replay-mp-action-finished-state-as-production` (ledger section 2): while such a replay runs, an action that finishes in a state other than Executing is logged, not thrown, as in the real client. Production keeps the throw (fail closed; the monitor `fidelity-monitor-action-finished-bad-state` counts 0 in every fuzz) |
| combat-entry sampling | `--compare-run-traces`, `--compare-run-traces-v2`, `replay --selftest` | a read-only event subscription (ledger: `replay-combat-start-samples`): the tool's dump and the game's own save at `CombatBegan` and after every combat-history entry of a fight's start |
| `--stop-at-first`, `--no-known-masks`, `--strict-sampling`, `--no-mcr-identity`, `--no-explained`, `--no-checksums`, `--wongo seed\|N` | `--compare-run-traces-v2` | negative controls and diagnostics of the comparison: stop a trace at its first divergence; no mask for the profile-dependent `discovered_*` save fields; no combat-entry sampling rule; identify recorded combat actions from combat.json only; apply none of the named rules that excuse a recorded item, a recorded click or the early end of a trace (each place then shows as a divergence; the rules that only count how UI items and steps map to host actions cannot be switched off, `notes\host-v2-parity.md` section 5.5); do not compare the recorded combat states; use the production WongoPoints rule instead of the traces' 0 |
| `EpisodeOptions.DebugWongoPoints`, `ICardSelectHostView.PreferInstance` | code only: set / called by `--compare-run-traces-v2` | the trace profile's WongoPoints (0) instead of the seed-derived value; hand the game the recorded instance of several identical pile cards (the real screen shows every instance; a host option is the class). Not reachable from the protocol or from any other command |
| `--loose-timing` | `--replay-mcr` | injects a recorded play as soon as the side is the player's instead of waiting for the collector's ready-for-input predicate (a weaker replay timing; default strict) |
| `--keep-checksums`, `--extended-trace`, `--expect-trace-hash H`, `--ledger FILE` | `--replay-mcr`; `--replay-run`; `--check-ledger`, `--list-patches`, `replay --selftest` | diagnostics: keep per-replay checksum lists in the output; an expected trace hash for the inline replay form; another ledger file to check against |
| `--skip-run-level`, `--skip-regressions`, `--skip-cases`, `--skip-sweep`, `--quick`, `--cases`, `--suite`, `--only` | the module self-tests | run part of a self-test (never used by a gate except the documented suites) |
| `--exact`, `--sections false`, `--auto`, `--probe`, `--reset-flags N`; `--special false` | `obs --bench-encode`; `obs --golden` | encoder tool options: every value from the game method itself, no section timing, encode auto-advanced decisions too, sample probe timings, another RESET flags word; no synthetic special frames |
| implicit forced HP of test tools | `events --selftest` (9999), `rooms --tree-runs` and `obs --leak-test` (99999) | these tools default to a forced HP so that their scenarios / random runs get far; `--hp 0` turns it off |
| `combat --crosscheck-mcr` probes | that command | four read-only probes (ledger: `combat-test-mcr-*`, tool-only) |
| `--fresh` | `--export-vocab` | a vocabulary with a new id space instead of an append-only merge |
| `--strict-hp-timing` (= `--strict-sampling`), `--no-known-masks` | `--compare-run-traces` | turn off the two documented rules of the trace comparison: the combat-entry sampling rule (a combat-room dump and its save must equal, as a whole, the host at its first decision point or ONE of the states the host went through in that fight's start sequence) and the mask for the profile-dependent `discovered_*` save fields |
| `--alter-event K` | `--replay-mcr` (used by `replay --replay-determinism`) | replaces one recorded card play by an end turn: the comparison must detect it. An event that is not a card play, or an index beyond the recording, is an error (never a pass) |
| `--allow-abandon`, `--auto-single false`, `--auto-gold false` | run-level tools (`--fuzz-runs`, `--determinism`, `--replay-run`) | episode options of the tools: keep actions flagged Abandon (Trial double-down YES) / ask forced single-action frames / ask gold rewards. The server never allows Abandon, always takes gold, and takes AutoSingle from RESET flags bit 0 |
| `--allow-aborts` | `--determinism` | exit 0 although an episode aborted (trace mismatches still fail). Not used by any gate |
| `Scenario` / `ScenarioRunner` / `ScenarioTools` | test code only | debug jumps and state construction after a real run start |
| `Scenario.Exempt` (`ScenarioRunner.Run(..., exempt:)`) | test code only | exemption of ONE named fidelity invariant for one scenario; used only by the rooms scenario `rest:soft-lock-reported` (`rest_options_inert`); counted in `fidelity.scenario_exemptions_used` |
| `SchedulingPatches.RunSaveBoundaryDisabledForTest` | set and reset only by the combat suite `turnend-control` | passes the run-save boundary through (negative control of ruling E2) |
| test class providers | combat suite `drawperm` | a local-key, a reversed and a broken canonical-class provider, each swapped in for one case and restored |
| `SpireHost.Driver.Tools.FrameAudit` | plugged in by the executable for `<area>` commands only | the module test probes run the real encoder's consistency checks on every decision |
| `Decision.SealForTest` | test code only | applies the central filter and canonical sort to a hand-built decision |
| leak-test perturbations, synthetic special frames | `obs --selftest`, `obs --leak-test`, `obs --golden` | write hidden state on purpose on sacrificial episodes |
| fake handlers of the driver self-test | `--selftest-driver` | registered and removed around each case |

Removed at the integration (no longer exist): `SPIREHOST_COMBAT_MODE`, `SPIREHOST_COMBAT_OBS_CLASSES`,
`--with-modules`, `--prototype-fixes`, `--temporary-handlers`, `--strict false`, `--profile`, the events console's
FoulPotion test extension, the H0 combat harness and its `--disable` experiments.

## The gates

```
bash ../bench/h1-integration/run_gates.sh <dist dir> [gate ...]
-> bench\h1-integration\gates\<dist name>\...   every command's JSON, stderr and exit code
-> bench\h1-integration\summary.json            machine-readable verdict per gate
```

Without gate names it runs G0 - G10 and the sample frames; with names (`G1 G5 frames G9`) only those. Every host
process is started at BelowNormal priority; it runs at most `SPIREHOST_MAX_PROCS` host processes at a time (default 4)
and never starts one while less than 14 GB of physical memory is free. Gates:

| Gate | Content |
|---|---|
| G0 | `--verify-dist`, boot report |
| G1 | `--list-patches` (0 Behaviour), `--check-ledger` (id, category, kind, target and registered text of every row), and the same production patch list in serve, fuzz, determinism, replay-run, replay-mcr, compare-run-traces and compare-run-traces-v2 mode (`--patch-report`) |
| G2 | driver self-test, every module's self-test, the combat encounter sweeps (natural and `--hp 99999`), the rooms proceed-tree runs, the fidelity site table |
| G3 | `--replay-mcr` over the corpus and the run-trace combats, `--compare-run-traces`, the combat cross-check on both |
| G4 | fuzz: natural HP 500 runs x 12 cells and `--hp 99999` 200 x 12; coverage table |
| G5 | `--determinism` 96 natural + 24 forced HP, in-process and across two processes; `--replay-run --selfcheck` 48 + 12 |
| G6 | the Python smoke client against `--serve` over stdio and the named pipe (private copy of `py\spire` + `py\scripts`); debug RESET refused without the flag |
| G7 | `--audit-threads`; the E2 tests (turn-end suite + negative control, the rooms repro tuple); the stress verification's hanging run ends as Aborted(Livelock) |
| G8 | cost: decisions per second per process in fuzz and serve mode, encoder cost |
| frames | `obs --sample-frames`: at least 20 frames of every DecisionKind + index |
| G9 | `--compare-run-traces-v2` over the 494 v2 run traces with exact expected numbers (494 / 494 to their end, 54,403 records, 12,813 dumps and saves, 6,371 fights with all 467,912 recorded combat states equal), and its negative controls: without the named rules / the sampling rule / the masks it must fail; copies of traces with one changed value must be reported at that value |
| G10 | `--replay-mcr --runs-v2`: 6,297 / 6,297 replays, 464,640 checksums; the 74 event fights and the 1 empty recording listed |

### Phase H5 results

Numbers and verdicts of the H5 gates: `..\notes\host-h5-4p.md`; the progress log of the phase:
`..\notes\host-h5-progress.md`. The gates are reproduced from the repository root with (never more than 4 host
processes in total; every process at BelowNormal priority):

| Gates | Command | Results |
|---|---|---|
| G0-G10 + frames of phase H1, on this build | `python -B bench\h5\h1gates\tools\gates.py --dist <dist> [gates]` - a copy of the H1 gate runner that writes under `bench\h5\h1gates`. Two expectations differ from H1: the fidelity self-test has 26 checks (+ `mend_local_targeting`), and G1 also runs the two H5 tool commands (`--selftest-mp --cases scaling`, `--replay-mcr --runs-mp`) | `bench\h5\h1gates\summary.json` |
| multi-seat gates | `python -B bench\h5\tools\m4_mp.py --dist <dist> [--max N] [phase ...]`: `serve_ident` and `ident` (single-seat frames and trace hashes byte-identical to `h1-v2parity` and to the live dist; 1-seat episodes between 4-seat episodes in one server), `fuzz_nat` and `fuzz_hp` (4 seats, 2 x 1,000 runs each: every seat asked / team votes), `det` (determinism for 2 / 3 / 4 seats, in-process and across processes), `wire` (wire round trip on / off), `golden4` (multi-seat frame files for the Python side: `bench\h5\golden4`), `mcr_mp` (the fights of the real 2-4 player sessions, `--replay-mcr --runs-mp`) | `bench\h5\m4\summary*.json` |
| serve soak | `python -B bench\h5\tools\serve_mp.py --exe <dist>\SpireHost.exe --vocab-dir <dist>\vocab --seats 4 --flag-cycle --seconds 7200 ...` (the Python client with 1-4 seats per episode; working-set and private-bytes samples; `--frame-hash` for comparisons between builds) | `bench\h5\m4\soak` |

Other tools of the phase: `bench\h5\tools` (`build.sh`, `hp.sh` / `runpar.py`, `snap.sh`, `lp.py`, `cmp_frames.py`,
`gen_ledger.py`). The ledger `FIDELITY.md` is generated by `bench\h5\tools\gen_ledger.py` (the H1 generator plus the
multi-seat rows) from `--list-patches` and the `--patch-report` files of the tool commands.

## Operational notes

* One process = one game instance = one `HostSession`; run more processes for more throughput. About 200-280 MB
  working set per process in whole runs. A multi-seat run is ONE process too (all seats in it); single-seat and
  multi-seat episodes can follow each other in one server process.
* Keep the default runtime settings (Workstation GC, non-concurrent, TieredCompilation + TieredPGO). Do not pin a
  host process to a single logical CPU (the background tier-1 JIT then starves).
* The NullGodot thunk table is generated for Windows x64 and Linux x64 (System V; which struct returns travel through
  a hidden buffer is measured from the runtime at startup and printed to stderr as `null-godot: System V struct returns ...`).
* Linux x64: `bash host/make-dist.sh <label>` builds the same dist layout with the apphost `SpireHost` instead of
  `SpireHost.exe`; start it with `DOTNET_ROOT` set to the .NET 9 runtime (`DOTNET_ROOT=/opt/dotnet <dist>/SpireHost ...`)
  or as `dotnet <dist>/SpireHost.dll ...`. `--pipe NAME` is a Windows named pipe; on Linux use the stdio transport.
  Evidence that it behaves like the Windows build: `..\notes\remote\host-linux.md`.
* Build with `-nodeReuse:false -p:UseSharedCompilation=false` so that no build server keeps files open.
* Episode logs, fuzz summaries and tool outputs carry the vocabulary hash and the production patch hash: results of
  different builds cannot be mixed up silently.

## Exit codes

`0` ok; `1` usage; `2` unexpected host exception; `3` startup error (game directory / SHA pin, vocabulary, dist
verification, configuration check); `4` a self-test or check failed / an episode aborted; `5` a determinism, replay
or oracle comparison did not match; `6` thread audit not clean.
