# SpireHost driver contract (H1 + H5)

Version: **`h5-4p`** (`SpireHost.Driver.DriverApi.Version`; recorded in every dist.json). This is the contract
of the INTEGRATED host: one executable, one production module set, one patch set. Phase H5 added runs with 2-4 seats
in one process (section 12); a single-seat run behaves exactly as in `h1-v2parity`. It supersedes the stage-0 text that
the six teams built against (frozen in `host\snapshot\current`, label `h1-stage0`; kept on disk, no longer referenced
by any build).
Audience: whoever changes a module, the Python side (section 13), verifiers and reviewers.
Binding upstream documents, in this order: `notes\DECISIONS-H1.md`, `notes\driver-spec-rooms.md` /
`notes\driver-spec-combat.md`, `notes\DESIGN.md` (section 0 hard rules apply to everything here). The rulings of the
H1 integration and how each was implemented: `notes\host-h1-integration.md`.
Ground truth for game APIs: `ref\sts2-v0113-src` (cited as `Dir/File.cs:line`, directory prefix
`MegaCrit.Sts2.Core.` omitted).

Contents

1. What exists
2. Solution layout, how to build, how to deploy
3. Execution model and threading rules
4. Types: session, episode, policy
5. Definitions: quiescent, decision point, stall, livelock, strict mode, what an abort reports
6. The decision model: kinds, verbs, references, contexts, canonical order, auto-advance
7. Writing a module: handlers, pending frames, nested decisions, tracked operations, waiting for the engine, checks, counters
8. What is forbidden
9. Testing a handler in isolation
10. Conventions: logging, faults, patches, FIDELITY.md
11. The temporary pieces (removed)
12. Multi-seat runs (phase H5); what is still open
13. Commands and exit codes
14. Changing this contract

---

## 1. What exists

| Piece | Where | State |
|---|---|---|
| Boot, NullGodot, engine pump + virtual clock, plumbing / scheduling patches, network guard, `PatchRegistry` (one patch per site), `PatchLedger` (registry == ledger), `GatePatch`, veteran profile, save store, signatures | `SpireHost.Core` | done |
| `HostSession` (engine thread, strict pump), `RunDriver` (run start chain, settle / classify / ask / apply loop, decision stack, tracked operations, auto-advance, strict checks, episode checks, module counters, teardown), decision model and contexts, policies, registry, MAP decision, `--fuzz-runs` / `--replay-run` / `--determinism` / `--selftest-driver` / `--driver-info` | `SpireHost.Driver` | done |
| COMBAT decisions, CARD_SELECT through the 15 `CardSelectCmd.From*` captures | `SpireHost.Driver.Combat` (module `combat`) | done (single seat) |
| REWARD / CARD_REWARD / TREASURE / REST / SHOP, the out-of-combat potion frame extension | `SpireHost.Driver.Rooms` (module `rooms`) | done (single seat) |
| EVENT, FakeMerchant, CRYSTAL, CONFIRM, BUNDLE | `SpireHost.Driver.Events` (module `events`) | done (single seat) |
| TestMode fidelity patch pack, per-run invariants, the quiescent-point assertion backstop, the audit site table | `SpireHost.Fidelity` (module `fidelity`) | done |
| Observation encoder (obs-schema-v1 + errata), canonical card-class provider, binary protocol server, vocabulary export | `SpireHost.Obs` (module `obs`) | done |
| Real-client oracle tools: `.mcr` combat replays, run-trace comparison | `SpireHost.Replay` (tools, not a module) | done |
| THE executable: every command, the production configuration (`ProductionHost.cs`), dist manifest | `SpireHost` | done |

Production module set, in registration order: **fidelity, combat, rooms, events, obs** - in every mode. There are no
temporary handlers and no fallback: a room screen or prompt without a registered owner aborts the episode, and the
executable refuses to start unless every room screen has a handler (`ProductionHost.Verify`).

`SpireHost --selftest-driver` runs 41 cases that demonstrate the definitions of section 5 in code (stalls, every
livelock bound incl. the two engine bounds, strict violations including the network guard and a run save without its
boundary, fault attribution, auto-advance, the abandon filter, nested decisions and the decision cap at a nested
decision, frame extensions, `Quiescence()`, the event context of a locked option, replay-log acceptance, profile
restore, "clean after abort"). Not every sub-code has a case: `unknown_screen`, `frame_built_nothing`,
`travel_incomplete`, `quiescence_waiter_spin`, `deferred_call_dropped`, `off_thread_continuation` and
`late_unobserved_fault` have none. Its fakes live inside the test (`SpireHost.Driver\Tools\DriverSelfTest.cs`); read it
next to this document: it is the executable form of the contract.

---

## 2. Solution layout, how to build, how to deploy

```
host\
  SpireHost.sln                    everything (IDE convenience; the build command is below)
  Directory.Build.props/.targets   shared settings: ONE build mode, ONE output root (host\out)
  game\                            unmodified game binaries (sts2.dll is SHA-pinned at every start)
  publicized\sts2.dll              COMPILE-TIME reference only: same assembly, every member public
  shared\IgnoresAccessChecksTo.cs  linked into assemblies that compile against the publicized reference
  tools\Publicizer\                the tool that writes publicized\sts2.dll (no NuGet, no Cecil)
  make-publicized.sh|.ps1          regenerates publicized\sts2.dll (only when the game version changes)
  make-dist.sh|.ps1                build + immutable deployment directory host\dist\<stamp>-<label>\ (2.4)
  vocab\vocab.json, catalog.json   the released vocabulary (copied next to the executable by the build)
  out\                             ALL build output (bin\<Project>\release, obj\<Project>\release); never committed to anything
  dist\<yyyymmdd-HHmmss>-<label>\  immutable deployments; training and verification run only from here
  snapshot\                        the stage-0 snapshot of Core/Driver (history; no build references it)
  SpireHost.Core\                  library   (compiled against the REAL sts2.dll)
  SpireHost.Driver\                library   (compiled against the publicized reference)
  SpireHost.Driver.Combat\  SpireHost.Driver.Rooms\  SpireHost.Driver.Events\
  SpireHost.Fidelity\  SpireHost.Obs\  SpireHost.Replay\      libraries (publicized reference)
  SpireHost\                       THE executable (plain reference): Program, Commands, ProductionHost, DistManifest
  README.md  FIDELITY.md  DRIVER-API.md  FIDELITY.<team>.md (history)  contract-requests\ (history + status)
```

### 2.1 Building

```
cd host                  (once before: host\game, publicized\sts2.dll, vocab\catalog.json - README.md "Build")
dotnet build SpireHost\SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false
-> host\out\bin\SpireHost\release\SpireHost.exe   (+ every module assembly, vocab\vocab.json, vocab\catalog.json)
```

Every project references the live sibling projects (`<SpireHostRef Include="Core" />` items, turned into
`ProjectReference` by `Directory.Build.targets`). There is no Snapshot switch any more. All output goes to `host\out`
through the SDK artifacts layout (`ArtifactsPath`); project-local `bin\` / `obj\` directories are never written and
are excluded from the Compile globs. Property `SpireSts2Ref=Publicized|Plain` selects the compile-time reference
(default `Publicized`; Core and the executable use `Plain`).

### 2.2 The publicized reference

`host\publicized\sts2.dll` is `host\game\sts2.dll` with only the accessibility bits of the TypeDef / Field /
MethodDef metadata rows rewritten to public (IL, names, MVID untouched; backing fields of field-like events
keep their accessibility so that `x.SomeEvent += ...` stays unambiguous). It is handed to the compiler
only (`Private=false`); at run time the SHA-pinned real `sts2.dll` is loaded, and `GameBoot` refuses to
continue if `sts2` was loaded from anywhere else. A module assembly carries
`[assembly: IgnoresAccessChecksTo("sts2")]`, which the CoreCLR honours: private members become plain
`ldfld` / `call` - no reflection on hot paths.

Rules: (1) reading private state is fine for *driving* and for the *encoder's white-listed public
information*; it never makes hidden information fair game (section 8). (2) Do not call private game methods
to "do the game's job": the driver uses the same entry points as the real UI. (3) One hazard: with every
overload visible, a call may bind to a formerly private overload. If you call an overloaded game method,
check which overload you got. (This is why `SpireHost.Core` compiles against the plain reference.)

### 2.3 One production configuration

`host\SpireHost\ProductionHost.cs` is the only place that builds `HostOptions`: the five modules in order, strict
mode, the vocabulary directory. Every command of the executable creates its session from it. There is no
command-line switch, environment variable or protocol field that changes the module or patch set (ruling A4). After
the session is created the executable checks (`ProductionHost.Verify`): the five modules, a handler for every room
screen, the canonical card-class provider, a loaded vocabulary, zero Behaviour patches, strict mode. A failure is a
startup error (exit 3).

The vocabulary is loaded at boot in EVERY mode (`ObsModule.Register`): `<exe dir>\vocab` or `--vocab-dir`. Missing
files, a vocabulary exported from another game build, a catalog of another vocabulary, or a name of the running game
that the vocabulary lacks: startup error. The one exception is `--export-vocab`, the command that adds missing names
(append-only merge; it refuses to write into the directory it loaded).

### 2.4 Deployment: dist directories

```
bash host\make-dist.sh <label>          (or: powershell -File host\make-dist.ps1 -Label <label>)
-> host\dist\<yyyymmdd-HHmmss>-<label>\  SpireHost.exe + assemblies, vocab\, FIDELITY.md, DRIVER-API.md, README.md, dist.json
```

The script builds, copies the output and the three documents, and lets the copied executable describe itself
(`SpireHost --write-dist-manifest --label L --stamp S`): dist.json holds the label and stamp, the SHA256 of `sts2.dll`
and of every shipped file, the production patch list with its hash, the vocabulary and catalog hashes, the
DRIVER-API version, and how the executable finds the game. A manifest is refused if the shipped ledger does not match
the patch registry. Files are then set read-only.

From then on every start of that executable verifies its directory before running a command: any changed, added or
removed file, another `sts2.dll`, or a live patch set / vocabulary / contract version different from the manifest is a
startup error. A dist directory is never modified; a new build is a new dist. The game is not part of a dist: the
executable finds it with `--game-dir`, else `SPIREHOST_GAME_DIR`, else the first `game\sts2.dll` found walking up
from its own directory (`host\dist\<stamp>-<label>\` -> `host\game`).

---

## 3. Execution model and threading rules

(DECISIONS-H1 section A.)

* **One process = one game instance = one `HostSession`.** The game's managers are process-wide singletons.
* **One engine thread.** `HostSession.Run(options, body)` creates a dedicated thread with a 256 MB stack,
  boots the game on it, installs `EnginePump` as its `SynchronizationContext` and runs `body` there.
  Everything that touches the game runs on that thread: game code, handlers, pending frames, lifecycles,
  **policies** and observers. The calling thread only waits.
* **The pump.** The game never uses `ConfigureAwait(false)`, so every `await` captures the pump; a
  continuation that cannot run inline is queued (FIFO) and executed by the driver core when it settles.
  Almost everything runs inline: per decision about 0.3 queued continuations in combat.
* **Blocking decisions.** A policy's `Decide` is called on the engine thread and may block (read a pipe,
  run a network). While it blocks nothing else can move. There are exactly two ways a decision is asked:
  1. **Pump-driven** (everything asynchronous: room screens, combat, `ICardSelector.GetSelectedCards`,
     `RewardsSet.testSelector` ...): the request is parked - a room handler builds it on demand, or a
     callback pushes a *pending frame* on the decision stack and returns an incomplete task. The core first
     **drains the pump to quiescence**, then asks, then applies. The observation is therefore always a
     settled state.
  2. **Nested** (synchronous game callbacks such as `ICardSelector.GetSelectedCardReward`): the callback
     calls `RunContext.DecideNested(decision)`; the policy is asked **without re-entering the pump** and
     the callback returns the answer to the game. (Measured: the pump queue was empty at all 8,111 nested
     decisions of the fuzz runs; the core counts the exceptions in `NestedWithQueuedWork`.)
* **Strict mode** (default): see 5.5.

Threading rules

* Never call anything of `HostSession`, `RunContext`, `EnginePump` or the game from another thread.
  (`HostSession.Run` is the only thread-crossing entry point.) A post from a foreign thread is rejected,
  counted and fails the episode.
* Never block on a task: no `.Wait()`, `.Result`, `GetAwaiter().GetResult()`, `Task.WaitAll`. Hand tasks
  to `ctx.Track`.
* Never create threads, timers, `Task.Run`, `Task.Delay`, `ConfigureAwait(false)` in driver code. If you
  need "later", use `ctx.Post`; if you need "when the engine has settled", `await ctx.Quiescence()`; if
  game code needs a delay, route it to `SchedulingPatches.Delay` (virtual clock).
* Never pump: `EnginePump.RunUntilQuiescent` / `PumpUntil` are for the driver core. A verification tool that
  drives the engine itself OUTSIDE an episode (the `.mcr` replay tool) uses the one public entry
  `EnginePump.DrainUntil(predicate, maxCallbacks, virtualBudget)`: drain, fire virtual timers in order, stop when the
  predicate holds, else `Stuck` with `LastStuckReason` = `quiescent` / `virtual_time_budget` / `pump_callbacks`. It
  throws when it is called while an episode is running (`EnginePump.EpisodeRunning`, set by the run driver).

---

## 4. Types: session, episode, policy

Namespace `SpireHost.Driver` unless noted. All members are engine-thread-only unless stated.

### 4.1 `HostOptions`, `HostSession`

```csharp
public sealed class HostOptions {
    public BootOptions Boot;                 // GameLog only: profile, virtual clock, gates and guards are not options
    public int StackBytes;                   // 256 MB
    public bool Strict = true;
    public List<IDriverModule> Modules;      // registered in order; built by ProductionHost only
    public HashSet<string> NativeAllowList;  // Godot natives that may be reached after boot
}

public sealed class HostSession {
    public static int Run(HostOptions options, Func<HostSession, int> body);   // any thread; creates the engine thread
    public static HostSession CreateOnCurrentThread(HostOptions options);      // you already are on EngineThread.Run
    public static HostSession? Current { get; }
    public RunContext? CurrentEpisode { get; }       // for Harmony patches that need the episode
    public DriverRegistry Registry { get; }
    public EnginePump Pump { get; }
    public IReadOnlyList<PatchRecord> ProductionPatches { get; }   // every registry record present when the session was created
    public string ProductionPatchHash { get; }                     // PatchLedger.Hash of them
    public IReadOnlyList<PatchRecord> ToolPatches { get; }         // what a verification command registered afterwards
    public long EpisodesRun { get; }
    public EpisodeResult RunEpisode(EpisodeSpec spec, IPolicy policy, EpisodeOptions? options = null, IEpisodeObserver? observer = null);
    public string NewRandomSeed();                   // game SeedHelper over a CSPRNG (never Rng.Chaotic)
    public static string SeedFromLabel(string label);// reproducible 12-char seed for tools
    public StrictCounters ReadStrict();
}
```

Creating the session boots the game, installs the driver core's two patches, registers the MAP handler and the core
counters, registers the modules in order, and then asserts one patch per site (section 10). `RunEpisode` never throws
for something that happens inside the episode; it returns an `EpisodeResult` whose `Outcome` is `Win`, `Loss`,
`Aborted` or `Truncated`. **An episode is never dropped.**

### 4.2 `EpisodeSpec`, `EpisodeOptions`, `Scenario`

```csharp
public sealed class EpisodeSpec {
    public string? Seed;                 // 12 chars; canonicalised; null = fresh random seed
    public string[] Characters;          // one id per seat: IRONCLAD SILENT REGENT NECROBINDER DEFECT FOOL (1 seat implemented)
    public int Ascension;                // 0..10, clamped to the profile's MaxAscension like the lobby does
    public Scenario? Scenario;           // tests only (section 9)
}

public sealed class EpisodeOptions {
    public bool AutoSingle = true;       // forced single-action frames are applied by the host
    public bool AutoGold = true;         // gold rewards are taken by the host
    public bool AllowAbandon = false;    // keep ActionFlags.Abandon actions
    public bool ComputeSignatures;       // fold RunSignature into EpisodeResult.TraceHash
    public bool RecordTrace;             // keep the per-decision signatures
    public bool RecordActions;           // keep the chosen indices (replayable with ScriptedPolicy)
    public long MaxPolicyDecisions;      // > 0: Truncated("max_policy_decisions") before decision N + 1 would be asked
    public int MaxCombatTurns = 500;     // Truncated("combat_turn_cap") once a player's turn number EXCEEDS it
    public long DecisionCap = 400_000;   // Aborted(Livelock, decision_cap)
    public int DebugForcedHp;            // DEBUG / TEST ONLY (forced max and current HP)
    public int DebugWongoPoints = -1;    // TOOL ONLY (--compare-run-traces-v2: the trace profile's 0); -1 = derived from the seed
}

public sealed class Scenario {           // TEST ONLY
    public string Name; public Func<RunContext, Task>? Setup; public Func<RunContext, bool>? StopWhen;
    public HashSet<string>? Exempt;      // ids of fidelity invariants this scenario may violate on purpose (ruling D3)
}
```

Training and evaluation use the same `AutoSingle` / `AutoGold` / `AllowAbandon` (DECISIONS-H1 C, B.6).
`MaxPolicyDecisions`, `MaxCombatTurns` and `DecisionCap` are enforced in the main loop AND inside the run start chain
(rooms requests R4 / R7): a prompt loop during the start cannot outrun them. `MaxPolicyDecisions` is also enforced at
a NESTED decision (`RunContext.DecideNested`: the card reward a REWARD pick opens): when the cap is reached there the
policy is not asked and the episode ends `Truncated("max_policy_decisions")`.

Two bounds live inside the game's own code, because a loop there never comes back to the driver (`LivelockGuard`,
ledger rows `livelock-guard-turn-loop`, `livelock-guard-combat-work`): more than 4,000 side-turn starts, or more than
1,000,000 combat-history entries, since the last decision point end the episode `Aborted(Livelock,
turns_without_decision | combat_work_without_decision)`. The second one bounds the work of ONE game call (a card
effect that draws and auto-plays for ever). It is a bound, not a game rule: a chain that would end after more than
the limit is cut off too, and such chains exist - every play of Voltaic doubles the Lightning channeled in the
combat (`Models.Cards/Voltaic.cs`; the card exhausts unless it is upgraded). Measured on the gate fuzz: at most 32 entries between two decision points
at natural HP (6,000 runs); with forced HP 99999 (2,400 runs) at most 710 in three of four processes, and in the
fourth one episode whose six Voltaic plays in rounds 103 to 142 of one fight had 12,399 / 24,795 / 49,587 / 99,171 /
198,339 / 396,675 entries (17 s for the last one); the play after its next would have been cut off. The stress verification's
hanging run (an endless cycle of auto-played Strikes) reaches the limit 30 s after its last decision point. Both
times: BelowNormal priority, machine shared with a training run. A client whose step timeout is shorter than that
sees its own timeout first. An abort carries the tuple that reproduces the run (section 5.6); its signature names only
the game method that added the last entry, which does not tell the two mechanisms apart.

In `--serve` a debug option is honoured only when the process was started with `--allow-debug-options`
(section 13); a production server answers a RESET that carries one with a protocol ERROR.

### 4.3 Run lifecycle of one episode

1. `VeteranProfile.Restore(wongo)`: `SaveManager.Instance.Progress` is replaced by a fresh `ProgressState`
   parsed from the immutable boot-time snapshot (about 3 ms). `WongoPoints` = `VeteranProfile.WongoPointsFor(seed)`,
   a pure function of the run seed, uniform in [0, 1999]. Nothing a previous run wrote survives.
2. Module counters are snapshotted; every `IEpisodeCheck.OnEpisodeBegin` runs (the place for a baseline of
   process-wide counters: the run set-up that follows belongs to the episode). `IEpisodeCheck.OnEpisodeStart(ctx)`
   runs later, in step 3, when the run was set up and the `RunContext` exists (after every `OnRunSetUp`, before
   `FinalizeStartingRelics`).
3. The real start chain (driver-spec-rooms.md section 1): canonical seed -> `UnlockState` from Progress with
   one `ToSerializable`/`FromSerializable` round trip -> `ActModel.GetRandomList(new Rng(hash(seed),
   "act_selection"), ...)` (real discovery branch on) -> ascension clamp -> `Player.CreateForNewRun(character,
   unlock, 1)` -> `RunState.CreateForNewRun(players, acts.ToMutable(), [], Standard, false, ascension, seed)`
   -> `ForceDiscoveryOrderModifications = (GameMode == Standard && !IsCustomGame)` (what
   `ShouldApplyTutorialModifications` returns in the real client) -> `SetUpNewSingleplayer(run, shouldSave: false)` ->
   **`IRunLifecycle.OnRunSetUp`** -> `FinalizeStartingRelics` -> `Launch` -> `EnterAct(0, doTransition: false)`
   (Neow) -> optional `Scenario.Setup`. Prompts raised inside the chain are answered like any other decision, under
   the same caps.
4. `IEpisodeObserver.OnEpisodeStart`, then the main loop (section 5) until terminal / truncated / aborted. After the
   settle that follows every `RunManager.RoomEntered` (and once for the first room) the observer's `OnRoomBoundary`
   runs, before the room's handler is asked for anything.
5. A final evaluation of the episode checks, `IEpisodeObserver.OnTerminalState` (run still alive) ->
   **`IRunLifecycle.OnEpisodeEnd`** -> `RunManager.CleanUp` + `NetLoadingHandle.Release` -> pump unwound, virtual
   clock cleared -> end-of-episode strict accounting, module-counter deltas -> `IEpisodeObserver.OnEpisodeEnd(result)`.

Terminal detection: `run.IsGameOver` at a quiescent point; **Win** iff `run.CurrentRoom.IsVictoryRoom`
(the room of `TheArchitect`, whose last option calls `RunManager.WinRun`, which kills all players), else
**Loss**. A natural death also runs the game's own end-of-run bookkeeping (`RunManager.OnEnded(false)`, restored by
the fidelity pack); nothing leaves the process (network guard).

### 4.4 `IPolicy`, `IEpisodeObserver`, built-in policies

```csharp
public interface IPolicy {
    int Decide(Decision decision);       // blocking; returns an index into decision.Actions (canonical order)
}

public interface IEpisodeObserver {
    void OnEpisodeStart(RunContext ctx);
    void OnRoomBoundary(RunContext ctx) { }                                      // default no-op (replay request R3)
    void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto);   // after the choice, before Apply
    void OnTerminalState(RunContext ctx, EpisodeResult provisional);             // run alive: encode the terminal observation here
    void OnEpisodeEnd(EpisodeResult result);                                      // after teardown: final result
}
```

* `OnRoomBoundary`: the first quiescent point after run start and after every `RunManager.RoomEntered`, before the
  room handler's first `Step` and before any pending frame is answered - the host analogue of "the room was entered
  and the transition finished". (The run-trace comparison dumps there.)
* `decision.Episode` is the `RunContext` (run state, seats, counters): a policy that encodes observations
  reads the game through it. A policy must not call game methods that change state, must not touch the
  pump and must not keep the `Decision` after returning.
* Throw `PolicyStopException` to end the episode as `Truncated`. Any other exception or an index outside
  the list is `Aborted(PolicyFault)`.
* `RandomLegalPolicy(seed)`: uniform over the legal actions, SplitMix64, independent of every game RNG.
* `ScriptedPolicy(indices, fallback?)`: plays back `EpisodeResult.ActionLog`; exhausted -> `Fallback` or
  `Truncated`.
* `FirstActionPolicy`: always index 0.

### 4.5 `EpisodeResult`

`Outcome`, `Abort` (section 5.6), `TruncatedBy`, `Seed`, `Character`, `Ascension` / `RequestedAscension`,
`Acts`, `ActIndex`, `ActFloor`, `TotalFloor` (map points entered in the whole run = "furthest floor"),
`FinalRoom`, `FinalHp`, `WongoPoints`, `Counters` (`EpisodeCounters`: `DecisionPoints`, `PolicyDecisions`,
`NestedDecisions`, `NestedWithQueuedWork`, `AutoSingle[kind]`, `AutoGold`, `PolicyByKind[kind]`,
`DecisionsByKind[kind]`, `VerbOffered[verb]`, `VerbChosen[verb]`, `RoomsEntered`, `EpisodeChecks`, `PumpCallbacks`,
`VirtualTimerFires`, `MaxStackBytes`, `MaxPendingStackDepth`, `TrackedOperations`, `Ticks*` phase timers, and the
PUBLIC HISTORY the encoder writes into GLOBAL 44-48: `CombatsWon`, `ElitesWon`, `BossesWon`, `LastWonRoomType`,
`LastWonTotalFloor`, maintained by the core from `CombatManager.CombatWon`), `Strict` (per-episode delta of
`StrictCounters`), `ModuleCounters` (below), `NativeCalls` / `UnexpectedNativeCalls`, `TeardownFaults`, `TraceHash`,
`Trace`, `ActionLog`, `WallMs`, `StartMs` (+ split).

`Counters.PolicyDecisions` is incremented AFTER `IPolicy.Decide` returns: the value a policy reads inside `Decide` is
the 0-based index of that decision (the protocol's STEP_ID, errata P-6); AUTO_STEPS = the growth of
`Counters.AutoSingleTotal + AutoGold` since the previous policy decision. `EpisodeOptions.MaxPolicyDecisions` is
checked before the policy is asked - in the main loop, in the start chain and at nested decisions - so a cap of N
yields exactly N policy decisions (N OBS frames, TERMINAL STEP_ID = N) (obs request O5; `--selftest-driver` checks it
for every N of a whole run).

**Module counters** (`EpisodeResult.ModuleCounters`, a sorted `"<module>.<name>" -> long` dictionary): every module
may register one `IModuleCounters` source of cumulative process-wide values; the core reads all sources at episode
start and at teardown and stores the per-episode difference (keys ending in `_max` are high-water marks: the value is
kept, not differenced). The core adds `core.*` (run-save boundary counts, inline hook continuations, virtual delays,
intercepted progress writes, network attempts, ...) and `driver.*` (`driver.kind.<Kind>`, `driver.kind_policy.<Kind>`,
`driver.verb_offered.<Verb>`, `driver.verb_chosen.<Verb>`, `driver.episode_checks`, `driver.elites_won`,
`driver.bosses_won`). `--fuzz-runs` sums them into `module_counters` and derives the `coverage` table (how often each
DecisionKind and Verb occurred; what never did); `--determinism` requires them to be equal between the two runs of an
episode (`_max` keys excepted).

---

## 5. Definitions

These are implemented in `SpireHost.Driver\RunDriver.cs` (class comment + `Settle`, `MainLoop`, `Classify`,
`Ask`) and demonstrated by `--selftest-driver`.

### 5.1 Quiescent

The engine is quiescent when **all** of the following hold:

1. the pump queue is empty;
2. no game-task fault is recorded (`TaskHelper.UnobservedFault`), and no strict counter moved;
3. every tracked operation that has completed has been observed (a faulted one aborts the episode);
4. nobody is awaiting `RunContext.Quiescence()` (waiters are released and the engine settles again);
5. no virtual timer is pending - **or** a pending frame is on the decision stack (while the game is waiting
   for an answer a timer loop must not be allowed to spin). With an empty stack, timers are fired in (due time,
   creation order) until none is left, bounded by the virtual time budget. (The paused `ActionExecutor` no longer
   polls on a timer: the fidelity pack wakes it from `Unpause`. The remaining virtual timers are the two Amalgamator
   delays and the 10 s turn-loop timeout.)
6. every registered `IEpisodeCheck` passes. The checks run at the end of every settle and once more before
   teardown; a failure aborts the episode with the reason and detail the check returns (7.9).

There is no wall-clock anywhere in this definition.

### 5.2 Decision point

Quiescent, and exactly one of, **in this priority order**:

| | Condition | Who builds the decision |
|---|---|---|
| T | `run.IsGameOver` | nobody: terminal (Win in the victory room, else Loss) |
| P1 | the decision stack is not empty | the frame on top (`IPendingDecision.Build`) |
| P2 | `CombatManager.Instance.IsInProgress` | the room handler of `RoomScreen.Combat` |
| P3 | no blocking tracked operation is pending, and the screen classifies as `Map`, `Event`, `Shop`, `Rest` or `Treasure` | that screen's room handler |

and the resulting decision has at least one legal action after the central filter (5.4).

Screen classification (`RunDriver.Classify`, only consulted when the stack is empty):

```
CombatManager.Instance.IsInProgress            -> Combat
ctx.MapOpen || run.CurrentRoom is MapRoom      -> Map
run.CurrentRoom is EventRoom                   -> Event      (Ancients, Neow, TheArchitect, FakeMerchant, Crystal Sphere ...)
run.CurrentRoom is MerchantRoom                -> Shop
run.CurrentRoom is RestSiteRoom                -> Rest
run.CurrentRoom is TreasureRoom                -> Treasure
otherwise                                      -> None (stall), e.g. a CombatRoom whose combat ended but nobody offered rewards or opened the map
```

Why this order: a selection prompt can be raised while a combat is in progress and outside the play phase
(start-of-turn hooks, KnowledgeDemon in the enemy turn), so pending frames come before the combat
predicate; a combat that an event started runs while the event's own operation may still be pending, so
P2 is exempt from the "no blocking operation" condition.

With 2-4 seats P3 is evaluated per seat and two entries are added between P1 and P3; the full order is in
section 12.2 (single seat: exactly the table above).

A room handler may answer `HandlerStep.Progressed` instead of a decision when it did work that the real
UI does without asking (starting a tracked operation, waiting for one); the core settles and classifies again.
Opening the treasure chest is NOT such work: the CHEST page is a TREASURE decision with the single action
`OPEN_CHEST`, auto-advanced by the central rules when it is the only action (rooms request R5).

### 5.3 Stall

Quiescent, not terminal, and no decision point. The episode ends **immediately** as
`Aborted(Stall, detail)` with a diagnostics snapshot - no waiting, no retry, no timeout:

| `detail` | Meaning |
|---|---|
| `unknown_screen:<RoomType>` | classification returned `None` |
| `no_handler:<Screen>` | nothing registered for the screen (there is no fallback handler; the executable refuses to start in that state, so this is reachable only in tests) |
| `handler_unrecognised:<handler>` | the handler returned `HandlerStep.Unrecognised(why)` |
| `pending_operation` | a blocking tracked operation is incomplete, the stack is empty and no combat is in progress |
| `no_legal_action:<Kind>` | a decision with zero legal actions (after the abandon filter) |
| `frame_built_nothing:<frame>` | a pending frame returned no decision but did not pop itself |
| `travel_incomplete` | a map vote was applied but the run did not arrive at that coordinate |
| `stall:<step>` (reason `StartFailed`) | a step of the start chain did not complete and nothing asked for input |
| anything a handler passes to `ctx.Stall(detail, message)` | handler-defined |

### 5.4 Livelock

The engine or the driver keeps moving without becoming a decision point: `Aborted(Livelock, detail)`.

| `detail` | Bound |
|---|---|
| `pump_callbacks` | more than 1,000,000 queued continuations in one settle |
| `virtual_time_budget` | more than 120 virtual seconds consumed by one settle (a hopeless poll loop; costs ~0 wall time) |
| `turns_without_decision` | more than 4,000 side-turn starts since the last decision point (`LivelockGuard` on `CombatManager.StartTurn`: the turn loop yields to the pump only once per round) |
| `handler_spin:<name>` | more than 256 `Progressed` steps (or self-removing frames) in a row |
| `quiescence_waiter_spin` | host code re-awaits `Quiescence()` more than 256 times in one settle |
| `auto_advance_spin:<Kind>` | more than 20,000 auto-advanced decisions in a row |
| `decision_cap` | `EpisodeOptions.DecisionCap` decision points in the episode |

Not a livelock: a combat that reaches `EpisodeOptions.MaxCombatTurns` ends the episode as
`Truncated("combat_turn_cap")` - endless fights are possible in the real game, and the consumer decides how
to score them.

### 5.5 Strict mode

With `HostOptions.Strict` (default) each of these marks the episode `Aborted(StrictViolation, detail)`:

| `detail` | Source | When detected |
|---|---|---|
| `foreign_post` | a `SynchronizationContext.Post` from a thread other than the engine thread. The callback is **rejected** (never executed) and attributed (`EnginePump.ForeignPostOrigins`). | next settle |
| `real_time_wait` | a game delay that did not go through the virtual clock (`SchedulingPatches.RealDelays`), or a pump wait in real time | next settle |
| `deferred_call_dropped` | a `Callable.CallDeferred` on a node method (nothing to run it) | next settle |
| `off_thread_continuation` | the hook-action continuation ran off the engine thread | next settle |
| `late_unobserved_fault` | an unobserved task exception that never went through `TaskHelper.RunSafely` (finalizer thread) | next settle |
| `network_attempt` | game code reached an outbound network entry point (`MetricUtilities.PutRequest`, `TimeServer.RequestTime`); the call is blocked, never executed | next settle |
| `native_call:<name>` | a Godot native entry point outside `HostOptions.NativeAllowList` was reached (NullGodot counts every native) | end of episode |

Related: `ProfileGateFired` (a tutorial / discovery branch that must be a no-op for the veteran profile
changed the run: the tutorial room, the act's boss, or anything in the act's room set). Informational only (reported, never fatal): `GameErrorsLogged` (the game logs handled
errors too), `DeferredCalls`, `VirtualDelays`, `InlineContinuations`, `InterceptedProgressWrites`.

After an aborted episode the core tears the run down, drops what is left in the pump queue, forces the
finalizers (so that leftovers are attributed to the broken episode) and the session is clean again -
verified by the last case of `--selftest-driver`.

### 5.6 What an aborted episode reports

`EpisodeResult.Abort` (`AbortInfo`):

| Field | Content |
|---|---|
| `Reason` | `StartFailed`, `GameTaskFault`, `ActionFault`, `HandlerFault`, `PolicyFault`, `Stall`, `Livelock`, `StrictViolation`, `ProfileGateFired`, `HostBug` |
| `Detail` | sub-code (tables above; for faults: `unobserved_fault`, `operation:<what>`, `apply:<source>`, `build:<source>`, `ask`, `bad_index`) |
| `Message` | first line of the exception / explanation |
| `ExceptionType`, `Frames` | type and first 6 stack frames of the root exception, the first game frame (`MegaCrit...`) first, no file paths |
| `Diagnostics` | one line: phase, seed, act / floor, room stack, `mapOpen`, HP, combat phase / turn / synchronizer state, executor running / paused / current action, decision stack depth and top frame, pending operations, pump queue, virtual timers, decision count, last decision kind |
| `Signature` | `Reason/Detail @ first frame`: the grouping key of `--fuzz-runs` |

Fault attribution: an exception thrown synchronously by `Apply` is `ActionFault`; by `Step` / `Build` /
`Extend` `HandlerFault`; by the policy `PolicyFault`; a faulted fire-and-forget game task or tracked
operation `GameTaskFault`. If a handler calls `ctx.Fail` (or throws `EpisodeAbortException`) from inside a
game callback and the game's task captures it, the original reason is restored when the fault surfaces.
Fail-closed (DECISIONS-H1 A.7): a game exception is never swallowed.

`--fuzz-runs` prints for every failing run the tuple that reproduces it:
`(seed, character, ascension, policy_seed, hp)`; `--record-dir` also writes the action log.

---

## 6. The decision model

### 6.1 Kinds, sub-kinds, verbs (obs-schema-v1 appendix A; values must not be renumbered)

```
DecisionKind    None 0, Map 1, Combat 2, CardSelect 3, Reward 4, CardReward 5, Event 6, Shop 7, Rest 8,
                Treasure 9, Confirm 10, Bundle 11, Crystal 12, Terminal 13
CardSelectKind  Unknown 0, SimpleGrid 1, SimpleGridRewards 2, CombatPile 3, Hand 4, HandDiscard 5, HandUpgrade 6,
                DeckUpgrade 7, DeckTransform 8, DeckEnchant 9, DeckRemoval 10, DeckGeneric 11, ChooseACard 12
TreasurePhase   Chest 1, Relic 2
Verb            Invalid 0, EndTurn 1, UndoEndTurn 2, PlayCard 3, UsePotion 4, DiscardPotion 5, Travel 6,
                SelectCard 7, ConfirmSelection 8, Cancel 9, Skip 10, ChooseOption 11, PickOfferCard 12,
                Proceed 13, OpenChest 14, CrystalClick 15
```

### 6.2 `Decision`

```csharp
public sealed class Decision {
    public DecisionKind Kind;
    public int SubKind;                  // CardSelectKind / TreasurePhase / 0
    public int ActingSeat;               // the seat this screen belongs to, or -1 = several seats may act
    public int SeatMask;                 // bit i = seat i has a legal action (computed by the core)
    public bool Nested;                  // asked from a synchronous game callback
    public IDecisionSource? Source;      // who applies (null only for nested decisions)
    public object? Context;              // one of the *Context classes: prompt / screen data for the encoder
    public readonly List<LegalAction> Actions;
    public long Sequence;                // index of the decision point in the episode (set by the core)
    public RunContext Episode;           // set by the core
    public Decision Reset(DecisionKind kind, IDecisionSource? source, int actingSeat = 0, int subKind = 0, object? context = null);
    public void Add(Verb verb, int seat, GameRef source = default, GameRef target = default, int aux = 0, ActionFlags flags = None, object? payload = null);
    public ulong ShapeHash();
}
```

A `Decision` object is valid only until the next decision is asked: sources may (and on hot paths
should) reuse one instance via `Reset`. Seats are indices into `RunState.Players`; single player is the
N = 1 case of the same model.

`Decision.SealForTest(allowAbandon)` applies the central filter and canonical sort to a hand-built decision
(tests only); `Decision.VerbMask` is the bit set of offered verbs.

**Contexts** (`DecisionContexts.cs`). The module that builds a decision fills its context COMPLETELY; the encoder
reads the context and white-listed public game members and nothing else of the builder (ruling D6: no side tables
keyed by decision or context objects, no reference from the encoder to a module). Add fields, never repurpose.

| Context | Fields |
|---|---|
| `MapContext` | `Map`, `Travelable` |
| `CombatContext` (combat C1) | `State`; `Seats[seat]` = `CombatSeatInfo` (`State`: `NotReady` reason or `Ready`, `ReadyToEndTurn`, `CanUndoEndTurn`, per hand card `HandPlayable` / `HandReason` / `HandPreventer`, per potion slot `PotionUsable` / `PotionDiscardable`); `Creatures` (allies, then enemies: the Creature ordinal of `GameRef`); `Hittable` per creature; `AllyCount` |
| `CardSelectContext` (combat C2) | `Kind`, `Player`, canonical `Options`, `OptionCounts` (instances per merged class), effective `MinSelect` / `MaxSelect`, `Cancelable`, `CanSkip`, `RequireManualConfirmation`, `Picked` (instances, pick order), `Prefs`, `SourceModel`, `Pile`, `InCombat`; `Entry` (which of the 15 `From*` overloads), `ContextKind` (game action / hook / blocking), `RawMin` / `RawMax`, `InstanceCount`, `Merged`, `PickedOrdinals`, `OptionPicked` (picks per option = CARDS.sel_count), `Enchantment` (+ `EnchantmentAmount`), `PromptTable` / `PromptKey` (the header the real screen shows: prefs prompt or the screen's fixed header), `SortedByComparison`, `OutsidePlayPhase`, `DuringEnemyTurn` |
| `RewardContext` | `Set`, `Terminal` |
| `CardRewardContext` | `Reward`, `Options`, `Alternatives` (a REROLL the screen would no longer show is absent) |
| `ShopContext` | `Inventory`, `FakeMerchant` |
| `RestContext` | `Options`, `Successes` |
| `TreasureContext` (rooms R1) | `Room`, `Relics`, `GoldGained`, `Finished` (the PROCEED-only page), `Empty`, `Awarded`. Pages are `Decision.SubKind` = `TreasurePhase.Chest` / `Relic`; no further sub-kind |
| `ConfirmContext` | `Key` (a name of vocabulary category `confirm_key`; every key any module uses is listed in `ConfirmKeys.All`, which the vocabulary export and an obs self-test enumerate - obs O7), `SourceModel` |
| `BundleContext` | `Player`, `Bundles`, `CanSkip` |
| `EventContext` (events R1) | `Event`, `Finished`, `EventId`, `PageKey`, `IsShared`, `IsAncient`, `PageVars` (numbers the page text prints, template order), `Options[]` = `EventOptionInfo` (`Index`, `Option`, `TextKey`, `IsLocked`, `IsProceed`, `WasChosen`, `WillKillPlayer`, `Legal`, `Synthesized`, `Relic`, `Vars`, `Cards`, `Models`); helpers `BeginPage`, `AddOption`, `AddSynthesizedProceed` |
| `CrystalContext` (events R2) | PUBLIC VIEW: `Width`, `Height`, `Divinations`, `Tool`, `Cells[]` (`X`, `Y`, `Hidden`, and for cleared cells `ItemType` / `ItemGood` / `ItemIndex`), `Exposed`, `ExposedComplete`; `FillPublicView()` rebuilds it from cleared cells only. **`Minigame` is HOST-ONLY** (hidden item layout): for the frame that applies the click; an encoder never reads it |

Host-only by contract, never read by an encoder: `RunContext.Seed`, `RunSignature` / `StateSignature`,
`CrystalContext.Minigame`, and `ICardSelectHostView` - implemented by the CARD_SELECT frame and reachable only through
`Decision.Source` (not the context): `RawOptions` (the candidates exactly as the game handed them to the selector;
for the draw pile that order is hidden information) and `OrdinalOf(instance)`. It exists for verification tools that
replay choices recorded on the real screens.

`CardSelectEntry` (combat C9): the 12 `CardSelectCmd.From*` names that go through `ICardSelector` have 15 public
overloads, each captured by its own read-only prefix + postfix; the 13th name, `FromChooseABundleScreen`, never
reaches a selector (its gate and BUNDLE frame are the events module's; the combat module carries no patch on it).
`PlayerChoiceSynchronizer.SyncLocalChoice` is not on the local-selector path: the game's own local-selector branch
does not call it (open item C10, section 12).

Which enemy intents the UI has already hidden (combat C3): `RunContext.IntentPerformed(enemy)` is true from the
moment `Creature.TakeTurn` begins for that enemy until the next player turn (driver patch
`enemy-turn-intent-performed`). A prompt raised inside the enemy turn (`CardSelectContext.DuringEnemyTurn`) is encoded
without the intents of those enemies (DECISIONS-H1 F.6).

### 6.3 `LegalAction` and `GameRef`

```csharp
public struct LegalAction {
    public Verb Verb;
    public int Seat;                 // who acts
    public GameRef Source;           // what is acted with (card, potion slot, map point, option ...) or None
    public GameRef Target;           // targeted creature or None
    public int Aux;                  // CRYSTAL: tool; else 0
    public ActionFlags Flags;        // AutoGold, Abandon
    public object? Payload;          // private to the applier
    public IDecisionSource? Owner;   // applier of THIS action if not Decision.Source (set by the core for frame extensions)
}

public readonly struct GameRef { public readonly RefKind Kind; public readonly object? Object; public readonly int Index; }
```

A `GameRef` = a typed reference to a **live game object** plus its **public ordinal**:

| `RefKind` | `Object` | `Index` (ordinal) |
|---|---|---|
| `Card` | `CardModel` instance | hand card: index in `PlayerCombatState.Hand.Cards`; selection candidate / offered card: index in the canonical candidate list (`CardSelectContext.Options`, `CardRewardContext.Options`) |
| `Creature` | `Creature` | index in `CombatState.Creatures` (allies, then enemies in slot order) |
| `PotionSlot` | owning `Player` | slot index |
| `MapPoint` | `MapPoint` | `row * 64 + col` |
| `EventOption` | `EventOption` | index in `EventModel.CurrentOptions` |
| `RestOption` | `RestSiteOption` | index in `RestSiteSynchronizer.GetLocalOptions()` |
| `Reward` | `Reward` | index in `RewardsSet.Rewards` |
| `ShopEntry` | `MerchantEntry` | index in `MerchantInventory.AllEntries` |
| `CardRewardAlt` | `CardRewardAlternative` | index in the alternatives list |
| `TreasureRelic` | `RelicModel` | index in `TreasureRoomRelicSynchronizer.CurrentRelics` |
| `Bundle` | the bundle's card list | bundle index |
| `ConfirmChoice` | null | 0 = no, 1 = yes |
| `CrystalCell` | the cell | `y * width + x` |

Use the factories (`GameRef.OfCard(card, ordinal)`, `OfCreature`, `OfPotionSlot`, `OfMapPoint`, ...).

### 6.4 Canonical order (binding for the encoder, replay logs and leak tests)

The core sorts every action list once, in `Decision.Seal` (stable insertion sort), by the key

```
(Verb, Seat, Source.Kind, Source.Index, Target.Index, Aux)        ascending, None = index -1
```

and rejects two actions with the same key (`HandlerFault`: give them distinct ordinals).
**The resulting index is the action index of the whole system**: what `IPolicy.Decide` returns, what
`ActionLog` records, what the protocol's `action_index` means. Sources add actions in any order and never
sort.

Consequences / rules:

* The order depends only on verbs, seats and **public ordinals**. An ordinal must come from a public
  order: hand order, slot order, option order as shown, (row, col), the canonical candidate order. Never
  from a `HashSet` enumeration, object identity, deck / draw-pile order, creation order of hidden things.
* **Pile-sourced candidates** (`CardSelectCmd.FromCombatPile`, the deck prompts): the game's own stable sort
  leaks the draw-pile order among identical cards. The owner of CARD_SELECT merges identical cards into
  classes, orders the classes canonically, exposes one candidate per class (`OptionCounts`), and maps the
  chosen class to its first not-yet-picked instance (DECISIONS-H1 B.4, driver-spec-combat.md 4.4-5).
* **The canonical card-class provider** (combat C7 = obs O1). Classes and their order come from ONE function
  registered in the driver registry:
  `delegate void CanonicalCardClasses(RunContext ctx, IReadOnlyList<CardModel> cards, PileType pile, int seat, bool unpoweredPreviews, List<CardModel> classes, List<int> counts)`.
  It is the observation encoder's content key (obs-schema-v1 5.2.3, `ObsEncoder.CanonicalClasses`), registered by
  `ObsModule` in EVERY mode - serve, fuzz, determinism, replay-run, the oracle tools, every self-test - because the
  order of legal actions must be identical in all of them (ruling B1). `RunContext.CanonicalCardClasses` hands it to
  the combat module; without a provider a pile-sourced prompt aborts (`HandlerFault/card_select_no_class_provider`).
  The key is a function of public card faces only. Episode logs and tool outputs record the vocabulary hash; a log
  recorded under another vocabulary or game build is refused by `--replay-run`.
* The obs encoder emits ACTIONS rows in exactly this order (obs-schema-v1 5.13 orders by
  `(verb, seat, src_table, src_row, tgt_row, aux)`; that is the same order as long as entity rows are
  emitted in ordinal order, which is the encoder's job; if the two ever disagree, **this order wins** and
  the schema text is amended).
* `Decision.ShapeHash()` (kind, sub-kind, seat, every key) is folded into the determinism trace, so a
  change in enumeration order or content shows up as a trace mismatch.

### 6.5 Central rules applied to every decision (`RunDriver.Ask`)

In this order, for pump-driven and nested decisions alike:

1. **Room-frame extensions** (pump-driven decisions of kind Map / Shop / Rest / Event / Reward / Treasure
   only): each registered `IRoomFrameExtension` may append actions (6.6).
2. **Abandon filter**: actions flagged `ActionFlags.Abandon` are removed unless
   `EpisodeOptions.AllowAbandon` (Trial double-down confirmation, any "abandon").
3. **Canonical sort**, duplicate check, `SeatMask`.
4. Zero actions -> stall.
5. **Auto gold**: if `AutoGold` and an action is flagged `ActionFlags.AutoGold`, the lowest-index such
   action is applied without asking; counted in `Counters.AutoGold`.
6. **Auto single**: else if `AutoSingle` and exactly one action is left, it is applied without asking;
   counted in `Counters.AutoSingle[kind]`.
7. Otherwise the state signature is taken (if enabled) and **the policy is asked**; counted in
   `PolicyDecisions`, `PolicyByKind[kind]`, `NestedDecisions`.
8. `IEpisodeObserver.OnDecision(ctx, decision, chosen, auto)`.

Sources do not implement any of this themselves: build the complete, honest action list and let the core
decide. (In particular: do not skip "obvious" frames yourself; a frame with one action is auto-advanced
and counted here, which the protocol's AUTO_STEPS needs.)

### 6.6 Room-frame extensions

```csharp
public interface IRoomFrameExtension : IDecisionSource {
    void Extend(RunContext ctx, Decision decision);     // append actions; Owner is set by the core
}
registry.SetFrameExtension(extension);
```

For actions that belong on room-level frames owned by somebody else: out-of-combat potion use / discard
(DECISIONS-H1 B.7) is offered on MAP (core), EVENT and the FakeMerchant SHOP frame (events module) and SHOP /
REST / REWARD / TREASURE (rooms module) frames by **one** extension of the rooms module (`rooms:potions`), without
any of those handlers knowing about potions. Never on nested selection frames (CARD_SELECT, CARD_REWARD, BUNDLE,
CRYSTAL, CONFIRM). Consequence for owners and their tests: a frame may hold USE_POTION / DISCARD_POTION actions the
owner did not add (e.g. a finished event's frame is "the synthesized PROCEED plus possibly potion actions"; the CHEST
page is asked, not auto-advanced, when potion actions are present).

---

## 7. Writing a module

### 7.1 Registering

```csharp
public sealed class CombatModule : IDriverModule {
    public string Name => "combat";
    public void Register(DriverRegistry registry) {
        SelectionCapture.Install();                                          // Harmony patches through PatchRegistry
        registry.SetLifecycle(new CombatLifecycle());                        // per-run hooks (7.8)
        registry.SetRoomHandler(RoomScreen.Combat, new CombatHandler(), owner: "combat");
        registry.SetModuleCounters(new CombatCounters());                    // "combat.*" (4.5)
    }
}
```

`Register` runs once, on the engine thread, after the game booted, before the first episode; it is the only place
where a module installs Harmony patches. Module constructors must not touch a game type (they run before boot). The
registry is frozen while an episode runs. `DriverRegistry`:

| Slot | Members |
|---|---|
| room handlers | `SetRoomHandler(screen, handler, owner)`, `GetRoomHandler`, `RemoveRoomHandler`, `ScreensWithoutHandler()` |
| lifecycles (7.8) | `SetLifecycle(lifecycle)` (same `Name` replaces), `RemoveLifecycle(name)`, `FindLifecycle<T>()`, `Lifecycles` |
| frame extensions (6.6) | `SetFrameExtension`, `RemoveFrameExtension`, `FrameExtensions` |
| episode checks (7.9) | `SetEpisodeCheck(check)`, `RemoveEpisodeCheck(name)`, `EpisodeChecks` |
| module counters (4.5) | `SetModuleCounters(source)`, `ModuleCounterSources`, `ReadModuleCounters()` |
| canonical card classes (6.4) | `SetCanonicalCardClasses(provider, name)`, `CanonicalCardClasses`, `CanonicalCardClassesName` |
| vocabulary identity | `SetVocabulary(vocabHash, catalogHash, source)`, `VocabHash`, `CatalogHash`, `VocabSource` |
| description | `Modules`, `Describe()` (what `--driver-info` and `--patch-report` print) |

### 7.2 Room handler

```csharp
public interface IRoomHandler : IDecisionSource {       // IDecisionSource: string Name; void Apply(ctx, decision, index)
    HandlerStep Step(RunContext ctx, int seat);         // Ask(decision) | Progressed(note) | Unrecognised(why)
}                                                       // multi-seat only: | Waiting(why) | Vote(decision)   (12.2)
```

`Step` is called only at a decision point of your screen (5.2): quiescent, stack empty, not terminal. `seat` is the
seat whose screen is asked for: always 0 in a single-seat run; in a multi-seat run the core calls it once per seat
(12.2) and the handler answers for THAT seat from that seat's game objects. `RoomScreen.Combat` is called with -1
(one decision for all seats).

```csharp
public HandlerStep Step(RunContext ctx, int seat) {
    RestState state = ctx.RoomData<RestState>();                                 // per-room scratch, created on first use (arrays indexed by seat)
    IReadOnlyList<RestSiteOption> options = ctx.RunManager.RestSiteSynchronizer.GetOptionsForPlayer(ctx.Seats[seat]);
    Decision d = _decision.Reset(DecisionKind.Rest, this, actingSeat: seat, context: _context);
    for (int i = 0; i < options.Count; i++)
        if (options[i].IsEnabled) d.Add(Verb.ChooseOption, seat, GameRef.OfRestOption(options[i], i));
    if (state.Successes[seat] > 0 || options.Count == 0) d.Add(Verb.Proceed, seat);  // UI rule: leave only after one success
    return d.Actions.Count == 0 ? HandlerStep.Unrecognised("no enabled option") : HandlerStep.Ask(d);
}

public void Apply(RunContext ctx, Decision decision, int actionIndex) {
    LegalAction a = decision.Actions[actionIndex];
    if (a.Verb == Verb.Proceed) { ctx.RunManager.Proceed(); return; }            // the UI's call; the core opens the host map (7.3)
    ctx.Track(Choose(ctx.RunManager.RestSiteSynchronizer, a.Source.Index, ctx.RoomData<RestState>()), "rest option");
}
```

Rules for `Step`: enumerate with the game's own predicates plus the UI-only rules (the specs list them);
no side effects on the game except through `Progressed` work that the real UI does unasked; no trial and
error. Rules for `Apply`: start the same game call the real UI makes (synchronizer / command / action
queue), hand every returned task to `ctx.Track`, return. Do not wait. The core settles afterwards.

### 7.3 Leaving a room: `RunManager.Proceed()` and `ctx.OpenMap`

`RunManager.Proceed()` (Runs/RunManager.cs:1289-1296) only talks to `NMapScreen`, which does not exist headlessly.
The driver core closes that gap in one place: the postfix `run-proceed-opens-map` calls `ctx.OpenMap` whenever the
game's `Proceed()` returns during an episode, exactly where the real method opens the map (not when it advanced a
Blitz node). From then on the screen classifies as `Map` until a room is entered (`RunManager.RoomEntered` resets it).

So every owner calls the game method its UI button calls - `RunManager.Proceed()` (LEAVE of shop and rest site, a
finished event's PROCEED, the FakeMerchant's PROCEED, an Ancient's PROCEED option) or
`RunManager.ProceedFromTerminalRewardsScreenAsync()` (terminal reward screens and the treasure room; the game calls
`Proceed()` inside it unless it resumes a parent event) - and then asserts `ctx.MapOpen` where the map must be open
(`HostBug/map_not_opened` otherwise). No module calls `ctx.OpenMap` itself any more. It stays public and idempotent
for a UI path that would enable travel without `Proceed()`; the flows of v0.113 have none (events request R3).

### 7.4 Pending frames (asynchronous game callbacks)

When the game asks through an async callback (`ICardSelector.GetSelectedCards`, `RewardsSet.testSelector`,
a patched `ShowScreen`, ...), do not answer inside the callback. Park the request:

```csharp
public Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int min, int max) {
    var frame = new CardSelectFrame(options.ToList(), min, max, capturedPromptContext);
    _ctx.Push(frame);                 // the core will ask it at the next quiescent point
    return frame.Completion.Task;     // the game awaits this
}

sealed class CardSelectFrame : IPendingDecision {
    public Decision? Build(RunContext ctx) { ... }                       // decision for the frame's CURRENT state
    public void Apply(RunContext ctx, Decision d, int i) {
        ... accumulate a pick, or:
        ctx.Pop(this);
        Completion.SetResult(pickedInOrder);                             // the game resumes inline; the core settles
    }
}
```

* The stack is LIFO: requests nest (taking a relic reward runs `AfterObtained`, which offers a reward set,
  whose card reward asks for a card ...). The top frame is always answered first; an outer frame is not
  asked while an inner one exists.
* The object that pushes a frame pops it. `Build` may return `null` only after the frame popped itself
  (the request became moot).
* Multi-step prompts (ordered multi-select) stay one frame: each `Apply` mutates the frame, the next
  `Build` shows the new state. The pick order is the order handed to the game (it is meaningful).
* Complete the game's `TaskCompletionSource` only from `Apply` (engine thread).

### 7.5 Nested decisions (synchronous game callbacks)

```csharp
public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alts) {
    Decision d = new Decision(DecisionKind.CardReward, null, actingSeat, context: new CardRewardContext { ... });
    for (int i = 0; i < options.Count; i++) d.Add(Verb.PickOfferCard, seat, GameRef.OfCard(options[i].Card, i));
    for (int k = 0; k < alts.Count; k++)    d.Add(Verb.ChooseOption, seat, GameRef.OfCardRewardAlt(alts[k], k));
    int index = _ctx.DecideNested(d);                  // blocks; no pump re-entry; same central rules; counted
    LegalAction a = d.Actions[index];                  // index refers to the SORTED list
    return a.Verb == Verb.PickOfferCard ? new CardRewardSelection { card = a.Source.Card }
                                        : new CardRewardSelection { alternative = a.Source.CardRewardAlt };
}
```

Use it only where the game leaves no choice (a synchronous interface). `Decision.Source` is null and is
not invoked: the callback itself turns the index into its return value. The game may call again (reroll,
"select more"): each call is its own nested decision.

`CardReward.OnSelect` only reads the **per-player selector stack** (`Rewards/CardReward.cs:185`), not the
local selector. So the reward owner pushes its own `ICardSelector` for the duration of that one reward
(`CardSelectorStack.PushSelector`) and forwards `GetSelectedCards` to `ctx.CardSelector`, the run's
selector that the CARD_SELECT owner installed with `UseLocalSelector` and published in `OnRunSetUp`. Never
use the global `_testSelector` (DECISIONS-H1 B.2).

### 7.6 Tracked operations

```csharp
ctx.Track(Task task, string what, bool blocking = true);
ctx.Post(Func<Task> work, string what, bool blocking = true);   // run from the pump queue, then Track
```

Every task a game API returns to driver code goes to `Track`. The core observes it at every quiescent
point: **faulted -> the episode aborts** (`GameTaskFault/operation:<what>`); `blocking` and incomplete ->
no room handler is consulted (pending frames and the combat handler still are); incomplete at quiescence
with an empty stack and no combat -> `Stall/pending_operation`. Use `blocking: false` only for tasks that
legitimately outlive the next room-level decision.

`Post` is for game *event* handlers (`CombatManager.CombatWon`, `RelicsAwarded`, ...): never `await` inside
the event callback; post the work, and the real UI's "on a later frame" is reproduced.

Multi-seat: `ctx.Track(task, what, seat)` marks the operation as one seat's (its event option, purchase, rest
option, chest). While it is incomplete only that seat's room handler is held back - the operation may be waiting for
teammates - and the other seats go on; an operation tracked without a seat holds every seat's room handler back
(12.2). With one seat there is no difference.

### 7.7 Waiting for the engine

There is exactly one correct way for host code to wait until the engine has settled:

```csharp
ctx.Post(async () => {
    await ctx.Quiescence();              // completes at the next quiescent point, on the engine thread
    if (room.Encounter.ShouldGiveRewards) await room.OfferRoomEndRewards();
    else await TerminalProceed(ctx);
}, "after combat won");
```

Everything else is wrong: polling a predicate, waiting on an event with a timeout, `Task.Yield` loops,
`Task.Delay`, pumping. Most of the time you do not need it at all: `Apply` returns and the core settles
before it asks again; `Step` / `Build` always see a settled engine.

### 7.8 Lifecycles

```csharp
public interface IRunLifecycle {
    string Name { get; }                      // stable owner name; same name replaces
    void OnRunSetUp(RunContext ctx);          // after SetUpNewSingleplayer, before FinalizeStartingRelics / Launch / EnterAct(0)
    void OnRoomEntered(RunContext ctx);       // RunManager.RoomEntered
    void OnEpisodeEnd(RunContext ctx);        // before CleanUp, for every outcome; must not throw
}
```

`OnRunSetUp` is where per-run hooks are installed: `CardSelectorStack.UseLocalSelector` (the stack is
created per run by `InitializeShared`), `RewardsSet.testSelector` (a **static**), event subscriptions on
synchronizers (per run) and on `CombatManager.Instance` (process-wide). `OnEpisodeEnd` removes every one
of them - a leftover static hook breaks the next episode, and a leftover subscription on a singleton leaks
one run per episode.

### 7.9 Episode checks and module counters

```csharp
public interface IEpisodeCheck {
    string Name { get; }
    void OnEpisodeStart(RunContext ctx);
    CheckFailure? Check(RunContext ctx, bool final);     // null = ok; else (AbortReason, detail, message)
}
public interface IModuleCounters {
    string Module { get; }                               // key prefix
    void Read(IDictionary<string, long> into);           // cumulative, process-wide values
}
```

An episode check is evaluated by the core at the end of EVERY settle (every quiescent point) and once more, with
`final = true`, before teardown (combat C4 = rooms R3 = fidelity R1). It must be cheap and read-only. The fidelity
module's check (`fidelity:invariants`) turns an assertion that fired inside game code - even when the game swallowed
the exception - into an abort at the next quiescent point, and re-checks the static test hooks that must stay inert.
A test scenario that violates one invariant on purpose names it in `Scenario.Exempt`; nothing else can exempt.

### 7.10 `RunContext` reference

```
Session, Pump, Spec, Options, Counters
Run (RunState), Seed (HOST ONLY), RunManager, CombatManager
Seats, SeatCount, LocalPlayer, SeatOf(player), CurrentRoom
CardSelector { get; set; }                       the run's ICardSelector (7.5)
CanonicalCardClasses                             the registered class provider (6.4), or null
IntentPerformed(enemy)                           the UI has hidden this enemy's intent (6.2)
MapOpen, OpenMap(why), RoomSerial
RoomData<T>(), RoomData<T>(room)                 per-room scratch keyed by the room object
Push(frame), Pop(frame), PendingCount, TopPending
Track(task, what, blocking), Post(work, what, blocking), HasBlockingOperation
Quiescence()
DecideNested(decision)
Fail(reason, detail, message), Stall(detail, message)      end the episode as Aborted (never swallow "cannot happen")
Diagnose()
```

Harmony patches that need the episode: `HostSession.Current?.CurrentEpisode` (null outside an episode:
then run the original / do nothing).

---

## 8. What is forbidden

1. **Blocking on tasks** (`Wait`, `Result`, `GetResult`), sleeping, real timers, `Task.Run`, new threads.
2. **Touching the pump** or the game from another thread; pumping from driver code.
3. **Trial-and-error legality.** In TestMode an action that cancels itself at execution time *throws*
   (`GameAction.cs:96-106`). Every action offered must be exactly what the engine will accept: the same
   predicates the engine checks at execution, plus the UI-only rules. A cancelled or rejected action is a
   host bug and aborts the episode.
4. **Reading hidden information to order, filter or shape anything a policy sees**: draw-pile order,
   seeds, RNG state or counters, unrevealed map-node contents, pre-rolled encounter / event sequences, relic
   grab bags, monster state machines / next move ids (only the displayed intent), `CombatId` and other
   ids assigned in hidden order. `RunContext.Seed`, `RunSignature`, `StateSignature` are host-only.
5. **Re-implementing game rules.** No damage math, no card effects, no reward generation in the host. If
   the UI does something the headless game does not (a UI-only flow), call the same game methods the UI
   calls, in the same order, and register it in FIDELITY.md.
6. **Swallowing faults**: no blanket `catch`, no finalizer patch that eats exceptions over code that has
   game logic after the throwing line (notes/host-spike.md 9.2). Unknown state -> `ctx.Fail`.
7. **Skipping the central rules**: no own auto-advance, no own sorting, no private action filtering "for
   training". Flags (`AutoGold`, `Abandon`) and `EpisodeOptions` exist for that.
8. **Global test hooks as shortcuts**: `CardSelectorStack.SetTestSelector` / `_testSelector`,
   `RunState.CreateForTest`, `UnlockState.all`, `CardCmd.AutoPlay`, `RunManager.EnterMapCoord` instead of
   the vote path, `ActionExecutor.Pause/Unpause`. (ScenarioTools' debug jumps are for tests only.)
9. **Per-decision reflection, LINQ and logging on hot paths** (combat enumeration, selection frames,
   encoder). Use the publicized reference; reuse `Decision` instances and lists.
10. Loading an assembly that `GameAssemblies` refuses by name; writing outside the project directory.

---

## 9. Testing a handler in isolation

All tests run a **real run start** on the production configuration and then use the game's own debug facilities on
the engine thread.

```csharp
EpisodeResult r = ScenarioRunner.Run(session, "silent vs byrdonis with toolbox", async ctx => {
    await ScenarioTools.GiveRelic(ctx, "TOOLBOX");            // RelicCmd.Obtain (AfterObtained runs)
    await ScenarioTools.GiveCard(ctx, "SURVIVOR", upgrades: 1);
    ScenarioTools.SetGold(ctx, 300);
    await ScenarioTools.JumpToEncounter(ctx, "BYRDONIS_ELITE"); // RunManager.EnterRoomDebug (dev console "fight")
}, character: "SILENT", policy: new RandomLegalPolicy(7), stopWhen: ScenarioTools.RoomFinished);
bool ok = ScenarioRunner.Passed(r);
```

* `Scenario.Setup` runs after the start chain as a tracked operation (frames it raises are answered by
  the policy). `StopWhen` is checked at every quiescent point; default: the room was left for the map.
  The episode then ends `Truncated("scenario")`; `ScenarioRunner.Passed` = not aborted, no teardown fault.
  `ScenarioRunner.Run(..., exempt: new[] { "rest_options_inert" })` is the explicit, per-scenario exemption from a
  named fidelity invariant (ruling D3); it is counted in `fidelity.scenario_exemptions_used`.
* `ScenarioTools`: `GiveRelic`, `GiveCard`, `GivePotion`, `SetGold`, `SetHp`, `JumpToEncounter`,
  `JumpToEvent(id | model)`, `JumpToRoom(RoomType)`, `EnterAct`, `EnterBoss`, `SetVisitedFloors`, and in combat
  `CombatState`, `AddCard`, `SetResources`, `ApplyPower`, `Afflict`; lookups `Encounter/Event/Relic/Card/Potion(id)`;
  `RoomFinished`. Test only: they are the game's debug facilities.
* Assertions: use an `IEpisodeObserver` or a scripted policy to look at every decision; `ScriptedPolicy` to force a
  path; `EpisodeOptions.RecordTrace` to compare signatures.
* Where tests live: every module has test code in its own assembly, reached through the executable as
  `SpireHost <area> --selftest` (areas: combat, rooms, events, fidelity, obs, replay; section 13). They all run on the
  production configuration; a test that needs a fake keeps it inside the test (the driver self-test registers and
  removes its fake handlers around each case).
* `SpireHost.Driver.Tools.FrameAudit` (TEST SEAM): the module self-test probes call `FrameAudit.Run(decision)`; for
  the area commands the executable plugs in the real encoder's consistency checks (canonical CHOICE and ACTIONS order,
  unmapped action references, overflow, frame well-formedness) without any module referencing the encoder. It is
  never plugged in for a top-level command.
* Gates kept green after every change: `--selftest-driver`; `--determinism`; `--replay-run --selfcheck N`;
  `--fuzz-runs` with and without `--hp 99999` (0 aborted); every area's `--selftest`; `--check-ledger`. The full gate
  list with the commands is in `host\README.md` (`bench\h1-integration\run_gates.sh`).
* `--hp N` / `EpisodeOptions.DebugForcedHp` (forced max and current HP) is how a random policy gets through
  whole runs. Debug / test only. It makes endless fights possible; `MaxCombatTurns` truncates them.

---

## 10. Conventions

**Logging.** stdout carries only the JSON result or protocol frames. Diagnostics: `HostLog.Info`
(only with `--verbose`) / `HostLog.Error` to stderr. Nothing per decision. Game log: `--game-log off|warn|info|debug`.

**Faults.** Use `ctx.Fail(reason, detail, message)` / `ctx.Stall(detail, message)`; `detail` is a short
stable snake_case code (it is the grouping key in fuzz summaries), `message` carries the specifics. Do not
invent new `AbortReason` values without section 14.

**Patches.** Every Harmony patch and every non-Harmony intervention goes through `PatchRegistry`
(`Prefix`, `Postfix`, `PrefixPostfix`, `Transpiler`, `Finalizer`, `Note`) with an id, a `PatchCategory` and
a reason that names the game site (`File.cs:line`). The Harmony instance itself is not public: nothing outside
`SpireHost.Core` can patch around the registry (the thread audit registers one id per audited method). Installation is fail-closed: a missing target or a duplicate id
throws at boot. Transpilers must check "exactly N hits". For the common "make this TestMode getter a constant" patch
use `GatePatch.ForceGetter(id, method, GatePatch.TestModeIsOn|TestModeIsOff|NonInteractiveIsActive, value,
expectedHits, category, reason, async: false)`.

**One patch per site** (ruling B3). A method - an async method and its state machine count as one site - may carry
the patches of ONE registry id. Where two ids must touch one method, the later installer declares it:
`PatchRegistry.DeclareSharedSite(reason, idA, idB)`. `PatchRegistry.AssertOnePatchPerSite()` runs when the session is
created and throws for an undeclared combination, for a declaration that names an unknown id, and for a declaration
that matches no site. A patch that another patch supersedes is removed, not left underneath.

Categories: `Plumbing` (engine / UI / IO leaf that cannot run headlessly, or an outbound network call; no game state,
no RNG), `Scheduling` (timer / deferred call / synchronously completing save -> pump; an order the real client can
produce), `Instrumentation` (read-only), `Fidelity` (**restores the real-game branch** of a TestMode / UI gate),
`Behaviour` (changes what the game computes; **the production set must contain none** - checked at every start).

**The production set and tool instrumentation.** The records present when the session was created are the production
set (`HostSession.ProductionPatches`): identical in every mode, hashed (`ProductionPatchHash`). A verification command
may register read-only instrumentation of its own afterwards (`ToolPatches`; ruling B4): the thread audit, the replay
tool's checksum interventions and coverage probe, the combat cross-check's probes. Nothing else may be registered
after the session exists.

**FIDELITY.md.** ONE ledger, `host\FIDELITY.md`: one row per registry id (`| \`id\` | category | kind | target |
what / why | self-check |`), production rows in section 1, tool-only rows in section 2. The tables are generated
from the registry (`bench\h1-integration\tools\gen_ledger.py`) and checked against it by the executable
(`--check-ledger`, `--list-patches`, `--patch-report`, the fidelity and replay self-tests, `make-dist`):
`SpireHost.Core\Patches\PatchLedger.cs`. A `Fidelity` row states: site, what the TestMode branch does, what the real
branch does, how it is patched, and the self-check. A UI-only flow that the host performs by proxy gets a row in the
ledger's section 6 with the UI source (`Nodes.../N*.cs:line`) and the game calls made. Adding a patch = add the row
(regenerate), or the host does not pass its own check.

**No network.** The host never performs network IO. The game's telemetry upload and time-server request are blocked
at their entry points; reaching one is a strict violation (`network_attempt`).

**Naming.** Handler / frame / lifecycle / check names: `<module>:<thing>` (`rooms:reward`, `combat:card-select`,
`fidelity:invariants`). Namespaces = project names.

**Performance.** Numbers of the integrated build (decisions per second per process in fuzz and serve mode, cost of the
per-settle checks and of the encoder) are in `notes\host-h1-integration.md` (cost table). Keep `Step` / `Build`
allocation-free where practical (reuse `Decision`, lists, contexts). A run start costs about 40 ms, most of it in the
game's own map generation.

---

## 11. The temporary pieces (removed)

`host\SpireHost.Driver\Temporary\TemporaryHandlers.cs` and `HostOptions.TemporaryHandlers` no longer exist (ruling
B2). Every slot the stage-0 stand-ins filled has its owner:

| Slot | Owner |
|---|---|
| room handler `Combat`; the run's card selector (`ctx.CardSelector`, lifecycle `combat:lifecycle`) | combat |
| lifecycles `rooms:rewards`, `rooms:treasure`, `rooms:potions-state`; room handlers `Treasure`, `Rest`, `Shop`; frame extension `rooms:potions` | rooms |
| room handler `Event` (events, Ancients, Neow, TheArchitect, FakeMerchant, Crystal Sphere, Trial, bundles) | events |
| room handler `Map` | driver core |

`ctx.CardSelector` is still the one shared object: the combat lifecycle installs the run's selector in `OnRunSetUp`
and the rooms card-reward selector forwards `GetSelectedCards` to it.

---

## 12. Multi-seat runs (phase H5); what is still open

Design note with the reasons: `notes\host-h5-design.md`. Ledger: `host\FIDELITY.md` section 11. Report with every
number: `notes\host-h5-4p.md`. Observation format: `host\SpireHost.Obs\SCHEMA-ERRATA-H5.md`.

### 12.1 What a multi-seat run is

`EpisodeSpec.Characters` with 2-4 entries (the same character may repeat; seat = index). The run is the game's real
multiplayer run - `RunManager.SetUpTest(run, service, disableCombatStateSync: false)` + `GenerateRooms()` - on the
host's `HeadlessHostNetService` (`RunContext.Mp.Net`; `NetGameType.Host`, no transport). Seat 0 is the local player
and uses every path of the single-seat host. Seats 1..N-1 are remote players: `RunContext.IsRemoteSeat(seat)`,
`RunContext.Mp.Client(seat)` is their `VirtualClient`. Net ids: seat 0 = 1, seat k = 1000 + k
(`MultiSeatRun.NetIdOfSeat`). `RunContext.Mp` is null in a single-seat run.

**The rule of the whole phase**: whatever the player of a remote seat does must reach the game as the message that
player's real client sends, with that seat's net id as sender - never as a direct call of a `*Local*` API, never
through a per-player selector. `VirtualClient`:

| Method | Client message |
|---|---|
| `SendAction(GameAction)` | `RequestEnqueueActionMessage` (the action must be owned by the client's player) |
| `SendPotionUse(potion, target, inCombat)` | the client's `PotionModel.EnqueueManualUse`: target completion, its own "queued" flag (`RunContext.IsPotionQueued`), then the action |
| `SendChoice(choiceId, PlayerChoiceResult)` | `PlayerChoiceMessage` |
| `SendHookEnqueue`, `SendResume`, `SendPlayerData` | sent by the core's observation points (`RemoteSeatPatches`, `MultiSeatRun`); modules never call them |
| `Send<T>(message)` | any room message (`OptionIndexChosenMessage`, `VotedForSharedEventOptionMessage`, `RewardSetSkippedMessage`, `RestSiteSkippedMessage`, `CrystalSphereRewardsMessage` ...) with `client.Location` |
| `SendTracked<T>(message)` | a room message whose host handler starts the player's work fire-and-forget (`RewardSelectedMessage`, `MerchantEntryPurchasedMessage`, `OptionIndexChosenMessage{RestSite}`, `TreasureChestOpenedMessage`): returns that task; track it with the seat |

### 12.2 Decision order with several seats (`RunDriver.NextMultiSeatDecision`)

| | Condition | Who builds the decision |
|---|---|---|
| T | `run.IsGameOver` | terminal (all players dead = Loss; seat 0's last Architect button = Win) |
| P1 | the decision stack is not empty | the frame on top. A remote seat's prompt is a frame like seat 0's (12.3) |
| P1' | a remote seat has an overlay screen open | `IRemoteOverlay.Step(ctx, seat)`, lowest seat first (the rooms module's reward screens of remote seats) |
| P2 | a combat is in progress | the Combat handler, once: the union of all seats' actions (`ActingSeat` -1 or the only seat that can act) |
| P3 | for seat 0, 1, ...: the seat is not ready for the next act, has no incomplete blocking operation of its own (or of no particular seat), and its screen (`RunDriver.ClassifySeat`: Map once `ctx.IsMapOpen(seat)`, else the room's screen) has a decision | that screen's handler, `Step(ctx, seat)`. `Waiting` = look at the next seat |
| P4 | no seat had a P3 decision and a handler answered `Vote` | the lowest such seat's vote (the map vote; the page vote of a shared event) |
| - | nothing of the above | `Stall/all_seats_waiting`, with what every seat waits for |

* `HandlerStep.Waiting(why)`: the seat cannot act until a teammate did (it voted; the treasure award waits).
* `HandlerStep.Vote(decision)`: the seat's TEAM VOTE - a vote whose result is one draw from all votes. Asked only at
  P4. With `EpisodeOptions.TeamVotes` the handler asks it once and casts the same vote for each other seat on that
  seat's next `Step` (returning `Progressed`), each as that seat's own message; a seat for which the option is not
  legal is asked itself. The treasure relic vote is an ordinary decision of the seat's chain (the seats vote for
  different relics); the act-change vote is the terminal PROCEED of the boss reward screen (`ctx.SetActReady`).
* `ctx.OpenMapFor(seat, why)` is what a remote seat's PROCEED / LEAVE does (a client sends nothing); seat 0 keeps the
  game call `RunManager.Proceed()` (7.3). `ctx.IsMapOpen(seat)`, `ctx.IsActReady(seat)`, `ctx.SetSeatFlag` /
  `SeatStatusBits` / `SeatScreen` (public per-seat status for the encoder) are reset at every room entry.
* The room-level frame extensions (6.6: out-of-combat potions) are offered every single-seat frame with its
  `ActingSeat`; they apply a remote seat's action through its client.

### 12.3 Prompts of remote seats: `IRemoteChoiceRouter`

The game asks a remote player by waiting: `PlayerChoiceSynchronizer.WaitForRemoteChoice(player, choiceId)`. The core
observes every such wait (`mp-remote-choice-wait`) and offers it to the registered routers in order
(`DriverRegistry.SetRemoteChoiceRouter`). A router that recognises the prompt from what it captured about the game
call parks a frame (`ctx.Push`) and returns true; the frame answers with `client.SendChoice(choiceId, result)`,
encoded as the local branch of the same call site encodes it. A router returns false for a wait that is not its own;
a wait nobody claims aborts the episode (`HandlerFault/remote_choice_unrouted`). `TryRoute` runs inside the game
call: no game work, no exception for "not mine". Registered: `combat:card-select-remote` (every `CardSelectCmd.From*`
prompt), `rooms:mend-target`, `rooms:remote-reward` (card reward), `events:bundle-remote`.

### 12.4 Options, tools, tests

`EpisodeOptions`: `TeamVotes` (default false; `--team-votes`; the protocol server: on unless RESET flag bit 6),
`OfferUndoEndTurn` (default true; `--undo false`; the protocol server: off unless RESET flag bit 7),
`WireRoundTrip` (`--wire`: every client message through the game's packet serialiser first; verification mode),
`DebugForcedHpPerSeat` (`--hp-seats a,b,c`). Tools: `--seats N` / `--lineup A+B+C` on `--fuzz-runs`, `--determinism`,
`--replay-run`, `obs --golden`. Self-tests: `--selftest-mp` (cases in `Tools\MultiSeatSelfTest.cs` and
`Tools\MultiSeatRoomTests.cs`; `--cases trace --seed S --lineup L --policy-seed P [--hp N]` lists the last decisions
of one fuzz tuple), `obs --selftest --cases frames_mp,protocol_mp,leak_team`, `obs --leak-test --seats N`.

### 12.5 Still open

| Item | Phase | Notes |
|---|---|---|
| Comparison with a real 2-4 client recording (`oracle\runs-mp`, format `notes\oracle-mp.md`) | verification | not built: plan in `notes\host-h5-4p.md`. Until then every injection point is derived from the decompiled client, not from an observed one |
| SEAT view of the encoder (one stream per seat) | - | a multi-seat RESET in SEAT view is refused |
| A hand prompt whose hand changes while it is open and still has selectable cards (several players' prompts open at once) | H5 follow-up | fails the episode (`card_select_hand_changed`); 0 occurrences so far. The emptied-hand / combat-over case is reproduced |
| Combat checksums as an episode option (replay R2), `SyncLocalChoice` for seat 0's prompts (combat C10) | H4 | unchanged from H1 |
| Save / restore of a run inside the host, room-boundary save hashes | H4 | out of scope |
| Real-client ground truth for hidden intents during the enemy turn (KnowledgeDemon) | verification | unchanged from H1 |

---

## 13. Commands and exit codes

One executable: `host\dist\<stamp>-<label>\SpireHost.exe` (development build: `host\out\bin\SpireHost\release\SpireHost.exe`).

```
SpireHost --<command> [options]                 top-level commands
SpireHost <area> --<command> [options]          area = combat | rooms | events | fidelity | obs | replay
```

Every command runs on the production configuration (2.3). The complete reference with every option is
`host\README.md`; the list:

| Command | Purpose |
|---|---|
| `--serve [--pipe NAME] [--max-decisions N] [--turn-cap N] [--decision-cap N] [--verbose] [--allow-debug-options [--hp N]]` | the binary protocol server (obs-schema-v1 section 7 + errata P) on stdio or a named pipe. Its command line is checked against an allow-list before anything boots: an option that is not listed (or a switch followed by a value that is not a boolean literal) is refused with exit code 1. A switch written with `false` / `0` / `off` / `no` is off. Debug options: below |
| `--fuzz-runs [--runs N \| --seconds T] [--characters ..] [--ascensions 0,10] [--seed-base S \| --seed S] [--policy-seed P] [--start-index I] [--hp N] [--turn-cap N] [--max-decisions N] [--signatures] [--record-dir DIR] [--out FILE] [--seats N \| --lineup A+B+C] [--team-votes] [--undo false] [--wire] [--hp-seats a,b,c]` | N whole runs with `RandomLegalPolicy`; JSON: outcomes, per character x ascension, aborts grouped by signature with repro tuples, auto-advance counts, strict counters, module counters, the coverage table, time split, decisions per second (wall and CPU). Exit 4 if any run aborted |
| `--determinism [--runs N] [--characters ..] [--ascensions ..] [--hp N] [--controls false] [--out FILE]` | each run twice (identical traces and module counters required) + negative controls; `trace_hashes` for the cross-process comparison. Exit 5 on mismatch or abort |
| `--replay-run --record FILE --seed S --character C [--ascension A] [--policy-seed P] [--hp N]` \| `--log FILE` \| `--selfcheck N` | record one random run / replay a log with `ScriptedPolicy`: every per-decision signature, the outcome and the final floor must be reproduced; a log that does not name its vocabulary and game build, or names another one, is refused. Exit 5 on mismatch |
| `--replay-mcr [--file F \| --dir D \| --run-traces \| --runs-v2 [sets] \| --runs-mp [corpus\|deep\|SESSION] [--mp-clients 0\|all]] [--stride S --offset K] [--coverage] ...` | real-client `.mcr` combat replays re-executed, the game's own post-action checksums compared. Exit 5 unless every replay passes. Empty recordings and fights started by an event option are listed, never counted. `--runs-mp` (phase H5): the fights of the real 2-4 player sessions in `oracle\runs-mp`, re-executed with the game's own multi-player replay sequence (checks the engine under the headless patch set with several players; it does not go through the Host-type service of section 12) |
| `--compare-run-traces [--dir D] [--traces a,b] ...` | run start and room-boundary parity against the golden run traces (format v1). Exit 5 unless every trace matches to its end |
| `--compare-run-traces-v2 [--dir D \| --set natural\|deep] [--traces a,b] [--stride S --offset K] ...` | every recorded decision of the v2 run traces (`oracle\runs-v2`): legal-action oracle, offers, state, dumps and saves, fights inside the run with the recorded per-action combat states. Exit 5 unless every trace matches to its end with no unexplained divergence (`notes\host-v2-parity.md`) |
| `--export-vocab DIR --merge OLD_VOCAB_JSON` | append-only vocabulary + catalog export into a NEW directory |
| `--selftest-driver` | the 41 contract cases of section 5. Exit 4 on failure |
| `--selftest-mp [--cases a,b] [--seats 2,3,4] [--episodes N] [--shard I --shards K] [--wire]` | the multi-seat self-test (12.4): fights (handshake, UI rules, death / revive / wipe, prompts, scaling against the decompiled formulas, determinism) and room flows (rest incl. MEND, treasure, shop, crystal sphere, bundle, shared events, isolation, whole runs). Exit 4 on failure |
| `--driver-info`, `--boot`, `--list-encounters` | registry, modules, profile, vocabulary, patch counts / boot report / encounter ids |
| `--list-patches`, `--check-ledger [--ledger FILE]` | the production patch set by category with its hash and shared sites; registry == ledger. Exit 4 on a Behaviour patch or a mismatch |
| `--verify-dist`, `--write-dist-manifest --label L --stamp S` | dist verification as a command; the manifest writer used by make-dist |
| `combat --selftest [--suite ..] [--seeds N]`, `combat --selftest --suite sweep [--hp N] [--shard I --shards N]`, `combat --crosscheck-mcr [--run-traces]` | combat module tests; the encounter sweeps; the handler against the real client's recorded decisions |
| `rooms --selftest [--characters all] [--group G] [--scenario S]`, `rooms --tree-runs`, `rooms --trace-run` | rooms scenarios; proceed-tree whole runs |
| `events --selftest [--skip-cases] [--skip-sweep] [--brief]` | event cases + the all-events sweep |
| `fidelity --selftest [--runs N]`, `fidelity --site-table`, `fidelity --scan` | fidelity checks; every non-presentation audit site -> handled by; static scans |
| `obs --selftest [--cases a,b]`, `obs --leak-test`, `obs --golden DIR`, `obs --sample-frames DIR`, `obs --bench-encode` | encoder / protocol / leak tests; frames for the Python side; encoder cost |
| `replay --selftest`, `replay --replay-determinism` | the oracle tools' own checks |

Process-level options (any command; `--serve` accepts all of them except `--audit-threads`): `--game-dir DIR`, `--vocab-dir DIR`,
`--game-log off|warn|info|debug`, `--patch-report FILE` (after the command: production patch list + hash, the tool
instrumentation the command added, registry == ledger), `--verbose`, `--audit-threads [--audit-out FILE]` (test
instrument: exit 6 unless every audited gameplay call ran on the engine thread).

**Debug options in serve mode** (ruling A4). `--hp N` on the command line and RESET word 15 (a non-zero value = forced
HP for that episode) are debug options. They are honoured only by a server started with `--allow-debug-options`;
without the flag `--hp` is refused before boot (exit code 1) and a RESET whose word 15 is not 0 is answered with ERROR 3 (bad message)
and starts no episode. A RESET whose flags word has a bit outside bits 0-5 (bits 0-7 for a RESET with 2-4 seats,
`SCHEMA-ERRATA-H5.md`) is refused the same way in every server; so is a multi-seat RESET in SEAT view.
A production client always sends 0 in word 15 (errata P-2: reserved).

Exit codes: 0 ok; 1 usage; 2 unexpected host exception; 3 startup error (game directory / SHA pin, vocabulary, dist
verification, configuration check); 4 a self-test or check failed / an episode aborted; 5 a determinism, replay or
oracle comparison did not match; 6 thread audit not clean.

Multi-process runs: `bench\h1-integration\tools\fuzz_mp.py` (fuzz shards + aggregation) and
`bench\h1-integration\tools\runpar.py` (any commands under a process cap, with a free-memory floor).

---

## 14. Changing this contract

The public surface of `SpireHost.Core` and `SpireHost.Driver` is the contract. If you need something that
is not there - a context field, a `RefKind`, a hook - do not work around it:

1. write the request (what, why, which spec paragraph) to the Core/Driver owner;
2. additive changes (new field, new enum value at the end, new optional parameter, new method) are made in
   the live projects and listed below;
3. a change of meaning or signature of an existing member needs the lead's ruling, an entry below, and a
   coordinated rebuild; `DriverApi.Version` changes with it.

| Version | Change |
|---|---|
| `h1-stage0` | initial contract (snapshot for the six parallel teams) |
| `h5-4p` | Runs with 2-4 seats in one process (section 12). **Changed signature**: `IRoomHandler.Step(RunContext ctx, int seat)` (was `Step(RunContext ctx)`; every handler was updated; seat is 0 with one seat). **Added**: `StepKind.Waiting` / `Vote`, `HandlerStep.Waiting` / `Vote`; `IRemoteOverlay` + `DriverRegistry.SetRemoteOverlay`; `IRemoteChoiceRouter` + `SetRemoteChoiceRouter`; `RunContext.Mp`, `IsRemoteSeat`, `IsPotionQueued`, `IsMapOpen`, `OpenMapFor`, `IsActReady`, `SetActReady`, `SetSeatFlag`, `ClearSeatFlag`, `SeatStatusBits`, `SeatScreen`, `Track(task, what, seat)`, `HasBlockingOperationFor`; `SeatStatus`; `HeadlessHostNetService`, `VirtualClient`, `MultiSeatRun`, `MultiSeatStats`; `EpisodeOptions.TeamVotes`, `OfferUndoEndTurn`, `WireRoundTrip`, `DebugForcedHpPerSeat`; `EpisodeResult.Characters`, `FinalHps`; `MapContext.SeatVotes`, `TreasureContext.SeatVotes`, `EventContext.SeatVotes`; `DriverApi.MaxSeats`; `CardSelectStats.RemoteFrames` / `RemoteAnswers` / `RevalidatedEmpty`; commands `--selftest-mp`, `--seats` / `--lineup` / `--team-votes` / `--undo` / `--wire` / `--hp-seats`, `obs --leak-test --seats N`, `obs --selftest --cases frames_mp,protocol_mp,leak_team`; RESET `n_seats` 2-4 (TEAM view) and flag bits 6 / 7 for it. **Changed meaning**: `EpisodeSpec.Characters` with 2-4 entries starts a multiplayer run (was refused); the stand-in for `Callable.CallDeferred` also runs parameterless delegates that return a value; the rebuild of a prompt's candidate list and the encoding of its answer are cross-checked on every local prompt (two new `HandlerFault` details, single-seat runs included); `ShopState` / `RestState` / `TreasureState` are indexed by seat. Production patch set: see `--list-patches` (the new ids: `mp-*`, `rooms-remote-*`, `events-bundle-remote-capture`, `events-crystal-sphere-remote`, `fidelity-mend-local-targeting`) |
| `h1-v2parity` | The host after the comparison with the real client's v2 run traces and the four verification reports (`notes\host-v2-parity.md`). **Added**: `IEpisodeCheck.OnEpisodeBegin` (default method; before the run is created); `StrictCounters.RunSaveBoundaryNotApplied` (`run_save_boundary_skipped`) and `StrictCounters` `operator +`; the engine bound `combat_work_without_decision` (`LivelockGuard.CombatWorkLimit`, `EngineLivelockException.Detail`); `EnginePump.EpisodeRunning`; `Args.SwitchValueError`; `PatchLedger.CompareLines / ParseLines / Cell` and `Row.Target / What`; `EpisodeOptions.DebugWongoPoints` and `ICardSelectHostView.PreferInstance` (tool only); `--compare-run-traces-v2`, `--replay-mcr --runs-v2`, `obs --leak-test --label / --policy-seed`. **Changed meaning**: the decision cap also ends an episode at a nested decision (exactly N policy decisions); `Args.Has` is false for a switch written with a false literal; `--replay-run --log` refuses a log that does not name its vocabulary and game build; the fidelity backstop's baseline is taken before the run set-up and the run-level switches are checked at every quiescent point; `EnginePump.DrainUntil` throws during an episode; a locked event option carries no hover-tip entities and no kill warning in `EventContext`; `--replay-mcr --alter-event` on a non-play is an error and a signature mismatch is a failed replay; the ledger check compares target and text; the v1 trace comparison's HP tolerance is the combat-entry sampling rule. **Removed**: `PatchRegistry.Harmony` as a public member (internal now). Production patch set: 97 (+ `livelock-guard-combat-work`) |
| `h1-integrated` | One executable, one production module set, no temporary handlers, no Snapshot build mode, output root `host\out`, dist directories. **Added**: decision contexts moved to `DecisionContexts.cs` and completed (`CombatContext`, `CardSelectContext` C2 fields, `TreasureContext.Finished / Empty / Awarded`, `EventContext` / `EventOptionInfo`, `CrystalContext` public view, `ConfirmKeys`), `NotReady`, `ICardSelectHostView`, `LocText`, `Decision.VerbMask` / `SealForTest`; registry slots for episode checks, module counters, the canonical card-class provider and the vocabulary identity; `IEpisodeCheck`, `IModuleCounters`, `CanonicalCardClasses`; `IEpisodeObserver.OnRoomBoundary` (default method); `EpisodeResult.ModuleCounters`, `EpisodeCounters` coverage arrays and public-history counters; `StrictCounters.NetworkAttempts` (`network_attempt`), `DiscoveryRoomsChanged`; `RunContext.CanonicalCardClasses`, `IntentPerformed`; `Scenario.Exempt`; `ScenarioTools` additions; `HostSession.ProductionPatches / ProductionPatchHash / ToolPatches`; `EnginePump.DrainUntil`; `PatchRegistry.DeclareSharedSite / SharedSites / AssertOnePatchPerSite`, `PatchLedger`; `VeteranProfile.WongoPointsFor`; `DriverApi.Version`. **Changed meaning**: `RunManager.Proceed()` opens the host map (core postfix) - owners call the game method instead of `OpenMap` where the UI does; decision and turn caps also apply inside the run start chain; the canonical order of pile-sourced CARD_SELECT candidates is the registered provider's in every mode; `ForceDiscoveryOrderModifications` follows the real client's condition; `--determinism` is always run-level and compares module counters; episode logs carry the vocabulary hash. **Removed**: `HostOptions.TemporaryHandlers`, `BootOptions.Profile / VirtualClock / FidelityGates / LivelockGuard`, `DriverRegistry.HasTemporary`, `SpireHost.Sim.CombatHarness` and the H0 commands (`--selftest`, `--selftest-stuck`, `--bench-combat`, combat-level `--determinism`, `--run-probe`), `TeamConsole`, `--strict false`, `--temporary-handlers`, `--list-patches --with-optional`; patches `virtual-clock-executor-unpause-poll`, `virtual-clock-creature-spawn-timeout`, `selection-context-choose-a-card` |
