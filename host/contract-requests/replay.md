# Contract requests of the replay team (H1)

## Status after the H1 integration (2026-11-15, contract version `h1-integrated`)

The lead's rulings are in the integration brief; how each was implemented: `notes\host-h1-integration.md`. "Shim removed" means the team's local stand-in described further down no longer exists. The text below the table is the original request, kept for history.

| Request | Subject | Status | Where / how |
|---|---|---|---|
| R1 | Pump loop for verification tools | **ACCEPTED, implemented** | `EnginePump.DrainUntil(predicate, maxCallbacks, virtualBudget)` + the documentation sentence (DRIVER-API.md section 3). `McrReplayer.PumpUntil` uses it. |
| R2 | Combat checksums as an episode option | **DEFERRED to H4 (ruling)** | Not small and not risk-free (needs the choice recording of combat C10). The replay tools keep their own replay run. Open item: DRIVER-API.md section 12. |
| R3 | Room-boundary callback | **ACCEPTED, implemented** | `IEpisodeObserver.OnRoomBoundary(RunContext)` (default interface method): first quiescent point after run start and after every `RoomEntered`, before the handler's first Step. `RunTraceCompare` dumps there. |
| R4 | Public WongoPointsFor | **ACCEPTED, implemented** | `VeteranProfile.WongoPointsFor(seed)`; the run driver uses it. (The combat replay tool keeps `Restore(0)`: a combat replay never reads the value.) |
| R5 | Treasure dump point | **INFORMATION, satisfied** | the CHEST page is a decision (rooms); the `replay:trace-dump` lifecycle is removed (ruling D2). |
| prototype fixes | the six `replay-prototype-*` patches | **REMOVED** | what they prototyped is in the production set; `--compare-run-traces` reaches 120 / 120 with no prototype fix and no test switch. |

---
From: replay team (`host\SpireHost.Replay`). To: Core/Driver owner / lead. Context: `notes\host-h1-replay.md`.
Nothing here blocks the replay team: every item has a local solution inside `SpireHost.Replay` (named below).
All requests are additive (DRIVER-API.md section 14, point 2).

---

## R1. Allow (and give a helper for) a combat-level pump loop in verification tools

* **What.** The `.mcr` replay harness drives the engine itself outside any `RunDriver` episode: it injects the
  recorded events through the game's replay facility (`NMultiplayerTest.RunReplay` call sequence) and has to "wait
  a frame" between events. It does so with `EnginePump.TryRunUntilQuiescent(budget)` + `VirtualClock.FireNext()`
  until a predicate holds (`McrReplayer.PumpUntil`).
* **Why.** DRIVER-API.md 3 / 8.2 say "never pump" for driver code; section 9 (last bullet) says the H0 harness
  "stays for micro-benchmarks and the replay team". The replay harness is exactly that kind of tool, but the
  existing `EnginePump.PumpUntil` blocks in real time (`_foreignSignal.Wait(stuckTimeoutMs)`) when nothing is
  runnable, which a strict tool does not want (it counts as a `ForeignWaits` real-time wait).
* **Spec.** DECISIONS-H1 H.2; DRIVER-API.md 9 last bullet; 12 "replay".
* **Proposed.** (a) one sentence in DRIVER-API.md 3: "verification tools that run outside an episode (the H0
  harness, `SpireHost.Replay`) may drain the pump themselves"; (b) in Core:
  `public PumpStop EnginePump.DrainUntil(Func<bool> predicate, long maxCallbacks, TimeSpan virtualBudget)` -
  run to quiescence, fire virtual timers while the predicate is false, never wait for another thread; returns
  `Decision` / `Stuck` (+ `LastStuckReason`: `quiescent`, `virtual_time_budget`, `pump_callbacks`).
* **Local shim.** `McrReplayer.PumpUntil` (same semantics).

## R2. Combat checksums as an episode option (for H4 and in-episode replay recording)

* **What.** `EpisodeOptions.CombatChecksums` (bool): inside the episode, enable the game's checksum tracker and
  restore the real branch's post-action checksum subscription; expose the generated `(id, checksum, context)`
  list (and optionally write the game's own `.mcr` via `CombatReplayWriter`).
* **Why.** Room-boundary save hashes and fork verification (DESIGN H4, game-state-fork.md 7.2) need exactly the
  checksum the real client computes; today only the replay tool can produce it.
* **Proposed.** `EpisodeOptions.CombatChecksums`, `EpisodeResult.CombatChecksums : List<(uint Id, uint Checksum, string Context)>?`.
  Implementation = the two interventions of `SpireHost.Replay\ReplayInterventions.cs`
  (`replay-checksum-tracker-enabled`, `replay-post-action-checksum`, ledger `host\FIDELITY.replay.md` A), moved to
  Core and switched per episode (category Instrumentation; read-only, the tracker's `NextId` is not game state).
* **Local shim.** `ReplayInterventions.Scope` (replay runs only).

## R3. A room-boundary callback for observers ("room entered and settled")

* **What.** A callback per room at the point where the real UI has finished the room transition and the player
  could act, BEFORE any work a room handler does unasked (`HandlerStep.Progressed`, e.g. the forced chest opening).
* **Why.** Room-boundary parity against the golden run traces (`--compare-run-traces`) needs the same dump point
  as the oracle (`RunTracer`: `RoomEntered` + `!IsChangingRooms`). Measured: the host's first decision point of a
  room is that point for combat / event / shop / rest rooms (1,256 of 1,277 dumps equal), but not for treasure
  rooms (the temporary treasure handler opens the chest before the first decision: +gold, +1 Rewards RNG draw);
  the plain `RunManager.RoomEntered` moment is too early for combats (monster AI not rolled yet: 48 traces fail
  with `monster_ai.counter`). H4's room-boundary snapshots need a defined point as well.
* **Proposed.** `IEpisodeObserver.OnRoomBoundary(RunContext ctx)` (default interface method, no-op), called by the
  core at the first quiescent point after `RunManager.RoomEntered`, before the room handler's first `Step` and
  before any pending frame is answered. Document it in DRIVER-API.md 4.4 / 5.
* **Local shim.** `RunTraceCompare`: first decision of the room via `IEpisodeObserver.OnDecision`, plus a lifecycle
  `replay:trace-dump` registered by the command (`DriverRegistry.SetLifecycle`) for treasure rooms only.

## R4. Public `WongoPointsFor(seed)`

* **What.** `RunDriver.WongoPointsFor` is `internal`; the replay tool restores the veteran profile before each
  replay and has to pick a Wongo value without it (uses 0; harmless: a combat replay never reads it).
* **Proposed.** `public static int VeteranProfile.WongoPointsFor(string seed)` (or on `HostSession`), same function.
* **Local shim.** `VeteranProfile.Restore(0)` in `McrReplayer.Prepare`.

## R5. (information for the rooms team, no API change) treasure dump point

If the rooms team keeps the forced chest opening as `HandlerStep.Progressed`, R3 is needed for exact trace parity;
if it becomes a single-action TREASURE frame (`Verb.OpenChest`, auto-advanced), the first-decision point already
matches the oracle and the `replay:trace-dump` lifecycle can be removed.
