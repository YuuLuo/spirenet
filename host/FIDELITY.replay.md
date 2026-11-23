# SpireHost fidelity ledger - fragment of the replay team (H1)

> **MERGED into `host\FIDELITY.md` at the H1 integration (2026-11-15).** This fragment is kept for history. It describes the
> team's stand-alone build against the stage-0 snapshot and is no longer checked against the registry: ids, counts,
> switches and "temporary" / "shim" remarks below may be out of date. The merged ledger is the source of truth
> (its section 4 lists what was removed or moved at the integration).

Owner: replay team, project `host\SpireHost.Replay` (module `replay`, `SpireHost.Replay.ReplayModule`).
Same format as `host\FIDELITY.md`; the lead merges this fragment at integration.

**Nothing in this fragment is part of the default patch set.** `ReplayModule.Register` registers nothing. Every
entry below is installed only by one of the replay team's *verification commands*, inside that command's own
process, and only when the command (or its flag) is used:

| Command / flag | Installs |
|---|---|
| `--replay-mcr`, `--replay-determinism`, `--selftest` | `replay-checksum-tracker-enabled`, `replay-post-action-checksum`, `replay-combat-state-sync-disabled` (per replay run the tool sets up) |
| `--replay-mcr --coverage` | `replay-coverage:<site>:<method>` (one per instrumented audit site, ~150) |
| `--prototype-fixes <ids>` / `all` (`--compare-run-traces`, `--replay-mcr`) | `replay-prototype-*` (prototypes of patches OWNED BY THE FIDELITY / EVENTS TEAMS - evidence only, never shipped) |
| `--compare-run-traces` | driver lifecycle `replay:trace-dump` (no game patch) |

Every Harmony patch goes through `PatchRegistry` (`Prefix` / `Transpiler` / `GatePatch.ForceGetter`), fails closed
when its target is missing, and transpilers check "exactly N hits" (coverage counters: >= 0, see C1).
Source references: `ref\sts2-v0113-src`, `Namespace/File.cs:line` with the `MegaCrit.Sts2.Core.` prefix omitted.
Evidence: `notes\host-h1-replay.md`, raw results `bench\h1-replay\`.

---

## A. Interventions of the replay tools (A3: game level, no Harmony)

| Id | Target | Kind | Category | What / why (source reference) | Self-check |
|---|---|---|---|---|---|
| `replay-checksum-tracker-enabled` | `ChecksumTracker.IsEnabled` (public setter) | property write, replay run only | Instrumentation | `RunManager.InitializeShared` sets `ChecksumTracker.IsEnabled = !TestMode.IsOn` (Runs/RunManager.cs:385). The tool sets it back to true right after `SetUpReplay` (notes\oracle-mod.md 10.3). The tracker only reads state (`NetFullCombatState.FromRun`, Entities.Multiplayer/NetFullCombatState.cs:287) and counts ids (`NextId`, not game state). | `--replay-mcr`: 77,112 / 77,112 corpus checksums generated with ids equal to the recording's (`checksums_generated == checksums_recorded`, no id outside the recording). |
| `replay-post-action-checksum` | `ActionExecutor.BeforeActionExecuted` (public event) | event subscription, replay run only | Instrumentation | The real (interactive) branch of `ActionExecutor.ExecuteActions` subscribes `readyAction.JustBeforeFinished += JustBeforeActionFinished` (GameActions/ActionExecutor.cs:127) and removes it after the action (:158); the NonInteractive branch taken under TestMode (:118-123) does not, so `RunManager.SendPostActionChecksum` (Runs/RunManager.cs:403, 485-491) never runs and the "finished action execution <action>" checksums (most of every .mcr) are missing. The tool subscribes the executor's own private handler from `BeforeActionExecuted` (raised at :117, immediately before the real branch would subscribe), first removing a previous subscription so that an action resumed after a player choice is subscribed once - exactly the real branch's life cycle. The game's handler runs with its own guards (`CurrentlyRunningAction == action`, `State == Finished`). Read-only w.r.t. game state. | 49,707 subscriptions over the corpus; every post-action checksum of the recording is reproduced (id and value). |
| `replay-combat-state-sync-disabled` | `CombatStateSynchronizer.IsDisabled` (public setter) | property write, replay run only | Plumbing | The replay run is set up exactly like the real client's replay facility: `RunManager.SetUpReplay` (Runs/RunManager.cs:334-353, `NetReplayGameService`) + `CombatStateSynchronizer.IsDisabled = true` (Nodes.Debug.Multiplayer/NMultiplayerTest.cs:417-419; the oracle's `ReplayCheck.Rerun` does the same). No peer exists to sync with. | same run as above |
| (lifecycle) `replay:trace-dump` | `DriverRegistry.SetLifecycle` | driver lifecycle, `--compare-run-traces` only | (not a patch) | Forwards `RunManager.RoomEntered` to the trace comparison so that TREASURE rooms are dumped before the host opens the chest unasked (the temporary treasure handler forces `DoNormalRewards` before the first decision; the oracle dumps before clicking the chest). No game call. | 118/120 traces match every room boundary (report section 4). |

## C. Optional instrumentation

### C1. `--coverage` (read-only)

| Id | Target | Kind | Category | What / why | Self-check |
|---|---|---|---|---|---|
| `replay-coverage:<File.cs:line>:<method>` (family; one id per instrumented method) | every method that contains a TestMode / NonInteractive gate listed by `notes\testmode-audit-core.json` or `testmode-audit-models-commands.json` (170 non-Node sites; 159 instrumented; static constructors and gates inside lambdas / already replaced by a default patch are reported as not instrumented) | transpiler (async: MoveNext) | Instrumentation | Each call of `TestMode.get_IsOn` / `get_IsOff` / `NonInteractiveMode.get_IsActive` / `AutoSlayerCheck` in the method is replaced by a wrapper that increments a per-method counter and returns the SAME value. Gives, per audit site, how many replays evaluated the gate (report section 2.4). | per-site counts in the `--replay-mcr --coverage` JSON; semantics unchanged: the corpus run with and without `--coverage` gives the same 1,488 passes and identical checksums. |

## P. Prototypes, NOT shipped (patch ownership: fidelity / events teams)

Installed only with `--prototype-fixes`. Each reproduces the fix the owning team planned for a divergence the
oracle found, to prove the divergence disappears; the finding and the evidence are in `notes\host-h1-replay.md`
section 4. The owning team's own patch (FIDELITY.fidelity.md) is what will be shipped.

| Id | Site | TestMode / headless branch | Real branch | Prototype | Evidence |
|---|---|---|---|---|---|
| `replay-prototype-runmanager-1592` | `RunManager.UpdatePlayerStatsInMapPointHistory` (Runs/RunManager.cs:1590-1594) | returns at once | writes current_gold / current_hp / max_hp of every player into the current map-point history entry | `GatePatch.ForceGetter`: exactly 1 `TestMode.get_IsOn` -> false | without it 0/120 traces get past their first room (`save.map_point_history[][].player_stats[].current_gold`); fidelity: `fidelity-map-history-player-stats` |
| `replay-prototype-merchantpotion-62` | `MerchantPotionEntry.CalcCost` (Entities.Merchant/MerchantPotionEntry.cs:55-64) | base price, no RNG draw | one `PlayerRng.Shops.NextFloat(0.95, 1.05)` per potion; the card entries are populated AFTER the potion entries (Entities.Merchant/MerchantInventory.cs:117-143), so every shop card / price differs | `GatePatch.ForceGetter`: exactly 1 `TestMode.get_IsOff` -> true | without it 61 traces stop at their first shop (`dump.shop[].cost`, `players[].rng.rngs.shops.counter` 42 vs 39); fidelity: `fidelity-merchant-potion-price` |
| `replay-prototype-merchantroom-exit-70` | `MerchantRoom.Exit` (Rooms/MerchantRoom.cs:66-73) | returns at once | `MerchantSynchronizer.OnMerchantExited` (shop card / relic / potion choices into the map-point history) | `GatePatch.ForceGetter`: exactly 1 `TestMode.get_IsOn` -> false | without it 61 traces stop after their first shop (`map_point_history[][].player_stats[].card_choices`); fidelity: `fidelity-merchant-room-exit` |
| `replay-prototype-event-option-title-vars` | `EventSynchronizer.SaveEventOptionToHistory` (Multiplayer.Game/EventSynchronizer.cs:245-264) <- UI-only `NEventOptionButton._Ready` (Nodes.Events/NEventOptionButton.cs:181-182) | headlessly no option button exists, `Event.DynamicVars.AddTo(Option.Title)` never runs, the history entry's variables stay empty (JSON: `variables` omitted) | the button adds the event's dynamic vars to the option title / description when the option is shown; the synchronizer copies them into `event_choices[].variables` | prefix on `SaveEventOptionToHistory`: the button's two `AddTo` calls for the chosen option | without it 102 traces stop at their first non-Neow event (`event_choices[].variables`); NOT in any audit (no TestMode gate) - finding for events / fidelity (section D proxy) |
| `replay-prototype-creaturecmd-382-onended` | `CreatureCmd.Kill` (Commands/CreatureCmd.cs:378-389) | only `LoseCombat`; `OnEnded(false)` skipped | `if (!HasRunEnded) OnEnded(false)` + game-over screen | `--compare-run-traces` calls `RunManager.OnEnded(isVictory: false)` itself at the terminal state of a run lost by death, before the end dump (the real branch's game call, no presentation); registered as a Note | without it 118 traces fail at their end dump (`map_point_history[][].player_stats[].current_gold`, `rooms[].turns_taken`); fidelity: `fidelity-natural-death-on-ended` |
| `replay-prototype-rollingboulder-per-target` | `RollingBoulderPower.AfterPlayerTurnStart` (Models.Powers/RollingBoulderPower.cs:36-52) | one `DoDamage` over all `HittableEnemies` | one `DoDamage` per creature hit by the VFX (Nodes.Vfx/NRollingBoulderVfx.cs) | transpiler: exactly 1 `DoDamage` call -> per-target sequential helper (the fidelity team's planned shape) | NOT a fix of a found divergence: tests the plan. Corpus replays 01124, 01132, 02269 (17 boulder triggers, 7 with two hittable enemies) pass with AND without it - the corpus does not discriminate. Not part of `--prototype-fixes all`. |

## D. UI-only flows (by proxy) found by the oracle

| Flow | UI source | Effect headlessly | Status |
|---|---|---|---|
| Event option title / description variables | `NEventOptionButton._Ready` (Nodes.Events/NEventOptionButton.cs:181-182): `Event.DynamicVars.AddTo(Option.Description)`, `AddTo(Option.Title)` | run-history bookkeeping only: `map_point_history[].player_stats[].event_choices[].variables` empty | Open - events / fidelity team (prototype `replay-prototype-event-option-title-vars` proves the fix). |
| Room-transition potion lock | `RunManager.FadeIn` (Runs/RunManager.cs:908-922, TestMode gate classified presentation) keeps `IsChangingRooms` true until `NGame.Transition.RoomFadeIn` finishes; `RunManager.IsPotionUseLocked` (:223-231) | in the real client potions cannot be used / discarded while the room fades in, although cards can already be played (11,890 collector decisions in 1,047 corpus replays had no potion option, always a prefix of the combat); the host has no fade, potions are usable at the first decision | Accepted difference (no time passes headlessly; a human cannot act on the faded screen). Documented for the combat / obs teams. |
