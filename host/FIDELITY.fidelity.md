# SpireHost fidelity ledger - fragment of the fidelity team (H1)

> **MERGED into `host\FIDELITY.md` at the H1 integration (2026-11-15).** This fragment is kept for history. It describes the
> team's stand-alone build against the stage-0 snapshot and is no longer checked against the registry: ids, counts,
> switches and "temporary" / "shim" remarks below may be out of date. The merged ledger is the source of truth
> (its section 4 lists what was removed or moved at the integration).

Owner: fidelity team, project `host\SpireHost.Fidelity` (module `fidelity`, `SpireHost.Fidelity.FidelityModule`).
Same format as `host\FIDELITY.md`; the lead merges this fragment into it at integration. One row per registry id
this pack installs (22 ids). Machine check: `SpireHost.Fidelity --selftest` (check `ledger_fragment_matches_registry`
compares the backticked ids of this file with `PatchRegistry.Records`); full records: `SpireHost.Fidelity --list-patches`.

Installed by `FidelityModule.Register` (engine thread, after boot, before the first episode). Every Harmony patch
goes through `PatchRegistry` / `GatePatch`, every transpiler checks "exactly N hits" and the process refuses to
start otherwise. Default set with this module: 57 registry entries = Core/Driver 35 + this pack 22; by category
22 plumbing, 9 scheduling, 16 instrumentation, 10 fidelity, **0 behaviour**.

Source references: `ref\sts2-v0113-src`, `Namespace/File.cs:line` with the `MegaCrit.Sts2.Core.` prefix omitted.

---

## A. Default set additions of this pack

### A2. Game level, Harmony

| Id | Target | Kind | Category | What / why (source reference) | Self-check (`--selftest`) |
|---|---|---|---|---|---|
| `fidelity-executor-unpause-signal` | `ActionExecutor.WaitForUnpause` (async MoveNext) | transpiler, exactly 1 delay call replaced | Scheduling | TestMode polls `while (_isPaused) await Task.Delay(50)` (GameActions/ActionExecutor.cs:205-222; Core already routes the delay to the virtual clock). DECISIONS-H1 A.5: the poll wait becomes a task completed by the next `Unpause` of the same executor (per-executor TCS, `RunContinuationsAsynchronously` = posted to the pump). No timer of any kind; the loop still re-checks `_isPaused`. | `executor_unpause_tcs`: paused executor -> wait pending, 0 virtual timers; Unpause -> 1 wake-up, wait completes. |
| `fidelity-executor-unpause-wake` | `ActionExecutor.Unpause` | postfix | Scheduling | Completes the wait task of the row above for this executor (ActionExecutor.cs:71-75). No game state. | same check |
| `fidelity-turn-loop-wait-virtual-timeout` | `CombatManager.WaitForPreviousTurnLoopToFinish` (async MoveNext) | transpiler, exactly 1 `Task.WaitAsync(TimeSpan)` replaced | Scheduling | `await previousTurnLoopTask.WaitAsync(10 s)` (Combat/CombatManager.cs:343-370) arms a real `System.Threading` timer. Replaced by the same wait with the 10 s timeout on the pump's virtual clock (TimeoutException on expiry, the task's own fault otherwise; the game's catch blocks unchanged). Reached only when a combat starts while the previous combat's turn loop is alive (never seen). | `turn_loop_wait_virtual_timeout`: live loop -> 1 virtual timer; never-ending loop -> TimeoutException after 10 virtual s at ~30 ms wall. |
| `fidelity-assert-creature-spawn-wait` | `CombatState.GodotTimerTask` | prefix (assertion) | Instrumentation | Unreachable in single player: `GetCreatureAsync` (Combat/CombatState.cs:483-515) only arms this 10 s timeout for a target id not yet spawned. A hit fails the episode (`StrictViolation/fidelity_assert:creature_spawn_wait`) instead of arming a timer. Core's `virtual-clock-creature-spawn-timeout` stays underneath (H0 harness). Revisit for 4 players. | `assert_creature_spawn_wait` (negative test: forced call aborts with that detail). |
| `fidelity-assert-rest-hover-delay` | `RestSiteSynchronizer.TrySendHoverMessage` | prefix (assertion) | Instrumentation | `Task.Delay` / `Task.Yield` hover throttling (Multiplayer.Game/RestSiteSynchronizer.cs:293-318); only caller `LocalOptionHovered` <- `NRestSiteRoom.cs:370/376` (UI hover). Unreachable; a hit fails the episode. | `assert_rest_hover_delay` (negative test). |
| `fidelity-assert-peer-input-delay` | `PeerInputSynchronizer.TrySendSyncMessage` | prefix (assertion) | Instrumentation | `Task.Delay` cursor / input sync of remote peers (Multiplayer.Game.PeerInput/PeerInputSynchronizer.cs:180-206); UI input only. Unreachable; a hit fails the episode. | static scan (c) lists no non-UI caller; fuzz: 0. |
| `fidelity-guard-test-energy-cost-sneckooil` | `SneckoOil.TestEnergyCostOverride` setter | prefix (assertion) | Instrumentation | Test-only cost override (Models.Potions/SneckoOil.cs:29-40): set, `NextEnergyCost` returns it instead of drawing `Rng.CombatEnergyCosts`. No production caller. Any call fails the episode. | `assert_test_energy_cost` (negative test). |
| `fidelity-guard-test-energy-cost-confusedpower` | `ConfusedPower.TestEnergyCostOverride` setter | prefix (assertion) | Instrumentation | Same (Models.Powers/ConfusedPower.cs:17-28; relayed only from SneckoEye / FakeSneckoEye when their own field, default -1, was set). | install check; fuzz: 0 |
| `fidelity-guard-test-energy-cost-slither` | `Slither.TestEnergyCostOverride` setter | prefix (assertion) | Instrumentation | Same (Models.Enchantments/Slither.cs:14-27). | install check; fuzz: 0 |
| `fidelity-guard-test-energy-cost-sneckoeye` | `SneckoEye.SetTestEnergyCostOverride` | prefix (assertion) | Instrumentation | Same (Models.Relics/SneckoEye.cs:53-59). | install check; fuzz: 0 |
| `fidelity-guard-test-energy-cost-fakesneckoeye` | `FakeSneckoEye.SetTestEnergyCostOverride` | prefix (assertion) | Instrumentation | Same (Models.Relics/FakeSneckoEye.cs:42-48). | install check; fuzz: 0 |
| `fidelity-monitor-action-finished-bad-state` | `GameAction.Execute` (async MoveNext) | transpiler, exactly 1 `NonInteractiveMode.get_IsActive` replaced | Instrumentation | Read-only. An action that finishes outside state Executing throws under NonInteractive (GameActions/GameAction.cs:96-106; the real game logs and continues). Kept fail-closed (DECISIONS-H1 A.7); the helper counts the event and returns the same value. | `fuzz_no_abort_in_fidelity_area`: counter 0. |
| `fidelity-discovery-order-rooms-check` | `ActModel.ApplyDiscoveryOrderModifications` | prefix + postfix | Instrumentation | Read-only complement of Core's `fidelity-discovery-order-noop-check`, which compares only the boss: the method also calls `ApplyActDiscoveryOrderModifications` (Models.Acts/Overgrowth.cs:110-127 fixed first-run order when `NumberOfRuns == 0`). Fingerprints the whole RoomSet (events, normal / elite encounters in order, boss, second boss, ancient); a change fails the episode (`ProfileGateFired/fidelity_assert:discovery_order_rooms`). | `veteran_gates_noop_fuzz`: checked >= 1 per run, changed 0. |

### A3. No Harmony (driver-level, read-only)

| Id | What | Why | Self-check |
|---|---|---|---|
| `fidelity-run-invariants` | lifecycle `fidelity:invariants`, `OnRunSetUp` | Turns "proven inert" audit sites into checked facts every run: `TestMode.IsOn`; `NonInteractiveMode.AutoSlayerCheck() == false` (AutoSlay/AutoSlayer.cs:66, GameActions/ActionExecutor.cs:64/207); `TestRngInjector` empty (Factories/RelicFactory.cs:32-51, Entities.Cards/CardPile.cs:71, Factories/CardFactory.cs:94); `TestFlags.ShouldSendResumeForRemotePlayers == false`; `RestSiteOption.generateForTests == null` (Entities.RestSite/RestSiteOption.cs:43); `CardSelectorStack._testSelector == null` (DECISIONS-H1 B.2); Standard non-custom run with `ForceDiscoveryOrderModifications` (so the forced flag equals `ShouldApplyTutorialModifications`, Runs/RunManager.cs:644-663); no `Deprived` seat (Models.Relics/LargeCapsule.cs:40/49, ScrollBoxes.cs:98). Violation = `Aborted(StartFailed, fidelity_assert:<id>)`. | every scenario / fuzz episode of `--selftest` runs them (0 violations) |
| `fidelity-assert-backstop` | frame extension `fidelity:invariants` (adds no action) | At every room-level decision: fails the episode if a fidelity assertion fired but game code swallowed its exception (`fidelity_assert_swallowed`; e.g. CombatManager.cs:369-371 catches every exception), and once per episode that `CardSelectorStack.LocalSelector` is installed (`fidelity_assert:local_selector_installed`: CardSelectCmd's UI branch dereferences null UI singletons, Commands/CardSelectCmd.cs:139-147). | fuzz: 0 |

---

## E. Fidelity gates restored (real-game branches) - this pack

| Id | Site | TestMode branch | Real branch (now running) | Patch | Self-check (`--selftest`) |
|---|---|---|---|---|---|
| `fidelity-merchant-potion-price` | `MerchantPotionEntry.CalcCost` (Entities.Merchant/MerchantPotionEntry.cs:62-65) | base price 50/75/100, no draw: the whole `PlayerRng.Shops` stream shifts afterwards (card prices MerchantCardEntry.cs:117, restocks, random potions) | `round(base x Shops.NextFloat(0.95, 1.05))`, one Shops draw | `GatePatch`: exactly 1 `TestMode.get_IsOff` -> true | `shop_potion_prices`: 18 potion entries over 6 seeds, 15 differ from the base, 0 outside the band; one `CalcCost` draws exactly 1 from `PlayerRng.Shops`. |
| `fidelity-calling-bell-rewards` | `CallingBell.GenerateRewards` (Models.Relics/CallingBell.cs:39-63) | fixed Anchor / GremlinHorn / MummifiedHand, grab bag untouched | `RelicReward(Common)`, `(Uncommon)`, `(Rare)`, populated from the grab bag (Rewards/RelicReward.cs:66-86) | `GatePatch`: exactly 1 `TestMode.get_IsOn` -> false | `calling_bell_rewards`: 3 seeds, 3 different C/U/R triples, grab bag 122 -> 119 at the offer. |
| `fidelity-cauldron-rewards` | `Cauldron.GenerateRewards` (Models.Relics/Cauldron.cs:40-53) | fixed Flex / Weak / Vulnerable / Strength / Dexterity, no draw (every later card / potion reward shifts) | 5 `PotionReward(owner)` populated from `PlayerRng.Rewards` (Rewards/PotionReward.cs:54-61) | `GatePatch`: exactly 1 `TestMode.get_IsOn` -> false | `cauldron_rewards`: 3 seeds, 3 different potion sets, `PlayerRng.Rewards` counter +10 at the offer. |
| `fidelity-merchant-room-exit` | `MerchantRoom.Exit` (Rooms/MerchantRoom.cs:68-76) | returns early: unbought entries not in the map-point history, `_inventories` survives into the next `OnMerchantEntered` (warning) | `MerchantSynchronizer.OnMerchantExited` (Multiplayer.Game/MerchantSynchronizer.cs:119-168) | `GatePatch`: exactly 1 `TestMode.get_IsOn` -> false | `merchant_exit_bookkeeping`: after leaving a shop the inventories are cleared and the unbought cards / relics / potions are in the history. |
| `fidelity-map-history-player-stats` | `RunManager.UpdatePlayerStatsInMapPointHistory` (Runs/RunManager.cs:1590-1605) | returns early: CurrentGold / CurrentHp / MaxHp of every map point stay 0 | written before every move (:763) and in `OnEnded` (:1506) | `GatePatch`: exactly 1 `TestMode.get_IsOn` -> false | `natural_loss_bookkeeping`: every map-point entry of a natural run carries the stats (8/8). |
| `fidelity-rolling-boulder-per-target` | `RollingBoulderPower.AfterPlayerTurnStart` (Models.Powers/RollingBoulderPower.cs:36-52) | one `DoDamage` for all `HittableEnemies` (batch: all HP losses, then the hooks, then one Kill) | the VFX captures `HittableEnemies.ToList()` and its `HitCreature` signal starts one `DoDamage` per creature as the boulder passes it (Nodes.Vfx/NRollingBoulderVfx.cs:122-126, 204-224); a creature whose node is gone is skipped | transpiler on the MoveNext: exactly 1 `DoDamage` call -> host helper that calls the game's own `DoDamage` once per target, sequentially, in list order (DECISIONS-H1 E; no extra IsHittable filter), skipping only creatures no longer in combat; `Flash` / `SetAmount` original | `rolling_boulder_per_target`: SLIMES_NORMAL, 4 hittable enemies -> 1 trigger, 4 single-target `DoDamage` calls in list order. Residual (INFERRED): the real order is screen X order; list order = slot order for unslotted encounters. |
| `fidelity-natural-death-on-ended` | `CreatureCmd.Kill` (Commands/CreatureCmd.cs:376-391) | only `LoseCombat`: `HasRunEnded` stays false, `OnEnded` bookkeeping (map stats, TurnsTaken, `CheckUpdateEnemyDiscoveryAfterLoss`, RunManager.cs:1498-1578) never runs on a loss | `if (!HasRunEnded) OnEnded(isVictory: false)` + game-over screen | transpiler on the MoveNext: exactly 1 `TestMode.get_IsOff` -> host helper doing `if (!HasRunEnded) OnEnded(false)` and returning false (music / screen block still skipped) | `natural_loss_bookkeeping`: natural loss -> `OnEnded(false)` exactly once, `HasRunEnded` true at the terminal state. |

---

## B. Status of FIDELITY.md section B after this pack

| Site | Status |
|---|---|
| `MerchantPotionEntry.CalcCost` (:62) | **Handled** (`fidelity-merchant-potion-price`). |
| `CallingBell` (:39), `Cauldron` (:40) | **Handled** (`fidelity-calling-bell-rewards`, `fidelity-cauldron-rewards`). |
| `MerchantRoom.Exit` (:70) | **Handled** (`fidelity-merchant-room-exit`). |
| `RunManager.UpdatePlayerStatsInMapPointHistory` (:1592) | **Handled** (`fidelity-map-history-player-stats`). |
| `CreatureCmd.Kill` (:382) | **Handled** (`fidelity-natural-death-on-ended`). |
| `RollingBoulderPower.AfterPlayerTurnStart` (:36) | **Handled** (`fidelity-rolling-boulder-per-target`). |
| `LocManager.SmartFormat` (:246) | **Kept TestMode (fail-closed), not patchable**: the method contains a `catch ... when` exception filter that Harmony / MonoMod cannot re-emit ("Incorrect code generation for exception block" at install). A formatting error aborts the episode with a `FormattingException` signature (visible in every fuzz summary; 0 so far). |
| `GameAction.Execute` finally (:102) | Kept fail-closed, now **counted** (`fidelity-monitor-action-finished-bad-state`). |
| `CardSelectCmd.FromChooseABundleScreen` (:670) | Not here: events team (bundle gate + BUNDLE frame) per the H1 task assignment; DRIVER-API.md section 12 lists the gate under fidelity - conflict recorded in `notes\host-h1-fidelity.md`. |
| `RewardsSet.Offer` (:142), `LordsParasol` (:57), FoulPotion at the merchant | rooms team (verified plan: report section 2). |
| `CombatState.GodotTimerTask`, `ActionExecutor.WaitForUnpause`, `CombatManager.cs:354`, `RestSiteSynchronizer.cs:311/317` | Handled (A2 rows above). |
| `SneckoOil` / `ConfusedPower` / `Slither` / `SneckoEye` / `FakeSneckoEye` test overrides | Guarded (A2 rows above). |

The full table "every non-presentation audit site -> handled by" (123 sites) is produced by
`SpireHost.Fidelity --site-table` and reproduced in `notes\host-h1-fidelity.md` section 1.
