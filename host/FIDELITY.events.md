# SpireHost fidelity ledger - events team fragment (H1)

> **MERGED into `host\FIDELITY.md` at the H1 integration (2026-11-15).** This fragment is kept for history. It describes the
> team's stand-alone build against the stage-0 snapshot and is no longer checked against the registry: ids, counts,
> switches and "temporary" / "shim" remarks below may be out of date. The merged ledger is the source of truth
> (its section 4 lists what was removed or moved at the integration).

Owner: events team (`SpireHost.Driver.Events`). Same format and rules as `host\FIDELITY.md`; the lead merges this
fragment into it. Machine-readable source: `SpireHost.Driver.Events --list-event-patches` (snapshot of the current
build: `bench\h1-events\patches_events.json`). Every entry below is registered through `PatchRegistry` by
`EventPatches.InstallOnce()` (called from `EventsModule.Register`), fails closed at boot (missing target, or a
transpiler that does not find exactly the expected number of instructions), and is a no-op outside an episode
unless stated. Source references are `ref\sts2-v0113-src` paths without the `MegaCrit.Sts2.Core.` prefix.

Totals with this module registered: 49 registry entries = 35 default + 14 events
(events: 6 Fidelity, 8 Plumbing, **0 Behaviour**). Hit counters per id: `--selftest` output `patch_hits`.

---

## A2. Game level, Harmony (events)

| Id | Target | Kind | Category | What / why (source reference) |
|---|---|---|---|---|
| `events-waterfall-enter-portrait` | `WaterfallOfObviousSecrets.EnterWaterfall` | transpiler (1 call swap) | Plumbing | `:53` `NEventRoom.Instance.SetPortrait(...)` (portrait texture) NREs headlessly before `SetEventState` of the ENTER page (`:55`). The `callvirt NEventRoom::SetPortrait` is swapped for a static stand-in: same call if the receiver exists, nothing (counted `stub:NEventRoom.SetPortrait`) if null - the `?.` the neighbouring code uses. Argument evaluation (`PreloadManager.Cache.GetTexture2D` -> null) and all following game calls unchanged. Self-check: case `site_waterfall_enter` (ENTER page reached with 3 options), sweep. |
| `events-dense-vegetation-rest-presentation` | `DenseVegetation.Rest` (MoveNext) | transpiler (2 x `NDebugAudioManager.Play`, 1 x `Stop`, 1 x `NGame.ScreenRumble`) | Plumbing | `:93-97` sounds + screen rumble (local player only) between `PlayerCmd.MimicRestSiteHeal` (`:90`) and `SetEventState` of the REST page with FIGHT (`:99`). Same null-safe call stand-ins. The `Cmd.CustomScaledWait` in between is already a no-op (non-interactive). Not patched, unreachable headlessly: `TrudgeOn` `:77` (inside `container != null`, container = `NEventRoom.Instance?.VfxContainer` = null). Self-check: `site_dense_vegetation_rest` (healed, REST page, FIGHT starts the combat). |
| `events-amalgamator-combine-strikes-shake` | `Amalgamator.CombineStrikes` (MoveNext) | transpiler (2 x `NGame.ScreenShakeTrauma`) | Plumbing | `:46`, `:49` camera shake between `CardPileCmd.RemoveFromDeck` (`:44`) and creating / adding `UltimateStrike` + `SetEventFinished` (`:50-52`). The `Task.Delay(300)` in between stays on Core's virtual clock (`virtual-clock-amalgamator-combinestrikes`; both transpilers coexist on the MoveNext). Self-check: `site_amalgamator_combine_strikes` (deck: two basics out, ULTIMATE_STRIKE in, final page). |
| `events-amalgamator-combine-defends-shake` | `Amalgamator.CombineDefends` (MoveNext) | transpiler (2 x `NGame.ScreenShakeTrauma`) | Plumbing | `:60`, `:63`, same structure (`:58` remove, `:64-66` UltimateDefend). Self-check: `site_amalgamator_combine_defends`. |
| `events-jungle-maze-safety-audio` | `JungleMazeAdventure.SafetyInNumbers` (MoveNext) | transpiler (1 x `NDebugAudioManager.Play`) | Plumbing | `:94` sound before `PlayerCmd.GainGold(JoinForcesGold)` (`:96`) and `SetEventFinished` (`:97`). Not patched, unreachable headlessly: `DontNeedHelp` `:80` (inside `control != null`). Self-check: `site_jungle_maze_safety_in_numbers` (gold delta == `JoinForcesGold`). |
| `events-punch-off-nab-shake` | `PunchOff.Nab` (MoveNext) | transpiler (1 x `NGame.ScreenShakeTrauma`) | Plumbing | `:106` between `AddCurseToDeck<Injury>` (`:105`) and the `RelicReward` offer / `SetEventFinished` (`:109-113`); `:107` is already `?.`. Self-check: `site_punch_off_nab` (Injury in deck, reward frame, final page). |
| `events-trial-accept-portrait` | `Trial.Accept` | transpiler (1 x `NEventRoom.get_Layout`, 1 x `NEventLayout.RemoveNodesOnPortrait`, 1 x `NEventRoom.SetPortrait`) | Plumbing | `:68` `NEventRoom.Instance.Layout.RemoveNodesOnPortrait()` and `:108` `SetPortrait` (local player only) around the case draw `Rng.NextInt(3)` (`:73`) and `SetEventState` of the case page (`:112`). Self-check: `site_trial_accept` (a MERCHANT / NOBLE / NONDESCRIPT page with GUILTY / INNOCENT, event finishes). |
| `events-trial-portrait-vfx` | `Trial.AddVfxAnchoredToPortrait` (private helper) | prefix (skip when `NEventRoom.Instance == null`) | Plumbing | `:134-142`: the whole helper is presentation (instantiate a VFX scene, position it, add it to the portrait); headlessly `GetScene` returns null and `.Instantiate` throws. Called from `Accept:105` between the case draw and `SetEventState`. With a real event room the original runs. Self-check: counter, `site_trial_accept`. |
| `events-crystal-sphere-screen` | `NCrystalSphereScreen.ShowScreen(CrystalSphereMinigame)` | prefix | Fidelity | see E-ev1 |
| `events-trial-double-down-confirm` | `Trial.DoubleDown` | prefix | Fidelity | see E-ev2 |
| `events-bundle-gate` | `CardSelectCmd.FromChooseABundleScreen` (MoveNext) | transpiler (`GatePatch`, 1 x `TestMode.get_IsOn` -> false) | Fidelity | see E-ev3 |
| `events-bundle-screen` | same MoveNext | transpiler (1 x `NChooseABundleSelectionScreen.ShowScreen`, 1 x `.CardsSelected`) | Fidelity | see E-ev3 |
| `events-fake-merchant-foul-potion-usable` | `FoulPotion.PassesCustomUsabilityCheck` (getter) | postfix | Fidelity | see E-ev4 |
| `events-fake-merchant-foul-potion-use` | `FoulPotion.OnUse` | prefix | Fidelity | see E-ev4 |

How the call-swap stand-ins are proven harmless (per site): each swapped instruction is a call on a UI node
(`NEventRoom`, `NEventLayout`, `NGame`, `NDebugAudioManager`) whose body only touches Godot nodes / audio players /
the camera shake (`NEventLayout.cs:191-213`, `NGame.cs:993-1010`, `NDebugAudioManager.cs:80-140`); the stand-in
calls it unchanged whenever the receiver exists (real game, UI present), and every IL instruction before and after
it is untouched, so the game-logic continuation (`SetEventState`, `GainGold`, `CardPileCmd.Add`, the reward offers)
runs exactly as in the real branch. Presentation-only consumption of `Rng.Chaotic` inside those UI methods
(`NGame.ScreenShake` angle, audio pitch) does not happen headlessly; `Rng.Chaotic` is not a gameplay stream.

## D. UI-only flows the events team performs by proxy

| Flow | UI source | What the host does | Where | Status |
|---|---|---|---|---|
| Event option dispatch | `NEventRoom.OptionButtonClicked` (NEventRoom.cs:233-250), `NEventOptionButton.OnRelease` (:260-289) | legal = `!IsLocked && !(DisableOnChosen && WasChosen)`; `WillKillPlayer` filtered only with >1 seat; `IsProceed` -> `option.Chosen()` directly (tracked); otherwise `EventSynchronizer.ChooseLocalOption(i)` (shared events: the vote path, one vote, host picks with the `event_synchronizer` RNG) and every new task in `_pendingOptionTasks` is tracked | `EventRoomHandler` | **final** (single seat) |
| Synthesized PROCEED of a finished event | `NEventRoom.SetOptions` (:217-232) -> `NEventRoom.Proceed` -> `RunManager.Proceed()` | `Verb.Proceed` -> `ctx.OpenMap` | `EventRoomHandler` | **final** |
| Ancient `PROCEED` (`IsProceed`, OnChosen = `NEventRoom.Proceed`) | AncientEventModel.cs:159-173 | `option.Chosen()` then `ctx.OpenMap` (RunManager.Proceed is a no-op without NMapScreen) | `EventRoomHandler` | final; **unreachable in v0.113** (no model overrides `ShouldAllowAncient`) |
| Displayed texts | `NEventRoom.SetDescription` (NEventRoom.cs:205-216), `NEventOptionButton._Ready` (:181-182) | the same `Character.AddDetailsTo` / `"IsMultiplayer"` / `Event.DynamicVars.AddTo(...)` on the page description and on every displayed option's Title / Description, at every frame (idempotent). Needed beyond display: `SaveEventOptionToHistory` copies `HistoryName.Variables` (HistoryName == Title) into the run history, which headlessly would otherwise lack the event variables. Writes LocString variable maps only. | `EventRoomHandler.MirrorDisplay` | final |
| FakeMerchant shop | `NFakeMerchant` (NFakeMerchant.cs:100-300), `NMerchantSlot.OnSelected` (:275-283) | while `!StartedFight`: SHOP frame over `FakeMerchant.Inventory` (`ShopContext.FakeMerchant = true`); BUY = `IsStocked && EnoughGold` -> `MerchantSynchronizer.PurchaseLocalEntry` (a `false` result aborts as `HostBug/purchase_rejected`); PROCEED -> `ctx.OpenMap` (`OnProceedButtonReleased` -> `RunManager.Proceed`; travel is enabled from entry, :153) | `EventRoomHandler` | final |
| Crystal Sphere screen | `NCrystalSphereScreen` (:146-331), `NCrystalSphereCell` (:68-99) | CRYSTAL frame: `CRYSTAL_CLICK(hidden cell, tool)`; apply = `SetTool(tool)` if it differs (the divination buttons) + `CellClicked(cell)` (tracked); pops before the last click; rewards follow as a non-terminal REWARD frame; the event's synthesized PROCEED opens the map (= the screen's proceed button, `RunManager.Proceed`). Hover highlighting not reproduced (presentation). | `CrystalFrame` | final |
| Abandon confirmation of Trial DOUBLE_DOWN | `NAbandonRunConfirmPopup` (:98-111) | CONFIRM frame: NO -> nothing; YES (`ActionFlags.Abandon`) -> `RunManager.Instance.Abandon()` | `TrialConfirmFrame` | final |
| Bundle screen | `NChooseABundleSelectionScreen` (:186-282) | BUNDLE frame: `CHOOSE_OPTION(bundle i)`, `SKIP` when `canSkip`; completes the awaited `CardsSelected` task with `[bundle]` or empty | `BundleFrame` | final |

Not reproduced (presentation only, VERIFIED): `EventModel.OnRoomEnter` (only called by `NEventLayout.SetEvent`:
Reflections / SlipperyBridge VFX, WelcomeToWongos music, TheArchitect score + speech bubbles; TheArchitect's option
chain dialogue -> PROCEED -> `WinRun` works without it, 15/15 + 6/6 wins in the tests), `PunchOff.PunchEachOther`
(returns at once without `NCombatRoom`), Ancient dialogue selection with `Rng.Chaotic` (NEventRoom.cs:184).

## E. Fidelity gates restored (events)

| Id | Site | TestMode / headless branch | Real branch (now running) | Patch | Self-check |
|---|---|---|---|---|---|
| E-ev1 `events-crystal-sphere-screen` | `CrystalSphereMinigame.PlayMinigame` -> `NCrystalSphereScreen.ShowScreen` (CrystalSphereMinigame.cs:135-143, NCrystalSphereScreen.cs:146-152; no TestMode gate) | scene instantiation on a null cache entry -> NRE, the minigame never completes | the player chooses tool + hidden cell until `DivinationCount == 0`; `CompleteMinigame` -> `OneOffSynchronizer.DoLocalCrystalSphereRewards` -> `RewardsCmd.OfferCustom` | prefix (episode only): push `CrystalFrame(grid)`, `__result = null` (the caller discards it), skip the original | cases `crystal_sphere_uncover_future` / `_payment_plan` (4 seeds each: exactly 3 / 6 CRYSTAL frames, a REWARD frame iff a non-curse item was revealed, event finishes, map opens), invariant on every CRYSTAL frame (actions = 2 x hidden cells, only hidden cells, view has no hidden-cell item data); sweep 15/15 |
| E-ev2 `events-trial-double-down-confirm` | `Trial.DoubleDown` (Trial.cs:128-132) | `NModalContainer.Instance.Add(NAbandonRunConfirmPopup.Create(null))`: `Create` returns null in TestMode (NAbandonRunConfirmPopup.cs:89), container null -> NRE | the popup; YES -> `RunManager.Abandon()`, NO -> close (the REJECT page stays, DOUBLE_DOWN can be chosen again: `disableOnChosen: false`) | prefix (episode only): push `TrialConfirmFrame`, return `Task.CompletedTask` (what DoubleDown returns) | `trial_double_down_no` (CONFIRM holds only NO by default and is auto-advanced, REJECT page again), `trial_double_down_yes_abandons` (`AllowAbandon`: YES -> Loss, `RunManager.IsAbandoned`) |
| E-ev3 `events-bundle-gate` + `events-bundle-screen` | `CardSelectCmd.FromChooseABundleScreen` (CardSelectCmd.cs:657-698; only caller `ScrollBoxes.AfterObtained`) | `:670-673` `readOnlyList = bundles[0]` (no choice, no `SyncLocalChoice`) | `ShouldSelectLocalCard` -> screen -> `(await CardsSelected()).FirstOrDefault() ?? Empty` -> `SyncLocalChoice(player, choiceId, FromIndex(IndexOf))` -> `LogChoice`; `ReserveChoiceId` (`:668`) is before the gate and always ran | gate: exactly 1 `TestMode.get_IsOn` -> `false`; screen: exactly 1 `ShowScreen` and 1 `CardsSelected` call swapped for `EventPatches.BundleShowScreen` (pushes `BundleFrame`) / `BundleCardsSelected` (returns its task). Nothing else in the method changes (testmode-audit-models-commands.md 2.1: unlike the old full-replacement patch, the choice-id protocol is kept) | `bundle_scroll_boxes_pick_1` (3 seeds: BUNDLE frame with 2 bundles, no skip; the deck receives exactly bundle 1; choice id +1; stand-in hit once), `bundle_skip_path` (`canSkip: true` -> SKIP returns empty; choice id +1) |
| E-ev4 `events-fake-merchant-foul-potion-usable` + `-use` | `FoulPotion.PassesCustomUsabilityCheck` (FoulPotion.cs:51-84, target via `GetFoulPotionMerchantTarget` :137-164) and `FoulPotion.OnUse` FakeMerchant branch (:104-123) | no `NFakeMerchant` node: unusable; and if used, `OnUse` returns without effect | usable out of combat in the FakeMerchant event while the merchant button exists and the inventory is closed; use = `FakeMerchant.FoulPotionThrown(potion)` for every player (Task.WhenAll) -> `StartedFight`, rewards (FakeMerchantsRug + unbought relics), `EnterCombatWithoutExitingEvent<FakeMerchantEventEncounter>` | usable: postfix, sets `true` only when the original said false, the room is the FakeMerchant `EventRoom`, the local event has no node and `!StartedFight` (the host SHOP frame is the "inventory closed" state); use: prefix only for that branch and only when the node is absent, runs exactly the branch's calls (`ShowPotionVfx` is TestMode-gated presentation; `FoulPotionThrown` skips its node part itself). All other branches run the original (the MerchantRoom branch of the same getter is the rooms team's). | `fake_merchant_foul_potion_fight` (2 seeds: gold 0 + FoulPotion -> SHOP frame with only the throw; combat `FAKE_MERCHANT_EVENT_ENCOUNTER` with parent event FAKE_MERCHANT; reward set contains FAKE_MERCHANTS_RUG), sweep `rich` variant |

## Test-only interventions (never in the default set, not registered as patches)

* `FoulPotionTestExtension` (`events:test-foul-potion`): a frame extension registered only by this console's
  `--selftest`, standing in for the rooms team's out-of-combat potion extension on the FakeMerchant SHOP frame
  (FoulPotion only; NPotionPopup rules; `EnqueueManualUse(null)`). Must not be registered after integration.
* Sweep state construction (`--selftest` only): `RunManager.EnterAct(n)` to reach act-gated events, gold / HP / potions /
  relics / an Exhaust card through `ScenarioTools`, and `RunState.AppendToMapPointHistory` to make
  `TotalFloor >= 7` (PunchOff / SlipperyBridge `IsAllowed`).

## Not covered / open

* Multi-seat: the shared-event vote with more than one vote, the multiplayer `WillKillPlayer` block (coded, never
  executed), remote crystal-sphere rewards (`CrystalSphereRewardsMessage`), remote bundle choices.
* `DeprecatedEvent` / `DeprecatedAncientEvent` (save-compatibility placeholders; `DeprecatedAncientEvent.DefineDialogues`
  throws) are not entered by any run and are excluded from the sweep.
