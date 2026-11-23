# Contract requests - rooms team (H1)

## Status after the H1 integration (2026-11-15, contract version `h1-integrated`)

The lead's rulings are in the integration brief; how each was implemented: `notes\host-h1-integration.md`. "Shim removed" means the team's local stand-in described further down no longer exists. The text below the table is the original request, kept for history.

| Request | Subject | Status | Where / how |
|---|---|---|---|
| R1 | TreasureContext page fields | **ACCEPTED, implemented** | `TreasureContext.Finished`, `Empty`, `Awarded` (context fields only, no new sub-kind). Filled by `TreasureHandler`; the encoder reads them. |
| R2 | CardRewardContext.Alternatives documentation | **ACCEPTED, done** | Doc comment in `DecisionContexts.cs`, DRIVER-API.md 6.2. |
| R3 | Per-module counters | **ACCEPTED, implemented (= combat C4 = fidelity R1)** | `RoomsCounters` = module `rooms`; see combat C4. |
| R4 | Bound the decisions asked while a scenario Setup runs | **ACCEPTED, implemented (with R7)** | `RunDriver.DriveStartTask` applies `MaxPolicyDecisions`, `MaxCombatTurns` and the decision cap like the main loop (`LimitReached`, `CheckDecisionCap`; the episode ends `Truncated` from inside the start chain). |
| R5 | Documentation (MapHandler, CHEST page) | **ACCEPTED, done** | `MapHandler` class comment; DRIVER-API.md 5.2 (the CHEST page is a TREASURE decision with the single action OPEN_CHEST, auto-advanced when it is the only action) and 6.6; FIDELITY.md sections 5 - 7 carry the rooms rows. |
| R6 | ScenarioTools.EnterAct / EnterBoss | **ACCEPTED, implemented** | `ScenarioTools.EnterAct(ctx, actIndex, doTransition)`, `EnterBoss(ctx)` (test only); the rooms scenarios use them. |
| R7 | Count decision points inside DriveStartTask | **ACCEPTED, implemented (with R4)** | see R4. |
| note: replay | trace policy maps OPEN_CHEST | **DONE (ruling D2)** | `RunTraceCompare`; the `replay:trace-dump` lifecycle is removed; `--compare-run-traces` 120 / 120. |
| note: events | finished-event invariant vs potion actions | **DONE (ruling D1)** | the events self-test invariant ignores USE_POTION / DISCARD_POTION added by `rooms:potions`; events cases 103 / 103. |
| note: fidelity | `rest_options_inert` vs `rest:soft-lock-reported` | **DONE (ruling D3)** | the invariant stays strict in production (checked at every quiescent point); the one rooms scenario names it in `Scenario.Exempt` (`ScenarioRunner.Run(..., exempt:)`), counted in `fidelity.exemptions_used`. |

---
To: the Core/Driver owner (lead). From: rooms team (`SpireHost.Driver.Rooms`, snapshot `h1-stage0`).
None of these blocks the rooms deliverable; each says what the rooms code does until it lands.

## R1. `TreasureContext`: say which TREASURE page the decision is (additive fields)

**What.** Add to `SpireHost.Driver.TreasureContext`:

```csharp
public bool Finished;                       // relic picking finished (or the chest was empty): the PROCEED-only page
public bool Empty;                          // the chest had no relic (SilverCrucible / Hook.ShouldGenerateTreasure false)
public IReadOnlyList<RelicModel> Awarded = Array.Empty<RelicModel>();   // relics obtained from this chest (public: shown when taken)
```

**Why.** obs-schema-v1 section 6 (TREASURE: RELIC page = RELICS OFFER + TREASURE_RELIC options) and DRIVER-API 6.2
("fill the contexts completely"). Today both pages use `SubKind = TreasurePhase.Relic`; the PROCEED page has
`Relics = []`, which the encoder can only tell apart from "no relic yet" by looking at the action list.
**Update (takeover, 2026-11-14 evening).** The CHEST page now exists as its own decision (`SubKind = TreasurePhase.Chest`,
single action `OPEN_CHEST`, see R5), so only the PROCEED-only page remains ambiguous with `SubKind = Relic`.

**Shim until then.** `TreasureState` (per room, `ctx.RoomData`) holds exactly these values inside the rooms project
(`Treasure.cs`); the handler fills `Relics = []` on the PROCEED page. A `TreasurePhase.Proceed = 3` sub-kind would
be the alternative (enum append, obs-schema A.x amendment).

## R2. `CardRewardContext.Alternatives`: document the exposed list (documentation)

**What.** DRIVER-API 6.3 `CardRewardAlt` row: "index in `CardRewardContext.Alternatives`", and 6.2: "Alternatives
= the buttons of the screen: the list the game passes, minus a REROLL that `CardRewardAlternative.Generate` would no
longer produce (`!CardReward.CanReroll`)".
**Why.** `CardReward.OnSelect` generates the alternatives once (Rewards/CardReward.cs:167) and passes the same list on
every call after a reroll, while the real screen is refreshed with regenerated buttons (`Populate` ->
`RefreshOptions`, :139-141). Offering the stale REROLL would allow an unlimited reroll the UI does not have.

## R3. Per-module counters in `EpisodeCounters` / fuzz summaries (additive)

**What.** `public readonly SortedDictionary<string, long> Module = new(...)` on `EpisodeCounters`, summed by
`--fuzz-runs` / `--determinism` into the JSON (`module_counters`). (The fidelity team asks for the same:
`EpisodeResult.ModuleCounters`, their R1.)
**Why.** The terminal proceed tree has six branches and post-condition checks; today they can only be seen through
the rooms console's own `--tree-runs` command, not in the common fuzz summary.
**Shim.** `SpireHost.Driver.Rooms.RoomsProbe` (process-wide static counters) + `--tree-runs` / `--selftest` output.

## R4. Bound the decisions asked while a scenario `Setup` runs (fix)

**What.** `RunDriver.DriveStartTask` asks pending frames in an unbounded loop and never checks
`EpisodeOptions.MaxPolicyDecisions` / `DecisionCap`. Apply the same two checks as `MainLoop`
(`Truncated("max_policy_decisions")`, `Livelock/decision_cap`).
**Why.** A scenario whose `Setup` offers a reward set (`RewardsCmd.OfferCustom`) is answered entirely inside
`DriveStartTask`; a scripted policy that keeps taking + skipping a card reward (legal, the reward stays) never
returns. Found by the rooms self-test (the scenario now leaves after two skips). Real runs are not affected (the
start chain itself offers no reward set).

## R5. `MapHandler` class comment and DRIVER-API.md 6.6 / 12 (documentation)

* `MapHandler`: "Not offered here yet (rooms team): out-of-combat potion use / discard" -> offered by the rooms
  module's frame extension `rooms:potions` (also on EVENT frames, including the events team's FakeMerchant SHOP frame).
* DRIVER-API.md 5.2 names "opening the chest" as the example of `HandlerStep.Progressed`. The rooms handler now asks a
  TREASURE decision `SubKind = TreasurePhase.Chest` with the single action `Verb.OpenChest` instead (obs-schema-v1 section 6
  CHEST page; replay request R5: the first decision point of a treasure room then matches the oracle's room-boundary
  dump). Please replace the example (e.g. by "work the real UI does unasked" without naming the chest).
* Section 12 "rooms": every item is resolved (report `notes\host-h1-rooms.md` section 3). FIDELITY.md section B row
  `RewardsSet.Offer` and the temporary rows of section D are replaced by `FIDELITY.rooms.md` D/E. Note the changed
  terminal semantics: a terminal set is no longer skipped at PROCEED but left for `BeforeLeavingRoom` (as the UI).

## R6. `ScenarioTools`: two test jumps the rooms tests needed (additive, tests only)

**What.** `ScenarioTools.EnterAct(ctx, actIndex)` (= `RunManager.EnterAct(i)`) and `ScenarioTools.EnterBoss(ctx)`
(= `RunManager.EnterMapCoord(run.Map.BossMapPoint.coord)`), documented as test-only like `JumpToEncounter`.
**Why.** The proceed-tree branches "last boss of an act" and "first of two bosses" depend on `run.CurrentMapPoint ==
BossMapPoint`, which `EnterRoomDebug` (all existing jumps) never sets. The rooms scenarios call the two game methods
directly from their `Setup` (`RoomScenarios.EnterBoss`); a shared helper keeps every team's tests on the same path.

## R7. Count decision points asked inside `DriveStartTask` like the main loop (fix, related to R4)

**What.** Decisions answered while a scenario `Setup` is still running are not checked against `MaxPolicyDecisions`
(R4); in addition a forced-HP scenario whose `Setup` jumps straight into a boss room runs that whole combat inside
`DriveStartTask`. Apply the R4 checks there and also `MaxCombatTurns`.
**Why.** Found at takeover: `tree:last-act-boss->architect` spent 27 s on one TestSubject fight (the boss revives with
fresh HP, `Models.Monsters/TestSubject.cs:264-346`) and the 6-character battery never finished; tests only. Fixed locally
by the test policy (enemies weakened at every COMBAT decision); real runs are not affected.

## Coordination notes (no contract change)

* **replay**: `RunTraceCompare`'s policy maps TREASURE to `ChooseOption` else `Proceed`. With the CHEST page the decision
  is auto-advanced when it has one action, but when the player holds a potion the page also carries potion actions
  and is asked: map `Verb.OpenChest` there (the trace's "treasure open" choice). The `replay:trace-dump` lifecycle for
  treasure rooms can then be removed (their R5).
* **events**: the events self-test invariant "a finished event frame is exactly the synthesized PROCEED" fails as soon as
  the rooms potion extension is registered (DECISIONS-H1 B.7: potion actions on EVENT frames; measured with
  `SpireHost.Driver.Rooms --team-command events:selftest --with-modules events --skip-sweep`: 94/103 cases, the 9
  failures are exactly this invariant on finished pages with a potion in the belt). The invariant should ignore
  `UsePotion` / `DiscardPotion` actions owned by `rooms:potions`. The FakeMerchant FoulPotion case passes with the
  rooms extension (`fake_merchant_foul_potion_fight` 2/2).
* **fidelity**: `fidelity_assert:rest_options_inert` asserts `RestSiteOption.generateForTests == null`; the rooms
  scenario `rest:soft-lock-reported` sets it (test only, reset in `Cleanup`) to build the soft-lock state. Measured with
  `--with-modules events,fidelity`: 216/216 scenarios pass, because the REST handler reports the stall before any frame
  extension runs. Brittle by construction - if the invariant moves to an episode-start check (their R1), that one
  scenario must be exempted.
