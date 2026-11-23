# Contract requests - combat team (H1)

## Status after the H1 integration (2026-11-15, contract version `h1-integrated`)

The lead's rulings are in the integration brief; how each was implemented: `notes\host-h1-integration.md`. "Shim removed" means the team's local stand-in described further down no longer exists. The text below the table is the original request, kept for history.

| Request | Subject | Status | Where / how |
|---|---|---|---|
| C1 | CombatContext fields | **ACCEPTED, implemented** | `SpireHost.Driver\DecisionContexts.cs`: `CombatContext` (`State`, `Seats`, `Creatures`, `Hittable`, `AllyCount`), `CombatSeatInfo`, `NotReady` moved to the driver. The combat handler fills the context directly; `CombatFrameInfo` and the side table are deleted. The encoder reads the context. |
| C2 | CardSelectContext fields | **ACCEPTED, implemented** | `CardSelectContext`: `Entry`, `ContextKind`, `RawMin` / `RawMax`, `InstanceCount`, `Merged`, `PickedOrdinals`, `OptionPicked`, `Enchantment` (+ `EnchantmentAmount`), `PromptTable` / `PromptKey`, `SortedByComparison`, `OutsidePlayPhase`, `DuringEnemyTurn`. The encoder takes sel_count from `OptionPicked` and the prompt key from `PromptTable` / `PromptKey` (two fixed headers were added to vocabulary v3). |
| C3 | Hidden enemy intents | **ACCEPTED, implemented in the driver core** | `RunContext.IntentPerformed(enemy)`; driver patch `enemy-turn-intent-performed` (prefix on `Creature.TakeTurn`). The combat module's `combat-intent-performed` is removed. Check: `obs --selftest` case `enemy_turn`. No real-client ground truth exists: open verification item (D8). |
| C4 | Per-module counters | **ACCEPTED, implemented (= rooms R3 = fidelity R1)** | `IModuleCounters` + `DriverRegistry.SetModuleCounters`; `EpisodeResult.ModuleCounters` (per-episode deltas, sorted); summed by `--fuzz-runs` (`module_counters`, `coverage`), compared by `--determinism`. `CombatCounters` = module `combat`. |
| C5 | Combat-end scheduling boundary to the core | **ACCEPTED, implemented as one uniform rule (ruling E2)** | Core patch `run-save-completes-after-pump-turn` (postfix on `SaveManager.SaveRun`): every awaiter of the run save resumes one pump turn later - the combat-end save, the room-entry save (E1) and the event-room save. `combat-end-run-save-yield` and `CombatScheduling.cs` are deleted. Arbiters: suite `turnend` 8 / 8, `turnend-control` 8 / 8 aborting in `PopAction`. |
| C6 | Retire `selection-context-choose-a-card` | **ACCEPTED, implemented** | The Core patch and `SelectionContextPatches.cs` are deleted with the temporary handlers (rulings B2 / B3). |
| C7 | Canonical classes registration (= obs O1) | **ACCEPTED, implemented** | `delegate CanonicalCardClasses(ctx, cards, pile, seat, unpoweredPreviews, classes, counts)`, `DriverRegistry.SetCanonicalCardClasses`, `RunContext.CanonicalCardClasses`. The provider is the obs encoder's (`ObsEncoder.CanonicalClasses`), registered by `ObsModule` in every mode (ruling B1). `CanonicalCandidates.Provider` and the `SPIREHOST_COMBAT_OBS_CLASSES` switch are deleted; without a provider a pile prompt aborts. The local class key survives only as a test provider of suite `drawperm`. |
| C8 | ScenarioTools mid-combat tools | **ACCEPTED, implemented** | `ScenarioTools.CombatState`, `AddCard`, `SetResources`, `ApplyPower`, `Afflict` (test only). The combat tests use them; the local copies are deleted. |
| C9 | Documentation | **ACCEPTED, done** | DRIVER-API.md 6.2 (`CardSelectEntry`: 15 overloads under 12 names + the bundle entry; `SyncLocalChoice` is not on the local-selector path) and FIDELITY.md section 6. |
| C10 | Who calls `SyncLocalChoice` for in-episode recording | **DEFERRED to H4 / H5 (ruling)** | Open item in DRIVER-API.md section 12 and FIDELITY.md section 10. The local code is unchanged. |

---
To: the Core/Driver owner (lead). From: combat team (`SpireHost.Driver.Combat`, snapshot `h1-stage0`).
None of these blocks the combat deliverable; each names the clearly marked local shim the combat code uses until
it lands. All are additive (DRIVER-API.md section 14, item 2) except C5 / C6 (moving / retiring a patch) and C9
(documentation). Report: `notes\host-h1-combat.md`. Ledger fragment: `host\FIDELITY.combat.md`.

---

## C1. `CombatContext`: what the COMBAT builder already knows (additive fields)

**What.** `SpireHost.Driver.CombatContext` carries `State` only. Add:

```csharp
public sealed class CombatContext
{
    public CombatState State = null!;

    /// <summary>Index = seat. Which seats may act in this decision and, if not, why.</summary>
    public IReadOnlyList<CombatSeatInfo> Seats = Array.Empty<CombatSeatInfo>();

    /// <summary>CombatState.Creatures (allies, then enemies) without the allocation; index = the Creature ordinal of GameRef.</summary>
    public IReadOnlyList<Creature> Creatures = Array.Empty<Creature>();

    /// <summary>Per creature: Creature.IsHittable as evaluated for this decision; 0 = not asked, 1 = yes, 2 = no.</summary>
    public IReadOnlyList<byte> Hittable = Array.Empty<byte>();

    public int AllyCount;
}

public sealed class CombatSeatInfo
{
    public int Seat;
    public Player Player = null!;
    public int NotReadyReason;                    // 0 = may act; else SpireHost.Driver.Combat.NotReady (move the enum to the driver)
    public bool ReadyToEndTurn;                   // the seat has ended its turn (hand / belt / button disabled by the UI)
    public bool CanUndoEndTurn;                   // multi-seat only: UNDO_END_TURN is offered
    public IReadOnlyList<bool> HandPlayable;      // per hand position: CardModel.CanPlay
    public IReadOnlyList<UnplayableReason> HandReason;      // its reason bits (CARDS col 19 unplayable_reason)
    public IReadOnlyList<AbstractModel?> HandPreventer;     // the model whose Hook.ShouldPlay vetoed (BlockedByHook)
    public IReadOnlyList<bool> PotionUsable;      // per belt slot: a USE_POTION action is offered
    public IReadOnlyList<bool> PotionDiscardable; // per belt slot: the DISCARD_POTION action is offered
}
```

**Why.** DRIVER-API.md 6.2 ("fill the contexts completely"), obs-schema-v1 CARDS col 19 `unplayable_reason`,
CREATURES col 9 `hittable`, SEATS col 19 `ready_to_end_turn`, GLOBAL 5 `acting_seat` (4-seat frames carry the
union of all acting seats' actions: DECISIONS-H1 F.7). The handler has evaluated exactly these with the game's own predicates while enumerating
(`CardModel.CanPlay`, `Creature.IsHittable` = one hook dispatch each); the encoder otherwise repeats every hook
dispatch per frame, and it cannot derive at all WHICH seat may act and why not (11 engine conditions of
driver-spec-combat.md 1.5, e.g. a paused per-seat queue).

**Shim until then.** `SpireHost.Driver.Combat.CombatFrameInfo` (`CombatFrameInfo.cs`), one instance per handler,
refilled at every COMBAT decision, valid until the next one; reached with `CombatFrameInfo.Of(decision)`
(`decision.Source` is the combat handler). Everything in it is public information (what the real UI shows as
greyed-out cards / disabled buttons).

## C2. `CardSelectContext`: prompt identity, picks per option, raw bounds (additive fields)

**What.** Add to `SpireHost.Driver.CardSelectContext`:

```csharp
public CardSelectEntry Entry;                 // which CardSelectCmd overload asked (15 values; enum moves to the driver)
public ChoiceContextKind ContextKind;         // None (deck screens) / GameAction (card, potion) / Hook / Blocking
public int RawMin, RawMax;                    // what the game handed to ICardSelector (before the UI-only rules)
public int InstanceCount;                     // candidate instances = sum of OptionCounts
public bool Merged;                           // OptionCounts != null
public readonly List<int> PickedOrdinals;     // index in Options of every pick, in pick order (parallel to Picked)
public int[] OptionPicked;                    // per option: how many of its instances are picked (CARDS col 24 sel_count)
public EnchantmentModel? Enchantment;         // FromDeckForEnchantment: shown by the real screen
public int EnchantmentAmount;
public string PromptTable, PromptKey;         // the header the real screen shows, as loc table + key (GLOBAL 24 prompt_key)
public bool SortedByComparison;               // simple grid shown sorted with CardSelectorPrefs.Comparison
public bool OutsidePlayPhase;                 // raised by a start-of-turn / shuffle / turn-end hook or in the enemy turn
public bool DuringEnemyTurn;                  // KnowledgeDemon: intents of enemies that already acted are hidden (F.6)
```

**Why.**
* `OptionPicked` / `PickedOrdinals`: obs-schema-v1 7.2 and CARDS col 24 `sel_count`. With merged classes
  (`OptionCounts`), `Picked` holds the instances handed to the game; the second pick of a class is a different
  instance than `Options[i]`, so `ReferenceEquals(Picked[k], Options[i])` (what the encoder does today,
  `SpireHost.Obs\Encoding\ObsEncoder.Cards.cs` `EncodeChoice`) undercounts. Only the frame knows the mapping.
* `PromptTable` / `PromptKey`: GLOBAL 24. `Prefs.Prompt` exists for 13 of the 15 entry points; the other two show a
  fixed header: `gameplay_ui/CHOOSE_CARD_HEADER` (`FromChooseACardScreen`,
  Nodes.Screens.CardSelection/NChooseACardSelectionScreen.cs:164) and `gameplay_ui/CHOOSE_CARD_UPGRADE_HEADER`
  (`FromHandForUpgrade`, Commands/CardSelectCmd.cs:641).
* `Enchantment(+Amount)`: the enchant screen shows what the pick will receive
  (Nodes.Screens.CardSelection/NDeckEnchantSelectScreen.cs); `Prefs` does not contain it.
* `OutsidePlayPhase` / `DuringEnemyTurn`: DECISIONS-H1 F.6; see C3.
* `RawMin` / `RawMax` / `Entry` / `ContextKind`: replay logs and diagnostics (`MinSelect` / `MaxSelect` are the
  effective bounds of the real screen, `SelectionRules.Compute`, not the numbers the game hands to the selector;
  e.g. `FromChooseACardScreen` always passes (0, 1) whether or not the screen has a skip button).

**Shim until then.** `SpireHost.Driver.Combat.CardSelectExtra` (`CardSelection.cs`), reached with
`CardSelectExtra.Of(decision)` (`decision.Source` is the `CardSelectFrame`). `CardSelectEntry` /
`ChoiceContextKind` live in `SelectionCapture.cs`.

## C3. Which enemy intents the UI has already hidden (additive, one query)

**What.** `bool RunContext.IntentPerformed(Creature enemy)` (or a per-creature flag list in `CombatContext`): true
while the enemy side's turn of this round is running and this enemy has begun its move.

**Why.** DECISIONS-H1 F.6 / obs-schema-v1: a prompt raised inside the enemy turn (KnowledgeDemon.ChooseCurse,
Models.Monsters/KnowledgeDemon.cs:175-198) must be encoded without the intents of the enemies that already acted:
the real UI hides an intent right before `Creature.TakeTurn` (Combat/CombatManager.cs:1142-1147 ->
Nodes.Combat/NCreature.cs:481-496). The game keeps no logical flag (`CombatHistory.MonsterPerformedMove` is written
after the move, Models/MonsterModel.cs:543-545).

**Shim until then.** `SpireHost.Driver.Combat.IntentTracker.IntentPerformed(creature)` (static), fed by the
read-only prefix `combat-intent-performed` on `Creature.TakeTurn` (FIDELITY.combat.md A2).

## C4. Per-module counters in `EpisodeCounters` / fuzz summaries (additive; = rooms R3, fidelity R1)

**What.** As rooms R3: `public readonly SortedDictionary<string, long> Module` on `EpisodeCounters`, summed by
`--fuzz-runs` / `--determinism` into the JSON.

**Why.** "Zero rejected or cancelled actions" is a hard requirement. Each violation aborts the episode, so the abort
list of a fuzz summary is the safety net, but the positive evidence - how many plays / potion uses / end turns were
submitted, how many CARD_SELECT frames by kind and context, how often the combat-end scheduling patch ran - is only
visible in this team's `--selftest` JSON.

**Shim until then.** Process-wide statics `CombatStats`, `CardSelectStats`, `SelectionCapture.EntryCalls /
EntryAsked`, `CombatScheduling.CombatEndSaveYields(Skipped)`, `CanonicalCandidates.ProviderCalls`; printed by
`SpireHost.Driver.Combat --selftest` (`combat`, `card_select`, `from_entry_points`, ...).

## C5. Move the scheduling patch `combat-end-run-save-yield` to the core (ownership)

**What.** `CombatScheduling` (`SpireHost.Driver.Combat\CombatScheduling.cs`: one fail-closed transpiler on the
MoveNext of `CombatManager.EndCombatInternal(CombatTurnState)`, exactly 1 call replaced) belongs next to
`hook-action-started-inline-continuation` in `SpireHost.Core` (FIDELITY.md A2, category Scheduling). Proposed:
move the class unchanged; the id stays.

**Why.** It is a property of the headless pump, not of the combat handler: the real client's
`await SaveManager.Instance.SaveRun(room, false)` (Combat/CombatManager.cs:1059) suspends (run save written through
the thread pool), the host's completes synchronously (`shouldSave: false`), and the remainder
(`ActionQueueSet.CombatEnded()`, :1077 -> GameActions.Multiplayer/ActionQueueSet.cs:327-356) can then run inside the
`finally` block of a hook action's `GameAction.Execute` (GameActions/GameAction.cs:92-97) -> `PopAction` throws
(ActionQueueSet.cs:525). It happens with the temporary combat handler too (found by the rooms team's forced-HP fuzz;
A/B on their tuple in `bench\h1-combat\popaction_*.json`). Root cause: `notes\host-h1-combat.md` section 5.
The H0 combat harness (`SpireHost.Sim.CombatHarness`) and the replay tool do not load driver modules and therefore
do not get the patch today. The only other awaited `SaveManager.SaveRun` in game logic is the one at room entry
(Runs/RunManager.cs:778); it is synchronous in the host as well. No effect of that one was observed in any run;
whether it should get the same boundary is a decision for the owner of the pump.

**Until then.** Installed by `CombatModule.Register` (and reported by `--list-patches` of any console that loads the
combat module). When a copy is added to the core, delete the `CombatScheduling.Install()` call from
`CombatModule.Register` in the same change: a second transpiler on that MoveNext finds 0 calls of
`SaveManager.SaveRun` left and fails closed at startup ("expected exactly 1").

## C6. Retire the core's `selection-context-choose-a-card` together with the temporary selector

**What.** `SpireHost.Core\Patches\SelectionContextPatches.cs` (a read-only prefix on
`CardSelectCmd.FromChooseACardScreen`) only feeds `TemporaryHandlers.CardSelectLifecycle`, which the combat module
removes (`registry.RemoveLifecycle`). After integration the method carries two read-only prefixes (the core's and
`combat-select-capture-choose-a-card`); harmless, but the core one has no reader.

## C7. One place to register "canonical classes of identical cards" (= obs request O1)

**What.** A driver-level registration point, so that the CARD_SELECT owner and the encoder need not reference each
other:

```csharp
// SpireHost.Driver
public delegate void CanonicalCardClasses(RunContext ctx, IReadOnlyList<CardModel> cards, PileType pile, int seat,
                                          List<CardModel> classes, List<int> counts);
public sealed class DriverRegistry { public void SetCanonicalCardClasses(CanonicalCardClasses provider, string name); ... }
public sealed class RunContext    { public CanonicalCardClasses? CanonicalCardClasses { get; } ... }
```

(the signature of `SpireHost.Obs.Encoding.ObsEncoder.CanonicalClasses`).

**Why.** obs O1: the deduplicated CHOICE rows must be strictly ascending by the ENCODER's content key
(obs-schema-v1 5.2.3: encoded row + printed values; it contains vocabulary ids, so nobody else can compute it), and
DRIVER-API.md 6.4 makes the CARD_SELECT owner responsible for the class order. The combat project builds against
the snapshot only and may not reference `SpireHost.Obs`.

**Shim until then (implemented and tested).** `SpireHost.Driver.Combat.CanonicalCandidates.Provider`
(`CardSelection.cs`), a static delegate of exactly that signature; `null` = the local key (`CardClassKey`: every
field `CardModel.CompareTo` reads plus the card face; also independent of the pile order, but a different order).
The integrator writes one line after creating the encoder:

```csharp
SpireHost.Driver.Combat.CanonicalCandidates.Provider = encoder.CanonicalClasses;
```

The frame takes classes, sizes and order from the provider, finds the members of each class with the provider
itself (pair queries), checks every answer and aborts the episode on an inconsistent one
(`HandlerFault/card_select_class_provider`). Proven with the real obs encoder through the test-only preview
`--with-modules ...,obs` (numbers in the report, section 4.6) and with a test provider in `--selftest --suite
drawperm`. Note for obs: `sel_count` of a merged class needs C2 `OptionPicked`.

## C8. `ScenarioTools`: mid-combat tools (additive, test only)

**What.** Move `SpireHost.Driver.Combat.Tests.CombatTestTools` into `SpireHost.Driver.Tools.ScenarioTools`:

```csharp
Task<CardModel> AddCard(RunContext ctx, string cardEntry, PileType pile = PileType.Hand, int upgrades = 0, int seat = 0);  // dev console "card"
Task SetResources(RunContext ctx, int energy, int stars, int seat = 0);                                                    // "energy" / "stars"
Task ApplyPower(RunContext ctx, string powerEntry, int amount, Creature target);                                           // "power"
Task Afflict(RunContext ctx, string afflictionEntry, int amount, CardModel card);                                          // "afflict"
```

**Why.** DRIVER-API.md 9 lists run-level tools only (`GiveRelic`, `GiveCard`, `JumpToEncounter` ...); every
selection-heavy combat scenario (and the obs leak tests on draw-pile prompts) needs a known hand / pile / resource
state. All four are the game's own dev-console paths (DevConsole.ConsoleCommands).

## C9. Documentation: DRIVER-API.md 12 and DECISIONS-H1 B.2 / B.3

* "The 13 `CardSelectCmd.From*` captures" (DRIVER-API.md 12; DECISIONS-H1 B.3 "全部 13 个"): v0.113 has **15
  overloads under 12 names** that ask an `ICardSelector`, plus `FromChooseABundleScreen`, which the events team owns
  (gate + BUNDLE frame). The combat module captures the 15 and asserts at boot that it installed exactly 15; it does
  not touch the bundle entry (self-test case `relic SCROLL_BOXES (bundle entry is not ours)`).
* DECISIONS-H1 B.2 says the local-selector path keeps "`ReserveChoiceId` / `SignalPlayerChoiceBegun` /
  `SyncLocalChoice`". The game's local-selector branch runs the first two and `SignalPlayerChoiceEnded` but does NOT
  call `SyncLocalChoice` (e.g. Commands/CardSelectCmd.cs:283-292 vs :294-300); only the real-screen branch does.
  `SyncLocalChoice` changes no game state (GameActions.Multiplayer/PlayerChoiceSynchronizer.cs:64-76: validates the
  id, raises `PlayerChoiceReceived`, sends a net message). See C10.
* design-review-eng.md 3.4 says the UI does not check `IsHittable` when aiming; DECISIONS-H1 B.1 and
  driver-spec-combat.md 2.3 say AnyEnemy -> `HittableEnemies`. The source agrees with DECISIONS
  (Nodes.Combat/NTargetManager.cs:302-313, :417-441); implemented that way.

## C10. In-episode `.mcr` recording (replay R2) needs the choices: decide who calls `SyncLocalChoice`

**What.** If replay R2 (`EpisodeOptions.CombatChecksums`, writing the game's own `.mcr` inside an episode) is
accepted: `CombatReplayWriter` records player choices from `PlayerChoiceSynchronizer.PlayerChoiceReceived`
(Multiplayer.Replay/CombatReplayWriter.cs:47), which for the local player only `SyncLocalChoice` raises
(GameActions.Multiplayer/PlayerChoiceSynchronizer.cs:74; :135 is the path of a received remote choice) - and the
local-selector path never calls it (C9). A recording made by the host would then contain the actions but not the
answers of selection prompts and could not be replayed.

**Proposed.** Either `HostCardSelector` calls `PlayerChoiceSynchronizer.SyncLocalChoice(player, choiceId,
PlayerChoiceResult.FromMutableCombatCards / FromMutableDeckCards(result))` when recording is on (the choice id is
the one `ReserveChoiceId` returned inside the captured `From*` call: capture it in the same prefix), or the replay
team records from the CARD_SELECT frames. Not implemented: nothing records in H1.
