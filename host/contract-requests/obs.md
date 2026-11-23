# Contract requests - obs team (`SpireHost.Obs`)

## Status after the H1 integration (2026-11-15, contract version `h1-integrated`)

The lead's rulings are in the integration brief; how each was implemented: `notes\host-h1-integration.md`. "Shim removed" means the team's local stand-in described further down no longer exists. The text below the table is the original request, kept for history.

| Request | Subject | Status | Where / how |
|---|---|---|---|
| O1 | Canonical candidate classes | **ACCEPTED, implemented (= combat C7)** | see combat C7. The encoder's `CanonicalClasses` takes the `unpoweredPreviews` parameter its key needs; the order check of the encoder (`ChoiceOrderViolations`) is 0 in every self-test and in serve mode. |
| O2 | EventContext / CrystalContext fields; the view seam | **ACCEPTED, implemented (follows events R1 / R2)** | The encoder reads the context fields; `IScreenViewProvider`, `ScreenViews.cs` and the adapter are deleted (ruling D6: contexts only). |
| O3 | Public history counters | **ACCEPTED, implemented in the core** | `EpisodeCounters.CombatsWon`, `ElitesWon`, `BossesWon`, `LastWonRoomType`, `LastWonTotalFloor`, maintained from `CombatManager.CombatWon`; the encoder's own `PublicHistory` and the protocol server's subscription are deleted. |
| O4 | `--serve` / `--export-vocab` from the main executable | **ACCEPTED, implemented (ruling A3)** | Top-level commands of `SpireHost.exe`; the obs console no longer exists (the project is a library). |
| O5 | Documentation: STEP_ID source, decision cap | **ACCEPTED, done** | DRIVER-API.md 4.5. |
| O6 | Decision.Seal for tests | **ACCEPTED, implemented** | `Decision.SealForTest(allowAbandon)`; the synthetic special frames use it. |
| O7 | ConfirmContext.Key registry | **ACCEPTED, implemented** | `SpireHost.Driver.ConfirmKeys` (`All`, `No`, `Yes`); the vocabulary category `confirm_key` is built from it; `obs --selftest` case `keys` enumerates it. |
| O8 | Rooms contexts the encoder relies on | **INFORMATION, satisfied** | the rooms module fills them; `TreasureContext.Finished / Empty / Awarded` (rooms R1) are read by the encoder. |

---
To: the Core/Driver owner (lead). From: obs team. Snapshot used: `h1-stage0` (`host\snapshot\current`, stamp
20261114-155134). All requests are additive (DRIVER-API.md section 14, item 2). Each names the local shim the obs
code uses until it lands; nothing here blocks the obs deliverable.

---

## O1. CARD_SELECT owner: canonical candidate classes for pile prompts (behavioural requirement + helper)

**What.** For every selection prompt whose candidates come from a pile (`Pile` in Draw / Discard / Exhaust / Deck),
`CardSelectContext.Options` must be the canonical classes and `OptionCounts` their sizes, in the order of the
encoder's content key (obs-schema-v1 5.2.3 / 5.2.4, DECISIONS-H1 B.4, DRIVER-API.md 6.4 "Draw-pile candidates").
The encoder exposes the key so the combat team does not re-implement it:

```csharp
// SpireHost.Obs.Encoding
encoder.CanonicalClasses(ctx, cards, pile, seat, classes, counts);   // classes in canonical order + class sizes
PublicCardKey.Classes(encoder, cards, classes, counts);              // same, ctx = HostSession.Current.CurrentEpisode
```

The representative of a class is its first instance in pile order (the instance handed to the game when the class is
chosen; instances encode identically, so the choice is not observable).

**Why.** The deduplicated CHOICE rows (pos = -1) must be strictly ascending by content key (Python's deep checker
asserts it, `spire.proto._multiset_problems`), and the ACTIONS rows (driver order) must follow the CHOICE rows. The
encoder cannot reorder candidates itself without breaking the driver's action order. With the temporary
CARD_SELECT handler (unmerged, pile order) every such frame leaks the draw order and is reported by
`EncoderStats.ChoiceOrderViolations`.

**Integration note.** Either the combat module references `SpireHost.Obs` (one call per prompt), or the driver
gets a hook `Func<RunContext, IReadOnlyList<CardModel>, PileType, (List<CardModel>, List<int>)>` that the obs module
registers. One detail: when `CardSelectorPrefs.UnpoweredPreviews` is set, the CHOICE rows are encoded without the
preview pass; the class key must then be computed the same way (pass `unpowered` - add a parameter if the combat team
needs it; v0.113 sets it only on non-pile prompts as far as the source shows).

**Shim.** none needed in obs; `EncoderStats.ChoiceOrderViolations` counts violations (0 in all self-test frames,
which build their pile prompts with `CanonicalClasses`).

## O2. EventContext / CrystalContext richer fields (= events team R1 / R2) and the view seam

**What.** Accept the events team's R1 / R2. The encoder already reads EVENT screens completely from
`EventContext.Event` (white-listed members, schema 2.2), but for CRYSTAL it must not read `CrystalContext.Minigame`
cells' `Item` (hidden layout). Until R2 lands, the encoder exposes a provider seam:

```csharp
// SpireHost.Obs.Encoding.ScreenViews.cs
public interface IScreenViewProvider {
    bool TryEvent(EventContext context, Player player, EventScreen into);
    bool TryCrystal(CrystalContext context, CrystalScreen into);
}
ScreenViews.Provider = new EventsViewAdapter();   // set once by the integrator
```

`EventScreen` / `CrystalScreen` carry exactly the fields of the events team's `EventView` / `CrystalView`
(`SpireHost.Driver.Events\EventViews.cs`); the adapter is a field copy (`CrystalCellView.ItemType -2 / -1 / type`,
`ItemGood`, `ItemIndex` map 1:1 onto `CrystalScreen.Cell`). After R2, `TryCrystal` reads the new `CrystalContext`
fields directly and the adapter is deleted.

**Until integration** (no provider): CELLS.item_type of revealed cells = 1 (UNK), item_good = item_idx = -1.
Everything else in EVENT / CRYSTAL / CONFIRM / BUNDLE frames is complete. (`--selftest` case `special` proves both
paths, including P11 with a test view: hidden cells' items never change the bytes.)

## O3. Public history counters in `EpisodeCounters` (low priority)

**What.** Add to `SpireHost.Driver.EpisodeCounters` (filled by the core from `CombatManager.CombatWon`):

```csharp
public int ElitesWon, BossesWon;            // combats won in rooms of type Elite / Boss
public int LastWonRoomType = -1;            // (int)RoomType of the most recent won combat, -1 = none
public int LastWonTotalFloor = -1;          // RunState.TotalFloor at that moment
```

**Why.** errata E-23 (GLOBAL 44-48: explicit last-combat-won signal the Python reward code asked for). Today the
obs observers (`ProtocolServer`, `FrameDump`) subscribe `CombatManager.Instance.CombatWon` themselves in
`OnEpisodeStart` and unsubscribe in `OnEpisodeEnd`; a core counter removes that duplicated subscription and makes the
value available to any encoder user (e.g. a future in-process actor).

**Shim.** `SpireHost.Obs.Encoding.PublicHistory`, maintained by the observers.

## O4. Main executable: route `--serve` and `--export-vocab` to `SpireHost.Obs`

**What.** At integration the main `SpireHost.exe` must offer `--serve [--pipe NAME] [--max-decisions N] [--turn-cap N]
[--vocab-dir DIR]` and `--export-vocab DIR [--merge OLD]` by calling `SpireHost.Obs.ObsCommands.Run` (or the obs
console becomes the shipped executable), and ship `vocab\vocab.json` / `vocab\catalog.json` next to it (the obs
csproj copies `SpireHost.Obs\data\*.json` to `bin\...\vocab\`).

**Why.** Python's `HostConfig.command` defaults to `(<repo>\host\SpireHost\bin\Release\net9.0\SpireHost.exe, "--serve")`
(`py\spire\host.py`). Until then the Python side passes `("dotnet", "<repo>\host\SpireHost.Obs\bin\Release\net9.0\SpireHost.Obs.dll", "--serve")`
(what `py\scripts\smoke_real_host.py` does).

## O5. Document the STEP_ID source and the decision-cap wiring (documentation)

**What.** DRIVER-API.md 12 "obs" says "STEP_ID = `Counters.PolicyDecisions`, AUTO_STEPS from
`Counters.AutoSingleTotal + AutoGold`". Please add to 4.5 that `PolicyDecisions` is incremented after
`IPolicy.Decide` returns (`RunDriver.cs:709-715`), so the value read inside `Decide` is the 0-based index of this
decision - the protocol relies on it (errata P-6) - and that `EpisodeOptions.MaxPolicyDecisions` is
checked before the policy is asked (so a cap of N yields exactly N OBS frames, TERMINAL STEP_ID = N).

## O6. `Decision.Seal` for tests (optional)

**What.** A public `Decision.SealForTest()` (canonical sort + duplicate check) so that test code which builds
synthetic decisions (obs `Tests\SpecialFrames.cs`, `EncodeChecker`) cannot hand the encoder an unsorted list by
mistake. Today the synthetic builders add actions in canonical order by hand.

## O7. `ConfirmContext.Key` registry (process)

**What.** Every driver that raises a CONFIRM decision registers its key in
`SpireHost.Obs.Vocab.VocabExport.ConfirmKeys` (append-only) and re-exports the vocabulary. Current keys: `NO`, `YES`
(the two buttons), `main_menu_ui/ABANDON_RUN_CONFIRMATION.header` (events team, Trial.DoubleDown), `trial_double_down`
(snapshot doc example). An unregistered key encodes as UNK and is reported once on stderr.

## O8. Rooms contexts the encoder relies on (for the rooms team, no contract change)

The encoder reads, per kind: `RewardContext.Set.Rewards` (skips `SuccessfullySelected`; `GoldReward.Amount`,
`PotionReward.Potion`, `RelicReward.Relic`, `SpecialCardReward._card`; never `CardReward` contents - F.4),
`RewardsSet.DisallowSkipping`, `CardRewardContext.{Reward, Options, Alternatives}`, `ShopContext.{Inventory,
FakeMerchant}`, `RestContext.{Options, Successes}`, `TreasureContext.Relics` (+ `Decision.SubKind` = TreasurePhase).
The rooms team's TREASURE R1 (`Finished`, `Empty`, `Awarded`) is welcome; until then the PROCEED-only page encodes as
RELIC phase with no TREASURE_RELIC rows, which is unambiguous from the action list.
