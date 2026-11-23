# Contract requests - events team (`SpireHost.Driver.Events`)

## Status after the H1 integration (2026-11-15, contract version `h1-integrated`)

The lead's rulings are in the integration brief; how each was implemented: `notes\host-h1-integration.md`. "Shim removed" means the team's local stand-in described further down no longer exists. The text below the table is the original request, kept for history.

| Request | Subject | Status | Where / how |
|---|---|---|---|
| R1 | EventContext fields | **ACCEPTED, implemented** | `EventContext` (`EventId`, `PageKey`, `IsShared`, `IsAncient`, `PageVars`, `Options[]` = `EventOptionInfo`) with `BeginPage` / `AddOption` / `AddSynthesizedProceed` in `SpireHost.Driver\DecisionContexts.cs`; printed variables through `LocText.PrintedVars` (template order). `EventViews.cs` is deleted; the encoder reads the context. |
| R2 | CrystalContext public view | **ACCEPTED, implemented** | `CrystalContext` public view (`Width`, `Height`, `Divinations`, `Tool`, `Cells`, `Exposed`, `ExposedComplete`, `FillPublicView`); `Minigame` documented HOST-ONLY. The encoder reads only the view; leak tests P11 / C6 run on the real CRYSTAL frames. |
| R3 | `RunManager.Proceed()` -> OpenMap in the core | **ACCEPTED (the lead left it to the integrator)** | Driver patch `run-proceed-opens-map` (postfix on `RunManager.Proceed`). Accepted because `OpenMap` is idempotent and every owner now calls the game's own `Proceed()` and asserts the map is open (`map_not_opened`): the synthesized PROCEED, the FakeMerchant PROCEED, the Ancient PROCEED path, rest / shop LEAVE and the terminal proceed. The events shim (direct `ctx.OpenMap` calls) is removed. |
| R4 | ScenarioTools helpers | **ACCEPTED, implemented** | `ScenarioTools.JumpToEvent(ctx, EventModel)`, `EnterAct`, `SetVisitedFloors` (test only); the local copies are deleted. |
| R5 | Ownership text | **ACCEPTED, done (no change of ownership)** | The events module owns the event null-dereference patches, the bundle gate + BUNDLE frame and FoulPotion at the FakeMerchant (FIDELITY.md 1.9, section 7 E-ev1 .. E-ev4; DRIVER-API.md section 11). |
| note: rooms | potion extension on the FakeMerchant SHOP frame | **DONE (ruling D4)** | `rooms:potions` acts on the frame (`ShopContext.FakeMerchant`); the two FoulPotion postfixes compose (declared shared site); `FoulPotionTestExtension` is deleted; tests: `events --selftest` `fake_merchant_foul_potion_fight`, `rooms --selftest` `shop:foul-potion`, `integration:fake-merchant-foul-potion`. |

---
Snapshot used: `h1-stage0` (`host\snapshot\current`, stamp 20261114-155134). All requests are additive (DRIVER-API.md
section 14, item 2) unless marked. Each one has a local shim in the events project so that the work did not block;
the shim is named in the request and is deleted when the request is integrated.

---

## R1. `EventContext`: the event screen as the UI shows it (additive)

**What.** Fields on `SpireHost.Driver.EventContext` (DecisionModel.cs:544-550) so the encoder gets option keys, the
printed numbers and the referenced entities without reading around the contract:

```csharp
public sealed class EventContext {
    public EventModel Event = null!;
    public bool Finished;
    // new:
    public string EventId = "";                       // Event.Id.Entry (vocab event)
    public string? PageKey;                           // Event.Description?.LocEntryKey (vocab event_page)
    public bool IsShared, IsAncient;                  // global 33 bits 0/1
    public readonly List<KeyValuePair<string,int>> PageVars;       // numeric vars printed by the page template
    public readonly List<EventOptionInfo> Options;    // every displayed button in display order (locked included);
                                                      // the synthesized PROCEED (Index -1) when Finished
}
public sealed class EventOptionInfo {
    public int Index;                                 // index in CurrentOptions = GameRef ordinal; -1 synthesized PROCEED
    public EventOption? Option;
    public string TextKey = "";                       // vocab event_option_key
    public bool IsLocked, IsProceed, WasChosen, WillKillPlayer, Legal, Synthesized;
    public RelicModel? Relic;                         // Ancient relic icon
    public readonly List<KeyValuePair<string,int>> Vars;           // printed numbers (title + description)
    public readonly List<CardModel> Cards;            // CardHoverTip.Card, hover-tip order
    public readonly List<AbstractModel> Models;       // other hover tips' CanonicalModel (relic, potion, power, enchantment)
}
```

**Why.** obs-schema-v1 5.11 (OPTIONS kind EVENT_OPTION: key_id, flags bit0/1/5/6, VARS, REFS, ref_row for the
relic) and section 4 global 31-33; DRIVER-API.md section 12 "events": "Fill the contexts completely for the
encoder". The UI evidence for each field is in `NEventOptionButton.cs:181-214, 266, 337` and `NEventRoom.cs:205-232`.

**Shim.** `SpireHost.Driver.Events.EventViews` (`EventViews.cs`): `EventViews.Of(EventContext)` returns an `EventView`
with exactly these fields, attached to the handler's context object (ConditionalWeakTable). Identical field names,
so the move is mechanical.

## R2. `CrystalContext`: a public-only view of the sphere (additive + one documentation rule)

**What.** Fields on `CrystalContext` (DecisionModel.cs:589-592):

```csharp
public int Width, Height, Divinations;
public CrystalSphereMinigame.CrystalSphereToolType Tool;
public readonly List<CrystalCellView> Cells;          // (y, x) order, index = y * Width + x = CrystalCell ordinal
public readonly List<CrystalSphereItem> Exposed;      // items with >= 1 cleared cell, by first cleared cell in scan order
public readonly List<bool> ExposedComplete;           // fully revealed (shape + position public)
public struct CrystalCellView { public int X, Y; public bool Hidden; public int ItemType /*-2 hidden, -1 empty*/, ItemGood, ItemIndex; }
```

and a sentence in DRIVER-API.md 6.2 / 8.4: **`CrystalContext.Minigame` is host-only** - its `cells[x,y].Item` and
`Items` expose the hidden item layout (`CrystalSphereItem.PlaceItem`, CrystalSphereMinigame.cs:145-189; obs-schema
5.12: "物品位置属于隐藏信息"); the encoder reads only the view.

**Why.** obs-schema-v1 5.12 (CELLS) and global 41-42; fair-play boundary DESIGN.md 0.5 / DRIVER-API.md 8.4.

**Shim.** `EventViews.Of(CrystalContext)` -> `CrystalView` (`EventViews.cs`), filled by `CrystalFrame.Build`.
Invariant tested on every CRYSTAL frame: hidden cells carry no item data.

## R3. `RunManager.Proceed()` -> `RunContext.OpenMap` in the core (additive, low priority)

**What.** A core-owned postfix (or an explicit hook) so that a game-initiated `RunManager.Proceed()` opens the host
map: `HostSession.Current?.CurrentEpisode?.OpenMap("RunManager.Proceed")`.

**Why.** `RunManager.Proceed()` (RunManager.cs:1289-1296) only talks to `NMapScreen`. Game code reaches it from event
options: the Ancient `PROCEED` with `OnChosen = NEventRoom.Proceed` (AncientEventModel.cs:172; dead in v0.113
because no model overrides `ShouldAllowAncient`, but the code path exists). DRIVER-API.md 7.3 currently requires
every owner to call `OpenMap` by hand.

**Shim.** `EventRoomHandler.Apply`: after `option.Chosen()` of an `IsProceed` option whose `OnChosen.Method` is
`NEventRoom.Proceed`, the handler calls `ctx.OpenMap`. Remove it when the core does this generically.

## R4. `ScenarioTools` helpers for event tests (additive)

```csharp
public static Task JumpToEvent(RunContext ctx, EventModel canonical);   // Ancients / TheArchitect are not in ModelDb.AllEvents
public static Task EnterAct(RunContext ctx, int actIndex);               // RunManager.EnterAct(actIndex, doTransition: false)
public static void SetVisitedFloors(RunContext ctx, int totalFloor);     // RunState.AppendToMapPointHistory until TotalFloor >= n
```

**Why.** DRIVER-API.md 9 ("construct those states with ScenarioTools"): 14 events read
`CurrentActIndex` in `IsAllowed` (10 of them need act >= 1 or exactly 1), PunchOff / SlipperyBridge on `TotalFloor`, and `ScenarioTools.Event(id)` only searches
`ModelDb.AllEvents` (62 of the 73 event types).

**Shim.** `EventIds.JumpTo`, `Cases.Act`, `Sweep.Apply` in the events project (test code only).

## R5. DRIVER-API.md section 12 text (documentation only)

Section 12 "events" says "The null-dereference patches themselves are fidelity's" and "BUNDLE frame ... (the gate
patch is fidelity's, the frame yours)". The H1 task assignment gives the events team exclusive ownership of the
patches inside event models / event UI stand-ins, the bundle gate and FoulPotion at the FakeMerchant; they are
implemented in `SpireHost.Driver.Events` (`host\FIDELITY.events.md`). Please amend section 12 and the "Open" rows of
FIDELITY.md B/D (`CardSelectCmd.FromChooseABundleScreen`, the seven event sites, FoulPotion FakeMerchant) when merging.

## Coordination notes (no contract change)

* **rooms**: the out-of-combat potion extension must also act on the FakeMerchant SHOP frame (`Decision.Kind == Shop`,
  `Source` = `events:event`, `ShopContext.FakeMerchant == true`): FoulPotion is `TargetedNoCreature` out of combat ->
  `USE_POTION` with no target -> `EnqueueManualUse(null)`; `PassesCustomUsabilityCheck` is true there through
  `events-fake-merchant-foul-potion-usable`. The rooms team's own postfix on the same getter must only handle the
  `MerchantRoom` branch (both postfixes compose: each only turns false into true under its own room condition).
  Until then the events console registers a test-only stand-in (`FoulPotionTestExtension`) under `--selftest` only.
* **rooms**: rewards offered by events (Crystal Sphere, Trial, PunchOff NAB, BattlewornDummy resume ...) are
  non-terminal `RewardsCmd.OfferCustom` sets; event combats come back through the terminal proceed tree
  (`ProceedFromTerminalRewardsScreenAsync` -> `ResumePreviousRoom` for BattlewornDummy). Both work with the temporary
  reward handler and are covered by the events sweep.
* **obs**: `ConfirmContext.Key` for Trial is `"main_menu_ui/ABANDON_RUN_CONFIRMATION.header"` (the popup header,
  NAbandonRunConfirmPopup.cs:65), `SourceModel` = the Trial event; CONFIRM actions are `CHOOSE_OPTION(ConfirmChoice
  0 = no / 1 = yes)`. CRYSTAL `Aux` = `(int)CrystalSphereMinigame.CrystalSphereToolType` (Small 1, Big 2).
  BUNDLE: `CHOOSE_OPTION(GameRef.OfBundle(cards, i))`, `SKIP` when `BundleContext.CanSkip`.
