# SpireHost fidelity ledger - obs team fragment (H1)

> **MERGED into `host\FIDELITY.md` at the H1 integration (2026-11-15).** This fragment is kept for history. It describes the
> team's stand-alone build against the stage-0 snapshot and is no longer checked against the registry: ids, counts,
> switches and "temporary" / "shim" remarks below may be out of date. The merged ledger is the source of truth
> (its section 4 lists what was removed or moved at the integration).

Owner: obs team (`SpireHost.Obs`). Same format and rules as `host\FIDELITY.md`; the lead merges this fragment.
Source references are `ref\sts2-v0113-src` paths without the `MegaCrit.Sts2.Core.` prefix.

**Registry entries added by this team: 0.** `ObsModule.Register` installs no handler, no lifecycle, no frame
extension and no Harmony patch (Plumbing 0, Scheduling 0, Instrumentation 0, Fidelity 0, Behaviour 0). The encoder
never patches the game; it only reads game objects at quiescent decision points (and at the terminal state), on the
engine thread, through the publicized reference (no reflection on the hot path).

No preview computation needed a guard: nothing in this fragment is an open request for a patch.

---

## A. Patches

None.

## B. Game methods the encoder calls that WRITE state (DECISIONS-H1 F.8, obs-schema-v1 2.1-4 / 14.4)

Each is undone before the encoder returns; the proof is the "no side effects" check of `--selftest` (case `fuzz` and
`scenarios`: `DeepSignature` before / after every encode of every decision, policy and auto-advanced, plus 1,000+
synthetic CARD_SELECT prompts per run; 0 differences) and the byte-identical second encode.

| Call | What it writes | How it is undone | Evidence |
|---|---|---|---|
| `CardModel.UpdateDynamicVarPreview(mode, target, set)` + `DynamicVarSet.ClearPreview()` (card + enchantment variable sets; untargeted, per target, upgrade preview) | `DynamicVar.PreviewValue`, `EnchantedValue` | values saved before (`ObsEncoder.Save`) and written back (`Restore`) in a `finally` | game logic never reads `PreviewValue` (only the two icon formatters do, schema 5.3.1) - restoring makes the encoder invisible even to the UI-state a later frame would read |
| `ICardScope.CloneCard(card)` + `UpgradeInternal()` + `UpgradePreviewType = Deck/Combat` (upgrade preview, `Nodes.Cards/NUpgradePreview.cs:93-99`) | adds the clone to `CombatState._allCards` / `RunState._allCards` | `scope.RemoveCard(clone)` in a `finally` | DeepSignature includes both `_allCards` counts |
| `LocalContext.NetId = Players[s].NetId` (per-seat `PowerModel.IsVisible`, `AttackIntent.GetSingleDamage`) | the process-wide "who am I" | saved and restored in a `finally` | DeepSignature includes `LocalContext.NetId` |
| `CardModel.LocalKeywords` (private getter, `Models/CardModel.cs:484-495`) | lazily creates `_keywords` from `CanonicalKeywords` on first access | not undone: the game's own `Keywords` getter performs the identical, idempotent initialisation whenever the UI or the engine reads a card's keywords | DeepSignature unchanged (it does not observe the lazily created set's identity) |

## C. Encoder-side computations that reproduce a game method (exactness tested)

`Encoding\HookCache.cs` evaluates five read-only display values per card with the frame's hook listeners enumerated
once instead of once per call (the game enumerates `CombatState.IterateHookListeners()` - a fresh list of every
power / relic / potion / orb / card / enchantment / affliction - on every hook call; that was 60-70 % of the encoder's
time). The replicas are line-for-line transcriptions with the same early returns, listener order and decimal
arithmetic; listeners whose type does not override the hook method (reflection once per type) run
`AbstractModel`'s identity implementation and are skipped.

| Value | Game method reproduced | Hook dispatch reproduced |
|---|---|---|
| CARDS.energy_cost | `CardEnergyCost.GetWithModifiers(CostModifiers.All)` (`Entities.Cards/CardEnergyCost.cs:56-83`) | `Hook.ModifyEnergyCostInCombat` (`Hooks/Hook.cs:1428-1444`, incl. `ShouldSkipCombatEffect`) |
| CARDS.star_cost | `CardModel.GetStarCostWithModifiers()` (`Models/CardModel.cs:1382-1394`) | `Hook.ModifyStarCost` (`:1820-1832`) |
| CARDS.keyword_bits | `CardModel.Keywords` = `GetKeywordsWithSources(All)` (`:1003-1031`) + `ShouldRetainThisTurn` / `IsSlyThisTurn` | `Hook.ModifyKeywordsInCombat` (`:1446-1452`) |
| CARDS.energy_cost_color, star_cost_color (hand) | `CardCostHelper.GetEnergyCostColor` / `GetStarCostColor` (`Helpers.Models/CardCostHelper.cs`), given the card's one `CanPlay` result | `TryModifyEnergyCostWithHooks` / `TryModifyStarCostWithHooks` (same file) |

Self-check: with `EncoderOptions.ExactGameCalls = true` every value comes from the game method itself; `--selftest`
encodes every frame both ways and requires identical bytes (`exact_compared` / `exact_mismatch` in the `fuzz` case;
0 mismatches over 3,449 frames in the run of notes/host-h1-obs.md 4.2). A card whose `CombatState` is not the
frame's combat state takes the game method. This is a dispatch cache, not a rule: no card / relic / power logic is
re-implemented, every `TryModify*` that can change the value is the game's own.

Also skipped as provably no-ops (same file / `ObsEncoder.Cards.cs`): per-target previews of hand cards whose printed
variables are all target-independent (only `DamageVar`, `OstyDamageVar`, `PowerVar<T>` and `CalculatedVar` read the
preview target in `Localization.DynamicVars/*.cs`; no card overrides `UpdateDynamicVarPreview`). The schema writes a
per-target row only when the value differs from the untargeted one, so skipping them changes no byte.

## D. UI-only flows performed by proxy

None.

## E. Accepted differences / notes for the integrator

* The real client locks potions while a room fades in (replay team, `FIDELITY.replay.md` D). The encoder derives
  POTIONS.flags bit0 ("usable now") from the actions the driver offers, so it follows whatever the potion extension
  decides; no encoder change.
* Encoder reads that are host-only by contract and never enter a frame: `RunContext.Seed`, `RunSignature`,
  `StateSignature`, `CrystalSphereCell.Item` (hidden item layout; the encoder reads only `IsHidden`, `DivinationCount`
  and the tool - item columns of revealed cells come from the events team's public view through
  `IScreenViewProvider`, see `notes\schema-errata-v1.md` E-30). Enforced by the IL audit of `--selftest` (case `audit`).
* Test-only code that writes hidden state on purpose lives in `SpireHost.Obs.Tests` (leak perturbations P1-P14,
  synthetic CONFIRM / BUNDLE / CRYSTAL frames on a sacrificial scenario) and is never reachable from `--serve`.
