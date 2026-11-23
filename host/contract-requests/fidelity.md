# Contract requests - fidelity team (SpireHost.Fidelity)

## Status after the H1 integration (2026-11-15, contract version `h1-integrated`)

The lead's rulings are in the integration brief; how each was implemented: `notes\host-h1-integration.md`. "Shim removed" means the team's local stand-in described further down no longer exists. The text below the table is the original request, kept for history.

| Request | Subject | Status | Where / how |
|---|---|---|---|
| R1 | IEpisodeCheck and module counters | **ACCEPTED, implemented (= combat C4 = rooms R3)** | `IEpisodeCheck` + `DriverRegistry.SetEpisodeCheck`: evaluated by the core at the end of every settle and before teardown. `FidelityInvariants` is the lifecycle (run set-up invariants) and the episode check (the swallowed-assertion backstop and the static-hook invariants at every quiescent point); the frame-extension shim is gone. `FidelityCounters` = module `fidelity`. |
| R2 | Discovery-order check compares the whole room set | **ACCEPTED, implemented** | Core `fidelity-discovery-order-noop-check` fingerprints boss + events + normal / elite encounters + second boss + ancient; `StrictCounters.DiscoveryRoomsChanged` (`ProfileGateFired/discovery_order_rooms`). The pack's duplicate patch `fidelity-discovery-order-rooms-check` is removed (one patch per site). |
| R3 | ForceDiscoveryOrderModifications = the real predicate | **ACCEPTED, implemented** | `RunDriver.StartRun`: `rm.ForceDiscoveryOrderModifications = run.GameMode == GameMode.Standard && !run.IsCustomGame`. |
| R4 | Two Core scheduling patches | **ACCEPTED, implemented (ruling B3)** | `virtual-clock-executor-unpause-poll` and `virtual-clock-creature-spawn-timeout` are removed from the Core; the pack's unpause transpiler now accepts exactly the original `Task.Delay(int)`. |
| R5 | NativeAllowList | **NO CHANGE (as requested)** | reviewed; `unexpected_native_calls` = {} in the integration fuzz. |

---
Against snapshot `h1-stage0` (`host\snapshot\current`). Each request: what, why, spec paragraph, proposed
signature, and the local shim the pack uses until the lead integrates it. All requests are additive.

---

## R1. Module-owned episode checks and counters (`IEpisodeCheck`)

**What.** A registry slot for read-only checks that the driver core evaluates at every quiescent point and at
the end of the episode, and whose counters are folded into `EpisodeResult` and the `--fuzz-runs` JSON.

**Why.** The fidelity pack fails an episode when a "proven unreachable" path is hit (assertions on
`CombatState.GodotTimerTask`, the hover / input-sync delays, the test-only energy-cost overrides, the
discovery-order room fingerprint). The assertion throws `EpisodeAbortException` at the site; but game code may
swallow it (e.g. `CombatManager.WaitForPreviousTurnLoopToFinish` has `catch (Exception) { }`,
Combat/CombatManager.cs:369-371), and then nothing fails the episode unless a later check looks at the counter.
The strict-mode counters that the core checks (`StrictCounters`, `RunDriver.CheckFaults`) are a fixed struct in
Driver, so a module cannot add one. Also: the pack's counters (boulder hits, natural-loss `OnEnded` calls,
unpause waits, room-set checks ...) do not appear in `--fuzz-runs` output.

**Spec.** DRIVER-API.md 5.5 (strict mode), 5.6 (abort reporting), 10 ("Patches ... self-check (a counter ...)"),
DECISIONS-H1 A.7 (fail-closed).

**Proposed signature (Driver).**

```csharp
public interface IEpisodeCheck
{
    string Name { get; }
    /// Called by the core at every quiescent point (inside CheckFaults) and once more before teardown.
    /// Return null if fine, else a short snake_case detail: the episode ends Aborted(StrictViolation, detail).
    string? Check(RunContext ctx);
    /// Called once at the end of every episode (after teardown, before OnEpisodeEnd); add counters (deltas of this episode).
    void Report(IDictionary<string, long> counters);
}
public sealed partial class DriverRegistry { public void SetEpisodeCheck(IEpisodeCheck check); }
public sealed partial class EpisodeResult { public SortedDictionary<string, long> ModuleCounters; }   // summed by --fuzz-runs
```

**Local shim.** `SpireHost.Fidelity.FidelityInvariants` is registered as an `IRoomFrameExtension` that adds no
action and throws when an assertion fired in this episode (`fidelity_assert_swallowed`): it only runs at
room-level decisions, so a swallowed assertion followed directly by the terminal state is not caught. Counters
are printed once per process on stderr (`[fidelity] counters {...}`) and summed by
`bench\h1-fidelity\summarize_fuzz.py`.

---

## R2. Discovery-order no-op check should compare the whole room set

**What.** Fold the room-set fingerprint of `fidelity-discovery-order-rooms-check` into Core's
`fidelity-discovery-order-noop-check` and expose its counter in `StrictCounters` (e.g. `DiscoveryRoomsChanged`,
checked by `CheckFaults` exactly like `DiscoveryBossChanged`).

**Why.** `ActModel.ApplyDiscoveryOrderModifications` (Models/ActModel.cs:332-343) changes the boss **and** calls
`ApplyActDiscoveryOrderModifications`, whose Overgrowth override reorders 7 normal encounters, 2 events and 2
elites when `NumberOfRuns == 0` (Models.Acts/Overgrowth.cs:110-127). The Core check compares only the boss
(`FidelityGatePatches.DiscoveryPrefix/Postfix`), so a profile with `NumberOfRuns == 0` would change the run
without tripping `ProfileGateFired`. DECISIONS-H1 D.5.

**Local shim.** This pack's `fidelity-discovery-order-rooms-check` (prefix + postfix on the same method):
fingerprint of events / normal / elite encounters in order, boss, second boss, ancient; a change fails the
episode as `ProfileGateFired/fidelity_assert:discovery_order_rooms`. Measured: 6,000 + 1,200 + 180 checks, 0 changes.
After the fold the lead removes the pack's patch (one registry id).

---

## R3. `ForceDiscoveryOrderModifications` should equal the real predicate

**What.** In `RunDriver.StartRun` (SpireHost.Driver/RunDriver.cs:333) set
`rm.ForceDiscoveryOrderModifications = run.GameMode == GameMode.Standard && !run.IsCustomGame;` instead of `true`.

**Why.** `ShouldApplyTutorialModifications` (Runs/RunManager.cs:644-663) returns true for a forced flag before it
checks the mode, whereas the real branch requires Standard and not custom. Today every host run is Standard and
not custom (RunDriver.cs:332), so both are equal; the literal `true` would silently be wrong for Daily / custom
runs later. testmode-audit-core.md P3. Behaviour-neutral now.

**Local shim.** `FidelityInvariants.OnRunSetUp` asserts Standard && !custom && flag set
(`fidelity_assert:standard_run`, `fidelity_assert:discovery_order_real_branch`).

---

## R4. Integration notes on two Core scheduling patches (no contract change)

* `virtual-clock-executor-unpause-poll` (Core) is superseded by `fidelity-executor-unpause-signal` +
  `fidelity-executor-unpause-wake` (DECISIONS-H1 A.5: TCS woken by `Unpause`). The pack's transpiler accepts
  either the original `Task.Delay(int)` or Core's `SchedulingPatches.Delay(int)` at that site (exactly 1), so the
  Core patch can stay or be removed at integration; the run is identical either way (0 executor waits in 2,400
  fuzz runs).
* `virtual-clock-creature-spawn-timeout` (Core) is pre-empted by the assertion `fidelity-assert-creature-spawn-wait`
  in the run driver; the Core patch stays for the H0 combat harness.

## R5. `HostOptions.NativeAllowList` - reviewed, no change requested

The two entries (`godotsharp_string_name_new_from_string`, `godotsharp_string_new_with_utf16_chars`) remain
sufficient. Evidence: static scan (b) + native probes (`SpireHost.Fidelity --scan`): `Callable.From(Action)`
as used by `GameAction.Cancel` reaches no native (the deferred call is the host stub); the only reachable
value-returning natives outside the list are `StringExtensions.Capitalize` (achievement ids; the only reader,
`AchievementsHelper.CheckForDefeatedAllEnemiesAchievement`, has an empty body in v0.113) and
`NodePath`/`ToSnakeCase` (UI only). Natural and forced-HP fuzz with the pack: `unexpected_native_calls` = {}.
