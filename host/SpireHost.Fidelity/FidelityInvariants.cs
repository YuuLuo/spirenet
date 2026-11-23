using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using SpireHost.Driver;
using SpireHost.Fidelity.Patches;
using SpireHost.Patches;

namespace SpireHost.Fidelity;

/// <summary>
/// Per-run invariants that turn "proven no-op / unreachable" audit sites into checked facts, plus the
/// assertion backstop. Read-only: nothing here changes game state.
///
/// As a lifecycle (OnRunSetUp; fail = Aborted(StartFailed, fidelity_assert:&lt;id&gt;)):
///   * TestMode.IsOn and NonInteractiveMode.AutoSlayerCheck() == false (AutoSlayer.cs:66, NonInteractiveMode.cs:8-21);
///   * every static test hook of the game is inert (see <see cref="StaticHooks"/>);
///   * the run is Standard and not custom and the driver set ForceDiscoveryOrderModifications to the value of the
///     real predicate (RunManager.cs:644-663), i.e. true;
///   * no seat plays the test character Deprived (LargeCapsule.cs:40/49, ScrollBoxes.cs:98);
///   * the net service fits the seat count (phase H5): one seat = the game's own NetSingleplayerGameService and no
///     multi-seat machinery; 2-4 seats = the host's HeadlessHostNetService of type Host - never the game's "fake
///     multiplayer" (several players on a Singleplayer service: CombatManager.cs:797, :829-845,
///     TreasureRoomRelicSynchronizer.cs:114 change rules there) - with a run lobby that lists the seats in order, the
///     combat-state sync enabled as RunManager.SetUpNewMultiplayer leaves it, and no run modifier.
///
/// As an episode check (driver slot IEpisodeCheck, fidelity request R1: evaluated by the core at EVERY quiescent
/// point and once more before teardown; fail = Aborted(StrictViolation, ...)):
///   * fidelity_assert_swallowed   an assertion of this pack fired in this episode but game code swallowed the
///                                 exception (e.g. CombatManager.cs:369-371 catches everything);
///   * fidelity_assert:local_selector_installed   CardSelectorStack.LocalSelector is installed (checked once per episode,
///                                 at the first quiescent point: CardSelectCmd's UI branch dereferences null singletons);
///   * the static test hooks stay inert for the whole episode (ruling D3: strict in production). A test scenario
///     that deliberately uses one of the game's test hooks to build its state names the invariant in
///     Scenario.Exempt - explicit, per scenario, test code only;
///   * fidelity_assert:local_player_is_seat_0   once the run is launched, LocalContext.NetId is seat 0's net id (the
///     driver's "local seat"; RunManager.Launch :596). Nothing in the host may leave another value behind (the encoder
///     switches it per seat for display values and restores it).
/// </summary>
public sealed class FidelityInvariants : IRunLifecycle, IEpisodeCheck
{
    private long _assertsAtStart;
    private bool _selectorChecked;

    public static long Episodes;
    public static long RunSetUpChecks;
    public static long QuiescentChecks;
    public static long ExemptionsUsed;

    public string Name => "fidelity:invariants";

    // ---------------------------------------------------------------------------------------- IRunLifecycle

    public void OnRunSetUp(RunContext ctx)
    {
        Episodes++;
        Require(TestMode.IsOn, "test_mode_on", "TestMode must be on (the host's headless switch).");
        Require(!NonInteractiveMode.AutoSlayerCheck(), "autoslayer_off", "NonInteractiveMode.AutoSlayerCheck() must be false (AutoSlayer is never touched).");
        (string id, string message)? hook = StaticHooks(ctx);
        if (hook.HasValue)
        {
            FidelityAssert.Fail(hook.Value.id, AbortReason.StartFailed, hook.Value.message);
        }
        RunState run = ctx.Run;
        Require(run.GameMode == GameMode.Standard && !run.IsCustomGame, "standard_run",
            "The host starts Standard, non-custom runs only (the lobby path it reproduces); anything else needs its own review of RunManager.ShouldApplyTutorialModifications (RunManager.cs:644-663).");
        Require(ctx.RunManager.ForceDiscoveryOrderModifications, "discovery_order_real_branch", "ForceDiscoveryOrderModifications must be true at OnRunSetUp for a Standard, non-custom run (DECISIONS-H1 D.5).");
        foreach (Player p in run.Players)
        {
            Require(p.Character is not MegaCrit.Sts2.Core.Models.Characters.Deprived, "no_test_character", "Seat plays the test character Deprived.");
        }
        RunManager rm = ctx.RunManager;
        if (run.Players.Count == 1)
        {
            Require(ctx.Mp == null && rm.NetService is MegaCrit.Sts2.Core.Multiplayer.NetSingleplayerGameService, "single_seat_service",
                "A single-seat run must be the game's single-player set-up (RunManager.SetUpNewSingleplayer: NetSingleplayerGameService).");
        }
        else
        {
            Require(ctx.Mp != null && ReferenceEquals(rm.NetService, ctx.Mp.Net) && rm.NetService.Type == MegaCrit.Sts2.Core.Multiplayer.Game.NetGameType.Host && !rm.IsSingleplayerOrFakeMultiplayer,
                "multi_seat_host_service", "A multi-seat run must run on the host's own Host-type net service (never several players on a Singleplayer service: the game's fake multiplayer changes rules).");
            bool lobbyOk = rm.RunLobby != null;
            int member = 0;
            if (rm.RunLobby != null)
            {
                foreach (ulong id in rm.RunLobby.PlayerIds)
                {
                    lobbyOk &= member < run.Players.Count && run.Players[member].NetId == id;
                    member++;
                }
            }
            Require(lobbyOk && member == run.Players.Count, "multi_seat_lobby", "The run lobby must list every seat's net id in seat order (CombatStateSynchronizer.CheckSyncCompleted waits for exactly these).");
            Require(!rm.CombatStateSynchronizer.IsDisabled, "multi_seat_combat_state_sync", "The combat-state sync must be enabled in a multi-seat run, as RunManager.SetUpNewMultiplayer leaves it (lead ruling Q2).");
            Require(run.Modifiers.Count == 0, "multi_seat_no_modifiers", "A multi-seat run of the host carries no run modifier (the treasure-room extra reward sets rely on it).");
        }
        RunSetUpChecks++;
    }

    public void OnRoomEntered(RunContext ctx)
    {
    }

    public void OnEpisodeEnd(RunContext ctx)
    {
    }

    // ---------------------------------------------------------------------------------------- IEpisodeCheck

    /// <summary>
    /// The baseline of the swallowed-assertion backstop is taken before the run is created (verifier finding R-8): an
    /// assertion that fires inside game code during the run set-up and is swallowed there fails the episode at its first
    /// quiescent point, like one that fires later.
    /// </summary>
    public void OnEpisodeBegin()
    {
        _assertsAtStart = FidelityAssert.Fired;
    }

    public void OnEpisodeStart(RunContext ctx)
    {
        _selectorChecked = false;
    }

    public CheckFailure? Check(RunContext ctx, bool final)
    {
        QuiescentChecks++;
        if (FidelityAssert.Fired != _assertsAtStart)
        {
            _assertsAtStart = FidelityAssert.Fired;
            return new CheckFailure(AbortReason.StrictViolation, "fidelity_assert_swallowed",
                "A fidelity assertion fired and the exception was swallowed by game code: " + FidelityAssert.Last);
        }
        if (!_selectorChecked)
        {
            _selectorChecked = true;
            // CardSelectCmd's UI branch (CardSelectCmd.cs:139-147 and the other From* entries) dereferences
            // NOverlayStack / NCombatRoom singletons; it is only reached when no local selector is installed.
            if (ctx.RunManager.CardSelectorStack.LocalSelector == null)
            {
                FidelityAssert.Count("local_selector_installed");
                return new CheckFailure(AbortReason.StrictViolation, "fidelity_assert:local_selector_installed",
                    "No CardSelectorStack.LocalSelector is installed (DECISIONS-H1 B.2): every CardSelectCmd.From* prompt would take the UI branch and dereference a null UI singleton.");
            }
        }
        (string id, string message)? hook = StaticHooks(ctx);
        if (hook.HasValue)
        {
            FidelityAssert.Count(hook.Value.id);
            return new CheckFailure(AbortReason.StrictViolation, "fidelity_assert:" + hook.Value.id, hook.Value.message);
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------- shared

    /// <summary>
    /// The static test hooks of the game that must be inert: TestRngInjector (relic / rarity / shuffle / card-generation
    /// overrides, consumed by RelicFactory.cs:32-51, CardPile.cs:71, CardFactory.cs:94), TestFlags
    /// (GameActionPlayerChoiceContext.cs:65, HookPlayerChoiceContext.cs:166/178), RestSiteOption.generateForTests
    /// (RestSiteOption.cs:43), CardSelectorStack._testSelector (CardSelectorStack.cs:115-136; DECISIONS-H1 B.2).
    /// Returns the first violated invariant that the running scenario (if any) does not explicitly exempt.
    /// </summary>
    private static (string id, string message)? StaticHooks(RunContext ctx)
    {
        Scenario? scenario = ctx.Spec.Scenario;
        (string id, string message)? Violation(bool ok, string id, string message)
        {
            if (ok)
            {
                return null;
            }
            if (scenario != null && scenario.IsExempt(id))
            {
                ExemptionsUsed++;
                return null;
            }
            return (id, message);
        }

        // Verifier finding F7: the three run-level switches below used to be looked at when the run was set up only. They are part of
        // what is evaluated at every quiescent point now (the site table's "invariant" rows rely on them for the whole episode).
        bool testCharacter = false;
        foreach (Player p in ctx.Run.Players)
        {
            testCharacter |= p.Character is MegaCrit.Sts2.Core.Models.Characters.Deprived;
        }
        ulong? localId = MegaCrit.Sts2.Core.Context.LocalContext.NetId;
        (string id, string message)? switches =
            Violation(!localId.HasValue || localId.Value == ctx.Run.Players[0].NetId, "local_player_is_seat_0", "LocalContext.NetId is " + localId + ", not seat 0's net id " + ctx.Run.Players[0].NetId + ".")
            ?? Violation(TestMode.IsOn, "test_mode_on", "TestMode must be on (the host's headless switch).")
            ?? Violation(!NonInteractiveMode.AutoSlayerCheck(), "autoslayer_off", "NonInteractiveMode.AutoSlayerCheck() must be false (AutoSlayer is never touched).")
            ?? Violation(!testCharacter, "no_test_character", "A seat plays the test character Deprived.");
        if (switches.HasValue)
        {
            return switches;
        }
        return Violation(TestRngInjector._relicOverrides.Count == 0 && TestRngInjector._relicRarityOverride == null
                         && TestRngInjector._initialShuffleOverride == null && TestRngInjector._combatCardGenerationOverride == null,
                   "test_rng_injector_inert", "TestRngInjector holds an override: relic / shuffle / card-generation results would not come from the game's RNG.")
               ?? Violation(!TestFlags.ShouldSendResumeForRemotePlayers, "test_flags_inert", "TestFlags.ShouldSendResumeForRemotePlayers must stay false.")
               ?? Violation(RestSiteOption.generateForTests == null, "rest_options_inert", "RestSiteOption.generateForTests must be null (RestSiteOption.cs:43).")
               ?? Violation(CardSelectorStack._testSelector == null, "global_test_selector_unused", "CardSelectorStack._testSelector must not be set (DECISIONS-H1 B.2).");
    }

    private static void Require(bool ok, string id, string message)
    {
        if (!ok)
        {
            FidelityAssert.Fail(id, AbortReason.StartFailed, message);
        }
    }
}

/// <summary>The counters of the fidelity module ("fidelity.*"): what each restored branch / assertion / monitor did.</summary>
public sealed class FidelityCounters : IModuleCounters
{
    public string Module => "fidelity";

    public void Read(IDictionary<string, long> c)
    {
        c["boulder_calls"] = FidelityPatches.BoulderCalls;
        c["boulder_hits"] = FidelityPatches.BoulderHits;
        c["boulder_skipped_removed"] = FidelityPatches.BoulderSkippedRemoved;
        c["natural_loss_on_ended"] = FidelityPatches.NaturalLossOnEnded;
        c["unpause_waits"] = FidelityPatches.UnpauseWaits;
        c["unpause_wakeups"] = FidelityPatches.UnpauseWakeups;
        c["previous_turn_loop_waits"] = FidelityPatches.PreviousTurnLoopWaits;
        c["previous_turn_loop_timeouts"] = FidelityPatches.PreviousTurnLoopTimeouts;
        c["action_finished_bad_state"] = FidelityPatches.ActionFinishedBadState;
        c["mend_local_targetings"] = FidelityPatches.MendLocalTargetings;
        c["asserts_fired"] = FidelityAssert.Fired;
        foreach (KeyValuePair<string, long> kv in FidelityAssert.ById)
        {
            c["assert." + kv.Key] = kv.Value;
        }
        c["run_setup_checks"] = FidelityInvariants.RunSetUpChecks;
        c["quiescent_checks"] = FidelityInvariants.QuiescentChecks;
        c["scenario_exemptions_used"] = FidelityInvariants.ExemptionsUsed;
        // the Core's profile gates (veteran profile: all 0 except the call counter)
        c["core_tutorial_room_forced"] = FidelityGatePatches.TutorialRoomForced;
        c["core_discovery_boss_changed"] = FidelityGatePatches.DiscoveryBossChanged;
        c["core_discovery_rooms_changed"] = FidelityGatePatches.DiscoveryRoomsChanged;
    }
}
