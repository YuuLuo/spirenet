using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using SpireHost.Patches;

namespace SpireHost.Fidelity;

/// <summary>
/// Every non-presentation site of the three TestMode audits (notes\testmode-audit-*.json) ->
/// handled by (patch id of any module | module without a patch | invariant | proven no-op with reason). The table is checked by
/// --selftest / --site-table: every non-presentation JSON site must have a row (or fall under the documented
/// default rule for UI screens that are never instantiated), and every patch id named here must be installed.
/// </summary>
public static class SiteTable
{
    public enum Kind
    {
        Patch,      // a PatchRegistry id of the production set (any module)
        Invariant,  // asserted every run by FidelityInvariants (fidelity_assert:<id>)
        Team,       // handled by a module without a registry id (a handler / frame), or by a tool
        NoOp,       // proven no-op / unreachable (reason)
    }

    public sealed record Entry(string Site, Kind Kind, string Target, string Reason);

    public sealed class Row
    {
        public string Site = "";
        public string Method = "";
        public string Classification = "";
        public string Audit = "";
        public string HandledBy = "";
        public string Target = "";
        public string Reason = "";
    }

    public sealed class Result
    {
        public bool Ok;
        public int Total;
        public int NonPresentation;
        public List<string> Problems = new List<string>();
        public List<Row> Rows = new List<Row>();
    }

    private static Entry P(string site, string id, string reason = "") => new Entry(site, Kind.Patch, id, reason);
    private static Entry I(string site, string id, string reason) => new Entry(site, Kind.Invariant, id, reason);
    private static Entry T(string site, string team, string reason) => new Entry(site, Kind.Team, team, reason);
    private static Entry N(string site, string reason) => new Entry(site, Kind.NoOp, "", reason);

    public static readonly Entry[] Entries =
    {
        // ---- Models / Commands (testmode-audit-models-commands.json) ----
        P("CardSelectCmd.cs:670", "events-bundle-gate", "Events module: the gate is forced to the real branch and its screen becomes the BUNDLE frame (events-bundle-screen on the same method)."),
        N("Cmd.cs:23", "Cmd.Wait: NonInteractive returns at once - the same path as the real game with FastMode=Instant (Cmd.cs:21-29); every await around it is linear."),
        N("Cmd.cs:49", "Cmd.CustomScaledWait: FastMode=Instant never waits either (Cmd.cs:47-63)."),
        P("CreatureCmd.cs:382", "fidelity-natural-death-on-ended"),
        P("Slither.cs:22", "fidelity-guard-test-energy-cost-slither"),
        P("SneckoOil.cs:37", "fidelity-guard-test-energy-cost-sneckooil"),
        P("ConfusedPower.cs:25", "fidelity-guard-test-energy-cost-confusedpower"),
        P("RollingBoulderPower.cs:36", "fidelity-rolling-boulder-per-target"),
        P("CallingBell.cs:39", "fidelity-calling-bell-rewards"),
        P("Cauldron.cs:40", "fidelity-cauldron-rewards"),
        P("FakeSneckoEye.cs:44", "fidelity-guard-test-energy-cost-fakesneckoeye"),
        I("LargeCapsule.cs:40", "no_test_character", "Only the test character Deprived takes this branch; asserted absent every run."),
        I("LargeCapsule.cs:49", "no_test_character", "Same as LargeCapsule.cs:40."),
        P("LordsParasol.cs:57", "rooms-lords-parasol-ui-yield", "Rooms module: run the original, UI block -> one yield, LastPurchaseEverythingTask blocking (DECISIONS-H1 E)."),
        I("ScrollBoxes.cs:98", "no_test_character", "Deprived card pool only; asserted absent every run."),
        P("SneckoEye.cs:55", "fidelity-guard-test-energy-cost-sneckoeye"),
        N("AbstractModel.cs:1094", "Only caller MockCardModel.cs:69 (Models.Cards.Mocks); the host never creates mock cards (the scan treats *.Mocks as UI code: 0 gameplay references)."),
        P("ActModel.cs:465", "fidelity-act-selection-discovery-gate", "Core E1, reviewed (report section 4)."),
        N("CardModel.cs:1116", "Only caller MockCardModel.cs:68."),

        // ---- core (testmode-audit-core.json) ----
        I("AutoSlayer.cs:66", "autoslayer_off", "AutoSlayer is never touched; NonInteractiveMode.AutoSlayerCheck() asserted false every run."),
        N("ScenarioRunner.cs:243", "AutoSlay scenario runner: never used by the host."),
        N("ScenarioRunner.cs:245", "AutoSlay scenario runner (exception text): never used."),
        N("CombatManager.cs:1656", "CombatManager.WaitForUnpause: IsPaused is only set by UI overlays (NCapstoneContainer.cs:148-208, NMapScreen.cs:1130/1185/1474); the TestMode branch returns at once."),
        P("CombatState.cs:658", "fidelity-assert-creature-spawn-wait", "Asserted unreachable for every seat (a virtual client is never ahead of the host); the method body never runs (the H0 virtual-clock patch on it was removed: one patch per site)."),
        N("SentryService.cs:70", "SentryService.Initialize is never called: IsEnabled stays false, every CaptureException is a no-op (SentryService.cs:280-302)."),
        N("MerchantEntry.cs:73", "Rejects purchases only when no run is in progress (MerchantEntry.cs:71-75); the host buys only inside a run."),
        P("MerchantPotionEntry.cs:62", "fidelity-merchant-potion-price"),
        I("RestSiteOption.cs:43", "rest_options_inert", "RestSiteOption.generateForTests asserted null every run."),
        I("ActionExecutor.cs:64", "autoslayer_off", "Pause skips only under AutoSlayer; asserted off."),
        N("ActionExecutor.cs:118", "NonInteractive branch awaits the action directly (no frames); same state sequence as the real branch; JustBeforeActionFinishedExecuting only feeds the disabled checksum. Action faults abort the episode (A.7); the GameAction.cs:102 subset is counted by fidelity-monitor-action-finished-bad-state."),
        I("ActionExecutor.cs:207", "autoslayer_off", "AutoSlayerCheck asserted false."),
        P("ActionExecutor.cs:213", "fidelity-executor-unpause-signal", "TCS woken by Unpause (DECISIONS-H1 A.5)."),
        P("GameAction.cs:102", "fidelity-monitor-action-finished-bad-state", "Kept fail-closed and counted."),
        N("PlayCardAction.cs:103", "CancelAction's TestMode early return only when no run is in progress (PlayCardAction.cs:100-117)."),
        N("NetCombatCardDb.cs:41", "Only matters for players without PlayerCombatState (none in a started combat)."),
        I("NonInteractiveMode.cs:15", "test_mode_on", "IsActive = TestMode.IsOn || AutoSlayerCheck(); both asserted every run; every IsActive site is in this table."),
        P("OneTimeInitialization.cs:43", "save-store-in-memory", "Core: ExecuteVeryEarly is not called; InitSettingsDataForTest / InitPrefsDataForTest instead (no gameplay fields)."),
        N("LocManager.cs:246", "Kept TestMode (fail-closed): the real branch only logs a SmartFormat error and returns the raw text. SmartFormat contains a `catch ... when` exception filter that Harmony/MonoMod cannot re-emit ('Incorrect code generation for exception block'), so any patch of it fails at install. A format error therefore aborts the episode with a FormattingException signature (visible in every fuzz summary; 0 in all runs). Text never feeds game state."),
        P("LocManager.cs:321", "localization-managed-tables", "Core: SetLanguage is never called (managed tables)."),
        P("Logger.cs:33", "godot-os-cmdline-args", "Core: the Logger..cctor natives are stubbed (godot-os-cmdline-args, godot-os-has-feature)."),
        P("ModManager.cs:225", "mod-manager-skipped", "Core."),
        N("ModManager.cs:297", "ResetForTests: never called."),
        N("ModManager.cs:895", "Mod loading: no mods."),
        N("ModManager.cs:912", "Mod loading: no mods."),
        N("ModManager.cs:917", "Mod loading: no mods."),
        N("ModManager.cs:938", "Mod loading: no mods."),
        N("ModManager.cs:968", "Mod loading: no mods."),
        N("JoinFlow.cs:43", "No client join in the host."),
        N("NullLeaderboardStrategy.cs:178", "Daily leaderboards only."),
        N("NullLeaderboardStrategy.cs:197", "Daily leaderboards only."),
        P("RewardsSet.cs:142", "rooms-rewards-test-selector", "Rooms module: RewardsSet.testSelector frame + UI-only proceed rule (NRewardsScreen.cs:496)."),
        P("MerchantRoom.cs:70", "fidelity-merchant-room-exit"),
        // (verifier finding F6: these two were labelled "team replay", but no production code handles them - they are no-ops)
        N("RunManager.cs:385", "ChecksumTracker off in TestMode: the tracker only reads state (NetFullCombatState.FromRun) and counts ids, so its absence has no effect on single-player results. Only the " +
                               "real-client comparison tools re-enable it, for the run they replay (tool-only replay-checksum-tracker-enabled); there the host's states equal the real client's recorded ones."),
        N("RunManager.cs:402", "CombatReplayWriter off in TestMode: it only records what happened (events and checksums of a fight); nothing reads it during a run. No production handling; no effect on results."),
        P("RunManager.cs:650", "fidelity-force-discovery-order", "Core E2, reviewed: set to the value of the real predicate; + invariant standard_run; the boss and the whole room set are fingerprinted by fidelity-discovery-order-noop-check."),
        P("RunManager.cs:883", "fidelity-tutorial-room-type-gate", "Core E3, reviewed; counter fidelity-tutorial-room-type-noop-check."),
        P("RunManager.cs:1592", "fidelity-map-history-player-stats"),
        N("RunManager.cs:1665", "ReturnToMainMenuWithError: multiplayer disconnect only."),
        N("RunRngSet.cs:47", "GetMockInstance: no callers (grep and IL scan)."),
        P("SaveManager.cs:499", "save-store-in-memory", "Core: own non-logging ISaveStore."),
        N("SaveManager.cs:602", "IncrementNumReloads: only on loading a save (H4); NumReloads excluded from signatures."),

        // ---- Nodes (testmode-audit-nodes.json): non-presentation ----
        // (verifier finding F6: this is a patch row - the bundle screen is the events module's registry patch, not an unregistered handler)
        P("NCardBundle.cs:97", "events-bundle-screen", "Events module: the bundle choice is the BUNDLE frame that events-bundle-screen puts in place of the screen (same method as events-bundle-gate, CardSelectCmd.cs:670)."),
        P("NAbandonRunConfirmPopup.cs:89", "events-trial-double-down-confirm", "Events module: Trial DoubleDown -> CONFIRM frame, YES flagged Abandon (DECISIONS-H1 B.5/B.6)."),
        T("NCombatRoom.cs:204", "rooms", "Combat-won proxy (NCombatUi.OnCombatWon -> OfferRoomEndRewards / ProceedWithoutRewards), FIDELITY.md D."),
        T("NEventRoom.cs:111", "events", "Event option dispatch / IsProceed / synthesized PROCEED / FakeMerchant node."),
        T("NRestSiteRoom.cs:185", "rooms", "REST frame (U-RS-1)."),
        P("NStandardMerchantRoom.cs:48", "rooms-foul-potion-merchant-usability", "Rooms module: SHOP frame + FoulPotion usability at the merchant (with the potion extension)."),
        T("NTreasureRoom.cs:129", "rooms", "Chest opening / relic award proxy (U-TR-1..4)."),
        N("NChooseARelicSelection.cs:112", "Only caller RelicSelectCmd.FromChooseARelicScreen has no caller in v0.113 (grep + IL scan)."),
        T("NCardRewardSelectionScreen.cs:146", "rooms", "CARD_REWARD nested decision through a per-reward stack selector (DRIVER-API 7.5)."),
        P("NChooseACardSelectionScreen.cs:149", "combat-local-card-selector", "Combat module: CARD_SELECT through UseLocalSelector + the 15 CardSelectCmd.From* captures (12 names)."),
        N("NGameOverScreen.cs:397", "Profile bookkeeping only (GrantNextUnlock, score, SaveProgressFile): the profile is a frozen snapshot per episode (DECISIONS-H1 D.2); the run-level OnEnded is restored by fidelity-natural-death-on-ended."),
        P("NRollingBoulderVfx.cs:118", "fidelity-rolling-boulder-per-target", "Kept TestMode at the VFX factory; fixed on the model side."),
    };

    /// <summary>Default for UI screens / popups / menus that the audit classified unreachable-headless and that are not listed explicitly.</summary>
    private const string UiDefaultReason = "UI screen / popup / menu factory or callback (Nodes namespace): never instantiated or invoked headlessly - the host never opens it and the TestMode factory returns null.";

    public static string DefaultNotesDir() => Path.GetFullPath(Path.Combine(Boot.GameAssemblies.GameDir, "..", "..", "notes"));

    public static Result Build(string notesDir)
    {
        Result r = new Result();
        Dictionary<string, Entry> byKey = new Dictionary<string, Entry>(StringComparer.Ordinal);
        foreach (Entry e in Entries)
        {
            if (!byKey.TryAdd(e.Site, e))
            {
                r.Problems.Add("duplicate table row " + e.Site);
            }
        }
        HashSet<string> registered = new HashSet<string>(PatchRegistry.Records.Select(x => x.Id), StringComparer.Ordinal);
        HashSet<string> used = new HashSet<string>(StringComparer.Ordinal);
        foreach (string file in new[] { "testmode-audit-models-commands.json", "testmode-audit-core.json", "testmode-audit-nodes.json" })
        {
            string path = Path.Combine(notesDir, file);
            if (!File.Exists(path))
            {
                r.Problems.Add("missing audit file " + path);
                continue;
            }
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
            foreach (JsonElement site in doc.RootElement.EnumerateArray())
            {
                r.Total++;
                string cls = site.GetProperty("classification").GetString() ?? "";
                if (cls == "presentation")
                {
                    continue;
                }
                r.NonPresentation++;
                string fileName = (site.GetProperty("file").GetString() ?? "").Replace('\\', '/').Split('/').Last();
                string key = fileName + ":" + site.GetProperty("line").GetInt32();
                Row row = new Row
                {
                    Site = (site.GetProperty("file").GetString() ?? "") + ":" + site.GetProperty("line").GetInt32(),
                    Method = site.GetProperty("method").GetString() ?? "",
                    Classification = cls,
                    Audit = file.Replace("testmode-audit-", "").Replace(".json", ""),
                };
                if (byKey.TryGetValue(key, out Entry? e))
                {
                    used.Add(key);
                    row.HandledBy = e.Kind switch { Kind.Patch => "patch", Kind.Invariant => "invariant", Kind.Team => "team", _ => "no-op" };
                    row.Target = e.Kind == Kind.Invariant ? "fidelity_assert:" + e.Target : e.Target;
                    row.Reason = e.Reason;
                    if (e.Kind == Kind.Patch && !registered.Contains(e.Target))
                    {
                        r.Problems.Add(key + ": patch id '" + e.Target + "' is not installed");
                    }
                }
                else if (cls == "unreachable-headless" && row.Site.Contains(".Nodes", StringComparison.OrdinalIgnoreCase))
                {
                    row.HandledBy = "no-op";
                    row.Reason = UiDefaultReason;
                }
                else
                {
                    r.Problems.Add(key + " (" + cls + "): no row in SiteTable");
                    row.HandledBy = "MISSING";
                }
                r.Rows.Add(row);
            }
        }
        foreach (string key in byKey.Keys)
        {
            if (!used.Contains(key))
            {
                r.Problems.Add("table row " + key + " matches no non-presentation audit site");
            }
        }
        r.Ok = r.Problems.Count == 0;
        return r;
    }
}
