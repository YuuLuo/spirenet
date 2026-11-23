using System.Collections.Generic;
using SpireHost.Patches;

namespace SpireHost.Replay.V2;

/// <summary>
/// The NAMED rules of the v2 comparison: every place where the recorded UI state and the host's decision are allowed
/// to differ. Each rule has a kind - "representation" (the real UI shows an item whose click changes nothing, or the
/// host folds several UI items / steps into one action) or "oracle" (a documented limit of the recording) - and the
/// source that proves it (game UI source under ref\sts2-v0113-src, the mod source under realgame\mod-src, or the
/// validation report notes\oracle-deep-validation.md). A rule is applied only where its exact condition holds, and
/// every application is counted in the per-trace "Explained" table and in the aggregate. There is no other tolerance.
/// </summary>
public static class V2Classes
{
    public const string Representation = "representation";
    public const string Oracle = "oracle";

    // ---- events ----
    public const string EventLockedOption = "event.locked_option_button_enabled_but_inert";

    // ---- rewards ----
    public const string RewardPotionFullBelt = "rewards.potion_reward_with_full_belt_click_fails";
    public const string RewardSkipFullBeltRecord = "rewards.skip_full_belt_record_is_not_an_action";

    // ---- rest ----
    public const string RestStaleButton = "rest.stale_button_of_a_consumed_option";
    public const string RestButtonFadingIn = "rest.remaining_option_recorded_during_its_fade_in";

    // ---- shop ----
    public const string ShopUnaffordableSlot = "shop.unaffordable_slot_click_fails";
    public const string ShopPotionFullBelt = "shop.potion_slot_with_full_belt_click_fails";
    public const string ShopPotionForbidden = "shop.potion_slot_forbidden_by_relic_click_fails";
    public const string ShopUiNavigation = "shop.ui_navigation_record_is_not_a_decision";
    public const string ShopRecordedClickWasNoop = "shop.recorded_click_was_refused_by_the_game";

    // ---- selections ----
    public const string SelectionMergedClass = "selection.identical_pile_cards_are_one_host_option";
    public const string SelectionNotFirstInstance = "selection.recorded_pick_is_not_the_first_instance_of_its_class";
    public const string SelectionPreviewConfirm = "selection.preview_confirmation_is_not_a_host_step";
    public const string SelectionCancelBehindPreview = "selection.close_button_disabled_while_a_preview_is_open";
    public const string SelectionEnchantZeroConfirmInert = "selection.enchant_confirm_with_no_card_is_inert";
    public const string SelectionGridWindow = "selection.grid_showed_a_35_card_window";
    public const string SelectionPileGridWindow = "selection.pile_grid_recorded_as_its_35_card_window";
    public const string SelectionHandScreenOrder = "selection.hand_holders_recorded_in_screen_order";
    public const string SelectionStrayResult = "selection.stray_selection_result_record";
    public const string SelectionInvisibleButton = "selection.enabled_flag_of_an_invisible_button";
    public const string SelectionResultMissing = "selection.result_record_written_elsewhere";

    // ---- bundle / crystal ----
    public const string BundlePreviewConfirm = "bundle.preview_confirmation_is_not_a_host_step";
    public const string CrystalToolFoldedIntoClick = "crystal.tool_button_is_folded_into_the_click";
    public const string CrystalProceedIsEventProceed = "crystal.screen_proceed_is_the_finished_event_proceed";

    // ---- combat ----
    public const string CombatPotionLockWindow = "combat.potions_locked_while_the_room_fades_in";
    public const string CombatNoMcrIdentity = "combat.fight_without_a_replay_file_identified_from_combat_json";
    public const string CombatStallEnd = "combat.trace_ended_by_a_client_stall";
    public const string CombatNoMcrIdentityEnd = "combat.fight_without_a_replay_file_not_comparable_after_an_unidentifiable_choice";
    public const string CombatMaxTurnsEnd = "combat.trace_ended_at_the_150_turn_cap";

    // ---- dumps / saves ----
    public const string DumpCombatEntrySampling = "dump.combat_room_sampled_before_turn_1_start_hooks";
    public const string DumpWrittenAfterFight = "dump.room_dump_written_after_the_rooms_fight";
    public const string SaveDiscovered = "save.players_discovered_lists_depend_on_the_profile_recipe";
    public const string SaveEntrantNumber = "save.trial_entrant_number_is_drawn_from_rng_chaotic";

    private static readonly Dictionary<string, (string Kind, string Rule)> _rules = new Dictionary<string, (string, string)>
    {
        [EventLockedOption] = (Representation,
            "A locked option's button is enabled and clickable at node level; NEventOptionButton.OnRelease returns at once when Option.IsLocked " +
            "(Nodes.Events/NEventOptionButton.cs:260-264). The host offers no action for it (EventRoomHandler: IsLocked)."),
        [RewardPotionFullBelt] = (Representation,
            "NRewardButton stays enabled for a PotionReward while every belt slot is filled; the click runs RewardsSetSynchronizer.SelectLocalReward -> PotionReward.OnSelect -> " +
            "PotionCmd.TryToProcure -> TooFull -> false and the button re-enables (Nodes.Rewards/NRewardButton.cs:184-215, Rewards/PotionReward.cs:76-93). Nothing changes, so the host does not offer it (RewardFrame.IsTakeEffective)."),
        [RewardSkipFullBeltRecord] = (Oracle,
            "The collector writes a rewards record with action skip_full_belt when its policy passes over a potion reward with a full belt; it clicks nothing (realgame\\mod-src\\SpireOracle\\RunTracer2.cs:797-814). No host action corresponds to it."),
        [RestStaleButton] = (Oracle,
            "NRestSiteRoom.UpdateRestSiteOptions frees the old buttons with QueueFree and creates new ones (Nodes.Rooms/NRestSiteRoom.cs:330-345); the collector still sees the freed, disabled buttons of options that left " +
            "RestSiteSynchronizer's list and records them with index -1 (validation report P5). They are not offers."),
        [RestButtonFadingIn] = (Oracle,
            "After a successful choice the remaining options (MiniatureTent) get NEW buttons whose MouseFilter becomes Stop only when their 0.5 s fade-in tween ends (Nodes.RestSite/NRestSiteButton.cs:127-135, " +
            "Nodes.Rooms/NRestSiteRoom.cs:441-463); the collector records after 8 unchanged frames (RunTracer2.cs:845), i.e. inside that tween, so it saw the option enabled but not yet clickable and could only proceed. " +
            "The host offers the option (RestSiteOption.IsEnabled), which is what the button becomes."),
        [ShopUnaffordableSlot] = (Representation,
            "Every stocked merchant slot is visible and clickable; NMerchantSlot.OnSelected -> MerchantSynchronizer.PurchaseLocalEntry -> MerchantEntry.OnTryPurchaseUnsynchronized returns FailureGold without changing anything when " +
            "!EnoughGold (Entities.Merchant/MerchantEntry.cs:71-86). The host offers only purchases that happen."),
        [ShopPotionFullBelt] = (Representation,
            "MerchantPotionEntry's purchase fails with FailureSpace when the belt is full, without spending gold (Entities.Merchant/MerchantPotionEntry.cs:68-80). The host does not offer it (ShopHandler.IsPurchasable)."),
        [ShopPotionForbidden] = (Representation,
            "MerchantPotionEntry's purchase fails with FailureForbidden when Hook.ShouldProcurePotion is false (Sozu), without spending gold (Entities.Merchant/MerchantPotionEntry.cs:68-80). The host does not offer it; " +
            "the collector's own 'purchasable' filter does not know the hook (oracle-deep.md 6.5: 1 refused purchase)."),
        [ShopUiNavigation] = (Representation,
            "Opening / closing the merchant inventory and the room's proceed button are UI navigation (Nodes.Rooms/NMerchantRoom.cs:179-185, Nodes.Screens.Shops/NMerchantInventory): the host's SHOP frame is the open inventory and its PROCEED is " +
            "'back, then proceed'. The collector's shop_ui records map to no host decision; their button states are checked against the frame."),
        [ShopRecordedClickWasNoop] = (Oracle,
            "The collector clicked a slot its own filter considered purchasable and the game refused the purchase (no gold spent, nothing gained: the next record shows the same gold and belt). The host offers no such action; the record is consumed without one."),
        [SelectionMergedClass] = (Representation,
            "The real grid shows every instance of identical cards as its own holder; the host offers one SELECT_CARD per class of identical cards (DECISIONS-H1 B.4, DRIVER-API 6.4: an instance-level ordinal of the draw pile would " +
            "leak its order, CardSelectCmd.cs:273-291). Each UI holder corresponds to the option of its class; the class's remaining count is its number of unpicked holders."),
        [SelectionNotFirstInstance] = (Representation,
            "The recorded click was on a holder that is not the first unpicked instance of its class in the order the game listed the cards. The host's action hands out instances in that order, so a policy cannot name this instance; " +
            "the tool hands the recorded instance to the game through the host-only view (ICardSelectHostView.PreferInstance) so that the run continues from the recorded state."),
        [SelectionPreviewConfirm] = (Representation,
            "Deck screens open a preview with Confirm / Cancel when the selection reaches its size (Nodes.Screens.CardSelection: NDeckCardSelectScreen.OnCardClicked -> PreviewSelection :170-225, confirmed by ConfirmSelection :253-265; " +
            "NDeckUpgradeSelectScreen.OnCardClicked :200-248, ConfirmSelection :294-311; NDeckTransformSelectScreen.OnCardClicked -> OpenPreviewScreen :189-205, :251-270; NDeckEnchantSelectScreen.OnCardClicked -> PreviewSelection :220-248, :295-318, " +
            "ConfirmSelection :322-336) and the hand shows its confirm button (Nodes.Combat/NPlayerHand.RefreshSelectModeConfirmButton :1011-1021); for a prompt without manual confirmation the host completes with the pick that reaches the size. " +
            "The reachable answers are the same; the UI's un-pick / preview-cancel has no host action."),
        [SelectionCancelBehindPreview] = (Representation,
            "While a preview is open the screen's Close button is disabled (NDeckCardSelectScreen.cs:199, NDeckUpgradeSelectScreen.cs:215 / :239, NDeckTransformSelectScreen.cs:264, NDeckEnchantSelectScreen.cs:299); the preview's Cancel returns to the grid, " +
            "where Close is enabled again for a cancelable prompt (NDeckCardSelectScreen.CancelSelection :233-251). The host's CANCEL stays offered on a cancelable prompt after a pick. Same reachable answer (empty), two UI clicks."),
        [SelectionEnchantZeroConfirmInert] = (Representation,
            "NDeckEnchantSelectScreen shows its Confirm button enabled with no card selected when MinSelect is 0, but ConfirmSelection ignores an empty selection (Nodes.Screens.CardSelection/NDeckEnchantSelectScreen.cs:240, 324, 334): " +
            "the click does nothing. The host offers no SKIP there (SelectionRules: DeckEnchant never completes with zero cards)."),
        [SelectionGridWindow] = (Oracle,
            "NCardGrid is virtualised: CurrentlyDisplayedCardHolders is a sliding window of 35 holders (Nodes.Cards/NCardGrid.cs:310, 769-800, 884-960). The record's 'displayed' is that window; 'candidates' is the screen's complete list and is what is compared (validation report P3)."),
        [SelectionPileGridWindow] = (Oracle,
            "NCombatPileCardSelectScreen keeps the pile, not a card list (its _cards is empty), so the collector records the grid's holders as the candidates (Screens2.AnswerGrid, candidate_order ui_display); the grid is virtualised and holds a " +
            "window of 35 holders (Nodes.Cards/NCardGrid.cs:310, 769-800, 884-960). For a pile of more than 35 cards the record lists 35 of them. Applied only when every recorded card is one of the cards the game handed to the host's selector; " +
            "the recorded picks are inside the window by construction."),
        [SelectionHandScreenOrder] = (Representation,
            "The collector lists NPlayerHand.ActiveHolders, i.e. the hand as it is laid out on screen. A card that a hand prompt selected and that stays in the hand gets a new holder (NPlayerHand.SelectCardInSimpleMode / SelectCardInUpgradeMode remove " +
            "the holder, :921-946; the card comes back through Add at HandLayoutHelper's insert index, :476-483, :955), so the screen order can differ from the order of the hand pile, which is what the game hands to the host's selector " +
            "(CardSelectCmd.FromHand filters PileType.Hand in pile order) and what the host uses as the public hand ordinal. Same cards; applied only when the two lists are equal as multisets."),
        [SelectionStrayResult] = (Oracle,
            "A hand selection that finishes as the fight ends writes its selection_result into decisions.jsonl instead of the fight's prompt records (validation report P2: Screens2.Emit decides by CombatManager.IsInProgress at emit time)."),
        [SelectionInvisibleButton] = (Oracle,
            "The enabled flag of a button that is not visible depends on animation timing (validation report section 5 / recommendation 1). Only visible buttons are compared."),
        [SelectionResultMissing] = (Oracle,
            "The selection record has no selection_result (it was written into the other log, validation report P2); the per-pick button states of this prompt are not compared."),
        [BundlePreviewConfirm] = (Representation,
            "NChooseABundleSelectionScreen opens a preview with confirm / cancel after a bundle is clicked (Nodes.Screens.CardSelection/NChooseABundleSelectionScreen.OpenPreviewScreen :238-248, CancelSelection :250); the host's CHOOSE_OPTION(bundle) is 'click, then confirm'."),
        [CrystalToolFoldedIntoClick] = (Representation,
            "The two divination buttons select the tool (NCrystalSphereScreen.SetBigDivination / SetSmallDivination -> CrystalSphereMinigame.SetTool, :234-246); the host folds the tool into CRYSTAL_CLICK(cell, tool). A tool is offered when it is the active one or its button is clickable."),
        [CrystalProceedIsEventProceed] = (Representation,
            "After the last divination the screen shows its Proceed button; the event is finished behind it. The host's decision at that point is the finished event's PROCEED (EVENT frame)."),
        [CombatPotionLockWindow] = (Oracle,
            "RunManager.IsPotionUseLocked = IsChangingRooms || CombatManager.IsStarting; the real client clears IsChangingRooms only after the room fade-in (Runs/RunManager.cs:222-231, 908-922), which TestMode skips. " +
            "The collector's option count therefore lacks every potion option for a leading window of a fight (host-h1-replay.md 3.2). Applied only when recorded == host count minus ALL potion actions."),
        [CombatNoMcrIdentity] = (Oracle,
            "The client writes a fight's replay when the fight is won (or lost, on leaving the run); a fight that ended at the collector's 150-turn cap or stalled has none. Its recorded actions name cards by id only: " +
            "identical cards are taken in hand / list order."),
        [CombatNoMcrIdentityEnd] = (Oracle,
            "In a fight without a replay file (it ends the trace at the collector's turn cap or stall) combat.json names a played card by id only. When several identical cards could be meant and the next recorded signature does not tell which " +
            "(or a prompt answer was among identical cards), the host takes the first; the real client may have taken another, after which hidden per-instance state (pile positions) differs. If a later recorded signature of that same fight then " +
            "differs, the trace's comparison ends there: everything before it matched, nothing after it is comparable. Applied only in the last fight of a trace that ended early, after at least one such unidentifiable choice."),
        [CombatStallEnd] = (Oracle,
            "REGENT-A0-DEP200000016: the headless real client hung in floor 13 after 87 player turns (side=Player phase=Start, executor idle); a re-run in another process continued (oracle-deep.md 5.3, validation report section 2). " +
            "The trace is compared up to its last recorded action."),
        [CombatMaxTurnsEnd] = (Oracle,
            "The collector ends a fight at 150 player turns (oracle-deep.md 5.3); the trace's end dump was taken at that decision point. The host is compared up to and including that dump."),
        [DumpCombatEntrySampling] = (Oracle,
            "The tracer dumps a combat room at the first frame after RoomEntered at which IsChangingRooms is false - a frame boundary inside the turn-1 start sequence of CombatManager.StartCombatInternal / StartTurn (Combat/CombatManager.cs:374-411, " +
            "470-560), after CombatBegan and somewhere between two of that sequence's commands (host-h1-replay.md 4.3: before the late turn-start hooks; here also: after k of the n cards of the first hand were drawn). The host has no frames. It samples its " +
            "own dump and save at CombatManager.CombatBegan, after every entry the game adds to CombatHistory (public event Changed: card drawn, power received, damage received, block gained, card generated ...) and at its first decision point; the " +
            "trace's dump and save - one snapshot of one moment - must equal, as a whole, the host at the decision point or ONE of those sampled states (never field by field from different states). The first recorded combat action's signature " +
            "(HP, hand, potions), every recorded per-action combat state of the fight and the next room's dump are compared exactly afterwards."),
        [DumpWrittenAfterFight] = (Oracle,
            "RunTracer2's loop dumps a room when RoomEntered was raised and IsChangingRooms is false, and plays a fight when CombatManager.IsInProgress (RunTracer2.cs:185-200, in that order). For an event's fight the room change can still be in " +
            "progress when the combat already is: the tracer then plays the whole fight first and writes the room's dump afterwards. It shows in the files themselves: the fight's combat.json / .mcr carry the same index as that room's dump " +
            "(normally the dump's index + 1). Applied only then; the dump is compared with the host when its fight is over and the combat room is still the current room, i.e. at CombatManager.CombatEnded " +
            "(1 fight in the corpus: DEFECT-A0-DEP200000009, BATTLEWORN_DUMMY)."),
        [SaveDiscovered] = (Oracle,
            "players[].discovered_* list what the run newly discovered relative to the PROFILE (Entities.Players/Player.cs:142-211); the trace profile marks nothing as seen, the host's veteran profile everything. The real client does not " +
            "reproduce discovered_cards itself (validation report section 5: UI render timing). Read only by UI and the epoch merge."),
        [SaveEntrantNumber] = (Oracle,
            "Trial.CalculateVars draws EntrantNumber from Rng.Chaotic.NextInt(101, 999) (Models.Events/Trial.cs:207-212): the real client does not reproduce it between two runs of the same seed (validation report P1). Display text only."),
    };

    public static string KindOf(string cls) => _rules.TryGetValue(Base(cls), out (string Kind, string Rule) r) ? r.Kind : "?";

    public static string RuleOf(string cls) => _rules.TryGetValue(Base(cls), out (string Kind, string Rule) r) ? r.Rule : "(no rule text)";

    public static bool IsKnown(string cls) => _rules.ContainsKey(Base(cls));

    /// <summary>A class may carry a ":detail" suffix (e.g. the screen type); the rule is that of the part before it.</summary>
    private static string Base(string cls)
    {
        int c = cls.IndexOf(':');
        return c > 0 ? cls.Substring(0, c) : cls;
    }
}

/// <summary>
/// The one read-only instrumentation of the run-trace comparison tools: subscriptions to the game's public events
/// CombatManager.CombatBegan and CombatHistory.Changed, at which the tool takes its own copy of the tracer's dump and of
/// RunManager.ToSave (reads only). Registered in PatchRegistry as tool-only instrumentation (ledger section 2); never part
/// of the production set.
/// </summary>
public static class V2Sampling
{
    public const string Id = "replay-combat-start-samples";
    private static bool _registered;

    public static void Register()
    {
        if (_registered)
        {
            return;
        }
        _registered = true;
        PatchRegistry.Note(Id, "MegaCrit.Sts2.Core.Combat.CombatManager::CombatBegan / CombatEnded, MegaCrit.Sts2.Core.Combat.History.CombatHistory::Changed (public events)", "event subscription (run-trace comparison tools only)", PatchCategory.Instrumentation,
            "The real client's tracer dumps a combat room at a frame boundary inside the turn-1 start sequence (after CombatBegan, CombatManager.cs:395; between two commands of StartTurn); the host's first decision point is after that sequence. " +
            "The run-trace comparison tools (--compare-run-traces, --compare-run-traces-v2) subscribe to these public events for the duration of an episode and build the tracer's dump and RunManager.ToSave at CombatBegan and after every " +
            "CombatHistory entry until the first decision point (reads only), so that a combat-room dump and its save are compared as a whole with each of the host's states of that sequence (they must equal one of them) instead of with a tolerance; " +
            "CombatEnded serves the one room dump of the v2 corpus that the tracer wrote after its room's fight. Not in the production set.");
    }
}
