using System.Collections.Generic;

namespace SpireHost.Driver.Rooms;

/// <summary>
/// Process-wide, read-only-for-the-game counters of the rooms sub-drivers: which branch of the terminal proceed
/// tree ran, how deep reward frames nested, what the UI-only rules filtered, how often the fidelity patches fired.
/// They never influence a decision; they exist for the self-tests and the reports (DRIVER-API.md has no
/// per-module counter slot in EpisodeCounters yet: see contract-requests\rooms.md, request R3).
/// Engine thread only (no locking needed: one engine thread per process).
/// </summary>
public static class RoomsProbe
{
    // ---- combat won / terminal proceed tree (NCombatUi.OnCombatWon, NRewardsScreen.OnProceedButtonPressed) ----
    public static long CombatsWon;
    public static long CombatWonWithRewards;
    public static long CombatWonWithoutRewards;
    public static long TreeMap;                 // terminal set proceed -> ProceedFromTerminalRewardsScreenAsync -> Proceed() -> map
    public static long TreeResumeEvent;         // terminal set proceed -> ResumePreviousRoom (parent event)
    public static long TreeFirstOfTwoBosses;    // A10 last act, first boss -> map with only the second boss travelable
    public static long TreeActChange;           // last boss of an act (not the last act) -> SetLocalPlayerReady -> next act
    public static long TreeArchitect;           // last boss of the last act -> SetLocalPlayerReady -> TheArchitect
    public static long TreeDebugOverride;       // RunManager.debugAfterCombatRewardsOverride (never set by the host)
    public static long NoRewardsMap;            // combat without rewards -> Proceed() -> map
    public static long NoRewardsResume;         // combat without rewards -> ResumePreviousRoom (BattlewornDummy)
    public static long TreeChecksPassed;        // post-condition of a branch verified at the next room entry / after the call
    public static long TreeChecksFailed;
    public static string? FirstTreeCheckFailure;

    // ---- reward frames ----
    public static long RewardSetsOffered;
    public static long TerminalSetsOffered;
    public static long RewardFrameMaxDepth;     // reward frames simultaneously on the decision stack
    public static long RewardTakes;
    public static long RewardTakeRejected;      // SelectLocalReward returned false (card reward skip/close, cancel)
    public static long RewardAutoClosed;        // non-terminal set fully taken: the screen closes itself
    public static long NonTerminalSkips;        // SkipLocalRewardsSet on a non-terminal PROCEED
    public static long TerminalLeftForRoomExit; // terminal PROCEED with untaken rewards (skipped later by BeforeLeavingRoom)
    public static long PotionRewardMasked;      // belt full and ShouldProcurePotion: a click would be a no-op
    public static long NoOpRewardMasked;        // card upgrade / removal reward with no eligible card
    public static long ProceedDisallowed;       // DisallowSkipping with untaken rewards
    public static readonly SortedDictionary<string, long> RewardTypesTaken = new SortedDictionary<string, long>(System.StringComparer.Ordinal);

    // ---- card reward (nested) ----
    public static long CardRewardDecisions;
    public static long CardRewardRerollFiltered;  // stale REROLL alternative hidden after a reroll (the UI refreshes its buttons)
    public static long CardRewardEarlyRelease;    // the pushed selector was popped as soon as the selection ended
    public static readonly SortedDictionary<string, long> CardRewardAlternativesOffered = new SortedDictionary<string, long>(System.StringComparer.Ordinal);
    public static readonly SortedDictionary<string, long> CardRewardAlternativesChosen = new SortedDictionary<string, long>(System.StringComparer.Ordinal);
    public static long CardRewardForwardedSelections; // GetSelectedCards forwarded to ctx.CardSelector

    // ---- treasure ----
    public static long ChestsOpened;
    public static long ChestGold;
    public static long ChestsEmpty;               // CompleteWithNoRelics
    public static long TreasurePicks;
    public static long TreasureSkips;
    public static long TreasureRelicsObtained;
    public static long TreasureFallbackMoves;

    // ---- rest ----
    public static long RestChoices;
    public static long RestSuccesses;
    public static long RestFailures;
    public static long RestChoicesAfterSuccess;   // a further option chosen after a success (MiniatureTent)
    public static long RestLeaves;

    // ---- shop ----
    public static long ShopPurchases;
    public static long ShopPurchaseRejectedLegit; // card removal cancelled in its CARD_SELECT
    public static long ShopLeaves;
    public static long ParasolWaits;
    public static long ParasolYields;
    public static readonly SortedDictionary<string, long> ShopEntryTypesBought = new SortedDictionary<string, long>(System.StringComparer.Ordinal);

    // ---- out-of-combat potions ----
    public static long PotionUses;
    public static long PotionDiscards;
    public static long FoulPotionMerchantOverrides;
    public static long FoulPotionMerchantKeptFalse; // MerchantRoom but not the SHOP screen (map opened by LEAVE, prompt open, no episode)
    public static readonly SortedDictionary<string, long> PotionUseOfferedByKind = new SortedDictionary<string, long>(System.StringComparer.Ordinal);
    public static readonly SortedDictionary<string, long> PotionsUsed = new SortedDictionary<string, long>(System.StringComparer.Ordinal);

    // ---- multi-seat (phase H5): the remote seats' screens; all 0 in single-seat runs ----
    public static long RemoteSetsOffered;            // reward screens opened for remote seats
    public static long RemoteTerminalSetsOffered;
    public static long RemoteSetsWithoutScreen;      // empty non-combat sets: RewardsSet.Offer returns without a screen
    public static long RemoteScreenMaxDepth;
    public static long RemoteTakes;
    public static long RemoteTakeRejected;
    public static long RemoteAutoClosed;
    public static long RemoteNonTerminalSkips;
    public static long RemoteTerminalLeftForRoomExit;
    public static long RemoteExitSkips;              // RewardSetSkippedMessage sent at a room exit
    public static long RemoteScreensClosedByExit;
    public static long RemoteTreeMap;
    public static long RemoteTreeFirstOfTwoBosses;
    public static long RemoteTreeActVote;
    public static long RemoteCardRewardFrames;
    public static long RemoteCardRewardDecisions;
    public static long RemoteChestsOpened;
    public static long RemoteTreasurePicks;
    public static long RemoteRestChoices;
    public static long RemoteRestExitSkips;          // RestSiteSkippedMessage sent at the rest-site exit
    public static long RemoteShopPurchases;
    public static long MendChoices;
    public static long MendTargetsAnswered;
    public static long ParasolRemoteOwner;

    private static SortedDictionary<string, long> Copy(SortedDictionary<string, long> d) => new SortedDictionary<string, long>(d, System.StringComparer.Ordinal);

    public static void Bump(SortedDictionary<string, long> d, string key)
    {
        d[key] = d.GetValueOrDefault(key) + 1;
    }

    public static void TreeCheck(bool ok, string what)
    {
        if (ok)
        {
            TreeChecksPassed++;
        }
        else
        {
            TreeChecksFailed++;
            FirstTreeCheckFailure ??= what;
        }
    }

    public static object Snapshot()
    {
        return new
        {
            tree = new
            {
                combats_won = CombatsWon,
                combat_won_with_rewards = CombatWonWithRewards,
                combat_won_without_rewards = CombatWonWithoutRewards,
                map = TreeMap,
                resume_event = TreeResumeEvent,
                first_of_two_bosses = TreeFirstOfTwoBosses,
                act_change = TreeActChange,
                architect = TreeArchitect,
                debug_override = TreeDebugOverride,
                no_rewards_map = NoRewardsMap,
                no_rewards_resume = NoRewardsResume,
                checks_passed = TreeChecksPassed,
                checks_failed = TreeChecksFailed,
                first_check_failure = FirstTreeCheckFailure,
            },
            rewards = new
            {
                sets_offered = RewardSetsOffered,
                terminal_sets_offered = TerminalSetsOffered,
                max_frame_depth = RewardFrameMaxDepth,
                takes = RewardTakes,
                take_rejected = RewardTakeRejected,
                auto_closed = RewardAutoClosed,
                non_terminal_skips = NonTerminalSkips,
                terminal_left_for_room_exit = TerminalLeftForRoomExit,
                potion_reward_masked = PotionRewardMasked,
                no_op_reward_masked = NoOpRewardMasked,
                proceed_disallowed = ProceedDisallowed,
                types_taken = Copy(RewardTypesTaken),
            },
            card_reward = new
            {
                decisions = CardRewardDecisions,
                reroll_filtered = CardRewardRerollFiltered,
                early_release = CardRewardEarlyRelease,
                alternatives_offered = Copy(CardRewardAlternativesOffered),
                alternatives_chosen = Copy(CardRewardAlternativesChosen),
                forwarded_selections = CardRewardForwardedSelections,
            },
            treasure = new
            {
                opened = ChestsOpened,
                gold = ChestGold,
                empty = ChestsEmpty,
                picks = TreasurePicks,
                skips = TreasureSkips,
                relics_obtained = TreasureRelicsObtained,
                fallback_moves = TreasureFallbackMoves,
            },
            rest = new
            {
                choices = RestChoices,
                successes = RestSuccesses,
                failures = RestFailures,
                choices_after_success = RestChoicesAfterSuccess,
                leaves = RestLeaves,
            },
            shop = new
            {
                purchases = ShopPurchases,
                purchase_cancelled = ShopPurchaseRejectedLegit,
                leaves = ShopLeaves,
                parasol_waits = ParasolWaits,
                parasol_yields = ParasolYields,
                bought = Copy(ShopEntryTypesBought),
            },
            potions = new
            {
                uses = PotionUses,
                discards = PotionDiscards,
                foul_potion_merchant_overrides = FoulPotionMerchantOverrides,
                foul_potion_merchant_kept_false = FoulPotionMerchantKeptFalse,
                use_offered_by_frame = Copy(PotionUseOfferedByKind),
                used = Copy(PotionsUsed),
            },
        };
    }

    public static void Reset()
    {
        CombatsWon = CombatWonWithRewards = CombatWonWithoutRewards = 0;
        TreeMap = TreeResumeEvent = TreeFirstOfTwoBosses = TreeActChange = TreeArchitect = TreeDebugOverride = 0;
        NoRewardsMap = NoRewardsResume = TreeChecksPassed = TreeChecksFailed = 0;
        FirstTreeCheckFailure = null;
        RewardSetsOffered = TerminalSetsOffered = RewardFrameMaxDepth = RewardTakes = RewardTakeRejected = 0;
        RewardAutoClosed = NonTerminalSkips = TerminalLeftForRoomExit = PotionRewardMasked = NoOpRewardMasked = ProceedDisallowed = 0;
        RewardTypesTaken.Clear();
        CardRewardDecisions = CardRewardRerollFiltered = CardRewardEarlyRelease = CardRewardForwardedSelections = 0;
        CardRewardAlternativesOffered.Clear();
        CardRewardAlternativesChosen.Clear();
        ChestsOpened = ChestGold = ChestsEmpty = TreasurePicks = TreasureSkips = TreasureRelicsObtained = TreasureFallbackMoves = 0;
        RestChoices = RestSuccesses = RestFailures = RestChoicesAfterSuccess = RestLeaves = 0;
        ShopPurchases = ShopPurchaseRejectedLegit = ShopLeaves = ParasolWaits = ParasolYields = 0;
        ShopEntryTypesBought.Clear();
        PotionUses = PotionDiscards = FoulPotionMerchantOverrides = FoulPotionMerchantKeptFalse = 0;
        PotionUseOfferedByKind.Clear();
        PotionsUsed.Clear();
        RemoteSetsOffered = RemoteTerminalSetsOffered = RemoteSetsWithoutScreen = RemoteScreenMaxDepth = RemoteTakes = RemoteTakeRejected = 0;
        RemoteAutoClosed = RemoteNonTerminalSkips = RemoteTerminalLeftForRoomExit = RemoteExitSkips = RemoteScreensClosedByExit = 0;
        RemoteTreeMap = RemoteTreeFirstOfTwoBosses = RemoteTreeActVote = RemoteCardRewardFrames = RemoteCardRewardDecisions = 0;
        RemoteChestsOpened = RemoteTreasurePicks = RemoteRestChoices = RemoteRestExitSkips = RemoteShopPurchases = 0;
        MendChoices = MendTargetsAnswered = ParasolRemoteOwner = 0;
    }
}
