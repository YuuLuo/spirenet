using System.Collections.Generic;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

/// <summary>
/// The counters of the rooms module as a module-counter source (contract request R3): the terminal proceed tree
/// with its post-condition checks, reward / card-reward / treasure / rest / shop frames, out-of-combat potions, the
/// module's fidelity patches. Values are the cumulative statics of <see cref="RoomsProbe"/>; the driver core stores
/// per-episode differences in EpisodeResult.ModuleCounters ("rooms.*") and --fuzz-runs / --determinism sum them.
/// </summary>
internal sealed class RoomsCounters : IModuleCounters
{
    public string Module => "rooms";

    public void Read(IDictionary<string, long> c)
    {
        c["tree.combats_won"] = RoomsProbe.CombatsWon;
        c["tree.combat_won_with_rewards"] = RoomsProbe.CombatWonWithRewards;
        c["tree.combat_won_without_rewards"] = RoomsProbe.CombatWonWithoutRewards;
        c["tree.map"] = RoomsProbe.TreeMap;
        c["tree.resume_event"] = RoomsProbe.TreeResumeEvent;
        c["tree.first_of_two_bosses"] = RoomsProbe.TreeFirstOfTwoBosses;
        c["tree.act_change"] = RoomsProbe.TreeActChange;
        c["tree.architect"] = RoomsProbe.TreeArchitect;
        c["tree.debug_override"] = RoomsProbe.TreeDebugOverride;
        c["tree.no_rewards_map"] = RoomsProbe.NoRewardsMap;
        c["tree.no_rewards_resume"] = RoomsProbe.NoRewardsResume;
        c["tree.checks_passed"] = RoomsProbe.TreeChecksPassed;
        c["tree.checks_failed"] = RoomsProbe.TreeChecksFailed;

        c["rewards.sets_offered"] = RoomsProbe.RewardSetsOffered;
        c["rewards.terminal_sets_offered"] = RoomsProbe.TerminalSetsOffered;
        c["rewards.frame_depth_max"] = RoomsProbe.RewardFrameMaxDepth;
        c["rewards.takes"] = RoomsProbe.RewardTakes;
        c["rewards.take_rejected"] = RoomsProbe.RewardTakeRejected;
        c["rewards.auto_closed"] = RoomsProbe.RewardAutoClosed;
        c["rewards.non_terminal_skips"] = RoomsProbe.NonTerminalSkips;
        c["rewards.terminal_left_for_room_exit"] = RoomsProbe.TerminalLeftForRoomExit;
        c["rewards.potion_reward_masked"] = RoomsProbe.PotionRewardMasked;
        c["rewards.no_op_reward_masked"] = RoomsProbe.NoOpRewardMasked;
        c["rewards.proceed_disallowed"] = RoomsProbe.ProceedDisallowed;
        Add(c, "rewards.types_taken.", RoomsProbe.RewardTypesTaken);

        c["card_reward.decisions"] = RoomsProbe.CardRewardDecisions;
        c["card_reward.reroll_filtered"] = RoomsProbe.CardRewardRerollFiltered;
        c["card_reward.early_release"] = RoomsProbe.CardRewardEarlyRelease;
        c["card_reward.forwarded_selections"] = RoomsProbe.CardRewardForwardedSelections;
        Add(c, "card_reward.alternatives_offered.", RoomsProbe.CardRewardAlternativesOffered);
        Add(c, "card_reward.alternatives_chosen.", RoomsProbe.CardRewardAlternativesChosen);

        c["treasure.opened"] = RoomsProbe.ChestsOpened;
        c["treasure.gold"] = RoomsProbe.ChestGold;
        c["treasure.empty"] = RoomsProbe.ChestsEmpty;
        c["treasure.picks"] = RoomsProbe.TreasurePicks;
        c["treasure.skips"] = RoomsProbe.TreasureSkips;
        c["treasure.relics_obtained"] = RoomsProbe.TreasureRelicsObtained;
        c["treasure.fallback_moves"] = RoomsProbe.TreasureFallbackMoves;

        c["rest.choices"] = RoomsProbe.RestChoices;
        c["rest.successes"] = RoomsProbe.RestSuccesses;
        c["rest.failures"] = RoomsProbe.RestFailures;
        c["rest.choices_after_success"] = RoomsProbe.RestChoicesAfterSuccess;
        c["rest.leaves"] = RoomsProbe.RestLeaves;

        c["shop.purchases"] = RoomsProbe.ShopPurchases;
        c["shop.purchase_cancelled"] = RoomsProbe.ShopPurchaseRejectedLegit;
        c["shop.leaves"] = RoomsProbe.ShopLeaves;
        c["shop.parasol_waits"] = RoomsProbe.ParasolWaits;
        c["shop.parasol_yields"] = RoomsProbe.ParasolYields;
        Add(c, "shop.bought.", RoomsProbe.ShopEntryTypesBought);

        c["potions.uses"] = RoomsProbe.PotionUses;
        c["potions.discards"] = RoomsProbe.PotionDiscards;
        c["potions.foul_potion_merchant_overrides"] = RoomsProbe.FoulPotionMerchantOverrides;
        c["potions.foul_potion_merchant_kept_false"] = RoomsProbe.FoulPotionMerchantKeptFalse;
        Add(c, "potions.use_offered_by_frame.", RoomsProbe.PotionUseOfferedByKind);
        Add(c, "potions.used.", RoomsProbe.PotionsUsed);

        // multi-seat (phase H5): the remote seats' screens
        c["remote.reward_screens"] = RoomsProbe.RemoteSetsOffered;
        c["remote.reward_screens_terminal"] = RoomsProbe.RemoteTerminalSetsOffered;
        c["remote.sets_without_screen"] = RoomsProbe.RemoteSetsWithoutScreen;
        c["remote.reward_screen_depth_max"] = RoomsProbe.RemoteScreenMaxDepth;
        c["remote.reward_takes"] = RoomsProbe.RemoteTakes;
        c["remote.reward_take_rejected"] = RoomsProbe.RemoteTakeRejected;
        c["remote.reward_auto_closed"] = RoomsProbe.RemoteAutoClosed;
        c["remote.reward_non_terminal_skips"] = RoomsProbe.RemoteNonTerminalSkips;
        c["remote.reward_terminal_left_for_room_exit"] = RoomsProbe.RemoteTerminalLeftForRoomExit;
        c["remote.reward_exit_skips"] = RoomsProbe.RemoteExitSkips;
        c["remote.reward_screens_closed_by_exit"] = RoomsProbe.RemoteScreensClosedByExit;
        c["remote.tree.map"] = RoomsProbe.RemoteTreeMap;
        c["remote.tree.first_of_two_bosses"] = RoomsProbe.RemoteTreeFirstOfTwoBosses;
        c["remote.tree.act_vote"] = RoomsProbe.RemoteTreeActVote;
        c["remote.card_reward_frames"] = RoomsProbe.RemoteCardRewardFrames;
        c["remote.card_reward_decisions"] = RoomsProbe.RemoteCardRewardDecisions;
        c["remote.chests_opened"] = RoomsProbe.RemoteChestsOpened;
        c["remote.treasure_picks"] = RoomsProbe.RemoteTreasurePicks;
        c["remote.rest_choices"] = RoomsProbe.RemoteRestChoices;
        c["remote.rest_exit_skips"] = RoomsProbe.RemoteRestExitSkips;
        c["remote.shop_purchases"] = RoomsProbe.RemoteShopPurchases;
        c["rest.mend_choices"] = RoomsProbe.MendChoices;
        c["rest.mend_targets_answered"] = RoomsProbe.MendTargetsAnswered;
        c["shop.parasol_remote_owner"] = RoomsProbe.ParasolRemoteOwner;
    }

    private static void Add(IDictionary<string, long> c, string prefix, SortedDictionary<string, long> source)
    {
        foreach (KeyValuePair<string, long> kv in source)
        {
            c[prefix + kv.Key] = kv.Value;
        }
    }
}
