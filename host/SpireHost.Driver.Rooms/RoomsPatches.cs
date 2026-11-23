using System;
using System.Reflection;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Driver;
using SpireHost.Patches;

namespace SpireHost.Driver.Rooms;

/// <summary>
/// The rooms team's patches and registered interventions (ledger: host\FIDELITY.rooms.md). Installed once per
/// process from RoomsModule.Register (after boot, on the engine thread), fail-closed through PatchRegistry: a missing
/// target throws at registration. Patch ownership (task): RewardsSet.Offer test-branch handling, LordsParasol
/// yield / wait, FoulPotion merchant-room usability / effect. NOT here: the MerchantPotionEntry price gate,
/// CallingBell / Cauldron, the MerchantRoom.Exit bookkeeping gate (fidelity team), FakeMerchant (events team).
/// </summary>
internal static class RoomsPatches
{
    private static bool _installed;

    public const string FoulPotionMerchantId = "rooms-foul-potion-merchant-usability";
    public const string ParasolYieldId = "rooms-lords-parasol-ui-yield";
    public const string RewardsTestSelectorId = "rooms-rewards-test-selector";
    public const string TerminalNoThrowId = "rooms-rewards-terminal-left-for-room-exit";
    public const string CardRewardSelectorId = "rooms-card-reward-stack-selector";
    public const string RemoteSetBegunId = "rooms-remote-reward-set-begun";
    public const string RemoteAlternativesId = "rooms-remote-card-reward-alternatives";
    public const string RemoteExitSkipsId = "rooms-remote-room-exit-skips";
    public const string RemoteRestExitSkipId = "rooms-remote-rest-exit-skip";

    public static void InstallOnce()
    {
        if (_installed)
        {
            return;
        }
        _installed = true;

        // ---- FoulPotion in the merchant room: usability (UI-node dependency, no TestMode gate) ----
        PatchRegistry.Postfix(FoulPotionMerchantId,
            AccessTools.PropertyGetter(typeof(FoulPotion), nameof(FoulPotion.PassesCustomUsabilityCheck)),
            PatchRegistry.M(typeof(RoomsPatches), nameof(FoulPotionUsabilityPostfix)),
            PatchCategory.Fidelity,
            "FoulPotion.PassesCustomUsabilityCheck (Models.Potions/FoulPotion.cs:51-84): out of combat in a MerchantRoom the real game requires " +
            "GetFoulPotionMerchantTarget(room).button != null (:137-150), i.e. NMerchantRoom.Instance with a closed inventory - a UI node that never exists " +
            "headlessly, so the potion was never usable at the merchant. Postfix: in a MerchantRoom, out of combat, with no NMerchantRoom node, on the " +
            "room-level SHOP screen (episode running, map not opened by LEAVE, no pending prompt - the states where the real button is visible with the " +
            "inventory closed) the result is true. Every other case keeps the original result. SHARED SITE (rulings B3 / D4): events-fake-merchant-foul-potion-usable " +
            "is a postfix on the same getter for the FakeMerchant EventRoom branch; the two compose. The effect needs no patch: OnUse in a MerchantRoom skips the null-safe UI calls " +
            "and runs PlayerCmd.GainGold(100) (:94-103). Self-check: rooms --selftest shop:foul-potion (usable, +100 gold).");

        // ---- LordsParasol: the UI block becomes one yield ----
        PatchRegistry.Prefix(ParasolYieldId,
            AccessTools.Method(typeof(LordsParasol), nameof(LordsParasol.PurchaseEverythingLocked)),
            PatchRegistry.M(typeof(RoomsPatches), nameof(ParasolPrefix)),
            PatchCategory.Fidelity,
            "LordsParasol.PurchaseEverythingLocked (Models.Relics/LordsParasol.cs:51-71): the real game (TestMode.IsOff) blocks the UI and awaits " +
            "NRun.AwaitProcessFrame (+ Cmd.Wait) before the first purchase, so AfterRoomEntered returns and the rest of the room entry runs first; the " +
            "TestMode branch buys synchronously inside AfterRoomEntered. DECISIONS-H1 E: run the original method and replace the UI block by one yield. " +
            "Prefix: the first call returns a task that does `await Task.Yield()` (one pump turn on the engine thread) and then calls the ORIGINAL method " +
            "(pass-through flag); the purchase lock taken by PurchaseEverything (:35-49) is held across the yield as in the real game. The SHOP frame " +
            "waits for LastPurchaseEverythingTask (ShopHandler). Self-check: rooms --selftest shop:lords-parasol.");

        InstallRemoteSeats();

        // ---- registered non-Harmony interventions (game test hooks / public switches) ----
        PatchRegistry.Note(RewardsTestSelectorId, "MegaCrit.Sts2.Core.Rewards.RewardsSet::testSelector", "static hook (per run)", PatchCategory.Fidelity,
            "RewardsSet.Offer (Rewards/RewardsSet.cs:140-158) TestMode branch: without testSelector every reward is taken in order and an incomplete set " +
            "throws. The rooms driver installs testSelector per run (OnRunSetUp, removed in OnEpisodeEnd) and parks a REWARD frame: rewards are taken in " +
            "any order through RewardsSetSynchronizer.SelectLocalReward and PROCEED follows NRewardsScreen (:421-504) - the real UI branch (:162).");
        PatchRegistry.Note(TerminalNoThrowId, "MegaCrit.Sts2.Core.Rewards.RewardsSet::ThrowInTestIfRewardsNotTaken", "property write (per set)", PatchCategory.Fidelity,
            "A terminal (combat room) reward screen does not skip its set on PROCEED: the set stays on the synchronizer stack until the room is left " +
            "(RewardsSetSynchronizer.BeforeLeavingRoom :446-466 via RunManager.ExitCurrentRoom). On PROCEED with untaken rewards the driver sets the " +
            "game's public ThrowInTestIfRewardsNotTaken = false on that set, so the TestMode-only assertion (:155-158) does not fire and OnSkipped runs at " +
            "room exit exactly as in the real game.");
        PatchRegistry.Note(CardRewardSelectorId, "MegaCrit.Sts2.Core.CardSelection.CardSelectorStack::PushSelector", "selector scope (per card reward)", PatchCategory.Fidelity,
            "CardReward.OnSelect only reads the per-player selector stack when NCardRewardSelectionScreen.ShowScreen returns null (TestMode; " +
            "Rewards/CardReward.cs:181-188). The driver pushes its own selector for that one reward and pops it as soon as the selection ends, so " +
            "prompts raised later in OnSelect go through the real local-selector path; GetSelectedCards on it is forwarded to the run's selector.");
    }

    // -----------------------------------------------------------------------------------------------------

    public static void FoulPotionUsabilityPostfix(FoulPotion __instance, ref bool __result)
    {
        if (__result || CombatManager.Instance.IsInProgress)
        {
            return;
        }
        if (__instance.Owner?.RunState?.CurrentRoom is not MerchantRoom || NMerchantRoom.Instance != null)
        {
            return; // not the MerchantRoom branch (FakeMerchant is the events team's postfix), or a real node decides
        }
        // GetFoulPotionMerchantTarget (FoulPotion.cs:137-150) needs the merchant button visible with the inventory closed.
        // Host stand-in: the room-level SHOP screen itself - not while a prompt raised by a purchase is open (a pending
        // frame: the inventory is open in the real UI) and not after LEAVE (the map screen covers the room; the host's
        // map is one-way). Outside an episode the original result stands.
        // Multi-seat: the screen in question is the OWNER's (its own map, its own reward overlay).
        RunContext? ctx = HostSession.Current?.CurrentEpisode;
        int seat = ctx == null ? 0 : ctx.SeatOf(__instance.Owner);
        if (ctx == null || ctx.IsMapOpen(seat) || ctx.PendingCount > 0 || RemoteRewards.Instance?.HasScreen(seat) == true)
        {
            RoomsProbe.FoulPotionMerchantKeptFalse++;
            return;
        }
        __result = true;
        RoomsProbe.FoulPotionMerchantOverrides++;
    }

    // -----------------------------------------------------------------------------------------------------
    //  MULTI-SEAT (phase H5): observation points of the remote seats' reward screens and room exits
    // -----------------------------------------------------------------------------------------------------

    private static void InstallRemoteSeats()
    {
        const string Inert = " MULTI-SEAT ONLY (phase H5; inert with one seat and outside an episode).";

        PatchRegistry.Postfix(RemoteSetBegunId,
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer), nameof(MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer.BeginRewardsSet)),
            PatchRegistry.M(typeof(RoomsPatches), nameof(RewardsSetBegunPostfix)), PatchCategory.Instrumentation,
            "Read-only: RewardsSetSynchronizer.BeginRewardsSet (Multiplayer.Game/RewardsSetSynchronizer.cs:246-278) is called by RewardsSet.Offer (Rewards/RewardsSet.cs:131) for every player. " +
            "For a player that is not the local one the host only awaits the set's completion (:165); the reward screen exists on that player's client. Postfix: a set of a remote " +
            "player is noted as that seat's reward screen (RemoteRewards), unless Offer returns without a screen (:132: no reward, not a combat room, empty sets not allowed). " +
            "Arguments, result and game state untouched." + Inert);

        PatchRegistry.Postfix(RemoteAlternativesId,
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative), nameof(MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative.Generate)),
            PatchRegistry.M(typeof(RoomsPatches), nameof(AlternativesGeneratedPostfix)), PatchCategory.Instrumentation,
            "Read-only: CardRewardAlternative.Generate (Entities.CardRewardAlternatives/CardRewardAlternative.cs:41-63) builds the alternative buttons of a card reward: at the start of " +
            "CardReward.OnSelect (Rewards/CardReward.cs:167, a local the host indexes the answers with) and in CardReward.Populate after a reroll (:139-141, the list the client's " +
            "refreshed screen shows). Postfix: the latest list of a remote player's card reward is remembered; it is the button row of that seat's CARD_REWARD frame, and the answer " +
            "the client sends is (offered cards) + (position of the button on its screen) (NCardRewardSelectionScreen.cs:267). Arguments, result and game state untouched." + Inert);

        PatchRegistry.Postfix(RemoteExitSkipsId,
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer), nameof(MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer.BeforeLeavingRoom)),
            PatchRegistry.M(typeof(RoomsPatches), nameof(BeforeLeavingRoomPostfix)), PatchCategory.Instrumentation,
            "Observation point of a client duty. RewardsSetSynchronizer.BeforeLeavingRoom (Multiplayer.Game/RewardsSetSynchronizer.cs:446-466; called by RunManager.ExitCurrentRoom, " +
            "Runs/RunManager.cs:1053, before the room is popped) skips the LOCAL player's open reward sets and broadcasts one RewardSetSkippedMessage per set. Every client runs the same " +
            "line for its own player when it leaves the room. Postfix: for every remote seat, that seat's client sends RewardSetSkippedMessage for each of its open sets, top first " +
            "(VirtualClient.Send; the host handles it in HandleRewardSetSkippedMessage :336-354), and the seat's reward screens are closed. Delivered inline, at the same point where the " +
            "game skips the local sets inline: the handler skips the sender's stack top whatever the set id says, so a later delivery could hit a set of the next room. Without it an " +
            "event room would never be left (EventRoom.Exit awaits the option tasks that await such sets, Rooms/EventRoom.cs:83). The patch itself changes no argument, result or state." + Inert);

        PatchRegistry.Postfix(RemoteRestExitSkipId,
            AccessTools.Method(typeof(MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer), nameof(MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer.BeforeLocalRestSiteExited)),
            PatchRegistry.M(typeof(RoomsPatches), nameof(BeforeLocalRestSiteExitedPostfix)), PatchCategory.Instrumentation,
            "Observation point of a client duty. RestSiteSynchronizer.BeforeLocalRestSiteExited (Multiplayer.Game/RestSiteSynchronizer.cs:215-230; called by RestSiteRoom.Exit, " +
            "Rooms/RestSiteRoom.cs:43) completes the LOCAL player's rest site when options are left and broadcasts RestSiteSkippedMessage; RestSiteRoom.Exit then awaits the completion " +
            "of EVERY player (:45). Every client runs the same line for its own player when it leaves. Postfix: for every remote seat that still has options, that seat's client sends " +
            "RestSiteSkippedMessage (handled at :126-137). Inline for the same reason as rooms-remote-room-exit-skips. The patch itself changes no argument, result or state." + Inert);
    }

    public static void RewardsSetBegunPostfix(MegaCrit.Sts2.Core.Rewards.RewardsSet set)
    {
        RemoteRewards.Instance?.OnSetBegun(set);
    }

    public static void AlternativesGeneratedPostfix(MegaCrit.Sts2.Core.Rewards.CardReward cardReward, System.Collections.Generic.IReadOnlyList<MegaCrit.Sts2.Core.Entities.CardRewardAlternatives.CardRewardAlternative> __result)
    {
        RemoteRewards.Instance?.OnAlternativesGenerated(cardReward, __result);
    }

    public static void BeforeLeavingRoomPostfix(MegaCrit.Sts2.Core.Multiplayer.Game.RewardsSetSynchronizer __instance)
    {
        RemoteRewards.Instance?.OnBeforeLeavingRoom(__instance);
    }

    public static void BeforeLocalRestSiteExitedPostfix(MegaCrit.Sts2.Core.Multiplayer.Game.RestSiteSynchronizer __instance)
    {
        RunContext? ctx = HostSession.Current?.CurrentEpisode;
        if (ctx?.Mp == null)
        {
            return;
        }
        for (int seat = 1; seat < ctx.SeatCount; seat++)
        {
            if (__instance.GetOptionsForPlayer(ctx.Seats[seat]).Count > 0)
            {
                VirtualClient client = ctx.Mp.Client(seat);
                RoomsProbe.RemoteRestExitSkips++;
                client.Send(new MegaCrit.Sts2.Core.Multiplayer.Messages.Game.Sync.RestSiteSkippedMessage { Location = client.Location });
            }
        }
    }

    [ThreadStatic]
    private static bool _parasolPassThrough;

    public static bool ParasolPrefix(LordsParasol __instance, MerchantInventory inventory, ref Task __result)
    {
        if (_parasolPassThrough)
        {
            return true; // the original method
        }
        if (!MegaCrit.Sts2.Core.Context.LocalContext.IsMe(__instance.Owner))
        {
            // Multi-seat, a remote owner: the real game blocks the UI and waits only when the owner is the local player
            // (LordsParasol.cs:54, :57: TestMode.IsOff && ownerIsLocal); for anybody else's parasol the host buys at once.
            RoomsProbe.ParasolRemoteOwner++;
            return true;
        }
        __result = ParasolAfterYield(__instance, inventory);
        return false;
    }

    private static async Task ParasolAfterYield(LordsParasol parasol, MerchantInventory inventory)
    {
        RoomsProbe.ParasolYields++;
        await Task.Yield(); // continuation posted to the engine pump: "on a later frame"
        Task original;
        _parasolPassThrough = true;
        try
        {
            original = parasol.PurchaseEverythingLocked(inventory);
        }
        finally
        {
            _parasolPassThrough = false;
        }
        await original;
    }
}
