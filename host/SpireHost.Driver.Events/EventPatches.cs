using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Patches;

namespace SpireHost.Driver.Events;

/// <summary>
/// Every Harmony patch of the events module (ledger: host\FIDELITY.md, same ids). All are installed once per
/// process from EventsModule.Register, fail closed (a missing target or a wrong instruction count throws at boot),
/// and do nothing outside an episode except where stated.
///
///   Fidelity (UI-node dependencies of game logic, DECISIONS-H1 E; testmode-audit-nodes.md U-EV-4/5/6, U-CS-2)
///     events-crystal-sphere-screen              NCrystalSphereScreen.ShowScreen        -> CRYSTAL frame
///     events-trial-double-down-confirm          Trial.DoubleDown                       -> CONFIRM frame
///     events-bundle-gate                        CardSelectCmd.FromChooseABundleScreen  TestMode.IsOn -> false (real branch)
///     events-bundle-screen                      the screen of that real branch         -> BUNDLE frame
///     events-fake-merchant-foul-potion-usable   FoulPotion.PassesCustomUsabilityCheck  FakeMerchant branch without NFakeMerchant
///     events-fake-merchant-foul-potion-use      FoulPotion.OnUse                       FakeMerchant branch without NFakeMerchant
///
///   Plumbing (presentation calls on UI singletons that are null headlessly; every game call around them runs)
///     events-waterfall-enter-portrait           WaterfallOfObviousSecrets.EnterWaterfall :53
///     events-dense-vegetation-rest-presentation DenseVegetation.Rest :93-97
///     events-amalgamator-combine-strikes-shake  Amalgamator.CombineStrikes :46,:49
///     events-amalgamator-combine-defends-shake  Amalgamator.CombineDefends :60,:63
///     events-jungle-maze-safety-audio           JungleMazeAdventure.SafetyInNumbers :94
///     events-punch-off-nab-shake                PunchOff.Nab :106
///     events-trial-accept-portrait              Trial.Accept :68,:108
///     events-trial-portrait-vfx                 Trial.AddVfxAnchoredToPortrait :136-141 (whole helper is presentation)
/// </summary>
public static class EventPatches
{
    private static bool _installed;
    private static readonly Dictionary<string, long> _counts = new Dictionary<string, long>(StringComparer.Ordinal);
    private static BundleFrame? _shownBundle;

    /// <summary>Hit counters per patch id (the self-check of each ledger row). Engine thread.</summary>
    public static IReadOnlyDictionary<string, long> Counts => _counts;

    internal static void Count(string id)
    {
        _counts[id] = _counts.GetValueOrDefault(id) + 1;
    }

    internal static void BeginEpisode()
    {
        _shownBundle = null;
        Array.Clear(_remoteBundles);
    }

    internal static void EndEpisode()
    {
        _shownBundle = null;
        Array.Clear(_remoteBundles);
    }

    private static RunContext? Episode => HostSession.Current?.CurrentEpisode;

    public static void InstallOnce()
    {
        if (_installed)
        {
            return;
        }
        _installed = true;
        InstallFrames();
        InstallFoulPotion();
        InstallPresentation();

        // Ruling B3 (one patch per site): the methods that must carry the patches of two ids. Every row of the ledger names its partner.
        PatchRegistry.DeclareSharedSite("CardSelectCmd.FromChooseABundleScreen: in the state machine the TestMode gate is forced to the real branch (events-bundle-gate) and the two UI calls " +
            "of that real branch are swapped for the host BUNDLE frame (events-bundle-screen); neither works without the other. On the method stub a read-only prefix remembers the " +
            "arguments of a REMOTE player's call (events-bundle-remote-capture, multi-seat only): the real branch of the same gate sends such a player to WaitForRemoteChoice, and the " +
            "wait needs the bundles to become that seat's BUNDLE frame.", "events-bundle-gate", "events-bundle-screen", "events-bundle-remote-capture");
        PatchRegistry.DeclareSharedSite("Amalgamator.CombineStrikes (state machine): the real Task.Delay(300) goes to the virtual clock (Core) and the two unguarded screen-shake calls " +
            "around it are made null-safe (events).", "virtual-clock-amalgamator-combinestrikes", "events-amalgamator-combine-strikes-shake");
        PatchRegistry.DeclareSharedSite("Amalgamator.CombineDefends (state machine): the real Task.Delay(300) goes to the virtual clock (Core) and the two unguarded screen-shake calls " +
            "around it are made null-safe (events).", "virtual-clock-amalgamator-combinedefends", "events-amalgamator-combine-defends-shake");
        PatchRegistry.DeclareSharedSite("FoulPotion.PassesCustomUsabilityCheck (getter): two room-specific postfixes that compose - each only turns false into true under its own room " +
            "condition (rooms: MerchantRoom; events: the FakeMerchant EventRoom). Ruling D4.", "rooms-foul-potion-merchant-usability", "events-fake-merchant-foul-potion-usable");
    }

    // =====================================================================================================
    //  Frames: Crystal Sphere, Trial double-down, bundles
    // =====================================================================================================

    private static void InstallFrames()
    {
        PatchRegistry.Prefix("events-crystal-sphere-screen",
            AccessTools.Method(typeof(NCrystalSphereScreen), nameof(NCrystalSphereScreen.ShowScreen), new[] { typeof(CrystalSphereMinigame) }),
            PatchRegistry.M(typeof(EventPatches), nameof(CrystalShowScreenPrefix)), PatchCategory.Fidelity,
            "NCrystalSphereScreen.ShowScreen (NCrystalSphereScreen.cs:146-152) has no TestMode gate and instantiates a scene; it is called by " +
            "CrystalSphereMinigame.PlayMinigame (CrystalSphereMinigame.cs:135-143) for the local player, which then awaits the minigame's completion source. " +
            "Inside an episode the prefix pushes the host CRYSTAL frame (the screen's SetTool / CellClicked calls) and returns null (the caller ignores the result). Outside an episode the original runs.");

        PatchRegistry.Prefix("events-trial-double-down-confirm",
            AccessTools.Method(typeof(Trial), "DoubleDown"),
            PatchRegistry.M(typeof(EventPatches), nameof(TrialDoubleDownPrefix)), PatchCategory.Fidelity,
            "Trial.DoubleDown (Trial.cs:128-132) = NModalContainer.Instance.Add(NAbandonRunConfirmPopup.Create(null)): NRE headlessly (TestMode Create returns null, the container is null). " +
            "Inside an episode the prefix pushes the CONFIRM frame of that popup (YES -> RunManager.Abandon(), NAbandonRunConfirmPopup.cs:98-108; NO -> close) and returns the same completed task.");

        MethodInfo bundle = AccessTools.Method(typeof(CardSelectCmd), nameof(CardSelectCmd.FromChooseABundleScreen))
            ?? throw new MissingMethodException("CardSelectCmd.FromChooseABundleScreen");
        GatePatch.ForceGetter("events-bundle-gate", bundle, GatePatch.TestModeIsOn, false, 1, PatchCategory.Fidelity,
            "CardSelectCmd.FromChooseABundleScreen (CardSelectCmd.cs:670-673): the TestMode branch always takes bundles[0] without asking. " +
            "Forced to the real branch: ShouldSelectLocalCard -> the bundle screen -> SyncLocalChoice(choiceId reserved at :668) -> LogChoice. " +
            "SHARED SITE (ruling B3): events-bundle-screen swaps the two UI calls of that real branch in the same method.",
            async: true);

        MethodBase bundleMoveNext = AccessTools.AsyncMoveNext(bundle) ?? throw new MissingMethodException("FromChooseABundleScreen.MoveNext");
        CallSwap.Install("events-bundle-screen", bundleMoveNext, PatchCategory.Fidelity,
            "The real branch of FromChooseABundleScreen (CardSelectCmd.cs:674-679) shows NChooseABundleSelectionScreen (scene instantiation) and awaits its CardsSelected(). " +
            "Exactly those two calls are swapped for host stand-ins: ShowScreen pushes the BUNDLE frame, CardsSelected returns the frame's task. ReserveChoiceId / SyncLocalChoice / LogChoice run unchanged. " +
            "SHARED SITE (ruling B3): events-bundle-gate forces the TestMode gate of the same method to the real branch.",
            new CallSwap.Spec(AccessTools.Method(typeof(NChooseABundleSelectionScreen), nameof(NChooseABundleSelectionScreen.ShowScreen)),
                PatchRegistry.M(typeof(EventPatches), nameof(BundleShowScreen)), 1),
            new CallSwap.Spec(AccessTools.Method(typeof(NChooseABundleSelectionScreen), nameof(NChooseABundleSelectionScreen.CardsSelected)),
                PatchRegistry.M(typeof(EventPatches), nameof(BundleCardsSelected)), 1));

        // ---- MULTI-SEAT (phase H5): the bundle prompt and the crystal sphere of a REMOTE player ----
        PatchRegistry.Prefix("events-bundle-remote-capture", bundle,
            PatchRegistry.M(typeof(EventPatches), nameof(BundleRemoteCapturePrefix)), PatchCategory.Instrumentation,
            "MULTI-SEAT ONLY (phase H5; inert with one seat and outside an episode). Read-only: the arguments of CardSelectCmd.FromChooseABundleScreen (Commands/CardSelectCmd.cs:657) " +
            "for a remote player - the bundles, canSkip and the choice id the call is about to reserve (:668) - are remembered so that the wait of its remote branch " +
            "(:681: WaitForRemoteChoice) becomes that seat's BUNDLE frame (events:bundle-remote router); the frame's answer is the client's PlayerChoiceMessage " +
            "FromIndex(bundle index, -1 = skip), what the client's local branch syncs (:679). Arguments, result and game state untouched. (The method's state machine carries " +
            "events-bundle-gate / events-bundle-screen; this prefix is on the method stub.)");

        PatchRegistry.Prefix("events-crystal-sphere-remote",
            AccessTools.Method(typeof(CrystalSphereMinigame), nameof(CrystalSphereMinigame.PlayMinigame)),
            PatchRegistry.M(typeof(EventPatches), nameof(CrystalPlayRemotePrefix)), PatchCategory.Instrumentation,
            "MULTI-SEAT ONLY (phase H5; lead ruling Q4 (a)). CrystalSphereMinigame.PlayMinigame (Events.Custom.CrystalSphereEvent/CrystalSphereMinigame.cs:135-143) returns at once " +
            "when the owner is not the local player: that player's client plays the minigame and sends its result (CrystalSphereRewardsMessage). Prefix, remote owners only: " +
            "parks the CRYSTAL frame of that seat over the host's own minigame object of that player (same constructor arguments and event Rng as on the client, so the same layout); " +
            "the original method runs unchanged. The seat's clicks call SetTool / CellClicked on that object; after the last click the seat's client sends " +
            "CrystalSphereRewardsMessage with the revealed items (OneOffSynchronizer.cs:113-143). KNOWN DIFFERENCE: a curse revealed by a click is added by " +
            "CrystalSphereCurse.RevealItem -> CardPileCmd.AddCurseToDeck on the host object (the code the client runs) instead of by the client's RewardObtainedMessage " +
            "(RewardSynchronizer.HandleRewardObtainedMessage :181-215): same card, same hooks, but the host path's CardChoiceHistoryEntry(card, wasPicked: true) is not written " +
            "and the host broadcasts a RewardObtainedMessage of its own (no receiver). Readers of CardChoices: FIDELITY section on this patch.");

        PatchRegistry.Prefix("events-trial-portrait-vfx",
            AccessTools.Method(typeof(Trial), "AddVfxAnchoredToPortrait"),
            PatchRegistry.M(typeof(EventPatches), nameof(TrialPortraitVfxPrefix)), PatchCategory.Plumbing,
            "Trial.AddVfxAnchoredToPortrait (Trial.cs:134-142) is a presentation-only helper (instantiate a VFX scene, position it, add it to NEventRoom.Instance.Layout); " +
            "headlessly GetScene returns null and .Instantiate throws. Skipped when NEventRoom.Instance is null; the caller (Trial.Accept) continues with SetEventState as in the real game.");
    }

    public static bool CrystalShowScreenPrefix(CrystalSphereMinigame grid, ref NCrystalSphereScreen? __result)
    {
        RunContext? ctx = Episode;
        if (ctx == null)
        {
            return true;
        }
        Count("events-crystal-sphere-screen");
        ctx.Push(new CrystalFrame(grid));
        __result = null;
        return false;
    }

    public static bool TrialDoubleDownPrefix(Trial __instance, ref Task __result)
    {
        RunContext? ctx = Episode;
        if (ctx == null)
        {
            return true;
        }
        Count("events-trial-double-down-confirm");
        ctx.Push(new TrialConfirmFrame(__instance));
        __result = Task.CompletedTask;
        return false;
    }

    // ---- remote bundle prompt ----

    private sealed class RemoteBundle
    {
        public RunContext Ctx = null!;
        public IReadOnlyList<IReadOnlyList<CardModel>> Bundles = null!;
        public bool CanSkip;
        public uint ChoiceId;
    }

    private static readonly RemoteBundle?[] _remoteBundles = new RemoteBundle?[DriverApi.MaxSeats];

    public static void BundleRemoteCapturePrefix(Player player, IReadOnlyList<IReadOnlyList<CardModel>> bundles, bool canSkip)
    {
        RunContext? ctx = Episode;
        if (ctx?.Mp == null)
        {
            return;
        }
        int seat = ctx.SeatOf(player);
        if (!ctx.IsRemoteSeat(seat))
        {
            return;
        }
        // The id the call reserves next (PlayerChoiceSynchronizer.ReserveChoiceId returns the slot's counter, :50-61).
        IReadOnlyList<uint> ids = ctx.RunManager.PlayerChoiceSynchronizer.ChoiceIds;
        _remoteBundles[seat] = new RemoteBundle { Ctx = ctx, Bundles = bundles, CanSkip = canSkip, ChoiceId = seat < ids.Count ? ids[seat] : 0u };
        Count("events-bundle-remote-capture");
    }

    /// <summary>The wait of FromChooseABundleScreen's remote branch for the prompt captured above (same seat, same choice id).</summary>
    internal sealed class BundleRemoteRouter : IRemoteChoiceRouter
    {
        public string Name => "events:bundle-remote";

        public bool TryRoute(RunContext ctx, Player player, uint choiceId)
        {
            int seat = ctx.SeatOf(player);
            RemoteBundle? b = _remoteBundles[seat];
            if (b == null)
            {
                return false;
            }
            _remoteBundles[seat] = null;
            if (!ReferenceEquals(b.Ctx, ctx) || b.ChoiceId != choiceId)
            {
                return false; // a capture whose call returned without waiting (no bundle, combat ending): not this wait
            }
            EventsStats.RemoteBundleFrames++;
            ctx.Push(new BundleFrame(player, b.Bundles, b.CanSkip, ctx.Mp!.Client(seat), choiceId));
            return true;
        }
    }

    // ---- remote crystal sphere ----

    public static void CrystalPlayRemotePrefix(CrystalSphereMinigame __instance)
    {
        RunContext? ctx = Episode;
        if (ctx?.Mp == null)
        {
            return;
        }
        VirtualClient? client = ctx.Mp.ClientOf(__instance.Owner.NetId);
        if (client == null)
        {
            return; // the local player: the game shows the screen (events-crystal-sphere-screen)
        }
        Count("events-crystal-sphere-remote");
        EventsStats.RemoteCrystalFrames++;
        ctx.Push(new CrystalFrame(__instance, client));
    }

    public static bool TrialPortraitVfxPrefix()
    {
        if (NEventRoom.Instance != null)
        {
            return true;
        }
        Count("events-trial-portrait-vfx");
        return false;
    }

    /// <summary>Stand-in for NChooseABundleSelectionScreen.ShowScreen(bundles, canSkip) inside the real branch.</summary>
    public static NChooseABundleSelectionScreen? BundleShowScreen(IReadOnlyList<IReadOnlyList<CardModel>> bundles, bool canSkip)
    {
        RunContext ctx = Episode ?? throw new InvalidOperationException("The bundle screen was requested outside an episode.");
        if (_shownBundle != null)
        {
            throw new InvalidOperationException("A bundle screen is already pending.");
        }
        Count("events-bundle-screen");
        BundleFrame frame = new BundleFrame(ctx.LocalPlayer, bundles, canSkip); // ShouldSelectLocalCard(player) held: the local player
        ctx.Push(frame);
        _shownBundle = frame;
        return null;
    }

    /// <summary>Stand-in for screen.CardsSelected(): the frame's completion (selected bundle, or empty = skip).</summary>
    public static Task<IEnumerable<IReadOnlyList<CardModel>>> BundleCardsSelected(NChooseABundleSelectionScreen? screen)
    {
        BundleFrame frame = _shownBundle ?? throw new InvalidOperationException("CardsSelected without a pending bundle screen.");
        _shownBundle = null;
        return frame.Completion.Task;
    }

    // =====================================================================================================
    //  FoulPotion at the FakeMerchant (Models.Potions/FoulPotion.cs:51-124)
    // =====================================================================================================

    private static void InstallFoulPotion()
    {
        PatchRegistry.Postfix("events-fake-merchant-foul-potion-usable",
            AccessTools.PropertyGetter(typeof(FoulPotion), nameof(FoulPotion.PassesCustomUsabilityCheck)),
            PatchRegistry.M(typeof(EventPatches), nameof(FoulPotionUsablePostfix)), PatchCategory.Fidelity,
            "FoulPotion.PassesCustomUsabilityCheck out of combat in the FakeMerchant event requires GetFoulPotionMerchantTarget(room).button != null, i.e. an NFakeMerchant " +
            "node whose inventory screen is closed (FoulPotion.cs:59-79, :151-161). Headlessly the event has no node. Postfix, FakeMerchant branch only: when the original said false, " +
            "the room is the FakeMerchant EventRoom, the local event has no node and has not started its fight, the result is true (the host SHOP frame = inventory closed). " +
            "SHARED SITE (rulings B3 / D4): rooms-foul-potion-merchant-usability is a postfix on the same getter for the MerchantRoom branch; the two compose (each only turns false into true under its own room condition).");

        PatchRegistry.Prefix("events-fake-merchant-foul-potion-use",
            AccessTools.Method(typeof(FoulPotion), "OnUse"),
            PatchRegistry.M(typeof(EventPatches), nameof(FoulPotionOnUsePrefix)), PatchCategory.Fidelity,
            "FoulPotion.OnUse out of combat in the FakeMerchant event returns without effect unless LocalMutableEvent.Node is NFakeMerchant (FoulPotion.cs:104-123). " +
            "Prefix, that branch only and only when the node is absent: runs the branch's game calls - FakeMerchant.FoulPotionThrown(potion) for every player, awaited with Task.WhenAll; " +
            "ShowPotionVfx is TestMode-gated presentation (:128). FoulPotionThrown skips its own node part when Node is not NFakeMerchant (FakeMerchant.cs:148-155). Every other branch runs the original.");
    }

    public static void FoulPotionUsablePostfix(FoulPotion __instance, ref bool __result)
    {
        if (__result || CombatManager.Instance.IsInProgress)
        {
            return;
        }
        Player? owner = __instance.Owner;
        if (owner == null || owner.RunState.CurrentRoom is not EventRoom room || room.CanonicalEvent is not FakeMerchant)
        {
            return;
        }
        if (room.LocalMutableEvent is FakeMerchant local && local.Node == null && !local.StartedFight)
        {
            Count("events-fake-merchant-foul-potion-usable");
            __result = true;
        }
    }

    public static bool FoulPotionOnUsePrefix(FoulPotion __instance, ref Task __result)
    {
        if (CombatManager.Instance.IsInProgress)
        {
            return true;
        }
        Player owner = __instance.Owner;
        AbstractRoom? current = owner.RunState.CurrentRoom;
        if (current is MerchantRoom || current is not EventRoom room || room.CanonicalEvent is not FakeMerchant)
        {
            return true;
        }
        EventModel? local = room.LocalMutableEvent;
        if (local == null || local.Node != null)
        {
            return true; // a real NFakeMerchant node (never headless): the original branch is correct
        }
        Count("events-fake-merchant-foul-potion-use");
        __result = ThrowAtFakeMerchant(__instance);
        return false;
    }

    /// <summary>The FakeMerchant branch of FoulPotion.OnUse (FoulPotion.cs:115-122) without the node-only presentation.</summary>
    private static async Task ThrowAtFakeMerchant(FoulPotion potion)
    {
        List<Task> list = new List<Task>();
        foreach (Player player in potion.Owner.RunState.Players)
        {
            FakeMerchant fakeMerchant = (FakeMerchant)RunManager.Instance.EventSynchronizer.GetEventForPlayer(player);
            list.Add(fakeMerchant.FoulPotionThrown(potion));
        }
        await Task.WhenAll(list);
    }

    // =====================================================================================================
    //  Presentation null dereferences in event models (headless-only failures)
    // =====================================================================================================

    private static void InstallPresentation()
    {
        MethodInfo setPortrait = AccessTools.Method(typeof(NEventRoom), nameof(NEventRoom.SetPortrait), new[] { typeof(Texture2D) });
        MethodInfo getLayout = AccessTools.PropertyGetter(typeof(NEventRoom), nameof(NEventRoom.Layout));
        MethodInfo removeNodes = AccessTools.Method(typeof(NEventLayout), nameof(NEventLayout.RemoveNodesOnPortrait));
        MethodInfo shakeTrauma = AccessTools.Method(typeof(NGame), nameof(NGame.ScreenShakeTrauma), new[] { typeof(ShakeStrength) });
        MethodInfo rumble = AccessTools.Method(typeof(NGame), nameof(NGame.ScreenRumble), new[] { typeof(ShakeStrength), typeof(ShakeDuration), typeof(RumbleStyle) });
        MethodInfo play = AccessTools.Method(typeof(NDebugAudioManager), nameof(NDebugAudioManager.Play), new[] { typeof(string), typeof(float), typeof(PitchVariance) });
        MethodInfo stop = AccessTools.Method(typeof(NDebugAudioManager), nameof(NDebugAudioManager.Stop), new[] { typeof(int), typeof(float) });

        MethodInfo sSetPortrait = PatchRegistry.M(typeof(PresentationStubs), nameof(PresentationStubs.SetPortrait));
        MethodInfo sGetLayout = PatchRegistry.M(typeof(PresentationStubs), nameof(PresentationStubs.Layout));
        MethodInfo sRemoveNodes = PatchRegistry.M(typeof(PresentationStubs), nameof(PresentationStubs.RemoveNodesOnPortrait));
        MethodInfo sShake = PatchRegistry.M(typeof(PresentationStubs), nameof(PresentationStubs.ScreenShakeTrauma));
        MethodInfo sRumble = PatchRegistry.M(typeof(PresentationStubs), nameof(PresentationStubs.ScreenRumble));
        MethodInfo sPlay = PatchRegistry.M(typeof(PresentationStubs), nameof(PresentationStubs.Play));
        MethodInfo sStop = PatchRegistry.M(typeof(PresentationStubs), nameof(PresentationStubs.Stop));

        const string NullSafe = " The instruction is swapped for a static stand-in that makes the same call when the receiver exists and does nothing (counted) when it is null, " +
                                "i.e. the C# `?.` the neighbouring lines already use; argument evaluation and every following game call are untouched.";

        CallSwap.Install("events-waterfall-enter-portrait", AccessTools.Method(typeof(WaterfallOfObviousSecrets), "EnterWaterfall"), PatchCategory.Plumbing,
            "WaterfallOfObviousSecrets.EnterWaterfall:53 calls NEventRoom.Instance.SetPortrait(...) (portrait texture only) before SetEventState of the ENTER page (:55)." + NullSafe,
            new CallSwap.Spec(setPortrait, sSetPortrait, 1));

        CallSwap.Install("events-dense-vegetation-rest-presentation", AccessTools.AsyncMoveNext(AccessTools.Method(typeof(DenseVegetation), "Rest")), PatchCategory.Plumbing,
            "DenseVegetation.Rest:93-97 (local player only) plays / stops two sounds and a screen rumble between MimicRestSiteHeal (:90) and SetEventState of the REST page (:99); " +
            "the Cmd.CustomScaledWait between them is already a no-op in non-interactive mode." + NullSafe,
            new CallSwap.Spec(play, sPlay, 2), new CallSwap.Spec(stop, sStop, 1), new CallSwap.Spec(rumble, sRumble, 1));

        CallSwap.Install("events-amalgamator-combine-strikes-shake", AccessTools.AsyncMoveNext(AccessTools.Method(typeof(Amalgamator), "CombineStrikes")), PatchCategory.Plumbing,
            "Amalgamator.CombineStrikes:46,:49 call NGame.Instance.ScreenShakeTrauma (camera shake) between RemoveFromDeck (:44) and creating / adding UltimateStrike (:50-52); " +
            "the Task.Delay(300) between them is on the virtual clock. SHARED SITE (ruling B3): Core's virtual-clock-amalgamator-combinestrikes is a transpiler on the same method." + NullSafe,
            new CallSwap.Spec(shakeTrauma, sShake, 2));

        CallSwap.Install("events-amalgamator-combine-defends-shake", AccessTools.AsyncMoveNext(AccessTools.Method(typeof(Amalgamator), "CombineDefends")), PatchCategory.Plumbing,
            "Amalgamator.CombineDefends:60,:63 call NGame.Instance.ScreenShakeTrauma between RemoveFromDeck (:58) and creating / adding UltimateDefend (:64-66). " +
            "SHARED SITE (ruling B3): Core's virtual-clock-amalgamator-combinedefends is a transpiler on the same method." + NullSafe,
            new CallSwap.Spec(shakeTrauma, sShake, 2));

        CallSwap.Install("events-jungle-maze-safety-audio", AccessTools.AsyncMoveNext(AccessTools.Method(typeof(JungleMazeAdventure), "SafetyInNumbers")), PatchCategory.Plumbing,
            "JungleMazeAdventure.SafetyInNumbers:94 plays a sound through NDebugAudioManager.Instance before GainGold (:96) and SetEventFinished (:97)." + NullSafe,
            new CallSwap.Spec(play, sPlay, 1));

        CallSwap.Install("events-punch-off-nab-shake", AccessTools.AsyncMoveNext(AccessTools.Method(typeof(PunchOff), "Nab")), PatchCategory.Plumbing,
            "PunchOff.Nab:106 calls NGame.Instance.ScreenShakeTrauma between AddCurseToDeck<Injury> (:105) and the RelicReward offer (:109-112) / SetEventFinished (:113)." + NullSafe,
            new CallSwap.Spec(shakeTrauma, sShake, 1));

        CallSwap.Install("events-trial-accept-portrait", AccessTools.Method(typeof(Trial), "Accept"), PatchCategory.Plumbing,
            "Trial.Accept:68 NEventRoom.Instance.Layout.RemoveNodesOnPortrait() and :108 NEventRoom.Instance.SetPortrait(...) (local player only, portrait only) around the " +
            "Rng.NextInt(3) case draw (:73) and SetEventState of the case page (:112)." + NullSafe,
            new CallSwap.Spec(getLayout, sGetLayout, 1), new CallSwap.Spec(removeNodes, sRemoveNodes, 1), new CallSwap.Spec(setPortrait, sSetPortrait, 1));
    }

    /// <summary>Static stand-ins for instance calls on UI nodes: the original call when the receiver exists, otherwise nothing (counted).</summary>
    public static class PresentationStubs
    {
        public static void SetPortrait(NEventRoom? room, Texture2D portrait)
        {
            if (room == null)
            {
                Count("stub:NEventRoom.SetPortrait");
                return;
            }
            room.SetPortrait(portrait);
        }

        public static NEventLayout? Layout(NEventRoom? room)
        {
            if (room == null)
            {
                Count("stub:NEventRoom.Layout");
                return null;
            }
            return room.Layout;
        }

        public static void RemoveNodesOnPortrait(NEventLayout? layout)
        {
            if (layout == null)
            {
                Count("stub:NEventLayout.RemoveNodesOnPortrait");
                return;
            }
            layout.RemoveNodesOnPortrait();
        }

        public static void ScreenShakeTrauma(NGame? game, ShakeStrength strength)
        {
            if (game == null)
            {
                Count("stub:NGame.ScreenShakeTrauma");
                return;
            }
            game.ScreenShakeTrauma(strength);
        }

        public static void ScreenRumble(NGame? game, ShakeStrength strength, ShakeDuration duration, RumbleStyle style)
        {
            if (game == null)
            {
                Count("stub:NGame.ScreenRumble");
                return;
            }
            game.ScreenRumble(strength, duration, style);
        }

        /// <summary>The returned sound handle is only ever passed back to Stop (DenseVegetation.cs:93-95).</summary>
        public static int Play(NDebugAudioManager? audio, string streamName, float volume, PitchVariance variance)
        {
            if (audio == null)
            {
                Count("stub:NDebugAudioManager.Play");
                return -1;
            }
            return audio.Play(streamName, volume, variance);
        }

        public static void Stop(NDebugAudioManager? audio, int id, float fadeTime)
        {
            if (audio == null)
            {
                Count("stub:NDebugAudioManager.Stop");
                return;
            }
            audio.Stop(id, fadeTime);
        }
    }
}

/// <summary>
/// Fail-closed call-site swap: in one method, replace every call / callvirt of a given method by a call of a static
/// stand-in with the receiver as first parameter (same remaining parameters, same return type). Each spec must hit
/// exactly its expected number of instructions, otherwise installation throws (DECISIONS-H1 E).
/// </summary>
internal static class CallSwap
{
    internal sealed record Spec(MethodInfo From, MethodInfo To, int Expected);

    private static readonly Dictionary<MethodBase, (string Id, Spec[] Specs)> _targets = new Dictionary<MethodBase, (string, Spec[])>();

    public static void Install(string id, MethodBase? target, PatchCategory category, string reason, params Spec[] specs)
    {
        if (target == null)
        {
            throw new MissingMethodException("Patch target for '" + id + "' was not found; the game assembly changed.");
        }
        foreach (Spec s in specs)
        {
            if (s.From == null || s.To == null)
            {
                throw new MissingMethodException("Call swap '" + id + "': a source or stand-in method was not found.");
            }
        }
        if (_targets.ContainsKey(target))
        {
            throw new InvalidOperationException("Call swap '" + id + "': " + target.Name + " already has a swap.");
        }
        _targets[target] = (id, specs);
        StringBuilderLite sb = new StringBuilderLite();
        foreach (Spec s in specs)
        {
            sb.Add($"exactly {s.Expected} call(s) of {s.From.DeclaringType?.Name}.{s.From.Name} -> {s.To.DeclaringType?.Name}.{s.To.Name}");
        }
        try
        {
            PatchRegistry.Transpiler(id, target, PatchRegistry.M(typeof(CallSwap), nameof(Transpile)), category, reason + " [transpiler: " + sb + "]");
        }
        catch
        {
            _targets.Remove(target);
            throw;
        }
    }

    public static IEnumerable<CodeInstruction> Transpile(IEnumerable<CodeInstruction> instructions, MethodBase original)
    {
        List<CodeInstruction> code = new List<CodeInstruction>(instructions);
        if (!_targets.TryGetValue(original, out (string Id, Spec[] Specs) t))
        {
            throw new InvalidOperationException("CallSwap: no spec for " + original.DeclaringType?.FullName + "::" + original.Name);
        }
        foreach (Spec s in t.Specs)
        {
            int hits = 0;
            foreach (CodeInstruction ci in code)
            {
                if ((ci.opcode == OpCodes.Call || ci.opcode == OpCodes.Callvirt) && ci.operand is MethodInfo m && m == s.From)
                {
                    ci.opcode = OpCodes.Call;
                    ci.operand = s.To;
                    hits++;
                }
            }
            if (hits != s.Expected)
            {
                throw new InvalidOperationException($"Call swap '{t.Id}': expected exactly {s.Expected} call(s) of {s.From.DeclaringType?.Name}.{s.From.Name} in {original.DeclaringType?.FullName}::{original.Name}, found {hits}; the game assembly changed.");
            }
        }
        return code;
    }

    private sealed class StringBuilderLite
    {
        private readonly List<string> _parts = new List<string>();

        public void Add(string s) => _parts.Add(s);

        public override string ToString() => string.Join("; ", _parts);
    }
}
