using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game.Sync;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;

namespace SpireHost.Driver.Rooms;

// =========================================================================================================
//  SHOP (driver-spec-rooms.md 5; testmode-audit-nodes.md U-SH-1..4)
//
//  Inventory: MerchantSynchronizer.GetLocalInventory() (MerchantRoom.EnterInternal -> OnMerchantEntered, before
//  Hook.AfterRoomEntered, so LordsParasol sees it). Buying = the UI's path NMerchantSlot.OnSelected
//  (:275-283) -> MerchantSynchronizer.PurchaseLocalEntry (purchase lock TryBeginPurchase, :76-103) ->
//  MerchantEntry.OnTryPurchaseUnsynchronized (:71-102). Card removal opens a CARD_SELECT (FromDeckForRemoval,
//  cancelable) answered by the run's card selector. Travel is enabled on entry (NMerchantRoom.cs:110); the proceed
//  button calls RunManager.Proceed() (:179-185). The OnMerchantExited bookkeeping of MerchantRoom.Exit is gated by
//  TestMode (MerchantRoom.cs:68-76) - that gate belongs to the fidelity team; leaving here runs MerchantRoom.Exit
//  exactly as the real game does once the map move exits the room.
//
//  MULTI-SEAT (phase H5). OnMerchantEntered builds one inventory per player (MerchantSynchronizer.cs:38-54); each
//  player sees and buys from its own (GetInventoryFor). A remote seat's purchase is its client's
//  MerchantEntryPurchasedMessage{Location, id} (PurchaseLocalEntry :91-96 -> host :105-117, which runs the purchase
//  without the lock and drops the result; the result is observed through mp-remote-purchase). The card removal of a
//  remote seat asks through the remote prompt path (CARD_SELECT frame of that seat). LEAVE of a remote seat opens
//  its own map and sends nothing. LordsParasol runs on the host for its owner whoever that is; the SHOP screen of the
//  OWNER waits for it.
// =========================================================================================================

internal sealed class ShopState
{
    /// <summary>Index = seat: the LordsParasol task of that seat that the handler is waiting for (or has waited for).</summary>
    public readonly Task?[] Parasol = new Task?[DriverApi.MaxSeats];
}

internal sealed class ShopHandler : IRoomHandler
{
    private readonly Decision _decision = new Decision();
    private readonly ShopContext _context = new ShopContext();

    public string Name => RoomsNames.ShopHandler;

    public HandlerStep Step(RunContext ctx, int seat)
    {
        if (ctx.CurrentRoom is not MerchantRoom room)
        {
            return HandlerStep.Unrecognised("shop handler outside a MerchantRoom");
        }
        Player player = ctx.Seats[seat];
        ShopState state = ctx.RoomData<ShopState>(room);

        // LordsParasol.AfterRoomEntered starts PurchaseEverything with RunSafely and keeps the task
        // (LordsParasol.cs:24-33). The real UI is blocked while it runs (map / deck / inventory input disabled,
        // :57-71, :121-142); DECISIONS-H1 E: run the original and treat LastPurchaseEverythingTask as blocking.
        IReadOnlyList<RelicModel> relics = player.Relics;
        for (int i = 0; i < relics.Count; i++)
        {
            if (relics[i] is LordsParasol parasol && parasol.LastPurchaseEverythingTask is Task t && !t.IsCompleted && !ReferenceEquals(t, state.Parasol[seat]))
            {
                state.Parasol[seat] = t;
                RoomsProbe.ParasolWaits++;
                ctx.Track(t, "lords parasol purchase everything", seat);
                return HandlerStep.Progressed("waiting for LordsParasol");
            }
        }

        MerchantInventory inventory = ctx.RunManager.MerchantSynchronizer.GetInventoryFor(player);
        if (inventory._purchaseInFlight)
        {
            // Nothing the host started is running (the core only asks when no blocking operation is pending).
            return HandlerStep.Unrecognised("a merchant purchase is in flight that the host did not start");
        }
        RunState run = ctx.Run;
        _context.Inventory = inventory;
        _context.FakeMerchant = false;
        Decision d = _decision.Reset(DecisionKind.Shop, this, actingSeat: seat, context: _context);
        List<MerchantEntry> entries = inventory.AllEntries;
        for (int i = 0; i < entries.Count; i++)
        {
            MerchantEntry e = entries[i];
            if (IsPurchasable(e, player, run))
            {
                d.Add(Verb.ChooseOption, seat, GameRef.OfShopEntry(e, i));
            }
        }
        d.Add(Verb.Proceed, seat);
        return HandlerStep.Ask(d);
    }

    /// <summary>
    /// OnTryPurchaseUnsynchronized refuses out-of-stock (FailureOutOfStock) and unaffordable (FailureGold) entries
    /// (MerchantEntry.cs:77-86). The UI does not pre-filter, but a refused click changes nothing, so it is not
    /// offered. The same holds for the entry-specific refusals: a potion fails with a full belt (FailureSpace) or when
    /// Hook.ShouldProcurePotion forbids it (FailureForbidden), without spending gold (MerchantPotionEntry.cs:68-80); a
    /// card removal without any removable card returns false before spending (FromDeckGeneric returns nothing,
    /// MerchantCardRemovalEntry.cs:43-53). Card purchases fail only when Hook.ShouldAddToDeck refuses, which no model
    /// overrides in v0.113 (AbstractModel.cs:942).
    /// </summary>
    private static bool IsPurchasable(MerchantEntry e, Player player, RunState run)
    {
        if (!e.IsStocked || !e.EnoughGold)
        {
            return false;
        }
        switch (e)
        {
            case MerchantPotionEntry pe:
                return pe.Model != null && player.HasOpenPotionSlots && Hook.ShouldProcurePotion(run, player.Creature.CombatState, pe.Model, player);
            case MerchantCardRemovalEntry:
                return RewardFrame.AnyDeckCard(player, upgradable: false);
            default:
                return true;
        }
    }

    public void Apply(RunContext ctx, Decision decision, int actionIndex)
    {
        LegalAction a = decision.Actions[actionIndex];
        int seat = a.Seat;
        VirtualClient? client = ctx.Mp != null && ctx.IsRemoteSeat(seat) ? ctx.Mp.Client(seat) : null;
        if (a.Verb == Verb.Proceed)
        {
            RoomsProbe.ShopLeaves++;
            if (client != null)
            {
                ctx.OpenMapFor(seat, "leave shop (remote)");
            }
            else
            {
                RoomsFlow.Proceed(ctx, "leave shop");
            }
            return;
        }
        MerchantEntry entry = a.Source.ShopEntry;
        RoomsProbe.ShopPurchases++;
        if (client != null)
        {
            if (!entry.Id.HasValue)
            {
                ctx.Fail(AbortReason.HandlerFault, "shop_entry_without_id", "Merchant entry " + entry.GetType().Name + " of seat " + seat + " has no id.");
            }
            // NMerchantSlot.OnSelected on the client -> PurchaseLocalEntry: this message, then the same purchase in its own instance.
            RoomsProbe.RemoteShopPurchases++;
            Task purchase = client.SendTracked(new MerchantEntryPurchasedMessage { Location = client.Location, id = entry.Id!.Value });
            ctx.Track(Observe(ctx, (Task<bool>)purchase, entry), "seat " + seat + " purchases " + entry.GetType().Name, seat);
        }
        else
        {
            // The UI's call (NMerchantSlot.OnSelected): takes the purchase lock that LordsParasol also uses.
            ctx.Track(Observe(ctx, ctx.RunManager.MerchantSynchronizer.PurchaseLocalEntry(entry), entry), "purchase " + entry.GetType().Name, seat);
        }
    }

    private static async Task Observe(RunContext ctx, Task<bool> purchase, MerchantEntry entry)
    {
        bool ok = await purchase;
        RoomsProbe.Bump(RoomsProbe.ShopEntryTypesBought, entry.GetType().Name + (ok ? "" : ":cancelled"));
        if (!ok)
        {
            if (entry is MerchantCardRemovalEntry)
            {
                RoomsProbe.ShopPurchaseRejectedLegit++; // the removal's card selection was cancelled
                return;
            }
            ctx.Fail(AbortReason.ActionFault, "purchase_rejected:" + entry.GetType().Name, "The purchase of " + entry.GetType().Name + " returned false (the host offered an entry the game refused).");
        }
    }
}
