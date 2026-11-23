using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace SpireHost.Driver;

/// <summary>
/// Building blocks for isolated handler tests (DRIVER-API.md section 9). Everything here goes through the
/// game's own commands and debug facilities (RelicCmd.Obtain, CardPileCmd.Add, PotionCmd.TryToProcure,
/// RunManager.EnterRoomDebug - the dev console's "fight" / "event" / "room" path), on the engine thread,
/// inside a Scenario.Setup task that the driver core tracks. TEST USE ONLY: never in training or evaluation.
/// </summary>
public static class ScenarioTools
{
    public static T Find<T>(IEnumerable<T> all, string entry) where T : AbstractModel
    {
        foreach (T m in all)
        {
            if (m.Id.Entry.Equals(entry, StringComparison.OrdinalIgnoreCase))
            {
                return m;
            }
        }
        throw new ArgumentException("Unknown " + typeof(T).Name + " id: " + entry);
    }

    public static EncounterModel Encounter(string entry) => Find(ModelDb.AllEncounters, entry);

    public static EventModel Event(string entry) => Find(ModelDb.AllEvents, entry);

    public static RelicModel Relic(string entry) => Find(ModelDb.AllRelics, entry);

    public static CardModel Card(string entry) => Find(ModelDb.AllCards, entry);

    public static PotionModel Potion(string entry) => Find(ModelDb.AllPotions, entry);

    /// <summary>Obtains a relic through RelicCmd.Obtain (AfterObtained runs; it may raise selection / reward frames).</summary>
    public static async Task GiveRelic(RunContext ctx, string relicEntry, int seat = 0)
    {
        await RelicCmd.Obtain(Relic(relicEntry).ToMutable(), ctx.Seats[seat]);
    }

    /// <summary>Adds a card to the deck through CardPileCmd.Add (deck hooks run).</summary>
    public static async Task GiveCard(RunContext ctx, string cardEntry, int upgrades = 0, int seat = 0)
    {
        Player player = ctx.Seats[seat];
        CardModel card = ctx.Run.CreateCard(Card(cardEntry), player);
        for (int i = 0; i < upgrades && card.IsUpgradable; i++)
        {
            card.UpgradeInternal();
        }
        await CardPileCmd.Add(card, PileType.Deck);
    }

    /// <summary>Procures a potion through PotionCmd.TryToProcure (fails silently when the belt is full, like the game).</summary>
    public static async Task GivePotion(RunContext ctx, string potionEntry, int seat = 0)
    {
        await PotionCmd.TryToProcure(Potion(potionEntry).ToMutable(), ctx.Seats[seat]);
    }

    public static void SetGold(RunContext ctx, int gold, int seat = 0)
    {
        ctx.Seats[seat].Gold = gold;
    }

    public static void SetHp(RunContext ctx, int currentHp, int? maxHp = null, int seat = 0)
    {
        Player p = ctx.Seats[seat];
        if (maxHp.HasValue)
        {
            p.Creature.SetMaxHpInternal(maxHp.Value);
        }
        p.Creature.SetCurrentHpInternal(currentHp);
    }

    /// <summary>Leaves the current room and enters a combat with this encounter (dev console "fight": RunManager.EnterRoomDebug).</summary>
    public static Task JumpToEncounter(RunContext ctx, string encounterEntry)
    {
        EncounterModel encounter = Encounter(encounterEntry).ToMutable();
        return ctx.RunManager.EnterRoomDebug(encounter.RoomType, MapPointType.Unassigned, encounter, showTransition: false);
    }

    /// <summary>Leaves the current room and enters this event (canonical model; the game creates the per-player mutable copy).</summary>
    public static Task JumpToEvent(RunContext ctx, string eventEntry)
    {
        return ctx.RunManager.EnterRoomDebug(RoomType.Event, MapPointType.Unassigned, Event(eventEntry), showTransition: false);
    }

    /// <summary>Leaves the current room and enters a fresh room of this type (Shop, RestSite, Treasure, Monster, Elite, Boss, Event).</summary>
    public static Task JumpToRoom(RunContext ctx, RoomType roomType)
    {
        return ctx.RunManager.EnterRoomDebug(roomType, MapPointType.Unassigned, null, showTransition: false);
    }

    /// <summary>
    /// Leaves the current room and enters a canonical event model (events R4). Works for the Ancients and TheArchitect,
    /// which are not in ModelDb.AllEvents (<see cref="Event"/> only searches that list).
    /// </summary>
    public static Task JumpToEvent(RunContext ctx, EventModel canonical)
    {
        return ctx.RunManager.EnterRoomDebug(RoomType.Event, MapPointType.Unassigned, canonical, showTransition: false);
    }

    /// <summary>
    /// RunManager.EnterAct(actIndex): the act change the game performs after the last boss of an act (rooms R6 / events R4).
    /// 14 events read CurrentActIndex in IsAllowed; the proceed-tree branches depend on the act.
    /// </summary>
    public static async Task EnterAct(RunContext ctx, int actIndex, bool doTransition = true)
    {
        if (ctx.Run.CurrentActIndex != actIndex)
        {
            await ctx.RunManager.EnterAct(actIndex, doTransition);
        }
    }

    /// <summary>
    /// Travels to the boss point of the act the way the game does once a map vote resolved (RunManager.EnterMapCoord):
    /// unlike EnterRoomDebug it sets RunState.CurrentMapPoint, which the terminal proceed tree reads (rooms R6).
    /// </summary>
    public static Task EnterBoss(RunContext ctx) => ctx.RunManager.EnterMapCoord(ctx.Run.Map.BossMapPoint.coord);

    /// <summary>
    /// TEST-ONLY state construction (events R4): appends map-point history entries until RunState.TotalFloor reaches
    /// <paramref name="totalFloor"/> (PunchOff / SlipperyBridge read it in IsAllowed).
    /// </summary>
    public static void SetVisitedFloors(RunContext ctx, int totalFloor)
    {
        while (ctx.Run.TotalFloor < totalFloor)
        {
            ctx.Run.AppendToMapPointHistory(MapPointType.Monster, RoomType.Monster, null);
        }
    }

    // ------------------------------------------------------------------------------------------------
    //  Mid-combat tools (combat C8): each one is the dev console path of the game (DevConsole.ConsoleCommands:
    //  "card", "energy", "stars", "power", "afflict"), awaited inside Scenario.Setup once a combat is in progress.
    // ------------------------------------------------------------------------------------------------

    public static CombatState CombatState(RunContext ctx) => ctx.CombatManager.DebugOnlyGetState() ?? throw new InvalidOperationException("No combat in progress.");

    /// <summary>Dev console "card &lt;id&gt; &lt;pile&gt;" (CardConsoleCmd.cs): CombatState.CreateCard + CardPileCmd.Add.</summary>
    public static async Task<CardModel> AddCard(RunContext ctx, string cardEntry, PileType pile = PileType.Hand, int upgrades = 0, int seat = 0)
    {
        Player player = ctx.Seats[seat];
        CardModel card = CombatState(ctx).CreateCard(Card(cardEntry), player);
        for (int i = 0; i < upgrades && card.IsUpgradable; i++)
        {
            card.UpgradeInternal();
        }
        await CardPileCmd.Add(card, pile);
        return card;
    }

    /// <summary>Dev console "energy" / "stars" (PlayerCmd.SetEnergy / SetStars).</summary>
    public static async Task SetResources(RunContext ctx, int energy, int stars, int seat = 0)
    {
        Player player = ctx.Seats[seat];
        await PlayerCmd.SetEnergy(energy, player);
        await PlayerCmd.SetStars(stars, player);
    }

    /// <summary>Dev console "power &lt;id&gt; &lt;amount&gt; &lt;target&gt;" (ApplyPowerConsoleCmd.cs:64-66).</summary>
    public static async Task ApplyPower(RunContext ctx, string powerEntry, int amount, Creature target)
    {
        PowerModel? canonical = null;
        foreach (PowerModel p in ModelDb.AllPowers)
        {
            if (p.Id.Entry == powerEntry)
            {
                canonical = p;
                break;
            }
        }
        if (canonical == null)
        {
            throw new ArgumentException("Unknown power " + powerEntry);
        }
        await PowerCmd.Apply(new BlockingPlayerChoiceContext(), canonical.ToMutable(), target, amount, null, null);
    }

    /// <summary>Dev console "afflict &lt;id&gt; &lt;amount&gt; &lt;hand index&gt;" (AfflictConsoleCmd.cs).</summary>
    public static async Task Afflict(RunContext ctx, string afflictionEntry, int amount, CardModel card)
    {
        ModelId id = new ModelId(ModelId.SlugifyCategory<AfflictionModel>(), afflictionEntry);
        AfflictionModel affliction = ModelDb.GetById<AfflictionModel>(id).ToMutable();
        await CardCmd.Afflict(affliction, card, amount);
    }

    /// <summary>Stop condition: the room that the scenario jumped into has been left for the map (or the run ended).</summary>
    public static bool RoomFinished(RunContext ctx) => ctx.MapOpen;
}
