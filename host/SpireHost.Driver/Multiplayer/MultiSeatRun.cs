using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;
using SpireHost.Sim;

namespace SpireHost.Driver;

/// <summary>
/// The multi-seat part of one episode (phase H5; null on <see cref="RunContext.Mp"/> for a single-seat run): the
/// Host-type net service of the run and one <see cref="VirtualClient"/> per remote seat.
///
/// Seat roles. Seat 0 is the LOCAL player of the host instance (RunManager.Launch sets LocalContext.NetId to the
/// service's NetId, RunManager.cs:596): every path the single-seat host uses stays as it is for seat 0. Seats 1..N-1
/// are REMOTE players: the host's simulation computes everything for them, but whatever a real client decides and
/// sends is sent by their virtual client, at a point a real client could send it.
///
/// Combat-state sync (decision D1 of notes\host-h5-progress.md). SetUpTest is called with disableCombatStateSync:
/// false, as RunManager.SetUpNewMultiplayer leaves it (RunManager.cs:275-293): at every map-point entry (and event
/// combat, debug jump) the host broadcasts its own player (CombatStateSynchronizer.StartSync, :100-122), waits for the
/// player data of every lobby member (:124-135, :190-196) and rebuilds every REMOTE player from what that client sent
/// (Player.SyncWithSerializedPlayer, Player.cs:325-365: deck, relics and potions are re-created). A lockstep client is
/// at the same point of its own StartSync with the same state, so the virtual client serialises its seat's Player at
/// the moment of the host's broadcast and sends it from the pump. <see cref="RemoteSeatPatches"/> checks that the
/// rebuild changes nothing a run signature sees (a difference would be a desync between host and client in the real game).
/// </summary>
public sealed class MultiSeatRun
{
    private readonly RunContext _ctx;
    private readonly Action _onSyncBroadcast;

    internal MultiSeatRun(RunContext ctx, HeadlessHostNetService net)
    {
        _ctx = ctx;
        Net = net;
        IReadOnlyList<Player> players = ctx.Run.Players;
        Clients = new VirtualClient?[players.Count];
        for (int seat = 1; seat < players.Count; seat++)
        {
            Clients[seat] = new VirtualClient(net, ctx.RunManager, players[seat], seat);
        }
        _onSyncBroadcast = OnHostSyncBroadcast;
        net.SyncPlayerDataBroadcast += _onSyncBroadcast;
    }

    public HeadlessHostNetService Net { get; }

    /// <summary>
    /// The net id of a seat: 1 for seat 0 - the host player of an ENet session (Multiplayer.Transport.ENet/ENetHost: NetId
    /// => 1) - and 1000 + k for seat k, the id the game gives the k-th client that joins with --clientId (NJoinFriendScreen
    /// .FastMpJoin, Nodes.Screens.MainMenu/NJoinFriendScreen.cs:158-167). These are the ids of the real multi-client
    /// recordings (notes\oracle-mp.md section 4), so that the host's action owners and combat checksums can be compared
    /// with a recording directly. No rule depends on the values; only their order matters (it is the seat order).
    /// </summary>
    public static ulong NetIdOfSeat(int seat) => seat == 0 ? 1UL : 1000UL + (ulong)seat;

    /// <summary>Index = seat; [0] is null (the local seat has no client).</summary>
    public VirtualClient?[] Clients { get; }

    public VirtualClient Client(int seat)
    {
        return Clients[seat] ?? throw new InvalidOperationException("Seat " + seat + " is the local seat: it has no virtual client.");
    }

    /// <summary>The client of this player, or null for the local player.</summary>
    public VirtualClient? ClientOf(ulong netId)
    {
        VirtualClient?[] clients = Clients;
        for (int i = 1; i < clients.Length; i++)
        {
            if (clients[i]!.NetId == netId)
            {
                return clients[i];
            }
        }
        return null;
    }

    internal void Detach()
    {
        Net.SyncPlayerDataBroadcast -= _onSyncBroadcast;
    }

    /// <summary>
    /// CombatStateSynchronizer.StartSync on the host just broadcast the host player's data. Each client's own StartSync
    /// runs at the same point of the same simulation: its message carries its player as it is NOW. The delivery is
    /// posted (a real packet arrives later, never inside the host's own call), the content is taken here.
    /// </summary>
    private void OnHostSyncBroadcast()
    {
        MultiSeatStats.SyncRounds++;
        VirtualClient?[] clients = Clients;
        for (int seat = 1; seat < clients.Length; seat++)
        {
            VirtualClient client = clients[seat]!;
            SerializablePlayer data = client.Player.ToSerializable();
            _ctx.Post(() =>
            {
                client.SendPlayerData(data);
                return Task.CompletedTask;
            }, "seat " + seat + " sends its player data (combat-state sync)", blocking: false);
        }
    }

    /// <summary>
    /// What a run signature sees of one player (RunSignature.Compute, the per-player part): gold, HP, energy, the deck in
    /// order, relics in order, potion slots, the four player RNG streams. HOST ONLY (hidden state).
    /// </summary>
    public static ulong SeatSignature(Player pl)
    {
        SigHasher h = SigHasher.Create();
        h.Add(pl.NetId);
        h.Add(pl.Gold);
        h.Add(pl.Creature.CurrentHp);
        h.Add(pl.Creature.MaxHp);
        h.Add(pl.MaxEnergy);
        h.Add(pl.MaxPotionCount);
        IReadOnlyList<CardModel> deck = pl.Deck.Cards;
        h.Add(deck.Count);
        for (int c = 0; c < deck.Count; c++)
        {
            CardModel card = deck[c];
            h.Add(card.Id);
            h.Add(card.CurrentUpgradeLevel);
            h.Add(card.Enchantment?.Id);
            h.Add(card.Enchantment?.Amount ?? 0);
            h.Add(card.Affliction?.Id);
        }
        IReadOnlyList<RelicModel> relics = pl.Relics;
        h.Add(relics.Count);
        for (int r = 0; r < relics.Count; r++)
        {
            h.Add(relics[r].Id);
        }
        IReadOnlyList<PotionModel?> potions = pl.PotionSlots;
        h.Add(potions.Count);
        for (int s = 0; s < potions.Count; s++)
        {
            h.Add(potions[s]?.Id);
        }
        AddRng(ref h, pl.PlayerRng.Rewards);
        AddRng(ref h, pl.PlayerRng.Shops);
        AddRng(ref h, pl.PlayerRng.Transformations);
        AddRng(ref h, pl.PlayerRng.NonCombatGeneration);
        return h.Finish();
    }

    private static void AddRng(ref SigHasher h, Rng rng)
    {
        SerializableRng s = rng.ToSerializable();
        h.Add(s.counter);
        h.Add(s.state0);
        h.Add(s.state1);
        h.Add(s.state2);
        h.Add(s.state3);
    }
}

/// <summary>Process-wide counters of the multi-seat machinery (module counters "mp.*"; all 0 in single-seat runs).</summary>
public static class MultiSeatStats
{
    public static long Runs;
    public static long SyncRounds;
    public static long SyncPlayersRebuilt;

    /// <summary>Player.SyncWithSerializedPlayer changed what a run signature sees of the player. Must stay 0 (aborts the episode).</summary>
    public static long SyncChangedState;

    public static long RemoteWaits;
    public static long RemoteWaitsUnrouted;
    public static long HookEnqueuesPosted;
    public static long ResumesPosted;
    public static long ReadyRelays;
    public static long RunLobbyResets;

    /// <summary>Messages of virtual clients handed to the host's net service (HeadlessHostNetService.Receive).</summary>
    public static long MessagesReceived;

    /// <summary>Of those, the ones that went through the game's packet code first (verification mode --wire), and their bytes.</summary>
    public static long WireRoundTrips;
    public static long WireBytes;
}

internal sealed class MultiSeatCounters : IModuleCounters
{
    public string Module => "mp";

    public void Read(IDictionary<string, long> c)
    {
        c["runs"] = MultiSeatStats.Runs;
        c["sync.rounds"] = MultiSeatStats.SyncRounds;
        c["sync.players_rebuilt"] = MultiSeatStats.SyncPlayersRebuilt;
        c["violation.sync_changed_state"] = MultiSeatStats.SyncChangedState;
        c["remote.waits"] = MultiSeatStats.RemoteWaits;
        c["violation.remote_wait_unrouted"] = MultiSeatStats.RemoteWaitsUnrouted;
        c["remote.hook_enqueues"] = MultiSeatStats.HookEnqueuesPosted;
        c["remote.resumes"] = MultiSeatStats.ResumesPosted;
        c["remote.ready_relays"] = MultiSeatStats.ReadyRelays;
        c["remote.tasks_captured"] = RemoteDelivery.Captured;
        c["remote.map_votes"] = MapHandler.RemoteVotesSent;
        c["team_votes.map"] = MapHandler.TeamVotesCast;
        c["run_lobby_resets"] = MultiSeatStats.RunLobbyResets;
        c["remote.messages"] = MultiSeatStats.MessagesReceived;
        c["wire.round_trips"] = MultiSeatStats.WireRoundTrips;
        c["wire.bytes"] = MultiSeatStats.WireBytes;
    }
}
