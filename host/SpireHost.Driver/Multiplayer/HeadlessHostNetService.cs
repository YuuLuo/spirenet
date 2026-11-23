using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Messages.Game;
using MegaCrit.Sts2.Core.Multiplayer.Quality;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Platform;

namespace SpireHost.Driver;

/// <summary>
/// The net service of a multi-seat run (phase H5): an <see cref="INetGameService"/> of type <b>Host</b> without any
/// transport. One instance per run, engine thread only, no IO of any kind.
///
/// Why it exists. RunManager.SetUpTest(state, service) (Runs/RunManager.cs:355-374) is the one set-up entry of the game
/// that accepts an arbitrary service. With <c>Type == Host</c> every rule of a real multiplayer game applies
/// (CombatManager.cs:797, :829-845; TreasureRoomRelicSynchronizer.cs:114; PlayerChoiceSynchronizer.cs:81-84 ...); the
/// game's own "fake multiplayer" - several players on a Singleplayer service - changes those rules and is never used.
///
/// What it does, member by member (callers: notes\h5-plan\game-net-api.md section 1):
///   NetId / Type / IsConnected / Platform / LocalVersion   constants of a connected host whose local player is seat 0
///   Register / UnregisterMessageHandler                   the handler table (same semantics as NetMessageBus.cs:127-159)
///   SendMessage                                           what the host broadcasts or sends to one client: counted and
///                                                         offered to <see cref="Sent"/>; NEVER dispatched to the host's
///                                                         own handlers (NetHostGameService.cs:95-130 does not loop back)
///   SetBufferMessages                                     NetMessageBus.cs:107-125: while on, buffer-able messages wait
///   SetGameLoading / IsGameLoading                        the flag only
///   Disconnect / Update / Disconnected                    inert: RunManager.CleanUp calls Disconnect (:1485); the event
///                                                         is never raised (raising it would run RunLobby.OnDisconnected)
///
/// <see cref="Receive{T}"/> is the host-only entry: "client <c>senderId</c> sent this message". It does what
/// NetHostGameService.OnPacketReceived does after the transport and the deserialisation (NetHostGameService.cs:142-166):
/// the registered handlers are called with the sender's id. Unlike NetMessageBus.SendMessageToAllHandlers (:78-105)
/// nothing is swallowed: a message without a handler and an exception inside a handler both propagate (fail closed).
/// VERIFICATION MODE <see cref="WireRoundTrip"/> (lead ruling Q11: off by default): the message is first put on the
/// wire with the game's own serializer (NetMessageBus.SerializeMessage: [type id][sender id][payload]) and read back
/// (TryDeserializeMessage); the handlers get the decoded message and the sender id of the packet header, so they see
/// exactly what a real packet carries (bit widths of ids). A run must be identical with the mode on and off.
///
/// Never call RunManager.Abandon() in a run that uses this service: RunLobby.AbandonRun casts to NetHostGameService
/// (RunLobby.cs:152). Not implementing INetHostGameService is deliberate (RunLobby.cs:52-56 subscribes only then).
/// </summary>
public sealed class HeadlessHostNetService : INetGameService
{
    private struct Handler
    {
        public object Original;
        public Action<INetMessage, ulong> Call;
    }

    private readonly Dictionary<Type, List<Handler>> _handlers = new Dictionary<Type, List<Handler>>();
    private readonly List<(INetMessage Message, ulong Sender)> _buffered = new List<(INetMessage, ulong)>();
    private readonly NetMessageBus _wire;
    private bool _buffering;
    private bool _loading;

    public HeadlessHostNetService(ulong localNetId)
    {
        NetId = localNetId;
        PacketWriter writer = new PacketWriter { WarnOnGrow = false };
        _wire = new NetMessageBus(new PacketReader(), writer);
    }

    // ------------------------------------------------------------------------------------------------ INetGameService

    /// <summary>Seat 0's Player.NetId. Read once by RunManager.InitializeShared (:383) and by Launch (:596, LocalContext.NetId).</summary>
    public ulong NetId { get; }

    public bool IsConnected => true;

    public bool IsGameLoading => _loading;

    public NetGameType Type => NetGameType.Host;

    public PlatformType Platform => PlatformType.None;

    public PeerVersionInfo LocalVersion { get; } = PeerVersionInfo.LocalDefault();

    /// <summary>Never raised.</summary>
    public event Action<NetErrorInfo>? Disconnected
    {
        add { }
        remove { }
    }

    public void SendMessage<T>(T message, ulong playerId) where T : INetMessage
    {
        SentToOne++;
        Sent?.Invoke(message, playerId);
    }

    public void SendMessage<T>(T message) where T : INetMessage
    {
        Broadcasts++;
        if (typeof(T) == typeof(SyncPlayerDataMessage))
        {
            SyncBroadcasts++;
            SyncPlayerDataBroadcast?.Invoke();
        }
        Sent?.Invoke(message, null);
    }

    public void RegisterMessageHandler<T>(MessageHandlerDelegate<T> handler) where T : INetMessage
    {
        if (typeof(T) == typeof(INetMessage))
        {
            throw new InvalidOperationException("RegisterMessageHandler must be called with a concrete implementation of INetMessage!");
        }
        if (!_handlers.TryGetValue(typeof(T), out List<Handler>? list))
        {
            list = new List<Handler>(2);
            _handlers[typeof(T)] = list;
        }
        list.Add(new Handler { Original = handler, Call = (m, s) => handler((T)m, s) });
    }

    public void UnregisterMessageHandler<T>(MessageHandlerDelegate<T> handler) where T : INetMessage
    {
        if (typeof(T) == typeof(INetMessage))
        {
            throw new InvalidOperationException("UnregisterMessageHandler must be called with a concrete implementation of INetMessage!");
        }
        if (_handlers.TryGetValue(typeof(T), out List<Handler>? list))
        {
            list.RemoveAll(h => (MessageHandlerDelegate<T>)h.Original == handler);
        }
    }

    public void Update()
    {
    }

    public void Disconnect(NetError reason, bool now = false)
    {
        Disconnects++;
    }

    public ConnectionStats? GetStatsForPeer(ulong peerId) => null;

    public void SetGameLoading(bool isLoading)
    {
        _loading = isLoading;
    }

    public void SetBufferMessages(bool bufferMessages)
    {
        if (_buffering == bufferMessages)
        {
            return;
        }
        _buffering = bufferMessages;
        if (bufferMessages)
        {
            return;
        }
        (INetMessage, ulong)[] pending = _buffered.ToArray();
        _buffered.Clear();
        foreach ((INetMessage message, ulong sender) in pending)
        {
            Deliver(message, sender);
        }
    }

    public string? GetRawLobbyIdentifier() => null;

    // ------------------------------------------------------------------------------------------------ host side

    /// <summary>Host -> clients traffic (broadcast: <c>to</c> = null). Observer for tests; a reaction never runs inside it.</summary>
    public event Action<INetMessage, ulong?>? Sent;

    /// <summary>
    /// The host broadcast its SyncPlayerDataMessage (CombatStateSynchronizer.StartSync, :109-112): every client is at the
    /// same point of its own StartSync and sends its own player data (see <see cref="MultiSeatRun"/>).
    /// </summary>
    public event Action? SyncPlayerDataBroadcast;

    public long Broadcasts;
    public long SentToOne;
    public long SyncBroadcasts;
    public long Received;
    public long ReceivedBytes;
    public long BufferedOnArrival;
    public long Disconnects;

    /// <summary>VERIFICATION MODE: every received message is serialised and deserialised with the game's own serializer before it is handled.</summary>
    public bool WireRoundTrip { get; set; }

    public long WireRoundTrips;

    /// <summary>Number of message types that have a handler (diagnostics).</summary>
    public int HandlerTypes => _handlers.Count;

    public bool HasHandler(Type messageType) => _handlers.TryGetValue(messageType, out List<Handler>? list) && list.Count > 0;

    /// <summary>
    /// "Client <paramref name="senderId"/> sent <paramref name="message"/>": wire round trip, then the registered handlers.
    /// Call it from driver code (an Apply, or work posted to the pump), never from inside a game callback: handlers
    /// complete TaskCompletionSources whose continuations run inline.
    /// </summary>
    public void Receive<T>(T message, ulong senderId) where T : INetMessage
    {
        Received++;
        MultiSeatStats.MessagesReceived++;
        if (!WireRoundTrip)
        {
            Deliver(message, senderId);
            return;
        }
        byte[] packet = _wire.SerializeMessage(senderId, message, out int length);
        WireRoundTrips++;
        ReceivedBytes += length;
        MultiSeatStats.WireRoundTrips++;
        MultiSeatStats.WireBytes += length;
        if (!_wire.TryDeserializeMessage(packet, out INetMessage? decoded, out ulong? headerSender) || decoded == null || !headerSender.HasValue)
        {
            throw new InvalidOperationException("The message " + typeof(T).Name + " of client " + senderId + " could not be read back from its own packet.");
        }
        if (decoded.GetType() != typeof(T) || headerSender.Value != senderId)
        {
            throw new InvalidOperationException("The packet of " + typeof(T).Name + " from client " + senderId + " decoded as " + decoded.GetType().Name + " from " + headerSender.Value + ".");
        }
        Deliver(decoded, headerSender.Value);
    }

    private void Deliver(INetMessage message, ulong senderId)
    {
        if (_buffering && message.ShouldBuffer)
        {
            BufferedOnArrival++;
            _buffered.Add((message, senderId));
            return;
        }
        if (!_handlers.TryGetValue(message.GetType(), out List<Handler>? list) || list.Count == 0)
        {
            throw new InvalidOperationException("Received a " + message.GetType().Name + " from client " + senderId + ", but no message handler is registered for that type.");
        }
        if (list.Count == 1)
        {
            list[0].Call(message, senderId);
            return;
        }
        Handler[] snapshot = list.ToArray(); // a handler may unregister itself
        for (int i = 0; i < snapshot.Length; i++)
        {
            snapshot[i].Call(message, senderId);
        }
    }
}
