using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Engine;

namespace SpireHost.Driver;

/// <summary>
/// The per-episode object every handler, pending frame and lifecycle works through. Engine thread only.
/// It owns: the seat list, the "map is open" flag, per-room scratch data, the decision stack, the tracked
/// operations and the entry point for nested (synchronous-callback) decisions.
/// </summary>
public sealed class RunContext
{
    private readonly RunDriver _driver;
    private readonly ConditionalWeakTable<AbstractRoom, Dictionary<Type, object>> _roomData = new ConditionalWeakTable<AbstractRoom, Dictionary<Type, object>>();

    internal RunContext(HostSession session, RunDriver driver, EpisodeSpec spec, EpisodeOptions options, RunState run, string seed)
    {
        Session = session;
        _driver = driver;
        Spec = spec;
        Options = options;
        Run = run;
        Seed = seed;
        Pump = session.Pump;
    }

    public HostSession Session { get; }

    public EnginePump Pump { get; }

    public EpisodeSpec Spec { get; }

    public EpisodeOptions Options { get; }

    public EpisodeCounters Counters => _driver.Counters;

    /// <summary>The live run. Valid from OnRunSetUp until the episode is torn down.</summary>
    public RunState Run { get; }

    /// <summary>The canonical game seed. HOST USE ONLY (reports, replay): never put it into an observation.</summary>
    public string Seed { get; }

    public RunManager RunManager => RunManager.Instance;

    public CombatManager CombatManager => CombatManager.Instance;

    /// <summary>Seats = RunState.Players (seat index = slot index). One seat in H1.</summary>
    public IReadOnlyList<Player> Seats => Run.Players;

    public int SeatCount => Run.Players.Count;

    /// <summary>
    /// The player the local game instance acts for: seat 0, in single-seat and in multi-seat runs (the run's net service
    /// has seat 0's net id, and RunManager.Launch makes that id LocalContext.NetId).
    /// </summary>
    public Player LocalPlayer => Run.Players[0];

    public int SeatOf(Player player) => Run.GetPlayerSlotIndex(player);

    /// <summary>
    /// MULTI-SEAT (phase H5): the Host-type net service of the run and the virtual clients of seats 1..N-1; null in a
    /// single-seat run (RunManager.SetUpNewSingleplayer, NetSingleplayerGameService). Set once, by the run start chain.
    /// </summary>
    public MultiSeatRun? Mp { get; internal set; }

    /// <summary>True for seats 1..N-1 of a multi-seat run: whatever the player of that seat does reaches the game as a message of its virtual client.</summary>
    public bool IsRemoteSeat(int seat) => seat != 0;

    /// <summary>
    /// "A use of this potion has been requested and is not executed yet" as the potion popup of the potion's OWNER sees it
    /// (NPotionPopup.cs:146: both buttons off): PotionModel.IsQueued for the local seat, the virtual client's own flag
    /// for a remote seat (<see cref="VirtualClient.IsPotionQueued"/>).
    /// </summary>
    public bool IsPotionQueued(MegaCrit.Sts2.Core.Models.PotionModel potion)
    {
        MultiSeatRun? mp = Mp;
        if (mp != null)
        {
            MegaCrit.Sts2.Core.Entities.Players.Player? owner = potion._owner; // no owner: a potion on a shelf or in a reward
            VirtualClient? client = owner == null ? null : mp.ClientOf(owner.NetId);
            if (client != null)
            {
                return client.IsPotionQueued(potion);
            }
        }
        return potion.IsQueued;
    }

    public AbstractRoom? CurrentRoom => Run.CurrentRoom;

    /// <summary>
    /// The host's card selector of this run: the object whose GetSelectedCards parks CARD_SELECT frames.
    /// Set in OnRunSetUp by the lifecycle that installs it (CardSelectorStack.UseLocalSelector). Code that
    /// has to push its own selector for a moment (card rewards: CardReward.OnSelect only reads the per-player
    /// stack) forwards GetSelectedCards to this one, so that prompts raised meanwhile keep their owner.
    /// </summary>
    public MegaCrit.Sts2.Core.TestSupport.ICardSelector? CardSelector { get; set; }

    // ------------------------------------------------------------------------------------------------
    //  Map / travel
    // ------------------------------------------------------------------------------------------------

    /// <summary>
    /// The host's stand-in for the real UI's "travel is enabled and the map is open"
    /// (NMapScreen.SetTravelEnabled(true) + Open, i.e. RunManager.Proceed(), which is a no-op headlessly).
    /// While true the screen classifies to RoomScreen.Map. Reset whenever a room is entered.
    /// Set by the core whenever the game calls RunManager.Proceed() (run-proceed-opens-map): room owners make that
    /// game call for their LEAVE / PROCEED action, exactly as the UI node does.
    /// </summary>
    public bool MapOpen { get; private set; }

    /// <summary>Number of RunManager.RoomEntered events seen in this episode.</summary>
    public int RoomSerial { get; private set; }

    /// <summary>
    /// "The real UI now lets the player leave for the map." Idempotent. Normally reached through the game call
    /// RunManager.Proceed() (see <see cref="MapOpen"/>); call it directly only where the real UI opens the map without
    /// that call (test scenarios).
    /// </summary>
    public void OpenMap(string why)
    {
        AssertEngineThread();
        MapOpen = true;
        _driver.Note("map_open:" + why);
    }

    internal void CloseMap()
    {
        MapOpen = false;
    }

    internal void OnRoomEntered()
    {
        RoomSerial++;
        MapOpen = false;
        for (int s = 0; s < _seatMapOpen.Length; s++)
        {
            _seatMapOpen[s] = false;
            _seatActReady[s] = false;
            _seatFlags[s] = 0;
        }
        Counters.RoomsEntered++;
    }

    // ------------------------------------------------------------------------------------------------
    //  MULTI-SEAT (phase H5): the per-seat part of "where is this player's UI"
    // ------------------------------------------------------------------------------------------------

    private readonly bool[] _seatMapOpen = new bool[RunDriver.MaxSeats];
    private readonly bool[] _seatActReady = new bool[RunDriver.MaxSeats];

    /// <summary>
    /// "This seat's map is open and travel is enabled": the seat pressed PROCEED / LEAVE in the current room. Seat 0 =
    /// <see cref="MapOpen"/> (the game call RunManager.Proceed). A remote seat's client makes that call in its own
    /// instance and sends nothing (notes\h5-plan\rooms-multiseat-duties.md 1.2), so its flag is host bookkeeping, set
    /// by the room owner that applies the seat's PROCEED (<see cref="OpenMapFor"/>). Reset at every room entry.
    /// </summary>
    public bool IsMapOpen(int seat) => seat == 0 ? MapOpen : _seatMapOpen[seat];

    /// <summary>The seat pressed the button that opens its map (see <see cref="IsMapOpen"/>). Seat 0: only where no game call does it.</summary>
    public void OpenMapFor(int seat, string why)
    {
        AssertEngineThread();
        if (seat == 0)
        {
            OpenMap(why);
            return;
        }
        _seatMapOpen[seat] = true;
        _driver.Note("map_open[" + seat + "]:" + why);
    }

    /// <summary>
    /// The seat pressed the terminal PROCEED of the act's last boss: its vote to move to the next act is cast
    /// (VoteToMoveToNextActAction) and cannot be taken back; it waits for the other seats. Reset at every room entry.
    /// </summary>
    public bool IsActReady(int seat) => _seatActReady[seat];

    public void SetActReady(int seat)
    {
        AssertEngineThread();
        _seatActReady[seat] = true;
        _driver.Note("act_ready[" + seat + "]");
    }

    private readonly int[] _seatFlags = new int[RunDriver.MaxSeats];

    /// <summary>A room owner records a public fact about a seat (a vote it cast, its rest site done). Reset at every room entry.</summary>
    public void SetSeatFlag(int seat, SeatStatus flag)
    {
        AssertEngineThread();
        _seatFlags[seat] |= (int)flag;
    }

    /// <summary>Clears a flag for every seat (a vote round resolved: the next page of a shared event starts without votes).</summary>
    public void ClearSeatFlag(SeatStatus flag)
    {
        AssertEngineThread();
        for (int s = 0; s < _seatFlags.Length; s++)
        {
            _seatFlags[s] &= ~(int)flag;
        }
    }

    /// <summary>
    /// PUBLIC per-seat status bits (<see cref="SeatStatus"/>; obs erratum H5, GLOBAL 56-59): what every player sees of a
    /// teammate's progress in the current room - it left its room screen, it voted, it is ready for the next act.
    /// </summary>
    public int SeatStatusBits(int seat)
    {
        return (IsMapOpen(seat) ? (int)SeatStatus.Proceeded : 0) | (IsActReady(seat) ? (int)SeatStatus.ActReady : 0) | _seatFlags[seat];
    }

    /// <summary>
    /// The kind of screen a seat has open right now, as far as it follows from the room state (obs erratum H5, GLOBAL
    /// 52-55; the real UI shows every teammate's screen type, NMultiplayerPlayerState): COMBAT while a combat is in
    /// progress; REWARD while a remote seat has a reward screen; None once the seat is ready for the next act; MAP
    /// once its map is open; else the room's screen (EVENT / SHOP / REST / TREASURE); None otherwise (a seat in a
    /// finished combat room whose reward screen is a frame of its own). A prompt of a seat is not reported here: it is
    /// the decision that is being asked.
    /// </summary>
    public DecisionKind SeatScreen(int seat)
    {
        if (CombatManager.IsInProgress)
        {
            return DecisionKind.Combat;
        }
        if (seat != 0 && Session.Registry.RemoteOverlay?.IsOpen(this, seat) == true)
        {
            return DecisionKind.Reward;
        }
        if (IsActReady(seat))
        {
            return DecisionKind.None;
        }
        AbstractRoom? room = Run.CurrentRoom;
        if (IsMapOpen(seat) || room is MapRoom)
        {
            return DecisionKind.Map;
        }
        return room switch
        {
            EventRoom => DecisionKind.Event,
            MerchantRoom => DecisionKind.Shop,
            RestSiteRoom => DecisionKind.Rest,
            TreasureRoom => DecisionKind.Treasure,
            _ => DecisionKind.None,
        };
    }

    /// <summary>
    /// Scratch state of a handler for one room instance (e.g. "chest opened", "rest successes"). Created on
    /// first use, dropped with the room. Keyed by the room object, so it survives an event-internal combat.
    /// </summary>
    public T RoomData<T>(AbstractRoom room) where T : class, new()
    {
        Dictionary<Type, object> bag = _roomData.GetOrCreateValue(room);
        if (!bag.TryGetValue(typeof(T), out object? v))
        {
            v = new T();
            bag[typeof(T)] = v;
        }
        return (T)v;
    }

    public T RoomData<T>() where T : class, new()
    {
        return RoomData<T>(Run.CurrentRoom ?? throw new InvalidOperationException("No current room."));
    }

    // ------------------------------------------------------------------------------------------------
    //  Decision stack (requests the game made through async callbacks)
    // ------------------------------------------------------------------------------------------------

    /// <summary>Parks a request on top of the decision stack. The driver core asks it at the next quiescent point.</summary>
    public void Push(IPendingDecision frame)
    {
        AssertEngineThread();
        _driver.Push(frame);
    }

    /// <summary>Removes a frame (normally the top one) once the game's request has been answered.</summary>
    public void Pop(IPendingDecision frame)
    {
        AssertEngineThread();
        _driver.Pop(frame);
    }

    public int PendingCount => _driver.PendingCount;

    public IPendingDecision? TopPending => _driver.TopPending;

    // ------------------------------------------------------------------------------------------------
    //  Tracked operations (tasks the driver started and must see through)
    // ------------------------------------------------------------------------------------------------

    /// <summary>
    /// Hands a task that a game API returned to the driver core. NEVER block on such a task and never ignore
    /// it: the core observes it at every quiescent point - a fault aborts the episode (fail-closed,
    /// DECISIONS-H1 A.7); while a <paramref name="blocking"/> operation is incomplete no room handler is
    /// consulted (only pending frames, and the combat handler, may ask), and if it is still incomplete when
    /// the engine is quiescent with an empty decision stack and no combat, the episode stalls.
    /// Use blocking: false for tasks that legitimately outlive the next room-level decision.
    /// </summary>
    public void Track(Task task, string what, bool blocking = true)
    {
        AssertEngineThread();
        _driver.Track(task, what, blocking, -1);
    }

    /// <summary>
    /// MULTI-SEAT (phase H5): as <see cref="Track(Task, string, bool)"/>, for work that belongs to one seat (its event
    /// option, purchase, rest option, chest). While it is incomplete only THAT seat's room handler is not consulted; the
    /// other seats go on (the operation may be waiting for them: a shared vote, a remote reward screen). An operation
    /// tracked without a seat holds every seat's room handler back. In a single-seat run there is no difference.
    /// </summary>
    public void Track(Task task, string what, int seat, bool blocking = true)
    {
        AssertEngineThread();
        _driver.Track(task, what, blocking, seat);
    }

    /// <summary>True while a blocking tracked operation of this seat, or of no particular seat, is incomplete.</summary>
    public bool HasBlockingOperationFor(int seat) => _driver.HasBlockingOperationFor(seat);

    /// <summary>
    /// Runs <paramref name="work"/> from the pump queue (not inside the caller's stack) and tracks the task it
    /// returns. For game event handlers that must not await inside the event callback (CombatWon,
    /// RelicsAwarded ...): the real UI reacts to those events on a later frame.
    /// </summary>
    public void Post(Func<Task> work, string what, bool blocking = true)
    {
        AssertEngineThread();
        _driver.PostWork(work, what, blocking);
    }

    public bool HasBlockingOperation => _driver.HasBlockingOperation;

    /// <summary>
    /// A task that completes at the next QUIESCENT point (pump queue empty, no fault, virtual timers drained),
    /// on the engine thread. The correct way for host-side async code to wait for the engine to settle:
    ///     ctx.Post(async () => { await ctx.Quiescence(); ...read settled state, call the next game API... }, "what");
    /// Never Wait() / .Result on any task and never spin the pump yourself.
    /// </summary>
    public Task Quiescence()
    {
        AssertEngineThread();
        return _driver.Quiescence();
    }

    // ------------------------------------------------------------------------------------------------
    //  Services of the core that handlers and encoders reach through the episode
    // ------------------------------------------------------------------------------------------------

    /// <summary>
    /// The registered "canonical classes of identical cards" provider (combat C7 = obs O1), or null when none is
    /// registered. The CARD_SELECT owner orders pile-sourced candidates with it; in production it is the content key
    /// of the observation encoder and is registered in every mode.
    /// </summary>
    public CanonicalCardClasses? CanonicalCardClasses => Session.Registry.CanonicalCardClasses;

    /// <summary>
    /// True while the enemy side's turn of this round is running and this enemy has begun its move: the real UI has
    /// hidden its intent (combat C3; DECISIONS-H1 F.6). Public information.
    /// </summary>
    public bool IntentPerformed(MegaCrit.Sts2.Core.Entities.Creatures.Creature enemy) => DriverPatches.IntentPerformed(enemy);

    // ------------------------------------------------------------------------------------------------
    //  Nested decisions (synchronous game callbacks)
    // ------------------------------------------------------------------------------------------------

    /// <summary>
    /// Asks the policy from inside a SYNCHRONOUS game callback (ICardSelector.GetSelectedCardReward and the
    /// like) and returns the chosen action index. Does not pump and does not re-enter the engine: the caller
    /// is game code that is in the middle of running. The decision goes through the same central path as any
    /// other (canonical ordering, auto-advance, counters, trace); its <see cref="Decision.Source"/> is not
    /// invoked - the caller turns the index into the callback's return value.
    /// </summary>
    public int DecideNested(Decision decision)
    {
        AssertEngineThread();
        return _driver.AskNested(decision);
    }

    // ------------------------------------------------------------------------------------------------
    //  Faults
    // ------------------------------------------------------------------------------------------------

    /// <summary>Ends the episode as Aborted. Use for "this cannot happen if the handler is right" conditions; never swallow them.</summary>
    [DoesNotReturn]
    public void Fail(AbortReason reason, string detail, string message)
    {
        throw new EpisodeAbortException(reason, detail, message);
    }

    /// <summary>Shorthand for Fail(AbortReason.Stall, ...).</summary>
    [DoesNotReturn]
    public void Stall(string detail, string message)
    {
        throw new EpisodeAbortException(AbortReason.Stall, detail, message);
    }

    public void AssertEngineThread()
    {
        if (!Pump.OnEngineThread)
        {
            throw new InvalidOperationException("RunContext may only be used on the engine thread.");
        }
    }

    /// <summary>One-line engine / driver snapshot (stall diagnostics, logs).</summary>
    public string Diagnose() => _driver.Diagnose();
}
