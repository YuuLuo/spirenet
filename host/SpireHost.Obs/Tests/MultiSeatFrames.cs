using System;
using System.Collections.Generic;
using System.Linq;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Protocol;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Tests;

/// <summary>
/// obs --selftest --cases frames_mp,protocol_mp (phase H5, milestone M3; opt-in cases, not part of the default list):
/// the frames of 2-4 seat episodes in TEAM view against the frozen v1 format plus erratum H5
/// (host\SpireHost.Obs\SCHEMA-ERRATA-H5.md).
///
///   frames_mp    whole runs with 2, 3 and 4 seats through the EncodeChecker of the single-seat fuzz case (every frame:
///                structure, pointers, slices, orders; no side effect; encoding twice is identical; hook cache == exact
///                game calls) plus the multi-seat rules checked on the BYTES of every frame: seat columns, acting seat,
///                GLOBAL 52-59, map / option vote bits, MEND aux, hand rows of every seat, UNDO only when asked for.
///   protocol_mp  the binary protocol with n_seats 2-4: refusals (SEAT view, undefined flag bits), whole episodes over a
///                scripted transport with and without team votes / UNDO, TERMINAL bookkeeping.
/// </summary>
public static class MultiSeatFrames
{
    private sealed class Rules
    {
        public readonly SortedDictionary<string, int> Problems = new SortedDictionary<string, int>(StringComparer.Ordinal);
        public readonly List<string> Examples = new List<string>();
        public readonly SortedDictionary<string, long> KindBySeat = new SortedDictionary<string, long>(StringComparer.Ordinal);
        public long Frames, MapVoteBits, OptionVoteBits, MendActions, UndoRows, StatusNonZero, ScreenNonZero, TeamHandRows;
        public int MaxRelicRows, MaxCardRows, MaxActionRows, MaxFrameBytes;
        public long FrameBytes;

        public void Problem(string what, string example)
        {
            Problems[what] = Problems.GetValueOrDefault(what) + 1;
            if (Examples.Count < 20)
            {
                Examples.Add(what + ": " + example);
            }
        }
    }

    /// <summary>The erratum-H5 rules, read from the encoded bytes of one frame.</summary>
    private static void Check(Rules rules, byte[] frame, int nSeats, bool undoAllowed, int mendId, string label)
    {
        List<string> structural = FrameCheck.Problems(frame, out FrameCheck.View v);
        foreach (string p in structural)
        {
            rules.Problem("frame: " + p, label);
        }
        if (structural.Count > 0)
        {
            return;
        }
        rules.Frames++;
        rules.FrameBytes += frame.Length;
        rules.MaxFrameBytes = Math.Max(rules.MaxFrameBytes, frame.Length);
        rules.MaxRelicRows = Math.Max(rules.MaxRelicRows, v.Rows[T.Relics]);
        rules.MaxCardRows = Math.Max(rules.MaxCardRows, v.Rows[T.Cards]);
        rules.MaxActionRows = Math.Max(rules.MaxActionRows, v.Rows[T.Actions]);
        bool terminal = v.H(Hd.MsgType) == MsgType.Terminal;
        int kind = v.G(Gl.Kind);
        int acting = v.G(Gl.ActingSeat);
        if ((v.H(Hd.Flags) & FrameFlag.Overflow) != 0)
        {
            rules.Problem("frame overflow (a table reached its row cap)", label);
        }
        if (v.G(Gl.NSeats) != nSeats || v.Rows[T.Seats] != nSeats)
        {
            rules.Problem("n_seats / SEATS rows", $"{label}: GLOBAL {v.G(Gl.NSeats)}, rows {v.Rows[T.Seats]}, expected {nSeats}");
        }
        if (v.G(Gl.ViewMode) != 0 || v.G(Gl.ViewSeat) != -1 || (v.H(Hd.Flags) & FrameFlag.TeamView) == 0)
        {
            rules.Problem("not a TEAM-view frame", label);
        }
        for (int s = 0; s < v.Rows[T.Seats]; s++)
        {
            if (v.Get(T.Seats, s, SeatC.Seat) != s || v.Get(T.Seats, s, SeatC.PrivateVisible) != 1)
            {
                rules.Problem("SEATS row order / private_visible", label);
            }
        }
        // ---- acting seat and action seats ----
        bool combat = kind == (int)DecisionKind.Combat;
        if (!terminal)
        {
            if (!combat && (acting < 0 || acting >= nSeats))
            {
                rules.Problem("acting_seat outside combat", $"{label}: {acting}");
            }
            rules.KindBySeat[((DecisionKind)kind) + ":" + (acting < 0 ? "union" : "seat" + acting)] = rules.KindBySeat.GetValueOrDefault(((DecisionKind)kind) + ":" + (acting < 0 ? "union" : "seat" + acting)) + 1;
        }
        for (int r = 0; r < v.Rows[T.Actions]; r++)
        {
            int seat = v.Get(T.Actions, r, ActionC.Seat);
            int verb = v.Get(T.Actions, r, ActionC.Verb);
            if (seat < 0 || seat >= nSeats)
            {
                rules.Problem("ACTIONS.seat out of range", $"{label}: {seat}");
            }
            if (!combat && seat != acting)
            {
                rules.Problem("an action of another seat on a single-seat frame", $"{label}: action seat {seat}, acting {acting}");
            }
            if (verb == (int)Verb.UndoEndTurn)
            {
                rules.UndoRows++;
                if (!undoAllowed)
                {
                    rules.Problem("UNDO_END_TURN offered without the option", label);
                }
            }
            // MEND: one action per target seat (aux), never the owner
            int srcTable = v.Get(T.Actions, r, ActionC.SrcTable), srcRow = v.Get(T.Actions, r, ActionC.SrcRow);
            if (kind == (int)DecisionKind.Rest && verb == (int)Verb.ChooseOption && srcTable == T.Options && srcRow >= 0
                && v.Get(T.Options, srcRow, OptionC.Kind) == OptionKind.RestOption && v.Get(T.Options, srcRow, OptionC.KeyId) == mendId)
            {
                rules.MendActions++;
                int aux = v.Get(T.Actions, r, ActionC.Aux);
                if (aux < 0 || aux >= nSeats || aux == seat)
                {
                    rules.Problem("MEND target seat (ACTIONS.aux)", $"{label}: owner {seat}, aux {aux}");
                }
            }
        }
        for (int r = 0; r < v.Rows[T.Options]; r++)
        {
            if (v.Get(T.Options, r, OptionC.Seat) != (acting >= 0 ? acting : 0))
            {
                rules.Problem("OPTIONS.seat is not the acting seat", label);
            }
        }
        // ---- GLOBAL 52-59 ----
        for (int s = 0; s < 4; s++)
        {
            int screen = v.G(Gl.SeatScreen0 + s), status = v.G(Gl.SeatStatus0 + s);
            if (s >= nSeats ? (screen != 0 || status != 0) : (screen < 0 || screen > (int)DecisionKind.Crystal || status < 0 || status >= 64))
            {
                rules.Problem("GLOBAL 52-59 out of range", $"{label}: seat {s} screen {screen} status {status}");
            }
            if (s < nSeats)
            {
                rules.StatusNonZero += status != 0 ? 1 : 0;
                rules.ScreenNonZero += screen != 0 ? 1 : 0;
                if (!terminal && s == acting && screen != kind)
                {
                    rules.Problem("seat_screen of the acting seat is not the frame's kind", $"{label}: {screen}");
                }
                if (!terminal && combat && screen != (int)DecisionKind.Combat)
                {
                    rules.Problem("seat_screen in a combat frame", $"{label}: seat {s} {screen}");
                }
                if (!terminal && kind == (int)DecisionKind.Map && s == acting && (status & (int)SeatStatus.VotedMap) != 0)
                {
                    rules.Problem("a seat that voted is asked for its map vote again", label);
                }
            }
        }
        for (int i = 60; i < 64; i++)
        {
            if (v.G(i) != 0)
            {
                rules.Problem("GLOBAL 60-63 not zero", label);
            }
        }
        // ---- vote bits ----
        int[] mapVotes = new int[4];
        for (int r = 0; r < v.Rows[T.MapNodes]; r++)
        {
            int f = v.Get(T.MapNodes, r, MapNodeC.Flags);
            if ((f & ~0xFF) != 0)
            {
                rules.Problem("MAP_NODES.flags has undefined bits", label);
            }
            for (int s = 0; s < 4; s++)
            {
                if ((f & (MapNodeFlag.VoteSeat0 << s)) != 0)
                {
                    mapVotes[s]++;
                    rules.MapVoteBits++;
                    if (s >= nSeats || (v.G(Gl.SeatStatus0 + s) & (int)SeatStatus.VotedMap) == 0)
                    {
                        rules.Problem("map vote bit of a seat whose status does not say VOTED_MAP", $"{label}: seat {s}");
                    }
                }
            }
        }
        int[] optionVotes = new int[4];
        for (int r = 0; r < v.Rows[T.Options]; r++)
        {
            int f = v.Get(T.Options, r, OptionC.Flags);
            int ok = v.Get(T.Options, r, OptionC.Kind);
            if ((f & ~0x7FF) != 0)
            {
                rules.Problem("OPTIONS.flags has undefined bits", label);
            }
            for (int s = 0; s < 4; s++)
            {
                if ((f & (OptionFlag.VoteSeat0 << s)) != 0)
                {
                    optionVotes[s]++;
                    rules.OptionVoteBits++;
                    if (s >= nSeats || (ok != OptionKind.TreasureRelic && ok != OptionKind.EventOption))
                    {
                        rules.Problem("option vote bit on a row that cannot be voted for", $"{label}: seat {s}, option kind {ok}");
                    }
                    if (s == acting)
                    {
                        rules.Problem("the acting seat already has a vote on its own vote frame", label);
                    }
                }
            }
        }
        for (int s = 0; s < 4; s++)
        {
            if (mapVotes[s] > 1 || optionVotes[s] > 1)
            {
                rules.Problem("a seat has more than one vote in a frame", $"{label}: seat {s}");
            }
        }
        // ---- TEAM view: every seat's hand is in a combat frame (count = SEATS.hand_count) ----
        if (v.G(Gl.InCombat) == 1)
        {
            int[] hand = new int[4];
            for (int r = 0; r < v.Rows[T.Cards]; r++)
            {
                if (v.Get(T.Cards, r, CardC.Zone) == Zone.Hand)
                {
                    int seat = v.Get(T.Cards, r, CardC.Seat);
                    if (seat >= 0 && seat < 4)
                    {
                        hand[seat]++;
                        rules.TeamHandRows += seat > 0 ? 1 : 0;
                    }
                }
            }
            for (int s = 0; s < nSeats; s++)
            {
                if (hand[s] != v.Get(T.Seats, s, SeatC.HandCount))
                {
                    rules.Problem("HAND rows of a seat != SEATS.hand_count", $"{label}: seat {s} rows {hand[s]} count {v.Get(T.Seats, s, SeatC.HandCount)}");
                }
            }
        }
    }

    // =====================================================================================================
    //  frames_mp
    // =====================================================================================================

    public static object FramesMp(HostSession session, ObsRuntime.Loaded l, int runsPerSeatCount, out bool ok)
    {
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = new EncoderOptions { MapInCombat = true, DeckInCombat = true, UpgradePreviews = true, TargetPreviews = true };
        EncodeChecker chk = new EncodeChecker(enc) { SyntheticEvery = 0, Reference = ObsRuntime.NewEncoder(l) };
        Rules rules = new Rules();
        int mendId = l.Vocab.Id("rest_option", "MEND");
        int nSeats = 0;
        bool undo = false;
        string label = "";
        ObsEncoder own = ObsRuntime.NewEncoder(l);
        own.Options = enc.Options.Clone();
        chk.Also = (ctx, d) =>
        {
            own.EncodeDecision(d, 1, (int)ctx.Counters.PolicyDecisions, 0);
            Check(rules, own.Payload.ToArray(), nSeats, undo, mendId, label + " " + d.Kind + " decision " + ctx.Counters.DecisionPoints);
        };
        Dictionary<string, int> outcomes = new Dictionary<string, int>();
        List<string> aborts = new List<string>();
        List<object> episodes = new List<object>();
        int index = 0;
        foreach (int n in new[] { 2, 3, 4 })
        {
            for (int i = 0; i < runsPerSeatCount; i++, index++)
            {
                nSeats = n;
                undo = i % 2 == 0;
                bool teamVotes = i % 3 == 1;
                bool hp = i % 2 == 1;
                string[] lineup = new string[n];
                for (int k = 0; k < n; k++)
                {
                    lineup[k] = DriverCommands.AllCharacters[(index + k) % 6];
                }
                EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel("OBS-SELFTEST-MP:" + n + ":" + i), Characters = lineup, Ascension = i % 4 == 3 ? 10 : 0 };
                label = n + " seats run " + i;
                EpisodeOptions o = new EpisodeOptions { DebugForcedHp = hp ? 99999 : 0, MaxPolicyDecisions = hp ? 4000 : 0, MaxCombatTurns = 60, TeamVotes = teamVotes, OfferUndoEndTurn = undo };
                EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)(900 + index)), o, chk);
                outcomes[r.Outcome.ToString()] = outcomes.GetValueOrDefault(r.Outcome.ToString()) + 1;
                if (r.Abort != null)
                {
                    aborts.Add(r.Abort.Signature + " | " + r.Abort.Message);
                }
                episodes.Add(new { seats = n, lineup = string.Join("+", lineup), team_votes = teamVotes, undo, hp = hp ? 99999 : 0, outcome = r.Outcome.ToString(), r.TruncatedBy, floor = r.TotalFloor, decisions = r.Counters.PolicyDecisions });
            }
        }
        EncoderStats s = enc.Stats;
        // coverage: the multi-seat fields must have been exercised, and every room-level kind must have been asked from a remote seat
        List<string> notCovered = new List<string>();
        foreach (string kind in new[] { "Map", "Reward", "CardReward", "Event", "Shop", "Rest", "Treasure", "CardSelect" })
        {
            if (!rules.KindBySeat.Keys.Any(k => k.StartsWith(kind + ":seat", StringComparison.Ordinal) && !k.EndsWith(":seat0", StringComparison.Ordinal)))
            {
                notCovered.Add(kind + " frame of a remote seat");
            }
        }
        if (rules.MapVoteBits == 0) notCovered.Add("map vote bits");
        if (rules.OptionVoteBits == 0) notCovered.Add("option vote bits");
        if (rules.MendActions == 0) notCovered.Add("MEND actions");
        if (rules.UndoRows == 0) notCovered.Add("UNDO_END_TURN rows");
        if (rules.StatusNonZero == 0 || rules.ScreenNonZero == 0) notCovered.Add("GLOBAL 52-59");
        if (rules.TeamHandRows == 0) notCovered.Add("hand rows of remote seats");
        ok = chk.Problems.Count == 0 && rules.Problems.Count == 0 && s.ActionOrderViolations == 0 && s.UnmappedActionRefs == 0 && s.ChoiceOrderViolations == 0 && s.Overflows == 0
             && aborts.Count == 0 && notCovered.Count == 0;
        return new
        {
            runs_per_seat_count = runsPerSeatCount,
            outcomes,
            aborts,
            frames_checked = chk.Frames,
            frames_rule_checked = rules.Frames,
            side_effects = chk.SideEffects,
            non_deterministic = chk.NonDeterministic,
            exact_compared = chk.ExactCompared,
            exact_mismatch = chk.ExactMismatches,
            problems = chk.Problems,
            examples = chk.Examples,
            rule_problems = rules.Problems,
            rule_examples = rules.Examples,
            not_covered = notCovered,
            frames_by_kind_and_seat = rules.KindBySeat,
            map_vote_bits = rules.MapVoteBits,
            option_vote_bits = rules.OptionVoteBits,
            mend_actions = rules.MendActions,
            undo_rows = rules.UndoRows,
            mean_frame_bytes = rules.Frames == 0 ? 0 : rules.FrameBytes / rules.Frames,
            max_frame_bytes = rules.MaxFrameBytes,
            max_rows = new { relics = rules.MaxRelicRows, relics_cap = SchemaV1.MaxRows[T.Relics], cards = rules.MaxCardRows, cards_cap = SchemaV1.MaxRows[T.Cards], actions = rules.MaxActionRows, actions_cap = SchemaV1.MaxRows[T.Actions] },
            stats = ObsCommands.Stats(s, l.Vocab),
            episodes,
        };
    }

    // =====================================================================================================
    //  protocol_mp
    // =====================================================================================================

    private static byte[] Reset(int[] characters, ulong seed, int flags, int maxDecisions, int viewMode = 0, int viewSeat = -1, int debugWord = 0)
    {
        byte[] b = ScriptTransport.Reset(characters[0], 0, seed, flags, maxDecisions, characters.Length, viewMode, viewSeat, debugWord);
        for (int s = 1; s < characters.Length; s++)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan((6 + s) * 4), characters[s]);
        }
        return b;
    }

    public static object ProtocolMp(HostSession session, ObsRuntime.Loaded l, out bool ok)
    {
        List<string> fails = new List<string>();
        Dictionary<string, object> facts = new Dictionary<string, object>();
        void Expect(bool cond, string what)
        {
            if (!cond)
            {
                fails.Add(what);
            }
        }
        int[] four = { l.Vocab.Id("character", "IRONCLAD"), l.Vocab.Id("character", "SILENT"), l.Vocab.Id("character", "REGENT"), l.Vocab.Id("character", "IRONCLAD") };
        int[] three = { l.Vocab.Id("character", "DEFECT"), l.Vocab.Id("character", "NECROBINDER"), l.Vocab.Id("character", "FOOL") };
        int[] two = { four[0], four[1] };
        int mendId = l.Vocab.Id("rest_option", "MEND");

        // ---- 1: refusals ----
        {
            int phase = 0;
            ScriptTransport t = new ScriptTransport(tr =>
            {
                switch (phase++)
                {
                    case 0:
                        return Reset(two, SeedCodec.Encode("SELFTESTMP01"), ResetFlag.Default, 0, viewMode: 1, viewSeat: 0);
                    case 1:
                        Expect(tr.Written.Count == 1 && ScriptTransport.Msg(tr.Last!) == MsgType.Error && ScriptTransport.GlobalOf(tr.Last!, 0) == ErrorCode.BadMessage, "multi-seat RESET in SEAT view -> ERROR(3)");
                        return Reset(four, SeedCodec.Encode("SELFTESTMP01"), ResetFlag.Default | 0x100, 0);
                    case 2:
                        Expect(tr.Written.Count == 2 && ScriptTransport.Msg(tr.Last!) == MsgType.Error && ScriptTransport.GlobalOf(tr.Last!, 0) == ErrorCode.BadMessage, "multi-seat RESET with an undefined flag bit (0x100) -> ERROR(3)");
                        return ScriptTransport.Reset(four[0], 0, SeedCodec.Encode("SELFTESTMP01"), flags: ResetFlag.Default | ResetFlag.OfferUndoEndTurn);
                    case 3:
                        Expect(tr.Written.Count == 3 && ScriptTransport.Msg(tr.Last!) == MsgType.Error, "single-seat RESET with a multi-seat flag bit (0x80) -> ERROR(3), as before H5");
                        return ScriptTransport.Quit();
                    default:
                        return null;
                }
            });
            ProtocolServer server = new ProtocolServer(session, t, ObsRuntime.NewEncoder(l), new ServeOptions());
            Expect(server.Serve() == 0 && server.Episodes == 0, "refusals: exit 0, no episode started");
        }

        // ---- 2 / 3: whole episodes, default flags (team votes, no UNDO) and with both option bits ----
        foreach ((string name, int[] chars, int flags, string seed) in new[]
                 {
                     ("default_4", four, ResetFlag.Default, "SELFTESTMP02"),
                     ("ask_every_seat_undo_3", three, ResetFlag.Default | ResetFlag.AskEverySeat | ResetFlag.OfferUndoEndTurn, "SELFTESTMP03"),
                     ("default_2", two, ResetFlag.Default, "SELFTESTMP04"),
                 })
        {
            int n = chars.Length;
            bool undo = (flags & ResetFlag.OfferUndoEndTurn) != 0;
            bool everySeat = (flags & ResetFlag.AskEverySeat) != 0;
            Rules rules = new Rules();
            Random rng = new Random(11 + n);
            int steps = 0, obs = 0, terminal = 0, travelsHere = 0, mapFramesAfterTravel = 0;
            long curNode = long.MinValue;
            List<int> travelsPerLocation = new List<int>();
            bool started = false;
            ScriptTransport t = new ScriptTransport(tr =>
            {
                if (!started)
                {
                    started = true;
                    return Reset(chars, SeedCodec.Encode(seed), flags, 900);
                }
                byte[] last = tr.Last!;
                int msg = ScriptTransport.Msg(last);
                if (msg == MsgType.Terminal)
                {
                    terminal++;
                    Check(rules, last, n, undo, mendId, name + " TERMINAL");
                    Expect(ScriptTransport.StepOf(last) == steps, name + ": TERMINAL step id == number of answered decisions");
                    return ScriptTransport.Quit();
                }
                if (msg != MsgType.Obs)
                {
                    Expect(false, name + ": unexpected message type " + msg);
                    return ScriptTransport.Quit();
                }
                obs++;
                Check(rules, last, n, undo, mendId, name + " step " + ScriptTransport.StepOf(last));
                FrameCheck.Problems(last, out FrameCheck.View v);
                long node = ((long)v.G(Gl.ActIndex) << 40) | ((long)(v.G(Gl.CurRow) + 1) << 20) | (uint)(v.G(Gl.CurCol) + 1);
                if (node != curNode)
                {
                    if (curNode != long.MinValue && travelsHere > 0)
                    {
                        travelsPerLocation.Add(travelsHere);
                    }
                    curNode = node;
                    travelsHere = 0;
                }
                int k = rng.Next(ScriptTransport.NActions(last));
                if (v.G(Gl.Kind) == (int)DecisionKind.Map)
                {
                    if (travelsHere > 0 && !everySeat)
                    {
                        mapFramesAfterTravel++;
                    }
                    if (v.Get(T.Actions, k, ActionC.Verb) == (int)Verb.Travel)
                    {
                        travelsHere++;
                    }
                }
                steps++;
                return ScriptTransport.Step(ScriptTransport.StepOf(last), k);
            });
            ProtocolServer server = new ProtocolServer(session, t, ObsRuntime.NewEncoder(l), new ServeOptions());
            int exit = server.Serve();
            Expect(exit == 0 && server.Episodes == 1 && terminal == 1 && server.Aborts == 0, $"{name}: exit {exit}, {server.Episodes} episode, {terminal} TERMINAL, {server.Aborts} aborted");
            Expect(rules.Problems.Count == 0, name + ": frame rules: " + string.Join("; ", rules.Examples.Take(4)));
            Expect(obs > 50, name + ": more than 50 OBS frames (" + obs + ")");
            if (!everySeat)
            {
                Expect(mapFramesAfterTravel == 0, name + ": team votes - no MAP frame after the team's TRAVEL at the same location (" + mapFramesAfterTravel + ")");
                Expect(travelsPerLocation.All(x => x == 1), name + ": team votes - one TRAVEL decision per location");
                Expect(rules.MapVoteBits == 0, name + ": team votes - no map vote is ever pending in a frame");
            }
            else
            {
                // (a seat whose MAP frame has one legal action is auto-advanced by the host and never reaches the client, so fewer than n is possible)
                Expect(travelsPerLocation.Count > 0 && travelsPerLocation.All(x => x <= n) && travelsPerLocation.Any(x => x == n), name + ": every seat asked - up to " + n + " TRAVEL decisions per location, " + n + " at least once (" + string.Join(",", travelsPerLocation.Take(8)) + ")");
                Expect(rules.MapVoteBits > 0, name + ": every seat asked - earlier map votes are visible");
            }
            Expect(undo ? rules.UndoRows > 0 : rules.UndoRows == 0, name + ": UNDO_END_TURN rows " + rules.UndoRows + " (option " + undo + ")");
            facts[name] = new { obs_frames = obs, steps, locations_left = travelsPerLocation.Count, undo_rows = rules.UndoRows, map_vote_bits = rules.MapVoteBits, mean_frame_bytes = rules.Frames == 0 ? 0 : rules.FrameBytes / rules.Frames, by_kind_and_seat = rules.KindBySeat };
        }
        // ---- 4: one server process: 1 seat, 4 seats, the same 1-seat episode again (the stale-lobby reset, ruling Q10) ----
        {
            List<byte[]>[] frames = { new List<byte[]>(), new List<byte[]>(), new List<byte[]>() };
            int episode = -1;
            int steps = 0;
            Random rng = new Random(1);
            const int Cap = 120;
            ScriptTransport t = new ScriptTransport(tr =>
            {
                byte[]? last = tr.Last;
                bool ended = last != null && (ScriptTransport.Msg(last) == MsgType.Terminal || ScriptTransport.Msg(last) == MsgType.Error);
                if (last != null && episode >= 0)
                {
                    frames[episode].Add(last);
                }
                if (last == null || ended)
                {
                    episode++;
                    steps = 0;
                    rng = new Random(77); // the same choices in episodes 0 and 2
                    switch (episode)
                    {
                        case 0:
                        case 2:
                            return ScriptTransport.Reset(four[0], 10, SeedCodec.Encode("SELFTESTMP05"), maxDecisions: Cap);
                        case 1:
                            return Reset(four, SeedCodec.Encode("SELFTESTMP06"), ResetFlag.Default, 300);
                        default:
                            return ScriptTransport.Quit();
                    }
                }
                steps++;
                return ScriptTransport.Step(ScriptTransport.StepOf(last), rng.Next(ScriptTransport.NActions(last)));
            });
            ProtocolServer server = new ProtocolServer(session, t, ObsRuntime.NewEncoder(l), new ServeOptions());
            int exit = server.Serve();
            Expect(exit == 0 && server.Episodes == 3 && server.Aborts == 0, $"mixed process: exit {exit}, {server.Episodes} episodes, {server.Aborts} aborted");
            bool same = frames[0].Count == frames[2].Count && frames[0].Count > 10;
            int firstDiff = -1;
            for (int i = 0; same && i < frames[0].Count; i++)
            {
                byte[] a = (byte[])frames[0][i].Clone(), b = (byte[])frames[2][i].Clone();
                // EPISODE_ID differs by construction (1 vs 3); HOST_US is a timing
                System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(a.AsSpan(Hd.EpisodeId * 4), 0);
                System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(Hd.EpisodeId * 4), 0);
                if (!EncodeChecker.SameIgnoringHostUs(a, b))
                {
                    same = false;
                    firstDiff = i;
                }
            }
            Expect(same, $"mixed process: the single-seat episode after a 4-seat episode is byte-identical to the one before it ({frames[0].Count} vs {frames[2].Count} frames, first difference at frame {firstDiff})");
            Expect(frames[1].Count > 10 && frames[1].All(f => ScriptTransport.Msg(f) != MsgType.Obs || ScriptTransport.GlobalOf(f, Gl.NSeats) == 4), "mixed process: the middle episode has 4 seats");
            Expect(frames[0].All(f => ScriptTransport.Msg(f) == MsgType.Error || Enumerable.Range(Gl.SeatScreen0, 8).All(g => ScriptTransport.GlobalOf(f, g) == 0)), "single-seat frames: GLOBAL 52-59 are 0");
            facts["mixed_process"] = new { single_seat_frames = frames[0].Count, four_seat_frames = frames[1].Count, identical = same };
        }
        ok = fails.Count == 0;
        return new { failures = fails, facts };
    }
}
