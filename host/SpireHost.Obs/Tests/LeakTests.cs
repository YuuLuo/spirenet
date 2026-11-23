using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Schema;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs.Tests;

/// <summary>
/// TEST ONLY - writes hidden state on purpose (obs-schema-v1 14.1 / 14.2). At sampled decision points of real episodes:
///   B0 = Encode(); apply a hidden perturbation; B1 = Encode(); undo it; B2 = Encode();   require B0 == B1 == B2
/// (header EPISODE_ID / STEP_ID / AUTO_STEPS / HOST_US ignored), for the real decision frame AND - in combat - a
/// synthetic draw-pile CARD_SELECT prompt whose candidate classes are rebuilt after the perturbation (P14). Positive
/// controls apply a visible change and require B1 != B0, B2 == B0. A DeepSignature before / after the whole probe proves
/// every perturbation was undone exactly (otherwise the rest of the episode would not be the same game).
///
/// Perturbations (single seat; the decisions come from the REAL handlers of the production set):
///   P1  draw pile order                     P2  discard / exhaust order          P3  every run / player / monster Rng replaced
///   P4  monster NextMove replaced by a move with the same intents + state log / first-move flag altered
///   P5  room sets of every act shuffled, visited counters changed                 P6  relic grab bags shuffled, odds changed
///   P7  REWARD: contents of unopened card rewards removed                        P8  power Amount / internal data where the
///       displayed amount and visibility do not change                            P9  private int / bool fields of relics
///       whose counter is not shown, where every displayed relic value stays the same
///   P10 unprinted DynamicVars of cards (cards without CalculatedVar)              P13 run / player seeds
///   P14 draw-pile CARD_SELECT candidates after P1 (classes rebuilt by the registered class provider, every probe)
///   P07b TREASURE CHEST page (LeakTests.ChestAb, an A/B pair of whole scenario episodes on the real treasure handler,
///        not an in-place probe: the frame is rebuilt by the handler in both): the relic inside the unopened chest replaced
///   P11 CRYSTAL: the items under hidden cells removed / planted (the context's public view is rebuilt before encoding)
///   positive controls C1-C6 (C6: revealing one more crystal cell)
/// Not buildable in H1: P12 SEAT views (one seat).
///
/// TEAM view with 2-4 seats (phase H5; <c>new LeakProbe(..., team: true)</c>, run by LeakTests.ExecuteTeam): P1-P13 as
/// above - they already cover every seat (piles, player RNG streams, relics, cards of all players) - plus what only
/// exists with several players. Hidden from EVERY player, so it must never reach a TEAM frame:
///   P15 the vote RNGs: map_point_selection (MapSelectionSynchronizer), event_synchronizer (EventSynchronizer) and the
///       treasure synchronizer's own reference to the relic stream
///   P16 the contents of every seat's unopened card rewards (all open reward sets of all seats, not only the one asked)
/// Private to a seat and not on its screen at this decision (the controller is shown a seat's shop / rest options only
/// in that seat's own decision, exactly as that player's client shows them):
///   P17 the merchant inventories of the seats whose SHOP screen is not the one being asked (prices changed)
///   P18 the rest-site option lists of the seats whose REST screen is not the one being asked
/// Positive controls: C7 a teammate's hand (a card moved to its discard pile), C8 a teammate's gold, C9 a teammate's
/// public status (GLOBAL 56-59), C10 a teammate's map vote (MAP frames).
/// </summary>
public sealed class LeakProbe : IEpisodeObserver
{
    public sealed class Stat
    {
        public bool Control;
        public long Applied;
        public long Passed;
        public long Failed;
        public long NotApplicable;
        public readonly List<string> Examples = new List<string>();

        /// <summary>For a failed perturbation that supports attribution: which single mutation changed the frame, how often.</summary>
        public readonly SortedDictionary<string, long> Culprits = new SortedDictionary<string, long>(StringComparer.Ordinal);
    }

    private delegate Action? Perturbation(RunContext ctx, Decision d);

    private readonly ObsEncoder _enc;
    private readonly Templates _tpl;
    private readonly System.Random _rng;
    private readonly List<(string name, bool control, Perturbation p)> _perts = new List<(string, bool, Perturbation)>();
    public readonly SortedDictionary<string, Stat> Stats = new SortedDictionary<string, Stat>(StringComparer.Ordinal);
    public int Every = 5;
    public long Probes;
    public long NotRestored;
    public readonly List<string> NotRestoredExamples = new List<string>();
    private long _n;
    private int _episode;
    private string _note = "";

    public LeakProbe(ObsEncoder encoder, Templates templates, int seed, bool team = false)
    {
        _enc = encoder;
        _tpl = templates;
        _rng = new System.Random(seed);
        _team = team;
        if (team)
        {
            Add("P15_vote_rngs", false, VoteRngs);
            Add("P16_all_seats_unopened_card_rewards", false, AllSeatsUnopenedCardRewards);
            Add("P17_other_seats_shop_inventories", false, OtherSeatsShopInventories);
            Add("P18_other_seats_rest_options", false, OtherSeatsRestOptions);
            Add("C7_teammate_hand", true, TeammateHand);
            Add("C8_teammate_gold", true, TeammateGold);
            Add("C9_teammate_status", true, TeammateStatus);
            Add("C10_teammate_map_vote", true, TeammateMapVote);
        }
        Add("P01_draw_order", false, DrawOrder);
        Add("P02_discard_exhaust_order", false, DiscardExhaustOrder);
        Add("P03_rng_streams", false, RngStreams);
        Add("P04_monster_move_state", false, MonsterMoveState);
        Add("P05_room_sets", false, RoomSets);
        Add("P06_grab_bags_odds", false, GrabBagsAndOdds);
        Add("P07_unopened_card_reward", false, UnopenedCardRewards);
        Add("P11_crystal_hidden_items", false, CrystalHiddenItems);
        Add("P08_power_hidden_amount", false, PowerHiddenAmount);
        Add("P09_relic_hidden_fields", false, RelicHiddenFields);
        Add("P10_unprinted_card_vars", false, UnprintedCardVars);
        Add("P13_seeds", false, Seeds);
        Add("C1_enemy_block", true, EnemyBlock);
        Add("C2_draw_to_discard", true, DrawToDiscard);
        Add("C3_gold", true, Gold);
        Add("C4_printed_card_var", true, PrintedCardVar);
        Add("C5_visible_power_amount", true, VisiblePowerAmount);
        Add("C6_crystal_reveal_cell", true, CrystalRevealCell);
    }

    /// <summary>The single mutations of the perturbation that is being applied (filled by perturbations that support attribution).</summary>
    private readonly List<(string name, Action apply, Action undo)> _items = new List<(string, Action, Action)>();

    private void Add(string name, bool control, Perturbation p)
    {
        _perts.Add((name, control, p));
        Stats[name] = new Stat { Control = control };
    }

    public void OnEpisodeStart(RunContext ctx)
    {
        _episode++;
    }

    public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
    {
        if (_n++ % Every != 0)
        {
            return;
        }
        Probe(ctx, decision);
    }

    public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
    {
    }

    public void OnEpisodeEnd(EpisodeResult result)
    {
    }

    // ---------------------------------------------------------------------------------------------- probe

    private void Probe(RunContext ctx, Decision d)
    {
        Probes++;
        ulong sig0 = DeepSignature.Compute(ctx.Run);
        byte[] b0 = Encode(ctx, d);
        byte[]? s0 = EncodeDrawPrompt(ctx);
        foreach ((string name, bool control, Perturbation p) in _perts)
        {
            Stat st = Stats[name];
            _note = "";
            Action? undo = p(ctx, d);
            if (undo == null)
            {
                st.NotApplicable++;
                continue;
            }
            st.Applied++;
            List<(string name, Action apply, Action undo)> items = new List<(string, Action, Action)>(_items);
            _items.Clear();
            byte[] b1 = Encode(ctx, d);
            byte[]? s1 = EncodeDrawPrompt(ctx);
            undo();
            byte[] b2 = Encode(ctx, d);
            byte[]? s2 = EncodeDrawPrompt(ctx);
            bool restored = Same(b0, b2) && Same(s0, s2);
            bool ok = control ? restored && !Same(b0, b1) : restored && Same(b0, b1) && Same(s0, s1);
            if (ok)
            {
                st.Passed++;
            }
            else
            {
                st.Failed++;
                // attribution: which single item of the perturbation changes the bytes
                List<string> culprits = new List<string>();
                foreach ((string item, Action apply, Action undoItem) in items)
                {
                    apply();
                    bool differs = !Same(b0, Encode(ctx, d)) || !Same(s0, EncodeDrawPrompt(ctx));
                    undoItem();
                    if (differs)
                    {
                        culprits.Add(item);
                        st.Culprits[item] = st.Culprits.GetValueOrDefault(item) + 1;
                    }
                }
                if (st.Examples.Count < 6)
                {
                    st.Examples.Add($"episode {_episode} seed {ctx.Run.Rng.StringSeed} {d.Kind} decision {ctx.Counters.DecisionPoints}: restored={restored} frame_same={Same(b0, b1)} prompt_same={Same(s0, s1)} {FirstDiff(b0, b1)} " +
                                    (culprits.Count > 0 ? "culprits: " + string.Join(", ", culprits) : _note) + " cells: " + CellDiffs(b0, b1));
                }
            }
        }
        if (DeepSignature.Compute(ctx.Run) != sig0)
        {
            NotRestored++;
            if (NotRestoredExamples.Count < 6)
            {
                NotRestoredExamples.Add($"episode {_episode} {d.Kind} decision {ctx.Counters.DecisionPoints}");
            }
        }
    }

    private byte[] Encode(RunContext ctx, Decision d)
    {
        // A context that carries a DERIVED public view is rebuilt from the (possibly perturbed) game state first, exactly as
        // its frame does on every Build - otherwise the probe would only compare a cached view with itself.
        if (d.Context is CrystalContext crystal)
        {
            crystal.FillPublicView();
        }
        _enc.EncodeDecision(d, _episode, 0, 0);
        return _enc.Payload.ToArray();
    }

    /// <summary>A CARD_SELECT prompt over the live draw pile (classes from PublicCardKey, as the CARD_SELECT owner must build them).</summary>
    private byte[]? EncodeDrawPrompt(RunContext ctx)
    {
        PlayerCombatState? pcs = ctx.LocalPlayer.PlayerCombatState;
        if (!ctx.CombatManager.IsInProgress || pcs == null || pcs.DrawPile.Cards.Count == 0)
        {
            return null;
        }
        List<CardModel> classes = new List<CardModel>();
        List<int> counts = new List<int>();
        _enc.CanonicalClasses(ctx, pcs.DrawPile.Cards, PileType.Draw, 0, false, classes, counts);
        CardSelectContext sc = new CardSelectContext
        {
            Kind = CardSelectKind.CombatPile,
            Player = ctx.LocalPlayer,
            Options = classes,
            OptionCounts = counts,
            MinSelect = 1,
            MaxSelect = 1,
            Pile = PileType.Draw,
            InCombat = true,
        };
        Decision d = new Decision(DecisionKind.CardSelect, null, 0, (int)CardSelectKind.CombatPile, sc) { Episode = ctx };
        for (int i = 0; i < classes.Count; i++)
        {
            d.Add(Verb.SelectCard, 0, GameRef.OfCard(classes[i], i));
        }
        _enc.EncodeDecision(d, _episode, 0, 0);
        return _enc.Payload.ToArray();
    }

    private static readonly int[] IgnoredWords = { Hd.EpisodeId, Hd.StepId, Hd.AutoSteps, Hd.HostUs };

    public static bool Same(byte[]? a, byte[]? b)
    {
        if (a == null || b == null)
        {
            return a == b;
        }
        if (a.Length != b.Length)
        {
            return false;
        }
        for (int i = 0; i < a.Length; i++)
        {
            if (i < SchemaV1.HeaderLen * 4 && Array.IndexOf(IgnoredWords, i / 4) >= 0)
            {
                continue;
            }
            if (a[i] != b[i])
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Up to 8 differing table cells of two frames with the same table shapes, as TABLE[row].col: before -> after; a CARDS
    /// row also names its zone, seat and card id (vocabulary index) so that the report says which card shows the change.
    /// </summary>
    private static string CellDiffs(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
        {
            return "(lengths differ)";
        }
        FrameCheck.Problems(a, out FrameCheck.View va);
        FrameCheck.Problems(b, out FrameCheck.View vb);
        List<string> o = new List<string>();
        for (int t = 0; t < 16 && o.Count < 8; t++)
        {
            if (va.Rows[t] != vb.Rows[t])
            {
                o.Add($"{SchemaV1.TableNames[t]} rows {va.Rows[t]} -> {vb.Rows[t]}");
                continue;
            }
            int nc = SchemaV1.NCols[t];
            for (int r = 0; r < va.Rows[t] && o.Count < 8; r++)
            {
                for (int c = 0; c < nc && o.Count < 8; c++)
                {
                    int x = va.Get(t, r, c), y = vb.Get(t, r, c);
                    if (x == y)
                    {
                        continue;
                    }
                    string where = t == T.Cards ? $" (zone {va.Get(t, r, CardC.Zone)} seat {va.Get(t, r, CardC.Seat)} card {va.Get(t, r, CardC.CardId)})" : "";
                    o.Add($"{SchemaV1.TableNames[t]}[{r}].col{c}: {x} -> {y}{where}");
                }
            }
        }
        return string.Join("; ", o);
    }

    /// <summary>Where two frames first differ (table name and row), for the report.</summary>
    private static string FirstDiff(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
        {
            return $"length {a.Length} vs {b.Length}";
        }
        for (int i = SchemaV1.HeaderLen * 4; i < a.Length; i++)
        {
            if (a[i] == b[i])
            {
                continue;
            }
            int w = i / 4;
            if (w < SchemaV1.DirOffsetWords)
            {
                return "GLOBAL[" + (w - SchemaV1.GlobalOffsetWords) + "]";
            }
            if (w < SchemaV1.DataOffsetWords)
            {
                return "directory of " + SchemaV1.TableNames[(w - SchemaV1.DirOffsetWords) / 5];
            }
            FrameCheck.Problems(a, out FrameCheck.View v);
            for (int t = 0; t < 16; t++)
            {
                int n = v.Rows[t] * SchemaV1.NCols[t];
                if (w >= v.Off[t] && w < v.Off[t] + n)
                {
                    int rel = w - v.Off[t];
                    return $"{SchemaV1.TableNames[t]}[{rel / SchemaV1.NCols[t]}].col{rel % SchemaV1.NCols[t]}";
                }
            }
            return "word " + w;
        }
        return "header only";
    }

    // ---------------------------------------------------------------------------------------------- helpers

    private void Shuffle<T>(List<T> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = _rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static Action RestoreList<T>(List<T> list)
    {
        List<T> copy = new List<T>(list);
        return () =>
        {
            list.Clear();
            list.AddRange(copy);
        };
    }

    private static Action? Combine(List<Action> undo)
    {
        if (undo.Count == 0)
        {
            return null;
        }
        return () =>
        {
            for (int i = undo.Count - 1; i >= 0; i--)
            {
                undo[i]();
            }
        };
    }

    private static IEnumerable<CardModel> AllCards(Player p)
    {
        foreach (CardModel c in p.Deck.Cards)
        {
            yield return c;
        }
        PlayerCombatState? pcs = p.PlayerCombatState;
        if (pcs != null)
        {
            foreach (CardPile pile in pcs.AllPiles)
            {
                foreach (CardModel c in pile.Cards)
                {
                    yield return c;
                }
            }
        }
    }

    private static CombatState? Combat(RunContext ctx) => ctx.CombatManager.IsInProgress && ctx.Run.CurrentRoom is CombatRoom r ? r.CombatState : null;

    // ---------------------------------------------------------------------------------------------- P1 / P2 / P14

    private Action? ShufflePiles(RunContext ctx, params PileType[] piles)
    {
        List<Action> undo = new List<Action>();
        foreach (Player p in ctx.Run.Players)
        {
            PlayerCombatState? pcs = p.PlayerCombatState;
            if (pcs == null || !ctx.CombatManager.IsInProgress)
            {
                continue;
            }
            foreach (PileType t in piles)
            {
                CardPile pile = t == PileType.Draw ? pcs.DrawPile : t == PileType.Discard ? pcs.DiscardPile : pcs.ExhaustPile;
                if (pile._cards.Count < 2)
                {
                    continue;
                }
                undo.Add(RestoreList(pile._cards));
                pile._cards.Reverse(); // guaranteed different order, then random
                Shuffle(pile._cards);
            }
        }
        return Combine(undo);
    }

    private Action? DrawOrder(RunContext ctx, Decision d) => ShufflePiles(ctx, PileType.Draw);

    private Action? DiscardExhaustOrder(RunContext ctx, Decision d) => ShufflePiles(ctx, PileType.Discard, PileType.Exhaust);

    // ---------------------------------------------------------------------------------------------- P3

    private Action? RngStreams(RunContext ctx, Decision d)
    {
        List<Action> undo = new List<Action>();
        void Replace<TK>(Dictionary<TK, Rng> rngs) where TK : notnull
        {
            foreach (TK k in rngs.Keys.ToList())
            {
                Rng old = rngs[k];
                rngs[k] = new Rng((ulong)_rng.NextInt64());
                undo.Add(() => rngs[k] = old);
            }
        }
        Replace(ctx.Run.Rng._rngs);
        foreach (Player p in ctx.Run.Players)
        {
            Replace(p.PlayerRng._rngs);
        }
        CombatState? cs = Combat(ctx);
        if (cs != null)
        {
            foreach (Creature e in cs._enemies)
            {
                MonsterModel? m = e.Monster;
                if (m != null && m._rng != null)
                {
                    Rng old = m._rng;
                    m._rng = new Rng((ulong)_rng.NextInt64());
                    undo.Add(() => m._rng = old);
                }
            }
        }
        return Combine(undo);
    }

    // ---------------------------------------------------------------------------------------------- P4

    private Action? MonsterMoveState(RunContext ctx, Decision d)
    {
        CombatState? cs = Combat(ctx);
        if (cs == null)
        {
            return null;
        }
        List<Action> undo = new List<Action>();
        foreach (Creature e in cs._enemies)
        {
            MonsterModel? m = e.Monster;
            if (m == null || !e.IsAlive)
            {
                continue;
            }
            MoveState old = m.NextMove;
            AbstractIntent[] intents = old.Intents.ToArray();
            m.NextMove = new MoveState("OBS_LEAKTEST_" + _rng.Next(), _ => Task.CompletedTask, intents) { FollowUpState = null };
            undo.Add(() => m.NextMove = old);
            MonsterMoveStateMachine? sm = m._moveStateMachine;
            if (sm != null)
            {
                bool first = sm._performedFirstMove;
                sm._performedFirstMove = !first;
                List<MonsterState> log = sm.StateLog;
                undo.Add(RestoreList(log));
                if (log.Count > 0)
                {
                    log.Add(log[0]);
                }
                Shuffle(log);
                undo.Add(() => sm._performedFirstMove = first);
            }
        }
        return Combine(undo);
    }

    // ---------------------------------------------------------------------------------------------- P5 / P6

    private Action? RoomSets(RunContext ctx, Decision d)
    {
        List<Action> undo = new List<Action>();
        foreach (ActModel act in ctx.Run.Acts)
        {
            RoomSet? rs = act._rooms;
            if (rs == null)
            {
                continue;
            }
            undo.Add(RestoreList(rs.events));
            undo.Add(RestoreList(rs.normalEncounters));
            undo.Add(RestoreList(rs.eliteEncounters));
            Shuffle(rs.events);
            Shuffle(rs.normalEncounters);
            Shuffle(rs.eliteEncounters);
            (int a, int b, int c, int e) = (rs.eventsVisited, rs.normalEncountersVisited, rs.eliteEncountersVisited, rs.bossEncountersVisited);
            rs.eventsVisited += 3;
            rs.normalEncountersVisited += 2;
            rs.eliteEncountersVisited += 1;
            undo.Add(() => (rs.eventsVisited, rs.normalEncountersVisited, rs.eliteEncountersVisited, rs.bossEncountersVisited) = (a, b, c, e));
        }
        return Combine(undo);
    }

    private Action? GrabBagsAndOdds(RunContext ctx, Decision d)
    {
        List<Action> undo = new List<Action>();
        void Bag(RelicGrabBag bag)
        {
            foreach (List<RelicModel> l in bag._deques.Values)
            {
                undo.Add(RestoreList(l));
                Shuffle(l);
            }
            undo.Add(RestoreList(bag._mpFallbackDequeue));
            Shuffle(bag._mpFallbackDequeue);
        }
        Bag(ctx.Run.SharedRelicGrabBag);
        foreach (Player p in ctx.Run.Players)
        {
            Bag(p.RelicGrabBag);
            float r = p.PlayerOdds.CardRarity.CurrentValue, q = p.PlayerOdds.PotionReward.CurrentValue;
            p.PlayerOdds.CardRarity.OverrideCurrentValue(r + 0.25f);
            p.PlayerOdds.PotionReward.OverrideCurrentValue(q + 0.25f);
            undo.Add(() =>
            {
                p.PlayerOdds.CardRarity.OverrideCurrentValue(r);
                p.PlayerOdds.PotionReward.OverrideCurrentValue(q);
            });
        }
        float u = ctx.Run.Odds.UnknownMapPoint.CurrentValue;
        ctx.Run.Odds.UnknownMapPoint.OverrideCurrentValue(u + 0.25f);
        undo.Add(() => ctx.Run.Odds.UnknownMapPoint.OverrideCurrentValue(u));
        return Combine(undo);
    }

    // ---------------------------------------------------------------------------------------------- P7

    private Action? UnopenedCardRewards(RunContext ctx, Decision d)
    {
        if (d.Context is not RewardContext rc)
        {
            return null;
        }
        List<Action> undo = new List<Action>();
        foreach (Reward r in rc.Set.Rewards)
        {
            if (r is CardReward cr && !cr.SuccessfullySelected && cr._cards.Count > 0)
            {
                undo.Add(RestoreList(cr._cards));
                cr._cards.Reverse();
                cr._cards.RemoveAt(0); // contents changed, not just the order
            }
        }
        return Combine(undo);
    }

    /// <summary>
    /// P11 - a real CRYSTAL frame (events module): the items under HIDDEN cells are hidden information. Every hidden
    /// cell's item is removed (or, where there is none, a curse item is planted); the frame must not change.
    /// </summary>
    private Action? CrystalHiddenItems(RunContext ctx, Decision d)
    {
        if (d.Context is not CrystalContext cc)
        {
            return null;
        }
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereCell[,] cells = cc.Minigame.cells;
        List<Action> undo = new List<Action>();
        MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItem planted = new MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItems.CrystalSphereCurse();
        foreach (MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereCell c in cells)
        {
            if (!c.IsHidden)
            {
                continue;
            }
            MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereItem? old = c.Item;
            c.Item = old == null ? planted : null;
            undo.Add(() => c.Item = old);
        }
        return Combine(undo);
    }

    /// <summary>C6 (positive control of P11): revealing one more cell of a real CRYSTAL frame changes the frame.</summary>
    private Action? CrystalRevealCell(RunContext ctx, Decision d)
    {
        if (d.Context is not CrystalContext cc)
        {
            return null;
        }
        foreach (MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereCell c in cc.Minigame.cells)
        {
            if (c.IsHidden)
            {
                c.IsHidden = false;
                return () => c.IsHidden = true;
            }
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------- P8 / P9

    /// <summary>
    /// What the real UI DISPLAYS on the face-up cards of every seat, computed with the game's own display routines only
    /// (never with the encoder). HAND: playable or not and why (CardModel.CanPlay, the hand's highlight), energy and star
    /// cost with modifiers, keywords, and the numbers of the card text as NCard prints them
    /// (CardModel.UpdateDynamicVarPreview -> PreviewValue; the values it writes are restored). PLAY (session 3, finding F13):
    /// the card that is being played sits in the play container face up and NCard prints its cost with modifiers for every
    /// pile (NCard.UpdateVisuals(PileType.Play, ...) NCard.cs:1164, UpdateEnergyCostVisuals NCard.cs:670-690; only the
    /// "unplayable" icon is hand-only, NCard.cs:692), so the same values without CanPlay. A hook such as VoidFormPower
    /// (TryModifyEnergyCostInCombatLate for Hand AND Play cards, Models.Powers/VoidFormPower.cs ShouldSkip) shows its
    /// private state there. Private state of a power or relic that changes any of this through a hook is displayed
    /// state: perturbing it is not a leak test.
    /// </summary>
    private static string CardDisplay(RunContext ctx, bool withPlay = true)
    {
        if (!ctx.CombatManager.IsInProgress)
        {
            return "";
        }
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        foreach (Player p in ctx.Run.Players)
        {
            PlayerCombatState? pcs = p.PlayerCombatState;
            if (pcs == null)
            {
                continue;
            }
            foreach (CardModel c in pcs.Hand.Cards)
            {
                bool can = c.CanPlay(out UnplayableReason reason, out AbstractModel? _);
                sb.Append(can ? 'P' : 'u').Append((int)reason).Append(' ');
                AppendShownValues(sb, c);
            }
            if (withPlay)
            {
                sb.Append("play:");
                foreach (CardModel c in pcs.PlayPile.Cards)
                {
                    AppendShownValues(sb, c);
                }
            }
            sb.Append("|\n");
        }
        return sb.ToString();
    }

    private static void AppendShownValues(System.Text.StringBuilder sb, CardModel c)
    {
        sb.Append(c.EnergyCost.GetWithModifiers(CostModifiers.All)).Append(' ').Append(c.GetStarCostWithModifiers()).Append(' ');
        foreach (CardKeyword k in c.Keywords)
        {
            sb.Append((int)k).Append(',');
        }
        DynamicVarSet set = c.DynamicVars;
        List<(decimal preview, decimal enchanted)> saved = set.Values.Select(v => (v.PreviewValue, v.EnchantedValue)).ToList();
        try
        {
            set.ClearPreview();
            c.UpdateDynamicVarPreview(CardPreviewMode.Normal, null, set);
            foreach (DynamicVar v in set.Values)
            {
                sb.Append(v.Name).Append('=').Append(v.PreviewValue).Append(';');
            }
        }
        finally
        {
            int i = 0;
            foreach (DynamicVar v in set.Values)
            {
                v.PreviewValue = saved[i].preview;
                v.EnchantedValue = saved[i].enchanted;
                i++;
            }
        }
        sb.Append('\n');
    }

    /// <summary>
    /// Counts a skip of the "displayed through cards" criterion; when the HAND alone shows no change (only a card in a
    /// play pile does), the key is also counted in <see cref="DisplayedThroughPlayPile"/>.
    /// </summary>
    private void NoteDisplayedThroughCards(RunContext ctx, string key, string handOnlyBefore, string playBefore)
    {
        DisplayedThroughCards[key] = DisplayedThroughCards.GetValueOrDefault(key) + 1;
        if (CardDisplay(ctx, withPlay: false) == handOnlyBefore)
        {
            DisplayedThroughPlayPile[key] = DisplayedThroughPlayPile.GetValueOrDefault(key) + 1;
            if (PlayPileExamples.Count < 6)
            {
                PlayPileExamples.Add($"{key} episode {_episode} seed {ctx.Run.Rng.StringSeed} decision {ctx.Counters.DecisionPoints}: play piles [{playBefore}] -> [{PlayPiles(ctx)}]");
            }
        }
    }

    /// <summary>Every seat's play pile as the UI prints it: card id, upgrade level, energy and star cost with modifiers.</summary>
    private static string PlayPiles(RunContext ctx)
    {
        if (!ctx.CombatManager.IsInProgress)
        {
            return "";
        }
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        IReadOnlyList<Player> players = ctx.Run.Players;
        for (int s = 0; s < players.Count; s++)
        {
            PlayerCombatState? pcs = players[s].PlayerCombatState;
            if (pcs == null)
            {
                continue;
            }
            foreach (CardModel c in pcs.PlayPile.Cards)
            {
                sb.Append("seat ").Append(s).Append(' ').Append(c.Id.Entry).Append('+').Append(c.CurrentUpgradeLevel)
                  .Append(" cost ").Append(c.EnergyCost.GetWithModifiers(CostModifiers.All)).Append(" star ").Append(c.GetStarCostWithModifiers())
                  .Append(" hand ").Append(pcs.Hand.Cards.Count).Append("; ");
            }
        }
        return sb.ToString();
    }

    public readonly List<string> PlayPileExamples = new List<string>();

    /// <summary>Power data / relic fields kept from being perturbed ONLY because a card in a play pile shows them (finding F13).</summary>
    public readonly SortedDictionary<string, long> DisplayedThroughPlayPile = new SortedDictionary<string, long>(StringComparer.Ordinal);

    private readonly bool _team;

    /// <summary>TEAM probe: power state left unperturbed because the intent numbers the game shows to some player depend on it.</summary>
    public readonly SortedDictionary<string, long> DisplayedThroughIntents = new SortedDictionary<string, long>(StringComparer.Ordinal);

    /// <summary>
    /// What the real UI DISPLAYS on the enemies' attack intents to each player, computed with the game's own routine
    /// (AttackIntent.GetSingleDamage against LocalContext.GetMe, MonsterMoves.Intents/AttackIntent.cs:84-93; the local id
    /// is switched per player and restored). With several players a power can redirect or change the damage a teammate
    /// is shown (InterceptPower: the covered creatures): such state is displayed, perturbing it is not a leak test.
    /// </summary>
    private static string IntentDisplay(RunContext ctx)
    {
        CombatState? cs = Combat(ctx);
        if (cs == null)
        {
            return "";
        }
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        ulong? saved = MegaCrit.Sts2.Core.Context.LocalContext.NetId;
        try
        {
            foreach (Creature e in cs._enemies)
            {
                MonsterModel? m = e.Monster;
                if (m == null || !e.IsAlive || m.NextMove.Intents == null)
                {
                    continue;
                }
                foreach (AbstractIntent it in m.NextMove.Intents)
                {
                    if (it is AttackIntent atk)
                    {
                        foreach (Player p in ctx.Run.Players)
                        {
                            MegaCrit.Sts2.Core.Context.LocalContext.NetId = p.NetId;
                            sb.Append(atk.GetSingleDamage(Array.Empty<Creature>(), e)).Append(',');
                        }
                    }
                    sb.Append('|');
                }
            }
        }
        finally
        {
            MegaCrit.Sts2.Core.Context.LocalContext.NetId = saved;
        }
        return sb.ToString();
    }

    private Action? PowerHiddenAmount(RunContext ctx, Decision d)
    {
        CombatState? cs = Combat(ctx);
        if (cs == null)
        {
            return null;
        }
        string hand = CardDisplay(ctx);
        string handOnly = CardDisplay(ctx, withPlay: false);
        string playBefore = PlayPiles(ctx);
        string intents = _team ? IntentDisplay(ctx) : "";
        List<Action> undo = new List<Action>();
        foreach (Creature c in cs._allies.Concat(cs._enemies))
        {
            foreach (PowerModel pw in c.Powers)
            {
                int shown = pw.DisplayAmount;
                bool vis = pw.IsVisible;
                int old = pw._amount;
                // the tooltip prints {Amount} (Templates.PowerPrinted): public, not perturbed
                bool amountPrinted = Array.IndexOf(_tpl.PrintedVars(pw), Templates.PowerAmount) >= 0;
                pw._amount = old + 7;
                bool amountOnIcon = amountPrinted || pw.DisplayAmount != shown || pw.IsVisible != vis;
                if (!amountOnIcon && _team && CardDisplay(ctx) == hand && IntentDisplay(ctx) != intents)
                {
                    pw._amount = old; // displayed through an intent number of some player (TEAM probe)
                    DisplayedThroughIntents[pw.Id.Entry + "._amount"] = DisplayedThroughIntents.GetValueOrDefault(pw.Id.Entry + "._amount") + 1;
                }
                else if (amountOnIcon || CardDisplay(ctx) != hand)
                {
                    if (!amountOnIcon)
                    {
                        // verifier finding R-6: this skip was not counted, so the report did not show which amounts the hand-card
                        // criterion (which looks at every variable of a hand card) kept from being perturbed
                        NoteDisplayedThroughCards(ctx, pw.Id.Entry + "._amount", handOnly, playBefore);
                    }
                    pw._amount = old; // the amount is displayed (on the power, or through a hook on a hand / play card): not hidden
                }
                else
                {
                    undo.Add(() => pw._amount = old);
                    _note += pw.Id.Entry + "(amount) ";
                    _items.Add((pw.Id.Entry + "._amount", () => pw._amount = old + 7, () => pw._amount = old));
                }
                object? data = pw._internalData;
                if (data != null)
                {
                    object? fresh = pw.InitInternalData();
                    pw._internalData = fresh;
                    bool onIcon = pw.DisplayAmount != shown || pw.IsVisible != vis;
                    bool onCards = !onIcon && CardDisplay(ctx) != hand;
                    if (!onIcon && !onCards && _team && IntentDisplay(ctx) != intents)
                    {
                        pw._internalData = data; // drives an intent number shown to some player (InterceptPower: who is covered)
                        DisplayedThroughIntents[pw.Id.Entry + "._internalData"] = DisplayedThroughIntents.GetValueOrDefault(pw.Id.Entry + "._internalData") + 1;
                    }
                    else if (onIcon || onCards)
                    {
                        if (onCards)
                        {
                            NoteDisplayedThroughCards(ctx, pw.Id.Entry + "._internalData", handOnly, playBefore);
                        }
                        // drives something the UI shows (e.g. ChainsOfBinding: which Bound cards are still playable; VoidForm: the
                        // cost of the owner's hand / play cards)
                        pw._internalData = data;
                    }
                    else
                    {
                        undo.Add(() => pw._internalData = data);
                        _note += pw.Id.Entry + "(data) ";
                        _items.Add((pw.Id.Entry + "._internalData", () => pw._internalData = fresh, () => pw._internalData = data));
                    }
                }
            }
        }
        return Combine(undo);
    }

    private static readonly Dictionary<Type, FieldInfo[]> s_relicFields = new Dictionary<Type, FieldInfo[]>();

    private static FieldInfo[] HiddenFields(Type t)
    {
        if (s_relicFields.TryGetValue(t, out FieldInfo[]? f))
        {
            return f;
        }
        List<FieldInfo> l = new List<FieldInfo>();
        for (Type? b = t; b != null && b != typeof(RelicModel); b = b.BaseType)
        {
            foreach (FieldInfo fi in b.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.DeclaredOnly))
            {
                if (!fi.IsInitOnly && (fi.FieldType == typeof(int) || fi.FieldType == typeof(bool)))
                {
                    l.Add(fi);
                }
            }
        }
        f = l.ToArray();
        s_relicFields[t] = f;
        return f;
    }

    private string RelicDisplay(RelicModel r)
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.Append(r.ShowCounter).Append(r.Status).Append(r.IsUsedUp).Append(r.IsWax).Append(r.IsMelted);
        foreach (string n in _tpl.PrintedVars(r))
        {
            if (r.DynamicVars.TryGetValue(n, out DynamicVar? v))
            {
                sb.Append(n).Append(v.PreviewValue).Append(v.BaseValue);
            }
        }
        return sb.ToString();
    }

    /// <summary>Power data / relic fields that the perturbations left alone because the game's own hand-card display depends on them (how often).</summary>
    public readonly SortedDictionary<string, long> DisplayedThroughCards = new SortedDictionary<string, long>(StringComparer.Ordinal);

    private Action? RelicHiddenFields(RunContext ctx, Decision d)
    {
        string hand = CardDisplay(ctx);
        string handOnly = CardDisplay(ctx, withPlay: false);
        string playBefore = PlayPiles(ctx);
        List<Action> undo = new List<Action>();
        foreach (Player p in ctx.Run.Players)
        {
            foreach (RelicModel r in p.Relics)
            {
                if (r.ShowCounter)
                {
                    continue;
                }
                string shown = RelicDisplay(r);
                foreach (FieldInfo fi in HiddenFields(r.GetType()))
                {
                    object old = fi.GetValue(r)!;
                    fi.SetValue(r, fi.FieldType == typeof(int) ? (object)((int)old + 3) : !(bool)old);
                    if (RelicDisplay(r) != shown)
                    {
                        fi.SetValue(r, old); // drives something the UI shows: not hidden
                        continue;
                    }
                    if (CardDisplay(ctx) != hand)
                    {
                        NoteDisplayedThroughCards(ctx, r.Id.Entry + "." + fi.Name, handOnly, playBefore);
                        fi.SetValue(r, old); // drives a number / flag the UI prints on a hand / play card (e.g. Vambrace: doubled block): not hidden
                        continue;
                    }
                    undo.Add(() => fi.SetValue(r, old));
                    object changed = fi.GetValue(r)!;
                    _items.Add((r.Id.Entry + "." + fi.Name, () => fi.SetValue(r, changed), () => fi.SetValue(r, old)));
                }
            }
        }
        return Combine(undo);
    }

    // ---------------------------------------------------------------------------------------------- P10 / P13

    private Action? UnprintedCardVars(RunContext ctx, Decision d)
    {
        List<Action> undo = new List<Action>();
        foreach (Player p in ctx.Run.Players)
        {
            foreach (CardModel c in AllCards(p))
            {
                DynamicVarSet set = c.DynamicVars;
                if (set.Values.Any(v => v is CalculatedVar || v is CalculationBaseVar || v is CalculationExtraVar))
                {
                    continue; // formula inputs feed printed values
                }
                string[] printed = _tpl.PrintedVars(c);
                foreach (DynamicVar v in set.Values)
                {
                    if (Array.IndexOf(printed, v.Name) >= 0 || v is BoolVar || v is StringVar)
                    {
                        continue;
                    }
                    (decimal b, decimal e, decimal pv) = (v._baseValue, v._enchantedValue, v._previewValue);
                    v._baseValue = b + 3;
                    v._enchantedValue = e + 3;
                    v._previewValue = pv + 3;
                    undo.Add(() =>
                    {
                        v._baseValue = b;
                        v._enchantedValue = e;
                        v._previewValue = pv;
                    });
                }
            }
        }
        return Combine(undo);
    }

    private Action? Seeds(RunContext ctx, Decision d)
    {
        List<Action> undo = new List<Action>();
        void SetAuto(object o, string prop, object value)
        {
            FieldInfo? f = o.GetType().GetField("<" + prop + ">k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic);
            if (f == null)
            {
                return;
            }
            object? old = f.GetValue(o);
            f.SetValue(o, value);
            undo.Add(() => f.SetValue(o, old));
        }
        SetAuto(ctx.Run.Rng, nameof(RunRngSet.StringSeed), "OBSLEAKTEST1");
        SetAuto(ctx.Run.Rng, nameof(RunRngSet.Seed), (ulong)_rng.NextInt64());
        foreach (Player p in ctx.Run.Players)
        {
            SetAuto(p.PlayerRng, nameof(PlayerRngSet.Seed), (ulong)_rng.NextInt64());
        }
        return Combine(undo);
    }

    // ---------------------------------------------------------------------------------------------- TEAM view, several seats (phase H5)

    private static readonly BindingFlags AnyInstance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    private Action? ReplaceRngField(object owner, string field, List<Action> undo)
    {
        FieldInfo f = owner.GetType().GetField(field, AnyInstance) ?? throw new MissingFieldException(owner.GetType().Name, field);
        object? old = f.GetValue(owner);
        f.SetValue(owner, new Rng((ulong)_rng.NextInt64()));
        Action a = () => f.SetValue(owner, old);
        undo.Add(a);
        return a;
    }

    /// <summary>P15: the three streams that resolve votes are drawn by the host only; no player sees them.</summary>
    private Action? VoteRngs(RunContext ctx, Decision d)
    {
        List<Action> undo = new List<Action>();
        ReplaceRngField(ctx.RunManager.MapSelectionSynchronizer, "_multiplayerMapPointSelection", undo);
        ReplaceRngField(ctx.RunManager.EventSynchronizer, "_multiplayerOptionSelectionRng", undo);
        ReplaceRngField(ctx.RunManager.TreasureRoomRelicSynchronizer, "_rng", undo);
        return Combine(undo);
    }

    /// <summary>P16: every seat's unopened card rewards (the one whose screen is open in this decision is displayed, so it is left alone).</summary>
    private Action? AllSeatsUnopenedCardRewards(RunContext ctx, Decision d)
    {
        CardReward? open = (d.Context as CardRewardContext)?.Reward;
        List<Action> undo = new List<Action>();
        var states = ctx.RunManager.RewardsSetSynchronizer._rewardStates;
        for (int s = 0; s < states.Count && s < ctx.SeatCount; s++)
        {
            foreach (var entry in states[s].rewardsStack)
            {
                foreach (Reward r in entry.set.Rewards)
                {
                    if (r is CardReward cr && !ReferenceEquals(cr, open) && !cr.SuccessfullySelected && cr._cards.Count > 0)
                    {
                        undo.Add(RestoreList(cr._cards));
                        cr._cards.Reverse();
                        cr._cards.RemoveAt(0);
                    }
                }
            }
        }
        return Combine(undo);
    }

    /// <summary>P17: the inventories of the seats whose shop is not the screen of this decision.</summary>
    private Action? OtherSeatsShopInventories(RunContext ctx, Decision d)
    {
        if (ctx.Run.CurrentRoom is not MerchantRoom)
        {
            return null;
        }
        List<Action> undo = new List<Action>();
        for (int s = 0; s < ctx.SeatCount; s++)
        {
            if (d.Kind == DecisionKind.Shop && s == d.ActingSeat)
            {
                continue;
            }
            MegaCrit.Sts2.Core.Entities.Merchant.MerchantInventory inv = ctx.RunManager.MerchantSynchronizer.GetInventoryFor(ctx.Seats[s]);
            foreach (MegaCrit.Sts2.Core.Entities.Merchant.MerchantEntry e in inv.AllEntries)
            {
                int old = e._cost;
                e._cost = old + 7;
                undo.Add(() => e._cost = old);
            }
        }
        return Combine(undo);
    }

    /// <summary>P18: the rest-site options of the seats whose rest screen is not the screen of this decision.</summary>
    private Action? OtherSeatsRestOptions(RunContext ctx, Decision d)
    {
        if (ctx.Run.CurrentRoom is not RestSiteRoom)
        {
            return null;
        }
        List<Action> undo = new List<Action>();
        var sites = ctx.RunManager.RestSiteSynchronizer._restSites;
        for (int s = 0; s < sites.Count && s < ctx.SeatCount; s++)
        {
            if (d.Kind == DecisionKind.Rest && s == d.ActingSeat)
            {
                continue;
            }
            List<MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption> options = sites[s].options;
            if (options.Count == 0)
            {
                continue;
            }
            undo.Add(RestoreList(options));
            options.Reverse();
            options.RemoveAt(0);
        }
        return Combine(undo);
    }

    /// <summary>C7: a teammate's hand is on the controller's screen in TEAM view.</summary>
    private Action? TeammateHand(RunContext ctx, Decision d)
    {
        if (Combat(ctx) == null)
        {
            return null;
        }
        for (int s = 1; s < ctx.SeatCount; s++)
        {
            PlayerCombatState? pcs = ctx.Seats[s].PlayerCombatState;
            if (pcs == null || pcs.Hand._cards.Count == 0 || (d.Context is CardSelectContext sel && ReferenceEquals(sel.Player, ctx.Seats[s])))
            {
                continue;
            }
            Action a = RestoreList(pcs.Hand._cards), b = RestoreList(pcs.DiscardPile._cards);
            CardModel c = pcs.Hand._cards[^1];
            pcs.Hand._cards.RemoveAt(pcs.Hand._cards.Count - 1);
            pcs.DiscardPile._cards.Add(c);
            return () =>
            {
                a();
                b();
            };
        }
        return null;
    }

    private Action? TeammateGold(RunContext ctx, Decision d)
    {
        if (ctx.SeatCount < 2)
        {
            return null;
        }
        Player p = ctx.Seats[ctx.SeatCount - 1];
        int old = p._gold;
        p._gold = old + 1;
        return () => p._gold = old;
    }

    /// <summary>C9: a teammate's public progress (GLOBAL 56-59).</summary>
    private Action? TeammateStatus(RunContext ctx, Decision d)
    {
        if (ctx.SeatCount < 2)
        {
            return null;
        }
        const SeatStatus flag = SeatStatus.VotedTreasure;
        int seat = ctx.SeatCount - 1;
        if ((ctx.SeatStatusBits(seat) & (int)flag) != 0)
        {
            return null;
        }
        List<int> had = new List<int>();
        for (int s = 0; s < ctx.SeatCount; s++)
        {
            if ((ctx.SeatStatusBits(s) & (int)flag) != 0)
            {
                had.Add(s);
            }
        }
        ctx.SetSeatFlag(seat, flag);
        return () =>
        {
            ctx.ClearSeatFlag(flag);
            foreach (int s in had)
            {
                ctx.SetSeatFlag(s, flag);
            }
        };
    }

    /// <summary>C10: a teammate's map vote is shown on the map.</summary>
    private Action? TeammateMapVote(RunContext ctx, Decision d)
    {
        if (d.Context is not MapContext mc || mc.SeatVotes.Count < 2 || mc.Travelable.Count == 0)
        {
            return null;
        }
        for (int s = 0; s < mc.SeatVotes.Count; s++)
        {
            if (s != d.ActingSeat && mc.SeatVotes[s] == null)
            {
                int seat = s;
                mc.SeatVotes[seat] = mc.Travelable[0];
                return () => mc.SeatVotes[seat] = null;
            }
        }
        return null;
    }

    // ---------------------------------------------------------------------------------------------- positive controls

    private Action? EnemyBlock(RunContext ctx, Decision d)
    {
        CombatState? cs = Combat(ctx);
        Creature? e = cs?._enemies.FirstOrDefault(x => x.IsAlive);
        if (e == null)
        {
            return null;
        }
        int old = e._block;
        e._block = old + 5;
        return () => e._block = old;
    }

    private Action? DrawToDiscard(RunContext ctx, Decision d)
    {
        PlayerCombatState? pcs = ctx.LocalPlayer.PlayerCombatState;
        if (Combat(ctx) == null || pcs == null || pcs.DrawPile._cards.Count == 0)
        {
            return null;
        }
        Action a = RestoreList(pcs.DrawPile._cards), b = RestoreList(pcs.DiscardPile._cards);
        CardModel c = pcs.DrawPile._cards[^1];
        pcs.DrawPile._cards.RemoveAt(pcs.DrawPile._cards.Count - 1);
        pcs.DiscardPile._cards.Add(c);
        return () =>
        {
            a();
            b();
        };
    }

    private Action? Gold(RunContext ctx, Decision d)
    {
        Player p = ctx.LocalPlayer;
        int old = p._gold;
        p._gold = old + 1;
        return () => p._gold = old;
    }

    private Action? PrintedCardVar(RunContext ctx, Decision d)
    {
        PlayerCombatState? pcs = ctx.LocalPlayer.PlayerCombatState;
        IEnumerable<CardModel> cards = Combat(ctx) != null && pcs != null ? pcs.Hand.Cards : ctx.LocalPlayer.Deck.Cards;
        foreach (CardModel c in cards)
        {
            foreach (string n in _tpl.PrintedVars(c))
            {
                if (c.DynamicVars.TryGetValue(n, out DynamicVar? v) && v is not BoolVar && v is not StringVar && v is not CalculatedVar)
                {
                    (decimal b, decimal e, decimal pv) = (v._baseValue, v._enchantedValue, v._previewValue);
                    v._baseValue = b + 1;
                    return () =>
                    {
                        v._baseValue = b;
                        v._enchantedValue = e;
                        v._previewValue = pv;
                    };
                }
            }
        }
        return null;
    }

    private Action? VisiblePowerAmount(RunContext ctx, Decision d)
    {
        CombatState? cs = Combat(ctx);
        if (cs == null)
        {
            return null;
        }
        foreach (Creature c in cs._allies.Concat(cs._enemies))
        {
            foreach (PowerModel pw in c.Powers)
            {
                if (!pw.IsVisible || pw.StackType != MegaCrit.Sts2.Core.Entities.Powers.PowerStackType.Counter)
                {
                    continue;
                }
                int old = pw._amount;
                int shown = pw.DisplayAmount;
                pw._amount = old + 1;
                if (pw.DisplayAmount == shown || !pw.IsVisible)
                {
                    pw._amount = old;
                    continue;
                }
                return () => pw._amount = old;
            }
        }
        return null;
    }
}

/// <summary>
/// --leak-test [--runs N] [--every K] [--hp N] [--label L] [--policy-seed P]: the leak probe on scenario states and whole runs.
/// --label / --policy-seed choose another sample of whole runs (obs verifier finding F4: the labels were fixed, so a rerun
/// could only reproduce the same sample); --hp 0 plays them at natural HP. The defaults are the gate's sample.
/// </summary>
public static class LeakTests
{
    public static int Run(HostSession session, Args a)
    {
        bool ok;
        object result = a.GetInt("seats", 1) > 1
            ? ExecuteTeam(session, ObsModule.Current, a.GetInt("seats", 4), a.GetInt("runs", 6), a.GetInt("every", 5), a.GetInt("hp", 99999), out ok, a.Get("label", "OBS-LEAK-TEAM"), a.GetULong("policy-seed", 7900))
            : Execute(session, ObsModule.Current, a.GetInt("runs", 12), a.GetInt("every", 5), a.GetInt("hp", 99999), out ok, a.Get("label", "OBS-LEAK"), a.GetULong("policy-seed", 7700));
        HostJson.Emit(result, a.Get("out"));
        return ok ? 0 : 4;
    }

    private static async Task TryRelicFor(RunContext ctx, string id, int seat)
    {
        try
        {
            await ScenarioTools.GiveRelic(ctx, id, seat);
        }
        catch (Exception e) when (e is ArgumentException || e is InvalidOperationException || e is KeyNotFoundException)
        {
            // relic id not in v0.113 - fine for a test setup
        }
    }

    /// <summary>
    /// --leak-test --seats N (2-4): the leak probe in TEAM view on multi-seat scenario states and whole runs (phase H5).
    /// Every seat is asked for its own votes (TeamVotes off), so that MAP frames with earlier votes occur.
    /// </summary>
    public static object ExecuteTeam(HostSession session, ObsRuntime.Loaded l, int seats, int runs, int every, int hp, out bool ok, string label = "OBS-LEAK-TEAM", ulong policySeedBase = 7900)
    {
        Stopwatch wall = Stopwatch.StartNew();
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = new EncoderOptions { MapInCombat = true, DeckInCombat = true, UpgradePreviews = true, TargetPreviews = true };
        LeakProbe probe = new LeakProbe(enc, l.Templates, 4343, team: true) { Every = every };
        List<object> episodes = new List<object>();
        string[] Lineup(int index)
        {
            string[] r = new string[seats];
            for (int k = 0; k < seats; k++)
            {
                r[k] = DriverCommands.AllCharacters[(index + k) % DriverCommands.AllCharacters.Length];
            }
            return r;
        }
        bool AllMapsOpen(RunContext ctx)
        {
            for (int s = 0; s < ctx.SeatCount; s++)
            {
                if (!ctx.IsMapOpen(s))
                {
                    return false;
                }
            }
            return true;
        }
        (string name, int probeEvery, Func<RunContext, Task> setup)[] scenarios =
        {
            // relics with hidden fields, potions and cards with unprinted variables on the local seat AND on a remote seat
            ("elite", every, async ctx =>
            {
                for (int s = 0; s < ctx.SeatCount; s++) { ScenarioTools.SetHp(ctx, 9999, 9999, s); }
                foreach (int s in new[] { 0, ctx.SeatCount - 1 })
                {
                    foreach (string r in new[] { "VAJRA", "PEN_NIB", "KUNAI", "ORICHALCUM", "NUNCHAKU", "LETTER_OPENER", "HAPPY_FLOWER", "SHURIKEN", "ART_OF_WAR", "BAG_OF_MARBLES", "ANCHOR", "LANTERN" })
                    {
                        await TryRelicFor(ctx, r, s);
                    }
                }
                await ScenarioTools.GivePotion(ctx, "FIRE_POTION", ctx.SeatCount - 1);
                await ScenarioTools.JumpToRoom(ctx, RoomType.Elite);
            }),
            ("monster", every, async ctx =>
            {
                for (int s = 0; s < ctx.SeatCount; s++) { ScenarioTools.SetHp(ctx, 9999, 9999, s); }
                foreach (int s in new[] { 0, 1 })
                {
                    foreach (string c in new[] { "FINESSE", "FLASH_OF_STEEL", "BACKFLIP", "SHADOW_STEP", "ADRENALINE", "METEOR_SHOWER", "PHOTON_CUT", "TUNE", "HEIRLOOM_HAMMER", "COLOSSUS", "BLUR", "REAVE" })
                    {
                        await ScenarioTools.GiveCard(ctx, c, 0, s);
                        await ScenarioTools.GiveCard(ctx, c, 1, s);
                    }
                }
                await ScenarioTools.JumpToRoom(ctx, RoomType.Monster);
            }),
            ("shop", 1, async ctx => { for (int s = 0; s < ctx.SeatCount; s++) { ScenarioTools.SetGold(ctx, 999, s); } await ScenarioTools.JumpToRoom(ctx, RoomType.Shop); }),
            ("rest", 1, async ctx => { for (int s = 0; s < ctx.SeatCount; s++) { ScenarioTools.SetHp(ctx, 10, null, s); } await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite); }),
            ("treasure", 1, ctx => ScenarioTools.JumpToRoom(ctx, RoomType.Treasure)),
            ("crystal-sphere", 1, async ctx => { for (int s = 0; s < ctx.SeatCount; s++) { ScenarioTools.SetGold(ctx, 500, s); } await ScenarioTools.JumpToEvent(ctx, "CRYSTAL_SPHERE"); }),
        };
        int k2 = 0;
        foreach (var s in scenarios)
        {
            int saved = probe.Every;
            probe.Every = s.probeEvery;
            bool jumped = false;
            bool combat = s.name == "elite" || s.name == "monster";
            EpisodeSpec spec = new EpisodeSpec
            {
                Seed = HostSession.SeedFromLabel(label + ":scenario:" + s.name),
                Characters = Lineup(k2),
                Scenario = new Scenario
                {
                    Name = "obs-leak-team-" + s.name,
                    Setup = async ctx =>
                    {
                        await s.setup(ctx);
                        jumped = true;
                    },
                    // a room scenario ends when every seat has left the room for the map; a fight when its rewards were handled too
                    StopWhen = ctx => jumped && !ctx.CombatManager.IsInProgress && AllMapsOpen(ctx) && !combat,
                },
            };
            EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)(3300 + k2++)), new EpisodeOptions { MaxPolicyDecisions = combat ? 500 : 400, MaxCombatTurns = 25 }, probe);
            probe.Every = saved;
            episodes.Add(new { scenario = s.name, outcome = r.Outcome.ToString(), r.TruncatedBy, abort = r.Abort?.Signature });
        }
        for (int i = 0; i < runs; i++)
        {
            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel(label + ":" + i), Characters = Lineup(i), Ascension = i % 2 == 0 ? 0 : 10 };
            EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy(policySeedBase + (ulong)i), new EpisodeOptions { DebugForcedHp = hp, MaxPolicyDecisions = 2500, MaxCombatTurns = 40 }, probe);
            episodes.Add(new { seed = r.Seed, lineup = string.Join("+", spec.Characters), outcome = r.Outcome.ToString(), r.TruncatedBy, floor = r.TotalFloor, abort = r.Abort?.Signature });
        }
        // every perturbation and every control must have been applied at least once: a leak test that never ran proves nothing
        bool negativesOk = probe.Stats.Values.Where(s => !s.Control).All(s => s.Failed == 0 && s.Applied > 0);
        bool controlsOk = probe.Stats.Values.Where(s => s.Control).All(s => s.Failed == 0 && s.Applied > 0);
        bool episodesOk = true;
        ok = negativesOk && controlsOk && probe.NotRestored == 0 && episodesOk;
        return new
        {
            kind = "obs-leak-test-team",
            ok,
            seats,
            view = "TEAM",
            wall_s = Math.Round(wall.Elapsed.TotalSeconds, 1),
            probes = probe.Probes,
            every,
            not_restored = probe.NotRestored,
            not_restored_examples = probe.NotRestoredExamples,
            perturbations = probe.Stats.ToDictionary(kv => kv.Key, kv => new { kv.Value.Control, kv.Value.Applied, kv.Value.Passed, kv.Value.Failed, not_applicable = kv.Value.NotApplicable, kv.Value.Examples, kv.Value.Culprits }),
            P08_P09_displayed_through_hand_cards = probe.DisplayedThroughCards,
            P08_P09_displayed_through_play_pile_only = probe.DisplayedThroughPlayPile,
            P08_P09_play_pile_examples = probe.PlayPileExamples,
            P08_displayed_through_intents = probe.DisplayedThroughIntents,
            episodes,
        };
    }

    public static object Execute(HostSession session, ObsRuntime.Loaded l, int runs, int every, int hp, out bool ok, string label = "OBS-LEAK", ulong policySeedBase = 7700)
    {
        Stopwatch wall = Stopwatch.StartNew();
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = new EncoderOptions { MapInCombat = true, DeckInCombat = true, UpgradePreviews = true, TargetPreviews = true };
        LeakProbe probe = new LeakProbe(enc, l.Templates, 4242) { Every = every };
        List<object> episodes = new List<object>();
        // scenario states with relics, potions, powers and many enemies (the temporary handlers fill only part of the contexts)
        (string name, string ch, Func<RunContext, Task> setup)[] scenarios =
        {
            ("elite-relics", "IRONCLAD", async ctx => { ScenarioTools.SetHp(ctx, 9999, 9999); foreach (string r in new[] { "VAJRA", "PEN_NIB", "KUNAI", "ORICHALCUM", "NUNCHAKU", "LETTER_OPENER", "HAPPY_FLOWER", "SHURIKEN", "ART_OF_WAR", "BAG_OF_MARBLES", "ANCHOR", "LANTERN" }) { await TryRelic(ctx, r); } await ScenarioTools.GivePotion(ctx, "FIRE_POTION"); await ScenarioTools.JumpToRoom(ctx, RoomType.Elite); }),
            ("boss-silent", "SILENT", async ctx => { ScenarioTools.SetHp(ctx, 9999, 9999); await ScenarioTools.JumpToRoom(ctx, RoomType.Boss); }),
            ("monster-defect", "DEFECT", async ctx => { ScenarioTools.SetHp(ctx, 9999, 9999); await ScenarioTools.JumpToRoom(ctx, RoomType.Monster); }),
            ("monster-necro", "NECROBINDER", async ctx => { ScenarioTools.SetHp(ctx, 9999, 9999); await ScenarioTools.JumpToRoom(ctx, RoomType.Monster); }),
            ("elite-regent", "REGENT", async ctx => { ScenarioTools.SetHp(ctx, 9999, 9999); await ScenarioTools.JumpToRoom(ctx, RoomType.Elite); }),
            ("monster-fool", "FOOL", async ctx => { ScenarioTools.SetHp(ctx, 9999, 9999); await ScenarioTools.JumpToRoom(ctx, RoomType.Monster); }),
            ("shop", "IRONCLAD", async ctx => { ScenarioTools.SetGold(ctx, 999); await ScenarioTools.JumpToRoom(ctx, RoomType.Shop); }),
            // frames that only the real handlers produce: the Crystal Sphere minigame (CRYSTAL frames of the events module)
            ("crystal-sphere", "IRONCLAD", async ctx => { ScenarioTools.SetGold(ctx, 500); await ScenarioTools.JumpToEvent(ctx, "CRYSTAL_SPHERE"); }),
            ("crystal-sphere-2", "REGENT", async ctx => { ScenarioTools.SetGold(ctx, 500); await ScenarioTools.JumpToEvent(ctx, "CRYSTAL_SPHERE"); }),
            // cards whose model has DynamicVars the template does not print (catalog vars.printed = false): P10
            ("unprinted-vars", "SILENT", async ctx =>
            {
                ScenarioTools.SetHp(ctx, 9999, 9999);
                foreach (string c in new[] { "FINESSE", "FLASH_OF_STEEL", "BACKFLIP", "SHADOW_STEP", "ADRENALINE", "METEOR_SHOWER", "PHOTON_CUT", "TUNE", "HEIRLOOM_HAMMER", "COLOSSUS", "BLUR", "REAVE" })
                {
                    await ScenarioTools.GiveCard(ctx, c);
                    await ScenarioTools.GiveCard(ctx, c, upgrades: 1);
                }
                await ScenarioTools.JumpToRoom(ctx, RoomType.Monster);
            }),
        };
        int k = 0;
        foreach (var s in scenarios)
        {
            // the scenarios that exist for one frame kind are probed at EVERY decision
            int savedEvery = probe.Every;
            if (s.name.StartsWith("crystal", StringComparison.Ordinal))
            {
                probe.Every = 1;
            }
            EpisodeResult r = ScenarioRunner.Run(session, "obs-leak-" + s.name, s.setup, character: s.ch, policy: new RandomLegalPolicy((ulong)(3100 + k++)), observer: probe,
                options: new EpisodeOptions { MaxPolicyDecisions = 300, MaxCombatTurns = 25 });
            probe.Every = savedEvery;
            episodes.Add(new { scenario = s.name, outcome = r.Outcome.ToString(), r.TruncatedBy, abort = r.Abort?.Signature });
        }
        for (int i = 0; i < runs; i++)
        {
            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel(label + ":" + i), Characters = new[] { DriverCommands.AllCharacters[i % 6] }, Ascension = i % 2 == 0 ? 0 : 10 };
            EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy(policySeedBase + (ulong)i), new EpisodeOptions { DebugForcedHp = hp, MaxPolicyDecisions = 1500, MaxCombatTurns = 40 }, probe);
            episodes.Add(new { seed = r.Seed, character = spec.Characters[0], outcome = r.Outcome.ToString(), r.TruncatedBy, floor = r.TotalFloor, abort = r.Abort?.Signature });
        }
        // every perturbation and every control must have been applied at least once: a leak test that never ran proves nothing
        object chest = ChestAb(session, l, out bool chestOk);
        bool negativesOk = chestOk && probe.Stats.Values.Where(s => !s.Control).All(s => s.Failed == 0 && s.Applied > 0);
        bool controlsOk = probe.Stats.Values.Where(s => s.Control).All(s => s.Failed == 0 && s.Applied > 0);
        ok = negativesOk && controlsOk && probe.NotRestored == 0;
        return new
        {
            kind = "obs-leak-test",
            ok,
            wall_s = Math.Round(wall.Elapsed.TotalSeconds, 1),
            probes = probe.Probes,
            every,
            not_restored = probe.NotRestored,
            not_restored_examples = probe.NotRestoredExamples,
            perturbations = probe.Stats.ToDictionary(kv => kv.Key, kv => new { kv.Value.Control, kv.Value.Applied, kv.Value.Passed, kv.Value.Failed, not_applicable = kv.Value.NotApplicable, kv.Value.Examples, kv.Value.Culprits }),
            P07b_unopened_chest_relics = chest,
            // P08 / P09: state left unperturbed because the game's own hand-card display depends on it (displayed, not hidden)
            P08_P09_displayed_through_hand_cards = probe.DisplayedThroughCards,
            P08_P09_displayed_through_play_pile_only = probe.DisplayedThroughPlayPile,
            P08_P09_play_pile_examples = probe.PlayPileExamples,
            episodes,
        };
    }

    /// <summary>Captures the encoded frames of the TREASURE decisions of one episode (auto-advanced ones included).</summary>
    private sealed class TreasureFrames : IEpisodeObserver
    {
        private readonly ObsEncoder _enc;
        public byte[]? Chest;
        public byte[]? FirstRelicPage;
        public string? RelicOnPage;

        public TreasureFrames(ObsEncoder enc)
        {
            _enc = enc;
        }

        public void OnEpisodeStart(RunContext ctx)
        {
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
        {
            if (decision.Kind != DecisionKind.Treasure)
            {
                return;
            }
            _enc.EncodeDecision(decision, 1, 0, 0);
            if (decision.SubKind == (int)TreasurePhase.Chest)
            {
                Chest ??= _enc.Payload.ToArray();
            }
            else if (FirstRelicPage == null && decision.Context is TreasureContext tc && tc.Relics.Count > 0)
            {
                FirstRelicPage = _enc.Payload.ToArray();
                RelicOnPage = tc.Relics[0].Id.Entry;
            }
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }
    }

    /// <summary>
    /// P07b. The relics of a chest are fixed when the treasure room is entered (TreasureRoomRelicSynchronizer.BeginRelicPicking)
    /// and hidden until the chest is open. Two episodes of the same scenario and seed on the REAL treasure handler; in the
    /// second one the relic inside the unopened chest is replaced by another relic right after the room was entered.
    /// Pass: the CHEST page frames are byte-identical (the policy cannot see the content); positive control: the RELIC
    /// page frames differ (the replacement really changed what the chest holds).
    /// </summary>
    public static object ChestAb(HostSession session, ObsRuntime.Loaded l, out bool ok)
    {
        List<object> rows = new List<object>();
        ok = true;
        foreach ((string ch, bool potion) in new[] { ("DEFECT", true), ("SILENT", false), ("IRONCLAD", true) })
        {
            ObsEncoder enc = ObsRuntime.NewEncoder(l);
            enc.Options = new EncoderOptions { MapInCombat = true, DeckInCombat = true, UpgradePreviews = true, TargetPreviews = true };
            string? original = null, replacement = null;
            TreasureFrames Play(bool perturb)
            {
                TreasureFrames frames = new TreasureFrames(enc);
                ScenarioRunner.Run(session, "obs-leak-chest-ab", async ctx =>
                {
                    if (potion)
                    {
                        await ScenarioTools.GivePotion(ctx, "FIRE_POTION"); // the CHEST page is then asked (potion actions), not auto-advanced
                    }
                    await ScenarioTools.JumpToRoom(ctx, RoomType.Treasure);
                    List<RelicModel>? relics = ctx.RunManager.TreasureRoomRelicSynchronizer._currentRelics;
                    if (relics != null && relics.Count > 0)
                    {
                        original = relics[0].Id.Entry;
                        if (perturb)
                        {
                            // TEST ONLY: writes hidden state on purpose
                            RelicModel other = ModelDb.AllRelics.First(r => r.Id.Entry != relics[0].Id.Entry && r.Rarity == relics[0].Rarity && ctx.LocalPlayer.Relics.All(have => have.Id != r.Id));
                            relics[0] = other.ToMutable();
                            replacement = other.Id.Entry;
                        }
                    }
                }, character: ch, seed: HostSession.SeedFromLabel("OBS-LEAK-CHEST:" + ch), policy: new FirstActionPolicy(), observer: frames,
                    options: new EpisodeOptions { MaxPolicyDecisions = 40 });
                return frames;
            }
            TreasureFrames a = Play(false);
            TreasureFrames b = Play(true);
            bool chestSame = a.Chest != null && b.Chest != null && LeakProbe.Same(a.Chest, b.Chest);
            bool relicDiffers = a.FirstRelicPage != null && b.FirstRelicPage != null && !LeakProbe.Same(a.FirstRelicPage, b.FirstRelicPage);
            bool rowOk = chestSame && relicDiffers && replacement != null;
            ok &= rowOk;
            rows.Add(new
            {
                character = ch, potion_in_belt = potion, ok = rowOk, chest_page_frames_identical = chestSame, relic_page_frames_differ = relicDiffers,
                chest_frame_bytes = a.Chest?.Length, relic_in_chest = original, replaced_by = replacement, relic_page_a = a.RelicOnPage, relic_page_b = b.RelicOnPage,
            });
        }
        return new { ok, pairs = rows };
    }

    private static async Task TryRelic(RunContext ctx, string id)
    {
        try
        {
            await ScenarioTools.GiveRelic(ctx, id);
        }
        catch (Exception e) when (e is ArgumentException || e is InvalidOperationException || e is KeyNotFoundException)
        {
            // relic id not in v0.113 - fine for a test setup
        }
    }
}
