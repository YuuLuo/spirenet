using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Driver;

namespace SpireHost.Replay.V2;

/// <summary>
/// Combat inside the run: the recorded actions (combat.json Actions[]) and prompt answers (prompt_records[]) are applied
/// through the production COMBAT and CARD_SELECT handlers, so that the run continues from the true post-combat state.
///
/// Per recorded action, before it is applied:
///   * the host must be at a COMBAT decision (the collector's decision point);
///   * the collector's public signature (round, turn, energy, HP, block, hand in order, pile sizes, potions, enemies)
///     must equal the host's - computed with the same function as the .mcr replay tool (McrReplayer.Signature);
///   * the collector's legal-option count must equal the handler's action count (named exception: the potion-lock
///     window of a fight that starts while the real client is still fading the room in);
///   * the action must be a legal host action.
/// Which card instance / potion slot / target the action names is taken from the fight's own replay file (.mcr: the
/// game's NetPlayCardAction etc., exact combat card ids and creature ids) and cross-checked with combat.json; a fight
/// without a replay file (ended at the collector's turn cap or stalled) is identified from combat.json alone.
/// The game's own per-action full-combat-state checksums of the same fights are compared too: inside the run (V2Replayer.Checksums.cs:
/// the host's states at the points where the real client recorded its states) and by the separate oracle --replay-mcr (every fight
/// re-entered from its replay's save, the recorded game actions injected).
/// </summary>
public sealed partial class V2Replayer
{
    private sealed class CombatCursor
    {
        public V2Combat Combat = null!;
        public int Action;
        public int Prompt;
        public List<INetAction>? McrActions;
        public List<NetPlayerChoiceResult>? McrChoices;
        public int McrChoice;
        public CombatReplay? Replay;
        public AbstractRoom? HostRoom;

        /// <summary>A fight without a replay file: recorded choices among identical cards that the recording does not tell apart.</summary>
        public int UnresolvedIdentities;

        public void PromptConsumed() => Prompt++;
    }

    private CombatCursor? _combat;
    private int _combatIndex;            // next fight of the trace

    /// <summary>The fight the host is in or was last in (noted at CombatManager.CombatBegan), and whether it has asked for a decision yet.</summary>
    private (int Floor, string? Encounter, string? RoomType, AbstractRoom? Room)? _hostFight;
    private bool _hostFightAsked;
    private readonly Queue<(int Floor, string? Encounter, string? RoomType, AbstractRoom? Room)> _silentHostFights = new Queue<(int, string?, string?, AbstractRoom?)>();

    /// <summary>CombatManager.CombatBegan: a fight starts in the host.</summary>
    private void NoteHostFightBegan(RunContext ctx)
    {
        NoteHostFightOver();
        CombatRoom? room = ctx.Run.CurrentRoom as CombatRoom;
        _hostFight = (ctx.Run.TotalFloor, room?.Encounter?.Id.Entry, ctx.Run.CurrentRoom?.RoomType.ToString(), ctx.Run.CurrentRoom);
        _hostFightAsked = false;
    }

    /// <summary>The previous host fight is over; if it never asked for a decision it is a fight the trace must have recorded without any action.</summary>
    private void NoteHostFightOver()
    {
        if (_hostFight.HasValue && !_hostFightAsked)
        {
            _silentHostFights.Enqueue(_hostFight.Value);
        }
        _hostFight = null;
    }

    /// <summary>
    /// Fights that ended before the first decision point (every enemy died in the start-of-combat sequence): the collector
    /// records them with no action and no prompt, and the host asks nothing in them. They are matched one to one, in order.
    /// </summary>
    private bool ReconcileSilentFights(RunContext ctx, bool hostFightInProgress)
    {
        if (!hostFightInProgress)
        {
            NoteHostFightOver();
        }
        while (_silentHostFights.Count > 0)
        {
            (int Floor, string? Encounter, string? RoomType, AbstractRoom? Room) h = _silentHostFights.Dequeue();
            string hostFight = $"floor {h.Floor} {h.Encounter} ({h.RoomType})";
            if (_combatIndex >= _t.Combats.Count)
            {
                Diverge("combat.host_fight_without_record", true, "host fight " + hostFight, "the host had a fight (it ended before any decision) that the trace does not have", null, hostFight);
                return false;
            }
            V2Combat c = _t.Combats[_combatIndex];
            if (c.Actions.Count != 0 || c.Prompts.Count != 0 || c.End != "win" || c.Floor != h.Floor || c.Encounter != h.Encounter || c.RoomType != h.RoomType)
            {
                Diverge("combat.host_fight_ended_before_any_decision", true, $"combat {c.CombatIndex} ({c.Encounter}, floor {c.Floor}, {c.File})",
                    "the host's fight ended before it asked for any decision; the trace's next fight is not such a fight", $"{c.Encounter} floor {c.Floor}: {c.Actions.Count} actions, {c.Prompts.Count} prompts, end {c.End}", hostFight);
                return false;
            }
            _combatIndex++;
            _r.CombatsReplayed++;
            Info("combat.fight_ended_before_any_decision_in_both");
            CompareChecksums(c, null, h.Room, $"combat {c.CombatIndex} ({c.Encounter}, floor {c.Floor}, {c.File})");
        }
        return true;
    }

    private string CombatWhere()
    {
        CombatCursor? c = _combat;
        return c == null ? "combat ?" : $"combat {c.Combat.CombatIndex} ({c.Combat.Encounter}, floor {c.Combat.Floor}, {c.Combat.File}) action {c.Action}/{c.Combat.Actions.Count}";
    }

    private static bool IsPolicyAction(INetAction? a) => a is NetPlayCardAction || a is NetUsePotionAction || a is NetDiscardPotionGameAction || a is NetEndPlayerTurnAction;

    private int CombatDecision(Decision d, RunContext ctx)
    {
        if (_combat == null && !OpenCombat(d, ctx))
        {
            return -1;
        }
        return d.Kind == DecisionKind.Combat ? CombatAction(d, ctx) : CombatPrompt(d, ctx);
    }

    private bool OpenCombat(Decision d, RunContext ctx)
    {
        _hostFightAsked = true;
        if (!ReconcileSilentFights(ctx, hostFightInProgress: true))
        {
            return false;
        }
        if (_combatIndex >= _t.Combats.Count)
        {
            Diverge("combat.host_fight_without_record", true, $"decision {d.Sequence} floor {ctx.Run.TotalFloor}", "the host is in a fight but the trace has no fight left", null, RunDriverRoom(ctx));
            return false;
        }
        V2Combat c = _t.Combats[_combatIndex++];
        CombatCursor cur = new CombatCursor { Combat = c, HostRoom = ctx.Run.CurrentRoom };
        _combat = cur;
        CombatRoom? room = ctx.Run.CurrentRoom as CombatRoom;
        string where = CombatWhere();
        void Field(string name, string? trace, string? host)
        {
            _r.ContentFieldsCompared++;
            if (trace != host)
            {
                Diverge("combat.identity." + name, true, where, "the recorded fight is not the fight the host is in: " + name, trace, host);
            }
        }
        Field("floor", c.Floor.ToString(), ctx.Run.TotalFloor.ToString());
        Field("encounter", c.Encounter, room?.Encounter?.Id.Entry);
        Field("room_type", c.RoomType, ctx.Run.CurrentRoom?.RoomType.ToString());
        // the tracer names a fight's files after the number of dumps written so far (RunTracer2.PlayCombat)
        Field("dump_index", c.Tag.ToString(), _nextDump.ToString());
        if (_stop)
        {
            return false;
        }
        if (c.McrPath != null && _opt.UseMcr)
        {
            CombatReplay replay = McrFile.Load(c.McrPath).Replay;
            cur.Replay = replay;
            cur.McrActions = new List<INetAction>();
            cur.McrChoices = new List<NetPlayerChoiceResult>();
            foreach (CombatReplayEvent ev in replay.events)
            {
                if (ev.eventType == CombatReplayEventType.GameAction && IsPolicyAction(ev.action))
                {
                    cur.McrActions.Add(ev.action!);
                }
                else if (ev.eventType == CombatReplayEventType.PlayerChoice && ev.playerChoiceResult.HasValue)
                {
                    cur.McrChoices.Add(ev.playerChoiceResult.Value);
                }
            }
            if (cur.McrActions.Count != c.Actions.Count || cur.McrChoices.Count != c.Prompts.Count)
            {
                Diverge("record.mcr_vs_combat_json", true, where, "the fight's replay file and its combat.json do not hold the same number of actions / prompt answers",
                    $"combat.json: {c.Actions.Count} actions, {c.Prompts.Count} prompts", $".mcr: {cur.McrActions.Count} policy actions, {cur.McrChoices.Count} choices");
                return false;
            }
        }
        return true;
    }

    private static string RunDriverRoom(RunContext ctx) => (ctx.Run.CurrentRoom?.GetType().Name ?? "null") + ":" + ctx.Run.CurrentRoom?.RoomType + ":" + ctx.Run.CurrentRoom?.ModelId?.Entry;

    /// <summary>The fight is over in the host. It must be over in the recording too, at the same point and with the same result.</summary>
    private void CloseCombat(RunContext ctx, bool hostWon)
    {
        CombatCursor cur = _combat!;
        V2Combat c = cur.Combat;
        string where = CombatWhere();
        _combat = null;
        if (_sel != null && _sel.InCombat)
        {
            _sel = null;
        }
        if (cur.Action < c.Actions.Count)
        {
            Diverge("combat.host_fight_ended_early", true, where, $"the host's fight ended ({(hostWon ? "won" : "lost")}) after {cur.Action} of {c.Actions.Count} recorded actions", Js(c.Actions[cur.Action]), hostWon ? "win" : "loss");
            return;
        }
        if (c.End != (hostWon ? "win" : "loss"))
        {
            Diverge("combat.end", true, where, "the fight ended differently", c.End, hostWon ? "win" : "loss");
            return;
        }
        if (cur.Prompt < c.Prompts.Count)
        {
            Diverge("combat.prompts_remain", true, where, $"the fight ended with {c.Prompts.Count - cur.Prompt} recorded selection prompts unanswered", Js(c.Prompts[cur.Prompt]["chosen"]), null);
            return;
        }
        _r.CombatsReplayed++;
        CompareChecksums(c, cur.Replay, cur.HostRoom, where);
    }

    // =====================================================================================================
    //  One recorded action
    // =====================================================================================================

    private int CombatAction(Decision d, RunContext ctx)
    {
        CombatCursor cur = _combat!;
        V2Combat c = cur.Combat;
        string where = CombatWhere();
        Player me = ctx.LocalPlayer;
        CombatState? cs = ctx.CombatManager.DebugOnlyGetState();
        if (cs == null)
        {
            Diverge("combat.no_state", true, where, "COMBAT decision without a combat state", null, Describe(d));
            return -1;
        }
        int k = cur.Action;
        if (k >= c.Actions.Count)
        {
            return RecordedFightHasNoMoreActions(d, ctx, cur, where);
        }
        JsonNode rec = c.Actions[k]!;
        _recordDiverged = false;

        // ---- state: the collector's public signature before the action ----
        string sig = McrReplayer.Signature(me, cs);
        string recSig = S(rec["Sig"]) ?? "";
        _r.CombatSigCompared++;
        if (sig != recSig)
        {
            if (cur.McrActions == null && cur.UnresolvedIdentities > 0 && lastFightOfAnEarlyEnd() && Rule(V2Classes.CombatNoMcrIdentityEnd))
            {
                // A fight the client wrote no replay for (it ends the trace at the collector's turn cap / stall): an earlier recorded choice
                // among identical cards could not be identified from combat.json, so from there on the host may hold another, equally
                // legal, state. Nothing after that point is comparable.
                Info("combat.no_replay_file.actions_compared_before_the_unidentifiable_choice_showed", cur.Action);
                _combat = null;
                Stop("trace_end:unidentifiable_choice_in_a_fight_without_replay", where + ": recorded '" + recSig + "' host '" + sig + "'");
                return -1;
            }
            Diverge("combat.signature", true, where, "the public combat state before this recorded action differs", recSig, sig);
            return -1;
        }
        bool lastFightOfAnEarlyEnd() => _combatIndex == _t.Combats.Count && (c.End == "max_turns" || c.End == "stall");
        _r.CombatSigMatched++;
        int turn = me.PlayerCombatState?.TurnNumber ?? -1;
        if (I(rec["Turn"]) != turn)
        {
            Diverge("combat.turn", true, where, "the player turn number differs", S(rec["Turn"]), turn.ToString());
            return -1;
        }

        // ---- legal-action count ----
        int mine = d.Actions.Count;
        int potionActions = d.Actions.Count(a => IsPotionVerb(a.Verb));
        int recorded = I(rec["Options"]);
        _r.CombatOptionCountsCompared++;
        _r.OfferItemsCompared += mine;
        if (recorded == mine)
        {
            _r.CombatOptionCountsEqual++;
        }
        else if (potionActions > 0 && recorded == mine - potionActions && Rule(V2Classes.CombatPotionLockWindow))
        {
            // the real client had every potion locked here (room fade-in); the host has no fade
        }
        else
        {
            Diverge("combat.option_count", false, where, "the collector's legal-option count differs from the handler's", recorded.ToString(), mine + " (" + potionActions + " potion actions): " + Describe(d));
        }

        // ---- the action ----
        string kind = S(rec["Kind"]) ?? "";
        string? card = S(rec["Card"]);
        string? potion = S(rec["Potion"]);
        int target = I(rec["Target"]);
        IReadOnlyList<Creature> enemies = cs.Enemies;
        int index = -1;
        if (cur.McrActions != null)
        {
            INetAction net = cur.McrActions[k];
            index = FindByReplayAction(d, net, kind, card, potion, target, me, enemies, where);
            if (_stop)
            {
                return -1;
            }
            if (index >= 0)
            {
                _r.CombatActionsByMcrIdentity++;
            }
        }
        else
        {
            index = FindByCombatJson(d, kind, card, potion, target, me, enemies, k + 1 < c.Actions.Count ? S(c.Actions[k + 1]!["Sig"]) : null, cur);
        }
        if (index < 0)
        {
            Diverge("chosen_not_offered.combat", true, where, $"the action the real client took ({kind} {card ?? potion} target {target}) is not a legal host action", Js(rec), Describe(d));
            return -1;
        }
        cur.Action++;
        _r.CombatActionsReplayed++;
        _r.ActionsApplied++;
        _r.DecisionsCompared++;
        _r.ComparedByKind["combat_action"] = _r.ComparedByKind.GetValueOrDefault("combat_action") + 1;
        if (!_recordDiverged)
        {
            _r.DecisionsMatched++;
        }
        return index;
    }

    /// <summary>The host is at a COMBAT decision and the recording has no action left: the trace ended inside this fight (turn cap / stall), or the fights differ.</summary>
    private int RecordedFightHasNoMoreActions(Decision d, RunContext ctx, CombatCursor cur, string where)
    {
        V2Combat c = cur.Combat;
        bool lastFight = _combatIndex == _t.Combats.Count;
        SkipAfterTheFactRecords();
        bool recordsDone = _rec == _t.Records.Count;
        if (c.End == "max_turns" && lastFight && recordsDone && _t.End == "combat_max_turns" && Rule(V2Classes.CombatMaxTurnsEnd))
        {
            // the collector stopped at this decision point (turn > 150) and dumped at once: compare that dump, then the trace is at its end
            int turn = ctx.LocalPlayer.PlayerCombatState?.TurnNumber ?? -1;
            if (turn <= 150)
            {
                Diverge("combat.max_turns_point", true, where, "the recording ended at its 150-turn cap but the host is in turn " + turn, "151", turn.ToString());
                return -1;
            }
            if (cur.Prompt < c.Prompts.Count)
            {
                Diverge("combat.prompts_remain", true, where, $"the recording ended with {c.Prompts.Count - cur.Prompt} selection prompts the host did not ask", Js(c.Prompts[cur.Prompt]["chosen"]), null);
                return -1;
            }
            _combat = null;
            _r.CombatsReplayed++;
            CompareDump(ctx, "end-combat_max_turns");
            if (!_stop)
            {
                if (_nextDump != _t.Dumps.Count)
                {
                    Diverge("end.dumps_remain", true, where, $"the trace has {_t.Dumps.Count - _nextDump} more dumps after its end", _t.Dumps[_nextDump].File, null);
                    return -1;
                }
                Stop("trace_end:combat_max_turns", where);
            }
            return -1;
        }
        if (c.End == "stall" && lastFight && recordsDone && _t.End == "combat_stall" && Rule(V2Classes.CombatStallEnd))
        {
            // the real client hung here (a timing hang of the headless client, not game state): nothing after its last action is comparable
            _combat = null;
            Stop("trace_end:combat_stall", where);
            return -1;
        }
        Diverge("combat.host_fight_continues", true, where, $"the recorded fight ended ({c.End}) after {c.Actions.Count} actions; the host is still asking", c.End, Describe(d));
        return -1;
    }

    private static bool TargetMatches(in LegalAction a, uint? targetId)
    {
        if (!targetId.HasValue)
        {
            return a.Target.IsNone;
        }
        return !a.Target.IsNone && a.Target.Creature.CombatId == targetId.Value;
    }

    /// <summary>The host action that IS the game action the real client enqueued (the fight's replay): exact card instance, potion slot and target.</summary>
    private int FindByReplayAction(Decision d, INetAction net, string kind, string? card, string? potion, int target, Player me, IReadOnlyList<Creature> enemies, string where)
    {
        void Inconsistent(string what)
        {
            Diverge("record.mcr_vs_combat_json", true, where, "the fight's replay and its combat.json disagree about this action: " + what, $"{kind} {card ?? potion} target {target}", net.ToString());
        }
        switch (net)
        {
            case NetPlayCardAction play:
            {
                if (kind != "play" || card == null || card.TrimEnd('+') != play.modelId.Entry)
                {
                    Inconsistent("card");
                    return -1;
                }
                if (target >= 0 && (target >= enemies.Count || enemies[target].CombatId != play.targetId))
                {
                    Inconsistent("target");
                    return -1;
                }
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    LegalAction a = d.Actions[i];
                    if (a.Verb == Verb.PlayCard && NetCombatCardDb.Instance.TryGetCardId(a.Source.Card, out uint id) && id == play.card.CombatCardIndex && TargetMatches(a, play.targetId))
                    {
                        return i;
                    }
                }
                return -1;
            }
            case NetUsePotionAction use:
            {
                if (kind != "potion")
                {
                    Inconsistent("kind");
                    return -1;
                }
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    LegalAction a = d.Actions[i];
                    if (a.Verb != Verb.UsePotion || a.Source.Index != (int)use.potionIndex || me.PotionSlots[a.Source.Index]?.Id.Entry != potion)
                    {
                        continue;
                    }
                    // an aimed potion names its enemy; every other potion is used on the owner or on nobody (the UI passes the owner / null)
                    bool aimedAtEnemy = !a.Target.IsNone && !ReferenceEquals(a.Target.Creature, me.Creature);
                    if (aimedAtEnemy ? a.Target.Creature.CombatId == use.targetId : true)
                    {
                        return i;
                    }
                }
                return -1;
            }
            case NetDiscardPotionGameAction dis:
                if (kind != "discard_potion")
                {
                    Inconsistent("kind");
                    return -1;
                }
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    LegalAction a = d.Actions[i];
                    if (a.Verb == Verb.DiscardPotion && a.Source.Index == (int)dis.potionSlotIndex && me.PotionSlots[a.Source.Index]?.Id.Entry == potion)
                    {
                        return i;
                    }
                }
                return -1;
            case NetEndPlayerTurnAction:
                if (kind != "end_turn")
                {
                    Inconsistent("kind");
                    return -1;
                }
                return Find(d, Verb.EndTurn);
            default:
                Inconsistent("unknown replay action type");
                return -1;
        }
    }

    /// <summary>
    /// A fight without a replay file: the action as combat.json names it (card id + upgrade flag, enemy index / self / none,
    /// potion id). Several hand cards / potion slots can carry that name. Which one the real client used shows in the NEXT
    /// recorded signature (the hand in order, the belt in order): the candidate whose removal leaves the hand / belt that
    /// the recording shows next is taken. When the recording cannot tell them apart (the same hand either way) the first
    /// is taken and counted as unresolved.
    /// </summary>
    private int FindByCombatJson(Decision d, string kind, string? card, string? potion, int target, Player me, IReadOnlyList<Creature> enemies, string? nextSig, CombatCursor cur)
    {
        int found = -1;
        int candidates = 0;
        List<int> all = new List<int>();
        for (int i = 0; i < d.Actions.Count; i++)
        {
            LegalAction x = d.Actions[i];
            bool match = false;
            switch (kind)
            {
                case "play":
                    if (x.Verb == Verb.PlayCard && card != null)
                    {
                        CardModel c = x.Source.Card;
                        if (c.Id.Entry + (c.IsUpgraded ? "+" : "") == card)
                        {
                            match = target >= 0 ? !x.Target.IsNone && target < enemies.Count && ReferenceEquals(x.Target.Creature, enemies[target])
                                : target == -2 ? !x.Target.IsNone && ReferenceEquals(x.Target.Creature, me.Creature)
                                : x.Target.IsNone;
                        }
                    }
                    break;
                case "potion":
                    if (x.Verb == Verb.UsePotion && potion != null && me.PotionSlots[x.Source.Index]?.Id.Entry == potion)
                    {
                        match = target >= 0 ? !x.Target.IsNone && target < enemies.Count && ReferenceEquals(x.Target.Creature, enemies[target])
                            : x.Target.IsNone || ReferenceEquals(x.Target.Creature, me.Creature);
                    }
                    break;
                case "discard_potion":
                    match = x.Verb == Verb.DiscardPotion && potion != null && me.PotionSlots[x.Source.Index]?.Id.Entry == potion;
                    break;
                case "end_turn":
                    match = x.Verb == Verb.EndTurn;
                    break;
            }
            if (match)
            {
                candidates++;
                all.Add(i);
                if (found < 0)
                {
                    found = i;
                }
            }
        }
        if (candidates > 1)
        {
            int resolved = ResolveByNextSignature(d, all, kind, me, nextSig);
            if (resolved >= 0)
            {
                Explain(V2Classes.CombatNoMcrIdentity + ":identified by the next recorded signature");
                return resolved;
            }
            Explain(V2Classes.CombatNoMcrIdentity + ":not identifiable, first taken");
            cur.UnresolvedIdentities++;
        }
        return found;
    }

    private static List<string>? SigList(string? sig, string open)
    {
        if (sig == null)
        {
            return null;
        }
        int a = sig.IndexOf(open, StringComparison.Ordinal);
        if (a < 0)
        {
            return null;
        }
        a += open.Length;
        int b = sig.IndexOf(']', a);
        if (b < 0)
        {
            return null;
        }
        string body = sig.Substring(a, b - a);
        return body.Length == 0 ? new List<string>() : body.Split(',').ToList();
    }

    /// <summary>
    /// Among host actions that the recording names identically (same card id / potion id and target), the one after which
    /// the hand (belt) agrees longest with the next recorded signature's hand (belt); -1 when two of them agree equally.
    /// </summary>
    private static int ResolveByNextSignature(Decision d, List<int> candidates, string kind, Player me, string? nextSig)
    {
        List<string>? next = kind == "play" ? SigList(nextSig, " H[") : SigList(nextSig, " P[");
        if (next == null)
        {
            return -1;
        }
        List<string> now = kind == "play"
            ? me.PlayerCombatState!.Hand.Cards.Select(x => x.Id.Entry + (x.IsUpgraded ? "+" : "")).ToList()
            : me.PotionSlots.Select(p => p?.Id.Entry ?? "-").ToList();
        int best = -1, bestScore = -1;
        bool tie = false;
        foreach (int i in candidates)
        {
            int at = d.Actions[i].Source.Index;
            List<string> after = new List<string>(now);
            if (at < 0 || at >= after.Count)
            {
                return -1;
            }
            if (kind == "play")
            {
                after.RemoveAt(at);
            }
            else
            {
                after[at] = "-";
            }
            int score = 0;
            while (score < after.Count && score < next.Count && after[score] == next[score])
            {
                score++;
            }
            if (score > bestScore)
            {
                best = i;
                bestScore = score;
                tie = false;
            }
            else if (score == bestScore)
            {
                tie = true;
            }
        }
        return tie ? -1 : best;
    }

    // =====================================================================================================
    //  One recorded selection prompt inside the fight
    // =====================================================================================================

    private int CombatPrompt(Decision d, RunContext ctx)
    {
        CombatCursor cur = _combat!;
        V2Combat c = cur.Combat;
        string where = CombatWhere();
        if (cur.Prompt >= c.Prompts.Count)
        {
            Diverge("combat.host_prompt_without_record", true, where, "the host asks a selection prompt inside the fight and the recording has none left", null, Describe(d));
            return -1;
        }
        JsonObject rec = c.Prompts[cur.Prompt];
        if (Kind(rec) != "selection")
        {
            Diverge("record.unsupported_kind", true, where, "an in-combat prompt record of a kind the comparison has no mapping for", Kind(rec), Describe(d));
            return -1;
        }
        NetPlayerChoiceResult? choice = null;
        if (cur.McrChoices != null)
        {
            if (cur.McrChoice >= cur.McrChoices.Count)
            {
                Diverge("record.mcr_vs_combat_json", true, where, "the fight's replay has no choice left for this prompt", Js(rec["chosen"]), null);
                return -1;
            }
            choice = cur.McrChoices[cur.McrChoice++];
        }
        return Selection(rec, d, ctx, choice);
    }
}
