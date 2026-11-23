using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using SpireHost.Driver;

namespace SpireHost.Replay.V2;

/// <summary>
/// Selection screens: the collector's "selection" record (written before the first click: screen type, prompt, prefs,
/// candidates, button states) and its "selection_result" (button states after every pick, every confirm click) against
/// the host's CARD_SELECT frame, step by step.
///
/// What is compared (per prompt):
///   identity     screen type <-> CardSelectKind, prompt key, prefs min / max / cancelable / manual confirmation, pile
///   candidates   every card the screen lists (id, upgrade, enchantment, affliction) against the cards the game handed
///                to the host's selector, in order (combat-pile grids: as a multiset, their record is in display order)
///   legality     every candidate has a SELECT_CARD action; before any pick: Close <-> CANCEL, "can finish with nothing"
///                (confirm enabled with nothing selected / the skip button) <-> SKIP; after every pick: "can finish now"
///                (a confirm button enabled and visible, or the screen closed) <-> the frame completed or offers CONFIRM
///   the answer   the recorded picks, one SELECT_CARD each in the recorded order, then CONFIRM where the frame asks
///                for it; cancel -> CANCEL, skip -> SKIP. The frame must end exactly where the real screen closed.
/// </summary>
public sealed partial class V2Replayer
{
    private sealed class Sel
    {
        public JsonObject Rec = null!;
        public bool InCombat;
        public string Where = "";
        public int Seq = -1;
        public IDecisionSource Frame = null!;
        public ICardSelectHostView View = null!;
        public CardSelectContext Ctx = null!;
        public string Plan = "";                 // picks | cancel | skip
        public List<CardModel> Picks = new List<CardModel>();
        public HashSet<CardModel> Picked = new HashSet<CardModel>(ReferenceEqualityComparer.Instance);
        public List<JsonNode> PickSteps = new List<JsonNode>();   // the "picked" steps of the selection_result, in order
        public int ConfirmClicks;                // "click" steps
        public bool HasResult;
        public int Applied;                      // picks applied so far
        public bool Ended;                       // cancel / skip / confirm applied: the frame must be gone at the next decision
        public bool Hand;
    }

    private Sel? _sel;

    private static bool IsConfirmButton(JsonNode b)
    {
        string type = S(b["type"]) ?? "";
        string node = S(b["node"]) ?? "";
        return type == "NConfirmButton" || node.EndsWith("/Confirm", StringComparison.Ordinal) || node.EndsWith("PreviewConfirm", StringComparison.Ordinal) || S(b["role"]) == "confirm";
    }

    private static bool IsCloseButton(JsonNode b)
    {
        string node = S(b["node"]) ?? "";
        return (S(b["type"]) ?? "") == "NBackButton" && node.EndsWith("/Close", StringComparison.Ordinal) && !node.Contains("Preview", StringComparison.Ordinal);
    }

    private static bool Shown(JsonNode b) => B(b["enabled"]) && B(b["visible"]);

    private static bool IsSubMultiset(List<string> part, List<string> whole)
    {
        Dictionary<string, int> left = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string w in whole)
        {
            left[w] = left.GetValueOrDefault(w) + 1;
        }
        foreach (string p in part)
        {
            if (left.GetValueOrDefault(p) <= 0)
            {
                return false;
            }
            left[p]--;
        }
        return true;
    }

    /// <summary>
    /// A button counts as offered when it is enabled AND visible. A button that is enabled but not visible cannot be clicked, and its
    /// enabled flag depends on animation timing (validation report section 5): every such button of the role that is being looked
    /// at is counted under the named rule, never silently passed over.
    /// </summary>
    private bool Offered(JsonNode b)
    {
        if (B(b["enabled"]) && !B(b["visible"]))
        {
            Explain(V2Classes.SelectionInvisibleButton);
        }
        return Shown(b);
    }

    /// <summary>"The selection can be finished now": a confirm-type button is enabled and visible.</summary>
    private bool UiCanFinish(JsonNode? buttons)
    {
        if (buttons is JsonArray list)
        {
            bool any = false;
            foreach (JsonNode? b in list)
            {
                any |= b != null && IsConfirmButton(b) && Offered(b);
            }
            return any;
        }
        return buttons is JsonObject one && Offered(one);
    }

    private bool UiCloseOffered(JsonNode? buttons)
    {
        bool any = false;
        if (buttons is JsonArray list)
        {
            foreach (JsonNode? b in list)
            {
                any |= b != null && IsCloseButton(b) && Offered(b);
            }
        }
        return any;
    }

    private static string KindOfScreen(string screen, string? mode, CardSelectKind host)
    {
        switch (screen)
        {
            case "NDeckUpgradeSelectScreen":
                return "DeckUpgrade";
            case "NDeckTransformSelectScreen":
                return "DeckTransform";
            case "NDeckEnchantSelectScreen":
                return "DeckEnchant";
            case "NDeckCardSelectScreen":
                return host == CardSelectKind.DeckGeneric ? "DeckGeneric" : "DeckRemoval";           // FromDeckForRemoval -> FromDeckGeneric: the same screen
            case "NSimpleCardSelectScreen":
                return host == CardSelectKind.SimpleGridRewards ? "SimpleGridRewards" : "SimpleGrid"; // both entries show the same screen
            case "NCombatPileCardSelectScreen":
                return "CombatPile";
            case "NChooseACardSelectionScreen":
                return "ChooseACard";
            case "NPlayerHand":
                return mode == "UpgradeSelect" ? "HandUpgrade" : host == CardSelectKind.HandDiscard ? "HandDiscard" : "Hand";   // FromHandForDiscard -> FromHand: the same hand mode
            default:
                return screen;
        }
    }

    /// <summary>
    /// A CARD_SELECT decision. <paramref name="rec"/> is the selection record (decisions.jsonl outside combat, the fight's
    /// prompt_records inside); <paramref name="mcr"/> is the answer the game itself recorded in the fight's replay
    /// (PlayerChoice event), when there is one.
    /// </summary>
    private int Selection(JsonObject rec, Decision d, RunContext ctx, NetPlayerChoiceResult? mcr)
    {
        bool inCombat = B(rec["in_combat"]);
        string where = inCombat ? CombatWhere() + " prompt " + _combat!.Prompt + " (" + S(rec["screen"]) + ")" : Where(rec);
        int seq = inCombat ? -1 : I(rec["seq"]);
        if (d.Kind != DecisionKind.CardSelect)
        {
            Diverge("decision_kind", true, where, $"the trace's next record is a selection ({S(rec["screen"])}) where the host asks a {d.Kind} decision", Js(rec["chosen"]), Describe(d), seq);
            return -1;
        }
        if (!inCombat && !CheckContext(rec, ctx))
        {
            return -1;
        }
        if (d.Context is not CardSelectContext c || d.Source is not ICardSelectHostView view || d.Source == null)
        {
            Diverge("decision_context", true, where, "CARD_SELECT decision without a CardSelectContext / host view", null, Describe(d), seq);
            return -1;
        }
        _recordDiverged = false;
        Sel s = new Sel { Rec = rec, InCombat = inCombat, Where = where, Seq = seq, Frame = d.Source, View = view, Ctx = c };
        string screen = S(rec["screen"]) ?? "";
        s.Hand = screen == "NPlayerHand";
        void Field(string name, string? trace, string? host)
        {
            _r.ContentFieldsCompared++;
            if (trace != host)
            {
                Diverge("content.selection." + name, false, where, "the selection prompt differs: " + name, trace, host, seq);
            }
        }

        // ---- identity ----
        Field("kind", KindOfScreen(screen, S(rec["mode"]), c.Kind), c.Kind.ToString());
        if (screen != "NChooseACardSelectionScreen")
        {
            // (the choose-a-card screen has a fixed header and no prefs: the collector records no prompt and min 0 / max 1)
            Field("prompt", S(rec["prompt"]), c.PromptTable.Length > 0 || c.PromptKey.Length > 0 ? c.PromptTable + "." + c.PromptKey : null);
        }
        if (c.Prefs.HasValue)
        {
            Field("min", S(rec["min"]), c.Prefs.Value.MinSelect.ToString());
            Field("max", S(rec["max"]), c.Prefs.Value.MaxSelect.ToString());
            if (rec.ContainsKey("cancelable"))
            {
                Field("cancelable", B(rec["cancelable"]) ? "true" : "false", c.Prefs.Value.Cancelable ? "true" : "false");
                Field("manual_confirm", B(rec["manual_confirm"]) ? "true" : "false", c.Prefs.Value.RequireManualConfirmation ? "true" : "false");
            }
        }
        else
        {
            Field("min", S(rec["min"]), c.RawMin.ToString());
            Field("max", S(rec["max"]), c.RawMax.ToString());
        }
        if (rec.ContainsKey("pile"))
        {
            Field("pile", S(rec["pile"]), c.Pile?.ToString());
        }

        // ---- candidates ----
        JsonArray cands = rec["candidates"]!.AsArray();
        IReadOnlyList<CardModel> raw = view.RawOptions;
        bool displayOrder = S(rec["candidate_order"]) == "ui_display";
        List<string> traceFaces = cands.Select(x => Face(x)).ToList();
        List<string> rawFaces = raw.Select(x => Face(x)).ToList();
        List<CardModel>? byIndex = null;         // the instance each candidate index names, when the order is established
        _r.ContentFieldsCompared += traceFaces.Count;
        if (traceFaces.SequenceEqual(rawFaces))
        {
            byIndex = raw.ToList();
            Info(displayOrder ? "selection.pile_grid.display_order_equals_the_list_the_game_hands_over" : "selection.candidates_in_list_order");
        }
        else if (traceFaces.OrderBy(x => x, StringComparer.Ordinal).SequenceEqual(rawFaces.OrderBy(x => x, StringComparer.Ordinal)))
        {
            if (!c.Merged && traceFaces.SequenceEqual(c.Options.Select(x => Face(x))))
            {
                byIndex = c.Options.ToList();    // a simple grid displays the list sorted with the prefs' comparison: the host's option order
                Info("selection.candidates_in_option_order");
            }
            else if (displayOrder)
            {
                Info("selection.pile_grid.display_order_differs_from_the_list_the_game_hands_over");
            }
            else if (s.Hand && Rule(V2Classes.SelectionHandScreenOrder))
            {
                // the collector lists the hand's holders in screen order; the same cards, another order than the hand pile
            }
            else
            {
                Diverge("content.selection.candidate_order", false, where, "the screen lists the same cards as the host's frame but in another order", string.Join(",", traceFaces), string.Join(",", rawFaces), seq);
            }
        }
        else if (displayOrder && traceFaces.Count >= 35 && traceFaces.Count < rawFaces.Count && IsSubMultiset(traceFaces, rawFaces) && Rule(V2Classes.SelectionPileGridWindow))
        {
            // NCombatPileCardSelectScreen keeps no list: the collector recorded the grid's holders, and the grid holds a window of the pile
            if (traceFaces.SequenceEqual(rawFaces.Take(traceFaces.Count)))
            {
                byIndex = raw.ToList();
                Info("selection.pile_grid.window_is_the_head_of_the_list_the_game_hands_over");
            }
            else
            {
                Info("selection.pile_grid.window_in_another_order");
            }
        }
        else
        {
            Diverge("content.selection.candidates", true, where, "the cards the real screen lists are not the cards the game handed to the host's selector", string.Join(",", traceFaces), string.Join(",", rawFaces), seq);
            return -1;
        }
        if (rec["displayed"] is JsonArray displayed && displayed.Count < cands.Count)
        {
            Explain(V2Classes.SelectionGridWindow);
        }
        if (c.Merged)
        {
            Explain(V2Classes.SelectionMergedClass, raw.Count - c.Options.Count);
        }
        // every listed card has a SELECT_CARD action (through its class), and every option stands for a listed card
        HashSet<int> ordinals = new HashSet<int>();
        for (int i = 0; i < raw.Count; i++)
        {
            int ordinal = view.OrdinalOf(raw[i]);
            ordinals.Add(ordinal);
            Item(rec, "candidate " + raw[i].Id.Entry, true, ordinal >= 0 && Has(d, Verb.SelectCard, RefKind.Card, ordinal));
        }
        foreach (LegalAction a in d.Actions)
        {
            if (a.Verb == Verb.SelectCard && !ordinals.Contains(a.Source.Index))
            {
                Item(rec, "option #" + a.Source.Index, false, true);
            }
        }

        // ---- before any pick: close <-> CANCEL, "finish with nothing" <-> SKIP ----
        JsonNode? before = rec["buttons_before"];
        bool hostCancel = Has(d, Verb.Cancel);
        bool hostSkip = Has(d, Verb.Skip);
        if (screen == "NChooseACardSelectionScreen")
        {
            JsonNode? skip = (before as JsonArray)?.FirstOrDefault(b => S(b!["role"]) == "skip");
            Item(rec, "skip button", B(skip?["clickable"]), hostSkip);
            Item(rec, "close", false, hostCancel);
        }
        else
        {
            bool uiZero = s.Hand ? UiCanFinish((before as JsonArray)?.FirstOrDefault()) : UiCanFinish(before);
            if (uiZero && !hostSkip && screen == "NDeckEnchantSelectScreen" && Rule(V2Classes.SelectionEnchantZeroConfirmInert))
            {
                _r.OfferItemsCompared++;
            }
            else
            {
                Item(rec, "finish with no card (confirm enabled before any pick)", uiZero, hostSkip);
            }
            Item(rec, "close", s.Hand ? false : B(rec["close_offered"]), hostCancel);
        }
        Item(rec, "confirm before any pick", false, Has(d, Verb.ConfirmSelection));

        // ---- the answer ----
        JsonNode? chosen = rec["chosen"];
        if (rec["_result"]?["steps"] is JsonArray steps)
        {
            s.HasResult = true;
            foreach (JsonNode? st in steps)
            {
                if (st is not JsonObject so)
                {
                    continue;
                }
                if (so.ContainsKey("picked"))
                {
                    s.PickSteps.Add(so);
                }
                else if (so.ContainsKey("click"))
                {
                    s.ConfirmClicks++;
                }
                else
                {
                    // extra_pick (the screen could not finish with the drawn picks) / warning (did not close): the recorded picks are not the whole answer
                    Diverge("record.unsupported_selection_step", true, where, "the selection_result has a step the comparison cannot replay", Js(so), null, seq);
                    return -1;
                }
            }
        }
        else if (screen != "NChooseACardSelectionScreen" && !(chosen is JsonValue cv && (S(cv) == "cancel" || S(cv) == "skip")))
        {
            Explain(V2Classes.SelectionResultMissing);
        }
        List<int> indices = new List<int>();
        if (chosen is JsonArray list)
        {
            indices.AddRange(list.Select(x => x!.GetValue<int>()));
            s.Plan = indices.Count == 0 ? "skip" : "picks";
        }
        else if (chosen is JsonValue v && v.GetValueKind() == System.Text.Json.JsonValueKind.Number)
        {
            indices.Add(v.GetValue<int>());
            s.Plan = "picks";
        }
        else
        {
            s.Plan = S(chosen) ?? "";
            if (s.Plan != "cancel" && s.Plan != "skip")
            {
                Diverge("record.inconsistent", true, where, "unknown selection answer", Js(chosen), null, seq);
                return -1;
            }
        }
        List<string> chosenFaces = indices.Select(i => i >= 0 && i < cands.Count ? Face(cands[i]) : "?").ToList();

        if (mcr.HasValue)
        {
            // the game's own record of the answer (PlayerChoice event of the fight's replay): exact instances / indexes
            List<CardModel>? fromMcr = McrPicks(mcr.Value, raw, where);
            if (fromMcr == null)
            {
                return -1;
            }
            List<string> mcrFaces = fromMcr.Select(x => Face(x)).ToList();
            if (!mcrFaces.OrderBy(x => x, StringComparer.Ordinal).SequenceEqual(chosenFaces.OrderBy(x => x, StringComparer.Ordinal)))
            {
                Diverge("record.mcr_vs_combat_json", true, where, "the answer in the fight's replay is not the answer in combat.json", string.Join(",", chosenFaces), string.Join(",", mcrFaces), seq);
                return -1;
            }
            s.Picks = fromMcr;
            if (fromMcr.Count == 0 && s.Plan == "picks")
            {
                s.Plan = "skip";
            }
        }
        else if (s.Plan == "picks")
        {
            foreach (int i in indices)
            {
                CardModel? instance = null;
                if (byIndex != null && i >= 0 && i < byIndex.Count)
                {
                    instance = byIndex[i];
                }
                else if (i >= 0 && i < cands.Count)
                {
                    // screen order (combat-pile grid, hand holders) without the game's replay: the first unpicked card with that face, in the order the game listed them
                    string face = Face(cands[i]);
                    instance = raw.FirstOrDefault(x => Face(x) == face && !s.Picks.Any(p => ReferenceEquals(p, x)));
                    if (raw.Count(x => Face(x) == face) > 1)
                    {
                        Explain(V2Classes.CombatNoMcrIdentity + ":prompt answer among identical cards");
                        if (_combat != null)
                        {
                            _combat.UnresolvedIdentities++;
                        }
                    }
                }
                if (instance == null)
                {
                    Diverge("record.inconsistent", true, where, "a chosen candidate index cannot be mapped to a card of the host's frame", Js(chosen), string.Join(",", rawFaces), seq);
                    return -1;
                }
                s.Picks.Add(instance);
            }
        }
        if (s.Picks.Count > 0 && s.HasResult && s.PickSteps.Count != s.Picks.Count)
        {
            Diverge("record.inconsistent", true, where, $"{s.Picks.Count} chosen cards but {s.PickSteps.Count} pick steps in the selection_result", Js(chosen), null, seq);
            return -1;
        }

        // the record is consumed now (the per-pick states are compared as the frame is answered)
        if (inCombat)
        {
            _combat!.PromptConsumed();
            _r.CombatPromptsReplayed++;
            _r.DecisionsCompared++;
            _r.ComparedByKind["selection(combat)"] = _r.ComparedByKind.GetValueOrDefault("selection(combat)") + 1;
            if (!_recordDiverged)
            {
                _r.DecisionsMatched++;
            }
        }
        else
        {
            Consume(rec, compared: true);
        }
        _r.ActionsApplied++;
        _sel = s;
        return SelectionStep(d);
    }

    /// <summary>The instances a PlayerChoice event of the fight's replay names (CardSelectCmd: indexes for the generated lists, combat cards for hand / piles).</summary>
    private List<CardModel>? McrPicks(NetPlayerChoiceResult net, IReadOnlyList<CardModel> raw, string where)
    {
        List<CardModel> picks = new List<CardModel>();
        if (net.indexes != null)
        {
            foreach (int i in net.indexes)
            {
                if (i < 0)
                {
                    continue;           // CardSelectCmd.FromChooseACardScreen: a negative index = skipped
                }
                if (i >= raw.Count)
                {
                    Diverge("record.mcr_choice_undecodable", true, where, "the replay's choice index is outside the host's candidate list", i.ToString(), raw.Count.ToString());
                    return null;
                }
                picks.Add(raw[i]);
            }
            return picks;
        }
        if (net.combatCards != null)
        {
            foreach (NetCombatCard nc in net.combatCards)
            {
                if (!NetCombatCardDb.Instance.TryGetCard(nc.CombatCardIndex, out CardModel? card) || card == null)
                {
                    Diverge("record.mcr_choice_undecodable", true, where, "the replay's choice names a combat card id the host's combat does not have", nc.CombatCardIndex.ToString(), null);
                    return null;
                }
                picks.Add(card);
            }
            return picks;
        }
        Diverge("record.mcr_choice_undecodable", true, where, "the replay's choice is of a type the comparison does not decode", net.type.ToString(), null);
        return null;
    }

    /// <summary>The next action of the selection being answered.</summary>
    private int SelectionStep(Decision d)
    {
        Sel s = _sel!;
        if (s.Plan == "cancel")
        {
            s.Ended = true;
            int cancel = Find(d, Verb.Cancel);
            if (cancel < 0)
            {
                Diverge("chosen_not_offered.selection", true, s.Where, "the real client closed the screen (cancel) and the host offers no CANCEL", Js(s.Rec["chosen"]), Describe(d), s.Seq);
            }
            return cancel;
        }
        if (s.Plan == "skip" && s.Applied == 0)
        {
            s.Ended = true;
            int skip = Find(d, Verb.Skip);
            if (skip < 0)
            {
                Diverge("chosen_not_offered.selection", true, s.Where, "the real client finished the screen with no card and the host offers no SKIP", Js(s.Rec["chosen"]), Describe(d), s.Seq);
            }
            return skip;
        }
        if (s.Applied < s.Picks.Count)
        {
            CardModel instance = s.Picks[s.Applied];
            int ordinal = s.View.OrdinalOf(instance);
            int index = ordinal < 0 ? -1 : Find(d, Verb.SelectCard, RefKind.Card, ordinal);
            if (index < 0)
            {
                Diverge("chosen_not_offered.selection", true, s.Where, $"the real client picked {Face(instance)} (pick {s.Applied + 1}) and the host has no SELECT_CARD for it", Js(s.Rec["chosen"]), Describe(d), s.Seq);
                return -1;
            }
            if (s.Ctx.Merged)
            {
                // which instance a pick of this class would hand out: the first unpicked one in the order the game listed them
                CardModel? first = s.View.RawOptions.FirstOrDefault(x => s.View.OrdinalOf(x) == ordinal && !s.Picked.Contains(x));
                if (!ReferenceEquals(first, instance))
                {
                    Explain(V2Classes.SelectionNotFirstInstance);
                }
                if (!s.View.PreferInstance(instance))
                {
                    Diverge("tool.prefer_instance", true, s.Where, "the host view refused the recorded instance", Face(instance), null, s.Seq);
                    return -1;
                }
            }
            s.Picked.Add(instance);
            s.Applied++;
            return index;
        }
        // every recorded pick is applied and the frame is still open: it wants a confirmation (or more cards)
        int confirm = Find(d, Verb.ConfirmSelection);
        if (confirm < 0)
        {
            Diverge("selection.host_needs_more_picks", true, s.Where, "the real screen finished with the recorded picks; the host's frame is still open and offers no CONFIRM", Js(s.Rec["chosen"]), Describe(d), s.Seq);
            return -1;
        }
        s.Ended = true;
        return confirm;
    }

    /// <summary>
    /// Called at the top of every decision: when a selection is being answered, what the frame did with the last action
    /// (still open and offering CONFIRM or not / gone) is compared with what the real screen showed after that pick.
    /// </summary>
    private void ResolveSelectionStep(Decision d)
    {
        Sel? s = _sel;
        if (s == null)
        {
            return;
        }
        bool same = ReferenceEquals(d.Source, s.Frame);
        if (s.Ended)
        {
            _sel = null;
            if (same)
            {
                Diverge("selection.host_frame_still_open", true, s.Where, "the real screen closed after this answer; the host's frame is still asking", Js(s.Rec["chosen"]), Describe(d), s.Seq);
            }
            return;
        }
        // a pick was applied
        JsonNode? step = s.HasResult && s.Applied - 1 < s.PickSteps.Count ? s.PickSteps[s.Applied - 1] : null;
        JsonNode? buttons = step == null ? null : s.Hand ? step["confirm"] : step["buttons"];
        bool uiClosed = buttons is JsonArray arr && arr.Count == 0;
        bool uiCanFinish = uiClosed || UiCanFinish(buttons);
        bool last = s.Applied == s.Picks.Count;
        if (!same)
        {
            _sel = null;
            if (!last)
            {
                Diverge("selection.host_completed_early", true, s.Where, $"the host's frame completed after {s.Applied} of {s.Picks.Count} recorded picks", Js(s.Rec["chosen"]), Describe(d), s.Seq);
                return;
            }
            if (step == null)
            {
                return;       // no result record: nothing to compare (counted when the prompt started)
            }
            _r.OfferItemsCompared++;
            if (!uiCanFinish)
            {
                Diverge("offer.selection.host_only", false, s.Where, "the host's frame completed with this pick while the real screen could not be finished yet", Js(step), "completed", s.Seq);
            }
            else if (!uiClosed)
            {
                // the real screen asks for one more click (preview confirm / the hand's confirm button); the host completes with the pick
                Explain(V2Classes.SelectionPreviewConfirm);   // a mapping, not a tolerance: counted wherever it is used
            }
            return;
        }
        if (step == null)
        {
            return;
        }
        bool hostConfirm = Has(d, Verb.ConfirmSelection);
        _r.OfferItemsCompared++;
        if (uiClosed)
        {
            Diverge("offer.selection.ui_only", false, s.Where, $"the real screen closed with pick {s.Applied}; the host's frame is still open", Js(step), Describe(d), s.Seq);
        }
        else if (uiCanFinish != hostConfirm)
        {
            Diverge("offer.selection." + (uiCanFinish ? "ui_only" : "host_only"), false, s.Where,
                uiCanFinish ? $"after pick {s.Applied} the real screen can be confirmed and the host offers no CONFIRM" : $"after pick {s.Applied} the host offers CONFIRM and the real screen has no enabled confirm button",
                Js(step), Describe(d), s.Seq);
        }
        if (!s.Hand)
        {
            bool uiClose = UiCloseOffered(buttons);
            bool hostCancel = Has(d, Verb.Cancel);
            _r.OfferItemsCompared++;
            if (hostCancel && !uiClose)
            {
                if (!(B(step["preview_open"]) && Rule(V2Classes.SelectionCancelBehindPreview)))
                {
                    Diverge("offer.selection.host_only", false, s.Where, $"after pick {s.Applied} the host offers CANCEL and the real screen's Close button is not enabled (no preview open)", Js(step), Describe(d), s.Seq);
                }
            }
            else if (!hostCancel && uiClose)
            {
                Diverge("offer.selection.ui_only", false, s.Where, $"after pick {s.Applied} the real screen's Close button is enabled and the host offers no CANCEL", Js(step), Describe(d), s.Seq);
            }
        }
    }
}
