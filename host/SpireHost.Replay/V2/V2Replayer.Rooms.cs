using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using SpireHost.Driver;

namespace SpireHost.Replay.V2;

public sealed partial class V2Replayer
{
    private readonly List<string> _hostCardOffers = new List<string>();   // the tracer's card_reward_offers list, rebuilt from the host's CARD_REWARD decisions
    private List<string>? _crystalRevealed;                               // items revealed in the minigame being played (the collector's strings)
    private CrystalSphereMinigame? _crystalMinigame;
    private JsonObject? _crystalPending;                                  // the crystal record whose "after" block is compared at the next decision
    private int _crystalHiddenBefore;
    private int _crystalRevealedBefore;
    private bool _awaitingShopClose;                                      // shop leave was applied: close_inventory / proceed UI records follow

    // =====================================================================================================
    //  Dispatch
    // =====================================================================================================

    private int RoomDecision(Decision d, RunContext ctx)
    {
        ResolveCrystalResult(d, ctx);
        while (!_stop)
        {
            JsonObject? rec = PeekRecord();
            if (rec == null)
            {
                Diverge("end.host_decision_without_record", true, $"decision {d.Sequence} floor {ctx.Run.TotalFloor}", "the host asks for a decision but the trace has no record left (trace end: " + _t.End + ")", null, Describe(d));
                return -1;
            }
            _recordDiverged = false;
            switch (Kind(rec))
            {
                case "shop_ui":
                {
                    int index = ShopUi(rec, d, ctx);
                    if (index >= 0)
                    {
                        return index;
                    }
                    continue;
                }
                case "potion":
                    return Potion(rec, d, ctx);
                case "event":
                    return Event(rec, d, ctx);
                case "map":
                    return Map(rec, d, ctx);
                case "rewards":
                {
                    int index = Rewards(rec, d, ctx);
                    if (index == NoAction)
                    {
                        continue;
                    }
                    return index;
                }
                case "card_reward":
                    return CardReward(rec, d, ctx);
                case "rest":
                    return Rest(rec, d, ctx);
                case "shop":
                {
                    int index = Shop(rec, d, ctx);
                    if (index == NoAction)
                    {
                        continue;
                    }
                    return index;
                }
                case "treasure":
                    return Treasure(rec, d, ctx);
                case "selection":
                    return Selection(rec, d, ctx, null);
                case "bundle":
                    return Bundle(rec, d, ctx);
                case "crystal":
                    return Crystal(rec, d, ctx);
                case "confirm":
                    return Confirm(rec, d, ctx);
                default:
                    Diverge("record.unsupported_kind", true, Where(rec), "the comparison has no mapping for this record kind", Js(rec["chosen"]), Describe(d), I(rec["seq"]));
                    return -1;
            }
        }
        return -1;
    }

    /// <summary>Returned by a record handler that consumed a record which maps to no host action.</summary>
    private const int NoAction = -2;

    private bool Expect(Decision d, JsonObject rec, DecisionKind want, bool alsoCheckContext = true)
    {
        if (d.Kind != want)
        {
            Diverge("decision_kind", true, Where(rec), $"the trace's next record is '{Kind(rec)}' where the host asks a {d.Kind} decision", Js(rec["chosen"]), Describe(d), I(rec["seq"]));
            return false;
        }
        return !alsoCheckContext || CheckContext(rec, d.Episode);
    }

    private void Content(JsonObject rec, string field, string? trace, string? host)
    {
        _r.ContentFieldsCompared++;
        if (trace != host)
        {
            Diverge("content." + Kind(rec) + "." + field, false, Where(rec), "public content differs: " + field, trace, host, I(rec["seq"]));
        }
    }

    private void Content(JsonObject rec, string field, bool trace, bool host) => Content(rec, field, trace ? "true" : "false", host ? "true" : "false");

    /// <summary>One UI item against one host action: both enabled, or both not.</summary>
    private void Item(JsonObject rec, string what, bool uiEnabled, bool hostLegal)
    {
        _r.OfferItemsCompared++;
        if (uiEnabled == hostLegal)
        {
            return;
        }
        Diverge("offer." + Kind(rec) + (uiEnabled ? ".ui_only" : ".host_only"), false, Where(rec),
            uiEnabled ? "the real UI offers '" + what + "' (enabled) and the host has no legal action for it" : "the host offers '" + what + "' and the real UI does not (absent or not enabled)",
            Js(rec["offer"] ?? rec["buttons"] ?? rec["cards"] ?? rec["belt"]), _d != null ? Describe(_d) : null, I(rec["seq"]));
    }

    private static string? LocKey(LocString? s) => s == null ? null : s.LocTable + "." + s.LocEntryKey;

    private static string Face(JsonNode? c)
    {
        if (c == null)
        {
            return "null";
        }
        return $"{S(c["id"])}|{I(c["upg"], 0)}|{S(c["ench"]) ?? ""}|{S(c["aff"]) ?? ""}";
    }

    private static string Face(CardModel? c)
    {
        if (c == null)
        {
            return "null";
        }
        return $"{c.Id.Entry}|{c.CurrentUpgradeLevel}|{(c.Enchantment != null ? c.Enchantment.Id.Entry + ":" + c.Enchantment.Amount : "")}|{c.Affliction?.Id.Entry ?? ""}";
    }

    private int Chosen(JsonObject rec, Decision d, int index, string what)
    {
        if (index < 0)
        {
            Diverge("chosen_not_offered." + Kind(rec), true, Where(rec), "the action the real client took (" + what + ") is not a legal host action", Js(rec["chosen"]), Describe(d), I(rec["seq"]));
            return -1;
        }
        Consume(rec, compared: true);
        _r.ActionsApplied++;
        return index;
    }

    // =====================================================================================================
    //  The potion belt on room-level frames (NPotionPopup of every slot: Use / Discard enabled)
    // =====================================================================================================

    /// <summary>
    /// rec["belt"]: per top-bar slot {slot, id, usage, usable, discardable, popup}; usable / discardable are the enabled
    /// states of the real popup's buttons. Host: USE_POTION(slot) / DISCARD_POTION(slot) on the frame (rooms:potions).
    /// </summary>
    private void Belt(JsonObject rec, Decision d, RunContext ctx)
    {
        if (rec["belt"] is not JsonArray belt)
        {
            return;
        }
        Player me = ctx.LocalPlayer;
        IReadOnlyList<PotionModel?> slots = me.PotionSlots;
        Content(rec, "belt.slots", belt.Count.ToString(), slots.Count.ToString());
        for (int i = 0; i < belt.Count && i < slots.Count; i++)
        {
            JsonNode b = belt[i]!;
            string? id = S(b["id"]);
            Content(rec, "belt.id", id, slots[i]?.Id.Entry);
            if (id == null || slots[i] == null)
            {
                Item(rec, "potion slot " + i + " (empty in the trace)", false, Has(d, Verb.UsePotion, RefKind.PotionSlot, i) || Has(d, Verb.DiscardPotion, RefKind.PotionSlot, i));
                continue;
            }
            Content(rec, "belt.usage", S(b["usage"]), slots[i]!.Usage.ToString());
            if (!B(b["popup"]))
            {
                // the popup did not open in the real client: its buttons were not read
                Diverge("offer.belt.popup_not_opened", false, Where(rec), "the real potion popup of slot " + i + " did not open; its Use / Discard state is unknown", Js(b), null, I(rec["seq"]));
                continue;
            }
            Item(rec, "use potion slot " + i + " " + id, B(b["usable"]), Has(d, Verb.UsePotion, RefKind.PotionSlot, i));
            Item(rec, "discard potion slot " + i + " " + id, B(b["discardable"]), Has(d, Verb.DiscardPotion, RefKind.PotionSlot, i));
        }
    }

    private int Potion(JsonObject rec, Decision d, RunContext ctx)
    {
        string where = S(rec["where"]) ?? "";
        DecisionKind want = where switch
        {
            "map" => DecisionKind.Map,
            "event" => DecisionKind.Event,
            "rewards" => DecisionKind.Reward,
            "shop" or "fake_merchant" => DecisionKind.Shop,
            "rest" => DecisionKind.Rest,
            "treasure_chest" or "treasure_relic" => DecisionKind.Treasure,
            _ => DecisionKind.None,
        };
        if (!Expect(d, rec, want))
        {
            return -1;
        }
        Belt(rec, d, ctx);
        JsonNode chosen = rec["chosen"]!;
        int slot = I(chosen["slot"]);
        bool use = S(chosen["action"]) == "use";
        Content(rec, "chosen.potion", S(chosen["id"]), slot >= 0 && slot < ctx.LocalPlayer.PotionSlots.Count ? ctx.LocalPlayer.PotionSlots[slot]?.Id.Entry : null);
        return Chosen(rec, d, Find(d, use ? Verb.UsePotion : Verb.DiscardPotion, RefKind.PotionSlot, slot), (use ? "use" : "discard") + " potion slot " + slot);
    }

    // =====================================================================================================
    //  EVENT (NEventRoom.Layout.OptionButtons)
    // =====================================================================================================

    private int Event(JsonObject rec, Decision d, RunContext ctx)
    {
        if (!Expect(d, rec, DecisionKind.Event))
        {
            return -1;
        }
        if (d.Context is not EventContext ec)
        {
            Diverge("decision_context", true, Where(rec), "EVENT decision without an EventContext", null, Describe(d), I(rec["seq"]));
            return -1;
        }
        Belt(rec, d, ctx);
        Content(rec, "event", S(rec["event"]), ec.EventId);
        Content(rec, "event_type", S(rec["event_type"]), ec.Event.GetType().Name);
        Content(rec, "page_description", S(rec["page_description"]), LocKey(ec.Event.Description));
        Content(rec, "is_finished", B(rec["is_finished"]), ec.Finished);
        Content(rec, "is_shared", B(rec["is_shared"]), ec.IsShared);
        JsonArray offer = rec["offer"]!.AsArray();
        Content(rec, "options", offer.Count.ToString(), ec.Options.Count.ToString());
        int legalOptions = 0;
        for (int i = 0; i < offer.Count && i < ec.Options.Count; i++)
        {
            JsonNode o = offer[i]!;
            EventOptionInfo h = ec.Options[i];
            bool synth = B(o["synthesized_proceed"]);
            Content(rec, "option.text_key", S(o["text_key"]), h.TextKey);
            Content(rec, "option.model_index", S(o["model_index"]), h.Index.ToString());
            Content(rec, "option.synthesized", synth, h.Synthesized);
            Content(rec, "option.locked", B(o["locked"]), h.IsLocked);
            Content(rec, "option.proceed", B(o["proceed"]), h.IsProceed);
            Content(rec, "option.was_chosen", B(o["was_chosen"]), h.WasChosen);
            Content(rec, "option.kills", B(o["kills"]), h.WillKillPlayer);
            Content(rec, "option.relic", S(o["relic"]), h.Relic?.Id.Entry);
            if (!synth)
            {
                Content(rec, "option.title_key", S(o["title_key"]), LocKey(h.Option?.Title));
            }
            bool clickable = B(o["clickable"]);
            bool locked = B(o["locked"]);
            bool uiEnabled = clickable && !locked;
            if (clickable && locked && !Rule(V2Classes.EventLockedOption))
            {
                uiEnabled = true;
            }
            bool hostLegal = h.Synthesized ? Has(d, Verb.Proceed) : Has(d, Verb.ChooseOption, RefKind.EventOption, h.Index);
            if (hostLegal)
            {
                legalOptions++;
            }
            Item(rec, "option " + S(o["text_key"]), uiEnabled, hostLegal);
        }
        int hostActions = d.Actions.Count(a => !IsPotionVerb(a.Verb));
        if (hostActions != legalOptions)
        {
            Diverge("offer.event.host_only", false, Where(rec), $"the host has {hostActions} event actions but only {legalOptions} of them correspond to a displayed option", Js(offer), Describe(d), I(rec["seq"]));
        }
        JsonNode chosen = rec["chosen"]!;
        int button = I(chosen["index"]);
        JsonNode? pick = offer.FirstOrDefault(o => I(o!["index"]) == button);
        if (pick == null)
        {
            Diverge("record.inconsistent", true, Where(rec), "the chosen option index is not in the recorded offer", Js(chosen), null, I(rec["seq"]));
            return -1;
        }
        int index = B(pick["synthesized_proceed"]) ? Find(d, Verb.Proceed) : Find(d, Verb.ChooseOption, RefKind.EventOption, I(pick["model_index"]));
        return Chosen(rec, d, index, "event option " + S(pick["text_key"]));
    }

    // =====================================================================================================
    //  MAP (NMapPoint.IsEnabled)
    // =====================================================================================================

    private int Map(JsonObject rec, Decision d, RunContext ctx)
    {
        if (!Expect(d, rec, DecisionKind.Map))
        {
            return -1;
        }
        Belt(rec, d, ctx);
        JsonArray offer = rec["offer"]!.AsArray();
        HashSet<int> ui = new HashSet<int>();
        foreach (JsonNode? p in offer)
        {
            int col = I(p!["col"]), row = I(p["row"]);
            int ordinal = row * 64 + col;
            ui.Add(ordinal);
            int at = Find(d, Verb.Travel, RefKind.MapPoint, ordinal);
            Item(rec, $"map point {col},{row}", true, at >= 0);
            if (at >= 0)
            {
                MapPoint mp = d.Actions[at].Source.MapPoint;
                Content(rec, "point.type", S(p["type"]), mp.PointType.ToString());
            }
        }
        foreach (LegalAction a in d.Actions)
        {
            if (a.Verb == Verb.Travel && !ui.Contains(a.Source.Index))
            {
                Item(rec, $"map point {a.Source.Index % 64},{a.Source.Index / 64}", false, true);
            }
        }
        JsonNode chosen = rec["chosen"]!;
        return Chosen(rec, d, Find(d, Verb.Travel, RefKind.MapPoint, I(chosen["row"]) * 64 + I(chosen["col"])), $"travel to {I(chosen["col"])},{I(chosen["row"])}");
    }

    // =====================================================================================================
    //  REWARDS (NRewardsScreen._rewardButtons, _proceedButton)
    // =====================================================================================================

    private static string RewardContent(Reward r)
    {
        switch (r)
        {
            case GoldReward g:
                return "GoldReward:" + g.Amount;
            case PotionReward p:
                return "PotionReward:" + p.Potion?.Id.Entry;
            case RelicReward rr:
                return "RelicReward:" + rr.Relic?.Id.Entry;
            case CardReward cr:
                return "CardReward:reroll=" + (cr.CanReroll ? "true" : "false") + ",skip=" + (cr.CanSkip ? "true" : "false");
            case SpecialCardReward sc:
                return "SpecialCardReward:" + Face(sc._card);
            default:
                return r.GetType().Name + ":";
        }
    }

    private static string RewardContent(JsonNode o)
    {
        string type = S(o["type"]) ?? "";
        switch (type)
        {
            case "GoldReward":
                return type + ":" + S(o["gold"]);
            case "PotionReward":
                return type + ":" + S(o["potion"]);
            case "RelicReward":
                return type + ":" + S(o["relic"]);
            case "CardReward":
                return type + ":reroll=" + (B(o["can_reroll"]) ? "true" : "false") + ",skip=" + (B(o["can_skip"]) ? "true" : "false");
            case "SpecialCardReward":
                return type + ":" + Face(o["card"]);
            default:
                return type + ":";
        }
    }

    private int Rewards(JsonObject rec, Decision d, RunContext ctx)
    {
        if (!Expect(d, rec, DecisionKind.Reward))
        {
            return -1;
        }
        if (d.Context is not RewardContext rc)
        {
            Diverge("decision_context", true, Where(rec), "REWARD decision without a RewardContext", null, Describe(d), I(rec["seq"]));
            return -1;
        }
        Player me = ctx.LocalPlayer;
        Belt(rec, d, ctx);
        Content(rec, "terminal", B(rec["terminal"]), rc.Terminal);
        List<Reward> rewards = rc.Set.Rewards;
        JsonArray offer = rec["offer"]!.AsArray();
        HashSet<int> shown = new HashSet<int>();
        bool beltFull = !me.HasOpenPotionSlots;
        foreach (JsonNode? o in offer)
        {
            int si = I(o!["set_index"]);
            shown.Add(si);
            if (si < 0 || si >= rewards.Count)
            {
                Diverge("content.rewards.set_index", false, Where(rec), "the real screen shows a reward whose set index the host's reward set does not have", Js(o), string.Join(",", rewards.Select(RewardContent)), I(rec["seq"]));
                continue;
            }
            Reward r = rewards[si];
            Content(rec, "reward", RewardContent(o), RewardContent(r));
            if (r.SuccessfullySelected)
            {
                Diverge("content.rewards.taken", false, Where(rec), "the real screen still shows a reward that the host has taken", Js(o), RewardContent(r), I(rec["seq"]));
            }
            bool ui = B(o["enabled"]) && B(o["clickable"]);
            bool host = Has(d, Verb.ChooseOption, RefKind.Reward, si);
            if (ui && !host && r is PotionReward pr && beltFull && pr.Potion != null && Hook.ShouldProcurePotion(ctx.Run, me.Creature.CombatState, pr.Potion, me) && Rule(V2Classes.RewardPotionFullBelt))
            {
                _r.OfferItemsCompared++;
                continue;
            }
            Item(rec, "reward " + RewardContent(o), ui, host);
        }
        for (int i = 0; i < rewards.Count; i++)
        {
            if (!rewards[i].SuccessfullySelected && !shown.Contains(i))
            {
                Diverge("content.rewards.missing_button", false, Where(rec), "the host's reward set has an untaken reward the real screen does not show", Js(offer), i + ":" + RewardContent(rewards[i]), I(rec["seq"]));
            }
        }
        foreach (LegalAction a in d.Actions)
        {
            if (a.Verb == Verb.ChooseOption && !shown.Contains(a.Source.Index))
            {
                Item(rec, "reward #" + a.Source.Index, false, true);
            }
        }
        JsonNode proceed = rec["proceed"]!;
        Item(rec, "proceed", B(proceed["clickable"]), Has(d, Verb.Proceed));
        Content(rec, "proceed.disallow_skipping", B(proceed["disallow_skipping"]), rc.Set.DisallowSkipping);
        Content(rec, "proceed.is_skip", B(proceed["is_skip"]), rewards.Any(r => !r.SuccessfullySelected));

        JsonNode chosen = rec["chosen"]!;
        string action = S(chosen["action"]) ?? "";
        switch (action)
        {
            case "take":
            case "take_forced":
                return Chosen(rec, d, Find(d, Verb.ChooseOption, RefKind.Reward, I(chosen["set_index"])), action + " reward " + I(chosen["set_index"]) + " " + S(chosen["type"]));
            case "proceed":
                return Chosen(rec, d, Find(d, Verb.Proceed), "rewards proceed");
            case "skip_full_belt":
            {
                // the collector passed over a potion reward because the belt is full: it clicked nothing
                int si = I(chosen["set_index"]);
                if (!beltFull || si < 0 || si >= rewards.Count || rewards[si] is not PotionReward)
                {
                    Diverge("record.inconsistent", true, Where(rec), "skip_full_belt on something that is not a potion reward with a full belt in the host", Js(chosen), si >= 0 && si < rewards.Count ? RewardContent(rewards[si]) : null, I(rec["seq"]));
                    return -1;
                }
                if (!Rule(V2Classes.RewardSkipFullBeltRecord))
                {
                    Diverge("record.no_host_action", true, Where(rec), "a skip_full_belt record maps to no host action", Js(chosen), null, I(rec["seq"]));
                    return -1;
                }
                Consume(rec, compared: true);
                return NoAction;
            }
            default:
                Diverge("record.unsupported_kind", true, Where(rec), "unknown rewards action '" + action + "'", Js(chosen), null, I(rec["seq"]));
                return -1;
        }
    }

    // =====================================================================================================
    //  CARD REWARD (NCardRewardSelectionScreen._cardRow, _rewardAlternativesContainer)
    // =====================================================================================================

    private int CardReward(JsonObject rec, Decision d, RunContext ctx)
    {
        if (!Expect(d, rec, DecisionKind.CardReward))
        {
            return -1;
        }
        if (d.Context is not CardRewardContext cc)
        {
            Diverge("decision_context", true, Where(rec), "CARD_REWARD decision without a CardRewardContext", null, Describe(d), I(rec["seq"]));
            return -1;
        }
        JsonArray cards = rec["cards"]!.AsArray();
        JsonArray alts = rec["alternatives"]!.AsArray();
        Content(rec, "cards", cards.Count.ToString(), cc.Options.Count.ToString());
        _hostCardOffers.Add(string.Join(",", cc.Options.Select(o => o.Card.Id.Entry + (o.Card.IsUpgraded ? "+" : ""))));
        HashSet<int> shown = new HashSet<int>();
        foreach (JsonNode? c in cards)
        {
            int oi = I(c!["index"]);
            shown.Add(oi);
            if (oi < 0 || oi >= cc.Options.Count)
            {
                Diverge("content.card_reward.index", false, Where(rec), "a displayed card is not one of the host's options", Js(c), null, I(rec["seq"]));
                continue;
            }
            CardCreationResult h = cc.Options[oi];
            Content(rec, "card", Face(c), Face(h.Card));
            Content(rec, "card.modified_by", c["modified_by"] is JsonArray m ? string.Join(",", m.Select(x => S(x))) : "", string.Join(",", h.ModifyingRelics.Select(x => x.Id.Entry)));
            Item(rec, "card " + S(c["id"]), B(c["clickable"]), Has(d, Verb.PickOfferCard, RefKind.Card, oi));
        }
        for (int i = 0; i < cc.Options.Count; i++)
        {
            if (!shown.Contains(i))
            {
                Item(rec, "card #" + i + " " + cc.Options[i].Card.Id.Entry, false, Has(d, Verb.PickOfferCard, RefKind.Card, i));
            }
        }
        Content(rec, "alternatives", alts.Count.ToString(), cc.Alternatives.Count.ToString());
        for (int k = 0; k < alts.Count; k++)
        {
            JsonNode a = alts[k]!;
            int ai = I(a["index"]);
            if (ai >= 0 && ai < cc.Alternatives.Count)
            {
                CardRewardAlternative h = cc.Alternatives[ai];
                Content(rec, "alternative.id", S(a["id"]), h.OptionId);
                Content(rec, "alternative.after", S(a["after"]), h.AfterSelected.ToString());
            }
            Item(rec, "alternative " + S(a["id"]), B(a["clickable"]), Has(d, Verb.ChooseOption, RefKind.CardRewardAlt, ai));
        }
        for (int k = alts.Count; k < cc.Alternatives.Count; k++)
        {
            Item(rec, "alternative #" + k + " " + cc.Alternatives[k].OptionId, false, Has(d, Verb.ChooseOption, RefKind.CardRewardAlt, k));
        }
        if (rec["chosen"] is not JsonObject chosen)
        {
            Diverge("record.unsupported_kind", true, Where(rec), "card reward record without a choice (nothing clickable)", Js(rec["chosen"]), Describe(d), I(rec["seq"]));
            return -1;
        }
        bool card = S(chosen["kind"]) == "card";
        int idx = I(chosen["index"]);
        return Chosen(rec, d, card ? Find(d, Verb.PickOfferCard, RefKind.Card, idx) : Find(d, Verb.ChooseOption, RefKind.CardRewardAlt, idx), (card ? "card " : "alternative ") + S(chosen["id"]));
    }

    // =====================================================================================================
    //  REST SITE (NRestSiteButton, NRestSiteRoom.ProceedButton)
    // =====================================================================================================

    private int Rest(JsonObject rec, Decision d, RunContext ctx)
    {
        if (!Expect(d, rec, DecisionKind.Rest))
        {
            return -1;
        }
        if (d.Context is not RestContext rc)
        {
            Diverge("decision_context", true, Where(rec), "REST decision without a RestContext", null, Describe(d), I(rec["seq"]));
            return -1;
        }
        Belt(rec, d, ctx);
        JsonArray offer = rec["offer"]!.AsArray();
        JsonNode proceed = rec["proceed"]!;
        bool proceedClickable = B(proceed["clickable"]);
        HashSet<int> shown = new HashSet<int>();
        foreach (JsonNode? o in offer)
        {
            int oi = I(o!["index"]);
            if (oi < 0)
            {
                // a freed button of an option that is no longer in the synchronizer's list
                if (!Rule(V2Classes.RestStaleButton))
                {
                    Item(rec, "stale rest button " + S(o["id"]), B(o["clickable"]), false);
                }
                else if (B(o["clickable"]))
                {
                    Diverge("offer.rest.ui_only", false, Where(rec), "a rest button that is not in the option list is clickable", Js(o), Describe(d), I(rec["seq"]));
                }
                continue;
            }
            shown.Add(oi);
            if (oi >= rc.Options.Count)
            {
                Diverge("content.rest.index", false, Where(rec), "a displayed rest option is not in the host's option list", Js(o), string.Join(",", rc.Options.Select(x => x.OptionId)), I(rec["seq"]));
                continue;
            }
            RestSiteOption h = rc.Options[oi];
            Content(rec, "option.id", S(o["id"]), h.OptionId);
            Content(rec, "option.enabled", B(o["option_enabled"]), h.IsEnabled);
            bool ui = B(o["clickable"]);
            bool host = Has(d, Verb.ChooseOption, RefKind.RestOption, oi);
            if (!ui && host && B(o["enabled"]) && B(o["option_enabled"]) && !B(o["unclickable"]) && proceedClickable && rc.Successes > 0 && Rule(V2Classes.RestButtonFadingIn))
            {
                // a NEW button of a remaining option, recorded inside its fade-in (MouseFilter not yet Stop)
                _r.OfferItemsCompared++;
                continue;
            }
            Item(rec, "rest option " + S(o["id"]), ui, host);
        }
        for (int i = 0; i < rc.Options.Count; i++)
        {
            if (!shown.Contains(i))
            {
                Diverge("content.rest.missing_button", false, Where(rec), "the host's option list has an option the real room shows no button for", Js(offer), i + ":" + rc.Options[i].OptionId, I(rec["seq"]));
            }
        }
        Item(rec, "proceed", proceedClickable, Has(d, Verb.Proceed));
        JsonNode chosen = rec["chosen"]!;
        if (S(chosen["action"]) == "proceed")
        {
            return Chosen(rec, d, Find(d, Verb.Proceed), "rest proceed");
        }
        return Chosen(rec, d, Find(d, Verb.ChooseOption, RefKind.RestOption, I(chosen["index"])), "rest option " + S(chosen["id"]));
    }

    // =====================================================================================================
    //  SHOP and FakeMerchant (NMerchantInventory.GetAllSlots)
    // =====================================================================================================

    private static string EntryContent(MerchantEntry e)
    {
        string s = e.GetType().Name + "|stocked=" + (e.IsStocked ? "true" : "false") + "|price=" + (e.IsStocked ? e.Cost.ToString() : "null") + "|enough=" + (e.IsStocked && e.EnoughGold ? "true" : "false");
        switch (e)
        {
            case MerchantCardEntry ce:
                return s + "|card=" + Face(ce.CreationResult?.Card) + "|sale=" + (ce.IsOnSale ? "true" : "false");
            case MerchantRelicEntry re:
                return s + "|relic=" + re.Model?.Id.Entry;
            case MerchantPotionEntry pe:
                return s + "|potion=" + pe.Model?.Id.Entry;
            case MerchantCardRemovalEntry rm:
                return s + "|used=" + (rm.Used ? "true" : "false");
            default:
                return s;
        }
    }

    private static string EntryContent(JsonNode o)
    {
        string s = S(o["entry"]) + "|stocked=" + (B(o["stocked"]) ? "true" : "false") + "|price=" + (o["price"] == null ? "null" : S(o["price"])) + "|enough=" + (B(o["enough_gold"]) ? "true" : "false");
        switch (S(o["kind"]))
        {
            case "card":
                return s + "|card=" + Face(o["card"]) + "|sale=" + (B(o["on_sale"]) ? "true" : "false");
            case "relic":
                return s + "|relic=" + S(o["model"]);
            case "potion":
                return s + "|potion=" + S(o["model"]);
            case "card_removal":
                return s + "|used=" + (B(o["used"]) ? "true" : "false");
            default:
                return s;
        }
    }

    private int Shop(JsonObject rec, Decision d, RunContext ctx)
    {
        if (!Expect(d, rec, DecisionKind.Shop))
        {
            return -1;
        }
        if (d.Context is not ShopContext sc)
        {
            Diverge("decision_context", true, Where(rec), "SHOP decision without a ShopContext", null, Describe(d), I(rec["seq"]));
            return -1;
        }
        Player me = ctx.LocalPlayer;
        Belt(rec, d, ctx);
        Content(rec, "fake_merchant", B(rec["fake_merchant"]), sc.FakeMerchant);
        List<MerchantEntry> entries = sc.Inventory.AllEntries;
        JsonArray offer = rec["offer"]!.AsArray();
        Content(rec, "entries", offer.Count.ToString(), entries.Count.ToString());
        bool beltFull = rec["belt"] is JsonArray belt ? belt.All(b => b!["id"] != null) : !me.HasOpenPotionSlots;
        HashSet<int> shown = new HashSet<int>();
        foreach (JsonNode? o in offer)
        {
            int ei = I(o!["index"]);
            shown.Add(ei);
            if (ei < 0 || ei >= entries.Count)
            {
                Diverge("content.shop.index", false, Where(rec), "a displayed slot is not one of the host's inventory entries", Js(o), null, I(rec["seq"]));
                continue;
            }
            MerchantEntry e = entries[ei];
            Content(rec, "entry", EntryContent(o), EntryContent(e));
            bool host = Has(d, Verb.ChooseOption, RefKind.ShopEntry, ei);
            bool stocked = B(o["stocked"]);
            bool visible = B(o["visible"]);
            // the real slot is clickable whenever it is stocked and visible; what a click does:
            bool effective = stocked && visible && B(o["enough_gold"]);
            if (stocked && visible && !B(o["enough_gold"]))
            {
                if (!Rule(V2Classes.ShopUnaffordableSlot))
                {
                    effective = true;
                }
            }
            else if (effective && S(o["kind"]) == "potion" && beltFull)
            {
                if (Rule(V2Classes.ShopPotionFullBelt))
                {
                    effective = false;
                }
            }
            else if (effective && S(o["kind"]) == "potion" && !host && e is MerchantPotionEntry pe && pe.Model != null && !Hook.ShouldProcurePotion(ctx.Run, me.Creature.CombatState, pe.Model, me))
            {
                if (Rule(V2Classes.ShopPotionForbidden))
                {
                    effective = false;
                }
            }
            Item(rec, "shop entry " + ei + " " + S(o["kind"]), effective, host);
        }
        foreach (LegalAction a in d.Actions)
        {
            if (a.Verb == Verb.ChooseOption && !shown.Contains(a.Source.Index))
            {
                Item(rec, "shop entry #" + a.Source.Index, false, true);
            }
        }
        Item(rec, "leave (inventory back button)", B(rec["back"]?["clickable"]), Has(d, Verb.Proceed));

        JsonNode chosen = rec["chosen"]!;
        if (S(chosen["action"]) == "leave")
        {
            _awaitingShopClose = true;
            return Chosen(rec, d, Find(d, Verb.Proceed), "leave the shop");
        }
        int buy = I(chosen["entry"]?["index"]);
        int index = Find(d, Verb.ChooseOption, RefKind.ShopEntry, buy);
        if (index < 0 && buy >= 0 && buy < entries.Count && RecordedClickWasRefused(rec, entries[buy], ctx))
        {
            Consume(rec, compared: true);
            return NoAction;
        }
        return Chosen(rec, d, index, "buy entry " + buy + " " + S(chosen["entry"]?["kind"]));
    }

    /// <summary>
    /// The collector clicked a potion slot that the game refuses (Hook.ShouldProcurePotion false: Sozu): the purchase
    /// did not happen in the real client either - the next record of the same shop shows the same gold and the entry
    /// still stocked. Only then is the record consumed without a host action.
    /// </summary>
    private bool RecordedClickWasRefused(JsonObject rec, MerchantEntry entry, RunContext ctx)
    {
        Player me = ctx.LocalPlayer;
        if (entry is not MerchantPotionEntry pe || pe.Model == null || !pe.IsStocked || !pe.EnoughGold || !me.HasOpenPotionSlots || Hook.ShouldProcurePotion(ctx.Run, me.Creature.CombatState, pe.Model, me))
        {
            return false;
        }
        int at = _rec + 1;
        if (at >= _t.Records.Count)
        {
            return false;
        }
        JsonObject next = _t.Records[at];
        if (S(next["gold"]) != S(rec["gold"]) || Js(next["belt"]) != Js(rec["belt"]) || S(next["floor"]) != S(rec["floor"]))
        {
            return false;
        }
        return Rule(V2Classes.ShopRecordedClickWasNoop);
    }

    /// <summary>
    /// shop_ui records: open_inventory (merchant button) before the first shop record, close_inventory (back button) and
    /// proceed (room button) after "leave". UI navigation; the button states are checked. Returns an action index only
    /// for a room proceed the collector clicked without a preceding leave (no merchant to open), else -1 (consumed).
    /// </summary>
    private int ShopUi(JsonObject rec, Decision d, RunContext ctx)
    {
        string chosen = S(rec["chosen"]) ?? "";
        JsonArray buttons = rec["buttons"] as JsonArray ?? new JsonArray();
        JsonNode? Button(string role) => buttons.FirstOrDefault(b => S(b!["role"]) == role);
        if (!CheckContext(rec, ctx))
        {
            return -1;
        }
        switch (chosen)
        {
            case "open_inventory":
            case "reopen_inventory":
                if (d.Kind != DecisionKind.Shop)
                {
                    Diverge("decision_kind", true, Where(rec), $"the trace opens the merchant inventory where the host asks a {d.Kind} decision", chosen, Describe(d), I(rec["seq"]));
                    return -1;
                }
                Item(rec, "merchant button", B(Button("merchant")?["clickable"]), true);
                Item(rec, "room proceed button", B(Button("proceed")?["clickable"]), Has(d, Verb.Proceed));
                break;
            case "close_inventory":
                if (!_awaitingShopClose)
                {
                    Diverge("record.inconsistent", true, Where(rec), "close_inventory without a preceding leave", chosen, Describe(d), I(rec["seq"]));
                    return -1;
                }
                Item(rec, "inventory back button", B(Button("back")?["clickable"]), true);
                break;
            case "proceed":
                Item(rec, "room proceed button", B(Button("proceed")?["clickable"]), true);
                if (!_awaitingShopClose)
                {
                    // no leave record: the collector found no merchant to open and clicked the room's proceed
                    if (d.Kind != DecisionKind.Shop)
                    {
                        Diverge("decision_kind", true, Where(rec), $"the trace clicks the shop room's proceed where the host asks a {d.Kind} decision", chosen, Describe(d), I(rec["seq"]));
                        return -1;
                    }
                    return Chosen(rec, d, Find(d, Verb.Proceed), "shop room proceed");
                }
                _awaitingShopClose = false;
                break;
            default:
                Diverge("record.unsupported_kind", true, Where(rec), "unknown shop_ui action", chosen, null, I(rec["seq"]));
                return -1;
        }
        Explain(V2Classes.ShopUiNavigation);   // a mapping, not a tolerance: counted wherever it is used (--no-explained does not switch it)
        Consume(rec, compared: true);
        return -1;
    }

    // =====================================================================================================
    //  TREASURE (NTreasureRoom._chestButton, _relicCollection.HoldersInUse, ProceedButton)
    // =====================================================================================================

    private int Treasure(JsonObject rec, Decision d, RunContext ctx)
    {
        if (!Expect(d, rec, DecisionKind.Treasure))
        {
            return -1;
        }
        if (d.Context is not TreasureContext tc)
        {
            Diverge("decision_context", true, Where(rec), "TREASURE decision without a TreasureContext", null, Describe(d), I(rec["seq"]));
            return -1;
        }
        Belt(rec, d, ctx);
        string phase = S(rec["phase"]) ?? "";
        JsonArray buttons = rec["buttons"] as JsonArray ?? new JsonArray();
        JsonNode? Button(string role) => buttons.FirstOrDefault(b => S(b!["role"]) == role);
        switch (phase)
        {
            case "chest":
                Content(rec, "page", "chest", d.SubKind == (int)TreasurePhase.Chest ? "chest" : "relic");
                Item(rec, "chest", B(Button("chest")?["clickable"]), Has(d, Verb.OpenChest));
                Item(rec, "proceed", B(Button("proceed")?["clickable"]), Has(d, Verb.Proceed));
                Item(rec, "skip", false, Has(d, Verb.Skip));
                return Chosen(rec, d, Find(d, Verb.OpenChest), "open the chest");
            case "relic":
            {
                Content(rec, "page", "relic", d.SubKind == (int)TreasurePhase.Relic && !tc.Finished ? "relic" : d.SubKind == (int)TreasurePhase.Chest ? "chest" : "proceed");
                JsonArray offer = rec["offer"]!.AsArray();
                Content(rec, "relics", offer.Count.ToString(), tc.Relics.Count.ToString());
                HashSet<int> shown = new HashSet<int>();
                foreach (JsonNode? o in offer)
                {
                    int ri = I(o!["index"]);
                    shown.Add(ri);
                    if (ri >= 0 && ri < tc.Relics.Count)
                    {
                        Content(rec, "relic", S(o["relic"]), tc.Relics[ri].Id.Entry);
                    }
                    Item(rec, "relic " + S(o["relic"]), B(o["clickable"]), Has(d, Verb.ChooseOption, RefKind.TreasureRelic, ri));
                }
                foreach (LegalAction a in d.Actions)
                {
                    if (a.Verb == Verb.ChooseOption && !shown.Contains(a.Source.Index))
                    {
                        Item(rec, "relic #" + a.Source.Index, false, true);
                    }
                }
                JsonNode? skip = rec["skip"];
                Item(rec, "skip", B(skip?["clickable"]) && B(skip?["is_skip"]), Has(d, Verb.Skip));
                Item(rec, "proceed", B(skip?["clickable"]) && !B(skip?["is_skip"]), Has(d, Verb.Proceed));
                Item(rec, "chest", false, Has(d, Verb.OpenChest));
                if (rec["chosen"] is JsonObject take)
                {
                    return Chosen(rec, d, Find(d, Verb.ChooseOption, RefKind.TreasureRelic, I(take["index"])), "take relic " + S(take["relic"]));
                }
                return Chosen(rec, d, Find(d, Verb.Skip), "skip the relic");
            }
            case "proceed":
                Content(rec, "page", "proceed", tc.Finished ? "proceed" : d.SubKind == (int)TreasurePhase.Chest ? "chest" : "relic");
                Item(rec, "proceed", B(Button("proceed")?["clickable"]), Has(d, Verb.Proceed));
                Item(rec, "chest", false, Has(d, Verb.OpenChest));
                Item(rec, "skip", false, Has(d, Verb.Skip));
                Item(rec, "relic", false, d.Actions.Any(a => a.Verb == Verb.ChooseOption));
                return Chosen(rec, d, Find(d, Verb.Proceed), "treasure proceed");
            default:
                Diverge("record.unsupported_kind", true, Where(rec), "unknown treasure phase", phase, null, I(rec["seq"]));
                return -1;
        }
    }

    // =====================================================================================================
    //  BUNDLE (NChooseABundleSelectionScreen)
    // =====================================================================================================

    private int Bundle(JsonObject rec, Decision d, RunContext ctx)
    {
        if (!Expect(d, rec, DecisionKind.Bundle))
        {
            return -1;
        }
        if (d.Context is not BundleContext bc)
        {
            Diverge("decision_context", true, Where(rec), "BUNDLE decision without a BundleContext", null, Describe(d), I(rec["seq"]));
            return -1;
        }
        JsonArray bundles = rec["bundles"]!.AsArray();
        Content(rec, "bundles", bundles.Count.ToString(), bc.Bundles.Count.ToString());
        for (int i = 0; i < bundles.Count; i++)
        {
            if (i < bc.Bundles.Count)
            {
                Content(rec, "bundle.cards", string.Join(",", bundles[i]!.AsArray().Select(c => Face(c))), string.Join(",", bc.Bundles[i].Select(c => Face(c))));
            }
            Item(rec, "bundle " + i, true, Has(d, Verb.ChooseOption, RefKind.Bundle, i));
        }
        for (int i = bundles.Count; i < bc.Bundles.Count; i++)
        {
            Item(rec, "bundle " + i, false, Has(d, Verb.ChooseOption, RefKind.Bundle, i));
        }
        JsonNode? skip = (rec["buttons_before"] as JsonArray)?.FirstOrDefault(b => S(b!["role"]) == "skip");
        Item(rec, "skip", B(skip?["clickable"]), Has(d, Verb.Skip));
        if (S(rec["chosen"]) == "skip")
        {
            return Chosen(rec, d, Find(d, Verb.Skip), "skip the bundles");
        }
        Explain(V2Classes.BundlePreviewConfirm);   // a mapping, not a tolerance (see ShopUiNavigation)
        return Chosen(rec, d, Find(d, Verb.ChooseOption, RefKind.Bundle, I(rec["chosen"])), "bundle " + I(rec["chosen"]));
    }

    // =====================================================================================================
    //  CONFIRM (NAbandonRunConfirmPopup: Trial double-down)
    // =====================================================================================================

    private int Confirm(JsonObject rec, Decision d, RunContext ctx)
    {
        if (!Expect(d, rec, DecisionKind.Confirm))
        {
            return -1;
        }
        JsonArray buttons = rec["buttons"] as JsonArray ?? new JsonArray();
        JsonNode? Button(string role) => buttons.FirstOrDefault(b => S(b!["role"]) == role);
        Item(rec, "yes", B(Button("yes")?["clickable"]), Has(d, Verb.ChooseOption, RefKind.ConfirmChoice, 1));
        Item(rec, "no", B(Button("no")?["clickable"]), Has(d, Verb.ChooseOption, RefKind.ConfirmChoice, 0));
        if (d.Context is ConfirmContext cc)
        {
            Content(rec, "key", "NAbandonRunConfirmPopup", cc.Key == ConfirmKeys.AbandonRun ? "NAbandonRunConfirmPopup" : cc.Key);
        }
        bool yes = S(rec["chosen"]) == "yes";
        return Chosen(rec, d, Find(d, Verb.ChooseOption, RefKind.ConfirmChoice, yes ? 1 : 0), yes ? "yes" : "no");
    }

    // =====================================================================================================
    //  CRYSTAL SPHERE (NCrystalSphereScreen)
    // =====================================================================================================

    private int Crystal(JsonObject rec, Decision d, RunContext ctx)
    {
        if (S(rec["phase"]) == "proceed")
        {
            // the screen's Proceed (NCrystalSphereScreen.OnProceedButtonPressed = RunManager.Proceed(), :328-331): the finished event's PROCEED in the host
            if (!Expect(d, rec, DecisionKind.Event))
            {
                return -1;
            }
            bool finished = d.Context is EventContext ec && ec.Finished && ec.EventId == "CRYSTAL_SPHERE";
            if (!finished || !Rule(V2Classes.CrystalProceedIsEventProceed))
            {
                Diverge("decision_kind", true, Where(rec), "the trace clicks the crystal screen's Proceed where the host is not at the finished CRYSTAL_SPHERE event", Js(rec["chosen"]), Describe(d), I(rec["seq"]));
                return -1;
            }
            JsonNode? pb = (rec["buttons"] as JsonArray)?.FirstOrDefault(b => S(b!["role"]) == "proceed");
            Item(rec, "proceed", B(pb?["clickable"]), Has(d, Verb.Proceed));
            if (_crystalRevealed != null && rec["revealed_all"] is JsonArray all)
            {
                Content(rec, "revealed_all", string.Join(";", all.Select(x => S(x))), string.Join(";", _crystalRevealed));
            }
            _crystalRevealed = null;
            _crystalMinigame = null;
            return Chosen(rec, d, Find(d, Verb.Proceed), "crystal proceed");
        }
        if (!Expect(d, rec, DecisionKind.Crystal))
        {
            return -1;
        }
        if (d.Context is not CrystalContext cc)
        {
            Diverge("decision_context", true, Where(rec), "CRYSTAL decision without a CrystalContext", null, Describe(d), I(rec["seq"]));
            return -1;
        }
        if (!ReferenceEquals(_crystalMinigame, cc.Minigame))
        {
            // HOST-ONLY member, read by this verification tool only: the collector subscribes to every item's Revealed event (RunTracer2.StepCrystal)
            _crystalMinigame = cc.Minigame;
            List<string> revealed = new List<string>();
            _crystalRevealed = revealed;
            foreach (CrystalSphereItem it in cc.Minigame.Items)
            {
                CrystalSphereItem item = it;
                item.Revealed += _ => revealed.Add($"{item.GetType().Name}@{item.Position.X},{item.Position.Y} size {item.Size.X}x{item.Size.Y} good={item.IsGood}");
            }
        }
        Content(rec, "grid", S(rec["grid"]), cc.Width + "x" + cc.Height);
        Content(rec, "divinations_left", S(rec["divinations_left"]), cc.Divinations.ToString());
        Content(rec, "tool_before", S(rec["tool_before"]), cc.Tool.ToString());
        JsonArray tools = rec["tools"] as JsonArray ?? new JsonArray();
        bool Tool(string role, CrystalSphereMinigame.CrystalSphereToolType t) => cc.Tool == t || B(tools.FirstOrDefault(b => S(b!["role"]) == role)?["clickable"]);
        bool big = Tool("big", CrystalSphereMinigame.CrystalSphereToolType.Big);
        bool small = Tool("small", CrystalSphereMinigame.CrystalSphereToolType.Small);
        Explain(V2Classes.CrystalToolFoldedIntoClick);   // a mapping, not a tolerance (see ShopUiNavigation)
        HashSet<int> ui = new HashSet<int>();
        foreach (JsonNode? cell in rec["clickable_cells"]!.AsArray())
        {
            string[] xy = S(cell)!.Split(',');
            int ordinal = int.Parse(xy[1]) * cc.Width + int.Parse(xy[0]);
            ui.Add(ordinal);
            Item(rec, "cell " + S(cell) + " small", small, Find(d, Verb.CrystalClick, RefKind.CrystalCell, ordinal, (int)CrystalSphereMinigame.CrystalSphereToolType.Small) >= 0);
            Item(rec, "cell " + S(cell) + " big", big, Find(d, Verb.CrystalClick, RefKind.CrystalCell, ordinal, (int)CrystalSphereMinigame.CrystalSphereToolType.Big) >= 0);
        }
        foreach (LegalAction a in d.Actions)
        {
            if (a.Verb == Verb.CrystalClick && !ui.Contains(a.Source.Index))
            {
                Item(rec, $"cell {a.Source.Index % cc.Width},{a.Source.Index / cc.Width}", false, true);
            }
        }
        JsonNode chosen = rec["chosen"]!;
        if (!Enum.TryParse(S(chosen["tool"]), out CrystalSphereMinigame.CrystalSphereToolType tool))
        {
            Diverge("record.inconsistent", true, Where(rec), "unknown crystal tool", Js(chosen), null, I(rec["seq"]));
            return -1;
        }
        _crystalPending = rec;
        _crystalHiddenBefore = cc.Cells.Count(c => c.Hidden);
        _crystalRevealedBefore = _crystalRevealed!.Count;
        return Chosen(rec, d, Find(d, Verb.CrystalClick, RefKind.CrystalCell, I(chosen["y"]) * cc.Width + I(chosen["x"]), (int)tool), $"crystal click {S(chosen["tool"])} {I(chosen["x"])},{I(chosen["y"])}");
    }

    /// <summary>crystal_result (the "after" block of the previous crystal record) against the minigame after the host applied the click.</summary>
    private void ResolveCrystalResult(Decision d, RunContext ctx)
    {
        JsonObject? rec = _crystalPending;
        if (rec == null)
        {
            return;
        }
        _crystalPending = null;
        if (rec["_result"]?["after"] is not JsonObject after || _crystalMinigame == null || _crystalRevealed == null)
        {
            return;
        }
        _recordDiverged = false;
        CrystalSphereMinigame m = _crystalMinigame;
        int hidden = 0;
        foreach (CrystalSphereCell c in m.cells)
        {
            if (c.IsHidden)
            {
                hidden++;
            }
        }
        Content(rec, "after.divinations_left", S(after["divinations_left"]), m.DivinationCount.ToString());
        Content(rec, "after.cells_uncovered", S(after["cells_uncovered"]), (_crystalHiddenBefore - hidden).ToString());
        Content(rec, "after.revealed", string.Join(";", (after["revealed"] as JsonArray ?? new JsonArray()).Select(x => S(x))), string.Join(";", _crystalRevealed.Skip(_crystalRevealedBefore)));
    }
}
