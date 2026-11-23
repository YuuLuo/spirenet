using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Schema;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs.Tests;

/// <summary>
/// The encoder's own consistency checks as the driver's test seam (SpireHost.Driver.Tools.FrameAudit): plugged in by
/// the executable for self-test commands, so that the test probes of the combat / rooms / events modules can audit
/// every decision they see with the real encoder without referencing this project.
/// </summary>
public static class ObsFrameAudit
{
    private static ObsEncoder? _enc;

    public static void Install()
    {
        FrameAudit.Check = Check;
    }

    private static string? Check(Decision decision)
    {
        _enc ??= ObsRuntime.NewEncoder(ObsModule.Current);
        ObsEncoder enc = _enc;
        enc.Options = new EncoderOptions { MapInCombat = true, DeckInCombat = true, UpgradePreviews = true, TargetPreviews = true };
        EncoderStats s = enc.Stats;
        long choice = s.ChoiceOrderViolations, action = s.ActionOrderViolations, unmapped = s.UnmappedActionRefs, overflows = s.Overflows;
        VocabSet vocab = ObsModule.Current.Vocab;
        vocab.UnknownLog.Clear();
        enc.EncodeDecision(decision, 0, (int)decision.Episode.Counters.PolicyDecisions, 0);
        byte[] b = enc.Payload.ToArray();
        List<string> problems = FrameCheck.Problems(b, out _);
        // A name the vocabulary does not know encodes as UNK: a finding - except the game's Mock* test models, which the
        // vocabulary excludes on purpose and which only a test loadout can put into a run.
        foreach (string unknown in vocab.UnknownLog)
        {
            string name = unknown.Substring(unknown.IndexOf('/') + 1);
            if (!name.StartsWith("MOCK_", StringComparison.Ordinal))
            {
                problems.Add("vocabulary UNK: " + unknown);
            }
        }
        vocab.UnknownLog.Clear();
        long dChoice = s.ChoiceOrderViolations - choice, dAction = s.ActionOrderViolations - action, dUnmapped = s.UnmappedActionRefs - unmapped, dOverflows = s.Overflows - overflows;
        if (problems.Count == 0 && dChoice + dAction + dUnmapped + dOverflows == 0 && ScriptTransport.Word(b, Hd.NActions) == decision.Actions.Count)
        {
            return null;
        }
        return $"obs encoder on a {decision.Kind} frame: choice-order violations {dChoice}, action-order violations {dAction}, unmapped action refs {dUnmapped}, " +
               $"overflows {dOverflows}, ACTIONS rows {ScriptTransport.Word(b, Hd.NActions)} for {decision.Actions.Count} actions" +
               (problems.Count > 0 ? ", frame check: " + string.Join("; ", problems.Take(3)) : "");
    }
}

/// <summary>Self-test cases that exist because of the integration (they need the real handlers of the other modules).</summary>
public static class IntegrationCases
{
    /// <summary>
    /// O7: every ConfirmContext.Key that a module can put into a CONFIRM decision (the driver's ConfirmKeys list) and the
    /// two button names are vocabulary names of category confirm_key - none encodes as UNK; the two fixed selection
    /// headers of contract C2 are prompt keys and exist in the game's localisation.
    /// </summary>
    public static object ConfirmAndPromptKeys(ObsRuntime.Loaded l, out bool ok)
    {
        List<string> missing = new List<string>();
        Dictionary<string, int> ids = new Dictionary<string, int>();
        VocabSet.Category confirm = l.Vocab["confirm_key"];
        foreach (string key in ConfirmKeys.All.Concat(new[] { ConfirmKeys.No, ConfirmKeys.Yes }))
        {
            if (confirm.Index.TryGetValue(key, out int id) && id > VocabSet.UnkId)
            {
                ids[key] = id;
            }
            else
            {
                missing.Add("confirm_key:" + key);
            }
        }
        VocabSet.Category prompt = l.Vocab["prompt_key"];
        List<string> notInGame = new List<string>();
        foreach (string key in VocabExport.FixedPromptKeys)
        {
            if (prompt.Index.TryGetValue(key, out int id) && id > VocabSet.UnkId)
            {
                ids[key] = id;
            }
            else
            {
                missing.Add("prompt_key:" + key);
            }
            int slash = key.IndexOf('/');
            if (!LocText.Exists(key.Substring(0, slash), key.Substring(slash + 1)))
            {
                notInGame.Add(key);
            }
        }
        ok = missing.Count == 0 && notInGame.Count == 0 && ConfirmKeys.All.Length > 0;
        return new { confirm_keys_of_the_modules = ConfirmKeys.All, vocab_ids = ids, missing_from_vocabulary = missing, fixed_headers_not_in_the_game_localisation = notInGame };
    }

    private sealed class Watch : IEpisodeObserver
    {
        public Action<RunContext, Decision>? On;

        public void OnEpisodeStart(RunContext ctx)
        {
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto) => On?.Invoke(ctx, decision);

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }
    }

    /// <summary>
    /// D8 / DECISIONS-H1 F.6 on the real combat handler: KnowledgeDemon raises a card choice INSIDE the enemy turn. In the
    /// frame of that prompt, an enemy that has begun its move (RunContext.IntentPerformed) has no INTENTS rows - the real UI
    /// hid its intent right before the move - and every other living enemy with a move keeps them.
    /// </summary>
    public static object EnemyTurnPrompt(HostSession session, ObsRuntime.Loaded l, out bool ok)
    {
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = new EncoderOptions { MapInCombat = false, DeckInCombat = false, UpgradePreviews = true, TargetPreviews = true };
        int prompts = 0, hiddenRows = 0, visibleRows = 0, wrong = 0, performedEnemies = 0, promptKeyIds = 0;
        int playerTurnFrames = 0, playerTurnIntentsVisible = 0;
        List<string> examples = new List<string>();
        List<string> episodes = new List<string>();
        Watch watch = new Watch
        {
            On = (ctx, d) =>
            {
                if (d.Kind == DecisionKind.Combat && ctx.CombatManager.DebugOnlyGetState() is CombatState ps)
                {
                    // positive control: at a player-turn decision of the same fight no enemy has begun a move, and every
                    // living enemy with a move shows its intents
                    enc.EncodeDecision(d, 1, 0, 0);
                    FrameCheck.Problems(enc.Payload.ToArray(), out FrameCheck.View pv);
                    IReadOnlyList<Creature> all = ps.Creatures;
                    for (int row = 0; row < all.Count && row < pv.Rows[T.Creatures]; row++)
                    {
                        Creature e = all[row];
                        if (!e.IsEnemy || !e.IsAlive || e.Monster == null || e.Monster.NextMove.Intents == null || e.Monster.NextMove.Intents.Count == 0)
                        {
                            continue;
                        }
                        playerTurnFrames++;
                        if (ctx.IntentPerformed(e))
                        {
                            wrong++;
                            examples.Add($"{e.Monster.Id.Entry} is marked as having begun its move at a player-turn decision");
                        }
                        else if (pv.Get(T.Creatures, row, CreatureC.IntentCnt) > 0)
                        {
                            playerTurnIntentsVisible++;
                        }
                        else
                        {
                            wrong++;
                            examples.Add($"{e.Monster.Id.Entry} shows no intent rows at a player-turn decision");
                        }
                    }
                    return;
                }
                if (d.Kind != DecisionKind.CardSelect || d.Context is not CardSelectContext c || !c.DuringEnemyTurn)
                {
                    return;
                }
                CombatState? cs = ctx.CombatManager.DebugOnlyGetState();
                if (cs == null)
                {
                    return;
                }
                prompts++;
                enc.EncodeDecision(d, 1, 0, 0);
                byte[] b = enc.Payload.ToArray();
                foreach (string p in FrameCheck.Problems(b, out FrameCheck.View v))
                {
                    wrong++;
                    examples.Add("frame: " + p);
                }
                FrameCheck.Problems(b, out FrameCheck.View view);
                if (ScriptTransport.GlobalOf(b, Gl.PromptKey) > VocabSet.UnkId)
                {
                    promptKeyIds++;
                }
                IReadOnlyList<Creature> creatures = cs.Creatures;
                for (int row = 0; row < creatures.Count && row < view.Rows[T.Creatures]; row++)
                {
                    Creature e = creatures[row];
                    if (!e.IsEnemy || !e.IsAlive || e.Monster == null)
                    {
                        continue;
                    }
                    int cnt = view.Get(T.Creatures, row, CreatureC.IntentCnt);
                    bool performed = ctx.IntentPerformed(e);
                    bool hasIntents = e.Monster.NextMove.Intents != null && e.Monster.NextMove.Intents.Count > 0;
                    if (performed)
                    {
                        performedEnemies++;
                        if (cnt != 0)
                        {
                            wrong++;
                            examples.Add($"{e.Monster.Id.Entry} began its move but the frame shows {cnt} intent rows");
                        }
                        else
                        {
                            hiddenRows++;
                        }
                    }
                    else if (hasIntents)
                    {
                        if (cnt == 0)
                        {
                            wrong++;
                            examples.Add($"{e.Monster.Id.Entry} has not acted but the frame shows no intent rows");
                        }
                        else
                        {
                            visibleRows++;
                        }
                    }
                }
            },
        };
        for (int k = 0; k < 4 && prompts < 3; k++)
        {
            EpisodeResult r = ScenarioRunner.Run(session, "obs-enemy-turn-prompt-" + k, ctx => ScenarioTools.JumpToEncounter(ctx, "KNOWLEDGE_DEMON_BOSS"),
                character: DriverCommands.AllCharacters[k % 6], policy: new EndTurnFirst((ulong)(40 + k)), observer: watch,
                options: new EpisodeOptions { DebugForcedHp = 99999, MaxPolicyDecisions = 600, MaxCombatTurns = 14 });
            episodes.Add(r.Outcome + "/" + r.TruncatedBy + (r.Abort == null ? "" : " " + r.Abort.Signature));
        }
        ok = prompts > 0 && performedEnemies > 0 && wrong == 0 && promptKeyIds == prompts && playerTurnIntentsVisible > 0 && playerTurnIntentsVisible == playerTurnFrames;
        return new
        {
            enemy_turn_prompts = prompts, enemies_that_had_begun_their_move = performedEnemies, their_intents_hidden = hiddenRows, other_enemies_with_visible_intents = visibleRows,
            control_player_turn_enemy_rows = playerTurnFrames, control_player_turn_intents_visible = playerTurnIntentsVisible,
            prompts_with_a_prompt_key = promptKeyIds, wrong, examples, episodes,
            note = "No real-client ground truth exists for the hidden intents (the oracle records no UI state): this checks the host against DECISIONS-H1 F.6, not against the client.",
        };
    }

    /// <summary>COMBAT: end the turn (the fight goes on); everything else random.</summary>
    private sealed class EndTurnFirst : IPolicy
    {
        private readonly RandomLegalPolicy _random;

        public EndTurnFirst(ulong seed)
        {
            _random = new RandomLegalPolicy(seed);
        }

        public int Decide(Decision d)
        {
            if (d.Kind == DecisionKind.Combat)
            {
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    if (d.Actions[i].Verb == Verb.EndTurn)
                    {
                        return i;
                    }
                }
            }
            return _random.Decide(d);
        }
    }

    /// <summary>
    /// The REAL special frames of the events module (not synthetic ones): CRYSTAL (Crystal Sphere), BUNDLE (Scroll Boxes)
    /// and CONFIRM (Trial double down, with and without AllowAbandon) are reached through the real handlers and every
    /// decision of those episodes passes the full encode checker (FrameCheck, side effects, encode twice, exact game
    /// calls). Also checks what only the integration can: on every CRYSTAL frame the item columns of revealed cells
    /// are filled from the public view and hidden cells carry none; merged CARD_SELECT classes report sel_count.
    /// </summary>
    public static object RealSpecialFrames(HostSession session, ObsRuntime.Loaded l, out bool ok)
    {
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = new EncoderOptions { MapInCombat = true, DeckInCombat = true, UpgradePreviews = true, TargetPreviews = true };
        EncodeChecker chk = new EncodeChecker(enc) { Reference = ObsRuntime.NewEncoder(l) };
        ObsEncoder probe = ObsRuntime.NewEncoder(l);
        probe.Options = enc.Options.Clone();
        SortedDictionary<string, int> seen = new SortedDictionary<string, int>(StringComparer.Ordinal);
        List<string> fails = new List<string>();
        int crystalRevealedWithItem = 0, crystalRevealed = 0, confirmBoth = 0, confirmSingle = 0;
        chk.Also = (ctx, d) =>
        {
            string k = d.Kind.ToString();
            seen[k] = seen.GetValueOrDefault(k) + 1;
            if (d.Kind == DecisionKind.Crystal || d.Kind == DecisionKind.Confirm)
            {
                probe.EncodeDecision(d, 1, 0, 0);
                byte[] b = probe.Payload.ToArray();
                FrameCheck.Problems(b, out FrameCheck.View v);
                if (d.Kind == DecisionKind.Confirm)
                {
                    if (d.Actions.Count == 2)
                    {
                        confirmBoth++;
                    }
                    else
                    {
                        confirmSingle++;
                    }
                    if (ScriptTransport.GlobalOf(b, Gl.ConfirmKey) <= VocabSet.UnkId)
                    {
                        fails.Add("CONFIRM: confirm_key is PAD / UNK");
                    }
                    return;
                }
                for (int r = 0; r < v.Rows[T.Cells]; r++)
                {
                    bool hidden = v.Get(T.Cells, r, CellC.Hidden) == 1;
                    int type = v.Get(T.Cells, r, CellC.ItemType);
                    if (hidden && (type != -1 || v.Get(T.Cells, r, CellC.ItemGood) != -1 || v.Get(T.Cells, r, CellC.ItemIdx) != -1))
                    {
                        fails.Add("CRYSTAL: a hidden cell carries item columns");
                    }
                    if (!hidden)
                    {
                        crystalRevealed++;
                        if (type == VocabSet.UnkId)
                        {
                            fails.Add("CRYSTAL: a revealed cell has item_type UNK");
                        }
                        if (type > VocabSet.UnkId)
                        {
                            crystalRevealedWithItem++;
                        }
                    }
                }
            }
        };
        List<object> runs = new List<object>();
        (string name, bool abandon, IPolicy policy, Func<RunContext, Task> setup)[] scenarios =
        {
            ("event CRYSTAL_SPHERE", false, new SuffixPolicy(11, "UNCOVER_FUTURE", "PAYMENT_PLAN"), async ctx => { ScenarioTools.SetGold(ctx, 500); await ScenarioTools.JumpToEvent(ctx, "CRYSTAL_SPHERE"); }),
            ("event CRYSTAL_SPHERE (2)", false, new SuffixPolicy(12, "PAYMENT_PLAN", "UNCOVER_FUTURE"), async ctx => { ScenarioTools.SetGold(ctx, 500); await ScenarioTools.JumpToEvent(ctx, "CRYSTAL_SPHERE"); }),
            ("relic SCROLL_BOXES", false, new RandomLegalPolicy(13), ctx => ScenarioTools.GiveRelic(ctx, "SCROLL_BOXES")),
            ("event TRIAL double down", false, new SuffixPolicy(14, "options.REJECT", "options.DOUBLE_DOWN", "options.ACCEPT"), ctx => ScenarioTools.JumpToEvent(ctx, "TRIAL")),
            ("event TRIAL double down (AllowAbandon)", true, new SuffixPolicy(15, "options.REJECT", "options.DOUBLE_DOWN", "options.ACCEPT"), ctx => ScenarioTools.JumpToEvent(ctx, "TRIAL")),
        };
        foreach (var sc in scenarios)
        {
            EpisodeResult r = ScenarioRunner.Run(session, "obs-real-" + sc.name, sc.setup, character: "IRONCLAD", policy: sc.policy, observer: chk,
                options: new EpisodeOptions { MaxPolicyDecisions = 300, MaxCombatTurns = 20, AllowAbandon = sc.abandon, DebugForcedHp = 999 });
            runs.Add(new { scenario = sc.name, outcome = r.Outcome.ToString(), r.TruncatedBy, abort = r.Abort?.Signature });
            if (r.Outcome == EpisodeOutcome.Aborted)
            {
                fails.Add(sc.name + " aborted: " + r.Abort?.Signature);
            }
        }
        foreach (string kind in new[] { "Crystal", "Bundle", "Confirm" })
        {
            if (seen.GetValueOrDefault(kind) == 0)
            {
                fails.Add("no real " + kind.ToUpperInvariant() + " frame was reached");
            }
        }
        if (confirmBoth == 0 || confirmSingle == 0)
        {
            fails.Add($"CONFIRM: {confirmBoth} frames with both buttons (AllowAbandon), {confirmSingle} with NO only");
        }
        if (crystalRevealedWithItem == 0)
        {
            fails.Add("CRYSTAL: no revealed cell with an item was seen");
        }
        ok = fails.Count == 0 && chk.Problems.Count == 0 && enc.Stats.ActionOrderViolations == 0 && enc.Stats.UnmappedActionRefs == 0 && enc.Stats.ChoiceOrderViolations == 0;
        return new
        {
            failures = fails, frames_by_kind = seen, frames_checked = chk.Frames, problems = chk.Problems, examples = chk.Examples,
            crystal_revealed_cells = crystalRevealed, crystal_revealed_cells_with_item = crystalRevealedWithItem,
            confirm_frames_with_both_buttons = confirmBoth, confirm_frames_no_only = confirmSingle, runs, stats = ObsCommands.Stats(enc.Stats, l.Vocab),
        };
    }

    /// <summary>Prefers event options by TextKey suffix (in order, each suffix at most twice per policy); everything else uniform random.</summary>
    private sealed class SuffixPolicy : IPolicy
    {
        private readonly string[] _suffixes;
        private readonly int[] _used;
        private readonly RandomLegalPolicy _random;

        public SuffixPolicy(ulong seed, params string[] suffixes)
        {
            _suffixes = suffixes;
            _used = new int[suffixes.Length];
            _random = new RandomLegalPolicy(seed);
        }

        public int Decide(Decision d)
        {
            if (d.Kind == DecisionKind.Event)
            {
                for (int s = 0; s < _suffixes.Length; s++)
                {
                    if (_used[s] >= 2)
                    {
                        continue;
                    }
                    for (int i = 0; i < d.Actions.Count; i++)
                    {
                        LegalAction a = d.Actions[i];
                        if (a.Source.Kind == RefKind.EventOption && a.Source.EventOption.TextKey.EndsWith(_suffixes[s], StringComparison.Ordinal))
                        {
                            _used[s]++;
                            return i;
                        }
                    }
                }
            }
            return _random.Decide(d);
        }
    }

    /// <summary>
    /// sel_count of MERGED classes on the real combat-module frames (contract C2). Deterministic part: the game's own
    /// CardSelectCmd.FromDeckForRemoval(player, "remove 2", manual confirmation) on a starting deck (several identical
    /// Strikes) with a policy that picks the same class twice - the frame after the second pick must show sel_count 2 on
    /// that class. Plus three whole runs in which every merged CARD_SELECT frame's sel_count column must equal the
    /// frame's own per-option pick counts.
    /// </summary>
    public static object MergedSelCount(HostSession session, ObsRuntime.Loaded l, out bool ok)
    {
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = new EncoderOptions { MapInCombat = false, DeckInCombat = true, UpgradePreviews = true, TargetPreviews = true };
        int mergedFrames = 0, framesWithPicks = 0, maxSelCount = 0, mismatches = 0;
        List<string> examples = new List<string>();
        Watch watch = new Watch
        {
            On = (ctx, d) =>
            {
                if (d.Kind != DecisionKind.CardSelect || d.Context is not CardSelectContext c || !c.Merged)
                {
                    return;
                }
                mergedFrames++;
                enc.EncodeDecision(d, 1, 0, 0);
                byte[] b = enc.Payload.ToArray();
                FrameCheck.Problems(b, out FrameCheck.View v);
                // CHOICE rows are written in option order: the i-th row of zone CHOICE is option i
                int option = 0;
                bool any = false;
                for (int r = 0; r < v.Rows[T.Cards]; r++)
                {
                    if (v.Get(T.Cards, r, CardC.Zone) != Zone.Choice)
                    {
                        continue;
                    }
                    int sel = v.Get(T.Cards, r, CardC.SelCount);
                    int expect = option < c.OptionPicked.Length ? c.OptionPicked[option] : 0;
                    if (sel != expect)
                    {
                        mismatches++;
                        if (examples.Count < 5)
                        {
                            examples.Add($"option {option}: sel_count {sel}, the frame picked {expect}");
                        }
                    }
                    maxSelCount = Math.Max(maxSelCount, sel);
                    any |= sel > 0;
                    option++;
                }
                if (any)
                {
                    framesWithPicks++;
                }
            },
        };
        List<string> episodes = new List<string>();
        int removed = -1;
        EpisodeResult two = ScenarioRunner.Run(session, "obs-selcount-two-of-a-class", async ctx =>
        {
            // The entry point events and relics use (blocking context: no action pauses), through the real selector path.
            IEnumerable<MegaCrit.Sts2.Core.Models.CardModel> gone = await MegaCrit.Sts2.Core.Commands.CardSelectCmd.FromDeckForRemoval(ctx.LocalPlayer,
                new MegaCrit.Sts2.Core.CardSelection.CardSelectorPrefs(MegaCrit.Sts2.Core.CardSelection.CardSelectorPrefs.RemoveSelectionPrompt, 2) { RequireManualConfirmation = true });
            removed = gone.Count();
        }, character: "IRONCLAD", policy: new SameClassAgain(69), observer: watch, options: new EpisodeOptions { MaxPolicyDecisions = 60 });
        episodes.Add("two-of-a-class: " + two.Outcome + "/" + two.TruncatedBy + (two.Abort == null ? "" : " " + two.Abort.Signature) + ", removed " + removed);
        int maxAfterScenario = maxSelCount;
        for (int k = 0; k < 3; k++)
        {
            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel("OBS-SELCOUNT:" + k), Characters = new[] { DriverCommands.AllCharacters[k % 6] } };
            EpisodeResult r = session.RunEpisode(spec, new SameClassAgain((ulong)(70 + k)), new EpisodeOptions { DebugForcedHp = 99999, MaxPolicyDecisions = 1200, MaxCombatTurns = 30 }, watch);
            episodes.Add(r.Outcome + "/" + r.TruncatedBy + (r.Abort == null ? "" : " " + r.Abort.Signature));
        }
        ok = mismatches == 0 && mergedFrames > 0 && maxAfterScenario == 2 && removed == 2 && two.Abort == null;
        return new
        {
            merged_card_select_frames = mergedFrames, frames_with_picks = framesWithPicks, max_sel_count_seen = maxSelCount, sel_count_mismatches = mismatches,
            two_of_a_class_scenario_max_sel_count = maxAfterScenario, two_of_a_class_cards_removed = removed, examples, episodes,
        };
    }

    /// <summary>CARD_SELECT: selects the option it selected last when it is still offered (to pick two instances of one class); else random.</summary>
    private sealed class SameClassAgain : IPolicy
    {
        private readonly RandomLegalPolicy _random;
        private int _last = -1;

        public SameClassAgain(ulong seed)
        {
            _random = new RandomLegalPolicy(seed);
        }

        public int Decide(Decision d)
        {
            if (d.Kind == DecisionKind.CardSelect && d.Context is CardSelectContext c)
            {
                if (c.Picked.Count == 0)
                {
                    _last = -1;
                }
                int best = -1;
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    if (d.Actions[i].Verb != Verb.SelectCard)
                    {
                        continue;
                    }
                    int ordinal = d.Actions[i].Source.Index;
                    if (ordinal == _last)
                    {
                        best = i;
                        break;
                    }
                    // prefer a class with several instances
                    if (best < 0 && c.OptionCounts != null && ordinal < c.OptionCounts.Count && c.OptionCounts[ordinal] > 1)
                    {
                        best = i;
                    }
                }
                if (best >= 0)
                {
                    _last = d.Actions[best].Source.Index;
                    return best;
                }
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    if (d.Actions[i].Verb == Verb.ConfirmSelection)
                    {
                        return i;
                    }
                }
            }
            return _random.Decide(d);
        }
    }
}
