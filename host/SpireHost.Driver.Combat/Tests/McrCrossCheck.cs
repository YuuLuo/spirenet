using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Patches;

namespace SpireHost.Driver.Combat.Tests;

/// <summary>
/// --crosscheck-mcr: the combat handler's legal-action enumeration against the REAL CLIENT, on the replay team's
/// ground truth (notes\host-h1-replay.md).
///
/// The oracle corpus (oracle\replays: 1,488 combats recorded in the real v0.113 client; oracle\runs: 621 map-node
/// combats of 120 golden runs) carries, for every action the real client's driver took, a sidecar record:
/// the public state signature, the number of legal options the collector enumerated in the real client
/// (realgame\mod-src\SpireOracle\CombatDriver.cs: CanPlay x targets, potion use / discard, end turn) and the
/// action itself (kind, card / potion, target), which the real engine then executed (every post-action checksum
/// of these replays is reproduced by the host: replay team, 2,109 / 2,109).
///
/// This command runs the replay tool (SpireHost.Replay.McrCommands.ReplayMcr, the assembly shipped in the same
/// executable, reached by reflection: the combat project has no reference to it) in THIS session - the production
/// patch set (ruling B4) - and looks over its shoulder with two read-only test patches on public methods of that tool:
///   McrReplayer.Run(file, options)             prefix:  which replay is running (its sidecar records)
///   McrReplayer.CountOptions(me, cs, potions)  postfix: the tool evaluates the collector's count right before it
///                                              injects recorded action k; at that moment the engine is in the
///                                              recorded pre-action state, and CombatHandler.Enumerate is run on it.
/// For every recorded decision it compares
///   (a) the handler's readiness: the state must be a decision point for the recorded seat;
///   (b) the handler's action count with the tool's count (same host state) and with the RECORDED count;
///   (c) membership: the action the real client took must be one of the handler's legal actions.
/// A count difference from the recording is classified as "potion lock window" when the recorded count equals
/// the handler's count minus all its potion actions: the accepted difference documented by the replay team (the
/// real client keeps potions locked while the room fades in, RunManager.cs:908-922 / :223-231; the host has no
/// fade). Anything else fails the command.
///
/// SELECTION PROMPTS. The recordings also contain every card-selection prompt the real client answered THROUGH
/// ITS REAL SCREENS (the collector clicks card holders and the confirm / skip buttons; the game then syncs the
/// choice, which the replay stores as a PlayerChoice event) and the collector's log line per prompt (screen
/// type, the prefs' min / max, number of offered cards, skip button). In the replay the game asks for the
/// "remote" choice at the same point (PlayerChoiceSynchronizer.WaitForRemoteChoice, called from inside the
/// CardSelectCmd.From* state machine, so SelectionCapture.Current is the captured prompt of that call). Two
/// more read-only test prefixes (WaitForRemoteChoice, ReceiveReplayChoice) pair every such request with the
/// recorded answer and check
///   (d) the captured context against the real screen: kind vs screen type, prefs min / max, candidate count
///       vs offered cards, canSkip vs the visibility of the skip button;
///   (e) the recorded answer against <see cref="SelectionRules"/>: its size must be one the frame's rules can
///       produce, and every chosen card must be one of the frame's candidates.
/// (The selector itself is not involved: the replay service answers through the remote path. What is checked
/// is the part a selector cannot learn from the game: the captured context and the UI-only rules.)
///
/// It also is the regression required of this module: the replay tool's own verdict (every post-action full
/// combat state checksum equal to the real client's) is produced with the combat module's Harmony patches in
/// place, which proves they do not change combat resolution.
/// Exit 0 = replays pass and no unexplained difference; 5 otherwise. TEST TOOL: nothing here is installed by the module.
/// </summary>
public static class McrCrossCheck
{
    /// <summary>The replay tool's assembly: part of the integrated executable, loaded by name.</summary>
    public const string ReplayAssemblyName = "SpireHost.Replay";

    private static readonly Decision _decision = new Decision();
    private static readonly CombatContext _info = new CombatContext();

    private static JsonArray? _actions;
    private static string _file = "";
    private static int _k;

    private static long _replays;
    private static long _compared;
    private static long _notReady;
    private static long _equalTool;
    private static long _differsTool;
    private static long _recordedAvailable;
    private static long _equalRecorded;
    private static long _potionWindow;
    private static long _differsOther;
    private static long _actionOffered;
    private static long _actionMissing;
    private static long _actionUnparsed;
    private static long _offeredPlays;
    private static long _offeredPotionUses;
    private static long _offeredDiscards;
    private static long _offeredEndTurns;
    private static long _withPotionActions;
    private static readonly Dictionary<string, long> _notReadyBy = new Dictionary<string, long>();
    private static readonly Dictionary<string, long> _recordedKinds = new Dictionary<string, long>();
    private static readonly HashSet<string> _replaysInWindow = new HashSet<string>();
    private static readonly List<string> _examples = new List<string>();
    private static string? _probeError;

    // ---- selection prompts ----
    private sealed class Ask
    {
        public SelectionPrompt Prompt = null!;
        public List<CardModel>? Candidates;
        public SelectionRules Rules;
        public ulong Player;
        public uint ChoiceId;
        public string Where = "";
        public int Result = -1;
    }

    private static readonly List<Ask> _asks = new List<Ask>();
    private static readonly Dictionary<(ulong, uint), NetPlayerChoiceResult> _earlyAnswers = new Dictionary<(ulong, uint), NetPlayerChoiceResult>();
    private static List<string> _promptLog = new List<string>();
    private static bool _hasPromptLog;
    private static long _logReplaysWithoutLog;
    private static long _logPromptsWithoutLog;
    private static long _promptsAsked;
    private static long _promptsWithoutContext;
    private static long _promptsUnsupportedKind;
    private static long _promptsAnswered;
    private static long _answersLegal;
    private static long _answersIllegalCount;
    private static long _answersNotCandidate;
    private static long _answersUndecodable;
    private static long _logCompared;
    private static long _logReplaysAligned;
    private static long _logReplaysMisaligned;
    private static long _logReplaysAlignedAfterMerge;
    private static long _logKindMismatch;
    private static long _logBoundsMismatch;
    private static long _logOfferedEqual;
    private static long _logOfferedLargerWithFilter;
    private static long _logOfferedMismatch;
    private static long _logSkipMismatch;
    private static readonly Dictionary<string, long> _promptsByKind = new Dictionary<string, long>();
    private static readonly Dictionary<string, long> _answersByShape = new Dictionary<string, long>();
    private static readonly Dictionary<string, long> _promptsByContext = new Dictionary<string, long>();

    public static int Run(HostSession session, Args a)
    {
        Assembly replay;
        try
        {
            replay = Assembly.Load(ReplayAssemblyName);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("crosscheck-mcr: the replay assembly " + ReplayAssemblyName + " is not part of this executable: " + e.Message);
            return 1;
        }
        string path = replay.Location;
        Type replayer = replay.GetType("SpireHost.Replay.McrReplayer", throwOnError: true)!;
        Type commands = replay.GetType("SpireHost.Replay.McrCommands", throwOnError: true)!;
        MethodInfo? run = replayer.GetMethods(BindingFlags.Instance | BindingFlags.Public).FirstOrDefault(m => m.Name == "Run" && m.GetParameters().Length == 2);
        MethodInfo? count = replayer.GetMethod("CountOptions", BindingFlags.Static | BindingFlags.Public, null, new[] { typeof(Player), typeof(CombatState), typeof(bool) }, null);
        MethodInfo? replayMcr = commands.GetMethod("ReplayMcr", BindingFlags.Static | BindingFlags.Public, null, new[] { typeof(HostSession), typeof(Args) }, null);
        if (replayMcr == null)
        {
            Console.Error.WriteLine("crosscheck-mcr: SpireHost.Replay.McrCommands.ReplayMcr(HostSession, Args) not found");
            return 1;
        }
        // Fail-closed like every other patch: a missing target throws.
        PatchRegistry.Prefix("combat-test-mcr-replay-begin", run, PatchRegistry.M(typeof(McrCrossCheck), nameof(BeginReplay)), PatchCategory.Instrumentation,
            "TEST TOOL (--crosscheck-mcr only; not installed by the module): read-only prefix on the replay team's SpireHost.Replay.McrReplayer.Run(file, options) that notes which .mcr replay is running and its sidecar action records.");
        PatchRegistry.Postfix("combat-test-mcr-option-count", count, PatchRegistry.M(typeof(McrCrossCheck), nameof(AfterCountOptions)), PatchCategory.Instrumentation,
            "TEST TOOL (--crosscheck-mcr only; not installed by the module): read-only postfix on the replay team's SpireHost.Replay.McrReplayer.CountOptions(me, cs, includePotions), which that tool evaluates right before it injects each recorded player action; the combat handler's enumeration is run on the same state and compared with the tool's count, the recorded count and the recorded action.");

        Type sync = typeof(PlayerChoiceSynchronizer);
        PatchRegistry.DeclareSharedSite("PlayerChoiceSynchronizer.WaitForRemoteChoice: two read-only prefixes. mp-remote-choice-wait (production, phase H5) routes the wait of a remote seat " +
            "inside a multi-seat episode and is inert otherwise; combat-test-mcr-choice-requested (this TEST TOOL, which replays single-player recordings outside an episode) notes the " +
            "captured prompt of the same call. They never act in the same process state.", "mp-remote-choice-wait", "combat-test-mcr-choice-requested");
        PatchRegistry.Prefix("combat-test-mcr-choice-requested", AccessTools.Method(sync, nameof(PlayerChoiceSynchronizer.WaitForRemoteChoice), new[] { typeof(Player), typeof(uint) }),
            PatchRegistry.M(typeof(McrCrossCheck), nameof(ChoiceRequested)), PatchCategory.Instrumentation,
            "TEST TOOL (--crosscheck-mcr only; not installed by the module): read-only prefix on PlayerChoiceSynchronizer.WaitForRemoteChoice (GameActions.Multiplayer/PlayerChoiceSynchronizer.cs:78), which CardSelectCmd calls where the real client opened a selection screen when a recording is replayed; notes the captured prompt context of that call.");
        PatchRegistry.Prefix("combat-test-mcr-choice-answered", AccessTools.Method(sync, nameof(PlayerChoiceSynchronizer.ReceiveReplayChoice), new[] { typeof(Player), typeof(uint), typeof(NetPlayerChoiceResult) }),
            PatchRegistry.M(typeof(McrCrossCheck), nameof(ChoiceAnswered)), PatchCategory.Instrumentation,
            "TEST TOOL (--crosscheck-mcr only; not installed by the module): read-only prefix on PlayerChoiceSynchronizer.ReceiveReplayChoice (:127), through which the replay tool injects the answer the real client's screen produced; the answer is checked against the frame's effective selection rules and candidates.");

        // The replay tool's own arguments (same names). Its JSON is captured in memory (and written next to --out, or to
        // --replay-out, when one is given - never to a temp directory); stdout carries this command's JSON only.
        string? outPath = a.Get("out");
        string? replayOut = a.Get("replay-out") ?? (outPath != null ? Path.ChangeExtension(Path.GetFullPath(outPath), ".replay.json") : null);
        List<string> argv = new List<string> { "--replay-mcr", "--no-records" };
        if (replayOut != null)
        {
            argv.Add("--out");
            argv.Add(replayOut);
        }
        foreach (string key in new[] { "file", "dir", "files", "start", "count", "stride", "offset" })
        {
            if (a.Get(key) is string v)
            {
                argv.Add("--" + key);
                argv.Add(v);
            }
        }
        foreach (string flag in new[] { "run-traces", "loose-timing", "progress" })
        {
            if (a.Has(flag))
            {
                argv.Add("--" + flag);
            }
        }
        TextWriter stdout = HostIo.Out;
        StringWriter captured = new StringWriter();
        HostIo.Out = captured;
        int code;
        try
        {
            code = (int)replayMcr.Invoke(null, new object[] { session, Args.Parse(argv.ToArray()) })!;
        }
        finally
        {
            HostIo.Out = stdout;
        }

        FlushPrompts();
        JsonNode? tool = null;
        try
        {
            tool = JsonNode.Parse(captured.ToString());
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("crosscheck-mcr: the replay tool printed no JSON document: " + e.Message);
        }
        bool ok = code == 0 && _probeError == null && _compared > 0 && _notReady == 0 && _differsTool == 0 && _differsOther == 0 && _actionMissing == 0 && _actionUnparsed == 0
            && _promptsWithoutContext == 0 && _promptsUnsupportedKind == 0 && _answersIllegalCount == 0 && _answersNotCandidate == 0 && _answersUndecodable == 0
            && _logKindMismatch == 0 && _logBoundsMismatch == 0 && _logOfferedMismatch == 0 && _logSkipMismatch == 0 && _logReplaysMisaligned == 0;
        HostJson.Emit(new
        {
            kind = "combat-mcr-crosscheck",
            ok,
            replay_assembly = path,
            corpus = a.Has("run-traces") ? "oracle\\runs (map-node combats of the golden run traces)" : a.Get("dir", "oracle\\replays (real-client combat corpus)"),
            combat_patches_installed = PatchRegistry.Records.Count(r => r.Id.StartsWith("combat-", StringComparison.Ordinal) && !r.Id.StartsWith("combat-test-", StringComparison.Ordinal)),
            replay_tool = new
            {
                exit_code = code,
                replays = tool?["replays"]?.GetValue<int>(),
                pass = tool?["pass"]?.GetValue<int>(),
                by_status = tool?["by_status"],
                checksums_recorded = tool?["checksums_recorded"]?.GetValue<int>(),
                checksums_matched = tool?["checksums_matched"]?.GetValue<int>(),
                checksums_mismatched = tool?["checksums_mismatched"]?.GetValue<int>(),
                events_injected = tool?["events_injected"]?.GetValue<int>(),
                extended = tool?["extended"],
                output = replayOut,
            },
            handler = new
            {
                replays_seen = _replays,
                recorded_decisions_compared = _compared,
                not_a_decision_point_for_the_handler = _notReady,
                not_ready_by_reason = _notReadyBy,
                count_equal_to_replay_tool = _equalTool,
                count_differs_from_replay_tool = _differsTool,
                recorded_count_available = _recordedAvailable,
                count_equal_to_recording = _equalRecorded,
                count_differs_potion_lock_window = _potionWindow,
                replays_with_potion_lock_window = _replaysInWindow.Count,
                count_differs_unexplained = _differsOther,
                recorded_action_is_offered = _actionOffered,
                recorded_action_not_offered = _actionMissing,
                recorded_action_unparsed = _actionUnparsed,
                recorded_action_kinds = _recordedKinds,
                actions_offered = new { play_card = _offeredPlays, use_potion = _offeredPotionUses, discard_potion = _offeredDiscards, end_turn = _offeredEndTurns },
                decisions_with_potion_actions = _withPotionActions,
                probe_error = _probeError,
                examples = _examples,
            },
            selection_prompts = new
            {
                requested_in_replays = _promptsAsked,
                by_kind = _promptsByKind,
                by_choice_context = _promptsByContext,
                without_captured_context = _promptsWithoutContext,
                unsupported_kind = _promptsUnsupportedKind,
                recorded_answers = _promptsAnswered,
                answer_allowed_by_frame_rules = _answersLegal,
                answer_count_not_allowed = _answersIllegalCount,
                answer_card_not_a_candidate = _answersNotCandidate,
                answer_undecodable = _answersUndecodable,
                answers_by_kind_and_shape = _answersByShape,
                collector_log = new
                {
                    replays_aligned = _logReplaysAligned,
                    replays_aligned_after_merging_click_retries = _logReplaysAlignedAfterMerge,
                    replays_not_aligned = _logReplaysMisaligned,
                    replays_without_a_collector_log = _logReplaysWithoutLog,
                    prompts_without_a_collector_log = _logPromptsWithoutLog,
                    prompts_compared = _logCompared,
                    kind_vs_screen_mismatch = _logKindMismatch,
                    prefs_min_max_mismatch = _logBoundsMismatch,
                    offered_equal_to_candidates = _logOfferedEqual,
                    offered_larger_filtered_hand = _logOfferedLargerWithFilter,
                    offered_mismatch = _logOfferedMismatch,
                    skip_button_mismatch = _logSkipMismatch,
                },
            },
        }, outPath);
        return ok ? 0 : 5;
    }

    // -----------------------------------------------------------------------------------------------------
    //  Harmony patch methods (parameter names = the replay tool's parameter names)
    // -----------------------------------------------------------------------------------------------------

    private static void BeginReplay(object file)
    {
        FlushPrompts();
        _k = 0;
        _actions = null;
        _file = "?";
        _replays++;
        try
        {
            Type t = file.GetType();
            _actions = t.GetField("Actions")?.GetValue(file) as JsonArray;
            _file = t.GetProperty("Name")?.GetValue(file) as string ?? "?";
            // The collector's prompt log: corpus sidecar "prompts", run-trace NNN-combat-fN.combat.json "Prompts".
            JsonArray? prompts = (t.GetField("Sidecar")?.GetValue(file) as JsonNode)?["prompts"] as JsonArray;
            if (prompts == null && t.GetField("Path")?.GetValue(file) is string path)
            {
                // The combat.json that belongs to this replay as the replay tool resolved it (McrFile.CombatJsonPath). Verifier
                // finding F2: the fatal fight's replay is stored as NNN-combat-final-loss.mcr while its combat.json keeps the fight's
                // own name, so the lookup by file name alone skipped the prompt log of every final-loss fight.
                string combat = t.GetField("CombatJsonPath")?.GetValue(file) as string ?? Path.ChangeExtension(path, ".combat.json");
                if (File.Exists(combat))
                {
                    prompts = JsonNode.Parse(File.ReadAllText(combat))?["Prompts"] as JsonArray;
                }
            }
            _hasPromptLog = prompts != null;
            if (prompts != null)
            {
                foreach (JsonNode? line in prompts)
                {
                    string text = line?.GetValue<string>() ?? "";
                    if (text.Contains(" hand-select ", StringComparison.Ordinal) || text.Contains(" grid ", StringComparison.Ordinal) || text.Contains(" choose-a-card ", StringComparison.Ordinal))
                    {
                        _promptLog.Add(text);
                    }
                }
            }
        }
        catch (Exception e)
        {
            _probeError ??= "BeginReplay: " + e.Message;
        }
    }

    // -----------------------------------------------------------------------------------------------------
    //  Selection prompts
    // -----------------------------------------------------------------------------------------------------

    /// <summary>Prefix of PlayerChoiceSynchronizer.WaitForRemoteChoice: the game asks for the choice of the From* call the current flow is inside of.</summary>
    private static void ChoiceRequested(Player player, uint choiceId)
    {
        try
        {
            _promptsAsked++;
            SelectionPrompt? p = SelectionCapture.Current;
            string where = $"{_file} choice {choiceId} turn {player.PlayerCombatState?.TurnNumber}";
            if (p == null)
            {
                _promptsWithoutContext++;
                Example(where + ": the game asked for a choice outside a captured CardSelectCmd.From* call");
                return;
            }
            string kind = p.Kind.ToString();
            _promptsByKind[kind] = _promptsByKind.GetValueOrDefault(kind) + 1;
            string context = p.ContextKind.ToString();
            _promptsByContext[context] = _promptsByContext.GetValueOrDefault(context) + 1;
            Ask ask = new Ask { Prompt = p, Player = player.NetId, ChoiceId = choiceId, Where = where + " " + p.Entry + (p.SourceModel != null ? " by " + p.SourceModel.Id.Entry : "") };
            ask.Candidates = Candidates(p);
            if (ask.Candidates == null)
            {
                _promptsUnsupportedKind++;
                Example(ask.Where + ": no candidate reconstruction for kind " + p.Kind);
            }
            else
            {
                ask.Rules = SelectionRules.Compute(p, ask.Candidates.Count);
            }
            _asks.Add(ask);
            if (_earlyAnswers.Remove((ask.Player, choiceId), out NetPlayerChoiceResult early))
            {
                Evaluate(ask, player, early);
            }
        }
        catch (Exception e)
        {
            _probeError ??= $"{_file} ChoiceRequested: {e.GetType().Name}: {e.Message}";
        }
    }

    /// <summary>
    /// The candidates exactly as CardSelectCmd computes them before it asks (CardSelectCmd.cs:64/75, :132-141, :189-198,
    /// :250-261, :554, :622). Only the entry points that can be reached inside a combat replay.
    /// </summary>
    private static List<CardModel>? Candidates(SelectionPrompt p)
    {
        switch (p.Kind)
        {
            case CardSelectKind.ChooseACard:
            case CardSelectKind.SimpleGrid:
                return p.GivenCards?.ToList();
            case CardSelectKind.SimpleGridRewards:
                return p.GivenResults?.Select(r => r.Card).ToList();
            case CardSelectKind.CombatPile:
                return p.Pile == null ? null : p.Filter == null ? p.Pile.Cards.ToList() : p.Pile.Cards.Where(p.Filter).ToList();
            case CardSelectKind.Hand:
            case CardSelectKind.HandDiscard:
            {
                IReadOnlyList<CardModel> hand = PileType.Hand.GetPile(p.Player).Cards;
                return p.Filter == null ? hand.ToList() : hand.Where(p.Filter).ToList();
            }
            case CardSelectKind.HandUpgrade:
                return PileType.Hand.GetPile(p.Player).Cards.Where(c => c.IsUpgradable).ToList();
            default:
                return null;
        }
    }

    /// <summary>Prefix of PlayerChoiceSynchronizer.ReceiveReplayChoice: the answer the real client's screen produced.</summary>
    private static void ChoiceAnswered(Player player, uint choiceId, NetPlayerChoiceResult result)
    {
        try
        {
            for (int i = _asks.Count - 1; i >= 0; i--)
            {
                if (_asks[i].Player == player.NetId && _asks[i].ChoiceId == choiceId && _asks[i].Result < 0)
                {
                    Evaluate(_asks[i], player, result);
                    return;
                }
            }
            _earlyAnswers[(player.NetId, choiceId)] = result; // the answer arrived before the game asked
        }
        catch (Exception e)
        {
            _probeError ??= $"{_file} ChoiceAnswered: {e.GetType().Name}: {e.Message}";
        }
    }

    private static void Evaluate(Ask ask, Player player, NetPlayerChoiceResult net)
    {
        _promptsAnswered++;
        ask.Result = 0;
        if (ask.Candidates == null)
        {
            return;
        }
        SelectionPrompt p = ask.Prompt;
        List<CardModel> chosen = new List<CardModel>();
        bool decodable = true;
        PlayerChoiceResult result = PlayerChoiceResult.FromNetData(player, player.RunState, net);
        switch (p.Kind)
        {
            case CardSelectKind.ChooseACard:
            {
                int index = result.AsIndex();                       // CardSelectCmd.cs:96-97: negative = skipped
                if (index >= ask.Candidates.Count)
                {
                    decodable = false;
                }
                else if (index >= 0)
                {
                    chosen.Add(ask.Candidates[index]);
                }
                break;
            }
            case CardSelectKind.SimpleGrid:
            case CardSelectKind.SimpleGridRewards:
                foreach (int index in result.AsIndexes())            // CardSelectCmd.cs:154-155 / :212-213
                {
                    if ((uint)index >= (uint)ask.Candidates.Count)
                    {
                        decodable = false;
                    }
                    else
                    {
                        chosen.Add(ask.Candidates[index]);
                    }
                }
                break;
            default:
                chosen.AddRange(result.AsCombatCards());             // CardSelectCmd.cs:305, :583, :647
                break;
        }
        if (!decodable)
        {
            _answersUndecodable++;
            Example(ask.Where + ": the recorded answer does not index the candidates");
            return;
        }
        int k = chosen.Count;
        ask.Result = k;
        SelectionRules rules = ask.Rules;
        string shape = p.Kind + (k == 0 ? ": none (skip)" : k == rules.EffMax ? ": upper bound" : k == rules.EffMin ? ": lower bound" : ": inside the range")
            + (rules.NeedConfirm ? " [confirm]" : "");
        _answersByShape[shape] = _answersByShape.GetValueOrDefault(shape) + 1;
        bool members = true;
        for (int i = 0; i < k; i++)
        {
            bool found = false;
            for (int c = 0; c < ask.Candidates.Count; c++)
            {
                found |= ReferenceEquals(ask.Candidates[c], chosen[i]);
            }
            for (int j = 0; j < i; j++)
            {
                found &= !ReferenceEquals(chosen[j], chosen[i]);
            }
            members &= found;
        }
        if (!members)
        {
            _answersNotCandidate++;
            Example($"{ask.Where}: the real client chose [{string.Join(",", chosen.Select(c => c.Id.Entry))}], not all (distinct) candidates of the frame ({ask.Candidates.Count} candidates)");
        }
        else if (!rules.Allows(k))
        {
            _answersIllegalCount++;
            Example($"{ask.Where}: the real screen finished with {k} cards; the frame allows {(rules.ZeroAllowed || rules.Cancelable ? "0 or " : "")}[{rules.EffMin}, {rules.EffMax}] (n {ask.Candidates.Count}, prefs {p.ExpectMin}..{p.ExpectMax}, canSkip {p.CanSkip})");
        }
        else
        {
            _answersLegal++;
        }
    }

    private static readonly Regex _handLine = new Regex(@"hand-select mode=(\w+) min=(-?\d+) max=(-?\d+) offered=(\d+) ", RegexOptions.Compiled);
    private static readonly Regex _gridLine = new Regex(@"grid (\w+) visInTree=\w+ min=(-?\d+) max=(-?\d+) offered=(\d+) ", RegexOptions.Compiled);
    private static readonly Regex _chooseLine = new Regex(@"choose-a-card offered=(\d+) skip=(True|False) ", RegexOptions.Compiled);

    /// <summary>End of a replay: the collector's prompt log lines against the prompts the game asked for, in order.</summary>
    private static void FlushPrompts()
    {
        try
        {
            if (!_hasPromptLog)
            {
                // No collector log next to this recording (the last, lost combat of a run trace is stored as
                // NNN-combat-final-loss.mcr without a .combat.json): the answers are still checked, the log is not.
                if (_asks.Count > 0)
                {
                    _logReplaysWithoutLog++;
                    _logPromptsWithoutLog += _asks.Count;
                }
            }
            else if (_asks.Count > 0 || _promptLog.Count > 0)
            {
                List<string> log = _promptLog;
                if (_asks.Count != log.Count)
                {
                    // The collector logs a line per ATTEMPT: the choose-a-card screen ignores clicks during its first
                    // 350 ms (NChooseACardSelectionScreen.SelectHolder), so one prompt can have two consecutive lines
                    // that differ only in the card the collector tried to click. Merge those and try again.
                    List<string> merged = new List<string>(log.Count);
                    foreach (string line in log)
                    {
                        int at = line.IndexOf(" chose=", StringComparison.Ordinal);
                        string head = at > 0 ? line.Substring(0, at) : line;
                        if (merged.Count > 0 && head.Contains(" choose-a-card ", StringComparison.Ordinal) && merged[merged.Count - 1].StartsWith(head + " chose=", StringComparison.Ordinal))
                        {
                            merged[merged.Count - 1] = line;
                            continue;
                        }
                        merged.Add(line);
                    }
                    if (merged.Count == _asks.Count)
                    {
                        _logReplaysAlignedAfterMerge++;
                        log = merged;
                    }
                }
                if (_asks.Count != log.Count)
                {
                    _logReplaysMisaligned++;
                    Example($"{_file}: {_asks.Count} prompts asked in the replay, {_promptLog.Count} prompt lines in the collector log (not compared)");
                }
                else
                {
                    _logReplaysAligned++;
                    for (int i = 0; i < _asks.Count; i++)
                    {
                        CompareWithLog(_asks[i], log[i]);
                    }
                }
            }
        }
        catch (Exception e)
        {
            _probeError ??= $"{_file} FlushPrompts: {e.GetType().Name}: {e.Message}";
        }
        _asks.Clear();
        _earlyAnswers.Clear();
        _promptLog = new List<string>();
        _hasPromptLog = false;
    }

    private static void CompareWithLog(Ask ask, string line)
    {
        _logCompared++;
        SelectionPrompt p = ask.Prompt;
        int n = ask.Candidates?.Count ?? -1;
        string screen;
        int min;
        int max;
        int offered;
        bool? skip = null;
        Match m;
        if ((m = _handLine.Match(line)).Success)
        {
            screen = "hand:" + m.Groups[1].Value;
            min = int.Parse(m.Groups[2].Value);
            max = int.Parse(m.Groups[3].Value);
            offered = int.Parse(m.Groups[4].Value);
        }
        else if ((m = _gridLine.Match(line)).Success)
        {
            screen = m.Groups[1].Value;
            min = int.Parse(m.Groups[2].Value);
            max = int.Parse(m.Groups[3].Value);
            offered = int.Parse(m.Groups[4].Value);
        }
        else if ((m = _chooseLine.Match(line)).Success)
        {
            screen = "NChooseACardSelectionScreen";
            min = p.ExpectMin;
            max = p.ExpectMax;
            offered = int.Parse(m.Groups[1].Value);
            skip = m.Groups[2].Value == "True";
        }
        else
        {
            _logKindMismatch++;
            Example(ask.Where + ": unparsed collector log line: " + line);
            return;
        }
        string expected = p.Kind switch
        {
            CardSelectKind.Hand or CardSelectKind.HandDiscard => "hand:SimpleSelect",
            CardSelectKind.HandUpgrade => "hand:UpgradeSelect",
            CardSelectKind.CombatPile => "NCombatPileCardSelectScreen",
            CardSelectKind.SimpleGrid or CardSelectKind.SimpleGridRewards => "NSimpleCardSelectScreen",
            CardSelectKind.ChooseACard => "NChooseACardSelectionScreen",
            _ => "?",
        };
        if (screen != expected)
        {
            _logKindMismatch++;
            Example($"{ask.Where}: captured kind {p.Kind}, the real client showed {screen}");
            return;
        }
        if (min != p.ExpectMin || max != p.ExpectMax)
        {
            _logBoundsMismatch++;
            Example($"{ask.Where}: captured prefs {p.ExpectMin}..{p.ExpectMax}, the real screen had {min}..{max}");
        }
        if (offered == n)
        {
            _logOfferedEqual++;
        }
        else if (offered > n && p.HasFilter && (p.Kind == CardSelectKind.Hand || p.Kind == CardSelectKind.HandDiscard))
        {
            // The collector counts every hand holder; the hand screen only hides the cards the filter rejects (NPlayerHand.cs:773-789).
            _logOfferedLargerWithFilter++;
        }
        else
        {
            _logOfferedMismatch++;
            Example($"{ask.Where}: {n} candidates, the real screen offered {offered} cards");
        }
        if (skip.HasValue && skip.Value != p.CanSkip)
        {
            _logSkipMismatch++;
            Example($"{ask.Where}: captured canSkip {p.CanSkip}, skip button visible {skip.Value}");
        }
    }

    private static void AfterCountOptions(Player me, CombatState cs, bool includePotions, int __result)
    {
        if (!includePotions)
        {
            return; // the tool's second call (its own potion-window classification)
        }
        int k = _k++;
        try
        {
            Compare(me, cs, k, __result);
        }
        catch (Exception e)
        {
            _probeError ??= $"{_file} action {k}: {e.GetType().Name}: {e.Message}";
        }
    }

    private static void Example(string text)
    {
        if (_examples.Count < 24)
        {
            _examples.Add(text);
        }
    }

    private static void Compare(Player me, CombatState cs, int k, int toolCount)
    {
        _compared++;
        CombatManager cm = CombatManager.Instance;
        RunManager rm = RunManager.Instance;
        IReadOnlyList<Player> seats = me.RunState.Players;
        int seat = -1;
        for (int i = 0; i < seats.Count; i++)
        {
            if (ReferenceEquals(seats[i], me))
            {
                seat = i;
            }
        }
        Decision d = _decision.Reset(DecisionKind.Combat, null);
        NotReady why = CombatHandler.Enumerate(cm, rm, seats, me, _info, d, out _);
        if (why == NotReady.Ready && seat >= 0)
        {
            why = _info.Seats[seat].State;
        }
        JsonNode? rec = _actions != null && k < _actions.Count ? _actions[k] : null;
        string where = $"{_file} action {k} turn {me.PlayerCombatState?.TurnNumber}";
        if (why != NotReady.Ready || seat < 0)
        {
            _notReady++;
            string reason = seat < 0 ? "seat_not_found" : why.ToString();
            _notReadyBy[reason] = _notReadyBy.GetValueOrDefault(reason) + 1;
            Example($"{where}: the handler would not ask here ({reason}); recorded {rec?["Kind"]}");
            return;
        }
        if (!ReferenceEquals(_info.State, cs))
        {
            _probeError ??= where + ": the handler enumerated a different combat state";
        }

        int mine = 0;
        int potionActions = 0;
        for (int i = 0; i < d.Actions.Count; i++)
        {
            LegalAction x = d.Actions[i];
            if (x.Seat != seat)
            {
                continue;
            }
            mine++;
            switch (x.Verb)
            {
                case Verb.PlayCard:
                    _offeredPlays++;
                    break;
                case Verb.UsePotion:
                    _offeredPotionUses++;
                    potionActions++;
                    break;
                case Verb.DiscardPotion:
                    _offeredDiscards++;
                    potionActions++;
                    break;
                case Verb.EndTurn:
                    _offeredEndTurns++;
                    break;
            }
        }
        if (potionActions > 0)
        {
            _withPotionActions++;
        }
        if (mine == toolCount)
        {
            _equalTool++;
        }
        else
        {
            _differsTool++;
            Example($"{where}: handler offers {mine} actions, the replay tool counts {toolCount}");
        }
        if (rec == null)
        {
            return;
        }
        int? recorded = rec["Options"]?.GetValue<int>();
        if (recorded.HasValue)
        {
            _recordedAvailable++;
            if (recorded.Value == mine)
            {
                _equalRecorded++;
            }
            else if (potionActions > 0 && recorded.Value == mine - potionActions)
            {
                _potionWindow++;
                _replaysInWindow.Add(_file);
            }
            else
            {
                _differsOther++;
                Example($"{where}: handler offers {mine} actions ({potionActions} potion actions), the real client's collector recorded {recorded.Value}");
            }
        }

        // ---- the action the real client took must be legal for the handler ----
        string kind = rec["Kind"]?.GetValue<string>() ?? "";
        _recordedKinds[kind] = _recordedKinds.GetValueOrDefault(kind) + 1;
        string? card = rec["Card"]?.GetValue<string>();
        string? potion = rec["Potion"]?.GetValue<string>();
        int target = rec["Target"]?.GetValue<int>() ?? -1;
        IReadOnlyList<Creature> enemies = cs.Enemies;
        bool found = false;
        bool parsed = true;
        for (int i = 0; i < d.Actions.Count && !found; i++)
        {
            LegalAction x = d.Actions[i];
            if (x.Seat != seat)
            {
                continue;
            }
            switch (kind)
            {
                case "play":
                    if (x.Verb == Verb.PlayCard && card != null)
                    {
                        CardModel c = x.Source.Card;
                        if (c.Id.Entry + (c.IsUpgraded ? "+" : "") == card)
                        {
                            // Target: index into CombatState.Enemies, -2 = the player's own creature, -1 = none.
                            found = target >= 0 ? !x.Target.IsNone && target < enemies.Count && ReferenceEquals(x.Target.Creature, enemies[target])
                                : target == -2 ? !x.Target.IsNone && ReferenceEquals(x.Target.Creature, me.Creature)
                                : x.Target.IsNone;
                        }
                    }
                    break;
                case "potion":
                    if (x.Verb == Verb.UsePotion && potion != null && me.PotionSlots[x.Source.Index]?.Id.Entry == potion)
                    {
                        // The collector passes the enemy for AnyEnemy potions; for every other potion the argument is the
                        // owner (Self) or null, which the game completes to the owner where that is valid (PotionModel.cs:220-223).
                        found = target >= 0 ? !x.Target.IsNone && target < enemies.Count && ReferenceEquals(x.Target.Creature, enemies[target])
                            : x.Target.IsNone || ReferenceEquals(x.Target.Creature, me.Creature);
                    }
                    break;
                case "discard_potion":
                    found = x.Verb == Verb.DiscardPotion && potion != null && me.PotionSlots[x.Source.Index]?.Id.Entry == potion;
                    break;
                case "end_turn":
                    found = x.Verb == Verb.EndTurn;
                    break;
                default:
                    parsed = false;
                    break;
            }
        }
        if (!parsed)
        {
            _actionUnparsed++;
            Example($"{where}: unknown recorded action kind '{kind}'");
        }
        else if (found)
        {
            _actionOffered++;
        }
        else
        {
            _actionMissing++;
            Example($"{where}: the real client took {kind} {card ?? potion} target {target}, which the handler does not offer");
        }
    }
}
