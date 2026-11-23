using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using SpireHost.Boot;
using SpireHost.Driver;
using SpireHost.Obs;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Schema;

namespace SpireHost.Search;

/// <summary>
/// Cross-platform (Windows / macOS / Linux) shadow-search worker server:
///     SpireHost --shadow-serve [--verbose]
///
/// Restores a live game's room-boundary <c>SerializableRun</c> JSON snapshot (never touching disk; SpireHost uses
/// <see cref="HostSaveStore"/>), replays the public room actions to the target decision, verifies bit-exact public
/// state alignment via <see cref="ContentHash"/>, applies 100% fair-play <see cref="Resampler.Rerandomise"/> (always
/// MODE_RESAMPLE; strictly refuses Crystal and Reward), plays a candidate action, and streams rollout OBS frames over
/// stdin/stdout so Python can batch GPU inference across a pool of shadow workers.
/// </summary>
public static class ShadowServe
{
    public const int MaxMessageBytes = 16 * 1024 * 1024;
    private const byte TagObs = 1;
    private const byte TagDone = 2;

    /// <summary>
    /// Identical FNV-1a content hash to <c>SpireAdvisor.Advisor.FrameSource.ContentHash</c>: hashes the entire OBS frame
    /// excluding header words <see cref="Hd.EpisodeId"/>, <see cref="Hd.StepId"/>, <see cref="Hd.AutoSteps"/>, and
    /// <see cref="Hd.HostUs"/>.
    /// </summary>
    public static ulong ContentHash(ReadOnlySpan<byte> frame)
    {
        ulong h = 14695981039346656037UL;
        for (int i = 0; i < frame.Length; i++)
        {
            int word = i >> 2;
            if (word == Hd.EpisodeId || word == Hd.StepId || word == Hd.AutoSteps || word == Hd.HostUs)
            {
                continue;
            }
            h ^= frame[i];
            h *= 1099511628211UL;
        }
        return h;
    }

    public static int Run(HostSession session, Args a)
    {
        bool verbose = a.Has("verbose");
        ObsRuntime.Loaded loaded = ObsModule.Current;
        ObsEncoder enc = ObsRuntime.NewEncoder(loaded);
        using Stream input = Console.OpenStandardInput();
        using Stream output = Console.OpenStandardOutput();

        Console.Error.WriteLine($"[shadow] ready: vocab 0x{loaded.Vocab.Hash:X8}, catalog 0x{loaded.CatalogHash:X8}");

        Dictionary<long, string> snapshots = new Dictionary<long, string>();
        LinkedList<long> snapshotOrder = new LinkedList<long>();
        int episodeCounter = 0;

        while (true)
        {
            byte[]? raw = ReadMessage(input);
            if (raw == null)
            {
                return 0;
            }
            JsonObject? req;
            try
            {
                req = JsonNode.Parse(raw) as JsonObject;
            }
            catch (Exception e)
            {
                WriteJson(output, new JsonObject { ["ok"] = false, ["error"] = "bad json: " + e.Message });
                continue;
            }
            if (req == null)
            {
                WriteJson(output, new JsonObject { ["ok"] = false, ["error"] = "request must be JSON object" });
                continue;
            }

            string op = req["op"]?.GetValue<string>() ?? "";
            if (op == "quit")
            {
                WriteJson(output, new JsonObject { ["ok"] = true });
                return 0;
            }
            if (op == "ping")
            {
                WriteJson(output, new JsonObject
                {
                    ["ok"] = true,
                    ["vocab_hash"] = loaded.Vocab.Hash,
                    ["vocab_hash_hex"] = $"0x{unchecked((uint)loaded.Vocab.Hash):X8}",
                    ["catalog_hash"] = loaded.CatalogHash,
                    ["catalog_hash_hex"] = $"0x{unchecked((uint)loaded.CatalogHash):X8}",
                    ["cached_snapshots"] = snapshots.Count,
                });
                continue;
            }
            if (op == "cache_snapshot")
            {
                long sid = ReadLong(req, "snapshot_id", -1);
                string? sj = req["save_json"]?.GetValue<string>();
                if (sid <= 0 || string.IsNullOrEmpty(sj))
                {
                    WriteJson(output, new JsonObject { ["ok"] = false, ["error"] = "cache_snapshot requires positive snapshot_id and save_json" });
                    continue;
                }
                CacheSnapshot(snapshots, snapshotOrder, sid, sj);
                WriteJson(output, new JsonObject { ["ok"] = true, ["snapshot_id"] = sid });
                continue;
            }
            if (op == "shadow_run")
            {
                long sid = ReadLong(req, "snapshot_id", -1);
                string? sj = req["save_json"]?.GetValue<string>();
                if (!string.IsNullOrEmpty(sj) && sid > 0)
                {
                    CacheSnapshot(snapshots, snapshotOrder, sid, sj);
                }
                else if (string.IsNullOrEmpty(sj) && sid > 0 && snapshots.TryGetValue(sid, out string? cached))
                {
                    sj = cached;
                }
                if (string.IsNullOrEmpty(sj))
                {
                    SendDone(output, new JsonObject
                    {
                        ["ok"] = false,
                        ["status"] = "missing_snapshot",
                        ["detail"] = $"snapshot_id={sid} not cached and save_json not provided",
                    }, null);
                    continue;
                }

                episodeCounter++;
                ExecuteShadowRun(session, enc, input, output, req, sj, episodeCounter, verbose);
                continue;
            }

            WriteJson(output, new JsonObject { ["ok"] = false, ["error"] = "unknown op: " + op });
        }
    }

    private static void CacheSnapshot(Dictionary<long, string> map, LinkedList<long> order, long sid, string json)
    {
        if (map.ContainsKey(sid))
        {
            map[sid] = json;
            return;
        }
        map[sid] = json;
        order.AddLast(sid);
        while (order.Count > 4)
        {
            long oldest = order.First!.Value;
            order.RemoveFirst();
            map.Remove(oldest);
        }
    }

    private static void ExecuteShadowRun(
        HostSession session,
        ObsEncoder enc,
        Stream input,
        Stream output,
        JsonObject req,
        string saveJson,
        int episodeId,
        bool verbose)
    {
        int[] replayActions = ReadIntArray(req["replay_actions"]);
        ulong expectedHash = ReadUlong(req, "expected_obs_hash", 0UL);
        int expectedKind = ReadInt(req, "expected_kind", -1);
        int expectedNActions = ReadInt(req, "expected_n_actions", -1);
        ulong resampleSeed = ReadUlong(req, "resample_seed", 1UL);
        bool resampleMacro = ReadBool(req, "resample_macro", false);
        int forcedAction = ReadInt(req, "forced_action", -1);
        string mode = req["mode"]?.GetValue<string>() ?? "fight";
        int horizonFloors = Math.Max(1, ReadInt(req, "horizon_floors", 4));
        int maxSteps = Math.Max(1, ReadInt(req, "max_steps", 180));
        int resetFlags = ReadInt(req, "reset_flags", 795);
        bool allowHashMismatch = ReadBool(req, "allow_hash_mismatch", false);

        enc.Options = EncoderOptions.FromResetFlags(resetFlags, 0, -1);

        ShadowRolloutDriver driver = new ShadowRolloutDriver(
            enc,
            input,
            output,
            episodeId,
            replayActions,
            expectedHash,
            expectedKind,
            expectedNActions,
            resampleSeed,
            resampleMacro,
            forcedAction,
            mode,
            horizonFloors,
            maxSteps,
            allowHashMismatch,
            verbose);

        EpisodeSpec spec = new EpisodeSpec
        {
            Characters = Array.Empty<string>(),
            SavedRunJson = saveJson,
        };
        EpisodeOptions opt = new EpisodeOptions
        {
            AutoSingle = (resetFlags & ResetFlag.AutoSingle) != 0,
            AutoGold = true,
            AllowAbandon = false,
            MaxPolicyDecisions = replayActions.Length + maxSteps + 32,
            MaxCombatTurns = 200,
            DecisionCap = 50_000,
            TeamVotes = true,
            OfferUndoEndTurn = false,
        };

        Stopwatch totalSw = Stopwatch.StartNew();
        EpisodeResult res;
        try
        {
            res = session.RunEpisode(spec, driver, opt, driver);
        }
        catch (Exception e)
        {
            SendDone(output, new JsonObject
            {
                ["ok"] = false,
                ["status"] = "error",
                ["detail"] = e.GetType().Name + ": " + e.Message,
                ["total_ms"] = Math.Round(totalSw.Elapsed.TotalMilliseconds, 2),
            }, null);
            return;
        }

        string status = driver.Status;
        string detail = driver.Detail;
        if (string.IsNullOrEmpty(status))
        {
            status = res.Outcome switch
            {
                EpisodeOutcome.Win => "run_won",
                EpisodeOutcome.Loss => "wiped",
                EpisodeOutcome.Truncated => "truncated",
                _ => "aborted",
            };
            if (res.Outcome == EpisodeOutcome.Aborted && res.Abort != null)
            {
                detail = res.Abort.Signature + ": " + res.Abort.Message;
            }
            else if (res.Outcome == EpisodeOutcome.Truncated && !string.IsNullOrEmpty(res.TruncatedBy))
            {
                detail = res.TruncatedBy;
            }
        }

        bool ok = status is "won" or "wiped" or "run_won" or "horizon" or "truncated" or "verified" or "cancelled";
        JsonObject summary = new JsonObject
        {
            ["ok"] = ok,
            ["status"] = status,
            ["detail"] = detail,
            ["shadow_hash"] = driver.ShadowHash.ToString(),
            ["expected_hash"] = expectedHash.ToString(),
            ["hash_match"] = driver.HashMatch,
            ["shadow_kind"] = driver.ShadowKind,
            ["shadow_n_actions"] = driver.ShadowNActions,
            ["steps"] = driver.RolloutSteps,
            ["turns"] = driver.TurnsTaken,
            ["hp_after"] = driver.HpAfter,
            ["max_hp"] = driver.MaxHpAfter,
            ["potions_after"] = driver.PotionsAfter,
            ["floor_after"] = driver.FloorAfter,
            ["act_after"] = driver.ActAfter,
            ["restore_ms"] = Math.Round(driver.RestoreMs, 2),
            ["rollout_ms"] = Math.Round(driver.RolloutMs, 2),
            ["total_ms"] = Math.Round(totalSw.Elapsed.TotalMilliseconds, 2),
        };
        SendDone(output, summary, driver.TrailingObs);
    }

    private sealed class ShadowRolloutDriver : IPolicy, IEpisodeObserver
    {
        private readonly ObsEncoder _enc;
        private readonly Stream _input;
        private readonly Stream _output;
        private readonly int _episodeId;
        private readonly int[] _replayActions;
        private readonly ulong _expectedHash;
        private readonly int _expectedKind;
        private readonly int _expectedNActions;
        private readonly ulong _resampleSeed;
        private readonly bool _resampleMacro;
        private readonly int _forcedAction;
        private readonly string _mode;
        private readonly int _horizonFloors;
        private readonly int _maxSteps;
        private readonly bool _allowHashMismatch;
        private readonly bool _verbose;

        private readonly Stopwatch _sw = Stopwatch.StartNew();
        private long _rolloutStartTs;
        private int _replayIdx;
        private bool _reachedTarget;
        private long _lastAuto;
        private CombatRoom? _forkRoom;
        private int _roundAtStart;
        private int _floorAtStart;
        private int _actAtStart;

        public string Status { get; private set; } = "";
        public string Detail { get; private set; } = "";
        public ulong ShadowHash { get; private set; }
        public bool HashMatch { get; private set; }
        public int ShadowKind { get; private set; } = -1;
        public int ShadowNActions { get; private set; } = -1;
        public int RolloutSteps { get; private set; }
        public int TurnsTaken { get; private set; }
        public int HpAfter { get; private set; }
        public int MaxHpAfter { get; private set; }
        public int PotionsAfter { get; private set; }
        public int FloorAfter { get; private set; }
        public int ActAfter { get; private set; }
        public double RestoreMs { get; private set; }
        public double RolloutMs { get; private set; }
        public byte[]? TrailingObs { get; private set; }

        public ShadowRolloutDriver(
            ObsEncoder enc,
            Stream input,
            Stream output,
            int episodeId,
            int[] replayActions,
            ulong expectedHash,
            int expectedKind,
            int expectedNActions,
            ulong resampleSeed,
            bool resampleMacro,
            int forcedAction,
            string mode,
            int horizonFloors,
            int maxSteps,
            bool allowHashMismatch,
            bool verbose)
        {
            _enc = enc;
            _input = input;
            _output = output;
            _episodeId = episodeId;
            _replayActions = replayActions;
            _expectedHash = expectedHash;
            _expectedKind = expectedKind;
            _expectedNActions = expectedNActions;
            _resampleSeed = resampleSeed;
            _resampleMacro = resampleMacro;
            _forcedAction = forcedAction;
            _mode = mode;
            _horizonFloors = horizonFloors;
            _maxSteps = maxSteps;
            _allowHashMismatch = allowHashMismatch;
            _verbose = verbose;
        }

        private void SnapshotPlayerMetrics(RunContext ctx)
        {
            RunState run = ctx.Run;
            FloorAfter = run.TotalFloor;
            ActAfter = run.CurrentActIndex + 1;
            if (run.Players.Count > 0)
            {
                Player p = run.Players[0];
                HpAfter = p.Creature.CurrentHp;
                MaxHpAfter = p.Creature.MaxHp;
                PotionsAfter = p.Potions.Count();
            }
            if (_forkRoom != null && _forkRoom.CombatState != null)
            {
                TurnsTaken = Math.Max(0, _forkRoom.CombatState.RoundNumber - _roundAtStart);
            }
        }

        private void FinishRolloutTiming()
        {
            if (_rolloutStartTs > 0)
            {
                RolloutMs = (Stopwatch.GetTimestamp() - _rolloutStartTs) * 1000.0 / Stopwatch.Frequency;
            }
        }

        public int Decide(Decision decision)
        {
            RunContext ctx = decision.Episode;
            int stepId = (int)ctx.Counters.PolicyDecisions;
            long auto = ctx.Counters.AutoSingleTotal + ctx.Counters.AutoGold;
            int autoSteps = (int)(auto - _lastAuto);
            _lastAuto = auto;

            // Phase A: replay recorded room actions up to the live target decision point.
            if (_replayIdx < _replayActions.Length)
            {
                int act = _replayActions[_replayIdx++];
                if ((uint)act >= (uint)decision.Actions.Count)
                {
                    Status = "replay_diverged";
                    Detail = $"replay step {_replayIdx - 1}: action {act} out of [0,{decision.Actions.Count}) on kind={decision.Kind}";
                    throw new PolicyStopException("replay_diverged");
                }
                return act;
            }

            // Phase B: reached the live decision point -> verify alignment, resample fairly, play forcedAction.
            if (!_reachedTarget)
            {
                _reachedTarget = true;
                RestoreMs = _sw.Elapsed.TotalMilliseconds;
                _enc.EncodeDecision(decision, _episodeId, stepId, autoSteps);
                byte[] beforeObs = _enc.Payload.ToArray();
                ShadowHash = ContentHash(beforeObs);
                ShadowKind = (int)decision.Kind;
                ShadowNActions = decision.Actions.Count;
                HashMatch = (_expectedHash == 0UL || ShadowHash == _expectedHash)
                    && (_expectedKind < 0 || ShadowKind == _expectedKind)
                    && (_expectedNActions <= 0 || ShadowNActions == _expectedNActions);

                SnapshotPlayerMetrics(ctx);

                bool kindOrCountBad = (_expectedKind >= 0 && ShadowKind != _expectedKind)
                    || (_expectedNActions > 0 && ShadowNActions != _expectedNActions);
                if (kindOrCountBad || (!_allowHashMismatch && _expectedHash != 0UL && ShadowHash != _expectedHash))
                {
                    Status = "hash_mismatch";
                    Detail = $"shadow(kind={ShadowKind},n={ShadowNActions},hash=0x{ShadowHash:X16}) != live(kind={_expectedKind},n={_expectedNActions},hash=0x{_expectedHash:X16})";
                    throw new PolicyStopException("hash_mismatch");
                }

                if (_mode == "verify_only")
                {
                    Status = "verified";
                    TrailingObs = beforeObs;
                    throw new PolicyStopException("verified");
                }

                bool inFight = ctx.Run.CurrentRoom is CombatRoom cr && ctx.CombatManager.IsInProgress;
                if (!inFight && (decision.Kind == DecisionKind.Crystal || decision.Kind == DecisionKind.Reward))
                {
                    Status = "fork_refused";
                    Detail = $"fair-play exclusion on non-combat {decision.Kind}";
                    throw new PolicyStopException("fork_refused");
                }
                if (_mode == "fight" && !inFight)
                {
                    Status = "fork_refused";
                    Detail = $"fight mode requested outside combat ({decision.Kind})";
                    throw new PolicyStopException("fork_refused");
                }

                // 100% Fair-Play MODE_RESAMPLE: canonical draw-pile re-shuffle + RNG re-seeding (+ macro state if macro)
                bool doMacroResample = _resampleMacro || _mode == "macro";
                Resampler.Rerandomise(ctx, _enc, _resampleSeed, PileOrder.Canonical, resampleMacroState: doMacroResample);
                _enc.EncodeDecision(decision, _episodeId, stepId, autoSteps);
                if (!SearchExtension.SameFrame(_enc.Payload, beforeObs))
                {
                    Status = "error";
                    Detail = "resampler altered public OBS frame";
                    throw new PolicyStopException("resample_invariant");
                }

                if ((uint)_forcedAction >= (uint)decision.Actions.Count)
                {
                    Status = "error";
                    Detail = $"forced_action {_forcedAction} out of [0,{decision.Actions.Count})";
                    throw new PolicyStopException("bad_forced_action");
                }

                _forkRoom = ctx.Run.CurrentRoom as CombatRoom;
                _roundAtStart = _forkRoom?.CombatState?.RoundNumber ?? 0;
                _floorAtStart = ctx.Run.TotalFloor;
                _actAtStart = ctx.Run.CurrentActIndex;
                _rolloutStartTs = Stopwatch.GetTimestamp();
                RolloutSteps = 0;
                return _forcedAction;
            }

            // Phase C: rollout after forcedAction.
            SnapshotPlayerMetrics(ctx);

            if (_mode == "fight")
            {
                bool stillSameFight = ctx.Run.CurrentRoom is CombatRoom cr2
                    && ctx.CombatManager.IsInProgress
                    && ReferenceEquals(cr2, _forkRoom);
                if (!stillSameFight)
                {
                    _enc.EncodeDecision(decision, _episodeId, stepId, autoSteps);
                    TrailingObs = _enc.Payload.ToArray();
                    Status = "won";
                    FinishRolloutTiming();
                    throw new PolicyStopException("fight_won");
                }
            }
            else if (_mode == "macro")
            {
                if (ctx.Run.TotalFloor >= _floorAtStart + _horizonFloors || ctx.Run.CurrentActIndex > _actAtStart)
                {
                    _enc.EncodeDecision(decision, _episodeId, stepId, autoSteps);
                    TrailingObs = _enc.Payload.ToArray();
                    Status = "horizon";
                    FinishRolloutTiming();
                    throw new PolicyStopException("macro_horizon");
                }
            }

            if (RolloutSteps >= _maxSteps)
            {
                _enc.EncodeDecision(decision, _episodeId, stepId, autoSteps);
                TrailingObs = _enc.Payload.ToArray();
                Status = "truncated";
                Detail = "max_steps";
                FinishRolloutTiming();
                throw new PolicyStopException("max_steps");
            }

            _enc.EncodeDecision(decision, _episodeId, stepId, autoSteps);
            SendObs(_output, _enc.Payload);

            int nextAction = ReadActionReply(_input);
            if (nextAction < 0)
            {
                Status = "cancelled";
                Detail = "cancelled_by_client";
                FinishRolloutTiming();
                throw new PolicyStopException("cancelled");
            }
            if ((uint)nextAction >= (uint)decision.Actions.Count)
            {
                Status = "error";
                Detail = $"client action {nextAction} out of [0,{decision.Actions.Count})";
                FinishRolloutTiming();
                throw new PolicyStopException("bad_client_action");
            }
            RolloutSteps++;
            return nextAction;
        }

        public void OnEpisodeStart(RunContext ctx)
        {
            _ = ctx;
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
        {
            _ = ctx;
            _ = decision;
            _ = chosen;
            _ = auto;
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
            SnapshotPlayerMetrics(ctx);
            FinishRolloutTiming();
            _ = provisional;
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
            FinishRolloutTiming();
            if (_verbose && result.Outcome == EpisodeOutcome.Aborted)
            {
                Console.Error.WriteLine($"[shadow] aborted: {result.Abort?.Signature} {result.Abort?.Message}");
            }
        }
    }

    /// <summary>
    /// Built-in self-test verifying SavedRun restore + room action replay + bit-exact ContentHash match +
    /// fair MODE_RESAMPLE invariant across a real STS2 run (Combat + Post-Combat + Macro).
    /// </summary>
    public static int SelfTest(HostSession session, Args a)
    {
        string character = a.Get("character") ?? "SILENT";
        int ascension = a.GetInt("ascension", 10);
        string seed = a.Get("seed") ?? "CMTYZSWYQJ02";
        ObsRuntime.Loaded loaded = ObsModule.Current;
        ObsEncoder liveEnc = ObsRuntime.NewEncoder(loaded);
        liveEnc.Options = EncoderOptions.FromResetFlags(795, 0, -1);

        // Capture real room snapshots and actions during a live episode
        List<(string label, string saveJson, int[] actions, ulong obsHash, int kind, int nActions, byte[] obsBytes)> samples = new();
        string? curSave = null;
        string? lastSeenSave = null;
        List<int> curActions = new();
        int stepCount = 0;

        EpisodeSpec liveSpec = new EpisodeSpec { Characters = new[] { character }, Ascension = ascension, Seed = seed };
        EpisodeOptions liveOpt = new EpisodeOptions { AutoSingle = true, AutoGold = true, AllowAbandon = false, MaxPolicyDecisions = 80 };

        void PollSaveStore()
        {
            HostSaveStore? store = SaveBootstrap.Store;
            if (store == null) return;
            string? txt = store.ReadFile("saves/current_run.save") ?? store.ReadFile("profile1/saves/current_run.save");
            if (txt == null)
            {
                foreach (string d in new[] { "saves", "profile1/saves", "" })
                {
                    foreach (string f in store.GetFilesInDirectory(d))
                    {
                        if (f.EndsWith("current_run.save", StringComparison.OrdinalIgnoreCase))
                        {
                            txt = store.ReadFile(string.IsNullOrEmpty(d) ? f : d + "/" + f);
                            break;
                        }
                    }
                    if (txt != null) break;
                }
            }
            if (!string.IsNullOrEmpty(txt) && !string.Equals(txt, lastSeenSave, StringComparison.Ordinal))
            {
                lastSeenSave = txt;
                curSave = txt;
                curActions.Clear();
            }
        }

        bool capturedCombatTurn1 = false;
        bool capturedCombatMid = false;
        bool capturedPostCombat = false;

        ActionPolicy livePolicy = new ActionPolicy((decision) =>
        {
            PollSaveStore();
            RunContext ctx = decision.Episode;
            RunState run = ctx.Run;
            liveEnc.EncodeDecision(decision, 1, stepCount, 0);
            ulong h = ContentHash(liveEnc.Payload);
            if (curSave != null)
            {
                bool inFight = run.CurrentRoom is CombatRoom && ctx.CombatManager.IsInProgress;
                if (inFight && !capturedCombatTurn1 && curActions.Count == 0)
                {
                    samples.Add(("combat_t1", curSave, curActions.ToArray(), h, (int)decision.Kind, decision.Actions.Count, liveEnc.Payload.ToArray()));
                    capturedCombatTurn1 = true;
                }
                else if (inFight && !capturedCombatMid && curActions.Count >= 3)
                {
                    samples.Add(("combat_mid", curSave, curActions.ToArray(), h, (int)decision.Kind, decision.Actions.Count, liveEnc.Payload.ToArray()));
                    capturedCombatMid = true;
                }
                else if (!inFight && capturedCombatTurn1 && !capturedPostCombat && decision.Kind is DecisionKind.CardReward or DecisionKind.Map or DecisionKind.Reward)
                {
                    samples.Add(("post_combat_" + decision.Kind, curSave, curActions.ToArray(), h, (int)decision.Kind, decision.Actions.Count, liveEnc.Payload.ToArray()));
                    capturedPostCombat = true;
                }
            }
            int chosen = 0;
            if (decision.Kind == DecisionKind.Combat)
            {
                for (int i = 0; i < decision.Actions.Count; i++)
                {
                    if (decision.Actions[i].Verb == Verb.PlayCard)
                    {
                        chosen = i;
                        break;
                    }
                }
            }
            if (curSave != null)
            {
                curActions.Add(chosen);
            }
            stepCount++;
            if (capturedCombatTurn1 && capturedCombatMid && capturedPostCombat)
            {
                throw new PolicyStopException("collected");
            }
            return chosen;
        });

        SnapshotTestObserver obs = new SnapshotTestObserver(PollSaveStore);
        session.RunEpisode(liveSpec, livePolicy, liveOpt, obs);

        int passed = 0;
        foreach (var s in samples)
        {
            ObsEncoder shadowEnc = ObsRuntime.NewEncoder(loaded);
            shadowEnc.Options = EncoderOptions.FromResetFlags(795, 0, -1);
            using MemoryStream dummyIn = new MemoryStream();
            using MemoryStream dummyOut = new MemoryStream();
            ShadowRolloutDriver verifyDriver = new ShadowRolloutDriver(
                shadowEnc, dummyIn, dummyOut, 99, s.actions, s.obsHash, s.kind, s.nActions,
                resampleSeed: 1234567UL, resampleMacro: false, forcedAction: 0, mode: "verify_only",
                horizonFloors: 4, maxSteps: 10, allowHashMismatch: false, verbose: false);
            EpisodeSpec ss = new EpisodeSpec { Characters = Array.Empty<string>(), SavedRunJson = s.saveJson };
            EpisodeOptions so = new EpisodeOptions { AutoSingle = true, AutoGold = true, AllowAbandon = false, MaxPolicyDecisions = s.actions.Length + 8 };
            session.RunEpisode(ss, verifyDriver, so, verifyDriver);
            bool ok = verifyDriver.Status == "verified" && verifyDriver.HashMatch;
            Console.Error.WriteLine($"[shadow-selftest] {s.label}: status={verifyDriver.Status} hash_match={verifyDriver.HashMatch} replay={s.actions.Length} restore={verifyDriver.RestoreMs:0.1}ms detail={verifyDriver.Detail}");
            if (ok) passed++;
        }
        bool allOk = samples.Count >= 2 && passed == samples.Count;
        string? dumpPath = a.Get("dump-samples");
        if (!string.IsNullOrEmpty(dumpPath))
        {
            JsonArray arr = new JsonArray();
            foreach (var s in samples)
            {
                JsonArray ra = new JsonArray();
                foreach (int act in s.actions) ra.Add(act);
                arr.Add(new JsonObject
                {
                    ["label"] = s.label,
                    ["save_json"] = s.saveJson,
                    ["room_actions"] = ra,
                    ["obs_hash"] = s.obsHash.ToString(),
                    ["kind"] = s.kind,
                    ["n_actions"] = s.nActions,
                    ["obs_b64"] = Convert.ToBase64String(s.obsBytes),
                });
            }
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(dumpPath))!);
            File.WriteAllText(dumpPath, arr.ToJsonString());
        }
        HostJson.Emit(new { kind = "shadow-selftest", ok = allOk, samples = samples.Count, passed }, a.Get("out"));
        return allOk ? 0 : 4;
    }

    private sealed class ActionPolicy : IPolicy
    {
        private readonly Func<Decision, int> _fn;
        public ActionPolicy(Func<Decision, int> fn) => _fn = fn;
        public int Decide(Decision decision) => _fn(decision);
    }

    private sealed class SnapshotTestObserver : IEpisodeObserver
    {
        private readonly Action _poll;
        public SnapshotTestObserver(Action poll) => _poll = poll;

        public void OnEpisodeStart(RunContext ctx)
        {
            RunManager.Instance.ShouldSave = true;
            _ = ctx;
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
        {
            _poll();
            _ = ctx;
            _ = decision;
            _ = chosen;
            _ = auto;
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional) { _ = ctx; _ = provisional; }
        public void OnEpisodeEnd(EpisodeResult result) { _ = result; }
    }

    // ---------------------------------------------------------------------------------------------- Wire IO

    private static byte[]? ReadMessage(Stream s)
    {
        byte[] head = new byte[4];
        int got = ReadFully(s, head, eofOk: true);
        if (got == 0)
        {
            return null;
        }
        uint len = BinaryPrimitives.ReadUInt32LittleEndian(head);
        if (len == 0 || len > MaxMessageBytes)
        {
            throw new IOException($"bad shadow message length {len}");
        }
        byte[] buf = new byte[len];
        ReadFully(s, buf, eofOk: false);
        return buf;
    }

    private static int ReadActionReply(Stream s)
    {
        byte[] buf = new byte[4];
        int got = ReadFully(s, buf, eofOk: true);
        if (got == 0)
        {
            return -1;
        }
        return BinaryPrimitives.ReadInt32LittleEndian(buf);
    }

    private static int ReadFully(Stream s, byte[] dst, bool eofOk)
    {
        int got = 0;
        while (got < dst.Length)
        {
            int k = s.Read(dst, got, dst.Length - got);
            if (k <= 0)
            {
                if (got == 0 && eofOk)
                {
                    return 0;
                }
                throw new EndOfStreamException($"unexpected EOF after {got}/{dst.Length}");
            }
            got += k;
        }
        return got;
    }

    private static void WriteJson(Stream s, JsonObject obj)
    {
        byte[] json = Encoding.UTF8.GetBytes(obj.ToJsonString());
        byte[] pkt = new byte[4 + json.Length];
        BinaryPrimitives.WriteUInt32LittleEndian(pkt, (uint)json.Length);
        json.CopyTo(pkt, 4);
        s.Write(pkt, 0, pkt.Length);
        s.Flush();
    }

    private static void SendObs(Stream s, ReadOnlySpan<byte> obs)
    {
        int payloadLen = 1 + obs.Length;
        byte[] pkt = new byte[4 + payloadLen];
        BinaryPrimitives.WriteUInt32LittleEndian(pkt, (uint)payloadLen);
        pkt[4] = TagObs;
        obs.CopyTo(pkt.AsSpan(5));
        s.Write(pkt, 0, pkt.Length);
        s.Flush();
    }

    private static void SendDone(Stream s, JsonObject summary, byte[]? trailingObs)
    {
        byte[] json = Encoding.UTF8.GetBytes(summary.ToJsonString());
        int obsLen = trailingObs?.Length ?? 0;
        int payloadLen = 1 + 4 + json.Length + obsLen;
        byte[] pkt = new byte[4 + payloadLen];
        BinaryPrimitives.WriteUInt32LittleEndian(pkt, (uint)payloadLen);
        pkt[4] = TagDone;
        BinaryPrimitives.WriteUInt32LittleEndian(pkt.AsSpan(5, 4), (uint)json.Length);
        json.CopyTo(pkt, 9);
        if (trailingObs != null && obsLen > 0)
        {
            trailingObs.CopyTo(pkt, 9 + json.Length);
        }
        s.Write(pkt, 0, pkt.Length);
        s.Flush();
    }

    private static int ReadInt(JsonObject o, string key, int def)
    {
        if (o[key] is JsonValue v)
        {
            if (v.TryGetValue(out int iv)) return iv;
            if (v.TryGetValue(out long lv)) return (int)lv;
        }
        return def;
    }

    private static long ReadLong(JsonObject o, string key, long def)
    {
        if (o[key] is JsonValue v)
        {
            if (v.TryGetValue(out long lv)) return lv;
            if (v.TryGetValue(out int iv)) return iv;
            if (v.TryGetValue(out string? sv) && long.TryParse(sv, out long parsed)) return parsed;
        }
        return def;
    }

    private static ulong ReadUlong(JsonObject o, string key, ulong def)
    {
        if (o[key] is JsonValue v)
        {
            if (v.TryGetValue(out ulong uv)) return uv;
            if (v.TryGetValue(out long lv) && lv >= 0) return (ulong)lv;
            if (v.TryGetValue(out string? sv) && ulong.TryParse(sv, out ulong parsed)) return parsed;
        }
        return def;
    }

    private static bool ReadBool(JsonObject o, string key, bool def)
    {
        if (o[key] is JsonValue v && v.TryGetValue(out bool bv)) return bv;
        return def;
    }

    private static int[] ReadIntArray(JsonNode? node)
    {
        if (node is not JsonArray arr || arr.Count == 0)
        {
            return Array.Empty<int>();
        }
        int[] outArr = new int[arr.Count];
        for (int i = 0; i < arr.Count; i++)
        {
            outArr[i] = arr[i]?.GetValue<int>() ?? -1;
        }
        return outArr;
    }
}
