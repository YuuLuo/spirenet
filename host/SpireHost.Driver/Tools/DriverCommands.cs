using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using SpireHost.Boot;
using SpireHost.Patches;

namespace SpireHost.Driver.Tools;

/// <summary>
/// The run-level commands (--fuzz-runs, --replay-run, --determinism --runs N, --driver-info). They live in
/// the Driver library so that the main executable and every team's own console entry run exactly the same
/// code against whatever handlers are registered in the session.
/// </summary>
public static class DriverCommands
{
    public static readonly string[] AllCharacters = { "IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT", "FOOL" };

    public static EpisodeOptions EpisodeOptionsFrom(Args a)
    {
        return new EpisodeOptions
        {
            AutoSingle = a.GetBool("auto-single", true),
            AutoGold = a.GetBool("auto-gold", true),
            AllowAbandon = a.GetBool("allow-abandon", false),
            DebugForcedHp = a.GetInt("hp", 0),
            MaxPolicyDecisions = a.GetLong("max-decisions", 0),
            DecisionCap = a.GetLong("decision-cap", 400_000),
            MaxCombatTurns = a.GetInt("turn-cap", 500),
            ComputeSignatures = a.GetBool("signatures", false),
            RecordActions = true,
            WireRoundTrip = a.GetBool("wire", false),
            TeamVotes = a.GetBool("team-votes", false),
            OfferUndoEndTurn = a.GetBool("undo", true),
            DebugForcedHpPerSeat = a.Get("hp-seats") is string perSeat ? perSeat.Split(',', StringSplitOptions.TrimEntries).Select(int.Parse).ToArray() : null,
        };
    }

    /// <summary>
    /// The seats of run <paramref name="index"/> of a tool (phase H5):
    ///   --lineup A+B+C     the same lineup for every run (2-4 characters; characters may repeat)
    ///   --seats N          N seats; seat k of run i plays characters[(i + k) % characters.Length] (--characters, default all six)
    ///   neither            one seat: characters[i % characters.Length] (unchanged)
    /// </summary>
    public static string[] LineupFor(Args a, string[] characters, int index)
    {
        if (a.Get("lineup") is string lineup)
        {
            return lineup.Split('+', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        }
        int seats = a.GetInt("seats", 1);
        string[] r = new string[seats];
        for (int k = 0; k < seats; k++)
        {
            r[k] = characters[(index + k) % characters.Length];
        }
        return r;
    }

    /// <summary>"IRONCLAD" for one seat, "IRONCLAD+SILENT+..." for several: the lineup as one string (cell labels, episode logs).</summary>
    public static string LineupLabel(EpisodeResult r) => r.Characters.Count > 1 ? string.Join("+", r.Characters) : r.Character;

    /// <summary>Returns true and the exit code when <paramref name="a"/> names one of the run-level commands.</summary>
    public static bool TryRun(HostSession session, Args a, out int exitCode)
    {
        switch (a.Command)
        {
            case "fuzz-runs":
                exitCode = FuzzRuns(session, a);
                return true;
            case "replay-run":
                exitCode = ReplayRun(session, a);
                return true;
            case "determinism":
                exitCode = DeterminismRuns(session, a);
                return true;
            case "driver-info":
                exitCode = DriverInfo(session, a);
                return true;
            case "selftest-driver":
                exitCode = DriverSelfTest.Run(session, a);
                return true;
            case "selftest-mp":
                exitCode = MultiSeatSelfTest.Run(session, a);
                return true;
            default:
                exitCode = 1;
                return false;
        }
    }

    public static bool IsRunLevelCommand(Args a)
    {
        return a.Command is "fuzz-runs" or "replay-run" or "determinism" or "driver-info" or "selftest-driver" or "selftest-mp";
    }

    /// <summary>Adds the module counters of one episode to a running total ("_max" keys: maximum, everything else: sum).</summary>
    public static void AddModuleCounters(SortedDictionary<string, long> total, SortedDictionary<string, long> episode)
    {
        foreach (KeyValuePair<string, long> kv in episode)
        {
            if (kv.Key.EndsWith("_max", StringComparison.Ordinal))
            {
                total[kv.Key] = Math.Max(total.GetValueOrDefault(kv.Key), kv.Value);
            }
            else
            {
                total[kv.Key] = total.GetValueOrDefault(kv.Key) + kv.Value;
            }
        }
    }

    /// <summary>
    /// Coverage table from the summed module counters (gate G4): how often each DecisionKind was a decision point and
    /// each Verb was offered / chosen, and which of them never occurred. DecisionKind.Terminal is not a decision
    /// point (it is the kind of the terminal observation) and is not listed.
    /// </summary>
    public static object Coverage(SortedDictionary<string, long> counters)
    {
        SortedDictionary<string, long> kinds = new SortedDictionary<string, long>(StringComparer.Ordinal);
        SortedDictionary<string, long> kindsPolicy = new SortedDictionary<string, long>(StringComparer.Ordinal);
        SortedDictionary<string, long> chosen = new SortedDictionary<string, long>(StringComparer.Ordinal);
        SortedDictionary<string, long> offered = new SortedDictionary<string, long>(StringComparer.Ordinal);
        List<string> kindsNever = new List<string>();
        List<string> verbsNeverOffered = new List<string>();
        List<string> verbsNeverChosen = new List<string>();
        foreach (DecisionKind k in Enum.GetValues<DecisionKind>())
        {
            if (k == DecisionKind.None || k == DecisionKind.Terminal)
            {
                continue;
            }
            long n = counters.GetValueOrDefault("driver.kind." + k);
            kinds[k.ToString()] = n;
            kindsPolicy[k.ToString()] = counters.GetValueOrDefault("driver.kind_policy." + k);
            if (n == 0)
            {
                kindsNever.Add(k.ToString());
            }
        }
        foreach (Verb v in Enum.GetValues<Verb>())
        {
            if (v == Verb.Invalid)
            {
                continue;
            }
            long c = counters.GetValueOrDefault("driver.verb_chosen." + v);
            long o = counters.GetValueOrDefault("driver.verb_offered." + v);
            chosen[v.ToString()] = c;
            offered[v.ToString()] = o;
            if (o == 0)
            {
                verbsNeverOffered.Add(v.ToString());
            }
            if (c == 0)
            {
                verbsNeverChosen.Add(v.ToString());
            }
        }
        return new
        {
            decision_points_by_kind = kinds,
            policy_decisions_by_kind = kindsPolicy,
            kinds_that_never_occurred = kindsNever,
            decision_points_offering_verb = offered,
            actions_chosen_by_verb = chosen,
            verbs_never_offered = verbsNeverOffered,
            verbs_never_chosen = verbsNeverChosen,
        };
    }

    // -----------------------------------------------------------------------------------------------------
    //  --driver-info
    // -----------------------------------------------------------------------------------------------------

    public static int DriverInfo(HostSession session, Args a)
    {
        HostJson.Emit(new
        {
            kind = "driver-info",
            sts2_sha256 = GameAssemblies.Sts2Sha256,
            strict = session.Options.Strict,
            registry = session.Registry.Describe(),
            modules = session.Registry.Modules,
            canonical_card_classes = session.Registry.CanonicalCardClassesName,
            vocab_hash = session.Registry.VocabHash,
            catalog_hash = session.Registry.CatalogHash,
            vocab_source = session.Registry.VocabSource,
            shared_patch_sites = PatchRegistry.SharedSites(),
            profile = VeteranProfile.Describe(),
            patches = PatchRegistry.Records.Count,
            patches_by_category = PatchRegistry.Records.GroupBy(r => r.Category.ToString()).ToDictionary(g => g.Key, g => g.Count()),
            native_allow_list = session.Options.NativeAllowList.OrderBy(x => x, StringComparer.Ordinal).ToList(),
        }, a.Get("out"));
        return 0;
    }

    // -----------------------------------------------------------------------------------------------------
    //  --fuzz-runs
    // -----------------------------------------------------------------------------------------------------

    private sealed class AbortGroup
    {
        public string Signature = "";
        public string Reason = "";
        public string Detail = "";
        public string Message = "";
        public string? ExceptionType;
        public List<string> Frames = new List<string>();
        public string Diagnostics = "";
        public int Count;
        public List<object> Examples = new List<object>();
    }

    /// <summary>
    /// N whole runs with RandomLegalPolicy over characters x ascensions.
    ///   --runs N  --characters A,B  --ascensions 0,10  --seed-base FUZZ  --policy-seed P  --start-index I
    ///   --seed S (one explicit game seed for every run)  --hp N (TEST ONLY forced HP)  --max-decisions N
    ///   --signatures  --record-dir DIR (episode logs of aborted runs, replayable with --replay-run --log)
    ///   --seconds T (run for T seconds instead of a fixed count)  --out FILE
    /// Exit code 4 when any run was aborted.
    /// </summary>
    public static int FuzzRuns(HostSession session, Args a)
    {
        string[] characters = a.Get("characters", string.Join(",", AllCharacters)).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int[] ascensions = a.Get("ascensions", "0").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        int runs = a.GetInt("runs", 12);
        double seconds = a.GetInt("seconds", 0);
        int startIndex = a.GetInt("start-index", 0);
        string seedBase = a.Get("seed-base", "FUZZ");
        ulong policyBase = a.GetULong("policy-seed", 1);
        string? fixedSeed = a.Get("seed");
        string? recordDir = a.Get("record-dir");
        bool verbose = a.Has("verbose");
        EpisodeOptions options = EpisodeOptionsFrom(a);
        LogBootstrap.ShowErrors = verbose || a.Get("game-log", "off") != "off";

        Dictionary<string, int> outcomes = new Dictionary<string, int> { ["Win"] = 0, ["Loss"] = 0, ["Aborted"] = 0, ["Truncated"] = 0 };
        SortedDictionary<string, int[]> byCell = new SortedDictionary<string, int[]>(StringComparer.Ordinal);
        SortedDictionary<int, int> floorHistogram = new SortedDictionary<int, int>();
        SortedDictionary<int, int> actHistogram = new SortedDictionary<int, int>();
        Dictionary<string, AbortGroup> aborts = new Dictionary<string, AbortGroup>(StringComparer.Ordinal);
        SortedDictionary<string, long> autoSingle = new SortedDictionary<string, long>(StringComparer.Ordinal);
        SortedDictionary<string, long> policyByKind = new SortedDictionary<string, long>(StringComparer.Ordinal);
        SortedDictionary<string, long> nativeCalls = new SortedDictionary<string, long>(StringComparer.Ordinal);
        SortedDictionary<string, long> unexpectedNative = new SortedDictionary<string, long>(StringComparer.Ordinal);
        SortedDictionary<string, int> truncatedBy = new SortedDictionary<string, int>(StringComparer.Ordinal);
        List<object> failures = new List<object>();
        List<string> traceHashes = new List<string>();
        SortedDictionary<string, long> moduleCounters = new SortedDictionary<string, long>(StringComparer.Ordinal);
        TimeSpan cpuStart = Process.GetCurrentProcess().TotalProcessorTime;
        StrictCounters strict = default;
        long decisionPoints = 0, policyDecisions = 0, nested = 0, nestedQueued = 0, autoGold = 0, autoSingleTotal = 0, rooms = 0, combatsWon = 0;
        long pumpCallbacks = 0, virtualTimerFires = 0, maxStack = 0, maxPending = 0, progressSteps = 0, tracked = 0;
        long ticksSettle = 0, ticksBuild = 0, ticksApply = 0, ticksPolicy = 0, ticksSignature = 0, ticksChecks = 0, episodeChecks = 0;
        int teardownFaults = 0, dirty = 0, maxFloor = 0, done = 0;
        long seatsPlayed = 0;
        double episodeMs = 0, startMs = 0, restoreMs = 0, setupMs = 0, actSelMs = 0, enterActMs = 0;
        string? firstTeardownFault = null;

        RandomLegalPolicy policy = new RandomLegalPolicy(0);
        Stopwatch wall = Stopwatch.StartNew();
        for (int n = 0; seconds > 0 ? wall.Elapsed.TotalSeconds < seconds : n < runs; n++)
        {
            int i = startIndex + n;
            int ascension = ascensions[(i / characters.Length) % ascensions.Length];
            string seed = fixedSeed ?? HostSession.SeedFromLabel(seedBase + ":" + i.ToString(CultureInfo.InvariantCulture));
            ulong policySeed = policyBase * 7919UL + (ulong)i;
            policy.Reseed(policySeed);
            EpisodeSpec spec = new EpisodeSpec { Seed = seed, Characters = LineupFor(a, characters, i), Ascension = ascension };
            EpisodeResult r = session.RunEpisode(spec, policy, options);
            done++;
            seatsPlayed += Math.Max(1, r.Characters.Count);

            outcomes[r.Outcome.ToString()]++;
            string cell = LineupLabel(r) + "/A" + r.Ascension;
            if (!byCell.TryGetValue(cell, out int[]? counts))
            {
                counts = new int[4];
                byCell[cell] = counts;
            }
            counts[(int)r.Outcome]++;
            floorHistogram[r.TotalFloor] = floorHistogram.GetValueOrDefault(r.TotalFloor) + 1;
            actHistogram[r.ActIndex] = actHistogram.GetValueOrDefault(r.ActIndex) + 1;
            maxFloor = Math.Max(maxFloor, r.TotalFloor);

            EpisodeCounters c = r.Counters;
            decisionPoints += c.DecisionPoints;
            policyDecisions += c.PolicyDecisions;
            nested += c.NestedDecisions;
            nestedQueued += c.NestedWithQueuedWork;
            autoGold += c.AutoGold;
            autoSingleTotal += c.AutoSingleTotal;
            rooms += c.RoomsEntered;
            combatsWon += c.CombatsWon;
            pumpCallbacks += c.PumpCallbacks;
            virtualTimerFires += c.VirtualTimerFires;
            progressSteps += c.HandlerProgressSteps;
            tracked += c.TrackedOperations;
            ticksSettle += c.TicksSettle;
            ticksBuild += c.TicksBuild;
            ticksApply += c.TicksApply;
            ticksPolicy += c.TicksPolicy;
            ticksSignature += c.TicksSignature;
            ticksChecks += c.TicksChecks;
            episodeChecks += c.EpisodeChecks;
            maxStack = Math.Max(maxStack, c.MaxStackBytes);
            maxPending = Math.Max(maxPending, c.MaxPendingStackDepth);
            for (int k = 0; k < c.AutoSingle.Length; k++)
            {
                if (c.AutoSingle[k] != 0)
                {
                    string key = ((DecisionKind)k).ToString();
                    autoSingle[key] = autoSingle.GetValueOrDefault(key) + c.AutoSingle[k];
                }
                if (c.PolicyByKind[k] != 0)
                {
                    string key = ((DecisionKind)k).ToString();
                    policyByKind[key] = policyByKind.GetValueOrDefault(key) + c.PolicyByKind[k];
                }
            }
            strict = Add(strict, r.Strict);
            AddModuleCounters(moduleCounters, r.ModuleCounters);
            foreach (KeyValuePair<string, long> kv in r.NativeCalls)
            {
                nativeCalls[kv.Key] = nativeCalls.GetValueOrDefault(kv.Key) + kv.Value;
            }
            foreach (KeyValuePair<string, long> kv in r.UnexpectedNativeCalls)
            {
                unexpectedNative[kv.Key] = unexpectedNative.GetValueOrDefault(kv.Key) + kv.Value;
            }
            teardownFaults += r.TeardownFaults;
            firstTeardownFault ??= r.TeardownFault;
            if (!r.Clean)
            {
                dirty++;
            }
            episodeMs += r.WallMs;
            startMs += r.StartMs;
            restoreMs += r.ProfileRestoreMs;
            setupMs += r.RunSetUpMs;
            actSelMs += r.ActSelectionMs;
            enterActMs += r.EnterActMs;
            if (options.ComputeSignatures)
            {
                traceHashes.Add(r.TraceHash.ToString("x16"));
            }
            if (r.Outcome == EpisodeOutcome.Truncated)
            {
                string by = r.TruncatedBy ?? "?";
                truncatedBy[by] = truncatedBy.GetValueOrDefault(by) + 1;
            }

            if (r.Outcome == EpisodeOutcome.Aborted && r.Abort != null)
            {
                AbortInfo info = r.Abort;
                if (!aborts.TryGetValue(info.Signature, out AbortGroup? g))
                {
                    g = new AbortGroup
                    {
                        Signature = info.Signature,
                        Reason = info.Reason.ToString(),
                        Detail = info.Detail,
                        Message = info.Message,
                        ExceptionType = info.ExceptionType,
                        Frames = info.Frames,
                        Diagnostics = info.Diagnostics,
                    };
                    aborts[info.Signature] = g;
                }
                g.Count++;
                object repro = new
                {
                    index = i,
                    seed = r.Seed,
                    character = LineupLabel(r),
                    ascension = r.Ascension,
                    policy_seed = policySeed,
                    hp = options.DebugForcedHp,
                    policy_decisions = c.PolicyDecisions,
                    act = r.ActIndex,
                    total_floor = r.TotalFloor,
                    room = r.FinalRoom,
                };
                if (g.Examples.Count < 5)
                {
                    g.Examples.Add(repro);
                }
                if (failures.Count < 200)
                {
                    failures.Add(new { reason = info.Signature, repro });
                }
                if (recordDir != null)
                {
                    WriteEpisodeLog(Path.Combine(recordDir, $"abort_{i}_{r.Seed}.json"), r, policySeed, options);
                }
            }
            if (verbose)
            {
                Console.Error.WriteLine($"[fuzz] #{i} {LineupLabel(r)} A{r.Ascension} {r.Seed}: {r.Outcome} act {r.ActIndex} floor {r.TotalFloor} decisions {c.PolicyDecisions}+{c.AutoSingleTotal + c.AutoGold} auto {r.WallMs:0} ms hp {string.Join("/", r.FinalHps)} {r.FinalRoom} {(r.Abort != null ? r.Abort.Signature : r.TruncatedBy ?? "")}");
            }
        }
        double sec = wall.Elapsed.TotalSeconds;
        double cpu = (Process.GetCurrentProcess().TotalProcessorTime - cpuStart).TotalSeconds;

        var result = new
        {
            kind = "fuzz-runs",
            vocab_hash = session.Registry.VocabHash,
            modules = session.Registry.Modules,
            runs = done,
            seats = done == 0 ? 0 : Math.Round((double)seatsPlayed / done, 2),
            lineup = a.Get("lineup"),
            wire_round_trip = options.WireRoundTrip,
            start_index = startIndex,
            seed_base = fixedSeed == null ? seedBase : null,
            fixed_seed = fixedSeed,
            policy_seed_base = policyBase,
            characters,
            ascensions,
            options = new { options.AutoSingle, options.AutoGold, options.AllowAbandon, options.DebugForcedHp, options.MaxPolicyDecisions, options.MaxCombatTurns, options.DecisionCap, options.ComputeSignatures },
            strict_mode = session.Options.Strict,
            registry = session.Registry.Describe(),
            outcomes,
            clean_runs = done - dirty,
            by_character_ascension = byCell.ToDictionary(kv => kv.Key, kv => new { win = kv.Value[0], loss = kv.Value[1], aborted = kv.Value[2], truncated = kv.Value[3] }),
            truncated_by = truncatedBy,
            furthest_floor = new
            {
                max = maxFloor,
                mean = done == 0 ? 0 : Math.Round(floorHistogram.Sum(kv => (double)kv.Key * kv.Value) / done, 2),
                histogram = floorHistogram.ToDictionary(kv => kv.Key.ToString(CultureInfo.InvariantCulture), kv => kv.Value),
                final_act_index_histogram = actHistogram.ToDictionary(kv => kv.Key.ToString(CultureInfo.InvariantCulture), kv => kv.Value),
            },
            aborted_by_reason = aborts.Values.OrderByDescending(g => g.Count).Select(g => new
            {
                count = g.Count,
                reason = g.Reason,
                detail = g.Detail,
                exception = g.ExceptionType,
                message = g.Message,
                first_frames = g.Frames,
                diagnostics = g.Diagnostics,
                examples = g.Examples,
            }).ToList(),
            auto_advance = new { single_total = autoSingleTotal, single_by_kind = autoSingle, gold = autoGold },
            decisions = new
            {
                decision_points = decisionPoints,
                policy = policyDecisions,
                policy_by_kind = policyByKind,
                nested = nested,
                nested_with_queued_work = nestedQueued,
                per_run_mean = done == 0 ? 0 : Math.Round((double)policyDecisions / done, 1),
            },
            strict_counters = new
            {
                foreign_posts = strict.ForeignPosts,
                real_time_waits = strict.RealDelays + strict.ForeignWaits,
                deferred_calls_dropped = strict.DeferredCallsDropped,
                late_unobserved_faults = strict.LateUnobservedFaults,
                off_thread_continuations = strict.OffThreadContinuations,
                network_attempts = strict.NetworkAttempts,
                run_save_boundary_not_applied = strict.RunSaveBoundaryNotApplied,
                unexpected_native_calls = unexpectedNative,
                tutorial_room_forced = strict.TutorialRoomForced,
                discovery_boss_changed = strict.DiscoveryBossChanged,
                discovery_rooms_changed = strict.DiscoveryRoomsChanged,
            },
            module_counters = moduleCounters,
            coverage = Coverage(moduleCounters),
            informational_counters = new
            {
                native_calls = nativeCalls,
                deferred_calls = strict.DeferredCalls,
                virtual_delays = strict.VirtualDelays,
                virtual_timer_fires = virtualTimerFires,
                inline_hook_continuations = strict.InlineContinuations,
                max_combat_history_entries_between_decision_points = SpireHost.Patches.LivelockGuard.MaxCombatWorkSeen,
                decision_points_after_more_than_10000_combat_history_entries = SpireHost.Patches.LivelockGuard.CombatWorkAbove10K,
                decision_points_after_more_than_100000_combat_history_entries = SpireHost.Patches.LivelockGuard.CombatWorkAbove100K,
                combat_work_limit = SpireHost.Patches.LivelockGuard.CombatWorkLimit,
                intercepted_progress_writes = strict.InterceptedProgressWrites,
                game_errors_logged = strict.GameErrorsLogged,
                last_game_error = HeadlessPatches.LastGameError == null ? null : HeadlessPatches.LastGameError.Split('\n')[0],
                pump_callbacks = pumpCallbacks,
                handler_progress_steps = progressSteps,
                tracked_operations = tracked,
                rooms_entered = rooms,
                combats_won = combatsWon,
                max_engine_stack_bytes = maxStack,
                max_pending_stack_depth = maxPending,
                teardown_faults = teardownFaults,
                first_teardown_fault = firstTeardownFault,
                metrics_upload_checks = NetworkGuard.MetricsUploadChecks,
                discovery_order_calls = FidelityGatePatches.DiscoveryOrderCalls,
            },
            throughput = new
            {
                wall_seconds = Math.Round(sec, 2),
                // CPU time of this process over the run loop: the load-independent measure (the machine is shared)
                cpu_seconds = Math.Round(cpu, 2),
                runs_per_second = Math.Round(done / sec, 3),
                policy_decisions_per_second = Math.Round(policyDecisions / sec, 1),
                decision_points_per_second = Math.Round(decisionPoints / sec, 1),
                policy_decisions_per_cpu_second = cpu <= 0 ? 0 : Math.Round(policyDecisions / cpu, 1),
                decision_points_per_cpu_second = cpu <= 0 ? 0 : Math.Round(decisionPoints / cpu, 1),
                mean_run_ms = done == 0 ? 0 : Math.Round(episodeMs / done, 1),
                mean_run_start_ms = done == 0 ? 0 : Math.Round(startMs / done, 2),
                mean_profile_restore_ms = done == 0 ? 0 : Math.Round(restoreMs / done, 2),
                mean_unlock_state_and_act_selection_ms = done == 0 ? 0 : Math.Round(actSelMs / done, 2),
                mean_create_and_setup_run_ms = done == 0 ? 0 : Math.Round(setupMs / done, 2),
                mean_finalize_launch_enter_act_ms = done == 0 ? 0 : Math.Round(enterActMs / done, 2),
                time_split_seconds = new
                {
                    episodes = Math.Round(episodeMs / 1000.0, 2),
                    run_start = Math.Round(startMs / 1000.0, 2),
                    apply = Math.Round(ticksApply / (double)Stopwatch.Frequency, 2),
                    settle = Math.Round(ticksSettle / (double)Stopwatch.Frequency, 2),
                    build_decisions = Math.Round(ticksBuild / (double)Stopwatch.Frequency, 2),
                    policy = Math.Round(ticksPolicy / (double)Stopwatch.Frequency, 2),
                    signatures = Math.Round(ticksSignature / (double)Stopwatch.Frequency, 2),
                    episode_checks_inside_settle = Math.Round(ticksChecks / (double)Stopwatch.Frequency, 3),
                },
                episode_checks = episodeChecks,
                episode_check_microseconds_per_decision_point = decisionPoints == 0 ? 0 : Math.Round(ticksChecks * 1e6 / Stopwatch.Frequency / decisionPoints, 2),
                episode_check_nanoseconds_per_check = episodeChecks == 0 ? 0 : Math.Round(ticksChecks * 1e9 / Stopwatch.Frequency / episodeChecks, 0),
                engine_microseconds_per_decision_point = decisionPoints == 0 ? 0 : Math.Round((ticksApply + ticksSettle) * 1e6 / Stopwatch.Frequency / decisionPoints, 1),
                host_microseconds_per_decision_point = decisionPoints == 0 ? 0 : Math.Round((episodeMs * 1000.0 - startMs * 1000.0 - (ticksApply + ticksSettle) * 1e6 / Stopwatch.Frequency) / decisionPoints, 1),
                working_set_mb = Math.Round(Process.GetCurrentProcess().WorkingSet64 / 1048576.0, 1),
                managed_heap_mb = Math.Round(GC.GetTotalMemory(false) / 1048576.0, 1),
            },
            failures,
            trace_hashes = options.ComputeSignatures ? traceHashes : null,
        };
        HostJson.Emit(result, a.Get("out"));
        return outcomes["Aborted"] == 0 ? 0 : 4;
    }

    private static StrictCounters Add(StrictCounters a, StrictCounters b) => a + b;

    // -----------------------------------------------------------------------------------------------------
    //  Episode log (the unit of replay): spec + host options + chosen action indices (+ signatures)
    // -----------------------------------------------------------------------------------------------------

    public sealed class EpisodeLog
    {
        public int format { get; set; } = 1;
        public string sts2_sha256 { get; set; } = "";

        /// <summary>
        /// VOCAB_HASH of the vocabulary loaded when the log was recorded (ruling B1): the canonical order of pile-sourced
        /// CARD_SELECT candidates is the content-key order of the encoder, so action indices are only meaningful with the
        /// same vocabulary. 0 = recorded before the integration / without a vocabulary.
        /// </summary>
        public uint vocab_hash { get; set; }
        public string seed { get; set; } = "";

        /// <summary>The character, or - for a multi-seat run - the lineup in seat order joined with '+' ("IRONCLAD+SILENT").</summary>
        public string character { get; set; } = "";
        public int ascension { get; set; }
        public ulong? policy_seed { get; set; }
        public bool auto_single { get; set; } = true;
        public bool auto_gold { get; set; } = true;
        public bool allow_abandon { get; set; }
        public int hp { get; set; }
        public int turn_cap { get; set; } = 500;
        public string outcome { get; set; } = "";
        public string? abort { get; set; }
        public int total_floor { get; set; }
        public List<int> actions { get; set; } = new List<int>();

        /// <summary>Hex, one per policy decision: state signature ^ decision shape. Empty when the log was recorded without signatures.</summary>
        public List<string> trace { get; set; } = new List<string>();
        public string? trace_hash { get; set; }
    }

    public static EpisodeLog ToLog(EpisodeResult r, ulong? policySeed, EpisodeOptions options)
    {
        return new EpisodeLog
        {
            sts2_sha256 = GameAssemblies.Sts2Sha256,
            vocab_hash = HostSession.Current?.Registry.VocabHash ?? 0,
            seed = r.Seed,
            character = LineupLabel(r),
            ascension = r.RequestedAscension,
            policy_seed = policySeed,
            auto_single = options.AutoSingle,
            auto_gold = options.AutoGold,
            allow_abandon = options.AllowAbandon,
            hp = options.DebugForcedHp,
            turn_cap = options.MaxCombatTurns,
            outcome = r.Outcome.ToString(),
            abort = r.Abort?.Signature,
            total_floor = r.TotalFloor,
            actions = r.ActionLog ?? new List<int>(),
            trace = r.Trace == null ? new List<string>() : r.Trace.Select(x => x.ToString("x16")).ToList(),
            trace_hash = r.TraceHash == 0 ? null : r.TraceHash.ToString("x16"),
        };
    }

    public static void WriteEpisodeLog(string path, EpisodeResult r, ulong? policySeed, EpisodeOptions options)
    {
        string full = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(full, JsonSerializer.Serialize(ToLog(r, policySeed, options), new JsonSerializerOptions { WriteIndented = false }));
    }

    public static EpisodeLog ReadEpisodeLog(string path)
    {
        return JsonSerializer.Deserialize<EpisodeLog>(File.ReadAllText(path)) ?? throw new InvalidDataException("Empty episode log: " + path);
    }

    // -----------------------------------------------------------------------------------------------------
    //  --replay-run
    // -----------------------------------------------------------------------------------------------------

    /// <summary>
    /// Replays recorded action indices with ScriptedPolicy and checks that every per-decision state signature
    /// is reproduced.
    ///   --record FILE --seed S --character C --ascension A [--policy-seed P] [--hp N] [--max-decisions N]
    ///        play one run with RandomLegalPolicy and write its episode log (actions + signatures)
    ///   --log FILE
    ///        replay an episode log; compare signatures (when recorded), outcome and final floor
    ///   --seed S --character C --ascension A --actions 0,3,1,...  [--expect-trace-hash HEX]
    ///        replay an inline action list
    ///   --selfcheck N [--characters ..] [--ascensions ..] [--hp N]
    ///        record N random runs and replay each one in the same process
    /// Exit code 5 on any mismatch.
    /// </summary>
    public static int ReplayRun(HostSession session, Args a)
    {
        LogBootstrap.ShowErrors = a.Has("verbose");
        if (a.Has("selfcheck"))
        {
            return ReplaySelfCheck(session, a);
        }
        EpisodeOptions options = EpisodeOptionsFrom(a);
        options.ComputeSignatures = true;
        options.RecordTrace = true;
        options.RecordActions = true;

        if (a.Get("record") is string recordPath)
        {
            ulong policySeed = a.GetULong("policy-seed", 1);
            EpisodeSpec spec = new EpisodeSpec { Seed = a.Get("seed"), Characters = a.Get("character", "IRONCLAD").Split('+', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries), Ascension = a.GetInt("ascension", 0) };
            EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy(policySeed), options);
            WriteEpisodeLog(recordPath, r, policySeed, options);
            HostJson.Emit(new
            {
                kind = "replay-record",
                log = Path.GetFullPath(recordPath),
                seed = r.Seed,
                character = r.Character,
                ascension = r.Ascension,
                policy_seed = policySeed,
                outcome = r.Outcome.ToString(),
                abort = r.Abort?.Signature,
                total_floor = r.TotalFloor,
                policy_decisions = r.Counters.PolicyDecisions,
                trace_hash = r.TraceHash.ToString("x16"),
                vocab_hash = session.Registry.VocabHash,
            }, a.Get("out"));
            return 0;
        }

        EpisodeLog log;
        bool recorded = false;
        if (a.Get("log") is string logPath)
        {
            log = ReadEpisodeLog(logPath);
            recorded = true;
        }
        else
        {
            if (a.Get("seed") == null || a.Get("actions") == null)
            {
                Console.Error.WriteLine("--replay-run needs --log FILE, or --seed S --character C --ascension A --actions i,j,k, or --record FILE, or --selfcheck N");
                return 1;
            }
            log = new EpisodeLog
            {
                seed = a.Get("seed")!,
                character = a.Get("character", "IRONCLAD"),
                ascension = a.GetInt("ascension", 0),
                auto_single = options.AutoSingle,
                auto_gold = options.AutoGold,
                allow_abandon = options.AllowAbandon,
                hp = options.DebugForcedHp,
                turn_cap = options.MaxCombatTurns,
                actions = a.Get("actions")!.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToList(),
                trace_hash = a.Get("expect-trace-hash"),
            };
        }
        object report = ReplayOne(session, log, out bool ok, recorded);
        HostJson.Emit(report, a.Get("out"));
        return ok ? 0 : 5;
    }

    /// <summary>
    /// Replays one episode log and compares. The log's host options (auto rules, hp) are applied, not the command line's.
    /// <paramref name="recorded"/>: the log is a recording (a file, or a log just recorded in this process). A recording must
    /// say which vocabulary and game build it was made with and both must be this process's - verifier finding R-9: a log
    /// WITHOUT those two fields used to be accepted, although the contract says a log of another vocabulary or game build is
    /// refused (a log that does not say cannot be shown to be of this one). Only the inline form (--seed / --actions on the
    /// command line: an action list, not a recording) has nothing to compare.
    /// </summary>
    public static object ReplayOne(HostSession session, EpisodeLog log, out bool ok, bool recorded = true)
    {
        EpisodeOptions options = new EpisodeOptions
        {
            AutoSingle = log.auto_single,
            AutoGold = log.auto_gold,
            AllowAbandon = log.allow_abandon,
            DebugForcedHp = log.hp,
            MaxCombatTurns = log.turn_cap,
            ComputeSignatures = true,
            RecordTrace = true,
            RecordActions = true,
        };
        EpisodeSpec spec = new EpisodeSpec { Seed = log.seed, Characters = log.character.Split('+', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries), Ascension = log.ascension };
        EpisodeResult r = session.RunEpisode(spec, new ScriptedPolicy(log.actions), options);

        List<ulong> got = r.Trace ?? new List<ulong>();
        int compared = 0, firstMismatch = -1;
        if (log.trace.Count > 0)
        {
            int n = Math.Min(got.Count, log.trace.Count);
            for (int i = 0; i < n; i++)
            {
                compared++;
                if (got[i].ToString("x16") != log.trace[i])
                {
                    firstMismatch = i;
                    break;
                }
            }
            if (firstMismatch < 0 && got.Count < log.trace.Count)
            {
                firstMismatch = got.Count; // the replay stopped before the recording did
            }
        }
        // A recording that ended in a terminal state must replay to the same outcome and floor. A recording that
        // was cut short replays either into the same abort, or to "script exhausted" (Truncated) at the same point.
        bool recordedTerminal = log.outcome is "Win" or "Loss";
        bool outcomeOk;
        if (log.outcome.Length == 0)
        {
            outcomeOk = r.Abort?.Reason != AbortReason.PolicyFault;
        }
        else if (recordedTerminal)
        {
            outcomeOk = r.Outcome.ToString() == log.outcome;
        }
        else
        {
            outcomeOk = r.Outcome == EpisodeOutcome.Truncated || (r.Outcome == EpisodeOutcome.Aborted && r.Abort?.Signature == log.abort);
        }
        bool consumedAll = (r.ActionLog?.Count ?? 0) == log.actions.Count;
        bool hashOk = log.trace_hash == null || !recordedTerminal || log.trace_hash == r.TraceHash.ToString("x16");
        bool floorOk = !recordedTerminal || log.total_floor == r.TotalFloor;
        // A log recorded with another vocabulary (or game build) cannot be replayed faithfully: never "ok".
        bool vocabOk = recorded ? log.vocab_hash != 0 && log.vocab_hash == session.Registry.VocabHash : log.vocab_hash == 0 || log.vocab_hash == session.Registry.VocabHash;
        bool gameOk = recorded ? log.sts2_sha256.Length != 0 && log.sts2_sha256 == GameAssemblies.Sts2Sha256 : log.sts2_sha256.Length == 0 || log.sts2_sha256 == GameAssemblies.Sts2Sha256;
        ok = firstMismatch < 0 && outcomeOk && consumedAll && hashOk && floorOk && vocabOk && gameOk;
        return new
        {
            kind = "replay-run",
            ok,
            seed = r.Seed,
            character = LineupLabel(r),
            ascension = r.Ascension,
            recorded = new { outcome = log.outcome, abort = log.abort, actions = log.actions.Count, signatures = log.trace.Count, total_floor = log.total_floor, trace_hash = log.trace_hash },
            replayed = new
            {
                outcome = r.Outcome.ToString(),
                truncated_by = r.TruncatedBy,
                abort = r.Abort?.Signature,
                abort_message = r.Abort?.Message,
                actions_consumed = r.ActionLog?.Count ?? 0,
                signatures = got.Count,
                total_floor = r.TotalFloor,
                trace_hash = r.TraceHash.ToString("x16"),
            },
            signatures_compared = compared,
            first_signature_mismatch_at_decision = firstMismatch,
            vocab_hash_recorded = log.vocab_hash,
            vocab_hash = session.Registry.VocabHash,
            vocab_hash_matches = vocabOk,
            game_build_matches = gameOk,
            log_names_its_vocabulary_and_game_build = log.vocab_hash != 0 && log.sts2_sha256.Length != 0,
            outcome_matches = outcomeOk,
            all_actions_consumed = consumedAll,
            trace_hash_matches = hashOk,
            wall_ms = Math.Round(r.WallMs, 1),
        };
    }

    private static int ReplaySelfCheck(HostSession session, Args a)
    {
        int n = a.GetInt("selfcheck", 6);
        string[] characters = a.Get("characters", string.Join(",", AllCharacters)).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int[] ascensions = a.Get("ascensions", "0").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        string seedBase = a.Get("seed-base", "REPLAY");
        ulong policyBase = a.GetULong("policy-seed", 3);
        EpisodeOptions options = EpisodeOptionsFrom(a);
        options.ComputeSignatures = true;
        options.RecordTrace = true;
        options.RecordActions = true;
        int okCount = 0;
        long signatures = 0, actions = 0;
        List<object> bad = new List<object>();
        Dictionary<string, int> recordedOutcomes = new Dictionary<string, int>();
        for (int i = 0; i < n; i++)
        {
            int ascension = ascensions[(i / characters.Length) % ascensions.Length];
            ulong policySeed = policyBase * 7919UL + (ulong)i;
            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel(seedBase + ":" + i), Characters = LineupFor(a, characters, i), Ascension = ascension };
            EpisodeResult rec = session.RunEpisode(spec, new RandomLegalPolicy(policySeed), options);
            recordedOutcomes[rec.Outcome.ToString()] = recordedOutcomes.GetValueOrDefault(rec.Outcome.ToString()) + 1;
            EpisodeLog log = ToLog(rec, policySeed, options);
            object report = ReplayOne(session, log, out bool ok);
            signatures += log.trace.Count;
            actions += log.actions.Count;
            if (ok)
            {
                okCount++;
            }
            else if (bad.Count < 10)
            {
                bad.Add(report);
            }
        }
        HostJson.Emit(new
        {
            kind = "replay-selfcheck",
            ok = okCount == n,
            runs = n,
            replayed_identically = okCount,
            recorded_outcomes = recordedOutcomes,
            actions_replayed = actions,
            signatures_compared = signatures,
            mismatches = bad,
        }, a.Get("out"));
        return okCount == n ? 0 : 5;
    }

    // -----------------------------------------------------------------------------------------------------
    //  --determinism --runs N   (run level)
    // -----------------------------------------------------------------------------------------------------

    /// <summary>
    /// Run-level determinism: every run is played twice with the same game seed and the same policy seed and
    /// the per-decision signature traces must be identical; plus two controls (different game seed, different
    /// policy seed) that must differ. Writes per-run trace hashes for cross-process comparison with
    /// bench\host-spike\scripts\compare_determinism.py (same JSON keys as the H0 combat-level check).
    ///   --runs N  --characters ..  --ascensions ..  --seed-base DETRUN  --policy-seed P  --hp N  --max-decisions N
    ///   --controls false  --allow-aborts  --out FILE
    /// Exit code 5 on mismatch or abort (with --allow-aborts: on mismatch only).
    /// </summary>
    public static int DeterminismRuns(HostSession session, Args a)
    {
        LogBootstrap.ShowErrors = a.Has("verbose");
        string[] characters = a.Get("characters", string.Join(",", AllCharacters)).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int[] ascensions = a.Get("ascensions", "0").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        int runs = a.GetInt("runs", 12);
        string seedBase = a.Get("seed-base", "DETRUN");
        ulong policyBase = a.GetULong("policy-seed", 7);
        bool controls = a.GetBool("controls", true);
        EpisodeOptions options = EpisodeOptionsFrom(a);
        options.ComputeSignatures = true;
        options.RecordTrace = true;

        int identical = 0, mismatched = 0, errors = 0, seedDiffers = 0, policyDiffers = 0, policyComparable = 0;
        long signatures = 0, decisions = 0;
        int wins = 0, losses = 0, truncated = 0, maxFloor = 0;
        List<object> mismatches = new List<object>();
        List<string> errorList = new List<string>();
        List<string> traceHashes = new List<string>(runs);
        SortedDictionary<string, long> moduleCounters = new SortedDictionary<string, long>(StringComparer.Ordinal);
        int counterMismatches = 0;
        for (int i = 0; i < runs; i++)
        {
            string[] lineup = LineupFor(a, characters, i);
            string character = string.Join("+", lineup);
            int ascension = ascensions[(i / characters.Length) % ascensions.Length];
            string seed = HostSession.SeedFromLabel(seedBase + ":" + i);
            ulong policySeed = policyBase * 7919UL + (ulong)i;
            EpisodeSpec spec = new EpisodeSpec { Seed = seed, Characters = lineup, Ascension = ascension };

            EpisodeResult ra = session.RunEpisode(spec, new RandomLegalPolicy(policySeed), options);
            EpisodeResult rb = session.RunEpisode(spec, new RandomLegalPolicy(policySeed), options);
            if (ra.Outcome == EpisodeOutcome.Aborted || rb.Outcome == EpisodeOutcome.Aborted)
            {
                // An aborted run must still abort identically (same trace up to the abort, same reason).
                bool same = ra.Outcome == rb.Outcome && ra.Abort?.Signature == rb.Abort?.Signature && FirstDifference(ra.Trace!, rb.Trace!) < 0;
                errors++;
                errorList.Add($"{character}/A{ascension}/{seed}: {(ra.Abort ?? rb.Abort)?.Signature} (aborts identically: {same})");
                traceHashes.Add("error:" + ra.TraceHash.ToString("x16"));
                if (!same)
                {
                    mismatched++;
                }
                continue;
            }
            traceHashes.Add(ra.TraceHash.ToString("x16"));
            AddModuleCounters(moduleCounters, ra.ModuleCounters);
            // The module counters of the two runs must agree as well (they count what the handlers did).
            bool countersEqual = ra.ModuleCounters.Count == rb.ModuleCounters.Count;
            if (countersEqual)
            {
                foreach (KeyValuePair<string, long> kv in ra.ModuleCounters)
                {
                    if (kv.Key.EndsWith("_max", StringComparison.Ordinal))
                    {
                        continue; // process-wide high-water marks
                    }
                    if (!rb.ModuleCounters.TryGetValue(kv.Key, out long other) || other != kv.Value)
                    {
                        countersEqual = false;
                        break;
                    }
                }
            }
            if (!countersEqual)
            {
                counterMismatches++;
            }
            signatures += ra.Trace!.Count;
            decisions += ra.Counters.PolicyDecisions;
            maxFloor = Math.Max(maxFloor, ra.TotalFloor);
            if (ra.Outcome == EpisodeOutcome.Win)
            {
                wins++;
            }
            else if (ra.Outcome == EpisodeOutcome.Loss)
            {
                losses++;
            }
            else
            {
                truncated++;
            }
            int firstDiff = FirstDifference(ra.Trace!, rb.Trace!);
            if (firstDiff < 0 && ra.Outcome == rb.Outcome && ra.TotalFloor == rb.TotalFloor && ra.FinalHp == rb.FinalHp && ra.FinalHps.SequenceEqual(rb.FinalHps) && ra.TraceHash == rb.TraceHash
                && ra.Counters.DecisionPoints == rb.Counters.DecisionPoints)
            {
                identical++;
            }
            else
            {
                mismatched++;
                if (mismatches.Count < 20)
                {
                    mismatches.Add(new { character, ascension, seed, policy_seed = policySeed, first_difference_at_decision = firstDiff, len_a = ra.Trace!.Count, len_b = rb.Trace!.Count, floor_a = ra.TotalFloor, floor_b = rb.TotalFloor });
                }
            }
            if (controls)
            {
                EpisodeSpec other = new EpisodeSpec { Seed = HostSession.SeedFromLabel(seedBase + ":" + i + ":X"), Characters = lineup, Ascension = ascension };
                EpisodeResult rc = session.RunEpisode(other, new RandomLegalPolicy(policySeed), options);
                if (rc.Trace != null && FirstDifference(ra.Trace!, rc.Trace) >= 0)
                {
                    seedDiffers++;
                }
                EpisodeResult rd = session.RunEpisode(spec, new RandomLegalPolicy(policySeed ^ 0x5DEECE66DUL), options);
                if (rd.Trace != null)
                {
                    policyComparable++;
                    if (FirstDifference(ra.Trace!, rd.Trace) >= 0)
                    {
                        policyDiffers++;
                    }
                }
            }
        }
        HostJson.Emit(new
        {
            kind = "determinism-runs",
            seats = a.Get("lineup") is string fixedLineup ? fixedLineup.Split('+').Length : a.GetInt("seats", 1),
            wire_round_trip = options.WireRoundTrip,
            pump = "on",
            toggles = "none",
            level = "run",
            combats = runs, // key kept for compare_determinism.py; at this level the unit is a whole run
            runs,
            characters = characters.Length,
            ascensions,
            seed_base = seedBase,
            policy_seed = policyBase,
            hp = options.DebugForcedHp,
            vocab_hash = session.Registry.VocabHash,
            module_counters = moduleCounters,
            same_seed_same_policy = new { identical, mismatched, errors, signatures_compared = signatures, actions = decisions, wins, losses, truncated, max_total_floor = maxFloor, module_counter_mismatches = counterMismatches },
            different_game_seed = controls ? new { trace_differs = seedDiffers, of = runs - errors } : null,
            different_policy_seed = controls ? new { trace_differs = policyDiffers, of = policyComparable } : null,
            mismatches,
            error_list = errorList.Take(20).ToList(),
            strict = session.ReadStrict(),
            trace_hashes = traceHashes,
        }, a.Get("out"));
        // --allow-aborts: aborted runs (e.g. known open items of another team) do not fail the check as long as
        // they abort identically; a trace mismatch always does.
        return mismatched == 0 && counterMismatches == 0 && (errors == 0 || a.Has("allow-aborts")) ? 0 : 5;
    }

    private static int FirstDifference(List<ulong> a, List<ulong> b)
    {
        int n = Math.Min(a.Count, b.Count);
        for (int i = 0; i < n; i++)
        {
            if (a[i] != b[i])
            {
                return i;
            }
        }
        return a.Count == b.Count ? -1 : n;
    }
}
