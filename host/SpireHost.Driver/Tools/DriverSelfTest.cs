using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Saves;
using SpireHost.Boot;
using SpireHost.Engine;
using SpireHost.Patches;

namespace SpireHost.Driver.Tools;

/// <summary>
/// --selftest-driver: demonstrates, case by case, the definitions of the contract (DRIVER-API.md section 5):
/// decision point, stall (each sub-code), livelock (each bound), strict-mode violations, fault attribution,
/// auto-advance, the abandon filter, nested decisions, truncation, profile restore - and that the session is
/// clean again after every aborted episode. Each case names the expected outcome and what was observed.
/// </summary>
public static class DriverSelfTest
{
    /// <summary>A scripted room handler for the first screen of a run (the Neow event): the test decides what Step / Apply do.</summary>
    private sealed class ProbeHandler : IRoomHandler
    {
        public Func<RunContext, ProbeHandler, HandlerStep>? OnStep;
        public Action<RunContext, Decision, int>? OnApply;
        public int Steps;
        public int Applies;
        public int LastActionCount = -1;
        public int NestedAnswer = -1;

        public string Name => "selftest:probe";

        public HandlerStep Step(RunContext ctx, int seat)
        {
            Steps++;
            return OnStep!(ctx, this);
        }

        public void Apply(RunContext ctx, Decision decision, int actionIndex)
        {
            Applies++;
            LastActionCount = decision.Actions.Count;
            OnApply?.Invoke(ctx, decision, actionIndex);
        }

        public Decision One(Verb verb = Verb.Proceed)
        {
            Decision d = new Decision(DecisionKind.Event, this);
            d.Add(verb, 0);
            return d;
        }
    }

    private sealed class FrameProbe : IRoomFrameExtension
    {
        public int Extended;
        public int Applied;

        public string Name => "selftest:frame-extension";

        public void Extend(RunContext ctx, Decision decision)
        {
            Extended++;
            decision.Add(Verb.DiscardPotion, 0, new GameRef(RefKind.PotionSlot, null, 0));
        }

        public void Apply(RunContext ctx, Decision decision, int actionIndex)
        {
            Applied++;
        }
    }

    /// <summary>Records, for every nested decision the policy answered, how many policy decisions preceded it (= the cap that falls on it).</summary>
    private sealed class NestedOrdinals : IEpisodeObserver
    {
        public readonly List<int> Ordinals = new List<int>();
        private int _policyDecisions;

        public void OnEpisodeStart(RunContext ctx)
        {
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
        {
            if (auto)
            {
                return;
            }
            if (decision.Nested)
            {
                Ordinals.Add(_policyDecisions);
            }
            _policyDecisions++;
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
        }
    }

    private sealed class LambdaPolicy : IPolicy
    {
        private readonly Func<Decision, int> _f;

        public LambdaPolicy(Func<Decision, int> f)
        {
            _f = f;
        }

        public int Decide(Decision decision) => _f(decision);
    }

    public static int Run(HostSession session, Args a)
    {
        LogBootstrap.ShowErrors = a.Has("verbose");
        List<object> cases = new List<object>();
        bool all = true;
        int n = 0;

        EpisodeSpec Spec() => new EpisodeSpec { Seed = HostSession.SeedFromLabel("DRIVERTEST:" + (n++)), Characters = new[] { "IRONCLAD" } };

        void Check(string name, string expected, EpisodeResult r, Func<EpisodeResult, bool> ok, object? extra = null)
        {
            string observed = r.Outcome == EpisodeOutcome.Aborted ? "Aborted(" + r.Abort!.Reason + "/" + r.Abort.Detail + ")"
                : r.Outcome == EpisodeOutcome.Truncated ? "Truncated(" + r.TruncatedBy + ")"
                : r.Outcome.ToString();
            bool pass = ok(r);
            all &= pass;
            cases.Add(new { name, pass, expected, observed, message = r.Abort?.Message, decisions = r.Counters.DecisionPoints, wall_ms = Math.Round(r.WallMs, 1), extra });
        }

        bool Aborted(EpisodeResult r, AbortReason reason, string detailPrefix) =>
            r.Outcome == EpisodeOutcome.Aborted && r.Abort!.Reason == reason && r.Abort.Detail.StartsWith(detailPrefix, StringComparison.Ordinal);

        EpisodeResult WithEventHandler(ProbeHandler probe, IPolicy? policy = null, EpisodeOptions? options = null)
        {
            IRoomHandler original = session.Registry.GetRoomHandler(RoomScreen.Event)!;
            session.Registry.SetRoomHandler(RoomScreen.Event, probe, "selftest");
            try
            {
                return session.RunEpisode(Spec(), policy ?? new RandomLegalPolicy(1), options);
            }
            finally
            {
                session.Registry.SetRoomHandler(RoomScreen.Event, original, "restored");
            }
        }

        // ---- decision points -------------------------------------------------------------------------------
        Check("normal_run", "Loss or Win, not aborted, on the first seed (the production handlers have no known fault to retry around)",
            session.RunEpisode(Spec(), new RandomLegalPolicy(11)),
            r => r.Outcome is EpisodeOutcome.Loss or EpisodeOutcome.Win && r.Counters.PolicyDecisions > 0 && r.Strict.FirstViolation() == null);

        Check("truncate_max_policy_decisions", "Truncated(max_policy_decisions) after exactly 5 policy decisions",
            session.RunEpisode(Spec(), new RandomLegalPolicy(12), new EpisodeOptions { MaxPolicyDecisions = 5 }),
            r => r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy == "max_policy_decisions" && r.Counters.PolicyDecisions == 5);

        Check("truncate_policy_stop", "Truncated(policy_stop: ...)",
            session.RunEpisode(Spec(), new ScriptedPolicy(new[] { 0, 0 })),
            r => r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy!.StartsWith("policy_stop", StringComparison.Ordinal) && r.Counters.PolicyDecisions == 2);

        // ---- faults ----------------------------------------------------------------------------------------
        Check("policy_bad_index", "Aborted(PolicyFault/bad_index)",
            session.RunEpisode(Spec(), new LambdaPolicy(d => d.Actions.Count)),
            r => Aborted(r, AbortReason.PolicyFault, "bad_index"));

        Check("policy_throws", "Aborted(PolicyFault/ask)",
            session.RunEpisode(Spec(), new LambdaPolicy(_ => throw new InvalidOperationException("policy exploded"))),
            r => Aborted(r, AbortReason.PolicyFault, "ask"));

        Check("action_throws", "Aborted(ActionFault/apply:selftest:probe)",
            WithEventHandler(new ProbeHandler { OnStep = (_, p) => HandlerStep.Ask(p.One()), OnApply = (_, _, _) => throw new InvalidOperationException("apply exploded") }),
            r => Aborted(r, AbortReason.ActionFault, "apply"));

        Check("handler_throws", "Aborted(HandlerFault/build:selftest:probe)",
            WithEventHandler(new ProbeHandler { OnStep = (_, _) => throw new InvalidOperationException("step exploded") }),
            r => Aborted(r, AbortReason.HandlerFault, "build"));

        Check("tracked_operation_faults", "Aborted(GameTaskFault/operation:boom)",
            WithEventHandler(new ProbeHandler
            {
                OnStep = (_, p) => HandlerStep.Ask(p.One()),
                OnApply = (ctx, _, _) => ctx.Track(Task.FromException(new InvalidOperationException("op exploded")), "boom"),
            }),
            r => Aborted(r, AbortReason.GameTaskFault, "operation:boom"));

        Check("handler_calls_fail", "Aborted(Stall/custom_detail) - RunContext.Fail keeps its reason",
            WithEventHandler(new ProbeHandler { OnStep = (ctx, _) => { ctx.Stall("custom_detail", "handler says no"); return default; } }),
            r => Aborted(r, AbortReason.Stall, "custom_detail"));

        // ---- stall -----------------------------------------------------------------------------------------
        Check("stall_handler_unrecognised", "Aborted(Stall/handler_unrecognised:selftest:probe)",
            WithEventHandler(new ProbeHandler { OnStep = (_, _) => HandlerStep.Unrecognised("test") }),
            r => Aborted(r, AbortReason.Stall, "handler_unrecognised"));

        Check("stall_no_legal_action", "Aborted(Stall/no_legal_action:Event)",
            WithEventHandler(new ProbeHandler { OnStep = (_, p) => HandlerStep.Ask(new Decision(DecisionKind.Event, p)) }),
            r => Aborted(r, AbortReason.Stall, "no_legal_action"));

        Check("stall_pending_operation", "Aborted(Stall/pending_operation): a blocking operation never completes and nothing asks for input",
            WithEventHandler(new ProbeHandler
            {
                OnStep = (_, p) => HandlerStep.Ask(p.One()),
                OnApply = (ctx, _, _) => ctx.Track(new TaskCompletionSource().Task, "never completes"),
            }),
            r => Aborted(r, AbortReason.Stall, "pending_operation"));

        {
            IRoomHandler original = session.Registry.GetRoomHandler(RoomScreen.Event)!;
            session.Registry.RemoveRoomHandler(RoomScreen.Event);
            EpisodeResult r0;
            try
            {
                r0 = session.RunEpisode(Spec(), new RandomLegalPolicy(1));
            }
            finally
            {
                session.Registry.SetRoomHandler(RoomScreen.Event, original, "restored");
            }
            Check("stall_no_handler", "Aborted(Stall/no_handler:Event): there is no fallback handler", r0, r => Aborted(r, AbortReason.Stall, "no_handler:Event"));
        }

        Check("start_stall", "Aborted(StartFailed/stall:scenario ...): a start-chain task never completes",
            session.RunEpisode(new EpisodeSpec { Seed = HostSession.SeedFromLabel("DRIVERTEST:start"), Scenario = new Scenario { Name = "never", Setup = _ => new TaskCompletionSource().Task } }, new RandomLegalPolicy(1)),
            r => Aborted(r, AbortReason.StartFailed, "stall:"));

        // ---- livelock --------------------------------------------------------------------------------------
        Check("livelock_handler_spin", "Aborted(Livelock/handler_spin:selftest:probe)",
            WithEventHandler(new ProbeHandler { OnStep = (_, _) => HandlerStep.Progressed("again") }),
            r => Aborted(r, AbortReason.Livelock, "handler_spin"));

        Check("livelock_auto_advance_spin", "Aborted(Livelock/auto_advance_spin:Event): a forced frame whose action changes nothing",
            WithEventHandler(new ProbeHandler { OnStep = (_, p) => HandlerStep.Ask(p.One()) }),
            r => Aborted(r, AbortReason.Livelock, "auto_advance_spin"));

        Check("livelock_decision_cap", "Aborted(Livelock/decision_cap)",
            WithEventHandler(new ProbeHandler { OnStep = (_, p) => HandlerStep.Ask(p.One()) }, options: new EpisodeOptions { DecisionCap = 50 }),
            r => Aborted(r, AbortReason.Livelock, "decision_cap"));

        Check("livelock_pump_callbacks", "Aborted(Livelock/pump_callbacks): a continuation that re-posts itself forever",
            WithEventHandler(new ProbeHandler
            {
                OnStep = (_, p) => HandlerStep.Ask(p.One()),
                OnApply = (ctx, _, _) => Repost(ctx.Pump),
            }),
            r => Aborted(r, AbortReason.Livelock, "pump_callbacks"));

        Check("livelock_virtual_time_budget", "Aborted(Livelock/virtual_time_budget): a poll loop on the virtual clock that never ends, in ~0 wall time",
            WithEventHandler(new ProbeHandler
            {
                OnStep = (_, p) => HandlerStep.Ask(p.One()),
                OnApply = (ctx, _, _) => ctx.Track(PollForever(), "poll forever", blocking: false),
            }),
            r => Aborted(r, AbortReason.Livelock, "virtual_time_budget"));

        {
            int saved = LivelockGuard.Limit;
            LivelockGuard.Limit = 1;
            EpisodeResult r0;
            try
            {
                r0 = session.RunEpisode(Spec(), new RandomLegalPolicy(5), new EpisodeOptions { DebugForcedHp = 9999 });
            }
            finally
            {
                LivelockGuard.Limit = saved;
            }
            Check("livelock_turns_without_decision", "Aborted(Livelock/turns_without_decision): with the limit set to 1 the first enemy turn trips the guard", r0,
                r => Aborted(r, AbortReason.Livelock, "turns_without_decision"));
        }

        {
            // Stress verifier finding F1: a bound on the work one game call may do between two decision points. With the limit set to
            // 5 combat-history entries the first fight passes it at once (5 cards are drawn before the first decision); the episode must end as a livelock.
            int saved = LivelockGuard.CombatWorkLimit;
            long fired0 = LivelockGuard.CombatWorkFired;
            LivelockGuard.CombatWorkLimit = 5;
            EpisodeResult r0;
            try
            {
                r0 = session.RunEpisode(Spec(), new LambdaPolicy(d => d.Actions.Count - 1));
            }
            finally
            {
                LivelockGuard.CombatWorkLimit = saved;
            }
            Check("livelock_combat_work_without_decision", "Aborted(Livelock/combat_work_without_decision): with the limit set to 5 history entries the start of the first fight trips the guard", r0,
                r => Aborted(r, AbortReason.Livelock, "combat_work_without_decision") && LivelockGuard.CombatWorkFired - fired0 == 1 && LivelockGuard.CombatWorkLimit == 1_000_000,
                new { fired = LivelockGuard.CombatWorkFired - fired0, limit_restored = LivelockGuard.CombatWorkLimit });
        }

        Check("truncate_combat_turn_cap", "Truncated(combat_turn_cap) with MaxCombatTurns = 2",
            session.RunEpisode(Spec(), new FirstActionPolicy(), new EpisodeOptions { DebugForcedHp = 99999, MaxCombatTurns = 2 }),
            r => r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy == "combat_turn_cap");

        // ---- strict mode -----------------------------------------------------------------------------------
        Check("strict_foreign_post", "Aborted(StrictViolation/foreign_post): another thread posted to the pump; the callback was rejected, never run",
            session.RunEpisode(Spec(), new LambdaPolicy(d =>
            {
                EnginePump pump = session.Pump;
                Task.Run(() => pump.Post(_ => ForeignCallbackRan++, null)).Wait();
                return 0;
            })),
            r => Aborted(r, AbortReason.StrictViolation, "foreign_post") && ForeignCallbackRan == 0,
            new { foreign_callback_executed = ForeignCallbackRan });

        Check("strict_real_time_wait", "Aborted(StrictViolation/real_time_wait): a game delay was created off the engine thread (real timer)",
            session.RunEpisode(Spec(), new LambdaPolicy(d =>
            {
                Task.Run(() => SchedulingPatches.Delay(1)).Wait();
                return 0;
            })),
            r => Aborted(r, AbortReason.StrictViolation, "real_time_wait"));

        Check("strict_native_call", "Aborted(StrictViolation/native_call:...): a Godot native entry point outside the allow-list was reached",
            session.RunEpisode(Spec(), new LambdaPolicy(d =>
            {
                NativeProbe.Touch();
                return 0;
            }), new EpisodeOptions { MaxPolicyDecisions = 3 }),
            r => Aborted(r, AbortReason.StrictViolation, "native_call"),
            new { note = "detected at the end of the episode (the native counters are compared once per episode)" });

        // ---- central rules ---------------------------------------------------------------------------------
        {
            ProbeHandler probe = new ProbeHandler();
            probe.OnStep = (_, p) =>
            {
                if (p.Applies > 0)
                {
                    return HandlerStep.Unrecognised("done");
                }
                Decision d = new Decision(DecisionKind.Confirm, p);
                d.Add(Verb.ChooseOption, 0, GameRef.OfConfirm(false));
                d.Add(Verb.ChooseOption, 0, GameRef.OfConfirm(true), flags: ActionFlags.Abandon);
                return HandlerStep.Ask(d);
            };
            EpisodeResult r0 = WithEventHandler(probe);
            Check("abandon_action_filtered", "the abandon-class action is removed; the remaining single action is auto-advanced (auto_single[Confirm] = 1)", r0,
                r => probe.LastActionCount == 1 && r.Counters.AutoSingle[(int)DecisionKind.Confirm] == 1 && r.Counters.PolicyDecisions == 0,
                new { actions_seen = probe.LastActionCount });

            ProbeHandler probe2 = new ProbeHandler();
            probe2.OnStep = probe.OnStep;
            EpisodeResult r1 = WithEventHandler(probe2, options: new EpisodeOptions { AllowAbandon = true });
            Check("abandon_action_allowed", "with AllowAbandon both actions are offered and the policy is asked", r1,
                r => probe2.LastActionCount == 2 && r.Counters.PolicyDecisions == 1, new { actions_seen = probe2.LastActionCount });
        }

        {
            ProbeHandler probe = new ProbeHandler();
            probe.OnStep = (_, p) => p.Applies > 0 ? HandlerStep.Unrecognised("done") : HandlerStep.Ask(p.One());
            probe.OnApply = (ctx, _, _) =>
            {
                // what a synchronous game callback does: build a decision, ask, turn the index into a return value
                Decision nested = new Decision(DecisionKind.CardReward, null);
                nested.Add(Verb.Skip, 0);
                nested.Add(Verb.PickOfferCard, 0, new GameRef(RefKind.Card, null, 1));
                nested.Add(Verb.PickOfferCard, 0, new GameRef(RefKind.Card, null, 0));
                probe.NestedAnswer = ctx.DecideNested(nested);
                probe.LastActionCount = nested.Actions[0].Verb == Verb.Skip && nested.Actions[1].Source.Index == 0 ? 3 : -3; // canonical: SelectCard..Skip(10) < PickOfferCard(12), ordinals ascending
            };
            EpisodeResult r0 = WithEventHandler(probe, new LambdaPolicy(d => d.Actions.Count - 1));
            Check("nested_decision", "a nested decision is asked without pumping, counted, canonically ordered, and its index comes back", r0,
                r => r.Counters.NestedDecisions == 1 && probe.NestedAnswer == 2 && probe.LastActionCount == 3 && r.Counters.NestedWithQueuedWork == 0,
                new { answer = probe.NestedAnswer });
        }

        {
            // Verifier finding R-1: the decision cap also holds at a nested decision. The outer decision is policy decision 1 = the cap;
            // its Apply asks a nested decision, which must not reach the policy.
            int asked = 0;
            ProbeHandler probe = new ProbeHandler();
            probe.OnStep = (_, p) => p.Applies > 0 ? HandlerStep.Unrecognised("done") : HandlerStep.Ask(p.One());
            probe.OnApply = (ctx, _, _) =>
            {
                Decision nested = new Decision(DecisionKind.CardReward, null);
                nested.Add(Verb.Skip, 0);
                nested.Add(Verb.PickOfferCard, 0, new GameRef(RefKind.Card, null, 0));
                probe.NestedAnswer = ctx.DecideNested(nested);
            };
            probe.NestedAnswer = -7;
            EpisodeResult r0 = WithEventHandler(probe, new LambdaPolicy(d => { asked++; return 0; }), new EpisodeOptions { MaxPolicyDecisions = 1, AutoSingle = false });
            Check("truncate_max_policy_decisions_at_nested_decision", "Truncated(max_policy_decisions): the nested decision is not asked, not counted, and exactly 1 policy decision was made", r0,
                r => r.Outcome == EpisodeOutcome.Truncated && r.TruncatedBy == "max_policy_decisions" && r.Counters.PolicyDecisions == 1 && asked == 1 && probe.NestedAnswer == -7
                     && r.Counters.NestedDecisions == 0 && r.Counters.DecisionPoints == 1,
                new { policy_asked = asked, nested_answer = probe.NestedAnswer });
        }

        {
            // R-1 on the real path (the reviewer's tuple): the card reward that a REWARD pick opens is asked from inside the game's
            // reward task. For every cap N below the uncapped length the episode must make exactly N policy decisions; the caps that
            // fall on a nested decision are counted, and there must be some.
            EpisodeSpec capSpec() => new EpisodeSpec { Seed = "REVIEWCAP1", Characters = new[] { "IRONCLAD" } };
            NestedOrdinals nestedAt = new NestedOrdinals();
            EpisodeResult full = session.RunEpisode(capSpec(), new RandomLegalPolicy(11), new EpisodeOptions(), nestedAt);
            int total = (int)full.Counters.PolicyDecisions;
            List<int> wrong = new List<int>();
            for (int cap = 1; cap < total; cap++)
            {
                EpisodeResult rc = session.RunEpisode(capSpec(), new RandomLegalPolicy(11), new EpisodeOptions { MaxPolicyDecisions = cap });
                if (rc.Outcome != EpisodeOutcome.Truncated || rc.TruncatedBy != "max_policy_decisions" || rc.Counters.PolicyDecisions != cap)
                {
                    wrong.Add(cap);
                }
            }
            Check("max_policy_decisions_is_exact_for_every_cap", "every cap N < the uncapped length ends Truncated(max_policy_decisions) after exactly N policy decisions, including the caps that fall on a nested card reward", full,
                r => r.Outcome is EpisodeOutcome.Loss or EpisodeOutcome.Win && total > 20 && wrong.Count == 0 && nestedAt.Ordinals.Count > 0,
                new { uncapped_policy_decisions = total, caps_tested = total - 1, caps_that_fall_on_a_nested_decision = nestedAt.Ordinals, caps_with_another_count = wrong });
        }

        {
            // A room-frame extension adds an action to a frame it does not own (the potion pattern, B.7).
            FrameProbe ext = new FrameProbe();
            session.Registry.SetFrameExtension(ext);
            ProbeHandler probe = new ProbeHandler();
            probe.OnStep = (_, p) => p.Applies + ext.Applied > 0 ? HandlerStep.Unrecognised("done") : HandlerStep.Ask(p.One());
            EpisodeResult r0;
            try
            {
                r0 = WithEventHandler(probe, new LambdaPolicy(d => 0)); // canonical order: DISCARD_POTION (5) sorts before PROCEED (13)
            }
            finally
            {
                session.Registry.RemoveFrameExtension(ext.Name);
            }
            Check("room_frame_extension", "the extension's action is offered next to the handler's (2 actions, canonical order) and applied by the extension, not by the frame's source", r0,
                r => ext.Extended >= 1 && ext.Applied == 1 && probe.Applies == 0 && r.Counters.PolicyDecisions == 1 && r.Counters.AutoSingleTotal == 0,
                new { extended = ext.Extended, applied_by_extension = ext.Applied, applied_by_source = probe.Applies });
        }

        {
            // Host async code that waits for the engine with RunContext.Quiescence().
            int stage = 0;
            ProbeHandler probe = new ProbeHandler();
            probe.OnStep = (_, p) => p.Applies > 0 ? HandlerStep.Unrecognised("done after stage " + stage) : HandlerStep.Ask(p.One());
            probe.OnApply = (ctx, _, _) => ctx.Post(async () =>
            {
                stage = 1;
                await ctx.Quiescence();
                stage = 2;
                await ctx.Quiescence();
                stage = 3;
            }, "two quiescence awaits");
            EpisodeResult r0 = WithEventHandler(probe);
            Check("await_quiescence", "posted host code resumes at each quiescent point (stage 3) before the next handler step", r0,
                r => stage == 3 && Aborted(r, AbortReason.Stall, "handler_unrecognised"), new { stage });
        }

        // ---- no network IO ---------------------------------------------------------------------------------
        {
            // The two outbound HTTP entry points of sts2.dll that game logic can reach are blocked (Core NetworkGuard). The
            // probe calls them directly with a host under the reserved TLD .invalid (RFC 2606: never resolves), so that even a
            // broken guard could not contact anyone. Outside an episode: the strict counter is a per-episode delta.
            System.Reflection.Assembly game = typeof(MegaCrit.Sts2.Core.Runs.RunManager).Assembly;
            const System.Reflection.BindingFlags any = System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
            long before = NetworkGuard.NetworkAttempts;
            long checksBefore = NetworkGuard.MetricsUploadChecks;
            object? put = game.GetType("MegaCrit.Sts2.Core.Runs.Metrics.MetricUtilities", true)!.GetMethod("PutRequest", any)!
                .Invoke(null, new object[] { "https://spirehost-selftest.invalid/record_data/", "{}", "selftest" });
            object? time = game.GetType("MegaCrit.Sts2.Core.Daily.TimeServer", true)!.GetMethod("RequestTime", any)!
                .Invoke(null, new object[] { "https://spirehost-selftest.invalid" });
            object? should = game.GetType("MegaCrit.Sts2.Core.Runs.Metrics.MetricUtilities", true)!.GetMethod("ShouldUploadMetrics", any)!
                .Invoke(null, new object[] { true });
            bool putBlocked = put is Task pt && pt.IsCompletedSuccessfully;
            bool timeBlocked = time is Task tt && tt.IsCompletedSuccessfully;
            long attempts = NetworkGuard.NetworkAttempts - before;
            EpisodeResult r0 = session.RunEpisode(Spec(), new RandomLegalPolicy(21), new EpisodeOptions { MaxPolicyDecisions = 3 });
            Check("network_guard", "PutRequest and RequestTime never run (completed task, 2 attempts counted), ShouldUploadMetrics is false, and an episode after it has network_attempts 0", r0,
                r => putBlocked && timeBlocked && attempts == 2 && should is false && NetworkGuard.MetricsUploadChecks - checksBefore == 1 && r.Strict.NetworkAttempts == 0 && r.Outcome == EpisodeOutcome.Truncated,
                new { put_blocked = putBlocked, time_blocked = timeBlocked, attempts_counted = attempts, should_upload = should, last = NetworkGuard.LastAttempt });
        }

        {
            // A network attempt INSIDE an episode is a strict violation.
            EpisodeResult r0 = session.RunEpisode(Spec(), new LambdaPolicy(d =>
            {
                typeof(MegaCrit.Sts2.Core.Runs.RunManager).Assembly.GetType("MegaCrit.Sts2.Core.Runs.Metrics.MetricUtilities", true)!
                    .GetMethod("PutRequest", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!
                    .Invoke(null, new object[] { "https://spirehost-selftest.invalid/record_data/", "{}", "selftest" });
                return 0;
            }));
            Check("strict_network_attempt", "Aborted(StrictViolation/network_attempt): game code reached an outbound HTTP entry point (blocked, counted)", r0,
                r => Aborted(r, AbortReason.StrictViolation, "network_attempt"));
        }

        {
            // Verifier finding R-13: a SaveManager.SaveRun call that the run-save boundary cannot be applied to (here: the patch's
            // own postfix entered from another thread, as game code saving off the engine thread would enter it) is a strict violation.
            EpisodeResult r0 = session.RunEpisode(Spec(), new LambdaPolicy(d =>
            {
                System.Threading.Thread t = new System.Threading.Thread(() =>
                {
                    Task save = Task.CompletedTask;
                    SchedulingPatches.RunSavePostfix(null, true, ref save);
                });
                t.Start();
                t.Join();
                return 0;
            }));
            Check("strict_run_save_boundary_skipped", "Aborted(StrictViolation/run_save_boundary_skipped): a run save outside the engine thread cannot get the pump-turn boundary; never silent", r0,
                r => Aborted(r, AbortReason.StrictViolation, "run_save_boundary_skipped"));
        }

        // ---- profile ---------------------------------------------------------------------------------------
        {
            ProgressState before = SaveManager.Instance.Progress;
            before.WongoPoints = 123456;                       // simulate drift left behind by a run
            string seed = HostSession.SeedFromLabel("DRIVERTEST:profile");
            int seenWongo = -1;
            ProgressState? during = null;
            ProbeHandler probe = new ProbeHandler();
            probe.OnStep = (_, _) =>
            {
                during = SaveManager.Instance.Progress;
                seenWongo = during.WongoPoints;
                return HandlerStep.Unrecognised("done");
            };
            IRoomHandler original = session.Registry.GetRoomHandler(RoomScreen.Event)!;
            session.Registry.SetRoomHandler(RoomScreen.Event, probe, "selftest");
            EpisodeResult r0;
            try
            {
                r0 = session.RunEpisode(new EpisodeSpec { Seed = seed }, new RandomLegalPolicy(1));
            }
            finally
            {
                session.Registry.SetRoomHandler(RoomScreen.Event, original, "restored");
            }
            bool invariants = true;
            try
            {
                VeteranProfile.AssertInvariants(during!);
            }
            catch (Exception)
            {
                invariants = false;
            }
            Check("profile_restored_at_reset", "Progress is a fresh object from the snapshot, invariants hold, WongoPoints = f(run seed) in [0, 1999]", r0,
                r => during != null && !ReferenceEquals(during, before) && invariants && seenWongo == r.WongoPoints && seenWongo == VeteranProfile.WongoPointsFor(seed) && seenWongo is >= 0 and < 2000,
                new { wongo_points = seenWongo, invariants });
        }

        {
            // Verifier finding R-10: EnginePump.DrainUntil (the verification tools' own pump entry) is refused while an episode runs.
            bool threw = false;
            EpisodeResult r0 = session.RunEpisode(Spec(), new LambdaPolicy(d =>
            {
                try
                {
                    session.Pump.DrainUntil(() => true, 10, TimeSpan.FromSeconds(1));
                }
                catch (InvalidOperationException)
                {
                    threw = true;
                }
                return 0;
            }), new EpisodeOptions { MaxPolicyDecisions = 1 });
            bool afterwards = session.Pump.DrainUntil(() => true, 10, TimeSpan.FromSeconds(1)) == PumpStop.Decision;
            Check("drain_until_is_refused_during_an_episode", "EnginePump.DrainUntil throws inside an episode and works again after it", r0,
                r => threw && afterwards && r.Outcome == EpisodeOutcome.Truncated, new { threw, works_after_the_episode = afterwards });
        }

        // ---- decision contexts ------------------------------------------------------------------------------
        {
            // Verifier finding R-14: the event context shows an option as its button shows it. Two options of the live Neow event with
            // the same two hover tips and the kill predicate true - one clickable, one locked (no OnChosen: EventOption.IsLocked).
            int tipsUnlocked = -1, tipsLocked = -1;
            bool killUnlocked = false, killLocked = true, lockedFlag = false;
            ProbeHandler probe = new ProbeHandler();
            probe.OnStep = (ctx, p) =>
            {
                MegaCrit.Sts2.Core.Models.EventModel ev = ctx.RunManager.EventSynchronizer.GetLocalEvent()!;
                MegaCrit.Sts2.Core.Events.EventOption src = ev.CurrentOptions[0];
                List<MegaCrit.Sts2.Core.HoverTips.IHoverTip> tips = new List<MegaCrit.Sts2.Core.HoverTips.IHoverTip>
                {
                    MegaCrit.Sts2.Core.HoverTips.HoverTipFactory.FromCard(System.Linq.Enumerable.First(MegaCrit.Sts2.Core.Models.ModelDb.AllCards)),
                    MegaCrit.Sts2.Core.HoverTips.HoverTipFactory.FromPotion(System.Linq.Enumerable.First(MegaCrit.Sts2.Core.Models.ModelDb.AllPotions)),
                };
                MegaCrit.Sts2.Core.Events.EventOption clickable = new MegaCrit.Sts2.Core.Events.EventOption(ev, () => Task.CompletedTask, src.Title, src.Description, src.TextKey, tips);
                MegaCrit.Sts2.Core.Events.EventOption locked = new MegaCrit.Sts2.Core.Events.EventOption(ev, null, src.Title, src.Description, src.TextKey, tips);
                EventContext c = new EventContext();
                c.BeginPage(ev);
                EventOptionInfo a0 = c.AddOption(clickable, 0, willKillPlayer: true);
                EventOptionInfo a1 = c.AddOption(locked, 1, willKillPlayer: true);
                tipsUnlocked = a0.Cards.Count + a0.Models.Count;
                tipsLocked = a1.Cards.Count + a1.Models.Count;
                killUnlocked = a0.WillKillPlayer;
                killLocked = a1.WillKillPlayer;
                lockedFlag = a1.IsLocked && !a0.IsLocked;
                return HandlerStep.Unrecognised("done");
            };
            EpisodeResult r0 = WithEventHandler(probe);
            Check("locked_event_option_shows_no_hover_tips_and_no_kill_warning", "a clickable option carries its 2 hover-tip entities and the kill warning; the same option locked carries neither", r0,
                r => lockedFlag && tipsUnlocked == 2 && killUnlocked && tipsLocked == 0 && !killLocked,
                new { clickable = new { hover_tip_entities = tipsUnlocked, kill_warning = killUnlocked }, locked = new { hover_tip_entities = tipsLocked, kill_warning = killLocked } });
        }

        // ---- run-level replay logs --------------------------------------------------------------------------
        {
            // Verifier finding R-9: a recorded log must name the vocabulary and the game build it was made with, and both must be ours.
            EpisodeOptions ro = new EpisodeOptions { ComputeSignatures = true, RecordTrace = true, RecordActions = true, MaxPolicyDecisions = 12 };
            EpisodeResult rec = session.RunEpisode(Spec(), new RandomLegalPolicy(31), ro);
            DriverCommands.EpisodeLog Log(Action<DriverCommands.EpisodeLog>? change = null)
            {
                DriverCommands.EpisodeLog l = DriverCommands.ToLog(rec, 31, ro);
                change?.Invoke(l);
                return l;
            }
            Dictionary<string, bool> got = new Dictionary<string, bool>();
            void Try(string name, bool recorded, Action<DriverCommands.EpisodeLog>? change)
            {
                DriverCommands.ReplayOne(session, Log(change), out bool okReplay, recorded);
                got[name] = okReplay;
            }
            Try("as recorded", true, null);
            Try("without vocab_hash", true, l => l.vocab_hash = 0);
            Try("without sts2_sha256", true, l => l.sts2_sha256 = "");
            Try("another vocab_hash", true, l => l.vocab_hash ^= 0x5555);
            Try("another sts2_sha256", true, l => l.sts2_sha256 = new string('0', 64));
            Try("inline action list (no recording)", false, l => { l.vocab_hash = 0; l.sts2_sha256 = ""; });
            bool pass = got["as recorded"] && !got["without vocab_hash"] && !got["without sts2_sha256"] && !got["another vocab_hash"] && !got["another sts2_sha256"] && got["inline action list (no recording)"];
            all &= pass;
            cases.Add(new { name = "replay_log_must_name_its_vocabulary_and_game_build", pass, expected = "a recorded log replays; the same log without / with another vocab_hash or sts2_sha256 is refused; an inline action list has nothing to compare",
                observed = string.Join(", ", got.Select(kv => kv.Key + ": " + (kv.Value ? "ok" : "refused"))), message = (string?)null, decisions = rec.Counters.DecisionPoints, wall_ms = Math.Round(rec.WallMs, 1), extra = (object?)null });
        }

        // ---- tool summaries --------------------------------------------------------------------------------
        {
            // Verifier finding R-2: every field of StrictCounters takes part in the sum and in the difference.
            List<string> missing = new List<string>();
            System.Reflection.FieldInfo[] fields = typeof(StrictCounters).GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            foreach (System.Reflection.FieldInfo f in fields)
            {
                object x = new StrictCounters(), y = new StrictCounters();
                f.SetValue(x, 7L);
                f.SetValue(y, 4L);
                StrictCounters sum = (StrictCounters)x + (StrictCounters)y;
                StrictCounters diff = (StrictCounters)x - (StrictCounters)y;
                if ((long)f.GetValue(sum)! != 11L || (long)f.GetValue(diff)! != 3L)
                {
                    missing.Add(f.Name);
                }
            }
            bool pass = fields.Length >= 15 && missing.Count == 0;
            all &= pass;
            cases.Add(new { name = "strict_counters_sum_covers_every_field", pass, expected = "operator + and operator - of StrictCounters include every field (the fuzz summary's totals)",
                observed = missing.Count == 0 ? fields.Length + " fields, all summed and subtracted" : "not covered: " + string.Join(",", missing), message = (string?)null, decisions = 0L, wall_ms = 0.0, extra = (object?)null });
        }

        // ---- the session is clean after all of the above ------------------------------------------------------
        {
            EpisodeResult a1 = session.RunEpisode(new EpisodeSpec { Seed = "CLEANCHECK01", Characters = new[] { "SILENT" } }, new RandomLegalPolicy(77), new EpisodeOptions { ComputeSignatures = true });
            EpisodeResult a2 = session.RunEpisode(new EpisodeSpec { Seed = "CLEANCHECK01", Characters = new[] { "SILENT" } }, new RandomLegalPolicy(77), new EpisodeOptions { ComputeSignatures = true });
            Check("clean_after_aborts", "after every abort above, a normal run completes, has no strict violation and is reproducible", a2,
                r => a1.Outcome is EpisodeOutcome.Loss or EpisodeOutcome.Win && a1.Outcome == a2.Outcome && a1.TraceHash == a2.TraceHash && a1.TraceHash != 0
                     && a1.Strict.FirstViolation() == null && a2.Strict.FirstViolation() == null && a2.UnexpectedNativeCalls.Count == 0,
                new { trace_a = a1.TraceHash.ToString("x16"), trace_b = a2.TraceHash.ToString("x16") });
        }

        HostJson.Emit(new { kind = "selftest-driver", ok = all, cases_total = cases.Count, cases }, a.Get("out"));
        return all ? 0 : 4;
    }

    private static int ForeignCallbackRan;

    private static void Repost(EnginePump pump)
    {
        pump.Post(static s => Repost((EnginePump)s!), pump);
    }

    private static async Task PollForever()
    {
        while (true)
        {
            await SchedulingPatches.Delay(50);
        }
    }

    /// <summary>Reaches one Godot native entry point that is not on the allow-list (GD.Randi: no object, no finalizer).</summary>
    private static class NativeProbe
    {
        public static void Touch()
        {
            try
            {
                _ = Godot.GD.Randi();
            }
            catch (Exception)
            {
                // the thunk returns zero; whatever GodotSharp makes of that is irrelevant - the call was counted
            }
        }
    }
}
