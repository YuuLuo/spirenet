using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Boot;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Protocol;
using SpireHost.Obs.Schema;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs.Tests;

/// <summary>
/// obs --selftest [--runs N] [--quick] [--cases a,b,...]: the observation module's gate, on the integrated host (every
/// decision comes from the real handlers of the production module set).
///   vocab        canonical JSON round trip; the loaded vocabulary knows every name of the running game (no UNK);
///                the append-only merge is a no-op on it and appends / retires correctly on a doctored older file
///   keys         O7: every ConfirmContext.Key of the driver's ConfirmKeys list, the two button names and the two fixed
///                selection headers are vocabulary names
///   audit        IL scan of the encoder namespace against the schema 14.3 blacklist
///   serve_args   a server's command line: the allow-list, and switches written with a value (verifier finding R-3)
///   fuzz         N whole random runs (half with --hp 99999 and a decision budget), every decision (policy + auto)
///                encoded twice: FrameCheck problems = 0, side effects (DeepSignature before/after) = 0, encode twice =
///                identical bytes, ACTIONS rows == decision actions, synthetic CARD_SELECT prompts (deck / hand upgrade
///                previews, deduplicated draw-pile classes) on the live state
///   scenarios    shop / rest site / treasure / elite / boss rooms through the debug jumps, same checks
///   protocol     the binary server driven in-process (ScriptTransport): every rule of errata section P, and ruling A4:
///                a RESET with a debug option (word 15) or an undefined flag bit is refused by a production server and
///                honoured (forced HP visible in the frame) only by a server started with --allow-debug-options
///   special      synthetic CONFIRM / BUNDLE / CRYSTAL frames in controlled states (P11 + positive control)
///   real_special the REAL CRYSTAL / BUNDLE / CONFIRM frames of the events module through the full encode checker
///   enemy_turn   D8: a card choice raised inside the enemy turn (KnowledgeDemon) hides the intents of the enemies that
///                already began their move and keeps the others
///   sel_count    CARD_SELECT sel_count of merged classes equals the frame's own pick counts
///   leak         the hidden-information perturbation suite (LeakTests) on real frames
/// Exit 0 when every case passed, 4 otherwise.
/// </summary>
public static class SelfTest
{
    private sealed class Case
    {
        public string Name = "";
        public bool Ok;
        public object? Detail;
    }

    public static int Run(HostSession session, Args a)
    {
        Stopwatch wall = Stopwatch.StartNew();
        ObsRuntime.Loaded l = ObsModule.Current; // the ONE vocabulary of the process (loaded at boot, ruling B1)
        l.Vocab.UnknownLog.Clear();
        List<Case> cases = new List<Case>();
        bool quick = a.Has("quick");
        string[] all = { "vocab", "keys", "audit", "serve_args", "fuzz", "scenarios", "protocol", "special", "real_special", "enemy_turn", "sel_count", "leak" };
        // Phase H5: the multi-seat cases are opt-in (--cases frames_mp,protocol_mp,leak_team); the default list is the H1 list.
        string[] multiSeat = { "frames_mp", "protocol_mp", "leak_team" };
        string[] only = (a.Get("cases") ?? string.Join(",", all)).Split(',', StringSplitOptions.RemoveEmptyEntries);
        foreach (string n in only)
        {
            if (Array.IndexOf(all, n) < 0 && Array.IndexOf(multiSeat, n) < 0)
            {
                Console.Error.WriteLine("obs --selftest: unknown case '" + n + "' (cases: " + string.Join(",", all) + "; multi-seat: " + string.Join(",", multiSeat) + ")");
                return 1;
            }
        }
        bool Want(string n) => Array.IndexOf(only, n) >= 0;
        if (Want("vocab"))
        {
            cases.Add(VocabCase(l));
        }
        if (Want("keys"))
        {
            object keys = IntegrationCases.ConfirmAndPromptKeys(l, out bool keysOk);
            cases.Add(new Case { Name = "keys", Ok = keysOk, Detail = keys });
        }
        if (Want("audit"))
        {
            cases.Add(AuditCase());
        }
        if (Want("serve_args"))
        {
            cases.Add(ServeArgsCase());
        }
        if (Want("fuzz"))
        {
            cases.Add(FuzzCase(session, l, a.GetInt("runs", quick ? 6 : 24)));
        }
        if (Want("scenarios"))
        {
            cases.Add(ScenarioCase(session, l));
        }
        if (Want("protocol"))
        {
            cases.Add(ProtocolCase(session, l));
        }
        if (Want("special"))
        {
            object sp = SpecialFrames.Run(session, l, null, out bool spOk);
            cases.Add(new Case { Name = "special", Ok = spOk, Detail = sp });
        }
        if (Want("real_special"))
        {
            object rs = IntegrationCases.RealSpecialFrames(session, l, out bool rsOk);
            cases.Add(new Case { Name = "real_special", Ok = rsOk, Detail = rs });
        }
        if (Want("enemy_turn"))
        {
            object et = IntegrationCases.EnemyTurnPrompt(session, l, out bool etOk);
            cases.Add(new Case { Name = "enemy_turn", Ok = etOk, Detail = et });
        }
        if (Want("sel_count"))
        {
            object sc = IntegrationCases.MergedSelCount(session, l, out bool scOk);
            cases.Add(new Case { Name = "sel_count", Ok = scOk, Detail = sc });
        }
        if (Want("leak"))
        {
            object leak = LeakTests.Execute(session, l, quick ? 2 : 6, quick ? 9 : 5, 99999, out bool leakOk);
            cases.Add(new Case { Name = "leak", Ok = leakOk, Detail = leak });
        }
        if (Want("frames_mp"))
        {
            object frames = MultiSeatFrames.FramesMp(session, l, a.GetInt("runs", quick ? 3 : 6), out bool framesOk);
            cases.Add(new Case { Name = "frames_mp", Ok = framesOk, Detail = frames });
        }
        if (Want("protocol_mp"))
        {
            object proto = MultiSeatFrames.ProtocolMp(session, l, out bool protoOk);
            cases.Add(new Case { Name = "protocol_mp", Ok = protoOk, Detail = proto });
        }
        if (Want("leak_team"))
        {
            object leakTeam = LeakTests.ExecuteTeam(session, l, a.GetInt("seats", 4), quick ? 1 : 3, quick ? 9 : 5, 99999, out bool leakTeamOk);
            cases.Add(new Case { Name = "leak_team", Ok = leakTeamOk, Detail = leakTeam });
        }
        // No name of a real run may encode as UNK: every lookup of every case above went through the one vocabulary.
        // (The synthetic and leak cases plant nothing the vocabulary does not know.)
        long unk = l.Vocab.UnknownLookups;
        cases.Add(new Case { Name = "no_unk_lookups", Ok = unk == 0, Detail = new { vocabulary_unk_lookups = unk, names = l.Vocab.UnknownLog.Distinct().ToList() } });
        bool ok = cases.All(c => c.Ok);
        HostJson.Emit(new
        {
            kind = "obs-selftest",
            ok,
            wall_s = Math.Round(wall.Elapsed.TotalSeconds, 1),
            vocab = l.Source,
            vocab_hash = l.Vocab.Hash,
            vocab_hash_hex = "0x" + l.Vocab.Hash.ToString("X8"),
            catalog_hash = l.CatalogHash,
            catalog_hash_hex = "0x" + l.CatalogHash.ToString("X8"),
            vocab_version = l.Vocab.VocabVersion,
            cases_run = cases.Count,
            cases_passed = cases.Count(c => c.Ok),
            cases = cases.Select(c => new { c.Name, c.Ok, c.Detail }).ToList(),
        }, a.Get("out"));
        return ok ? 0 : 4;
    }

    // ---------------------------------------------------------------------------------------------- serve_args

    /// <summary>
    /// Verifier finding R-3 (ruling A4, "by accident"): a server's switches written with a value. "--allow-debug-options false"
    /// must leave the debug options OFF (it used to switch them on, because only the presence of the name was tested), so
    /// "--hp" next to it is refused; a stray token after the switch is refused; the bare switch and a true literal switch it on.
    /// </summary>
    private static Case ServeArgsCase()
    {
        List<object> rows = new List<object>();
        bool ok = true;
        void Row(string line, bool wantError, bool wantDebug)
        {
            Args parsed = Args.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries));
            string? error = ObsCommands.ServeArgumentError(parsed);
            bool debug = error == null && parsed.Has("allow-debug-options");
            bool pass = (error != null) == wantError && debug == wantDebug;
            ok &= pass;
            rows.Add(new { command_line = line, refused = error, debug_options = debug, pass });
        }
        Row("--serve", false, false);
        Row("--serve --allow-debug-options", false, true);
        Row("--serve --allow-debug-options --hp 5", false, true);
        Row("--serve --allow-debug-options true --hp 5", false, true);
        Row("--serve --allow-debug-options false", false, false);
        Row("--serve --allow-debug-options 0", false, false);
        Row("--serve --allow-debug-options off", false, false);
        Row("--serve --allow-debug-options false --hp 5", true, false);   // the reviewer's command line: debug options stay off, --hp is refused
        Row("--serve --allow-debug-options maybe", true, false);          // a switch takes no value
        Row("--serve --hp 5", true, false);
        Row("--serve --hp 0", true, false);
        Row("--serve --verbose false", false, false);
        Row("--serve --fuzz-runs", true, false);                         // not on the allow-list
        return new Case { Name = "serve_args", Ok = ok, Detail = rows };
    }

    // ---------------------------------------------------------------------------------------------- vocab

    private static Case VocabCase(ObsRuntime.Loaded l)
    {
        VocabSet fresh = VocabExport.BuildFresh(GameAssemblies.Sts2Sha256);
        JObj json = fresh.ToJson();
        object? back = CanonicalJson.Parse(CanonicalJson.Pretty(json));
        bool roundTrip = CanonicalJson.Hash(back) == CanonicalJson.Hash(json);
        // informational: a merged vocabulary keeps the ids of its predecessors (names appended at the end), so it equals a
        // fresh export only for the very first version
        bool sameAsFile = fresh.Hash == l.Vocab.Hash;

        // append-only merge (schema 12.2): merging the current game into the shipped file changes nothing ...
        VocabSet same = VocabExport.Merge(l.Vocab, GameAssemblies.Sts2Sha256, out List<string> sameReport);
        bool mergeNoop = sameReport.Count == 0 && same.Hash == l.Vocab.Hash && same.VocabVersion == l.Vocab.VocabVersion;

        // ... and an "older" vocabulary that lacks two cards and has one card the game no longer has: the two are appended at
        // the end in ordinal order, the vanished one keeps its id and is marked retired, every other id is unchanged
        JObj old = (JObj)CanonicalJson.Parse(CanonicalJson.Pretty(fresh.ToJson()))!;
        List<object?> cards = (List<object?>)((JObj)((JObj)old["categories"]!)["card"]!)["entries"]!;
        string dropA = (string)cards[10]!, dropB = (string)cards[5]!;
        cards.Remove(dropA);
        cards.Remove(dropB);
        cards.Add("ZZZ_OBS_SELFTEST_RETIRED");
        VocabSet older = VocabSet.FromJson(old);
        VocabSet merged = VocabExport.Merge(older, GameAssemblies.Sts2Sha256, out List<string> report);
        List<string> mc = merged["card"].Entries;
        string first = string.CompareOrdinal(dropA, dropB) < 0 ? dropA : dropB, second = first == dropA ? dropB : dropA;
        bool appendOk = mc.Count == cards.Count + 2 && mc[^2] == first && mc[^1] == second
            && merged["card"].Retired.Contains("ZZZ_OBS_SELFTEST_RETIRED") && mc.IndexOf("ZZZ_OBS_SELFTEST_RETIRED") == cards.Count - 1
            && merged.VocabVersion == older.VocabVersion + 1;
        for (int i = 0; appendOk && i < cards.Count; i++)
        {
            appendOk = mc[i] == (string)cards[i]!;
        }
        return new Case
        {
            Name = "vocab",
            Ok = roundTrip && l.MissingNames == 0 && mergeNoop && appendOk,
            Detail = new
            {
                canonical_round_trip = roundTrip, fresh_export_equals_file = sameAsFile, missing_current_names = l.MissingNames, fresh_hash = fresh.Hash, file_hash = l.Vocab.Hash,
                merge_noop = mergeNoop, merge_append_retire = appendOk, merge_report = report,
            },
        };
    }

    // ---------------------------------------------------------------------------------------------- audit

    private static Case AuditCase()
    {
        // 1. the encoder namespace against the blacklist (which names the driver's host-only members too)
        List<string> hits = EncoderAudit.Scan(typeof(ObsEncoder).Assembly, "SpireHost.Obs.Encoding", out int methods, out int refs);
        // 2. the driver's public views (verifier finding R-7: that is where the reads of hidden game members live now)
        List<string> viewHits = EncoderAudit.ScanPublicViews(typeof(SpireHost.Driver.Decision).Assembly, out int viewMethods, out int viewRefs);
        // 3. every driver type the encoder references is a public view or a driver API type
        List<string> unaudited = EncoderAudit.UnauditedDriverTypes(typeof(ObsEncoder).Assembly, "SpireHost.Obs.Encoding");
        // the canary: both rule sets must report each of its five forbidden reads (Item, Minigame, Source, RawOptions, DrawPile)
        List<string> canaryEncoder = EncoderAudit.ScanCanary(viewRules: false, out int canaryMethods);
        List<string> canaryViews = EncoderAudit.ScanCanary(viewRules: true, out _);
        bool Reports(List<string> found, string member) => found.Exists(h => h.Contains(member, StringComparison.Ordinal));
        string[] mustReport = { "CrystalSphereCell::get_Item", "CrystalContext::Minigame", "Decision::Source", "ICardSelectHostView::get_RawOptions", "::get_DrawPile" };
        bool canaryOk = canaryMethods > 0 && Array.TrueForAll(mustReport, m => Reports(canaryEncoder, m) && Reports(canaryViews, m));
        return new Case
        {
            Name = "audit",
            Ok = hits.Count == 0 && viewHits.Count == 0 && unaudited.Count == 0 && canaryOk && viewMethods > 0,
            Detail = new
            {
                methods_scanned = methods,
                member_refs = refs,
                forbidden = hits,
                public_views = new { types = EncoderAudit.PublicViewTypes, methods_scanned = viewMethods, member_refs = viewRefs, forbidden = viewHits },
                driver_types_referenced_by_the_encoder_and_not_audited = unaudited,
                canary = new { must_report = mustReport, reported_with_encoder_rules = canaryEncoder, reported_with_view_rules = canaryViews, ok = canaryOk },
            },
        };
    }

    // ---------------------------------------------------------------------------------------------- fuzz

    private static Case FuzzCase(HostSession session, ObsRuntime.Loaded l, int runs)
    {
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = new EncoderOptions { MapInCombat = true, DeckInCombat = true, UpgradePreviews = true, TargetPreviews = true };
        EncodeChecker chk = new EncodeChecker(enc) { SyntheticEvery = 9, Reference = ObsRuntime.NewEncoder(l) };
        Dictionary<string, int> outcomes = new Dictionary<string, int>();
        List<string> aborts = new List<string>();
        for (int i = 0; i < runs; i++)
        {
            bool hp = i % 2 == 1;
            EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel("OBS-SELFTEST:" + i), Characters = new[] { DriverCommands.AllCharacters[i % 6] }, Ascension = i % 4 == 3 ? 10 : 0 };
            EpisodeOptions o = new EpisodeOptions { DebugForcedHp = hp ? 99999 : 0, MaxPolicyDecisions = hp ? 2500 : 0, MaxCombatTurns = 60 };
            EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)(500 + i)), o, chk);
            outcomes[r.Outcome.ToString()] = outcomes.GetValueOrDefault(r.Outcome.ToString()) + 1;
            if (r.Abort != null)
            {
                aborts.Add(r.Abort.Signature);
            }
        }
        // integrated host: NO episode may abort (the production handlers are the real ones)
        EncoderStats s = enc.Stats;
        bool ok = chk.Problems.Count == 0 && s.ActionOrderViolations == 0 && s.UnmappedActionRefs == 0 && s.ChoiceOrderViolations == 0 && aborts.Count == 0;
        return new Case
        {
            Name = "fuzz",
            Ok = ok,
            Detail = new
            {
                runs,
                outcomes,
                aborts,
                frames_checked = chk.Frames,
                synthetic_prompts = chk.Synthetic,
                side_effects = chk.SideEffects,
                non_deterministic = chk.NonDeterministic,
                exact_compared = chk.ExactCompared,
                exact_mismatch = chk.ExactMismatches,
                problems = chk.Problems,
                examples = chk.Examples,
                by_kind = chk.FramesByKind,
                stats = ObsCommands.Stats(s, l.Vocab),
            },
        };
    }

    // ---------------------------------------------------------------------------------------------- scenarios

    private static Case ScenarioCase(HostSession session, ObsRuntime.Loaded l)
    {
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = new EncoderOptions { MapInCombat = true, DeckInCombat = true };
        EncodeChecker chk = new EncodeChecker(enc) { SyntheticEvery = 3, Reference = ObsRuntime.NewEncoder(l) };
        (string name, string ch, Func<RunContext, Task> setup)[] scenarios =
        {
            ("shop", "IRONCLAD", async ctx => { ScenarioTools.SetGold(ctx, 999); await ScenarioTools.JumpToRoom(ctx, RoomType.Shop); }),
            ("rest", "SILENT", async ctx => { ScenarioTools.SetHp(ctx, 20); await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite); }),
            ("treasure", "DEFECT", ctx => ScenarioTools.JumpToRoom(ctx, RoomType.Treasure)),
            ("elite", "REGENT", async ctx => { ScenarioTools.SetHp(ctx, 9999, 9999); await ScenarioTools.GivePotion(ctx, "FIRE_POTION"); await ScenarioTools.JumpToRoom(ctx, RoomType.Elite); }),
            ("boss", "NECROBINDER", async ctx => { ScenarioTools.SetHp(ctx, 9999, 9999); await ScenarioTools.JumpToRoom(ctx, RoomType.Boss); }),
            ("fool-combat", "FOOL", async ctx => { ScenarioTools.SetHp(ctx, 9999, 9999); await ScenarioTools.JumpToRoom(ctx, RoomType.Monster); }),
            ("event", "IRONCLAD", ctx => ScenarioTools.JumpToRoom(ctx, RoomType.Event)),
        };
        List<object> res = new List<object>();
        bool ok = true;
        int k = 0;
        foreach (var s in scenarios)
        {
            EpisodeResult r = ScenarioRunner.Run(session, "obs-" + s.name, s.setup, character: s.ch, policy: new RandomLegalPolicy((ulong)(900 + k++)), observer: chk,
                options: new EpisodeOptions { MaxPolicyDecisions = 400, MaxCombatTurns = 30 });
            bool passed = ScenarioRunner.Passed(r) || r.Outcome == EpisodeOutcome.Truncated;
            if (r.Abort != null)
            {
                ok = false; // integrated host: no scenario may abort
            }
            res.Add(new { scenario = s.name, outcome = r.Outcome.ToString(), truncated_by = r.TruncatedBy, abort = r.Abort?.Signature, passed });
        }
        ok &= chk.Problems.Count == 0 && enc.Stats.ActionOrderViolations == 0 && enc.Stats.UnmappedActionRefs == 0;
        return new Case
        {
            Name = "scenarios",
            Ok = ok,
            Detail = new { scenarios = res, frames_checked = chk.Frames, synthetic_prompts = chk.Synthetic, problems = chk.Problems, examples = chk.Examples, by_kind = chk.FramesByKind, stats = ObsCommands.Stats(enc.Stats, l.Vocab) },
        };
    }

    // ---------------------------------------------------------------------------------------------- protocol

    private static Case ProtocolCase(HostSession session, ObsRuntime.Loaded l)
    {
        List<string> fails = new List<string>();
        Dictionary<string, object> facts = new Dictionary<string, object>();
        int ironclad = l.Vocab.Id("character", "IRONCLAD");
        void Expect(bool cond, string what)
        {
            if (!cond)
            {
                fails.Add(what);
            }
        }

        // ---- 1: errors, resend, a whole episode to TERMINAL ----
        {
            int phase = 0;
            int steps = 0;
            Random rng = new Random(5);
            ScriptTransport t = new ScriptTransport(tr =>
            {
                byte[]? last = tr.Last;
                switch (phase)
                {
                    case 0:
                        phase = 1;
                        return ScriptTransport.Step(0, 0); // STEP before any RESET
                    case 1:
                        Expect(last != null && ScriptTransport.Msg(last) == MsgType.Error && ScriptTransport.GlobalOf(last, 0) == ErrorCode.BadMessage, "STEP outside an episode -> ERROR(3)");
                        phase = 2;
                        return new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }; // garbage
                    case 2:
                        Expect(last != null && ScriptTransport.Msg(last) == MsgType.Error && ScriptTransport.GlobalOf(last, 0) == ErrorCode.BadMessage, "garbage -> ERROR(3)");
                        phase = 3;
                        return ScriptTransport.Reset(ironclad, 0, SeedCodec.Encode("SELFTEST0001"));
                    case 3:
                        Expect(last != null && ScriptTransport.Msg(last) == MsgType.Obs && ScriptTransport.StepOf(last) == 0 && ScriptTransport.EpisodeOf(last) == 1, "RESET -> OBS step 0 episode 1");
                        facts["first_obs_bytes"] = last!.Length;
                        phase = 4;
                        return ScriptTransport.Step(0, 9999);
                    case 4:
                    {
                        byte[] err = tr.Written[^2];
                        Expect(ScriptTransport.Msg(err) == MsgType.Error && ScriptTransport.GlobalOf(err, 0) == ErrorCode.BadIndex && ScriptTransport.StepOf(err) == 0, "bad index -> ERROR(1)");
                        Expect(EncodeChecker.SameIgnoringHostUs(tr.Written[^1], tr.Written[^3]), "ERROR is followed by the same OBS bytes");
                        phase = 5;
                        return ScriptTransport.Step(7, 0);
                    }
                    case 5:
                    {
                        byte[] err = tr.Written[^2];
                        Expect(ScriptTransport.Msg(err) == MsgType.Error && ScriptTransport.GlobalOf(err, 0) == ErrorCode.StaleStep, "stale step -> ERROR(2)");
                        phase = 6;
                        return new byte[12]; // malformed during a step
                    }
                    case 6:
                        Expect(ScriptTransport.Msg(tr.Written[^2]) == MsgType.Error && ScriptTransport.Msg(tr.Written[^1]) == MsgType.Obs, "malformed during step -> ERROR(3) + resend");
                        phase = 7;
                        goto case 7;
                    case 7:
                        if (last != null && ScriptTransport.Msg(last) == MsgType.Terminal)
                        {
                            Expect(ScriptTransport.StepOf(last) == steps, $"TERMINAL STEP_ID ({ScriptTransport.StepOf(last)}) = policy decisions ({steps})");
                            Expect(ScriptTransport.Word(last, Hd.Result) is Result.Win or Result.Loss or Result.Aborted, "TERMINAL has a result");
                            Expect(ScriptTransport.GlobalOf(last, Gl.TerminalReasonIdx) != 0, "TERMINAL has a terminal_reason");
                            Expect(FrameCheck.Problems(last, out _).Count == 0, "TERMINAL frame is well formed");
                            facts["terminal_result"] = ScriptTransport.Word(last, Hd.Result);
                            facts["terminal_reason"] = ScriptTransport.GlobalOf(last, Gl.TerminalReasonIdx);
                            facts["episode1_steps"] = steps;
                            phase = 8;
                            return ScriptTransport.Quit();
                        }
                        Expect(ScriptTransport.Msg(last!) == MsgType.Obs && ScriptTransport.StepOf(last!) == steps, $"OBS step {steps} in sequence");
                        int n = ScriptTransport.NActions(last!);
                        steps++;
                        return ScriptTransport.Step(steps - 1, rng.Next(n));
                    default:
                        return null;
                }
            });
            ObsEncoder enc = ObsRuntime.NewEncoder(l);
            ProtocolServer server = new ProtocolServer(session, t, enc, new ServeOptions { MaxPolicyDecisions = 5000 });
            int code = server.Serve();
            Expect(code == 0, "QUIT -> exit 0");
            Expect(phase == 8, "scenario 1 reached QUIT (phase " + phase + ")");
        }

        // ---- 2: RESET mid-episode abandons it (no TERMINAL), decision cap, same seed + same actions = same frames ----
        {
            List<byte[]> obsA = new List<byte[]>(), obsB = new List<byte[]>();
            int phase = 0, k = 0;
            ScriptTransport t = new ScriptTransport(tr =>
            {
                byte[]? last = tr.Last;
                switch (phase)
                {
                    case 0:
                        phase = 1;
                        return ScriptTransport.Reset(ironclad, 0, SeedCodec.Encode("SELFTEST0002"));
                    case 1:
                        obsA.Add(last!);
                        if (++k < 12)
                        {
                            return ScriptTransport.Step(ScriptTransport.StepOf(last!), k % ScriptTransport.NActions(last!));
                        }
                        phase = 2;
                        k = 0;
                        return ScriptTransport.Reset(ironclad, 0, SeedCodec.Encode("SELFTEST0002")); // abandon, same seed again
                    case 2:
                        if (k == 0)
                        {
                            Expect(ScriptTransport.Msg(last!) == MsgType.Obs && ScriptTransport.EpisodeOf(last!) == 2 && ScriptTransport.StepOf(last!) == 0, "RESET while an OBS is pending -> new episode 2, step 0, no TERMINAL");
                        }
                        obsB.Add(last!);
                        if (++k < 12)
                        {
                            return ScriptTransport.Step(ScriptTransport.StepOf(last!), k % ScriptTransport.NActions(last!));
                        }
                        phase = 3;
                        return ScriptTransport.Reset(ironclad, 0, null, ResetFlag.Default, maxDecisions: 3); // host seed, cap 3
                    case 3:
                        if (ScriptTransport.Msg(last!) == MsgType.Error)
                        {
                            Expect(false, $"scenario 2 phase 3: unexpected ERROR({ScriptTransport.GlobalOf(last!, 0)}) after {tr.Reads} reads");
                            phase = 5;
                            return null;
                        }
                        if (ScriptTransport.Msg(last!) == MsgType.Terminal)
                        {
                            Expect(ScriptTransport.Word(last!, Hd.Result) == Result.Loss && ScriptTransport.GlobalOf(last!, Gl.TerminalReasonIdx) == TerminalReason.DecisionCap && ScriptTransport.StepOf(last!) == 3,
                                "decision cap -> TERMINAL(LOSS, DECISION_CAP) at step 3");
                            phase = 4;
                            return null; // end of stream
                        }
                        if (tr.Reads % 1000 == 0)
                        {
                            Console.Error.WriteLine($"[selftest] scenario 2 phase 3: {tr.Reads} reads, last msg {ScriptTransport.Msg(last!)} ep {ScriptTransport.EpisodeOf(last!)} step {ScriptTransport.StepOf(last!)} kind {ScriptTransport.GlobalOf(last!, 0)}");
                        }
                        return ScriptTransport.Step(ScriptTransport.StepOf(last!), 0);
                    default:
                        return null;
                }
            });
            ObsEncoder enc = ObsRuntime.NewEncoder(l);
            ProtocolServer server = new ProtocolServer(session, t, enc, new ServeOptions());
            int code = server.Serve();
            Expect(code == 0 && phase == 4, "scenario 2 completed (phase " + phase + ", exit " + code + ")");
            Expect(!t.Overrun, $"scenario 2 looped ({t.Reads} reads, phase {phase})");
            bool same = obsA.Count == obsB.Count;
            for (int i = 0; same && i < obsA.Count; i++)
            {
                byte[] x = (byte[])obsA[i].Clone(), y = (byte[])obsB[i].Clone();
                foreach (int w in new[] { Hd.EpisodeId, Hd.HostUs })
                {
                    Array.Clear(x, w * 4, 4);
                    Array.Clear(y, w * 4, 4);
                }
                same &= x.AsSpan().SequenceEqual(y);
            }
            Expect(same, "same seed + same actions -> identical frames (episode id / HOST_US ignored)");
            Expect(server.Abandoned == 2, $"two abandoned episodes counted ({server.Abandoned})");
            facts["determinism_frames"] = obsA.Count;
        }

        // ---- 3: malformed RESETs ----
        {
            int phase = 0;
            ScriptTransport t = new ScriptTransport(tr =>
            {
                switch (phase++)
                {
                    case 0:
                        return ScriptTransport.Reset(1); // UNK is not a character
                    case 1:
                        Expect(ScriptTransport.Msg(tr.Last!) == MsgType.Error, "RESET with a non-character id -> ERROR");
                        return ScriptTransport.Reset(ironclad, 0, ulong.MaxValue);
                    case 2:
                        Expect(ScriptTransport.Msg(tr.Last!) == MsgType.Error, "RESET with a seed word >= 34^12 -> ERROR");
                        return ScriptTransport.Reset(ironclad, nSeats: 2);
                    case 3:
                        Expect(ScriptTransport.Msg(tr.Last!) == MsgType.Error, "RESET with 2 seats (H1: 1) -> ERROR");
                        return ScriptTransport.Quit();
                    default:
                        return null;
                }
            });
            ProtocolServer server = new ProtocolServer(session, t, ObsRuntime.NewEncoder(l), new ServeOptions());
            Expect(server.Serve() == 0, "scenario 3 exit 0");
            Expect(server.Episodes == 0, "no episode started by malformed RESETs");
        }
        // ---- 4: ruling A4 - debug options in a RESET, on a PRODUCTION server (no --allow-debug-options) ----
        int HpOf(byte[] frame)
        {
            FrameCheck.Problems(frame, out FrameCheck.View v);
            return v.Rows[T.Seats] > 0 ? v.Get(T.Seats, 0, SeatC.Hp) : -1;
        }
        int plainHp = -1;
        {
            int phase = 0;
            ScriptTransport t = new ScriptTransport(tr =>
            {
                switch (phase++)
                {
                    case 0:
                        return ScriptTransport.Reset(ironclad, 0, SeedCodec.Encode("SELFTEST0004"), debugWord: 99999);
                    case 1:
                        Expect(tr.Written.Count == 1 && ScriptTransport.Msg(tr.Last!) == MsgType.Error && ScriptTransport.GlobalOf(tr.Last!, 0) == ErrorCode.BadMessage,
                            "production server: RESET with a debug option (word 15) -> ERROR(3) and nothing else");
                        return ScriptTransport.Reset(ironclad, 0, SeedCodec.Encode("SELFTEST0004"), flags: ResetFlag.Default | 0x40);
                    case 2:
                        Expect(tr.Written.Count == 2 && ScriptTransport.Msg(tr.Last!) == MsgType.Error && ScriptTransport.GlobalOf(tr.Last!, 0) == ErrorCode.BadMessage,
                            "RESET with an undefined flag bit (0x40) -> ERROR(3)");
                        return ScriptTransport.Reset(ironclad, 0, SeedCodec.Encode("SELFTEST0004"));
                    case 3:
                        Expect(ScriptTransport.Msg(tr.Last!) == MsgType.Obs && ScriptTransport.EpisodeOf(tr.Last!) == 1, "the plain RESET after the refused ones starts episode 1");
                        plainHp = HpOf(tr.Last!);
                        return ScriptTransport.Quit();
                    default:
                        return null;
                }
            });
            ProtocolServer server = new ProtocolServer(session, t, ObsRuntime.NewEncoder(l), new ServeOptions());
            Expect(server.Serve() == 0, "scenario 4 exit 0");
            Expect(server.DebugResetsRefused == 1 && server.DebugEpisodes == 0 && server.Episodes == 1,
                $"production server: 1 debug RESET refused, 0 debug episodes, 1 episode (got {server.DebugResetsRefused} / {server.DebugEpisodes} / {server.Episodes})");
            facts["a4_plain_hp"] = plainHp;
        }

        // ---- 5: the same debug RESET on a server started with --allow-debug-options ----
        {
            int phase = 0;
            int debugHp = -1;
            ScriptTransport t = new ScriptTransport(tr =>
            {
                switch (phase++)
                {
                    case 0:
                        return ScriptTransport.Reset(ironclad, 0, SeedCodec.Encode("SELFTEST0004"), debugWord: 99999);
                    case 1:
                        Expect(ScriptTransport.Msg(tr.Last!) == MsgType.Obs, "debug server: RESET with word 15 = 99999 -> OBS");
                        debugHp = ScriptTransport.Msg(tr.Last!) == MsgType.Obs ? HpOf(tr.Last!) : -1;
                        return ScriptTransport.Reset(ironclad, 0, SeedCodec.Encode("SELFTEST0004"), debugWord: -5);
                    case 2:
                        Expect(ScriptTransport.Msg(tr.Last!) == MsgType.Error, "debug server: RESET with a negative debug word -> ERROR(3)");
                        return ScriptTransport.Quit();
                    default:
                        return null;
                }
            });
            ProtocolServer server = new ProtocolServer(session, t, ObsRuntime.NewEncoder(l), new ServeOptions { AllowDebugOptions = true });
            Expect(server.Serve() == 0, "scenario 5 exit 0");
            Expect(debugHp == 99999 && plainHp > 0 && plainHp < 1000, $"forced HP of the debug episode is visible in SEATS.hp (debug {debugHp}, plain {plainHp})");
            Expect(server.DebugEpisodes == 1 && server.DebugResetsRefused == 0, $"debug server: 1 debug episode, 0 refused (got {server.DebugEpisodes} / {server.DebugResetsRefused})");
            facts["a4_debug_hp"] = debugHp;
        }
        Expect(SeedCodec.TryDecode(SeedCodec.Encode("ABCDEFGH1234"), out string? back) && back == "ABCDEFGH1234", "seed codec round trip");
        // golden values of py/spire/actor.seed_to_wire (computed with the Python implementation)
        Expect(SeedCodec.Encode("ABCDEFGH1234") == 725349170157637066UL, "seed codec == Python seed_to_wire('ABCDEFGH1234')");
        Expect(SeedCodec.Encode("0oi1zzzzzzzz") == 64288580576255UL, "seed codec canonicalises like Python canonical_seed (O->0, I->1)");
        return new Case { Name = "protocol", Ok = fails.Count == 0, Detail = new { failures = fails, facts } };
    }
}
