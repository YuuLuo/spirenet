using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Rooms;
using SpireHost.Driver;
using SpireHost.Driver.Tools;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Protocol;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Tests;

/// <summary>
/// obs --sample-frames DIR [--per-kind N] [--runs R]: sample frames of EVERY DecisionKind from real runs of the
/// integrated host (the production handlers; nothing synthetic), for the Python side and for reviewers.
///
///   DIR\&lt;Kind&gt;.bin    [u32 LE length][frame payload] records (readable with spire.proto.read_message)
///   DIR\index.json    per frame: kind, file, record index, byte offset, bytes, sub-kind, actions, where it came from
///
/// Sources: whole random runs (natural HP and forced HP 99999, all six characters, A0 / A10) for the common kinds,
/// and scenario episodes (ScenarioTools debug jumps into the real rooms / events) for the kinds a random run
/// reaches rarely: TREASURE, REST, SHOP, CRYSTAL (Crystal Sphere), BUNDLE (Scroll Boxes), CONFIRM (Trial double down;
/// with AllowAbandon the frame carries both buttons, with the default options only NO). Every decision is sampled the
/// moment the driver asks it (policy-asked and auto-advanced alike); TERMINAL frames are the final observations.
/// At most N/4 frames of a kind are taken from one episode, so that a kind's samples come from several episodes.
/// Every written frame passes FrameCheck. Exit 0 when every kind has at least N frames, 4 otherwise.
/// </summary>
public static class SampleFrames
{
    private sealed class Sampler : IEpisodeObserver
    {
        private readonly ObsEncoder _enc;
        private readonly string _dir;
        private readonly int _perKind;
        private readonly Dictionary<string, FileStream> _files = new Dictionary<string, FileStream>(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _thisEpisode = new Dictionary<string, int>(StringComparer.Ordinal);
        public readonly SortedDictionary<string, int> Counts = new SortedDictionary<string, int>(StringComparer.Ordinal);
        public readonly List<object> Index = new List<object>();
        public readonly List<string> Problems = new List<string>();
        public string Source = "";
        public bool TakeAll;
        private int _episode;
        private bool _terminalEncoded;
        private string _seed = "", _character = "";

        public Sampler(ObsEncoder enc, string dir, int perKind)
        {
            _enc = enc;
            _dir = dir;
            _perKind = perKind;
        }

        public void OnEpisodeStart(RunContext ctx)
        {
            _episode++;
            _thisEpisode.Clear();
            _seed = ctx.Seed; // HOST-side index only (never in a frame)
            _character = ctx.Spec.Character;
        }

        private bool Wants(string kind)
        {
            int have = Counts.GetValueOrDefault(kind);
            if (TakeAll)
            {
                return have < _perKind * 3;
            }
            return have < _perKind * 2 && _thisEpisode.GetValueOrDefault(kind) < Math.Max(1, _perKind / 4);
        }

        public void OnDecision(RunContext ctx, Decision decision, int chosen, bool auto)
        {
            string kind = decision.Kind.ToString();
            if (!Wants(kind))
            {
                return;
            }
            _enc.EncodeDecision(decision, _episode, (int)ctx.Counters.PolicyDecisions, 0);
            Write(kind, decision.SubKind, decision.Actions.Count, auto, (int)ctx.Counters.PolicyDecisions);
        }

        private void Write(string kind, int subKind, int actions, bool auto, int step)
        {
            byte[] b = _enc.Payload.ToArray();
            foreach (string p in FrameCheck.Problems(b, out _))
            {
                if (Problems.Count < 20)
                {
                    Problems.Add(kind + ": " + p);
                }
            }
            if (!_files.TryGetValue(kind, out FileStream? fs))
            {
                fs = File.Create(Path.Combine(_dir, kind + ".bin"));
                _files[kind] = fs;
            }
            long offset = fs.Position;
            Span<byte> head = stackalloc byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(head, b.Length);
            fs.Write(head);
            fs.Write(b);
            int n = Counts.GetValueOrDefault(kind);
            Counts[kind] = n + 1;
            _thisEpisode[kind] = _thisEpisode.GetValueOrDefault(kind) + 1;
            Index.Add(new
            {
                kind, file = kind + ".bin", record = n, offset, bytes = b.Length, sub_kind = subKind, n_actions = actions, auto_advanced = auto,
                step, source = Source, character = _character, host_seed = _seed,
            });
        }

        public void OnTerminalState(RunContext ctx, EpisodeResult provisional)
        {
            _terminalEncoded = false;
            if (!Wants("Terminal") || provisional.Outcome == EpisodeOutcome.Aborted)
            {
                return;
            }
            _enc.EncodeTerminal(ctx, _episode, (int)ctx.Counters.PolicyDecisions, 0);
            _terminalEncoded = true;
        }

        public void OnEpisodeEnd(EpisodeResult result)
        {
            if (!_terminalEncoded)
            {
                return;
            }
            // the final RESULT / terminal fields exactly as the protocol server patches them
            (int res, int reason) = ProtocolServer.Outcome(result);
            FrameWriter w = _enc.Writer;
            w.PatchHeader(Hd.Result, res);
            w.PatchHeader(Hd.StepId, (int)result.Counters.PolicyDecisions);
            w.PatchGlobal(Gl.TerminalReasonIdx, reason);
            w.PatchGlobal(Gl.AbortReasonIdx, 0);
            w.PatchGlobal(Gl.PolicyDecisionsIdx, (int)result.Counters.PolicyDecisions);
            Write("Terminal", 0, 0, false, (int)result.Counters.PolicyDecisions);
            _terminalEncoded = false;
        }

        public void Close()
        {
            foreach (FileStream fs in _files.Values)
            {
                fs.Dispose();
            }
            _files.Clear();
        }
    }

    /// <summary>Prefers event options by TextKey suffix (in order), the YES / NO of a CONFIRM, else uniform random.</summary>
    private sealed class PreferKeys : IPolicy
    {
        private readonly string[] _suffixes;
        private readonly int[] _used;
        private readonly bool _confirmYes;
        private readonly RandomLegalPolicy _random;

        public PreferKeys(ulong seed, bool confirmYes, params string[] suffixes)
        {
            _suffixes = suffixes;
            _used = new int[suffixes.Length];
            _confirmYes = confirmYes;
            _random = new RandomLegalPolicy(seed);
        }

        public int Decide(Decision d)
        {
            if (d.Kind == DecisionKind.Confirm)
            {
                for (int i = 0; i < d.Actions.Count; i++)
                {
                    if (d.Actions[i].Source.Kind == RefKind.ConfirmChoice && d.Actions[i].Source.Index == (_confirmYes ? 1 : 0))
                    {
                        return i;
                    }
                }
            }
            if (d.Kind == DecisionKind.Event)
            {
                for (int s = 0; s < _suffixes.Length; s++)
                {
                    if (_used[s] >= 3)
                    {
                        continue; // e.g. Trial DOUBLE_DOWN answered NO returns to the same page: move on after three rounds
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

    public static int Run(HostSession session, Args a)
    {
        string dir = Path.GetFullPath(a.Get("sample-frames") ?? "frames");
        int perKind = a.GetInt("per-kind", 20);
        int runs = a.GetInt("runs", 36);
        Directory.CreateDirectory(dir);
        ObsRuntime.Loaded l = ObsModule.Current;
        ObsEncoder enc = ObsRuntime.NewEncoder(l);
        enc.Options = EncoderOptions.FromResetFlags(ResetFlag.Default, 0, -1); // what a default Python RESET asks for
        Sampler s = new Sampler(enc, dir, perKind);
        List<object> episodes = new List<object>();
        try
        {
            // ---- whole random runs ----
            for (int i = 0; i < runs; i++)
            {
                bool hp = i % 3 == 2;
                string ch = DriverCommands.AllCharacters[i % 6];
                EpisodeSpec spec = new EpisodeSpec { Seed = HostSession.SeedFromLabel("SAMPLE:" + i), Characters = new[] { ch }, Ascension = (i / 6) % 2 == 0 ? 0 : 10 };
                s.Source = hp ? "random run, forced HP 99999" : "random run";
                s.TakeAll = false;
                EpisodeResult r = session.RunEpisode(spec, new RandomLegalPolicy((ulong)(9000 + i)),
                    new EpisodeOptions { DebugForcedHp = hp ? 99999 : 0, MaxPolicyDecisions = hp ? 3000 : 0, MaxCombatTurns = 80 }, s);
                episodes.Add(new { source = s.Source, character = ch, outcome = r.Outcome.ToString(), r.TruncatedBy, abort = r.Abort?.Signature, floor = r.TotalFloor });
            }

            // ---- scenario episodes for the kinds a random run reaches rarely ----
            (string name, string ch, bool abandon, string[]? prefer, Func<RunContext, Task> setup)[] scenarios =
            {
                ("treasure room", "DEFECT", false, null, ctx => ScenarioTools.JumpToRoom(ctx, RoomType.Treasure)),
                ("treasure room with a potion (CHEST page asked)", "SILENT", false, null, async ctx => { await ScenarioTools.GivePotion(ctx, "FIRE_POTION"); await ScenarioTools.JumpToRoom(ctx, RoomType.Treasure); }),
                ("rest site", "IRONCLAD", false, null, async ctx => { ScenarioTools.SetHp(ctx, 20); await ScenarioTools.JumpToRoom(ctx, RoomType.RestSite); }),
                ("shop", "REGENT", false, null, async ctx => { ScenarioTools.SetGold(ctx, 999); await ScenarioTools.JumpToRoom(ctx, RoomType.Shop); }),
                ("event CRYSTAL_SPHERE", "NECROBINDER", false, new[] { "UNCOVER_FUTURE", "PAYMENT_PLAN" }, async ctx => { ScenarioTools.SetGold(ctx, 500); await ScenarioTools.JumpToEvent(ctx, "CRYSTAL_SPHERE"); }),
                ("relic SCROLL_BOXES", "FOOL", false, null, ctx => ScenarioTools.GiveRelic(ctx, "SCROLL_BOXES")),
                ("event TRIAL, double down (default options: only NO, auto-advanced)", "IRONCLAD", false, new[] { "options.REJECT", "options.DOUBLE_DOWN", "options.ACCEPT" }, ctx => ScenarioTools.JumpToEvent(ctx, "TRIAL")),
                ("event TRIAL, double down (AllowAbandon: NO and YES)", "SILENT", true, new[] { "options.REJECT", "options.DOUBLE_DOWN", "options.ACCEPT" }, ctx => ScenarioTools.JumpToEvent(ctx, "TRIAL")),
            };
            for (int round = 0; round < 40; round++)
            {
                bool need = false;
                foreach (DecisionKind k in Enum.GetValues<DecisionKind>())
                {
                    need |= k != DecisionKind.None && s.Counts.GetValueOrDefault(k.ToString()) < perKind;
                }
                if (!need)
                {
                    break;
                }
                foreach (var sc in scenarios)
                {
                    s.Source = "scenario: " + sc.name;
                    s.TakeAll = true;
                    string[] chars = DriverCommands.AllCharacters;
                    string ch = chars[(Array.IndexOf(chars, sc.ch) + round) % chars.Length];
                    ulong policySeed = (ulong)(100 + round * 17 + sc.name.Length);
                    IPolicy policy = sc.prefer != null ? new PreferKeys(policySeed, false, sc.prefer) : new RandomLegalPolicy(policySeed);
                    EpisodeResult r = ScenarioRunner.Run(session, "sample " + sc.name + " #" + round, sc.setup, character: ch,
                        seed: HostSession.SeedFromLabel("SAMPLE-SCENARIO:" + sc.name + ":" + round), policy: policy, observer: s,
                        options: new EpisodeOptions { MaxPolicyDecisions = 200, MaxCombatTurns = 30, AllowAbandon = sc.abandon, DebugForcedHp = 999 });
                    if (round == 0)
                    {
                        episodes.Add(new { source = s.Source, character = ch, outcome = r.Outcome.ToString(), r.TruncatedBy, abort = r.Abort?.Signature, floor = r.TotalFloor });
                    }
                }
            }
        }
        finally
        {
            s.Close();
        }
        List<string> shortKinds = new List<string>();
        foreach (DecisionKind k in Enum.GetValues<DecisionKind>())
        {
            if (k != DecisionKind.None && s.Counts.GetValueOrDefault(k.ToString()) < perKind)
            {
                shortKinds.Add(k + " (" + s.Counts.GetValueOrDefault(k.ToString()) + ")");
            }
        }
        bool ok = shortKinds.Count == 0 && s.Problems.Count == 0 && enc.Stats.ActionOrderViolations == 0 && enc.Stats.UnmappedActionRefs == 0 && enc.Stats.ChoiceOrderViolations == 0
                  && l.Vocab.UnknownLookups == 0; // no sampled frame may carry an UNK id
        object index = new
        {
            kind = "obs-sample-frames",
            ok,
            dir,
            record_format = "[u32 LE length][frame payload] (obs-schema-v1 3.1)",
            schema_version = SchemaV1.Version,
            vocab_hash = unchecked((uint)l.Vocab.Hash),
            vocab_hash_hex = "0x" + l.Vocab.Hash.ToString("X8"),
            catalog_hash = unchecked((uint)l.CatalogHash),
            catalog_hash_hex = "0x" + l.CatalogHash.ToString("X8"),
            vocab_version = l.Vocab.VocabVersion,
            reset_flags = ResetFlag.Default,
            per_kind_required = perKind,
            frames_by_kind = s.Counts,
            kinds_below_required = shortKinds,
            frame_check_problems = s.Problems,
            encoder = ObsCommands.Stats(enc.Stats, l.Vocab),
            episodes,
            frames = s.Index,
        };
        File.WriteAllText(Path.Combine(dir, "index.json"), HostJson.Serialize(index));
        HostJson.Emit(new { kind = "obs-sample-frames", ok, dir, frames_by_kind = s.Counts, kinds_below_required = shortKinds, frame_check_problems = s.Problems, encoder = ObsCommands.Stats(enc.Stats, l.Vocab) }, a.Get("out"));
        return ok ? 0 : 4;
    }
}
