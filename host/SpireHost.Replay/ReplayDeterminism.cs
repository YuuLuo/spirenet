using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using SpireHost.Driver;

namespace SpireHost.Replay;

/// <summary>
/// --replay-determinism [--files a,b,.. | --sample N] [--out FILE]
/// Task 4 of the replay team: (1) the same replay injected twice in this process gives identical states:
/// every generated checksum, and a host-only extended trace per checksum (round number, side, per monster next
/// move id / intent types / state-machine log length / current state; ExtendedState), must be equal between the
/// two runs; (2) the same replay with ONE recorded event altered (a card play replaced by an end turn) must be
/// detected as a divergence at or after the altered event. trace_hashes allow a cross-process comparison
/// (run the command in two processes and compare the JSON fields). Exit 0 = all equal and all alterations
/// detected, 5 otherwise.
/// </summary>
public static class ReplayDeterminism
{
    public static List<string> DefaultSample(int n)
    {
        // a spread over the corpus: every block, all characters (index order is character-interleaved)
        List<string> all = Directory.GetFiles(McrCommands.DefaultCorpus, "*.mcr").Where(f => !f.EndsWith(".rerun.mcr", StringComparison.OrdinalIgnoreCase)).OrderBy(f => f, StringComparer.Ordinal).ToList();
        List<string> pick = new List<string>();
        if (all.Count == 0)
        {
            return pick;
        }
        int step = Math.Max(1, all.Count / Math.Max(1, n));
        for (int i = 0; i < all.Count && pick.Count < n; i += step)
        {
            pick.Add(all[(i + 7) % all.Count]);
        }
        return pick;
    }

    public sealed class Row
    {
        public string File = "";
        public string Status1 = "";
        public string Status2 = "";
        public int Checksums;
        public bool ChecksumsEqual;
        public bool ExtendedEqual;
        public ulong TraceHash1;
        public ulong TraceHash2;
        public int AlteredEvent = -1;
        public string? AlteredStatus;
        public int AlteredFirstOrdinal = -1;
        public int AlteredFirstEventIndex = -1;
        public bool AlterationDetected;
        public string? AlteredContext;
    }

    public static int Run(HostSession session, Args a)
    {
        List<string> files = a.Has("files") || a.Has("file") ? McrCommands.SelectFiles(a) : DefaultSample(a.GetInt("sample", 24));
        List<Row> rows = Check(session, files);
        bool ok = rows.All(r => r.ChecksumsEqual && r.ExtendedEqual && r.Status1 == "pass" && r.Status2 == "pass" && (r.AlteredEvent < 0 || r.AlterationDetected));
        HostJson.Emit(new
        {
            command = "replay-determinism",
            replays = rows.Count,
            identical_twice = rows.Count(r => r.ChecksumsEqual && r.ExtendedEqual),
            both_pass = rows.Count(r => r.Status1 == "pass" && r.Status2 == "pass"),
            alterations = rows.Count(r => r.AlteredEvent >= 0),
            alterations_detected = rows.Count(r => r.AlteredEvent >= 0 && r.AlterationDetected),
            ok,
            trace_hashes = rows.ToDictionary(r => r.File, r => r.TraceHash1.ToString("x16")),
            rows,
        }, a.Get("out"));
        return ok ? 0 : 5;
    }

    public static List<Row> Check(HostSession session, List<string> files)
    {
        McrReplayer replayer = new McrReplayer(session.Pump);
        List<Row> rows = new List<Row>();
        foreach (string f in files)
        {
            McrFile file = McrFile.Load(f);
            ReplayOptions opt = new ReplayOptions { ExtendedTrace = true };
            McrReplayResult r1 = replayer.Run(file, opt);
            McrReplayResult r2 = replayer.Run(McrFile.Load(f), opt);
            Row row = new Row
            {
                File = r1.File,
                Status1 = r1.Status,
                Status2 = r2.Status,
                Checksums = r1.ChecksumsGenerated,
                ChecksumsEqual = r1.Checksums!.SequenceEqual(r2.Checksums!),
                ExtendedEqual = r1.ExtendedTrace!.SequenceEqual(r2.ExtendedTrace!),
                TraceHash1 = r1.TraceHash,
                TraceHash2 = r2.TraceHash,
            };
            // alter one card play in the middle of the combat
            List<int> plays = new List<int>();
            for (int i = 0; i < file.Replay.events.Count; i++)
            {
                CombatReplayEvent ev = file.Replay.events[i];
                if (ev.eventType == CombatReplayEventType.GameAction && ev.action is NetPlayCardAction)
                {
                    plays.Add(i);
                }
            }
            if (plays.Count > 0)
            {
                int alter = plays[plays.Count / 2];
                McrReplayResult r3 = replayer.Run(McrFile.Load(f), new ReplayOptions { AlterEventIndex = alter, ContinueAfterDivergence = false });
                row.AlteredEvent = alter;
                row.AlteredStatus = r3.Status;
                row.AlteredFirstOrdinal = r3.First?.ChecksumOrdinal ?? -1;
                row.AlteredFirstEventIndex = r3.First?.EventIndex ?? -1;
                row.AlteredContext = r3.First?.LocalContext ?? r3.StatusDetail;
                row.AlterationDetected = r3.Status != "pass" && (r3.First == null || r3.First.EventIndex >= alter);
            }
            rows.Add(row);
        }
        return rows;
    }
}
