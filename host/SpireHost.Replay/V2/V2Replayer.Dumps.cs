using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using SpireHost.Driver;

namespace SpireHost.Replay.V2;

/// <summary>
/// Room boundaries: the tracer's dump (RunTracer2.Dump, field by field) and the game's own RunManager.ToSave(null) JSON
/// against the trace's NNN-*.json / NNN-*.save.json.
///
/// Ignored in the save (and nothing else): save_time, start_time, run_time, win_time (wall clock), map_drawings (UI node),
/// as in the v1 tool; players[].discovered_* (profile recipe; counted) and the Trial EntrantNumber variable (Rng.Chaotic;
/// counted) - the two fields the validation report proves the real client itself does not reproduce.
/// Ignored in the dump: the v2 block's "frame" (engine frame counter).
///
/// Combat-room dumps: the tracer samples at a frame boundary inside the turn-1 start sequence, i.e. between two of the
/// game's commands (a card drawn, a power applied, damage dealt ...) of that sequence. The host has no frames: it builds
/// the same dump and save at CombatManager.CombatBegan, after every entry the game adds to its own CombatHistory until
/// the first decision point (public event CombatHistory.Changed; reads only), and at that decision point. The trace's
/// dump and save - one snapshot of one moment - must equal, as a whole, the host at its first decision point or - at a
/// combat room's entry dump only - ONE of those samples (never field by field from different states). Never anything
/// else. (--strict-sampling: the first decision point only.)
/// </summary>
public sealed partial class V2Replayer
{
    private int _nextDump;
    private int _lastMapActDumped = -1;
    private Action<CombatState>? _onCombatBegan;
    private Action? _onHistoryChanged;
    private Action<CombatRoom>? _onCombatEnded;
    private object? _sampleRoom;                                              // the combat room whose turn-1 start is being sampled
    private readonly List<(JsonObject Dump, string Save)> _samples = new List<(JsonObject, string)>();
    private const int MaxSamples = 400;
    private bool _dumpDeferred;                                               // the next trace dump was written after its room's fight: compare it then
    private bool _dumpDeferredNow;

    private static readonly HashSet<string> IgnoredSaveFields = new HashSet<string>(StringComparer.Ordinal)
    {
        "save_time", "start_time", "run_time", "win_time", "map_drawings",
    };

    private sealed class Diff
    {
        public string Kind = "";
        public string Path = "";
        public string? Trace;
        public string? Host;
    }

    private void SubscribeSampling(RunContext ctx)
    {
        if (_onCombatBegan != null)
        {
            return;
        }
        // read-only: the tracer's dump and the game's own save at the moments the real client's tracer may be dumping
        _onCombatBegan = _ =>
        {
            RunContext? c = HostSession.Current?.CurrentEpisode;
            if (c == null || _stop)
            {
                return;
            }
            NoteHostFightBegan(c);
            _samples.Clear();
            _sampleRoom = c.Run.CurrentRoom;
            _samples.Add((BuildDump(c, "room", peekMap: true), SaveJson()));
        };
        _onHistoryChanged = () =>
        {
            RunContext? c = HostSession.Current?.CurrentEpisode;
            if (c == null || _stop || _sampleRoom == null || !ReferenceEquals(_sampleRoom, c.Run.CurrentRoom) || _samples.Count >= MaxSamples)
            {
                return;
            }
            _samples.Add((BuildDump(c, "room", peekMap: true), SaveJson()));
        };
        // A room dump that the tracer wrote after the room's fight (see CompareDump): the tracer's loop dumps as soon as the fight is
        // no longer in progress, with the combat room still the current room. The host's equivalent moment is the end of
        // CombatManager.EndCombatInternal (public event CombatEnded, Combat/CombatManager.cs:1079); the room is left afterwards.
        _onCombatEnded = _ =>
        {
            RunContext? c = HostSession.Current?.CurrentEpisode;
            if (c == null || _stop || !_dumpDeferred)
            {
                return;
            }
            _dumpDeferred = false;
            _dumpDeferredNow = true;
            try
            {
                CompareDump(c, "room");
            }
            catch (Exception e)
            {
                _r.Error = "tool exception at the deferred room dump: " + e.GetType().Name + ": " + e.Message;
                Stop("tool_error", _r.Error);
            }
        };
        ctx.CombatManager.CombatBegan += _onCombatBegan;
        ctx.CombatManager.History.Changed += _onHistoryChanged;
        ctx.CombatManager.CombatEnded += _onCombatEnded;
    }

    private void UnsubscribeSampling(RunContext ctx)
    {
        if (_onCombatBegan != null)
        {
            ctx.CombatManager.CombatBegan -= _onCombatBegan;
            _onCombatBegan = null;
        }
        if (_onHistoryChanged != null)
        {
            ctx.CombatManager.History.Changed -= _onHistoryChanged;
            _onHistoryChanged = null;
        }
        if (_onCombatEnded != null)
        {
            ctx.CombatManager.CombatEnded -= _onCombatEnded;
            _onCombatEnded = null;
        }
        _sampleRoom = null;
        _samples.Clear();
    }

    private static string SaveJson() => JsonSerializer.Serialize(RunManager.Instance.ToSave(null), JsonSerializationUtility.GetTypeInfo<SerializableRun>());

    private void CompareDump(RunContext ctx, string kind)
    {
        if (_nextDump >= _t.Dumps.Count)
        {
            Diverge("dump.host_has_more_rooms", true, $"host room boundary {_nextDump} ({kind}) floor {ctx.Run.TotalFloor}", "the host reached a room boundary the trace has no dump for (trace end: " + _t.End + ")", null, RunDriverRoom(ctx));
            return;
        }
        V2Dump td = _t.Dumps[_nextDump];
        string where = $"dump {td.Index} ({td.File}) floor {ctx.Run.TotalFloor}";
        if (td.Kind != kind)
        {
            Diverge("dump.kind", true, where, "the trace's next dump is of another kind than the host's boundary", td.Kind, kind);
            return;
        }
        if (kind == "room" && !_dumpDeferredNow && _combatIndex < _t.Combats.Count && _t.Combats[_combatIndex].Tag == td.Index && ctx.Run.CurrentRoom is CombatRoom
            && _t.Combats[_combatIndex].Encounter == S(td.Dump["room_model"]) && _t.Combats[_combatIndex].Floor == I(td.Dump["total_floor"]) && Rule(V2Classes.DumpWrittenAfterFight))
        {
            // The trace's next fight carries THIS dump's index in its file names: the tracer played the fight first (the combat was
            // already in progress while IsChangingRooms was still true) and wrote the room's dump when the fight was over. The dump is
            // compared when the host's fight is over, at the first decision after it.
            _dumpDeferred = true;
            _samples.Clear();
            _sampleRoom = null;
            return;
        }
        bool afterFight = _dumpDeferredNow;
        _dumpDeferredNow = false;
        int index = _nextDump++;
        _r.DumpsCompared++;
        // the samples of this combat room's turn-1 start (taken since CombatBegan); sampling ends here, at the first decision point
        List<(JsonObject Dump, string Save)> samples = new List<(JsonObject, string)>(_samples);
        bool sampled = kind == "room" && !afterFight && !_opt.StrictSampling && samples.Count > 0 && samples.Count < MaxSamples && ctx.Run.CurrentRoom is CombatRoom && ReferenceEquals(_sampleRoom, ctx.Run.CurrentRoom);
        _samples.Clear();
        _sampleRoom = null;

        JsonObject host = BuildDump(ctx, kind, peekMap: false);
        string hostSave = SaveJson();
        if (_opt.DumpDir != null)
        {
            string dir = Path.Combine(_opt.DumpDir, _t.Set, _t.Name);
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, $"{index:D3}-{kind}.host.json"), host.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
            File.WriteAllText(Path.Combine(dir, $"{index:D3}-{kind}.host.save.json"), hostSave);
            if (sampled)
            {
                File.WriteAllText(Path.Combine(dir, $"{index:D3}-{kind}.host-combat-began.json"), samples[0].Dump.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
                File.WriteAllText(Path.Combine(dir, $"{index:D3}-{kind}.host-combat-began.save.json"), samples[0].Save);
            }
        }
        JsonNode? traceSave = _opt.CompareSave ? td.LoadSave() : null;
        Dictionary<string, int> masked = new Dictionary<string, int>(StringComparer.Ordinal);
        List<Diff> diffs = new List<Diff>();
        CompareDumpNodes(td.Dump, host, diffs);
        if (traceSave != null)
        {
            _r.SavesCompared++;
            CompareNodes(traceSave, JsonNode.Parse(hostSave), "save", "save", diffs, masked);
        }
        foreach (KeyValuePair<string, int> kv in masked)
        {
            Explain(kv.Key, kv.Value);
        }
        if (diffs.Count == 0)
        {
            _r.DumpsMatched++;
            return;
        }
        int closest = int.MaxValue;
        if (sampled)
        {
            // The tracer's dump and save are ONE snapshot of ONE moment of the real client's turn-1 start sequence. The rule: ONE of the
            // host's own states of that sequence (CombatBegan, then after every CombatHistory entry) must equal the WHOLE dump and save.
            // (Field-by-field agreement with different samples is not accepted.)
            for (int i = 0; i < samples.Count; i++)
            {
                List<Diff> early = new List<Diff>();
                CompareDumpNodes(td.Dump, samples[i].Dump, early, hostIsEarlySample: true);
                if (traceSave != null)
                {
                    CompareNodes(traceSave, JsonNode.Parse(samples[i].Save), "save", "save", early, new Dictionary<string, int>());
                }
                closest = Math.Min(closest, early.Count);
                if (early.Count == 0 && Rule(V2Classes.DumpCombatEntrySampling))
                {
                    _r.DumpsMatchedBySampling++;
                    Info("dump.sampling.sample_index_that_equals_the_dump:" + (i == 0 ? "0 (CombatBegan)" : i < 10 ? i.ToString() : i < 30 ? "10-29" : "30+"));
                    foreach (IGrouping<string, Diff> g in diffs.GroupBy(x => NormalizePath(x.Path)))
                    {
                        Explain(V2Classes.DumpCombatEntrySampling + ":" + g.Key, g.Count());
                    }
                    return;
                }
            }
        }
        Diff first = diffs[0];
        Diverge("dump." + first.Kind + ":" + NormalizePath(first.Path), true, where, $"{diffs.Count} field(s) differ; first: {first.Path}" + (sampled ? $" (the dump equals the host neither at its first decision point nor at any of the {samples.Count} states of the turn-1 start sequence; the closest state differs in {closest} field(s))" : ""),
            string.Join(" | ", diffs.Take(_opt.MaxDiffs).Select(x => x.Path + "=" + x.Trace)), string.Join(" | ", diffs.Take(_opt.MaxDiffs).Select(x => x.Path + "=" + x.Host)));
    }

    /// <summary>
    /// Every key of the trace dump against the host's, and every key of the host's dump the trace lacks. The act map
    /// (map / map_boss / map_start) is part of a dump the first time an act is seen - by the same rule on both sides; the
    /// host's sample taken at CombatBegan always carries it, so there it is compared only when the trace has it.
    /// </summary>
    private void CompareDumpNodes(JsonObject trace, JsonObject host, List<Diff> diffs, bool hostIsEarlySample = false)
    {
        foreach (KeyValuePair<string, JsonNode?> kv in trace)
        {
            if (kv.Key == "v2")
            {
                JsonObject tv = kv.Value!.AsObject();
                JsonObject hv = host["v2"]!.AsObject();
                foreach (KeyValuePair<string, JsonNode?> f in tv)
                {
                    if (f.Key != "frame")
                    {
                        CompareNodes(f.Value, hv[f.Key], "dump.v2." + f.Key, "structure", diffs, null);
                    }
                }
                continue;
            }
            CompareNodes(kv.Value, host[kv.Key], "dump." + kv.Key, "structure", diffs, null);
        }
        foreach (KeyValuePair<string, JsonNode?> kv in host)
        {
            if (trace.ContainsKey(kv.Key) || (hostIsEarlySample && (kv.Key == "map" || kv.Key == "map_boss" || kv.Key == "map_start")))
            {
                continue;
            }
            diffs.Add(new Diff { Kind = "structure", Path = "dump." + kv.Key, Trace = "(absent)", Host = ShortJson(kv.Value) });
        }
    }

    /// <summary>RunTracer2.Dump, field by field, from the host's run state.</summary>
    private JsonObject BuildDump(RunContext ctx, string kind, bool peekMap)
    {
        RunState rs = ctx.Run;
        Player me = ctx.LocalPlayer;
        static JsonArray Arr(IEnumerable<string?> xs) => new JsonArray(xs.Select(x => (JsonNode?)JsonValue.Create(x)).ToArray());
        // the collector counts the records it has written; the host has consumed exactly those when it reaches the same boundary
        int consumed = _rec;
        while (consumed < _t.Records.Count && IsAfterTheFact(_t.Records[consumed], attachedOnly: true))
        {
            consumed++;
        }
        JsonObject o = new JsonObject
        {
            ["index"] = _nextDump - (peekMap ? 0 : 1),
            ["kind"] = kind,
            ["total_floor"] = rs.TotalFloor,
            ["act_index"] = rs.CurrentActIndex,
            ["act"] = rs.Act.Id.Entry,
            ["act_floor"] = rs.ActFloor,
            ["acts"] = Arr(rs.Acts.Select(a => a.Id.Entry)),
            ["bosses"] = Arr(rs.Acts.Select(a => $"{a.BossEncounter?.Id.Entry}|{a.SecondBossEncounter?.Id.Entry}")),
            ["room_type"] = rs.CurrentRoom?.RoomType.ToString(),
            ["room_model"] = rs.CurrentRoom?.ModelId?.Entry,
            ["coord"] = rs.CurrentMapCoord is MapCoord c ? $"{c.col},{c.row}" : null,
            ["hp"] = $"{me.Creature.CurrentHp}/{me.Creature.MaxHp}",
            ["gold"] = me.Gold,
            ["deck"] = Arr(me.Deck.Cards.Select(x => x.Id.Entry + (x.IsUpgraded ? "+" : ""))),
            ["relics"] = Arr(me.Relics.Select(x => x.Id.Entry)),
            ["potions"] = Arr(me.PotionSlots.Select(x => x?.Id.Entry ?? "-")),
            ["card_reward_offers_so_far"] = Arr(_hostCardOffers),
            ["choices_so_far"] = consumed,
        };
        bool withMap = rs.Map != null && (peekMap || rs.CurrentActIndex != _lastMapActDumped);
        if (!peekMap && withMap)
        {
            _lastMapActDumped = rs.CurrentActIndex;
        }
        if (withMap)
        {
            o["map"] = new JsonArray(rs.Map!.GetAllMapPoints().OrderBy(p => p.coord.row).ThenBy(p => p.coord.col).Select(p => (JsonNode)new JsonObject
            {
                ["col"] = p.coord.col,
                ["row"] = p.coord.row,
                ["type"] = p.PointType.ToString(),
                ["children"] = Arr(p.Children.OrderBy(ch => ch.coord.col).Select(ch => $"{ch.coord.col},{ch.coord.row}")),
            }).ToArray());
            o["map_boss"] = $"{rs.Map.BossMapPoint?.coord.col},{rs.Map.BossMapPoint?.coord.row}";
            o["map_start"] = $"{rs.Map.StartingMapPoint?.coord.col},{rs.Map.StartingMapPoint?.coord.row}";
        }
        if (rs.CurrentRoom is EventRoom er)
        {
            o["event"] = er.CanonicalEvent?.Id.Entry;
            // the tracer lists the option buttons of NEventRoom: the local event's current options (a finished event shows the synthesized PROCEED)
            EventModel? local = RunManager.Instance.EventSynchronizer.GetLocalEvent();
            if (local != null)
            {
                o["event_options_visible"] = EventButtons(local);
            }
        }
        if (rs.CurrentRoom is MerchantRoom)
        {
            MerchantInventory inv = RunManager.Instance.MerchantSynchronizer.GetLocalInventory();
            o["shop"] = new JsonArray(inv.AllEntries.Select(e => (JsonNode)new JsonObject
            {
                ["type"] = e.GetType().Name,
                ["cost"] = e.IsStocked ? e.Cost : -1,
                ["item"] = e switch
                {
                    MerchantCardEntry ce => ce.CreationResult?.Card?.Id.Entry + ((ce.CreationResult?.Card?.IsUpgraded ?? false) ? "+" : ""),
                    MerchantRelicEntry re => re.Model?.Id.Entry,
                    MerchantPotionEntry pe => pe.Model?.Id.Entry,
                    _ => null,
                },
            }).ToArray());
        }
        o["v2"] = new JsonObject
        {
            ["decisions_so_far"] = consumed,
            ["room_count"] = rs.CurrentRoomCount,
            ["second_boss_point"] = rs.Map?.SecondBossMapPoint == null ? null : $"{rs.Map.SecondBossMapPoint.coord.col},{rs.Map.SecondBossMapPoint.coord.row}",
        };
        return o;
    }

    /// <summary>The TextKeys of the option buttons the real event room shows for this event state (NEventRoom.SetOptions, Nodes.Rooms/NEventRoom.cs:217-231).</summary>
    private static JsonArray EventButtons(EventModel local)
    {
        // An event with a custom scene (FakeMerchant) has no NEventLayout: NEventRoom.SetupLayout returns before SetOptions
        // (Nodes.Rooms/NEventRoom.cs:86, :163-166) and the room shows no option button at all.
        IEnumerable<string?> keys = local.LayoutType == MegaCrit.Sts2.Core.Events.EventLayoutType.Custom ? Array.Empty<string?>()
            : local.IsFinished ? new string?[] { "PROCEED" } : local.CurrentOptions.Select(x => (string?)x.TextKey);
        return new JsonArray(keys.Select(x => (JsonNode?)JsonValue.Create(x)).ToArray());
    }

    private void CompareNodes(JsonNode? a, JsonNode? b, string path, string kind, List<Diff> diffs, Dictionary<string, int>? masked)
    {
        if (diffs.Count >= _opt.MaxDiffs * 8)
        {
            return;
        }
        if (a is JsonObject oa && b is JsonObject ob)
        {
            foreach (KeyValuePair<string, JsonNode?> kv in oa)
            {
                if (kind == "save" && IgnoredSaveFields.Contains(kv.Key))
                {
                    continue;
                }
                if (kind == "save" && Masked(kv.Key, path, kv.Value, ob[kv.Key], masked))
                {
                    continue;
                }
                CompareNodes(kv.Value, ob[kv.Key], path + "." + kv.Key, kind, diffs, masked);
            }
            foreach (KeyValuePair<string, JsonNode?> kv in ob)
            {
                if (oa.ContainsKey(kv.Key) || (kind == "save" && IgnoredSaveFields.Contains(kv.Key)))
                {
                    continue;
                }
                if (kind == "save" && Masked(kv.Key, path, null, kv.Value, masked))
                {
                    continue;
                }
                diffs.Add(new Diff { Kind = kind, Path = path + "." + kv.Key, Trace = "(absent)", Host = ShortJson(kv.Value) });
            }
            return;
        }
        if (a is JsonArray la && b is JsonArray lb)
        {
            int n = Math.Max(la.Count, lb.Count);
            for (int i = 0; i < n; i++)
            {
                if (i >= la.Count || i >= lb.Count)
                {
                    diffs.Add(new Diff { Kind = kind, Path = path + "[" + i + "]", Trace = i < la.Count ? ShortJson(la[i]) : "(absent)", Host = i < lb.Count ? ShortJson(lb[i]) : "(absent)" });
                    return;
                }
                CompareNodes(la[i], lb[i], path + "[" + i + "]", kind, diffs, masked);
            }
            return;
        }
        string sa = a?.ToJsonString() ?? "null";
        string sb = b?.ToJsonString() ?? "null";
        if (sa != sb)
        {
            diffs.Add(new Diff { Kind = kind, Path = path, Trace = ShortJson(a), Host = ShortJson(b) });
        }
    }

    /// <summary>The two save fields the real client itself does not reproduce (validation report P1): compared, a difference is counted under its named rule and not reported.</summary>
    private bool Masked(string key, string path, JsonNode? trace, JsonNode? host, Dictionary<string, int>? masked)
    {
        string? cls = null;
        if (_opt.KnownSaveDiffs.Contains(key) && path.StartsWith("save.players[", StringComparison.Ordinal))
        {
            cls = V2Classes.SaveDiscovered + ":" + key;
        }
        else if (key == "EntrantNumber" && path.EndsWith(".variables", StringComparison.Ordinal) && path.Contains(".event_choices[", StringComparison.Ordinal) && _opt.KnownSaveDiffs.Count > 0)
        {
            cls = V2Classes.SaveEntrantNumber;
        }
        if (cls == null)
        {
            return false;
        }
        if ((trace?.ToJsonString() ?? "null") != (host?.ToJsonString() ?? "null") && masked != null)
        {
            masked[cls] = masked.GetValueOrDefault(cls) + 1;
        }
        return true;
    }

    private static string ShortJson(JsonNode? n)
    {
        string s = n?.ToJsonString() ?? "null";
        return s.Length > 200 ? s.Substring(0, 200) + "..." : s;
    }

    private static string NormalizePath(string p)
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        bool inIndex = false;
        foreach (char ch in p)
        {
            if (ch == '[')
            {
                inIndex = true;
                sb.Append("[]");
                continue;
            }
            if (ch == ']')
            {
                inIndex = false;
                continue;
            }
            if (!inIndex)
            {
                sb.Append(ch);
            }
        }
        return sb.ToString();
    }
}
