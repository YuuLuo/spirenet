using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;

namespace SpireHost.Replay.V2;

/// <summary>One room-boundary dump of a v2 trace (NNN-start / NNN-room / NNN-end-&lt;reason&gt; .json + .save.json).</summary>
public sealed class V2Dump
{
    public int Index;
    public string Kind = "";          // start | room | end-<reason>
    public string File = "";
    public JsonObject Dump = null!;
    public string? SavePath;

    public JsonNode? LoadSave() => SavePath != null && System.IO.File.Exists(SavePath) ? JsonNode.Parse(System.IO.File.ReadAllText(SavePath)) : null;
}

/// <summary>One fight of a v2 trace: NNN-combat-f&lt;floor&gt;.combat.json (+ .mcr when the client wrote one).</summary>
public sealed class V2Combat
{
    public int CombatIndex;           // 1-based (combat_list)
    public int Tag;                   // the NNN of the file name = the dump index at the time of the fight
    public int Floor;
    public string Encounter = "";
    public string RoomType = "";
    public string End = "";           // win | loss | max_turns | stall | timeout | error
    public string File = "";
    public JsonArray Actions = new JsonArray();

    /// <summary>The in-combat selection records in order (kind "selection" / "bundle"); each carries its result under "_result" when one was written.</summary>
    public List<JsonObject> Prompts = new List<JsonObject>();

    /// <summary>selection_result records of this fight that have no selection record (never expected).</summary>
    public int OrphanResults;

    /// <summary>The game's own replay of this fight; null when the client wrote none (max_turns / stall fights).</summary>
    public string? McrPath;
}

/// <summary>
/// A run trace of format "spireoracle-run-trace-v2" (notes\oracle-deep.md section 2): decisions.jsonl with one record
/// BEFORE every action outside combat, room-boundary dumps + the game's own run save, one combat.json (+ .mcr) per fight.
/// </summary>
public sealed class V2Trace
{
    public string Dir = "";
    public string Name = "";
    public string Set = "";            // natural | deep (the header's mode)
    public string Character = "";
    public int Ascension;
    public string Seed = "";
    public string End = "";
    public string ModVersion = "";
    public int ForcedHp;               // 0 = no fixture (natural)
    public bool FixtureDeclared;
    public JsonNode Summary = null!;
    public List<JsonObject> Records = new List<JsonObject>();
    public List<V2Dump> Dumps = new List<V2Dump>();
    public List<V2Combat> Combats = new List<V2Combat>();

    /// <summary>selection_result records in decisions.jsonl whose of_seq does not point at a selection record (validation report P2).</summary>
    public int StraySelectionResults;

    public static bool IsTraceDir(string dir) => File.Exists(Path.Combine(dir, "summary.json")) && File.Exists(Path.Combine(dir, "decisions.jsonl"));

    public static V2Trace Load(string dir)
    {
        JsonNode summary = JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "summary.json")))!;
        string format = summary["format"]?.GetValue<string>() ?? "";
        if (format != "spireoracle-run-trace-v2")
        {
            throw new InvalidDataException("not a v2 run trace (format '" + format + "'): " + dir);
        }
        V2Trace t = new V2Trace
        {
            Dir = dir,
            Name = Path.GetFileName(dir.TrimEnd('\\', '/')),
            Summary = summary,
            Character = summary["character"]!.GetValue<string>(),
            Ascension = summary["ascension"]!.GetValue<int>(),
            Seed = summary["seed"]!.GetValue<string>(),
            Set = summary["mode"]?.GetValue<string>() ?? "",
            End = summary["end"]?.GetValue<string>() ?? "",
            ModVersion = summary["mod_version"]?.GetValue<string>() ?? "",
        };
        // The fixture exactly as the header declares it (oracle-deep.md section 4): the only declared rule change.
        if (summary["fixture"] is JsonObject fx)
        {
            string name = fx["name"]?.GetValue<string>() ?? "";
            if (name != "forced_hp")
            {
                throw new InvalidDataException("trace " + t.Name + " declares an unknown fixture '" + name + "'");
            }
            t.FixtureDeclared = true;
            t.ForcedHp = fx["value"]!.GetValue<int>();
        }
        if ((t.Set == "deep") != t.FixtureDeclared)
        {
            throw new InvalidDataException("trace " + t.Name + ": mode '" + t.Set + "' but fixture " + (t.FixtureDeclared ? "declared" : "absent"));
        }

        // ---- decisions.jsonl ----
        foreach (string line in File.ReadLines(Path.Combine(dir, "decisions.jsonl")))
        {
            if (line.Length == 0)
            {
                continue;
            }
            t.Records.Add(JsonNode.Parse(line)!.AsObject());
        }
        for (int i = 0; i < t.Records.Count; i++)
        {
            if (t.Records[i]["seq"]!.GetValue<int>() != i)
            {
                throw new InvalidDataException("trace " + t.Name + ": record " + i + " has seq " + t.Records[i]["seq"]);
            }
        }
        // selection_result / crystal_result -> attached to the record they describe (of_seq)
        for (int i = 0; i < t.Records.Count; i++)
        {
            JsonObject r = t.Records[i];
            string kind = r["kind"]!.GetValue<string>();
            if (kind != "selection_result" && kind != "crystal_result")
            {
                continue;
            }
            int of = r["of_seq"]?.GetValue<int>() ?? -1;
            string want = kind == "selection_result" ? "selection" : "crystal";
            if (of >= 0 && of < i && t.Records[of]["kind"]!.GetValue<string>() == want && !t.Records[of].ContainsKey("_result"))
            {
                t.Records[of]["_result"] = r.DeepClone();
                r["_attached"] = true;      // written right after the record it describes, before anything else happens
            }
            else
            {
                t.StraySelectionResults++;
            }
        }

        // ---- dumps ----
        foreach (string f in Directory.GetFiles(dir, "*.json").OrderBy(f => f, StringComparer.Ordinal))
        {
            string name = Path.GetFileName(f);
            if (name.EndsWith(".save.json", StringComparison.Ordinal) || name.EndsWith(".combat.json", StringComparison.Ordinal) || name == "summary.json")
            {
                continue;
            }
            JsonObject d = JsonNode.Parse(File.ReadAllText(f))!.AsObject();
            string save = Path.Combine(dir, Path.GetFileNameWithoutExtension(f) + ".save.json");
            t.Dumps.Add(new V2Dump { Index = d["index"]!.GetValue<int>(), Kind = d["kind"]!.GetValue<string>(), File = name, Dump = d, SavePath = File.Exists(save) ? save : null });
        }
        for (int i = 0; i < t.Dumps.Count; i++)
        {
            if (t.Dumps[i].Index != i)
            {
                throw new InvalidDataException("trace " + t.Name + ": dump files are not contiguous at " + i);
            }
        }

        // ---- combats ----
        List<string> orphanLoss = Directory.GetFiles(dir, "*-combat-final-loss.mcr").ToList();
        JsonArray list = summary["combat_list"]!.AsArray();
        for (int i = 0; i < list.Count; i++)
        {
            JsonNode c = list[i]!;
            string file = c["file"]!.GetValue<string>();
            string path = Path.Combine(dir, file);
            JsonNode body = JsonNode.Parse(File.ReadAllText(path))!;
            V2Combat combat = new V2Combat
            {
                CombatIndex = c["combat_index"]!.GetValue<int>(),
                Floor = c["floor"]!.GetValue<int>(),
                Encounter = c["encounter"]?.GetValue<string>() ?? "",
                RoomType = c["room_type"]?.GetValue<string>() ?? "",
                End = c["end"]?.GetValue<string>() ?? "",
                File = file,
                Tag = int.Parse(file.Substring(0, 3)),
                Actions = body["Actions"]!.AsArray(),
            };
            if (body["prompt_records"] is JsonArray prompts)
            {
                List<JsonObject> all = prompts.Select(p => p!.AsObject()).ToList();
                for (int k = 0; k < all.Count; k++)
                {
                    JsonObject p = all[k];
                    string kind = p["kind"]!.GetValue<string>();
                    if (kind == "selection_result")
                    {
                        int of = p["of_seq"]?.GetValue<int>() ?? -1;
                        if (of >= 0 && of < k && all[of]["kind"]!.GetValue<string>() == "selection" && !all[of].ContainsKey("_result"))
                        {
                            all[of]["_result"] = p.DeepClone();
                        }
                        else
                        {
                            combat.OrphanResults++;
                        }
                        continue;
                    }
                    combat.Prompts.Add(p);
                }
            }
            string mcr = Path.Combine(dir, Path.GetFileNameWithoutExtension(Path.GetFileNameWithoutExtension(file)) + ".mcr");
            if (File.Exists(mcr))
            {
                combat.McrPath = mcr;
            }
            else if (combat.End == "loss" && i == list.Count - 1 && orphanLoss.Count == 1)
            {
                // The fatal fight: the client writes its replay when the run is left (RunManager.CleanUp), and the tracer stores it as
                // <end dump index + 1>-combat-final-loss.mcr (RunTracer2.TraceRun), while the fight's combat.json keeps its own tag.
                combat.McrPath = orphanLoss[0];
            }
            t.Combats.Add(combat);
        }
        return t;
    }

    /// <summary>
    /// The collector's sidecar of a run-trace replay file (verifier finding F2): NNN-combat-fN.mcr -> NNN-combat-fN.combat.json;
    /// NNN-combat-final-loss.mcr -> the trace's LAST fight when that fight was lost (its combat.json has another tag and no .mcr twin).
    /// </summary>
    public static string? SidecarOf(string mcrPath)
    {
        string direct = Path.ChangeExtension(mcrPath, ".combat.json");
        if (File.Exists(direct))
        {
            return direct;
        }
        if (!Path.GetFileName(mcrPath).EndsWith("-combat-final-loss.mcr", StringComparison.Ordinal))
        {
            return null;
        }
        string dir = Path.GetDirectoryName(mcrPath)!;
        string? last = Directory.GetFiles(dir, "*.combat.json").OrderBy(f => f, StringComparer.Ordinal).LastOrDefault();
        if (last == null || File.Exists(Path.Combine(dir, Path.GetFileNameWithoutExtension(Path.GetFileNameWithoutExtension(last)) + ".mcr")))
        {
            return null;
        }
        JsonNode? body = JsonNode.Parse(File.ReadAllText(last));
        return body?["End"]?.GetValue<string>() == "loss" ? last : null;
    }
}
