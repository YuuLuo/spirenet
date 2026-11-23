using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace SpireHost.Patches;

/// <summary>
/// Mechanical check "registry == ledger" (H1 integration, gate G1): every id of the patch registry has exactly one
/// row in host\FIDELITY.md and every row names an id that exists.
///
/// Ledger format (host\FIDELITY.md): a row is a Markdown table line whose FIRST cell is the id in backticks,
/// followed by the category and the kind:    | `id` | Category | kind | ...
/// Rows under a heading that contains "tool-only" describe read-only instrumentation that only a named
/// verification command installs; rows under any other heading are the PRODUCTION set. The check compares
///   production rows  ==  the registry right after the session was created (every command, every mode),
///   tool rows        >=  whatever a command registered afterwards,
/// the category, kind, TARGET and registered text of every row with the registry record (verifier finding R-4: the
/// target was not compared, so a patch whose target moved kept passing; a family row `prefix:*` stands for ids
/// generated per run-time target and is compared in category and kind only), and - for every site that carries the
/// patches of several PRODUCTION ids (ruling B3) - that each of their rows names the other ids. (Tool-only
/// instrumentation on a production site is counted, not treated as a shared site.)
/// </summary>
public static class PatchLedger
{
    public sealed class Row
    {
        public string Id = "";
        public string Category = "";
        public string Kind = "";

        /// <summary>Fourth cell without its backticks: the patched member / the thing written, as registered.</summary>
        public string Target = "";

        /// <summary>Fifth cell: the registered text (production rows), "installed by. registered text" (tool-only rows).</summary>
        public string What = "";
        public bool ToolOnly;
        public int Line;
        public string Text = "";
    }

    public sealed class Result
    {
        public bool Ok;
        public string Path = "";
        public string? Sha256;
        public int ProductionIds;
        public int ProductionRows;
        public int ToolIds;
        public int ToolRows;
        public List<string> MissingRows = new List<string>();        // registry id without a row
        public List<string> StaleRows = new List<string>();          // production row without a registry id
        public List<string> DuplicateRows = new List<string>();      // id with more than one row
        public List<string> WrongSection = new List<string>();       // production id listed as tool-only, or a tool id listed as production
        public List<string> FieldMismatches = new List<string>();    // category / kind / target / registered text differ from the registry
        public List<string> SharedSiteProblems = new List<string>(); // a row of a shared site does not name its partner(s)
        public int ToolInstrumentationOnProductionSites;             // sites that carry production patches AND tool-only instrumentation (information)
        public List<string> Problems = new List<string>();
    }

    /// <summary>The ledger next to the executable (a dist directory ships its own copy), else host\FIDELITY.md.</summary>
    public static string DefaultPath()
    {
        string local = System.IO.Path.Combine(AppContext.BaseDirectory, "FIDELITY.md");
        if (File.Exists(local))
        {
            return local;
        }
        return System.IO.Path.GetFullPath(System.IO.Path.Combine(SpireHost.Boot.GameAssemblies.GameDir, "..", "FIDELITY.md"));
    }

    public static List<Row> Parse(string path) => ParseLines(File.ReadAllLines(path));

    /// <summary>What a table cell holds for a registered string: the generator replaces '|' by '/' and collapses white space (gen_ledger.py cell()).</summary>
    public static string Cell(string text) => string.Join(" ", text.Replace('|', '/').Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    public static List<Row> ParseLines(string[] lines)
    {
        List<Row> rows = new List<Row>();
        bool tool = false;
        for (int n = 0; n < lines.Length; n++)
        {
            string line = lines[n];
            string t = line.TrimStart();
            if (t.StartsWith("#", StringComparison.Ordinal))
            {
                tool = t.Contains("tool-only", StringComparison.OrdinalIgnoreCase);
                continue;
            }
            if (!t.StartsWith("| `", StringComparison.Ordinal))
            {
                continue;
            }
            int s = t.IndexOf('`') + 1;
            int e = t.IndexOf('`', s);
            if (e <= s)
            {
                continue;
            }
            string[] cells = t.Split('|');
            rows.Add(new Row
            {
                Id = t.Substring(s, e - s),
                Category = cells.Length > 2 ? cells[2].Trim() : "",
                Kind = cells.Length > 3 ? cells[3].Trim() : "",
                Target = cells.Length > 4 ? cells[4].Trim().Trim('`') : "",
                What = cells.Length > 5 ? cells[5].Trim() : "",
                ToolOnly = tool,
                Line = n + 1,
                Text = line,
            });
        }
        return rows;
    }

    /// <summary>
    /// <paramref name="production"/> = the records present when the session was created; <paramref name="toolExtras"/> = what
    /// the running command registered afterwards (may be empty).
    /// </summary>
    public static Result Compare(string? path, IReadOnlyList<PatchRecord> production, IReadOnlyList<PatchRecord> toolExtras, IReadOnlyList<SharedSite> sharedSites)
    {
        Result r = new Result { Path = path ?? DefaultPath() };
        if (!File.Exists(r.Path))
        {
            r.Problems.Add("ledger not found: " + r.Path);
            return r;
        }
        r.Sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(r.Path))).ToLowerInvariant();
        return CompareRows(r, Parse(r.Path), production, toolExtras, sharedSites);
    }

    /// <summary>The comparison itself, on ledger text (the self-tests doctor a copy of the ledger in memory to prove what the check sees).</summary>
    public static Result CompareLines(string[] ledgerLines, IReadOnlyList<PatchRecord> production, IReadOnlyList<PatchRecord> toolExtras, IReadOnlyList<SharedSite> sharedSites)
    {
        return CompareRows(new Result { Path = "(in memory)" }, ParseLines(ledgerLines), production, toolExtras, sharedSites);
    }

    private static Result CompareRows(Result r, List<Row> rows, IReadOnlyList<PatchRecord> production, IReadOnlyList<PatchRecord> toolExtras, IReadOnlyList<SharedSite> sharedSites)
    {
        Dictionary<string, Row> byId = new Dictionary<string, Row>(StringComparer.Ordinal);
        foreach (Row row in rows)
        {
            if (!byId.TryAdd(row.Id, row))
            {
                r.DuplicateRows.Add(row.Id + " (lines " + byId[row.Id].Line + " and " + row.Line + ")");
            }
        }
        r.ProductionIds = production.Count;
        r.ToolIds = toolExtras.Count;
        r.ProductionRows = rows.Count(x => !x.ToolOnly);
        r.ToolRows = rows.Count(x => x.ToolOnly);

        HashSet<string> productionIds = new HashSet<string>(production.Select(p => p.Id), StringComparer.Ordinal);
        foreach (PatchRecord p in production)
        {
            if (!byId.TryGetValue(p.Id, out Row? row))
            {
                r.MissingRows.Add(p.Id);
                continue;
            }
            if (row.ToolOnly)
            {
                r.WrongSection.Add(p.Id + ": in the production registry but listed under a tool-only heading");
            }
            CheckFields(r, p, row);
        }
        foreach (PatchRecord p in toolExtras)
        {
            if (!byId.TryGetValue(p.Id, out Row? row))
            {
                // families registered per run-time target (one row for the family: `replay-coverage:*`)
                int colon = p.Id.IndexOf(':');
                if (colon > 0 && byId.TryGetValue(p.Id.Substring(0, colon + 1) + "*", out Row? family) && family.ToolOnly)
                {
                    CheckFields(r, p, family, familyRow: true);
                    continue;
                }
                r.MissingRows.Add(p.Id + " (tool)");
                continue;
            }
            if (!row.ToolOnly)
            {
                r.WrongSection.Add(p.Id + ": registered by a tool command but listed in the production sections");
            }
            CheckFields(r, p, row);
        }
        foreach (Row row in rows)
        {
            if (!row.ToolOnly && !productionIds.Contains(row.Id))
            {
                r.StaleRows.Add(row.Id + " (line " + row.Line + ")");
            }
        }
        foreach (SharedSite site in sharedSites)
        {
            // Ruling B3 is a rule of the production set. Read-only instrumentation that a verification command puts on a
            // production site afterwards (its tool-only row says what it sits on) does not make the site a shared one.
            List<string> ids = site.Ids.Where(productionIds.Contains).ToList();
            if (ids.Count < site.Ids.Count && ids.Count > 0)
            {
                r.ToolInstrumentationOnProductionSites++;
            }
            if (ids.Count < 2)
            {
                continue;
            }
            foreach (string id in ids)
            {
                if (!byId.TryGetValue(id, out Row? row))
                {
                    continue; // reported as missing above
                }
                foreach (string other in ids)
                {
                    if (other != id && !row.Text.Contains(other, StringComparison.Ordinal))
                    {
                        r.SharedSiteProblems.Add("row " + id + " does not name " + other + " (shared site " + site.Site + ")");
                    }
                }
            }
        }
        r.Ok = r.Problems.Count == 0 && r.MissingRows.Count == 0 && r.StaleRows.Count == 0 && r.DuplicateRows.Count == 0 && r.WrongSection.Count == 0
               && r.FieldMismatches.Count == 0 && r.SharedSiteProblems.Count == 0;
        return r;
    }

    private static void CheckFields(Result r, PatchRecord p, Row row, bool familyRow = false)
    {
        if (!string.Equals(row.Category, p.Category.ToString(), StringComparison.Ordinal))
        {
            r.FieldMismatches.Add(p.Id + ": category '" + row.Category + "' in the ledger, " + p.Category + " in the registry");
        }
        if (!string.Equals(row.Kind, p.Kind, StringComparison.Ordinal))
        {
            r.FieldMismatches.Add(p.Id + ": kind '" + row.Kind + "' in the ledger, '" + p.Kind + "' in the registry");
        }
        if (familyRow)
        {
            return; // one row for ids generated per run-time target: there is no single target / text to compare
        }
        if (!string.Equals(row.Target, Cell(p.Target), StringComparison.Ordinal))
        {
            r.FieldMismatches.Add(p.Id + ": target '" + row.Target + "' in the ledger, '" + Cell(p.Target) + "' in the registry");
        }
        // the row's text is the registered text (production rows: followed by the shared-site note; tool rows: preceded by "installed by")
        if (!row.What.Contains(Cell(p.Reason), StringComparison.Ordinal))
        {
            r.FieldMismatches.Add(p.Id + ": the row's text is not the text the code registers");
        }
    }

    /// <summary>A stable hash of a patch list: sha256 over "id|target|kind|category" lines sorted by id.</summary>
    public static string Hash(IEnumerable<PatchRecord> records)
    {
        StringBuilder sb = new StringBuilder();
        foreach (PatchRecord p in records.OrderBy(x => x.Id, StringComparer.Ordinal))
        {
            sb.Append(p.Id).Append('|').Append(p.Target).Append('|').Append(p.Kind).Append('|').Append(p.Category).Append('\n');
        }
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString()))).ToLowerInvariant();
    }
}
