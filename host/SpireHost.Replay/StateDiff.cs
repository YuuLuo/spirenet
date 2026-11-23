using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Entities.Multiplayer;

namespace SpireHost.Replay;

/// <summary>
/// Field-level diff of two NetFullCombatState values (the checksum input). Both sides are rendered with the
/// game's own NetFullCombatState.ToString() (the same dump the game logs on "Replay state divergence!",
/// ChecksumTracker.cs:266) and compared line by line with an LCS diff. The recorded side is the fullState that
/// the real client stored next to every checksum (ReplayChecksumData.fullState), so the diff is exact: both
/// sides carry the anonymised player id of the replay save.
/// </summary>
public static class StateDiff
{
    public sealed class Result
    {
        public List<string> Lines = new List<string>();   // "-recorded" / "+host" lines with a "@<line>" position
        public int Removed;
        public int Added;
        public List<string> Ids = new List<string>();     // CARD./POWER./RELIC./MONSTER./ORB./POTION. ids that occur on differing lines
    }

    private static readonly Regex IdPattern = new Regex(@"\b(CARD|POWER|RELIC|MONSTER|ORB|POTION|ENCHANTMENT|AFFLICTION|CHARACTER)\.[A-Z0-9_]+", RegexOptions.Compiled);

    public static Result Diff(NetFullCombatState recorded, NetFullCombatState host, int maxLines = 60)
    {
        string[] a = Split(recorded.ToString());
        string[] b = Split(host.ToString());
        Result r = new Result();
        int n = a.Length, m = b.Length;
        // LCS table; dumps are a few hundred lines, so O(n*m) ints is fine (only computed on the first divergence).
        int[,] lcs = new int[n + 1, m + 1];
        for (int i = n - 1; i >= 0; i--)
        {
            for (int j = m - 1; j >= 0; j--)
            {
                lcs[i, j] = a[i] == b[j] ? lcs[i + 1, j + 1] + 1 : Math.Max(lcs[i + 1, j], lcs[i, j + 1]);
            }
        }
        int x = 0, y = 0;
        HashSet<string> ids = new HashSet<string>(StringComparer.Ordinal);
        void Add(string line)
        {
            if (r.Lines.Count < maxLines)
            {
                r.Lines.Add(line);
            }
            foreach (Match mt in IdPattern.Matches(line))
            {
                ids.Add(mt.Value);
            }
        }
        while (x < n || y < m)
        {
            if (x < n && y < m && a[x] == b[y])
            {
                x++;
                y++;
            }
            else if (y < m && (x == n || lcs[x, y + 1] >= lcs[x + 1, y]))
            {
                r.Added++;
                Add("+host     @" + y + " " + b[y].TrimEnd());
                y++;
            }
            else
            {
                r.Removed++;
                Add("-recorded @" + x + " " + a[x].TrimEnd());
                x++;
            }
        }
        r.Ids.AddRange(ids);
        r.Ids.Sort(StringComparer.Ordinal);
        return r;
    }

    private static string[] Split(string s) => s.Replace("\r", "").Split('\n', StringSplitOptions.RemoveEmptyEntries);
}
