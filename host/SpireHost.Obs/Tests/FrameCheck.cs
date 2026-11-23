using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using SpireHost.Obs.Encoding;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Tests;

/// <summary>
/// Independent C# reader of a frame (does not use FrameWriter / ObsEncoder internals): layout + directory + hashes,
/// pointer and CSR-slice bounds and ownership, canonical orders (VARS, ACTIONS, MAP, CELLS, multiset zones),
/// SEATS rows. The C# twin of py/spire/proto.frame_problems (schema 14.5), run on every frame of the self-test.
/// </summary>
public static class FrameCheck
{
    private static readonly int[] KeyCols =
    {
        CardC.Zone, CardC.Seat, CardC.CardId, CardC.UpgradeLevel, CardC.EnergyCost, CardC.CostFlags, CardC.EnergyCostColor,
        CardC.StarCost, CardC.StarCostColor, CardC.KeywordBits, CardC.EnchantId, CardC.EnchantAmount, CardC.EnchantStatus,
        CardC.AfflictionId, CardC.AfflictionAmount, CardC.ReplayCount, CardC.HandFlags, CardC.UnplayableReason, CardC.Aux,
    };

    public sealed class View
    {
        public int[] W = Array.Empty<int>();
        public int[] Rows = new int[16];
        public int[] Off = new int[16];

        public int Get(int t, int r, int c) => W[Off[t] + r * SchemaV1.NCols[t] + c];

        public int H(int i) => W[i];

        public int G(int i) => W[SchemaV1.GlobalOffsetWords + i];
    }

    public static List<string> Problems(ReadOnlySpan<byte> payload, out View v)
    {
        List<string> p = new List<string>();
        v = new View();
        if (payload.Length < SchemaV1.DataOffsetWords * 4 || payload.Length % 4 != 0)
        {
            p.Add("frame too short / unaligned");
            return p;
        }
        v.W = MemoryMarshal.Cast<byte, int>(payload).ToArray();
        int[] w = v.W;
        if (w[Hd.Magic] != SchemaV1.FrameMagic || w[Hd.SchemaVersion] != 1 || w[Hd.NTables] != 16 || w[Hd.GlobalLen] != 64)
        {
            p.Add("bad header");
        }
        if (w[Hd.FrameBytes] != payload.Length)
        {
            p.Add("frame_bytes mismatch");
        }
        int expect = SchemaV1.DataOffsetWords * 4;
        for (int t = 0; t < 16; t++)
        {
            int d = SchemaV1.DirOffsetWords + t * 5;
            if (w[d] != t || w[d + 2] != SchemaV1.NCols[t])
            {
                p.Add($"directory entry {t} malformed");
            }
            int n = w[d + 1];
            if (n < 0 || n > SchemaV1.MaxRows[t])
            {
                p.Add($"{SchemaV1.TableNames[t]}: n_rows {n}");
                n = Math.Max(0, Math.Min(n, SchemaV1.MaxRows[t]));
            }
            if (w[d + 3] != expect)
            {
                p.Add($"{SchemaV1.TableNames[t]}: offset {w[d + 3]} != {expect}");
            }
            v.Rows[t] = n;
            v.Off[t] = w[d + 3] / 4;
            int bytes = n * SchemaV1.NCols[t] * 4;
            if (w[d + 3] + bytes <= payload.Length)
            {
                uint h = FrameWriter.Fnv1a(payload.Slice(w[d + 3], bytes));
                if (unchecked((int)h) != w[d + 4])
                {
                    p.Add($"{SchemaV1.TableNames[t]}: hash mismatch");
                }
            }
            expect += bytes;
        }
        if (expect != payload.Length)
        {
            p.Add("table data does not end at frame end");
            return p;
        }
        int msg = w[Hd.MsgType];
        if (w[Hd.NActions] != v.Rows[T.Actions])
        {
            p.Add("n_actions != ACTIONS rows");
        }
        if (msg == MsgType.Obs && (v.Rows[T.Actions] == 0 || w[Hd.Result] != 0 || (w[Hd.Flags] & FrameFlag.Terminal) != 0))
        {
            p.Add("OBS frame without actions / with result / terminal flag");
        }
        if (msg == MsgType.Terminal && (v.Rows[T.Actions] != 0 || (w[Hd.Flags] & FrameFlag.Terminal) == 0))
        {
            p.Add("TERMINAL frame with actions / without terminal flag");
        }
        Pointers(v, p);
        Slices(v, p);
        Orders(v, p);
        return p;
    }

    private static void Ptr(View v, List<string> p, int t, int col, int target, bool allowNull = true)
    {
        for (int r = 0; r < v.Rows[t]; r++)
        {
            int x = v.Get(t, r, col);
            if (x == -1 && allowNull)
            {
                continue;
            }
            if (x < 0 || x >= v.Rows[target])
            {
                p.Add($"{SchemaV1.TableNames[t]}[{r}].{col} -> {x} outside {SchemaV1.TableNames[target]}");
                return;
            }
        }
    }

    private static void Pointers(View v, List<string> p)
    {
        Ptr(v, p, T.Seats, SeatC.CreatureRow, T.Creatures);
        Ptr(v, p, T.Cards, CardC.LinkRow, T.Cards);
        Ptr(v, p, T.Vars, VarC.TargetRow, T.Creatures);
        Ptr(v, p, T.Powers, PowerC.OwnerRow, T.Creatures, false);
        Ptr(v, p, T.Powers, PowerC.ApplierRow, T.Creatures);
        Ptr(v, p, T.Intents, IntentC.OwnerRow, T.Creatures, false);
        Ptr(v, p, T.MapEdges, MapEdgeC.From, T.MapNodes, false);
        Ptr(v, p, T.MapEdges, MapEdgeC.To, T.MapNodes, false);
        Ptr(v, p, T.Actions, ActionC.TgtRow, T.Creatures);
        Dyn(v, p, T.Vars, VarC.OwnerTable, VarC.OwnerRow);
        Dyn(v, p, T.Options, OptionC.RefTable, OptionC.RefRow);
        Dyn(v, p, T.Refs, RefC.OwnerTable, RefC.OwnerRow);
        Dyn(v, p, T.Refs, RefC.EntTable, RefC.EntRow);
        Dyn(v, p, T.Actions, ActionC.SrcTable, ActionC.SrcRow);
    }

    private static void Dyn(View v, List<string> p, int t, int tcol, int rcol)
    {
        for (int r = 0; r < v.Rows[t]; r++)
        {
            int tt = v.Get(t, r, tcol), rr = v.Get(t, r, rcol);
            if (tt == -1)
            {
                continue;
            }
            if (tt < 0 || tt >= 16 || rr < 0 || rr >= v.Rows[tt])
            {
                p.Add($"{SchemaV1.TableNames[t]}[{r}] dynamic pointer ({tt},{rr}) out of range");
                return;
            }
        }
    }

    private static void Slices(View v, List<string> p)
    {
        (int owner, int off, int cnt, int child, int ownRow, int ownTab)[] specs =
        {
            (T.Cards, CardC.VarOff, CardC.VarCnt, T.Vars, VarC.OwnerRow, VarC.OwnerTable),
            (T.Creatures, CreatureC.PowerOff, CreatureC.PowerCnt, T.Powers, PowerC.OwnerRow, -1),
            (T.Creatures, CreatureC.IntentOff, CreatureC.IntentCnt, T.Intents, IntentC.OwnerRow, -1),
            (T.Powers, PowerC.VarOff, PowerC.VarCnt, T.Vars, VarC.OwnerRow, VarC.OwnerTable),
            (T.Relics, RelicC.VarOff, RelicC.VarCnt, T.Vars, VarC.OwnerRow, VarC.OwnerTable),
            (T.Potions, PotionC.VarOff, PotionC.VarCnt, T.Vars, VarC.OwnerRow, VarC.OwnerTable),
            (T.Options, OptionC.VarOff, OptionC.VarCnt, T.Vars, VarC.OwnerRow, VarC.OwnerTable),
            (T.Options, OptionC.RefOff, OptionC.RefCnt, T.Refs, RefC.OwnerRow, RefC.OwnerTable),
        };
        Dictionary<int, int[]> claimed = new Dictionary<int, int[]>
        {
            [T.Vars] = new int[v.Rows[T.Vars]],
            [T.Powers] = new int[v.Rows[T.Powers]],
            [T.Intents] = new int[v.Rows[T.Intents]],
            [T.Refs] = new int[v.Rows[T.Refs]],
        };
        foreach (var s in specs)
        {
            for (int r = 0; r < v.Rows[s.owner]; r++)
            {
                int off = v.Get(s.owner, r, s.off), cnt = v.Get(s.owner, r, s.cnt);
                if (cnt == 0)
                {
                    if (off != -1)
                    {
                        p.Add($"{SchemaV1.TableNames[s.owner]}[{r}] cnt 0 with off {off}");
                    }
                    continue;
                }
                if (cnt < 0 || off < 0 || off + cnt > v.Rows[s.child])
                {
                    p.Add($"{SchemaV1.TableNames[s.owner]}[{r}] slice out of bounds");
                    continue;
                }
                for (int k = off; k < off + cnt; k++)
                {
                    claimed[s.child][k]++;
                    if (v.Get(s.child, k, s.ownRow) != r || (s.ownTab >= 0 && v.Get(s.child, k, s.ownTab) != s.owner))
                    {
                        p.Add($"{SchemaV1.TableNames[s.child]}[{k}] does not name its owner {SchemaV1.TableNames[s.owner]}[{r}]");
                        break;
                    }
                }
            }
        }
        foreach (var kv in claimed)
        {
            for (int k = 0; k < kv.Value.Length; k++)
            {
                if (kv.Value[k] != 1)
                {
                    p.Add($"{SchemaV1.TableNames[kv.Key]}[{k}] claimed {kv.Value[k]} times");
                    break;
                }
            }
        }
    }

    private static int CompareRows(View v, int t, int a, int b, int[] cols)
    {
        foreach (int c in cols)
        {
            int k = v.Get(t, a, c).CompareTo(v.Get(t, b, c));
            if (k != 0)
            {
                return k;
            }
        }
        return 0;
    }

    private static void Orders(View v, List<string> p)
    {
        int[] varKey = { VarC.OwnerTable, VarC.OwnerRow, VarC.TargetRow, VarC.VarId };
        for (int r = 1; r < v.Rows[T.Vars]; r++)
        {
            if (CompareRows(v, T.Vars, r - 1, r, varKey) >= 0)
            {
                p.Add("VARS not strictly ascending");
                break;
            }
        }
        int[] actKey = { 0, 1, 2, 3, 4, 5 };
        for (int r = 1; r < v.Rows[T.Actions]; r++)
        {
            if (CompareRows(v, T.Actions, r - 1, r, actKey) >= 0)
            {
                p.Add("ACTIONS not strictly ascending");
                break;
            }
        }
        int[] edgeKey = { 0, 1 };
        for (int r = 1; r < v.Rows[T.MapEdges]; r++)
        {
            if (CompareRows(v, T.MapEdges, r - 1, r, edgeKey) >= 0)
            {
                p.Add("MAP_EDGES not strictly ascending");
                break;
            }
        }
        int nodes = v.Rows[T.MapNodes];
        int lastRegular = -1;
        bool bossSeen = false;
        for (int r = 0; r < nodes; r++)
        {
            bool boss = (v.Get(T.MapNodes, r, MapNodeC.Flags) & (MapNodeFlag.Boss | MapNodeFlag.SecondBoss)) != 0;
            if (boss)
            {
                bossSeen = true;
                continue;
            }
            if (bossSeen)
            {
                p.Add("MAP_NODES: boss nodes not last");
                break;
            }
            if (lastRegular >= 0 && CompareRows(v, T.MapNodes, lastRegular, r, new[] { 0, 1 }) >= 0)
            {
                p.Add("MAP_NODES not ascending");
                break;
            }
            lastRegular = r;
        }
        for (int r = 1; r < v.Rows[T.Cells]; r++)
        {
            if (CompareRows(v, T.Cells, r - 1, r, new[] { CellC.Y, CellC.X }) >= 0)
            {
                p.Add("CELLS not ascending");
                break;
            }
        }
        int seats = v.Rows[T.Seats];
        int msg = v.H(Hd.MsgType);
        if (msg != MsgType.Error && seats != v.G(Gl.NSeats))
        {
            p.Add("SEATS rows != n_seats");
        }
        for (int s = 0; s < seats; s++)
        {
            if (v.Get(T.Seats, s, SeatC.Seat) != s)
            {
                p.Add("SEATS not in seat order");
            }
        }
        for (int r = 0; r < v.Rows[T.Actions]; r++)
        {
            int seat = v.Get(T.Actions, r, ActionC.Seat);
            if (seat < 0 || seat >= seats)
            {
                p.Add("ACTIONS seat out of range");
                break;
            }
        }
        // multiset zones: strictly ascending content keys per (zone, seat)
        Dictionary<long, int> last = new Dictionary<long, int>();
        for (int r = 0; r < v.Rows[T.Cards]; r++)
        {
            int zone = v.Get(T.Cards, r, CardC.Zone);
            bool dedup = zone is Zone.Draw or Zone.Discard or Zone.Exhaust or Zone.Deck || (zone == Zone.Choice && v.Get(T.Cards, r, CardC.Pos) == -1);
            if (!dedup)
            {
                continue;
            }
            if (v.Get(T.Cards, r, CardC.Count) < 1 || v.Get(T.Cards, r, CardC.Pos) != -1)
            {
                p.Add($"CARDS[{r}] multiset row with count < 1 or pos != -1");
            }
            long g = ((long)zone << 32) | (uint)v.Get(T.Cards, r, CardC.Seat);
            if (last.TryGetValue(g, out int prev) && CompareCardKeys(v, prev, r) >= 0)
            {
                p.Add($"CARDS[{r}] multiset zone {zone} not strictly ascending");
            }
            last[g] = r;
        }
    }

    private static List<int> CardKey(View v, int r)
    {
        List<int> k = new List<int>(40);
        foreach (int c in KeyCols)
        {
            k.Add(v.Get(T.Cards, r, c));
        }
        int off = v.Get(T.Cards, r, CardC.VarOff), cnt = v.Get(T.Cards, r, CardC.VarCnt);
        for (int i = 0; i < cnt; i++)
        {
            int vr = off + i; // VARS rows of one owner are already sorted by (target_row, var_id)
            k.Add(v.Get(T.Vars, vr, VarC.VarId));
            k.Add(v.Get(T.Vars, vr, VarC.TargetRow));
            k.Add(v.Get(T.Vars, vr, VarC.Shown));
            k.Add(v.Get(T.Vars, vr, VarC.Base));
            k.Add(v.Get(T.Vars, vr, VarC.Enchanted));
            k.Add(v.Get(T.Vars, vr, VarC.Flags));
        }
        return k;
    }

    private static int CompareCardKeys(View v, int a, int b)
    {
        List<int> x = CardKey(v, a), y = CardKey(v, b);
        for (int i = 0; i < Math.Min(x.Count, y.Count); i++)
        {
            int c = x[i].CompareTo(y[i]);
            if (c != 0)
            {
                return c;
            }
        }
        return x.Count.CompareTo(y.Count);
    }
}
