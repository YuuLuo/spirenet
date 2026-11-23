using System;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Encoding;

/// <summary>
/// One reusable frame buffer (obs-schema-v1 section 3.2): 16 row-major int32 tables that grow on demand,
/// the 64-word GLOBAL vector, and a build step that lays out header + global + directory + table data with
/// the per-table FNV-1a hashes. Nothing is allocated per frame once the buffers have reached their
/// steady-state size.
///
/// VARS rows are collected with <see cref="AddVar"/> in any order and sorted at build time by
/// (owner_table, owner_row, target_row, var_id); the owners' var_off / var_cnt are filled in then.
/// Rows beyond the v1 caps are dropped at build time and FrameFlag.Overflow is set (the episode must be
/// discarded by the consumer, schema 11).
/// </summary>
public sealed class FrameWriter
{
    private static readonly int[] VarOffCol = BuildVarOffCols();

    private readonly int[][] _data = new int[SchemaV1.NTables][];
    private readonly int[] _rows = new int[SchemaV1.NTables];

    // extension table FORECAST (SchemaX1): written only by Build(forecast: true)
    private int[] _forecast = new int[SchemaX1.ForecastNCols * 32];
    private int _forecastRows;

    // calc trailer tables TURN / FOE / ROUTE (SchemaX1, notes/calc/contract.md 2.2): written only by Build(calc: true);
    // fixed capacity = the row caps, a row beyond a cap is an encoder bug (the tables never overflow)
    private static readonly int[] CalcIds = { SchemaX1.TurnTable, SchemaX1.FoeTable, SchemaX1.RouteTable };
    private static readonly int[] CalcCols = { SchemaX1.TurnNCols, SchemaX1.FoeNCols, SchemaX1.RouteNCols };
    private static readonly int[] CalcCaps = { SchemaX1.TurnMaxRows, SchemaX1.FoeMaxRows, SchemaX1.RouteMaxRows };
    private readonly int[][] _calc =
    {
        new int[SchemaX1.TurnMaxRows * SchemaX1.TurnNCols], new int[SchemaX1.FoeMaxRows * SchemaX1.FoeNCols],
        new int[SchemaX1.RouteMaxRows * SchemaX1.RouteNCols],
    };
    private readonly int[] _calcRows = new int[3];

    // VARS staging: rows + sort keys
    private int[] _vars = new int[256 * 8];
    private long[] _varKeys = new long[256];
    private int[] _varOrder = new int[256];
    private int _varCount;

    private byte[] _out = new byte[64 * 1024];
    private int _outLength;

    public readonly int[] Global = new int[SchemaV1.GlobalLen];
    public readonly int[] Header = new int[SchemaV1.HeaderLen];

    public FrameWriter()
    {
        for (int t = 0; t < SchemaV1.NTables; t++)
        {
            _data[t] = new int[Math.Max(16, SchemaV1.NCols[t] * 32)];
        }
    }

    public bool Overflowed { get; private set; }

    /// <summary>The bytes of the last <see cref="Build"/> (payload, no length prefix).</summary>
    public ReadOnlySpan<byte> Payload => new ReadOnlySpan<byte>(_out, 0, _outLength);

    public byte[] PayloadBuffer => _out;

    public int PayloadLength => _outLength;

    public void Clear()
    {
        Array.Clear(_rows);
        _forecastRows = 0;
        Array.Clear(_calcRows);
        Array.Clear(Global);
        Array.Clear(Header);
        _varCount = 0;
        Overflowed = false;
        _outLength = 0;
    }

    public int Count(int table) => _rows[table];

    /// <summary>Appends a row filled with <paramref name="fill"/> and returns its index.</summary>
    public int AddRow(int table, int fill = 0)
    {
        int cols = SchemaV1.NCols[table];
        int r = _rows[table];
        int[] d = _data[table];
        if ((r + 1) * cols > d.Length)
        {
            Array.Resize(ref _data[table], d.Length * 2);
            d = _data[table];
        }
        Span<int> row = new Span<int>(d, r * cols, cols);
        row.Fill(fill);
        int vo = VarOffCol[table];
        if (vo >= 0)
        {
            row[vo] = -1;
            row[vo + 1] = 0;
        }
        _rows[table] = r + 1;
        return r;
    }

    public int ForecastCount => _forecastRows;

    /// <summary>Appends a FORECAST row filled with 0 and returns it.</summary>
    public Span<int> AddForecastRow()
    {
        int cols = SchemaX1.ForecastNCols;
        int r = _forecastRows;
        if ((r + 1) * cols > _forecast.Length)
        {
            Array.Resize(ref _forecast, _forecast.Length * 2);
        }
        Span<int> row = new Span<int>(_forecast, r * cols, cols);
        row.Clear();
        _forecastRows = r + 1;
        return row;
    }

    public Span<int> ForecastRow(int row) => new Span<int>(_forecast, row * SchemaX1.ForecastNCols, SchemaX1.ForecastNCols);

    /// <summary>Calc table index: 0 TURN, 1 FOE, 2 ROUTE.</summary>
    public const int CalcTurn = 0, CalcFoe = 1, CalcRoute = 2;

    public int CalcCount(int k) => _calcRows[k];

    /// <summary>Appends a row of calc table <paramref name="k"/> filled with 0 and returns it; a row beyond the cap throws.</summary>
    public Span<int> AddCalcRow(int k)
    {
        int r = _calcRows[k];
        if (r >= CalcCaps[k])
        {
            throw new InvalidOperationException($"calc table {CalcIds[k]}: more than {CalcCaps[k]} rows");
        }
        Span<int> row = new Span<int>(_calc[k], r * CalcCols[k], CalcCols[k]);
        row.Clear();
        _calcRows[k] = r + 1;
        return row;
    }

    public Span<int> Row(int table, int row)
    {
        int cols = SchemaV1.NCols[table];
        return new Span<int>(_data[table], row * cols, cols);
    }

    public int Get(int table, int row, int col) => _data[table][row * SchemaV1.NCols[table] + col];

    public void Set(int table, int row, int col, int value) => _data[table][row * SchemaV1.NCols[table] + col] = value;

    /// <summary>Removes rows from the end (used when a speculative row is not needed).</summary>
    public void Truncate(int table, int rows) => _rows[table] = rows;

    /// <summary>
    /// Copies <paramref name="count"/> rows of <paramref name="table"/> starting at <paramref name="first"/>
    /// in the given order (a permutation of that range) - used by canonical multiset ordering.
    /// </summary>
    public void Permute(int table, int first, ReadOnlySpan<int> order, Span<int> scratch)
    {
        int cols = SchemaV1.NCols[table];
        int n = order.Length;
        Span<int> src = new Span<int>(_data[table], first * cols, n * cols);
        src.CopyTo(scratch);
        for (int i = 0; i < n; i++)
        {
            scratch.Slice(order[i] * cols, cols).CopyTo(src.Slice(i * cols, cols));
        }
    }

    public void AddVar(int ownerTable, int ownerRow, int targetRow, int varId, int shown, int baseValue, int enchanted, int flags)
    {
        int n = _varCount;
        if ((n + 1) * 8 > _vars.Length)
        {
            Array.Resize(ref _vars, _vars.Length * 2);
            Array.Resize(ref _varKeys, _varKeys.Length * 2);
            Array.Resize(ref _varOrder, _varOrder.Length * 2);
        }
        int o = n * 8;
        _vars[o] = ownerTable;
        _vars[o + 1] = ownerRow;
        _vars[o + 2] = targetRow;
        _vars[o + 3] = varId;
        _vars[o + 4] = shown;
        _vars[o + 5] = baseValue;
        _vars[o + 6] = enchanted;
        _vars[o + 7] = flags;
        _varKeys[n] = VarKey(ownerTable, ownerRow, targetRow, varId);
        _varCount = n + 1;
    }

    public int PendingVarCount => _varCount;

    /// <summary>
    /// Canonical multiset reordering moved / merged owner rows [firstRow, firstRow+count): pending VARS rows of
    /// those owners get owner firstRow + newRel[old - firstRow], or are dropped when that is -1.
    /// </summary>
    public void RemapVars(int ownerTable, int firstRow, int count, ReadOnlySpan<int> newRel)
    {
        int w = 0;
        for (int i = 0; i < _varCount; i++)
        {
            int o = i * 8;
            int r = _vars[o + 1];
            if (_vars[o] == ownerTable && r >= firstRow && r < firstRow + count)
            {
                int nr = newRel[r - firstRow];
                if (nr < 0)
                {
                    continue; // merged into another instance of the same class
                }
                _vars[o + 1] = firstRow + nr;
                _varKeys[i] = VarKey(ownerTable, firstRow + nr, _vars[o + 2], _vars[o + 3]);
            }
            if (w != i)
            {
                Array.Copy(_vars, o, _vars, w * 8, 8);
                _varKeys[w] = _varKeys[i];
            }
            w++;
        }
        _varCount = w;
    }

    /// <summary>True if a pending VARS row (owner, target, var_id) exists.</summary>
    public bool HasVar(int ownerTable, int ownerRow, int targetRow, int varId)
    {
        long k = VarKey(ownerTable, ownerRow, targetRow, varId);
        for (int i = _varCount - 1; i >= 0; i--)
        {
            if (_varKeys[i] == k)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Pending VARS rows of one owner (unsorted), for content keys: (row index in staging) list.</summary>
    public int CollectVarsOf(int ownerTable, int ownerRow, Span<int> outIdx)
    {
        int k = 0;
        for (int i = 0; i < _varCount; i++)
        {
            int o = i * 8;
            if (_vars[o] == ownerTable && _vars[o + 1] == ownerRow)
            {
                outIdx[k++] = i;
            }
        }
        return k;
    }

    public ReadOnlySpan<int> StagedVar(int index) => new ReadOnlySpan<int>(_vars, index * 8, 8);

    private static long VarKey(int ownerTable, int ownerRow, int targetRow, int varId)
    {
        return ((long)ownerTable << 52) | ((long)(ownerRow & 0xFFFFF) << 32) | ((long)((targetRow + 1) & 0xFFFF) << 16) | (long)(varId & 0xFFFF);
    }

    private static int[] BuildVarOffCols()
    {
        int[] a = new int[SchemaV1.NTables];
        Array.Fill(a, -1);
        a[T.Cards] = CardC.VarOff;
        a[T.Powers] = PowerC.VarOff;
        a[T.Relics] = RelicC.VarOff;
        a[T.Potions] = PotionC.VarOff;
        a[T.Options] = OptionC.VarOff;
        return a;
    }

    /// <summary>
    /// Lays out the frame. Header fields other than layout ones must be set in <see cref="Header"/> before. With
    /// <paramref name="forecast"/> the FORECAST trailer follows the v1 data (SchemaX1): bytes [0, X0) are those of the same
    /// frame without it except header words FRAME_BYTES and N_TABLES.
    /// </summary>
    public void Build(bool forecast, bool calc)
    {
        if (!forecast && _forecastRows != 0)
        {
            throw new InvalidOperationException("FORECAST rows in a frame without the forecast trailer");
        }
        if (!calc && (_calcRows[0] | _calcRows[1] | _calcRows[2]) != 0)
        {
            throw new InvalidOperationException("calc rows in a frame without the calc trailer");
        }
        // ---- VARS: sort staged rows, write the table, fill owner slices ----
        int nv = _varCount;
        for (int i = 0; i < nv; i++)
        {
            _varOrder[i] = i;
        }
        // keys are unique per (owner, target, var): an unstable sort is deterministic; a duplicate is an encoder bug
        if (nv > 1)
        {
            Array.Sort(_varKeys, _varOrder, 0, nv);
            for (int i = 1; i < nv; i++)
            {
                if (_varKeys[i] == _varKeys[i - 1])
                {
                    int s = _varOrder[i] * 8;
                    throw new InvalidOperationException($"duplicate VARS row: owner {SchemaV1.TableNames[_vars[s]]}[{_vars[s + 1]}] target {_vars[s + 2]} var {_vars[s + 3]}");
                }
            }
        }
        _rows[T.Vars] = 0;
        int lastOwnerTable = -1, lastOwnerRow = -1, sliceStart = 0;
        for (int i = 0; i < nv; i++)
        {
            int src = _varOrder[i] * 8;
            int r = AddRow(T.Vars);
            Span<int> row = Row(T.Vars, r);
            new ReadOnlySpan<int>(_vars, src, 8).CopyTo(row);
            int ot = row[VarC.OwnerTable], or = row[VarC.OwnerRow];
            if (ot != lastOwnerTable || or != lastOwnerRow)
            {
                CloseSlice(lastOwnerTable, lastOwnerRow, sliceStart, r);
                lastOwnerTable = ot;
                lastOwnerRow = or;
                sliceStart = r;
            }
        }
        CloseSlice(lastOwnerTable, lastOwnerRow, sliceStart, _rows[T.Vars]);

        // ---- caps ----
        for (int t = 0; t < SchemaV1.NTables; t++)
        {
            if (_rows[t] > SchemaV1.MaxRows[t])
            {
                _rows[t] = SchemaV1.MaxRows[t];
                Overflowed = true;
            }
        }
        if (_forecastRows > SchemaX1.ForecastMaxRows)
        {
            _forecastRows = SchemaX1.ForecastMaxRows;
            Overflowed = true;
        }
        if (Overflowed)
        {
            Header[Hd.Flags] |= FrameFlag.Overflow;
        }

        // ---- layout ----
        int words = SchemaV1.DataOffsetWords;
        for (int t = 0; t < SchemaV1.NTables; t++)
        {
            words += _rows[t] * SchemaV1.NCols[t];
        }
        int x0 = words;
        if (forecast)
        {
            words += SchemaV1.DirStride + _forecastRows * SchemaX1.ForecastNCols;
        }
        int x1 = words;
        if (calc)
        {
            for (int k = 0; k < 3; k++)
            {
                words += SchemaV1.DirStride + _calcRows[k] * CalcCols[k];
            }
        }
        int bytes = words * 4;
        if (_out.Length < bytes)
        {
            _out = new byte[Math.Max(bytes, _out.Length * 2)];
        }
        Span<int> o = MemoryMarshal.Cast<byte, int>(new Span<byte>(_out, 0, bytes));
        Header[Hd.Magic] = SchemaV1.FrameMagic;
        Header[Hd.SchemaVersion] = SchemaV1.Version;
        Header[Hd.FrameBytes] = bytes;
        Header[Hd.NTables] = SchemaV1.NTables + (forecast ? 1 : 0) + (calc ? 3 : 0);
        Header[Hd.GlobalLen] = SchemaV1.GlobalLen;
        Header[Hd.NActions] = _rows[T.Actions];
        int pos = SchemaV1.DataOffsetWords;
        for (int t = 0; t < SchemaV1.NTables; t++)
        {
            int n = _rows[t] * SchemaV1.NCols[t];
            Span<int> dst = o.Slice(pos, n);
            new ReadOnlySpan<int>(_data[t], 0, n).CopyTo(dst);
            uint h = Fnv1a(MemoryMarshal.AsBytes(dst));
            int d = SchemaV1.DirOffsetWords + t * SchemaV1.DirStride;
            o[d] = t;
            o[d + 1] = _rows[t];
            o[d + 2] = SchemaV1.NCols[t];
            o[d + 3] = pos * 4;
            o[d + 4] = unchecked((int)h);
            pos += n;
        }
        if (forecast)
        {
            int n = _forecastRows * SchemaX1.ForecastNCols;
            Span<int> dst = o.Slice(x0 + SchemaV1.DirStride, n);
            new ReadOnlySpan<int>(_forecast, 0, n).CopyTo(dst);
            o[x0] = SchemaX1.ForecastTable;
            o[x0 + 1] = _forecastRows;
            o[x0 + 2] = SchemaX1.ForecastNCols;
            o[x0 + 3] = (x0 + SchemaV1.DirStride) * 4;
            o[x0 + 4] = unchecked((int)Fnv1a(MemoryMarshal.AsBytes(dst)));
        }
        if (calc)
        {
            int e = x1;
            for (int k = 0; k < 3; k++)
            {
                int n = _calcRows[k] * CalcCols[k];
                Span<int> dst = o.Slice(e + SchemaV1.DirStride, n);
                new ReadOnlySpan<int>(_calc[k], 0, n).CopyTo(dst);
                o[e] = CalcIds[k];
                o[e + 1] = _calcRows[k];
                o[e + 2] = CalcCols[k];
                o[e + 3] = (e + SchemaV1.DirStride) * 4;
                o[e + 4] = unchecked((int)Fnv1a(MemoryMarshal.AsBytes(dst)));
                e += SchemaV1.DirStride + n;
            }
        }
        Global.AsSpan().CopyTo(o.Slice(SchemaV1.GlobalOffsetWords, SchemaV1.GlobalLen));
        Header.AsSpan().CopyTo(o.Slice(0, SchemaV1.HeaderLen));
        if (!BitConverter.IsLittleEndian)
        {
            throw new PlatformNotSupportedException("big-endian host");
        }
        _outLength = bytes;
    }

    private void CloseSlice(int ownerTable, int ownerRow, int start, int end)
    {
        if (ownerTable < 0)
        {
            return;
        }
        int vo = VarOffCol[ownerTable];
        if (vo < 0 || ownerRow >= _rows[ownerTable])
        {
            throw new InvalidOperationException($"VARS owner {SchemaV1.TableNames[ownerTable]}[{ownerRow}] does not exist / cannot own VARS");
        }
        Span<int> row = Row(ownerTable, ownerRow);
        row[vo] = start;
        row[vo + 1] = end - start;
    }

    /// <summary>Patches one header word in the already built payload (RESULT / STEP_ID / AUTO_STEPS / HOST_US ...).</summary>
    public void PatchHeader(int index, int value)
    {
        Header[index] = value;
        BinaryPrimitives.WriteInt32LittleEndian(new Span<byte>(_out, index * 4, 4), value);
    }

    /// <summary>Patches one GLOBAL word in the already built payload (GLOBAL is not covered by any hash).</summary>
    public void PatchGlobal(int index, int value)
    {
        Global[index] = value;
        BinaryPrimitives.WriteInt32LittleEndian(new Span<byte>(_out, (SchemaV1.GlobalOffsetWords + index) * 4, 4), value);
    }

    public static uint Fnv1a(ReadOnlySpan<byte> data)
    {
        uint h = SchemaV1.FnvOffsetBasis;
        for (int i = 0; i < data.Length; i++)
        {
            h ^= data[i];
            h *= SchemaV1.FnvPrime;
        }
        return h;
    }
}
