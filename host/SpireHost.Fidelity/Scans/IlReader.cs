using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

namespace SpireHost.Fidelity.Scans;

/// <summary>One decoded IL instruction (offset, opcode, resolved operand).</summary>
internal sealed class Ins
{
    public int Index;
    public int Offset;
    public OpCode Op;

    /// <summary>MemberInfo for tokens, int target offset for branches, int[] for switch, boxed constants otherwise.</summary>
    public object? Operand;

    public MethodBase? Callee => Operand as MethodBase;

    public FieldInfo? Field => Operand as FieldInfo;

    public override string ToString() => $"IL_{Offset:X4} {Op.Name} {Operand}";
}

/// <summary>
/// Minimal IL decoder over System.Reflection (the assembly is already loaded in the host process; nothing is
/// written, the game file is never opened for writing). Used only by the static scans of the fidelity team.
/// </summary>
internal static class IlReader
{
    private static readonly OpCode[] _one = new OpCode[0x100];
    private static readonly OpCode[] _two = new OpCode[0x100];

    static IlReader()
    {
        foreach (FieldInfo f in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (f.GetValue(null) is OpCode op)
            {
                ushort v = (ushort)op.Value;
                if (v < 0x100)
                {
                    _one[v] = op;
                }
                else if ((v & 0xFF00) == 0xFE00)
                {
                    _two[v & 0xFF] = op;
                }
            }
        }
    }

    public static List<Ins>? Read(MethodBase m)
    {
        MethodBody? body;
        try
        {
            body = m.GetMethodBody();
        }
        catch
        {
            return null;
        }
        if (body == null)
        {
            return null;
        }
        byte[]? il = body.GetILAsByteArray();
        if (il == null)
        {
            return null;
        }
        Type[]? typeArgs = m.DeclaringType != null && m.DeclaringType.IsGenericType ? m.DeclaringType.GetGenericArguments() : null;
        Type[]? methodArgs = m is MethodInfo mi && mi.IsGenericMethod ? mi.GetGenericArguments() : null;
        Module module = m.Module;
        List<Ins> list = new List<Ins>(il.Length / 3 + 4);
        int pos = 0;
        while (pos < il.Length)
        {
            int start = pos;
            OpCode op;
            byte b = il[pos++];
            if (b == 0xFE)
            {
                op = _two[il[pos++]];
            }
            else
            {
                op = _one[b];
            }
            object? operand = null;
            switch (op.OperandType)
            {
                case OperandType.InlineNone:
                    break;
                case OperandType.ShortInlineBrTarget:
                    operand = pos + 1 + (sbyte)il[pos];
                    pos += 1;
                    break;
                case OperandType.InlineBrTarget:
                    operand = pos + 4 + BitConverter.ToInt32(il, pos);
                    pos += 4;
                    break;
                case OperandType.ShortInlineI:
                    operand = op == OpCodes.Ldc_I4_S ? (sbyte)il[pos] : il[pos];
                    pos += 1;
                    break;
                case OperandType.ShortInlineVar:
                    operand = (int)il[pos];
                    pos += 1;
                    break;
                case OperandType.InlineVar:
                    operand = (int)BitConverter.ToUInt16(il, pos);
                    pos += 2;
                    break;
                case OperandType.InlineI:
                    operand = BitConverter.ToInt32(il, pos);
                    pos += 4;
                    break;
                case OperandType.InlineI8:
                    operand = BitConverter.ToInt64(il, pos);
                    pos += 8;
                    break;
                case OperandType.ShortInlineR:
                    operand = BitConverter.ToSingle(il, pos);
                    pos += 4;
                    break;
                case OperandType.InlineR:
                    operand = BitConverter.ToDouble(il, pos);
                    pos += 8;
                    break;
                case OperandType.InlineSwitch:
                {
                    int n = BitConverter.ToInt32(il, pos);
                    pos += 4;
                    int baseOffset = pos + 4 * n;
                    int[] targets = new int[n];
                    for (int i = 0; i < n; i++)
                    {
                        targets[i] = baseOffset + BitConverter.ToInt32(il, pos);
                        pos += 4;
                    }
                    operand = targets;
                    break;
                }
                case OperandType.InlineString:
                {
                    int tok = BitConverter.ToInt32(il, pos);
                    pos += 4;
                    try
                    {
                        operand = module.ResolveString(tok);
                    }
                    catch
                    {
                        operand = null;
                    }
                    break;
                }
                case OperandType.InlineSig:
                    pos += 4;
                    break;
                case OperandType.InlineMethod:
                case OperandType.InlineField:
                case OperandType.InlineType:
                case OperandType.InlineTok:
                {
                    int tok = BitConverter.ToInt32(il, pos);
                    pos += 4;
                    try
                    {
                        operand = module.ResolveMember(tok, typeArgs, methodArgs);
                    }
                    catch
                    {
                        operand = null;
                    }
                    break;
                }
                default:
                    throw new InvalidOperationException("Unknown operand type " + op.OperandType + " in " + m);
            }
            list.Add(new Ins { Index = list.Count, Offset = start, Op = op, Operand = operand });
        }
        return list;
    }

    public static bool IsStateMachineMoveNext(MethodBase m)
    {
        if (m.Name != "MoveNext" || m.DeclaringType == null)
        {
            return false;
        }
        return typeof(IAsyncStateMachine).IsAssignableFrom(m.DeclaringType)
            || typeof(System.Collections.IEnumerator).IsAssignableFrom(m.DeclaringType);
    }
}

/// <summary>
/// Reachability of the instructions of one method body when the TestMode / NonInteractive getters are
/// constants (the host always runs with TestMode on): a conditional branch fed directly by such a getter only
/// follows the edge that the constant selects. Exception handlers are reachable when any instruction of their
/// try block is. For async / iterator state machines the resume labels of the dispatch switch are reachable
/// only when the instruction just before them (the await's `leave` / the iterator's `ret`) is reachable, so code
/// after an await inside a TestMode-dead block stays dead.
/// </summary>
internal sealed class Cfg
{
    public readonly List<Ins> Code;
    public readonly bool[] Reachable;
    private readonly Dictionary<int, int> _byOffset = new Dictionary<int, int>();

    public Cfg(MethodBase method, List<Ins> code, Func<MethodBase, MethodBase, bool?> constantOf)
    {
        Code = code;
        Reachable = new bool[code.Count];
        foreach (Ins i in code)
        {
            _byOffset[i.Offset] = i.Index;
        }
        HashSet<int> labelled = new HashSet<int>();
        foreach (Ins i in code)
        {
            if (i.Operand is int t && (i.Op.OperandType == OperandType.InlineBrTarget || i.Op.OperandType == OperandType.ShortInlineBrTarget))
            {
                labelled.Add(t);
            }
            else if (i.Operand is int[] ts)
            {
                foreach (int x in ts)
                {
                    labelled.Add(x);
                }
            }
        }

        // constant-fed branches
        Dictionary<int, bool> constBranch = new Dictionary<int, bool>();
        for (int k = 0; k < code.Count; k++)
        {
            if ((code[k].Op == OpCodes.Call || code[k].Op == OpCodes.Callvirt) && code[k].Callee is MethodBase callee && constantOf(method, callee) is bool v)
            {
                int j = k + 1;
                // non-short-circuit `TestMode.IsOff & flag` / `flag & TestMode.IsOff` (Roslyn emits `and` for cheap operands)
                int p = j;
                while (p < code.Count && p < j + 3 && IsPureLoad(code[p].Op) && !labelled.Contains(code[p].Offset))
                {
                    p++;
                }
                if (p < code.Count && (code[p].Op == OpCodes.And || code[p].Op == OpCodes.Or) && !labelled.Contains(code[p].Offset))
                {
                    bool absorbing = code[p].Op == OpCodes.And ? !v : v; // false & x == false, true | x == true
                    if (!absorbing)
                    {
                        continue;
                    }
                    j = p + 1;
                }
                if (j + 1 < code.Count && code[j].Op == OpCodes.Ldc_I4_0 && code[j + 1].Op == OpCodes.Ceq && !labelled.Contains(code[j].Offset) && !labelled.Contains(code[j + 1].Offset))
                {
                    v = !v;
                    j += 2;
                }
                if (j < code.Count && !labelled.Contains(code[j].Offset) && IsBoolBranch(code[j].Op))
                {
                    constBranch[j] = v;
                }
            }
        }

        // state machine dispatch: branches fed by ldloc.0 (the state) in MoveNext
        HashSet<int> dispatch = new HashSet<int>();
        if (IlReader.IsStateMachineMoveNext(method))
        {
            for (int k = 0; k < code.Count; k++)
            {
                OpCode op = code[k].Op;
                if (op.FlowControl != FlowControl.Cond_Branch)
                {
                    continue;
                }
                int p = k - 1;
                int guard = 0;
                while (p >= 0 && guard < 4 && (IsLdcI4(code[p].Op) || code[p].Op == OpCodes.Sub))
                {
                    p--;
                    guard++;
                }
                if (p >= 0 && code[p].Op == OpCodes.Ldloc_0)
                {
                    dispatch.Add(k);
                }
            }
        }
        Dictionary<int, List<int>> resumeByPredecessor = new Dictionary<int, List<int>>();

        // exception regions
        List<(int tryStart, int tryEnd, int handler)> regions = new List<(int, int, int)>();
        try
        {
            MethodBody? body = method.GetMethodBody();
            if (body != null)
            {
                foreach (ExceptionHandlingClause c in body.ExceptionHandlingClauses)
                {
                    int ts = IndexAtOrAfter(c.TryOffset);
                    int te = IndexAtOrAfter(c.TryOffset + c.TryLength);
                    regions.Add((ts, te, IndexAtOrAfter(c.HandlerOffset)));
                    if ((c.Flags & ExceptionHandlingClauseOptions.Filter) != 0)
                    {
                        regions.Add((ts, te, IndexAtOrAfter(c.FilterOffset)));
                    }
                }
            }
        }
        catch
        {
            // no clauses
        }

        Stack<int> work = new Stack<int>();
        void Mark(int idx)
        {
            if (idx >= 0 && idx < code.Count && !Reachable[idx])
            {
                Reachable[idx] = true;
                work.Push(idx);
            }
        }
        Mark(0);
        while (work.Count > 0)
        {
            int i = work.Pop();
            Ins ins = code[i];
            foreach ((int ts, int te, int h) in regions)
            {
                if (i >= ts && i < te)
                {
                    Mark(h);
                }
            }
            if (resumeByPredecessor.TryGetValue(i, out List<int>? resumes))
            {
                foreach (int r in resumes)
                {
                    Mark(r);
                }
            }
            FlowControl fc = ins.Op.FlowControl;
            if (fc == FlowControl.Return || fc == FlowControl.Throw)
            {
                continue;
            }
            if (fc == FlowControl.Branch)
            {
                Mark(Target(ins));
                continue;
            }
            if (fc == FlowControl.Cond_Branch)
            {
                if (constBranch.TryGetValue(i, out bool v))
                {
                    bool taken = (ins.Op == OpCodes.Brtrue || ins.Op == OpCodes.Brtrue_S) ? v : !v;
                    Mark(taken ? Target(ins) : i + 1);
                    continue;
                }
                List<int> targets = new List<int>();
                if (ins.Operand is int[] sw)
                {
                    foreach (int t in sw)
                    {
                        targets.Add(_byOffset.TryGetValue(t, out int x) ? x : -1);
                    }
                }
                else
                {
                    targets.Add(Target(ins));
                }
                foreach (int t in targets)
                {
                    if (dispatch.Contains(i) && t > 0)
                    {
                        Ins before = code[t - 1];
                        if (before.Op == OpCodes.Leave || before.Op == OpCodes.Leave_S || before.Op == OpCodes.Ret)
                        {
                            if (Reachable[t - 1])
                            {
                                Mark(t);
                            }
                            else
                            {
                                if (!resumeByPredecessor.TryGetValue(t - 1, out List<int>? l))
                                {
                                    l = new List<int>();
                                    resumeByPredecessor[t - 1] = l;
                                }
                                l.Add(t);
                            }
                            continue;
                        }
                    }
                    Mark(t);
                }
                Mark(i + 1);
                continue;
            }
            Mark(i + 1);
        }
    }

    private int Target(Ins ins) => ins.Operand is int t && _byOffset.TryGetValue(t, out int x) ? x : -1;

    private int IndexAtOrAfter(int offset)
    {
        if (_byOffset.TryGetValue(offset, out int x))
        {
            return x;
        }
        for (int i = 0; i < Code.Count; i++)
        {
            if (Code[i].Offset >= offset)
            {
                return i;
            }
        }
        return Code.Count;
    }

    private static bool IsPureLoad(OpCode op) =>
        op.Name.StartsWith("ldarg", StringComparison.Ordinal) || op.Name.StartsWith("ldloc", StringComparison.Ordinal)
        || op == OpCodes.Ldfld || op == OpCodes.Ldsfld || IsLdcI4(op);

    private static bool IsBoolBranch(OpCode op) =>
        op == OpCodes.Brtrue || op == OpCodes.Brtrue_S || op == OpCodes.Brfalse || op == OpCodes.Brfalse_S;

    public static bool IsLdcI4(OpCode op) =>
        op == OpCodes.Ldc_I4 || op == OpCodes.Ldc_I4_S || op == OpCodes.Ldc_I4_0 || op == OpCodes.Ldc_I4_1 || op == OpCodes.Ldc_I4_2
        || op == OpCodes.Ldc_I4_3 || op == OpCodes.Ldc_I4_4 || op == OpCodes.Ldc_I4_5 || op == OpCodes.Ldc_I4_6 || op == OpCodes.Ldc_I4_7
        || op == OpCodes.Ldc_I4_8 || op == OpCodes.Ldc_I4_M1;
}
