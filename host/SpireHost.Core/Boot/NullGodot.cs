using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using SpireHost.Patches;

namespace SpireHost.Boot;

/// <summary>
/// "Null Godot": GodotSharp talks to the native engine through one table of 225 unmanaged function
/// pointers (Godot.NativeInterop.NativeFuncs.UnmanagedCallbacks). Without the engine the table is all
/// null and the first native call (even a static <c>StringName</c> field initializer) kills the process
/// with an uncatchable 0xC0000005.
///
/// Instead of patching every game type that owns such a static (the old transport's approach), the host
/// fills the table with tiny generated thunks, one per entry, that
///   1. increment a per-entry call counter (so the host can report exactly which natives were reached),
///   2. return zero (integer and float registers; hidden return buffers are zero-filled).
/// The table is handed to GodotSharp through its own public entry point <c>NativeFuncs.Initialize</c>.
///
/// Consequences: object-returning natives yield null handles, so GodotSharp's own guards turn node /
/// singleton usage into managed exceptions (catchable, with a stack trace) instead of process death, and
/// value-type statics (StringName, NodePath) become inert empty handles. No game code is modified.
/// </summary>
public static unsafe class NullGodot
{
    private static long* _counters;
    private static string[] _names = Array.Empty<string>();
    private static bool _installed;

    public static int EntryCount => _names.Length;

    public static void Install()
    {
        if (_installed)
        {
            return;
        }
        if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            throw new PlatformNotSupportedException("NullGodot thunks are generated for x64 only.");
        }
        bool sysV = OperatingSystem.IsLinux();
        if (!sysV && !OperatingSystem.IsWindows())
        {
            throw new PlatformNotSupportedException("NullGodot thunks are generated for Windows x64 and Linux x64 (System V) only.");
        }
        Type nativeFuncs = typeof(Godot.GodotObject).Assembly.GetType("Godot.NativeInterop.NativeFuncs", throwOnError: true)!;
        Type table = nativeFuncs.GetNestedType("UnmanagedCallbacks", BindingFlags.NonPublic | BindingFlags.Public)
            ?? throw new MissingMemberException("NativeFuncs.UnmanagedCallbacks");
        FieldInfo[] fields = table.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        // Sequential layout: metadata order == memory order. Verify with the declared size.
        int tableSize = Marshal.SizeOf(table);
        if (tableSize != fields.Length * IntPtr.Size)
        {
            throw new InvalidOperationException($"Unexpected UnmanagedCallbacks layout: {fields.Length} fields, {tableSize} bytes.");
        }
        Array.Sort(fields, (a, b) => Marshal.OffsetOf(table, a.Name).ToInt64().CompareTo(Marshal.OffsetOf(table, b.Name).ToInt64()));

        int n = fields.Length;
        _names = new string[n];
        _counters = (long*)NativeMemory.AllocZeroed((nuint)n, sizeof(long));
        if (sysV)
        {
            InstallSysV(nativeFuncs, fields, tableSize);
            return;
        }
        const int thunkSlot = 96;
        byte* code = (byte*)VirtualAlloc(IntPtr.Zero, (nuint)(n * thunkSlot), 0x3000 /* MEM_COMMIT|MEM_RESERVE */, 0x04 /* PAGE_READWRITE */);
        if (code == null)
        {
            throw new OutOfMemoryException("VirtualAlloc for NullGodot thunks failed.");
        }
        nint* ptrs = (nint*)NativeMemory.AllocZeroed((nuint)n, (nuint)IntPtr.Size);
        for (int i = 0; i < n; i++)
        {
            _names[i] = fields[i].Name;
            int hidden = HiddenReturnSize(fields[i].FieldType);
            byte* p = code + i * thunkSlot;
            ptrs[i] = (nint)p;
            // mov rax, &counter[i] ; lock inc qword ptr [rax]
            *p++ = 0x48; *p++ = 0xB8; *(long**)p = _counters + i; p += 8;
            *p++ = 0xF0; *p++ = 0x48; *p++ = 0xFF; *p++ = 0x00;
            // xor eax, eax ; xorps xmm0, xmm0
            *p++ = 0x31; *p++ = 0xC0;
            *p++ = 0x0F; *p++ = 0x57; *p++ = 0xC0;
            if (hidden > 0)
            {
                // zero-fill the caller's return buffer at [rcx], then return its address in rax
                int off = 0;
                for (; off + 8 <= hidden; off += 8)
                {
                    *p++ = 0x48; *p++ = 0x89; *p++ = 0x41; *p++ = (byte)off;   // mov [rcx+off], rax
                }
                for (; off < hidden; off++)
                {
                    *p++ = 0xC6; *p++ = 0x41; *p++ = (byte)off; *p++ = 0x00;   // mov byte ptr [rcx+off], 0
                }
                *p++ = 0x48; *p++ = 0x89; *p++ = 0xC8;                          // mov rax, rcx
            }
            *p++ = 0xC3;                                                        // ret
            if (p - (code + i * thunkSlot) > thunkSlot)
            {
                throw new InvalidOperationException("NullGodot thunk overflow for " + fields[i].Name);
            }
        }
        if (!VirtualProtect((IntPtr)code, (nuint)(n * thunkSlot), 0x20 /* PAGE_EXECUTE_READ */, out _))
        {
            throw new InvalidOperationException("VirtualProtect for NullGodot thunks failed.");
        }
        MethodInfo init = nativeFuncs.GetMethod("Initialize", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new MissingMethodException("NativeFuncs.Initialize");
        init.Invoke(null, new object[] { (nint)ptrs, tableSize });
        _installed = true;
        PatchRegistry.Note("null-godot-table", "Godot.NativeInterop.NativeFuncs::Initialize (GodotSharp, not the game)", "native thunk table",
            PatchCategory.Plumbing,
            $"All {n} GodotSharp->engine function pointers point at counting no-op thunks that return zero, so a stray native call is a counted no-op / managed exception instead of an uncatchable access violation. Game code untouched.");
    }

    /// <summary>
    /// Windows x64: a by-value struct return whose size is not 1, 2, 4 or 8 bytes is written through a hidden
    /// pointer passed in rcx. Returns that size, or 0 when the value comes back in rax / xmm0.
    /// </summary>
    private static int HiddenReturnSize(Type functionPointerType)
    {
        if (!functionPointerType.IsFunctionPointer)
        {
            throw new InvalidOperationException("UnmanagedCallbacks field is not a function pointer: " + functionPointerType);
        }
        Type ret = functionPointerType.GetFunctionPointerReturnType();
        if (ret == typeof(void) || ret.IsPointer || ret.IsFunctionPointer || ret.IsPrimitive || ret.IsEnum || ret == typeof(IntPtr) || ret == typeof(UIntPtr))
        {
            return 0;
        }
        int size = Marshal.SizeOf(ret);
        return size is 1 or 2 or 4 or 8 ? 0 : size;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // System V x64 (Linux). A by-value struct comes back either in registers (rax / rdx for INTEGER eightbytes, xmm0 /
    // xmm1 for SSE eightbytes) or, when it is of class MEMORY, through a hidden buffer whose address the caller passes
    // in rdi and the callee returns in rax. The thunk zeroes rax, rdx, xmm0 and xmm1; for a MEMORY-class return it also
    // zero-fills [rdi, rdi+size) and returns rdi. Which return types are MEMORY class is not a hand-written rule: it is
    // measured from the runtime at startup (ProbeMemoryClass).
    // ---------------------------------------------------------------------------------------------------------------

    private const int PROT_READ = 0x1, PROT_WRITE = 0x2, PROT_EXEC = 0x4, MAP_PRIVATE = 0x02, MAP_ANONYMOUS = 0x20;

    private static void InstallSysV(Type nativeFuncs, FieldInfo[] fields, int tableSize)
    {
        int n = fields.Length;
        Dictionary<Type, int> hiddenSize = new Dictionary<Type, int>();
        foreach (FieldInfo f in fields)
        {
            Type? ret = StructReturnType(f.FieldType);
            if (ret != null && !hiddenSize.ContainsKey(ret))
            {
                hiddenSize[ret] = ProbeMemoryClass(ret) ? Marshal.SizeOf(ret) : 0;
            }
        }
        _sysVReturnClasses = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (KeyValuePair<Type, int> kv in hiddenSize)
        {
            _sysVReturnClasses[kv.Key.FullName!] = kv.Value;
        }
        Console.Error.WriteLine("null-godot: System V struct returns (hidden-buffer bytes, 0 = registers): " +
            string.Join(", ", System.Linq.Enumerable.Select(_sysVReturnClasses, kv => kv.Key + "=" + kv.Value)));

        const int thunkSlot = 128;
        nuint codeSize = (nuint)(n * thunkSlot);
        byte* code = MapReadWrite(codeSize);
        nint* ptrs = (nint*)NativeMemory.AllocZeroed((nuint)n, (nuint)IntPtr.Size);
        for (int i = 0; i < n; i++)
        {
            _names[i] = fields[i].Name;
            Type? ret = StructReturnType(fields[i].FieldType);
            int hidden = ret == null ? 0 : hiddenSize[ret];
            if (hidden >= 128)
            {
                throw new InvalidOperationException($"NullGodot SysV thunk: MEMORY-class return of {hidden} bytes for {fields[i].Name} exceeds the disp8 encoding.");
            }
            byte* p = code + i * thunkSlot;
            ptrs[i] = (nint)p;
            // mov rax, &counter[i] ; lock inc qword ptr [rax]
            *p++ = 0x48; *p++ = 0xB8; *(long**)p = _counters + i; p += 8;
            *p++ = 0xF0; *p++ = 0x48; *p++ = 0xFF; *p++ = 0x00;
            // xor eax, eax ; xor edx, edx ; xorps xmm0, xmm0 ; xorps xmm1, xmm1
            *p++ = 0x31; *p++ = 0xC0;
            *p++ = 0x31; *p++ = 0xD2;
            *p++ = 0x0F; *p++ = 0x57; *p++ = 0xC0;
            *p++ = 0x0F; *p++ = 0x57; *p++ = 0xC9;
            if (hidden > 0)
            {
                // zero-fill the caller's return buffer at [rdi], then return its address in rax
                int off = 0;
                for (; off + 8 <= hidden; off += 8)
                {
                    *p++ = 0x48; *p++ = 0x89; *p++ = 0x47; *p++ = (byte)off;   // mov [rdi+off], rax
                }
                for (; off < hidden; off++)
                {
                    *p++ = 0xC6; *p++ = 0x47; *p++ = (byte)off; *p++ = 0x00;   // mov byte ptr [rdi+off], 0
                }
                *p++ = 0x48; *p++ = 0x89; *p++ = 0xF8;                          // mov rax, rdi
            }
            *p++ = 0xC3;                                                        // ret
            if (p - (code + i * thunkSlot) > thunkSlot)
            {
                throw new InvalidOperationException("NullGodot thunk overflow for " + fields[i].Name);
            }
        }
        ProtectReadExec(code, codeSize);
        MethodInfo init = nativeFuncs.GetMethod("Initialize", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new MissingMethodException("NativeFuncs.Initialize");
        init.Invoke(null, new object[] { (nint)ptrs, tableSize });
        _installed = true;
        PatchRegistry.Note("null-godot-table", "Godot.NativeInterop.NativeFuncs::Initialize (GodotSharp, not the game)", "native thunk table",
            PatchCategory.Plumbing,
            $"All {n} GodotSharp->engine function pointers point at counting no-op thunks that return zero, so a stray native call is a counted no-op / managed exception instead of an uncatchable access violation. Game code untouched.");
    }

    private static SortedDictionary<string, int>? _sysVReturnClasses;

    /// <summary>Linux only: every distinct by-value struct return type of the table -> its hidden-buffer size as measured
    /// by the probe (0 = returned in registers). Null on Windows.</summary>
    public static SortedDictionary<string, int>? SysVReturnClasses => _sysVReturnClasses;

    /// <summary>The return type of a table entry if it is a by-value struct (the only kind whose return path depends on
    /// the ABI's classification), else null.</summary>
    private static Type? StructReturnType(Type functionPointerType)
    {
        if (!functionPointerType.IsFunctionPointer)
        {
            throw new InvalidOperationException("UnmanagedCallbacks field is not a function pointer: " + functionPointerType);
        }
        Type ret = functionPointerType.GetFunctionPointerReturnType();
        if (ret == typeof(void) || ret.IsPointer || ret.IsFunctionPointer || ret.IsPrimitive || ret.IsEnum || ret == typeof(IntPtr) || ret == typeof(UIntPtr))
        {
            return null;
        }
        if (!ret.IsValueType)
        {
            throw new InvalidOperationException("UnmanagedCallbacks entry returns a non-value type: " + ret);
        }
        return ret;
    }

    private delegate void ProbeCall(nint fn, nint arg, nint dst);

    // Two probe sets: { rdx, xmm0, xmm1 } sentinels and the seed of the marker the probe writes through rdi.
    private static readonly ulong[][] ProbeSentinels =
    {
        new ulong[] { 0x93D1E8476AF2B05CUL, 0x2C8B7F15D9E3460AUL, 0xE64A0D9C37B18F25UL, 0x3B },
        new ulong[] { 0x0E9C35D8F1A7426BUL, 0xB3276E0F4C9AD851UL, 0x48D5B1C27E06F39AUL, 0xC6 },
    };

    private static byte Marker(ulong seed, int off) => (byte)((int)seed + 29 * off + 1);

    /// <summary>
    /// Measures how the runtime returns <paramref name="t"/> from an unmanaged call. A probe thunk
    ///     mov rax, rdi ; [rdi + k] := marker(k) for k in [0, size) ; rdx, xmm0, xmm1 := sentinels ; ret
    /// is called through a DynamicMethod calli (unmanaged, ONE native-int argument: the address of a scratch buffer).
    ///  * returned in registers: the argument arrived in rdi, so the probe wrote the marker into the scratch buffer and
    ///    returned the scratch address in rax; the result's eightbytes are, in order, rax / rdx (INTEGER) and xmm0 /
    ///    xmm1 (SSE);
    ///  * MEMORY class: rdi was the runtime's hidden buffer, the probe filled it with the marker and returned it in rax;
    ///    the result is exactly the marker and the scratch buffer is untouched.
    /// (The lead's first design - a probe that writes no memory, MEMORY recognised by the absence of the sentinels -
    /// does not hold: the runtime's hidden buffer is a stack temporary whose garbage contained the argument sentinel,
    /// measured for Godot.Basis. Here both cases are recognised positively.) Every type must fall cleanly into exactly
    /// one case, identically for both probe sets.
    /// </summary>
    private static bool ProbeMemoryClass(Type t)
    {
        int size = Marshal.SizeOf(t);
        if (size >= 128)
        {
            throw new InvalidOperationException($"NullGodot probe: {t.FullName} has {size} bytes, beyond the disp8 encoding.");
        }
        DynamicMethod dm = new DynamicMethod("NullGodotProbe_" + t.Name, typeof(void), new[] { typeof(nint), typeof(nint), typeof(nint) },
            typeof(NullGodot).Module, skipVisibility: true);
        ILGenerator il = dm.GetILGenerator();
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(OpCodes.Ldarg_1);
        il.Emit(OpCodes.Ldarg_0);
        il.EmitCalli(OpCodes.Calli, CallingConvention.Cdecl, t, new[] { typeof(nint) });
        il.Emit(OpCodes.Stobj, t);
        il.Emit(OpCodes.Ret);
        ProbeCall call = (ProbeCall)dm.CreateDelegate(typeof(ProbeCall));

        bool? memory = null;
        foreach (ulong[] s in ProbeSentinels)
        {
            nuint len = 4096;
            byte* thunk = MapReadWrite(len);
            byte* p = thunk;
            *p++ = 0x48; *p++ = 0x89; *p++ = 0xF8;                                  // mov rax, rdi
            for (int k = 0; k < size; k++)
            {
                *p++ = 0xC6; *p++ = 0x47; *p++ = (byte)k; *p++ = Marker(s[3], k);    // mov byte ptr [rdi+k], marker
            }
            *p++ = 0x48; *p++ = 0xBA; *(ulong*)p = s[0]; p += 8;                     // mov rdx, imm64
            *p++ = 0x49; *p++ = 0xBB; *(ulong*)p = s[1]; p += 8;                     // mov r11, imm64
            *p++ = 0x66; *p++ = 0x49; *p++ = 0x0F; *p++ = 0x6E; *p++ = 0xC3;         // movq xmm0, r11
            *p++ = 0x49; *p++ = 0xBB; *(ulong*)p = s[2]; p += 8;                     // mov r11, imm64
            *p++ = 0x66; *p++ = 0x49; *p++ = 0x0F; *p++ = 0x6E; *p++ = 0xCB;         // movq xmm1, r11
            *p++ = 0xC3;                                                            // ret
            ProtectReadExec(thunk, len);

            byte[] scratch = new byte[128];
            byte[] result = new byte[size];
            Array.Fill(scratch, (byte)0xA5);
            Array.Fill(result, (byte)0x5A);
            fixed (byte* scr = scratch)
            fixed (byte* dst = result)
            {
                call((nint)thunk, (nint)scr, (nint)dst);
                bool mem = ClassifyProbeResult(t, result, scratch, (ulong)scr, s);
                if (memory != null && memory.Value != mem)
                {
                    throw new InvalidOperationException($"NullGodot probe: {t.FullName} classified differently by the two probe sets.");
                }
                memory = mem;
            }
            if (munmap((IntPtr)thunk, len) != 0)
            {
                throw new InvalidOperationException("munmap of the NullGodot probe failed, errno " + Marshal.GetLastPInvokeError());
            }
        }
        return memory!.Value;
    }

    /// <summary>true = MEMORY class, false = registers; anything that is not cleanly one of the two throws.</summary>
    private static bool ClassifyProbeResult(Type t, byte[] result, byte[] scratch, ulong scratchAddress, ulong[] s)
    {
        int size = result.Length;
        bool resultIsMarker = true, scratchIsMarker = true, scratchUntouched = true;
        for (int k = 0; k < size; k++)
        {
            resultIsMarker &= result[k] == Marker(s[3], k);
            scratchIsMarker &= scratch[k] == Marker(s[3], k);
        }
        for (int k = 0; k < scratch.Length; k++)
        {
            scratchUntouched &= scratch[k] == 0xA5;
        }
        int nextInt = 0, nextSse = 0;
        bool decodes = true;
        ulong[] intRegs = { scratchAddress, s[0] }, sseRegs = { s[1], s[2] };
        for (int off = 0; off < size; off += 8)
        {
            int len = Math.Min(8, size - off);
            bool isInt = nextInt < 2 && PrefixEquals(result, off, len, intRegs[nextInt]);
            bool isSse = nextSse < 2 && PrefixEquals(result, off, len, sseRegs[nextSse]);
            if (isInt == isSse)
            {
                decodes = false;
                break;
            }
            if (isInt)
            {
                nextInt++;
            }
            else
            {
                nextSse++;
            }
        }
        bool registers = decodes && scratchIsMarker;
        bool memoryClass = resultIsMarker && scratchUntouched;
        if (registers == memoryClass)
        {
            throw new InvalidOperationException($"NullGodot probe: {t.FullName} ({size} bytes) is not cleanly register-returned or MEMORY class: result {Convert.ToHexString(result)}, scratch {Convert.ToHexString(scratch, 0, size)}");
        }
        return memoryClass;
    }
    private static bool PrefixEquals(byte[] buf, int off, int len, ulong value)
    {
        for (int i = 0; i < len; i++)
        {
            if (buf[off + i] != (byte)(value >> (8 * i)))
            {
                return false;
            }
        }
        return true;
    }

    private static byte* MapReadWrite(nuint len)
    {
        IntPtr m = mmap(IntPtr.Zero, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == new IntPtr(-1))
        {
            throw new OutOfMemoryException("mmap for NullGodot thunks failed, errno " + Marshal.GetLastPInvokeError());
        }
        return (byte*)m;
    }

    private static void ProtectReadExec(byte* code, nuint len)
    {
        if (mprotect((IntPtr)code, len, PROT_READ | PROT_EXEC) != 0)
        {
            throw new InvalidOperationException("mprotect for NullGodot thunks failed, errno " + Marshal.GetLastPInvokeError());
        }
    }

    [DllImport("libc.so.6", SetLastError = true)]
    private static extern IntPtr mmap(IntPtr addr, nuint length, int prot, int flags, int fd, nint offset);

    [DllImport("libc.so.6", SetLastError = true)]
    private static extern int mprotect(IntPtr addr, nuint len, int prot);

    [DllImport("libc.so.6", SetLastError = true)]
    private static extern int munmap(IntPtr addr, nuint length);

    public static long[] SnapshotCounters()
    {
        long[] r = new long[_names.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = _counters[i];
        }
        return r;
    }

    /// <summary>Native calls made since <paramref name="since"/>, by entry name (only non-zero deltas).</summary>
    public static SortedDictionary<string, long> Delta(long[]? since)
    {
        SortedDictionary<string, long> d = new SortedDictionary<string, long>(StringComparer.Ordinal);
        for (int i = 0; i < _names.Length; i++)
        {
            long v = _counters[i] - (since == null ? 0 : since[i]);
            if (v != 0)
            {
                d[_names[i]] = v;
            }
        }
        return d;
    }

    [DllImport("kernel32", SetLastError = true)]
    private static extern IntPtr VirtualAlloc(IntPtr lpAddress, nuint dwSize, uint flAllocationType, uint flProtect);

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool VirtualProtect(IntPtr lpAddress, nuint dwSize, uint flNewProtect, out uint lpflOldProtect);
}
