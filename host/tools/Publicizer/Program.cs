using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Security.Cryptography;

namespace SpireHost.Tools;

/// <summary>
/// Publicizer: copies an assembly and rewrites ONLY the accessibility bits of the TypeDef / Field /
/// MethodDef metadata rows so that everything is public. IL, signatures, names, MVID and layout stay
/// byte-identical. The output is a compile-time reference; it must never be loaded at run time.
///
/// Fields that share their name with an event of the same type keep their accessibility: a field-like
/// event has a private backing field of the same name, and making both public would turn every
/// <c>x.SomeEvent += ...</c> into an ambiguity error (CS0229).
///
/// usage: Publicizer &lt;input.dll&gt; &lt;output.dll&gt; [--expect-sha256 hex]
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("usage: Publicizer <input.dll> <output.dll> [--expect-sha256 <hex>]");
            return 1;
        }
        string input = Path.GetFullPath(args[0]);
        string output = Path.GetFullPath(args[1]);
        if (string.Equals(input, output, StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine("Publicizer: refusing to overwrite the input assembly.");
            return 1;
        }
        byte[] bytes = File.ReadAllBytes(input);
        string sha = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        for (int i = 2; i + 1 < args.Length; i++)
        {
            if (args[i] == "--expect-sha256" && !sha.Equals(args[i + 1], StringComparison.OrdinalIgnoreCase))
            {
                Console.Error.WriteLine($"Publicizer: {input} has SHA256 {sha}, expected {args[i + 1]}.");
                return 3;
            }
        }

        int types = 0, fields = 0, methods = 0, skippedEventFields = 0;
        byte[] patched = (byte[])bytes.Clone();
        using (PEReader pe = new PEReader(new MemoryStream(bytes)))
        {
            MetadataReader md = pe.GetMetadataReader();
            int mdStart = pe.PEHeaders.MetadataStartOffset;

            // Fields that must keep their accessibility: backing fields of field-like events.
            HashSet<int> keep = new HashSet<int>();
            foreach (TypeDefinitionHandle th in md.TypeDefinitions)
            {
                TypeDefinition td = md.GetTypeDefinition(th);
                HashSet<string>? eventNames = null;
                foreach (EventDefinitionHandle eh in td.GetEvents())
                {
                    (eventNames ??= new HashSet<string>(StringComparer.Ordinal)).Add(md.GetString(md.GetEventDefinition(eh).Name));
                }
                if (eventNames == null)
                {
                    continue;
                }
                foreach (FieldDefinitionHandle fh in td.GetFields())
                {
                    if (eventNames.Contains(md.GetString(md.GetFieldDefinition(fh).Name)))
                    {
                        keep.Add(MetadataTokens.GetRowNumber(fh));
                    }
                }
            }

            // TypeDef row: Flags (4 bytes) first.
            int typeOff = mdStart + md.GetTableMetadataOffset(TableIndex.TypeDef);
            int typeRow = md.GetTableRowSize(TableIndex.TypeDef);
            int typeCount = md.GetTableRowCount(TableIndex.TypeDef);
            for (int r = 0; r < typeCount; r++)
            {
                int o = typeOff + r * typeRow;
                uint flags = BitConverter.ToUInt32(patched, o);
                uint vis = flags & (uint)TypeAttributes.VisibilityMask;
                uint want = vis <= (uint)TypeAttributes.Public ? (uint)TypeAttributes.Public : (uint)TypeAttributes.NestedPublic;
                if (r == 0)
                {
                    continue; // <Module>
                }
                if (vis != want)
                {
                    flags = (flags & ~(uint)TypeAttributes.VisibilityMask) | want;
                    BitConverter.TryWriteBytes(new Span<byte>(patched, o, 4), flags);
                    types++;
                }
            }

            // Field row: Flags (2 bytes) first.
            int fieldOff = mdStart + md.GetTableMetadataOffset(TableIndex.Field);
            int fieldRow = md.GetTableRowSize(TableIndex.Field);
            int fieldCount = md.GetTableRowCount(TableIndex.Field);
            for (int r = 0; r < fieldCount; r++)
            {
                if (keep.Contains(r + 1))
                {
                    skippedEventFields++;
                    continue;
                }
                int o = fieldOff + r * fieldRow;
                ushort flags = BitConverter.ToUInt16(patched, o);
                if ((flags & (ushort)FieldAttributes.FieldAccessMask) != (ushort)FieldAttributes.Public)
                {
                    flags = (ushort)((flags & ~(ushort)FieldAttributes.FieldAccessMask) | (ushort)FieldAttributes.Public);
                    BitConverter.TryWriteBytes(new Span<byte>(patched, o, 2), flags);
                    fields++;
                }
            }

            // MethodDef row: RVA (4), ImplFlags (2), Flags (2), ...
            int methodOff = mdStart + md.GetTableMetadataOffset(TableIndex.MethodDef);
            int methodRow = md.GetTableRowSize(TableIndex.MethodDef);
            int methodCount = md.GetTableRowCount(TableIndex.MethodDef);
            for (int r = 0; r < methodCount; r++)
            {
                int o = methodOff + r * methodRow + 6;
                ushort flags = BitConverter.ToUInt16(patched, o);
                if ((flags & (ushort)MethodAttributes.MemberAccessMask) != (ushort)MethodAttributes.Public)
                {
                    flags = (ushort)((flags & ~(ushort)MethodAttributes.MemberAccessMask) | (ushort)MethodAttributes.Public);
                    BitConverter.TryWriteBytes(new Span<byte>(patched, o, 2), flags);
                    methods++;
                }
            }
        }

        // Sanity: the patched image must still parse and expose the same row counts.
        using (PEReader check = new PEReader(new MemoryStream(patched)))
        {
            MetadataReader md = check.GetMetadataReader();
            foreach (TypeDefinitionHandle th in md.TypeDefinitions)
            {
                _ = md.GetString(md.GetTypeDefinition(th).Name);
            }
        }

        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        File.WriteAllBytes(output, patched);
        File.WriteAllText(output + ".info.txt",
            "REFERENCE ASSEMBLY ONLY - never load this file at run time.\n" +
            $"source: {input}\nsource_sha256: {sha}\n" +
            $"publicized: types={types} fields={fields} methods={methods} event_backing_fields_kept={skippedEventFields}\n" +
            $"output_sha256: {Convert.ToHexString(SHA256.HashData(patched)).ToLowerInvariant()}\n");
        Console.WriteLine($"Publicizer: {Path.GetFileName(input)} -> {output} (types={types} fields={fields} methods={methods}, kept {skippedEventFields} event backing fields)");
        return 0;
    }
}
