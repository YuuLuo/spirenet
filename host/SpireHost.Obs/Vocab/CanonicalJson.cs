using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace SpireHost.Obs.Vocab;

/// <summary>
/// A tiny JSON object model with a writer that reproduces Python's
/// <c>json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")</c> byte for byte
/// (schema-errata-v1 E-11: this is the definition of "canonical JSON"; the hash is SHA-256 of these bytes, first 4
/// bytes read as little-endian int32 = py/spire/vocab.canonical_hash).
///
/// Values: <see cref="JObj"/> (keys sorted by code point, i.e. ordinal), <see cref="List{T}"/> of object?,
/// string, long/int, double (finite, written in Python repr form), bool, null.
/// </summary>
public sealed class JObj : SortedDictionary<string, object?>
{
    public JObj() : base(StringComparer.Ordinal)
    {
    }
}

public static class CanonicalJson
{
    public static byte[] Serialize(object? value)
    {
        StringBuilder sb = new StringBuilder(1 << 16);
        Write(sb, value, indent: -1, depth: 0);
        return new UTF8Encoding(false).GetBytes(sb.ToString());
    }

    /// <summary>Human-readable form (sorted keys, 1-space indent); parses to the same object, so it hashes the same.</summary>
    public static string Pretty(object? value)
    {
        StringBuilder sb = new StringBuilder(1 << 16);
        Write(sb, value, indent: 1, depth: 0);
        sb.Append('\n');
        return sb.ToString();
    }

    public static int Hash(object? value)
    {
        byte[] digest = SHA256.HashData(Serialize(value));
        return BitConverter.ToInt32(digest, 0); // little-endian on every supported platform
    }

    private static void Newline(StringBuilder sb, int indent, int depth)
    {
        if (indent < 0)
        {
            return;
        }
        sb.Append('\n');
        sb.Append(' ', indent * depth);
    }

    private static void Write(StringBuilder sb, object? v, int indent, int depth)
    {
        switch (v)
        {
            case null:
                sb.Append("null");
                return;
            case bool b:
                sb.Append(b ? "true" : "false");
                return;
            case string s:
                WriteString(sb, s);
                return;
            case int i:
                sb.Append(i.ToString(CultureInfo.InvariantCulture));
                return;
            case long l:
                sb.Append(l.ToString(CultureInfo.InvariantCulture));
                return;
            case double d:
                sb.Append(FormatDouble(d));
                return;
            case JObj o:
            {
                if (o.Count == 0)
                {
                    sb.Append("{}");
                    return;
                }
                sb.Append('{');
                bool first = true;
                foreach (KeyValuePair<string, object?> kv in o)
                {
                    if (!first)
                    {
                        sb.Append(',');
                    }
                    first = false;
                    Newline(sb, indent, depth + 1);
                    WriteString(sb, kv.Key);
                    sb.Append(indent < 0 ? ":" : ": ");
                    Write(sb, kv.Value, indent, depth + 1);
                }
                Newline(sb, indent, depth);
                sb.Append('}');
                return;
            }
            case List<object?> list:
            {
                if (list.Count == 0)
                {
                    sb.Append("[]");
                    return;
                }
                sb.Append('[');
                for (int k = 0; k < list.Count; k++)
                {
                    if (k > 0)
                    {
                        sb.Append(',');
                    }
                    Newline(sb, indent, depth + 1);
                    Write(sb, list[k], indent, depth + 1);
                }
                Newline(sb, indent, depth);
                sb.Append(']');
                return;
            }
            default:
                throw new ArgumentException("CanonicalJson: unsupported value type " + v.GetType());
        }
    }

    /// <summary>Python repr of a finite float (shortest round trip, with ".0" for integral values; no exponent form here).</summary>
    public static string FormatDouble(double d)
    {
        if (double.IsNaN(d) || double.IsInfinity(d))
        {
            throw new ArgumentException("non-finite double in canonical JSON");
        }
        string s = d.ToString("R", CultureInfo.InvariantCulture);
        if (s.Contains('E') || s.Contains('e'))
        {
            throw new ArgumentException("double needs exponent form in canonical JSON: " + s);
        }
        if (!s.Contains('.'))
        {
            s += ".0";
        }
        return s;
    }

    private static void WriteString(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"':
                    sb.Append("\\\"");
                    break;
                case '\\':
                    sb.Append("\\\\");
                    break;
                case '\n':
                    sb.Append("\\n");
                    break;
                case '\r':
                    sb.Append("\\r");
                    break;
                case '\t':
                    sb.Append("\\t");
                    break;
                case '\b':
                    sb.Append("\\b");
                    break;
                case '\f':
                    sb.Append("\\f");
                    break;
                default:
                    if (c < 0x20)
                    {
                        sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        sb.Append(c);
                    }
                    break;
            }
        }
        sb.Append('"');
    }

    // ---------------------------------------------------------------------------------------------- parsing

    public static object? Parse(string text)
    {
        using JsonDocument doc = JsonDocument.Parse(text, new JsonDocumentOptions { MaxDepth = 256 });
        return Convert(doc.RootElement);
    }

    public static object? ParseFile(string path) => Parse(File.ReadAllText(path, System.Text.Encoding.UTF8));

    private static object? Convert(JsonElement e)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.Object:
            {
                JObj o = new JObj();
                foreach (JsonProperty p in e.EnumerateObject())
                {
                    o[p.Name] = Convert(p.Value);
                }
                return o;
            }
            case JsonValueKind.Array:
            {
                List<object?> l = new List<object?>();
                foreach (JsonElement x in e.EnumerateArray())
                {
                    l.Add(Convert(x));
                }
                return l;
            }
            case JsonValueKind.String:
                return e.GetString();
            case JsonValueKind.Number:
                if (e.TryGetInt64(out long l64))
                {
                    return l64;
                }
                return e.GetDouble();
            case JsonValueKind.True:
                return true;
            case JsonValueKind.False:
                return false;
            default:
                return null;
        }
    }
}
