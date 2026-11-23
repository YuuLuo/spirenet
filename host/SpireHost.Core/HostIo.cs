using System;
using System.Collections.Generic;

namespace SpireHost;

public static class HostIo
{
    /// <summary>The real stdout. Console.Out is redirected to stderr at startup.</summary>
    public static System.IO.TextWriter Out = Console.Out;
}

/// <summary>The JSON options every command uses for its result on stdout.</summary>
public static class HostJson
{
    public static readonly System.Text.Json.JsonSerializerOptions Options = new System.Text.Json.JsonSerializerOptions
    {
        WriteIndented = true,
        IncludeFields = true,
        Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() },
    };

    public static string Serialize(object value) => System.Text.Json.JsonSerializer.Serialize(value, Options);

    /// <summary>Writes the result to stdout and, when given, to a file (directories are created).</summary>
    public static void Emit(object value, string? outPath)
    {
        string json = Serialize(value);
        if (!string.IsNullOrEmpty(outPath))
        {
            string full = System.IO.Path.GetFullPath(outPath);
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(full)!);
            System.IO.File.WriteAllText(full, json);
        }
        HostIo.Out.WriteLine(json);
    }
}

/// <summary>Minimal "--key value" / "--flag" / "--key=value" parser.</summary>
public sealed class Args
{
    private readonly Dictionary<string, string?> _values = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);

    public string Command { get; private set; } = "";

    public static Args Parse(string[] args)
    {
        Args a = new Args();
        for (int i = 0; i < args.Length; i++)
        {
            string s = args[i];
            if (!s.StartsWith("--", StringComparison.Ordinal))
            {
                continue;
            }
            string key = s.Substring(2);
            string? value = null;
            int eq = key.IndexOf('=');
            if (eq > 0)
            {
                value = key.Substring(eq + 1);
                key = key.Substring(0, eq);
            }
            else if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                value = args[++i];
            }
            if (a.Command.Length == 0)
            {
                a.Command = key;
                if (value != null)
                {
                    a._values[key] = value;
                }
                continue;
            }
            a._values[key] = value;
        }
        return a;
    }

    /// <summary>
    /// True when the option was given. The parser hands the next non-dash token to an option as its value, so a switch
    /// written as "--x false" (or 0 / off / no) arrives here with that value: it means "not given". Verifier finding R-3:
    /// "--serve --allow-debug-options false" used to switch the debug options ON, because only the presence of the name
    /// was tested. An option with any other value, or with none, is given.
    /// </summary>
    public bool Has(string key) => _values.TryGetValue(key, out string? v) && !IsFalseLiteral(v);

    /// <summary>
    /// For a switch that takes no value: null when it is absent, bare, or written with a boolean literal; otherwise what is
    /// wrong (a switch followed by a stray token is refused where it matters instead of being read as "on").
    /// </summary>
    public string? SwitchValueError(string key)
    {
        if (!_values.TryGetValue(key, out string? v) || v == null || IsFalseLiteral(v) || IsTrueLiteral(v))
        {
            return null;
        }
        return "--" + key + " is a switch and takes no value (got '" + v + "'; write --" + key + " or leave it out)";
    }

    private static bool IsFalseLiteral(string? v) =>
        v != null && (v == "0" || v.Equals("false", StringComparison.OrdinalIgnoreCase) || v.Equals("off", StringComparison.OrdinalIgnoreCase) || v.Equals("no", StringComparison.OrdinalIgnoreCase));

    private static bool IsTrueLiteral(string v) =>
        v == "1" || v.Equals("true", StringComparison.OrdinalIgnoreCase) || v.Equals("on", StringComparison.OrdinalIgnoreCase) || v.Equals("yes", StringComparison.OrdinalIgnoreCase);

    /// <summary>Every option name that was given (the command word included when it carried a value).</summary>
    public IEnumerable<string> Keys => _values.Keys;

    public string? Get(string key) => _values.TryGetValue(key, out string? v) ? v : null;

    public string Get(string key, string fallback) => Get(key) ?? fallback;

    public int GetInt(string key, int fallback) => Get(key) is string v ? int.Parse(v) : fallback;

    public long GetLong(string key, long fallback) => Get(key) is string v ? long.Parse(v) : fallback;

    public ulong GetULong(string key, ulong fallback) => Get(key) is string v ? ulong.Parse(v) : fallback;

    public bool GetBool(string key, bool fallback)
    {
        if (!_values.TryGetValue(key, out string? v))
        {
            return fallback;
        }
        return v == null || v == "1" || v.Equals("true", StringComparison.OrdinalIgnoreCase) || v.Equals("on", StringComparison.OrdinalIgnoreCase);
    }
}
