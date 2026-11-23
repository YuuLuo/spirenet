using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Saves;

namespace SpireHost.Boot;

/// <summary>
/// The host's own in-memory <see cref="ISaveStore"/> (DECISIONS-H1 D.2). Replaces the game's
/// MockGodotFileIo test double, which appends every call with its arguments (whole file contents) to a
/// public list and therefore leaks without bound in a long-lived process. This store keeps only the latest
/// content per path, records nothing and never touches the disk. Engine thread only.
/// </summary>
public sealed class HostSaveStore : ISaveStore
{
    private readonly Dictionary<string, string> _files = new Dictionary<string, string>(StringComparer.Ordinal);
    private readonly HashSet<string> _dirs = new HashSet<string>(StringComparer.Ordinal);
    private readonly Dictionary<string, DateTimeOffset> _mtimes = new Dictionary<string, DateTimeOffset>(StringComparer.Ordinal);

    public long Writes;
    public long Reads;

    public int FileCount => _files.Count;

    public long TotalChars
    {
        get
        {
            long n = 0;
            foreach (KeyValuePair<string, string> kv in _files)
            {
                n += kv.Value.Length;
            }
            return n;
        }
    }

    private static string Norm(string path) => path.Replace('\\', '/').TrimEnd('/');

    public string? ReadFile(string path)
    {
        Reads++;
        return _files.TryGetValue(Norm(path), out string? v) ? v : null;
    }

    public Task<string?> ReadFileAsync(string path) => Task.FromResult(ReadFile(path));

    public void WriteFile(string path, string content)
    {
        Writes++;
        string p = Norm(path);
        _files[p] = content;
        _mtimes[p] = DateTimeOffset.UnixEpoch;
    }

    public void WriteFile(string path, byte[] content) => WriteFile(path, Encoding.UTF8.GetString(content));

    public Task WriteFileAsync(string path, string content)
    {
        WriteFile(path, content);
        return Task.CompletedTask;
    }

    public Task WriteFileAsync(string path, byte[] content)
    {
        WriteFile(path, content);
        return Task.CompletedTask;
    }

    public bool FileExists(string path) => _files.ContainsKey(Norm(path));

    public bool DirectoryExists(string path) => _dirs.Contains(Norm(path));

    public void DeleteFile(string path)
    {
        string p = Norm(path);
        _files.Remove(p);
        _mtimes.Remove(p);
    }

    public void RenameFile(string sourcePath, string destinationPath)
    {
        string s = Norm(sourcePath);
        if (_files.Remove(s, out string? v))
        {
            _mtimes.Remove(s);
            WriteFile(destinationPath, v);
        }
    }

    public string[] GetFilesInDirectory(string directoryPath)
    {
        string prefix = Norm(directoryPath) + "/";
        List<string> r = new List<string>();
        foreach (string k in _files.Keys)
        {
            if (k.StartsWith(prefix, StringComparison.Ordinal) && k.IndexOf('/', prefix.Length) < 0)
            {
                r.Add(k.Substring(prefix.Length));
            }
        }
        r.Sort(StringComparer.Ordinal);
        return r.ToArray();
    }

    public string[] GetDirectoriesInDirectory(string directoryPath)
    {
        string prefix = Norm(directoryPath) + "/";
        List<string> r = new List<string>();
        foreach (string k in _dirs)
        {
            if (k.StartsWith(prefix, StringComparison.Ordinal) && k.IndexOf('/', prefix.Length) < 0)
            {
                r.Add(k.Substring(prefix.Length));
            }
        }
        r.Sort(StringComparer.Ordinal);
        return r.ToArray();
    }

    public void CreateDirectory(string directoryPath) => _dirs.Add(Norm(directoryPath));

    public void DeleteDirectory(string directoryPath) => _dirs.Remove(Norm(directoryPath));

    public void DeleteTemporaryFiles(string directoryPath)
    {
    }

    public DateTimeOffset GetLastModifiedTime(string path) => _mtimes.TryGetValue(Norm(path), out DateTimeOffset t) ? t : DateTimeOffset.UnixEpoch;

    public int GetFileSize(string path) => _files.TryGetValue(Norm(path), out string? v) ? Encoding.UTF8.GetByteCount(v) : 0;

    public void SetLastModifiedTime(string path, DateTimeOffset time) => _mtimes[Norm(path)] = time;

    public string GetFullPath(string filename) => filename;
}
