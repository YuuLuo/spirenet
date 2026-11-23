using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Managers;
using SpireHost.Patches;

namespace SpireHost.Boot;

public static class LogBootstrap
{
    public static LogLevel MinLevel = LogLevel.Error;
    public static bool ShowErrors = true;

    public static void Install(string level)
    {
        MinLevel = level switch
        {
            "debug" => LogLevel.Debug,
            "verydebug" => LogLevel.VeryDebug,
            "info" => LogLevel.Info,
            "warn" => LogLevel.Warn,
            _ => LogLevel.Error,
        };
        // The game's own knobs: with these set, Logger.WillLog rejects everything below MinLevel before the
        // printer is reached. (Call sites still build their interpolated strings; see the profile section.)
        Logger.GlobalLogLevel = MinLevel;
        foreach (LogType t in Enum.GetValues<LogType>())
        {
            Logger.SetLogLevelForType(t, MinLevel);
        }
        TaskHelper.UnobservedFault += e =>
        {
            HostFaults.Record(e);
        };
        PatchRegistry.Note("log-levels", "MegaCrit.Sts2.Core.Logging.Logger::GlobalLogLevel / SetLogLevelForType", "game API call", PatchCategory.Plumbing,
            "Log level raised to Error by default through the game's public API (observational only).");
        PatchRegistry.Note("unobserved-fault-hook", "MegaCrit.Sts2.Core.Helpers.TaskHelper::UnobservedFault", "event subscription", PatchCategory.Plumbing,
            "Faults of fire-and-forget game tasks (TaskHelper.RunSafely) are recorded by the host and fail the episode instead of vanishing.");
    }
}

public static class SaveBootstrap
{
    public const int ProfileId = 1;

    public static HostSaveStore? Store { get; private set; }

    /// <summary>
    /// Installs a SaveManager over the host's own in-memory store (HostSaveStore) through the game's public
    /// test seam SaveManager.MockInstanceForTesting. The game's default TestMode store (MockGodotFileIo) is
    /// never constructed: it records every call with its arguments and leaks ~29 KB per won combat.
    /// </summary>
    public static void Install()
    {
        HostSaveStore store = new HostSaveStore();
        SaveManager sm = new SaveManager(store, forceSynchronous: true);
        SaveManager.MockInstanceForTesting(sm);
        if (!ReferenceEquals(SaveManager.Instance, sm) || !ReferenceEquals(Backend(), store))
        {
            throw new InvalidOperationException("SaveManager is not backed by the host's in-memory store.");
        }
        Store = store;
        sm.InitProfileId(ProfileId);
        sm.InitSettingsDataForTest();
        sm.InitPrefsDataForTest();
        PatchRegistry.Note("net-loading-handle-release", "MegaCrit.Sts2.Core.Multiplayer.Game.NetLoadingHandle::Release (public static)", "game API call", PatchCategory.Plumbing,
            "After RunManager.CleanUp the host calls NetLoadingHandle.Release(netService): the static load-count dictionary otherwise keeps one NetSingleplayerGameService per run alive forever.");
        PatchRegistry.Note("save-store-in-memory", "MegaCrit.Sts2.Core.Saves.SaveManager::MockInstanceForTesting(new SaveManager(HostSaveStore, forceSynchronous: true))", "game API call", PatchCategory.Plumbing,
            "Saves, settings, prefs and progress live in the host's own in-memory ISaveStore (latest content per path, no call log, no disk). " +
            "The game's default TestMode store (MockGodotFileIo) is not used because it records every call with its arguments and grows without bound.");
    }

    public static object Backend()
    {
        FieldInfo store = typeof(SaveManager).GetField("_saveStore", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingFieldException("SaveManager._saveStore");
        return store.GetValue(SaveManager.Instance)!;
    }
}

/// <summary>
/// Builds the game's LocManager from a pinned English bundle (extracted from the game's own
/// res://localization/eng/*.json) without Godot FileAccess. The game's own formatter set
/// (LocManager.LoadLocFormatters) is used unchanged, so LocString formatting runs original code.
/// </summary>
public static class LocalizationBootstrap
{
    public const string BundleName = "fair-localization-eng.json";

    /// <summary>
    /// SHA256 -> game version of the accepted bundles: the v0.113.0 bundle, and the v0.113.1 bundle that
    /// host\tools\prepare_game_dir.py builds from SlayTheSpire2.pck. Same 46 tables and 7749 keys; they differ only
    /// in four values of the table "epochs" (timeline text).
    /// </summary>
    public static readonly IReadOnlyDictionary<string, string> AcceptedBundles = new Dictionary<string, string>
    {
        { "3c157b72e3f2f3a462b0caf854168a65ca165ddb1891ef6d425c39db41c9eae1", "v0.113.0" },
        { "7ed2a457ca4073b5f7b2b8abe497b53da428e63a02b838aefb6230106ccb669e", "v0.113.1" },
    };

    public static int Tables;
    public static int Entries;

    public static void Install()
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(GameAssemblies.GameDir, BundleName));
        string sha = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        if (!AcceptedBundles.TryGetValue(sha, out string? version))
        {
            throw new InvalidOperationException("Localization bundle hash mismatch: " + sha + " (build host\\game with host\\tools\\prepare_game_dir.py)");
        }
        using JsonDocument doc = JsonDocument.Parse(bytes);
        JsonElement root = doc.RootElement;
        if (root.GetProperty("version").GetString() != version || root.GetProperty("language").GetString() != "eng")
        {
            throw new InvalidOperationException("Localization bundle is not " + version + "/eng.");
        }
        if (LocManager.Instance != null)
        {
            throw new InvalidOperationException("LocManager already exists.");
        }
        Type t = typeof(LocManager);
        LocManager loc = (LocManager)RuntimeHelpers.GetUninitializedObject(t);
        Dictionary<string, LocTable> tables = new Dictionary<string, LocTable>();
        int entries = 0;
        foreach (JsonProperty table in root.GetProperty("tables").EnumerateObject())
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            foreach (JsonProperty p in table.Value.EnumerateObject())
            {
                dict[p.Name] = p.Value.GetString() ?? "";
            }
            entries += dict.Count;
            tables.Add(table.Name, new LocTable(table.Name, dict, null));
        }
        SetField(loc, "_tables", tables);
        SetField(loc, "_engTables", tables);
        SetField(loc, "_languageKeyCount", new Dictionary<string, int> { { "eng", entries } });
        FieldInfo callbacks = t.GetField("_localeChangeCallbacks", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingFieldException("LocManager._localeChangeCallbacks");
        callbacks.SetValue(loc, Activator.CreateInstance(callbacks.FieldType));
        SetProperty(loc, "CultureInfo", CultureInfo.GetCultureInfo("en"));
        SetProperty(loc, "Language", "eng");
        SetProperty(loc, "StringComparer", StringComparer.Create(CultureInfo.GetCultureInfo("en"), CompareOptions.None));
        SetProperty(loc, "ValidationErrors", Array.Empty<LocValidationError>());
        PropertyInfo instance = t.GetProperty("Instance", BindingFlags.Static | BindingFlags.Public)!;
        instance.GetSetMethod(nonPublic: true)!.Invoke(null, new object[] { loc });
        MethodInfo formatters = t.GetMethod("LoadLocFormatters", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingMethodException("LocManager.LoadLocFormatters");
        formatters.Invoke(loc, null);
        Tables = tables.Count;
        Entries = entries;
        PatchRegistry.Note("localization-managed-tables", "MegaCrit.Sts2.Core.Localization.LocManager (constructed without its Godot file loader)", "reflection construct", PatchCategory.Plumbing,
            $"LocManager.Instance is populated from a pinned eng bundle ({tables.Count} tables, {entries} entries, sha256 one of LocalizationBootstrap.AcceptedBundles) instead of res:// FileAccess. " +
            "Formatters are the game's own (LoadLocFormatters). Text never reaches the policy; it exists so that game code that formats strings behaves as in the real game.");
    }

    private static void SetField(object target, string name, object value)
    {
        FieldInfo f = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingFieldException(target.GetType().FullName, name);
        f.SetValue(target, value);
    }

    private static void SetProperty(object target, string name, object value)
    {
        PropertyInfo p = target.GetType().GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new MissingMemberException(target.GetType().FullName, name);
        MethodInfo set = p.GetSetMethod(nonPublic: true) ?? throw new MissingMethodException(target.GetType().FullName, "set_" + name);
        set.Invoke(target, new[] { value });
    }
}
