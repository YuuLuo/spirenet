using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using HarmonyLib;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Managers;
using MegaCrit.Sts2.Core.Saves.Validation;
using MegaCrit.Sts2.Core.Timeline;
using MegaCrit.Sts2.Core.Timeline.Epochs;
using MegaCrit.Sts2.Core.Unlocks;
using SpireHost.Patches;

namespace SpireHost.Boot;

/// <summary>
/// The fixed "veteran, everything unlocked" profile (DECISIONS-H1 D.1-D.3; recipe: notes\testmode-audit-core.md
/// section 5.4). Built once at boot with the game's own <see cref="ProgressState"/> API, serialised with the
/// game's serializer into an immutable JSON snapshot, checked against the invariants below, and restored
/// from that snapshot at every episode reset, so nothing a run writes into the live ProgressState (encounter
/// statistics, epochs, seen cards, Wongo points) can reach the next run.
///
/// Why not UnlockState.all: it disagrees with the live Progress that TryGetRoomTypeForTutorial,
/// ActModel.GetRandomList and WelcomeToWongos read, and its NumberOfRuns does not survive serialisation.
/// </summary>
public static class VeteranProfile
{
    private static string? _snapshotJson;

    public static bool Installed { get; private set; }

    /// <summary>SHA256 of the snapshot JSON, for reports. Epoch obtain dates are wall-clock, so it differs between processes.</summary>
    public static string SnapshotSha256 { get; private set; } = "";

    public static int SnapshotChars => _snapshotJson?.Length ?? 0;

    /// <summary>ProgressSaveManager.SaveProgress calls that were intercepted (not serialised, not stored).</summary>
    public static long InterceptedProgressWrites;

    public static long Restores;

    public static int EpochCount { get; private set; }
    public static int EncounterCount { get; private set; }
    public static int ActCount { get; private set; }

    /// <summary>Builds the profile, installs it as SaveManager.Instance.Progress, asserts the invariants and intercepts progress writes.</summary>
    public static void Install()
    {
        if (Installed)
        {
            return;
        }
        ProgressState p = Build();
        SerializableProgress ser = p.ToSerializable();
        ser.SchemaVersion = SaveManager.Instance.GetLatestSchemaVersion<SerializableProgress>();
        _snapshotJson = JsonSerializationUtility.ToJson(ser);
        SnapshotSha256 = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(_snapshotJson))).ToLowerInvariant();

        MethodInfo save = AccessTools.Method(typeof(ProgressSaveManager), nameof(ProgressSaveManager.SaveProgress))
            ?? throw new MissingMethodException("ProgressSaveManager.SaveProgress");
        PatchRegistry.Prefix("progress-save-intercept", save, PatchRegistry.M(typeof(VeteranProfile), nameof(SaveProgressPrefix)), PatchCategory.Plumbing,
            "ProgressSaveManager.SaveProgress serialises the whole ProgressState to JSON and writes progress.save (after every won combat and on every map move, " +
            "SaveManager.cs:570-581, CombatManager.cs:1061-1063; not gated by ShouldSave). The host never persists progress: the write is skipped and counted. " +
            "The live in-memory ProgressState is still updated by the game during the run (WelcomeToWongos etc. read it) and is replaced by the pinned veteran snapshot at every reset.");
        PatchRegistry.Note("profile-veteran-recipe", "MegaCrit.Sts2.Core.Saves.SaveManager::Progress (public setter)", "game API calls", PatchCategory.Plumbing,
            "Progress = veteran profile built with ProgressState API (all epochs Revealed, all acts discovered, every encounter / enemy with statistics, every character with wins+losses and MaxAscension 10, " +
            "FTUE off), serialised once with the game's serializer and restored from that immutable snapshot at every episode reset (ProgressState.FromSerializable). WongoPoints is set per run (DECISIONS-H1 D.3).");

        Installed = true;
        Restore(0);
        AssertInvariants(SaveManager.Instance.Progress);
    }

    public static bool SaveProgressPrefix()
    {
        InterceptedProgressWrites++;
        return false;
    }

    private static ProgressState Build()
    {
        ProgressState p = ProgressState.CreateDefault();
        foreach (string id in EpochModel.AllEpochIds)
        {
            p.ObtainEpochOverride(id, EpochState.Revealed);
        }
        List<CharacterModel> characters = ModelDb.AllCharacters.ToList();
        ModelId anyCharacter = characters[0].Id;
        foreach (ActModel a in ModelDb.Acts)
        {
            p.MarkActAsSeen(a.Id);
        }
        foreach (EncounterModel e in ModelDb.AllEncounters)
        {
            EncounterStats s = p.GetOrCreateEncounterStats(e.Id);
            if (s.FightStats.Count == 0)
            {
                s.FightStats.Add(new FightStats { Character = anyCharacter, Wins = 1, Losses = 0 });
            }
        }
        foreach (MonsterModel m in ModelDb.Monsters)
        {
            EnemyStats s = p.GetOrCreateEnemyStats(m.Id);
            if (s.FightStats.Count == 0)
            {
                s.FightStats.Add(new FightStats { Character = anyCharacter, Wins = 1, Losses = 0 });
            }
        }
        foreach (CharacterModel c in characters)
        {
            CharacterStats cs = p.GetOrCreateCharacterStats(c.Id);
            cs.TotalWins = 1;
            cs.TotalLosses = 1;
            cs.MaxAscension = 10;
            cs.PreferredAscension = 10;
        }
        foreach (CardModel c in ModelDb.AllCards)
        {
            p.MarkCardAsSeen(c.Id);
        }
        foreach (RelicModel r in ModelDb.AllRelics)
        {
            p.MarkRelicAsSeen(r.Id);
        }
        foreach (PotionModel po in ModelDb.AllPotions)
        {
            p.MarkPotionAsSeen(po.Id);
        }
        foreach (EventModel ev in ModelDb.AllEvents)
        {
            p.MarkEventAsSeen(ev.Id);
        }
        p.EnableFtues = false;
        p.WongoPoints = 0;
        p.MaxMultiplayerAscension = 10;
        p.PreferredMultiplayerAscension = 10;
        return p;
    }

    /// <summary>
    /// Replaces SaveManager.Instance.Progress with a fresh ProgressState parsed from the immutable snapshot.
    /// <paramref name="wongoPoints"/> is the per-run WelcomeToWongos point total (hidden cross-run state in the real game).
    /// </summary>
    public static void Restore(int wongoPoints)
    {
        if (_snapshotJson == null)
        {
            throw new InvalidOperationException("VeteranProfile.Install was not called.");
        }
        ReadSaveResult<SerializableProgress> r = JsonSerializationUtility.FromJson<SerializableProgress>(_snapshotJson);
        if (!r.Success || r.SaveData == null)
        {
            throw new InvalidOperationException("Veteran profile snapshot failed to parse: " + r.Status + " " + r.ErrorMessage);
        }
        DeserializationContext ctx = new DeserializationContext();
        ProgressState p = ProgressState.FromSerializable(r.SaveData, ctx);
        if (ctx.Errors.Count > 0)
        {
            throw new InvalidOperationException("Veteran profile snapshot produced validation errors: " + ctx.Errors[0]);
        }
        p.WongoPoints = wongoPoints;
        SaveManager.Instance.Progress = p;
        Restores++;
        // Cheap per-reset subset of the invariants (the full set is asserted at boot on the same snapshot).
        if (p.NumberOfRuns <= 0 || p.EnableFtues || p.Epochs.Count < EpochCount)
        {
            throw new InvalidOperationException("Veteran profile invariants broken after restore.");
        }
    }

    /// <summary>
    /// WelcomeToWongos reads Progress.WongoPoints (hidden cross-run state of the real profile). The host gives every
    /// run its own value: uniform in [0, 1999], a pure function of the run seed (DECISIONS-H1 D.3). Public for
    /// verification tools that restore the profile themselves (replay request R4).
    /// </summary>
    public static int WongoPointsFor(string seed)
    {
        ulong z = MegaCrit.Sts2.Core.Helpers.StringHelper.GetDeterministicHashCode(seed) ^ 0x574F4E474F504F49UL; // "WONGOPOI"
        z += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (int)(z % 2000UL);
    }

    /// <summary>The unlock state exactly as the real lobby hands it to Player.CreateForNewRun: new UnlockState(Progress) after one ToSerializable/FromSerializable round trip (StartRunLobby.cs:198, :955).</summary>
    public static UnlockState CreateUnlockState()
    {
        UnlockState fromProgress = SaveManager.Instance.GenerateUnlockStateFromProgress();
        return UnlockState.FromSerializable(fromProgress.ToSerializable());
    }

    /// <summary>
    /// The invariants that make the TestMode gates of run start equivalent to their real-game branch
    /// (design-review-eng.md section 3.2, testmode-audit-core.md section 5). Throws on the first violation.
    /// </summary>
    public static void AssertInvariants(ProgressState p)
    {
        void Require(bool ok, string what)
        {
            if (!ok)
            {
                throw new InvalidOperationException("Veteran profile invariant violated: " + what);
            }
        }

        IReadOnlyList<string> epochs = EpochModel.AllEpochIds;
        foreach (string id in epochs)
        {
            Require(p.IsEpochRevealed(id), "epoch not Revealed: " + id);
        }
        UnlockState unlock = new UnlockState(p);
        Require(unlock.EpochUnlockCount() == epochs.Count, $"UnlockState epochs {unlock.EpochUnlockCount()} != {epochs.Count}");
        Require(unlock.IsEpochRevealed<NeowEpoch>(), "NeowEpoch not revealed (StartedWithNeow would be false)");
        Require(unlock.NumberOfRuns > 0 && p.NumberOfRuns > 0, "NumberOfRuns must be > 0 (tutorial branches)");
        Require(!p.EnableFtues, "FTUEs must be off");

        int acts = 0;
        foreach (ActModel a in ModelDb.Acts)
        {
            acts++;
            Require(p.DiscoveredActs.Contains(a.Id), "act not discovered: " + a.Id.Entry + " (ActModel.GetRandomList would force it)");
            Require(a.IsUnlocked(unlock), "act not unlocked: " + a.Id.Entry);
            foreach (EncounterModel boss in a.BossDiscoveryOrder)
            {
                Require(unlock.HasSeenEncounter(boss), "boss not seen: " + boss.Id.Entry + " (ApplyDiscoveryOrderModifications would force it)");
            }
        }
        int encounters = 0;
        foreach (EncounterModel e in ModelDb.AllEncounters)
        {
            encounters++;
            Require(unlock.HasSeenEncounter(e), "encounter not seen: " + e.Id.Entry);
        }
        foreach (MonsterModel m in ModelDb.Monsters)
        {
            Require(p.EnemyStats.TryGetValue(m.Id, out EnemyStats? es) && es.FightStats.Count > 0, "enemy without fight statistics: " + m.Id.Entry);
        }
        int characters = 0;
        foreach (CharacterModel c in ModelDb.AllCharacters)
        {
            characters++;
            CharacterStats? cs = p.GetStatsForCharacter(c.Id);
            Require(cs != null && cs.MaxAscension == 10 && cs.TotalWins + cs.TotalLosses > 0, "character stats incomplete: " + c.Id.Entry);
        }
        Require(unlock.Characters.Count() == characters, "not every character is unlocked");
        Require(unlock.SharedAncients.Count() == ModelDb.AllSharedAncients.Count(), "not every shared ancient is unlocked");

        // The lobby round trip must not lose anything.
        UnlockState round = UnlockState.FromSerializable(unlock.ToSerializable());
        Require(round.EpochUnlockCount() == unlock.EpochUnlockCount() && round.NumberOfRuns == unlock.NumberOfRuns, "UnlockState round trip changed epochs / NumberOfRuns");
        foreach (EncounterModel e in ModelDb.AllEncounters)
        {
            Require(round.HasSeenEncounter(e), "UnlockState round trip lost encounter " + e.Id.Entry);
        }

        EpochCount = epochs.Count;
        EncounterCount = encounters;
        ActCount = acts;
    }

    /// <summary>Summary for --boot / reports.</summary>
    public static object Describe()
    {
        UnlockState u = new UnlockState(SaveManager.Instance.Progress);
        return new
        {
            installed = Installed,
            snapshot_chars = SnapshotChars,
            snapshot_sha256 = SnapshotSha256,
            epochs = EpochCount,
            encounters = EncounterCount,
            acts = ActCount,
            unlocked_cards = u.Cards.Count(),
            unlocked_relics = u.Relics.Count(),
            unlocked_potions = u.Potions.Count(),
            shared_ancients = u.SharedAncients.Count(),
            number_of_runs = u.NumberOfRuns,
            restores = Restores,
            intercepted_progress_writes = InterceptedProgressWrites,
        };
    }
}
