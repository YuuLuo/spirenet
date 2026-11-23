using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using SpireHost.Driver;

namespace SpireHost.Replay.V2;

/// <summary>
/// The game's own per-action full-combat-state checksums, inside the run.
///
/// The real client records, for every fight, the state behind each checksum it generates while the fight is in progress
/// (CombatReplayWriter.RecordChecksum, Multiplayer.Replay/CombatReplayWriter.cs:146-161: only while
/// CombatManager.IsInProgress): after every executed game action except the two turn-end actions
/// (RunManager.SendPostActionChecksum, Runs/RunManager.cs:485-491) and at the five turn points of
/// Combat/CombatManager.cs (:580 after player turn start, :627 after enemy turn start, :1157 after enemy turn end,
/// :1313 after player turn phase one end, :1500 after player turn phase two end). Each record is the complete
/// NetFullCombatState: every creature (HP, block, powers), the player (turn, phase, energy, stars, gold, every pile with
/// every card in order, potions, relics with their counters, orbs, arrows, the player rngs, the relic grab bag), the run's
/// rng set, and the synchronizers' counters (next choice ids, next reward ids, last executed hook / action id).
///
/// The comparison enables the game's tracker for the episode it replays (the two read-only interventions of the .mcr
/// replay tool, ReplayInterventions) and keeps the host's states under the SAME condition as the writer. When a fight is
/// over, the host's sequence is compared one to one, in order, with the recorded one: the context string (which action /
/// which turn point) and the serialized state, byte for byte.
///
/// Exactly one field is rewritten before the bytes are compared: the player's net id (the replay file carries the
/// anonymised id of the recording client - CombatReplay.Anonymized - and the host episode its own id; the game's own
/// replay facility avoids the difference by loading the run from the anonymised save). The same id inside a context
/// string ("... for player N", "owner N", "UsePotionAction N ...": GameAction.ToString of the three action types that
/// print it) is masked on both sides.
/// </summary>
public sealed partial class V2Replayer
{
    private ReplayInterventions.Scope? _csScope;
    private ChecksumTracker? _csTracker;
    private Action<NetChecksumData, string, NetFullCombatState>? _onChecksum;
    private readonly Dictionary<AbstractRoom, List<(string Context, NetFullCombatState State)>> _csByRoom =
        new Dictionary<AbstractRoom, List<(string, NetFullCombatState)>>(ReferenceEqualityComparer.Instance);
    private readonly PacketWriter _csA = new PacketWriter { WarnOnGrow = false };
    private readonly PacketWriter _csB = new PacketWriter { WarnOnGrow = false };

    private static readonly Regex PlayerIdInContext = new Regex(@"(for player |owner |UsePotionAction |ReadyToBeginEnemyTurnAction )\d+", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    // AbstractModel.ToString() is "{Id} ({RuntimeHelpers.GetHashCode(this)})" (Models/AbstractModel.cs:1087-1090): the number is the object's
    // identity hash in the process that printed it, not game state.
    private static readonly Regex ModelHashInContext = new Regex(@"\b([A-Z][A-Z_]*\.[A-Z0-9_]+) \(\d+\)", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    // UsePotionAction.ToString() prints its target as "(Creature <LogName>)" (GameActions/UsePotionAction.cs:141-146); LogName is the creature's
    // localized title (Entities.Creatures/Creature.cs:137-149). The target's combat id is printed next to it ("target: N") and is compared.
    private static readonly Regex CreatureNameInContext = new Regex(@"\(Creature [^)]*\)", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// A checksum's context string with the three things masked that are not game state: the player's net id, the identity hash that
    /// AbstractModel.ToString appends to a model id, and the localized creature title of a potion's target.
    /// </summary>
    private static string NormalizeContext(string s)
    {
        s = PlayerIdInContext.Replace(s, "$1<player>");
        s = ModelHashInContext.Replace(s, "$1 (#)");
        return CreatureNameInContext.Replace(s, "(Creature)");
    }

    private void SubscribeChecksums(RunContext ctx)
    {
        if (!_opt.CompareChecksums || _onChecksum != null)
        {
            return;
        }
        RunManager rm = ctx.RunManager;
        _csScope = new ReplayInterventions.Scope(rm, replayRun: false);
        _csTracker = rm.ChecksumTracker;
        _onChecksum = (data, context, state) =>
        {
            // the writer's own condition (CombatReplayWriter.cs:148): a checksum belongs to a fight's recording only while the fight is in progress
            if (_stop || !CombatManager.Instance.IsInProgress)
            {
                return;
            }
            AbstractRoom? room = HostSession.Current?.CurrentEpisode?.Run.CurrentRoom;
            if (room == null)
            {
                return;
            }
            if (!_csByRoom.TryGetValue(room, out List<(string, NetFullCombatState)>? list))
            {
                list = new List<(string, NetFullCombatState)>();
                _csByRoom[room] = list;
            }
            list.Add((context, state));
        };
        _csTracker.ChecksumGenerated += _onChecksum;
    }

    private void UnsubscribeChecksums()
    {
        if (_csTracker != null && _onChecksum != null)
        {
            _csTracker.ChecksumGenerated -= _onChecksum;
        }
        _csScope?.Dispose();
        _csScope = null;
        _csTracker = null;
        _onChecksum = null;
    }

    private ReadOnlySpan<byte> Bytes(PacketWriter w, NetFullCombatState s)
    {
        // what ChecksumTracker.GenerateChecksum hashes (Multiplayer.Game/ChecksumTracker.cs:271-278)
        w.Reset();
        w.Write(s);
        w.ZeroByteRemainder();
        return w.Buffer.AsSpan(0, w.BytePosition);
    }

    /// <summary>The recorded state with the recording client's (anonymised) player id replaced by the host episode's.</summary>
    private static void RewritePlayerId(NetFullCombatState recorded, ulong hostId)
    {
        for (int i = 0; i < recorded.Creatures.Count; i++)
        {
            NetFullCombatState.CreatureState c = recorded.Creatures[i];
            if (c.playerId.HasValue)
            {
                c.playerId = hostId;
                recorded.Creatures[i] = c;
            }
        }
        for (int i = 0; i < recorded.Players.Count; i++)
        {
            NetFullCombatState.PlayerState p = recorded.Players[i];
            p.playerId = hostId;
            recorded.Players[i] = p;
        }
    }

    /// <summary>
    /// The fight is over in the host and in the recording: the recorded per-action states of its replay file against the
    /// host's states of the same fight.
    /// </summary>
    private void CompareChecksums(V2Combat c, CombatReplay? replay, AbstractRoom? hostRoom, string where)
    {
        List<(string Context, NetFullCombatState State)>? host = null;
        if (hostRoom != null)
        {
            _csByRoom.Remove(hostRoom, out host);
        }
        if (!_opt.CompareChecksums || _stop)
        {
            return;
        }
        if (c.McrPath == null)
        {
            Info("combat.checksums.fight_without_replay_file");
            return;
        }
        replay ??= McrFile.Load(c.McrPath).Replay;
        host ??= new List<(string, NetFullCombatState)>();
        List<ReplayChecksumData> rec = replay.checksumData;
        ulong hostId = _ctx!.LocalPlayer.NetId;
        _r.FightsChecksumsCompared++;
        _r.ChecksumsRecorded += rec.Count;
        int n = Math.Min(rec.Count, host.Count);
        bool equal = true;
        for (int i = 0; i < n; i++)
        {
            _r.ChecksumsCompared++;
            string rc = NormalizeContext(rec[i].context);
            string hc = NormalizeContext(host[i].Context);
            if (rc != hc)
            {
                equal = false;
                Diverge("combat.checksum.context", false, $"{where} checksum {i}/{rec.Count}", "the host generated a checksum at another point than the real client (different action or turn point)", rc, hc);
                break;
            }
            RewritePlayerId(rec[i].fullState, hostId);
            if (Bytes(_csA, rec[i].fullState).SequenceEqual(Bytes(_csB, host[i].State)))
            {
                _r.ChecksumsMatched++;
                continue;
            }
            equal = false;
            StateDiff.Result diff = StateDiff.Diff(rec[i].fullState, host[i].State, _opt.MaxDiffs * 4);
            string cls = diff.Lines.Count == 0 ? "combat.checksum.state_bytes" : "combat.checksum.state";
            Diverge(cls, false, $"{where} checksum {i}/{rec.Count} '{rc}'" + (i > 0 ? $" (previous: '{NormalizeContext(rec[i - 1].context)}')" : ""),
                $"the full combat state differs from the state the real client recorded here ({diff.Removed} recorded lines, {diff.Added} host lines differ; ids: {string.Join(",", diff.Ids)})",
                string.Join("\n", diff.Lines.FindAll(l => l.StartsWith("-", StringComparison.Ordinal))), string.Join("\n", diff.Lines.FindAll(l => l.StartsWith("+", StringComparison.Ordinal))));
            break;
        }
        if (equal && rec.Count != host.Count)
        {
            equal = false;
            string next = rec.Count > n ? "recorded next: '" + NormalizeContext(rec[n].context) + "'" : "host next: '" + NormalizeContext(host[n].Context) + "'";
            Diverge("combat.checksum.count", false, where, "the number of checksums of the fight differs (" + next + ")", rec.Count.ToString(), host.Count.ToString());
        }
        if (equal)
        {
            _r.FightsChecksumsEqual++;
        }
    }
}
