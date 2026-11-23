using System;
using System.Collections.Generic;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Powers;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using SpireHost.Driver;
using SpireHost.Engine;
using SpireHost.Obs.Schema;
using SpireHost.Obs.Vocab;

namespace SpireHost.Obs.Encoding;

public sealed partial class ObsEncoder
{
    private static readonly Creature[] NoCreatures = Array.Empty<Creature>();

    // =====================================================================================================
    //  SEATS 23-31: public combat history of the current turn / combat
    // =====================================================================================================

    /// <summary>MelodyCmd.PlayedThisTurn(player, Wah/Cho/Ki) (Commands/MelodyCmd.cs:33), shown as symbols by NFoolCombatUi.</summary>
    private static int MelodyBits(Player p)
    {
        if (p.Character.Id.Entry != "FOOL")
        {
            return 0; // only the Fool's combat UI shows the melody; the history query is pointless for others
        }
        int bits = 0;
        if (MelodyCmd.PlayedThisTurn(p, CardTag.Wah))
        {
            bits |= 1;
        }
        if (MelodyCmd.PlayedThisTurn(p, CardTag.Cho))
        {
            bits |= 2;
        }
        if (MelodyCmd.PlayedThisTurn(p, CardTag.Ki))
        {
            bits |= 4;
        }
        return bits;
    }

    /// <summary>
    /// Public-history counters (schema 5.1 cols 24-31), with the game's own "this turn" predicate
    /// CombatHistoryEntry.HappenedThisTurn (Combat.History/CombatHistoryEntry.cs:35) as cards like Finisher use it.
    /// </summary>
    private const int HPlayed = 0, HAttacks = 1, HSkills = 2, HPowers = 3, HDrawn = 4, HEnergy = 5, HStars = 6, HHpLoss = 7, HCols = 8;
    private readonly int[] _history = new int[4 * HCols];

    /// <summary>
    /// The public-history counters of every seat in one pass over the combat history (the per-seat predicates below are
    /// those of the per-seat pass this replaces: same entry types, same tests, CombatHistoryEntry.HappenedThisTurn evaluated
    /// once per entry instead of once per entry and seat; every test is a pure read of the quiescent state).
    /// </summary>
    private void CountHistory()
    {
        CombatState cs = _cs!;
        IReadOnlyList<Player> players = _run.Players;
        int n = _nSeats;
        if (n > 4)
        {
            throw new InvalidOperationException("more than 4 seats");
        }
        Array.Clear(_history);
        foreach (CombatHistoryEntry e in CombatManager.Instance.History._entries)
        {
            switch (e)
            {
                case CardPlayFinishedEntry f:
                {
                    int s = -1;
                    for (int k = 0; k < n; k++)
                    {
                        if (f.CardPlay.Player == players[k])
                        {
                            s = k;
                            break;
                        }
                    }
                    if (s < 0 || !f.HappenedThisTurn(cs))
                    {
                        break;
                    }
                    // a player is one seat: the first match is the only one (Player identity)
                    int o = s * HCols;
                    _history[o + HPlayed]++;
                    switch (f.CardPlay.Card.Type)
                    {
                        case CardType.Attack:
                            _history[o + HAttacks]++;
                            break;
                        case CardType.Skill:
                            _history[o + HSkills]++;
                            break;
                        case CardType.Power:
                            _history[o + HPowers]++;
                            break;
                    }
                    break;
                }
                case CardDrawnEntry dr:
                {
                    int s = SeatOfCreature(players, n, dr.Actor);
                    if (s >= 0 && dr.HappenedThisTurn(cs))
                    {
                        _history[s * HCols + HDrawn]++;
                    }
                    break;
                }
                case EnergySpentEntry es:
                {
                    int s = SeatOfCreature(players, n, es.Actor);
                    if (s >= 0 && es.HappenedThisTurn(cs))
                    {
                        _history[s * HCols + HEnergy] += es.Amount;
                    }
                    break;
                }
                case StarsModifiedEntry sm:
                {
                    int s = SeatOfCreature(players, n, sm.Actor);
                    if (s >= 0 && sm.Amount > 0 && sm.HappenedThisTurn(cs))
                    {
                        _history[s * HCols + HStars] += sm.Amount;
                    }
                    break;
                }
                case DamageReceivedEntry dmg:
                {
                    int s = SeatOfCreature(players, n, dmg.Receiver);
                    if (s >= 0 && dmg.Result.UnblockedDamage > 0)
                    {
                        _history[s * HCols + HHpLoss]++;
                    }
                    break;
                }
            }
        }
    }

    /// <summary>The seat whose Player.Creature == <paramref name="c"/> (a creature is one player's), or -1.</summary>
    private static int SeatOfCreature(IReadOnlyList<Player> players, int n, Creature? c)
    {
        for (int k = 0; k < n; k++)
        {
            if (c == players[k].Creature)
            {
                return k;
            }
        }
        return -1;
    }

    private void HistoryCounters(int seat, Span<int> row)
    {
        int o = seat * HCols;
        row[SeatC.CardsPlayedTurn] = _history[o + HPlayed];
        row[SeatC.AttacksPlayedTurn] = _history[o + HAttacks];
        row[SeatC.SkillsPlayedTurn] = _history[o + HSkills];
        row[SeatC.PowersPlayedTurn] = _history[o + HPowers];
        row[SeatC.CardsDrawnTurn] = _history[o + HDrawn];
        row[SeatC.EnergySpentTurn] = _history[o + HEnergy];
        row[SeatC.StarsGainedTurn] = _history[o + HStars];
        row[SeatC.HpLossEventsCombat] = _history[o + HHpLoss];
    }

    // =====================================================================================================
    //  CREATURES / POWERS / INTENTS (schema 5.4-5.6)
    // =====================================================================================================

    private void EncodeCreatures()
    {
        CombatState cs = _cs!;
        List<Creature> allies = cs._allies;
        List<Creature> enemies = cs._enemies;
        for (int i = 0; i < allies.Count; i++)
        {
            AddCreatureRow(allies[i], i, enemy: false);
        }
        for (int i = 0; i < enemies.Count; i++)
        {
            AddCreatureRow(enemies[i], i, enemy: true);
        }
        // powers and intents need every creature row (applier pointers), so they come second, in creature-row order
        int n = _w.Count(T.Creatures);
        bool showIntents = true;
        bool enemyTurn = cs.CurrentSide != CombatSide.Player;
        for (int r = 0; r < n; r++)
        {
            Creature c = r < allies.Count ? allies[r] : enemies[r - allies.Count];
            EncodePowers(c, r);
            if (showIntents && c.IsEnemy && c.IsAlive && c.Monster != null)
            {
                // DECISIONS-H1 F.6: during the enemy turn the intents of the enemies that have BEGUN their move are hidden
                // (the UI hides an intent right before Creature.TakeTurn, CombatManager.cs:1142-1147 -> NCreature.PerformIntent,
                // Nodes.Combat/NCreature.cs:482-496); the others stay visible. The core tracks it (RunContext.IntentPerformed,
                // enemy-turn-intent-performed): the combat history's MonsterPerformedMove entry is written only AFTER the move
                // (Models/MonsterModel.cs:543-545), i.e. too late for a prompt raised inside the move (KnowledgeDemon).
                if (!enemyTurn || !_ctx.IntentPerformed(c))
                {
                    EncodeIntents(c, r);
                    if (Options.Forecast)
                    {
                        if (HostProfile.On)
                        {
                            HostProfile.EncLap(2);
                        }
                        EncodeForecast(c, r);
                        if (HostProfile.On)
                        {
                            HostProfile.EncLap(ProfileForecast);
                        }
                    }
                }
            }
        }
        for (int s = 0; s < _nSeats; s++)
        {
            _w.Set(T.Seats, s, SeatC.CreatureRow, _seatCreatureRow[s]);
        }
    }

    private void AddCreatureRow(Creature c, int pos, bool enemy)
    {
        int r = _w.AddRow(T.Creatures);
        _creatureRow[c] = r;
        Span<int> row = _w.Row(T.Creatures, r);
        int kind;
        int seat;
        if (c.IsPlayer)
        {
            kind = CreatureKind.Player;
            seat = SeatOf(c.Player);
            if (seat >= 0 && seat < 4)
            {
                _seatCreatureRow[seat] = r;
            }
        }
        else if (!enemy)
        {
            kind = CreatureKind.Pet;
            seat = SeatOf(c.PetOwner);
        }
        else
        {
            kind = CreatureKind.Enemy;
            seat = -1;
        }
        row[CreatureC.Kind] = kind;
        row[CreatureC.Seat] = seat;
        row[CreatureC.Pos] = pos;
        row[CreatureC.ModelId] = c.IsPlayer ? _v.Model("character", c.Player!.Character) : _v.Model("monster", c.Monster);
        // HP numbers are shown only when HpDisplay.ShowsNumbers() (Nodes.Combat/NHealthBar.cs:414-418); the 9999-HP
        // sentinel creatures (pets / NPCs, catalog hp_sentinel) are written as -1 too (schema 5.4, errata E-20).
        bool showHp = c.HpDisplay.ShowsNumbers() && (c.IsPlayer || c.MaxHp < 9999);
        row[CreatureC.Hp] = showHp ? c.CurrentHp : -1;
        row[CreatureC.MaxHp] = showHp ? c.MaxHp : -1;
        row[CreatureC.HpDisplay] = Enum(Ev.HpDisplay, (int)c.HpDisplay);
        row[CreatureC.Block] = c.Block;
        row[CreatureC.Alive] = c.IsAlive ? 1 : 0;
        row[CreatureC.Hittable] = KnownHittable(c, r) ? 1 : 0;
        row[CreatureC.SlotId] = string.IsNullOrEmpty(c.SlotName) ? 0 : _v.Id("slot_name", c.SlotName);
        row[CreatureC.PowerOff] = -1;
        row[CreatureC.PowerCnt] = 0;
        row[CreatureC.IntentOff] = -1;
        row[CreatureC.IntentCnt] = 0;
        PoisonPower? poison = c.GetPower<PoisonPower>();
        row[CreatureC.PoisonNextTurn] = poison == null ? 0 : _hooks.PoisonDamageNextTurn(poison);
    }

    /// <summary>
    /// Creature.IsHittable, from the COMBAT handler's CombatContext.Hittable when it evaluated it for this decision (same
    /// creature ordinal = CREATURES row: allies, then enemies; quiescent engine), else from the game.
    /// </summary>
    private bool KnownHittable(Creature c, int row)
    {
        if (!Options.ExactGameCalls && _d?.Context is CombatContext cc && row < cc.Hittable.Count && ReferenceEquals(cc.Creatures[row], c) && cc.Hittable[row] != 0)
        {
            return cc.Hittable[row] == 1;
        }
        return _hooks.IsHittable(c);
    }

    private int CreatureRowOf(Creature? c)
    {
        if (c == null)
        {
            return -1;
        }
        return _creatureRow.TryGetValue(c, out int r) ? r : -1;
    }

    private void EncodePowers(Creature c, int ownerRow)
    {
        IReadOnlyList<PowerModel> powers = c.Powers;
        int first = _w.Count(T.Powers);
        int order = 0;
        ulong? savedNetId = LocalContext.NetId;
        try
        {
            for (int i = 0; i < powers.Count; i++)
            {
                PowerModel pw = powers[i];
                int mask = VisibleMask(pw);
                if (mask == 0)
                {
                    continue; // NPowerContainer only creates nodes for visible powers (Nodes.Combat/NPowerContainer.cs:80-89)
                }
                int r = _w.AddRow(T.Powers);
                Span<int> row = _w.Row(T.Powers, r);
                row[PowerC.OwnerRow] = ownerRow;
                row[PowerC.PowerId] = _v.Model("power", pw);
                // NPower.RefreshAmount: only Counter powers print DisplayAmount (Nodes.Combat/NPower.cs:167-172)
                row[PowerC.Amount] = pw.StackType == PowerStackType.Counter ? pw.DisplayAmount : 0;
                row[PowerC.StackType] = Enum(Ev.PowerStackType, (int)pw.StackType);
                row[PowerC.TypeNow] = Enum(Ev.PowerType, (int)pw.TypeForCurrentAmount);
                row[PowerC.ApplierRow] = CreatureRowOf(pw.Applier);
                row[PowerC.VisibleMask] = mask;
                row[PowerC.Order] = order++;
                AddModelVars(T.Powers, r, pw, pw.DynamicVars, 0);
            }
        }
        finally
        {
            LocalContext.NetId = savedNetId;
        }
        int n = _w.Count(T.Powers) - first;
        if (n > 0)
        {
            _w.Set(T.Creatures, ownerRow, CreatureC.PowerOff, first);
            _w.Set(T.Creatures, ownerRow, CreatureC.PowerCnt, n);
        }
    }

    /// <summary>bit s = PowerModel.IsVisible as seen by seat s (LocalContext.NetId = seat s; Models/PowerModel.cs:141-151).</summary>
    private int VisibleMask(PowerModel pw)
    {
        int mask = 0;
        for (int s = 0; s < _nSeats; s++)
        {
            if (Options.ViewMode != 0 && s != Options.ViewSeat)
            {
                continue;
            }
            LocalContext.NetId = _run.Players[s].NetId;
            if (pw.IsVisible)
            {
                mask |= 1 << s;
            }
        }
        return mask;
    }

    private void EncodeIntents(Creature enemy, int ownerRow)
    {
        IReadOnlyList<AbstractIntent> intents = enemy.Monster!.NextMove.Intents;
        if (intents == null || intents.Count == 0)
        {
            return;
        }
        int first = _w.Count(T.Intents);
        ulong? savedNetId = LocalContext.NetId;
        try
        {
            for (int i = 0; i < intents.Count; i++)
            {
                int r = _w.AddRow(T.Intents);
                IntentColumns(_w.Row(T.Intents, r), ownerRow, i, intents[i], enemy);
            }
        }
        finally
        {
            LocalContext.NetId = savedNetId;
        }
        _w.Set(T.Creatures, ownerRow, CreatureC.IntentOff, first);
        _w.Set(T.Creatures, ownerRow, CreatureC.IntentCnt, _w.Count(T.Intents) - first);
    }

    /// <summary>
    /// INTENTS columns 0-9 of intent <paramref name="idx"/> (FORECAST columns 0-9 are the same). Sets LocalContext.NetId per
    /// seat: the caller restores it.
    /// </summary>
    private void IntentColumns(Span<int> row, int ownerRow, int idx, AbstractIntent it, Creature enemy)
    {
        row[IntentC.OwnerRow] = ownerRow;
        row[IntentC.Idx] = idx;
        row[IntentC.IntentType] = Enum(Ev.IntentType, (int)it.IntentType);
        row[IntentC.Hits] = it is AttackIntent atk ? atk.Repeats : 0;
        row[IntentC.CardCount] = it is StatusIntent st ? st.CardCount : 0;
        for (int s = 0; s < 4; s++)
        {
            row[IntentC.DmgSeat0 + s] = -1;
        }
        if (it is AttackIntent attack)
        {
            for (int s = 0; s < _nSeats && s < 4; s++)
            {
                if (Options.ViewMode != 0 && s != Options.ViewSeat)
                {
                    continue;
                }
                // AttackIntent.GetSingleDamage computes against LocalContext.GetMe (MonsterMoves.Intents/AttackIntent.cs:84-93)
                LocalContext.NetId = _run.Players[s].NetId;
                row[IntentC.DmgSeat0 + s] = _hooks.SingleDamage(attack, NoCreatures, enemy);
            }
        }
        row[IntentC.Flags] = it.HasIntentTip ? 1 : 0;
    }

    // =====================================================================================================
    //  VARS of non-card models (relics, potions, powers): printed DynamicVars only (schema 5.3.3)
    // =====================================================================================================

    private void AddModelVars(int ownerTable, int ownerRow, AbstractModel model, DynamicVarSet vars, int flags)
    {
        if (vars.Count == 0 && model is not PowerModel)
        {
            return;
        }
        string[] printed = _tpl.PrintedVars(model);
        for (int i = 0; i < printed.Length; i++)
        {
            if (model is PowerModel pw && printed[i] == Templates.PowerAmount)
            {
                // the tooltip's {Amount} when the icon does not show it (Templates.PowerPrinted, errata E-34)
                int a = pw.Amount;
                _w.AddVar(ownerTable, ownerRow, -1, _v.Id("var_name", Templates.PowerAmount), a, a, a, flags);
                continue;
            }
            if (!vars.TryGetValue(printed[i], out DynamicVar? dv) || dv is BoolVar || dv is StringVar)
            {
                continue;
            }
            int b = (int)dv.BaseValue;
            _w.AddVar(ownerTable, ownerRow, -1, _v.Id("var_name", printed[i]), (int)dv.PreviewValue, b, b, flags | (dv.WasJustUpgraded ? VarFlag.JustUpgraded : 0));
        }
    }
}
