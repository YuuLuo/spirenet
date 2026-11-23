using System;
using System.Collections.Generic;
using System.Numerics;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Relics;
using SpireHost.Engine;
using SpireHost.Obs.Schema;

namespace SpireHost.Obs.Encoding;

/// <summary>
/// The calc trailer tables TURN / FOE / ROUTE (notes/calc/contract.md; RESET flag CALC): turn-end forecast of the acting
/// seat, per-enemy kill sums, per-reachable-node route summary.
///
/// Every number is a pure function of the rows this encoder has just staged (the v1 tables: GLOBAL, SEATS, CREATURES,
/// POWERS, INTENTS, RELICS, CARDS + their staged VARS, MAP_NODES, MAP_EDGES) plus fixed rule knowledge: the vocab ids of a
/// closed list of names, Orichalcum's canonical Block value, and of the acting seat's hand cards CardModel.TargetType and
/// CardModel.HasTurnEndInHandEffect (getters). Nothing of the game is written, no RNG, pile order or hidden state is read.
/// </summary>
public sealed partial class ObsEncoder
{
    // ---- vocab ids, resolved once (a missing name throws) ----
    private bool _calcReady;
    private int _cPlating, _cConstrict, _cBuffer, _cIntangible, _cHardenedShell, _cSlippery, _cSkittish;
    private int _cOrichalcum, _cTungstenRod, _cBeatingRemnant;
    private int _cDamage, _cCalcDamage, _cHitCount, _cCalcHits, _cBlock, _cCalcBlock, _cHpLoss;
    private int _cHidden, _cUnknownIntent, _cPlayerSide;
    private int _cRest, _cShop, _cElite, _cMonster, _cUnknownNode;
    private int _cOrichalcumBlock;

    private int CalcId(string cat, string name)
    {
        if (!_v[cat].Index.TryGetValue(name, out int id))
        {
            throw new InvalidOperationException($"calc: vocab {cat} has no entry {name}");
        }
        return id;
    }

    private void CalcResolve()
    {
        _cPlating = CalcId("power", "PLATING_POWER");
        _cConstrict = CalcId("power", "CONSTRICT_POWER");
        _cBuffer = CalcId("power", "BUFFER_POWER");
        _cIntangible = CalcId("power", "INTANGIBLE_POWER");
        _cHardenedShell = CalcId("power", "HARDENED_SHELL_POWER");
        _cSlippery = CalcId("power", "SLIPPERY_POWER");
        _cSkittish = CalcId("power", "SKITTISH_POWER");
        _cOrichalcum = CalcId("relic", "ORICHALCUM");
        _cTungstenRod = CalcId("relic", "TUNGSTEN_ROD");
        _cBeatingRemnant = CalcId("relic", "BEATING_REMNANT");
        _cDamage = CalcId("var_name", "Damage");
        _cCalcDamage = CalcId("var_name", "CalculatedDamage");
        _cHitCount = CalcId("var_name", "HitCount");
        _cCalcHits = CalcId("var_name", "CalculatedHits");
        _cBlock = CalcId("var_name", "Block");
        _cCalcBlock = CalcId("var_name", "CalculatedBlock");
        _cHpLoss = CalcId("var_name", "HpLoss");
        _cHidden = CalcId(Ev.IntentType, "Hidden");
        _cUnknownIntent = CalcId(Ev.IntentType, "Unknown");
        _cPlayerSide = CalcId(Ev.CombatSide, "Player");
        _cRest = CalcId(Ev.MapPointType, "RestSite");
        _cShop = CalcId(Ev.MapPointType, "Shop");
        _cElite = CalcId(Ev.MapPointType, "Elite");
        _cMonster = CalcId(Ev.MapPointType, "Monster");
        _cUnknownNode = CalcId(Ev.MapPointType, "Unknown");
        _cOrichalcumBlock = (int)ModelDb.Relic<Orichalcum>().DynamicVars.Block.BaseValue;
        _calcReady = true;
    }

    /// <summary>Stages the three calc tables of this frame (called by Finish when Options.Calc, before Build).</summary>
    private void EncodeCalc()
    {
        if (!_calcReady)
        {
            CalcResolve();
        }
        CalcCombat();
        if (HostProfile.On)
        {
            HostProfile.EncLap(ProfileCalc);
        }
        if (_w.Count(T.MapNodes) > 0)
        {
            CalcRoute();
        }
        if (HostProfile.On)
        {
            HostProfile.EncLap(ProfileCalcRoute);
        }
    }

    private static int CalcCheck(int value, int lo, string what)
    {
        if (value < lo)
        {
            throw new InvalidOperationException($"calc: {what} = {value} < {lo}");
        }
        return value;
    }

    // =====================================================================================================
    //  TURN + FOE (contract 1.1-1.6)
    // =====================================================================================================

    /// <summary>One candidate hand card (contract 1.4).</summary>
    private struct CalcCard
    {
        public int Row, Pos, Cost, Star;
        public bool IsX;
        public TargetType Tt;
        public int Block;      // block(c)
        public int Hits;       // hits(c) (1 when no hit-count variable)
        public bool HasHits;   // a CalculatedHits / HitCount row exists
        public int DmgVar;     // var id used for per(c, .): CalculatedDamage, Damage, or -1
        public int DmgUntargeted;
    }

    private CalcCard[] _calcCards = new CalcCard[16];
    private readonly int[] _calcFoes = new int[SchemaX1.FoeMaxRows];
    private int[] _calcFoeOf = new int[64];
    private readonly int[] _calcVarIdx = new int[256];

    // knapsack scratch (at most CalcMaxItems paying items)
    private readonly int[] _kCost = new int[SchemaX1.CalcMaxItems], _kStar = new int[SchemaX1.CalcMaxItems], _kVal = new int[SchemaX1.CalcMaxItems];
    private readonly bool[] _kX = new bool[SchemaX1.CalcMaxItems];
    private readonly int[] _mCost = new int[1 << SchemaX1.CalcMaxItems], _mStar = new int[1 << SchemaX1.CalcMaxItems],
        _mVal = new int[1 << SchemaX1.CalcMaxItems], _mX = new int[1 << SchemaX1.CalcMaxItems];
    private readonly List<int> _paying = new List<int>(16);
    private readonly List<long> _payKeys = new List<long>(16);

    private void CalcCombat()
    {
        int[] g = _w.Global;
        if (g[Gl.InCombat] != 1)
        {
            return;
        }
        int s = g[Gl.ActingSeat];
        if (s < 0 || s >= _w.Count(T.Seats))
        {
            return;
        }
        int p = _w.Get(T.Seats, s, SeatC.CreatureRow);
        if (p < 0 || _w.Get(T.Creatures, p, CreatureC.Alive) != 1 || _w.Get(T.Seats, s, SeatC.PrivateVisible) != 1)
        {
            return;
        }
        int nCr = _w.Count(T.Creatures);
        if (_calcFoeOf.Length < nCr)
        {
            _calcFoeOf = new int[Math.Max(nCr, _calcFoeOf.Length * 2)];
        }
        // ---- FOE rows: living enemies in CREATURES row order ----
        int nFoe = 0, nHittable = 0;
        bool pet = false;
        for (int r = 0; r < nCr; r++)
        {
            _calcFoeOf[r] = -1;
            int kind = _w.Get(T.Creatures, r, CreatureC.Kind);
            bool alive = _w.Get(T.Creatures, r, CreatureC.Alive) == 1;
            if (kind == CreatureKind.Enemy && alive)
            {
                if (nFoe == SchemaX1.FoeMaxRows)
                {
                    throw new InvalidOperationException("calc: more than " + SchemaX1.FoeMaxRows + " living enemies");
                }
                _calcFoeOf[r] = nFoe;
                _calcFoes[nFoe++] = r;
                if (_w.Get(T.Creatures, r, CreatureC.Hittable) == 1)
                {
                    nHittable++;
                }
            }
            else if (kind == CreatureKind.Pet && alive && _w.Get(T.Creatures, r, CreatureC.Seat) == s)
            {
                pet = true;
            }
        }
        // ---- 1.2 incoming ----
        Span<int> incomingFrom = stackalloc int[SchemaX1.FoeMaxRows];
        incomingFrom.Clear();
        bool unknownIncoming = false;
        int nIt = _w.Count(T.Intents);
        for (int i = 0; i < nIt; i++)
        {
            int f = _calcFoeOf[_w.Get(T.Intents, i, IntentC.OwnerRow)];
            if (f < 0)
            {
                continue;
            }
            int type = _w.Get(T.Intents, i, IntentC.IntentType);
            if (type == _cHidden || type == _cUnknownIntent)
            {
                unknownIncoming = true;
            }
            int dmg = _w.Get(T.Intents, i, IntentC.DmgSeat0 + s);
            if (dmg >= 0)
            {
                incomingFrom[f] += dmg * _w.Get(T.Intents, i, IntentC.Hits);
            }
        }
        int incoming = 0;
        for (int f = 0; f < nFoe; f++)
        {
            incoming += incomingFrom[f];
        }
        // ---- powers of P and of the enemies ----
        Span<bool> guarded = stackalloc bool[SchemaX1.FoeMaxRows];
        guarded.Clear();
        int plating = 0, constrict = 0, nPlating = 0, nConstrict = 0;
        bool reduced = pet;
        int nPw = _w.Count(T.Powers);
        for (int i = 0; i < nPw; i++)
        {
            int owner = _w.Get(T.Powers, i, PowerC.OwnerRow);
            int id = _w.Get(T.Powers, i, PowerC.PowerId);
            if (owner == p)
            {
                if (id == _cPlating)
                {
                    plating = _w.Get(T.Powers, i, PowerC.Amount);
                    nPlating++;
                }
                else if (id == _cConstrict)
                {
                    constrict = _w.Get(T.Powers, i, PowerC.Amount);
                    nConstrict++;
                }
                else if (id == _cBuffer || id == _cIntangible)
                {
                    reduced = true;
                }
            }
            else if (_calcFoeOf[owner] >= 0 && (id == _cHardenedShell || id == _cSlippery || id == _cBuffer || id == _cSkittish))
            {
                guarded[_calcFoeOf[owner]] = true;
            }
        }
        if (nPlating > 1 || nConstrict > 1)
        {
            throw new InvalidOperationException($"calc: player has {nPlating} PLATING / {nConstrict} CONSTRICT power rows");
        }
        bool orichalcum = false;
        int nRel = _w.Count(T.Relics);
        for (int i = 0; i < nRel; i++)
        {
            if (_w.Get(T.Relics, i, RelicC.Seat) != s || _w.Get(T.Relics, i, RelicC.Zone) != ItemZone.Owned)
            {
                continue;
            }
            int id = _w.Get(T.Relics, i, RelicC.RelicId);
            if (id == _cOrichalcum)
            {
                orichalcum = true;
            }
            else if (id == _cTungstenRod || id == _cBeatingRemnant)
            {
                reduced = true;
            }
        }
        int hp = CalcCheck(_w.Get(T.Creatures, p, CreatureC.Hp), 1, "TURN.HP");
        int block = CalcCheck(_w.Get(T.Creatures, p, CreatureC.Block), 0, "TURN.BLOCK");
        bool playerSide = g[Gl.CombatSide] == _cPlayerSide;

        // ---- hand: turn-end cards (1.3) and knapsack candidates (1.4) ----
        Player player = _run.Players[s];
        IReadOnlyList<CardModel> hand = player.PlayerCombatState!.Hand.Cards;
        int energy = CalcCheck(_w.Get(T.Seats, s, SeatC.Energy), 0, "SEATS.ENERGY");
        int stars = Math.Max(_w.Get(T.Seats, s, SeatC.Stars), 0);
        int endDmg = 0, endLoss = 0;
        int nCand = 0;
        int nCards = _w.Count(T.Cards);
        for (int r = 0; r < nCards; r++)
        {
            if (_w.Get(T.Cards, r, CardC.Zone) != Zone.Hand || _w.Get(T.Cards, r, CardC.Seat) != s)
            {
                continue;
            }
            int pos = _w.Get(T.Cards, r, CardC.Pos);
            CardModel card = hand[pos];
            if (!_cardRow.TryGetValue(card, out int mapped) || mapped != r)
            {
                throw new InvalidOperationException($"calc: HAND row {r} (pos {pos}) is not the row of hand card {card.Id.Entry}");
            }
            int nv = _w.CollectVarsOf(T.Cards, r, _calcVarIdx);
            if (playerSide && card.HasTurnEndInHandEffect)
            {
                if (CalcVar(nv, _cDamage, -1, out int d))
                {
                    endDmg += d;
                }
                if (CalcVar(nv, _cHpLoss, -1, out int l))
                {
                    endLoss += l;
                }
            }
            int cf = _w.Get(T.Cards, r, CardC.CostFlags);
            if ((_w.Get(T.Cards, r, CardC.HandFlags) & HandFlag.CanPlay) == 0 || (cf & CostFlag.StarX) != 0)
            {
                continue;
            }
            if (nCand == _calcCards.Length)
            {
                Array.Resize(ref _calcCards, nCand * 2);
            }
            ref CalcCard c = ref _calcCards[nCand++];
            c.Row = r;
            c.Pos = pos;
            c.IsX = (cf & CostFlag.CostsX) != 0;
            c.Cost = CalcCheck(_w.Get(T.Cards, r, CardC.EnergyCost), 0, "CARDS.ENERGY_COST of a playable card");
            c.Star = Math.Max(_w.Get(T.Cards, r, CardC.StarCost), 0);
            c.Tt = card.TargetType;
            c.Block = 0;
            if (!c.IsX && c.Tt is not (TargetType.AnyAlly or TargetType.AllAllies or TargetType.AnyPlayer))
            {
                if (CalcVar(nv, _cCalcBlock, -1, out int b) || CalcVar(nv, _cBlock, -1, out b))
                {
                    c.Block = Math.Max(b, 0);
                }
            }
            c.HasHits = CalcVar(nv, _cCalcHits, -1, out int h) || CalcVar(nv, _cHitCount, -1, out h);
            c.Hits = c.HasHits ? Math.Max(h, 0) : 1;
            if (CalcVar(nv, _cCalcDamage, -1, out int du))
            {
                c.DmgVar = _cCalcDamage;
            }
            else if (CalcVar(nv, _cDamage, -1, out du))
            {
                c.DmgVar = _cDamage;
            }
            else
            {
                c.DmgVar = -1;
                du = 0;
            }
            c.DmgUntargeted = du;
        }
        int endBlock = 0;
        if (playerSide)
        {
            endBlock = plating + (orichalcum && block == 0 ? _cOrichalcumBlock : 0);
            endDmg += constrict;
        }
        else
        {
            endDmg = 0;
            endLoss = 0;
        }
        CalcCheck(endBlock, 0, "TURN.END_BLOCK");
        CalcCheck(endDmg, 0, "TURN.END_DMG");
        CalcCheck(endLoss, 0, "TURN.END_LOSS");

        // ---- BLOCK_REACHABLE ----
        _paying.Clear();
        _payKeys.Clear();
        int freeSum = 0;
        for (int i = 0; i < nCand; i++)
        {
            ref CalcCard c = ref _calcCards[i];
            if (c.Block <= 0)
            {
                continue; // X cards have block 0
            }
            if (c.Cost == 0 && c.Star == 0)
            {
                freeSum += c.Block;
            }
            else
            {
                _paying.Add(i);
                _payKeys.Add(CalcKey(c.Block, c.Pos));
            }
        }
        bool blockCapped = CalcCap();
        int nk = _paying.Count;
        for (int k = 0; k < nk; k++)
        {
            ref CalcCard c = ref _calcCards[_paying[k]];
            _kCost[k] = c.Cost;
            _kStar[k] = c.Star;
            _kVal[k] = c.Block;
            _kX[k] = false;
        }
        int blockReachable = CalcCheck(freeSum + CalcBest(nk, energy, stars, null, 0), 0, "TURN.BLOCK_REACHABLE");

        // ---- 1.5 TURN row ----
        int t = endDmg + incoming;
        int aNow = block + endBlock;
        int aBest = Math.Max(aNow, block + plating + blockReachable);
        int lossNow = Math.Max(0, t - aNow) + endLoss;
        int lossMin = Math.Max(0, t - aBest) + endLoss;
        Span<int> tr = _w.AddCalcRow(FrameWriter.CalcTurn);
        tr[TurnC.OwnerRow] = p;
        tr[TurnC.Hp] = hp;
        tr[TurnC.Block] = block;
        tr[TurnC.Incoming] = CalcCheck(incoming, 0, "TURN.INCOMING");
        tr[TurnC.EndBlock] = endBlock;
        tr[TurnC.EndDmg] = endDmg;
        tr[TurnC.EndLoss] = endLoss;
        tr[TurnC.HpLossNow] = lossNow;
        tr[TurnC.BlockReachable] = blockReachable;
        tr[TurnC.HpLossMin] = lossMin;
        tr[TurnC.Flags] = (lossNow >= hp ? TurnFlag.LethalNow : 0) | (lossMin >= hp ? TurnFlag.LethalUnavoidable : 0)
            | (unknownIncoming ? TurnFlag.UnknownIncoming : 0) | (reduced ? TurnFlag.Reduced : 0) | (blockCapped ? TurnFlag.Capped : 0);
        Stats.CalcTurnRows++;
        Stats.CalcMaxima++;
        if (blockCapped)
        {
            Stats.CalcCapped++;
        }

        // ---- 1.6 FOE rows ----
        for (int f = 0; f < nFoe; f++)
        {
            int e = _calcFoes[f];
            bool hittable = _w.Get(T.Creatures, e, CreatureC.Hittable) == 1;
            int ehp = _w.Get(T.Creatures, e, CreatureC.Hp);
            int toKill = ehp >= 0 ? ehp + _w.Get(T.Creatures, e, CreatureC.Block) : -1;
            if (toKill != -1)
            {
                CalcCheck(toKill, 1, "FOE.TO_KILL");
            }
            int dmg = 0;
            bool capped = false;
            if (hittable)
            {
                Stats.CalcMaxima++;
                dmg = CalcDamage(e, nCand, nHittable, energy, stars, out capped);
                if (capped)
                {
                    Stats.CalcCapped++;
                }
            }
            Span<int> fr = _w.AddCalcRow(FrameWriter.CalcFoe);
            fr[FoeC.OwnerRow] = e;
            fr[FoeC.ToKill] = toKill;
            fr[FoeC.DmgReachable] = CalcCheck(dmg, 0, "FOE.DMG_REACHABLE");
            fr[FoeC.IncomingFrom] = CalcCheck(incomingFrom[f], 0, "FOE.INCOMING_FROM");
            fr[FoeC.Flags] = (toKill > 0 && dmg >= toKill && !guarded[f] ? FoeFlag.Kill : 0) | (guarded[f] ? FoeFlag.Guarded : 0)
                | (ehp < 0 ? FoeFlag.HpHidden : 0) | (capped ? FoeFlag.Capped : 0) | (hittable ? 0 : FoeFlag.Unhittable);
            Stats.CalcFoeRows++;
        }
    }

    /// <summary>SHOWN of the card's staged VARS row (var id, target row), not an enchantment row; negative values clamp to 0.</summary>
    private bool CalcVar(int nv, int varId, int targetRow, out int shown)
    {
        for (int k = 0; k < nv; k++)
        {
            ReadOnlySpan<int> v = _w.StagedVar(_calcVarIdx[k]);
            if (v[VarC.VarId] == varId && v[VarC.TargetRow] == targetRow && (v[VarC.Flags] & VarFlag.Enchantment) == 0)
            {
                shown = Math.Max(v[VarC.Shown], 0);
                return true;
            }
        }
        shown = 0;
        return false;
    }

    /// <summary>Cap key: larger value first, then lower POS (sorted descending).</summary>
    private static long CalcKey(long value, int pos) => (value << 20) | (uint)(0xFFFFF - pos);

    /// <summary>Keeps the CalcMaxItems paying items with the largest key (contract 1.4); true = CAPPED.</summary>
    private bool CalcCap()
    {
        int n = _paying.Count;
        if (n <= SchemaX1.CalcMaxItems)
        {
            return false;
        }
        // selection by key, descending; keys are unique (POS is unique in a hand)
        for (int i = 0; i < SchemaX1.CalcMaxItems; i++)
        {
            int best = i;
            for (int j = i + 1; j < n; j++)
            {
                if (_payKeys[j] > _payKeys[best])
                {
                    best = j;
                }
            }
            (_payKeys[i], _payKeys[best]) = (_payKeys[best], _payKeys[i]);
            (_paying[i], _paying[best]) = (_paying[best], _paying[i]);
        }
        _paying.RemoveRange(SchemaX1.CalcMaxItems, n - SchemaX1.CalcMaxItems);
        _payKeys.RemoveRange(SchemaX1.CalcMaxItems, n - SchemaX1.CalcMaxItems);
        return true;
    }

    /// <summary>DMG_REACHABLE(e) (contract 1.4) for a hittable enemy row <paramref name="e"/>.</summary>
    private int CalcDamage(int e, int nCand, int nHittable, int energy, int stars, out bool capped)
    {
        _paying.Clear();
        _payKeys.Clear();
        int freeSum = 0;
        Span<int> per = stackalloc int[nCand];
        Span<int> val = stackalloc int[nCand];
        Span<bool> aimsAt = stackalloc bool[nCand];
        for (int i = 0; i < nCand; i++)
        {
            ref CalcCard c = ref _calcCards[i];
            int pe = 0;
            if (c.DmgVar >= 0)
            {
                int nv = _w.CollectVarsOf(T.Cards, c.Row, _calcVarIdx);
                pe = CalcVar(nv, c.DmgVar, e, out int targeted) ? targeted : c.DmgUntargeted;
            }
            // e is hittable here (the caller skips the rest)
            bool aims = c.Tt is TargetType.AnyEnemy or TargetType.AllEnemies || (c.Tt == TargetType.RandomEnemy && nHittable == 1);
            per[i] = pe;
            aimsAt[i] = aims;
            int v = aims ? pe * c.Hits : 0;
            val[i] = v;
            if (!(v > 0 || (c.IsX && pe > 0)))
            {
                continue;
            }
            if (!c.IsX && c.Cost == 0 && c.Star == 0)
            {
                freeSum += v;
                continue;
            }
            _paying.Add(i);
            _payKeys.Add(CalcKey(c.IsX ? (long)pe * energy : v, c.Pos));
        }
        capped = CalcCap();
        int nk = _paying.Count;
        Span<int> xPer = stackalloc int[nk];
        for (int k = 0; k < nk; k++)
        {
            int i = _paying[k];
            ref CalcCard c = ref _calcCards[i];
            _kCost[k] = c.Cost;
            _kStar[k] = c.Star;
            _kX[k] = c.IsX;
            // X card: with a printed hit count its value is per * hits; without one, per * (energy left) - computed in CalcBest
            _kVal[k] = c.IsX && !c.HasHits ? 0 : val[i];
            xPer[k] = c.IsX && !c.HasHits && aimsAt[i] ? per[i] : 0;
        }
        return freeSum + CalcBest(nk, energy, stars, xPer, nk);
    }

    /// <summary>
    /// Exact 0/1 maximum over subsets of the nk paying items (_kCost/_kStar/_kVal/_kX) within energy and stars, at most one
    /// X item per subset; an X item without a fixed value adds xPer * (energy - sum cost of the subset).
    /// </summary>
    private int CalcBest(int nk, int energy, int stars, Span<int> xPer, int nx)
    {
        int best = 0;
        int full = 1 << nk;
        _mCost[0] = 0;
        _mStar[0] = 0;
        _mVal[0] = 0;
        _mX[0] = 0;
        for (int m = 1; m < full; m++)
        {
            int low = BitOperations.TrailingZeroCount(m);
            int prev = m & (m - 1);
            _mCost[m] = _mCost[prev] + _kCost[low];
            _mStar[m] = _mStar[prev] + _kStar[low];
            _mVal[m] = _mVal[prev] + _kVal[low];
            _mX[m] = _mX[prev] + (_kX[low] ? 1 : 0);
            if (_mCost[m] > energy || _mStar[m] > stars || _mX[m] > 1)
            {
                continue;
            }
            int v = _mVal[m];
            if (_mX[m] == 1 && nx > 0)
            {
                for (int k = 0; k < nk; k++)
                {
                    if (_kX[k] && (m & (1 << k)) != 0)
                    {
                        v += xPer[k] * (energy - _mCost[m]);
                        break;
                    }
                }
            }
            if (v > best)
            {
                best = v;
            }
        }
        return best;
    }

    // =====================================================================================================
    //  ROUTE (contract 1.7)
    // =====================================================================================================

    private const int RouteInf = int.MaxValue / 4;
    private readonly int[] _rType = new int[SchemaX1.RouteMaxRows];
    private readonly int[] _rChildStart = new int[SchemaX1.RouteMaxRows + 1];
    private readonly int[] _rChild = new int[512];
    private readonly int[] _rOutDeg = new int[SchemaX1.RouteMaxRows];
    private readonly int[] _rColor = new int[SchemaX1.RouteMaxRows];
    private readonly int[] _rOrder = new int[SchemaX1.RouteMaxRows];
    private readonly int[] _rStack = new int[SchemaX1.RouteMaxRows];
    private readonly int[] _rIter = new int[SchemaX1.RouteMaxRows];
    private readonly int[] _rBoss = new int[SchemaX1.RouteMaxRows], _rRest = new int[SchemaX1.RouteMaxRows], _rShop = new int[SchemaX1.RouteMaxRows],
        _rERest = new int[SchemaX1.RouteMaxRows], _rFRest = new int[SchemaX1.RouteMaxRows], _rEBoss = new int[SchemaX1.RouteMaxRows],
        _rFBoss = new int[SchemaX1.RouteMaxRows];
    private readonly ulong[] _rReachLo = new ulong[SchemaX1.RouteMaxRows], _rReachHi = new ulong[SchemaX1.RouteMaxRows];

    private void CalcRoute()
    {
        int n = _w.Count(T.MapNodes);
        if (n > SchemaX1.RouteMaxRows)
        {
            throw new InvalidOperationException($"calc: {n} MAP_NODES rows > {SchemaX1.RouteMaxRows}");
        }
        int boss = -1, second = -1;
        for (int i = 0; i < n; i++)
        {
            int fl = _w.Get(T.MapNodes, i, MapNodeC.Flags);
            if ((fl & MapNodeFlag.Boss) != 0)
            {
                if (boss >= 0)
                {
                    throw new InvalidOperationException("calc: two BOSS map nodes");
                }
                boss = i;
            }
            if ((fl & MapNodeFlag.SecondBoss) != 0)
            {
                if (second >= 0)
                {
                    throw new InvalidOperationException("calc: two SECOND_BOSS map nodes");
                }
                second = i;
            }
            _rType[i] = _w.Get(T.MapNodes, i, MapNodeC.PointType);
            _rOutDeg[i] = 0;
        }
        if (boss < 0)
        {
            throw new InvalidOperationException("calc: no BOSS map node");
        }
        // ---- adjacency (CSR): MAP_EDGES + (m -> BOSS) for sinks + (BOSS -> SECOND_BOSS) ----
        int ne = _w.Count(T.MapEdges);
        for (int k = 0; k < ne; k++)
        {
            int from = _w.Get(T.MapEdges, k, MapEdgeC.From), to = _w.Get(T.MapEdges, k, MapEdgeC.To);
            if ((uint)from >= (uint)n || (uint)to >= (uint)n)
            {
                throw new InvalidOperationException($"calc: MAP_EDGES row {k} ({from} -> {to}) outside {n} nodes");
            }
            _rOutDeg[from]++;
        }
        bool bossToSecond = false;
        if (second >= 0)
        {
            for (int k = 0; k < ne; k++)
            {
                if (_w.Get(T.MapEdges, k, MapEdgeC.From) == boss && _w.Get(T.MapEdges, k, MapEdgeC.To) == second)
                {
                    bossToSecond = true;
                }
            }
        }
        int total = 0;
        for (int i = 0; i < n; i++)
        {
            _rChildStart[i] = total;
            int deg = _rOutDeg[i];
            if (deg == 0 && i != boss && i != second)
            {
                deg = 1; // the added edge to the boss
            }
            if (i == boss && second >= 0 && !bossToSecond)
            {
                deg++;
            }
            total += deg;
        }
        _rChildStart[n] = total;
        if (total > _rChild.Length)
        {
            throw new InvalidOperationException($"calc: {total} map edges");
        }
        Span<int> fill = stackalloc int[n];
        for (int i = 0; i < n; i++)
        {
            fill[i] = _rChildStart[i];
        }
        for (int k = 0; k < ne; k++)
        {
            int from = _w.Get(T.MapEdges, k, MapEdgeC.From);
            _rChild[fill[from]++] = _w.Get(T.MapEdges, k, MapEdgeC.To);
        }
        for (int i = 0; i < n; i++)
        {
            if (_rOutDeg[i] == 0 && i != boss && i != second)
            {
                _rChild[fill[i]++] = boss;
            }
            if (i == boss && second >= 0 && !bossToSecond)
            {
                _rChild[fill[i]++] = second;
            }
        }

        // ---- post-order DFS (acyclicity asserted), then the minima child-first ----
        Array.Clear(_rColor, 0, n);
        int nOrd = 0;
        for (int root = 0; root < n; root++)
        {
            if (_rColor[root] != 0)
            {
                continue;
            }
            int sp = 0;
            _rStack[sp] = root;
            _rIter[sp] = _rChildStart[root];
            _rColor[root] = 1;
            sp++;
            while (sp > 0)
            {
                int v = _rStack[sp - 1];
                if (_rIter[sp - 1] < _rChildStart[v + 1])
                {
                    int c = _rChild[_rIter[sp - 1]++];
                    if (_rColor[c] == 1)
                    {
                        throw new InvalidOperationException($"calc: the map graph has a cycle through node {c}");
                    }
                    if (_rColor[c] == 0)
                    {
                        _rColor[c] = 1;
                        _rStack[sp] = c;
                        _rIter[sp] = _rChildStart[c];
                        sp++;
                    }
                    continue;
                }
                _rColor[v] = 2;
                _rOrder[nOrd++] = v;
                sp--;
            }
        }
        for (int oi = 0; oi < nOrd; oi++)
        {
            int v = _rOrder[oi];
            int type = _rType[v];
            bool rest = type == _cRest, shop = type == _cShop, elite = type == _cElite, fight = elite || type == _cMonster;
            int mBoss = RouteInf, mRest = RouteInf, mShop = RouteInf, mERest = RouteInf, mFRest = RouteInf, mEBoss = RouteInf, mFBoss = RouteInf;
            ulong lo = 0, hi = 0;
            for (int k = _rChildStart[v]; k < _rChildStart[v + 1]; k++)
            {
                int c = _rChild[k];
                mBoss = Math.Min(mBoss, _rBoss[c]);
                mRest = Math.Min(mRest, _rRest[c]);
                mShop = Math.Min(mShop, _rShop[c]);
                mERest = Math.Min(mERest, _rERest[c]);
                mFRest = Math.Min(mFRest, _rFRest[c]);
                mEBoss = Math.Min(mEBoss, _rEBoss[c]);
                mFBoss = Math.Min(mFBoss, _rFBoss[c]);
                lo |= _rReachLo[c];
                hi |= _rReachHi[c];
            }
            int ev = elite ? 1 : 0, fv = fight ? 1 : 0;
            // the SECOND_BOSS node is its own "boss" target (notes/calc/deviations.md D1)
            bool target = v == boss || v == second;
            _rBoss[v] = target ? 0 : RoutePlus(mBoss, 1);
            _rEBoss[v] = target ? 0 : RoutePlus(mEBoss, ev);
            _rFBoss[v] = target ? 0 : RoutePlus(mFBoss, fv);
            _rRest[v] = rest ? 0 : RoutePlus(mRest, 1);
            _rShop[v] = shop ? 0 : RoutePlus(mShop, 1);
            _rERest[v] = rest ? 0 : RoutePlus(mERest, ev);
            _rFRest[v] = rest ? 0 : RoutePlus(mFRest, fv);
            if (v < 64)
            {
                lo |= 1UL << v;
            }
            else
            {
                hi |= 1UL << (v - 64);
            }
            _rReachLo[v] = lo;
            _rReachHi[v] = hi;
        }
        // type masks over the non-boss nodes
        ulong restLo = 0, restHi = 0, shopLo = 0, shopHi = 0, eliteLo = 0, eliteHi = 0, unkLo = 0, unkHi = 0;
        for (int i = 0; i < n; i++)
        {
            if (i == boss || i == second)
            {
                continue;
            }
            int type = _rType[i];
            ulong blo = i < 64 ? 1UL << i : 0, bhi = i < 64 ? 0 : 1UL << (i - 64);
            if (type == _cRest)
            {
                restLo |= blo;
                restHi |= bhi;
            }
            else if (type == _cShop)
            {
                shopLo |= blo;
                shopHi |= bhi;
            }
            else if (type == _cElite)
            {
                eliteLo |= blo;
                eliteHi |= bhi;
            }
            else if (type == _cUnknownNode)
            {
                unkLo |= blo;
                unkHi |= bhi;
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (_w.Get(T.MapNodes, i, MapNodeC.State) != MapNodeState.Reachable)
            {
                continue;
            }
            if (_rBoss[i] >= RouteInf)
            {
                throw new InvalidOperationException($"calc: the BOSS node is not reachable from REACHABLE map node {i}");
            }
            Span<int> row = _w.AddCalcRow(FrameWriter.CalcRoute);
            row[RouteC.OwnerRow] = i;
            row[RouteC.RowsToBoss] = _rBoss[i];
            row[RouteC.RestDist] = RouteOut(_rRest[i]);
            row[RouteC.ShopDist] = RouteOut(_rShop[i]);
            row[RouteC.ElitesToRest] = RouteOut(_rERest[i]);
            row[RouteC.FightsToRest] = RouteOut(_rFRest[i]);
            row[RouteC.ElitesToBoss] = _rEBoss[i];
            row[RouteC.FightsToBoss] = _rFBoss[i];
            ulong lo = _rReachLo[i], hi = _rReachHi[i];
            row[RouteC.NRest] = BitOperations.PopCount(lo & restLo) + BitOperations.PopCount(hi & restHi);
            row[RouteC.NShop] = BitOperations.PopCount(lo & shopLo) + BitOperations.PopCount(hi & shopHi);
            row[RouteC.NElite] = BitOperations.PopCount(lo & eliteLo) + BitOperations.PopCount(hi & eliteHi);
            row[RouteC.NUnknown] = BitOperations.PopCount(lo & unkLo) + BitOperations.PopCount(hi & unkHi);
            Stats.CalcRouteRows++;
        }
    }

    private static int RoutePlus(int m, int add) => m >= RouteInf ? RouteInf : m + add;

    private static int RouteOut(int m) => m >= RouteInf ? -1 : m;
}
