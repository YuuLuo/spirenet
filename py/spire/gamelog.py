"""Per-decision game logs of 4-seat evaluation games (audit task "logs"; format: notes/audit/logs.md).

:class:`GameLogger` is an ``evaluate(on_decision=...)`` observer like :class:`~spire.h5eval.CoopTelemetry`: for every
policy decision it decodes the frame the policy decided on and the chosen ACTIONS row into one JSON record (names from
the vocabulary, never raw ids) and streams it to a gzip file per game. :meth:`GameLogger.finish` joins the games of
each host in order to the evaluation's episode records (result, final floor, seed), writes each game's header as the
first line of its file and returns the index rows.

File = gzip member 1 (the ``game`` header line) + gzip member 2 (the body: ``combat`` / ``inv`` / ``d`` records in
decision order); ``gzip.open`` reads the concatenation as one text stream."""
from __future__ import annotations

import gzip
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Mapping, Sequence

from .metrics import lineup_key
from .proto import Frame
from .schema import (CREATURE_MODEL_VOCAB, OPTION_KEY_VOCAB, REF_CATEGORY_VOCAB, ActionCol, CardCol, CardSelectSub,
                     CellCol, CostFlag, CreatureCol, CreatureKind, G, HandFlag, IntentCol, ItemZone, Kind, MapNodeCol,
                     OptionCol, OptionFlag, OptionKind, OrbCol, PotionCol, PowerCol, RefCol, RelicCol, SeatCol, Table,
                     TreasurePhase, VarCol, Verb, Zone)
from .vocab import Vocab

LOG_FORMAT = 1
CHAR_CODE = {"DEFECT": "D", "FOOL": "F", "IRONCLAD": "I", "NECROBINDER": "N", "REGENT": "R", "SILENT": "S"}
OPTION_FLAGS = (("locked", OptionFlag.LOCKED), ("lethal", OptionFlag.LETHAL), ("on_sale", OptionFlag.ON_SALE),
                ("stocked", OptionFlag.STOCKED), ("affordable", OptionFlag.AFFORDABLE),
                ("proceed", OptionFlag.IS_PROCEED), ("was_chosen", OptionFlag.WAS_CHOSEN))
VOTE_BIT0 = 7  # OPTIONS.flags bit 7 + s = seat s voted for this option (erratum H5-5); MAP_NODES.flags bit 4 + s
MAP_VOTE_BIT0 = 4
DECISION_KEYS = ("t", "i", "step", "kind", "seat", "floor", "act", "room", "n", "a")


def lineup_code(characters: Sequence[str]) -> str:
    """``NNNN``, ``DINN``, ...: one letter per seat, in canonical (sorted name) order."""
    return "".join(CHAR_CODE[c] for c in sorted(characters))


def _dumps(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, separators=(",", ":"), ensure_ascii=False)


class _View:
    """One frame's tables as Python lists (decoded once per decision)."""

    __slots__ = ("f", "_t", "_vars", "_powers", "_intents")

    def __init__(self, f: Frame) -> None:
        self.f = f
        self._t: dict[int, list[list[int]]] = {}
        self._vars: dict[tuple[int, int], list[list[int]]] | None = None
        self._powers: dict[int, list[list[int]]] | None = None
        self._intents: dict[int, list[list[int]]] | None = None

    def t(self, table: int) -> list[list[int]]:
        rows = self._t.get(table)
        if rows is None:
            rows = self._t[table] = self.f.table(table).tolist()
        return rows

    def vars_of(self, table: int, row: int) -> list[list[int]]:
        if self._vars is None:
            by: dict[tuple[int, int], list[list[int]]] = {}
            for v in self.t(Table.VARS):
                by.setdefault((v[VarCol.OWNER_TABLE], v[VarCol.OWNER_ROW]), []).append(v)
            self._vars = by
        return self._vars.get((table, row), [])

    def powers_of(self, creature_row: int) -> list[list[int]]:
        if self._powers is None:
            by: dict[int, list[list[int]]] = {}
            for p in self.t(Table.POWERS):
                by.setdefault(p[PowerCol.OWNER_ROW], []).append(p)
            self._powers = by
        return self._powers.get(creature_row, [])

    def intents_of(self, creature_row: int) -> list[list[int]]:
        if self._intents is None:
            by: dict[int, list[list[int]]] = {}
            for i in self.t(Table.INTENTS):
                by.setdefault(i[IntentCol.OWNER_ROW], []).append(i)
            self._intents = by
        return self._intents.get(creature_row, [])


@dataclass
class _SeatInv:
    deck: list[dict[str, Any]] | None = None  # None until the first frame outside combat
    deck_at: int = -1  # decision index of the frame the deck was read from
    relics: list[str] = field(default_factory=list)
    potions: list[str] = field(default_factory=list)
    max_hp: int = -1
    hp: int = -1
    gold: int = -1


@dataclass
class _Game:
    env: int
    episode_id: int
    chars: list[str]
    ascension: int
    key: str
    tmp: Path
    body: IO[str]
    decisions: int = 0
    card_plays: int = 0
    potion_uses: int = 0
    combats: int = 0
    combat_key: tuple[int, int] | None = None  # (TOTAL_FLOOR, COMBATS_WON) of the open combat
    inv: list[_SeatInv] = field(default_factory=list)
    kinds: dict[str, int] = field(default_factory=dict)


class GameLogger:
    """Write one ``<out_dir>/<lineup code>-A<asc>/<seed index>-<seed>.jsonl.gz`` per game (see module doc).

    ``meta`` goes into every header (run, panel, checkpoint path / version / sha256, weights, mode)."""

    def __init__(self, vocab: Vocab, out_dir: Path, meta: Mapping[str, Any]) -> None:
        self.out_dir = out_dir
        self.meta = dict(meta)
        n = {cat: vocab[cat].entries for cat in vocab.categories}
        self.n = n
        self.characters = n["character"]
        self.mend = vocab.id("rest_option", "MEND")
        assert n["rest_option"][self.mend] == "MEND"
        # bodies of the open games; one directory per process (evaluations may share ``out_dir``)
        self.tmp = out_dir / f"tmp-{os.getpid()}"
        self.tmp.mkdir(parents=True)
        self._open: dict[int, _Game] = {}
        self._done: dict[int, list[_Game]] = {}
        self._serial = 0

    # ------------------------------------------------------------------ names
    def _name(self, cat: str, idx: int) -> str | None:
        """Vocabulary name; ``None`` for PAD (0 = no entity)."""
        if idx == 0:
            return None
        entries = self.n[cat]
        assert 0 < idx < len(entries), f"{cat}: id {idx} outside the vocabulary"
        return entries[idx]

    def _vars(self, v: _View, table: int, row: int, target: int = -1) -> dict[str, int]:
        return {self.n["var_name"][x[VarCol.VAR_ID]]: x[VarCol.SHOWN] for x in v.vars_of(table, row)
                if x[VarCol.TARGET_ROW] == target}

    def _card(self, v: _View, row: int) -> dict[str, Any]:
        c = v.t(Table.CARDS)[row]
        out: dict[str, Any] = {"c": self._name("card", c[CardCol.CARD_ID]), "u": c[CardCol.UPGRADE_LEVEL],
                               "e": "X" if c[CardCol.COST_FLAGS] & CostFlag.COSTS_X else c[CardCol.ENERGY_COST]}
        if c[CardCol.STAR_COST] >= 0 or c[CardCol.COST_FLAGS] & CostFlag.STAR_X:
            out["s"] = "X" if c[CardCol.COST_FLAGS] & CostFlag.STAR_X else c[CardCol.STAR_COST]
        if c[CardCol.ZONE] == Zone.HAND:
            out["p"] = int(bool(c[CardCol.HAND_FLAGS] & HandFlag.CAN_PLAY))
        if c[CardCol.COUNT] > 1:
            out["n"] = c[CardCol.COUNT]
        if c[CardCol.ENCHANT_ID]:
            out["ench"] = self._name("enchantment", c[CardCol.ENCHANT_ID])
        if c[CardCol.AFFLICTION_ID]:
            out["affl"] = self._name("affliction", c[CardCol.AFFLICTION_ID])
        var = self._vars(v, Table.CARDS, row)
        if var:
            out["v"] = var
        return out

    def _powers(self, v: _View, creature_row: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in v.powers_of(creature_row):
            name = self.n["power"][p[PowerCol.POWER_ID]]
            out[name] = out.get(name, 0) + p[PowerCol.AMOUNT]
        return out

    def _target(self, v: _View, row: int) -> dict[str, Any]:
        cre = v.t(Table.CREATURES)
        c = cre[row]
        kind = c[CreatureCol.KIND]
        if kind == CreatureKind.ENEMY:
            idx = sum(1 for r in cre[:row] if r[CreatureCol.KIND] == CreatureKind.ENEMY)
            return {"enemy": idx, "id": self._name("monster", c[CreatureCol.MODEL_ID])}
        if kind == CreatureKind.PLAYER:
            return {"seat": c[CreatureCol.SEAT]}
        assert kind == CreatureKind.PET, kind
        return {"pet": c[CreatureCol.SEAT], "id": self._name("monster", c[CreatureCol.MODEL_ID])}

    def _entity(self, v: _View, table: int, row: int) -> dict[str, Any]:
        if table == Table.CARDS:
            return {"card": self._card(v, row)}
        if table == Table.RELICS:
            return {"relic": self._name("relic", v.t(Table.RELICS)[row][RelicCol.RELIC_ID])}
        assert table == Table.POTIONS, table
        return {"potion": self._name("potion", v.t(Table.POTIONS)[row][PotionCol.POTION_ID])}

    def _option(self, v: _View, row: int) -> dict[str, Any]:
        o = v.t(Table.OPTIONS)[row]
        kind = OptionKind(o[OptionCol.KIND])
        out: dict[str, Any] = {"kind": kind.name}
        cat = OPTION_KEY_VOCAB.get(kind)
        if cat is not None:
            out["key"] = self._name(cat, o[OptionCol.KEY_ID])
        fl = o[OptionCol.FLAGS]
        flags = [name for name, bit in OPTION_FLAGS if fl & bit]
        if flags:
            out["flags"] = flags
        votes = [s for s in range(4) if fl & (1 << (VOTE_BIT0 + s))]
        if votes:
            out["votes"] = votes
        if o[OptionCol.PRICE] >= 0:
            out["price"] = o[OptionCol.PRICE]
        if o[OptionCol.AMOUNT] >= 0:
            out["amount"] = o[OptionCol.AMOUNT]
        if o[OptionCol.GROUP] >= 0:
            out["group"] = o[OptionCol.GROUP]
        if o[OptionCol.REF_TABLE] >= 0:
            out.update(self._entity(v, o[OptionCol.REF_TABLE], o[OptionCol.REF_ROW]))
        if o[OptionCol.REF_CNT] > 0:
            refs = v.t(Table.REFS)[o[OptionCol.REF_OFF]:o[OptionCol.REF_OFF] + o[OptionCol.REF_CNT]]
            out["refs"] = [{"cat": REF_CATEGORY_VOCAB[r[RefCol.CATEGORY]],
                            "id": self._name(REF_CATEGORY_VOCAB[r[RefCol.CATEGORY]], r[RefCol.VOCAB_ID])}
                           for r in refs]
        var = self._vars(v, Table.OPTIONS, row)
        if var:
            out["v"] = var
        return out

    def _node(self, v: _View, row: int) -> dict[str, Any]:
        m = v.t(Table.MAP_NODES)[row]
        out: dict[str, Any] = {"row": m[MapNodeCol.ROW], "col": m[MapNodeCol.COL],
                               "type": self._name("enum:MapPointType", m[MapNodeCol.POINT_TYPE])}
        votes = [s for s in range(4) if m[MapNodeCol.FLAGS] & (1 << (MAP_VOTE_BIT0 + s))]
        if votes:
            out["votes"] = votes
        return out

    # ------------------------------------------------------------------ records
    def _action(self, v: _View, a: list[int]) -> dict[str, Any]:
        verb, seat, st, sr, tgt, aux = a[:ActionCol.FLAGS]
        d: dict[str, Any] = {"verb": Verb(verb).name, "seat": seat}
        if st == Table.CARDS:
            d["card"] = self._card(v, sr)
            d["zone"] = Zone(v.t(Table.CARDS)[sr][CardCol.ZONE]).name
        elif st == Table.POTIONS:
            d["potion"] = self._name("potion", v.t(Table.POTIONS)[sr][PotionCol.POTION_ID])
        elif st == Table.MAP_NODES:
            d["node"] = self._node(v, sr)
        elif st == Table.OPTIONS:
            d["opt"] = sr  # index into the record's "opts"
            if v.t(Table.OPTIONS)[sr][OptionCol.KIND] == OptionKind.REST_OPTION and \
                    v.t(Table.OPTIONS)[sr][OptionCol.KEY_ID] == self.mend:
                assert aux != seat, "erratum H6: MEND never targets its owner"
                d["mend_target"] = aux
        elif st == Table.CELLS:
            cell = v.t(Table.CELLS)[sr]
            d["cell"] = [cell[CellCol.X], cell[CellCol.Y]]
            d["tool"] = self._name("enum:CrystalSphereToolType", aux)
        else:
            assert st == -1 and sr == -1, (st, sr)
        if tgt >= 0:
            d["tgt"] = self._target(v, tgt)
            if st == Table.CARDS:
                tv = self._vars(v, Table.CARDS, sr, tgt)
                if tv:
                    d["tvars"] = tv  # the card's numbers against this target where they differ (VARS target rows)
        return d

    def _combat(self, g: _Game, v: _View) -> dict[str, Any]:
        gv = v.f.gvals
        n_seats = gv[G.N_SEATS]
        cre = v.t(Table.CREATURES)
        seats_t = v.t(Table.SEATS)
        seats: list[dict[str, Any]] = []
        for s in seats_t:
            r = s[SeatCol.CREATURE_ROW]
            seats.append({"hp": s[SeatCol.HP], "mhp": s[SeatCol.MAX_HP], "blk": s[SeatCol.BLOCK],
                          "en": s[SeatCol.ENERGY], "men": s[SeatCol.MAX_ENERGY], "stars": s[SeatCol.STARS],
                          "alive": s[SeatCol.ALIVE], "ready": s[SeatCol.READY_TO_END_TURN],
                          "hand": s[SeatCol.HAND_COUNT], "draw": s[SeatCol.DRAW_COUNT],
                          "disc": s[SeatCol.DISCARD_COUNT], "exh": s[SeatCol.EXHAUST_COUNT],
                          "played": s[SeatCol.CARDS_PLAYED_TURN], "pw": self._powers(v, r) if r >= 0 else {},
                          "orbs": [], "pets": []})
        for o in v.t(Table.ORBS):
            seats[o[OrbCol.SEAT]]["orbs"].append(self._name("orb", o[OrbCol.ORB_ID]))
        enemies: list[dict[str, Any]] = []
        for r, c in enumerate(cre):
            kind = c[CreatureCol.KIND]
            if kind == CreatureKind.PET:
                seats[c[CreatureCol.SEAT]]["pets"].append(
                    {"id": self._name(CREATURE_MODEL_VOCAB[CreatureKind.PET], c[CreatureCol.MODEL_ID]),
                     "hp": c[CreatureCol.HP], "mhp": c[CreatureCol.MAX_HP], "blk": c[CreatureCol.BLOCK],
                     "alive": c[CreatureCol.ALIVE], "pw": self._powers(v, r)})
            elif kind == CreatureKind.ENEMY:
                intents = [{"type": self._name("enum:IntentType", i[IntentCol.INTENT_TYPE]), "hits": i[IntentCol.HITS],
                            **({"dmg": i[IntentCol.DMG_SEAT0:IntentCol.DMG_SEAT0 + n_seats]}
                               if i[IntentCol.DMG_SEAT0] >= 0 else {})}
                           for i in v.intents_of(r)]
                enemies.append({"id": self._name("monster", c[CreatureCol.MODEL_ID]),
                                "slot": self._name("slot_name", c[CreatureCol.SLOT_ID]),
                                "hp": c[CreatureCol.HP], "mhp": c[CreatureCol.MAX_HP], "blk": c[CreatureCol.BLOCK],
                                "alive": c[CreatureCol.ALIVE], "hittable": c[CreatureCol.HITTABLE],
                                "pw": self._powers(v, r), "intent": intents})
        hands: list[list[dict[str, Any]]] = [[] for _ in seats_t]
        for row, c in enumerate(v.t(Table.CARDS)):
            if c[CardCol.ZONE] == Zone.HAND:
                hands[c[CardCol.SEAT]].append(self._card(v, row))
        return {"cmb": g.combats, "round": gv[G.COMBAT_ROUND], "side": self._name("enum:CombatSide", gv[G.COMBAT_SIDE]),
                "seats": seats, "enemies": enemies, "hands": hands}

    def _inventory(self, g: _Game, v: _View, i: int, in_combat: bool) -> list[dict[str, Any]]:
        """``inv`` records of the seats whose deck (read outside combat only), relics, potions or max HP changed."""
        n = len(g.inv)
        relics: list[list[str]] = [[] for _ in range(n)]
        for r in v.t(Table.RELICS):
            if r[RelicCol.ZONE] == ItemZone.OWNED:
                relics[r[RelicCol.SEAT]].append(self._name("relic", r[RelicCol.RELIC_ID]))
        potions: list[list[str]] = [[] for _ in range(n)]
        for p in v.t(Table.POTIONS):
            if p[PotionCol.ZONE] == ItemZone.OWNED:
                potions[p[PotionCol.SEAT]].append(self._name("potion", p[PotionCol.POTION_ID]))
        decks: list[list[dict[str, Any]]] | None = None
        if not in_combat:
            decks = [[] for _ in range(n)]
            for row, c in enumerate(v.t(Table.CARDS)):
                if c[CardCol.ZONE] == Zone.DECK:
                    decks[c[CardCol.SEAT]].append(self._card(v, row))
            assert all(decks), "a frame outside combat carries every seat's deck (schema section 6)"
        out = []
        for s, (st, row) in enumerate(zip(g.inv, v.t(Table.SEATS))):
            changed = st.relics != relics[s] or st.potions != potions[s] or st.max_hp != row[SeatCol.MAX_HP]
            if decks is not None and st.deck != decks[s]:
                changed = True
                st.deck = decks[s]
            if decks is not None:
                st.deck_at = i
            st.relics, st.potions, st.max_hp = relics[s], potions[s], row[SeatCol.MAX_HP]
            st.hp, st.gold = row[SeatCol.HP], row[SeatCol.GOLD]
            if changed:
                out.append({"t": "inv", "i": i, "seat": s, "mhp": st.max_hp, "relics": st.relics,
                            "potions": st.potions, "deck": st.deck})
        return out

    def observe(self, env_id: int, f: Frame, action: int) -> None:
        gv = f.gvals
        assert gv[G.N_SEATS] > 1, "game log of a single-seat episode"
        g = self._open.get(env_id)
        if g is None or g.episode_id != f.episode_id:
            if g is not None:
                self._close(env_id)
            g = self._open[env_id] = self._new_game(env_id, f)
        i = g.decisions
        g.decisions += 1
        v = _View(f)
        kind = Kind(gv[G.KIND])
        g.kinds[kind.name] = g.kinds.get(kind.name, 0) + 1
        in_combat = bool(gv[G.IN_COMBAT])
        lines = []
        if in_combat:
            ck = (gv[G.TOTAL_FLOOR], gv[G.COMBATS_WON])
            if g.combat_key != ck:
                g.combat_key = ck
                g.combats += 1
                lines.append({"t": "combat", "cmb": g.combats, "i": i, "floor": gv[G.TOTAL_FLOOR],
                              "act": gv[G.ACT_INDEX], "room": self._name("enum:RoomType", gv[G.ROOM_TYPE]),
                              "enemies": [self._name("monster", c[CreatureCol.MODEL_ID]) for c in v.t(Table.CREATURES)
                                          if c[CreatureCol.KIND] == CreatureKind.ENEMY]})
        else:
            g.combat_key = None
        lines.extend(self._inventory(g, v, i, in_combat))
        a = v.t(Table.ACTIONS)[action]
        rec: dict[str, Any] = {"t": "d", "i": i, "step": f.step_id, "kind": kind.name, "seat": gv[G.ACTING_SEAT],
                               "floor": gv[G.TOTAL_FLOOR], "act": gv[G.ACT_INDEX],
                               "room": self._name("enum:RoomType", gv[G.ROOM_TYPE]), "n": len(v.t(Table.ACTIONS)),
                               "a": self._action(v, a)}
        verb = a[ActionCol.VERB]
        if kind == Kind.COMBAT:
            g.card_plays += verb == Verb.PLAY_CARD
            g.potion_uses += verb == Verb.USE_POTION
        if in_combat:
            rec["c"] = self._combat(g, v)
        else:
            rec["team"] = [{"hp": s[SeatCol.HP], "mhp": s[SeatCol.MAX_HP], "gold": s[SeatCol.GOLD]}
                           for s in v.t(Table.SEATS)]
        if v.t(Table.OPTIONS):
            rec["opts"] = [self._option(v, r) for r in range(len(v.t(Table.OPTIONS)))]
        if kind == Kind.CARD_REWARD:
            rec["offer"] = [self._card(v, r) for r, c in enumerate(v.t(Table.CARDS)) if c[CardCol.ZONE] == Zone.OFFER]
            if verb == Verb.PICK_OFFER_CARD:
                offers = [r for r, c in enumerate(v.t(Table.CARDS)) if c[CardCol.ZONE] == Zone.OFFER]
                rec["a"]["offer"] = offers.index(a[ActionCol.SRC_ROW])
        elif kind == Kind.CARD_SELECT:
            rec["sub"] = CardSelectSub(gv[G.KIND_SUB]).name
            rec["prompt"] = self._name("prompt_key", gv[G.PROMPT_KEY])
            if gv[G.PROMPT_SOURCE_CAT]:
                cat = REF_CATEGORY_VOCAB[gv[G.PROMPT_SOURCE_CAT]]
                rec["source"] = {"cat": cat, "id": self._name(cat, gv[G.PROMPT_SOURCE_ID])}
            rec["sel"] = [gv[G.SEL_MIN], gv[G.SEL_MAX], gv[G.SEL_COUNT]]
            rec["choices"] = [{**self._card(v, r), "sel": c[CardCol.SEL_COUNT]}
                              for r, c in enumerate(v.t(Table.CARDS)) if c[CardCol.ZONE] == Zone.CHOICE]
        elif kind == Kind.MAP:
            rows = sorted({x[ActionCol.SRC_ROW] for x in v.t(Table.ACTIONS) if x[ActionCol.VERB] == Verb.TRAVEL})
            rec["nodes"] = [self._node(v, r) for r in rows]
            rec["cur"] = [gv[G.CUR_ROW], gv[G.CUR_COL]]
        elif kind == Kind.EVENT:
            rec["event"] = self._name("event", gv[G.EVENT_ID])
            rec["page"] = self._name("event_page", gv[G.EVENT_PAGE_KEY])
        elif kind == Kind.TREASURE:
            rec["phase"] = TreasurePhase(gv[G.KIND_SUB]).name
        elif kind == Kind.BUNDLE:  # the bundles' cards (OFFER rows, CARDS.aux = bundle number); opts = the bundles
            rec["offer"] = [{**self._card(v, r), "bundle": c[CardCol.AUX]}
                            for r, c in enumerate(v.t(Table.CARDS)) if c[CardCol.ZONE] == Zone.OFFER]
        lines.append(rec)
        g.body.write("".join(_dumps(x) + "\n" for x in lines))

    def _new_game(self, env_id: int, f: Frame) -> _Game:
        gv = f.gvals
        chars = [self.characters[s[SeatCol.CHARACTER_ID]] for s in f.table(Table.SEATS).tolist()]
        self._serial += 1
        tmp = self.tmp / f"env{env_id}-ep{f.episode_id}-{self._serial}.body.gz"
        body = gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6)
        return _Game(env_id, f.episode_id, chars, gv[G.ASCENSION], lineup_key("+".join(chars), gv[G.ASCENSION]),
                     tmp, body, inv=[_SeatInv() for _ in chars])

    def _close(self, env_id: int) -> None:
        g = self._open.pop(env_id)
        g.body.close()
        self._done.setdefault(env_id, []).append(g)

    # ------------------------------------------------------------------ end of the evaluation
    def finish(self, episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Join the games of each host in order to ``episodes`` (the evaluation's records without shutdown ones;
        records with 0 decisions have no game) and write every game file; a host's last game without a record is
        ``unfinished``. Returns one index row per game."""
        for env_id in list(self._open):
            self._close(env_id)
        records: dict[int, list[Mapping[str, Any]]] = {}
        for rec in episodes:
            records.setdefault(int(rec["env"]), []).append(rec)
        rows = []
        for env_id in sorted(set(records) | set(self._done)):
            games = list(reversed(self._done.get(env_id, [])))
            for rec in records.get(env_id, []):
                if rec["decisions"] == 0:
                    continue
                g = games.pop()
                assert g.decisions == rec["decisions"] and g.key == lineup_key(rec["lineup"], rec["ascension"]), \
                    f"env {env_id}: logged game {g.key} / {g.decisions} decisions vs record {rec}"
                rows.append(self._write(g, rec))
            assert len(games) <= 1, f"env {env_id}: {len(games)} logged games without an episode record"
            for g in games:
                rows.append(self._write(g, None))
        self._done.clear()
        self.tmp.rmdir()
        return rows

    def _write(self, g: _Game, rec: Mapping[str, Any] | None) -> dict[str, Any]:
        code = lineup_code(g.chars)
        d = self.out_dir / f"{code}-A{g.ascension}"
        d.mkdir(parents=True, exist_ok=True)
        if rec is not None:
            path = d / f"{int(rec['seed_id']):02d}-{rec['seed']}.jsonl.gz"
        else:
            path = d / f"unfinished-env{g.env}-ep{g.episode_id}.jsonl.gz"
        assert not path.exists(), path
        result = rec["result"] if rec is not None else "unfinished"
        final = [{"seat": s, "character": ch, "deck": st.deck, "deck_at": st.deck_at, "relics": st.relics,
                  "potions": st.potions, "mhp": st.max_hp, "hp": st.hp, "gold": st.gold}
                 for s, (ch, st) in enumerate(zip(g.chars, g.inv))]
        header: dict[str, Any] = {
            "t": "game", "format": LOG_FORMAT, **self.meta, "lineup": code, "lineup_key": g.key, "seats": g.chars,
            "ascension": g.ascension, "env": g.env, "episode_id": g.episode_id, "result": result,
            "decisions": g.decisions, "card_plays": g.card_plays, "potion_uses": g.potion_uses, "combats": g.combats,
            "kinds": dict(sorted(g.kinds.items())), "final": final}
        if rec is not None:
            header.update({k: rec[k] for k in ("seed", "seed_id", "floor", "act", "room", "in_combat", "enemies",
                                               "terminal_reason", "combats_won", "elites_won", "bosses_won",
                                               "final_hp", "final_alive", "seat_deaths", "seconds", "auto_steps")})
        with open(path, "wb") as out:
            out.write(gzip.compress((_dumps(header) + "\n").encode("utf-8"), compresslevel=6))
            with open(g.tmp, "rb") as body:
                shutil.copyfileobj(body, out)
        g.tmp.unlink()
        return {"path": path.relative_to(self.out_dir).as_posix(), "lineup": code, "ascension": g.ascension,
                "seed": header.get("seed"), "seed_id": header.get("seed_id"), "result": result,
                "floor": header.get("floor"), "decisions": g.decisions, "card_plays": g.card_plays,
                "bytes": path.stat().st_size}


def read_game(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(header, body records) of one game file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert lines and lines[0]["t"] == "game", path
    return lines[0], lines[1:]
