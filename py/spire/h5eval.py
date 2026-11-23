"""Zero-shot H5 evaluation: one fixed panel, greedy and sampled, with frame-cost, co-op and debuff telemetry."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, NamedTuple, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from .actor import Policy, PolicyOut, TorchPolicy
from .evalpanel import evaluate, load_catalog, load_panel, load_policy_net, make_panel, save_panel
from .host import Host, real_host_spec
from .metrics import lineup_key
from .proto import Frame
from .schema import (ActionCol, CardCol, CreatureCol, CreatureKind, G, IntentCol, Kind, OptionCol, PowerCol, SeatCol,
                     Table, Verb)
from .synth import FakeHost, FakeHostConfig
from .tokenize import Sample
from .train import FAKE_HOST_SCRIPT, HostSection, make_hosts
from .vocab import Catalog, Vocab


def distribution(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    x = np.asarray(values)
    return {"n": len(values), "mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "max": int(x.max())}


class MeasuredPolicy:
    """Measure the actual policy inputs, including decisions in interrupted episodes."""
    needs_samples = True

    def __init__(self, policy: Policy, vocab: Vocab) -> None:
        self.policy, self.vocab = policy, vocab
        self.version = policy.version
        self.frames: dict[str, tuple[list[int], list[int]]] = {}
        # per decision kind (multi-action decisions only): [n, sum entropy, sum probability of the chosen action,
        # sum legal actions] of the distribution the policy acted with
        self.spread: dict[int, list[float]] = {}
        self._kinds: dict[int, list[tuple[int, int]]] = {}

    def launch(self, samples: Sequence[Sample], n_actions: np.ndarray) -> Any:
        assert len(samples) == len(n_actions)
        for sample in samples:
            n = int(sample.cond[1])
            key = "+".join(sorted(self.vocab["character"].entries[int(c)] for c in sample.cond[2:2 + n]))
            tokens, actions = self.frames.setdefault(key, ([], []))
            tokens.append(sample.n_tokens)
            actions.append(sample.n_actions)
        handle = self.policy.launch(samples, n_actions)
        kind_slot = getattr(self.policy, "_kind_slot", None)
        if kind_slot is not None:
            self._kinds[id(handle)] = [(int(s.raw[0, kind_slot]), int(n)) for s, n in zip(samples, n_actions)]
        return handle

    def done(self, handle: Any) -> bool:
        return self.policy.done(handle)

    def finish(self, handle: Any) -> PolicyOut:
        out = self.policy.finish(handle)
        kinds = self._kinds.pop(id(handle), None)
        if kinds is not None and out.entropy is not None:
            for (kind, n), entropy, logp in zip(kinds, out.entropy.tolist(), out.logp.tolist()):
                if n > 1:
                    row = self.spread.setdefault(kind, [0.0, 0.0, 0.0, 0.0])
                    row[0] += 1
                    row[1] += entropy
                    row[2] += float(np.exp(logp))
                    row[3] += n
        return out

    def spread_report(self) -> dict[str, Any]:
        from .schema import Kind
        return {Kind(k).name: {"n": int(r[0]), "entropy": r[1] / r[0], "p_chosen": r[2] / r[0], "legal": r[3] / r[0]}
                for k, r in sorted(self.spread.items())}

    def refresh(self) -> bool:
        changed = self.policy.refresh()
        self.version = self.policy.version
        return changed

    def report(self) -> dict[str, Any]:
        return {key: {"tokens_per_frame": distribution(t), "actions_per_frame": distribution(a)}
                for key, (t, a) in sorted(self.frames.items())}


MP_ONLY, ANY_ALLY, ALL_ALLIES = 1, 2, 4  # card_facts bits
CARD_CLASSES = (("mp_only", MP_ONLY), ("ally_target", ANY_ALLY | ALL_ALLIES), ("other", 0))


def card_facts(catalog: Catalog, vocab: Vocab) -> NDArray[np.int8]:
    """(card id, upgrade level) -> MP_ONLY | ANY_ALLY | ALL_ALLIES from catalog.json (0 for ids without an entry)."""
    entries = catalog.entries["card"]
    levels = max(len(e) for e in entries.values())
    out = np.zeros((vocab.size("card"), levels), np.int8)
    for i, e in entries.items():
        for lv in range(levels):
            rec = e[min(lv, len(e) - 1)]
            out[i, lv] = (MP_ONLY * (rec["multiplayer_constraint"] == "MultiplayerOnly")
                          | ANY_ALLY * (rec["target_type"] == "AnyAlly") | ALL_ALLIES * (rec["target_type"] == "AllAllies"))
    return out


def _in_class(bits: Any, mask: int) -> Any:
    """Card class membership of ``bits``: any of ``mask``, or (mask 0, "other") none of the co-op bits."""
    return (bits & mask) != 0 if mask else (bits & (MP_ONLY | ANY_ALLY | ALL_ALLIES)) == 0


def _share(n: float, of: float) -> dict[str, float | None]:
    return {"n": n, "of": of, "share": n / of if of else None}


class CoopTelemetry:
    """Co-operative play of the 4-seat controller, from the frames it decided on (``evaluate(on_decision=...)``).

    Per ``lineup|A<asc>`` counters of: card plays / potion uses aimed at another seat's creature and AllAllies plays;
    MultiplayerOnly plays and CARD_REWARD offers / picks by card class; REST choices when MEND was offered and the
    MEND target's HP rank; per combat turn (episode, floor, combats won, round) the seat order of the card plays and
    END_TURNs. Decisions of unfinished episodes count (they are decisions the policy took)."""

    def __init__(self, vocab: Vocab, catalog: Catalog) -> None:
        self.characters = vocab["character"].entries
        self.card_names = vocab["card"].entries
        self.rest_names = vocab["rest_option"].entries
        self.mend = vocab.id("rest_option", "MEND")
        self.cards = card_facts(catalog, vocab)
        self.counts: dict[str, Counter[str]] = {}
        self._turns: dict[int, tuple[tuple[int, ...], str, list[tuple[int, int]]]] = {}  # env -> key, lineup, events

    def _bits(self, cards: NDArray[np.int32]) -> NDArray[np.int8]:
        return self.cards[cards[:, CardCol.CARD_ID], np.minimum(cards[:, CardCol.UPGRADE_LEVEL], self.cards.shape[1] - 1)]

    def observe(self, env_id: int, f: Frame, action: int) -> None:
        g = f.gvals
        assert g[G.N_SEATS] > 1, "co-op telemetry of a single-seat episode"
        seats = f.table(Table.SEATS)
        key = lineup_key("+".join(self.characters[c] for c in seats[:, SeatCol.CHARACTER_ID]), g[G.ASCENSION])
        c = self.counts.setdefault(key, Counter())
        c["decisions"] += 1
        acts = f.actions
        verb, seat, _, src, _, aux = (int(x) for x in acts[action, :ActionCol.FLAGS])
        turn_key = (f.episode_id, g[G.TOTAL_FLOOR], g[G.COMBATS_WON], g[G.COMBAT_ROUND])
        turn = self._turns.get(env_id)
        if turn is not None and (not g[G.IN_COMBAT] or turn[0] != turn_key):
            self.close_turn(*self._turns.pop(env_id)[1:])
        kind = g[G.KIND]
        if kind == Kind.COMBAT:
            self._combat(c, f, acts, action, verb, src)
            if verb in (Verb.PLAY_CARD, Verb.END_TURN):
                self._turns.setdefault(env_id, (turn_key, key, []))[2].append((verb, seat))
        elif kind == Kind.CARD_REWARD:
            picks = np.flatnonzero(acts[:, ActionCol.VERB] == Verb.PICK_OFFER_CARD)
            bits = self._bits(f.table(Table.CARDS)[acts[picks, ActionCol.SRC_ROW]])
            chosen = int(bits[picks == action][0]) if verb == Verb.PICK_OFFER_CARD else None
            c["reward_decisions"] += 1
            c["reward_no_pick"] += chosen is None
            for name, mask in CARD_CLASSES:
                offered = _in_class(bits, mask)
                picked = chosen is not None and bool(_in_class(chosen, mask))
                c[f"reward_offered:{name}"] += int(offered.sum())
                c[f"reward_picked:{name}"] += picked
                c[f"reward_decisions_offering:{name}"] += bool(offered.any())
                c[f"reward_picked_when_offering:{name}"] += picked and bool(offered.any())
        elif kind == Kind.REST:
            opts = f.table(Table.OPTIONS)
            choose = acts[:, ActionCol.VERB] == Verb.CHOOSE_OPTION
            mend = np.zeros(len(acts), bool)
            mend[choose] = opts[acts[choose, ActionCol.SRC_ROW], OptionCol.KEY_ID] == self.mend
            label = self.rest_names[opts[src, OptionCol.KEY_ID]] if verb == Verb.CHOOSE_OPTION else Verb(verb).name
            c["rest_decisions"] += 1
            c[f"rest_choice:{label}"] += 1
            if mend.any():
                c["rest_mend_offered"] += 1
                c[f"rest_choice_mend_offered:{label}"] += 1
                for name in {self.rest_names[k] for k in opts[acts[choose, ActionCol.SRC_ROW], OptionCol.KEY_ID]}:
                    c[f"rest_offered_mend_offered:{name}"] += 1
            if mend[action]:
                targets = acts[mend, ActionCol.AUX]
                assert aux in targets and seat not in targets, "erratum H6: one MEND action per teammate"
                hp = seats[targets, SeatCol.HP] / np.maximum(seats[targets, SeatCol.MAX_HP], 1)
                lowest = hp == hp.min()
                c["mend_chosen"] += 1
                c["mend_target_lowest_hp"] += bool(lowest[targets == aux][0])
                c["mend_candidates"] += len(targets)
                c["mend_candidates_lowest_hp"] += int(lowest.sum())
                c["mend_target_full_hp"] += bool(seats[aux, SeatCol.HP] >= seats[aux, SeatCol.MAX_HP])
                c[f"mend_target_position:{int(np.flatnonzero(targets == aux)[0])}"] += 1  # among targets, seat order

    def _combat(self, c: Counter[str], f: Frame, acts: NDArray[np.int32], action: int, verb: int, src: int) -> None:
        cre = f.table(Table.CREATURES)
        tgt = acts[:, ActionCol.TGT_ROW]
        aimed = tgt >= 0
        ally = np.zeros(len(acts), bool)  # aimed at a player / pet of another seat
        ally[aimed] = (cre[tgt[aimed], CreatureCol.KIND] != CreatureKind.ENEMY) & \
            (cre[tgt[aimed], CreatureCol.SEAT] != acts[aimed, ActionCol.SEAT])
        c["combat_decisions"] += 1
        c["ally_legal"] += bool(ally.any())
        c["ally_taken"] += bool(ally[action])
        if verb == Verb.PLAY_CARD:
            bits = int(self._bits(f.table(Table.CARDS)[src:src + 1])[0])
            c["card_plays"] += 1
            c["ally_target_plays"] += bool(ally[action])
            c["all_allies_plays"] += bool(bits & ALL_ALLIES)
            if bits & MP_ONLY:
                c["mp_only_plays"] += 1
                c[f"mp_only_card:{self.card_names[f.table(Table.CARDS)[src, CardCol.CARD_ID]]}"] += 1
        elif verb == Verb.USE_POTION:
            c["potion_uses"] += 1
            c["potion_ally_target"] += bool(ally[action])

    def close_turn(self, key: str, events: list[tuple[int, int]]) -> None:
        """Count one combat turn of ``key``: its (verb, seat) PLAY_CARD / END_TURN decisions in order."""
        c = self.counts[key]
        plays = [s for v, s in events if v == Verb.PLAY_CARD]
        ends = [(i, s) for i, (v, s) in enumerate(events) if v == Verb.END_TURN]
        active = len(set(plays))
        switches = sum(a != b for a, b in zip(plays, plays[1:]))
        c["turns"] += 1
        c["turn_plays"] += len(plays)
        c["switches"] += switches
        c["min_switches"] += max(active - 1, 0)
        c["multi_seat_turns"] += active > 1
        c["block_turns"] += active > 1 and switches == active - 1
        if plays:
            c[f"first_play_seat:{plays[0]}"] += 1
        c["end_order_turns"] += len(ends) > 1
        c["end_order_ascending"] += len(ends) > 1 and [s for _, s in ends] == sorted(s for _, s in ends)
        c["end_turns"] += len(ends)
        c["end_then_teammate_plays"] += sum(any(v == Verb.PLAY_CARD and t != s for v, t in events[i + 1:])
                                            for i, s in ends)

    def report(self) -> dict[str, Any]:
        """Per ``lineup|A<asc>`` and pooled (``"all"``); closes the turns still open."""
        for _, key, events in self._turns.values():
            self.close_turn(key, events)
        self._turns.clear()
        pooled = sum(self.counts.values(), Counter[str]())
        return {key: coop_summary(c) for key, c in (*sorted(self.counts.items()), ("all", pooled))}


def coop_summary(c: Counter[str]) -> dict[str, Any]:
    plays, turn_plays = c["card_plays"], c["turn_plays"]

    def hist(prefix: str) -> dict[str, int]:
        return {k[len(prefix):]: v for k, v in sorted(c.items()) if k.startswith(prefix)}

    return {
        "decisions": c["decisions"],
        "ally_plays": {"card_plays": plays, "ally_target": _share(c["ally_target_plays"], plays),
                       "all_allies": _share(c["all_allies_plays"], plays),
                       "potion_ally_target": _share(c["potion_ally_target"], c["potion_uses"]),
                       "combat_decisions_with_ally_action": _share(c["ally_legal"], c["combat_decisions"]),
                       "ally_action_taken_when_legal": _share(c["ally_taken"], c["ally_legal"])},
        "mp_only_plays": {
            "plays": _share(c["mp_only_plays"], plays), "cards": hist("mp_only_card:"),
            "card_reward": {"decisions": c["reward_decisions"], "no_pick": c["reward_no_pick"], **{
                name: {"offered": c[f"reward_offered:{name}"], "picked": c[f"reward_picked:{name}"],
                       "pick_rate": _share(c[f"reward_picked:{name}"], c[f"reward_offered:{name}"])["share"],
                       "picked_when_offered": _share(c[f"reward_picked_when_offering:{name}"],
                                                     c[f"reward_decisions_offering:{name}"])}
                for name, _ in CARD_CLASSES}}},
        "mend": {"rest_decisions": c["rest_decisions"], "mend_offered": _share(c["rest_mend_offered"],
                                                                                c["rest_decisions"]),
                 "offered_when_mend_offered": hist("rest_offered_mend_offered:"),
                 "choices_when_mend_offered": hist("rest_choice_mend_offered:"), "choices": hist("rest_choice:"),
                 "target_lowest_hp": _share(c["mend_target_lowest_hp"], c["mend_chosen"]),
                 "target_lowest_hp_chance": _share(c["mend_candidates_lowest_hp"], c["mend_candidates"])["share"],
                 "target_full_hp": _share(c["mend_target_full_hp"], c["mend_chosen"]),
                 "target_position": hist("mend_target_position:")},
        "interleave": {"turns": c["turns"], "plays": turn_plays, "switches": c["switches"],
                       "min_switches": c["min_switches"], "extra_switches": c["switches"] - c["min_switches"],
                       "switches_per_play": _share(c["switches"], turn_plays)["share"],
                       "min_switches_per_play": _share(c["min_switches"], turn_plays)["share"],
                       "block_turns": _share(c["block_turns"], c["multi_seat_turns"])},
        "turn_end_order": {"first_play_seat": hist("first_play_seat:"),
                           "ascending": _share(c["end_order_ascending"], c["end_order_turns"]),
                           "end_then_teammate_plays": _share(c["end_then_teammate_plays"], c["end_turns"])},
    }


VULN, WEAK, ATTACK, MULTI_ENEMY = 1, 2, 4, 8  # debuff_cards bits
DEBUFFS = (("vulnerable", VULN), ("weak", WEAK))
DEBUFF_POWERS = ("VULNERABLE_POWER", "WEAK_POWER", "ARTIFACT_POWER")  # columns of _Enemies.amount / .has
ENEMY_TARGETS = ("AnyEnemy", "AllEnemies", "RandomEnemy")
ATTACK_INTENTS = ("Attack", "DeathBlow")
FIGHT_ROOMS = ("Monster", "Elite", "Boss")  # any other room type of a combat counts as "Other" (event fights)
REWARD_CLASSES = (("vulnerable", VULN), ("weak", WEAK), ("debuff", VULN | WEAK), ("other", 0))
RESULTS = ("win", "loss", "other", "unfinished")
_SENTENCE = re.compile(r"\.(?:\s|$)")
_APPLY = re.compile(r"\bappl(?:y|ies)\b", re.IGNORECASE)


def debuff_cards(catalog: Catalog, vocab: Vocab) -> NDArray[np.int8]:
    """(card id, upgrade level) -> VULN | WEAK | ATTACK | MULTI_ENEMY from catalog.json (0 for ids without an entry).

    A card applies Vulnerable (Weak) to enemies when its target type is an enemy one (``ENEMY_TARGETS``) and a
    sentence of its text (level 0; the catalog prints it once) says apply / applies ... Vulnerable (Weak). Cards that
    only read or scale the debuff (BULLY, DISMANTLE, MOLTEN_FIST, DEBILITATE, CRUELTY, ...) are not in the set."""
    entries = catalog.entries["card"]
    levels = max(len(e) for e in entries.values())
    out = np.zeros((vocab.size("card"), levels), np.int8)
    for i, e in entries.items():
        applies = [s for s in _SENTENCE.split(e[0]["text"]) if _APPLY.search(s)]
        for lv in range(levels):
            rec = e[min(lv, len(e) - 1)]
            enemy = rec["target_type"] in ENEMY_TARGETS
            out[i, lv] = (VULN * (enemy and any("Vulnerable" in s for s in applies))
                          | WEAK * (enemy and any("Weak" in s for s in applies))
                          | ATTACK * (rec["type"] == "Attack") | MULTI_ENEMY * (rec["target_type"] in ENEMY_TARGETS[1:]))
    return out


class _Enemies(NamedTuple):
    """The enemy rows of one combat frame: CREATURES rows, identity (slot, monster, n-th of that pair), whether
    alive, and per DEBUFF_POWERS column the summed amount and whether the power is present."""
    rows: NDArray[np.int64]
    keys: list[tuple[int, int, int]]
    alive: NDArray[np.bool_]
    amount: NDArray[np.int64]
    has: NDArray[np.bool_]


@dataclass
class _Combat:
    floor_won: tuple[int, int]  # (TOTAL_FLOOR, COMBATS_WON): one combat of the episode
    room: str
    act: int
    hp: NDArray[np.int64]  # per seat, at the previous combat decision
    alive: NDArray[np.bool_]
    turns: int = 0
    round: int = -1  # COMBAT_ROUND of the current team turn
    last: tuple[Frame, _Enemies] | None = None  # latest frame of the current team turn


@dataclass
class _Pending:
    """A card play / potion use whose effect on the enemies is read at the next COMBAT decision of its turn."""
    round: int
    before: _Enemies
    char: str
    label: str  # "card" | "potion"
    bits: int


@dataclass
class _Episode:
    episode_id: int
    key: str
    chars: list[str]
    decisions: int = 0
    counts: Counter[tuple[Any, ...]] = field(default_factory=Counter)
    tail: Counter[tuple[Any, ...]] = field(default_factory=Counter)  # added only if the episode was lost
    combat: _Combat | None = None
    pending: _Pending | None = None


class DebuffTelemetry:
    """Vulnerable / Weak on enemies in 4-seat evaluation games, from the frames decided on (``on_decision``).

    Per ``lineup|A<asc>`` and game result (win / loss / other / unfinished), joined to the episode records by
    :meth:`report`. Counted per combat (room type Monster / Elite / Boss / Other, act):
    - uptime: one sample per team turn (its first decision): living enemies with Vulnerable / Weak;
    - applications: a card play / potion use after which an enemy's Vulnerable / Weak amount went up (read at the
      next COMBAT decision of the same turn), by the acting seat's character; Artifact decrements;
    - use: attack card plays that hit a Vulnerable enemy; enemies with an attack intent at the last decision of a
      team turn followed by an enemy turn, and whether they were Weak;
    - fights: team turns, HP lost per seat (sum of decreases between decisions, including the first frame after
      the combat; in a lost game the HP still left at the last decision counts as lost) and seat deaths;
    - card rewards: offers / picks of the Vulnerable / Weak card set (``debuff_cards``) by the seat's character."""

    def __init__(self, vocab: Vocab, catalog: Catalog) -> None:
        self.characters = vocab["character"].entries
        self.rooms = vocab["enum:RoomType"].entries
        for cat, names in (("power", DEBUFF_POWERS), ("enum:IntentType", ATTACK_INTENTS)):
            assert all(n in vocab[cat].index for n in names), f"vocab {cat} lacks one of {names}"
        self.powers = vocab.ids("power", DEBUFF_POWERS)
        self.attack_intents = vocab.ids("enum:IntentType", ATTACK_INTENTS)
        self.cards = debuff_cards(catalog, vocab)
        self._open: dict[int, _Episode] = {}
        self._done: dict[int, list[_Episode]] = {}
        self.episode_rows: list[dict[str, Any]] = []  # filled by report(): one row of raw counts per episode

    def observe(self, env_id: int, f: Frame, action: int) -> None:
        g = f.gvals
        assert g[G.N_SEATS] > 1, "debuff telemetry of a single-seat episode"
        e = self._open.get(env_id)
        if e is None or e.episode_id != f.episode_id:
            if e is not None:
                self._end_episode(env_id)
            chars = [self.characters[c] for c in f.table(Table.SEATS)[:, SeatCol.CHARACTER_ID]]
            e = self._open[env_id] = _Episode(f.episode_id, lineup_key("+".join(chars), g[G.ASCENSION]), chars)
        e.decisions += 1
        if g[G.IN_COMBAT]:
            self._combat_frame(e, f, g, action)
            return
        if e.combat is not None:
            self._close_combat(e, f.table(Table.SEATS))
        if g[G.KIND] == Kind.CARD_REWARD:
            self._reward(e, f, action)

    def _enemies(self, f: Frame) -> _Enemies:
        cre, pw = f.table(Table.CREATURES), f.table(Table.POWERS)
        rows = np.flatnonzero(cre[:, CreatureCol.KIND] == CreatureKind.ENEMY)
        amount = np.zeros((len(cre), len(self.powers)), np.int64)
        has = np.zeros((len(cre), len(self.powers)), bool)
        for j, pid in enumerate(self.powers):
            sel = pw[:, PowerCol.POWER_ID] == pid
            np.add.at(amount[:, j], pw[sel, PowerCol.OWNER_ROW], pw[sel, PowerCol.AMOUNT])
            has[pw[sel, PowerCol.OWNER_ROW], j] = True
        seen: Counter[tuple[int, int]] = Counter()
        keys = []
        for r in rows:
            k = (int(cre[r, CreatureCol.SLOT_ID]), int(cre[r, CreatureCol.MODEL_ID]))
            keys.append((*k, seen[k]))
            seen[k] += 1
        return _Enemies(rows, keys, cre[rows, CreatureCol.ALIVE] != 0, amount[rows], has[rows])

    def _combat_frame(self, e: _Episode, f: Frame, g: tuple[int, ...], action: int) -> None:
        seats = f.table(Table.SEATS)
        floor_won = (g[G.TOTAL_FLOOR], g[G.COMBATS_WON])
        if e.combat is not None and e.combat.floor_won != floor_won:
            self._close_combat(e, None)
        c = e.counts
        if e.combat is None:
            name = self.rooms[g[G.ROOM_TYPE]]
            e.combat = _Combat(floor_won, name if name in FIGHT_ROOMS else "Other", int(g[G.ACT_INDEX]),
                               seats[:, SeatCol.HP].astype(np.int64), seats[:, SeatCol.ALIVE] != 0)
            assert (e.combat.hp >= 0).all(), "TEAM view shows every seat's HP"
            cb = e.combat
            c[("cmb", cb.room, cb.act, "combats")] += 1
            c[("cmb", cb.room, cb.act, "seat_combats")] += len(e.chars)
            for ch in e.chars:
                c[("cmbc", cb.room, ch, "seat_combats")] += 1
        else:
            self._hp(e, c, seats)
        cb = e.combat
        en = self._enemies(f)
        rnd = int(g[G.COMBAT_ROUND])
        if rnd != cb.round:
            if cb.last is not None:  # the previous team turn of this combat was followed by an enemy turn
                self._enemy_turn(c, cb, *cb.last)
            cb.round = rnd
            cb.turns += 1
            self._uptime(c, cb, en)
        cb.last = (f, en)
        kind = g[G.KIND]
        if kind != Kind.COMBAT:  # a prompt inside the turn (card selection, ...): a pending effect may follow it
            return
        if e.pending is not None:
            p, e.pending = e.pending, None
            if p.round == rnd:
                self._resolve(c, p, en)
            else:
                c[("app_unresolved", p.label)] += 1
        acts = f.actions
        verb, seat, _, src, tgt = (int(x) for x in acts[action, :ActionCol.AUX])
        if verb == Verb.PLAY_CARD:
            card = f.table(Table.CARDS)[src]
            bits = int(self.cards[card[CardCol.CARD_ID], min(int(card[CardCol.UPGRADE_LEVEL]), self.cards.shape[1] - 1)])
            if bits & ATTACK:
                self._use(c, cb, en, tgt, bits)
            e.pending = _Pending(rnd, en, e.chars[seat], "card", bits)
        elif verb == Verb.USE_POTION:
            e.pending = _Pending(rnd, en, e.chars[seat], "potion", 0)

    def _hp(self, e: _Episode, c: Counter[tuple[Any, ...]], seats: NDArray[np.int32]) -> None:
        cb = e.combat
        assert cb is not None
        hp, alive = seats[:, SeatCol.HP].astype(np.int64), seats[:, SeatCol.ALIVE] != 0
        assert (hp >= 0).all(), "TEAM view shows every seat's HP"
        lost = np.maximum(cb.hp - hp, 0)
        for s, ch in enumerate(e.chars):
            c[("cmbc", cb.room, ch, "hp_lost")] += int(lost[s])
        c[("cmb", cb.room, cb.act, "hp_lost")] += int(lost.sum())
        c[("cmb", cb.room, cb.act, "deaths")] += int((cb.alive & ~alive).sum())
        cb.hp, cb.alive = hp, alive

    def _close_combat(self, e: _Episode, seats: NDArray[np.int32] | None) -> None:
        """End the open combat; ``seats`` = SEATS of the first frame after it (None: no such frame)."""
        cb = e.combat
        assert cb is not None
        if seats is not None:
            self._hp(e, e.counts, seats)
        e.counts[("cmb", cb.room, cb.act, "turns")] += cb.turns
        if e.pending is not None:
            e.counts[("app_unresolved", e.pending.label)] += 1
        e.combat = e.pending = None

    def _uptime(self, c: Counter[tuple[Any, ...]], cb: _Combat, en: _Enemies) -> None:
        live = en.alive
        if not live.any():
            c[("up", cb.room, cb.act, "turns_no_enemy")] += 1
            return
        v, w = en.has[live, 0], en.has[live, 1]
        c[("up", cb.room, cb.act, "turns")] += 1
        c[("up", cb.room, cb.act, "turns_vulnerable")] += bool(v.any())
        c[("up", cb.room, cb.act, "turns_weak")] += bool(w.any())
        c[("up", cb.room, cb.act, "turns_both")] += bool(v.any() and w.any())
        c[("up", cb.room, cb.act, "enemies")] += int(live.sum())
        c[("up", cb.room, cb.act, "enemies_vulnerable")] += int(v.sum())
        c[("up", cb.room, cb.act, "enemies_weak")] += int(w.sum())

    def _enemy_turn(self, c: Counter[tuple[Any, ...]], cb: _Combat, f: Frame, en: _Enemies) -> None:
        """Enemies that attack after the team turn whose last decision was ``f``, and whether they are Weak."""
        intents = f.table(Table.INTENTS)
        attackers = intents[np.isin(intents[:, IntentCol.INTENT_TYPE], self.attack_intents), IntentCol.OWNER_ROW]
        att = en.alive & np.isin(en.rows, attackers)
        c[("eatk", cb.room, cb.act, "attackers")] += int(att.sum())
        c[("eatk", cb.room, cb.act, "attackers_weak")] += int((att & en.has[:, 1]).sum())
        c[("eatk", cb.room, cb.act, "enemy_turns")] += 1
        c[("eatk", cb.room, cb.act, "enemy_turns_attack")] += bool(att.any())

    def _use(self, c: Counter[tuple[Any, ...]], cb: _Combat, en: _Enemies, tgt: int, bits: int) -> None:
        hit = np.flatnonzero(en.rows == tgt)
        c[("use", cb.room, cb.act, "attack_plays")] += 1
        if len(hit):
            vuln = bool(en.has[hit[0], 0])
            c[("use", cb.room, cb.act, "targeted")] += 1
            c[("use", cb.room, cb.act, "targeted_vulnerable")] += vuln
        elif bits & MULTI_ENEMY:
            vuln = bool((en.has[:, 0] & en.alive).any())
            c[("use", cb.room, cb.act, "multi")] += 1
            c[("use", cb.room, cb.act, "multi_any_vulnerable")] += vuln
        else:  # no enemy target (Osty / self-targeted attacks)
            vuln = False
            c[("use", cb.room, cb.act, "no_enemy_target")] += 1
        c[("use", cb.room, cb.act, "hit_vulnerable")] += vuln

    def _resolve(self, c: Counter[tuple[Any, ...]], p: _Pending, en: _Enemies) -> None:
        now = {k: i for i, k in enumerate(en.keys)}
        pairs = [(i, now[k]) for i, k in enumerate(p.before.keys) if k in now]
        before = p.before.amount[[i for i, _ in pairs]]
        after = en.amount[[j for _, j in pairs]]
        up = (after > before).sum(axis=0) if pairs else np.zeros(len(self.powers), np.int64)
        artifact_down = bool(pairs and (after[:, 2] < before[:, 2]).any())
        c[("app", p.char, p.label, "plays")] += 1
        c[("app", p.char, p.label, "artifact_down")] += artifact_down
        for j, (name, bit) in enumerate(DEBUFFS):
            c[("app", p.char, p.label, name)] += bool(up[j])
            c[("app", p.char, p.label, f"{name}_enemies")] += int(up[j])
            if p.bits & bit:
                c[("set", p.char, name, "plays")] += 1
                c[("set", p.char, name, "landed")] += bool(up[j])
                c[("set", p.char, name, "artifact_blocked")] += not up[j] and artifact_down

    def _reward(self, e: _Episode, f: Frame, action: int) -> None:
        acts = f.actions
        picks = np.flatnonzero(acts[:, ActionCol.VERB] == Verb.PICK_OFFER_CARD)
        seats = set(acts[picks, ActionCol.SEAT].tolist())
        assert len(seats) == 1, f"card reward offers of seats {seats}"
        ch = e.chars[seats.pop()]
        offers = f.table(Table.CARDS)[acts[picks, ActionCol.SRC_ROW]]
        bits = self.cards[offers[:, CardCol.CARD_ID], np.minimum(offers[:, CardCol.UPGRADE_LEVEL], self.cards.shape[1] - 1)]
        chosen = int(bits[picks == action][0]) if acts[action, ActionCol.VERB] == Verb.PICK_OFFER_CARD else None
        c = e.counts
        c[("rew", ch, "decisions")] += 1
        c[("rew", ch, "no_pick")] += chosen is None
        for name, mask in REWARD_CLASSES:
            offered = (bits & mask) != 0 if mask else (bits & (VULN | WEAK)) == 0
            picked = chosen is not None and bool((chosen & mask) != 0 if mask else (chosen & (VULN | WEAK)) == 0)
            c[("rew", ch, name, "offered")] += int(offered.sum())
            c[("rew", ch, name, "picked")] += picked
            c[("rew", ch, name, "decisions_offering")] += bool(offered.any())
            c[("rew", ch, name, "picked_when_offering")] += picked and bool(offered.any())

    def _end_episode(self, env_id: int) -> None:
        e = self._open.pop(env_id)
        cb = e.combat
        if cb is not None:  # the episode ended inside this combat: if it was lost, the team died in it
            alive = cb.alive
            for s, ch in enumerate(e.chars):
                e.tail[("cmbc", cb.room, ch, "hp_lost")] += int(cb.hp[s]) if alive[s] else 0
            e.tail[("cmb", cb.room, cb.act, "hp_lost")] += int(cb.hp[alive].sum())
            e.tail[("cmb", cb.room, cb.act, "deaths")] += int(alive.sum())
            if cb.last is not None:
                self._enemy_turn(e.tail, cb, *cb.last)
            self._close_combat(e, None)
        self._done.setdefault(env_id, []).append(e)

    def report(self, episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Per ``lineup|A<asc>`` and pooled (``"all"``), per result and ``"all"``: the telemetry episodes of each
        host joined in order to ``episodes`` (the evaluation's episode records, without shutdown ones); a host's
        last telemetry episode without a record is ``unfinished``. Ends the open episodes."""
        for env_id in list(self._open):
            self._end_episode(env_id)
        records: dict[int, list[Mapping[str, Any]]] = {}
        for rec in episodes:
            records.setdefault(int(rec["env"]), []).append(rec)
        counts: dict[str, dict[str, Counter[tuple[Any, ...]]]] = {}
        games: dict[str, Counter[str]] = {}

        def add(ep: _Episode, result: str, rec: Mapping[str, Any] | None) -> None:
            c = ep.counts + ep.tail if result == "loss" else ep.counts
            for key in (ep.key, "all"):
                counts.setdefault(key, {}).setdefault(result, Counter()).update(c)
                games.setdefault(key, Counter())[result] += 1
            self.episode_rows.append({"env": env_id, "key": ep.key, "result": result, "decisions": ep.decisions,
                                      **({} if rec is None else {k: rec[k] for k in ("seed", "seed_id", "floor")}),
                                      "counts": {"|".join(map(str, k)): v for k, v in sorted(c.items(), key=str)}})

        for env_id in sorted(set(records) | set(self._done)):
            tel = list(self._done.get(env_id, []))
            tel.reverse()
            for rec in records.get(env_id, []):
                if rec["decisions"] == 0:
                    continue
                ep = tel.pop()
                assert ep.decisions == rec["decisions"] and ep.key == lineup_key(rec["lineup"], rec["ascension"]), \
                    f"env {env_id}: telemetry episode {ep.key} / {ep.decisions} decisions vs record {rec}"
                add(ep, rec["result"] if rec["result"] in ("win", "loss") else "other", rec)
            assert len(tel) <= 1, f"env {env_id}: {len(tel)} telemetry episodes without an episode record"
            for ep in tel:
                add(ep, "unfinished", None)
        self._done.clear()
        out: dict[str, Any] = {}
        for key in sorted(counts, key=lambda k: (k == "all", k)):
            by_result = counts[key]
            by_result["all"] = sum(by_result.values(), Counter())
            out[key] = {r: {"games": games[key][r] if r != "all" else sum(games[key].values()),
                            **debuff_summary(by_result[r])}
                        for r in (*RESULTS, "all") if r in by_result}
        return out


def _marginals(c: Counter[tuple[Any, ...]], section: str, fields: Sequence[str],
               make: Callable[[Counter[str]], dict[str, Any]], joint: bool = False) -> dict[str, Any]:
    """``make`` over (section, room, act, field) counts: all, per act ("act1".."act3"), per room and (``joint``)
    per act and room ("act1:Elite")."""
    total: Counter[str] = Counter()
    groups: dict[str, dict[str, Counter[str]]] = {"by_act": {}, "by_room": {}, "by_act_room": {}}
    for k, v in c.items():
        if k[0] == section:
            _, room, act, name = k
            assert name in fields, k
            total[name] += v
            groups["by_act"].setdefault(f"act{act + 1}", Counter())[name] += v
            groups["by_room"].setdefault(room, Counter())[name] += v
            groups["by_act_room"].setdefault(f"act{act + 1}:{room}", Counter())[name] += v
    if not joint:
        del groups["by_act_room"]
    return {"all": make(total), **{g: {k: make(x) for k, x in sorted(xs.items())} for g, xs in groups.items()}}


def _ratio(n: float, of: float) -> float | None:
    return n / of if of else None


def debuff_summary(c: Counter[tuple[Any, ...]]) -> dict[str, Any]:
    def uptime(x: Counter[str]) -> dict[str, Any]:
        return {"turns": x["turns"], "turns_no_enemy": x["turns_no_enemy"],
                "any_vulnerable": _ratio(x["turns_vulnerable"], x["turns"]),
                "any_weak": _ratio(x["turns_weak"], x["turns"]), "both": _ratio(x["turns_both"], x["turns"]),
                "enemies": x["enemies"], "enemies_vulnerable": _ratio(x["enemies_vulnerable"], x["enemies"]),
                "enemies_weak": _ratio(x["enemies_weak"], x["enemies"])}

    def use(x: Counter[str]) -> dict[str, Any]:
        return {"attack_plays": x["attack_plays"], "hit_vulnerable": _share(x["hit_vulnerable"], x["attack_plays"]),
                "targeted_vulnerable": _share(x["targeted_vulnerable"], x["targeted"]),
                "multi_any_vulnerable": _share(x["multi_any_vulnerable"], x["multi"]),
                "no_enemy_target": x["no_enemy_target"]}

    def enemy_attacks(x: Counter[str]) -> dict[str, Any]:
        return {"enemy_turns": x["enemy_turns"], "with_attack": _share(x["enemy_turns_attack"], x["enemy_turns"]),
                "attacker_weak": _share(x["attackers_weak"], x["attackers"])}

    def fights(x: Counter[str]) -> dict[str, Any]:
        return {"combats": x["combats"], "turns_per_combat": _ratio(x["turns"], x["combats"]),
                "hp_lost_per_seat_combat": _ratio(x["hp_lost"], x["seat_combats"]),
                "deaths_per_combat": _ratio(x["deaths"], x["combats"]), "turns": x["turns"], "hp_lost": x["hp_lost"],
                "deaths": x["deaths"]}

    per_char: dict[str, dict[tuple[str, ...], int]] = {}
    for k, v in c.items():
        if k[0] in ("app", "set", "rew"):
            per_char.setdefault(k[1], {})[(k[0], *k[2:])] = v
            per_char.setdefault("all", {})
            all_key = (k[0], *k[2:])
            per_char["all"][all_key] = per_char["all"].get(all_key, 0) + v
    seat_combats: Counter[str] = Counter()
    hp_char: dict[str, dict[str, dict[str, Any]]] = {}
    for k, v in c.items():
        if k[0] == "cmbc":
            _, room, ch, name = k
            row = hp_char.setdefault(ch, {}).setdefault(room, {"seat_combats": 0, "hp_lost": 0})
            row[name] += v
            if name == "seat_combats":
                seat_combats[ch] += v
                seat_combats["all"] += v
    for rows in hp_char.values():
        for row in rows.values():
            row["hp_lost_per_seat_combat"] = _ratio(row["hp_lost"], row["seat_combats"])

    def applications(x: dict[tuple[str, ...], int], ch: str) -> dict[str, Any]:
        plays = x.get(("app", "card", "plays"), 0)
        out: dict[str, Any] = {"card_plays": plays, "seat_combats": seat_combats[ch],
                               "unresolved_plays": c[("app_unresolved", "card")] if ch == "all" else None,
                               "artifact_down_plays": x.get(("app", "card", "artifact_down"), 0)}
        for name, _ in DEBUFFS:
            n = x.get(("app", "card", name), 0)
            out[name] = {"plays_applying": n, "enemies": x.get(("app", "card", f"{name}_enemies"), 0),
                         "per_100_plays": _ratio(100 * n, plays), "per_seat_combat": _ratio(n, seat_combats[ch]),
                         "potion_uses_applying": x.get(("app", "potion", name), 0),
                         "set_card_plays": x.get(("set", name, "plays"), 0),
                         "set_landed": _share(x.get(("set", name, "landed"), 0), x.get(("set", name, "plays"), 0)),
                         "set_artifact_blocked": _share(x.get(("set", name, "artifact_blocked"), 0),
                                                        x.get(("set", name, "plays"), 0))}
        out["potion_uses"] = x.get(("app", "potion", "plays"), 0)
        return out

    def rewards(x: dict[tuple[str, ...], int]) -> dict[str, Any]:
        out: dict[str, Any] = {"decisions": x.get(("rew", "decisions"), 0), "no_pick": x.get(("rew", "no_pick"), 0)}
        for name, _ in REWARD_CLASSES:
            offered, picked = x.get(("rew", name, "offered"), 0), x.get(("rew", name, "picked"), 0)
            out[name] = {"offered": offered, "picked": picked, "pick_rate": _ratio(picked, offered),
                         "picked_when_offered": _share(x.get(("rew", name, "picked_when_offering"), 0),
                                                       x.get(("rew", name, "decisions_offering"), 0))}
        return out

    return {
        "uptime": _marginals(c, "up", ("turns", "turns_no_enemy", "turns_vulnerable", "turns_weak", "turns_both",
                                       "enemies", "enemies_vulnerable", "enemies_weak"), uptime, joint=True),
        "use": _marginals(c, "use", ("attack_plays", "hit_vulnerable", "targeted", "targeted_vulnerable", "multi",
                                     "multi_any_vulnerable", "no_enemy_target"), use),
        "enemy_attacks": _marginals(c, "eatk", ("attackers", "attackers_weak", "enemy_turns", "enemy_turns_attack"),
                                    enemy_attacks),
        "fights": _marginals(c, "cmb", ("combats", "seat_combats", "turns", "hp_lost", "deaths"), fights,
                             joint=True),
        "hp_lost_by_character": hp_char,
        "applications": {ch: applications(x, ch) for ch, x in sorted(per_char.items())
                         if ch == "all" or seat_combats[ch]},
        "card_rewards": {ch: rewards(x) for ch, x in sorted(per_char.items()) if ("rew", "decisions") in x},
    }


FOCUS_ROOMS = (("boss", "Boss"), ("elite", "Elite"))
FOCUS_REST = ("HEAL", "SMITH", "MEND")  # REST choices counted by name where MEND is offered; the rest is "other"


def _round(x: float | None, digits: int) -> float | None:
    return None if x is None else round(x, digits)


def focus_summary(summary: Mapping[str, Mapping[str, Any]], coop: Mapping[str, Mapping[str, Any]],
                  debuff: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A few numbers per ``lineup|A<asc>`` (the keys of ``summary``, which include ``"all"``) for the hourly history
    rows, from the ``summary``, ``coop`` and ``debuff`` blocks of one mode:

    - ``wins`` / ``n`` (finished episodes);
    - ``boss`` / ``elite`` (all results pooled; ``None`` when no game of the cell reached such a fight): ``fights``;
      ``attackers`` (enemies with an attack intent at the team's last decision before an enemy turn) and the share of
      them that are Weak (``attackers_weak``); the team's ``attack_plays`` and the share that hit a Vulnerable enemy
      (``hit_vulnerable``); ``turns_per_fight`` (team turns), ``hp_lost_per_player`` (per seat and fight) and
      ``deaths_per_fight``. A share of nothing (no attacker, no attack play) is ``None``;
    - ``rest``: REST decisions where MEND was offered (``mend_offered``) and the choices there (``HEAL`` / ``SMITH`` /
      ``MEND`` / ``other``); of the MEND choices, the share whose target was at full HP (``mend_full_hp``) and the
      share on the lowest-HP team mate (``mend_lowest_hp``) next to its chance level (``mend_lowest_hp_chance``:
      lowest-HP candidates / candidates).
    ``summary`` holds finished episodes only, so every key of it has co-op and debuff telemetry; telemetry keys
    without a finished episode (a partial round) are not in the result."""
    out: dict[str, Any] = {}
    for key, s in summary.items():
        row: dict[str, Any] = {"wins": s["wins"], "n": s["n"]}
        d = debuff[key]["all"]
        for name, room in FOCUS_ROOMS:
            if room not in d["fights"]["by_room"]:
                row[name] = None
                continue
            f = d["fights"]["by_room"][room]
            # use / enemy_attacks have a room only once it saw an attack play / an enemy turn
            use = d["use"]["by_room"].get(room, {"attack_plays": 0, "hit_vulnerable": {"share": None}})
            atk = d["enemy_attacks"]["by_room"].get(room, {"attacker_weak": {"of": 0, "share": None}})
            row[name] = {"fights": f["combats"], "attackers": atk["attacker_weak"]["of"],
                         "attackers_weak": _round(atk["attacker_weak"]["share"], 4),
                         "attack_plays": use["attack_plays"], "hit_vulnerable": _round(use["hit_vulnerable"]["share"], 4),
                         "turns_per_fight": _round(f["turns_per_combat"], 3),
                         "hp_lost_per_player": _round(f["hp_lost_per_seat_combat"], 3),
                         "deaths_per_fight": _round(f["deaths_per_combat"], 4)}
        m = coop[key]["mend"]
        choices = m["choices_when_mend_offered"]
        assert sum(choices.values()) == m["mend_offered"]["n"], (key, choices, m["mend_offered"])
        assert m["target_full_hp"]["of"] == choices.get("MEND", 0), (key, m)
        row["rest"] = {"mend_offered": m["mend_offered"]["n"], **{c: choices.get(c, 0) for c in FOCUS_REST},
                       "other": sum(v for c, v in choices.items() if c not in FOCUS_REST),
                       "mend_full_hp": _round(m["target_full_hp"]["share"], 4),
                       "mend_lowest_hp": _round(m["target_lowest_hp"]["share"], 4),
                       "mend_lowest_hp_chance": _round(m["target_lowest_hp_chance"], 4)}
        out[key] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--panel", type=Path, required=True, help="existing panel or destination of a new frozen panel")
    ap.add_argument("--lineups", nargs="+", default=["DEFECT+DEFECT+IRONCLAD+IRONCLAD"])
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--host-config", help="stable host launch TOML; omit for in-process fake protocol host")
    ap.add_argument("--fake-proc", action="store_true", help="serve fake H5 over stdio instead of in-process")
    ap.add_argument("--vocab", type=Path, help="explicit checkpoint vocabulary location (fake mode)")
    ap.add_argument("--catalog", type=Path, help="explicit checkpoint catalog location (fake mode)")
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", choices=("ema", "online"), default="ema")
    ap.add_argument("--modes", nargs="+", choices=("greedy", "sample"), default=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-episode-decisions", type=int, default=20000, help="per-seat cap; actor multiplies by seats")
    ap.add_argument("--round-budget-s", type=float, default=1200.0)
    ap.add_argument("--fake-mean-decisions", type=float, default=100.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--debuff-episodes", action="store_true",
                    help="also write the per-episode debuff counts (debuff_episodes) into the per-mode report")
    a = ap.parse_args(argv)
    assert 1 <= a.envs <= 16, "at most 16 hosts per evaluation (the machine also runs the training hosts)"
    assert not (a.host_config and a.fake_proc)
    torch.set_num_threads(4)
    spec = real_host_spec(a.host_config) if a.host_config else None
    net, tok, version, extra = load_policy_net(a.checkpoint, device=a.device, weights=a.weights,
                                             vocab_path=spec.vocab if spec else a.vocab,
                                             catalog_path=spec.catalog if spec else a.catalog)
    if a.panel.exists():
        panel = load_panel(a.panel)
    else:
        panel = make_panel(a.panel.stem, a.seeds, a.lineups, [a.ascension], rng_seed=a.seed)
        save_panel(panel, a.panel.parent)
    assert all(len(lineup.split("+")) == 4 for lineup in panel.lineups), "H5 zero-shot panel must have four seats"
    catalog = load_catalog(str(spec.catalog if spec else a.catalog or extra["catalog"]), tok.vocab)
    assert catalog is not None, "co-op telemetry needs the card catalog"
    a.out.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for mode_name in a.modes:
        mode: Literal["greedy", "sample"] = mode_name
        policy = TorchPolicy(net, tok, mode=mode, temperature=a.temperature, seed=a.seed,
                             version=version, use_graphs=a.device != "cpu")
        measured = MeasuredPolicy(policy, tok.vocab)
        coop = CoopTelemetry(tok.vocab, catalog)
        debuff = DebuffTelemetry(tok.vocab, catalog)

        def on_decision(env_id: int, f: Frame, action: int) -> None:
            coop.observe(env_id, f, action)
            debuff.observe(env_id, f, action)
        hosts: list[Host]
        if spec is not None:
            hosts = make_hosts(HostSection(real=True, real_config=a.host_config, transport="stdio",
                                          priority="below_normal", max_restarts=0), a.envs, seed=a.seed,
                               log_dir=a.out / "host-logs", name=mode,
                               expected_hashes=(tok.vocab.hash, int(extra["catalog_hash"])))
        elif a.fake_proc:
            command = [sys.executable, "-B", str(FAKE_HOST_SCRIPT), "--no-hashes", "--abort-prob", "0",
                       "--mean-decisions", str(a.fake_mean_decisions), "--catalog-hash", str(extra["catalog_hash"])]
            if a.vocab is not None:
                command.extend(["--vocab", str(a.vocab)])
            else:
                assert extra["vocab"] == "synthetic", "custom fake subprocess vocabulary requires --vocab"
            hosts = make_hosts(HostSection(command=tuple(command), transport="stdio", priority="below_normal",
                                          max_restarts=0), a.envs, seed=a.seed,
                               log_dir=a.out / "host-logs", name=mode,
                               expected_hashes=(tok.vocab.hash, int(extra["catalog_hash"])))
        else:
            cfg = FakeHostConfig(mean_decisions=a.fake_mean_decisions, abort_prob=0, hashes=False)
            hosts = [FakeHost(tok.vocab, a.seed + i, cfg, catalog_hash=int(extra["catalog_hash"]))
                     for i in range(a.envs)]
        try:
            result, path = evaluate(panel, measured, tok, hosts, tag=f"v{version}-{a.weights}-{mode}",
                                    policy_info={"checkpoint": str(a.checkpoint), "mode": mode, "weights": a.weights,
                                                 "temperature": a.temperature, "seed": a.seed},
                                    out_dir=a.out, hosts_info=str(a.host_config or ("stdio fake H5" if a.fake_proc else "in-process fake H5")),
                                    max_batch=a.envs, max_episode_decisions=a.max_episode_decisions,
                                    round_budget_s=a.round_budget_s, on_decision=on_decision)
        finally:
            policy.close()
            for host in hosts:
                host.close()
        report = result.to_dict()
        report["frame_cost"] = measured.report()
        report["spread_by_kind"] = measured.spread_report()
        report["coop"] = coop.report()
        report["debuff"] = debuff.report(result.episodes)
        if a.debuff_episodes:
            report["debuff_episodes"] = debuff.episode_rows  # per-mode report only, not zero-shot.json
        report["focus"] = focus_summary(report["summary"], report["coop"], report["debuff"])
        report["decisions_per_second"] = result.decisions / result.seconds
        for key, summary in report["summary"].items():
            episodes = [e for e in result.episodes if key == "all" or f"{e['lineup']}|A{e['ascension']}" == key]
            summary["decisions_per_game"] = distribution([e["decisions"] for e in episodes])
            summary["floors"] = distribution([e["floor"] for e in episodes])
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports[mode] = {"path": str(path), "summary": report["summary"], "frame_cost": report["frame_cost"],
                         "spread_by_kind": report["spread_by_kind"], "coop": report["coop"],
                         "debuff": report["debuff"], "focus": report["focus"],
                         "decisions_per_second": report["decisions_per_second"], "partial": result.partial}
    (a.out / "zero-shot.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
