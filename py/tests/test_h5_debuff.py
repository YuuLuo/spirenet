from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from spire.actor import UniformPolicy
from spire.evalpanel import evaluate, make_panel
from spire.h5eval import (ATTACK, MULTI_ENEMY, VULN, WEAK, CoopTelemetry, DebuffTelemetry, coop_summary, debuff_cards,
                          focus_summary, main)
from spire.model import PRESETS, SpireNet
from spire.proto import Frame
from spire.schema import (ActionCol, CardCol, CreatureCol, CreatureKind, G, IntentCol, Kind, PowerCol, SeatCol, Table,
                          Verb)
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.vocab import Catalog, Vocab, make_synthetic_catalog

CHARS = ["DEFECT", "DEFECT", "IRONCLAD", "IRONCLAD"]
KEY = "DEFECT+DEFECT+IRONCLAD+IRONCLAD|A9"
V_ATTACK, W_SKILL, ATTACK_ONE, ATTACK_ALL, SKILL_SELF = 10, 11, 12, 13, 14
CARDS = {V_ATTACK: ("Attack", "AnyEnemy", "Deal {Damage:diff()} damage. Apply {VulnerablePower:diff()} "
                                          "[gold]Vulnerable[/gold]."),
         W_SKILL: ("Skill", "AnyEnemy", "Apply {WeakPower:diff()} [gold]Weak[/gold]."),
         ATTACK_ONE: ("Attack", "AnyEnemy", "Deal {Damage:diff()} damage. Deals more to [gold]Vulnerable[/gold] "
                                            "enemies."),
         ATTACK_ALL: ("Attack", "AllEnemies", "Deal {Damage:diff()} damage to ALL enemies."),
         SKILL_SELF: ("Skill", "Self", "Gain {Block:diff()} [gold]Block[/gold]. Apply 1 [gold]Weak[/gold] to "
                                       "yourself.")}
NCOLS: dict[int, int] = {Table.SEATS: len(SeatCol), Table.CARDS: len(CardCol), Table.CREATURES: len(CreatureCol),
         Table.POWERS: len(PowerCol), Table.INTENTS: len(IntentCol), Table.ACTIONS: len(ActionCol)}


def _catalog(vocab: Vocab) -> Catalog:
    raw = make_synthetic_catalog(vocab, seed=1)
    for k, levels in raw["categories"]["card"].items():
        kind, target, text = CARDS.get(int(k), ("Skill", "Self", "Gain {Block:diff()} [gold]Block[/gold]."))
        levels[0]["text"] = text
        for rec in levels:
            rec["type"], rec["target_type"] = kind, target
    return Catalog(raw, vocab)


class FakeFrame:
    """The parts of a Frame that the telemetry reads."""

    def __init__(self, episode_id: int, gvals: dict[int, int], tables: dict[int, list[list[int]]],
                 actions: list[list[int]]) -> None:
        g = [0] * 64
        for k, v in gvals.items():
            g[k] = v
        self.episode_id, self.gvals = episode_id, tuple(g)
        self._tables = {t: np.array(rows, np.int32).reshape(len(rows), NCOLS[t]) for t, rows in tables.items()}
        self.actions = np.array(actions, np.int32).reshape(len(actions), len(ActionCol))

    def table(self, t: int) -> np.ndarray:
        return self._tables.get(t, np.zeros((0, NCOLS[t]), np.int32))


class Scene:
    """Hand-made 4-seat frames: seats 0-3 (CHARS) are CREATURES rows 0-3, the enemies follow from row 4."""

    def __init__(self, vocab: Vocab) -> None:
        self.v = vocab

    def row(self, table: int, **cols: int) -> list[int]:
        enums: dict[int, type[IntEnum]] = {Table.SEATS: SeatCol, Table.CARDS: CardCol, Table.CREATURES: CreatureCol,
                                           Table.POWERS: PowerCol, Table.INTENTS: IntentCol}
        r = [0] * NCOLS[table]
        for k, x in cols.items():
            r[enums[table][k.upper()]] = x
        return r

    def frame(self, episode: int, kind: Kind, *, combat: bool, rnd: int = 1, room: str = "Elite", act: int = 0,
              floor: int = 5, won: int = 2, hp: tuple[int, ...] = (50, 50, 60, 60), alive: tuple[int, ...] = (1,) * 4,
              enemies: tuple[tuple[int, int, int], ...] = ((2, 5, 1), (3, 6, 1)),
              powers: tuple[tuple[int, str, int], ...] = (), intents: tuple[tuple[int, str], ...] = (),
              cards: tuple[tuple[int, int, int], ...] = (), actions: tuple[tuple[int, ...], ...] = ()) -> FakeFrame:
        v = self.v
        seats = [self.row(Table.SEATS, seat=s, character_id=v.id("character", ch), hp=hp[s], max_hp=80, alive=alive[s],
                          creature_row=s if combat else -1) for s, ch in enumerate(CHARS)]
        tables: dict[int, list[list[int]]] = {Table.SEATS: seats}
        if combat:
            tables[Table.CREATURES] = [self.row(Table.CREATURES, kind=CreatureKind.PLAYER, seat=s, alive=alive[s])
                                       for s in range(4)] + \
                [self.row(Table.CREATURES, kind=CreatureKind.ENEMY, seat=-1, slot_id=slot, model_id=model, alive=a)
                 for slot, model, a in enemies]
            tables[Table.POWERS] = [self.row(Table.POWERS, owner_row=o, power_id=v.id("power", p), amount=n)
                                    for o, p, n in powers]
            tables[Table.INTENTS] = [self.row(Table.INTENTS, owner_row=o, intent_type=v.id("enum:IntentType", t))
                                     for o, t in intents]
        tables[Table.CARDS] = [self.row(Table.CARDS, seat=s, card_id=cid, upgrade_level=lv) for cid, s, lv in cards]
        g: dict[int, int] = {G.KIND: kind, G.N_SEATS: 4, G.ASCENSION: 9, G.IN_COMBAT: int(combat), G.COMBAT_ROUND: rnd,
             G.ROOM_TYPE: v.id("enum:RoomType", room), G.ACT_INDEX: act, G.TOTAL_FLOOR: floor, G.COMBATS_WON: won}
        acts = [list(a) + [0] * (len(ActionCol) - len(a)) for a in actions]
        return FakeFrame(episode, g, tables, acts)


def play(seat: int, card_row: int, tgt: int = -1) -> tuple[int, ...]:
    return (Verb.PLAY_CARD, seat, Table.CARDS, card_row, tgt, 0)


END = (Verb.END_TURN, 3, -1, -1, -1, 0)
HAND = ((V_ATTACK, 0, 0), (W_SKILL, 2, 0), (ATTACK_ALL, 1, 1), (ATTACK_ONE, 0, 0))


def _win_episode(s: Scene) -> list[FakeFrame]:
    """One elite fight (2 team turns) and a card reward, episode 7; see test_combat_and_reward_counts."""
    att = ((4, "Attack"), (5, "Buff"))
    acts = (play(0, 0, 4), play(2, 1, 5), play(1, 2), play(0, 3, 4), END)
    return [
        s.frame(7, Kind.COMBAT, combat=True, powers=((5, "ARTIFACT_POWER", 1),), intents=att, cards=HAND,
                actions=acts[:1]),
        s.frame(7, Kind.COMBAT, combat=True, powers=((4, "VULNERABLE_POWER", 2), (5, "ARTIFACT_POWER", 1)),
                intents=att, cards=HAND, actions=acts[1:2]),
        s.frame(7, Kind.COMBAT, combat=True, powers=((4, "VULNERABLE_POWER", 2),), intents=att, cards=HAND,
                actions=acts[2:3]),
        s.frame(7, Kind.CARD_SELECT, combat=True, powers=((4, "VULNERABLE_POWER", 2),), intents=att, cards=HAND,
                actions=((Verb.SELECT_CARD, 1, Table.CARDS, 0, -1, 0),)),
        s.frame(7, Kind.COMBAT, combat=True, powers=((4, "VULNERABLE_POWER", 2),), intents=att, cards=HAND,
                actions=(END,)),
        s.frame(7, Kind.COMBAT, combat=True, rnd=2, hp=(40, 50, 60, 60),
                powers=((4, "VULNERABLE_POWER", 1), (5, "WEAK_POWER", 2)), intents=att, cards=HAND,
                actions=((Verb.USE_POTION, 3, Table.POTIONS, 0, 4, 0),)),
        s.frame(7, Kind.COMBAT, combat=True, rnd=2, hp=(40, 50, 60, 60),
                powers=((4, "VULNERABLE_POWER", 1), (4, "WEAK_POWER", 1), (5, "WEAK_POWER", 2)), intents=att,
                cards=HAND, actions=(acts[3],)),
        s.frame(7, Kind.REWARD, combat=False, won=3, hp=(38, 50, 60, 60),
                actions=((Verb.PROCEED, 0, -1, -1, -1, 0),)),
        s.frame(7, Kind.CARD_REWARD, combat=False, won=3, hp=(38, 50, 60, 60),
                cards=((V_ATTACK, 1, 0), (W_SKILL, 1, 0), (ATTACK_ONE, 1, 1)),
                actions=tuple((Verb.PICK_OFFER_CARD, 1, Table.CARDS, r, -1, 0) for r in range(3))
                + ((Verb.SKIP, 1, -1, -1, -1, 0),)),
    ]


def _feed(tel: DebuffTelemetry, env: int, frames: list[FakeFrame], actions: list[int] | None = None) -> None:
    for i, f in enumerate(frames):
        tel.observe(env, f, 0 if actions is None else actions[i])  # type: ignore[arg-type]


def _record(env: int, decisions: int, result: str) -> dict[str, Any]:
    return {"env": env, "decisions": decisions, "result": result, "lineup": "DEFECT+IRONCLAD+DEFECT+IRONCLAD",
            "ascension": 9, "seed": f"SEED{env}", "seed_id": str(env), "floor": 10}


def test_debuff_cards_from_text_and_target(vocab: Vocab) -> None:
    bits = debuff_cards(_catalog(vocab), vocab)
    assert bits.shape[0] == vocab.size("card")
    assert (bits[V_ATTACK] == VULN | ATTACK).all()
    assert (bits[W_SKILL] == WEAK).all()
    assert (bits[ATTACK_ONE] == ATTACK).all()  # mentions Vulnerable without applying it
    assert (bits[ATTACK_ALL] == ATTACK | MULTI_ENEMY).all()
    assert (bits[SKILL_SELF] == 0).all()  # applies Weak, but not to an enemy


def test_debuff_telemetry_needs_the_power_names(vocab: Vocab) -> None:
    raw = json.loads(json.dumps(vocab.raw))
    entries = raw["categories"]["power"]["entries"]
    entries[entries.index("WEAK_POWER")] = "WEAK_POWER_RENAMED"
    v2 = Vocab(raw)
    with pytest.raises(AssertionError, match="lacks"):
        DebuffTelemetry(v2, _catalog(v2))


def test_combat_and_reward_counts(vocab: Vocab) -> None:
    tel = DebuffTelemetry(vocab, _catalog(vocab))
    frames = _win_episode(Scene(vocab))
    _feed(tel, 0, frames, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    rep = tel.report([_record(0, len(frames), "win")])
    assert set(rep) == {KEY, "all"} and set(rep[KEY]) == {"win", "all"}
    r = rep[KEY]["win"]
    assert r == rep["all"]["all"] and r["games"] == 1
    up = r["uptime"]
    assert up["all"] == {"turns": 2, "turns_no_enemy": 0, "any_vulnerable": 0.5, "any_weak": 0.5, "both": 0.5,
                         "enemies": 4, "enemies_vulnerable": 0.25, "enemies_weak": 0.25}
    assert set(up["by_act"]) == {"act1"} and set(up["by_room"]) == {"Elite"}
    assert up["by_act_room"]["act1:Elite"] == up["all"] and set(up["by_act_room"]) == {"act1:Elite"}
    assert r["fights"]["by_act_room"]["act1:Elite"] == r["fights"]["all"]
    (row,) = tel.episode_rows
    assert row["result"] == "win" and row["decisions"] == 9 and row["key"] == KEY
    assert row["counts"]["up|Elite|0|turns"] == 2 and row["counts"]["cmb|Elite|0|hp_lost"] == 12
    assert r["use"]["all"] == {"attack_plays": 3, "hit_vulnerable": {"n": 2, "of": 3, "share": 2 / 3},
                               "targeted_vulnerable": {"n": 1, "of": 2, "share": 0.5},
                               "multi_any_vulnerable": {"n": 1, "of": 1, "share": 1.0}, "no_enemy_target": 0}
    assert r["enemy_attacks"]["all"] == {"enemy_turns": 1, "with_attack": {"n": 1, "of": 1, "share": 1.0},
                                         "attacker_weak": {"n": 0, "of": 1, "share": 0.0}}
    fights = r["fights"]["by_room"]["Elite"]
    assert fights == {"combats": 1, "turns_per_combat": 2.0, "hp_lost_per_seat_combat": 3.0, "deaths_per_combat": 0.0,
                      "turns": 2, "hp_lost": 12, "deaths": 0}
    assert r["hp_lost_by_character"] == {"DEFECT": {"Elite": {"seat_combats": 2, "hp_lost": 12,
                                                              "hp_lost_per_seat_combat": 6.0}},
                                         "IRONCLAD": {"Elite": {"seat_combats": 2, "hp_lost": 0,
                                                                "hp_lost_per_seat_combat": 0.0}}}
    d, i, a = (r["applications"][k] for k in ("DEFECT", "IRONCLAD", "all"))
    assert d["card_plays"] == 2 and d["seat_combats"] == 2 and d["potion_uses"] == 0
    assert d["vulnerable"]["plays_applying"] == 1 and d["vulnerable"]["per_100_plays"] == 50.0
    assert d["vulnerable"]["per_seat_combat"] == 0.5 and d["vulnerable"]["enemies"] == 1
    assert d["vulnerable"]["set_landed"] == {"n": 1, "of": 1, "share": 1.0}
    assert d["weak"]["plays_applying"] == 0 and d["weak"]["set_card_plays"] == 0
    assert i["card_plays"] == 1 and i["artifact_down_plays"] == 1 and i["potion_uses"] == 1
    assert i["weak"]["set_landed"] == {"n": 0, "of": 1, "share": 0.0}
    assert i["weak"]["set_artifact_blocked"] == {"n": 1, "of": 1, "share": 1.0}
    assert i["weak"]["potion_uses_applying"] == 1 and i["weak"]["plays_applying"] == 0
    assert a["card_plays"] == 3 and a["unresolved_plays"] == 1 and a["seat_combats"] == 4
    rw = r["card_rewards"]["DEFECT"]
    assert rw["decisions"] == 1 and rw["no_pick"] == 0
    assert rw["vulnerable"] == {"offered": 1, "picked": 1, "pick_rate": 1.0,
                                "picked_when_offered": {"n": 1, "of": 1, "share": 1.0}}
    assert rw["weak"]["offered"] == 1 and rw["weak"]["picked"] == 0
    assert rw["debuff"]["offered"] == 2 and rw["debuff"]["picked"] == 1
    assert rw["other"] == {"offered": 1, "picked": 0, "pick_rate": 0.0,
                           "picked_when_offered": {"n": 0, "of": 1, "share": 0.0}}
    assert "IRONCLAD" not in r["card_rewards"]


def test_lost_combat_tail_and_unfinished_episode(vocab: Vocab) -> None:
    s = Scene(vocab)
    tel = DebuffTelemetry(vocab, _catalog(vocab))
    boss = s.frame(8, Kind.COMBAT, combat=True, room="Boss", act=2, hp=(10, 0, 5, 20), alive=(1, 0, 1, 1),
                   enemies=((2, 5, 1),), powers=((4, "WEAK_POWER", 1),), intents=((4, "DeathBlow"),), actions=(END,))
    tel.observe(1, boss, 0)  # type: ignore[arg-type]
    tel.observe(1, s.frame(9, Kind.MAP, combat=False, actions=((Verb.TRAVEL, 0, -1, -1, -1, 0),)),  # type: ignore[arg-type]
                0)
    _feed(tel, 0, _win_episode(s))
    rep = tel.report([_record(1, 1, "loss"), _record(0, 9, "win")])
    assert {k: v["games"] for k, v in rep[KEY].items()} == {"win": 1, "loss": 1, "unfinished": 1, "all": 3}
    loss = rep[KEY]["loss"]
    assert loss["fights"]["by_room"]["Boss"] == {"combats": 1, "turns_per_combat": 1.0, "hp_lost_per_seat_combat": 8.75,
                                                 "deaths_per_combat": 3.0, "turns": 1, "hp_lost": 35, "deaths": 3}
    assert set(loss["fights"]["by_act"]) == {"act3"}
    assert loss["enemy_attacks"]["all"]["attacker_weak"] == {"n": 1, "of": 1, "share": 1.0}
    assert rep[KEY]["unfinished"]["uptime"]["all"]["turns"] == 0
    assert rep[KEY]["all"]["fights"]["all"]["combats"] == 2
    rows = {r["result"]: r for r in tel.episode_rows}
    assert set(rows) == {"win", "loss", "unfinished"} and "seed_id" not in rows["unfinished"]
    assert rows["loss"]["counts"]["cmb|Boss|2|hp_lost"] == 35  # the tail counts in a lost episode


def test_report_join_checks_decisions(vocab: Vocab) -> None:
    s = Scene(vocab)
    tel = DebuffTelemetry(vocab, _catalog(vocab))
    _feed(tel, 0, _win_episode(s))
    with pytest.raises(AssertionError, match="vs record"):
        tel.report([_record(0, 8, "win")])
    tel = DebuffTelemetry(vocab, _catalog(vocab))
    _feed(tel, 0, _win_episode(s))
    for episode in (8, 9):
        _feed(tel, 0, [s.frame(episode, Kind.MAP, combat=False, actions=((Verb.TRAVEL, 0, -1, -1, -1, 0),))])
    with pytest.raises(AssertionError, match="without an episode record"):
        tel.report([])


def test_single_seat_frames_are_rejected(vocab: Vocab) -> None:
    f = Scene(vocab).frame(1, Kind.MAP, combat=False)
    f.gvals = tuple(1 if i == G.N_SEATS else x for i, x in enumerate(f.gvals))
    with pytest.raises(AssertionError, match="single-seat"):
        DebuffTelemetry(vocab, _catalog(vocab)).observe(0, f, 0)  # type: ignore[arg-type]


def test_evaluate_feeds_the_debuff_telemetry(vocab: Vocab, catalog: Catalog, tmp_path: Path) -> None:
    panel = make_panel("h5-debuff", 3, ["IRONCLAD+DEFECT+IRONCLAD+DEFECT", "DEFECT+DEFECT+DEFECT+DEFECT"], [0],
                       rng_seed=5)
    tel = DebuffTelemetry(vocab, catalog)
    coop = CoopTelemetry(vocab, catalog)
    kinds: Counter[int] = Counter()

    def observe(env_id: int, f: Frame, action: int) -> None:
        kinds[int(f.gvals[G.IN_COMBAT])] += 1
        coop.observe(env_id, f, action)
        tel.observe(env_id, f, action)

    hosts = [FakeHost(vocab, i, FakeHostConfig(mean_decisions=60, length_sigma=0, abort_prob=0, hashes=False))
             for i in range(2)]
    try:
        result, _ = evaluate(panel, UniformPolicy(3), Tokenizer(vocab), hosts, tag="t", policy_info={},
                             out_dir=tmp_path, max_batch=2, on_decision=observe)
    finally:
        for h in hosts:
            h.close()
    rep = tel.report(result.episodes)
    assert set(rep) == {*result.summary}
    assert rep["all"]["all"]["games"] == len(result.episodes) == 6
    assert sum(v["games"] for k, v in rep["all"].items() if k != "all") == 6
    assert kinds[1] > 0 and rep["all"]["all"]["fights"]["all"]["combats"] > 0


def test_zero_shot_report_has_debuff_block(vocab: Vocab, tmp_path: Path) -> None:
    catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
    tok = Tokenizer(vocab)
    cfg = PRESETS["2.5m"]
    state = SpireNet(tok.layout, cfg, catalog).state_dict()
    ck = tmp_path / "ck.pt"
    torch.save({"model": state, "ema": state, "counters": {"version": 3},
                "extra": {"vocab": "synthetic", "vocab_hash": vocab.hash, "catalog": "synthetic",
                          "catalog_hash": catalog.hash, "tokenizer": asdict(tok.config),
                          "model_config": asdict(cfg)}}, ck)
    out = tmp_path / "out"
    assert main(["--checkpoint", str(ck), "--panel", str(tmp_path / "p.json"), "--seeds", "2", "--ascension", "9",
                 "--lineups", "DEFECT+DEFECT+IRONCLAD+IRONCLAD", "--modes", "greedy", "--fake-mean-decisions", "40",
                 "--debuff-episodes", "--out", str(out)]) == 0
    rep = json.loads((out / "zero-shot.json").read_text(encoding="utf-8"))["greedy"]
    debuff = rep["debuff"]
    assert set(debuff) == {KEY, "all"} and debuff[KEY]["all"]["games"] == rep["summary"][KEY]["n"] == 2
    assert set(debuff[KEY]["all"]) == {"games", "uptime", "use", "enemy_attacks", "fights", "hp_lost_by_character",
                                       "applications", "card_rewards"}
    full = json.loads(Path(rep["path"]).read_text(encoding="utf-8"))
    assert full["debuff"] == debuff
    assert len(full["debuff_episodes"]) == 2 and "debuff_episodes" not in rep
    assert full["focus"] == rep["focus"] == focus_summary(rep["summary"], rep["coop"], debuff)
    assert set(rep["focus"]) == {KEY, "all"} and rep["focus"][KEY]["n"] == 2
    out2 = tmp_path / "out2"  # without --debuff-episodes the per-mode report has no per-episode rows
    assert main(["--checkpoint", str(ck), "--panel", str(tmp_path / "p.json"), "--modes", "greedy",
                 "--fake-mean-decisions", "40", "--out", str(out2)]) == 0
    rep2 = json.loads((out2 / "zero-shot.json").read_text(encoding="utf-8"))["greedy"]
    assert "debuff_episodes" not in json.loads(Path(rep2["path"]).read_text(encoding="utf-8"))
    assert set(rep2["focus"]) == {KEY, "all"}


def test_focus_summary_numbers(vocab: Vocab) -> None:
    tel = DebuffTelemetry(vocab, _catalog(vocab))
    _feed(tel, 0, _win_episode(Scene(vocab)), [0, 0, 0, 0, 0, 0, 0, 0, 0])
    debuff = tel.report([_record(0, 9, "win")])
    mend = Counter({"rest_decisions": 6, "rest_mend_offered": 5, "rest_choice_mend_offered:MEND": 3,
                    "rest_choice_mend_offered:HEAL": 1, "rest_choice_mend_offered:DIG": 1, "rest_choice:MEND": 3,
                    "rest_choice:HEAL": 2, "rest_choice:DIG": 1, "mend_chosen": 3, "mend_target_full_hp": 2,
                    "mend_target_lowest_hp": 1, "mend_candidates": 9, "mend_candidates_lowest_hp": 4})
    coop = {KEY: coop_summary(mend), "all": coop_summary(mend)}
    summary = {KEY: {"n": 1, "wins": 1}, "all": {"n": 1, "wins": 1}}
    f = focus_summary(summary, coop, debuff)
    assert set(f) == {KEY, "all"} and f[KEY] == f["all"]
    assert f[KEY] == {"wins": 1, "n": 1, "boss": None,
                      "elite": {"fights": 1, "attackers": 1, "attackers_weak": 0.0, "attack_plays": 3,
                                "hit_vulnerable": 0.6667, "turns_per_fight": 2.0, "hp_lost_per_player": 3.0,
                                "deaths_per_fight": 0.0},
                      "rest": {"mend_offered": 5, "HEAL": 1, "SMITH": 0, "MEND": 3, "other": 1, "mend_full_hp": 0.6667,
                               "mend_lowest_hp": 0.3333, "mend_lowest_hp_chance": 0.4444}}
    with pytest.raises(KeyError):  # a finished episode always has telemetry
        focus_summary({**summary, "DEFECT+DEFECT+DEFECT+DEFECT|A0": {"n": 1, "wins": 0}}, coop, debuff)
    no_turn = json.loads(json.dumps(debuff))  # an Elite fight without an enemy turn or an attack play
    for block in ("use", "enemy_attacks"):
        del no_turn[KEY]["all"][block]["by_room"]["Elite"]
    e = focus_summary({KEY: summary[KEY]}, coop, no_turn)[KEY]["elite"]
    assert e["attackers"] == e["attack_plays"] == 0 and e["attackers_weak"] is None and e["hit_vulnerable"] is None
    mend["rest_choice_mend_offered:MEND"] = 2  # choices must add up to the MEND offers
    with pytest.raises(AssertionError):
        focus_summary(summary, {KEY: coop_summary(mend), "all": coop["all"]}, debuff)
