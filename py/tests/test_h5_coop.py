from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

from spire.actor import UniformPolicy
from spire.evalpanel import evaluate, make_panel
from spire.h5eval import ALL_ALLIES, ANY_ALLY, MP_ONLY, CoopTelemetry, card_facts, coop_summary, main
from spire.model import PRESETS, SpireNet
from spire.proto import Frame, parse_frame
from spire.schema import (ActionCol, CardCol, CreatureCol, CreatureKind, G, Kind, OptionCol, SeatCol, Table, Verb,
                          ViewMode)
from spire.synth import FakeHost, FakeHostConfig, FrameGen, random_frame
from spire.tokenize import Tokenizer
from spire.vocab import Catalog, Vocab, make_synthetic_catalog

KEY = "DEFECT+DEFECT+IRONCLAD+IRONCLAD|A0"


def _frame(gen: FrameGen, kind: Kind, ok: Callable[[Frame], bool]) -> Frame:
    """The first synthetic 4-seat TEAM frame of ``kind`` (seeds 0, 1, ...) that satisfies ``ok``."""
    for seed in range(500):
        f = parse_frame(random_frame(gen, np.random.default_rng(seed), kind, n_seats=4, view=ViewMode.TEAM))
        if ok(f):
            return f
    raise AssertionError(f"no {kind.name} frame")


def _catalog(vocab: Vocab, flags: dict[int, tuple[str, str]]) -> Catalog:
    """Synthetic catalog whose cards are plain (Self, no constraint) except ``flags``: id -> (target, constraint)."""
    raw = make_synthetic_catalog(vocab, seed=1)
    for k, levels in raw["categories"]["card"].items():
        target, constraint = flags.get(int(k), ("Self", "None"))
        for rec in levels:
            rec["target_type"], rec["multiplayer_constraint"] = target, constraint
    return Catalog(raw, vocab)


def _ally_rows(f: Frame) -> np.ndarray:
    cre, a = f.table(Table.CREATURES), f.actions
    t = a[:, ActionCol.TGT_ROW]
    return np.array([t[i] >= 0 and cre[t[i], CreatureCol.KIND] != CreatureKind.ENEMY
                     and cre[t[i], CreatureCol.SEAT] != a[i, ActionCol.SEAT] for i in range(len(a))])


def _lineup(f: Frame, vocab: Vocab) -> str:
    names = sorted(vocab.name("character", int(c)) for c in f.table(Table.SEATS)[:, SeatCol.CHARACTER_ID])
    return "+".join(names) + f"|A{f.gvals[G.ASCENSION]}"


def test_card_facts_bits(vocab: Vocab) -> None:
    bits = card_facts(_catalog(vocab, {2: ("AnyAlly", "MultiplayerOnly"), 3: ("AllAllies", "None"),
                                       4: ("AnyEnemy", "MultiplayerOnly")}), vocab)
    assert bits.shape[0] == vocab.size("card")
    assert (bits[2] == MP_ONLY | ANY_ALLY).all() and (bits[3] == ALL_ALLIES).all() and (bits[4] == MP_ONLY).all()
    assert not bits[5:].any() and not bits[:2].any()


def test_combat_ally_targets_all_allies_and_mp_only_plays(gen: FrameGen, vocab: Vocab) -> None:
    def ok(f: Frame) -> bool:
        play = f.actions[:, ActionCol.VERB] == Verb.PLAY_CARD
        return bool((play & _ally_rows(f)).any() and (play & ~_ally_rows(f)).any())

    f = _frame(gen, Kind.COMBAT, ok)
    a, ally = f.actions, _ally_rows(f)
    play = a[:, ActionCol.VERB] == Verb.PLAY_CARD
    to_ally, other = int(np.flatnonzero(play & ally)[0]), int(np.flatnonzero(play & ~ally)[0])
    cards = f.table(Table.CARDS)
    ally_card, other_card = (int(cards[a[i, ActionCol.SRC_ROW], CardCol.CARD_ID]) for i in (to_ally, other))
    flags = {other_card: ("AllAllies", "MultiplayerOnly")} if other_card != ally_card else {}
    tel = CoopTelemetry(vocab, _catalog(vocab, flags))
    tel.observe(0, f, to_ally)
    tel.observe(0, f, other)
    end = int(np.flatnonzero(a[:, ActionCol.VERB] == Verb.END_TURN)[0])
    tel.observe(0, f, end)
    rep = tel.report()
    assert set(rep) == {_lineup(f, vocab), "all"}
    r = rep["all"]
    assert r["decisions"] == 3
    assert r["ally_plays"]["card_plays"] == 2
    assert r["ally_plays"]["ally_target"] == {"n": 1, "of": 2, "share": 0.5}
    assert r["ally_plays"]["combat_decisions_with_ally_action"] == {"n": 3, "of": 3, "share": 1.0}
    assert r["ally_plays"]["ally_action_taken_when_legal"]["n"] == 1
    assert r["ally_plays"]["all_allies"]["n"] == r["mp_only_plays"]["plays"]["n"] == (1 if flags else 0)
    assert r["mp_only_plays"]["cards"] == ({vocab.name("card", other_card): 1} if flags else {})
    seats = [int(a[i, ActionCol.SEAT]) for i in (to_ally, other)]
    assert r["interleave"]["turns"] == 1 and r["interleave"]["plays"] == 2
    assert r["interleave"]["switches"] == r["interleave"]["min_switches"] == int(seats[0] != seats[1])
    assert r["turn_end_order"]["first_play_seat"] == {str(seats[0]): 1}


def test_card_reward_pick_rates_by_class(gen: FrameGen, vocab: Vocab) -> None:
    def offered(f: Frame) -> np.ndarray:
        a = f.actions
        return f.table(Table.CARDS)[a[a[:, ActionCol.VERB] == Verb.PICK_OFFER_CARD, ActionCol.SRC_ROW], CardCol.CARD_ID]

    f = _frame(gen, Kind.CARD_REWARD, lambda f: len(set(offered(f).tolist())) == len(offered(f)) >= 2
               and (f.actions[:, ActionCol.VERB] != Verb.PICK_OFFER_CARD).any())
    ids = offered(f).tolist()
    tel = CoopTelemetry(vocab, _catalog(vocab, {ids[0]: ("AnyAlly", "MultiplayerOnly"),
                                                ids[1]: ("AnyEnemy", "MultiplayerOnly")}))
    picks = np.flatnonzero(f.actions[:, ActionCol.VERB] == Verb.PICK_OFFER_CARD)
    tel.observe(0, f, int(picks[0]))
    tel.observe(0, f, int(np.flatnonzero(f.actions[:, ActionCol.VERB] != Verb.PICK_OFFER_CARD)[0]))
    cr = tel.report()["all"]["mp_only_plays"]["card_reward"]
    n = len(ids)
    assert cr["decisions"] == 2 and cr["no_pick"] == 1
    assert cr["mp_only"] == {"offered": 4, "picked": 1, "pick_rate": 0.25,
                             "picked_when_offered": {"n": 1, "of": 2, "share": 0.5}}
    assert cr["ally_target"]["offered"] == 2 and cr["ally_target"]["picked"] == 1
    assert cr["other"] == {"offered": 2 * (n - 2), "picked": 0, "pick_rate": 0.0 if n > 2 else None,
                           "picked_when_offered": {"n": 0, "of": 2 if n > 2 else 0, "share": 0.0 if n > 2 else None}}


def test_rest_mend_choices_and_target_rank(gen: FrameGen, catalog: Catalog, vocab: Vocab) -> None:
    mend_id = vocab.id("rest_option", "MEND")

    def mend_rows(f: Frame) -> np.ndarray:
        a, o = f.actions, f.table(Table.OPTIONS)
        return np.flatnonzero([v == Verb.CHOOSE_OPTION and o[s, OptionCol.KEY_ID] == mend_id
                               for v, s in a[:, [ActionCol.VERB, ActionCol.SRC_ROW]]])

    f = _frame(gen, Kind.REST, lambda f: len(mend_rows(f)) == 3)
    a, seats, rows = f.actions, f.table(Table.SEATS), mend_rows(f)
    targets = a[rows, ActionCol.AUX]
    assert f.gvals[G.ACTING_SEAT] not in targets
    ratio = seats[targets, SeatCol.HP] / seats[targets, SeatCol.MAX_HP]
    lowest, highest = int(rows[np.argmin(ratio)]), int(rows[np.argmax(ratio)])
    other = int(np.flatnonzero(~np.isin(np.arange(len(a)), rows))[0])
    tel = CoopTelemetry(vocab, catalog)
    for action in (lowest, highest, other):
        tel.observe(0, f, action)
    m = tel.report()["all"]["mend"]
    label = vocab.name("rest_option", int(f.table(Table.OPTIONS)[a[other, ActionCol.SRC_ROW], OptionCol.KEY_ID])) \
        if a[other, ActionCol.VERB] == Verb.CHOOSE_OPTION else Verb(int(a[other, ActionCol.VERB])).name
    assert m["rest_decisions"] == 3 and m["mend_offered"] == {"n": 3, "of": 3, "share": 1.0}
    assert m["choices_when_mend_offered"] == m["choices"] == dict(sorted(Counter(["MEND", "MEND", label]).items()))
    assert m["target_lowest_hp"] == {"n": 1, "of": 2, "share": 0.5}
    assert m["target_lowest_hp_chance"] == pytest.approx(1 / 3)
    assert m["target_position"] == dict(sorted(Counter(str(int(np.flatnonzero(rows == r)[0]))
                                                       for r in (lowest, highest)).items()))
    assert m["offered_when_mend_offered"]["MEND"] == 3


def test_turn_interleave_counts() -> None:
    p, e = int(Verb.PLAY_CARD), int(Verb.END_TURN)
    c = Counter[str]()
    tel = CoopTelemetry.__new__(CoopTelemetry)
    tel.counts = {"k": c}
    tel.close_turn("k", [(p, 0), (p, 0), (e, 0), (p, 1), (e, 1), (p, 2), (p, 2), (e, 2), (e, 3)])  # seat blocks
    tel.close_turn("k", [(p, 1), (p, 0), (p, 1), (p, 0), (e, 1), (e, 0)])  # alternating
    tel.close_turn("k", [(p, 3), (p, 3), (e, 3)])  # one seat
    r = coop_summary(c)
    assert r["interleave"] == {"turns": 3, "plays": 5 + 4 + 2, "switches": 2 + 3, "min_switches": 2 + 1,
                               "extra_switches": 2, "switches_per_play": 5 / 11, "min_switches_per_play": 3 / 11,
                               "block_turns": {"n": 1, "of": 2, "share": 0.5}}
    assert r["turn_end_order"]["first_play_seat"] == {"0": 1, "1": 1, "3": 1}
    assert r["turn_end_order"]["ascending"] == {"n": 1, "of": 2, "share": 0.5}
    assert r["turn_end_order"]["end_then_teammate_plays"] == {"n": 2, "of": 7, "share": 2 / 7}


def test_single_seat_frames_are_rejected(gen: FrameGen, catalog: Catalog, vocab: Vocab) -> None:
    f = parse_frame(random_frame(gen, np.random.default_rng(0), Kind.COMBAT))
    with pytest.raises(AssertionError, match="single-seat"):
        CoopTelemetry(vocab, catalog).observe(0, f, 0)


def test_evaluate_feeds_every_decision_to_the_telemetry(vocab: Vocab, catalog: Catalog, tmp_path: Path) -> None:
    panel = make_panel("h5-coop", 3, ["IRONCLAD+DEFECT+IRONCLAD+DEFECT", "DEFECT+DEFECT+DEFECT+DEFECT"], [0],
                       rng_seed=5)
    tel = CoopTelemetry(vocab, catalog)
    verbs: Counter[int] = Counter()

    def observe(env_id: int, f: Frame, action: int) -> None:
        verbs[int(f.actions[action, ActionCol.VERB])] += f.kind == Kind.COMBAT
        tel.observe(env_id, f, action)

    hosts = [FakeHost(vocab, i, FakeHostConfig(mean_decisions=60, length_sigma=0, abort_prob=0, hashes=False))
             for i in range(2)]
    try:
        result, _ = evaluate(panel, UniformPolicy(3), Tokenizer(vocab), hosts, tag="t", policy_info={},
                             out_dir=tmp_path, max_batch=2, on_decision=observe)
    finally:
        for h in hosts:
            h.close()
    rep = tel.report()
    assert set(rep) == {*result.summary} == {KEY, "DEFECT+DEFECT+DEFECT+DEFECT|A0", "all"}
    assert rep["all"]["decisions"] == result.decisions == sum(e["decisions"] for e in result.episodes)
    assert rep["all"]["ally_plays"]["card_plays"] == verbs[Verb.PLAY_CARD] > 0
    assert rep["all"]["interleave"]["plays"] == verbs[Verb.PLAY_CARD]
    assert sum(rep["all"]["turn_end_order"]["first_play_seat"].values()) <= rep["all"]["interleave"]["turns"]


def test_zero_shot_report_has_coop_block(vocab: Vocab, tmp_path: Path) -> None:
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
    assert main(["--checkpoint", str(ck), "--panel", str(tmp_path / "p.json"), "--seeds", "1", "--ascension", "0",
                 "--lineups", "DEFECT+DEFECT+IRONCLAD+IRONCLAD", "--modes", "greedy", "--fake-mean-decisions", "15",
                 "--out", str(out)]) == 0
    coop = json.loads((out / "zero-shot.json").read_text(encoding="utf-8"))["greedy"]["coop"]
    assert set(coop) == {KEY, "all"} and coop["all"]["decisions"] == coop[KEY]["decisions"] > 0
    assert set(coop["all"]) == {"decisions", "ally_plays", "mp_only_plays", "mend", "interleave", "turn_end_order"}
