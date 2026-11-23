"""Schema conformance from an independent encoder (py-review).

Frames are built here straight from ``notes/obs-schema-v1.md`` (sections 3-8, appendix A) with
literal numbers - deliberately **not** through ``spire.synth`` or the ``spire.schema`` enums - and
fed through ``proto`` -> ``rewards`` -> ``tokenize`` -> ``model``. Three decision kinds (COMBAT,
MAP, EVENT) plus a TERMINAL frame.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from spire import schema
from spire.model import ModelConfig, SpireNet
from spire.proto import check_frame, encode_reset, encode_step, parse_frame, ResetCmd
from spire.rewards import RewardConfig, RoomIds, episode_targets, frame_facts
from spire.tokenize import Ent, FeatureDecoder, Tokenizer
from spire.vocab import Vocab

# --------------------------------------------------------------------------- the document, literally

# schema 3.3 / 5.x: column names in column order
DOC_COLUMNS = {
    0: "seat character_id creature_row private_visible alive hp max_hp block gold potion_slots energy max_energy "
       "stars orb_capacity hand_count draw_count discard_count exhaust_count deck_count ready_to_end_turn "
       "turn_number arrow_slots arrows_available melody_bits cards_played_turn attacks_played_turn "
       "skills_played_turn powers_played_turn cards_drawn_turn energy_spent_turn stars_gained_turn "
       "hp_loss_events_combat",
    1: "zone seat pos count card_id upgrade_level energy_cost cost_flags energy_cost_color star_cost "
       "star_cost_color keyword_bits enchant_id enchant_amount enchant_status affliction_id affliction_amount "
       "replay_count hand_flags unplayable_reason var_off var_cnt link_row aux sel_count",
    2: "owner_table owner_row target_row var_id shown base enchanted flags",
    3: "kind seat pos model_id hp max_hp hp_display block alive hittable slot_id power_off power_cnt intent_off "
       "intent_cnt poison_next_turn",
    4: "owner_row power_id amount stack_type type_now applier_row visible_mask var_off var_cnt order",
    5: "owner_row idx intent_type hits card_count dmg_seat0 dmg_seat1 dmg_seat2 dmg_seat3 flags",
    6: "seat zone pos relic_id counter status flags var_off var_cnt",
    7: "seat zone slot potion_id flags var_off var_cnt",
    8: "seat pos orb_id passive evoke",
    9: "seat slot status",
    10: "row col point_type state resolved_room flags",
    11: "from_node_row to_node_row",
    12: "kind seat pos key_id flags price amount ref_table ref_row group var_off var_cnt ref_off ref_cnt",
    13: "owner_table owner_row category vocab_id ent_table ent_row",
    14: "x y hidden item_type item_good item_idx",
    15: "verb seat src_table src_row tgt_row aux flags",
}
NCOL = {t: len(c.split()) for t, c in DOC_COLUMNS.items()}
COL = {t: {n: i for i, n in enumerate(c.split())} for t, c in DOC_COLUMNS.items()}
SEATS, CARDS, VARS, CREATURES, POWERS, INTENTS, RELICS, POTIONS = range(8)
MAP_NODES, MAP_EDGES, OPTIONS, REFS, ACTIONS = 10, 11, 12, 13, 15
MAGIC, CMD = 0x424F5053, 0x4D435053
FNV_BASIS, FNV_PRIME = 2166136261, 16777619
# appendix A
K_MAP, K_COMBAT, K_EVENT, K_TERMINAL = 1, 2, 6, 13
Z_HAND, Z_DRAW, Z_DECK, Z_OFFER = 1, 2, 6, 8
CK_PLAYER, CK_ENEMY = 1, 3
OK_EVENT_OPTION = 1
V_END_TURN, V_PLAY_CARD, V_USE_POTION, V_DISCARD_POTION, V_TRAVEL, V_CHOOSE_OPTION = 1, 3, 4, 5, 6, 11
REF_CARD = 1  # provisional numbering (stage-1 schema issue 1)


def _fnv(b: bytes) -> int:
    h = FNV_BASIS
    for x in b:
        h = ((h ^ x) * FNV_PRIME) & 0xFFFFFFFF
    return h


def _i32(u: int) -> int:
    return u - (1 << 32) if u >= 1 << 31 else u


def doc_encode(msg_type: int, glob: dict[int, int], tables: dict[int, list[list[int]]], *, vocab_hash: int,
               flags: int = 0, result: int = 0, step_id: int = 0) -> bytes:
    """Schema 3.2: header 16 | global 64 | directory 16x5 | table data in table-id order."""
    g = [0] * 64
    for k, v in glob.items():
        g[k] = v
    data = b""
    directory: list[int] = []
    off = 640
    for t in range(16):
        rows = tables.get(t, [])
        assert all(len(r) == NCOL[t] for r in rows), t
        blob = struct.pack(f"<{len(rows) * NCOL[t]}i", *[x for r in rows for x in r])
        directory += [t, len(rows), NCOL[t], off, _i32(_fnv(blob))]
        data += blob
        off += len(blob)
    n_act = len(tables.get(ACTIONS, []))
    head = [MAGIC, 1, msg_type, vocab_hash, 0, 7, step_id, off, 16, 64, flags, result, n_act, 0, 0, 0]
    return struct.pack("<16i64i80i", *head, *g, *directory) + data


def row(t: int, **vals: int) -> list[int]:
    """A row of table ``t``: named columns from ``vals``; other columns -1 for pointers/offsets, 0 else."""
    r = [0] * NCOL[t]
    for n, i in COL[t].items():
        if n.endswith(("_row", "_off")) and n not in vals:
            r[i] = -1
    for n, v in vals.items():
        r[COL[t][n]] = v
    return r


# --------------------------------------------------------------------------- frames


@pytest.fixture(scope="module")
def ids(vocab: Vocab) -> dict[str, int]:
    v = vocab
    return {"ironclad": v.id("character", "IRONCLAD"), "elite": v.id("enum:RoomType", "Elite"),
            "event_room": v.id("enum:RoomType", "Event"), "monster_room": v.id("enum:RoomType", "Monster"),
            "attack": v.id("enum:IntentType", "Attack"), "buff": v.id("enum:IntentType", "Buff"),
            "counter": v.id("enum:PowerStackType", "Counter"), "debuff": v.id("enum:PowerType", "Debuff"),
            "monster_pt": v.id("enum:MapPointType", "Monster")}


def combat_frame(ids: dict[str, int], vh: int, *, hp: int = 60, floor: int = 3) -> bytes:
    seats = [row(SEATS, seat=0, character_id=ids["ironclad"], creature_row=0, private_visible=1, alive=1, hp=hp,
                 max_hp=80, gold=99, potion_slots=3, energy=3, max_energy=3, hand_count=2, draw_count=3,
                 deck_count=12, turn_number=1)]
    # HAND rows keep positions; DRAW is a multiset sorted by content key (card_id differs -> ascending)
    cards = [row(CARDS, zone=Z_HAND, seat=0, pos=0, count=1, card_id=10, energy_cost=1, hand_flags=1, var_off=0,
                 var_cnt=2),
             row(CARDS, zone=Z_HAND, seat=0, pos=1, count=1, card_id=20, energy_cost=1, hand_flags=1, var_off=2,
                 var_cnt=1),
             row(CARDS, zone=Z_DRAW, seat=0, pos=-1, count=2, card_id=10, energy_cost=1),
             row(CARDS, zone=Z_DRAW, seat=0, pos=-1, count=1, card_id=30, energy_cost=2)]
    # VARS grouped by owner (table, row), inside by (target_row, var_id), -1 first
    vars_ = [row(VARS, owner_table=CARDS, owner_row=0, target_row=-1, var_id=5, shown=6, base=6, enchanted=6),
             row(VARS, owner_table=CARDS, owner_row=0, target_row=1, var_id=5, shown=9, base=6, enchanted=6),
             row(VARS, owner_table=CARDS, owner_row=1, target_row=-1, var_id=7, shown=5, base=5, enchanted=5),
             row(VARS, owner_table=POTIONS, owner_row=0, target_row=-1, var_id=9, shown=20, base=20, enchanted=20)]
    creatures = [row(CREATURES, kind=CK_PLAYER, seat=0, pos=0, model_id=ids["ironclad"], hp=hp, max_hp=80, alive=1,
                     hittable=1, power_off=0, power_cnt=1),
                 row(CREATURES, kind=CK_ENEMY, seat=-1, pos=0, model_id=5, hp=40, max_hp=44, alive=1, hittable=1,
                     power_off=1, power_cnt=1, intent_off=0, intent_cnt=1),
                 row(CREATURES, kind=CK_ENEMY, seat=-1, pos=1, model_id=6, hp=12, max_hp=12, alive=1, hittable=1,
                     intent_off=1, intent_cnt=1)]
    powers = [row(POWERS, owner_row=0, power_id=3, amount=2, stack_type=ids["counter"], visible_mask=1, order=0),
              row(POWERS, owner_row=1, power_id=4, amount=1, stack_type=ids["counter"], type_now=ids["debuff"],
                  applier_row=0, visible_mask=1, order=0)]
    intents = [row(INTENTS, owner_row=1, idx=0, intent_type=ids["attack"], hits=2, dmg_seat0=6, dmg_seat1=-1,
                   dmg_seat2=-1, dmg_seat3=-1, flags=1),
               row(INTENTS, owner_row=2, idx=0, intent_type=ids["buff"], dmg_seat0=-1, dmg_seat1=-1, dmg_seat2=-1,
                   dmg_seat3=-1, flags=1)]
    relics = [row(RELICS, seat=0, zone=1, pos=0, relic_id=8, counter=-1)]
    potions = [row(POTIONS, seat=0, zone=1, slot=0, potion_id=11, flags=3, var_off=3, var_cnt=1)]
    # ACTIONS in canonical order (verb, seat, src_table, src_row, tgt_row, aux)
    actions = [row(ACTIONS, verb=V_END_TURN, seat=0, src_table=-1, src_row=-1, tgt_row=-1),
               row(ACTIONS, verb=V_PLAY_CARD, seat=0, src_table=CARDS, src_row=0, tgt_row=1),
               row(ACTIONS, verb=V_PLAY_CARD, seat=0, src_table=CARDS, src_row=0, tgt_row=2),
               row(ACTIONS, verb=V_PLAY_CARD, seat=0, src_table=CARDS, src_row=1, tgt_row=-1),
               row(ACTIONS, verb=V_USE_POTION, seat=0, src_table=POTIONS, src_row=0, tgt_row=-1),
               row(ACTIONS, verb=V_DISCARD_POTION, seat=0, src_table=POTIONS, src_row=0, tgt_row=-1)]
    glob = {0: K_COMBAT, 2: 1, 3: 0, 4: -1, 5: 0, 8: 0, 10: floor, 11: floor, 12: 17, 13: 2, 14: 3,
            15: ids["elite"], 19: 1, 20: 1, 43: 1}
    return doc_encode(1, glob, {SEATS: seats, CARDS: cards, VARS: vars_, CREATURES: creatures, POWERS: powers,
                                INTENTS: intents, RELICS: relics, POTIONS: potions, ACTIONS: actions},
                      vocab_hash=vh)


def map_frame(ids: dict[str, int], vh: int, *, floor: int = 2) -> bytes:
    seats = [row(SEATS, seat=0, character_id=ids["ironclad"], creature_row=-1, private_visible=1, alive=1, hp=70,
                 max_hp=80, energy=-1, max_energy=3, deck_count=3)]
    deck = [row(CARDS, zone=Z_DECK, seat=0, pos=-1, count=2, card_id=10, energy_cost=1),
            row(CARDS, zone=Z_DECK, seat=0, pos=-1, count=1, card_id=20, energy_cost=1)]
    # nodes by (row, col), boss last; state 3 = current, 1 = reachable
    nodes = [row(MAP_NODES, row=1, col=2, point_type=ids["monster_pt"], state=3),
             row(MAP_NODES, row=2, col=1, point_type=ids["monster_pt"], state=1),
             row(MAP_NODES, row=2, col=3, point_type=ids["monster_pt"], state=1),
             row(MAP_NODES, row=3, col=2, point_type=ids["monster_pt"], state=0),
             row(MAP_NODES, row=4, col=3, point_type=ids["monster_pt"], state=0, flags=2)]
    edges = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]]
    actions = [row(ACTIONS, verb=V_TRAVEL, seat=0, src_table=MAP_NODES, src_row=1, tgt_row=-1),
               row(ACTIONS, verb=V_TRAVEL, seat=0, src_table=MAP_NODES, src_row=2, tgt_row=-1)]
    glob = {0: K_MAP, 2: 1, 4: -1, 5: 0, 10: floor, 11: floor, 12: 5, 13: 1, 14: 2, 15: ids["monster_room"], 43: 1}
    return doc_encode(1, glob, {SEATS: seats, CARDS: deck, MAP_NODES: nodes, MAP_EDGES: edges, ACTIONS: actions},
                      vocab_hash=vh, flags=1 << 3)


def event_frame(ids: dict[str, int], vh: int, *, floor: int = 4, history: dict[int, int] | None = None) -> bytes:
    seats = [row(SEATS, seat=0, character_id=ids["ironclad"], creature_row=-1, private_visible=1, alive=1, hp=50,
                 max_hp=80, energy=-1, max_energy=3, deck_count=1)]
    cards = [row(CARDS, zone=Z_DECK, seat=0, pos=-1, count=1, card_id=10, energy_cost=1),
             row(CARDS, zone=Z_OFFER, seat=0, pos=0, count=1, card_id=40, energy_cost=0)]
    opts = [row(OPTIONS, kind=OK_EVENT_OPTION, seat=0, pos=0, key_id=3, flags=0, price=-1, amount=-1, ref_table=-1,
                group=-1, var_off=0, var_cnt=1, ref_off=0, ref_cnt=1),
            row(OPTIONS, kind=OK_EVENT_OPTION, seat=0, pos=1, key_id=4, flags=2, price=-1, amount=-1, ref_table=-1,
                group=-1)]
    vars_ = [row(VARS, owner_table=OPTIONS, owner_row=0, target_row=-1, var_id=12, shown=8, base=8, enchanted=8,
                 flags=2)]
    refs = [row(REFS, owner_table=OPTIONS, owner_row=0, category=REF_CARD, vocab_id=40, ent_table=CARDS, ent_row=1)]
    actions = [row(ACTIONS, verb=V_CHOOSE_OPTION, seat=0, src_table=OPTIONS, src_row=0, tgt_row=-1),
               row(ACTIONS, verb=V_CHOOSE_OPTION, seat=0, src_table=OPTIONS, src_row=1, tgt_row=-1)]
    glob = {0: K_EVENT, 2: 1, 4: -1, 5: 0, 10: floor, 11: floor, 15: ids["event_room"], 31: 5, 32: 6, 43: 1, 48: -1,
            **(history or {})}
    return doc_encode(1, glob, {SEATS: seats, CARDS: cards, VARS: vars_, OPTIONS: opts, REFS: refs,
                                ACTIONS: actions}, vocab_hash=vh)


def terminal_frame(ids: dict[str, int], vh: int, *, result: int, floor: int, hp: int,
                   history: dict[int, int] | None = None) -> bytes:
    seats = [row(SEATS, seat=0, character_id=ids["ironclad"], private_visible=1, alive=int(hp > 0), hp=hp,
                 max_hp=80, energy=-1)]
    return doc_encode(2, {0: K_TERMINAL, 2: 1, 4: -1, 10: floor, 11: floor, 48: -1, **(history or {})}, {SEATS: seats},
                      vocab_hash=vh, flags=1, result=result)


# --------------------------------------------------------------------------- tests


def test_constants_match_the_document() -> None:
    for t, names in DOC_COLUMNS.items():
        assert schema.N_COLS[t] == NCOL[t]
    enum_of = {0: schema.SeatCol, 1: schema.CardCol, 2: schema.VarCol, 3: schema.CreatureCol, 4: schema.PowerCol,
               5: schema.IntentCol, 6: schema.RelicCol, 7: schema.PotionCol, 8: schema.OrbCol, 9: schema.ArrowCol,
               10: schema.MapNodeCol, 11: schema.MapEdgeCol, 12: schema.OptionCol, 13: schema.RefCol,
               14: schema.CellCol, 15: schema.ActionCol}
    for t, e in enum_of.items():
        assert [m.name.lower() for m in e] == DOC_COLUMNS[t].split(), t
    g = schema.G
    assert (g.KIND, g.ACTING_SEAT, g.ROOM_TYPE, g.IN_COMBAT, g.TOTAL_FLOOR, g.PROMPT_KEY, g.SEL_FLAGS,
            g.EVENT_ID, g.PLAYERS_ALIVE) == (0, 5, 15, 19, 11, 24, 30, 31, 43)
    assert [int(v) for v in schema.Verb][1:] == list(range(1, 16))
    assert schema.Verb.CRYSTAL_CLICK == 15 and schema.Verb.PICK_OFFER_CARD == 12
    assert (schema.Kind.TERMINAL, schema.Zone.PREVIEW, schema.OptionKind.CONFIRM_CHOICE) == (13, 9, 8)
    assert (schema.FRAME_MAGIC, schema.CMD_MAGIC) == (MAGIC, CMD)
    assert (schema.H.FLAGS, schema.H.RESULT, schema.H.N_ACTIONS, schema.H.AUTO_STEPS) == (10, 11, 12, 13)
    # errata-v1 section 5: GLOBAL 44..51, TerminalReason values
    assert [int(x) for x in (g.COMBATS_WON, g.ELITES_WON, g.BOSSES_WON, g.LAST_WON_ROOM_TYPE, g.LAST_WON_TOTAL_FLOOR,
                             g.TERMINAL_REASON, g.ABORT_REASON, g.POLICY_DECISIONS)] == list(range(44, 52))
    assert [int(r) for r in schema.TerminalReason] == list(range(7))
    assert schema.GLOBAL_VOCAB[g.LAST_WON_ROOM_TYPE] == "enum:RoomType"


def test_commands_match_the_document() -> None:
    # 7.1 STEP = [len][SPCM, 2, step_id, action_index]
    assert encode_step(5, 3) == struct.pack("<I4i", 16, CMD, 2, 5, 3)
    # 7.1 RESET = SPCM, 1, n_seats, view_mode, view_seat, ascension, char x4, seed_present, lo, hi, flags, 0, 0
    seed = (7 << 32) | 9
    msg = encode_reset(ResetCmd((4, 4), 3, seed, flags=0b10001))
    assert msg == struct.pack("<I11i2I3i", 64, CMD, 1, 2, 0, -1, 3, 4, 4, 0, 0, 1, 9, 7, 0b10001, 0, 0)
    # errata-v1 P-2: word 14 = the episode's policy-decision cap
    msg = encode_reset(ResetCmd((4,), 0, None, flags=25, max_decisions=5000))
    assert msg == struct.pack("<I11i2I3i", 64, CMD, 1, 1, 0, -1, 0, 4, 0, 0, 0, 0, 0, 0, 25, 5000, 0)


@pytest.mark.parametrize("which", ["combat", "map", "event"])
def test_doc_frames_parse_check_and_tokenize(which: str, vocab: Vocab, ids: dict[str, int]) -> None:
    payload = {"combat": combat_frame, "map": map_frame, "event": event_frame}[which](ids, vocab.hash)
    f = parse_frame(payload, verify_hashes=True)  # directory hashes computed independently
    check_frame(f)  # schema 14.5 deep checks accept the document's canonical orders
    assert f.is_obs and f.n_actions == len(f.actions)
    tok = Tokenizer(vocab)
    s = tok.encode(f)
    n = {t: f.table(t).shape[0] for t in range(16)}
    tokens = 2 + sum(n[t] for t in (SEATS, CARDS, CREATURES, RELICS, POTIONS, 8, 9, MAP_NODES, OPTIONS, 14))
    assert s.counts == (tokens, n[POWERS] + n[INTENTS] + n[REFS], n[VARS], n[ACTIONS])
    assert s.edges.shape[0] == 2 * n[MAP_EDGES]
    # semantic ids: decoded embedding rows land in the vocab region named by the document's column
    L = tok.layout
    dec = FeatureDecoder(L)
    id_rows, _ = dec(torch.from_numpy(s.types.astype(np.int64)), torch.from_numpy(s.raw))
    id_rows = id_rows.numpy()
    card_tok = 2 + n[SEATS]
    for i, card in enumerate(f.table(CARDS)):
        assert L.regions["card"][0] + card[4] in id_rows[card_tok + i]
    act0 = sum(s.counts[:3])
    verbs = [r[0] for r in f.actions]
    for j, v in enumerate(verbs):
        assert L.regions["p:verb"][0] + v in id_rows[act0 + j]
    # action -> source / target token relations follow (src_table, src_row) / tgt_row
    base = {SEATS: 2}
    k = 2
    for t in (SEATS, CARDS, CREATURES, RELICS, POTIONS, 8, 9, MAP_NODES, OPTIONS, 14):
        base[t] = k
        k += n[t]
    for j, a in enumerate(f.actions):
        src_t, src_r, tgt = int(a[2]), int(a[3]), int(a[4])
        assert s.rel[act0 + j, 0] == (base[src_t] + src_r if src_t >= 0 else -1)
        assert s.rel[act0 + j, 1] == (base[CREATURES] + tgt if tgt >= 0 else -1)
        assert s.rel[act0 + j, 2] == base[SEATS] + int(a[1])
    if which == "combat":
        # per-target VARS row (CARDS row 0, target creature 1) feeds the PLAY_CARD(card 0 -> creature 1) action
        var0 = sum(s.counts[:2])
        tv = [i for i, r in enumerate(f.table(VARS)) if r[2] >= 0]
        assert [s.rel[var0 + i, 0] for i in tv] == [act0 + 1]
        # untargeted VARS pool into their owner token (card 0, card 1, potion 0)
        assert s.rel[var0 + 0, 0] == base[CARDS] + 0 and s.rel[var0 + 2, 0] == base[CARDS] + 1
        assert s.rel[var0 + 3, 0] == base[POTIONS] + 0
        # POWERS / INTENTS pool into their owner creature
        kid0 = s.counts[0]
        assert list(s.rel[kid0:kid0 + 4, 0]) == [base[CREATURES], base[CREATURES] + 1, base[CREATURES] + 1,
                                                 base[CREATURES] + 2]
    if which == "event":
        kid0 = s.counts[0]
        assert s.rel[kid0, 0] == base[OPTIONS] and s.rel[kid0, 1] == base[CARDS] + 1  # REFS owner / entity


def test_doc_frames_through_the_model(vocab: Vocab, ids: dict[str, int]) -> None:
    tok = Tokenizer(vocab)
    frames = [parse_frame(fn(ids, vocab.hash)) for fn in (combat_frame, map_frame, event_frame)]
    samples = [tok.encode(f) for f in frames]
    torch.manual_seed(0)
    net = SpireNet(tok.layout, ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, d_id=16, d_cond=16,
                                           d_static=16)).eval()
    batch = tok.collate(samples)
    with torch.no_grad():
        out = net(batch)
    A = batch["act_mask"].shape[1]
    for i, f in enumerate(frames):
        lg = out["logits"][i]
        assert torch.isfinite(lg[: f.n_actions]).all() and torch.isinf(lg[f.n_actions:]).all()
    # logits column j of sample i is entity row act_base + j: dest of the action rows is in order
    dest = batch["dest"].numpy()
    BT = batch["tok_mask"].numel()
    a0 = BT + batch["kid_mask"].shape[0] + batch["var_mask"].shape[0]
    r0 = 0
    for i, s in enumerate(samples):
        rows = dest[r0 + sum(s.counts[:3]): r0 + sum(s.counts)]
        assert list(rows) == [a0 + i * A + j for j in range(s.n_actions)]
        r0 += sum(s.counts)
    assert torch.isfinite(out["value"]).all()


def test_doc_episode_rewards(vocab: Vocab, ids: dict[str, int]) -> None:
    """MAP(f2) -> elite COMBAT(f3) x2 -> EVENT(f4) -> TERMINAL LOSS: rewards by hand (schema-level fields).

    The elite kill is read from the host's counters (errata-v1 section 5: GLOBAL 44 combats_won, 45 elites_won,
    47 last_won_room_type, 48 last_won_total_floor), the death from TERMINAL_REASON 2 (GLOBAL 49)."""
    vh = vocab.hash
    won = {44: 1, 45: 1, 47: ids["elite"], 48: 3}
    frames = [map_frame(ids, vh, floor=2), combat_frame(ids, vh, hp=60, floor=3), combat_frame(ids, vh, hp=52, floor=3),
              event_frame(ids, vh, floor=4, history=won),
              terminal_frame(ids, vh, result=2, floor=4, hp=0, history={**won, 49: 2, 51: 4})]
    facts = [frame_facts(parse_frame(p)) for p in frames]
    cfg = RewardConfig(gamma=0.99, hp_coef=0.1)
    t = episode_targets(facts, RoomIds.from_vocab(vocab), cfg)
    phi = [0.1 * 70 / 80, 0.1 * 60 / 80, 0.1 * 52 / 80, 0.1 * 50 / 80, 0.0]
    hp = [cfg.gamma * phi[i + 1] - phi[i] for i in range(4)]
    want = [0.01 + hp[0], hp[1], 0.01 + 0.02 + hp[2], -0.1 + hp[3]]
    np.testing.assert_allclose(t.rewards, want, rtol=0, atol=1e-6)
    assert list(t.combat_id) == [-1, 0, 0, -1]
    np.testing.assert_allclose(t.hp_post[1:3], [50 / 80] * 2, atol=1e-6)
    assert t.win == 0.0 and t.floors_left is not None and list(t.floors_left) == [2, 1, 1, 0]
