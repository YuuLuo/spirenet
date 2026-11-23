"""The removal rule (spire.removal, notes/solo/removal-spec.md section 7): target order, one legal row, the mask in the
model / UniformPolicy / learner, metrics, and off = the tree without the rule (SPIRE_HEAD_TREE)."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy, UniformPolicy
from spire.cycle import CycleBan
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.proto import Frame, parse_frame
from spire.removal import RemovalRule, build_table, is_removal_prompt, load_rule
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore
from spire.schema import ActionCol, CardCol, CardSelectSub, G, Kind, Table, Verb, Zone
from spire.synth import FakeHost, FakeHostConfig, FrameBuilder
from spire.tokenize import CYCLE_BAN_SLOTS, Tokenizer, TokenizerConfig
from spire.vocab import Catalog, Vocab

REF = Path(__file__).with_name("removal_ref.py")
CHARS = ("DEFECT", "FOOL", "IRONCLAD", "NECROBINDER", "REGENT", "SILENT")
CURSE_A, CURSE_B = "CARD_0040", "CARD_0012"  # CURSE_B has the lower card id
LOW, LOW2, HIGH = "CARD_0050", "CARD_0051", "CARD_0060"  # ratings 1, 1, 5


def strike(c: str) -> str:
    return f"CARD_{100 + 2 * CHARS.index(c):04d}"


def defend(c: str) -> str:
    return f"CARD_{101 + 2 * CHARS.index(c):04d}"


def synthetic_table(vocab: Vocab) -> dict:
    names = [n for n in vocab["card"].entries[2:]]
    ratings = {n: 3 for n in names}
    ratings.update({LOW: 1, LOW2: 1, HIGH: 5, CURSE_A: 0, CURSE_B: 0})
    return {"version": 1, "vocab_hash": vocab.hash, "game_version": vocab.game_version, "ratings": ratings,
            "curses": [CURSE_B, CURSE_A], "strike": {c: [strike(c)] for c in CHARS},
            "defend": {c: [defend(c)] for c in CHARS}, "dropped": []}


@pytest.fixture(scope="module")
def table_path(tmp_path_factory: pytest.TempPathFactory, vocab: Vocab) -> str:
    p = tmp_path_factory.mktemp("removal") / "removal-ratings.json"
    p.write_text(json.dumps(synthetic_table(vocab)), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def rule(vocab: Vocab, table_path: str) -> RemovalRule:
    return load_rule(vocab, table_path)


def prompt(vocab: Vocab, cards: list[tuple[str, int]], *, char: str = "SILENT", sel_count: int = 0, sel_max: int = 1,
           confirm: bool = False, cancel: bool = True, sub: CardSelectSub = CardSelectSub.DECK_REMOVAL,
           in_combat: int = 0, step: int = 0, also: tuple[str, ...] = ()) -> Frame:
    """A removal prompt offering ``cards`` ((name, upgrade level), each a SELECT_CARD row) plus CONFIRM / CANCEL;
    ``also``: deck cards the prompt does not offer."""
    b = FrameBuilder()
    b.episode_id, b.step_id = 1, step
    g = b.glob
    g[G.KIND], g[G.KIND_SUB], g[G.N_SEATS], g[G.TOTAL_FLOOR], g[G.IN_COMBAT] = Kind.CARD_SELECT, sub, 1, 3, in_combat
    g[G.SEL_MIN], g[G.SEL_MAX], g[G.SEL_COUNT] = 1, sel_max, sel_count
    b.add(Table.SEATS, seat=0, character_id=vocab.id("character", char), hp=60, max_hp=70, alive=1, creature_row=-1)
    for i, name in enumerate(also):
        b.add(Table.CARDS, zone=Zone.DECK, seat=0, pos=-1, count=1, card_id=vocab.id("card", name))
    for i, (name, up) in enumerate(cards):
        row = b.add(Table.CARDS, zone=Zone.CHOICE, seat=0, pos=i, count=2, card_id=vocab.id("card", name),
                    upgrade_level=up)
        b.add(Table.ACTIONS, verb=Verb.SELECT_CARD, seat=0, src_table=Table.CARDS, src_row=row)
    if confirm:
        b.add(Table.ACTIONS, verb=Verb.CONFIRM_SELECTION, seat=0)
    if cancel:
        b.add(Table.ACTIONS, verb=Verb.CANCEL, seat=0)
    return parse_frame(b.build(hashes=False))


def card_of(frame: Frame, row: int) -> tuple[str, int]:
    a = frame.actions[row]
    assert a[ActionCol.VERB] == Verb.SELECT_CARD
    c = frame.table(Table.CARDS)[a[ActionCol.SRC_ROW]]
    return frame_vocab.name("card", int(c[CardCol.CARD_ID])), int(c[CardCol.UPGRADE_LEVEL])


frame_vocab: Vocab


@pytest.fixture(autouse=True)
def _frame_vocab(vocab: Vocab) -> None:
    global frame_vocab
    frame_vocab = vocab


def target(rule: RemovalRule, frame: Frame) -> tuple[str, int] | str:
    row, cls = rule.choose(frame)
    if cls is None:
        assert frame.actions[row, ActionCol.VERB] == Verb.CONFIRM_SELECTION
        return "CONFIRM"
    return card_of(frame, row)


# ------------------------------------------------------------------ the order of section 2


def test_a_curse_comes_first_lowest_card_id(vocab: Vocab, rule: RemovalRule) -> None:
    f = prompt(vocab, [(strike("SILENT"), 0), (CURSE_A, 0), (LOW, 0), (CURSE_B, 0), (defend("SILENT"), 0)])
    assert target(rule, f) == (CURSE_B, 0)
    assert rule.choose(f)[1] == "curse"


def test_only_strikes_and_defends_strike_first(vocab: Vocab, rule: RemovalRule) -> None:
    f = prompt(vocab, [(defend("SILENT"), 0), (strike("SILENT"), 0)])
    assert target(rule, f) == (strike("SILENT"), 0) and rule.choose(f)[1] == "strike"
    f = prompt(vocab, [(defend("SILENT"), 0), (defend("SILENT"), 1)])
    assert target(rule, f) == (defend("SILENT"), 0) and rule.choose(f)[1] == "defend"


def test_upgraded_and_plain_copies(vocab: Vocab, rule: RemovalRule) -> None:
    s = strike("SILENT")
    assert target(rule, prompt(vocab, [(s, 1), (s, 0), (defend("SILENT"), 0)])) == (s, 0)
    # an upgraded Strike still goes before a plain Defend (class before upgrade level)
    assert target(rule, prompt(vocab, [(defend("SILENT"), 0), (s, 1)])) == (s, 1)
    # rated cards: lowest rating, then not upgraded, then the lowest card id
    assert target(rule, prompt(vocab, [(HIGH, 0), (LOW2, 0), (LOW, 1)])) == (LOW2, 0)
    assert target(rule, prompt(vocab, [(HIGH, 0), (LOW2, 0), (LOW, 0)])) == (LOW, 0)


def test_starter_cards_are_those_of_the_acting_character(vocab: Vocab, rule: RemovalRule) -> None:
    # an Ironclad Strike in a Silent deck is an ordinary rated card (rating 3), behind the Silent Defend
    f = prompt(vocab, [(strike("IRONCLAD"), 0), (defend("SILENT"), 0), (HIGH, 0)])
    assert target(rule, f) == (defend("SILENT"), 0)
    f = prompt(vocab, [(strike("IRONCLAD"), 0), (defend("SILENT"), 0)], char="IRONCLAD")
    assert target(rule, f) == (strike("IRONCLAD"), 0)


def test_a_prompt_asking_for_two_cards(vocab: Vocab, rule: RemovalRule) -> None:
    s = strike("SILENT")
    cards = [(HIGH, 0), (s, 0), (CURSE_A, 0)]
    assert target(rule, prompt(vocab, cards, sel_max=2)) == (CURSE_A, 0)
    # one picked, the second pick follows the same order (the curse row is gone once both copies are taken)
    assert target(rule, prompt(vocab, cards[:2], sel_max=2, sel_count=1, confirm=True)) == (s, 0)
    assert target(rule, prompt(vocab, cards[:2], sel_max=2, sel_count=2, confirm=True)) == "CONFIRM"
    # selection complete although SELECT rows remain: CONFIRM
    assert target(rule, prompt(vocab, cards, sel_max=1, sel_count=1, confirm=True)) == "CONFIRM"


def test_a_prompt_with_only_non_basic_cards_and_unoffered_basics(vocab: Vocab, rule: RemovalRule) -> None:
    f = prompt(vocab, [(HIGH, 0), (LOW2, 1), ("CARD_0070", 0)], also=(strike("SILENT"), CURSE_A))
    assert target(rule, f) == (LOW2, 1) and rule.choose(f)[1] == "other"


def test_no_row_for_the_rule_and_unrated_cards_fail(vocab: Vocab, table_path: str) -> None:
    rule = load_rule(vocab, table_path)
    with pytest.raises(AssertionError, match="no row for the rule"):
        rule.choose(prompt(vocab, [(HIGH, 0)], sel_count=1, sel_max=1))
    t = synthetic_table(vocab)
    del t["ratings"][HIGH]
    with pytest.raises(AssertionError, match="without a rating"):
        RemovalRule(vocab, t).choose(prompt(vocab, [(HIGH, 0)]))
    t = synthetic_table(vocab)
    t["vocab_hash"] += 1
    with pytest.raises(AssertionError, match="another vocabulary"):
        RemovalRule(vocab, t)


def test_scope_out_of_combat_removal_only(vocab: Vocab) -> None:
    assert is_removal_prompt(prompt(vocab, [(HIGH, 0)]))
    assert not is_removal_prompt(prompt(vocab, [(HIGH, 0)], in_combat=1))
    assert not is_removal_prompt(prompt(vocab, [(HIGH, 0)], sub=CardSelectSub.DECK_TRANSFORM))
    assert not is_removal_prompt(prompt(vocab, [(HIGH, 0)], sub=CardSelectSub.DECK_UPGRADE))


# ------------------------------------------------------------------ the mask: CycleBan, model, UniformPolicy


def legal_rows(net: SpireNet, tok: Tokenizer, samples: list) -> list[list[int]]:
    with torch.no_grad():
        out = net(tok.collate(samples))
    return [np.flatnonzero(np.isfinite(row.numpy())).tolist() for row in out["logits"]]


def test_cancel_is_never_legal_one_row_in_model_and_uniform_policy(vocab: Vocab, rule: RemovalRule) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, public_history=True, commit_select=True))
    net = SpireNet(tok.layout, ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16,
                                           d_static=32)).eval()
    cycle = CycleBan(vocab, commit_select=True, removal=rule)
    plain = CycleBan(vocab, commit_select=True)
    s = strike("SILENT")
    frames = [prompt(vocab, [(HIGH, 0), (s, 0), (LOW, 0), (CURSE_A, 1)] * 2, sel_max=2, step=0),
              prompt(vocab, [(HIGH, 0), (s, 0)], sel_max=2, sel_count=1, confirm=True, step=1),
              prompt(vocab, [(HIGH, 0), (s, 0)], sel_max=2, sel_count=2, confirm=True, step=2)]
    samples, rows = [], []
    for f in frames:
        sample = tok.encode(f)
        st = cycle.observe(f, sample)
        assert st.removal_rule and not st.banned and cycle.rule_row is not None
        assert f.actions[cycle.rule_row, ActionCol.VERB] != Verb.CANCEL
        assert sorted((*cycle.active_bans, cycle.rule_row)) == list(range(f.n_actions))
        assert sample.history[-CYCLE_BAN_SLOTS] == -(cycle.rule_row + 1) and not sample.history[-5:].any()
        with pytest.raises(AssertionError, match="banned"):
            cycle.record(f, cycle.active_bans[0])
        rows.append(cycle.rule_row)
        cycle.record(f, cycle.rule_row)
        samples.append(sample)
        # without the rule the removal prompt keeps every row, CANCEL included (commit_select leaves it alone)
        other = tok.encode(f)
        assert not plain.observe(f, other).banned and not other.history[-CYCLE_BAN_SLOTS:].any()
        plain.record(f, 0)
    assert legal_rows(net, tok, samples) == [[r] for r in rows]
    out = UniformPolicy(seed=3).launch(samples, np.asarray([f.n_actions for f in frames]))
    assert out.actions.tolist() == rows and np.allclose(out.logp, 0.0)
    # other prompts are untouched by the rule: an upgrade prompt keeps its commit_select mask (CANCEL only)
    up = prompt(vocab, [(HIGH, 0), (s, 0)], sub=CardSelectSub.DECK_UPGRADE, step=3)
    sample = tok.encode(up)
    st = cycle.observe(up, sample)
    assert st.banned and not st.removal_rule and cycle.rule_row is None
    assert [int(v) - 1 for v in sample.history[-CYCLE_BAN_SLOTS:] if v] == \
        np.flatnonzero(up.actions[:, ActionCol.VERB] == Verb.CANCEL).tolist()


def test_config_keys(table_path: str) -> None:
    TokenizerConfig(cycle_ban=True, removal_rule=True, removal_ratings=table_path)
    with pytest.raises(AssertionError, match="removal_rule iff"):
        TokenizerConfig(cycle_ban=True, removal_rule=True)
    with pytest.raises(AssertionError, match="removal_rule iff"):
        TokenizerConfig(cycle_ban=True, removal_ratings=table_path)
    with pytest.raises(AssertionError, match="cycle-ban mask"):
        TokenizerConfig(removal_rule=True, removal_ratings=table_path)
    assert TokenizerConfig().removal_rule is False and TokenizerConfig().removal_ratings == ""


# ------------------------------------------------------------------ actor + learner on fake hosts


def run_actor(vocab: Vocab, table_path: str, on: bool, decisions: int = 3000) -> tuple[list, list, Learner, dict]:
    torch.manual_seed(0)
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, public_history=True, commit_select=True,
                                           removal_rule=on, removal_ratings=table_path if on else ""))
    net = SpireNet(tok.layout, ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16,
                                           d_static=32))
    expl = ((int(Kind.CARD_REWARD), 0.1), (int(Kind.CARD_SELECT), 0.2))
    learner = Learner(net, tok, PPOConfig(minibatch=512, micro_batch=256, epochs=1, exploration_by_kind=expl),
                      device="cpu", fast=False)
    policy = TorchPolicy(learner.model, tok, seed=11, exploration_by_kind=expl)
    segs: list = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=80, hashes=False, abort_prob=0.0))
             for i in range(4)]
    episodes: list = []
    frames: list[tuple[Frame, int]] = []
    actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0), ("IRONCLAD", 1.0))),
                                                     vocab, 0), ActorConfig(), store=store,
                  on_episode=episodes.append, on_decision=lambda e, f, a: frames.append((f, a)))
    actor.run(max_decisions=decisions)
    return segs, episodes, learner, {"frames": frames}


def test_rule_on_fake_hosts_one_legal_row_learner_reproduces_and_metrics(vocab: Vocab, table_path: str,
                                                                         rule: RemovalRule) -> None:
    segs, episodes, learner, extra = run_actor(vocab, table_path, True)
    removal = [(f, a) for f, a in extra["frames"] if is_removal_prompt(f)]
    assert len(removal) >= 5, len(removal)
    for f, a in removal:
        assert a == rule.choose(f)[0] and f.actions[a, ActionCol.VERB] != Verb.CANCEL
    # the stored rule steps: probability 1 (log-prob 0, one legal row), and the learner's model rebuilds the same
    # single legal row from the stored sample (the history side record carries the mask)
    picked, acts = [], []
    for s in segs:
        for i in range(len(s)):
            if s.samples[i].history[-CYCLE_BAN_SLOTS] < 0:
                assert abs(s.logp[i]) < 1e-6 and s.n_legal[i] == 1 and s.kinds[i] == Kind.CARD_SELECT
                picked.append(s.samples[i])
                acts.append(int(s.actions[i]))
    assert len(picked) == len(removal)
    learner.model.eval()
    assert legal_rows(learner.model, learner.tokenizer, picked) == [[a] for a in acts]
    m = learner.update(segs)
    assert np.isfinite(m["loss_pi"])
    done = [e for e in episodes if e.result in ("win", "loss")]
    assert done and all(set(e.removal) == {"steps", "curse", "strike", "defend", "other"} for e in done)
    assert sum(e.removal["steps"] for e in episodes) >= 1
    assert sum(sum(v for k, v in e.removal.items() if k != "steps") for e in episodes) >= 1
    assert all("removal" in e.as_dict() for e in episodes)


def test_off_records_have_no_removal_key(vocab: Vocab, table_path: str) -> None:
    _, episodes, _, _ = run_actor(vocab, table_path, False, decisions=800)
    assert episodes and all("removal" not in e.as_dict() and e.removal == {} for e in episodes)


# ------------------------------------------------------------------ the real table builder


def test_build_table_asserts_every_deckable_card_is_rated(vocab: Vocab, catalog: Catalog) -> None:
    raw = json.loads(json.dumps(catalog.raw))
    cards = raw["categories"]["card"]
    for i, (k, levels) in enumerate(sorted(cards.items(), key=lambda kv: int(kv[0]))):
        for e in levels:
            e["is_basic_strike_or_defend"] = i < 4
            e["tags"] = ["Strike"] if i in (0, 2) else (["Defend"] if i in (1, 3) else [])
            e["character"] = "SILENT" if i < 2 else "IRONCLAD"
            e["type"] = "Curse" if i == 10 else ("Status" if i == 11 else "Skill")
    cat = Catalog(raw, vocab)
    names = vocab["card"].entries
    deck = [names[int(k)] for k in cards if cards[k][0]["type"] != "Status"]
    ratings = {n: 3 for n in deck}
    t = build_table(vocab, cat, {**ratings, "NOT_A_CARD": 2})
    assert t["dropped"] == ["NOT_A_CARD"] and sorted(t["strike"]) == ["IRONCLAD", "SILENT"]
    assert len(t["curses"]) == 1
    RemovalRule(vocab, t)
    with pytest.raises(AssertionError, match="without a removal rating"):
        build_table(vocab, cat, {n: r for n, r in ratings.items() if n != deck[5]})


# ------------------------------------------------------------------ off = the tree without the rule


def _ref(tree: Path, mode: str, table: str) -> dict:
    env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1",
           "REMOVAL_MODE": mode, "REMOVAL_TABLE": table}
    out = subprocess.run([sys.executable, "-B", str(REF)], env=env, capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_off_is_the_reference_tree(table_path: str) -> None:
    """SPIRE_HEAD_TREE = the tree without the rule: a fixed-seed actor run (production tokenizer flags, fake hosts
    with removal prompts) and one learner update hash identically with the rule off; with it on they differ."""
    head = os.environ.get("SPIRE_HEAD_TREE")
    if not head:
        pytest.skip("SPIRE_HEAD_TREE (the reference py tree) not set")
    mine = Path(__file__).resolve().parents[1]
    ref, off, on = _ref(Path(head), "base", table_path), _ref(mine, "off", table_path), _ref(mine, "on", table_path)
    assert ref["removal_frames"] >= 5 and ref["decisions"] >= 2000
    assert {k: v for k, v in off.items() if k != "mode"} == {k: v for k, v in ref.items() if k != "mode"}
    assert on["steps"] != ref["steps"] and on["rule_one_row"] is True


# ------------------------------------------------------------------ a run resumed with the rule switched on


def test_run_resumes_with_the_rule_on(tmp_path: Path, vocab: Vocab, table_path: str) -> None:
    """A fake-host run trains without the rule, stops, and resumes from its checkpoint with tokenizer.removal_rule on
    (the move of a running run, notes/solo/tree-m.md): versions continue, the start event names the table, train
    rows carry the per-lineup removal metrics and every finished episode its removal record."""
    from spire.metrics import read_jsonl
    from spire.train import Supervisor
    from test_train import small_config
    tok = {"cycle_ban": True, "public_history": True, "commit_select": True}
    over = {"topology": "threads", "hosts": {"envs_per_actor": 4, "fake": {"learnable": True, "mean_decisions": 80,
                                                                           "hashes": False}}}
    assert Supervisor(small_config(tmp_path, "mv", max_updates=2, tokenizer=tok, **over)).run() == 0
    run = tmp_path / "mv"
    n0 = len(list(read_jsonl(run / "episodes.jsonl")))
    cfg = small_config(tmp_path, "mv", max_updates=6, tokenizer={**tok, "removal_rule": True,
                                                                 "removal_ratings": table_path}, **over)
    assert Supervisor(cfg, resume=True).run() == 0
    rows = list(read_jsonl(run / "metrics.jsonl"))
    assert [r["version"] for r in rows if r["kind"] == "learner"] == [1, 2, 3, 4, 5, 6]
    assert any(r["kind"] == "event" and r.get("what") == "removal_rule" and r["file"] == table_path for r in rows)
    train = [r for r in rows if r["kind"] == "train" and "removal" in r]
    assert train and all(set(v) == {"episodes", "rule_steps", "rule_steps_per_episode", "removed",
                                     "removed_per_episode"} for v in train[-1]["removal"].values())
    eps = list(read_jsonl(run / "episodes.jsonl"))
    assert all("removal" not in e for e in eps[:n0])
    later = [e for e in eps[n0:] if e["result"] in ("win", "loss")]
    assert later and all("removal" in e for e in later)
    assert json.loads((run / "config.json").read_text(encoding="utf-8"))["tokenizer"]["removal_rule"] is True
