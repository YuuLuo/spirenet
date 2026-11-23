"""Rules out 1.2 (spire.rules, notes/next/p4.md): teach the removal rule, then lift it. R1 - R6."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from spire.actor import RULES_KEYS, Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy, UniformPolicy
from spire.cycle import CycleBan
from spire.kickstart import load_teacher
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet, packed_layout, packed_views
from spire.proto import Command, Frame, QuitCmd, ResetCmd, StepCmd, parse_frame
from spire.removal import load_rule
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.rules import RulesConfig, coefficient, free_legal, mask_legal, removal_terms
from spire.schema import ActionCol, CardCol, CardSelectSub, FrameFlag, G, Kind, MsgType, OptionKind, Result, Table, TerminalReason, Verb,     Zone
from spire.synth import FakeGame, FakeHost, FakeHostConfig, FrameBuilder
from spire.tokenize import CYCLE_BAN_SLOTS, Sample, SampleBlock, Tokenizer, TokenizerConfig
from spire.train import Supervisor, TrainConfig, build, load_config
from spire.vocab import Vocab

from test_cycle import menu, rows, select
from test_cycle_integration import _host
from test_removal import HIGH, LOW, prompt, synthetic_table

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16)
MODES = ("mask", "teach", "free")
HERE = Path(__file__).resolve().parent
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")


@pytest.fixture(scope="module")
def table_path(tmp_path_factory: pytest.TempPathFactory, vocab: Vocab) -> str:
    p = tmp_path_factory.mktemp("rules") / "removal-ratings.json"
    p.write_text(json.dumps(synthetic_table(vocab)), encoding="utf-8")
    return str(p)


def tok_of(vocab: Vocab, mode: str, table: str, **kw: Any) -> Tokenizer:
    return Tokenizer(vocab, TokenizerConfig(cycle_ban=True, commit_select=True, removal_rule=True,
                                            removal_ratings=table, rules_mode=mode, **kw))


def slots(sample: Sample) -> list[int]:
    return [int(v) for v in sample.history[-CYCLE_BAN_SLOTS:]]


def removal_frame(vocab: Vocab, step: int = 0) -> Frame:
    """Two cards (the rule removes LOW), CONFIRM (fewer than allowed) and CANCEL."""
    return prompt(vocab, [(HIGH, 0), (LOW, 0)], confirm=True, step=step)


# =========================================================================== R3: the slot cases in the three modes


def _case(name: str, vocab: Vocab, rule: Any) -> tuple[list[tuple[Frame, int]], Frame]:
    """(decisions to observe + record, the frame observed last)."""
    if name == "removal":
        return [], removal_frame(vocab)
    if name == "commit":  # an upgrade prompt: CANCEL first, then a cycle row
        verbs = (Verb.PROCEED, Verb.PROCEED, Verb.CANCEL)
        f = select(CardSelectSub.DECK_UPGRADE, verbs)
        return [(f, rows(f, Verb.PROCEED)[0])], select(CardSelectSub.DECK_UPGRADE, verbs, step=1)
    if name == "cancel_six":  # CANCEL first, then 5 of the 6 cycle rows (capacity: six slots)
        verbs = (Verb.PROCEED,) * 7 + (Verb.CANCEL,)
        f0 = select(CardSelectSub.DECK_UPGRADE, verbs)
        proceed = rows(f0, Verb.PROCEED)
        return ([(select(CardSelectSub.DECK_UPGRADE, verbs, step=s), proceed[s]) for s in range(6)],
                select(CardSelectSub.DECK_UPGRADE, verbs, step=6))
    if name == "six":
        return [(menu(actions=8, step=s), s) for s in range(6)], menu(actions=8, step=6)
    if name == "all_banned":
        return [(menu(actions=2, step=s), s) for s in range(2)], menu(actions=2, step=2)
    raise AssertionError(name)


CASES = ("removal", "commit", "cancel_six", "six", "all_banned")


def _observe(name: str, vocab: Vocab, table: str, mode: str, fork_at: int | None = None
             ) -> tuple[CycleBan, Sample, Any]:
    tok = tok_of(vocab, mode, table)
    rule = load_rule(vocab, table)
    cycle = CycleBan(vocab, commit_select=True, removal=rule, mode=mode)
    steps, last = _case(name, vocab, rule)
    for i, (f, a) in enumerate(steps):
        if fork_at == i:
            cycle = cycle.fork()
        cycle.observe(f, tok.encode(f))
        cycle.record(f, a)
    if fork_at == len(steps):
        cycle = cycle.fork()
    sample = tok.encode(last)
    status = cycle.observe(last, sample)
    return cycle, sample, status


def _model_finite(vocab: Vocab, table: str, mode: str, samples: list[Sample]) -> tuple[list[list[int]], dict]:
    tok = tok_of(vocab, mode, table)
    torch.manual_seed(5)
    net = SpireNet(tok.layout, TINY).eval()
    with torch.no_grad():
        out = net(tok.collate(samples))
    return [torch.isfinite(out["logits"][i, :s.n_actions]).nonzero().flatten().tolist()
            for i, s in enumerate(samples)], out


@pytest.mark.parametrize("name", CASES)
def test_slot_cases_in_the_tracker_the_actor_decode_and_the_model(vocab: Vocab, table_path: str, name: str) -> None:
    """One encoding in every mode (no bit 14): the three modes write the same slots; "mask" / "teach" act on the
    removal rule's one row, "free" on every row of the removal prompt; every other case is the same in all modes."""
    got = {m: _observe(name, vocab, table_path, m) for m in MODES}
    (mc, ms, mst), (tc, ts, tst), (fc, fs, fst) = got["mask"], got["teach"], got["free"]
    assert slots(ms) == slots(ts) == slots(fs)
    n = ms.n_actions
    legal = mask_legal(ms.history[-CYCLE_BAN_SLOTS:], n).tolist()
    free_rows = free_legal(fs.history[-CYCLE_BAN_SLOTS:], n).tolist()
    assert sorted(set(range(n)) - set(mc.active_bans)) == legal and tc.active_bans == mc.active_bans
    assert sorted(set(range(n)) - set(fc.active_bans)) == free_rows
    if name == "removal":
        r = mc.rule_row
        assert slots(ms) == [-(r + 1)] + [0] * (CYCLE_BAN_SLOTS - 1) and legal == [r]
        assert free_rows == list(range(n)) and fc.rule_row == r and fst.removal_rule and not fst.banned
        assert mst == tst and mst.removal_rule
    else:
        assert free_rows == legal and mst == tst == fst and fc.active_bans == mc.active_bans
        assert mc.rule_row is None and fc.rule_row is None
    if name == "all_banned":
        assert mst.all_banned and slots(ms) == [0] * CYCLE_BAN_SLOTS and legal == list(range(n))
    for mode, sample in (("mask", ms), ("teach", ts), ("free", fs)):
        pol = UniformPolicy(3, rules_mode=mode)
        seen = {int(pol.launch([sample], np.asarray([n])).actions[0]) for _ in range(300)}
        assert seen == set(free_rows if mode == "free" else legal), mode
    for mode, sample in (("mask", ms), ("teach", ts), ("free", fs)):
        finite, out = _model_finite(vocab, table_path, mode, [sample])
        assert finite[0] == (free_rows if mode == "free" else legal), mode
        if mode != "mask":
            assert out["rules_covered"].tolist() == [float(name == "removal")]
        else:
            assert "teach_removal" not in out


def test_removal_with_cycle_rows_in_free(vocab: Vocab, table_path: str) -> None:
    """Only "free" can close a loop at a removal prompt (the rule fixes the row otherwise): slot 0 = the rule's row,
    slots 1.. = the cycle rows (masked in "free", as in "mask"); a loop through the rule's own row makes the prompt an
    ordinary one (no slot 0). "mask" / "teach" refuse a non-rule row at record."""
    rule = load_rule(vocab, table_path)
    f0 = removal_frame(vocab)
    r, _ = rule.choose(f0)
    cancel = rows(f0, Verb.CANCEL)[0]
    for mode in ("mask", "teach"):
        cycle = CycleBan(vocab, commit_select=True, removal=rule, mode=mode)
        cycle.observe(f0, tok_of(vocab, mode, table_path).encode(f0))
        with pytest.raises(AssertionError, match="cycle-banned"):
            cycle.record(f0, cancel)
    tok = tok_of(vocab, "free", table_path)
    n = f0.n_actions
    for loop in (cancel, r):
        cycle = CycleBan(vocab, commit_select=True, removal=rule, mode="free")
        cycle.observe(f0, tok.encode(f0))
        cycle.record(f0, loop)  # "free": any row the host offers
        f1 = removal_frame(vocab, step=1)
        s = tok.encode(f1)
        status = cycle.observe(f1, s)
        assert cycle.active_bans == (loop,) and status.banned
        if loop == cancel:
            assert status.removal_rule and cycle.rule_row == r
            assert slots(s) == [-(r + 1), cancel + 1] + [0] * (CYCLE_BAN_SLOTS - 2)
        else:
            assert not status.removal_rule and cycle.rule_row is None
            assert slots(s) == [r + 1] + [0] * (CYCLE_BAN_SLOTS - 1)
        want = [a for a in range(n) if a != loop]
        assert free_legal(s.history[-CYCLE_BAN_SLOTS:], n).tolist() == want
        assert mask_legal(s.history[-CYCLE_BAN_SLOTS:], n).tolist() == ([r] if loop == cancel else want)
        pol = UniformPolicy(1, rules_mode="free")
        seen = {int(pol.launch([s], np.asarray([n])).actions[0]) for _ in range(300)}
        assert seen == set(want)
        for mode in MODES:
            finite, out = _model_finite(vocab, table_path, mode, [s])
            assert finite[0] == (want if mode == "free" or loop == r else [r]), (mode, loop)
            if mode != "mask":
                assert out["rules_covered"].tolist() == [float(loop == cancel)]
        with pytest.raises(AssertionError, match="cycle-banned"):
            cycle.record(f1, loop)
        cycle.record(f1, want[0])


def test_six_cycle_rows_at_a_removal_prompt_keep_five(vocab: Vocab, table_path: str) -> None:
    """"free", six rows: slot 0 is the rule's, so at most five cycle rows fit; the sixth is not masked."""
    rule = load_rule(vocab, table_path)
    tok = tok_of(vocab, "free", table_path)
    cards = [(HIGH, 0)] * 6 + [(LOW, 0)]
    f = prompt(vocab, cards, confirm=True)
    r, _ = rule.choose(f)
    others = [a for a in range(f.n_actions) if a != r][:6]
    cycle = CycleBan(vocab, commit_select=True, removal=rule, mode="free")
    for i, a in enumerate(others):
        fi = prompt(vocab, cards, confirm=True, step=i)
        cycle.observe(fi, tok.encode(fi))
        cycle.record(fi, a)
    last = prompt(vocab, cards, confirm=True, step=6)
    s = tok.encode(last)
    st = cycle.observe(last, s)
    assert st.removal_rule and st.banned and len(cycle.bans[cycle.key]) == 6
    assert slots(s) == [-(r + 1)] + [a + 1 for a in others[:5]] and cycle.active_bans == tuple(others[:5])


def test_collators_keep_the_slots(vocab: Vocab, table_path: str) -> None:
    samples = [_observe(n, vocab, table_path, m)[1] for n in CASES for m in MODES]
    want = np.stack([s.history for s in samples])
    assert (want[:, -CYCLE_BAN_SLOTS] < 0).any() and (want[:, -CYCLE_BAN_SLOTS:] > 0).any()
    paths = {}
    for native in (False, True):
        tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, commit_select=True, removal_rule=True,
                                               removal_ratings=table_path, rules_mode="free"), native=native)
        paths[f"collate-{native}"] = tok.collate(samples)["history"].numpy()
        blk = SampleBlock.from_samples(samples)
        idx = np.arange(len(samples))
        shape = tok.shape_block(blk, idx)
        layout, nb = packed_layout(shape, tok.width)
        buf = torch.zeros(nb, dtype=torch.uint8)
        arr = {k: v.numpy() for k, v in packed_views(buf, layout).items()}
        tok.collate_block_into(blk, idx, shape, arr)
        paths[f"block-{native}"] = arr["history"][:len(samples)]
    for name, got in paths.items():
        assert np.array_equal(got[:, -CYCLE_BAN_SLOTS:], want[:, -CYCLE_BAN_SLOTS:]), name


@pytest.mark.parametrize("mode", MODES)
def test_fork_in_each_mode(vocab: Vocab, table_path: str, mode: str) -> None:
    rule = load_rule(vocab, table_path)
    for name in CASES:
        n = len(_case(name, vocab, rule)[0])
        ref = _observe(name, vocab, table_path, mode)
        for at in range(n + 1):
            cyc, s, st = _observe(name, vocab, table_path, mode, fork_at=at)
            assert slots(s) == slots(ref[1]) and st == ref[2] and cyc.mode == mode, (name, at)
            assert (cyc.active_bans, cyc.rule_row) == (ref[0].active_bans, ref[0].rule_row)


# =========================================================================== R1 / R2: identity


def _ref(tree: Path, mode: str, table: str) -> dict:
    env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1",
           "RULES_MODE": mode, "RULES_TABLE": table}
    out = subprocess.run([sys.executable, "-B", str(HERE / "rules_ref.py")], env=env, capture_output=True, text=True,
                         timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_mask_is_the_base_and_teach_without_coefficient_is_mask(table_path: str) -> None:
    """R1: SPIRE_RULES_BASE = tree p3 (alt/p4/base): initialisation (keys, values), samples and ban slots, actions,
    log-probs, values, the updated parameters and the update's terms equal. R2: "teach" with coef_removal 0 = "mask"
    in all of it, the ban slots included."""
    base = os.environ.get("SPIRE_RULES_BASE")
    if not base:
        pytest.skip("SPIRE_RULES_BASE (tree p3) not set")
    mine = HERE.parent
    ref, mask, teach = _ref(Path(base), "base", table_path), _ref(mine, "mask", table_path), \
        _ref(mine, "teach0", table_path)
    assert ref["decisions"] >= 3000 and min(ref["slots"].values()) >= 5, (ref["decisions"], ref["slots"])
    assert {k: v for k, v in mask.items() if k != "mode"} == {k: v for k, v in ref.items() if k != "mode"}
    assert mask["rules"] is None
    assert {k: v for k, v in teach.items() if k not in ("mode", "rules")} == \
        {k: v for k, v in mask.items() if k not in ("mode", "rules")}
    assert teach["rules"]["mode"] == "teach" and teach["rules"]["coef_removal"] == 0.0 and teach["rules"]["n"] > 0
    assert 0 < teach["rules"]["z"] < 1 and teach["rules"]["t"] > 0


def test_modes_add_no_parameter(vocab: Vocab, table_path: str) -> None:
    """R1 (in-tree part): rules_mode defaults to "mask"; every mode has the same state-dict keys, the same fixed-seed
    initialisation and draws the same random numbers; teach / free need cycle_ban and removal_rule."""
    assert TokenizerConfig().rules_mode == "mask"
    nets, rng = {}, {}
    for mode in (None, *MODES):
        flags: dict[str, Any] = {"cycle_ban": True, "removal_rule": True, "removal_ratings": table_path}
        tok = Tokenizer(vocab, TokenizerConfig(**flags) if mode is None else TokenizerConfig(**flags, rules_mode=mode))
        torch.manual_seed(0)
        nets[mode] = SpireNet(tok.layout, TINY)
        rng[mode] = torch.get_rng_state()
    base = nets[None].state_dict()
    for mode in MODES:
        sd = nets[mode].state_dict()
        assert list(sd) == list(base) and all(torch.equal(sd[k], base[k]) for k in base), mode
        assert torch.equal(rng[mode], rng[None])
    with pytest.raises(AssertionError, match="needs cycle_ban and removal_rule"):
        TokenizerConfig(cycle_ban=True, rules_mode="teach")
    with pytest.raises(AssertionError, match="rules_mode"):
        TokenizerConfig(cycle_ban=True, rules_mode="strict")


# =========================================================================== R4: the term


def test_term_against_hand_computed_values() -> None:
    """T_REMOVAL = -log P(rule row), P = softmax over the kept rows (host-legal minus masked); 0 without a removal
    prompt; P, agree; a padded sample contributes nothing and no NaN; the gradient reaches the kept rows only."""
    score = torch.tensor([[0.0, 1.0, 2.0, 5.0]] * 4, requires_grad=True)
    keep = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    rule = torch.zeros(4, 4, dtype=torch.bool)
    rule[0, 1] = rule[1, 1] = True  # sample 2: no removal prompt; sample 3: padding
    out = removal_terms(score, keep, rule)

    def lse(*x: float) -> float:
        return math.log(sum(math.exp(v) for v in x))

    assert out["teach_removal"].tolist() == pytest.approx([lse(0, 1, 2) - 1, lse(0, 1) - 1, 0, 0], abs=1e-6)
    assert out["rules_covered"].tolist() == [1, 1, 0, 0]
    assert out["rules_p"].tolist() == pytest.approx([math.exp(1 - lse(0, 1, 2)), math.exp(1 - lse(0, 1)), 0, 0])
    assert out["rules_agree"].tolist() == [0, 1, 0, 0]
    out["teach_removal"].sum().backward()
    g = score.grad
    assert g is not None and bool(torch.isfinite(g).all())
    assert g[0, 3] == 0 and g[1, 2] == 0 and g[1, 3] == 0 and bool((g[2:] == 0).all())
    assert g[0, 1] < 0 and g[0, 0] > 0 and g[0, 2] > 0  # the rule's row up, the other kept rows down


def test_the_term_drives_the_rule_row_up(vocab: Vocab, table_path: str) -> None:
    """R4 toy: the removal term alone (teach, the model's own output) drives P(rule row) above 0.97 and the arg-max to
    the rule's row; CANCEL and CONFIRM (fewer than allowed) lose their mass."""
    tok = tok_of(vocab, "teach", table_path)
    rule = load_rule(vocab, table_path)
    torch.manual_seed(2)
    net = SpireNet(tok.layout, TINY)
    samples = []
    for i in range(4):
        f = removal_frame(vocab, step=i)
        s = tok.encode(f)
        CycleBan(vocab, commit_select=True, removal=rule, mode="teach").observe(f, s)
        samples.append(s)
    batch = tok.collate(samples)
    f = removal_frame(vocab)
    cancel, confirm = rows(f, Verb.CANCEL)[0], rows(f, Verb.CONFIRM_SELECTION)[0]
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    assert float(net(batch)["rules_p"].mean()) < 0.9
    for _ in range(300):
        opt.zero_grad()
        net(batch)["teach_removal"].sum().backward()
        opt.step()
    out = net(batch)
    assert float(out["rules_p"].min()) > 0.97 and bool(out["rules_agree"].all())
    tok_f = tok_of(vocab, "free", table_path)
    net.rules_mode = "free"
    with torch.no_grad():
        lf = net(tok_f.collate(samples))["logits"]
    p = torch.softmax(lf, 1)
    assert bool(torch.isfinite(lf[:, cancel]).all()) and float(p[:, cancel].max()) < 0.02
    assert float(p[:, confirm].max()) < 0.02


# =========================================================================== R5: "free", fade, train state


class _RemovalGame(FakeGame):
    """REST (an option that opens the removal prompt, PROCEED) -> removal prompt (SELECT HIGH, SELECT LOW, CONFIRM =
    fewer than allowed, CANCEL = back to REST) -> after a SELECT the confirm step (CONFIRM removes the card, CANCEL back
    to REST); a removal or CONFIRM at the prompt closes the option; PROCEED ends the game: WIN iff LOW was removed;
    14 decisions at most. CANCEL -> REST -> option returns to the same prompt: a cycle in "free"."""

    def __init__(self, vocab: Vocab, seed: int) -> None:
        super().__init__(vocab, seed)
        self.step, self.screen, self.open, self.picked, self.removed = 0, "rest", True, -1, -1

    def handle(self, cmd: Command) -> list[bytes]:
        if isinstance(cmd, QuitCmd):
            return []
        if isinstance(cmd, ResetCmd):
            self.episode_id += 1
            self.step, self.screen, self.open, self.picked, self.removed = 0, "rest", True, -1, -1
        else:
            assert isinstance(cmd, StepCmd) and cmd.step_id == self.step
            self.step += 1
            verb = self.rows[cmd.action_index]
            if self.screen == "rest":
                self.screen = "grid" if verb == Verb.CHOOSE_OPTION else "end"
            elif self.screen == "grid":
                if verb == Verb.SELECT_CARD:
                    self.screen, self.picked = "confirm", self.cards[cmd.action_index]
                else:
                    self.screen, self.open = "rest", verb == Verb.CANCEL
            else:
                if verb == Verb.CONFIRM_SELECTION:
                    self.removed, self.open = self.picked, False
                self.screen = "rest"
            if self.step == 14 and self.screen != "end":
                self.screen = "cap"
        b = FrameBuilder(MsgType.TERMINAL if self.screen in ("end", "cap") else MsgType.OBS)
        b.episode_id, b.step_id = self.episode_id, self.step
        g = b.glob
        g[G.N_SEATS], g[G.TOTAL_FLOOR] = 1, 3
        g[G.ROOM_TYPE] = self.vocab.id("enum:RoomType", "RestSite")
        b.add(Table.SEATS, seat=0, hp=60, max_hp=70, alive=1, creature_row=-1,
              character_id=self.vocab.id("character", "SILENT"))
        verbs: list[int] = []
        cards: list[int] = []
        if self.screen == "rest":
            g[G.KIND] = Kind.REST
            if self.open:
                b.add(Table.OPTIONS, kind=OptionKind.REST_OPTION, seat=0, pos=0,
                      key_id=self.vocab.id("rest_option", "SMITH"))
                b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=0)
            b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=0)
        elif self.screen in ("grid", "confirm"):
            g[G.KIND], g[G.KIND_SUB], g[G.SEL_MIN], g[G.SEL_MAX] = Kind.CARD_SELECT, CardSelectSub.DECK_REMOVAL, 1, 1
            if self.screen == "grid":
                for pos, name in enumerate((HIGH, LOW)):
                    row = b.add(Table.CARDS, zone=Zone.CHOICE, seat=0, pos=pos, count=1,
                                card_id=self.vocab.id("card", name))
                    b.add(Table.ACTIONS, verb=Verb.SELECT_CARD, seat=0, src_table=Table.CARDS, src_row=row)
            else:
                g[G.SEL_COUNT] = 1
            b.add(Table.ACTIONS, verb=Verb.CONFIRM_SELECTION, seat=0)
            b.add(Table.ACTIONS, verb=Verb.CANCEL, seat=0)
        else:
            g[G.KIND] = Kind.TERMINAL
            win = self.screen == "end" and self.removed == self.vocab.id("card", LOW)
            b.flags, b.result = FrameFlag.TERMINAL, Result.WIN if win else Result.LOSS
            g[G.TERMINAL_REASON] = TerminalReason.DECISION_CAP if self.screen == "cap" else                 TerminalReason.NONE if win else TerminalReason.DEATH
            g[G.POLICY_DECISIONS] = self.step
        raw = b.build(hashes=False)
        f = parse_frame(raw)
        self.rows = [int(v) for v in f.actions[:, ActionCol.VERB]]
        self.cards = [int(f.table(Table.CARDS)[r][CardCol.CARD_ID]) if v == Verb.SELECT_CARD else -1
                      for r, v in zip(f.actions[:, ActionCol.SRC_ROW], self.rows)]
        return [raw]


def _removal_host(vocab: Vocab, seed: int) -> FakeHost:
    host = FakeHost(vocab)
    host.game = _RemovalGame(vocab, seed)
    return host


def test_fade_schedule() -> None:
    cfg = RulesConfig(coef_removal=2.0, fade_updates=1000, floor_removal=0.1)
    assert coefficient(cfg, "mask", None) == 0.0 and coefficient(cfg, "teach", None) == 2.0
    for k, want in ((0, 2.0), (250, 2.0 * 0.775), (500, 2.0 * 0.55), (999, 2.0 * (1 - 0.9 * 0.999)), (1000, 0.2),
                    (1500, 0.2), (10 ** 6, 0.2)):
        assert coefficient(cfg, "free", k) == pytest.approx(want), k
    assert RulesConfig() == RulesConfig(coef_removal=0.05, fade_updates=1000, floor_removal=0.1)
    with pytest.raises(AssertionError):
        RulesConfig(fade_updates=0)
    with pytest.raises(AssertionError):
        RulesConfig(floor_removal=1.5)


def _learner(vocab: Vocab, table: str, mode: str, rules: RulesConfig = RulesConfig()) -> tuple[Learner, Tokenizer]:
    tok = tok_of(vocab, mode, table)
    torch.manual_seed(0)
    net = SpireNet(tok.layout, TINY)
    return Learner(net, tok, PPOConfig(minibatch=256, micro_batch=256, epochs=1, rules=rules), device="cpu",
                   fast=False), tok


def _segments(learner: Learner, tok: Tokenizer, seed: int, decisions: int = 400
              ) -> tuple[list[Segment], dict[str, int], tuple[int, int]]:
    """Segments of an actor run, the actor's RULES_KEYS totals (its reports + the rest) and, from the recorded
    samples, (removal prompts, those whose action is the rule's row)."""
    segs: list[Segment] = []
    reps: list[dict[str, Any]] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(tok.vocab), config=RolloutConfig(segment_len=32))
    hosts = [FakeHost(tok.vocab, seed=seed + i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
             for i in range(3)] + [_host(tok.vocab)] + [_removal_host(tok.vocab, seed + 10 + i) for i in range(2)]
    actor = Actor(hosts, TorchPolicy(learner.model, tok, seed=seed), tok,
                  EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0),)), tok.vocab, seed), ActorConfig(), store=store,
                  on_report=reps.append)
    actor.run(max_decisions=decisions)
    tot = {k: sum(r["rules"][k] for r in reps) + actor.rules_counts[k] for k in RULES_KEYS}         if learner.rules_mode != "mask" else {}
    slot0 = [(int(x.history[-CYCLE_BAN_SLOTS]), int(a)) for g in segs for x, a in zip(g.samples, g.actions)]
    rem = [(-v - 1, a) for v, a in slot0 if v < 0]
    return segs, tot, (len(rem), sum(r == a for r, a in rem))


def test_free_play_records_and_the_fade_survives_a_resume(vocab: Vocab, table_path: str, tmp_path: Path) -> None:
    """mask -> teach (--migrate-rules: weights and optimizer state unchanged) -> free (--rules-free); in "free" the
    actor plays non-rule rows of removal prompts (recorded and counted); the fade counts from the first update in
    "free" and its field survives a plain resume."""
    rules = RulesConfig(coef_removal=1.0, fade_updates=4)
    want = [1.0, 0.775, 0.55, 0.325, 0.1, 0.1]
    mask, mtok = _learner(vocab, table_path, "mask", rules)
    segs, _, _ = _segments(mask, mtok, 1)
    mask.update(segs)
    mask.save(tmp_path / "mask.pt")
    teach, ttok = _learner(vocab, table_path, "teach", rules)
    with pytest.raises(AssertionError, match="migrate-rules"):
        teach.load(tmp_path / "mask.pt")
    teach, ttok = _learner(vocab, table_path, "teach", rules)
    teach.load(tmp_path / "mask.pt", migrate_rules=True)
    for a, b in ((teach.model.state_dict(), mask.model.state_dict()), (teach.ema.state_dict(), mask.ema.state_dict())):
        assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in b)
    opt, mopt = teach.opt.state_dict(), mask.opt.state_dict()
    assert opt["param_groups"] == mopt["param_groups"] and set(opt["state"]) == set(mopt["state"])
    assert all(torch.equal(opt["state"][k][n], mopt["state"][k][n]) for k in opt["state"] for n in opt["state"][k])
    segs, tot, (n_rem, n_rule) = _segments(teach, ttok, 2)
    assert tot["removal_prompts"] == tot["removal_rule"] > 0 and n_rem == n_rule > 0
    m = teach.update(segs)
    assert m["rules"]["mode"] == "teach" and m["rules"]["coef_removal"] == 1.0 and m["rules"]["n"] > 0
    teach.save(tmp_path / "teach.pt")
    free, ftok = _learner(vocab, table_path, "free", rules)
    with pytest.raises(AssertionError, match="rules-free"):
        free.load(tmp_path / "teach.pt")
    free, ftok = _learner(vocab, table_path, "free", rules)
    free.load(tmp_path / "teach.pt", rules_free=True)
    assert free.rules_free_first is None
    got, counts = [], {k: 0 for k in RULES_KEYS}
    for u in range(3):
        segs, tot, (n_rem, n_rule) = _segments(free, ftok, 10 + u)
        for k in RULES_KEYS:
            counts[k] += tot[k]
        counts["seg_prompts"] = counts.get("seg_prompts", 0) + n_rem
        counts["seg_rule"] = counts.get("seg_rule", 0) + n_rule
        got.append(free.update(segs)["rules"]["coef_removal"])
    assert free.rules_free_first == 2  # updates 0 (mask) and 1 (teach) came before
    free.save(tmp_path / "free.pt")
    again, atok = _learner(vocab, table_path, "free", rules)
    again.load(tmp_path / "free.pt")  # a plain resume: the train-state field survives
    assert again.rules_free_first == 2
    for u in range(3):
        segs, _, _ = _segments(again, atok, 20 + u)
        got.append(again.update(segs)["rules"]["coef_removal"])
    assert got == pytest.approx(want)
    assert counts["removal_prompts"] == sum(counts[k] for k in RULES_KEYS[1:]) > 0
    # "free" chose other rows at removal prompts (counted, recorded in the segments: record accepted them)
    assert counts["removal_prompts"] > counts["removal_rule"] and counts["seg_prompts"] > counts["seg_rule"], counts
    with pytest.raises(AssertionError, match="mask -> teach"):
        _learner(vocab, table_path, "teach", rules)[0].load(tmp_path / "teach.pt", migrate_rules=True)


def test_kickstart_teacher_needs_mask_and_config_keys(vocab: Vocab, tmp_path: Path, table_path: str) -> None:
    with pytest.raises(AssertionError, match="rules_mode mask"):
        load_teacher(tmp_path / "t.pt", tok_of(vocab, "teach", table_path), None, TINY)
    cfg = build(TrainConfig, {"run_name": "x", "tokenizer": {"cycle_ban": True, "removal_rule": True,
                                                            "removal_ratings": table_path, "rules_mode": "teach"},
                              "ppo": {"rules": {"coef_removal": 0.5, "fade_updates": 10}}})
    assert cfg.tokenizer.rules_mode == "teach" and cfg.ppo.rules == RulesConfig(coef_removal=0.5, fade_updates=10)
    with pytest.raises(ValueError, match="unknown keys"):
        build(TrainConfig, {"run_name": "x", "ppo": {"rules": {"coef_cycle": 1.0}}})


# =========================================================================== R6: a full run in each mode


def _running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            s = fh.read().decode()
    except FileNotFoundError:
        return False
    return s[s.rindex(")") + 2] != "Z"


@linux
def test_supervisor_runs_in_each_mode_and_leaves_no_process(tmp_path: Path, table_path: str) -> None:
    """R6: Supervisor + learner + actor processes on fake hosts, 3 updates in "mask", then resumed with
    --migrate-rules ("teach", 3 more), then with --rules-free ("free", 3 more): versions continue, learner rows carry
    the rules block of their mode, "free" train rows the removal split; no actor or learner process is left."""
    import dataclasses
    from spire.metrics import read_jsonl
    from test_train import small_config
    tok = {"cycle_ban": True, "public_history": True, "commit_select": True, "removal_rule": True,
           "removal_ratings": table_path}
    hosts = {"envs_per_actor": 4, "fake": {"learnable": True, "mean_decisions": 60, "hashes": False}}
    run = tmp_path / "r6"
    assert Supervisor(small_config(tmp_path, "r6", max_updates=3, tokenizer=tok, hosts=hosts)).run() == 0
    for mode, flag, upto in (("teach", "migrate_rules", 6), ("free", "rules_free", 9)):
        cfg = load_config(run / "config.json", max_updates=upto)
        cfg = dataclasses.replace(cfg, tokenizer=dataclasses.replace(cfg.tokenizer, rules_mode=mode), **{flag: True})
        assert Supervisor(cfg, resume=True).run() == 0
        saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
        assert saved["tokenizer"]["rules_mode"] == mode and not saved["migrate_rules"] and not saved["rules_free"]
    rows_ = list(read_jsonl(run / "metrics.jsonl"))
    learner = [r for r in rows_ if r["kind"] == "learner"]
    assert [r["version"] for r in learner] == list(range(1, 10))
    assert [r.get("rules", {}).get("mode") for r in learner] == [None] * 3 + ["teach"] * 3 + ["free"] * 3
    assert all(r["rules"]["coef_removal"] == RulesConfig().coef_removal for r in learner[3:7])
    assert learner[7]["rules"]["coef_removal"] < learner[6]["rules"]["coef_removal"]
    pids = {p for r in rows_ if r["kind"] == "train" for p in r.get("actor_pids", [])}
    learners = {r["pid"] for r in rows_ if r["kind"] == "event" and r.get("what") == "learner_ready"}
    free_rows = [r for r in rows_ if r["kind"] == "train" and "rules_removal_prompts" in r]
    assert len(pids) >= 3 and len(learners) == 3 and free_rows  # one actor and one learner process per start
    eps = list(read_jsonl(run / "episodes.jsonl"))
    assert any("rules" in e for e in eps) and not any("rules" in e for e in eps[:5])
    deadline = time.monotonic() + 30
    while any(_running(p) for p in pids | learners) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not [p for p in pids | learners if _running(p)]
