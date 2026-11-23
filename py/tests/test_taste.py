"""spire.taste (notes/solo/taste-spec.md): the five invariants of spec section 4 (with the thin case of v1.1 B),
the table / config checks, the actor -> shared memory -> learner path and a supervisor run on fake hosts."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
from spire.distributions import exploration_table
from spire.ipc import decode_segment, pack_segment
from spire.learner import Learner, PPOConfig, _Accum
from spire.model import ModelConfig, SpireNet
from spire.proto import Frame, parse_frame
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import ActionCol, CardCol, G, Kind, SeatCol, Table, Verb, Zone
from spire.synth import FakeHost, FakeHostConfig, FrameGen
from spire.taste import (Family, Taste, TasteConfig, episode_values, family_rows, load_table, skip_rows, target_rows,
                         taste_log_prob)
from spire.tokenize import BatchShape, Tokenizer
from spire.trainstep import DEN_COLS, NF, PF_COLS, MicroStepper, _Res
from spire.vocab import Vocab

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32)
REF = Path(__file__).with_name("taste_ref.py")
EPS = 0.6


def card_names(vocab: Vocab, residue: int) -> list[str]:
    return [vocab["card"].entries[i] for i in range(2, vocab.size("card")) if i % 3 == residue]


def write_table(path: Path, families: dict[str, dict[str, dict[str, int]]]) -> str:
    path.write_text(json.dumps({"version": 1, "characters": {c: {"families": {n: {"cards": cards} for n, cards in
                                                                               fams.items()}}
                                                             for c, fams in families.items()}}), encoding="utf-8")
    return str(path)


def placeholder_table(path: Path, vocab: Vocab, chars: tuple[str, ...] = ("IRONCLAD", "SILENT")) -> str:
    """The two-family placeholder of spec 5 (synthetic card names, caps 1-3)."""
    return write_table(path, {c: {"r0": {n: 1 + i % 3 for i, n in enumerate(card_names(vocab, 0))},
                                  "r1": {n: 2 for n in card_names(vocab, 1)}} for c in chars})


def card_reward_frame(vocab: Vocab, seed: int, n_seats: int = 1) -> Frame:
    f = parse_frame(FrameGen(vocab).frame(np.random.default_rng(seed), Kind.CARD_REWARD, n_seats=n_seats))
    assert (f.actions[:, ActionCol.VERB] == Verb.PICK_OFFER_CARD).sum() >= 3
    return f


def edited(f: Frame, seat: int, *, card: int = -1, count: int = 0, deck_count: int = -1) -> Frame:
    """``f`` with a DECK row of ``count`` copies of ``card`` for ``seat`` and / or the seat's DECK_COUNT set."""
    tabs = [None if t is None else t.copy() for t in f.tables]
    if card >= 0:
        cards = tabs[Table.CARDS]
        row = np.zeros((1, cards.shape[1]), cards.dtype)
        row[0, CardCol.ZONE], row[0, CardCol.SEAT], row[0, CardCol.CARD_ID] = Zone.DECK, seat, card
        row[0, CardCol.COUNT] = count
        tabs[Table.CARDS] = np.concatenate([cards, row])
    if deck_count >= 0:
        tabs[Table.SEATS][seat, SeatCol.DECK_COUNT] = deck_count
    g = copy.copy(f)
    g.tables = tabs
    return g


def offered(f: Frame) -> tuple[np.ndarray, np.ndarray]:
    picks = np.flatnonzero(f.actions[:, ActionCol.VERB] == Verb.PICK_OFFER_CARD)
    return picks, f.table(Table.CARDS)[f.actions[picks, ActionCol.SRC_ROW], CardCol.CARD_ID].astype(np.int64)


def frame_taste(tmp_path: Path, vocab: Vocab, f: Frame, cards: dict[int, int], *, thin: bool = False,
                eps: float = EPS, thin_deck: int = 20) -> Taste:
    """A taste whose single family (cards: vocab id -> cap) is drawn for every seat of env 0 (p_episode 1)."""
    chars = sorted({vocab["character"].entries[int(c)] for c in f.table(Table.SEATS)[:, SeatCol.CHARACTER_ID]})
    fam = {vocab["card"].entries[c]: cap for c, cap in cards.items()}
    path = write_table(tmp_path / "f.json", {c: {"fam": fam} for c in chars})
    t = Taste(TasteConfig(path, 1.0, eps, p_thin=1.0 if thin else 0.0, thin_deck=thin_deck), vocab, (0, 1, 2, 3, 4))
    t.begin(0, [int(c) for c in f.table(Table.SEATS)[:, SeatCol.CHARACTER_ID]])
    return t


def not_offered(vocab: Vocab, f: Frame, n: int) -> list[int]:
    return [int(c) for c in np.setdiff1d(np.arange(2, vocab.size("card")), offered(f)[1])[:n]]


# ------------------------------------------------------------------ table / config


def test_config_and_table_checks(tmp_path: Path, vocab: Vocab) -> None:
    assert not PPOConfig().taste.on and PPOConfig().taste == TasteConfig()
    assert TasteConfig("x", {"SILENT": 0.5, "IRONCLAD": 0.25}, 0.7).p_for("DEFECT") == 0.0
    assert not TasteConfig("x", {"SILENT": 0.0}, 0.7).on
    for bad in (dict(p_episode=0.5, eps=0.7), dict(file="x", p_episode=0.5, eps=0.0), dict(file="x", p_episode=1.5,
                                                                                         eps=0.5),
                dict(file="x", p_episode=0.5, eps=1.0), dict(file="x", p_episode={"SILENT": 2.0}, eps=0.5)):
        with pytest.raises(AssertionError):
            TasteConfig(**bad)  # type: ignore[arg-type]
    for anchor in (dict(recompute_values=True), dict(decoupled_clip_anchor=True)):
        PPOConfig(**anchor)  # type: ignore[arg-type]
        with pytest.raises(AssertionError, match="re-derive"):
            PPOConfig(taste=TasteConfig("x", 0.5, 0.7), **anchor)  # type: ignore[arg-type]
    version, table = load_table(placeholder_table(tmp_path / "t.json", vocab), vocab)
    assert version == 1 and len(table) == 2 and all(len(v) == 2 for v in table.values())
    raw = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    for v, ok in ((2, True), (7, True), (0, False), ("2", False)):
        (tmp_path / "v.json").write_text(json.dumps({**raw, "version": v}), encoding="utf-8")
        if ok:
            assert load_table(str(tmp_path / "v.json"), vocab)[0] == v
        else:
            with pytest.raises(AssertionError, match="version"):
                load_table(str(tmp_path / "v.json"), vocab)
    assert TasteConfig().thin_deck == 24 and TasteConfig().start_floor == 0
    for fams, msg in (({"SILENT": {"x": {"NOT_A_CARD": 2}}}, "unknown cards"),
                      ({"NOT_A_HERO": {"x": {card_names(vocab, 0)[0]: 2}}}, "unknown character"),
                      ({"SILENT": {"x": {card_names(vocab, 0)[0]: 0}}}, "caps"),
                      ({"SILENT": {"x": {}}}, "SILENT/x")):
        with pytest.raises(AssertionError, match=msg):
            load_table(write_table(tmp_path / "bad.json", fams), vocab)


def test_episode_hash_is_unique_and_seat_independent() -> None:
    vals = {episode_values((0, 20261185, run, slot, inc, env, k), s)
            for run in (1, 2) for slot in (0, 1) for inc in (0, 1) for env in range(3) for k in range(20) for s in range(4)}
    assert len(vals) == 2 * 2 * 2 * 3 * 20 * 4
    u = np.asarray([v[0] for v in vals])
    assert 0.45 < (u < 0.5).mean() < 0.55 and all(0 <= x < 1 for v in vals for x in v)


# ------------------------------------------------------------------ invariant 3: unbiased


def test_unbiased_policy_gradient_by_enumeration() -> None:
    """5 actions, 2 of them in F: sum_a b(a) grad[b_theta(a) / b(a) A(a)] = t_keep sum_a A(a) grad q_theta(a)."""
    torch.manual_seed(0)
    z = torch.randn(5, dtype=torch.float64, requires_grad=True)
    adv = torch.randn(5, dtype=torch.float64)
    eps_k, eps_t = 0.1, 0.7
    q = (1 - eps_k) * z.softmax(-1) + eps_k / 5
    inside = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float64)
    t_keep = torch.full((5,), 1 - eps_t, dtype=torch.float64)
    t_add = eps_t * inside / inside.sum()
    log_b = taste_log_prob(q.log(), t_keep, t_add)
    b = log_b.exp()
    assert torch.allclose(b, (1 - eps_t) * q + t_add) and abs(float(b.detach().sum()) - 1) < 1e-12
    surrogate = (b.detach() * (log_b - log_b.detach()).exp() * adv).sum()  # sum_a b(a) * ratio(a) * A(a)
    g_taste, = torch.autograd.grad(surrogate, z, retain_graph=True)
    g_base, = torch.autograd.grad((adv * q).sum(), z)
    torch.testing.assert_close(g_taste, (1 - eps_t) * g_base, rtol=1e-12, atol=1e-14)


# ------------------------------------------------------------------ invariants 2 (formula) and 4 (sampling)


def mixture_sample(f: Frame, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The actor's own sampler (TorchPolicy._sample_all with the CARD_REWARD mixture) on one frame's fixed logits,
    ``n`` times: (logits, actions, log q rows, q)."""
    pol = TorchPolicy.__new__(TorchPolicy)
    pol.mode, pol.temperature, pol.rng = "sample", 1.0, np.random.default_rng(seed)
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, (1, f.n_actions + 3)).astype(np.float32)
    z[0, f.n_actions:] = -np.inf  # padding
    picks, _ = offered(f)
    banned = int(picks[-1])  # a cycle-banned PICK row: never a target
    z[0, banned] = -np.inf
    expl = exploration_table(((int(Kind.CARD_REWARD), 0.1),))[[int(Kind.CARD_REWARD)] * n]
    a, _, _, logq = pol._sample_all(np.repeat(z, n, 0), expl)
    q = np.where(np.isfinite(z[0]), np.exp(logq[0]), 0.0)
    assert abs(q.sum() - 1) < 1e-5
    return z[0], a, logq, q


@pytest.mark.parametrize("thin", [False, True])
def test_sampling_frequencies_and_stored_log_prob(tmp_path: Path, vocab: Vocab, thin: bool) -> None:
    """200,000 draws of the actor's composite sampler (policy mixture, then taste) match b = (1 - eps) q + eps U_F
    within 3 sigma; the stored log b is exact, and the learner formula on log q reproduces it (|d| < 1e-5)."""
    f = card_reward_frame(vocab, 3)
    picks, cids = offered(f)
    seat = int(f.gvals[G.ACTING_SEAT])
    if thin:  # no family card on offer, a 25-card deck: the target is the Skip row
        f = edited(f, seat, deck_count=25)
        taste = frame_taste(tmp_path, vocab, f, {c: 2 for c in not_offered(vocab, f, 3)}, thin=True)
    else:  # two offered cards in the family; the banned row's card too
        taste = frame_taste(tmp_path, vocab, f, {int(cids[0]): 2, int(cids[1]): 2, int(cids[-1]): 2})
    z, a, logq, q = mixture_sample(f, 200_000, 1)
    legal = np.isfinite(z)
    fam = taste.table[int(f.table(Table.SEATS)[seat, SeatCol.CHARACTER_ID])][0]
    rows, is_skip = target_rows(f, seat, fam, thin, 20, legal, taste.skip_key)
    assert is_skip == thin and len(rows) >= 1 and (thin or np.isin(rows, picks[:-1]).all())
    assert int(picks[-1]) not in rows
    b = (1 - EPS) * q
    b[rows] += EPS / len(rows)
    out = np.empty(len(a), np.int64)
    logb = np.empty(len(a))
    keep = np.empty(len(a))
    add = np.empty(len(a))
    for k in range(len(a)):
        out[k], logb[k], keep[k], add[k] = taste.steer(0, f, z, logq[k], int(a[k]))
    freq = np.bincount(out, minlength=len(b)) / len(a)
    sd = np.sqrt(b * (1 - b) / len(a))
    assert (np.abs(freq - b) <= 3 * sd + 1e-12).all(), (freq, b)
    assert np.allclose(logb, np.log(b[out]), rtol=0, atol=1e-6)
    assert (keep == 1 - EPS).all() and np.array_equal(add > 0, np.isin(out, rows))
    back = taste_log_prob(torch.as_tensor(logq[0][out]), torch.as_tensor(keep), torch.as_tensor(add))
    assert float((back - torch.as_tensor(logb)).abs().max()) < 1e-5
    st = taste.seats[0][seat]
    assert st.active == len(a) and (st.skips if thin else st.family_taken) == int((add > 0).sum())


# ------------------------------------------------------------------ invariant 5: cap, empty F, thin, seats


def test_cap_counts_plain_and_upgraded_copies_and_empty_f_is_ordinary(tmp_path: Path, vocab: Vocab) -> None:
    for seed in range(50):
        f = card_reward_frame(vocab, seed)
        picks, cids = offered(f)
        seat = int(f.gvals[G.ACTING_SEAT])
        c = int(cids[0])
        cards = f.table(Table.CARDS)
        if not ((cards[:, CardCol.ZONE] == Zone.DECK) & (cards[:, CardCol.CARD_ID] == c)).any():
            break
    legal = np.ones(f.n_actions, bool)
    n_rows = int((cids == c).sum())
    cap2, cap3 = (Family("f", np.array([c]), np.array([k])) for k in (2, 3))
    assert len(family_rows(f, seat, cap2, legal)) == n_rows
    assert len(family_rows(edited(f, seat, card=c, count=1), seat, cap2, legal)) == n_rows
    assert len(family_rows(edited(f, seat, card=c, count=2), seat, cap2, legal)) == 0
    up = edited(edited(f, seat, card=c, count=1), seat, card=c, count=1)  # a plain and an upgraded copy: two rows
    up.tables[Table.CARDS][-1, CardCol.UPGRADE_LEVEL] = 1
    assert len(family_rows(up, seat, cap2, legal)) == 0 and len(family_rows(up, seat, cap3, legal)) == n_rows
    banned = legal.copy()
    banned[picks[cids == c]] = False
    assert len(family_rows(f, seat, cap2, banned)) == 0  # rows the sampler cannot take are never targets
    # F empty (the card at its cap): an ordinary step, t_keep 1, t_add 0, nothing consumed from the taste generator
    g = edited(f, seat, card=c, count=2)
    t = frame_taste(tmp_path, vocab, g, {c: 2})
    state = t.rng.bit_generator.state
    z = np.zeros(f.n_actions)
    assert t.steer(0, g, z, np.full(f.n_actions, -1.5), int(picks[0])) == (int(picks[0]), -1.5, 1.0, 0.0)
    assert t.rng.bit_generator.state == state and t.seats[0][seat].active == 0


def test_thin_rule_needs_an_empty_family_and_a_big_deck(tmp_path: Path, vocab: Vocab) -> None:
    f = card_reward_frame(vocab, 4)
    picks, cids = offered(f)
    seat = int(f.gvals[G.ACTING_SEAT])
    legal = np.ones(f.n_actions, bool)
    t = frame_taste(tmp_path, vocab, f, {c: 2 for c in not_offered(vocab, f, 2)}, thin=True)
    skip = skip_rows(f, legal, t.skip_key)
    assert len(skip) == 1
    fam = Family("f", np.array(not_offered(vocab, f, 2)), np.array([2, 2]))
    offer = Family("g", np.array([int(cids[0])]), np.array([3]))
    assert target_rows(edited(f, seat, deck_count=19), seat, fam, True, 20, legal, t.skip_key)[0].size == 0
    rows, is_skip = target_rows(edited(f, seat, deck_count=20), seat, fam, True, 20, legal, t.skip_key)
    assert is_skip and np.array_equal(rows, skip)
    assert target_rows(edited(f, seat, deck_count=30), seat, fam, False, 20, legal, t.skip_key)[0].size == 0
    rows, is_skip = target_rows(edited(f, seat, deck_count=30), seat, offer, True, 20, legal, t.skip_key)
    assert not is_skip and len(rows) >= 1  # a family card on offer: the family rows, not Skip
    no_skip = legal.copy()
    no_skip[skip] = False
    assert target_rows(edited(f, seat, deck_count=30), seat, fam, True, 20, no_skip, t.skip_key)[0].size == 0


def test_four_seat_frame_uses_the_acting_seat(tmp_path: Path, vocab: Vocab) -> None:
    for seed in range(60):
        f = card_reward_frame(vocab, seed, n_seats=4)
        if f.table(Table.SEATS).shape[0] == 4 and int(f.gvals[G.ACTING_SEAT]) > 0:
            break
    seat = int(f.gvals[G.ACTING_SEAT])
    picks, cids = offered(f)
    c = int(cids[0])
    cards = f.table(Table.CARDS)
    own = int(cards[(cards[:, CardCol.ZONE] == Zone.DECK) & (cards[:, CardCol.SEAT] == seat)
                    & (cards[:, CardCol.CARD_ID] == c), CardCol.COUNT].sum())
    g = edited(f, (seat + 1) % 4, card=c, count=9)  # another seat holds nine: does not count
    t = frame_taste(tmp_path, vocab, g, {c: own + 1}, eps=0.99)
    z = np.zeros(f.n_actions)
    lq = np.full(f.n_actions, -2.0)
    hits = [t.steer(0, g, z, lq, int(picks[-1]))[0] for _ in range(300)]
    rows = picks[cids == c]
    assert np.mean(np.isin(hits, rows)) > 0.95
    assert t.steer(0, edited(g, seat, card=c, count=1), z, lq, int(picks[-1]))[2:] == (1.0, 0.0)  # own deck at cap
    t.seats[0][seat].family = -1  # the acting seat without taste: untouched although every other seat has one
    assert t.steer(0, g, z, lq, int(picks[-1])) == (int(picks[-1]), -2.0, 1.0, 0.0)
    assert int(f.table(Table.SEATS)[seat, SeatCol.CHARACTER_ID]) == t.seats[0][seat].character
    t.end(0, "win", 30, 7, 3)
    eps = t.take()["episodes"]
    assert len(eps) == 4 and {e[1] for e in eps} >= {"none", "fam"} and all(e[2] == 7 and e[3] == 1 for e in eps)
    assert all(e[8] == 0 for e in eps)  # a win is no act-1 death
    t.begin(0, [int(c) for c in f.table(Table.SEATS)[:, SeatCol.CHARACTER_ID]])
    t.end(0, "loss", 12, 7, 0)
    t.begin(0, [int(c) for c in f.table(Table.SEATS)[:, SeatCol.CHARACTER_ID]])
    t.end(0, "loss", 30, 7, 1)
    eps = t.take()["episodes"]
    assert [e[8] for e in eps] == [1] * 4 + [0] * 4


def test_start_floor_keeps_taste_inactive_below_it(tmp_path: Path, vocab: Vocab) -> None:
    """v1.2 J: below start_floor a taste seat's CARD_REWARD is an ordinary step (no draw consumed); from it on the
    family rows are targets again; the hash draw is the same."""
    f = card_reward_frame(vocab, 3)
    picks, cids = offered(f)
    floor = int(f.gvals[G.TOTAL_FLOOR])
    path = write_table(tmp_path / "s.json", {vocab["character"].entries[int(c)]: {"fam": {
        vocab["card"].entries[int(cids[0])]: 3}} for c in f.table(Table.SEATS)[:, SeatCol.CHARACTER_ID]})
    chars = [int(c) for c in f.table(Table.SEATS)[:, SeatCol.CHARACTER_ID]]
    z = np.zeros(f.n_actions)
    lq = np.full(f.n_actions, -1.0)
    for start, active in ((floor + 1, False), (floor, True)):
        t = Taste(TasteConfig(path, 1.0, 0.99, start_floor=start), vocab, (0, 1, 2, 3, 4))
        t.begin(0, chars)
        state = t.rng.bit_generator.state
        out = t.steer(0, f, z, lq, int(picks[-1]))
        assert (out[2] < 1) == active and (t.rng.bit_generator.state != state) == active



# ------------------------------------------------------------------ the captured learner body


class _StaticLogits(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        zeros = self.logits.new_zeros(len(self.logits))
        return {"logits": self.logits, "value": zeros, "pwin_logit": zeros, "floors_pred": zeros, "hp_pred": zeros}


def _microbody(taste: TasteConfig, keep: torch.Tensor, add: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor([[2.0, -2.0, 1.0], [-1.0, 0.5, 2.0], [0.0, -torch.inf, -torch.inf],
                           [-torch.inf, -torch.inf, -torch.inf]])
    model = _StaticLogits(logits)
    cfg = PPOConfig(exploration_by_kind=((int(Kind.CARD_REWARD), 0.2),), ent_coef=0, vf_coef=0, aux_win=0,
                    aux_floors=0, aux_hp=0, taste=taste)
    st = MicroStepper.__new__(MicroStepper)
    st.model, st.fwd, st.cfg, st.autocast = model, model, cfg, False
    st._choice_verb_slot = 0
    st._exploration = torch.from_numpy(exploration_table(cfg.exploration_by_kind))
    st._uniform_kl = None
    st.acc = _Accum(torch.device("cpu"))
    st.den = torch.ones(len(DEN_COLS))
    st.den[DEN_COLS.index("pol")] = 2
    st.den[DEN_COLS.index("all")] = 3
    st.mb_kl = torch.zeros(())
    pf = torch.zeros(4, NF)
    pf[:, PF_COLS.index("logp_b")] = torch.tensor([0.5, 0.2, 1.0, 1.0]).log()
    pf[:, PF_COLS.index("logp_ref")] = pf[:, PF_COLS.index("logp_b")]
    pf[:, PF_COLS.index("adv")] = torch.tensor([1.0, -0.5, 0.0, 0.0])
    pf[:2, PF_COLS.index("pol")] = 1
    pf[:3, PF_COLS.index("valid")] = 1
    pf[:, PF_COLS.index("log_legal")] = torch.tensor([3.0, 3.0, 2.0, 2.0]).log()
    pf[:, PF_COLS.index("t_keep")] = keep
    pf[:, PF_COLS.index("t_add")] = add
    pi = torch.tensor([[0, int(Kind.CARD_REWARD)], [1, int(Kind.COMBAT)], [0, int(Kind.REST)], [0, 0]])
    views = {"raw": torch.zeros(12, 1, dtype=torch.int16), "rel": torch.zeros(12, 3, dtype=torch.int64),
             "dest": torch.arange(4, 16), "act_mask": torch.isfinite(logits), "tok_mask": torch.ones(4, 1, dtype=torch.bool),
             "kid_mask": torch.empty(0, dtype=torch.bool), "var_mask": torch.empty(0, dtype=torch.bool)}
    st._body(_Res(BatchShape(4, 1, 3, 0, 0, 0, 12), torch.empty(0), views, pf, pi))
    assert model.logits.grad is not None
    return model.logits.grad.clone(), st.mb_kl.clone()


def test_fast_microbody_taste_gradient_and_off_identity() -> None:
    """The CUDA-graph micro body on CPU: neutral columns (1, 0) are bit-identical to taste off; on a steered
    CARD_REWARD row the gradient is that of log((1 - e) q(a) + e u)."""
    g0, k0 = _microbody(TasteConfig(), torch.ones(4), torch.zeros(4))
    g1, k1 = _microbody(TasteConfig("x", 0.5, 0.7), torch.ones(4), torch.zeros(4))
    assert torch.equal(g0, g1) and torch.equal(k0, k1)
    e, u = 0.7, 0.5
    g2, _ = _microbody(TasteConfig("x", 0.5, e), torch.tensor([1 - e, 1, 1, 1]), torch.tensor([e * u, 0, 0, 0]))
    z = torch.tensor([[2.0, -2.0, 1.0], [-1.0, 0.5, 2.0]], requires_grad=True)
    mix = (1 - torch.tensor([0.2, 0.0])[:, None]) * z.softmax(-1) + torch.tensor([0.2, 0.0])[:, None] / 3
    p = mix[torch.arange(2), torch.tensor([0, 1])]
    p = torch.stack([(1 - e) * p[0] + e * u, p[1]])
    ratio = p / torch.tensor([0.5, 0.2])
    adv = torch.tensor([1.0, -0.5])
    loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).sum() / 2
    want, = torch.autograd.grad(loss, z)
    torch.testing.assert_close(g2[:2], want, rtol=1e-5, atol=1e-7)
    assert not torch.allclose(g2[:1], g0[:1])


# ------------------------------------------------------------------ actor -> store -> shared memory -> learner


def taste_actor_run(tmp_path: Path, vocab: Vocab, net: SpireNet, n: int, lineup: str
                    ) -> tuple[list[Segment], list[dict]]:
    tok = Tokenizer(vocab)
    pol = TorchPolicy(net, tok, seed=7, exploration_by_kind=((int(Kind.CARD_REWARD), 0.1),))
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32), taste=True)
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
             for i in range(4)]
    taste = Taste(TasteConfig(placeholder_table(tmp_path / "t.json", vocab), 1.0, 0.7, p_thin=0.5, thin_deck=0),
                  vocab, (0, 1, 2, 3, 4))
    reps: list[dict] = []
    Actor(hosts, pol, tok, EpisodeSampler(EpisodeDist(lineups=((lineup, 1.0),)), vocab, 0),
          ActorConfig(report_interval=0.05), store=store, taste=taste, on_report=reps.append).run(max_decisions=n)
    return segs, reps


@pytest.mark.parametrize("lineup", ["SILENT", "IRONCLAD+SILENT+SILENT+IRONCLAD"])
def test_learner_reproduces_the_stored_behaviour(tmp_path: Path, vocab: Vocab, lineup: str) -> None:
    """Invariant 2 on real actor output: fake hosts, a CPU policy, every seat with a taste (half thin, Skip from any
    deck size); the records survive the shared-memory codec, and the learner's first minibatch (the whole batch,
    before any step) has ratio 1 on every step with the taste transform - and not without it."""
    torch.manual_seed(0)
    net = SpireNet(Tokenizer(vocab).layout, TINY)
    learner_net = copy.deepcopy(net)
    segs, reps = taste_actor_run(tmp_path, vocab, net, 1200, lineup)
    segs = [decode_segment(pack_segment(s)) for s in segs]
    keep = np.concatenate([s.t_keep for s in segs])
    add = np.concatenate([s.t_add for s in segs])
    kinds = np.concatenate([s.kinds for s in segs])
    assert (keep < 1).sum() > 4 and (add > 0).sum() > 2, ((keep < 1).sum(), (add > 0).sum())
    assert (kinds[keep < 1] == Kind.CARD_REWARD).all() and np.allclose(keep[keep < 1], 0.3) and (add[keep == 1] == 0).all()
    assert sum(len(r["taste"]["episodes"]) for r in reps) > 0
    tok = Tokenizer(vocab)
    n = sum(len(s) for s in segs)
    for on in (True, False):
        cfg = PPOConfig(minibatch=4 * n, micro_batch=256, epochs=1, exploration_by_kind=((int(Kind.CARD_REWARD), 0.1),),
                        taste=TasteConfig("x", 1.0, 0.7) if on else TasteConfig())
        lr = Learner(copy.deepcopy(learner_net), tok, cfg, device="cpu", fast=False)
        data = segs if on else [dataclasses.replace(s, t_keep=None, t_add=None) for s in segs]
        m = lr.update(data)
        if on:
            assert m["first_minibatch_kl"] < 1e-10 and m["clipfrac"] == 0.0, m
            assert m["taste"]["steps"] == int((keep < 1).sum()) and m["taste"]["target"]["n"] == int((add > 0).sum())
        else:
            assert m["first_minibatch_kl"] > 1e-6 and "taste" not in m


# ------------------------------------------------------------------ invariant 1: off = the reference tree


def _ref(tree: Path, p0: bool) -> dict:
    env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1",
           "TASTE_P0": "1" if p0 else ""}
    out = subprocess.run([sys.executable, "-B", str(REF)], env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_off_is_the_reference_tree() -> None:
    """SPIRE_HEAD_TREE = the py tree before taste (py-stage-h3): a fixed-seed actor run samples the same actions with
    the same log-probs and values, and one learner update gives the same parameters, with taste absent and with
    p_episode 0."""
    head = os.environ.get("SPIRE_HEAD_TREE")
    if not head:
        pytest.skip("SPIRE_HEAD_TREE (the reference py tree) not set")
    mine = Path(__file__).resolve().parents[1]
    ref, a, b = _ref(Path(head), False), _ref(mine, False), _ref(mine, True)
    assert ref["taste_module"] is False and a["taste_module"] is True and b["p0"] is True
    strip = ("taste_module", "p0")
    for x in (a, b):
        assert {k: v for k, v in x.items() if k not in strip} == {k: v for k, v in ref.items() if k not in strip}
    assert ref["decisions"] >= 2000


# ------------------------------------------------------------------ end to end (fake hosts, supervisor)


def test_training_run_reports_taste(tmp_path: Path, vocab: Vocab) -> None:
    from spire.metrics import read_jsonl
    from spire.train import Supervisor
    from test_train import small_config

    table = placeholder_table(tmp_path / "t.json", vocab)
    spec = {"file": table, "p_episode": {"SILENT": 0.5, "IRONCLAD": 0.25}, "eps": 0.7, "seed": 3, "p_thin": 0.5,
            "thin_deck": 0, "start_floor": 0}
    cfg = small_config(tmp_path, "taste", topology="threads", max_updates=3,
                       ppo={"minibatch": 256, "micro_batch": 256, "epochs": 1, "lr": 1e-3, "taste": spec},
                       learner={"device": "cpu", "iteration_samples": 1024, "micro_tokens": None,
                                "checkpoint_interval_s": 3600, "allow_ppo_config_change": True})
    assert Supervisor(cfg).run() == 0
    rows = list(read_jsonl(cfg.run_dir / "metrics.jsonl"))
    learner = [r for r in rows if r["kind"] == "learner"]
    train = [r for r in rows if r["kind"] == "train"]
    assert [r["version"] for r in learner] == [1, 2, 3] and all("taste" in r for r in learner)
    assert sum(r["taste"]["steps"] for r in learner) > 0 and all(set(r["taste"]) == {"steps", "target", "own", "off"}
                                                                for r in learner)
    last = train[-1]["taste"]
    groups = last["groups"]
    assert any("|none|" in k for k in groups) and any("|r0" in k or "|r1" in k for k in groups), groups
    assert all({"n", "win_rate", "mean_floor", "window_n", "active_steps", "family_taken", "thin_skips",
                "act1_death_share", "window_act1_death_share"} <= set(v) for v in groups.values())
    ev = [r for r in rows if r["kind"] == "event" and r.get("what") == "taste_table"]
    assert len(ev) == 1 and ev[0]["version"] == 1 and set(ev[0]["families"]) == {"IRONCLAD", "SILENT"}
    assert last["none_outcomes"] and all(k.count("|A") == 1 for k in last["none_outcomes"])
    assert last["base_pi_none"] and all(0 <= v["mean"] <= 1 for v in last["base_pi_none"].values())
    st = torch.load(cfg.run_dir / "ckpt" / "latest.pt", map_location="cpu", weights_only=True)
    assert st["config"]["taste"] == spec
