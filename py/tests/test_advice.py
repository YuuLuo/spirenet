"""Advice (spire.advice): where the rule fires, the exact policy-advantage offset, transport, metrics, config."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import struct
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ppo_stubs import StubNet, StubSample, StubTokenizer
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.advice import (NO_ADVICE, NO_JUDGE, Advisor, advice_metrics, advice_offset, advice_rules, habit_metrics,
                          habit_offset, habit_rules, parse_advice)
from spire.cycle import CycleBan
from spire.habit import _Names
from spire.ipc import ShmRing, decode_segment, pack_segment
from spire.learner import Learner, PPOConfig
from spire.metrics import read_jsonl
from spire.proto import Frame, parse_frame
from spire.rewards import RoomIds
from spire.rollout import EpisodeMeta, GAEConfig, RolloutConfig, RolloutStore, Segment, SegmentEnd, build_batch
from spire.schema import ActionCol, G, Kind, OptionCol, OptionKind, SeatCol, Table, Verb
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import CYCLE_BAN_SLOTS, Tokenizer, TokenizerConfig
from spire.train import REPO, Supervisor, config_dict, load_config
from spire.trainstep import PF_COLS
from spire.vocab import Vocab, make_synthetic_vocab
from test_train import small_config

SMITH = ("REST", "SMITH", 0.5, 0.05)
ROOT = Path(os.environ.get("SPIRE_REPO", REPO))  # the repository with bench/ and host/ (a private py copy: set it)
GOLDEN = ROOT / "bench" / "h5" / "golden4" / "frames4_nat.bin"
GOLDEN_VOCAB = ROOT / "host" / "dist" / "20261115-194451-h5-4p" / "vocab" / "vocab.json"
HEAD_REF = Path(__file__).with_name("advice_head_ref.py")


# =========================================================================== rules and config


def test_rules_validate_names_and_numbers(vocab: Vocab) -> None:
    (r,) = advice_rules([SMITH])
    assert (r.kind, r.option, r.hp_min, r.bonus, r.name) == (Kind.REST, "SMITH", 0.5, 0.05, "REST:SMITH")
    for bad, msg in ((("COMBAT", "SMITH", 0.5, 0.05), "kind must be one of"),
                     (("RESTING", "SMITH", 0.5, 0.05), "kind must be one of"),
                     (("REST", "SMITH", 1.5, 0.05), "hp_min"), (("REST", "SMITH", 0.5, -0.1), "bonus"),
                     (("REST", "SMITH", 0.5, math.nan), "bonus"), (("REST", "SMITH", 0.5), "expected")):
        with pytest.raises(AssertionError, match=msg):
            advice_rules([bad])
        with pytest.raises(AssertionError, match=msg):
            PPOConfig(advice=(bad,))  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="one rule per decision kind"):
        advice_rules([SMITH, ("REST", "HEAL", 0.0, 0.01)])
    with pytest.raises(AssertionError, match="is not a rest_option key"):
        Advisor(vocab, [("REST", "SMITHY", 0.5, 0.05)])
    Advisor(vocab, [SMITH])


def test_config_json_round_trip_and_supervisor_start_check(tmp_path: Path) -> None:
    cfg = small_config(tmp_path, "cfg", ppo={"advice": [["REST", "SMITH", 0.5, 0.05]]})
    assert cfg.ppo.advice == (SMITH,) and isinstance(cfg.ppo.advice[0][2], float)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config_dict(cfg)), encoding="utf-8")
    assert load_config(p).ppo.advice == (SMITH,)
    two = small_config(tmp_path, "two", ppo={"advice": [["REST", "SMITH", 0.75, 0.05, 0.03]]})  # the loader keeps ``below``
    assert two.ppo.advice == (("REST", "SMITH", 0.75, 0.05, 0.03),) and advice_rules(two.ppo.advice)[0].below == 0.03
    assert small_config(tmp_path, "plain").ppo.advice == ()
    with pytest.raises(AssertionError, match="is not a rest_option key"):
        Supervisor(small_config(tmp_path, "bad", ppo={"advice": [["REST", "SMITHY", 0.5, 0.05]]}))


def test_old_checkpoint_without_advice_loads_and_advice_can_change_on_resume(tmp_path: Path) -> None:
    tok = StubTokenizer()
    old = Learner(StubNet(tok.n_features), tok, PPOConfig(), device="cpu")
    st = old.state()
    del st["config"]["advice"]  # a checkpoint written before the field existed
    torch.save(st, tmp_path / "old.pt")
    Learner(StubNet(tok.n_features), tok, PPOConfig(), device="cpu").load(tmp_path / "old.pt")
    with pytest.raises(ValueError, match="config differs"):
        Learner(StubNet(tok.n_features), tok, PPOConfig(advice=(SMITH,)), device="cpu").load(tmp_path / "old.pt")
    new = Learner(StubNet(tok.n_features), tok, PPOConfig(advice=(SMITH,)), device="cpu")
    new.load(tmp_path / "old.pt", allow_config_change=True)
    new.save(tmp_path / "new.pt")
    annealed = PPOConfig(advice=(("REST", "SMITH", 0.5, 0.0),))
    Learner(StubNet(tok.n_features), tok, annealed, device="cpu").load(tmp_path / "new.pt", allow_config_change=True)


# =========================================================================== where the rule fires (real 4-seat frames)


def golden_frames() -> list[Frame]:
    if not GOLDEN.exists() or not GOLDEN_VOCAB.exists():
        pytest.skip(f"recorded 4-seat frames {GOLDEN} / vocab {GOLDEN_VOCAB} not present")
    frames = []
    with GOLDEN.open("rb") as fh:
        while head := fh.read(4):
            (size,) = struct.unpack("<I", head)
            frames.append(parse_frame(fh.read(size)))
    return frames


def rest_option(f: Frame, row: int) -> bool:
    """``row`` is a CHOOSE_OPTION row of a REST_OPTION."""
    a = f.actions[row]
    return bool(a[ActionCol.VERB] == Verb.CHOOSE_OPTION and a[ActionCol.SRC_TABLE] == Table.OPTIONS
                and f.table(Table.OPTIONS)[a[ActionCol.SRC_ROW], OptionCol.KIND] == OptionKind.REST_OPTION)


def reference_candidate(vocab: Vocab, f: Frame, banned: tuple[int, ...], action: int) -> tuple[int, float]:
    """The rule, written out row by row with names (independent of Advisor's vectorised code)."""
    if f.msg_type != 1 or Kind(f.gvals[G.KIND]) != Kind.REST or not rest_option(f, action):
        return NO_ADVICE  # only a choice among the rest options is advised (notes/advfix)
    opts = f.table(Table.OPTIONS)
    for row in range(f.n_actions):
        a = f.actions[row]
        if a[ActionCol.VERB] != Verb.CHOOSE_OPTION or a[ActionCol.SRC_TABLE] != Table.OPTIONS:
            continue
        o = opts[a[ActionCol.SRC_ROW]]
        if o[OptionCol.KIND] == OptionKind.REST_OPTION and vocab.name("rest_option", int(o[OptionCol.KEY_ID])) == "SMITH":
            if row in banned:
                return NO_ADVICE
            seat = f.table(Table.SEATS)[f.gvals[G.ACTING_SEAT]]
            return row, int(seat[SeatCol.HP]) / int(seat[SeatCol.MAX_HP])
    return NO_ADVICE


def test_rule_fires_exactly_on_rest_frames_offering_smith() -> None:
    frames = golden_frames()
    vocab = Vocab.load(GOLDEN_VOCAB)
    assert all(f.vocab_hash == vocab.hash for f in frames[:50])
    advisor = Advisor(vocab, [SMITH])
    obs = [f for f in frames if f.msg_type == 1]
    got = [[advisor.candidate(f, (), a) for a in range(f.n_actions)] for f in obs]
    assert got == [[reference_candidate(vocab, f, (), a) for a in range(f.n_actions)] for f in obs]
    fired = []  # (frame, its record): one per frame, the same for every rest-option action
    for f, recs in zip(obs, got):
        on = {c for c in recs if c[0] >= 0}
        assert len(on) <= 1
        if on:
            fired.append((f, on.pop()))
    rest = [f for f in obs if f.gvals[G.KIND] == Kind.REST]
    assert len(frames) == 19389 and len(rest) == 162 and len(fired) == 85  # bench/h5/golden4, 4-seat natural games
    assert all(len(f.table(Table.SEATS)) == 4 for f, _ in fired)
    hp = np.array([c[1] for _, c in fired], np.float32)
    assert (hp < 0.5).sum() > 5 and ((hp >= 0.5) & (hp < 0.75)).sum() > 5 and ((hp >= 0.75) & (hp < 1)).sum() > 5
    assert (hp == 1.0).sum() >= 3
    for f, (row, _) in fired[:20]:  # a banned SMITH row is not advised; a ban elsewhere changes nothing
        assert all(advisor.candidate(f, (row,), a) == NO_ADVICE for a in range(f.n_actions))
        other = next(i for i in range(f.n_actions) if i != row)
        assert advisor.candidate(f, (other,), row) == advisor.candidate(f, (), row)


def test_cycle_banned_smith_is_not_advised() -> None:
    frames = golden_frames()
    vocab = Vocab.load(GOLDEN_VOCAB)
    advisor = Advisor(vocab, [SMITH])
    f, row = next((f, a) for f in frames if f.msg_type == 1 for a in range(f.n_actions)
                  if advisor.candidate(f, (), a)[0] == a)
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, mend_target=True, coop_features=True))
    cycle = CycleBan(vocab)
    cycle.observe(f, tok.encode(f))
    assert cycle.active_bans == ()
    cycle.record(f, row)
    cycle.observe(f, tok.encode(f))  # the same public state again: the SMITH that left it closed a cycle
    assert cycle.active_bans == (row,)
    assert all(advisor.candidate(f, cycle.active_bans, a) == NO_ADVICE for a in range(f.n_actions))


# =========================================================================== actor recording (fake host)


class _LoggingStore(RolloutStore):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.log: list[tuple[int, tuple[int, ...], tuple[int, float] | None, int]] = []
        self.judged_log: list[tuple[int, int] | None] = []

    def add_step(self, env_id: int, facts: Any, sample: Any, action: int, logp: float, value: float,
                 policy_version: int, advice: tuple[int, float] | None = None,
                 judged: tuple[int, int] | None = None) -> bool:
        slots = sample.history[-CYCLE_BAN_SLOTS:]
        self.log.append((env_id, tuple(int(x) - 1 for x in slots[slots > 0]), advice, action))
        self.judged_log.append(judged)
        return super().add_step(env_id, facts, sample, action, logp, value, policy_version, advice, judged)


def actor_run(vocab: Vocab, advice: tuple[tuple[Any, ...], ...], n: int = 3000,
              seg_len: int = 32) -> tuple[_LoggingStore, list[tuple[int, Frame, int]], list[Segment]]:
    segs: list[Segment] = []
    old, habits = parse_advice(advice)
    store = _LoggingStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=seg_len),
                          advice=bool(old), judged=bool(habits))
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=150, hashes=False, abort_prob=0.0))
             for i in range(4)]
    decisions: list[tuple[int, Frame, int]] = []
    tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, commit_select=True))
    Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(lineups=(("IRONCLAD+SILENT", 1.0),)), vocab, 0),
          ActorConfig(), store=store, on_decision=lambda e, f, a: decisions.append((e, f, a)),
          advice=advice).run(max_decisions=n)
    return store, decisions, segs


def n_candidates(segs: list[Segment]) -> int:
    rows = [s.advice_row for s in segs]
    assert all(r is not None for r in rows)
    return sum(int((r >= 0).sum()) for r in rows if r is not None)


def test_actor_records_the_candidate_of_every_step(vocab: Vocab) -> None:
    store, decisions, segs = actor_run(vocab, (SMITH,))
    assert len(store.log) == len(decisions) > 2000
    n_fired = 0
    for (env, bans, advice, action), (env2, f, action2) in zip(store.log, decisions):
        assert (env, action) == (env2, action2)
        assert advice == reference_candidate(vocab, f, bans, action)
        n_fired += advice[0] >= 0
    assert n_fired > 20
    for s in segs:  # released segments carry the record; candidates only at REST steps
        assert s.advice_row is not None and s.advice_hp is not None
        assert s.advice_row.dtype == np.int16 and s.advice_hp.dtype == np.float32 and len(s.advice_row) == len(s)
        assert (s.kinds[s.advice_row >= 0] == Kind.REST).all() and (s.advice_hp[s.advice_row < 0] == 0).all()
    assert n_candidates(segs) > 10


def test_store_and_actor_agree_on_advice(vocab: Vocab) -> None:
    store = RolloutStore(lambda s: None, RoomIds.from_vocab(vocab), advice=False)
    with pytest.raises(AssertionError, match="iff the store records advice"):
        Actor([FakeHost(vocab, seed=0)], UniformPolicy(0), Tokenizer(vocab), EpisodeSampler(EpisodeDist(), vocab, 0),
              ActorConfig(), store=store, advice=(SMITH,))
    store.begin_episode(0, (2,))
    with pytest.raises(AssertionError, match="iff the store records advice"):
        store.add_step(0, None, None, 0, 0.0, 0.0, 0, (1, 0.5))  # type: ignore[arg-type]


def test_without_rules_segments_carry_no_record(vocab: Vocab) -> None:
    store, _, segs = actor_run(vocab, (), n=400)
    assert all(a is None for _, _, a, _ in store.log)
    assert segs and all(s.advice_row is None and s.advice_hp is None for s in segs)
    assert build_batch(segs, GAEConfig(), 0).advice_row is None


# =========================================================================== transport


def assert_same_advice(a: Segment, b: Segment) -> None:
    assert np.array_equal(a.actions, b.actions) and a.meta == b.meta and a.start == b.start
    if a.advice_row is None:
        assert b.advice_row is None and b.advice_hp is None
        return
    assert b.advice_row is not None and b.advice_hp is not None and a.advice_hp is not None
    assert b.advice_row.dtype == np.int16 and b.advice_hp.dtype == np.float32
    assert np.array_equal(a.advice_row, b.advice_row) and np.array_equal(a.advice_hp, b.advice_hp)


def test_blob_round_trip_with_and_without_advice(vocab: Vocab) -> None:
    for advice in ((SMITH,), ()):
        _, _, segs = actor_run(vocab, advice, n=800)
        for s in segs:
            assert_same_advice(s, decode_segment(pack_segment(s)))


def _producer(spec: Any, conn: Any) -> None:
    vocab = Vocab(make_synthetic_vocab())
    ring = ShmRing.attach(spec, conn)
    _, _, segs = actor_run(vocab, (SMITH,), n=1200)
    for s in segs:
        assert ring.put(s)
    conn.send(("done", len(segs)))
    ring.close()


def test_segments_from_another_process_carry_the_record(vocab: Vocab) -> None:
    ctx = mp.get_context("spawn")
    owner, spec = ShmRing.create(64 << 20)
    rx, tx = ctx.Pipe(duplex=False)
    cons = ShmRing.attach(spec, rx)
    p = ctx.Process(target=_producer, args=(spec, tx))
    p.start()
    try:
        got = []
        while True:
            assert rx.poll(180)
            msg = rx.recv()
            if msg[0] == "done":
                break
            got.append(cons.get(msg))
        p.join(30)
        assert p.exitcode == 0 and msg[1] == len(got) > 5
        _, _, ref = actor_run(vocab, (SMITH,), n=1200)  # the same deterministic fake run
        assert len(ref) == len(got)
        for a, b in zip(ref, got):
            assert_same_advice(a, b)
        assert n_candidates(got) > 0
    finally:
        if p.is_alive():
            p.kill()
        cons.close()
        owner.close()


# =========================================================================== learner: offset, targets, metrics


def synthetic_segments(seed: int = 0, n_segments: int = 12, length: int = 40, advice: bool = True) -> list[Segment]:
    """REST / COMBAT / EVENT steps; at REST steps a candidate row (70%) with HP fractions on the bucket edges."""
    rng = np.random.default_rng(seed)
    tok = StubTokenizer()
    out = []
    for i in range(n_segments):
        kinds = rng.choice(np.array([Kind.REST, Kind.COMBAT, Kind.EVENT], np.int8), length)
        n_legal = rng.integers(1, 5, length).astype(np.int16)
        actions = (rng.random(length) * n_legal).astype(np.int64)
        row = np.where((kinds == Kind.REST) & (rng.random(length) < 0.7), (rng.random(length) * n_legal), -1)
        row = row.astype(np.int16)
        pick = (row >= 0) & (rng.random(length) < 0.4)
        actions[pick] = row[pick]
        hp = np.where(row >= 0, rng.choice(np.array([0.2, 0.4999, 0.5, 0.7, 0.75, 0.9, 1.0], np.float32), length), 0)
        x = rng.standard_normal((length, tok.n_features)).astype(np.float32)
        out.append(Segment(
            EpisodeMeta(i % 3, i, (2, 2, 2, 2), 9), 0, [StubSample(x[t], int(n_legal[t])) for t in range(length)],
            actions, np.log(1.0 / n_legal).astype(np.float32), rng.standard_normal(length).astype(np.float32) * 0.1,
            kinds, n_legal, np.zeros(length, np.int64), end=SegmentEnd.TERMINAL if i % 2 else SegmentEnd.CUT,
            bootstrap_value=0.3, rewards=(rng.random(length) * 0.02).astype(np.float32),
            combat_id=np.where(kinds == Kind.COMBAT, i, -1).astype(np.int32), win=np.full(length, i % 2, np.float32),
            win_mask=np.ones(length, bool), floors_left=np.full(length, 10.0, np.float32),
            hp_post=np.full(length, 0.5, np.float32), hp_mask=kinds == Kind.COMBAT,
            advice_row=row if advice else None, advice_hp=hp.astype(np.float32) if advice else None))
    return out


def _learner(advice: tuple[tuple[str, str, float, float], ...], **kw: Any) -> Learner:
    torch.manual_seed(0)
    tok = StubTokenizer()
    cfg = PPOConfig(epochs=1, minibatch=128, micro_batch=64, lr=1e-3, advice=advice, seed=0, **kw)
    return Learner(StubNet(tok.n_features), tok, cfg, device="cpu", size_fn=StubTokenizer.size)


def test_policy_advantage_differs_by_exactly_the_bonus_on_advised_steps() -> None:
    segs = synthetic_segments()
    rb = build_batch(segs, GAEConfig(), 0)
    assert rb.advice_row is not None and rb.advice_hp is not None
    plain, advised = _learner((), adv_norm=False), _learner((SMITH,), adv_norm=False)
    a0, a1 = plain._policy_advantages(rb), advised._policy_advantages(rb)
    on = (rb.kinds == Kind.REST) & (rb.advice_row >= 0) & (rb.advice_hp >= np.float32(0.5))
    chosen = rb.actions == rb.advice_row
    assert on.sum() > 30 and (on & chosen).sum() > 5 and (on & ~chosen).sum() > 5
    assert ((rb.advice_row >= 0) & ~on).sum() > 5  # candidates below hp_min stay plain
    b = np.float32(0.05)
    off = np.where(on, np.where(chosen, b, -b), np.float32(0)).astype(np.float32)
    assert np.array_equal(advice_offset(advised._advice, rb.kinds, rb.actions, rb.advice_row, rb.advice_hp), off)
    assert np.array_equal(a1, a0 + off) and np.array_equal(a1[~on], a0[~on])
    assert np.allclose(a1[on] - a0[on], np.where(chosen[on], 0.05, -0.05), atol=1e-7)
    # the training columns: only the advantage column differs; returns, values and every other target are equal
    for norm in (False, True):
        p0, p1 = _learner((), adv_norm=norm), _learner((SMITH,), adv_norm=norm)
        pf0, pi0, m0 = p0._columns(rb)
        pf1, pi1, m1 = p1._columns(rb)
        col = PF_COLS.index("adv")
        rest = [c for c in range(len(PF_COLS)) if c != col]
        assert np.array_equal(pf0[:, rest], pf1[:, rest]) and np.array_equal(pi0, pi1)
        assert all(np.array_equal(m0[k], m1[k]) for k in m0)
        if not norm:
            assert np.array_equal(pf1[:, col], a0 + off) and np.array_equal(pf0[:, col], a0)
        else:  # the offset enters before the normalisation over the policy steps
            pol = rb.n_legal > 1
            x = a0 + off
            assert np.array_equal(pf1[:, col], (x - x[pol].mean()) / (x[pol].std() + 1e-8))
        t0, t1 = p0._tensors(rb), p1._tensors(rb)
        for name in t0.__dataclass_fields__:
            same = torch.equal(getattr(t0, name), getattr(t1, name))
            assert same == (name != "adv"), name
        assert np.array_equal(t1.adv.numpy(), pf1[:, col])


def test_returns_and_value_targets_do_not_depend_on_advice() -> None:
    with_record, without = build_batch(synthetic_segments(), GAEConfig(), 0), \
        build_batch(synthetic_segments(advice=False), GAEConfig(), 0)
    for name in ("advantages", "returns", "values", "rewards", "win", "floors_left", "hp_post", "actions", "logp"):
        assert np.array_equal(getattr(with_record, name), getattr(without, name)), name


def _update(advice: tuple[tuple[str, str, float, float], ...], segs: list[Segment]) -> tuple[dict, list[torch.Tensor]]:
    lr = _learner(advice)
    m = lr.update(segs)
    return m, [p.detach().clone() for p in lr.model.state_dict().values()]


def _strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("time_") and k not in ("samples_per_s", "lag_s", "advice")}


def test_zero_bonus_or_no_advised_step_is_the_plain_update() -> None:
    segs = synthetic_segments()
    m0, p0 = _update((), synthetic_segments(advice=False))
    for advice, data in (((("REST", "SMITH", 0.5, 0.0),), segs), (((("REST", "SMITH", 1.0, 0.05),),
                                                                   [replace(s, advice_hp=np.zeros(len(s), np.float32))
                                                                    for s in segs]))):
        m1, p1 = _update(advice, data)
        assert _strip(m0) == _strip(m1) and all(torch.equal(a, b) for a, b in zip(p0, p1))
    m2, p2 = _update((SMITH,), segs)
    assert m2["loss_pi"] != m0["loss_pi"] and not all(torch.equal(a, b) for a, b in zip(p0, p2))
    assert m2["explained_variance"] == m0["explained_variance"] and m2["return_mean"] == m0["return_mean"]


def test_learner_metrics() -> None:
    kinds = np.array([Kind.REST] * 9 + [Kind.COMBAT], np.int8)
    row = np.array([1, 1, 2, 0, 1, 1, 0, -1, 1, 3], np.int16)  # step 9: a COMBAT step never counts
    hp = np.array([0.5, 0.6, 0.75, 0.9, 1.0, 1.0, 0.3, 0.0, 0.49, 1.0], np.float32)
    actions = np.array([1, 0, 2, 1, 1, 1, 0, 2, 0, 3], np.int64)
    adv = np.array([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 7.0, 8.0, 9.0, 10.0], np.float32)
    (m,) = advice_metrics(advice_rules([SMITH]), kinds, actions, row, hp, adv).values()
    assert m["steps"] == 6 and m["chosen"] == round(4 / 6, 4)
    assert m["steps_by_hp"] == {"lo": 2, "hi": 2, "full": 2}
    assert m["chosen_by_hp"] == {"lo": 0.5, "hi": 0.5, "full": 1.0}
    assert m["raw_adv_chosen"] == pytest.approx((0.1 + 0.3 + 0.5 + 0.6) / 4)
    assert m["raw_adv_other"] == pytest.approx((-0.2 + 0.4) / 2)
    assert m["below_steps"] == 2 and m["below_chosen"] == 0.5
    off = advice_offset(advice_rules([SMITH]), kinds, actions, row, hp)
    assert off.tolist() == pytest.approx([0.05, -0.05, 0.05, -0.05, 0.05, 0.05, 0, 0, 0, 0])
    # two-sided: below hp_min the offset is against the option (steps 6 and 8; step 7 offers no option)
    two = advice_offset(advice_rules([(*SMITH, 0.02)]), kinds, actions, row, hp)
    assert two.tolist() == pytest.approx([0.05, -0.05, 0.05, -0.05, 0.05, 0.05, -0.02, 0, 0.02, 0])
    (empty,) = advice_metrics(advice_rules([SMITH]), kinds[:0], actions[:0], row[:0], hp[:0], adv[:0]).values()
    assert empty["steps"] == 0 and empty["chosen"] is None and empty["raw_adv_chosen"] is None


def test_update_reports_advice_metrics() -> None:
    m, _ = _update((SMITH,), synthetic_segments())
    a = m["advice"]["REST:SMITH"]
    assert a["steps"] > 30 and 0 < a["chosen"] < 1 and a["below_steps"] > 5 and a["bonus"] == 0.05
    assert "advice" not in _update((), synthetic_segments(advice=False))[0]


def _run_head_ref(tree: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"}
    out = subprocess.run([sys.executable, "-B", str(HEAD_REF)], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_no_rule_update_is_bit_identical_to_the_reference_tree() -> None:
    """SPIRE_HEAD_TREE = the py tree of the commit before this patch (e.g. a checkout of the previous commit)."""
    head = os.environ.get("SPIRE_HEAD_TREE")
    if not head:
        pytest.skip("SPIRE_HEAD_TREE (the reference py tree) not set")
    ref, mine = _run_head_ref(Path(head)), _run_head_ref(Path(__file__).resolve().parents[1])
    assert ref["tree"] != mine["tree"] and ref["advice_module"] is False and mine["advice_module"] is True
    assert ref["metrics"] == mine["metrics"] and ref["params"] == mine["params"] and ref["samples"] >= 512


# =========================================================================== end to end (fake hosts, both topologies)


def _learner_rows(run: Path) -> list[dict]:
    return [r for r in read_jsonl(run / "metrics.jsonl") if r["kind"] == "learner"]


@pytest.mark.parametrize("topology", ["threads", "processes"])
def test_training_runs_report_advice(tmp_path: Path, topology: str) -> None:
    cfg = small_config(tmp_path, f"adv-{topology}", topology=topology, max_updates=3,
                       ppo={"minibatch": 256, "micro_batch": 256, "epochs": 1, "lr": 1e-3,
                            "advice": [["REST", "SMITH", 0.5, 0.05]]},
                       tokenizer={"cycle_ban": True, "commit_select": True},
                       learner={"device": "cpu", "iteration_samples": 1024, "micro_tokens": None,
                                "checkpoint_interval_s": 3600, "allow_ppo_config_change": True})
    assert Supervisor(cfg).run() == 0
    rows = _learner_rows(cfg.run_dir)
    assert [r["version"] for r in rows] == [1, 2, 3]
    adv = [r["advice"]["REST:SMITH"] for r in rows]
    assert sum(a["steps"] for a in adv) > 0 and all(a["bonus"] == 0.05 for a in adv)
    if topology == "processes":  # anneal: edit the bonus, resume
        raw = json.loads((cfg.run_dir / "config.json").read_text(encoding="utf-8"))
        raw["ppo"]["advice"] = [["REST", "SMITH", 0.5, 0.0]]
        raw["max_updates"] = 4
        (cfg.run_dir / "config.json").write_text(json.dumps(raw), encoding="utf-8")
        assert Supervisor(load_config(cfg.run_dir / "config.json"), resume=True).run() == 0
        rows = _learner_rows(cfg.run_dir)
        assert [r["version"] for r in rows] == [1, 2, 3, 4] and rows[-1]["advice"]["REST:SMITH"]["bonus"] == 0.0


# =========================================================================== habit rules ([habit spec, bonus])

POTION = ("potion_hold", 0.05)
POTION_R1 = ("potion_r1:Elite,Boss", 0.05)
SMITH_H = ("smith:0.5", 0.05)
TREE_REF = Path(__file__).with_name("advice_tree_ref.py")


def test_habit_rules_validate() -> None:
    old, new = parse_advice([("REST", "SMITH", 0.75, 0.05, 0.05), POTION, ("enrage:none", 0.05)])
    assert [r.name for r in old] == ["REST:SMITH"] and [r.name for r in new] == ["potion_hold", "enrage:none"]
    assert new[0].kinds == {Kind.COMBAT} and new[0].combat and not parse_advice([("smith:0.75", 0.1)])[1][0].combat
    assert parse_advice([("shop_relic", 0)])[1][0].bonus == 0.0 and isinstance(parse_advice([("shop_relic", 0)])[1][0].bonus, float)
    for bad, msg in (((("remove_basic", 0.05),), "evaluation-only"), ((("combo", 0.05),), "one habit per rule"),
                     ((("smith:0.5+potion_hold", 0.05),), "one habit per rule"), ((("nope", 0.05),), "unknown habit"),
                     ((("potion_hold", -0.1),), "bonus"), ((("potion_hold", math.inf),), "bonus"),
                     ((POTION, ("potion_boss", 0.05)), "can judge one step"),
                     ((("smith:0.75", 0.05), ("smith_pre:0.75:1.0", 0.05)), "can judge one step"),
                     ((("shop_relic", 0.05), ("shop_nocard", 0.05)), "can judge one step"),
                     (((*SMITH, 0.05), ("smith:0.75", 0.05)), "both judge REST"),
                     ((("shop_nocard", 0.05), ("shop_nopotion", 0.05)), "can judge one step"),
                     ((POTION_R1, POTION), "can judge one step"),
                     ((("potion_boss", 0.05), POTION_R1), "can judge one step"),
                     ((("potion_r1:Elite,Shop", 0.05),), "rooms must be"), ((("potion_r1:", 0.05),), "rooms must be"),
                     ((("potion_r1:Boss,Boss", 0.05),), "rooms must be")):
        with pytest.raises((AssertionError, ValueError), match=msg):
            parse_advice(bad)
        with pytest.raises((AssertionError, ValueError), match=msg):
            PPOConfig(advice=bad)  # type: ignore[arg-type]
    PPOConfig(advice=((*SMITH, 0.05), POTION, ("enrage:block", 0.05), ("no_elite:1-2", 0.01)))  # kinds disjoint
    # potion_r1 (COMBAT, USE_POTION only): with the REST rule and enrage (disjoint verbs), positive-only
    old, new = parse_advice([("REST", "SMITH", 0.75, 0.05, 0.05), POTION_R1, ("enrage:none", 0.05)])
    assert [r.name for r in old] == ["REST:SMITH"] and [r.name for r in new] == ["potion_r1:Elite,Boss", "enrage:none"]
    assert new[0].kinds == {Kind.COMBAT} and new[0].combat and new[0].habit.VERBS == {int(Verb.USE_POTION)}
    assert new[0].habit.rooms == ("Elite", "Boss")


def test_mixed_config_json_round_trip_and_start_check(tmp_path: Path) -> None:
    spec = [["REST", "SMITH", 0.75, 0.05, 0.05], ["potion_hold", 0.05], ["no_elite:1-2", 0.02]]
    cfg = small_config(tmp_path, "mixed", ppo={"advice": spec})
    assert cfg.ppo.advice == (("REST", "SMITH", 0.75, 0.05, 0.05), ("potion_hold", 0.05), ("no_elite:1-2", 0.02))
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config_dict(cfg)), encoding="utf-8")
    back = load_config(p)
    assert back.ppo.advice == cfg.ppo.advice and parse_advice(back.ppo.advice) == parse_advice(cfg.ppo.advice)
    assert json.loads(p.read_text(encoding="utf-8"))["ppo"]["advice"] == spec
    # names the vocab lacks fail at start-up (the synthetic vocab has no ENRAGE_POWER; catalog "none": no card types)
    with pytest.raises(AssertionError, match="lacks ENRAGE_POWER"):
        Supervisor(small_config(tmp_path, "enrage", ppo={"advice": [["enrage:none", 0.05]]}, catalog="synthetic"))
    with pytest.raises(AssertionError, match="both judge REST"):
        small_config(tmp_path, "overlap", ppo={"advice": [["REST", "SMITH", 0.75, 0.05], ["smith:0.75", 0.05]]})


def test_old_checkpoints_load_into_habit_advice(tmp_path: Path) -> None:
    """A checkpoint of a live run (old-format rule in its config) resumes with the same rules, or with habit rules added
    (``allow_config_change``, as when a run's advice is edited)."""
    tok = StubTokenizer()
    live = Learner(StubNet(tok.n_features), tok, PPOConfig(advice=((*SMITH[:2], 0.75, 0.05, 0.05),)), device="cpu")
    live.save(tmp_path / "live.pt")
    Learner(StubNet(tok.n_features), tok, PPOConfig(advice=((*SMITH[:2], 0.75, 0.05, 0.05),)), device="cpu").load(
        tmp_path / "live.pt")
    mixed = PPOConfig(advice=(("REST", "SMITH", 0.75, 0.05, 0.05), POTION, ("enrage:none", 0.05)))
    with pytest.raises(ValueError, match="config differs"):
        Learner(StubNet(tok.n_features), tok, mixed, device="cpu").load(tmp_path / "live.pt")
    new = Learner(StubNet(tok.n_features), tok, mixed, device="cpu")
    new.load(tmp_path / "live.pt", allow_config_change=True)
    new.save(tmp_path / "mixed.pt")
    st = torch.load(tmp_path / "mixed.pt", weights_only=False)
    assert parse_advice(st["config"]["advice"]) == parse_advice(mixed.advice)


# ------------------------------------------------------------------ card rewards: prefs + deck_cap


def card_names(vocab: Vocab, residue: int) -> str:
    """Comma-separated synthetic card names whose vocab id is ``residue`` mod 3 (from id 2: not PAD / UNK)."""
    return ",".join(vocab["card"].entries[i] for i in range(2, vocab.size("card")) if i % 3 == residue)


def test_prefs_and_deck_cap_rules_validate_and_round_trip(tmp_path: Path) -> None:
    """prefs + deck_cap may share a config (precedence, spire.advice.YIELDS) in either order; two of either may not.
    The inline table survives json -> config -> json -> checkpoint unchanged; card names fail at start-up."""
    vocab = Vocab(make_synthetic_vocab())
    prefs = f"prefs:list:{card_names(vocab, 0)}/{card_names(vocab, 1)}"
    for spec in ([[prefs, 0.05], ["deck_cap:26", 0.03]], [["deck_cap:26", 0.03], [prefs, 0.05]]):
        _, new = parse_advice(spec)
        assert [r.spec for r in new] == [s for s, _ in spec]
        PPOConfig(advice=tuple(tuple(x) for x in [*spec, ("REST", "SMITH", 0.75, 0.05, 0.05), POTION]))
    for bad, msg in (([[prefs, 0.05], ["prefs:list:A/B", 0.05]], "can judge one step"),
                     ([["deck_cap:20", 0.05], ["deck_cap:26", 0.05]], "can judge one step"),
                     ([["prefs:list:A/A", 0.05]], "both PREFER and AVOID"), ([["prefs:list:A", 0.05]], "expected file"),
                     ([["prefs:list:A/B+deck_cap:20", 0.05]], "one habit per rule")):
        with pytest.raises(AssertionError, match=msg):
            PPOConfig(advice=bad)  # type: ignore[arg-type]
    spec = [["REST", "SMITH", 0.75, 0.05, 0.05], [prefs, 0.05], ["deck_cap:26", 0.03]]
    cfg = small_config(tmp_path, "cards", ppo={"advice": spec})
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config_dict(cfg)), encoding="utf-8")
    back = load_config(p)
    assert json.loads(p.read_text(encoding="utf-8"))["ppo"]["advice"] == spec
    assert back.ppo.advice == cfg.ppo.advice and parse_advice(back.ppo.advice) == parse_advice(spec)
    tok = StubTokenizer()
    lr = Learner(StubNet(tok.n_features), tok, back.ppo, device="cpu")
    lr.save(tmp_path / "cards.pt")
    st = torch.load(tmp_path / "cards.pt", weights_only=False)
    assert [list(x) for x in st["config"]["advice"]] == spec
    Learner(StubNet(tok.n_features), tok, back.ppo, device="cpu").load(tmp_path / "cards.pt")
    Supervisor(cfg)  # every name resolves on the synthetic vocab
    with pytest.raises(AssertionError, match="vocab card lacks NOT_A_CARD"):
        Supervisor(small_config(tmp_path, "unknown", ppo={"advice": [["prefs:list:NOT_A_CARD/", 0.05]]}))


def test_actor_records_prefs_and_deck_cap_with_precedence(vocab: Vocab) -> None:
    """Fake hosts, prefs + deck_cap + the old REST rule: every step's record is deck_cap's judgement where deck_cap
    applies, else prefs's; both rules and both signs occur."""
    prefs = f"prefs:list:{card_names(vocab, 0)}/{card_names(vocab, 1)}"
    rules = (("REST", "SMITH", 0.75, 0.05, 0.05), (prefs, 0.05), ("deck_cap:16", 0.03))
    store, decisions, segs = actor_run(vocab, rules, n=6000)
    names = _Names(vocab, None)
    _, (hp, hc) = parse_advice(rules)
    n = Counter()
    for (env, bans, advice, action), judged, (_, f, _) in zip(store.log, store.judged_log, decisions):
        g = f.gvals
        if hc.habit.applies(names, f, g, bans):
            s, i = hc.habit.judge(names, f, g, action, bans), 1
        else:
            s, i = hp.habit.judge(names, f, g, action, bans), 0
        assert judged == ((i, s) if s else NO_JUDGE)
        n[judged] += 1
    assert min(n[(0, 1)], n[(0, -1)], n[(1, 1)], n[(1, -1)]) > 5, n
    for s in segs:
        assert (s.kinds[s.advice_rule >= 0] == Kind.CARD_REWARD).all()


# ------------------------------------------------------------------ recorded frames: the old rule == smith:0.75


def test_smith_habit_rule_gives_the_old_two_sided_offsets_on_recorded_frames() -> None:
    """On every recorded REST frame, every action and with / without a banned SMITH row: the offset of the habit rule
    ``["smith:0.75", b]`` equals that of the live runs' ``["REST", "SMITH", 0.75, b, b]`` (0.75 is exact in binary;
    for other thresholds the old rule compares float32 fractions, the habit ``hp >= t * max_hp``)."""
    frames = golden_frames()
    vocab = Vocab.load(GOLDEN_VOCAB)
    b = 0.05
    old = Advisor(vocab, [("REST", "SMITH", 0.75, b, b)])
    new = Advisor(vocab, [("smith:0.75", b)])
    rest = [f for f in frames if f.msg_type == 1 and f.gvals[G.KIND] == Kind.REST]
    kinds, actions, rows, hps, rule, sign = [], [], [], [], [], []
    option = []  # the action is a rest option (the only rows either path may grade, notes/advfix)
    for f in rest:
        row = max(old.candidate(f, (), a)[0] for a in range(f.n_actions))
        for banned in ((), (row,)) if row >= 0 else ((),):
            for a in range(f.n_actions):
                if a in banned:
                    continue
                r, hp = old.candidate(f, banned, a)
                i, s = new.judge(f, banned, a, int(f.actions[a, ActionCol.VERB]))
                kinds.append(Kind.REST), actions.append(a), rows.append(r), hps.append(hp), rule.append(i), sign.append(s)
                option.append(rest_option(f, a))
    a_old = advice_offset(old.rules, np.array(kinds, np.int8), np.array(actions, np.int64), np.array(rows, np.int16),
                          np.array(hps, np.float32))
    a_new = habit_offset(new.habits, np.array(rule, np.int8), np.array(sign, np.int8))
    assert len(a_old) > 300 and (a_old != 0).sum() > 200 and (a_old > 0).sum() > 50 and (a_old < 0).sum() > 50
    assert np.array_equal(a_old, a_new)
    assert not a_old[~np.array(option)].any() and (~np.array(option)).sum() > 20


# ------------------------------------------------------------------ actor recording (fake host)


def reference_judged(vocab: Vocab, rules: tuple[tuple[Any, ...], ...], f: Frame, bans: tuple[int, ...],
                     action: int) -> tuple[int, int]:
    """The habit rules' judgement through the habits themselves (independent of Advisor's kind table)."""
    names = _Names(vocab, None)
    out = NO_JUDGE
    for i, r in enumerate(parse_advice(rules)[1]):
        s = r.habit.judge(names, f, f.gvals, action, bans)
        if s:
            assert out == NO_JUDGE
            out = (i, s)
    return out


def test_actor_records_the_habit_judgement_of_every_step(vocab: Vocab) -> None:
    rules = ((*SMITH[:2], 0.75, 0.05, 0.05), POTION, ("shop_nocard", 0.05))  # old REST rule + two habit rules
    store, decisions, segs = actor_run(vocab, rules)
    assert len(store.judged_log) == len(decisions) > 2000
    n = Counter()
    for (env, bans, advice, action), judged, (_, f, _) in zip(store.log, store.judged_log, decisions):
        assert advice == reference_candidate(vocab, f, bans, action)
        assert judged == reference_judged(vocab, rules, f, bans, action)
        n[judged] += 1
    assert n[(0, -1)] > 10 and n[(1, -1)] > 5, n  # potions used in Monster rooms, cards bought in shops
    for s in segs:
        assert s.advice_rule is not None and s.advice_sign is not None and s.advice_row is not None
        assert s.advice_rule.dtype == np.int8 and s.advice_sign.dtype == np.int8 and len(s.advice_rule) == len(s)
        assert ((s.advice_rule >= 0) == (s.advice_sign != 0)).all()
        assert (s.kinds[s.advice_rule == 0] == Kind.COMBAT).all() and (s.kinds[s.advice_rule == 1] == Kind.SHOP).all()
    only_habits, _, segs2 = actor_run(vocab, (POTION,), n=600)
    assert all(a is None for _, _, a, _ in only_habits.log) and all(s.advice_row is None for s in segs2)
    assert all(s.advice_rule is not None for s in segs2)


def test_actor_records_the_rollout_rule_set(vocab: Vocab) -> None:
    """The REST rule + potion_r1 (fake hosts, every combat room): the old record only at rest-option choices, potion_r1
    +1 at round-1 potion uses and never -1."""
    rules = (("REST", "SMITH", 0.75, 0.05, 0.05), ("potion_r1:Monster,Elite,Boss", 0.05))
    store, decisions, segs = actor_run(vocab, rules, n=6000)
    n = Counter()
    for (env, bans, advice, action), judged, (_, f, _) in zip(store.log, store.judged_log, decisions):
        assert advice == reference_candidate(vocab, f, bans, action)
        assert judged == reference_judged(vocab, rules, f, bans, action)
        assert advice[0] < 0 or rest_option(f, action)
        n[judged] += 1
        n["rest"] += advice[0] >= 0
    assert n[(0, 1)] > 5 and n[(0, -1)] == 0 and n["rest"] > 5, n
    for s in segs:
        assert (s.kinds[s.advice_rule >= 0] == Kind.COMBAT).all() and (s.advice_sign >= 0).all()


def test_actor_times_the_advice_stage(vocab: Vocab) -> None:
    for rules in ((), (POTION, SMITH_H)):
        reps: list[dict] = []
        segs: list[Segment] = []
        p, h = parse_advice(rules)
        store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), advice=bool(p), judged=bool(h))
        hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=150, hashes=False, abort_prob=0.0))
                 for i in range(4)]
        tok = Tokenizer(vocab, TokenizerConfig(cycle_ban=True, commit_select=True))
        Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(lineups=(("IRONCLAD+SILENT", 1.0),)), vocab, 0),
              ActorConfig(report_interval=0.2), store=store, on_report=reps.append, advice=rules).run(max_decisions=3000)
        stage = sum(r["stage_ns"]["advice"] for r in reps)
        assert (stage > 0) == bool(rules) and all("advice" in r["stage_ns"] for r in reps)


def test_store_and_actor_agree_on_habit_records(vocab: Vocab) -> None:
    store = RolloutStore(lambda s: None, RoomIds.from_vocab(vocab), advice=False, judged=False)
    with pytest.raises(AssertionError, match="habit rules iff"):
        Actor([FakeHost(vocab, seed=0)], UniformPolicy(0), Tokenizer(vocab), EpisodeSampler(EpisodeDist(), vocab, 0),
              ActorConfig(), store=store, advice=(POTION,))
    store.begin_episode(0, (2,))
    with pytest.raises(AssertionError, match="habit-rule record given iff"):
        store.add_step(0, None, None, 0, 0.0, 0.0, 0, None, (0, -1))  # type: ignore[arg-type]


# ------------------------------------------------------------------ transport


def test_blob_round_trip_with_habit_records(vocab: Vocab) -> None:
    for rules in ((POTION,), ((*SMITH[:2], 0.75, 0.05), POTION)):
        _, _, segs = actor_run(vocab, rules, n=800)
        for s in segs:
            d = decode_segment(pack_segment(s))
            assert_same_advice(s, d)
            assert d.advice_rule is not None and d.advice_sign is not None
            assert d.advice_rule.dtype == np.int8 and np.array_equal(s.advice_rule, d.advice_rule)
            assert np.array_equal(s.advice_sign, d.advice_sign)
    _, _, plain = actor_run(vocab, (), n=200)
    assert all(decode_segment(pack_segment(s)).advice_rule is None for s in plain)


def _habit_producer(spec: Any, conn: Any) -> None:
    vocab = Vocab(make_synthetic_vocab())
    ring = ShmRing.attach(spec, conn)
    _, _, segs = actor_run(vocab, (SMITH, POTION), n=1200)
    for s in segs:
        assert ring.put(s)
    conn.send(("done", len(segs)))
    ring.close()


def test_habit_records_from_another_process(vocab: Vocab) -> None:
    ctx = mp.get_context("spawn")
    owner, spec = ShmRing.create(64 << 20)
    rx, tx = ctx.Pipe(duplex=False)
    cons = ShmRing.attach(spec, rx)
    p = ctx.Process(target=_habit_producer, args=(spec, tx))
    p.start()
    try:
        got = []
        while True:
            assert rx.poll(180)
            msg = rx.recv()
            if msg[0] == "done":
                break
            got.append(cons.get(msg))
        p.join(30)
        assert p.exitcode == 0 and msg[1] == len(got) > 5
        _, _, ref = actor_run(vocab, (SMITH, POTION), n=1200)
        assert len(ref) == len(got)
        for a, b in zip(ref, got):
            assert_same_advice(a, b)
            assert np.array_equal(a.advice_rule, b.advice_rule) and np.array_equal(a.advice_sign, b.advice_sign)
        assert sum(int((s.advice_rule >= 0).sum()) for s in got) > 0
    finally:
        if p.is_alive():
            p.kill()
        cons.close()
        owner.close()


# ------------------------------------------------------------------ learner


def habit_segments(seed: int = 0, advice: bool = True) -> list[Segment]:
    """``synthetic_segments`` plus a habit-rule record: rule 0 (a COMBAT rule) at 20% of COMBAT steps with sign -1,
    rule 1 at 30% of EVENT steps with sign +1 / -1."""
    rng = np.random.default_rng(100 + seed)
    out = []
    for s in synthetic_segments(seed, advice=advice):
        rule = np.full(len(s), -1, np.int8)
        rule[(s.kinds == Kind.COMBAT) & (rng.random(len(s)) < 0.2)] = 0
        rule[(s.kinds == Kind.EVENT) & (rng.random(len(s)) < 0.3)] = 1
        sign = np.where(rule == 0, -1, np.where(rule == 1, rng.choice(np.array([-1, 1]), len(s)), 0)).astype(np.int8)
        out.append(replace(s, advice_rule=rule, advice_sign=sign))
    return out


HABITS = (POTION, ("no_elite:1-2", 0.02))  # stand-ins for the synthetic kinds: rule 0 judges COMBAT, rule 1 EVENT


def test_habit_offset_reaches_the_policy_advantage_once() -> None:
    rb = build_batch(habit_segments(), GAEConfig(), 0)
    assert rb.advice_rule is not None and rb.advice_sign is not None
    for rules in (HABITS, (SMITH, *HABITS)):
        plain, advised = _learner((), adv_norm=False), _learner(rules, adv_norm=False)
        a0, a1 = plain._policy_advantages(rb), advised._policy_advantages(rb)
        b = np.array([0.05, 0.02], np.float32)
        hab = np.where(rb.advice_rule >= 0, rb.advice_sign * b[np.maximum(rb.advice_rule, 0)], 0).astype(np.float32)
        assert np.array_equal(habit_offset(advised._habits, rb.advice_rule, rb.advice_sign), hab)
        old = advice_offset(advised._advice, rb.kinds, rb.actions, rb.advice_row, rb.advice_hp) if advised._advice else 0
        assert np.array_equal(a1, a0 + old + hab)
        assert ((rb.advice_rule == 0).sum() > 30 and (rb.advice_sign > 0).sum() > 10 and (rb.advice_sign < 0).sum() > 30)
        for norm in (False, True):  # only the advantage column differs; it is (A + offsets), normalised once
            p0, p1 = _learner((), adv_norm=norm), _learner(rules, adv_norm=norm)
            pf0, pi0, m0 = p0._columns(rb)
            pf1, pi1, m1 = p1._columns(rb)
            col = PF_COLS.index("adv")
            rest = [c for c in range(len(PF_COLS)) if c != col]
            assert np.array_equal(pf0[:, rest], pf1[:, rest]) and np.array_equal(pi0, pi1)
            assert all(np.array_equal(m0[k], m1[k]) for k in m0)
            x = a0 + old + hab
            pol = rb.n_legal > 1
            assert np.array_equal(pf1[:, col], (x - x[pol].mean()) / (x[pol].std() + 1e-8) if norm else x)
            t0, t1 = p0._tensors(rb), p1._tensors(rb)
            for name in t0.__dataclass_fields__:
                assert torch.equal(getattr(t0, name), getattr(t1, name)) == (name != "adv"), name
            assert np.array_equal(t1.adv.numpy(), pf1[:, col])


def test_habit_metrics() -> None:
    kinds = np.array([Kind.COMBAT] * 6 + [Kind.SHOP] * 4, np.int8)
    rule = np.array([0, 0, -1, -1, 0, -1, 1, 1, 1, -1], np.int8)
    sign = np.array([-1, -1, 0, 0, -1, 0, 1, -1, 1, 0], np.int8)
    adv = np.array([0.1, 0.2, 9, 9, 0.3, 9, 0.5, -0.4, 0.7, 9], np.float32)
    m = habit_metrics(habit_rules([POTION, ("shop_relic", 0.03)]), kinds, rule, sign, adv)
    assert m["potion_hold"] == {"bonus": 0.05, "judged": 3, "plus": 0, "minus": 3, "raw_adv_plus": None,
                                "raw_adv_minus": pytest.approx(0.2), "per_combat": 0.5}
    s = m["shop_relic"]
    assert (s["judged"], s["plus"], s["minus"], "per_combat" in s) == (3, 2, 1, False)
    assert s["raw_adv_plus"] == pytest.approx(0.6) and s["raw_adv_minus"] == pytest.approx(-0.4)
    assert habit_metrics(habit_rules([POTION]), kinds[:0], rule[:0], sign[:0], adv[:0])["potion_hold"]["per_combat"] is None


def test_zero_bonus_habit_rules_are_the_plain_update() -> None:
    m0, p0 = _update((), synthetic_segments(advice=False))
    stripped = [replace(s, advice_row=None, advice_hp=None) for s in habit_segments()]
    m1, p1 = _update((("potion_hold", 0.0), ("no_elite:1-2", 0.0)), stripped)
    assert _strip(m0) == _strip(m1) and all(torch.equal(a, b) for a, b in zip(p0, p1))
    m2, p2 = _update(HABITS, stripped)
    assert m2["loss_pi"] != m0["loss_pi"] and m2["explained_variance"] == m0["explained_variance"]
    assert set(m2["advice"]) == {"potion_hold", "no_elite:1-2"} and m2["advice"]["potion_hold"]["judged"] > 30
    m3, _ = _update((SMITH, *HABITS), habit_segments())
    assert set(m3["advice"]) == {"REST:SMITH", "potion_hold", "no_elite:1-2"}


def _run_tree_ref(tree: Path, advice: list) -> dict:
    env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"}
    out = subprocess.run([sys.executable, "-B", str(TREE_REF), json.dumps(advice)], env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_no_rule_and_old_rule_updates_are_bit_identical_to_the_advice_commit() -> None:
    """SPIRE_ADVICE_TREE = the py tree of b89108a (e.g. alt/unify/head/py, or py-stage-e2): the update without rules
    and with the live runs' old-format rule is bit-identical (metrics and parameters) to that tree's."""
    head = os.environ.get("SPIRE_ADVICE_TREE")
    if not head:
        pytest.skip("SPIRE_ADVICE_TREE (the b89108a py tree) not set")
    mine = Path(__file__).resolve().parents[1]
    for advice in ([], [["REST", "SMITH", 0.75, 0.05, 0.05]], [["REST", "SMITH", 0.5, 0.05]]):
        ref, got = _run_tree_ref(Path(head), advice), _run_tree_ref(mine, advice)
        assert ref["tree"] != got["tree"] and ref["samples"] == got["samples"] >= 400
        assert ref["metrics"] == got["metrics"] and ref["params"] == got["params"], advice
    a, b = _run_head_ref(Path(head)), _run_head_ref(mine)  # the corridor rollout path of the existing helper
    assert a["metrics"] == b["metrics"] and a["params"] == b["params"]


@pytest.mark.parametrize("topology", ["threads", "processes"])
def test_training_runs_report_habit_advice(tmp_path: Path, topology: str) -> None:
    rules = [["REST", "SMITH", 0.5, 0.05, 0.05], ["potion_hold", 0.05], ["shop_nocard", 0.05]]
    cfg = small_config(tmp_path, f"hab-{topology}", topology=topology, max_updates=3,
                       ppo={"minibatch": 256, "micro_batch": 256, "epochs": 1, "lr": 1e-3, "advice": rules},
                       tokenizer={"cycle_ban": True, "commit_select": True},
                       learner={"device": "cpu", "iteration_samples": 1024, "micro_tokens": None,
                                "checkpoint_interval_s": 3600, "allow_ppo_config_change": True})
    assert Supervisor(cfg).run() == 0
    rows = _learner_rows(cfg.run_dir)
    assert [r["version"] for r in rows] == [1, 2, 3]
    assert all(set(r["advice"]) == {"REST:SMITH", "potion_hold", "shop_nocard"} for r in rows)
    pot = [r["advice"]["potion_hold"] for r in rows]
    assert sum(a["judged"] for a in pot) > 0 and all(a["plus"] == 0 and a["per_combat"] is not None for a in pot)
    assert sum(r["advice"]["REST:SMITH"]["steps"] for r in rows) > 0


# ------------------------------------------------------------------ round-3 habits (notes/gym/merge.md)

WEAK_SET = [["REST", "SMITH", 0.75, 0.05, 0.05], ["potion_r1:Elite,Boss", 0.1], ["weak_attacker:cards", 0.05]]


def test_round3_rules_validate_and_round_trip(tmp_path: Path) -> None:
    """The roll-out set + weak_attacker:cards passes PPOConfig and survives config.json -> config -> checkpoint; the
    round-3 training forms validate; two-sided / evaluation-only forms and pairs that can judge one step are refused;
    a vocab without the habit's names fails at the Supervisor's start-up (the synthetic vocab has no OSTY)."""
    old, new = parse_advice(WEAK_SET)
    assert [r.name for r in old] == ["REST:SMITH"] and [(r.spec, r.bonus) for r in new] == \
        [("potion_r1:Elite,Boss", 0.1), ("weak_attacker:cards", 0.05)]
    assert [r.habit.VERBS for r in new] == [{int(Verb.USE_POTION)}, {int(Verb.PLAY_CARD)}]
    cfg = small_config(tmp_path, "weak", ppo={"advice": WEAK_SET})
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config_dict(cfg)), encoding="utf-8")
    back = load_config(p)
    assert json.loads(p.read_text(encoding="utf-8"))["ppo"]["advice"] == WEAK_SET
    assert parse_advice(back.ppo.advice) == parse_advice(WEAK_SET)
    tok = StubTokenizer()
    lr = Learner(StubNet(tok.n_features), tok, back.ppo, device="cpu")
    lr.save(tmp_path / "weak.pt")
    assert [list(x) for x in torch.load(tmp_path / "weak.pt", weights_only=False)["config"]["advice"]] == WEAK_SET
    for spec in (["potion_boss_take", 0.05], ["potion_lethal", 0.05], ["weak_attacker", 0.05], ["kill_dangerous", 0.05],
                 ["lethal_block", 0.05], ["lethal_eot", 0.05], ["potion_reserve2:all", 0.05], ["potion_none", 0.05],
                 ["potion_boss_use:late", 0.05], ["heal_late:1.0:3:2", 0.05],
                 ["explore:page:ROOM_FULL_OF_CHEESE.INITIAL=ROOM_FULL_OF_CHEESE.pages.INITIAL.options.SEARCH", 0.05]):
        PPOConfig(advice=(tuple(spec),))
    for bad, msg in (([["kill_dangerous:1.0", 0.05]], "evaluation-only"),
                     ([["potion_boss_use", 0.05], ["potion_r1:Elite,Boss", 0.05]], "can judge one step"),
                     ([["potion_boss_use", 0.05], ["weak_attacker:cards", 0.05]], "can judge one step"),
                     ([["potion_boss_take", 0.05], ["potion_r1:Elite,Boss", 0.05]], "can judge one step"),
                     ([["heal_late:1.0", 0.05], ["smith:0.75", 0.05]], "can judge one step"),
                     ([["REST", "SMITH", 0.75, 0.05, 0.05], ["heal_late:1.0", 0.05]], "both judge REST"),
                     ([["potion_none", 0.05], ["potion_hold", 0.05]], "can judge one step"),
                     ([["weak_attacker:bogus", 0.05]], "weak_attacker:bogus")):
        with pytest.raises(AssertionError, match=msg):
            PPOConfig(advice=bad)  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="lacks OSTY"):
        Supervisor(small_config(tmp_path, "osty", ppo={"advice": [["weak_attacker:cards", 0.05]]}, catalog="synthetic"))
