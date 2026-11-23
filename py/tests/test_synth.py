from __future__ import annotations

import collections

import numpy as np
import pytest

from spire.host import ActionRejected, HostStateError
from spire.proto import ResetCmd, StepCmd, frame_problems, parse_frame
from spire.schema import (
    DECISION_KINDS,
    ActionCol,
    CardCol,
    ErrorCode,
    FrameFlag,
    Kind,
    MsgType,
    Result,
    SeatCol,
    Table,
    VarCol,
    ViewMode,
    Zone,
)
from spire.synth import FakeGame, FakeHost, FakeHostConfig, FrameBuilder, FrameGen, random_frame
from spire.vocab import Vocab


def test_generator_is_deterministic(gen: FrameGen) -> None:
    a = [random_frame(gen, np.random.default_rng(7), k, n_seats=3) for k in DECISION_KINDS]
    b = [random_frame(gen, np.random.default_rng(7), k, n_seats=3) for k in DECISION_KINDS]
    assert a == b


def test_seat_view_hides_teammates(gen: FrameGen) -> None:
    rng = np.random.default_rng(11)
    for _ in range(10):
        f = parse_frame(random_frame(gen, rng, Kind.COMBAT, n_seats=4, view=ViewMode.SEAT))
        k = int(f.glob[4])
        seats = f.table(Table.SEATS)
        assert seats[:, SeatCol.PRIVATE_VISIBLE].tolist() == [int(s == k) for s in range(4)]
        others = np.arange(4) != k
        assert (seats[others, SeatCol.DRAW_COUNT] == -1).all()
        cards = f.table(Table.CARDS)
        private = np.isin(cards[:, CardCol.ZONE], [Zone.HAND, Zone.DRAW, Zone.DISCARD, Zone.EXHAUST])
        assert (cards[private, CardCol.SEAT] == k).all()
        assert (f.actions[:, ActionCol.SEAT] == k).all()
        assert not f.flags & FrameFlag.TEAM_VIEW
        assert frame_problems(f) == []


def test_extreme_sizes_stay_within_caps(gen: FrameGen) -> None:
    rng = np.random.default_rng(12)
    f = parse_frame(random_frame(gen, rng, Kind.COMBAT, n_seats=4, size="extreme", view=ViewMode.TEAM))
    assert f.n_actions > 250 and f.table(Table.CARDS).shape[0] > 350 and f.table(Table.VARS).shape[0] > 1000
    assert f.table(Table.ORBS).shape[0] == 40 and f.table(Table.ARROWS).shape[0] == 16
    c = parse_frame(random_frame(gen, rng, Kind.CRYSTAL, size="extreme"))
    assert c.table(Table.CELLS).shape[0] == 121 and c.n_actions == 242


def test_builder_rejects_misuse() -> None:
    b = FrameBuilder()
    with pytest.raises(ValueError, match="attach"):
        b.add(Table.VARS, var_id=3)
    with pytest.raises(ValueError, match="attach"):
        b.add(Table.POWERS, power_id=3)
    b.add(Table.CREATURES, kind=1)
    b.add(Table.CREATURES, kind=3)
    b.attach(Table.POWERS, 1, [b.new_rows(Table.POWERS, 1)[0]])
    with pytest.raises(ValueError, match="order"):
        b.attach(Table.POWERS, 0, b.new_rows(Table.POWERS, 1))
    with pytest.raises(ValueError, match="does not exist"):
        b.attach(Table.INTENTS, 5, b.new_rows(Table.INTENTS, 1))
    b.add(Table.CARDS, zone=1)
    v = b.new_rows(Table.VARS, 2)
    v[:, VarCol.VAR_ID] = 4
    b.attach_vars(Table.CARDS, 0, v)
    with pytest.raises(ValueError, match="duplicate"):
        b.build()
    b2 = FrameBuilder()
    for _ in range(5):
        b2.add(Table.SEATS)
    with pytest.raises(ValueError, match="cap"):
        b2.build()
    b3 = FrameBuilder()
    b3.add(Table.ACTIONS, verb=1)
    b3.add(Table.ACTIONS, verb=1)
    with pytest.raises(ValueError, match="duplicate ACTIONS"):
        b3.build()


def test_multiset_groups_identical_instances() -> None:
    b = FrameBuilder()
    inst = np.tile(b.new_rows(Table.CARDS, 1), (5, 1))
    inst[:, CardCol.ZONE] = Zone.DRAW
    inst[:, CardCol.CARD_ID] = [9, 3, 9, 3, 9]
    inst[:, CardCol.POS] = [0, 1, 2, 3, 4]  # pos is not part of the key
    v = b.new_rows(Table.VARS, 1)
    v[0, VarCol.VAR_ID], v[0, VarCol.SHOWN] = 2, 6
    v2 = v.copy()
    v2[0, VarCol.SHOWN] = 7  # different printed value -> separate group
    rows = b.add_multiset(inst, [None, None, v, None, v2])
    b.add(Table.ACTIONS, verb=1)
    b.glob[2] = 1  # n_seats
    f = parse_frame(b.build(), verify_hashes=True)
    c = f.table(Table.CARDS)
    assert len(rows) == 4  # 3 x2, then 9 without vars (a key prefix sorts first), 9 shown=6, 9 shown=7
    assert c[:, [CardCol.CARD_ID, CardCol.COUNT, CardCol.POS, CardCol.VAR_CNT]].tolist() == [
        [3, 2, -1, 0], [9, 1, -1, 0], [9, 1, -1, 1], [9, 1, -1, 1]]
    assert f.table(Table.VARS)[:, VarCol.SHOWN].tolist() == [6, 7]
    assert frame_problems(f) == []


# ---------------------------------------------------------------- FakeGame / FakeHost


def run_episode(h: FakeHost, seed: int | None, policy_seed: int, chars: tuple[int, ...] = (4,)) -> list[bytes]:
    rng = np.random.default_rng(policy_seed)
    f = h.reset(seed, chars, 0)
    frames = [bytes(f.payload)]
    while not f.is_terminal:
        f = h.step(int(rng.integers(f.n_actions)))
        frames.append(bytes(f.payload))
    return frames


def test_fake_host_deterministic_per_seed(vocab: Vocab) -> None:
    cfg = FakeHostConfig(mean_decisions=60)
    a = run_episode(FakeHost(vocab, seed=1, config=cfg), 12345, 0)
    b = run_episode(FakeHost(vocab, seed=999, config=cfg), 12345, 0)  # explicit seed: host seed irrelevant
    assert a == b
    c = run_episode(FakeHost(vocab, seed=1, config=cfg), 12345, 1)  # other actions -> other frames
    assert len(c) == len(a) and c[1:] != a[1:]
    h1, h2 = FakeHost(vocab, seed=5, config=cfg), FakeHost(vocab, seed=5, config=cfg)
    assert run_episode(h1, None, 3) == run_episode(h2, None, 3)  # unseeded: deterministic per host seed


def test_fake_host_episodes_are_plausible(vocab: Vocab) -> None:
    cfg = FakeHostConfig(mean_decisions=80, win_prob=0.6, abort_prob=0.1, auto_step_prob=0.2)
    h = FakeHost(vocab, seed=2, config=cfg, verify_hashes=True, expected_hashes=(vocab.hash, 0))
    kinds: collections.Counter[int] = collections.Counter()
    rng = np.random.default_rng(0)
    lengths = []
    for ep in range(40):
        f = h.reset(None, (2, 4) if ep % 3 == 0 else (4,), ep % 11)
        assert f.step_id == 0 and f.episode_id == ep + 1
        n = 0
        while not f.is_terminal:
            kinds[f.kind] += 1
            if n % 25 == 0:
                assert frame_problems(f) == []
            f = h.step(int(rng.integers(f.n_actions)))
            n += 1
        assert h.last_episode is not None and h.last_episode.decisions == n and len(h.last_episode.actions) == n
        assert frame_problems(f) == [] and f.n_actions == 0
        lengths.append(n)
    s = h.stats
    assert s.episodes == s.finished == 40 and s.wins + s.losses + s.aborted == 40
    assert 0 < s.aborted < 14 and s.wins > s.losses / 2 and s.auto_steps > 0 and s.errors == 0
    assert {Kind.MAP, Kind.COMBAT, Kind.REWARD, Kind.CARD_REWARD, Kind.CARD_SELECT} <= set(kinds)
    assert kinds[Kind.COMBAT] > kinds[Kind.MAP] > kinds[Kind.SHOP]
    assert 30 < np.mean(lengths) < 200


def test_fake_schedule_reaches_every_kind(vocab: Vocab) -> None:
    cfg = FakeHostConfig(mean_decisions=900, length_sigma=0.01, abort_prob=0.0, hashes=False)
    h = FakeHost(vocab, seed=0, config=cfg)
    seen: set[int] = set()
    rng = np.random.default_rng(1)
    for _ in range(3):
        f = h.reset(None, (4,), 0)
        while not f.is_terminal:
            seen.add(f.kind)
            f = h.step(int(rng.integers(f.n_actions)))
    assert seen == set(DECISION_KINDS)


def test_fake_host_errors_and_resend(vocab: Vocab) -> None:
    h = FakeHost(vocab, seed=3, config=FakeHostConfig(mean_decisions=50, abort_prob=0.0))
    f0 = h.reset(1, (4,), 0)
    with pytest.raises(ActionRejected) as e:
        h.step(f0.n_actions)
    assert e.value.code == ErrorCode.BAD_INDEX and e.value.frame is not None
    assert e.value.frame.payload == f0.payload and h.last_frame is e.value.frame
    with pytest.raises(ActionRejected) as e:
        h.step(0, step_id=f0.step_id + 5)
    assert e.value.code == ErrorCode.STALE_STEP
    f1 = h.step(0)
    assert f1.step_id == 1 and h.stats.errors == 2 and h.episode is not None and h.episode.errors == 2
    flaky = FakeHost(vocab, seed=3, config=FakeHostConfig(mean_decisions=50, error_prob=0.3, abort_prob=0.0))
    f = flaky.reset(1, (4,), 0)
    rejected = 0
    while not f.is_terminal:
        try:
            f = flaky.step(0)
        except ActionRejected as err:
            assert err.code == ErrorCode.BAD_MESSAGE and err.frame is not None
            f = err.frame
            rejected += 1
    assert rejected > 3 and flaky.stats.errors == rejected


def test_fake_host_state_errors(vocab: Vocab) -> None:
    h = FakeHost(vocab, config=FakeHostConfig(mean_decisions=3, min_decisions=3, length_sigma=0.0, abort_prob=0.0))
    with pytest.raises(HostStateError):
        h.step(0)
    f = h.reset(None, (4,), 0)
    for _ in range(3):
        f = h.step(0)
    assert f.is_terminal and f.result in (Result.WIN, Result.LOSS)
    with pytest.raises(HostStateError):
        h.step(0)
    h.reset(None, (4,), 0)
    h.reset(None, (4,), 0)
    assert h.stats.abandoned == 1
    h.close()


def test_fake_game_direct_messages(vocab: Vocab) -> None:
    g = FakeGame(vocab, seed=0)
    (err,) = g.handle(StepCmd(0, 0))  # no episode yet: ERROR without re-send
    assert parse_frame(err).msg_type == MsgType.ERROR
    (obs,) = g.handle(ResetCmd((4, 7, 2), ascension=10, seed=77))
    f = parse_frame(obs, verify_hashes=True)
    assert f.n_seats == 3 and int(f.glob[6]) == 10 and frame_problems(f) == []
    assert f.table(Table.SEATS)[:, SeatCol.CHARACTER_ID].tolist() == [4, 7, 2]
