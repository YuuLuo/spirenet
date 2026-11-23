from __future__ import annotations

from typing import Generator

import numpy as np
import pytest

from ppo_stubs import obs_frame, room_ids, terminal_frame
from spire import synth
from spire.rewards import (
    Outcome,
    RewardConfig,
    RoomIds,
    episode_targets,
    frame_facts,
    inferred_kills,
    team_hp_fraction,
)
from spire.schema import Kind, Result, SeatCol, Table, TerminalReason
from spire.synth import FakeHost, FakeHostConfig, RunCtx
from spire.vocab import Vocab

CFG = RewardConfig(gamma=0.99)


@pytest.fixture(scope="module")
def R(vocab: Vocab) -> dict[str, int]:
    return room_ids(vocab)


@pytest.fixture(scope="module")
def rooms(vocab: Vocab) -> RoomIds:
    return RoomIds.from_vocab(vocab)


def phi(h: float) -> float:
    return CFG.hp_coef * h


def run_frames(R: dict[str, int], final: int) -> list:
    """MAP f0 -> Monster combat f1 (2 decisions) -> REWARD -> MAP -> Elite f2 (1) -> REWARD
    -> MAP -> Boss f3 (2) -> TERMINAL(final). The host's win counters (GLOBAL ELITES_WON / BOSSES_WON)
    go up on the first frame after a won fight; the final boss win is on the TERMINAL(WIN) frame."""
    M, E, B, EV, MP = R["Monster"], R["Elite"], R["Boss"], R["Event"], R["Map"]
    return [
        obs_frame(Kind.MAP, 0, MP, False, [80]),            # 0
        obs_frame(Kind.COMBAT, 1, M, True, [80]),           # 1 combat 0
        obs_frame(Kind.COMBAT, 1, M, True, [70]),           # 2 combat 0
        obs_frame(Kind.REWARD, 1, M, False, [70]),          # 3
        obs_frame(Kind.MAP, 1, MP, False, [70]),            # 4
        obs_frame(Kind.COMBAT, 2, E, True, [70]),           # 5 combat 1 (elite)
        obs_frame(Kind.CARD_REWARD, 2, E, False, [50], elites=1),  # 6
        obs_frame(Kind.MAP, 2, MP, False, [50], elites=1),  # 7
        obs_frame(Kind.COMBAT, 3, B, True, [50], elites=1),  # 8 combat 2 (boss)
        obs_frame(Kind.CARD_SELECT, 3, B, True, [40], elites=1),  # 9 combat 2
        terminal_frame(final, 3, B, [0], elites=1, bosses=int(final == Result.WIN),
                       steps=10),
    ]


def test_win_episode_components(R: dict[str, int], rooms: RoomIds) -> None:
    facts = [frame_facts(f) for f in run_frames(R, Result.WIN)]
    t = episode_targets(facts, rooms, CFG)
    assert t.outcome == Outcome.WIN and len(t.rewards) == 10
    c = t.components
    np.testing.assert_allclose(c["floor"], [0.01, 0, 0, 0, 0.01, 0, 0, 0.01, 0, 0], atol=1e-7)
    np.testing.assert_allclose(c["elite"], [0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(c["boss"], [0] * 9 + [0.1], atol=1e-7)  # final boss credited into TERMINAL(WIN)
    np.testing.assert_allclose(c["terminal"], [0] * 9 + [1.0], atol=1e-7)
    hp = [0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.4]
    expect_hp = [CFG.gamma * phi(hp[i + 1]) - phi(hp[i]) for i in range(9)] + [0.0 - phi(0.4)]
    np.testing.assert_allclose(c["hp"], expect_hp, atol=1e-6)
    np.testing.assert_allclose(t.rewards, sum(c.values()), atol=1e-6)
    np.testing.assert_array_equal(t.combat_id, [-1, 0, 0, -1, -1, 1, -1, -1, 2, 2])
    np.testing.assert_allclose(t.hp_post[t.hp_mask], [0.7, 0.7, 0.5, 0.4, 0.4], atol=1e-6)
    np.testing.assert_array_equal(t.hp_mask, t.combat_id >= 0)
    assert t.win == 1.0 and t.floors_left is not None
    np.testing.assert_allclose(t.floors_left, [3, 2, 2, 2, 2, 1, 1, 1, 0, 0])


def test_loss_in_boss_gets_no_kill(R: dict[str, int], rooms: RoomIds) -> None:
    t = episode_targets([frame_facts(f) for f in run_frames(R, Result.LOSS)], rooms, CFG)
    assert t.outcome == Outcome.LOSS and t.win == 0.0
    assert t.components["boss"].sum() == 0
    assert t.components["terminal"][-1] == pytest.approx(-0.1)
    assert t.components["elite"].sum() == pytest.approx(0.02)
    assert t.hp_post[-1] == 0.0 and t.hp_mask[-1]


@pytest.mark.parametrize("frame_pool", [0, 2])
def test_fake_win_terminal_zero_hp_labels_final_fight_from_last_decision(
        vocab: Vocab, rooms: RoomIds, monkeypatch: pytest.MonkeyPatch, frame_pool: int) -> None:
    scripted_hps: list[int] = []

    def final_fight(ctx: RunCtx) -> Generator[tuple[int, int], int, None]:
        ctx.room, ctx.max_hp = "Boss", [100, 80]
        ctx.acting_seat = -1
        for hp in (50, 40, 30):
            scripted_hps.append(hp)
            ctx.hp = [hp, 0]
            ctx.ready_seats.clear()
            yield Kind.COMBAT, 0

    monkeypatch.setattr(synth, "_schedule_h5", lambda rng, ctx, gen, size: final_fight(ctx))
    cfg = FakeHostConfig(mean_decisions=3, min_decisions=3, max_decisions=3, length_sigma=0.0,
                         win_prob=1.0, abort_prob=0.0, hashes=False, frame_pool=frame_pool)
    with FakeHost(vocab, config=cfg) as host:
        frame = host.reset(5, (2, 4), 0)
        facts = [frame_facts(frame)]
        while frame.is_obs:
            assert frame.kind == Kind.COMBAT
            assert facts[-1].room_type == vocab.id("enum:RoomType", "Boss")
            frame = host.step(0)
            facts.append(frame_facts(frame))
    assert scripted_hps == [50, 40, 30]
    assert frame.result == Result.WIN
    np.testing.assert_array_equal(frame.table(Table.SEATS)[:, SeatCol.HP], [0, 0])
    assert facts[-2].hp_frac > 0
    targets = episode_targets(facts, rooms)
    assert targets.hp_mask.all() and targets.win == 1.0
    np.testing.assert_allclose(targets.hp_post, [facts[-2].hp_frac] * 3)


@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_shaping_telescopes_to_minus_phi0(R: dict[str, int], rooms: RoomIds, gamma: float) -> None:
    cfg = RewardConfig(gamma=gamma)
    for final in (Result.WIN, Result.LOSS):
        facts = [frame_facts(f) for f in run_frames(R, final)]
        t = episode_targets(facts, rooms, cfg)
        reference = []
        for before, after in zip(facts, facts[1:]):
            next_phi = 0.0 if after.terminal else cfg.hp_coef * after.hp_frac
            reference.append(gamma * next_phi - cfg.hp_coef * before.hp_frac)
        np.testing.assert_allclose(t.components["hp"], reference, atol=1e-7)
        discounted = 0.0
        for reward in reversed(reference):
            discounted = reward + gamma * discounted
        assert discounted == pytest.approx(-cfg.hp_coef * facts[0].hp_frac, abs=1e-12)
        assert float((gamma ** np.arange(len(t.rewards)) * t.components["hp"]).sum()) == pytest.approx(
            discounted, abs=1e-6)


def test_aborted_and_overflow_are_dropped_without_terminal_reward(R: dict[str, int], rooms: RoomIds) -> None:
    frames = run_frames(R, Result.WIN)[:-1] + [terminal_frame(Result.ABORTED, 3, R["Boss"], [40])]
    t = episode_targets([frame_facts(f) for f in frames], rooms, CFG)
    assert t.outcome == Outcome.ABORTED and t.outcome.dropped
    assert len(t.rewards) == 9 and t.components["terminal"].sum() == 0 and t.components["boss"].sum() == 0
    assert t.win is None and t.floors_left is None
    assert not t.hp_mask[8:].any()  # the boss combat never resolved
    frames = run_frames(R, Result.WIN)
    frames[4] = obs_frame(Kind.MAP, 1, R["Map"], False, [70], overflow=True)
    assert episode_targets([frame_facts(f) for f in frames], rooms, CFG).outcome == Outcome.OVERFLOW


def test_prefix_consistent_with_full_episode(R: dict[str, int], rooms: RoomIds) -> None:
    facts = [frame_facts(f) for f in run_frames(R, Result.WIN)]
    full = episode_targets(facts, rooms, CFG)
    for k in range(1, len(facts) - 1):
        part = episode_targets(facts[:k + 1], rooms, CFG)
        assert part.outcome == Outcome.ONGOING and len(part.rewards) == k
        np.testing.assert_allclose(part.rewards, full.rewards[:k], atol=1e-7)
        np.testing.assert_array_equal(part.combat_id, full.combat_id[:k + 1])
        resolved = part.hp_mask
        np.testing.assert_allclose(part.hp_post[resolved], full.hp_post[:k + 1][resolved])


def test_double_boss_back_to_back_and_team_hp(R: dict[str, int], rooms: RoomIds) -> None:
    B = R["Boss"]
    frames = [
        obs_frame(Kind.COMBAT, 16, B, True, [50, 0], [100, 80], bosses=2),
        obs_frame(Kind.COMBAT, 17, B, True, [40, 0], [100, 80], bosses=3),  # second boss: next floor, still in combat
        obs_frame(Kind.COMBAT, 17, B, True, [30, 0], [100, 80], bosses=3),
        terminal_frame(Result.WIN, 17, B, [0, 0], [100, 80], bosses=4, steps=3),
    ]
    facts = [frame_facts(f) for f in frames]
    assert facts[0].hp_frac == pytest.approx(0.25)  # mean(50/100, 0/80): a dead seat counts 0
    t = episode_targets(facts, rooms, CFG)
    np.testing.assert_allclose(t.components["boss"], [0.1, 0, 0.1], atol=1e-7)
    np.testing.assert_array_equal(t.combat_id, [0, 1, 1])
    np.testing.assert_allclose(t.hp_post, [0.2, 0.15, 0.15], atol=1e-6)


def test_event_combat_on_same_floor(R: dict[str, int], rooms: RoomIds) -> None:
    """errata-v1 E-22: an event-started fight is a Monster room fight (ROOM_TYPE Monster, not Event)."""
    EV, M = R["Event"], R["Monster"]
    frames = [
        obs_frame(Kind.EVENT, 5, EV, False, [60]),
        obs_frame(Kind.COMBAT, 5, M, True, [60]),  # an event fight: same floor, Monster room, no kill bonus
        obs_frame(Kind.COMBAT, 5, M, True, [55]),
        obs_frame(Kind.EVENT, 5, EV, False, [55]),
        obs_frame(Kind.MAP, 5, R["Map"], False, [55]),
    ]
    facts = [frame_facts(f) for f in frames]
    for kills in ("counters", "rooms"):
        t = episode_targets(facts, rooms, RewardConfig(gamma=0.99, kills=kills))
        assert t.outcome == Outcome.ONGOING
        np.testing.assert_array_equal(t.combat_id, [-1, 0, 0, -1, -1])
        np.testing.assert_allclose(t.hp_post[1:3], [0.55, 0.55], atol=1e-6)
        assert t.components["elite"].sum() == t.components["boss"].sum() == t.components["floor"].sum() == 0


def test_team_hp_fraction_edge_cases() -> None:
    seats = np.zeros((3, 32), np.int32)
    seats[:, SeatCol.HP] = [120, -5, 10]
    seats[:, SeatCol.MAX_HP] = [100, 50, 0]
    assert team_hp_fraction(seats) == pytest.approx(1 / 3)  # clipped to 1, negative hp 0, max_hp 0 -> 0
    assert team_hp_fraction(seats[:0]) == 0.0


def test_room_ids_require_vocab_names(vocab: Vocab) -> None:
    r = RoomIds.from_vocab(vocab)
    assert vocab.name("enum:RoomType", r.elite) == "Elite" and vocab.name("enum:RoomType", r.boss) == "Boss"


# --------------------------------------------------------------------------- errata-v1: counters, terminal reasons


def test_counter_kills_equal_inferred_kills_on_the_run(R: dict[str, int], rooms: RoomIds) -> None:
    """PY-5: kill bonuses from the host counters equal the old ROOM_TYPE inference on a normal run."""
    for final in (Result.WIN, Result.LOSS):
        facts = [frame_facts(f) for f in run_frames(R, final)]
        a = episode_targets(facts, rooms, CFG)
        b = episode_targets(facts, rooms, RewardConfig(gamma=0.99, kills="rooms"))
        np.testing.assert_allclose(a.rewards, b.rewards, atol=1e-7)
        assert inferred_kills(facts, rooms) == (1, int(final == Result.WIN))


def test_escaped_elite_gets_no_bonus_from_counters(R: dict[str, int], rooms: RoomIds) -> None:
    """The case the counters fix (S2b-2): an elite fight that ends without a win is not a kill."""
    E, MP = R["Elite"], R["Map"]
    frames = [obs_frame(Kind.COMBAT, 2, E, True, [70]), obs_frame(Kind.COMBAT, 2, E, True, [60]),
              obs_frame(Kind.MAP, 2, MP, False, [60])]  # the elite fled: counters unchanged
    facts = [frame_facts(f) for f in frames]
    assert episode_targets(facts, rooms, CFG).components["elite"].sum() == 0
    assert episode_targets(facts, rooms, RewardConfig(gamma=0.99, kills="rooms")).components["elite"].sum() > 0


@pytest.mark.parametrize("reason,caps_as_failure", [
    (TerminalReason.DECISION_CAP, False), (TerminalReason.COMBAT_TURN_CAP, False),
    (TerminalReason.OTHER_TRUNCATION, True), (TerminalReason.OTHER_TRUNCATION, False),
])
def test_truncation_is_not_a_death(R: dict[str, int], rooms: RoomIds, reason: TerminalReason,
                                 caps_as_failure: bool) -> None:
    """Other truncations and the explicit legacy cap setting retain no death reward, Phi(s_T) of the real
    HP (the TERMINAL frame is a state), outcome targets masked, an unfinished combat has no hp_post."""
    B = R["Boss"]
    in_combat = reason == TerminalReason.COMBAT_TURN_CAP
    frames = run_frames(R, Result.WIN)[:-1] + [
        terminal_frame(Result.LOSS, 3, B, [35], elites=1, reason=int(reason), in_combat=in_combat, steps=10)]
    facts = [frame_facts(f) for f in frames]
    assert facts[-1].truncated and not facts[-2].truncated
    t = episode_targets(facts, rooms, RewardConfig(gamma=CFG.gamma, caps_as_failure=caps_as_failure))
    assert t.outcome == Outcome.TRUNCATED and not t.outcome.finished and not t.outcome.dropped
    assert len(t.rewards) == 10 and t.components["terminal"].sum() == 0 and t.components["boss"].sum() == 0
    assert t.components["hp"][-1] == pytest.approx(CFG.gamma * phi(0.35) - phi(0.4), abs=1e-6)
    assert t.win is None and t.floors_left is None
    if in_combat:  # the boss fight was still running at the cut: never resolved
        assert not t.hp_mask[8:].any()
    else:  # decision cap outside a fight: the boss fight "ended" into the cut frame
        np.testing.assert_allclose(t.hp_post[8:], [0.35, 0.35], atol=1e-6)
    death = episode_targets([*facts[:-1], frame_facts(terminal_frame(Result.LOSS, 3, B, [0], elites=1, steps=10))],
                            rooms, CFG)
    assert death.outcome == Outcome.LOSS and death.components["terminal"][-1] == pytest.approx(CFG.death)


@pytest.mark.parametrize("reason", [TerminalReason.DECISION_CAP, TerminalReason.COMBAT_TURN_CAP])
@pytest.mark.parametrize("in_combat", [False, True])
def test_policy_cap_is_terminal_failure(R: dict[str, int], rooms: RoomIds, reason: TerminalReason,
                                        in_combat: bool) -> None:
    frames = run_frames(R, Result.WIN)[:-1] + [terminal_frame(
        Result.LOSS, 3, R["Boss"], [35], elites=1, reason=reason, in_combat=in_combat, steps=10)]
    t = episode_targets([frame_facts(f) for f in frames], rooms)
    assert t.outcome == Outcome.LOSS and t.outcome.finished and not t.outcome.dropped
    np.testing.assert_allclose(t.components["terminal"], [0] * 9 + [-0.1], atol=1e-7)
    assert t.components["hp"][-1] == pytest.approx(-0.04)
    assert t.components["boss"].sum() == 0
    assert t.win == 0.0
    np.testing.assert_array_equal(t.floors_left, [3, 2, 2, 2, 2, 1, 1, 1, 0, 0])
    np.testing.assert_allclose(t.hp_post[[1, 2, 5]], [0.7, 0.7, 0.5])
    if in_combat:
        assert not t.hp_mask[8:].any()  # a failed run has not observed this combat's post-fight HP
    else:
        np.testing.assert_allclose(t.hp_post[8:], [0.35, 0.35])


@pytest.mark.parametrize("decisions", [1, 10, 1000])
@pytest.mark.parametrize("reason", [TerminalReason.DEATH, TerminalReason.DECISION_CAP,
                                    TerminalReason.COMBAT_TURN_CAP])
def test_gamma_one_stalling_never_reduces_failure_cost(R: dict[str, int], rooms: RoomIds,
                                                      decisions: int, reason: TerminalReason) -> None:
    start = frame_facts(obs_frame(Kind.COMBAT, 8, R["Elite"], True, [50]))
    terminal = frame_facts(terminal_frame(Result.LOSS, 8, R["Elite"],
                                        [0 if reason == TerminalReason.DEATH else 50],
                                        reason=reason, in_combat=reason != TerminalReason.DEATH,
                                        steps=decisions))
    t = episode_targets([start] * decisions + [terminal], rooms, RewardConfig(gamma=1.0))
    scalar_return = 0.0
    for reward in reversed(t.rewards.tolist()):
        scalar_return = reward + scalar_return
    assert scalar_return == pytest.approx(-0.1 - 0.1 * 0.5)


def test_reasonless_loss_is_a_death(R: dict[str, int], rooms: RoomIds) -> None:
    """TERMINAL_REASON NONE (hosts predating errata-v1) keeps the old reading: RESULT=LOSS is a death."""
    frames = run_frames(R, Result.WIN)[:-1] + [terminal_frame(Result.LOSS, 3, R["Boss"], [0], reason=0, steps=10)]
    assert episode_targets([frame_facts(f) for f in frames], rooms, CFG).outcome == Outcome.LOSS
