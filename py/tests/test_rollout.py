from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest

from ppo_stubs import obs_frame, room_ids, terminal_frame
from spire.rewards import Facts, Outcome, RewardConfig, RoomIds, episode_targets, frame_facts
from spire.rollout import (
    EpisodeMeta,
    GAEConfig,
    RolloutConfig,
    RolloutStore,
    Segment,
    SegmentEnd,
    SegmentRing,
    build_batch,
    gae,
    seat_trace_decay,
)
from spire.schema import Kind, Result, TerminalReason
from spire.synth import FakeHost, FakeHostConfig
from spire.vocab import Vocab

# =========================================================================== brute-force references


def lambda_mixture(rets: list[float], lam: float) -> float:
    """(1 - lam) * sum_{n<N} lam^(n-1) G_n + lam^(N-1) G_N."""
    n = len(rets)
    return sum((1 - lam) * lam ** i * g for i, g in enumerate(rets[:-1])) + lam ** (n - 1) * rets[-1]


def ref_standard(seg: Segment, gamma: float, lam: float) -> np.ndarray:
    r, v, L = seg.rewards, seg.values, len(seg)
    vb = 0.0 if seg.end == SegmentEnd.TERMINAL else seg.bootstrap_value
    V = lambda j: v[j] if j < L else vb  # noqa: E731
    out = []
    for t in range(L):
        rets = [sum(gamma ** k * r[t + k] for k in range(n)) + gamma ** n * V(t + n) for n in range(1, L - t + 1)]
        out.append(lambda_mixture(rets, lam) - v[t])
    return np.array(out)


def ref_macro_micro(seg: Segment, cfg: GAEConfig) -> np.ndarray:
    r, v, c, L, g = seg.rewards, seg.values, seg.combat_id, len(seg), cfg.gamma
    vb = 0.0 if seg.end == SegmentEnd.TERMINAL else seg.bootstrap_value
    V = lambda j: v[j] if j < L else vb  # noqa: E731
    out = []
    for t in range(L):
        if c[t] >= 0:  # micro: n-step returns up to the end of this combat (or the segment)
            h = next((j for j in range(t + 1, L) if c[j] != c[t]), L)
            ends, lam = list(range(t + 1, h + 1)), cfg.lam_micro
        else:  # macro: options ending at each following macro step, then the segment end
            ends, lam = [j for j in range(t + 1, L) if c[j] < 0] + [L], cfg.lam_macro
        rets = [sum(g ** (k - t) * r[k] for k in range(t, e)) + g ** (e - t) * V(e) for e in ends]
        out.append(lambda_mixture(rets, lam) - v[t])
    return np.array(out)


def make_segment(rng: np.random.Generator, length: int, end: SegmentEnd, combat: bool = False) -> Segment:
    meta = EpisodeMeta(0, 0, (2,), 0)
    seg = Segment(meta, 0, [None] * length, np.zeros(length, np.int64), np.zeros(length, np.float32),
                  rng.normal(size=length).astype(np.float32), np.zeros(length, np.int8),
                  np.full(length, 2, np.int16), np.zeros(length, np.int64), end=end,
                  bootstrap_value=float(rng.normal()))
    seg.rewards = rng.normal(size=length).astype(np.float32)
    if combat:  # random runs of macro steps and combats
        cid, cur, nxt = [], -1, 0
        for _ in range(length):
            x = rng.random()
            if x < 0.3:
                cur = -1
            elif x < 0.45 or cur < 0:
                cur, nxt = nxt, nxt + 1
            cid.append(cur)
        seg.combat_id = np.array(cid, np.int32)
    else:
        seg.combat_id = np.full(length, -1, np.int32)
    return seg


# =========================================================================== GAE


@pytest.mark.parametrize("macro_micro", [False, True])
@pytest.mark.parametrize("gamma", [0.97, 1.0])
def test_gae_matches_brute_force(macro_micro: bool, gamma: float) -> None:
    rng = np.random.default_rng(7)
    cfg = GAEConfig(gamma=gamma, lam=0.9, macro_micro=macro_micro, lam_macro=0.8, lam_micro=0.6)
    segs = [make_segment(rng, int(rng.integers(1, 14)), SegmentEnd(int(rng.integers(3))), macro_micro)
            for _ in range(40)]
    adv, ret = gae(segs, cfg)
    ref = np.concatenate([ref_macro_micro(s, cfg) if macro_micro else ref_standard(s, cfg.gamma, cfg.lam)
                          for s in segs])
    np.testing.assert_allclose(adv, ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, adv + np.concatenate([s.values for s in segs]), rtol=1e-6, atol=1e-6)


def test_gamma_one_gae_and_returns_match_scalar_recursion() -> None:
    rng = np.random.default_rng(19)
    segs = [make_segment(rng, length, end) for length in (1, 7, 256) for end in SegmentEnd]
    expected_adv: list[float] = []
    expected_returns: list[float] = []
    for seg in segs:
        next_value = 0.0 if seg.end == SegmentEnd.TERMINAL else seg.bootstrap_value
        advantage = 0.0
        backward = []
        for reward, value in zip(reversed(seg.rewards.tolist()), reversed(seg.values.tolist())):
            advantage = reward + next_value - value + 0.99 * advantage
            backward.append(advantage)
            next_value = value
        reference = list(reversed(backward))
        expected_adv.extend(reference)
        expected_returns.extend(a + float(v) for a, v in zip(reference, seg.values))
    adv, ret = gae(segs, GAEConfig(gamma=1.0, lam=0.99))
    np.testing.assert_allclose(adv, expected_adv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ret, expected_returns, rtol=1e-6, atol=1e-6)


def test_macro_micro_reduces_to_standard() -> None:
    rng = np.random.default_rng(3)
    segs = [make_segment(rng, 11, SegmentEnd.CUT), make_segment(rng, 5, SegmentEnd.TERMINAL)]
    std = gae(segs, GAEConfig(gamma=0.95, lam=0.7))[0]
    all_macro = gae(segs, GAEConfig(gamma=0.95, macro_micro=True, lam_macro=0.7))[0]
    np.testing.assert_allclose(all_macro, std, rtol=1e-6, atol=1e-6)
    for s in segs:
        s.combat_id = np.zeros(len(s), np.int32)  # one combat spanning the segment
    all_micro = gae(segs, GAEConfig(gamma=0.95, macro_micro=True, lam_micro=0.7))[0]
    np.testing.assert_allclose(all_micro, std, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("stitch", [False, True])
def test_mixed_seat_trace_decay_matches_independent_reference(stitch: bool) -> None:
    cfg = GAEConfig(gamma=0.9998, macro_micro=True, lam_macro=0.81, lam_micro=0.65,
                    stitch_segments=stitch)
    wholes: list[Segment] = []
    fragments = []
    reference = []
    for seats in (1, 4, 2, 1):
        whole = make_segment(np.random.default_rng(92 + seats), 13, SegmentEnd.TERMINAL)
        whole.meta = EpisodeMeta(0, len(wholes), (2,) * seats, 10)
        # Seat-major rewards: four macro decisions; one union combat (including prompts);
        # another seat-major room chain, then the next union combat.
        whole.combat_id = np.asarray([-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 1, 1, -1], np.int32)
        per_episode = replace(cfg, lam_macro=cfg.lam_macro ** (1 / seats),
                              lam_micro=cfg.lam_micro ** (1 / seats))
        reference.append(ref_macro_micro(whole, per_episode))
        wholes.append(whole)
        fragments.extend(_split_for_gae(whole, (2, 6, 9)) if stitch else [whole])
    actual, _ = gae(fragments, cfg)
    np.testing.assert_allclose(actual, np.concatenate(reference), rtol=1e-6, atol=1e-6)


def test_single_seat_trace_is_bit_identical_and_full_lineup_is_retained() -> None:
    for lam in (0.0, 0.65, 0.95, 0.99, 1.0):
        assert seat_trace_decay(lam, 1).hex() == lam.hex()
        assert seat_trace_decay(lam, 4) ** 4 == pytest.approx(lam)
    whole = make_segment(np.random.default_rng(81), 19, SegmentEnd.TERMINAL, combat=True)
    cfg = GAEConfig(macro_micro=True, lam_macro=0.99, lam_micro=0.97)
    # Scalar stitched recursion was the existing path; the n=1 decay is exactly unchanged.
    a, r = gae([whole], cfg)
    b, s = gae(_split_for_gae(whole, (3, 8)), replace(cfg, stitch_segments=True))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(r, s)
    multi = replace(whole, meta=EpisodeMeta(0, 1, (4, 4, 2, 2), 10))
    # Complete auxiliary arrays are normally filled by RolloutStore.
    for segment in (whole, multi):
        segment.win = segment.floors_left = segment.hp_post = np.zeros(len(segment), np.float32)
        segment.win_mask = segment.hp_mask = np.ones(len(segment), bool)
    batch = build_batch([whole, multi], cfg, 0)
    assert batch.characters.shape == (38, 4)
    np.testing.assert_array_equal(batch.characters[0], [2, 0, 0, 0])
    np.testing.assert_array_equal(batch.characters[-1], [4, 4, 2, 2])
    np.testing.assert_array_equal(batch.n_seats, [1] * 19 + [4] * 19)


def test_gae_overrides_and_terminal_ignores_bootstrap() -> None:
    rng = np.random.default_rng(0)
    segs = [make_segment(rng, 6, SegmentEnd.TERMINAL), make_segment(rng, 4, SegmentEnd.TRUNCATED)]
    base = gae(segs, GAEConfig(gamma=0.9, lam=0.8))[0]
    shifted = gae(segs, GAEConfig(gamma=0.9, lam=0.8), bootstrap=np.array([123.0, segs[1].bootstrap_value], np.float32))[0]
    np.testing.assert_allclose(base, shifted, atol=1e-6)  # TERMINAL segments never bootstrap
    vals = np.concatenate([s.values for s in segs]) + 1.0
    over = gae(segs, GAEConfig(gamma=0.9, lam=0.8), values=vals)[0]
    assert not np.allclose(over, base)
    with pytest.raises(ValueError):
        gae(segs, GAEConfig(), values=vals[:-1])


def _split_for_gae(seg: Segment, cuts: tuple[int, ...]) -> list[Segment]:
    ends = (*cuts, len(seg))
    starts = (0, *cuts)
    return [replace(seg, start=lo, samples=seg.samples[lo:hi], actions=seg.actions[lo:hi],
                    rewards=seg.rewards[lo:hi], values=seg.values[lo:hi], combat_id=seg.combat_id[lo:hi],
                    end=seg.end if hi == len(seg) else SegmentEnd.CUT,
                    bootstrap_value=seg.bootstrap_value if hi == len(seg) else float(seg.values[hi]))
            for lo, hi in zip(starts, ends)]


@pytest.mark.parametrize("macro_micro", [False, True])
@pytest.mark.parametrize("end", list(SegmentEnd))
def test_stitched_gae_equals_uncut_reference(macro_micro: bool, end: SegmentEnd) -> None:
    whole = make_segment(np.random.default_rng(81), 47, end, combat=True)
    pieces = _split_for_gae(whole, (9, 17, 28))
    cfg = GAEConfig(gamma=0.97, lam=0.89, macro_micro=macro_micro, lam_macro=0.93, lam_micro=0.91)
    ref = ref_macro_micro(whole, cfg) if macro_micro else ref_standard(whole, cfg.gamma, cfg.lam)
    adv, ret = gae(pieces, replace(cfg, stitch_segments=True))
    np.testing.assert_allclose(adv, ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, ref + whole.values, rtol=1e-5, atol=1e-5)
    # Explicitly off retains the old segmented n-step-return mixture.
    old = np.concatenate([ref_macro_micro(s, cfg) if macro_micro else ref_standard(s, cfg.gamma, cfg.lam)
                          for s in pieces])
    np.testing.assert_allclose(gae(pieces, replace(cfg, stitch_segments=False))[0], old, rtol=1e-5, atol=1e-5)
    vals = whole.values + np.linspace(0, 1, len(whole), dtype=np.float32)
    boot = np.array([vals[9], vals[17], vals[28], 4.0], np.float32)
    override = replace(whole, values=vals, bootstrap_value=4.0)
    ref_override = ref_macro_micro(override, cfg) if macro_micro else ref_standard(override, cfg.gamma, cfg.lam)
    np.testing.assert_allclose(gae(pieces, replace(cfg, stitch_segments=True), vals, boot)[0],
                               ref_override, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("boundary", ["gap", "episode", "environment", "truncated", "terminal"])
@pytest.mark.parametrize("macro_micro", [False, True])
def test_stitched_gae_does_not_cross_missing_data_or_episode_end(boundary: str, macro_micro: bool) -> None:
    whole = make_segment(np.random.default_rng(8), 12, SegmentEnd.TERMINAL, combat=True)
    first, second = _split_for_gae(whole, (6,))
    if boundary == "gap":
        second = replace(second, start=7)
    elif boundary == "episode":
        second = replace(second, meta=replace(second.meta, episode_uid=1))
    elif boundary == "environment":
        second = replace(second, meta=replace(second.meta, env_id=1))
    else:
        first = replace(first, end=SegmentEnd.TRUNCATED if boundary == "truncated" else SegmentEnd.TERMINAL)
    cfg = GAEConfig(gamma=1.0, macro_micro=macro_micro)
    pieces = [first, second]
    np.testing.assert_array_equal(gae(pieces, replace(cfg, stitch_segments=True))[0], gae(pieces, cfg)[0])


def test_macro_trace_carries_delayed_strategic_credit_across_combat_and_storage_cut() -> None:
    whole = make_segment(np.random.default_rng(9), 202, SegmentEnd.TERMINAL)
    whole.values.fill(0)
    whole.rewards.fill(0)
    whole.rewards[-1] = 1
    whole.combat_id[1:-1] = 0
    pieces = _split_for_gae(whole, (100,))
    cfg = GAEConfig(gamma=1.0, lam=0.99, macro_micro=True, lam_macro=0.99, lam_micro=0.99,
                    stitch_segments=True)
    assert gae(pieces, cfg)[0][0] == pytest.approx(0.99)
    assert gae([whole], replace(cfg, macro_micro=False))[0][0] == pytest.approx(0.99 ** 201)
    assert gae(pieces, replace(cfg, stitch_segments=False))[0][0] == 0


# =========================================================================== RolloutStore


@pytest.fixture(scope="module")
def R(vocab: Vocab) -> dict[str, int]:
    return room_ids(vocab)


@pytest.fixture(scope="module")
def rooms(vocab: Vocab) -> RoomIds:
    return RoomIds.from_vocab(vocab)


def episode_facts(R: dict[str, int], n: int, result: int, rng: np.random.Generator) -> list[Facts]:
    """n decision frames alternating map / combat blocks, then a TERMINAL frame."""
    out, floor, hp = [], 0, 90
    for i in range(n):
        if i % 5 == 0:
            floor += 1
        hp = max(1, hp - int(rng.integers(0, 3)))
        ic = i % 5 >= 2
        room = R["Elite"] if ic and floor % 3 == 0 else (R["Monster"] if ic else R["Map"])
        out.append(frame_facts(obs_frame(Kind.COMBAT if ic else Kind.MAP, floor, room, ic, [hp],
                                         n_actions=1 + i % 3)))
    out.append(frame_facts(terminal_frame(result, floor, R["Map"], [hp if result == Result.WIN else 0])))
    return out


def feed(store: RolloutStore, env: int, facts: list[Facts], values: np.ndarray, version: int = 0) -> None:
    store.begin_episode(env, (2,), 0, seed_id=env)
    for t, f in enumerate(facts[:-1]):
        store.add_step(env, f, ("s", env, t), t % 2, -0.5, float(values[t]), version)


def test_store_segments_rewards_and_targets(R: dict[str, int], rooms: RoomIds) -> None:
    rng = np.random.default_rng(1)
    facts = episode_facts(R, 23, Result.WIN, rng)
    values = rng.normal(size=23).astype(np.float32)
    out: list[Segment] = []
    cfg = RewardConfig(gamma=0.99)
    store = RolloutStore(out.append, rooms, config=RolloutConfig(segment_len=8), rewards=cfg)
    feed(store, 3, facts, values)
    assert not out  # held until the episode resolves
    assert store.end_episode(3, facts[-1]) == Outcome.WIN
    assert [len(s) for s in out] == [8, 8, 7] and [s.start for s in out] == [0, 8, 16]
    assert [s.end for s in out] == [SegmentEnd.CUT, SegmentEnd.CUT, SegmentEnd.TERMINAL]
    assert out[0].bootstrap_value == pytest.approx(values[8]) and out[0].bootstrap_sample == ("s", 3, 8)
    t = episode_targets(facts, rooms, cfg)
    np.testing.assert_allclose(np.concatenate([s.rewards for s in out]), t.rewards)
    np.testing.assert_array_equal(np.concatenate([s.combat_id for s in out]), t.combat_id)
    assert all(s.win_mask.all() and (s.win == 1).all() for s in out)
    assert np.concatenate([s.n_legal for s in out]).tolist() == [1 + i % 3 for i in range(23)]
    assert [x for s in out for x in s.samples] == [("s", 3, i) for i in range(23)]
    assert store.stats.steps_released == 23 and store.stats.wins == 1
    # GAE over the store's segments equals the brute force over the whole episode
    g = GAEConfig(gamma=0.99, lam=0.9)
    adv = gae(out, g)[0]
    whole = make_segment(rng, 23, SegmentEnd.TERMINAL)
    whole.rewards, whole.values = t.rewards, values
    cut_ref = np.concatenate([ref_standard(s, 0.99, 0.9) for s in out])
    np.testing.assert_allclose(adv, cut_ref, rtol=1e-5, atol=1e-5)
    # segment cuts only truncate the lambda trace: TD(0) (lam=0) is cut-invariant
    np.testing.assert_allclose(gae(out, GAEConfig(gamma=0.99, lam=0.0))[0],
                               ref_standard(whole, 0.99, 0.0), rtol=1e-5, atol=1e-5)


def test_store_aborted_overflow_and_truncated(R: dict[str, int], rooms: RoomIds) -> None:
    rng = np.random.default_rng(2)
    out: list[Segment] = []
    store = RolloutStore(out.append, rooms, config=RolloutConfig(segment_len=4))
    ab = episode_facts(R, 10, Result.ABORTED, rng)
    feed(store, 0, ab, np.zeros(10))
    assert store.end_episode(0, ab[-1]) == Outcome.ABORTED
    assert not out and store.stats.aborted == 1 and store.stats.steps_dropped == 10
    # overflow mid-episode drops it at once
    store.begin_episode(1, (2,))
    f = frame_facts(obs_frame(Kind.MAP, 1, R["Map"], False, [50]))
    assert store.add_step(1, f, None, 0, 0.0, 0.0, 0)
    assert not store.add_step(1, f._replace(overflow=True), None, 0, 0.0, 0.0, 0)
    assert not store.in_episode(1) and store.stats.overflow == 1
    # truncation: all steps released, last segment bootstraps from the given next state
    tr = episode_facts(R, 6, Result.WIN, rng)
    feed(store, 2, tr, np.arange(6, dtype=np.float32))
    store.truncate_episode(2, tr[3], next_value=7.5, next_sample="next")
    assert [len(s) for s in out] == [4, 2] and out[-1].end == SegmentEnd.TRUNCATED
    assert out[-1].bootstrap_value == 7.5 and out[-1].bootstrap_sample == "next"
    assert not any(s.win_mask.any() for s in out) and store.stats.truncated == 1
    assert out[-1].truncated[-1] and not out[-1].done.any()
    # host loss
    feed(store, 4, tr, np.zeros(6))
    store.abort_episode(4, "lost")
    assert store.stats.lost == 1
    with pytest.raises(ValueError):
        store.abort_episode(4, "bogus")


def test_store_force_release_then_abort(R: dict[str, int], rooms: RoomIds) -> None:
    rng = np.random.default_rng(4)
    out: list[Segment] = []
    store = RolloutStore(out.append, rooms, config=RolloutConfig(segment_len=4, max_pending_steps=9))
    facts = episode_facts(R, 30, Result.ABORTED, rng)
    feed(store, 0, facts, np.zeros(30))
    released = sum(len(s) for s in out)
    assert store.stats.forced_releases >= 1 and released >= 16
    assert all(s.end == SegmentEnd.CUT and not s.win_mask.any() for s in out)
    full = episode_targets(facts, rooms, store.reward_cfg)
    np.testing.assert_allclose(np.concatenate([s.rewards for s in out]), full.rewards[:released])
    store.end_episode(0, facts[-1])
    st = store.stats
    assert st.aborted_after_release == 1 and st.steps_dropped == 30 - released
    assert st.steps_released + st.steps_dropped == st.steps_added


@pytest.mark.parametrize("reason", [TerminalReason.DECISION_CAP, TerminalReason.COMBAT_TURN_CAP])
@pytest.mark.parametrize("caps_as_failure", [False, True])
def test_cap_rollout_uses_configured_failure_semantics(R: dict[str, int], rooms: RoomIds,
                                                     reason: TerminalReason, caps_as_failure: bool) -> None:
    facts = [frame_facts(obs_frame(Kind.COMBAT, 8, R["Elite"], True, [50]))] * 5
    facts.append(frame_facts(terminal_frame(Result.LOSS, 8, R["Elite"], [50], reason=reason,
                                            in_combat=True, steps=5)))
    out: list[Segment] = []
    store = RolloutStore(out.append, rooms, config=RolloutConfig(segment_len=3),
                         rewards=RewardConfig(gamma=1.0, caps_as_failure=caps_as_failure))
    feed(store, 0, facts, np.full(5, 2.0))
    outcome = store.end_episode(0, facts[-1], bootstrap_value=7.0, bootstrap_sample="terminal")
    assert store.stats.host_truncated == 1 and store.stats.losses == 0
    assert out[0].end == SegmentEnd.CUT and out[0].bootstrap_value == 2.0
    last = out[-1]
    assert not any(s.hp_mask.any() for s in out)
    if caps_as_failure:
        assert outcome == Outcome.LOSS and last.end == SegmentEnd.TERMINAL
        assert last.bootstrap_value == 0.0 and last.bootstrap_sample is None
        assert last.done[-1] and not last.truncated.any()
        assert all(s.win_mask.all() and (s.win == 0).all() and (s.floors_left == 0).all() for s in out)
        assert gae(out, GAEConfig(gamma=1.0, lam=0.99))[1][-1] == pytest.approx(-0.15)
    else:
        assert outcome == Outcome.TRUNCATED and last.end == SegmentEnd.TRUNCATED
        assert last.bootstrap_value == 7.0 and last.bootstrap_sample == "terminal"
        assert not any(s.win_mask.any() for s in out)
        assert gae(out, GAEConfig(gamma=1.0, lam=0.99))[1][-1] == pytest.approx(7.0)


def test_store_with_fake_host_accounting(vocab: Vocab, rooms: RoomIds) -> None:
    host = FakeHost(vocab, seed=5, config=FakeHostConfig(mean_decisions=40, abort_prob=0.3, win_prob=0.5,
                                                         hashes=False))
    out: list[Segment] = []
    store = RolloutStore(out.append, rooms, config=RolloutConfig(segment_len=16))
    rng = np.random.default_rng(0)
    results = []
    for ep in range(25):
        f = host.reset(ep, (2,), 0)
        store.begin_episode(0, (2,), 0, seed_id=ep)
        while f.is_obs:
            a = int(rng.integers(f.n_actions))
            store.add_step(0, frame_facts(f), None, a, -1.0, 0.0, ep)
            f = host.step(a)
        results.append(store.end_episode(0, frame_facts(f)))
    st = store.stats
    assert st.steps_added == host.stats.decisions
    assert st.steps_released + st.steps_dropped == st.steps_added
    assert st.aborted == results.count(Outcome.ABORTED) == host.stats.aborted > 0
    assert st.wins + st.losses == host.stats.wins + host.stats.losses
    by_ep: dict[int, list[Segment]] = {}
    for s in out:
        by_ep.setdefault(s.meta.episode_uid, []).append(s)
    for segs in by_ep.values():
        assert [s.start for s in segs] == list(range(0, 16 * len(segs), 16))
        assert segs[-1].end == SegmentEnd.TERMINAL and all(s.end == SegmentEnd.CUT for s in segs[:-1])
    b = build_batch(out, GAEConfig(), current_version=30)
    assert len(b) == st.steps_released and b.staleness.min() >= 6 and len(b.segment_staleness) == len(out)


# =========================================================================== ring


def seg_of(n: int, version: int) -> Segment:
    s = make_segment(np.random.default_rng(n), n, SegmentEnd.CUT)
    s.policy_version[:] = version
    return s


def test_ring_take_evict_and_staleness() -> None:
    ring = SegmentRing(capacity_samples=30)
    for v in range(5):
        ring.put(seg_of(10, v))
    assert ring.n_samples == 30 and ring.stats.evicted_samples == 20
    assert ring.take(40, timeout=0.01) is None and ring.n_samples == 30
    got = ring.take(15, current_version=5, max_staleness=2)
    assert got is not None and [s.version_min for s in got] == [3, 4]  # version 2 is 3 behind: dropped
    assert ring.stats.stale_dropped_samples == 10 and ring.n_samples == 0
    with pytest.raises(ValueError):
        ring.take(1, max_staleness=1)


def test_ring_threaded_producer() -> None:
    ring = SegmentRing()

    def produce() -> None:
        for v in range(20):
            ring.put(seg_of(8, v))

    th = threading.Thread(target=produce)
    th.start()
    taken: list[Segment] = []
    while sum(len(s) for s in taken) < 160:
        got = ring.take(32, timeout=5.0)
        assert got is not None
        taken.extend(got)
    th.join()
    assert [s.version_min for s in taken] == list(range(20))
