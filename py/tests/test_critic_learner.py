"""Critic packages P3 (actor and store) and P4 (learner) of notes/critic/contract.md 8 on tree p4 (build-v1.md):
the gates PG4 (warm-up leaves the trunk alone), PG5 (split and linearity), PG6 (children, T1 / T9 with the critic
record), PG7 (switch), eager = captured with the tower, plus the store / ipc / actor plumbing and the train.py
config rules. PG3 (off = tree p4) is the identity tool (alt/critic/tools/critic_identity.py)."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import critic_data as cd
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy
from spire.critic import CriticConfig, CriticPPOConfig, crit_width, side
from spire.ipc import decode_segment, pack_segment
from spire.learner import Learner, PPOConfig, switch_test, transition_td
from spire.model import ModelConfig, SpireNet
from spire.rewards import RewardConfig, RoomIds, episode_targets
from spire.rollout import (GAEConfig, RolloutConfig, RolloutStore, Segment, SegmentEnd, build_batch, critic_streams,
                           gae, gae_columns, reward_groups)
from spire.schema import TerminalReason
from spire.synth import FakeHost, FakeHostConfig
from spire.train import TrainConfig, build, check_critic, config_dict
from spire.vocab import Vocab
from test_practice import _game, fights

K = 32
TINY_C = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32,
                     critic=CriticConfig())
TINY = dataclasses.replace(TINY_C, critic=None)
SOLO_GAE = GAEConfig(gamma=1.0, lam=0.99, macro_micro=True, lam_macro=0.99, lam_micro=0.99, stitch_segments=True)
GAES = {"standard": GAEConfig(gamma=0.9998, lam=0.95), "macro_micro": dataclasses.replace(SOLO_GAE, stitch_segments=False),
        "stitched": SOLO_GAE}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def strip_critic(segs: list[Segment]) -> list[Segment]:
    """The same segments without the critic record (what a tree-p4 actor stores)."""
    return [dataclasses.replace(s, values_c=None, crit=None, bootstrap_c=None, rewards_c=None) for s in segs]


@pytest.fixture(scope="module")
def critic_run() -> dict[str, Any]:
    """A CPU critic actor (TorchPolicy critic=True, fake hosts with RESET bit 10) and its segments."""
    torch.manual_seed(0)
    on, _off = cd.tokenizers("calc")
    net = SpireNet(on.layout, TINY_C)
    policy = TorchPolicy(net, on, use_graphs=False, seed=3, critic=True)
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(cd.vocab()), config=RolloutConfig(segment_len=32), critic_k=K)
    hosts = [FakeHost(cd.vocab(), seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0),
                      training_hidden=True) for i in range(4)]
    frames: dict[int, list[Any]] = {}
    reps: list[dict] = []
    actor = Actor(hosts, policy, on, EpisodeSampler(EpisodeDist(), cd.vocab(), 0),
                  ActorConfig(reset_flags=cd.flags_of("calc"), max_batch=8, min_group=2, report_interval=0.1),
                  store=store, on_decision=lambda e, f, a: frames.setdefault(e, []).append(f),
                  on_report=reps.append)
    actor.run(max_decisions=1200)
    rep = actor.take_report()
    return {"segs": segs, "net": net, "tok": on, "frames": frames, "reps": reps + [rep]}


# =========================================================================== store, actor, ipc


def test_store_and_actor_records(critic_run: dict[str, Any]) -> None:
    """Every critic segment carries values_c (n, 3), CRIT (n, 2 + K) = critic.side of its frames, bootstrap_c and
    rewards_c with R_WIN + R_PROG + R_SURV = rewards; values_c = the tower's forward_critic on the stored sample with
    its CRIT (CPU fp32); a TERMINAL segment's bootstrap_c is zero; the actor reports its CRIT counts."""
    segs, net, tok = critic_run["segs"], critic_run["net"], critic_run["tok"]
    assert sum(len(s) for s in segs) > 600
    n_rows = 0
    for s in segs:
        n = len(s)
        assert s.values_c.shape == (n, 3) and s.values_c.dtype == np.float32
        assert s.crit.shape == (n, crit_width(K)) and s.crit.dtype == np.int16
        assert s.rewards_c.shape == (n, 3) and s.bootstrap_c.shape == (3,)
        np.testing.assert_allclose(s.rewards_c.astype(np.float64).sum(1), s.rewards, atol=2e-7)
        if s.end == SegmentEnd.TERMINAL:
            assert not s.bootstrap_c.any()
        n_rows += int(s.crit[:, 1].sum())
    assert n_rows > 0, "the fake hosts drew HDRAW rows"
    # CRIT = critic.side of the frame (its sample's card_base), values_c = forward_critic of the stored sample
    seg = max(segs, key=lambda s: int(s.crit[:, 1].sum()))
    samples = list(seg.samples)
    batch = tok.collate(samples)
    with torch.no_grad():
        out = net.eval().forward_critic(batch, torch.as_tensor(seg.crit))
    np.testing.assert_allclose(out["vc"].numpy(), seg.values_c, rtol=0, atol=1e-5)
    np.testing.assert_allclose(out["value"].numpy(), seg.values, rtol=0, atol=1e-5)
    rep = critic_run["reps"]
    tot = {k: sum(r["critic"][k] for r in rep) for k in ("frames", "hdraw_rows", "rows_kept", "truncated")}
    assert tot["frames"] >= 1200 and tot["hdraw_rows"] == tot["rows_kept"] + tot["truncated"] > 0


def test_crit_rows_are_the_frames_side(critic_run: dict[str, Any]) -> None:
    """The stored CRIT row of every step is critic.side(frame, card_base, n_tokens, K) of the frame the step was
    taken at (the actor's decisions per env, in order, against that env's released steps)."""
    tok, frames = critic_run["tok"], critic_run["frames"]
    checked = 0
    for env in {s.meta.env_id for s in critic_run["segs"]}:
        segs = sorted((s for s in critic_run["segs"] if s.meta.env_id == env), key=lambda s: (s.meta.episode_uid, s.start))
        rows = np.concatenate([s.crit for s in segs])
        fr = frames[env][:len(rows)]
        assert len(fr) == len(rows)
        for f, row in zip(fr, rows):
            enc = tok.encode_batch([f])
            assert np.array_equal(side(f, int(enc.card_base[0]), int(enc.n_tokens[0]), K), row)
            checked += 1
    assert checked > 600


def test_ipc_round_trip(critic_run: dict[str, Any]) -> None:
    for s in critic_run["segs"][:20]:
        d = decode_segment(pack_segment(s))
        for name in ("values_c", "crit", "bootstrap_c", "rewards_c", "values", "rewards"):
            a, b = getattr(s, name), getattr(d, name)
            assert a.dtype == b.dtype and np.array_equal(a, b), name
    plain = strip_critic(critic_run["segs"][:3])
    for s in plain:
        d = decode_segment(pack_segment(s))
        assert d.values_c is None and d.crit is None and d.bootstrap_c is None and d.rewards_c is None


def test_store_refusals_and_scouts(vocab: Vocab) -> None:
    """A critic store needs the critic record on every step and the tower's bootstrap at a truncation; a plain store
    refuses it; a scout (facts only) releases nothing, so its steps never reach the learner."""
    rooms = RoomIds.from_vocab(vocab)
    facts, term = _game(vocab, 3)
    segs: list[Segment] = []
    st = RolloutStore(segs.append, rooms, critic_k=K)
    st.begin_episode(0, (2,))
    with pytest.raises(AssertionError, match="critic record"):
        st.add_step(0, facts[0], 1, 0, -1.0, 0.0, 0)
    vc, cr = np.zeros(3, np.float32), np.full(crit_width(K), -1, np.int16)
    st.add_step(0, facts[0], 1, 0, -1.0, 0.0, 0, critic=(vc, cr))
    with pytest.raises(AssertionError, match="bootstrap values"):
        st.truncate_episode(0, facts[1], 0.5, 2)
    st.truncate_episode(0, facts[1], 0.5, 2, np.ones(3, np.float32))
    assert len(segs) == 1 and segs[0].end == SegmentEnd.TRUNCATED and (segs[0].bootstrap_c == 1).all()
    plain = RolloutStore(segs.append, rooms)
    plain.begin_episode(0, (2,))
    with pytest.raises(AssertionError, match="critic record"):
        plain.add_step(0, facts[0], 1, 0, -1.0, 0.0, 0, critic=(vc, cr))
    segs.clear()
    st.begin_episode(5, (2,), scout=True)
    for f in facts[:20]:
        st.add_scout_step(5, f)
    st.end_episode(5, term)
    assert not segs and st.scout_stats.steps_added == 20 and st.stats.steps_released == 1


# =========================================================================== PG5: split and linearity


def _episode_facts(vocab: Vocab) -> dict[str, tuple[list, Any]]:
    """Fake games ending in a WIN, a death, a decision cap and a host truncation (TERMINAL facts)."""
    out: dict[str, tuple[list, Any]] = {}
    for seed in range(400):
        facts, term = _game(vocab, seed)
        name = {1: "win", 2: "death"}.get(term.result)
        if name and not term.truncated and name not in out:
            out[name] = facts, term
        if len(out) == 2:
            break
    facts, term = out["death"]
    out["cap"] = facts, term._replace(terminal_reason=int(TerminalReason.DECISION_CAP))
    out["truncated"] = facts, term._replace(terminal_reason=int(TerminalReason.OTHER_TRUNCATION))
    return out


def test_pg5_reward_split(vocab: Vocab) -> None:
    """R_WIN + R_PROG + R_SURV against the float64 reward within 4 ulp on episodes with a win, a death, a cap and a
    truncation (and an unfinished prefix); exactly one group carries the terminal term."""
    rooms = RoomIds.from_vocab(vocab)
    seen = set()
    for name, (facts, term) in _episode_facts(vocab).items():
        for cfg in (RewardConfig(), RewardConfig(caps_as_failure=False)):
            for fs in ([*facts, term], facts[:57]):
                t = episode_targets(fs, rooms, cfg)
                r64 = sum(t.components64.values(), np.zeros(len(t.rewards)))
                g = reward_groups(t)
                assert g.shape == (len(t.rewards), 3)
                assert (np.abs(g.sum(1) - r64) <= 4 * np.spacing(np.abs(r64))).all(), (name, cfg)
                assert np.array_equal(r64.astype(np.float32), t.rewards)
                seen.add(t.outcome.name)
                if t.outcome.name == "WIN":
                    assert g[-1, 0] == 1.0 and not g[:-1, 0].any()
                else:
                    assert not g[:, 0].any()
    assert {"WIN", "LOSS", "TRUNCATED", "ONGOING"} <= seen, seen


def _critic_segments(vocab: Vocab, seg_len: int, seed: int = 0) -> list[Segment]:
    """Critic segments of several fake games (random values, values_c, bootstraps), in release order."""
    rooms = RoomIds.from_vocab(vocab)
    rng = np.random.default_rng(seed)
    segs: list[Segment] = []
    st = RolloutStore(segs.append, rooms, config=RolloutConfig(segment_len=seg_len), critic_k=K)
    games = _episode_facts(vocab)
    jobs = [*games.items(), ("cut", games["win"])]  # "cut": stopped by the caller (truncate_episode)
    for env, (name, (facts, term)) in enumerate(jobs):
        st.begin_episode(env, (2,) if env % 2 else (2, 3), 10)
        cr = np.full(crit_width(K), -1, np.int16)
        cut = len(facts) if name != "cut" else len(facts) - 11
        for t in range(cut):
            st.add_step(env, facts[t], 100 + t, int(rng.integers(0, 5)), -float(rng.random()), float(rng.normal()), 1,
                        critic=(rng.normal(size=3).astype(np.float32), cr))
        if name == "cut":
            st.truncate_episode(env, facts[cut], float(rng.normal()), 7, rng.normal(size=3).astype(np.float32))
        else:
            st.end_episode(env, term)
    return segs


@pytest.mark.parametrize("seg_len", [256, 7])
@pytest.mark.parametrize("mode", list(GAES))
def test_pg5_gae_columns(vocab: Vocab, seg_len: int, mode: str) -> None:
    """Column 0 of the 4-column GAE = today's gae bit for bit; sum_k A_k = the advantage of (sum_k R_k, sum_k V_k)
    within 1e-12 in float64 (macro_micro with stitching included, segment lengths 256 and 7); build_batch's
    advantages / returns of critic segments = today's."""
    cfg = GAES[mode]
    segs = _critic_segments(vocab, seg_len)
    assert any(s.end == SegmentEnd.CUT for s in segs)
    v4, b4, r4 = critic_streams(segs)
    a4, vv = gae_columns(segs, cfg, v4, b4, r4)
    adv0, ret0 = gae(strip_critic(segs), cfg)
    assert np.array_equal(a4[:, 0].astype(np.float32), adv0) and np.array_equal((a4[:, 0] + vv[:, 0]).astype(np.float32), ret0)
    a_sum, _ = gae_columns(segs, cfg, v4[:, 1:].sum(1, keepdims=True), b4[:, 1:].sum(1, keepdims=True),
                           r4[:, 1:].sum(1, keepdims=True))
    assert np.abs(a4[:, 1:].sum(1) - a_sum[:, 0]).max() <= 1e-12
    rb = build_batch(segs, cfg, 1)
    rb0 = build_batch(strip_critic(segs), cfg, 1)
    assert np.array_equal(rb.advantages, rb0.advantages) and np.array_equal(rb.returns, rb0.returns)
    assert np.array_equal(rb.ret_c, a4[:, 1:] + vv[:, 1:]) and rb.values_c.shape == (len(rb), 3)
    f32, _ = gae(segs, cfg, v4, b4, r4)  # the component-axis form of gae itself
    assert f32.shape == (len(rb), 4) and np.array_equal(f32[:, 0], adv0)


# =========================================================================== PG6: children


def _flat4(segs: list[Segment], cfg: GAEConfig) -> dict[str, np.ndarray]:
    segs = sorted(segs, key=lambda s: s.start)
    a4, v4 = gae_columns(segs, cfg, *critic_streams(segs))
    cat = lambda k: np.concatenate([getattr(s, k) for s in segs])  # noqa: E731
    return {"adv4": a4, "ret4": a4 + v4, "rewards_c": cat("rewards_c"), "values_c": cat("values_c"),
            "rewards": cat("rewards"), "crit": cat("crit")}


@pytest.mark.parametrize("game_child", [False, True])
def test_pg6_children_numbers_equal_the_full_run(vocab: Vocab, game_child: bool) -> None:
    """T1 (fight-only children, truncated at the first frame after the fight with that frame's value and values_c)
    and T9 (GAME children to the run's end) with the critic record: the child's rewards_c, values_c, CRIT and the
    4-column advantages and returns are the full run's on every decision of the child, bit for bit, at segment
    lengths 256 and 7, from a trained parent and from a scout (facts only)."""
    rooms = RoomIds.from_vocab(vocab)
    games = _episode_facts(vocab)
    checked = 0
    for gname in ("win", "death"):
        facts, term = games[gname]
        n = len(facts)
        rng = np.random.default_rng(n)
        values = rng.normal(size=n + 1).astype(np.float32)
        vcs = rng.normal(size=(n + 1, 3)).astype(np.float32)
        crit = rng.integers(-1, 50, size=(n + 1, crit_width(K))).astype(np.int16)
        logp = -rng.random(n).astype(np.float32)
        actions = rng.integers(0, 5, n)

        def step(st: RolloutStore, env: int, t: int) -> None:
            st.add_step(env, facts[t], 100 + t, int(actions[t]), float(logp[t]), float(values[t]), 1,
                        critic=(vcs[t], crit[t]))

        for seg_len in (256, 7):
            segs: list[Segment] = []
            st = RolloutStore(segs.append, rooms, config=RolloutConfig(segment_len=seg_len),
                              rewards=RewardConfig(gamma=1.0), critic_k=K)
            st.begin_episode(0, (2,), 10, 5)
            for t in range(n):
                step(st, 0, t)
            st.end_episode(0, term)
            full = _flat4([s for s in segs if s.meta.env_id == 0], SOLO_GAE)
            for k, (a, b) in enumerate(fights(facts)):
                for scout in (False, True):
                    segs.clear()
                    st.begin_episode(1, (2,), 10, 5, scout=scout)
                    for t in range(a):
                        st.add_scout_step(1, facts[t]) if scout else step(st, 1, t)
                    st.fork_episode(st.fork_point(1), 2, game=game_child)
                    st.abort_episode(1, "shutdown")
                    last = n if game_child else b + 1
                    for t in range(a, last):
                        step(st, 2, t)
                    if last < n:
                        st.truncate_episode(2, facts[last], float(values[last]), 100 + last, vcs[last])
                    else:
                        st.end_episode(2, term)
                    kid = _flat4([s for s in segs if s.meta.env_id == 2], SOLO_GAE)
                    for key in ("adv4", "ret4", "rewards_c", "values_c", "rewards", "crit"):
                        assert np.array_equal(kid[key], full[key][a:last]), (gname, key, a, b, seg_len, scout)
                    checked += 1
    assert checked >= 2 * 2 * 2 * 4


# =========================================================================== PG4: warm-up leaves the trunk alone


def _pg4_learners(tok: Any, critic: CriticPPOConfig, seed: int = 5) -> tuple[Learner, Learner]:
    cfg = PPOConfig(epochs=2, minibatch=256, micro_batch=128, lr=1e-3, seed=0, gae=SOLO_GAE, kl_lr_target=0.002,
                    weight_decay=0.01)
    torch.manual_seed(seed)
    off = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu", fast=False)
    torch.manual_seed(seed)
    on = Learner(SpireNet(tok.layout, TINY_C), tok, dataclasses.replace(cfg, critic=critic), device="cpu", fast=False)
    return off, on


@pytest.mark.parametrize("source", ["public", "auto"])
def test_pg4_warmup_leaves_the_trunk_alone(critic_run: dict[str, Any], source: str) -> None:
    """Fixed segments, CPU fp32, eager: two updates with ppo.critic (adv_source "public", and "auto" in its warm-up)
    and two without the critic (= tree p4, PG3): every non-critic parameter, its Adam moments and step, its EMA and
    the update's PPO metrics bit-identical; the tower moved (its own groups, lr 3e-4)."""
    torch.set_num_threads(1)
    tok = critic_run["tok"]
    segs = critic_run["segs"]
    off, on = _pg4_learners(tok, CriticPPOConfig(adv_source=source))
    tower = {id(p) for p in on.model.critic.parameters()}
    assert [n for n, _ in off.model.named_parameters()] == [n for n, p in on.model.named_parameters()
                                                             if id(p) not in tower]
    t0 = [p.detach().clone() for p in on.model.critic.parameters()]
    for _ in range(2):
        m_off = off.update(strip_critic(segs))
        m_on = on.update(segs)
        assert m_on["critic"]["beta"] == 0.0
    for key in ("loss_pi", "loss_v", "loss_win", "loss_hp", "approx_kl", "grad_norm", "kl_lr_scale_min",
                "explained_variance", "adv_std"):
        assert m_off[key] == m_on[key], key
    trunk_on = [(n, p) for n, p in on.model.named_parameters() if id(p) not in tower]
    for (n, a), b in zip(trunk_on, off.model.parameters()):
        assert torch.equal(a, b), n
    ema_on = [p for n, p in on.ema.named_parameters() if not n.startswith("critic.")]
    for a, b in zip(ema_on, off.ema.parameters()):
        assert torch.equal(a, b)
    st_on, st_off = on.opt.state, off.opt.state
    for (n, a), b in zip(trunk_on, off.model.parameters()):
        for k in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(st_on[a][k], st_off[b][k]), (n, k)
    assert any(not torch.equal(a, b) for a, b in zip(t0, on.model.critic.parameters()))
    crit_groups = [g for g in on.opt.param_groups if g.get("critic")]
    assert len(crit_groups) == 2 and all(g["lr"] == 3e-4 for g in crit_groups)
    row = m_on["critic"]
    for k in ("ev_win", "ev_prog", "ev_surv", "ev_total", "ev_public", "var_adv_combat_pub", "var_adv_combat_tower",
              "loss_win", "loss_prog", "loss_surv", "td", "switch_test", "switch_overdue"):
        assert k in row, k
    assert set(row["td"]) == {"td_end_won", "td_enter_fight", "td_in_fight", "td_reward_screens"}


def test_beta_mixes_the_advantages(critic_run: dict[str, Any]) -> None:
    """adv_source "tower": A = A_TOWER = sum_k A_k and the public head regresses onto RET_WIN + RET_PROG + RET_SURV
    (public_target "tower"); "public" with public_target "public" keeps today's numbers."""
    tok, segs = critic_run["tok"], critic_run["segs"]
    _, on = _pg4_learners(tok, CriticPPOConfig(adv_source="tower"))
    rb = build_batch(segs, on.cfg.gae, on.version)
    a_pub = rb.advantages.copy()
    st = on._critic_begin(rb, segs)
    assert on._beta == 1.0 and np.array_equal(st["a_pub"], a_pub)
    assert np.array_equal(rb.advantages, rb.adv_c.sum(1).astype(np.float32))
    assert np.array_equal(rb.returns, rb.ret_c.sum(1).astype(np.float32))
    m = on.update(segs)
    assert m["critic"]["beta"] == 1.0


# =========================================================================== PG7: switch


def test_pg7_switch(tmp_path: Path, critic_run: dict[str, Any]) -> None:
    """Synthetic metric streams through Learner.switch_step: no switch before warmup_updates even when both
    conditions hold; the switch at the first update where both hold (each alone is not enough); beta 0, 0.5, 1 at 0,
    50, 100 updates after it; one-way; switched_at, the start and the window survive a checkpoint resume."""
    tok = critic_run["tok"]
    c = CriticPPOConfig(warmup_updates=300, switch_window=50, blend_updates=100)
    _, lr = _pg4_learners(tok, c)
    good = [100.0, 0.5, 1000.0, 0.0, 200.0, 0.0, 150.0]  # td 0.005, var ratio 0.75
    bad_td = [100.0, 2.0, 1000.0, 0.0, 200.0, 0.0, 150.0]  # td 0.02
    bad_var = [100.0, 0.5, 1000.0, 0.0, 200.0, 0.0, 250.0]  # ratio 1.25
    assert switch_test([good] * 50, c)["pass"] and not switch_test([good] * 49, c)["ready"]
    assert not switch_test([bad_td] * 50, c)["pass"] and not switch_test([bad_var] * 50, c)["pass"]
    # (a) both conditions hold all along: no switch in the warm-up, the switch at its end (300 updates after the start)
    _, la = _pg4_learners(tok, c)
    la.critic_first = la.updates = 1000
    for i in range(300):
        assert la.critic_beta() == 0.0 and la.switched_at is None
        la.updates += 1
        out = la.switch_step(good)
    assert out.get("switched_now") and la.switched_at == 1300
    # (b) a stream where first the TD condition, then the variance condition fails: the switch exactly at the first
    # update (after the warm-up) whose window passes both, as switch_test computes it on the same window
    stream = [bad_td] * 330 + [bad_var] * 70 + [good] * 80
    lr.critic_first = lr.updates = 1000
    expect = None
    for k, stats in enumerate(stream):
        if expect is None and k + 1 >= 300 and switch_test((stream[:k + 1])[-50:], c)["pass"]:
            expect = 1000 + k + 1
        assert lr.critic_beta() == 0.0
        lr.updates += 1
        out = lr.switch_step(stats)
        if lr.switched_at is not None:
            break
    assert expect is not None and expect > 1400 and lr.switched_at == expect == lr.updates
    assert out.get("switched_now")
    s = lr.switched_at
    window = [list(r) for r in lr._switch_window]
    betas = {}
    for u in range(0, 151):
        lr.updates = s + u
        betas[u] = lr.critic_beta()
        lr.switch_step(bad_td)  # one-way: bad statistics after the switch change nothing
    assert betas[0] == 0.0 and betas[50] == 0.5 and betas[100] == 1.0 and betas[150] == 1.0
    assert lr.switched_at == s
    # resume
    lr.updates = s + 20
    lr._switch_window = window
    path = tmp_path / "ck.pt"
    lr.save(path)
    _, lr2 = _pg4_learners(tok, c)
    lr2.load(path)
    assert (lr2.switched_at, lr2.critic_first, lr2.updates) == (s, 1000, s + 20)
    assert lr2._switch_window == window and lr2.critic_beta() == 0.2
    # adv_source "public" / "tower" force beta
    for src, beta in (("public", 0.0), ("tower", 1.0)):
        _, l3 = _pg4_learners(tok, dataclasses.replace(c, adv_source=src))
        assert l3.critic_beta() == beta


def test_transition_td_classes(critic_run: dict[str, Any]) -> None:
    segs = critic_run["segs"]
    td = transition_td(segs, 0.9998)
    n = sum(len(s) for s in segs)
    cls = td["classes"]
    assert all(m.shape == (n,) for m in cls.values())
    tot = sum(m.astype(int) for m in cls.values())
    assert tot.max() <= 1 and cls["td_in_fight"].sum() > 0 and cls["td_end_won"].sum() > 0
    rb = build_batch(segs, SOLO_GAE, 1)
    assert (rb.in_combat[cls["td_end_won"]]).all() and not rb.in_combat[cls["td_enter_fight"]].any()


# =========================================================================== migration through Learner.load


def test_migrate_critic_through_learner_load(tmp_path: Path, critic_run: dict[str, Any]) -> None:
    """A tree-p4 learner's checkpoint loads into a critic learner with migrate_critic only: every old weight, EMA and
    Adam entry exact, the tower as at birth, its groups empty; then it trains; the one-migration rule holds."""
    tok, segs = critic_run["tok"], critic_run["segs"]
    off, _ = _pg4_learners(tok, CriticPPOConfig())
    off.update(strip_critic(segs))
    path = tmp_path / "p4.pt"
    off.save(path)
    _, on = _pg4_learners(tok, CriticPPOConfig(), seed=9)
    with pytest.raises((AssertionError, ValueError)):
        on.load(path)
    _, on = _pg4_learners(tok, CriticPPOConfig(), seed=9)
    with pytest.raises(AssertionError, match="separately"):
        on.load(path, migrate_critic=True, migrate_rules=True)
    _, on = _pg4_learners(tok, CriticPPOConfig(), seed=9)
    on.load(path, migrate_critic=True)
    tower = {id(p) for p in on.model.critic.parameters()}
    for (n, a), b in zip([(n, p) for n, p in on.model.named_parameters() if id(p) not in tower],
                         off.model.parameters()):
        assert torch.equal(a, b), n
        for k in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(on.opt.state[a][k], off.opt.state[b][k])
    assert on.updates == off.updates and on.critic_first is None
    torch.manual_seed(0)
    birth = SpireNet(tok.layout, TINY_C).critic.state_dict()
    for k, v in on.model.critic.state_dict().items():
        assert torch.equal(v, birth[k]), k
    m = on.update(segs)
    assert m["critic"]["first_update"] == off.updates and on.critic_first == off.updates
    p2 = tmp_path / "critic.pt"
    on.save(p2)
    _, again = _pg4_learners(tok, CriticPPOConfig(), seed=9)
    with pytest.raises(AssertionError):
        again.load(p2, migrate_critic=True)
    _, again = _pg4_learners(tok, CriticPPOConfig(), seed=9)
    again.load(p2)
    assert again.critic_first == off.updates


# =========================================================================== train.py


def _raw(tmp_path: Path, **over: Any) -> dict[str, Any]:
    from test_train import small_config
    d = config_dict(small_config(tmp_path, "c"))
    d.update(over)
    return d


def test_train_config_rules(tmp_path: Path) -> None:
    """config.json without the critic keys = tree p4's form; ppo.critic and model.critic come together; bit 10 never
    in actor.reset_flags; ppo.critic round-trips through config.json; --migrate-critic is never persisted."""
    d = _raw(tmp_path)
    assert "critic" not in d["ppo"] and "critic" not in d["model"] and "migrate_critic" not in d
    on = json.loads(json.dumps(d))
    on["model"]["critic"] = {}
    on["ppo"]["critic"] = {"adv_source": "public"}
    cfg = build(TrainConfig, on)
    check_critic(cfg)
    assert cfg.ppo.critic == CriticPPOConfig(adv_source="public") and cfg.model.config().critic == CriticConfig()
    back = config_dict(cfg)
    assert back["ppo"]["critic"]["adv_source"] == "public" and build(TrainConfig, back).ppo.critic == cfg.ppo.critic
    assert "migrate_critic" not in back and config_dict(dataclasses.replace(cfg, migrate_critic=True))["migrate_critic"]
    half = json.loads(json.dumps(d))
    half["ppo"]["critic"] = {}
    with pytest.raises(AssertionError, match="come together"):
        check_critic(build(TrainConfig, half))
    half = json.loads(json.dumps(d))
    half["model"]["critic"] = {}
    with pytest.raises(AssertionError, match="come together"):
        check_critic(build(TrainConfig, half))
    bad = json.loads(json.dumps(on))
    bad["actor"]["reset_flags"] = int(bad["actor"]["reset_flags"]) | 1024
    with pytest.raises(AssertionError, match="bit 10"):
        check_critic(build(TrainConfig, bad))
    bad = json.loads(json.dumps(on))
    bad["ppo"]["recompute_values"] = True
    with pytest.raises(AssertionError, match="recompute_values"):
        build(TrainConfig, bad)
    bad = json.loads(json.dumps(on))
    bad["ppo"]["critic"]["adv_weights"] = {"10": [1, 0.3, 1]}
    with pytest.raises(AssertionError, match="adv_weights"):
        build(TrainConfig, bad)
    with pytest.raises(AssertionError, match="--migrate-critic"):
        check_critic(dataclasses.replace(build(TrainConfig, d), migrate_critic=True))


def test_training_run_with_the_critic(tmp_path: Path) -> None:
    """A processes-topology run with fake hosts: a tree-p4 run (no critic keys) for 2 updates, then resumed once with
    --migrate-critic semantics (model.critic + ppo.critic, migrate_critic) for 2 more: the actors send critic
    segments through the shm ring (RESET bit 10 on training_hidden hosts), the learner rows carry ``critic``, the
    train rows the CRIT counts; config.json keeps ppo.critic and drops the migration flag."""
    from spire.metrics import read_jsonl
    from spire.train import Supervisor
    from test_train import small_config
    cfg = small_config(tmp_path, "crit", max_updates=2)
    assert Supervisor(cfg).run() == 0
    run = tmp_path / "crit"
    on = config_dict(cfg)
    on["model"]["critic"] = {}
    on["ppo"]["critic"] = {}
    on["max_updates"] = 4
    cfg2 = dataclasses.replace(build(TrainConfig, on), migrate_critic=True)
    assert Supervisor(cfg2, resume=True).run() == 0
    rows = [r for r in read_jsonl(run / "metrics.jsonl") if r["kind"] == "learner"]
    crit_rows = [r for r in rows if "critic" in r]
    assert len(crit_rows) == 2 and crit_rows[0]["critic"]["first_update"] == 2 and crit_rows[-1]["updates"] == 4
    trains = [r for r in read_jsonl(run / "metrics.jsonl") if r["kind"] == "train" and "critic_frames" in r]
    assert trains and sum(r["critic_frames"] for r in trains) > 0
    saved = json.loads((run / "config.json").read_text())
    assert "critic" in saved["ppo"] and "migrate_critic" not in saved
    ck = torch.load(run / "ckpt" / "latest.pt", map_location="cpu", weights_only=True)
    assert ck["counters"]["critic"]["first_update"] == 2 and any(k.startswith("critic.") for k in ck["model"])


# =========================================================================== eager = captured


@cuda
def test_critic_fast_and_graph_updates_match_the_generic_path(critic_run: dict[str, Any]) -> None:
    """One update with the tower through the generic eager path, the fast eager path, the fast CUDA-graph path and
    (tree p7) the compiled fast CUDA-graph path (``forward_critic`` torch.compile'd), from the same weights on the same
    critic segments: the same parameters (tower included) and diagnostics within the tolerance of
    tests/test_learner_fast.py, the critic losses too."""
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    tok, segs = critic_run["tok"], critic_run["segs"]
    cfg = PPOConfig(epochs=2, minibatch=256, micro_batch=128, micro_tokens=4096, lr=1e-4, seed=3,
                    critic=CriticPPOConfig(adv_source="tower", lr=1e-4))
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY_C)
    ls = {name: Learner(copy.deepcopy(base), tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                        fast=fast, graphs=graphs, compile_model=comp)
          for name, fast, graphs, comp in (("generic", False, False, False), ("fast", True, False, False),
                                           ("graphs", True, True, False), ("compiled", True, True, True))}
    assert ls["compiled"].stepper is not None and ls["compiled"].stepper.compiled
    p0 = [p.detach().float().cpu().clone() for p in ls["generic"].model.parameters()]
    ms = {k: lr.update(segs) for k, lr in ls.items()}
    assert not ms["generic"]["fast_path"] and ms["fast"]["fast_path"] and ms["graphs"]["fast_path"]
    assert ms["compiled"]["fast_path"] and ms["compiled"]["stepper"]["captures"] == ms["graphs"]["stepper"]["captures"]
    st = ls["compiled"].state()  # checkpoints hold the plain module's keys (no torch.compile wrapper prefix)
    assert set(st["model"]) == set(ls["graphs"].state()["model"]) and not any("_orig_mod" in k for k in st["model"])
    deltas = {k: [p.detach().float().cpu() - q for p, q in zip(lr.model.parameters(), p0)] for k, lr in ls.items()}
    for k in ("fast", "graphs", "compiled"):
        for d_ref, d in zip(deltas["generic"], deltas[k]):
            assert (d - d_ref).abs().max() <= 2.5 * cfg.lr * cfg.epochs * 2, k
        num = sum(float((d - r).abs().sum()) for d, r in zip(deltas[k], deltas["generic"]))
        den = sum(float(r.abs().sum()) for r in deltas["generic"])
        assert num / den < 0.05, (k, num / den)
        for key in ("loss_pi", "loss_v", "entropy", "approx_kl", "grad_norm"):
            assert ms[k][key] == pytest.approx(ms["generic"][key], rel=0.05, abs=2e-3), (k, key)
        for key in ("loss_win", "loss_prog", "loss_surv", "grad_norm"):
            assert ms[k]["critic"][key] == pytest.approx(ms["generic"]["critic"][key], rel=0.05, abs=2e-3), (k, key)
        assert ms[k]["critic"]["ev_total"] == ms["generic"]["critic"]["ev_total"]
    for k in ("graphs", "compiled"):
        tower_moved = [float(d.abs().max()) for d, (n, _) in zip(deltas[k], ls[k].model.named_parameters())
                       if n.startswith("critic.")]
        assert max(tower_moved) > 0, k
    # a second update: no new capture for the shapes already seen, as on the graph path
    for k in ("graphs", "compiled"):
        ls[k].update(segs)
    assert ls["compiled"].stepper.captures == ls["graphs"].stepper.captures  # type: ignore[union-attr]


@cuda
@pytest.mark.parametrize("compiled", [False, True])
def test_gpu_actor_graph_records_the_tower(critic_run: dict[str, Any], compiled: bool) -> None:
    """The training actor on CUDA (packed CUDA graphs, bf16, forward_critic in the graph; tree p7: also the graphs of a
    torch.compile'd forward_critic): the stored values_c are the tower's values of the stored sample and CRIT (CPU
    fp32 recomputation within bf16 tolerance) on a tower whose weights were perturbed (at birth it is constant), the
    stored public values those of forward."""
    if compiled:
        import torch._dynamo as dynamo
        dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    tok = critic_run["tok"]
    torch.manual_seed(0)
    net = copy.deepcopy(critic_run["net"])
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in net.critic.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    ref = copy.deepcopy(net).eval()
    policy = TorchPolicy(net.cuda(), tok, use_graphs=True, seed=3, critic=True, compile_model=compiled)
    assert policy.runner is not None and (policy.runner._critic_forward == net.forward_critic) != compiled
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(cd.vocab()), config=RolloutConfig(segment_len=32), critic_k=K)
    hosts = [FakeHost(cd.vocab(), seed=10 + i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0),
                      training_hidden=True) for i in range(8)]
    actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(), cd.vocab(), 0),
                  ActorConfig(reset_flags=cd.flags_of("calc"), max_batch=16, min_group=4), store=store)
    actor.run(max_decisions=800)
    assert policy.runner is not None and policy.runner.captures > 0
    n = 0
    for s in segs:
        with torch.no_grad():
            out = ref.forward_critic(tok.collate(list(s.samples)), torch.as_tensor(s.crit))
        assert np.abs(out["vc"].numpy() - s.values_c).max() < 0.05
        assert np.abs(out["value"].numpy() - s.values).max() < 0.05
        n += len(s)
    assert n > 300 and float(np.std(np.concatenate([s.values_c[:, 1] for s in segs]))) > 1e-3


@pytest.mark.skipif(__import__("sys").platform != "linux", reason="practice forks (Linux)")
@pytest.mark.parametrize("scouts", [False, True])
def test_practice_actor_with_the_critic(tmp_path: Path, scouts: bool) -> None:
    """A critic training actor with practice (fake fork server started for RESET bit 10, scouts on / off): every
    released segment carries the critic record; a fight-only child's truncated last segment bootstraps the tower
    with its post-fight frame's values (not zeros); scouts release nothing (their steps never reach the learner)."""
    import sys
    from spire import gym
    from spire.critic import training_reset_flags
    from spire.gym import PracticeConfig, Priors
    from spire.host import HostConfig, HostProcess
    from spire.schema import RESET_FLAGS_DEFAULT
    from spire.tokenize import Tokenizer, TokenizerConfig
    import test_practice as tp
    vocab = cd.vocab()
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True, cycle_ban=True, commit_select=True))
    tok.allow_hidden = True
    torch.manual_seed(0)
    policy = TorchPolicy(SpireNet(tok.layout, TINY_C), tok, use_graphs=False, seed=1, critic=True)
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16), critic_k=K)
    tmp_path.mkdir(parents=True, exist_ok=True)
    host = HostProcess(HostConfig(command=(sys.executable, "-B", tp.SERVER, "--seed", "0", "--mean-decisions", "80",
                                           "--marker", tmp_path.name), start_timeout=30.0, reset_timeout=10.0,
                                  step_timeout=10.0, log_path=str(tmp_path / "h0.log"), name="g0"),
                       training_hidden=True)
    recs: list = []
    errs: list = []
    practice = PracticeConfig(actors=1, scouts=scouts, priors=Priors(0.25, 0.35, 0.55, 0.9),
                              socket_dir=str(tmp_path / "s"))
    try:
        actor = Actor([host], policy, tok, EpisodeSampler(EpisodeDist(ascensions=((0, 1.0),)), vocab, 0),
                      ActorConfig(max_batch=16, min_group=1, gym_env_id_base=tp.KID_BASE,
                                  reset_flags=training_reset_flags(int(RESET_FLAGS_DEFAULT))),
                      store=store, on_episode=recs.append, on_error=errs.append, practice=practice, practice_q=0.5,
                      practice_children=8, gym_seed=5)
        assert actor.gym is not None and actor.gym.mode == gym.MODE_RESAMPLE
        actor.run(lambda: sum(not r.gym for r in recs) >= 3, max_seconds=300.0)
    finally:
        host.close()
    assert not errs
    kids = [s for s in segs if s.meta.env_id >= tp.KID_BASE]
    assert kids and all(s.values_c is not None and s.rewards_c is not None for s in segs)
    trunc = [s for s in kids if s.end == SegmentEnd.TRUNCATED]
    assert trunc and all(np.array_equal(s.bootstrap_c, np.asarray([0.5, 0.0, 0.0], np.float32)) for s in trunc)
    if scouts:
        assert store.scout_stats.episodes > 0 and store.scout_stats.steps_released == 0
        assert not [s for s in segs if s.meta.env_id < tp.KID_BASE]  # the scouts' own games release nothing


def test_resume_from_a_cpu_migrated_checkpoint(tmp_path: Path, critic_run: dict[str, Any]) -> None:
    """A checkpoint migrated on the CPU (scripts/migrate_critic.py: tower at birth, its Adam groups, the proof in
    extra["critic_migration"], config and counters of the source) starts a critic learner with migrate_critic and no
    change; a tower that is not at birth, or no proof, is refused."""
    from spire.critic import optimizer_groups
    from spire.critic_migration import migrate_critic_optimizer, migrate_critic_weights
    tok, segs = critic_run["tok"], critic_run["segs"]
    off, _ = _pg4_learners(tok, CriticPPOConfig())
    off.update(strip_critic(segs))
    src = tmp_path / "p4.pt"
    off.save(src)
    ck = torch.load(src, map_location="cpu", weights_only=True)
    torch.manual_seed(0)
    net = SpireNet(tok.layout, TINY_C)
    opt = torch.optim.AdamW(optimizer_groups(net, 0.01, 3e-4))
    out = {**ck, "model": migrate_critic_weights(ck["model"], net), "ema": migrate_critic_weights(ck["ema"], net),
           "optimizer": migrate_critic_optimizer(ck["optimizer"], net, opt),
           "extra": {**ck.get("extra", {}), "critic_migration": {"tool": "test"}}}
    cpu = tmp_path / "cpu-migrated.pt"
    torch.save(out, cpu)
    _, on = _pg4_learners(tok, CriticPPOConfig(), seed=9)
    on.load(cpu, migrate_critic=True)
    for k, v in on.model.state_dict().items():
        assert torch.equal(v, out["model"][k]), k
    on.update(segs)
    bad = {**out, "extra": {}}
    torch.save(bad, tmp_path / "bad.pt")
    _, again = _pg4_learners(tok, CriticPPOConfig(), seed=9)
    with pytest.raises(AssertionError, match="proof"):
        again.load(tmp_path / "bad.pt", migrate_critic=True)
    moved = {**out, "model": {k: v + 1 if k.startswith("critic.") and v.is_floating_point() else v
                              for k, v in out["model"].items()}}
    torch.save(moved, tmp_path / "moved.pt")
    _, again = _pg4_learners(tok, CriticPPOConfig(), seed=9)
    with pytest.raises(AssertionError, match="birth"):
        again.load(tmp_path / "moved.pt", migrate_critic=True)
