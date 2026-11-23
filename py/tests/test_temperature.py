"""spire.temperature (notes/next/temperature-spec.md v1): the invariants of spec section 4 on synthetic data and fake
hosts (the real-frame halves of invariants 2 and 6 are scripts/temp_check.py on real hosts), the start-up rules and
the metrics."""
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
from spire.distributions import choice_uniform_probs_numpy, exploration_table
from spire.ipc import decode_segment, pack_segment
from spire.learner import Learner, PPOConfig, _Accum
from spire.model import ModelConfig, SpireNet
from spire.proto import Frame
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import G, Kind
from spire.synth import FakeHost, FakeHostConfig
from spire.taste import TasteConfig
from spire.temperature import combat_steps, step_temperatures, tempered_log_probs
from spire.tokenize import BatchShape, Tokenizer, TokenizerConfig
from spire.trainstep import DEN_COLS, NF, PF_COLS, MicroStepper, _Res
from spire.vocab import Vocab

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32)
EXPL = ((int(Kind.CARD_REWARD), 0.1), (int(Kind.CARD_SELECT), 0.1), (int(Kind.COMBAT), 0.05))
T_Q = 0.5
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def noise_eval_combat(f: Frame) -> bool:
    """alt/solo/noise/noise_eval.py line 77, verbatim: combat = IN_COMBAT and kind COMBAT / CARD_SELECT."""
    g = f.gvals
    kind = Kind(int(g[G.KIND]))
    return bool(g[G.IN_COMBAT]) and kind in (Kind.COMBAT, Kind.CARD_SELECT)


def actor_run(vocab: Vocab, net: SpireNet, n: int, temperature: float, *, beh_temp: bool = True, seed: int = 7,
              on_decision=None, tok: Tokenizer | None = None) -> tuple[list[Segment], Actor]:
    tok = tok or Tokenizer(vocab)
    pol = TorchPolicy(net, tok, seed=seed, exploration_by_kind=EXPL, combat_temperature=temperature)
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32), beh_temp=beh_temp)
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
             for i in range(4)]
    actor = Actor(hosts, pol, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0), ("IRONCLAD", 1.0))), vocab, 0),
                  ActorConfig(), store=store, on_decision=on_decision)
    actor.run(max_decisions=n)
    return segs, actor


def policy(vocab: Vocab, temperature: float, seed: int = 0) -> TorchPolicy:
    tok = Tokenizer(vocab)
    return TorchPolicy(SpireNet(tok.layout, TINY), tok, seed=seed, exploration_by_kind=EXPL,
                       combat_temperature=temperature)


# ------------------------------------------------------------------ start-up rules


def test_config_rules(vocab: Vocab) -> None:
    assert PPOConfig().combat_temperature == 1.0
    PPOConfig(combat_temperature=0.5)
    PPOConfig(combat_temperature=1.0, advice=(("REST", "SMITH", 0.5, 0.05, 0.0),), taste=TasteConfig("x", 0.5, 0.7))
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(AssertionError, match="finite"):
            PPOConfig(combat_temperature=bad)
    with pytest.raises(AssertionError, match="advice"):
        PPOConfig(combat_temperature=0.5, advice=(("REST", "SMITH", 0.5, 0.05, 0.0),))
    with pytest.raises(AssertionError, match="taste"):
        PPOConfig(combat_temperature=0.5, taste=TasteConfig("x", 0.5, 0.7))
    with pytest.raises(AssertionError, match="anchor"):
        PPOConfig(combat_temperature=0.5, decoupled_clip_anchor=True)
    tok = Tokenizer(vocab)
    net = SpireNet(tok.layout, TINY)
    with pytest.raises(AssertionError, match="sampling-policy"):
        TorchPolicy(net, tok, mode="greedy", combat_temperature=0.5)
    # a tempered policy needs a store that records the column
    store = RolloutStore(lambda s: None, RoomIds.from_vocab(vocab))
    with pytest.raises(AssertionError, match="beh_temp"):
        Actor([FakeHost(vocab, seed=0)], TorchPolicy(net, tok, combat_temperature=0.5), tok,
              EpisodeSampler(EpisodeDist(), vocab, 0), ActorConfig(), store=store)


# ------------------------------------------------------------------ invariant 3: the formula


def _rows(n_legal: list[int], width: int = 5) -> np.ndarray:
    rng = np.random.default_rng(5)
    z = np.full((len(n_legal), width), -np.inf, np.float32)
    for i, n in enumerate(n_legal):
        z[i, :n] = rng.normal(0, 2, n).astype(np.float32)
    return z


def _hand(z: np.ndarray, t: float, eps: float, ref: np.ndarray) -> np.ndarray:
    """q = (1 - eps) softmax(z / t) + eps ref over the legal entries, by hand in float64."""
    legal = np.isfinite(z)
    e = np.where(legal, np.exp((np.where(legal, z, 0) - z[legal].max()) / t), 0.0)
    return (1 - eps) * e / e.sum() + eps * ref


@pytest.mark.parametrize("eps", [0.0, 0.2])
@pytest.mark.parametrize("choice", [False, True])
def test_formula_by_hand(vocab: Vocab, eps: float, choice: bool) -> None:
    """Steps with 1, 2 and 5 legal rows, combat and non-combat: the actor's log q (TorchPolicy._sample_all) and the
    learner's (tempered_log_probs) equal the hand value; a non-combat step does not change with T; one row: log q 0."""
    n_legal = [1, 2, 5, 1, 2, 5]
    combat = np.array([True, True, True, False, False, False])
    z = _rows(n_legal)
    legal = np.isfinite(z)
    keys = np.tile(np.array([0, 0, 1, 2, 2], np.int64), (len(z), 1))  # choices {0, 1}, {2}, {3, 4}
    ref = choice_uniform_probs_numpy(legal, keys) if choice else legal / legal.sum(1, keepdims=True)
    for t in (1.0, T_Q, 2.0):
        temps = step_temperatures(combat, t)
        assert temps.dtype == np.float32 and (temps[~combat] == 1).all() and (temps[combat] == np.float32(t)).all()
        pol = policy(vocab, t)
        _, _, _, lq_actor = pol._sample_all(z.copy(), np.full(len(z), eps), keys if choice else None,
                                            temps if t != 1.0 else None)
        lq_learner = tempered_log_probs(torch.from_numpy(z), torch.from_numpy(temps), torch.full((len(z),), eps),
                                        torch.from_numpy(ref.astype(np.float32))).double().numpy()
        for i in range(len(z)):
            want = np.log(_hand(z[i], float(temps[i]), eps, ref[i])[legal[i]])
            np.testing.assert_allclose(lq_actor[i, legal[i]], want, rtol=0, atol=1e-5)
            np.testing.assert_allclose(lq_learner[i, legal[i]], want, rtol=0, atol=1e-5)
            assert (lq_learner[i, ~legal[i]] == 0).all()
        # single row: log q = 0 in the actor; in the learner 0 without mixing, and with eps > 0 the float32 rounding
        # of the existing T = 1 mixture formula, log((1 - eps) + eps) = -1.5e-8, unchanged by T
        assert (lq_actor[[0, 3], 0] == 0).all()
        if eps == 0:
            assert (lq_learner[[0, 3], 0] == 0).all()
        if t == 1.0:
            base = lq_learner
        else:
            np.testing.assert_array_equal(lq_learner[[0, 3], 0], base[[0, 3], 0])
            np.testing.assert_array_equal(lq_learner[~combat], base[~combat])  # non-combat: T-independent
            assert not np.allclose(lq_learner[[1, 2]], base[[1, 2]])


def test_predicate_and_temperatures() -> None:
    kinds = np.array([Kind.COMBAT, Kind.CARD_SELECT, Kind.CARD_SELECT, Kind.COMBAT, Kind.MAP, Kind.CARD_REWARD,
                      Kind.REST], np.int64)
    ic = np.array([1, 1, 0, 0, 1, 1, 0])
    assert combat_steps(kinds, ic).tolist() == [True, True, False, False, False, False, False]
    assert step_temperatures(combat_steps(kinds, ic), 0.5).tolist() == [0.5, 0.5, 1, 1, 1, 1, 1]


# ------------------------------------------------------------------ invariant 4: sampling


def _chi2_sf_even(x: float, df: int) -> float:
    """Chi-square survival function for an even number of degrees of freedom (closed form)."""
    assert df % 2 == 0
    h = x / 2
    return math.exp(-h) * sum(h ** k / math.factorial(k) for k in range(df // 2))


def test_sampling_matches_the_tempered_softmax(vocab: Vocab) -> None:
    """200,000 draws of the actor's sampler on one logit vector at T = 0.5 (eps 0, a combat row) match
    softmax(z / 0.5): chi-square p > 0.001; and they do not match softmax(z)."""
    z = np.array([1.0, 0.3, -0.4, 0.8, -1.5], np.float32)
    n = 200_000
    pol = policy(vocab, T_Q, seed=3)
    a, lp, _, _ = pol._sample_all(np.tile(z, (n, 1)), np.zeros(n), None, np.full(n, T_Q, np.float32))
    counts = np.bincount(a, minlength=len(z))
    for t, ok in ((T_Q, True), (1.0, False)):
        p = np.exp(z / t - (z / t).max())
        p /= p.sum()
        chi2 = float(((counts - n * p) ** 2 / (n * p)).sum())
        pv = _chi2_sf_even(chi2, len(z) - 1)
        assert (pv > 1e-3) == ok, (t, chi2, pv)
        if ok:
            np.testing.assert_allclose(lp, np.log(p[a]).astype(np.float32), rtol=0, atol=1e-5)


# ------------------------------------------------------------------ invariant 6: the combat-step predicate


def test_actor_flag_is_noise_evals_predicate(vocab: Vocab) -> None:
    """On fake-host frames: the actor's per-decision flag (asserted inside the actor to equal the policy's flag from
    the sample's global token) counts exactly noise_eval's combat steps; the policy's flag recomputed from the
    encoded frames equals noise_eval's frame by frame; the stored beh_temp is 0.5 exactly there."""
    frames: list[Frame] = []
    flags: list[bool] = []

    def on_decision(env_id: int, f: Frame, a: int) -> None:
        frames.append(f)
        flags.append(noise_eval_combat(f))

    torch.manual_seed(0)
    tok = Tokenizer(vocab)
    segs, actor = actor_run(vocab, SpireNet(tok.layout, TINY), 2000, T_Q, on_decision=on_decision, tok=tok)
    nf = np.asarray(flags)
    assert 200 < nf.sum() < len(nf) - 200, nf.sum()
    assert actor.counters.combat_steps == int(nf.sum()) and actor.counters.noncombat_steps == len(nf) - int(nf.sum())
    enc = tok.encode_batch(frames, sort=False)
    pol = TorchPolicy(SpireNet(tok.layout, TINY), tok)
    g0 = enc.row0[:len(frames)]
    assert np.array_equal(combat_steps(enc.raw[g0, pol._kind_slot], enc.raw[g0, pol._in_combat_slot]), nf)
    temps = np.concatenate([s.beh_temp for s in segs])
    combat = np.concatenate([(s.combat_id >= 0) & np.isin(s.kinds, (Kind.COMBAT, Kind.CARD_SELECT)) for s in segs])
    assert temps.dtype == np.float32 and (temps[combat] == T_Q).all() and (temps[~combat] == 1).all()


# ------------------------------------------------------------------ invariant 1: off = identical


def test_off_is_identical_in_this_tree(vocab: Vocab) -> None:
    """T = 1 with the beh_temp column recorded: the same actions, log-probs and values as without it (no extra draw),
    beh_temp all ones, the column survives the codec; one learner update with and without the column (and with the
    config value 0.5 on all-ones data) gives bit-identical parameters."""
    torch.manual_seed(0)
    net = SpireNet(Tokenizer(vocab).layout, TINY)
    a, _ = actor_run(vocab, copy.deepcopy(net), 1500, 1.0, beh_temp=True)
    b, _ = actor_run(vocab, copy.deepcopy(net), 1500, 1.0, beh_temp=False)
    assert len(a) == len(b) and all(x.beh_temp is not None and y.beh_temp is None for x, y in zip(a, b))
    for x, y in zip(a, b):
        for f in ("actions", "logp", "values", "kinds", "n_legal"):
            assert np.array_equal(getattr(x, f), getattr(y, f)), f
        assert x.beh_temp.dtype == np.float32 and (x.beh_temp == 1).all()
    a = [decode_segment(pack_segment(s)) for s in a]
    assert all(s.beh_temp is not None and (s.beh_temp == 1).all() for s in a)
    tok = Tokenizer(vocab)
    params = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # CPU bit identity across learner instances needs one thread (as tests/identity_ref.py)
    try:
        for data, t in ((a, 1.0), (b, 1.0), (a, T_Q)):
            cfg = PPOConfig(minibatch=512, micro_batch=256, epochs=1, exploration_by_kind=EXPL, combat_temperature=t)
            lr = Learner(copy.deepcopy(net), tok, cfg, device="cpu", fast=False)
            m = lr.update(data)
            assert m["beh_temp_mean"] == 1.0 and not lr._tempered
            params.append([p.detach().clone() for p in lr.model.state_dict().values()])
    finally:
        torch.set_num_threads(threads)
    for other in params[1:]:
        assert all(torch.equal(x, y) for x, y in zip(params[0], other))
    with pytest.raises(AssertionError, match="beh_temp records"):
        Learner(copy.deepcopy(net), tok, PPOConfig(exploration_by_kind=EXPL, combat_temperature=T_Q), device="cpu",
                fast=False).update(b)


def _ref(tree: Path, script: str) -> dict:
    env = {**os.environ, "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1",
           "FEATURES": ""}
    out = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name(script))], env=env, capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_off_is_the_base_tree() -> None:
    """SPIRE_BASE_TREE = tree n (alt/calc/py): its tests/identity_ref.py (features off) and this tree's
    tests/temperature_ref.py (beh_temp recorded, combat_temperature 1.0) print the same hashes of every recorded
    action / log-prob / value / kind / legal count and of the parameters after one learner update."""
    base = os.environ.get("SPIRE_BASE_TREE")
    if not base:
        pytest.skip("SPIRE_BASE_TREE (tree n) not set")
    mine = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": base, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1",
           "FEATURES": ""}
    out = subprocess.run([sys.executable, "-B", str(Path(base) / "tests" / "identity_ref.py")], env=env,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    ref = json.loads(out.stdout.strip().splitlines()[-1])
    new = _ref(mine, "temperature_ref.py")
    assert new.pop("beh_temp_ones") is True and new.pop("combat_steps") > 100
    assert new == ref and ref["decisions"] >= 2000


# ------------------------------------------------------------------ invariant 2: consistency (fake hosts)


def test_learner_reproduces_the_stored_behaviour(vocab: Vocab) -> None:
    """T = 0.5, eps > 0 on COMBAT / CARD_SELECT: after the codec, the learner's first minibatch (the whole batch, before
    any step, the acting weights) has ratio 1 on every step - and not when the column is dropped (T = 1 accounting)."""
    torch.manual_seed(0)
    tok = Tokenizer(vocab)
    net = SpireNet(tok.layout, TINY)
    segs, actor = actor_run(vocab, copy.deepcopy(net), 1500, T_Q)
    segs = [decode_segment(pack_segment(s)) for s in segs]
    temps = np.concatenate([s.beh_temp for s in segs])
    assert (temps == T_Q).sum() > 200 and (temps == 1).sum() > 200
    n = sum(len(s) for s in segs)
    for on in (True, False):
        cfg = PPOConfig(minibatch=4 * n, micro_batch=256, epochs=1, exploration_by_kind=EXPL,
                        combat_temperature=T_Q if on else 1.0)
        lr = Learner(copy.deepcopy(net), tok, cfg, device="cpu", fast=False)
        m = lr.update(segs if on else [dataclasses.replace(s, beh_temp=None) for s in segs])
        if on:
            assert lr._tempered and m["first_minibatch_kl"] < 1e-10 and m["clipfrac"] == 0.0, m
            assert m["beh_temp_mean"] == pytest.approx(float(temps.mean()))
            assert m["entropy_beh_combat"] < m["entropy_base_combat"] and m["combat_policy_samples"] > 100
        else:
            assert m["first_minibatch_kl"] > 1e-6, m


# ------------------------------------------------------------------ invariant 5: paths


class _StaticLogits(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        zeros = self.logits.new_zeros(len(self.logits))
        return {"logits": self.logits, "value": zeros, "pwin_logit": zeros, "floors_pred": zeros, "hp_pred": zeros}


def _microbody(temps: torch.Tensor, tempered: bool) -> tuple[torch.Tensor, torch.Tensor, _Accum]:
    logits = torch.tensor([[2.0, -2.0, 1.0], [-1.0, 0.5, 2.0], [0.0, -torch.inf, -torch.inf],
                           [-torch.inf, -torch.inf, -torch.inf]])
    model = _StaticLogits(logits)
    cfg = PPOConfig(exploration_by_kind=((int(Kind.COMBAT), 0.2),), ent_coef=0, vf_coef=0, aux_win=0,
                    aux_floors=0, aux_hp=0, combat_temperature=T_Q)
    st = MicroStepper.__new__(MicroStepper)
    st.model, st.fwd, st.cfg, st.autocast, st.tempered = model, model, cfg, False, tempered
    st._choice_verb_slot = 0
    st._exploration = torch.from_numpy(exploration_table(cfg.exploration_by_kind))
    st._uniform_kl = None
    st.acc = _Accum(torch.device("cpu"))
    st.den = torch.ones(len(DEN_COLS))
    st.den[DEN_COLS.index("pol")] = 2
    st.den[DEN_COLS.index("all")] = 3
    st.mb_kl = torch.zeros(())
    pf = torch.zeros(4, NF)
    pf[:, PF_COLS.index("logp_b")] = torch.tensor([0.7, 0.2, 1.0, 1.0]).log()
    pf[:, PF_COLS.index("logp_ref")] = pf[:, PF_COLS.index("logp_b")]
    pf[:, PF_COLS.index("adv")] = torch.tensor([1.0, -0.5, 0.0, 0.0])
    pf[:2, PF_COLS.index("pol")] = 1
    pf[:3, PF_COLS.index("valid")] = 1
    pf[:, PF_COLS.index("log_legal")] = torch.tensor([3.0, 3.0, 2.0, 2.0]).log()
    pf[:, PF_COLS.index("t_keep")] = 1.0
    pf[:, PF_COLS.index("beh_temp")] = temps
    pf[:, PF_COLS.index("combat")] = (temps != 1).float()
    pi = torch.tensor([[0, int(Kind.COMBAT)], [1, int(Kind.MAP)], [0, int(Kind.REST)], [0, 0]])
    views = {"raw": torch.zeros(12, 1, dtype=torch.int16), "rel": torch.zeros(12, 3, dtype=torch.int64),
             "dest": torch.arange(4, 16), "act_mask": torch.isfinite(logits), "tok_mask": torch.ones(4, 1, dtype=torch.bool),
             "kid_mask": torch.empty(0, dtype=torch.bool), "var_mask": torch.empty(0, dtype=torch.bool)}
    st._body(_Res(BatchShape(4, 1, 3, 0, 0, 0, 12), torch.empty(0), views, pf, pi))
    assert model.logits.grad is not None
    return model.logits.grad.clone(), st.mb_kl.clone(), st.acc


def test_fast_microbody_tempered_gradient_and_off_identity() -> None:
    """The captured micro body on CPU: the tempered body on all-ones temperatures is bit-identical to the untempered
    body; on a tempered combat row the gradient is that of log((1 - e) softmax(z / T)(a) + e / 3)."""
    ones = torch.ones(4)
    g0, k0, _ = _microbody(ones, False)
    g1, k1, _ = _microbody(ones, True)
    assert torch.equal(g0, g1) and torch.equal(k0, k1)
    g2, k2, acc = _microbody(torch.tensor([T_Q, 1, 1, 1]), True)
    z = torch.tensor([[2.0, -2.0, 1.0], [-1.0, 0.5, 2.0]], requires_grad=True)
    t = torch.tensor([T_Q, 1.0])[:, None]
    e = torch.tensor([0.2, 0.0])[:, None]
    mix = (1 - e) * (z / t).softmax(-1) + e / 3
    p = mix[torch.arange(2), torch.tensor([0, 1])]
    ratio = p / torch.tensor([0.7, 0.2])
    adv = torch.tensor([1.0, -0.5])
    loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).sum() / 2
    want, = torch.autograd.grad(loss, z)
    torch.testing.assert_close(g2[:2], want, rtol=1e-5, atol=1e-7)
    assert not torch.allclose(g2[:1], g0[:1]) and torch.equal(g2[1:], g0[1:])
    q = mix[0].detach()
    assert float(acc.n_c) == 1 and float(acc.ent_beh_c) == pytest.approx(float(-(q * q.log()).sum()), rel=1e-6)


@pytest.fixture(scope="module")
def tempered_segments(vocab: Vocab) -> list[Segment]:
    torch.manual_seed(0)
    tok = Tokenizer(vocab, TokenizerConfig(token_buckets=(20, 28, 40, 64, 80, 96, 128, 192, 256, 384)))
    segs, _ = actor_run(vocab, SpireNet(tok.layout, TINY), 3000, T_Q, tok=tok)
    return [decode_segment(pack_segment(s)) for s in segs]


def _params(lr: Learner) -> list[torch.Tensor]:
    return [p.detach().float().cpu().clone() for p in lr.model.parameters()]


@cuda
def test_tempered_paths_agree(vocab: Vocab, tempered_segments: list[Segment]) -> None:
    """Generic eager, fast eager, fast CUDA graphs and compiled CUDA graphs on tempered segments: the same parameters
    within the bf16 / summation-order tolerance of tests/test_learner_fast.py, the same diagnostics (the tempered
    ones included); the graph path captured its graphs under the tempered body."""
    import torch._dynamo as dynamo
    dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)
    tok = Tokenizer(vocab, TokenizerConfig(token_buckets=(20, 28, 40, 64, 80, 96, 128, 192, 256, 384)))
    cfg = PPOConfig(epochs=2, minibatch=512, micro_batch=256, micro_tokens=4096, lr=1e-4, seed=3, kl_lr_target=.01,
                    exploration_by_kind=EXPL, combat_temperature=T_Q)
    torch.manual_seed(0)
    base = SpireNet(tok.layout, ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16,
                                            d_static=32))
    ls = {}
    for name, fast, graphs, comp in (("generic", False, False, False), ("fast", True, False, False),
                                     ("graphs", True, True, False), ("compiled", True, True, True)):
        net = copy.deepcopy(base)
        net.bag_atomic = comp
        ls[name] = Learner(net, tok, cfg, device="cuda", size_fn=lambda s: tok.config.bucket(s.n_tokens),
                           fast=fast, graphs=graphs, compile_model=comp)
    p0 = _params(ls["generic"])
    ms = {k: lr.update(tempered_segments) for k, lr in ls.items()}
    assert all(lr._tempered for lr in ls.values())
    assert set(ls["graphs"].stepper._res[next(iter(ls["graphs"].stepper._res))].graphs) == {True}
    deltas = {k: [a - b for a, b in zip(_params(lr), p0)] for k, lr in ls.items()}
    for k in ("fast", "graphs", "compiled"):
        for d_ref, d in zip(deltas["generic"], deltas[k]):
            assert (d - d_ref).abs().max() <= 2.5 * cfg.lr * cfg.epochs * 2, k
        num = sum(float((d - r).abs().sum()) for d, r in zip(deltas[k], deltas["generic"]))
        den = sum(float(r.abs().sum()) for r in deltas["generic"])
        assert num / den < 0.05, (k, num / den)
        for key in ("loss_pi", "loss_v", "entropy", "approx_kl", "clipfrac_epoch0", "ratio_dev_epoch0", "grad_norm",
                    "first_minibatch_kl", "entropy_beh_combat", "entropy_base_combat"):
            assert ms[k][key] == pytest.approx(ms["generic"][key], rel=0.05, abs=2e-3), (k, key)
        for key in ("samples", "minibatches", "nonfinite_skipped", "illegal_chosen", "beh_temp_mean",
                    "combat_policy_samples"):
            assert ms[k][key] == ms["generic"][key], (k, key)
    # an untempered iteration on the same learner captures its own graphs and keeps the tempered ones
    plain = [dataclasses.replace(s, beh_temp=np.ones(len(s), np.float32)) for s in tempered_segments]
    ls["graphs"].update(plain)
    assert not ls["graphs"]._tempered
    assert any(set(r.graphs) == {True, False} for r in ls["graphs"].stepper._res.values())


# ------------------------------------------------------------------ metrics end to end (fake hosts, supervisor)


def test_training_run_reports_the_temperature_metrics(tmp_path: Path) -> None:
    from spire.metrics import read_jsonl
    from spire.train import Supervisor
    from test_train import small_config

    cfg = small_config(tmp_path, "temp", topology="threads", max_updates=3,
                       ppo={"minibatch": 256, "micro_batch": 256, "epochs": 1, "lr": 1e-3, "combat_temperature": T_Q,
                            "exploration_by_kind": [[int(Kind.CARD_REWARD), 0.1]]},
                       learner={"device": "cpu", "iteration_samples": 1024, "micro_tokens": None,
                                "checkpoint_interval_s": 3600, "allow_ppo_config_change": True})
    assert Supervisor(cfg).run() == 0
    rows = list(read_jsonl(cfg.run_dir / "metrics.jsonl"))
    learner = [r for r in rows if r["kind"] == "learner"]
    train = [r for r in rows if r["kind"] == "train" and r["combat_steps"]]
    assert [r["version"] for r in learner] == [1, 2, 3]
    assert all(0.5 < r["beh_temp_mean"] < 1 and r["entropy_beh_combat"] < r["entropy_base_combat"] for r in learner)
    assert train and all(0 <= r["combat_nonargmax"] <= 1 and 0 <= r["noncombat_nonargmax"] <= 1 for r in train)
