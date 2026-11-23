"""Legal exploration floors, exact behaviour probabilities, and differentiable PPO accounting."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spire.actor import TorchPolicy
from spire.distributions import exploration_table, kind_coefficients, legal_uniform_log_probs, uniform_cross_entropy
from spire.learner import PPOConfig, _Accum
from spire.model import ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.schema import G, Kind
from spire.synth import FrameGen, random_frame
from spire.tokenize import BatchShape, Tokenizer
from spire.trainstep import DEN_COLS, NF, PF_COLS, MicroStepper, _Res
from spire.vocab import Vocab

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32)


def test_off_reproduces_original_log_probs_and_gradients() -> None:
    z = torch.tensor([[2.0, -2.0, -torch.inf], [-200.0, 0.0, 4.0]], requires_grad=True)
    legal = torch.isfinite(z)
    old = torch.where(legal, torch.log_softmax(z.masked_fill(~legal, -1e30), -1), 0.0)
    new = legal_uniform_log_probs(z)
    assert torch.equal(old, new)
    old_grad, = torch.autograd.grad(old.sum(), z, retain_graph=True)
    new_grad, = torch.autograd.grad(new.sum(), z)
    assert torch.equal(old_grad, new_grad)


def test_mixture_probabilities_gradients_and_padding() -> None:
    z = torch.tensor([[2.0, -2.0, -torch.inf], [-5.0, 3.0, 1.0]], requires_grad=True)
    eps = torch.tensor([0.2, 0.05])
    legal = torch.isfinite(z)
    expected = (1 - eps[:, None]) * z.softmax(-1) + eps[:, None] * legal / legal.sum(-1, keepdim=True)
    got = legal_uniform_log_probs(z, eps)
    assert torch.allclose(torch.where(legal, got.exp(), 0.0), expected)
    weights = torch.tensor([[0.25, -0.7, 0.0], [-0.2, 0.3, 0.5]])
    reference = torch.where(legal, expected, 1.0).log()
    grad_ref, = torch.autograd.grad((reference * weights).sum(), z, retain_graph=True)
    grad_got, = torch.autograd.grad((got * weights).sum(), z)
    assert torch.allclose(grad_got, grad_ref, atol=1e-7)
    assert torch.isfinite(grad_got).all()
    padded = torch.full((1, 3), -torch.inf, requires_grad=True)
    lp = legal_uniform_log_probs(padded, torch.tensor([0.0]))
    lp.sum().backward()
    assert torch.equal(lp, torch.zeros_like(lp))
    assert padded.grad is not None and torch.equal(padded.grad, torch.zeros_like(padded))


def test_zero_epsilon_in_mixed_batch_preserves_tiny_probabilities() -> None:
    z = torch.tensor([[-1000.0, 0.0], [-1000.0, 0.0]], requires_grad=True)
    lp = legal_uniform_log_probs(z, torch.tensor([0.0, 0.2]))
    assert lp[0, 0] == -1000.0
    assert float(lp[1, 0].detach().exp()) == pytest.approx(0.1)
    grad, = torch.autograd.grad(lp.sum(), z)
    assert torch.isfinite(grad).all()
    assert grad[0, 0] == 1.0


def test_uniform_barrier_restores_collapsed_base_policy() -> None:
    z = torch.tensor([[0.0, np.log(1e-7), -torch.inf]], requires_grad=True)
    ce = uniform_cross_entropy(z)
    gradient, = torch.autograd.grad(ce.sum(), z)
    expected = z.detach().softmax(-1) - torch.tensor([[0.5, 0.5, 0.0]])
    assert torch.allclose(gradient, expected, atol=1e-7)
    assert float(gradient[0, 1]) < -0.499999  # pushes the collapsed logit up under gradient descent
    epsilon = torch.tensor([0.2])
    logq = legal_uniform_log_probs(z, epsilon)[0, 1]
    mixed_gradient, = torch.autograd.grad(logq, z)
    assert abs(float(mixed_gradient[0, 1])) < 1e-6  # mixture alone cannot quickly revive it


def test_exploration_config_rejects_invalid_values() -> None:
    for pairs in (((8, -0.1),), ((8, 1.0),), ((8, float("nan")),), ((8, 0.1), (8, 0.2)), ((0, 0.1),)):
        with pytest.raises(AssertionError):
            exploration_table(pairs)
    table = exploration_table(((int(Kind.REST), 0.2),))
    assert table[Kind.REST] == pytest.approx(0.2) and table[Kind.COMBAT] == 0


def test_actor_mixture_has_floor_and_exact_learner_probabilities(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    pol = TorchPolicy(SpireNet(tok.layout, TINY), tok, seed=19)
    logits = np.tile(np.array([0.0, -1000.0, -np.inf], np.float32), (20000, 1))
    eps = np.full(len(logits), 0.2, np.float32)
    a, lp, ent = pol._sample(logits, eps)
    assert set(a) == {0, 1} and abs((a == 1).mean() - 0.1) < 0.01
    ref = legal_uniform_log_probs(torch.from_numpy(logits), torch.from_numpy(eps)).numpy()
    assert np.allclose(lp, ref[np.arange(len(a)), a], atol=1e-6)
    assert np.allclose(ent, -(0.1 * np.log(0.1) + 0.9 * np.log(0.9)), atol=1e-6)


@pytest.mark.parametrize("epsilon", [0.0, 0.15])
@pytest.mark.parametrize("scores", [[0.0, -1000.0, -np.inf], [10000.0, 9999.0, -np.inf],
                                   [0.0, -np.inf, -np.inf]])
def test_single_frame_actor_learner_mixture_agreement(vocab: Vocab, epsilon: float, scores: list[float]) -> None:
    tok = Tokenizer(vocab)
    pol = TorchPolicy(SpireNet(tok.layout, TINY), tok, seed=47)
    logits = np.array([scores], np.float32)
    eps = np.array([epsilon], np.float32)
    action, logp, entropy = pol._sample(logits, eps)
    reference = legal_uniform_log_probs(torch.from_numpy(logits), torch.from_numpy(eps)).numpy()
    legal = np.isfinite(logits)
    probability = np.where(legal, np.exp(reference), 0.0)
    assert np.allclose(logp, reference[0, action], atol=1e-6)
    assert np.allclose(entropy, -(probability * reference).sum(1), atol=1e-6)


def test_actor_off_reproduces_original_samples(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    pol = TorchPolicy(SpireNet(tok.layout, TINY), tok, seed=31)
    z = np.array([[1.0, -2.0, -np.inf]] * 128, np.float32)
    rng = np.random.default_rng(31)
    g = -np.log(-np.log(rng.random(z.shape).clip(1e-12, 1.0 - 1e-7)))
    expected = (z + g).argmax(1)
    a, lp, ent = pol._sample(z)
    old_lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    assert np.array_equal(a, expected)
    assert np.allclose(lp, old_lp[np.arange(len(a)), a], atol=1e-6)
    assert np.isfinite(ent).all()


def test_mixture_rejects_training_temperature_mismatch(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    with pytest.raises(AssertionError, match="temperature=1"):
        TorchPolicy(SpireNet(tok.layout, TINY), tok, temperature=2.0,
                    exploration_by_kind=((int(Kind.REST), 0.2),))


def test_encoded_actor_uses_each_samples_decision_kind(vocab: Vocab, gen: FrameGen) -> None:
    tok = Tokenizer(vocab)
    rng = np.random.default_rng(17)
    frames = [parse_frame(random_frame(gen, rng, kind, hashes=False))
              for kind in (Kind.REST, Kind.MAP, Kind.SHOP, Kind.CARD_REWARD)]
    samples = [tok.encode(f) for f in frames]
    policy = TorchPolicy(SpireNet(tok.layout, TINY), tok, mode="greedy", use_graphs=False,
                         exploration_by_kind=((int(Kind.REST), 0.2), (int(Kind.SHOP), 0.1)))
    counts = np.array([s.n_actions for s in samples])
    handle = policy.launch(samples, counts)
    assert handle.exploration is not None
    assert np.allclose(handle.exploration, [0.2, 0.0, 0.1, 0.0])
    enc = tok.encode_batch(frames, sort=True)
    order = np.arange(len(frames)) if enc.order is None else enc.order  # documented identity-permutation sentinel
    encoded = policy.launch_encoded(enc, 0, len(frames), counts)
    assert encoded.exploration is not None
    expected = exploration_table(((int(Kind.REST), 0.2), (int(Kind.SHOP), 0.1)))[
        np.array([int(frames[int(i)].glob[G.KIND]) for i in order])]
    assert np.array_equal(encoded.exploration, expected)
    a, b = policy.finish(handle), policy.finish(encoded)
    assert np.array_equal(a.actions[order], b.actions)
    assert np.allclose(a.logp[order], b.logp, atol=1e-6)


class _StaticLogits(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        zeros = self.logits.new_zeros(len(self.logits))
        return {"logits": self.logits, "value": zeros, "pwin_logit": zeros,
                "floors_pred": zeros, "hp_pred": zeros}


@pytest.mark.parametrize("exploration", [False, True])
@pytest.mark.parametrize("decoupled", [False, True])
@pytest.mark.parametrize("uniform_penalty", [False, True])
@pytest.mark.parametrize("choice_uniform", [False, True])
def test_fast_microbody_matches_independent_cpu_loss(exploration: bool, decoupled: bool,
                                                    uniform_penalty: bool, choice_uniform: bool) -> None:
    """Exercise the exact body captured by CUDA graphs using CPU tensors and no CUDA API."""
    logits = torch.tensor([[2.0, -2.0, 1.0], [-1.0, 0.5, 2.0], [0.0, -torch.inf, -torch.inf],
                           [-torch.inf, -torch.inf, -torch.inf]])
    model = _StaticLogits(logits)
    cfg = PPOConfig(exploration_by_kind=((int(Kind.REST), 0.2),) if exploration else (),
                    uniform_kl_by_kind=((int(Kind.REST), 0.03),) if uniform_penalty else (),
                    exploration_choice_uniform=choice_uniform,
                    decoupled_clip_anchor=decoupled, ent_coef=0, vf_coef=0,
                    aux_win=0, aux_floors=0, aux_hp=0)
    stepper = MicroStepper.__new__(MicroStepper)
    stepper.model, stepper.fwd, stepper.cfg, stepper.autocast = model, model, cfg, False
    stepper._choice_verb_slot = 0
    stepper._exploration = torch.from_numpy(exploration_table(cfg.exploration_by_kind)) if exploration else None
    stepper._uniform_kl = torch.from_numpy(kind_coefficients(cfg.uniform_kl_by_kind)) if uniform_penalty else None
    stepper.acc = _Accum(torch.device("cpu"))
    stepper.den = torch.ones(len(DEN_COLS))
    stepper.den[DEN_COLS.index("pol")] = 2
    stepper.den[DEN_COLS.index("all")] = 3
    stepper.mb_kl = torch.zeros(())
    pf = torch.zeros(4, NF)
    pf[:, PF_COLS.index("logp_b")] = torch.tensor([0.4, 0.2, 1.0, 1.0]).log()
    pf[:, PF_COLS.index("logp_ref")] = torch.tensor([0.5, 0.3, 1.0, 1.0]).log()
    pf[:, PF_COLS.index("adv")] = torch.tensor([1.0, -0.5, 0.0, 0.0])
    pf[:2, PF_COLS.index("pol")] = 1
    pf[:3, PF_COLS.index("valid")] = 1
    pf[:, PF_COLS.index("log_legal")] = torch.tensor([3.0, 3.0, 2.0, 2.0]).log()
    pi = torch.tensor([[0, int(Kind.REST)], [1, int(Kind.COMBAT)], [0, int(Kind.REST)], [0, 0]])
    rel = torch.zeros(12, 3, dtype=torch.int64)
    rel[:, 0] = torch.tensor([0, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0])
    views = {"raw": torch.zeros(12, 1, dtype=torch.int16), "rel": rel, "dest": torch.arange(4, 16),
             "act_mask": torch.isfinite(logits), "tok_mask": torch.ones(4, 1, dtype=torch.bool),
             "kid_mask": torch.empty(0, dtype=torch.bool), "var_mask": torch.empty(0, dtype=torch.bool)}
    res = _Res(BatchShape(4, 1, 3, 0, 0, 0, 12), torch.empty(0), views, pf, pi)
    stepper._body(res)

    independent = logits[:2].clone().requires_grad_()
    base = independent.softmax(-1)
    epsilon = torch.tensor([0.2 if exploration else 0.0, 0.0])[:, None]
    legal = torch.isfinite(independent)
    uniform = (torch.tensor([[0.5, 0.25, 0.25], [1 / 3, 1 / 3, 1 / 3]]) if choice_uniform
               else legal / legal.sum(-1, keepdim=True))
    mixed = (1 - epsilon) * base + epsilon * uniform
    logp = mixed[torch.arange(2), torch.tensor([0, 1])].log()
    logb = pf[:2, PF_COLS.index("logp_b")]
    logr = pf[:2, PF_COLS.index("logp_ref")] if decoupled else logb
    adv = pf[:2, PF_COLS.index("adv")]
    ratio = (logp - logb).exp()
    ratio_ref = (logp - logr).exp()
    anchor_weight = (logr - logb).exp()
    loss = -torch.minimum(ratio * adv, anchor_weight * ratio_ref.clamp(0.8, 1.2) * adv).mean()
    if uniform_penalty:
        loss = loss - 0.03 * (uniform[0] * base[0].log()).sum() / 2
    gradient, = torch.autograd.grad(loss, independent)
    assert model.logits.grad is not None
    torch.testing.assert_close(model.logits.grad[:2], gradient, rtol=1e-5, atol=1e-7)
    assert torch.equal(model.logits.grad[2:], torch.zeros_like(model.logits.grad[2:]))
    kl = (anchor_weight * (ratio_ref - 1 - (logp - logr))).sum()
    torch.testing.assert_close(stepper.mb_kl, kl)
    expected_base_entropy = -(base[0] * base[0].log()).sum()
    torch.testing.assert_close(stepper.acc.ent_base_k[Kind.REST], expected_base_entropy)
