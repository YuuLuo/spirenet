"""Choice-first exploration uses one reference in the actor and every PPO term."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from spire.actor import TorchPolicy
from spire.distributions import (batch_choice_keys, choice_uniform_probs, choice_uniform_probs_numpy,
                                legal_uniform_log_probs, sample_choice_keys, uniform_cross_entropy)
from spire.learner import Learner, PPOConfig, _Accum, _Tensors
from spire.model import ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.rollout import EpisodeMeta, Segment, SegmentEnd
from spire.schema import ActionCol, Kind, Verb
from spire.synth import FrameGen, random_frame
from spire.tokenize import Ent, Sample, Tokenizer
from spire.vocab import Vocab

TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32)


def _choices(tok: Tokenizer) -> Sample:
    """HEAL, SMITH, three MEND targets, then same source with a different verb/seat."""
    raw = np.zeros((13, tok.width), np.int16)
    rel = np.full((13, 3), -1, np.int16)
    raw[6:, tok.layout.slot(Ent.ACTION, ActionCol.VERB)] = int(Verb.CHOOSE_OPTION)
    raw[11, tok.layout.slot(Ent.ACTION, ActionCol.VERB)] = int(Verb.USE_POTION)
    rel[6:, 0] = [3, 4, 5, 5, 5, 5, 5]
    rel[6:, 2] = [1, 1, 1, 1, 1, 1, 2]
    rel[8:11, 1] = [1, 2, 3]  # target is deliberately irrelevant to choice identity
    types = np.full(13, int(Ent.ACTION), np.int8)
    types[:6] = int(Ent.SEAT)
    return Sample(types, raw, rel, np.empty((0, 3), np.int16), np.zeros(6, np.int16), (6, 0, 0, 7))


def test_choice_keys_match_collation_and_keep_verb_and_seat_distinct(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    sample = _choices(tok)
    batch = tok.collate([sample, sample])
    slot = tok.layout.slot(Ent.ACTION, ActionCol.VERB)
    keys = batch_choice_keys(batch, slot)
    legal = batch["act_mask"].clone()
    expected = np.array([1 / 5, 1 / 5, 1 / 15, 1 / 15, 1 / 15, 1 / 5, 1 / 5], np.float32)
    reference = choice_uniform_probs(legal, keys)
    torch.testing.assert_close(reference[:, :7], torch.from_numpy(np.stack([expected, expected])))
    host_keys = sample_choice_keys([sample, sample], slot, legal.shape[1])
    host = choice_uniform_probs_numpy(legal.numpy(), host_keys)
    np.testing.assert_allclose(reference.numpy(), host, atol=1e-7)
    # Cycle bans / other model masks remove targets before counting choices.
    legal[1, 3:5] = False
    reference = choice_uniform_probs(legal, keys)
    np.testing.assert_allclose(reference[1, :7].numpy(), [0.2, 0.2, 0.2, 0, 0, 0.2, 0.2])
    legal[1] = False
    assert torch.equal(choice_uniform_probs(legal, keys)[1], torch.zeros(legal.shape[1]))


def test_choice_reference_is_permutation_equivariant() -> None:
    keys = torch.tensor([[40, 10, 10, 10, 30, -1]])
    legal = torch.tensor([[True, True, True, True, True, False]])
    permutation = torch.tensor([3, 5, 0, 4, 2, 1])
    before = choice_uniform_probs(legal, keys)
    after = choice_uniform_probs(legal[:, permutation], keys[:, permutation])
    assert torch.equal(after, before[:, permutation])
    torch.testing.assert_close(before, torch.tensor([[1 / 3, 1 / 9, 1 / 9, 1 / 9, 1 / 3, 0]]))


def test_choice_actor_sampling_logprob_entropy_and_barrier(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    policy = TorchPolicy(SpireNet(tok.layout, TINY), tok, seed=17)
    logits = np.tile(np.array([0, -1000, -1000, -1000, -1000, -np.inf], np.float32), (20000, 1))
    keys = np.tile(np.array([0, 1, 2, 2, 2, -1]), (len(logits), 1))
    epsilon = np.full(len(logits), 0.15, np.float32)
    action, stored_logp, entropy = policy._sample(logits, epsilon, keys)
    expected = np.array([0.9, 0.05, 1 / 60, 1 / 60, 1 / 60, 0])
    frequencies = np.bincount(action, minlength=6) / len(action)
    np.testing.assert_allclose(frequencies, expected, atol=0.006)
    z = torch.from_numpy(logits[:1]).requires_grad_()
    reference = choice_uniform_probs(torch.isfinite(z), torch.from_numpy(keys[:1]))
    learner_logp = legal_uniform_log_probs(z, torch.tensor([0.15]), reference)
    np.testing.assert_allclose(stored_logp, learner_logp.detach().numpy()[0, action], atol=1e-6)
    np.testing.assert_allclose(entropy, -(expected[:5] * np.log(expected[:5])).sum(), atol=1e-6)
    gradient, = torch.autograd.grad(uniform_cross_entropy(z, reference).sum(), z)
    torch.testing.assert_close(gradient, z.detach().softmax(-1) - reference)


@pytest.mark.parametrize("choice_uniform", [False, True])
def test_actor_encoded_and_eager_learner_anchor_agree(vocab: Vocab, gen: FrameGen,
                                                    choice_uniform: bool) -> None:
    tok = Tokenizer(vocab)
    rng = np.random.default_rng(21)
    kinds = [Kind.REST, Kind.COMBAT, Kind.SHOP, Kind.MAP]
    frames = [parse_frame(random_frame(gen, rng, kind, n_seats=4, hashes=False)) for kind in kinds]
    samples = [tok.encode(frame) for frame in frames]
    net = SpireNet(tok.layout, TINY)
    pairs = tuple((int(kind), 0.15) for kind in kinds)
    cfg = PPOConfig(exploration_by_kind=pairs, exploration_choice_uniform=choice_uniform,
                    decoupled_clip_anchor=True)
    policy = TorchPolicy(net, tok, use_graphs=False, mode="greedy", exploration_by_kind=pairs,
                         exploration_choice_uniform=choice_uniform)
    counts = np.array([sample.n_actions for sample in samples])
    launched = policy.launch(samples, counts)
    result = policy.finish(launched)
    encoded = tok.encode_batch(frames, sort=False)
    encoded_result = policy.finish(policy.launch_encoded(encoded, 0, len(frames), counts))
    assert np.array_equal(result.actions, encoded_result.actions)
    assert np.array_equal(result.logp, encoded_result.logp)
    learner = Learner(copy.deepcopy(net), tok, cfg, device="cpu")
    segment = Segment(EpisodeMeta(0, 1, (2, 2, 2, 2), 0), 0, samples, result.actions, result.logp,
                      result.value, np.array(kinds, np.int8), counts.astype(np.int16), np.zeros(4, np.int64),
                      end=SegmentEnd.TERMINAL)
    _, _, anchor = learner._recompute([segment])
    assert anchor is not None
    np.testing.assert_allclose(anchor, result.logp, atol=1e-6)
    output = learner._forward(learner.model, samples)
    reference = (choice_uniform_probs(torch.isfinite(output["logits"]), output["choice_keys"])
                 if choice_uniform else None)
    learner_logp = legal_uniform_log_probs(output["logits"], torch.full((4,), 0.15), reference)
    np.testing.assert_allclose(learner_logp.detach().numpy()[np.arange(4), result.actions], result.logp, atol=1e-6)


def test_default_off_is_bit_identical_even_with_mixture(vocab: Vocab, gen: FrameGen) -> None:
    tok = Tokenizer(vocab)
    frame = parse_frame(random_frame(gen, np.random.default_rng(92), Kind.COMBAT, n_seats=4, hashes=False))
    sample = tok.encode(frame)
    net = SpireNet(tok.layout, TINY)
    pairs = ((int(Kind.COMBAT), 0.15),)
    default = TorchPolicy(net, tok, seed=42, use_graphs=False, exploration_by_kind=pairs)
    explicit = TorchPolicy(net, tok, seed=42, use_graphs=False, exploration_by_kind=pairs,
                           exploration_choice_uniform=False)
    a = default.finish(default.launch([sample], np.array([sample.n_actions])))
    b = explicit.finish(explicit.launch([sample], np.array([sample.n_actions])))
    assert np.array_equal(a.actions, b.actions)
    assert np.array_equal(a.logp, b.logp)
    assert a.entropy is not None and b.entropy is not None
    assert np.array_equal(a.entropy, b.entropy)
    z = np.array([[3.0, -2.0, -2.0, -2.0, -np.inf]], np.float32)
    # Freeze the pre-change actor mixture arithmetic, including its NumPy precision.
    legal = np.isfinite(z)
    zm = np.where(legal, z, -np.inf)
    mx = zm.max(1, keepdims=True)
    e = np.where(legal, np.exp(zm - mx), 0.0)
    logp = np.where(legal, zm - mx - np.log(e.sum(1, keepdims=True)), 0.0)
    epsilon = np.array([0.15], np.float32)
    eps = epsilon[:, None]
    old = np.where(legal, np.logaddexp(logp + np.log1p(-eps),
                                      np.log(eps) - np.log(legal.sum(1, keepdims=True))), 0.0)
    action, logp_new, entropy = default._sample(z, epsilon)
    assert np.array_equal(logp_new, old[np.arange(len(action)), action].astype(np.float32))
    assert np.array_equal(entropy, -(np.where(legal, np.exp(old), 0) * old).sum(1).astype(np.float32))


def test_default_reference_preserves_original_learner_bits_and_gradients() -> None:
    logits = torch.tensor([[3.0, -2.0, -2.0, -2.0, -torch.inf], [0.0, -1000.0, 2.0, 1.0, 0.5]],
                          requires_grad=True)
    legal = torch.isfinite(logits)
    epsilon = torch.tensor([0.15, 0.0])
    eps = epsilon[:, None]
    base = torch.log_softmax(logits.masked_fill(~legal, -1e30), -1)
    original = torch.where(legal, torch.logaddexp(base + torch.log1p(-eps),
                                                 eps.log() - legal.sum(-1, keepdim=True).clamp_min(1).log()), 0.0)
    current = legal_uniform_log_probs(logits, epsilon)
    assert torch.equal(original, current)
    old_grad, = torch.autograd.grad(original.sum(), logits, retain_graph=True)
    new_grad, = torch.autograd.grad(current.sum(), logits, retain_graph=True)
    assert torch.equal(old_grad, new_grad)
    original_ce = -torch.where(legal, base, 0.0).sum(-1) / legal.sum(-1).clamp_min(1)
    current_ce = uniform_cross_entropy(logits)
    assert torch.equal(original_ce, current_ce)
    old_grad, = torch.autograd.grad(original_ce.sum(), logits, retain_graph=True)
    new_grad, = torch.autograd.grad(current_ce.sum(), logits)
    assert torch.equal(old_grad, new_grad)


class _Fixed(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[2.0, -1.0, -3.0, 0.5, -0.5, 0.0, -1.0]]))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = torch.nn.functional.pad(self.logits, (0, batch["act_mask"].shape[1] - 7), value=-torch.inf)
        zero = logits.new_zeros(1)
        return {"logits": logits, "value": zero, "pwin_logit": zero, "floors_pred": zero, "hp_pred": zero}


@pytest.mark.parametrize("choice_uniform", [False, True])
def test_eager_surrogate_and_barrier_gradients_share_reference(vocab: Vocab, choice_uniform: bool) -> None:
    tok = Tokenizer(vocab)
    model = _Fixed()
    cfg = PPOConfig(exploration_by_kind=((int(Kind.REST), 0.3),),
                    exploration_choice_uniform=choice_uniform, uniform_kl_by_kind=((int(Kind.REST), 0.07),),
                    lr=0, grad_clip=1e6, ent_coef=0, vf_coef=0, aux_win=0, aux_floors=0, aux_hp=0)
    learner = Learner(model, tok, cfg, device="cpu")
    independent = model.logits.detach().clone().requires_grad_()
    reference = (torch.tensor([[0.2, 0.2, 1 / 15, 1 / 15, 1 / 15, 0.2, 0.2]]) if choice_uniform
                 else torch.full((1, 7), 1 / 7))
    base = independent.softmax(-1)
    mixed = 0.7 * base + 0.3 * reference
    logb = mixed[:, 2].detach().log()
    loss = -mixed[:, 2] / logb.exp() - 0.07 * (reference * independent.log_softmax(-1)).sum(-1)
    expected, = torch.autograd.grad(loss.sum(), independent)
    zero = torch.zeros(1)
    false = torch.zeros(1, dtype=torch.bool)
    tensors = _Tensors(actions=torch.tensor([2]), logp_b=logb, logp_ref=logb, adv=torch.ones(1),
                       ret=zero, v_old=zero, pol_mask=~false, log_legal=torch.tensor([7.0]).log(),
                       ent_coef=zero, kinds=torch.tensor([int(Kind.REST)]), win=zero, win_mask=false,
                       floors=zero, floors_mask=false, hp=zero, hp_mask=false)
    kl = learner._minibatch([_choices(tok)], np.array([0]), None, tensors, _Accum(torch.device("cpu")), 0)
    assert abs(kl) < 1e-6
    assert model.logits.grad is not None
    torch.testing.assert_close(model.logits.grad, expected, atol=1e-7, rtol=1e-5)
