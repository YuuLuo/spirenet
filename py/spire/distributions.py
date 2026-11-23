"""Legal-row or choice-uniform behaviour mixtures, shared by eager and captured PPO."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import TYPE_CHECKING, Sequence

from .schema import DECISION_KINDS, Kind, Verb

if TYPE_CHECKING:
    from .tokenize import Sample

_N_VERBS = max(int(v) for v in Verb) + 1


def sample_choice_keys(samples: Sequence["Sample"], verb_slot: int, actions: int) -> np.ndarray:
    """Choice = (source entity, verb, acting seat), independent of target and aux.

    Sources have unique entity rows, including the distinct REST option rows. A missing
    source still has a verb and seat. Padding is -1; only finite-logit rows participate.
    """
    keys = np.full((len(samples), actions), -1, np.int64)
    for i, sample in enumerate(samples):
        n = sample.n_actions
        assert n <= actions
        start = sum(sample.counts[:3])
        rel = sample.rel[start:].astype(np.int64)
        verb = sample.raw[start:, verb_slot].astype(np.int64)
        keys[i, :n] = ((rel[:, 0] + 1) * _N_VERBS + verb) * (len(sample.raw) + 1) + rel[:, 2] + 1
    return keys


def batch_choice_keys(batch: dict[str, Tensor], verb_slot: int) -> Tensor:
    """The same choice equivalence on a collated batch, with static shapes for CUDA graphs."""
    b, a = batch["act_mask"].shape
    start = batch["tok_mask"].numel() + batch["kid_mask"].numel() + batch["var_mask"].numel()
    n = start + b * a
    rel = batch["rel"]
    keys = (rel[:, 0] * _N_VERBS + batch["raw"][:, verb_slot].long()) * (n + 1) + rel[:, 2]
    return keys.new_full((n + 2,), -1).index_copy(0, batch["dest"], keys)[start:n].view(b, a)


def choice_uniform_probs(legal: Tensor, keys: Tensor) -> Tensor:
    """Uniform over choices, then over each choice's currently legal target rows.

    Sorting plus fixed-size scatter avoids an action-squared comparison tensor and works
    inside the captured learner body. Empty padding rows have probability zero.
    """
    ordered, order = torch.where(legal, keys, -1).sort(-1)
    valid = legal.gather(-1, order)
    first = torch.cat((torch.ones_like(valid[:, :1]), ordered[:, 1:] != ordered[:, :-1]), -1) & valid
    group = first.long().cumsum(-1)
    counts = torch.zeros((*legal.shape[:-1], legal.shape[-1] + 1), device=legal.device,
                         dtype=torch.float32).scatter_add(
        -1, group, valid.float())
    probability = valid.float() / counts.gather(-1, group).clamp_min(1) / first.sum(-1, keepdim=True).clamp_min(1)
    return torch.zeros_like(probability).scatter(-1, order, probability)


def choice_uniform_probs_numpy(legal: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Host sampler counterpart of :func:`choice_uniform_probs`."""
    assert legal.shape == keys.shape
    probability = np.zeros(legal.shape, np.float64)
    for i in range(len(legal)):
        assert legal[i].any(), "actor frame has no legal action"
        _, inverse, counts = np.unique(keys[i, legal[i]], return_inverse=True, return_counts=True)
        probability[i, legal[i]] = 1.0 / (len(counts) * counts[inverse])
    return probability


def kind_coefficients(pairs: tuple[tuple[int, float], ...]) -> np.ndarray:
    """Nonnegative per-kind coefficients, with zero for unconfigured decision kinds."""
    assert len({k for k, _ in pairs}) == len(pairs), "duplicate decision kind"
    table = np.zeros(max(int(k) for k in Kind) + 1, np.float32)
    for kind, coefficient in pairs:
        assert kind in DECISION_KINDS, "coefficient requires a decision kind"
        assert np.isfinite(coefficient) and coefficient >= 0, "coefficient must be finite and nonnegative"
        table[kind] = coefficient
    return table


def exploration_table(pairs: tuple[tuple[int, float], ...]) -> np.ndarray:
    """Per-kind legal-uniform mixture weights."""
    table = kind_coefficients(pairs)
    assert (table < 1).all(), "exploration epsilon must be in [0, 1)"
    return table


def legal_uniform_log_probs(logits: Tensor, epsilon: Tensor | None = None,
                            reference: Tensor | None = None) -> Tensor:
    """Log q for q=(1-epsilon) softmax(logits)+epsilon*reference; illegal entries are zero.

    The default reference is uniform over legal rows; callers may provide choice-uniform.
    Zero illegal entries keep entropy products finite. Empty rows are CUDA micro-batch
    padding and have no loss weight. Log-space mixing retains gradients for very small
    policy probabilities and for rows whose configured epsilon is zero.
    """
    legal = torch.isfinite(logits)
    logp = torch.log_softmax(logits.masked_fill(~legal, -1e30), -1)
    if epsilon is not None:
        eps = epsilon[:, None]
        log_uniform = (eps.log() - legal.sum(-1, keepdim=True).clamp_min(1).log() if reference is None
                       else eps.log() + reference.masked_fill(~legal, 1.0).log())
        logp = torch.logaddexp(logp + torch.log1p(-eps), log_uniform)
    return torch.where(legal, logp, 0.0)


def uniform_cross_entropy(logits: Tensor, reference: Tensor | None = None) -> Tensor:
    """CE(reference, base policy), with a legal-row-uniform reference by default.

    The gradient on a legal logit is pi(a)-reference(a), which continues to restore a
    collapsed action even when its probability is arbitrarily small. Empty padded
    rows contribute zero, matching their zero training weight.
    """
    logp = legal_uniform_log_probs(logits)
    if reference is not None:
        return -(reference * logp).sum(-1)
    legal = torch.isfinite(logits)
    return -logp.sum(-1) / legal.sum(-1).clamp_min(1)
