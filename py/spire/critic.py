"""The training-only critic tower that sees the draw order (notes/critic/contract.md 1.2, 2.1-2.2; build-v1.md).

* :func:`training_reset_flags` - the only way to RESET flag ``ResetFlag.HIDDEN`` (``schema.reset_flags_for`` refuses
  it); its only caller is the training actor construction of a critic run (spire.train, mode "train").
* :func:`side` / :func:`side_batch` - the side input CRIT int16[2 + K] of a frame, built from ``Frame.hdraw`` and the
  frame's ``card_base``: [GLOBAL[KIND], rows kept, local token index of the DRAW card token of the j-th HDRAW row in
  (SEAT, POS) order, -1 beyond]. The policy's ``Sample`` never sees it.
* :class:`CriticConfig` - ``model.critic`` (persisted in the checkpoint); v1 builds the 4 M tower only.
* :class:`CriticTower` - ``SpireNet.critic``: blocks over the detached final token states (after ``out_norm``), FiLM on
  the detached condition vector, CRIT as a kind embedding on the global token and position embeddings summed into the
  card tokens; outputs V_WIN (sigmoid), V_PROG, V_SURV and the WIN logit.
* :func:`optimizer_groups` - the trunk's AdamW groups exactly as ``learner._param_groups`` builds them for a model
  without the tower, then the tower's own two groups (``critic: True``, their own ``lr``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .proto import Frame
from .schema import CardCol, G, HDrawCol, ResetFlag, Table, Zone

CRITIC_INIT_SEED = 0x43524954  # "CRIT"
N_KINDS = 16  # GLOBAL[KIND] values (0..15)
N_COMPONENTS = 3  # WIN, PROG, SURV
_HIDDEN = int(ResetFlag.HIDDEN)
_KIND = int(G.KIND)
_H_SEAT, _H_POS, _H_OWNER = int(HDrawCol.SEAT), int(HDrawCol.POS), int(HDrawCol.OWNER_ROW)
_K_ZONE, _K_SEAT, _DRAW = int(CardCol.ZONE), int(CardCol.SEAT), int(Zone.DRAW)


@dataclass(frozen=True)
class CriticConfig:
    """``model.critic`` (notes/critic/contract.md 5 "Config"). build-v1.md 3: only the 4 M trunk's tower is built
    (d 224, 2 blocks, 7 heads, d_ff 896), on any trunk depth; other sizes are refused."""

    d: int = 224
    layers: int = 2
    heads: int = 7
    d_ff: int = 896
    draw_k: int = 32

    def __post_init__(self) -> None:
        assert (self.d, self.layers, self.heads, self.d_ff) == (224, 2, 7, 896), \
            "model.critic: v1 builds the 4 M tower only (d 224, layers 2, heads 7, d_ff 896; build-v1.md 3)"
        assert 1 <= self.draw_k <= 1024, "model.critic.draw_k must be in 1..1024"


ADV_SOURCES = ("public", "tower", "auto")
PUBLIC_TARGETS = ("public", "tower")


@dataclass(frozen=True)
class CriticPPOConfig:
    """``ppo.critic`` (notes/critic/contract.md 5 "Config"; build-v1.md 3). ``adv_weights`` other than empty (the
    default (1, 1, 1) at every ascension) and ``aux_fight_death`` (the public fight heads) are out of v1."""

    adv_source: str = "auto"
    warmup_updates: int = 300
    switch_window: int = 50
    switch_td_end: float = 0.010
    switch_var_ratio: float = 1.0
    blend_updates: int = 100
    adv_weights: dict[str, Any] = field(default_factory=dict)
    coef: tuple[float, float, float] = (1.0, 1.0, 1.0)
    lr: float = 3e-4
    public_target: str = "tower"

    def __post_init__(self) -> None:
        assert self.adv_source in ADV_SOURCES, f"ppo.critic.adv_source {self.adv_source!r} not in {ADV_SOURCES}"
        assert self.public_target in PUBLIC_TARGETS, f"ppo.critic.public_target {self.public_target!r}"
        assert not self.adv_weights, "ppo.critic.adv_weights: v1 keeps w = (1, 1, 1) at every ascension (build-v1.md 3)"
        assert len(self.coef) == N_COMPONENTS and all(c >= 0 for c in self.coef)
        assert self.warmup_updates >= 0 and self.switch_window >= 1 and self.blend_updates >= 1 and self.lr > 0

    @staticmethod
    def weights(ascension: int) -> tuple[float, float, float]:
        """w_k(ascension) of A_TOWER = sum_k w_k A_k (contract 4.2): (1, 1, 1) in v1."""
        return (1.0, 1.0, 1.0)


def carries_hidden(flags: int) -> bool:
    """RESET flags ``flags`` carry the hidden-trailer bit (10)."""
    return bool(int(flags) & _HIDDEN)


def training_reset_flags(flags: int) -> int:
    """``flags | ResetFlag.HIDDEN`` (notes/critic/contract.md 1.2). Only the actor construction of spire.train calls it,
    inside mode "train" with ``ppo.critic`` set (the caller asserts both)."""
    assert not flags & _HIDDEN, "training_reset_flags: the flags already carry HIDDEN"
    return int(flags) | _HIDDEN


# --------------------------------------------------------------------------- side input CRIT


def crit_width(draw_k: int) -> int:
    return 2 + draw_k


def empty_crit(n: int, draw_k: int) -> NDArray[np.int16]:
    """``n`` CRIT rows of frames without hidden rows and kind 0 (graph padding rows): [0, 0, -1 ...]."""
    out = np.full((n, crit_width(draw_k)), -1, np.int16)
    out[:, :2] = 0
    return out


def side(frame: Frame, card_base: int, n_tokens: int, draw_k: int) -> NDArray[np.int16]:
    """CRIT int16[2 + K] of one frame (notes/critic/contract.md 2.1). Rows of the HDRAW table in (SEAT, POS) order
    with POS < K, at most K of them (seats concatenated, seat 0 first); the dropped ones are
    ``len(frame.hdraw) - CRIT[1]`` (``critic_draw_truncated``). A frame without the table: [kind, 0, -1 ...]."""
    out = np.full(crit_width(draw_k), -1, np.int16)
    out[0] = frame.gvals[_KIND]
    assert 0 <= out[0] < N_KINDS, f"GLOBAL[KIND] {int(out[0])} outside 0..15"
    hd = frame.hdraw
    if hd is None or not hd.shape[0]:
        out[1] = 0
        return out
    keep = hd[hd[:, _H_POS] < draw_k][:draw_k]
    owner = keep[:, _H_OWNER].astype(np.int64)
    tok = card_base + owner
    assert int(tok.max()) < n_tokens and int(owner.min()) >= 0, "CRIT: a DRAW card token outside the sample"
    cards = frame.table(Table.CARDS)
    assert (cards[owner, _K_ZONE] == _DRAW).all() and (cards[owner, _K_SEAT] == keep[:, _H_SEAT]).all(), \
        "CRIT: an HDRAW row does not name a DRAW card of its seat"
    out[1] = keep.shape[0]
    out[2 : 2 + keep.shape[0]] = tok
    return out


def side_batch(frames: Sequence[Frame], card_base: NDArray[np.integer], n_tokens: NDArray[np.integer],
               draw_k: int) -> NDArray[np.int16]:
    """(B, 2 + K) CRIT of ``frames`` with their ``EncodedBatch.card_base`` / ``n_tokens`` in the same order (an
    EncodedBatch built with ``sort=True`` lists them in ``enc.order``: pass ``[frames[i] for i in enc.order]``)."""
    assert len(frames) == len(card_base) == len(n_tokens)
    return np.stack([side(f, int(b), int(n), draw_k) for f, b, n in zip(frames, card_base, n_tokens)])


def truncated(frame: Frame, crit: NDArray[np.int16]) -> int:
    """HDRAW rows of ``frame`` that CRIT ``crit`` dropped (POS >= K, or beyond K rows in total)."""
    hd = frame.hdraw
    return 0 if hd is None else int(hd.shape[0]) - int(crit[1])


# --------------------------------------------------------------------------- the tower


class CriticTower(nn.Module):
    """notes/critic/contract.md 2.2. Reads only detached trunk tensors: the hidden loss cannot shape the trunk."""

    def __init__(self, cfg: CriticConfig, d_model: int, d_cond: int, init_std: float) -> None:
        super().__init__()
        from .model import Block, _norm_mlp  # model.py imports this module at its top

        self.cfg = cfg
        d = cfg.d
        self.crit_in = nn.Linear(d_model, d)
        self.kind_emb = nn.Embedding(N_KINDS, d)
        self.pos_emb = nn.Embedding(cfg.draw_k, d)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.d_ff, d_cond) for _ in range(cfg.layers)])
        self.out_norm = nn.LayerNorm(d)
        self.head = _norm_mlp(d, N_COMPONENTS)
        self.kind_bias = nn.Embedding(N_KINDS, N_COMPONENTS)
        self.init_weights(init_std)

    @torch.no_grad()
    def init_weights(self, std: float) -> None:
        """From ``torch.Generator().manual_seed(CRITIC_INIT_SEED)`` in module order: linears and embeddings
        normal(0, std), the blocks' residual output layers std / sqrt(2 * layers), biases 0, LayerNorms (1, 0),
        ``kind_bias`` 0 and the head's last layer 0: the tower starts at (0.5, 0, 0) on every frame."""
        g = torch.Generator().manual_seed(CRITIC_INIT_SEED)
        out_std = std / math.sqrt(2 * self.cfg.layers)
        residual = {id(b.proj.weight) for b in self.blocks} | {id(b.ff2.weight) for b in self.blocks}  # type: ignore
        last = self.head[1][2]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                s = out_std if id(m.weight) in residual else std
                m.weight.copy_(torch.empty(m.weight.shape).normal_(0.0, s, generator=g))
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.copy_(torch.empty(m.weight.shape).normal_(0.0, std, generator=g))
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                m.weight.fill_(1.0)
                m.bias.zero_()
        self.kind_bias.weight.zero_()
        last.weight.zero_()
        last.bias.zero_()

    def forward(self, h: Tensor, cond: Tensor, tok_mask: Tensor, crit: Tensor) -> tuple[Tensor, Tensor]:
        """``h`` (B, T, d_model) final token states, ``cond`` (B, d_cond), ``tok_mask`` (B, T), ``crit`` (B, 2 + K)
        -> ``vc`` (B, 3) float32 = (V_WIN, V_PROG, V_SURV) and the WIN logit (B,) float32."""
        B, T, _ = h.shape
        assert crit.shape == (B, crit_width(self.cfg.draw_k)), (tuple(crit.shape), B)
        x = self.crit_in(h.detach())
        c = cond.detach()
        kind = crit[:, 0].long()
        x = torch.cat([(x[:, 0] + self.kind_emb(kind).to(x.dtype))[:, None], x[:, 1:]], 1)
        # position embedding j into the card token CRIT[2 + j] (-1 matches no token): a one-hot product, so a
        # multiset token gets the sum of its copies' positions, deterministically and with static shapes
        idx = crit[:, 2:].long()
        hit = (idx[:, :, None] == torch.arange(T, device=idx.device)).to(x.dtype)  # (B, K, T)
        x = x + torch.einsum("bkt,kd->btd", hit, self.pos_emb.weight.to(x.dtype))
        mask = tok_mask[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, mask, c)
        g = self.out_norm(x)[:, 0]
        o = self.head(g).float() + self.kind_bias(kind).float()
        return torch.stack([torch.sigmoid(o[:, 0]), o[:, 1], o[:, 2]], 1), o[:, 0]


# --------------------------------------------------------------------------- optimizer groups


def _groups(params: list[nn.Parameter], no_decay: set[int], weight_decay: float) -> list[dict[str, Any]]:
    decay = [p for p in params if p.requires_grad and p.ndim >= 2 and id(p) not in no_decay]
    other = [p for p in params if p.requires_grad and (p.ndim < 2 or id(p) in no_decay)]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": other, "weight_decay": 0.0}]
    return [g for g in groups if g["params"]]


def optimizer_groups(net: nn.Module, weight_decay: float, critic_lr: float) -> list[dict[str, Any]]:
    """AdamW groups of a critic network (notes/critic/contract.md 4.3): the trunk's groups are exactly
    ``learner._param_groups`` of the network without the tower (same parameters, same order, same settings), then
    the tower's decay / no-decay groups by the same rule, tagged ``critic: True`` with ``lr = critic_lr``."""
    tower = getattr(net, "critic")
    assert isinstance(tower, CriticTower), "optimizer_groups needs a network with the critic tower"
    no_decay: set[int] = set()
    for mod in net.modules():
        if isinstance(mod, (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            no_decay.update(id(p) for p in mod.parameters(recurse=False))
    mine = {id(p) for p in tower.parameters()}
    trunk = [p for p in net.parameters() if id(p) not in mine]
    crit = [{**g, "lr": critic_lr, "critic": True} for g in _groups(list(tower.parameters()), no_decay, weight_decay)]
    return _groups(trunk, no_decay, weight_decay) + crit


def tower_parameters(net: nn.Module) -> set[int]:
    """ids of the tower's parameters."""
    return {id(p) for p in getattr(net, "critic").parameters()}
