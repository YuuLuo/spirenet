"""SpireNet: entity transformer with pointer-style action scoring (py-stack-spec, schema 13).

Pipeline for one :data:`~spire.tokenize.Batch`:

1. **Entity embedding** (every entity row - token, kid, var, action): type embedding + one linear
   layer over ``[sum of id embeddings (d_id) | per-type normalised numeric block]``. Id embeddings
   come from one table: learned rows per vocab category / protocol table (sized from the vocab),
   plus optional *static prior* rows computed from ``catalog.json`` features by small MLPs
   (cards x upgrade level, relics, potions, powers, monsters); optional training-time dropout of
   the learned id rows of those categories (``ModelConfig.id_dropout``).
2. **Relations**: a token adds a projection of its related token (SEATS->creature, CARDS->link_row,
   OPTIONS->ref_row; child tokens additionally their owner); kids add their related token.
3. **Pooling** (DeepSets): VARS rows -> owner token / power kid / matching action;
   kids (POWERS, INTENTS, REFS, FORECAST) and both directions of every map edge -> owner token. FORECAST kids
   (``tokenizer.forecast``) get their own extra projection ``forecast_in`` and their pooled contribution is
   multiplied by the scalar ``forecast_gate`` (0 at migration: notes/forecast/contract.md 3.3). Calc kids TURN /
   FOE / ROUTE (``tokenizer.calc``) likewise get ``calc_in`` and the three gates ``calc_gate`` (0 at migration:
   notes/calc/contract.md 3.3). The AHEAD kid of the GLOBAL token (``tokenizer.route_summary``) gets ``ahead_in`` and
   the scalar ``ahead_gate`` (0 at birth and at migration: notes/inputs2/contract.md 2.4).
4. **Transformer** over tokens: pre-norm blocks whose LayerNorms are modulated (FiLM / AdaLN)
   by a conditioning vector from (ascension, seat count, characters) - injected into every layer.
   No positional encoding: order information only enters through the schema's own position
   columns (hand position, orb queue index, enemy slot ...), so the model is permutation
   equivariant over rows.
5. **Policy**: action vector = own embedding (+ pooled per-target VARS) + projections of the final
   source-, target- and actor-seat-token states and of the global token; scored by an MLP (additive
   pointer attention). Padded actions get ``-inf``.
6. **Heads** on the global token: value, P(win) logit, remaining floors, post-combat HP ratio.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .tokenize import (CALC_ENTS, CYCLE_BAN_SLOTS, HISTORY_ROWS, HISTORY_STATS, N_ENT_CALC, N_ENT_FORECAST,
                       N_ENT_LEGACY, SPECS,
                       SYMLOG_CLIP, Batch, BatchShape, EncodedBatch, Ent, FeatureDecoder, Mode, Sample, TokenLayout,
                       Tokenizer, history_width)
from .critic import CriticConfig, CriticTower, crit_width
from .rules import removal_terms
from .vocab import Catalog

OUTPUT_KEYS = ("logits", "value", "pwin_logit", "floors_pred", "hp_pred")
CALC_INIT_SEED = 0x43414C43  # "CALC": calc_in and the new type_emb rows (notes/calc/contract.md 3.3)
AHEAD_INIT_SEED = 0x41484541  # "AHEA": type_emb row 24, then ahead_in (notes/inputs2/contract.md 2.4)


def calc_type_rows(forecast: bool) -> int:
    """type_emb rows a calc layout keeps from the model before calc (rows after them are new)."""
    return N_ENT_FORECAST if forecast else N_ENT_LEGACY


def ahead_init(d: int, in_shape: torch.Size, init_std: float) -> tuple[Tensor, Tensor]:
    """(type_emb row 24, ahead_in.weight) drawn normal(0, init_std) from the generator AHEAD_INIT_SEED, row first
    (notes/inputs2/contract.md 2.4; SpireNet and the migration both draw them here)."""
    g = torch.Generator().manual_seed(AHEAD_INIT_SEED)
    row = torch.empty(d).normal_(0.0, init_std, generator=g)
    return row, torch.empty(in_shape).normal_(0.0, init_std, generator=g)
CAPTURE_LOCK = threading.RLock()
"""Serialises CUDA-graph captures of one process (inference graphs of an actor thread, training graphs of a
learner thread): ``torch.cuda.graph`` synchronises the device and empties the allocator caches before it
starts capturing, which would invalidate a capture running in another thread."""


def capture_graph(graph: torch.cuda.CUDAGraph, fn: Any, *, pool: Any, stream: torch.cuda.Stream) -> Any:
    """Capture ``fn()`` into ``graph`` (on ``stream``, memory from ``pool``) like ``torch.cuda.graph`` but
    without its device synchronisation and allocator cache flush (``empty_cache``): under WDDM contention
    the re-allocations after every flush made each training-graph capture cost ~0.5 s (py-stage2-2 profile).
    Thread-local capture mode, serialised by :data:`CAPTURE_LOCK`. Returns ``fn()``'s result."""
    with CAPTURE_LOCK:
        cur = torch.cuda.current_stream(stream.device)
        stream.wait_stream(cur)
        with torch.cuda.stream(stream):
            graph.capture_begin(pool, capture_error_mode="thread_local")
            try:
                out = fn()
            finally:
                graph.capture_end()
        cur.wait_stream(stream)
    return out


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    d_ff: int = 768
    d_id: int = 64  # width of id embeddings (summed per entity, then projected to d_model)
    d_cond: int = 64
    d_static: int = 128  # hidden width of the static-prior MLPs
    init_std: float = 0.02
    id_dropout: float = 0.0
    """Training only, with a catalog: probability that an entity's learned id rows of the
    statically described categories (card, relic, potion, power, monster) are dropped, so the
    static prior has to carry rare / newly added ids (schema 13)."""
    seat_symmetry: bool = False
    """Explicit H5 migration architecture; preserves the legacy one-seat intent-missing constants."""
    critic: CriticConfig | None = None
    """The training-only critic tower (notes/critic/contract.md 2.2), ``SpireNet.critic``; never in ``forward``.
    Absent (None) = the network of tree p4: no key in :func:`model_config_dict`, no parameter, the same RNG path."""
    value_bins: int = 0
    """p12: HL-Gauss distributional value head bin count (0 = scalar value_head, omitted from model_config_dict)."""
    value_vmin: float = -0.5
    value_vmax: float = 2.5
    value_sigma: float = 0.05
    char_experts: int = 0
    """p12: character-routed FFN experts per transformer block (0 = off, omitted from model_config_dict)."""
    char_expert_ff: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.critic, Mapping):  # ModelConfig(**asdict(...)) round trip (checkpoints, kickstart)
            object.__setattr__(self, "critic", CriticConfig(**self.critic))
        assert (isinstance(self.value_bins, int) and not isinstance(self.value_bins, bool)
                and (self.value_bins == 0 or self.value_bins >= 2)), f"value_bins {self.value_bins}: 0 or >= 2"
        assert (math.isfinite(self.value_vmin) and math.isfinite(self.value_vmax)
                and self.value_vmin < self.value_vmax), (self.value_vmin, self.value_vmax)
        assert math.isfinite(self.value_sigma) and self.value_sigma > 0.0, self.value_sigma
        assert (isinstance(self.char_experts, int) and not isinstance(self.char_experts, bool)
                and self.char_experts >= 0), self.char_experts
        assert (isinstance(self.char_expert_ff, int) and not isinstance(self.char_expert_ff, bool)
                and self.char_expert_ff >= 0), self.char_expert_ff
        assert (self.char_experts > 0) == (self.char_expert_ff > 0), \
            "char_experts and char_expert_ff must both be > 0 or both 0"


def model_config_dict(config: ModelConfig) -> dict[str, Any]:
    """``asdict(config)`` omitting disabled optional feature keys (bit-identical to tree p4/p11 when off)."""
    d = asdict(config)
    if config.critic is None:
        del d["critic"]
    if config.value_bins == 0:
        for k in ("value_bins", "value_vmin", "value_vmax", "value_sigma"):
            del d[k]
    if config.char_experts == 0:
        for k in ("char_experts", "char_expert_ff"):
            del d[k]
    return d


def hl_gauss_probs(target: Tensor, edges: Tensor, vmin: float, vmax: float, sigma: float) -> Tensor:
    """Exact truncated HL-Gauss histogram target probabilities (B, K) over ``edges`` (K + 1,)."""
    y = target.float().clamp(vmin, vmax).unsqueeze(-1)
    cdf = torch.erf((edges.unsqueeze(0) - y) * (1.0 / (math.sqrt(2.0) * sigma)))
    probs = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)
    return probs / (cdf[:, -1:] - cdf[:, :1]).clamp_min(1e-12)


PRESETS: Mapping[str, ModelConfig] = {
    # total parameters with the v0.113-sized synthetic vocab + catalog: 2.51M / 4.12M / 6.61M / ~19M
    "2.5m": ModelConfig(d_model=192, n_layers=4, n_heads=6, d_ff=512, d_id=32),
    "4m": ModelConfig(d_model=224, n_layers=4, n_heads=7, d_ff=896, d_id=64),
    "6.5m": ModelConfig(d_model=256, n_layers=6, n_heads=8, d_ff=960, d_id=64),
    "19m": ModelConfig(d_model=384, n_layers=8, n_heads=8, d_ff=1536, d_id=96),
    "19m-moe": ModelConfig(d_model=384, n_layers=8, n_heads=8, d_ff=1280, d_id=96,
                           char_experts=8, char_expert_ff=512),
}


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(approximate="tanh"), nn.Linear(d_hidden, d_out))


def _norm_mlp(d: int, d_out: int) -> nn.Sequential:
    """LayerNorm (no affine: the next linear layer absorbs it) followed by a 2-layer MLP."""
    return nn.Sequential(nn.LayerNorm(d, elementwise_affine=False), _mlp(d, d, d_out))


def _take(src: Tensor, idx: Tensor, valid: Tensor) -> Tensor:
    """``src[idx]`` where ``valid``, zero elsewhere.

    Invalid lookups read distinct rows (then masked) instead of one shared "none" row: the
    accumulating backward of a gather serialises on duplicate indices, and "none" is by far the
    most common relation (measured: 73% of a training step's GPU time before this).
    """
    alt = torch.arange(idx.shape[0], device=idx.device) % src.shape[0]
    return src[torch.where(valid, idx, alt)] * valid.unsqueeze(-1).to(src.dtype)


class _BagSum(torch.autograd.Function):
    """``F.embedding_bag(ids, table, mode="sum", padding_idx=0)`` with an atomic (``index_add_``) table gradient.

    The library's dense backward sorts all ids and sums every row's occurrences in one sequential loop per row;
    the "none" rows of the id regions occur ~24k times in one training micro-batch (measured: 10 ms per micro-step,
    15% of the learner's GPU time). Here each id column adds its rows' gradients atomically; padding ids go to
    distinct scratch rows (no contention on row 0) that are dropped. Same gradient up to fp32 summation order."""

    @staticmethod
    def forward(ctx: Any, ids: Tensor, table: Tensor) -> Tensor:
        ctx.save_for_backward(ids)
        ctx.n_rows = table.shape[0]
        return F.embedding_bag(ids, table, mode="sum", padding_idx=0)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[None, Tensor]:
        (ids,) = ctx.saved_tensors
        n, scratch = ctx.n_rows, 1024
        out = grad.new_zeros(n + scratch, grad.shape[1])
        spread = n + torch.arange(ids.shape[0], device=ids.device) % scratch
        for j in range(ids.shape[1]):
            col = ids[:, j]
            out.index_add_(0, torch.where(col > 0, col, spread), grad)
        out[0].zero_()
        return None, out[:n]


def embedding_bag_sum(ids: Tensor, table: Tensor, atomic: bool) -> Tensor:
    return _BagSum.apply(ids, table) if atomic else F.embedding_bag(ids, table, mode="sum", padding_idx=0)


class StaticPrior(nn.Module):
    """Embedding rows computed from catalog static features (schema 12.4 / 13 "static prior").

    For each category in ``layout.static`` the catalog features (categorical fields one-hot,
    numeric fields symlog) form a fixed matrix; an MLP maps every row to ``d_id``. Rows without a
    catalog entry (PAD/UNK/slack) are zero. Without a catalog the whole region is zero.
    """

    def __init__(self, layout: TokenLayout, catalog: Catalog | None, d_id: int, d_hidden: int) -> None:
        super().__init__()
        self.n_rows = layout.n_rows - layout.n_learned
        self.d_id = d_id
        self.cats: list[str] = []
        self.mlps = nn.ModuleDict()
        if catalog is None:
            return
        for cat, (_, rows, levels) in layout.static.items():
            x, valid = _static_matrix(catalog, cat, rows, levels)
            self.register_buffer(f"x_{cat}", torch.from_numpy(x), persistent=False)
            self.register_buffer(f"v_{cat}", torch.from_numpy(valid), persistent=False)
            self.mlps[cat] = _mlp(x.shape[1], d_hidden, d_id)
            self.cats.append(cat)

    def forward(self, like: Tensor) -> Tensor:
        if not self.cats:
            return like.new_zeros(self.n_rows, self.d_id)
        parts = [self.mlps[c](getattr(self, f"x_{c}")) * getattr(self, f"v_{c}")[:, None] for c in self.cats]
        return torch.cat(parts).to(like.dtype)


def _static_matrix(catalog: Catalog, cat: str, rows: int, levels: int) -> tuple[np.ndarray, np.ndarray]:
    sf = catalog.features(cat)
    ids, nums = sf.ids, sf.nums
    if ids.ndim == 2:  # non-leveled category: add a level axis
        ids, nums = ids[:, None], nums[:, None]
    V, L = ids.shape[:2]
    vocab = catalog.vocab
    parts = [np.eye(vocab.size(c), dtype=np.float32)[np.clip(ids[..., j], 0, vocab.size(c) - 1)]
             for j, c in enumerate(sf.id_cats)]
    # symlog clipped to +-SYMLOG_CLIP like every frame feature (schema 13): the real catalog carries 999999999
    # sentinels (relic merchant_cost of unsold relics, card max_upgrade_level of endlessly upgradable cards)
    # whose raw symlog (20.7) would dwarf every other static input
    sym = np.clip(np.sign(nums) * np.log1p(np.abs(nums)), -SYMLOG_CLIP, SYMLOG_CLIP)
    feat = np.concatenate([*parts, sym], axis=-1).astype(np.float32)
    lv = np.minimum(np.arange(levels), L - 1)
    out = np.zeros((rows, levels, feat.shape[-1]), np.float32)
    out[:V] = feat[:, lv]
    valid = np.zeros((rows, levels), np.float32)
    known = np.array(sorted(catalog.entries.get(cat, {})), np.int64)
    if known.size:
        valid[known] = 1.0
    return out.reshape(rows * levels, -1), valid.reshape(-1)


class Block(nn.Module):
    """Pre-norm transformer block; both LayerNorms are FiLM-modulated by the conditioning vector."""

    def __init__(self, d: int, n_heads: int, d_ff: int, d_cond: int,
                 char_experts: int = 0, char_expert_ff: int = 0) -> None:
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d_model {d} not divisible by n_heads {n_heads}")
        self.h = n_heads
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)  # affine supplied by FiLM
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)
        self.film = nn.Linear(d_cond, 4 * d)
        self.char_experts = char_experts
        if char_experts > 0:
            assert char_expert_ff > 0
            self.exp_w1 = nn.Parameter(torch.empty(char_experts, d, char_expert_ff))
            self.exp_b1 = nn.Parameter(torch.zeros(char_experts, char_expert_ff))
            self.exp_w2 = nn.Parameter(torch.zeros(char_experts, char_expert_ff, d))
            self.exp_b2 = nn.Parameter(torch.zeros(char_experts, d))

    def forward(self, x: Tensor, mask: Tensor, cond: Tensor, char_id: Tensor | None = None) -> Tensor:
        B, T, d = x.shape
        s1, b1, s2, b2 = self.film(cond).unsqueeze(1).chunk(4, dim=-1)
        h = self.ln1(x) * (1 + s1) + b1
        q, k, v = self.qkv(h).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        h = self.ln2(x) * (1 + s2) + b2
        out = x + self.ff2(F.gelu(self.ff1(h), approximate="tanh"))
        if self.char_experts > 0:
            assert char_id is not None
            w1 = self.exp_w1[char_id]
            eb1 = self.exp_b1[char_id].unsqueeze(1)
            w2 = self.exp_w2[char_id]
            eb2 = self.exp_b2[char_id].unsqueeze(1)
            eh = torch.bmm(h, w1)
            e = F.gelu(eh + eb1.to(eh.dtype), approximate="tanh")
            eo = torch.bmm(e, w2)
            out = out + eo + eb2.to(eo.dtype)
        return out


class SpireNet(nn.Module):
    """Entity transformer policy/value network.

    ``forward(batch)`` returns ``logits`` (B, A) float32 with padded actions at ``-inf``,
    ``value``, ``pwin_logit``, ``floors_pred``, ``hp_pred`` (B,) float32. Works under
    ``torch.autocast(dtype=torch.bfloat16)`` and ``torch.inference_mode``.
    """

    id_droppable: Tensor
    _intent_missing_columns: Tensor
    val_edges: Tensor
    val_centers: Tensor

    def __init__(self, layout: TokenLayout, config: ModelConfig = PRESETS["2.5m"],
                 catalog: Catalog | None = None) -> None:
        super().__init__()
        if catalog is not None and catalog.vocab.hash != layout.vocab_hash:
            raise ValueError("catalog and layout were built from different vocabularies")
        self.config = config
        self.layout_hash = layout.vocab_hash
        d, c = config.d_model, config.d_cond
        self.decoder = FeatureDecoder(layout)
        self.ids = nn.Embedding(layout.n_learned, config.d_id, padding_idx=0)
        self.static = StaticPrior(layout, catalog, config.d_id, config.d_static)
        droppable = torch.zeros(layout.n_rows, dtype=torch.bool)
        for cat in self.static.cats:
            off, rows = layout.regions[cat]
            droppable[off : off + rows] = True
        self.register_buffer("id_droppable", droppable, persistent=False)
        self.type_emb = nn.Embedding(layout.n_types, d)
        self._fc_lo, self._fc_hi = layout.forecast_cols  # FORECAST's own wide columns; in_proj reads [:fc_lo]
        self._cc_lo, self._cc_hi = layout.calc_cols  # the calc rows' own wide columns (after the forecast's)
        self.in_proj = nn.Linear(config.d_id + self._fc_lo, d)
        self.link_own = nn.Linear(d, d, bias=False)
        self.link_rel = nn.Linear(d, d, bias=False)
        self.phi_var = _norm_mlp(d, d)
        self.phi_kid = _norm_mlp(d, d)
        self.in_norm = nn.LayerNorm(d)
        self.cond_asc = nn.Embedding(16, c)
        self.cond_seats = nn.Embedding(5, c)
        self.cond_char = nn.Embedding(layout.character_rows, c)
        self.cond_mlp = _mlp(c, c, c)
        self.blocks = nn.ModuleList([Block(d, config.n_heads, config.d_ff, c,
                                           config.char_experts, config.char_expert_ff)
                                     for _ in range(config.n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.act_refs = nn.Linear(3 * d, d, bias=False)  # [source | target | acting seat] token states
        self.act_ctx = nn.Linear(d, d, bias=False)
        # no final bias: a shift common to all logits gets no gradient through the softmax
        self.act_score = nn.Sequential(nn.LayerNorm(d, elementwise_affine=False), nn.Linear(d, d),
                                       nn.GELU(approximate="tanh"), nn.Linear(d, 1, bias=False))
        self.value_head: nn.Sequential | None = None
        self.val_dist: nn.Sequential | None = None
        if config.value_bins > 0:
            self.val_dist = _norm_mlp(d, config.value_bins)
            edges = torch.linspace(config.value_vmin, config.value_vmax, config.value_bins + 1, dtype=torch.float32)
            self.register_buffer("val_edges", edges, persistent=False)
            self.register_buffer("val_centers", 0.5 * (edges[:-1] + edges[1:]), persistent=False)
        else:
            self.value_head = _norm_mlp(d, 1)
        self.aux_head = _norm_mlp(d, 3)
        self.history_proj: nn.Linear | None = None
        self.reset_parameters()
        # Construct after the original initialization: disabled history retains the original RNG path.
        if layout.public_history:
            self.history_proj = nn.Linear(HISTORY_ROWS * d + 2 * config.d_id + HISTORY_STATS, d, bias=False)
            nn.init.zeros_(self.history_proj.weight)
        self._history_kind = layout.regions["p:kind"][0]
        self._history_kind_sub = layout.regions["p:kind_sub"][0]
        self.cycle_ban = layout.cycle_ban
        # spire.rules: unless "mask" the forward adds the removal teach term (no parameter); a plain attribute, so an
        # evaluation may play a "teach" network "free" (spire.habit_eval --rules-mode free)
        self.rules_mode = layout.rules_mode
        self.multiseat_intent_missing: nn.Parameter | None = None
        missing = [config.d_id + layout.num_offsets[Ent.INTENT] + j
                   for j, feature in enumerate(SPECS[Ent.INTENT].nums) if feature.mode == Mode.MISS]
        assert len(missing) == 4
        self.register_buffer("_intent_missing_columns", torch.tensor(missing), persistent=False)
        if config.seat_symmetry:
            # Missing seat 1..3 features are ONE, not zero, in single-player frames. Their
            # trained constant columns must survive exactly on that path. Multi-seat frames
            # use a separate four-column matrix; all other projection weights stay shared.
            self.multiseat_intent_missing = nn.Parameter(self.in_proj.weight[:, missing].detach().clone())
        self.coop_screen: nn.Embedding | None = None
        self._legacy_input_width = config.d_id + layout.legacy_n_wide
        self._seat_screen_slot = 0
        if layout.coop_features:
            # Keep the legacy embedding-bag reduction unchanged. This categorical
            # contribution is separate, zero initially, and attached to SEATS only.
            self.coop_screen = nn.Embedding(16, d, padding_idx=0)
            nn.init.zeros_(self.coop_screen.weight)
            self._seat_screen_slot = layout.derived_slot[Ent.SEAT]["acts"]
            with torch.no_grad():
                self.in_proj.weight[:, self._legacy_input_width:].zero_()
        self.bag_atomic = False  # atomic embedding-bag table gradient (_BagSum); identical forward
        self.mend_target_gate: nn.Parameter | None = None
        if layout.mend_target:
            self.mend_target_gate = nn.Parameter(torch.zeros(()))
        # Constructed last: a layout without forecast keeps every earlier RNG draw and parameter.
        self.forecast_in: nn.Linear | None = None
        self.forecast_gate: nn.Parameter | None = None
        if layout.forecast:
            self.forecast_in = nn.Linear(self._fc_hi - self._fc_lo, d, bias=False)
            nn.init.normal_(self.forecast_in.weight, std=config.init_std)
            self.forecast_gate = nn.Parameter(torch.zeros(()))
        # Calc: constructed after the forecast's parameters; drawn from a fixed generator (as the migration does).
        self.calc_in: nn.Linear | None = None
        self.calc_gate: nn.Parameter | None = None
        if layout.calc:
            self.calc_in = nn.Linear(self._cc_hi - self._cc_lo, d, bias=False)
            g = torch.Generator().manual_seed(CALC_INIT_SEED)
            with torch.no_grad():
                new_rows = calc_type_rows(layout.forecast)
                self.type_emb.weight[new_rows:N_ENT_CALC] = torch.empty(N_ENT_CALC - new_rows, d).normal_(
                    0.0, config.init_std, generator=g)
                self.calc_in.weight.copy_(torch.empty(self.calc_in.weight.shape).normal_(
                    0.0, config.init_std, generator=g))
            self.calc_gate = nn.Parameter(torch.zeros(len(CALC_ENTS)))
        # Route summary: constructed after the calc parameters (a model without it keeps its RNG path); type_emb row
        # 24 and ahead_in from a fixed generator, as the migration draws them; the gate is 0 for born and migrated
        # networks alike.
        self._ah_lo, self._ah_hi = layout.ahead_cols
        self.ahead_in: nn.Linear | None = None
        self.ahead_gate: nn.Parameter | None = None
        if layout.route_summary:
            self.ahead_in = nn.Linear(self._ah_hi - self._ah_lo, d, bias=False)
            row, w = ahead_init(d, self.ahead_in.weight.shape, config.init_std)
            with torch.no_grad():
                self.type_emb.weight[int(Ent.AHEAD)] = row
                self.ahead_in.weight.copy_(w)
            self.ahead_gate = nn.Parameter(torch.zeros(()))
        # The critic tower: constructed last, under a forked RNG (a network with the tower leaves the global RNG where
        # one without it does) and initialised from its own generator (notes/critic/contract.md 2.2).
        self.critic: CriticTower | None = None
        if config.critic is not None:
            with torch.random.fork_rng(devices=[]):
                self.critic = CriticTower(config.critic, d, c, config.init_std)

    def reset_parameters(self) -> None:
        std = self.config.init_std
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()
        out_std = std / (2 * self.config.n_layers) ** 0.5  # GPT-2 style residual scaling
        for b in cast(Sequence[Block], self.blocks):
            nn.init.normal_(b.proj.weight, std=out_std)
            nn.init.normal_(b.ff2.weight, std=out_std)
            if b.char_experts > 0:
                nn.init.normal_(b.exp_w1, std=std)
                nn.init.zeros_(b.exp_b1)
                nn.init.zeros_(b.exp_w2)
                nn.init.zeros_(b.exp_b2)
        if self.history_proj is not None:
            nn.init.zeros_(self.history_proj.weight)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ forward
    def character_condition(self, chars: Tensor) -> Tensor:
        """Mean of present characters; the zero-seat row is collation padding."""
        present = chars > 0
        summed = (self.cond_char(chars) * present.unsqueeze(-1)).sum(1)
        return summed / present.sum(1).clamp_min(1).unsqueeze(-1)

    def _project_entities(self, features: Tensor, types: Tensor, multi: Tensor) -> Tensor:
        if self.coop_screen is None:
            legacy = self.in_proj(features)
            weight = self.in_proj.weight
        else:
            # Widening GEMM's reduction can change old floating-point bits even
            # when the added columns are zero. Preserve its original shape.
            base = features[:, :self._legacy_input_width].contiguous()
            weight = self.in_proj.weight[:, :self._legacy_input_width].contiguous()
            legacy = F.linear(base, weight, self.in_proj.bias)
            extra = F.linear(features[:, self._legacy_input_width:],
                             self.in_proj.weight[:, self._legacy_input_width:])
            features = base
        if self.multiseat_intent_missing is None:
            result = legacy
        else:
            weight = weight.index_copy(1, self._intent_missing_columns, self.multiseat_intent_missing)
            projected = F.linear(features, weight, self.in_proj.bias)
            intent_like = types == int(Ent.INTENT)
            if self.forecast_in is not None:  # forecast rows mirror the intent columns, so their override too
                intent_like = intent_like | (types == int(Ent.FORECAST))
            result = torch.where((intent_like & multi).unsqueeze(-1), projected, legacy)
        return result if self.coop_screen is None else result + extra

    def _coop_screen_features(self, types: Tensor, raw: Tensor) -> Tensor:
        assert self.coop_screen is not None
        screen = torch.where(types == int(Ent.SEAT), (raw[:, self._seat_screen_slot].long() >> 1) & 15, 0)
        return self.coop_screen(screen)

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        return self._forward(batch)[0]

    def forward_critic(self, batch: Batch, crit: Tensor) -> dict[str, Tensor]:
        """``forward``'s outputs (the same tensors: the logits are computed before CRIT is read) plus the tower's
        ``vc`` (B, 3) = (V_WIN, V_PROG, V_SURV) and ``win_logit_c`` (B,) from the detached final token states and
        condition vector and ``crit`` (B, 2 + K) (notes/critic/contract.md 2.2)."""
        assert self.critic is not None, "forward_critic needs model.critic"
        out, h, c = self._forward(batch)
        vc, win_logit = self.critic(h, c, batch["tok_mask"], crit)
        return {**out, "vc": vc, "win_logit_c": win_logit}

    def _forward(self, batch: Batch) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        """(outputs, final token states after ``out_norm`` (B, T, d), condition vector (B, d_cond))."""
        tok_mask, act_mask = batch["tok_mask"], batch["act_mask"]
        B, T = tok_mask.shape
        A = act_mask.shape[1]
        Nk, Nv = batch["kid_mask"].shape[0], batch["var_mask"].shape[0]
        BT = B * T
        k0, v0, a0 = BT, BT + Nk, BT + Nk + Nv
        N = a0 + B * A
        d = self.config.d_model

        # 1. entity embedding (unpadded rows), scattered into the padded layout: rows 0..N-1, zero
        #    row N ("none"), sink row N+1 for capacity padding (dropped)
        types, dest = batch["types"], batch["dest"]
        ids, wide = self.decoder(types, batch["raw"])
        if self.ahead_in is not None:
            wide_ah = wide[:, self._ah_lo : self._ah_hi]
        if self.calc_in is not None:
            wide_cc = wide[:, self._cc_lo : self._cc_hi]
        if self.forecast_in is not None:
            wide_fc = wide[:, self._fc_lo : self._fc_hi]
        if self.forecast_in is not None or self.calc_in is not None:
            wide = wide[:, : self._fc_lo]
        p = self.config.id_dropout
        if self.training and p > 0 and self.static.cats:
            drop = self.id_droppable[ids] & (torch.rand(ids.shape[0], 1, device=ids.device) < p)
            ids = ids.masked_fill(drop, 0)
        table = torch.cat([self.ids.weight, self.static(self.ids.weight)])
        e_id = embedding_bag_sum(ids, table, self.bag_atomic)
        if self.multiseat_intent_missing is None and self.coop_screen is None:
            x = self.in_proj(torch.cat([e_id, wide], 1)) + self.type_emb(types)
        else:
            # INTENT owner links point at creature tokens (also with pool_children=False).
            # Other rows' links can be null/actions; entity type masks their selector.
            owner_sample = (batch["rel"][:, 0] // T).clamp(0, B - 1)
            multi = batch["cond"][owner_sample, 1] > 1
            x = self._project_entities(torch.cat([e_id, wide], 1), types, multi) + self.type_emb(types)
        if self.coop_screen is not None:
            x = x + self._coop_screen_features(types, batch["raw"])
        if self.forecast_in is not None:  # rows of other types are selected unchanged
            x = torch.where((types == int(Ent.FORECAST)).unsqueeze(-1), x + self.forecast_in(wide_fc), x)
        if self.calc_in is not None:  # TURN / FOE / ROUTE rows; rows of other types are selected unchanged
            is_calc = (types >= int(Ent.TURN)) & (types <= int(Ent.ROUTE))
            x = torch.where(is_calc.unsqueeze(-1), x + self.calc_in(wide_cc), x)
        if self.ahead_in is not None:  # the AHEAD rows; rows of other types are selected unchanged
            x = torch.where((types == int(Ent.AHEAD)).unsqueeze(-1), x + self.ahead_in(wide_ah), x)
        H = x.new_zeros(N + 2, d).index_copy(0, dest, x)[: N + 1]
        R = torch.full((N + 2, 3), N, dtype=torch.long, device=x.device).index_copy(0, dest, batch["rel"])[: N + 1]

        # 2. relations: token owner links, token / kid / edge related-entity links (one gather)
        edges = batch["edges"]
        Ne = edges.shape[0]
        idx = torch.cat([R[:BT, 0], R[:BT, 1], R[k0:v0, 1], edges[:, 1]])
        g_own, g_rel = _take(H, idx, idx < N).split([BT, BT + Nk + Ne])
        l_tok, l_kid, l_edge = self.link_rel(g_rel).split([BT, Nk, Ne])
        tok = H[:BT] + self.link_own(g_own) + l_tok
        kid = H[k0:v0] + l_kid
        edge = self.type_emb(edges[:, 2] + int(Ent.EDGE_OUT)) + l_edge

        # 3. pooling: vars -> tokens / kids / actions, then kids + edges -> tokens
        P = torch.cat([tok, kid, H[v0:]])
        P = P.index_add(0, R[v0:a0, 0], self.phi_var(H[v0:a0]).to(P.dtype))
        pooled = self.phi_kid(torch.cat([P[k0:v0], edge.to(P.dtype)]))
        if self.forecast_gate is not None or self.calc_gate is not None:
            # kid types of the padded layout come from the unpadded types and their destination rows; edges: 0
            kid_type = torch.cat([types.new_zeros(N + 2).index_copy(0, dest, types)[k0:v0], types.new_zeros(Ne)])
        if self.forecast_gate is not None:
            # gate the FORECAST kids' pooled contribution (exactly +-0 while the gate is 0)
            is_fc = kid_type == int(Ent.FORECAST)
            pooled = torch.where(is_fc.unsqueeze(-1), self.forecast_gate.to(pooled.dtype) * pooled, pooled)
        if self.calc_gate is not None:
            # gate k of the TURN / FOE / ROUTE kids (exactly +-0 while the gates are 0: notes/calc/contract.md 3.3)
            for k, e in enumerate(CALC_ENTS):
                pooled = torch.where((kid_type == int(e)).unsqueeze(-1), self.calc_gate[k].to(pooled.dtype) * pooled,
                                     pooled)
        if self.ahead_gate is not None:
            # the AHEAD kid's pooled contribution to its GLOBAL token (exactly +-0 while the gate is 0: contract 2.4)
            pooled = torch.where((kid_type == int(Ent.AHEAD)).unsqueeze(-1), self.ahead_gate.to(pooled.dtype) * pooled,
                                 pooled)
        P = P.index_add(0, torch.cat([R[k0:v0, 0], edges[:, 0]]), pooled.to(P.dtype))

        # 4. transformer over tokens
        cond = batch["cond"]
        chars = cond[:, 2:].clamp(0, self.cond_char.num_embeddings - 1)
        c = (self.cond_asc(cond[:, 0].clamp(0, 15)) + self.cond_seats(cond[:, 1].clamp(0, 4))
             + self.character_condition(chars))
        c = self.cond_mlp(c)
        h = self.in_norm(P[:BT]).view(B, T, d)
        if self.history_proj is not None:
            # A separate residual preserves every original matrix shape and operation at migration.
            # Historical entities use the same public feature decoder/embeddings as current entities.
            history = batch["history"]
            ht = history[:, :HISTORY_ROWS].reshape(-1).long()
            hr = history[:, HISTORY_ROWS:-HISTORY_STATS].reshape(B * HISTORY_ROWS, -1)
            hi, hw = self.decoder(ht, hr)
            hw = hw[:, : self._fc_lo]  # history rows are never FORECAST / calc kids
            he = embedding_bag_sum(hi, table, self.bag_atomic)
            if self.multiseat_intent_missing is None and self.coop_screen is None:
                hx = self.in_proj(torch.cat([he, hw], 1)) + self.type_emb(ht)
            else:
                hm = (cond[:, 1] > 1).repeat_interleave(HISTORY_ROWS)
                hx = self._project_entities(torch.cat([he, hw], 1), ht, hm) + self.type_emb(ht)
            if self.coop_screen is not None:
                hx = hx + self._coop_screen_features(ht, hr)
            hx = (hx * (ht > 0).unsqueeze(1)).reshape(B, HISTORY_ROWS, d)
            if self.mend_target_gate is not None:
                # PublicHistory descriptor3 is the previous action's target.
                # Newly recorded MEND targets must share the live-action gate;
                # otherwise an already-trained history projection sees new input.
                mend_history = ht.reshape(B, HISTORY_ROWS)[:, 3] == int(Ent.SEAT)
                scale = torch.where(mend_history, self.mend_target_gate, 1)
                hx = torch.cat([hx[:, :3], hx[:, 3:4] * scale[:, None, None], hx[:, 4:]], dim=1)
            hx = hx.reshape(B, -1)
            stats = history[:, -HISTORY_STATS:].long()
            kind = stats[:, 0].clamp(0, 15)
            sub = kind * 16 + stats[:, 1].clamp(0, 15)
            kinds = torch.cat([self.ids(kind + self._history_kind),
                               self.ids(sub + self._history_kind_sub)], 1) * (kind > 0).unsqueeze(1)
            nums = stats.float()
            nums = nums.sign() * nums.abs().log1p()
            residual = self.history_proj(torch.cat([hx, kinds, nums.to(hx.dtype)], 1))
            h = torch.cat([h[:, :1] + residual[:, None], h[:, 1:]], 1)
        attn_mask = tok_mask[:, None, None, :]
        char_id = cond[:, 2].clamp(0, self.config.char_experts - 1) if self.config.char_experts > 0 else None
        for blk in self.blocks:
            h = blk(h, attn_mask, c, char_id) if char_id is not None else blk(h, attn_mask, c)
        h = self.out_norm(h)

        # 5. policy over action rows
        ra = R[a0:N].reshape(-1)
        refs = _take(h.reshape(BT, d), ra, ra < BT).view(B * A, 3 * d)  # token rows; "none" is N
        if self.mend_target_gate is not None:
            # SEATS are tokens [2, 2+n_seats); all ordinary action targets are
            # CREATURES. The only target-SEATS relation is the MEND target.
            target = R[a0:N, 1]
            local = target % T
            is_mend = (target < BT) & (local >= 2) & (local < 2 + cond[:, 1].repeat_interleave(A))
            source_ref, target_ref, seat_ref = refs.chunk(3, dim=1)
            scale = torch.where(is_mend, self.mend_target_gate, 1).unsqueeze(1)
            refs = torch.cat([source_ref, target_ref * scale, seat_ref], dim=1)
        a = P[a0:N].to(h.dtype) + self.act_refs(refs) + self.act_ctx(h[:, 0]).repeat_interleave(A, 0)
        if self.cycle_ban:
            # All collators and graph inputs already carry the stored history.
            # One mask keeps behavior/learner mixtures and entropy on the same legal set.
            # A negative first slot -(row + 1) is the removal rule's "only this row" (spire.cycle).
            bans = batch["history"][:, -CYCLE_BAN_SLOTS:]
            action_ids = torch.arange(1, A + 1, device=act_mask.device)
            only = bans[:, :1]
            if self.rules_mode != "mask":
                return self._rules_policy(a, act_mask, action_ids, bans, h[:, 0], B, A), h, c
            act_mask = (act_mask & ~(action_ids[None, :, None] == bans[:, None, :]).any(2)
                        & ((only >= 0) | (action_ids[None, :] == -only)))
        logits = self.act_score(a).view(B, A).float().masked_fill(~act_mask, float("-inf"))

        # 6. heads on the global token
        g = h[:, 0]
        aux = self.aux_head(g).float()
        return {"logits": logits, **self._value_outputs(g),
                "pwin_logit": aux[:, 0], "floors_pred": aux[:, 1], "hp_pred": aux[:, 2]}, h, c

    def _value_outputs(self, g: Tensor) -> dict[str, Tensor]:
        if self.val_dist is None:
            assert self.value_head is not None
            return {"value": self.value_head(g).squeeze(-1).float()}
        v_logits = self.val_dist(g).float()
        v = (F.softmax(v_logits, dim=-1) * self.val_centers).sum(-1)
        return {"value": v, "value_logits": v_logits}

    def _rules_policy(self, a: Tensor, act_mask: Tensor, action_ids: Tensor, bans: Tensor, g: Tensor, B: int,
                      A: int) -> dict[str, Tensor]:
        """Steps 5 and 6 in "teach" / "free" (spire.rules): ``keep`` = host-legal minus the commit / cycle masked rows
        (the positive slots); acting logits: "teach" = the "mask" mode's (``keep`` and the removal rule's one row),
        "free" = ``keep``; plus the removal teach term over ``keep``."""
        assert self.rules_mode in ("teach", "free"), self.rules_mode
        keep = act_mask & ~(action_ids[None, :, None] == bans[:, None, :]).any(2)
        only = bans[:, :1]
        rule = (only < 0) & (action_ids[None, :] == -only)
        legal = keep if self.rules_mode == "free" else keep & ((only >= 0) | rule)
        score = self.act_score(a).view(B, A).float()
        aux = self.aux_head(g).float()
        return {"logits": score.masked_fill(~legal, float("-inf")), **self._value_outputs(g),
                "pwin_logit": aux[:, 0], "floors_pred": aux[:, 1], "hp_pred": aux[:, 2],
                **removal_terms(score, keep, rule)}


# =========================================================================== CUDA-graph inference


def _pow2(n: int, lo: int) -> int:
    return max(lo, 1 << max(0, n - 1).bit_length())


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    inputs: dict[str, Tensor]
    outputs: dict[str, Tensor]
    packed: Tensor | None = None  # device byte buffer behind ``inputs`` (packed graphs)


BATCH_FIELDS: tuple[tuple[str, torch.dtype], ...] = (
    ("types", torch.int64), ("raw", torch.int16), ("rel", torch.int64), ("dest", torch.int64),
    ("edges", torch.int64), ("tok_mask", torch.bool), ("act_mask", torch.bool), ("kid_mask", torch.bool),
    ("var_mask", torch.bool), ("cond", torch.int64), ("history", torch.int16))
"""Keys and dtypes of a collated :data:`~spire.tokenize.Batch` (see :meth:`Tokenizer.collate`)."""
_PACK_ALIGN = 64


def batch_field_shapes(shape: BatchShape, width: int, draw_k: int = 0) -> dict[str, tuple[int, ...]]:
    """Tensor shapes of a batch collated with ``shape=shape`` (``draw_k`` > 0: plus the critic's ``crit``)."""
    B, T, A, M = shape.batch, shape.tokens, shape.actions, shape.rows
    out = {"types": (M,), "raw": (M, width), "rel": (M, 3), "dest": (M,), "edges": (shape.edges, 3),
           "tok_mask": (B, T), "act_mask": (B, A), "kid_mask": (shape.kids,), "var_mask": (shape.vars,),
           "cond": (B, 6), "history": (B, history_width(width))}
    if draw_k:
        out["crit"] = (B, crit_width(draw_k))
    return out


def packed_layout(shape: BatchShape, width: int,
                  draw_k: int = 0) -> tuple[dict[str, tuple[int, tuple[int, ...], torch.dtype]], int]:
    """Byte layout of all batch tensors of ``shape`` in one buffer: ``{key: (offset, shape, dtype)}``, size. With
    ``draw_k`` > 0 the critic's side input ``crit`` (B, 2 + K) int16 follows the batch fields (notes/critic/contract.md
    3: CRIT is one more packed input of the actor's graph)."""
    shapes = batch_field_shapes(shape, width, draw_k)
    out: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
    off = 0
    for k, dt in BATCH_FIELDS + ((("crit", torch.int16),) if draw_k else ()):
        n = math.prod(shapes[k]) * torch.empty((), dtype=dt).element_size()
        out[k] = (off, shapes[k], dt)
        off += -(-n // _PACK_ALIGN) * _PACK_ALIGN
    return out, max(off, _PACK_ALIGN)


def packed_views(buf: Tensor, layout: dict[str, tuple[int, tuple[int, ...], torch.dtype]]) -> Batch:
    """Typed views of a uint8 buffer laid out by :func:`packed_layout`."""
    out: Batch = {}
    for k, (off, shp, dt) in layout.items():
        n = math.prod(shp) * torch.empty((), dtype=dt).element_size()
        out[k] = buf[off : off + n].view(dt).view(shp)
    return out


class GraphRunner:
    """CUDA-graph inference for :class:`SpireNet`: one captured graph per padded :class:`BatchShape`.

    Eager PyTorch on Windows is launch-bound for this model (about 5 ms of CPU per forward at
    any batch size up to 256); replaying a captured graph removes the per-op CPU cost. Shapes are
    rounded up (batch and actions to powers of two >= 8, kid / var / edge / entity rows to powers of
    two) so a few dozen graphs cover a run; the least recently used graph is dropped beyond
    ``max_graphs``. Weights may be updated in place between calls (``import_weights`` /
    ``load_state_dict``); replacing parameter tensors (``.to()``, new modules) requires :meth:`clear`.
    Returned tensors are fresh device tensors (first ``len(samples)`` rows).
    """

    def __init__(self, net: SpireNet, tokenizer: Tokenizer, *, max_graphs: int = 32,
                 dtype: torch.dtype = torch.bfloat16, critic: bool = False,
                 critic_forward: Callable[[Batch, Tensor], dict[str, Tensor]] | None = None,
                 include_pwin: bool = False) -> None:
        """``critic=True`` (a critic run's training actor): the graphs run ``forward_critic`` on the batch's ``crit``
        and the packed readback row is ``logits | value | vc[3]``; else ``forward`` (the tower is not in the graph).
        ``include_pwin=True`` appends ``pwin_logit`` as one extra float32 column after ``value`` (or ``vc``) so
        search/ExIt workers read ``pwin_logit`` from the same CUDA-graphed forward with zero extra passes.
        ``critic_forward`` (critic only): the callable the graphs run instead of ``net.forward_critic`` - its
        ``torch.compile`` (tree p7)."""
        self.net, self.tok, self.dtype, self.max_graphs = net, tokenizer, dtype, max_graphs
        assert not critic or net.critic is not None, "GraphRunner(critic=True) needs a network with model.critic"
        assert critic_forward is None or critic, "critic_forward is the entry of a critic graph"
        self.critic = critic
        self.include_pwin = include_pwin
        self._critic_forward = (critic_forward if critic_forward is not None else net.forward_critic) if critic else None
        self.draw_k = net.critic.cfg.draw_k if critic and net.critic is not None else 0
        self.device = next(net.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("GraphRunner needs a CUDA model")
        self.pool = torch.cuda.graph_pool_handle()
        # Keep one tiny graph in the pool for the runner's lifetime: if every graph of a private pool is
        # destroyed (e.g. the only graph replaced by a grown shape), the pool's use count drops to zero
        # and the next capture into it trips an internal allocator assert (torch 2.11).
        self._pin = torch.zeros(1, device=self.device)
        self._pin_graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            self._pin.add_(0)
        torch.cuda.current_stream(self.device).wait_stream(side)
        with CAPTURE_LOCK, torch.cuda.graph(self._pin_graph, pool=self.pool, capture_error_mode="thread_local"):
            self._pin.add_(0)
        self.graphs: OrderedDict[BatchShape, _Graph] = OrderedDict()
        self.captures = 0

    def shape_for(self, samples: Sequence[Sample]) -> BatchShape:
        s = self.tok.shape_of(samples)
        return BatchShape(_pow2(s.batch, 8), s.tokens, _pow2(s.actions, 8), _pow2(s.kids, 64),
                          _pow2(s.vars, 64), _pow2(s.edges, 64), _pow2(s.rows, 256))

    def shape_for_encoded(self, enc: EncodedBatch, lo: int = 0, hi: int | None = None) -> BatchShape:
        """:meth:`shape_for` of frames ``lo:hi`` of an :class:`~spire.tokenize.EncodedBatch`."""
        return self.pad_shape(self.tok.shape_encoded(enc, lo, hi))

    @staticmethod
    def pad_shape(s: BatchShape) -> BatchShape:
        """The graph shape of a batch of natural shape ``s`` (powers of two, see class docstring)."""
        return BatchShape(_pow2(s.batch, 8), s.tokens, _pow2(s.actions, 8), _pow2(s.kids, 64),
                          _pow2(s.vars, 64), _pow2(s.edges, 64), _pow2(s.rows, 256))

    def clear(self) -> None:
        self.graphs.clear()

    def _net(self, inputs: Batch) -> dict[str, Tensor]:
        if self.critic:
            assert self._critic_forward is not None
            return self._critic_forward({k: v for k, v in inputs.items() if k != "crit"}, inputs["crit"])
        return self.net(inputs)

    @torch.no_grad()
    def run_packed(self, host: Tensor, shape: BatchShape, n: int) -> Tensor:
        """Replay the graph of ``shape`` on a batch collated into the pinned uint8 buffer ``host``
        laid out by :func:`packed_layout`: the graph's inputs are views of one device buffer, so a
        replay costs one host->device copy instead of one per tensor. Returns the first ``n`` rows
        of the static ``(B, A + 1)`` output ``logits | value`` (``critic``: ``(B, A + 4)`` = ``logits | value | vc``;
        fp32; concatenated inside the graph; valid until the next replay of any graph)."""
        g = self.graphs.get(shape)
        if g is None or g.packed is None:
            g = self._capture(shape, packed=host)
        else:
            self.graphs.move_to_end(shape)
            g.packed.copy_(host, non_blocking=True)
        g.graph.replay()
        return g.outputs["_lv"][:n]

    def __call__(self, samples: Sequence[Sample]) -> dict[str, Tensor]:
        shape = self.shape_for(samples)
        return self.run(self.tok.collate(samples, shape=shape), shape, len(samples))

    @torch.no_grad()
    def run(self, batch: Batch, shape: BatchShape, n: int, *, clone: bool = True) -> dict[str, Tensor]:
        """Replay (capturing on first use) the graph of ``shape`` on a batch collated with
        ``shape=shape``; outputs of the first ``n`` samples. ``clone=False`` returns views of the
        graph's static outputs, valid only until the next replay of any graph."""
        g = self.graphs.get(shape)
        if g is None:
            g = self._capture(shape, batch)
        else:
            self.graphs.move_to_end(shape)
            for k, v in batch.items():
                g.inputs[k].copy_(v, non_blocking=True)
        g.graph.replay()
        return {k: v[:n].clone() if clone else v[:n] for k, v in g.outputs.items()}

    def _capture(self, shape: BatchShape, batch: Batch | None = None, *, packed: Tensor | None = None) -> _Graph:
        with CAPTURE_LOCK:
            return self._capture_locked(shape, batch, packed=packed)

    def _capture_locked(self, shape: BatchShape, batch: Batch | None = None, *,
                        packed: Tensor | None = None) -> _Graph:
        if self.net.training:
            raise RuntimeError("GraphRunner requires net.eval()")
        dev_buf = None
        if packed is not None:
            dev_buf = packed.to(self.device)
            inputs = packed_views(dev_buf, packed_layout(shape, self.tok.width, self.draw_k)[0])
        else:
            assert batch is not None
            inputs = {k: v.to(self.device) for k, v in batch.items()}
        assert ("crit" in inputs) == self.critic, "a critic graph takes crit, any other graph none"
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.autocast("cuda", dtype=self.dtype, cache_enabled=False):
            for _ in range(2):
                self._net(inputs)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        # thread_local: a learner thread may run CUDA work while an actor thread captures
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"), \
                torch.autocast("cuda", dtype=self.dtype, cache_enabled=False):
            outputs = self._net(inputs)
            if dev_buf is not None:  # packed graphs also produce the actor's readback block
                lv = [outputs["logits"].float(), outputs["value"].float()[:, None]]
                if self.critic:
                    lv.append(outputs["vc"].float())
                if self.include_pwin:
                    lv.append(outputs["pwin_logit"].float()[:, None])
                outputs["_lv"] = torch.cat(lv, 1)
        g = _Graph(graph, inputs, outputs, dev_buf)
        assert not self.critic or "vc" in outputs
        self.graphs[shape] = g
        self.captures += 1
        while len(self.graphs) > self.max_graphs:
            self.graphs.popitem(last=False)
        return g


# =========================================================================== weights


def export_weights(net: SpireNet) -> dict[str, Tensor]:
    """Plain CPU state dict (parameters only; all buffers are rebuilt from layout/catalog)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}


def import_weights(net: SpireNet, state: Mapping[str, Tensor]) -> None:
    """Load a state dict produced by :func:`export_weights` (strict)."""
    net.load_state_dict(state, strict=True)


def migrate_public_history_weights(state: Mapping[str, Tensor], net: SpireNet) -> dict[str, Tensor]:
    """Explicit one-way migration of pre-history weights; all original keys and shapes stay strict."""
    assert net.history_proj is not None, "public-history migration requires tokenizer.public_history=true"
    key = "history_proj.weight"
    expected = net.state_dict()
    assert set(state) == set(expected) - {key}, "migration requires exactly the pre-history model keys"
    assert all(state[k].shape == expected[k].shape for k in state), "migration cannot change existing shapes"
    return {**state, key: torch.zeros_like(expected[key], device="cpu")}


def state_to_cpu(module: nn.Module) -> dict[str, Tensor]:
    """CPU copy of ``module.state_dict()`` with ONE device->host copy per dtype (the tensors are flattened
    into one device buffer first). Per-tensor ``.cpu()`` costs one synchronous copy each: under WDDM GPU
    contention every one of them waits for a time slice of this process (measured: ~45 ms each, 4.4 s per
    publication of the 2.5m model next to an actor process)."""
    sd = module.state_dict()
    out: dict[str, Tensor] = {}
    by_dtype: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
    for k, v in sd.items():
        if v.device.type == "cpu":
            out[k] = v.detach().clone()
        else:
            by_dtype.setdefault(v.dtype, []).append((k, v.detach()))
    for items in by_dtype.values():
        flat = torch.cat([v.reshape(-1) for _, v in items]).cpu()
        off = 0
        for k, v in items:  # views of the one host buffer (torch.save stores the shared storage once)
            n = v.numel()
            out[k] = flat[off:off + n].view(v.shape)
            off += n
    return {k: out[k] for k in sd}


class PinnedState:
    """A state dict staged in pinned host memory (one buffer per dtype) for one asynchronous host->device copy
    per dtype followed by device-side copies into the parameters (:meth:`load_into`). Built off the hot loop
    (the actor's weight-poll thread); replaces ``load_state_dict``'s one synchronous pageable copy per tensor
    (~30 ms per weight load under GPU contention, measured)."""

    def __init__(self, state: Mapping[str, Tensor]) -> None:
        self.keys = list(state)
        self.groups: list[tuple[Tensor, list[tuple[str, int, int, torch.Size]]]] = []
        by_dtype: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
        for k, v in state.items():
            by_dtype.setdefault(v.dtype, []).append((k, v))
        for dt, items in by_dtype.items():
            total = sum(v.numel() for _, v in items)
            buf = torch.empty(total, dtype=dt, pin_memory=torch.cuda.is_available())
            spans = []
            off = 0
            for k, v in items:
                n = v.numel()
                buf[off:off + n].copy_(v.reshape(-1))
                spans.append((k, off, n, v.shape))
                off += n
            self.groups.append((buf, spans))

    @torch.no_grad()
    def load_into(self, module: nn.Module) -> None:
        """Copy into ``module``'s parameters / buffers in place (strict on the key set); on CUDA the copies are
        queued on the current stream without waiting (captured graphs keep their addresses)."""
        dst = module.state_dict(keep_vars=True)
        if set(dst) != set(self.keys):
            missing, extra = set(dst) - set(self.keys), set(self.keys) - set(dst)
            raise KeyError(f"state keys differ: missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")
        for buf, spans in self.groups:
            dev = dst[spans[0][0]].device
            src = buf.to(dev, non_blocking=True) if dev.type == "cuda" else buf
            for k, off, n, shape in spans:
                t = dst[k].data
                if t.shape != shape:
                    raise ValueError(f"{k}: shape {tuple(shape)} != {tuple(t.shape)}")
                t.copy_(src[off:off + n].view(shape), non_blocking=True)


def checkpoint(net: SpireNet) -> dict[str, Any]:
    """Weights plus what is needed to rebuild the network (config, vocab hash)."""
    return {"config": model_config_dict(net.config), "vocab_hash": net.layout_hash, "state_dict": export_weights(net)}


def from_checkpoint(ckpt: Mapping[str, Any], layout: TokenLayout, catalog: Catalog | None = None) -> SpireNet:
    if ckpt["vocab_hash"] != layout.vocab_hash:
        raise ValueError("checkpoint was trained with a different vocab")
    net = SpireNet(layout, ModelConfig(**ckpt["config"]), catalog)
    import_weights(net, ckpt["state_dict"])
    return net

