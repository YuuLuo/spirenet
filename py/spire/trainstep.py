"""Fast PPO micro-steps on CUDA for :class:`spire.learner.Learner` (py-stage2-2).

Why: the eager learner is CPU-bound on this machine (one core busy, GPU ~60% busy, 32k samples/s on real
frames) and synchronises with the device several times per micro-batch (non-pinned host->device copies,
``isfinite(loss)``, the minibatch denominators, the grad-norm read). Under WDDM the GPU is time-sliced
between processes, so every synchronisation waits for a slice of the actor's context: next to one actor
process the eager learner fell to ~11k samples/s. Here every micro-batch is

1. collated by the C kernel ``collate_idx`` straight from the iteration's concatenated sample block into a
   recycled pinned staging buffer (no GIL; ~9x faster than ``Tokenizer.collate``), together with the
   per-sample training columns (:data:`PF_COLS`);
2. copied to a static device buffer with ONE asynchronous host->device copy;
3. run as forward + PPO loss + backward + diagnostics accumulation - either eagerly (``graphs=False``) or by
   replaying a CUDA graph captured once per padded micro-batch shape (``graphs=True``), which removes the
   per-op Python / launch cost (a few hundred kernels per micro-batch).

The minibatch denominators live in a device tensor written by an asynchronous copy, gradients accumulate
into persistent ``.grad`` buffers (never set to None in this mode: the graphs hold their addresses), the
optimiser step skips non-finite minibatches on the device (fused AdamW ``found_inf``) and the diagnostics
are device sums read once per update - so the CPU queues a whole minibatch ahead of the GPU and never
waits for it.

Micro-batches here are strictly per token bucket (padded to their bucket, at most
``min(micro_batch, micro_tokens // bucket)`` samples) and padded with empty samples to a small set of batch
sizes (the bucket's cap, or a power of two >= 8 for the last partial chunk); the other dimensions are rounded
up and only ever grow per (batch, tokens), so a run captures a few dozen graphs in its first iterations and
none after that. Padded samples have ``valid = 0`` and contribute nothing. The loss is the one of
:meth:`spire.learner.Learner._minibatch` term for term (tests/test_learner_fast.py compares the two paths).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .distributions import (batch_choice_keys, choice_uniform_probs, exploration_table, kind_coefficients,
                            legal_uniform_log_probs, uniform_cross_entropy)
from .kickstart import distil_sums, kick_loss, smooth_table
from .critic import empty_crit
from .model import CAPTURE_LOCK, SpireNet, capture_graph, hl_gauss_probs, packed_layout, packed_views
from .rules import teach_loss
from .schema import ActionCol
from .taste import taste_log_prob
from .temperature import tempered_log_probs
from .tokenize import BatchShape, Ent, SampleBlock, Tokenizer

if TYPE_CHECKING:
    from .learner import PPOConfig, _Accum

PF_COLS = ("logp_b", "adv", "ret", "v_old", "pol", "ent_coef", "log_legal", "win", "win_mask", "floors", "hp",
           "hp_mask", "valid", "logp_ref", "t_keep", "t_add", "beh_temp", "combat")
"""Per-sample float32 training columns of a micro-batch (``valid`` = 1 for real samples, 0 for padding;
``t_keep`` / ``t_add``: the :mod:`spire.taste` terms, read only when taste is on; ``beh_temp`` / ``combat``: the
:mod:`spire.temperature` behaviour temperature of the step (read only in a tempered iteration) and its combat flag)."""
(C_LOGP, C_ADV, C_RET, C_VOLD, C_POL, C_ENT, C_LOGL, C_WIN, C_WINM, C_FLO, C_HP, C_HPM, C_VALID,
 C_REF, C_TKEEP, C_TADD, C_BTEMP, C_COMBAT) = range(len(PF_COLS))
NF = len(PF_COLS)
CRITIC_PF_COLS = ("ret_win", "ret_prog", "ret_surv")
"""The critic's columns, after :data:`PF_COLS` in a critic learner only (notes/critic/contract.md 4.3; the fight-head
columns of the contract are out of v1): the tower's targets RET_WIN (in [0, 1]), RET_PROG, RET_SURV."""
C_RWIN, C_RPROG, C_RSURV = range(NF, NF + len(CRITIC_PF_COLS))
NF_CRITIC = NF + len(CRITIC_PF_COLS)
DEN_COLS = ("pol", "win", "floors", "hp", "all", "v_scale", "epoch0")
"""Minibatch-level scalars of the loss: denominators, the value-loss scale and the epoch-0 flag."""
(D_POL, D_WIN, D_FLO, D_HP, D_ALL, D_VSCALE, D_E0) = range(len(DEN_COLS))
_ALIGN = 64


def round_dim(n: int, lo: int) -> int:
    """Padded size of a dimension: a power of two up to 1024, then the next multiple of 1/8 of the power of
    two above (<= 12.5% padding; few distinct sizes)."""
    n = max(n, lo)
    p = 1 << max(0, n - 1).bit_length()
    if p <= 1024:
        return p
    q = p // 8
    return -(-n // q) * q


def batch_cap(cfg: "PPOConfig", bucket: int) -> int:
    """Samples per micro-batch of token bucket ``bucket`` (the eager learner's micro_batch / micro_tokens)."""
    cap = cfg.micro_batch
    if cfg.micro_tokens is not None:
        cap = min(cap, max(1, cfg.micro_tokens // max(1, bucket)))
    return max(1, cap)


def bucket_micros(mb: np.ndarray, buckets: np.ndarray, cfg: "PPOConfig") -> list[tuple[int, np.ndarray]]:
    """Micro-batches of minibatch ``mb`` (sample indices): per token bucket (largest first, minibatch order
    inside a bucket), the bucket's samples split into the fewest chunks of at most :func:`batch_cap` samples,
    of equal size (so the padding to :func:`pad_batch` stays small). Returns ``(bucket, indices)`` pairs."""
    b = buckets[mb]
    order = np.argsort(-b, kind="stable")
    idx, bs = mb[order], b[order]
    out: list[tuple[int, np.ndarray]] = []
    if not len(idx):
        return out
    cuts = np.flatnonzero(np.diff(bs)) + 1
    for lo, hi in zip(np.r_[0, cuts].tolist(), np.r_[cuts, len(idx)].tolist()):
        t = int(bs[lo])
        k = -(-(hi - lo) // batch_cap(cfg, t))
        out.extend((t, part) for part in np.array_split(idx[lo:hi], k))
    return out


def pad_batch(b: int, cap: int) -> int:
    """Padded batch size of a micro-batch of ``b`` samples: 8 for small ones, else the next multiple of the
    largest power of two <= b / 8 (< 12.5% padding; 8-16 sizes per octave). ``cap`` (the bucket's micro-batch
    cap) is accepted for the caller's bookkeeping; chunks never exceed it."""
    del cap
    if b <= 8:
        return 8
    q = 1 << max(0, (b // 8).bit_length() - 1)
    return -(-b // q) * q


@dataclass(eq=False)
class _Stage:
    """Pinned host staging of one micro-batch of a shape (packed batch | per-sample floats | ints)."""

    buf: Tensor
    arr: dict[str, np.ndarray]
    pf: np.ndarray
    pi: np.ndarray
    event: Any = None  # torch.cuda.Event of the last host->device copy out of ``buf``
    prepared_for: np.ndarray | None = None


@dataclass(eq=False)
class _Res:
    """Static device buffers (and the graph) of one padded micro-batch shape."""

    shape: BatchShape
    dev: Tensor
    views: dict[str, Tensor]
    pf: Tensor
    pi: Tensor
    stages: list[_Stage] = field(default_factory=list)
    k: int = 0
    consumed: Any = None  # last use of dev on the compute stream when copies are overlapped
    # torch.cuda.CUDAGraph per value of MicroStepper.tempered (the body of a tempered iteration differs)
    graphs: dict[bool, Any] = field(default_factory=dict)


class MicroStepper:
    """Runs the PPO micro-steps of a :class:`~spire.learner.Learner` on CUDA (module docstring).

    The learner owns the model, optimiser and the persistent diagnostics accumulator ``acc``; this class
    owns the staging / static buffers, the device-side minibatch scalars :attr:`den` and the graphs."""
    # the iteration has steps with a behaviour temperature != 1 (set by the learner per iteration, spire.temperature)
    tempered: bool = False
    allow_lookahead: bool = False

    teacher: nn.Module | None = None  # the kickstart teacher (spire.kickstart); set in __init__ / set_teacher
    kick_acc: Tensor | None = None
    distil = False  # kickstart phase 0: kick term + value / auxiliary losses only
    _smooth: Tensor | None = None  # kickstart smoothing eps per decision kind
    rules_coef: Tensor | None = None  # spire.rules ("teach" / "free"): the update's removal coefficient, read at replay
    rules_acc: Tensor | None = None
    critic_acc: Tensor | None = None  # the critic's loss sums (WIN, PROG, SURV) of the update (a critic learner)
    critic = False  # a critic learner's stepper (set in __init__; class defaults for steppers built without it)
    draw_k = 0
    nf = NF
    _hl_gauss: bool = False  # HL-Gauss distributional value head (set in __init__; class defaults for __new__ tests)
    _val_edges: Tensor | None = None
    _val_vmin: float = 0.0
    _val_vmax: float = 0.0
    _val_sigma: float = 0.0

    def __init__(self, model: nn.Module, tokenizer: Tokenizer, cfg: "PPOConfig", device: torch.device,
                 acc: "_Accum", *, graphs: bool, autocast: bool = True, n_stages: int = 3,
                 compile_model: bool = False, teacher: nn.Module | None = None, kick_acc: Tensor | None = None,
                 rules_coef: Tensor | None = None, rules_acc: Tensor | None = None,
                 critic_acc: Tensor | None = None) -> None:
        """``critic_acc`` (iff ``cfg.critic``): the device sums of the critic losses; the micro-batch then carries
        CRIT (:func:`spire.model.packed_layout` with the tower's draw_k), the columns :data:`CRITIC_PF_COLS` and
        the forward is ``forward_critic`` (notes/critic/contract.md 4.3)."""
        if device.type != "cuda":
            raise ValueError("MicroStepper needs a CUDA device")
        self.model, self.tok, self.cfg, self.device, self.acc = model, tokenizer, cfg, device, acc
        self.critic = cfg.critic is not None
        assert self.critic == (critic_acc is not None), "critic loss sums iff ppo.critic"
        mcfg = getattr(model, "config", None)
        self._hl_gauss = bool(mcfg is not None and getattr(mcfg, "value_bins", 0) > 0)
        if self._hl_gauss:
            assert cfg.value_clip is None, "model.value_bins > 0 refuses ppo.value_clip"
            assert not cfg.normalize_returns, "model.value_bins > 0 refuses ppo.normalize_returns"
            self._val_edges: Tensor | None = model.val_edges  # type: ignore[assignment]
            self._val_vmin = float(mcfg.value_vmin)  # type: ignore[union-attr]
            self._val_vmax = float(mcfg.value_vmax)  # type: ignore[union-attr]
            self._val_sigma = float(mcfg.value_sigma)  # type: ignore[union-attr]
        else:
            self._val_edges = None
            self._val_vmin = self._val_vmax = self._val_sigma = 0.0
        self.critic_acc = critic_acc
        self.draw_k = getattr(model, "critic").cfg.draw_k if self.critic else 0
        self.nf = NF_CRITIC if self.critic else NF
        # torch.compile of the forward (dynamic shapes; the loss stays eager). Compiling inside a graph capture is not
        # allowed, so with graphs every capture is preceded by an eager warm-up step (which may recompile).
        self.compiled = compile_model
        self.fwd: nn.Module = torch.compile(model, dynamic=True) if compile_model else model  # type: ignore[assignment]
        # a critic learner's forward: ``forward_critic`` (torch.compile wraps ``forward`` only, so the critic entry is
        # compiled on its own; tree p7)
        self.fwd_critic: Any = ((torch.compile(model.forward_critic, dynamic=True) if compile_model  # type: ignore[operator]
                                 else model.forward_critic) if self.critic else None)
        self._choice_verb_slot = tokenizer.layout.slot(Ent.ACTION, ActionCol.VERB)
        self._exploration = (torch.as_tensor(exploration_table(cfg.exploration_by_kind), device=device)
                             if cfg.exploration_by_kind else None)
        self._uniform_kl = (torch.as_tensor(kind_coefficients(cfg.uniform_kl_by_kind), device=device)
                            if cfg.uniform_kl_by_kind else None)
        self.graphs = graphs
        self.autocast = autocast
        self.n_stages = n_stages
        self.den = torch.zeros(len(DEN_COLS), device=device)
        self._den_host = [torch.zeros(len(DEN_COLS), pin_memory=True) for _ in range(4)]
        self._den_ev: list[Any] = [None] * 4
        self._den_k = 0
        # A's KL is known before clipping, AdamW, EMA and the copy back to B. Read
        # only this prefix of the CUDA stream when deciding whether to enqueue more work.
        self.stop_host = torch.empty((), dtype=torch.bool, pin_memory=True)
        self.stop_divisor = torch.empty((), device=device, dtype=torch.float64)
        self.kl_ready = torch.cuda.Event()
        self.stop_pending = False
        self._saved_sums: list[Tensor] = []
        self.next_micros: tuple[np.ndarray, list[tuple[int, np.ndarray]]] | None = None
        self.mb_kl = torch.zeros((), device=device)  # minibatch KL sum (early stop)
        # kickstart (spire.kickstart): the frozen teacher, run eagerly under no_grad on the micro-batch's static inputs
        # (inside the captured graph), this update's (pi_coef, v_coef) and the learner's metric sums
        self.teacher = teacher
        self.kick_coef = torch.zeros(2, device=device)
        self.kick_acc = kick_acc
        assert teacher is None or kick_acc is not None
        self.distil = cfg.kickstart.on and cfg.kickstart.phase == 0
        self._smooth = smooth_table(cfg.kickstart, device)
        # spire.rules: the learner's live coefficient and metric sums (both or neither: rules_mode != "mask")
        assert (rules_coef is None) == (rules_acc is None)
        self.rules_coef, self.rules_acc = rules_coef, rules_acc
        self.pool = torch.cuda.graph_pool_handle() if graphs else None
        self._caps: dict[tuple[int, int], BatchShape] = {}
        self._res: dict[BatchShape, _Res] = {}
        self.captures = 0
        self._side: torch.cuda.Stream | None = None  # capture stream
        self.micro_steps = 0
        self.padded_samples = 0
        # One device input buffer for every shape (replays and their input copies are ordered on one stream, so
        # the shapes' static inputs may alias); a larger shape appends a new buffer, captured graphs keep theirs.
        self.copy_stream: torch.cuda.Stream | None = None
        self.shared_input = False
        self._shared: list[Tensor] = []

    # ------------------------------------------------------------------ minibatch scalars
    def set_den(self, pol: float, win: float, floors: float, hp: float, n_all: float, v_scale: float,
                epoch0: bool) -> None:
        k = self._den_k
        self._den_k = (k + 1) % len(self._den_host)
        if self._den_ev[k] is not None:
            self._den_ev[k].synchronize()
        h = self._den_host[k]
        h.copy_(torch.tensor([pol, win, floors, hp, n_all, v_scale, float(epoch0)], dtype=torch.float32))
        self.den.copy_(h, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._den_ev[k] = ev

    # ------------------------------------------------------------------ kickstart
    def set_kick(self, pi_coef: float, v_coef: float) -> None:
        """This update's kickstart coefficients (read by the captured graphs at replay)."""
        self.kick_coef[0].fill_(pi_coef)
        self.kick_coef[1].fill_(v_coef)

    def set_teacher(self, teacher: nn.Module | None) -> None:
        """Change the teacher; every captured graph is dropped (recaptured at its next use, into a new pool: a pool
        whose graphs are all gone cannot be captured into again)."""
        self.teacher = teacher
        for r in self._res.values():
            r.graphs.clear()
        if self.graphs:
            self.pool = torch.cuda.graph_pool_handle()

    # ------------------------------------------------------------------ shapes / buffers
    def shape_for(self, nat: BatchShape, cap: int) -> BatchShape:
        B = pad_batch(nat.batch, cap)
        need = BatchShape(B, nat.tokens, round_dim(nat.actions, 8), round_dim(nat.kids, 64), round_dim(nat.vars, 64),
                          round_dim(nat.edges, 64), round_dim(nat.rows, 256))
        key = (B, nat.tokens)
        cur = self._caps.get(key)
        if cur is not None and cur.fits(need):
            return cur
        if cur is not None:  # superseded: a larger shape for the same (batch, tokens); grow with 25% headroom
            self._res.pop(cur, None)
            grown = [max(x, y) for x, y in zip(cur.astuple(), need.astuple())]
            lo = (8, 8, 8, 64, 64, 64, 256)
            need = BatchShape(*grown[:2], *(round_dim(int(v * 1.25), m) if v > c else v
                                            for v, c, m in zip(grown[2:], cur.astuple()[2:], lo[2:])))
        self._caps[key] = need
        return need

    def _resources(self, shape: BatchShape) -> _Res:
        r = self._res.get(shape)
        if r is not None:
            return r
        layout, nb = packed_layout(shape, self.tok.width, self.draw_k)
        B = shape.batch
        NF = self.nf  # noqa: N806  (a critic learner: PF_COLS + CRITIC_PF_COLS)
        off_f = -(-nb // _ALIGN) * _ALIGN
        off_i = off_f + -(-(B * NF * 4) // _ALIGN) * _ALIGN
        size = off_i + B * 2 * 8
        if self.shared_input:
            if not self._shared or self._shared[-1].numel() < size:
                grow = int(1.5 * self._shared[-1].numel()) if self._shared else 0
                self._shared.append(torch.empty(max(size, grow), dtype=torch.uint8, device=self.device))
            dev = self._shared[-1][:size]
        else:
            dev = torch.empty(size, dtype=torch.uint8, device=self.device)
        views = packed_views(dev, layout)
        r = _Res(shape, dev, views, dev[off_f:off_f + B * NF * 4].view(torch.float32).view(B, NF),
                 dev[off_i:off_i + B * 16].view(torch.int64).view(B, 2))
        for _ in range(self.n_stages):
            buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            hv = packed_views(buf, layout)
            r.stages.append(_Stage(buf, {k: v.numpy() for k, v in hv.items()},
                                   buf[off_f:off_f + B * NF * 4].view(torch.float32).view(B, NF).numpy(),
                                   buf[off_i:off_i + B * 16].view(torch.int64).view(B, 2).numpy()))
        self._res[shape] = r
        return r

    def stage_stop(self, total: Tensor, count: float, threshold: float) -> None:
        # Match Python's float(total) / count > threshold, including strict
        # comparison at the boundary. A NaN does not request an early stop.
        # Use a device divisor: the scalar CUDA division kernel may replace a
        # host divisor by its reciprocal, changing a comparison by one double ULP.
        self.stop_divisor.fill_(count)
        stop = total.to(torch.float64).div(self.stop_divisor).gt(threshold)
        self.stop_host.copy_(stop, non_blocking=True)
        self.kl_ready.record(torch.cuda.current_stream(total.device))

    def should_stop(self) -> bool:
        self.kl_ready.synchronize()
        return self.stop_host.item()

    def already_captured(self, blk: SampleBlock, idx: np.ndarray, bucket: int) -> bool:
        """A speculative backward may only replay a graph already in this shape cache."""
        nat = self.tok.shape_block(blk, idx)
        shape = self._caps.get((pad_batch(nat.batch, batch_cap(self.cfg, bucket)), nat.tokens))
        if shape is None or not shape.fits(nat):
            return False
        r = self._res.get(shape)
        return r is not None and self.tempered in r.graphs

    def replay_is_pure_backward(self) -> bool:
        """Extra replays may mutate only registered diagnostic sums and persistent gradients."""
        model = self.model
        if type(model) is not SpireNet or model.config.id_dropout or torch.is_anomaly_enabled():
            return False
        if self.teacher is not None or self.critic:
            return False
        hooks = torch.nn.modules.module
        if (hooks._global_forward_hooks or hooks._global_forward_pre_hooks or hooks._global_backward_hooks
                or getattr(hooks, "_global_backward_pre_hooks", None)):
            return False
        return (not any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks
                        for m in model.modules())
                and not any(p._backward_hooks or getattr(p, "_post_accumulate_grad_hooks", None)
                            for p in model.parameters()))

    def _micro_state(self) -> list[Tensor]:
        tensors = [*self.acc.tensors(), self.den, self.mb_kl]
        if self.rules_acc is not None:
            tensors.append(self.rules_acc)
        return tensors

    def save_micro_state(self) -> tuple[int, int]:
        tensors = self._micro_state()
        if not self._saved_sums:
            self._saved_sums = [torch.empty_like(t) for t in tensors]
        torch._foreach_copy_(self._saved_sums, tensors)
        return self.micro_steps, self.padded_samples

    def restore_micro_state(self, counters: tuple[int, int]) -> None:
        torch._foreach_copy_(self._micro_state(), self._saved_sums)
        self.micro_steps, self.padded_samples = counters

    def prepack(self, blk: SampleBlock, idx: np.ndarray, pf: np.ndarray, pi: np.ndarray, bucket: int) -> None:
        """Use an idle pinned slot while the previous minibatch's stop test is pending.

        Do not grow a shape, advance the ring, or launch model work. The caller
        reuses this exact indices object only if the next minibatch is accepted.
        """
        nat = self.tok.shape_block(blk, idx)
        key = (pad_batch(nat.batch, batch_cap(self.cfg, bucket)), nat.tokens)
        shape = self._caps.get(key)
        if shape is None or not shape.fits(nat):
            return
        r = self._res.get(shape)
        if r is None or r.consumed is None or self.copy_stream is None or self.shared_input:
            return
        slot = r.stages[r.k]
        if slot.prepared_for is not None or (slot.event is not None and not slot.event.query()):
            return
        self._fill(slot, blk, idx, shape, pf[idx], pi[idx], None)
        self._upload(r, slot, self.copy_stream)
        slot.prepared_for = idx

    def _upload(self, r: _Res, slot: _Stage, stream: torch.cuda.Stream) -> None:
        stream.wait_event(r.consumed)
        with torch.cuda.stream(stream):
            r.dev.copy_(slot.buf, non_blocking=True)
            if slot.event is None:
                slot.event = torch.cuda.Event()
            slot.event.record(stream)

    def _fill(self, st: _Stage, blk: SampleBlock, idx: np.ndarray, shape: BatchShape,
              pf_rows: np.ndarray, pi_rows: np.ndarray, crit_rows: np.ndarray | None) -> None:
        b = len(idx)
        self.tok.collate_block_into(blk, idx, shape, st.arr)
        st.pf[:b] = pf_rows
        st.pf[b:] = 0.0
        st.pi[:b] = pi_rows
        st.pi[b:] = 0
        if crit_rows is not None:
            st.arr["crit"][:b] = crit_rows
            st.arr["crit"][b:] = empty_crit(shape.batch - b, self.draw_k)

    # ------------------------------------------------------------------ one micro-batch
    def run(self, blk: SampleBlock, idx: np.ndarray, pf_rows: np.ndarray, pi_rows: np.ndarray, bucket: int,
            crit_rows: np.ndarray | None = None) -> None:
        """One micro-step over samples ``idx`` of ``blk`` with their training columns ``pf_rows`` (b, NF)
        float32 and ``pi_rows`` (b, 2) int64 (action, kind); gradients accumulate into ``.grad``. A critic
        learner: ``crit_rows`` (b, 2 + K) int16, the samples' CRIT."""
        assert (crit_rows is not None) == self.critic, "CRIT rows iff a critic learner"
        nat = self.tok.shape_block(blk, idx)
        shape = self.shape_for(nat, batch_cap(self.cfg, bucket))
        r = self._resources(shape)
        st = r.stages[r.k]
        r.k = (r.k + 1) % len(r.stages)
        uploaded = st.prepared_for is idx
        if not uploaded:
            if st.event is not None:
                st.event.synchronize()
            self._fill(st, blk, idx, shape, pf_rows, pi_rows, crit_rows)
        st.prepared_for = None
        b = len(idx)
        copy_stream = self.copy_stream
        if copy_stream is None or self.shared_input:
            r.dev.copy_(st.buf, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            st.event = ev
        else:
            # An input of a different shape can be transferred while backward
            # runs. Reusing this shape's device buffer must wait for its last replay.
            compute = torch.cuda.current_stream(self.device)
            if r.consumed is None:
                r.dev.record_stream(copy_stream)
                r.consumed = torch.cuda.Event()
                r.consumed.record(compute)
            if not uploaded:
                self._upload(r, st, copy_stream)
            compute.wait_event(st.event)
        self.micro_steps += 1
        self.padded_samples += shape.batch - b
        if not self.graphs:
            self._body(r)
        else:
            if self.tempered not in r.graphs:
                self._capture(r)
            r.graphs[self.tempered].replay()
        if r.consumed is not None:
            r.consumed.record(torch.cuda.current_stream(self.device))

    def _body(self, r: _Res) -> None:
        """Forward + PPO loss + backward + diagnostics of the micro-batch in ``r``'s static buffers."""
        cfg, den, acc = self.cfg, self.den, self.acc
        views = r.views
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast, cache_enabled=False):
            if self.critic:  # forward_critic: the same trunk outputs plus the tower's (notes/critic/contract.md 2.2)
                views = {k: v for k, v in r.views.items() if k != "crit"}
                out = self.fwd_critic(views, r.views["crit"])
            else:
                out = self.fwd(views)
            if self.teacher is not None:
                with torch.no_grad():
                    t_out = self.teacher(views)
                out = {**out, "teacher_logits": t_out["logits"], "teacher_value": t_out["value"]}
        out = {k: v.float() for k, v in out.items()}
        pf, pi = r.pf, r.pi
        logits = out["logits"]
        legal = torch.isfinite(logits)
        reference_probs = (choice_uniform_probs(legal, batch_choice_keys(r.views, self._choice_verb_slot))
                           if cfg.exploration_choice_uniform else None)
        eps = None if self._exploration is None else self._exploration[pi[:, 1]]
        logp_all = legal_uniform_log_probs(logits, eps, reference_probs)
        # the behaviour's log q of every row, with the stored temperature of the step (spire.temperature)
        logp_beh = tempered_log_probs(logits, pf[:, C_BTEMP], eps, reference_probs) if self.tempered else logp_all
        a = pi[:, 0]
        valid = pf[:, C_VALID] > 0
        chosen_legal = legal.gather(1, a[:, None]).squeeze(1)
        pm = (pf[:, C_POL] > 0) & chosen_legal
        logp_b = pf[:, C_LOGP]
        chosen = logp_beh.gather(1, a[:, None]).squeeze(1)
        if cfg.taste.on:
            chosen = taste_log_prob(chosen, pf[:, C_TKEEP], pf[:, C_TADD])
        logp = torch.where(pm, chosen, logp_b)
        log_ratio = logp - logp_b
        ratio = log_ratio.exp()
        adv = pf[:, C_ADV]
        if cfg.decoupled_clip_anchor:
            logp_ref = torch.where(pm, pf[:, C_REF], logp_b)
            log_ratio_ref = logp - logp_ref
            ratio_ref = log_ratio_ref.exp()
            anchor_weight = (logp_ref - logp_b).exp()
            surr = torch.minimum(ratio * adv, anchor_weight * ratio_ref.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
        else:
            log_ratio_ref, ratio_ref = log_ratio, ratio
            surr = torch.minimum(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
        pmf = pm.float()
        loss_pi = -(surr * pmf).sum() / den[D_POL]
        if self.distil:  # kickstart phase 0: no PPO surrogate, entropy bonus or uniform barrier
            loss_pi = logits.new_zeros(())
        probs = torch.where(legal, logp_all.exp(), 0.0)
        ent = -(probs * logp_all).sum(-1)
        ent_n = ent / pf[:, C_LOGL].clamp_min(1e-6)
        base_logp = legal_uniform_log_probs(logits) if self._exploration is not None else logp_all
        base_ent = -(torch.where(legal, base_logp.exp(), 0.0) * base_logp).sum(-1)
        base_ent_n = base_ent / pf[:, C_LOGL].clamp_min(1e-6)
        bonus = ((ent_n if cfg.ent_normalize else ent) * pf[:, C_ENT] * pmf).sum() / den[D_POL]
        if self.distil:
            bonus = logits.new_zeros(())
        uniform_ce = logits.new_zeros(())
        if self._uniform_kl is not None and not self.distil:
            uniform_ce = (uniform_cross_entropy(logits, reference_probs)
                          * self._uniform_kl[pi[:, 1]] * pmf).sum() / den[D_POL]
        v = out["value"]
        ret = pf[:, C_RET]
        if self._hl_gauss:
            assert self._val_edges is not None
            p_tgt = hl_gauss_probs(ret, self._val_edges, self._val_vmin, self._val_vmax, self._val_sigma)
            ce = -(p_tgt * F.log_softmax(out["value_logits"], dim=-1)).sum(-1)
            loss_v = (ce * pf[:, C_VALID]).sum() / den[D_ALL]
        else:
            err2 = (v - ret).square()
            if cfg.value_clip is not None:
                v_old = pf[:, C_VOLD]
                v_c = v_old + (v - v_old).clamp(-cfg.value_clip, cfg.value_clip)
                err2 = torch.maximum(err2, (v_c - ret).square())
            loss_v = 0.5 * (err2 * pf[:, C_VALID]).sum() / den[D_ALL] * den[D_VSCALE]
        wm, hm = pf[:, C_WINM], pf[:, C_HPM]
        loss_win = (F.binary_cross_entropy_with_logits(out["pwin_logit"], pf[:, C_WIN], reduction="none")
                    * wm).sum() / den[D_WIN]
        loss_fl = ((out["floors_pred"] - pf[:, C_FLO]).square() * wm).sum() / den[D_FLO]
        loss_hp = ((out["hp_pred"] - pf[:, C_HP]).square() * hm).sum() / den[D_HP]
        loss = (loss_pi - bonus + uniform_ce + cfg.vf_coef * loss_v + cfg.aux_win * loss_win
                + cfg.aux_floors * loss_fl + cfg.aux_hp * loss_hp)
        if self.teacher is not None:
            assert self.kick_acc is not None
            loss = loss + kick_loss(out, pmf, pf[:, C_VALID], den[D_POL], den[D_ALL], self.kick_coef[0],
                                    self.kick_coef[1], self.kick_acc,
                                    None if self._smooth is None else self._smooth[pi[:, 1]])
        if self.distil:
            assert self.kick_acc is not None
            distil_sums(v, ret, pf[:, C_VALID], loss, self.kick_acc)
        if self.rules_coef is not None:  # spire.rules: the removal teach term
            assert self.rules_acc is not None
            loss = loss + teach_loss(out, pf[:, C_VALID], den[D_ALL], self.rules_coef, den[D_E0], self.rules_acc)
        if self.critic:  # the tower's losses (its parameters only: every tensor it reads is detached)
            assert self.critic_acc is not None and cfg.critic is not None
            loss = loss + critic_loss(out, pf[:, C_RWIN], pf[:, C_RPROG], pf[:, C_RSURV], pf[:, C_VALID], den[D_ALL],
                                      cfg.critic.coef, self.critic_acc)
        loss.backward()
        with torch.no_grad():
            kl = ((ratio_ref - 1) - log_ratio_ref) * pmf
            if cfg.decoupled_clip_anchor:
                kl = kl * anchor_weight
            if self.distil:  # no behaviour ratio: zeros for the PPO terms
                kl, ratio_ref = torch.zeros_like(pmf), torch.ones_like(pmf)
            ks = kl.sum()
            self.mb_kl += ks
            cm = pf[:, C_COMBAT] * pmf
            acc.add_combat(-(torch.where(legal, logp_beh.exp(), 0.0) * logp_beh).sum(-1) * cm, base_ent * cm, cm)
            acc.add_micro(den[D_E0], loss_pi, loss_v, loss_win, loss_fl, loss_hp, bonus, ks,
                          ((ratio_ref - 1).abs() > cfg.clip).float().mul(pmf).sum(),
                          (ratio_ref - 1).abs().mul(pmf).sum(),
                          pmf.sum(), pi[:, 1], ent * pmf, ent_n * pmf, pmf, ((~chosen_legal) & valid).sum(), 0,
                          uniform_ce, base_ent * pmf, base_ent_n * pmf)

    def _capture(self, r: _Res) -> None:
        """Capture the micro-step graph of ``r``: warm-up on a side stream (cuBLAS / autograd workspaces),
        then capture into the shared pool. The warm-up really runs, so gradients, diagnostics and the KL
        sum are saved first and restored afterwards; the capture itself runs nothing. Holds
        :data:`spire.model.CAPTURE_LOCK` (an actor thread of the same process may capture too)."""
        with CAPTURE_LOCK:
            self._capture_locked(r)

    def _capture_locked(self, r: _Res) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if any(p.grad is None for p in params):
            raise RuntimeError("MicroStepper: gradient buffers must exist before a capture")
        ptrs = [p.grad.data_ptr() for p in params]  # type: ignore[union-attr]
        if self._side is None:
            self._side = torch.cuda.Stream(self.device)
        if not self.captures or self.compiled:
            # the first capture warms up (cuBLAS handles, autograd / attention workspaces) by really running
            # two micro-steps on the capture stream (a compiled forward: before every capture, so that any
            # recompilation for the new shape happens outside the capture): save and restore what they accumulate
            grads0 = [p.grad.clone() for p in params]  # type: ignore[union-attr]
            acc0 = self.acc.snapshot()
            kl0 = self.mb_kl.clone()
            kick0 = None if self.kick_acc is None else self.kick_acc.clone()
            rules0 = None if self.rules_acc is None else self.rules_acc.clone()
            crit0 = None if self.critic_acc is None else self.critic_acc.clone()
            cur = torch.cuda.current_stream(self.device)
            self._side.wait_stream(cur)
            # The first graph primes cuBLAS/autograd. Later shapes of the
            # deterministic public model need one eager execution to finish
            # any shape-specific compilation before capture. Keep both calls
            # for stochastic or externally instrumented models.
            deterministic = self.copy_stream is not None and self.captures > 0 and self.replay_is_pure_backward()
            with torch.cuda.stream(self._side):
                for _ in range(1 if deterministic else 2):
                    self._body(r)
            cur.wait_stream(self._side)
            with torch.no_grad():
                for p, g0 in zip(params, grads0):
                    p.grad.copy_(g0)  # type: ignore[union-attr]
                self.acc.restore(acc0)
                self.mb_kl.copy_(kl0)
                if kick0 is not None:
                    self.kick_acc.copy_(kick0)  # type: ignore[union-attr]
                if rules0 is not None:
                    self.rules_acc.copy_(rules0)  # type: ignore[union-attr]
                if crit0 is not None:
                    self.critic_acc.copy_(crit0)  # type: ignore[union-attr]
            del grads0, acc0, kl0, kick0, rules0, crit0
            if self.compiled:
                # the eager warm-up's blocks (side-stream cache) are not usable by the graph pool: without this the
                # compiled learner holds the largest eager micro-step twice (measured: OOM at a 4.5 GB cap)
                if self.copy_stream is None or self._capture_needs_cache():
                    torch.cuda.empty_cache()
        g = torch.cuda.CUDAGraph()
        capture_graph(g, lambda: self._body(r), pool=self.pool, stream=self._side)
        if [p.grad.data_ptr() for p in params] != ptrs:  # type: ignore[union-attr]
            raise RuntimeError("MicroStepper: a capture replaced gradient buffers (accumulation not in place)")
        r.graphs[self.tempered] = g
        self.captures += 1

    def _capture_needs_cache(self) -> bool:
        # When both cards have ample headroom, releasing the allocator cache on
        # each shape change stalls work already enqueued on the other card. Keep
        # it if another entire reserved working set fits, including a process cap.
        free, total = torch.cuda.mem_get_info(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        limit = torch.cuda.get_per_process_memory_fraction(self.device) * total
        return min(free, limit - reserved) < reserved

    def stats(self) -> dict[str, Any]:
        return {"graphs": self.graphs, "captures": self.captures, "graph_shapes": len(self._res),
                "micro_steps": self.micro_steps, "padded_samples": self.padded_samples}


def critic_loss(out: dict[str, Tensor], ret_win: Tensor, ret_prog: Tensor, ret_surv: Tensor, valid: Tensor,
                den_all: Tensor | float, coef: tuple[float, float, float], acc: Tensor) -> Tensor:
    """L_crit of notes/critic/contract.md 4.3 on one micro-batch: coef_win * BCE(win_logit_c, RET_WIN) + coef_prog *
    0.5 (V_PROG - RET_PROG)^2 + coef_surv * 0.5 (V_SURV - RET_SURV)^2, each a sum over the valid samples divided by
    the minibatch's sample count (means over the minibatch once its micro-batches are summed). ``acc`` (3,) collects
    the three unweighted terms (detached)."""
    vc = out["vc"]
    l_win = (F.binary_cross_entropy_with_logits(out["win_logit_c"], ret_win, reduction="none") * valid).sum() / den_all
    l_prog = (0.5 * (vc[:, 1] - ret_prog).square() * valid).sum() / den_all
    l_surv = (0.5 * (vc[:, 2] - ret_surv).square() * valid).sum() / den_all
    with torch.no_grad():
        acc += torch.stack([l_win, l_prog, l_surv]).detach()
    return coef[0] * l_win + coef[1] * l_prog + coef[2] * l_surv


def ema_weight(decay: float, step: int) -> float:
    """EMA weight of the online parameters at EMA step ``step`` (0-based) with warm-up:
    ``decay_t = min(decay, (1 + t) / (10 + t))`` (py-review 5.1)."""
    return 1.0 - min(decay, (1.0 + step) / (10.0 + step))

