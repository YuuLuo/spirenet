"""``learner.replica_device`` (notes/next/dual-spec.md): the learner's model replicated across one or more extra cards.

One process, one submission thread per replica CUDA device. The learner keeps ONE optimizer, ONE EMA and the
parameters of record on its own device A; :class:`Replica` holds copies of the model on N replica devices
``workers[0..N-1]``, each with its own :class:`~spire.trainstep.MicroStepper` (own static buffers, graphs, pool,
minibatch scalars, compiled forward, device accumulators). Per minibatch the learner splits the micro-batches of
:func:`~spire.trainstep.bucket_micros` across all ``K = 1 + N`` cards (:meth:`Replica.assign`), adds each replica's
gradients and minibatch KL to A's in fixed deterministic worker order (:meth:`Replica.combine`), runs clip / AdamW /
EMA on A and broadcasts A's updated parameters to every replica (:meth:`Replica.sync`). Each replica's parameters and
gradients are views of one flat buffer each, so each direction is one cross-device copy per replica with event-ordered
reuse of A's flat send buffer (no host synchronisation). Replicas are never saved."""
from __future__ import annotations

import copy
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .kickstart import N_ACC
from .rules import N_RULES_ACC
from .trainstep import MicroStepper, batch_cap, pad_batch

if TYPE_CHECKING:
    from .learner import PPOConfig, _Accum
    from .tokenize import SampleBlock, Tokenizer

SPLITS = ("balance", "a", "b")
_ALIGN_EL = 128  # float32 elements = 512 bytes


def normalize_replica_devices(
    rd: str | torch.device | Sequence[str | torch.device] | None,
) -> tuple[str, ...]:
    """Normalize ``replica_device`` (empty, single ``"cuda:1"``, comma-separated string, or list/tuple of devices)
    into a canonical tuple of non-empty device strings. Invalid types or empty entries raise ``AssertionError``."""
    if rd is None:
        return ()
    if isinstance(rd, torch.device):
        return (str(rd),)
    if isinstance(rd, str):
        s = rd.strip()
        if not s:
            return ()
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            assert all(parts), f"learner.replica_device {rd!r}: empty device entry"
            return tuple(parts)
        return (s,)
    if isinstance(rd, Sequence) and not isinstance(rd, (bytes, bytearray)):
        out: list[str] = []
        for item in rd:
            if isinstance(item, torch.device):
                s = str(item)
            else:
                assert isinstance(item, str), f"learner.replica_device: expected str or torch.device, got {item!r}"
                s = item.strip()
            assert s and "," not in s, f"learner.replica_device: invalid device entry {item!r}"
            out.append(s)
        return tuple(out)
    raise AssertionError(f"learner.replica_device: expected str or sequence of str, got {rd!r}")


def stepper_sums(acc: "_Accum") -> list[Tensor]:
    """The device sums of an :class:`~spire.learner._Accum` that the micro-steps write (``add_micro`` /
    ``add_combat``); the learner-side sums (gradient norms, skipped steps, the KL brake's scale) are A's alone."""
    return [*acc.loss.values(), acc.kl, acc.clipfrac, acc.ratio_dev, acc.n_pol, acc.ratio_dev0, acc.n_pol0, acc.clip0,
            acc.ent_k, acc.ent_nk, acc.ent_base_k, acc.ent_base_nk, acc.n_k, acc.illegal_chosen, acc.ent_beh_c,
            acc.ent_base_c, acc.n_c]


class _ReplicaWorker:
    """State of one replica card: its model copy, flat param/grad buffers, A-side recv buffers, stepper and thread."""

    def __init__(self, index: int, model_a: nn.Module, params_a: list[nn.Parameter], names_a: list[str],
                 offs: list[int], sizes: list[int], total: int, a_dev: torch.device, tokenizer: "Tokenizer",
                 cfg: "PPOConfig", device: torch.device, acc: "_Accum", *, graphs: bool, autocast: bool,
                 compile_model: bool, rules_on: bool) -> None:
        self.index = index
        self.device = device
        self.model = copy.deepcopy(model_a).to(device)
        names_b = [n for n, p in self.model.named_parameters() if p.requires_grad]
        assert names_a == names_b
        params_b = [p for p in self.model.parameters() if p.requires_grad]
        # Replica: parameters and gradients are views of one flat buffer each (the graphs hold the gradients' addresses)
        self.pflat = torch.zeros(total, device=device)
        self.gflat = torch.zeros(total, device=device)
        with torch.no_grad():
            for p, o, n in zip(params_b, offs, sizes):
                self.pflat[o:o + n].copy_(p.reshape(-1))
                p.data = self.pflat[o:o + n].view_as(p)
                p.grad = self.gflat[o:o + n].view_as(p)
        self.params_b = params_b
        # A-side staging for this replica's gradients and minibatch KL sum
        self.recv = torch.empty(total, device=a_dev)
        self.recv_views = [self.recv[o:o + n].view_as(p) for p, o, n in zip(params_a, offs, sizes)]
        self.kl_recv = torch.zeros((), device=a_dev)
        self.submitter = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ppo-replica-{index}")
        self.acc = acc
        self.rules_coef = torch.zeros((), device=device) if rules_on else None
        self.rules_acc = torch.zeros(N_RULES_ACC, device=device) if rules_on else None
        self.kick_acc = torch.zeros(N_ACC, device=device)
        with torch.cuda.device(device):
            self.stepper = MicroStepper(self.model, tokenizer, cfg, device, acc, graphs=graphs, autocast=autocast,
                                        compile_model=compile_model, teacher=None, kick_acc=self.kick_acc,
                                        rules_coef=self.rules_coef, rules_acc=self.rules_acc)
            self._sync_ev = torch.cuda.Event()
        self._sync_ready = False


class Replica:
    """One or more model replicas on extra CUDA cards, their steppers and the cross-device copy buffers.

    ``params_a``: A's trained parameters (the optimizer's, in A's model order); every worker holds the same names.
    When configured with a single replica device, single-card attributes (``device``, ``model``, ``gflat``,
    ``stepper``, ...) delegate to ``workers[0]`` for full backwards compatibility."""

    # test-only switch: "balance" (LPT across all K cards), "a" (all on A), "b" (all on replicas, A idle),
    # or an int card index 0..K-1 (0 = A, 1..N = workers[idx - 1])
    split: str | int = "balance"

    def __init__(self, model: nn.Module, tokenizer: "Tokenizer", cfg: "PPOConfig",
                 devices: torch.device | Sequence[torch.device], accs: "_Accum | Sequence[_Accum]", *,
                 graphs: bool, autocast: bool, compile_model: bool, rules_on: bool) -> None:
        dev_list = [devices] if isinstance(devices, torch.device) else list(devices)
        acc_list = [accs] if not isinstance(accs, Sequence) else list(accs)
        assert len(dev_list) >= 1, "Replica requires at least one replica device"
        assert len(dev_list) == len(acc_list), "Replica devices and accumulators count mismatch"
        a_dev = next(model.parameters()).device
        seen: set[int] = set()
        for dev in dev_list:
            assert dev.type == "cuda" and dev.index is not None, f"replica device {dev}: cuda:N"
            assert a_dev != dev, f"the replica device {dev} is the learner's"
            assert dev.index not in seen, f"duplicate replica device {dev}"
            seen.add(dev.index)
        self.a_device = a_dev
        names_a = [n for n, p in model.named_parameters() if p.requires_grad]
        self.params_a = [p for p in model.parameters() if p.requires_grad]
        assert all(p.dtype == torch.float32 and p.is_contiguous() for p in self.params_a), "fp32 contiguous parameters"
        sizes = [p.numel() for p in self.params_a]
        # every view starts at a multiple of 512 bytes, as the caching allocator's blocks do: kernels (cuBLAS) choose
        # their algorithm by the pointers' alignment, so a misaligned view would sum in another order than on A
        offs = [0]
        for n in sizes:
            offs.append(offs[-1] + -(-n // _ALIGN_EL) * _ALIGN_EL)
        total = offs[-1]
        # A: shared flat buffer of parameters to broadcast to every replica
        self.send = torch.zeros(total, device=a_dev)
        self.send_views = [self.send[o:o + n].view_as(p) for p, o, n in zip(self.params_a, offs, sizes)]
        self._cfg = cfg
        self.workers: list[_ReplicaWorker] = [
            _ReplicaWorker(idx, model, self.params_a, names_a, offs, sizes, total, a_dev, tokenizer, cfg, dev, acc,
                           graphs=graphs, autocast=autocast, compile_model=compile_model, rules_on=rules_on)
            for idx, (dev, acc) in enumerate(zip(dev_list, acc_list))
        ]
        self.devices: tuple[torch.device, ...] = tuple(w.device for w in self.workers)
        self.n_cards = 1 + len(self.workers)
        self.micro = [0] * self.n_cards
        self.load = [0] * self.n_cards

    # ------------------------------------------------------------------ single-replica compatibility properties
    @property
    def _single(self) -> _ReplicaWorker:
        assert len(self.workers) == 1, f"single-replica attribute used with {len(self.workers)} replicas; use .workers"
        return self.workers[0]

    @property
    def device(self) -> torch.device:
        return self._single.device

    @property
    def model(self) -> nn.Module:
        return self._single.model

    @property
    def pflat(self) -> Tensor:
        return self._single.pflat

    @property
    def gflat(self) -> Tensor:
        return self._single.gflat

    @property
    def params_b(self) -> list[nn.Parameter]:
        return self._single.params_b

    @property
    def recv(self) -> Tensor:
        return self._single.recv

    @property
    def recv_views(self) -> list[Tensor]:
        return self._single.recv_views

    @property
    def kl_recv(self) -> Tensor:
        return self._single.kl_recv

    @property
    def submitter(self) -> ThreadPoolExecutor:
        return self._single.submitter

    @property
    def acc(self) -> "_Accum":
        return self._single.acc

    @property
    def rules_coef(self) -> Tensor | None:
        return self._single.rules_coef

    @property
    def rules_acc(self) -> Tensor | None:
        return self._single.rules_acc

    @property
    def stepper(self) -> MicroStepper:
        return self._single.stepper

    # ------------------------------------------------------------------ per update / minibatch
    def assign(self, micros: list[tuple[int, np.ndarray]], *, record: bool = True) -> list[int]:
        """Card index of every micro-batch (0 = primary A, 1..N = ``workers[side - 1]``): greedy in the list's order
        to the lightest card (load = bucket tokens x padded batch), ties to the lowest index. The test switch
        ``split`` accepts ``"balance"``, ``"a"``, ``"b"`` (replicas only, balanced across ``1..N``) or an explicit
        card index ``0..K-1``."""
        sp = self.split
        k = self.n_cards
        if isinstance(sp, int) and not isinstance(sp, bool):
            assert 0 <= sp < k, f"replica.split {sp}: expected 0 <= idx < {k}"
        else:
            assert sp in SPLITS, sp
        out: list[int] = []
        load = [0] * k
        for bucket, idx in micros:
            w = bucket * pad_batch(len(idx), batch_cap(self._cfg, bucket))
            if sp == "balance":
                side = min(range(k), key=load.__getitem__)
            elif sp == "a":
                side = 0
            elif sp == "b":
                side = min(range(1, k), key=load.__getitem__)
            else:
                side = int(sp)
            load[side] += w
            if record:
                self.micro[side] += 1
                self.load[side] += w
            out.append(side)
        return out

    def set_den(self, *args: Any) -> None:
        for w in self.workers:
            with torch.cuda.device(w.device):
                w.stepper.set_den(*args)
                w.stepper.mb_kl.zero_()

    def ensure_copy_streams(self) -> None:
        for w in self.workers:
            if w.stepper.copy_stream is None:
                w.stepper.copy_stream = torch.cuda.Stream(device=w.device)

    def save_micro_states(self) -> list[tuple[int, int]]:
        saved: list[tuple[int, int]] = []
        for w in self.workers:
            with torch.cuda.device(w.device):
                saved.append(w.stepper.save_micro_state())
        return saved

    def restore_micro_states(self, saved: Sequence[tuple[int, int]]) -> None:
        assert len(saved) == len(self.workers)
        for w, counters in zip(self.workers, saved):
            with torch.cuda.device(w.device):
                w.stepper.restore_micro_state(counters)

    @torch.no_grad()
    def zero_grads(self) -> None:
        for w in self.workers:
            with torch.cuda.device(w.device):
                w.gflat.zero_()

    def run(self, blk: "SampleBlock", idx: np.ndarray, pf_rows: np.ndarray, pi_rows: np.ndarray, bucket: int,
            *, worker_idx: int = 0) -> None:
        w = self.workers[worker_idx]
        with torch.cuda.device(w.device):
            w.stepper.run(blk, idx, pf_rows, pi_rows, bucket)

    def enqueue(self, blk: "SampleBlock", micros: list[tuple[int, np.ndarray]], pf: np.ndarray,
                pi: np.ndarray) -> Future[None]:
        """Single-replica helper: pack and launch worker 0's subsequence while the learner runs on A."""
        w = self._single
        stream = torch.cuda.current_stream(w.device)

        def submit() -> None:
            with torch.cuda.device(w.device), torch.cuda.stream(stream):
                for bucket, idx in micros:
                    w.stepper.run(blk, idx, pf[idx], pi[idx], bucket)

        return w.submitter.submit(submit)

    def enqueue_all(self, blk: "SampleBlock", micros: list[tuple[int, np.ndarray]], sides: Sequence[int],
                    pf: np.ndarray, pi: np.ndarray) -> list[Future[None]]:
        """Pack and launch every replica's subsequence in parallel across per-device submission threads."""
        per_worker: list[list[tuple[int, np.ndarray]]] = [[] for _ in self.workers]
        for m, side in zip(micros, sides):
            if side > 0:
                per_worker[side - 1].append(m)
        futures: list[Future[None]] = []
        for w, items in zip(self.workers, per_worker):
            if not items:
                continue
            stream = torch.cuda.current_stream(w.device)

            def submit(worker: _ReplicaWorker = w, batch_items: list[tuple[int, np.ndarray]] = items,
                       st: torch.cuda.Stream = stream) -> None:
                with torch.cuda.device(worker.device), torch.cuda.stream(st):
                    for bucket, idx in batch_items:
                        worker.stepper.run(blk, idx, pf[idx], pi[idx], bucket)

            futures.append(w.submitter.submit(submit))
        return futures

    @torch.no_grad()
    def combine(self, grads_a: list[Tensor], mb_kl_a: Tensor) -> Tensor:
        """Every replica's gradients added to A's (``grads_a``: the ``.grad`` of :attr:`params_a`) in fixed
        deterministic worker order; returns A's minibatch KL sum plus every replica's (a new device scalar on A)."""
        mb_kl = mb_kl_a
        for w in self.workers:
            w.recv.copy_(w.gflat, non_blocking=True)
            torch._foreach_add_(grads_a, w.recv_views)
            w.kl_recv.copy_(w.stepper.mb_kl, non_blocking=True)
            mb_kl = mb_kl + w.kl_recv
        return mb_kl

    @torch.no_grad()
    def sync(self) -> None:
        """After A's optimizer step: every replica's parameters = A's, gradients zeroed in place.

        A's stream waits on every replica's prior broadcast event before refilling ``self.send`` so back-to-back
        syncs (e.g. PPO last minibatch followed by an ExIt auxiliary step) can never overwrite ``self.send`` while a
        replica's peer copy is still reading it."""
        cur_a = torch.cuda.current_stream(self.a_device)
        for w in self.workers:
            if w._sync_ready:
                cur_a.wait_event(w._sync_ev)
        torch._foreach_copy_(self.send_views, self.params_a)
        for w in self.workers:
            with torch.cuda.device(w.device):
                w.pflat.copy_(self.send, non_blocking=True)
                w._sync_ev.record(torch.cuda.current_stream(w.device))
                w._sync_ready = True
                w.gflat.zero_()

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        """Every parameter and buffer of ``model`` (A's) into every replica (construction and after a load)."""
        for w in self.workers:
            with torch.cuda.device(w.device):
                for pb, pa in zip(w.model.parameters(), model.parameters()):
                    pb.copy_(pa)
                for bb, ba in zip(w.model.buffers(), model.buffers()):
                    bb.copy_(ba)
                w.gflat.zero_()

    def begin(self, tempered: bool, rules_coef: float | None) -> None:
        """Start of an update: replicas' sums zeroed, the tempered flag and the removal coefficient written."""
        for w in self.workers:
            with torch.cuda.device(w.device):
                w.model.train()
                w.acc.reset()
                w.stepper.tempered = tempered
                if rules_coef is not None:
                    assert w.rules_coef is not None and w.rules_acc is not None
                    w.rules_coef.fill_(rules_coef)
                    w.rules_acc.zero_()
        self.micro = [0] * self.n_cards
        self.load = [0] * self.n_cards

    @torch.no_grad()
    def merge_into(self, acc: "_Accum", rules_acc: Tensor | None) -> None:
        """End of an update (after the synchronize): replicas' micro-step sums (and rules sums) added to A's in
        fixed deterministic worker order."""
        for w in self.workers:
            for ta, tb in zip(stepper_sums(acc), stepper_sums(w.acc)):
                ta.add_(tb.to(ta.device))
            if rules_acc is not None:
                assert w.rules_acc is not None
                rules_acc.add_(w.rules_acc.to(rules_acc.device))

    def row(self) -> dict[str, Any]:
        tot = max(1, sum(self.load))
        if len(self.workers) == 1:
            w = self.workers[0]
            return {"device": str(w.device), "share_a": self.load[0] / tot, "micro_a": self.micro[0],
                    "micro_b": self.micro[1], "stepper": w.stepper.stats()}
        return {
            "devices": [str(w.device) for w in self.workers],
            "share_a": self.load[0] / tot,
            "shares": [ld / tot for ld in self.load],
            "micro_a": self.micro[0],
            "micro_b": sum(self.micro[1:]),
            "micros": list(self.micro),
            "steppers": [w.stepper.stats() for w in self.workers],
        }

    def close(self) -> None:
        for w in self.workers:
            w.submitter.shutdown(wait=True)
