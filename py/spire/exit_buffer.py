"""Compact shard I/O and sliding ring buffer for the Async ExIt Search Teacher (py-stage-p13).

Shards are stored as plain CPU tensor/primitive dicts loadable under ``torch.load(..., weights_only=True)``
and replaced atomically via temp-file + ``os.replace``. Each record carries a single model-ready
:class:`~spire.tokenize.Sample` (compact int8/int16 arrays, ~2.5 KB/record) plus its search-verified
canonical target action, legal mask, and diagnostic metadata.
"""
from __future__ import annotations

import collections
import dataclasses
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .tokenize import Sample

SHARD_FORMAT_VERSION = 1


def _retry_fs(fn: Any, attempts: int = 50, delay: float = 0.01) -> Any:
    """Windows refuses to replace/open a file another process holds open; retry briefly."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


@dataclasses.dataclass(frozen=True)
class ExitRecord:
    """One verified high-gain tactical correction emitted by :mod:`spire.exit_teacher`."""

    sample: Sample
    target_action: int
    legal_mask: np.ndarray | None  # (n_actions,) bool, or None if unavailable
    cand_rows: tuple[int, ...]
    greedy_action: int
    gain: float
    gain_s2: float
    t_stat: float
    dv_after: float
    p_old: float
    p_group_old: float
    teacher_version: int
    stage: str
    verb: str
    char: str = "SILENT"
    ascension: int = 10
    ts: float = 0.0
    cand_scores: tuple[float, ...] = ()  # p13.1: optional per-candidate search scores matching cand_rows
    cand_p_old: tuple[float, ...] = ()   # p13.1: optional per-candidate prior probs matching cand_rows


def record_to_dict(rec: ExitRecord) -> dict[str, Any]:
    """Serialize an :class:`ExitRecord` into ``weights_only=True``-safe CPU tensors & primitives."""
    s = rec.sample
    n_act = s.n_actions
    assert 0 <= rec.target_action < n_act, (rec.target_action, n_act)
    d: dict[str, Any] = {
        "types": torch.from_numpy(np.ascontiguousarray(s.types, dtype=np.int8)).clone(),
        "raw": torch.from_numpy(np.ascontiguousarray(s.raw, dtype=np.int16)).clone(),
        "rel": torch.from_numpy(np.ascontiguousarray(s.rel, dtype=np.int16)).clone(),
        "edges": torch.from_numpy(np.ascontiguousarray(s.edges, dtype=np.int16)).clone(),
        "cond": torch.from_numpy(np.ascontiguousarray(s.cond, dtype=np.int16)).clone(),
        "counts": [int(c) for c in s.counts],
        "history": torch.from_numpy(np.ascontiguousarray(s.history, dtype=np.int16)).clone(),
        "target_action": int(rec.target_action),
        "cand_rows": [int(r) for r in rec.cand_rows],
        "greedy_action": int(rec.greedy_action),
        "gain": float(rec.gain),
        "gain_s2": float(rec.gain_s2),
        "t_stat": float(rec.t_stat),
        "dv_after": float(rec.dv_after),
        "p_old": float(rec.p_old),
        "p_group_old": float(rec.p_group_old),
        "teacher_version": int(rec.teacher_version),
        "stage": str(rec.stage),
        "verb": str(rec.verb),
        "char": str(rec.char),
        "ascension": int(rec.ascension),
        "ts": float(rec.ts),
    }
    if rec.legal_mask is not None:
        lm = np.ascontiguousarray(rec.legal_mask, dtype=np.bool_)
        assert lm.shape == (n_act,), (lm.shape, n_act)
        d["legal_mask"] = torch.from_numpy(lm).clone()
    if rec.cand_scores:
        assert len(rec.cand_scores) == len(rec.cand_rows), (len(rec.cand_scores), len(rec.cand_rows))
        d["cand_scores"] = [float(x) for x in rec.cand_scores]
    if rec.cand_p_old:
        assert len(rec.cand_p_old) == len(rec.cand_rows), (len(rec.cand_p_old), len(rec.cand_rows))
        d["cand_p_old"] = [float(x) for x in rec.cand_p_old]
    return d


def record_from_dict(d: dict[str, Any]) -> ExitRecord:
    """Deserialize and strictly validate one :class:`ExitRecord` dict."""
    if not isinstance(d, dict):
        raise ValueError("record is not a dict")
    types = d["types"].detach().cpu().numpy().astype(np.int8, copy=False)
    raw = d["raw"].detach().cpu().numpy().astype(np.int16, copy=False)
    rel = d["rel"].detach().cpu().numpy().astype(np.int16, copy=False)
    edges = d["edges"].detach().cpu().numpy().astype(np.int16, copy=False)
    cond = d["cond"].detach().cpu().numpy().astype(np.int16, copy=False)
    history = d["history"].detach().cpu().numpy().astype(np.int16, copy=False)
    counts_list = d["counts"]
    if len(counts_list) != 4:
        raise ValueError(f"bad counts length: {counts_list!r}")
    counts = (int(counts_list[0]), int(counts_list[1]), int(counts_list[2]), int(counts_list[3]))
    n_rows = sum(counts)
    if types.ndim != 1 or types.shape[0] != n_rows:
        raise ValueError(f"types shape {types.shape} != n_rows {n_rows}")
    if raw.ndim != 2 or raw.shape[0] != n_rows:
        raise ValueError(f"raw shape {raw.shape} != n_rows {n_rows}")
    if rel.ndim != 2 or rel.shape != (n_rows, 3):
        raise ValueError(f"rel shape {rel.shape} != ({n_rows}, 3)")
    if edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError(f"edges shape {edges.shape}")
    if cond.ndim != 1 or cond.shape[0] != 6:
        raise ValueError(f"cond shape {cond.shape}")
    n_act = counts[3]
    target_action = int(d["target_action"])
    if not (0 <= target_action < n_act):
        raise ValueError(f"target_action {target_action} out of bounds [0, {n_act})")
    legal_mask: np.ndarray | None = None
    if "legal_mask" in d and d["legal_mask"] is not None:
        lm = d["legal_mask"].detach().cpu().numpy().astype(np.bool_, copy=False)
        if lm.shape != (n_act,):
            raise ValueError(f"legal_mask shape {lm.shape} != ({n_act},)")
        if not bool(lm[target_action]):
            raise ValueError(f"target_action {target_action} is illegal in stored legal_mask")
        legal_mask = lm
    gain = float(d["gain"])
    if not math.isfinite(gain):
        raise ValueError(f"non-finite gain {gain}")
    cand_rows = tuple(int(r) for r in d.get("cand_rows", ()))
    cand_scores = tuple(float(x) for x in d.get("cand_scores", ()))
    cand_p_old = tuple(float(x) for x in d.get("cand_p_old", ()))
    if cand_scores:
        if len(cand_scores) != len(cand_rows) or not all(math.isfinite(x) for x in cand_scores):
            raise ValueError(f"bad cand_scores {cand_scores!r} for cand_rows {cand_rows!r}")
    if cand_p_old:
        if len(cand_p_old) != len(cand_rows) or not all(math.isfinite(x) and x >= 0.0 for x in cand_p_old):
            raise ValueError(f"bad cand_p_old {cand_p_old!r} for cand_rows {cand_rows!r}")
    sample = Sample(types, raw, rel, edges, cond, counts, history)
    return ExitRecord(
        sample=sample,
        target_action=target_action,
        legal_mask=legal_mask,
        cand_rows=cand_rows,
        greedy_action=int(d.get("greedy_action", -1)),
        gain=gain,
        gain_s2=float(d.get("gain_s2", 0.0)),
        t_stat=float(d.get("t_stat", 0.0)),
        dv_after=float(d.get("dv_after", 0.0)),
        p_old=float(d.get("p_old", 0.0)),
        p_group_old=float(d.get("p_group_old", d.get("p_old", 0.0))),
        teacher_version=int(d["teacher_version"]),
        stage=str(d.get("stage", "")),
        verb=str(d.get("verb", "")),
        char=str(d.get("char", "SILENT")),
        ascension=int(d.get("ascension", 10)),
        ts=float(d.get("ts", 0.0)),
        cand_scores=cand_scores,
        cand_p_old=cand_p_old,
    )


def save_exit_shard(
    exit_dir: str | Path,
    worker_id: str | int,
    seq: int,
    records: Sequence[ExitRecord],
    *,
    max_shards: int = 500,
    vocab_hash: int = 0,
    catalog_hash: int = 0,
) -> Path:
    """Atomically write ``shard-<worker>-<seq:06d>.pt`` and prune oldest shards beyond ``max_shards``."""
    assert records, "refuse to write empty ExIt shard"
    out_dir = Path(exit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"shard-{worker_id}-{seq:06d}.pt"
    dst = out_dir / name
    tmp = out_dir / f".{name}.{os.getpid()}.tmp"
    payload = {
        "format": SHARD_FORMAT_VERSION,
        "worker": str(worker_id),
        "seq": int(seq),
        "vocab_hash": int(vocab_hash),
        "catalog_hash": int(catalog_hash),
        "ts": float(time.time()),
        "records": [record_to_dict(r) for r in records],
    }
    torch.save(payload, tmp)
    _retry_fs(lambda: os.replace(tmp, dst))
    if max_shards > 0:
        prune_shards(out_dir, max_shards)
    return dst


def prune_shards(exit_dir: Path, max_shards: int) -> int:
    """Keep at most ``max_shards`` newest ``shard-*.pt`` files in ``exit_dir``."""
    files = list(exit_dir.glob("shard-*.pt"))
    if len(files) <= max_shards:
        return 0
    keyed: list[tuple[float, str, Path]] = []
    for p in files:
        try:
            st = p.stat()
            keyed.append((st.st_mtime, p.name, p))
        except OSError:
            continue
    keyed.sort()
    removed = 0
    for _, _, old in keyed[: max(0, len(keyed) - max_shards)]:
        try:
            old.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def load_exit_shard(path: str | Path, *, expected_vocab_hash: int | None = None) -> list[ExitRecord]:
    """Load and validate an ExIt shard with ``weights_only=True``."""
    p = Path(path)
    obj = _retry_fs(lambda: torch.load(p, map_location="cpu", weights_only=True))
    if not isinstance(obj, dict) or obj.get("format") != SHARD_FORMAT_VERSION:
        raise ValueError(f"{p}: bad ExIt shard format {obj.get('format') if isinstance(obj, dict) else type(obj)}")
    if expected_vocab_hash is not None and int(obj.get("vocab_hash", 0)) not in (0, int(expected_vocab_hash)):
        raise ValueError(f"{p}: vocab_hash {obj.get('vocab_hash')} != expected {expected_vocab_hash}")
    raw_recs = obj.get("records")
    if not isinstance(raw_recs, list):
        raise ValueError(f"{p}: records is not a list")
    return [record_from_dict(r) for r in raw_recs]


class ExitBuffer:
    """Learner-side sliding ring buffer ingesting atomic teacher shards with age and capacity eviction.

    Optional p13.1 prioritization (all default OFF = bit-identical to p13):
      * ``recency_halflife`` > 0: exponential recency decay ``2 ** (-age / halflife)`` when sampling.
      * ``legacy_weight`` < 1.0 or ``retire_legacy_at`` > 0: classifies cross-lineage bank records
        (``teacher_version > current_update + 100`` at ingest) as legacy, aging them from ingest update
        instead of letting negative ``current_update - teacher_version`` make them immortal, down-weighting
        them relative to live on-policy records, and auto-evicting them once ``n_live >= retire_legacy_at``.
    """

    def __init__(
        self,
        exit_dir: str | Path,
        *,
        capacity: int = 8192,
        max_age_updates: int = 2000,
        min_gain: float = 0.025,
        expected_vocab_hash: int | None = None,
        recency_halflife: int = 0,
        legacy_weight: float = 1.0,
        retire_legacy_at: int = 0,
    ) -> None:
        assert capacity >= 1 and max_age_updates >= 0 and min_gain >= 0.0
        assert isinstance(recency_halflife, int) and not isinstance(recency_halflife, bool) and recency_halflife >= 0
        assert math.isfinite(legacy_weight) and 0.0 <= legacy_weight <= 1.0
        assert isinstance(retire_legacy_at, int) and not isinstance(retire_legacy_at, bool) and retire_legacy_at >= 0
        self.exit_dir = Path(exit_dir)
        self.capacity = int(capacity)
        self.max_age_updates = int(max_age_updates)
        self.min_gain = float(min_gain)
        self.expected_vocab_hash = expected_vocab_hash
        self.recency_halflife = int(recency_halflife)
        self.legacy_weight = float(legacy_weight)
        self.retire_legacy_at = int(retire_legacy_at)
        self._legacy_mode = (self.legacy_weight < 1.0) or (self.retire_legacy_at > 0)
        self._seen: set[str] = set()
        self._buf: collections.deque[ExitRecord] = collections.deque()
        self._meta: collections.deque[tuple[bool, int]] = collections.deque()  # (is_legacy, ingested_update)

    def __len__(self) -> int:
        return len(self._buf)

    def counts(self) -> tuple[int, int]:
        """Return ``(n_live, n_legacy)`` counts currently in buffer."""
        n_leg = sum(1 for leg, _ in self._meta if leg)
        return len(self._buf) - n_leg, n_leg

    def _record_age(self, rec: ExitRecord, is_legacy: bool, ingested_at: int, current_update: int) -> int:
        if is_legacy:
            return current_update - ingested_at
        return current_update - rec.teacher_version

    def poll_and_evict(self, current_update: int) -> tuple[int, int, int]:
        """Ingest unseen ``shard-*.pt`` files and evict stale/overflow entries.

        Returns ``(ingested, evicted, corrupt_skipped)``. Never raises if ``exit_dir`` is missing,
        empty, or contains partially-written/corrupt files.
        """
        ingested = 0
        evicted = 0
        corrupt = 0

        if self.exit_dir.is_dir():
            # Sort by (mtime, name) so oldest shards are ingested first; ignore .tmp files.
            candidates: list[tuple[float, str, Path]] = []
            for p in self.exit_dir.glob("shard-*.pt"):
                if p.name in self._seen:
                    continue
                try:
                    candidates.append((p.stat().st_mtime, p.name, p))
                except OSError:
                    continue
            candidates.sort()
            for _, name, p in candidates:
                self._seen.add(name)
                try:
                    recs = load_exit_shard(p, expected_vocab_hash=self.expected_vocab_hash)
                except FileNotFoundError:
                    continue
                except Exception:  # noqa: BLE001 - resilient to corrupt/truncated shards
                    corrupt += 1
                    continue
                for r in recs:
                    if r.gain < self.min_gain:
                        continue
                    is_leg = self._legacy_mode and (r.teacher_version > current_update + 100)
                    if self._record_age(r, is_leg, current_update, current_update) > self.max_age_updates:
                        evicted += 1
                        continue
                    self._buf.append(r)
                    self._meta.append((is_leg, current_update))
                    ingested += 1

        # Evict stale entries by age (and retire legacy bank once n_live >= retire_legacy_at)
        if self._buf:
            n_live_valid = sum(
                1
                for r, (leg, ing) in zip(self._buf, self._meta)
                if (not leg) and (self._record_age(r, False, ing, current_update) <= self.max_age_updates)
            )
            retire_leg = self.retire_legacy_at > 0 and n_live_valid >= self.retire_legacy_at
            kept_buf: collections.deque[ExitRecord] = collections.deque()
            kept_meta: collections.deque[tuple[bool, int]] = collections.deque()
            for r, (leg, ing) in zip(self._buf, self._meta):
                if (leg and retire_leg) or self._record_age(r, leg, ing, current_update) > self.max_age_updates:
                    evicted += 1
                else:
                    kept_buf.append(r)
                    kept_meta.append((leg, ing))
            self._buf = kept_buf
            self._meta = kept_meta

        # Enforce capacity (evict legacy FIFO first when legacy_mode is active, then live FIFO)
        if len(self._buf) > self.capacity:
            overflow = len(self._buf) - self.capacity
            if self._legacy_mode:
                drop_idx: set[int] = set()
                for i, (leg, _) in enumerate(self._meta):
                    if leg and len(drop_idx) < overflow:
                        drop_idx.add(i)
                for i in range(len(self._buf)):
                    if len(drop_idx) >= overflow:
                        break
                    if i not in drop_idx:
                        drop_idx.add(i)
                self._buf = collections.deque(r for i, r in enumerate(self._buf) if i not in drop_idx)
                self._meta = collections.deque(m for i, m in enumerate(self._meta) if i not in drop_idx)
                evicted += len(drop_idx)
            else:
                while len(self._buf) > self.capacity:
                    self._buf.popleft()
                    self._meta.popleft()
                    evicted += 1

        return ingested, evicted, corrupt

    def sample_with_flags(
        self, batch_size: int, rng: np.random.Generator, *, current_update: int | None = None
    ) -> tuple[list[ExitRecord], list[bool]]:
        """Sample up to ``batch_size`` records (uniform when v2 knobs are off; recency/live prioritized when on)."""
        n = len(self._buf)
        if n == 0 or batch_size <= 0:
            return [], []
        items = list(self._buf)
        metas = list(self._meta)
        if self.recency_halflife == 0 and not self._legacy_mode:
            k = min(batch_size, n)
            idx = rng.choice(n, size=k, replace=False)
            return [items[int(i)] for i in idx], [metas[int(i)][0] for i in idx]

        n_live = sum(1 for leg, _ in metas if not leg)
        taper = (max(0.0, 1.0 - n_live / self.retire_legacy_at) if self.retire_legacy_at > 0 else 1.0)
        eff_leg = self.legacy_weight * taper
        w = np.empty(n, dtype=np.float64)
        inv_hl = (math.log(2.0) / self.recency_halflife) if self.recency_halflife > 0 else 0.0
        for i, (r, (leg, ing)) in enumerate(zip(items, metas)):
            base = eff_leg if leg else 1.0
            if base <= 0.0:
                w[i] = 0.0
            elif inv_hl > 0.0 and current_update is not None:
                age = max(0, self._record_age(r, leg, ing, current_update))
                w[i] = base * math.exp(-inv_hl * age)
            else:
                w[i] = base
        nz = int(np.count_nonzero(w > 0.0))
        if nz == 0:
            return [], []
        k = min(batch_size, nz)
        p = w / float(w.sum())
        idx = rng.choice(n, size=k, replace=False, p=p)
        return [items[int(i)] for i in idx], [metas[int(i)][0] for i in idx]

    def sample(self, batch_size: int, rng: np.random.Generator, *, current_update: int | None = None) -> list[ExitRecord]:
        """Sample ``min(batch_size, len(self))`` records without replacement."""
        return self.sample_with_flags(batch_size, rng, current_update=current_update)[0]
