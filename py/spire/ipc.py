"""Segment transfer between processes: binary segment blobs in a shared-memory byte ring.

Split out of ``train.py`` (process topology) to keep that module small.

* :func:`encode_segment` / :func:`decode_segment` turn a :class:`~spire.rollout.Segment`
  (its ``Sample`` records, per-step arrays and metadata) into one contiguous blob and back:
  a small JSON header followed by a handful of concatenated arrays (the samples' ``types``,
  ``raw``, ``rel``, ``edges``, ``cond`` and counts are concatenated across the segment, so a
  256-step segment is ~20 arrays, not thousands of pickled objects). Decoding makes zero-copy
  read-only views into the blob.
* :class:`ShmRing` is a single-producer / single-consumer byte ring in one
  ``multiprocessing.shared_memory`` block: the producer copies a blob in, advances ``head`` and
  sends ``(start, length)`` over a pipe; the consumer copies the blob out and advances ``tail``.
  Both counters are monotonic byte positions in the block header, each written by one side
  only. A blob never wraps (the producer skips to the start of the block instead). A dead
  producer leaves at most an unannounced gap that the consumer's next ``tail`` update reclaims.
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Protocol

import numpy as np

from .rollout import EpisodeMeta, Segment, SegmentEnd
from .tokenize import Sample, SampleBlock

_U32 = struct.Struct("<I")
_ALIGN = 8
_HDR = 64  # ring header bytes: int64 head at 0, int64 tail at 8

_STEP_ARRAYS = ("actions", "logp", "values", "kinds", "n_legal", "policy_version", "rewards", "combat_id", "win",
                "win_mask", "floors_left", "hp_post", "hp_mask")
_ADVICE_ARRAYS = ("advice_row", "advice_hp")  # only in blobs of segments that carry an advice record
_JUDGE_ARRAYS = ("advice_rule", "advice_sign")  # only in blobs of segments that carry a habit-rule record
_TASTE_ARRAYS = ("t_keep", "t_add")  # only in blobs of segments that carry a taste record (spire.taste)
_TEMP_ARRAYS = ("beh_temp",)  # only in blobs of segments that carry a behaviour temperature (spire.temperature)
# only in blobs of segments that carry the critic record (notes/critic/contract.md 3); rewards_c is filled at release
_CRITIC_ARRAYS = ("values_c", "crit", "bootstrap_c", "rewards_c")
_PROG_ARRAYS = ("rewards_prog",)  # p12: floor + elite + boss rewards (omitted when None)


class Conn(Protocol):
    """One end of a ``multiprocessing.Pipe`` (``Connection`` / Windows ``PipeConnection``)."""

    def send(self, obj: Any) -> None: ...

    def recv(self) -> Any: ...

    def poll(self, timeout: float | None = 0.0) -> bool: ...


def _pad(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


# =========================================================================== codec


def encode_segment(seg: Segment) -> tuple[bytes, list[np.ndarray]]:
    """Header bytes (length-prefixed JSON, padded) and the arrays that follow it in the blob."""
    has_boot = seg.bootstrap_sample is not None
    blk = seg.samples
    if isinstance(blk, SampleBlock):  # already concatenated when the segment was sealed
        arrays: dict[str, np.ndarray] = {
            "s_counts": blk.counts.astype(np.int32), "s_edges_n": np.diff(blk.edge0).astype(np.int32),
            "s_types": blk.types, "s_raw": blk.raw, "s_rel": blk.rel, "s_edges": blk.edges, "s_cond": blk.cond,
            "s_history": blk.history}
        if has_boot:
            b: Sample = seg.bootstrap_sample
            arrays = {
                "s_counts": np.concatenate([arrays["s_counts"], np.array([b.counts], np.int32)]),
                "s_edges_n": np.append(arrays["s_edges_n"], np.int32(b.edges.shape[0])),
                "s_types": np.concatenate([blk.types, b.types]), "s_raw": np.concatenate([blk.raw, b.raw]),
                "s_rel": np.concatenate([blk.rel, b.rel]), "s_edges": np.concatenate([blk.edges, b.edges]),
                "s_cond": np.concatenate([blk.cond, b.cond[None, :]]),
                "s_history": np.concatenate([blk.history, b.history[None, :]])}
    else:
        samples: list[Sample] = list(seg.samples)
        if has_boot:
            samples.append(seg.bootstrap_sample)
        counts = np.array([s.counts for s in samples], np.int32).reshape(-1, 4)
        n_edges = np.fromiter((s.edges.shape[0] for s in samples), np.int32, len(samples))
        arrays = {
            "s_counts": counts, "s_edges_n": n_edges,
            "s_types": np.concatenate([s.types for s in samples]) if samples else np.zeros(0, np.int8),
            "s_raw": np.concatenate([s.raw for s in samples]) if samples else np.zeros((0, 0), np.int16),
            "s_rel": np.concatenate([s.rel for s in samples]) if samples else np.zeros((0, 3), np.int16),
            "s_edges": np.concatenate([s.edges for s in samples]) if samples else np.zeros((0, 3), np.int16),
            "s_cond": np.stack([s.cond for s in samples]) if samples else np.zeros((0, 6), np.int16),
            "s_history": np.stack([s.history for s in samples]) if samples else np.zeros((0, 0), np.int16),
        }
    for name in _STEP_ARRAYS:
        arrays[name] = np.ascontiguousarray(getattr(seg, name))
    if seg.advice_row is not None:
        for name in _ADVICE_ARRAYS:
            arrays[name] = np.ascontiguousarray(getattr(seg, name))
    if seg.advice_rule is not None:
        for name in _JUDGE_ARRAYS:
            arrays[name] = np.ascontiguousarray(getattr(seg, name))
    if seg.t_keep is not None:
        for name in _TASTE_ARRAYS:
            arrays[name] = np.ascontiguousarray(getattr(seg, name))
    if seg.beh_temp is not None:
        for name in _TEMP_ARRAYS:
            arrays[name] = np.ascontiguousarray(getattr(seg, name))
    if seg.values_c is not None:
        assert seg.rewards_c is not None, "a critic segment travels released (with its component rewards)"
        for name in _CRITIC_ARRAYS:
            arrays[name] = np.ascontiguousarray(getattr(seg, name))
    if seg.rewards_prog is not None:
        for name in _PROG_ARRAYS:
            arrays[name] = np.ascontiguousarray(getattr(seg, name))
    m = seg.meta
    manifest = []
    off = 0
    for name, a in arrays.items():
        manifest.append([name, a.dtype.str, list(a.shape), off])
        off += _pad(a.nbytes)
    meta = [m.env_id, m.episode_uid, list(m.characters), m.ascension, m.seed_id]
    if m.fork_group:
        meta.extend([m.fork_group, m.fork_start])
    head = {"meta": meta,
            "start": seg.start, "end": int(seg.end), "boot_value": seg.bootstrap_value, "has_boot": has_boot,
            "t": [seg.t_first, seg.t_last], "arrays": manifest}
    hj = json.dumps(head, separators=(",", ":")).encode()
    header = _U32.pack(len(hj)) + hj
    header += b"\0" * (_pad(len(header)) - len(header))
    return header, list(arrays.values())


def blob_size(header: bytes, arrays: list[np.ndarray]) -> int:
    return len(header) + sum(_pad(a.nbytes) for a in arrays)


def write_blob(dest: memoryview, header: bytes, arrays: list[np.ndarray]) -> int:
    """Copy a blob into ``dest`` (at least :func:`blob_size` bytes); returns its size."""
    out = np.frombuffer(dest, np.uint8)
    out[: len(header)] = np.frombuffer(header, np.uint8)
    off = len(header)
    for a in arrays:
        n = a.nbytes
        if n:
            out[off : off + n] = a.reshape(-1).view(np.uint8)
        off += _pad(n)
    return off


def pack_segment(seg: Segment) -> bytes:
    header, arrays = encode_segment(seg)
    buf = bytearray(blob_size(header, arrays))
    write_blob(memoryview(buf), header, arrays)
    return bytes(buf)


def decode_segment(blob: bytes | bytearray | memoryview) -> Segment:
    """Rebuild a :class:`Segment` whose samples / arrays are views into ``blob``."""
    mv = memoryview(blob)
    (hn,) = _U32.unpack_from(mv, 0)
    head = json.loads(bytes(mv[4 : 4 + hn]))
    base = _pad(4 + hn)
    arr: dict[str, np.ndarray] = {}
    for name, dt, shape, off in head["arrays"]:
        dtype = np.dtype(dt)
        count = int(np.prod(shape)) if shape else 1
        arr[name] = np.frombuffer(mv, dtype, count, base + off).reshape(shape)
    counts = arr["s_counts"]
    rows = counts.sum(1, dtype=np.int64)
    r_end = np.cumsum(rows)
    e_end = np.cumsum(arr["s_edges_n"], dtype=np.int64)
    types, raw, rel, edges, cond = arr["s_types"], arr["s_raw"], arr["s_rel"], arr["s_edges"], arr["s_cond"]
    # the segment's samples as one SampleBlock over the blob (the learner's fast collation reads its arrays);
    # the bootstrap sample, if any, is the last record
    n = len(counts) - (1 if head["has_boot"] else 0)
    samples: Any
    if n > 0:
        R, E = int(r_end[n - 1]), int(e_end[n - 1])
        samples = SampleBlock(types[:R], raw[:R], rel[:R], edges[:E], cond[:n], counts[:n].astype(np.int64),
                              arr["s_history"][:n])
        samples.set_edge_counts(arr["s_edges_n"][:n])
    else:
        R = E = 0
        samples = []
    boot = None
    if head["has_boot"]:
        c = counts[n]
        boot = Sample(types[R:], raw[R:], rel[R:], edges[E:], cond[n], (int(c[0]), int(c[1]), int(c[2]), int(c[3])),
                      arr["s_history"][n])
    meta_fields = head["meta"]
    env_id, uid, chars, asc, seed_id = meta_fields[:5]
    fork_group = int(meta_fields[5]) if len(meta_fields) > 5 else 0
    fork_start = int(meta_fields[6]) if len(meta_fields) > 6 else 0
    seg = Segment(meta=EpisodeMeta(env_id, uid, tuple(chars), asc, seed_id, fork_group, fork_start),
                  start=head["start"], samples=samples,
                  actions=arr["actions"], logp=arr["logp"], values=arr["values"], kinds=arr["kinds"],
                  n_legal=arr["n_legal"], policy_version=arr["policy_version"], end=SegmentEnd(head["end"]),
                  bootstrap_value=head["boot_value"], bootstrap_sample=boot,
                  t_first=float(head.get("t", (0.0, 0.0))[0]), t_last=float(head.get("t", (0.0, 0.0))[1]))
    for name in _STEP_ARRAYS[6:]:
        setattr(seg, name, arr[name])
    if "advice_row" in arr:
        for name in _ADVICE_ARRAYS:
            setattr(seg, name, arr[name])
    if "advice_rule" in arr:
        for name in _JUDGE_ARRAYS:
            setattr(seg, name, arr[name])
    if "t_keep" in arr:
        for name in _TASTE_ARRAYS:
            setattr(seg, name, arr[name])
    if "beh_temp" in arr:
        for name in _TEMP_ARRAYS:
            setattr(seg, name, arr[name])
    if "values_c" in arr:
        for name in _CRITIC_ARRAYS:
            setattr(seg, name, arr[name])
    if "rewards_prog" in arr:
        for name in _PROG_ARRAYS:
            setattr(seg, name, arr[name])
    return seg


# =========================================================================== shared-memory ring


@dataclass(frozen=True)
class RingSpec:
    """Picklable description of a ring (shared-memory block name and data capacity)."""

    name: str
    capacity: int


@dataclass
class RingStats:
    segments: int = 0
    bytes: int = 0
    encode_ns: int = 0  # producer: encoding + copying into shared memory
    stall_ns: int = 0  # producer: waiting for free space
    dropped: int = 0  # producer: gave up after the put timeout (counted, never silent)
    decode_ns: int = 0  # consumer: copy out + decode

    def take(self) -> dict[str, int]:
        out = dict(self.__dict__)
        for k in out:
            setattr(self, k, 0)
        return out


class ShmRing:
    """One end of a shared-memory segment ring (see module docstring).

    The creator (:meth:`create`) owns the block; producer and consumer attach with
    :meth:`attach` in their own processes, each with its end of a one-way pipe.
    """

    def __init__(self, shm: SharedMemory, capacity: int, conn: Conn | None, owner: bool) -> None:
        self.shm = shm
        self.capacity = capacity
        self.conn = conn
        self.owner = owner
        buf = shm.buf
        assert buf is not None
        self.ctr = np.ndarray((2,), np.int64, buffer=buf[:16])
        self.data = buf[_HDR : _HDR + capacity]
        self.stats = RingStats()

    @classmethod
    def create(cls, capacity: int) -> tuple["ShmRing", RingSpec]:
        shm = SharedMemory(create=True, size=_HDR + capacity)
        ring = cls(shm, capacity, None, owner=True)
        ring.ctr[:] = 0
        return ring, RingSpec(shm.name, capacity)

    @classmethod
    def attach(cls, spec: RingSpec, conn: Conn) -> "ShmRing":
        return cls(SharedMemory(name=spec.name), spec.capacity, conn, owner=False)

    # ------------------------------------------------------------------ producer
    def put(self, seg: Segment, timeout: float = 30.0) -> bool:
        """Copy ``seg`` into the ring and announce it; False if no space within ``timeout``."""
        assert self.conn is not None
        t0 = time.perf_counter_ns()
        header, arrays = encode_segment(seg)
        n = _pad(blob_size(header, arrays))
        cap = self.capacity
        if n > cap:
            raise ValueError(f"segment blob of {n} bytes exceeds the ring capacity {cap}")
        head = int(self.ctr[0])
        pos = head % cap
        start = head + (cap - pos if pos + n > cap else 0)
        t_wait = time.perf_counter_ns()
        deadline = time.monotonic() + timeout
        while start + n - int(self.ctr[1]) > cap:
            if time.monotonic() > deadline:
                self.stats.dropped += 1
                self.stats.stall_ns += time.perf_counter_ns() - t_wait
                return False
            time.sleep(0.001)
        t_stall = time.perf_counter_ns() - t_wait
        p = start % cap
        write_blob(self.data[p : p + n], header, arrays)
        self.ctr[0] = start + n
        self.conn.send((start, n))
        st = self.stats
        st.segments += 1
        st.bytes += n
        st.stall_ns += t_stall
        st.encode_ns += time.perf_counter_ns() - t0 - t_stall
        return True

    # ------------------------------------------------------------------ consumer
    def get(self, announcement: tuple[int, int]) -> Segment:
        """Copy out and decode the blob of one ``(start, length)`` announcement; frees its space."""
        t0 = time.perf_counter_ns()
        start, n = announcement
        p = start % self.capacity
        # copied with numpy (it releases the GIL for the copy; bytes(memoryview) holds it) into a fresh read-only
        # array: the decoded arrays are read-only views exactly as with a bytes blob
        blob = np.empty(n, np.uint8)
        np.copyto(blob, np.frombuffer(self.data, np.uint8, n, p))
        blob.flags.writeable = False
        self.ctr[1] = start + n
        seg = decode_segment(blob)
        self.stats.segments += 1
        self.stats.bytes += n
        self.stats.decode_ns += time.perf_counter_ns() - t0
        return seg

    def close(self) -> None:
        del self.ctr
        self.data.release()
        self.shm.close()
        if self.owner:
            self.shm.unlink()


def take_ring_stats(rings: list[ShmRing]) -> dict[str, Any]:
    """Merge and reset the counters of several ring ends."""
    tot: dict[str, int] = {}
    for r in rings:
        for k, v in r.stats.take().items():
            tot[k] = tot.get(k, 0) + v
    return tot
