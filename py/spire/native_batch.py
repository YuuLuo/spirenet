"""In-process ctypes adapter for Rust Cut #42+ ``libfast_native_runtime.so`` Parallel NativeBatch.

Provides Option B in-process ``Operation::ResampleFork`` (op=9) + multi-threaded ``step_batch``
(``FAST_NATIVE_BATCH_THREADS``) with zero ``libc::fork()``, zero AF_UNIX sockets, and strict
fair-play ``MODE_RESAMPLE`` enforcement.
"""
from __future__ import annotations

import ctypes as C
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .gym import MODE_RESAMPLE, same_obs
from .host import HostError
from .proto import Frame, ResetCmd, encode_reset, parse_frame

(
    RESET,
    IMPORT,
    STEP,
    CLONE,
    RELEASE,
    COMMAND,
    RESTORE,
    FORK,
    RESAMPLE_FORK,
    CONSTRUCT_COMBAT,
    CLONE_WITH_DECK,
) = range(1, 12)


def _encode_spec(spec: object) -> bytes:
    if isinstance(spec, dict):
        return json.dumps(spec, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if isinstance(spec, str):
        return spec.encode("utf-8")
    if isinstance(spec, (bytes, bytearray, memoryview)):
        return bytes(spec)
    raise TypeError(f"expected dict, str, or bytes spec, got {type(spec).__name__}")


class NativeTransportError(HostError):
    def __init__(self, status: int, message: str):
        super().__init__(f"native transport {status}: {message}")
        self.status = status


@dataclass(frozen=True)
class Request:
    operation: int
    handle: int = 0
    step_id: int = 0
    action_index: int = 0
    data: bytes = b""


@dataclass(frozen=True)
class BatchResult:
    status: int
    handle: int
    frames: tuple[bytes, ...]
    quit: bool = False
    error: str = ""

    def checked(self) -> "BatchResult":
        if self.status < 0:
            raise NativeTransportError(self.status, self.error)
        return self


class _Request(C.Structure):
    _fields_ = [
        ("operation", C.c_int32),
        ("handle", C.c_uint64),
        ("step_id", C.c_int32),
        ("action_index", C.c_int32),
        ("data", C.POINTER(C.c_uint8)),
        ("data_len", C.c_size_t),
    ]


class _Result(C.Structure):
    _fields_ = [
        ("status", C.c_int32),
        ("handle", C.c_uint64),
        ("quit", C.c_int32),
        ("data", C.POINTER(C.c_uint8)),
        ("data_len", C.c_size_t),
        ("error", C.POINTER(C.c_uint8)),
        ("error_len", C.c_size_t),
    ]


def unpack_frames(data: bytes) -> tuple[bytes, ...]:
    frames: list[bytes] = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < 4:
            raise ValueError("native result has a truncated frame prefix")
        (length,) = struct.unpack_from("<I", data, offset)
        offset += 4
        end = offset + length
        if end > len(data):
            raise ValueError("native result has a truncated frame payload")
        frames.append(data[offset:end])
        offset = end
    return tuple(frames)


def _integer(value: int, low: int, high: int, name: str) -> int:
    if not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} out of range: {value!r}")
    return value


class NativeBatch:
    """One native context wrapping ``libfast_native_runtime.so`` (parallelized internally via Rust BatchPool)."""

    def __init__(self, library: str | Path, *, threads: int | None = None):
        if threads is not None and threads > 0:
            os.environ["FAST_NATIVE_BATCH_THREADS"] = str(int(threads))
        self._dll = C.CDLL(str(Path(library).resolve()))
        self._dll.ft_abi_version.argtypes = []
        self._dll.ft_abi_version.restype = C.c_uint32
        if self._dll.ft_abi_version() != 1:
            raise NativeTransportError(-3, "unsupported native batch ABI")
        self._dll.ft_create.argtypes = []
        self._dll.ft_create.restype = C.c_void_p
        self._dll.ft_destroy.argtypes = [C.c_void_p]
        self._dll.ft_destroy.restype = None
        self._dll.ft_execute.argtypes = [
            C.c_void_p,
            C.POINTER(_Request),
            C.c_size_t,
            C.POINTER(C.POINTER(_Result)),
        ]
        self._dll.ft_execute.restype = C.c_int32
        self._dll.ft_free_results.argtypes = [C.POINTER(_Result), C.c_size_t]
        self._dll.ft_free_results.restype = None
        self._context = self._dll.ft_create()
        if not self._context:
            raise NativeTransportError(-2, "native context allocation failed")

    def close(self) -> None:
        if self._context:
            self._dll.ft_destroy(self._context)
            self._context = None

    def __enter__(self) -> "NativeBatch":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def execute(self, requests: Iterable[Request]) -> list[BatchResult]:
        if not self._context:
            raise NativeTransportError(-3, "context is closed")
        req_tuple = tuple(requests)
        if not req_tuple:
            return []
        buffers = []
        native = (_Request * len(req_tuple))()
        for i, request in enumerate(req_tuple):
            data = bytes(request.data)
            buf = (C.c_uint8 * len(data)).from_buffer_copy(data)
            buffers.append(buf)
            native[i] = _Request(
                _integer(request.operation, -(2**31), 2**31 - 1, "operation"),
                _integer(request.handle, 0, 2**64 - 1, "handle"),
                _integer(request.step_id, -(2**31), 2**31 - 1, "step_id"),
                _integer(request.action_index, -(2**31), 2**31 - 1, "action_index"),
                buf,
                len(data),
            )
        output = C.POINTER(_Result)()
        status = self._dll.ft_execute(self._context, native, len(native), C.byref(output))
        if status:
            raise NativeTransportError(status, "invalid native call arguments")
        try:
            results: list[BatchResult] = []
            for i in range(len(req_tuple)):
                item = output[i]
                payload = C.string_at(item.data, item.data_len) if item.data_len else b""
                error = C.string_at(item.error, item.error_len).decode("utf-8", errors="replace") if item.error_len else ""
                results.append(BatchResult(item.status, item.handle, unpack_frames(payload), bool(item.quit), error))
            return results
        finally:
            self._dll.ft_free_results(output, len(req_tuple))

    def reset(self, payload: bytes) -> BatchResult:
        return self.execute([Request(RESET, data=payload)])[0].checked()

    def step(self, handle: int, action_index: int, expected_step_id: int) -> BatchResult:
        return self.execute([Request(STEP, handle, expected_step_id, action_index)])[0].checked()

    def step_batch(self, actions: Iterable[tuple[int, int, int]]) -> list[BatchResult]:
        return self.execute(Request(STEP, handle, step_id, action) for handle, action, step_id in actions)

    def resample_fork_batch(self, parent_handle: int, seeds: Iterable[int], *, flags: int = 0) -> list[BatchResult]:
        """Strictly fair-play ``MODE_RESAMPLE`` in-process fork batch (``op=9``)."""
        return self.execute(
            Request(RESAMPLE_FORK, parent_handle, MODE_RESAMPLE, flags, struct.pack("<Q", int(s) & ((1 << 64) - 1)))
            for s in seeds
        )

    def construct_combat(self, spec: dict | str | bytes) -> BatchResult:
        """Construct an independent combat world from a UTF-8 JSON/dict spec (Cut #51, op=10)."""
        return self.execute([Request(CONSTRUCT_COMBAT, data=_encode_spec(spec))])[0].checked()

    def construct_combat_batch(self, specs: Iterable[dict | str | bytes]) -> list[BatchResult]:
        """Construct combat worlds in parallel (op=10); per-request errors remain unchecked."""
        return self.execute(Request(CONSTRUCT_COMBAT, data=_encode_spec(spec)) for spec in specs)

    def clone_with_deck(self, handle: int, spec: dict | str | bytes) -> BatchResult:
        """Clone handle and swap deck/re-seed combat while preserving parent handle (Cut #51, op=11)."""
        return self.execute([Request(CLONE_WITH_DECK, handle, data=_encode_spec(spec))])[0].checked()

    def clone_with_deck_batch(
        self, items: Iterable[tuple[int, dict | str | bytes]]
    ) -> list[BatchResult]:
        """Input tuples are (handle, spec) for op=11; per-request errors remain unchecked."""
        return self.execute(
            Request(CLONE_WITH_DECK, handle, data=_encode_spec(spec)) for handle, spec in items
        )

    def release_batch(self, handles: Iterable[int]) -> list[BatchResult]:
        hs = [h for h in handles if h]
        if not hs:
            return []
        return self.execute(Request(RELEASE, h) for h in hs)


class NativeBatchHost:
    """Drop-in parent+child search host backed by in-process ``NativeBatch`` (Option B)."""

    def __init__(
        self,
        so_path: str | Path,
        *,
        threads: int = 8,
        expected_hashes: tuple[int, int] | None = None,
    ) -> None:
        self.so_path = Path(so_path)
        self.threads = int(threads)
        self.expected_hashes = expected_hashes
        self.batch = NativeBatch(self.so_path, threads=self.threads)
        self.handle: int | None = None
        self.step_id: int = 0
        self.pid: int = os.getpid()

    def reset_with(self, cmd: ResetCmd) -> Frame:
        if self.handle is not None:
            try:
                self.batch.release_batch([self.handle])
            except Exception:  # noqa: BLE001
                pass
            self.handle = None
        wire = encode_reset(cmd)
        payload = wire[4:] if len(wire) == 68 else wire
        res = self.batch.reset(payload)
        if not res.frames:
            raise NativeTransportError(-2, "native reset returned no frames")
        fr = parse_frame(res.frames[-1])
        if self.expected_hashes is not None:
            vh, ch = self.expected_hashes
            if (fr.vocab_hash, fr.catalog_hash) != (vh, ch):
                self.batch.release_batch([res.handle])
                raise HostError(
                    f"native vocab/catalog hash mismatch: got {(fr.vocab_hash, fr.catalog_hash)}, expected {(vh, ch)}"
                )
        self.handle = res.handle
        self.step_id = fr.step_id
        return fr

    def step(self, action_row: int) -> Frame:
        if self.handle is None:
            raise HostError("NativeBatchHost.step called without active handle")
        fr = self.step_handle(self.handle, action_row, self.step_id)
        self.step_id = fr.step_id
        return fr

    def step_handle(self, handle: int, action_row: int, expected_step_id: int) -> Frame:
        res = self.batch.step(int(handle), int(action_row), int(expected_step_id))
        if not res.frames:
            raise NativeTransportError(-2, f"native step_handle returned no frames: {res.error}")
        return parse_frame(res.frames[-1])

    def fork_handle_resample(
        self,
        parent_handle: int,
        parent_frame: Frame,
        seeds: list[int],
        *,
        flags: int = 0,
    ) -> list[tuple[int, Frame]]:
        """Strictly fair-play ``MODE_RESAMPLE`` fork from any active handle (``op=9``)."""
        results = self.batch.resample_fork_batch(int(parent_handle), seeds, flags=flags)
        created = [r.handle for r in results if r.status == 0 and r.handle]
        if any(r.status != 0 or not r.handle for r in results):
            self.release_handles(created)
            errs = [f"{r.status}:{r.error}" for r in results if r.status != 0 or not r.handle][:2]
            raise NativeTransportError(-2, f"resample_fork failed (flags={flags}): {errs}")
        out: list[tuple[int, Frame]] = []
        for r in results:
            if r.frames:
                if not same_obs(r.frames[0], parent_frame.payload):
                    self.release_handles(created)
                    raise NativeTransportError(-2, "resample_fork child initial OBS differed from parent")
                out.append((r.handle, parse_frame(r.frames[0])))
            else:
                out.append((r.handle, parent_frame))
        return out

    def fork_resample(self, parent_frame: Frame, seeds: list[int], *, flags: int = 0) -> list[tuple[int, Frame]]:
        if self.handle is None:
            raise HostError("NativeBatchHost.fork_resample called without active handle")
        return self.fork_handle_resample(self.handle, parent_frame, seeds, flags=flags)

    def step_batch(self, actions: Iterable[tuple[int, int, int]]) -> list[BatchResult]:
        return self.batch.step_batch(actions)

    def release_handles(self, handles: Iterable[int]) -> None:
        try:
            self.batch.release_batch(handles)
        except Exception:  # noqa: BLE001
            pass

    def close_handle(self, handle: int) -> None:
        self.release_handles([handle])

    def close_handles(self, handles: Iterable[int]) -> None:
        self.release_handles(handles)

    def close(self) -> None:
        if self.handle is not None:
            self.release_handles([self.handle])
            self.handle = None
        self.batch.close()
