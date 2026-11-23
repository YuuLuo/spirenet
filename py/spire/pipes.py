"""Named-pipe transport for host processes: one I/O completion port multiplexes every host of an
actor in the actor's own thread (Windows; specification: ``notes/transport-spec.md``).

Why: with stdin/stdout pipes every host needs a blocking reader thread, and every frame costs a
thread wake-up, a GIL hand-off and a queue round trip in the actor (measured ~12 us of actor CPU
per decision for recv + send at 32-128 hosts, before any parsing). Overlapped named pipes on one
completion port need no threads at all: ``PipeMux.poll`` dequeues completed reads and hands whole
messages to each connection's sink; writes complete synchronously into the pipe buffer
(``FILE_SKIP_COMPLETION_PORT_ON_SUCCESS``). Measured ~5 us per decision for recv + send.

The byte protocol is unchanged (``[u32 len][payload]`` both ways, schema 3.1). Per connection:

1. the actor creates the pipe ``\\\\.\\pipe\\spire-<actor pid>-<host name>-<128-bit hex nonce>`` (first
   instance, one instance, remote clients rejected) *before* starting the host, and passes the
   name on the host's command line (``--pipe <name>``);
2. the host opens it read/write and sends ``HELLO`` (:data:`HELLO_MAGIC`, transport version,
   its pid, 0) as its first message, then serves commands exactly as over stdio;
3. commands the actor sends before the host connected are queued and flushed on connect;
4. EOF / broken pipe on either side ends the connection: the actor treats it like a closed
   stdout (host crashed -> restart with a new pipe name), the host exits.

A ``PipeMux`` is single-threaded: all ``poll`` / ``send`` / ``open`` / ``close`` calls come from
the thread that owns it. Overlapped operations keep their buffers alive until the completion is
dequeued, so connections are only forgotten after their pending operations completed.

Two implementations of the overlapped I/O (connect, reads with length-prefix framing, writes,
completion dequeue) sit behind the same class: the C module ``spire._pipemux`` (native/pipemux.c,
default when built; several microseconds less interpreter time per frame) and the pure-Python one
below (``_overlapped`` objects). Pipe creation, HELLO validation, the outbox, sinks and the error
policy are shared. ``PipeMux(native=False)`` or ``SPIRE_PIPEMUX=python`` selects the Python one;
tests/test_pipes.py runs every test against both.
"""
from __future__ import annotations

import ctypes
import os
import secrets
import struct
import sys
import time
from typing import Any, Callable

from .proto import ProtocolError
from .schema import MAX_FRAME_BYTES_HIDDEN

if sys.platform == "win32":
    import _overlapped  # type: ignore[import-not-found]
    import _winapi

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    _k32.SetFileCompletionNotificationModes.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
    _k32.SetFileCompletionNotificationModes.restype = ctypes.c_int
    _k32.GetNamedPipeClientProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    _k32.GetNamedPipeClientProcessId.restype = ctypes.c_int

    class _Entry(ctypes.Structure):  # OVERLAPPED_ENTRY
        _fields_ = [("key", ctypes.c_size_t), ("overlapped", ctypes.c_void_p), ("internal", ctypes.c_size_t),
                    ("transferred", ctypes.c_uint32)]

    assert ctypes.sizeof(_Entry) == 32
    _GQCSEX = _k32.GetQueuedCompletionStatusEx
    _GQCSEX.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Entry), ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                        ctypes.c_ulong, ctypes.c_int]
    _GQCSEX.restype = ctypes.c_int

try:
    from . import _pipemux  # type: ignore[attr-defined]
except ImportError:  # not built (python -B native/build.py) or not Windows
    _pipemux = None
NATIVE_DEFAULT = os.environ.get("SPIRE_PIPEMUX", "native") != "python"
_NATIVE_VERSION = 1

HELLO_MAGIC = 0x4C485053  # b"SPHL" little-endian
TRANSPORT_VERSION = 1
HELLO = struct.Struct("<Iiii")  # magic, transport version, host pid, reserved (0)
_EV_CONNECTED, _EV_EOF, _EV_ERROR, _EV_TOO_LONG = 0, 1, 2, 3  # native event kinds (native/pipemux.c)
PIPE_PREFIX = "\\\\.\\pipe\\spire-"
PIPE_BUFFER = 1 << 18  # pipe in/out buffer sizes (advisory)
READ_BUFFER = 1 << 17  # initial per-connection receive buffer; grows for larger frames
DEQUEUE_BATCH = 128  # completions per GetQueuedCompletionStatusEx call

_LEN = struct.Struct("<I")
_SKIP_PORT_ON_SUCCESS = 0x1
_PIPE_REJECT_REMOTE_CLIENTS = 0x8
_ERROR_PIPE_CONNECTED = 535
_ERROR_BROKEN_PIPE = 109
_ERROR_NO_DATA = 232
_ERROR_OPERATION_ABORTED = 995

Sink = Callable[[object], None]
"""Receives each message payload (``bytes``) of a connection, or one terminal event: :data:`EOF`
(the host closed the pipe) or an exception (broken framing / I/O error)."""

EOF = object()


class PipeConn:
    """One host connection of a :class:`PipeMux` (create with :meth:`PipeMux.open`)."""

    __slots__ = ("mux", "key", "name", "handle", "sink", "state", "buf", "view", "have", "outbox", "client_pid",
                 "_read", "max_bytes", "messages", "bytes_in")

    def __init__(self, mux: "PipeMux", key: int, name: str, handle: int, sink: Sink, max_bytes: int) -> None:
        self.mux = mux
        self.key = key
        self.name = name
        self.handle = handle
        self.sink = sink
        self.state = "listening"  # -> "connected" -> "closed"
        self.buf = bytearray(READ_BUFFER)
        self.view = memoryview(self.buf)
        self.have = 0
        self.outbox: list[bytes] = []
        self.client_pid = -1  # from HELLO
        self._read: Any = None  # pending read Overlapped
        self.max_bytes = max_bytes
        self.messages = 0
        self.bytes_in = 0

    @property
    def connected(self) -> bool:
        return self.state == "connected"

    def send(self, message: bytes) -> None:
        """Write one complete wire message (length prefix included). Raises ``BrokenPipeError``
        when the connection is gone."""
        if self.state == "connected":
            self.mux._write(self, message)
        elif self.state == "listening":
            self.outbox.append(message)
        else:
            raise BrokenPipeError(f"{self.name}: connection closed")

    def close(self) -> None:
        """Close the pipe (pending operations are cancelled and reaped by the mux)."""
        self.mux._close(self)


class PipeMux:
    """Completion port shared by the pipe connections of one actor thread."""

    def __init__(self, max_bytes: int = MAX_FRAME_BYTES_HIDDEN, *, auto_close: bool = True,
                 native: bool | None = None) -> None:
        """``native``: use ``spire._pipemux`` (default: when built, unless ``SPIRE_PIPEMUX=python``);
        ``True`` requires it."""
        if sys.platform != "win32":
            raise OSError("PipeMux needs Windows named pipes; use the stdio transport")
        if native and _pipemux is None:
            raise ImportError("spire._pipemux is not built (python -B native/build.py)")
        if _pipemux is not None and getattr(_pipemux, "VERSION", 0) != _NATIVE_VERSION:
            raise RuntimeError("spire._pipemux was built from another source version; run python -B native/build.py")
        self.port = _overlapped.CreateIoCompletionPort(_overlapped.INVALID_HANDLE_VALUE, 0, 0, 1)
        self.max_bytes = max_bytes
        self.auto_close = auto_close  # close when the last user (host) releases it
        self._users = 0
        self._ops: dict[int, tuple[PipeConn, str, Any]] = {}  # OVERLAPPED address -> (conn, kind, ov)
        self._conns: dict[int, PipeConn] = {}  # completion key -> conn
        self._next_key = 1
        self._closed = False
        self._wait_py = 0  # Python implementation: ns in blocking or empty GetQueuedCompletionStatusEx calls
        self._writes_py = 0
        self._port_h = ctypes.c_void_p(self.port)
        self._entries = (_Entry * DEQUEUE_BATCH)()
        self._entries_mv = memoryview(self._entries).cast("B").cast("Q")  # 4 x u64 per OVERLAPPED_ENTRY
        self._removed = ctypes.c_ulong(0)
        use = NATIVE_DEFAULT if native is None else native
        self._nat: Any = _pipemux.Mux(self.port, DEQUEUE_BATCH) if use and _pipemux is not None else None

    @property
    def native(self) -> bool:
        """True when the C implementation (``spire._pipemux``) does the I/O."""
        return self._nat is not None

    @property
    def wait_ns(self) -> int:
        """ns spent in blocking or empty completion-port polls (cumulative)."""
        return int(self._nat.wait_ns) if self._nat is not None else self._wait_py

    @property
    def writes_pending(self) -> int:
        return int(self._nat.writes_pending) if self._nat is not None else self._writes_py

    # ------------------------------------------------------------------ connections
    def open(self, tag: str, sink: Sink) -> PipeConn:
        """Create a fresh pipe instance (unique random name) and start listening on it."""
        if self._closed:
            raise OSError("PipeMux is closed")
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)[:64]
        name = f"{PIPE_PREFIX}{os.getpid()}-{safe}-{secrets.token_hex(16)}"
        h = _winapi.CreateNamedPipe(
            name, _winapi.PIPE_ACCESS_DUPLEX | _winapi.FILE_FLAG_OVERLAPPED | _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE,
            _winapi.PIPE_WAIT | _PIPE_REJECT_REMOTE_CLIENTS,  # byte stream, byte reads
            1, PIPE_BUFFER, PIPE_BUFFER, 0, _winapi.NULL)
        key = self._next_key
        self._next_key += 1
        try:
            _overlapped.CreateIoCompletionPort(h, self.port, key, 0)
            if not _k32.SetFileCompletionNotificationModes(h, _SKIP_PORT_ON_SUCCESS):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            _winapi.CloseHandle(h)
            raise
        conn = PipeConn(self, key, name, h, sink, self.max_bytes)
        self._conns[key] = conn
        if self._nat is not None:
            try:
                now = self._nat.attach(key, h, self.max_bytes)
            except OSError:
                self._close(conn)
                raise
            if now:
                self._on_connect(conn)
            return conn
        ov = _overlapped.Overlapped(0)
        try:
            ov.ConnectNamedPipe(h)
        except OSError as e:
            if getattr(e, "winerror", None) != _ERROR_PIPE_CONNECTED:
                self._close(conn)
                raise
        if ov.pending:
            self._ops[ov.address] = (conn, "connect", ov)
        else:
            self._on_connect(conn)
        return conn

    def _on_connect(self, conn: PipeConn) -> None:
        conn.state = "connected"
        pid = ctypes.c_ulong(0)
        if _k32.GetNamedPipeClientProcessId(conn.handle, ctypes.byref(pid)):
            conn.client_pid = int(pid.value)
        out, conn.outbox = conn.outbox, []
        try:
            for m in out:
                self._write(conn, m)
        except OSError as e:
            self._fail(conn, e)
            return
        if self._nat is not None:
            self._dispatch(self._nat.start(conn.key))
        else:
            self._post_read(conn)

    # ------------------------------------------------------------------ I/O
    def _write(self, conn: PipeConn, message: bytes) -> None:
        nat = self._nat
        if nat is not None:  # raises OSError (e.g. broken pipe); a pending write completes on the port
            nat.write(conn.key, message)
            return
        ov = _overlapped.Overlapped(0)
        ov.WriteFile(conn.handle, message)  # raises OSError (e.g. broken pipe)
        if ov.pending:  # pipe buffer full: completes later on the port
            self._ops[ov.address] = (conn, "write", ov)
            self._writes_py += 1

    def _post_read(self, conn: PipeConn) -> None:
        """Keep one read outstanding; consume reads that complete synchronously."""
        while conn.state == "connected":
            if conn.have == len(conn.buf):  # a message larger than the buffer: grow it
                nb = bytearray(max(2 * len(conn.buf), conn.have + READ_BUFFER))
                nb[: conn.have] = conn.buf[: conn.have]
                conn.buf, conn.view = nb, memoryview(nb)
            ov = _overlapped.Overlapped(0)
            try:
                ov.ReadFileInto(conn.handle, conn.view[conn.have :])
            except OSError as e:
                self._fail(conn, EOF if getattr(e, "winerror", None) in (_ERROR_BROKEN_PIPE, _ERROR_NO_DATA) else e)
                return
            if ov.pending:
                conn._read = ov
                self._ops[ov.address] = (conn, "read", ov)
                return
            try:
                n = ov.getresult()
            except OSError as e:
                self._fail(conn, EOF if getattr(e, "winerror", None) == _ERROR_BROKEN_PIPE else e)
                return
            if not self._consume(conn, n):
                return

    def _consume(self, conn: PipeConn, n: int) -> bool:
        """Account ``n`` new bytes and deliver every complete message; False if the conn failed."""
        if n == 0:
            self._fail(conn, EOF)
            return False
        conn.have += n
        conn.bytes_in += n
        buf, have, start = conn.buf, conn.have, 0
        sink = conn.sink
        while have - start >= 4:
            (m,) = _LEN.unpack_from(buf, start)
            if m > conn.max_bytes:
                self._fail(conn, ProtocolError(f"message length {m} exceeds limit {conn.max_bytes}"))
                return False
            if have - start < 4 + m:
                if 4 + m > len(buf) - start and start == 0:  # grow before the next read
                    nb = bytearray(4 + m)
                    nb[:have] = buf[:have]
                    conn.buf, conn.view = nb, memoryview(nb)
                break
            payload = bytes(conn.view[start + 4 : start + 4 + m])
            start += 4 + m
            if conn.messages == 0:
                if not self._hello(conn, payload):
                    return False
                conn.messages += 1
                continue
            conn.messages += 1
            sink(payload)
            if conn.state != "connected":  # the sink closed it (host restarted)
                return False
        if start:
            rest = have - start
            if rest:
                conn.buf[:rest] = conn.buf[start:have]
            conn.have = rest
        return True

    def _hello(self, conn: PipeConn, payload: bytes) -> bool:
        if len(payload) != HELLO.size:
            self._fail(conn, ProtocolError(f"expected HELLO ({HELLO.size} bytes), got {len(payload)} bytes"))
            return False
        magic, version, pid, _ = HELLO.unpack(payload)
        if magic != HELLO_MAGIC or version != TRANSPORT_VERSION:
            self._fail(conn, ProtocolError(f"bad HELLO magic 0x{magic:08X} / version {version}"))
            return False
        if conn.client_pid > 0 and pid != conn.client_pid:
            self._fail(conn, ProtocolError(f"HELLO pid {pid} != pipe client pid {conn.client_pid}"))
            return False
        conn.client_pid = pid
        return True

    def _fail(self, conn: PipeConn, event: object) -> None:
        if conn.state == "closed":
            return
        self._close(conn)
        conn.sink(event)

    def _close(self, conn: PipeConn) -> None:
        if conn.state == "closed":
            return
        conn.state = "closed"
        conn.outbox = []
        self._conns.pop(conn.key, None)
        if self._nat is not None:
            self._nat.detach(conn.key)  # cancels its I/O; C frees it once the cancellations are dequeued
        for addr, (c, _, ov) in list(self._ops.items()):
            if c is conn and ov.pending:
                try:
                    ov.cancel()
                except OSError:
                    pass
        try:
            _winapi.CloseHandle(conn.handle)
        except OSError:
            pass
        # pending operations stay in _ops until their (aborted) completions are dequeued

    # ------------------------------------------------------------------ event loop
    def poll(self, timeout: float) -> int:
        """Process completions: wait up to ``timeout`` seconds for the first (0 = do not block),
        then everything already queued. Returns the number of completions processed.

        Completions are dequeued in batches with ``GetQueuedCompletionStatusEx`` (one system call
        for up to :data:`DEQUEUE_BATCH` completions)."""
        ms = 0 if timeout < 0.001 else int(timeout * 1000.0 + 0.999)  # sub-millisecond: poll, the caller loops
        if self._nat is not None:
            return self._dispatch(self._nat.poll(ms))
        n = 0
        entries, mv, removed = self._entries, self._entries_mv, self._removed
        if ms:  # only a blocking call is waiting; a non-blocking one is plain CPU work
            t0 = time.perf_counter_ns()
            ok = _GQCSEX(self._port_h, entries, DEQUEUE_BATCH, ctypes.byref(removed), ms, 0)
            self._wait_py += time.perf_counter_ns() - t0
        else:
            t0 = time.perf_counter_ns()
            ok = _GQCSEX(self._port_h, entries, DEQUEUE_BATCH, ctypes.byref(removed), 0, 0)
            if not ok:  # an empty poll is idle time too
                self._wait_py += time.perf_counter_ns() - t0
        while ok:
            k = removed.value
            n += k
            for i in range(k):
                self._complete(mv[4 * i + 1])
            if k < DEQUEUE_BATCH:
                break
            ok = _GQCSEX(self._port_h, entries, DEQUEUE_BATCH, ctypes.byref(removed), 0, 0)
        return n

    def _complete(self, address: int) -> None:
        """One dequeued completion (``address`` = its OVERLAPPED)."""
        rec = self._ops.pop(address, None)
        if rec is None:
            return
        conn, kind, ov = rec
        if kind == "read":
            conn._read = None
            if conn.state == "connected":
                try:
                    got = ov.getresult()
                except OSError as e:
                    self._fail(conn, EOF if getattr(e, "winerror", None) == _ERROR_BROKEN_PIPE else e)
                else:
                    if self._consume(conn, got):
                        self._post_read(conn)
        elif kind == "connect":
            if conn.state == "listening":
                try:
                    ov.getresult()
                except OSError as e:
                    self._fail(conn, e)
                else:
                    self._on_connect(conn)
        else:  # write
            self._writes_py -= 1
            if conn.state == "connected":
                try:
                    ov.getresult()
                except OSError as e:
                    self._fail(conn, EOF if getattr(e, "winerror", None) in (_ERROR_BROKEN_PIPE, _ERROR_NO_DATA) else e)

    def _dispatch(self, events: list[tuple[Any, ...]]) -> int:
        """Deliver the events of the native implementation (native/pipemux.c): ``(key, payload)``
        messages and ``(key, None, kind, value)`` connection events. Returns the number of events."""
        conns = self._conns
        for ev in events:
            conn = conns.get(ev[0])
            if conn is None:  # closed meanwhile (e.g. by a sink earlier in this list)
                continue
            if len(ev) == 2:
                payload = ev[1]
                conn.bytes_in += 4 + len(payload)
                if conn.messages:
                    conn.messages += 1
                    conn.sink(payload)
                elif self._hello(conn, payload):
                    conn.messages = 1
                continue
            kind, value = ev[2], ev[3]
            if kind == _EV_CONNECTED:
                if conn.state == "listening":
                    self._on_connect(conn)
            elif kind == _EV_EOF:
                self._fail(conn, EOF)
            elif kind == _EV_TOO_LONG:
                self._fail(conn, ProtocolError(f"message length {value} exceeds limit {conn.max_bytes}"))
            else:
                self._fail(conn, ctypes.WinError(value))
        return len(events)

    def close(self, timeout: float = 2.0) -> None:
        """Close every connection, reap their cancelled operations, close the port. Idempotent."""
        if self._closed:
            return
        for conn in list(self._conns.values()):
            self._close(conn)
        end = time.monotonic() + timeout
        nat = self._nat
        if nat is not None:
            while nat.live_ops and time.monotonic() < end:
                nat.poll(50)  # events of closed connections are dropped
            if nat.live_ops:  # never free an OVERLAPPED the kernel may still write
                _LEAKED.append(nat)
        while self._ops and time.monotonic() < end:
            if _GQCSEX(self._port_h, self._entries, DEQUEUE_BATCH, ctypes.byref(self._removed), 50, 0):
                for i in range(self._removed.value):
                    self._ops.pop(self._entries_mv[4 * i + 1], None)
        if self._ops:  # never deallocate an Overlapped with a pending operation (it would crash)
            _LEAKED.extend(self._ops.values())
            self._ops.clear()
        self._conns.clear()
        self._closed = True
        try:
            _winapi.CloseHandle(self.port)
        except OSError:
            pass

    @property
    def n_open(self) -> int:
        return len(self._conns)

    def acquire(self) -> None:
        """Register a user (a host wrapper); see :meth:`release`."""
        self._users += 1

    def release(self) -> None:
        """Unregister a user; the last one closes an ``auto_close`` mux."""
        self._users -= 1
        if self._users <= 0 and self.auto_close:
            self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close(timeout=0.5)
        except Exception:
            pass


_LEAKED: list[object] = []  # operations that never completed (kept alive on purpose)


def hello_message(pid: int | None = None) -> bytes:
    """The host's first message (fake hosts / reference implementation)."""
    return _LEN.pack(HELLO.size) + HELLO.pack(HELLO_MAGIC, TRANSPORT_VERSION, os.getpid() if pid is None else pid, 0)


class PosixPipeMux:
    """Linux counterpart of :class:`PipeMux` (same interface, same single-thread contract): the host keeps its
    stdin / stdout - two anonymous pipes created here - and the actor thread reads every host's stdout through
    one level-triggered ``epoll`` instance, without reader threads. No ``--pipe`` argument and no ``HELLO``:
    a connection is connected from :meth:`open` on; the host process (started by the caller with
    :attr:`PosixPipeConn.child_fds` as stdin / stdout) sees exactly the stdio transport.

    Writes are blocking (one command per host in flight; commands are far below the pipe buffer); EOF on
    the host's stdout ends the connection like a closed named pipe."""

    def __init__(self, max_bytes: int = MAX_FRAME_BYTES_HIDDEN, *, auto_close: bool = True) -> None:
        import select
        if not hasattr(select, "epoll"):
            raise OSError("PosixPipeMux needs Linux epoll")
        self.max_bytes = max_bytes
        self.auto_close = auto_close
        self._users = 0
        self._ep = select.epoll()  # type: ignore[attr-defined]  # Linux only
        self._in = select.EPOLLIN  # type: ignore[attr-defined]
        self._conns: dict[int, PosixPipeConn] = {}  # stdout read fd -> conn
        self._closed = False
        self.wait_ns = 0  # ns in blocking or empty epoll waits (cumulative)
        self.writes_pending = 0

    native = False

    def open(self, tag: str, sink: Sink) -> "PosixPipeConn":
        if self._closed:
            raise OSError("PosixPipeMux is closed")
        out_r, out_w = os.pipe()  # host stdout -> actor
        in_r, in_w = os.pipe()  # actor -> host stdin
        os.set_blocking(out_r, False)  # type: ignore[attr-defined]  # POSIX only
        conn = PosixPipeConn(self, out_r, tag, in_w, sink, self.max_bytes)
        conn.child_fds = (in_r, out_w)
        self._conns[out_r] = conn
        self._ep.register(out_r, self._in)
        return conn

    def _write(self, conn: "PosixPipeConn", message: bytes) -> None:
        view = memoryview(message)
        while view:  # raises BrokenPipeError when the host is gone
            view = view[os.write(conn.handle, view):]

    def poll(self, timeout: float) -> int:
        """Read every readable host stdout, waiting up to ``timeout`` seconds for the first (0 = do not
        block); complete messages go to the sinks. Returns the number of readable connections handled."""
        t0 = time.perf_counter_ns()
        events = self._ep.poll(timeout if timeout > 0 else 0, 1024)
        if not events or timeout > 0:
            self.wait_ns += time.perf_counter_ns() - t0
        conns = self._conns
        for fd, _ in events:
            conn = conns.get(fd)
            if conn is not None:  # None: closed by an earlier sink in this batch
                self._read(conn)
        return len(events)

    def _read(self, conn: "PosixPipeConn") -> None:
        if conn.have == len(conn.buf):  # a message larger than the buffer: grow it
            nb = bytearray(max(2 * len(conn.buf), conn.have + READ_BUFFER))
            nb[: conn.have] = conn.buf[: conn.have]
            conn.buf, conn.view = nb, memoryview(nb)
        n = os.readv(conn.key, [conn.view[conn.have:]])  # type: ignore[attr-defined]  # POSIX only
        if n == 0:
            self._fail(conn, EOF)
            return
        conn.have += n
        conn.bytes_in += n
        buf, have, start = conn.buf, conn.have, 0
        while have - start >= 4:
            (m,) = _LEN.unpack_from(buf, start)
            if m > conn.max_bytes:
                self._fail(conn, ProtocolError(f"message length {m} exceeds limit {conn.max_bytes}"))
                return
            if have - start < 4 + m:
                if 4 + m > len(buf) - start and start == 0:  # grow before the next read
                    nb = bytearray(4 + m)
                    nb[:have] = buf[:have]
                    conn.buf, conn.view = nb, memoryview(nb)
                break
            payload = bytes(conn.view[start + 4: start + 4 + m])
            start += 4 + m
            conn.messages += 1
            conn.sink(payload)
            if conn.state != "connected":  # the sink closed it (host restarted)
                return
        if start:
            rest = have - start
            if rest:
                conn.buf[:rest] = conn.buf[start:have]
            conn.have = rest

    def _fail(self, conn: "PosixPipeConn", event: object) -> None:
        if conn.state == "closed":
            return
        self._close(conn)
        conn.sink(event)

    def _close(self, conn: "PosixPipeConn") -> None:
        if conn.state == "closed":
            return
        conn.state = "closed"
        self._conns.pop(conn.key)
        self._ep.unregister(conn.key)
        conn.spawned()
        os.close(conn.key)
        os.close(conn.handle)

    def close(self, timeout: float = 2.0) -> None:
        """Close every connection and the epoll instance. Idempotent."""
        if self._closed:
            return
        for conn in list(self._conns.values()):
            self._close(conn)
        self._ep.close()
        self._closed = True

    @property
    def n_open(self) -> int:
        return len(self._conns)

    def acquire(self) -> None:
        self._users += 1

    def release(self) -> None:
        self._users -= 1
        if self._users <= 0 and self.auto_close:
            self.close()


class PosixPipeConn(PipeConn):
    """One host of a :class:`PosixPipeMux`: ``key`` = the read end of the host's stdout, ``handle`` = the
    write end of its stdin, ``child_fds`` = (stdin read end, stdout write end) for the host process, closed
    here by :meth:`spawned` once the child holds them."""

    __slots__ = ("child_fds",)

    def __init__(self, mux: "PosixPipeMux", key: int, name: str, handle: int, sink: Sink, max_bytes: int) -> None:
        super().__init__(mux, key, name, handle, sink, max_bytes)  # type: ignore[arg-type]
        self.state = "connected"
        self.child_fds: tuple[int, ...] = ()

    def spawned(self) -> None:
        """Close the host's pipe ends in this process (after the host was started, or when it never will)."""
        for fd in self.child_fds:
            os.close(fd)
        self.child_fds = ()


def new_mux() -> "PipeMux | PosixPipeMux":
    """The multiplexer of the "pipe" transport on this platform."""
    return PipeMux() if sys.platform == "win32" else PosixPipeMux()
