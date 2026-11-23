/*
 * spire._pipemux - the I/O half of spire.pipes.PipeMux in C (py-throughput-1).
 *
 * The Python PipeMux spends ~6-8 us of interpreter time per frame on overlapped-I/O bookkeeping
 * (an Overlapped object per read and per write, a dict of pending operations, the completion
 * dequeue loop, length-prefix framing and buffer management). This module does exactly that work
 * in C; everything else (pipe creation, HELLO validation, outbox, sinks, error policy) stays in
 * spire/pipes.py. The byte protocol and the transport behaviour are unchanged
 * (notes/transport-spec.md); tests/test_pipes.py runs against both implementations.
 *
 * One Mux serves the pipe handles of one completion port (owned by Python). Every overlapped
 * operation on those handles is issued here (connect, reads, writes), so every completion packet
 * on the port belongs to this module. Per connection, C keeps one read outstanding and frames the
 * byte stream into messages ([u32 LE length][payload]).
 *
 *   Mux(port, max_events)
 *   attach(key, handle, max_bytes) -> bool        issue ConnectNamedPipe; True: already connected
 *   start(key) -> list[event]                     start reading (after connect); synchronous reads
 *   write(key, data) -> bool                      WriteFile; True: pending (completes on the port)
 *   poll(timeout_ms) -> list[event]               dequeue completions (all queued, in batches)
 *   detach(key)                                   cancel the connection's I/O; the caller closes
 *                                                 the handle; C frees the connection once its
 *                                                 cancelled operations were dequeued
 *   live_ops, writes_pending, wait_ns             attributes (int)
 *
 * Events: (key, payload: bytes) for a message, or (key, None, kind, value) with kind
 * EV_CONNECTED (value: 0), EV_EOF (value: winerror, 0 / 109 / 232), EV_ERROR (value: winerror),
 * EV_TOO_LONG (value: announced length). After any non-message event the connection delivers
 * nothing more.
 *
 * Memory safety: an OVERLAPPED must stay valid until its completion is dequeued. Connections are
 * freed only when no operation of theirs is pending; a Mux deallocated with pending operations
 * leaks them on purpose (spire.pipes drains the port first and keeps undrained muxes alive).
 *
 * Build: python -B native/build.py (MSVC; spire/_pipemux.<abi>.pyd).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <string.h>

#define PIPEMUX_VERSION 1
#define READ_BUFFER (1 << 17)
#define MIN_READ 4096

enum { EV_CONNECTED = 0, EV_EOF = 1, EV_ERROR = 2, EV_TOO_LONG = 3 };
enum { OP_CONNECT = 1, OP_READ = 2, OP_WRITE = 3 };

typedef struct Conn Conn;

typedef struct Op {
    OVERLAPPED ov; /* first member: the OVERLAPPED* of a completion is the Op* */
    int kind;
    Conn *conn;
    PyObject *data; /* write: the bytes object kept alive while pending */
    struct Op *next_free;
} Op;

struct Conn {
    Op cop, rop; /* connect, read */
    HANDLE h;
    long long key;
    int open;     /* until detach() */
    int failed;   /* a terminal event was produced: deliver nothing more */
    int pending;  /* this connection's operations not yet dequeued */
    char *buf;
    size_t cap, have;
    uint32_t max_bytes;
};

typedef struct {
    PyObject_HEAD
    HANDLE port;
    PyObject *conns; /* dict key -> PyLong(Conn*) of attached (not detached) connections */
    OVERLAPPED_ENTRY *entries;
    ULONG max_events;
    long long live_ops, writes_pending, wait_ns;
    Op *free_ops;
    int leaked;
    LARGE_INTEGER qpf;
} Mux;

/* ------------------------------------------------------------------ helpers */

static Conn *lookup(Mux *m, PyObject *key) {
    PyObject *v = PyDict_GetItemWithError(m->conns, key);
    if (v == NULL) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_BrokenPipeError, "pipemux: connection is not attached");
        return NULL;
    }
    return (Conn *)PyLong_AsVoidPtr(v);
}

static void conn_free_if_done(Conn *c) {
    if (!c->open && c->pending == 0) {
        PyMem_RawFree(c->buf);
        PyMem_RawFree(c);
    }
}

static int emit_msg(PyObject *events, Conn *c, const char *p, size_t n) {
    PyObject *b = PyBytes_FromStringAndSize(p, (Py_ssize_t)n);
    if (b == NULL) return -1;
    PyObject *t = PyTuple_New(2);
    if (t == NULL) { Py_DECREF(b); return -1; }
    PyTuple_SET_ITEM(t, 0, PyLong_FromLongLong(c->key));
    PyTuple_SET_ITEM(t, 1, b);
    int r = PyList_Append(events, t);
    Py_DECREF(t);
    return r;
}

static int emit_status(PyObject *events, Conn *c, int kind, long long value) {
    if (kind != EV_CONNECTED) c->failed = 1;
    PyObject *t = Py_BuildValue("(LOiL)", c->key, Py_None, kind, value);
    if (t == NULL) return -1;
    int r = PyList_Append(events, t);
    Py_DECREF(t);
    return r;
}

/* Frame the bytes in c->buf; emits complete messages. Returns -1 on a Python error, 1 if the
   connection failed (too long), 0 otherwise. Ensures room for the next read. */
static int frame(PyObject *events, Conn *c) {
    size_t start = 0, have = c->have;
    while (have - start >= 4) {
        uint32_t m;
        memcpy(&m, c->buf + start, 4);
        if (m > c->max_bytes) {
            c->have = 0;
            return emit_status(events, c, EV_TOO_LONG, (long long)m) < 0 ? -1 : 1;
        }
        if (have - start < 4 + (size_t)m) {
            break;
        }
        if (emit_msg(events, c, c->buf + start + 4, m) < 0) return -1;
        start += 4 + (size_t)m;
    }
    if (start) {
        size_t rest = have - start;
        if (rest) memmove(c->buf, c->buf + start, rest);
        c->have = have = rest;
    }
    /* room for the rest of a partial message (and at least MIN_READ bytes) */
    size_t need = have + MIN_READ;
    if (have >= 4) {
        uint32_t m;
        memcpy(&m, c->buf, 4);
        if (4 + (size_t)m > need) need = 4 + (size_t)m;
    }
    if (need > c->cap) {
        size_t nc = c->cap * 2;
        if (nc < need) nc = need;
        char *nb = (char *)PyMem_RawRealloc(c->buf, nc);
        if (nb == NULL) { PyErr_NoMemory(); return -1; }
        c->buf = nb;
        c->cap = nc;
    }
    return 0;
}

static int is_eof(DWORD err) { return err == ERROR_BROKEN_PIPE || err == ERROR_NO_DATA || err == ERROR_PIPE_NOT_CONNECTED; }

/* Consume n bytes that a read delivered, then keep one read outstanding (consuming reads that
   complete synchronously). Returns -1 on a Python error. */
static int read_loop(Mux *m, PyObject *events, Conn *c, DWORD n) {
    for (;;) {
        if (n == 0) {
            return emit_status(events, c, EV_EOF, 0);
        }
        c->have += n;
        int r = frame(events, c);
        if (r != 0) return r < 0 ? -1 : 0;
        memset(&c->rop.ov, 0, sizeof(OVERLAPPED));
        if (ReadFile(c->h, c->buf + c->have, (DWORD)(c->cap - c->have), NULL, &c->rop.ov)) {
            /* completed synchronously; FILE_SKIP_COMPLETION_PORT_ON_SUCCESS: no packet is queued */
            if (!GetOverlappedResult(c->h, &c->rop.ov, &n, FALSE)) {
                DWORD e = GetLastError();
                return emit_status(events, c, is_eof(e) ? EV_EOF : EV_ERROR, (long long)e);
            }
            continue;
        }
        DWORD e = GetLastError();
        if (e == ERROR_IO_PENDING) {
            c->pending++;
            m->live_ops++;
            return 0;
        }
        return emit_status(events, c, is_eof(e) ? EV_EOF : EV_ERROR, (long long)e);
    }
}

/* ------------------------------------------------------------------ type */

static int Mux_init(Mux *m, PyObject *args, PyObject *kw) {
    unsigned long long port;
    unsigned long max_events = 128;
    if (!PyArg_ParseTuple(args, "K|k", &port, &max_events)) return -1;
    if (max_events < 1) max_events = 1;
    m->port = (HANDLE)(uintptr_t)port;
    m->conns = PyDict_New();
    if (m->conns == NULL) return -1;
    m->entries = (OVERLAPPED_ENTRY *)PyMem_RawCalloc(max_events, sizeof(OVERLAPPED_ENTRY));
    if (m->entries == NULL) { PyErr_NoMemory(); return -1; }
    m->max_events = max_events;
    QueryPerformanceFrequency(&m->qpf);
    return 0;
}

static void Mux_dealloc(Mux *m) {
    if (m->conns != NULL) {
        PyObject *k, *v;
        Py_ssize_t pos = 0;
        while (PyDict_Next(m->conns, &pos, &k, &v)) {
            Conn *c = (Conn *)PyLong_AsVoidPtr(v);
            c->open = 0;
            CancelIoEx(c->h, NULL);
            if (m->live_ops == 0) conn_free_if_done(c);
        }
        Py_CLEAR(m->conns);
    }
    if (m->live_ops == 0) { /* otherwise OVERLAPPEDs may still be written by the kernel: leak */
        PyMem_RawFree(m->entries);
        while (m->free_ops) { Op *o = m->free_ops; m->free_ops = o->next_free; PyMem_RawFree(o); }
    }
    Py_TYPE(m)->tp_free((PyObject *)m);
}

static PyObject *Mux_attach(Mux *m, PyObject *args) {
    PyObject *key;
    unsigned long long h;
    unsigned long max_bytes;
    if (!PyArg_ParseTuple(args, "OKk", &key, &h, &max_bytes)) return NULL;
    long long k = PyLong_AsLongLong(key);
    if (k == -1 && PyErr_Occurred()) return NULL;
    if (PyDict_Contains(m->conns, key)) { PyErr_SetString(PyExc_ValueError, "pipemux: key already attached"); return NULL; }
    Conn *c = (Conn *)PyMem_RawCalloc(1, sizeof(Conn));
    if (c == NULL) return PyErr_NoMemory();
    c->buf = (char *)PyMem_RawMalloc(READ_BUFFER);
    if (c->buf == NULL) { PyMem_RawFree(c); return PyErr_NoMemory(); }
    c->cap = READ_BUFFER;
    c->h = (HANDLE)(uintptr_t)h;
    c->key = k;
    c->open = 1;
    c->max_bytes = (uint32_t)max_bytes;
    c->cop.kind = OP_CONNECT;
    c->cop.conn = c;
    c->rop.kind = OP_READ;
    c->rop.conn = c;
    PyObject *ptr = PyLong_FromVoidPtr(c);
    if (ptr == NULL || PyDict_SetItem(m->conns, key, ptr) < 0) {
        Py_XDECREF(ptr);
        PyMem_RawFree(c->buf);
        PyMem_RawFree(c);
        return NULL;
    }
    Py_DECREF(ptr);
    if (ConnectNamedPipe(c->h, &c->cop.ov)) { /* overlapped: normally FALSE */
        Py_RETURN_TRUE;
    }
    DWORD e = GetLastError();
    if (e == ERROR_PIPE_CONNECTED) Py_RETURN_TRUE;
    if (e == ERROR_IO_PENDING) {
        c->pending++;
        m->live_ops++;
        Py_RETURN_FALSE;
    }
    PyDict_DelItem(m->conns, key);
    c->open = 0;
    conn_free_if_done(c);
    PyErr_SetFromWindowsErr((int)e);
    return NULL;
}

static PyObject *Mux_start(Mux *m, PyObject *key) {
    Conn *c = lookup(m, key);
    if (c == NULL) return NULL;
    PyObject *events = PyList_New(0);
    if (events == NULL) return NULL;
    memset(&c->rop.ov, 0, sizeof(OVERLAPPED));
    /* read_loop with n = 0 would mean EOF: post the first read directly */
    if (ReadFile(c->h, c->buf + c->have, (DWORD)(c->cap - c->have), NULL, &c->rop.ov)) {
        DWORD n = 0;
        if (!GetOverlappedResult(c->h, &c->rop.ov, &n, FALSE)) {
            DWORD e = GetLastError();
            if (emit_status(events, c, is_eof(e) ? EV_EOF : EV_ERROR, (long long)e) < 0) goto fail;
        } else if (read_loop(m, events, c, n) < 0) {
            goto fail;
        }
    } else {
        DWORD e = GetLastError();
        if (e == ERROR_IO_PENDING) {
            c->pending++;
            m->live_ops++;
        } else if (emit_status(events, c, is_eof(e) ? EV_EOF : EV_ERROR, (long long)e) < 0) {
            goto fail;
        }
    }
    return events;
fail:
    Py_DECREF(events);
    return NULL;
}

static PyObject *Mux_write(Mux *m, PyObject *args) {
    PyObject *key, *data;
    if (!PyArg_ParseTuple(args, "OS", &key, &data)) return NULL;
    Conn *c = lookup(m, key);
    if (c == NULL) return NULL;
    Op *o = m->free_ops;
    if (o != NULL) {
        m->free_ops = o->next_free;
    } else {
        o = (Op *)PyMem_RawMalloc(sizeof(Op));
        if (o == NULL) return PyErr_NoMemory();
    }
    memset(&o->ov, 0, sizeof(OVERLAPPED));
    o->kind = OP_WRITE;
    o->conn = c;
    o->data = NULL;
    o->next_free = NULL;
    if (WriteFile(c->h, PyBytes_AS_STRING(data), (DWORD)PyBytes_GET_SIZE(data), NULL, &o->ov)) {
        o->next_free = m->free_ops; /* completed synchronously (skip-port-on-success): no packet */
        m->free_ops = o;
        Py_RETURN_FALSE;
    }
    DWORD e = GetLastError();
    if (e == ERROR_IO_PENDING) {
        Py_INCREF(data);
        o->data = data;
        c->pending++;
        m->live_ops++;
        m->writes_pending++;
        Py_RETURN_TRUE;
    }
    o->next_free = m->free_ops;
    m->free_ops = o;
    PyErr_SetFromWindowsErr((int)e);
    return NULL;
}

static PyObject *Mux_detach(Mux *m, PyObject *key) {
    PyObject *v = PyDict_GetItemWithError(m->conns, key);
    if (v == NULL) {
        if (PyErr_Occurred()) return NULL;
        Py_RETURN_NONE;
    }
    Conn *c = (Conn *)PyLong_AsVoidPtr(v);
    if (PyDict_DelItem(m->conns, key) < 0) return NULL;
    c->open = 0;
    if (c->pending) CancelIoEx(c->h, NULL); /* completions (aborted) arrive on the port */
    conn_free_if_done(c);
    Py_RETURN_NONE;
}

/* one dequeued completion */
static int complete(Mux *m, PyObject *events, OVERLAPPED_ENTRY *e) {
    Op *o = (Op *)e->lpOverlapped;
    if (o == NULL) return 0;
    Conn *c = o->conn;
    m->live_ops--;
    c->pending--;
    int kind = o->kind;
    if (kind == OP_WRITE) {
        m->writes_pending--;
        Py_CLEAR(o->data);
        o->next_free = m->free_ops;
        m->free_ops = o;
    }
    if (!c->open || c->failed) {
        conn_free_if_done(c);
        return 0;
    }
    DWORD n = 0;
    BOOL ok = GetOverlappedResult(c->h, &o->ov, &n, FALSE);
    DWORD err = ok ? 0 : GetLastError();
    if (kind == OP_READ) {
        if (!ok) return emit_status(events, c, is_eof(err) ? EV_EOF : EV_ERROR, (long long)err);
        return read_loop(m, events, c, n);
    }
    if (kind == OP_CONNECT) {
        if (!ok && err != ERROR_PIPE_CONNECTED) return emit_status(events, c, EV_ERROR, (long long)err);
        return emit_status(events, c, EV_CONNECTED, 0);
    }
    if (!ok) return emit_status(events, c, is_eof(err) ? EV_EOF : EV_ERROR, (long long)err);
    return 0;
}

static PyObject *Mux_poll(Mux *m, PyObject *arg) {
    unsigned long ms = PyLong_AsUnsignedLong(arg);
    if (ms == (unsigned long)-1 && PyErr_Occurred()) return NULL;
    PyObject *events = PyList_New(0);
    if (events == NULL) return NULL;
    ULONG got = 0;
    BOOL ok;
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    if (ms) {
        Py_BEGIN_ALLOW_THREADS
        ok = GetQueuedCompletionStatusEx(m->port, m->entries, m->max_events, &got, ms, FALSE);
        Py_END_ALLOW_THREADS
    } else {
        ok = GetQueuedCompletionStatusEx(m->port, m->entries, m->max_events, &got, 0, FALSE);
    }
    if (ms || !ok) { /* a blocking wait, or an empty poll, is idle time */
        QueryPerformanceCounter(&t1);
        m->wait_ns += (long long)((double)(t1.QuadPart - t0.QuadPart) * 1e9 / (double)m->qpf.QuadPart);
    }
    while (ok) {
        for (ULONG i = 0; i < got; i++) {
            if (complete(m, events, &m->entries[i]) < 0) { Py_DECREF(events); return NULL; }
        }
        if (got < m->max_events) break;
        ok = GetQueuedCompletionStatusEx(m->port, m->entries, m->max_events, &got, 0, FALSE);
    }
    return events;
}

static PyMethodDef Mux_methods[] = {
    {"attach", (PyCFunction)Mux_attach, METH_VARARGS, "attach(key, handle, max_bytes) -> connected now"},
    {"start", (PyCFunction)Mux_start, METH_O, "start(key) -> events of synchronously completed reads"},
    {"write", (PyCFunction)Mux_write, METH_VARARGS, "write(key, data) -> pending"},
    {"detach", (PyCFunction)Mux_detach, METH_O, "detach(key): cancel; the caller closes the handle"},
    {"poll", (PyCFunction)Mux_poll, METH_O, "poll(timeout_ms) -> events"},
    {NULL, NULL, 0, NULL},
};

static PyMemberDef Mux_members[] = {
    {"live_ops", T_LONGLONG, offsetof(Mux, live_ops), READONLY, "operations not yet dequeued"},
    {"writes_pending", T_LONGLONG, offsetof(Mux, writes_pending), READONLY, "pending writes"},
    {"wait_ns", T_LONGLONG, offsetof(Mux, wait_ns), READONLY, "ns in blocking or empty polls (cumulative)"},
    {NULL, 0, 0, 0, NULL},
};

static PyTypeObject MuxType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spire._pipemux.Mux",
    .tp_basicsize = sizeof(Mux),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Mux_init,
    .tp_dealloc = (destructor)Mux_dealloc,
    .tp_methods = Mux_methods,
    .tp_members = Mux_members,
    .tp_doc = "Overlapped named-pipe I/O on one completion port (see module docstring).",
};

static struct PyModuleDef moddef = {PyModuleDef_HEAD_INIT, "_pipemux", "named-pipe completion-port I/O", -1, NULL};

PyMODINIT_FUNC PyInit__pipemux(void) {
    if (PyType_Ready(&MuxType) < 0) return NULL;
    PyObject *mod = PyModule_Create(&moddef);
    if (mod == NULL) return NULL;
    Py_INCREF(&MuxType);
    if (PyModule_AddObject(mod, "Mux", (PyObject *)&MuxType) < 0) return NULL;
    PyModule_AddIntConstant(mod, "VERSION", PIPEMUX_VERSION);
    PyModule_AddIntConstant(mod, "EV_CONNECTED", EV_CONNECTED);
    PyModule_AddIntConstant(mod, "EV_EOF", EV_EOF);
    PyModule_AddIntConstant(mod, "EV_ERROR", EV_ERROR);
    PyModule_AddIntConstant(mod, "EV_TOO_LONG", EV_TOO_LONG);
    PyModule_AddIntConstant(mod, "READ_BUFFER", READ_BUFFER);
    return mod;
}
