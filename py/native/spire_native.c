/*
 * spire._native - compiled hot-path kernels of the Python actor (py-throughput-1).
 *
 * encode_plan / encode_fill: the pooled-children tokenizer (spire.tokenize.Tokenizer.encode_batch)
 * as one C loop over the frames of a batch. Semantics are those of the frozen reference
 * spire/tokenize_ref.py (tests/test_tokenize_batch.py proves bit-identical output, including
 * fuzzed frames): raw slots are the value as stored into an int32 array (two's-complement wrap)
 * saturated to int16; relations / edges / cond are wrapped to int16; derived int32 arithmetic
 * wraps like numpy int32 arithmetic (computed exactly in int64, then truncated).
 *
 * All schema constants (table ids, column indices, derived slot positions, embedding-row
 * offsets) come from Python in the `params` int64 array (layout: NP_* below, built by
 * spire.tokenize._native_params); nothing about the schema is hard-coded here.
 *
 * Build: python -B native/build.py  (MSVC; produces spire/_native.<abi>.pyd).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> /* concat_into reads ndarray data pointers directly (headers only) */

#define NATIVE_VERSION 6
#define N_TABLES 16 /* v1 tables (references name only these) */
#define N_TAB 21    /* + the FORECAST trailer (16), the calc trailers TURN FOE ROUTE (17-19) and the AHEAD row (20) */
#define N_SEG 20 /* token tables (10) + kids (POWERS INTENTS REFS FORECAST TURN FOE ROUTE AHEAD) + VARS + ACTIONS */
#define N_KID_END 18 /* segments [10, 18) are kids */
#define N_CALC 3
#define N_AH_PAIRS 10 /* AHEAD: ROUTE columns with a MIN and a MAX slot */
#define N_AH_TYPES 6  /* AHEAD: room types counted in slots 1-6 */
#define N_AH_MAX 32   /* AHEAD slots fit the record width */

enum {
    NP_COUNT, NP_VERSION, NP_WIDTH, NP_N_ENT, NP_LV,
    NP_ST_CARD, NP_ST_MONSTER, NP_ST_RELIC, NP_ST_POTION, NP_ST_POWER,
    NP_D_KIND_SUB, NP_D_PROMPT, NP_D_ACTS, NP_D_C_STATIC, NP_D_C_PILE, NP_D_C_GROUP, NP_D_CR_MODEL,
    NP_D_CR_STATIC, NP_D_CR_LETHAL, NP_D_RELIC, NP_D_POTION, NP_D_DROW, NP_D_KEY, NP_D_PW_STATIC, NP_D_PW_AK,
    NP_D_REF,
    NP_ZONE_CHOICE, NP_ZONE_OFFER, NP_PLAYER,
    NP_G_KIND, NP_G_SUB, NP_G_PCAT, NP_G_PSID, NP_G_CUR_ROW, NP_G_ASC, NP_G_NSEATS,
    NP_S_CROW, NP_S_CHAR,
    NP_K_ZONE, NP_K_ID, NP_K_UP, NP_K_LINK, NP_K_AUX,
    NP_CR_KIND, NP_CR_MODEL, NP_CR_HP, NP_CR_POISON,
    NP_RL_ID, NP_PT_ID, NP_MN_ROW,
    NP_O_KIND, NP_O_KEY, NP_O_REF_T, NP_O_REF_R,
    NP_P_ID, NP_P_OWNER, NP_P_APPLIER,
    NP_I_OWNER,
    NP_RF_CAT, NP_RF_VOC, NP_RF_OT, NP_RF_OR, NP_RF_ET, NP_RF_ER,
    NP_V_OT, NP_V_OR, NP_V_TGT,
    NP_A_SEAT, NP_A_SRC_T, NP_A_SRC_R, NP_A_TGT,
    NP_E_FROM, NP_E_TO,
    NP_T_SEATS, NP_T_CARDS, NP_T_VARS, NP_T_CREATURES, NP_T_POWERS, NP_T_INTENTS, NP_T_RELICS, NP_T_POTIONS,
    NP_T_ORBS, NP_T_ARROWS, NP_T_NODES, NP_T_EDGES, NP_T_OPTIONS, NP_T_REFS, NP_T_CELLS, NP_T_ACTIONS,
    NP_DIR_W0, NP_GLOB_W0, NP_DIR_STRIDE,
    NP_ENT_GLOBAL, NP_ENT_DECISION,
    NP_N_CRE_POLY, NP_N_OPT_POLY, NP_N_REF_POLY,
    NP_NCOLS,                      /* 21 entries: columns per table (v1 + FORECAST + TURN FOE ROUTE + AHEAD slots) */
    NP_SEG_TAB = NP_NCOLS + N_TAB,   /* 20 entries: tables of the pooled segments, in order */
    NP_SEG_ENT = NP_SEG_TAB + N_SEG,  /* 20 entries: entity type of each segment */
    NP_MEND_TARGET = NP_SEG_ENT + N_SEG, NP_COOP_FEATURES,
    NP_D_VOTE_COUNT,
    NP_G_SEAT_SCREEN, NP_G_SEAT_STATUS, NP_S_SEAT, NP_O_FLAGS,
    NP_REST_OPTION, NP_MEND_KEY, NP_A_AUX,
    /* version 3: FORECAST trailer (notes/forecast/contract.md 2.2) */
    NP_FORECAST, NP_T_FORECAST, NP_FC_NCOLS, NP_ENT_FORECAST, NP_FC_OWNER, NP_H_N_TABLES, NP_DATA_W0,
    /* version 4: calc trailers (notes/calc/contract.md 2.2): on/off, 3 table ids, 3 owner columns, 3 owner tables */
    NP_CALC, NP_T_CALC, NP_CC_OWNER = NP_T_CALC + N_CALC, NP_CC_OWNER_T = NP_CC_OWNER + N_CALC,
    /* version 5 (notes/inputs2/contract.md 1-2): DROP rule on/off, route summary on/off, GLOBAL / header columns, map
       node columns and flag masks, the AHEAD pseudo-table and entity type, ROUTE columns, the six room type ids */
    NP_DROP = NP_CC_OWNER_T + N_CALC, NP_ROUTE_SUMMARY, NP_G_IN_COMBAT, NP_KIND_MAP, NP_H_STEP, NP_MN_PT, NP_MN_FLAGS,
    NP_FL_BOSS, NP_FL_QUEST, NP_T_AHEAD, NP_ENT_AHEAD, NP_RC_TO_BOSS, NP_RC_PAIRS,
    NP_PT_IDS = NP_RC_PAIRS + N_AH_PAIRS,
    /* version 6 (notes/critic/contract.md 1.2-1.3): the frames carry the HDRAW trailer iff NP_HIDDEN (its entry is
       checked, its rows are never read); encode_fill writes each frame's card_base */
    NP_HIDDEN = NP_PT_IDS + N_AH_TYPES, NP_T_HDRAW, NP_HD_NCOLS,
    NP__END
};

typedef struct {
    Py_buffer b;
    int ok;
} Buf;

static int get_buf(PyObject *o, Buf *out, int writable, Py_ssize_t itemsize, const char *name) {
    if (PyObject_GetBuffer(o, &out->b, (writable ? PyBUF_WRITABLE : 0) | PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return -1;
    out->ok = 1;
    if (out->b.itemsize != itemsize) {
        PyErr_Format(PyExc_TypeError, "%s: itemsize %zd, expected %zd", name, out->b.itemsize, itemsize);
        return -1;
    }
    return 0;
}

static void rel_bufs(Buf *bs, int n) {
    for (int i = 0; i < n; i++)
        if (bs[i].ok) PyBuffer_Release(&bs[i].b);
}

static inline int16_t sat16(int64_t v) {
    int32_t x = (int32_t)(uint32_t)(uint64_t)v; /* the int32 store, then saturation */
    return (int16_t)(x > 32767 ? 32767 : (x < -32768 ? -32768 : x));
}

static inline int16_t wrap16(int64_t v) { return (int16_t)(uint16_t)(uint64_t)v; }

static inline int64_t clip64(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* _resolve(off, rows, disc, ids): global row of vocab id `ids` in the region chosen by `disc` */
static inline int64_t resolve(const int64_t *off, const int64_t *rows, int64_t n, int64_t disc, int64_t ids) {
    int64_t d = clip64(disc, 0, n - 1);
    return (rows[d] > 0 && ids > 0) ? off[d] + ids : 0;
}

/* _ref(base, table, row): entity row of a (table, row) reference, -1 for none */
static inline int64_t ref(const int64_t *base, int64_t table, int64_t row) {
    int64_t b = base[clip64(table, 0, N_TABLES - 1)];
    return (table >= 0 && b >= 0) ? b + row : -1;
}

/* Directory of frame i: rows and absolute first word per table (N_TAB entries); checks the tables lie in the
   frame. The frame must carry the FORECAST trailer iff NP_FORECAST and the calc trailers iff NP_CALC (header
   N_TABLES 16 + [forecast] + 3 [calc], entries chained from X0 = end of the v1 data, ending at the frame end);
   absent trailer tables have 0 rows. */
static int frame_dir(const int32_t *W, int64_t nW, int64_t fb, int64_t len, const int64_t *P, int64_t *n,
                     int64_t *o) {
    if (fb < 0 || len < 0 || fb + len > nW || len < P[NP_DIR_W0] + N_TABLES * P[NP_DIR_STRIDE]) {
        PyErr_SetString(PyExc_ValueError, "frame outside the word buffer");
        return -1;
    }
    if (W[fb + P[NP_H_N_TABLES]] !=
        N_TABLES + (P[NP_FORECAST] ? 1 : 0) + (P[NP_CALC] ? N_CALC : 0) + (P[NP_HIDDEN] ? 1 : 0)) {
        PyErr_Format(PyExc_AssertionError, "frame has N_TABLES %d, tokenizer.forecast=%d calc=%d allow_hidden=%d",
                     (int)W[fb + P[NP_H_N_TABLES]], (int)P[NP_FORECAST], (int)P[NP_CALC], (int)P[NP_HIDDEN]);
        return -1;
    }
    const int32_t *d = W + fb + P[NP_DIR_W0];
    int64_t x0 = P[NP_DATA_W0];
    for (int t = 0; t < N_TABLES; t++) {
        int64_t rows = d[t * P[NP_DIR_STRIDE] + 1], off = d[t * P[NP_DIR_STRIDE] + 3] >> 2;
        int64_t nc = P[NP_NCOLS + t];
        if (rows < 0 || off < 0 || off + rows * nc > len) {
            PyErr_Format(PyExc_ValueError, "table %d of a frame is elided or out of bounds", t);
            return -1;
        }
        n[t] = rows;
        o[t] = fb + off;
        x0 += rows * nc;
    }
    const int64_t tf = P[NP_T_FORECAST];
    n[tf] = 0;
    o[tf] = fb;
    if (P[NP_FORECAST]) {
        if (x0 + P[NP_DIR_STRIDE] > len) {
            PyErr_SetString(PyExc_ValueError, "FORECAST entry beyond the frame");
            return -1;
        }
        const int32_t *e = W + fb + x0;
        int64_t rows = e[1], off = e[3] >> 2;
        if (e[0] != tf || e[2] != P[NP_FC_NCOLS] || rows < 0 || off != x0 + P[NP_DIR_STRIDE] ||
            off + rows * P[NP_FC_NCOLS] > len) {
            PyErr_SetString(PyExc_ValueError, "FORECAST entry of a frame is malformed or out of bounds");
            return -1;
        }
        n[tf] = rows;
        o[tf] = fb + off;
        x0 = off + rows * P[NP_FC_NCOLS];
    }
    for (int k = 0; k < N_CALC; k++) {
        const int64_t tc = P[NP_T_CALC + k];
        n[tc] = 0;
        o[tc] = fb;
    }
    if (P[NP_CALC]) {
        for (int k = 0; k < N_CALC; k++) {
            const int64_t tc = P[NP_T_CALC + k], nc = P[NP_NCOLS + tc];
            if (x0 + P[NP_DIR_STRIDE] > len) {
                PyErr_SetString(PyExc_ValueError, "calc entry beyond the frame");
                return -1;
            }
            const int32_t *e = W + fb + x0;
            int64_t rows = e[1], off = e[3] >> 2;
            if (e[0] != tc || e[2] != nc || rows < 0 || off != x0 + P[NP_DIR_STRIDE] || off + rows * nc > len) {
                PyErr_SetString(PyExc_ValueError, "calc entry of a frame is malformed or out of bounds");
                return -1;
            }
            n[tc] = rows;
            o[tc] = fb + off;
            x0 = off + rows * nc;
        }
        if (!P[NP_HIDDEN] && x0 != len) {
            PyErr_SetString(PyExc_ValueError, "calc trailer chain does not end at the frame end");
            return -1;
        }
    }
    if (P[NP_HIDDEN]) { /* HDRAW closes the chain; only its entry is checked (the rows are not the tokenizer's) */
        if (x0 + P[NP_DIR_STRIDE] > len) {
            PyErr_SetString(PyExc_ValueError, "HDRAW entry beyond the frame");
            return -1;
        }
        const int32_t *e = W + fb + x0;
        int64_t rows = e[1], off = e[3] >> 2;
        if (e[0] != P[NP_T_HDRAW] || e[2] != P[NP_HD_NCOLS] || rows < 0 || off != x0 + P[NP_DIR_STRIDE] ||
            off + rows * P[NP_HD_NCOLS] != len) {
            PyErr_SetString(PyExc_ValueError, "HDRAW entry of a frame is malformed or does not end the frame");
            return -1;
        }
    }
    return 0;
}

/* Version 5, after frame_dir (notes/inputs2/contract.md 1-2, v2a.md 2). With the route summary: asserts that a frame
   with IN_COMBAT = 1 has MAP_NODES rows, writes the AHEAD slots of the frame's ROUTE rows (read before the DROP rule)
   into ah and gives table NP_T_AHEAD 1 row. With the DROP rule, in a frame with IN_COMBAT = 1 and KIND != MAP:
   assertion A1, then MAP_NODES, MAP_EDGES and ROUTE get 0 rows. */
static int frame_v5(const int32_t *W, int64_t fb, const int64_t *P, int64_t *n, int64_t *o, int32_t *ah) {
    const int64_t ta = P[NP_T_AHEAD], tn = P[NP_T_NODES], tr = P[NP_T_CALC + 2];
    const int32_t *g = W + fb + P[NP_GLOB_W0];
    const int in_combat = g[P[NP_G_IN_COMBAT]] == 1;
    n[ta] = 0;
    o[ta] = fb;
    if (P[NP_ROUTE_SUMMARY]) {
        if (in_combat && n[tn] == 0) {
            PyErr_SetString(PyExc_AssertionError, "tokenizer.route_summary: a frame with IN_COMBAT = 1 has no "
                            "MAP_NODES rows (hosts need RESET bit 1)");
            return -1;
        }
        const int64_t ns = P[NP_NCOLS + ta], ncr = P[NP_NCOLS + tr], ncn = P[NP_NCOLS + tn], nr = n[tr];
        const int32_t *route = W + o[tr], *nodes = W + o[tn];
        if (ns > N_AH_MAX) {
            PyErr_SetString(PyExc_ValueError, "AHEAD slot count");
            return -1;
        }
        for (int64_t k = 0; k < ns; k++) ah[k] = (k >= 8 && k <= 28) ? -1 : 0;
        ah[0] = (int32_t)nr;
        for (int64_t r = 0; r < nr; r++) {
            int64_t owner = route[r * ncr + P[NP_CC_OWNER + 2]];
            if (owner < 0 || owner >= n[tn]) {
                PyErr_SetString(PyExc_ValueError, "ROUTE row owner outside MAP_NODES");
                return -1;
            }
            const int32_t *node = nodes + owner * ncn;
            int64_t ty = node[P[NP_MN_PT]], fl = node[P[NP_MN_FLAGS]];
            for (int k = 0; k < N_AH_TYPES; k++) ah[1 + k] += ty == P[NP_PT_IDS + k];
            ah[7] += (fl & P[NP_FL_BOSS]) != 0;
            ah[29] += (fl & P[NP_FL_QUEST]) != 0;
        }
        /* MINV(c): min over the rows with c >= 0, else -1; MAXV(c): -1 without rows or with a row c = -1, else max */
        for (int j = 0; j <= N_AH_PAIRS; j++) {
            int64_t c = j == 0 ? P[NP_RC_TO_BOSS] : P[NP_RC_PAIRS + j - 1], mn = INT64_MAX, mx = INT64_MIN;
            int neg = 0;
            for (int64_t r = 0; r < nr; r++) {
                int64_t v = route[r * ncr + c];
                if (v >= 0 && v < mn) mn = v;
                if (v > mx) mx = v;
                neg |= v == -1;
            }
            int32_t minv = (int32_t)(mn == INT64_MAX ? -1 : mn), maxv = (int32_t)((nr == 0 || neg) ? -1 : mx);
            if (j == 0) {
                ah[8] = minv;
            } else {
                ah[9 + 2 * (j - 1)] = minv;
                ah[10 + 2 * (j - 1)] = maxv;
            }
        }
        n[ta] = 1;
    }
    if (P[NP_DROP] && in_combat && g[P[NP_G_KIND]] != P[NP_KIND_MAP]) {
        static const char *what[3] = {"ACTIONS row %lld has SRC_TABLE", "OPTIONS row %lld has REF_TABLE",
                                      "REFS row %lld has ENT_TABLE"};
        const int64_t tabs[3] = {P[NP_T_ACTIONS], P[NP_T_OPTIONS], P[NP_T_REFS]};
        const int64_t cols[3] = {P[NP_A_SRC_T], P[NP_O_REF_T], P[NP_RF_ET]};
        for (int k = 0; k < 3; k++) {
            const int32_t *c = W + o[tabs[k]];
            const int64_t nc = P[NP_NCOLS + tabs[k]];
            for (int64_t r = 0; r < n[tabs[k]]; r++)
                if (c[r * nc + cols[k]] == tn) {
                    char row[96];
                    snprintf(row, sizeof row, what[k], (long long)r);
                    PyErr_Format(PyExc_AssertionError, "A1: frame step %d (IN_COMBAT = 1, KIND != MAP: map tables "
                                 "dropped): %s = MAP_NODES", (int)W[fb + P[NP_H_STEP]], row);
                    return -1;
                }
        }
        n[tn] = 0;
        n[P[NP_T_EDGES]] = 0;
        n[tr] = 0;
    }
    return 0;
}

/* encode_plan(W, fb, lens, params, counts(B,4) i64 out, rows(B) i64 out, edges(B) i64 out) */
static PyObject *encode_plan(PyObject *self, PyObject *args) {
    PyObject *oW, *ofb, *olen, *oP, *oc, *orow, *oedge;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &oW, &ofb, &olen, &oP, &oc, &orow, &oedge)) return NULL;
    Buf bs[7] = {{0}};
    PyObject *res = NULL;
    if (get_buf(oW, &bs[0], 0, 4, "W") || get_buf(ofb, &bs[1], 0, 8, "fb") || get_buf(olen, &bs[2], 0, 8, "lens") ||
        get_buf(oP, &bs[3], 0, 8, "params") || get_buf(oc, &bs[4], 1, 8, "counts") ||
        get_buf(orow, &bs[5], 1, 8, "rows") || get_buf(oedge, &bs[6], 1, 8, "edges"))
        goto done;
    {
        const int32_t *W = (const int32_t *)bs[0].b.buf;
        int64_t nW = bs[0].b.len / 4, B = bs[1].b.len / 8;
        const int64_t *fb = (const int64_t *)bs[1].b.buf, *lens = (const int64_t *)bs[2].b.buf;
        const int64_t *P = (const int64_t *)bs[3].b.buf;
        int64_t *counts = (int64_t *)bs[4].b.buf, *rows = (int64_t *)bs[5].b.buf, *edges = (int64_t *)bs[6].b.buf;
        if (bs[3].b.len / 8 != NP__END || P[NP_COUNT] != NP__END || P[NP_VERSION] != NATIVE_VERSION) {
            PyErr_SetString(PyExc_ValueError, "native params layout mismatch (rebuild spire._native)");
            goto done;
        }
        if (bs[2].b.len / 8 != B || bs[4].b.len / 8 != 4 * B || bs[5].b.len / 8 != B || bs[6].b.len / 8 != B) {
            PyErr_SetString(PyExc_ValueError, "encode_plan: array sizes");
            goto done;
        }
        for (int64_t i = 0; i < B; i++) {
            int64_t n[N_TAB], o[N_TAB];
            int32_t ah[N_AH_MAX];
            if (frame_dir(W, nW, fb[i], lens[i], P, n, o) || frame_v5(W, fb[i], P, n, o, ah)) goto done;
            int64_t tok = 2, kid = 0;
            for (int s = 0; s < 10; s++) tok += n[P[NP_SEG_TAB + s]];
            for (int s = 10; s < N_KID_END; s++) kid += n[P[NP_SEG_TAB + s]];
            int64_t nv = n[P[NP_T_VARS]], na = n[P[NP_T_ACTIONS]];
            counts[4 * i] = tok;
            counts[4 * i + 1] = kid;
            counts[4 * i + 2] = nv;
            counts[4 * i + 3] = na;
            rows[i] = tok + kid + nv + na;
            edges[i] = 2 * n[P[NP_T_EDGES]];
        }
    }
    Py_INCREF(Py_None);
    res = Py_None;
done:
    rel_bufs(bs, 7);
    return res;
}

/* Isolate opt-in work so it does not enlarge the legacy fill loop's live register
   set. Keep source/target assertions and nonzero SP fields identical to Python. */
#ifdef _MSC_VER
__declspec(noinline)
#else
__attribute__((noinline))
#endif
static int fill_coop(const int64_t *P, const int32_t *W, const int32_t *g,
                     const int64_t *n, const int64_t *o, const int64_t *base,
                     int64_t act_base, int16_t *raw, int16_t *rel) {
    const int64_t width = P[NP_WIDTH], ts = P[NP_T_SEATS], to = P[NP_T_OPTIONS], ta = P[NP_T_ACTIONS];
    const int64_t ns = n[ts], no = n[to], na = n[ta], bs = base[ts], bo = base[to];
    const int64_t ncs = P[NP_NCOLS + ts], nco = P[NP_NCOLS + to], nca = P[NP_NCOLS + ta];
    const int32_t *seats = W + o[ts], *options = W + o[to], *actions = W + o[ta];
    if (P[NP_COOP_FEATURES]) {
        for (int64_t r = 0; r < ns; r++) {
            int64_t seat = seats[r * ncs + P[NP_S_SEAT]];
            int64_t extra = ((int64_t)g[P[NP_G_SEAT_SCREEN] + seat] << 1) |
                            ((int64_t)g[P[NP_G_SEAT_STATUS] + seat] << 5);
            if (extra) {
                int16_t *rr = raw + (bs + r) * width;
                rr[P[NP_D_ACTS]] = sat16((int64_t)rr[P[NP_D_ACTS]] | extra);
            }
        }
        for (int64_t r = 0; r < no; r++) {
            uint32_t votes = ((uint32_t)options[r * nco + P[NP_O_FLAGS]] >> 7) & 15;
            if (votes)
                raw[(bo + r) * width + P[NP_D_VOTE_COUNT]] =
                    (int16_t)((votes & 1) + ((votes >> 1) & 1) + ((votes >> 2) & 1) + ((votes >> 3) & 1));
        }
    }
    if (P[NP_MEND_TARGET]) {
        int has_mend = 0;
        for (int64_t r = 0; r < no; r++) {
            const int32_t *option = options + r * nco;
            has_mend |= option[P[NP_O_KIND]] == P[NP_REST_OPTION] && option[P[NP_O_KEY]] == P[NP_MEND_KEY];
        }
        for (int64_t a = 0; a < na; a++) {
            const int32_t *action = actions + a * nca;
            if (action[P[NP_A_SRC_T]] != to) continue;
            int64_t source = action[P[NP_A_SRC_R]];
            if (source < 0 || source >= no) {
                PyErr_SetString(PyExc_AssertionError, "MEND source outside OPTIONS");
                return -1;
            }
            if (has_mend) {
                const int32_t *option = options + source * nco;
                if (option[P[NP_O_KIND]] == P[NP_REST_OPTION] && option[P[NP_O_KEY]] == P[NP_MEND_KEY]) {
                    int64_t target = action[P[NP_A_AUX]];
                    if (target < 0 || target >= ns || target == action[P[NP_A_SEAT]]) {
                        PyErr_SetString(PyExc_AssertionError, "MEND target must be another seat");
                        return -1;
                    }
                    rel[(act_base + a) * 3 + 1] = wrap16(bs + target);
                }
            }
        }
    }
    return 0;
}

/* encode_fill(W, fb, lens, params, prog(N_ENT*width) i64, progn(N_ENT) i64, poly i64, row0(B+1), edge0(B+1),
 *             types i8 out, raw i16 out (zeroed), rel i16 out, edges i16 out, cond i16 out (zeroed),
 *             act0 i64 out, card_base(B) i32 out) */
static PyObject *encode_fill(PyObject *self, PyObject *args) {
    PyObject *ob[16];
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOO", &ob[0], &ob[1], &ob[2], &ob[3], &ob[4], &ob[5], &ob[6], &ob[7],
                          &ob[8], &ob[9], &ob[10], &ob[11], &ob[12], &ob[13], &ob[14], &ob[15]))
        return NULL;
    static const Py_ssize_t isz[16] = {4, 8, 8, 8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 8, 4};
    static const int wr[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1};
    static const char *nm[16] = {"W", "fb", "lens", "params", "prog", "progn", "poly", "row0", "edge0",
                                 "types", "raw", "rel", "edges", "cond", "act0", "card_base"};
    Buf bs[16] = {{0}};
    PyObject *res = NULL;
    for (int k = 0; k < 16; k++)
        if (get_buf(ob[k], &bs[k], wr[k], isz[k], nm[k])) goto done;
    {
        const int32_t *W = (const int32_t *)bs[0].b.buf;
        int64_t nW = bs[0].b.len / 4, B = bs[1].b.len / 8;
        const int64_t *fb = (const int64_t *)bs[1].b.buf, *lens = (const int64_t *)bs[2].b.buf;
        const int64_t *P = (const int64_t *)bs[3].b.buf, *prog = (const int64_t *)bs[4].b.buf;
        const int64_t *progn = (const int64_t *)bs[5].b.buf, *poly = (const int64_t *)bs[6].b.buf;
        const int64_t *row0 = (const int64_t *)bs[7].b.buf, *edge0 = (const int64_t *)bs[8].b.buf;
        int8_t *types = (int8_t *)bs[9].b.buf;
        int16_t *raw = (int16_t *)bs[10].b.buf, *rel = (int16_t *)bs[11].b.buf, *edges = (int16_t *)bs[12].b.buf;
        int16_t *cond = (int16_t *)bs[13].b.buf;
        int64_t *act0 = (int64_t *)bs[14].b.buf;
        int32_t *card_base = (int32_t *)bs[15].b.buf;
        if (bs[3].b.len / 8 != NP__END || P[NP_COUNT] != NP__END || P[NP_VERSION] != NATIVE_VERSION) {
            PyErr_SetString(PyExc_ValueError, "native params layout mismatch (rebuild spire._native)");
            goto done;
        }
        const int64_t width = P[NP_WIDTH], n_ent = P[NP_N_ENT];
        const int64_t M = bs[9].b.len, E = bs[12].b.len / 6;
        if (bs[2].b.len / 8 != B || bs[7].b.len / 8 != B + 1 || bs[8].b.len / 8 != B + 1 ||
            bs[4].b.len / 8 != n_ent * width || bs[5].b.len / 8 != n_ent || bs[10].b.len / 2 != M * width ||
            bs[11].b.len / 2 != 3 * M || bs[13].b.len / 2 != 6 * B || bs[14].b.len / 8 != B || bs[15].b.len / 4 != B ||
            row0[B] != M || edge0[B] != E) {
            PyErr_SetString(PyExc_ValueError, "encode_fill: array sizes");
            goto done;
        }
        const int64_t ncre_p = P[NP_N_CRE_POLY], nopt_p = P[NP_N_OPT_POLY], nref_p = P[NP_N_REF_POLY];
        const int64_t *cre_off = poly, *cre_rows = poly + ncre_p, *opt_off = poly + 2 * ncre_p;
        const int64_t *opt_rows = opt_off + nopt_p, *ref_off = opt_rows + nopt_p, *ref_rows = ref_off + nref_p;
        const int64_t T_SEATS = P[NP_T_SEATS], T_CARDS = P[NP_T_CARDS], T_VARS = P[NP_T_VARS];
        const int64_t T_CRE = P[NP_T_CREATURES], T_POW = P[NP_T_POWERS], T_INT = P[NP_T_INTENTS];
        const int64_t T_RELICS = P[NP_T_RELICS], T_POTIONS = P[NP_T_POTIONS], T_NODES = P[NP_T_NODES];
        const int64_t T_EDGES = P[NP_T_EDGES], T_OPT = P[NP_T_OPTIONS], T_REFS = P[NP_T_REFS];
        const int64_t T_ACT = P[NP_T_ACTIONS];
        const int64_t lv = P[NP_LV];

        const int64_t T_AHEAD = P[NP_T_AHEAD];
        for (int64_t i = 0; i < B; i++) {
            int64_t n[N_TAB], o[N_TAB], base[N_TAB];
            int32_t ah[N_AH_MAX];
            if (frame_dir(W, nW, fb[i], lens[i], P, n, o) || frame_v5(W, fb[i], P, n, o, ah)) goto done;
            const int32_t *g = W + fb[i] + P[NP_GLOB_W0];
            const int64_t R = row0[i];
            /* segment layout (pooled): GLOBAL, DECISION, token tables, kid tables, VARS, ACTIONS */
            for (int t = 0; t < N_TAB; t++) base[t] = -1;
            int64_t s = 2, seg0[N_SEG];
            for (int k = 0; k < N_SEG; k++) {
                int64_t t = P[NP_SEG_TAB + k];
                seg0[k] = s;
                if (k < N_KID_END) base[t] = s; /* VARS / ACTIONS keep base -1 (reference semantics) */
                s += n[t];
            }
            const int64_t var_base = seg0[N_KID_END], act_base = seg0[N_KID_END + 1], total = s;
            if (R + total != row0[i + 1] || edge0[i] + 2 * n[T_EDGES] != edge0[i + 1]) {
                PyErr_SetString(PyExc_ValueError, "encode_fill: row / edge offsets do not match the frames");
                goto done;
            }
            int16_t *rr0 = raw + R * width;
            int16_t *rl0 = rel + R * 3;
            for (int64_t r = 0; r < 3 * total; r++) rl0[r] = -1;
            /* types */
            types[R] = (int8_t)P[NP_ENT_GLOBAL];
            types[R + 1] = (int8_t)P[NP_ENT_DECISION];
            for (int k = 0; k < N_SEG; k++) {
                int64_t t = P[NP_SEG_TAB + k];
                memset(types + R + seg0[k], (int)P[NP_SEG_ENT + k], (size_t)n[t]);
            }
            /* GLOBAL / DECISION rows */
            {
                int64_t e = P[NP_ENT_GLOBAL];
                const int64_t *pc = prog + e * width;
                for (int64_t j = 0; j < progn[e]; j++) rr0[j] = sat16(g[pc[j]]);
                rr0[P[NP_D_KIND_SUB]] = sat16((int64_t)g[P[NP_G_KIND]] * 16 + clip64(g[P[NP_G_SUB]], 0, 15));
                e = P[NP_ENT_DECISION];
                pc = prog + e * width;
                int16_t *rd = rr0 + width;
                for (int64_t j = 0; j < progn[e]; j++) rd[j] = sat16(g[pc[j]]);
                int64_t cat = g[P[NP_G_PCAT]], sid = g[P[NP_G_PSID]];
                if (cat > 0 && cat < nref_p && ref_rows[cat] > 0 && sid > 0) rd[P[NP_D_PROMPT]] = sat16(ref_off[cat] + sid);
            }
            /* slot columns of every table row */
            for (int k = 0; k < N_SEG; k++) {
                int64_t t = P[NP_SEG_TAB + k], e = P[NP_SEG_ENT + k], nc = P[NP_NCOLS + t], kk = progn[e];
                const int64_t *pc = prog + e * width;
                const int32_t *src = t == T_AHEAD ? ah : W + o[t]; /* the AHEAD row lives in ah (frame_v5) */
                int16_t *rr = rr0 + seg0[k] * width;
                for (int64_t r = 0; r < n[t]; r++, src += nc, rr += width)
                    for (int64_t j = 0; j < kk; j++) rr[j] = sat16(src[pc[j]]);
            }
            const int64_t ns = n[T_SEATS], bs_ = base[T_SEATS], bcr = base[T_CRE], ncre = n[T_CRE];
            const int32_t *seats = W + o[T_SEATS], *cre = W + o[T_CRE], *acts = W + o[T_ACT];
            const int64_t nc_seat = P[NP_NCOLS + T_SEATS], nc_cre = P[NP_NCOLS + T_CRE], nc_act = P[NP_NCOLS + T_ACT];
            const int64_t na = n[T_ACT];
            /* SEATS: "acts" bit, creature relation */
            for (int64_t r = 0; r < ns; r++) {
                int16_t *rr = rr0 + (bs_ + r) * width;
                int hit = 0;
                for (int64_t a = 0; a < na && !hit; a++) hit = acts[a * nc_act + P[NP_A_SEAT]] == r;
                rr[P[NP_D_ACTS]] = (int16_t)hit;
                int64_t crow = seats[r * nc_seat + P[NP_S_CROW]];
                rl0[(bs_ + r) * 3 + 1] = wrap16(crow >= 0 ? bcr + crow : -1);
            }
            /* CARDS */
            {
                const int32_t *c = W + o[T_CARDS];
                const int64_t nc = P[NP_NCOLS + T_CARDS], b0 = base[T_CARDS];
                for (int64_t r = 0; r < n[T_CARDS]; r++, c += nc) {
                    int16_t *rr = rr0 + (b0 + r) * width;
                    int64_t up = c[P[NP_K_UP]], zone = c[P[NP_K_ZONE]], aux = c[P[NP_K_AUX]], lr = c[P[NP_K_LINK]];
                    rr[P[NP_D_C_STATIC]] = sat16(P[NP_ST_CARD] + (int64_t)c[P[NP_K_ID]] * lv + (up < lv - 1 ? up : lv - 1));
                    rr[P[NP_D_C_PILE]] = sat16(zone == P[NP_ZONE_CHOICE] ? aux : 0);
                    rr[P[NP_D_C_GROUP]] = sat16(zone == P[NP_ZONE_OFFER] ? aux : -1);
                    rl0[(b0 + r) * 3 + 1] = wrap16(lr >= 0 ? b0 + lr : -1);
                }
            }
            /* CREATURES */
            for (int64_t r = 0; r < ncre; r++) {
                const int32_t *c = cre + r * nc_cre;
                int16_t *rr = rr0 + (bcr + r) * width;
                int64_t kind = c[P[NP_CR_KIND]], mid = c[P[NP_CR_MODEL]], hp = c[P[NP_CR_HP]], poi = c[P[NP_CR_POISON]];
                rr[P[NP_D_CR_MODEL]] = sat16(resolve(cre_off, cre_rows, ncre_p, kind, mid));
                rr[P[NP_D_CR_STATIC]] = sat16(kind == P[NP_PLAYER] ? 0 : P[NP_ST_MONSTER] + mid);
                rr[P[NP_D_CR_LETHAL]] = (int16_t)(poi > 0 && poi >= hp && hp > 0);
            }
            /* RELICS / POTIONS / MAP_NODES */
            {
                const int32_t *c = W + o[T_RELICS];
                int64_t nc = P[NP_NCOLS + T_RELICS], b0 = base[T_RELICS];
                for (int64_t r = 0; r < n[T_RELICS]; r++)
                    rr0[(b0 + r) * width + P[NP_D_RELIC]] = sat16(P[NP_ST_RELIC] + (int64_t)c[r * nc + P[NP_RL_ID]]);
                c = W + o[T_POTIONS];
                nc = P[NP_NCOLS + T_POTIONS];
                b0 = base[T_POTIONS];
                for (int64_t r = 0; r < n[T_POTIONS]; r++)
                    rr0[(b0 + r) * width + P[NP_D_POTION]] = sat16(P[NP_ST_POTION] + (int64_t)c[r * nc + P[NP_PT_ID]]);
                c = W + o[T_NODES];
                nc = P[NP_NCOLS + T_NODES];
                b0 = base[T_NODES];
                int64_t cur = g[P[NP_G_CUR_ROW]];
                for (int64_t r = 0; r < n[T_NODES]; r++)
                    rr0[(b0 + r) * width + P[NP_D_DROW]] = sat16((int64_t)c[r * nc + P[NP_MN_ROW]] - cur);
            }
            /* OPTIONS */
            {
                const int32_t *c = W + o[T_OPT];
                int64_t nc = P[NP_NCOLS + T_OPT], b0 = base[T_OPT];
                for (int64_t r = 0; r < n[T_OPT]; r++, c += nc) {
                    rr0[(b0 + r) * width + P[NP_D_KEY]] =
                        sat16(resolve(opt_off, opt_rows, nopt_p, c[P[NP_O_KIND]], c[P[NP_O_KEY]]));
                    rl0[(b0 + r) * 3 + 1] = wrap16(ref(base, c[P[NP_O_REF_T]], c[P[NP_O_REF_R]]));
                }
            }
            /* POWERS */
            {
                const int32_t *c = W + o[T_POW];
                int64_t nc = P[NP_NCOLS + T_POW], b0 = base[T_POW];
                for (int64_t r = 0; r < n[T_POW]; r++, c += nc) {
                    int16_t *rr = rr0 + (b0 + r) * width;
                    int64_t app = c[P[NP_P_APPLIER]];
                    rr[P[NP_D_PW_STATIC]] = sat16(P[NP_ST_POWER] + (int64_t)c[P[NP_P_ID]]);
                    int64_t ak = 0;
                    if (app >= 0 && ncre > 0) ak = cre[clip64(app, 0, ncre - 1) * nc_cre + P[NP_CR_KIND]];
                    rr[P[NP_D_PW_AK]] = sat16(ak);
                    rl0[(b0 + r) * 3] = wrap16(bcr + c[P[NP_P_OWNER]]);
                    rl0[(b0 + r) * 3 + 1] = wrap16(app >= 0 ? bcr + app : -1);
                }
            }
            /* INTENTS */
            {
                const int32_t *c = W + o[T_INT];
                int64_t nc = P[NP_NCOLS + T_INT], b0 = base[T_INT];
                for (int64_t r = 0; r < n[T_INT]; r++)
                    rl0[(b0 + r) * 3] = wrap16(bcr + c[r * nc + P[NP_I_OWNER]]);
            }
            /* FORECAST: kids of their creature */
            {
                const int64_t tf = P[NP_T_FORECAST];
                const int32_t *c = W + o[tf];
                int64_t nc = P[NP_NCOLS + tf], b0 = base[tf];
                for (int64_t r = 0; r < n[tf]; r++)
                    rl0[(b0 + r) * 3] = wrap16(bcr + c[r * nc + P[NP_FC_OWNER]]);
            }
            /* TURN / FOE / ROUTE: kids of the creature / creature / map node of their OWNER_ROW */
            for (int k = 0; k < N_CALC; k++) {
                const int64_t tc = P[NP_T_CALC + k], oc = P[NP_CC_OWNER + k], ob = base[P[NP_CC_OWNER_T + k]];
                const int32_t *c = W + o[tc];
                int64_t nc = P[NP_NCOLS + tc], b0 = base[tc];
                for (int64_t r = 0; r < n[tc]; r++)
                    rl0[(b0 + r) * 3] = wrap16(ob + c[r * nc + oc]);
            }
            /* AHEAD: the kid of the sample's GLOBAL token (row 0) */
            if (n[T_AHEAD]) rl0[base[T_AHEAD] * 3] = 0;
            /* REFS */
            {
                const int32_t *c = W + o[T_REFS];
                int64_t nc = P[NP_NCOLS + T_REFS], b0 = base[T_REFS];
                for (int64_t r = 0; r < n[T_REFS]; r++, c += nc) {
                    rr0[(b0 + r) * width + P[NP_D_REF]] =
                        sat16(resolve(ref_off, ref_rows, nref_p, c[P[NP_RF_CAT]], c[P[NP_RF_VOC]]));
                    rl0[(b0 + r) * 3] = wrap16(ref(base, c[P[NP_RF_OT]], c[P[NP_RF_OR]]));
                    rl0[(b0 + r) * 3 + 1] = wrap16(ref(base, c[P[NP_RF_ET]], c[P[NP_RF_ER]]));
                }
            }
            /* VARS: per-target rows go to the first action with the same (src_table, src_row, tgt) key */
            {
                const int32_t *c = W + o[T_VARS];
                int64_t nc = P[NP_NCOLS + T_VARS];
                for (int64_t r = 0; r < n[T_VARS]; r++, c += nc) {
                    int64_t ot = c[P[NP_V_OT]], orow = c[P[NP_V_OR]], vt = c[P[NP_V_TGT]];
                    int64_t dst = ref(base, ot, orow);
                    if (vt >= 0) {
                        int64_t vkey = (ot * 4096 + orow) * 64 + vt;
                        for (int64_t a = 0; a < na; a++) {
                            const int32_t *ar = acts + a * nc_act;
                            int64_t st = ar[P[NP_A_SRC_T]], tg = ar[P[NP_A_TGT]];
                            int64_t akey = (st >= 0 && tg >= 0) ? (st * 4096 + ar[P[NP_A_SRC_R]]) * 64 + tg : -1;
                            if (akey == vkey) {
                                dst = act_base + a;
                                break;
                            }
                        }
                    }
                    rl0[(var_base + r) * 3] = wrap16(dst);
                }
            }
            /* ACTIONS */
            for (int64_t a = 0; a < na; a++) {
                const int32_t *ar = acts + a * nc_act;
                int64_t tg = ar[P[NP_A_TGT]], seat = ar[P[NP_A_SEAT]];
                int16_t *rl = rl0 + (act_base + a) * 3;
                rl[0] = wrap16(ref(base, ar[P[NP_A_SRC_T]], ar[P[NP_A_SRC_R]]));
                rl[1] = wrap16(tg >= 0 ? bcr + tg : -1);
                rl[2] = wrap16(seat < ns ? bs_ + seat : -1);
            }
            act0[i] = o[T_ACT];
            card_base[i] = (int32_t)base[T_CARDS];
            /* MAP_EDGES: (from, to, 0) per edge, then (to, from, 1) */
            {
                const int32_t *c = W + o[T_EDGES];
                int64_t nc = P[NP_NCOLS + T_EDGES], ne = n[T_EDGES], bn = base[T_NODES];
                int16_t *eo = edges + edge0[i] * 3, *ei = eo + ne * 3;
                for (int64_t r = 0; r < ne; r++, c += nc, eo += 3, ei += 3) {
                    int64_t a = (int64_t)c[P[NP_E_FROM]] + bn, b = (int64_t)c[P[NP_E_TO]] + bn;
                    eo[0] = wrap16(a);
                    eo[1] = wrap16(b);
                    eo[2] = 0;
                    ei[0] = wrap16(b);
                    ei[1] = wrap16(a);
                    ei[2] = 1;
                }
            }
            /* cond: ascension, n_seats, character of seats 0..3 */
            {
                int16_t *cd = cond + 6 * i;
                cd[0] = wrap16(g[P[NP_G_ASC]]);
                cd[1] = wrap16(g[P[NP_G_NSEATS]]);
                for (int64_t r = 0; r < ns && r < 4; r++) cd[2 + r] = wrap16(seats[r * nc_seat + P[NP_S_CHAR]]);
            }
            if ((P[NP_COOP_FEATURES] || P[NP_MEND_TARGET]) &&
                fill_coop(P, W, g, n, o, base, act_base, rr0, rl0)) goto done;
        }
    }
    Py_INCREF(Py_None);
    res = Py_None;
done:
    rel_bufs(bs, 16);
    return res;
}

/* collate_fill(types i8, raw i16, rel i16, edges i16, cond i16, counts i64(B,4), row0 i64(B+1), edge0 i64(B+1),
 *              lo, hi, T, A, Nk, Nv,
 *              out: types i64(M), raw i16(M,W), rel i64(M,3), dest i64(M), edges i64(Ne,3), tok_mask u8(Bp,T),
 *                   act_mask u8(Bp,A), kid_mask u8(Nk), var_mask u8(Nv), cond i64(Bp,6))
 * Tokenizer.collate of frames lo:hi of an EncodedBatch into fixed-shape buffers (every element written). */
static PyObject *collate_fill(PyObject *self, PyObject *args) {
    PyObject *ob[18];
    long long lo, hi, T, A, Nk, Nv;
    if (!PyArg_ParseTuple(args, "OOOOOOOOLLLLLLOOOOOOOOOO", &ob[0], &ob[1], &ob[2], &ob[3], &ob[4], &ob[5], &ob[6],
                          &ob[7], &lo, &hi, &T, &A, &Nk, &Nv, &ob[8], &ob[9], &ob[10], &ob[11], &ob[12], &ob[13],
                          &ob[14], &ob[15], &ob[16], &ob[17]))
        return NULL;
    static const Py_ssize_t isz[18] = {1, 2, 2, 2, 2, 8, 8, 8, 8, 2, 8, 8, 8, 1, 1, 1, 1, 8};
    static const int wr[18] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    static const char *nm[18] = {"types", "raw", "rel", "edges", "cond", "counts", "row0", "edge0", "o_types",
                                 "o_raw", "o_rel", "o_dest", "o_edges", "o_tok_mask", "o_act_mask", "o_kid_mask",
                                 "o_var_mask", "o_cond"};
    Buf bs[18] = {{0}};
    PyObject *res = NULL;
    for (int k = 0; k < 18; k++)
        if (get_buf(ob[k], &bs[k], wr[k], isz[k], nm[k])) goto done;
    {
        const int8_t *types = (const int8_t *)bs[0].b.buf;
        const int16_t *raw = (const int16_t *)bs[1].b.buf, *rel = (const int16_t *)bs[2].b.buf;
        const int16_t *edges = (const int16_t *)bs[3].b.buf, *cond = (const int16_t *)bs[4].b.buf;
        const int64_t *counts = (const int64_t *)bs[5].b.buf, *row0 = (const int64_t *)bs[6].b.buf;
        const int64_t *edge0 = (const int64_t *)bs[7].b.buf;
        int64_t *o_types = (int64_t *)bs[8].b.buf, *o_rel = (int64_t *)bs[10].b.buf, *o_dest = (int64_t *)bs[11].b.buf;
        int16_t *o_raw = (int16_t *)bs[9].b.buf;
        int64_t *o_edges = (int64_t *)bs[12].b.buf, *o_cond = (int64_t *)bs[17].b.buf;
        uint8_t *tok_mask = (uint8_t *)bs[13].b.buf, *act_mask = (uint8_t *)bs[14].b.buf;
        uint8_t *kid_mask = (uint8_t *)bs[15].b.buf, *var_mask = (uint8_t *)bs[16].b.buf;
        const int64_t nB = bs[6].b.len / 8 - 1, M = bs[8].b.len / 8, Ne = bs[12].b.len / 24;
        const int64_t Bp = bs[17].b.len / 48;
        if (M <= 0 || bs[9].b.len % (2 * M)) {
            PyErr_SetString(PyExc_ValueError, "collate_fill: output raw shape");
            goto done;
        }
        const int64_t W = bs[9].b.len / 2 / M;
        if (lo < 0 || hi > nB || hi <= lo || bs[5].b.len / 8 != 4 * nB || bs[7].b.len / 8 != nB + 1 ||
            bs[1].b.len / 2 != bs[0].b.len * W || bs[2].b.len / 2 != 3 * bs[0].b.len || bs[4].b.len / 2 != 6 * nB ||
            bs[10].b.len / 8 != 3 * M || bs[11].b.len / 8 != M || bs[13].b.len != Bp * T || bs[14].b.len != Bp * A ||
            bs[15].b.len != Nk || bs[16].b.len != Nv || Bp < hi - lo) {
            PyErr_SetString(PyExc_ValueError, "collate_fill: array sizes");
            goto done;
        }
        const int64_t b = hi - lo, BT = Bp * T, N = BT + Nk + Nv + Bp * A;
        const int64_t R0 = row0[lo], R1 = row0[hi], total = R1 - R0, E0 = edge0[lo], E1 = edge0[hi], se = E1 - E0;
        if (total > M || se > Ne) {
            PyErr_SetString(PyExc_ValueError, "collate_fill: batch does not fit the shape");
            goto done;
        }
        /* dest: tokens j*T.., kids / vars packed, actions j*A.. */
        int64_t ck = 0, cv = 0;
        for (int64_t j = 0; j < b; j++) {
            const int64_t *c = counts + 4 * (lo + j);
            int64_t p = row0[lo + j] - R0;
            int64_t g0 = j * T, g1 = BT + ck, g2 = BT + Nk + cv, g3 = BT + Nk + Nv + j * A;
            for (int64_t x = 0; x < c[0]; x++) o_dest[p++] = g0 + x;
            for (int64_t x = 0; x < c[1]; x++) o_dest[p++] = g1 + x;
            for (int64_t x = 0; x < c[2]; x++) o_dest[p++] = g2 + x;
            for (int64_t x = 0; x < c[3]; x++) o_dest[p++] = g3 + x;
            ck += c[1];
            cv += c[2];
            if (p != row0[lo + j + 1] - R0) {
                PyErr_SetString(PyExc_ValueError, "collate_fill: counts do not match the row offsets");
                goto done;
            }
        }
        for (int64_t p = total; p < M; p++) o_dest[p] = N + 1;
        for (int64_t p = 0; p < total; p++) o_types[p] = types[R0 + p];
        for (int64_t p = total; p < M; p++) o_types[p] = 0;
        memcpy(o_raw, raw + R0 * W, (size_t)(total * W) * 2);
        memset(o_raw + total * W, 0, (size_t)((M - total) * W) * 2);
        for (int64_t j = 0; j < b; j++) {
            int64_t a = row0[lo + j] - R0, e = row0[lo + j + 1] - R0;
            for (int64_t p = a; p < e; p++)
                for (int c = 0; c < 3; c++) {
                    int64_t r = rel[(R0 + p) * 3 + c], v;
                    if (r < 0) {
                        v = N;
                    } else {
                        int64_t x = r + a;
                        if (x > total) {
                            PyErr_SetString(PyExc_IndexError, "collate_fill: relation outside the batch rows");
                            goto done;
                        }
                        v = x == total ? N : o_dest[x];
                    }
                    o_rel[p * 3 + c] = v;
                }
        }
        for (int64_t p = total; p < M; p++) o_rel[3 * p] = o_rel[3 * p + 1] = o_rel[3 * p + 2] = N;
        for (int64_t k = 0; k < Ne; k++) {
            o_edges[3 * k] = o_edges[3 * k + 1] = N;
            o_edges[3 * k + 2] = 0;
        }
        for (int64_t j = 0; j < b; j++) {
            int64_t off = row0[lo + j] - R0;
            for (int64_t k = edge0[lo + j]; k < edge0[lo + j + 1]; k++) {
                int64_t x0 = edges[3 * k] + off, x1 = edges[3 * k + 1] + off, q = k - E0;
                if (x0 < -M || x0 >= M || x1 < -M || x1 >= M) { /* numpy indexing: negative counts from the end */
                    PyErr_SetString(PyExc_IndexError, "collate_fill: edge outside the batch rows");
                    goto done;
                }
                if (x0 < 0) x0 += M;
                if (x1 < 0) x1 += M;
                o_edges[3 * q] = o_dest[x0];
                o_edges[3 * q + 1] = o_dest[x1];
                o_edges[3 * q + 2] = edges[3 * k + 2];
            }
        }
        for (int64_t j = 0; j < Bp; j++) {
            if (j < b) {
                const int64_t *c = counts + 4 * (lo + j);
                for (int k = 0; k < 6; k++) o_cond[6 * j + k] = cond[6 * (lo + j) + k];
                for (int64_t t = 0; t < T; t++) tok_mask[j * T + t] = (uint8_t)(t < c[0]);
                for (int64_t t = 0; t < A; t++) act_mask[j * A + t] = (uint8_t)(t < c[3]);
            } else { /* empty filler samples attend to their (zero) global token: no NaN rows */
                for (int k = 0; k < 6; k++) o_cond[6 * j + k] = 0;
                for (int64_t t = 0; t < T; t++) tok_mask[j * T + t] = (uint8_t)(t == 0);
                for (int64_t t = 0; t < A; t++) act_mask[j * A + t] = 0;
            }
        }
        for (int64_t x = 0; x < Nk; x++) kid_mask[x] = (uint8_t)(x < ck);
        for (int64_t x = 0; x < Nv; x++) var_mask[x] = (uint8_t)(x < cv);
    }
    Py_INCREF(Py_None);
    res = Py_None;
done:
    rel_bufs(bs, 18);
    return res;
}

/* collate_idx(types i8, raw i16, rel i16, edges i16, cond i16, counts i64(S,4), row0 i64(S+1), edge0 i64(S+1),
 *             idx i64(b), T, A, Nk, Nv, out: as collate_fill)
 * Tokenizer.collate([block[i] for i in idx], shape=...) of samples stored back to back (a learner iteration:
 * every sample of every segment concatenated) into fixed-shape buffers; every element written. Identical
 * output to collate_fill on a contiguous range and to Tokenizer.collate (tests/test_learner_fast.py).
 * The copy loops run without the GIL (the learner thread must not stall an actor thread of the same
 * process); errors found there are raised after it is re-acquired. */
static PyObject *collate_idx(PyObject *self, PyObject *args) {
    PyObject *ob[19];
    long long T, A, Nk, Nv;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOLLLLOOOOOOOOOO", &ob[0], &ob[1], &ob[2], &ob[3], &ob[4], &ob[5], &ob[6],
                          &ob[7], &ob[8], &T, &A, &Nk, &Nv, &ob[9], &ob[10], &ob[11], &ob[12], &ob[13], &ob[14],
                          &ob[15], &ob[16], &ob[17], &ob[18]))
        return NULL;
    static const Py_ssize_t isz[19] = {1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 2, 8, 8, 8, 1, 1, 1, 1, 8};
    static const int wr[19] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    static const char *nm[19] = {"types", "raw", "rel", "edges", "cond", "counts", "row0", "edge0", "idx", "o_types",
                                 "o_raw", "o_rel", "o_dest", "o_edges", "o_tok_mask", "o_act_mask", "o_kid_mask",
                                 "o_var_mask", "o_cond"};
    Buf bs[19] = {{0}};
    PyObject *res = NULL;
    const char *err = NULL;
    PyObject *err_type = PyExc_ValueError;
    int64_t *p0 = NULL;
    for (int k = 0; k < 19; k++)
        if (get_buf(ob[k], &bs[k], wr[k], isz[k], nm[k])) goto done;
    {
        const int8_t *types = (const int8_t *)bs[0].b.buf;
        const int16_t *raw = (const int16_t *)bs[1].b.buf, *rel = (const int16_t *)bs[2].b.buf;
        const int16_t *edges = (const int16_t *)bs[3].b.buf, *cond = (const int16_t *)bs[4].b.buf;
        const int64_t *counts = (const int64_t *)bs[5].b.buf, *row0 = (const int64_t *)bs[6].b.buf;
        const int64_t *edge0 = (const int64_t *)bs[7].b.buf, *idx = (const int64_t *)bs[8].b.buf;
        int64_t *o_types = (int64_t *)bs[9].b.buf, *o_rel = (int64_t *)bs[11].b.buf, *o_dest = (int64_t *)bs[12].b.buf;
        int16_t *o_raw = (int16_t *)bs[10].b.buf;
        int64_t *o_edges = (int64_t *)bs[13].b.buf, *o_cond = (int64_t *)bs[18].b.buf;
        uint8_t *tok_mask = (uint8_t *)bs[14].b.buf, *act_mask = (uint8_t *)bs[15].b.buf;
        uint8_t *kid_mask = (uint8_t *)bs[16].b.buf, *var_mask = (uint8_t *)bs[17].b.buf;
        const int64_t nS = bs[6].b.len / 8 - 1, b = bs[8].b.len / 8, M = bs[9].b.len / 8, Ne = bs[13].b.len / 24;
        const int64_t Bp = bs[18].b.len / 48;
        if (M <= 0 || bs[10].b.len % (2 * M)) {
            PyErr_SetString(PyExc_ValueError, "collate_idx: output raw shape");
            goto done;
        }
        const int64_t W = bs[10].b.len / 2 / M;
        if (nS < 0 || b <= 0 || bs[5].b.len / 8 != 4 * nS || bs[7].b.len / 8 != nS + 1 ||
            bs[1].b.len / 2 != bs[0].b.len * W || bs[2].b.len / 2 != 3 * bs[0].b.len || bs[4].b.len / 2 != 6 * nS ||
            bs[11].b.len / 8 != 3 * M || bs[12].b.len / 8 != M || bs[14].b.len != Bp * T || bs[15].b.len != Bp * A ||
            bs[16].b.len != Nk || bs[17].b.len != Nv || Bp < b) {
            PyErr_SetString(PyExc_ValueError, "collate_idx: array sizes");
            goto done;
        }
        p0 = (int64_t *)PyMem_Malloc((size_t)(b + 1) * sizeof(int64_t));
        if (p0 == NULL) {
            PyErr_NoMemory();
            goto done;
        }
        Py_BEGIN_ALLOW_THREADS
        int64_t total = 0, se = 0;
        for (int64_t j = 0; j < b && !err; j++) {
            int64_t s = idx[j];
            if (s < 0 || s >= nS) { err = "collate_idx: sample index out of range"; err_type = PyExc_IndexError; break; }
            const int64_t *c = counts + 4 * s;
            if (row0[s + 1] - row0[s] != c[0] + c[1] + c[2] + c[3]) { err = "collate_idx: counts do not match the row offsets"; break; }
            p0[j] = total;
            total += row0[s + 1] - row0[s];
            se += edge0[s + 1] - edge0[s];
        }
        p0[b] = total;
        if (!err && (total > M || se > Ne)) err = "collate_idx: batch does not fit the shape";
        const int64_t BT = Bp * T, N = BT + Nk + Nv + Bp * A;
        int64_t ck = 0, cv = 0;
        if (!err) {
            for (int64_t j = 0; j < b; j++) {
                const int64_t *c = counts + 4 * idx[j];
                int64_t p = p0[j];
                int64_t g0 = j * T, g1 = BT + ck, g2 = BT + Nk + cv, g3 = BT + Nk + Nv + j * A;
                for (int64_t x = 0; x < c[0]; x++) o_dest[p++] = g0 + x;
                for (int64_t x = 0; x < c[1]; x++) o_dest[p++] = g1 + x;
                for (int64_t x = 0; x < c[2]; x++) o_dest[p++] = g2 + x;
                for (int64_t x = 0; x < c[3]; x++) o_dest[p++] = g3 + x;
                ck += c[1];
                cv += c[2];
            }
            for (int64_t p = total; p < M; p++) o_dest[p] = N + 1;
            for (int64_t j = 0; j < b; j++) {
                int64_t s = idx[j], n = row0[s + 1] - row0[s];
                for (int64_t q = 0; q < n; q++) o_types[p0[j] + q] = types[row0[s] + q];
                memcpy(o_raw + p0[j] * W, raw + row0[s] * W, (size_t)(n * W) * 2);
            }
            for (int64_t p = total; p < M; p++) o_types[p] = 0;
            memset(o_raw + total * W, 0, (size_t)((M - total) * W) * 2);
        }
        for (int64_t j = 0; j < b && !err; j++) {
            int64_t s = idx[j], a = p0[j], n = row0[s + 1] - row0[s];
            for (int64_t q = 0; q < n && !err; q++)
                for (int c = 0; c < 3; c++) {
                    int64_t r = rel[(row0[s] + q) * 3 + c], v;
                    if (r < 0) {
                        v = N;
                    } else {
                        int64_t x = r + a;
                        if (x > total) { err = "collate_idx: relation outside the batch rows"; err_type = PyExc_IndexError; break; }
                        v = x == total ? N : o_dest[x];
                    }
                    o_rel[(a + q) * 3 + c] = v;
                }
        }
        if (!err) {
            for (int64_t p = total; p < M; p++) o_rel[3 * p] = o_rel[3 * p + 1] = o_rel[3 * p + 2] = N;
            for (int64_t k = 0; k < Ne; k++) {
                o_edges[3 * k] = o_edges[3 * k + 1] = N;
                o_edges[3 * k + 2] = 0;
            }
        }
        int64_t q = 0;
        for (int64_t j = 0; j < b && !err; j++) {
            int64_t s = idx[j], off = p0[j];
            for (int64_t k = edge0[s]; k < edge0[s + 1]; k++, q++) {
                int64_t x0 = edges[3 * k] + off, x1 = edges[3 * k + 1] + off;
                if (x0 < -M || x0 >= M || x1 < -M || x1 >= M) { /* numpy indexing: negative counts from the end */
                    err = "collate_idx: edge outside the batch rows";
                    err_type = PyExc_IndexError;
                    break;
                }
                if (x0 < 0) x0 += M;
                if (x1 < 0) x1 += M;
                o_edges[3 * q] = o_dest[x0];
                o_edges[3 * q + 1] = o_dest[x1];
                o_edges[3 * q + 2] = edges[3 * k + 2];
            }
        }
        if (!err) {
            for (int64_t j = 0; j < Bp; j++) {
                if (j < b) {
                    const int64_t s = idx[j], *c = counts + 4 * s;
                    for (int k = 0; k < 6; k++) o_cond[6 * j + k] = cond[6 * s + k];
                    for (int64_t t = 0; t < T; t++) tok_mask[j * T + t] = (uint8_t)(t < c[0]);
                    for (int64_t t = 0; t < A; t++) act_mask[j * A + t] = (uint8_t)(t < c[3]);
                } else { /* empty filler samples attend to their (zero) global token: no NaN rows */
                    for (int k = 0; k < 6; k++) o_cond[6 * j + k] = 0;
                    for (int64_t t = 0; t < T; t++) tok_mask[j * T + t] = (uint8_t)(t == 0);
                    for (int64_t t = 0; t < A; t++) act_mask[j * A + t] = 0;
                }
            }
            for (int64_t x = 0; x < Nk; x++) kid_mask[x] = (uint8_t)(x < ck);
            for (int64_t x = 0; x < Nv; x++) var_mask[x] = (uint8_t)(x < cv);
        }
        Py_END_ALLOW_THREADS
        if (err) {
            PyErr_SetString(err_type, err);
            goto done;
        }
    }
    Py_INCREF(Py_None);
    res = Py_None;
done:
    if (p0 != NULL) PyMem_Free(p0);
    rel_bufs(bs, 19);
    return res;
}

/* concat_into(arrays, out) -> items: copy a sequence of C-contiguous arrays with out's item size back
   to back into the C-contiguous buffer `out` (np.concatenate of many small arrays without numpy's per-array
   overhead; used to compact rollout segments, spire.tokenize.SampleBlock.from_samples). The caller sizes
   `out` exactly; a size mismatch raises ValueError. */
static PyObject *concat_into(PyObject *self, PyObject *args) {
    PyObject *seq, *out;
    if (!PyArg_ParseTuple(args, "OO", &seq, &out)) return NULL;
    PyObject *fast = PySequence_Fast(seq, "concat_into: arrays must be a sequence");
    if (fast == NULL) return NULL;
    Buf ob = {0};
    if (PyObject_GetBuffer(out, &ob.b, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) { Py_DECREF(fast); return NULL; }
    ob.ok = 1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast), pos = 0;
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *it = items[i];
        if (PyArray_Check(it) && PyArray_IS_C_CONTIGUOUS((PyArrayObject *)it)) {
            /* fast path: an ndarray (exporting a buffer costs ~0.2 us per array) */
            PyArrayObject *a = (PyArrayObject *)it;
            Py_ssize_t len = (Py_ssize_t)PyArray_NBYTES(a);
            if ((Py_ssize_t)PyArray_ITEMSIZE(a) != ob.b.itemsize || pos + len > ob.b.len) {
                PyErr_Format(PyExc_ValueError, "concat_into: item %zd (itemsize %zd, %zd bytes) does not fit (itemsize %zd, %zd of %zd bytes used)",
                             i, (Py_ssize_t)PyArray_ITEMSIZE(a), len, ob.b.itemsize, pos, ob.b.len);
                goto fail;
            }
            if (len) memcpy((char *)ob.b.buf + pos, PyArray_DATA(a), (size_t)len);
            pos += len;
            continue;
        }
        Py_buffer b;
        if (PyObject_GetBuffer(it, &b, PyBUF_C_CONTIGUOUS) < 0) goto fail;
        if (b.itemsize != ob.b.itemsize || pos + b.len > ob.b.len) {
            PyErr_Format(PyExc_ValueError, "concat_into: item %zd (itemsize %zd, %zd bytes) does not fit (itemsize %zd, %zd of %zd bytes used)",
                         i, b.itemsize, b.len, ob.b.itemsize, pos, ob.b.len);
            PyBuffer_Release(&b);
            goto fail;
        }
        memcpy((char *)ob.b.buf + pos, b.buf, (size_t)b.len);
        pos += b.len;
        PyBuffer_Release(&b);
    }
    if (pos != ob.b.len) {
        PyErr_Format(PyExc_ValueError, "concat_into: %zd of %zd bytes filled", pos, ob.b.len);
        goto fail;
    }
    PyBuffer_Release(&ob.b);
    Py_DECREF(fast);
    return PyLong_FromSsize_t(n);
fail:
    PyBuffer_Release(&ob.b);
    Py_DECREF(fast);
    return NULL;
}

static PyMethodDef methods[] = {
    {"encode_plan", encode_plan, METH_VARARGS, "Per-frame counts, rows and edges of a pooled batch."},
    {"encode_fill", encode_fill, METH_VARARGS, "Fill the pooled batch arrays (see spire.tokenize)."},
    {"collate_fill", collate_fill, METH_VARARGS, "Collate frames lo:hi of an encoded batch into fixed buffers."},
    {"collate_idx", collate_idx, METH_VARARGS, "Collate samples idx of a sample block into fixed buffers (no GIL)."},
    {"concat_into", concat_into, METH_VARARGS, "Copy contiguous arrays back to back into a sized buffer."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moddef = {PyModuleDef_HEAD_INIT, "_native", "spire compiled hot-path kernels", -1, methods};

PyMODINIT_FUNC PyInit__native(void) {
    import_array();
    PyObject *m = PyModule_Create(&moddef);
    if (m == NULL) return NULL;
    if (PyModule_AddIntConstant(m, "VERSION", NATIVE_VERSION) < 0 || PyModule_AddIntConstant(m, "N_PARAMS", NP__END) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
