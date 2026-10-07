#include "levels.h"
#include "library.h"          // vendor rung registry

#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ---------------------------------------------------------------------------
// Compile-time register-tile extents, chosen for the vector width the
// build's -march actually enables:
//   AVX-512: fp64 4x16, fp32 4x32      AVX2/baseline: fp64 4x8, fp32 4x16
// Like the GPU ladder's TM/TN, the micro-kernel accumulators must be
// constant-bounded to live in vector registers; MC/NC/KC remain
// runtime-tunable via -T.
// ---------------------------------------------------------------------------
#define MRK 4
#if defined(__AVX512F__)
#  ifdef DATATYPE_FP32
#    define NRK 32
#  else
#    define NRK 16
#  endif
#else
#  ifdef DATATYPE_FP32
#    define NRK 16
#  else
#    define NRK 8
#  endif
#endif

// Thread-local A-panel scratch (MC x KC), grown on demand.
static _Thread_local GEMM_T *tls_ab = NULL;
static _Thread_local size_t  tls_ab_cap = 0;

static GEMM_T *tls_scratch(size_t elems)
{
    if (tls_ab_cap < elems) {
        free(tls_ab);
        tls_ab = (GEMM_T *)malloc(elems * sizeof(GEMM_T));
        tls_ab_cap = tls_ab ? elems : 0;
    }
    return tls_ab;
}

// Row band owned by rank r of a team of nt threads.
static void rank_rowband(size_t M, int r, int nt, size_t *r0, size_t *r1)
{
    *r0 = M * (size_t)r / (size_t)nt;
    *r1 = M * (size_t)(r + 1) / (size_t)nt;
}

// Which NUMA node owns rank r (grouped plan); 0 if no plan.
static int rank_node(const gemm_numa_ctx *nz, int r)
{
    if (nz && nz->plan && r < nz->plan->nthreads) return nz->plan->node[r];
    return 0;
}

// ---------------------------------------------------------------------------
// V0 — naive ijk, parallelized over C rows.  Deliberately the textbook
// loop: strided B-column reads, no blocking, no locality planning.
// ---------------------------------------------------------------------------
int rung_v0_naive(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz)
{
    (void)t; (void)nz;
    const size_t M = c->M, N = c->N, K = c->K;
    const GEMM_T alpha = c->alpha, beta = c->beta;
    const GEMM_T *A = c->A, *B = c->B;
    GEMM_T *C = c->C;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            GEMM_T s = 0;
            for (size_t k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * s + beta * C[i * N + j];
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// V1 — pure loop-reordering lesson: ikj instead of ijk, so every inner
// update is a row-contiguous vector FMA (daxpy-like).  NO blocking: B is
// re-streamed once per row of A, so this rung collapses once B no longer
// fits in cache.  That failure is what V2's blocking fixes.
// ---------------------------------------------------------------------------
int rung_v1_block(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz)
{
    (void)t; (void)nz;
    const size_t M = c->M, N = c->N, K = c->K;
    const GEMM_T alpha = c->alpha, beta = c->beta;
    const GEMM_T *A = c->A, *B = c->B;
    GEMM_T *C = c->C;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int rank = 0, nt = 1;
#ifdef _OPENMP
        rank = omp_get_thread_num(); nt = omp_get_num_threads();
#endif
        size_t r0, r1;
        rank_rowband(M, rank, nt, &r0, &r1);

        for (size_t i = r0; i < r1; i++) {
            GEMM_T *crow = C + i * N;
            const GEMM_T *arow = A + i * K;
            if (beta != (GEMM_T)1.0) {
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (size_t j = 0; j < N; j++) crow[j] *= beta;
            }
            for (size_t k = 0; k < K; k++) {
                GEMM_T av = alpha * arow[k];
                if (av == 0) continue;
                const GEMM_T *brow = B + k * N;
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (size_t j = 0; j < N; j++)
                    crow[j] += av * brow[j];
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// V2/V3 micro-kernel: per-thread row band, MC/NC/KC blocking, an MRK x NRK
// register tile per (sub)tile and a packed A panel with alpha pre-applied
// (the B tile is read contiguously row-major, which needs no packing on
// this layout).  Interior tiles take an unmasked fast path; boundary tiles
// a masked one (masked lanes multiply against zeros and are never stored).
// ---------------------------------------------------------------------------
static void micro_range(const GEMM_T *A, const GEMM_T *B, GEMM_T *C,
                        size_t N, size_t K,
                        size_t r0, size_t r1, size_t c0, size_t c1,
                        GEMM_T alpha, GEMM_T beta, const gemm_tiles *t)
{
    const size_t MC = (size_t)t->MC, NC = (size_t)t->NC, KC = (size_t)t->KC;
    GEMM_T *Ab = tls_scratch(MC * KC);
    if (!Ab) return;

    for (size_t ib = r0; ib < r1; ib += MC) {
        size_t im = MIN(MC, r1 - ib);
        for (size_t kb = 0; kb < K; kb += KC) {
            size_t km = MIN(KC, K - kb);

            /* pack A panel [im][km] with alpha folded in; reused for all jb */
            for (size_t i = 0; i < im; i++) {
                const GEMM_T *src = A + (ib + i) * K + kb;
                GEMM_T *dst = &Ab[i * km];
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (size_t k = 0; k < km; k++) dst[k] = alpha * src[k];
            }

            for (size_t jb = c0; jb < c1; jb += NC) {
                size_t jm = MIN(NC, c1 - jb);
                for (size_t ii = 0; ii < im; ii += MRK) {
                    size_t mm = MIN((size_t)MRK, im - ii);
                    for (size_t jj = 0; jj < jm; jj += NRK) {
                        size_t nn = MIN((size_t)NRK, jm - jj);
                        int full = (mm == MRK && nn == NRK);
                        GEMM_T g[MRK][NRK];
                        GEMM_T *cptr = &C[(ib + ii) * N + jb + jj];
                        const GEMM_T *bptr = &B[kb * N + jb + jj];

                        for (int m = 0; m < MRK; m++)
#ifdef _OPENMP
                            #pragma omp simd
#endif
                            for (int n = 0; n < NRK; n++)
                                g[m][n] = ((size_t)m < mm && (size_t)n < nn)
                                        ? cptr[(size_t)m * N + n] : (GEMM_T)0;
                        if (kb == 0 && beta != (GEMM_T)1.0)
                            for (int m = 0; m < MRK; m++)
                                for (int n = 0; n < NRK; n++)
                                    g[m][n] *= beta;

                        if (full) {
                            for (size_t k = 0; k < km; k++) {
                                GEMM_T b[NRK];
#ifdef _OPENMP
                                #pragma omp simd
#endif
                                for (int n = 0; n < NRK; n++)
                                    b[n] = bptr[k * N + n];
                                for (int m = 0; m < MRK; m++) {
                                    const GEMM_T av = Ab[(ii + m) * km + k];
#ifdef _OPENMP
                                    #pragma omp simd
#endif
                                    for (int n = 0; n < NRK; n++)
                                        g[m][n] += av * b[n];
                                }
                            }
                        } else {
                            for (size_t k = 0; k < km; k++) {
                                GEMM_T b[NRK];
                                for (int n = 0; n < NRK; n++)
                                    b[n] = ((size_t)n < nn)
                                        ? bptr[k * N + n] : (GEMM_T)0;
                                for (int m = 0; m < MRK; m++) {
                                    const GEMM_T av = ((size_t)m < mm)
                                        ? Ab[(ii + m) * km + k] : (GEMM_T)0;
                                    for (int n = 0; n < NRK; n++)
                                        g[m][n] += av * b[n];
                                }
                            }
                        }

                        for (size_t m = 0; m < mm; m++)
#ifdef _OPENMP
                            #pragma omp simd
#endif
                            for (size_t n = 0; n < nn; n++)
                                cptr[m * N + n] = g[m][n];
                    }
                }
            }
        }
    }
}

int rung_v2_micro(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz)
{
    const size_t M = c->M, N = c->N, K = c->K;
    const GEMM_T alpha = c->alpha, beta = c->beta;
    const GEMM_T *A = c->A, *B = c->B;
    GEMM_T *C = c->C;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int rank = 0, nt = 1;
#ifdef _OPENMP
        rank = omp_get_thread_num(); nt = omp_get_num_threads();
#endif
        size_t r0, r1;
        rank_rowband(M, rank, nt, &r0, &r1);
        const GEMM_T *Bp = B;
        if (nz && nz->Brep[0]) Bp = nz->Brep[rank_node(nz, rank)];
        micro_range(A, Bp, C, N, K, r0, r1, 0, N, alpha, beta, t);
    }
    return 0;
}

// V3: identical kernel; the RUNNER has restaged A/B/C with spread-placed
// first-touch and pinned the team (that is the optimization being shown).
int rung_v3_firsttouch(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz)
{
    return rung_v2_micro(c, t, nz);
}

// V4: staged buffers + per-node B replicas; rank picks its node's copy.
int rung_v4_repl(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz)
{
    return rung_v2_micro(c, t, nz);
}

// V5: V4 + a P x Q grid of row/column bands per node.  Row bands stay
// inside the node's staged slice (locality preserved); the extra column
// split gives sibling threads disjoint B panels that are re-used across
// the P row bands -> panel reuse in LLC.
int rung_v5_grid(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz)
{
    const size_t M = c->M, N = c->N, K = c->K;
    const GEMM_T alpha = c->alpha, beta = c->beta;
    const GEMM_T *A = c->A, *B = c->B;
    GEMM_T *C = c->C;
    const numa_plan_t *pl = nz ? nz->plan : NULL;
    if (!pl) return rung_v4_repl(c, t, nz);

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int r = 0, nt = 1;
#ifdef _OPENMP
        r = omp_get_thread_num(); nt = omp_get_num_threads();
#endif
        int ni = rank_node(nz, r);
        int base = pl->node_start_rank[ni];
        int cnt  = pl->node_rank_count[ni];
        int slot = r - base;
        if (cnt <= 0 || slot < 0 || slot >= cnt) {
            size_t r0, r1;
            rank_rowband(M, r, nt, &r0, &r1);
            micro_range(A, B, C, N, K, r0, r1, 0, N, alpha, beta, t);
        } else {
            /* grid: P row bands x Q column bands with P*Q = cnt */
            int P = (int)(sqrt((double)cnt) + 0.5);
            if (P < 1) P = 1;
            while (P > 1 && cnt % P) P--;
            int Q = cnt / P;
            int p = slot / Q, q = slot % Q;

            size_t nb0 = M * (size_t)base / (size_t)nt;
            size_t nb1 = M * (size_t)(base + cnt) / (size_t)nt;
            size_t rr0 = nb0 + (nb1 - nb0) * (size_t)p / (size_t)P;
            size_t rr1 = nb0 + (nb1 - nb0) * (size_t)(p + 1) / (size_t)P;
            size_t cc0 = N * (size_t)q / (size_t)Q;
            size_t cc1 = N * (size_t)(q + 1) / (size_t)Q;

            const GEMM_T *Bp = nz->Brep[0] ? nz->Brep[ni] : B;
            micro_range(A, Bp, C, N, K, rr0, rr1, cc0, cc1, alpha, beta, t);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// One row per rung of the ladder.  The hand-written rungs come first; the
// vendor rungs are labelled from the library registry so that a new library
// appears in the table by adding a row there.
// ---------------------------------------------------------------------------
const level_desc LEVEL_TABLE[LVL_COUNT] = {
    { LVL_V0,  "V0 naive-omp ijk",           0, 0 },
    { LVL_V1,  "V1 ikj row locality",        0, 0 },
    { LVL_V2,  "V2 blocked micro-kernel",    1, 0 },
    { LVL_V3,  "V3 + NUMA first-touch",      1, 1 },
    { LVL_V4,  "V4 + B replication",         1, 1 },
    { LVL_V5,  "V5 + 2D tile grid",          1, 1 },
    { LVL_MKL,      BLAS_NAME_MKL      " (" CBLAS_GEMM_SYM ")", 0, 0 },
    { LVL_AOCL,     BLAS_NAME_AOCL     " (" CBLAS_GEMM_SYM ")", 0, 0 },
    { LVL_OPENBLAS, BLAS_NAME_OPENBLAS " (" CBLAS_GEMM_SYM ")", 0, 0 },
};

// ---------------------------------------------------------------------------
int level_compiled(int id)
{
    if (id < 0 || id >= LVL_COUNT) return 0;
    if (!(LEVEL_MASK & LVL_BIT(id))) return 0;
    // vendor rungs additionally need their library to have been configured
    if (LVL_IS_LIB(id)) return blas_lib_compiled(LIB_FOR_LVL(id));
    return 1;
}

gemm_tiles tiles_default(void)
{
    gemm_tiles t;
    t.MC = 128; t.NC = 192; t.KC = 128;
    t.MR = MRK;
    t.NR = NRK;
    return t;
}

int tiles_candidates(gemm_tiles *out, int max)
{
    /* (MC, NC, KC) combinations tried by -T tune; MR/NR are compile-time */
    static const int combos[][3] = {
        {  64,  64,  64 }, { 128, 128, 128 }, { 128, 192, 128 },
        { 128, 256, 128 }, { 192, 128, 128 }, { 256, 128, 128 },
        { 256, 256,  64 }, {  64, 256, 128 }, { 128, 128, 256 },
        { 256, 192,  64 },
    };
    gemm_tiles base = tiles_default();
    int n = 0;
    for (size_t i = 0;
         i < sizeof(combos) / sizeof(combos[0]) && n < max; i++) {
        gemm_tiles t = base;
        t.MC = combos[i][0]; t.NC = combos[i][1]; t.KC = combos[i][2];
        out[n++] = t;
    }
    return n;
}
