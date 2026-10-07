#ifndef LEVELS_H
#define LEVELS_H

#include <stddef.h>
#include "numa.h"

// ---------------------------------------------------------------------------
// CPU GEMM optimization ladder.  Every rung computes the same row-major
// problem, fully multithreaded with OpenMP:
//
//     C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
//
//   V0 naive-omp          ijk, one row-band per thread, no locality tricks
//   V1 ikj locality       pure loop reordering: every inner update is a
//                         row-contiguous vector FMA; no blocking, so it
//                         falls apart once B stops fitting in cache
//   V2 blocked microkernel  MC x NC x KC blocking + MRK x NRK register tile
//                         + packed A panel + omp simd (runtime -T tiles)
//   V3 + NUMA first-touch same kernel; A/B/C restaged with pinned ranks
//                         (spread across nodes) so pages sit where they
//                         are computed
//   V4 + B replication    V3 with grouped (per-node contiguous) placement
//                         and a socket-local replica of B per NUMA node
//   V5 + 2D tile grid     V4 with per-node P x Q thread grid: row x column
//                         bands maximize B/A panel reuse in LLC
//
//   vendor rungs   the library references from LVL_LIB_FIRST on: their
//                         kernels are not here, they are dlopen'd — see the
//                         registry in library.h for what can be built in
//
// Runes 3-5 receive their buffers through the gemm_numa_ctx (staged copies);
// the kernels themselves only differ in how B is picked and how the C plane
// is tiled across ranks.
// ---------------------------------------------------------------------------

#ifdef DATATYPE_FP32
  typedef float GEMM_T;
# define GEMM_DATATYPE_NAME "FP32"
#else
  typedef double GEMM_T;
# define GEMM_DATATYPE_NAME "FP64"
#endif

typedef struct {
    size_t M, N, K;
    GEMM_T alpha, beta;
    GEMM_T *A, *B, *C;
} gemm_ctx;

typedef struct {
    int MC, NC, KC;   // outer (cache) blocking granularity
    int MR, NR;       // register tile (micro-kernel) granularity
} gemm_tiles;

typedef struct {
    const numa_topology_t *topo;
    const numa_plan_t     *plan;    // rank -> node/cpu placement
    GEMM_T *Brep[NUMA_MAX_NODES];   // per-node replicas of B (V4/V5) or NULL
} gemm_numa_ctx;

// Every rung returns 0 on success.
int rung_v0_naive(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz);
int rung_v1_block(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz);
int rung_v2_micro(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz);
int rung_v3_firsttouch(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz);
int rung_v4_repl(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz);
int rung_v5_grid(gemm_ctx *c, const gemm_tiles *t, const gemm_numa_ctx *nz);

// Level ids.  LVL_V0..LVL_V5 are the hand-written rungs; from LVL_LIB_FIRST
// up sit the vendor BLAS rungs, one slot per row of the registry in
// src/library.h (blas_lib_id == level id - LVL_LIB_FIRST, checked there with
// _Static_assert).  The Makefile assigns bit (1 << id) to each of them in
// LEVEL_MASK, so the two lists must stay in step.
#define LVL_V0 0
#define LVL_V1 1
#define LVL_V2 2
#define LVL_V3 3
#define LVL_V4 4
#define LVL_V5 5

#define LVL_LIB_FIRST  6
#define LVL_LIB_SLOTS  3                 // == LIB_COUNT in library.h
#define LVL_MKL        (LVL_LIB_FIRST + 0)
#define LVL_AOCL       (LVL_LIB_FIRST + 1)
#define LVL_OPENBLAS   (LVL_LIB_FIRST + 2)

#define LVL_HAND_COUNT LVL_LIB_FIRST     // hand-written rungs
#define LVL_COUNT      (LVL_LIB_FIRST + LVL_LIB_SLOTS)

#define LVL_BIT(i) (1u << (i))
#define LVL_IS_LIB(id) ((id) >= LVL_LIB_FIRST && (id) < LVL_COUNT)
#define LVL_OF_LIB(slot) (LVL_LIB_FIRST + (int)(slot))

// every rung of the ladder (used for 'all' at run time)
#define LEVEL_ALL ((1u << LVL_COUNT) - 1u)

#ifndef LEVEL_MASK
#  define LEVEL_MASK LEVEL_ALL
#endif

typedef struct {
    int id;
    const char *name;
    int uses_tiles;         // tile params meaningful for this rung
    int needs_numa;         // requires staged buffers / placement
} level_desc;

#ifdef DATATYPE_FP32
#  define CBLAS_GEMM_SYM "cblas_sgemm"
#else
#  define CBLAS_GEMM_SYM "cblas_dgemm"
#endif

extern const level_desc LEVEL_TABLE[LVL_COUNT];

// compiled-in mask
int level_compiled(int id);

// Sane per-datatype default tile parameters (tunable at runtime with -T).
gemm_tiles tiles_default(void);
// Tuning candidates tried by -T tune (per level class: blocking rungs).
int tiles_candidates(gemm_tiles *out, int max);

#endif // LEVELS_H
