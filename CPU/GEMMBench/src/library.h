#ifndef LIBRARY_H
#define LIBRARY_H

#include "levels.h"

// ---------------------------------------------------------------------------
// Vendor BLAS rungs, loaded at run time with dlopen().
//
// A single ladder binary can drive every vendor library because none of them
// is linked: MKL, AOCL-BLIS and OpenBLAS all export cblas_dgemm, so linking
// even two of them in one binary is impossible.  Instead the build resolves
// the locations given in config.mk (BLAS_LIBS + MKL_PATH / AOCL_PATH /
// OPENBLAS_PATH) and bakes absolute soname paths into the binary:
//
//   -DMKL_LIB_DEFAULT='"/abs/libmkl_rt.so"'      -DHAS_MKL_LEVEL=1
//   -DAOCL_LIB_DEFAULT='"/abs/libblis-mt.so"'    -DHAS_AOCL_LEVEL=1
//   -DOPENBLAS_LIB_DEFAULT='"/abs/libopenblas.so"' -DHAS_OPENBLAS_LEVEL=1
//
// A library that is not configured compiles its rung out (HAS_*_LEVEL = 0 ->
// blas_lib_compiled() == 0 -> level_compiled() == 0).  A configured library
// whose object cannot be loaded fails soft instead of aborting the run: the
// rung prints NOT AVAILABLE plus the dlerror() text.
//
// Only the few BLAS constants we need are declared locally: the cblas.h
// layout/transpose values are the same numbers in MKL, BLIS and OpenBLAS, so
// no vendor header is required at compile time.
//
// ADDING A LIBRARY (uarch-tuned BLAS for a new CPU vendor):
//   1. a rung id in levels.h        (LVL_<NAME>, LVL_LIB_SLOTS +1)
//   2. an enum value here           (LIB_<NAME>, LIB_COUNT +1) + its name/tag
//      macros, a HAS_<NAME>_LEVEL default, and a row in BLAS_REGISTRY
//   3. a stanza in the Makefile's vendor-BLAS registry (path var, sonames,
//      macros, level bit) and, for 'auto' selection, its CPU vendor
//   Nothing else in the harness branches on library identity.
// ---------------------------------------------------------------------------

#define GEMM_ROW_MAJOR 101
#define GEMM_OP_N      111

// Library slots; keep this order in step with the library rungs in levels.h.
typedef enum {
    LIB_MKL      = 0,
    LIB_AOCL     = 1,
    LIB_OPENBLAS = 2,
    LIB_COUNT    = 3
} blas_lib_id;

#define LIB_NONE (-1)            // blas_lib_by_tag() result for unknown tokens

#define LIB_FOR_LVL(id) ((blas_lib_id)((id) - LVL_LIB_FIRST))

_Static_assert(LIB_COUNT == LVL_LIB_SLOTS,
               "library registry and the library rungs of levels.h diverged");
_Static_assert(LVL_MKL == LVL_LIB_FIRST + LIB_MKL,
               "LVL_MKL must map onto LIB_MKL");
_Static_assert(LVL_AOCL == LVL_LIB_FIRST + LIB_AOCL,
               "LVL_AOCL must map onto LIB_AOCL");
_Static_assert(LVL_OPENBLAS == LVL_LIB_FIRST + LIB_OPENBLAS,
               "LVL_OPENBLAS must map onto LIB_OPENBLAS");

// tags = the tokens accepted in config.mk (BLAS_LIBS, LEVELS) and by -l
// names = how the rung is labelled in the ladder table
#define BLAS_TAG_MKL        "mkl"
#define BLAS_NAME_MKL       "MKL"
#define BLAS_TAG_AOCL       "aocl"
#define BLAS_NAME_AOCL      "AOCL/BLIS"
#define BLAS_TAG_OPENBLAS   "openblas"
#define BLAS_NAME_OPENBLAS  "OpenBLAS"

// Build-time availability (1 when the Makefile resolved a shared object).
#ifndef HAS_MKL_LEVEL
#  define HAS_MKL_LEVEL 0
#endif
#ifndef HAS_AOCL_LEVEL
#  define HAS_AOCL_LEVEL 0
#endif
#ifndef HAS_OPENBLAS_LEVEL
#  define HAS_OPENBLAS_LEVEL 0
#endif

// exact cblas_<d,s>gemm signature for the compiled datatype
typedef void (*blas_gemm_fn)(int layout, int ta, int tb,
                             int m, int n, int k,
                             GEMM_T alpha, const GEMM_T *A, int lda,
                             const GEMM_T *B, int ldb,
                             GEMM_T beta, GEMM_T *C, int ldc);

// Static description of one vendor library — the registry in library.c is the
// only place that knows what a library's symbols are called.
typedef struct {
    const char *tag;              // config.mk / -l token
    const char *name;             // ladder rung label
    const char *col;              // comparison-table column header
    const char *pathvar;          // config.mk variable that locates it
    const char *path;             // soname baked in by the build (NULL if off)
    const char *gemm_sym;         // cblas entry point for the datatype
    const char *set_threads_sym;  // void (*)(int) thread-count setter
    const char *info_sym[3];      // optional const char* (*)(void) build probes
} blas_lib_desc;

extern const blas_lib_desc BLAS_REGISTRY[LIB_COUNT];

// Order in which -v picks a library as the verification reference.
extern const blas_lib_id BLAS_REF_PRIORITY[LIB_COUNT];

// Live handle for one library (copied out of the loader's cache).
typedef struct {
    const blas_lib_desc *desc;
    void        *handle;
    const char  *path;
    int          ok;
    char         err[256];
    char         info[192];       // build/uarch string, "" when unavailable
    blas_gemm_fn gemm;            // cblas_dgemm / cblas_sgemm
    void       (*set_threads)(int);
} blas_lib_t;

// --- registry queries -------------------------------------------------------
int          blas_lib_compiled(blas_lib_id id);   // 0 = rung not built
int          blas_lib_by_tag(const char *tag);    // slot or LIB_NONE
const char  *blas_lib_name(blas_lib_id id);
const char  *blas_lib_tag(blas_lib_id id);
const char  *blas_tag_list(void);                 // "mkl, aocl, openblas"

// --- loading ---------------------------------------------------------------
// Open (or return the cached handle for) a library.  0 on success, -1 on
// failure with the reason in lib->err.  Safe to call repeatedly.
int  blas_open(blas_lib_id id, blas_lib_t *lib);

// --- running ---------------------------------------------------------------
// Ask the library for `nthreads` worker threads (no-op when unsupported).
void blas_use_threads(blas_lib_t *lib, int nthreads);

// Run C[M,N] = alpha*A*B + beta*C row-major through the loaded library.
// Returns 0 on success.
int  blas_run_gemm(blas_lib_t *lib, gemm_ctx *c, int nthreads);

#endif // LIBRARY_H
