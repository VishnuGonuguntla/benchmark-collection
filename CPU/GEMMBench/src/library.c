#include "library.h"

#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

// Absolute library paths resolved by the Makefile from config.mk's *_PATH
// variables.  Unconfigured libraries default to NULL, which keeps their rung
// out of the ladder (see blas_lib_compiled() / level_compiled()).
#ifndef MKL_LIB_DEFAULT
#  define MKL_LIB_DEFAULT NULL
#endif
#ifndef AOCL_LIB_DEFAULT
#  define AOCL_LIB_DEFAULT NULL
#endif
#ifndef OPENBLAS_LIB_DEFAULT
#  define OPENBLAS_LIB_DEFAULT NULL
#endif

#define BLAS_NO_INFO { NULL, NULL, NULL }

// ---------------------------------------------------------------------------
// The vendor registry.  Slot order must match LVL_LIB_FIRST + n in levels.h.
// Everything the harness needs to know about a library lives in its row:
// where it came from, which symbol computes the gemm, how to size its thread
// team, and which (optional) symbols describe the build it loaded.
// ---------------------------------------------------------------------------
const blas_lib_desc BLAS_REGISTRY[LIB_COUNT] = {
    [LIB_MKL] = {
        BLAS_TAG_MKL, BLAS_NAME_MKL, "MKL", "MKL_PATH", MKL_LIB_DEFAULT,
        CBLAS_GEMM_SYM,
        // NOTE: the lowercase mkl_set_num_threads binds to the Fortran entry
        // point whose argument is a POINTER (int*) — calling it by value
        // segfaults.  The uppercase C name is the one that takes an int.
        "MKL_Set_Num_Threads", BLAS_NO_INFO,
    },
    [LIB_AOCL] = {
        // AMD AOCL-BLIS: BLIS compiled for the local uarch (libblis*.so is
        // the only AMD library with an AVX-512/Zen-optimised kernel set).
        BLAS_TAG_AOCL, BLAS_NAME_AOCL, "AOCL", "AOCL_PATH", AOCL_LIB_DEFAULT,
        CBLAS_GEMM_SYM, "blis_thread_set_num_threads", BLAS_NO_INFO,
    },
    [LIB_OPENBLAS] = {
        // OpenBLAS: portable across every x86-64/arm64 uarch (DYNAMIC_ARCH
        // picks the kernel set at run time), so it is the one rung that is
        // meaningful on all CPUs.  It also reports what it detected, which
        // is worth printing: the same binary can pick different kernels on
        // different machines.
        BLAS_TAG_OPENBLAS, BLAS_NAME_OPENBLAS, "OpenBLAS", "OPENBLAS_PATH",
        OPENBLAS_LIB_DEFAULT, CBLAS_GEMM_SYM, "openblas_set_num_threads",
        { "openblas_get_corename", "openblas_get_config", NULL },
    },
};

// Reference for -v: prefer the uarch-tuned vendor library, fall back to the
// portable one, then to the deterministic in-tree V2 kernel (runner.c).
const blas_lib_id BLAS_REF_PRIORITY[LIB_COUNT] = {
    LIB_AOCL, LIB_MKL, LIB_OPENBLAS,
};

// ---------------------------------------------------------------------------
// registry queries
// ---------------------------------------------------------------------------
int blas_lib_compiled(blas_lib_id id)
{
    // designated initializers: a registry row that was never filled in
    // simply reports "not built" instead of reading past the array
    static const int built[LIB_COUNT] = {
        [LIB_MKL] = HAS_MKL_LEVEL,
        [LIB_AOCL] = HAS_AOCL_LEVEL,
        [LIB_OPENBLAS] = HAS_OPENBLAS_LEVEL,
    };
    if (id < 0 || id >= LIB_COUNT) return 0;
    return built[id] && BLAS_REGISTRY[id].path != NULL;
}

int blas_lib_by_tag(const char *tag)
{
    if (!tag) return LIB_NONE;
    for (int i = 0; i < LIB_COUNT; i++)
        if (BLAS_REGISTRY[i].tag && !strcmp(BLAS_REGISTRY[i].tag, tag))
            return i;
    return LIB_NONE;
}

const char *blas_lib_name(blas_lib_id id)
{
    return (id >= 0 && id < LIB_COUNT && BLAS_REGISTRY[id].name)
           ? BLAS_REGISTRY[id].name : "?";
}

const char *blas_lib_tag(blas_lib_id id)
{
    return (id >= 0 && id < LIB_COUNT && BLAS_REGISTRY[id].tag)
           ? BLAS_REGISTRY[id].tag : "?";
}

const char *blas_tag_list(void)
{
    static char buf[128];
    size_t o = 0;
    for (int i = 0; i < LIB_COUNT && o + 1 < sizeof(buf); i++) {
        if (!BLAS_REGISTRY[i].tag) continue;
        o += (size_t)snprintf(buf + o, sizeof(buf) - o, "%s%s",
                              o ? ", " : "", BLAS_REGISTRY[i].tag);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------
static blas_lib_t g_libs[LIB_COUNT];
static int        g_tried[LIB_COUNT];

// Optional build description: every info symbol of the registry row that this
// library actually exports is appended ("ZEN4 | OpenBLAS 0.3.29 ...").
static void probe_info(blas_lib_t *L, void *handle)
{
    const blas_lib_desc *d = L->desc;
    size_t o = 0;
    for (int i = 0; i < 3 && d->info_sym[i]; i++) {
        const char *(*f)(void) = (const char *(*)(void))dlsym(handle, d->info_sym[i]);
        const char *s = f ? f() : NULL;
        if (!s || !*s) continue;
        int n = snprintf(L->info + o, sizeof(L->info) - o, "%s%s",
                         o ? " | " : "", s);
        if (n < 0 || (size_t)n >= sizeof(L->info) - o) break;
        o += (size_t)n;
    }
}

static void lib_load(blas_lib_id id, blas_lib_t *L)
{
    const blas_lib_desc *d = &BLAS_REGISTRY[id];
    memset(L, 0, sizeof(*L));
    L->desc = d;
    L->path = d->path;

    if (!L->path || !*L->path) {
        snprintf(L->err, sizeof(L->err),
                 "not configured - add '%s' to BLAS_LIBS and set %s in config.mk",
                 blas_lib_tag(id),
                 d->pathvar ? d->pathvar : "the matching *_PATH");
        return;
    }

    L->handle = dlopen(L->path, RTLD_NOW | RTLD_LOCAL);
    if (!L->handle) {
        snprintf(L->err, sizeof(L->err), "%s", dlerror());
        return;
    }

    L->gemm = (blas_gemm_fn)dlsym(L->handle, d->gemm_sym);
    if (!L->gemm) {
        snprintf(L->err, sizeof(L->err), "%s: symbol %s not found",
                 L->path, d->gemm_sym);
        dlclose(L->handle);
        L->handle = NULL;
        return;
    }

    if (d->set_threads_sym)
        L->set_threads = (void (*)(int))dlsym(L->handle, d->set_threads_sym);

    probe_info(L, L->handle);
    L->ok = 1;
}

int blas_open(blas_lib_id id, blas_lib_t *lib)
{
    if (id < 0 || id >= LIB_COUNT) return -1;
    if (!g_tried[id]) {
        g_tried[id] = 1;
        lib_load(id, &g_libs[id]);
    }
    *lib = g_libs[id];
    return lib->ok ? 0 : -1;
}

// ---------------------------------------------------------------------------
// running
// ---------------------------------------------------------------------------
void blas_use_threads(blas_lib_t *lib, int nthreads)
{
    if (lib && lib->set_threads && nthreads > 0) lib->set_threads(nthreads);
}

int blas_run_gemm(blas_lib_t *lib, gemm_ctx *c, int nthreads)
{
    if (!lib || !lib->ok) return -1;
    blas_use_threads(lib, nthreads);

    // Row-major: pass the problem through directly (no transpose trick
    // needed by the cblas layout argument).  int casts are safe: the
    // benchmark rejects shapes exceeding 2^31 per dimension up front.
    lib->gemm(GEMM_ROW_MAJOR, GEMM_OP_N, GEMM_OP_N,
              (int)c->M, (int)c->N, (int)c->K,
              c->alpha, c->A, (int)c->K,
              c->B, (int)c->N,
              c->beta, c->C, (int)c->N);
    return 0;
}
