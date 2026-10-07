#include "runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <getopt.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
static void *xalloc64(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes ? bytes : 64) != 0 || !p) {
        fprintf(stderr, "out of memory allocating %zu bytes\n", bytes);
        exit(1);
    }
    return p;
}

static GEMM_T grand(unsigned long long *s)
{   // xorshift64*, uniform in [0,1)
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (GEMM_T)((double)((*s * 2685821657736338717ULL) >> 11)
                    * (1.0 / 9007199254740992.0));
}

static void fill_matrices(gemm_ctx *c, int constant);   /* fwd */

// Public setup: allocate 64-byte-aligned A/B/C for the ctx and fill them.
int ladder_allocate(gemm_ctx *c, size_t M, size_t N, size_t K, int constant)
{
    c->M = M; c->N = N; c->K = K;
    c->A = (GEMM_T *)xalloc64(M * K * sizeof(GEMM_T));
    c->B = (GEMM_T *)xalloc64(K * N * sizeof(GEMM_T));
    c->C = (GEMM_T *)xalloc64(M * N * sizeof(GEMM_T));
    fill_matrices(c, constant);
    return 0;
}

void ladder_free(gemm_ctx *c)
{
    free(c->A); free(c->B); free(c->C);
    c->A = c->B = c->C = NULL;
}

static void fill_matrices(gemm_ctx *c, int constant)
{
    size_t nA = c->M * c->K, nB = c->K * c->N, nC = c->M * c->N;
    if (constant) {
        GEMM_T *A = c->A, *B = c->B, *C = c->C;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nA; i++) A[i] = (GEMM_T)1.0;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nB; i++) B[i] = (GEMM_T)0.5;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nC; i++) C[i] = (GEMM_T)0.0;
    } else {
        GEMM_T *A = c->A, *B = c->B, *C = c->C;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nA; i++) {
            unsigned long long s = (unsigned long long)i * 0x9E3779B97F4A7C15ULL + 1;
            A[i] = grand(&s);
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nB; i++) {
            unsigned long long s = (unsigned long long)i * 0xBF58476D1CE4E5B9ULL + 1;
            B[i] = grand(&s);
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nC; i++) {
            unsigned long long s = (unsigned long long)i * 0x94D049BB133111EBULL + 1;
            C[i] = grand(&s);
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
static void usage(const char *prog, int is_sweep)
{
    printf("Usage: %s [options]\n\n"
           "CPU GEMM optimization ladder: V0 naive-omp .. V5 2D NUMA grid,\n"
           "plus the vendor library rungs (%s), loaded at run time from the\n"
           "paths baked in by the build.\n\n"
           "Options:\n"
           "  -h              this help\n"
           "  -m <int>        M: rows of A / C%s\n", prog, blas_tag_list(),
           is_sweep ? " (base)" : "");
    printf("  -n <int>        N: cols of B / C%s\n",
           is_sweep ? " (base)" : "");
    printf("  -k <int>        K: cols of A / rows of B%s\n",
           is_sweep ? " (base)" : "");
    printf("  -e <int>        e: Max. size of M/N/K %s\n",
           is_sweep ? " (base)" : "");
    printf("  -s <int>        cube shorthand M=N=K%s%s\n",
           is_sweep ? " (default shape)" : "",
           is_sweep ? "; sweep stops when the smallest dim exceeds this cap" : "");
    printf("  -r <int>        repeats per rung (default 5)\n");
    printf("  -t <double>     minutes per rung; 0 = use -r (default 0)\n");
    printf("  -a <float>      alpha (default 1.0)\n");
    printf("  -b <float>      beta  (default 0.0)\n");
    printf("  -i <mode>       data init: random|constant (default random)\n");
    printf("  -l <list>       rungs to run: 0..%d, %s, all"
           " (default: every compiled rung)\n",
           LVL_LIB_FIRST - 1, blas_tag_list());
    printf("  -T <MC,NC,KC|tune|help>   blocking params or autotune\n");
    printf("  -v              verify each rung against the reference\n");
    printf("  -c <file>       append CSV rows to <file>\n");
    printf("  -R              NUMA topology + per-rung placement report\n");
    printf("\nThread count comes from the environment (Slurm/OMP_NUM_THREADS).\n");
}

// -l list: digits for the hand-written rungs, library tags (see the registry
// in library.c) for the vendor ones, 'all' = every compiled rung (= no
// runtime subset, which is also the default).
static int parse_levels(const char *s, unsigned *mask)
{
    char buf[256];
    *mask = 0;
    strncpy(buf, s, sizeof(buf) - 1); buf[sizeof(buf)-1] = 0;
    char *tok = strtok(buf, ",");
    while (tok) {
        int slot;
        if (!strcmp(tok, "all"))          *mask = 0;
        else if ((slot = blas_lib_by_tag(tok)) >= 0)
            *mask |= LVL_BIT(LVL_OF_LIB(slot));
        else if (strlen(tok) == 1 && tok[0] >= '0' && tok[0] < '0' + LVL_LIB_FIRST)
            *mask |= LVL_BIT(tok[0] - '0');
        else return -1;
        tok = strtok(NULL, ",");
    }
    return 0;
}

static int parse_tiles(const char *s, gemm_tiles *t)
{
    int v[5] = {0,0,0,0,0};
    int n = 0;
    const char *p = s;
    while (n < 5) {
        char *end;
        long x = strtol(p, &end, 10);
        if (end == p || x <= 0 || x > 4096) return -1;
        v[n++] = (int)x;
        if (*end == ',') { p = end + 1; continue; }
        if (*end == '\0') break;
        return -1;
    }
    if (n < 3) return -1;
    gemm_tiles d = tiles_default();
    t->MC = v[0]; t->NC = v[1]; t->KC = v[2];
    if (n >= 4 && (v[3] != d.MR || (n >= 5 && v[4] != d.NR)))
        printf("note: MR/NR are compile-time register-tile constants "
               "(%d x %d); extra -T values ignored (see -T help)\n",
               d.MR, d.NR);
    t->MR = d.MR; t->NR = d.NR;
    return 0;
}

int ladder_parse_args(int argc, char **argv, int is_sweep, ladder_args *a)
{
    static struct option lopts[] = {
        {"m", required_argument, 0, 'm'}, {"n", required_argument, 0, 'n'},
        {"k", required_argument, 0, 'k'}, {"s", required_argument, 0, 's'},
        {"r", required_argument, 0, 'r'}, {"t", required_argument, 0, 't'},
        {"a", required_argument, 0, 'a'}, {"b", required_argument, 0, 'b'},
        {"i", required_argument, 0, 'i'}, {"l", required_argument, 0, 'l'},
        {"T", required_argument, 0, 'T'}, {"c", required_argument, 0, 'c'},
        {"v", no_argument, 0, 'v'},       {"R", no_argument, 0, 'R'},
        {"help", no_argument, 0, 'h'},    {0,0,0,0}
    };
    memset(a, 0, sizeof(*a));
    a->M = a->N = a->K = 0;
    a->cap = 7000;
    a->repeats = 20;
    a->alpha = (GEMM_T)1.0;
    a->beta  = (GEMM_T)0.0;
    a->tile_mode = TUNE_DEFAULT;
    a->tiles = tiles_default();

    int opt;
    while ((opt = getopt_long(argc, argv, "hm:n:k:s:r:t:a:b:i:l:T:vc:Re:",
                              lopts, NULL)) != -1) {
        switch (opt) {
        case 'h': usage(argv[0], is_sweep); return 1;
        case 'm': a->M = strtoull(optarg, NULL, 10); break;
        case 'n': a->N = strtoull(optarg, NULL, 10); break;
        case 'k': a->K = strtoull(optarg, NULL, 10); break;
        case 's': {
            size_t v = strtoull(optarg, NULL, 10);
            if (is_sweep && (a->M || a->N || a->K)) a->cap = v;
            else a->M = a->N = a->K = v;
            break;
        }
        case 'r': a->repeats = atoi(optarg); break;
        case 't': a->minutes = strtod(optarg, NULL); break;
        case 'a': a->alpha = (GEMM_T)strtod(optarg, NULL); break;
        case 'b': a->beta  = (GEMM_T)strtod(optarg, NULL); break;
        case 'i':
            if (!strcmp(optarg, "random")) a->init_constant = 0;
            else if (!strcmp(optarg, "constant")) a->init_constant = 1;
            else { fprintf(stderr, "bad -i mode\n"); return -1; }
            break;
        case 'l': if (parse_levels(optarg, &a->levels)) {
                      fprintf(stderr, "bad -l list (use 0..%d, %s, or all)\n",
                              LVL_LIB_FIRST - 1, blas_tag_list());
                      return -1; }
                  break;
        case 'T':
            if (!strcmp(optarg, "tune")) a->tile_mode = TUNE_TUNE;
            else if (!strcmp(optarg, "help")) a->tile_mode = TUNE_HELP;
            else if (parse_tiles(optarg, &a->tiles)) {
                fprintf(stderr, "bad -T (want MC,NC,KC[,MR,NR] | tune | help)\n");
                return -1;
            } else a->tiles_set = 1;
            break;
        case 'v': a->verify = 1; break;
        case 'R': a->report = 1; break;
        case 'c': a->csv = optarg; break;
        case 'e': a->cap = atoi(optarg); break;
        default: usage(argv[0], is_sweep); return -1;
        }
    }
    if (!a->M && !a->N && !a->K) a->M = a->N = a->K = 100;
    if (!a->M || !a->N || !a->K) {
        fprintf(stderr, "-m, -n, -k must all be set (or use -s)\n");
        return -1;
    }
    if (is_sweep && !a->cap) a->cap = a->M;   // default cap = base cube
    return 0;
}

void ladder_print_tiles(void)
{
    gemm_tiles d = tiles_default();
    printf("Blocking parameters (rungs V1-V5; -T MC,NC,KC):\n");
    printf("  defaults: MC=%d NC=%d KC=%d   |   register tile MRxNR = %dx%d "
           "(compile-time)\n", d.MC, d.NC, d.KC, d.MR, d.NR);
    gemm_tiles cand[16];
    int n = tiles_candidates(cand, 16);
    printf("  -T tune tries %d (MC,NC,KC) combos:\n  ", n);
    for (int i = 0; i < n; i++)
        printf("{%d,%d,%d} ", cand[i].MC, cand[i].NC, cand[i].KC);
    printf("\n");
}

// ---------------------------------------------------------------------------
// footprint report
// ---------------------------------------------------------------------------
static void print_footprint(const numa_topology_t *topo, const char *what,
                            const void *p, size_t bytes)
{
    long cnt[NUMA_MAX_NODES];
    long total = numa_footprint(p, bytes, cnt, topo->n_nodes);
    if (total <= 0) return;
    printf("    %-4s placement:", what);
    for (int i = 0; i < topo->n_nodes; i++)
        printf(" node%d=%.0f%%", i, 100.0 * (double)cnt[i] / (double)total);
    printf("\n");
}

// ---------------------------------------------------------------------------
// staged buffers for the NUMA rungs
// ---------------------------------------------------------------------------
typedef struct {
    GEMM_T *A, *B, *C;
    GEMM_T *Brep[NUMA_MAX_NODES];
    numa_plan_t plan;
    int active;
} staged_t;

static void stage_free(staged_t *s)
{
    free(s->A); free(s->B); free(s->C);
    for (int i = 0; i < NUMA_MAX_NODES; i++) free(s->Brep[i]);
    memset(s, 0, sizeof(*s));
}

// Build staged copies so every page of A/B/C (and each B replica) is
// first-touched by a thread that is already pinned to its compute node.
static void stage_build(staged_t *s, const gemm_ctx *base, int mode,
                        const numa_topology_t *topo, int nthreads,
                        int replicate_B)
{
    size_t nA = base->M * base->K, nB = base->K * base->N, nC = base->M * base->N;
    memset(s, 0, sizeof(*s));
    s->A = (GEMM_T *)xalloc64(nA * sizeof(GEMM_T));
    s->B = (GEMM_T *)xalloc64(nB * sizeof(GEMM_T));
    s->C = (GEMM_T *)xalloc64(nC * sizeof(GEMM_T));
    numa_plan(topo, mode, nthreads, &s->plan);
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
    numa_apply_plan(topo, &s->plan);
    numa_first_touch_bytes(s->A, base->A, nA * sizeof(GEMM_T));
    numa_first_touch_bytes(s->B, base->B, nB * sizeof(GEMM_T));
    numa_first_touch_bytes(s->C, base->C, nC * sizeof(GEMM_T));

    if (replicate_B && topo->n_nodes > 0) {
        for (int ni = 0; ni < topo->n_nodes; ni++) {
            int band = s->plan.node_rank_count[ni];
            if (band <= 0 && ni > 0) continue;
            if (band <= 0) band = 1;
            s->Brep[ni] = (GEMM_T *)xalloc64(nB * sizeof(GEMM_T));
            // sub-plan: `band` threads, all on node ni's cpus
            numa_plan_t sp;
            memset(&sp, 0, sizeof(sp));
            sp.mode = mode; sp.nthreads = band; sp.n_nodes = 1;
            const numa_node_t *nd = &topo->node[ni];
            for (int i = 0; i < band; i++) {
                sp.cpu[i] = nd->cpus[i % nd->ncpus];
                sp.node[i] = ni;
            }
            sp.node_start_rank[0] = 0;
            sp.node_rank_count[0] = band;
#ifdef _OPENMP
            omp_set_num_threads(band);
#endif
            numa_apply_plan(topo, &sp);
            numa_first_touch_bytes(s->Brep[ni], base->B, nB * sizeof(GEMM_T));
        }
#ifdef _OPENMP
        omp_set_num_threads(nthreads);
#endif
        numa_apply_plan(topo, &s->plan);   // restore full-team affinity
    }
    s->active = 1;
}

// ---------------------------------------------------------------------------
// rung dispatch + timing helpers
// ---------------------------------------------------------------------------
#include "timing.h"

typedef struct { gemm_ctx *ctx; const gemm_tiles *t; const gemm_numa_ctx *nz;
                 int id; blas_lib_t *lib; } rung_arg;

static void rung_dispatch(void *v)
{
    rung_arg *r = (rung_arg *)v;
    if (r->lib) {                       // vendor rung: kernel lives in the .so
        blas_run_gemm(r->lib, r->ctx, 0);
        return;
    }
    switch (r->id) {
    case LVL_V0: rung_v0_naive(r->ctx, r->t, NULL); break;
    case LVL_V1: rung_v1_block(r->ctx, r->t, NULL); break;
    case LVL_V2: rung_v2_micro(r->ctx, r->t, NULL); break;
    case LVL_V3: rung_v2_micro(r->ctx, r->t, r->nz); break;
    case LVL_V4: rung_v4_repl(r->ctx, r->t, r->nz); break;
    case LVL_V5: rung_v5_grid(r->ctx, r->t, r->nz); break;
    default: break;                     // unreachable: rungs are enumerated
    }
}

static double rung_gflops(const gemm_ctx *c, double total_s, int reps)
{
    double fl = 2.0 * (double)c->M * (double)c->N * (double)c->K
                * (double)reps;
    return (total_s > 0) ? fl / total_s / 1e9 : 0.0;
}

// ---------------------------------------------------------------------------
// verification
// ---------------------------------------------------------------------------
static int verify_against(const gemm_ctx *c, const GEMM_T *got,
                          const GEMM_T *ref, const char *label)
{
    double atol =
#ifdef DATATYPE_FP32
        1e-2;  double rtol = 5e-4;
#else
        1e-8;  double rtol = 1e-11;
#endif
    size_t n = c->M * c->N;
    size_t fails = 0;
    double maxrel = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:fails) \
            reduction(max:maxrel)
    for (size_t i = 0; i < n; i++) {
        double a = (double)got[i], b = (double)ref[i];
        double err = fabs(a - b);
        if (err > atol + rtol * fabs(b)) fails++;
        double rel = err / (fabs(b) + 1e-30);
        if (rel > maxrel) maxrel = rel;
    }
    printf("    verify %-26s %s  (max rel err %.1e, %zu mismatches)\n",
           label, fails ? "FAIL" : "PASS", maxrel, fails);
    return fails ? -1 : 0;
}

// ---------------------------------------------------------------------------
// per-rung tile choice (tune)
// ---------------------------------------------------------------------------
static gemm_tiles choose_tiles(int id, const gemm_ctx *c, const gemm_ctx *stage_ctx,
                               const gemm_numa_ctx *nz, const ladder_args *a,
                               int reps)
{
    gemm_tiles t = a->tiles;
    if (a->tile_mode != TUNE_TUNE) return t;
    if (!LEVEL_TABLE[id].uses_tiles) return t;

    gemm_tiles cand[16];
    int n = tiles_candidates(cand, 16);
    int best = 0; double best_gf = -1.0;
    for (int i = 0; i < n; i++) {
        rung_arg r = { (gemm_ctx*)stage_ctx, &cand[i], nz, id, NULL };
        double tot = 0.0;
        rung_dispatch(&r);                    // warmup
        double t0 = timestamp();
        for (int k = 0; k < reps; k++) rung_dispatch(&r);
        tot = timestamp() - t0;
        double gf = rung_gflops(c, tot, reps);
        printf("    tune %-24s MC=%3d NC=%3d KC=%3d -> %8.2f GFlop/s\n",
               LEVEL_TABLE[id].name, cand[i].MC, cand[i].NC, cand[i].KC, gf);
        if (gf > best_gf) { best_gf = gf; best = i; }
    }
    printf("    -> %s using MC=%d NC=%d KC=%d\n", LEVEL_TABLE[id].name,
           cand[best].MC, cand[best].NC, cand[best].KC);
    return cand[best];
}

// ---------------------------------------------------------------------------
// ladder entry point
// ---------------------------------------------------------------------------
int ladder_run(gemm_ctx *ctx, ladder_args *a,
               const numa_topology_t *topo, int nthreads,
               ladder_row *rows, int *nrows, int rows_cap,
               int print_table_out)
{
    unsigned want = a->levels;
    size_t nA = ctx->M * ctx->K, nB = ctx->K * ctx->N, nC = ctx->M * ctx->N;
    GEMM_T *C0 = (GEMM_T *)xalloc64(nC * sizeof(GEMM_T));
    memcpy(C0, ctx->C, nC * sizeof(GEMM_T));

    // reset OpenMP affinity to system default before the NUMA-oblivious rungs
    {
        numa_plan_t dp;
        numa_plan(topo, NUMA_MODE_DEFAULT, nthreads, &dp);
        numa_apply_plan(topo, &dp);
    }

    staged_t st_spread = {0}, st_group = {0};
    int reps = a->repeats;

    // ---- vendor library rungs: open whatever this build was configured for ----
    blas_lib_t libs[LIB_COUNT];
    int have[LIB_COUNT];
    memset(libs, 0, sizeof(libs));
    memset(have, 0, sizeof(have));
    for (int id = LVL_LIB_FIRST; id < LVL_COUNT; id++) {
        blas_lib_id slot = LIB_FOR_LVL(id);
        if (level_compiled(id)) {
            blas_open(slot, &libs[slot]);
            have[slot] = libs[slot].ok;
        }
    }

    // reference for -v: first library that loaded (uarch-tuned first, then
    // the portable one); BLAS_REF_PRIORITY keeps this in step with the registry
    blas_lib_t *ref_lib = NULL;
    for (int i = 0; i < LIB_COUNT; i++) {
        blas_lib_t *l = &libs[BLAS_REF_PRIORITY[i]];
        if (l->ok) { ref_lib = l; break; }
    }

    GEMM_T *ref = NULL;
    if (a->verify) {
        ref = (GEMM_T *)xalloc64(nC * sizeof(GEMM_T));
        memcpy(ref, C0, nC * sizeof(GEMM_T));
        gemm_ctx rctx = *ctx; rctx.C = ref;
        if (ref_lib) blas_run_gemm(ref_lib, &rctx, nthreads);
        else {
            gemm_tiles t = tiles_default();
            rung_v2_micro(&rctx, &t, NULL);   // deterministic fallback ref
        }
    }

    int count = 0;
    #define ADDROW(id_, gf_, tot_, reps_) do {                                \
        if (rows && count < rows_cap) {                                       \
            rows[count].label = (id_) < LVL_COUNT ? LEVEL_TABLE[id_].name     \
                                                  : "?";                      \
            rows[count].level = (id_);                                        \
            rows[count].M = ctx->M; rows[count].N = ctx->N; rows[count].K = ctx->K; \
            rows[count].time_s = (tot_); rows[count].repeats = (reps_);       \
            rows[count].gflops = (gf_);                                       \
        }                                                                     \
        count++;                                                              \
    } while (0)

    for (int id = 0; id < LVL_COUNT; id++) {
        int slot = LVL_IS_LIB(id) ? (int)LIB_FOR_LVL(id) : -1;

        if (!level_compiled(id)) {
            if (want && ((want >> id) & 1))
                printf("    %-30s not compiled in this build "
                       "(see LEVELS/BLAS_LIBS in config.mk)\n",
                       LEVEL_TABLE[id].name);
            continue;
        }
        if (want && !((want >> id) & 1)) continue;

        blas_lib_t *lp = NULL;
        if (slot >= 0) {
            if (!have[slot]) {                 // configured but not loadable
                printf("    %-30s NOT AVAILABLE: %s\n",
                       LEVEL_TABLE[id].name, libs[slot].err);
                continue;
            }
            lp = &libs[slot];
            if (a->report) {
                printf("    %-8s %s\n", "library:", lp->path);
                if (lp->info[0]) printf("    %-8s %s\n", "build:", lp->info);
            }
        }

        // ---- select buffers / placement for this rung ----
        int uses_numa = (id >= LVL_V3 && id < LVL_LIB_FIRST);
        gemm_ctx *use_ctx = ctx;
        gemm_ctx staged_ctx;
        gemm_numa_ctx nz = { topo, NULL, {0} };

        if (id == LVL_V3) {
            if (!st_spread.active)
                stage_build(&st_spread, ctx, NUMA_MODE_SPREAD, topo, nthreads, 0);
            staged_ctx = *ctx;
            staged_ctx.A = st_spread.A; staged_ctx.B = st_spread.B;
            staged_ctx.C = st_spread.C;
            use_ctx = &staged_ctx;
            nz.plan = &st_spread.plan;
        } else if (id == LVL_V4 || id == LVL_V5) {
            if (!st_group.active)
                stage_build(&st_group, ctx, NUMA_MODE_GROUPED, topo, nthreads, 1);
            staged_ctx = *ctx;
            staged_ctx.A = st_group.A; staged_ctx.B = st_group.B;
            staged_ctx.C = st_group.C;
            use_ctx = &staged_ctx;
            nz.plan = &st_group.plan;
            memcpy(nz.Brep, st_group.Brep, sizeof(nz.Brep));
        }

        gemm_tiles t = choose_tiles(id, ctx, use_ctx, (uses_numa ? &nz : NULL),
                                    a, reps < 2 ? reps : 2);
        const gemm_tiles *tp = &t;

        // ---- verification (fresh C, single call) ----
        if (a->verify) {
            memcpy(use_ctx->C, C0, nC * sizeof(GEMM_T));
            gemm_ctx vctx = *use_ctx;
            vctx.C = use_ctx->C;
            rung_arg r = { &vctx, tp, (uses_numa ? &nz : NULL), id, lp };
            blas_use_threads(lp, nthreads);
            rung_dispatch(&r);
            verify_against(use_ctx, use_ctx->C, ref, LEVEL_TABLE[id].name);
        }

        // ---- timed run ----
        memcpy(use_ctx->C, C0, nC * sizeof(GEMM_T));
        rung_arg r = { use_ctx, tp, (uses_numa ? &nz : NULL), id, lp };
        blas_use_threads(lp, nthreads);

        double t_total = 0.0; int done = 0;
        rung_dispatch(&r);                    // warmup
        double t0 = timestamp();
        while (1) {
            if (a->minutes > 0.0) {
                if (timestamp() - t0 > a->minutes * 60.0) break;
            } else if (done >= reps) break;
            rung_dispatch(&r);
            done++;
        }
        t_total = timestamp() - t0;
        if (done < 1) done = 1;

        double gf = rung_gflops(ctx, t_total, done);
        printf("  %-32s %10.2f GFlop/s  (%d reps, %.3f s)\n",
               LEVEL_TABLE[id].name, gf, done, t_total);
        if (a->report) {
            print_footprint(topo, "C", use_ctx->C, nC * sizeof(GEMM_T));
            print_footprint(topo, "A", use_ctx->A, nA * sizeof(GEMM_T));
            print_footprint(topo, "B", use_ctx->B, nB * sizeof(GEMM_T));
        }
        ADDROW(id, gf, t_total, done);
    }

    if (ref) free(ref);
    stage_free(&st_spread);
    stage_free(&st_group);
    free(C0);

    if (nrows) *nrows = count;
    if (print_table_out && rows) ladder_print_table(rows, count, 0, NULL);
    return 0;
    #undef ADDROW
}

// ---------------------------------------------------------------------------
// table
// ---------------------------------------------------------------------------
// GFlop/s of `base` for the same shape, so a comparison column is like for
// like; -1.0 when the reference rung did not run at that shape.
static double table_base(const ladder_row *rows, int n, int base,
                         size_t M, size_t N, size_t K)
{
    for (int i = 0; i < n; i++)
        if (rows[i].level == base &&
            rows[i].M == M && rows[i].N == N && rows[i].K == K)
            return rows[i].gflops;
    return -1.0;
}

static int table_has(const ladder_row *rows, int n, int base)
{
    for (int i = 0; i < n; i++)
        if (rows[i].level == base) return 1;
    return 0;
}

// Comparison columns: the V0 speed-up plus one "% of the vendor rung" column
// for every library rung that actually has rows — derived from the registry,
// so a newly added library gets its column without touching this function.
typedef struct {
    char hdr[32];        // "%MKL" / "x V0"
    int  base;           // level id to compare against
    int  pct;            // 1 = percentage of the reference, 0 = speed-up
    int  fw;             // header field width
} table_col;

static int table_column(table_col *c, int base, const char *hdr, int pct)
{
    snprintf(c->hdr, sizeof(c->hdr), "%s", hdr);
    c->base = base;
    c->pct  = pct;
    // wide enough for the data: percentages print "%6.1f%%" ("100.0%"),
    // speed-ups "%7.2f" ("1234.56")
    int min = pct ? 6 : 7;
    int len = (int)strlen(c->hdr);
    c->fw = len > min ? len : min;
    return 1;
}

static int table_columns(const ladder_row *rows, int n, table_col *cols, int cap)
{
    int ncol = 0;
    if (ncol < cap && table_has(rows, n, LVL_V0))
        ncol += table_column(&cols[ncol], LVL_V0, "x V0", 0);
    for (int s = 0; s < LIB_COUNT && ncol < cap; s++) {
        int base = LVL_OF_LIB(s);
        if (!table_has(rows, n, base)) continue;
        char hdr[32];
        snprintf(hdr, sizeof(hdr), "%%%s", BLAS_REGISTRY[s].col);
        ncol += table_column(&cols[ncol], base, hdr, 1);
    }
    return ncol;
}

void ladder_print_table(const ladder_row *rows, int n, int show_shape,
                        const unsigned *mask)
{
    (void)mask;
    table_col col[1 + LIB_COUNT];
    int ncol = table_columns(rows, n, col, 1 + LIB_COUNT);
    static const char dashes[] =
        "------------------------------------------------------------";

    // cell layout: ' ' + field(fw or fw-1) + ['%'] + ' ' + '|'  =>  fw + 3
    char line[512];
    int o = snprintf(line, sizeof(line), "+----------------------------------+");
    if (show_shape) o += snprintf(line + o, sizeof(line) - o, "----------------------+");
    o += snprintf(line + o, sizeof(line) - o, "---------+---------+----------+");
    for (int i = 0; i < ncol; i++)
        o += snprintf(line + o, sizeof(line) - o, "%.*s+", col[i].fw + 2, dashes);
    snprintf(line + o, sizeof(line) - o, "\n");

    fputs(line, stdout);
    printf("| %-32s |", "Level");
    if (show_shape) printf(" %-20s |", "M x N x K");
    printf(" Time(s) | Reps    |  GFlop/s |");
    for (int i = 0; i < ncol; i++)
        printf(" %-*s |", col[i].fw, col[i].hdr);
    printf("\n%s", line);

    for (int i = 0; i < n; i++) {
        printf("| %-32s |", rows[i].label);
        if (show_shape) {
            char sh[32];
            snprintf(sh, sizeof(sh), "%zux%zux%zu",
                     rows[i].M, rows[i].N, rows[i].K);
            printf(" %-20s |", sh);
        }
        printf(" %-7.3f | %-7d | %8.2f |",
               rows[i].time_s, rows[i].repeats, rows[i].gflops);
        for (int c = 0; c < ncol; c++) {
            double base = table_base(rows, n, col[c].base,
                                     rows[i].M, rows[i].N, rows[i].K);
            double v = base > 0 ? rows[i].gflops / base : 0.0;
            if (col[c].pct) printf(" %*.1f%% |", col[c].fw - 1, 100.0 * v);
            else            printf(" %*.2f |",     col[c].fw,        v);
        }
        printf("\n");
    }
    fputs(line, stdout);
}

void ladder_write_csv(const char *path, const ladder_row *rows, int n)
{
    FILE *f = fopen(path, "a");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return; }
    long cur = ftell(f);
    if (cur == 0)
        fprintf(f, "level,M,N,K,time_s,repeats,gflops\n");
    for (int i = 0; i < n; i++)
        fprintf(f, "%s,%zu,%zu,%zu,%.6f,%d,%.6f\n",
                rows[i].label, rows[i].M, rows[i].N, rows[i].K,
                rows[i].time_s, rows[i].repeats, rows[i].gflops);
    fclose(f);
}
