// CPU GEMM optimization-ladder benchmark — size sweep.
//
// The base shape (-m/-n/-k, or the -s cube) is scaled by 1.2 per step
// (all three dimensions grow proportionally) until the smallest dimension
// exceeds the cap (-s in shape mode); the whole compiled ladder runs at
// each size.  A combined table (one row per size x rung) is printed at the
// end; -c additionally streams CSV per step.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
static void omp_set_num_threads(int n) { (void)n; }
#endif

#include "runner.h"

#define SWEEP_ROW_CAP 512

int main(int argc, char **argv)
{
    ladder_args a;
    int err = ladder_parse_args(argc, argv, 1 /* sweep */, &a);
    if (err == 1) return 0;
    if (err < 0) return 1;
    if (a.tile_mode == TUNE_HELP) { ladder_print_tiles(); return 0; }

    numa_topology_t topo;
    if (numa_probe(&topo) != 0) {
        fprintf(stderr, "failed to probe NUMA topology\n");
        return 1;
    }
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
    omp_set_num_threads(nthreads);

    const size_t bM = a.M, bN = a.N, bK = a.K, cap = a.cap;
    printf("# CPU GEMM ladder sweep (%s) base %zux%zux%zu cap %zu\n",
           GEMM_DATATYPE_NAME, bM, bN, bK, cap);
    if (a.report || topo.n_nodes > 1) fputs(numa_describe(&topo), stdout);
    fflush(stdout);

    static ladder_row rows[SWEEP_ROW_CAP];
    int nrows = 0;

    size_t pM = 0, pN = 0, pK = 0;
    for (double f = 1.0; ; f *= 1.2) {
        size_t M = (size_t)(bM * f);
        size_t N = (size_t)(bN * f);
        size_t K = (size_t)(bK * f);
        if (M == 0 || N == 0 || K == 0) break;
        if (M > 0x7fffffffu || N > 0x7fffffffu || K > 0x7fffffffu) break;
        size_t mn = M < N ? (M < K ? M : K) : (N < K ? N : K);
        if (mn > cap) break;
        if (M == pM && N == pN && K == pK) break;   // growth stalled
        pM = M; pN = N; pK = K;

        printf("size %zux%zux%zu\n", M, N, K);
        fflush(stdout);

        gemm_ctx ctx;
        ladder_allocate(&ctx, M, N, K, a.init_constant);
        ctx.alpha = a.alpha;
        ctx.beta  = a.beta;

        int step = 0;
        ladder_run(&ctx, &a, &topo, nthreads,
                   nrows < SWEEP_ROW_CAP ? rows + nrows : NULL,
                   &step, SWEEP_ROW_CAP - nrows, 0 /* no per-step table */);
        if (nrows < SWEEP_ROW_CAP) nrows += step;
        if (a.csv)
            ladder_write_csv(a.csv, rows + (nrows - step), step);

        ladder_free(&ctx);
    }

    // The CSV (-c) already carries every row; the combined table would only
    // duplicate it.  Print the table for human runs, stay silent for CSV runs.
    if (!a.csv && nrows > 0)
        ladder_print_table(rows, nrows, 1 /* shapes */, NULL);
    return 0;
}
