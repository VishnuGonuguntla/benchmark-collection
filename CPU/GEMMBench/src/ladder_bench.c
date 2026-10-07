// Ladder benchmark, fixed rectangular problem:
//   C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
// Runs every compiled rung (V0..V5 + MKL/AOCL library rungs) in order and
// prints the comparison table.  Thread count comes from the environment
// (Slurm --cpus-per-task / OMP_NUM_THREADS).
#include "runner.h"
#include "levels.h"
#include "numa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
#endif

int main(int argc, char **argv)
{
    ladder_args a;
    int rc = ladder_parse_args(argc, argv, 0, &a);
    if (rc == 1) return 0;            // -h
    if (rc < 0) return 1;
    if (a.tile_mode == TUNE_HELP) { ladder_print_tiles(); return 0; }

    if (a.M > 0x7fffffffu || a.N > 0x7fffffffu || a.K > 0x7fffffffu) {
        fprintf(stderr, "dimensions must be < 2^31 (cblas LP64 interface)\n");
        return 1;
    }

    numa_topology_t topo;
    if (numa_probe(&topo) != 0) {
        fprintf(stderr, "failed to probe NUMA topology\n");
        return 1;
    }
    int nthreads = omp_get_max_threads();

    printf("GEMM optimization ladder (%s)  |  %d thread(s)\n",
           GEMM_DATATYPE_NAME, nthreads);
    printf("shape: C[%zux%zu] = A[%zux%zu] x B[%zux%zu]   alpha=%g beta=%g\n",
           a.M, a.N, a.M, a.K, a.K, a.N, (double)a.alpha, (double)a.beta);
    if (a.report || topo.n_nodes > 1) fputs(numa_describe(&topo), stdout);

    gemm_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ladder_allocate(&ctx, a.M, a.N, a.K, a.init_constant);
    ctx.alpha = a.alpha;
    ctx.beta = a.beta;

    ladder_row rows[16];
    int nrows = 0;
    // With -c the CSV is the machine-readable record; the trailing table would
    // just repeat the per-rung lines, so only print it for human runs.
    ladder_run(&ctx, &a, &topo, nthreads, rows, &nrows, 16, a.csv ? 0 : 1);

    if (a.csv) ladder_write_csv(a.csv, rows, nrows);

    ladder_free(&ctx);
    return 0;
}
