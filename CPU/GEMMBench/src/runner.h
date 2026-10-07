#ifndef RUNNER_H
#define RUNNER_H

#include <stddef.h>
#include "levels.h"
#include "numa.h"
#include "library.h"

// ---------------------------------------------------------------------------
// Ladder harness: CLI, buffer lifecycle (including the NUMA-staged copies
// used by V3-V5), per-rung timing loops, -T autotuning, -v verification,
// table and CSV output.  Shared by ladder_bench.c (fixed shape) and
// ladder_sweep.c (proportionally grown shapes).
// ---------------------------------------------------------------------------

#define TUNE_DEFAULT (-1)
#define TUNE_TUNE    (-2)
#define TUNE_HELP    (-3)

#define HLINE_CPU \
  "---------------------------------------------------------------------------\n"

typedef struct {
    size_t M, N, K;
    size_t cap;                 // sweep cap (ignored by fixed-size mode)
    int    repeats;
    double minutes;             // 0 = use repeats
    GEMM_T alpha, beta;
    int    init_constant;       // 1 = constant fill, 0 = pseudo-random
    unsigned levels;            // runtime subset (0 = all compiled)
    int    tile_mode;           // TUNE_DEFAULT / TUNE_TUNE / explicit index?
    gemm_tiles tiles;           // explicit values when tile_mode==TUNE_DEFAULT&set
    int    tiles_set;
    int    verify;
    const char *csv;
    int    report;              // topology + per-rung footprint lines
    int    help;
} ladder_args;

typedef struct {
    const char *label;
    int         level;              // ladder level id (table compares by id)
    size_t M, N, K;
    double time_s;
    int    repeats;
    double gflops;
} ladder_row;

// Allocate aligned A/B/C for ctx and initialize (random or constant fill).
int  ladder_allocate(gemm_ctx *c, size_t M, size_t N, size_t K, int constant);
void ladder_free(gemm_ctx *c);

// -l/-T/-... parsing; returns 0, 1 (=print help), or -1 (error).
int ladder_parse_args(int argc, char **argv, int is_sweep, ladder_args *a);

// Full ladder for one problem instance.  Appends one row per executed rung
// to rows (may be NULL).  Prints per-rung status lines and the final table
// for fixed-size mode (table printing is the sweep's job for its rows).
int ladder_run(gemm_ctx *ctx, ladder_args *a,
               const numa_topology_t *topo, int nthreads,
               ladder_row *rows, int *nrows, int rows_cap,
               int print_table);

void ladder_print_table(const ladder_row *rows, int n, int show_shape,
                        const unsigned *mask);
void ladder_write_csv(const char *path, const ladder_row *rows, int n);
void ladder_print_tiles(void);

#endif // RUNNER_H
