#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <getopt.h>
#include <string.h>
#include <omp.h>

#include "util.h"
#include "cli.h"
#include "timing.h"

#ifdef DGEMM_BENCH_WITH_MKL
#include "mkl.h"
#endif

#if defined(DGEMM_BENCH_WITH_CBLAS) || defined(DGEMM_BENCH_WITH_OPENBLAS)
#include "cblas.h"
#endif

#if defined(DGEMM_BENCH_WITH_AOCL)
#include "blis.h"
#endif

#ifdef DOUBLE
#define DGEMM_BENCH_DATATYPE double
#else
#define DGEMM_BENCH_DATATYPE float
#endif


#define DGEMM_IDX(i, j, N) ((i) * (N) + (j))

#ifndef _OPENMP
int omp_get_max_threads(void)
{
    return 1;
}
int omp_get_num_threads(void)
{
    return 1;
}
void omp_set_num_threads(int num_threads)
{
    fprintf("ERROR: Cannot set number of threads to %d, built without OpenMP support.\n", num_threads);
    return;
}
#endif

int main(int argc, char* argv[]) {
    int err = 0;
    cli_options options;

    err = get_cli_options(argc, argv, &options);
    if (err < 0)
    {
        return err;
    }
    if (options._help)
    {
        print_help(&options);
        return 0;
    }
#ifdef DOUBLE
    printf("Precision: DOUBLE\n");
#else
    printf("Precision: SINGLE\n");
#endif
#if defined(DGEMM_BENCH_WITH_MKL) || defined(DGEMM_BENCH_WITH_CBLAS) || defined(DGEMM_BENCH_WITH_AOCL)
    printf("Optimization: MKL | CBLAS\n");
#else
#if defined _OPENMP
    printf("Optimization: NAIVE\n");
#endif
#endif

    printf("Options: repeat %d alpha %f beta %f num_threads %d debug %d\n", options.repeat, options.alpha, options.beta, options.num_threads, options.debug);

    omp_set_num_threads(options.num_threads);

    DGEMM_BENCH_DATATYPE * DGEMM_BENCH_RESTRICT A = NULL;
    DGEMM_BENCH_DATATYPE * DGEMM_BENCH_RESTRICT B = NULL;
    DGEMM_BENCH_DATATYPE * DGEMM_BENCH_RESTRICT C = NULL;

    if (options.debug) printf("--- Allocating 3 matrices, each %ld bytes\n", sizeof(DGEMM_BENCH_DATATYPE) * options.N * options.N);
    err = allocate_matrix(options.N, &A);
    if (err < 0)
    {
        return err;
    }
    err = allocate_matrix(options.N, &B);
    if (err < 0)
    {
        free(A);
        return err;
    }
    err = allocate_matrix(options.N, &C);
    if (err < 0)
    {
        free(A);
        free(B);
        return err;
    }

    if (options.debug) printf("--- Initializing 3 matrices\n");
    err = fill_matrices(A, B, C, options.N);
    if (err < 0)
    {
        free(A);
        free(B);
        free(C);
        return err;
    }

    if (options.debug) printf("--- Run DGEMM %d times\n", options.repeat);
    double starttime = timestamp();
    for(int r = 0; r < options.repeat; r++) {
        if (options.debug) printf(".");
#if defined(DGEMM_BENCH_WITH_MKL) || defined(DGEMM_BENCH_WITH_CBLAS) || defined(DGEMM_BENCH_WITH_AOCL)

#ifdef double
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                options.N, options.N, options.N, options.alpha, A, options.N, B, options.N, options.beta, C, options.N);
#else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                options.N, options.N, options.N, options.alpha, A, options.N, B, options.N, options.beta, C, options.N);

#endif
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i = 0; i < options.N; i++) {
            for(size_t j = 0; j < options.N; j++) {
                double s = 0;
                for(size_t k = 0; k < options.N; k++) {
                    s += A[DGEMM_IDX(i, k, options.N)] * B[DGEMM_IDX(k, j, options.N)];
                }
                C[DGEMM_IDX(i, j, options.N)] = (options.alpha * s) + (options.beta * C[DGEMM_IDX(i, j, options.N)]);
            }
        }
#endif
    }
    double endtime = timestamp();
    if (options.debug) printf("\n");

    if (options.debug) printf("--- Do verification\n");
    double verify_s = 0;
    size_t verify_c = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:verify_s, verify_c)
#endif
    for(size_t i = 0; i < options.N; i++) {
        for(size_t j = 0; j < options.N; j++) {
            verify_s += C[DGEMM_IDX(i, j, options.N)];
            verify_c += 1;
        }
    }
    if (options.debug) printf("-- Verification sum is : %f\n", (verify_s / (verify_c * options.repeat)));
    if (options.debug) printf("-- Memory for matrices:  %f MB\n", ((double)(3 * sizeof(DGEMM_BENCH_DATATYPE) * options.N * options.N) / (1024 * 1024)));

    double dbl_N = (double)options.N;
    double dbl_repeat = (double)options.repeat;
    double fp_ops = (dbl_N * dbl_N * dbl_N * 2.0  + dbl_N * dbl_N  * 2.0);
    double runtime = endtime - starttime;
    print_horizontal_line();
    print_column_title();
    print_horizontal_line();
    print_stats(dbl_N, runtime, options.repeat, fp_ops);

    // printf("size : %f\n", dbl_N);
    // printf("Per DGEMM FP ops : %f\n", (dbl_N * dbl_N * dbl_N * 2.0) + (dbl_N * dbl_N  * 2.0));
    // printf("Per DGEMM Runtime : %f\n", runtime / options.repeat);
    // printf("Total FP ops : %f\n", fp_ops);
    // printf("Total Runtime : %f\n", runtime);
    // printf("FP rate : %f MFLOPS/s |\n", 1E-6*(fp_ops/runtime));
    print_horizontal_line();

    if (options.debug) printf("--- Cleanup 3 matrices\n");
    free(A);
    free(B);
    free(C);

    return 0;
}
