#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <omp.h>

#include "cli.h"
#include "util.h"
#include "timing.h"

#ifdef DGEMM_BENCH_WITH_MKL
#include "mkl.h"
#endif

#if defined(DGEMM_BENCH_WITH_CBLAS) || defined(DGEMM_BENCH_WITH_AOCL)
#include "cblas.h"
#endif

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
    size_t matrix_dimension = 5000;
    size_t max_bytes = matrix_dimension * matrix_dimension * sizeof(DGEMM_BENCH_DATATYPE);
    printf("Max allocation per matrix: %f GB\n", max_bytes / 1e9);
  

    printf("repeat %d alpha %f beta %f num_threads %d debug %d\n", options.repeat, options.alpha, options.beta, options.num_threads, options.debug);

    omp_set_num_threads(options.num_threads);
    print_column_title();
    print_horizontal_line();
    for (size_t size = 1000; size <= matrix_dimension; size = (size_t)(size * 1.2))
    {
        DGEMM_BENCH_DATATYPE * DGEMM_BENCH_RESTRICT A = NULL;
        DGEMM_BENCH_DATATYPE * DGEMM_BENCH_RESTRICT B = NULL;
        DGEMM_BENCH_DATATYPE * DGEMM_BENCH_RESTRICT C = NULL;

        if (options.debug) printf("--- Allocating 3 matrices, each %ld bytes\n", sizeof(DGEMM_BENCH_DATATYPE) * size * size);
        err = allocate_matrix(size, &A);
        if (err < 0)
        {
            return err;
        }
        err = allocate_matrix(size, &B);
        if (err < 0)
        {
            free(A);
            return err;
        }
        err = allocate_matrix(size, &C);
        if (err < 0)
        {
            free(A);
            free(B);
            return err;
        }

        if (options.debug) printf("--- Initializing 3 matrices\n");
        err = fill_matrices(A, B, C, size);
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
                    size, size, size, options.alpha, A, size, B, size, options.beta, C, size);
#else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, size, size, options.alpha, A, size, B, size, options.beta, C, size);

#endif
#else
#ifdef _OPENMP
        #pragma omp parallel for
#endif
            for(size_t i = 0; i < size; i++) {
                for(size_t j = 0; j < size; j++) {
                    double s = 0;
                    for(size_t k = 0; k < size; k++) {
                        s += A[DGEMM_IDX(i, k, size)] * B[DGEMM_IDX(k, j, size)];
                    }
                    C[DGEMM_IDX(i, j, size)] = (options.alpha * s) + (options.beta * C[DGEMM_IDX(i, j, size)]);
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
        for(size_t i = 0; i < size; i++) {
            for(size_t j = 0; j < size; j++) {
                verify_s += C[DGEMM_IDX(i, j, size)];
                verify_c += 1;
            }
        }
        if (options.debug) printf("-- Verification sum is : %f\n", (verify_s / (verify_c * options.repeat)));
        if (options.debug) printf("-- Memory for matrices:  %f MB\n", ((double)(3 * sizeof(DGEMM_BENCH_DATATYPE) * size * size) / (1024 * 1024)));

        double dbl_N = (double)size;
        double dbl_repeat = (double)options.repeat;
        double fp_ops = dbl_N * dbl_N * dbl_N * 2.0 + dbl_N * dbl_N  * 2.0;
        double runtime = endtime - starttime;
        
        print_stats(dbl_N, runtime, options.repeat, fp_ops);

        if (options.debug) printf("--- Cleanup 3 matrices\n");
        free(A);
        free(B);
        free(C);
    
    }
    print_horizontal_line();

    return 0;
}