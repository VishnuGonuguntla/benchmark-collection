#ifndef __UTIL_H
#define __UTIL_H
#include <stdio.h>
#include <errno.h>
#include <string.h>

#define DGEMM_BENCH_RESTRICT __restrict__
#define DGEMM_IDX(i, j, N) ((i) * (N) + (j))

#ifdef DOUBLE
#define DGEMM_BENCH_DATATYPE double
#else
#define DGEMM_BENCH_DATATYPE float
#endif


int allocate_matrix(size_t N, DGEMM_BENCH_DATATYPE* DGEMM_BENCH_RESTRICT *  matrix)
{
    DGEMM_BENCH_DATATYPE* mat = (DGEMM_BENCH_DATATYPE*) malloc(sizeof(DGEMM_BENCH_DATATYPE) * N * N);
    if (!mat)
    {
        fprintf(stderr, "Failed to allocate %ld bytes: %s\n", sizeof(DGEMM_BENCH_DATATYPE) * N * N, strerror(ENOMEM));
        return -ENOMEM;
    }
    *matrix = mat;
    return 0;
}

int fill_matrices(DGEMM_BENCH_DATATYPE* A, DGEMM_BENCH_DATATYPE *B, DGEMM_BENCH_DATATYPE *C, size_t N)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            A[DGEMM_IDX(i, j, N)] = 1.0;
            B[DGEMM_IDX(i, j, N)] = 0.5;
            C[DGEMM_IDX(i, j, N)] = 2.0;
        }
    }
    return 0;
}

static void print_column_title() {
    printf("| %-5s | %-16s | %-17s | %-16s | %-17s | %-18s |\n", "Size", "Per DGEMM FP ops", "Per DGEMM Runtime", "Total FP ops", "Total Runtime (s)", "FP_Rate (MFLOPS/s)");
}

static void print_horizontal_line() {    
    printf("+-------+------------------+-------------------+------------------+-------------------+--------------------+\n");    
}

static void print_stats(double dbl_N, double runtime, int repeat, double fp_ops) {
    printf("| %-5.0f ",   dbl_N);
    // Use .0f to print the double with 0 decimal places instead of casting to int
    printf("| %-16.0f ", fp_ops); 
    printf("| %-17.3f ", runtime / repeat);
    // Multiply the doubles directly, then format to 0 decimal places
    printf("| %-16.0f ", (double)repeat * fp_ops); 
    printf("| %-17.3f ", runtime);
    printf("| %-18.3f |\n", 1E-6 * ((double)repeat * fp_ops / runtime));
}
#endif /*UTIL_H*/