#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>
#include <getopt.h>
#include <string.h>
#include <omp.h>

#ifdef DGEMM_BENCH_WITH_MKL
#include "mkl.h"
#endif

#if defined(DGEMM_BENCH_WITH_CBLAS) || defined(DGEMM_BENCH_WITH_AOCL)
#include "cblas.h"
#endif

#define DGEMM_BENCH_DEFAULT_N       256
#define DGEMM_BENCH_DEFAULT_REPEAT  20
#define DGEMM_BENCH_DEFAULT_ALPHA   1.0
#define DGEMM_BENCH_DEFAULT_BETA    1.0
#define DGEMM_BENCH_DEFAULT_DEBUG   0

#define DGEMM_BENCH_DATATYPE double

#define DGEMM_BENCH_RESTRICT __restrict__

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

static struct option cli_options_list[] =
{
    {"N", required_argument, NULL, 'n'},
    {"n", required_argument, NULL, 'n'},
    {"repeat", required_argument, NULL, 'r'},
    {"alpha", required_argument, NULL, 'a'},
    {"beta", required_argument, NULL, 'b'},
    {"debug", no_argument, NULL, 'd'},
    {"threads", required_argument, NULL, 't'},
    {NULL, 0, NULL, 0}
};

typedef struct {
    size_t N;
    int repeat;
    DGEMM_BENCH_DATATYPE alpha;
    DGEMM_BENCH_DATATYPE beta;
    int debug;
    int num_threads;
    int max_threads;
    int _help;
} cli_options;

void print_help(cli_options* options)
{
    printf("dgemm_bench - Benchmark for DGEMM implementations of various implementations\n");
    printf("\n");
    printf("Options:\n");
    printf("\t-n/-N/--n/--N <int> : Dim size (N) for NxNxN matrices (N=%ld)\n", options->N);
    printf("\t-r/--repeat <int> : Number of repeats (repeats=%d)\n", options->repeat);
    printf("\t-a/--alpha <float> : Alpha factor (alpha=%f)\n", options->alpha);
    printf("\t-b/--beta <float> : Beta factor (beta=%f)\n", options->beta);
    printf("\t-t/--threads <float> : Amount of OpenMP threads (threads=%d)\n", options->num_threads);
    printf("\t-d/--debug      : Enable debug output (debug=%d)\n", options->debug);
}

int get_cli_options(int argc, char* argv[], cli_options* options)
{
    char ch;
    int err = 0;
    int num_threads = 0;
    options->N = DGEMM_BENCH_DEFAULT_N;
    options->repeat = DGEMM_BENCH_DEFAULT_REPEAT;
    options->alpha = DGEMM_BENCH_DEFAULT_ALPHA;
    options->beta = DGEMM_BENCH_DEFAULT_BETA;
    options->debug = DGEMM_BENCH_DEFAULT_DEBUG;
    options->_help = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
{
#ifdef _OPENMP
#pragma omp master
#endif
    {
        options->max_threads = omp_get_max_threads();
        options->num_threads = omp_get_num_threads();
    }
}

    while ((ch = getopt_long(argc, argv, "hN:n:r:a:b:d", cli_options_list, NULL)) != -1)
    {
        switch (ch)
        {
             case 'h':
                options->_help = 1;
                break;
             case 'n':
             case 'N':
                options->N = strtoull(optarg, NULL, 10);
                break;
             case 'r':
                options->repeat = atoi(optarg);
                break;
             case 't':
                num_threads = atoi(optarg);
                if (num_threads <= options->num_threads)
                {
                    options->num_threads = num_threads;
                }
                else
                {
                    fprintf(stderr, "ERROR: Cannot use %d threads, only maximal %d possible.\n", num_threads, options->num_threads);
                    
                }
                break;
             case 'a':
                options->alpha = strtod(optarg, NULL);
                break;
             case 'b':
                options->beta = strtod(optarg, NULL);
                break;
             case 'd':
                options->debug = 1;
                break;
             default:
                errno = EINVAL;
                fprintf(stderr, "Unknown command line option -%c: %s\n", ch, strerror(errno));
                err = -EINVAL;
                break;
        }
    }
    return err;
}

double timestamp(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}

double resolution(void)
{
    struct timespec ts;
    clock_getres(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}


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
    printf("Options: N %ld repeat %d alpha %f beta %f num_threads %d debug %d\n", options.N, options.repeat, options.alpha, options.beta, options.num_threads, options.debug);

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
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                options.N, options.N, options.N, options.alpha, A, options.N, B, options.N, options.beta, C, options.N);
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
    double fp_ops = (dbl_N * dbl_N * dbl_N * 2.0 * dbl_repeat) + (dbl_N * dbl_N  * 2.0 * dbl_repeat);
    double runtime = endtime - starttime;
    printf("Per DGEMM FP ops : %f\n", (dbl_N * dbl_N * dbl_N * 2.0) + (dbl_N * dbl_N  * 2.0));
    printf("Per DGEMM Runtime : %f\n", runtime / options.repeat);
    printf("Total FP ops : %f\n", fp_ops);
    printf("Total Runtime : %f\n", runtime);
    printf("FP rate : %f MFLOPS/s\n", 1E-6*(fp_ops/runtime));

    if (options.debug) printf("--- Cleanup 3 matrices\n");
    free(A);
    free(B);
    free(C);

    return 0;
}
