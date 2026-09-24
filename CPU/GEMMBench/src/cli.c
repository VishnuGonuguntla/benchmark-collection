#include "cli.h"

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