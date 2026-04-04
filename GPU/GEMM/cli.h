#ifndef CLI_H
#define CLI_H

// CLI argument parsing for the GEMM benchmark.
// Inspired by TheBandwidthBenchmark/src/cli.c / cli.h
// Header-only (static functions) so it can be included by both
// gemm.cu and gemm_sweep.cu without ODR violations.

#include <ctype.h>
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Compile-time defaults (overridden by -DSIZE=... etc. in the mk files)
// ---------------------------------------------------------------------------
#ifdef SIZE
#  define DEFAULT_SIZE        ((size_t)(SIZE))
#else
#  define DEFAULT_SIZE        ((size_t)5120)
#endif

#ifdef NTIMES
#  define DEFAULT_NTIMES      NTIMES
#else
#  define DEFAULT_NTIMES      500
#endif

#ifdef SWEEPSIZE
#  define DEFAULT_SWEEPSIZE   ((size_t)(SWEEPSIZE))
#else
#  define DEFAULT_SWEEPSIZE   ((size_t)90000)
#endif

#ifdef SWEEPNTIMES
#  define DEFAULT_SWEEPNTIMES SWEEPNTIMES
#else
#  define DEFAULT_SWEEPNTIMES 500
#endif

#ifdef TARGET_MINUTES
#  define DEFAULT_TARGET_MINUTES ((double)(TARGET_MINUTES))
#else
#  define DEFAULT_TARGET_MINUTES 0.0
#endif

// ---------------------------------------------------------------------------
// Argument struct
// ---------------------------------------------------------------------------
typedef struct {
    int    device_id;       // GPU device index
    size_t matrix_size;     // gemm: fixed size; gemm_sweep: max size of sweep
    int    repeats;         // iterations per run (used when target_minutes == 0)
    double target_minutes;  // >0: run for this many minutes; 0: use repeats
} GemmArgs;

// ---------------------------------------------------------------------------
// Help text
// ---------------------------------------------------------------------------
static void printHelp(const char *prog, int is_sweep)
{
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -h              Show this help text\n");
    printf("  -d <int>        GPU device ID (default: 0)\n");
    if (is_sweep) {
        printf("  -s <int>        Maximum matrix size for sweep (default: %zu)\n",
               DEFAULT_SWEEPSIZE);
        printf("  -n <int>        Iterations per matrix size (default: %d)\n",
               DEFAULT_SWEEPNTIMES);
    } else {
        printf("  -s <int>        Matrix size (default: %zu)\n", DEFAULT_SIZE);
        printf("  -n <int>        Number of iterations (default: %d)\n", DEFAULT_NTIMES);
    }
    printf("  -t <double>     Target run duration in minutes per size\n"
           "                  0 = use -n (fixed repeats), >0 = time-based (default: %.1f)\n",
           DEFAULT_TARGET_MINUTES);
    printf("\nExamples:\n");
    if (is_sweep) {
        printf("  %s -d 0 -s 32768 -n 100\n", prog);
        printf("  %s -d 1 -t 2.5\n", prog);
    } else {
        printf("  %s -d 0 -s 16384 -n 500\n", prog);
        printf("  %s -d 0 -s 16384 -t 5.0\n", prog);
    }
}

// ---------------------------------------------------------------------------
// Argument parser
// ---------------------------------------------------------------------------
static GemmArgs parseArguments(int argc, char **argv, int is_sweep)
{
    GemmArgs args;
    args.device_id      = 0;
    args.matrix_size    = is_sweep ? DEFAULT_SWEEPSIZE : DEFAULT_SIZE;
    args.repeats        = is_sweep ? DEFAULT_SWEEPNTIMES : DEFAULT_NTIMES;
    args.target_minutes = DEFAULT_TARGET_MINUTES;

    int co;
    opterr = 0;

    while ((co = getopt(argc, argv, "hd:s:n:t:")) != -1) {
        switch (co) {

        case 'h':
            printHelp(argv[0], is_sweep);
            exit(EXIT_SUCCESS);

        case 'd': {
            char *end; errno = 0;
            long val = strtol(optarg, &end, 10);
            if (*end != '\0' || errno != 0 || val < 0) {
                fprintf(stderr, "Invalid device ID for -d: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            args.device_id = (int)val;
            break;
        }

        case 's': {
            char *end; errno = 0;
            long val = strtol(optarg, &end, 10);
            if (*end != '\0' || errno != 0 || val <= 0) {
                fprintf(stderr, "Invalid matrix size for -s: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            args.matrix_size = (size_t)val;
            break;
        }

        case 'n': {
            char *end; errno = 0;
            long val = strtol(optarg, &end, 10);
            if (*end != '\0' || errno != 0 || val <= 0) {
                fprintf(stderr, "Invalid iteration count for -n: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            args.repeats = (int)val;
            break;
        }

        case 't': {
            char *end; errno = 0;
            double val = strtod(optarg, &end);
            if (*end != '\0' || errno != 0 || val < 0.0) {
                fprintf(stderr, "Invalid target minutes for -t: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            args.target_minutes = val;
            break;
        }

        case '?':
            if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'. Use -h for help.\n", optopt);
            else
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            exit(EXIT_FAILURE);

        default:
            abort();
        }
    }

    return args;
}

#endif // CLI_H
