#ifndef CLI_H
#define CLI_H

// CLI argument parsing for the GEMM optimization-ladder benchmark.
// Inspired by TheBandwidthBenchmark/src/cli.c / cli.h
// Header-only (static functions) so it can be included by both
// gemm.cu and gemm_sweep.cu without ODR violations.

#include <ctype.h>
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Compile-time defaults (overridden by -DSIZE=..., -DNTIMES=..., etc.)
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
typedef enum {
    INIT_RANDOM   = 0,  // fill matrices with curand uniform random values
    INIT_CONSTANT = 1   // fill matrices with a fixed constant value
} InitMode;

// -T tile selection codes (tile_cfg field)
#define TILE_DEFAULT  (-1)   // per-level default configuration (index 0)
#define TILE_TUNE     (-2)   // time every compiled tile config, keep the best
#define TILE_HELP     (-3)   // list the compiled tile configurations

typedef struct {
    int      device_id;       // GPU device ID
    size_t   M, N, K;         // problem shape:  C[MxN] = A[MxK] * B[KxN]
    size_t   max_dim;         // sweep: largest dimension reached by the base shape
    int      repeats;         // iterations per run (used when target_minutes == 0)
    double   target_minutes;  // >0: run for this many minutes; 0: use repeats
    InitMode init_mode;       // data initialisation: random or constant
    unsigned levels;          // bitmask of levels to run (0 = all compiled)
    int      tile_cfg;        // -T: tile option index, TILE_DEFAULT/TUNE/HELP
    int      verify;          // -v: compare each level against the vendor BLAS
    const char *csv_path;     // -c: append machine-readable rows to this file
} GemmArgs;

// ---------------------------------------------------------------------------
// Help text
// ---------------------------------------------------------------------------
static void printHelp(const char *prog, int is_sweep)
{
    printf("Usage: %s [options]\n\n", prog);
    printf("Computes C[MxN] = alpha*A[MxK]*B[KxN] + beta*C and benchmarks\n"
           "the optimization ladder V0..V5 plus the vendor BLAS.\n\n");
    printf("Options:\n");
    printf("  -h              Show this help text\n");
    printf("  -d <int>        GPU device ID (default: 0)\n");
    printf("  -m <int>        M: rows of A / C%s\n", is_sweep ? " (base shape)" : "");
    printf("  -n <int>        N: cols of B / C%s\n", is_sweep ? " (base shape)" : "");
    printf("  -k <int>        K: hidden size, cols of A / rows of B%s\n",
           is_sweep ? " (base shape)" : "");
    printf("  -e <int>        Max. size of M/N/K%s\n",
           is_sweep ? " (base shape)" : "");
    printf("  -s <int>        Cube shorthand: sets M=N=K=<int>%s\n",
           is_sweep ? ", OR the sweep size cap when -m/-n/-k are given" : "");
    if (is_sweep) {
        printf("                  (sweep multiplies the base shape by 1.2 per step\n"
               "                   until any dim exceeds the cap; default cap %zu)\n",
               DEFAULT_SWEEPSIZE);
    }
    printf("  -r <int>        Number of iterations per level%s (default: %d)\n",
           is_sweep ? " per size" : "", is_sweep ? DEFAULT_SWEEPNTIMES : DEFAULT_NTIMES);
    printf("  -t <double>     Target run duration in MINUTES per level\n"
           "                  0 = use -r (fixed repeats), >0 = time-based (default: %.1f)\n",
           DEFAULT_TARGET_MINUTES);
    printf("  -i <mode>       Data initialisation: 'random' or 'constant' (default: random)\n");
    printf("  -l <list>       Comma-separated levels to run: 0..5 or 'blas' (default: all compiled)\n");
    printf("                  e.g. -l 2,3,blas ; 'all' selects every compiled level\n");
    printf("  -T <int|tune|help>  Tile configuration index for the SIMT tiled levels\n"
           "                  (see '-T help'), or 'tune' to try every compiled\n"
           "                  config per level and keep the best\n");
    printf("  -v              Verify each level against the vendor BLAS result (PASS/FAIL)\n");
    printf("  -c <file>       Append one CSV row per level to <file>\n");
    printf("\nExamples:\n");
    printf("  %s -m 8192 -n 4096 -k 16384 -r 10        rectangular problem\n", prog);
    printf("  %s -s 4096 -l blas,4,5 -t 1.0            cube shorthand, 1 min per level\n", prog);
    printf("  %s -m 2048 -n 16384 -k 8192 -v -T tune   verify + per-level tile tuning\n", prog);
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static long parsePosLong(const char *opt, const char *flag)
{
    char *end; errno = 0;
    long val = strtol(opt, &end, 10);
    if (*end != '\0' || errno != 0 || val <= 0) {
        fprintf(stderr, "Invalid value for %s: %s\n", flag, opt);
        exit(EXIT_FAILURE);
    }
    return val;
}

// Device IDs may be zero.
static long parseNonNegLong(const char *opt, const char *flag)
{
    char *end; errno = 0;
    long val = strtol(opt, &end, 10);
    if (*end != '\0' || errno != 0 || val < 0) {
        fprintf(stderr, "Invalid value for %s: %s\n", flag, opt);
        exit(EXIT_FAILURE);
    }
    return val;
}

// Parse "-l 2,3,blas" into a bitmask. Unknown tokens are fatal.
static unsigned parseLevelList(const char *arg)
{
    unsigned mask = 0;
    char buf[256];
    strncpy(buf, arg, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *tok = strtok(buf, ",");
    while (tok) {
        if (!strcmp(tok, "all")) mask = 0x7F;
        else if (!strcmp(tok, "blas") || !strcmp(tok, "6")) mask |= 1u << 6;
        else if (strlen(tok) == 1 && tok[0] >= '0' && tok[0] <= '5') mask |= 1u << (tok[0] - '0');
        else { fprintf(stderr, "Invalid level token in -l: '%s' (use 0..5, blas, all)\n", tok); exit(EXIT_FAILURE); }
        tok = strtok(NULL, ",");
    }
    return mask;
}

// ---------------------------------------------------------------------------
// Argument parser
// ---------------------------------------------------------------------------
static GemmArgs parseArguments(int argc, char **argv, int is_sweep)
{
    GemmArgs args;
    args.device_id      = 0;
    args.M = args.N = args.K = 0;
    args.max_dim        = is_sweep ? DEFAULT_SWEEPSIZE : 0;
    args.repeats        = is_sweep ? DEFAULT_SWEEPNTIMES : DEFAULT_NTIMES;
    args.target_minutes = DEFAULT_TARGET_MINUTES;
    args.init_mode      = INIT_RANDOM;
    args.levels         = 0;         // 0 = all compiled levels
    args.tile_cfg       = TILE_DEFAULT;
    args.verify         = 0;
    args.csv_path       = NULL;

    int co;
    opterr = 0;

    while ((co = getopt(argc, argv, "hd:s:m:n:k:r:t:i:l:T:vc:e:")) != -1) {
        switch (co) {

        case 'h':
            printHelp(argv[0], is_sweep);
            exit(EXIT_SUCCESS);

        case 'd':
            args.device_id = (int)parseNonNegLong(optarg, "-d");
            break;

        case 's': {
            long v = parsePosLong(optarg, "-s");
            if (is_sweep)
                args.max_dim = (size_t)v;           // sweep: -s is the cap
            else
                args.M = args.N = args.K = (size_t)v; // fixed: -s = cube shorthand
            break;
        }
        case 'm': args.M = (size_t)parsePosLong(optarg, "-m"); break;
        case 'n': args.N = (size_t)parsePosLong(optarg, "-n"); break;
        case 'k': args.K = (size_t)parsePosLong(optarg, "-k"); break;

        case 'r': args.repeats = (int)parsePosLong(optarg, "-r"); break;

        case 'e': args.repeats = (int)parsePosLong(optarg, "-e"); break;

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

        case 'i':
            if (strcmp(optarg, "random") == 0)      args.init_mode = INIT_RANDOM;
            else if (strcmp(optarg, "constant") == 0) args.init_mode = INIT_CONSTANT;
            else {
                fprintf(stderr, "Invalid init mode for -i: '%s'. Use 'random' or 'constant'.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;

        case 'l':
            args.levels = parseLevelList(optarg);
            break;

        case 'T':
            if (!strcmp(optarg, "tune"))      args.tile_cfg = TILE_TUNE;
            else if (!strcmp(optarg, "help")) args.tile_cfg = TILE_HELP;
            else {
                char *end; errno = 0;
                long val = strtol(optarg, &end, 10);
                if (*end != '\0' || errno != 0 || val < 0) {
                    fprintf(stderr, "Invalid tile config for -T: %s\n", optarg);
                    exit(EXIT_FAILURE);
                }
                args.tile_cfg = (int)val;
            }
            break;

        case 'v': args.verify = 1; break;

        case 'c': args.csv_path = optarg; break;

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

    // Defaults: -m/-n/-k (or -s cube) required; cube otherwise
    if (!args.M && !args.N && !args.K) {
        args.M = args.N = args.K = DEFAULT_SIZE;
    } else if (!args.M || !args.N || !args.K) {
        fprintf(stderr, "-m, -n and -k must be given together (or use -s for a cube).\n");
        exit(EXIT_FAILURE);
    } else 

    if (is_sweep && !args.max_dim) args.max_dim = DEFAULT_SWEEPSIZE;

    return args;
}

#endif // CLI_H
