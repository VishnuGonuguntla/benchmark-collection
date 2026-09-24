#ifndef CLI_H
#define CLI_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <getopt.h>
#include <errno.h>
#include <string.h>

#define DGEMM_BENCH_DEFAULT_N       256
#define DGEMM_BENCH_DEFAULT_REPEAT  20
#define DGEMM_BENCH_DEFAULT_ALPHA   1.0
#define DGEMM_BENCH_DEFAULT_BETA    1.0
#define DGEMM_BENCH_DEFAULT_DEBUG   0

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
    float alpha;
    float beta;
    int debug;
    int num_threads;
    int max_threads;
    int _help;
} cli_options;

extern void print_help(cli_options* options);

extern int get_cli_options(int argc, char* argv[], cli_options* options);

#endif /*CLI_H*/