// bf16_bench.c — AVX512-BF16 (VDPBF16PS) GEMM rung for BF16 throughput runs.
//
//   C(fp32)[M,N] = A(bf16) x B(bf16), outer-product formulation:
//   one _mm512_dpbf16_ps consumes one (2 x bf16) pair of A broadcast to 16
//   output lanes and 16 packed (2 x bf16) pairs of B -> 32 FLOPs/instruction.
//   B is transposed + pair-packed once per size; A pairs are converted on the
//   fly.  OpenMP over rows of C.
//
// CSV schema is identical to the ladder gemm_sweep CSV:
//   level,M,N,K,time_s,repeats,gflops
// so parse_likwid.py / visualize.ipynb treat BF16 as just another datatype.
//
// Exits with code 3 (and a "# BF16 unsupported..." line) when the CPU lacks
// AVX512_BF16, so run_bench.sh can skip cleanly on Zen4/Ice Lake etc.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
static int omp_get_thread_num(void) { return 0; }
#endif

#ifdef __AVX512BF16__
#include <immintrin.h>
#endif

#define LEVEL_NAME "BF16 vdpbf16ps outer"

static double timestamp(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static float bf2f(uint16_t b)  /* bfloat16 bit pattern -> float */
{
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}

static uint16_t f2bf(float f)  /* round-to-nearest-even float -> bfloat16 */
{
    uint32_t u;
    memcpy(&u, &f, 4);
    if (((u >> 23) & 0xff) == 0xff) return (uint16_t)(u >> 16);  /* inf/nan */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static void usage(const char *p)
{
    printf("Usage: %s [-m N -n N -k N | -s cube] [-r reps] [-c csv] [-v] [-h]\n"
           "BF16 GEMM rung (requires AVX512_BF16 at build+run time).\n", p);
}

#ifdef __AVX512BF16__

/* GCC has no _mm512_castsi512_bh; LLVM/icx does. Same 512-bit vector cast works on both. */
static inline __m512bh si2bh(__m512i v)
{
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
    return _mm512_castsi512_bh(v);
#else
    return (__m512bh)v;
#endif
}

static void bf16_gemm_run(const float *A, const uint16_t *Bp, uint32_t *C,
                          size_t M, size_t N, size_t npairs, size_t njb)
{
    #pragma omp parallel
    {
        __m512 acc[64];
        size_t jb_lim = njb < 64 ? njb : 64;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < M; i++) {
            for (size_t ib = 0; ib < njb; ib += jb_lim) {
                size_t nb = (njb - ib < jb_lim) ? (njb - ib) : jb_lim;
                for (size_t b = 0; b < nb; b++)
                    acc[b] = _mm512_setzero_ps();
                for (size_t p = 0; p < npairs; p++) {
                    uint16_t a0 = f2bf(A[i * (npairs * 2) + 2 * p + 0]);
                    uint16_t a1 = f2bf(A[i * (npairs * 2) + 2 * p + 1]);
                    __m512bh sa = si2bh(
                        _mm512_set1_epi32((int)((uint32_t)a1 << 16 | a0)));
                    for (size_t b = 0; b < nb; b++) {
                        __m512bh sb = si2bh(
                            _mm512_loadu_si512(Bp + ((ib + b) * npairs + p) * 32));
                        acc[b] = _mm512_dpbf16_ps(acc[b], sa, sb);
                    }
                }
                for (size_t b = 0; b < nb; b++) {
                    size_t n = (ib + b) * 16;
                    __mmask16 mk = (N - n >= 16) ? (__mmask16)0xffff
                                                 : (__mmask16)(((__mmask16)1 << (N - n)) - 1);
                    _mm512_mask_storeu_ps(&C[i * N + n], mk, acc[b]);
                }
            }
        }
    }
}

#endif /* __AVX512BF16__ */

int main(int argc, char **argv)
{
    size_t M = 0, N = 0, K = 0;
    int reps = 3, verify = 0;
    const char *csv = NULL;
    int opt;
    while ((opt = getopt(argc, argv, "hm:n:k:s:r:c:v")) != -1) {
        switch (opt) {
        case 'm': M = strtoull(optarg, NULL, 10); break;
        case 'n': N = strtoull(optarg, NULL, 10); break;
        case 'k': K = strtoull(optarg, NULL, 10); break;
        case 's': M = N = K = strtoull(optarg, NULL, 10); break;
        case 'r': reps = atoi(optarg); break;
        case 'c': csv = optarg; break;
        case 'v': verify = 1; break;
        default: usage(argv[0]); return opt == 'h' ? 0 : 1;
        }
    }
    if (!M || !N || !K) { fprintf(stderr, "dimensions must be > 0 (-s or -m/-n/-k)\n"); return 1; }
    if (M > 0x7fffffffu || N > 0x7fffffffu || K > 0x7fffffffu) {
        fprintf(stderr, "dimensions must be < 2^31\n"); return 1;
    }

#ifndef __AVX512BF16__
    printf("# BF16 unsupported: build without -mavx512bf16\n");
    return 3;
#elif defined(__GNUC__)
    if (!__builtin_cpu_supports("avx512bf16")) {
        printf("# BF16 unsupported: CPU has no AVX512_BF16\n");
        return 3;
    }
#endif

    const size_t Kp = (K + 1) & ~(size_t)1;     /* even: pairs along K */
    const size_t npairs = Kp / 2;
    const size_t njb    = (N + 15) / 16;        /* 16-wide output blocks */

    printf("# BF16 GEMM (" LEVEL_NAME ") M=%zu N=%zu K=%zu | %d thread(s)\n",
           M, N, K, omp_get_max_threads());

    float   *A = aligned_alloc(64, M * Kp * sizeof(float));
    float   *B = aligned_alloc(64, Kp * N * sizeof(float));
    uint32_t *C = aligned_alloc(64, M * N * sizeof(uint32_t));
    uint16_t *Bp = aligned_alloc(64, njb * npairs * 32 * sizeof(uint16_t));
    if (!A || !B || !C || !Bp) { fprintf(stderr, "out of memory\n"); return 1; }

    srand(42);
    for (size_t i = 0; i < M * Kp; i++)
        A[i] = ((rand() % 2000) - 1000) / 500.0f;
    for (size_t i = 0; i < Kp * N; i++)
        B[i] = ((rand() % 2000) - 1000) / 500.0f;
    for (size_t i = 0; i < Kp * N; i++) /* pad K tail to pairs with zeros */
        if ((i / N) >= K) B[i] = 0.0f;
    memset(C, 0, M * N * sizeof(uint32_t));

    /* pair-pack B transposed: Bp[jb][p][lane]{bf16 b[2p][n], b[2p+1][n]} */
    #pragma omp parallel for schedule(static)
    for (size_t jb = 0; jb < njb; jb++) {
        for (size_t p = 0; p < npairs; p++) {
            uint16_t *d = Bp + (jb * npairs + p) * 32;
            for (int l = 0; l < 16; l++) {
                size_t n = jb * 16 + l;
                if (n >= N) { d[2 * l] = d[2 * l + 1] = 0; continue; }
                d[2 * l]     = f2bf(B[(2 * p + 0) * N + n]);
                d[2 * l + 1] = f2bf(B[(2 * p + 1) * N + n]);
            }
        }
    }

    /* timed run: one warmup, then `reps` measured passes */
    double dt = 0.0, gflops = 0.0;
#ifdef __AVX512BF16__
    bf16_gemm_run(A, Bp, C, M, N, npairs, njb);              /* warmup */
    double t0 = timestamp();
    for (int r = 0; r < reps; r++)
        bf16_gemm_run(A, Bp, C, M, N, npairs, njb);
    dt = timestamp() - t0;
    if (dt <= 0.0) dt = 1e-12;
    gflops = (double)M * (double)N * (double)K * 2.0 * reps / dt / 1e9;
#endif

    if (verify) {
        /* corner check against naive double bf16-arithmetic reference */
        int bad = 0, checked = 0;
        size_t mi = M < 8 ? M : 8, nj = N < 8 ? N : 8;
        for (size_t i = 0; i < mi; i++) {
            for (size_t j = 0; j < nj; j++) {
                double s = 0.0;
                for (size_t p = 0; p < npairs; p++) {
                    uint16_t a0 = f2bf(A[i * Kp + 2 * p]), a1 = f2bf(A[i * Kp + 2 * p + 1]);
                    double d0 = bf2f(a0), d1 = bf2f(a1);
                    /* NOTE: lane order b[2p][n],b[2p+1][n] matches f2bf(B[k][n]) */
                    s += d0 * bf2f(f2bf(B[(2 * p + 0) * N + j]))
                       + d1 * bf2f(f2bf(B[(2 * p + 1) * N + j]));
                }
                uint32_t cb = C[i * N + j]; float c; memcpy(&c, &cb, 4);
                double scale = s < 0 ? -s : s; if (scale < 1) scale = 1;
                if ((c - s > 0.02 * scale || s - c > 0.02 * scale)) bad++;
                checked++;
            }
        }
        printf("  verify %-20s %s (%d/%d mismatched)\n", LEVEL_NAME,
               bad ? "FAIL" : "OK", bad, checked);
    }

    printf("  %-32s %10.2f GFlop/s  (%d reps, %.3f s)\n", LEVEL_NAME, gflops, reps, dt);

    if (csv) {
        FILE *f = fopen(csv, "a");
        if (f) {
            long cur = ftell(f);
            if (cur == 0) fprintf(f, "level,M,N,K,time_s,repeats,gflops\n");
            fprintf(f, "%s,%zu,%zu,%zu,%.6f,%d,%.6f\n", LEVEL_NAME, M, N, K, dt, reps, gflops);
            fclose(f);
        } else fprintf(stderr, "cannot open %s\n", csv);
    }

    free(A); free(B); free(C); free(Bp);
    return 0;
}
