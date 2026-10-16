// V6/BF16 rung: AVX512-BF16 (VDPBF16PS) GEMM kernel, ported from the
// standalone src/bf16_bench.c binary into the ladder harness.
//
// C(fp32)[M,N] = alpha * A(bf16) x B(bf16) + beta * C, outer-product
// formulation: one _mm512_dpbf16_ps consumes one (2 x bf16) pair of A
// broadcast to 16 output lanes and 16 packed (2 x bf16) pairs of B
// -> 32 FLOPs/instruction.
//
// This TU is compiled with -mavx512bf16 when the ladder Makefile enables
// the rung (target-specific LFLAGS, see Makefile); the other rungs never
// see that flag, so their codegen is unchanged.  Without the flag / the
// FP32 datatype / the CPU feature, bf16_supported() returns 0 and the
// harness prints NOT AVAILABLE and skips the rung.
#include "bf16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define BF16_LEVEL_NAME "BF16 vdpbf16ps outer"

int bf16_supported(void)
{
#if defined(__AVX512BF16__) && defined(DATATYPE_FP32)
#  if defined(__GNUC__)
    return __builtin_cpu_supports("avx512bf16") ? 1 : 0;
#  else
    return 1;
#  endif
#else
    return 0;
#endif
}

#if defined(__AVX512BF16__) && defined(DATATYPE_FP32)

#include <immintrin.h>

static uint16_t f2bf(float f)  /* round-to-nearest-even float -> bfloat16 */
{
    uint32_t u;
    memcpy(&u, &f, 4);
    if (((u >> 23) & 0xff) == 0xff) return (uint16_t)(u >> 16);  /* inf/nan */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static float bf2f(uint16_t b)  /* bfloat16 bit pattern -> float */
{
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}

/* GCC has no _mm512_castsi512_bh; LLVM/icx does. Same 512-bit vector cast works on both. */
static inline __m512bh si2bh(__m512i v)
{
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
    return _mm512_castsi512_bh(v);
#else
    return (__m512bh)v;
#endif
}

static void *xalign64(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes ? bytes : 64) != 0 || !p) return NULL;
    return p;
}

int bf16_prepare(gemm_ctx *c, bf16_state *s)
{
    const size_t M = c->M, N = c->N, K = c->K;
    memset(s, 0, sizeof(*s));
    if (!bf16_supported()) return -1;

    s->Kp     = (K + 1) & ~(size_t)1;          /* even: pairs along K */
    s->npairs = s->Kp / 2;
    s->njb    = (N + 15) / 16;                 /* 16-wide output blocks */

    s->Ap = (GEMM_T *)xalign64(M * s->Kp * sizeof(GEMM_T));
    s->Bp = (uint16_t *)xalign64(s->njb * s->npairs * 32 * sizeof(uint16_t));
    if (!s->Ap || !s->Bp) {
        fprintf(stderr, "bf16 rung: out of memory\n");
        bf16_free(s);
        return -1;
    }

    /* padded copy of A: rows re-strided to Kp, tail zeroed so odd-K pairs
       multiply against 0 exactly as in the old standalone binary */
    memset(s->Ap, 0, M * s->Kp * sizeof(GEMM_T));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; i++)
        memcpy(s->Ap + i * s->Kp, c->A + i * K, K * sizeof(GEMM_T));

    /* pair-pack B transposed: Bp[jb][p][lane]{bf16 b[2p][n], b[2p+1][n]} */
    #pragma omp parallel for schedule(static)
    for (size_t jb = 0; jb < s->njb; jb++) {
        for (size_t p = 0; p < s->npairs; p++) {
            uint16_t *d = s->Bp + (jb * s->npairs + p) * 32;
            for (int l = 0; l < 16; l++) {
                size_t n = jb * 16 + l;
                if (n >= N) { d[2 * l] = d[2 * l + 1] = 0; continue; }
                d[2 * l]     = f2bf(c->B[(2 * p + 0) * N + n]);
                d[2 * l + 1] = (2 * p + 1 < K) ? f2bf(c->B[(2 * p + 1) * N + n])
                                               : (uint16_t)0;
            }
        }
    }
    s->active = 1;
    return 0;
}

int bf16_run(const gemm_ctx *c, const bf16_state *s)
{
    const size_t M = c->M, N = c->N, npairs = s->npairs, njb = s->njb;
    const GEMM_T *Ap = s->Ap;
    const uint16_t *Bp = s->Bp;
    GEMM_T *C = c->C;
    const __m512 valpha = _mm512_set1_ps(c->alpha);
    const __m512 vbeta  = _mm512_set1_ps(c->beta);
    const int scale = (c->alpha != (GEMM_T)1.0f) || (c->beta != (GEMM_T)0.0f);

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
                    uint16_t a0 = f2bf(Ap[i * (npairs * 2) + 2 * p + 0]);
                    uint16_t a1 = f2bf(Ap[i * (npairs * 2) + 2 * p + 1]);
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
                    __m512 v = acc[b];
                    if (scale) {
                        v = _mm512_mul_ps(v, valpha);
                        if (c->beta != (GEMM_T)0.0f)
                            v = _mm512_fmadd_ps(vbeta,
                                    _mm512_maskz_loadu_ps(mk, &C[i * N + n]), v);
                    }
                    _mm512_mask_storeu_ps(&C[i * N + n], mk, v);
                }
            }
        }
    }
    return 0;
}

int bf16_verify(const gemm_ctx *c, const GEMM_T *C0, const bf16_state *s)
{
    /* corner check against a naive double bf16-arithmetic reference (the
       old bf16_bench -v): bf16 rounding makes a strict compare against the
       fp32/dgemm reference meaningless, so this checks the kernel against
       its own operand rounding instead. */
    const size_t K = c->K;
    int bad = 0, checked = 0;
    size_t mi = c->M < 8 ? c->M : 8, nj = c->N < 8 ? c->N : 8;
    for (size_t i = 0; i < mi; i++) {
        for (size_t j = 0; j < nj; j++) {
            double sref = 0.0;
            for (size_t p = 0; p < s->npairs; p++) {
                double d0 = (double)bf2f(f2bf((float)s->Ap[i * s->Kp + 2 * p]));
                double d1 = (double)bf2f(f2bf((float)s->Ap[i * s->Kp + 2 * p + 1]));
                double b0 = (double)bf2f(f2bf((float)c->B[(2 * p + 0) * c->N + j]));
                double b1 = (2 * p + 1 < K)
                    ? (double)bf2f(f2bf((float)c->B[(2 * p + 1) * c->N + j]))
                    : 0.0;
                sref += d0 * b0 + d1 * b1;
            }
            double expect = (double)c->alpha * sref
                          + (double)c->beta * (double)C0[i * c->N + j];
            double got = (double)c->C[i * c->N + j];
            double scale = expect < 0 ? -expect : expect;
            if (scale < 1) scale = 1;
            if (got - expect > 0.02 * scale || expect - got > 0.02 * scale) bad++;
            checked++;
        }
    }
    printf("    verify %-26s %s  (%d/%d mismatched, bf16 corner check)\n",
           BF16_LEVEL_NAME, bad ? "FAIL" : "PASS", bad, checked);
    return bad ? -1 : 0;
}

void bf16_free(bf16_state *s)
{
    free(s->Ap); free(s->Bp);
    memset(s, 0, sizeof(*s));
}

#else   /* no AVX512-BF16 build / datatype: the rung degrades to unsupported */

int bf16_prepare(gemm_ctx *c, bf16_state *s)
{
    (void)c;
    memset(s, 0, sizeof(*s));
    return -1;
}

int bf16_run(const gemm_ctx *c, const bf16_state *s)
{
    (void)c; (void)s;
    return -1;
}

int bf16_verify(const gemm_ctx *c, const GEMM_T *C0, const bf16_state *s)
{
    (void)c; (void)C0; (void)s;
    return -1;
}

void bf16_free(bf16_state *s)
{
    memset(s, 0, sizeof(*s));
}

#endif
