#ifndef BF16_H
#define BF16_H

#include "levels.h"

// ---------------------------------------------------------------------------
// V6/BF16: AVX512-BF16 (VDPBF16PS) GEMM rung for BF16 throughput runs,
// ported from the standalone bf16_bench binary into the ladder.
//
//   C(fp32)[M,N] = alpha * A(bf16) x B(bf16) + beta * C
//
// outer-product formulation: one _mm512_dpbf16_ps consumes one (2 x bf16)
// pair of A broadcast to 16 output lanes and 16 packed (2 x bf16) pairs of
// B -> 32 FLOPs/instruction.  B is transposed + pair-packed once per
// problem; A is copied with its K stride padded to whole pairs and its
// bf16 conversion happens on the fly inside the timed kernel.  OpenMP over
// rows of C.
//
// The kernel TU is compiled with -mavx512bf16 (Makefile target rule);
// without that flag, or without DATATYPE=FP32, everything degrades to
// bf16_supported() == 0 and the harness skips the rung with a NOT AVAILABLE
// line.  Meaningful only on CPUs with AVX512_BF16 (Sapphire Rapids and
// newer Intel, Zen 5 EPYC, ...).
// ---------------------------------------------------------------------------

typedef struct {
    GEMM_T   *Ap;       // copy of A, K padded to whole bf16 pairs
    uint16_t *Bp;       // pair-packed B: Bp[jb][p][lane]{b[2p][n], b[2p+1][n]}
    size_t Kp, npairs, njb;
    int    active;
} bf16_state;

// 1 = compiled with the AVX512-BF16 kernel, FP32 datatype build, and the
// running CPU supports the instruction.
int bf16_supported(void);

// Stage Ap/Bp for ctx's problem (setup cost, deliberately outside timed /
// marker regions).  Returns 0, or -1 when unsupported.
int  bf16_prepare(gemm_ctx *c, bf16_state *s);

// One timed call: the VDPBF16PS outer-product kernel over the staged inputs.
int  bf16_run(const gemm_ctx *c, const bf16_state *s);

// Corner check (top-left 8x8) against a double-precision reference that
// replays the bf16 rounding of both operands, mirroring the old
// bf16_bench -v: compares C against alpha*ref + beta*C0.
int  bf16_verify(const gemm_ctx *c, const GEMM_T *C0, const bf16_state *s);

void bf16_free(bf16_state *s);

#endif // BF16_H
