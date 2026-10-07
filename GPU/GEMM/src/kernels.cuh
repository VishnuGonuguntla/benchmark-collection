#ifndef KERNELS_CUH
#define KERNELS_CUH
#include "portability.h"
#include <stdio.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// The optimization ladder.  Every level computes, at its own depth of
// optimization, the same row-major problem
//
//     C[M x N] = alpha * A[M x K] * B[K x N] + beta * C[M x N]
//
//   V0  naive             one thread = one C element, global memory only
//   V1  shared tiling     classic TILE x TILE shared-memory tiles (8/16/32)
//   V2  register tiling   BMxBN block tile, each thread owns a TMxTN
//                         micro-tile of accumulators
//   V3  + prefetch        padded shared tiles; the next K-tile is staged
//                         into registers while the current tile is computed
//   V4  tuned SIMT        double-buffered shared memory, 4-wide vectorized
//                         cooperative global loads, transposed B tile
//   V5  tensor cores      WMMA (NVIDIA) / rocWMMA (AMD): fp16->fp32 on both,
//                         TF32 on NVIDIA FP32 builds
//
// Levels 2..4 are templated on (BM,BN,BK,TM,TN) and instantiated for a
// curated set of tile configs selected at runtime with -T.  They require
// M%BM==0, N%BN==0, K%BK==0; the harness aligns all dimensions to 256,
// which satisfies every instantiation.  All kernels use 64-bit global
// addressing so that huge shapes (e.g. 37632^2) do not overflow.
// ---------------------------------------------------------------------------

#define LEVEL_V0   0
#define LEVEL_V1   1
#define LEVEL_V2   2
#define LEVEL_V3   3
#define LEVEL_V4   4
#define LEVEL_V5   5
#define LEVEL_BLAS 6

#ifndef LEVEL_MASK
#define LEVEL_MASK 0x7F   // bit per compiled level (V0..V5 + BLAS)
#endif

// ---------------------------------------------------------------------------
// Shared launch arguments for every ladder kernel.  Launchers return false
// when the configuration cannot run on this device (e.g. the dynamic shared
// memory request exceeds the GPU's opt-in limit); the harness then skips
// that level with a message instead of aborting.
// ---------------------------------------------------------------------------
struct LaunchArgs {
  int M, N, K;
  ACCUM alpha, beta;
  const STORAGE *A, *B;
  ACCUM *C;
};
typedef bool (*LaunchFn)(const LaunchArgs &);

// Bank-conflict padding: one 16-byte line worth of elements
// (float:4, double:2, half:8).
#define SMEM_PAD (16 / (int)sizeof(STORAGE))

// 4-element staging vector (the compiler emits wide global loads).
// alignas(4*sizeof(T)) keeps the vector's alignment requirement exactly
// what the 4-element-multiple offsets guarantee (16 B for float, 8 B for
// half; doubles split into two 16 B loads).
template <typename T>
struct alignas(4 * sizeof(T)) Vec4 { T v[4]; };

// Opt-in to dynamic shared memory beyond the 48 KB default.
static inline bool smem_optin(void *func, size_t bytes)
{
  if (bytes <= 48u * 1024u) return true;
  cudaError_t e = cudaFuncSetAttribute((const void *)func,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)bytes);
  return e == cudaSuccess;
}

// ===========================================================================
// V0 — naive: one thread per output element, no shared memory.
// Any shape (boundary-masked).
// ===========================================================================
__global__ void gemm_v0_naive(int M, int N, int K, ACCUM alpha,
                              const STORAGE *A, const STORAGE *B,
                              ACCUM beta, ACCUM *C)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;

  ACCUM sum = 0;
  const size_t arow = (size_t)row * K;
  for (int k = 0; k < K; ++k)
    sum += (ACCUM)A[arow + k] * (ACCUM)B[(size_t)k * N + col];

  size_t c = (size_t)row * N + col;
  C[c] = alpha * sum + beta * C[c];
}

static bool launch_v0(const LaunchArgs &a)
{
  dim3 thr(16, 16);
  dim3 grid((a.N + 15) / 16, (a.M + 15) / 16);
  gemm_v0_naive<<<grid, thr>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  return true;
}

// ===========================================================================
// V1 — shared-memory tiling (generalized version of the original NAIVE
// variant).  TILE x TILE block tile, one thread per shared slot, boundary
// masking -> works for any M/N/K.
// ===========================================================================
template <int TILE>
__global__ void gemm_v1_shared(int M, int N, int K, ACCUM alpha,
                               const STORAGE *A, const STORAGE *B,
                               ACCUM beta, ACCUM *C)
{
  __shared__ STORAGE sA[TILE * TILE];
  __shared__ STORAGE sB[TILE * TILE];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  ACCUM sum = 0;
  const int numTiles = (K + TILE - 1) / TILE;

  for (int t = 0; t < numTiles; ++t) {
    int k = t * TILE + tx;
    sA[ty * TILE + tx] = (row < M && k < K) ? A[(size_t)row * K + k] : (STORAGE)0;
    k = t * TILE + ty;
    sB[ty * TILE + tx] = (k < K && col < N) ? B[(size_t)k * N + col] : (STORAGE)0;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TILE; ++i)
      sum += (ACCUM)sA[ty * TILE + i] * (ACCUM)sB[i * TILE + tx];
    __syncthreads();
  }

  if (row < M && col < N) {
    size_t c = (size_t)row * N + col;
    C[c] = alpha * sum + beta * C[c];
  }
}

template <int TILE>
static bool launch_v1(const LaunchArgs &a)
{
  dim3 thr(TILE, TILE);
  dim3 grid((a.N + TILE - 1) / TILE, (a.M + TILE - 1) / TILE);
  gemm_v1_shared<TILE><<<grid, thr>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  return true;
}

// ===========================================================================
// V2 — register tiling.  Block computes a BMxBN tile; thread (x,y) owns a
// TMxTN micro-tile of accumulators so every shared-memory read feeds
// TM*TN FMAs.  Plain single-buffered shared tiles, cooperative strided
// global loads, one pair of __syncthreads() per K-tile.
// ===========================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v2_reg(int M, int N, int K, ACCUM alpha,
                            const STORAGE *A, const STORAGE *B,
                            ACCUM beta, ACCUM *C)
{
  extern __shared__ STORAGE smem[];
  STORAGE *sA = smem;                       // [BM][BK]
  STORAGE *sB = smem + BM * BK;             // [BK][BN]

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int NT  = blockDim.x * blockDim.y;
  const int m0  = blockIdx.y * BM;
  const int n0  = blockIdx.x * BN;

  ACCUM acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0;

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int e = tid; e < BM * BK; e += NT) {
      int r = e / BK, k = e % BK;
      sA[e] = A[(size_t)(m0 + r) * K + k0 + k];
    }
    for (int e = tid; e < BK * BN; e += NT) {
      int k = e / BN, n = e % BN;
      sB[e] = B[(size_t)(k0 + k) * N + n0 + n];
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      ACCUM fa[TM], fb[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) fa[i] = (ACCUM)sA[(threadIdx.y * TM + i) * BK + k];
      #pragma unroll
      for (int j = 0; j < TN; ++j) fb[j] = (ACCUM)sB[k * BN + threadIdx.x * TN + j];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += fa[i] * fb[j];
    }
    __syncthreads();
  }

  const int ra = m0 + threadIdx.y * TM;
  const int cb = n0 + threadIdx.x * TN;
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      size_t c = (size_t)(ra + i) * N + cb + j;
      C[c] = alpha * acc[i][j] + beta * C[c];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
static bool launch_v2(const LaunchArgs &a)
{
  dim3 thr(BN / TN, BM / TM);
  dim3 grid(a.N / BN, a.M / BM);
  size_t smem = (size_t)(BM * BK + BK * BN) * sizeof(STORAGE);
  if (!smem_optin((void *)&gemm_v2_reg<BM, BN, BK, TM, TN>, smem)) return false;
  gemm_v2_reg<BM, BN, BK, TM, TN><<<grid, thr, smem>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  return true;
}

// ===========================================================================
// V3 — V2 plus bank-conflict padding and register prefetching: the next
// K-tile's global loads are issued into per-thread registers *before* the
// current tile's FMAs run and only written to shared memory afterwards,
// hiding global-memory latency.  Each thread stages TM*KS A values (its
// rows) and TN*KS B values (its columns), KS = BK/blockDim.x; the threads
// together cover each tile exactly once.  Requires blockDim.x == blockDim.y
// (true for all instantiations below).
// ===========================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v3_prefetch(int M, int N, int K, ACCUM alpha,
                                 const STORAGE *A, const STORAGE *B,
                                 ACCUM beta, ACCUM *C)
{
  extern __shared__ STORAGE smem[];
  const int SA_STR = BK + SMEM_PAD;
  const int SB_STR = BN + SMEM_PAD;
  STORAGE *sA = smem;                        // [BM][SA_STR]
  STORAGE *sB = smem + BM * SA_STR;          // [BK][SB_STR]

  constexpr int KS = (BK * TN) / BN;         // k-slice width per thread
                                              // (blockDim.x == BN/TN by launch)
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  const int rowA = m0 + threadIdx.y * TM;
  const int colB = n0 + threadIdx.x * TN;

  ACCUM acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0;

  STORAGE ra[TM * KS], rb[TN * KS];

  auto load_stage = [&](int k0) {
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      #pragma unroll
      for (int q = 0; q < KS; ++q)
        ra[i * KS + q] = A[(size_t)(rowA + i) * K + k0 + threadIdx.x * KS + q];
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      #pragma unroll
      for (int q = 0; q < KS; ++q)
        rb[j * KS + q] = B[(size_t)(k0 + threadIdx.y * KS + q) * N + colB + j];
  };
  auto store_stage = [&]() {
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      #pragma unroll
      for (int q = 0; q < KS; ++q)
        sA[(threadIdx.y * TM + i) * SA_STR + threadIdx.x * KS + q] = ra[i * KS + q];
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      #pragma unroll
      for (int q = 0; q < KS; ++q)
        sB[(threadIdx.y * KS + q) * SB_STR + threadIdx.x * TN + j] = rb[j * KS + q];
  };

  load_stage(0);
  store_stage();
  __syncthreads();

  for (int k0 = 0; k0 < K; k0 += BK) {
    if (k0 + BK < K) load_stage(k0 + BK);    // issue next global loads early

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      ACCUM fa[TM], fb[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) fa[i] = (ACCUM)sA[(threadIdx.y * TM + i) * SA_STR + k];
      #pragma unroll
      for (int j = 0; j < TN; ++j) fb[j] = (ACCUM)sB[k * SB_STR + threadIdx.x * TN + j];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += fa[i] * fb[j];
    }

    __syncthreads();                         // all threads done reading
    if (k0 + BK < K) {
      store_stage();                         // deposit prefetched tile
      __syncthreads();
    }
  }

  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      size_t c = (size_t)(rowA + i) * N + colB + j;
      C[c] = alpha * acc[i][j] + beta * C[c];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
static bool launch_v3(const LaunchArgs &a)
{
  dim3 thr(BN / TN, BM / TM);
  dim3 grid(a.N / BN, a.M / BM);
  size_t smem = (size_t)(BM * (BK + SMEM_PAD) + BK * (BN + SMEM_PAD)) * sizeof(STORAGE);
  if (!smem_optin((void *)&gemm_v3_prefetch<BM, BN, BK, TM, TN>, smem)) return false;
  gemm_v3_prefetch<BM, BN, BK, TM, TN><<<grid, thr, smem>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  return true;
}

// ===========================================================================
// V4 — tuned SIMT.  Double-buffered shared memory (the next tile is loaded
// while the current one is computed, with a single __syncthreads() per
// step), 4-wide vectorized cooperative global loads, and a transposed B
// tile so the register fragments are read contiguously.  The practical
// ceiling of CUDA-/stream-processor-only kernels.
// ===========================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v4_tuned(int M, int N, int K, ACCUM alpha,
                              const STORAGE *A, const STORAGE *B,
                              ACCUM beta, ACCUM *C)
{
  extern __shared__ STORAGE smem[];
  const int SA_STR  = BK + SMEM_PAD;
  const int SBT_STR = BK + SMEM_PAD;
  const int BUF_SZ  = BM * SA_STR + BN * SBT_STR;

  STORAGE *buf[2] = { smem, smem + BUF_SZ };

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int NT  = blockDim.x * blockDim.y;
  const int m0  = blockIdx.y * BM;
  const int n0  = blockIdx.x * BN;

  const int CKB = BK * (BN / 4);             // B chunks: (k, n-quad) pairs

  ACCUM acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0;

  auto load_all = [&](int k0, STORAGE *dst) {
    // A tile [BM][BK]: 4-element vectors along k
    for (int e = tid; e < BM * (BK / 4); e += NT) {
      int r = e / (BK / 4), kc = e % (BK / 4);
      Vec4<STORAGE> v = *(const Vec4<STORAGE> *)&A[(size_t)(m0 + r) * K + k0 + kc * 4];
      *(Vec4<STORAGE> *)&dst[r * SA_STR + kc * 4] = v;
    }
    // B tile [BK][BN]: 4-element vectors along n, stored transposed
    STORAGE *sBt = dst + BM * SA_STR;        // [BN][SBT_STR]
    for (int e = tid; e < CKB; e += NT) {
      int k = e / (BN / 4), nc = e % (BN / 4);
      Vec4<STORAGE> v = *(const Vec4<STORAGE> *)&B[(size_t)(k0 + k) * N + n0 + nc * 4];
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        sBt[(nc * 4 + j) * SBT_STR + k] = v.v[j];
    }
  };

  load_all(0, buf[0]);
  __syncthreads();

  const int KT = K / BK;
  for (int kt = 0; kt < KT; ++kt) {
    int cur = kt & 1;
    if (kt + 1 < KT)
      load_all((kt + 1) * BK, buf[cur ^ 1]);     // overlaps with compute below

    STORAGE *sA  = buf[cur];
    STORAGE *sBt = buf[cur] + BM * SA_STR;

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      ACCUM fa[TM], fb[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) fa[i] = (ACCUM)sA[(threadIdx.y * TM + i) * SA_STR + k];
      #pragma unroll
      for (int j = 0; j < TN; ++j) fb[j] = (ACCUM)sBt[(threadIdx.x * TN + j) * SBT_STR + k];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += fa[i] * fb[j];
    }
    __syncthreads();
  }

  const int ra = m0 + threadIdx.y * TM;
  const int cb = n0 + threadIdx.x * TN;
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      size_t c = (size_t)(ra + i) * N + cb + j;
      C[c] = alpha * acc[i][j] + beta * C[c];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
static bool launch_v4(const LaunchArgs &a)
{
  dim3 thr(BN / TN, BM / TM);
  dim3 grid(a.N / BN, a.M / BM);
  size_t smem = 2 * (size_t)(BM * (BK + SMEM_PAD) + BN * (BK + SMEM_PAD)) * sizeof(STORAGE);
  if (!smem_optin((void *)&gemm_v4_tuned<BM, BN, BK, TM, TN>, smem)) return false;
  gemm_v4_tuned<BM, BN, BK, TM, TN><<<grid, thr, smem>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  return true;
}

// ===========================================================================
// V5 — tensor cores via the WMMA API.
//   NVIDIA: nvcuda::wmma (mma.h) — fp16 inputs on FP16 builds, TF32 on
//           FP32 builds (compute capability 8.0+).
//   AMD:    rocWMMA (MFMA instructions, gfx908+) — fp16 inputs only.
// Each wave computes a 32x32 warp tile (2x2 fragments of 16x16).  The
// accumulator fragments are staged through shared memory so the alpha/beta
// epilogue runs cooperatively across the whole block.
// ===========================================================================
#ifdef ENABLE_V5

#ifdef _HIP
#include <rocwmma/rocwmma.hpp>
#define WM_NS        rocwmma
#define WM_ROW       rocwmma::row_major
#define WM_MEM_ROW   rocwmma::mem_row_major
#else
#include <mma.h>
#define WM_NS        nvcuda::wmma
#define WM_ROW       nvcuda::wmma::row_major          // fragment layout TAG
#define WM_MEM_ROW   nvcuda::wmma::mem_row_major      // store_matrix_sync enum
#endif

#ifdef DTYPE_FP16
#  ifdef _HIP
     typedef rocwmma::float16_t WM_T;
#  else
     typedef __half WM_T;
#  endif
#  define WM_K 16
#else /* FP32 build -> TF32, NVIDIA only */
#  define WM_T float
#  define WM_K 8
#endif

// Thread-block geometry: WAVES_X x WAVES_Y waves, WM_TPW threads each.
#ifdef _HIP
constexpr int WM_TPW  = 64;   // CDNA wavefront is 64 lanes
constexpr int WM_WAVES_X = 1, WM_WAVES_Y = 2;   // block tile 64 x 32
#else
constexpr int WM_TPW  = 32;   // NVIDIA warp is 32 lanes
constexpr int WM_WAVES_X = 2, WM_WAVES_Y = 2;   // block tile 64 x 64
#endif
constexpr int WM_BM = WM_WAVES_Y * 32;
constexpr int WM_BN = WM_WAVES_X * 32;

__global__ void gemm_v5_wmma(int M, int N, int K, ACCUM alpha,
                             const STORAGE *A, const STORAGE *B,
                             ACCUM beta, ACCUM *C)
{
  constexpr int WAVES = WM_WAVES_X * WM_WAVES_Y;
  constexpr int SP = 32 + 4;                 // staged tile pitch (pad to kill conflicts)

  __shared__ float stage[WAVES * 32 * SP];

#ifdef DTYPE_FP16
  WM_NS::fragment<WM_NS::matrix_a, 16, 16, WM_K, WM_T, WM_ROW> af[2];
  WM_NS::fragment<WM_NS::matrix_b, 16, 16, WM_K, WM_T, WM_ROW> bf[2];
#else
  WM_NS::fragment<WM_NS::matrix_a, 16, 16, WM_K, WM_NS::precision::tf32, WM_ROW> af[2];
  WM_NS::fragment<WM_NS::matrix_b, 16, 16, WM_K, WM_NS::precision::tf32, WM_ROW> bf[2];
#endif
  WM_NS::fragment<WM_NS::accumulator, 16, 16, WM_K, float> acc[2][2];
  #pragma unroll
  for (int i = 0; i < 2; ++i)
    #pragma unroll
    for (int j = 0; j < 2; ++j) WM_NS::fill_fragment(acc[i][j], 0.0f);

  const int wid = (threadIdx.y * (WM_TPW * WM_WAVES_X) + threadIdx.x) / WM_TPW;
  const int wm  = wid / WM_WAVES_X, wn = wid % WM_WAVES_X;
  const int m0  = blockIdx.y * WM_BM + wm * 32;
  const int n0  = blockIdx.x * WM_BN + wn * 32;

  for (int k0 = 0; k0 < K; k0 += WM_K) {
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      WM_NS::load_matrix_sync(af[i], (const WM_T *)&A[(size_t)(m0 + i * 16) * K + k0], K);
    #pragma unroll
    for (int j = 0; j < 2; ++j)
      WM_NS::load_matrix_sync(bf[j], (const WM_T *)&B[(size_t)k0 * N + n0 + j * 16], N);
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      #pragma unroll
      for (int j = 0; j < 2; ++j)
        WM_NS::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
  }

  // Dump accumulators to shared memory, then apply alpha/beta cooperatively.
  #pragma unroll
  for (int i = 0; i < 2; ++i)
    #pragma unroll
    for (int j = 0; j < 2; ++j)
      WM_NS::store_matrix_sync(&stage[wid * 32 * SP + i * 16 * SP + j * 16],
                               acc[i][j], SP, WM_MEM_ROW);
  __syncthreads();

  const int bm0 = blockIdx.y * WM_BM, bn0 = blockIdx.x * WM_BN;
  const int nthr = WM_TPW * WM_WAVES_X * WM_WAVES_Y;
  for (int t = threadIdx.y * (WM_TPW * WM_WAVES_X) + threadIdx.x; t < WM_BM * WM_BN; t += nthr) {
    int r = t / WM_BN, cix = t % WM_BN;
    int w = (r / 32) * WM_WAVES_X + (cix / 32);          // wave owning the element
    float v = stage[w * 32 * SP + (r % 32) * SP + (cix % 32)];
    size_t c = (size_t)(bm0 + r) * N + bn0 + cix;
    C[c] = alpha * (ACCUM)v + beta * C[c];
  }
}

static bool launch_v5(const LaunchArgs &a)
{
  dim3 thr(WM_TPW * WM_WAVES_X, WM_WAVES_Y);
  dim3 grid(a.N / WM_BN, a.M / WM_BM);
  gemm_v5_wmma<<<grid, thr>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  return true;
}

#endif // ENABLE_V5

// ===========================================================================
// Level registry: per level, the curated tile configurations compiled into
// the binary.  -T selects among them at runtime (index 0 = textbook
// default).  Tile fields describe the config for divisibility checks.
// ===========================================================================
struct TileOpt {
  const char *label;
  LaunchFn    fn;
  int BM, BN, BK;             // block-tile granularity (V0/V5/BLAS: 0/loose)
};

static const TileOpt V0_OPTS[] = {{"default", launch_v0, 0, 0, 0}};

static const TileOpt V1_OPTS[] = {
  {"shared 16x16",  launch_v1<16>, 16, 16, 16},
  {"shared 8x8",    launch_v1<8>,  8,  8,  8},
  {"shared 32x32",  launch_v1<32>, 32, 32, 32},
};

static const TileOpt V2_OPTS[] = {
  {"32x32x16/4x4",   launch_v2<32, 32, 16, 4, 4>,   32, 32, 16},
  {"64x64x16/4x4",   launch_v2<64, 64, 16, 4, 4>,   64, 64, 16},
  {"64x64x32/8x8",   launch_v2<64, 64, 32, 8, 8>,   64, 64, 32},
};

static const TileOpt V3_OPTS[] = {
  {"64x64x16/4x4",   launch_v3<64, 64, 16, 4, 4>,   64, 64, 16},
  {"64x64x32/4x4",   launch_v3<64, 64, 32, 4, 4>,   64, 64, 32},
  {"128x64x16/8x4",  launch_v3<128, 64, 16, 8, 4>, 128, 64, 16},
};

static const TileOpt V4_OPTS[] = {
  {"64x64x16/8x4",   launch_v4<64, 64, 16, 8, 4>,   64, 64, 16},
  {"128x128x16/8x8", launch_v4<128, 128, 16, 8, 8>, 128, 128, 16},
  {"128x128x32/8x8", launch_v4<128, 128, 32, 8, 8>, 128, 128, 32},
};

#ifdef ENABLE_V5
#  ifdef DTYPE_FP16
static const TileOpt V5_OPTS[] = {{"wmma 16x16x16 fp16->fp32", launch_v5, 16, 16, 16}};
#  else
static const TileOpt V5_OPTS[] = {{"wmma 16x16x8 tf32",        launch_v5, 16, 16, 8}};
#  endif
#endif

struct LevelInfo {
  int             id;
  const char     *name;
  const TileOpt  *opts;
  int             nopts;
  bool            compiled;
};

static inline bool level_compiled(int id)
{
  if (!((LEVEL_MASK >> id) & 1)) return false;
  if (id == LEVEL_V5) {
#ifdef ENABLE_V5
    return true;
#else
    return false;
#endif
  }
  return true;
}

static inline const char *level_base_name(int id)
{
  switch (id) {
    case LEVEL_V0:   return "V0 naive (global only)";
    case LEVEL_V1:   return "V1 shared tiling";
    case LEVEL_V2:   return "V2 register tiling";
    case LEVEL_V3:   return "V3 + prefetch padding";
    case LEVEL_V4:   return "V4 tuned SIMT";
    case LEVEL_V5:   return "V5 tensor cores";
    default:         return "BLAS";
  }
}

static inline LevelInfo get_level(int id)
{
  LevelInfo li = {id, level_base_name(id), nullptr, 0, level_compiled(id)};
  switch (id) {
    case LEVEL_V0: li.opts = V0_OPTS; li.nopts = 1; break;
    case LEVEL_V1: li.opts = V1_OPTS; li.nopts = 3; break;
    case LEVEL_V2: li.opts = V2_OPTS; li.nopts = 3; break;
    case LEVEL_V3: li.opts = V3_OPTS; li.nopts = 3; break;
    case LEVEL_V4: li.opts = V4_OPTS; li.nopts = 3; break;
#ifdef ENABLE_V5
    case LEVEL_V5: li.opts = V5_OPTS; li.nopts = 1; break;
#endif
    default: break;   // BLAS: no tile options
  }
  return li;
}

// Per-level correctness tolerances for -v (element check |x-y|<=atol+rtol|y|).
static inline void level_tolerance(int id, double &atol, double &rtol)
{
#if defined(DTYPE_FP64)
  atol = 1e-6; rtol = 1e-10;
#elif defined(DTYPE_FP16)
  atol = 1e-2; rtol = 5e-3;
#else
  atol = 1e-2; rtol = 5e-3;
#ifdef ENABLE_V5_TF32
  if (id == LEVEL_V5) rtol = 5e-2;           // TF32 rounds inputs to 10 bits
#endif
#endif
}

#endif // KERNELS_CUH
