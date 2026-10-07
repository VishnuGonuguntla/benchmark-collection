#include "portability.h"   // HIP/CUDA portability + dtype typedefs
#include "cli.h"            // CLI argument parsing
#include "metrics.h"        // GPU telemetry monitoring
#include "util.cuh"         // kernels: init / compare, output helpers
#include "kernels.cuh"      // optimization ladder V0..V5
#include "runner.h"         // shared benchmark harness
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <algorithm>

using namespace std;

// ---------------------------------------------------------------------------
// Sweep over rectangular problem sizes: the base shape (M, N, K) given by
// -m/-n/-k is multiplied by 1.2 each step (all three dims grow
// proportionally), every dim re-aligned to 256, until the smallest dim
// exceeds the cap (-s, default SWEEPSIZE).
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  GemmArgs args = parseArguments(argc, argv, 1 /* is sweep */);

  if (args.tile_cfg == TILE_HELP) { print_tile_help(); return 0; }

  int gpu_id    = args.device_id;
  int device_id = resolve_device(gpu_id);
  checkCuda(cudaSetDevice(device_id));

  size_t baseM = args.M, baseN = args.N, baseK = args.K;
  align_dims(baseM, baseN, baseK);
  size_t cap = args.max_dim;

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, device_id));

  std::cout << "# GEMM ladder sweep (" << GEMM_DT_NAME << ", "
#ifdef _HIP
       << "HIP/ROCm"
#else
       << "CUDA"
#endif
       << ", " << prop.name << ") base " << baseM << "x" << baseN << "x" << baseK
       << " cap " << cap
       << (args.target_minutes > 0.0
             ? ("  minutes/level " + std::to_string(args.target_minutes))
             : ("  reps " + std::to_string(args.repeats)))
       << endl;

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));

#ifdef METRICS
  GpuMonitor monitor;
  if (!monitor.init(device_id, gpu_id))
    return 1;
#endif

#if defined(DTYPE_FP64)
  const ACCUM alf = 1.012, bet = 4.019;
#else
  const ACCUM alf = 1.0579f, bet = 0.0127f;
#endif

  std::vector<LevelRow> all_rows;

  const size_t oM = baseM, oN = baseN, oK = baseK;   // original aligned base
  size_t prevM = 0, prevN = 0, prevK = 0;

  for (double f = 1.0; ; f *= 1.2) {
    size_t M = (size_t)(oM * f); align_one(M);
    size_t N = (size_t)(oN * f); align_one(N);
    size_t K = (size_t)(oK * f); align_one(K);

    if (std::min(M, std::min(N, K)) > cap) break;
    if (M == prevM && N == prevN && K == prevK) break;  // growth stalled
    prevM = M; prevN = N; prevK = K;

    size_t bytesA = M * K * sizeof(STORAGE);
    size_t bytesB = K * N * sizeof(STORAGE);
    size_t bytesC = M * N * sizeof(ACCUM);
    std::cout << "size " << M << "x" << N << "x" << K
         << " footprint(A+B+C) " << (bytesA + bytesB + bytesC) / 1e9 << " GB" << endl;

    STORAGE *d_A, *d_B;
    ACCUM   *d_C, *d_C0;
    checkCuda(cudaMalloc(&d_A, bytesA));
    checkCuda(cudaMalloc(&d_B, bytesB));
    checkCuda(cudaMalloc(&d_C, bytesC));
    checkCuda(cudaMalloc(&d_C0, bytesC));

    long threads = 256;
    size_t nA = M * K, nB = K * N, nC = M * N;
    long blocks = (long)((std::max(nA, std::max(nB, nC)) + threads - 1) / threads);
    if (args.init_mode == INIT_RANDOM)
      init_data<<<blocks, threads>>>(d_A, d_B, d_C, nA, nB, nC, (unsigned long long)time(NULL));
    else
      init_data_constant<<<blocks, threads>>>(d_A, d_B, d_C, nA, nB, nC);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(d_C0, d_C, bytesC, cudaMemcpyDeviceToDevice));

    LadderCtx cx;
    cx.handle  = handle;
    cx.M = M; cx.N = N; cx.K = K;
    cx.alpha = alf; cx.beta = bet;
    cx.dA = d_A; cx.dB = d_B; cx.dC = d_C; cx.dC0 = d_C0;
    cx.dCref = nullptr;                 // verify is fixed-size mode only
    cx.repeats = args.repeats;
    cx.minutes = args.target_minutes;
    cx.level_mask = args.levels;
    cx.tile_cfg = args.tile_cfg;
    cx.verify = false;
    cx.prop = &prop;
#ifdef METRICS
    cx.monitor = &monitor;
#endif

    std::vector<LevelRow> rows = run_ladder(cx);
    all_rows.insert(all_rows.end(), rows.begin(), rows.end());
    if (args.csv_path) write_csv(args.csv_path, rows);

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));
    checkCuda(cudaFree(d_C0));
  }

  // CSV (-c) already carries every row; the combined table would duplicate it.
  if (!args.csv_path) print_table(all_rows, true /* show shape */);

  checkCublas(cublasDestroy(handle));
#ifdef METRICS
  monitor.shutdown();
#endif
  return 0;
}
