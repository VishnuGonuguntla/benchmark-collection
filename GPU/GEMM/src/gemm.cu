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
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  GemmArgs args = parseArguments(argc, argv, 0 /* not sweep */);

  if (args.tile_cfg == TILE_HELP) { print_tile_help(); return 0; }

  int gpu_id    = args.device_id;
  int device_id = resolve_device(gpu_id);

  size_t M = args.M, N = args.N, K = args.K;
  align_dims(M, N, K);

  checkCuda(cudaSetDevice(device_id));

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, device_id));

  cout << HLINE;
  cout << "GEMM optimization-ladder benchmark (" << GEMM_DT_NAME << ", "
#ifdef _HIP
       << "HIP/ROCm"
#else
       << "CUDA"
#endif
       << "):" << endl << HLINE;
  cout << "Summary:" << endl
       << "GPU ID: "      << gpu_id << endl
#ifdef _HIP
       << "HIP device: "  << device_id << endl
#endif
       << "Device Name: " << prop.name << endl
       << "Shape: C(" << M << "x" << N << ") = A(" << M << "x" << K
       << ") * B(" << K << "x" << N << ")" << endl;
  if (args.target_minutes > 0.0)
    cout << "Mode: time-based (" << args.target_minutes << " min per level)" << endl;
  else
    cout << "Repeats per level: " << args.repeats << endl;
  cout << "Init mode: " << (args.init_mode == INIT_RANDOM ? "random" : "constant") << endl;
  cout << "Levels (compiled): ";
  {
    std::vector<int> lv = selected_levels(args.levels, prop);
    for (size_t i = 0; i < lv.size(); i++) cout << (i ? "," : "") << lv[i];
    if ((args.levels == 0 || ((args.levels >> LEVEL_BLAS) & 1)) &&
        ((LEVEL_MASK >> LEVEL_BLAS) & 1))
      cout << (lv.empty() ? "" : ",") << "blas";
  }
  cout << endl;
  cout << "Datatype: " << GEMM_DT_NAME
#ifdef ENABLE_V5
     << " (tensor-core level enabled)"
#endif
       << endl;
  cout << HLINE;

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));

#ifdef METRICS
  GpuMonitor monitor;
  if (!monitor.init(device_id, gpu_id))
    return 1;
#endif

  // ---------------------------------------------------------------------------
  // Memory allocation and data initialization
  // ---------------------------------------------------------------------------
  size_t bytesA = M * K * sizeof(STORAGE);
  size_t bytesB = K * N * sizeof(STORAGE);
  size_t bytesC = M * N * sizeof(ACCUM);
  cout << "Allocating device buffers: A " << bytesA / 1e9 << " GB + B "
       << bytesB / 1e9 << " GB + C " << bytesC / 1e9 << " GB (x"
       << (args.verify ? 3 : 2) << " copies of C)" << endl << HLINE;

  STORAGE *d_A, *d_B;
  ACCUM   *d_C, *d_C0, *d_Cref = nullptr;
  checkCuda(cudaMalloc(&d_A, bytesA));
  checkCuda(cudaMalloc(&d_B, bytesB));
  checkCuda(cudaMalloc(&d_C, bytesC));
  checkCuda(cudaMalloc(&d_C0, bytesC));
  if (args.verify) checkCuda(cudaMalloc(&d_Cref, bytesC));

  long threads = 256;
  size_t nA = M * K, nB = K * N, nC = M * N;
  long blocks = (long)((std::max(nA, std::max(nB, nC)) + threads - 1) / threads);
  if (args.init_mode == INIT_RANDOM)
    init_data<<<blocks, threads>>>(d_A, d_B, d_C, nA, nB, nC, (unsigned long long)time(NULL));
  else
    init_data_constant<<<blocks, threads>>>(d_A, d_B, d_C, nA, nB, nC);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(d_C0, d_C, bytesC, cudaMemcpyDeviceToDevice));

#if defined(DTYPE_FP64)
  const ACCUM alf = 1.012, bet = 4.019;
#else
  const ACCUM alf = 1.0579f, bet = 0.0127f;
#endif

  LadderCtx cx;
  cx.handle  = handle;
  cx.M = M; cx.N = N; cx.K = K;
  cx.alpha = alf; cx.beta = bet;
  cx.dA = d_A; cx.dB = d_B; cx.dC = d_C; cx.dC0 = d_C0; cx.dCref = d_Cref;
  cx.repeats = args.repeats;
  cx.minutes = args.target_minutes;
  cx.level_mask = args.levels;
  cx.tile_cfg = args.tile_cfg;
  cx.verify = args.verify != 0;
  cx.prop = &prop;
#ifdef METRICS
  cx.monitor = &monitor;
#endif

  std::vector<LevelRow> rows = run_ladder(cx);

  // ---------------------------------------------------------------------------
  // Output
  // ---------------------------------------------------------------------------
  cout << HLINE;
  print_table(rows, false);
  cout << HLINE;
  if (args.csv_path) write_csv(args.csv_path, rows);

  checkCuda(cudaFree(d_A));
  checkCuda(cudaFree(d_B));
  checkCuda(cudaFree(d_C));
  checkCuda(cudaFree(d_C0));
  if (d_Cref) checkCuda(cudaFree(d_Cref));
  checkCublas(cublasDestroy(handle));
#ifdef METRICS
  monitor.shutdown();
#endif
  return 0;
}
