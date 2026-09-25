#include "portability.h"   // HIP/CUDA portability + backend metric headers
#include "cli.h"           // CLI argument parsing
#include "metrics.h"       // GPU telemetry monitoring
#include "util.cuh"
#include <stdlib.h>
#include <assert.h>
#include <chrono>          // Wall-clock timing for time-based execution

using namespace std;

// Select precision via -DDOUBLE (double precision) or default (single precision)
#ifdef DOUBLE
typedef double GEMM_FLOAT;
#else
typedef float GEMM_FLOAT;
#endif

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  GemmArgs args = parseArguments(argc, argv, 0 /* not sweep */);

  int      device_id        = args.device_id;
  int      gpu_id           = device_id; // GPU ID in BDF order (matches amd-smi)
  size_t   size = args.matrix_size;
  int      repeats          = args.repeats;
  double   target_minutes   = args.target_minutes;
  InitMode init_mode        = args.init_mode;

#ifdef _HIP
  // On AMD, GPU ID (BDF order, as shown by amd-smi) differs from HIP device index.
  // The -d flag specifies GPU ID; resolve it to the corresponding HIP device index.
  {
    int num_devices = 0;
    checkCuda(cudaGetDeviceCount(&num_devices));
    if (gpu_id < 0 || gpu_id >= num_devices) {
      cerr << "GPU ID " << gpu_id << " out of range (0.." << num_devices - 1 << ")" << endl;
      return 1;
    }
    int bdf_keys[64], hip_indices[64];
    for (int i = 0; i < num_devices && i < 64; i++) {
      cudaDeviceProp p;
      checkCuda(cudaGetDeviceProperties(&p, i));
      bdf_keys[i] = (p.pciDomainID << 16) | (p.pciBusID << 8) | p.pciDeviceID;
      hip_indices[i] = i;
    }
    for (int i = 1; i < num_devices; i++) {
      int kb = bdf_keys[i], ki = hip_indices[i];
      int j = i - 1;
      while (j >= 0 && bdf_keys[j] > kb) {
        bdf_keys[j + 1] = bdf_keys[j];
        hip_indices[j + 1] = hip_indices[j];
        j--;
      }
      bdf_keys[j + 1] = kb;
      hip_indices[j + 1] = ki;
    }
    device_id = hip_indices[gpu_id];
  }
#endif

  // Align to 256
  const size_t ALIGN = 256;
  size = (size + ALIGN - 1) / ALIGN * ALIGN;

  checkCuda(cudaSetDevice(device_id));

 cout << HLINE;

#ifdef DOUBLE
  cout << "Double Precision GEMMM sweep results :" << endl;
#else
  cout << "Single Precision GEMMM sweep results :" << endl;
#endif

  cout << HLINE;

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, device_id));

  cout << "Summary:" << endl
       << "GPU ID: "           << gpu_id           << endl
#ifdef _HIP
       << "HIP device: "       << device_id        << endl
#endif
       << "Device Name: "      << prop.name        << endl
       << "Matrix_dimension: " << size << endl;
  if (target_minutes > 0.0)
    cout << "Mode: time-based (" << target_minutes << " min)" << endl;
  else
    cout << "Repeats: " << repeats << endl;
  cout << "Init mode: " << (init_mode == INIT_RANDOM ? "random" : "constant") << endl;
#ifdef NAIVE
    cout << "Optimization: " << "NAIVE"  << endl;
#endif
#ifdef VENDOR
    cout << "Optimization: " << "VENDOR"  << endl;
#endif
    cout << HLINE;

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));

#if !defined(DOUBLE) && defined(TENSOR)
  checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
#endif

#ifdef METRICS
  GpuMonitor monitor;
  if (!monitor.init(device_id, gpu_id))
    return 1;
#endif

  // ---------------------------------------------------------------------------
  // Memory allocation and data initialization
  // ---------------------------------------------------------------------------
  size_t bytes = (size_t)size * (size_t)size * sizeof(GEMM_FLOAT);
  cout << "Allocating device variables with total size: " << 3 * bytes / 1e9 << " GB" << endl;
  cout << HLINE;

  GEMM_FLOAT *d_A, *d_B, *d_C;
  checkCuda(cudaMalloc(&d_A, bytes));
  checkCuda(cudaMalloc(&d_B, bytes));
  checkCuda(cudaMalloc(&d_C, bytes));

  long threads = 256;
  long blocks  = ((long)size * (long)size) / threads;
  if (init_mode == INIT_RANDOM)
    init_data<<<blocks, threads>>>(d_A, d_B, d_C, size, (unsigned long long)time(NULL));
  else
    init_data_constant<<<blocks, threads>>>(d_A, d_B, d_C, size);
  checkCuda(cudaDeviceSynchronize());

#ifdef DOUBLE
  const GEMM_FLOAT alf = 1.012,   bet = 4.019;
#else
  const GEMM_FLOAT alf = 1.0579f, bet = 0.0127f;
#endif
  const GEMM_FLOAT *alpha = &alf, *beta = &bet;

  cudaEvent_t ev_start, ev_stop;
  checkCuda(cudaEventCreate(&ev_start));
  checkCuda(cudaEventCreate(&ev_stop));

  // ---------------------------------------------------------------------------
  // Execution loop: time-based or fixed-count
  // ---------------------------------------------------------------------------
  double sum         = 0.0;
  int actual_repeats = 0;
  auto wall_start    = chrono::high_resolution_clock::now();

#ifdef METRICS
  monitor.start();
#endif

  while (true)
  {
    if (target_minutes > 0.0) {
      auto now = chrono::high_resolution_clock::now();
      if (chrono::duration<double>(now - wall_start).count() > target_minutes * 60.0)
        break;
    } else {
      if (actual_repeats >= repeats) break;
    }

    int64_t m = size, n = size, k = size;
    int64_t lda = m, ldb = k, ldc = m;

    checkCuda(cudaEventRecord(ev_start, 0));
#ifdef VENDOR
#ifdef DOUBLE
    cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
#else
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
#endif
#endif
#ifdef NAIVE
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    // Calculate how many blocks we need in the grid. 
    // Adding (TILE_SIZE - 1) ensures we round up if dimensions aren't perfect multiples of 16.
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, 
                       (m + TILE_SIZE - 1) / TILE_SIZE);
    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, *alpha, d_A, d_B, *beta, d_C);
#endif

    checkCuda(cudaEventRecord(ev_stop, 0));
    checkCuda(cudaEventSynchronize(ev_stop));
#ifdef VENDOR
    assert(stat == CUBLAS_STATUS_SUCCESS);
#endif
    assert(!cudaGetLastError());

    float elapsed_ms;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    sum += elapsed_ms / 1000.0f;
    actual_repeats++;
  }

#ifdef METRICS
  monitor.stop();
#endif


  if (actual_repeats == 0) actual_repeats = 1;

  checkCuda(cudaFree(d_A));
  checkCuda(cudaFree(d_B));
  checkCuda(cudaFree(d_C));


  // ---------------------------------------------------------------------------
  // Output
  // ---------------------------------------------------------------------------
  print_horizantal_line();
  print_stats_header();
  print_horizantal_line();
#ifdef METRICS
    MetricsAvg avg = monitor.averages();
  print_stats(size, sum, actual_repeats, avg);
  monitor.shutdown();
#else
  print_stats(size, sum, actual_repeats);
#endif
  print_horizantal_line();
  return 0;
}
