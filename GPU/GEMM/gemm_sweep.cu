#include "portability.h" // HIP/CUDA portability + backend metric headers
#include "cli.h"         // CLI argument parsing
#include "metrics.h"     // GPU telemetry monitoring

#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <chrono> // Wall-clock timing for time-based execution

using namespace std;

// Select precision via -DDOUBLE (double precision) or default (single precision)
#ifdef DOUBLE
typedef double GEMM_FLOAT;
#else
typedef float GEMM_FLOAT;
#endif

// ---------------------------------------------------------------------------
// Error handling — void return avoids HIP [[nodiscard]] warnings on callers.
// ---------------------------------------------------------------------------
const char *cublasGetErrorString(cublasStatus_t status)
{
  switch (status)
  {
#ifdef _HIP
  // Only codes guaranteed to exist in all hipBLAS versions (ROCm >= 6 removed
  // HIPBLAS_STATUS_ARCH_MISMATCH and HIPBLAS_STATUS_MAPPING_ERROR).
  case HIPBLAS_STATUS_SUCCESS:
    return "HIPBLAS_STATUS_SUCCESS";
  case HIPBLAS_STATUS_NOT_INITIALIZED:
    return "HIPBLAS_STATUS_NOT_INITIALIZED";
  case HIPBLAS_STATUS_ALLOC_FAILED:
    return "HIPBLAS_STATUS_ALLOC_FAILED";
  case HIPBLAS_STATUS_INVALID_VALUE:
    return "HIPBLAS_STATUS_INVALID_VALUE";
  case HIPBLAS_STATUS_EXECUTION_FAILED:
    return "HIPBLAS_STATUS_EXECUTION_FAILED";
  case HIPBLAS_STATUS_INTERNAL_ERROR:
    return "HIPBLAS_STATUS_INTERNAL_ERROR";
#else
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#endif
  }
  return "unknown error";
}

inline void checkCuda(cudaError_t result)
{
  if (result != cudaSuccess)
  {
    fprintf(stderr, "GPU Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
}

inline void checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "BLAS Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
}

// ---------------------------------------------------------------------------
// Kernel: Initialize matrices with random values
// ---------------------------------------------------------------------------
__global__ void init_data(GEMM_FLOAT *A, GEMM_FLOAT *B, GEMM_FLOAT *C, size_t size, unsigned long long seed)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)size * (long)size;
  if (idx >= total)
    return;

  curandState state;
  curand_init(seed, idx, 0, &state);
  A[idx] = (GEMM_FLOAT)curand_uniform(&state);
  B[idx] = (GEMM_FLOAT)curand_uniform(&state);
  C[idx] = (GEMM_FLOAT)curand_uniform(&state);
}

// ---------------------------------------------------------------------------
// Kernel: Initialize matrices with a fixed constant value
// ---------------------------------------------------------------------------
__global__ void init_data_constant(GEMM_FLOAT *A, GEMM_FLOAT *B, GEMM_FLOAT *C, size_t size)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)size * (long)size;
  if (idx >= total)
    return;

  A[idx] = (GEMM_FLOAT)0.1529;
  B[idx] = (GEMM_FLOAT)1.2631;
  C[idx] = (GEMM_FLOAT)0.0;
}

#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void gemm_kernel(int M, int N, int K, 
                            float alpha, const GEMM_FLOAT *A, 
                            const GEMM_FLOAT *B, float beta, GEMM_FLOAT *C) {
    // 1D Thread and Block coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column index in the flattened 1D C matrix
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Allocate shared memory as flat 1D arrays
    __shared__ GEMM_FLOAT sA[TILE_SIZE * TILE_SIZE];
    __shared__ GEMM_FLOAT sB[TILE_SIZE * TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles of the inputs
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        
        // Load A into 1D shared memory
        // 1D Shared Memory Index: ty * TILE_SIZE + tx
        // 1D Global Memory Index: row * K + (t * TILE_SIZE + tx)
        if (row < M && (t * TILE_SIZE + tx) < K) {
            sA[ty * TILE_SIZE + tx] = A[row * K + (t * TILE_SIZE + tx)];
        } else {
            sA[ty * TILE_SIZE + tx] = 0.0f;
        }

        // Load B into 1D shared memory
        // 1D Global Memory Index: (t * TILE_SIZE + ty) * N + col
        if ((t * TILE_SIZE + ty) < K && col < N) {
            sB[ty * TILE_SIZE + tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            sB[ty * TILE_SIZE + tx] = 0.0f;
        }

        __syncthreads();

        // Perform the dot product using purely 1D indexing
        for (int i = 0; i < TILE_SIZE; ++i) {
            // sA accesses row 'ty' and column 'i' -> ty * TILE_SIZE + i
            // sB accesses row 'i' and column 'tx' -> i * TILE_SIZE + tx
            sum += sA[ty * TILE_SIZE + i] * sB[i * TILE_SIZE + tx];
        }

        __syncthreads();
    }

    // Write final output to 1D global memory array C
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  GemmArgs args = parseArguments(argc, argv, 1 /* is_sweep */);

  int device_id = args.device_id;
  int gpu_id = device_id;                     // GPU ID in BDF order (matches amd-smi)
  size_t matrix_dimension = args.matrix_size; // upper bound of sweep
  int repeats = args.repeats;
  double target_minutes = args.target_minutes;
  InitMode init_mode = args.init_mode;

#ifdef _HIP
  // On AMD, GPU ID (BDF order, as shown by amd-smi) differs from HIP device index.
  // The -d flag specifies GPU ID; resolve it to the corresponding HIP device index.
  {
    int num_devices = 0;
    checkCuda(cudaGetDeviceCount(&num_devices));
    if (gpu_id < 0 || gpu_id >= num_devices)
    {
      cerr << "GPU ID " << gpu_id << " out of range (0.." << num_devices - 1 << ")" << endl;
      return 1;
    }
    int bdf_keys[64], hip_indices[64];
    for (int i = 0; i < num_devices && i < 64; i++)
    {
      cudaDeviceProp p;
      checkCuda(cudaGetDeviceProperties(&p, i));
      bdf_keys[i] = (p.pciDomainID << 16) | (p.pciBusID << 8) | p.pciDeviceID;
      hip_indices[i] = i;
    }
    for (int i = 1; i < num_devices; i++)
    {
      int kb = bdf_keys[i], ki = hip_indices[i];
      int j = i - 1;
      while (j >= 0 && bdf_keys[j] > kb)
      {
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
       << "GPU ID: " << gpu_id << endl
#ifdef _HIP
       << "HIP device: " << device_id << endl
#endif
       << "Device Name: " << prop.name << endl
       << "Max Matrix dimension: " << matrix_dimension << endl;
  if (target_minutes > 0.0)
    cout << "Mode: time-based (" << target_minutes << " min per size)" << endl;
  else
    cout << "Repeats per size: " << repeats << endl;
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

  size_t max_bytes = matrix_dimension * matrix_dimension * sizeof(GEMM_FLOAT);
  cout << "Max allocation per matrix: " << max_bytes / 1e9 << " GB" << endl ;
  cout << HLINE;

#ifdef DOUBLE
  const GEMM_FLOAT alf = 1.012, bet = 4.019;
#else
  const GEMM_FLOAT alf = 1.0579f, bet = 0.0127f;
#endif
  const GEMM_FLOAT *alpha = &alf, *beta = &bet;

  cudaEvent_t ev_start, ev_stop;
  checkCuda(cudaEventCreate(&ev_start));
  checkCuda(cudaEventCreate(&ev_stop));

  // ---------------------------------------------------------------------------
  // Sweep over matrix sizes
  // ---------------------------------------------------------------------------
  for (size_t size = 2560; size <= matrix_dimension; size = (size_t)(size * 1.2))
  {
    const size_t ALIGN = 256;
    size = (size + ALIGN - 1) / ALIGN * ALIGN;

    size_t bytes = (size_t)size * (size_t)size * sizeof(GEMM_FLOAT);

    GEMM_FLOAT *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, bytes));
    checkCuda(cudaMalloc(&d_B, bytes));
    checkCuda(cudaMalloc(&d_C, bytes));

    long threads = 256;
    long blocks = ((long)size * (long)size) / threads;
    if (init_mode == INIT_RANDOM)
      init_data<<<blocks, threads>>>(d_A, d_B, d_C, size, (unsigned long long)time(NULL));
    else
      init_data_constant<<<blocks, threads>>>(d_A, d_B, d_C, size);
    checkCuda(cudaDeviceSynchronize());

#ifdef METRICS
    monitor.start();
#endif

    double sum = 0.0;
    int actual_repeats = 0;

    // --- Inner execution loop: time-based or fixed-count ---
    auto wall_start = chrono::high_resolution_clock::now();

    while (true)
    {
      if (target_minutes > 0.0)
      {
        auto now = chrono::high_resolution_clock::now();
        if (chrono::duration<double>(now - wall_start).count() > target_minutes * 60.0)
          break;
      }
      else
      {
        if (actual_repeats >= repeats)
          break;
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

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    if (actual_repeats == 0)
      actual_repeats = 1;

    double totalFlops = 2.0 * size * size * size * actual_repeats;

    // --- Print results for this size ---
#ifdef METRICS
    MetricsAvg avg = monitor.averages();
    cout << "size " << size
         << " | Time: " << sum << " s"
         << " | Repeats: " << actual_repeats
         << " | Perf: " << totalFlops / sum / 1e12 << " TFlop/s"
         << " | Avg Power: " << avg.power << " W"
         << " | Avg Clock: " << avg.clock << " MHz"
         << " | Avg Temp: " << avg.temp << " C"
         << " | " << GpuMonitor::util_label() << ": " << avg.gpu_util << " %"
         << " | Mem Util: " << avg.mem_util << " %" << endl;
#else
    cout << "size " << size
         << " | Time: " << sum << " s"
         << " | Repeats: " << actual_repeats
         << " | Perf: " << totalFlops / sum / 1e12 << " TFlop/s" << endl;
#endif
  }

#ifdef METRICS
  monitor.shutdown();
#endif

  return 0;
}
