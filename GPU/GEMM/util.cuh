#ifndef __UTIL_H
#define __UTIL_H
#include "portability.h" // HIP/CUDA portability + backend metric headers
#include <iostream>
#include <iomanip>
#include <errno.h>
#include <string.h>

#ifdef DOUBLE
typedef double GEMM_FLOAT;
#else
typedef float GEMM_FLOAT;
#endif

// ---------------------------------------------------------------------------
// Kernel: Initialize matrices with random values
// ---------------------------------------------------------------------------
__global__ void init_data(GEMM_FLOAT *A, GEMM_FLOAT *B, GEMM_FLOAT *C, size_t size, unsigned long long seed)
{
  size_t idx   = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = size * size;
  if (idx >= total) return;

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
  size_t idx   = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = size * size;
  if (idx >= total) return;

  A[idx] = (GEMM_FLOAT)0.1529;
  B[idx] = (GEMM_FLOAT)1.2631;
  C[idx] = (GEMM_FLOAT)0.0;
}

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
  case HIPBLAS_STATUS_SUCCESS:           return "HIPBLAS_STATUS_SUCCESS";
  case HIPBLAS_STATUS_NOT_INITIALIZED:   return "HIPBLAS_STATUS_NOT_INITIALIZED";
  case HIPBLAS_STATUS_ALLOC_FAILED:      return "HIPBLAS_STATUS_ALLOC_FAILED";
  case HIPBLAS_STATUS_INVALID_VALUE:     return "HIPBLAS_STATUS_INVALID_VALUE";
  case HIPBLAS_STATUS_EXECUTION_FAILED:  return "HIPBLAS_STATUS_EXECUTION_FAILED";
  case HIPBLAS_STATUS_INTERNAL_ERROR:    return "HIPBLAS_STATUS_INTERNAL_ERROR";
#else
  case CUBLAS_STATUS_SUCCESS:            return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:       return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:      return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:      return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:      return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:   return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:     return "CUBLAS_STATUS_INTERNAL_ERROR";
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




#ifdef METRICS
void print_horizantal_line() {
    std::cout << "+-------+-----------+---------+---------------+-----------+------------+----------+-------------+------------+\n";
}
// 2. Prints the centered/aligned column headers
void print_stats_header() {
    std::cout << std::left;
    std::cout << "| " << std::setw(5)  << "Size"
              << " | " << std::setw(9)  << "Time (s)"
              << " | " << std::setw(7)  << "Repeats"
              << " | " << std::setw(13) << "Perf(TFlop/s)"
              << " | " << std::setw(9)  << "Power (W)"
              << " | " << std::setw(9)  << "Clock(MHz)"
              << " | " << std::setw(8)  << "Temp (C)"
              << " | " << std::setw(11) << GpuMonitor::util_label()
              << " | " << std::setw(10) << "Mem Util"
              << " |\n";
}

// 3. Prints the actual benchmark metrics row
void print_stats(size_t size, double sum, int repeats, MetricsAvg avg) {
    double total_flops = 2.0 * size * size * size * repeats;
    double tflops = (sum > 0.0) ? (total_flops / sum / 1e12) : 0.0;

    std::cout << std::left << std::fixed;
    std::cout << "| " << std::setw(5)  << size
              << " | " << std::setw(9)  << std::setprecision(4) << sum
              << " | " << std::setw(7)  << repeats
              << " | " << std::setw(13) << std::setprecision(3) << tflops
              << " | " << std::setw(9)  << std::setprecision(1) << avg.power
              << " | " << std::setw(10)  << std::setprecision(1) << avg.clock
              << " | " << std::setw(8)  << std::setprecision(1) << avg.temp
              << " | " << std::setw(9)  << std::setprecision(1) << avg.gpu_util << " %"
              << " | " << std::setw(8)  << std::setprecision(1) << avg.mem_util << " %"
              << " |\n";
}
#else
void print_horizantal_line() {
    std::cout << "+-------+-----------+---------+---------------+\n";
}
void print_stats_header() {
    std::cout << std::left;
    std::cout << "| " << std::setw(5)  << "Size"
              << " | " << std::setw(9)  << "Time (s)"
              << " | " << std::setw(7)  << "Repeats"
              << " | " << std::setw(13) << "Perf(TFlop/s)"
              << " |\n";
}

static void print_stats(size_t size, double sum, int repeats) {
  double total_flops = 2.0 * size * size * size * repeats;
  double tflops = (sum > 0.0) ? (total_flops / sum / 1e12) : 0.0;

  std::cout << std::left;
  std::cout << "| " << std::setw(5)  << size
            << " | " << std::setw(9)  << std::setprecision(4) << sum
            << " | " << std::setw(7)  << repeats
            << " | " << std::setw(13) <<  std::setprecision(3) << tflops
            << " |\n";
}
#endif

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

#endif