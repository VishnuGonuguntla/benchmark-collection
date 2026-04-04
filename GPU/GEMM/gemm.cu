#include "portability.h"   // HIP/CUDA portability + backend metric headers
#include "cli.h"           // CLI argument parsing

#include <iostream>
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

// ---------------------------------------------------------------------------
// Kernel: Initialize matrices with random values
// ---------------------------------------------------------------------------
__global__ void init_data(GEMM_FLOAT *A, GEMM_FLOAT *B, GEMM_FLOAT *C, int size, unsigned long long seed)
{
  long idx   = blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)size * (long)size;
  if (idx >= total) return;

  curandState state;
  curand_init(seed, idx, 0, &state);
  A[idx] = (GEMM_FLOAT)curand_uniform(&state);
  B[idx] = (GEMM_FLOAT)curand_uniform(&state);
  C[idx] = (GEMM_FLOAT)curand_uniform(&state);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  GemmArgs args = parseArguments(argc, argv, 0 /* not sweep */);

  int    device_id        = args.device_id;
  int    gpu_id           = device_id; // GPU ID in BDF order (matches amd-smi)
  size_t matrix_dimension = args.matrix_size;
  int    repeats          = args.repeats;
  double target_minutes   = args.target_minutes;

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
  matrix_dimension = (matrix_dimension + ALIGN - 1) / ALIGN * ALIGN;

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
       << "Matrix_dimension: " << matrix_dimension << endl;
  if (target_minutes > 0.0)
    cout << "Mode: time-based (" << target_minutes << " min)" << endl;
  else
    cout << "Repeats: " << repeats << endl;
  cout << HLINE;

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));

#if !defined(DOUBLE) && defined(TENSOR)
  checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
#endif

  // ---------------------------------------------------------------------------
  // Telemetry — single flag -DMETRICS selects NVML (CUDA) or ROCm SMI (HIP)
  // ---------------------------------------------------------------------------
#ifdef METRICS
  double             totalPower = 0.0, totalClock = 0.0, totalTemp = 0.0;
  unsigned long long totalGpuUtil = 0, totalMemUtil = 0;
  int                monitor_samples = 0;

#  ifndef _HIP  // --- NVML (CUDA) ---
  nvmlReturn_t nvmlResult;
  nvmlDevice_t nvmlDev;
  nvmlResult = nvmlInit();
  if (nvmlResult != NVML_SUCCESS) {
    cerr << "Failed to init NVML: " << nvmlErrorString(nvmlResult) << endl;
    return 1;
  }
  nvmlResult = nvmlDeviceGetHandleByIndex(device_id, &nvmlDev);
  if (nvmlResult != NVML_SUCCESS) {
    cerr << "Failed to get NVML device: " << nvmlErrorString(nvmlResult) << endl;
    return 1;
  }
#  else         // --- ROCm SMI (HIP) ---
  rsmi_status_t rsmiResult = rsmi_init(0);
  if (rsmiResult != RSMI_STATUS_SUCCESS) {
    const char *err_str; rsmi_status_string(rsmiResult, &err_str);
    cerr << "Failed to init ROCm SMI: " << err_str << endl;
    return 1;
  }
  // gpu_id matches ROCm SMI device ordering (both use BDF/PCI bus order)
  uint32_t rsmi_dev_idx = (uint32_t)gpu_id;
#  endif
#endif // METRICS

  // ---------------------------------------------------------------------------
  // Memory allocation and data initialization
  // ---------------------------------------------------------------------------
  size_t bytes = (size_t)matrix_dimension * (size_t)matrix_dimension * sizeof(GEMM_FLOAT);
  cout << "Allocating device variables with total size: " << 3 * bytes / 1e9 << " GB" << endl;
  cout << HLINE;

  GEMM_FLOAT *d_A, *d_B, *d_C;
  checkCuda(cudaMalloc(&d_A, bytes));
  checkCuda(cudaMalloc(&d_B, bytes));
  checkCuda(cudaMalloc(&d_C, bytes));

  long threads = 256;
  long blocks  = ((long)matrix_dimension * (long)matrix_dimension) / threads;
  init_data<<<blocks, threads>>>(d_A, d_B, d_C, matrix_dimension, (unsigned long long)time(NULL));
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

  while (true)
  {
    if (target_minutes > 0.0) {
      auto now = chrono::high_resolution_clock::now();
      if (chrono::duration<double>(now - wall_start).count() > target_minutes * 60.0)
        break;
    } else {
      if (actual_repeats >= repeats) break;
    }

    int m = matrix_dimension, n = matrix_dimension, k = matrix_dimension;
    int lda = m, ldb = k, ldc = m;

    checkCuda(cudaEventRecord(ev_start, 0));

#ifdef DOUBLE
    cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
#else
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
#endif

    // --- Sample telemetry while GPU is still computing (before sync) ---
#ifdef METRICS
#  ifndef _HIP  // NVML
    {
      unsigned int power_mW, clock_MHz, temp_C;
      nvmlUtilization_t util;
      if (nvmlDeviceGetPowerUsage(nvmlDev, &power_mW) == NVML_SUCCESS)
        totalPower += power_mW / 1000.0;                             // mW  -> W
      if (nvmlDeviceGetClockInfo(nvmlDev, NVML_CLOCK_GRAPHICS, &clock_MHz) == NVML_SUCCESS)
        totalClock += clock_MHz;
      if (nvmlDeviceGetTemperature(nvmlDev, NVML_TEMPERATURE_GPU, &temp_C) == NVML_SUCCESS)
        totalTemp += temp_C;
      if (nvmlDeviceGetUtilizationRates(nvmlDev, &util) == NVML_SUCCESS) {
        totalGpuUtil += util.gpu;
        totalMemUtil += util.memory;
      }
    }
#  else         // ROCm SMI
    {
      uint64_t power_uW;
      RSMI_POWER_TYPE power_type;
      rsmi_frequencies_t freqs;
      int64_t  temp_mC;
      uint32_t busy_pct;
      if (rsmi_dev_power_get(rsmi_dev_idx, &power_uW, &power_type) == RSMI_STATUS_SUCCESS)
        totalPower += (double)power_uW / 1e6;                        // μW  -> W
      if (rsmi_dev_gpu_clk_freq_get(rsmi_dev_idx, RSMI_CLK_TYPE_SYS, &freqs) == RSMI_STATUS_SUCCESS)
        totalClock += (double)freqs.frequency[freqs.current] / 1e6;  // Hz  -> MHz
      if (rsmi_dev_temp_metric_get(rsmi_dev_idx, RSMI_TEMP_TYPE_JUNCTION,
                                   RSMI_TEMP_CURRENT, &temp_mC) == RSMI_STATUS_SUCCESS)
        totalTemp += temp_mC / 1000.0;                               // m°C -> °C
      if (rsmi_dev_busy_percent_get(rsmi_dev_idx, &busy_pct) == RSMI_STATUS_SUCCESS)
        totalGpuUtil += busy_pct;
      if (rsmi_dev_memory_busy_percent_get(rsmi_dev_idx, &busy_pct) == RSMI_STATUS_SUCCESS)
        totalMemUtil += busy_pct;
    }
#  endif
    monitor_samples++;
#endif // METRICS

    checkCuda(cudaEventRecord(ev_stop, 0));
    checkCuda(cudaEventSynchronize(ev_stop));

    assert(stat == CUBLAS_STATUS_SUCCESS);
    assert(!cudaGetLastError());

    float elapsed_ms;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    sum += elapsed_ms / 1000.0f;
    actual_repeats++;
  }

  if (actual_repeats == 0) actual_repeats = 1;

  checkCuda(cudaFree(d_A));
  checkCuda(cudaFree(d_B));
  checkCuda(cudaFree(d_C));

  double totalFlops = 2.0 * matrix_dimension * matrix_dimension * matrix_dimension * actual_repeats;

  // ---------------------------------------------------------------------------
  // Output
  // ---------------------------------------------------------------------------
#ifdef METRICS
  double avgPower = (monitor_samples > 0) ? totalPower / monitor_samples : 0.0;
  double avgClock = (monitor_samples > 0) ? totalClock / monitor_samples : 0.0;
  double avgTemp  = (monitor_samples > 0) ? totalTemp  / monitor_samples : 0.0;
  double avgGpu   = (monitor_samples > 0) ? (double)totalGpuUtil / monitor_samples : 0.0;
  double avgMem   = (monitor_samples > 0) ? (double)totalMemUtil / monitor_samples : 0.0;
  // SM = Streaming Multiprocessors (NVIDIA); CU = Compute Units (AMD)
#  ifndef _HIP
  const char *util_label = "SM Util";
#  else
  const char *util_label = "CU Util";
#  endif

  cout << "size "        << matrix_dimension
       << " | Time: "    << sum << " s"
       << " | Repeats: " << actual_repeats
       << " | Perf: "    << totalFlops / sum / 1e12 << " TFlop/s"
       << " | Avg Power: " << avgPower << " W"
       << " | Avg Clock: " << avgClock << " MHz"
       << " | Avg Temp: "  << avgTemp  << " C"
       << " | " << util_label << ": " << avgGpu << " %"
       << " | Mem Util: "  << avgMem   << " %" << endl;
#else
  cout << "size "        << matrix_dimension
       << " | Time: "    << sum << " s"
       << " | Repeats: " << actual_repeats
       << " | Perf: "    << totalFlops / sum / 1e12 << " TFlop/s" << endl;
#endif

  // ---------------------------------------------------------------------------
  // Shutdown telemetry
  // ---------------------------------------------------------------------------
#ifdef METRICS
#  ifndef _HIP
  nvmlShutdown();
#  else
  rsmi_shut_down();
#  endif
#endif

  return 0;
}
