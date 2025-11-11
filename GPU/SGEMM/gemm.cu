#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <fstream> // === Added for file writing ===

#include <nvml.h>

using namespace std;

const char *cublasGetErrorString(cublasStatus_t status)
{
  switch (status)
  {
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
  }
  return "unknown error";
}

// Kernel: Initialize arrays with specific values or random values
__global__ void init_data(float *A, float *B, float *C, int size, unsigned long long seed)
{
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)size * (long)size;

  if (idx >= total)
    return;

  // Declare and initialize RNG state
  curandState state;
  curand_init(seed, idx, 0, &state); // seed, sequence number, offset, &state

  // Generate random numbers between 0 and 1
  A[idx] = (float)curand_uniform(&state);
  B[idx] = (float)curand_uniform(&state);
  C[idx] = (float)curand_uniform(&state);

  // B[idx] = (float)0.0012478f;
  // A[idx] = (float)0.0121290f;
  // C[idx] = (float)0.0f;
}

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "CUBLAS Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}

int main(int argc, char **argv)
{
  size_t matrix_dimension = 90000;
  int repeats = 500;

#ifdef SIZE
  matrix_dimension = SIZE;
#endif

#ifdef NTIMES
  repeats = NTIMES;
#endif

  int device_id = 0;

  if (argc > 1)
  {
    device_id = stoi(argv[1]);
  }

  checkCuda(cudaSetDevice(device_id));

  cout
      << "\ncublasSgemm test result:\n"
      << endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);

  cout << "Summary:" << endl
       << "Device: " << device_id << endl
       << "Device Name : " << prop.name << endl
       << "Matrix_dimension: " << matrix_dimension << endl
       << "Repeats: " << repeats
       << endl;

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));

#ifdef TENSOR

  checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

#endif

#ifdef NVML

  // --- Initialize NVML ---
  nvmlReturn_t nvmlResult;
  nvmlDevice_t device;
  nvmlResult = nvmlInit();
  if (nvmlResult != NVML_SUCCESS)
  {
    cerr << "Failed to initialize NVML: " << nvmlErrorString(nvmlResult) << endl;
    return 1;
  }
  nvmlResult = nvmlDeviceGetHandleByIndex(device_id, &device);
  if (nvmlResult != NVML_SUCCESS)
  {
    cerr << "Failed to get NVML device handle: " << nvmlErrorString(nvmlResult) << endl;
    return 1;
  }

#endif

  size_t bytes = (size_t)matrix_dimension * (size_t)matrix_dimension * sizeof(float);
  cout << "Allocating device variables with total size: " << 3 * bytes / 1e9 << " GB" << endl;

  float *d_A, *d_B, *d_C;
  checkCuda(cudaMalloc(&d_A, bytes));
  checkCuda(cudaMalloc(&d_B, bytes));
  checkCuda(cudaMalloc(&d_C, bytes));

  // Launch configuration
  long threads = 256;
  long blocks = ((long)matrix_dimension * (long)matrix_dimension) / threads;
  unsigned long long seed = time(NULL); // unique seed

  // Launch kernel
  init_data<<<blocks, threads>>>(d_A, d_B, d_C, matrix_dimension, seed);
  cudaDeviceSynchronize();

  const float alf = 1.0579f;
  const float bet = 0.0127f;
  const float *alpha = &alf;
  const float *beta = &bet;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

#ifdef NVML

  double totalPower = 0.0;
  double totalClock = 0.0;
  double totalTemp = 0.0;

  int samples = 0;

#endif

  double sum = 0.0;

  for (int rep = 0; rep < repeats; rep++)
  {
    cudaEventRecord(start, 0);
    int m = matrix_dimension, n = matrix_dimension, k = matrix_dimension;
    int lda = m, ldb = k, ldc = m;

    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    assert(stat == CUBLAS_STATUS_SUCCESS);
    assert(!cudaGetLastError());

#ifdef NVML

    // --- Sample NVML power and frequency ---
    unsigned int power_mW, clock_MHz, temp_C;
    if (nvmlDeviceGetPowerUsage(device, &power_mW) == NVML_SUCCESS)
      totalPower += power_mW / 1000.0;
    if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock_MHz) == NVML_SUCCESS)
      totalClock += clock_MHz;
    if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_C) == NVML_SUCCESS)

      totalTemp += temp_C;
    samples++;

#endif
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    sum += elapsed / 1000.0f; // convert ms to s
  }

#ifdef NVML
  double avgPower = totalPower / samples;
  double avgClock = totalClock / samples;
  double avgTemp = totalTemp / samples;
  cout << "size " << matrix_dimension
       << " | Time: " << sum << " s"
       << " | Perf: " << 2.0 * matrix_dimension * matrix_dimension * matrix_dimension * repeats / sum / 1e12 << " TFlop/s"
       << " | Avg Power: " << avgPower << " W"
       << " | Avg Clock: " << avgClock << " MHz"
       << " | Avg Temp: " << avgTemp << " C" << endl;
#else
  cout << "size " << matrix_dimension
       << " | Time: " << sum << " s"
       << " | Perf: " << 2.0 * matrix_dimension * matrix_dimension * matrix_dimension * repeats / sum / 1e12 << " TFlop/s" << endl;
#endif

  // float *h_C;

  // size_t alignment = 32; // 64-byte alignment is common for AVX512, CUDA, etc.

  // if (posix_memalign((void **)&h_C, alignment, bytes) != 0)
  // {
  //   perror("posix_memalign h_C");
  //   return 1;
  // }

  // checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // ofstream outfile("output_matrix.txt");
  // if (!outfile.is_open())
  // {
  //   cerr << "Error: could not open output_matrix.txt for writing!" << endl;
  // }
  // else
  // {
  //   outfile << "Matrix C (" << matrix_dimension << "x" << matrix_dimension << ")\n";
  //   for (size_t i = 0; i < matrix_dimension ; ++i)
  //   {
  //     for (size_t j = 0; j < matrix_dimension ; ++j)
  //     {
  //       outfile << h_C[i * matrix_dimension + j] << " ";
  //     }
  //     outfile << "\n";
  //   }
  //   outfile.close();
  //   cout << "Matrix C written to output_matrix.txt" << endl;
  // }
  // // === End of added section ===
  // free(h_C);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  nvmlShutdown(); // cleanup NVML
  return 0;
}
