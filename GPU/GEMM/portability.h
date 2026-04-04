#ifndef PORTABILITY_H
#define PORTABILITY_H

// Inspired by TheBandwidthBenchmark/src/portability.h
// Provides a portability layer so gemm.cu / gemm_sweep.cu compile with
// either nvcc (CUDA) or hipcc (HIP/ROCm).
//
// -D_NVCC  set by mk/include_<NVIDIA_GPU>.mk  → CUDA path
// -D_HIP   set by mk/include_<AMD_GPU>.mk     → HIP path
//
// -DMETRICS (single flag, backend-agnostic)
//   CUDA build → NVML  (header included below; links -lnvidia-ml)
//   HIP  build → ROCm SMI (header included below; links -lrocm_smi64)

#define HLINE "------------------------------------------------------------------------\n"
 
#ifdef _HIP

// ---------------------------------------------------------------------------
// HIP backend
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>      // device-side RNG (mirrors curand_kernel.h)
#include <hipblas/hipblas.h>     // hipBLAS (mirrors cublas_v2.h)

// ROCm SMI: AMD equivalent of NVML — included when -DMETRICS is set.
// The Makefile always links -lrocm_smi64 for HIP builds.
#ifdef METRICS
#include <rocm_smi/rocm_smi.h>
#endif

// CUDA runtime -> HIP runtime
#define cudaSuccess                     hipSuccess
#define cudaError_t                     hipError_t
#define cudaGetErrorString              hipGetErrorString
#define cudaSetDevice                   hipSetDevice
#define cudaGetDevice                   hipGetDevice
#define cudaGetDeviceCount              hipGetDeviceCount
#define cudaFree                        hipFree
#define cudaMalloc                      hipMalloc
#define cudaDeviceSynchronize           hipDeviceSynchronize
#define cudaDeviceProp                  hipDeviceProp_t
#define cudaGetDeviceProperties         hipGetDeviceProperties
#define cudaGetLastError                hipGetLastError
#define cudaEvent_t                     hipEvent_t
#define cudaEventCreate                 hipEventCreate
#define cudaEventRecord                 hipEventRecord
#define cudaEventSynchronize            hipEventSynchronize
#define cudaEventElapsedTime            hipEventElapsedTime

// cuRAND -> hipRAND (device-side)
#define curandState                     hiprandState
#define curand_init                     hiprand_init
#define curand_uniform                  hiprand_uniform

// cuBLAS -> hipBLAS
#define cublasHandle_t                  hipblasHandle_t
#define cublasStatus_t                  hipblasStatus_t
#define cublasCreate                    hipblasCreate
#define cublasDestroy                   hipblasDestroy
#define cublasDgemm                     hipblasDgemm
#define cublasSgemm                     hipblasSgemm
#define cublasSetMathMode               hipblasSetMathMode
#define CUBLAS_OP_N                     HIPBLAS_OP_N
#define CUBLAS_STATUS_SUCCESS           HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED   HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED      HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE     HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_EXECUTION_FAILED  HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR    HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_TF32_TENSOR_OP_MATH      HIPBLAS_TF32_TENSOR_OP_MATH
// HIPBLAS_STATUS_ARCH_MISMATCH and HIPBLAS_STATUS_MAPPING_ERROR do not exist
// in all hipBLAS versions (absent in ROCm >= 6) and are intentionally not mapped.

#else

// ---------------------------------------------------------------------------
// CUDA backend
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <nvml.h>   // always included; used when -DMETRICS is set

#endif // _HIP

#endif // PORTABILITY_H
