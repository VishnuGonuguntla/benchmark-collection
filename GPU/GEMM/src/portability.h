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
//
// -DDTYPE_FP64 / -DDTYPE_FP32 (default) / -DDTYPE_FP16 select the problem
//   storage type (STORAGE) and compute type (ACCUM):
//   FP64: double  × double → double
//   FP32: float   × float  → float
//   FP16: __half  × __half → float   (C is stored in float)

#define HLINE "------------------------------------------------------------------------\n"

#ifdef _HIP

// ---------------------------------------------------------------------------
// HIP backend
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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
#define cudaMemcpy                      hipMemcpy
#define cudaMemcpyDeviceToDevice        hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost          hipMemcpyDeviceToHost
#define cudaMemset                      hipMemset
#define cudaDeviceSynchronize           hipDeviceSynchronize
#define cudaDeviceProp                  hipDeviceProp_t
#define cudaGetDeviceProperties         hipGetDeviceProperties
#define cudaGetLastError                hipGetLastError
#define cudaFuncSetAttribute            hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaStream_t                    hipStream_t
#define cudaEvent_t                     hipEvent_t
#define cudaEventCreate                 hipEventCreate
#define cudaEventDestroy                hipEventDestroy
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
#define cublasGemmEx                    hipblasGemmEx
#define cublasGemmExCompute_t           hipblasComputeType_t
#define CUBLAS_OP_N                     HIPBLAS_OP_N
#define CUBLAS_R_64F                    HIPBLAS_R_64F
#define CUBLAS_R_32F                    HIPBLAS_R_32F
#define CUBLAS_R_16F                    HIPBLAS_R_16F
#define CUBLAS_COMPUTE_64F              HIPBLAS_COMPUTE_64F
#define CUBLAS_COMPUTE_32F              HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT             HIPBLAS_GEMM_DEFAULT
#define CUBLAS_STATUS_SUCCESS           HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED   HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED      HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE     HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_EXECUTION_FAILED  HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR    HIPBLAS_STATUS_INTERNAL_ERROR
// TF32 compute mode only exists on NVIDIA hardware; on AMD it falls back to
// plain FP32 so shared call sites compile (the ladder never requests it on
// HIP — see ENABLE_V5_TF32 in this header).
#define CUBLAS_COMPUTE_32F_FAST_TF32    HIPBLAS_COMPUTE_32F

// hipBLAS's datatype enum type is hipblasDatatype_t across ROCm versions.
typedef hipDataType blasDataType_t;

// HIPBLAS_STATUS_ARCH_MISMATCH and HIPBLAS_STATUS_MAPPING_ERROR do not exist
// in all hipBLAS versions (absent in ROCm >= 6) and are intentionally not mapped.

#else

// ---------------------------------------------------------------------------
// CUDA backend
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <nvml.h>   // always included; used when -DMETRICS is set

typedef cudaDataType_t blasDataType_t;

// cublasGemmEx spelling shared with the HIP path (runner.h uses one call
// site for both backends).
#define cublasGemmExCompute_t           cublasComputeType_t
#define CUBLAS_R_64F                    CUDA_R_64F
#define CUBLAS_R_32F                    CUDA_R_32F
#define CUBLAS_R_16F                    CUDA_R_16F

#endif // _HIP

// ---------------------------------------------------------------------------
// Problem data types
// ---------------------------------------------------------------------------
#if defined(DTYPE_FP64)
#  define GEMM_DT_NAME "FP64"
typedef double STORAGE;   // storage type of A and B
typedef double ACCUM;     // accumulator / C type
#elif defined(DTYPE_FP16)
#  define GEMM_DT_NAME "FP16"
typedef __half STORAGE;
typedef float  ACCUM;
#else /* FP32 default */
#  ifndef DTYPE_FP32
#    define DTYPE_FP32
#  endif
#  define GEMM_DT_NAME "FP32"
typedef float STORAGE;
typedef float ACCUM;
#endif

// ---------------------------------------------------------------------------
// Tensor-core (V5) availability
//   - TENSOR=off  → never
//   - FP16        → NVIDIA (WMMA) + AMD (rocWMMA, when headers installed)
//   - FP32        → NVIDIA TF32 WMMA only (AMD has no tensor path for FP32)
//   - FP64        → never
// ---------------------------------------------------------------------------
#if !defined(TENSOR_OFF) && !defined(DTYPE_FP64)
#  if defined(DTYPE_FP16) && (!defined(_HIP) || defined(HAVE_ROCWMA))
#    define ENABLE_V5 1
#  elif defined(DTYPE_FP32) && !defined(_HIP)
#    define ENABLE_V5 1
#    define ENABLE_V5_TF32 1
#  endif
#endif

#if defined(TENSOR_ON) && !defined(ENABLE_V5)
#  error "TENSOR=on requested but no tensor-core path exists for this backend/datatype (FP64 never; FP32 on AMD never; FP16 on AMD requires rocWMMA headers, see config.mk)."
#endif

#endif // PORTABILITY_H
