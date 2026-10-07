#ifndef __UTIL_H
#define __UTIL_H
#include "portability.h" // HIP/CUDA portability + dtype typedefs
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Kernel: Initialize A (nA elems), B (nB), C (nC) with random values
// ---------------------------------------------------------------------------
__global__ void init_data(STORAGE *A, STORAGE *B, ACCUM *C,
                          size_t nA, size_t nB, size_t nC, unsigned long long seed)
{
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (idx >= nA && idx >= nB && idx >= nC) return;

  curandState state;
  curand_init(seed, idx, 0, &state);
  if (idx < nA) A[idx] = (STORAGE)curand_uniform(&state);
  if (idx < nB) B[idx] = (STORAGE)curand_uniform(&state);
  if (idx < nC) C[idx] = (ACCUM)curand_uniform(&state);
}

// ---------------------------------------------------------------------------
// Kernel: Initialize A, B, C with fixed constant values
// ---------------------------------------------------------------------------
__global__ void init_data_constant(STORAGE *A, STORAGE *B, ACCUM *C,
                                   size_t nA, size_t nB, size_t nC)
{
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (idx >= nA && idx >= nB && idx >= nC) return;

  if (idx < nA) A[idx] = (STORAGE)0.1529;
  if (idx < nB) B[idx] = (STORAGE)1.2631;
  if (idx < nC) C[idx] = (ACCUM)0.0;
}

// ---------------------------------------------------------------------------
// Kernel: Element-wise correctness check  |x - y| > atol + rtol*|y| fails.
// Counts failures and records the worst relative error (atomicMax over the
// bit pattern of non-negative floats is valid for IEEE ordering).
// ---------------------------------------------------------------------------
__global__ void compare_kernel(const ACCUM *x, const ACCUM *y, size_t n,
                               double atol, double rtol,
                               size_t *fails, unsigned int *max_rel_bits)
{
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    double a = (double)x[i], b = (double)y[i];
    double err = fabs(a - b);
    if (err > atol + rtol * fabs(b))
      atomicAdd((unsigned long long *)fails, 1ULL);
    double rel = err / (fabs(b) + 1e-30);
    if (rel > 1e10) rel = 1e10;
    unsigned int bits;
    float rf = (float)rel;
    memcpy(&bits, &rf, sizeof(bits));
    atomicMax(max_rel_bits, bits);
  }
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

// ---------------------------------------------------------------------------
// Result rows: one per optimization level (plus the vendor BLAS rung).
// ---------------------------------------------------------------------------
struct LevelRow {
  std::string level;
  double time_s;
  int    repeats;
  double tflops;
  long   M, N, K;              // shape (the sweep table shows it)
#ifdef METRICS
  bool   has_metrics;
  double power, clock, temp, gpu_util, mem_util;
#endif
};

static inline LevelRow make_row(const std::string &level, size_t M, size_t N, size_t K,
                                double sum, int repeats)
{
  LevelRow r;
  r.level = level;
  r.M = (long)M; r.N = (long)N; r.K = (long)K;
  r.time_s = sum;
  r.repeats = repeats;
  double total_flops = 2.0 * (double)M * (double)N * (double)K * (double)repeats;
  r.tflops = (sum > 0.0) ? (total_flops / sum / 1e12) : 0.0;
#ifdef METRICS
  r.has_metrics = false;
  r.power = r.clock = r.temp = r.gpu_util = r.mem_util = 0.0;
#endif
  return r;
}

// Formatting helpers for the table.
static inline std::string dstr(double v, int prec) {
  std::ostringstream os; os << std::fixed << std::setprecision(prec) << v; return os.str();
}
static inline std::string lpad(const std::string &s, int w) {
  if ((int)s.size() >= w) return s.substr(0, w);
  return s + std::string(w - (int)s.size(), ' ');
}
static inline std::string cpad(const std::string &s, int w) {
  if ((int)s.size() >= w) return s;
  int left = (w - (int)s.size()) / 2;
  return std::string(left, ' ') + s + std::string(w - (int)s.size() - left, ' ');
}

// ---------------------------------------------------------------------------
// Table printing.  show_shape enables the M/N/K column (sweep mode).
// The %BLAS column appears automatically when a vendor-BLAS row exists
// (its TFlop/s is the reference).  %Peak is compiled in when the build
// defines PEAK_TFLOPS (set per GPU + datatype in the mk/ files).
// ---------------------------------------------------------------------------
static inline void print_table(const std::vector<LevelRow> &rows, bool show_shape)
{
  using namespace std;
  cout << left << fixed;

  double blas_tflops = 0.0;
  for (const LevelRow &r : rows)
    if (r.level.find("BLAS") != string::npos) blas_tflops = r.tflops;

#ifdef METRICS
  bool metrics = false;
  for (const LevelRow &r : rows) if (r.has_metrics) { metrics = true; break; }
#else
  const bool metrics = false;
#endif

  const int W_LVL = 40;
  string hl = "+" + string(W_LVL + 2, '-') + "+";
  if (show_shape)        hl += string(22, '-') + "+";
  hl += string(11, '-') + "+" + string(9, '-') + "+" + string(13, '-') + "+";
  if (blas_tflops > 0.0) hl += string(9, '-') + "+";
#ifdef PEAK_TFLOPS
  hl += string(8, '-') + "+";
#endif
  if (metrics)           hl += string(48, '-') + "+";

  cout << hl << "\n| " << lpad("Level", W_LVL) << " |";
  if (show_shape) cout << " " << cpad("M x N x K", 20) << " |";
  cout << " " << cpad("Time (s)", 9) << " |"
       << " " << cpad("Reps", 7) << " |"
       << " " << cpad("TFlop/s", 11) << " |";
  if (blas_tflops > 0.0) cout << " " << cpad("% BLAS", 7) << " |";
#ifdef PEAK_TFLOPS
  cout << " " << cpad("% Peak", 6) << " |";
#endif
  if (metrics) cout << " " << cpad("Power W", 8) << " |"
                      " " << cpad("Clock MHz", 9) << " |"
                      " " << cpad("Temp C", 7) << " |"
                      " " << cpad(GpuMonitor::util_label(), 9) << " |"
                      " " << cpad("Mem %", 7) << " |";
  cout << "\n" << hl << "\n";

  for (const LevelRow &r : rows) {
    ostringstream shape;
    shape << r.M << "x" << r.N << "x" << r.K;
    cout << "| " << lpad(r.level, W_LVL) << " |";
    if (show_shape) cout << " " << lpad(shape.str(), 20) << " |";
    cout << " " << lpad(dstr(r.time_s, 4), 9) << " |"
         << " " << lpad(to_string(r.repeats), 7) << " |"
         << " " << lpad(dstr(r.tflops, 3), 11) << " |";
    if (blas_tflops > 0.0)
      cout << " " << lpad(dstr(100.0 * r.tflops / blas_tflops, 1), 7) << " |";
#ifdef PEAK_TFLOPS
    cout << " " << lpad(dstr(100.0 * r.tflops / (double)(PEAK_TFLOPS), 1), 6) << " |";
#endif
#ifdef METRICS
    if (metrics && r.has_metrics)
      cout << " " << lpad(dstr(r.power, 1), 8) << " |"
           << " " << lpad(dstr(r.clock, 1), 9) << " |"
           << " " << lpad(dstr(r.temp, 1), 7) << " |"
           << " " << lpad(dstr(r.gpu_util, 1), 9) << " |"
           << " " << lpad(dstr(r.mem_util, 1), 7) << " |";
    else if (metrics)
      cout << " " << lpad("-", 8) << " |" << " " << lpad("-", 9) << " |"
           << " " << lpad("-", 7) << " |" << " " << lpad("-", 9) << " |"
           << " " << lpad("-", 7) << " |";
#endif
    cout << "\n";
  }
  cout << hl << "\n";
}

// CSV writer (used with -c <file>); one row per level.
static inline void write_csv(const std::string &path, const std::vector<LevelRow> &rows)
{
  std::ofstream f(path.c_str(), std::ios::app);
  if (!f.good()) { std::cerr << "Cannot open CSV file: " << path << "\n"; return; }
  if (f.tellp() == std::streampos(0)) {
    f << "level,M,N,K,time_s,repeats,tflops";
#ifdef METRICS
    f << ",power_W,clock_MHz,temp_C,gpu_util,mem_util";
#endif
    f << "\n";
  }
  for (const LevelRow &r : rows) {
    f << r.level << ',' << r.M << ',' << r.N << ',' << r.K << ','
      << r.time_s << ',' << r.repeats << ',' << r.tflops;
#ifdef METRICS
    f << ',' << r.power << ',' << r.clock << ',' << r.temp
      << ',' << r.gpu_util << ',' << r.mem_util;
#endif
    f << "\n";
  }
  f.close();
}

#endif
