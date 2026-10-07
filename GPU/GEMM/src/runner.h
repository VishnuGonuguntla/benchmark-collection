#ifndef RUNNER_H
#define RUNNER_H

// Shared benchmark harness for gemm.cu and gemm_sweep.cu:
// device resolution, vendor BLAS rung, per-level timing loops,
// runtime tile selection (-T) with optional autotuning, and the
// correctness pass (-v).  Header-only static functions, like cli.h.

#include "portability.h"
#include "util.cuh"
#include "kernels.cuh"
#include <chrono>
#include <string>
#include <algorithm>

// ---------------------------------------------------------------------------
// On AMD, the -d flag addresses the GPU in BDF order (as shown by amd-smi),
// which differs from the HIP device index.  Resolve it.
// ---------------------------------------------------------------------------
static inline int resolve_device(int gpu_id)
{
  int device_id = gpu_id;
#ifdef _HIP
  int num_devices = 0;
  checkCuda(cudaGetDeviceCount(&num_devices));
  if (gpu_id < 0 || gpu_id >= num_devices) {
    std::cerr << "GPU ID " << gpu_id << " out of range (0.." << num_devices - 1 << ")\n";
    exit(1);
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
#else
  int num_devices = 0;
  checkCuda(cudaGetDeviceCount(&num_devices));
  if (gpu_id < 0 || gpu_id >= num_devices) {
    std::cerr << "GPU ID " << gpu_id << " out of range (0.." << num_devices - 1 << ")\n";
    exit(1);
  }
#endif
  return device_id;
}

// Round a dimension up to the 256-element alignment used throughout
// (guarantees every tile config and vector load stays valid).
static inline void align_one(size_t &v)
{
  const size_t ALIGN = 256;
  v = (v + ALIGN - 1) / ALIGN * ALIGN;
}

static inline void align_dims(size_t &M, size_t &N, size_t &K)
{
  align_one(M); align_one(N); align_one(K);
}

// ---------------------------------------------------------------------------
// Vendor BLAS rung.  Row-major C = A*B is executed as the column-major
// identity C^T = B^T * A^T; the storage layouts of the row-major arrays are
// exactly the required column-major views, so no transpose ops are needed.
// ---------------------------------------------------------------------------
static inline blasDataType_t blas_storage_type()
{
#if defined(DTYPE_FP64)
  return CUBLAS_R_64F;
#elif defined(DTYPE_FP16)
  return CUBLAS_R_16F;
#else
  return CUBLAS_R_32F;
#endif
}
static inline blasDataType_t blas_c_type()
{
#if defined(DTYPE_FP64)
  return CUBLAS_R_64F;
#else
  return CUBLAS_R_32F;   // C is float for FP32 and FP16 builds, double for FP64
#endif
}

// allow_tf32 selects FP32 fast-TF32 tensor math for the library rung
// (no-op on AMD / other datatypes, where it silently stays exact).
static inline void blas_run(cublasHandle_t h, const LaunchArgs &a, bool allow_tf32)
{
  cublasGemmExCompute_t comp;
#if defined(DTYPE_FP64)
  comp = CUBLAS_COMPUTE_64F;
#else
  comp = (allow_tf32) ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
#endif
  checkCublas(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                           a.N, a.M, a.K,
                           &a.alpha, a.B, blas_storage_type(), a.N,
                                       a.A, blas_storage_type(), a.K,
                           &a.beta,  a.C, blas_c_type(), a.N,
                           comp, CUBLAS_GEMM_DEFAULT));
}

static inline bool tf32_runtime_supported(const cudaDeviceProp &prop)
{
#if defined(ENABLE_V5_TF32)
  return prop.major >= 8;   // wmma tf32 + cuBLAS fast-TF32
#else
  (void)prop; return false;
#endif
}

static inline bool v5_runtime_supported(const cudaDeviceProp &prop)
{
#ifdef _HIP
  (void)prop; return true;          // wave ISA fixed at compile time (gfx908+)
#else
#  ifdef ENABLE_V5_TF32
  return prop.major >= 8;           // TF32 fragments need Ampere+
#  else
  return prop.major >= 7;           // fp16 wmma needs Volta+
#  endif
#endif
}

// ---------------------------------------------------------------------------
// One ladder pass: run `once()` under CUDA/HIP events, honoring either the
// fixed-repeat (-r) or wall-clock (-t) mode, and produce the result row.
// ---------------------------------------------------------------------------
template <typename OnceFn>
static inline LevelRow timed_loop(const std::string &label,
                                  size_t M, size_t N, size_t K,
                                  int repeats, double target_minutes,
                                  OnceFn once)
{
  using namespace std;
  cudaEvent_t ev_start, ev_stop;
  checkCuda(cudaEventCreate(&ev_start));
  checkCuda(cudaEventCreate(&ev_stop));

  once();                                   // warmup (algo selection, page-in)
  checkCuda(cudaDeviceSynchronize());

  double sum = 0.0;
  int actual_repeats = 0;
  auto wall_start = chrono::high_resolution_clock::now();

  while (true) {
    if (target_minutes > 0.0) {
      auto now = chrono::high_resolution_clock::now();
      if (chrono::duration<double>(now - wall_start).count() > target_minutes * 60.0)
        break;
    } else {
      if (actual_repeats >= repeats) break;
    }

    checkCuda(cudaEventRecord(ev_start, 0));
    once();
    checkCuda(cudaEventRecord(ev_stop, 0));
    checkCuda(cudaEventSynchronize(ev_stop));
    assert(!cudaGetLastError());

    float elapsed_ms;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    sum += elapsed_ms / 1000.0f;
    actual_repeats++;
  }

  checkCuda(cudaEventDestroy(ev_start));
  checkCuda(cudaEventDestroy(ev_stop));
  if (actual_repeats == 0) actual_repeats = 1;
  return make_row(label, M, N, K, sum, actual_repeats);
}

// ---------------------------------------------------------------------------
// Run one level exactly once (kernel dispatch or BLAS call).
// Returns false when the launch could not be made (e.g. tile config needs
// more shared memory than the device offers).
// ---------------------------------------------------------------------------
static inline bool level_run_once(int id, int tile_idx, cublasHandle_t h,
                                  const LaunchArgs &a, bool blas_tf32)
{
  if (id == LEVEL_BLAS) { blas_run(h, a, blas_tf32); return true; }
  LevelInfo li = get_level(id);
  if (tile_idx >= li.nopts) tile_idx = 0;
  return li.opts[tile_idx].fn(a);
}

// Choose the tile option index for a level given the user's -T request.
static inline int pick_tile(const LevelInfo &li, int tile_cfg)
{
  if (li.nopts <= 1) return 0;
  if (tile_cfg == TILE_DEFAULT) return 0;
  if (tile_cfg < 0 || tile_cfg >= li.nopts) {
    std::cerr << "  note: -T " << tile_cfg << " not compiled for '" << li.name
              << "' (valid 0.." << li.nopts - 1 << "); using default\n";
    return 0;
  }
  return tile_cfg;
}

// ---------------------------------------------------------------------------
// -T tune: quick-time every compiled tile config of a level (fixed small
// repeat count) and return the index of the fastest.
// ---------------------------------------------------------------------------
static inline int tune_tile(LevelInfo &li, cublasHandle_t h, LaunchArgs &a,
                            const cudaDeviceProp &prop)
{
  const int TUNE_REPS = 3;
  int best = 0; double best_tflops = -1.0;
  for (int t = 0; t < li.nopts; ++t) {
    LaunchArgs probe = a;
    if (!level_run_once(li.id, t, h, probe, false)) {
      std::cout << "  tune " << li.name << " [" << li.opts[t].label
                << "]: skipped (not runnable on this device)\n";
      continue;
    }
    checkCuda(cudaDeviceSynchronize());
    LevelRow r = timed_loop(std::string(), a.M, a.N, a.K, TUNE_REPS, 0.0,
        [&]() { li.opts[t].fn(a); });
    std::cout << "  tune " << li.name << " [" << li.opts[t].label
              << "]: " << r.tflops << " TFlop/s\n";
    if (r.tflops > best_tflops) { best_tflops = r.tflops; best = t; }
  }
  (void)prop;
  std::cout << "  -> " << li.name << ": using [" << li.opts[best].label << "]\n";
  return best;
}

// ---------------------------------------------------------------------------
// Build the list of levels to run: user -l mask ∩ compiled levels.
// V5 is additionally gated at runtime (needs a tensor-capable device).
// ---------------------------------------------------------------------------
static inline std::vector<int> selected_levels(unsigned user_mask,
                                               const cudaDeviceProp &prop)
{
  std::vector<int> out;
  for (int id = 0; id < LEVEL_BLAS; ++id) {   // BLAS handled separately
    if (user_mask && !((user_mask >> id) & 1)) continue;
    if (!level_compiled(id)) continue;
    if (id == LEVEL_V5 && !v5_runtime_supported(prop)) continue; // sm70/80+, gfx908+
    out.push_back(id);
  }
  return out;
}

static inline std::string level_label(int id, int tile_idx, bool blas_tf32_row)
{
  if (id == LEVEL_BLAS) {
#ifndef _HIP
    return blas_tf32_row ? "cuBLAS (TF32)" : "cuBLAS (library)";
#else
    return "hipBLAS (library)";
#endif
  }
  LevelInfo li = get_level(id);
  std::string s = li.name;
  if (li.nopts > 1 && tile_idx >= 0)
    s += std::string(" [") + li.opts[tile_idx].label + "]";
  return s;
}

// ---------------------------------------------------------------------------
// The full ladder for one problem instance (assumes data already allocated
// and initialized; C0 holds the pristine copy of C).
// ---------------------------------------------------------------------------
struct LadderCtx {
  cublasHandle_t   handle;
  size_t M, N, K;
  ACCUM alpha, beta;
  STORAGE *dA, *dB;
  ACCUM *dC, *dC0;
  ACCUM *dCref;              // verification reference (nullptr unless -v)
  int    repeats;
  double minutes;
  unsigned level_mask;
  int    tile_cfg;
  bool   verify;
  const cudaDeviceProp *prop;
#ifdef METRICS
  GpuMonitor *monitor;
#endif
};

static inline void attach_metrics(LevelRow &r, LadderCtx &cx)
{
#ifdef METRICS
  if (cx.monitor) {
    cx.monitor->stop();
    MetricsAvg avg = cx.monitor->averages();
    r.has_metrics = true;
    r.power = avg.power; r.clock = avg.clock; r.temp = avg.temp;
    r.gpu_util = avg.gpu_util; r.mem_util = avg.mem_util;
  }
#endif
}

static inline std::vector<LevelRow> run_ladder(LadderCtx &cx)
{
  using namespace std;
  std::vector<LevelRow> rows;

  LaunchArgs la;
  la.M = (int)cx.M; la.N = (int)cx.N; la.K = (int)cx.K;
  la.alpha = cx.alpha; la.beta = cx.beta;
  la.A = cx.dA; la.B = cx.dB; la.C = cx.dC;

  std::vector<int> levels = selected_levels(cx.level_mask, *cx.prop);

  // cuBLAS may appear twice (FP32 builds with tensor support): exact FP32
  // and fast-TF32.  AMD / FP64 / FP16 build a single vendor row.
  std::vector<int> blas_rows;   // 0 = exact compute, 1 = TF32 twin (NVIDIA FP32 only)
  if ((cx.level_mask == 0 || (cx.level_mask >> LEVEL_BLAS) & 1) &&
      (LEVEL_MASK >> LEVEL_BLAS) & 1) {
    bool tf32_ok = tf32_runtime_supported(*cx.prop);
    blas_rows.push_back(0);                 // exact FP32/FP64
    if (tf32_ok) blas_rows.push_back(1);    // tensor-core cuBLAS row
  }

  // -------- correctness pass against the exact vendor result -------------
  if (cx.verify && cx.dCref) {
    std::cout << "Correctness check (vs vendor BLAS, exact compute):" << std::endl;
    checkCuda(cudaMemcpy(cx.dCref, cx.dC0, cx.M * cx.N * sizeof(ACCUM),
                         cudaMemcpyDeviceToDevice));
    LaunchArgs ref = la; ref.C = cx.dCref;
    blas_run(cx.handle, ref, false);
    checkCuda(cudaDeviceSynchronize());

    size_t *d_fails = nullptr; unsigned *d_maxrel = nullptr;
    checkCuda(cudaMalloc((void**)&d_fails, sizeof(size_t)));
    checkCuda(cudaMalloc((void**)&d_maxrel, sizeof(unsigned)));
    for (int id : levels) {
      LaunchArgs v = la; v.C = cx.dC;
      checkCuda(cudaMemcpy(cx.dC, cx.dC0, cx.M * cx.N * sizeof(ACCUM),
                           cudaMemcpyDeviceToDevice));
      bool ok = level_run_once(id, pick_tile(get_level(id), cx.tile_cfg),
                               cx.handle, v, false);
      checkCuda(cudaDeviceSynchronize());
      if (!ok) { std::cout << "  " << level_label(id, 0, false)
                           << ": SKIPPED (unlaunchable tile)\n"; continue; }
      double atol, rtol; level_tolerance(id, atol, rtol);
      size_t n = cx.M * cx.N;
      checkCuda(cudaMemset(d_fails, 0, sizeof(size_t)));
      checkCuda(cudaMemset(d_maxrel, 0, sizeof(unsigned)));
      int thr = 256, blocks = (int)std::min<size_t>((n + thr - 1) / thr, 8192);
      compare_kernel<<<blocks, thr>>>(cx.dC, cx.dCref, n, atol, rtol, d_fails, d_maxrel);
      checkCuda(cudaDeviceSynchronize());
      size_t fails; unsigned bits; float maxrel;
      checkCuda(cudaMemcpy(&fails, d_fails, sizeof(size_t), cudaMemcpyDeviceToHost));
      checkCuda(cudaMemcpy(&bits, d_maxrel, sizeof(unsigned), cudaMemcpyDeviceToHost));
      memcpy(&maxrel, &bits, sizeof(float));
      std::cout << "  " << level_label(id, 0, false) << ": "
                << (fails == 0 ? "PASS" : "FAIL")
                << "  (" << fails << " mismatching elems, max rel err "
                << maxrel << ")" << std::endl;
    }
    cudaFree(d_fails); cudaFree(d_maxrel);
    std::cout << HLINE;
  }

  // -------- timing pass ---------------------------------------------------
  for (int id : levels) {
    LevelInfo li = get_level(id);
    int tile_idx = (cx.tile_cfg == TILE_TUNE && li.nopts > 1)
                 ? tune_tile(li, cx.handle, la, *cx.prop)
                 : pick_tile(li, cx.tile_cfg);
    std::string label = level_label(id, tile_idx, false);

    // Probe once: skips levels whose tile cannot run on this device
    // (e.g. dynamic shared memory beyond the GPU's per-block limit).
    if (!level_run_once(id, tile_idx, cx.handle, la, false)) {
      std::cout << "  skipping " << label << ": tile not runnable on this device\n";
      continue;
    }

#ifdef METRICS
    if (cx.monitor) cx.monitor->start();
#endif
    LevelRow r = timed_loop(label, cx.M, cx.N, cx.K, cx.repeats, cx.minutes,
        [&]() { li.opts[tile_idx < li.nopts ? tile_idx : 0].fn(la); });
    attach_metrics(r, cx);
    rows.push_back(r);
  }
  for (int flag : blas_rows) {
    std::string label = level_label(LEVEL_BLAS, -1, flag == 1);
#ifdef METRICS
    if (cx.monitor) cx.monitor->start();
#endif
    LevelRow r = timed_loop(label, cx.M, cx.N, cx.K, cx.repeats, cx.minutes,
        [&]() { blas_run(cx.handle, la, flag == 1); });
    attach_metrics(r, cx);
    rows.push_back(r);
  }

  return rows;
}

static inline void print_tile_help()
{
  std::cout << "Compiled optimization levels and tile configurations (-T index):\n";
  for (int id = 0; id <= LEVEL_BLAS; ++id) {
    if (id == LEVEL_BLAS) {
      std::cout << "  6  " << level_base_name(id) << " (no tile options)\n";
      continue;
    }
    LevelInfo li = get_level(id);
    std::cout << "  " << id << "  " << li.name
              << (li.compiled ? "" : "  [not compiled for this build]") << "\n";
    for (int t = 0; t < li.nopts; ++t)
      std::cout << "      -T " << t << "  " << li.opts[t].label << "\n";
  }
}

#endif // RUNNER_H
