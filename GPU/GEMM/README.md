# GEMM optimization-ladder benchmark (cuBLAS / hipBLAS + hand-written kernels)

A repeatable GPU benchmark that walks a **six-rung optimization ladder** for
`C[M,N] = alpha · A[M,K] · B[K,N] + beta · C[M,N]` — from a naive kernel with
no shared memory, up to tensor-core WMMA kernels — and compares every rung
against the **vendor library** (cuBLAS on NVIDIA, hipBLAS on AMD) in one run.
The same portable source compiles with `nvcc` (CUDA) and `hipcc` (HIP/ROCm),
and the problem is fully **rectangular**: M (rows of A), N (cols of B) and K
(the hidden size) are independent.

Supports FP64, FP32 and FP16 datatypes and an on/off switch for the
tensor-core rung, all selected at build time.

## Layout

| File | Description |
|------|-------------|
| `src/gemm.cu` | Runs the ladder for one fixed problem shape |
| `src/gemm_sweep.cu` | Runs the ladder across a sweep of proportionally grown shapes |
| `src/kernels.cuh` | The optimization ladder V0–V5 + the level/tile registry |
| `src/runner.h` | Shared harness: vendor BLAS rung, timing loops, `-T` tuning, `-v` verification |
| `src/util.cuh` | Init/compare kernels, error checks, result table + CSV output |
| `src/portability.h` | HIP/CUDA portability layer (runtime, cuBLAS→hipBLAS, dtypes) |
| `src/cli.h` | CLI argument parser |
| `src/metrics.h` | GPU telemetry monitoring (NVML / ROCm SMI) |
| `config.mk` | User-facing build configuration |
| `Makefile` | Build rules |
| `mk/` | Per-GPU architecture flags, default problem sizes, optional peaks |

## Prerequisites

**CUDA (NVIDIA):**
- NVHPC SDK or CUDA Toolkit (nvcc, cuBLAS, NVML)
- V5 (tensor cores): compute capability 8.0+ for TF32, 7.0+ for FP16 WMMA

**HIP / ROCm (AMD):**
- ROCm toolkit (hipcc, hipBLAS)
- hipRAND (device-side header `hiprand_kernel.h`)
- ROCm SMI library (`rocm_smi_lib`)
- V5 in FP16 additionally needs the rocWMMA headers (`/opt/rocm*/include/rocwmma/`);
  the Makefile auto-detects them and silently drops level 5 when absent

## Build configuration (`config.mk`)

```makefile
GPU      = H100     # A40 | A100 | H100 | MI300X — also picks CUDA vs HIP
DATATYPE = FP32     # FP64 | FP32 | FP16   (legacy PRECISION=DP/SP also works)
TENSOR   = auto     # auto | on | off — the V5 tensor-core rung
LEVELS   = all      # e.g. "0,1,2,blas" — compile-time subset of the ladder
```

```bash
make                              # GPU/DATATYPE from config.mk
make GPU=MI300X DATATYPE=FP16     # AMD, tensor-core ladder included
make GPU=H100 TENSOR=off LEVELS="2,3,4,blas"
make clean
```

The V5 rung follows simple rules: FP16 builds enable it on **both** vendors,
FP32 builds enable it on **NVIDIA only** (TF32 — AMD GPUs have no FP32 tensor
path, the level is reported as skipped), FP64 builds never have it (there are
no FP64 tensor cores; the ladder ends at V4).

## The optimization ladder

| Level | Source | What it adds | Notes |
|-------|--------|--------------|-------|
| **V0** naive | 1 thread = 1 output, global loads only | — (baseline) | any shape |
| **V1** shared tiling | 8/16/32-tile shared-memory version of V0 | shared memory reuse | any shape (boundary-masked) |
| **V2** register tiling | BM×BN block tile + TM×TN register micro-tile | every shared read feeds TM·TN FMAs | shape must divide the tile |
| **V3** + prefetch | padded shared tiles; next K-tile staged in registers during compute | bank-conflict avoidance + latency hiding | shape must divide the tile |
| **V4** tuned SIMT | double-buffered shared memory, 4-wide vectorized loads, transposed B tile, 128×128 tiles | practical CUDA-core ceiling | shape must divide the tile |
| **V5** tensor cores | WMMA / rocWMMA: fp16→fp32 (both) or TF32 (NVIDIA) | matrix units | 16×16 fragments |
| **BLAS** | `cublasGemmEx` / `hipblasGemmEx` | vendor reference | on NVIDIA FP32, both an exact-FP32 and a fast-TF32 row are measured |

Levels V2–V4 are instantiated for a curated set of tile configurations
(`-T help` lists them) and the configuration is selected **at runtime** —
either explicitly (`-T <index>`) or by autotuning (`-T tune`, times every
compiled config per level and keeps the fastest). Dynamic shared memory with
opt-in beyond 48 KB makes large tile configs possible; configs exceeding the
device's shared-memory limit are skipped with a notice.

## Runtime flags

| Flag | Description | Default |
|------|-------------|---------|
| `-h` | Help | — |
| `-d <int>` | GPU device ID (AMD: BDF order as in `amd-smi`) | `0` |
| `-m <int>` | **M** — rows of A / C | `SIZE` (from `mk/`) |
| `-n <int>` | **N** — cols of B / C | `SIZE` |
| `-k <int>` | **K** — hidden size (cols A = rows B) | `SIZE` |
| `-s <int>` | Cube shorthand `M=N=K` (`gemm`); size cap (`gemm_sweep`) | sweep: `SWEEPSIZE` |
| `-r <int>` | Iterations per level | `NTIMES` / `SWEEPNTIMES` |
| `-t <double>` | Minutes per level, `0` = use `-r` | `0` |
| `-i <mode>` | `random` or `constant` data init | `random` |
| `-l <list>` | Runtime subset of compiled levels: `0..5`, `blas`, `all` | all compiled |
| `-T <int\|tune\|help>` | Tile configuration per level, autotune, or list options | per-level default |
| `-v` | Verify every level against the exact vendor result (PASS/FAIL) | off |
| `-c <file>` | Append one CSV row per level (for `postprocess.ipynb`) | — |

All dimensions are rounded up to a multiple of 256 (printed when adjusted);
this keeps every tile configuration and vector load valid. `2·M·N·K` flops
are used for every reported TFlop/s.

### Examples

```bash
# Fixed-iterations ladder at one LLM-MLP-like shape, with verification
./gemm -m 4096 -n 4096 -k 16384 -r 10 -v

# Time-based: half a minute per rung, autotuned tiles
./gemm -m 8192 -n 8192 -k 8192 -t 0.5 -T tune

# Hand-written levels only, compare vs cuBLAS TF32
./gemm -s 4096 -l 2,3,4,5,blas -t 1.0

# List the tile configurations compiled into this binary
./gemm -T help

# Proportional rectangular sweep (base shape grown ×1.2 per step,
# stopping when the smallest dim passes the cap)
./gemm_sweep -m 2560 -n 2560 -k 5120 -s 30000 -r 5 -c sweep.csv

# Legacy invocations still work (NAIVE == V1, VENDOR == BLAS only)
make GPU=H100 PRECISION=SP OPTIMIZATION=VENDOR && ./gemm -s 16384 -r 10
```

## Output

```
+----------------------------------+-----------+---------+-------------+----------+
| Level                            |  Time (s) |  Reps   |  TFlop/s    | % BLAS   |
+----------------------------------+-----------+---------+-------------+----------+
| V0 naive (global only)           | 312.4021  |   10    |  1.321      |  4.1 %   |
| V1 shared tiling [shared 16x16]  | 41.9003   |   10    |  9.842      | 30.6 %   |
| V2 register tiling [64x64x16/4x4]| 8.7310    |   10    |  47.230     | 146.9%   |
| ...                              |           |         |             |          |
| cuBLAS (library)                 | 0.9102    |   10    |  453.2      | 100.0 %  |
| cuBLAS (TF32)                    | 0.3041    |   10    |  1357.1     | ...      |
+----------------------------------+-----------+---------+-------------+----------+
```

One row per level (plus the vendor row(s)); the `% BLAS` column references
the last vendor-BLAS row.  `gemm_sweep` adds an `M x N x K` column.  With
`-DMETRICS` (default in the `mk/` files) each row also carries per-level
power, clock, temperature, GPU and memory utilization, so telemetry is
compared rung-by-rung, not just performance.  A `% Peak` column appears when
`PEAK_FP64/FP32/FP16` are defined in the active `mk/include_<GPU>.mk`.

The `-c` CSV rows (`level,M,N,K,time_s,repeats,tflops,...`) are directly
plottable from `postprocess.ipynb`.

## Compile-time options

| Option | Effect |
|--------|--------|
| `-DMETRICS` | GPU telemetry per level (NVML on CUDA, ROCm SMI on HIP) |
| `-DLEVEL_MASK=<bits>` | Compile only selected ladder rungs (set by `LEVELS=` in `config.mk`) |
| `-DTENSOR_ON/-DTENSOR_OFF` | Force tensor rung on/off (set by `TENSOR=`) |
| `-DDTYPE_FP64/-DDTYPE_FP32/-DDTYPE_FP16` | Datatype (set by `DATATYPE=`) |
| `-DPEAK_TFLOPS=<val>` | Enable the `% Peak` column (set from `PEAK_*` in `mk/`) |
| `-DTARGET_MINUTES=<val>` | Compile-time default for `-t` |

## Portability & accuracy notes

- Levels V0–V4 are single-source CUDA written to compile unmodified under
  `hipcc`; global addressing is 64-bit throughout (square runs at MI300X
  default sizes overflow 32-bit indexing).
- The vendor rung always computes row-major `A·B` by the column-major
  identity `Cᵀ = Bᵀ·Aᵀ`, with correct `lda/ldb/ldc` for rectangular shapes.
- **FP16 builds** store A/B as `__half`, accumulate in `float`, and store C
  as `float` — matching the WMMA/MFMA rung and the `GemmEx` library call, so
  the ladder difference is purely optimization, not numerics.
- **TF32 is a lossy mode** (10-bit mantissa). V5/TF32 rows are verified with
  a relaxed tolerance; `-v` on such builds with `-i constant` gives the
  cleanest comparison.
- Repeated timing iterations keep `beta ≠ 0` (library-idiomatic accumulation
  across reps); verification always restarts from a pristine copy of C.

## See also

Heavily inspired by: https://github.com/hma02/cublasgemm-benchmark/tree/master

* [CUDA C++ Best Practices Guide — implicit GEMM chapters](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
* [The Matrix-Cores FAQ (NVIDIA)](https://github.com/LeftNotEasy/MATRIX-FAQ)
* [rocWMMA API guide (AMD)](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html)
* [BLAS Interface for Different Precision](http://www.netlib.org/utk/people/JackDongarra/WEB-PAGES/Batched-BLAS-2016/Day1/precision-blas.pdf)
* [OpenAI gemm](https://github.com/openai/openai-gemm)
* [Useful nvidia-smi Queries](http://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries)
* [TheBandwidthBenchmark](https://github.com/RRZE-HPC/TheBandwidthBenchmark) — portability layer inspiration
