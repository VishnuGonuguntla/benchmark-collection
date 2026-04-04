# CuBLAS / hipBLAS GEMM benchmark

A simple and repeatable benchmark for validating GPU performance based on cuBLAS (NVIDIA) or hipBLAS (AMD ROCm) matrix multiplication. Supports double precision (DGEMM) and single precision (SGEMM) through a single `config.mk` switch, and runs on both CUDA and HIP backends.

## Contents

| File | Description |
|------|-------------|
| `gemm.cu` | GEMM for a single fixed matrix size |
| `gemm_sweep.cu` | GEMM swept across a range of matrix sizes |
| `portability.h` | HIP/CUDA portability layer (maps cuBLAS → hipBLAS, etc.) |
| `cli.h` | CLI argument parser (`-d`, `-s`, `-n`, `-t`, `-h`) |
| `config.mk` | User-facing build configuration |
| `Makefile` | Build rules |
| `mk/` | Per-GPU architecture flags and default problem sizes |

The code computes `C = alpha*A*B + beta*C` with square matrices and reports performance in TFlop/s. Both executables accept the same flags and support either a fixed iteration count or a wall-clock time limit.

## Prerequisites

**CUDA (NVIDIA):**
- NVHPC SDK or CUDA Toolkit (nvcc, cuBLAS, NVML)

**HIP / ROCm (AMD):**
- ROCm toolkit (hipcc, hipBLAS)
- hipRAND (device-side header `hiprand_kernel.h`)
- ROCm SMI library (`rocm_smi_lib`, always present on ROCm systems)

## Build configuration (`config.mk`)

```makefile
# Target GPU — also determines the backend (CUDA or HIP).
# Each mk/include_$(GPU).mk sets _NVCC=1 or _HIP=1,
# which the Makefile uses to select nvcc or hipcc automatically.
GPU = H100

# Floating-point precision
#   DP — double precision (cublasDgemm / hipblasDgemm)
#   SP — single precision (cublasSgemm / hipblasSgemm)
PRECISION = DP
```

### Building for NVIDIA (CUDA)

```bash
make                        # defaults: GPU=H100 PRECISION=DP
make GPU=A100 PRECISION=SP  # override at command line
```

### Building for AMD (HIP / ROCm)

```bash
make GPU=MI300X PRECISION=DP
```

### Cleaning

```bash
make clean
```

## Runtime flags

Both `gemm` and `gemm_sweep` accept the same CLI flags:

| Flag | Description | Default |
|------|-------------|---------|
| `-h` | Show help and exit | — |
| `-d <int>` | GPU device ID | `0` |
| `-s <int>` | Matrix size (`gemm`) or max sweep size (`gemm_sweep`) | from `config.mk` |
| `-n <int>` | Number of iterations (used when `-t 0`) | from `config.mk` |
| `-t <double>` | Target duration in minutes; `0` = use `-n` | `0` |

The compile-time defines (`-DSIZE`, `-DNTIMES`, etc. from the `mk/` files) become the defaults for the respective flags, so CLI args always override them without recompiling.

### Examples

```bash
# Fixed iterations
./gemm -d 0 -s 16384 -n 500

# Time-based (run for 5 minutes)
./gemm -d 0 -s 16384 -t 5.0

# Sweep, device with gpu-id=1 and max size
./gemm_sweep -d 1 -s 32768 -n 100

# Sweep, time-based (1 minute per matrix size)
./gemm_sweep -d 0 -t 1.0
```

## Output format

**Without `-DMETRICS`:**
```
size 16384 | Time: 42.3 s | Repeats: 200 | Perf: 18.4 TFlop/s
```

**With `-DMETRICS` (CUDA, NVML):**
```
size 16384 | Time: 42.3 s | Repeats: 200 | Perf: 18.4 TFlop/s | Avg Power: 397.2 W | Avg Clock: 1410 MHz | Avg Temp: 77.1 C | SM Util: 99.8 % | Mem Util: 62.4 %
```

**With `-DMETRICS` (HIP, ROCm SMI):**
```
size 65536 | Time: 38.1 s | Repeats: 100 | Perf: 96.2 TFlop/s | Avg Power: 512.4 W | Avg Clock: 2100 MHz | Avg Temp: 68.3 C | CU Util: 99.5 % | Mem Util: 71.2 %
```

## Architecture-specific settings (`mk/`)

Each file in `mk/` sets the CUDA/HIP architecture flags and the default problem sizes for a given GPU. The correct file is selected automatically based on `GPU` in `config.mk`.

| GPU | Arch | Backend flag | File |
|-----|------|-------------|------|
| A100 | sm_80 | `_NVCC = 1` | `mk/include_A100.mk` |
| A40 | sm_86 | `_NVCC = 1` | `mk/include_A40.mk` |
| H100 | sm_90 | `_NVCC = 1` | `mk/include_H100.mk` |
| MI300X | gfx942 | `_HIP = 1` | `mk/include_MI300X.mk` |

You can add your own GPU file under `mk/` and reference it via `config.mk`.

## Compile-time options (set in `mk/` files or via `OPTIONS +=`)

| Option | Effect |
|--------|--------|
| `-DMETRICS` | Enable GPU telemetry: power, clock, temperature, GPU utilization, memory utilization. Automatically uses NVML on CUDA builds and ROCm SMI on HIP builds — no separate flag needed. |
| `-DTENSOR` | Enable TF32 Tensor core math mode (`cublasSgemm` only; auto-disabled on HIP and when `PRECISION=DP`) |
| `-DTARGET_MINUTES=<val>` | Set compile-time default for the `-t` flag (can still be overridden at runtime) |


## See also

Heavily inspired by: https://github.com/hma02/cublasgemm-benchmark/tree/master

* [BLAS Interface for Different Precision](http://www.netlib.org/utk/people/JackDongarra/WEB-PAGES/Batched-BLAS-2016/Day1/precision-blas.pdf)
* [OpenAI gemm](https://github.com/openai/openai-gemm)
* [Useful nvidia-smi Queries](http://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries)
* [TheBandwidthBenchmark](https://github.com/RRZE-HPC/TheBandwidthBenchmark) — portability layer inspiration
