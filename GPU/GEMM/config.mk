# ---------------------------------------------------------------------------
# Build configuration for the GEMM optimization-ladder benchmark.
#
# Everything below is OPTIONAL: uncomment / edit the lines you want to
# change.  Omitted settings fall back to the defaults in the Makefile
# (GPU=MI300X, DATATYPE=FP32, TENSOR=auto, LEVELS=all).
# ---------------------------------------------------------------------------

# GPU target — selects mk/include_$(GPU).mk, which sets the backend
# (_NVCC for CUDA, _HIP for ROCm) and the architecture flags.
#   NVIDIA (CUDA):  A40 | A100 | H100
#   AMD    (HIP):   MI300X
GPU = H100

# ---------------------------------------------------------------------------
# Datatype of A/B:
#   FP64 — double           (no V5 tensor rung; ladder ends at V4)
#   FP32 — float            (V5 = TF32 on NVIDIA; no tensor rung on AMD)
#   FP16 — half in, float C (V5 = WMMA/rocWMMA on both vendors)
# The legacy PRECISION=DP/SP switch still works (DP -> FP64, SP -> FP32).
DATATYPE = FP32

# ---------------------------------------------------------------------------
# Tensor-core ladder rung + cuBLAS tensor math:
#   auto — V5 enabled wherever the (backend, datatype) supports it
#   on   — same, but fails the build when no tensor path exists
#   off  — SIMT ladder only (V0..V4) + exact vendor BLAS
TENSOR = auto

# ---------------------------------------------------------------------------
# Optimization levels compiled into the binary: 'all' or a comma/space list
# from 0 1 2 3 4 5 blas.  The runtime -l flag can only select from these.
# (Legacy OPTIMIZATION=NAIVE maps to level 1, VENDOR maps to blas.)
LEVELS = all

# ---------------------------------------------------------------------------
# Optional: compile-time default for the -t runtime flag (minutes).
# ---------------------------------------------------------------------------
# OPTIONS += -DTARGET_MINUTES=5.0
