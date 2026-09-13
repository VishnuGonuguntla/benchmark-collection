# ---------------------------------------------------------------------------
# GPU target — selects the mk/include_$(GPU).mk file, which also sets the
# backend (_NVCC for CUDA, _HIP for ROCm) and architecture flags.
#
# NVIDIA (CUDA):  A40 | A100 | H100
# AMD    (HIP):   MI300X
# ---------------------------------------------------------------------------
GPU ?= MI300X

# ---------------------------------------------------------------------------
# Floating-point precision
#   DP — double precision  (cublasDgemm / hipblasDgemm)
#   SP — single precision  (cublasSgemm / hipblasSgemm)
# ---------------------------------------------------------------------------
PRECISION ?= SP
OPTIMIZATION ?= NAIVE

# ---------------------------------------------------------------------------
# Optional: compile-time default for the -t runtime flag (minutes).
# Uncomment to enable time-based execution by default.
# ---------------------------------------------------------------------------
# OPTIONS += -DTARGET_MINUTES=5.0
