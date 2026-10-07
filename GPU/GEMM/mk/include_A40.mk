# NVIDIA A40 (compute_86) — CUDA backend
_NVCC = 1

CUDA_ARCH_FLAGS ?= -gencode arch=compute_86,code=sm_86

OPTIONS  =  -D_NVCC
OPTIONS +=  -DSIZE=42000
OPTIONS +=  -DNTIMES=2
OPTIONS +=  -DSWEEPSIZE=42000
OPTIONS +=  -DSWEEPNTIMES=30
OPTIONS +=  -DMETRICS
# OPTIONS +=  -DTENSOR
# OPTIONS +=  -DTARGET_MINUTES=1.0

# Optional dense peaks (TFlop/s) enabling the '% Peak' output column.
# Values: A40 datasheet (FP64 rate is heavily cut on GA102; FP16 tensor
# with FP32 accumulate).
# PEAK_FP64 = 0.58
# PEAK_FP32 = 37.4
# PEAK_FP16 = 74.8