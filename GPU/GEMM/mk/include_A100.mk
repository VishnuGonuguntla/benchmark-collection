# NVIDIA A100 (compute_80) — CUDA backend
_NVCC = 1

CUDA_ARCH_FLAGS ?= -gencode arch=compute_80,code=sm_80

OPTIONS  =  -D_NVCC
OPTIONS +=  -DSIZE=14080
OPTIONS +=  -DNTIMES=20
OPTIONS +=  -DSWEEPSIZE=30000
OPTIONS +=  -DSWEEPNTIMES=20
OPTIONS +=  -DMETRICS
# OPTIONS +=  -DTENSOR
# OPTIONS +=  -DTARGET_MINUTES=1.0

# Optional dense peaks (TFlop/s) enabling the '% Peak' output column.
# Values: A100 SXM4 datasheet.
# PEAK_FP64 = 9.7
# PEAK_FP32 = 19.5
# PEAK_FP16 = 312.5