# NVIDIA H100 (compute_90) — CUDA backend
_NVCC = 1

CUDA_ARCH_FLAGS ?= -gencode arch=compute_90,code=sm_90

OPTIONS  =  -D_NVCC
OPTIONS +=  -DSIZE=4096
OPTIONS +=  -DNTIMES=10
OPTIONS +=  -DSWEEPSIZE=18000
OPTIONS +=  -DSWEEPNTIMES=20
OPTIONS +=  -DMETRICS
# OPTIONS +=  -DTENSOR
# OPTIONS +=  -DTARGET_MINUTES=1.0

# Optional dense peaks (TFlop/s) enabling the '% Peak' output column.
# Convention: vector-peak for FP64/FP32, tensor-peak (fp16->fp32, no
# sparsity) for FP16.  Tensor-core rows above 100% of a vector peak is
# exactly the point of the ladder.  Values: H100 SXM5 datasheet.
# PEAK_FP64 = 33.5
# PEAK_FP32 = 66.9
# PEAK_FP16 = 989.4