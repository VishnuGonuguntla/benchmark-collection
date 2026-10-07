# AMD Instinct MI300X (gfx942) — HIP/ROCm backend
_HIP = 1

CUDA_ARCH_FLAGS ?= --offload-arch=gfx942 -Wno-switch
INCLUDES  = -I/opt/rocm-7.2.4/include/hiprand

OPTIONS  =  -D_HIP
OPTIONS +=  -DSIZE=37632
OPTIONS +=  -DNTIMES=30
OPTIONS +=  -DSWEEPSIZE=90000
OPTIONS +=  -DSWEEPNTIMES=100
OPTIONS +=  -DMETRICS
# OPTIONS +=  -DTENSOR
# OPTIONS +=  -DTARGET_MINUTES=1.0

# Optional dense peaks (TFlop/s) enabling the '% Peak' output column.
# Convention: vector-peak for FP64/FP32, matrix-peak for FP16.
# Values: MI300X datasheet (FP64 matrix: 163.4).
# PEAK_FP64 = 81.7
# PEAK_FP32 = 163.4
# PEAK_FP16 = 1307.4