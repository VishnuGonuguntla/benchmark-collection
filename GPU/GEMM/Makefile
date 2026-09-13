MAKE_DIR = ./mk

include config.mk
include $(MAKE_DIR)/include_$(GPU).mk

# ---------------------------------------------------------------------------
# Backend detection: _HIP or _NVCC is set by the GPU-specific mk file.
# This mirrors the approach used in TheBandwidthBenchmark.
# ---------------------------------------------------------------------------
ifdef _HIP
  COMPILER   = hipcc
  LIB_FLAGS += -lhipblas -lrocm_smi64   # rocm_smi64 always present on ROCm systems
else
  COMPILER   = nvcc
  LIB_FLAGS += -lcublas -lnvidia-ml -lcuda
endif

# ---------------------------------------------------------------------------
# Precision: DP adds -DDOUBLE to select cublasDgemm/hipblasDgemm + double type
# ---------------------------------------------------------------------------
ifeq ($(PRECISION), DP)
  CC_FLAGS += -DDOUBLE
endif

ifeq ($(OPTIMIZATION), NAIVE)
  CC_FLAGS += -DNAIVE
else
  CC_FLAGS += -DVENDOR
endif

CC_FLAGS += -O3 --std=c++17 $(CUDA_ARCH_FLAGS) $(OPTIONS)

EXE       = gemm
EXE_SWEEP = gemm_sweep

all: $(EXE) $(EXE_SWEEP)

% : %.cu
	$(COMPILER) $< $(CC_FLAGS) $(LIB_FLAGS) -o $@

clean:
	rm -f $(EXE) $(EXE_SWEEP)