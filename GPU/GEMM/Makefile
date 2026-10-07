MAKE_DIR = ./mk

include config.mk
GPU ?= MI300X
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
# Datatype.  The modern knob is DATATYPE = FP64 | FP32 | FP16.
# The legacy PRECISION=DP/SP switch is still accepted (DP -> FP64, SP -> FP32).
# ---------------------------------------------------------------------------
ifeq ($(DATATYPE),)
  ifeq ($(PRECISION),DP)
    DATATYPE = FP64
  else
    DATATYPE = FP32
  endif
endif

ifeq ($(DATATYPE), FP64)
  CC_FLAGS += -DDTYPE_FP64
else ifeq ($(DATATYPE), FP16)
  CC_FLAGS += -DDTYPE_FP16
else ifeq ($(DATATYPE), FP32)
  CC_FLAGS += -DDTYPE_FP32
else
  $(error Unknown DATATYPE '$(DATATYPE)' - use FP64, FP32 or FP16)
endif

# Legacy OPTIMIZATION=NAIVE|VENDOR maps onto the new ladder (before LEVELS is
# evaluated; a user-set LEVELS still wins because it comes from the command
# line or config.mk).
ifdef OPTIMIZATION
  ifeq ($(OPTIMIZATION),NAIVE)
    LEVELS = 1
  else ifeq ($(OPTIMIZATION),VENDOR)
    LEVELS = blas
  endif
endif

# ---------------------------------------------------------------------------
# Tensor-core ladder rung: TENSOR = auto | on | off
#   auto - enable V5 where the backend/datatype combination supports it
#   on   - same, but hard-fails at compile time when no path exists
#   off  - never (SIMT ladder only)
# FP16 on AMD additionally requires the rocWMMA headers to be installed.
# ---------------------------------------------------------------------------
TENSOR ?= auto
ifeq ($(TENSOR),off)
  CC_FLAGS += -DTENSOR_OFF
else ifeq ($(TENSOR),on)
  CC_FLAGS += -DTENSOR_ON
endif

ifdef _HIP
  ROCWMMA_HDR := $(firstword $(wildcard /opt/rocm/include/rocwmma/rocwmma.hpp \
                                        /opt/rocm-*/include/rocwmma/rocwmma.hpp))
  ifneq ($(ROCWMMA_HDR),)
    CC_FLAGS += -DHAVE_ROCWMA
    CC_FLAGS += -I$(patsubst %/rocwmma/rocwmma.hpp,%,$(dir $(ROCWMMA_HDR))..)
  endif
endif

# ---------------------------------------------------------------------------
# Compile-time level selection: LEVELS = all | space/comma list of 0..5 blas
# Encoded as a bitmask (bit i = level i, bit 6 = vendor BLAS).
# ---------------------------------------------------------------------------
comma := ,
empty :=
space := $(empty) $(empty)
LEVELS ?= all
LEVEL_TOKENS = $(subst $(comma),$(space),$(LEVELS))

# Per-token bit values (used for anything but 'all').
LEVEL_BIT_0    := 1
LEVEL_BIT_1    := 2
LEVEL_BIT_2    := 4
LEVEL_BIT_3    := 8
LEVEL_BIT_4    := 16
LEVEL_BIT_5    := 32
LEVEL_BIT_blas := 64
LEVEL_BIT_BLAS := 64
LEVEL_BIT_6    := 64

ifeq ($(filter all,$(LEVEL_TOKENS)),all)
  LEVEL_MASK := 127
else
  LEVEL_BITS := $(foreach t,$(LEVEL_TOKENS),$(LEVEL_BIT_$(t)))
  # Sum the bits.  Reject empty result / unknown tokens (unknown tokens expand
  # to nothing, so a bogus-only list yields an empty LEVEL_BITS).
  ifeq ($(strip $(LEVEL_BITS)),)
    $(error Failed to parse LEVELS='$(LEVELS)' — use 'all' or tokens 0..5/blas)
  endif
  LEVEL_MASK := $(shell printf '%s\n' $(LEVEL_BITS) | paste -sd+ | bc)
endif
CC_FLAGS += -DLEVEL_MASK=$(LEVEL_MASK)

# ---------------------------------------------------------------------------
# Optional theoretical dense peak ('% Peak' column).  Each mk file may define
# PEAK_FP64 / PEAK_FP32 / PEAK_FP16 (dense, non-sparse datasheet TFlop/s);
# tensor-core rows exceeding 100% of a vector peak is the whole point.
# ---------------------------------------------------------------------------
PEAK_VAL = $(PEAK_$(DATATYPE))
ifneq ($(strip $(PEAK_VAL)),)
  CC_FLAGS += -DPEAK_TFLOPS=$(PEAK_VAL)
endif

CC_FLAGS += -O3 --std=c++17 $(CUDA_ARCH_FLAGS) $(OPTIONS) $(INCLUDES)

SRC_DIR   = src
EXE       = gemm
EXE_SWEEP = gemm_sweep
HEADERS   = $(wildcard $(SRC_DIR)/*.h $(SRC_DIR)/*.cuh) Makefile config.mk $(MAKE_DIR)/include_$(GPU).mk

all: $(EXE) $(EXE_SWEEP)

% : $(SRC_DIR)/%.cu $(HEADERS)
	$(COMPILER) $< $(CC_FLAGS) $(LIB_FLAGS) -o $@

clean:
	rm -f $(EXE) $(EXE_SWEEP)
