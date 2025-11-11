MAKE_DIR   = ./mk

include config.mk
include $(MAKE_DIR)/include_$(GPU).mk

CC_FLAGS += -O3

CC_FLAGS += --std=c++17 $(CUDA_ARCH_FLAGS) $(OPTIONS)

# LIB_FLAGS += -lcurand
LIB_FLAGS += -lcublas -lnvidia-ml

EXE = gemm
EXE_SWEEP = gemm_sweep

all: $(EXE) $(EXE_SWEEP)

% : %.cu
	nvcc $< $(CC_FLAGS) $(LIB_FLAGS) -o $@

clean:
	rm -f $(EXE) $(EXE_SWEEP)