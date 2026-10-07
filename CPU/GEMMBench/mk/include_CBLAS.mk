# ---------------------------------------------------------------------------
# Legacy link-time build against the system BLAS (reference CBLAS interface,
# whatever libblas/libcblas the environment provides).
# Used by: make OPTIMIZATION=CBLAS   (TOOLCHAIN=GCC|AOCC|ICC|ICX)
# ---------------------------------------------------------------------------
CBLAS_INCDIR ?= $(if $(CBLAS_PATH),$(CBLAS_PATH)/include)
CBLAS_LIBDIR ?= $(if $(CBLAS_PATH),$(CBLAS_PATH)/lib)

ENV_INCDIRS ?= $(if $(CBLAS_INCDIR),-I$(CBLAS_INCDIR))
ENV_LIBDIRS ?= $(if $(CBLAS_LIBDIR),-L$(CBLAS_LIBDIR))
ENV_DEFINES ?= -DDGEMM_BENCH_WITH_CBLAS
ENV_LIBS    ?= -lcblas -lblas -lm
