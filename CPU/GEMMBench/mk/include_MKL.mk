# MKL location: the module's MKLROOT wins, then config.mk's MKL_PATH (shared
# with the ladder's runtime-loaded rung), then the pinned install below so
# past measurements on this cluster stay reproducible.
ifeq ($(strip $(MKLROOT)),)
  MKLROOT := $(if $(MKL_PATH),$(MKL_PATH),\
    /apps/spack/1.0.2/opt/linux-almalinux9-icelake/none-none/intel-oneapi-mkl-2024.2.2-qvjxvr2b5cebxqymrzkfopa2tr6c2g4m/mkl/2024.2)
endif

ENV_INCDIRS ?= -I$(MKLROOT)/include
ENV_DEFINES ?= -DDGEMM_BENCH_WITH_MKL

ifeq ($(TOOLCHAIN),ICX)
ENV_LIBDIRS ?= -L$(MKLROOT)/lib/intel64_lin
ENV_LIBS ?= -lmkl_intel_thread -lmkl_rt -lmkl_core -lmkl_intel_lp64
ENV_MODULES ?= intel mkl
endif

ifeq ($(TOOLCHAIN),GCC)
ENV_LIBDIRS ?= -L$(MKLROOT)/lib/intel64
ENV_LIBS ?= -lmkl_rt -lpthread -lm -ldl
ENV_MODULES ?= mkl
endif

