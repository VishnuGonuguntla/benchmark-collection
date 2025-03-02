ENV_CC ?= icx
ENV_CFLAGS ?= -O3 -qmkl
ENV_INCDIRS ?= -I$(MKLROOT)/include
ENV_LIBDIRS ?= -L$(MKLROOT)/lib/intel64_lin
ENV_OMP_FLAG ?= -qopenmp
ENV_DEFINES ?= -DDGEMM_BENCH_WITH_MKL
ENV_LIBS ?= -lmkl_intel_thread -lmkl_rt -lmkl_core -lmkl_intel_lp64
ENV_MODULES ?= intel mkl
