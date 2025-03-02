ENV_CC ?= icc
ENV_CFLAGS ?= -diag-disable?=10441 -O3
ENV_INCDIRS ?= -I$(AOCL_INCDIR)
ENV_LIBDIRS ?= -L$(AOCL_LIBDIR)
ENV_OMP_FLAG ?= -qopenmp
ENV_DEFINES ?= -DDGEMM_BENCH_WITH_AOCL
ENV_LIBS ?= -lblis-mt -liomp5

ENV_WRAP_CMD ?= LD_PRELOAD?=$(CMPLR_ROOT)/linux/compiler/lib/intel64_lin/libiomp5.so
ENV_MODULES ?= intel aocl
