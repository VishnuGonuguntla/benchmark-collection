ENV_CC ?= gcc
ENV_CFLAGS ?= -O3
ENV_INCDIRS ?= -I$(MKLROOT)/include
ENV_LIBDIRS ?= -L$(MKLROOT)/lib/intel64
ENV_OMP_FLAG ?= -fopenmp
ENV_DEFINES ?= -DDGEMM_BENCH_WITH_MKL
ENV_LIBS ?= -lmkl_rt -lpthread -lm -ldl
ENV_MODULES ?= mkl
