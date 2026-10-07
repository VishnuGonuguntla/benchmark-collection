# ---------------------------------------------------------------------------
# Legacy link-time build against OpenBLAS.
# Used by: make OPTIMIZATION=OPENBLAS   (TOOLCHAIN=GCC|AOCC|ICC|ICX)
#
# Portable across every x86-64 / arm64 microarchitecture (DYNAMIC_ARCH builds
# choose the kernel set at run time), which makes this the one legacy variant
# that is meaningful on any CPU.  Location comes from config.mk's
# OPENBLAS_PATH, or from OPENBLAS_INCDIR/OPENBLAS_LIBDIR when the library is
# provided by a module.
# ---------------------------------------------------------------------------
OPENBLAS_INCDIR ?= $(if $(OPENBLAS_PATH),$(OPENBLAS_PATH)/include)
OPENBLAS_LIBDIR ?= $(if $(OPENBLAS_PATH),$(OPENBLAS_PATH)/lib)

ENV_INCDIRS ?= $(if $(OPENBLAS_INCDIR),-I$(OPENBLAS_INCDIR))
ENV_LIBDIRS ?= $(if $(OPENBLAS_LIBDIR),-L$(OPENBLAS_LIBDIR))
ENV_DEFINES ?= -DDGEMM_BENCH_WITH_OPENBLAS
# OpenBLAS ships the reference CBLAS interface (cblas_sgemm/cblas_dgemm);
# the soname encodes the threading/build flavour (-p-, -openrpc-, 64_ ...).
ENV_LIBS    ?= -lopenblas -lm -lpthread

$(if $(OPENBLAS_LIBDIR),,\
  $(error OPTIMIZATION=OPENBLAS needs a location: set OPENBLAS_PATH=<install root> in config.mk, or export OPENBLAS_LIBDIR))
$(if $(wildcard $(OPENBLAS_LIBDIR)/libopenblas.so $(OPENBLAS_LIBDIR)/libopenblas.so.*),,\
  $(info note: OPTIMIZATION=OPENBLAS found no libopenblas*.so in $(OPENBLAS_LIBDIR)))
