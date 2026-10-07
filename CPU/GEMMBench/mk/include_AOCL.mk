# ---------------------------------------------------------------------------
# Legacy link-time build against AMD AOCL-BLIS (uarch-tuned BLIS, AMD only).
# Used by: make OPTIMIZATION=AOCL   (with TOOLCHAIN=GCC|AOCC|ICC|ICX)
#
# Location: config.mk's AOCL_PATH (the same variable the ladder's runtime-
# loaded rung reads), or AOCL_INCDIR/AOCL_LIBDIR as exported by an AOCL
# module.  The combined mk/include_<TOOLCHAIN>+AOCL.mk files remain available
# for reproducing past module-based measurements.
# ---------------------------------------------------------------------------
AOCL_INCDIR ?= $(if $(AOCL_PATH),$(AOCL_PATH)/include)
AOCL_LIBDIR ?= $(if $(AOCL_PATH),$(AOCL_PATH)/lib)

ENV_INCDIRS ?= $(if $(AOCL_INCDIR),-I$(AOCL_INCDIR))
ENV_LIBDIRS ?= $(if $(AOCL_LIBDIR),-L$(AOCL_LIBDIR))
ENV_DEFINES ?= -DDGEMM_BENCH_WITH_AOCL
# libblis-mt is the threaded (pthread) build of BLIS
ENV_LIBS    ?= -lblis-mt -lm -lpthread

$(if $(AOCL_LIBDIR),,\
  $(error OPTIMIZATION=AOCL needs a location: set AOCL_PATH=<install root> in config.mk, or export AOCL_LIBDIR via the aocl module))
$(if $(wildcard $(AOCL_LIBDIR)/libblis-mt.so $(AOCL_LIBDIR)/libblis.so),,\
  $(info note: OPTIMIZATION=AOCL found no libblis*.so in $(AOCL_LIBDIR)))
