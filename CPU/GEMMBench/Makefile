# ---------------------------------------------------------------------------
# CPU GEMM benchmark collection
#
#   OPTIMIZATION = LADDER   → the new multi-level ladder binary
#                             (V0..V5 + runtime-loaded vendor BLAS rungs:
#                              MKL, AOCL-BLIS, OpenBLAS — see BLAS_LIBS)
#   OPTIMIZATION = MKL|AOCL|OPENBLAS|CBLAS|NAIVE
#                           → legacy single-variant binaries (link-time BLAS)
# ---------------------------------------------------------------------------

include config.mk

# ---------------------------------------------------------------------------
# Ladder build
# ---------------------------------------------------------------------------
ifeq ($(OPTIMIZATION),LADDER)

include mk/include_$(TOOLCHAIN).mk

CC  = $(ENV_CC)
LD  = $(ENV_CC)

# ---- datatype (DATATYPE wins; legacy PRECISION as alias) -------------------
ifeq ($(strip $(DATATYPE)),)
  ifeq ($(PRECISION),SP)
    DATATYPE = FP32
  else
    DATATYPE = FP64
  endif
endif
ifeq ($(DATATYPE),FP64)
  LDEFS += -DDATATYPE_FP64
else ifeq ($(DATATYPE),FP32)
  LDEFS += -DDATATYPE_FP32
else
  $(error Unknown DATATYPE '$(DATATYPE)' - use FP64 or FP32)
endif

# ---- vendor BLAS registry ---------------------------------------------------
# One stanza per runtime-loaded library: the config.mk variable that locates
# it, where to look inside an installation root, which sonames count, the
# macros src/library.c reads and the ladder level bit it owns.
# Adding a library = a stanza here + a row in BLAS_REGISTRY (src/library.c)
# + a rung id (src/levels.h) +, to make 'auto' pick it up, its CPU vendor.
comma := ,
empty :=
space := $(empty) $(empty)

BLAS_TAGS := mkl aocl openblas

BLAS_PATH_mkl         := $(MKL_PATH)
BLAS_PATHVAR_mkl      := MKL_PATH
BLAS_DIRS_mkl         := lib/intel64 lib64 lib mkl/latest/lib/intel64 mkl/2024.2/lib/intel64
BLAS_SONAMES_mkl      := libmkl_rt.so
BLAS_MACRO_mkl        := MKL_LIB_DEFAULT
BLAS_HAS_mkl          := HAS_MKL_LEVEL
BLAS_BIT_mkl          := 128

BLAS_PATH_aocl        := $(AOCL_PATH)
BLAS_PATHVAR_aocl     := AOCL_PATH
BLAS_DIRS_aocl        := lib lib64
BLAS_SONAMES_aocl     := libblis-mt.so libblis.so
BLAS_MACRO_aocl       := AOCL_LIB_DEFAULT
BLAS_HAS_aocl         := HAS_AOCL_LEVEL
BLAS_BIT_aocl         := 256

BLAS_PATH_openblas    := $(OPENBLAS_PATH)
BLAS_PATHVAR_openblas := OPENBLAS_PATH
BLAS_DIRS_openblas    := lib lib64
BLAS_SONAMES_openblas := libopenblas.so libopenblas64_.so libopenblas*.so
BLAS_MACRO_openblas   := OPENBLAS_LIB_DEFAULT
BLAS_HAS_openblas     := HAS_OPENBLAS_LEVEL
BLAS_BIT_openblas     := 512

# Which libraries get a rung: auto | none | all | list of BLAS_TAGS
# (config.mk documents the choice; the default is the per-CPU-vendor 'auto')
BLAS_LIBS ?= auto
BLAS_WANTED := $(subst $(comma),$(space),$(BLAS_LIBS))
$(if $(filter-out $(BLAS_TAGS) auto none all,$(BLAS_WANTED)),\
     $(error BLAS_LIBS='$(BLAS_LIBS)' - use auto, none, all or any of: $(BLAS_TAGS)))

# 'auto': which rung per target.  The toolchain is the strongest hint about
# what the build is for (AOCC is the AMD compiler, ICC/ICX ship MKL), so it
# wins over the build host's CPU vendor — useful when you build for a
# cluster's compute nodes on a differently-populated login node.  OpenBLAS
# accompanies every choice because its kernel set is picked at run time, so
# the same library is meaningful on Intel, AMD and Arm alike.
ifeq ($(BLAS_WANTED),auto)
  ifeq ($(TOOLCHAIN),AOCC)
    BLAS_AUTO_WHY := AOCC toolchain (AMD target)
    BLAS_WANTED   := aocl openblas
  else ifeq ($(TOOLCHAIN),ICC)
    BLAS_AUTO_WHY := ICC toolchain (Intel target)
    BLAS_WANTED   := mkl openblas
  else ifeq ($(TOOLCHAIN),ICX)
    BLAS_AUTO_WHY := ICX toolchain (Intel target)
    BLAS_WANTED   := mkl openblas
  else
    BLAS_CPU_VENDOR := $(shell awk -F': *' '/vendor_id/{print $$2; exit}' /proc/cpuinfo 2>/dev/null)
    ifeq ($(BLAS_CPU_VENDOR),AuthenticAMD)
      BLAS_AUTO_WHY := $(BLAS_CPU_VENDOR) host
      BLAS_WANTED   := aocl openblas
    else ifeq ($(BLAS_CPU_VENDOR),GenuineIntel)
      BLAS_AUTO_WHY := $(BLAS_CPU_VENDOR) host
      BLAS_WANTED   := mkl openblas
    else
      BLAS_AUTO_WHY := portable default
      BLAS_WANTED   := openblas
    endif
  endif
  BLAS_WHY := auto via $(BLAS_AUTO_WHY)
  # a detected target may simply have no library installed: skip with a note
  BLAS_LENIENT := 1
endif
# 'none' / 'all' are wildcards too: 'all' means "every library that resolves"
ifeq ($(filter none,$(BLAS_WANTED)),none)
  BLAS_WANTED :=
else ifeq ($(filter all,$(BLAS_WANTED)),all)
  BLAS_WANTED := $(BLAS_TAGS)
  BLAS_WHY := all
  BLAS_LENIENT := 1
endif

# $(call blas-resolve,<tag>) — the library's path variable holds either a full
# shared-object path or an installation root (its lib*/ subdirectories are
# searched); returns the absolute path of the object to load, or the empty
# string when nothing was found.  Nothing is linked here, this is only what
# the rung will dlopen() at run time.
define blas-resolve
$(strip $(if $(BLAS_PATH_$(1)),\
  $(if $(wildcard $(BLAS_PATH_$(1))/.),\
    $(realpath $(firstword \
      $(foreach d,$(BLAS_PATH_$(1)) $(addprefix $(BLAS_PATH_$(1))/,$(BLAS_DIRS_$(1))),\
        $(foreach s,$(BLAS_SONAMES_$(1)),$(wildcard $(d)/$(s) $(d)/$(s).*))))),\
    $(if $(wildcard $(BLAS_PATH_$(1))),$(realpath $(BLAS_PATH_$(1)))))))
endef

$(foreach t,$(BLAS_WANTED),$(eval BLAS_SO_$(t) := $(call blas-resolve,$(t))))

BLAS_ENABLED :=
$(foreach t,$(BLAS_WANTED),\
  $(if $(BLAS_SO_$(t)),\
    $(eval BLAS_ENABLED += $(t))\
    $(eval LDEFS += -D$(BLAS_MACRO_$(t))='"$(BLAS_SO_$(t))"' -D$(BLAS_HAS_$(t))=1),\
    $(if $(BLAS_LENIENT),\
      $(info note: BLAS_LIBS=$(BLAS_WHY) - '$(t)' skipped: $(BLAS_PATHVAR_$(t)) $(if $(BLAS_PATH_$(t)),holds no matching library,is unset)),\
      $(error BLAS_LIBS lists '$(t)' but $(BLAS_PATHVAR_$(t)) $(if $(BLAS_PATH_$(t)),holds no matching library,is unset) - set it to the install root or the full .so path))))

# ---- compile-time level mask ----------------------------------------------
# tokens: 0..5 (hand-written rungs), bf16 (V6, see below) and the library
# tags of the registry above
LEVEL_BIT_0    := 1
LEVEL_BIT_1    := 2
LEVEL_BIT_2    := 4
LEVEL_BIT_3    := 8
LEVEL_BIT_4    := 16
LEVEL_BIT_5    := 32
LEVEL_BIT_bf16 := 64
LEVEL_HAND     := 0 1 2 3 4 5

# ---- BF16 rung (V6) ---------------------------------------------------------
# src/bf16.c is the only TU compiled with -mavx512bf16 (target-specific
# LFLAGS below), so the codegen of the other rungs never changes.  The
# kernel computes the fp32-output ladder problem, so it needs DATATYPE=FP32;
# LEVELS=all enables it whenever the compiler takes the flags, an explicit
# 'bf16' token fails loudly when it cannot be built.  CPUs without
# AVX512_BF16 skip the rung at run time (the harness prints NOT AVAILABLE).
BF16_FLAGS := -mavx512f -mavx512bf16
BF16_PROBE := $(shell echo | $(ENV_CC) $(BF16_FLAGS) -O0 -x c -c -o /dev/null - >/dev/null 2>&1 && echo yes)

LEVELS ?= all
LEVEL_TOKENS = $(subst $(comma),$(space),$(LEVELS))

$(if $(filter all,$(LEVEL_TOKENS)),,\
  $(if $(filter-out $(LEVEL_HAND) bf16 $(BLAS_TAGS),$(LEVEL_TOKENS)),\
       $(error Unknown LEVELS token(s) '$(filter-out $(LEVEL_HAND) bf16 $(BLAS_TAGS),$(LEVEL_TOKENS))' - use $(LEVEL_HAND), bf16, the library tags $(BLAS_TAGS), or 'all')))

# BF16 enablement: 'all' picks it leniently, an explicit token demands it
# (and gets a hard error when it cannot be built)
BF16_ON :=
ifneq ($(filter bf16,$(LEVEL_TOKENS)),)
  ifneq ($(DATATYPE),FP32)
    $(error LEVELS lists 'bf16' but the BF16 rung computes the fp32-output ladder problem - it needs DATATYPE=FP32)
  endif
  ifeq ($(BF16_PROBE),)
    $(error LEVELS lists 'bf16' but $(ENV_CC) does not accept $(BF16_FLAGS))
  endif
  BF16_ON := yes
endif
ifeq ($(filter all,$(LEVEL_TOKENS)),all)
  BF16_ON := $(if $(filter FP32,$(DATATYPE)),$(BF16_PROBE),)
endif
# same for the vendor tags
$(foreach t,$(filter $(BLAS_TAGS),$(LEVEL_TOKENS)),\
  $(if $(filter $(t),$(BLAS_ENABLED)),,\
    $(error LEVELS references '$(t)' but that rung is not built - enable it in BLAS_LIBS and set $(BLAS_PATHVAR_$(t)))))

ifeq ($(filter all,$(LEVEL_TOKENS)),all)
  LEVEL_BITS := $(foreach i,$(LEVEL_HAND),$(LEVEL_BIT_$(i)))\
                $(if $(BF16_ON),$(LEVEL_BIT_bf16))\
                $(foreach t,$(BLAS_ENABLED),$(BLAS_BIT_$(t)))
else
  LEVEL_BITS := $(foreach t,$(LEVEL_TOKENS),\
                 $(if $(filter $(t),$(BLAS_TAGS)),$(BLAS_BIT_$(t)),$(LEVEL_BIT_$(t))))
endif
ifeq ($(strip $(LEVEL_BITS)),)
  $(error LEVELS='$(LEVELS)' selects no rung)
endif
LEVEL_MASK := $(shell echo $$(( $(subst $(space),+,$(LEVEL_BITS))+0 )))
LDEFS += -DLEVEL_MASK=$(LEVEL_MASK)

$(info ladder: $(DATATYPE) datatype, level mask $(LEVEL_MASK))
$(info ladder: vendor rungs $(if $(BLAS_ENABLED),$(BLAS_ENABLED),none))
$(foreach t,$(BLAS_ENABLED),$(info ladder:   $(t) <- $(BLAS_SO_$(t))))
$(info ladder: BF16 rung $(if $(BF16_ON),on ($(BF16_FLAGS) on src/bf16.c only),off$(if $(filter all,$(LEVEL_TOKENS)), - LEVELS=all: $(if $(filter FP32,$(DATATYPE)),compiler lacks $(BF16_FLAGS),needs DATATYPE=FP32))))

# ---- rules -----------------------------------------------------------------
LADDER_COMMON = obj/runner.o obj/levels.o obj/library.o obj/numa.o obj/timing.o obj/profiler.o obj/bf16.o
LFLAGS = $(ENV_CFLAGS) $(ENV_OMP_FLAG) $(ENV_INCDIRS) $(LDEFS)
# Ladder-specific arch default: x86-64-v3 (AVX2+FMA era).  On the target
# machine itself, overriding with  make LADDER_ARCH=-march=native  is the
# honest way to benchmark SIMD rungs (V2+ benefit directly).
LADDER_ARCH ?=  -march=x86-64-v4 #  -mcpu=neoverse-v2 
LFLAGS += $(LADDER_ARCH)
LLIBS  = $(ENV_LIBDIRS) $(ENV_LIBS) -ldl -lm

# ---- LIKWID marker API ------------------------------------------------------
# LIKWID=1 builds the rungs with marker regions (see src/profiler.h) so the
# ladder can be run through  likwid-perfctr -m -g <GROUP> ./gemm_bench_...
# LIKWID_PATH may point at an installation root; when empty, likwid-perfctr
# from $PATH locates it (e.g. after "module load likwid").
LIKWID ?= 0
ifeq ($(LIKWID),1)
  ifeq ($(strip $(LIKWID_PATH)),)
    LIKWID_BIN := $(shell command -v likwid-perfctr 2>/dev/null)
    ifneq ($(LIKWID_BIN),)
      LIKWID_PATH := $(realpath $(dir $(LIKWID_BIN))/..)
    endif
  endif
  ifeq ($(strip $(LIKWID_PATH)),)
    $(error LIKWID=1 needs LIKWID_PATH or likwid-perfctr on $$PATH - module load likwid or set LIKWID_PATH=/path/to/likwid)
  endif
  LIKWID_LIBDIR := $(firstword $(wildcard $(LIKWID_PATH)/lib $(LIKWID_PATH)/lib64))
  ifeq ($(strip $(LIKWID_LIBDIR)),)
    $(error LIKWID_PATH='$(LIKWID_PATH)' has no lib/ or lib64/ directory)
  endif
  LFLAGS += -DLIKWID_PERFMON -I$(LIKWID_PATH)/include
  LLIBS  += -L$(LIKWID_LIBDIR) -llikwid -Wl,-rpath,$(LIKWID_LIBDIR)
  $(info ladder: LIKWID markers <- $(LIKWID_LIBDIR))
endif

BSUF = $(shell echo "$(strip $(TOOLCHAIN))_LADDER" | tr '[:upper:]' '[:lower:]')
EXE1 = gemm_bench_$(BSUF)
EXE2 = gemm_sweep_$(BSUF)

all: $(EXE1) $(EXE2) $(EXE1).sh

obj:
	mkdir -p obj

# Rebuild whenever the effective compile flags change (DATATYPE, LEVELS,
# library paths, ...) — not just when sources change.  kbuild-style stamp:
# GEMM_T must be identical across all translation units!
obj/.flags: FORCE
	@mkdir -p obj
	@echo '$(LFLAGS)' | cmp -s - $@ || echo '$(LFLAGS)' > $@
FORCE:

obj/%.o: src/%.c obj/.flags $(MAKEFILE_LIST) | obj
	$(CC) $(LFLAGS) -c $< -o $@

# The BF16 kernel is the only TU that sees -mavx512bf16; every other object
# keeps its exact codegen (target-specific variables apply through the
# pattern rule above).
ifneq ($(BF16_PROBE),)
obj/bf16.o: LFLAGS += $(BF16_FLAGS)
endif

$(EXE1): obj/ladder_bench.o $(LADDER_COMMON)
	$(LD) $(LFLAGS) $^ -o $@ $(LLIBS)

$(EXE2): obj/ladder_sweep.o $(LADDER_COMMON)
	$(LD) $(LFLAGS) $^ -o $@ $(LLIBS)

$(EXE1).sh: mk/run_template.sh
	sed -e "s#ENV_LOAD_MODULES=.*#ENV_LOAD_MODULES='$(ENV_MODULES)'#g" \
	    -e "s#ENV_PRERUN_CMD=.*#ENV_PRERUN_CMD='$(ENV_PRERUN_CMD)'#g" \
	    -e "s#ENV_POSTRUN_CMD=.*#ENV_POSTRUN_CMD='$(ENV_POSTRUN_CMD)'#g" \
	    -e "s#ENV_WRAP_CMD=.*#ENV_WRAP_CMD='$(ENV_WRAP_CMD)'#g" \
	    -e "s#ENV_VERBOSE=.*#ENV_VERBOSE='$(RUN_VERBOSE)'#g" \
	    -e "s#ENV_ENV_VARS=.*#ENV_ENV_VARS='$(ENV_VARIABLES)'#g" \
	    -e "s#ENV_BIN_SUFFIX=.*#ENV_BIN_SUFFIX='$(BSUF)'#g" \
	    -e "s#dgemm_bench_#gemm_bench_#g" \
	    -e "s#\.exe # #" \
	    mk/run_template.sh > $@
	chmod +x $@

clean:
	rm -f $(EXE1) $(EXE2) $(EXE1).sh
	rm -rf obj

.PHONY: all clean

# ---------------------------------------------------------------------------
# Legacy builds (kept verbatim from the previous Makefile)
# ---------------------------------------------------------------------------
else

# Legacy flavors link one BLAS in at build time; the compiler and the library
# fragments live in mk/include_<TOOLCHAIN>.mk and mk/include_<OPTIMIZATION>.mk.
$(if $(wildcard mk/include_$(OPTIMIZATION).mk),,\
  $(if $(filter NAIVE,$(OPTIMIZATION)),,\
    $(error OPTIMIZATION='$(OPTIMIZATION)' has no mk/include_$(OPTIMIZATION).mk — legacy flavors are NAIVE MKL AOCL OPENBLAS CBLAS (or TOOLCHAIN+LIBRARY), and the library rungs of the LADDER build are selected with BLAS_LIBS)))

include mk/include_$(TOOLCHAIN).mk
ifneq ($(strip $(OPTIMIZATION)),NAIVE)
include mk/include_$(OPTIMIZATION).mk
endif

CC = $(ENV_CC)
CFLAGS = $(ENV_CFLAGS) $(ENV_OMP_FLAG) $(ENV_DEFINES) $(ENV_INCDIRS) $(ENV_LIBDIRS)
LIBS = $(ENV_LIBS)

ifeq ($(strip $(OPTIMIZATION)),NAIVE)
  # If OPTIMIZATION is NAIVE, just use the lowercased toolchain
  ENV_BIN_SUFFIX := $(shell echo "$(strip $(TOOLCHAIN))" | tr '[:upper:]' '[:lower:]')
else
  # For all other flags, use toolchain_optimization
  RAW_SUFFIX := $(strip $(TOOLCHAIN))_$(strip $(OPTIMIZATION))
  ENV_BIN_SUFFIX := $(shell echo "$(RAW_SUFFIX)" | tr '[:upper:]' '[:lower:]')
endif

ifeq ($(BUILD_VERBOSE),0)
Q = @
endif
ifeq ($(PRECISION), DP)
  CFLAGS += -DDOUBLE
endif

MAIN_SRCS = src/gemm_bench.c src/gemm_sweep.c
LADDER_SRCS = src/runner.c src/levels.c src/library.c src/numa.c \
              src/ladder_bench.c src/ladder_sweep.c src/profiler.c src/bf16.c

# 3. Safely collect ALL OTHER .c files using wildcard and filter-out
ALL_SRCS = $(wildcard src/*.c)
COMMON_SRCS = $(filter-out $(MAIN_SRCS) $(LADDER_SRCS), $(ALL_SRCS))

# 4. Generate object file lists
COMMON_OBJS = $(COMMON_SRCS:.c=.o)
OBJ1 = src/gemm_bench.o
OBJ2 = src/gemm_sweep.o
HEADERS = $(wildcard src/*.h)

all: gemm_bench_$(ENV_BIN_SUFFIX) gemm_bench_$(ENV_BIN_SUFFIX).sh gemm_sweep_$(ENV_BIN_SUFFIX) check

check:
	$(Q)echo $(ENV_BIN_SUFFIX)

src/%.o: src/%.c $(HEADERS)
	$(Q)$(CC) $(CFLAGS) -c $< -o $@

gemm_bench_$(ENV_BIN_SUFFIX): $(OBJ1) $(COMMON_OBJS)
	$(Q)mk/test_modules.sh $(ENV_MODULES)
	$(Q)$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
gemm_sweep_$(ENV_BIN_SUFFIX): $(OBJ2) $(COMMON_OBJS)
	$(Q)mk/test_modules.sh $(ENV_MODULES)
	$(Q)$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
gemm_bench_$(ENV_BIN_SUFFIX).sh: mk/run_template.sh
	$(Q)sed -e "s#ENV_LOAD_MODULES=.*#ENV_LOAD_MODULES=\'$(ENV_MODULES)\'#g" \
	    -e "s#ENV_PRERUN_CMD=.*#ENV_PRERUN_CMD=\'$(ENV_PRERUN_CMD)\'#g" \
	    -e "s#ENV_POSTRUN_CMD=.*#ENV_POSTRUN_CMD=\'$(ENV_POSTRUN_CMD)\'#g" \
	    -e "s#ENV_WRAP_CMD=.*#ENV_WRAP_CMD=\'$(ENV_WRAP_CMD)\'#g" \
	    -e "s#ENV_VERBOSE=.*#ENV_VERBOSE=\'$(RUN_VERBOSE)\'#g" \
	    -e "s#ENV_ENV_VARS=.*#ENV_ENV_VARS=\'$(ENV_VARIABLES)\'#g" \
	    -e "s#ENV_BIN_SUFFIX=.*#ENV_BIN_SUFFIX=\'$(ENV_BIN_SUFFIX)\'#g" mk/run_template.sh > $@
	$(Q)chmod +x $@

test_modules:


.PHONY: clean distclean
clean:
	$(Q)$(RM) gemm_bench_$(ENV_BIN_SUFFIX)
	$(Q)$(RM) gemm_sweep_$(ENV_BIN_SUFFIX)
	$(Q)$(RM) gemm_bench_$(ENV_BIN_SUFFIX).sh

distclean:
	$(Q)$(RM) gemm_bench_*.exe
	$(Q)$(RM) gemm_bench_*.sh

endif
