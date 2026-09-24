include config.mk

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
  CC_FLAGS += -DDOUBLE
endif

MAIN_SRCS = src/gemm_bench.c src/gemm_sweep.c

# 3. Safely collect ALL OTHER .c files using wildcard and filter-out
ALL_SRCS = $(wildcard src/*.c)
COMMON_SRCS = $(filter-out $(MAIN_SRCS), $(ALL_SRCS))

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
