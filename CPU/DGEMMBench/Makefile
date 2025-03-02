
include config.mk

CC = $(ENV_CC)
CFLAGS = $(ENV_CFLAGS) $(ENV_OMP_FLAG) $(ENV_DEFINES) $(ENV_INCDIRS) $(ENV_LIBDIRS)
LIBS = $(ENV_LIBS)
ENV_BIN_SUFFIX = $(shell echo "$(ENV)" | tr '[:upper:]' '[:lower:]')
ifeq ($(BUILD_VERBOSE),0)
Q = @
endif

all: dgemm_bench_$(ENV_BIN_SUFFIX) dgemm_bench_$(ENV_BIN_SUFFIX).sh

dgemm_bench_$(ENV_BIN_SUFFIX): src/dgemm_bench.c
	$(Q)make/test_modules.sh $(ENV_MODULES)
	$(Q)$(CC) $(CFLAGS) src/dgemm_bench.c -o $@ $(LIBS)

dgemm_bench_$(ENV_BIN_SUFFIX).sh: make/run_template.sh
	$(Q)sed -e "s#ENV_LOAD_MODULES=.*#ENV_LOAD_MODULES=\'$(ENV_MODULES)\'#g" \
	    -e "s#ENV_PRERUN_CMD=.*#ENV_PRERUN_CMD=\'$(ENV_PRERUN_CMD)\'#g" \
	    -e "s#ENV_POSTRUN_CMD=.*#ENV_POSTRUN_CMD=\'$(ENV_POSTRUN_CMD)\'#g" \
	    -e "s#ENV_WRAP_CMD=.*#ENV_WRAP_CMD=\'$(ENV_WRAP_CMD)\'#g" \
	    -e "s#ENV_VERBOSE=.*#ENV_VERBOSE=\'$(RUN_VERBOSE)\'#g" \
	    -e "s#ENV_ENV_VARS=.*#ENV_ENV_VARS=\'$(ENV_VARIABLES)\'#g" \
	    -e "s#ENV_BIN_SUFFIX=.*#ENV_BIN_SUFFIX=\'$(ENV_BIN_SUFFIX)\'#g" make/run_template.sh > $@
	$(Q)chmod +x $@

test_modules:


.PHONY: clean distclean
clean:
	$(Q)$(RM) dgemm_bench_$(ENV_BIN_SUFFIX)
	$(Q)$(RM) dgemm_bench_$(ENV_BIN_SUFFIX).sh

distclean:
	$(Q)$(RM) dgemm_bench_*.exe
	$(Q)$(RM) dgemm_bench_*.sh
