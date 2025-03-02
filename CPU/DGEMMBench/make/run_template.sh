#!/bin/bash -l


####
ENV_LOAD_MODULES=
ENV_BIN_SUFFIX=
ENV_ENV_VARS=
ENV_WRAP_CMD=
ENV_PRERUN_CMD=
ENV_POSTRUN_CMD=
ENV_VERBOSE=
####

VERBOSE=1
if [ "x${ENV_VERBOSE}" != "x1" ]; then
    VERBOSE=0
fi

module purge
module load ${ENV_LOAD_MODULES}
if [ "${VERBOSE}" == "1" ]; then
    module list
fi

if [ "${VERBOSE}" == "1" ]; then
    if [ ! -z ${ENV_ENV_VARS} ]; then
        echo "Environment:"
    fi
fi
for VAR in $(echo "${ENV_ENV_VARS}" | xargs); do
    if [ "${VERBOSE}" == "1" ]; then
        echo "\t${VAR}"
    fi
    export ${VAR}
done

if [ "${VERBOSE}" == "1" ]; then
    echo "Exec pre-run: ${ENV_PRERUN_CMD}"
fi
${ENV_PRERUN_CMD}

if [ "${VERBOSE}" == "1" ]; then
    echo "Exec : ${ENV_WRAP_CMD} ./dgemm_bench_${ENV_BIN_SUFFIX}.exe $@"
fi
${ENV_WRAP_CMD} ./dgemm_bench_${ENV_BIN_SUFFIX}.exe $@


if [ "${VERBOSE}" == "1" ]; then
    echo "Exec post-run: ${ENV_POSTRUN_CMD}"
fi
${ENV_POSTRUN_CMD}

