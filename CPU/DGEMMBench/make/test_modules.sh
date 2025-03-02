#!/bin/bash -l

MODIN=$(echo "$@" | xargs)

for MOD in ${MODIN}; do
    MODOUT=$(module list "${MOD}")

    if [ $(echo "${MOD}" | grep "${MOD}" | wc -l) != "1" ]; then
        echo "Module ${MOD} not loaded. Please load for building."
        exit 1
    fi
done
