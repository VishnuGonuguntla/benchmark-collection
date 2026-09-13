
ENV ?= ICX+MKL

BUILD_VERBOSE ?= 1
RUN_VERBOSE ?= 1

PRECISION ?= DP

MKLROOT=/apps/spack/1.0.2/opt/linux-almalinux9-icelake/none-none/intel-oneapi-mkl-2024.2.2-qvjxvr2b5cebxqymrzkfopa2tr6c2g4m/mkl/2024.2
include make/include_$(ENV).mk
