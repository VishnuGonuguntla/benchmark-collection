# CuBLASS DGEMM benchmark

A simple and repeatable benchmark for validating the GPU performance based on cublas matrix multiplication.

## Contents:

**gemm**: Performs DGEMM for a specific matrix size.

**gemm_sweep**: Performs DGEMM over a range of problem size. Range starts from matrix size of 2096 to 90000.

* The code does `C=alpha*A*B+beta*C` with square matrices A, B and C and repeats 500 times (adjustable to test longer for more stable result). 

* The sizes of A,B and C are upto (90000,90000) in default test (also adjustable to fit your GPU memory size).

## Prerequisites:

1. **NVHPC SDK** (With **nvcc** compiler and to link **NVML**)
2. **CUDA Toolkit** (to link **CuBLAS**)

## Compiling specific to architecture:

There are different files for different GPU in **mk/** directory. One of these will be used by **config.mk**, where GPU = option is specified.

e.g. If **config.mk** has 

```
GPU = A100
```

then the Makefile will use A100 file under **mk/** directory. It was structured such way to keep architecture specific compile commands separate. 

You can create your own files under **mk/** directory and use then via **config.mk**.

When you want to compile your code, simply use:

```
make
```

To clean the builds, use:

```
make clean
```

## Options:

1. OPTIONS +=  -DNVML

Enables compiling this code with NVMK support to measure power, frequency and temperature.

This option is enabled, then it also measures POWER, FREQUENCY and TEMPERATURE over all the iterations of GEMM. It gives you average value over the runtime of the DGEMM for the given matrix size.

e.g. 
`
size 26368 | Time: 610.108 s | Perf: 18.0292 TFlop/s | Avg Power: 396.723 W | Avg Clock: 1377.75 MHz | Avg Temp: 76.9333 C
`

Without this option, it gives output like this:
`
size 26368 | Time: 610.108 s | Perf: 18.0292 TFlop/s
`

## See also

Heavily inspired by: https://github.com/hma02/cublasgemm-benchmark/tree/master

* [BLAS Interface for Different Precision](http://www.netlib.org/utk/people/JackDongarra/WEB-PAGES/Batched-BLAS-2016/Day1/precision-blas.pdf)
* [OpenAI gemm](https://github.com/openai/openai-gemm)
* [Useful nvidia-smi Queries](http://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries)
