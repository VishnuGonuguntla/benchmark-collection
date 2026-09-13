#
# Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
# All rights reserved.
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE file.
#

include config.mk

CPU_dir = CPU
GPU_dir = GPU

IMB_dir = $(CPU_dir)/IMB
DGEMM_dir = $(CPU_dir)/DGEMMBench
TheBandwidthBenchmark_dir = $(CPU_dir)/TheBandwidthBenchmark
CpuHPL_dir = $(INTEL_MKL_PATH)/benchmarks/linpack
CpuHPCG_dir = $(INTEL_MKL_PATH)/benchmarks/hpcg
NvidiaBench_dir = $(GPU_dir)/NvidiaBench
GpuHPL_dir = $(NvidiaBench_dir)/hpl-linux-x86_64
GpuHPCG_dir = $(NvidiaBench_dir)/hpcg-linux-x86_64
GpuStream_dir = $(GPU_dir)/GpuBenches/gpu-stream

all: IMB DGEMMBench TheBandwidthBenchmark CpuHPL CpuHPCG GpuStream GpuHPL GpuHPCG

IMB:
	make -C $(IMB_dir) -f Makefile IMB-MPI1

DGEMM:
	make -C $(DGEMM_dir) -f Makefile

TheBandwidthBenchmark:
	make -C $(TheBandwidthBenchmark_dir) -f Makefile

GpuStream:
	make -C $(GpuStream_dir) -f Makefile

NvidiaBench:
	@wget -P $(GPU_dir) https://developer.download.nvidia.com/compute/nvidia-hpc-benchmarks/redist/nvidia_hpc_benchmarks_openmpi/linux-x86_64/nvidia_hpc_benchmarks_openmpi-linux-x86_64-24.09.02-archive.tar.xz
	@tar xvf $(GPU_dir)/nvidia_hpc_benchmarks_openmpi-linux-x86_64-24.09.02-archive.tar.xz -C $(GPU_dir)/.

clean:
	make -C $(IMB_dir) -f Makefile clean
	make -C $(DGEMM_dir) -f Makefile clean
	make -C $(TheBandwidthBenchmark_dir) -f Makefile clean
	make -C $(GpuStream_dir) -f Makefile clean

distclean:
	make -C $(IMB_dir) -f Makefile distclean
	make -C $(DGEMM_dir) -f Makefile distclean
	make -C $(TheBandwidthBenchmark_dir) -f Makefile distclean
	make -C $(GpuStream_dir) -f Makefile distclean