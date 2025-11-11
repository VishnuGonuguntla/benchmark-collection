# NHR benchmarking suite

A collection of handpicked benchmarks aimed at continuous regression stress testing. Each benchmark is associated with a component within 1 HPC node. The goal is to apply as much stress possible as one can on the associated component.

## Components within a HPC node:
1. CPU
2. GPU
3. Memory/DRAM
4. GPU DRAM
5. Ethernet adapter or other communication fabric
6. Local disk

## Benchmarks and the associated components (aimed at):

1. Vendor optimized HPL -> CPU (Peak Flop/s)
2. Vendor optimized DGEMM/SGEMM -> GPU (Peak Flop/s)
3. TheBandwidthBenchmark -> CPU and GPU Memory/DRAM (Peak Memory Bandwidth)
4. IMB/OSU -> Ethernet/Communication Fabric including GPU-GPU communication (Peak Communication Bandwidth)
5. ior + mdtest -> Local disk (Peak Disk Bandwidth)

## Cloning the collection

In order to clone all the benchmarks within this repo:

```
git clone --recursive https://github.com/RRZE-HPC/benchmark-collection.git 
```
