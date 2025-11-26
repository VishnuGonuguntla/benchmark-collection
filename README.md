# NHR benchmarking suite

A collection of handpicked benchmarks aimed at continuous regression stress testing. Each benchmark is associated with a component within 1 HPC node. The goal is to apply as much stress possible as one can on the associated component.

These benchmarks are currently being used with [ClusterBench](https://github.com/ClusterCockpit/cc-clusterbench) for stress testing all the components within a HPC node.

## Components within a HPC node:
1. CPU
2. GPU
3. Memory/DRAM
4. GPU DRAM
5. Ethernet adapter or other communication fabric
6. Local disk

## Benchmarks and the associated components (aimed at):

| Benchmark / Tool                 | Target Component(s)                             | Purpose / Metric                         |
| -------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| Vendor-optimized **HPL**         | CPU                                             | Peak floating-point performance (FLOP/s) |
| Vendor-optimized **DGEMM/SGEMM** | GPU                                             | Peak GPU floating-point performance      |
| **TheBandwidthBenchmark**        | CPU & GPU Memory / DRAM                         | Peak memory bandwidth                    |
| **IMB / OSU**                    | Ethernet / Communication fabric (incl. GPU–GPU) | Peak communication bandwidth             |
| **ior** + **mdtest**             | Local disk                                      | Peak disk I/O bandwidth + metadata rate  |


## Cloning the collection

In order to clone all the benchmarks within this repo:

```
git clone --recursive https://github.com/RRZE-HPC/benchmark-collection.git 
```

## License

MIT License

Copyright (c) 2025 RRZE-HPC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
