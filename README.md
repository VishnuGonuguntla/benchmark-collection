# NHR@FAU benchmarking suite

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

## GEMM cache-locality sweep pipeline

A scripted end-to-end workflow: **run → parse → visualize**, covering the CPU
(`CPU/GEMMBench`) and GPU (`GPU/GEMM`) optimization-ladder benchmarks with
optional LIKWID hardware counters (SP/DP FLOPs, energy, memory traffic).

### 1. LIKWID perf groups (one-time setup)

The measurements use `CUSTOM` performance groups that count single-precision,
double-precision, RAPL energy and DRAM traffic in one pass. LIKWID finds them
in **`$HOME/.likwid/groups/<arch>/`**:

| Architecture | Group files | Note |
|---|---|---|
| GNR / SPR / ICX | `CUSTOM.txt` | SP+DP in one pass (uses all 8 core counters) |
| GNR / SPR | `CUSTOM_BF16.txt` | energy + memory context only: LIKWID 5.5.2 has no counter that fires for `VDPBF16PS` on these parts (verified on GNR); BF16 rate comes from the app CSV |
| zen3 / zen4 | `CUSTOM_SP.txt`, `CUSTOM_DP.txt` | HW has no SP/DP split — each group reports the combined FLOP event under its precision label |
| zen5 / zen5c | `CUSTOM_SP.txt`, `CUSTOM_DP.txt` | full scalar+packed per precision; 6 counters force the two-pass split |
| zen5 / zen5c | `CUSTOM_BF16.txt` | dedicated hardware BF16 event `RETIRED_SSE_AVX_FLOPS_ALL_BF16` + energy + memory |

If you keep copies in the repo instead, symlink them: `ln -s <repo>/.likwid $HOME/.likwid`.
Requirements on the measured node: counter access (`hwperf`-style partition or
LIKWID access daemon) and `kernel.nmi_watchdog=0` for the 8-counter Intel group.

BF16 is measured by a dedicated kernel (`CPU/GEMMBench/src/bf16_bench.c`,
AVX512 `VDPBF16PS` outer product, OpenMP) that emits the same CSV schema as
the ladder; it is auto-skipped (exit 3) on CPUs without AVX512_BF16
(Ice Lake, Zen4 and older). GPU BF16 is not covered by GPU/GEMM's datatypes
(FP64/FP32/FP16 only).

### 2. Running the sweep

On one node (auto-detects CPU arch and GPUs):

```bash
module load likwid                 # optional, enables USE_LIKWID
bash scripts/run_bench.sh
```

Knobs (environment variables):

```
USE_LIKWID=0|1        wrap each run in likwid-perfctr          (default 1)
SIZES="256 512 ..."   cube sizes spanning L1 -> L2 -> L3 -> DRAM
DATATYPES="FP32 FP64 BF16"
THREADS=<int>         OMP threads / CPUs measured (default: all in cpuset)
REPEATS=<int>         reps per rung             (default 3)
TOOLCHAIN=GCC         GEMMBench toolchain (GCC|ICX|AOCC|...)
LIKWID_GROUP=auto     override group name (e.g. CUSTOM_DP)
PIN=auto              processor pinning for non-likwid runs:
                      inside a SLURM allocation the app is launched as a
                      `srun --ntasks=1 --cpus-per-task=$THREADS
                       --cpu-bind=$CPU_BIND` step; otherwise likwid-pin is
                      used, then OMP_PROC_BIND; 0 disables
CPU_BIND=cores        srun cpu-bind mode (threads|cores|sockets|none)
PIN_FREQ=1            frequency pin passed to `srun --cpu-freq` (applied by
                      slurmd — no root needed):
                      <MHz>|<any srun spec> | 1|max -> high-high:performance
                      (governor 'performance', min=max=boost) | 0 (off).
                      Requires a SLURM allocation; warned + ignored outside.
SKIP_CPU=1 / SKIP_GPU=1
```

Inside a SLURM allocation every measured run (including LIKWID-wrapped
ones, so the frequency pin applies) is launched as an
`srun --ntasks=1 --cpus-per-task=$THREADS --cpu-bind=$CPU_BIND
--cpu-freq=$PIN_FREQ` step; `likwid-perfctr -C` additionally pins the
measured threads itself. Outside Slurm the fallback is `likwid-pin`, then
OpenMP affinity. The chosen pin mode and the resolved frequency spec are
recorded in `results/<arch>/meta.csv` (`pin_mode`, `cpu_bind`,
`cpu_freq_spec`).

For all architectures at once (one `--exclusive` SLURM job per node):

```bash
DRY_RUN=1 bash scripts/submit_all.sh      # preview the sbatch commands
bash scripts/submit_all.sh                # submit; edit the node map at the top
```

### 3. Output layout

```
results/<arch>[_<gpu>]/
  meta.csv                    node description incl. per-instance cache sizes
  gemm_sweep_FP32.csv         level,M,N,K,time_s,repeats,gflops    (one per size)
  gpu_gemm_FP32.csv           ...same + power_W,clock,temp,util (GPU METRICS build)
  likwid_FP32_CUSTOM_SP_512.csv   LIKWID -O metrics per size (when USE_LIKWID=1)
  raw/                        full logs (debug only; the CSVs are the record)
```

The benchmark stdout is intentionally minimal: a `#` header, a `size MxNxK`
marker per step, and one `V2 blocked micro-kernel  1014.12 GFlop/s` line per
rung. The big ASCII summary tables and the 100-column LIKWID event dumps are
suppressed in favour of the CSVs (the human tables still print when you run
without `-c`).

### 4. Parsing with pandas

```bash
python3 scripts/parse_likwid.py                    # reads results/, writes output/
python3 scripts/parse_likwid.py --legacy CPU/GEMMBench/check.txt   # old txt captures
```

Produces per processor: `output/<proc>/{gemm,likwid,likwid_perthread,summary}.csv`
with socket-aggregated metrics and a `gflops_per_joule` efficiency column.
Dependencies: `module load python` (pandas, numpy, matplotlib).

### 5. Visualizing

Open `scripts/visualize.ipynb` (Jupyter / VS Code notebook UI) and run all
cells. For each processor it saves PNGs into `output/<proc>/`:

- `gemm_ladder_<DT>.png` — GFlop/s vs matrix size per ladder rung, with
  shaded L1/L2/L3 bands: where the naive rungs collapse marks the cache exit
- `likwid_profile_<DT>.png` — SP/DP rates, energy, memory bandwidth vs size
- `threads_<DT>_<size>.png` — per-thread FLOP-rate violin (load balance)

and cross-processor bar charts (`peak GFlop/s`, `GFlop/s per Joule`,
`memory bandwidth`) into `output/comparison/`.

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
