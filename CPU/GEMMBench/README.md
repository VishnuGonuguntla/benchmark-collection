# CPU GEMM benchmark collection

GEMM benchmarks for CPUs with two build flavors:

* **Ladder** (new, default) — one binary runs a six-rung optimization ladder
  from a naive threaded `ijk` loop up to a NUMA-aware 2D-tiled micro-kernel,
  and compares every rung against vendor BLAS libraries — **Intel MKL**,
  **AMD AOCL/BLIS** and **OpenBLAS** — which are loaded at runtime from paths
  configured in the makefile.  The ladder's optimization paradigm is
  **memory localization** (loop order, cache blocking, register tiling)
  followed by **ccNUMA domain localization** (first-touch placement,
  socket-local replication, NUMA-aware compute scheduling).
* **Legacy** (`OPTIMIZATION=NAIVE|MKL|AOCL|OPENBLAS|CBLAS`) — the original
  single-variant binaries with link-time BLAS selection, kept for reproducing
  past measurements.

The rectangular problem `C[M,N] = alpha·A[M,K]·B[K,N] + beta·C[M,N]` is fully
supported (`-m/-n/-k`), matching the interface of the sibling
`GPU/GEMM` ladder benchmark.

## Layout

| File | Description |
|------|-------------|
| `src/levels.c/.h` | The ladder rungs V0–V5 + level/tile registry |
| `src/numa.c/.h` | `/sys` NUMA topology, thread pinning, first-touch staging, `/proc/self/numa_maps` footprint report |
| `src/library.c/.h` | Vendor-BLAS registry: runtime `dlopen` loading of MKL / AOCL-BLIS / OpenBLAS |
| `src/runner.c/.h` | CLI, timing loops, `-T` autotune, `-v` verification, table/CSV output |
| `src/ladder_bench.c` | Ladder binary, fixed shape |
| `src/ladder_sweep.c` | Ladder binary, proportionally grown shape sweep |
| `src/gemm_bench.c`, `src/gemm_sweep.c`, `src/cli.c`, `src/util.h` | Legacy sources (link-time BLAS) |
| `config.mk` / `Makefile` / `mk/` | Build configuration, per toolchain/library |

## The ladder

| Rung | Adds | Mechanism |
|------|------|-----------|
| **V0** naive-omp `ijk` | — baseline | row band per thread; strided B-column reads |
| **V1** `ikj` row locality | loop reordering | every inner update is a contiguous vector FMA; **no blocking**, so it collapses once B exceeds cache |
| **V2** blocked micro-kernel | cache + register blocking | MC×NC×KC blocking (runtime-tuned), compile-time 4×N register tile, packed A panel, `#pragma omp simd` |
| **V3** + NUMA first-touch | *data* localization | threads pinned round-robin across nodes before pages are touched, so each page lands on its computing node |
| **V4** + B replication | *domain* localization | contiguous per-node thread bands, row bands aligned to sockets, and a **socket-local copy of B** → no remote reads in the inner loop |
| **V5** + 2D tile grid | compute scheduling | per-node P×Q row/column tile grid → A/B panel reuse in LLC, C tiles remain disjoint (no cross-node reduction) |
| **MKL** | vendor reference | `cblas_dgemm`/`sgemm` via `dlopen` |
| **AOCL/BLIS** | vendor reference | same, from `libblis*.so` via `dlopen` |
| **OpenBLAS** | vendor reference | same, from `libopenblas*.so` via `dlopen` |

The three library rungs are not code paths: they are rows of the registry in
`src/library.c`, built in only when `BLAS_LIBS` selects them (see below).

V3–V5 run the V2 kernel — each rung isolates exactly one placement policy
decision, and the `-R` report shows the `/proc/self/numa_maps` page
footprint as evidence of the localization working.

## Build (`config.mk` / make variables)

```makefile
TOOLCHAIN    = GCC        # GCC | ICX | ICC | AOCC
OPTIMIZATION = LADDER     # LADDER (new) or legacy: NAIVE/MKL/AOCL/OPENBLAS/CBLAS
DATATYPE     = FP64       # FP64 | FP32 (legacy PRECISION=DP/SP also honored)
LEVELS       = all        # compile-time subset: 0..5, mkl, aocl, openblas
BLAS_LIBS    = auto       # which vendor libraries get a rung (see below)
MKL_PATH       = /path/to/mkl        # root or full libmkl_rt.so path
AOCL_PATH      = /path/to/aocl       # root or full libblis*.so path
OPENBLAS_PATH  = /path/to/openblas   # root or full libopenblas*.so path
LADDER_ARCH ?= -march=x86-64-v3      # override to -march=native on the target machine
```

```bash
# GCC, both library rungs configured
make TOOLCHAIN=GCC MKL_PATH=/apps/SPACK/.../intel-oneapi-mkl-2024.2.2-... \
                   AOCL_PATH=/opt/amd/aocl/5.0

make DATATYPE=FP32 LEVELS="2,3,4,5,openblas"  # subset build
make TOOLCHAIN=ICX                            # Intel oneAPI compiler
make TOOLCHAIN=AOCC BLAS_LIBS=aocl            # AMD: AOCL-BLIS rung only
make BLAS_LIBS=openblas OPENBLAS_PATH=/apps/spack/.../openblas-0.3.29
make TOOLCHAIN=GCC LADDER_ARCH=-march=native  # unlock the wide register tile
```

## Vendor library rungs (`BLAS_LIBS`)

The libraries are **dlopen'd** — MKL, AOCL-BLIS and OpenBLAS all export
`cblas_dgemm`, so a binary that *linked* more than one of them would collide;
instead the makefile resolves the paths, bakes the absolute sonames into the
binary and the rungs load them at run time.  Nothing is linked against BLAS,
only `-ldl -lm -fopenmp`, so one binary can carry every rung at once.

| | Intel CPUs | AMD CPUs |
|---|---|---|
| **MKL** (oneAPI) | tuned (AVX-512/AMX kernels) | runs, but not tuned |
| **AOCL/BLIS** | not supported | tuned (Zen/AVX-512 kernels) |
| **OpenBLAS** | runs (DYNAMIC_ARCH) | runs (DYNAMIC_ARCH) |

`BLAS_LIBS = auto` (the default) therefore asks for `mkl openblas` on an Intel
target and `aocl openblas` on an AMD one: OpenBLAS is the common rung on every
CPU, the uarch-tuned library is added on top.  The target is read from the
toolchain when it names one (`TOOLCHAIN=AOCC` → AMD, `ICC`/`ICX` → Intel) and
otherwise from `/proc/cpuinfo` of the build host — pass `BLAS_LIBS=...`
explicitly when cross-building for a different vendor.  Other values: `none`
(hand-written ladder only), `all`, or any list of the tags `mkl aocl openblas`.

Every listed library must resolve, or the build stops with a message; under
`auto` a library without a `*_PATH` is dropped with a note.  A path may be an
installation root (its `lib*/` subdirectories are searched for the soname) or
the full `.so` path.  A rung whose object cannot be loaded at run time is not
fatal: the ladder prints `NOT AVAILABLE` with the `dlerror()` text and carries
on.  `LEVELS` also accepts the library tags, so `LEVELS="2,3,4,5,openblas"`
builds a trimmed binary that still keeps the OpenBLAS reference.

The rungs are data, not branches: `src/library.c`'s `BLAS_REGISTRY` holds one
row per library (tag, labels, path variable, gemm symbol, thread-count setter,
optional build-info symbols) and the Makefile holds the matching stanza (soname
patterns, level bit, macro names).  Adding a uarch-tuned library for a new CPU
means one row, one stanza, one rung id in `src/levels.h` — the CLI (`-l`), the
comparison table, the `-v` reference choice and the build messages all follow
the registry.  `-R` prints each loaded library's path and, where the library
reports it, its build string (OpenBLAS: detected core + kernel config).

* Objects track the effective compile flags (kbuild-style stamp), so
  switching `DATATYPE`/`LEVELS`/`BLAS_LIBS`/paths always triggers a correct
  rebuild.
* The register tile (4×8 fp64, 4×16 fp32; doubled to 4×16 / 4×32 when the
  build enables AVX-512) is **compile-time** — like the GPU ladder's
  register tiles — while MC/NC/KC blocking is tunable at runtime.

## Run

Thread count comes from the environment (Slurm / OpenMP), never from the
binary:

```bash
srun --cpus-per-task=64 ./gemm_bench_gcc_ladder -s 4096 -r 5 -v

# or via the generated module wrapper
./gemm_bench_gcc_ladder.sh -m 8192 -n 8192 -k 4096 -t 1.0 -T tune
```

| Flag | Description |
|------|-------------|
| `-m/-n/-k <int>` | rectangular shape (cols A = rows B = K) |
| `-s <int>` | cube shorthand `M=N=K`; in sweep mode: size cap |
| `-r <int>` / `-t <min>` | repeats per rung / minutes per rung (0 = use -r) |
| `-a/-b <val>` | alpha / beta (defaults 1.0 / 0.0) |
| `-i random\|constant` | data initialization |
| `-l <list>` | runtime subset of compiled rungs: `0..5, mkl, aocl, openblas, all` |
| `-T <MC,NC,KC>\|tune\|help` | blocking parameters, per-rung autotune, or list options |
| `-v` | verify every rung against the reference (AOCL → MKL → OpenBLAS → V2) |
| `-R` | NUMA topology, loaded library paths/builds, per-rung page footprint |
| `-c <file>` | append CSV rows (`level,M,N,K,time_s,repeats,gflops`) |

`gemm_sweep_gcc_ladder` takes a base shape (`-m/-n/-k` or `-s` cube) and
grows all three dimensions ×1.2 per step until the smallest exceeds the
`-s` cap, running the full ladder at each size.

## Output

Actual run on a 4-core, single-NUMA-node of AMD EPYC 9355P, built
with the MKL and OpenBLAS rungs (`make DATATYPE=FP64 BLAS_LIBS=mkl,openblas …`,
then `./gemm_bench_gcc_ladder -m 2048 -n 1024 -k 512 -r 2`):

```
GEMM optimization ladder (FP64)  |  4 thread(s)
shape: C[2048x1024] = A[2048x512] x B[512x1024]   alpha=1 beta=0
---------------------------------------------------------------------------
+----------------------------------+---------+---------+----------+---------+--------+-----------+
| Level                            | Time(s) | Reps    |  GFlop/s | x V0    | %MKL   | %OpenBLAS |
+----------------------------------+---------+---------+----------+---------+--------+-----------+
| V0 naive-omp ijk                 | 1.707   | 2       |     2.52 |    1.00 |   2.9% |      2.6% |
| V1 ikj row locality              | 0.084   | 2       |    51.06 |   20.29 |  58.5% |     52.0% |
| V2 blocked micro-kernel          | 0.070   | 2       |    61.36 |   24.38 |  70.3% |     62.5% |
| V3 + NUMA first-touch            | 0.071   | 2       |    60.18 |   23.91 |  69.0% |     61.3% |
| V4 + B replication               | 0.080   | 2       |    54.01 |   21.46 |  61.9% |     55.1% |
| V5 + 2D tile grid                | 0.081   | 2       |    52.95 |   21.04 |  60.7% |     54.0% |
| MKL (cblas_dgemm)                | 0.049   | 2       |    87.26 |   34.67 | 100.0% |     88.9% |
| OpenBLAS (cblas_dgemm)           | 0.044   | 2       |    98.11 |   38.98 | 112.4% |    100.0% |
+----------------------------------+---------+---------+----------+---------+--------+-----------+
```

The comparison columns are generated from the rungs that actually ran: on an
Intel target you get `%MKL`, on an AMD one `%AOCL`, and a library that was not
configured simply contributes no column (no empty placeholder rows).

(V2 leading V1 here is the expected behavior once B no longer fits in cache —
see the crossover table below.  On a multi-socket machine the interesting part
is the V2/V3/V4/V5 relationship at problem sizes exceeding one node's
cache/memory.)

With `-v` every rung prints `PASS/FAIL (max rel err …)` against the reference
before timing; with `-R` each rung also prints the per-NUMA-node page split of
its A/B/C buffers, and every library rung prints the object it loaded — plus
the library's own build string where it exposes one:

```
    library: /apps/.../lib/libopenblasp-r0.3.29.so
    build:   Cooperlake | OpenBLAS 0.3.29 DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=128
```

which is exactly how you notice a portable library that picked the wrong
kernel set for the machine you happen to be sitting on.

## What the numbers demonstrate (and where)

Measured on a 4-core (single-NUMA-node) login slice of an EPYC 9355P, FP64:

| | V1 `ikj` | V2 blocked |
|---|---|---|
| 1024³ (B fits L3) | **98 GF/s** | 76 GF/s |
| 4096³ (B = 128 MB ≫ L3) | 35 GF/s | **70 GF/s** |

That crossover *is* the cache-blocking lesson: V1's unblocked row streaming
wins inside cache and falls off a cliff beyond it. On a real ccNUMA
machine the V2→V3→V4→V5 sequence likewise shows up as first-touch gains at
large sizes (B/C streaming over the inter-socket link), the
`-R` footprint lines print split percentages, and the libraries pull away
further (MKL/BLIS use hand-tuned assembly micro-kernels — matching them is
not the ladder's goal; it shows the trajectory toward them).

## Legacy builds

`make TOOLCHAIN=<tc> OPTIMIZATION=<opt>` still produces the original
`gemm_bench_<tc>_<opt>` / `gemm_sweep_<tc>_<opt>` binaries, where the BLAS is
chosen at **link** time by `mk/include_<opt>.mk` (`MKL`, `AOCL`, `OPENBLAS`,
`CBLAS`) instead of being loaded at run time.  `OPTIMIZATION=NAIVE` builds the
hand-written loop only.  Those include fragments now take their location from
the same `MKL_PATH` / `AOCL_PATH` / `OPENBLAS_PATH` variables as the ladder
(exported module variables still win), and the sources share one
`DGEMM_BENCH_WITH_BLAS` switch in `src/util.h` instead of repeating the
per-library `#if` ladder.  Note the legacy sources are square-matrix only
(`-n`); the rectangular interface is the ladder's.  Two deliberate fixes:
`PRECISION=DP` previously had no effect (the flag was appended to an unused
variable); it now correctly defines `DOUBLE`, and the library call previously
switched on `#ifdef double` (a keyword, never defined), so a `DP` build
silently ran the *single* precision entry point.

## See also

* Sibling GPU project: `GPU/GEMM` (same ladder philosophy on CUDA/HIP)
* [DGEMM on 64 cores](https://cea-hpc.github.io/dgemm/) — the classic ladder write-up
* [BLIS framework docs](https://github.com/flame/blis) — the micro-kernel model
* [OpenBLAS](https://www.openblas.net/) — portable, per-uarch kernel sets
* AMD AOCL-BLIS installation guide, Intel MKL `cblas` reference
