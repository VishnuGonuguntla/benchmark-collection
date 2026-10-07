# ---------------------------------------------------------------------------
# CPU GEMM benchmark configuration
# ---------------------------------------------------------------------------

# Compiler toolchain — selects mk/include_$(TOOLCHAIN).mk
#   GCC | ICX | ICC | AOCC
TOOLCHAIN ?= GCC

# ---------------------------------------------------------------------------
# Build flavor:
#   LADDER  — the new optimization-ladder binary: V0 naive-omp .. V5 2D NUMA
#             grid, plus runtime-loaded vendor BLAS rungs (MKL / AOCL-BLIS /
#             OpenBLAS, selected by BLAS_LIBS below).
#   NAIVE / MKL / AOCL / OPENBLAS / CBLAS
#           — legacy single-variant builds (link-time BLAS, kept for
#             reproducing past measurements).
# ---------------------------------------------------------------------------
OPTIMIZATION ?= LADDER

# ---------------------------------------------------------------------------
# Ladder options
#
# Datatype of the problem: FP64 | FP32.
# (Leave empty to fall back to the legacy PRECISION switch: DP->FP64,
#  SP->FP32.)
DATATYPE = FP64

# ---------------------------------------------------------------------------
# Which vendor BLAS libraries the ladder gets a rung for.  The libraries are
# dlopen'd at run time (nothing is linked), so several can coexist in one
# binary.  Each one still needs its *_PATH below to be resolvable.
#
#   auto        pick per CPU vendor of the build host:
#                 AuthenticAMD  -> aocl openblas   (AOCL-BLIS is AMD-tuned)
#                 GenuineIntel  -> mkl  openblas   (MKL is Intel-tuned)
#                 anything else ->        openblas (portable kernels, all CPUs)
#   none        hand-written ladder only (V0..V5)
#   all         every library whose path resolves
#   <list>      comma/space list of: mkl aocl openblas
#
# A library named here without a working *_PATH stops the build; under 'auto'
# it is dropped with a note.  Building for a machine other than this one?
# Pass BLAS_LIBS explicitly (e.g. BLAS_LIBS=aocl,openblas on EPYC).
# Adding a library: see the checklist in src/library.h.
BLAS_LIBS ?= auto

# Library locations: give either an installation root (the Makefile searches
# its lib*/ directories) or the full .so path.  The resolved absolute path is
# baked into the binary and loaded via dlopen — an unset path omits the rung.
#
# MKL:       <root>/lib/intel64/libmkl_rt.so
# AOCL-BLIS: <root>/lib/libblis-mt.so
# OpenBLAS:  <root>/lib/libopenblas*.so
#
MKL_PATH       = /apps/SPACK/0.23.1/opt/linux-ubuntu24.04-x86_64_v3/gcc-13.3.0/intel-oneapi-mkl-2024.2.2-ypxe56o6gvnp36cijjhjl2ajfusjriqt
AOCL_PATH      = /home/hpc/ihpc/ihpc171h/blas/blis/rt
OPENBLAS_PATH  = /home/hpc/ihpc/ihpc171h/blas/OpenBLAS/icx

# Compile-time level selection: 'all' or a comma/space list of
#   0 1 2 3 4 5 mkl aocl openblas
# Library tokens select the matching vendor rung, which must also be listed in
# BLAS_LIBS and have a *_PATH.  The runtime -l flag can only select from these.
LEVELS ?= all

# ---------------------------------------------------------------------------
# Legacy options (used by the non-LADDER builds)
# ---------------------------------------------------------------------------
BUILD_VERBOSE ?= 1
RUN_VERBOSE ?= 1

PRECISION ?= DP#SP/DP
