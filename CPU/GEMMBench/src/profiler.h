/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.  MIT style license, see TheBandwidthBenchmark.
 *
 * LIKWID marker integration for the GEMM ladder. */
#ifndef __PROFILER_H_
#define __PROFILER_H_

#include "runner.h"

// ---------------------------------------------------------------------------
// Hardware-counter instrumentation via the LIKWID MarkerAPI.
//
// Every ladder rung — the hand-written V0..V5 *and* the dlopen'd vendor
// libraries — is one marker region, tagged by rung_marker() below.  Built
// with LIKWID=1 (config.mk) and run as
//
//     likwid-perfctr -m -g <GROUP> ./gemm_bench_<tc>_ladder ...
//
// the application prints one performance-group table per rung.  One region
// call corresponds to one timed repetition (warmup, -v verification and
// -T tune calls are deliberately unmarked).
//
// Without -DLIKWID_PERFMON everything compiles to an empty function; with
// the perfmon build but outside likwid-perfctr the markers stay disabled
// (likwid_markerIsEnabled()) so no parallel regions are spawned.
// ---------------------------------------------------------------------------

// Region tag of one ladder level id: "V0".."V5" for the hand-written rungs,
// for the vendor rungs the tag from their BLAS_REGISTRY row, so a library
// added to the registry gets its marker region without touching this file.
const char *rung_marker(int id);

// LIKWID_MARKER_INIT plus a parallel region in which every thread performs
// LIKWID_MARKER_THREADINIT and registers all compiled rung tags.  Call once,
// as the very first thing in main after argument parsing: INIT must precede
// the first OpenMP region.
void ladder_markers_init(void);

// Start/stop one region call of `tag`.  These wrap a single rung dispatch:
// START/STOP execute on the full ambient OpenMP team (TheBandwidthBenchmark
// wrapper pattern), so every hardware thread the kernel — or the hidden
// threading of a vendor BLAS — subsequently runs on contributes its counters
// to the region.  No-ops when markers are not compiled in or the binary is
// not running under likwid-perfctr.
void ladder_marker_begin(const char *tag);
void ladder_marker_end(const char *tag);

// LIKWID_MARKER_CLOSE: writes the per-region results for likwid-perfctr.
// Call once just before main returns.
void ladder_markers_finish(void);

#endif // __PROFILER_H_
