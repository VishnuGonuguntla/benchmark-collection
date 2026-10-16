/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved.  MIT style license, see TheBandwidthBenchmark.
 *
 * LIKWID MarkerAPI wrappers for the GEMM ladder (see profiler.h). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "likwid-marker.h"
#include "profiler.h"

// Region tags are derived from the rung id so nothing here branches on
// library identity: hand rungs are V0..V5, vendor rungs carry their marker
// string in their BLAS_REGISTRY row (src/library.c).
static const char *const HAND_MARKERS[LVL_LIB_FIRST] =
    { "V0", "V1", "V2", "V3", "V4", "V5", "BF16" };

const char *rung_marker(int id)
{
    if (id >= 0 && id < LVL_LIB_FIRST) return HAND_MARKERS[id];
    if (LVL_IS_LIB(id)) return BLAS_REGISTRY[LIB_FOR_LVL(id)].marker;
    return "?";
}

#ifdef LIKWID_PERFMON

// likwid-perfctr -m hands the event set to the application through
// LIKWID_EVENTS (older builds: LIKWID_NREVENTS).  Outside the wrapper the
// marker API has nothing to program, so skip the parallel START/STOP teams
// entirely - an unwrapped LIKWID=1 build then costs nothing.
static int markers_enabled(void)
{
    return getenv("LIKWID_EVENTS") != NULL || getenv("LIKWID_NREVENTS") != NULL;
}

void ladder_markers_init(void)
{
    if (!markers_enabled()) return;  // plain run, not under likwid-perfctr
    LIKWID_MARKER_INIT;              // must run before the first OMP region
#ifdef _OPENMP
    #pragma omp parallel             // implicit barrier at the end
    {
        LIKWID_MARKER_THREADINIT;
        for (int id = 0; id < LVL_COUNT; id++)
            if (level_compiled(id))
                LIKWID_MARKER_REGISTER(rung_marker(id));
    }
#else
    for (int id = 0; id < LVL_COUNT; id++)
        if (level_compiled(id))
            LIKWID_MARKER_REGISTER(rung_marker(id));
#endif
}

// START/STOP on every thread of the ambient team around an otherwise empty
// parallel region: the rung's own parallel regions reuse the same pool, so
// all hardware threads running the kernel participate in the counters.  For
// the vendor rungs this is the documented recipe for hidden library threads:
// LIKWID counts per hardware thread, and the BLAS threads land on exactly
// the cores the team occupies (blas_use_threads keeps the team sizes equal).
void ladder_marker_begin(const char *tag)
{
    if (!markers_enabled()) return;  // not run under likwid-perfctr -m
#ifdef _OPENMP
    #pragma omp parallel
#endif
    LIKWID_MARKER_START(tag);
}

void ladder_marker_end(const char *tag)
{
    if (!markers_enabled()) return;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    LIKWID_MARKER_STOP(tag);
}

void ladder_markers_finish(void)
{
    if (!markers_enabled()) return;
    LIKWID_MARKER_CLOSE;             // writes the region results likwid-perfctr reads
}

#else   /* no perfmon build: every entry point is a no-op */

void ladder_markers_init(void) {}
void ladder_marker_begin(const char *tag) { (void)tag; }
void ladder_marker_end(const char *tag) { (void)tag; }
void ladder_markers_finish(void) {}

#endif
