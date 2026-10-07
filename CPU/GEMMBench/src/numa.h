#ifndef NUMA_H
#define NUMA_H

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// ccNUMA topology discovery + placement helpers.
//
// Everything is read from /sys (no libnuma dependency):
//   /sys/devices/system/node/online            -> node list
//   /sys/devices/system/node/node<N>/cpulist   -> logical CPUs per node
//
// Placement model:
//   - numa_plan_*() builds a rank -> logical-CPU table for a team of T
//     OpenMP threads ("spread" = round-robin across nodes, "grouped" =
//     contiguous per-node bands).
//   - numa_apply_*() spawns a tiny parallel region in which every thread
//     pins itself with sched_setaffinity() according to that table.  This
//     works independently of OMP_PROC_BIND and can be (re)applied between
//     ladder rungs.
//   - numa_first_touch() copies src -> dst (or zeroes) in per-rank slices
//     inside such a pinned region, so the Linux first-touch policy places
//     each page on the NUMA node of the thread that will compute it.
// ---------------------------------------------------------------------------

#define NUMA_MAX_NODES 16
#define NUMA_MAX_CPUS  4096

typedef struct {
    int  valid;
    int  id;
    int  ncpus;
    int  cpus[NUMA_MAX_CPUS];
} numa_node_t;

typedef struct {
    int           n_nodes;
    int           n_cpus;                 // total logical CPUs across nodes
    numa_node_t   node[NUMA_MAX_NODES];
    int           node_of_cpu[NUMA_MAX_CPUS];
} numa_topology_t;

// rank -> cpu assignment tables (entry -1 = leave unpinned)
typedef struct {
    int mode;                 // NUMA_MODE_*
    int nthreads;
    int cpu[NUMA_MAX_CPUS];   // pinned cpu per rank
    int node[NUMA_MAX_CPUS];  // owning node per rank
    // per-node rank bands (grouped placement): ranks
    // [node_start_rank[i], +node_rank_count[i]) of a team belong to node i
    int n_nodes;
    int node_start_rank[NUMA_MAX_NODES];
    int node_rank_count[NUMA_MAX_NODES];
} numa_plan_t;

enum {
    NUMA_MODE_DEFAULT = 0,    // unpin (full cpu set) — OS decides
    NUMA_MODE_SPREAD  = 1,    // round-robin threads over nodes (V3)
    NUMA_MODE_GROUPED = 2     // contiguous per-node thread bands (V4/V5)
};

// Probe /sys.  Returns 0 on success.  On systems without a NUMA sysfs the
// topology collapses to a single node covering sched_getaffinity()'s CPUs.
int numa_probe(numa_topology_t *t);

// Human-readable one-line-per-node description; returns t's static buffer.
const char *numa_describe(const numa_topology_t *t);

// Build a plan for nthreads worker ranks under the given mode.
void numa_plan(const numa_topology_t *t, int mode, int nthreads, numa_plan_t *p);

// Open a parallel region of p->nthreads (or the ambient team for
// numa_pin_self) applying affinity.  Call these OUTSIDE timed sections.
void numa_apply_plan(const numa_topology_t *t, const numa_plan_t *p);

// First-touch staging: the ambient pinned team writes [0, nbytes) of dst
// in contiguous per-rank slices, so the Linux first-touch policy places
// each page on the NUMA node of the thread that will compute it.
// src may be NULL (zero-fill).  Must be called after numa_apply_plan().
void numa_first_touch_bytes(void *dst, const void *src, size_t nbytes);

// ---------------------------------------------------------------------------
// /proc/self/numa_maps page footprint of one buffer (best effort: small
// allocations live in shared heap mappings and cannot be attributed).
// Fills counts[] with pages per node for mappings starting exactly at the
// buffer's first page.  Returns total pages, -1 on failure.
// ---------------------------------------------------------------------------
long numa_footprint(const void *ptr, size_t bytes, long *counts, int max_nodes);

// Returns the node owning `ptr`'s pages per numa_maps, or -1.  Used to
// sanity-check placement in the report.
int numa_footprint_dominant(const void *ptr, size_t bytes);

#endif // NUMA_H
