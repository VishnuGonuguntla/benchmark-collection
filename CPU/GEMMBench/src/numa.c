#define _GNU_SOURCE
#include "numa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// range-list parsing: "0-15,64,128-143"
// ---------------------------------------------------------------------------
static int parse_cpu_list(const char *s, int *out, int max)
{
    int n = 0;
    const char *p = s;
    while (*p && n < max) {
        char *end;
        long a = strtol(p, &end, 10);
        if (end == p) break;
        long b = a;
        if (*end == '-') { p = end + 1; b = strtol(p, &end, 10); p = end; }
        else p = end;
        for (long c = a; c <= b && n < max; c++)
            if (c >= 0) out[n++] = (int)c;
        while (*p == ',') p++;
    }
    return n;
}

static int sysfs_read(const char *path, char *buf, size_t cap)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, (int)cap, f)) { fclose(f); return -1; }
    fclose(f);
    size_t l = strlen(buf);
    while (l && (buf[l-1] == '\n' || buf[l-1] == '\r')) buf[--l] = '\0';
    return 0;
}

static void pin_to_cpu(int cpu)
{
    int maxcpu = NUMA_MAX_CPUS;
    size_t sz = CPU_ALLOC_SIZE(maxcpu);
    cpu_set_t *set = CPU_ALLOC(maxcpu);
    if (!set) return;
    CPU_ZERO_S(sz, set);
    if (cpu >= 0) {
        CPU_SET_S(cpu, sz, set);
    } else {
        // unpin: allow every cpu in the current (intersection-)set.  Query
        // the existing affinity first so we never widen beyond cgroups.
        sched_getaffinity(0, sz, set);
    }
    if (sched_setaffinity(0, sz, set) != 0 && cpu >= 0) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "numa: sched_setaffinity(cpu=%d) failed: %s\n",
                    cpu, strerror(errno));
        }
    }
    CPU_FREE(set);
}

// ---------------------------------------------------------------------------
int numa_probe(numa_topology_t *t)
{
    memset(t, 0, sizeof(*t));

    char buf[8192];
    int ids[NUMA_MAX_NODES], nid = 0;

    if (sysfs_read("/sys/devices/system/node/online", buf, sizeof(buf)) == 0) {
        int tmp[NUMA_MAX_NODES];
        nid = parse_cpu_list(buf, tmp, NUMA_MAX_NODES);
        for (int i = 0; i < nid; i++) ids[i] = tmp[i];
    }
    if (nid == 0) { ids[0] = 0; nid = 1; }

    t->n_nodes = 0;
    for (int i = 0; i < nid && t->n_nodes < NUMA_MAX_NODES; i++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/node/node%d/cpulist", ids[i]);
        if (sysfs_read(path, buf, sizeof(buf)) != 0) continue;
        numa_node_t *nd = &t->node[t->n_nodes];
        nd->valid = 1;
        nd->id = ids[i];
        nd->ncpus = parse_cpu_list(buf, nd->cpus, NUMA_MAX_CPUS);
        for (int c = 0; c < nd->ncpus; c++) {
            if (nd->cpus[c] >= 0 && nd->cpus[c] < NUMA_MAX_CPUS)
                t->node_of_cpu[nd->cpus[c]] = t->n_nodes;
        }
        t->n_cpus += nd->ncpus;
        t->n_nodes++;
    }

    if (t->n_nodes == 0 || t->n_cpus == 0) {
        // fallback: current affinity mask as a single node
        size_t sz = CPU_ALLOC_SIZE(NUMA_MAX_CPUS);
        cpu_set_t *set = CPU_ALLOC(NUMA_MAX_CPUS);
        int n = 0;
        if (set && sched_getaffinity(0, sz, set) == 0) {
            numa_node_t *nd = &t->node[0];
            nd->valid = 1; nd->id = 0;
            for (int c = 0; c < NUMA_MAX_CPUS; c++)
                if (CPU_ISSET_S(c, sz, set)) nd->cpus[n++] = c;
            nd->ncpus = n;
            t->n_nodes = 1; t->n_cpus = n;
        }
        if (set) CPU_FREE(set);
    }
    return (t->n_nodes > 0 && t->n_cpus > 0) ? 0 : -1;
}

const char *numa_describe(const numa_topology_t *t)
{
    static char buf[2048];
    int off = snprintf(buf, sizeof(buf),
                       "NUMA topology: %d node(s), %d logical CPUs\n",
                       t->n_nodes, t->n_cpus);
    for (int i = 0; i < t->n_nodes && off < (int)sizeof(buf) - 96; i++) {
        const numa_node_t *nd = &t->node[i];
        off += snprintf(buf + off, sizeof(buf) - off,
                        "  node%-2d id=%-3d cpus=%-5d [%d",
                        i, nd->id, nd->ncpus, nd->cpus[0]);
        off += snprintf(buf + off, sizeof(buf) - off,
                        nd->ncpus > 1 ? "-%d]\n" : "]\n",
                        nd->cpus[nd->ncpus - 1]);
    }
    return buf;
}

// ---------------------------------------------------------------------------
void numa_plan(const numa_topology_t *t, int mode, int nthreads, numa_plan_t *p)
{
    memset(p, 0, sizeof(*p));
    if (nthreads > NUMA_MAX_CPUS) nthreads = NUMA_MAX_CPUS;
    p->mode = mode;
    p->nthreads = nthreads;

    for (int r = 0; r < nthreads; r++) { p->cpu[r] = -1; p->node[r] = 0; }
    if (mode == NUMA_MODE_DEFAULT || t->n_nodes == 0) return;

    if (mode == NUMA_MODE_SPREAD) {
        int next[NUMA_MAX_NODES];
        for (int i = 0; i < t->n_nodes; i++) next[i] = 0;
        for (int r = 0; r < nthreads; r++) {
            int ni = -1;
            for (int tries = 0, i = r % t->n_nodes; tries < t->n_nodes;
                 tries++, i = (i + 1) % t->n_nodes) {
                if (next[i] < t->node[i].ncpus) { ni = i; break; }
            }
            if (ni < 0) {                       // exhausted: wrap node 0
                const numa_node_t *nd = &t->node[0];
                p->cpu[r] = nd->cpus[r % nd->ncpus];
                p->node[r] = 0;
            } else {
                p->cpu[r] = t->node[ni].cpus[next[ni]++];
                p->node[r] = ni;
            }
        }
        return;
    }

    // GROUPED: contiguous rank bands sized proportional to node core counts
    int base = 0;
    p->n_nodes = t->n_nodes;
    for (int i = 0; i < t->n_nodes; i++) {
        p->node_start_rank[i] = base;
        p->node_rank_count[i] = 0;
    }
    for (int i = 0; i < t->n_nodes && base < nthreads; i++) {
        int band;
        if (i == t->n_nodes - 1)
            band = nthreads - base;                       // last gets the rest
        else
            band = (int)((long)nthreads * t->node[i].ncpus / t->n_cpus);
        if (band < 1) band = 1;
        if (base + band > nthreads) band = nthreads - base;
        const numa_node_t *nd = &t->node[i];
        for (int s = 0; s < band; s++) {
            p->cpu[base + s]  = nd->cpus[s % nd->ncpus];
            p->node[base + s] = i;
        }
        p->node_start_rank[i] = base;
        p->node_rank_count[i] = band;
        base += band;
    }
    for (int r = base; r < nthreads; r++) {              // (safety) tail
        int li = t->n_nodes - 1;
        p->cpu[r] = t->node[li].cpus[(r - base) % t->node[li].ncpus];
        p->node[r] = li;
        p->node_rank_count[li]++;
    }
}

// ---------------------------------------------------------------------------
static numa_plan_t g_plan;

static void numa_pin_self(void)
{
    int r = 0;
#ifdef _OPENMP
    r = omp_get_thread_num();
#endif
    pin_to_cpu(r < g_plan.nthreads ? g_plan.cpu[r] : -1);
}

void numa_apply_plan(const numa_topology_t *t, const numa_plan_t *p)
{
    (void)t;
    g_plan = *p;
#ifdef _OPENMP
    if (p->nthreads > 1) {
        #pragma omp parallel
        numa_pin_self();
    } else {
        numa_pin_self();
    }
#else
    numa_pin_self();
#endif
}

void numa_first_touch_bytes(void *dst, const void *src, size_t nbytes)
{
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int rank = 0, nthreads = 1;
#ifdef _OPENMP
        rank = omp_get_thread_num(); nthreads = omp_get_num_threads();
#endif
        size_t a = nbytes * (size_t)rank     / (size_t)nthreads;
        size_t b = nbytes * (size_t)(rank + 1) / (size_t)nthreads;
        if (nbytes >= (1u << 20)) {          // page-align slices for big
            a &= ~(size_t)4095;              // buffers so whole pages are
            b &= ~(size_t)4095;              // faulted by one rank
        }
        if (rank == nthreads - 1) b = nbytes;
        if (b < a) b = a;
        if (b > nbytes) b = nbytes;
        if (src) memcpy((char *)dst + a, (const char *)src + a, b - a);
        else     memset((char *)dst + a, 0, b - a);
    }
}

// ---------------------------------------------------------------------------
// /proc/self/numa_maps footprint: sum "nodeN=" pages for mappings overlapping
// [ptr, ptr+bytes), attributed proportionally to the overlap.
// ---------------------------------------------------------------------------
#define FT_MAX_MAPS 16384

typedef struct {
    unsigned long start;
    long count[NUMA_MAX_NODES];
} ft_map_t;

long numa_footprint(const void *ptr, size_t bytes, long *counts, int max_nodes)
{
    FILE *f = fopen("/proc/self/numa_maps", "r");
    if (!f) return -1;

    long pagesz = sysconf(_SC_PAGESIZE);
    if (pagesz <= 0) pagesz = 4096;
    unsigned long lo = (unsigned long)(uintptr_t)ptr & ~(unsigned long)(pagesz - 1);
    unsigned long hi = ((unsigned long)(uintptr_t)ptr + bytes + pagesz - 1)
                       & ~(unsigned long)(pagesz - 1);

    if (max_nodes > NUMA_MAX_NODES) max_nodes = NUMA_MAX_NODES;
    for (int i = 0; i < max_nodes; i++) counts[i] = 0;

    ft_map_t *maps = calloc(FT_MAX_MAPS, sizeof(ft_map_t));
    if (!maps) { fclose(f); return -1; }
    int n = 0;

    char line[4096];
    while (n < FT_MAX_MAPS && fgets(line, sizeof(line), f)) {
        unsigned long start;
        if (sscanf(line, "%lx", &start) != 1) continue;
        maps[n].start = start;
        const char *sp = line;
        char *sp2 = strchr(line, ' ');
        while (sp2) {
            sp = sp2 + 1;
            int nid; long pages;
            // kernels emit either "node0=123" or "N0=123" per-node counts
            if (sscanf(sp, "node%d=%ld", &nid, &pages) == 2 ||
                sscanf(sp, "N%d=%ld", &nid, &pages) == 2) {
                if (nid >= 0 && nid < max_nodes)
                    maps[n].count[nid] += pages;
            }
            sp2 = strchr(sp, ' ');
        }
        n++;
    }
    fclose(f);

    long total = 0;
    for (int i = 0; i < n; i++) {
        unsigned long mstart = maps[i].start;
        unsigned long mend = (i + 1 < n) ? maps[i + 1].start : ~0UL;
        unsigned long olo = mstart > lo ? mstart : lo;
        unsigned long ohi = mend  < hi ? mend   : hi;
        if (ohi <= olo) continue;
        double frac = (double)(ohi - olo) / (double)(mend - mstart);
        for (int k = 0; k < max_nodes; k++) {
            long c = (long)(maps[i].count[k] * frac);
            counts[k] += c;
            total += c;
        }
    }
    free(maps);
    return total;
}

int numa_footprint_dominant(const void *ptr, size_t bytes)
{
    long counts[NUMA_MAX_NODES];
    long total = numa_footprint(ptr, bytes, counts, NUMA_MAX_NODES);
    if (total <= 0) return -1;
    int best = 0;
    for (int i = 1; i < NUMA_MAX_NODES; i++)
        if (counts[i] > counts[best]) best = i;
    return best;
}
