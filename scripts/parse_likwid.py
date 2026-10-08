#!/usr/bin/env python3
"""parse_likwid.py — turn GEMM benchmark captures into tidy pandas CSVs.

Understands three input dialects (auto-detected):

1.  results/<arch>/gemm_sweep_<DT>.csv / gpu_gemm_<DT>.csv   — app ladder CSV
2.  results/<arch>/likwid_<DT>_<group>_<size>.csv            — `likwid-perfctr -O`
       (csv lines: `TABLE,Group 1 Metric,<grp>,<n>,...` headers followed by
        `metric,value,value,...` rows; uncore/RAPL metrics appear on one
        thread per socket, core metrics on every thread — see AGGREGATION)
3.  legacy `check*.txt` stdout captures (ASCII tables from likwid-perfctr
    without -O, interleaved app output) — CPU/GEMMBench/check*.txt

Outputs, per processor directory `output/<proc>/`:
    gemm.csv      one row per (dtype, size, ladder rung)
    likwid.csv    one row per (dtype, size, group, metric) socket-aggregated
    summary.csv   one row per (dtype, size): best GFlop/s, likwid SP/DP,
                  Energy, Memory bandwidth/volume, GFLOP/J ...

Run:  python3 scripts/parse_likwid.py [--results DIR] [--out DIR] [--legacy FILE...]
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

import pandas as pd

# metric-name -> how to combine per-HWThread values into a socket total
#   core FLOP rates: every thread has its share        -> sum
#   uncore/RAPL: only one thread per socket is non-zero -> sum
#   Runtime: identical on all threads                   -> max
#   clock/CPI/temperature: per-thread scalar            -> mean
def aggregate(metric_name: str, values):
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return float("nan")
    n = metric_name
    if "Runtime" in n:
        return max(vals)
    if any(k in n for k in ("Clock", "CPI", "Temperature")):
        return sum(vals) / len(vals)
    return sum(vals)


def fnum(s):
    s = s.strip().strip('"')
    if s in ("", "-", "*"):
        return float("nan")
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# 1. app ladder CSV  (level,M,N,K,time_s,repeats,gflops[,power_W,...])
# ---------------------------------------------------------------------------
def parse_app_csv(path, proc, dtype):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    perf_col = "gflops" if "gflops" in df.columns else "tflops"
    size_map = {"FP32": 4, "FP64": 8, "FP16": 2, "BF16": 2}
    esz = size_map.get(dtype, 4)
    df.insert(0, "dtype", dtype)
    if "M" in df.columns:
        df["footprint_bytes"] = (df["M"] * df["K"] + df["K"] * df["N"]
                                 + df["M"] * df["N"]) * esz
        # ladder can grow sizes; a single-size capture is one M value
        df["cube"] = df["M"]
    df = df.rename(columns={perf_col: "perf"})
    df["unit"] = "GFlop/s" if perf_col == "gflops" else "TFlop/s"
    df.insert(0, "proc", proc)
    df["source"] = "gpu" if "gpu_gemm" in os.path.basename(path) else "cpu"
    return df


# ---------------------------------------------------------------------------
# 2. likwid -O csv (stdout capture: lines of `name,v,v,...` with TABLE headers)
# ---------------------------------------------------------------------------
def parse_likwid_csv(path):
    """returns (meta dict, {metric: [values]}, {event: (counter, [values])})"""
    meta, metrics, events = {}, {}, {}
    section = None
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            first = row[0].strip()
            if first == "STRUCT":
                continue
            if first == "CPU name:":
                meta["cpu_name"] = row[1].strip() if len(row) > 1 else ""
                continue
            if first == "CPU type:":
                meta["cpu_type"] = row[1].strip() if len(row) > 1 else ""
                continue
            if first == "CPU clock:":
                meta["cpu_clock"] = row[1].strip() if len(row) > 1 else ""
                continue
            if first == "TABLE":
                kind = row[1] if len(row) > 1 else ""
                section = None
                if "Metric" in kind and "STAT" not in kind:
                    section = "metric"
                elif "Raw" in kind and "STAT" not in kind:
                    section = "raw"
                elif "STAT" in kind:
                    section = None      # ignore: we aggregate ourselves
                continue
            if section == "metric" and first and first != "Metric":
                vals = [fnum(v) for v in row[1:]]
                metrics.setdefault(first, []).extend(v for v in vals if not math.isnan(v))
            elif section == "raw" and first and first != "Event":
                counter = row[1].strip() if len(row) > 1 else ""
                vals = [fnum(v) for v in row[2:]]
                events.setdefault(first, (counter, []))[1].extend(v for v in vals if not math.isnan(v))
    return meta, metrics, events


# filename convention created by run_bench.sh:  likwid_<DT>_<GROUP>_<size>.csv
LIKWID_FNAME = re.compile(r"likwid_(?P<dt>FP\d+|BF16)_(?P<group>[A-Za-z0-9_]+)_(?P<size>\d+)\.csv$")


def parse_results_dir(res_dir, out_dir):
    proc = os.path.basename(res_dir.rstrip("/"))
    odir = os.path.join(out_dir, proc)
    os.makedirs(odir, exist_ok=True)

    # meta.csv
    meta = {}
    mp = os.path.join(res_dir, "meta.csv")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = dict((r["key"], r["value"]) for r in csv.DictReader(f))

    gemm_frames = []
    for p in sorted(glob.glob(os.path.join(res_dir, "gemm_sweep_*.csv"))):
        dt = re.search(r"_(FP\d+|BF16)\.csv$", p).group(1)
        gemm_frames.append(parse_app_csv(p, proc, dt))
    for p in sorted(glob.glob(os.path.join(res_dir, "gpu_gemm_*.csv"))):
        dt = re.search(r"_(FP\d+|BF16)\.csv$", p).group(1)
        gemm_frames.append(parse_app_csv(p, proc, dt))
    gemm = pd.concat(gemm_frames, ignore_index=True) if gemm_frames else pd.DataFrame()

    likw_rows, pt_rows = [], []
    for p in sorted(glob.glob(os.path.join(res_dir, "likwid_*.csv"))):
        m = LIKWID_FNAME.search(os.path.basename(p))
        if not m:
            continue
        _, metrics, _ = parse_likwid_csv(p)
        rec = {"proc": proc, "dtype": m["dt"], "group": m["group"],
               "size": int(m["size"])}
        for name, vals in metrics.items():
            rec[name] = aggregate(name, vals)
            for i, v in enumerate(vals):
                pt_rows.append({"proc": proc, "dtype": m["dt"], "group": m["group"],
                                "size": int(m["size"]), "metric": name,
                                "hwthread": i, "value": v})
        likw_rows.append(rec)
    likwid = pd.DataFrame(likw_rows)
    perthread = pd.DataFrame(pt_rows)

    if len(gemm):
        gemm.to_csv(os.path.join(odir, "gemm.csv"), index=False)
    if len(likwid):
        likwid.to_csv(os.path.join(odir, "likwid.csv"), index=False)
    if len(perthread):
        perthread.to_csv(os.path.join(odir, "likwid_perthread.csv"), index=False)

    summary = build_summary(proc, meta, gemm, likwid)
    if len(summary):
        summary.to_csv(os.path.join(odir, "summary.csv"), index=False)
    with open(os.path.join(odir, "meta.csv"), "w") as f:
        for k, v in meta.items():
            f.write(f"{k},{v}\n")
    return odir


def build_summary(proc, meta, gemm, likwid):
    """one row per (dtype, size) combining app perf + likwid counters."""
    rows = []
    if len(gemm):
        cpu_g = gemm[gemm["source"] == "cpu"]
    else:
        cpu_g = pd.DataFrame()
    keys = set()
    if len(cpu_g):
        keys |= {(r.dtype, int(r.cube)) for r in cpu_g.itertuples()}
    if len(likwid):
        keys |= {(r.dtype, int(r.size)) for r in likwid.itertuples()}
    for dt, sz in sorted(keys):
        rec = {"proc": proc, "dtype": dt, "size": sz,
               "arch": meta.get("arch", proc),
               "cpu_name": meta.get("cpu_name", ""),
               "l1d_bytes": meta.get("l1d_bytes", ""),
               "l2_bytes": meta.get("l2_bytes", ""),
               "l3_bytes": meta.get("l3_bytes", "")}
        if len(cpu_g):
            sub = cpu_g[(cpu_g["dtype"] == dt) & (cpu_g["cube"] == sz)]
            if len(sub):
                rec["gemm_best_gflops"] = float(sub["perf"].max())
                best = sub.loc[sub["perf"].idxmax()]
                rec["gemm_best_level"] = best["level"]
                # the two rungs the cache-locality story cares about
                v0 = sub[sub["level"].str.startswith("V0")]
                v2 = sub[sub["level"].str.startswith("V2")]
                if len(v0):
                    rec["gemm_v0_gflops"] = float(v0["perf"].iloc[0])
                if len(v2):
                    rec["gemm_v2_gflops"] = float(v2["perf"].iloc[0])
        if len(likwid):
            sub = likwid[(likwid["dtype"] == dt) & (likwid["size"] == sz)]
            if len(sub):
                for col in ("SP [MFLOP/s]", "DP [MFLOP/s]", "BF16 [MFLOP/s]",
                            "512B BF16 [MFLOP/s]", "FP rate [MFLOP/s]",
                            "Energy PKG [J]", "Energy [J]", "Power PKG [W]", "Power [W]",
                            "Memory bandwidth [MBytes/s]", "Memory data volume [GBytes]",
                            "Memory read bandwidth [MBytes/s]", "Memory write bandwidth [MBytes/s]",
                            "Runtime (RDTSC) [s]"):
                    if col in sub.columns:
                        vals = sub[col].dropna()
                        if len(vals):
                            # rate metrics: take the max across group passes
                            # (identical physics measured in two likwid runs
                            #  is not additive); volume/energy likewise max
                            rec[col.replace(" [", "_").replace("]", "").replace(" ", "_")
                                .replace("(", "").replace(")", "")] = float(vals.max())
            if "Energy_PKG_J" in rec and "gemm_best_gflops" in rec and rec["Energy_PKG_J"]:
                rt = rec.get("Runtime_RDTSC_s", float("nan"))
                if rt == rt and rt > 0:
                    rec["gflops_per_joule"] = rec["gemm_best_gflops"] / (rec["Energy_PKG_J"] / rt)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. legacy txt captures (likwid ascii tables + app output)
# ---------------------------------------------------------------------------
def parse_legacy_txt(path):
    """returns (proc, meta, gemm_df, likwid_df) from a check*.txt capture."""
    txt = open(path, errors="replace").read()
    meta = {}
    for k in ("CPU name", "CPU type", "CPU clock"):
        m = re.search(rf"^{k}:\s*(.+)$", txt, re.M)
        meta[k.lower().replace(" ", "_")] = m.group(1).strip() if m else ""
    arch = {"Intel GraniteRapids processor": "GNR",
            "Intel SapphireRapids processor": "SPR",
            "Intel IceLakeX processor": "ICX"}.get(meta.get("cpu_type", ""),
            meta.get("cpu_type", "legacy").split()[0].lower())
    proc = os.path.splitext(os.path.basename(path))[0]

    # app ladder lines: "  V0 naive-omp ijk   61.76 GFlop/s  (5 reps, 1.391 s)"
    g = []
    for m in re.finditer(r"^\s{2}(\S.*?)\s+([\d.]+)\s+GFlop/s\s+\((\d+) reps, ([\d.]+) s\)",
                         txt, re.M):
        g.append({"level": m.group(1).strip(), "perf": float(m.group(2)),
                  "repeats": int(m.group(3)), "time_s": float(m.group(4)),
                  "unit": "GFlop/s"})
    size = None
    ms = re.search(r"shape: C\[(\d+)x\d+\]", txt)
    if ms:
        size = int(ms.group(1))
    ms = re.search(r"base (\d+)x", txt)
    if ms:
        size = int(ms.group(1))
    gemm = pd.DataFrame(g)
    if len(gemm):
        gemm["cube"] = size
        gemm["M"] = gemm["N"] = gemm["K"] = size
        gemm["dtype"] = "FP64" if re.search(r"ladder \(FP64", txt) else "FP32"
        gemm["source"] = "cpu"
        gemm.insert(0, "proc", proc)

    # metric tables: last per-thread metric table + STAT lines
    dt_guess = "FP64" if re.search(r"ladder \(FP64", txt) else "FP32"
    metrics = {}
    for m in re.finditer(r"\|\s*([A-Za-z][^|\[]+?(?:\[[^\]]*\])?)\s*\|\s*"
                         r"((?:\|?\s*[\d.eE+-]+\s*\|)+)", txt):
        pass  # too ambiguous; prefer the STAT lines below
    for m in re.finditer(r"^\|\s*(.+?) STAT\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|", txt, re.M):
        metrics[m.group(1).strip()] = {"sum": float(m.group(2)), "min": float(m.group(3)),
                                       "max": float(m.group(4)), "avg": float(m.group(5))}
    grp = re.search(r"Group \d+: (\w+)", txt)
    rec = {"proc": proc, "dtype": dt_guess, "group": grp.group(1) if grp else "",
           "size": size or 0}
    for name, st in metrics.items():
        # mirror the csv-path rules: rates/energy/volumes sum over threads,
        # runtime/clock/cpi are identical or per-thread scalars -> avg
        if any(k in name for k in ("Runtime", "Clock", "CPI", "Temperature")):
            rec[name] = st["avg"]
        else:
            rec[name] = st["sum"]
    likwid = pd.DataFrame([rec])
    return proc, meta, gemm, likwid


def write_legacy_outputs(path, out_root):
    proc, meta, gemm, likwid = parse_legacy_txt(path)
    odir = os.path.join(out_root, proc)
    os.makedirs(odir, exist_ok=True)
    if len(gemm):
        gemm.to_csv(os.path.join(odir, "gemm.csv"), index=False)
    if len(likwid):
        likwid.to_csv(os.path.join(odir, "likwid.csv"), index=False)
    s = build_summary(proc, meta, gemm, likwid)
    if len(s):
        s.to_csv(os.path.join(odir, "summary.csv"), index=False)
    with open(os.path.join(odir, "meta.csv"), "w") as f:
        for k, v in meta.items():
            f.write(f"{k},{v}\n")
    return odir


# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=os.path.join(repo, "results"))
    ap.add_argument("--out", default=os.path.join(repo, "output"))
    ap.add_argument("--legacy", nargs="*", default=[],
                    help="extra check*.txt files to parse")
    args = ap.parse_args()

    n = 0
    if os.path.isdir(args.results):
        for d in sorted(glob.glob(os.path.join(args.results, "*/"))):
            if glob.glob(os.path.join(d, "gemm_sweep_*.csv")) or \
               glob.glob(os.path.join(d, "gpu_gemm_*.csv")) or \
               glob.glob(os.path.join(d, "likwid_*.csv")):
                odir = parse_results_dir(d, args.out)
                print(f"parsed {d} -> {odir}")
                n += 1
    for p in args.legacy:
        odir = write_legacy_outputs(p, args.out)
        print(f"parsed legacy {p} -> {odir}")
        n += 1
    if not args.legacy and n == 0 and os.path.isdir(repo):
        # no results/ yet: fall back to any check*.txt in the repo
        for p in sorted(glob.glob(os.path.join(repo, "**/check*.txt"), recursive=True)):
            odir = write_legacy_outputs(p, args.out)
            print(f"parsed legacy {p} -> {odir}")
            n += 1
    if n == 0:
        print("nothing to parse — run scripts/run_bench.sh first "
              "(or pass --legacy check.txt)", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {n} processor folder(s) to {args.out}")


if __name__ == "__main__":
    main()
