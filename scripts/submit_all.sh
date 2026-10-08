#!/bin/bash
# submit_all.sh — fan run_bench.sh out over every processor architecture.
#
# Submits one --exclusive Slurm job per node (CPU or GPU) and lets
# run_bench.sh detect the actual hardware there.  Results land in
# results/<arch>[_<gpu>]/ on the shared home filesystem.
#
# Knobs (environment):
#   NODES="..."        override the default node map (whitespace-separated
#                      node names); default below reflects the cluster state
#                      of Oct 2026 — edit to add GPU nodes you own
#   SIZES DATATYPES REPEATS TOOLCHAIN USE_LIKWID LIKWID_GROUP
#   PIN CPU_BIND PIN_FREQ
#                      all forwarded to run_bench.sh (see its header)
#   WALLTIME=2:00:00   per-job time limit
#   DRY_RUN=1          print the sbatch commands without submitting
#
# After all jobs finish:  python3 scripts/parse_likwid.py
#                         then open scripts/visualize.ipynb
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# node:label — label only names the log file; run_bench.sh auto-detects the
# architecture on the node itself.  Add GPU nodes here, e.g.
#   "h100node1:h100" "mi300x1:mi300x"
DEFAULT_NODES="
turin1:turin
genoa1:genoa
milan1:milan
icx32:icx
saprap2:spr
granrap2:gnr
gracesup1:grace
"
NODES="${NODES:-$DEFAULT_NODES}"
WALLTIME="${WALLTIME:-2:00:00}"

# sweep config forwarded to every node (small cubes first: cache locality)
SIZES="${SIZES:-256 512 1024 2048 4096 6144 8192 12288}"
DATATYPES="${DATATYPES:-FP32 FP64 BF16}"
REPEATS="${REPEATS:-3}"
TOOLCHAIN="${TOOLCHAIN:-GCC}"
USE_LIKWID="${USE_LIKWID:-1}"
PIN="${PIN:-auto}"
CPU_BIND="${CPU_BIND:-cores}"
PIN_FREQ="${PIN_FREQ:-1}"
LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

submitted=0
for entry in $NODES; do
    node="${entry%%:*}"; label="${entry##*:}"

    state="$(scontrol show node "$node" 2>/dev/null | awk -F= '/State=/{print $2; exit}' | cut -d, -f1)"
    if [ -z "$state" ]; then
        echo "[submit_all] $node ($label): NOT KNOWN to Slurm, skipped"
        continue
    fi
    case "$state" in
        *DOWN*|*DRAIN*|*NOT_RESPONDING*)
            echo "[submit_all] $node ($label): state=$state, skipped"; continue ;;
        *RESERVED*|*ALLOCATED*|*COMPLETING*)
            echo "[submit_all] $node ($label): state=$state, BUSY — skipped (re-run later)"
            continue ;;
    esac

    cmd="sbatch --job-name=bench_$label --nodes=1 --exclusive --time=$WALLTIME \
        -w $node --constraint=hwperf --output=$LOG_DIR/bench_$label.%j.log \
        --wrap='USE_LIKWID=$USE_LIKWID PIN=$PIN CPU_BIND=$CPU_BIND PIN_FREQ=$PIN_FREQ SIZES=\"$SIZES\" DATATYPES=\"$DATATYPES\" REPEATS=$REPEATS TOOLCHAIN=$TOOLCHAIN bash $REPO_ROOT/scripts/run_bench.sh'"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] $cmd"
    else
        eval "$cmd"
    fi
    submitted=$((submitted + 1))
done

echo "[submit_all] processed $submitted node(s); results -> $REPO_ROOT/results/<arch>/"
echo "[submit_all] when jobs are done, run:  python3 scripts/parse_likwid.py"
