#!/bin/bash
# run_bench.sh — GEMM optimization-ladder sweep driver for ONE node (CPU + GPU).
#
# Designed to run directly on a compute node (via sbatch/srun) of any
# supported architecture.  It detects the CPU (via LIKWID's short name), any
# GPUs, builds the matching ladder binaries, sweeps cube sizes (cache
# locality profile L1 -> L2 -> L3 -> DRAM) for FP32 and FP64, and optionally
# wraps every run in likwid-perfctr using the CUSTOM / CUSTOM_SP / CUSTOM_DP
# perf groups.  Results are CSV (app rows + LIKWID -O metrics + per-size
# likwid CSV); full logs go to raw/.
#
# Knobs (environment):
#   USE_LIKWID=0|1       wrap runs with likwid-perfctr          (default 1)
#   LIKWID_GROUP=auto    auto|CUSTOM|CUSTOM_SP|<name>           (default auto)
#   SIZES="256 512 ..."  cube sizes to sweep (cache-locality ladder)
#   DATATYPES="FP32 FP64 BF16"  BF16 uses a dedicated AVX512-BF16 kernel
#                      (CPU/GEMMBench/src/bf16_bench.c); auto-skipped when the
#                      CPU lacks AVX512_BF16
#   THREADS=<int>        measuring threads / OMP_NUM_THREADS    (default: all)
#   REPEATS=<int>        reps per rung                          (default 3)
#   TOOLCHAIN=GCC        GEMMBench toolchain (GCC|ICX|AOCC ...)
#   RESULT_ROOT=DIR      output tree root (default <repo>/results)
#   SKIP_CPU=1 / SKIP_GPU=1
#   CPU_BIND=cores       srun --cpu-bind mode for app steps:
#                        threads|cores|sockets|ldom|none  (default cores)
#   PIN=auto             processor pinning for runs NOT already pinned by
#                        likwid-perfctr -C: auto|1 uses `srun --cpu-bind`
#                        inside a SLURM allocation, else likwid-pin, else
#                        OMP_PROC_BIND; 0 disables                   (default auto)
#   PIN_FREQ=high-high:performance
#                        CPU frequency pin, passed straight to
#                        `srun --cpu-freq=` (SLURM applies it via slurmd —
#                        no root needed).  1/max -> high-high:performance
#                        (governor 'performance', min=max=boost),
#                        0/off -> untouched.  Requires a SLURM allocation.
#
# Usage on a node:  bash scripts/run_bench.sh
# Via Slurm:        sbatch --wrap="bash scripts/run_bench.sh" (see submit_all.sh)

#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --job-name=gracehop1-gemm
#SBATCH -w gracehop1
#SBATCH --exclusive
#SBATCH --constraint=hwperf

# set -u -o pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

USE_LIKWID="${USE_LIKWID:-1}"
PIN_FREQ="${PIN_FREQ:-1}"
CPU_BIND="${CPU_BIND:-cores}"
PIN="${PIN:-auto}"
LIKWID_GROUP="${LIKWID_GROUP:-auto}"
SIZES="${SIZES:-8000}"
DATATYPES="${DATATYPES:-FP32 FP64 BF16}"
REPEATS="${REPEATS:-30}"
TOOLCHAIN="${TOOLCHAIN:-GCC}"
THREADS="${THREADS:-$(nproc)}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results}"
export OMP_NUM_THREADS="$THREADS"

log() { echo "[run_bench] $*"; }

# ---------------------------------------------------------------------------
# tool detection
# ---------------------------------------------------------------------------
LIKWID_BIN=""
LIKWID_PIN_BIN=""
if command -v likwid-perfctr >/dev/null 2>&1; then
    LIKWID_BIN="$(command -v likwid-perfctr)"
elif module load likwid 2>/dev/null && command -v likwid-perfctr >/dev/null 2>&1; then
    LIKWID_BIN="$(command -v likwid-perfctr)"
fi
[ -n "$LIKWID_BIN" ] || { log "likwid-perfctr not found -> USE_LIKWID=0"; USE_LIKWID=0; }
command -v likwid-pin >/dev/null 2>&1 && LIKWID_PIN_BIN="$(command -v likwid-pin)"

# srun usable inside a SLURM allocation (sbatch job shell or interactive
# srun).  Interactive steps may briefly contend with nested steps; the
# capability probe below time-outs and falls back cleanly in that case.
have_srun() { command -v srun >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; }

# ---------------------------------------------------------------------------
# CPU architecture (likwid short name == $HOME/.likwid/groups/<arch> name)
# ---------------------------------------------------------------------------
ARCH=""
CPU_NAME=""
if [ -n "$LIKWID_BIN" ]; then
    ARCH="$($LIKWID_BIN -i 2>/dev/null | awk -F':\t' '/CPU short:/{print $2; exit}' | awk '{print $1}')"
    CPU_NAME="$($LIKWID_BIN -i 2>/dev/null | awk -F':\t' '/CPU name:/{print $2; exit}' | sed 's/[[:space:]]*$//')"
fi
[ -n "$CPU_NAME" ] || CPU_NAME="$(awk -F': *' '/Model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null)"
if [ -z "$ARCH" ]; then
    case "$CPU_NAME" in
        *Granite*)        ARCH=GNR ;;
        *Sapphire*)       ARCH=SPR ;;
        *Emerald*)        ARCH=EMR ;;
        *IceLake*|*"Ice Lake"*) ARCH=ICX ;;
        *Genoa*)          ARCH=zen4 ;;
        *Bergamo*)        ARCH=zen4c ;;
        *Milan*)          ARCH=zen3 ;;
        *Turin*|*EPYC*9005*) ARCH=zen5 ;;
        *Grace*)          ARCH=nvidia_grace ;;
        *)                ARCH=unknown ;;
    esac
fi
[ -n "$CPU_NAME" ] || CPU_NAME="$ARCH"

group_exists() {
    [ -n "$LIKWID_BIN" ] || return 1
    "$LIKWID_BIN" -a 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}
# group(s) for this datatype on this arch; 'auto' honours the CUSTOM layout
pick_groups() {
    local dt="$1"
    if [ "$LIKWID_GROUP" != "auto" ]; then echo "$LIKWID_GROUP"; return; fi
    if [ "$dt" = BF16 ]; then
        case "$ARCH" in
            GNR|SPR|EMR|zen5|zen5c) echo CUSTOM_BF16 ;;
            *) echo "" ;;   # no BF16 counters on zen3/zen4/ICX/...
        esac
        return
    fi
    case "$ARCH" in
        zen3|zen4|zen4c|zen5|zen5c)
            if [ "$dt" = "FP32" ]; then echo CUSTOM_SP; else echo CUSTOM_DP; fi ;;
        GNR|SPR|EMR|ICX|SRF)  echo CUSTOM ;;
        *)   [ "$dt" = "FP32" ] && echo FLOPS_SP || echo FLOPS_DP ;;
    esac
}

# ---------------------------------------------------------------------------
# GPU detection (name -> GPU= make tag of GPU/GEMM/mk/include_<tag>.mk)
# ---------------------------------------------------------------------------
GPU_KIND="" GPU_NAMES="" GPU_TAG=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_KIND=nvidia
    GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//')"
elif command -v rocm-smi >/dev/null 2>&1; then
    GPU_KIND=amd
    GPU_NAMES="$(rocm-smi --showproductname 2>/dev/null | awk '/card[0-9]/{print $NF; exit}')"
fi
if [ -n "$GPU_NAMES" ]; then
    case "$GPU_NAMES" in
        *A100*)  GPU_TAG=A100 ;;
        *A40*)   GPU_TAG=A40 ;;
        *H100*)  GPU_TAG=H100 ;;
        *H200*)  GPU_TAG=H100 ;;   # same sm90 build flags
        *MI300X*) GPU_TAG=MI300X ;;
        *MI300A*) GPU_TAG=MI300X ;;
        *)       GPU_TAG="" ;;      # no matching mk/include_ file
    esac
fi

PROC="${ARCH}${GPU_TAG:+_$GPU_TAG}"
OUT="$RESULT_ROOT/$PROC"
mkdir -p "$OUT/raw"

# ---------------------------------------------------------------------------
# CPU ids to measure: honour the (possibly sparse) job cpuset
# ---------------------------------------------------------------------------
expand_list() {
    local r
    for r in $(tr ',' ' ' <<<"$1"); do
        case "$r" in
            *-*) seq "${r%%-*}" "${r##*-}" ;;
            *)   echo "$r" ;;
        esac
    done
}
CPULIST=""
for f in /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/cpuset/cpuset.cpus; do
    [ -r "$f" ] || continue
    CPULIST="$(expand_list "$(cat "$f")" | head -n "$THREADS" | paste -sd, -)"
    [ -n "$CPULIST" ] && break
done
[ -n "$CPULIST" ] || CPULIST="0-$((THREADS-1))"

# ---------------------------------------------------------------------------
# CPU frequency pinning via `srun --cpu-freq` (applied by slurmd — no root).
#   PIN_FREQ=<MHz>|<srun spec> | 1|max -> high-high:performance
#            (governor performance, scaling min=max=boost) | 0/off -> none
# Runs outside a SLURM allocation cannot use it -> warned + ignored.
# ---------------------------------------------------------------------------
CPUFREQ=""
case "${PIN_FREQ:-1}" in
    0|off|"") ;;
    1|max)  CPUFREQ="high-high:performance" ;;
    *)      CPUFREQ="$PIN_FREQ" ;;
esac

# ---------------------------------------------------------------------------
# Processor pinning + step prefix.
#   - srun available (inside a job): every measurement run is launched as a
#     step with --cpu-bind and --cpu-freq; likwid-perfctr -C additionally
#     pins the measured threads itself.
#   - no srun: likwid-pin fallback, then OpenMP affinity.
# ---------------------------------------------------------------------------
RUN_WRAP=()
PIN_MODE="none"
need_step=0
if [ "$PIN" != 0 ] && [ "$CPU_BIND" != none ]; then need_step=1; fi
[ -n "$CPUFREQ" ] && need_step=1

if have_srun && [ "$need_step" = 1 ]; then
    RUN_WRAP=(srun --ntasks=1 --cpus-per-task="$THREADS")
    if [ "$PIN" != 0 ] && [ "$CPU_BIND" != none ]; then
        RUN_WRAP+=(--cpu-bind="$CPU_BIND")
        PIN_MODE="srun:$CPU_BIND"
    else
        PIN_MODE="srun"
    fi
    if [ -n "$CPUFREQ" ]; then
        RUN_WRAP+=(--cpu-freq="$CPUFREQ")
        PIN_MODE="$PIN_MODE+freq:$CPUFREQ"
    fi
else
    if [ -n "$CPUFREQ" ]; then
        log "WARNING: PIN_FREQ needs sbatch job context (srun --cpu-freq steps) -> frequency pin ignored"
        CPUFREQ=""
    fi
    if [ "$PIN" != 0 ]; then
        if [ -n "$LIKWID_PIN_BIN" ]; then
            RUN_WRAP=("$LIKWID_PIN_BIN" -q -c "$CPULIST")
            PIN_MODE="likwid-pin:$CPULIST"
        else
            export OMP_PROC_BIND=spread OMP_PLACES=cores
            PIN_MODE="omp-proc-bind"
        fi
    fi
fi

# one-time capability probe: Slurm refuses to start steps whose frequency
# request cannot be honoured (or nested steps cannot get CPUs when run_bench
# itself runs as a held step).  Fail fast here — then drop the srun wrapper
# entirely and continue with the likwid-pin/OpenMP fallback.
if [ -n "$CPUFREQ" ] && [ "${RUN_WRAP[0]:-}" = srun ]; then
    if ! timeout 120 srun --ntasks=1 --cpu-freq="$CPUFREQ" true >>"$OUT/raw/freq.log" 2>&1; then
        log "WARNING: srun --cpu-freq='$CPUFREQ' unavailable (see $OUT/raw/freq.log) -> frequency pin disabled, srun step wrapper dropped"
        CPUFREQ=""
        RUN_WRAP=()
        if [ "$PIN" != 0 ]; then
            if [ -n "$LIKWID_PIN_BIN" ]; then
                RUN_WRAP=("$LIKWID_PIN_BIN" -q -c "$CPULIST")
                PIN_MODE="likwid-pin:$CPULIST"
            else
                export OMP_PROC_BIND=spread OMP_PLACES=cores
                PIN_MODE="omp-proc-bind"
            fi
        else
            PIN_MODE="none"
        fi
    fi
fi

log "pinning: $PIN_MODE"

# prefix for likwid-measured runs too, but ONLY the srun variant
# (likwid-perfctr -C pins threads itself; no likwid-pin double-wrap)
LIKWID_PRE=()
[ "${RUN_WRAP[0]:-}" = srun ] && LIKWID_PRE=("${RUN_WRAP[@]}")

# ---------------------------------------------------------------------------
# meta.csv (per-instance cache bytes for the visualizer's locality bands)
# ---------------------------------------------------------------------------
cache_bytes() {  # $1 = lscpu pattern -> prints per-instance bytes (POSIX awk/sed)
    local line v u cnt mult
    line="$(lscpu 2>/dev/null | awk -F': *' -v pat="$1" '$0 ~ pat {print $2; exit}')"
    [ -n "$line" ] || { echo 0; return; }
    v="$(awk '{print $1}' <<<"$line")"
    v="${v%%.*}"                       # lscpu may print "1.1 MiB"
    [ -n "$v" ] || v=0
    u="$(awk '{print $2}' <<<"$line")"
    cnt="$(sed -nE 's/.*\(([0-9]+) instances?\).*/\1/p' <<<"$line")"
    [ -n "$cnt" ] || cnt=1
    case "$u" in
        KiB) mult=1024 ;;
        MiB) mult=1048576 ;;
        GiB) mult=1073741824 ;;
        *)   mult=1 ;;
    esac
    echo $(( v * mult / cnt ))
}
L1D=$(cache_bytes "L1d cache");  L1D=${L1D:-0}
L2=$(cache_bytes "L2 cache");    L2=${L2:-0}
L3=$(cache_bytes "L3 cache");    L3=${L3:-0}
SOCKS=$(lscpu 2>/dev/null | awk -F': *' '/^Socket\(s\)/{print $2}')
CPS=$(lscpu 2>/dev/null | awk -F': *' '/^Core\(s\) per socket/{print $2}')
TPC=$(lscpu 2>/dev/null | awk -F': *' '/^Thread\(s\) per core/{print $2}')
{
    echo "key,value"
    echo "arch,$ARCH"
    echo "cpu_name,$CPU_NAME"
    echo "hostname,$(hostname)"
    echo "threads,$THREADS"
    echo "cpulist,$CPULIST"
    echo "sockets,${SOCKS:-1}"
    echo "cores_per_socket,${CPS:-0}"
    echo "threads_per_core,${TPC:-1}"
    echo "l1d_bytes,$L1D"
    echo "l2_bytes,$L2"
    echo "l3_bytes,$L3"
    echo "gpu_kind,${GPU_KIND:-none}"
    echo "gpu_name,${GPU_NAMES:-}"
    echo "gpu_tag,${GPU_TAG:-}"
    echo "toolchain,$TOOLCHAIN"
    echo "use_likwid,$USE_LIKWID"
    echo "pin_mode,$PIN_MODE"
    echo "cpu_bind,$CPU_BIND"
    echo "pin_freq,$PIN_FREQ"
    echo "cpu_freq_spec,${CPUFREQ:-off}"
    echo "slurm_job,${SLURM_JOB_ID:-none}"
    echo "sizes,$SIZES"
    echo "datatypes,$DATATYPES"
    echo "repeats,$REPEATS"
} > "$OUT/meta.csv"

# ---------------------------------------------------------------------------
# CPU: CPU/GEMMBench ladder — one sweep step per cube size
#      plus BF16: dedicated AVX512-BF16 kernel (gemm_bf16_*), skipped at
#      runtime (exit 3) when the CPU has no AVX512_BF16.
# ---------------------------------------------------------------------------
run_cpu() {
    local DT="$1"
    local csv="$OUT/gemm_sweep_${DT}.csv"
    local BSUF log_f BIN
    BSUF="$(echo "${TOOLCHAIN}_LADDER" | tr '[:upper:]' '[:lower:]')"
    log_f="$OUT/raw/cpu_${DT}.log"
    rm -f "$csv"; : > "$log_f"

    if [ "$DT" = BF16 ]; then
        local cc
        case "$TOOLCHAIN" in
            GCC) cc=gcc ;; ICX|ICC) cc=icx ;; AOCC|AOCL) cc=clang ;;
            *) cc=gcc ;;
        esac
        BIN="CPU/GEMMBench/gemm_bf16_${BSUF}"
        log "CPU $ARCH / BF16: building bf16_bench ($cc -march=native)"
        "$cc" -O3 -fopenmp -march=native -o "$BIN" \
            CPU/GEMMBench/src/bf16_bench.c -lm >> "$log_f" 2>&1 \
            || { log "BF16 BUILD FAILED, see $log_f"; return 1; }
    else
        log "CPU $ARCH / $DT: building gemm_sweep_${BSUF}"
        make -s -C CPU/GEMMBench TOOLCHAIN="$TOOLCHAIN" OPTIMIZATION=LADDER DATATYPE="$DT" \
            >> "$log_f" 2>&1 || { log "BUILD FAILED ($DT), see $log_f"; return 1; }
        BIN="CPU/GEMMBench/gemm_sweep_${BSUF}"
    fi
    [ -x "$BIN" ] || { log "missing $BIN"; return 1; }

    local sz rc g GRPSET
    for sz in $SIZES; do
        rc=0
        GRPSET=""
        if [ "$USE_LIKWID" = 1 ]; then
            GRPSET="$(pick_groups "$DT")"
            for g in $GRPSET; do group_exists "$g" || GRPSET=""; done
        fi
        if [ -n "$GRPSET" ]; then
            for g in $GRPSET; do
                "${LIKWID_PRE[@]}" "$LIKWID_BIN" -C "$CPULIST" -g "$g" -o "$OUT/likwid_${DT}_${g}_${sz}.csv" \
                    "$BIN" -r "$REPEATS" -c "$csv" -e "$sz" \
                    >> "$log_f" 2>&1 || rc=$?
            done
        else
            "${RUN_WRAP[@]}" "$BIN" -e "$sz" -r "$REPEATS" -c "$csv" \
                >> "$log_f" 2>&1 || rc=$?
        fi
        if [ "$rc" = 3 ] && [ "$DT" = BF16 ]; then
            log "BF16 unsupported on $CPU_NAME (no AVX512_BF16) -> skipping BF16 sweep"
            rm -f "$csv"
            return 0
        fi
        [ "$rc" = 0 ] || log "size $sz ($DT): run had failures (rc=$rc)"
    done
    log "CPU $DT done -> $csv"
}

# ---------------------------------------------------------------------------
# GPU: GPU/GEMM ladder (mk/include_<tag>.mk always compiles with METRICS
#      → the CSV gains power/clock/temp/util columns)
# ---------------------------------------------------------------------------
run_gpu() {
    local DT="$1"
    local csv="$OUT/gpu_gemm_${DT}.csv"
    local log_f="$OUT/raw/gpu_${DT}.log"
    rm -f "$csv"; : > "$log_f"
    log "GPU $GPU_TAG / $DT: building GPU/GEMM"
    ( cd GPU/GEMM && make -s GPU="$GPU_TAG" DATATYPE="$DT" ) >> "$log_f" 2>&1 \
        || { log "GPU build failed, see $log_f"; return 1; }
    local sz g rc
    for sz in $SIZES; do
        rc=0
        if [ "$USE_LIKWID" = 1 ] && [ "$GPU_KIND" = nvidia ] && [ -n "$LIKWID_BIN" ]; then
            g="$([ "$DT" = FP32 ] && echo FLOPS_SP || echo FLOPS_DP)"
            "${LIKWID_PRE[@]}" "$LIKWID_BIN" -G 0 -W "$g" -o "$OUT/gpu_likwid_${DT}_${sz}.csv" \
                GPU/GEMM/gemm_sweep -e "$sz" -r "$REPEATS" -c "$csv" \
                >> "$log_f" 2>&1 || rc=$?
        else
            "${RUN_WRAP[@]}" GPU/GEMM/gemm_sweep -e "$sz" -r "$REPEATS" -c "$csv" \
                >> "$log_f" 2>&1 || rc=$?
        fi
        [ "$rc" = 0 ] || log "GPU size $sz ($DT) failed (rc=$rc)"
    done
    log "GPU $DT done -> $csv"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
log "node=$(hostname) arch=$ARCH cpu='$CPU_NAME' threads=$THREADS gpu='${GPU_KIND:-none}:${GPU_TAG:-}'"
log "USE_LIKWID=$USE_LIKWID sizes='$SIZES' datatypes='$DATATYPES' -> $OUT"

if [ "${SKIP_CPU:-0}" != "1" ]; then
    for dt in $DATATYPES; do run_cpu "$dt"; done
fi
if [ "${SKIP_GPU:-0}" != "1" ]; then
    if [ -n "$GPU_TAG" ]; then
        for dt in $DATATYPES; do run_gpu "$dt"; done
    else
        log "no GPU (or no mk/include_<tag>.mk match) -> skipping GPU section"
    fi
fi

log "ALL DONE -> $OUT"
ls -la "$OUT"
