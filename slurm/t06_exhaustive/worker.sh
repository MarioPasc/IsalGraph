#!/usr/bin/env bash
# T06_exhaustive encoding campaign -- Picasso worker.
#
# One array task runs a contiguous SLICE of the unit list, sequentially.
# A unit is one (suite, dataset, representation) cell. See launcher.sh for why
# the array is sized in tasks rather than in units.
#
# NO #SBATCH HEADER, deliberately. The launcher supplies every resource flag on
# the sbatch command line -- the IsalGraph convention, and the right one when a
# single worker serves several resource profiles. A header here would silently
# win over the launcher for any flag it also names.
set -uo pipefail

START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID:-N/A} of ${N_TASKS:-?}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "CPU model:    $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
echo "Git commit:   $(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

# ============================================================================
# ENVIRONMENT
# ============================================================================
module_loaded=0
for m in miniconda/3 miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>&1 | grep -qiE "(^|/)${m}([[:space:]]|/|$)"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module; using the env's python directly."

# The env is given as an absolute PREFIX, so the interpreter is addressed
# directly. conda is not on PATH on every node and `conda activate` on a prefix
# is the step most likely to fail silently into the base interpreter.
PY="${CONDA_PREFIX_DIR}/bin/python"
[ -x "$PY" ] || { echo "[FATAL] no interpreter at $PY"; exit 2; }

cd "${REPO_DIR}" || { echo "[FATAL] no repo at ${REPO_DIR}"; exit 2; }

# 🔴 REPO_DIR ONLY -- never "${REPO_DIR}/src".
# The generic Picasso template puts src/ first. For THIS project that is wrong
# and silently so: the C++ extension installs into site-packages, so a src-first
# path shadows the installed package, `isalgraph.engine()` falls back to
# 'python', and the campaign measures the pure-Python reference at ~1/100 the
# speed with every timing and every censoring rate wrong. src/ is not needed
# here anyway -- the package is pip-installed; PYTHONPATH exists only so
# `benchmarks.*` imports.
export PYTHONPATH="${REPO_DIR}"
export PYTHONUNBUFFERED=1

# Picasso exports a SHARED PYTHONPYCACHEPREFIX per user, so concurrent tasks on
# one node race writing identical .pyc paths and a fraction of them die with an
# intermittent ModuleNotFoundError on a module that is present. Per-task prefix.
export PYTHONPYCACHEPREFIX="${LOCALSCRATCH:-/tmp}/${USER}/pycache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${PYTHONPYCACHEPREFIX}"

# The encoder is single-threaded by design (measured: 4 threads are 1.8x SLOWER
# at n=6). Pin the numeric libraries so they cannot oversubscribe the allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ============================================================================
# ENGINE GATE -- abort rather than measure the wrong encoder
# ============================================================================
# The .so installs into site-packages and does NOT rsync; it must be built on
# the cluster. Falling back to Python here would not fail, it would just be
# 23-1025x slower and produce a completely different censoring rate at the same
# nominal budget -- a wrong number with no error.
"$PY" - <<'EOF' || exit 2
import sys
import isalgraph
info = isalgraph.build_info()
print(f"[engine] {isalgraph.engine()}")
print(f"[build ] {info}")
if isalgraph.engine() != "cpp":
    print("[FATAL] engine is not 'cpp'. The extension is not built in this env.", file=sys.stderr)
    print("        Build it on the cluster: pip install -e '.[dev,native]'", file=sys.stderr)
    print("        Flags must be -march=x86-64-v3, NEVER -march=native:", file=sys.stderr)
    print("        Picasso is heterogeneous and native produces SIGILL on some nodes.", file=sys.stderr)
    sys.exit(2)
EOF

# ============================================================================
# UNIT LIST -- decode this task's slice
# ============================================================================
# Shipped colon-separated: --export splits on COMMAS, so a comma inside a value
# is truncated and its tail becomes a junk variable, with no error.
IFS=':' read -r -a UNITS <<< "${UNIT_LIST}"
N_UNITS=${#UNITS[@]}
TASK_IDX=$(( ${SLURM_ARRAY_TASK_ID:-0} ))

# STRIDED, not contiguous. An equal split by COUNT is not an equal split by
# WORK here: the launcher orders units expensive-first, so contiguous blocks
# hand task 0 every expensive cell (protein, coil-del, mutagenicity, aids-iam,
# ... all exhaustive) and task 4 six greedy cells that finish in seconds. Task 0
# would then race the wall clock while four tasks idle.
#
# Striding over an expensive-first list deals the costly units round-robin, so
# each task gets at most one of the top-N. Counts stay within one unit of each
# other, which is what the 2 h floor actually needs.
MY_UNITS=()
for (( i = TASK_IDX; i < N_UNITS; i += N_TASKS )); do
    MY_UNITS+=( "${UNITS[$i]}" )
done

echo "[decode] ${N_UNITS} units, ${N_TASKS} tasks -> this task runs ${#MY_UNITS[@]}: ${MY_UNITS[*]}"
if [ "${#MY_UNITS[@]}" -eq 0 ]; then
    echo "[decode] empty slice; nothing to do"; exit 0
fi

OUT="${OUT_ROOT}"
mkdir -p "${OUT}/encodings/suite1" "${OUT}/encodings/suite2" "${OUT}/logs"

fail=0; ok=0; skip=0
for (( i = 0; i < ${#MY_UNITS[@]}; i++ )); do
    unit="${MY_UNITS[$i]}"
    suite="${unit%%/*}"; rest="${unit#*/}"
    dataset="${rest%%/*}"; rep="${rest##*/}"
    echo "--- unit ${i}: ${suite}/${dataset}/${rep}"

    target="${OUT}/encodings/${suite}/${dataset}__${rep}.npz"
    if [ -s "${target}" ]; then
        skip=$(( skip + 1 )); echo "    [skip] exists"; continue
    fi

    # Deadline: refuse to START a unit whose full budget no longer fits, but
    # ALWAYS run the first one. Without the exemption a task whose start-up
    # alone exceeds the cutoff defers its whole slice, a resume pass derives the
    # same slice and defers it again, and the array livelocks.
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$i" -gt 0 ] && [ "${ELAPSED}" -ge "${START_CUTOFF_S}" ]; then
        echo "    [defer] ${ELAPSED}s elapsed >= ${START_CUTOFF_S}s cutoff; a resume pass takes it"
        continue
    fi

    t0=$(date +%s)
    if "$PY" -m benchmarks.real_data.eval_encoding.t06_encode \
         --suite "${suite}" --dataset "${dataset}" --representation "${rep}" \
         --out "${OUT}" --budget-s "${BUDGET_S}" --jobs "${ENCODE_JOBS}" --require-cpp
    then
        t1=$(date +%s); ok=$(( ok + 1 )); echo "    [ok] in $(( t1 - t0 )) s"
    else
        fail=$(( fail + 1 )); echo "    [FAIL] ${suite}/${dataset}/${rep}"
    fi
done

END_TIME=$(date +%s); ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $(( ELAPSED / 3600 ))h $(( (ELAPSED / 60) % 60 ))m $(( ELAPSED % 60 ))s"
echo "DONE_MARKER task=${TASK_IDX} ok=${ok} skip=${skip} fail=${fail}"
[ "${fail}" -eq 0 ] || exit 1
