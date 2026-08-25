#!/usr/bin/env bash
# T06_exhaustive DISTANCE stage -- Picasso worker.
#
# One unit = one (suite, dataset, arm). Each unit computes BOTH of that cell's
# new matrices, because they share the same encodings file and splitting them
# would double the I/O for no scheduling gain:
#
#   levenshtein   symbol-level, per CONTRACTS 3.1. The unit of edit is
#                 Encoding.symbols, never characters.
#   size_null     ONE PER (representation, dataset), per CONTRACTS 4.1. The new
#                 arms censor a different set of graphs from the pruned arm, so
#                 each needs its OWN null -- sharing one would compute the
#                 baseline over pairs the arm was never evaluated on, which is
#                 the exact defect 4.1 was written against.
#
# NO GED MATRIX IS RECOMPUTED and NO COMPETITOR DISTANCE IS RECOMPUTED. Both are
# unchanged by this work and are reused verbatim; the worker asserts they are
# reachable rather than silently producing a tree F2 cannot use.
#
# This track NEVER opens a cohort file -- distance_runner's docstring is explicit
# that node counts travel inside the encodings .npz precisely so the ownership
# sets stay disjoint. So no COHORT_ROOT here, deliberately, unlike the encode
# worker.
#
# NO #SBATCH HEADER: the launcher supplies every resource flag, matching the
# encode pair.
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

PY="${CONDA_PREFIX_DIR}/bin/python"
[ -x "$PY" ] || { echo "[FATAL] no interpreter at $PY"; exit 2; }

cd "${REPO_DIR}" || { echo "[FATAL] no repo at ${REPO_DIR}"; exit 2; }

# 🔴 REPO_DIR ONLY -- never "${REPO_DIR}/src". A src-first path shadows the
# installed package and silently drops the engine to pure Python. Verified on
# the cluster: the build installs into site-packages.
export PYTHONPATH="${REPO_DIR}"
export PYTHONUNBUFFERED=1

# Picasso exports a SHARED PYTHONPYCACHEPREFIX per user; concurrent tasks on one
# node race writing identical .pyc paths and a fraction die with an intermittent
# ModuleNotFoundError on a module that is present.
export PYTHONPYCACHEPREFIX="${LOCALSCRATCH:-/tmp}/${USER}/pycache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${PYTHONPYCACHEPREFIX}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" - <<'EOF' || exit 2
import sys
import isalgraph
print(f"[engine] {isalgraph.engine()}")
print(f"[build ] {isalgraph.build_info()}")
if isalgraph.engine() != "cpp":
    print("[FATAL] engine is not 'cpp'; refusing to compute a production matrix.", file=sys.stderr)
    sys.exit(2)
EOF

# ============================================================================
# GATES -- both are preconditions, not formalities
# ============================================================================
OUT_ROOT="${OUT_ROOT}"
DIST="${OUT_ROOT}/distances"
GATE="${OUT_ROOT}/gates/gate_T06_reproduction.json"

if [ ! -s "${GATE}" ]; then
    echo "[FATAL] reproduction gate missing at ${GATE}" >&2
    echo "        T-04a's corrected table must be reproduced at max |delta| = 0.0000" >&2
    echo "        before any production matrix is computed. Stage T06/gates/ first." >&2
    exit 2
fi
"$PY" - "${GATE}" <<'EOF' || exit 2
import json, sys
r = json.load(open(sys.argv[1]))
ok = r["passed"] and r["max_abs_delta_observed"] == 0.0
print(f" reproduction gate: {'PASS' if ok else 'FAIL'} "
      f"({r['numeric_values_compared']} values, max |delta| = {r['max_abs_delta_observed']:.10f})")
sys.exit(0 if ok else 1)
EOF

# The competitor matrices are REUSED, never recomputed. If they are not
# reachable the tree is incomplete and F2 would silently run on a reduced
# comparator set -- which is exactly the cardinality change the campaign must
# not make by accident.
missing=0
for s in suite1 suite2; do
    n=$(ls -1 "${DIST}/${s}"/*__levenshtein.npz 2>/dev/null | grep -vc "__isalgraph_" || true)
    echo "[reuse] ${s}: ${n} competitor levenshtein matrices reachable"
    [ "${n}" -eq 0 ] && missing=$(( missing + 1 ))
done
if [ "${missing}" -gt 0 ]; then
    echo "[FATAL] no competitor distance matrices under ${DIST}." >&2
    echo "        They are REUSED verbatim from data/source/T06/distances and must be" >&2
    echo "        staged (copied or symlinked) into this tree before F2 can run." >&2
    exit 2
fi

# ============================================================================
# UNIT LIST -- strided, not contiguous
# ============================================================================
# The launcher orders units expensive-first, so a count-equal contiguous split
# hands task 0 every expensive cell while the last task finishes in seconds.
# Striding deals the costly units round-robin; counts stay within one unit.
IFS=':' read -r -a UNITS <<< "${UNIT_LIST}"
N_UNITS=${#UNITS[@]}
TASK_IDX=$(( ${SLURM_ARRAY_TASK_ID:-0} ))

MY_UNITS=()
for (( i = TASK_IDX; i < N_UNITS; i += N_TASKS )); do
    MY_UNITS+=( "${UNITS[$i]}" )
done

echo "[decode] ${N_UNITS} units, ${N_TASKS} tasks -> this task runs ${#MY_UNITS[@]}: ${MY_UNITS[*]}"
[ "${#MY_UNITS[@]}" -eq 0 ] && { echo "[decode] empty slice"; exit 0; }

mkdir -p "${DIST}/suite1" "${DIST}/suite2" "${OUT_ROOT}/logs"

fail=0; ok=0; skip=0
for (( i = 0; i < ${#MY_UNITS[@]}; i++ )); do
    unit="${MY_UNITS[$i]}"
    suite="${unit%%/*}"; rest="${unit#*/}"
    dataset="${rest%%/*}"; rep="${rest##*/}"
    enc="${OUT_ROOT}/encodings/${suite}/${dataset}__${rep}.npz"
    echo "--- unit ${i}: ${suite}/${dataset}/${rep}"

    if [ ! -s "${enc}" ]; then
        echo "    [skip] no encoding at ${enc}"; skip=$(( skip + 1 )); continue
    fi

    # Start-cutoff: never START a unit whose budget no longer fits, but ALWAYS
    # run index 0 of this task's own slice. Without the exemption a task whose
    # start-up alone exceeds the cutoff defers its whole slice, a resume pass
    # derives the same slice and defers it again, and the array livelocks.
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$i" -gt 0 ] && [ "${ELAPSED}" -ge "${START_CUTOFF_S}" ]; then
        echo "    [defer] ${ELAPSED}s >= ${START_CUTOFF_S}s cutoff; a resume pass takes it"
        continue
    fi

    lev="${DIST}/${suite}/${dataset}__${rep}__levenshtein.npz"
    if [ -s "${lev}" ]; then
        skip=$(( skip + 1 )); echo "    [skip] lev exists"
    else
        t0=$(date +%s)
        if "$PY" -m benchmarks.real_data.eval_distance.distance_runner \
             --encodings "${enc}" --metric levenshtein --out "${DIST}/${suite}" \
             --n-chunks 1 --chunk-index 0 --jobs "${DIST_JOBS}" --suite "${suite}" \
           && [ -s "${lev}" ]; then
            ok=$(( ok + 1 )); echo "    [ok] lev in $(( $(date +%s) - t0 )) s"
            rm -f "${DIST}/${suite}/${dataset}__${rep}__levenshtein.shard"*.npz
        else
            fail=$(( fail + 1 )); echo "    [FAIL] lev"
        fi
    fi

    null="${DIST}/${suite}/${dataset}__${rep}__size_null.npz"
    if [ -s "${null}" ]; then
        skip=$(( skip + 1 )); echo "    [skip] null exists"
    elif "$PY" -m benchmarks.real_data.eval_distance.size_null \
           --encodings "${enc}" --out "${DIST}/${suite}" --suite "${suite}" \
         && [ -s "${null}" ]; then
        ok=$(( ok + 1 )); echo "    [ok] size_null"
    else
        fail=$(( fail + 1 )); echo "    [FAIL] size_null"
    fi
done

END_TIME=$(date +%s); ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $(( ELAPSED / 3600 ))h $(( (ELAPSED / 60) % 60 ))m $(( ELAPSED % 60 ))s"
echo "DONE_MARKER stage=distances task=${TASK_IDX} ok=${ok} skip=${skip} fail=${fail}"
[ "${fail}" -eq 0 ] || exit 1
