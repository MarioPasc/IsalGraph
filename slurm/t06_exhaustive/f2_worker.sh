#!/usr/bin/env bash
# T06_exhaustive F2 / STATISTICS stage -- Picasso worker.
#
# One unit = one (suite, dataset) shard, matching experiments/paper_pipeline/run_f2.sh.
# Each shard writes its own partial, so a failure costs one dataset rather than
# the campaign and a rerun is a rerun of that shard.
#
# TWO MODES, selected by STAGE:
#   shards   run this task's slice of the fifteen shards
#   merge    merge every partial and emit family_F2.json / rho_table.json
# The launcher submits `merge` with --dependency=afterok on the shard array, so
# the merge cannot run on a partial set.
#
# THE REFERENCE ARM IS isalgraph_exhaustive, via T06_REFERENCE_ARM. The default
# is unchanged (isalgraph_pruned), so every T-06 artifact still reproduces
# byte-identically; this campaign overrides it and the override is echoed into
# the log so no number can be quoted without its arm.
#
# THE COMPARATOR SET IS FULL. graph6, sparse6 and adjacency stay in the data.
# COMPARATOR_SET=reduced emits the reduced reporting view as a SEPARATE run and
# is never the primary. Dropping a competitor from a table costs nothing;
# dropping it from the campaign changes the cardinality of a pre-registered
# confirmatory family.
#
# NO #SBATCH HEADER: the launcher supplies every resource flag.
set -uo pipefail

START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Stage:        ${STAGE}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID:-N/A} of ${N_TASKS:-?}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Reference arm: ${T06_REFERENCE_ARM}"
echo "Comparators:   ${COMPARATOR_SET}"
echo "Git commit:   $(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

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

# 🔴 REPO_DIR ONLY -- never src-first.
export PYTHONPATH="${REPO_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${LOCALSCRATCH:-/tmp}/${USER}/pycache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${PYTHONPYCACHEPREFIX}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# T06_REFERENCE_ARM is read at MODULE IMPORT time by t06_f2_inputs, t06_gates,
# t06_ladder and t06_censoring. Exporting it here is what re-points all four.
export T06_REFERENCE_ARM

# ============================================================================
# DATA ROOTS -- validated once, explicitly
# ============================================================================
# Every root that has a workstation default must be threaded through and
# checked. The encode array's first submission died 6/6 units in seven seconds
# because one of these fell through to a /media/... path, and the message named
# a directory that has never existed on the cluster.
fatal_root() {
    echo "[FATAL] $1 unset or missing: '${2:-}'" >&2
    echo "        It has a /media/... workstation default that does not exist here." >&2
    exit 2
}
[ -n "${GED_ROOT:-}" ]    && [ -d "${GED_ROOT}" ]    || fatal_root GED_ROOT "${GED_ROOT:-}"
[ -n "${APPROX_ROOT:-}" ] && [ -d "${APPROX_ROOT}" ] || fatal_root APPROX_ROOT "${APPROX_ROOT:-}"
[ -n "${OUT_ROOT:-}" ]    && [ -d "${OUT_ROOT}" ]    || fatal_root OUT_ROOT "${OUT_ROOT:-}"
echo "[roots] ged=${GED_ROOT}"
echo "[roots] approx=${APPROX_ROOT}"
echo "[roots] out=${OUT_ROOT}"

DIST="${OUT_ROOT}/distances"
ENC="${OUT_ROOT}/encodings"
FAM="${OUT_ROOT}/families"
PARTIALS="${FAM}/f2_partials"
COMPLETION="${OUT_ROOT}/completion_rates.json"
mkdir -p "${FAM}" "${PARTIALS}" "${OUT_ROOT}/logs"

# The reference arm must actually be present, or every shard logs "no reference
# arm, B rows skipped" and the campaign produces an empty family with rc=0.
if ! ls "${DIST}/suite1/"*"__${T06_REFERENCE_ARM}__levenshtein.npz" >/dev/null 2>&1 \
   && ! ls "${DIST}/suite2/"*"__${T06_REFERENCE_ARM}__levenshtein.npz" >/dev/null 2>&1; then
    echo "[FATAL] no ${T06_REFERENCE_ARM} levenshtein matrix under ${DIST}." >&2
    echo "        Every shard would skip its B rows and the family would come back" >&2
    echo "        empty with a zero exit status. Run the distance stage first." >&2
    exit 2
fi

# The same failure on the other axis, and it is the one that actually happened.
# Run 1 (2106063/2106064, 2026-08-26) staged distances/ and NOT encodings/: the
# tree held only the 30 cells the encode array wrote. Claim B read distances/ and
# was complete; Claim A read encodings/, found the reference arm alone, and all
# 15 shards wrote `a1_cells: []` while reporting ok=3 fail=0. The merge died
# 6 h 50 m later in the A2 post-hoc with "need >= 2 named methods, got 1".
#
# A1 emits one cell per COMPARATOR, never per IsalGraph arm -- measured on
# suite2/linux, the 12 cells are 6 competitors (adjacency, agm_cam, graph6,
# min_dfs, nauty_graph6, sparse6) x 2 arms (primary, complete_case). So the test
# is not "how many representations" -- run 1 had two and still emitted nothing --
# it is "is there anything here that is not one of ours".
#
# `find`, not `ls`: a cell may legitimately be a symlink into T06/encodings/, and
# an `ls` that renders "name -> target" would feed the target path to sed and
# count nonsense. find prints the path and nothing else.
ENC_REPS=$(find "${ENC}" -maxdepth 2 -name '*__*.npz' 2>/dev/null \
    | sed 's/.*__//; s/\.npz$//' | sort -u)
# awk 'NF', not grep -c: an empty ENC_REPS still makes printf emit one blank
# line, and `grep -cv '^isalgraph_'` counts that blank as a comparator -- so an
# EMPTY encodings tree, the worst case of all, would sail past this guard.
N_ENC_REPS=$(printf '%s\n' "${ENC_REPS}" | awk 'NF' | wc -l)
N_ENC_COMPARATORS=$(printf '%s\n' "${ENC_REPS}" | awk 'NF && $0 !~ /^isalgraph_/' | wc -l)
if [ "${N_ENC_COMPARATORS}" -lt 1 ]; then
    echo "[FATAL] ${N_ENC_REPS} representation(s) under ${ENC} and NOT ONE comparator." >&2
    echo "        Claim A emits one cell per comparator, so every shard would write" >&2
    echo "        a1_cells=[] and still report ok, and the merge would fail hours" >&2
    echo "        later in the A2 post-hoc with 'need >= 2 named methods, got 1'." >&2
    echo "        Stage the competitor encodings with 'rsync -aL' so the Sandisk" >&2
    echo "        symlinks into T06/encodings/ resolve into real files here." >&2
    exit 2
fi
echo "[input] encodings: ${N_ENC_REPS} representations, ${N_ENC_COMPARATORS} comparators"

# Completion rates describe THIS arm. The T-06 file describes the pruned arm and
# would silently mis-describe this one.
#
# 🔴 AND THEY DESCRIBE WHATEVER ENC HELD WHEN THEY WERE WRITTEN. A plain
# `[ ! -s ]` test reuses a file built from a smaller tree, which is how run 2
# (2106210/2106211, 2026-08-26) reported N_actual=86: the file had been generated
# during run 1 when ENC held only the 30 new cells, so it carried rows for
# isalgraph_exhaustive and isalgraph_greedy ALONE. `c` is the count of cells
# excluded for completion, computed at merge time from this file -- with no
# competitor row present, nothing could be excluded and c came out 0 against
# T-06's 7. The family was internally consistent (discrepancy=0) and wrong.
#
# So: regenerate whenever the file covers fewer representations than ENC does.
if [ -s "${COMPLETION}" ]; then
    N_COMP_REPS=$("$PY" - "${COMPLETION}" <<'PYEOF' 2>/dev/null || echo 0
import json, sys
reps = set()
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "representation" and isinstance(v, str):
                reps.add(v)
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(json.load(open(sys.argv[1])))
print(len(reps))
PYEOF
)
    if [ "${N_COMP_REPS}" -lt "${N_ENC_REPS}" ]; then
        echo "[stale] ${COMPLETION} covers ${N_COMP_REPS} representations but ${ENC}" >&2
        echo "        holds ${N_ENC_REPS}. Regenerating; a short completion table makes" >&2
        echo "        c too small and N_actual too large, with discrepancy still 0." >&2
        mv "${COMPLETION}" "${COMPLETION%.json}_stale_$(date +%Y%m%dT%H%M%SZ).json"
    fi
fi

if [ ! -s "${COMPLETION}" ]; then
    "$PY" -m benchmarks.real_data.eval_encoding.t06_completion \
        --encodings "${ENC}" --out "${COMPLETION}" \
        || { echo "[FATAL] completion rates failed" >&2; exit 2; }
fi
echo "[input] completion rates: ${COMPLETION}"

f2() {
    "$PY" -m benchmarks.real_data.eval_stats.t06_f2 \
        --distances "${DIST}" \
        --encodings "${ENC}" \
        --completion-rates "${COMPLETION}" \
        --ged-root "${GED_ROOT}" \
        --approx-root "${APPROX_ROOT}" \
        --out-dir "${FAM}" \
        --comparator-set "${COMPARATOR_SET}" \
        "$@"
}

fail=0; ok=0; skip=0

if [ "${STAGE}" = "shards" ]; then
    IFS=':' read -r -a UNITS <<< "${UNIT_LIST}"
    N_UNITS=${#UNITS[@]}
    TASK_IDX=$(( ${SLURM_ARRAY_TASK_ID:-0} ))

    # Strided, matching the encode and distance workers: the list is ordered
    # expensive-first (mutagenicity and coil_del are the critical path), so a
    # contiguous split would hand task 0 every heavy shard.
    MY_UNITS=()
    for (( i = TASK_IDX; i < N_UNITS; i += N_TASKS )); do MY_UNITS+=( "${UNITS[$i]}" ); done
    echo "[decode] ${N_UNITS} shards, ${N_TASKS} tasks -> this task runs ${#MY_UNITS[@]}: ${MY_UNITS[*]}"
    [ "${#MY_UNITS[@]}" -eq 0 ] && { echo "[decode] empty slice"; exit 0; }

    for (( i = 0; i < ${#MY_UNITS[@]}; i++ )); do
        unit="${MY_UNITS[$i]}"
        suite="${unit%%/*}"; dataset="${unit##*/}"
        partial="${PARTIALS}/${suite}__${dataset}.json"
        echo "--- shard ${i}: ${suite}/${dataset}"
        if [ -s "${partial}" ]; then
            skip=$(( skip + 1 )); echo "    [skip] partial exists"; continue
        fi
        ELAPSED=$(( $(date +%s) - START_TIME ))
        if [ "$i" -gt 0 ] && [ "${ELAPSED}" -ge "${START_CUTOFF_S}" ]; then
            echo "    [defer] ${ELAPSED}s >= ${START_CUTOFF_S}s cutoff"; continue
        fi
        t0=$(date +%s)
        if f2 --suites "${suite}" --datasets "${dataset}" --emit-partial "${partial}" \
             >"${OUT_ROOT}/logs/f2_${suite}__${dataset}.log" 2>&1 && [ -s "${partial}" ]; then
            ok=$(( ok + 1 )); echo "    [ok] in $(( $(date +%s) - t0 )) s"
        else
            fail=$(( fail + 1 )); echo "    [FAIL] see logs/f2_${suite}__${dataset}.log"
        fi
    done

elif [ "${STAGE}" = "merge" ]; then
    have=$(ls -1 "${PARTIALS}"/*.json 2>/dev/null | wc -l)
    echo "[merge] ${have} partials of ${EXPECTED_SHARDS} expected"
    if [ "${have}" -ne "${EXPECTED_SHARDS}" ]; then
        echo "[FATAL] merging an incomplete partial set would silently shrink the family." >&2
        exit 2
    fi
    if f2 --merge-partials "${PARTIALS}"; then ok=1; else fail=1; fi

    n_actual=$("$PY" -c "import json;print(json.load(open('${FAM}/family_F2.json'))['cardinality']['n_actual'])" 2>/dev/null || echo 0)
    discrep=$("$PY" -c "import json;print(json.load(open('${FAM}/family_F2.json'))['cardinality']['discrepancy'])" 2>/dev/null || echo 999)
    rows=$("$PY" -c "import json;print(json.load(open('${FAM}/rho_table.json'))['n_rows'])" 2>/dev/null || echo 0)

    # N_actual is REPORTED, not asserted at 79. N_actual = 101 - 5*3 - c and c is
    # the completion-based exclusions; the exhaustive arm has a different
    # completion profile, so c can legitimately move and a moved value here is a
    # FINDING, not a defect. Aborting on it would discard the result the campaign
    # exists to produce. T-06's own 79 lives in source/T06 and is not rewritten.
    echo "=== N_actual: ${n_actual} (T-06 measured 79 on the pruned arm) ==="
    echo "=== rho rows: ${rows} ==="
    # The discrepancy IS asserted: it compares the enumeration against the closed
    # form WITHIN this run, so a non-zero value is an internal inconsistency
    # whatever the arm.
    echo "=== discrepancy: ${discrep} (expect 0) ==="
    [ "${discrep}" -eq 0 ] || { echo "!!! enumeration disagrees with the closed form"; fail=$(( fail + 1 )); }
    [ "${rows}" -gt 0 ] || { echo "!!! rho_table.json is empty"; fail=$(( fail + 1 )); }
else
    echo "[FATAL] unknown STAGE '${STAGE}'; expected 'shards' or 'merge'" >&2
    exit 2
fi

END_TIME=$(date +%s); ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $(( ELAPSED / 3600 ))h $(( (ELAPSED / 60) % 60 ))m $(( ELAPSED % 60 ))s"
echo "DONE_MARKER stage=${STAGE} task=${SLURM_ARRAY_TASK_ID:-0} ok=${ok} skip=${skip} fail=${fail}"
[ "${fail}" -eq 0 ] || exit 1
