#!/usr/bin/env bash
# Submit the T-03 exact-GED programme to Picasso.
#
#   bash launcher.sh --dry-run --stage all --sec-per-pair 15
#   bash launcher.sh --stage gates
#   bash launcher.sh --stage aids --sec-per-pair 15        # stages 1 and 2 together
#
# Design: .claude/notes/review/tasks/T-03-design.md  (read the Changelog first)
#
# The single human entry point. Every sbatch flag lives here; the workers carry no
# #SBATCH header, so one launcher can dispatch heterogeneous jobs without four headers
# drifting apart. This follows the existing IsalGraph convention.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- configuration
export CONDA_ENV_PREFIX="/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalgraph"
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph"
export GEDLIB_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/build_gedlib/graphkit-learn"
export DATA_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph"
export OUT_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/exact_ged"
export LOGS_DIR="${OUT_DIR}/logs"
ACCOUNT="tic_163_uma"

# Pin the node family. Per-pair wall time is a REPORTED quantity for the D12 censoring
# analysis, so a mixed Intel/AMD pool would make the timing a measurement of the
# scheduler rather than of the solver. sr = 128 cores / 450 GB, the largest homogeneous
# idle pool measured 2026-08-12 (~45 nodes).
CONSTRAINT="sr"
CORES_PER_NODE=128

# SCBI's two-hour floor (Manuel, soporte@scbi.uma.es, 2026-08-07): a job must run >= 2 h
# or it costs the scheduler more to place than to run. This account has already been
# written to about it. TARGET_HOURS carries margin above the floor.
FLOOR_SECONDS=7200
TARGET_HOURS=2.5

STAGE="all"
SEC_PER_PAIR=""
MAX_CONCURRENT=8
DRY_RUN=false
TIMEOUT_PER_PAIR=60          # the submission's ged_computer default; recorded, not chosen

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=true; shift ;;
        --stage)          STAGE="$2"; shift 2 ;;
        --sec-per-pair)   SEC_PER_PAIR="$2"; shift 2 ;;
        --target-hours)   TARGET_HOURS="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --timeout-per-pair) TIMEOUT_PER_PAIR="$2"; shift 2 ;;
        -h|--help)        sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
case "${STAGE}" in gates|small|aids|merge|all) ;; *) echo "bad --stage ${STAGE}" >&2; exit 2 ;; esac

TARGET_SECONDS=$(python3 -c "print(int(${TARGET_HOURS}*3600))")

# ---------------------------------------------------------------- helpers
# Picasso's Lua sbatch wrapper writes ANSI codes and a multi-line warning to stdout, so
# --parsable does NOT return just the id. A line-wise sed leaves a multi-line "id" and
# the guard then fires AFTER submission, leaving an untracked job on the cluster. Take
# the LAST line first, then strip.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

submit() {                       # echoes a verified-numeric job id, or fails
    if ${DRY_RUN}; then
        printf '[DRY-RUN] sbatch --parsable' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        echo "000000"
        return 0
    fi
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}

assert_dependency_took() {       # a bad id is ACCEPTED and recorded Dependency=(null)
    local id="$1"
    ${DRY_RUN} && return 0
    if scontrol show job "${id}" | grep -q 'Dependency=(null)'; then
        echo "FATAL: dependency dropped on ${id}; cancelling" >&2
        scancel "${id}"; return 1
    fi
}

# cores such that a single task clears the floor: ceil(core_seconds / target_seconds)
cores_for_single_task() {        # $1 = total core-seconds
    python3 -c "
import math,sys
cs=float(sys.argv[1]); tgt=${TARGET_SECONDS}; cap=${CORES_PER_NODE}
print(max(1, min(cap, math.ceil(cs/tgt))))" "$1"
}

# tasks such that EVERY task clears the floor, given fixed cores per task
tasks_for_array() {              # $1 = total core-seconds, $2 = cores per task
    python3 -c "
import math,sys
cs=float(sys.argv[1]); cores=int(sys.argv[2])
tgt=${TARGET_SECONDS}; floor=${FLOOR_SECONDS}
n=max(1, int(cs//(cores*tgt)))
while n>1 and cs/(cores*n) < floor: n-=1     # refuse to put a task under the floor
print(n, int(math.ceil(cs/(cores*n))))" "$1" "$2"
}

need_rate() {
    [[ -n "${SEC_PER_PAIR}" ]] || {
        echo "FATAL: --sec-per-pair is required for stage '${STAGE}'." >&2
        echo "       Take it from the gate 3 report; do not guess. Measured on real" >&2
        echo "       AIDS 2026-08-12: median 6.5 s, mean 10.1 s among completions," >&2
        echo "       ~27% exceeding 30 s -> ~15 s/pair expected at a 60 s timeout." >&2
        exit 2; }
}

COMMON_EXPORT="ALL,CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX},REPO_DIR=${REPO_DIR},GEDLIB_DIR=${GEDLIB_DIR},DATA_DIR=${DATA_DIR},OUT_DIR=${OUT_DIR},TIMEOUT_PER_PAIR=${TIMEOUT_PER_PAIR}"
# NOTE: --export splits on EVERY comma, so no value above may contain one. All values
# here are single paths or integers. Any list-valued variable must be shipped
# colon-separated and translated in the worker.

mkdir -p "${LOGS_DIR}" 2>/dev/null || true
echo "stage=${STAGE}  constraint=${CONSTRAINT}  target=${TARGET_HOURS}h  floor=$((FLOOR_SECONDS/3600))h  timeout/pair=${TIMEOUT_PER_PAIR}s"
echo "logs -> ${LOGS_DIR}"
echo

GATES_ID=""; SMALL_ID=""; S1_ID=""; S2_ID=""

# ---------------------------------------------------------------- 1. gates
if [[ "${STAGE}" == "gates" || "${STAGE}" == "all" ]]; then
    GATES_ID=$(submit \
        --job-name=ged-gates --account="${ACCOUNT}" \
        --time=0-04:00:00 --ntasks=1 --cpus-per-task=4 --mem=32G \
        --constraint="${CONSTRAINT}" \
        --output="${LOGS_DIR}/gates_%j.out" --error="${LOGS_DIR}/gates_%j.err" \
        --export="${COMMON_EXPORT}" \
        "${SCRIPT_DIR}/worker_gates.sh")
    echo "gates      -> ${GATES_ID}   (BLOCKS everything; must exit 0)"
fi

# ---------------------------------------------------------------- 2. small suite
# Letter LOW/MED/HIGH + LINUX. This is the end-to-end proof on real data before AIDS.
# LINUX alone would be ~18 min on 8 cores and would violate the floor, so the four
# datasets run in one task -- which is exactly the grouping SCBI asked for.
if [[ "${STAGE}" == "small" || "${STAGE}" == "all" ]]; then
    need_rate
    # measured: Letter ~0.004-0.008 s/pair, LINUX 2.48 s/pair (local smoke, 12 workers)
    SMALL_CS=$(python3 -c "print(695610*0.004 + 784378*0.004 + 2118711*0.008 + 3916*2.48)")
    SMALL_CORES=$(cores_for_single_task "${SMALL_CS}")
    echo "small      core-seconds=${SMALL_CS%.*}  -> ${SMALL_CORES} cores"
    DEP=(); [[ -n "${GATES_ID}" ]] && DEP=(--dependency="afterok:${GATES_ID}")
    SMALL_ID=$(submit \
        --job-name=ged-small --account="${ACCOUNT}" \
        --time=0-08:00:00 --ntasks=1 --cpus-per-task="${SMALL_CORES}" --mem=64G \
        --constraint="${CONSTRAINT}" "${DEP[@]}" \
        --output="${LOGS_DIR}/small_%j.out" --error="${LOGS_DIR}/small_%j.err" \
        --export="${COMMON_EXPORT},N_WORKERS=${SMALL_CORES}" \
        "${SCRIPT_DIR}/worker_small.sh")
    echo "small      -> ${SMALL_ID}"
    [[ -n "${GATES_ID}" ]] && assert_dependency_took "${SMALL_ID}"
fi

# ---------------------------------------------------------------- 3+4. AIDS
# Stages 1 and 2 are submitted TOGETHER -- signed decision 21. Stage 1 is the
# pre-declared reported analysis; the census runs unattended behind it and supersedes
# stage 1 only if it lands before the T-20 freeze, on the calendar, not on the values.
if [[ "${STAGE}" == "aids" || "${STAGE}" == "all" ]]; then
    need_rate
    S1_PAIRS=22106                                    # from ged_sampling, seed 42
    S2_PAIRS=295296                                   # C(769,2)
    S1_CS=$(python3 -c "print(${S1_PAIRS}*${SEC_PER_PAIR})")
    S1_CORES=$(cores_for_single_task "${S1_CS}")
    read -r S2_TASKS S2_WALL < <(tasks_for_array "$(python3 -c "print(${S2_PAIRS}*${SEC_PER_PAIR})")" 64)
    S2_WALL_H=$(python3 -c "print(f'{${S2_WALL}/3600:.2f}')")

    echo "aids s1    ${S1_PAIRS} pairs, core-seconds=${S1_CS%.*} -> ${S1_CORES} cores, $(python3 -c "print(f'{${S1_CS}/${S1_CORES}/3600:.2f}')") h"
    echo "aids s2    ${S2_PAIRS} pairs -> ${S2_TASKS} tasks x 64 cores, ${S2_WALL_H} h/task"
    if (( S2_WALL < FLOOR_SECONDS )); then
        echo "FATAL: projected ${S2_WALL}s/task is under the ${FLOOR_SECONDS}s floor." >&2
        echo "       Reduce the task count rather than submitting short jobs." >&2
        exit 3
    fi

    DEP=(); [[ -n "${GATES_ID}" ]] && DEP=(--dependency="afterok:${GATES_ID}")
    S1_ID=$(submit \
        --job-name=ged-aids-s1 --account="${ACCOUNT}" \
        --time=0-12:00:00 --ntasks=1 --cpus-per-task="${S1_CORES}" --mem=64G \
        --constraint="${CONSTRAINT}" "${DEP[@]}" \
        --output="${LOGS_DIR}/aids_s1_%j.out" --error="${LOGS_DIR}/aids_s1_%j.err" \
        --export="${COMMON_EXPORT},N_WORKERS=${S1_CORES},AIDS_STAGE=1,N_CHUNKS=1,CHUNK_OFFSET=0" \
        "${SCRIPT_DIR}/worker_aids.sh")
    echo "aids s1    -> ${S1_ID}"
    [[ -n "${GATES_ID}" ]] && assert_dependency_took "${S1_ID}"

    # Stage 2 waits on stage 1 so it can seed from it -- the overlap is computed once,
    # and agreement on the shared pairs is asserted at merge.
    S2_ID=$(submit \
        --job-name=ged-aids-s2 --account="${ACCOUNT}" \
        --array="0-$((S2_TASKS-1))%${MAX_CONCURRENT}" \
        --time=0-16:00:00 --ntasks=1 --cpus-per-task=64 --mem=64G \
        --constraint="${CONSTRAINT}" --dependency="afterok:${S1_ID}" \
        --output="${LOGS_DIR}/aids_s2_%A_%a.out" --error="${LOGS_DIR}/aids_s2_%A_%a.err" \
        --export="${COMMON_EXPORT},N_WORKERS=64,AIDS_STAGE=2,N_CHUNKS=${S2_TASKS},CHUNK_OFFSET=0" \
        "${SCRIPT_DIR}/worker_aids.sh")
    echo "aids s2    -> ${S2_ID}   (array 0-$((S2_TASKS-1))%${MAX_CONCURRENT})"
    assert_dependency_took "${S2_ID}"
fi

# ---------------------------------------------------------------- 5. merge
if [[ "${STAGE}" == "merge" || "${STAGE}" == "all" ]]; then
    DEPS=()
    for id in "${SMALL_ID}" "${S2_ID}"; do [[ -n "${id}" ]] && DEPS+=("${id}"); done
    DEP=()
    if [[ ${#DEPS[@]} -gt 0 ]]; then
        DEP=(--dependency="afterany:$(IFS=:; echo "${DEPS[*]}")")
    fi
    MERGE_ID=$(submit \
        --job-name=ged-merge --account="${ACCOUNT}" \
        --time=0-02:00:00 --ntasks=1 --cpus-per-task=2 --mem=32G \
        --constraint="${CONSTRAINT}" "${DEP[@]}" \
        --output="${LOGS_DIR}/merge_%j.out" --error="${LOGS_DIR}/merge_%j.err" \
        --export="${COMMON_EXPORT}" \
        "${SCRIPT_DIR}/worker_merge.sh")
    echo "merge      -> ${MERGE_ID}"
    [[ ${#DEPS[@]} -gt 0 ]] && assert_dependency_took "${MERGE_ID}"
fi

echo
echo "monitor:  ssh picasso 'squeue'          # -u is REJECTED by Picasso's wrapper"
echo "states:   ssh picasso 'sacct -j <id> -X -n -P -o JobID,State,Elapsed,NodeList'"
echo "memory:   ssh picasso 'sacct -j <id> -n -P -o JobID,MaxRSS | grep .batch'   # NO -X"
echo "results:  ${OUT_DIR}"
