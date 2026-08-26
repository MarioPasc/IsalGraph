#!/usr/bin/env bash
# =============================================================================
# T-13 complexity campaign -- Picasso launcher
# =============================================================================
#
# Submits the controlled-experiment runner as a whole-node CPU array. One array
# task owns one node exclusively and runs SHARDS_PER_TASK single-threaded shards
# concurrently, each pinned to its own core with taskset.
#
# Usage:
#   bash slurm/launch.sh --source constructed            # submit
#   bash slurm/launch.sh --source cohort --dry-run       # print, do not submit
#   bash slurm/launch.sh --source constructed --test-only # sbatch --test-only
#
# Follows the IsalGraph convention: the worker carries NO #SBATCH header and
# every resource flag is supplied here, on the sbatch command line. See
# experiments/paper_pipeline/launch.sh.
#
# -----------------------------------------------------------------------------
# WHY ONE SHARD PER DEDICATED CORE, AND NOT A THREAD POOL
# -----------------------------------------------------------------------------
# Measured on this workload, parallelising inside the process is NEGATIVE-
# scaling: 1 worker 36 core-s, 4 -> 212, 15 -> 928, 32 -> 5,260 on identical
# work. Independent single-threaded processes, one per dedicated core, are the
# measured optimum and the only defensible fan-out. Every shard therefore runs
# with ISALGRAPH_THREADS=1 and every BLAS pinned to one thread -- time.process_time
# sums CPU across threads, so an unpinned library would inflate every reading by
# its thread count.
#
# -----------------------------------------------------------------------------
# SIZING, AND THE 2 h FLOOR
# -----------------------------------------------------------------------------
# SCBI asked this account in writing (Manuel, soporte@scbi.uma.es, 2026-08-07)
# that every submitted task run for at least two hours: placing a job costs the
# scheduler about the same whether it then runs for ten seconds or ten hours, so
# short tasks are almost pure overhead. A task below the floor will be refused.
#
#   CE (constructed) is estimated at 400-700 core-h (design note S6).
#   One node of the sr family carries 128 cores, so 128 shards.
#     400 core-h / 128 shards = 3.1 h per shard   >= 2 h  OK
#     700 core-h / 128 shards = 5.5 h per shard   >= 2 h  OK
#   Two nodes (256 shards) would give 1.6-2.7 h and BREAKS the floor at the
#   low estimate, so the fan-out stops at one node per source. The campaign is
#   two array tasks total (one per source), not two hundred.
#
#   Worst case is bounded by construction, since nothing runs past its budget:
#     units_per_shard x BUDGET_S. Hence the wallclock defaults below.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Picasso paths -- measured 2026-08-26. The C++ engine IS built in this env.
# ---------------------------------------------------------------------------
REPO_DIR="${ISALGRAPH_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph}"
CONDA_ENV_PATH="${ISALGRAPH_CONDA_ENV:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalgraph}"
COHORT_ROOT="${ISALGRAPH_COHORT_ROOT:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph}"
RESULTS_ROOT="${T13_RESULTS_ROOT:-/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/t13}"
LOGS_DIR="${T13_LOGS_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/logs}"
ACCOUNT="tic_163_uma"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# sr = AMD EPYC 7H12, 128 cores, 439 GB usable, 2.6 GHz base.
#
# THE FAMILY IS PINNED, AND THAT IS NOT OPTIONAL HERE. Wall-clock is the
# reported quantity of this entire ticket, and sd (Intel, 2.1 GHz) against sr
# (AMD, 2.6 GHz) differ enough that an unpinned pool would turn the cost law
# into a measurement of the scheduler. sr is chosen over bc (256 cores,
# 2.25 GHz) because a single-threaded latency benchmark is served better by the
# higher base clock than by more cores it cannot use.
NODE_FAMILY="${T13_NODE_FAMILY:-sr}"
SHARDS_PER_TASK="${T13_SHARDS_PER_TASK:-128}"
N_SHARDS="${T13_N_SHARDS:-128}"
BUDGET_S="${T13_BUDGET_S:-300}"
SEED="${T13_SEED:-13}"
ARMS="${T13_ARMS:-default}"
# The campaign runs the ladders alone at one replicate: design rule 7 makes the
# within-(n, m) ladder contrast the primary evidence, and the other nine
# families are supporting. Empty FAMILIES means the full 664-spec grid.
FAMILIES="${T13_FAMILIES:-}"
REPLICATES="${T13_REPLICATES:-}"
SOURCE=""
DRY_RUN=false
TEST_ONLY=false
TIME_LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)          SOURCE="$2"; shift 2 ;;
        --n-shards)        N_SHARDS="$2"; shift 2 ;;
        --shards-per-task) SHARDS_PER_TASK="$2"; shift 2 ;;
        --arms)            ARMS="$2"; shift 2 ;;
        --families)        FAMILIES="$2"; shift 2 ;;
        --replicates)      REPLICATES="$2"; shift 2 ;;
        --budget-s)        BUDGET_S="$2"; shift 2 ;;
        --seed)            SEED="$2"; shift 2 ;;
        --node-family)     NODE_FAMILY="$2"; shift 2 ;;
        --time)            TIME_LIMIT="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --test-only)       TEST_ONLY=true; shift ;;
        -h|--help)
            sed -n '2,40p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "${SOURCE}" != "constructed" && "${SOURCE}" != "cohort" ]]; then
    echo "ERROR: --source must be 'constructed' or 'cohort'" >&2
    exit 1
fi

# Constructed censors in the high-|Aut| cells and is bounded well under a day;
# the cohort arm is 16,370 graphs x 13 representations and needs more room.
if [[ -z "${TIME_LIMIT}" ]]; then
    if [[ "${SOURCE}" == "constructed" ]]; then TIME_LIMIT="1-00:00:00"; else TIME_LIMIT="2-00:00:00"; fi
fi

if (( N_SHARDS % SHARDS_PER_TASK != 0 )); then
    echo "ERROR: --n-shards ${N_SHARDS} is not a multiple of --shards-per-task ${SHARDS_PER_TASK}." >&2
    echo "       A ragged last task is exactly the short job SCBI asked us to stop submitting." >&2
    exit 1
fi
N_TASKS=$(( N_SHARDS / SHARDS_PER_TASK ))
ARRAY_MAX=$(( N_TASKS - 1 ))

# ---------------------------------------------------------------------------
# Job-ID capture.  Picasso's Lua sbatch wrapper writes ANSI codes and a
# multi-line warning to stdout, so --parsable does NOT return just the id.  Take
# the LAST line first: a line-by-line sed leaves the warning's newlines in place
# and a guard that then rejects the result fires AFTER the job was submitted,
# leaving an untracked job on the cluster.
# ---------------------------------------------------------------------------
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

RUN_ID="${T13_RUN_ID:-t13_$(date -u +%Y%m%dT%H%M%SZ)_${SOURCE}}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_ID}"

# --export splits on EVERY comma, so a comma inside a VALUE is truncated and its
# tail is parsed as the next variable name -- silently. Ship the arm list
# colon-separated and let the worker translate it back.
ARMS_COLON="${ARMS//,/:}"

echo "=============================================="
echo "T-13 complexity campaign"
echo "=============================================="
echo "Run ID:          ${RUN_ID}"
echo "Source:          ${SOURCE}"
echo "Arms:            ${ARMS}"
echo "Shards:          ${N_SHARDS} (${SHARDS_PER_TASK} per task, one per core)"
echo "Array tasks:     ${N_TASKS} (indices 0-${ARRAY_MAX})"
echo "Node family:     ${NODE_FAMILY} (pinned: wall clock is the reported quantity)"
echo "Budget:          ${BUDGET_S} s per (graph, representation, arm)"
echo "Wallclock:       ${TIME_LIMIT}"
echo "Repo:            ${REPO_DIR}"
echo "Env:             ${CONDA_ENV_PATH}"
echo "Results:         ${RESULTS_DIR}"
echo "Logs:            ${LOGS_DIR}"
echo ""

SBATCH_ARGS=(
    --parsable
    --job-name="t13_${SOURCE}"
    --array="0-${ARRAY_MAX}"
    --time="${TIME_LIMIT}"
    --nodes=1
    --ntasks=1
    --exclusive
    --constraint="${NODE_FAMILY}"
    --account="${ACCOUNT}"
    --chdir="${REPO_DIR}"
    --output="${LOGS_DIR}/t13_${SOURCE}_%A_%a.out"
    --error="${LOGS_DIR}/t13_${SOURCE}_%A_%a.err"
    --export="ALL,ISALGRAPH_REPO_DIR=${REPO_DIR},ISALGRAPH_CONDA_ENV=${CONDA_ENV_PATH},ISALGRAPH_COHORT_ROOT=${COHORT_ROOT},T13_RESULTS_DIR=${RESULTS_DIR},T13_RUN_ID=${RUN_ID},T13_SOURCE=${SOURCE},T13_N_SHARDS=${N_SHARDS},T13_SHARDS_PER_TASK=${SHARDS_PER_TASK},T13_ARMS_COLON=${ARMS_COLON},T13_FAMILIES_COLON=${FAMILIES//,/:},T13_REPLICATES=${REPLICATES},T13_BUDGET_S=${BUDGET_S},T13_SEED=${SEED}"
    "${SCRIPT_DIR}/worker.sh"
)

if ${DRY_RUN}; then
    echo "[DRY-RUN] mkdir -p ${LOGS_DIR} ${RESULTS_DIR}"
    printf '[DRY-RUN] sbatch'
    printf ' %q' "${SBATCH_ARGS[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"

if ${TEST_ONLY}; then
    # Catches an unsatisfiable request in a second. A live submission just sits
    # PENDING forever and is indistinguishable from queue pressure.
    sbatch --test-only "${SBATCH_ARGS[@]}"
    exit $?
fi

RAW_OUTPUT="$(sbatch "${SBATCH_ARGS[@]}" 2>&1)" || {
    echo "ERROR: sbatch failed:" >&2
    echo "${RAW_OUTPUT}" >&2
    exit 1
}
JOB_ID="$(_clean_job_id "${RAW_OUTPUT}")"
if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: unparsable job id from sbatch: ${RAW_OUTPUT@Q}" >&2
    echo "       Run 'squeue' NOW -- assume the job exists until proven otherwise." >&2
    exit 1
fi

echo "Submitted array ${JOB_ID} (${N_TASKS} task(s))"
echo ""
echo "Monitor:  squeue"
echo "States:   sacct -j ${JOB_ID} -X -n -P -o JobID,State,Elapsed,NodeList"
echo "Memory:   sacct -j ${JOB_ID} -n -P -o JobID,MaxRSS | awk -F'|' '\$1 ~ /\\.batch\$/'"
echo "Logs:     ${LOGS_DIR}/t13_${SOURCE}_${JOB_ID}_*.out"
echo "Results:  ${RESULTS_DIR}/records_${SOURCE}_*of${N_SHARDS}.jsonl"
