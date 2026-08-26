#!/usr/bin/env bash
# =============================================================================
# T-13 complexity campaign -- Picasso worker
# =============================================================================
#
# One array task owns one node exclusively and runs T13_SHARDS_PER_TASK
# single-threaded shards concurrently, each pinned to its own core.
#
# NO #SBATCH HEADER, DELIBERATELY. Every resource flag is supplied by
# slurm/launch.sh on the sbatch command line -- the IsalGraph convention, see
# experiments/paper_pipeline/launch.sh. A stray #SBATCH line here would be a
# second source of truth for the resources, and the manual's rule that any
# #SBATCH after the first uncommented line is silently ignored makes that a
# failure nobody sees.
#
# Reads from the environment: ISALGRAPH_REPO_DIR, ISALGRAPH_CONDA_ENV,
# ISALGRAPH_COHORT_ROOT, T13_RESULTS_DIR, T13_RUN_ID, T13_SOURCE, T13_N_SHARDS,
# T13_SHARDS_PER_TASK, T13_ARMS_COLON, T13_BUDGET_S, T13_SEED,
# T13_FAMILIES_COLON, T13_REPLICATES.
#
set -euo pipefail

START_TIME=$(date +%s)

REPO_DIR="${ISALGRAPH_REPO_DIR:?ISALGRAPH_REPO_DIR is not set}"
CONDA_ENV_PATH="${ISALGRAPH_CONDA_ENV:?ISALGRAPH_CONDA_ENV is not set}"
RESULTS_DIR="${T13_RESULTS_DIR:?T13_RESULTS_DIR is not set}"
SOURCE="${T13_SOURCE:?T13_SOURCE is not set}"
N_SHARDS="${T13_N_SHARDS:?T13_N_SHARDS is not set}"
SHARDS_PER_TASK="${T13_SHARDS_PER_TASK:?T13_SHARDS_PER_TASK is not set}"
BUDGET_S="${T13_BUDGET_S:-300}"
SEED="${T13_SEED:-13}"
RUN_ID="${T13_RUN_ID:-t13_unknown}"

# The launcher ships the arm list colon-separated because --export splits on
# every comma and would truncate "default,no_bnb" to "default" in silence.
ARMS="${T13_ARMS_COLON//:/,}"

# Same reason, same encoding: the campaign runs the three ladder families and
# "spider_ladder,symmetry_ladder" inside --export would arrive as
# "spider_ladder" with "symmetry_ladder" parsed as a junk variable name, in
# silence. Empty means "every family", which is the full 664-spec grid.
FAMILIES="${T13_FAMILIES_COLON//:/,}"
REPLICATES="${T13_REPLICATES:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_LO=$(( TASK_ID * SHARDS_PER_TASK ))
SHARD_HI=$(( SHARD_LO + SHARDS_PER_TASK - 1 ))

echo "=========================================="
echo "T-13 complexity campaign -- worker"
echo "=========================================="
echo "Job:           ${SLURM_JOB_ID:-local}"
echo "Array task:    ${TASK_ID}"
echo "Node:          $(hostname)"
echo "CPU model:     $(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
echo "Start:         $(date)"
echo "Run ID:        ${RUN_ID}"
echo "Source:        ${SOURCE}"
echo "Arms:          ${ARMS}"
echo "Shards:        ${SHARD_LO}..${SHARD_HI} of ${N_SHARDS}"
echo "Budget:        ${BUDGET_S} s"
echo "Git commit:    $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------
# Conda is not on PATH on every Picasso node, and activation is not needed: the
# interpreter is addressed by absolute path. Only the env's lib directory has to
# be reachable, for the extension's libstdc++.
PY="${CONDA_ENV_PATH}/bin/python"
[[ -x "${PY}" ]] || { echo "[FATAL] no interpreter at ${PY}" >&2; exit 1; }
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

# NEVER set PYTHONPATH=<repo>/src. A src-first path shadows the editable install
# and the engine falls back to pure Python SILENTLY -- 23x-1025x slower, no
# error, and every timing in the campaign becomes fiction. sbatch --chdir puts
# the repo on sys.path for `-m benchmarks...`, which is all that is needed.
unset PYTHONPATH

# time.process_time sums CPU across ALL threads of the process, so an unpinned
# BLAS would inflate every reading by its thread count. One thread everywhere.
export ISALGRAPH_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# ENGINE GATE -- fail on the node, before 128 shards start
# ---------------------------------------------------------------------------
# measure.py asserts this per shard too, but doing it once here turns a
# 128-way identical failure into one legible line.
"${PY}" - <<'ENGINE_CHECK'
import sys
import isalgraph
from benchmarks.real_data.eval_t13_complexity import measure

info = measure.assert_engine()
sys.stderr.write(
    f"[engine] {isalgraph.engine()} build_hash={info['build_hash']} "
    f"compiler={info.get('compiler')} isa={info.get('isa')}\n"
)
ENGINE_CHECK

# ---------------------------------------------------------------------------
# LOCALSCRATCH staging
# ---------------------------------------------------------------------------
# 128 shards appending JSONL with a flush per record is a heavy small-write load
# on shared GPFS, which the manual (S4.9) asks us to keep off it. Results are
# written on the node and mirrored back at the end.
MYLOCALSCRATCH="${LOCALSCRATCH:-/tmp}/${USER}/${SLURM_JOB_ID:-local}_${TASK_ID}"
mkdir -p "${MYLOCALSCRATCH}/out"

# Picasso exports a SHARED PYTHONPYCACHEPREFIX, so every task of one user on a
# node writes the same .pyc paths -- 128 of them here. The symptom is an
# intermittent ModuleNotFoundError on a module that is plainly present, hitting
# a small fraction of shards. An import that is wrong is wrong every time; one
# that fails occasionally is a race.
export PYTHONPYCACHEPREFIX="${MYLOCALSCRATCH}/pycache"

FINALIZED=0
finalize() {
    [[ "${FINALIZED}" == 1 ]] && return 0
    FINALIZED=1
    mkdir -p "${RESULTS_DIR}"
    # Mirror the WHOLE tree. Never enumerate the files expected: a copy-back
    # that names paths loses anything written beside them, and every file it
    # did name comes back, so nothing looks wrong until the analysis needs the
    # one that is missing.
    local ok=1
    shopt -s nullglob dotglob
    for src in "${MYLOCALSCRATCH}/out"/*; do
        local base tmp
        base="$(basename "${src}")"
        tmp="${RESULTS_DIR}/.${base}.${SLURM_JOB_ID:-local}_${TASK_ID}.tmp"
        # Copy to a temp name in the DESTINATION directory, then rename. A
        # rename within one directory is atomic, so a reader sees the old file
        # or the new one, never half of one.
        if cp -a "${src}" "${tmp}" && mv -f "${tmp}" "${RESULTS_DIR}/${base}"; then
            :
        else
            ok=0
            rm -f "${tmp}" 2>/dev/null || true
        fi
    done
    shopt -u nullglob dotglob
    if [[ "${ok}" == 1 ]]; then
        echo "[finalize] mirrored $(ls -1 "${MYLOCALSCRATCH}/out" | wc -l) file(s) -> ${RESULTS_DIR}"
        [[ -n "${MYLOCALSCRATCH}" ]] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
    else
        echo "[FATAL] copy-back failed; results remain at ${MYLOCALSCRATCH} on $(hostname)" >&2
    fi
}
trap finalize EXIT
# SLURM sends SIGTERM and waits KillWait before SIGKILL. Without these the shell
# dies WITHOUT running the EXIT trap, and the whole task's output dies with it.
trap 'exit 143' TERM
trap 'exit 130' INT

# Stage existing shard files in, so measure.py's resume sees them. Without this
# a requeued task starts from an empty tree and redoes work it already paid for.
for shard in $(seq "${SHARD_LO}" "${SHARD_HI}"); do
    existing="${RESULTS_DIR}/records_${SOURCE}_${shard}of${N_SHARDS}.jsonl"
    [[ -f "${existing}" ]] && cp -a "${existing}" "${MYLOCALSCRATCH}/out/"
done
echo "[stage-in] $(ls -1 "${MYLOCALSCRATCH}/out" | wc -l) existing shard file(s) restored"

# ---------------------------------------------------------------------------
# RUN: one shard per dedicated core
# ---------------------------------------------------------------------------
# --exclusive gives us the whole node, so cores 0..SHARDS_PER_TASK-1 are ours.
# taskset pins each shard to one, which is what keeps a timing a property of the
# work rather than of the scheduler's migration decisions.
declare -a PIDS=()
declare -a SHARDS=()

for offset in $(seq 0 $(( SHARDS_PER_TASK - 1 ))); do
    shard=$(( SHARD_LO + offset ))
    (( shard >= N_SHARDS )) && break
    log="${MYLOCALSCRATCH}/shard_${shard}.log"
    declare -a SUBSET_ARGS=()
    [[ -n "${FAMILIES}" ]]   && SUBSET_ARGS+=(--families "${FAMILIES}")
    [[ -n "${REPLICATES}" ]] && SUBSET_ARGS+=(--replicates "${REPLICATES}")
    taskset -c "${offset}" "${PY}" -m benchmarks.eval_t13_complexity.measure \
        --source "${SOURCE}" \
        --shard "${shard}" \
        --n-shards "${N_SHARDS}" \
        "${SUBSET_ARGS[@]}" \
        --arms "${ARMS}" \
        --budget-s "${BUDGET_S}" \
        --seed "${SEED}" \
        --run-id "${RUN_ID}" \
        --out "${MYLOCALSCRATCH}/out" \
        > "${log}" 2>&1 &
    PIDS+=("$!")
    SHARDS+=("${shard}")
done

echo "[run] launched ${#PIDS[@]} shard(s), one per core"

FAILURES=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        :
    else
        code=$?
        FAILURES=$(( FAILURES + 1 ))
        echo "[FAIL] shard ${SHARDS[$i]} exited ${code}" >&2
        tail -n 20 "${MYLOCALSCRATCH}/shard_${SHARDS[$i]}.log" >&2 || true
    fi
done

# Keep the per-shard logs beside the records: a censored cell is a result, and
# the reason it censored is in the log.
cp -a "${MYLOCALSCRATCH}"/shard_*.log "${MYLOCALSCRATCH}/out/" 2>/dev/null || true

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $(( ELAPSED / 3600 ))h $(( (ELAPSED / 60) % 60 ))m $(( ELAPSED % 60 ))s"
echo "Failures:  ${FAILURES} of ${#PIDS[@]} shard(s)"

if (( ELAPSED < 7200 )); then
    # Not fatal -- a resume pass legitimately finishes in seconds -- but a FRESH
    # campaign finishing under two hours means the array was sized too wide and
    # the next submission must use fewer shards. SCBI asked for the floor in
    # writing after a 12,600-task campaign of minute-long jobs.
    echo "[NOTE] task ran under the 2 h floor. If this was a fresh run and not a" >&2
    echo "       resume, lower --n-shards before the next submission." >&2
fi

exit $(( FAILURES > 0 ? 1 : 0 ))
