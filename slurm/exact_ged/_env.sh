#!/usr/bin/env bash
# Sourced by every T-03 worker. Carries no #SBATCH header -- the launcher supplies every
# resource flag on the sbatch command line, which is the existing IsalGraph convention
# and keeps four workers from drifting apart.
#
# Sets up: the job header, conda, PYTHONPATH, $LOCALSCRATCH staging, and the traps.

set -euo pipefail
START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node:         $(hostname)"
echo "CPU model:    $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
echo "Cores:        ${SLURM_CPUS_PER_TASK:-?}"
echo "Start:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git commit:   $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Timeout/pair: ${TIMEOUT_PER_PAIR}s"
echo "=========================================="

# The env is a bare prefix, not a named env, and conda is not on PATH on compute nodes.
# Invoking the interpreter by absolute path avoids the whole activation dance.
PY="${CONDA_ENV_PREFIX}/bin/python"
[[ -x "${PY}" ]] || { echo "FATAL: no interpreter at ${PY}" >&2; exit 1; }

# 🔴 Repo ROOT, never ${REPO_DIR}/src. isalgraph is not installed on this cluster and
# T-03 must not import it -- exact GED does not touch the encoder, so there is no C++
# engine in this ticket. A src-first path is the documented way this project silently
# loads the wrong thing.
export PYTHONPATH="${REPO_DIR}:${GEDLIB_DIR}"
export PYTHONUNBUFFERED=1
# Every solver here is single-threaded per worker process; the parallelism is the
# process pool. Letting BLAS spawn threads inside 64 pool workers oversubscribes badly.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Picasso exports a SHARED PYTHONPYCACHEPREFIX, so every task of one user on a node
# writes the same .pyc paths. Symptom is an intermittent ModuleNotFoundError on a module
# that is plainly present. Give each task its own.
MYLOCAL="${LOCALSCRATCH:-/tmp}/${USER}/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${MYLOCAL}/out"
export PYTHONPYCACHEPREFIX="${MYLOCAL}/pycache"

RESULTS_DIR="${OUT_DIR}"
mkdir -p "${RESULTS_DIR}"

FINALIZED=0
finalize() {
    [[ "${FINALIZED}" == 1 ]] && return 0
    FINALIZED=1
    # Mirror the WHOLE tree. Never enumerate expected files: everything you listed comes
    # back, so nothing looks wrong until the analysis needs the one you forgot.
    if [[ -d "${MYLOCAL}/out" ]] && [[ -n "$(ls -A "${MYLOCAL}/out" 2>/dev/null)" ]]; then
        if cp -a "${MYLOCAL}/out/." "${RESULTS_DIR}/"; then
            echo "[finalize] mirrored $(find "${MYLOCAL}/out" -type f | wc -l) files -> ${RESULTS_DIR}"
            rm -rf --one-file-system "${MYLOCAL}"      # only AFTER the copy worked
        else
            echo "[FATAL] copy-back failed; results remain at ${MYLOCAL} on $(hostname)" >&2
        fi
    fi
    local elapsed=$(( $(date +%s) - START_TIME ))
    echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)  Duration: $((elapsed/3600))h $(((elapsed/60)%60))m $((elapsed%60))s"
}
trap finalize EXIT
# 🔴 SLURM sends SIGTERM and waits 30 s before SIGKILL. Without these the shell dies
# WITHOUT running the EXIT trap and the whole task's output dies on the node.
trap 'echo "[signal] SIGTERM"; exit 143' TERM
trap 'echo "[signal] SIGINT";  exit 130' INT

run_py() { "${PY}" -m "$@"; }
