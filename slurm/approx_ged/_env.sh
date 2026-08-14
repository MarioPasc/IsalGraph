#!/usr/bin/env bash
# Sourced by every T-05 approximate-GED worker. Carries no #SBATCH header -- the launcher
# supplies every resource flag on the sbatch command line. That is the convention
# slurm/exact_ged/ established and it is what lets one launcher dispatch four
# heterogeneously-sized jobs without four headers drifting apart.
#
# Sets up: the job header, the interpreter, PYTHONPATH, $LOCALSCRATCH staging, the traps.
#
# Environment contract -- CONTRACTS.md §8, verbatim. Every variable below is exported by
# the launcher through --export; none of them is discovered here.

set -euo pipefail
START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Job name:     ${SLURM_JOB_NAME:-n/a}"
echo "Node:         $(hostname)"
echo "CPU model:    $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
echo "Cores:        ${SLURM_CPUS_PER_TASK:-?}"
echo "Start:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git commit:   $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Roles:        ${ROLES:-n/a}"
echo "=========================================="

# The env is a bare prefix, not a named env, and conda is NOT on PATH on compute nodes.
# Invoking the interpreter by absolute path avoids the whole activation dance.
PY="${CONDA_ENV_PREFIX}/bin/python"
[[ -x "${PY}" ]] || { echo "FATAL: no interpreter at ${PY}" >&2; exit 1; }

# 🔴 Repo ROOT, never ${REPO_DIR}/src. isalgraph is NOT imported by this ticket
# (CONTRACTS.md §9) -- T-05 does not touch the encoder and must not acquire a dependency
# on the C++ engine. A src-first path is the documented way this project silently loads
# the wrong thing.
export PYTHONPATH="${REPO_DIR}:${GEDLIB_DIR}"
export PYTHONUNBUFFERED=1
# 🔴 The cluster checkout is populated by rsync of the source trees, NOT by git, so its
# .git stays pinned at whatever was last pulled there and `git rev-parse` names code that
# is not running. Measured 2026-08-13: the banner announced d6a9f4b while executing code
# eleven commits ahead. The submitting side passes the true sha in ISALGRAPH_CODE_COMMIT
# and the merge prefers it over rev-parse. Provenance naming the wrong commit is worse
# than none, because it looks checkable.
export ISALGRAPH_CODE_COMMIT="${ISALGRAPH_CODE_COMMIT:-unknown-rsync-checkout}"
# One GEDLIB environment per dataset rather than a rebuild per pair. Proven output-
# preserving: cohort mode reproduces T-27's LINUX census byte-for-byte on all three
# roles (sha256 identical, max abs diff 0.0), verified on two machines. The runner
# still DEFAULTS to per-pair so T-03's behaviour is untouched; this campaign opts in.
# Worth 1.8-5.8x on `lb` and ~1.1x on `ubs` -- the saving is a flat per-pair constant
# equal to the bare rebuild (276 us on protein), so it matters most where the solve is
# cheapest. It is NOT the 33x an earlier diagnosis claimed; measured on Picasso the
# rebuild is 22% of per-pair cost and the solve is 78%.
export ENV_MODE="${ENV_MODE:-cohort}"
# Every solver here runs single-threaded per pool worker; the parallelism is the process
# pool and `--threads 1` inside every GEDLIB options string (CONTRACTS.md §3). Letting
# BLAS spawn threads inside N pool workers oversubscribes badly.
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

# ---------------------------------------------------------------- role dispatch table
# CONTRACTS.md §3, frozen. THE OPTIONS STRING IS PART OF THE METHOD NAME: GEDLIB's upper
# bounds change on 91.5-93.6 % of pairs between runs at library defaults (T-27 §4.2), so
# a run whose metadata does not record these strings verbatim is invalid at the gate.
#
# Bash 4 associative arrays. Keyed by role id.
declare -A ROLE_METHOD=(
    [lb]=BRANCH_FAST
    [ub]=BIPARTITE
    [ubs]=BP_BEAM
    [ubt]=IPFP
)
declare -A ROLE_OPTIONS=(
    [lb]="--threads 1"
    [ub]="--threads 1"
    [ubs]="--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1"
    [ubt]="--threads 1 --randomness PSEUDO --initial-solutions 10"
)
# Which end of the bracket the role computes -> the runner's --compute value, and which
# array the merge promotes into ged_matrix (--ged-from).
declare -A ROLE_COMPUTE=( [lb]=lb [ub]=ub [ubs]=ub [ubt]=ub )
# Output directory per role -- CONTRACTS.md §4.
declare -A ROLE_OUTDIR=( [lb]=LB [ub]=UB [ubs]=UB_SENSITIVITY [ubt]=UB_TIGHT )

# The ten Suite-2 datasets, CONTRACTS.md §1. Cheapest first, so a wallclock overrun costs
# the largest dataset rather than all ten.
DATASET_ORDER=(linux protein grec aids_graphedx iam_letter_low iam_letter_med aids_iam
               iam_letter_high coil_del mutagenicity)
declare -A NGRAPHS=(
    [iam_letter_low]=1180  [iam_letter_med]=1253 [iam_letter_high]=2059
    [linux]=89             [aids_graphedx]=819   [grec]=650
    [aids_iam]=1811        [coil_del]=3900       [mutagenicity]=4040
    [protein]=569
)

# Fail loudly on an unknown role rather than running BRANCH_FAST under an empty options
# string, which GEDLIB accepts and which silently produces a different number.
assert_known_role() {
    local r="$1"
    [[ -n "${ROLE_METHOD[$r]:-}" ]] || { echo "FATAL: unknown role '${r}'" >&2; exit 2; }
}
