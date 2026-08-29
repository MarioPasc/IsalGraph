#!/usr/bin/env bash
# Submit the T-28 size profile + rho-vs-size figures to Picasso.
#
#   bash slurm/t28_metrics/figure_launcher.sh --dry-run
#   bash slurm/t28_metrics/figure_launcher.sh
#   REFERENCES="wl spectral" bash slurm/t28_metrics/figure_launcher.sh
#
# ONE profile pass over every reference, then one figure per alternative
# distance -- same cohorts, same representations and same T-04a distances as the
# GED figure, with only the reference changed.
#
# A short CPU job rather than an array: the representation distances are read
# from the T-06 cache and the per-stratum bootstrap is skipped because the
# figures never read it, so the whole fifteen-cell profile is seconds of compute.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FSCRATCH="/mnt/home/users/tic_163_uma/mpascual/fscratch"
DATA="${FSCRATCH}/datasets/isalgraph"

export CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-${FSCRATCH}/conda_envs/isalgraph}"
export REPO_DIR="${REPO_DIR:-${FSCRATCH}/repos/IsalGraph-t28}"

# Every root named explicitly. The workstation archive and this staging tree do
# not share a layout, so a single prefix cannot address both.
export DISTANCES="${DISTANCES:-${DATA}/T06/distances}"
export GED_ROOT="${GED_ROOT:-${DATA}/eval/ged_matrices}"
export APPROX_ROOT="${APPROX_ROOT:-${DATA}/APPROX_GED}"
export T28_REFERENCE_ROOT="${T28_REFERENCE_ROOT:-${DATA}/T28_metrics/references}"
export FIG_OUT="${FIG_OUT:-${DATA}/T28_metrics/figures}"

# Space-separated for the human, COLON-separated on the wire.
#
# sbatch --export takes a COMMA-separated list, and a value containing spaces
# does not survive it: "...,T28_FIG_REFERENCES=wl spectral ..." puts everything
# after the first space outside the option. Caught by reading the --dry-run
# line, which is the only place the assembled argv is visible.
REFERENCES="${REFERENCES:-wl spectral spectral_comb spectral_adj spectral_esd}"
T28_FIG_REFERENCES=$(printf '%s' "${REFERENCES}" | tr ' ' ':')
export T28_FIG_REFERENCES

LOGS_DIR="${LOGS_DIR:-${HOME}/execs/isalgraph/logs}"
ACCOUNT="${ACCOUNT:-tic_163_uma}"
mkdir -p "${LOGS_DIR}" "${FIG_OUT}"

fail_pre() { echo "[PREFLIGHT FAIL] $*" >&2; exit 2; }
[ -d "${REPO_DIR}" ]           || fail_pre "no repo at ${REPO_DIR}"
[ -d "${DISTANCES}" ]          || fail_pre "no distances at ${DISTANCES}"
[ -d "${GED_ROOT}" ]           || fail_pre "no GED_ROOT at ${GED_ROOT}"
[ -d "${APPROX_ROOT}" ]        || fail_pre "no APPROX_ROOT at ${APPROX_ROOT}"
[ -d "${T28_REFERENCE_ROOT}" ] || fail_pre "no reference tree at ${T28_REFERENCE_ROOT}"
[ -x "${CONDA_PREFIX_DIR}/bin/python" ] || fail_pre "no interpreter at ${CONDA_PREFIX_DIR}/bin/python"
N_REF=$(find "${T28_REFERENCE_ROOT}" -name '*__*.npz' | wc -l)
[ "${N_REF}" -ge 1 ] || fail_pre "${T28_REFERENCE_ROOT} holds no matrices"

EXPORTS="CONDA_PREFIX_DIR=${CONDA_PREFIX_DIR},REPO_DIR=${REPO_DIR},DISTANCES=${DISTANCES}"
EXPORTS="${EXPORTS},GED_ROOT=${GED_ROOT},APPROX_ROOT=${APPROX_ROOT}"
EXPORTS="${EXPORTS},T28_REFERENCE_ROOT=${T28_REFERENCE_ROOT},FIG_OUT=${FIG_OUT}"
EXPORTS="${EXPORTS},T28_FIG_REFERENCES=${T28_FIG_REFERENCES}"

ARGS=(
  --job-name=t28fig
  --time=0-02:00:00 --ntasks=1 --cpus-per-task=4 --mem=48G
  --constraint=cpu --account="${ACCOUNT}" --chdir="${REPO_DIR}"
  --output="${LOGS_DIR}/t28fig_%j.out" --error="${LOGS_DIR}/t28fig_%j.err"
  --export="ALL,${EXPORTS}"
  "${SCRIPT_DIR}/figure_worker.sh"
)

cat <<EOF
references: ${REFERENCES}
distances:  ${DISTANCES}   [T-04a's, read from cache -- NOT recomputed]
ged root:   ${GED_ROOT}
approx:     ${APPROX_ROOT}
t28 refs:   ${T28_REFERENCE_ROOT}  (${N_REF} matrices)
out:        ${FIG_OUT}
EOF

if [[ "${1:-}" == "--dry-run" ]]; then
    echo ""
    echo "[DRY-RUN] sbatch ${ARGS[*]}"
    exit 0
fi

_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }
RAW=$(sbatch --parsable "${ARGS[@]}") || { echo "sbatch failed" >&2; exit 1; }
JOB=$(_clean_job_id "${RAW}")
[[ "${JOB}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${RAW@Q}" >&2; exit 1; }
echo "submitted t28fig: ${JOB}"
echo "Logs:    ${LOGS_DIR}/t28fig_${JOB}.out"
echo "Figures: ${FIG_OUT}/fig1_rho_vs_size_<reference>.pdf"
