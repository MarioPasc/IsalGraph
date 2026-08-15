#!/usr/bin/env bash
# Resume the array chunks that hit their wallclock, then merge the dataset.
#
#   bash slurm/approx_ged/resume_stragglers.sh --dry-run
#   bash slurm/approx_ged/resume_stragglers.sh
#
# Defaults to ubs/mutagenicity chunks 5,9,10 -- the three tasks of array 2005391 that
# TIMEOUT at 12:00:06, leaving 68,230 of 8,158,780 pairs (0.84 %) uncomputed and merge
# 2005392 permanently unsatisfiable. Override ROLE/KEY/N_CHUNKS/CHUNKS to reuse it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACCOUNT="tic_163_uma"
CONSTRAINT="sr"
FSCRATCH="${FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
# 🔴 _env.sh consumes these WITHOUT a default (`PY="${CONDA_ENV_PREFIX}/bin/python"`), so
# omitting either kills the job in 3 s with `unbound variable`. Job 2010228 died exactly
# that way: a launcher written beside finish_remaining.sh rather than derived from its
# export list. CONTRACTS §8 is the authority for both paths.
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${FSCRATCH}/conda_envs/isalgraph}"
GEDLIB_DIR="${GEDLIB_DIR:-${FSCRATCH}/build_gedlib/graphkit-learn}"
REPO_DIR="${REPO_DIR:-${FSCRATCH}/repos/IsalGraph}"
OUT_DIR="${OUT_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/approx_ged}"
DATA_DIR="${DATA_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph/suite2}"
LOGS_DIR="${LOGS_DIR:-${OUT_DIR}/logs}"

ROLE="${ROLE:-ubs}"
KEY="${KEY:-mutagenicity}"
N_CHUNKS="${N_CHUNKS:-19}"
CHUNKS="${CHUNKS:-5:9:10}"
N_GRAPHS="${N_GRAPHS:-4040}"

# ~2.05 h of real work at the rate these chunks actually achieved; 6 h is ~3x headroom,
# and the runner checkpoints every 20,000 pairs so an overrun costs wall time, not work.
WALLCLOCK="${WALLCLOCK:-0-06:00:00}"
START_CUTOFF_S="${START_CUTOFF_S:-16200}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "${LOGS_DIR}" 2>/dev/null || true

# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning banner to
# --parsable output. Taking the LAST line first is load-bearing: a line-wise
# `sed 's/[^0-9]//g'` leaves the banner's newlines intact, so the guard fires only after
# the job is already queued -- leaving an untracked job running.
_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }
submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}

# 🔴 The commit recorded is the ORIGINAL run's, not this checkout's HEAD, and that is
# deliberate. 93 % of these chunks' pairs were computed by the original submission at
# 10752df3; the resume only finishes the tail. `git diff 10752df3..HEAD` touches
# ged_exact_runner, ged_backends, ged_merge_shards, ged_pair_index, worker_range.sh and
# _env.sh in ZERO lines -- the compute path is byte-identical, and the only commits since
# add analysis modules the runner never imports. So this sha identifies the code that
# produced every value in the file, which is what the field is for. Passing HEAD would
# name a run that never happened. The resume is recorded in T-05-design.md, not here.
CODE_COMMIT="${ISALGRAPH_CODE_COMMIT:-10752df35d4c318a36cf5c932654f3724f6e72e9}"

EXPORTS="ALL,REPO_DIR=${REPO_DIR},OUT_DIR=${OUT_DIR},DATA_DIR=${DATA_DIR}"
EXPORTS+=",CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX},GEDLIB_DIR=${GEDLIB_DIR}"
EXPORTS+=",ISALGRAPH_CODE_COMMIT=${CODE_COMMIT}"
EXPORTS+=",ROLE=${ROLE},KEY=${KEY},N_CHUNKS=${N_CHUNKS},CHUNKS=${CHUNKS}"
EXPORTS+=",N_GRAPHS=${N_GRAPHS},START_CUTOFF_S=${START_CUTOFF_S}"

echo "role=${ROLE} key=${KEY} chunks=${CHUNKS} of ${N_CHUNKS}  wall=${WALLCLOCK}"

if (( DRY_RUN )); then
    echo "[DRY-RUN] resume: sbatch --time=${WALLCLOCK} --cpus-per-task=1 --mem=16G"
    echo "[DRY-RUN]         --constraint=${CONSTRAINT} --account=${ACCOUNT}"
    echo "[DRY-RUN]         --export=${EXPORTS}"
    echo "[DRY-RUN]         ${SCRIPT_DIR}/worker_resume_chunks.sh"
    echo "[DRY-RUN] merge:  afterok, worker_merge_range.sh, n_graphs=${N_GRAPHS}"
    exit 0
fi

RID=$(submit --job-name="ag-resume-${ROLE}-${KEY}" --account="${ACCOUNT}" \
    --time="${WALLCLOCK}" --ntasks=1 --cpus-per-task=1 --mem=16G \
    --constraint="${CONSTRAINT}" \
    --output="${LOGS_DIR}/ag-resume-${ROLE}-${KEY}_%j.out" \
    --error="${LOGS_DIR}/ag-resume-${ROLE}-${KEY}_%j.err" \
    --export="${EXPORTS}" "${SCRIPT_DIR}/worker_resume_chunks.sh") || exit 1

MID=$(submit --job-name="agm-${ROLE}-${KEY}" --account="${ACCOUNT}" \
    --time="0-04:00:00" --ntasks=1 --cpus-per-task=2 --mem=64G \
    --constraint="${CONSTRAINT}" --dependency="afterok:${RID}" \
    --output="${LOGS_DIR}/agm-${ROLE}-${KEY}_%j.out" \
    --error="${LOGS_DIR}/agm-${ROLE}-${KEY}_%j.err" \
    --export="${EXPORTS}" "${SCRIPT_DIR}/worker_merge_range.sh") || exit 1

# sbatch ACCEPTS a malformed --dependency and records Dependency=(null), which would let
# the merge start immediately against an incomplete shard set. Verify it took.
if scontrol show job "${MID}" | grep -q 'Dependency=(null)'; then
    echo "FATAL: dependency dropped on merge ${MID}; cancelling it" >&2
    scancel "${MID}"
    exit 1
fi

echo "resume=${RID}  merge=${MID} (afterok)"
echo "watch: sacct -j ${RID},${MID} -X -n -P -o JobID,State,Elapsed"
