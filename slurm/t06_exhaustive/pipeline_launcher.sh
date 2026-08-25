#!/usr/bin/env bash
# T06_exhaustive stages 2 and 3 on Picasso: DISTANCES then F2, chained.
#
#   bash slurm/t06_exhaustive/pipeline_launcher.sh --dry-run
#   bash slurm/t06_exhaustive/pipeline_launcher.sh --test-only
#   bash slurm/t06_exhaustive/pipeline_launcher.sh                      # submit
#   AFTER_ENCODE=2102929 bash slurm/t06_exhaustive/pipeline_launcher.sh # chain on the encode array
#
# THREE JOBS, CHAINED BY DEPENDENCY so nothing is submitted by hand in sequence:
#
#   distances  array, N_DIST_TASKS tasks over 30 units (15 cells x 2 arms)
#              --dependency=afterok:${AFTER_ENCODE}   when AFTER_ENCODE is set
#   f2-shards  array, N_F2_TASKS tasks over 15 (suite, dataset) shards
#              --dependency=afterok:<distances>
#   f2-merge   single job, merges every partial and emits family_F2.json
#              --dependency=afterok:<f2-shards>
#
# `afterok`, not `afterany`, at every link: a distance stage that failed leaves a
# tree F2 would happily run on and report a family computed from whatever
# matrices happen to exist. Wrong numbers with a zero exit status is the failure
# mode this whole campaign is guarding against.
#
# 🔴 A DEPENDENCY THAT sbatch ACCEPTS IS NOT A DEPENDENCY THAT TOOK. A malformed
# upstream id is accepted and recorded as Dependency=(null), and the downstream
# job then starts IMMEDIATELY against partial input. Every link is verified with
# scontrol after submission and the chain is cancelled if one did not take.
#
# WHAT IS NOT RECOMPUTED: every GED matrix and every competitor distance. They
# are unchanged by this work and are reused verbatim. The distance worker
# asserts the competitor matrices are reachable rather than producing a tree F2
# cannot use.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
FSCRATCH="/mnt/home/users/tic_163_uma/mpascual/fscratch"
export CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-${FSCRATCH}/conda_envs/isalgraph}"
export REPO_DIR="${REPO_DIR:-${FSCRATCH}/repos/IsalGraph}"
export OUT_ROOT="${OUT_ROOT:-${FSCRATCH}/datasets/isalgraph/T06_exhaustive}"

# Roots with a /media/... workstation default. Both are VALIDATED in the worker;
# an unset one fails at the first shard with a message naming a path that has
# never existed on this cluster. This is the defect that killed array 2102923.
export GED_ROOT="${GED_ROOT:-${FSCRATCH}/datasets/isalgraph/eval/ged_matrices}"
export APPROX_ROOT="${APPROX_ROOT:-${FSCRATCH}/datasets/isalgraph/APPROX_GED}"

LOGS_DIR="${LOGS_DIR:-${HOME}/execs/isalgraph/logs}"
ACCOUNT="${ACCOUNT:-tic_163_uma}"

export T06_REFERENCE_ARM="${T06_REFERENCE_ARM:-isalgraph_exhaustive}"
export COMPARATOR_SET="${COMPARATOR_SET:-full}"

# distances: rapidfuzz over ~4000-graph cohorts; the tier-3 matrices are
# 4040^2 float64 plus condensed vectors.
DIST_WALL="${DIST_WALL:-0-12:00:00}"
DIST_CPUS="${DIST_CPUS:-16}"
DIST_MEM="${DIST_MEM:-48G}"
export DIST_JOBS="${DIST_JOBS:-12}"        # below DIST_CPUS, deliberately
N_DIST_TASKS="${N_DIST_TASKS:-5}"
DIST_CONCURRENT="${DIST_CONCURRENT:-5}"

# F2: single-process numpy per shard; run_f2.sh caps concurrency at 5 because a
# tier-3 shard peaks around 3 GB.
F2_WALL="${F2_WALL:-0-12:00:00}"
F2_CPUS="${F2_CPUS:-4}"
F2_MEM="${F2_MEM:-24G}"
N_F2_TASKS="${N_F2_TASKS:-5}"
F2_CONCURRENT="${F2_CONCURRENT:-5}"

export START_CUTOFF_S="${START_CUTOFF_S:-30600}"   # 8.5 h of a 12 h wall

SUITE1=(linux aids iam_letter_low iam_letter_med iam_letter_high)
SUITE2=(linux grec protein aids_graphedx iam_letter_low iam_letter_med
        aids_iam iam_letter_high coil_del mutagenicity)
ARMS=(isalgraph_exhaustive isalgraph_greedy)

# Expensive first. The workers STRIDE over this list, so ordering it by cost is
# what makes the round-robin deal balanced.
COST_ORDER=(mutagenicity coil_del iam_letter_high aids_iam aids_graphedx
            iam_letter_med iam_letter_low protein grec linux)

DIST_UNITS=()
for arm in "${ARMS[@]}"; do
  for d in "${COST_ORDER[@]}"; do DIST_UNITS+=("suite2/${d}/${arm}"); done
  for d in iam_letter_high iam_letter_med iam_letter_low aids linux; do
    DIST_UNITS+=("suite1/${d}/${arm}")
  done
done
N_DIST_UNITS=${#DIST_UNITS[@]}
EXPECTED_DIST=$(( (${#SUITE1[@]} + ${#SUITE2[@]}) * ${#ARMS[@]} ))
[ "${N_DIST_UNITS}" -eq "${EXPECTED_DIST}" ] || {
  echo "FATAL: built ${N_DIST_UNITS} distance units, expected ${EXPECTED_DIST}" >&2; exit 1; }

F2_UNITS=()
for d in "${COST_ORDER[@]}"; do F2_UNITS+=("suite2/${d}"); done
for d in iam_letter_high iam_letter_med iam_letter_low aids linux; do F2_UNITS+=("suite1/${d}"); done
EXPECTED_SHARDS=${#F2_UNITS[@]}
export EXPECTED_SHARDS
[ "${EXPECTED_SHARDS}" -eq $(( ${#SUITE1[@]} + ${#SUITE2[@]} )) ] || {
  echo "FATAL: built ${EXPECTED_SHARDS} shards, expected 15" >&2; exit 1; }

# 🔴 COLON-separated. --export splits on COMMAS, truncating a value silently:
# low task indices then return plausible results and high ones die out of range.
DIST_UNIT_LIST=$(IFS=':'; echo "${DIST_UNITS[*]}")
F2_UNIT_LIST=$(IFS=':'; echo "${F2_UNITS[*]}")

DRY_RUN=false; TEST_ONLY=false
case "${1:-}" in
  --dry-run) DRY_RUN=true ;;
  --test-only) TEST_ONLY=true ;;
  "") ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

mkdir -p "${LOGS_DIR}"

COMMON_EXPORT="CONDA_PREFIX_DIR=${CONDA_PREFIX_DIR},REPO_DIR=${REPO_DIR},OUT_ROOT=${OUT_ROOT},START_CUTOFF_S=${START_CUTOFF_S}"

DIST_ARGS=(
  --job-name=t06dist
  --array="0-$(( N_DIST_TASKS - 1 ))%${DIST_CONCURRENT}"
  --time="${DIST_WALL}" --ntasks=1 --cpus-per-task="${DIST_CPUS}" --mem="${DIST_MEM}"
  --constraint=cpu --account="${ACCOUNT}" --chdir="${REPO_DIR}"
  --output="${LOGS_DIR}/t06dist_%A_%a.out" --error="${LOGS_DIR}/t06dist_%A_%a.err"
  --export="ALL,${COMMON_EXPORT},DIST_JOBS=${DIST_JOBS},N_TASKS=${N_DIST_TASKS},UNIT_LIST=${DIST_UNIT_LIST}"
  "${SCRIPT_DIR}/distances_worker.sh"
)

F2_EXPORT="${COMMON_EXPORT},GED_ROOT=${GED_ROOT},APPROX_ROOT=${APPROX_ROOT},T06_REFERENCE_ARM=${T06_REFERENCE_ARM},COMPARATOR_SET=${COMPARATOR_SET},EXPECTED_SHARDS=${EXPECTED_SHARDS}"

F2_SHARD_ARGS=(
  --job-name=t06f2s
  --array="0-$(( N_F2_TASKS - 1 ))%${F2_CONCURRENT}"
  --time="${F2_WALL}" --ntasks=1 --cpus-per-task="${F2_CPUS}" --mem="${F2_MEM}"
  --constraint=cpu --account="${ACCOUNT}" --chdir="${REPO_DIR}"
  --output="${LOGS_DIR}/t06f2s_%A_%a.out" --error="${LOGS_DIR}/t06f2s_%A_%a.err"
  --export="ALL,${F2_EXPORT},STAGE=shards,N_TASKS=${N_F2_TASKS},UNIT_LIST=${F2_UNIT_LIST}"
  "${SCRIPT_DIR}/f2_worker.sh"
)

F2_MERGE_ARGS=(
  --job-name=t06f2m
  --time=0-02:00:00 --ntasks=1 --cpus-per-task=4 --mem=32G
  --constraint=cpu --account="${ACCOUNT}" --chdir="${REPO_DIR}"
  --output="${LOGS_DIR}/t06f2m_%j.out" --error="${LOGS_DIR}/t06f2m_%j.err"
  --export="ALL,${F2_EXPORT},STAGE=merge,N_TASKS=1"
  "${SCRIPT_DIR}/f2_worker.sh"
)

echo "reference arm:  ${T06_REFERENCE_ARM}"
echo "comparators:    ${COMPARATOR_SET}"
echo "distance units: ${N_DIST_UNITS} over ${N_DIST_TASKS} tasks"
echo "f2 shards:      ${EXPECTED_SHARDS} over ${N_F2_TASKS} tasks, then 1 merge job"
echo "out:            ${OUT_ROOT}"
echo "ged root:       ${GED_ROOT}"
echo "approx root:    ${APPROX_ROOT}"
echo "chain:          ${AFTER_ENCODE:+encode ${AFTER_ENCODE} -> }distances -> f2-shards -> f2-merge"

if ${DRY_RUN}; then
  printf '[DRY-RUN 1/3 distances] sbatch'
  [ -n "${AFTER_ENCODE:-}" ] && printf ' --dependency=afterok:%s' "${AFTER_ENCODE}"
  printf ' %q' "${DIST_ARGS[@]}"; printf '\n\n'
  printf '[DRY-RUN 2/3 f2-shards] sbatch --dependency=afterok:<distances>'
  printf ' %q' "${F2_SHARD_ARGS[@]}"; printf '\n\n'
  printf '[DRY-RUN 3/3 f2-merge ] sbatch --dependency=afterok:<f2-shards>'
  printf ' %q' "${F2_MERGE_ARGS[@]}"; printf '\n'
  exit 0
fi

if ${TEST_ONLY}; then
  echo "--- distances ---"; sbatch --test-only "${DIST_ARGS[@]}"
  echo "--- f2 shards ---"; sbatch --test-only "${F2_SHARD_ARGS[@]}"
  echo "--- f2 merge  ---"; sbatch --test-only "${F2_MERGE_ARGS[@]}"
  exit 0
fi

# Picasso's Lua wrapper prepends ANSI and a multi-line warning to --parsable
# output. Take the LAST line FIRST: a line-by-line sed leaves the newlines in
# place, and the guard then fires AFTER submission, leaving an untracked job.
_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }
submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || {
        echo "FATAL: unparsable job id: ${raw@Q}" >&2
        echo "A submission may still have happened -- run squeue before retrying." >&2
        return 1; }
    echo "${id}"
}
verify_dep() {   # job_id label
    if scontrol show job "$1" 2>/dev/null | grep -q 'Dependency=(null)'; then
        echo "FATAL: dependency dropped on $2 (job $1); cancelling the chain" >&2
        scancel "$1" 2>/dev/null || true
        return 1
    fi
    echo "  dependency on $2 verified"
}

echo "--- live state ---"; quota 2>/dev/null || echo "(quota unavailable)"; echo "------------------"

# Explicit if-blocks, not `[ ] && ...`: an empty array expanded as
# "${ARR[@]}" is an unbound-variable error under `set -u` on bash < 4.4, and a
# trailing `&&` list that short-circuits is a control-flow foot-gun next to
# `set -e`. Neither would be visible until a live submission.
if [ -n "${AFTER_ENCODE:-}" ]; then
    DIST_ID=$(submit --dependency="afterok:${AFTER_ENCODE}" "${DIST_ARGS[@]}") || exit 1
    echo "Submitted distances  ${DIST_ID}"
    verify_dep "${DIST_ID}" "encode ${AFTER_ENCODE}" || exit 1
else
    DIST_ID=$(submit "${DIST_ARGS[@]}") || exit 1
    echo "Submitted distances  ${DIST_ID}"
fi

SHARD_ID=$(submit --dependency="afterok:${DIST_ID}" "${F2_SHARD_ARGS[@]}") || exit 1
echo "Submitted f2-shards  ${SHARD_ID}"
verify_dep "${SHARD_ID}" "distances ${DIST_ID}" || exit 1

MERGE_ID=$(submit --dependency="afterok:${SHARD_ID}" "${F2_MERGE_ARGS[@]}") || exit 1
echo "Submitted f2-merge   ${MERGE_ID}"
verify_dep "${MERGE_ID}" "f2-shards ${SHARD_ID}" || exit 1

echo ""
echo "Chain: ${AFTER_ENCODE:+${AFTER_ENCODE} -> }${DIST_ID} -> ${SHARD_ID} -> ${MERGE_ID}"
echo "Monitor: squeue"
echo "States:  sacct -j ${DIST_ID},${SHARD_ID},${MERGE_ID} -X -n -P -o JobID,State,Elapsed"
echo "Memory:  sacct -j ${DIST_ID} -n -P -o JobID,MaxRSS | grep '\.batch'"
echo "Logs:    ${LOGS_DIR}/t06{dist,f2s,f2m}_*"
echo "Resume:  rerun this launcher -- every stage skips work already on disk"
