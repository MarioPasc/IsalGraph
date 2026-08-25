#!/usr/bin/env bash
# Submit the T06_exhaustive encoding campaign to Picasso.
#
#   bash slurm/t06_exhaustive/launcher.sh --dry-run   # print, do not submit
#   bash slurm/t06_exhaustive/launcher.sh --test-only # sbatch --test-only probe
#   bash slurm/t06_exhaustive/launcher.sh             # submit
#
# WHAT IT RUNS. Thirty units: fifteen (suite, dataset) cells x two IsalGraph
# arms. `isalgraph_exhaustive` is the true w*_G, replacing the length-suboptimal
# pruned form; `isalgraph_greedy` is the declared canonicalisation ablation.
# NO COMPETITOR IS RE-ENCODED and NO GED MATRIX IS RECOMPUTED -- both are
# unchanged by this work and are reused verbatim from data/source/T06.
#
# WHY THE ARRAY IS SIZED IN TASKS, NOT UNITS. SCBI asked (soporte@scbi.uma.es,
# 2026-08-07) that every submitted job run for at least two hours: placing a job
# costs the scheduler about the same whether it then runs for ten seconds or ten
# hours, so short jobs are almost pure overhead. Unit cost here is wildly
# uneven -- suite1/linux is 89 graphs and finishes in seconds, suite2/protein is
# the cell where the canonical search actually bites -- so one task per unit
# would submit two dozen jobs of which most last seconds. Units are therefore
# bundled into N_TASKS tasks run sequentially.
#
# Grouping is makespan-neutral: N units of duration T under throttle K finish at
# N*T/K; grouped into N/B tasks of B*T they finish at (N/B)*(B*T)/K = N*T/K. The
# two conditions are that the array keeps at least K tasks (so no slot idles)
# and that the last round is not mostly empty -- hence the EVEN split in the
# worker rather than fixed blocks with a ragged tail.
#
# 🔴 THE EXTENSION MUST BE BUILT ON THE CLUSTER FIRST. The .so installs into
# site-packages and does not rsync. The worker asserts isalgraph.engine()=='cpp'
# and aborts otherwise, because a silent fall back to the pure-Python reference
# would not fail -- it would produce a completely different censoring rate at
# the same nominal budget, with no error. Build with -march=x86-64-v3, NEVER
# -march=native: Picasso is heterogeneous and native gives SIGILL on a fraction
# of nodes, which reads like flaky hardware rather than a build fault.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
FSCRATCH="/mnt/home/users/tic_163_uma/mpascual/fscratch"
export CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-${FSCRATCH}/conda_envs/isalgraph}"
export REPO_DIR="${REPO_DIR:-${FSCRATCH}/repos/IsalGraph}"
export OUT_ROOT="${OUT_ROOT:-${FSCRATCH}/datasets/isalgraph/T06_exhaustive}"
LOGS_DIR="${LOGS_DIR:-${HOME}/execs/isalgraph/logs}"
ACCOUNT="${ACCOUNT:-tic_163_uma}"

export BUDGET_S="${BUDGET_S:-30}"
# Deliberately below --cpus-per-task: a contended graph that would finish in 25 s
# alone can be killed at 40 s, inflating the very censoring rate the run exists
# to measure. Headroom is the point.
export ENCODE_JOBS="${ENCODE_JOBS:-6}"

WALL="${WALL:-0-08:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-16G}"
N_TASKS="${N_TASKS:-5}"
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"

# Refuse to START a unit after this many seconds, so a bundled task cannot
# overrun its wall. wall - (worst-case unit) - teardown.
export START_CUTOFF_S="${START_CUTOFF_S:-19800}"   # 5.5 h of an 8 h wall

SUITE1=(linux aids iam_letter_low iam_letter_med iam_letter_high)
SUITE2=(linux grec protein aids_graphedx iam_letter_low iam_letter_med
        aids_iam iam_letter_high coil_del mutagenicity)
ARMS=(isalgraph_exhaustive isalgraph_greedy)

# Expensive cells first: the campaign finishes when the slowest task does, and
# an even split over a cost-sorted list keeps the tail short.
UNITS=()
for arm in "${ARMS[@]}"; do
  for d in protein coil_del mutagenicity aids_iam iam_letter_high aids_graphedx \
           iam_letter_med iam_letter_low grec linux; do
    UNITS+=("suite2/${d}/${arm}")
  done
  for d in "${SUITE1[@]}"; do UNITS+=("suite1/${d}/${arm}"); done
done
N_UNITS=${#UNITS[@]}

EXPECTED_UNITS=$(( (${#SUITE1[@]} + ${#SUITE2[@]}) * ${#ARMS[@]} ))
if [ "${N_UNITS}" -ne "${EXPECTED_UNITS}" ]; then
  echo "FATAL: built ${N_UNITS} units, expected ${EXPECTED_UNITS}" >&2; exit 1
fi

# 🔴 Ship the list COLON-separated. --export splits on COMMAS, so a comma inside
# a value truncates it and its tail is parsed as the next variable name. Nothing
# errors: low task indices return plausible results and high ones die out of
# range. This has cost two separate incidents in IsalSR.
UNIT_LIST=$(IFS=':'; echo "${UNITS[*]}")

DRY_RUN=false; TEST_ONLY=false
case "${1:-}" in
  --dry-run)  DRY_RUN=true ;;
  --test-only) TEST_ONLY=true ;;
  "") ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

mkdir -p "${LOGS_DIR}"

# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning to
# --parsable output. Take the LAST line FIRST: a line-by-line sed leaves the
# newlines in place and the guard then fires AFTER the job was submitted,
# leaving an untracked job on the cluster.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

SBATCH_ARGS=(
  --job-name=t06exh
  --array="0-$(( N_TASKS - 1 ))%${MAX_CONCURRENT}"
  --time="${WALL}"
  --ntasks=1
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --constraint=cpu
  --account="${ACCOUNT}"
  --chdir="${REPO_DIR}"
  --output="${LOGS_DIR}/t06exh_%A_%a.out"
  --error="${LOGS_DIR}/t06exh_%A_%a.err"
  --export="ALL,CONDA_PREFIX_DIR=${CONDA_PREFIX_DIR},REPO_DIR=${REPO_DIR},OUT_ROOT=${OUT_ROOT},BUDGET_S=${BUDGET_S},ENCODE_JOBS=${ENCODE_JOBS},START_CUTOFF_S=${START_CUTOFF_S},N_TASKS=${N_TASKS},UNIT_LIST=${UNIT_LIST}"
  "${SCRIPT_DIR}/worker.sh"
)

echo "units:        ${N_UNITS}  (15 cells x ${#ARMS[@]} arms)"
echo "tasks:        ${N_TASKS} (throttle ${MAX_CONCURRENT}), ~$(( N_UNITS / N_TASKS )) units each"
echo "budget:       ${BUDGET_S} s per graph, ${ENCODE_JOBS} encode jobs on ${CPUS} cores"
echo "out:          ${OUT_ROOT}"
echo "logs:         ${LOGS_DIR}/t06exh_%A_%a.out"

if ${DRY_RUN}; then
  printf '[DRY-RUN] sbatch'; printf ' %q' "${SBATCH_ARGS[@]}"; printf '\n'
  exit 0
fi

if ${TEST_ONLY}; then
  sbatch --test-only "${SBATCH_ARGS[@]}"
  exit $?
fi

# Read live cluster state before committing. Every number in the skill notes is
# account state that drifts; quota in particular is a FILE COUNT limit and a
# campaign that hits the hard limit at 80 % keeps burning wallclock while every
# write fails.
echo "--- live state ---"
quota 2>/dev/null || echo "(quota unavailable)"
squeue 2>/dev/null | tail -5 || true
echo "------------------"

RAW=$(sbatch --parsable "${SBATCH_ARGS[@]}") || { echo "sbatch failed" >&2; exit 1; }
JOB_ID=$(_clean_job_id "${RAW}")
if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "FATAL: unparsable job id: ${RAW@Q}" >&2
  echo "A submission may still have happened -- run 'squeue' before resubmitting." >&2
  exit 1
fi

echo "Submitted array ${JOB_ID} (${N_TASKS} tasks)"
echo "Monitor:   squeue"
echo "States:    sacct -j ${JOB_ID} -X -n -P -o JobID,State,Elapsed"
echo "Memory:    sacct -j ${JOB_ID} -n -P -o JobID,MaxRSS | grep '\.batch'"
echo "Logs:      ${LOGS_DIR}/t06exh_${JOB_ID}_*.out"
echo "Resume:    rerun this launcher -- a completed cell is skipped by the worker"
