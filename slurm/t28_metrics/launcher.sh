#!/usr/bin/env bash
# T-28 -- alternative similarity references for section 5.4. Picasso launcher.
#
#   bash slurm/t28_metrics/launcher.sh --dry-run   # print, submit nothing
#   bash slurm/t28_metrics/launcher.sh             # submit
#
# TWO STAGES, chained: f2-shards (array over the 15 (suite, dataset) cells) then
# one f2-merge with --dependency=afterok.
#
# THERE IS NO DISTANCE STAGE, and that is the point of the ticket. Every
# representation distance is T-04a's, already computed, and is reused from
# ${OUT_ROOT}/distances verbatim -- `levenshtein` for the six serialisations and
# `kernel` for wl_subtree. T-28 changes only the REFERENCE side of the
# correlation. Skipping the 12 h distance stage is what makes this fit the
# revision deadline.
#
# WHAT IS NEW: ${T28_REFERENCE_ROOT}, holding one dense NPZ per (cell, reference)
# for `wl` and the spectral family. The `wl` matrix is not recomputed either --
# it IS the cached wl_subtree kernel matrix, which makes the degeneracy under
# that reference exact rather than approximate, and it is excluded from the win
# counts for that reason.
#
# 🔴 A DEPENDENCY THAT sbatch ACCEPTS IS NOT A DEPENDENCY THAT TOOK. A malformed
# upstream id is accepted and recorded as Dependency=(null); the merge then runs
# IMMEDIATELY against an incomplete partial set. The link is verified with
# scontrol after submission and the chain is cancelled if it did not take.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
FSCRATCH="/mnt/home/users/tic_163_uma/mpascual/fscratch"
DATA="${FSCRATCH}/datasets/isalgraph"

export CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-${FSCRATCH}/conda_envs/isalgraph}"
# A SEPARATE checkout, deliberately. The shared repos/IsalGraph carries
# uncommitted local edits and is the target of the env's editable install; T-28
# must not disturb it. The only `import isalgraph` in the F2 chain is inside
# _metadata() for provenance, so resolving the package from the older tree
# changes no computed number -- but it does mean the manifest's src_commit
# describes that tree and not this one. Recorded, not silently accepted.
export REPO_DIR="${REPO_DIR:-${FSCRATCH}/repos/IsalGraph-t28}"

# Read-only inputs, all reused.
#
# 🔴 T-06's OWN tree, NOT T06_exhaustive's. Both hold the arms T-28 needs and the
# shared matrices are identical, but T06_exhaustive/encodings ALSO holds
# isalgraph_exhaustive and isalgraph_greedy. With T06_REFERENCE_ARM=isalgraph_pruned
# those two are undeclared, the completion table names them, and every shard dies
# in family.validate() with
#     FamilyError: c names an undeclared representation 'isalgraph_exhaustive'
# Caught by a login-node smoke run; queued, it would have surfaced after a ~10 h
# wait. Using T-06's own inputs is also what gate G1 compares against.
export OUT_ROOT="${OUT_ROOT:-${DATA}/T06}"
export GED_ROOT="${GED_ROOT:-${DATA}/eval/ged_matrices}"
export APPROX_ROOT="${APPROX_ROOT:-${DATA}/APPROX_GED}"

# T-28's own outputs.
export FAM_ROOT="${FAM_ROOT:-${DATA}/T28_metrics/families}"
export T28_REFERENCE_ROOT="${T28_REFERENCE_ROOT:-${DATA}/T28_metrics/references}"

LOGS_DIR="${LOGS_DIR:-${HOME}/execs/isalgraph/logs}"
ACCOUNT="${ACCOUNT:-tic_163_uma}"

# THE DEFAULT ARM. T-28 changes the reference, never the arm: section 5.4 reports
# isalgraph_pruned and gate G1 requires its GED columns to come back unchanged.
export T06_REFERENCE_ARM="${T06_REFERENCE_ARM:-isalgraph_pruned}"
export COMPARATOR_SET="${COMPARATOR_SET:-full}"

# Measured on this cluster: T06_exhaustive/families/family_F2.json reports 79,
# and so does T-06 itself. The merge ABORTS if T-28's run moves it -- see the
# worker's merge stage for why that is a guard failure and not a finding.
export EXPECTED_N_ACTUAL="${EXPECTED_N_ACTUAL:-79}"

# F2: single-process numpy per shard; a tier-3 shard peaks around 3 GB. T-28
# adds references to Claim B, so a shard does roughly 2x the correlation work of
# T-06's -- hence a wider wall than the 12 h that sufficed there.
F2_WALL="${F2_WALL:-0-20:00:00}"
F2_CPUS="${F2_CPUS:-4}"
F2_MEM="${F2_MEM:-32G}"
N_F2_TASKS="${N_F2_TASKS:-8}"
F2_CONCURRENT="${F2_CONCURRENT:-8}"
export START_CUTOFF_S="${START_CUTOFF_S:-54000}"   # 15 h of a 20 h wall

# QOS. Empty means the account default, which is medium_uma. Measured on this
# cluster 2026-08-29 with `sacctmgr show qos`:
#
#   short        priority 10000   MaxWall 02:00:00
#   medium_uma   priority  1000   MaxWall 3-00:00:00
#   long_uma     priority   500   MaxWall 7-00:00:00
#
# The site weights QOS at 100000, so the factor is qos_priority/10000 and the
# contribution is 100000 under `short` against 10000 under `medium_uma`. Job
# 2132238 sat on Priority/ for six hours at a total of 29,485 with 22 pending
# jobs ahead of it; +90,000 puts it at the top of that queue instead. The whole
# cost is the two-hour wall, so this is right for the light cells and wrong for
# mutagenicity and coil_del.
#
# The shard loop is idempotent -- it skips any cell whose partial already
# exists -- so a `short` array and a `medium_uma` array can target the SAME
# FAM_ROOT, and the second one only picks up what the first did not finish.
# Do not run them CONCURRENTLY against one FAM_ROOT: the skip test is a
# check-then-write, so two live arrays can both start the same cell and race on
# the partial. Hold one while the other runs.
F2_QOS="${F2_QOS:-}"

# Submit the shard array only. The merge needs all fifteen partials and aborts
# otherwise, so when the campaign is being completed in more than one pass the
# merge is submitted by hand once the fifteenth partial lands.
SKIP_MERGE="${SKIP_MERGE:-false}"

SUITE1=(linux aids iam_letter_low iam_letter_med iam_letter_high)
SUITE2=(linux grec protein aids_graphedx iam_letter_low iam_letter_med
        aids_iam iam_letter_high coil_del mutagenicity)

# Expensive first. The worker STRIDES over this list, so ordering by cost is what
# makes the round-robin deal balanced rather than dumping every heavy cell on
# task 0.
COST_ORDER=(mutagenicity coil_del iam_letter_high aids_iam aids_graphedx
            iam_letter_med iam_letter_low grec protein aids linux)

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ---- Build the shard unit list ---------------------------------------------
UNITS=()
for d in "${COST_ORDER[@]}"; do
    for s in "${SUITE2[@]}"; do [[ "$s" == "$d" ]] && UNITS+=( "suite2/$d" ) && break; done
done
for d in "${COST_ORDER[@]}"; do
    for s in "${SUITE1[@]}"; do [[ "$s" == "$d" ]] && UNITS+=( "suite1/$d" ) && break; done
done
EXPECTED_SHARDS=${#UNITS[@]}
F2_UNIT_LIST=$(IFS=':'; echo "${UNITS[*]}")

mkdir -p "${LOGS_DIR}" "${FAM_ROOT}"

# ---- Preflight --------------------------------------------------------------
# Each of these has killed a campaign on this cluster. Check them here, on the
# login node, where the failure costs a second rather than a queue wait.
fail_pre() { echo "[PREFLIGHT FAIL] $*" >&2; exit 2; }
[ -d "${REPO_DIR}" ]              || fail_pre "no repo at ${REPO_DIR}"
[ -x "${CONDA_PREFIX_DIR}/bin/python" ] || fail_pre "no interpreter at ${CONDA_PREFIX_DIR}/bin/python"
[ -d "${OUT_ROOT}/distances" ]    || fail_pre "no distances under ${OUT_ROOT}"
[ -d "${GED_ROOT}" ]              || fail_pre "no GED_ROOT at ${GED_ROOT}"
[ -d "${APPROX_ROOT}" ]           || fail_pre "no APPROX_ROOT at ${APPROX_ROOT}"
[ -d "${T28_REFERENCE_ROOT}" ]    || fail_pre "no T28_REFERENCE_ROOT at ${T28_REFERENCE_ROOT} -- build the reference matrices first"
N_T28=$(find "${T28_REFERENCE_ROOT}" -name '*__*.npz' 2>/dev/null | wc -l)
[ "${N_T28}" -ge 1 ] || fail_pre "${T28_REFERENCE_ROOT} holds no matrices; the run would be a silent no-op"
N_ARM=$(find "${OUT_ROOT}/distances" -name "*__${T06_REFERENCE_ARM}__levenshtein.npz" | wc -l)
[ "${N_ARM}" -ge 1 ] || fail_pre "no ${T06_REFERENCE_ARM} matrices under ${OUT_ROOT}/distances"

T28_KEYS=$(find "${T28_REFERENCE_ROOT}" -name '*__*.npz' | sed 's/.*__//; s/\.npz$//' | sort -u | tr '\n' ' ')

# ---- Job-id capture ---------------------------------------------------------
# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning to
# --parsable output. `sed 's/[^0-9]//g'` alone runs line-by-line and returns a
# MULTI-LINE "id"; the guard then trips AFTER the job was submitted, leaving an
# untracked job on the cluster. Take the last line first.
_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }
submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}

COMMON_EXPORT="CONDA_PREFIX_DIR=${CONDA_PREFIX_DIR},REPO_DIR=${REPO_DIR},OUT_ROOT=${OUT_ROOT},START_CUTOFF_S=${START_CUTOFF_S}"
F2_EXPORT="${COMMON_EXPORT},GED_ROOT=${GED_ROOT},APPROX_ROOT=${APPROX_ROOT},FAM_ROOT=${FAM_ROOT},T28_REFERENCE_ROOT=${T28_REFERENCE_ROOT},T06_REFERENCE_ARM=${T06_REFERENCE_ARM},COMPARATOR_SET=${COMPARATOR_SET},EXPECTED_SHARDS=${EXPECTED_SHARDS},EXPECTED_N_ACTUAL=${EXPECTED_N_ACTUAL}"

# Built as its own array so an unset F2_QOS contributes no argument at all.
# `--qos=` with an empty value is not the same as omitting the flag: sbatch
# takes it as a request for a QOS named "" and rejects the submission.
QOS_ARGS=()
[ -n "${F2_QOS}" ] && QOS_ARGS+=( --qos="${F2_QOS}" )

F2_SHARD_ARGS=(
  --job-name=t28f2s
  --array="0-$(( N_F2_TASKS - 1 ))%${F2_CONCURRENT}"
  --time="${F2_WALL}" --ntasks=1 --cpus-per-task="${F2_CPUS}" --mem="${F2_MEM}"
  ${QOS_ARGS[@]+"${QOS_ARGS[@]}"}
  --constraint=cpu --account="${ACCOUNT}" --chdir="${REPO_DIR}"
  --output="${LOGS_DIR}/t28f2s_%A_%a.out" --error="${LOGS_DIR}/t28f2s_%A_%a.err"
  --export="ALL,${F2_EXPORT},STAGE=shards,N_TASKS=${N_F2_TASKS},UNIT_LIST=${F2_UNIT_LIST}"
  "${SCRIPT_DIR}/f2_worker.sh"
)
F2_MERGE_ARGS=(
  --job-name=t28f2m
  --time=0-03:00:00 --ntasks=1 --cpus-per-task=4 --mem=32G
  --constraint=cpu --account="${ACCOUNT}" --chdir="${REPO_DIR}"
  --output="${LOGS_DIR}/t28f2m_%j.out" --error="${LOGS_DIR}/t28f2m_%j.err"
  --export="ALL,${F2_EXPORT},STAGE=merge,N_TASKS=1"
  "${SCRIPT_DIR}/f2_worker.sh"
)

cat <<EOF
reference arm:  ${T06_REFERENCE_ARM}   (unchanged -- gate G1)
comparators:    ${COMPARATOR_SET}
t28 references: ${N_T28} matrices; keys: ${T28_KEYS}
f2 shards:      ${EXPECTED_SHARDS} over ${N_F2_TASKS} tasks, then 1 merge
repo:           ${REPO_DIR}
distances (RO): ${OUT_ROOT}/distances   [${N_ARM} arm matrices -- NOT recomputed]
ged root:       ${GED_ROOT}
approx root:    ${APPROX_ROOT}
families out:   ${FAM_ROOT}
expected N_actual: ${EXPECTED_N_ACTUAL}  (asserted at merge)
chain:          f2-shards -> f2-merge
EOF

if ${DRY_RUN}; then
    echo ""
    echo "[DRY-RUN] sbatch ${F2_SHARD_ARGS[*]}"
    echo ""
    echo "[DRY-RUN] sbatch --dependency=afterok:<shards> ${F2_MERGE_ARGS[*]}"
    exit 0
fi

SHARDS=$(submit "${F2_SHARD_ARGS[@]}") || exit 1
echo "submitted f2-shards: ${SHARDS}"

if ${SKIP_MERGE}; then
    echo "submitted f2-merge:  SKIPPED (SKIP_MERGE=true)"
    echo ""
    echo "Fifteen partials are required before the merge will run. Check with"
    echo "  ls -1 ${FAM_ROOT}/f2_partials/*.json | wc -l"
    echo "and submit the merge by hand once it reads 15."
    echo "Monitor:  squeue --me"
    echo "Logs:     ${LOGS_DIR}/t28f2s_${SHARDS}_*.out"
    exit 0
fi

MERGE=$(submit --dependency="afterok:${SHARDS}" "${F2_MERGE_ARGS[@]}") || {
    echo "FATAL: merge submission failed; cancelling ${SHARDS}" >&2
    scancel "${SHARDS}"; exit 1
}
# The dependency must have TAKEN, not merely been accepted.
if scontrol show job "${MERGE}" | grep -q 'Dependency=(null)'; then
    echo "FATAL: dependency dropped on ${MERGE}; cancelling both" >&2
    scancel "${MERGE}" "${SHARDS}"; exit 1
fi
echo "submitted f2-merge:  ${MERGE}  (afterok:${SHARDS})"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ${LOGS_DIR}/t28f2s_${SHARDS}_*.out"
echo "Results:  ${FAM_ROOT}/rho_table.json"
