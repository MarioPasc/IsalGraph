#!/usr/bin/env bash
# Finish the two Suite-2 datasets the pooled campaigns could not: coil_del, mutagenicity.
#
#   bash finish_remaining.sh --dry-run
#   bash finish_remaining.sh --roles lb,ub,ubs
#
# 🔴 WHY A SECOND LAUNCHER. launcher.sh submits one pooled job per role over all ten
# datasets. That shape is retired for this workload: the pool is negative-scaling
# (T-05-design amendment 11), and lb/ub/ubs all hit their 12 h wallclock after 8 of 10
# datasets with 15/37/126 workers. The eight completed matrices are already merged and
# mirrored; recomputing them would be pure waste, so this launcher covers only the
# remainder and does it as an array of single-process pair ranges.
#
# SIZING. Rates below are SINGLE-PROCESS, random-sampled on an sr core in cohort mode --
# the only regime in which a per-pair number here means anything. lb is measured directly;
# ub and ubs are scaled by T-27 §5's method ratios (BIPARTITE 690/285 = 2.42x,
# BP_BEAM_DET 2322/285 = 8.15x, both already doubled for two-orientation symmetrisation).
# Those two are therefore ESTIMATES; the 12 h wallclock against a ~3 h target is the margin
# that covers them being wrong, and checkpointing bounds the loss if they are wrong badly.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph}"
FSCRATCH="/mnt/home/users/tic_163_uma/mpascual/fscratch"
OUT_DIR="${OUT_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/approx_ged}"
DATA_DIR="${DATA_DIR:-${FSCRATCH}/datasets/isalgraph/suite2}"
ACCOUNT="tic_163_uma"
CONSTRAINT="sr"
WALLCLOCK="0-12:00:00"
TARGET_SECONDS=10800          # ~3 h per task
FLOOR_SECONDS=7200            # SCBI, 2026-08-07. Non-negotiable.

DRY_RUN=0
ROLES="lb,ub,ubs"
while (( $# )); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift;;
        --roles)   ROLES="$2"; shift 2;;
        *) echo "unknown flag $1" >&2; exit 2;;
    esac
done

# dataset -> pairs, graphs
declare -A PAIRS=( [coil_del]=7603050 [mutagenicity]=8158780 )
declare -A GRAPHS=( [coil_del]=3900   [mutagenicity]=4040 )
# dataset -> measured single-process lb microseconds/pair (cohort mode, sr core)
declare -A LB_US=( [coil_del]=4315    [mutagenicity]=3107 )
# role -> multiplier on the lb rate
declare -A ROLE_MULT=( [lb]=1.0 [ub]=2.42 [ubs]=8.15 )

_clean_job_id() { tr -cd '0-9\n' <<< "$1" | grep -E '^[0-9]+$' | tail -1; }

submit_one() {
    local role="$1" key="$2"
    local pairs="${PAIRS[$key]}" graphs="${GRAPHS[$key]}"
    local core_s n_tasks per_task
    core_s=$(awk -v p="$pairs" -v u="${LB_US[$key]}" -v m="${ROLE_MULT[$role]}" \
             'BEGIN{printf "%d", p*u*m/1000000}')
    n_tasks=$(awk -v c="$core_s" -v t="$TARGET_SECONDS" 'BEGIN{n=int(c/t); print (n<1?1:n)}')
    # Never submit a task under the floor: shrink the task count until each clears it.
    while (( n_tasks > 1 )) && (( core_s / n_tasks < FLOOR_SECONDS )); do n_tasks=$((n_tasks-1)); done
    per_task=$(( core_s / n_tasks ))
    printf '  %-4s %-13s %10d pairs %8d core-s -> %2d tasks x %.2f h\n' \
        "$role" "$key" "$pairs" "$core_s" "$n_tasks" "$(awk -v s="$per_task" 'BEGIN{print s/3600}')"
    if (( per_task < FLOOR_SECONDS )); then
        echo "    REFUSING: ${per_task}s/task is under the ${FLOOR_SECONDS}s floor" >&2
        return 3
    fi

    local exports="ALL,REPO_DIR=${REPO_DIR},CONDA_ENV_PREFIX=${FSCRATCH}/conda_envs/isalgraph"
    exports+=",GEDLIB_DIR=${FSCRATCH}/build_gedlib/graphkit-learn,DATA_DIR=${DATA_DIR}"
    exports+=",OUT_DIR=${OUT_DIR},ROLE=${role},KEY=${key},N_CHUNKS=${n_tasks}"
    exports+=",ISALGRAPH_CODE_COMMIT=${ISALGRAPH_CODE_COMMIT:-unknown}"

    if (( DRY_RUN )); then echo "    [DRY-RUN] array 0-$((n_tasks-1)) + afterok merge"; return 0; fi

    local aid mid
    aid=$(sbatch --parsable --job-name="ag-${role}-${key}" --account="${ACCOUNT}" \
        --time="${WALLCLOCK}" --ntasks=1 --cpus-per-task=1 --mem=16G --constraint="${CONSTRAINT}" \
        --array="0-$((n_tasks-1))%${n_tasks}" \
        --output="${OUT_DIR}/logs/ag-${role}-${key}_%A_%a.out" \
        --error="${OUT_DIR}/logs/ag-${role}-${key}_%A_%a.err" \
        --export="${exports}" "${REPO_DIR}/slurm/approx_ged/worker_range.sh" 2>&1)
    aid=$(_clean_job_id "$aid")
    mid=$(sbatch --parsable --job-name="agm-${role}-${key}" --account="${ACCOUNT}" \
        --time="0-04:00:00" --ntasks=1 --cpus-per-task=2 --mem=64G --constraint="${CONSTRAINT}" \
        --dependency="afterok:${aid}" \
        --output="${OUT_DIR}/logs/agm-${role}-${key}_%j.out" \
        --error="${OUT_DIR}/logs/agm-${role}-${key}_%j.err" \
        --export="${exports},N_GRAPHS=${graphs}" \
        "${REPO_DIR}/slurm/approx_ged/worker_merge_range.sh" 2>&1)
    mid=$(_clean_job_id "$mid")
    echo "    array=${aid}  merge=${mid} (afterok)"
}

mkdir -p "${OUT_DIR}/logs"
echo "finishing coil_del + mutagenicity, roles=${ROLES}, floor=${FLOOR_SECONDS}s target=${TARGET_SECONDS}s"
IFS=',' read -r -a ROLE_LIST <<< "${ROLES}"
for role in "${ROLE_LIST[@]}"; do
    for key in coil_del mutagenicity; do
        submit_one "${role}" "${key}"
    done
done
