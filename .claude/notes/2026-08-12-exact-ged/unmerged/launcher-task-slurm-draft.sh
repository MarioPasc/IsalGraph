#!/usr/bin/env bash
# =============================================================================
# IsalGraph T-03 -- exact GED on Picasso.  The single human entry point.
# =============================================================================
#
# Submits the five jobs of T-03-design.md section 6, in order, with dependencies:
#
#   gates  ->  small  ->  aids1  ->  aids2  ->  merge
#
# Every task is sized so that it runs at least --min-hours (SCBI's two-hour
# floor; Manuel, soporte@scbi.uma.es, 2026-08-07).  The launcher REFUSES to
# submit a stage whose projection falls under that floor; it reduces the task
# count -- or, for a single-task stage where no task-count lever exists, the
# core count -- until the projection clears it, and stops if it cannot.
#
# Usage
#   bash slurm/exact_ged/launcher.sh --dry-run --stage all --sec-per-pair 12
#   bash slurm/exact_ged/launcher.sh --stage gates
#   bash slurm/exact_ged/launcher.sh --stage aids1 --sec-per-pair 6.5 --stage1-pairs 22106
#   bash slurm/exact_ged/launcher.sh --stage merge --after 1234567
#
# The per-pair rate is NEVER hard-coded.  Pass the figure measured by the gate
# job (`--sec-per-pair`); the small datasets have their own, far smaller rate
# (`--sec-per-pair-small`) because Letter/LINUX graphs are a third the size of
# AIDS graphs and A* is superexponential in node count.
#
# This script only ever calls `sbatch`.  It never runs the payload itself.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Cluster facts, measured 2026-08-12.  Override on the command line, not here.
# -----------------------------------------------------------------------------
REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph"
ENV_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalgraph"
GEDLIB_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/build_gedlib/graphkit-learn"
DATA_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph/exact_ged"
WORK_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph/exact_ged/work"
EXEC_ROOT="/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/exact_ged"

ACCOUNT="tic_163_uma"
CONSTRAINT="sr"          # 128c/450G AMD EPYC 7H12, ~45 idle.  Pinned: per-pair
                         # wall time is a REPORTED quantity (D12 censoring), so
                         # a mixed Intel/AMD pool would measure the scheduler.
QOS_LONG="medium_uma"    # 3 d cap; every compute stage lives here
QOS_SHORT="short"        # 2 h cap; the merge barrier only

# -----------------------------------------------------------------------------
# Cohort -- T-03-design.md section 2.  Locked; a mismatch stops the ticket.
# -----------------------------------------------------------------------------
SMALL_KEYS="iam_letter_low:iam_letter_med:iam_letter_high:linux"
SMALL_NGRAPHS="1180:1253:2059:89"
SMALL_PAIRS=3602615      # 695610 + 784378 + 2118711 + 3916
AIDS_KEY="aids"
AIDS_NGRAPHS=769
AIDS_PAIRS=295296
ALL_KEYS="iam_letter_low:iam_letter_med:iam_letter_high:linux:aids"
ALL_NGRAPHS="1180:1253:2059:89:769"

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
STAGE="all"
DRY_RUN=false
SEC_PER_PAIR=""          # AIDS rate; required for aids1/aids2/all
SEC_PER_PAIR_SMALL="0.02"
TARGET_HOURS="2.5"
MIN_HOURS="2.0"
MAX_CONCURRENT=16
CORES=64                 # AIDS worker cores, and the ceiling when auto-sizing
SMALL_CORES=8
GATES_CORES=4
MERGE_CORES=2
GATES_HOURS="4.0"        # fixed workload (design section 6 estimates 2.5-3.5 h)
TIME_MARGIN="2.0"        # wall-clock request = margin x projection
STAGE1_PAIRS=23000       # projection only; real value from sampling_report.json
BACKEND="networkx"       # amendment 2: ANCHOR_AWARE_GED retired, A* is exact
GATES_BACKEND="gedlib"
COST_MODEL="unit"
TIMEOUT_PER_PAIR="300"
CHECKPOINT_EVERY="2000"
GATE_N_PAIRS=500
GATE_SEED=42
GATE_MAX_NODES=10
GATE_N_MAX=12
SAMPLING_K=180
SAMPLING_Q=10
SAMPLING_F=30
SAMPLING_SEED=42
EXTERNAL_DEP=""
FORCE_SHORT=false

MAX_ARRAY_SIZE=4096      # scontrol show config: MaxArraySize
QOS_LONG_CAP=259200      # 3 d, in seconds
QOS_SHORT_CAP=7200       # 2 h, in seconds

# -----------------------------------------------------------------------------
# Plumbing
# -----------------------------------------------------------------------------
log()   { printf '%s\n' "$*"; }
warn()  { printf 'WARNING: %s\n' "$*" >&2; }
fatal() { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options
  --stage {gates,small,aids1,aids2,merge,all}   default: all
  --sec-per-pair <float>      measured AIDS seconds/pair; required for aids stages
  --sec-per-pair-small <float>  default 0.02, the Letter/LINUX rate
  --target-hours <float>      sizing target per task, default 2.5
  --min-hours <float>         hard floor below which we refuse, default 2.0
  --max-concurrent <int>      array throttle (%N), default 16
  --cores <int>               AIDS cores per task / auto-size ceiling, default 64
  --small-cores <int>         default 8
  --gates-cores <int>         default 4
  --merge-cores <int>         default 2
  --gates-hours <float>       wall clock for the fixed gate workload, default 4.0
  --time-margin <float>       wall request = margin x projection, default 2.0
  --stage1-pairs <int>        stage-1 sample size for sizing, default 23000
  --backend <name>            production solver, default networkx (amendment 2)
  --gates-backend <name>      default gedlib
  --cost-model {unit,graphedx}  default unit
  --timeout-per-pair <float>  default 300
  --after <jobid>             attach an external dependency to a single stage
  --force-short               submit even under the floor (loud; you own SCBI)
  --dry-run                   print every sbatch line, submit nothing
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)              STAGE="$2"; shift 2 ;;
        --sec-per-pair)       SEC_PER_PAIR="$2"; shift 2 ;;
        --sec-per-pair-small) SEC_PER_PAIR_SMALL="$2"; shift 2 ;;
        --target-hours)       TARGET_HOURS="$2"; shift 2 ;;
        --min-hours)          MIN_HOURS="$2"; shift 2 ;;
        --max-concurrent)     MAX_CONCURRENT="$2"; shift 2 ;;
        --cores)              CORES="$2"; shift 2 ;;
        --small-cores)        SMALL_CORES="$2"; shift 2 ;;
        --gates-cores)        GATES_CORES="$2"; shift 2 ;;
        --merge-cores)        MERGE_CORES="$2"; shift 2 ;;
        --gates-hours)        GATES_HOURS="$2"; shift 2 ;;
        --time-margin)        TIME_MARGIN="$2"; shift 2 ;;
        --stage1-pairs)       STAGE1_PAIRS="$2"; shift 2 ;;
        --backend)            BACKEND="$2"; shift 2 ;;
        --gates-backend)      GATES_BACKEND="$2"; shift 2 ;;
        --cost-model)         COST_MODEL="$2"; shift 2 ;;
        --timeout-per-pair)   TIMEOUT_PER_PAIR="$2"; shift 2 ;;
        --repo-dir)           REPO_DIR="$2"; shift 2 ;;
        --env-dir)            ENV_DIR="$2"; shift 2 ;;
        --gedlib-dir)         GEDLIB_DIR="$2"; shift 2 ;;
        --data-dir)           DATA_DIR="$2"; shift 2 ;;
        --work-dir)           WORK_DIR="$2"; shift 2 ;;
        --exec-root)          EXEC_ROOT="$2"; shift 2 ;;
        --account)            ACCOUNT="$2"; shift 2 ;;
        --constraint)         CONSTRAINT="$2"; shift 2 ;;
        --after)              EXTERNAL_DEP="$2"; shift 2 ;;
        --force-short)        FORCE_SHORT=true; shift ;;
        --dry-run)            DRY_RUN=true; shift ;;
        -h|--help)            usage; exit 0 ;;
        *) fatal "unknown argument: $1  (try --help)" ;;
    esac
done

case "${STAGE}" in
    gates|small|aids1|aids2|merge|all) ;;
    *) fatal "--stage must be one of gates, small, aids1, aids2, merge, all" ;;
esac

LOG_DIR="${EXEC_ROOT}/logs"
TARGET_SECONDS="$(awk -v h="${TARGET_HOURS}" 'BEGIN{printf "%.0f", h*3600}')"
MIN_SECONDS="$(awk -v h="${MIN_HOURS}" 'BEGIN{printf "%.0f", h*3600}')"

if (( MIN_SECONDS > TARGET_SECONDS )); then
    fatal "--min-hours (${MIN_HOURS}) exceeds --target-hours (${TARGET_HOURS})"
fi

needs_rate=false
case "${STAGE}" in aids1|aids2|all) needs_rate=true ;; esac
if [[ "${needs_rate}" == true && -z "${SEC_PER_PAIR}" ]]; then
    fatal "--sec-per-pair is required for stage '${STAGE}'.  Do not guess it: take the
       median seconds/pair from the gate job's report and pass it.  Measured on
       real AIDS 2026-08-12: median 6.5 s, mean 10.1 s, ~27 % over 30 s."
fi

# -----------------------------------------------------------------------------
# Trap 3 -- sbatch --export splits on EVERY comma.  A comma inside a value
# truncates it and the tail silently becomes a junk variable.  Every list-valued
# export is colon-separated, and add_export refuses a comma outright.
# -----------------------------------------------------------------------------
EXPORTS=()
reset_exports() { EXPORTS=(); }
add_export() {
    local name="$1" value="$2"
    if [[ "${value}" == *,* ]]; then
        fatal "export ${name} contains a comma ('${value}'). sbatch --export splits on
       every comma, so this would truncate silently. Use colon separation."
    fi
    EXPORTS+=( "${name}=${value}" )
}
join_exports() {
    local IFS=','
    printf 'ALL,%s' "${EXPORTS[*]}"
}
add_common_exports() {
    add_export EG_REPO_DIR         "${REPO_DIR}"
    add_export EG_ENV_DIR          "${ENV_DIR}"
    add_export EG_GEDLIB_DIR       "${GEDLIB_DIR}"
    add_export EG_DATA_DIR         "${DATA_DIR}"
    add_export EG_WORK_DIR         "${WORK_DIR}"
    add_export EG_EXEC_ROOT        "${EXEC_ROOT}"
    add_export EG_LOG_DIR          "${LOG_DIR}"
}

# -----------------------------------------------------------------------------
# Trap 2 -- `sbatch --parsable` does NOT return a bare id on Picasso.  The Lua
# wrapper prepends ANSI colour and a multi-line warning.  A line-oriented
# `sed 's/[^0-9]//g'` leaves a MULTI-LINE "id", the guard then fires after the
# job was already submitted, and an untracked job keeps running.  Take the last
# line FIRST, strip ANSI, strip non-digits, then assert.  One helper, always.
# -----------------------------------------------------------------------------
LAST_JOB_ID=""
print_cmd() {
    local first=1 tok
    for tok in "$@"; do
        if (( first )); then printf '  %s' "${tok}"; first=0
        else printf ' \\\n      %s' "${tok}"; fi
    done
    printf '\n'
}
submit() {
    local label="$1"; shift
    local -a cmd=( sbatch --parsable "$@" )
    printf '\n### %s\n' "${label}"
    print_cmd "${cmd[@]}"
    if [[ "${DRY_RUN}" == true ]]; then
        LAST_JOB_ID="<${label}-jobid>"
        return 0
    fi
    local raw id
    if ! raw="$( "${cmd[@]}" 2>&1 )"; then
        fatal "sbatch rejected ${label}:
${raw}"
    fi
    id="$(printf '%s\n' "${raw}" | tail -n 1 \
          | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g')"
    if [[ ! "${id}" =~ ^[0-9]+$ ]]; then
        fatal "could not parse a job id for ${label}. sbatch said:
${raw}
Parsed: '${id}'.  A job MAY have been submitted -- check 'squeue' (bare, not -u)
before resubmitting."
    fi
    LAST_JOB_ID="${id}"
    log "    submitted ${label} as job ${id}"
}

# -----------------------------------------------------------------------------
# Trap 7 -- a dependency built from a bad id is ACCEPTED, recorded as
# Dependency=(null), and the dependent job starts IMMEDIATELY against partial
# input.  Assert after every dependent submit.
# -----------------------------------------------------------------------------
assert_dependency() {
    local id="$1" label="$2"
    [[ "${DRY_RUN}" == true ]] && return 0
    local info
    info="$(scontrol show job "${id}" 2>&1 || true)"
    if grep -q 'Dependency=(null)' <<<"${info}"; then
        fatal "job ${id} (${label}) was accepted with Dependency=(null): SLURM discarded
       the dependency and this job will start against PARTIAL input.
       scancel ${id} now."
    fi
    log "    dependency ok: $(grep -o 'Dependency=[^ ]*' <<<"${info}" | head -n 1)"
}

# -----------------------------------------------------------------------------
# Sizing.  Trap 1 -- SCBI's two-hour floor is binding, so we size in TASKS, not
# in units of work, and refuse rather than emit a short task.
#
#   n_tasks = clamp( floor( pairs * sec_per_pair / (cores * target_seconds) ),
#                    1, MaxArraySize )
#
# The floor check is not redundant: whenever floor(.) >= 1 the projection is
# >= target by construction, so the only way to fall short is a stage whose
# TOTAL work is under one target-length task.  That is precisely the case the
# design bundles away (LINUX alone is 18 min, hence the four-dataset small job).
# -----------------------------------------------------------------------------
PLAN_TASKS=0
PLAN_SECONDS=0
PLAN_CORES=0

plan_array() {   # plan_array <pairs> <sec_per_pair> <cores>
    local pairs="$1" spp="$2" cores="$3"
    read -r PLAN_TASKS PLAN_SECONDS <<<"$(
        awk -v p="${pairs}" -v s="${spp}" -v c="${cores}" \
            -v t="${TARGET_SECONDS}" -v m="${MAX_ARRAY_SIZE}" 'BEGIN{
            total = p * s;
            n = int(total / (c * t));
            if (n < 1) n = 1;
            if (n > m) n = m;
            printf "%d %.0f", n, total / (c * n);
        }')"
    PLAN_CORES="${cores}"
}

plan_single() {  # plan_single <pairs> <sec_per_pair> <cores_max> -- one task, size the CORES
    local pairs="$1" spp="$2" cmax="$3"
    read -r PLAN_CORES PLAN_SECONDS <<<"$(
        awk -v p="${pairs}" -v s="${spp}" -v cmax="${cmax}" -v t="${TARGET_SECONDS}" 'BEGIN{
            total = p * s;
            c = int(total / t);
            if (c < 1) c = 1;
            if (c > cmax) c = cmax;
            printf "%d %.0f", c, total / c;
        }')"
    PLAN_TASKS=1
}

check_floor() {  # check_floor <label> <remedy-hint>
    local label="$1" hint="$2"
    if (( PLAN_SECONDS >= MIN_SECONDS )); then return 0; fi
    local msg
    msg="stage '${label}' projects ${PLAN_SECONDS} s per task ($(hours "${PLAN_SECONDS}") h),
       under the ${MIN_HOURS} h SCBI floor. ${hint}"
    if [[ "${FORCE_SHORT}" == true ]]; then
        warn "${msg}
       --force-short given: submitting anyway. This account has already been
       written to about short jobs; you own that conversation."
        return 0
    fi
    fatal "${msg}
       Refusing to submit. Re-run with a corrected --sec-per-pair, bundle this
       stage with another, or pass --force-short if you accept the cost."
}

hours() { awk -v s="$1" 'BEGIN{printf "%.2f", s/3600}'; }

fmt_hms() {      # fmt_hms <seconds> <cap_seconds>  -- round up to the next 15 min
    awk -v s="$1" -v cap="$2" 'BEGIN{
        q = 900;
        n = int((s + q - 1) / q) * q;
        if (n < q) n = q;
        if (n > cap) n = cap;
        printf "%02d:%02d:%02d", int(n/3600), int((n%3600)/60), n%60;
    }'
}

request_time() { # request_time <projected_seconds> <cap>
    awk -v s="$1" -v m="${TIME_MARGIN}" 'BEGIN{printf "%.0f", s*m}'
}

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
for w in worker_gates worker_small worker_aids worker_merge; do
    [[ -f "${SCRIPT_DIR}/${w}.sh" ]] || fatal "missing worker script ${SCRIPT_DIR}/${w}.sh"
done

WORKER_DIR="${REPO_DIR}/slurm/exact_ged"
if [[ "${DRY_RUN}" == false ]]; then
    command -v sbatch >/dev/null 2>&1 || fatal "sbatch not on PATH -- are you on a Picasso login node?"
    [[ -d "${REPO_DIR}" ]]  || fatal "repo not found at ${REPO_DIR}"
    [[ -x "${ENV_DIR}/bin/python" ]] || fatal "no interpreter at ${ENV_DIR}/bin/python"
    [[ -d "${DATA_DIR}" ]]  || fatal "CONTRACT A directory not found at ${DATA_DIR}"
    for w in worker_gates worker_small worker_aids worker_merge; do
        [[ -f "${WORKER_DIR}/${w}.sh" ]] || fatal "missing ${WORKER_DIR}/${w}.sh -- rsync the repo first"
    done
    mkdir -p "${LOG_DIR}" "${WORK_DIR}" "${EXEC_ROOT}"
else
    log "# dry run: no directory is created and nothing is submitted."
    log "# on the cluster the launcher would first run:"
    log "  mkdir -p ${LOG_DIR} ${WORK_DIR} ${EXEC_ROOT}"
fi

log ""
log "=============================================================================="
log "IsalGraph T-03 -- exact GED launcher"
log "=============================================================================="
log "  stage            ${STAGE}"
log "  dry run          ${DRY_RUN}"
log "  constraint       ${CONSTRAINT}   account ${ACCOUNT}"
log "  AIDS s/pair      ${SEC_PER_PAIR:-<unset>}"
log "  small s/pair     ${SEC_PER_PAIR_SMALL}"
log "  target / floor   ${TARGET_HOURS} h / ${MIN_HOURS} h per task"
log "  solver           ${BACKEND}   cost model ${COST_MODEL}"
log "  repo             ${REPO_DIR}"
log "  data             ${DATA_DIR}"
log "  work             ${WORK_DIR}"
log "  logs             ${LOG_DIR}"
log "=============================================================================="

# -----------------------------------------------------------------------------
# Stage builders
# -----------------------------------------------------------------------------
JID_GATES=""; JID_SMALL=""; JID_AIDS1=""; JID_AIDS2=""

submit_gates() {
    local tlimit
    tlimit="$(fmt_hms "$(awk -v h="${GATES_HOURS}" 'BEGIN{printf "%.0f", h*3600}')" "${QOS_LONG_CAP}")"
    log ""
    log "-- gates: fixed workload, ${GATES_CORES} cores, wall ${tlimit}"
    log "   (no pair-rate derivation: the gate suite's size is set by --n-pairs,"
    log "    not by the cohort. Design section 6 estimates 2.5-3.5 h.)"
    reset_exports
    add_common_exports
    add_export EG_GATE            "all"
    add_export EG_GATE_BACKEND    "${GATES_BACKEND}"
    add_export EG_WORKERS         "${GATES_CORES}"
    add_export EG_N_PAIRS         "${GATE_N_PAIRS}"
    add_export EG_SEED            "${GATE_SEED}"
    add_export EG_TIMEOUT_PER_PAIR "${TIMEOUT_PER_PAIR}"
    add_export EG_MAX_NODES       "${GATE_MAX_NODES}"
    add_export EG_N_MAX           "${GATE_N_MAX}"
    local -a args=(
        --job-name=eg-gates
        --account="${ACCOUNT}" --qos="${QOS_LONG}" --constraint="${CONSTRAINT}"
        --nodes=1 --ntasks=1 --cpus-per-task="${GATES_CORES}" --mem-per-cpu=4G
        --time="${tlimit}"
        --output="${LOG_DIR}/eg-gates_%j.out"
        --error="${LOG_DIR}/eg-gates_%j.err"
        --export="$(join_exports)"
    )
    [[ -n "${EXTERNAL_DEP}" ]] && args+=( --dependency="afterok:${EXTERNAL_DEP}" )
    submit "eg-gates" "${args[@]}" "${WORKER_DIR}/worker_gates.sh"
    JID_GATES="${LAST_JOB_ID}"
    [[ -n "${EXTERNAL_DEP}" ]] && assert_dependency "${JID_GATES}" "eg-gates"
    return 0
}

submit_small() {
    local dep="$1"
    plan_array "${SMALL_PAIRS}" "${SEC_PER_PAIR_SMALL}" "${SMALL_CORES}"
    PLAN_TASKS=1                                   # one job, four datasets in sequence
    PLAN_SECONDS="$(awk -v p="${SMALL_PAIRS}" -v s="${SEC_PER_PAIR_SMALL}" \
                        -v c="${SMALL_CORES}" 'BEGIN{printf "%.0f", p*s/c}')"
    check_floor "small" "LINUX alone is ~18 min; the design already bundles it with the
       three Letter sets for exactly this reason. If it is still short, raise
       --sec-per-pair-small to the measured Letter rate or lower --small-cores."
    local tlimit
    tlimit="$(fmt_hms "$(request_time "${PLAN_SECONDS}")" "${QOS_LONG_CAP}")"
    log ""
    log "-- small: ${SMALL_PAIRS} pairs x ${SEC_PER_PAIR_SMALL} s / ${SMALL_CORES} cores"
    log "          = ${PLAN_SECONDS} s = $(hours "${PLAN_SECONDS}") h projected, wall ${tlimit}"
    reset_exports
    add_common_exports
    add_export EG_KEYS             "${SMALL_KEYS}"
    add_export EG_NGRAPHS          "${SMALL_NGRAPHS}"
    add_export EG_BACKEND          "${BACKEND}"
    add_export EG_COST_MODEL       "${COST_MODEL}"
    add_export EG_WORKERS          "${SMALL_CORES}"
    add_export EG_TIMEOUT_PER_PAIR "${TIMEOUT_PER_PAIR}"
    add_export EG_CHECKPOINT_EVERY "${CHECKPOINT_EVERY}"
    local -a args=(
        --job-name=eg-small
        --account="${ACCOUNT}" --qos="${QOS_LONG}" --constraint="${CONSTRAINT}"
        --nodes=1 --ntasks=1 --cpus-per-task="${SMALL_CORES}" --mem-per-cpu=4G
        --time="${tlimit}" --signal=B:TERM@120
        --output="${LOG_DIR}/eg-small_%j.out"
        --error="${LOG_DIR}/eg-small_%j.err"
        --export="$(join_exports)"
    )
    [[ -n "${dep}" ]] && args+=( --dependency="afterok:${dep}" )
    submit "eg-small" "${args[@]}" "${WORKER_DIR}/worker_small.sh"
    JID_SMALL="${LAST_JOB_ID}"
    [[ -n "${dep}" ]] && assert_dependency "${JID_SMALL}" "eg-small"
    return 0
}

submit_aids1() {
    local dep="$1"
    plan_single "${STAGE1_PAIRS}" "${SEC_PER_PAIR}" "${CORES}"
    check_floor "aids1" "Stage 1 is a single pre-registered sample, so there is no
       task-count lever; the core count is the lever and it is already at 1."
    local tlimit
    tlimit="$(fmt_hms "$(request_time "${PLAN_SECONDS}")" "${QOS_LONG_CAP}")"
    log ""
    log "-- aids1: ${STAGE1_PAIRS} pairs x ${SEC_PER_PAIR} s = $(awk -v p="${STAGE1_PAIRS}" \
         -v s="${SEC_PER_PAIR}" 'BEGIN{printf "%.0f", p*s}') core-s"
    log "          single task, cores auto-sized to ${PLAN_CORES} (ceiling ${CORES})"
    log "          = ${PLAN_SECONDS} s = $(hours "${PLAN_SECONDS}") h projected, wall ${tlimit}"
    if (( PLAN_CORES < CORES )); then
        log "   NOTE: ${PLAN_CORES} < ${CORES} cores. Per-pair wall time is a reported"
        log "         quantity, and 30 vs 64 concurrent solvers on one node is not the"
        log "         same contention regime as stage 2. If you need stage 1 and stage 2"
        log "         timings to be directly comparable, pass --cores ${PLAN_CORES} so"
        log "         both stages run at the same width, and accept the longer census."
    fi
    reset_exports
    add_common_exports
    add_export EG_KEY              "${AIDS_KEY}"
    add_export EG_STAGE            "1"
    add_export EG_N_CHUNKS         "1"
    add_export EG_BACKEND          "${BACKEND}"
    add_export EG_COST_MODEL       "${COST_MODEL}"
    add_export EG_WORKERS          "${PLAN_CORES}"
    add_export EG_TIMEOUT_PER_PAIR "${TIMEOUT_PER_PAIR}"
    add_export EG_CHECKPOINT_EVERY "${CHECKPOINT_EVERY}"
    add_export EG_PAIR_LIST        "${WORK_DIR}/pairs/${AIDS_KEY}_stage1_pair_list.npz"
    add_export EG_SAMPLING_REPORT  "${WORK_DIR}/pairs/${AIDS_KEY}_sampling_report.json"
    add_export EG_SAMPLING_K       "${SAMPLING_K}"
    add_export EG_SAMPLING_Q       "${SAMPLING_Q}"
    add_export EG_SAMPLING_F       "${SAMPLING_F}"
    add_export EG_SAMPLING_SEED    "${SAMPLING_SEED}"
    local -a args=(
        --job-name=eg-aids-s1
        --account="${ACCOUNT}" --qos="${QOS_LONG}" --constraint="${CONSTRAINT}"
        --nodes=1 --ntasks=1 --cpus-per-task="${PLAN_CORES}" --mem-per-cpu=3500M
        --array="0-0%${MAX_CONCURRENT}"
        --time="${tlimit}" --signal=B:TERM@120
        --output="${LOG_DIR}/eg-aids-s1_%A_%a.out"
        --error="${LOG_DIR}/eg-aids-s1_%A_%a.err"
        --export="$(join_exports)"
    )
    [[ -n "${dep}" ]] && args+=( --dependency="afterok:${dep}" )
    submit "eg-aids-s1" "${args[@]}" "${WORKER_DIR}/worker_aids.sh"
    JID_AIDS1="${LAST_JOB_ID}"
    [[ -n "${dep}" ]] && assert_dependency "${JID_AIDS1}" "eg-aids-s1"
    return 0
}

submit_aids2() {
    local dep="$1"
    local census
    census=$(( AIDS_PAIRS - STAGE1_PAIRS ))
    (( census > 0 )) || fatal "--stage1-pairs ${STAGE1_PAIRS} >= the AIDS census ${AIDS_PAIRS}"
    plan_array "${census}" "${SEC_PER_PAIR}" "${CORES}"
    check_floor "aids2" "Reduce the task count by lowering --target-hours, or check that
       --sec-per-pair really is the measured figure."
    local tlimit last
    tlimit="$(fmt_hms "$(request_time "${PLAN_SECONDS}")" "${QOS_LONG_CAP}")"
    last=$(( PLAN_TASKS - 1 ))
    log ""
    log "-- aids2: census ${AIDS_PAIRS} - stage1 ${STAGE1_PAIRS} = ${census} pairs to compute"
    log "          ${census} x ${SEC_PER_PAIR} s = $(awk -v p="${census}" -v s="${SEC_PER_PAIR}" \
         'BEGIN{printf "%.0f", p*s}') core-s over ${CORES} cores"
    log "          n_tasks = floor(core-s / (${CORES} x ${TARGET_SECONDS})) = ${PLAN_TASKS}"
    log "          = ${PLAN_SECONDS} s = $(hours "${PLAN_SECONDS}") h per task, wall ${tlimit}"
    log "          array 0-${last}%${MAX_CONCURRENT}; chunking spans the FULL upper"
    log "          triangle and --seed-from skips the stage-1 pairs inside each chunk."
    reset_exports
    add_common_exports
    add_export EG_KEY              "${AIDS_KEY}"
    add_export EG_STAGE            "2"
    add_export EG_N_CHUNKS         "${PLAN_TASKS}"
    add_export EG_BACKEND          "${BACKEND}"
    add_export EG_COST_MODEL       "${COST_MODEL}"
    add_export EG_WORKERS          "${CORES}"
    add_export EG_TIMEOUT_PER_PAIR "${TIMEOUT_PER_PAIR}"
    add_export EG_CHECKPOINT_EVERY "${CHECKPOINT_EVERY}"
    add_export EG_SEED_FROM        "${WORK_DIR}/shards/${AIDS_KEY}/${AIDS_KEY}_s1_c0000.npz"
    local -a args=(
        --job-name=eg-aids-s2
        --account="${ACCOUNT}" --qos="${QOS_LONG}" --constraint="${CONSTRAINT}"
        --nodes=1 --ntasks=1 --cpus-per-task="${CORES}" --mem-per-cpu=3500M
        --array="0-${last}%${MAX_CONCURRENT}"
        --time="${tlimit}" --signal=B:TERM@120
        --output="${LOG_DIR}/eg-aids-s2_%A_%a.out"
        --error="${LOG_DIR}/eg-aids-s2_%A_%a.err"
        --export="$(join_exports)"
    )
    [[ -n "${dep}" ]] && args+=( --dependency="afterok:${dep}" )
    submit "eg-aids-s2" "${args[@]}" "${WORKER_DIR}/worker_aids.sh"
    JID_AIDS2="${LAST_JOB_ID}"
    [[ -n "${dep}" ]] && assert_dependency "${JID_AIDS2}" "eg-aids-s2"
    return 0
}

submit_merge() {
    local dep="$1"
    log ""
    log "-- merge: ${MERGE_CORES} cores, QOS ${QOS_SHARP:-${QOS_SHORT}}, minutes of work."
    log "   This is the ONE job that does not clear the two-hour floor, deliberately:"
    log "   it is a structural barrier that must wait on 'afterany' of the array, so it"
    log "   cannot be bundled into anything. It is 2 cores; the placement cost is real"
    log "   but bounded, and the alternative is no gate 4."
    reset_exports
    add_common_exports
    add_export EG_MERGE_KEYS     "${ALL_KEYS}"
    add_export EG_MERGE_NGRAPHS  "${ALL_NGRAPHS}"
    add_export EG_DELETE_SHARDS  "1"
    add_export EG_RSYNC          "1"
    local -a args=(
        --job-name=eg-merge
        --account="${ACCOUNT}" --qos="${QOS_SHORT}" --constraint="${CONSTRAINT}"
        --nodes=1 --ntasks=1 --cpus-per-task="${MERGE_CORES}" --mem-per-cpu=8G
        --time="$(fmt_hms "${QOS_SHORT_CAP}" "${QOS_SHORT_CAP}")"
        --output="${LOG_DIR}/eg-merge_%j.out"
        --error="${LOG_DIR}/eg-merge_%j.err"
        --export="$(join_exports)"
    )
    [[ -n "${dep}" ]] && args+=( --dependency="afterany:${dep}" )
    submit "eg-merge" "${args[@]}" "${WORKER_DIR}/worker_merge.sh"
    [[ -n "${dep}" ]] && assert_dependency "${LAST_JOB_ID}" "eg-merge"
    return 0
}

# -----------------------------------------------------------------------------
# Drive
# -----------------------------------------------------------------------------
case "${STAGE}" in
    gates) submit_gates ;;
    small) submit_small "${EXTERNAL_DEP}" ;;
    aids1) submit_aids1 "${EXTERNAL_DEP}" ;;
    aids2) submit_aids2 "${EXTERNAL_DEP}" ;;
    merge) submit_merge "${EXTERNAL_DEP}" ;;
    all)
        submit_gates
        submit_small "${JID_GATES}"
        submit_aids1 "${JID_SMALL}"
        submit_aids2 "${JID_AIDS1}"
        # Decision 21: stages 3 and 4 are submitted TOGETHER. They are, in this
        # one invocation; the afterok chain only orders their execution.
        submit_merge "${JID_SMALL}:${JID_AIDS1}:${JID_AIDS2}"
        ;;
esac

log ""
log "=============================================================================="
if [[ "${DRY_RUN}" == true ]]; then
    log "Dry run complete. Nothing was submitted and no directory was created."
else
    log "Submitted. Monitor with:  squeue          # bare -- Picasso rejects 'squeue -u'"
    log "                          sacct -X --format=JobID,JobName%18,State,Elapsed,ExitCode"
fi
log "=============================================================================="
