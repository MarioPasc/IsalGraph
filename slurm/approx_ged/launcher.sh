#!/usr/bin/env bash
# Submit the T-05 approximate-GED bound programme to Picasso.
#
#   bash launcher.sh --dry-run --stage all
#   bash launcher.sh --stage lb
#   bash launcher.sh --stage all --probe-json /path/probe.json --bins /path/bins.json
#   bash launcher.sh --dry-run --stage all --group lb,ub          # two roles, one job
#
# Design: .claude/notes/review/tasks/T-05-design.md §5
# Contract: .claude/notes/2026-08-13-t05-bounds/CONTRACTS.md §3, §4, §6, §7, §8
#
# The single human entry point. Every sbatch flag lives here; the workers carry no
# #SBATCH header, so one launcher can dispatch four differently-sized jobs without four
# headers drifting apart. This mirrors slurm/exact_ged/launcher.sh, from which the
# helpers below are taken unchanged unless a comment says otherwise.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- configuration
# CONTRACTS.md §8, verbatim. Do not edit these here -- edit the contract, then here.
export CONDA_ENV_PREFIX="/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalgraph"
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalGraph"
export GEDLIB_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/build_gedlib/graphkit-learn"
export DATA_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph/suite2"
export OUT_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/approx_ged"
export LOGS_DIR="${OUT_DIR}/logs"
ACCOUNT="tic_163_uma"

# Pin the node family. Per-pair wall time is a REPORTED quantity (D12, and T-27 §5
# extended to Suite 2), so a mixed Intel/AMD pool would make the timing a measurement of
# the scheduler rather than of the solver. sr = 128 cores / 450 GB, AMD EPYC 7H12,
# homogeneous.
CONSTRAINT="sr"
CORES_PER_NODE=128

# SCBI's two-hour floor (Manuel, soporte@scbi.uma.es, 2026-08-07). THIS IS THE ONLY
# BINDING CONSTRAINT ON THIS WORKLOAD: the whole programme is ~133 core-hours, which on
# one node is under an hour. The design problem is not how to split the work but how to
# keep from splitting it (T-05-design §5).
FLOOR_SECONDS=7200
TARGET_SECONDS=10800

# Suite-2 pair total, CONTRACTS.md §1. Used for the three full-Suite roles.
SUITE2_PAIRS=21710892

# Projected core-hours per role, CONTRACTS §3 verbatim. The per-pair rate is DERIVED from
# these at use time rather than written out as a rounded decimal: writing the rate down
# costs a core to float error (11.957142857 x 28,000 = 334,799.99996, which floors to 30
# cores instead of 31), and the contract states core-hours, not rates, so deriving keeps
# the provenance visible.
#
# THESE ARE PROJECTIONS, NOT MEASUREMENTS, and T-27 limitation 3 says they are LOWER
# bounds on true cost: per-pair cost scales roughly as max(n1,n2)^3 and T-27's rate was
# probed at n-bar = 29.5 while Suite 2 reaches n = 98. Override with --probe-json.
DEFAULT_COREH_lb="3.4"
DEFAULT_COREH_ub="8.4"
DEFAULT_COREH_ubs="28"
DEFAULT_COREH_ubt="93"

STAGE="all"
DRY_RUN=false
PROBE_JSON=""
BINS_JSON=""
UBT_PAIRS=28000                   # CONTRACTS §5 ceiling: 14 bins x min(2000, population)
PROBE_PAIRS=3000                  # T-05-design §5: the in-job probe sample
PROBE_PAIRS_UBT=200               # ubt costs ~12 s/pair against ~1 ms for the others
CHECKPOINT_EVERY=2000
RATE_lb=""; RATE_ub=""; RATE_ubs=""; RATE_ubt=""
# 🔴 NOT named GROUPS. `GROUPS` is a bash BUILTIN array holding the user's group ids;
# assignment to it fails (rc=1) and under `set -e` that kills the launcher during flag
# parsing, or -- worse, depending on context -- leaves the array empty so --group is
# silently ignored and four jobs are submitted where two were asked for. Measured on
# bash 5.2.15. The same trap applies to SECONDS, PIPESTATUS, RANDOM, LINENO, FUNCNAME.
ROLE_GROUPS=()                    # each element is a comma-separated role list

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=true; shift ;;
        --stage)           STAGE="$2"; shift 2 ;;
        --probe-json)      PROBE_JSON="$2"; shift 2 ;;
        --bins)            BINS_JSON="$2"; shift 2 ;;
        --group)           ROLE_GROUPS+=("$2"); shift 2 ;;
        --rate-lb)         RATE_lb="$2"; shift 2 ;;
        --rate-ub)         RATE_ub="$2"; shift 2 ;;
        --rate-ubs)        RATE_ubs="$2"; shift 2 ;;
        --rate-ubt)        RATE_ubt="$2"; shift 2 ;;
        --ubt-pairs)       UBT_PAIRS="$2"; shift 2 ;;
        --probe-pairs)     PROBE_PAIRS="$2"; shift 2 ;;
        --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
        --target-seconds)  TARGET_SECONDS="$2"; shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
case "${STAGE}" in probe|lb|ub|ubs|ubt|merge|all) ;;
    *) echo "bad --stage ${STAGE} (probe|lb|ub|ubs|ubt|merge|all)" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- helpers
# Picasso's Lua sbatch wrapper writes ANSI codes and a multi-line warning to stdout, so
# --parsable does NOT return just the id. A line-wise sed leaves a multi-line "id" and
# the guard then fires AFTER submission, leaving an untracked job on the cluster. Take
# the LAST line first, then strip.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

submit() {                       # echoes a verified-numeric job id, or fails
    if ${DRY_RUN}; then
        printf '[DRY-RUN] sbatch --parsable' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        echo "000000"
        return 0
    fi
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}

assert_dependency_took() {       # a bad id is ACCEPTED and recorded Dependency=(null)
    local id="$1"
    ${DRY_RUN} && return 0
    if scontrol show job "${id}" | grep -q 'Dependency=(null)'; then
        echo "FATAL: dependency dropped on ${id}; cancelling" >&2
        scancel "${id}"; return 1
    fi
}

# ---------------------------------------------------------------- sizing
# core-seconds for one role, by the best available evidence, in this order:
#   1. per-bin measured rates x the per-bin pair table   (orchestrator amendment 1)
#   2. a flat measured rate from probe.json
#   3. the CONTRACTS §3 projection
# Falling back LOGS LOUDLY, because an n-bar-based projection is wrong by a large factor
# on a cohort running from n-bar = 4.07 to n-bar = 31.68 with a tail to n = 98, and
# Jensen's inequality makes it an UNDER-estimate on the right-skewed sets.
#
# Echoes "<core_seconds> <evidence-tag>".
core_seconds_for() {
    local role="$1" npairs="$2" fallback_rate="$3"
    python3 - "${role}" "${npairs}" "${fallback_rate}" "${PROBE_JSON}" "${BINS_JSON}" <<'PYEOF'
import json, sys, os

role, npairs, fallback = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
probe_path, bins_path = sys.argv[4], sys.argv[5]

def load(p):
    if not p or not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None

probe, bins = load(probe_path), load(bins_path)

# 1. per-bin: sum_b count_b * rate_b. Both sides must name the same bin edges, or the
#    products are meaningless -- refuse rather than silently mixing binnings.
if probe and bins:
    per_bin = (probe.get("per_bin_seconds_per_pair") or {}).get(role) \
        or probe.get("per_bin_seconds_per_pair_%s" % role)
    counts = bins.get("totals")
    if counts is None and isinstance(bins.get("datasets"), dict):
        cols = list(bins["datasets"].values())
        if cols:
            counts = [sum(c[i] for c in cols) for i in range(len(cols[0]))]
    edges_ok = (probe.get("bin_edges") is None or bins.get("bin_edges") is None
                or probe["bin_edges"] == bins["bin_edges"])
    if per_bin and counts and edges_ok and len(per_bin) >= len(counts):
        rates = [float(per_bin[str(i)] if isinstance(per_bin, dict) else per_bin[i])
                 for i in range(len(counts))]
        print("%.6f binned" % sum(c * r for c, r in zip(counts, rates)))
        raise SystemExit(0)

# 2. flat measured
if probe:
    flat = (probe.get("seconds_per_pair") or {})
    rate = flat.get(role) if isinstance(flat, dict) else (flat if role == probe.get("role") else None)
    if rate:
        print("%.6f flat-measured" % (npairs * float(rate)))
        raise SystemExit(0)

# 3. projection
print("%.6f projected" % (npairs * fallback))
PYEOF
}

# cores such that ONE task clears the floor.
#
# 🔴 FLOOR, not ceil -- this is the one place this launcher deliberately differs from
# slurm/exact_ged/launcher.sh:95. ceil(cs/target) makes the projected wall <= TARGET and
# therefore able to land UNDER the floor; floor(cs/target) makes it >= TARGET. T-03 could
# use ceil because 2,081 core-hours never approached the floor; here the floor is the only
# binding constraint, so the rounding direction is load-bearing.
cores_for_single_task() {        # $1 = total core-seconds
    python3 -c "
import math,sys
cs=float(sys.argv[1]); tgt=float(${TARGET_SECONDS}); cap=${CORES_PER_NODE}
print(max(1, min(cap, int(math.floor(cs/tgt)))))" "$1"
}

wall_seconds() {                 # $1 = core-seconds, $2 = cores
    python3 -c "import sys; print(int(float(sys.argv[1])/int(sys.argv[2])))" "$1" "$2"
}

fmt_hours() { python3 -c "import sys; print(f'{float(sys.argv[1])/3600:.2f}')" "$1"; }

# 🔴 The refusal. A job projected under the floor is NOT submitted short. Reduce cores
# (already done by the floor division), or merge the role into an adjacent job with
# --group. Exit 3, matching slurm/exact_ged/launcher.sh:183.
assert_clears_floor() {          # $1 = label, $2 = wall seconds, $3 = roles
    local label="$1" wall="$2" roles="$3"
    if (( wall < FLOOR_SECONDS )); then
        echo "FATAL: ${label} projects ${wall}s of wall, under the ${FLOOR_SECONDS}s floor." >&2
        echo "       SCBI wrote to THIS account about minute-long jobs (2026-08-07); a" >&2
        echo "       short job costs the scheduler more to place than to run." >&2
        echo "       Remedies, in order:" >&2
        echo "         1. group this role with an adjacent one:  --group ${roles},<other>" >&2
        echo "         2. re-measure: --probe-json <probe.json> (the projection is a LOWER" >&2
        echo "            bound on true cost, so a measurement usually moves this up)" >&2
        echo "       Submitting short is not one of them." >&2
        exit 3
    fi
}

# ---------------------------------------------------------------- role -> job resolution
# A role runs alone unless --group names it. Groups run their roles sequentially in one
# job and their core-seconds add, which is how "merge the role into an adjacent job"
# (T-05-design §5) becomes executable rather than advice.
role_group() {                   # $1 = role -> the comma-separated group it belongs to
    local role="$1" g
    for g in "${ROLE_GROUPS[@]+"${ROLE_GROUPS[@]}"}"; do
        if [[ ",${g}," == *",${role},"* ]]; then echo "${g}"; return 0; fi
    done
    echo "${role}"
}

pairs_for_role() {               # $1 = role
    case "$1" in
        ubt) echo "${UBT_PAIRS}" ;;
        *)   echo "${SUITE2_PAIRS}" ;;
    esac
}

rate_for_role() {                # $1 = role -- CLI override, else core-hours / pairs
    local role="$1" override coreh npairs
    override="$(eval "printf '%s' \"\${RATE_${role}:-}\"")"
    if [[ -n "${override}" ]]; then echo "${override}"; return 0; fi
    coreh="$(eval "printf '%s' \"\${DEFAULT_COREH_${role}}\"")"
    npairs="$(pairs_for_role "${role}")"
    python3 -c "import sys; print(repr(float(sys.argv[1])*3600.0/float(sys.argv[2])))" \
        "${coreh}" "${npairs}"
}

# ---------------------------------------------------------------- preflight
# --export splits on EVERY comma, so no value below may contain one. All values here are
# single paths or integers; the roles list is COLON-separated for exactly this reason,
# and the GEDLIB options strings never cross --export at all -- the workers read them
# from the frozen table in _env.sh.
COMMON_EXPORT="ALL,CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX},REPO_DIR=${REPO_DIR},GEDLIB_DIR=${GEDLIB_DIR},DATA_DIR=${DATA_DIR},OUT_DIR=${OUT_DIR},CHECKPOINT_EVERY=${CHECKPOINT_EVERY},PROBE_PAIRS=${PROBE_PAIRS}"

mkdir -p "${LOGS_DIR}" 2>/dev/null || true
echo "stage=${STAGE}  constraint=${CONSTRAINT}  target=$(fmt_hours "${TARGET_SECONDS}")h  floor=$(fmt_hours "${FLOOR_SECONDS}")h"
echo "logs -> ${LOGS_DIR}"
if [[ -z "${BINS_JSON}" || -z "${PROBE_JSON}" ]]; then
    echo "WARNING: sizing WITHOUT the per-bin table and/or measured probe."
    echo "         bins=${BINS_JSON:-<none>}  probe=${PROBE_JSON:-<none>}"
    echo "         Falling back to a flat n-bar projection. T-27 limitation 3: that is a"
    echo "         LOWER bound on true cost -- per-pair cost scales ~max(n1,n2)^3 and this"
    echo "         cohort runs to n = 98. Treat every core count below as a floor."
fi
echo

# Reading the real subsample size beats trusting --ubt-pairs. The sampler writes the pair
# list ahead of the run (CONTRACTS §5, orchestrator amendment 3); if it is on disk, use
# its length. numpy lives in the conda env, not necessarily in the login node's python3.
SUBSAMPLE_PAIRS_FILE="${DATA_DIR}/UB_TIGHT/subsample_pairs.npz"
if [[ -f "${SUBSAMPLE_PAIRS_FILE}" ]] && [[ -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
    _n=$("${CONDA_ENV_PREFIX}/bin/python" -c "
import numpy as np,sys
try:
    print(int(np.load(sys.argv[1], allow_pickle=False)['pair_i'].shape[0]))
except Exception:
    print('')" "${SUBSAMPLE_PAIRS_FILE}" 2>/dev/null || echo "")
    if [[ "${_n}" =~ ^[0-9]+$ ]]; then
        echo "subsample:  ${_n} pairs read from ${SUBSAMPLE_PAIRS_FILE} (overrides --ubt-pairs=${UBT_PAIRS})"
        UBT_PAIRS="${_n}"
    else
        echo "subsample:  could not read ${SUBSAMPLE_PAIRS_FILE}; using --ubt-pairs=${UBT_PAIRS}"
    fi
fi

declare -A JOB_ID_OF_ROLE=()
SUBMITTED_IDS=()

# ---------------------------------------------------------------- the production stages
# One single-node job per role (or per --group). NOT a job array over datasets: Letter LOW
# is ~90 core-seconds, so nine of ten array tasks would be minutes long -- the 12,600-task
# pattern SCBI wrote to this account about. NOT an array over pair chunks: correct for
# T-03's 2,081 core-h, absurd for 133. ged_pair_index.py's chunking is retained for
# RESUMABILITY INSIDE ONE TASK, not for fan-out (T-05-design §5).
submit_bounds_job() {            # $1 = comma-separated roles, e.g. "lb" or "lb,ub"
    local roles_csv="$1"
    local roles_colon="${roles_csv//,/:}"
    local total_cs=0 evidence="" role cs tag npairs rate
    for role in ${roles_csv//,/ }; do
        npairs=$(pairs_for_role "${role}")
        rate=$(rate_for_role "${role}")
        read -r cs tag < <(core_seconds_for "${role}" "${npairs}" "${rate}")
        total_cs=$(python3 -c "print(float('${total_cs}')+float('${cs}'))")
        evidence="${evidence}${evidence:+,}${role}:${tag}"
        printf '  %-4s %12s pairs  %12.0f core-s  (%s)\n' "${role}" "${npairs}" "${cs}" "${tag}"
    done

    local cores wall wall_h
    cores=$(cores_for_single_task "${total_cs}")
    wall=$(wall_seconds "${total_cs}" "${cores}")
    wall_h=$(fmt_hours "${wall}")

    # ubt is the only role on the subsample and the only one whose per-pair cost is
    # seconds rather than milliseconds; give it a day. 12 h for the full-Suite roles is
    # 3-4x the projected wall, which is the headroom T-27 limitation 3 demands.
    local time_flag="0-12:00:00" mem="64G"
    [[ ",${roles_csv}," == *",ubt,"* ]] && time_flag="1-00:00:00"

    local name="aged-${roles_csv//,/-}"
    echo "  -> ${name}: ${cores} cores, projected wall ${wall_h} h, --time=${time_flag}, evidence=${evidence}"
    assert_clears_floor "${name}" "${wall}" "${roles_csv}"

    local worker="worker_bounds.sh"
    [[ "${roles_csv}" == "ubt" ]] && worker="worker_subsample.sh"

    local id
    id=$(submit \
        --job-name="${name}" --account="${ACCOUNT}" \
        --time="${time_flag}" --ntasks=1 --cpus-per-task="${cores}" --mem="${mem}" \
        --constraint="${CONSTRAINT}" \
        --output="${LOGS_DIR}/${name}_%j.out" --error="${LOGS_DIR}/${name}_%j.err" \
        --export="${COMMON_EXPORT},ROLES=${roles_colon},N_WORKERS=${cores},PROJ_WALL_SECONDS=${wall},PROJ_CORE_SECONDS=${total_cs%.*},SIZING_EVIDENCE=${evidence//,/;}" \
        "${SCRIPT_DIR}/${worker}")
    echo "  ${name} -> ${id}"
    for role in ${roles_csv//,/ }; do JOB_ID_OF_ROLE[$role]="${id}"; done
    SUBMITTED_IDS+=("${id}")
}

# --stage probe is a SIZING REPORT AND A REFUSAL, and it submits nothing by design.
#
# T-05-design §5 settled this: "A probe stage runs first INSIDE the same job ... A separate
# probe job would itself violate the floor." Rather than assert that, this stage computes
# it. At the projected rates the standalone probe is ~0.7 core-hours -- 40 minutes on one
# core, well under the 7,200 s floor -- so the same assert_clears_floor that guards every
# production job refuses it, with exit 3, from the same code path. Nothing here is special
# -cased to produce the refusal; the arithmetic produces it.
if [[ "${STAGE}" == "probe" ]]; then
    echo "probe (standalone) -- sizing report, submits nothing:"
    TOTAL_CS=0
    for role in lb ub ubs ubt; do
        RATE=$(rate_for_role "${role}")
        # ubt's per-pair cost is seconds where the others are milliseconds, and its whole
        # scope is 28,000 pairs. Probing it on 3,000 would burn a third of the campaign to
        # measure it. PROBE_PAIRS_UBT is a separate, much smaller sample.
        NP="${PROBE_PAIRS}"
        [[ "${role}" == "ubt" ]] && NP="${PROBE_PAIRS_UBT}"
        CS=$(python3 -c "print(float('${RATE}')*${NP})")
        TOTAL_CS=$(python3 -c "print(float('${TOTAL_CS}')+float('${CS}'))")
        printf '  %-4s %6s probe pairs  %10.1f core-s\n' "${role}" "${NP}" "${CS}"
    done
    CORES=$(cores_for_single_task "${TOTAL_CS}")
    WALL=$(wall_seconds "${TOTAL_CS}" "${CORES}")
    echo "  -> aged-probe: ${CORES} cores, projected wall $(fmt_hours "${WALL}") h"
    assert_clears_floor "aged-probe" "${WALL}" "probe"
    echo
    echo "The standalone probe clears the floor at these rates -- but it is STILL not"
    echo "submitted. T-05-design §5: the probe belongs inside the production job, on the"
    echo "hardware that does the work. Submit --stage lb (or all); each worker probes"
    echo "first and writes probe.json, which you then feed back via --probe-json."
    exit 0
fi

for ROLE in lb ub ubs ubt; do
    [[ "${STAGE}" == "${ROLE}" || "${STAGE}" == "all" ]] || continue
    GRP=$(role_group "${ROLE}")
    # A grouped role is submitted once, by its first member.
    FIRST="${GRP%%,*}"
    [[ "${FIRST}" == "${ROLE}" ]] || continue
    echo "=== ${GRP} ==="
    submit_bounds_job "${GRP}"
    echo
done

# ---------------------------------------------------------------- cross-fill + gates
# CONTRACTS §4.2: after the three role campaigns merge, ONE step opens LB/, UB/ and
# UB_SENSITIVITY/, writes the same lb_matrix / ub_matrix / certified_mask into all three,
# and rewrites them atomically. ged_matrix and seconds_matrix are never touched.
#
# afterok, not afterany: cross-filling from a partial campaign would write a certified_mask
# derived from an incomplete lower bound, and certified_mask is a PROOF (CONTRACTS §4.1).
# A hole in it is not a missing value, it is a false one.
if [[ "${STAGE}" == "merge" || "${STAGE}" == "all" ]]; then
    echo "=== cross-fill + gates ==="
    DEPS=()
    for role in lb ub ubs; do
        id="${JOB_ID_OF_ROLE[$role]:-}"
        # An explicit `if`, not an `&&` chain. Under `set -e` a top-level `a && b && c`
        # whose guard fails returns non-zero, and whether bash then exits depends on the
        # context the list sits in -- which is exactly the kind of thing that works in a
        # dry run and kills a real submission. Grouped roles share one id, so the
        # duplicate test is reached on every real --group invocation.
        if [[ -n "${id}" ]]; then
            if [[ " ${DEPS[*]:-} " != *" ${id} "* ]]; then
                DEPS+=("${id}")
            fi
        fi
    done
    DEP=()
    if [[ ${#DEPS[@]} -gt 0 ]]; then
        DEP=(--dependency="afterok:$(IFS=:; echo "${DEPS[*]}")")
        echo "  waiting on: ${DEPS[*]}"
    else
        echo "  no upstream job in this invocation; running against whatever is in ${OUT_DIR}"
    fi
    XF_ID=$(submit \
        --job-name=aged-crossfill --account="${ACCOUNT}" \
        --time=0-04:00:00 --ntasks=1 --cpus-per-task=4 --mem=64G \
        --constraint="${CONSTRAINT}" "${DEP[@]}" \
        --output="${LOGS_DIR}/crossfill_%j.out" --error="${LOGS_DIR}/crossfill_%j.err" \
        --export="${COMMON_EXPORT}" \
        "${SCRIPT_DIR}/worker_crossfill.sh")
    echo "  aged-crossfill -> ${XF_ID}"
    [[ ${#DEPS[@]} -gt 0 ]] && assert_dependency_took "${XF_ID}"
    SUBMITTED_IDS+=("${XF_ID}")
fi

echo
echo "submitted: ${SUBMITTED_IDS[*]:-none}"
echo
echo "monitor:  ssh picasso 'squeue'          # -u is REJECTED by Picasso's wrapper"
echo "states:   ssh picasso 'sacct -j <id> -X -n -P -o JobID,State,Elapsed,NodeList'"
echo "memory:   ssh picasso 'sacct -j <id> -n -P -o JobID,MaxRSS | grep .batch'   # NO -X"
echo "results:  ${OUT_DIR}"
echo
echo "AFTER crossfill, run the independent gates and READ THEM (they are not automatic):"
echo "  python -m benchmarks.real_data.eval_setup.approx_ged_gates --gate all \\"
echo "     --lb-dir ${OUT_DIR}/LB --ub-dir ${OUT_DIR}/UB --ubs-dir ${OUT_DIR}/UB_SENSITIVITY \\"
echo "     --t27-cells <T-27 cells> --exact-dir <extended_merged_exact_ged/computed> \\"
echo "     --out ${OUT_DIR}/gates"
