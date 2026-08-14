#!/usr/bin/env bash
# One array task = one (role, dataset, contiguous pair range), at --workers 1.
#
# 🔴 WHY THIS EXISTS, AND WHY IT IS NOT worker_bounds.sh.
# worker_bounds.sh runs one job per role with an N-worker process pool. Measured on an
# identical 211,871-pair Letter HIGH slice, varying ONLY --workers:
#     1 worker   36.4 s wall,     36 core-s
#     4 workers  53.0 s wall,    212 core-s
#    15 workers  61.9 s wall,    928 core-s
#    32 workers 164.4 s wall,  5,260 core-s
# The pool is NEGATIVE-SCALING: more workers makes even the wall clock worse. The lb, ub
# and ubs campaigns ran at 15, 37 and 126 workers and all three hit their 12 h wallclock
# after 8 of 10 datasets. Cores do not fix that; they cause it. T-05-design amendment 11.
#
# So parallelism moves OUT of the pool and into the array, which is what T-03 did and what
# §5 talked itself out of. Each task is single-process, so per-pair cost is the honest
# solver cost, and the SCBI two-hour floor is met by sizing the RANGE rather than by
# stacking cores.
#
# Shards land on SHARED storage, not $LOCALSCRATCH: array tasks land on different nodes and
# a later merge job has to see all of them. That is a deliberate, bounded file-count cost
# (a few tens of files, deleted at merge), not an oversight.
set -euo pipefail
source "${REPO_DIR:?REPO_DIR must be exported by the launcher}/slurm/approx_ged/_env.sh"

ROLE="${ROLE:?ROLE must be exported}"
KEY="${KEY:?KEY must be exported}"
N_CHUNKS="${N_CHUNKS:?N_CHUNKS must be exported}"
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"

assert_known_role "${ROLE}"

METHOD="${ROLE_METHOD[$ROLE]}"
OPTIONS="${ROLE_OPTIONS[$ROLE]}"
COMPUTE="${ROLE_COMPUTE[$ROLE]}"
INPUT="${DATA_DIR}/${KEY}.npz"
SHARDS="${OUT_DIR}/shards/${ROLE}_${KEY}"

[[ -f "${INPUT}" ]] || { echo "FATAL: no exported cohort at ${INPUT}" >&2; exit 2; }
mkdir -p "${SHARDS}"

echo "role=${ROLE} key=${KEY} chunk=${CHUNK}/${N_CHUNKS} method=${METHOD} options='${OPTIONS}'"
echo "workers=1 (deliberate -- see the header)"

# --workers 1 is not a conservative default here, it is the measured optimum. Do not raise
# it: every value above 1 was slower in BOTH wall clock and core-seconds.
run_py benchmarks.real_data.eval_setup.ged_exact_runner \
    --input "${INPUT}" \
    --out "${SHARDS}/${KEY}_c$(printf '%04d' "${CHUNK}").npz" \
    --backend gedlib --cost-model unit \
    --compute "${COMPUTE}" --role "${ROLE}" \
    $( [[ "${COMPUTE}" == "lb" ]] \
        && echo --lb-method "${METHOD}" --lb-options "${OPTIONS}" \
        || echo --ub-method "${METHOD}" --ub-options "${OPTIONS}" ) \
    --env-mode "${ENV_MODE}" \
    --chunk-index "${CHUNK}" --n-chunks "${N_CHUNKS}" \
    --workers 1 \
    --checkpoint-every "${CHECKPOINT_EVERY:-20000}" \
    --checkpoint "${SHARDS}/${KEY}_c$(printf '%04d' "${CHUNK}").ckpt.npz"

echo "chunk ${CHUNK} of ${ROLE}/${KEY} complete"
