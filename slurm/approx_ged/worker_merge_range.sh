#!/usr/bin/env bash
# Merge one (role, dataset)'s array shards into the CONTRACTS §4 dense matrix.
#
# Runs `afterok` on the whole array, so every chunk is present before the merge asserts
# completeness. ged_merge_shards raises MergeError on a pair no shard covers rather than
# writing `inf`, which is the check that makes a lost array task loud instead of silent.
set -euo pipefail
source "${REPO_DIR:?REPO_DIR must be exported by the launcher}/slurm/approx_ged/_env.sh"

ROLE="${ROLE:?ROLE must be exported}"
KEY="${KEY:?KEY must be exported}"
N_GRAPHS="${N_GRAPHS:?N_GRAPHS must be exported}"

assert_known_role "${ROLE}"
OUTDIR="${ROLE_OUTDIR[$ROLE]}"
GED_FROM="${ROLE_COMPUTE[$ROLE]}"
SHARDS="${OUT_DIR}/shards/${ROLE}_${KEY}"
DEST="${OUT_DIR}/${OUTDIR}"

mkdir -p "${DEST}"
echo "merging ${SHARDS} -> ${DEST}/${KEY}.npz (ged-from=${GED_FROM}, ${N_GRAPHS} graphs)"

run_py benchmarks.real_data.eval_setup.ged_merge_shards \
    --shards "${SHARDS}" \
    --key "${KEY}" \
    --n-graphs "${N_GRAPHS}" \
    --input "${DATA_DIR}/${KEY}.npz" \
    --out "${DEST}/${KEY}.npz" \
    --ged-from "${GED_FROM}" \
    --role "${ROLE}" \
    --delete-shards

echo "merged ${ROLE}/${KEY}"
