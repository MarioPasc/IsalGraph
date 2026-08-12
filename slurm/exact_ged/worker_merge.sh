#!/usr/bin/env bash
# Merge the AIDS census shards into one CONTRACT D matrix, run gate 4, delete the
# shards, and leave the results where the analysis expects them.
#
# Runs under --dependency=afterany, so it must tolerate a partly failed array: it
# reports what is missing rather than merging a hole into silence.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

KEY=aids
SHARDS="${OUT_DIR}/shards_${KEY}_s2"

if [[ ! -d "${SHARDS}" ]] || [[ -z "$(ls -A "${SHARDS}"/*.npz 2>/dev/null)" ]]; then
    echo "[merge] no stage-2 shards at ${SHARDS}; nothing to merge."
    echo "[merge] stage 1 stands alone as the reported analysis -- that is decision 21"
    echo "[merge] working as designed, not a failure."
    exit 0
fi

echo "[merge] $(ls -1 "${SHARDS}"/*.npz | grep -vc ckpt) shards"

# Gate 4 lives inside the merge and exits non-zero on any violation: symmetry, zero
# diagonal, 0 < v < inf or explicitly censored, lb <= ged <= ub, and no conflicting
# value on any pair index present in more than one shard -- which is how stage 1 and
# stage 2 are checked against each other.
run_py benchmarks.real_data.eval_setup.ged_merge_shards \
    --shards "${SHARDS}" --key "${KEY}" --n-graphs 769 \
    --input "${DATA_DIR}/${KEY}.npz" \
    --out "${MYLOCAL}/out/${KEY}.npz" \
    --delete-shards
RC=$?
if [[ ${RC} -ne 0 ]]; then
    echo "[merge] FAILED rc=${RC}: shards KEPT for diagnosis at ${SHARDS}" >&2
    exit ${RC}
fi

# Leave the checkpoint files behind only if they are still needed; they are not once the
# merge has passed every assertion.
rm -f "${SHARDS}"/*.ckpt.npz 2>/dev/null || true
rmdir "${SHARDS}" 2>/dev/null || true

echo "[merge] file-count check -- the whole GED programme should be ~30 files"
find "${OUT_DIR}" -type f | wc -l
echo "[merge] ok"
