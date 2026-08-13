#!/usr/bin/env bash
# Role ubt: IPFP with 10 initial solutions, on the CONTRACTS §5 subsample only.
#
# This is the 93-core-hour arm and the reason it has its own job: T-05-design §5 --
# "a failure in the 93-core-h IPFP_MS arm would take the 3.4-core-h primary lower bound
# down with it. The primary deliverable must not depend on the arms."
#
# SLURM copies the batch script into a per-job spool directory, so BASH_SOURCE resolves to
# /var/spool/slurmd/... and a sibling path finds nothing. Source from REPO_DIR, which the
# launcher exports. Keep set -euo here too.
set -euo pipefail
source "${REPO_DIR:?REPO_DIR must be exported by the launcher}/slurm/approx_ged/_env.sh"

ROLE=ubt
assert_known_role "${ROLE}"
W="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
METHOD="${ROLE_METHOD[$ROLE]}"
OPTIONS="${ROLE_OPTIONS[$ROLE]}"
OUTDIR="${ROLE_OUTDIR[$ROLE]}"          # UB_TIGHT

# 🔴 TWO FILES, NEVER ONE PATH (CONTRACTS §5, orchestrator amendment 3).
#   subsample_pairs.npz  the sampler's pair list, written AHEAD of this run, read-only here
#   subsample.npz        this campaign's RESULT, which adds value/value_fwd/value_rev/seconds
# Pointing --out at the pair list would destroy the only reproducible record of which pairs
# were drawn, and the draw is reproducible from seed 42 only if that file survives.
PAIR_LIST="${DATA_DIR}/UB_TIGHT/subsample_pairs.npz"
DEST="${MYLOCAL}/out/${OUTDIR}"
RESULT="${DEST}/subsample.npz"
mkdir -p "${DEST}"

[[ -f "${PAIR_LIST}" ]] || {
    echo "FATAL: no subsample pair list at ${PAIR_LIST}." >&2
    echo "       It is emitted by the sampler BEFORE this run and is reproducible from" >&2
    echo "       seed 42 (CONTRACTS §5). Do not regenerate it here: a list drawn inside" >&2
    echo "       the job is a list nobody can audit." >&2
    exit 2
}
[[ "$(readlink -f "${PAIR_LIST}")" != "$(readlink -f "${RESULT}")" ]] || {
    echo "FATAL: --pair-list and --out resolve to the same file. Refusing." >&2
    exit 2
}

echo "role=${ROLE} method=${METHOD} options='${OPTIONS}'"
echo "pair-list=${PAIR_LIST}"
echo "out=${RESULT}"
"${PY}" -c "
import numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=False)
print(f'[pairs] {z[\"pair_i\"].shape[0]} pairs, keys={sorted(z.files)}')" "${PAIR_LIST}"

SHARDS="${MYLOCAL}/shards_${ROLE}"
mkdir -p "${SHARDS}"

# The subsample is pooled across datasets by construction (CONTRACTS §5), so this is one
# flat run over the pair list, not a loop over the ten datasets. UB_TIGHT/ holds ONE flat
# file, not ten: a dense per-dataset matrix would be 99.9 % missing.
run_py benchmarks.real_data.eval_setup.ged_exact_runner \
    --input "${DATA_DIR}" \
    --pair-list "${PAIR_LIST}" \
    --out "${SHARDS}/${ROLE}_c0000.npz" \
    --backend gedlib --cost-model unit \
    --compute ub --role "${ROLE}" \
    --ub-method "${METHOD}" --ub-options "${OPTIONS}" \
    --chunk-index 0 --n-chunks 1 \
    --workers "${W}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-2000}" \
    --checkpoint "${SHARDS}/${ROLE}.ckpt.npz"

# --delete-shards is honoured only after the merge's structural gate passes (CONTRACTS
# §6.2, §7). A gate failure raises MergeError, set -e aborts, and the shards survive.
run_py benchmarks.real_data.eval_setup.ged_merge_shards \
    --shards "${SHARDS}" --key "${ROLE}" \
    --pair-list "${PAIR_LIST}" \
    --out "${RESULT}" \
    --ged-from ub --role "${ROLE}" --seconds-role "${ROLE}" \
    --delete-shards

mkdir -p "${OUT_DIR}/${OUTDIR}"
cp -a "${RESULT}" "${OUT_DIR}/${OUTDIR}/subsample.npz.part.$$" && \
    mv -f "${OUT_DIR}/${OUTDIR}/subsample.npz.part.$$" "${OUT_DIR}/${OUTDIR}/subsample.npz"

REALISED=$(( $(date +%s) - START_TIME ))
{
    printf '{\n'
    printf '  "job_id": "%s",\n' "${SLURM_JOB_ID:-local}"
    printf '  "roles": "%s",\n' "${ROLES:-ubt}"
    printf '  "cores": %s,\n' "${W}"
    printf '  "node": "%s",\n' "$(hostname)"
    printf '  "cpu_model": "%s",\n' "$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
    printf '  "projected_wall_seconds": %s,\n' "${PROJ_WALL_SECONDS:-null}"
    printf '  "projected_core_seconds": %s,\n' "${PROJ_CORE_SECONDS:-null}"
    printf '  "sizing_evidence": "%s",\n' "${SIZING_EVIDENCE:-unknown}"
    printf '  "realised_wall_seconds": %s,\n' "${REALISED}"
    printf '  "realised_core_seconds": %s,\n' "$(( REALISED * W ))"
    printf '  "floor_seconds": 7200,\n'
    printf '  "cleared_floor": %s,\n' "$( (( REALISED >= 7200 )) && echo true || echo false )"
    printf '  "finished_utc": "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '}\n'
} > "${MYLOCAL}/out/run_report_${SLURM_JOB_ID:-local}.json"
cat "${MYLOCAL}/out/run_report_${SLURM_JOB_ID:-local}.json"
