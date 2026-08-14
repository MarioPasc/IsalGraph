#!/usr/bin/env bash
# T-05 §6 -- the calibration ladder. Exact GED on a size-stratified sample at each rung
# from n = 13 upward, so the proven bracket is calibrated closer to the regime it is
# licensed to (AE.1). One job, not an array: a rung is a few hundred core-hours, and the
# 12,600-task pattern SCBI wrote to this account about on 2026-08-07 is exactly what
# splitting six rungs into six submissions would recreate.
#
# NO RESOURCE HEADER. The launcher supplies every resource flag on the sbatch command
# line, the same convention slurm/exact_ged/ established and the one that lets a single
# launcher dispatch heterogeneously-sized jobs without headers drifting apart.
# `test_workers_carry_no_sbatch_header` greps every worker for the directive token as a
# literal, so do not write it here even inside a comment -- the test cannot distinguish a
# comment from a directive, and it is right not to try.
#
# THIS SCRIPT SUBMITS NOTHING. It is the payload; the orchestrator owns submission.
#
# SLURM copies the batch script into a per-job spool directory, so BASH_SOURCE resolves
# to /var/spool/slurmd/... and a sibling path finds nothing. Source from REPO_DIR, which
# the launcher exports. Keep set -euo here too: _env.sh sets it, so failing to source it
# would otherwise silently disable it as well.
set -euo pipefail
source "${REPO_DIR:?REPO_DIR must be exported by the launcher}/slurm/approx_ged/_env.sh"

W="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
RUNGS="${LADDER_RUNGS:-13,14,15,16,17,18}"
PAIRS_PER_RUNG="${LADDER_PAIRS_PER_RUNG:-250}"
MIN_PER_DATASET="${LADDER_MIN_PER_DATASET:-20}"
BUDGET_SECONDS="${LADDER_BUDGET_SECONDS:-1200}"
SEED="${LADDER_SEED:-42}"
TRUNCATE_BELOW="${LADDER_TRUNCATE_BELOW:-0.25}"
LADDER_LOCAL="${MYLOCAL}/out/ladder"
LADDER_SHARED="${OUT_DIR}/ladder"
REPORT="${MYLOCAL}/out/ladder_report_${SLURM_JOB_ID:-local}.json"

mkdir -p "${LADDER_LOCAL}" "${LADDER_SHARED}"

echo "=========================================="
echo "T-05 calibration ladder"
echo "  rungs             ${RUNGS}"
echo "  pairs per rung    ${PAIRS_PER_RUNG} (floor ${MIN_PER_DATASET} per contributor)"
echo "  seed              ${SEED}"
echo "  per-pair budget   ${BUDGET_SECONDS} s"
echo "  truncate below    ${TRUNCATE_BELOW}"
echo "  pool workers      ${W}"
echo "  data              ${DATA_DIR}"
echo "  out (node-local)  ${LADDER_LOCAL}"
echo "  out (shared)      ${LADDER_SHARED}"
echo "=========================================="

# ---------------------------------------------------------------- resume
# Rungs already landed in the shared directory are staged back to node-local scratch so
# --resume can skip them. A requeue after a wallclock overrun therefore costs the rung in
# flight and nothing else. Copy rather than symlink: the EXIT trap mirrors the whole
# node-local tree back, and a dangling symlink would mirror as a dangling symlink.
shopt -s nullglob
for f in "${LADDER_SHARED}"/rung_*.npz; do
    cp -a "${f}" "${LADDER_LOCAL}/"
    echo "[resume] staged $(basename "${f}") -- it will be skipped"
done
shopt -u nullglob

# ---------------------------------------------------------------- sample first
# The sample is drawn from seed 42 alone and costs milliseconds, so it is written before
# any solver runs. Two consequences worth the extra second: the per-rung stratification is
# on disk even if the job is killed on its first pair, and the pairs a later rerun solves
# are demonstrably the same pairs rather than merely the same recipe.
echo "### drawing the samples (seed ${SEED})"
run_py benchmarks.real_data.eval_setup.ged_ladder \
    --exported-dir "${DATA_DIR}" \
    --out-dir "${LADDER_LOCAL}" \
    --rungs "${RUNGS}" \
    --pairs-per-rung "${PAIRS_PER_RUNG}" \
    --min-per-dataset "${MIN_PER_DATASET}" \
    --seed "${SEED}" \
    --sample-only
cp -a "${LADDER_LOCAL}"/sample_rung_*.npz "${LADDER_SHARED}/" 2>/dev/null || true

# ---------------------------------------------------------------- production
# One invocation for the whole ladder. Truncation at the first rung below
# TRUNCATE_BELOW is a property of the ladder, not of a rung, so the decision lives in one
# tested place rather than in a bash loop that would have to re-read each .npz to make it.
#
# --mirror-dir copies each rung to the shared filesystem the moment it lands. That is the
# checkpoint: the EXIT trap's whole-tree mirror guarantees completeness, and this bounds
# the loss to the rung in flight if the node dies mid-ladder.
#
# --bounds gedlib with the CONTRACTS §3 role strings verbatim. THE OPTIONS STRING IS PART
# OF THE METHOD NAME: GEDLIB's upper bounds change on 74-94 % of pairs between runs at
# library defaults (T-05-design amendment 6), so a run whose metadata omits them records
# a number nobody can reproduce.
#
# ANCHOR_AWARE_GED appears nowhere. It is retired (T-03-design amendment 2): measured
# non-deterministic on 14 of 15 real AIDS pairs, wrong on 4 of 18 against brute force, and
# it reports LB == UB -- a false optimality certificate. The exact solver is networkx A*,
# and completion is established by the search terminating, never by a value coming back.
echo "### solving the ladder"
run_py benchmarks.real_data.eval_setup.ged_ladder \
    --exported-dir "${DATA_DIR}" \
    --out-dir "${LADDER_LOCAL}" \
    --mirror-dir "${LADDER_SHARED}" \
    --rungs "${RUNGS}" \
    --pairs-per-rung "${PAIRS_PER_RUNG}" \
    --min-per-dataset "${MIN_PER_DATASET}" \
    --seed "${SEED}" \
    --budget-seconds "${BUDGET_SECONDS}" \
    --truncate-below "${TRUNCATE_BELOW}" \
    --cost-model unit \
    --bounds gedlib \
    --lb-method "${ROLE_METHOD[lb]}" --lb-options "${ROLE_OPTIONS[lb]}" \
    --ub-method "${ROLE_METHOD[ub]}" --ub-options "${ROLE_OPTIONS[ub]}" \
    --workers "${W}" \
    --resume

# ---------------------------------------------------------------- projected vs realised
# T-27 limitation 3 says its Suite-2 projections are LOWER bounds on true cost, and the
# ladder is the part of T-05 whose cost is least well characterised: exact GED grows
# steeply in n and the pilot that sized this job was 25 pairs at one rung. Recording the
# realised numbers costs nothing and is what makes the next sizing a measurement.
REALISED=$(( $(date +%s) - START_TIME ))
{
    printf '{\n'
    printf '  "job_id": "%s",\n' "${SLURM_JOB_ID:-local}"
    printf '  "stage": "ladder",\n'
    printf '  "rungs": "%s",\n' "${RUNGS}"
    printf '  "pairs_per_rung": %s,\n' "${PAIRS_PER_RUNG}"
    printf '  "min_per_dataset": %s,\n' "${MIN_PER_DATASET}"
    printf '  "budget_seconds": %s,\n' "${BUDGET_SECONDS}"
    printf '  "seed": %s,\n' "${SEED}"
    printf '  "truncate_below": %s,\n' "${TRUNCATE_BELOW}"
    printf '  "cores": %s,\n' "${W}"
    printf '  "node": "%s",\n' "$(hostname)"
    printf '  "cpu_model": "%s",\n' "$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
    printf '  "projected_wall_seconds": %s,\n' "${PROJ_WALL_SECONDS:-null}"
    printf '  "sizing_evidence": "%s",\n' "${SIZING_EVIDENCE:-rung13-pilot-25-pairs}"
    printf '  "realised_wall_seconds": %s,\n' "${REALISED}"
    printf '  "realised_core_seconds": %s,\n' "$(( REALISED * W ))"
    printf '  "rungs_landed": %s,\n' "$(find "${LADDER_LOCAL}" -maxdepth 1 -name 'rung_*.npz' | wc -l)"
    printf '  "finished_utc": "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '}\n'
} > "${REPORT}"
echo "[report] ${REPORT}"
cat "${REPORT}"

# The manifest carries the measured exact-GED ceiling. Echo it so it is in the job log as
# well as in the file -- it is the number AE.1 is answered with.
if [[ -f "${LADDER_LOCAL}/manifest.json" ]]; then
    echo "### ladder manifest"
    "${PY}" -c "
import json, sys
m = json.load(open('${LADDER_LOCAL}/manifest.json'))
print(f\"measured exact-GED ceiling: n = {m['exact_ged_ceiling']}\")
print(f\"truncated at rung: {m['truncated_at_rung']}\")
for r in m['rungs']:
    print(f\"  rung {r['rung']:>3}: {r['n_certified']:>4}/{r['n_pairs']:<4} certified \"
          f\"({100*r['certification_rate']:5.1f} %), censoring {100*r['censoring_rate']:5.1f} %\")
"
fi
