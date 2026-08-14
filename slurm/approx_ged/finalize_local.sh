#!/usr/bin/env bash
# Pull the Picasso results, cross-fill the bracket, gate them, and organise them on disk.
#
# Run from the LOCAL workstation once all six finish_remaining merges have completed.
# Idempotent: safe to re-run, and it refuses to publish anything that fails a gate.
#
#   bash slurm/approx_ged/finalize_local.sh            # pull, cross-fill, gate, publish
#   bash slurm/approx_ged/finalize_local.sh --no-pull  # re-gate what is already local
set -euo pipefail

SANDISK="/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph"
DEST="${SANDISK}/data/source/APPROX_GED"
SRC_PICASSO="picasso:/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/approx_ged"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$HOME/.conda/envs/isalgraph-cpp/bin/python}"
export PYTHONPATH="${HOME}/opt/build_gedlib/graphkit-learn"

KEYS=(iam_letter_low iam_letter_med iam_letter_high linux aids_graphedx grec aids_iam coil_del mutagenicity protein)
PULL=1
[[ "${1:-}" == "--no-pull" ]] && PULL=0

cd "${REPO}"

# ---------------------------------------------------------------- 1. pull
if (( PULL )); then
    echo "== pulling from Picasso =="
    mkdir -p "${DEST}"/{LB,UB,UB_SENSITIVITY,UB_TIGHT,ladder,gates,run_reports}
    rsync -az "${SRC_PICASSO}/LB/"              "${DEST}/LB/"
    rsync -az "${SRC_PICASSO}/UB/"              "${DEST}/UB/"
    rsync -az "${SRC_PICASSO}/UB_SENSITIVITY/"  "${DEST}/UB_SENSITIVITY/"
    rsync -az "${SRC_PICASSO}/UB_TIGHT/"        "${DEST}/UB_TIGHT/"
    rsync -az "${SRC_PICASSO}/ladder/"          "${DEST}/ladder/"
    rsync -az --include='run_report_*.json' --exclude='*' \
              "${SRC_PICASSO}/" "${DEST}/run_reports/" || true
fi

# ---------------------------------------------------------------- 2. completeness
echo "== completeness =="
missing=0
for role in LB UB UB_SENSITIVITY; do
    for k in "${KEYS[@]}"; do
        [[ -f "${DEST}/${role}/${k}.npz" ]] || { echo "  MISSING ${role}/${k}.npz"; missing=1; }
    done
done
(( missing == 0 )) || { echo "FATAL: incomplete; not cross-filling a partial cohort." >&2; exit 2; }
echo "  30/30 dense matrices present"

# ---------------------------------------------------------------- 3. cross-fill
# CONTRACTS §4.2: the same lb/ub/certified_mask written into all three role files, so the
# bracket and the certification rate travel with ANY single file. ged_matrix and
# seconds_matrix are never touched.
echo "== cross-fill =="
for k in "${KEYS[@]}"; do
    "${PY}" -m benchmarks.real_data.eval_setup.approx_ged_crossfill \
        --lb  "${DEST}/LB/${k}.npz" \
        --ub  "${DEST}/UB/${k}.npz" \
        --ubs "${DEST}/UB_SENSITIVITY/${k}.npz" 2>&1 | grep -E 'cross-fill:' || {
            echo "FATAL: cross-fill failed on ${k}" >&2; exit 3; }
done

# ---------------------------------------------------------------- 4. gates
# G2 is the strong one: the four datasets whose Suite-2 cohort is IDENTICAL to Suite 1 must
# reproduce T-27's recorded census element-wise. One comparison covers loader, cost model,
# options string, symmetrisation and pair ordering.
echo "== gates =="
T27="${SANDISK}/results/reports/T-27-ged-bound-bakeoff/data"
"${PY}" -m benchmarks.real_data.eval_setup.approx_ged_gates --gate all \
    --lb-dir "${DEST}/LB" --ub-dir "${DEST}/UB" --ubs-dir "${DEST}/UB_SENSITIVITY" \
    --t27-cells "${T27}/cells" --t27-index "${T27}/index" \
    --exact-dir "${SANDISK}/data/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed" \
    --input-dir "${DEST}/exported_suite2" \
    --datasets "$(IFS=,; echo "${KEYS[*]}")" \
    --out "${DEST}/gates" 2>&1 | grep -E 'PASS|FAIL|G2:|G3:|G4:'

if grep -q '"passed": false' "${DEST}"/gates/*.json 2>/dev/null; then
    echo "FATAL: a gate failed. Read ${DEST}/gates/*.json before using any of this." >&2
    exit 4
fi
echo "== all gates passed =="
