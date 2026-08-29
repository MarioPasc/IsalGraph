#!/usr/bin/env bash
# Pull the T-28 campaign results off Picasso and report where they stand.
#
#   bash slurm/t28_metrics/fetch_results.sh            # status + fetch whatever exists
#   bash slurm/t28_metrics/fetch_results.sh --status   # status only, fetch nothing
#
# Safe to run repeatedly and safe to run while the campaign is still going: it
# fetches the partials that exist and tells you which of the fifteen are missing.
# Runs from either workstation -- it needs only an ssh alias `picasso` and a
# writable DEST.
set -euo pipefail

REMOTE="${REMOTE:-picasso}"
RDATA="${RDATA:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph/T28_metrics}"
RLOGS="${RLOGS:-/mnt/home/users/tic_163_uma/mpascual/execs/isalgraph/logs}"
DEST="${DEST:-/home/mpascual/research/data/isalgraph_archive/data/source/T28}"

STATUS_ONLY=false
[[ "${1:-}" == "--status" ]] && STATUS_ONLY=true

echo "=== queue ==="
ssh "${REMOTE}" "squeue --me 2>/dev/null | tail -6" || echo "(could not reach ${REMOTE})"

echo ""
echo "=== shard partials on ${REMOTE} ==="
ssh "${REMOTE}" "
P=${RDATA}/families/f2_partials
n=\$(ls -1 \$P/*.json 2>/dev/null | wc -l)
echo \"\$n / 15 partials\"
[ \$n -gt 0 ] && ls -1 \$P/*.json 2>/dev/null | sed 's|.*/||; s/\.json//' | sed 's/^/  have  /'
for s in suite2/mutagenicity suite2/coil_del suite2/iam_letter_high suite2/aids_iam \
         suite2/aids_graphedx suite2/iam_letter_med suite2/iam_letter_low suite2/grec \
         suite2/protein suite2/linux suite1/iam_letter_high suite1/iam_letter_med \
         suite1/iam_letter_low suite1/aids suite1/linux; do
    f=\${s/\//__}
    [ -s \$P/\$f.json ] || echo \"  MISSING \$f\"
done
echo ''
if [ -s ${RDATA}/families/rho_table.json ]; then
    echo 'MERGE DONE: rho_table.json present'
else
    echo 'merge not done yet (rho_table.json absent)'
fi
"

${STATUS_ONLY} && exit 0

echo ""
echo "=== fetching to ${DEST} ==="
mkdir -p "${DEST}/families" "${DEST}/logs"
# --ignore-missing-args so a not-yet-written rho_table.json is not an error.
rsync -az --info=stats2 "${REMOTE}:${RDATA}/families/" "${DEST}/families/" 2>&1 | tail -3 || true
rsync -az "${REMOTE}:${RLOGS}/t28f2*" "${DEST}/logs/" 2>/dev/null || true

echo ""
echo "=== fetched ==="
find "${DEST}/families" -name '*.json' | wc -l | sed 's/^/  json files: /'
if [ -s "${DEST}/families/rho_table.json" ]; then
    echo "  rho_table.json IS present -- summarise with:"
    echo "    python benchmarks/real_data/eval_reference_metrics/headtohead_t28.py \\"
    echo "        --probe ${DEST}/families/rho_table.json --view all_pairs"
fi
