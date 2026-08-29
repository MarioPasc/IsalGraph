#!/usr/bin/env bash
# Run the T-28 F2 shards LOCALLY, cheapest cell first.
#
# The Picasso array is the campaign of record. This exists because the queue can
# sit on Priority/ for hours and the paired bootstrap intervals are what decide
# whether the gap to min_dfs under the WL reference is a loss or a TIE -- a
# question a point estimate cannot answer.
#
# The two heaviest cells (mutagenicity 8.16 M pairs, coil_del 7.60 M) are NOT run
# here; they are tier-3 and belong on the cluster. Everything else fits.
set -uo pipefail

A="${A:-/home/mpascual/research/data/isalgraph_archive}"
PY="${PY:-/home/mpascual/.conda/envs/isalgraph-cpp/bin/python}"
OUT="${OUT:-${A}/data/source/T28/families_local}"
JOBS="${JOBS:-5}"

export T28_REFERENCE_ROOT="${A}/data/source/T28/references"
export T06_REFERENCE_ARM=isalgraph_pruned
export PYTHONUNBUFFERED=1

DIST="${A}/data/source/T06/distances"
ENC="${A}/data/source/T06/encodings"
GED="${A}/data/eval/ged_matrices"
APPROX="${A}/data/source/APPROX_GED"
COMPLETION="${OUT}/completion_rates.json"
PARTIALS="${OUT}/f2_partials"
LOGS="${OUT}/logs"
mkdir -p "${PARTIALS}" "${LOGS}"

if [ ! -s "${COMPLETION}" ]; then
    echo "[setup] generating completion rates from ${ENC}"
    "$PY" -m benchmarks.real_data.eval_encoding.t06_completion \
        --encodings "${ENC}" --out "${COMPLETION}" \
        >"${LOGS}/completion.log" 2>&1 \
        || { echo "[FATAL] completion rates failed; see ${LOGS}/completion.log"; exit 2; }
fi
echo "[setup] completion rates: ${COMPLETION}"

# Cheapest first, so the intervals that answer the min_dfs question arrive early.
UNITS=(
  suite1/linux suite2/linux suite1/aids suite2/grec suite2/protein
  suite1/iam_letter_low suite1/iam_letter_med suite1/iam_letter_high
  suite2/aids_graphedx suite2/iam_letter_low suite2/iam_letter_med
  suite2/iam_letter_high suite2/aids_iam
)

run_one() {
    local unit="$1" suite dataset partial t0
    suite="${unit%%/*}"; dataset="${unit##*/}"
    partial="${PARTIALS}/${suite}__${dataset}.json"
    [ -s "${partial}" ] && { echo "[skip] ${unit}"; return 0; }
    t0=$(date +%s)
    if "$PY" -m benchmarks.real_data.eval_stats.t06_f2 \
        --distances "${DIST}" --encodings "${ENC}" \
        --completion-rates "${COMPLETION}" \
        --ged-root "${GED}" --approx-root "${APPROX}" \
        --out-dir "${OUT}" --suites "${suite}" --datasets "${dataset}" \
        --emit-partial "${partial}" \
        >"${LOGS}/f2_${suite}__${dataset}.log" 2>&1 && [ -s "${partial}" ]; then
        echo "[ok]   ${unit} in $(( $(date +%s) - t0 )) s"
    else
        echo "[FAIL] ${unit} -- see ${LOGS}/f2_${suite}__${dataset}.log"
    fi
}
export -f run_one
export PY DIST ENC COMPLETION GED APPROX OUT PARTIALS LOGS

printf '%s\n' "${UNITS[@]}" | xargs -P "${JOBS}" -I{} bash -c 'run_one "$@"' _ {}
echo "[done] $(ls -1 "${PARTIALS}"/*.json 2>/dev/null | wc -l) partials in ${PARTIALS}"
