#!/usr/bin/env bash
# Size-profile experiment: Spearman rho against GED within equal-n strata,
# plus the three figures built from it.
#
# ORCHESTRATION ONLY. The science lives in
# benchmarks/real_data/eval_size_profile/{size_profile,figures}.py; this file
# only says where the inputs are, where the outputs go, and how to know it
# worked.
#
# What the figures answer that the per-dataset tables cannot: within a stratum
# every pair has n_i == n_j, so the size null |n_i - n_j| is identically zero
# and its rank correlation is undefined. There is nothing to subtract, and raw
# rho inside a stratum is the structural signal with the size channel removed by
# construction rather than by adjustment.
#
# Reference follows the exact-computability ceiling: exact GED at n <= 12, the
# proven LB/UB bracket above it, reported as two series and never interpolated
# into a midpoint.
#
# Runtime: ~45 min on 24 cores, dominated by the graph-level bootstrap.
set -uo pipefail

PY=${PY:-/home/mpascual/.conda/envs/isalgraph-cpp/bin/python}
REPO=${REPO:-/home/mpascual/research/code/IsalGraph-T06}
DATA=${DATA:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data}
T06="$DATA/source/T06"
OUT_FIGS=${OUT_FIGS:-$T06/figures/size_profile}

cd "$REPO" || exit 1
mkdir -p "$OUT_FIGS" "$T06/logs"

echo "=== T-06 size profile ==="
echo "code_commit = $(git rev-parse --short HEAD)"
echo "src_commit  = $(git -C /home/mpascual/research/code/IsalGraph rev-parse --short HEAD)"

# The shared checkout is another session's and can move under us; an encoder
# swap would be invisible in the output. Two git calls, loud failure.
if ! git diff --quiet "$(git -C /home/mpascual/research/code/IsalGraph rev-parse HEAD)" HEAD -- src/isalgraph/ 2>/dev/null; then
  echo "!!! src/isalgraph differs between this branch and the shared checkout we import from"
  echo "!!! see T-06-design.md 1.4b.1 -- ABORTING rather than measuring the wrong encoder"
  exit 2
fi

"$PY" -c "import isalgraph,sys; e=isalgraph.engine(); print(' engine:',e,'build:',isalgraph.build_info()['build_hash']); sys.exit(0 if e=='cpp' else 1)" || {
  echo "!!! engine is not cpp -- ABORTING"; exit 2; }

date -u +"start %Y-%m-%dT%H:%M:%SZ"

fail=0
"$PY" -m benchmarks.real_data.eval_size_profile.size_profile \
  --encodings "$T06/encodings" \
  --ged-root "$DATA/eval/ged_matrices" \
  --approx-root "$DATA/source/APPROX_GED" \
  --out "$T06/size_profile.json" || fail=$((fail + 1))

"$PY" -m benchmarks.real_data.eval_size_profile.figures \
  --profile "$T06/size_profile.json" \
  --out-dir "$OUT_FIGS" || fail=$((fail + 1))

date -u +"end %Y-%m-%dT%H:%M:%SZ"

rows=$("$PY" -c "import json;print(len(json.load(open('$T06/size_profile.json'))['rows']))" 2>/dev/null || echo 0)
figs=$(ls -1 "$OUT_FIGS"/*.pdf 2>/dev/null | wc -l)
echo "=== rows: $rows   figures: $figs (expect 3) ==="
[ "$rows" -gt 0 ] || { echo "!!! no rows written"; fail=$((fail + 1)); }
[ "$figs" -eq 3 ] || { echo "!!! expected 3 figures"; fail=$((fail + 1)); }

echo "DONE_MARKER rc=$fail rows=$rows figs=$figs"
[ "$fail" -eq 0 ] || exit 1
