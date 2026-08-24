#!/usr/bin/env bash
# The ten small F2 shards, run into their OWN partials directory so an interim
# rho_table.json exists within minutes instead of hours.
#
# Why this exists. run_f2.sh orders its shards largest-first, because the
# campaign finishes when its slowest shard does. That is right for total
# wall clock and wrong for time-to-first-number: the five long shards occupy
# every slot for hours and the ten cheap ones, which cover ten of the fifteen
# (suite, dataset) cells, do not start at all. A methodology decision is waiting
# on those numbers, so they are computed here in parallel with the main run.
#
# The separate PARTIALS directory is not cosmetic. run_f2.sh will reach these
# same datasets as its long shards retire, and two processes writing one partial
# path would race and corrupt it. Disjoint directories make the duplicate work
# harmless; it costs a few minutes on small matrices.
#
# The authoritative artifact is still run_f2.sh's merge over all fifteen.
set -uo pipefail

PY=${PY:-/home/mpascual/.conda/envs/isalgraph-cpp/bin/python}
REPO=${REPO:-/home/mpascual/research/code/IsalGraph-T06}
DATA=${DATA:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data}
T06="$DATA/source/T06"
OUT=${OUT:-$T06/families}
PARTIALS=${PARTIALS:-$T06/families/f2_partials_early}
JOBS=${JOBS:-3}

cd "$REPO" || exit 1
mkdir -p "$OUT" "$PARTIALS" "$T06/logs"

echo "=== T-06 F2, early pass (small datasets) ==="
echo "code_commit = $(git rev-parse --short HEAD)"
echo "src_commit  = $(git -C /home/mpascual/research/code/IsalGraph rev-parse --short HEAD)"

if ! git diff --quiet "$(git -C /home/mpascual/research/code/IsalGraph rev-parse HEAD)" HEAD -- src/isalgraph/ 2>/dev/null; then
  echo "!!! src/isalgraph differs from the shared checkout -- ABORTING"
  echo "DONE_MARKER rc=2 reason=src_drift"; exit 2
fi

date -u +"start %Y-%m-%dT%H:%M:%SZ"

SHARDS=(
  "suite1 linux" "suite2 linux" "suite2 protein" "suite2 grec" "suite2 aids_graphedx"
  "suite1 aids" "suite2 iam_letter_low" "suite2 iam_letter_med"
  "suite1 iam_letter_low" "suite1 iam_letter_med"
)

run_shard() {
  local suite="$1" dataset="$2"
  "$PY" -m benchmarks.real_data.eval_stats.t06_f2 \
    --distances "$T06/distances" --encodings "$T06/encodings" \
    --completion-rates "$T06/completion_rates.json" \
    --ged-root "$DATA/eval/ged_matrices" --approx-root "$DATA/source/APPROX_GED" \
    --out-dir "$OUT" --suites "$suite" --datasets "$dataset" \
    --emit-partial "$PARTIALS/${suite}__${dataset}.json" \
    >"$T06/logs/f2early_${suite}__${dataset}.log" 2>&1
  local rc=$?
  echo "  early shard ${suite}/${dataset} rc=$rc $(date -u +%H:%M:%SZ)"
  return $rc
}
export -f run_shard
export PY T06 DATA OUT PARTIALS

printf '%s\n' "${SHARDS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_shard {}'
shard_rc=$?

date -u +"end %Y-%m-%dT%H:%M:%SZ"

want=${#SHARDS[@]}
have=$(ls -1 "$PARTIALS"/*.json 2>/dev/null | wc -l)
fail=0
echo "=== early partials: $have of $want ==="
[ "$have" -eq "$want" ] || { echo "!!! missing $((want - have)) partials"; fail=$((fail + 1)); }
[ "$shard_rc" -eq 0 ] || { echo "!!! at least one early shard exited non-zero"; fail=$((fail + 1)); }

echo "DONE_MARKER rc=$fail partials=$have/$want"
[ "$fail" -eq 0 ] || exit 1
