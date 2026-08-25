#!/usr/bin/env bash
# F2 for the T06_exhaustive campaign. Descends from experiments/paper_pipeline/run_f2.sh.
#
# Four differences from the T-06 driver, all deliberate.
#
#   * T06="$DATA/source/T06_exhaustive". Nothing here writes into source/T06.
#   * T06_REFERENCE_ARM=isalgraph_exhaustive re-points the reference arm. The
#     comparators are untouched, and so are the GED matrices, which are read
#     from the SAME roots as T-06 read them.
#   * The comparator set is the FULL one. graph6, sparse6 and adjacency stay in
#     the data. `--comparator-set reduced` emits the reduced reporting view as a
#     SEPARATE output; it is never the primary run. Dropping a competitor from a
#     table costs nothing; dropping it from the campaign changes the cardinality
#     of a pre-registered confirmatory family.
#   * N_actual is REPORTED, not asserted at 79.
#
#     This is the one substantive relaxation and it needs saying. In T-06,
#     N_actual = 101 - 5*3 - 7, where the trailing 7 is the cells excluded for
#     completion. The exhaustive arm has a DIFFERENT completion profile from the
#     pruned arm, so that term can legitimately move, and a moved N_actual here
#     is a finding rather than a defect. Aborting on it would discard the result
#     the campaign exists to produce. T-06's own 79 is unaffected -- it lives in
#     source/T06 and is not rewritten.
#
# Concurrency is capped at 5, not 24: the tier-3 datasets hold ~12 matrices of
# 4040^2 float64 plus their condensed vectors, so each of those processes peaks
# around 3 GB.
set -uo pipefail

PY=${PY:-/home/mpascual/.conda/envs/isalgraph-cpp/bin/python}
REPO=${REPO:-/home/mpascual/research/code/IsalGraph}
DATA=${DATA:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data}
T06="$DATA/source/T06_exhaustive"
OUT=${OUT:-$T06/families}
PARTIALS=${PARTIALS:-$T06/families/f2_partials}
JOBS=${JOBS:-5}
EXTRA=${EXTRA:-}
COMPARATOR_SET=${COMPARATOR_SET:-full}

export PYTHONPATH="$REPO"
export T06_REFERENCE_ARM=isalgraph_exhaustive

cd "$REPO" || exit 1
mkdir -p "$OUT" "$PARTIALS" "$T06/logs"

echo "=== T06_exhaustive F2 (reference arm: $T06_REFERENCE_ARM, comparators: $COMPARATOR_SET) ==="
echo "src_commit = $(git rev-parse --short HEAD)"

"$PY" -c "import isalgraph,sys; e=isalgraph.engine(); print(' engine:',e,'build:',isalgraph.build_info()['build_hash']); sys.exit(0 if e=='cpp' else 1)" || {
  echo "!!! engine is not cpp -- ABORTING"; echo "DONE_MARKER rc=2 reason=engine"; exit 2; }

# Completion rates for THIS arm. The T-06 file describes the pruned arm and
# would silently mis-describe this one.
COMPLETION="$T06/completion_rates.json"
if [ ! -s "$COMPLETION" ]; then
  "$PY" -m benchmarks.real_data.eval_encoding.t06_completion \
    --encodings "$T06/encodings" --out "$COMPLETION" || {
    echo "!!! completion rates failed"; echo "DONE_MARKER rc=2 reason=completion"; exit 2; }
fi
echo " completion rates: $COMPLETION"

date -u +"start %Y-%m-%dT%H:%M:%SZ"

SHARDS=(
  "suite2 mutagenicity" "suite2 coil_del" "suite2 iam_letter_high" "suite2 aids_iam"
  "suite1 iam_letter_high" "suite1 iam_letter_med" "suite1 iam_letter_low"
  "suite2 iam_letter_med" "suite2 iam_letter_low" "suite2 aids_graphedx"
  "suite2 grec" "suite2 protein" "suite1 aids" "suite2 linux" "suite1 linux"
)

run_shard() {
  local suite="$1" dataset="$2"
  local log="$T06/logs/f2_${suite}__${dataset}.log"
  "$PY" -m benchmarks.real_data.eval_stats.t06_f2 \
    --distances "$T06/distances" \
    --encodings "$T06/encodings" \
    --completion-rates "$T06/completion_rates.json" \
    --ged-root "$DATA/eval/ged_matrices" \
    --approx-root "$DATA/source/APPROX_GED" \
    --out-dir "$OUT" \
    --suites "$suite" \
    --datasets "$dataset" \
    --comparator-set "$COMPARATOR_SET" \
    --emit-partial "$PARTIALS/${suite}__${dataset}.json" \
    $EXTRA >"$log" 2>&1
  local rc=$?
  echo "  shard ${suite}/${dataset} rc=$rc $(date -u +%H:%M:%SZ)"
  return $rc
}
export -f run_shard
export PY T06 DATA OUT PARTIALS EXTRA COMPARATOR_SET T06_REFERENCE_ARM PYTHONPATH

printf '%s\n' "${SHARDS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_shard {}'
shard_rc=$?

date -u +"shards done %Y-%m-%dT%H:%M:%SZ"

want=${#SHARDS[@]}
have=$(ls -1 "$PARTIALS"/*.json 2>/dev/null | wc -l)
echo "=== partials: $have of $want ==="
fail=0
[ "$have" -eq "$want" ] || { echo "!!! missing $((want - have)) partials"; fail=$((fail + 1)); }

"$PY" -m benchmarks.real_data.eval_stats.t06_f2 \
  --distances "$T06/distances" \
  --encodings "$T06/encodings" \
  --completion-rates "$T06/completion_rates.json" \
  --ged-root "$DATA/eval/ged_matrices" \
  --approx-root "$DATA/source/APPROX_GED" \
  --out-dir "$OUT" \
  --comparator-set "$COMPARATOR_SET" \
  --merge-partials "$PARTIALS" || fail=$((fail + 1))

date -u +"end %Y-%m-%dT%H:%M:%SZ"

n_actual=$("$PY" -c "import json;print(json.load(open('$OUT/family_F2.json'))['cardinality']['n_actual'])" 2>/dev/null || echo 0)
discrep=$("$PY" -c "import json;print(json.load(open('$OUT/family_F2.json'))['cardinality']['discrepancy'])" 2>/dev/null || echo 999)
rows=$("$PY" -c "import json;print(json.load(open('$OUT/rho_table.json'))['n_rows'])" 2>/dev/null || echo 0)

# REPORTED, not asserted: see the header. T-06's own N_actual = 79 is untouched.
echo "=== N_actual: $n_actual (T-06 measured 79 on the pruned arm) ==="
echo "=== discrepancy: $discrep (expect 0 -- enumeration vs closed form) ==="
echo "=== rho rows: $rows ==="
# The discrepancy IS asserted: it compares the enumeration against the closed
# form within this run, so a non-zero value is an internal inconsistency
# whatever the arm.
[ "$discrep" -eq 0 ] || { echo "!!! enumeration disagrees with the closed form"; fail=$((fail + 1)); }
[ "$rows" -gt 0 ] || { echo "!!! rho_table.json is empty"; fail=$((fail + 1)); }

echo "DONE_MARKER rc=$fail n_actual=$n_actual discrepancy=$discrep rows=$rows partials=$have/$want"
[ "$fail" -eq 0 ] || exit 1
