#!/usr/bin/env bash
# F2, the primary family, over the 79 admissible cells -- plus the 81 cells F0's
# majority branch demoted to descriptive, which are computed rather than skipped.
#
# ORCHESTRATION ONLY. The science lives in
# benchmarks/real_data/eval_stats/{t06_f2,t06_f2_inputs}.py; this file says where
# the inputs are, how the work is split, and how to know it worked.
#
# Why sharded per (suite, dataset). The campaign is ~21 core-hours, dominated by
# the graph-level bootstrap on the two tier-3 datasets, and a single process
# would put all of it on one core behind one point of failure. One shard per
# dataset writes its own partial, so a failure costs one dataset rather than the
# campaign, a rerun is a rerun of that shard, and the fifteen shards use the
# machine. Largest first, because the critical path is mutagenicity and coil_del.
#
# Not on Picasso, deliberately: the job is ~21 core-hours of numpy over
# precomputed matrices, it needs no GPU and no encoder, the distance tree is
# ~2 GB that would have to cross the wire twice, and Picasso's queue wait is
# unbounded against an eight-day deadline. Twenty-four local cores finish the
# critical path in about three hours with no transfer and no queue.
#
# Concurrency is capped at 5, not 24: the tier-3 datasets hold ~12 matrices of
# 4040^2 float64 plus their condensed vectors and integer codes, so each of those
# processes peaks around 3 GB against 18 GB free.
set -uo pipefail

PY=${PY:-/home/mpascual/.conda/envs/isalgraph-cpp/bin/python}
REPO=${REPO:-/home/mpascual/research/code/IsalGraph-T06}
DATA=${DATA:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data}
T06="$DATA/source/T06"
OUT=${OUT:-$T06/families}
PARTIALS=${PARTIALS:-$T06/families/f2_partials}
JOBS=${JOBS:-5}
EXTRA=${EXTRA:-}

cd "$REPO" || exit 1
mkdir -p "$OUT" "$PARTIALS" "$T06/logs"

echo "=== T-06 F2 ==="
echo "code_commit = $(git rev-parse --short HEAD)"
echo "src_commit  = $(git -C /home/mpascual/research/code/IsalGraph rev-parse --short HEAD)"

# The shared checkout is another session's and can move under us; an encoder
# swap would be invisible in the output. Design note 1.4b.1.
if ! git diff --quiet "$(git -C /home/mpascual/research/code/IsalGraph rev-parse HEAD)" HEAD -- src/isalgraph/ 2>/dev/null; then
  echo "!!! src/isalgraph differs between this branch and the shared checkout we import from"
  echo "!!! see T-06-design.md 1.4b.1 -- ABORTING rather than measuring the wrong code"
  echo "DONE_MARKER rc=2 reason=src_drift"
  exit 2
fi

"$PY" -c "import isalgraph,sys; e=isalgraph.engine(); print(' engine:',e,'build:',isalgraph.build_info()['build_hash']); sys.exit(0 if e=='cpp' else 1)" || {
  echo "!!! engine is not cpp -- ABORTING"; echo "DONE_MARKER rc=2 reason=engine"; exit 2; }

date -u +"start %Y-%m-%dT%H:%M:%SZ"

# Largest first: the campaign finishes when the slowest shard does.
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
    --emit-partial "$PARTIALS/${suite}__${dataset}.json" \
    $EXTRA >"$log" 2>&1
  local rc=$?
  echo "  shard ${suite}/${dataset} rc=$rc $(date -u +%H:%M:%SZ)"
  return $rc
}
export -f run_shard
export PY T06 DATA OUT PARTIALS EXTRA

printf '%s\n' "${SHARDS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_shard {}'
shard_rc=$?

date -u +"shards done %Y-%m-%dT%H:%M:%SZ"

# File-count assertion: every shard must have left a partial. xargs reports only
# the last failure, so count the artifacts rather than trust the exit code.
want=${#SHARDS[@]}
have=$(ls -1 "$PARTIALS"/*.json 2>/dev/null | wc -l)
echo "=== partials: $have of $want ==="
fail=0
[ "$have" -eq "$want" ] || { echo "!!! missing $((want - have)) partials"; fail=$((fail + 1)); }
[ "$shard_rc" -eq 0 ] || { echo "!!! at least one shard exited non-zero"; fail=$((fail + 1)); }

"$PY" -m benchmarks.real_data.eval_stats.t06_f2 \
  --distances "$T06/distances" \
  --encodings "$T06/encodings" \
  --completion-rates "$T06/completion_rates.json" \
  --ged-root "$DATA/eval/ged_matrices" \
  --approx-root "$DATA/source/APPROX_GED" \
  --out-dir "$OUT" \
  --merge-partials "$PARTIALS" || fail=$((fail + 1))

date -u +"end %Y-%m-%dT%H:%M:%SZ"

# N_actual is the one number a silent reduction would move, so assert it here
# rather than read it off a log afterwards. 79 is frozen by design note 18.7.
n_actual=$("$PY" -c "import json;print(json.load(open('$OUT/family_F2.json'))['cardinality']['n_actual'])" 2>/dev/null || echo 0)
discrep=$("$PY" -c "import json;print(json.load(open('$OUT/family_F2.json'))['cardinality']['discrepancy'])" 2>/dev/null || echo 999)
rows=$("$PY" -c "import json;print(json.load(open('$OUT/rho_table.json'))['n_rows'])" 2>/dev/null || echo 0)
echo "=== N_actual: $n_actual (expect 79)  discrepancy: $discrep (expect 0)  rho rows: $rows ==="
[ "$n_actual" -eq 79 ] || { echo "!!! N_actual is not 79 -- escalate, do not proceed"; fail=$((fail + 1)); }
[ "$discrep" -eq 0 ] || { echo "!!! enumeration disagrees with the closed form"; fail=$((fail + 1)); }
[ "$rows" -gt 0 ] || { echo "!!! rho_table.json is empty"; fail=$((fail + 1)); }

echo "DONE_MARKER rc=$fail n_actual=$n_actual discrepancy=$discrep rows=$rows partials=$have/$want"
[ "$fail" -eq 0 ] || exit 1
