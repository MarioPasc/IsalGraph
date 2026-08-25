#!/usr/bin/env bash
# T-06-exhaustive encoding campaign -- the isalgraph_exhaustive arm ONLY.
#
# Descends from run_encode_isalgraph.sh. Three differences, all deliberate:
#
#   * representation is isalgraph_exhaustive, which computes the true w*_G
#     rather than the length-suboptimal pruned form. Measured 8-12 % shorter at
#     n = 13-20 and 12-22 % at n = 23-26, and never longer on any of the 5,350
#     Suite-1 graphs.
#   * BUDGET is 30 s, not the frozen 300 s of T-06. The completion distribution
#     is heavy-tailed -- 100 % through n = 20 at 60 s with a median of 9 ms
#     against a 33 s maximum -- so 30 s captures nearly all the benefit at a
#     tenth of the worst-case cost. THE BUDGET IS RECORDED IN EVERY CELL'S
#     metadata.encode_budget_s: a censoring rate is a property of its budget and
#     quoting one without the other is meaningless.
#   * output goes to T06_exhaustive, NEVER to T06. The original campaign is the
#     pre-registered record and must survive byte-identical.
#
# NO COMPETITOR IS RE-ENCODED and NO GED MATRIX IS RECOMPUTED. Both are
# unchanged by this work and are reused verbatim through symlinks.
#
# jobs=6 on 24 cores with a single-threaded engine, so each worker has headroom
# and the wall-clock kill is not tripped by contention. That matters: a
# contended graph that would finish in 25 s alone could be killed at 40 s and
# inflate the very censoring rate this run exists to measure.
#
# NOTE: no timing from this run is publishable. It shares the box.
set -uo pipefail

PY=${PY:-/home/mpascual/.conda/envs/isalgraph-cpp/bin/python}
REPO=${REPO:-/home/mpascual/research/code/IsalGraph}
OUT=${OUT:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06_exhaustive}
BUDGET=${BUDGET:-30}
JOBS=${JOBS:-6}
REP=${REP:-isalgraph_exhaustive}

SUITE2=(linux grec protein aids_graphedx iam_letter_low iam_letter_med
        aids_iam iam_letter_high coil_del mutagenicity)
SUITE1=(linux aids iam_letter_low iam_letter_med iam_letter_high)

export PYTHONPATH="$REPO"
mkdir -p "$OUT/logs"
cd "$REPO" || exit 1

LOG="$OUT/logs/encode_exhaustive_$(date -u +%Y%m%dT%H%M%SZ).log"

fail=0; ok=0; skip=0
run_cell() {  # suite dataset
  local t0 t1 target
  target="$OUT/encodings/$1/$2__${REP}.npz"
  # Resumable: a completed cell is never recomputed, so an interrupted run
  # restarts where it stopped rather than from the beginning.
  if [ -s "$target" ] && [ ! -L "$target" ]; then
    skip=$((skip + 1)); echo "    [skip] $1/$2 (exists)"; return
  fi
  t0=$(date +%s)
  if "$PY" -m benchmarks.real_data.eval_encoding.t06_encode \
       --suite "$1" --dataset "$2" --representation "$REP" \
       --out "$OUT" --budget-s "$BUDGET" --jobs "$JOBS" --require-cpp 2>&1 | tail -3
  then
    t1=$(date +%s); ok=$((ok + 1)); echo "    [ok] $1/$2 in $((t1 - t0)) s"
  else
    fail=$((fail + 1)); echo "    [FAIL] $1/$2"
  fi
}

{
echo "=== T-06-exhaustive encoding: $REP, budget ${BUDGET}s, jobs ${JOBS} ==="
echo "=== src_commit = $(git -C "$REPO" rev-parse --short HEAD) ==="
"$PY" -c "import isalgraph,sys; e=isalgraph.engine(); print(' engine:',e,'build:',isalgraph.build_info()['build_hash']); sys.exit(0 if e=='cpp' else 1)" \
  || { echo "!!! engine is not cpp -- ABORTING"; exit 2; }
date -u +"start %Y-%m-%dT%H:%M:%SZ"

# Suite 1 first: it is small, and a defect surfaces in minutes rather than hours.
for d in "${SUITE1[@]}"; do run_cell suite1 "$d"; done
# Suite 2 cheapest-first.
for d in "${SUITE2[@]}"; do run_cell suite2 "$d"; done

date -u +"end %Y-%m-%dT%H:%M:%SZ"
echo "=== ok: $ok  skipped: $skip  FAILED: $fail ==="
n1=$(ls -1 "$OUT"/encodings/suite1/*__${REP}.npz 2>/dev/null | wc -l)
n2=$(ls -1 "$OUT"/encodings/suite2/*__${REP}.npz 2>/dev/null | wc -l)
echo "=== suite1 cells: $n1 (expect 5)  suite2 cells: $n2 (expect 10) ==="
echo "DONE_MARKER rc=$fail ok=$ok skip=$skip suite1=$n1 suite2=$n2"
} 2>&1 | tee "$LOG"

[ "$fail" -eq 0 ] || exit 1
