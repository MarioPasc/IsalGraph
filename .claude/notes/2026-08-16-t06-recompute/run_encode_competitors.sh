#!/usr/bin/env bash
# T-06 production encoding campaign -- the NINE competitor representations.
#
# Companion to run_encode_isalgraph.sh, which already delivered the reference
# arm (isalgraph_pruned on both suites, isalgraph_canonical on Suite 1). This
# script delivers everything else, and it is what sets `c`: the
# per-(representation, dataset) completion rates feed the pre-registration's
# suite-restricted term.
#
# isalgraph_canonical is deliberately ABSENT from Suite 2. SUITE1_ONLY is a
# frozen T-04 policy -- the registry raises SuiteScopeError above n = 12 before
# attempting an encode -- so a Suite-2 cell would measure the guard, not the
# encoder. See T-06-design.md 11.4, where F-1 closes on exactly that ground.
#
# Conventions inherited from run_encode_isalgraph.sh v2, each one a bug it hit:
#   * /usr/bin/time is ABSENT on this workstation -- never wrap a cell in it.
#   * `set -uo pipefail` without -e lets a script exit 0 having done nothing,
#     so there is a per-cell failure counter AND a file-count assertion.
#   * --out takes the ROOT. The driver appends encodings/<suite>/ itself.
#
# jobs=6 on 24 cores with a single-threaded engine: each worker keeps headroom
# so the 300 s wall-clock kill is not tripped by contention. A contended graph
# that would finish in 250 s alone could be killed at 400 s and inflate the
# censoring rate this campaign exists to measure.
#
# NO TIMING FROM THIS RUN IS PUBLISHABLE. Cells run concurrently and the box is
# shared; Fig. 2 timings are measured separately, alone, and language-matched.
set -uo pipefail

PY=/home/mpascual/.conda/envs/isalgraph-cpp/bin/python
REPO=/home/mpascual/research/code/IsalGraph-T06
OUT=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06
BUDGET=300
JOBS=6

# Cheapest datasets first, so a failure surfaces in seconds rather than an hour.
SUITE2=(linux grec protein aids_graphedx iam_letter_low iam_letter_med
        aids_iam iam_letter_high coil_del mutagenicity)
SUITE1=(linux aids iam_letter_low iam_letter_med iam_letter_high)

# Cheapest representations first, for the same reason. agm_cam and min_dfs are
# the expensive tail and run last within each dataset.
REPS=(graph6 sparse6 nauty_graph6 sparse6_nauty adjacency size_null
      wl_subtree agm_cam min_dfs)

mkdir -p "$OUT"
cd "$REPO" || exit 1

fail=0
skip=0
done_cells=0

run_cell() {  # suite dataset representation
  local suite="$1" ds="$2" rep="$3" t0 t1 target
  target="$OUT/encodings/$suite/${ds}__${rep}.npz"
  if [ -s "$target" ]; then
    skip=$((skip + 1)); echo "    [skip] $suite/$ds/$rep (exists)"; return
  fi
  t0=$(date +%s)
  if "$PY" -m benchmarks.real_data.eval_encoding.t06_encode \
       --suite "$suite" --dataset "$ds" --representation "$rep" \
       --out "$OUT" --budget-s "$BUDGET" --jobs "$JOBS" --require-cpp 2>&1 | tail -2
  then
    t1=$(date +%s)
    if [ -s "$target" ]; then
      done_cells=$((done_cells + 1)); echo "    [ok] $suite/$ds/$rep in $((t1 - t0)) s"
    else
      # Exit 0 with no artifact is the exact failure mode v1 of the sibling
      # script hit. Treat a missing file as a failure regardless of status.
      fail=$((fail + 1)); echo "    [FAIL] $suite/$ds/$rep exited 0 but wrote NO FILE"
    fi
  else
    fail=$((fail + 1)); echo "    [FAIL] $suite/$ds/$rep"
  fi
}

echo "=== T-06 competitor encoding campaign: 9 representations, budget ${BUDGET}s, jobs ${JOBS} ==="
echo "=== src_commit(shared) = $(git -C /home/mpascual/research/code/IsalGraph rev-parse --short HEAD) ==="
echo "=== code_commit(T-06)  = $(git rev-parse --short HEAD) ==="
echo "=== engine check ==="
"$PY" -c "import isalgraph,sys; e=isalgraph.engine(); print(' engine:',e,'build:',isalgraph.build_info()['build_hash']); sys.exit(0 if e=='cpp' else 1)" || {
  echo "!!! engine is not cpp -- ABORTING before any cell"; exit 2; }
date -u +"start %Y-%m-%dT%H:%M:%SZ"

for d in "${SUITE2[@]}"; do
  echo "--- suite2/$d"
  for r in "${REPS[@]}"; do run_cell suite2 "$d" "$r"; done
done

for d in "${SUITE1[@]}"; do
  echo "--- suite1/$d"
  for r in "${REPS[@]}"; do run_cell suite1 "$d" "$r"; done
done

date -u +"end %Y-%m-%dT%H:%M:%SZ"
n2=$(ls -1 "$OUT"/encodings/suite2/*.npz 2>/dev/null | wc -l)
n1=$(ls -1 "$OUT"/encodings/suite1/*.npz 2>/dev/null | wc -l)
echo "=== cells ok: $done_cells  skipped: $skip  FAILED: $fail ==="
echo "=== suite2 files: $n2 (expect 100 = 10 datasets x 10 reps incl. isalgraph_pruned) ==="
echo "=== suite1 files: $n1 (expect 55 = 5 datasets x 11 reps incl. both isalgraph arms) ==="

rc=0
[ "$fail" -eq 0 ] || rc=1
[ "$n2" -eq 100 ] || { echo "!!! suite2 file-count assertion FAILED"; rc=1; }
[ "$n1" -eq 55 ]  || { echo "!!! suite1 file-count assertion FAILED"; rc=1; }

# DONE_MARKER is the last line on every path, success or failure, so a watcher
# can distinguish "finished badly" from "still running" from "died silently".
echo "DONE_MARKER rc=$rc fail=$fail ok=$done_cells skip=$skip suite2=$n2 suite1=$n1"
exit "$rc"
