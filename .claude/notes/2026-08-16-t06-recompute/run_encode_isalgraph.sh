#!/usr/bin/env bash
# T-06 production encoding campaign -- the IsalGraph reference arm only.
#
# Unblocked by T-04a, which decides comparator DISTANCES, not encodings.
# Delivers the two things this ticket needs first:
#   * the D14 censoring rate at the FROZEN 300 s budget, per dataset -- never
#     measured on the full cohort, and a headline result rather than a footnote;
#   * a canonical string per Suite-2 graph, which is what T-05 deferred in full
#     to T-06 as its section 7.5 debt (rho(Lev, .) has no other input).
#
# jobs=6 on 24 cores with a single-threaded engine, so each worker has headroom
# and the 300 s wall-clock kill is not tripped by contention. That matters: a
# contended graph that would finish in 250 s alone could be killed at 400 s and
# inflate the very censoring rate this run exists to measure.
#
# NOTE: no timing from this run is publishable. Fig. 2 timings are measured
# separately, alone, and language-matched (competitors/README finding 11).
#
# v2, after v1 exited 0 having done nothing for Suite 2:
#   * /usr/bin/time is ABSENT on this workstation. v1 wrapped every Suite-2 cell
#     in it, so all ten failed; `set -uo pipefail` without -e, plus a pipe to
#     tail, swallowed it and the script still exited 0. Removed, and a per-cell
#     failure counter now makes a silent no-op impossible.
#   * --out takes the ROOT. The driver appends encodings/<suite>/ itself, so
#     v1's --out "$OUT/suite1" produced .../encodings/suite1/encodings/suite1/.
set -uo pipefail

PY=/home/mpascual/.conda/envs/isalgraph-cpp/bin/python
REPO=/home/mpascual/research/code/IsalGraph-T06
OUT=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06
BUDGET=300
JOBS=6

SUITE2=(linux grec protein aids_graphedx iam_letter_low iam_letter_med
        aids_iam iam_letter_high coil_del mutagenicity)
SUITE1=(linux aids iam_letter_low iam_letter_med iam_letter_high)

mkdir -p "$OUT"
cd "$REPO" || exit 1

fail=0
run_cell() {  # suite dataset representation
  local t0 t1
  t0=$(date +%s)
  if "$PY" -m benchmarks.real_data.eval_encoding.t06_encode \
       --suite "$1" --dataset "$2" --representation "$3" \
       --out "$OUT" --budget-s "$BUDGET" --jobs "$JOBS" --require-cpp 2>&1 | tail -2
  then
    t1=$(date +%s); echo "    [ok] $1/$2/$3 in $((t1 - t0)) s"
  else
    fail=$((fail + 1)); echo "    [FAIL] $1/$2/$3"
  fi
}

echo "=== T-06 encoding campaign: isalgraph_pruned, budget ${BUDGET}s, jobs ${JOBS} ==="
echo "=== src_commit(shared) = $(git -C /home/mpascual/research/code/IsalGraph rev-parse --short HEAD) ==="
echo "=== code_commit(T-06)  = $(git rev-parse --short HEAD) ==="
date -u +"start %Y-%m-%dT%H:%M:%SZ"

# Cheapest datasets first, so a failure surfaces in seconds rather than an hour.
for d in "${SUITE2[@]}"; do run_cell suite2 "$d" isalgraph_pruned; done

# Suite 1 carries BOTH arms: pruned is the reference, canonical is the
# Suite-1-only descriptive arm that gives the pruned-vs-exhaustive gap.
for d in "${SUITE1[@]}"; do
  run_cell suite1 "$d" isalgraph_pruned
  run_cell suite1 "$d" isalgraph_canonical
done

date -u +"end %Y-%m-%dT%H:%M:%SZ"
echo "=== cells failed: $fail ==="
echo "=== suite2 files: $(ls -1 "$OUT"/encodings/suite2 2>/dev/null | wc -l) (expect 10) ==="
echo "=== suite1 files: $(ls -1 "$OUT"/encodings/suite1 2>/dev/null | wc -l) (expect 10) ==="
[ "$fail" -eq 0 ] || exit 1
