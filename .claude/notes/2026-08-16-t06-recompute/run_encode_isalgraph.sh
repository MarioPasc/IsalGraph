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
set -uo pipefail

PY=/home/mpascual/.conda/envs/isalgraph-cpp/bin/python
REPO=/home/mpascual/research/code/IsalGraph-T06
OUT=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06/encodings
BUDGET=300
JOBS=6

SUITE2=(linux grec protein aids_graphedx iam_letter_low iam_letter_med
        aids_iam iam_letter_high coil_del mutagenicity)
SUITE1=(linux aids iam_letter_low iam_letter_med iam_letter_high)

mkdir -p "$OUT/suite2" "$OUT/suite1"
cd "$REPO" || exit 1

echo "=== T-06 encoding campaign: isalgraph_pruned, budget ${BUDGET}s, jobs ${JOBS} ==="
echo "=== src_commit(shared) = $(git -C /home/mpascual/research/code/IsalGraph rev-parse --short HEAD) ==="
date -u +"start %Y-%m-%dT%H:%M:%SZ"

# Cheapest datasets first, so a failure surfaces in seconds rather than an hour.
for d in "${SUITE2[@]}"; do
  echo "--- suite2/$d ---"
  /usr/bin/time -f "  wall %e s  maxrss %M KB" \
    "$PY" -m benchmarks.real_data.eval_encoding.t06_encode \
      --suite suite2 --dataset "$d" --representation isalgraph_pruned \
      --out "$OUT/suite2" --budget-s "$BUDGET" --jobs "$JOBS" --require-cpp \
    2>&1 | tail -4
done

# Suite 1 carries BOTH arms: pruned is the reference, canonical is the
# Suite-1-only descriptive arm that gives the pruned-vs-exhaustive gap.
for d in "${SUITE1[@]}"; do
  for rep in isalgraph_pruned isalgraph_canonical; do
    echo "--- suite1/$d/$rep ---"
    "$PY" -m benchmarks.real_data.eval_encoding.t06_encode \
      --suite suite1 --dataset "$d" --representation "$rep" \
      --out "$OUT/suite1" --budget-s "$BUDGET" --jobs "$JOBS" --require-cpp \
      2>&1 | tail -3
  done
done

date -u +"end %Y-%m-%dT%H:%M:%SZ"
echo "=== files written ==="
ls -1 "$OUT"/suite2 "$OUT"/suite1 | wc -l
