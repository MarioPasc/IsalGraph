#!/usr/bin/env bash
# F-1 RE-VERIFICATION -- does the reference-arm decision survive the retraction
# of the probe's timings?
#
# WHY THIS EXISTS. F-1 froze `isalgraph_pruned` as the reference arm on both
# suites, on a probe that measured `canonical` killed on 20/45 graphs of the
# three largest Suite-2 datasets at a 30 s budget. That probe was later found to
# encode ONE GRAPH PER SUBPROCESS, so every measurement was a cold first call
# into the C++ extension and paid one-time warm-up -- inflating a single COIL-DEL
# graph from a true 4.95 ms to a measured 578.95 ms, a factor of 117.
#
# Kill counts are more robust than medians: a kill needs the budget's worth of
# measured time to elapse. But a two-order inflation could turn a 2 s encode
# into a 30 s kill, and F-1 is a frozen decision resting on those counts. So it
# is re-measured with the AMORTISED driver, which pays warm-up once per process.
#
# DESIGN. 200 graphs per dataset on the three that killed (protein 10/15,
# coil_del 7/15, mutagenicity 3/15), budget 60 s, jobs 1.
#   * jobs=1, not 6: the budget is a WALL-CLOCK kill and this run exists to
#     measure kills. Concurrency would inflate exactly the quantity under test,
#     which is the mistake being corrected, in a different disguise.
#   * 60 s, not 300 s: bounds the worst case at 200*60 = 3.3 h per dataset while
#     still being 2x the probe's 30 s. A cell that does not kill at 60 s
#     amortised, having killed at 30 s cold, settles the question.
#   * 200, not the full cohort: enough to separate 10/15 from 0/15; the
#     production 300 s rate is a separate deliverable.
#
# READING THE RESULT.
#   censored stays high  -> F-1 STANDS, the probe was directionally right.
#   censored collapses   -> F-1 must be REOPENED with the PI: `canonical` may be
#                           viable on Suite 2 after all, and it is the arm T-04
#                           measured as stronger on two equal-n columns.
set -uo pipefail

PY=/home/mpascual/.conda/envs/isalgraph-cpp/bin/python
REPO=/home/mpascual/research/code/IsalGraph-T06
OUT=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06/f1_recheck
BUDGET=60
LIMIT=200

cd "$REPO" || exit 1
mkdir -p "$OUT"

date -u +"f1 recheck start %Y-%m-%dT%H:%M:%SZ"
echo "budget=${BUDGET}s limit=${LIMIT} jobs=1 (deliberate: a kill test must not be contended)"
for d in protein coil_del mutagenicity; do
  for rep in isalgraph_canonical isalgraph_pruned; do
    echo "--- $d / $rep ---"
    "$PY" -m benchmarks.real_data.eval_encoding.t06_encode \
      --suite suite2 --dataset "$d" --representation "$rep" \
      --out "$OUT" --budget-s "$BUDGET" --jobs 1 --limit "$LIMIT" --require-cpp 2>&1 | tail -2
  done
done
date -u +"f1 recheck end %Y-%m-%dT%H:%M:%SZ"
echo "F1_RECHECK_DONE"
