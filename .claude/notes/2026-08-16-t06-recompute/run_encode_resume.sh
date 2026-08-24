#!/usr/bin/env bash
# Resume the T-06 IsalGraph encoding campaign: skip cells already on disk.
#
# v2 completed 9 of 10 Suite-2 cells (all censored=0) and was killed by the
# orchestrator's 10-minute tool cap while running mutagenicity -- the one
# dataset with genuinely hard graphs. COIL-DEL's 3,900 finished in 1 s;
# mutagenicity was still going after 7 minutes, which is consistent with
# canonicalisation cost tracking |Aut(G)| rather than size (data.md section 4).
#
# So this runs DETACHED via setsid+nohup and is polled, rather than being held
# open by a tool call it will outlive.
set -uo pipefail

PY=/home/mpascual/.conda/envs/isalgraph-cpp/bin/python
REPO=/home/mpascual/research/code/IsalGraph-T06
OUT=/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06
BUDGET=300
JOBS=6

cd "$REPO" || exit 1
fail=0

run_cell() {  # suite dataset representation
  local out="$OUT/encodings/$1/$2__$3.npz" t0 t1
  if [ -f "$out" ]; then echo "    [skip] $1/$2/$3 (exists)"; return; fi
  t0=$(date +%s)
  if "$PY" -m benchmarks.real_data.eval_encoding.t06_encode \
       --suite "$1" --dataset "$2" --representation "$3" \
       --out "$OUT" --budget-s "$BUDGET" --jobs "$JOBS" --require-cpp 2>&1 | tail -2
  then t1=$(date +%s); echo "    [ok] $1/$2/$3 in $((t1 - t0)) s"
  else fail=$((fail + 1)); echo "    [FAIL] $1/$2/$3"
  fi
}

date -u +"resume start %Y-%m-%dT%H:%M:%SZ"
for d in linux grec protein aids_graphedx iam_letter_low iam_letter_med \
         aids_iam iam_letter_high coil_del mutagenicity; do
  run_cell suite2 "$d" isalgraph_pruned
done
for d in linux aids iam_letter_low iam_letter_med iam_letter_high; do
  run_cell suite1 "$d" isalgraph_pruned
  run_cell suite1 "$d" isalgraph_canonical
done
date -u +"resume end %Y-%m-%dT%H:%M:%SZ"
echo "=== cells failed: $fail ==="
echo "=== suite2: $(ls -1 "$OUT"/encodings/suite2 2>/dev/null | wc -l)/10  suite1: $(ls -1 "$OUT"/encodings/suite1 2>/dev/null | wc -l)/10 ==="
echo "DONE_MARKER"
