#!/usr/bin/env bash
# T-06 production distance campaign.
#
# GATED. Acceptance criterion 5 must pass first -- the shipped module must
# reproduce T-04a's corrected table at max |delta| = 0.0000 -- and this script
# re-checks the gate artefact before computing anything. Passed 2026-08-23:
# 113 numeric values, max |delta| = 0.0000000000.
#
# Three kinds of cell, and the split is not cosmetic:
#
#   levenshtein   symbol-level, per CONTRACTS 3.1. The unit is Encoding.symbols,
#                 never characters: a min_dfs symbol is a whole DFS tuple, so a
#                 character-level distance charges ~4 edits for one deleted
#                 tuple on the comparator competitors.md calls "the single most
#                 important" one.
#   kernel        wl_subtree only, via wl_driver, which needs the exported CSR
#                 cohort rather than the encodings. h is frozen at WL_ROUNDS = 2
#                 and is deliberately not a flag.
#   size_null     ONE PER (representation, dataset), per CONTRACTS 4.1 -- not one
#                 per dataset. Censoring is not independent of size and differs
#                 by representation, so a shared null would be computed over
#                 pairs the arm was never evaluated on. Measured on Mutagenicity:
#                 whole-cohort null 0.7538 against restricted 0.6363, while the
#                 arm itself does not move.
#
# isalgraph_canonical is Suite-1 only. SUITE1_ONLY is a frozen T-04 policy, not
# a performance outcome -- see T-06-design.md 11.4, where F-1 closes on it.
#
# adjacency, graph6 and sparse6 are absent by design: T-04a's k = 3 excludes
# them for failing the metric axioms at F3 = 1/50, so they carry no Claim-B cell
# and no distance is admissible for them.
#
# NO TIMING FROM THIS RUN IS PUBLISHABLE: it shares the box.
set -uo pipefail

PY=${PY:-/home/mpascual/.conda/envs/isalgraph-cpp/bin/python}
REPO=${REPO:-/home/mpascual/research/code/IsalGraph-T06}
DATA=${DATA:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data}
T06="$DATA/source/T06"
OUT="$T06/distances"
JOBS=${JOBS:-8}

LEV_REPS=(nauty_graph6 sparse6_nauty agm_cam min_dfs isalgraph_pruned)
SUITE2=(linux grec protein aids_graphedx iam_letter_low iam_letter_med
        aids_iam iam_letter_high coil_del mutagenicity)
SUITE1=(linux aids iam_letter_low iam_letter_med iam_letter_high)

cd "$REPO" || exit 1
mkdir -p "$OUT/suite1" "$OUT/suite2"

echo "=== T-06 distance campaign ==="
echo "code_commit = $(git rev-parse --short HEAD)"
echo "src_commit  = $(git -C /home/mpascual/research/code/IsalGraph rev-parse --short HEAD)"

# The reproduction gate is a precondition, not a formality.
GATE="$T06/gates/gate_T06_reproduction.json"
"$PY" -c "
import json,sys
r=json.load(open('$GATE'))
ok = r['passed'] and r['max_abs_delta_observed'] == 0.0
print(' reproduction gate:', 'PASS' if ok else 'FAIL',
      f\"({r['numeric_values_compared']} values, max |delta| = {r['max_abs_delta_observed']:.10f})\")
sys.exit(0 if ok else 1)
" || { echo "!!! reproduction gate has not passed -- refusing to compute a production matrix"; exit 2; }

# 1.4b.1: the shared checkout is another session's and can move under us.
"$PY" -c "import isalgraph,sys; e=isalgraph.engine(); print(' engine:',e,'build:',isalgraph.build_info()['build_hash']); sys.exit(0 if e=='cpp' else 1)" || {
  echo "!!! engine is not cpp -- ABORTING"; exit 2; }

date -u +"start %Y-%m-%dT%H:%M:%SZ"
fail=0; ok=0; skip=0

run_lev() {  # suite dataset representation
  local suite="$1" ds="$2" rep="$3" enc target
  enc="$T06/encodings/$suite/${ds}__${rep}.npz"
  target="$OUT/$suite/${ds}__${rep}__levenshtein.npz"
  [ -f "$enc" ] || { echo "    [skip] $suite/$ds/$rep (no encoding)"; skip=$((skip+1)); return; }
  [ -s "$target" ] && { skip=$((skip+1)); echo "    [skip] $suite/$ds/$rep lev (exists)"; return; }
  if "$PY" -m benchmarks.real_data.eval_distance.distance_runner \
       --encodings "$enc" --metric levenshtein --out "$OUT/$suite" \
       --n-chunks 1 --chunk-index 0 --jobs "$JOBS" --suite "$suite" >/dev/null 2>&1 \
     && [ -s "$target" ]; then
    ok=$((ok+1)); echo "    [ok] $suite/$ds/$rep lev"
    rm -f "$OUT/$suite/${ds}__${rep}__levenshtein.shard"*.npz
  else
    fail=$((fail+1)); echo "    [FAIL] $suite/$ds/$rep lev"
  fi
}

run_null() {  # suite dataset representation
  local suite="$1" ds="$2" rep="$3" enc target
  enc="$T06/encodings/$suite/${ds}__${rep}.npz"
  target="$OUT/$suite/${ds}__${rep}__size_null.npz"
  [ -f "$enc" ] || { skip=$((skip+1)); return; }
  [ -s "$target" ] && { skip=$((skip+1)); return; }
  if "$PY" -m benchmarks.real_data.eval_distance.size_null \
       --encodings "$enc" --out "$OUT/$suite" --suite "$suite" >/dev/null 2>&1 \
     && [ -s "$target" ]; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "    [FAIL] $suite/$ds/$rep size_null"
  fi
}

run_wl() {  # suite dataset cohort_dir
  local suite="$1" ds="$2" cohort_dir="$3" target
  target="$OUT/$suite/${ds}__wl_subtree__kernel.npz"
  [ -s "$target" ] && { skip=$((skip+1)); echo "    [skip] $suite/$ds wl (exists)"; return; }
  if "$PY" -m benchmarks.real_data.eval_distance.wl_driver \
       --cohort "$cohort_dir/${ds}.npz" \
       --reference-encodings "$T06/encodings/$suite/${ds}__wl_subtree.npz" \
       --out "$OUT/$suite" --suite "$suite" >/dev/null 2>&1 \
     && [ -s "$target" ]; then
    ok=$((ok+1)); echo "    [ok] $suite/$ds wl kernel"
  else
    fail=$((fail+1)); echo "    [FAIL] $suite/$ds wl kernel"
  fi
}

for d in "${SUITE2[@]}"; do
  echo "--- suite2/$d"
  for r in "${LEV_REPS[@]}"; do run_lev suite2 "$d" "$r"; run_null suite2 "$d" "$r"; done
  run_wl suite2 "$d" "$DATA/exported_suite2"
  run_null suite2 "$d" wl_subtree
done

for d in "${SUITE1[@]}"; do
  echo "--- suite1/$d"
  for r in "${LEV_REPS[@]}" isalgraph_canonical; do run_lev suite1 "$d" "$r"; run_null suite1 "$d" "$r"; done
  run_wl suite1 "$d" "$DATA/exported"
  run_null suite1 "$d" wl_subtree
done

date -u +"end %Y-%m-%dT%H:%M:%SZ"
n2=$(ls -1 "$OUT"/suite2/*.npz 2>/dev/null | wc -l)
n1=$(ls -1 "$OUT"/suite1/*.npz 2>/dev/null | wc -l)
echo "=== ok: $ok  skipped: $skip  FAILED: $fail ==="
echo "=== suite2 files: $n2 (expect 120 = 10 x (5 lev + 6 null + 1 wl)) ==="
echo "=== suite1 files: $n1 (expect  70 =  5 x (6 lev + 7 null + 1 wl)) ==="

rc=0
[ "$fail" -eq 0 ] || rc=1
[ "$n2" -eq 120 ] || { echo "!!! suite2 file-count assertion FAILED"; rc=1; }
[ "$n1" -eq 70 ]  || { echo "!!! suite1 file-count assertion FAILED"; rc=1; }
echo "DONE_MARKER rc=$rc fail=$fail ok=$ok skip=$skip suite2=$n2 suite1=$n1"
exit "$rc"
