#!/usr/bin/env bash
# T-06-exhaustive distance campaign -- the isalgraph_exhaustive arm ONLY.
#
# GATED, exactly as run_distances.sh is: acceptance criterion 5 must pass -- the
# shipped module must reproduce T-04a's corrected table at max |delta| = 0.0000
# -- and this script re-checks the gate artefact before computing anything.
#
# NO GED MATRIX IS RECOMPUTED and NO COMPETITOR DISTANCE IS RECOMPUTED. Every
# competitor cell already exists under T06/distances and is reachable here
# through a symlink; the `[ -s "$target" ]` guard follows symlinks, so those
# cells are skipped rather than recomputed. This is the bulk of the saving.
#
# Two cells per (suite, dataset) are new:
#   levenshtein   symbol-level, per CONTRACTS 3.1.
#   size_null     ONE PER (representation, dataset), per CONTRACTS 4.1 -- the
#                 new arm censors a different set of graphs from the pruned arm,
#                 so it needs its OWN null. Sharing the pruned arm's null would
#                 compute the baseline over pairs this arm was never evaluated
#                 on, which is the exact defect 4.1 was written against.
#
# NO TIMING FROM THIS RUN IS PUBLISHABLE: it shares the box.
set -uo pipefail

PY=${PY:-/home/mpascual/.conda/envs/isalgraph-cpp/bin/python}
REPO=${REPO:-/home/mpascual/research/code/IsalGraph}
DATA=${DATA:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data}
NEW="$DATA/source/T06_exhaustive"
OUT="$NEW/distances"
JOBS=${JOBS:-8}
REP=isalgraph_exhaustive

SUITE2=(linux grec protein aids_graphedx iam_letter_low iam_letter_med
        aids_iam iam_letter_high coil_del mutagenicity)
SUITE1=(linux aids iam_letter_low iam_letter_med iam_letter_high)

export PYTHONPATH="$REPO"
cd "$REPO" || exit 1
mkdir -p "$OUT/suite1" "$OUT/suite2" "$NEW/logs"
LOG="$NEW/logs/distances_exhaustive_$(date -u +%Y%m%dT%H%M%SZ).log"

{
echo "=== T-06-exhaustive distance campaign: $REP ==="
echo "src_commit = $(git -C "$REPO" rev-parse --short HEAD)"

GATE="$NEW/gates/gate_T06_reproduction.json"
"$PY" -c "
import json,sys
r=json.load(open('$GATE'))
ok = r['passed'] and r['max_abs_delta_observed'] == 0.0
print(' reproduction gate:', 'PASS' if ok else 'FAIL',
      f\"({r['numeric_values_compared']} values, max |delta| = {r['max_abs_delta_observed']:.10f})\")
sys.exit(0 if ok else 1)
" || { echo "!!! reproduction gate has not passed -- refusing to compute a production matrix"; exit 2; }

"$PY" -c "import isalgraph,sys; e=isalgraph.engine(); print(' engine:',e,'build:',isalgraph.build_info()['build_hash']); sys.exit(0 if e=='cpp' else 1)" \
  || { echo "!!! engine is not cpp -- ABORTING"; exit 2; }

date -u +"start %Y-%m-%dT%H:%M:%SZ"
fail=0; ok=0; skip=0

run_lev() {  # suite dataset
  local suite="$1" ds="$2" enc target
  enc="$NEW/encodings/$suite/${ds}__${REP}.npz"
  target="$OUT/$suite/${ds}__${REP}__levenshtein.npz"
  [ -f "$enc" ] || { echo "    [skip] $suite/$ds (no encoding)"; skip=$((skip+1)); return; }
  [ -s "$target" ] && { skip=$((skip+1)); echo "    [skip] $suite/$ds lev (exists)"; return; }
  if "$PY" -m benchmarks.real_data.eval_distance.distance_runner \
       --encodings "$enc" --metric levenshtein --out "$OUT/$suite" \
       --n-chunks 1 --chunk-index 0 --jobs "$JOBS" --suite "$suite" >/dev/null 2>&1 \
     && [ -s "$target" ]; then
    ok=$((ok+1)); echo "    [ok] $suite/$ds lev"
    rm -f "$OUT/$suite/${ds}__${REP}__levenshtein.shard"*.npz
  else
    fail=$((fail+1)); echo "    [FAIL] $suite/$ds lev"
  fi
}

run_null() {  # suite dataset
  local suite="$1" ds="$2" enc target
  enc="$NEW/encodings/$suite/${ds}__${REP}.npz"
  target="$OUT/$suite/${ds}__${REP}__size_null.npz"
  [ -f "$enc" ] || { skip=$((skip+1)); return; }
  [ -s "$target" ] && { skip=$((skip+1)); return; }
  if "$PY" -m benchmarks.real_data.eval_distance.size_null \
       --encodings "$enc" --out "$OUT/$suite" --suite "$suite" >/dev/null 2>&1 \
     && [ -s "$target" ]; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "    [FAIL] $suite/$ds size_null"
  fi
}

# Suite 1 first: small, so a defect surfaces in minutes.
for d in "${SUITE1[@]}"; do echo "--- suite1/$d"; run_lev suite1 "$d"; run_null suite1 "$d"; done
for d in "${SUITE2[@]}"; do echo "--- suite2/$d"; run_lev suite2 "$d"; run_null suite2 "$d"; done

date -u +"end %Y-%m-%dT%H:%M:%SZ"
n1=$(ls -1 "$OUT"/suite1/*__${REP}__*.npz 2>/dev/null | wc -l)
n2=$(ls -1 "$OUT"/suite2/*__${REP}__*.npz 2>/dev/null | wc -l)
echo "=== ok: $ok  skipped: $skip  FAILED: $fail ==="
echo "=== new suite1 cells: $n1 (expect 10 = 5 x (lev + null)) ==="
echo "=== new suite2 cells: $n2 (expect 20 = 10 x (lev + null)) ==="
echo "DONE_MARKER rc=$fail ok=$ok skip=$skip suite1=$n1 suite2=$n2"
} 2>&1 | tee "$LOG"

exit 0
