#!/usr/bin/env bash
# Curate T-06's results into the report tree the completed/ archive expects.
#
# ORCHESTRATION ONLY. This copies finished artifacts; it computes nothing. Run
# it after run_f2.sh reports DONE_MARKER rc=0.
#
# The layout matches T-05-bounded-ged and T-27-ged-bound-bakeoff:
#
#   results/reports/T-06-full-recompute/
#       REPORT.md        the headline document (DECISION_SUMMARY.md, promoted)
#       PROVENANCE.md    which run produced this -- GENERATED, never hand-written,
#                        because a provenance page maintained by hand drifts from
#                        the run it claims to describe and a later reader cannot tell
#       figures/         pdf AND png, as T-27 ships both
#       data/            the analysis JSONs a reader needs
#
# 🔴 COPY, NEVER MOVE. Nothing under data/source/T06/ is relocated or deleted:
# the analysis modules resolve paths there and a later rerun must still work.
# The report tree is a curated view, not a relocation.
#
# 🔴 encodings/ (9 MB) and distances/ (518 MB) STAY under data/source/T06/. They
# are source and intermediate data, not results. If this tree ever exceeds
# single-digit MB, a distance matrix has been copied by mistake and the
# assertion at the end will say so.
set -uo pipefail

PY=${PY:-/home/mpascual/.conda/envs/isalgraph-cpp/bin/python}
DATA=${DATA:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data}
ROOT=${ROOT:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports}
REPO=${REPO:-/home/mpascual/research/code/IsalGraph-T06}
T06="$DATA/source/T06"
OUT="$ROOT/T-06-full-recompute"

cd "$REPO" || exit 1
mkdir -p "$OUT/data" "$OUT/figures" || exit 1
fail=0
echo "=== T-06 archive -> $OUT ==="

# --- the analysis JSONs a reader needs -------------------------------------
for f in \
  "$T06/families/family_F0.json" "$T06/families/family_F1.json" \
  "$T06/families/family_F2.json" "$T06/families/rho_table.json" \
  "$T06/claim_a_strata.json" "$T06/claim_a_suite1.json" "$T06/claim_a_suite2.json" \
  "$T06/censoring.json" "$T06/ladder.json" "$T06/ladder_suite1.json" \
  "$T06/size_profile.json" "$T06/size_profile_censoring_confound.json" \
  "$T06/completion_rates.json" "$T06/encodings/manifest.json"
do
  if [ -f "$f" ]; then cp -p "$f" "$OUT/data/"; else echo "!!! missing $f"; fail=$((fail + 1)); fi
done

# The whole gates/ tree, as its own subdirectory so the three files stay together.
if [ -d "$T06/gates" ]; then cp -rp "$T06/gates" "$OUT/data/"; else
  echo "!!! missing $T06/gates"; fail=$((fail + 1)); fi

# --- figures, both formats, as T-27 ships both -----------------------------
figs=0
for ext in pdf png; do
  for f in "$T06"/figures/size_profile/*."$ext"; do
    [ -f "$f" ] && { cp -p "$f" "$OUT/figures/"; figs=$((figs + 1)); }
  done
done

# --- the headline document and the record of what was rejected -------------
if [ -f "$T06/DECISION_SUMMARY.md" ]; then
  cp -p "$T06/DECISION_SUMMARY.md" "$OUT/REPORT.md"
else
  echo "!!! missing DECISION_SUMMARY.md"; fail=$((fail + 1))
fi

# The framing document records the framings that were measured and REJECTED.
# That record is worth more to whoever writes T-20 than any single JSON here,
# because it says which sentences the data refused rather than which it allowed.
for f in "$REPO/.claude/notes/review/tasks/T-06-FRAMING.md" \
         "$REPO/.claude/notes/review/tasks/T-06-design.md"; do
  [ -f "$f" ] && cp -p "$f" "$OUT/"
done

# --- provenance, generated so it cannot drift from the run it describes -----
"$PY" -m benchmarks.real_data.eval_stats.t06_f2 \
  --provenance "$OUT/PROVENANCE.md" --out-dir "$T06/families" || fail=$((fail + 1))

# --- verification ----------------------------------------------------------
n_data=$(ls -1 "$OUT/data"/*.json 2>/dev/null | wc -l)
n_gates=$(ls -1 "$OUT/data/gates"/*.json 2>/dev/null | wc -l)
size_mb=$(du -sm "$OUT" | cut -f1)
echo "=== data JSONs: $n_data | gates: $n_gates | figures: $figs | size: ${size_mb} MB ==="

[ -s "$OUT/PROVENANCE.md" ] || { echo "!!! PROVENANCE.md missing or empty"; fail=$((fail + 1)); }
[ "$figs" -eq 6 ] || { echo "!!! expected 6 figures (3 pdf + 3 png), got $figs"; fail=$((fail + 1)); }
[ "$n_gates" -eq 3 ] || { echo "!!! expected 3 gate JSONs, got $n_gates"; fail=$((fail + 1)); }
# A distance matrix copied here by mistake is the failure this guards against:
# distances/ alone is 518 MB, so a single stray file dwarfs the whole report.
[ "$size_mb" -lt 100 ] || {
  echo "!!! report tree is ${size_mb} MB -- source data has been copied by mistake"
  fail=$((fail + 1)); }
# Nothing must have left the source tree.
for d in encodings distances; do
  [ -d "$T06/$d" ] || { echo "!!! $T06/$d is GONE -- copy, never move"; fail=$((fail + 1)); }
done

echo "DONE_MARKER rc=$fail data=$n_data gates=$n_gates figures=$figs size_mb=$size_mb"
[ "$fail" -eq 0 ] || exit 1
