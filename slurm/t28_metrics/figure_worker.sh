#!/usr/bin/env bash
# T-28 size profile + one rho-vs-size figure per alternative reference.
#
# ONE profile pass over every reference, then one figure each. The earlier
# WL-only run is why a spectral figure needed a second pass at all: the profile
# is the expensive half and it costs almost nothing to emit every reference from
# it, so it does.
#
# Cheap by construction: representation distances are READ from the T-06 cache
# (T-04a's selections, never recomputed), and --no-bootstrap skips the
# per-stratum interval the figures never read -- figures.aggregate derives its
# own from the Fisher-z mean of rho and n_graphs. Measured: 21 s for all fifteen
# cells that way, against hours with the bootstrap on.
#
# NO #SBATCH HEADER: the launcher supplies every resource flag.
set -uo pipefail

START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
# Colon-separated on the wire; see the launcher for why.
REFERENCE_LIST=$(printf '%s' "${T28_FIG_REFERENCES}" | tr ':' ' ')
echo "References:   ${REFERENCE_LIST}"
echo "Git commit:   $(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

module_loaded=0
for m in miniconda/3 miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>&1 | grep -qiE "(^|/)${m}([[:space:]]|/|$)"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module; using the env's python directly."

PY="${CONDA_PREFIX_DIR}/bin/python"
[ -x "$PY" ] || { echo "[FATAL] no interpreter at $PY"; exit 2; }
cd "${REPO_DIR}" || { echo "[FATAL] no repo at ${REPO_DIR}"; exit 2; }

export PYTHONPATH="${REPO_DIR}"          # REPO_DIR ONLY -- never src-first
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${LOCALSCRATCH:-/tmp}/${USER}/pycache_${SLURM_JOB_ID:-0}"
mkdir -p "${PYTHONPYCACHEPREFIX}"

fatal_root() { echo "[FATAL] $1 unset or missing: '${2:-}'" >&2; exit 2; }
[ -n "${DISTANCES:-}" ]          && [ -d "${DISTANCES}" ]          || fatal_root DISTANCES "${DISTANCES:-}"
[ -n "${GED_ROOT:-}" ]           && [ -d "${GED_ROOT}" ]           || fatal_root GED_ROOT "${GED_ROOT:-}"
[ -n "${APPROX_ROOT:-}" ]        && [ -d "${APPROX_ROOT}" ]        || fatal_root APPROX_ROOT "${APPROX_ROOT:-}"
[ -n "${T28_REFERENCE_ROOT:-}" ] && [ -d "${T28_REFERENCE_ROOT}" ] || fatal_root T28_REFERENCE_ROOT "${T28_REFERENCE_ROOT:-}"
[ -n "${FIG_OUT:-}" ] || fatal_root FIG_OUT "${FIG_OUT:-}"
mkdir -p "${FIG_OUT}"

echo "[roots] distances=${DISTANCES}"
echo "[roots] ged=${GED_ROOT}"
echo "[roots] approx=${APPROX_ROOT}"
echo "[roots] t28=${T28_REFERENCE_ROOT}"

PROFILE="${FIG_OUT}/size_profile_all_references.json"

echo ""
echo "[1/2] size profile -- every reference, cached distances, no per-stratum bootstrap"
"$PY" -m benchmarks.real_data.eval_reference_metrics.size_profile_cached \
    --distances "${DISTANCES}" \
    --ged-root "${GED_ROOT}" \
    --approx-root "${APPROX_ROOT}" \
    --t28-root "${T28_REFERENCE_ROOT}" \
    --no-bootstrap \
    --out "${PROFILE}" || { echo "[FATAL] size profile failed"; exit 2; }

# An empty profile still writes valid JSON, and every figure would then draw an
# empty axes and exit 0. That is the silent no-op this guard exists to catch.
N_ROWS=$("$PY" -c "import json;print(len(json.load(open('${PROFILE}'))['rows']))" 2>/dev/null || echo 0)
echo "[input] ${N_ROWS} stratum rows"
[ "${N_ROWS}" -gt 0 ] || { echo "[FATAL] profile is empty; every figure would be blank" >&2; exit 2; }

echo ""
echo "[2/2] one figure per reference"
ok=0; fail=0
for ref in ${REFERENCE_LIST}; do
    case "${ref}" in
        wl)             label="WL kernel";              degen="wl_subtree" ;;
        spectral)       label="spectral (norm. L)";     degen="" ;;
        spectral_comb)  label="spectral (comb. L)";     degen="" ;;
        spectral_adj)   label="spectral (adjacency)";   degen="" ;;
        spectral_esd)   label="spectral ESD";           degen="" ;;
        *)              label="${ref}";                 degen="" ;;
    esac
    # Every reference is checked against the profile first: a name with no rows
    # would otherwise produce an empty axes and a zero exit status.
    have=$("$PY" -c "
import json,sys
rows=json.load(open('${PROFILE}'))['rows']
print(sum(1 for r in rows if r['reference']=='${ref}' and r['rho'] is not None))
" 2>/dev/null || echo 0)
    if [ "${have}" -lt 1 ]; then
        echo "  [SKIP] ${ref}: no rows in the profile"
        fail=$(( fail + 1 )); continue
    fi
    if "$PY" -m benchmarks.real_data.eval_size_profile.figures \
        --profile "${PROFILE}" --out-dir "${FIG_OUT}" \
        --reference "${ref}" --reference-label "${label}" \
        ${degen:+--degenerate "${degen}"} \
        --stem "fig1_rho_vs_size_${ref}" >/dev/null 2>&1; then
        echo "  [ok]   ${ref}: ${have} rows -> fig1_rho_vs_size_${ref}.{pdf,png}"
        ok=$(( ok + 1 ))
    else
        echo "  [FAIL] ${ref}"
        fail=$(( fail + 1 ))
    fi
done

echo ""
ls -la "${FIG_OUT}"/fig1_rho_vs_size_*.pdf 2>/dev/null

END_TIME=$(date +%s); ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $(( ELAPSED / 3600 ))h $(( (ELAPSED / 60) % 60 ))m $(( ELAPSED % 60 ))s"
echo "DONE_MARKER rows=${N_ROWS} figures_ok=${ok} figures_failed=${fail}"
[ "${fail}" -eq 0 ] || exit 1
