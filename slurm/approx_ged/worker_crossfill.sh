#!/usr/bin/env bash
# Cross-fill the three role campaigns into each other, then run the independent gates.
#
# CONTRACTS §4.2: one step opens LB/{key}.npz, UB/{key}.npz and UB_SENSITIVITY/{key}.npz,
# writes the same lb_matrix / ub_matrix / certified_mask into all three, and rewrites them
# atomically. ged_matrix and seconds_matrix are NEVER touched by cross-fill.
#
# Why certified_mask is produced HERE and not by a backend (CONTRACTS §4.1):
# GedlibBackend.pair() returns certified=False always, deliberately, after
# ANCHOR_AWARE_GED was measured issuing a FALSE optimality certificate (T-03 amendment 2).
# The mask is not a self-report. It is the derived statement "a proven lower bound of k and
# an exhibited edit path of cost k together prove GED = k" -- two independent proofs
# meeting, computed by a separate step over two separate campaigns. Do not source it from
# any backend field.
#
# SLURM copies the batch script into a per-job spool directory, so BASH_SOURCE resolves to
# /var/spool/slurmd/... . Source from REPO_DIR, which the launcher exports.
set -euo pipefail
source "${REPO_DIR:?REPO_DIR must be exported by the launcher}/slurm/approx_ged/_env.sh"

LB_DIR="${OUT_DIR}/LB"
UB_DIR="${OUT_DIR}/UB"
UBS_DIR="${OUT_DIR}/UB_SENSITIVITY"
GATES_DIR="${MYLOCAL}/out/gates"
mkdir -p "${GATES_DIR}"

# Refuse to cross-fill a partial campaign. certified_mask is a PROOF; derived from an
# incomplete lower bound it is not a missing value but a FALSE one, and it would be
# indistinguishable from a correct one downstream.
MISSING=0
for d in "${LB_DIR}" "${UB_DIR}" "${UBS_DIR}"; do
    for key in "${DATASET_ORDER[@]}"; do
        [[ -f "${d}/${key}.npz" ]] || { echo "[crossfill] MISSING ${d}/${key}.npz" >&2; MISSING=1; }
    done
done
if (( MISSING )); then
    echo "FATAL: the three role campaigns are not all complete. Cross-fill would write a" >&2
    echo "       certified_mask derived from an incomplete lower bound -- a false proof," >&2
    echo "       not a missing value. Re-run the failed role first." >&2
    exit 4
fi

echo "[crossfill] all 30 role files present"
run_py benchmarks.real_data.eval_setup.approx_ged_crossfill \
    --lb-dir "${LB_DIR}" --ub-dir "${UB_DIR}" --ubs-dir "${UBS_DIR}" \
    --datasets "$(IFS=,; echo "${DATASET_ORDER[*]}")"

# ---------------------------------------------------------------- the independent gates
# These duplicate the merge's own structural gate ON PURPOSE: an independent reader of the
# finished file, written against CONTRACTS rather than against the writer. They are the
# reason a number from this campaign may be printed in the paper.
#
# G2 needs T-27's recorded cells and G3 needs T-03's exact census. Both live on the
# workstation, not here, so they run there. What runs on the cluster is G4-verify (which
# needs only the written files) and lb-consistency (which needs GEDLIB, and GEDLIB exists
# only here). This split is the same one slurm/exact_ged/worker_gates.sh makes.
RC=0
for G in G4 lb-consistency; do
    echo "--- gate ${G} ---"
    if ! run_py benchmarks.real_data.eval_setup.approx_ged_gates \
        --gate "${G}" \
        --lb-dir "${LB_DIR}" --ub-dir "${UB_DIR}" --ubs-dir "${UBS_DIR}" \
        --input-dir "${DATA_DIR}" \
        --datasets "$(IFS=,; echo "${DATASET_ORDER[*]}")" \
        --seed 42 --sample-size 5000 --workers "${SLURM_CPUS_PER_TASK:-4}" \
        --out "${GATES_DIR}"; then
        echo "[crossfill] gate ${G} FAILED"
        RC=1
    fi
done

echo
echo "[crossfill] G2 and G3 are NOT run here -- they need T-27's cells and T-03's exact"
echo "[crossfill] census, which live on the workstation. Run them there before quoting a"
echo "[crossfill] single number from this campaign:"
echo "  python -m benchmarks.real_data.eval_setup.approx_ged_gates --gate G2 --gate G3 ..."
echo
echo "[crossfill] file-count check -- the whole programme should be ~35 files"
find "${OUT_DIR}" -type f | wc -l
echo "[crossfill] exit=${RC}"
exit ${RC}
