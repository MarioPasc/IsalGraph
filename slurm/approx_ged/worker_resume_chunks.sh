#!/usr/bin/env bash
# Resume a LIST of array chunks sequentially, in ONE task.
#
# 🔴 WHY THIS EXISTS. Array 2005391 (ubs/mutagenicity, 19 chunks) finished 16 chunks and
# hit the 12 h wallclock on chunks 5, 9 and 10. The TERM trap in ged_exact_runner flushed
# each one's checkpoint, so those three chunks are 400,000 / 400,000 / 420,000 pairs into
# their 429,410 -- 68,230 pairs remain out of Mutagenicity's 8,158,780, i.e. 0.84 %.
# Their afterok merge (2005392) can therefore never run.
#
# 🔴 WHY ONE TASK AND NOT A 3-TASK ARRAY. At the rate those chunks actually achieved
# (43,200 s / 400,000 pairs = 108 ms/pair) the three residuals are ~53, ~53 and ~17
# minutes. Submitted as an array that is three tasks all under SCBI's two-hour floor --
# the exact pattern soporte@scbi.uma.es wrote to this account about on 2026-08-07. Run
# sequentially in one task it is ~2.05 h, which clears the floor by construction.
#
# 🔴 WHY IT DELEGATES TO worker_range.sh RATHER THAN REIMPLEMENTING IT. That worker
# carries the accumulated environment fixes -- the METHOD_ARGS array that stops
# `--threads 1` word-splitting, the --workers 1 optimum, the shared-storage shard path,
# the checkpoint path convention. A fresh worker would be missing whichever of those
# nobody remembered. Resume identity depends on the chunk's pair-set fingerprint, so the
# ROLE / KEY / N_CHUNKS triple MUST match the original submission exactly or
# _load_checkpoint refuses the checkpoint rather than corrupting the shard.
set -euo pipefail
source "${REPO_DIR:?REPO_DIR must be exported by the launcher}/slurm/approx_ged/_env.sh"

CHUNKS="${CHUNKS:?CHUNKS must be exported, colon-separated (e.g. 5:9:10)}"
START_CUTOFF_S="${START_CUTOFF_S:-16200}"

# Colon-separated, not comma-separated: --export splits its value on commas, so a comma
# inside CHUNKS would truncate the list silently and we would resume a subset without
# any error. The merge would then fail on the missing pairs -- loud, but a wasted job.
IFS=':' read -r -a CHUNK_LIST <<< "${CHUNKS}"

START_TIME=$(date +%s)
echo "=========================================="
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Role/key:   ${ROLE:?}/${KEY:?}  n_chunks=${N_CHUNKS:?}"
echo "Chunks:     ${CHUNK_LIST[*]}"
echo "Cutoff:     ${START_CUTOFF_S}s (first chunk exempt)"
echo "=========================================="

deferred=()
idx=0
for c in "${CHUNK_LIST[@]}"; do
    idx=$((idx + 1))
    ELAPSED=$(( $(date +%s) - START_TIME ))
    # The first chunk is exempt. Without that exemption a task whose start-up alone
    # exceeds the cutoff defers its entire list, a resubmission re-derives the same list
    # and defers it again, and the recovery livelocks.
    if (( idx > 1 && ELAPSED >= START_CUTOFF_S )); then
        echo "DEFERRED: chunk ${c} -- ${ELAPSED}s elapsed >= ${START_CUTOFF_S}s cutoff"
        deferred+=("${c}")
        continue
    fi
    echo "--- resuming chunk ${c} (${idx}/${#CHUNK_LIST[@]}), ${ELAPSED}s elapsed ---"
    SLURM_ARRAY_TASK_ID="${c}" bash "${REPO_DIR}/slurm/approx_ged/worker_range.sh"
    echo "--- chunk ${c} done ---"
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "Finished:   $(date)"
echo "Duration:   $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"

if (( ${#deferred[@]} > 0 )); then
    # Exit non-zero so the afterok merge does NOT run against an incomplete shard set.
    echo "FATAL: ${#deferred[@]} chunk(s) deferred and not computed: ${deferred[*]}" >&2
    echo "Resubmit with CHUNKS=$(IFS=:; echo "${deferred[*]}")" >&2
    exit 3
fi
echo "all ${#CHUNK_LIST[@]} chunks complete"
