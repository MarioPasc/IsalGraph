#!/usr/bin/env bash
# Letter LOW/MED/HIGH + LINUX, all pairs. The end-to-end proof on real cluster data
# before AIDS is submitted.
#
# The four datasets run in ONE task deliberately. LINUX alone is 3,916 pairs -- about
# 18 minutes on 8 cores -- and would violate SCBI's two-hour floor. Grouping short units
# into one submission is exactly what SCBI asked for (2026-08-07) and it is
# makespan-neutral: the work is the same, the scheduler places it once.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

W="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
declare -A NGRAPHS=( [iam_letter_low]=1180 [iam_letter_med]=1253 [iam_letter_high]=2059 [linux]=89 )

# Cheapest first, so a wallclock overrun costs the largest dataset rather than all four.
for KEY in linux iam_letter_low iam_letter_med iam_letter_high; do
    N=${NGRAPHS[$KEY]}
    echo "=== ${KEY} (${N} graphs) ==="
    SHARDS="${MYLOCAL}/out/shards_${KEY}"; mkdir -p "${SHARDS}"

    run_py benchmarks.real_data.eval_setup.ged_exact_runner \
        --input "${DATA_DIR}/${KEY}.npz" \
        --out "${SHARDS}/${KEY}_c0000.npz" \
        --backend networkx --cost-model unit \
        --chunk-index 0 --n-chunks 1 \
        --workers "${W}" \
        --timeout-per-pair "${TIMEOUT_PER_PAIR}" \
        --checkpoint-every 2000 \
        --checkpoint "${SHARDS}/${KEY}.ckpt.npz"

    run_py benchmarks.real_data.eval_setup.ged_merge_shards \
        --shards "${SHARDS}" --key "${KEY}" --n-graphs "${N}" \
        --input "${DATA_DIR}/${KEY}.npz" \
        --out "${MYLOCAL}/out/${KEY}.npz" \
        --delete-shards

    # Per-dataset copy-back bounds the loss if the node dies later in the loop; the
    # EXIT trap's whole-tree mirror is what guarantees completeness.
    cp -a "${MYLOCAL}/out/${KEY}.npz" "${OUT_DIR}/${KEY}.npz.part" && \
        mv -f "${OUT_DIR}/${KEY}.npz.part" "${OUT_DIR}/${KEY}.npz"
    echo "=== ${KEY} done at $(( $(date +%s) - START_TIME ))s ==="
done
