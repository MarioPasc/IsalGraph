#!/usr/bin/env bash
# AIDS exact GED. Serves BOTH stages of signed decision 21:
#
#   AIDS_STAGE=1  the pre-declared stratified sample -- 22,106 pairs over ALL 769 graphs,
#                 a complete 180-graph core block plus a stratified halo and top-up.
#                 This is the REPORTED analysis.
#   AIDS_STAGE=2  the full 295,296-pair census, as an array over contiguous
#                 upper-triangle index ranges, seeded from stage 1 so the overlap is
#                 computed once and agreement on it is asserted at merge.
#
# Supersession is decided on the CALENDAR, not on the values -- see the design note.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

KEY=aids
W="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-64}}"
STAGE="${AIDS_STAGE:?AIDS_STAGE must be 1 or 2}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
CHUNK=$(( TASK + ${CHUNK_OFFSET:-0} ))      # offset the RANGE, never renumber the decode

# Always echo the decoded tuple. A silently wrong decode produces a complete, plausible,
# wrong result set -- the most expensive failure mode available here.
echo "[decode] stage=${STAGE} array_task=${TASK} chunk=${CHUNK}/${N_CHUNKS} workers=${W}"

SHARDS="${OUT_DIR}/shards_${KEY}_s${STAGE}"; mkdir -p "${SHARDS}"
LOCAL_OUT="${MYLOCAL}/out/shards_${KEY}_s${STAGE}"; mkdir -p "${LOCAL_OUT}"
SHARD_NAME=$(printf "%s_s%s_c%04d.npz" "${KEY}" "${STAGE}" "${CHUNK}")

EXTRA=()
if [[ "${STAGE}" == "1" ]]; then
    PAIR_LIST="${OUT_DIR}/${KEY}_stage1_pairs.npz"
    if [[ ! -f "${PAIR_LIST}" ]]; then
        echo "[stage1] building the frozen sample (K=180, q=10, f=30, seed 42)"
        run_py benchmarks.real_data.eval_setup.ged_sampling \
            --input "${DATA_DIR}/${KEY}.npz" \
            --out-pairs "${PAIR_LIST}" \
            --out-report "${OUT_DIR}/${KEY}_stage1_sampling_report.json" \
            -K 180 -q 10 -f 30 --seed 42 --expect-graphs 769
    fi
    EXTRA+=(--pair-list "${PAIR_LIST}")
else
    # Reuse stage 1 rather than recomputing its 22,106 pairs. The seeded values are
    # CARRIED into this shard, not omitted -- that is what gives the merge's
    # "no conflicting values on any k" check something to compare.
    S1="${OUT_DIR}/${KEY}_stage1.npz"
    [[ -f "${S1}" ]] && EXTRA+=(--seed-from "${S1}") && echo "[stage2] seeding from ${S1}"
fi

run_py benchmarks.real_data.eval_setup.ged_exact_runner \
    --input "${DATA_DIR}/${KEY}.npz" \
    --out "${LOCAL_OUT}/${SHARD_NAME}" \
    --backend networkx --cost-model unit \
    --chunk-index "${CHUNK}" --n-chunks "${N_CHUNKS}" \
    --workers "${W}" \
    --timeout-per-pair "${TIMEOUT_PER_PAIR}" \
    --checkpoint-every 2000 \
    --checkpoint "${SHARDS}/${SHARD_NAME%.npz}.ckpt.npz" \
    "${EXTRA[@]}"

# Copy to a per-task temp name in the destination, then rename. cp truncates before it
# writes, so two concurrent array tasks can otherwise leave a short file; a rename
# within one directory is atomic.
cp -a "${LOCAL_OUT}/${SHARD_NAME}" "${SHARDS}/.${SHARD_NAME}.$$" \
    && mv -f "${SHARDS}/.${SHARD_NAME}.$$" "${SHARDS}/${SHARD_NAME}"
echo "[shard] ${SHARDS}/${SHARD_NAME}"

# Stage 1 is NOT merged into a CONTRACT D matrix, deliberately. It computes 22,106 of
# 295,296 pairs, so a full 769x769 matrix would be 92.5% empty -- a shape that invites
# every downstream consumer to treat missing as censored. The CONTRACT C shard already
# carries exactly what stage 1 is for: the pair indices, the values, the certification
# flags and the per-pair wall times the D12 censoring analysis needs. It is also the
# format --seed-from consumes, so stage 2 reuses it without conversion.
if [[ "${STAGE}" == "1" ]]; then
    cp -a "${SHARDS}/${SHARD_NAME}" "${OUT_DIR}/.${KEY}_stage1.npz.$$" \
        && mv -f "${OUT_DIR}/.${KEY}_stage1.npz.$$" "${OUT_DIR}/${KEY}_stage1.npz"
    echo "[stage1] result -> ${OUT_DIR}/${KEY}_stage1.npz (CONTRACT C, sparse by design)"
fi
