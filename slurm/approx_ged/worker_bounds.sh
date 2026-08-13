#!/usr/bin/env bash
# One role (or a --group of roles, run sequentially) over all ten Suite-2 datasets.
# Roles lb / ub / ubs. Role ubt runs the subsample and has its own worker.
#
# ONE JOB, NOT AN ARRAY. Letter LOW is ~90 core-seconds; an array over datasets would
# make nine of ten tasks minutes long, which is the 12,600-task pattern SCBI wrote to this
# account about (2026-08-07). Grouping short units into one submission is exactly what
# they asked for, and it is makespan-neutral. See T-05-design §5.
#
# SLURM copies the batch script into a per-job spool directory, so BASH_SOURCE resolves to
# /var/spool/slurmd/... and a sibling path finds nothing. Source from REPO_DIR, which the
# launcher exports. Keep set -euo here too: _env.sh sets it, so a failure to source it
# would otherwise silently disable it as well.
set -euo pipefail
source "${REPO_DIR:?REPO_DIR must be exported by the launcher}/slurm/approx_ged/_env.sh"

W="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
IFS=':' read -r -a ROLE_LIST <<< "${ROLES:?ROLES must be exported by the launcher (colon-separated)}"
PROBE_PAIR_LIST="${DATA_DIR}/probe_pairs.npz"
REPORT="${MYLOCAL}/out/run_report_${SLURM_JOB_ID:-local}.json"

for R in "${ROLE_LIST[@]}"; do assert_known_role "${R}"; done

# ---------------------------------------------------------------- probe
# T-05-design §5: the rate is MEASURED, on the hardware that does the work, inside the
# same job -- a separate probe job would itself violate the two-hour floor. The launcher
# consumes the probe.json this writes when sizing the NEXT submission.
#
# The stratified 3,000-pair list (every dataset, every n decile) is the sampler's output.
# Without it the fallback probe is a contiguous first chunk of one dataset, which is NOT
# stratified: contiguous upper-triangle indices over a size-ordered export oversample the
# small-n corner, and per-pair cost scales ~max(n1,n2)^3. The fallback rate is therefore
# biased LOW and is logged as such rather than quietly used.
probe_role() {
    local role="$1" t0 t1 elapsed npairs rate
    local method="${ROLE_METHOD[$role]}" options="${ROLE_OPTIONS[$role]}"
    local compute="${ROLE_COMPUTE[$role]}"
    local probe_out="${MYLOCAL}/probe_${role}.npz"
    local args=()

    if [[ -f "${PROBE_PAIR_LIST}" ]]; then
        echo "[probe:${role}] stratified list ${PROBE_PAIR_LIST}"
        args=(--input "${DATA_DIR}/mutagenicity.npz" --pair-list "${PROBE_PAIR_LIST}")
        npairs="${PROBE_PAIRS:-3000}"
    else
        echo "[probe:${role}] WARNING: no ${PROBE_PAIR_LIST}."
        echo "[probe:${role}] Falling back to a contiguous first chunk of grec. That sample"
        echo "[probe:${role}] is NOT stratified by n, so the rate it yields is biased LOW."
        echo "[probe:${role}] Do not feed it to --probe-json as if it were the design's probe."
        args=(--input "${DATA_DIR}/grec.npz" --chunk-index 0 --n-chunks 200)
        npairs=$(( 210925 / 200 ))
    fi

    t0=$(date +%s)
    if [[ "${compute}" == "lb" ]]; then
        run_py benchmarks.real_data.eval_setup.ged_exact_runner \
            "${args[@]}" --out "${probe_out}" \
            --backend gedlib --cost-model unit --compute lb --role "${role}" \
            --lb-method "${method}" --lb-options "${options}" \
            --workers 1 || { echo "[probe:${role}] FAILED -- production is not attempted" >&2; return 1; }
    else
        run_py benchmarks.real_data.eval_setup.ged_exact_runner \
            "${args[@]}" --out "${probe_out}" \
            --backend gedlib --cost-model unit --compute ub --role "${role}" \
            --ub-method "${method}" --ub-options "${options}" \
            --workers 1 || { echo "[probe:${role}] FAILED -- production is not attempted" >&2; return 1; }
    fi
    t1=$(date +%s)
    elapsed=$(( t1 - t0 ))
    rate=$("${PY}" -c "print(f'{${elapsed}/max(1,${npairs}):.9f}')")
    echo "[probe:${role}] ${npairs} pairs in ${elapsed}s -> ${rate} s/pair on 1 core"
    echo "[probe:${role}] projected wall on ${W} cores for this role's full scope:"
    "${PY}" -c "
n = 21710892
print(f'[probe:${role}]   {n*${rate}/${W}/3600:.2f} h  (floor is 2.00 h)')
if n*${rate}/${W} < 7200:
    print('[probe:${role}]   BELOW THE FLOOR at this core count. The job continues -- the')
    print('[probe:${role}]   launcher is where a submission is refused -- but record this:')
    print('[probe:${role}]   the next submission of this role should use fewer cores or be')
    print('[probe:${role}]   grouped with an adjacent role (--group).')"
    rm -f "${probe_out}" 2>/dev/null || true
    PROBE_RATES+=("\"${role}\": ${rate}")
}

# ---------------------------------------------------------------- production
produce_role() {
    local role="$1"
    local method="${ROLE_METHOD[$role]}" options="${ROLE_OPTIONS[$role]}"
    local compute="${ROLE_COMPUTE[$role]}" outdir="${ROLE_OUTDIR[$role]}"
    local dest="${MYLOCAL}/out/${outdir}"
    mkdir -p "${dest}"

    echo "############ role=${role} method=${method} options='${options}' -> ${outdir}/"

    local key n shards
    for key in "${DATASET_ORDER[@]}"; do
        n="${NGRAPHS[$key]}"
        echo "=== ${role} / ${key} (${n} graphs) ==="
        shards="${MYLOCAL}/shards_${role}_${key}"
        mkdir -p "${shards}"

        # --chunk-index 0 --n-chunks 1: the chunking exists for RESUMABILITY INSIDE THIS
        # TASK (the checkpoint below), not for fan-out. T-05-design §5.
        if [[ "${compute}" == "lb" ]]; then
            run_py benchmarks.real_data.eval_setup.ged_exact_runner \
                --input "${DATA_DIR}/${key}.npz" \
                --out "${shards}/${key}_c0000.npz" \
                --backend gedlib --cost-model unit \
                --compute lb --role "${role}" \
                --lb-method "${method}" --lb-options "${options}" \
                --chunk-index 0 --n-chunks 1 \
                --workers "${W}" \
                --checkpoint-every "${CHECKPOINT_EVERY:-2000}" \
                --checkpoint "${shards}/${key}.ckpt.npz"
        else
            run_py benchmarks.real_data.eval_setup.ged_exact_runner \
                --input "${DATA_DIR}/${key}.npz" \
                --out "${shards}/${key}_c0000.npz" \
                --backend gedlib --cost-model unit \
                --compute ub --role "${role}" \
                --ub-method "${method}" --ub-options "${options}" \
                --chunk-index 0 --n-chunks 1 \
                --workers "${W}" \
                --checkpoint-every "${CHECKPOINT_EVERY:-2000}" \
                --checkpoint "${shards}/${key}.ckpt.npz"
        fi

        # --delete-shards is honoured by the merge only AFTER its structural gate passes
        # (CONTRACTS §6.2, §7). Passing it here is therefore not "delete the evidence";
        # a gate failure raises MergeError, set -e aborts, and the shards survive for
        # diagnosis. Do not add an `rm` of your own after this line.
        run_py benchmarks.real_data.eval_setup.ged_merge_shards \
            --shards "${shards}" --key "${key}" --n-graphs "${n}" \
            --input "${DATA_DIR}/${key}.npz" \
            --out "${dest}/${key}.npz" \
            --ged-from "${compute}" --role "${role}" --seconds-role "${role}" \
            --delete-shards

        # Per-dataset copy-back bounds the loss if the node dies later in the loop; the
        # EXIT trap's whole-tree mirror is what guarantees completeness. Copy to a
        # per-task temp name then rename -- cp truncates before it writes, so a reader
        # of a shared destination can otherwise see a half-written file.
        mkdir -p "${OUT_DIR}/${outdir}"
        cp -a "${dest}/${key}.npz" "${OUT_DIR}/${outdir}/${key}.npz.part.$$" && \
            mv -f "${OUT_DIR}/${outdir}/${key}.npz.part.$$" "${OUT_DIR}/${outdir}/${key}.npz"
        echo "=== ${role}/${key} done at $(( $(date +%s) - START_TIME ))s ==="
    done
}

PROBE_RATES=()
for R in "${ROLE_LIST[@]}"; do probe_role "${R}"; done
for R in "${ROLE_LIST[@]}"; do produce_role "${R}"; done

# ---------------------------------------------------------------- projected vs realised
# T-27 limitation 3 says its Suite-2 projections are LOWER bounds on true cost. This is
# the measurement that tests that claim, and it costs nothing.
REALISED=$(( $(date +%s) - START_TIME ))
{
    printf '{\n'
    printf '  "job_id": "%s",\n' "${SLURM_JOB_ID:-local}"
    printf '  "roles": "%s",\n' "${ROLES}"
    printf '  "cores": %s,\n' "${W}"
    printf '  "node": "%s",\n' "$(hostname)"
    printf '  "cpu_model": "%s",\n' "$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
    printf '  "projected_wall_seconds": %s,\n' "${PROJ_WALL_SECONDS:-null}"
    printf '  "projected_core_seconds": %s,\n' "${PROJ_CORE_SECONDS:-null}"
    printf '  "sizing_evidence": "%s",\n' "${SIZING_EVIDENCE:-unknown}"
    printf '  "realised_wall_seconds": %s,\n' "${REALISED}"
    printf '  "realised_core_seconds": %s,\n' "$(( REALISED * W ))"
    printf '  "floor_seconds": 7200,\n'
    printf '  "cleared_floor": %s,\n' "$( (( REALISED >= 7200 )) && echo true || echo false )"
    printf '  "probe_seconds_per_pair": {%s},\n' "$(IFS=,; echo "${PROBE_RATES[*]:-}")"
    printf '  "probe_stratified": %s,\n' "$( [[ -f "${PROBE_PAIR_LIST}" ]] && echo true || echo false )"
    printf '  "finished_utc": "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '}\n'
} > "${REPORT}"
echo "[report] ${REPORT}"
cat "${REPORT}"
