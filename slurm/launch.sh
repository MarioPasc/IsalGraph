#!/usr/bin/env bash
# =============================================================================
# IsalGraph Benchmark Launcher for Picasso HPC
# =============================================================================
#
# Master executor that reads slurm/config.yaml and dispatches SLURM jobs
# for each enabled benchmark.
#
# Usage:
#   bash slurm/launch.sh              # Submit all enabled benchmarks
#   bash slurm/launch.sh --dry-run    # Print sbatch commands without submitting
#   bash slurm/launch.sh --benchmark random_roundtrip  # Submit single benchmark
#
set -euo pipefail

CONDA_ENV_NAME="isalgraph"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=false
SINGLE_BENCHMARK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --benchmark)
            SINGLE_BENCHMARK="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Parse config.yaml using Python (available on Picasso login nodes)
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi
parse_config() {
    python3 -c "
import yaml, json, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg, sys.stdout)
"
}

CONFIG_JSON=$(parse_config)

REPO_DIR=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['repo_dir'])")
RESULTS_DIR=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['results_dir'])")
CONDA_ENV=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['conda_env'])")
CONSTRAINT=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['constraint'])")
ACCOUNT=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['account'])")

echo "=============================================="
echo "IsalGraph Benchmark Launcher"
echo "=============================================="
echo "Repo:       ${REPO_DIR}"
echo "Results:    ${RESULTS_DIR}"
echo "Conda env:  ${CONDA_ENV}"
echo "Constraint: ${CONSTRAINT}"
echo "Account:    ${ACCOUNT}"
echo "Dry run:    ${DRY_RUN}"
echo ""

# ---------------------------------------------------------------------------
# Get benchmark config as JSON
# ---------------------------------------------------------------------------
get_benchmark_config() {
    local bench_name="$1"
    echo "$CONFIG_JSON" | python3 -c "
import json, sys
cfg = json.load(sys.stdin)
bench = cfg['benchmarks'].get('${bench_name}', {})
json.dump(bench, sys.stdout)
"
}

# ---------------------------------------------------------------------------
# Submit a benchmark job
# ---------------------------------------------------------------------------
submit_benchmark() {
    local bench_name="$1"
    local bench_config
    bench_config=$(get_benchmark_config "$bench_name")

    local enabled
    enabled=$(echo "$bench_config" | python3 -c "import json,sys; print(json.load(sys.stdin).get('enabled', False))")
    if [[ "$enabled" != "True" ]]; then
        echo "[SKIP] ${bench_name}: disabled in config"
        return
    fi

    local time_limit cpus mem_gb
    time_limit=$(echo "$bench_config" | python3 -c "import json,sys; print(json.load(sys.stdin)['time_limit'])")
    cpus=$(echo "$bench_config" | python3 -c "import json,sys; print(json.load(sys.stdin)['cpus'])")
    mem_gb=$(echo "$bench_config" | python3 -c "import json,sys; print(json.load(sys.stdin)['mem_gb'])")

    local out_dir="${RESULTS_DIR}/${bench_name}"
    local worker_script="${SCRIPT_DIR}/workers/${bench_name}_slurm.sh"

    if [[ ! -f "$worker_script" ]]; then
        echo "[ERROR] Worker script not found: ${worker_script}"
        return 1
    fi

    # Create output directory
    mkdir -p "${out_dir}"

    local sbatch_cmd="sbatch \
        --job-name=isalgraph_${bench_name} \
        --output=${out_dir}/slurm_%j.out \
        --error=${out_dir}/slurm_%j.err \
        --time=${time_limit} \
        --cpus-per-task=${cpus} \
        --mem=${mem_gb}G \
        --constraint=${CONSTRAINT} \
        --account=${ACCOUNT} \
        --chdir=${REPO_DIR} \
        --export=ALL,ISALGRAPH_REPO_DIR=${REPO_DIR} \
        ${worker_script}"

    echo "[${bench_name}]"
    echo "  Time:  ${time_limit}"
    echo "  CPUs:  ${cpus}"
    echo "  Mem:   ${mem_gb}G"
    echo "  Out:   ${out_dir}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would execute: ${sbatch_cmd}"
    else
        echo "  Submitting..."
        eval "${sbatch_cmd}"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BENCHMARKS=("random_roundtrip" "canonical_invariance" "string_length_analysis" "levenshtein_vs_ged")

if [[ -n "$SINGLE_BENCHMARK" ]]; then
    submit_benchmark "$SINGLE_BENCHMARK"
else
    for bench in "${BENCHMARKS[@]}"; do
        submit_benchmark "$bench"
    done
fi

echo "Done."
