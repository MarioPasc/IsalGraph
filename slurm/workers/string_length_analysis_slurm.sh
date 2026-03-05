#!/usr/bin/env bash
# SLURM compute worker for string_length_analysis benchmark
set -euo pipefail

echo "=== String Length Analysis: SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-1}"
echo "Start:  $(date)"

CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_DIR"

RESULTS_DIR=$(python3 -c "
import yaml
with open('slurm/config.yaml') as f:
    print(yaml.safe_load(f)['results_dir'])
")
BENCH_CFG=$(python3 -c "
import yaml, json, sys
with open('slurm/config.yaml') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg['benchmarks']['string_length_analysis'], sys.stdout)
")

MAX_NODES=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('max_nodes', 200))")
SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")
OUT_DIR="${RESULTS_DIR}/string_length_analysis"

mkdir -p "$OUT_DIR"

echo "Config: max_nodes=$MAX_NODES, seed=$SEED"
echo "Output: $OUT_DIR"

python -m benchmarks.string_length_analysis.string_length_analysis \
    --max-nodes "$MAX_NODES" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --mode picasso \
    --csv --plot --table

echo "Finished: $(date)"
