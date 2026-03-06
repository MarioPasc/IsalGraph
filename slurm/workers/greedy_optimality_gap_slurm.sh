#!/usr/bin/env bash
# SLURM compute worker for greedy_optimality_gap benchmark
set -euo pipefail

echo "=== Greedy Optimality Gap: SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-1}"
echo "Start:  $(date)"

CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

# ISALGRAPH_REPO_DIR is passed via --export from launch.sh
REPO_DIR="${ISALGRAPH_REPO_DIR:?ERROR: ISALGRAPH_REPO_DIR not set. Run via launch.sh or set manually.}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

RESULTS_DIR=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    print(yaml.safe_load(f)['results_dir'])
")
BENCH_CFG=$(python3 -c "
import yaml, json, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg['benchmarks']['greedy_optimality_gap'], sys.stdout)
")

SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")
N_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
OUT_DIR="${RESULTS_DIR}/greedy_optimality_gap"

mkdir -p "$OUT_DIR"

echo "Config: seed=$SEED"
echo "Workers: $N_WORKERS, Output: $OUT_DIR"

python -m benchmarks.greedy_optimality_gap.greedy_optimality_gap \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --mode picasso \
    --n-workers "$N_WORKERS" \
    --csv --plot --table

echo "Finished: $(date)"
