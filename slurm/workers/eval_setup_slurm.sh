#!/usr/bin/env bash
# SLURM compute worker for eval_setup benchmark
set -euo pipefail

echo "=== Eval Setup: SLURM Worker ==="
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

BENCH_CFG=$(python3 -c "
import yaml, json, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg['benchmarks']['eval_setup'], sys.stdout)
")

DATA_ROOT=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['data_root'])")
IAM_LETTER_PATH=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['iam_letter_path'])")
N_MAX=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_max', 12))")
SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")
TIMEOUT=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('timeout_per_graph', 600))")
N_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$DATA_ROOT"

echo "Config: data_root=$DATA_ROOT, iam_letter_path=$IAM_LETTER_PATH"
echo "  n_max=$N_MAX, seed=$SEED, timeout=$TIMEOUT, workers=$N_WORKERS"

python -m benchmarks.eval_setup.eval_setup \
    --data-root "$DATA_ROOT" \
    --iam-letter-path "$IAM_LETTER_PATH" \
    --n-max "$N_MAX" \
    --seed "$SEED" \
    --timeout-per-graph "$TIMEOUT" \
    --n-workers "$N_WORKERS" \
    --mode picasso

echo "Finished: $(date)"
