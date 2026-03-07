#!/usr/bin/env bash
# SLURM compute worker for eval_correlation benchmark
set -euo pipefail

echo "=== Eval Correlation: SLURM Worker ==="
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
json.dump(cfg['benchmarks']['eval_correlation'], sys.stdout)
")

DATA_ROOT=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['data_root'])")
OUTPUT_DIR=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_dir'])")
N_BOOTSTRAP=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_bootstrap', 10000))")
N_PERMUTATIONS=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_permutations', 9999))")
SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")

mkdir -p "$OUTPUT_DIR"

echo "Config: data_root=$DATA_ROOT, output_dir=$OUTPUT_DIR"
echo "  n_bootstrap=$N_BOOTSTRAP, n_permutations=$N_PERMUTATIONS, seed=$SEED"

python -m benchmarks.eval_correlation.eval_correlation \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --n-permutations "$N_PERMUTATIONS" \
    --seed "$SEED" \
    --mode picasso \
    --csv --plot --table

echo "Finished: $(date)"
