#!/usr/bin/env bash
# SLURM compute worker for eval_embedding benchmark
set -euo pipefail

echo "=== Eval Embedding: SLURM Worker ==="
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
json.dump(cfg['benchmarks']['eval_embedding'], sys.stdout)
")

DATA_ROOT=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['data_root'])")
OUTPUT_DIR=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_dir'])")
DIMENSIONS=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('dimensions', '2,3,5,10'))")
N_PROCRUSTES_PERMS=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_procrustes_perms', 9999))")
SMACOF_MAX_ITER=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('smacof_max_iter', 300))")
SMACOF_N_INIT=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('smacof_n_init', 4))")
SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")

mkdir -p "$OUTPUT_DIR"

echo "Config: data_root=$DATA_ROOT, output_dir=$OUTPUT_DIR"
echo "  dimensions=$DIMENSIONS, n_procrustes_perms=$N_PROCRUSTES_PERMS"
echo "  smacof_max_iter=$SMACOF_MAX_ITER, smacof_n_init=$SMACOF_N_INIT, seed=$SEED"

python -m benchmarks.eval_embedding.eval_embedding \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --dimensions "$DIMENSIONS" \
    --n-procrustes-perms "$N_PROCRUSTES_PERMS" \
    --smacof-max-iter "$SMACOF_MAX_ITER" \
    --smacof-n-init "$SMACOF_N_INIT" \
    --seed "$SEED" \
    --mode picasso \
    --plot --table

echo "Finished: $(date)"
