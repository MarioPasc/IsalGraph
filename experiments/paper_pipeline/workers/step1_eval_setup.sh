#!/usr/bin/env bash
# =============================================================================
# Step 1: Eval Setup — Compute GED, canonical strings, Levenshtein matrices
# =============================================================================
set -euo pipefail

echo "=== Paper Pipeline Step 1: Eval Setup ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-1}"
echo "Start:  $(date)"

# --- Environment ---
RUN_DIR="${ISALGRAPH_RUN_DIR:?ERROR: ISALGRAPH_RUN_DIR not set}"
REPO_DIR="${ISALGRAPH_REPO_DIR:?ERROR: ISALGRAPH_REPO_DIR not set}"
CONFIG="${RUN_DIR}/config.yaml"
cd "$REPO_DIR"

CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

# --- Parse config ---
STEP_CFG=$(python3 -c "
import yaml, json, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
out = cfg['steps']['eval_setup']
out['seed'] = cfg['experiment']['seed']
out['source_dir'] = cfg['paths']['source_dir']
json.dump(out, sys.stdout)
")

N_MAX=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_max'])")
SEED=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['seed'])")
TIMEOUT=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['timeout_per_graph'])")
SOURCE_DIR=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['source_dir'])")
N_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

DATA_ROOT="${RUN_DIR}/data"
mkdir -p "$DATA_ROOT"

echo "Config: data_root=$DATA_ROOT, source_dir=$SOURCE_DIR"
echo "  n_max=$N_MAX, seed=$SEED, timeout=$TIMEOUT, workers=$N_WORKERS"

python -m benchmarks.eval_setup.eval_setup \
    --data-root "$DATA_ROOT" \
    --source-dir "$SOURCE_DIR" \
    --n-max "$N_MAX" \
    --seed "$SEED" \
    --timeout-per-graph "$TIMEOUT" \
    --n-workers "$N_WORKERS" \
    --mode picasso

echo "Finished: $(date)"
