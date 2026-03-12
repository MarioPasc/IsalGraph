#!/usr/bin/env bash
# =============================================================================
# Step 2d: Eval Message Length — IsalGraph vs GED encoding efficiency
# =============================================================================
set -euo pipefail

echo "=== Paper Pipeline Step 2d: Eval Message Length ==="
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
out = cfg['steps']['eval_message_length']
out['seed'] = cfg['experiment']['seed']
out['datasets'] = cfg['experiment']['datasets']
out['algorithms'] = cfg['experiment']['algorithms']
json.dump(out, sys.stdout)
")

SEED=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['seed'])")

# Map algorithm names to file-level method names
METHODS=$(echo "$STEP_CFG" | python3 -c "
import json, sys
cfg = json.load(sys.stdin)
algo_map = {
    'canonical': 'exhaustive',
    'canonical_pruned': 'pruned_exhaustive',
    'greedy_min': 'greedy',
    'greedy_single': 'greedy_single',
}
algos = cfg.get('algorithms', [])
methods = [algo_map.get(a, a) for a in algos]
print(','.join(methods))
")

DATASETS=$(echo "$STEP_CFG" | python3 -c "
import json, sys
cfg = json.load(sys.stdin)
print(','.join(cfg.get('datasets', [])))
")

DATA_ROOT="${RUN_DIR}/data"
OUTPUT_DIR="${RUN_DIR}/message_length"
mkdir -p "$OUTPUT_DIR"

echo "Config: data_root=$DATA_ROOT output_dir=$OUTPUT_DIR"
echo "  datasets=$DATASETS, methods=$METHODS"

python -m benchmarks.eval_message_length.eval_message_length \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --datasets "$DATASETS" \
    --methods "$METHODS" \
    --csv --plot --table

echo "Finished: $(date)"
