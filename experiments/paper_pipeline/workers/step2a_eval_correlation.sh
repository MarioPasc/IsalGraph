#!/usr/bin/env bash
# =============================================================================
# Step 2a: Eval Correlation — Bootstrap correlation analysis
# =============================================================================
set -euo pipefail

echo "=== Paper Pipeline Step 2a: Eval Correlation ==="
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
out = cfg['steps']['eval_correlation']
out['seed'] = cfg['experiment']['seed']

# Map algorithm config names to internal method names
algo_to_method = {
    'canonical': 'exhaustive',
    'canonical_pruned': 'pruned_exhaustive',
    'greedy_min': 'greedy',
    'greedy_single': 'greedy_single',
}
algos = cfg['experiment'].get('algorithms', ['canonical', 'greedy_min', 'greedy_single'])
methods = [algo_to_method[a] for a in algos if a in algo_to_method]
out['methods'] = ','.join(methods)

json.dump(out, sys.stdout)
")

N_BOOTSTRAP=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_bootstrap'])")
N_PERMUTATIONS=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_permutations'])")
SEED=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['seed'])")
METHODS=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['methods'])")

DATA_ROOT="${RUN_DIR}/data"
OUTPUT_DIR="${RUN_DIR}/correlation"
mkdir -p "$OUTPUT_DIR"

echo "Config: data_root=$DATA_ROOT, output_dir=$OUTPUT_DIR"
echo "  n_bootstrap=$N_BOOTSTRAP, n_permutations=$N_PERMUTATIONS, seed=$SEED"
echo "  methods=$METHODS"

python -m benchmarks.eval_correlation.eval_correlation \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --n-permutations "$N_PERMUTATIONS" \
    --seed "$SEED" \
    --methods "$METHODS" \
    --mode picasso \
    --csv --plot --table

echo "Finished: $(date)"
