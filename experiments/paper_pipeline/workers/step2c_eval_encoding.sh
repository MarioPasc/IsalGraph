#!/usr/bin/env bash
# =============================================================================
# Step 2c: Eval Encoding — Empirical complexity on synthetic graphs
# =============================================================================
set -euo pipefail

echo "=== Paper Pipeline Step 2c: Eval Encoding ==="
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
out = cfg['steps']['eval_encoding']
out['seed'] = cfg['experiment']['seed']
json.dump(out, sys.stdout)
")

N_INSTANCES=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_instances'])")
N_REPS=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_reps'])")
MAX_N_GREEDY=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['max_n_greedy'])")
MAX_N_CANONICAL=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['max_n_canonical'])")
SEED=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['seed'])")

OUTPUT_DIR="${RUN_DIR}/encoding"
mkdir -p "$OUTPUT_DIR"

echo "Config: output_dir=$OUTPUT_DIR"
echo "  n_instances=$N_INSTANCES, n_reps=$N_REPS, max_n_greedy=$MAX_N_GREEDY"
echo "  max_n_canonical=$MAX_N_CANONICAL, seed=$SEED"

python -m benchmarks.eval_encoding.eval_encoding \
    --output-dir "$OUTPUT_DIR" \
    --n-instances "$N_INSTANCES" \
    --n-reps "$N_REPS" \
    --max-n-greedy "$MAX_N_GREEDY" \
    --max-n-canonical "$MAX_N_CANONICAL" \
    --seed "$SEED" \
    --csv --plot --table

echo "Finished: $(date)"
