#!/usr/bin/env bash
# =============================================================================
# Step 3b: Topology & Complexity Figures — Neighbourhood and complexity plots
# =============================================================================
set -euo pipefail

echo "=== Paper Pipeline Step 3b: Topology & Complexity Figures ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
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
out = cfg['steps']['topology_figs']
out['seed'] = cfg['experiment']['seed']
json.dump(out, sys.stdout)
")

SEED=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['seed'])")
N_GRAPHS=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_graphs'])")
N_MIN=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_min'])")
N_MAX=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_max'])")
EDGE_PROB=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['edge_prob'])")
EXHAUSTIVE_TIMEOUT=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['exhaustive_timeout'])")

OUTPUT_DIR="${RUN_DIR}/figures/_intermediate/topology"
mkdir -p "$OUTPUT_DIR"

echo "Config: output_dir=$OUTPUT_DIR"
echo "  seed=$SEED, n_graphs=$N_GRAPHS, n_min=$N_MIN, n_max=$N_MAX"
echo "  edge_prob=$EDGE_PROB, exhaustive_timeout=$EXHAUSTIVE_TIMEOUT"

python -m benchmarks.eval_visualizations.illustrative.topology_and_complexity \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --n-graphs "$N_GRAPHS" \
    --n-min "$N_MIN" \
    --n-max "$N_MAX" \
    --edge-prob "$EDGE_PROB" \
    --exhaustive-timeout "$EXHAUSTIVE_TIMEOUT"

echo "Finished: $(date)"
