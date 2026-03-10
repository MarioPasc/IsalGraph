#!/usr/bin/env bash
# =============================================================================
# Step 2b: Eval Computational — Timing comparison (GED vs Levenshtein)
# =============================================================================
set -euo pipefail

echo "=== Paper Pipeline Step 2b: Eval Computational ==="
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
out = cfg['steps']['eval_computational']
out['seed'] = cfg['experiment']['seed']
out['source_dir'] = cfg['paths']['source_dir']
json.dump(out, sys.stdout)
")

N_TIMING_REPS=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_timing_reps'])")
N_PAIRS_PER_BIN=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_pairs_per_bin'])")
SEED=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['seed'])")
SOURCE_DIR=$(echo "$STEP_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['source_dir'])")

DATA_ROOT="${RUN_DIR}/data"
OUTPUT_DIR="${RUN_DIR}/computational"
mkdir -p "$OUTPUT_DIR"

echo "Config: data_root=$DATA_ROOT, source_dir=$SOURCE_DIR, output_dir=$OUTPUT_DIR"
echo "  n_timing_reps=$N_TIMING_REPS, n_pairs_per_bin=$N_PAIRS_PER_BIN, seed=$SEED"

python -m benchmarks.eval_computational.eval_computational \
    --data-root "$DATA_ROOT" \
    --source-dir "$SOURCE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-timing-reps "$N_TIMING_REPS" \
    --n-pairs-per-bin "$N_PAIRS_PER_BIN" \
    --seed "$SEED" \
    --mode picasso \
    --csv --plot --table

echo "Finished: $(date)"
