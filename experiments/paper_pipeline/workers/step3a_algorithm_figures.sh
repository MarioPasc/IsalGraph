#!/usr/bin/env bash
# =============================================================================
# Step 3a: Algorithm Figures — Standalone illustrative figures
# =============================================================================
set -euo pipefail

echo "=== Paper Pipeline Step 3a: Algorithm Figures ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

# --- Environment ---
RUN_DIR="${ISALGRAPH_RUN_DIR:?ERROR: ISALGRAPH_RUN_DIR not set}"
REPO_DIR="${ISALGRAPH_REPO_DIR:?ERROR: ISALGRAPH_REPO_DIR not set}"
cd "$REPO_DIR"

CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

OUTPUT_DIR="${RUN_DIR}/figures/_intermediate/algorithm"
mkdir -p "$OUTPUT_DIR"

echo "Config: output_dir=$OUTPUT_DIR"

python -m benchmarks.eval_visualizations.illustrative.algorithm_figures \
    --output-dir "$OUTPUT_DIR"

echo "Finished: $(date)"
