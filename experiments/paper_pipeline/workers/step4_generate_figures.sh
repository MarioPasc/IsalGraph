#!/usr/bin/env bash
# =============================================================================
# Step 4: Generate Figures — Assemble all paper outputs from intermediate data
# =============================================================================
set -euo pipefail

echo "=== Paper Pipeline Step 4: Generate Figures ==="
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

FIGURES_DIR="${RUN_DIR}/figures"
mkdir -p "$FIGURES_DIR"

echo "Config: run_dir=$RUN_DIR, figures_dir=$FIGURES_DIR"

python "${REPO_DIR}/experiments/paper_pipeline/generate_figures.py" \
    --run-dir "$RUN_DIR"

echo "Finished: $(date)"
