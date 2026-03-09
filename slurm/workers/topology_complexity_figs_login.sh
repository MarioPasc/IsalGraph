#!/usr/bin/env bash
# Login-node (local) runner for topology & complexity figures
# Usage: bash slurm/workers/topology_complexity_figs_login.sh [--output-dir DIR]
set -euo pipefail

echo "=== Topology & Complexity Figures: Login Node ==="
echo "Host:  $(hostname)"
echo "Start: $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_DIR"

CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

# Default output — override with first positional arg or --output-dir
OUTPUT_DIR="${1:-/media/mpascual/Sandisk2TB/research/isalgraph/results/figures/test_algo_figs_v7}"
if [[ "$OUTPUT_DIR" == "--output-dir" ]]; then
    OUTPUT_DIR="${2:?ERROR: --output-dir requires a path}"
fi

mkdir -p "$OUTPUT_DIR"

python -m benchmarks.eval_visualizations.illustrative.topology_and_complexity \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    --n-graphs 8 \
    --n-min 4 \
    --n-max 13 \
    --edge-prob 0.35 \
    --exhaustive-timeout 60.0

echo "Finished: $(date)"
