#!/usr/bin/env bash
# Login-node preparation for random_roundtrip benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$SLURM_DIR")"

echo "=== Random Roundtrip: Login Node Preparation ==="

# Validate paths
if [[ ! -d "$REPO_DIR/src/isalgraph" ]]; then
    echo "ERROR: IsalGraph package not found at $REPO_DIR/src/isalgraph"
    exit 1
fi

CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

# Install package
conda run -n isalgraph pip install -e "${REPO_DIR}[all]" --quiet

# Validate import
conda run -n isalgraph python -c "from isalgraph.core.string_to_graph import StringToGraph; print('OK')"

echo "Login preparation complete. Submit via: bash slurm/launch.sh --benchmark random_roundtrip"
