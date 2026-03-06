#!/usr/bin/env bash
# Login-node preparation for alphabet_entropy benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Alphabet Entropy: Login Node Preparation ==="

if [[ ! -d "$REPO_DIR/src/isalgraph" ]]; then
    echo "ERROR: IsalGraph package not found at $REPO_DIR/src/isalgraph"
    exit 1
fi

CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

conda run -n isalgraph pip install -e "${REPO_DIR}[all]" --quiet

# Validate critical imports (canonical needed for Regime A)
conda run -n isalgraph python -c "
from isalgraph.core.canonical import canonical_string
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
print('OK')
"

echo "Login preparation complete. Submit via: bash slurm/launch.sh --benchmark alphabet_entropy"
