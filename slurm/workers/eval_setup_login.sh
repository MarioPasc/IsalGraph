#!/usr/bin/env bash
# Login-node preparation for eval_setup benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Eval Setup: Login Node Preparation ==="

if [[ ! -d "$REPO_DIR/src/isalgraph" ]]; then
    echo "ERROR: IsalGraph package not found at $REPO_DIR/src/isalgraph"
    exit 1
fi

CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi
conda run -n isalgraph pip install -e "${REPO_DIR}[eval]" --quiet
conda run -n isalgraph python -c "
from isalgraph.core.canonical import canonical_string
from isalgraph.core.graph_to_string import GraphToString
print('Core imports OK')
"

# Verify source data exists
CONFIG="${REPO_DIR}/slurm/config.yaml"
SOURCE_DIR=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['benchmarks']['eval_setup']['source_dir'])
")

echo "Checking source data at: ${SOURCE_DIR}"
for ds in Letter/LOW Letter/MED Letter/HIGH LINUX AIDS; do
    if [[ -d "${SOURCE_DIR}/${ds}" ]]; then
        echo "  OK: ${ds}"
    else
        echo "  MISSING: ${ds}"
    fi
done

echo "Login preparation complete. Submit via: bash slurm/launch.sh --benchmark eval_setup"
