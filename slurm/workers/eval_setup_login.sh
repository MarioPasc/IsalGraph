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

# Download PyG datasets (requires internet, only available on login node)
echo "Downloading PyG datasets (LINUX, ALKANE)..."
CONFIG="${REPO_DIR}/slurm/config.yaml"
DATA_ROOT=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['benchmarks']['eval_setup']['data_root'])
")

conda run -n isalgraph python -c "
from torch_geometric.datasets import GEDDataset
import os
root = os.path.join('${DATA_ROOT}', 'datasets')
print('Downloading LINUX...')
GEDDataset(root=root, name='LINUX', train=True)
GEDDataset(root=root, name='LINUX', train=False)
print('Downloading ALKANE...')
GEDDataset(root=root, name='ALKANE', train=True)
GEDDataset(root=root, name='ALKANE', train=False)
print('PyG datasets downloaded OK')
"

echo "Login preparation complete. Submit via: bash slurm/launch.sh --benchmark eval_setup"
