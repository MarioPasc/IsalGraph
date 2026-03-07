#!/usr/bin/env bash
# Login-node preparation for eval_correlation benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Eval Correlation: Login Node Preparation ==="

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
from benchmarks.eval_correlation.correlation_metrics import lins_ccc
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
ccc = lins_ccc(x, y)
print(f'Smoke test: CCC = {ccc:.4f} (expected ~0.99)')
assert 0.9 < ccc < 1.0, f'CCC out of range: {ccc}'
print('Core imports OK')
"

# Verify data exists
CONFIG="${REPO_DIR}/slurm/config.yaml"
DATA_ROOT=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['benchmarks']['eval_correlation']['data_root'])
")

echo "Checking eval data at: ${DATA_ROOT}"
for subdir in ged_matrices levenshtein_matrices canonical_strings graph_metadata; do
    if [[ -d "${DATA_ROOT}/${subdir}" ]]; then
        echo "  OK: ${subdir}"
    else
        echo "  MISSING: ${subdir}"
    fi
done

echo "Login preparation complete. Submit via: bash slurm/launch.sh --benchmark eval_correlation"
