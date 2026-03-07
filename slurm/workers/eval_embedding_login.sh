#!/usr/bin/env bash
# Login-node preparation for eval_embedding benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Eval Embedding: Login Node Preparation ==="

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
from benchmarks.eval_embedding.embedding_methods import smacof, classical_mds_eigenanalysis
import numpy as np
# Smoke test: embed a small random distance matrix
rng = np.random.default_rng(42)
D = np.abs(rng.standard_normal((10, 10)))
D = (D + D.T) / 2
np.fill_diagonal(D, 0)
result = smacof(D, n_components=2, max_iter=50, n_init=1, seed=42)
print(f'Smoke test: stress_1={result.stress_1:.4f}, shape={result.coords.shape}')
assert result.coords.shape == (10, 2), f'Bad shape: {result.coords.shape}'
assert result.stress_1 < 1.0, f'Stress too high: {result.stress_1}'
print('Core imports and SMACOF OK')
"

# Verify data exists
CONFIG="${REPO_DIR}/slurm/config.yaml"
DATA_ROOT=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['benchmarks']['eval_embedding']['data_root'])
")

echo "Checking eval data at: ${DATA_ROOT}"
for subdir in ged_matrices levenshtein_matrices; do
    if [[ -d "${DATA_ROOT}/${subdir}" ]]; then
        echo "  OK: ${subdir}"
    else
        echo "  MISSING: ${subdir}"
    fi
done

echo "Login preparation complete. Submit via: bash slurm/launch.sh --benchmark eval_embedding"
