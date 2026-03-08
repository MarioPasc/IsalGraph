#!/usr/bin/env bash
# Login-node preparation for eval_computational benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Eval Computational: Login Node Preparation ==="

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
from benchmarks.eval_computational.timing_utils import time_function, get_hardware_info
import numpy as np

# Smoke test: time a trivial function
result = time_function(lambda: sum(range(1000)), n_reps=10)
print(f'Smoke test: median={result[\"median_s\"]:.6f}s, n_reps={result[\"n_reps\"]}')
assert result['median_s'] > 0, 'Timing returned 0'
assert len(result['all_times_s']) == 10

hw = get_hardware_info()
print(f'Hardware: {hw[\"platform\"]}, {hw[\"cpu_count\"]} CPUs')
print('Core imports and timing OK')
"

# Verify data exists
CONFIG="${REPO_DIR}/slurm/config.yaml"
DATA_ROOT=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['benchmarks']['eval_computational']['data_root'])
")
SOURCE_DIR=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['benchmarks']['eval_computational']['source_dir'])
")

echo "Checking eval data at: ${DATA_ROOT}"
for subdir in canonical_strings graph_metadata ged_matrices; do
    if [[ -d "${DATA_ROOT}/${subdir}" ]]; then
        echo "  OK: ${subdir}"
    else
        echo "  MISSING: ${subdir}"
    fi
done

echo "Checking source data at: ${SOURCE_DIR}"
for subdir in Letter LINUX AIDS; do
    if [[ -d "${SOURCE_DIR}/${subdir}" ]]; then
        echo "  OK: ${subdir}"
    else
        echo "  MISSING: ${subdir}"
    fi
done

echo "Login preparation complete. Submit via: bash slurm/launch.sh --benchmark eval_computational"
