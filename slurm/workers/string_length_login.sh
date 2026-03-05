#!/usr/bin/env bash
# Login-node preparation for string_length_analysis benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== String Length Analysis: Login Node Preparation ==="

if [[ ! -d "$REPO_DIR/src/isalgraph" ]]; then
    echo "ERROR: IsalGraph package not found at $REPO_DIR/src/isalgraph"
    exit 1
fi

conda run -n isalgraph pip install -e "${REPO_DIR}[all]" --quiet
conda run -n isalgraph python -c "from isalgraph.core.graph_to_string import GraphToString; print('OK')"

echo "Login preparation complete. Submit via: bash slurm/launch.sh --benchmark string_length_analysis"
