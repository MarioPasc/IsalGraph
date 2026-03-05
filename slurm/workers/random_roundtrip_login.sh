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

# Check conda env
if ! conda env list 2>/dev/null | grep -q isalgraph; then
    echo "WARNING: conda env 'isalgraph' not found. Creating..."
    conda create -n isalgraph python=3.11 -y
fi

# Install package
conda run -n isalgraph pip install -e "${REPO_DIR}[all]" --quiet

# Validate import
conda run -n isalgraph python -c "from isalgraph.core.string_to_graph import StringToGraph; print('OK')"

echo "Login preparation complete. Submit via: bash slurm/launch.sh --benchmark random_roundtrip"
