#!/usr/bin/env bash
# SLURM compute worker for random_roundtrip benchmark
#
# This script is submitted by launch.sh with SLURM directives.
# It runs on a compute node with allocated resources.
set -euo pipefail

echo "=== Random Roundtrip: SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-1}"
echo "Start:  $(date)"

CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

# ISALGRAPH_REPO_DIR is passed via --export from launch.sh
REPO_DIR="${ISALGRAPH_REPO_DIR:?ERROR: ISALGRAPH_REPO_DIR not set. Run via launch.sh or set manually.}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

# Parse config
RESULTS_DIR=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg['results_dir'])
")
BENCH_CFG=$(python3 -c "
import yaml, json, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg['benchmarks']['random_roundtrip'], sys.stdout)
")

NUM_TESTS=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('num_tests', 10000))")
MAX_STRING_LEN=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('max_string_len', 100))")
MAX_NODES=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('max_nodes', 50))")
SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")
N_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
OUT_DIR="${RESULTS_DIR}/random_roundtrip"

mkdir -p "$OUT_DIR"

echo "Config: num_tests=$NUM_TESTS, max_string_len=$MAX_STRING_LEN, max_nodes=$MAX_NODES"
echo "Workers: $N_WORKERS"
echo "Output: $OUT_DIR"

# Run benchmark
python -m benchmarks.random_roundtrip.random_roundtrip \
    --num-tests "$NUM_TESTS" \
    --max-string-len "$MAX_STRING_LEN" \
    --max-nodes "$MAX_NODES" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --mode picasso \
    --n-workers "$N_WORKERS" \
    --csv --plot --table

echo "Finished: $(date)"
