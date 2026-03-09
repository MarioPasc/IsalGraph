#!/usr/bin/env bash
# SLURM compute worker for topology & complexity publication figures
set -euo pipefail

echo "=== Topology & Complexity Figures: SLURM Worker ==="
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

BENCH_CFG=$(python3 -c "
import yaml, json, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg['benchmarks']['topology_complexity_figs'], sys.stdout)
")

OUTPUT_DIR=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_dir'])")
SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")
N_GRAPHS=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_graphs', 8))")
N_MIN=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_min', 4))")
N_MAX=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_max', 13))")
EDGE_PROB=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('edge_prob', 0.35))")
EXHAUSTIVE_TIMEOUT=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('exhaustive_timeout', 60.0))")

mkdir -p "$OUTPUT_DIR"

echo "Config: output_dir=$OUTPUT_DIR"
echo "  seed=$SEED, n_graphs=$N_GRAPHS, n_min=$N_MIN, n_max=$N_MAX"
echo "  edge_prob=$EDGE_PROB, exhaustive_timeout=$EXHAUSTIVE_TIMEOUT"

python -m benchmarks.eval_visualizations.illustrative.topology_and_complexity \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --n-graphs "$N_GRAPHS" \
    --n-min "$N_MIN" \
    --n-max "$N_MAX" \
    --edge-prob "$EDGE_PROB" \
    --exhaustive-timeout "$EXHAUSTIVE_TIMEOUT"

echo "Finished: $(date)"
