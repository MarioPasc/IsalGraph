#!/usr/bin/env bash
# SLURM compute worker for eval_encoding benchmark
set -euo pipefail

echo "=== Eval Encoding: SLURM Worker ==="
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
json.dump(cfg['benchmarks']['eval_encoding'], sys.stdout)
")

DATA_ROOT=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('data_root', ''))")
OUTPUT_DIR=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_dir'])")
N_INSTANCES=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_instances', 5))")
N_REPS=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_reps', 25))")
MAX_N_GREEDY=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('max_n_greedy', 50))")
MAX_N_CANONICAL=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('max_n_canonical', 20))")
SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")

mkdir -p "$OUTPUT_DIR"

echo "Config: output_dir=$OUTPUT_DIR"
echo "  n_instances=$N_INSTANCES, n_reps=$N_REPS, max_n_greedy=$MAX_N_GREEDY"
echo "  max_n_canonical=$MAX_N_CANONICAL, seed=$SEED"

CMD="python -m benchmarks.eval_encoding.eval_encoding \
    --output-dir $OUTPUT_DIR \
    --n-instances $N_INSTANCES \
    --n-reps $N_REPS \
    --max-n-greedy $MAX_N_GREEDY \
    --max-n-canonical $MAX_N_CANONICAL \
    --seed $SEED \
    --csv --plot --table"

# Add data-root if specified
if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root $DATA_ROOT"
fi

eval "$CMD"

echo "Finished: $(date)"
