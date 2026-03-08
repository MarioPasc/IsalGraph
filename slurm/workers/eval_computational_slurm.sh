#!/usr/bin/env bash
# SLURM compute worker for eval_computational benchmark
set -euo pipefail

echo "=== Eval Computational: SLURM Worker ==="
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
json.dump(cfg['benchmarks']['eval_computational'], sys.stdout)
")

DATA_ROOT=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['data_root'])")
SOURCE_DIR=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['source_dir'])")
OUTPUT_DIR=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin)['output_dir'])")
N_TIMING_REPS=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_timing_reps', 25))")
N_PAIRS_PER_BIN=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('n_pairs_per_bin', 50))")
SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")

mkdir -p "$OUTPUT_DIR"

echo "Config: data_root=$DATA_ROOT, source_dir=$SOURCE_DIR, output_dir=$OUTPUT_DIR"
echo "  n_timing_reps=$N_TIMING_REPS, n_pairs_per_bin=$N_PAIRS_PER_BIN, seed=$SEED"

python -m benchmarks.eval_computational.eval_computational \
    --data-root "$DATA_ROOT" \
    --source-dir "$SOURCE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-timing-reps "$N_TIMING_REPS" \
    --n-pairs-per-bin "$N_PAIRS_PER_BIN" \
    --seed "$SEED" \
    --mode picasso \
    --csv --plot --table

echo "Finished: $(date)"
