#!/usr/bin/env bash
# SLURM compute worker for levenshtein_vs_ged benchmark
set -euo pipefail

echo "=== Levenshtein vs GED: SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-1}"
echo "Start:  $(date)"

module load conda 2>/dev/null || true
conda activate isalgraph

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_DIR"

RESULTS_DIR=$(python3 -c "
import yaml
with open('slurm/config.yaml') as f:
    print(yaml.safe_load(f)['results_dir'])
")
BENCH_CFG=$(python3 -c "
import yaml, json, sys
with open('slurm/config.yaml') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg['benchmarks']['levenshtein_vs_ged'], sys.stdout)
")

MAX_NODES=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('max_nodes', 8))")
NUM_RANDOM_PAIRS=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('num_random_pairs', 500))")
SEED=$(echo "$BENCH_CFG" | python3 -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")
N_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
OUT_DIR="${RESULTS_DIR}/levenshtein_vs_ged"

mkdir -p "$OUT_DIR"

echo "Config: max_nodes=$MAX_NODES, num_random_pairs=$NUM_RANDOM_PAIRS, seed=$SEED"
echo "Workers: $N_WORKERS, Output: $OUT_DIR"

python -m benchmarks.levenshtein_vs_ged.levenshtein_vs_ged \
    --max-nodes "$MAX_NODES" \
    --num-random-pairs "$NUM_RANDOM_PAIRS" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --mode picasso \
    --n-workers "$N_WORKERS" \
    --csv --plot --table

echo "Finished: $(date)"
