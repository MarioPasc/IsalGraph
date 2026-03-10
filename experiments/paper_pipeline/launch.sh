#!/usr/bin/env bash
# =============================================================================
# IsalGraph Paper Pipeline Launcher
# =============================================================================
#
# Orchestrates the full paper figure reproduction pipeline.
# Submits SLURM jobs with dependency chains, or runs locally.
#
# Usage:
#   bash experiments/paper_pipeline/launch.sh                  # Submit to SLURM
#   bash experiments/paper_pipeline/launch.sh --dry-run        # Print commands only
#   bash experiments/paper_pipeline/launch.sh --local          # Run sequentially
#   bash experiments/paper_pipeline/launch.sh --step eval_setup  # Single step
#   bash experiments/paper_pipeline/launch.sh --config path/to/config.yaml
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/config.yaml"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=false
LOCAL_MODE=false
SINGLE_STEP=""
CONFIG_PATH="${DEFAULT_CONFIG}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --step)
            SINGLE_STEP="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash experiments/paper_pipeline/launch.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run           Print commands without executing"
            echo "  --local             Run steps sequentially as bash subprocesses"
            echo "  --step <name>       Run a single step (eval_setup, eval_correlation,"
            echo "                      eval_computational, eval_encoding, algorithm_figures,"
            echo "                      topology_figs, generate_figures)"
            echo "  --config <path>     Use custom config file (default: config.yaml)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Activate conda
# ---------------------------------------------------------------------------
CONDA_ENV_NAME="isalgraph"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Parse config.yaml
# ---------------------------------------------------------------------------
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}"
    exit 1
fi

CONFIG_JSON=$(python3 -c "
import yaml, json, sys
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg, sys.stdout)
")

# Extract paths (use local overrides for --local mode)
if [[ "$LOCAL_MODE" == "true" ]]; then
    CFG_REPO_DIR="$REPO_DIR"
else
    CFG_REPO_DIR=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['paths']['repo_dir'])")
fi
CFG_RUNS_DIR=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['paths']['runs_dir'])")
CFG_ACCOUNT=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['slurm']['account'])")
CFG_CONSTRAINT=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['slurm']['constraint'])")
CFG_CONDA_ENV=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['slurm']['conda_env'])")

# ---------------------------------------------------------------------------
# Verify benchmark symlinks
# ---------------------------------------------------------------------------
REQUIRED_SYMLINKS=(eval_setup eval_correlation eval_computational eval_encoding eval_visualizations)
SYMLINKS_OK=true
for pkg in "${REQUIRED_SYMLINKS[@]}"; do
    link="${REPO_DIR}/benchmarks/${pkg}"
    if [[ ! -e "$link" ]]; then
        echo "ERROR: Missing symlink: ${link}"
        echo "  Run: ln -sfn real_data/${pkg} benchmarks/${pkg}"
        SYMLINKS_OK=false
    fi
done
if [[ "$SYMLINKS_OK" == "false" ]]; then
    echo "ERROR: Required benchmark symlinks are missing. See above."
    exit 1
fi

# ---------------------------------------------------------------------------
# Generate run ID and create run directory
# ---------------------------------------------------------------------------
GIT_HASH=$(cd "$REPO_DIR" && git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(cd "$REPO_DIR" && git diff --quiet 2>/dev/null && echo "false" || echo "true")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIMESTAMP}_${GIT_HASH}"
if [[ "$GIT_DIRTY" == "true" ]]; then
    RUN_ID="${RUN_ID}_dirty"
fi

if [[ "$LOCAL_MODE" == "true" ]]; then
    RUN_DIR="${REPO_DIR}/runs/${RUN_ID}"
else
    RUN_DIR="${CFG_RUNS_DIR}/${RUN_ID}"
fi

echo "=============================================="
echo "IsalGraph Paper Pipeline"
echo "=============================================="
echo "Run ID:     ${RUN_ID}"
echo "Run dir:    ${RUN_DIR}"
echo "Repo:       ${CFG_REPO_DIR}"
echo "Config:     ${CONFIG_PATH}"
echo "Local mode: ${LOCAL_MODE}"
echo "Dry run:    ${DRY_RUN}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would create: ${RUN_DIR}"
else
    mkdir -p "${RUN_DIR}"/{data,correlation,computational,encoding,figures/_intermediate/{algorithm,topology},logs}

    # Copy and fill config
    python3 -c "
import yaml, sys, os
with open('${CONFIG_PATH}') as f:
    cfg = yaml.safe_load(f)
cfg['meta'] = {
    'git_commit': '${GIT_HASH}',
    'git_dirty': '${GIT_DIRTY}' == 'true',
    'launched_at': '$(date -Iseconds)',
    'launched_by': '$(whoami)@$(hostname)',
}
with open('${RUN_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    echo "Config frozen: ${RUN_DIR}/config.yaml"

    # Save conda environment
    if command -v conda >/dev/null 2>&1; then
        conda list --export > "${RUN_DIR}/conda_env.txt" 2>/dev/null || true
    fi
fi

# ---------------------------------------------------------------------------
# Helper: check if a step is enabled
# ---------------------------------------------------------------------------
step_enabled() {
    local step_name="$1"
    echo "$CONFIG_JSON" | python3 -c "
import json, sys
cfg = json.load(sys.stdin)
print(cfg['steps'].get('${step_name}', {}).get('enabled', False))
"
}

# ---------------------------------------------------------------------------
# Helper: get step resource limits
# ---------------------------------------------------------------------------
step_resources() {
    local step_name="$1"
    local field="$2"
    echo "$CONFIG_JSON" | python3 -c "
import json, sys
cfg = json.load(sys.stdin)
print(cfg['steps']['${step_name}']['${field}'])
"
}

# ---------------------------------------------------------------------------
# Helper: submit SLURM job
# ---------------------------------------------------------------------------
submit_job() {
    local job_name="$1"
    local step_name="$2"
    local worker_script="$3"
    local dependency="$4"  # "" for no dependency, or "afterok:ID1:ID2"

    local time_limit cpus mem_gb
    time_limit=$(step_resources "$step_name" "time_limit")
    cpus=$(step_resources "$step_name" "cpus")
    mem_gb=$(step_resources "$step_name" "mem_gb")

    local dep_flag=""
    if [[ -n "$dependency" ]]; then
        dep_flag="--dependency=${dependency}"
    fi

    local sbatch_cmd="sbatch --parsable \
        --job-name=paper_${job_name} \
        --output=${RUN_DIR}/logs/${job_name}_%j.out \
        --error=${RUN_DIR}/logs/${job_name}_%j.err \
        --time=${time_limit} \
        --cpus-per-task=${cpus} \
        --mem=${mem_gb}G \
        --constraint=${CFG_CONSTRAINT} \
        --account=${CFG_ACCOUNT} \
        --chdir=${CFG_REPO_DIR} \
        --export=ALL,ISALGRAPH_RUN_DIR=${RUN_DIR},ISALGRAPH_REPO_DIR=${CFG_REPO_DIR} \
        ${dep_flag} \
        ${worker_script}"

    # Info to stderr so command substitution only captures the job ID
    echo "  [${job_name}] time=${time_limit} cpus=${cpus} mem=${mem_gb}G ${dep_flag}" >&2

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] ${sbatch_cmd}" >&2
        echo "DRY_${job_name}"
    else
        local job_id
        job_id=$(eval "${sbatch_cmd}")
        echo "  -> Job ID: ${job_id}" >&2
        echo "${job_id}"
    fi
}

# ---------------------------------------------------------------------------
# Helper: run step locally
# ---------------------------------------------------------------------------
run_local() {
    local job_name="$1"
    local worker_script="$2"

    echo "  [${job_name}] Running locally..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would run: bash ${worker_script}"
        return 0
    fi

    local log_out="${RUN_DIR}/logs/${job_name}_local.out"
    local log_err="${RUN_DIR}/logs/${job_name}_local.err"

    export ISALGRAPH_RUN_DIR="${RUN_DIR}"
    export ISALGRAPH_REPO_DIR="${REPO_DIR}"

    if bash "${worker_script}" > "${log_out}" 2> "${log_err}"; then
        echo "  -> OK (log: ${log_out})"
    else
        local exit_code=$?
        echo "  -> FAILED (exit ${exit_code}, log: ${log_err})"
        echo "  Last 10 lines of stderr:"
        tail -n 10 "${log_err}" 2>/dev/null | sed 's/^/    /'
        return ${exit_code}
    fi
}

# ---------------------------------------------------------------------------
# Collect worker script paths
# ---------------------------------------------------------------------------
W_DIR="${SCRIPT_DIR}/workers"
W_STEP1="${W_DIR}/step1_eval_setup.sh"
W_STEP2A="${W_DIR}/step2a_eval_correlation.sh"
W_STEP2B="${W_DIR}/step2b_eval_computational.sh"
W_STEP2C="${W_DIR}/step2c_eval_encoding.sh"
W_STEP3A="${W_DIR}/step3a_algorithm_figures.sh"
W_STEP3B="${W_DIR}/step3b_topology_figs.sh"
W_STEP4="${W_DIR}/step4_generate_figures.sh"

# ---------------------------------------------------------------------------
# Single step mode
# ---------------------------------------------------------------------------
if [[ -n "$SINGLE_STEP" ]]; then
    echo "Running single step: ${SINGLE_STEP}"
    echo ""

    case "$SINGLE_STEP" in
        eval_setup)         run_local "step1" "$W_STEP1" ;;
        eval_correlation)   run_local "step2a" "$W_STEP2A" ;;
        eval_computational) run_local "step2b" "$W_STEP2B" ;;
        eval_encoding)      run_local "step2c" "$W_STEP2C" ;;
        algorithm_figures)  run_local "step3a" "$W_STEP3A" ;;
        topology_figs)      run_local "step3b" "$W_STEP3B" ;;
        generate_figures)   run_local "step4" "$W_STEP4" ;;
        *)
            echo "ERROR: Unknown step '${SINGLE_STEP}'"
            echo "Valid steps: eval_setup, eval_correlation, eval_computational,"
            echo "  eval_encoding, algorithm_figures, topology_figs, generate_figures"
            exit 1
            ;;
    esac

    echo ""
    echo "Done."
    exit 0
fi

# ---------------------------------------------------------------------------
# Full pipeline: Local mode
# ---------------------------------------------------------------------------
if [[ "$LOCAL_MODE" == "true" ]]; then
    echo "Running full pipeline locally (sequential)..."
    echo ""

    ERRORS=0

    # Step 1
    if [[ "$(step_enabled eval_setup)" == "True" ]]; then
        run_local "step1" "$W_STEP1" || ERRORS=$((ERRORS + 1))
    fi

    # Steps 2a, 2b, 2c (sequential in local mode)
    if [[ "$(step_enabled eval_correlation)" == "True" ]]; then
        run_local "step2a" "$W_STEP2A" || ERRORS=$((ERRORS + 1))
    fi
    if [[ "$(step_enabled eval_computational)" == "True" ]]; then
        run_local "step2b" "$W_STEP2B" || ERRORS=$((ERRORS + 1))
    fi
    if [[ "$(step_enabled eval_encoding)" == "True" ]]; then
        run_local "step2c" "$W_STEP2C" || ERRORS=$((ERRORS + 1))
    fi

    # Steps 3a, 3b
    if [[ "$(step_enabled algorithm_figures)" == "True" ]]; then
        run_local "step3a" "$W_STEP3A" || ERRORS=$((ERRORS + 1))
    fi
    if [[ "$(step_enabled topology_figs)" == "True" ]]; then
        run_local "step3b" "$W_STEP3B" || ERRORS=$((ERRORS + 1))
    fi

    # Step 4
    if [[ "$(step_enabled generate_figures)" == "True" ]]; then
        run_local "step4" "$W_STEP4" || ERRORS=$((ERRORS + 1))
    fi

    echo ""
    echo "=============================================="
    echo "Pipeline complete: ${RUN_DIR}"
    if [[ "$ERRORS" -gt 0 ]]; then
        echo "WARNING: ${ERRORS} step(s) failed. Check logs in ${RUN_DIR}/logs/"
    fi
    echo "=============================================="
    exit 0
fi

# ---------------------------------------------------------------------------
# Full pipeline: SLURM mode
# ---------------------------------------------------------------------------
echo "Submitting SLURM dependency chain..."
echo ""

# Track job IDs for manifest
declare -A JOB_IDS

# Step 1: eval_setup (no deps)
JOB1=""
if [[ "$(step_enabled eval_setup)" == "True" ]]; then
    JOB1=$(submit_job "step1" "eval_setup" "$W_STEP1" "")
    JOB_IDS[step1]="$JOB1"
fi

# Step 2a: eval_correlation (afterok:Step1)
JOB2A=""
if [[ "$(step_enabled eval_correlation)" == "True" ]]; then
    DEP=""
    [[ -n "$JOB1" ]] && DEP="afterok:${JOB1}"
    JOB2A=$(submit_job "step2a" "eval_correlation" "$W_STEP2A" "$DEP")
    JOB_IDS[step2a]="$JOB2A"
fi

# Step 2b: eval_computational (afterok:Step1)
JOB2B=""
if [[ "$(step_enabled eval_computational)" == "True" ]]; then
    DEP=""
    [[ -n "$JOB1" ]] && DEP="afterok:${JOB1}"
    JOB2B=$(submit_job "step2b" "eval_computational" "$W_STEP2B" "$DEP")
    JOB_IDS[step2b]="$JOB2B"
fi

# Step 2c: eval_encoding (no deps -- synthetic data)
JOB2C=""
if [[ "$(step_enabled eval_encoding)" == "True" ]]; then
    JOB2C=$(submit_job "step2c" "eval_encoding" "$W_STEP2C" "")
    JOB_IDS[step2c]="$JOB2C"
fi

# Step 3a: algorithm_figures (no deps -- standalone)
JOB3A=""
if [[ "$(step_enabled algorithm_figures)" == "True" ]]; then
    JOB3A=$(submit_job "step3a" "algorithm_figures" "$W_STEP3A" "")
    JOB_IDS[step3a]="$JOB3A"
fi

# Step 3b: topology_figs (no deps -- standalone)
JOB3B=""
if [[ "$(step_enabled topology_figs)" == "True" ]]; then
    JOB3B=$(submit_job "step3b" "topology_figs" "$W_STEP3B" "")
    JOB_IDS[step3b]="$JOB3B"
fi

# Step 4: generate_figures (afterok: all previous)
JOB4=""
if [[ "$(step_enabled generate_figures)" == "True" ]]; then
    # Build dependency string from all submitted jobs
    DEP_JOBS=""
    for jid in "$JOB2A" "$JOB2B" "$JOB2C" "$JOB3A" "$JOB3B"; do
        if [[ -n "$jid" ]]; then
            if [[ -n "$DEP_JOBS" ]]; then
                DEP_JOBS="${DEP_JOBS}:${jid}"
            else
                DEP_JOBS="${jid}"
            fi
        fi
    done
    DEP=""
    [[ -n "$DEP_JOBS" ]] && DEP="afterok:${DEP_JOBS}"
    JOB4=$(submit_job "step4" "generate_figures" "$W_STEP4" "$DEP")
    JOB_IDS[step4]="$JOB4"
fi

# ---------------------------------------------------------------------------
# Write manifest.json
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" != "true" ]]; then
    python3 -c "
import json, sys
manifest = {
    'run_id': '${RUN_ID}',
    'git_commit': '${GIT_HASH}',
    'git_dirty': '${GIT_DIRTY}' == 'true',
    'launched_at': '$(date -Iseconds)',
    'config_path': '${CONFIG_PATH}',
    'jobs': {
        'step1_eval_setup': '${JOB_IDS[step1]:-}',
        'step2a_eval_correlation': '${JOB_IDS[step2a]:-}',
        'step2b_eval_computational': '${JOB_IDS[step2b]:-}',
        'step2c_eval_encoding': '${JOB_IDS[step2c]:-}',
        'step3a_algorithm_figures': '${JOB_IDS[step3a]:-}',
        'step3b_topology_figs': '${JOB_IDS[step3b]:-}',
        'step4_generate_figures': '${JOB_IDS[step4]:-}',
    },
    'dependency_chain': {
        'step1': [],
        'step2a': ['step1'] if '${JOB1}' else [],
        'step2b': ['step1'] if '${JOB1}' else [],
        'step2c': [],
        'step3a': [],
        'step3b': [],
        'step4': [k for k in ['step2a','step2b','step2c','step3a','step3b']
                   if k in {'step2a': '${JOB2A}', 'step2b': '${JOB2B}',
                             'step2c': '${JOB2C}', 'step3a': '${JOB3A}',
                             'step3b': '${JOB3B}'} and
                   {'step2a': '${JOB2A}', 'step2b': '${JOB2B}',
                    'step2c': '${JOB2C}', 'step3a': '${JOB3A}',
                    'step3b': '${JOB3B}'}[k]],
    },
}
with open('${RUN_DIR}/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('Manifest: ${RUN_DIR}/manifest.json')
"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Pipeline submitted!"
echo "=============================================="
echo "Run directory: ${RUN_DIR}"
echo ""
echo "Dependency chain:"
echo "  Step 1  (eval_setup)       : ${JOB_IDS[step1]:-skipped}"
echo "  Step 2a (eval_correlation) : ${JOB_IDS[step2a]:-skipped}  -> depends on Step 1"
echo "  Step 2b (eval_computational): ${JOB_IDS[step2b]:-skipped}  -> depends on Step 1"
echo "  Step 2c (eval_encoding)    : ${JOB_IDS[step2c]:-skipped}  -> independent"
echo "  Step 3a (algorithm_figures): ${JOB_IDS[step3a]:-skipped}  -> independent"
echo "  Step 3b (topology_figs)    : ${JOB_IDS[step3b]:-skipped}  -> independent"
echo "  Step 4  (generate_figures) : ${JOB_IDS[step4]:-skipped}  -> depends on 2a,2b,2c,3a,3b"
echo ""
echo "Monitor: squeue -u $(whoami) --name=paper_step*"
echo "Logs:    ls ${RUN_DIR}/logs/"
echo "Figures: ls ${RUN_DIR}/figures/ (after completion)"
echo "=============================================="
