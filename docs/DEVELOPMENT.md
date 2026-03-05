# IsalGraph Development Guide

## Setup

```bash
conda activate isalgraph
cd /home/mpascual/research/code/IsalGraph
python -m pip install -e ".[all]"
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only (fast, no external deps)
python -m pytest tests/unit/ -v

# Integration tests (needs networkx, igraph)
python -m pytest tests/integration/ -v

# Property tests (hypothesis, may be slow)
python -m pytest tests/property/ -v

# With coverage
python -m pytest tests/ -v --cov=isalgraph --cov-report=term-missing
```

## Code Quality

```bash
# Lint
python -m ruff check src/ tests/ benchmarks/
python -m ruff check --fix src/ tests/ benchmarks/

# Format
python -m ruff format src/ tests/ benchmarks/

# Type checking
python -m mypy src/isalgraph/
```

## Development Workflow

1. **Phase-gated development.** Phase N+1 cannot start until Phase N tests pass.
2. **Test-first.** Write the test, then implement the code.
3. **Round-trip is king.** Every code change must preserve round-trip correctness.
4. **No external deps in core.** `isalgraph.core` must never import external libraries.

## Phase Progression

| Phase | Description | Test location | Status |
|-------|-------------|---------------|--------|
| 1 | Short string round-trip | tests/unit/test_roundtrip.py | DONE |
| 2 | Massive random testing | benchmarks/random_roundtrip/ | DONE |
| 3 | Canonical string | benchmarks/canonical_invariance/ | DONE |
| 4 | String length analysis | benchmarks/string_length_analysis/ | DONE |
| 5 | Levenshtein vs GED | benchmarks/levenshtein_vs_ged/ | DONE |

## Benchmarks

Benchmarks are NOT part of the test suite. Each has its own subdirectory under `benchmarks/` with a README.md describing the scientific claim being tested.

### Local execution

```bash
# Quick smoke test (all 4 benchmarks)
python -m benchmarks.random_roundtrip.random_roundtrip \
    --num-tests 50 --seed 42 --output-dir /tmp/rr --csv --plot --table
python -m benchmarks.canonical_invariance.canonical_invariance \
    --num-tests 20 --max-nodes 7 --seed 42 --output-dir /tmp/ci --csv --plot --table
python -m benchmarks.string_length_analysis.string_length_analysis \
    --seed 42 --max-nodes 30 --output-dir /tmp/sla --csv --plot --table
python -m benchmarks.levenshtein_vs_ged.levenshtein_vs_ged \
    --seed 42 --max-nodes 6 --num-random-pairs 10 --output-dir /tmp/lvg --csv --plot --table
```

### Parallel execution (local, multi-core)

```bash
python -m benchmarks.random_roundtrip.random_roundtrip \
    --num-tests 1000 --n-workers 8 --csv --plot --table
python -m benchmarks.canonical_invariance.canonical_invariance \
    --num-tests 200 --n-workers 8 --csv --plot --table
```

### Output files per benchmark

Each benchmark can produce:
- `*_results.json` -- Raw JSON results (always generated)
- `*_results.csv` -- Tabular data for downstream analysis (`--csv`)
- `*_figure.pdf` + `*_figure.png` -- Publication-quality figure (`--plot`)
- `*_table.tex` -- LaTeX table for papers (`--table`)

---

## Picasso HPC

### Architecture

The Picasso supercomputer uses SLURM for job scheduling. Our infrastructure follows a **login-worker + compute-worker** pattern:

```
Login Node (internet, no compute)
  |-- slurm/launch.sh (reads config.yaml, dispatches sbatch)
  |-- slurm/workers/*_login.sh (validates env, creates dirs)
  |
  v  sbatch
Compute Node (no internet, allocated CPUs/RAM)
  |-- slurm/workers/*_slurm.sh (loads modules, runs benchmark)
```

### Filesystem Layout on Picasso

```
/mnt/home/users/tic_163_uma/mpascual/
  fscratch/repos/IsalGraph/          <-- Git repo (cloned once)
/mnt/home/users/tic_163_uma/mpascual/
  execs/isalgraph/                   <-- Benchmark results
    random_roundtrip/
    canonical_invariance/
    string_length_analysis/
    levenshtein_vs_ged/
```

**Important**: `$FSCRATCH` is auto-purged after inactivity. Keep the repo there but push results to a permanent location or download them.

### Configuration

All Picasso parameters are centralized in `slurm/config.yaml`:
- Paths (repo_dir, results_dir)
- SLURM defaults (constraint, account)
- Per-benchmark: num_tests, max_nodes, time_limit, cpus, mem_gb

### Launching Jobs

```bash
# From the login node:

# Submit all enabled benchmarks
bash slurm/launch.sh

# Submit a single benchmark
bash slurm/launch.sh --benchmark random_roundtrip

# Dry run (print sbatch commands without submitting)
bash slurm/launch.sh --dry-run

# Check job status
squeue -u $USER
```

### SLURM Resource Allocation

| Benchmark | Time | CPUs | RAM | Constraint |
|-----------|------|------|-----|------------|
| random_roundtrip | 4h | 32 | 32G | sr (AMD EPYC) |
| canonical_invariance | 2 days | 64 | 64G | sr |
| string_length_analysis | 8h | 16 | 16G | sr |
| levenshtein_vs_ged | 2 days | 64 | 64G | sr |

### Coding Standards for Picasso

When writing code that runs on Picasso compute nodes:

1. **No internet access on compute nodes.** All dependencies must be pre-installed via `pip install -e ".[all]"` on the login node.

2. **Module system.** SLURM workers must load modules before activating conda:
   ```bash
   module load conda 2>/dev/null || true
   conda activate isalgraph
   ```

3. **Parallelization.** Use `ProcessPoolExecutor` from `concurrent.futures` (stdlib). Do NOT use MPI or multiprocessing.Pool for benchmark scripts. The number of workers should match `$SLURM_CPUS_PER_TASK`.

4. **Output paths.** Always write results to the configured `results_dir`, not to the repo directory. Use `os.makedirs(output_dir, exist_ok=True)` before writing.

5. **Constraint types.** Available node types on Picasso:
   - `sr` -- AMD EPYC 128 cores, 439GB RAM (recommended for CPU benchmarks)
   - `sd` -- Intel Xeon 52 cores
   - `bl` / `bc` -- GPU nodes (not needed for IsalGraph benchmarks)

6. **Time limits.** Max recommended: 3 days. For expensive benchmarks (canonical, GED), request 2 days and use `--max-nodes 8` to keep computation tractable.

7. **Reproducibility.** Always pass `--seed` explicitly. SLURM workers read seeds from `config.yaml`.

8. **Benchmark CLI interface.** Every benchmark must support these flags:
   - `--mode {local,picasso}` -- In picasso mode, enables all outputs automatically
   - `--n-workers N` -- For ProcessPoolExecutor parallelization
   - `--csv` -- Save raw results as CSV
   - `--plot` -- Generate publication figure (PDF + PNG)
   - `--table` -- Generate LaTeX table

## Git Conventions

- Commit messages: imperative mood, describe the "why"
- Branch naming: `feature/<description>`, `fix/<description>`, `phase-N/<description>`
- No force pushes to main
- Original code is read-only at `docs/original_code_and_files/`
