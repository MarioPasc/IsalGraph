# Evaluation Infrastructure Setup (Dual-Mode)

Computes evaluation artifacts for the exhaustive canonical vs. greedy-min
comparison across 5 datasets.

## Scientific Question

> Does the Levenshtein-GED correlation differ significantly between
> exhaustive canonical strings and greedy-min strings?

## Datasets

| Dataset | Domain | N graphs | Avg |V| | Max |V| | GED Method |
|---------|--------|----------|---------|---------|------------|
| IAM Letter LOW | Handwriting | ~750 | 4.7 | 9 | NetworkX A* (computed) |
| IAM Letter MED | Handwriting | ~750 | 4.7 | 9 | NetworkX A* (computed) |
| IAM Letter HIGH | Handwriting | ~750 | 4.7 | 9 | NetworkX A* (computed) |
| LINUX | Software deps | ~89 | 7.6 | 12 | Precomputed (GraphEdX) |
| AIDS | Molecular topology | ~860 | 8.9 | 12 | Precomputed (GraphEdX) |

All datasets are pre-downloaded and stored under `--source-dir`:
```
source/
    Letter/{LOW,MED,HIGH}/   # GXL/CXL files (Zenodo)
    LINUX/                   # GraphEdX .pt files
    AIDS/                    # GraphEdX .pt files
```

## Connectivity Caveat

IAM Letter graphs have significant disconnection rates (47% LOW, 44% MED,
8% HIGH). Disconnected graphs are filtered out since IsalGraph's G2S
algorithm requires connectivity.

## Usage

```bash
# Smoke test (2 graphs per dataset)
python -m benchmarks.eval_setup.eval_setup \
    --datasets iam_letter_low --max-graphs 2 --n-workers 1

# Full pipeline (local)
python -m benchmarks.eval_setup.eval_setup \
    --data-root data/eval --source-dir /path/to/source \
    --n-max 12 --n-workers 12 --seed 42

# Validate only
python -m benchmarks.eval_setup.eval_setup --data-root data/eval --validate-only
```

## Output Structure

```
data/eval/
    filtering_report.json
    validation_report.json
    ged_matrices/{dataset}.npz
    graph_metadata/{dataset}.json
    canonical_strings/{dataset}_{exhaustive,greedy}.json
    levenshtein_matrices/{dataset}_{exhaustive,greedy}.npz
    method_comparison/{dataset}_comparison.json
```

## Module Map

| Module | Purpose |
|--------|---------|
| `eval_setup.py` | CLI orchestrator |
| `dataset_filter.py` | Node-count + connectivity filtering |
| `iam_letter_loader.py` | GXL/CXL parser (IAM Letter) |
| `graphedx_loader.py` | GraphEdX .pt loader (LINUX, AIDS) |
| `ged_computer.py` | All-pairs exact GED (IAM Letter) |
| `canonical_computer.py` | Dual canonical computation |
| `levenshtein_computer.py` | All-pairs Levenshtein |
| `method_comparator.py` | Exhaustive vs greedy comparison |
| `validator.py` | Validation suite |
