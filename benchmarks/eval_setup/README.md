# Evaluation Infrastructure Setup (Dual-Mode)

Computes evaluation artifacts for the exhaustive canonical vs. greedy-min
comparison across 5 datasets.

## Scientific Question

> Does the Levenshtein-GED correlation differ significantly between
> exhaustive canonical strings and greedy-min strings?

## Datasets

| Dataset | Source | Max |V| | GED Method |
|---------|--------|---------|------------|
| IAM Letter LOW/MED/HIGH | Zenodo GXL | 9 | NetworkX A* |
| LINUX | PyG GEDDataset | ~21 | Precomputed (Bai 2019) |
| ALKANE | PyG GEDDataset | ~12 | Precomputed (Bai 2019) |

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
    --data-root data/eval --n-max 12 --n-workers 12 --seed 42

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
| `iam_letter_loader.py` | GXL/CXL parser |
| `pyg_ged_extractor.py` | PyG GED extraction |
| `ged_computer.py` | All-pairs exact GED |
| `canonical_computer.py` | Dual canonical computation |
| `levenshtein_computer.py` | All-pairs Levenshtein |
| `method_comparator.py` | Exhaustive vs greedy comparison |
| `validator.py` | Validation suite |
