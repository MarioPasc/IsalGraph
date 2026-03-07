# Agent 0: Evaluation Infrastructure Setup (Dual-Mode: Exhaustive + Greedy)

**Priority**: MUST RUN FIRST -- all other agents depend on this.
**Estimated Picasso time**: ~1-2 hours (parallelized).

---

## 1. Scientific Context

IsalGraph encodes graph topology as an instruction string over alphabet Sigma = {N,n,P,p,V,v,C,c,W}. We study **two** methods for computing the representative string of a graph:

1. **Exhaustive canonical** (`w*_G`): Lexicographically minimal shortest string across all starting nodes and all valid neighbor orderings at V/v branch points. This is a **complete graph invariant** (w*_G = w*_H iff G ~ H). Exponential worst case (`src/isalgraph/core/canonical.py`).

2. **Greedy-min** (`w'_G`): Minimum-length string produced by the greedy `GraphToString` algorithm across all starting nodes. Polynomial time, but **not** a complete graph invariant. (`src/isalgraph/core/graph_to_string.py`).

### 1.1 The Research Question

> **Q_method**: Does the Levenshtein-GED correlation differ significantly between exhaustive canonical strings and greedy-min strings?

### 1.2 Datasets

| Dataset | Domain | N graphs | Avg |V| | Max |V| | GED method |
|---------|--------|----------|---------|---------|------------|
| IAM Letter LOW | Handwriting | ~750 | 4.7 | 9 | Compute via NX A* |
| IAM Letter MED | Handwriting | ~750 | 4.7 | 9 | Compute via NX A* |
| IAM Letter HIGH | Handwriting | ~750 | 4.7 | 9 | Compute via NX A* |
| LINUX | Software deps | ~89 | 7.6 | 12 | Precomputed (GraphEdX) |
| AIDS | Molecular topology | ~860 | 8.9 | 12 | Precomputed (GraphEdX) |

**Source data** is pre-downloaded at `--source-dir`:
```
source/
    Letter/{LOW,MED,HIGH}/   # GXL/CXL files (Zenodo)
    LINUX/                   # GraphEdX .pt files
    AIDS/                    # GraphEdX .pt files
```

See `DATASET_STRATEGY_UPDATED.md` for full dataset rationale.

### 1.3 Dataset Filtering

$$\mathcal{D}_{\text{filtered}} = \{ G \in \mathcal{D} \mid |V(G)| \leq N_{\max} \land G \text{ is connected} \}$$

**Default**: `N_max = 12`. Connectivity is required by IsalGraph's G2S algorithm.

### 1.4 GED Cost Functions

- **IAM Letter**: Uniform topology-only (node ins/del = 1, edge ins/del = 1, node sub = 0). Computed via `networkx.graph_edit_distance()`.
- **LINUX / AIDS**: Precomputed by Bai et al. (2019), curated by Jain et al. (NeurIPS 2024). GraphEdX `no_attr_data` uses Cn=(0,0,0), Ce=(1,2,0). Within-split GED only; cross-split pairs are inf.

---

## 2. Dependencies

```toml
[project.optional-dependencies]
eval = [
    "isalgraph[networkx,viz,bench]",
    "torch>=2.0",
    "python-Levenshtein>=0.21",
]
```

---

## 3. Output Specification

### 3.1 Directory Structure

```
data/eval/
    ged_matrices/{dataset}.npz
    graph_metadata/{dataset}.json
    canonical_strings/{dataset}_{exhaustive,greedy}.json
    levenshtein_matrices/{dataset}_{exhaustive,greedy}.npz
    method_comparison/{dataset}_comparison.json
    filtering_report.json
    validation_report.json
```

### 3.2 GED Matrix Format (.npz)

| Key | Type | Description |
|-----|------|-------------|
| `ged_matrix` | `float64 [N, N]` | Pairwise exact GED. `inf` for unavailable pairs. Symmetric. Diagonal = 0. |
| `node_counts` | `int32 [N]` | Number of nodes per graph. |
| `edge_counts` | `int32 [N]` | Number of edges per graph. |
| `graph_ids` | `str [N]` | Unique identifier per graph. |
| `labels` | `str [N]` | Class label. |
| `metadata` | `dict` | Dataset, method, cost function, source, n_graphs, n_valid_pairs. |

### 3.3 Canonical String Format (.json)

```json
{
    "dataset": "linux",
    "method": "exhaustive",
    "n_max_filter": 12,
    "n_graphs": 89,
    "strings": {
        "linux_train_0000": {"string": "VVVVVnvnvv", "length": 10, "time_s": 0.05}
    },
    "stats": { "mean_length": ..., "median_length": ..., ... }
}
```

---

## 4. Module Structure

```
benchmarks/eval_setup/
    __init__.py
    eval_setup.py              # CLI orchestrator
    iam_letter_loader.py       # GXL/CXL parser (IAM Letter)
    graphedx_loader.py         # GraphEdX .pt loader (LINUX, AIDS)
    ged_computer.py            # NetworkX GED computation (IAM Letter)
    canonical_computer.py      # Dual canonical string computation
    levenshtein_computer.py    # All-pairs Levenshtein
    dataset_filter.py          # Node-count + connectivity filtering
    method_comparator.py       # Exhaustive vs greedy comparison
    validator.py               # Validation suite
    README.md
```

---

## 5. Pipeline Execution Order

```
Step 0: Load all datasets + filter (node count + connectivity)
    |
    +-> Step 1a: Compute IAM Letter GED (parallel, ~25 min)
    +-> Step 1b: Extract LINUX/AIDS GED submatrices (fast)
    |
Step 2: Dual canonical strings (parallel)
    |
    +-> Step 3a: Levenshtein matrix (exhaustive)
    +-> Step 3b: Levenshtein matrix (greedy)
    |
Step 4: Method comparison analysis
    |
Step 5: Validation
```

---

## 6. CLI Interface

```bash
# Smoke test
python -m benchmarks.eval_setup.eval_setup \
    --datasets iam_letter_high --max-graphs 4 --n-workers 1

# Full pipeline
python -m benchmarks.eval_setup.eval_setup \
    --data-root data/eval --source-dir /path/to/source \
    --n-max 12 --n-workers 12 --seed 42

# Picasso mode
python -m benchmarks.eval_setup.eval_setup \
    --data-root $DATA_ROOT --source-dir $SOURCE_DIR \
    --n-max 12 --n-workers 64 --mode picasso

# Validate only
python -m benchmarks.eval_setup.eval_setup --data-root data/eval --validate-only
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root` | `data/eval` | Root output directory |
| `--source-dir` | (local default) | Path to source data (Letter/, LINUX/, AIDS/) |
| `--n-max` | `12` | Maximum node count filter |
| `--n-workers` | `1` | Parallel workers |
| `--seed` | `42` | Random seed |
| `--mode` | `local` | `local` or `picasso` |
| `--datasets` | `all` | Comma-separated dataset names |
| `--validate-only` | `False` | Only validate existing files |
| `--skip-ged` | `False` | Skip GED computation |
| `--skip-canonical` | `False` | Skip canonical computation |
| `--skip-levenshtein` | `False` | Skip Levenshtein computation |
| `--timeout-per-graph` | `600` | Per-graph canonical timeout (seconds) |
| `--max-graphs` | `None` | Limit graphs per dataset (smoke testing) |

---

## 7. Acceptance Criteria

1. `filtering_report.json` exists with per-dataset statistics.
2. 5 GED matrices pass validation (symmetry, diagonal=0, non-negative, triangle inequality).
3. 10 canonical string files (5 datasets x 2 methods), 0 failures.
4. 10 Levenshtein matrices (5 x 2) pass validation.
5. 5 method comparison files exist.
6. Exhaustive: GED=0 => Levenshtein=0 (completeness check).
7. `validation_report.json` all checks passed.
8. `ruff check` passes.
