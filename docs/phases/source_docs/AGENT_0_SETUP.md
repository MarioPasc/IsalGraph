# Agent 0: Evaluation Infrastructure Setup (Dual-Mode: Exhaustive + Greedy)

**Priority**: MUST RUN FIRST — all other agents depend on this.
**Estimated local time**: ~4–6 hours (dominated by exhaustive canonical computation + IAM Letter GED).
**Estimated Picasso time**: ~1–2 hours (parallelized).

---

## 1. Scientific Context

IsalGraph encodes graph topology as an instruction string over alphabet Σ = {N,n,P,p,V,v,C,c,W}. We study **two** methods for computing the representative string of a graph:

1. **Exhaustive canonical** (`w*_G`): Lexicographically minimal shortest string across all starting nodes and all valid neighbor orderings at V/v branch points. This is the theoretically ideal representation — a **complete graph invariant** (w*_G = w*_H ⟺ G ≅ H). Its computation is exponential in the worst case (implemented in `src/isalgraph/core/canonical.py` via backtracking search).

2. **Greedy-min** (`w'_G`): Minimum-length string produced by the greedy `GraphToString` algorithm across all starting nodes. Polynomial time (see Agent 4 for exact characterization), but **not** a complete graph invariant — two isomorphic graphs may produce different greedy-min strings depending on internal tie-breaking. Implemented in `src/isalgraph/core/graph_to_string.py`.

### 1.1 The New Research Question

From the existing `canonical_invariance` benchmark: **greedy ≠ canonical in ~32% of test cases**. But this only reports a binary match/fail — it never quantifies whether the difference matters for the Levenshtein–GED proxy quality. This evaluation answers:

> **Q_method**: Does the Levenshtein–GED correlation differ significantly between exhaustive canonical strings and greedy-min strings? Specifically:
>
> - Is Spearman ρ(Lev_exhaustive, GED) > Spearman ρ(Lev_greedy, GED)?
> - If so, does the difference justify the exponential cost of exhaustive search?
> - For which graph families does the gap matter most?

This is a practically important question. If greedy-min yields comparable proxy quality, then the expensive exhaustive search is unnecessary for applications — making the entire IsalGraph pipeline polynomial-time with no meaningful quality loss. Conversely, if exhaustive is significantly better, it justifies the cost and makes the canonical string theoretically essential.

### 1.2 Datasets (from DATASET_STRATEGY.md)

| Dataset | Source | N graphs (raw) | Avg |V| | Max |V| | GED method |
|---------|--------|----------------|---------|---------|------------|
| IAM Letter LOW | Zenodo (GXL) | ~750 | 4.7 | 9 | Compute via NX A* |
| IAM Letter MED | Zenodo (GXL) | ~750 | 4.7 | 9 | Compute via NX A* |
| IAM Letter HIGH | Zenodo (GXL) | ~750 | 4.7 | 9 | Compute via NX A* |
| LINUX | PyG `GEDDataset` | 1,000 | 7.6 | ~21 | Extract precomputed |
| ALKANE | PyG `GEDDataset` | 150 | 8.9 | ~12 | Extract precomputed |

### 1.3 Dataset Filtering

To ensure exhaustive canonical computation is feasible for **every** graph without fallback, we apply a **node-count filter** before any computation:

$$\mathcal{D}_{\text{filtered}} = \{ G \in \mathcal{D} \mid |V(G)| \leq N_{\max} \}$$

**Default threshold**: `N_max = 12`.

**Justification**: The codebase's empirically validated limits are `CANONICAL_LIMIT_SPARSE = 12` (trees, paths, stars, cycles) and `CANONICAL_LIMIT_DEFAULT = 8` (dense graphs like complete, high-density GNP). At N=12:
- Trees: canonical completes in < 1s (minimal branching).
- Sparse graphs (density < 0.3): a few seconds.
- Moderate density (0.3–0.5): 10–60s.
- Dense graphs (density > 0.5): potentially minutes, but very few real-world graphs in our datasets are both large AND dense.

**Expected filtering impact**:

| Dataset | Max |V| (raw) | N_max = 12 | Graphs kept | Graphs dropped | % kept |
|---------|----------------|------------|-------------|----------------|--------|
| IAM Letter LOW | 9 | 12 | ~750 | 0 | 100% |
| IAM Letter MED | 9 | 12 | ~750 | 0 | 100% |
| IAM Letter HIGH | 9 | 12 | ~750 | 0 | 100% |
| ALKANE | ~12 | 12 | ~150 | ~0 | ~100% |
| LINUX | ~21 | 12 | ~900+ | ~60–100 | ~90–94% |

For LINUX, the dropped graphs are rare outliers (13–21 nodes). We report the exact filtering statistics and note it in the paper's experimental setup.

**Safety net**: Even after filtering, a **per-graph timeout of 600s** guards against unexpectedly slow exhaustive computations. Any graph exceeding this timeout is excluded post-hoc and logged as a "timeout exclusion."

**CLI-configurable**: The threshold `--n-max` allows experimentation with stricter (10) or more generous (14) limits.

### 1.4 GED Cost Function (Uniform, Topology-Only)

For IAM Letter, we compute GED using NetworkX's `graph_edit_distance()` with:
- Node insertion/deletion cost: 1
- Edge insertion/deletion cost: 1
- Node substitution cost: 0 (all nodes identical after stripping coordinates)

This is exact A* — not approximate. Feasible for max 9 nodes (Blumenthal & Gamper, 2020, showed timeout at ~16).

For LINUX and ALKANE, GED was precomputed by Bai et al. (2019) using exact A* and is available via `GEDDataset.ged`.

**Important**: After filtering LINUX, we must extract the **submatrix** of the GED matrix corresponding to the kept graphs. The original PyG GED matrix indexes graphs 0..999; we keep only those with ≤ N_max nodes and take the corresponding rows/columns.

---

## 2. Dependencies

### 2.1 Conda Environment

```bash
conda activate isalgraph
pip install -e ".[all]" --break-system-packages
pip install python-Levenshtein tqdm --break-system-packages
```

**The agent must add an `eval` optional dependency group to `pyproject.toml`**:

```toml
[project.optional-dependencies]
eval = [
    "isalgraph[networkx,pyg,viz,bench]",
    "python-Levenshtein>=0.21",
    "scikit-learn>=1.3",
    "scikit-bio>=0.6",
    "tqdm>=4.60",
]
```

### 2.2 PyTorch Geometric

`GEDDataset` requires `torch` and `torch_geometric`. These auto-download from Google Drive, so **internet access is required on first run** (login node on Picasso).

### 2.3 IAM Letter Download

Available at: https://zenodo.org/records/13763793 — download and extract the ZIP. The dataset contains LOW/MED/HIGH subdirectories with `.gxl` graph files and `.cxl` index files.

---

## 3. Output Specification

### 3.1 Directory Structure

The key design: **every artifact that depends on the string method is duplicated** with `_exhaustive` and `_greedy` suffixes. Artifacts that are method-independent (GED matrices, graph metadata) exist once.

```
data/
    eval/
        datasets/
            iam_letter_low/              # Raw GXL files (or symlinks)
            iam_letter_med/
            iam_letter_high/
            linux/                       # PyG cache
            alkane/                      # PyG cache

        # ── Method-INDEPENDENT (computed once) ──────────────────
        ged_matrices/
            iam_letter_low.npz
            iam_letter_med.npz
            iam_letter_high.npz
            linux.npz                    # Submatrix after filtering
            alkane.npz
            README.md

        graph_metadata/
            iam_letter_low.json
            iam_letter_med.json
            iam_letter_high.json
            linux.json                   # Only filtered graphs
            alkane.json

        filtering_report.json            # What was dropped and why

        # ── Method-DEPENDENT (duplicated per method) ────────────
        canonical_strings/
            iam_letter_low_exhaustive.json
            iam_letter_low_greedy.json
            iam_letter_med_exhaustive.json
            iam_letter_med_greedy.json
            iam_letter_high_exhaustive.json
            iam_letter_high_greedy.json
            linux_exhaustive.json
            linux_greedy.json
            alkane_exhaustive.json
            alkane_greedy.json

        levenshtein_matrices/
            iam_letter_low_exhaustive.npz
            iam_letter_low_greedy.npz
            iam_letter_med_exhaustive.npz
            iam_letter_med_greedy.npz
            iam_letter_high_exhaustive.npz
            iam_letter_high_greedy.npz
            linux_exhaustive.npz
            linux_greedy.npz
            alkane_exhaustive.npz
            alkane_greedy.npz

        # ── Method comparison (unique to this setup) ────────────
        method_comparison/
            iam_letter_low_comparison.json
            iam_letter_med_comparison.json
            iam_letter_high_comparison.json
            linux_comparison.json
            alkane_comparison.json

        validation_report.json
```

### 3.2 GED Matrix Format (`.npz`) — Unchanged

| Key | Type | Description |
|-----|------|-------------|
| `ged_matrix` | `float64 [N, N]` | Pairwise exact GED. `inf` for unavailable pairs. Symmetric. Diagonal = 0. |
| `node_counts` | `int32 [N]` | Number of nodes per graph. |
| `edge_counts` | `int32 [N]` | Number of edges per graph. |
| `graph_ids` | `str [N]` | Unique identifier per graph. |
| `labels` | `str [N]` | Class label. |
| `metadata` | `dict` | `{"dataset", "ged_method", "ged_cost_function", "source", "n_graphs", "n_valid_pairs", "n_max_filter", "n_dropped"}` |

### 3.3 Canonical String Format (`.json`) — Per Method

```json
{
    "dataset": "iam_letter_low",
    "method": "exhaustive",
    "n_max_filter": 12,
    "n_graphs": 750,
    "strings": {
        "AP1_0001": {
            "string": "VVC",
            "length": 3,
            "time_s": 0.12
        }
    },
    "stats": {
        "mean_length": 12.3,
        "median_length": 11,
        "std_length": 4.1,
        "max_length": 28,
        "min_length": 0,
        "mean_time_s": 0.15,
        "median_time_s": 0.08,
        "max_time_s": 45.2,
        "total_time_s": 112.5,
        "n_timeout_exclusions": 0
    }
}
```

For `method = "greedy"`, the format is identical. The `string` field contains the greedy-min string (shortest over all starting nodes).

### 3.4 Method Comparison Format (`.json`)

Per-dataset file comparing exhaustive vs. greedy-min for every graph.

```json
{
    "dataset": "iam_letter_low",
    "n_graphs": 750,
    "per_graph": [
        {
            "graph_id": "AP1_0001",
            "exhaustive_string": "VVC",
            "greedy_string": "VVC",
            "exhaustive_length": 3,
            "greedy_length": 3,
            "strings_identical": true,
            "length_gap": 0,
            "levenshtein_between_methods": 0,
            "exhaustive_time_s": 0.12,
            "greedy_time_s": 0.003,
            "speedup": 40.0,
            "n_nodes": 5,
            "n_edges": 4,
            "density": 0.4
        }
    ],
    "aggregate": {
        "n_identical_strings": 510,
        "pct_identical_strings": 68.0,
        "n_identical_lengths": 680,
        "pct_identical_lengths": 90.7,
        "mean_length_gap": 0.4,
        "max_length_gap": 3,
        "mean_levenshtein_between_methods": 0.8,
        "max_levenshtein_between_methods": 5,
        "mean_speedup": 85.3,
        "median_speedup": 42.0,
        "total_exhaustive_time_s": 112.5,
        "total_greedy_time_s": 1.3
    },
    "matrix_comparison": {
        "spearman_r": 0.97,
        "pearson_r": 0.98,
        "max_abs_diff": 4,
        "mean_abs_diff": 0.3,
        "frac_identical_entries": 0.72,
        "n_pairs": 280875
    }
}
```

### 3.5 Filtering Report (`filtering_report.json`)

```json
{
    "n_max_filter": 12,
    "datasets": {
        "iam_letter_low": {
            "n_raw": 750,
            "n_kept": 750,
            "n_dropped": 0,
            "pct_kept": 100.0,
            "max_nodes_raw": 9,
            "dropped_graph_ids": [],
            "dropped_node_counts": []
        },
        "linux": {
            "n_raw": 1000,
            "n_kept": 938,
            "n_dropped": 62,
            "pct_kept": 93.8,
            "max_nodes_raw": 21,
            "max_nodes_kept": 12,
            "dropped_graph_ids": ["train_042", "train_089"],
            "dropped_node_counts": [13, 14, 15, 21],
            "node_count_histogram_raw": {"3": 12, "4": 45, "5": 78},
            "node_count_histogram_kept": {"3": 12, "4": 45, "5": 78}
        }
    }
}
```

### 3.6 Levenshtein Matrix Format (`.npz`) — Per Method

| Key | Type | Description |
|-----|------|-------------|
| `levenshtein_matrix` | `int32 [N, N]` | Pairwise Levenshtein distance. Symmetric. Diagonal = 0. |
| `graph_ids` | `str [N]` | Same ordering as GED matrix. |
| `method` | `str` | `"exhaustive"` or `"greedy"`. |

---

## 4. Implementation Plan

### 4.1 Module Structure

```
benchmarks/eval_setup/
    __init__.py
    eval_setup.py                  # Main orchestrator (CLI entry point)
    iam_letter_loader.py           # GXL parser for IAM Letter
    pyg_ged_extractor.py           # GED extraction from PyTorch Geometric
    ged_computer.py                # NetworkX GED computation (IAM Letter)
    canonical_computer.py          # Dual canonical string computation
    levenshtein_computer.py        # All-pairs Levenshtein computation
    dataset_filter.py              # Node-count filtering + reporting
    method_comparator.py           # Exhaustive vs greedy comparison
    validator.py                   # Comprehensive validation suite
    README.md
```

### 4.2 Step 0: Dataset Filtering (`dataset_filter.py`)

**This is the FIRST step after loading.** Every downstream operation works only on filtered graphs.

```python
"""Dataset filtering by node count.

Ensures all graphs are within the exhaustive canonical search
feasibility range, eliminating any need for fallback mechanisms.
"""

import logging
from dataclasses import dataclass

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filtering a dataset."""
    n_raw: int
    n_kept: int
    n_dropped: int
    kept_indices: list[int]
    dropped_indices: list[int]
    dropped_node_counts: list[int]
    dropped_graph_ids: list[str]


def filter_by_node_count(
    graphs: list[nx.Graph],
    graph_ids: list[str],
    n_max: int,
) -> FilterResult:
    """Filter graphs to those with at most n_max nodes.

    Args:
        graphs: List of NetworkX graphs.
        graph_ids: Corresponding graph identifiers.
        n_max: Maximum allowed node count (inclusive).

    Returns:
        FilterResult with indices and statistics.
    """
    kept_idx, dropped_idx, dropped_counts, dropped_ids = [], [], [], []

    for i, (g, gid) in enumerate(zip(graphs, graph_ids)):
        if g.number_of_nodes() <= n_max:
            kept_idx.append(i)
        else:
            dropped_idx.append(i)
            dropped_counts.append(g.number_of_nodes())
            dropped_ids.append(gid)
            logger.info("Dropped %s: %d nodes > n_max=%d", gid, g.number_of_nodes(), n_max)

    return FilterResult(
        n_raw=len(graphs), n_kept=len(kept_idx), n_dropped=len(dropped_idx),
        kept_indices=kept_idx, dropped_indices=dropped_idx,
        dropped_node_counts=dropped_counts, dropped_graph_ids=dropped_ids,
    )
```

For LINUX/ALKANE with precomputed GED, extract the submatrix after filtering:

```python
def extract_submatrix(ged_matrix: np.ndarray, kept_indices: list[int]) -> np.ndarray:
    """Extract submatrix for kept graphs."""
    idx = np.array(kept_indices)
    return ged_matrix[np.ix_(idx, idx)]
```

### 4.3 Step 1: IAM Letter GXL Parser (`iam_letter_loader.py`)

Parse `.cxl` index files for (filename → class_label) mapping, then parse each `.gxl` file for topology (strip x,y coordinates), build `nx.Graph` with integer node labels (0..N-1). Store all splits (train/validation/test) unified with a `split` metadata field.

After parsing, apply `filter_by_node_count`. For IAM Letter (max 9 < 12), nothing drops.

### 4.4 Step 2: PyG GED Extraction (`pyg_ged_extractor.py`)

Load via `GEDDataset`, convert PyG Data to NetworkX (topology only), apply filter, extract GED submatrix. Handle `inf` entries (test-test for LINUX, train-test for ALKANE). Verify the submatrix still has ≥ 20 graphs and ≥ 190 valid pairs.

### 4.5 Step 3: GED Computation for IAM Letter (`ged_computer.py`)

All-pairs exact GED via `networkx.graph_edit_distance()` with uniform costs. Parallelized with `ProcessPoolExecutor`. Checkpoints every 10,000 pairs.

### 4.6 Step 4: Dual Canonical String Computation (`canonical_computer.py`)

**Core modification.** For every graph, compute BOTH exhaustive and greedy-min:

```python
"""Compute canonical strings via both exhaustive and greedy-min methods.

For each graph G:
  - exhaustive: canonical_string(G) from src/isalgraph/core/canonical.py
  - greedy-min: min over all starting nodes of GraphToString(G, v)
"""

import time
import logging
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.canonical import canonical_string, levenshtein as _lev
from isalgraph.core.graph_to_string import GraphToString

logger = logging.getLogger(__name__)
TIMEOUT_S = 600


def compute_both_methods(nx_graph, graph_id):
    """Compute exhaustive canonical and greedy-min for one graph.

    Returns dict with both methods' strings, times, and comparison.
    """
    adapter = NetworkXAdapter()
    sg = adapter.from_external(nx_graph, directed=False)
    n = sg.node_count()

    # ---- Exhaustive canonical ----
    t0 = time.perf_counter()
    try:
        exhaustive_str = canonical_string(sg)
        exhaustive_time = time.perf_counter() - t0
    except (ValueError, RuntimeError) as e:
        logger.warning("Exhaustive failed for %s (%d nodes): %s", graph_id, n, e)
        exhaustive_str = None
        exhaustive_time = time.perf_counter() - t0

    # ---- Greedy-min (all starting nodes) ----
    t0 = time.perf_counter()
    greedy_results = []
    for v in range(n):
        try:
            gts = GraphToString(sg)
            s, _ = gts.run(initial_node=v)
            greedy_results.append((v, len(s), s))
        except (ValueError, RuntimeError):
            continue
    greedy_time = time.perf_counter() - t0

    if greedy_results:
        greedy_results.sort(key=lambda x: (x[1], x[2]))
        best_start, best_len, best_str = greedy_results[0]
    else:
        best_start, best_len, best_str = -1, -1, None

    # ---- Comparison ----
    strings_identical = (exhaustive_str == best_str) if (exhaustive_str and best_str) else None
    length_gap = (best_len - len(exhaustive_str)) if (exhaustive_str and best_str) else None
    lev_between = _lev(exhaustive_str, best_str) if (exhaustive_str and best_str) else None

    return {
        "graph_id": graph_id,
        "n_nodes": n,
        "n_edges": sg.logical_edge_count(),
        "density": sg.logical_edge_count() / (n * (n - 1) / 2) if n > 1 else 0.0,
        "exhaustive_string": exhaustive_str,
        "exhaustive_length": len(exhaustive_str) if exhaustive_str else -1,
        "exhaustive_time_s": round(exhaustive_time, 4),
        "greedy_string": best_str,
        "greedy_length": best_len,
        "greedy_time_s": round(greedy_time, 4),
        "greedy_best_start_node": best_start,
        "greedy_all_starts": [{"start_node": v, "length": l} for v, l, _ in greedy_results],
        "strings_identical": strings_identical,
        "length_gap": length_gap,
        "levenshtein_between_methods": lev_between,
        "speedup": round(exhaustive_time / greedy_time, 1) if greedy_time > 0 else None,
    }
```

### 4.7 Step 5: Dual Levenshtein Matrix (`levenshtein_computer.py`)

Compute all-pairs Levenshtein twice: on exhaustive strings and on greedy-min strings. Cross-validate C extension against our own implementation on first 50 graphs.

### 4.8 Step 6: Method Comparison (`method_comparator.py`)

Aggregate per-graph comparison data. Additionally compute Levenshtein matrix correlation (Spearman between vectorized upper-triangles of the two matrices).

### 4.9 Step 7: Validation (`validator.py`)

Validates BOTH sets of canonical strings and BOTH Levenshtein matrices. Key checks:
- Exhaustive: wherever GED=0, Levenshtein must be 0 (completeness).
- Greedy: wherever GED=0, document Levenshtein mismatches (expected since greedy is not invariant).
- Cross-consistency of graph_id ordering across all files.

---

## 5. Pipeline Execution Order

```
Step 0: Parse all datasets + apply node-count filter
        |
        +-> Step 1a: Compute IAM Letter GED matrices (parallel, ~25 min)
        +-> Step 1b: Extract LINUX/ALKANE GED submatrices (fast, ~2 min)
        |
        v
Step 2: Compute dual canonical strings (parallel, ~80 min)
        |
        +-> Step 3a: Levenshtein matrix (exhaustive)
        +-> Step 3b: Levenshtein matrix (greedy)
        |
        v
Step 4: Method comparison analysis
        |
        v
Step 5: Validation
```

---

## 6. CLI Interface

```bash
# Full pipeline (local)
python -m benchmarks.eval_setup.eval_setup \
    --data-root data/eval \
    --iam-letter-path /path/to/Letter \
    --n-max 12 --n-workers 12 --seed 42

# Tighter filter for faster testing
python -m benchmarks.eval_setup.eval_setup \
    --data-root /tmp/eval_test \
    --iam-letter-path /path/to/Letter \
    --n-max 10 --datasets iam_letter_low --n-workers 4

# Picasso mode
python -m benchmarks.eval_setup.eval_setup \
    --data-root $FSCRATCH/repos/IsalGraph/data/eval \
    --iam-letter-path $FSCRATCH/datasets/Letter \
    --n-max 12 --n-workers 64 --mode picasso

# Validate only
python -m benchmarks.eval_setup.eval_setup --data-root data/eval --validate-only
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root` | `data/eval` | Root output directory |
| `--iam-letter-path` | `data/eval/datasets/iam_letter_raw` | Zenodo archive path |
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

---

## 7. Local Testing Plan

### 7.1 Smoke Test (< 5 min)

```python
def test_filter():
    graphs = [nx.path_graph(n) for n in [3, 5, 8, 10, 13, 15]]
    ids = [f"g_{n}" for n in [3, 5, 8, 10, 13, 15]]
    result = filter_by_node_count(graphs, ids, n_max=12)
    assert result.n_kept == 4 and result.n_dropped == 2

def test_dual_canonical():
    G = nx.cycle_graph(4)
    result = compute_both_methods(G, "cycle_4")
    assert result["exhaustive_length"] <= result["greedy_length"]
    assert result["length_gap"] >= 0

def test_submatrix():
    M = np.array([[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]], dtype=float)
    sub = extract_submatrix(M, [0, 2, 3])
    assert sub.shape == (3, 3) and sub[0, 1] == 2

def test_method_comparison():
    G = nx.star_graph(4)
    result = compute_both_methods(G, "star_5")
    assert result["exhaustive_length"] == result["greedy_length"]
```

### 7.2 Integration Test (< 15 min)

```bash
python -m benchmarks.eval_setup.eval_setup \
    --data-root /tmp/eval_smoke --iam-letter-path /path/to/Letter \
    --datasets iam_letter_low --n-max 12 --n-workers 4 --seed 42
# Verify dual outputs exist
ls /tmp/eval_smoke/canonical_strings/iam_letter_low_{exhaustive,greedy}.json
ls /tmp/eval_smoke/levenshtein_matrices/iam_letter_low_{exhaustive,greedy}.npz
ls /tmp/eval_smoke/method_comparison/iam_letter_low_comparison.json
```

---

## 8. Error Handling

1. **Exhaustive timeout**: Exclude graph entirely. Log in `filtering_report.json`.
2. **Disconnected graphs**: Skip (canonical requires connectivity).
3. **GED pair timeout** (>60s at max 9 nodes): Store `inf`, continue.
4. **Download failures**: Clear error messages.
5. **Checkpointing**: GED every 10K pairs, canonical every 100 graphs.

---

## 9. Acceptance Criteria

1. ✅ `filtering_report.json` exists with per-dataset statistics.
2. ✅ 5 GED matrices pass validation.
3. ✅ 10 canonical string files (5 datasets × 2 methods), 0 failures, 0 timeouts.
4. ✅ 10 Levenshtein matrices (5 × 2) pass validation.
5. ✅ 5 method comparison files exist.
6. ✅ Exhaustive: GED=0 ⟹ Levenshtein=0 (completeness check).
7. ✅ `validation_report.json` all checks passed.
8. ✅ Pipeline runs end-to-end on ≥1 dataset locally.
9. ✅ `pytest tests/unit/ -v` passes.
10. ✅ `ruff check` passes.
