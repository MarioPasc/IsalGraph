# Agent 3: Computational Advantage Analysis

**Priority**: Key supporting benchmark — demonstrates practical utility.
**Dependencies**: Agent 0 (SETUP) must complete first. Additionally, this agent uses the **real dataset graphs** loaded by Agent 0, but performs its own timing measurements.
**Parallelizable with**: Agents 1, 2, 4.
**Estimated local time**: ~2–4 hours (dominated by GED timing on larger graphs).
**Estimated Picasso time**: ~1 hour (parallelized).

---

## 1. Scientific Context

### 1.1 Research Question

The advisor's email explicitly states: *"computational advantage de la distancia de Levenshtein con respecto a la GED"*. We must demonstrate that the IsalGraph pipeline (encode graph → compute Levenshtein) is **faster** than direct GED computation, and characterize precisely when and by how much.

### 1.2 The Complexity Argument

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| IsalGraph encoding (graph → string) | O(f(n)) — empirically characterized | One-time cost per graph |
| Canonical string (exhaustive) | Exponential worst-case | Practical for n ≤ 12 |
| Levenshtein distance (string × string) | O(s₁ · s₂) | Conditionally optimal under SETH (Backurs & Indyk, 2015) |
| **Total: Encode + Compare** | **O(f(n₁) + f(n₂) + s₁·s₂)** | **Polynomial in string length** |
| GED (exact, A*) | **NP-hard** (exponential) | Intractable for n > ~16 (Blumenthal & Gamper, 2020) |
| GED (bipartite approx.) | O(n³) | Upper bound only (Riesen & Bunke, 2009) |

The critical insight: **IsalGraph encoding is a one-time preprocessing cost**. For a dataset of N graphs, encoding costs O(N · f(n)). Then all N(N-1)/2 pairwise comparisons use Levenshtein at O(s₁ · s₂) each. The total is:

$$T_{\text{IsalGraph}} = N \cdot f(n) + \binom{N}{2} \cdot O(s_1 \cdot s_2)$$

vs.

$$T_{\text{GED}} = \binom{N}{2} \cdot T_{\text{GED\_pair}}(n)$$

For datasets with many graphs, the amortized encoding cost becomes negligible, and the O(s₁·s₂) Levenshtein dominates — which is polynomial vs. exponential GED.

### 1.3 What We Measure

1. **Per-graph encoding time**: Time to compute canonical string (or greedy fallback).
2. **Per-pair Levenshtein time**: Time to compute Levenshtein distance between two strings.
3. **Per-pair GED time**: Time to compute exact GED via NetworkX A*.
4. **Total pipeline time**: Encoding + all-pairs Levenshtein vs. all-pairs GED.
5. **Speedup factor**: T_GED / T_IsalGraph per pair and amortized over dataset.
6. **Crossover point**: Graph size at which IsalGraph becomes faster than GED.

### 1.4 Benchmarking Best Practices

Following Weber et al. (2019), *Genome Biology* 20:125 and Bartz-Beielstein et al. (2020):

- **CPU time**: Use `time.process_time()` (not `time.perf_counter()`) for single-threaded measurements. This excludes I/O and sleep, measuring only CPU cycles.
- **Repetitions**: Each timing measurement repeated **25 times** (minimum 10, ideally 25+).
- **Report median and IQR** (not mean/SD) for skewed runtime distributions.
- **Warm-up**: Run each operation once before timing to avoid JIT/cache effects.
- **Single-threaded**: All timing comparisons are single-threaded for fairness.
- **Hardware specification**: Report CPU model, clock speed, RAM, OS, Python version.

---

## 2. Input Data

From Agent 0:

| File | Content |
|------|---------|
| `ged_matrices/{dataset}.npz` | GED matrix (for reference values) |
| `canonical_strings/{dataset}.json` | Precomputed canonical strings (for Levenshtein timing) |
| `graph_metadata/{dataset}.json` | Graph metadata |

Additionally, this agent **reloads the original graphs** (NetworkX format) to perform its own encoding timing. The agent reads graphs from:
- IAM Letter: parsed by the loader in Agent 0 (or re-parse from GXL).
- LINUX/ALKANE: loaded from PyG and converted to NetworkX.

---

## 3. Output Specification

### 3.1 Directory Structure

```
benchmarks/eval_computational/
    __init__.py
    eval_computational.py        # Main orchestrator
    timing_utils.py              # Timing infrastructure
    README.md

results/eval_computational/
    raw/
        {dataset}_encoding_times.csv        # Per-graph encoding times
        {dataset}_levenshtein_times.csv     # Per-pair Levenshtein times
        {dataset}_ged_times.csv             # Per-pair GED times (subset)
        {dataset}_amortized_comparison.csv  # Amortized pipeline comparison
    stats/
        {dataset}_timing_stats.json         # Summary statistics
        crossover_analysis.json             # Crossover point results
        scaling_regression.json             # log-log regression results
        hardware_info.json                  # System specification
    figures/
        computational_main_figure.pdf       # Main paper figure
        scaling_loglog_{dataset}.pdf        # Log-log scaling plots
        speedup_bar.pdf                     # Speedup bar chart
    tables/
        computational_summary.tex           # LaTeX table
```

### 3.2 Raw Data — Per-Graph Encoding Times

`{dataset}_encoding_times.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `graph_id` | str | Graph identifier |
| `n_nodes` | int | Node count |
| `n_edges` | int | Edge count |
| `density` | float | Edge density |
| `method` | str | "exhaustive" or "greedy_min" |
| `canonical_time_median_s` | float | Median of 25 runs |
| `canonical_time_iqr_s` | float | IQR of 25 runs |
| `canonical_time_all_s` | str | JSON array of all 25 measurements |
| `greedy_time_median_s` | float | Median greedy time (min over starts) |
| `string_length` | int | Length of canonical/greedy string |

### 3.3 Raw Data — Per-Pair Timing

Due to scale (hundreds of thousands of pairs), we time a **stratified subsample**:

For each dataset, select pairs stratified by max(n_i, n_j):
- 50 pairs per size bin: [3-4], [5-6], [7-8], [9-10], [11-12], [13+]
- Total: ~200-300 pairs per dataset (practical for 25 repetitions each)

`{dataset}_ged_times.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `graph_i` | str | First graph ID |
| `graph_j` | str | Second graph ID |
| `max_n` | int | max(n_i, n_j) |
| `ged_time_median_s` | float | Median of 25 GED computations |
| `ged_time_iqr_s` | float | IQR |
| `ged_value` | float | GED value |
| `ged_times_all_s` | str | JSON array of 25 measurements |

`{dataset}_levenshtein_times.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `graph_i` | str | First graph ID |
| `graph_j` | str | Second graph ID |
| `lev_time_median_s` | float | Median of 25 Levenshtein computations |
| `lev_time_iqr_s` | float | IQR |
| `levenshtein_value` | int | Levenshtein value |
| `string_len_i` | int | Length of string i |
| `string_len_j` | int | Length of string j |

### 3.4 Amortized Comparison

`{dataset}_amortized_comparison.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `n_graphs` | int | Number of graphs (hypothetical dataset sizes: 10, 50, 100, 200, 500, 1000) |
| `total_encoding_time_s` | float | Sum of encoding times for n_graphs |
| `total_levenshtein_time_s` | float | Sum of all-pairs Levenshtein |
| `total_isalgraph_time_s` | float | Encoding + Levenshtein |
| `total_ged_time_s` | float | Estimated all-pairs GED (extrapolated from sampled pairs) |
| `speedup` | float | total_ged_time / total_isalgraph_time |

---

## 4. Implementation Plan

### 4.1 Module Structure

```
benchmarks/eval_computational/
    __init__.py
    eval_computational.py     # CLI orchestrator
    timing_utils.py           # Timing functions
    README.md
```

### 4.2 Timing Infrastructure (`timing_utils.py`)

```python
"""Timing utilities for computational benchmarking.

Follows Weber et al. (2019) and Bartz-Beielstein et al. (2020) guidelines:
- CPU time (not wall clock) via time.process_time()
- 25 repetitions per measurement
- Report median and IQR
- Single-threaded execution
"""

import time
import platform
import psutil  # Optional, for hardware info
import numpy as np
from typing import Callable, Any


def time_function(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    n_reps: int = 25,
    warmup: int = 1,
) -> dict:
    """Time a function with multiple repetitions.
    
    Uses time.process_time() for CPU time measurement.
    
    Args:
        func: Function to time.
        args: Positional arguments.
        kwargs: Keyword arguments.
        n_reps: Number of timing repetitions.
        warmup: Number of warmup calls before timing.
    
    Returns:
        Dict with median, iqr, all_times, result (from last call).
    """
    kwargs = kwargs or {}
    
    # Warmup
    for _ in range(warmup):
        result = func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_reps):
        t0 = time.process_time()
        result = func(*args, **kwargs)
        t1 = time.process_time()
        times.append(t1 - t0)
    
    times_arr = np.array(times)
    return {
        "median_s": float(np.median(times_arr)),
        "iqr_s": float(np.percentile(times_arr, 75) - np.percentile(times_arr, 25)),
        "mean_s": float(np.mean(times_arr)),
        "std_s": float(np.std(times_arr)),
        "min_s": float(np.min(times_arr)),
        "max_s": float(np.max(times_arr)),
        "all_times_s": times,
        "n_reps": n_reps,
        "result": result,
    }


def get_hardware_info() -> dict:
    """Collect hardware and software specification for reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        info["cpu_freq_mhz"] = psutil.cpu_freq().max if psutil.cpu_freq() else None
    except ImportError:
        info["ram_gb"] = "unknown (psutil not installed)"
    return info
```

### 4.3 Encoding Time Measurement

For each graph in the dataset:

```python
def time_canonical_encoding(nx_graph: nx.Graph, n_reps: int = 25) -> dict:
    """Time the full canonical string computation.
    
    This includes:
    1. NetworkX → SparseGraph conversion
    2. Exhaustive canonical search (or greedy fallback)
    """
    adapter = NetworkXAdapter()
    
    def encode():
        sg = adapter.from_external(nx_graph, directed=False)
        try:
            return canonical_string(sg)
        except (ValueError, RuntimeError):
            # Fallback to greedy min
            gts = GraphToString(sg)
            best = None
            for v in range(sg.node_count()):
                try:
                    s, _ = gts.run(initial_node=v)
                    if best is None or (len(s), s) < (len(best), best):
                        best = s
                except:
                    continue
            return best
    
    return time_function(encode, n_reps=n_reps)
```

**Important**: For graphs where canonical search takes > 60s, reduce n_reps to 5 and use greedy-only timing. Log which graphs use reduced repetitions.

### 4.4 Levenshtein Time Measurement

```python
import Levenshtein

def time_levenshtein(s1: str, s2: str, n_reps: int = 25) -> dict:
    """Time Levenshtein distance computation between two strings."""
    return time_function(Levenshtein.distance, args=(s1, s2), n_reps=n_reps)
```

### 4.5 GED Time Measurement

```python
def time_ged(g1: nx.Graph, g2: nx.Graph, n_reps: int = 25) -> dict:
    """Time exact GED computation via NetworkX A*.
    
    Warning: For graphs > 10 nodes, single GED computation can take
    seconds to minutes. With 25 reps, this becomes expensive.
    For graphs > 12 nodes, reduce to 5 reps.
    """
    def compute_ged():
        return nx.graph_edit_distance(
            g1, g2,
            node_subst_cost=lambda n1, n2: 0,
            node_del_cost=lambda n: 1,
            node_ins_cost=lambda n: 1,
            edge_subst_cost=lambda e1, e2: 0,
            edge_del_cost=lambda e: 1,
            edge_ins_cost=lambda e: 1,
        )
    
    max_n = max(g1.number_of_nodes(), g2.number_of_nodes())
    actual_reps = 5 if max_n > 12 else n_reps
    
    return time_function(compute_ged, n_reps=actual_reps, warmup=1)
```

### 4.6 Crossover Analysis

The **crossover point** is the graph size n* at which IsalGraph becomes faster than GED for a single pair comparison. Above n*, the IsalGraph pipeline (encode both graphs + Levenshtein) is faster than direct GED.

```python
def analyze_crossover(encoding_times: pd.DataFrame, pair_times: pd.DataFrame) -> dict:
    """Find the crossover point where IsalGraph beats GED.
    
    For each size bin:
    - T_isalgraph(n) = median_encode(n) * 2 + median_levenshtein(n)
    - T_ged(n) = median_ged(n)
    
    The crossover is where T_isalgraph < T_ged.
    
    Also extrapolate via log-log regression for larger n.
    """
    ...
```

### 4.7 Scaling Regression

Fit T(n) ≈ c · n^α on log-log scale:

```python
def fit_scaling_exponent(sizes: np.ndarray, times: np.ndarray) -> dict:
    """Fit scaling exponent via OLS on log-log data.
    
    log(T) = α * log(n) + log(c)
    
    Returns: alpha, c, R², residuals.
    """
    log_n = np.log(sizes)
    log_t = np.log(times)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_t)
    
    return {
        "alpha": float(slope),
        "c": float(np.exp(intercept)),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }
```

### 4.8 Amortized Dataset Comparison

For hypothetical dataset sizes N ∈ {10, 50, 100, 200, 500, 1000}:

```python
def compute_amortized(
    N: int,
    mean_encode_time: float,  # Average encoding time per graph
    mean_lev_time: float,     # Average Levenshtein time per pair
    mean_ged_time: float,     # Average GED time per pair
) -> dict:
    """Compute amortized time for full dataset comparison."""
    n_pairs = N * (N - 1) // 2
    t_isalgraph = N * mean_encode_time + n_pairs * mean_lev_time
    t_ged = n_pairs * mean_ged_time
    return {
        "n_graphs": N,
        "n_pairs": n_pairs,
        "t_isalgraph_s": t_isalgraph,
        "t_ged_s": t_ged,
        "speedup": t_ged / t_isalgraph if t_isalgraph > 0 else float("inf"),
    }
```

---

## 5. Visualization

### Main Figure (`computational_main_figure.pdf`)

**Layout**: 2 × 2 panels.

| Panel | Content |
|-------|---------|
| (a) | Log-log scatter: per-pair GED time vs max(n_i, n_j). Separate series for IsalGraph total (encode+Lev) and GED. Reference lines for O(n²), O(n³), O(2^n). Mark crossover. |
| (b) | Bar chart: median speedup (GED_time / IsalGraph_time) per dataset. Error bars = IQR. |
| (c) | Line plot: amortized total time vs N (dataset size) for IsalGraph pipeline vs GED. Log-log scale. Show where curves cross. |
| (d) | Stacked bar: breakdown of IsalGraph time (encoding vs Levenshtein) vs GED time, per size bin. |

---

## 6. CLI Interface

```bash
python -m benchmarks.eval_computational.eval_computational \
    --data-root data/eval \
    --output-dir results/eval_computational \
    --n-timing-reps 25 \
    --n-pairs-per-bin 50 \
    --seed 42 \
    --csv --plot --table

# Quick test (fewer reps and pairs)
python -m benchmarks.eval_computational.eval_computational \
    --data-root data/eval \
    --output-dir /tmp/comp_test \
    --datasets iam_letter_low \
    --n-timing-reps 5 \
    --n-pairs-per-bin 10 \
    --csv --plot --table
```

---

## 7. Local Testing Plan

### 7.1 Smoke Test

```python
def test_timing_infrastructure():
    # Time a known function (e.g., sum of range(1000))
    result = time_function(lambda: sum(range(1000)), n_reps=10)
    assert result["median_s"] > 0
    assert result["median_s"] < 0.1  # Should be very fast
    assert len(result["all_times_s"]) == 10

def test_ged_timing():
    # Two small graphs (path_3 vs path_4)
    g1 = nx.path_graph(3)
    g2 = nx.path_graph(4)
    result = time_ged(g1, g2, n_reps=5)
    assert result["result"] == 1.0  # GED(P3, P4) = 1 (add one node + one edge)
    assert result["median_s"] < 1.0
```

### 7.2 Integration Test

```bash
python -m benchmarks.eval_computational.eval_computational \
    --data-root data/eval \
    --output-dir /tmp/comp_test \
    --datasets iam_letter_low \
    --n-timing-reps 5 \
    --n-pairs-per-bin 5 \
    --csv --plot --table
```

---

## 8. Methodological Notes

### 8.1 GED Timeout for LINUX Outliers

LINUX has rare graphs with 15-21 nodes. GED for pairs involving these graphs may take minutes. Set a **per-pair timeout of 300s**. If GED times out, record `inf` for that pair's GED time and exclude from scaling regression but include in the amortized comparison as a lower bound for GED time.

### 8.2 python-Levenshtein vs Pure Python

Use `python-Levenshtein` (C extension) for timing, as this represents the realistic fast-path a user would employ. Also time our own implementation (`isalgraph.core.canonical.levenshtein`) on a subset for comparison. Report both.

### 8.3 Encoding Time Decomposition

Decompose encoding time into:
1. NetworkX → SparseGraph conversion
2. GraphToString (greedy, one starting node)
3. Canonical search (exhaustive)

This helps identify bottlenecks. Report each component separately.

---

## 9. Acceptance Criteria

1. ✅ Per-graph encoding time CSV for all 5 datasets.
2. ✅ Per-pair timing CSV (Levenshtein and GED) for stratified subsample.
3. ✅ Crossover analysis JSON with identified crossover point.
4. ✅ Scaling regression JSON with exponents.
5. ✅ Hardware info JSON.
6. ✅ Main figure (4-panel) in PDF.
7. ✅ Summary LaTeX table.
8. ✅ Speedup > 1 for at least some graph sizes (the method should be faster somewhere).
9. ✅ Code passes `ruff check`.
