# Agent 1: Levenshtein–GED Correlation Analysis

**Priority**: Core benchmark — this is the FUNDAMENTAL claim of the paper.
**Dependencies**: Agent 0 (SETUP) must complete first.
**Parallelizable with**: Agents 2, 3, 4 (all independent after setup).
**Estimated local time**: ~30 min (computation is on precomputed matrices).
**Estimated Picasso time**: ~15 min.

---

## 1. Scientific Context

### 1.1 The Central Hypothesis

IsalGraph's core proposition is that Levenshtein distance between canonical strings serves as a **practical proxy** for Graph Edit Distance (GED). Formally:

> **H₁**: For graphs G₁, G₂ with canonical strings w*₁, w*₂:  
> d_Levenshtein(w*₁, w*₂) correlates positively with GED(G₁, G₂)

This is novel: **no prior work establishes, for general graphs, that Levenshtein distance on a canonical string representation correlates with GED** (see Experimental Evaluation Design, Section 3). The closest results are for trees only (Akutsu et al., 2010, *Algorithmica* 57:325–348, proved O(n^{3/4})-approximation for ordered rooted trees of bounded degree).

### 1.2 Hypotheses to Test

**H₁ (Global correlation)**: Across each dataset, Spearman ρ(Levenshtein, GED) > 0 with p < 0.001 (Mantel test, permutation-based).

**H₂ (Monotone degradation)**: Across IAM Letter LOW → MED → HIGH, Spearman ρ decreases monotonically. Tested via the Jonckheere-Terpstra ordered-alternative test:
- H₀: ρ_LOW = ρ_MED = ρ_HIGH  
- H₁: ρ_LOW ≥ ρ_MED ≥ ρ_HIGH (with at least one strict inequality)

**H₃ (Tree superiority)**: ALKANE (all trees) exhibits the highest Spearman ρ among all datasets, consistent with theoretical results that tree edit distance approximation via string edit distance has known bounds (Akutsu et al., 2010).

**H₄ (Same-class discrimination)**: For within-class pairs (same label), Levenshtein distances are significantly smaller than for between-class pairs, as measured by Cohen's d effect size.

### 1.3 Statistical Methods

**Mantel test** (Mantel, 1967, *Cancer Research* 27:209–220): The standard method for assessing correlation between two distance matrices. Computes Pearson or Spearman correlation between the n(n−1)/2 upper-triangular entries, with significance assessed via **permutation testing** (9,999 permutations). Pairwise distances are not independent, so parametric p-values are invalid — permutation p-values are required.

**Bootstrap confidence intervals**: For Pearson r and Spearman ρ, compute 10,000 bootstrap resamples of the (GED, Levenshtein) pairs to obtain BCa (bias-corrected and accelerated) 95% confidence intervals. This is important because with hundreds of thousands of pairs, even trivial correlations will be "significant" — the CI width and point estimate magnitude matter more than the p-value.

**Lin's Concordance Correlation Coefficient (CCC)** (Lin, 1989, *Biometrics* 45:255–268): Evaluates how closely paired observations fall on the 45° identity line, penalizing both poor correlation and systematic bias. Unlike Pearson r, CCC detects scale and location shifts. Interpretation: >0.99 almost perfect, 0.95–0.99 substantial, <0.90 poor (McBride, 2005). **Note**: We do NOT expect CCC ≈ 1 because Levenshtein and GED are on different scales. CCC quantifies the scale relationship after optimal linear transformation. Compute it on both raw and z-normalized distances.

**Kendall's τ**: Preferred over Spearman for datasets with many tied values (LINUX has ~8.9% unique graphs → many GED=0 pairs). More robust to ties and more interpretable for ordinal data.

**Precision@k**: Fraction of true k-nearest neighbors (by GED) that are also k-nearest neighbors by Levenshtein, averaged over all graphs. Standard in neural GED papers (SimGNN, GRAPHEDX). Use k ∈ {5, 10, 20}.

**Jonckheere-Terpstra test** (for H₂): Non-parametric test for ordered alternatives in independent samples. Available in `scipy.stats` (as of scipy 1.11+, `page_trend_test` or a custom implementation). If not available in the installed scipy version, implement using the Mann-Kendall statistic approach.

---

## 2. Input Data

All data is produced by Agent 0 and located under `data/eval/`:

| File | Content |
|------|---------|
| `ged_matrices/{dataset}.npz` | Pairwise exact GED matrix |
| `levenshtein_matrices/{dataset}.npz` | Pairwise Levenshtein distance matrix |
| `canonical_strings/{dataset}.json` | Canonical strings with metadata |
| `graph_metadata/{dataset}.json` | Node/edge counts, class labels |

Datasets: `iam_letter_low`, `iam_letter_med`, `iam_letter_high`, `linux`, `alkane`.

---

## 3. Output Specification

### 3.1 Directory Structure

```
benchmarks/eval_correlation/
    __init__.py
    eval_correlation.py          # Main orchestrator
    correlation_metrics.py       # Statistical computation functions
    README.md
    
results/eval_correlation/        # Output directory
    raw/
        {dataset}_pair_data.csv            # Every valid pair's distances
        {dataset}_bootstrap_samples.npz    # Bootstrap correlation samples
        {dataset}_permutation_dist.npz     # Mantel test permutation distribution
    stats/
        {dataset}_correlation_stats.json   # All statistical results
        cross_distortion_analysis.json     # H₂ test results
        summary_table.json                 # Summary across all datasets
    figures/
        correlation_scatter_{dataset}.pdf  # Per-dataset scatter
        correlation_main_figure.pdf        # Main paper figure (multi-panel)
        bland_altman_{dataset}.pdf         # Bland-Altman plots
        heatmap_comparison_{dataset}.pdf   # Side-by-side heatmaps
    tables/
        correlation_summary.tex            # LaTeX table for paper
        precision_at_k.tex                 # Precision@k table
```

### 3.2 Raw Data — Pair-Level CSV

**Critical**: Save every valid pair. This enables downstream reanalysis with different statistical methods without recomputation.

Columns for `{dataset}_pair_data.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `graph_i` | str | Graph ID of first graph |
| `graph_j` | str | Graph ID of second graph |
| `ged` | float | Graph Edit Distance |
| `levenshtein` | int | Levenshtein distance between canonical strings |
| `n_i` | int | Node count of graph i |
| `n_j` | int | Node count of graph j |
| `m_i` | int | Edge count of graph i |
| `m_j` | int | Edge count of graph j |
| `label_i` | str | Class label of graph i |
| `label_j` | str | Class label of graph j |
| `same_class` | bool | Whether both graphs share the same class label |
| `canonical_len_i` | int | Length of canonical string i |
| `canonical_len_j` | int | Length of canonical string j |
| `canonical_method_i` | str | "exhaustive" or "greedy_min" |
| `canonical_method_j` | str | "exhaustive" or "greedy_min" |
| `abs_size_diff` | int | abs(n_i - n_j) |
| `ged_normalized` | float | GED / (0.5 * (n_i + n_j)) |
| `levenshtein_normalized` | float | Levenshtein / (0.5 * (canonical_len_i + canonical_len_j)) |

### 3.3 Statistical Results JSON

```json
{
    "dataset": "iam_letter_low",
    "n_graphs": 750,
    "n_valid_pairs": 280875,
    "n_unique_canonical": 523,
    "n_isomorphic_pairs": 12450,
    "global": {
        "pearson_r": 0.84,
        "pearson_p": 1.2e-50,
        "pearson_95ci": [0.83, 0.85],
        "spearman_rho": 0.79,
        "spearman_p": 3.1e-45,
        "spearman_95ci": [0.78, 0.80],
        "kendall_tau": 0.62,
        "kendall_p": 1.5e-40,
        "lin_ccc_raw": 0.45,
        "lin_ccc_znorm": 0.78,
        "mantel_r": 0.79,
        "mantel_p_perm": 0.0001,
        "mantel_n_perms": 9999,
        "cohens_d_same_vs_diff_class": 1.23,
        "precision_at_5": 0.72,
        "precision_at_10": 0.68,
        "precision_at_20": 0.65,
        "ols_slope": 1.32,
        "ols_intercept": 0.45,
        "ols_r_squared": 0.71
    },
    "within_class": {
        "pearson_r": 0.91,
        "spearman_rho": 0.88,
        "n_pairs": 35000
    },
    "between_class": {
        "pearson_r": 0.76,
        "spearman_rho": 0.72,
        "n_pairs": 245875
    },
    "by_size_bin": [
        {"bin": "3-4 nodes", "n_pairs": 50000, "spearman_rho": 0.82},
        {"bin": "5-6 nodes", "n_pairs": 150000, "spearman_rho": 0.80},
        {"bin": "7-9 nodes", "n_pairs": 80875, "spearman_rho": 0.75}
    ]
}
```

---

## 4. Implementation Plan

### 4.1 Module Structure

```
benchmarks/eval_correlation/
    __init__.py
    eval_correlation.py         # CLI orchestrator
    correlation_metrics.py      # All statistical functions
    README.md
```

### 4.2 Core Statistical Functions (`correlation_metrics.py`)

```python
"""Statistical functions for Levenshtein–GED correlation analysis.

All functions operate on vectorized upper-triangular distance pairs,
not full matrices, to avoid redundancy.

References:
    - Mantel (1967), Cancer Research 27:209-220.
    - Lin (1989), Biometrics 45:255-268.
    - Demšar (2006), JMLR 7:1-30.
"""

import numpy as np
from scipy import stats
from typing import NamedTuple

class CorrelationResult(NamedTuple):
    """Result of a correlation analysis."""
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_pairs: int
    method: str


def extract_upper_tri(matrix: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Extract upper-triangular entries from a symmetric matrix.
    
    Args:
        matrix: Symmetric N×N matrix.
        valid_mask: Optional N×N boolean mask (True = valid pair).
        
    Returns:
        1D array of upper-triangular values where both matrices have valid entries.
    """
    ...


def mantel_test(
    D1: np.ndarray,
    D2: np.ndarray,
    method: str = "spearman",
    n_permutations: int = 9999,
    seed: int = 42,
) -> dict:
    """Mantel test for matrix correlation.
    
    Computes correlation between vectorized upper-triangular entries
    of two distance matrices, with permutation-based significance.
    
    Args:
        D1: First distance matrix (N×N, symmetric).
        D2: Second distance matrix (N×N, symmetric).
        method: "pearson" or "spearman".
        n_permutations: Number of permutations for p-value.
        seed: Random seed for reproducibility.
        
    Returns:
        Dict with observed_r, p_value, permutation_distribution.
    """
    ...


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> CorrelationResult:
    """Bootstrap confidence interval for correlation.
    
    Uses BCa (bias-corrected and accelerated) method.
    
    Important: For large N (>100,000 pairs), subsample to 50,000
    for bootstrap efficiency without losing CI precision.
    """
    ...


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient.
    
    CCC = 2 * r * σ_x * σ_y / (σ_x² + σ_y² + (μ_x - μ_y)²)
    
    where r is Pearson correlation, σ is standard deviation, μ is mean.
    
    Reference: Lin (1989), Biometrics 45:255-268.
    """
    ...


def precision_at_k(
    D_true: np.ndarray,
    D_proxy: np.ndarray,
    k_values: list[int] = [5, 10, 20],
) -> dict[int, float]:
    """Precision@k: fraction of true k-NN preserved by proxy metric.
    
    For each graph i, find its k nearest neighbors under D_true,
    and check how many are also in its k nearest neighbors under D_proxy.
    Average over all graphs.
    
    Args:
        D_true: Ground truth distance matrix (GED).
        D_proxy: Proxy distance matrix (Levenshtein).
        k_values: List of k values to evaluate.
        
    Returns:
        Dict mapping k -> precision.
    """
    ...


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size (pooled standard deviation variant).
    
    d = (mean_1 - mean_2) / s_pooled
    where s_pooled = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))
    """
    ...


def jonckheere_terpstra(groups: list[np.ndarray]) -> dict:
    """Jonckheere-Terpstra test for ordered alternatives.
    
    Tests H₀: medians are equal vs H₁: medians are ordered.
    
    Args:
        groups: List of arrays [group_1, group_2, ..., group_k]
                ordered from expected smallest to largest.
    
    Returns:
        Dict with statistic, p_value, z_score.
    """
    ...
```

### 4.3 Analysis Pipeline (`eval_correlation.py`)

The main script does the following **per dataset**:

**Step 1: Load precomputed data**
```python
ged_data = np.load(f"data/eval/ged_matrices/{dataset}.npz", allow_pickle=True)
lev_data = np.load(f"data/eval/levenshtein_matrices/{dataset}.npz")
with open(f"data/eval/graph_metadata/{dataset}.json") as f:
    metadata = json.load(f)
with open(f"data/eval/canonical_strings/{dataset}.json") as f:
    canonical = json.load(f)
```

**Step 2: Build pair-level DataFrame**
```python
# Extract upper-triangular pairs where both GED and Levenshtein are finite
ged_matrix = ged_data["ged_matrix"]
lev_matrix = lev_data["levenshtein_matrix"]
valid = np.isfinite(ged_matrix) & (ged_matrix >= 0)
# Build DataFrame with all columns specified in 3.2
```

**Step 3: Compute correlation metrics**
- Mantel test (Spearman, 9999 permutations)
- Pearson r with bootstrap 95% CI (10,000 resamples)
- Spearman ρ with bootstrap 95% CI
- Kendall τ
- Lin's CCC (raw and z-normalized)
- Precision@k for k ∈ {5, 10, 20}
- OLS regression: Levenshtein ~ GED (slope, intercept, R²)
- Cohen's d: same-class vs between-class Levenshtein distances

**Step 4: Within-class and between-class analysis**
- Separate pairs by `same_class` flag
- Compute Spearman ρ within each subset
- Compute Cohen's d

**Step 5: Size-stratified analysis**
- Bin pairs by max(n_i, n_j): [3-4], [5-6], [7-9], [10-12], [13+]
- Compute Spearman ρ per bin

**Step 6: Save all raw data and statistics**

Then, **across datasets**:

**Step 7: Cross-distortion analysis (H₂)**
- For IAM Letter LOW/MED/HIGH, compute Spearman ρ per distortion level
- Run Jonckheere-Terpstra test on [ρ_LOW, ρ_MED, ρ_HIGH]
- Save results to `cross_distortion_analysis.json`

**Step 8: Summary comparison**
- Table with one row per dataset, columns: N_graphs, N_pairs, Spearman ρ, Pearson r, CCC, P@10
- Highlight best/worst datasets

### 4.4 Visualization

#### Figure 1: Main Paper Figure (`correlation_main_figure.pdf`)

**Layout**: 2 rows × 3 columns, full-page width (7.5" × 5").

| Panel | Content |
|-------|---------|
| (a) | Scatter: Levenshtein vs GED for IAM Letter LOW. Hexbin density + OLS regression line. Annotate: Spearman ρ, Pearson r, n_pairs. |
| (b) | Scatter: Levenshtein vs GED for LINUX. Same format. |
| (c) | Scatter: Levenshtein vs GED for ALKANE. Same format. |
| (d) | Bar chart: Spearman ρ per dataset (all 5), with 95% CI error bars. Horizontal dashed line at ρ=0. Color by domain (handwriting blue, software green, molecular orange). |
| (e) | Line plot: Spearman ρ vs distortion level (LOW→MED→HIGH), with 95% CI band. Annotate Jonckheere-Terpstra p-value. |
| (f) | Grouped bar: Precision@k for k=5,10,20, grouped by dataset. |

Use `matplotlib` with publication-quality settings:
```python
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
```

Import project-level style if available at `benchmarks/plotting_styles.py`.

#### Figure 2: Bland-Altman Plots (`bland_altman_{dataset}.pdf`)

One per dataset. X-axis: mean of (GED, Levenshtein) per pair. Y-axis: difference (Levenshtein − GED). Horizontal lines at mean difference ± 1.96×SD. This reveals systematic bias, proportional bias, and heteroscedasticity (Bland & Altman, 1986).

**Important**: Before plotting Bland-Altman, z-normalize both distance vectors to make them comparable in scale (since Levenshtein and GED have different scales).

#### Figure 3: Heatmap Comparison (`heatmap_comparison_{dataset}.pdf`)

Side-by-side heatmaps of GED matrix and Levenshtein matrix for a **subset** (e.g., first 100 graphs sorted by class label). Shared color scale (after min-max normalization). Hierarchical clustering on rows/columns using GED to order both heatmaps identically. This visually demonstrates that the two matrices have similar structure.

### 4.5 Tables

#### Table 1: Correlation Summary (`correlation_summary.tex`)

| Dataset | N | Pairs | Spearman ρ [95% CI] | Pearson r [95% CI] | Kendall τ | CCC | P@10 | Mantel p |
|---------|---|-------|---------------------|--------------------|-----------|----|------|----------|
| IAM LOW | 750 | 280,875 | 0.79 [0.78, 0.80] | 0.84 [0.83, 0.85] | 0.62 | 0.45 | 0.68 | <0.001 |
| ... | | | | | | | | |

#### Table 2: Precision@k (`precision_at_k.tex`)

| Dataset | P@5 | P@10 | P@20 |
|---------|-----|------|------|
| IAM LOW | 0.72 | 0.68 | 0.65 |
| ... | | | |

---

## 5. CLI Interface

```bash
# Full analysis (all datasets)
python -m benchmarks.eval_correlation.eval_correlation \
    --data-root data/eval \
    --output-dir results/eval_correlation \
    --seed 42 \
    --n-bootstrap 10000 \
    --n-permutations 9999 \
    --csv --plot --table

# Single dataset (debugging)
python -m benchmarks.eval_correlation.eval_correlation \
    --data-root data/eval \
    --output-dir /tmp/corr_test \
    --datasets iam_letter_low \
    --n-bootstrap 1000 \
    --n-permutations 999 \
    --csv --plot --table

# Picasso mode
python -m benchmarks.eval_correlation.eval_correlation \
    --data-root data/eval \
    --output-dir $RESULTS_DIR/eval_correlation \
    --mode picasso \
    --seed 42
```

---

## 6. Local Testing Plan

### 6.1 Smoke Test (< 2 min)

```python
# Test with tiny synthetic data
def test_correlation_pipeline():
    # Create 20 random graphs (path, star, cycle, complete, tree)
    # Compute GED manually (small graphs, instant)
    # Compute canonical strings
    # Compute Levenshtein matrix
    # Run full correlation pipeline
    # Verify: Spearman ρ > 0, Mantel p < 0.05
    ...

def test_mantel_test():
    # Create two correlated distance matrices
    D1 = np.random.rand(10, 10); D1 = (D1 + D1.T) / 2; np.fill_diagonal(D1, 0)
    D2 = D1 + np.random.rand(10, 10) * 0.1; D2 = (D2 + D2.T) / 2; np.fill_diagonal(D2, 0)
    result = mantel_test(D1, D2, n_permutations=999)
    assert result["observed_r"] > 0.9
    assert result["p_value"] < 0.01
    ...

def test_lins_ccc():
    # Perfect agreement: CCC = 1
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    assert abs(lins_ccc(x, x) - 1.0) < 1e-10
    # Perfect correlation but different scale: CCC < 1
    assert lins_ccc(x, 2*x) < 1.0
    ...

def test_precision_at_k():
    # Identical matrices: P@k = 1.0 for all k
    D = np.random.rand(20, 20); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    result = precision_at_k(D, D, k_values=[5, 10])
    assert result[5] == 1.0
    assert result[10] == 1.0
    ...
```

### 6.2 Integration Test (< 10 min)

```bash
# Run on IAM Letter LOW only with reduced bootstrap
python -m benchmarks.eval_correlation.eval_correlation \
    --data-root data/eval \
    --output-dir /tmp/corr_test \
    --datasets iam_letter_low \
    --n-bootstrap 500 \
    --n-permutations 499 \
    --csv --plot --table

# Verify outputs exist
ls /tmp/corr_test/raw/iam_letter_low_pair_data.csv
ls /tmp/corr_test/stats/iam_letter_low_correlation_stats.json
ls /tmp/corr_test/figures/correlation_scatter_iam_letter_low.pdf
```

---

## 7. Methodological Notes

### 7.1 Handling Many Tied Values

LINUX has ~8.9% unique graphs (Jain et al., 2024), meaning many GED=0 pairs. This causes:
- Inflated Spearman ρ (many tied ranks).
- Kendall τ is more robust to ties — always report it alongside Spearman.
- Precision@k may be artificially high near k=1 if many graphs are identical.

**Recommendation**: Report correlation both on the full pair set AND on the subset excluding GED=0 pairs (non-isomorphic pairs only). The latter tests whether IsalGraph distinguishes degrees of structural difference, not just identity.

### 7.2 Multiple Comparisons

We compute correlation on 5 datasets. Apply **Holm-Bonferroni correction** to the 5 Mantel test p-values. Report both raw and adjusted p-values. With expected p < 0.001 on each, the correction is unlikely to change conclusions, but scientific rigor demands it.

### 7.3 Subsample for Bootstrap Efficiency

For IAM Letter with ~281,000 pairs, computing 10,000 bootstrap resamples of the full pair vector is expensive (280K × 10K = 2.8 billion operations). Instead:
- Compute point estimates on full data (exact Spearman, exact Pearson).
- For bootstrap CI, subsample 50,000 pairs uniformly. The CI width scales as O(1/√n), so 50K pairs give tight CIs.
- Report that bootstrap was computed on a subsample with the subsample size.

### 7.4 Expected Results

From existing benchmark on synthetic graphs (max_nodes=7, 85 pairs): Pearson r = 0.83, Spearman ρ = 0.59.

On real datasets we expect:
- IAM Letter LOW: high ρ (sanity check, low distortion, near-isomorphic within-class).
- IAM Letter HIGH: lower ρ (more structural noise).
- LINUX: moderate ρ (many duplicates inflate it, but genuine structural variation exists).
- ALKANE: highest ρ (all trees, Akutsu theory applies).

---

## 8. Acceptance Criteria

1. ✅ Pair-level CSV exists for all 5 datasets with correct column count.
2. ✅ Statistical results JSON exists for all 5 datasets.
3. ✅ Cross-distortion analysis JSON exists with Jonckheere-Terpstra results.
4. ✅ Main figure (6-panel) is generated in PDF format.
5. ✅ Bland-Altman plots generated for all 5 datasets.
6. ✅ LaTeX correlation summary table generated.
7. ✅ All Mantel test p-values < 0.05 (if the method works at all).
8. ✅ Bootstrap CIs are plausible (not [0,0] or [1,1]).
9. ✅ Code passes `ruff check`.
