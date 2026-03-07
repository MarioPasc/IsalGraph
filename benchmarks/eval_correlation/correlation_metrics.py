"""Pure statistical functions for Levenshtein-GED correlation analysis.

No I/O, no logging side effects. All functions take numpy arrays
and return dataclasses or dicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class CorrelationResult:
    """Result of a correlation test with bootstrap CI."""

    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_pairs: int
    method: str


@dataclass
class MantelResult:
    """Result of a Mantel permutation test."""

    observed_r: float
    p_value: float
    n_permutations: int
    n_more_extreme: int
    permutation_mean: float
    permutation_std: float


@dataclass
class TrendTestResult:
    """Result of a Jonckheere-Terpstra trend test."""

    statistic: float
    p_value: float
    z_score: float
    is_monotone: bool
    group_medians: list[float]


@dataclass
class PrecisionAtKResult:
    """Precision@k for neighbor retrieval."""

    k: int
    precision: float
    n_graphs: int


# =============================================================================
# Core functions
# =============================================================================


def extract_upper_tri(
    *matrices: np.ndarray,
    mask_inf: bool = True,
    sentinel: float = -1.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract aligned upper-triangle vectors from square matrices.

    Args:
        matrices: One or more square matrices of the same shape.
        mask_inf: If True, exclude entries where any matrix has inf or sentinel.
        sentinel: Value to treat as invalid (default -1.0).

    Returns:
        (vectors, valid_mask) where vectors[i] is the masked upper-triangle
        of matrices[i] and valid_mask is a boolean array over the full
        upper-triangle.
    """
    n = matrices[0].shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vectors = [m[triu_idx].astype(np.float64) for m in matrices]

    if mask_inf:
        valid = np.ones(len(vectors[0]), dtype=bool)
        for v in vectors:
            valid &= np.isfinite(v) & (v >= 0) & (v != sentinel)
        vectors = [v[valid] for v in vectors]
        return vectors, valid

    return vectors, np.ones(len(vectors[0]), dtype=bool)


def mantel_test(
    d1: np.ndarray,
    d2: np.ndarray,
    method: str = "spearman",
    n_permutations: int = 9999,
    seed: int = 42,
) -> MantelResult:
    """Permutation-based matrix correlation test (Mantel test).

    For Spearman, pre-ranks vectors so each permutation only needs
    a Pearson correlation on ranks.

    Args:
        d1: First distance matrix (square, symmetric).
        d2: Second distance matrix (same shape as d1).
        method: "spearman" or "pearson".
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        MantelResult with observed correlation and permutation p-value.
    """
    n = d1.shape[0]
    rng = np.random.default_rng(seed)
    triu_idx = np.triu_indices(n, k=1)

    # Build validity mask across both matrices
    v1_full = d1[triu_idx].astype(np.float64)
    v2_full = d2[triu_idx].astype(np.float64)
    valid = np.isfinite(v1_full) & np.isfinite(v2_full) & (v1_full >= 0) & (v2_full >= 0)

    v1 = v1_full[valid]
    v2 = v2_full[valid]

    if len(v1) < 3:
        return MantelResult(
            observed_r=np.nan,
            p_value=1.0,
            n_permutations=n_permutations,
            n_more_extreme=n_permutations,
            permutation_mean=0.0,
            permutation_std=0.0,
        )

    # Compute observed correlation
    if method == "spearman":
        observed_r = float(stats.spearmanr(v1, v2).statistic)
    else:
        observed_r = float(stats.pearsonr(v1, v2).statistic)

    # Permutation distribution
    # For each permutation, permute rows+cols of d1, re-extract upper tri
    perm_rs = np.empty(n_permutations)
    for i in range(n_permutations):
        sigma = rng.permutation(n)
        d1_perm = d1[np.ix_(sigma, sigma)]
        v1_perm = d1_perm[triu_idx].astype(np.float64)[valid]
        if method == "spearman":
            perm_rs[i] = stats.spearmanr(v1_perm, v2).statistic
        else:
            perm_rs[i] = stats.pearsonr(v1_perm, v2).statistic

    n_more_extreme = int(np.sum(perm_rs >= observed_r))
    p_value = (n_more_extreme + 1) / (n_permutations + 1)

    return MantelResult(
        observed_r=observed_r,
        p_value=p_value,
        n_permutations=n_permutations,
        n_more_extreme=n_more_extreme,
        permutation_mean=float(np.mean(perm_rs)),
        permutation_std=float(np.std(perm_rs)),
    )


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    max_subsample: int = 50000,
    seed: int = 42,
) -> CorrelationResult:
    """Percentile bootstrap CI for a correlation statistic.

    If len(x) > max_subsample, randomly subsamples pairs for bootstrap
    (but computes point estimate on full data).

    Args:
        x: First array.
        y: Second array (same length).
        method: "spearman", "pearson", or "kendall".
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (default 0.95).
        max_subsample: Maximum pairs for bootstrap resampling.
        seed: Random seed.

    Returns:
        CorrelationResult with point estimate and CI.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Point estimate on full data
    if method == "spearman":
        res = stats.spearmanr(x, y)
        point = float(res.statistic)
        p_val = float(res.pvalue)
    elif method == "pearson":
        res = stats.pearsonr(x, y)
        point = float(res.statistic)
        p_val = float(res.pvalue)
    elif method == "kendall":
        res = stats.kendalltau(x, y)
        point = float(res.statistic)
        p_val = float(res.pvalue)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Subsample for bootstrap if too large
    n = len(x)
    if n > max_subsample:
        idx = rng.choice(n, size=max_subsample, replace=False)
        x_boot = x[idx]
        y_boot = y[idx]
        n_boot = max_subsample
    else:
        x_boot = x
        y_boot = y
        n_boot = n

    alpha = 1.0 - ci_level
    boot_stats = np.empty(n_bootstrap)

    corr_func = {
        "spearman": lambda a, b: stats.spearmanr(a, b).statistic,
        "pearson": lambda a, b: stats.pearsonr(a, b).statistic,
        "kendall": lambda a, b: stats.kendalltau(a, b).statistic,
    }[method]

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_boot, size=n_boot)
        boot_stats[i] = corr_func(x_boot[idx], y_boot[idx])

    ci_lower = float(np.nanpercentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))

    return CorrelationResult(
        statistic=point,
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_pairs=n,
        method=method,
    )


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient.

    CCC = 2 * r * s_x * s_y / (s_x^2 + s_y^2 + (mu_x - mu_y)^2)

    Args:
        x: First array.
        y: Second array.

    Returns:
        CCC value in [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu_x, mu_y = np.mean(x), np.mean(y)
    s_x, s_y = np.std(x, ddof=1), np.std(y, ddof=1)
    r = float(np.corrcoef(x, y)[0, 1])

    if s_x == 0 and s_y == 0:
        return 1.0 if mu_x == mu_y else 0.0

    numerator = 2 * r * s_x * s_y
    denominator = s_x**2 + s_y**2 + (mu_x - mu_y) ** 2
    return float(numerator / denominator) if denominator > 0 else 0.0


def precision_at_k(
    d_true: np.ndarray,
    d_proxy: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[int, PrecisionAtKResult]:
    """Fraction of true k-nearest neighbors preserved by proxy metric.

    Handles inf by excluding from neighbor candidates.
    Excludes graphs with fewer than k valid neighbors.

    Args:
        d_true: True distance matrix (square).
        d_proxy: Proxy distance matrix (same shape).
        k_values: List of k values to evaluate.

    Returns:
        Dict mapping k -> PrecisionAtKResult.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    n = d_true.shape[0]
    results: dict[int, PrecisionAtKResult] = {}

    for k in k_values:
        precisions = []
        for i in range(n):
            # Valid neighbors: finite in both matrices, not self
            valid = np.isfinite(d_true[i]) & np.isfinite(d_proxy[i])
            valid[i] = False
            valid_idx = np.where(valid)[0]

            if len(valid_idx) < k:
                continue

            # True k-NN
            true_dists = d_true[i, valid_idx]
            true_knn = set(valid_idx[np.argsort(true_dists)[:k]])

            # Proxy k-NN
            proxy_dists = d_proxy[i, valid_idx]
            proxy_knn = set(valid_idx[np.argsort(proxy_dists)[:k]])

            precisions.append(len(true_knn & proxy_knn) / k)

        results[k] = PrecisionAtKResult(
            k=k,
            precision=float(np.mean(precisions)) if precisions else 0.0,
            n_graphs=len(precisions),
        )

    return results


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Pooled-variance Cohen's d.

    Args:
        group1: First group values.
        group2: Second group values.

    Returns:
        Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def jonckheere_terpstra(
    groups: list[np.ndarray],
    alternative: str = "decreasing",
) -> TrendTestResult:
    """Jonckheere-Terpstra test for ordered alternatives.

    Uses Mann-Whitney U counts for all ordered pairs of groups.
    Normal approximation for p-value (valid for group sizes > 10).

    Args:
        groups: Ordered list of sample arrays.
        alternative: "decreasing" (expect groups[0] > groups[1] > ...)
                     or "increasing".

    Returns:
        TrendTestResult with JT statistic, z-score, and p-value.
    """
    k = len(groups)
    ns = [len(g) for g in groups]
    n_total = sum(ns)

    # JT statistic: sum of U_ij for all i < j
    jt_stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            if alternative == "decreasing":
                # Count how many in group[i] > group[j]
                u_stat = float(
                    stats.mannwhitneyu(groups[i], groups[j], alternative="greater").statistic
                )
            else:
                u_stat = float(
                    stats.mannwhitneyu(groups[i], groups[j], alternative="less").statistic
                )
            jt_stat += u_stat

    # Expected value and variance under H0 (normal approximation)
    # E[J] = (N^2 - sum(n_i^2)) / 4
    e_j = (n_total**2 - sum(ni**2 for ni in ns)) / 4.0

    # Var[J] = (N^2(2N+3) - sum(n_i^2(2n_i+3))) / 72
    var_j = (n_total**2 * (2 * n_total + 3) - sum(ni**2 * (2 * ni + 3) for ni in ns)) / 72.0

    if var_j <= 0:
        return TrendTestResult(
            statistic=jt_stat,
            p_value=1.0,
            z_score=0.0,
            is_monotone=False,
            group_medians=[float(np.median(g)) for g in groups],
        )

    z = (jt_stat - e_j) / np.sqrt(var_j)
    # One-sided test
    p_value = float(stats.norm.sf(z))

    medians = [float(np.median(g)) for g in groups]
    is_monotone = all(medians[i] >= medians[i + 1] for i in range(len(medians) - 1))
    if alternative == "increasing":
        is_monotone = all(medians[i] <= medians[i + 1] for i in range(len(medians) - 1))

    return TrendTestResult(
        statistic=jt_stat,
        p_value=p_value,
        z_score=float(z),
        is_monotone=is_monotone,
        group_medians=medians,
    )


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of raw p-values.

    Returns:
        List of adjusted p-values (same order as input).
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort by p-value, keeping original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cummax = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (m - rank)
        cummax = max(cummax, corrected)
        adjusted[orig_idx] = min(cummax, 1.0)

    return adjusted


def ols_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Ordinary least squares regression via scipy.stats.linregress.

    Args:
        x: Independent variable.
        y: Dependent variable.

    Returns:
        Dict with slope, intercept, r_squared, p_value, stderr.
    """
    res = stats.linregress(x, y)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r_squared": float(res.rvalue**2),
        "r_value": float(res.rvalue),
        "p_value": float(res.pvalue),
        "stderr": float(res.stderr),
    }
