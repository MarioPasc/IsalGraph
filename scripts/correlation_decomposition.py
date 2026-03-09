"""Neighborhood topology analysis: correlation decomposition experiments.

Implements Experiments B, C, and F from the neighborhood topology
analysis document. These operate on precomputed distance matrices
and require no additional encoding.

Experiments:
    B — Partial Spearman correlation controlling for graph size
    C — GED-to-Levenshtein distortion factor distribution
    F — Normalised Levenshtein correlation

Usage:
    python -m benchmarks.eval_neighborhood.correlation_decomposition \
        --data-root data/eval \
        --output-dir results/eval_neighborhood \
        --datasets iam_letter_low iam_letter_med iam_letter_high linux aids

References:
    - Conover (1999), Practical Nonparametric Statistics, 3rd ed., Wiley.
    - Öztürk et al. (2016), BMC Bioinformatics 17:128.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class PartialCorrelationResult:
    """Result of partial Spearman correlation analysis (Experiment B)."""
    dataset: str
    method: str
    rho_raw: float               # raw Spearman rho(d_L, GED)
    rho_partial: float           # Spearman rho(d_L, GED | n_bar)
    delta_rho: float             # rho_raw - rho_partial
    pct_size_driven: float       # 100 * (1 - rho_partial / rho_raw)
    rho_normlev_normged: float   # rho(d_L^norm, GED^norm)
    n_pairs: int

@dataclass
class DistortionResult:
    """Result of distortion factor analysis (Experiment C)."""
    dataset: str
    method: str
    median_ratio: float          # median(d_L / GED) for GED > 0
    iqr_ratio: tuple[float, float]
    mean_ratio: float
    std_ratio: float
    pct_underestimate: float     # % of pairs where d_L < GED
    pct_overestimate: float      # % of pairs where d_L > GED
    pct_exact: float             # % of pairs where d_L == GED
    n_pairs: int
    # Stratified by GED value: {ged_val: mean_ratio}
    ratio_by_ged: dict[int, float] = field(default_factory=dict)

@dataclass
class NormalisedCorrelationResult:
    """Result of normalised Levenshtein analysis (Experiment F)."""
    dataset: str
    method: str
    rho_raw: float
    rho_normalised_lev: float
    rho_both_normalised: float
    delta_rho_norm: float        # rho_normalised - rho_raw
    n_pairs: int


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------

def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    """Partial Spearman rank correlation rho(x, y | z).

    Computes partial rank correlation by regressing rank-transformed
    x and y on rank-transformed z and correlating the residuals
    (Conover, 1999, Section 5.6).

    Args:
        x: First variable (e.g., Levenshtein distances).
        y: Second variable (e.g., GED values).
        z: Conditioning variable (e.g., mean pair size).

    Returns:
        Partial Spearman correlation coefficient.
    """
    # Rank-transform all variables
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    # Regress ranked x on ranked z → residuals
    slope_xz, intercept_xz, _, _, _ = stats.linregress(rz, rx)
    residual_x = rx - (slope_xz * rz + intercept_xz)

    # Regress ranked y on ranked z → residuals
    slope_yz, intercept_yz, _, _, _ = stats.linregress(rz, ry)
    residual_y = ry - (slope_yz * rz + intercept_yz)

    # Pearson correlation of residuals = partial Spearman
    rho_partial, _ = stats.pearsonr(residual_x, residual_y)
    return rho_partial


def compute_distortion_ratios(
    lev_vec: np.ndarray,
    ged_vec: np.ndarray,
) -> np.ndarray:
    """Compute distortion ratios d_L / GED for pairs with GED > 0.

    Args:
        lev_vec: Levenshtein distance vector (upper triangular).
        ged_vec: GED vector (upper triangular).

    Returns:
        Array of ratios for pairs where GED > 0.
    """
    mask = ged_vec > 0
    return lev_vec[mask] / ged_vec[mask]


# ---------------------------------------------------------------------------
# Experiment implementations
# ---------------------------------------------------------------------------

def run_experiment_b(
    lev_vec: np.ndarray,
    ged_vec: np.ndarray,
    nbar_vec: np.ndarray,
    string_lengths: np.ndarray,
    node_counts: np.ndarray,
    dataset: str,
    method: str,
) -> PartialCorrelationResult:
    """Experiment B: Partial correlation controlling for graph size.

    Decomposes the raw Spearman rho into size-driven and structural
    components by computing the partial rank correlation
    rho(d_L, GED | n_bar).

    Args:
        lev_vec: Upper-triangular Levenshtein distances.
        ged_vec: Upper-triangular GED values.
        nbar_vec: Upper-triangular mean pair sizes.
        string_lengths: Per-graph canonical string lengths.
        node_counts: Per-graph node counts.
        dataset: Dataset name.
        method: Encoding method (exhaustive/greedy).

    Returns:
        PartialCorrelationResult with raw and partial correlations.
    """
    # Raw Spearman
    rho_raw, _ = stats.spearmanr(lev_vec, ged_vec)

    # Partial Spearman controlling for mean pair size
    rho_partial = partial_spearman(lev_vec, ged_vec, nbar_vec)

    # Normalised distances
    # d_L^norm = 2 * d_L / (|w_i| + |w_j|)
    n = len(node_counts)
    idx_i, idx_j = np.triu_indices(n, k=1)
    mean_string_len = (string_lengths[idx_i] + string_lengths[idx_j]) / 2.0
    lev_norm = lev_vec / np.maximum(mean_string_len, 1e-10)

    # GED^norm = GED / n_bar
    ged_norm = ged_vec / np.maximum(nbar_vec, 1e-10)

    rho_normlev_normged, _ = stats.spearmanr(lev_norm, ged_norm)

    delta = rho_raw - rho_partial
    pct = 100.0 * (1.0 - rho_partial / rho_raw) if rho_raw != 0 else 0.0

    result = PartialCorrelationResult(
        dataset=dataset,
        method=method,
        rho_raw=round(float(rho_raw), 4),
        rho_partial=round(float(rho_partial), 4),
        delta_rho=round(float(delta), 4),
        pct_size_driven=round(float(pct), 1),
        rho_normlev_normged=round(float(rho_normlev_normged), 4),
        n_pairs=len(lev_vec),
    )

    logger.info(
        f"[ExpB] {dataset}/{method}: rho_raw={result.rho_raw}, "
        f"rho_partial={result.rho_partial}, "
        f"delta={result.delta_rho} ({result.pct_size_driven}% size-driven)"
    )
    return result


def run_experiment_c(
    lev_vec: np.ndarray,
    ged_vec: np.ndarray,
    nbar_vec: np.ndarray,
    dataset: str,
    method: str,
) -> DistortionResult:
    """Experiment C: GED-to-Levenshtein distortion factor distribution.

    Characterises R_{ij} = d_L / GED for all pairs, stratified by GED
    value and mean pair size.

    Args:
        lev_vec: Upper-triangular Levenshtein distances.
        ged_vec: Upper-triangular GED values.
        nbar_vec: Upper-triangular mean pair sizes.
        dataset: Dataset name.
        method: Encoding method.

    Returns:
        DistortionResult with ratio statistics.
    """
    mask = ged_vec > 0
    ratios = lev_vec[mask] / ged_vec[mask]

    # Global statistics
    median_r = float(np.median(ratios))
    q25, q75 = float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))
    mean_r = float(np.mean(ratios))
    std_r = float(np.std(ratios))

    # Classification: under/over/exact
    pct_under = 100.0 * np.mean(lev_vec[mask] < ged_vec[mask])
    pct_over = 100.0 * np.mean(lev_vec[mask] > ged_vec[mask])
    pct_exact = 100.0 * np.mean(lev_vec[mask] == ged_vec[mask])

    # Stratified by GED value
    ged_unique = np.unique(ged_vec[mask].astype(int))
    ratio_by_ged = {}
    for g in ged_unique:
        g_mask = ged_vec[mask].astype(int) == g
        if np.sum(g_mask) >= 10:  # require minimum sample for stable mean
            ratio_by_ged[int(g)] = round(float(np.mean(ratios[g_mask])), 3)

    result = DistortionResult(
        dataset=dataset,
        method=method,
        median_ratio=round(median_r, 3),
        iqr_ratio=(round(q25, 3), round(q75, 3)),
        mean_ratio=round(mean_r, 3),
        std_ratio=round(std_r, 3),
        pct_underestimate=round(pct_under, 1),
        pct_overestimate=round(pct_over, 1),
        pct_exact=round(pct_exact, 1),
        n_pairs=int(np.sum(mask)),
        ratio_by_ged=ratio_by_ged,
    )

    logger.info(
        f"[ExpC] {dataset}/{method}: median_R={result.median_ratio}, "
        f"IQR=[{q25:.3f}, {q75:.3f}], "
        f"under={result.pct_underestimate}%, over={result.pct_overestimate}%"
    )
    return result


def run_experiment_f(
    lev_vec: np.ndarray,
    ged_vec: np.ndarray,
    string_lengths: np.ndarray,
    nbar_vec: np.ndarray,
    dataset: str,
    method: str,
) -> NormalisedCorrelationResult:
    """Experiment F: Normalised Levenshtein correlation.

    Tests whether normalising Levenshtein by string length improves
    the correlation with GED, following the normalised edit distance
    approach from Öztürk et al. (2016, BMC Bioinformatics 17:128).

    Args:
        lev_vec: Upper-triangular Levenshtein distances.
        ged_vec: Upper-triangular GED values.
        string_lengths: Per-graph canonical string lengths.
        nbar_vec: Upper-triangular mean pair sizes.
        dataset: Dataset name.
        method: Encoding method.

    Returns:
        NormalisedCorrelationResult.
    """
    n = int((1 + np.sqrt(1 + 8 * len(lev_vec))) / 2)
    idx_i, idx_j = np.triu_indices(n, k=1)

    # Normalised Levenshtein
    mean_slen = (string_lengths[idx_i] + string_lengths[idx_j]) / 2.0
    lev_norm = lev_vec / np.maximum(mean_slen, 1e-10)

    # Normalised GED
    ged_norm = ged_vec / np.maximum(nbar_vec, 1e-10)

    rho_raw, _ = stats.spearmanr(lev_vec, ged_vec)
    rho_norm_lev, _ = stats.spearmanr(lev_norm, ged_vec)
    rho_both, _ = stats.spearmanr(lev_norm, ged_norm)

    result = NormalisedCorrelationResult(
        dataset=dataset,
        method=method,
        rho_raw=round(float(rho_raw), 4),
        rho_normalised_lev=round(float(rho_norm_lev), 4),
        rho_both_normalised=round(float(rho_both), 4),
        delta_rho_norm=round(float(rho_norm_lev - rho_raw), 4),
        n_pairs=len(lev_vec),
    )

    logger.info(
        f"[ExpF] {dataset}/{method}: rho_raw={result.rho_raw}, "
        f"rho_normLev={result.rho_normalised_lev}, "
        f"rho_bothNorm={result.rho_both_normalised}, "
        f"delta={result.delta_rho_norm}"
    )
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_upper_triangular(matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangular entries as a flat vector."""
    n = matrix.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    return matrix[idx_i, idx_j].astype(np.float64)


def load_dataset(
    data_root: Path,
    dataset: str,
    method: str = "exhaustive",
) -> Optional[dict]:
    """Load GED matrix, Levenshtein matrix, and metadata for a dataset.

    Adapts to the project's eval pipeline output structure.

    Args:
        data_root: Root of eval pipeline output.
        dataset: Dataset name.
        method: Encoding method (exhaustive/greedy).

    Returns:
        Dict with lev_vec, ged_vec, nbar_vec, string_lengths,
        node_counts, or None if files missing.
    """
    ged_path = data_root / "ged_matrices" / f"{dataset}.npz"
    lev_path = data_root / "levenshtein_matrices" / f"{dataset}_{method}.npz"
    meta_path = data_root / "graph_metadata" / f"{dataset}.json"
    strings_path = data_root / "canonical_strings" / f"{dataset}_{method}.json"

    for p in [ged_path, lev_path, meta_path]:
        if not p.exists():
            logger.warning(f"Missing file: {p}")
            return None

    # Load matrices
    ged_data = np.load(ged_path)
    ged_matrix = ged_data["ged_matrix"] if "ged_matrix" in ged_data else ged_data[ged_data.files[0]]

    lev_data = np.load(lev_path)
    lev_matrix = lev_data["lev_matrix"] if "lev_matrix" in lev_data else lev_data[lev_data.files[0]]

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)

    node_counts = np.array(meta.get("node_counts", []))
    n = ged_matrix.shape[0]

    # Compute string lengths
    if strings_path.exists():
        with open(strings_path) as f:
            strings_data = json.load(f)
        # Handle both list and dict formats
        if isinstance(strings_data, list):
            string_lengths = np.array([len(s) if isinstance(s, str) else len(s.get("string", "")) for s in strings_data])
        elif isinstance(strings_data, dict) and "strings" in strings_data:
            string_lengths = np.array([len(s) if isinstance(s, str) else len(s.get("string", "")) for s in strings_data["strings"]])
        else:
            logger.warning(f"Unexpected strings format for {dataset}, estimating lengths")
            string_lengths = node_counts * 2  # rough estimate
    else:
        string_lengths = node_counts * 2

    # Ensure consistent dimensions
    assert ged_matrix.shape[0] == lev_matrix.shape[0] == len(node_counts), \
        f"Dimension mismatch: GED={ged_matrix.shape[0]}, Lev={lev_matrix.shape[0]}, meta={len(node_counts)}"

    # Compute vectors
    ged_vec = load_upper_triangular(ged_matrix)
    lev_vec = load_upper_triangular(lev_matrix)

    # Mean pair size
    idx_i, idx_j = np.triu_indices(n, k=1)
    nbar_vec = (node_counts[idx_i] + node_counts[idx_j]) / 2.0

    return {
        "lev_vec": lev_vec,
        "ged_vec": ged_vec,
        "nbar_vec": nbar_vec,
        "string_lengths": string_lengths,
        "node_counts": node_counts,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neighborhood topology: correlation decomposition (Experiments B, C, F)"
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of eval pipeline output")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output JSON results")
    parser.add_argument("--datasets", nargs="+",
                        default=["iam_letter_low", "iam_letter_med",
                                 "iam_letter_high", "linux", "aids"],
                        help="Datasets to process")
    parser.add_argument("--methods", nargs="+",
                        default=["exhaustive", "greedy"],
                        help="Encoding methods to process")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "experiment_b": [],
        "experiment_c": [],
        "experiment_f": [],
    }

    for dataset in args.datasets:
        for method in args.methods:
            logger.info(f"Processing {dataset} / {method}")
            data = load_dataset(data_root, dataset, method)
            if data is None:
                logger.warning(f"Skipping {dataset}/{method} (missing data)")
                continue

            # Experiment B: Partial correlation
            res_b = run_experiment_b(
                lev_vec=data["lev_vec"],
                ged_vec=data["ged_vec"],
                nbar_vec=data["nbar_vec"],
                string_lengths=data["string_lengths"],
                node_counts=data["node_counts"],
                dataset=dataset,
                method=method,
            )
            all_results["experiment_b"].append(asdict(res_b))

            # Experiment C: Distortion factor
            res_c = run_experiment_c(
                lev_vec=data["lev_vec"],
                ged_vec=data["ged_vec"],
                nbar_vec=data["nbar_vec"],
                dataset=dataset,
                method=method,
            )
            all_results["experiment_c"].append(asdict(res_c))

            # Experiment F: Normalised Levenshtein
            res_f = run_experiment_f(
                lev_vec=data["lev_vec"],
                ged_vec=data["ged_vec"],
                string_lengths=data["string_lengths"],
                nbar_vec=data["nbar_vec"],
                dataset=dataset,
                method=method,
            )
            all_results["experiment_f"].append(asdict(res_f))

    # Write results
    output_path = output_dir / "correlation_decomposition.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results written to {output_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("EXPERIMENT B: Partial Correlation (controlling for graph size)")
    print("=" * 90)
    print(f"{'Dataset':<20} {'Method':<12} {'rho_raw':>8} {'rho_partial':>12} "
          f"{'delta':>7} {'%size':>7}")
    print("-" * 90)
    for r in all_results["experiment_b"]:
        print(f"{r['dataset']:<20} {r['method']:<12} {r['rho_raw']:>8.4f} "
              f"{r['rho_partial']:>12.4f} {r['delta_rho']:>7.4f} "
              f"{r['pct_size_driven']:>6.1f}%")

    print("\n" + "=" * 90)
    print("EXPERIMENT C: Distortion Factor (d_L / GED)")
    print("=" * 90)
    print(f"{'Dataset':<20} {'Method':<12} {'median_R':>9} {'IQR':>16} "
          f"{'%under':>7} {'%over':>7}")
    print("-" * 90)
    for r in all_results["experiment_c"]:
        iqr_str = f"[{r['iqr_ratio'][0]:.2f}, {r['iqr_ratio'][1]:.2f}]"
        print(f"{r['dataset']:<20} {r['method']:<12} {r['median_ratio']:>9.3f} "
              f"{iqr_str:>16} {r['pct_underestimate']:>6.1f}% "
              f"{r['pct_overestimate']:>6.1f}%")

    print("\n" + "=" * 90)
    print("EXPERIMENT F: Normalised Levenshtein")
    print("=" * 90)
    print(f"{'Dataset':<20} {'Method':<12} {'rho_raw':>8} {'rho_normL':>10} "
          f"{'rho_both':>10} {'delta':>7}")
    print("-" * 90)
    for r in all_results["experiment_f"]:
        print(f"{r['dataset']:<20} {r['method']:<12} {r['rho_raw']:>8.4f} "
              f"{r['rho_normalised_lev']:>10.4f} "
              f"{r['rho_both_normalised']:>10.4f} "
              f"{r['delta_rho_norm']:>7.4f}")


if __name__ == "__main__":
    main()
