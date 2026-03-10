# ruff: noqa: E402
"""Performance comparison table across datasets and encoding methods.

Generates a LaTeX table with:
- Dataset properties: N (graphs), valid pairs, mean edges (proxy for complexity)
- Spearman ρ with significance stars and Δ from best method per dataset

The proxy \\bar{m} (mean edges per graph) monotonically tracks ρ degradation:
larger, denser graphs produce longer instruction strings, making
Levenshtein distance a less faithful approximation to GED.

Usage:
    python -m benchmarks.eval_visualizations.table_performance_summary \
        --data-root /media/mpascual/Sandisk2TB/research/isalgraph/data/eval \
        --stats-dir /media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_correlation/stats \
        --output-dir /media/mpascual/Sandisk2TB/research/isalgraph/results/figures/tables
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = ["iam_letter_low", "iam_letter_med", "iam_letter_high", "linux", "aids"]
DATASET_DISPLAY = {
    "iam_letter_low": "IAM LOW",
    "iam_letter_med": "IAM MED",
    "iam_letter_high": "IAM HIGH",
    "linux": "LINUX",
    "aids": "AIDS",
}

METHODS = ["exhaustive", "greedy", "greedy_single"]
METHOD_DISPLAY = {
    "exhaustive": "Canonical",
    "greedy": "Greedy-Min",
    "greedy_single": r"Greedy-rnd($v_0$)",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DatasetProps:
    """Dataset-level properties for one dataset."""

    name: str
    n_graphs: int
    n_valid_pairs: int
    mean_edges: float
    mean_density: float


@dataclass
class MethodResult:
    """Spearman ρ result for one (dataset, method) pair."""

    dataset: str
    method: str
    rho: float
    p_value: float
    n_pairs: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_dataset_props(data_root: str) -> dict[str, DatasetProps]:
    """Load graph properties from GED matrix NPZ files."""
    props = {}
    for ds in DATASETS:
        ged_path = os.path.join(data_root, "ged_matrices", f"{ds}.npz")
        if not os.path.isfile(ged_path):
            logger.warning("GED matrix not found: %s", ged_path)
            continue

        data = np.load(ged_path, allow_pickle=True)
        ged_matrix = data["ged_matrix"]
        nc = data["node_counts"]
        ec = data["edge_counts"]
        density = 2 * ec / (nc * (nc - 1) + 1e-10)

        n = len(nc)
        # Count valid pairs (finite GED)
        iu = np.triu_indices(n, k=1)
        ged_upper = ged_matrix[iu]
        n_valid = int(np.sum(np.isfinite(ged_upper)))

        props[ds] = DatasetProps(
            name=ds,
            n_graphs=n,
            n_valid_pairs=n_valid,
            mean_edges=float(np.mean(ec)),
            mean_density=float(np.mean(density)),
        )
    return props


def _load_correlation_from_json(
    stats_dir: str,
    dataset: str,
    method: str,
) -> MethodResult | None:
    """Load Spearman ρ from pre-computed correlation stats JSON."""
    # Method mapping: 'exhaustive' → file has 'exhaustive', 'greedy' → 'greedy'
    path = os.path.join(stats_dir, f"{dataset}_{method}_correlation_stats.json")
    if not os.path.isfile(path):
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    spearman = raw.get("spearman", {})
    return MethodResult(
        dataset=dataset,
        method=method,
        rho=spearman.get("statistic", 0.0),
        p_value=spearman.get("p_value", 1.0),
        n_pairs=raw.get("n_valid_pairs", 0),
    )


def _compute_spearman_from_matrices(
    data_root: str,
    dataset: str,
    method: str,
) -> MethodResult | None:
    """Compute Spearman ρ directly from GED and Levenshtein matrices."""
    ged_path = os.path.join(data_root, "ged_matrices", f"{dataset}.npz")
    lev_path = os.path.join(data_root, "levenshtein_matrices", f"{dataset}_{method}.npz")

    if not os.path.isfile(ged_path) or not os.path.isfile(lev_path):
        logger.warning("Matrix not found: %s or %s", ged_path, lev_path)
        return None

    ged_data = np.load(ged_path, allow_pickle=True)
    lev_data = np.load(lev_path, allow_pickle=True)

    ged_matrix = ged_data["ged_matrix"]
    lev_matrix = lev_data["levenshtein_matrix"]

    n = ged_matrix.shape[0]
    if lev_matrix.shape[0] != n:
        logger.warning(
            "Matrix size mismatch for %s_%s: GED=%d, Lev=%d",
            dataset,
            method,
            n,
            lev_matrix.shape[0],
        )
        return None

    # Extract upper triangle
    iu = np.triu_indices(n, k=1)
    ged_vals = ged_matrix[iu].astype(float)
    lev_vals = lev_matrix[iu].astype(float)

    # Filter to valid pairs (both finite)
    valid = np.isfinite(ged_vals) & np.isfinite(lev_vals)
    ged_valid = ged_vals[valid]
    lev_valid = lev_vals[valid]

    if len(ged_valid) < 10:
        logger.warning("Too few valid pairs for %s_%s: %d", dataset, method, len(ged_valid))
        return None

    result = sp_stats.spearmanr(ged_valid, lev_valid)
    return MethodResult(
        dataset=dataset,
        method=method,
        rho=float(result.statistic),
        p_value=float(result.pvalue),
        n_pairs=int(np.sum(valid)),
    )


def _significance_stars(p: float) -> str:
    """Map p-value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _format_number(n: int) -> str:
    """Format integer with LaTeX-safe thousands separators."""
    s = f"{n:,}"
    return s.replace(",", "{,}")


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def generate_performance_table(
    data_root: str,
    stats_dir: str,
    output_dir: str,
) -> str:
    """Generate the performance comparison LaTeX table.

    Args:
        data_root: Root of eval data (ged_matrices/, levenshtein_matrices/).
        stats_dir: Directory with pre-computed correlation stats JSONs.
        output_dir: Directory to save the .tex file.

    Returns:
        Path to the saved .tex file.
    """
    # --- Load dataset properties ---
    props = _load_dataset_props(data_root)

    # --- Load / compute Spearman ρ per (dataset, method) ---
    results: dict[tuple[str, str], MethodResult] = {}

    for ds in DATASETS:
        for method in METHODS:
            if method in ("exhaustive", "greedy"):
                # Use pre-computed JSON
                r = _load_correlation_from_json(stats_dir, ds, method)
            else:
                # Compute from matrices (greedy_single)
                r = _compute_spearman_from_matrices(data_root, ds, method)

            if r is not None:
                results[(ds, method)] = r
                logger.info(
                    "%s / %s: ρ=%.4f, p=%.2e, pairs=%d",
                    ds,
                    method,
                    r.rho,
                    r.p_value,
                    r.n_pairs,
                )

    # --- Determine best ρ per dataset ---
    best_rho: dict[str, float] = {}
    for ds in DATASETS:
        rhos = [results[(ds, m)].rho for m in METHODS if (ds, m) in results]
        best_rho[ds] = max(rhos) if rhos else 0.0

    # --- Build LaTeX table ---
    n_ds = len(DATASETS)
    col_spec = "cl" + "c" * n_ds

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Dataset properties and Spearman $\rho$ correlation between GED and "
        r"IsalGraph Levenshtein distance across encoding methods. "
        r"$\bar{m}$: mean edges per graph (complexity proxy). "
        r"$\Delta$: difference from best method per dataset. "
        r"Best $\rho$ per dataset in \textbf{bold}.}",
        r"\label{tab:performance-summary}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    header_cells = ["", ""]
    for ds in DATASETS:
        header_cells.append(rf"\textbf{{{DATASET_DISPLAY[ds]}}}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # --- Properties block ---
    # Row: N
    row_n = [r"\multirow{3}{*}{\rotatebox[origin=c]{90}{\scriptsize Prop.}}", "$N$"]
    for ds in DATASETS:
        if ds in props:
            row_n.append(_format_number(props[ds].n_graphs))
        else:
            row_n.append("--")
    lines.append(" & ".join(row_n) + r" \\")

    # Row: Pairs
    row_pairs = ["", "Pairs"]
    for ds in DATASETS:
        if ds in props:
            row_pairs.append(_format_number(props[ds].n_valid_pairs))
        else:
            row_pairs.append("--")
    lines.append(" & ".join(row_pairs) + r" \\")

    # Row: mean edges (proxy)
    row_proxy = ["", r"$\bar{m}$"]
    for ds in DATASETS:
        if ds in props:
            row_proxy.append(f"{props[ds].mean_edges:.2f}")
        else:
            row_proxy.append("--")
    lines.append(" & ".join(row_proxy) + r" \\")

    lines.append(r"\midrule")

    # --- Spearman ρ block ---
    for i, method in enumerate(METHODS):
        if i == 0:
            prefix = r"\multirow{3}{*}{\rotatebox[origin=c]{90}{\small Spearman $\rho$}}"
        else:
            prefix = ""

        row = [prefix, METHOD_DISPLAY[method]]

        for ds in DATASETS:
            key = (ds, method)
            if key not in results:
                row.append("--")
                continue

            r = results[key]
            stars = _significance_stars(r.p_value)
            delta = r.rho - best_rho[ds]
            is_best = abs(delta) < 1e-6

            # Format ρ value
            rho_str = f"{r.rho:.3f}"
            if is_best:
                rho_str = rf"\textbf{{{rho_str}}}"

            # Format delta (omit for best method)
            if is_best:
                cell = rf"{rho_str}$^{{{stars}}}$"
            else:
                delta_str = f"{delta:+.3f}"
                cell = rf"{rho_str}$^{{{stars}}}$ ({delta_str})"
            row.append(cell)

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\vspace{1mm}\par\footnotesize "
        r"$^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$. "
        r"$\Delta = \rho_{\mathrm{method}} - \rho_{\mathrm{best}}$. "
        r"$\bar{m}$ increases monotonically as $\rho$ degrades across datasets."
    )
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "table_performance_summary.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    logger.info("Table saved to %s", output_path)

    # Also print to stdout for inspection
    print("\n" + "=" * 80)
    print("GENERATED LaTeX TABLE")
    print("=" * 80)
    print(latex)
    print("=" * 80 + "\n")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance comparison LaTeX table.")
    parser.add_argument(
        "--data-root",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/data/eval",
        help="Root directory for eval data.",
    )
    parser.add_argument(
        "--stats-dir",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_correlation/stats",
        help="Directory with correlation stats JSONs.",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/mpascual/Sandisk2TB/research/isalgraph/results/figures/tables",
        help="Output directory for the .tex file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_performance_table(args.data_root, args.stats_dir, args.output_dir)


if __name__ == "__main__":
    main()
