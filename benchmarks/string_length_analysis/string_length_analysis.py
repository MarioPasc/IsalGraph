"""Empirical analysis of IsalGraph string length vs graph size and density.

Tests the central compactness claim of the preprint (Lopez-Rubio 2025,
Section 2.2). The preprint derives (Eq. 9) for the matrix-pointer algorithm:

    E[|I_G|] ~ (sqrt(pi) / (2*sqrt(2))) * N^2 * sqrt(rho)

where N = node count and rho = |E| / N^2 is edge density. For large sparse
graphs, |I_G| << N^2, achieving significant compression over the binary
adjacency representation.

Our CDLL-based algorithm uses a different instruction set ({N,n,P,p,V,v,C,c,W})
and may exhibit different scaling. This benchmark empirically characterizes:

1. **String length vs N** for fixed graph families.
2. **String length vs edge count** across families.
3. **Compression ratio**: |w| / N^2.
4. **Greedy vs canonical** string length for small graphs.

Usage:
    python benchmarks/string_length_analysis.py --seed 42
    python benchmarks/string_length_analysis.py --seed 42 --no-canonical --max-nodes 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import networkx as nx

from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_SEED = 42
# Graphs up to this size try all starting nodes; above, sample.
GREEDY_ALL_STARTS_LIMIT = 50
# Canonical string computation limit (exhaustive search is expensive).
CANONICAL_LIMIT = 8

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StringLengthRecord:
    """One data point: a single graph and its string length."""

    family: str
    num_nodes: int
    num_edges: int
    density: float
    binary_length: int  # N*(N-1)/2 for undirected, N^2 for directed
    greedy_length_best: int  # min over starting nodes
    greedy_length_node0: int  # from node 0 only
    canonical_length: int  # -1 if not computed
    compression_ratio: float  # greedy_length_best / binary_length
    theoretical_prediction: float  # Eq. 9 (reference, not our algorithm)
    directed: bool
    time_s: float


@dataclass
class AnalysisSummary:
    """Aggregate results."""

    total_graphs: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    total_time_s: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# String length computation
# ---------------------------------------------------------------------------


def _greedy_string_length(sg: SparseGraph, start_node: int) -> int:
    """Compute greedy G2S string length from a single starting node.

    Returns -1 if the starting node cannot reach all other nodes.
    """
    try:
        g2s = GraphToString(sg)
        w, _ = g2s.run(initial_node=start_node)
        return len(w)
    except ValueError:
        return -1


def _greedy_min_length(sg: SparseGraph, max_starts: int | None = None) -> tuple[int, int]:
    """Compute min greedy string length over starting nodes.

    Returns:
        (min_length, length_from_node_0).
        min_length is -1 if no starting node succeeds.
    """
    n = sg.node_count()
    if n <= 1:
        return 0, 0

    len_0 = _greedy_string_length(sg, 0)

    if max_starts is None or max_starts >= n:
        starts = list(range(n))
    else:
        rng = random.Random(42)
        others = rng.sample(range(1, n), min(max_starts - 1, n - 1))
        starts = [0] + others

    best = -1
    for v in starts:
        length = len_0 if v == 0 else _greedy_string_length(sg, v)
        if length >= 0 and (best < 0 or length < best):
            best = length

    return best, len_0 if len_0 >= 0 else -1


def _theoretical_eq9(n: int, rho: float) -> float:
    """Preprint Eq. 9: E[|I_G|] ~ (sqrt(pi) / (2*sqrt(2))) * N^2 * sqrt(rho).

    This is for the matrix-pointer algorithm, not our CDLL-based one.
    Serves as a reference point.
    """
    if n <= 1 or rho <= 0:
        return 0.0
    return (math.sqrt(math.pi) / (2.0 * math.sqrt(2.0))) * n * n * math.sqrt(rho)


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


def _generate_graph_suite(
    rng: random.Random, max_nodes: int = 100
) -> list[tuple[str, nx.Graph, bool]]:
    """Generate a diverse suite of graphs for string length analysis.

    Returns list of (family_name, nx_graph, directed).
    """
    graphs: list[tuple[str, nx.Graph, bool]] = []

    node_sizes_sparse = [n for n in [4, 6, 8, 10, 15, 20, 30, 50, 75, 100] if n <= max_nodes]
    node_sizes_dense = [n for n in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20] if n <= max_nodes]

    # ---- Trees (minimum connected, density ~ 2/N) ----
    for n in node_sizes_sparse:
        seed = rng.randint(0, 2**31)
        graphs.append(("tree", nx.random_labeled_tree(n, seed=seed), False))

    # ---- Paths (linear chain) ----
    for n in node_sizes_sparse:
        graphs.append(("path", nx.path_graph(n), False))

    # ---- Stars (hub-and-spoke, same edge count as trees) ----
    for n in node_sizes_sparse:
        graphs.append(("star", nx.star_graph(n - 1), False))

    # ---- Cycles ----
    for n in node_sizes_sparse:
        graphs.append(("cycle", nx.cycle_graph(n), False))

    # ---- Complete graphs (density = 1) ----
    for n in node_sizes_dense:
        graphs.append(("complete", nx.complete_graph(n), False))

    # ---- Barabasi-Albert (scale-free, controlled density via m) ----
    for n in [n for n in node_sizes_sparse if n >= 10]:
        for m in [1, 2, 3]:
            seed = rng.randint(0, 2**31)
            graphs.append((f"ba_m{m}", nx.barabasi_albert_graph(n, m, seed=seed), False))

    # ---- Erdos-Renyi at controlled densities ----
    for n in [n for n in [10, 15, 20, 30, 50] if n <= max_nodes]:
        for p in [0.1, 0.2, 0.3, 0.5, 0.7]:
            for _attempt in range(20):
                seed = rng.randint(0, 2**31)
                g = nx.gnp_random_graph(n, p, seed=seed)
                if nx.is_connected(g):
                    graphs.append((f"gnp_p{p:.1f}", g, False))
                    break

    # ---- Watts-Strogatz small-world ----
    for n in [n for n in [10, 20, 30, 50] if n <= max_nodes]:
        for k in [4, 6]:
            if k >= n:
                continue
            seed = rng.randint(0, 2**31)
            g = nx.watts_strogatz_graph(n, k, 0.3, seed=seed)
            if nx.is_connected(g):
                graphs.append((f"ws_k{k}", g, False))

    # ---- Grids (regular, sparse, 2D lattice) ----
    for m in [3, 4, 5, 6, 7, 8]:
        if m * m > max_nodes:
            break
        g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(m, m))
        graphs.append(("grid", g, False))

    # ---- Ladder ----
    for n in [n for n in [4, 6, 8, 10, 15, 20] if 2 * n <= max_nodes]:
        graphs.append(("ladder", nx.ladder_graph(n), False))

    # ---- Special graphs ----
    graphs.append(("petersen", nx.petersen_graph(), False))
    for n in [n for n in [5, 8, 10, 15, 20] if n <= max_nodes]:
        graphs.append(("wheel", nx.wheel_graph(n), False))

    return graphs


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def _analyze_graph(
    family: str,
    nx_graph: nx.Graph,
    directed: bool,
    compute_canonical: bool,
) -> StringLengthRecord:
    """Analyze string length for a single graph."""
    adapter = NetworkXAdapter()
    sg = adapter.from_external(nx_graph, directed=directed)

    n = sg.node_count()
    m = sg.logical_edge_count()
    binary_len = n * n if directed else n * (n - 1) // 2
    density = m / binary_len if binary_len > 0 else 0.0

    t0 = time.perf_counter()

    max_starts = None if n <= GREEDY_ALL_STARTS_LIMIT else 20
    greedy_best, greedy_0 = _greedy_min_length(sg, max_starts=max_starts)

    canonical_len = -1
    if compute_canonical and n <= CANONICAL_LIMIT:
        import contextlib

        from isalgraph.core.canonical import canonical_string

        with contextlib.suppress(ValueError, RuntimeError):
            canonical_len = len(canonical_string(sg))

    elapsed = time.perf_counter() - t0
    theo = _theoretical_eq9(n, density)
    compression = greedy_best / binary_len if binary_len > 0 and greedy_best >= 0 else -1.0

    return StringLengthRecord(
        family=family,
        num_nodes=n,
        num_edges=m,
        density=round(density, 6),
        binary_length=binary_len,
        greedy_length_best=greedy_best,
        greedy_length_node0=greedy_0,
        canonical_length=canonical_len,
        compression_ratio=round(compression, 6),
        theoretical_prediction=round(theo, 2),
        directed=directed,
        time_s=round(elapsed, 4),
    )


def run_analysis(
    seed: int,
    output_dir: str,
    max_nodes: int = 100,
    compute_canonical: bool = True,
) -> tuple[AnalysisSummary, list[dict[str, Any]]]:
    """Run the full string length analysis."""
    rng = random.Random(seed)
    summary = AnalysisSummary()

    print("String Length Analysis Benchmark")
    print(f"{'=' * 60}")
    print(f"Seed: {seed}")
    print(f"Max nodes: {max_nodes}")
    print(f"Canonical: {'yes (N <= ' + str(CANONICAL_LIMIT) + ')' if compute_canonical else 'no'}")
    print(f"Output dir: {output_dir}")
    print()

    graphs = _generate_graph_suite(rng, max_nodes=max_nodes)
    print(f"Generated {len(graphs)} graphs.\n")

    for i, (family, nx_g, directed) in enumerate(graphs):
        try:
            rec = _analyze_graph(family, nx_g, directed, compute_canonical)
            summary.records.append(asdict(rec))
        except Exception as exc:
            summary.errors.append(
                {
                    "index": i,
                    "family": family,
                    "num_nodes": nx_g.number_of_nodes(),
                    "error": str(exc),
                }
            )

        if (i + 1) % max(1, len(graphs) // 20) == 0:
            print(f"  ... {i + 1}/{len(graphs)} done")

    summary.total_graphs = len(summary.records)
    summary.total_time_s = sum(r["time_s"] for r in summary.records)

    # ---- Print results ----
    print(f"\n{'=' * 80}")
    print(
        f"{'Family':<20} {'N':>4} {'M':>5} {'rho':>6} "
        f"{'|w|':>6} {'|w0|':>6} {'|can|':>6} {'N^2':>6} {'ratio':>7} {'Eq9':>7}"
    )
    print("-" * 80)

    for rec in summary.records:
        can_s = f"{rec['canonical_length']:>6}" if rec["canonical_length"] >= 0 else "     -"
        print(
            f"{rec['family']:<20} {rec['num_nodes']:>4} {rec['num_edges']:>5} "
            f"{rec['density']:>6.3f} "
            f"{rec['greedy_length_best']:>6} {rec['greedy_length_node0']:>6} "
            f"{can_s} "
            f"{rec['binary_length']:>6} "
            f"{rec['compression_ratio']:>7.3f} "
            f"{rec['theoretical_prediction']:>7.1f}"
        )

    # ---- Canonical vs Greedy ----
    can_recs = [r for r in summary.records if r["canonical_length"] >= 0]
    if can_recs:
        print(f"\n{'=' * 60}")
        print(f"CANONICAL vs GREEDY (N <= {CANONICAL_LIMIT})")
        print(f"{'=' * 60}")
        total_saved = 0
        for r in can_recs:
            saved = r["greedy_length_best"] - r["canonical_length"]
            total_saved += saved
            print(
                f"  {r['family']:<15} N={r['num_nodes']:>2}  "
                f"canonical={r['canonical_length']:>3}  greedy={r['greedy_length_best']:>3}  "
                f"saved={saved}"
            )
        print(f"  Total chars saved by canonical: {total_saved}")

    # ---- Summary stats ----
    ratios = [r["compression_ratio"] for r in summary.records if r["compression_ratio"] >= 0]
    print(f"\n{'=' * 60}")
    print(
        f"Total graphs: {summary.total_graphs}, Time: {summary.total_time_s:.2f}s, "
        f"Errors: {len(summary.errors)}"
    )
    if ratios:
        print(
            f"Compression ratio |w|/N^2: "
            f"mean={sum(ratios) / len(ratios):.4f}, "
            f"min={min(ratios):.4f}, max={max(ratios):.4f}"
        )

    # ---- Save JSON ----
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "string_length_analysis.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "benchmark": "string_length_analysis",
                "config": {
                    "seed": seed,
                    "max_nodes": max_nodes,
                    "compute_canonical": compute_canonical,
                },
                "summary": {
                    "total_graphs": summary.total_graphs,
                    "total_time_s": round(summary.total_time_s, 4),
                    "errors": len(summary.errors),
                },
                "records": summary.records,
                "errors": summary.errors,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {output_path}")

    return summary, summary.records


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def save_csv(records: list[dict[str, Any]], output_dir: str) -> str:
    """Save analysis records as CSV using pandas.

    Args:
        records: List of record dicts from AnalysisSummary.records.
        output_dir: Directory to write the CSV file.

    Returns:
        Path to the saved CSV file.
    """
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "string_length_analysis.csv")
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    print(f"CSV saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure generation (2x2 panel)
# ---------------------------------------------------------------------------


def _get_family_color(family: str) -> str:
    """Resolve a family name to its color from FAMILY_COLORS.

    Falls back to grey for unknown families. Handles gnp_pX.X -> gnp mapping.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from plotting_styles import FAMILY_COLORS  # noqa: E402

    if family in FAMILY_COLORS:
        return FAMILY_COLORS[family]
    # Try prefix matching (e.g. gnp_p0.3 -> gnp, ws_k4 -> watts_strogatz)
    prefix_map = {
        "gnp_p": "gnp",
        "ws_k": "watts_strogatz",
        "ba_m": "barabasi_albert",
    }
    for prefix, key in prefix_map.items():
        if family.startswith(prefix):
            # Check if exact family key exists first (e.g. ba_m1)
            if family in FAMILY_COLORS:
                return FAMILY_COLORS[family]
            return FAMILY_COLORS.get(key, "#BBBBBB")
    return "#BBBBBB"


def generate_figure(records: list[dict[str, Any]], output_dir: str) -> list[str]:
    """Generate a 2x2 publication-quality figure of string length analysis.

    Panel A (top-left): Log-log scatter of greedy_length_best vs num_nodes.
    Panel B (top-right): Scatter of compression_ratio vs density.
    Panel C (bottom-left): Bar chart of mean compression ratio by family.
    Panel D (bottom-right): Canonical vs greedy length scatter.

    Args:
        records: List of record dicts from AnalysisSummary.records.
        output_dir: Directory to save the figure.

    Returns:
        List of saved file paths.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    from plotting_styles import (  # noqa: E402
        PLOT_SETTINGS,
        apply_ieee_style,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()

    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", 0.85))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Collect unique families and their colors
    families_seen: dict[str, str] = {}
    for r in records:
        fam = r["family"]
        if fam not in families_seen:
            families_seen[fam] = _get_family_color(fam)

    # ---- Panel A: Log-log scatter (greedy_length_best vs num_nodes) ----
    for fam, color in families_seen.items():
        fam_recs = [r for r in records if r["family"] == fam and r["greedy_length_best"] >= 0]
        if not fam_recs:
            continue
        xs = [r["num_nodes"] for r in fam_recs]
        ys = [r["greedy_length_best"] for r in fam_recs]
        ax_a.scatter(
            xs,
            ys,
            c=color,
            label=fam,
            s=PLOT_SETTINGS["scatter_size"],
            alpha=PLOT_SETTINGS["scatter_alpha"],
            edgecolors="white",
            linewidths=PLOT_SETTINGS["scatter_edgewidth"],
        )

    # Reference lines
    all_nodes = [r["num_nodes"] for r in records if r["greedy_length_best"] >= 0]
    if all_nodes:
        x_ref = np.linspace(max(min(all_nodes), 2), max(all_nodes), 100)
        ax_a.plot(x_ref, x_ref, "--", color="0.5", linewidth=0.8, label=r"$y = N$")
        ax_a.plot(x_ref, x_ref**2, ":", color="0.3", linewidth=0.8, label=r"$y = N^2$")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Number of nodes $N$")
    ax_a.set_ylabel("Greedy string length $|w|$")
    ax_a.set_title("(A) String length vs graph size")
    ax_a.legend(fontsize=6, ncol=2, loc="upper left")

    # ---- Panel B: Scatter (compression_ratio vs density) ----
    for fam, color in families_seen.items():
        fam_recs = [r for r in records if r["family"] == fam and r["compression_ratio"] >= 0]
        if not fam_recs:
            continue
        xs = [r["density"] for r in fam_recs]
        ys = [r["compression_ratio"] for r in fam_recs]
        ax_b.scatter(
            xs,
            ys,
            c=color,
            label=fam,
            s=PLOT_SETTINGS["scatter_size"],
            alpha=PLOT_SETTINGS["scatter_alpha"],
            edgecolors="white",
            linewidths=PLOT_SETTINGS["scatter_edgewidth"],
        )

    ax_b.set_xlabel("Edge density $\\rho$")
    ax_b.set_ylabel("Compression ratio $|w| / N^2$")
    ax_b.set_title("(B) Compression vs density")
    ax_b.legend(fontsize=6, ncol=2, loc="upper left")

    # ---- Panel C: Bar chart (mean compression ratio by family) ----
    family_ratios: dict[str, list[float]] = {}
    for r in records:
        if r["compression_ratio"] >= 0:
            family_ratios.setdefault(r["family"], []).append(r["compression_ratio"])

    # Only families with >= 3 data points
    bar_families = [f for f, vals in family_ratios.items() if len(vals) >= 3]
    bar_families.sort(key=lambda f: sum(family_ratios[f]) / len(family_ratios[f]))
    bar_means = [sum(family_ratios[f]) / len(family_ratios[f]) for f in bar_families]
    bar_colors = [_get_family_color(f) for f in bar_families]

    ax_c.barh(
        range(len(bar_families)),
        bar_means,
        color=bar_colors,
        alpha=PLOT_SETTINGS["bar_alpha"],
        edgecolor="white",
    )
    ax_c.set_yticks(range(len(bar_families)))
    ax_c.set_yticklabels(bar_families, fontsize=7)
    ax_c.set_xlabel("Mean compression ratio $|w| / N^2$")
    ax_c.set_title("(C) Mean compression by family")

    # ---- Panel D: Canonical vs Greedy ----
    can_recs = [r for r in records if r["canonical_length"] >= 0]
    if can_recs:
        can_xs = [r["canonical_length"] for r in can_recs]
        can_ys = [r["greedy_length_best"] for r in can_recs]
        can_colors = [_get_family_color(r["family"]) for r in can_recs]
        ax_d.scatter(
            can_xs,
            can_ys,
            c=can_colors,
            s=PLOT_SETTINGS["scatter_size"] * 2,
            alpha=PLOT_SETTINGS["scatter_alpha"],
            edgecolors="white",
            linewidths=PLOT_SETTINGS["scatter_edgewidth"],
        )
        # y=x reference line
        max_val = max(max(can_xs), max(can_ys))
        ref_line = np.linspace(0, max_val * 1.1, 50)
        ax_d.plot(ref_line, ref_line, "--", color="0.5", linewidth=0.8, label="$y = x$")

        total_saved = sum(g - c for g, c in zip(can_ys, can_xs, strict=True))
        ax_d.annotate(
            f"Total saved: {total_saved} chars",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            verticalalignment="top",
        )
        ax_d.legend(fontsize=7)
    else:
        ax_d.text(
            0.5,
            0.5,
            "No canonical data\n(N too large)",
            transform=ax_d.transAxes,
            ha="center",
            va="center",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
        )

    ax_d.set_xlabel("Canonical string length $|w^*|$")
    ax_d.set_ylabel("Greedy string length $|w|$")
    ax_d.set_title("(D) Canonical vs greedy")

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "string_length_analysis")
    saved = save_figure(fig, base_path)
    plt.close(fig)
    print(f"Figure saved to: {saved}")
    return saved


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def generate_table(records: list[dict[str, Any]], output_dir: str) -> str:
    """Generate a LaTeX summary table grouped by family.

    Columns: Family, N_range, Mean_|w|, Mean_ratio, Best_ratio, Mean_time_s.

    Args:
        records: List of record dicts from AnalysisSummary.records.
        output_dir: Directory to save the .tex file.

    Returns:
        Path to the saved .tex file.
    """
    import pandas as pd

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from plotting_styles import save_latex_table  # noqa: E402

    # Group by family
    family_data: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        family_data.setdefault(r["family"], []).append(r)

    rows = []
    for family in sorted(family_data.keys()):
        recs = family_data[family]
        nodes = [r["num_nodes"] for r in recs]
        lengths = [r["greedy_length_best"] for r in recs if r["greedy_length_best"] >= 0]
        ratios = [r["compression_ratio"] for r in recs if r["compression_ratio"] >= 0]
        times = [r["time_s"] for r in recs]

        n_range = f"{min(nodes)}--{max(nodes)}" if len(nodes) > 1 else str(nodes[0])
        mean_len = sum(lengths) / len(lengths) if lengths else -1.0
        mean_ratio = sum(ratios) / len(ratios) if ratios else -1.0
        best_ratio = min(ratios) if ratios else -1.0
        mean_time = sum(times) / len(times) if times else 0.0

        rows.append(
            {
                "Family": family,
                "N_range": n_range,
                "Mean_|w|": round(mean_len, 1),
                "Mean_ratio": round(mean_ratio, 4),
                "Best_ratio": round(best_ratio, 4),
                "Mean_time_s": round(mean_time, 4),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "string_length_analysis.tex")
    save_latex_table(
        df,
        path,
        caption="String length analysis by graph family.",
        label="tab:string_length_analysis",
    )
    print(f"LaTeX table saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Empirical analysis of IsalGraph string length vs graph size/density."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-nodes", type=int, default=100)
    parser.add_argument("--no-canonical", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["local", "picasso"],
        default="local",
        help="Execution mode: local (default) or picasso (HPC).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of workers (kept for interface consistency).",
    )
    parser.add_argument("--csv", action="store_true", help="Save results as CSV.")
    parser.add_argument("--plot", action="store_true", help="Generate 2x2 figure.")
    parser.add_argument("--table", action="store_true", help="Generate LaTeX table.")
    args = parser.parse_args()

    # In picasso mode, default all outputs to True
    if args.mode == "picasso":
        if not args.csv:
            args.csv = True
        if not args.plot:
            args.plot = True
        if not args.table:
            args.table = True

    summary, records = run_analysis(
        seed=args.seed,
        output_dir=args.output_dir,
        max_nodes=args.max_nodes,
        compute_canonical=not args.no_canonical,
    )

    if args.csv:
        save_csv(records, args.output_dir)

    if args.plot:
        generate_figure(records, args.output_dir)

    if args.table:
        generate_table(records, args.output_dir)

    if summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
