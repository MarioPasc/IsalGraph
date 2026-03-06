"""Starting Node Sensitivity benchmark.

For a fixed graph G, how does |G2S(G, v)| vary across starting nodes v?
This benchmark measures the sensitivity of the greedy G2S encoding to the
choice of starting node.

Key metrics:
  - Coefficient of variation CV = std(|w_v|) / mean(|w_v|)
  - Range ratio = worst / best starting node string length
  - Correlation between node centrality and encoding quality

Hypotheses:
  H1: Symmetric families (complete, cycle, Petersen) have CV ~0;
      asymmetric families (star, BA, GNP) have high CV.
  H2: The worst-to-best ratio grows with N for asymmetric families.
  H3: The best starting node tends to have high degree centrality.

No canonical computation required -- scales to large graphs (N=200+).

Authors: Ezequiel Lopez-Rubio (supervisor), Mario Pascual Gonzalez.
University of Malaga.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import stats as sp_stats

from benchmarks.plotting_styles import (
    FAMILY_COLORS,
    FAMILY_MARKERS,
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    binomial_ci,
    family_display,
    get_figure_size,
    save_figure,
    save_latex_table,
)
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.graph_to_string import GraphToString

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_SEED = 42
# For large N, sample starting nodes instead of trying all
GREEDY_ALL_STARTS_LIMIT = 50
GREEDY_SAMPLE_SIZE = 20

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class SensitivityRecord:
    """Result for a single graph: G2S length variation across starting nodes."""

    test_id: int
    family: str
    num_nodes: int
    num_edges: int
    density: float
    directed: bool
    # Graph edges for reconstruction in figures
    edges: list[tuple[int, int]]
    # Per-node data
    per_node_lengths: list[tuple[int, int]]  # (v, |w'_v|)
    n_nodes_tried: int
    all_nodes_tried: bool
    # Aggregate statistics
    best_length: int
    best_node: int
    worst_length: int
    worst_node: int
    mean_length: float
    median_length: float
    std_length: float
    cv: float  # coefficient of variation
    iqr: float
    range_ratio: float  # worst / best
    # Centrality of best node
    best_node_degree_centrality: float
    best_node_betweenness_centrality: float
    best_node_closeness_centrality: float
    # Spearman correlation: degree centrality vs string length rank
    spearman_degree_length: float
    spearman_degree_pvalue: float
    # Timing
    time_s: float
    error: str = ""


# =============================================================================
# Graph generation
# =============================================================================


def _generate_test_graphs(seed: int) -> list[tuple[str, nx.Graph]]:
    """Generate diverse graph instances for sensitivity analysis."""
    rng = np.random.default_rng(seed)
    graphs: list[tuple[str, nx.Graph]] = []

    def _add(family: str, g: nx.Graph) -> None:
        if g.number_of_nodes() >= 3:
            graphs.append((family, g))

    # Trees: various N, 10 samples
    for n in [5, 10, 20, 50, 100]:
        for _ in range(10):
            _add("tree", nx.random_labeled_tree(n, seed=int(rng.integers(0, 2**31))))

    # Path
    for n in [5, 10, 20, 50, 100]:
        _add("path", nx.path_graph(n))

    # Cycle
    for n in [5, 10, 20, 50, 100]:
        _add("cycle", nx.cycle_graph(n))

    # Star
    for n in [5, 10, 20, 50, 100]:
        _add("star", nx.star_graph(n - 1))

    # Complete (smaller N due to O(N^2) edges)
    for n in [3, 5, 10, 15, 20]:
        _add("complete", nx.complete_graph(n))

    # BA m=1
    for n in [5, 10, 20, 50, 100]:
        for _ in range(5):
            _add("ba_m1", nx.barabasi_albert_graph(n, 1, seed=int(rng.integers(0, 2**31))))

    # BA m=2
    for n in [5, 10, 20, 50, 100]:
        for _ in range(5):
            _add("ba_m2", nx.barabasi_albert_graph(n, 2, seed=int(rng.integers(0, 2**31))))

    # GNP p=0.2
    for n in [5, 10, 20, 50, 100]:
        for _ in range(5):
            for _retry in range(20):
                g = nx.gnp_random_graph(n, 0.2, seed=int(rng.integers(0, 2**31)))
                if nx.is_connected(g) and g.number_of_edges() > 0:
                    _add("gnp_p0.2", g)
                    break

    # GNP p=0.5
    for n in [5, 10, 20, 50]:
        for _ in range(5):
            for _retry in range(20):
                g = nx.gnp_random_graph(n, 0.5, seed=int(rng.integers(0, 2**31)))
                if nx.is_connected(g) and g.number_of_edges() > 0:
                    _add("gnp_p0.5", g)
                    break

    # Watts-Strogatz k=4
    for n in [10, 20, 50, 100]:
        for _ in range(5):
            _add(
                "watts_strogatz",
                nx.watts_strogatz_graph(n, 4, 0.3, seed=int(rng.integers(0, 2**31))),
            )

    # Grid
    for side in [3, 4, 5, 6, 8, 10]:
        _add("grid", nx.grid_2d_graph(side, side))

    # Wheel
    for n in [5, 10, 20, 50]:
        _add("wheel", nx.wheel_graph(n))

    # Petersen
    _add("petersen", nx.petersen_graph())

    logger.info("Generated %d test graphs across families", len(graphs))
    return graphs


# =============================================================================
# Core computation
# =============================================================================


def _run_single_test(
    test_id: int,
    family: str,
    nxg: nx.Graph,
    seed: int,
) -> SensitivityRecord:
    """Run G2S from all (or sampled) starting nodes for a single graph."""
    adapter = NetworkXAdapter()
    sg = adapter.from_external(nxg, directed=False)
    n = sg.node_count()
    m = sg.logical_edge_count()
    max_edges = n * (n - 1) / 2
    density = m / max_edges if max_edges > 0 else 0.0

    # Store edges for graph reconstruction in figures
    edge_list: list[tuple[int, int]] = list(nxg.edges())

    t0 = time.perf_counter()

    # Determine which starting nodes to try
    rng = np.random.default_rng(seed + test_id)
    if n <= GREEDY_ALL_STARTS_LIMIT:
        nodes_to_try = list(range(n))
        all_tried = True
    else:
        nodes_to_try = sorted(rng.choice(n, size=GREEDY_SAMPLE_SIZE, replace=False).tolist())
        all_tried = False

    # Run G2S from each starting node
    per_node: list[tuple[int, int]] = []
    for v in nodes_to_try:
        try:
            w, _ = GraphToString(sg).run(initial_node=v)
            per_node.append((v, len(w)))
        except (ValueError, RuntimeError):
            pass

    elapsed = time.perf_counter() - t0

    if len(per_node) < 2:
        return SensitivityRecord(
            test_id=test_id,
            family=family,
            num_nodes=n,
            num_edges=m,
            density=density,
            directed=False,
            edges=edge_list,
            per_node_lengths=per_node,
            n_nodes_tried=len(per_node),
            all_nodes_tried=all_tried,
            best_length=0,
            best_node=-1,
            worst_length=0,
            worst_node=-1,
            mean_length=0.0,
            median_length=0.0,
            std_length=0.0,
            cv=0.0,
            iqr=0.0,
            range_ratio=1.0,
            best_node_degree_centrality=0.0,
            best_node_betweenness_centrality=0.0,
            best_node_closeness_centrality=0.0,
            spearman_degree_length=0.0,
            spearman_degree_pvalue=1.0,
            time_s=elapsed,
            error="fewer than 2 valid starting nodes",
        )

    # Compute statistics
    lengths = [l for _, l in per_node]
    best_idx = int(np.argmin(lengths))
    worst_idx = int(np.argmax(lengths))
    best_v, best_len = per_node[best_idx]
    worst_v, worst_len = per_node[worst_idx]
    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    std_len = float(np.std(lengths))
    cv = std_len / mean_len if mean_len > 0 else 0.0
    q75, q25 = np.percentile(lengths, [75, 25])
    iqr = float(q75 - q25)
    range_ratio = worst_len / best_len if best_len > 0 else 1.0

    # Centrality of best node
    deg_cent = nx.degree_centrality(nxg)
    bet_cent = nx.betweenness_centrality(nxg)
    clo_cent = nx.closeness_centrality(nxg)

    # Map best_v to NX node (sorted label order)
    node_list = sorted(nxg.nodes())
    best_nx_node = node_list[best_v] if best_v < len(node_list) else node_list[0]

    best_deg = deg_cent.get(best_nx_node, 0.0)
    best_bet = bet_cent.get(best_nx_node, 0.0)
    best_clo = clo_cent.get(best_nx_node, 0.0)

    # Spearman correlation: degree centrality rank vs string length
    spearman_rho = 0.0
    spearman_p = 1.0
    if len(per_node) >= 5:
        tried_nodes = [v for v, _ in per_node]
        tried_deg = [
            deg_cent.get(node_list[v] if v < len(node_list) else node_list[0], 0.0)
            for v in tried_nodes
        ]
        tried_len = [l for _, l in per_node]
        # Higher degree -> shorter string? (expect negative correlation)
        try:
            res = sp_stats.spearmanr(tried_deg, tried_len)
            spearman_rho = float(res.statistic)
            spearman_p = float(res.pvalue)
        except Exception:
            pass

    return SensitivityRecord(
        test_id=test_id,
        family=family,
        num_nodes=n,
        num_edges=m,
        density=density,
        directed=False,
        edges=edge_list,
        per_node_lengths=per_node,
        n_nodes_tried=len(per_node),
        all_nodes_tried=all_tried,
        best_length=best_len,
        best_node=best_v,
        worst_length=worst_len,
        worst_node=worst_v,
        mean_length=mean_len,
        median_length=median_len,
        std_length=std_len,
        cv=cv,
        iqr=iqr,
        range_ratio=range_ratio,
        best_node_degree_centrality=best_deg,
        best_node_betweenness_centrality=best_bet,
        best_node_closeness_centrality=best_clo,
        spearman_degree_length=spearman_rho,
        spearman_degree_pvalue=spearman_p,
        time_s=elapsed,
    )


# Parallel worker
def _parallel_worker(
    args: tuple[int, str, int, list[tuple[int, int]], int],
) -> SensitivityRecord:
    """Reconstruct graph in subprocess and run test."""
    test_id, family, num_nodes, edges, seed = args
    nxg = nx.Graph()
    nxg.add_nodes_from(range(num_nodes))
    for u, v in edges:
        nxg.add_edge(u, v)
    return _run_single_test(test_id, family, nxg, seed)


def _serialize_for_parallel(
    test_id: int,
    family: str,
    nxg: nx.Graph,
    seed: int,
) -> tuple[int, str, int, list[tuple[int, int]], int]:
    """Serialize NX graph for subprocess transport."""
    return (test_id, family, nxg.number_of_nodes(), list(nxg.edges()), seed)


# =============================================================================
# Benchmark runner
# =============================================================================


def run_benchmark(
    seed: int = DEFAULT_SEED,
    n_workers: int = 1,
) -> list[SensitivityRecord]:
    """Run the full benchmark."""
    graphs = _generate_test_graphs(seed)
    results: list[SensitivityRecord] = []

    if n_workers <= 1:
        for i, (family, nxg) in enumerate(graphs):
            rec = _run_single_test(i, family, nxg, seed)
            results.append(rec)
            if (i + 1) % max(1, len(graphs) // 10) == 0:
                logger.info("  Progress: %d/%d tests", i + 1, len(graphs))
    else:
        tasks = [_serialize_for_parallel(i, fam, nxg, seed) for i, (fam, nxg) in enumerate(graphs)]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_parallel_worker, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                results.append(rec)
                done += 1
                if done % max(1, len(tasks) // 10) == 0:
                    logger.info("  Progress: %d/%d tests", done, len(tasks))
        results.sort(key=lambda r: r.test_id)

    return results


# =============================================================================
# Results saving
# =============================================================================


def save_results(
    results: list[SensitivityRecord],
    output_dir: str,
    seed: int,
) -> None:
    """Save JSON summary."""
    valid = [r for r in results if r.error == ""]
    families = sorted(set(r.family for r in valid))

    by_family: dict[str, Any] = {}
    for fam in families:
        recs = [r for r in valid if r.family == fam]
        cvs = [r.cv for r in recs]
        rrs = [r.range_ratio for r in recs]
        by_family[fam] = {
            "count": len(recs),
            "mean_cv": float(np.mean(cvs)),
            "std_cv": float(np.std(cvs)),
            "mean_range_ratio": float(np.mean(rrs)),
            "mean_spearman": float(np.mean([r.spearman_degree_length for r in recs])),
        }

    obj = {
        "benchmark": "starting_node_sensitivity",
        "config": {"seed": seed},
        "summary": {
            "total_tests": len(results),
            "valid_tests": len(valid),
            "overall_mean_cv": float(np.mean([r.cv for r in valid])) if valid else 0.0,
            "overall_mean_range_ratio": float(np.mean([r.range_ratio for r in valid]))
            if valid
            else 0.0,
        },
        "results_by_family": by_family,
    }

    path = os.path.join(output_dir, "starting_node_sensitivity_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Saved JSON: %s", path)


def save_csv(
    results: list[SensitivityRecord],
    output_dir: str,
) -> None:
    """Save per-test CSV."""
    rows = []
    for r in results:
        rows.append(
            {
                "test_id": r.test_id,
                "family": r.family,
                "num_nodes": r.num_nodes,
                "num_edges": r.num_edges,
                "density": round(r.density, 4),
                "n_nodes_tried": r.n_nodes_tried,
                "all_nodes_tried": r.all_nodes_tried,
                "best_length": r.best_length,
                "best_node": r.best_node,
                "worst_length": r.worst_length,
                "worst_node": r.worst_node,
                "mean_length": round(r.mean_length, 2),
                "median_length": round(r.median_length, 2),
                "std_length": round(r.std_length, 2),
                "cv": round(r.cv, 4),
                "iqr": round(r.iqr, 2),
                "range_ratio": round(r.range_ratio, 4),
                "best_node_degree_centrality": round(r.best_node_degree_centrality, 4),
                "best_node_betweenness_centrality": round(r.best_node_betweenness_centrality, 4),
                "best_node_closeness_centrality": round(r.best_node_closeness_centrality, 4),
                "spearman_degree_length": round(r.spearman_degree_length, 4),
                "spearman_degree_pvalue": round(r.spearman_degree_pvalue, 6),
                "time_s": round(r.time_s, 4),
                "error": r.error,
            }
        )
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "starting_node_sensitivity_results.csv")
    df.to_csv(path, index=False)
    logger.info("Saved CSV: %s", path)


# =============================================================================
# Population figure (2x2)
# =============================================================================


def generate_population_figure(
    results: list[SensitivityRecord],
    output_dir: str,
) -> None:
    """Generate 2x2 population-level figure."""
    apply_ieee_style()
    valid = [r for r in results if r.error == ""]
    if not valid:
        logger.warning("No valid results for population figure.")
        return

    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", 0.85))
    families = sorted(set(r.family for r in valid))
    family_data: dict[str, list[SensitivityRecord]] = {f: [] for f in families}
    for r in valid:
        family_data[r.family].append(r)

    # ---- Panel (a): CV vs N by family ----
    ax = axes[0, 0]
    for fam in families:
        recs = family_data[fam]
        if not recs:
            continue
        # Group by N
        by_n: dict[int, list[float]] = {}
        for r in recs:
            by_n.setdefault(r.num_nodes, []).append(r.cv)
        ns = sorted(by_n.keys())
        means = [float(np.mean(by_n[n])) for n in ns]
        stds = [float(np.std(by_n[n])) for n in ns]
        ax.errorbar(
            ns,
            means,
            yerr=stds,
            color=FAMILY_COLORS.get(fam, "#BBBBBB"),
            marker=FAMILY_MARKERS.get(fam, "o"),
            markersize=PLOT_SETTINGS["marker_size"],
            lw=PLOT_SETTINGS["line_width"],
            capsize=PLOT_SETTINGS["errorbar_capsize"],
            label=family_display(fam),
            alpha=0.8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Number of nodes $N$")
    ax.set_ylabel("CV = $\\sigma / \\mu$")
    ax.set_title("(a) Coefficient of variation vs $N$", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax.legend(
        fontsize=5, ncol=2, loc="upper left", frameon=False, handletextpad=0.3, columnspacing=0.5
    )

    # ---- Panel (b): Range ratio (worst/best) vs N ----
    ax = axes[0, 1]
    for fam in families:
        recs = family_data[fam]
        if not recs:
            continue
        by_n: dict[int, list[float]] = {}
        for r in recs:
            by_n.setdefault(r.num_nodes, []).append(r.range_ratio)
        ns = sorted(by_n.keys())
        means = [float(np.mean(by_n[n])) for n in ns]
        stds = [float(np.std(by_n[n])) for n in ns]
        ax.errorbar(
            ns,
            means,
            yerr=stds,
            color=FAMILY_COLORS.get(fam, "#BBBBBB"),
            marker=FAMILY_MARKERS.get(fam, "o"),
            markersize=PLOT_SETTINGS["marker_size"],
            lw=PLOT_SETTINGS["line_width"],
            capsize=PLOT_SETTINGS["errorbar_capsize"],
            label=family_display(fam),
            alpha=0.8,
        )
    ax.set_xscale("log")
    ax.axhline(1.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel("Number of nodes $N$")
    ax.set_ylabel(r"Range ratio $|w_{\mathrm{worst}}| / |w_{\mathrm{best}}|$")
    ax.set_title("(b) Range ratio vs $N$", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # ---- Panel (c): Spearman corr(degree, length) by family ----
    ax = axes[1, 0]
    fam_corrs: list[tuple[str, float, float]] = []
    for fam in families:
        recs = [r for r in family_data[fam] if abs(r.spearman_degree_length) > 0]
        if len(recs) >= 3:
            corrs = [r.spearman_degree_length for r in recs]
            mean_c = float(np.mean(corrs))
            std_c = float(np.std(corrs))
            fam_corrs.append((fam, mean_c, std_c))

    if fam_corrs:
        fam_corrs.sort(key=lambda x: x[1])
        fam_names = [family_display(f) for f, _, _ in fam_corrs]
        mean_vals = [m for _, m, _ in fam_corrs]
        std_vals = [s for _, _, s in fam_corrs]
        colors = [FAMILY_COLORS.get(f, "#BBBBBB") for f, _, _ in fam_corrs]
        y_pos = range(len(fam_names))
        ax.barh(
            y_pos,
            mean_vals,
            xerr=std_vals,
            color=colors,
            alpha=0.8,
            edgecolor="0.3",
            linewidth=0.5,
            capsize=PLOT_SETTINGS["errorbar_capsize"],
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(fam_names, fontsize=PLOT_SETTINGS["tick_labelsize"])
        ax.axvline(0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel(r"Spearman $\rho$(degree, $|w_v|$)")
    ax.set_title("(c) Centrality-length correlation", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # ---- Panel (d): Boxplot of CV by family ----
    ax = axes[1, 1]
    box_data = [[r.cv for r in family_data[f]] for f in families if family_data[f]]
    box_families = [f for f in families if family_data[f]]
    if box_data:
        bp = ax.boxplot(
            box_data,
            labels=[family_display(f) for f in box_families],
            patch_artist=True,
            widths=PLOT_SETTINGS["boxplot_width"],
            flierprops={"markersize": PLOT_SETTINGS["boxplot_flier_size"]},
        )
        for patch, fam in zip(bp["boxes"], box_families):
            patch.set_facecolor(FAMILY_COLORS.get(fam, "#BBBBBB"))
            patch.set_alpha(0.7)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("CV")
    ax.set_title("(d) CV distribution by family", fontsize=PLOT_SETTINGS["axes_titlesize"])

    fig.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "starting_node_sensitivity_population")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Saved population figure: %s.pdf", path)


# =============================================================================
# Individual figure (1x3)
# =============================================================================


def _select_showcase(
    results: list[SensitivityRecord],
) -> SensitivityRecord | None:
    """Select a graph with high CV and moderate N for visualization."""
    valid = [
        r
        for r in results
        if r.error == "" and r.cv > 0 and 8 <= r.num_nodes <= 15 and r.n_nodes_tried >= 5
    ]
    if not valid:
        valid = [r for r in results if r.error == "" and r.cv > 0 and r.n_nodes_tried >= 3]
    if not valid:
        return None
    return max(valid, key=lambda r: r.cv)


def generate_single_case_figure(
    results: list[SensitivityRecord],
    output_dir: str,
) -> None:
    """Generate detailed visualization of one representative case."""
    apply_ieee_style()
    rec = _select_showcase(results)
    if rec is None:
        logger.warning("No suitable case for individual figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=get_figure_size("double", 0.45))

    # Reconstruct the NX graph from stored edges
    nxg_show = nx.Graph()
    nxg_show.add_nodes_from(range(rec.num_nodes))
    nxg_show.add_edges_from(rec.edges)

    # ---- Panel (a): Graph with nodes colored by |G2S(G,v)| ----
    ax = axes[0]
    if rec.per_node_lengths:
        len_map = {v: l for v, l in rec.per_node_lengths}
        node_list = sorted(nxg_show.nodes())
        node_colors = []
        for node in node_list:
            if node in len_map:
                node_colors.append(len_map[node])
            else:
                node_colors.append(rec.mean_length)

        pos = nx.spring_layout(nxg_show, seed=42)
        norm = Normalize(vmin=rec.best_length, vmax=rec.worst_length)

        nx.draw_networkx_edges(nxg_show, pos, ax=ax, edge_color="0.5", width=0.8)
        sc = nx.draw_networkx_nodes(
            nxg_show,
            pos,
            ax=ax,
            node_color=node_colors,
            cmap=plt.cm.RdYlBu_r,
            vmin=rec.best_length,
            vmax=rec.worst_length,
            node_size=120,
            edgecolors="0.3",
            linewidths=0.5,
        )
        nx.draw_networkx_labels(nxg_show, pos, ax=ax, font_size=5, font_color="0.1")
        fig.colorbar(sc, ax=ax, shrink=0.7, label=r"$|w_v|$")
    ax.set_title(
        f"(a) {family_display(rec.family)}, N={rec.num_nodes}\nCV={rec.cv:.3f}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )
    ax.axis("off")

    # ---- Panel (b): Bar chart of |G2S(G,v)| per node ----
    ax = axes[1]
    if rec.per_node_lengths:
        sorted_pn = sorted(rec.per_node_lengths, key=lambda x: x[1])
        nodes_v = [v for v, _ in sorted_pn]
        lens = [l for _, l in sorted_pn]
        colors = []
        for v, l in sorted_pn:
            if v == rec.best_node:
                colors.append(PAUL_TOL_BRIGHT["green"])
            elif v == rec.worst_node:
                colors.append(PAUL_TOL_BRIGHT["red"])
            else:
                colors.append(PAUL_TOL_BRIGHT["blue"])
        ax.bar(
            range(len(nodes_v)),
            lens,
            color=colors,
            alpha=0.8,
            edgecolor="0.3",
            linewidth=0.5,
        )
        ax.axhline(
            rec.mean_length, color="0.4", ls="--", lw=0.8, label=f"mean={rec.mean_length:.0f}"
        )
        # Only show tick labels if not too many nodes
        if len(nodes_v) <= 20:
            ax.set_xticks(range(len(nodes_v)))
            ax.set_xticklabels([str(v) for v in nodes_v], fontsize=5, rotation=45)
        else:
            ax.set_xlabel("Starting nodes (sorted by length)")
        ax.set_ylabel(r"$|G2S(G, v)|$")
        ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], frameon=False)
    ax.set_title(
        f"(b) String length by node\nbest={rec.best_length}, worst={rec.worst_length}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    # ---- Panel (c): Length distribution histogram ----
    ax = axes[2]
    if rec.per_node_lengths:
        lens_all = [l for _, l in rec.per_node_lengths]
        ax.hist(
            lens_all,
            bins=min(20, max(5, len(set(lens_all)))),
            color=PAUL_TOL_BRIGHT["blue"],
            alpha=0.7,
            edgecolor="0.3",
            linewidth=0.5,
        )
        ax.axvline(
            rec.best_length,
            color=PAUL_TOL_BRIGHT["green"],
            ls="--",
            lw=1.2,
            label=f"best={rec.best_length}",
        )
        ax.axvline(
            rec.worst_length,
            color=PAUL_TOL_BRIGHT["red"],
            ls="--",
            lw=1.2,
            label=f"worst={rec.worst_length}",
        )
        ax.set_xlabel(r"$|G2S(G, v)|$")
        ax.set_ylabel("Count")
        ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], frameon=False)
    ax.set_title(
        f"(c) Length distribution\nrange ratio={rec.range_ratio:.2f}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    fig.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "starting_node_sensitivity_single_case")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Saved single-case figure: %s.pdf", path)


# =============================================================================
# LaTeX table
# =============================================================================


def generate_table(
    results: list[SensitivityRecord],
    output_dir: str,
) -> None:
    """Generate LaTeX summary table."""
    valid = [r for r in results if r.error == ""]
    if not valid:
        logger.warning("No valid results for table.")
        return

    families = sorted(set(r.family for r in valid))
    rows = []
    for fam in families:
        recs = [r for r in valid if r.family == fam]
        cvs = [r.cv for r in recs]
        rrs = [r.range_ratio for r in recs]
        n_cv0 = sum(1 for r in recs if r.cv < 0.001)
        ci_lo, ci_hi = binomial_ci(n_cv0, len(recs))
        corrs = [r.spearman_degree_length for r in recs if abs(r.spearman_degree_length) > 0]
        n_range = f"{min(r.num_nodes for r in recs)}-{max(r.num_nodes for r in recs)}"
        rows.append(
            {
                "Family": family_display(fam),
                "N range": n_range,
                "Tests": len(recs),
                "Mean CV": f"{np.mean(cvs):.4f}",
                "SD(CV)": f"{np.std(cvs):.4f}",
                "Mean range": f"{np.mean(rrs):.3f}",
                "P(CV=0)": f"{n_cv0 / len(recs):.2f}",
                "Corr(deg,len)": f"{np.mean(corrs):.3f}" if corrs else "---",
                "Mean time (ms)": f"{1000 * np.mean([r.time_s for r in recs]):.0f}",
            }
        )

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "starting_node_sensitivity_table.tex")
    save_latex_table(
        df,
        path,
        caption="Starting node sensitivity by graph family. "
        "CV = coefficient of variation of $|G2S(G, v)|$ across starting nodes; "
        "Corr(deg,len) = mean Spearman correlation between degree centrality and string length.",
        label="tab:node_sensitivity",
    )
    logger.info("Saved table: %s", path)


# =============================================================================
# Main / CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Starting Node Sensitivity benchmark",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["local", "picasso"], default="local")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()

    if args.mode == "picasso":
        args.csv = True
        args.plot = True
        args.table = True

    logger.info("=== Starting Node Sensitivity ===")
    logger.info("Seed=%d, workers=%d, output=%s", args.seed, args.n_workers, args.output_dir)

    t0 = time.perf_counter()
    results = run_benchmark(seed=args.seed, n_workers=args.n_workers)
    elapsed = time.perf_counter() - t0

    valid = [r for r in results if r.error == ""]
    logger.info("Done: %d tests (%d valid) in %.1fs", len(results), len(valid), elapsed)
    if valid:
        logger.info(
            "  CV: mean=%.4f, max=%.4f",
            np.mean([r.cv for r in valid]),
            np.max([r.cv for r in valid]),
        )
        logger.info(
            "  Range ratio: mean=%.3f, max=%.3f",
            np.mean([r.range_ratio for r in valid]),
            np.max([r.range_ratio for r in valid]),
        )

    save_results(results, args.output_dir, args.seed)
    if args.csv:
        save_csv(results, args.output_dir)
    if args.plot:
        generate_population_figure(results, args.output_dir)
        generate_single_case_figure(results, args.output_dir)
    if args.table:
        generate_table(results, args.output_dir)


if __name__ == "__main__":
    main()
