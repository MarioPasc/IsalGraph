"""Greedy Optimality Gap Distribution benchmark.

Measures how close the greedy G2S algorithm is to the canonical (globally
optimal) encoding. For each graph, computes:
  - Best greedy encoding: min over all starting nodes v of |G2S(G, v)|
  - Canonical encoding: |w*_G| via exhaustive backtracking search

The optimality ratio |w'_best| / |w*| characterizes the quality of the
greedy algorithm. Ratio = 1.0 means greedy is optimal; ratio > 1.0 means
the greedy found a longer-than-necessary string.

Hypotheses:
  H1: Greedy is near-optimal for sparse/tree-like families (ratio ~1.0)
      but exhibits a growing gap for dense families.
  H2: The gap grows with N for dense graph families.

Expected results: Trees always optimal (ratio=1.0), complete/dense GNP
have the largest gaps, the gap increases with N for dense families.

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
from scipy import stats as sp_stats

from benchmarks.plotting_styles import (
    FAMILY_COLORS,
    FAMILY_MARKERS,
    INSTRUCTION_COLORS,
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
from isalgraph.core.canonical import canonical_string
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.sparse_graph import SparseGraph

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_SEED = 42
CANONICAL_TIMEOUT_S = 120.0

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class OptimalityGapRecord:
    """Result for a single graph: greedy vs canonical comparison."""

    test_id: int
    family: str
    num_nodes: int
    num_edges: int
    density: float
    directed: bool
    # Greedy data
    greedy_best_length: int
    greedy_best_node: int
    greedy_best_string: str
    greedy_worst_length: int
    greedy_worst_node: int
    greedy_mean_length: float
    greedy_std_length: float
    n_starting_nodes: int
    # Canonical data
    canonical_length: int
    canonical_str: str
    # Derived metrics
    optimality_ratio: float
    gap_absolute: int
    greedy_is_optimal: bool
    # Per-node lengths for individual figure
    per_node_lengths: list[tuple[int, int]]
    # Timing
    time_greedy_s: float
    time_canonical_s: float
    time_total_s: float
    error: str = ""


# =============================================================================
# Graph generation
# =============================================================================


def _generate_test_graphs(
    seed: int,
) -> list[tuple[str, SparseGraph]]:
    """Generate diverse graph instances for optimality gap analysis.

    Returns list of (family_name, SparseGraph) tuples.
    """
    rng = np.random.default_rng(seed)
    adapter = NetworkXAdapter()
    graphs: list[tuple[str, SparseGraph]] = []

    def _add(family: str, nxg: nx.Graph) -> None:
        sg = adapter.from_external(nxg, directed=False)
        if sg.node_count() >= 2:
            graphs.append((family, sg))

    # Trees: 3-12, 10 samples per N
    for n in range(3, 13):
        for _ in range(10):
            _add("tree", nx.random_labeled_tree(n, seed=int(rng.integers(0, 2**31))))

    # Path: 3-12
    for n in range(3, 13):
        _add("path", nx.path_graph(n))

    # Cycle: 3-12
    for n in range(3, 13):
        _add("cycle", nx.cycle_graph(n))

    # Star: 3-12
    for n in range(3, 13):
        _add("star", nx.star_graph(n - 1))

    # Complete: 3-7 (cap for canonical cost)
    for n in range(3, 8):
        _add("complete", nx.complete_graph(n))

    # GNP p=0.2: 4-10, 5 samples
    for n in range(4, 11):
        for _ in range(5):
            for _retry in range(20):
                g = nx.gnp_random_graph(n, 0.2, seed=int(rng.integers(0, 2**31)))
                if nx.is_connected(g) and g.number_of_edges() > 0:
                    _add("gnp_p0.2", g)
                    break

    # GNP p=0.3: 4-9, 5 samples
    for n in range(4, 10):
        for _ in range(5):
            for _retry in range(20):
                g = nx.gnp_random_graph(n, 0.3, seed=int(rng.integers(0, 2**31)))
                if nx.is_connected(g) and g.number_of_edges() > 0:
                    _add("gnp_p0.3", g)
                    break

    # GNP p=0.5: 4-8, 5 samples
    for n in range(4, 9):
        for _ in range(5):
            for _retry in range(20):
                g = nx.gnp_random_graph(n, 0.5, seed=int(rng.integers(0, 2**31)))
                if nx.is_connected(g) and g.number_of_edges() > 0:
                    _add("gnp_p0.5", g)
                    break

    # BA m=1: 4-12, 5 samples
    for n in range(4, 13):
        for _ in range(5):
            _add("ba_m1", nx.barabasi_albert_graph(n, 1, seed=int(rng.integers(0, 2**31))))

    # BA m=2: 4-10, 5 samples
    for n in range(4, 11):
        for _ in range(5):
            _add("ba_m2", nx.barabasi_albert_graph(n, 2, seed=int(rng.integers(0, 2**31))))

    # Watts-Strogatz k=2: 4-10, 5 samples (need n >= 2k)
    for n in range(4, 11):
        for _ in range(5):
            _add(
                "watts_strogatz",
                nx.watts_strogatz_graph(n, 2, 0.3, seed=int(rng.integers(0, 2**31))),
            )

    # Ladder: 3-10
    for n in range(3, 11):
        _add("ladder", nx.ladder_graph(n))

    # Wheel: 4-10
    for n in range(4, 11):
        _add("wheel", nx.wheel_graph(n))

    logger.info("Generated %d test graphs across families", len(graphs))
    return graphs


# =============================================================================
# Core computation
# =============================================================================


def _run_single_test(
    test_id: int,
    family: str,
    sg: SparseGraph,
) -> OptimalityGapRecord:
    """Run greedy from all starting nodes + canonical for a single graph."""
    n = sg.node_count()
    m = sg.logical_edge_count()
    max_edges = n * (n - 1) / 2 if not sg.directed() else n * (n - 1)
    density = m / max_edges if max_edges > 0 else 0.0

    t0 = time.perf_counter()

    # -- Greedy from all starting nodes --
    per_node: list[tuple[int, int]] = []
    t_greedy_start = time.perf_counter()
    for v in range(n):
        try:
            w, _ = GraphToString(sg).run(initial_node=v)
            per_node.append((v, len(w)))
        except (ValueError, RuntimeError):
            pass
    t_greedy = time.perf_counter() - t_greedy_start

    if not per_node:
        return OptimalityGapRecord(
            test_id=test_id,
            family=family,
            num_nodes=n,
            num_edges=m,
            density=density,
            directed=sg.directed(),
            greedy_best_length=0,
            greedy_best_node=-1,
            greedy_best_string="",
            greedy_worst_length=0,
            greedy_worst_node=-1,
            greedy_mean_length=0.0,
            greedy_std_length=0.0,
            n_starting_nodes=0,
            canonical_length=0,
            canonical_str="",
            optimality_ratio=0.0,
            gap_absolute=0,
            greedy_is_optimal=False,
            per_node_lengths=[],
            time_greedy_s=t_greedy,
            time_canonical_s=0.0,
            time_total_s=time.perf_counter() - t0,
            error="no valid starting node",
        )

    # Best and worst greedy
    lengths = [l for _, l in per_node]
    best_idx = int(np.argmin(lengths))
    worst_idx = int(np.argmax(lengths))
    best_v, best_len = per_node[best_idx]
    worst_v, worst_len = per_node[worst_idx]

    # Get the actual best greedy string
    best_w, _ = GraphToString(sg).run(initial_node=best_v)

    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))

    # -- Canonical string --
    t_canon_start = time.perf_counter()
    try:
        w_canon = canonical_string(sg)
        canon_len = len(w_canon)
    except Exception as exc:
        t_canon = time.perf_counter() - t_canon_start
        return OptimalityGapRecord(
            test_id=test_id,
            family=family,
            num_nodes=n,
            num_edges=m,
            density=density,
            directed=sg.directed(),
            greedy_best_length=best_len,
            greedy_best_node=best_v,
            greedy_best_string=best_w,
            greedy_worst_length=worst_len,
            greedy_worst_node=worst_v,
            greedy_mean_length=mean_len,
            greedy_std_length=std_len,
            n_starting_nodes=len(per_node),
            canonical_length=-1,
            canonical_str="",
            optimality_ratio=-1.0,
            gap_absolute=-1,
            greedy_is_optimal=False,
            per_node_lengths=per_node,
            time_greedy_s=t_greedy,
            time_canonical_s=t_canon,
            time_total_s=time.perf_counter() - t0,
            error=f"canonical failed: {exc}",
        )
    t_canon = time.perf_counter() - t_canon_start

    # -- Derived metrics --
    ratio = best_len / canon_len if canon_len > 0 else 1.0
    gap = best_len - canon_len
    is_opt = best_len == canon_len

    return OptimalityGapRecord(
        test_id=test_id,
        family=family,
        num_nodes=n,
        num_edges=m,
        density=density,
        directed=sg.directed(),
        greedy_best_length=best_len,
        greedy_best_node=best_v,
        greedy_best_string=best_w,
        greedy_worst_length=worst_len,
        greedy_worst_node=worst_v,
        greedy_mean_length=mean_len,
        greedy_std_length=std_len,
        n_starting_nodes=len(per_node),
        canonical_length=canon_len,
        canonical_str=w_canon,
        optimality_ratio=ratio,
        gap_absolute=gap,
        greedy_is_optimal=is_opt,
        per_node_lengths=per_node,
        time_greedy_s=t_greedy,
        time_canonical_s=t_canon,
        time_total_s=time.perf_counter() - t0,
    )


# Parallel worker (must be top-level for pickling)
def _parallel_worker(
    args: tuple[int, str, int, list[tuple[int, int]], bool],
) -> OptimalityGapRecord:
    """Reconstruct graph in subprocess and run test."""
    test_id, family, max_nodes, edges, directed = args
    sg = SparseGraph(max_nodes, directed)
    for _ in range(max_nodes):
        sg.add_node()
    for u, v in edges:
        sg.add_edge(u, v)
    return _run_single_test(test_id, family, sg)


def _serialize_for_parallel(
    test_id: int,
    family: str,
    sg: SparseGraph,
) -> tuple[int, str, int, list[tuple[int, int]], bool]:
    """Serialize SparseGraph for subprocess transport."""
    edges: list[tuple[int, int]] = []
    for u in range(sg.node_count()):
        for v in sg.neighbors(u):
            if sg.directed() or u < v:
                edges.append((u, v))
    return (test_id, family, sg.node_count(), edges, sg.directed())


# =============================================================================
# Benchmark runner
# =============================================================================


def run_benchmark(
    seed: int = DEFAULT_SEED,
    n_workers: int = 1,
) -> list[OptimalityGapRecord]:
    """Run the full benchmark, returning all records."""
    graphs = _generate_test_graphs(seed)
    results: list[OptimalityGapRecord] = []

    if n_workers <= 1:
        for i, (family, sg) in enumerate(graphs):
            rec = _run_single_test(i, family, sg)
            results.append(rec)
            if (i + 1) % max(1, len(graphs) // 10) == 0:
                logger.info("  Progress: %d/%d tests", i + 1, len(graphs))
    else:
        tasks = [_serialize_for_parallel(i, fam, sg) for i, (fam, sg) in enumerate(graphs)]
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
# Results aggregation and saving
# =============================================================================


def save_results(
    results: list[OptimalityGapRecord],
    output_dir: str,
    seed: int,
) -> None:
    """Save JSON summary."""
    valid = [r for r in results if r.error == "" and r.canonical_length > 0]
    n_errors = sum(1 for r in results if r.error != "")

    by_family: dict[str, dict[str, Any]] = {}
    for r in valid:
        if r.family not in by_family:
            by_family[r.family] = {
                "count": 0,
                "n_optimal": 0,
                "sum_ratio": 0.0,
                "sum_gap": 0,
                "max_gap": 0,
                "ratios": [],
            }
        d = by_family[r.family]
        d["count"] += 1
        d["n_optimal"] += int(r.greedy_is_optimal)
        d["sum_ratio"] += r.optimality_ratio
        d["sum_gap"] += r.gap_absolute
        d["max_gap"] = max(d["max_gap"], r.gap_absolute)
        d["ratios"].append(r.optimality_ratio)

    summary: dict[str, Any] = {}
    for fam, d in by_family.items():
        ratios = d["ratios"]
        summary[fam] = {
            "count": d["count"],
            "n_optimal": d["n_optimal"],
            "p_optimal": d["n_optimal"] / d["count"] if d["count"] > 0 else 0.0,
            "mean_ratio": d["sum_ratio"] / d["count"] if d["count"] > 0 else 0.0,
            "std_ratio": float(np.std(ratios)) if ratios else 0.0,
            "max_gap": d["max_gap"],
        }

    obj = {
        "benchmark": "greedy_optimality_gap",
        "config": {"seed": seed},
        "summary": {
            "total_tests": len(results),
            "valid_tests": len(valid),
            "n_errors": n_errors,
            "overall_p_optimal": (
                sum(1 for r in valid if r.greedy_is_optimal) / len(valid) if valid else 0.0
            ),
            "overall_mean_ratio": float(np.mean([r.optimality_ratio for r in valid]))
            if valid
            else 0.0,
        },
        "results_by_family": summary,
        "errors": [
            {"test_id": r.test_id, "family": r.family, "error": r.error}
            for r in results
            if r.error != ""
        ][:50],
    }

    path = os.path.join(output_dir, "greedy_optimality_gap_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Saved JSON: %s", path)


def save_csv(
    results: list[OptimalityGapRecord],
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
                "directed": r.directed,
                "greedy_best_length": r.greedy_best_length,
                "greedy_best_node": r.greedy_best_node,
                "greedy_worst_length": r.greedy_worst_length,
                "greedy_mean_length": round(r.greedy_mean_length, 2),
                "greedy_std_length": round(r.greedy_std_length, 2),
                "n_starting_nodes": r.n_starting_nodes,
                "canonical_length": r.canonical_length,
                "optimality_ratio": round(r.optimality_ratio, 4),
                "gap_absolute": r.gap_absolute,
                "greedy_is_optimal": r.greedy_is_optimal,
                "time_greedy_s": round(r.time_greedy_s, 4),
                "time_canonical_s": round(r.time_canonical_s, 4),
                "time_total_s": round(r.time_total_s, 4),
                "error": r.error,
            }
        )
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "greedy_optimality_gap_results.csv")
    df.to_csv(path, index=False)
    logger.info("Saved CSV: %s", path)


# =============================================================================
# Population figure (2x2)
# =============================================================================


def _make_instruction_cmap() -> tuple[Any, dict[str, int]]:
    """Create a ListedColormap for instruction heatmaps."""
    from matplotlib.colors import ListedColormap

    instr_order = ["N", "P", "n", "p", "V", "v", "C", "c", "W", "_"]
    colors = [INSTRUCTION_COLORS.get(i, "#FFFFFF") for i in instr_order]
    cmap = ListedColormap(colors)
    char_to_idx = {ch: i for i, ch in enumerate(instr_order)}
    return cmap, char_to_idx


def generate_population_figure(
    results: list[OptimalityGapRecord],
    output_dir: str,
) -> None:
    """Generate 2x2 population-level analysis figure."""
    apply_ieee_style()
    valid = [r for r in results if r.error == "" and r.canonical_length > 0]
    if not valid:
        logger.warning("No valid results for population figure.")
        return

    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", 0.85))

    # Collect families in consistent order
    families = sorted(set(r.family for r in valid))
    family_data: dict[str, list[OptimalityGapRecord]] = {f: [] for f in families}
    for r in valid:
        family_data[r.family].append(r)

    # ---- Panel (a): Boxplot of optimality ratio by family ----
    ax = axes[0, 0]
    box_data = [[r.optimality_ratio for r in family_data[f]] for f in families if family_data[f]]
    box_families = [f for f in families if family_data[f]]
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
    ax.axhline(1.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.set_ylabel(r"Optimality ratio $|w'_{\mathrm{best}}| / |w^*|$")
    ax.tick_params(axis="x", rotation=45)
    n_opt = sum(1 for r in valid if r.greedy_is_optimal)
    ax.set_title(
        f"(a) Gap by family  [optimal: {n_opt}/{len(valid)} = {100 * n_opt / len(valid):.0f}%]",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    # ---- Panel (b): Ratio vs N, colored by family ----
    ax = axes[0, 1]
    for fam in families:
        recs = family_data[fam]
        if not recs:
            continue
        ns = [r.num_nodes for r in recs]
        rats = [r.optimality_ratio for r in recs]
        ax.scatter(
            ns,
            rats,
            c=FAMILY_COLORS.get(fam, "#BBBBBB"),
            marker=FAMILY_MARKERS.get(fam, "o"),
            s=PLOT_SETTINGS["scatter_size"] * 2,
            alpha=PLOT_SETTINGS["scatter_alpha"],
            label=family_display(fam),
            edgecolors="0.3",
            linewidths=PLOT_SETTINGS["scatter_edgewidth"],
        )
    ax.axhline(1.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel("Number of nodes $N$")
    ax.set_ylabel(r"Optimality ratio")
    ax.set_title("(b) Gap vs graph size", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax.legend(
        fontsize=6,
        ncol=2,
        loc="upper left",
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.5,
    )

    # ---- Panel (c): Ratio vs density, colored by N ----
    ax = axes[1, 0]
    densities = [r.density for r in valid]
    ratios = [r.optimality_ratio for r in valid]
    nodes = [r.num_nodes for r in valid]
    sc = ax.scatter(
        densities,
        ratios,
        c=nodes,
        cmap="viridis",
        s=PLOT_SETTINGS["scatter_size"] * 2,
        alpha=PLOT_SETTINGS["scatter_alpha"],
        edgecolors="0.3",
        linewidths=PLOT_SETTINGS["scatter_edgewidth"],
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("$N$", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.axhline(1.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel(r"Edge density $\rho$")
    ax.set_ylabel(r"Optimality ratio")
    ax.set_title("(c) Gap vs density", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # Spearman correlation annotation
    if len(valid) > 5:
        rho_sp, p_sp = sp_stats.spearmanr(densities, ratios)
        ax.annotate(
            f"$\\rho_s$={rho_sp:.2f}, p={p_sp:.1e}",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            va="top",
        )

    # ---- Panel (d): CDF of optimality ratio by family ----
    ax = axes[1, 1]
    for fam in families:
        recs = family_data[fam]
        if not recs:
            continue
        rats = sorted([r.optimality_ratio for r in recs])
        cdf = np.arange(1, len(rats) + 1) / len(rats)
        ax.step(
            rats,
            cdf,
            where="post",
            color=FAMILY_COLORS.get(fam, "#BBBBBB"),
            lw=PLOT_SETTINGS["line_width"],
            label=family_display(fam),
        )
    ax.axvline(1.0, color="0.4", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel(r"Optimality ratio")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("(d) CDF of optimality ratio", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax.legend(fontsize=6, ncol=2, loc="lower right", frameon=False)

    fig.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "greedy_optimality_gap_population")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Saved population figure: %s.pdf", path)


# =============================================================================
# Individual figure (1x3)
# =============================================================================


def _select_showcase(
    results: list[OptimalityGapRecord],
) -> OptimalityGapRecord | None:
    """Select a representative case with a large optimality gap."""
    valid = [
        r
        for r in results
        if r.error == ""
        and r.canonical_length > 0
        and r.optimality_ratio > 1.0
        and 5 <= r.num_nodes <= 8
        and r.n_starting_nodes >= 3
    ]
    if not valid:
        # Fallback: any suboptimal case
        valid = [
            r
            for r in results
            if r.error == "" and r.canonical_length > 0 and r.optimality_ratio > 1.0
        ]
    if not valid:
        # Fallback: any valid case
        valid = [r for r in results if r.error == "" and r.canonical_length > 0]
    if not valid:
        return None

    # Score: prefer large gap, moderate N, many starting nodes
    def score(r: OptimalityGapRecord) -> float:
        return (
            min(r.optimality_ratio - 1.0, 1.0) * 10
            + r.num_nodes * 0.3
            + r.n_starting_nodes * 0.1
            + r.gap_absolute * 2.0
        )

    return max(valid, key=score)


def generate_single_case_figure(
    results: list[OptimalityGapRecord],
    output_dir: str,
) -> None:
    """Generate detailed visualization of one representative case."""
    apply_ieee_style()
    rec = _select_showcase(results)
    if rec is None:
        logger.warning("No suitable case for individual figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=get_figure_size("double", 0.45))

    # ---- Panel (a): Graph G ----
    ax = axes[0]
    adapter = NetworkXAdapter()
    sg = SparseGraph(rec.num_nodes, rec.directed)
    for _ in range(rec.num_nodes):
        sg.add_node()
    # Reconstruct edges from greedy_best_string via S2G
    from isalgraph.core.string_to_graph import StringToGraph

    sg_show, _ = StringToGraph(rec.greedy_best_string, directed_graph=rec.directed).run()
    nxg = adapter.to_external(sg_show)

    pos = nx.spring_layout(nxg, seed=42)
    nx.draw_networkx_edges(nxg, pos, ax=ax, edge_color="0.5", width=0.8)
    nx.draw_networkx_nodes(
        nxg,
        pos,
        ax=ax,
        node_color=PAUL_TOL_BRIGHT["blue"],
        node_size=150,
        edgecolors="0.3",
        linewidths=0.5,
    )
    nx.draw_networkx_labels(nxg, pos, ax=ax, font_size=7, font_color="white")
    ax.set_title(
        f"(a) Graph: {family_display(rec.family)}\nN={rec.num_nodes}, M={rec.num_edges}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )
    ax.axis("off")

    # ---- Panel (b): Instruction heatmap: greedy vs canonical ----
    ax = axes[1]
    cmap, char_to_idx = _make_instruction_cmap()

    w_greedy = rec.greedy_best_string
    w_canon = rec.canonical_str
    max_len = max(len(w_greedy), len(w_canon))

    def _to_col(w: str) -> list[int]:
        col = [char_to_idx.get(ch, char_to_idx["_"]) for ch in w]
        col.extend([char_to_idx["_"]] * (max_len - len(col)))
        return col

    mat = np.array([_to_col(w_greedy), _to_col(w_canon)]).T
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=len(cmap.colors) - 1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Greedy\n|w'|={len(w_greedy)}", f"Canonical\n|w*|={len(w_canon)}"],
        fontsize=PLOT_SETTINGS["tick_labelsize"],
    )
    ax.set_ylabel("Instruction position")
    ax.set_title(
        f"(b) Encoding comparison\nratio={rec.optimality_ratio:.2f}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    # Legend patches
    from matplotlib.patches import Patch

    legend_patches = [Patch(facecolor=INSTRUCTION_COLORS[i], label=i) for i in "NnPpVvCcW"]
    ax.legend(
        handles=legend_patches,
        fontsize=6,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        frameon=False,
    )

    # ---- Panel (c): Bar chart of per-node lengths ----
    ax = axes[2]
    if rec.per_node_lengths:
        nodes_v = [v for v, _ in rec.per_node_lengths]
        lens = [l for _, l in rec.per_node_lengths]
        colors = [
            PAUL_TOL_BRIGHT["green"] if v == rec.greedy_best_node else PAUL_TOL_BRIGHT["blue"]
            for v in nodes_v
        ]
        ax.bar(range(len(nodes_v)), lens, color=colors, alpha=0.8, edgecolor="0.3", linewidth=0.5)
        ax.axhline(
            rec.canonical_length,
            color=PAUL_TOL_BRIGHT["red"],
            ls="--",
            lw=1.2,
            label=f"|w*|={rec.canonical_length}",
        )
        ax.set_xticks(range(len(nodes_v)))
        ax.set_xticklabels([str(v) for v in nodes_v], fontsize=PLOT_SETTINGS["tick_labelsize"])
        ax.set_xlabel("Starting node $v$")
        ax.set_ylabel(r"$|G2S(G, v)|$")
        ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], frameon=False)
    ax.set_title(
        "(c) String length by starting node",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    fig.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "greedy_optimality_gap_single_case")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Saved single-case figure: %s.pdf", path)


# =============================================================================
# LaTeX table
# =============================================================================


def generate_table(
    results: list[OptimalityGapRecord],
    output_dir: str,
) -> None:
    """Generate LaTeX summary table."""
    valid = [r for r in results if r.error == "" and r.canonical_length > 0]
    if not valid:
        logger.warning("No valid results for table.")
        return

    families = sorted(set(r.family for r in valid))
    rows = []
    for fam in families:
        recs = [r for r in valid if r.family == fam]
        ratios = [r.optimality_ratio for r in recs]
        n_opt = sum(1 for r in recs if r.greedy_is_optimal)
        ci_lo, ci_hi = binomial_ci(n_opt, len(recs))
        n_range = f"{min(r.num_nodes for r in recs)}-{max(r.num_nodes for r in recs)}"
        rows.append(
            {
                "Family": family_display(fam),
                "N range": n_range,
                "Tests": len(recs),
                "E[ratio]": f"{np.mean(ratios):.3f}",
                "SD(ratio)": f"{np.std(ratios):.3f}",
                "P(optimal)": f"{n_opt / len(recs):.2f}",
                "95\\% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                "Max gap": max(r.gap_absolute for r in recs),
                "Mean time (s)": f"{np.mean([r.time_total_s for r in recs]):.2f}",
            }
        )

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "greedy_optimality_gap_table.tex")
    save_latex_table(
        df,
        path,
        caption="Greedy optimality gap distribution by graph family. "
        "E[ratio] = mean of $|w'_{\\text{best}}| / |w^*|$; P(optimal) = "
        "fraction where greedy achieves canonical length.",
        label="tab:greedy_gap",
    )
    logger.info("Saved table: %s", path)


# =============================================================================
# Main / CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy Optimality Gap Distribution benchmark",
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

    logger.info("=== Greedy Optimality Gap Distribution ===")
    logger.info("Seed=%d, workers=%d, output=%s", args.seed, args.n_workers, args.output_dir)

    t0 = time.perf_counter()
    results = run_benchmark(seed=args.seed, n_workers=args.n_workers)
    elapsed = time.perf_counter() - t0

    # Print summary
    valid = [r for r in results if r.error == "" and r.canonical_length > 0]
    n_errors = sum(1 for r in results if r.error != "")
    n_opt = sum(1 for r in valid if r.greedy_is_optimal)
    logger.info(
        "Done: %d tests (%d valid, %d errors) in %.1fs",
        len(results),
        len(valid),
        n_errors,
        elapsed,
    )
    if valid:
        ratios = [r.optimality_ratio for r in valid]
        logger.info(
            "  Optimality ratio: mean=%.3f, std=%.3f, max=%.3f",
            np.mean(ratios),
            np.std(ratios),
            np.max(ratios),
        )
        logger.info("  Greedy optimal: %d/%d (%.1f%%)", n_opt, len(valid), 100 * n_opt / len(valid))

    # Save outputs
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
