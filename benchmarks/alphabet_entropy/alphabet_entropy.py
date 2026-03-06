"""Alphabet Utilization and Entropy Analysis benchmark.

Computes Shannon entropy, information density, and instruction composition
for IsalGraph encodings. Two regimes:

  Regime A (with canonical, N<=10): Compare H(w') vs H(w*) vs H_max.
  Regime B (greedy only, N<=200): Entropy scaling with graph size.

Key formulas:
  H(w) = -sum p_i * log2(p_i)  where p_i = count(instr_i) / |w|
  H_max = log2(9) = 3.17 bits  (or log2(8) if W never appears)
  eta = log2(#iso_classes(N,M)) / (|w*| * log2(9))  (efficiency ratio)

Hypotheses:
  H1: Shannon entropy of greedy strings is well below H_max, with
      navigation instructions (N/P/n/p) dominating.
  H2: Information density is lower for sparse graphs (more navigation
      overhead) and higher for dense graphs.
  H3: Canonical strings have higher information density than greedy.

Authors: Ezequiel Lopez-Rubio (supervisor), Mario Pascual Gonzalez.
University of Malaga.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from benchmarks.plotting_styles import (
    FAMILY_COLORS,
    INSTRUCTION_COLORS,
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    family_display,
    get_figure_size,
    save_figure,
    save_latex_table,
)
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.canonical import canonical_string
from isalgraph.core.graph_to_string import GraphToString

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_SEED = 42
ALPHABET = list("NnPpVvCcW")
H_MAX = math.log2(9)  # ~3.17 bits
H_MAX_NO_W = math.log2(8)  # 3.0 bits (W never appears in greedy/canonical)

# =============================================================================
# Information-theoretic functions
# =============================================================================


def shannon_entropy(w: str) -> float:
    """Compute Shannon entropy H(w) in bits."""
    if not w:
        return 0.0
    n = len(w)
    counts: dict[str, int] = {}
    for ch in w:
        counts[ch] = counts.get(ch, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def instruction_counts(w: str) -> dict[str, int]:
    """Count occurrences of each instruction."""
    counts = {ch: 0 for ch in ALPHABET}
    for ch in w:
        if ch in counts:
            counts[ch] += 1
    return counts


def log2_n_iso_classes(n: int, m: int, directed: bool = False) -> float:
    """Approximate log2 of # isomorphism classes of graphs with n nodes, m edges.

    Uses Burnside approximation: #iso_classes ~ C(max_edges, m) / n!
    """
    if n <= 1:
        return 0.0
    max_edges = n * (n - 1) if directed else n * (n - 1) // 2
    if m > max_edges or m < 0:
        return 0.0
    ln2 = math.log(2)
    log2_comb = (
        math.lgamma(max_edges + 1) - math.lgamma(m + 1) - math.lgamma(max_edges - m + 1)
    ) / ln2
    log2_nfact = math.lgamma(n + 1) / ln2
    return max(0.0, log2_comb - log2_nfact)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class EntropyRecord:
    """Entropy analysis for one graph encoding."""

    test_id: int
    family: str
    regime: str  # "A" (with canonical) or "B" (greedy only)
    num_nodes: int
    num_edges: int
    density: float
    directed: bool
    # Greedy string data
    greedy_string: str
    greedy_length: int
    greedy_instr_counts: dict[str, int]
    greedy_entropy: float
    greedy_entropy_normalized: float  # H / H_max
    greedy_info_density: float  # H bits per character
    # Instruction fractions
    greedy_frac_node: float  # (V+v) / |w|
    greedy_frac_edge: float  # (C+c) / |w|
    greedy_frac_nav: float  # (N+P+n+p) / |w|
    greedy_frac_noop: float  # W / |w|
    # Canonical string data (Regime A only)
    canonical_string: str
    canonical_length: int
    canonical_entropy: float
    canonical_entropy_normalized: float
    canonical_info_density: float
    canonical_instr_counts: dict[str, int]
    # Information-theoretic bounds
    info_lower_bound_bits: float
    efficiency_greedy: float  # info_lower / (|w'| * log2(9))
    efficiency_canonical: float  # info_lower / (|w*| * log2(9)), -1 if no canonical
    # Timing
    time_s: float
    error: str = ""


# =============================================================================
# Graph generation
# =============================================================================


def _generate_test_graphs(
    seed: int,
) -> list[tuple[str, str, nx.Graph]]:
    """Generate test graphs for both regimes.

    Returns (regime, family, nxg) tuples.
    """
    rng = np.random.default_rng(seed)
    graphs: list[tuple[str, str, nx.Graph]] = []

    def _add(regime: str, family: str, g: nx.Graph) -> None:
        if g.number_of_nodes() >= 3 and nx.is_connected(g):
            graphs.append((regime, family, g))

    # --- Regime A (with canonical, small N) ---
    # Trees
    for n in range(3, 11):
        for _ in range(5):
            _add("A", "tree", nx.random_labeled_tree(n, seed=int(rng.integers(0, 2**31))))

    for n in range(3, 11):
        _add("A", "path", nx.path_graph(n))
        _add("A", "cycle", nx.cycle_graph(n))
        _add("A", "star", nx.star_graph(n - 1))

    for n in range(3, 8):
        _add("A", "complete", nx.complete_graph(n))

    for n in range(4, 10):
        for _ in range(3):
            for _retry in range(20):
                g = nx.gnp_random_graph(n, 0.3, seed=int(rng.integers(0, 2**31)))
                if nx.is_connected(g) and g.number_of_edges() > 0:
                    _add("A", "gnp_p0.3", g)
                    break

    for n in range(4, 10):
        for _ in range(3):
            _add("A", "ba_m2", nx.barabasi_albert_graph(n, 2, seed=int(rng.integers(0, 2**31))))

    # --- Regime B (greedy only, larger N) ---
    for n in [10, 20, 50, 100, 200]:
        for _ in range(5):
            _add("B", "tree", nx.random_labeled_tree(n, seed=int(rng.integers(0, 2**31))))

    for n in [10, 20, 50, 100, 200]:
        _add("B", "path", nx.path_graph(n))
        _add("B", "cycle", nx.cycle_graph(n))
        _add("B", "star", nx.star_graph(n - 1))

    for n in [10, 15, 20]:
        _add("B", "complete", nx.complete_graph(n))

    for n in [10, 20, 50, 100]:
        for _ in range(3):
            for _retry in range(20):
                g = nx.gnp_random_graph(n, 0.3, seed=int(rng.integers(0, 2**31)))
                if nx.is_connected(g) and g.number_of_edges() > 0:
                    _add("B", "gnp_p0.3", g)
                    break

    for n in [10, 20, 50, 100]:
        for _ in range(3):
            _add("B", "ba_m2", nx.barabasi_albert_graph(n, 2, seed=int(rng.integers(0, 2**31))))

    for n in [10, 20, 50, 100]:
        for _ in range(3):
            _add(
                "B",
                "watts_strogatz",
                nx.watts_strogatz_graph(n, 4, 0.3, seed=int(rng.integers(0, 2**31))),
            )

    for side in [3, 5, 7, 10]:
        g = nx.grid_2d_graph(side, side)
        _add("B", "grid", nx.convert_node_labels_to_integers(g))

    logger.info(
        "Generated %d test graphs (A=%d, B=%d)",
        len(graphs),
        sum(1 for r, _, _ in graphs if r == "A"),
        sum(1 for r, _, _ in graphs if r == "B"),
    )
    return graphs


# =============================================================================
# Core computation
# =============================================================================


def _run_single_test(
    test_id: int,
    regime: str,
    family: str,
    nxg: nx.Graph,
) -> EntropyRecord:
    """Compute entropy analysis for a single graph."""
    t0 = time.perf_counter()
    adapter = NetworkXAdapter()
    sg = adapter.from_external(nxg, directed=False)
    n = sg.node_count()
    m = sg.logical_edge_count()
    max_edges = n * (n - 1) / 2
    density = m / max_edges if max_edges > 0 else 0.0

    # Best greedy string (try all starting nodes, or sample for large N)
    best_w = ""
    best_len = float("inf")
    if n <= 50:
        nodes_to_try = range(n)
    else:
        rng_local = np.random.default_rng(test_id)
        nodes_to_try = sorted(rng_local.choice(n, size=20, replace=False).tolist())

    for v in nodes_to_try:
        try:
            w, _ = GraphToString(sg).run(initial_node=v)
            if len(w) < best_len:
                best_w = w
                best_len = len(w)
        except (ValueError, RuntimeError):
            pass

    if not best_w and n >= 2:
        return EntropyRecord(
            test_id=test_id,
            family=family,
            regime=regime,
            num_nodes=n,
            num_edges=m,
            density=density,
            directed=False,
            greedy_string="",
            greedy_length=0,
            greedy_instr_counts={c: 0 for c in ALPHABET},
            greedy_entropy=0.0,
            greedy_entropy_normalized=0.0,
            greedy_info_density=0.0,
            greedy_frac_node=0.0,
            greedy_frac_edge=0.0,
            greedy_frac_nav=0.0,
            greedy_frac_noop=0.0,
            canonical_string="",
            canonical_length=-1,
            canonical_entropy=-1.0,
            canonical_entropy_normalized=-1.0,
            canonical_info_density=-1.0,
            canonical_instr_counts={c: 0 for c in ALPHABET},
            info_lower_bound_bits=0.0,
            efficiency_greedy=0.0,
            efficiency_canonical=-1.0,
            time_s=time.perf_counter() - t0,
            error="no valid greedy string",
        )

    # Greedy analysis
    g_counts = instruction_counts(best_w)
    g_entropy = shannon_entropy(best_w)
    g_len = len(best_w)
    g_norm = g_entropy / H_MAX if H_MAX > 0 else 0.0
    g_density_info = g_entropy  # bits per character (entropy IS bits/char already)
    frac_node = (g_counts["V"] + g_counts["v"]) / g_len if g_len > 0 else 0.0
    frac_edge = (g_counts["C"] + g_counts["c"]) / g_len if g_len > 0 else 0.0
    frac_nav = (
        (g_counts["N"] + g_counts["P"] + g_counts["n"] + g_counts["p"]) / g_len
        if g_len > 0
        else 0.0
    )
    frac_noop = g_counts["W"] / g_len if g_len > 0 else 0.0

    # Canonical (Regime A only)
    c_str = ""
    c_len = -1
    c_entropy = -1.0
    c_norm = -1.0
    c_density_info = -1.0
    c_counts: dict[str, int] = {c: 0 for c in ALPHABET}

    if regime == "A":
        try:
            c_str = canonical_string(sg)
            c_len = len(c_str)
            c_counts = instruction_counts(c_str)
            c_entropy = shannon_entropy(c_str)
            c_norm = c_entropy / H_MAX if H_MAX > 0 else 0.0
            c_density_info = c_entropy
        except Exception:
            pass

    # Information-theoretic bound
    info_lb = log2_n_iso_classes(n, m, directed=False)
    eff_greedy = info_lb / (g_len * H_MAX) if g_len > 0 and H_MAX > 0 else 0.0
    eff_canon = info_lb / (c_len * H_MAX) if c_len > 0 and H_MAX > 0 else -1.0

    return EntropyRecord(
        test_id=test_id,
        family=family,
        regime=regime,
        num_nodes=n,
        num_edges=m,
        density=density,
        directed=False,
        greedy_string=best_w,
        greedy_length=g_len,
        greedy_instr_counts=g_counts,
        greedy_entropy=g_entropy,
        greedy_entropy_normalized=g_norm,
        greedy_info_density=g_density_info,
        greedy_frac_node=frac_node,
        greedy_frac_edge=frac_edge,
        greedy_frac_nav=frac_nav,
        greedy_frac_noop=frac_noop,
        canonical_string=c_str,
        canonical_length=c_len,
        canonical_entropy=c_entropy,
        canonical_entropy_normalized=c_norm,
        canonical_info_density=c_density_info,
        canonical_instr_counts=c_counts,
        info_lower_bound_bits=info_lb,
        efficiency_greedy=eff_greedy,
        efficiency_canonical=eff_canon,
        time_s=time.perf_counter() - t0,
    )


# Parallel worker
def _parallel_worker(
    args: tuple[int, str, str, int, list[tuple[int, int]]],
) -> EntropyRecord:
    """Reconstruct graph in subprocess and run test."""
    test_id, regime, family, num_nodes, edges = args
    nxg = nx.Graph()
    nxg.add_nodes_from(range(num_nodes))
    for u, v in edges:
        nxg.add_edge(u, v)
    return _run_single_test(test_id, regime, family, nxg)


# =============================================================================
# Benchmark runner
# =============================================================================


def run_benchmark(
    seed: int = DEFAULT_SEED,
    n_workers: int = 1,
) -> list[EntropyRecord]:
    """Run the full benchmark."""
    graphs = _generate_test_graphs(seed)
    results: list[EntropyRecord] = []

    if n_workers <= 1:
        for i, (regime, family, nxg) in enumerate(graphs):
            rec = _run_single_test(i, regime, family, nxg)
            results.append(rec)
            if (i + 1) % max(1, len(graphs) // 10) == 0:
                logger.info("  Progress: %d/%d tests", i + 1, len(graphs))
    else:
        tasks = [
            (i, regime, family, nxg.number_of_nodes(), list(nxg.edges()))
            for i, (regime, family, nxg) in enumerate(graphs)
        ]
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
    results: list[EntropyRecord],
    output_dir: str,
    seed: int,
) -> None:
    """Save JSON summary."""
    valid = [r for r in results if r.error == ""]
    regime_a = [r for r in valid if r.regime == "A"]
    regime_b = [r for r in valid if r.regime == "B"]

    families = sorted(set(r.family for r in valid))
    by_family: dict[str, Any] = {}
    for fam in families:
        recs = [r for r in valid if r.family == fam]
        by_family[fam] = {
            "count": len(recs),
            "mean_greedy_entropy": float(np.mean([r.greedy_entropy for r in recs])),
            "mean_greedy_frac_nav": float(np.mean([r.greedy_frac_nav for r in recs])),
            "mean_greedy_frac_node": float(np.mean([r.greedy_frac_node for r in recs])),
            "mean_greedy_frac_edge": float(np.mean([r.greedy_frac_edge for r in recs])),
            "mean_efficiency_greedy": float(np.mean([r.efficiency_greedy for r in recs])),
        }
        canon_recs = [r for r in recs if r.canonical_length > 0]
        if canon_recs:
            by_family[fam]["mean_canonical_entropy"] = float(
                np.mean([r.canonical_entropy for r in canon_recs])
            )
            by_family[fam]["mean_efficiency_canonical"] = float(
                np.mean([r.efficiency_canonical for r in canon_recs])
            )

    obj = {
        "benchmark": "alphabet_entropy",
        "config": {"seed": seed, "H_max": H_MAX},
        "summary": {
            "total_tests": len(results),
            "valid_tests": len(valid),
            "regime_A_tests": len(regime_a),
            "regime_B_tests": len(regime_b),
            "mean_greedy_entropy": float(np.mean([r.greedy_entropy for r in valid]))
            if valid
            else 0.0,
            "mean_greedy_frac_nav": float(np.mean([r.greedy_frac_nav for r in valid]))
            if valid
            else 0.0,
        },
        "results_by_family": by_family,
    }

    path = os.path.join(output_dir, "alphabet_entropy_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Saved JSON: %s", path)


def save_csv(
    results: list[EntropyRecord],
    output_dir: str,
) -> None:
    """Save per-test CSV."""
    rows = []
    for r in results:
        rows.append(
            {
                "test_id": r.test_id,
                "family": r.family,
                "regime": r.regime,
                "num_nodes": r.num_nodes,
                "num_edges": r.num_edges,
                "density": round(r.density, 4),
                "greedy_length": r.greedy_length,
                "greedy_entropy": round(r.greedy_entropy, 4),
                "greedy_entropy_normalized": round(r.greedy_entropy_normalized, 4),
                "greedy_frac_node": round(r.greedy_frac_node, 4),
                "greedy_frac_edge": round(r.greedy_frac_edge, 4),
                "greedy_frac_nav": round(r.greedy_frac_nav, 4),
                "greedy_frac_noop": round(r.greedy_frac_noop, 4),
                "canonical_length": r.canonical_length,
                "canonical_entropy": round(r.canonical_entropy, 4)
                if r.canonical_entropy >= 0
                else "",
                "info_lower_bound_bits": round(r.info_lower_bound_bits, 2),
                "efficiency_greedy": round(r.efficiency_greedy, 4),
                "efficiency_canonical": round(r.efficiency_canonical, 4)
                if r.efficiency_canonical >= 0
                else "",
                "time_s": round(r.time_s, 4),
                "error": r.error,
            }
        )
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "alphabet_entropy_results.csv")
    df.to_csv(path, index=False)
    logger.info("Saved CSV: %s", path)


# =============================================================================
# Population figure (2x2)
# =============================================================================


def generate_population_figure(
    results: list[EntropyRecord],
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

    # ---- Panel (a): Greedy entropy H(w') vs N by family ----
    ax = axes[0, 0]
    for fam in families:
        recs = [r for r in valid if r.family == fam]
        by_n: dict[int, list[float]] = {}
        for r in recs:
            by_n.setdefault(r.num_nodes, []).append(r.greedy_entropy)
        ns = sorted(by_n.keys())
        means = [float(np.mean(by_n[n_])) for n_ in ns]
        stds = [float(np.std(by_n[n_])) for n_ in ns]
        ax.errorbar(
            ns,
            means,
            yerr=stds,
            color=FAMILY_COLORS.get(fam, "#BBBBBB"),
            marker="o",
            markersize=3,
            lw=PLOT_SETTINGS["line_width"],
            capsize=2,
            label=family_display(fam),
            alpha=0.8,
        )
    ax.axhline(H_MAX, color="0.4", ls="--", lw=0.8, label=f"$H_{{max}}$={H_MAX:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of nodes $N$")
    ax.set_ylabel("Shannon entropy $H(w')$ [bits]")
    ax.set_title("(a) Greedy entropy vs $N$", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax.legend(fontsize=5, ncol=2, loc="lower right", frameon=False)

    # ---- Panel (b): Instruction composition vs N ----
    ax = axes[0, 1]
    # Aggregate by family and N for a few representative families
    rep_families = ["tree", "complete", "gnp_p0.3"]
    available_rep = [f for f in rep_families if f in families]
    if not available_rep:
        available_rep = families[:3]

    width = 0.25
    for idx, fam in enumerate(available_rep):
        recs = sorted([r for r in valid if r.family == fam], key=lambda r: r.num_nodes)
        by_n: dict[int, list[EntropyRecord]] = {}
        for r in recs:
            by_n.setdefault(r.num_nodes, []).append(r)
        ns_list = sorted(by_n.keys())
        # Take up to 6 representative N values
        if len(ns_list) > 6:
            step = max(1, len(ns_list) // 6)
            ns_list = ns_list[::step][:6]

        nav_fracs = [float(np.mean([r.greedy_frac_nav for r in by_n[n_]])) for n_ in ns_list]
        node_fracs = [float(np.mean([r.greedy_frac_node for r in by_n[n_]])) for n_ in ns_list]
        edge_fracs = [float(np.mean([r.greedy_frac_edge for r in by_n[n_]])) for n_ in ns_list]

        x = np.arange(len(ns_list))
        ax.bar(
            x + idx * width,
            nav_fracs,
            width,
            color=PAUL_TOL_BRIGHT["blue"],
            alpha=0.7,
            label="Nav" if idx == 0 else "",
            edgecolor="0.3",
            linewidth=0.3,
        )
        ax.bar(
            x + idx * width,
            node_fracs,
            width,
            bottom=nav_fracs,
            color=PAUL_TOL_BRIGHT["green"],
            alpha=0.7,
            label="Node" if idx == 0 else "",
            edgecolor="0.3",
            linewidth=0.3,
        )
        bottom2 = [a + b for a, b in zip(nav_fracs, node_fracs)]
        ax.bar(
            x + idx * width,
            edge_fracs,
            width,
            bottom=bottom2,
            color=PAUL_TOL_BRIGHT["red"],
            alpha=0.7,
            label="Edge" if idx == 0 else "",
            edgecolor="0.3",
            linewidth=0.3,
        )

    if available_rep:
        combined_ns = set()
        for fam in available_rep:
            for r in valid:
                if r.family == fam:
                    combined_ns.add(r.num_nodes)
        n_labels = sorted(combined_ns)
        if len(n_labels) > 6:
            step = max(1, len(n_labels) // 6)
            n_labels = n_labels[::step][:6]
        ax.set_xticks(range(len(n_labels)))
        ax.set_xticklabels([str(n_) for n_ in n_labels], fontsize=PLOT_SETTINGS["tick_labelsize"])

    ax.set_xlabel("$N$")
    ax.set_ylabel("Fraction of string")
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], frameon=False, loc="upper right")
    ax.set_title("(b) Instruction composition", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # ---- Panel (c): Canonical vs greedy info density (Regime A) ----
    ax = axes[1, 0]
    regime_a = [r for r in valid if r.regime == "A" and r.canonical_length > 0]
    if regime_a:
        for fam in families:
            recs = [r for r in regime_a if r.family == fam]
            if not recs:
                continue
            g_ent = [r.greedy_entropy for r in recs]
            c_ent = [r.canonical_entropy for r in recs]
            ax.scatter(
                c_ent,
                g_ent,
                c=FAMILY_COLORS.get(fam, "#BBBBBB"),
                s=PLOT_SETTINGS["scatter_size"] * 2,
                alpha=PLOT_SETTINGS["scatter_alpha"],
                label=family_display(fam),
                edgecolors="0.3",
                linewidths=PLOT_SETTINGS["scatter_edgewidth"],
            )
        lims = [0, max(H_MAX, max(r.greedy_entropy for r in regime_a) + 0.2)]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="$y=x$")
        ax.set_xlabel("$H(w^*)$ [bits]")
        ax.set_ylabel("$H(w')$ [bits]")
        ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.set_title("(c) Canonical vs greedy entropy", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # ---- Panel (d): Efficiency ratio vs density (Regime A) ----
    ax = axes[1, 1]
    if regime_a:
        densities = [r.density for r in regime_a]
        eff_g = [r.efficiency_greedy for r in regime_a]
        nodes_a = [r.num_nodes for r in regime_a]
        sc = ax.scatter(
            densities,
            eff_g,
            c=nodes_a,
            cmap="viridis",
            s=PLOT_SETTINGS["scatter_size"] * 2,
            alpha=PLOT_SETTINGS["scatter_alpha"],
            edgecolors="0.3",
            linewidths=PLOT_SETTINGS["scatter_edgewidth"],
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("$N$", fontsize=PLOT_SETTINGS["axes_labelsize"])
        ax.axhline(1.0, color="0.4", ls="--", lw=0.8, label="$\\eta=1$ (optimal)")
        ax.set_xlabel(r"Edge density $\rho$")
        ax.set_ylabel(r"Efficiency $\eta_{greedy}$")
        ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], frameon=False)
    ax.set_title("(d) Encoding efficiency", fontsize=PLOT_SETTINGS["axes_titlesize"])

    fig.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "alphabet_entropy_population")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Saved population figure: %s.pdf", path)


# =============================================================================
# Individual figure (1x3)
# =============================================================================


def _select_showcase(results: list[EntropyRecord]) -> EntropyRecord | None:
    """Select a Regime A case with meaningful entropy difference."""
    candidates = [
        r
        for r in results
        if r.error == ""
        and r.regime == "A"
        and r.canonical_length > 0
        and 5 <= r.num_nodes <= 8
        and abs(r.greedy_entropy - r.canonical_entropy) > 0.1
    ]
    if not candidates:
        candidates = [
            r for r in results if r.error == "" and r.regime == "A" and r.canonical_length > 0
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: abs(r.greedy_entropy - r.canonical_entropy))


def generate_single_case_figure(
    results: list[EntropyRecord],
    output_dir: str,
) -> None:
    """Generate individual-case figure (1x3)."""
    apply_ieee_style()
    rec = _select_showcase(results)
    if rec is None:
        logger.warning("No suitable case for individual figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=get_figure_size("double", 0.45))

    # ---- Panel (a): Pie chart for greedy ----
    ax = axes[0]
    g_counts = rec.greedy_instr_counts
    labels = [k for k in ALPHABET if g_counts.get(k, 0) > 0]
    sizes = [g_counts[k] for k in labels]
    colors = [INSTRUCTION_COLORS.get(k, "#BBBBBB") for k in labels]
    if sizes:
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": PLOT_SETTINGS["tick_labelsize"]},
        )
    ax.set_title(
        f"(a) Greedy w'\nH={rec.greedy_entropy:.2f} bits, |w'|={rec.greedy_length}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    # ---- Panel (b): Pie chart for canonical ----
    ax = axes[1]
    c_counts = rec.canonical_instr_counts
    labels_c = [k for k in ALPHABET if c_counts.get(k, 0) > 0]
    sizes_c = [c_counts[k] for k in labels_c]
    colors_c = [INSTRUCTION_COLORS.get(k, "#BBBBBB") for k in labels_c]
    if sizes_c:
        ax.pie(
            sizes_c,
            labels=labels_c,
            colors=colors_c,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": PLOT_SETTINGS["tick_labelsize"]},
        )
    ax.set_title(
        f"(b) Canonical w*\nH={rec.canonical_entropy:.2f} bits, |w*|={rec.canonical_length}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    # ---- Panel (c): Grouped bar chart ----
    ax = axes[2]
    instrs = [k for k in ALPHABET if g_counts.get(k, 0) > 0 or c_counts.get(k, 0) > 0]
    x = np.arange(len(instrs))
    w_bar = 0.35
    ax.bar(
        x - w_bar / 2,
        [g_counts.get(i, 0) for i in instrs],
        w_bar,
        label=f"Greedy (H={rec.greedy_entropy:.2f})",
        color=PAUL_TOL_BRIGHT["blue"],
        alpha=0.8,
        edgecolor="0.3",
        linewidth=0.5,
    )
    ax.bar(
        x + w_bar / 2,
        [c_counts.get(i, 0) for i in instrs],
        w_bar,
        label=f"Canonical (H={rec.canonical_entropy:.2f})",
        color=PAUL_TOL_BRIGHT["red"],
        alpha=0.8,
        edgecolor="0.3",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(instrs, fontsize=PLOT_SETTINGS["tick_labelsize"])
    ax.set_xlabel("Instruction")
    ax.set_ylabel("Count")
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], frameon=False)
    ax.set_title(
        f"(c) {family_display(rec.family)}, N={rec.num_nodes}",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    fig.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "alphabet_entropy_single_case")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Saved single-case figure: %s.pdf", path)


# =============================================================================
# LaTeX table
# =============================================================================


def generate_table(
    results: list[EntropyRecord],
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
        canon_recs = [r for r in recs if r.canonical_length > 0]
        n_range = f"{min(r.num_nodes for r in recs)}-{max(r.num_nodes for r in recs)}"
        rows.append(
            {
                "Family": family_display(fam),
                "N range": n_range,
                "Mean H_greedy": f"{np.mean([r.greedy_entropy for r in recs]):.2f}",
                "Mean H_canon": f"{np.mean([r.canonical_entropy for r in canon_recs]):.2f}"
                if canon_recs
                else "---",
                "H_max": f"{H_MAX:.2f}",
                "Mean eta_greedy": f"{np.mean([r.efficiency_greedy for r in recs]):.3f}",
                "Mean eta_canon": f"{np.mean([r.efficiency_canonical for r in canon_recs]):.3f}"
                if canon_recs
                else "---",
                "Mean nav%": f"{100 * np.mean([r.greedy_frac_nav for r in recs]):.1f}",
                "Mean node%": f"{100 * np.mean([r.greedy_frac_node for r in recs]):.1f}",
                "Mean edge%": f"{100 * np.mean([r.greedy_frac_edge for r in recs]):.1f}",
            }
        )

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "alphabet_entropy_table.tex")
    save_latex_table(
        df,
        path,
        caption="Alphabet utilization and entropy analysis by graph family. "
        "H = Shannon entropy in bits; $\\eta$ = encoding efficiency ratio; "
        "nav/node/edge = fraction of instructions by type.",
        label="tab:alphabet_entropy",
    )
    logger.info("Saved table: %s", path)


# =============================================================================
# Main / CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alphabet Utilization / Entropy Analysis benchmark",
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

    logger.info("=== Alphabet Utilization / Entropy Analysis ===")
    logger.info("Seed=%d, workers=%d, output=%s", args.seed, args.n_workers, args.output_dir)

    t0 = time.perf_counter()
    results = run_benchmark(seed=args.seed, n_workers=args.n_workers)
    elapsed = time.perf_counter() - t0

    valid = [r for r in results if r.error == ""]
    logger.info("Done: %d tests (%d valid) in %.1fs", len(results), len(valid), elapsed)
    if valid:
        logger.info(
            "  Mean greedy entropy: %.3f bits (H_max=%.2f)",
            np.mean([r.greedy_entropy for r in valid]),
            H_MAX,
        )
        logger.info(
            "  Mean nav fraction: %.1f%%", 100 * np.mean([r.greedy_frac_nav for r in valid])
        )
        regime_a = [r for r in valid if r.regime == "A" and r.canonical_length > 0]
        if regime_a:
            logger.info(
                "  Regime A: mean canonical H=%.3f, mean efficiency=%.3f",
                np.mean([r.canonical_entropy for r in regime_a]),
                np.mean([r.efficiency_canonical for r in regime_a]),
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
