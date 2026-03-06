"""String Processing Pipeline benchmark: w -> G -> w'_v -> w*.

Traces the full IsalGraph encoding pipeline for each test graph:

1. Start with a random instruction string w (or encode an NX graph via G2S).
2. Decode to graph G = S2G(w).
3. Compute greedy encodings w'_v = G2S(G, v) for **every** starting node v.
4. Compute the canonical string w* = canonical(G).

Reports normalization (|w| vs |w'_best|), greedy optimality gap (|w'_best|/|w*|),
starting-node sensitivity, and representation-space geometry (Levenshtein distances).

Usage:
    python benchmarks/string_pipeline/string_pipeline.py --num-tests 30 --seed 42 \
        --output-dir /tmp/pipeline_test --csv --plot --table
    python benchmarks/string_pipeline/string_pipeline.py --num-tests 200 --seed 42 \
        --output-dir /media/mpascual/Sandisk2TB/research/isalgraph --csv --plot --table
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.canonical import canonical_string, levenshtein
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_NUM_TESTS = 200
DEFAULT_MAX_NODES = 8
DEFAULT_MAX_STRING_LEN = 15
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_NODE_INSTRS = ["V", "v"]
_MOVE_INSTRS = ["N", "n", "P", "p"]
_EDGE_INSTRS = ["C", "c"]


@dataclass
class PipelineResult:
    """Result of a single pipeline test."""

    test_id: int
    source: str
    num_nodes: int
    num_edges: int
    directed: bool
    w_random: str
    w_greedy_best: str
    w_canonical: str
    greedy_strings: list[tuple[int, str]]
    len_w_random: int
    len_w_greedy_best: int
    len_w_canonical: int
    lev_w_wbest: int
    lev_wbest_wcanon: int
    lev_w_wcanon: int
    optimality_ratio: float
    time_s: float
    error: str = ""


@dataclass
class BenchmarkSummary:
    """Aggregate results."""

    total_tests: int = 0
    total_time_s: float = 0.0
    results_by_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Random string generation (same logic as random_roundtrip)
# ---------------------------------------------------------------------------


def _generate_random_string(rng: random.Random, length: int) -> str:
    """Generate a random valid IsalGraph instruction string."""
    if length <= 0:
        return ""
    instructions: list[str] = []
    node_count = 1
    for i in range(length):
        if i == 0 and length <= 3:
            instr = rng.choice(_NODE_INSTRS)
        else:
            pool = list(_NODE_INSTRS) + list(_MOVE_INSTRS)
            if node_count >= 2:
                pool.extend(_EDGE_INSTRS)
            instr = rng.choice(_NODE_INSTRS) if rng.random() < 0.4 else rng.choice(pool)
        instructions.append(instr)
        if instr in ("V", "v"):
            node_count += 1
    return "".join(instructions)


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


def _generate_test_inputs(
    rng: random.Random,
    num_tests: int,
    max_nodes: int,
    max_string_len: int,
) -> list[tuple[str, Any, bool]]:
    """Generate (source, graph_or_string, directed) tuples.

    Returns a mix of random strings and NX graph families.
    For random strings, the second element is the string itself.
    For NX families, the second element is an nx.Graph.
    """
    inputs: list[tuple[str, Any, bool]] = []

    # --- Random strings (~50%) ---
    n_random = num_tests // 2
    attempts = 0
    while (
        len([x for x in inputs if x[0] == "random_string"]) < n_random and attempts < n_random * 5
    ):
        attempts += 1
        length = rng.randint(3, max_string_len)
        w = _generate_random_string(rng, length)
        s2g = StringToGraph(w, directed_graph=False)
        sg, _ = s2g.run()
        if 2 <= sg.node_count() <= max_nodes:
            inputs.append(("random_string", w, False))

    # --- NX families (~50%) ---
    n_nx = num_tests - len(inputs)
    n_per_family = max(n_nx // 8, 1)

    # Trees
    for _ in range(n_per_family):
        n = rng.randint(3, max_nodes)
        seed = rng.randint(0, 2**31)
        g = nx.random_labeled_tree(n, seed=seed)
        inputs.append(("tree", g, False))

    # Paths
    for n in range(3, min(max_nodes + 1, 3 + n_per_family)):
        inputs.append(("path", nx.path_graph(n), False))

    # Cycles
    for n in range(3, min(max_nodes + 1, 3 + n_per_family)):
        inputs.append(("cycle", nx.cycle_graph(n), False))

    # Stars
    for n in range(4, min(max_nodes + 1, 4 + n_per_family)):
        inputs.append(("star", nx.star_graph(n - 1), False))

    # Complete (cap at N=6 for canonical speed)
    complete_max = min(max_nodes, 6)
    for n in range(3, min(complete_max + 1, 3 + n_per_family)):
        inputs.append(("complete", nx.complete_graph(n), False))

    # GNP (retry on disconnected)
    gnp_attempts = 0
    gnp_count = 0
    while gnp_count < n_per_family and gnp_attempts < n_per_family * 5:
        gnp_attempts += 1
        n = rng.randint(3, max_nodes)
        p = rng.uniform(0.2, 0.5)
        seed = rng.randint(0, 2**31)
        g = nx.gnp_random_graph(n, p, seed=seed)
        if nx.is_connected(g) and g.number_of_nodes() >= 3:
            inputs.append(("gnp", g, False))
            gnp_count += 1

    # Barabasi-Albert
    for _ in range(n_per_family):
        n = rng.randint(4, max_nodes)
        m = rng.choice([1, 2])
        seed = rng.randint(0, 2**31)
        g = nx.barabasi_albert_graph(n, m, seed=seed)
        inputs.append(("barabasi_albert", g, False))

    # Watts-Strogatz (retry on disconnected)
    ws_attempts = 0
    ws_count = 0
    while ws_count < n_per_family and ws_attempts < n_per_family * 5:
        ws_attempts += 1
        n = rng.randint(4, max_nodes)
        seed = rng.randint(0, 2**31)
        g = nx.watts_strogatz_graph(n, k=2, p=0.3, seed=seed)
        if nx.is_connected(g):
            inputs.append(("watts_strogatz", g, False))
            ws_count += 1

    return inputs


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(
    test_id: int,
    source: str,
    sg: Any,
    directed: bool,
    w_random_input: str,
) -> PipelineResult:
    """Run the full pipeline on a single SparseGraph.

    Args:
        test_id: Test identifier.
        source: Graph family name.
        sg: SparseGraph instance.
        directed: Whether the graph is directed.
        w_random_input: The input string (random string or G2S from node 0).

    Returns:
        PipelineResult with all pipeline metrics.
    """
    t0 = time.perf_counter()
    try:
        n = sg.node_count()
        m = sg.logical_edge_count()

        # Step 1: G2S from every starting node
        greedy_strings: list[tuple[int, str]] = []
        for v in range(n):
            try:
                g2s = GraphToString(sg)
                w_v, _ = g2s.run(initial_node=v)
                greedy_strings.append((v, w_v))
            except ValueError:
                continue

        if not greedy_strings:
            elapsed = time.perf_counter() - t0
            return PipelineResult(
                test_id=test_id,
                source=source,
                num_nodes=n,
                num_edges=m,
                directed=directed,
                w_random=w_random_input,
                w_greedy_best="",
                w_canonical="",
                greedy_strings=[],
                len_w_random=len(w_random_input),
                len_w_greedy_best=0,
                len_w_canonical=0,
                lev_w_wbest=0,
                lev_wbest_wcanon=0,
                lev_w_wcanon=0,
                optimality_ratio=1.0,
                time_s=elapsed,
                error="No valid starting node",
            )

        # Step 2: Best greedy (shortest, then lexmin)
        w_greedy_best = min(greedy_strings, key=lambda x: (len(x[1]), x[1]))[1]

        # Step 3: Canonical string
        w_canon = canonical_string(sg)

        # Step 4: Levenshtein distances
        lev_w_wbest = levenshtein(w_random_input, w_greedy_best)
        lev_wbest_wcanon = levenshtein(w_greedy_best, w_canon)
        lev_w_wcanon = levenshtein(w_random_input, w_canon)

        # Step 5: Optimality ratio
        opt_ratio = len(w_greedy_best) / len(w_canon) if len(w_canon) > 0 else 1.0

        elapsed = time.perf_counter() - t0
        return PipelineResult(
            test_id=test_id,
            source=source,
            num_nodes=n,
            num_edges=m,
            directed=directed,
            w_random=w_random_input,
            w_greedy_best=w_greedy_best,
            w_canonical=w_canon,
            greedy_strings=greedy_strings,
            len_w_random=len(w_random_input),
            len_w_greedy_best=len(w_greedy_best),
            len_w_canonical=len(w_canon),
            lev_w_wbest=lev_w_wbest,
            lev_wbest_wcanon=lev_wbest_wcanon,
            lev_w_wcanon=lev_w_wcanon,
            optimality_ratio=opt_ratio,
            time_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return PipelineResult(
            test_id=test_id,
            source=source,
            num_nodes=0,
            num_edges=0,
            directed=directed,
            w_random=w_random_input,
            w_greedy_best="",
            w_canonical="",
            greedy_strings=[],
            len_w_random=len(w_random_input),
            len_w_greedy_best=0,
            len_w_canonical=0,
            lev_w_wbest=0,
            lev_wbest_wcanon=0,
            lev_w_wcanon=0,
            optimality_ratio=1.0,
            time_s=elapsed,
            error=f"Exception: {exc!r}",
        )


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------


def _parallel_worker(
    args_tuple: tuple[int, str, Any, bool, int],
) -> PipelineResult:
    """Worker for ProcessPoolExecutor.

    Receives (test_id, source, graph_or_string, directed, seed).
    Reconstructs SparseGraph inside worker for picklability.
    """
    test_id, source, payload, directed, _seed = args_tuple
    adapter = NetworkXAdapter()

    if source == "random_string":
        w_random = payload
        s2g = StringToGraph(w_random, directed_graph=directed)
        sg, _ = s2g.run()
    else:
        nx_graph = payload
        sg = adapter.from_external(nx_graph, directed=directed)
        g2s = GraphToString(sg)
        w0, _ = g2s.run(initial_node=0)
        w_random = w0

    return _run_pipeline(test_id, source, sg, directed, w_random)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_benchmark(
    num_tests: int,
    max_nodes: int,
    max_string_len: int,
    seed: int,
    output_dir: str,
) -> tuple[BenchmarkSummary, list[PipelineResult]]:
    """Run the string pipeline benchmark sequentially."""
    rng = random.Random(seed)
    adapter = NetworkXAdapter()

    print("String Pipeline Benchmark")
    print(f"{'=' * 60}")
    print(f"Tests: {num_tests}, Max nodes: {max_nodes}, Seed: {seed}")
    print(f"Output: {output_dir}")
    print()

    inputs = _generate_test_inputs(rng, num_tests, max_nodes, max_string_len)
    print(f"Generated {len(inputs)} test inputs.")

    all_results: list[PipelineResult] = []
    for i, (source, payload, directed) in enumerate(inputs):
        if source == "random_string":
            w_random = payload
            s2g = StringToGraph(w_random, directed_graph=directed)
            sg, _ = s2g.run()
        else:
            sg = adapter.from_external(payload, directed=directed)
            g2s = GraphToString(sg)
            w0, _ = g2s.run(initial_node=0)
            w_random = w0

        result = _run_pipeline(i, source, sg, directed, w_random)
        all_results.append(result)

        if (i + 1) % max(1, len(inputs) // 10) == 0:
            print(f"  ... {i + 1}/{len(inputs)} done ({result.time_s:.2f}s)")

    return _finalize(all_results, num_tests, max_nodes, seed, output_dir)


def run_benchmark_parallel(
    num_tests: int,
    max_nodes: int,
    max_string_len: int,
    seed: int,
    output_dir: str,
    n_workers: int = 4,
) -> tuple[BenchmarkSummary, list[PipelineResult]]:
    """Run the string pipeline benchmark with ProcessPoolExecutor."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rng = random.Random(seed)

    print(f"String Pipeline Benchmark (parallel, {n_workers} workers)")
    print(f"{'=' * 60}")
    print(f"Tests: {num_tests}, Max nodes: {max_nodes}, Seed: {seed}")
    print()

    inputs = _generate_test_inputs(rng, num_tests, max_nodes, max_string_len)
    print(f"Generated {len(inputs)} test inputs.")

    tasks = []
    for i, (source, payload, directed) in enumerate(inputs):
        task_seed = rng.randint(0, 2**31)
        tasks.append((i, source, payload, directed, task_seed))

    all_results: list[PipelineResult] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_parallel_worker, t): t[0] for t in tasks}
        for done_count, fut in enumerate(as_completed(futures), 1):
            all_results.append(fut.result())
            if done_count % max(1, len(tasks) // 20) == 0:
                print(f"  ... {done_count}/{len(tasks)} done")

    return _finalize(all_results, num_tests, max_nodes, seed, output_dir)


def _finalize(
    all_results: list[PipelineResult],
    num_tests: int,
    max_nodes: int,
    seed: int,
    output_dir: str,
) -> tuple[BenchmarkSummary, list[PipelineResult]]:
    """Aggregate and save results."""
    summary = BenchmarkSummary()
    summary.total_tests = len(all_results)
    summary.total_time_s = sum(r.time_s for r in all_results)

    for r in all_results:
        key = r.source
        if key not in summary.results_by_source:
            summary.results_by_source[key] = {
                "count": 0,
                "errors": 0,
                "sum_len_w": 0,
                "sum_len_wbest": 0,
                "sum_len_wcanon": 0,
                "sum_opt_ratio": 0.0,
                "sum_lev_w_wbest": 0,
                "sum_lev_wbest_wcanon": 0,
            }
        stats = summary.results_by_source[key]
        if r.error:
            stats["errors"] += 1
            summary.errors.append({"test_id": r.test_id, "source": r.source, "error": r.error})
        else:
            stats["count"] += 1
            stats["sum_len_w"] += r.len_w_random
            stats["sum_len_wbest"] += r.len_w_greedy_best
            stats["sum_len_wcanon"] += r.len_w_canonical
            stats["sum_opt_ratio"] += r.optimality_ratio
            stats["sum_lev_w_wbest"] += r.lev_w_wbest
            stats["sum_lev_wbest_wcanon"] += r.lev_wbest_wcanon

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total tests: {summary.total_tests}")
    print(f"Total time:  {summary.total_time_s:.2f}s")
    print(f"Errors:      {len(summary.errors)}")
    print()

    valid = [r for r in all_results if not r.error]
    if valid:
        opt_ratios = [r.optimality_ratio for r in valid]
        greedy_optimal = sum(1 for r in valid if r.optimality_ratio == 1.0)
        pct = greedy_optimal / len(valid) * 100
        print(f"Greedy optimal: {greedy_optimal}/{len(valid)} ({pct:.1f}%)")
        print(f"Mean optimality ratio: {sum(opt_ratios) / len(opt_ratios):.3f}")
        print(f"Max optimality ratio:  {max(opt_ratios):.3f}")
    print()

    hdr_wbest = "Mean|w'|"
    print(
        f"{'Source':<20} {'N':>5} {'Mean|w|':>8} {hdr_wbest:>9} {'Mean|w*|':>9} {'Opt.ratio':>10}"
    )
    print(f"{'-' * 20} {'-' * 5} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 10}")
    for src in sorted(summary.results_by_source.keys()):
        s = summary.results_by_source[src]
        n = s["count"]
        if n == 0:
            continue
        print(
            f"{src:<20} {n:>5} {s['sum_len_w'] / n:>8.1f} "
            f"{s['sum_len_wbest'] / n:>9.1f} {s['sum_len_wcanon'] / n:>9.1f} "
            f"{s['sum_opt_ratio'] / n:>10.3f}"
        )
    print()

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "string_pipeline_results.json")
    results_json = {
        "benchmark": "string_pipeline",
        "config": {"num_tests": num_tests, "max_nodes": max_nodes, "seed": seed},
        "summary": {
            "total_tests": summary.total_tests,
            "total_time_s": round(summary.total_time_s, 4),
            "n_errors": len(summary.errors),
        },
        "results_by_source": summary.results_by_source,
        "errors": summary.errors[:50],
    }
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: {output_path}")

    return summary, all_results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def save_csv(results: list[PipelineResult], output_dir: str) -> str:
    """Save raw results as CSV."""
    import csv as csv_mod

    rows = []
    for r in results:
        rows.append(
            {
                "test_id": r.test_id,
                "source": r.source,
                "num_nodes": r.num_nodes,
                "num_edges": r.num_edges,
                "directed": r.directed,
                "len_w_random": r.len_w_random,
                "len_w_greedy_best": r.len_w_greedy_best,
                "len_w_canonical": r.len_w_canonical,
                "lev_w_wbest": r.lev_w_wbest,
                "lev_wbest_wcanon": r.lev_wbest_wcanon,
                "lev_w_wcanon": r.lev_w_wcanon,
                "optimality_ratio": round(r.optimality_ratio, 4),
                "n_starting_nodes": len(r.greedy_strings),
                "time_s": round(r.time_s, 4),
                "error": r.error,
            }
        )

    path = os.path.join(output_dir, "string_pipeline_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# Instruction heatmap helpers
# ---------------------------------------------------------------------------

_INSTR_ORDER = "NPnpVvCcW"
_INSTR_TO_IDX = {ch: i for i, ch in enumerate(_INSTR_ORDER)}
_PAD_IDX = 9  # White padding (not a real instruction)


def _string_to_int_column(s: str, pad_char: str | None = None) -> list[int]:
    """Convert an instruction string to a list of integer indices.

    If pad_char is provided, that character maps to _PAD_IDX (white).
    """
    result = []
    for ch in s:
        if pad_char and ch == pad_char:
            result.append(_PAD_IDX)
        else:
            result.append(_INSTR_TO_IDX.get(ch, 8))
    return result


def _make_instruction_cmap():
    """Create a ListedColormap for 9 instructions + 1 white padding."""
    from matplotlib.colors import ListedColormap

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from plotting_styles import INSTRUCTION_COLORS  # noqa: E402

    colors = [INSTRUCTION_COLORS[ch] for ch in _INSTR_ORDER]
    colors.append("#FFFFFF")  # index 9 = white padding
    return ListedColormap(colors, name="isalgraph_instructions")


def _draw_string_heatmap(ax, string: str, title: str, cmap, show_ylabel: bool = True) -> None:
    """Draw a single instruction string as a vertical heatmap."""
    import numpy as np

    col = np.array(_string_to_int_column(string)).reshape(-1, 1)
    ax.imshow(col, cmap=cmap, vmin=0, vmax=9, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([])
    if show_ylabel:
        ax.set_ylabel("Instruction position", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)


# ---------------------------------------------------------------------------
# Population figure (1x3)
# ---------------------------------------------------------------------------


def generate_population_figure(results: list[PipelineResult], output_dir: str) -> list[str]:
    """Generate the 1x3 population-level figure.

    Panel (a): |w| vs |w'_best| scatter (normalization).
    Panel (b): Optimality ratio |w'_best|/|w*| boxplot by family.
    Panel (c): Mean normalized Levenshtein distances by family.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    from plotting_styles import (  # noqa: E402
        FAMILY_COLORS,
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        family_display,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()
    w, h = get_figure_size("double", 0.55)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(w, h))

    valid = [r for r in results if not r.error and r.len_w_canonical > 0]
    _default_color = "#555555"

    # =================================================================
    # Panel (a): |w| vs |w'_best| (random_string tests only)
    # =================================================================
    rand_results = [r for r in valid if r.source == "random_string"]
    if rand_results:
        lens_w = [r.len_w_random for r in rand_results]
        lens_wbest = [r.len_w_greedy_best for r in rand_results]
        nodes = [r.num_nodes for r in rand_results]

        sc = ax1.scatter(
            lens_w,
            lens_wbest,
            c=nodes,
            cmap="viridis",
            s=PLOT_SETTINGS["scatter_size"] * 1.5,
            alpha=PLOT_SETTINGS["scatter_alpha"],
            edgecolors="none",
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax1, shrink=0.7, pad=0.02)
        cbar.set_label("$N$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        max_val = max(max(lens_w), max(lens_wbest)) * 1.1
        ax1.plot(
            [0, max_val],
            [0, max_val],
            color="0.4",
            linewidth=PLOT_SETTINGS["line_width"],
            linestyle="--",
            zorder=1,
        )

        compressed = sum(1 for lw, lb in zip(lens_w, lens_wbest, strict=True) if lb < lw)
        pct = compressed / len(rand_results) * 100
        ratios = [lb / lw for lw, lb in zip(lens_w, lens_wbest, strict=True) if lw > 0]
        median_ratio = float(np.median(ratios)) if ratios else 1.0
        ax1.text(
            0.03,
            0.97,
            f"{pct:.0f}% compressed\nMedian $|w'|/|w|$ = {median_ratio:.2f}",
            transform=ax1.transAxes,
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.9,
            },
        )

    ax1.set_xlabel("Random string length $|w|$")
    ax1.set_ylabel("Best greedy length $|w'_{\\mathrm{best}}|$")
    ax1.text(-0.15, 1.05, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold")

    # =================================================================
    # Panel (b): Optimality ratio boxplot by family
    # =================================================================
    ratio_by_family: dict[str, list[float]] = {}
    for r in valid:
        ratio_by_family.setdefault(r.source, []).append(r.optimality_ratio)

    families_sorted = sorted(ratio_by_family.keys())
    box_data = [ratio_by_family[f] for f in families_sorted]
    x_pos = np.arange(1, len(families_sorted) + 1)

    bp = ax2.boxplot(
        box_data,
        positions=x_pos,
        widths=PLOT_SETTINGS["boxplot_width"],
        patch_artist=True,
        flierprops={
            "markersize": PLOT_SETTINGS["boxplot_flier_size"],
            "marker": ".",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        capprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        boxprops={"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
    )
    for patch, family in zip(bp["boxes"], families_sorted, strict=True):
        patch.set_facecolor(FAMILY_COLORS.get(family, _default_color))
        patch.set_alpha(0.6)
    for flier, family in zip(bp["fliers"], families_sorted, strict=True):
        flier.set_markerfacecolor(FAMILY_COLORS.get(family, _default_color))

    ax2.axhline(y=1.0, color="0.4", linewidth=PLOT_SETTINGS["line_width"], linestyle="--", zorder=1)

    greedy_optimal = sum(1 for r in valid if r.optimality_ratio == 1.0)
    ax2.text(
        0.97,
        0.97,
        f"Greedy optimal:\n{greedy_optimal}/{len(valid)}",
        transform=ax2.transAxes,
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [family_display(f) for f in families_sorted],
        rotation=45,
        ha="right",
        fontsize=PLOT_SETTINGS["tick_labelsize"],
    )
    ax2.set_ylabel("Optimality ratio $|w'_{\\mathrm{best}}|/|w^*|$")
    ax2.text(-0.15, 1.05, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold")

    # =================================================================
    # Panel (c): Grouped bars of normalized Levenshtein distances
    # =================================================================
    lev_data: dict[str, dict[str, list[float]]] = {}
    for r in valid:
        fam = r.source
        if fam not in lev_data:
            lev_data[fam] = {"w_wbest": [], "wbest_wcanon": [], "w_wcanon": []}
        if r.len_w_random > 0:
            lev_data[fam]["w_wbest"].append(r.lev_w_wbest / r.len_w_random)
            lev_data[fam]["w_wcanon"].append(r.lev_w_wcanon / r.len_w_random)
        if r.len_w_greedy_best > 0:
            lev_data[fam]["wbest_wcanon"].append(r.lev_wbest_wcanon / r.len_w_greedy_best)

    fam_order = sorted(lev_data.keys())
    bar_width = PLOT_SETTINGS["bar_width"]
    x = np.arange(len(fam_order))

    colors_3 = [PAUL_TOL_BRIGHT["blue"], PAUL_TOL_BRIGHT["green"], PAUL_TOL_BRIGHT["red"]]
    labels_3 = [
        r"$\mathrm{Lev}(w, w')/|w|$",
        r"$\mathrm{Lev}(w', w^*)/|w'|$",
        r"$\mathrm{Lev}(w, w^*)/|w|$",
    ]
    for j, metric_key in enumerate(["w_wbest", "wbest_wcanon", "w_wcanon"]):
        means = []
        for fam in fam_order:
            vals = lev_data[fam][metric_key]
            means.append(float(np.mean(vals)) if vals else 0.0)
        ax3.bar(
            x + (j - 1) * bar_width,
            means,
            bar_width,
            color=colors_3[j],
            alpha=PLOT_SETTINGS["bar_alpha"],
            label=labels_3[j],
        )

    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [family_display(f) for f in fam_order],
        rotation=45,
        ha="right",
        fontsize=PLOT_SETTINGS["tick_labelsize"],
    )
    ax3.set_ylabel("Normalized Levenshtein distance")
    ax3.legend(fontsize=6, loc="upper left", frameon=False)
    ax3.text(-0.15, 1.05, "(c)", transform=ax3.transAxes, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, wspace=0.4)
    paths = save_figure(fig, os.path.join(output_dir, "string_pipeline_population"))
    plt.close(fig)
    print(f"Population figure saved: {paths}")
    return paths


# ---------------------------------------------------------------------------
# Single-case figure (1x4) -- the paper figure
# ---------------------------------------------------------------------------


def _select_showcase(results: list[PipelineResult]) -> PipelineResult | None:
    """Pick the best single-case example for visualization."""
    candidates = [
        r
        for r in results
        if not r.error
        and r.source == "random_string"
        and 5 <= r.num_nodes <= 7
        and len(r.greedy_strings) >= 4
        and r.len_w_canonical > 0
    ]
    if not candidates:
        candidates = [
            r
            for r in results
            if not r.error
            and r.num_nodes >= 3
            and len(r.greedy_strings) >= 3
            and r.len_w_canonical > 0
        ]
    if not candidates:
        return None

    def _score(r: PipelineResult) -> float:
        # Prefer: suboptimal greedy, more nodes, more variation in greedy strings
        ratio_bonus = min(r.optimality_ratio - 1.0, 0.5) * 10
        node_bonus = r.num_nodes * 0.5
        n_strings = len(r.greedy_strings)
        lengths = [len(s) for _, s in r.greedy_strings]
        variation = (max(lengths) - min(lengths)) / max(max(lengths), 1) if lengths else 0
        return ratio_bonus + node_bonus + variation * 5 + n_strings * 0.2

    return max(candidates, key=_score)


def generate_single_case_figure(results: list[PipelineResult], output_dir: str) -> list[str]:
    """Generate the 1x4 single-case paper figure.

    Panel (a): w (random) as vertical heatmap.
    Panel (b): Graph G = S2G(w).
    Panel (c): Clustered heatmap of all w'_v.
    Panel (d): w* (canonical) as vertical heatmap.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np  # noqa: E402
    from matplotlib.patches import Patch  # noqa: E402
    from plotting_styles import (  # noqa: E402
        INSTRUCTION_COLORS,
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()

    result = _select_showcase(results)
    if result is None:
        print("WARNING: No suitable showcase found. Skipping single-case figure.")
        return []

    print(
        f"Showcase: test_id={result.test_id}, source={result.source}, "
        f"N={result.num_nodes}, M={result.num_edges}, "
        f"|w|={result.len_w_random}, |w'|={result.len_w_greedy_best}, "
        f"|w*|={result.len_w_canonical}, ratio={result.optimality_ratio:.2f}"
    )

    cmap = _make_instruction_cmap()

    # Reconstruct graph for plotting
    s2g = StringToGraph(result.w_random, directed_graph=result.directed)
    sg, _ = s2g.run()
    adapter = NetworkXAdapter()
    nx_graph = adapter.to_external(sg)

    # Prepare greedy strings for heatmap, cluster by Levenshtein distance
    greedy_strs = result.greedy_strings
    max_len = max(len(s) for _, s in greedy_strs)
    # Pad shorter strings with sentinel '_' (rendered as white, not W)
    padded = [(v, s + "_" * (max_len - len(s))) for v, s in greedy_strs]

    # Compute pairwise Levenshtein distance and cluster
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n_strings = len(padded)
    linkage_matrix = None
    if n_strings > 1:
        dist_matrix = np.zeros((n_strings, n_strings))
        for i in range(n_strings):
            for j in range(i + 1, n_strings):
                d = levenshtein(padded[i][1], padded[j][1])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        condensed = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method="average")
        order = leaves_list(linkage_matrix)
    else:
        order = [0]

    padded_ordered = [padded[i] for i in order]

    # Build figure using GridSpec: 4 columns, the heatmap column split into
    # dendrogram (top) + heatmap (bottom) rows.
    from matplotlib.gridspec import GridSpec
    from scipy.cluster.hierarchy import dendrogram

    w_fig, h_fig = get_figure_size("double", 0.75)
    fig = plt.figure(figsize=(w_fig, h_fig))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1, 2.5, 3, 1],
        height_ratios=[1, 5],
        hspace=0.05,
    )

    # Panels (a), (b), (d) span both rows
    ax_w = fig.add_subplot(gs[:, 0])
    ax_graph = fig.add_subplot(gs[:, 1])
    ax_dendro = fig.add_subplot(gs[0, 2])
    ax_heatmap = fig.add_subplot(gs[1, 2])
    ax_canon = fig.add_subplot(gs[:, 3])

    # ---- Panel (a): w (random input) ----
    _draw_string_heatmap(ax_w, result.w_random, "$w$", cmap)

    # ---- Panel (b): Graph G ----
    ax_graph.set_aspect("equal")
    pos = nx.spring_layout(nx_graph, seed=42)
    nx.draw(
        nx_graph,
        pos,
        ax=ax_graph,
        node_size=150,
        node_color=PAUL_TOL_BRIGHT["blue"],
        edge_color="0.5",
        with_labels=True,
        font_size=7,
        font_color="white",
        width=PLOT_SETTINGS["line_width"],
    )
    ax_graph.set_title(
        f"$G = \\mathrm{{S2G}}(w)$\n$N={result.num_nodes},\\; M={result.num_edges}$",
        fontsize=9,
        pad=4,
    )

    # ---- Panel (c) top: Dendrogram ----
    if n_strings > 1 and linkage_matrix is not None:
        dendrogram(
            linkage_matrix,
            ax=ax_dendro,
            orientation="top",
            no_labels=True,
            above_threshold_color="0.5",
            color_threshold=0,
        )
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    ax_dendro.spines["top"].set_visible(False)
    ax_dendro.spines["right"].set_visible(False)
    ax_dendro.spines["bottom"].set_visible(False)
    ax_dendro.spines["left"].set_visible(False)
    ax_dendro.set_title("$w'_v$ (all starting nodes)", fontsize=9, pad=4)

    # ---- Panel (c) bottom: Clustered heatmap ----
    mat = np.array(
        [_string_to_int_column(s, pad_char="_") for _, s in padded_ordered]
    ).T  # rows=positions, cols=nodes
    ax_heatmap.imshow(mat, cmap=cmap, vmin=0, vmax=9, aspect="auto", interpolation="nearest")
    ax_heatmap.set_xlabel("Starting node $v$", fontsize=8)
    ax_heatmap.set_xticks(range(n_strings))
    ax_heatmap.set_xticklabels([str(padded_ordered[i][0]) for i in range(n_strings)], fontsize=7)
    ax_heatmap.set_ylabel("Instruction position", fontsize=8)
    ax_heatmap.tick_params(axis="y", labelsize=7)

    # ---- Panel (d): w* (canonical) ----
    _draw_string_heatmap(ax_canon, result.w_canonical, "$w^*$", cmap, show_ylabel=False)

    # ---- Shared legend ----
    legend_patches = [
        Patch(facecolor=INSTRUCTION_COLORS[ch], edgecolor="0.3", label=ch) for ch in _INSTR_ORDER
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=9,
        fontsize=7,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    # Panel labels — all at the same y-height (top of figure)
    # Use the dendrogram top as reference for consistent alignment
    label_y = 1.05
    for ax, label in zip(
        [ax_w, ax_graph, ax_dendro, ax_canon], ["(a)", "(b)", "(c)", "(d)"], strict=True
    ):
        # Convert dendrogram's label_y to this axes' coordinate system
        fig_point = ax_dendro.transAxes.transform((0, label_y))
        ax_point = ax.transAxes.inverted().transform(fig_point)
        ax.text(-0.1, ax_point[1], label, transform=ax.transAxes, fontsize=11, fontweight="bold")

    gs.tight_layout(fig)
    fig.subplots_adjust(bottom=0.1)
    paths = save_figure(fig, os.path.join(output_dir, "string_pipeline_single_case"))
    plt.close(fig)
    print(f"Single-case figure saved: {paths}")
    return paths


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def generate_table(results: list[PipelineResult], output_dir: str) -> str:
    """Generate LaTeX table: pipeline statistics by graph family."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import pandas as pd  # noqa: E402
    from plotting_styles import save_latex_table  # noqa: E402

    valid = [r for r in results if not r.error and r.len_w_canonical > 0]

    family_data: dict[str, list[PipelineResult]] = {}
    for r in valid:
        family_data.setdefault(r.source, []).append(r)

    rows = []
    for fam in sorted(family_data.keys()):
        fam_results = family_data[fam]
        n = len(fam_results)
        mean_w = sum(r.len_w_random for r in fam_results) / n
        mean_wbest = sum(r.len_w_greedy_best for r in fam_results) / n
        mean_wcanon = sum(r.len_w_canonical for r in fam_results) / n
        mean_opt = sum(r.optimality_ratio for r in fam_results) / n
        mean_lev_w_wb = sum(r.lev_w_wbest for r in fam_results) / n
        mean_lev_wb_wc = sum(r.lev_wbest_wcanon for r in fam_results) / n
        mean_nodes = sum(r.num_nodes for r in fam_results) / n

        rows.append(
            {
                "Family": fam.replace("_", " ").title(),
                "Tests": n,
                "Nodes": f"{mean_nodes:.1f}",
                "|w|": f"{mean_w:.1f}",
                "|w'|": f"{mean_wbest:.1f}",
                "|w*|": f"{mean_wcanon:.1f}",
                "Ratio": f"{mean_opt:.3f}",
                "Lev(w,w')": f"{mean_lev_w_wb:.1f}",
                "Lev(w',w*)": f"{mean_lev_wb_wc:.1f}",
            }
        )

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "string_pipeline_table.tex")
    save_latex_table(
        df,
        path,
        caption=(
            "String pipeline statistics by graph family. "
            "\\textit{|w|}: input string length, "
            "\\textit{|w'|}: best greedy length, "
            "\\textit{|w*|}: canonical string length, "
            "\\textit{Ratio}: $|w'_{\\mathrm{best}}|/|w^*|$."
        ),
        label="tab:string_pipeline",
    )
    print(f"Table saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="String Processing Pipeline benchmark: w -> G -> w'_v -> w*"
    )
    parser.add_argument("--num-tests", type=int, default=DEFAULT_NUM_TESTS)
    parser.add_argument("--max-string-len", type=int, default=DEFAULT_MAX_STRING_LEN)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["local", "picasso"], default="local")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--csv", action="store_true", help="Save CSV")
    parser.add_argument("--plot", action="store_true", help="Generate figures")
    parser.add_argument("--table", action="store_true", help="Generate LaTeX table")
    args = parser.parse_args()

    if args.mode == "picasso":
        args.csv = True
        args.plot = True
        args.table = True

    if args.n_workers > 1:
        summary, results = run_benchmark_parallel(
            args.num_tests,
            args.max_nodes,
            args.max_string_len,
            args.seed,
            args.output_dir,
            args.n_workers,
        )
    else:
        summary, results = run_benchmark(
            args.num_tests,
            args.max_nodes,
            args.max_string_len,
            args.seed,
            args.output_dir,
        )

    if args.csv:
        save_csv(results, args.output_dir)
    if args.plot:
        generate_population_figure(results, args.output_dir)
        generate_single_case_figure(results, args.output_dir)
    if args.table:
        generate_table(results, args.output_dir)


if __name__ == "__main__":
    main()
