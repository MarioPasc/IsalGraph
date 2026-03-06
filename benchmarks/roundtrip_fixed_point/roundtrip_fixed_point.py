"""Round-trip Fixed Point Property benchmark.

Tests whether the greedy G2S algorithm is idempotent at the string level.
Given w' = G2S(S2G(w), 0), does G2S(S2G(w'), 0) = w'?

If yes: G2S always produces stable/fixed-point strings from its own
perspective -- the greedy encoding is idempotent after one round-trip.

If no: the encoding oscillates between different string representations,
which would be a surprising finding worth reporting.

Key design decision: always use initial_node=0 because S2G creates
node 0 first, making it the natural starting node for re-encoding.

Hypotheses:
  H1: G2S is idempotent (w'' = w') with high probability.
  H2: When not a fixed point, Lev(w', w'') is small relative to |w'|.
  H3: Further iterations converge within 2-3 rounds.

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

from benchmarks.plotting_styles import (
    FAMILY_COLORS,
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
from isalgraph.core.canonical import levenshtein
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph"
DEFAULT_SEED = 42
MAX_ROUNDS = 5  # for multi-round convergence tests

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class FixedPointRecord:
    """Result for one round-trip fixed-point test."""

    test_id: int
    source: str  # "random_string" or graph family name
    family: str
    num_nodes: int
    num_edges: int
    density: float
    directed: bool
    # Round 1 & 2
    w_round1: str
    w_round2: str
    len_round1: int
    len_round2: int
    # Fixed point comparison
    is_fixed_point: bool
    lev_round1_round2: int
    length_change: int  # len_round2 - len_round1
    # Graph isomorphism verification
    graph_iso_verified: bool
    # Multi-round convergence (for subset)
    multi_round: bool
    convergence_round: int  # round at which w_k == w_{k+1}, -1 if > MAX_ROUNDS
    all_round_lengths: list[int]  # [|w_0|, |w_1|, ..., |w_k|]
    # Timing
    time_s: float
    error: str = ""


# =============================================================================
# Random string generation
# =============================================================================

_VALID_INSTRUCTIONS = list("NnPpVvCcW")
_NODE_INSTRS = {"V", "v"}


def _generate_random_string(
    rng: np.random.Generator,
    min_len: int = 3,
    max_len: int = 50,
    max_nodes: int = 50,
) -> str | None:
    """Generate a random valid IsalGraph instruction string.

    Returns None if the resulting graph doesn't meet criteria.
    """
    length = rng.integers(min_len, max_len + 1)
    chars = []
    for _ in range(length):
        if rng.random() < 0.4:
            chars.append(rng.choice(["V", "v"]))
        else:
            chars.append(rng.choice(_VALID_INSTRUCTIONS))
    w = "".join(chars)

    # Check node count
    n_nodes = 1 + sum(1 for c in w if c in _NODE_INSTRS)
    if n_nodes < 2 or n_nodes > max_nodes:
        return None
    return w


# =============================================================================
# Graph generation
# =============================================================================


def _generate_test_inputs(
    seed: int,
) -> list[tuple[str, str, str | None]]:
    """Generate test inputs: (source_label, family, initial_string_or_None).

    For random strings, initial_string is the raw string.
    For NX families, initial_string is None (graph generated later).
    Returns list of (source, family, w_or_none).
    """
    rng = np.random.default_rng(seed)
    inputs: list[tuple[str, str, str | None]] = []

    # Random strings: 400 tests
    count = 0
    while count < 400:
        w = _generate_random_string(rng, min_len=3, max_len=50, max_nodes=50)
        if w is not None:
            inputs.append(("random_string", "random_string", w))
            count += 1

    # NX families with explicit graph generation
    adapter = NetworkXAdapter()

    def _add_nx(family: str, nxg: nx.Graph) -> None:
        if nxg.number_of_nodes() >= 2 and nx.is_connected(nxg):
            sg = adapter.from_external(nxg, directed=False)
            w, _ = GraphToString(sg).run(initial_node=0)
            inputs.append(("nx_graph", family, w))

    # Trees
    for n in [3, 5, 8, 10, 15, 20, 30, 50]:
        for _ in range(4):
            _add_nx("tree", nx.random_labeled_tree(n, seed=int(rng.integers(0, 2**31))))

    # Cycles
    for n in [3, 5, 8, 10, 15, 20, 30, 50]:
        _add_nx("cycle", nx.cycle_graph(n))

    # Stars
    for n in [3, 5, 8, 10, 15, 20, 30, 50]:
        _add_nx("star", nx.star_graph(n - 1))

    # Complete
    for n in [3, 4, 5, 6, 7, 8, 10, 15]:
        _add_nx("complete", nx.complete_graph(n))

    # GNP p=0.3
    for n in [4, 6, 8, 10, 15, 20, 30, 50]:
        for _ in range(6):
            for _retry in range(20):
                g = nx.gnp_random_graph(n, 0.3, seed=int(rng.integers(0, 2**31)))
                if nx.is_connected(g) and g.number_of_edges() > 0:
                    _add_nx("gnp_p0.3", g)
                    break

    # BA m=2
    for n in [4, 6, 8, 10, 15, 20, 30, 50]:
        for _ in range(4):
            _add_nx("ba_m2", nx.barabasi_albert_graph(n, 2, seed=int(rng.integers(0, 2**31))))

    # WS k=4
    for n in [6, 10, 15, 20, 30, 50]:
        for _ in range(5):
            _add_nx(
                "watts_strogatz",
                nx.watts_strogatz_graph(n, 4, 0.3, seed=int(rng.integers(0, 2**31))),
            )

    # Grid
    for side in [2, 3, 4, 5, 6, 8, 10]:
        g = nx.grid_2d_graph(side, side)
        g_relabeled = nx.convert_node_labels_to_integers(g)
        _add_nx("grid", g_relabeled)

    logger.info("Generated %d test inputs across families", len(inputs))
    return inputs


# =============================================================================
# Core computation
# =============================================================================


def _run_single_test(
    test_id: int,
    source: str,
    family: str,
    w_input: str,
    do_multi_round: bool,
) -> FixedPointRecord:
    """Run the fixed-point test for a single string."""
    t0 = time.perf_counter()

    try:
        # Round 1: S2G(w_input) -> G1, then G2S(G1, 0) -> w_round1
        sg1, _ = StringToGraph(w_input, directed_graph=False).run()
        n = sg1.node_count()
        m = sg1.logical_edge_count()
        max_edges = n * (n - 1) / 2
        density = m / max_edges if max_edges > 0 else 0.0

        w_round1, _ = GraphToString(sg1).run(initial_node=0)

        # Round 2: S2G(w_round1) -> G2, then G2S(G2, 0) -> w_round2
        sg2, _ = StringToGraph(w_round1, directed_graph=False).run()
        w_round2, _ = GraphToString(sg2).run(initial_node=0)

        # Fixed-point check
        is_fp = w_round1 == w_round2
        lev_12 = levenshtein(w_round1, w_round2) if not is_fp else 0

        # Graph isomorphism verification (sanity check)
        iso_ok = sg1.is_isomorphic(sg2)

        # Multi-round convergence
        conv_round = 1 if is_fp else -1
        all_lengths = [len(w_input), len(w_round1), len(w_round2)]

        if do_multi_round and not is_fp:
            w_prev = w_round2
            for rnd in range(3, MAX_ROUNDS + 2):
                sg_tmp, _ = StringToGraph(w_prev, directed_graph=False).run()
                w_next, _ = GraphToString(sg_tmp).run(initial_node=0)
                all_lengths.append(len(w_next))
                if w_next == w_prev:
                    conv_round = rnd - 1  # rounds are 1-indexed
                    break
                w_prev = w_next
            else:
                conv_round = -1  # did not converge within MAX_ROUNDS

        elapsed = time.perf_counter() - t0

        return FixedPointRecord(
            test_id=test_id,
            source=source,
            family=family,
            num_nodes=n,
            num_edges=m,
            density=density,
            directed=False,
            w_round1=w_round1,
            w_round2=w_round2,
            len_round1=len(w_round1),
            len_round2=len(w_round2),
            is_fixed_point=is_fp,
            lev_round1_round2=lev_12,
            length_change=len(w_round2) - len(w_round1),
            graph_iso_verified=iso_ok,
            multi_round=do_multi_round,
            convergence_round=conv_round,
            all_round_lengths=all_lengths,
            time_s=elapsed,
        )
    except Exception as exc:
        return FixedPointRecord(
            test_id=test_id,
            source=source,
            family=family,
            num_nodes=0,
            num_edges=0,
            density=0.0,
            directed=False,
            w_round1="",
            w_round2="",
            len_round1=0,
            len_round2=0,
            is_fixed_point=False,
            lev_round1_round2=0,
            length_change=0,
            graph_iso_verified=False,
            multi_round=do_multi_round,
            convergence_round=-1,
            all_round_lengths=[],
            time_s=time.perf_counter() - t0,
            error=str(exc),
        )


# Parallel worker
def _parallel_worker(
    args: tuple[int, str, str, str, bool],
) -> FixedPointRecord:
    """Run single test in subprocess."""
    test_id, source, family, w_input, do_multi = args
    return _run_single_test(test_id, source, family, w_input, do_multi)


# =============================================================================
# Benchmark runner
# =============================================================================


def run_benchmark(
    seed: int = DEFAULT_SEED,
    n_workers: int = 1,
) -> list[FixedPointRecord]:
    """Run the full benchmark."""
    inputs = _generate_test_inputs(seed)
    results: list[FixedPointRecord] = []

    # First ~100 tests get multi-round analysis
    rng = np.random.default_rng(seed)
    multi_round_ids = set(
        rng.choice(len(inputs), size=min(100, len(inputs)), replace=False).tolist()
    )

    if n_workers <= 1:
        for i, (source, family, w) in enumerate(inputs):
            do_multi = i in multi_round_ids
            rec = _run_single_test(i, source, family, w, do_multi)
            results.append(rec)
            if (i + 1) % max(1, len(inputs) // 10) == 0:
                logger.info("  Progress: %d/%d tests", i + 1, len(inputs))
    else:
        tasks = [
            (i, source, family, w, i in multi_round_ids)
            for i, (source, family, w) in enumerate(inputs)
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
    results: list[FixedPointRecord],
    output_dir: str,
    seed: int,
) -> None:
    """Save JSON summary."""
    valid = [r for r in results if r.error == ""]
    families = sorted(set(r.family for r in valid))

    by_family: dict[str, Any] = {}
    for fam in families:
        recs = [r for r in valid if r.family == fam]
        n_fp = sum(1 for r in recs if r.is_fixed_point)
        by_family[fam] = {
            "count": len(recs),
            "n_fixed_point": n_fp,
            "p_fixed_point": n_fp / len(recs) if recs else 0.0,
            "mean_lev_nonfp": float(
                np.mean([r.lev_round1_round2 for r in recs if not r.is_fixed_point])
            )
            if any(not r.is_fixed_point for r in recs)
            else 0.0,
        }

    obj = {
        "benchmark": "roundtrip_fixed_point",
        "config": {"seed": seed, "max_rounds": MAX_ROUNDS},
        "summary": {
            "total_tests": len(results),
            "valid_tests": len(valid),
            "n_errors": sum(1 for r in results if r.error != ""),
            "n_fixed_point": sum(1 for r in valid if r.is_fixed_point),
            "p_fixed_point": (
                sum(1 for r in valid if r.is_fixed_point) / len(valid) if valid else 0.0
            ),
            "n_iso_failures": sum(1 for r in valid if not r.graph_iso_verified),
        },
        "results_by_family": by_family,
    }

    path = os.path.join(output_dir, "roundtrip_fixed_point_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Saved JSON: %s", path)


def save_csv(
    results: list[FixedPointRecord],
    output_dir: str,
) -> None:
    """Save per-test CSV."""
    rows = []
    for r in results:
        rows.append(
            {
                "test_id": r.test_id,
                "source": r.source,
                "family": r.family,
                "num_nodes": r.num_nodes,
                "num_edges": r.num_edges,
                "density": round(r.density, 4),
                "len_round1": r.len_round1,
                "len_round2": r.len_round2,
                "is_fixed_point": r.is_fixed_point,
                "lev_round1_round2": r.lev_round1_round2,
                "length_change": r.length_change,
                "graph_iso_verified": r.graph_iso_verified,
                "multi_round": r.multi_round,
                "convergence_round": r.convergence_round,
                "time_s": round(r.time_s, 4),
                "error": r.error,
            }
        )
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "roundtrip_fixed_point_results.csv")
    df.to_csv(path, index=False)
    logger.info("Saved CSV: %s", path)


# =============================================================================
# Population figure (2x2)
# =============================================================================


def generate_population_figure(
    results: list[FixedPointRecord],
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

    # ---- Panel (a): Fixed-point rate by family ----
    ax = axes[0, 0]
    fam_rates: list[tuple[str, float, float, float]] = []
    for fam in families:
        recs = [r for r in valid if r.family == fam]
        n_fp = sum(1 for r in recs if r.is_fixed_point)
        rate = n_fp / len(recs) if recs else 0.0
        ci_lo, ci_hi = binomial_ci(n_fp, len(recs))
        fam_rates.append((fam, rate, ci_lo, ci_hi))

    fam_rates.sort(key=lambda x: x[1], reverse=True)
    y_pos = list(range(len(fam_rates)))
    rates = [r for _, r, _, _ in fam_rates]
    errors_lo = [r - lo for _, r, lo, _ in fam_rates]
    errors_hi = [hi - r for _, r, _, hi in fam_rates]
    colors = [FAMILY_COLORS.get(f, "#BBBBBB") for f, _, _, _ in fam_rates]

    ax.barh(
        y_pos,
        rates,
        xerr=[errors_lo, errors_hi],
        color=colors,
        alpha=0.8,
        edgecolor="0.3",
        linewidth=0.5,
        capsize=PLOT_SETTINGS["errorbar_capsize"],
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [family_display(f) for f, _, _, _ in fam_rates], fontsize=PLOT_SETTINGS["tick_labelsize"]
    )
    overall_rate = sum(1 for r in valid if r.is_fixed_point) / len(valid)
    ax.axvline(overall_rate, color="0.4", ls="--", lw=0.8, label=f"overall={overall_rate:.2f}")
    ax.set_xlabel("P(fixed point)")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], frameon=False)
    ax.set_title("(a) Fixed-point rate by family", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # ---- Panel (b): Convergence round distribution ----
    ax = axes[0, 1]
    multi = [r for r in valid if r.multi_round]
    if multi:
        round_counts: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, ">5": 0}
        for r in multi:
            if r.is_fixed_point:
                round_counts["1"] += 1
            elif r.convergence_round > 0 and r.convergence_round <= 5:
                round_counts[str(r.convergence_round)] += 1
            else:
                round_counts[">5"] += 1

        labels = list(round_counts.keys())
        counts = [round_counts[l] for l in labels]
        ax.bar(
            labels,
            counts,
            color=PAUL_TOL_BRIGHT["blue"],
            alpha=0.8,
            edgecolor="0.3",
            linewidth=0.5,
        )
        ax.set_xlabel("Convergence round")
        ax.set_ylabel("Count")
        for i, c in enumerate(counts):
            if c > 0:
                ax.text(
                    i, c + 0.5, str(c), ha="center", fontsize=PLOT_SETTINGS["annotation_fontsize"]
                )
    ax.set_title("(b) Convergence round distribution", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # ---- Panel (c): Normalized Lev distance for non-FP cases ----
    ax = axes[1, 0]
    non_fp = [r for r in valid if not r.is_fixed_point and r.len_round1 > 0]
    if non_fp:
        for fam in families:
            recs = [r for r in non_fp if r.family == fam]
            if not recs:
                continue
            ns = [r.num_nodes for r in recs]
            norm_levs = [r.lev_round1_round2 / r.len_round1 for r in recs]
            ax.scatter(
                ns,
                norm_levs,
                c=FAMILY_COLORS.get(fam, "#BBBBBB"),
                s=PLOT_SETTINGS["scatter_size"] * 2,
                alpha=PLOT_SETTINGS["scatter_alpha"],
                label=family_display(fam),
                edgecolors="0.3",
                linewidths=PLOT_SETTINGS["scatter_edgewidth"],
            )
        ax.set_xlabel("Number of nodes $N$")
        ax.set_ylabel(r"Lev$(w', w'') / |w'|$")
        ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper right")
    else:
        ax.text(
            0.5,
            0.5,
            "All tests are\nfixed points!",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    ax.set_title("(c) Deviation when not fixed point", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # ---- Panel (d): Length trajectory for multi-round tests ----
    ax = axes[1, 1]
    if multi:
        for r in multi:
            if len(r.all_round_lengths) >= 2:
                rounds = list(range(len(r.all_round_lengths)))
                color = FAMILY_COLORS.get(r.family, "#BBBBBB")
                ax.plot(
                    rounds,
                    r.all_round_lengths,
                    color=color,
                    alpha=0.15,
                    lw=0.5,
                )

        # Mean trajectory per family
        for fam in families:
            fam_multi = [r for r in multi if r.family == fam and len(r.all_round_lengths) >= 3]
            if not fam_multi:
                continue
            max_len_traj = max(len(r.all_round_lengths) for r in fam_multi)
            # Pad shorter trajectories with their last value
            padded = []
            for r in fam_multi:
                traj = r.all_round_lengths[:]
                while len(traj) < max_len_traj:
                    traj.append(traj[-1])
                padded.append(traj)
            mean_traj = np.mean(padded, axis=0)
            ax.plot(
                range(len(mean_traj)),
                mean_traj,
                color=FAMILY_COLORS.get(fam, "#BBBBBB"),
                lw=PLOT_SETTINGS["line_width_thick"],
                label=family_display(fam),
            )
        ax.set_xlabel("Round $k$")
        ax.set_ylabel(r"$|w_k|$")
        ax.legend(fontsize=5, ncol=2, frameon=False, loc="upper right")
    ax.set_title("(d) Length trajectory over rounds", fontsize=PLOT_SETTINGS["axes_titlesize"])

    fig.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "roundtrip_fixed_point_population")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Saved population figure: %s.pdf", path)


# =============================================================================
# Individual figure (1x4)
# =============================================================================


def _select_showcase(
    results: list[FixedPointRecord],
) -> FixedPointRecord | None:
    """Select a non-fixed-point case with convergence at round 2-3."""
    candidates = [
        r
        for r in results
        if r.error == ""
        and not r.is_fixed_point
        and r.multi_round
        and 2 <= r.convergence_round <= 3
        and 5 <= r.num_nodes <= 10
    ]
    if not candidates:
        candidates = [
            r for r in results if r.error == "" and not r.is_fixed_point and r.num_nodes >= 3
        ]
    if not candidates:
        # All are fixed points -- pick one with interesting structure
        candidates = [r for r in results if r.error == "" and r.num_nodes >= 5]
    if not candidates:
        return None

    def score(r: FixedPointRecord) -> float:
        return r.lev_round1_round2 * 2 + r.num_nodes * 0.3 + len(r.all_round_lengths) * 0.5

    return max(candidates, key=score)


def _make_instruction_cmap() -> tuple[Any, dict[str, int]]:
    """Create a ListedColormap for instruction heatmaps."""
    from matplotlib.colors import ListedColormap

    instr_order = ["N", "P", "n", "p", "V", "v", "C", "c", "W", "_"]
    colors = [INSTRUCTION_COLORS.get(i, "#FFFFFF") for i in instr_order]
    cmap = ListedColormap(colors)
    char_to_idx = {ch: i for i, ch in enumerate(instr_order)}
    return cmap, char_to_idx


def generate_single_case_figure(
    results: list[FixedPointRecord],
    output_dir: str,
) -> None:
    """Generate detailed visualization of one representative case."""
    apply_ieee_style()
    rec = _select_showcase(results)
    if rec is None:
        logger.warning("No suitable case for individual figure.")
        return

    n_panels = 4 if rec.multi_round and len(rec.all_round_lengths) >= 3 else 3
    fig, axes = plt.subplots(1, n_panels, figsize=get_figure_size("double", 0.45))

    # ---- Panel (a): Graph G ----
    ax = axes[0]
    sg1, _ = StringToGraph(rec.w_round1, directed_graph=False).run()
    adapter = NetworkXAdapter()
    nxg = adapter.to_external(sg1)
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

    # ---- Panel (b): Instruction heatmap: w' vs w'' ----
    ax = axes[1]
    cmap, char_to_idx = _make_instruction_cmap()
    w1 = rec.w_round1
    w2 = rec.w_round2
    max_len = max(len(w1), len(w2))

    def _to_col(w: str) -> list[int]:
        col = [char_to_idx.get(ch, char_to_idx["_"]) for ch in w]
        col.extend([char_to_idx["_"]] * (max_len - len(col)))
        return col

    mat = np.array([_to_col(w1), _to_col(w2)]).T
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=len(cmap.colors) - 1)
    ax.set_xticks([0, 1])
    fp_label = "FP" if rec.is_fixed_point else f"Lev={rec.lev_round1_round2}"
    ax.set_xticklabels(
        [f"w' (|{len(w1)}|)", f"w'' (|{len(w2)}|)"],
        fontsize=PLOT_SETTINGS["tick_labelsize"],
    )
    ax.set_ylabel("Position")
    ax.set_title(f"(b) Round 1 vs 2 [{fp_label}]", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # ---- Panel (c): Diff visualization ----
    ax = axes[2]
    # Show character-level diff
    min_len_diff = min(len(w1), len(w2))
    max_len_diff = max(len(w1), len(w2))
    diff_colors = []
    for i in range(max_len_diff):
        c1 = w1[i] if i < len(w1) else "_"
        c2 = w2[i] if i < len(w2) else "_"
        if c1 == c2:
            diff_colors.append(0)  # match
        else:
            diff_colors.append(1)  # mismatch

    ax.barh(
        range(max_len_diff),
        [1] * max_len_diff,
        color=[PAUL_TOL_BRIGHT["green"] if d == 0 else PAUL_TOL_BRIGHT["red"] for d in diff_colors],
        alpha=0.7,
        edgecolor="none",
    )
    ax.set_ylim(max_len_diff - 0.5, -0.5)
    ax.set_xlabel("Match / Mismatch")
    ax.set_ylabel("Position")
    n_diff = sum(diff_colors)
    ax.set_title(
        f"(c) Character diff\n{n_diff}/{max_len_diff} differ",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )
    ax.set_xticks([])

    # ---- Panel (d): Length trajectory (if multi-round) ----
    if n_panels == 4:
        ax = axes[3]
        rounds = list(range(len(rec.all_round_lengths)))
        ax.plot(
            rounds,
            rec.all_round_lengths,
            "o-",
            color=PAUL_TOL_BRIGHT["blue"],
            lw=PLOT_SETTINGS["line_width_thick"],
            markersize=PLOT_SETTINGS["marker_size"],
        )
        ax.set_xlabel("Round $k$")
        ax.set_ylabel(r"$|w_k|$")
        if rec.convergence_round > 0:
            ax.axvline(
                rec.convergence_round,
                color="0.4",
                ls="--",
                lw=0.8,
                label=f"conv @{rec.convergence_round}",
            )
            ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], frameon=False)
        ax.set_title("(d) Length over rounds", fontsize=PLOT_SETTINGS["axes_titlesize"])

    fig.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "roundtrip_fixed_point_single_case")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Saved single-case figure: %s.pdf", path)


# =============================================================================
# LaTeX table
# =============================================================================


def generate_table(
    results: list[FixedPointRecord],
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
        n_fp = sum(1 for r in recs if r.is_fixed_point)
        ci_lo, ci_hi = binomial_ci(n_fp, len(recs))
        non_fp = [r for r in recs if not r.is_fixed_point]
        mean_lev = float(np.mean([r.lev_round1_round2 for r in non_fp])) if non_fp else 0.0
        multi_recs = [r for r in recs if r.multi_round and r.convergence_round > 0]
        median_conv = (
            float(np.median([r.convergence_round for r in multi_recs])) if multi_recs else 1.0
        )
        rows.append(
            {
                "Family": family_display(fam),
                "Tests": len(recs),
                "P(FP)": f"{n_fp / len(recs):.3f}",
                "95\\% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                "Mean Lev": f"{mean_lev:.1f}" if non_fp else "---",
                "Med conv": f"{median_conv:.0f}",
                "Mean |w'|": f"{np.mean([r.len_round1 for r in recs]):.1f}",
                "Mean time (ms)": f"{1000 * np.mean([r.time_s for r in recs]):.1f}",
            }
        )

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "roundtrip_fixed_point_table.tex")
    save_latex_table(
        df,
        path,
        caption="Round-trip fixed-point property by graph family. "
        "P(FP) = fraction where $w'' = w'$; Mean Lev = mean Levenshtein distance for non-fixed-point cases; "
        "Med conv = median convergence round for multi-round tests.",
        label="tab:fixed_point",
    )
    logger.info("Saved table: %s", path)


# =============================================================================
# Main / CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-trip Fixed Point Property benchmark",
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

    logger.info("=== Round-trip Fixed Point Property ===")
    logger.info("Seed=%d, workers=%d, output=%s", args.seed, args.n_workers, args.output_dir)

    t0 = time.perf_counter()
    results = run_benchmark(seed=args.seed, n_workers=args.n_workers)
    elapsed = time.perf_counter() - t0

    valid = [r for r in results if r.error == ""]
    n_fp = sum(1 for r in valid if r.is_fixed_point)
    n_errors = sum(1 for r in results if r.error != "")
    n_iso_fail = sum(1 for r in valid if not r.graph_iso_verified)
    logger.info(
        "Done: %d tests (%d valid, %d errors) in %.1fs", len(results), len(valid), n_errors, elapsed
    )
    if valid:
        logger.info("  Fixed points: %d/%d (%.1f%%)", n_fp, len(valid), 100 * n_fp / len(valid))
        logger.info("  Isomorphism failures: %d", n_iso_fail)
        non_fp = [r for r in valid if not r.is_fixed_point]
        if non_fp:
            logger.info(
                "  Non-FP: mean Lev=%0.1f, mean |len change|=%.1f",
                np.mean([r.lev_round1_round2 for r in non_fp]),
                np.mean([abs(r.length_change) for r in non_fp]),
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
