# ruff: noqa: N803, N806, E402
"""Publication figures: graph-space topology and empirical complexity.

Figure 1 -- Neighbourhood topology:
    For a base graph G₀, display its GED-1 neighbours (edge edits) and its
    Levenshtein-1 neighbours (single-character string edits) side by side,
    showing that the two notions of closeness largely agree.

Figure 2 -- Empirical complexity:
    Measure CPU time of greedy single-start, greedy-min, exhaustive
    canonical encoding, and exact GED on random connected graphs of
    increasing size.  X = n (node count), Y = time (log scale).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.gridspec import GridSpec

from benchmarks.plotting_styles import (
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
    get_figure_size,
    render_colored_string,
    save_figure,
)
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.canonical import canonical_string as compute_canonical
from isalgraph.core.graph_to_string import GraphToString
from isalgraph.core.string_to_graph import StringToGraph

logger = logging.getLogger(__name__)

_ADAPTER = NetworkXAdapter()
_ALPHABET = list("NnPpVvCcW")


# ============================================================================
# Shared helpers
# ============================================================================


def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance (pure-Python, fine for short strings)."""
    n, m = len(s1), len(s2)
    if n < m:
        return _levenshtein(s2, s1)
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(n):
        curr = [i + 1]
        for j in range(m):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (s1[i] != s2[j])))
        prev = curr
    return prev[m]


def _nx_to_sg(G: nx.Graph):  # type: ignore[type-arg]
    return _ADAPTER.from_external(G, directed=False)


def _greedy_min(G: nx.Graph) -> str:
    """Greedy-min encoding: shortest then lexmin across all starting nodes."""
    sg = _nx_to_sg(G)
    best: str | None = None
    for v in range(sg.node_count()):
        try:
            gts = GraphToString(sg)
            s, _ = gts.run(initial_node=v)
        except (ValueError, RuntimeError):
            continue
        if best is None or len(s) < len(best) or (len(s) == len(best) and s < best):
            best = s
    assert best is not None
    return best


def _decode(w: str) -> nx.Graph | None:
    """Decode an IsalGraph string to a NetworkX graph; None on failure."""
    try:
        s2g = StringToGraph(w, directed=False)
        sg, _ = s2g.run()
        G = _ADAPTER.to_external(sg)
        if G.number_of_nodes() == 0:
            return None
        return G
    except Exception:  # noqa: BLE001
        return None


def _ged(G1: nx.Graph, G2: nx.Graph) -> int:
    """Exact GED between two small unlabelled graphs."""
    return int(
        nx.graph_edit_distance(
            G1,
            G2,
            node_subst_cost=lambda _a, _b: 0,
            edge_subst_cost=lambda _a, _b: 0,
        )
    )


# ============================================================================
# Figure 1 — Neighbourhood topology
# ============================================================================


def _ged1_neighbors(
    G0: nx.Graph,
    w0: str,
) -> list[dict[str, Any]]:
    """All non-isomorphic GED-1 neighbours of G0 (edge add / remove)."""
    results: list[dict[str, Any]] = []
    seen: list[nx.Graph] = []

    def _try(G: nx.Graph, edit_type: str, edge: tuple[int, int]) -> None:
        for G_prev in seen:
            if nx.is_isomorphic(G, G_prev):
                return
        if nx.is_isomorphic(G, G0):
            return
        seen.append(G)
        w = _greedy_min(G)
        results.append(
            {
                "graph": G,
                "string": w,
                "lev": _levenshtein(w0, w),
                "edit_type": edit_type,
                "edge": edge,
            }
        )

    # Edge deletions
    for u, v in list(G0.edges()):
        G = G0.copy()
        G.remove_edge(u, v)
        if nx.is_connected(G):
            _try(G, "del", (u, v))

    # Edge additions
    for u, v in nx.non_edges(G0):
        G = G0.copy()
        G.add_edge(u, v)
        _try(G, "add", (u, v))

    results.sort(key=lambda d: (d["lev"], d["string"]))
    return results


def _lev1_neighbors(
    G0: nx.Graph,
    w0: str,
) -> list[dict[str, Any]]:
    """Non-isomorphic Levenshtein-1 neighbours of G0 in greedy-min space."""
    candidates: set[str] = set()

    # Substitutions
    for i in range(len(w0)):
        for c in _ALPHABET:
            if c != w0[i]:
                candidates.add(w0[:i] + c + w0[i + 1 :])
    # Deletions
    for i in range(len(w0)):
        candidates.add(w0[:i] + w0[i + 1 :])
    # Insertions
    for i in range(len(w0) + 1):
        for c in _ALPHABET:
            candidates.add(w0[:i] + c + w0[i:])

    seen: list[nx.Graph] = []
    results: list[dict[str, Any]] = []
    for w_cand in sorted(candidates):
        G = _decode(w_cand)
        if G is None or not nx.is_connected(G):
            continue
        dup = False
        for G_prev in seen:
            if nx.is_isomorphic(G, G_prev):
                dup = True
                break
        if dup or nx.is_isomorphic(G, G0):
            continue
        seen.append(G)
        w_gm = _greedy_min(G)
        ged_val = _ged(G0, G)
        results.append({"graph": G, "string": w_gm, "ged": ged_val, "orig": w_cand})

    results.sort(key=lambda d: (d["ged"], d["string"]))
    return results


def _draw_cell(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    w: str,
    *,
    title: str = "",
    ghost_edge: tuple[int, int] | None = None,
    highlight_edge: tuple[int, int] | None = None,
    node_size: int = 150,
) -> None:
    """Draw one graph cell with string annotation."""
    from benchmarks.eval_visualizations.graph_drawing import _DEFAULT_NODE_COLOR

    # Normal edges
    all_edges = list(G.edges())
    hl_set: set[tuple[int, int]] = set()
    if highlight_edge is not None:
        hl_set = {(min(highlight_edge), max(highlight_edge))}

    normal = [e for e in all_edges if (min(e), max(e)) not in hl_set]
    nx.draw_networkx_edges(G, pos, edgelist=normal, edge_color="0.5", width=0.8, ax=ax)

    # Highlighted (added) edge in red
    if highlight_edge is not None:
        hl = [e for e in all_edges if (min(e), max(e)) in hl_set]
        if hl:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=hl,
                edge_color=PAUL_TOL_BRIGHT["red"],
                width=1.5,
                ax=ax,
            )

    # Ghost (removed) edge as dashed red
    if ghost_edge is not None:
        u, v = ghost_edge
        if u in pos and v in pos:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                "--",
                color=PAUL_TOL_BRIGHT["red"],
                linewidth=1.0,
                zorder=0,
                alpha=0.6,
            )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=_DEFAULT_NODE_COLOR,
        node_size=node_size,
        edgecolors="0.3",
        linewidths=0.5,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=6, font_color="white", ax=ax)

    if title:
        ax.set_title(title, fontsize=7, pad=3)

    # Axis limits (same padding as draw_graph)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
    pad = 0.25 * span
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    y_lo = min(ys) - pad
    y_hi = max(ys) + pad
    # Extra bottom space for the string
    ax.set_ylim(y_lo - 0.35 * span, y_hi)

    # Render coloured string below graph
    x_center = np.mean(xs)
    render_colored_string(ax, w, x=x_center, y=min(ys) - 0.22 * span, fontsize=5, mono=True)

    ax.set_aspect("equal")
    ax.axis("off")


def generate_neighborhood_figure(output_dir: str) -> str:
    """Generate the GED-1 vs Levenshtein-1 neighbourhood figure."""
    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)

    # --- Base graph: house graph (5 nodes, 6 edges) -----------------------
    G0 = nx.house_graph()
    w0 = _greedy_min(G0)
    pos0 = nx.spring_layout(G0, seed=42)
    logger.info("Base graph: house, w0=%r (len=%d)", w0, len(w0))

    # --- GED-1 neighbours --------------------------------------------------
    ged1 = _ged1_neighbors(G0, w0)
    logger.info("GED-1 neighbours: %d unique", len(ged1))

    # --- Lev-1 neighbours --------------------------------------------------
    lev1 = _lev1_neighbors(G0, w0)
    logger.info("Lev-1 neighbours: %d unique", len(lev1))

    # Select 4 from each: prefer diversity in distance values
    n_show = 4
    sel_ged1 = ged1[:n_show]
    sel_lev1 = lev1[:n_show]

    # --- Figure layout: 2 rows × (1 + n_show) columns --------------------
    n_cols = 1 + n_show
    w_fig, _ = get_figure_size("double")
    fig = plt.figure(figsize=(w_fig, 4.0))
    gs = GridSpec(
        2,
        n_cols,
        figure=fig,
        wspace=0.15,
        hspace=0.45,
        left=0.02,
        right=0.98,
        top=0.88,
        bottom=0.04,
    )

    # Row (a): GED-1 neighbourhood
    ax0a = fig.add_subplot(gs[0, 0])
    _draw_cell(ax0a, G0, pos0, w0, title="$G_0$ (base)")

    for i, d in enumerate(sel_ged1):
        ax = fig.add_subplot(gs[0, i + 1])
        # Reuse base positions (same node set for edge edits)
        ghost = d["edge"] if d["edit_type"] == "del" else None
        hl = d["edge"] if d["edit_type"] == "add" else None
        _draw_cell(
            ax,
            d["graph"],
            pos0,
            d["string"],
            title=f"Lev = {d['lev']}",
            ghost_edge=ghost,
            highlight_edge=hl,
        )

    # Row (b): Lev-1 neighbourhood
    ax0b = fig.add_subplot(gs[1, 0])
    _draw_cell(ax0b, G0, pos0, w0, title="$G_0$ (base)")

    for i, d in enumerate(sel_lev1):
        ax = fig.add_subplot(gs[1, i + 1])
        pos_i = nx.spring_layout(d["graph"], seed=42)
        _draw_cell(ax, d["graph"], pos_i, d["string"], title=f"GED = {d['ged']}")

    # Row headers
    fig.text(
        0.01,
        0.92,
        r"$\mathbf{(a)}$ GED-1 neighbours of $G_0$"
        r"  $\longrightarrow$  Levenshtein distance to $w_0$",
        fontsize=8,
        va="bottom",
    )
    fig.text(
        0.01,
        0.46,
        r"$\mathbf{(b)}$ Levenshtein-1 neighbours of $G_0$"
        r"  $\longrightarrow$  GED to $G_0$",
        fontsize=8,
        va="bottom",
    )

    path = os.path.join(output_dir, "fig_neighborhood_topology")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Neighbourhood figure saved: %s", path)
    return path


# ============================================================================
# Figure 2 — Empirical complexity
# ============================================================================


def _random_connected(n: int, p: float, rng: np.random.Generator) -> nx.Graph:
    """Generate a random connected G(n,p) graph."""
    for _ in range(500):
        seed = int(rng.integers(0, 2**31))
        G = nx.gnp_random_graph(n, p, seed=seed)
        if nx.is_connected(G):
            return G
    # Fallback: complete graph
    return nx.complete_graph(n)


def _time_it(func, *args, n_reps: int = 3) -> float:
    """Return median CPU time (seconds) of func(*args) over n_reps calls."""
    times = []
    for _ in range(n_reps):
        t0 = time.process_time()
        func(*args)
        t1 = time.process_time()
        times.append(t1 - t0)
    return float(np.median(times))


def generate_complexity_figure(
    output_dir: str,
    *,
    seed: int = 42,
    n_graphs: int = 8,
    n_range: tuple[int, int] = (4, 13),
    p: float = 0.35,
    exhaustive_timeout: float = 60.0,
) -> str:
    """Generate the empirical complexity figure.

    Args:
        output_dir: Output directory.
        seed: Random seed.
        n_graphs: Graphs per node-count level.
        n_range: (min_n, max_n_exclusive) for graph sizes.
        p: Edge probability for G(n,p).
        exhaustive_timeout: Skip exhaustive if a single probe exceeds this.
    """
    apply_ieee_style()
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    sizes = list(range(n_range[0], n_range[1]))
    data: dict[str, dict[int, list[float]]] = {
        "greedy_single": {},
        "greedy_min": {},
        "exhaustive": {},
        "exact_ged": {},
    }
    for key in data:
        for n in sizes:
            data[key][n] = []

    for n in sizes:
        logger.info("Timing n=%d ...", n)
        graphs = [_random_connected(n, p, rng) for _ in range(n_graphs)]

        for G in graphs:
            sg = _nx_to_sg(G)

            # Greedy single-start (node 0)
            def _gs(sg_=sg):  # noqa: E731
                GraphToString(sg_).run(initial_node=0)

            data["greedy_single"][n].append(_time_it(_gs, n_reps=5))

            # Greedy-min (all starts)
            def _gm(G_=G):  # noqa: E731
                _greedy_min(G_)

            data["greedy_min"][n].append(_time_it(_gm, n_reps=3))

            # Exhaustive canonical (with timeout probe)
            t_probe = time.process_time()
            import contextlib  # noqa: PLC0415

            with contextlib.suppress(Exception):
                compute_canonical(sg)
            t_probe = time.process_time() - t_probe

            if t_probe < exhaustive_timeout:
                data["exhaustive"][n].append(
                    _time_it(
                        lambda sg_=sg: compute_canonical(sg_),
                        n_reps=max(1, min(3, int(5.0 / max(t_probe, 1e-6)))),
                    )
                )

        # Exact GED: time a random pair
        G1, G2 = graphs[0], graphs[min(1, len(graphs) - 1)]
        for _ in range(min(n_graphs, 5)):
            data["exact_ged"][n].append(_time_it(lambda g1=G1, g2=G2: _ged(g1, g2), n_reps=3))

    # --- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=get_figure_size("double", height_ratio=0.55))

    _COLORS = {
        "greedy_single": PAUL_TOL_BRIGHT["cyan"],
        "greedy_min": PAUL_TOL_BRIGHT["red"],
        "exhaustive": PAUL_TOL_BRIGHT["purple"],
        "exact_ged": PAUL_TOL_BRIGHT["grey"],
    }
    _MARKERS = {
        "greedy_single": "^",
        "greedy_min": "D",
        "exhaustive": "s",
        "exact_ged": "o",
    }
    _LABELS = {
        "greedy_single": "Greedy (single start)",
        "greedy_min": r"Greedy-min ($n$ starts)",
        "exhaustive": "Exhaustive canonical",
        "exact_ged": "Exact GED (one pair)",
    }

    for key in ["exact_ged", "exhaustive", "greedy_min", "greedy_single"]:
        xs, ys_med, ys_lo, ys_hi = [], [], [], []
        for n in sizes:
            vals = data[key][n]
            if not vals:
                continue
            med = np.median(vals)
            q25 = np.percentile(vals, 25)
            q75 = np.percentile(vals, 75)
            xs.append(n)
            ys_med.append(med)
            ys_lo.append(med - q25)
            ys_hi.append(q75 - med)
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys_med,
            yerr=[ys_lo, ys_hi],
            marker=_MARKERS[key],
            color=_COLORS[key],
            linewidth=1.2,
            markersize=5,
            capsize=2,
            capthick=0.8,
            label=_LABELS[key],
            zorder=3,
        )

    # Fit power-law reference lines for greedy methods
    for key, ls in [("greedy_single", ":"), ("greedy_min", "--")]:
        xs_fit, ys_fit = [], []
        for n in sizes:
            vals = data[key][n]
            if vals:
                xs_fit.append(n)
                ys_fit.append(np.median(vals))
        if len(xs_fit) >= 3:
            log_x = np.log(xs_fit)
            log_y = np.log(ys_fit)
            alpha, log_c = np.polyfit(log_x, log_y, 1)
            x_ref = np.linspace(sizes[0], sizes[-1], 50)
            y_ref = np.exp(log_c) * x_ref**alpha
            ax.plot(
                x_ref,
                y_ref,
                ls,
                color=_COLORS[key],
                alpha=0.5,
                linewidth=0.8,
                label=f"$\\alpha = {alpha:.1f}$  (fit)",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes ($n$)", fontsize=9)
    ax.set_ylabel("CPU time (s)", fontsize=9)
    ax.set_xticks(sizes)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, ncol=2)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.4, linestyle=":")

    path = os.path.join(output_dir, "fig_empirical_complexity")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("Complexity figure saved: %s", path)
    return path


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate topology and complexity publication figures.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures/topology_and_complexity",
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for complexity figure.",
    )
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=8,
        help="Graphs per node-count level.",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=4,
        help="Minimum node count.",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=13,
        help="Maximum node count (exclusive).",
    )
    parser.add_argument(
        "--edge-prob",
        type=float,
        default=0.35,
        help="Edge probability for G(n,p).",
    )
    parser.add_argument(
        "--exhaustive-timeout",
        type=float,
        default=60.0,
        help="Skip exhaustive canonical if probe exceeds this (seconds).",
    )
    parser.add_argument(
        "--only",
        choices=["neighborhood", "complexity"],
        default=None,
        help="Generate only one figure.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.only != "complexity":
        generate_neighborhood_figure(args.output_dir)

    if args.only != "neighborhood":
        generate_complexity_figure(
            args.output_dir,
            seed=args.seed,
            n_graphs=args.n_graphs,
            n_range=(args.n_min, args.n_max),
            p=args.edge_prob,
            exhaustive_timeout=args.exhaustive_timeout,
        )

    logger.info("All figures generated in %s", args.output_dir)


if __name__ == "__main__":
    main()
