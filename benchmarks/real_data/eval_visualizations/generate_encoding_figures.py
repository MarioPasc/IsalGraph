# ruff: noqa: N803, N806, E402
"""Generate encoding complexity figures (C4.1, C4.2, C4.3).

Reads data from eval_encoding outputs and produces:
  - C4.1: Scaling exponents (log-log + bar + validation) + individual + table
  - C4.2: Density dependence heatmap + individual + table
  - C4.3: Greedy vs canonical overhead bar + individual + table

Usage:
    python -m benchmarks.eval_visualizations.generate_encoding_figures \
        --enc-dir results/eval_benchmarks/eval_encoding \
        --output-dir results/figures/encoding
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import stats as sp_stats

from benchmarks.eval_encoding.synthetic_generator import generate_graph_family
from benchmarks.eval_visualizations.table_generator import generate_dual_table
from benchmarks.plotting_styles import (
    FAMILY_COLORS,
    FAMILY_MARKERS,
    PAUL_TOL_BRIGHT,
    PAUL_TOL_MUTED,
    PLOT_SETTINGS,
    apply_ieee_style,
    family_display,
    get_figure_size,
    save_figure,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_ENC_DIR = (
    "/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_benchmarks/eval_encoding"
)
DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/results/figures/encoding"

ALL_FAMILIES = [
    "path",
    "star",
    "cycle",
    "complete",
    "binary_tree",
    "ba_m1",
    "ba_m2",
    "gnp_03",
    "gnp_05",
    "grid",
]

# Colors
GREEDY_COLOR = PAUL_TOL_BRIGHT["blue"]
CANONICAL_COLOR = PAUL_TOL_BRIGHT["red"]
VALIDATION_COLOR = PAUL_TOL_BRIGHT["green"]


# =============================================================================
# Data loading
# =============================================================================


def _load_scaling_exponents(enc_dir: str) -> dict:
    path = os.path.join(enc_dir, "stats", "scaling_exponents.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_density_analysis(enc_dir: str) -> dict:
    path = os.path.join(enc_dir, "stats", "density_analysis.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_greedy_csv(enc_dir: str) -> pd.DataFrame:
    path = os.path.join(enc_dir, "raw", "synthetic_greedy_times.csv")
    return pd.read_csv(path)


def _load_canonical_csv(enc_dir: str) -> pd.DataFrame:
    path = os.path.join(enc_dir, "raw", "synthetic_canonical_times.csv")
    return pd.read_csv(path)


def _load_density_csv(enc_dir: str) -> pd.DataFrame:
    path = os.path.join(enc_dir, "raw", "density_dependence.csv")
    return pd.read_csv(path)


def _load_validation_csv(enc_dir: str) -> pd.DataFrame:
    path = os.path.join(enc_dir, "raw", "real_validation.csv")
    return pd.read_csv(path)


# =============================================================================
# C4.1 — Scaling Exponents per Graph Family
# =============================================================================


def generate_c41_population(
    scaling: dict,
    greedy_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """C4.1 Population: 2x2 panel — greedy log-log, canonical log-log, bar, validation."""
    fig_dir = os.path.join(output_dir, "C4_1_scaling_exponents")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", height_ratio=0.85))

    # --- (a) Greedy single-start log-log ---
    ax_a = axes[0, 0]
    # Use best (minimum) greedy time per (family, n_nodes)
    greedy_best = greedy_df.groupby(["family", "n_nodes"])["greedy_time_s"].median().reset_index()

    for fam in ALL_FAMILIES:
        fam_data = greedy_best[greedy_best["family"] == fam]
        if fam_data.empty:
            continue
        color = FAMILY_COLORS.get(fam, "#333333")
        marker = FAMILY_MARKERS.get(fam, "o")
        ax_a.plot(
            fam_data["n_nodes"],
            fam_data["greedy_time_s"],
            color=color,
            marker=marker,
            markersize=3,
            linewidth=0.8,
            label=family_display(fam),
            alpha=0.8,
        )

    # Reference lines
    n_ref = np.linspace(3, 50, 100)
    for exp_val, label, ls in [
        (2, r"$O(N^2)$", ":"),
        (3, r"$O(N^3)$", "--"),
        (4, r"$O(N^4)$", "-."),
    ]:
        t_ref = 1e-6 * n_ref**exp_val
        ax_a.plot(n_ref, t_ref, color="0.6", linestyle=ls, linewidth=0.5, alpha=0.4, label=label)

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("N (nodes)")
    ax_a.set_ylabel("Time (s)")
    ax_a.set_title("(a) Greedy single-start", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax_a.legend(fontsize=5, ncol=2, loc="upper left", handletextpad=0.3, columnspacing=0.5)

    # --- (b) Canonical log-log ---
    ax_b = axes[0, 1]
    canon_best = (
        canonical_df.groupby(["family", "n_nodes"])["canonical_time_s"].median().reset_index()
    )

    for fam in ALL_FAMILIES:
        fam_data = canon_best[canon_best["family"] == fam]
        if fam_data.empty:
            continue
        color = FAMILY_COLORS.get(fam, "#333333")
        marker = FAMILY_MARKERS.get(fam, "o")
        # Determine if this family has exponential scaling
        fam_info = scaling.get("canonical", {}).get(fam, {})
        ls = "--" if fam_info.get("model") == "exponential" else "-"
        ax_b.plot(
            fam_data["n_nodes"],
            fam_data["canonical_time_s"],
            color=color,
            marker=marker,
            markersize=3,
            linewidth=0.8,
            linestyle=ls,
            label=family_display(fam),
            alpha=0.8,
        )

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("N (nodes)")
    ax_b.set_ylabel("Time (s)")
    ax_b.set_title("(b) Canonical", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax_b.legend(fontsize=5, ncol=2, loc="upper left", handletextpad=0.3, columnspacing=0.5)

    # --- (c) Bar chart: alpha per family ---
    ax_c = axes[1, 0]
    families_with_data = [f for f in ALL_FAMILIES if f in scaling.get("greedy_single", {})]
    x = np.arange(len(families_with_data))
    bar_w = 0.35

    greedy_alphas = [scaling["greedy_single"][f]["alpha"] for f in families_with_data]
    canon_alphas = [scaling["canonical"].get(f, {}).get("alpha", 0) for f in families_with_data]
    greedy_r2 = [scaling["greedy_single"][f]["r_squared"] for f in families_with_data]
    canon_r2 = [scaling["canonical"].get(f, {}).get("r_squared", 0) for f in families_with_data]

    ax_c.bar(x - bar_w / 2, greedy_alphas, bar_w, color=GREEDY_COLOR, alpha=0.85, label="Greedy")
    ax_c.bar(
        x + bar_w / 2, canon_alphas, bar_w, color=CANONICAL_COLOR, alpha=0.85, label="Canonical"
    )

    # R² annotations on greedy bars
    for i, (ga, gr) in enumerate(zip(greedy_alphas, greedy_r2, strict=True)):
        ax_c.text(i - bar_w / 2, ga + 0.15, f"{gr:.2f}", ha="center", fontsize=5, color="0.3")
    for i, (ca, cr) in enumerate(zip(canon_alphas, canon_r2, strict=True)):
        if ca > 0:
            ax_c.text(i + bar_w / 2, ca + 0.15, f"{cr:.2f}", ha="center", fontsize=5, color="0.3")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        [family_display(f) for f in families_with_data], rotation=45, ha="right", fontsize=6
    )
    ax_c.set_ylabel(r"Exponent $\alpha$")
    ax_c.set_title(r"(c) Scaling exponent $\alpha$", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax_c.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="upper left")
    ax_c.axhline(y=3, color="0.5", linestyle=":", linewidth=0.5, alpha=0.5)

    # --- (d) Scatter: observed vs predicted (validation) ---
    ax_d = axes[1, 1]
    valid = validation_df.dropna(subset=["observed_time_s", "predicted_time_s"])
    valid = valid[(valid["observed_time_s"] > 0) & (valid["predicted_time_s"] > 0)]

    ax_d.scatter(
        valid["predicted_time_s"],
        valid["observed_time_s"],
        c=VALIDATION_COLOR,
        alpha=0.3,
        s=8,
        edgecolors="none",
    )

    # Identity line
    lims = [
        min(valid["predicted_time_s"].min(), valid["observed_time_s"].min()),
        max(valid["predicted_time_s"].max(), valid["observed_time_s"].max()),
    ]
    ax_d.plot(lims, lims, color="0.3", linestyle="--", linewidth=0.8, label="Identity")

    # Fit and annotate R²
    log_pred = np.log10(valid["predicted_time_s"])
    log_obs = np.log10(valid["observed_time_s"])
    mask = np.isfinite(log_pred) & np.isfinite(log_obs)
    if mask.sum() > 2:
        slope, intercept, r_value, _p_value, _se = sp_stats.linregress(
            log_pred[mask], log_obs[mask]
        )
        ax_d.text(
            0.05,
            0.92,
            f"$R^2 = {r_value**2:.3f}$",
            transform=ax_d.transAxes,
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
        )

    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Predicted time (s)")
    ax_d.set_ylabel("Observed time (s)")
    ax_d.set_title("(d) Synthetic → Real validation", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax_d.legend(fontsize=7, loc="lower right")

    fig.suptitle("C4.1: Encoding Scaling Exponents", fontsize=PLOT_SETTINGS["axes_titlesize"] + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, os.path.join(fig_dir, "population_scaling_loglog"))
    plt.close(fig)
    logger.info("C4.1 population figure saved.")


def generate_c41_individual(output_dir: str) -> None:
    """C4.1 Individual: Path P8, BA(8,m=2), K8 — graph + string + time."""
    fig_dir = os.path.join(output_dir, "C4_1_scaling_exponents")
    os.makedirs(fig_dir, exist_ok=True)

    import networkx as nx

    from isalgraph.adapters.networkx_adapter import NetworkXAdapter
    from isalgraph.core.canonical import canonical_string
    from isalgraph.core.graph_to_string import GraphToString

    families = [("path", 8), ("ba_m2", 8), ("complete", 8)]
    labels = [r"Path $P_8$", r"BA$(8, m\!=\!2)$", r"$K_8$"]

    fig, axes = plt.subplots(1, 3, figsize=get_figure_size("double", height_ratio=0.4))

    for idx, ((fam, n), label) in enumerate(zip(families, labels, strict=True)):
        ax = axes[idx]
        G_nx = generate_graph_family(fam, n, seed=42, instance=0)
        sg = NetworkXAdapter().from_external(G_nx, directed=False)

        # Canonical string + time
        import time as _time

        t0 = _time.process_time()
        canon_str = canonical_string(sg)
        t_canon = _time.process_time() - t0

        # Greedy string + time
        t0 = _time.process_time()
        greedy_str, _cost = GraphToString(sg).run(initial_node=0)
        t_greedy = _time.process_time() - t0

        # Draw graph
        pos = nx.spring_layout(G_nx, seed=42)
        nx.draw_networkx(
            G_nx,
            pos,
            ax=ax,
            node_size=80,
            font_size=6,
            node_color=PAUL_TOL_BRIGHT["blue"],
            edge_color="0.6",
            width=0.8,
            with_labels=True,
            font_color="white",
        )
        ax.set_title(label, fontsize=PLOT_SETTINGS["axes_titlesize"])

        # Annotate
        info_text = (
            f'Greedy: "{greedy_str}" ({t_greedy * 1000:.2f} ms)\n'
            f'Canon.: "{canon_str}" ({t_canon * 1000:.1f} ms)'
        )
        ax.text(
            0.5,
            -0.05,
            info_text,
            transform=ax.transAxes,
            fontsize=6,
            ha="center",
            va="top",
            family="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
        )
        ax.axis("off")

    fig.suptitle("C4.1: Family Examples (N=8, seed=42)", fontsize=PLOT_SETTINGS["axes_titlesize"])
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_figure(fig, os.path.join(fig_dir, "individual_family_examples"))
    plt.close(fig)
    logger.info("C4.1 individual figure saved.")


def generate_c41_table(scaling: dict, output_dir: str) -> None:
    """C4.1 Table: Family, Density, alpha_greedy, R², alpha_canonical, R², Model."""
    fig_dir = os.path.join(output_dir, "C4_1_scaling_exponents")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    for fam in ALL_FAMILIES:
        gr = scaling["greedy_single"].get(fam, {})
        ca = scaling["canonical"].get(fam, {})

        rows.append(
            {
                "Family": family_display(fam),
                r"$\alpha_{\text{greedy}}$": f"{gr.get('alpha', 0):.2f}",
                r"$R^2_{\text{gr}}$": f"{gr.get('r_squared', 0):.3f}",
                r"$\alpha_{\text{canon}}$": f"{ca.get('alpha', 0):.2f}"
                if ca.get("alpha")
                else "---",
                r"$R^2_{\text{ca}}$": f"{ca.get('r_squared', 0):.3f}"
                if ca.get("r_squared")
                else "---",
                "Model": ca.get("model", "---"),
            }
        )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        fig_dir,
        "table_scaling_exponents",
        caption="Encoding scaling exponents per graph family.",
        label="tab:scaling_exponents",
        highlight_cols={r"$R^2_{\text{gr}}$", r"$R^2_{\text{ca}}$"},
    )
    logger.info("C4.1 table saved.")


# =============================================================================
# C4.2 — Density Dependence
# =============================================================================


def generate_c42_population(
    density_analysis: dict,
    density_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """C4.2 Population: Heatmap of log10(canonical time) vs (N, density p)."""
    fig_dir = os.path.join(output_dir, "C4_2_density_dependence")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.9))

    # Build pivot: rows=n_nodes, cols=p, values=median canonical time
    pivot_data = density_df.groupby(["n_nodes", "p"])["canonical_time_s"].median().reset_index()
    pivot = pivot_data.pivot(index="n_nodes", columns="p", values="canonical_time_s")

    # Replace inf/nan with a large sentinel for visualization
    max_finite = pivot.replace([np.inf, -np.inf], np.nan).max().max()
    timeout_val = max(max_finite * 10, 300) if np.isfinite(max_finite) else 300
    pivot = pivot.replace([np.inf], timeout_val)
    pivot = pivot.fillna(timeout_val)

    # Ensure values are positive for log scale
    pivot = pivot.clip(lower=1e-6)

    n_vals = pivot.index.values
    p_vals = pivot.columns.values

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="cividis",
        norm=LogNorm(vmin=pivot.values[pivot.values > 0].min(), vmax=pivot.values.max()),
        origin="lower",
        extent=[p_vals[0] - 0.05, p_vals[-1] + 0.05, n_vals[0] - 0.5, n_vals[-1] + 0.5],
    )

    cbar = fig.colorbar(im, ax=ax, label="Canonical time (s)")
    cbar.ax.tick_params(labelsize=7)

    # Contour lines at 1s, 10s, 60s, 300s
    contour_levels = [1, 10, 60, 300]
    valid_levels = [lv for lv in contour_levels if lv <= pivot.values.max()]
    if valid_levels:
        cs = ax.contour(
            p_vals,
            n_vals,
            pivot.values,
            levels=valid_levels,
            colors="white",
            linewidths=0.8,
            linestyles="--",
        )
        ax.clabel(cs, fmt=lambda x: f"{x:.0f}s", fontsize=6, colors="white")

    # Annotate timeout cells
    for i, n_val in enumerate(n_vals):
        for j, p_val in enumerate(p_vals):
            val = pivot.values[i, j]
            if val >= timeout_val * 0.9:
                ax.text(
                    p_val,
                    n_val,
                    "TO",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )

    ax.set_xlabel("Edge probability (p)")
    ax.set_ylabel("Number of nodes (N)")
    ax.set_title("C4.2: Canonical Encoding Feasibility", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax.set_xticks(p_vals)
    ax.set_yticks(n_vals)

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "population_density_heatmap"))
    plt.close(fig)
    logger.info("C4.2 population figure saved.")


def generate_c42_individual(
    density_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """C4.2 Individual: G(10,0.2) vs G(10,0.8) with times annotated."""
    fig_dir = os.path.join(output_dir, "C4_2_density_dependence")
    os.makedirs(fig_dir, exist_ok=True)

    import networkx as nx

    fig, axes = plt.subplots(1, 2, figsize=get_figure_size("single", height_ratio=0.6))

    for p_val, ax in zip([0.2, 0.8], axes, strict=True):
        # Generate a representative graph
        n = 10
        G = nx.erdos_renyi_graph(n, p_val, seed=42)
        # Ensure connected
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            G = nx.convert_node_labels_to_integers(G)

        # Get timing from CSV
        subset = density_df[(density_df["p"] == p_val) & (density_df["n_nodes"] <= n)]
        if not subset.empty:
            med_greedy = subset["greedy_min_time_s"].median()
            med_canon = subset["canonical_time_s"].median()
        else:
            med_greedy = float("nan")
            med_canon = float("nan")

        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G,
            pos,
            ax=ax,
            node_size=60,
            font_size=5,
            node_color=PAUL_TOL_BRIGHT["blue"],
            edge_color="0.6",
            width=0.6,
            with_labels=True,
            font_color="white",
        )

        time_str = f"Greedy: {med_greedy * 1000:.1f} ms"
        if np.isfinite(med_canon):
            time_str += f"\nCanon.: {med_canon * 1000:.1f} ms"
        else:
            time_str += "\nCanon.: timeout"

        ax.set_title(
            f"G({G.number_of_nodes()}, {p_val})\n{G.number_of_edges()} edges",
            fontsize=PLOT_SETTINGS["tick_labelsize"],
        )
        ax.text(
            0.5,
            -0.02,
            time_str,
            transform=ax.transAxes,
            fontsize=7,
            ha="center",
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
        )
        ax.axis("off")

    # Density increase annotation
    sub_02 = density_df[density_df["p"] == 0.2]["greedy_min_time_s"].median()
    sub_08 = density_df[density_df["p"] == 0.8]["greedy_min_time_s"].median()
    ratio = sub_08 / sub_02 if sub_02 > 0 else float("nan")
    fig.suptitle(
        f"C4.2: Density Effect (4x density → {ratio:.1f}x greedy time)",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_figure(fig, os.path.join(fig_dir, "individual_density_pair"))
    plt.close(fig)
    logger.info("C4.2 individual figure saved.")


def generate_c42_table(
    density_analysis: dict,
    density_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """C4.2 Table: Grid of N x p with times. 'TO' for timeout."""
    fig_dir = os.path.join(output_dir, "C4_2_density_dependence")
    os.makedirs(fig_dir, exist_ok=True)

    pivot = density_df.groupby(["n_nodes", "p"])["canonical_time_s"].median().reset_index()
    table = pivot.pivot(index="n_nodes", columns="p", values="canonical_time_s")

    # Format cells
    def _fmt(val: float) -> str:
        if not np.isfinite(val):
            return "TO"
        if val < 0.001:
            return f"{val * 1000:.2f} ms"
        if val < 1:
            return f"{val:.3f} s"
        if val < 60:
            return f"{val:.1f} s"
        return f"{val / 60:.1f} min"

    formatted = table.map(_fmt)
    formatted.index.name = "N"
    formatted.columns = [f"p={p}" for p in formatted.columns]
    formatted = formatted.reset_index()

    generate_dual_table(
        formatted,
        fig_dir,
        "table_density_dependence",
        caption="Canonical encoding time by graph size and density. TO = timeout (>300s).",
        label="tab:density",
    )
    logger.info("C4.2 table saved.")


# =============================================================================
# C4.3 — Greedy vs Canonical Overhead
# =============================================================================


def generate_c43_population(
    canonical_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """C4.3 Population: Bar chart of T_canonical / T_greedy per family (log scale)."""
    fig_dir = os.path.join(output_dir, "C4_3_greedy_vs_canonical")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.9))

    # Compute median overhead per family
    families = []
    overheads = []
    colors = []
    densities = []

    for fam in ALL_FAMILIES:
        fam_data = canonical_df[canonical_df["family"] == fam]
        if fam_data.empty:
            continue
        valid = fam_data[
            (fam_data["canonical_time_s"] > 0)
            & (fam_data["greedy_min_time_s"] > 0)
            & np.isfinite(fam_data["canonical_time_s"])
        ]
        if valid.empty:
            continue
        ratio = (valid["canonical_time_s"] / valid["greedy_min_time_s"]).median()
        families.append(fam)
        overheads.append(ratio)
        # Color by density class
        med_density = valid["density"].median() if "density" in valid.columns else 0.5
        if med_density < 0.3:
            colors.append(PAUL_TOL_MUTED[3])  # green (sparse)
        elif med_density < 0.6:
            colors.append(PAUL_TOL_MUTED[2])  # sand (medium)
        else:
            colors.append(PAUL_TOL_MUTED[0])  # rose (dense)
        densities.append(med_density)

    x = np.arange(len(families))
    bars = ax.bar(x, overheads, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Annotate values
    for bar, ov in zip(bars, overheads, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.1,
            f"{ov:.1f}x",
            ha="center",
            fontsize=7,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([family_display(f) for f in families], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"$T_{\mathrm{canonical}} / T_{\mathrm{greedy}}$")
    ax.set_title("C4.3: Canonical vs Greedy Overhead", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax.axhline(y=1, color="0.5", linestyle="--", linewidth=0.5, alpha=0.5)

    # Legend for density classes
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PAUL_TOL_MUTED[3], label="Sparse (d<0.3)"),
        Patch(facecolor=PAUL_TOL_MUTED[2], label="Medium (0.3-0.6)"),
        Patch(facecolor=PAUL_TOL_MUTED[0], label="Dense (d>0.6)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper left")

    fig.tight_layout()
    save_figure(fig, os.path.join(fig_dir, "population_overhead_ratio"))
    plt.close(fig)
    logger.info("C4.3 population figure saved.")


def generate_c43_individual(output_dir: str) -> None:
    """C4.3 Individual: BA(8,m=2) — graph + greedy strings per start node + canonical."""
    fig_dir = os.path.join(output_dir, "C4_3_greedy_vs_canonical")
    os.makedirs(fig_dir, exist_ok=True)

    import time as _time

    import networkx as nx

    from isalgraph.adapters.networkx_adapter import NetworkXAdapter
    from isalgraph.core.canonical import canonical_string
    from isalgraph.core.graph_to_string import GraphToString

    G_nx = generate_graph_family("ba_m2", 8, seed=42, instance=0)
    sg = NetworkXAdapter().from_external(G_nx, directed=False)

    # Canonical
    t0 = _time.process_time()
    canon_str = canonical_string(sg)
    t_canon = _time.process_time() - t0

    # Greedy from each starting node (new instance per run — G2S is stateful)
    greedy_results = []
    for v in range(G_nx.number_of_nodes()):
        t0 = _time.process_time()
        s, _cost = GraphToString(sg).run(initial_node=v)
        t_g = _time.process_time() - t0
        greedy_results.append((v, s, len(s), t_g))

    # Sort by string length then lexicographically
    greedy_results.sort(key=lambda x: (x[2], x[1]))
    best_greedy = greedy_results[0]

    fig = plt.figure(figsize=get_figure_size("double", height_ratio=0.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1])

    # (a) Graph
    ax_graph = fig.add_subplot(gs[0])
    pos = nx.spring_layout(G_nx, seed=42)
    nx.draw_networkx(
        G_nx,
        pos,
        ax=ax_graph,
        node_size=100,
        font_size=7,
        node_color=PAUL_TOL_BRIGHT["blue"],
        edge_color="0.6",
        width=0.8,
        with_labels=True,
        font_color="white",
    )
    ax_graph.set_title("(a) BA(8, m=2)", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax_graph.axis("off")

    # (b) Table of greedy strings
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")
    table_data = []
    cell_colors = []
    for v, s, slen, t_g in greedy_results:
        is_best = v == best_greedy[0]
        table_data.append(
            [f"v={v}", s[:25] + ("..." if len(s) > 25 else ""), str(slen), f"{t_g * 1000:.2f}"]
        )
        if is_best:
            cell_colors.append([PAUL_TOL_BRIGHT["green"] + "40"] * 4)
        else:
            cell_colors.append(["white"] * 4)

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=["Start", "String", "Len", "Time (ms)"],
        cellColours=cell_colors,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1, 1.2)
    ax_table.set_title("(b) Greedy strings", fontsize=PLOT_SETTINGS["axes_titlesize"])

    # (c) Canonical string
    ax_canon = fig.add_subplot(gs[2])
    ax_canon.axis("off")
    canon_text = (
        f'Canonical string:\n"{canon_str}"\n'
        f"Length: {len(canon_str)}\n"
        f"Time: {t_canon * 1000:.1f} ms\n\n"
        f'Best greedy (v={best_greedy[0]}):\n"{best_greedy[1]}"\n'
        f"Length: {best_greedy[2]}\n"
        f"Time: {best_greedy[3] * 1000:.2f} ms\n\n"
        f"Match: {'Yes' if canon_str == best_greedy[1] else 'No'}"
    )
    ax_canon.text(
        0.1,
        0.95,
        canon_text,
        transform=ax_canon.transAxes,
        fontsize=7,
        va="top",
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "lightyellow", "alpha": 0.8},
    )
    ax_canon.set_title("(c) Canonical", fontsize=PLOT_SETTINGS["axes_titlesize"])

    fig.suptitle("C4.3: Greedy vs Canonical — BA(8, m=2)", fontsize=PLOT_SETTINGS["axes_titlesize"])
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, os.path.join(fig_dir, "individual_overhead_example"))
    plt.close(fig)
    logger.info("C4.3 individual figure saved.")


def generate_c43_table(
    canonical_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """C4.3 Table: Family, T_greedy_min, T_canonical, Overhead, String match %."""
    fig_dir = os.path.join(output_dir, "C4_3_greedy_vs_canonical")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    for fam in ALL_FAMILIES:
        fam_data = canonical_df[canonical_df["family"] == fam]
        if fam_data.empty:
            continue
        valid = fam_data[
            np.isfinite(fam_data["canonical_time_s"]) & (fam_data["canonical_time_s"] > 0)
        ]
        if valid.empty:
            continue

        t_greedy = valid["greedy_min_time_s"].median() * 1000
        t_canon = valid["canonical_time_s"].median() * 1000
        overhead = t_canon / t_greedy if t_greedy > 0 else float("nan")
        match_pct = (
            (valid["length_gap"] == 0).mean() * 100
            if "length_gap" in valid.columns
            else float("nan")
        )

        rows.append(
            {
                "Family": family_display(fam),
                r"$T_{\text{greedy}}$ (ms)": f"{t_greedy:.2f}",
                r"$T_{\text{canon}}$ (ms)": f"{t_canon:.1f}",
                "Overhead": f"{overhead:.1f}x",
                "Match %": f"{match_pct:.0f}" if np.isfinite(match_pct) else "---",
            }
        )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        fig_dir,
        "table_overhead",
        caption="Canonical vs greedy encoding overhead per family.",
        label="tab:overhead",
        highlight_cols={"Overhead"},
        minimize_cols={"Overhead"},
    )
    logger.info("C4.3 table saved.")


# =============================================================================
# Main
# =============================================================================


def generate_all(enc_dir: str, output_dir: str) -> None:
    """Generate all encoding complexity figures and tables."""
    scaling = _load_scaling_exponents(enc_dir)
    density_analysis = _load_density_analysis(enc_dir)
    greedy_df = _load_greedy_csv(enc_dir)
    canonical_df = _load_canonical_csv(enc_dir)
    density_df = _load_density_csv(enc_dir)
    validation_df = _load_validation_csv(enc_dir)

    # C4.1
    generate_c41_population(scaling, greedy_df, canonical_df, validation_df, output_dir)
    generate_c41_individual(output_dir)
    generate_c41_table(scaling, output_dir)

    # C4.2
    generate_c42_population(density_analysis, density_df, output_dir)
    generate_c42_individual(density_df, output_dir)
    generate_c42_table(density_analysis, density_df, output_dir)

    # C4.3
    generate_c43_population(canonical_df, output_dir)
    generate_c43_individual(output_dir)
    generate_c43_table(canonical_df, output_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate encoding complexity figures (C4.1, C4.2, C4.3).",
    )
    parser.add_argument(
        "--enc-dir",
        default=DEFAULT_ENC_DIR,
        help="Directory with eval_encoding outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for figures and tables.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()

    generate_all(args.enc_dir, args.output_dir)
    logger.info("All encoding figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
