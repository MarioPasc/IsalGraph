# ruff: noqa: N803, N806
"""Population figures and tables for embedding hypotheses H2.1-H2.3.

Generates:
  - H2.1: Stress-1 vs dimension plot + embedding quality table
  - H2.2: Procrustes m² grouped bar + table
  - H2.3: Shepard R² grouped bar + table
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.eval_visualizations.embedding_loader import (
    DIMENSIONS,
    PROCRUSTES_DIMS,
    EmbeddingData,
)
from benchmarks.eval_visualizations.result_loader import (
    ALL_DATASETS,
    DATASET_DISPLAY,
)
from benchmarks.eval_visualizations.table_generator import (
    format_significance,
    generate_dual_table,
)
from benchmarks.plotting_styles import PAUL_TOL_MUTED, save_figure

logger = logging.getLogger(__name__)

# Dataset line styles for stress plot
_DS_COLORS = {ds: PAUL_TOL_MUTED[i] for i, ds in enumerate(ALL_DATASETS)}

# Kruskal stress interpretation thresholds
_STRESS_THRESHOLDS = [
    (0.20, "Poor", "0.6"),
    (0.10, "Fair", "0.6"),
    (0.05, "Good", "0.6"),
]


# =====================================================================
# H2.1 — Stress-1 vs dimension (Low Distortion)
# =====================================================================


def generate_h2_1_population(
    emb: EmbeddingData,
    output_dir: str,
    method: str = "exhaustive",
) -> str:
    """Generate H2.1 population figure: Stress-1 vs embedding dimension.

    Lines for each dataset: solid = Lev(method), dashed = GED.
    Horizontal threshold lines at Kruskal interpretation boundaries.
    """
    fig, ax = plt.subplots(figsize=(3.39, 3.39 * 0.85))

    dims = np.array(DIMENSIONS)

    for dataset in ALL_DATASETS:
        ds_stats = emb.stats.get(dataset)
        if ds_stats is None:
            continue

        color = _DS_COLORS[dataset]
        label = DATASET_DISPLAY[dataset]

        # GED stress
        ged_stress = [
            ds_stats.ged_smacof[d].stress_1 if d in ds_stats.ged_smacof else np.nan
            for d in DIMENSIONS
        ]
        ax.plot(
            dims,
            ged_stress,
            linestyle="--",
            color=color,
            marker="s",
            markersize=3,
            linewidth=1.0,
            alpha=0.6,
        )

        # Lev stress (solid)
        lev_stress = [
            ds_stats.lev_smacof[(method, d)].stress_1
            if (method, d) in ds_stats.lev_smacof
            else np.nan
            for d in DIMENSIONS
        ]
        ax.plot(
            dims,
            lev_stress,
            linestyle="-",
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
        )

    # Threshold lines
    for thresh, interpretation, gray in _STRESS_THRESHOLDS:
        ax.axhline(thresh, color=gray, linestyle=":", linewidth=0.7, zorder=0)
        ax.text(
            dims[-1] + 0.4,
            thresh,
            interpretation,
            fontsize=6,
            va="center",
            color="0.4",
        )

    ax.set_xlabel("Embedding dimension $d$")
    ax.set_ylabel("Stress-1")
    ax.set_xticks(DIMENSIONS)
    ax.set_xlim(1.5, 11.5)
    ax.set_ylim(bottom=0)

    # Custom legend: solid = Lev, dashed = GED
    legend = ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
    legend.set_title("Solid = Lev, Dashed = GED", prop={"size": 5.5})

    fig.tight_layout()
    path = os.path.join(output_dir, "population_stress_by_dimension")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H2.1 population figure saved: %s", path)
    return path


def generate_h2_1_table(
    emb: EmbeddingData,
    output_dir: str,
) -> str:
    """Generate H2.1 table: embedding quality summary per dataset × method."""
    rows = []
    for dataset in ALL_DATASETS:
        ds = emb.stats.get(dataset)
        if ds is None:
            continue

        for method in ["exhaustive", "greedy"]:
            ged_s2 = ds.ged_smacof.get(2)
            ged_s5 = ds.ged_smacof.get(5)
            lev_s2 = ds.lev_smacof.get((method, 2))
            lev_s5 = ds.lev_smacof.get((method, 5))

            rows.append(
                {
                    "Dataset": DATASET_DISPLAY[dataset],
                    "Method": method.capitalize(),
                    "NEV (GED)": f"{ds.ged_nev_ratio:.3f}",
                    "NEV (Lev)": f"{ds.lev_nev_ratio.get(method, 0):.3f}",
                    "Stress-1 2D (GED)": f"{ged_s2.stress_1:.4f}" if ged_s2 else "--",
                    "Stress-1 2D (Lev)": f"{lev_s2.stress_1:.4f}" if lev_s2 else "--",
                    "Stress-1 5D (GED)": f"{ged_s5.stress_1:.4f}" if ged_s5 else "--",
                    "Stress-1 5D (Lev)": f"{lev_s5.stress_1:.4f}" if lev_s5 else "--",
                }
            )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        output_dir,
        "table_embedding_quality",
        caption="Embedding quality: NEV ratio and SMACOF Stress-1.",
        label="tab:embedding_quality",
        highlight_cols={"Stress-1 2D (Lev)", "Stress-1 5D (Lev)"},
        minimize_cols={"Stress-1 2D (Lev)", "Stress-1 5D (Lev)"},
    )
    logger.info("H2.1 table saved to %s", output_dir)
    return output_dir


# =====================================================================
# H2.2 — Procrustes m² (Geometric Agreement)
# =====================================================================


def generate_h2_2_population(
    emb: EmbeddingData,
    output_dir: str,
    method: str = "exhaustive",
) -> str:
    """Generate H2.2 population figure: Procrustes m² grouped bar + significance."""
    fig, ax = plt.subplots(figsize=(3.39, 3.39 * 0.85))

    datasets = [ds for ds in ALL_DATASETS if ds in emb.stats]
    n_ds = len(datasets)
    bar_width = 0.35
    x = np.arange(n_ds)

    # m² at d=2 and d=5
    m2_d2 = []
    m2_d5 = []
    p_d2 = []
    p_d5 = []

    for ds in datasets:
        ds_stats = emb.stats[ds]
        proc_2 = ds_stats.procrustes.get((method, 2))
        proc_5 = ds_stats.procrustes.get((method, 5))

        m2_d2.append(proc_2.m_squared if proc_2 else 0)
        m2_d5.append(proc_5.m_squared if proc_5 else 0)
        p_d2.append(proc_2.p_value if proc_2 else 1.0)
        p_d5.append(proc_5.p_value if proc_5 else 1.0)

    bars_2d = ax.bar(
        x - bar_width / 2,
        m2_d2,
        bar_width,
        label="$d = 2$",
        color=PAUL_TOL_MUTED[4],
        edgecolor="white",
        linewidth=0.5,
    )
    bars_5d = ax.bar(
        x + bar_width / 2,
        m2_d5,
        bar_width,
        label="$d = 5$",
        color=PAUL_TOL_MUTED[1],
        edgecolor="white",
        linewidth=0.5,
    )

    # Significance stars above bars
    for i, (bar2, bar5) in enumerate(zip(bars_2d, bars_5d, strict=True)):
        stars_2 = format_significance(p_d2[i])
        stars_5 = format_significance(p_d5[i])
        ax.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar2.get_height() + 0.01,
            stars_2,
            ha="center",
            va="bottom",
            fontsize=6,
        )
        ax.text(
            bar5.get_x() + bar5.get_width() / 2,
            bar5.get_height() + 0.01,
            stars_5,
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # Reference: m²=1.0 means no agreement (random)
    ax.axhline(1.0, color="0.6", linestyle=":", linewidth=0.7, zorder=0)
    ax.text(n_ds - 0.5, 1.02, "No agreement", fontsize=5.5, color="0.4", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_DISPLAY[ds] for ds in datasets], fontsize=6, rotation=25, ha="right"
    )
    ax.set_ylabel("Procrustes $m^2$")
    ax.set_ylim(0, min(max(max(m2_d2), max(m2_d5)) * 1.25, 1.15))
    ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    path = os.path.join(output_dir, "population_procrustes_comparison")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H2.2 population figure saved: %s", path)
    return path


def generate_h2_2_table(
    emb: EmbeddingData,
    output_dir: str,
) -> str:
    """Generate H2.2 table: Procrustes m² and p-values per dataset × method × dim."""
    rows = []
    for dataset in ALL_DATASETS:
        ds = emb.stats.get(dataset)
        if ds is None:
            continue

        for method in ["exhaustive", "greedy"]:
            for dim in PROCRUSTES_DIMS:
                proc = ds.procrustes.get((method, dim))
                if proc is None:
                    continue

                stars = format_significance(proc.p_value)
                rows.append(
                    {
                        "Dataset": DATASET_DISPLAY[dataset],
                        "Method": method.capitalize(),
                        "Dim": f"{dim}D",
                        "$m^2$": f"{proc.m_squared:.4f}",
                        "$p$-value": f"{proc.p_value:.4f} {stars}",
                    }
                )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        output_dir,
        "table_procrustes",
        caption="Procrustes analysis: $m^2$ disparity between GED and Levenshtein MDS.",
        label="tab:procrustes",
        highlight_cols={"$m^2$"},
        minimize_cols={"$m^2$"},
    )
    logger.info("H2.2 table saved to %s", output_dir)
    return output_dir


# =====================================================================
# H2.3 — Shepard R² (Fidelity)
# =====================================================================


def generate_h2_3_population(
    emb: EmbeddingData,
    output_dir: str,
    method: str = "exhaustive",
) -> str:
    """Generate H2.3 population figure: Shepard R² grouped bar per dataset × dim."""
    fig, ax = plt.subplots(figsize=(3.39, 3.39 * 0.85))

    datasets = [ds for ds in ALL_DATASETS if ds in emb.stats]
    n_ds = len(datasets)
    bar_width = 0.2
    x = np.arange(n_ds)

    # 4 groups: GED 2D, GED 5D, Lev 2D, Lev 5D
    groups = [
        ("GED $d{=}2$", "ged", 2, PAUL_TOL_MUTED[0]),
        ("GED $d{=}5$", "ged", 5, PAUL_TOL_MUTED[2]),
        ("Lev $d{=}2$", "lev", 2, PAUL_TOL_MUTED[4]),
        ("Lev $d{=}5$", "lev", 5, PAUL_TOL_MUTED[1]),
    ]

    for g_idx, (label, source, dim, color) in enumerate(groups):
        vals = []
        for ds in datasets:
            ds_stats = emb.stats[ds]
            shep = ds_stats.shepard.get((method, dim))
            if shep is None:
                vals.append(0)
                continue
            if source == "ged":
                vals.append(shep.ged_r_squared)
            else:
                vals.append(shep.lev_r_squared)

        offset = (g_idx - 1.5) * bar_width
        ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_DISPLAY[ds] for ds in datasets], fontsize=6, rotation=25, ha="right"
    )
    ax.set_ylabel("Shepard $R^2$")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=5.5, loc="lower left", ncol=2)

    fig.tight_layout()
    path = os.path.join(output_dir, "population_shepard_r2_summary")
    save_figure(fig, path)
    plt.close(fig)
    logger.info("H2.3 population figure saved: %s", path)
    return path


def generate_h2_3_table(
    emb: EmbeddingData,
    output_dir: str,
) -> str:
    """Generate H2.3 table: Shepard R² per dataset × source × dim."""
    rows = []
    for dataset in ALL_DATASETS:
        ds = emb.stats.get(dataset)
        if ds is None:
            continue

        for method in ["exhaustive", "greedy"]:
            for dim in PROCRUSTES_DIMS:
                shep = ds.shepard.get((method, dim))
                if shep is None:
                    continue

                rows.append(
                    {
                        "Dataset": DATASET_DISPLAY[dataset],
                        "Method": method.capitalize(),
                        "Dim": f"{dim}D",
                        "$R^2$ (GED)": f"{shep.ged_r_squared:.4f}",
                        "$R^2$ (Lev)": f"{shep.lev_r_squared:.4f}",
                        "$R^2_{\\text{mono}}$ (GED)": f"{shep.ged_monotonic_r_squared:.4f}",
                        "$R^2_{\\text{mono}}$ (Lev)": f"{shep.lev_monotonic_r_squared:.4f}",
                    }
                )

    df = pd.DataFrame(rows)
    generate_dual_table(
        df,
        output_dir,
        "table_shepard",
        caption="Shepard diagram fidelity: $R^2$ and monotonic $R^2$.",
        label="tab:shepard",
        highlight_cols={"$R^2$ (Lev)", "$R^2_{\\text{mono}}$ (Lev)"},
    )
    logger.info("H2.3 table saved to %s", output_dir)
    return output_dir


# =====================================================================
# Orchestration
# =====================================================================


def generate_all_embedding_population(
    emb: EmbeddingData,
    output_root: str,
) -> None:
    """Generate all population figures and tables for H2.1-H2.3."""
    # H2.1
    h2_1_dir = os.path.join(output_root, "H2_1_low_distortion")
    os.makedirs(h2_1_dir, exist_ok=True)
    generate_h2_1_population(emb, h2_1_dir)
    generate_h2_1_table(emb, h2_1_dir)

    # H2.2
    h2_2_dir = os.path.join(output_root, "H2_2_geometric_agreement")
    os.makedirs(h2_2_dir, exist_ok=True)
    generate_h2_2_population(emb, h2_2_dir)
    generate_h2_2_table(emb, h2_2_dir)

    # H2.3
    h2_3_dir = os.path.join(output_root, "H2_3_shepard_fidelity")
    os.makedirs(h2_3_dir, exist_ok=True)
    generate_h2_3_population(emb, h2_3_dir)
    generate_h2_3_table(emb, h2_3_dir)
