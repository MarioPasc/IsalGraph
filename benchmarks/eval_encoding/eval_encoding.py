# ruff: noqa: N803, N806
"""CLI orchestrator for encoding complexity analysis.

Characterises the empirical scaling of IsalGraph encoding algorithms
using synthetic graphs with controlled properties. Measures greedy
single-start, greedy-min, and canonical encoding times as a function
of graph size and density.

Usage:
    python -m benchmarks.eval_encoding.eval_encoding \
        --output-dir results/eval_encoding \
        --n-instances 5 --n-reps 25 --seed 42 \
        --csv --plot --table
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import signal
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from benchmarks.eval_computational.timing_utils import (
    fit_scaling_exponent,
    get_hardware_info,
    time_function,
)
from benchmarks.eval_encoding.synthetic_generator import (
    ALL_FAMILIES,
    FAMILY_CONFIGS,
    compute_density,
    generate_density_sweep,
    generate_graph_family,
    get_family_n_range,
)
from benchmarks.plotting_styles import (
    FAMILY_COLORS,
    FAMILY_MARKERS,
    PLOT_SETTINGS,
    apply_ieee_style,
    family_display,
    save_figure,
    save_latex_table,
)
from isalgraph.adapters.networkx_adapter import NetworkXAdapter
from isalgraph.core.canonical import canonical_string
from isalgraph.core.graph_to_string import GraphToString

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalgraph/results/eval_encoding"
DEFAULT_DATA_ROOT = "/media/mpascual/Sandisk2TB/research/isalgraph/data/eval"

# Per-family canonical feasibility limits
CANONICAL_N_LIMITS: dict[str, int] = {
    "path": 20,
    "star": 20,
    "cycle": 20,
    "binary_tree": 20,
    "ba_m1": 15,
    "grid": 15,
    "ba_m2": 12,
    "gnp_03": 12,
    "complete": 8,
    "gnp_05": 8,
}

DENSITY_P_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DENSITY_N = 10
DENSITY_N_INSTANCES = 10

GREEDY_TIMEOUT_S = 60.0
CANONICAL_PROBE_TIMEOUT_S = 60.0


# =============================================================================
# Timeout helper
# =============================================================================


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise _TimeoutError("Timed out")


# =============================================================================
# Experiment 1: Greedy single-start scaling
# =============================================================================


def _experiment_greedy_scaling(
    families: list[str],
    max_n: int,
    n_instances: int,
    n_reps: int,
    seed: int,
) -> pd.DataFrame:
    """Measure greedy GraphToString time vs graph size per family."""
    adapter = NetworkXAdapter()
    rows: list[dict] = []

    for family in families:
        n_range = get_family_n_range(family, max_n=max_n)
        n_inst = 1 if FAMILY_CONFIGS[family]["deterministic"] else n_instances
        family_timed_out = False

        logger.info(
            "  Greedy scaling: %s (n=%d..%d, %d instances)", family, n_range[0], n_range[-1], n_inst
        )

        for n in n_range:
            if family_timed_out:
                break

            for inst in range(n_inst):
                G = generate_graph_family(family, n, seed=seed, instance=inst)
                sg = adapter.from_external(G, directed=False)
                actual_n = sg.node_count()
                n_edges = G.number_of_edges()
                dens = compute_density(G)

                # Time up to 5 starting nodes
                n_starts = min(actual_n, 5)

                for v in range(n_starts):
                    # Quick probe for timeout
                    t_probe = time.process_time()
                    try:
                        GraphToString(sg).run(initial_node=v)
                    except Exception:
                        logger.warning("Greedy failed: %s n=%d v=%d", family, n, v)
                        continue
                    t_probe_elapsed = time.process_time() - t_probe

                    if t_probe_elapsed > GREEDY_TIMEOUT_S:
                        logger.warning(
                            "Greedy timeout: %s n=%d (%.1fs)", family, n, t_probe_elapsed
                        )
                        family_timed_out = True
                        rows.append(
                            {
                                "family": family,
                                "n_nodes": actual_n,
                                "n_edges": n_edges,
                                "density": dens,
                                "instance": inst,
                                "start_node": v,
                                "greedy_time_s": float("inf"),
                                "string_length": -1,
                                "iqr_s": float("nan"),
                                "n_reps": 0,
                            }
                        )
                        break

                    # Actual timed measurement (probe counts as warmup)
                    def _greedy_run(sg_l=sg, v_l=v):  # noqa: B023
                        return GraphToString(sg_l).run(initial_node=v_l)

                    reps = n_reps if actual_n <= 30 else max(5, n_reps // 3)
                    timing = time_function(_greedy_run, n_reps=reps, warmup=0)

                    result_str = timing["result"][0]
                    rows.append(
                        {
                            "family": family,
                            "n_nodes": actual_n,
                            "n_edges": n_edges,
                            "density": dens,
                            "instance": inst,
                            "start_node": v,
                            "greedy_time_s": timing["median_s"],
                            "string_length": len(result_str),
                            "iqr_s": timing["iqr_s"],
                            "n_reps": reps,
                        }
                    )

                if family_timed_out:
                    break

        logger.info("  Greedy scaling: %s done (%d rows)", family, len(rows))

    return pd.DataFrame(rows)


# =============================================================================
# Experiment 2: Canonical scaling
# =============================================================================


def _experiment_canonical_scaling(
    families: list[str],
    max_n_canonical: int,
    n_instances: int,
    n_reps: int,
    seed: int,
) -> pd.DataFrame:
    """Measure canonical and greedy-min time vs graph size."""
    adapter = NetworkXAdapter()
    rows: list[dict] = []

    for family in families:
        family_limit = min(
            CANONICAL_N_LIMITS.get(family, 8),
            max_n_canonical,
        )
        n_range = get_family_n_range(family, max_n=family_limit)
        n_inst = 1 if FAMILY_CONFIGS[family]["deterministic"] else n_instances
        family_timed_out = False

        logger.info(
            "  Canonical scaling: %s (n=%d..%d, limit=%d)",
            family,
            n_range[0],
            n_range[-1],
            family_limit,
        )

        for n in n_range:
            if family_timed_out:
                break

            for inst in range(n_inst):
                G = generate_graph_family(family, n, seed=seed, instance=inst)
                sg = adapter.from_external(G, directed=False)
                actual_n = sg.node_count()
                n_edges = G.number_of_edges()
                dens = compute_density(G)

                # Probe: run canonical once untimed with timeout
                try:
                    # Use SIGALRM for timeout on Unix
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(int(CANONICAL_PROBE_TIMEOUT_S) + 1)
                    t_probe_start = time.process_time()
                    canonical_string(sg)  # probe only, result discarded
                    t_probe_elapsed = time.process_time() - t_probe_start
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except (_TimeoutError, Exception) as e:
                    signal.alarm(0)
                    with contextlib.suppress(Exception):
                        signal.signal(signal.SIGALRM, old_handler)
                    logger.warning("Canonical probe timeout/error: %s n=%d (%s)", family, n, e)
                    family_timed_out = True
                    rows.append(
                        {
                            "family": family,
                            "n_nodes": actual_n,
                            "n_edges": n_edges,
                            "density": dens,
                            "instance": inst,
                            "canonical_time_s": float("inf"),
                            "greedy_min_time_s": float("nan"),
                            "canonical_length": -1,
                            "greedy_min_length": -1,
                            "length_gap": 0,
                            "n_reps": 0,
                        }
                    )
                    break

                if t_probe_elapsed > CANONICAL_PROBE_TIMEOUT_S:
                    logger.warning(
                        "Canonical probe slow: %s n=%d (%.1fs)", family, n, t_probe_elapsed
                    )
                    family_timed_out = True
                    break

                # Timed canonical measurement
                reps = n_reps if actual_n <= 10 else max(3, n_reps // 5)

                def _canonical_run(sg_l=sg):
                    return canonical_string(sg_l)

                can_timing = time_function(_canonical_run, n_reps=reps, warmup=0)
                can_str = can_timing["result"]

                # Timed greedy-min measurement
                def _greedy_min_run(sg_l=sg):
                    best_s = None
                    for v in range(sg_l.node_count()):
                        s, _ = GraphToString(sg_l).run(initial_node=v)
                        if (
                            best_s is None
                            or len(s) < len(best_s)
                            or (len(s) == len(best_s) and s < best_s)
                        ):
                            best_s = s
                    return best_s

                gm_timing = time_function(_greedy_min_run, n_reps=reps, warmup=0)
                gm_str = gm_timing["result"]

                can_len = len(can_str)
                gm_len = len(gm_str) if gm_str else -1

                rows.append(
                    {
                        "family": family,
                        "n_nodes": actual_n,
                        "n_edges": n_edges,
                        "density": dens,
                        "instance": inst,
                        "canonical_time_s": can_timing["median_s"],
                        "greedy_min_time_s": gm_timing["median_s"],
                        "canonical_length": can_len,
                        "greedy_min_length": gm_len,
                        "length_gap": gm_len - can_len if gm_len > 0 else 0,
                        "n_reps": reps,
                    }
                )

            if family_timed_out:
                break

        logger.info("  Canonical scaling: %s done (%d rows)", family, len(rows))

    return pd.DataFrame(rows)


# =============================================================================
# Experiment 3: Density dependence
# =============================================================================


def _experiment_density_dependence(
    n_reps: int,
    seed: int,
) -> pd.DataFrame:
    """Measure encoding time as function of edge density at fixed n."""
    adapter = NetworkXAdapter()
    rows: list[dict] = []

    sweep = generate_density_sweep(
        DENSITY_N,
        DENSITY_P_VALUES,
        DENSITY_N_INSTANCES,
        seed=seed,
    )

    logger.info(
        "  Density dependence: n=%d, %d p-values, %d instances each",
        DENSITY_N,
        len(DENSITY_P_VALUES),
        DENSITY_N_INSTANCES,
    )

    for entry in sweep:
        G = entry["graph"]
        sg = adapter.from_external(G, directed=False)
        actual_n = sg.node_count()

        # Greedy-min
        def _greedy_min_run(sg_l=sg):
            best_s = None
            for v in range(sg_l.node_count()):
                s, _ = GraphToString(sg_l).run(initial_node=v)
                if best_s is None or len(s) < len(best_s) or (len(s) == len(best_s) and s < best_s):
                    best_s = s
            return best_s

        gm_timing = time_function(_greedy_min_run, n_reps=n_reps, warmup=1)

        # Canonical (with timeout protection)
        can_time_s = float("nan")
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(CANONICAL_PROBE_TIMEOUT_S) + 1)
            can_timing = time_function(
                lambda sg_l=sg: canonical_string(sg_l),
                n_reps=max(3, n_reps // 5),
                warmup=0,
            )
            can_time_s = can_timing["median_s"]
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except (_TimeoutError, Exception):
            signal.alarm(0)
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGALRM, old_handler)
            can_time_s = float("inf")

        rows.append(
            {
                "p": entry["p"],
                "density": entry["density"],
                "instance": entry["instance"],
                "n_nodes": actual_n,
                "n_edges": entry["n_edges"],
                "greedy_min_time_s": gm_timing["median_s"],
                "canonical_time_s": can_time_s,
            }
        )

    logger.info("  Density dependence done (%d rows)", len(rows))
    return pd.DataFrame(rows)


# =============================================================================
# Experiment 4: Real dataset validation
# =============================================================================


def _experiment_real_validation(
    data_root: str,
    greedy_exponents: dict[str, dict],
) -> pd.DataFrame:
    """Validate synthetic scaling predictions against real dataset data."""
    rows: list[dict] = []

    # Try to load precomputed canonical string data
    cs_dir = os.path.join(data_root, "canonical_strings")
    if not os.path.isdir(cs_dir):
        logger.warning("No canonical_strings directory at %s, skipping real validation", cs_dir)
        return pd.DataFrame(rows)

    datasets = ["iam_letter_low", "linux", "aids"]

    for dataset in datasets:
        # Try exhaustive first, then greedy
        cs_path = None
        for method in ["exhaustive", "greedy"]:
            candidate = os.path.join(cs_dir, f"{dataset}_{method}.json")
            if os.path.isfile(candidate):
                cs_path = candidate
                break

        if cs_path is None:
            logger.warning("No canonical strings for %s, skipping", dataset)
            continue

        with open(cs_path) as f:
            cs_data = json.load(f)

        # Load graph metadata for node counts
        meta_path = os.path.join(data_root, "graph_metadata", f"{dataset}.json")
        node_count_map: dict[str, int] = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            for gid, nc in zip(
                meta.get("graph_ids", []), meta.get("node_counts", []), strict=False
            ):
                node_count_map[gid] = nc

        # Get overall alpha from greedy exponents
        alpha_pred = greedy_exponents.get("overall", {}).get("alpha", float("nan"))
        c_pred = greedy_exponents.get("overall", {}).get("c", float("nan"))

        # Format: {"strings": {"graph_id": {"string": ..., "length": ..., "time_s": ...}}}
        strings_dict = cs_data.get("strings", {})
        for graph_id, entry in strings_dict.items():
            obs_time = entry.get("time_s", float("nan"))
            str_len = entry.get("length", len(entry.get("string", "")))
            n_nodes = node_count_map.get(graph_id, 0)

            if n_nodes < 2 or not np.isfinite(obs_time):
                continue

            # Predicted time from synthetic regression
            pred_time = c_pred * n_nodes**alpha_pred if np.isfinite(alpha_pred) else float("nan")

            rows.append(
                {
                    "dataset": dataset,
                    "graph_id": graph_id,
                    "n_nodes": n_nodes,
                    "observed_time_s": obs_time,
                    "predicted_time_s": pred_time,
                    "string_length": str_len,
                }
            )

    logger.info("  Real validation: %d entries", len(rows))
    return pd.DataFrame(rows)


# =============================================================================
# Scaling regression
# =============================================================================


def fit_polynomial_or_exponential(
    n_values: np.ndarray,
    times: np.ndarray,
) -> dict:
    """Fit polynomial and exponential models, select best.

    Polynomial: T(n) = c * n^alpha  (log-log OLS)
    Exponential: T(n) = c * b^n     (semi-log OLS)

    Selects exponential if R^2_exp > R^2_poly + 0.05.

    Args:
        n_values: Array of graph sizes.
        times: Array of corresponding median times.

    Returns:
        Dict with model type, parameters, and R^2.
    """
    mask = (n_values > 0) & (times > 0) & np.isfinite(times)
    if mask.sum() < 3:
        return {
            "model": "insufficient_data",
            "alpha": float("nan"),
            "c": float("nan"),
            "r_squared": float("nan"),
            "n_points": int(mask.sum()),
        }

    n_valid = n_values[mask].astype(np.float64)
    t_valid = times[mask].astype(np.float64)

    log_n = np.log(n_valid)
    log_t = np.log(t_valid)

    # Polynomial fit: log(T) = alpha * log(n) + log(c)
    slope_poly, intercept_poly, r_poly, _, _ = stats.linregress(log_n, log_t)
    r2_poly = float(r_poly**2)

    # Exponential fit: log(T) = n * log(b) + log(c)
    slope_exp, intercept_exp, r_exp, _, _ = stats.linregress(n_valid, log_t)
    r2_exp = float(r_exp**2)

    if r2_exp > r2_poly + 0.05:
        return {
            "model": "exponential",
            "base": float(np.exp(slope_exp)),
            "c": float(np.exp(intercept_exp)),
            "r_squared": r2_exp,
            "r_squared_poly": r2_poly,
            "alpha_poly": float(slope_poly),
            "n_points": int(mask.sum()),
        }

    return {
        "model": "polynomial",
        "alpha": float(slope_poly),
        "c": float(np.exp(intercept_poly)),
        "r_squared": r2_poly,
        "r_squared_exp": r2_exp,
        "n_points": int(mask.sum()),
    }


def _compute_all_scaling_exponents(
    greedy_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
) -> dict:
    """Compute scaling exponents for all families and methods."""
    result: dict = {
        "greedy_single": {},
        "greedy_min": {},
        "canonical": {},
    }

    # Greedy single-start: median time per n_nodes, per family
    for family in greedy_df["family"].unique():
        fdf = greedy_df[greedy_df["family"] == family]
        medians = fdf.groupby("n_nodes")["greedy_time_s"].median().reset_index()
        medians = medians[np.isfinite(medians["greedy_time_s"])]
        if len(medians) < 3:
            continue

        reg = fit_polynomial_or_exponential(
            medians["n_nodes"].values,
            medians["greedy_time_s"].values,
        )
        result["greedy_single"][family] = reg

    # Overall greedy single-start
    medians_all = greedy_df.groupby("n_nodes")["greedy_time_s"].median().reset_index()
    medians_all = medians_all[np.isfinite(medians_all["greedy_time_s"])]
    if len(medians_all) >= 3:
        result["greedy_single"]["overall"] = fit_scaling_exponent(
            medians_all["n_nodes"].values,
            medians_all["greedy_time_s"].values,
        )

    # Canonical and greedy-min per family
    if not canonical_df.empty:
        for family in canonical_df["family"].unique():
            fdf = canonical_df[canonical_df["family"] == family]

            # Canonical
            can_medians = fdf.groupby("n_nodes")["canonical_time_s"].median().reset_index()
            can_medians = can_medians[np.isfinite(can_medians["canonical_time_s"])]
            if len(can_medians) >= 3:
                result["canonical"][family] = fit_polynomial_or_exponential(
                    can_medians["n_nodes"].values,
                    can_medians["canonical_time_s"].values,
                )

            # Greedy-min
            gm_medians = fdf.groupby("n_nodes")["greedy_min_time_s"].median().reset_index()
            gm_medians = gm_medians[np.isfinite(gm_medians["greedy_min_time_s"])]
            if len(gm_medians) >= 3:
                result["greedy_min"][family] = fit_polynomial_or_exponential(
                    gm_medians["n_nodes"].values,
                    gm_medians["greedy_min_time_s"].values,
                )

        # Overall canonical
        can_all = canonical_df.groupby("n_nodes")["canonical_time_s"].median().reset_index()
        can_all = can_all[np.isfinite(can_all["canonical_time_s"])]
        if len(can_all) >= 3:
            result["canonical"]["overall"] = fit_polynomial_or_exponential(
                can_all["n_nodes"].values,
                can_all["canonical_time_s"].values,
            )

        # Overall greedy-min
        gm_all = canonical_df.groupby("n_nodes")["greedy_min_time_s"].median().reset_index()
        gm_all = gm_all[np.isfinite(gm_all["greedy_min_time_s"])]
        if len(gm_all) >= 3:
            result["greedy_min"]["overall"] = fit_polynomial_or_exponential(
                gm_all["n_nodes"].values,
                gm_all["greedy_min_time_s"].values,
            )

    return result


# =============================================================================
# Visualization
# =============================================================================


def _plot_main_figure(
    greedy_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    scaling_exponents: dict,
    output_dir: str,
) -> None:
    """Create 2x2 main figure.

    (a) Log-log greedy single-start scaling by family
    (b) Log-log canonical scaling by family
    (c) Density heatmap (canonical time vs n and density)
    (d) Scaling exponent bar chart
    """
    apply_ieee_style()
    fig, axes = plt.subplots(2, 2, figsize=(PLOT_SETTINGS["figure_width_double"], 5.0))

    # ---- Panel (a): Greedy single-start scaling ----
    ax_a = axes[0, 0]
    for family in greedy_df["family"].unique():
        fdf = greedy_df[greedy_df["family"] == family]
        fdf_finite = fdf[np.isfinite(fdf["greedy_time_s"])]
        if fdf_finite.empty:
            continue

        medians = fdf_finite.groupby("n_nodes")["greedy_time_s"].median()
        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")

        ax_a.scatter(
            medians.index,
            medians.values,
            c=color,
            marker=marker,
            s=12,
            alpha=0.7,
            label=family_display(family),
        )

        # Regression line
        fam_scaling = scaling_exponents.get("greedy_single", {}).get(family, {})
        alpha = fam_scaling.get("alpha", float("nan"))
        c = fam_scaling.get("c", float("nan"))
        if (
            np.isfinite(alpha)
            and np.isfinite(c)
            and fam_scaling.get("model", "polynomial") == "polynomial"
        ):
            ns = np.linspace(medians.index.min(), medians.index.max(), 50)
            ax_a.plot(ns, c * ns**alpha, "-", color=color, alpha=0.4, lw=0.8)

    # Reference lines
    ns_ref = np.array([3, 50])
    for exp, ls, label_str in [
        (2, ":", r"$O(n^2)$"),
        (3, "--", r"$O(n^3)$"),
        (4, "-.", r"$O(n^4)$"),
    ]:
        # Scale to fit visible range
        ref_y = (ns_ref.astype(float) / 10.0) ** exp * 1e-5
        ax_a.plot(ns_ref, ref_y, ls, color="grey", alpha=0.4, lw=0.7, label=label_str)

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("n (nodes)")
    ax_a.set_ylabel("Greedy time (s)")
    ax_a.set_title("(a) Greedy single-start")
    ax_a.legend(fontsize=5, loc="upper left", ncol=2)

    # ---- Panel (b): Canonical scaling ----
    ax_b = axes[0, 1]
    if not canonical_df.empty:
        for family in canonical_df["family"].unique():
            fdf = canonical_df[canonical_df["family"] == family]
            fdf_finite = fdf[np.isfinite(fdf["canonical_time_s"])]
            if fdf_finite.empty:
                continue

            medians = fdf_finite.groupby("n_nodes")["canonical_time_s"].median()
            color = FAMILY_COLORS.get(family, "#333333")
            marker = FAMILY_MARKERS.get(family, "o")

            fam_scaling = scaling_exponents.get("canonical", {}).get(family, {})
            is_exp = fam_scaling.get("model") == "exponential"

            ax_b.scatter(
                medians.index,
                medians.values,
                c=color,
                marker=marker,
                s=12,
                alpha=0.7,
                label=family_display(family) + (" (exp)" if is_exp else ""),
            )

            # Regression line
            if is_exp:
                base = fam_scaling.get("base", 1.0)
                c_val = fam_scaling.get("c", 1.0)
                ns = np.linspace(medians.index.min(), medians.index.max(), 50)
                ax_b.plot(ns, c_val * base**ns, "--", color=color, alpha=0.4, lw=0.8)
            else:
                alpha = fam_scaling.get("alpha", float("nan"))
                c_val = fam_scaling.get("c", float("nan"))
                if np.isfinite(alpha) and np.isfinite(c_val):
                    ns = np.linspace(medians.index.min(), medians.index.max(), 50)
                    ax_b.plot(ns, c_val * ns**alpha, "-", color=color, alpha=0.4, lw=0.8)

    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("n (nodes)")
    ax_b.set_ylabel("Canonical time (s)")
    ax_b.set_title("(b) Canonical encoding")
    ax_b.legend(fontsize=5, loc="upper left", ncol=2)

    # ---- Panel (c): Density heatmap ----
    ax_c = axes[1, 0]
    density_csv = os.path.join(output_dir, "raw", "density_dependence.csv")
    if os.path.isfile(density_csv):
        dd_df = pd.read_csv(density_csv)
        if not dd_df.empty and "canonical_time_s" in dd_df.columns:
            # Pivot for heatmap
            dd_finite = dd_df[
                np.isfinite(dd_df["canonical_time_s"]) & (dd_df["canonical_time_s"] > 0)
            ]
            if not dd_finite.empty:
                pivot = dd_finite.groupby("p")["canonical_time_s"].median()
                ax_c.bar(
                    pivot.index,
                    pivot.values,
                    width=0.08,
                    color=FAMILY_COLORS.get("gnp_03", "#DDCC77"),
                    alpha=0.8,
                )
                ax_c.set_yscale("log")
                ax_c.set_xlabel("Edge probability p")
                ax_c.set_ylabel("Canonical time (s)")
                ax_c.set_title("(c) Density dependence (n=10)")
            else:
                ax_c.text(0.5, 0.5, "No finite data", transform=ax_c.transAxes, ha="center")
                ax_c.set_title("(c) Density dependence")
        else:
            ax_c.text(0.5, 0.5, "No data", transform=ax_c.transAxes, ha="center")
            ax_c.set_title("(c) Density dependence")
    else:
        ax_c.text(0.5, 0.5, "No data", transform=ax_c.transAxes, ha="center")
        ax_c.set_title("(c) Density dependence")

    # ---- Panel (d): Scaling exponent bar chart ----
    ax_d = axes[1, 1]
    families_with_data = sorted(
        set(list(scaling_exponents.get("greedy_single", {}).keys())) - {"overall"}
    )

    if families_with_data:
        x_pos = np.arange(len(families_with_data))
        width = 0.35

        greedy_alphas = []
        canonical_alphas = []
        for fam in families_with_data:
            g_info = scaling_exponents.get("greedy_single", {}).get(fam, {})
            c_info = scaling_exponents.get("canonical", {}).get(fam, {})

            g_alpha = g_info.get("alpha", float("nan"))
            c_alpha = c_info.get("alpha", float("nan"))
            # For exponential models, use a large placeholder
            if g_info.get("model") == "exponential":
                g_alpha = float("nan")
            if c_info.get("model") == "exponential":
                c_alpha = float("nan")

            greedy_alphas.append(g_alpha)
            canonical_alphas.append(c_alpha)

        ga = np.array(greedy_alphas)
        ca = np.array(canonical_alphas)

        # Plot bars (skip NaN)
        ga_finite = np.where(np.isfinite(ga), ga, 0)
        ca_finite = np.where(np.isfinite(ca), ca, 0)

        ax_d.bar(
            x_pos - width / 2,
            ga_finite,
            width,
            label="Greedy",
            color=FAMILY_COLORS.get("ba_m1", "#4477AA"),
            alpha=0.8,
        )
        ax_d.bar(
            x_pos + width / 2,
            ca_finite,
            width,
            label="Canonical",
            color=FAMILY_COLORS.get("complete", "#EE6677"),
            alpha=0.8,
        )

        # Mark exponential with "E" annotation
        for i, fam in enumerate(families_with_data):
            c_info = scaling_exponents.get("canonical", {}).get(fam, {})
            if c_info.get("model") == "exponential":
                ax_d.annotate(
                    "E", (x_pos[i] + width / 2, 0.5), fontsize=6, ha="center", color="red"
                )

        # Reference lines
        for ref_alpha in [2, 3, 4]:
            ax_d.axhline(ref_alpha, ls="--", color="grey", alpha=0.4, lw=0.7)
            ax_d.text(
                len(families_with_data) - 0.3,
                ref_alpha + 0.1,
                f"$\\alpha$={ref_alpha}",
                fontsize=6,
                color="grey",
            )

        display_labels = [family_display(f) for f in families_with_data]
        ax_d.set_xticks(x_pos)
        ax_d.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=6)
        ax_d.set_ylabel(r"Scaling exponent $\alpha$")
        ax_d.set_title(r"(d) Exponents ($T \sim n^\alpha$)")
        ax_d.legend(fontsize=7)

    fig.tight_layout()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    save_figure(fig, os.path.join(fig_dir, "encoding_main_figure"))
    plt.close(fig)
    logger.info("Saved main figure")


def _plot_scaling_by_family(
    greedy_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    scaling_exponents: dict,
    output_dir: str,
) -> None:
    """Plot individual family log-log scaling with regression lines."""
    apply_ieee_style()

    families = sorted(greedy_df["family"].unique())
    n_fam = len(families)
    ncols = min(4, n_fam)
    nrows = math.ceil(n_fam / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(PLOT_SETTINGS["figure_width_double"], 2.0 * nrows),
        squeeze=False,
    )

    for idx, family in enumerate(families):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        # Greedy data
        fdf = greedy_df[greedy_df["family"] == family]
        fdf_f = fdf[np.isfinite(fdf["greedy_time_s"])]
        if not fdf_f.empty:
            medians = fdf_f.groupby("n_nodes")["greedy_time_s"].median()
            ax.scatter(
                medians.index,
                medians.values,
                s=8,
                alpha=0.7,
                color=FAMILY_COLORS.get(family, "#333333"),
                label="Greedy",
            )

            # Regression
            g_info = scaling_exponents.get("greedy_single", {}).get(family, {})
            alpha = g_info.get("alpha", float("nan"))
            c_val = g_info.get("c", float("nan"))
            r2 = g_info.get("r_squared", float("nan"))
            if (
                np.isfinite(alpha)
                and np.isfinite(c_val)
                and g_info.get("model", "polynomial") == "polynomial"
            ):
                ns = np.linspace(medians.index.min(), medians.index.max(), 50)
                ax.plot(
                    ns,
                    c_val * ns**alpha,
                    "-",
                    alpha=0.5,
                    color=FAMILY_COLORS.get(family, "#333333"),
                    lw=0.8,
                )
                ax.text(
                    0.05,
                    0.95,
                    rf"$\alpha$={alpha:.2f}, $R^2$={r2:.2f}",
                    transform=ax.transAxes,
                    fontsize=6,
                    va="top",
                )

        # Canonical data
        if not canonical_df.empty:
            cdf = canonical_df[canonical_df["family"] == family]
            cdf_f = cdf[np.isfinite(cdf["canonical_time_s"])]
            if not cdf_f.empty:
                medians_c = cdf_f.groupby("n_nodes")["canonical_time_s"].median()
                ax.scatter(
                    medians_c.index,
                    medians_c.values,
                    s=8,
                    alpha=0.7,
                    marker="^",
                    color="red",
                    label="Canonical",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(family_display(family), fontsize=8)
        ax.legend(fontsize=5)

    # Hide empty subplots
    for idx in range(n_fam, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.supxlabel("n (nodes)", fontsize=8)
    fig.supylabel("Time (s)", fontsize=8)
    fig.tight_layout()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    save_figure(fig, os.path.join(fig_dir, "scaling_by_family"))
    plt.close(fig)
    logger.info("Saved scaling by family figure")


def _plot_string_length_scaling(
    greedy_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Plot string length vs n per family."""
    apply_ieee_style()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(PLOT_SETTINGS["figure_width_single"], PLOT_SETTINGS["figure_width_single"] * 0.75),
    )

    for family in greedy_df["family"].unique():
        fdf = greedy_df[greedy_df["family"] == family]
        fdf_valid = fdf[fdf["string_length"] > 0]
        if fdf_valid.empty:
            continue

        medians = fdf_valid.groupby("n_nodes")["string_length"].median()
        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")
        ax.plot(
            medians.index,
            medians.values,
            marker=marker,
            ms=4,
            color=color,
            alpha=0.7,
            label=family_display(family),
            lw=0.8,
        )

    ax.set_xlabel("n (nodes)")
    ax.set_ylabel("|w| (string length)")
    ax.set_title("String length scaling")
    ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    save_figure(fig, os.path.join(fig_dir, "string_length_scaling"))
    plt.close(fig)
    logger.info("Saved string length scaling figure")


# =============================================================================
# Table generation
# =============================================================================


def _generate_summary_table(
    scaling_exponents: dict,
    output_dir: str,
) -> None:
    """Generate LaTeX summary table of scaling exponents per family."""
    rows: list[dict] = []

    all_families_set = set()
    for method_key in ["greedy_single", "canonical"]:
        all_families_set.update(k for k in scaling_exponents.get(method_key, {}) if k != "overall")

    for family in sorted(all_families_set):
        cfg = FAMILY_CONFIGS.get(family, {})
        g_info = scaling_exponents.get("greedy_single", {}).get(family, {})
        c_info = scaling_exponents.get("canonical", {}).get(family, {})
        # Density class
        if family in ("complete", "gnp_05"):
            density_class = "Dense"
        elif family in ("gnp_03", "ba_m2"):
            density_class = "Moderate"
        else:
            density_class = "Sparse"

        # Greedy alpha (use alpha_poly fallback if model is exponential)
        g_model = g_info.get("model", "polynomial")
        g_alpha = g_info.get("alpha", float("nan"))
        if g_model == "exponential":
            g_alpha = g_info.get("alpha_poly", float("nan"))
        g_r2 = g_info.get("r_squared", float("nan"))
        # Canonical
        c_alpha = c_info.get("alpha", float("nan"))
        c_r2 = c_info.get("r_squared", float("nan"))
        c_model = c_info.get("model", "polynomial")

        c_alpha_str = (
            f"{c_alpha:.2f}"
            if c_model == "polynomial" and np.isfinite(c_alpha)
            else f"exp (b={c_info.get('base', 0):.2f})"
            if c_model == "exponential"
            else "---"
        )

        rows.append(
            {
                "Family": family_display(family),
                "n range": f"{cfg.get('n_min', '?')}-{cfg.get('n_max', '?')}",
                "a greedy": f"{g_alpha:.2f}" if np.isfinite(g_alpha) else "---",
                "R2 greedy": f"{g_r2:.2f}" if np.isfinite(g_r2) else "---",
                "a canonical": c_alpha_str,
                "R2 canonical": f"{c_r2:.2f}" if np.isfinite(c_r2) else "---",
                "Model": c_model if c_model != "insufficient_data" else "---",
                "Density": density_class,
            }
        )

    df = pd.DataFrame(rows)
    table_dir = os.path.join(output_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    save_latex_table(
        df,
        os.path.join(table_dir, "encoding_complexity.tex"),
        caption="Encoding complexity scaling exponents per graph family.",
        label="tab:encoding_complexity",
    )
    logger.info("Saved summary table")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Encoding complexity analysis for IsalGraph.",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root directory of eval pipeline output (for real validation).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=5,
        help="Instances per family/size for stochastic families.",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=25,
        help="Timing repetitions per measurement.",
    )
    parser.add_argument(
        "--max-n-greedy",
        type=int,
        default=50,
        help="Max n for greedy timing.",
    )
    parser.add_argument(
        "--max-n-canonical",
        type=int,
        default=20,
        help="Max n for canonical timing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", action="store_true", help="Save raw CSVs.")
    parser.add_argument("--plot", action="store_true", help="Generate figures.")
    parser.add_argument("--table", action="store_true", help="Generate LaTeX table.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_raw = os.path.join(args.output_dir, "raw")
    out_stats = os.path.join(args.output_dir, "stats")
    os.makedirs(out_raw, exist_ok=True)
    os.makedirs(out_stats, exist_ok=True)

    # Save hardware info
    hw_info = get_hardware_info()
    hw_info["n_instances"] = args.n_instances
    hw_info["n_reps"] = args.n_reps
    hw_info["max_n_greedy"] = args.max_n_greedy
    hw_info["max_n_canonical"] = args.max_n_canonical
    hw_info["seed"] = args.seed
    with open(os.path.join(out_stats, "hardware_info.json"), "w") as f:
        json.dump(hw_info, f, indent=2)
    logger.info("Hardware: %s", hw_info)

    families = ALL_FAMILIES

    # --- Experiment 1: Greedy scaling ---
    logger.info("=" * 60)
    logger.info("Experiment 1: Greedy single-start scaling")
    logger.info("=" * 60)
    t0 = time.perf_counter()
    greedy_df = _experiment_greedy_scaling(
        families,
        args.max_n_greedy,
        args.n_instances,
        args.n_reps,
        args.seed,
    )
    logger.info("Experiment 1 done in %.1fs (%d rows)", time.perf_counter() - t0, len(greedy_df))
    if args.csv:
        greedy_df.to_csv(os.path.join(out_raw, "synthetic_greedy_times.csv"), index=False)

    # --- Experiment 2: Canonical scaling ---
    logger.info("=" * 60)
    logger.info("Experiment 2: Canonical scaling")
    logger.info("=" * 60)
    t0 = time.perf_counter()
    canonical_df = _experiment_canonical_scaling(
        families,
        args.max_n_canonical,
        args.n_instances,
        args.n_reps,
        args.seed,
    )
    logger.info("Experiment 2 done in %.1fs (%d rows)", time.perf_counter() - t0, len(canonical_df))
    if args.csv:
        canonical_df.to_csv(os.path.join(out_raw, "synthetic_canonical_times.csv"), index=False)

    # --- Experiment 3: Density dependence ---
    logger.info("=" * 60)
    logger.info("Experiment 3: Density dependence")
    logger.info("=" * 60)
    t0 = time.perf_counter()
    density_df = _experiment_density_dependence(args.n_reps, args.seed)
    logger.info("Experiment 3 done in %.1fs (%d rows)", time.perf_counter() - t0, len(density_df))
    if args.csv:
        density_df.to_csv(os.path.join(out_raw, "density_dependence.csv"), index=False)

    # --- Scaling analysis ---
    logger.info("=" * 60)
    logger.info("Computing scaling exponents")
    logger.info("=" * 60)
    scaling_exponents = _compute_all_scaling_exponents(greedy_df, canonical_df)
    with open(os.path.join(out_stats, "scaling_exponents.json"), "w") as f:
        json.dump(scaling_exponents, f, indent=2, default=str)
    logger.info("Scaling exponents: %s", json.dumps(scaling_exponents, indent=2, default=str))

    # --- Density analysis summary ---
    if not density_df.empty:
        density_summary = {}
        for p_val in density_df["p"].unique():
            pdf = density_df[density_df["p"] == p_val]
            density_summary[str(p_val)] = {
                "median_greedy_min_s": float(pdf["greedy_min_time_s"].median()),
                "median_canonical_s": float(pdf["canonical_time_s"].median())
                if np.isfinite(pdf["canonical_time_s"]).any()
                else float("nan"),
                "n_instances": len(pdf),
            }
        with open(os.path.join(out_stats, "density_analysis.json"), "w") as f:
            json.dump(density_summary, f, indent=2, default=str)

    # --- Experiment 4: Real validation ---
    logger.info("=" * 60)
    logger.info("Experiment 4: Real dataset validation")
    logger.info("=" * 60)
    real_df = _experiment_real_validation(
        args.data_root,
        scaling_exponents.get("greedy_single", {}),
    )
    if args.csv and not real_df.empty:
        real_df.to_csv(os.path.join(out_raw, "real_validation.csv"), index=False)

    # --- Figures ---
    if args.plot:
        logger.info("Generating figures...")
        _plot_main_figure(greedy_df, canonical_df, scaling_exponents, args.output_dir)
        _plot_scaling_by_family(greedy_df, canonical_df, scaling_exponents, args.output_dir)
        _plot_string_length_scaling(greedy_df, args.output_dir)

    # --- Table ---
    if args.table:
        logger.info("Generating table...")
        _generate_summary_table(scaling_exponents, args.output_dir)

    # --- Summary ---
    summary = {
        "n_greedy_rows": len(greedy_df),
        "n_canonical_rows": len(canonical_df),
        "n_density_rows": len(density_df),
        "n_real_rows": len(real_df),
        "scaling_exponents": scaling_exponents,
    }
    with open(os.path.join(out_stats, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("All done. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
