"""Dual-mode LaTeX table generation (color + grayscale).

Generates paired .tex files with booktabs formatting and
\\best{}/\\worst{} highlighting for publication tables.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Significance formatting
# =============================================================================


def format_significance(p: float) -> str:
    """Map p-value to significance marker.

    Args:
        p: p-value.

    Returns:
        Significance string: ***, **, *, or n.s.
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# =============================================================================
# Dual table generation
# =============================================================================


def generate_dual_table(
    df: pd.DataFrame,
    output_dir: str,
    basename: str,
    caption: str,
    label: str,
    highlight_cols: set[str] | None = None,
    minimize_cols: set[str] | None = None,
    note: str | None = None,
) -> tuple[Path, Path]:
    """Generate paired color and grayscale LaTeX tables.

    Args:
        df: DataFrame with table data.
        output_dir: Output directory.
        basename: Base filename (without extension).
        caption: LaTeX table caption.
        label: LaTeX table label.
        highlight_cols: Columns to apply best/worst highlighting.
        minimize_cols: Columns where lower is better (default: higher is better).
        note: Optional table footnote.

    Returns:
        Tuple of (color_path, gray_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    highlight_cols = highlight_cols or set()
    minimize_cols = minimize_cols or set()

    color_path = Path(output_dir) / f"{basename}_color.tex"
    gray_path = Path(output_dir) / f"{basename}_gray.tex"

    # Build highlighted DataFrame
    df_color = _highlight_best_worst(df, highlight_cols, minimize_cols, mode="color")
    df_gray = _highlight_best_worst(df, highlight_cols, minimize_cols, mode="gray")

    for out_path, highlighted_df in [(color_path, df_color), (gray_path, df_gray)]:
        latex = _render_booktabs(highlighted_df, caption, label, note)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(latex)

    logger.info("Tables saved: %s, %s", color_path, gray_path)
    return color_path, gray_path


def _highlight_best_worst(
    df: pd.DataFrame,
    highlight_cols: set[str],
    minimize_cols: set[str],
    mode: str,
) -> pd.DataFrame:
    """Apply best/worst markers to numeric columns.

    Args:
        df: Input DataFrame.
        highlight_cols: Columns to highlight.
        minimize_cols: Columns where lower is better.
        mode: "color" for \\best{}/\\worst{}, "gray" for \\textbf{}/\\textit{}.

    Returns:
        DataFrame with string-formatted highlighted values.
    """
    df_out = df.copy()

    for col in highlight_cols:
        if col not in df_out.columns:
            continue
        # Try to parse as numeric
        vals = pd.to_numeric(df_out[col], errors="coerce")
        if vals.isna().all():
            continue

        if col in minimize_cols:
            best_idx = vals.idxmin()
            worst_idx = vals.idxmax()
        else:
            best_idx = vals.idxmax()
            worst_idx = vals.idxmin()

        for idx in df_out.index:
            cell = str(df_out.at[idx, col])
            if idx == best_idx:
                if mode == "color":
                    df_out.at[idx, col] = f"\\best{{{cell}}}"
                else:
                    df_out.at[idx, col] = f"\\textbf{{{cell}}}"
            elif idx == worst_idx:
                if mode == "color":
                    df_out.at[idx, col] = f"\\worst{{{cell}}}"
                else:
                    df_out.at[idx, col] = f"\\textit{{{cell}}}"

    return df_out


def _render_booktabs(
    df: pd.DataFrame,
    caption: str,
    label: str,
    note: str | None,
) -> str:
    """Render DataFrame as booktabs LaTeX table."""
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1),
    )
    # Wrap in table environment with booktabs
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        latex.strip(),
    ]
    if note:
        lines.append(f"\\smallskip\\footnotesize {note}")
    lines.append("\\end{table}")
    return "\n".join(lines)
