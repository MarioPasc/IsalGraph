"""Levenshtein alignment via DP backtrace with visual rendering.

Computes the optimal alignment between two IsalGraph strings and
renders it as a color-coded two-row character grid.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Paul Tol colors for alignment operations
_OP_COLORS: dict[str, str] = {
    "match": "#228833",  # green
    "substitute": "#EE6677",  # red
    "insert": "#4477AA",  # blue
    "delete": "#CCBB44",  # yellow
}

_OP_ALPHA: dict[str, float] = {
    "match": 0.3,
    "substitute": 0.7,
    "insert": 0.7,
    "delete": 0.7,
}


def levenshtein_alignment(
    s: str,
    t: str,
) -> list[tuple[str, str | None, str | None]]:
    """Compute Levenshtein alignment via DP with backtrace.

    Args:
        s: Source string.
        t: Target string.

    Returns:
        List of (operation, char_from_s, char_from_t) tuples.
        Operations: "match", "substitute", "insert", "delete".
    """
    n, m = len(s), len(t)
    # DP matrix
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])

    # Backtrace
    alignment: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s[i - 1] == t[j - 1] and dp[i, j] == dp[i - 1, j - 1]:
            alignment.append(("match", s[i - 1], t[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            alignment.append(("substitute", s[i - 1], t[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            alignment.append(("delete", s[i - 1], None))
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            alignment.append(("insert", None, t[j - 1]))
            j -= 1
        else:
            break

    alignment.reverse()
    return alignment


def draw_alignment(
    alignment: list[tuple[str, str | None, str | None]],
    ax: plt.Axes,
    *,
    s_label: str = "w\u2081",
    t_label: str = "w\u2082",
    cell_width: float = 0.4,
    cell_height: float = 0.35,
) -> None:
    """Render alignment as a color-coded two-row character grid.

    Args:
        alignment: Output from levenshtein_alignment().
        ax: Matplotlib axes.
        s_label: Label for source row.
        t_label: Label for target row.
        cell_width: Width of each character cell.
        cell_height: Height of each character cell.
    """
    n_cols = len(alignment)
    if n_cols == 0:
        ax.axis("off")
        return

    label_offset = 0.6
    total_width = label_offset + n_cols * cell_width

    for col_idx, (op, cs, ct) in enumerate(alignment):
        x = label_offset + col_idx * cell_width
        color = _OP_COLORS[op]
        alpha = _OP_ALPHA[op]

        # Source row (top)
        rect_s = mpatches.FancyBboxPatch(
            (x, cell_height + 0.05),
            cell_width - 0.02,
            cell_height - 0.02,
            boxstyle="round,pad=0.02",
            facecolor=color,
            alpha=alpha,
            edgecolor="0.3",
            linewidth=0.5,
        )
        ax.add_patch(rect_s)
        char_s = cs if cs is not None else "\u2013"
        ax.text(
            x + cell_width / 2,
            cell_height + cell_height / 2 + 0.04,
            char_s,
            ha="center",
            va="center",
            fontsize=8,
            fontfamily="monospace",
            fontweight="bold",
        )

        # Target row (bottom)
        rect_t = mpatches.FancyBboxPatch(
            (x, 0.0),
            cell_width - 0.02,
            cell_height - 0.02,
            boxstyle="round,pad=0.02",
            facecolor=color,
            alpha=alpha,
            edgecolor="0.3",
            linewidth=0.5,
        )
        ax.add_patch(rect_t)
        char_t = ct if ct is not None else "\u2013"
        ax.text(
            x + cell_width / 2,
            cell_height / 2 - 0.01,
            char_t,
            ha="center",
            va="center",
            fontsize=8,
            fontfamily="monospace",
            fontweight="bold",
        )

    # Row labels
    ax.text(0.0, cell_height + cell_height / 2 + 0.04, s_label, ha="left", va="center", fontsize=8)
    ax.text(0.0, cell_height / 2 - 0.01, t_label, ha="left", va="center", fontsize=8)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=_OP_COLORS[op], alpha=_OP_ALPHA[op], label=op.capitalize())
        for op in ["match", "substitute", "insert", "delete"]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=6, ncol=4, framealpha=0.8)

    ax.set_xlim(-0.1, total_width + 0.1)
    ax.set_ylim(-0.15, 2 * cell_height + 0.25)
    ax.set_aspect("equal")
    ax.axis("off")
