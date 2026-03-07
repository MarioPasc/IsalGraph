# Agent 5: Publication-Quality Visualization and Graphical Examples

**Priority**: Explicitly requested by the advisor — *"Interesa incluir ejemplos gráficos de ello"* (graphical examples showing that similar graphs produce similar strings).
**Dependencies**: Agent 0 (SETUP) must complete first. Ideally, Agent 1 (CORRELATION) runs first too so we can select the most illustrative examples using its results. However, this agent can proceed with Agent 0 outputs alone.
**Parallelizable with**: Can run after Agent 0; benefits from Agent 1 results if available.
**Estimated local time**: ~30 min.
**Estimated Picasso time**: ~15 min.

---

## 1. Scientific Context

### 1.1 Purpose

The advisor requires **visual evidence** that:

> "La topología del espacio de las cadenas IsalGraph es similar a la que hay en el espacio de los grafos. Es decir, hay que mostrar que grafos parecidos se representan mediante cadenas parecidas."

This agent produces figures that make this claim intuitively compelling to a reviewer who skims the paper. These are not statistical analyses (that's Agent 1) — these are **illustrative examples** that build visual intuition before the formal results.

### 1.2 Types of Visual Evidence

1. **Example pairs**: Side-by-side drawings of two similar graphs with their canonical strings, showing that small graph changes produce small string changes.
2. **Example pairs (dissimilar)**: Two structurally different graphs with very different canonical strings, showing discrimination.
3. **String alignment**: Levenshtein edit alignment between two canonical strings, visualized with colored character-level operations (insert, delete, substitute, match).
4. **Similarity heatmaps**: Side-by-side GED and Levenshtein distance matrices for a subset of graphs, showing similar structure.
5. **Nearest-neighbor examples**: For a query graph, show its k-nearest neighbors under GED and under Levenshtein — they should overlap.
6. **Class-level clustering**: For IAM Letter, show that same-class graphs (e.g., all "A"s) cluster in both GED and Levenshtein space.

---

## 2. Input Data

From Agent 0:

| File | Content |
|------|---------|
| `ged_matrices/{dataset}.npz` | GED matrix |
| `levenshtein_matrices/{dataset}.npz` | Levenshtein matrix |
| `canonical_strings/{dataset}.json` | Canonical strings |
| `graph_metadata/{dataset}.json` | Class labels, node/edge counts |

From Agent 1 (optional, enhances example selection):

| File | Content |
|------|---------|
| `raw/{dataset}_pair_data.csv` | All pairs with GED and Levenshtein |

Additionally, the agent needs the **original graphs** (NetworkX format) for drawing. These are obtained by:
- IAM Letter: Re-parsing GXL files (same loader as Agent 0).
- LINUX/ALKANE: Loading from PyG.

---

## 3. Output Specification

### 3.1 Directory Structure

```
benchmarks/eval_visualization/
    __init__.py
    eval_visualization.py        # Main orchestrator
    graph_drawing.py             # Graph drawing utilities
    string_alignment.py          # Levenshtein alignment visualization
    example_selector.py          # Intelligent example selection
    README.md

results/eval_visualization/
    figures/
        # Main paper figures (numbered for paper inclusion)
        fig_example_pairs.pdf              # 2-3 illustrative pairs
        fig_string_alignment.pdf           # Alignment visualization
        fig_heatmap_comparison.pdf         # Side-by-side heatmaps
        fig_nearest_neighbors.pdf          # k-NN overlap examples
        fig_class_clustering.pdf           # Class-level structure
        
        # Per-dataset supplementary
        heatmap_{dataset}.pdf
        knn_examples_{dataset}.pdf
    
    raw/
        selected_examples.json             # Which examples were chosen and why
```

### 3.2 Figure Specifications

All figures use **matplotlib** with publication-quality settings matching the project style:

```python
PUBLICATION_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,  # Use mathtext, not LaTeX (faster, more portable)
}
```

---

## 4. Implementation Plan

### 4.1 Module Structure

```
benchmarks/eval_visualization/
    __init__.py
    eval_visualization.py     # CLI orchestrator
    graph_drawing.py          # NetworkX graph drawing with consistent style
    string_alignment.py       # Levenshtein alignment + visualization
    example_selector.py       # Selects illustrative examples
    README.md
```

### 4.2 Graph Drawing (`graph_drawing.py`)

```python
"""Graph drawing utilities for publication figures.

Uses NetworkX's spring layout with fixed seed for reproducibility.
Consistent styling: nodes as filled circles, edges as lines,
no self-loops, no multi-edges.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def draw_graph(
    G: nx.Graph,
    ax: plt.Axes,
    title: str = "",
    canonical_string: str = "",
    highlight_edges: set | None = None,
    highlight_color: str = "#e74c3c",
    node_color: str = "#3498db",
    edge_color: str = "#2c3e50",
    node_size: int = 300,
    layout_seed: int = 42,
) -> None:
    """Draw a graph on a matplotlib axes with publication styling.
    
    Args:
        G: NetworkX graph.
        ax: Matplotlib axes to draw on.
        title: Title above the graph.
        canonical_string: IsalGraph canonical string (displayed below).
        highlight_edges: Set of (u,v) edges to highlight (e.g., edited edges).
        highlight_color: Color for highlighted edges.
        node_color: Default node color.
        edge_color: Default edge color.
        node_size: Node marker size.
        layout_seed: Seed for spring layout reproducibility.
    """
    pos = nx.spring_layout(G, seed=layout_seed, k=1.5/np.sqrt(max(G.number_of_nodes(), 1)))
    
    # Draw edges
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if highlight_edges and (u, v) in highlight_edges or (v, u) in highlight_edges:
            edge_colors.append(highlight_color)
            edge_widths.append(2.5)
        else:
            edge_colors.append(edge_color)
            edge_widths.append(1.0)
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, node_size=node_size, 
                           edgecolors="#2c3e50", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white", font_weight="bold")
    
    ax.set_title(title, fontsize=10, pad=8)
    if canonical_string:
        ax.text(0.5, -0.08, f"w* = {canonical_string}", transform=ax.transAxes,
                ha="center", fontsize=8, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#bdc3c7"))
    ax.set_axis_off()
```

### 4.3 String Alignment Visualization (`string_alignment.py`)

```python
"""Levenshtein alignment visualization.

Computes the optimal edit alignment (backtrace through the DP matrix)
and renders it as a colored character-by-character comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Operation colors
MATCH_COLOR = "#2ecc71"      # Green
SUBSTITUTE_COLOR = "#e74c3c"  # Red
INSERT_COLOR = "#3498db"      # Blue  
DELETE_COLOR = "#f39c12"      # Orange

def levenshtein_alignment(s: str, t: str) -> list[tuple[str, str | None, str | None]]:
    """Compute Levenshtein alignment via DP backtrace.
    
    Returns:
        List of (operation, char_from_s, char_from_t) where operation
        is one of "match", "substitute", "insert", "delete".
    """
    m, n = len(s), len(t)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i-1] == t[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    
    # Backtrace
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s[i-1] == t[j-1]:
            alignment.append(("match", s[i-1], t[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + 1:
            alignment.append(("substitute", s[i-1], t[j-1]))
            i -= 1; j -= 1
        elif j > 0 and dp[i, j] == dp[i, j-1] + 1:
            alignment.append(("insert", None, t[j-1]))
            j -= 1
        else:
            alignment.append(("delete", s[i-1], None))
            i -= 1
    
    alignment.reverse()
    return alignment


def draw_alignment(
    alignment: list[tuple[str, str | None, str | None]],
    ax: plt.Axes,
    s_label: str = "w*(G₁)",
    t_label: str = "w*(G₂)",
) -> None:
    """Draw colored alignment on matplotlib axes.
    
    Top row: characters from s (with gaps for insertions)
    Bottom row: characters from t (with gaps for deletions)
    Color-coded by operation type.
    """
    cell_w, cell_h = 0.6, 0.4
    gap = 0.15
    
    colors = {
        "match": MATCH_COLOR,
        "substitute": SUBSTITUTE_COLOR,
        "insert": INSERT_COLOR,
        "delete": DELETE_COLOR,
    }
    
    n_cols = len(alignment)
    
    # Labels
    ax.text(-0.8, cell_h + gap/2, s_label, ha="right", va="center", fontsize=8, fontweight="bold")
    ax.text(-0.8, -gap/2, t_label, ha="right", va="center", fontsize=8, fontweight="bold")
    
    for col, (op, cs, ct) in enumerate(alignment):
        x = col * cell_w
        color = colors[op]
        
        # Top cell (s character)
        rect_s = FancyBboxPatch((x, gap/2), cell_w - 0.05, cell_h,
                                boxstyle="round,pad=0.02", facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=1)
        ax.add_patch(rect_s)
        ax.text(x + cell_w/2 - 0.025, gap/2 + cell_h/2,
                cs if cs else "—", ha="center", va="center",
                fontsize=9, fontfamily="monospace", fontweight="bold")
        
        # Bottom cell (t character)
        rect_t = FancyBboxPatch((x, -cell_h - gap/2), cell_w - 0.05, cell_h,
                                boxstyle="round,pad=0.02", facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=1)
        ax.add_patch(rect_t)
        ax.text(x + cell_w/2 - 0.025, -gap/2 - cell_h/2,
                ct if ct else "—", ha="center", va="center",
                fontsize=9, fontfamily="monospace", fontweight="bold")
    
    # Legend
    legend_items = [
        mpatches.Patch(color=MATCH_COLOR, alpha=0.5, label="Match"),
        mpatches.Patch(color=SUBSTITUTE_COLOR, alpha=0.5, label="Substitute"),
        mpatches.Patch(color=INSERT_COLOR, alpha=0.5, label="Insert"),
        mpatches.Patch(color=DELETE_COLOR, alpha=0.5, label="Delete"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7, framealpha=0.8)
    
    ax.set_xlim(-1, n_cols * cell_w + 0.5)
    ax.set_ylim(-cell_h - gap, cell_h + gap + 0.3)
    ax.set_axis_off()
    ax.set_aspect("equal")
```

### 4.4 Example Selection (`example_selector.py`)

Intelligent selection of illustrative examples:

```python
"""Select the most illustrative graph pairs for visualization.

Strategy:
1. SIMILAR pair: Two graphs with small GED (1-2) and small Levenshtein (1-3).
   Ideally same class in IAM Letter.
2. DISSIMILAR pair: Two graphs with large GED and large Levenshtein.
   Different classes.
3. CONCORDANT pair: GED and Levenshtein agree on ranking.
4. DISCORDANT pair (optional): GED and Levenshtein disagree — a failure case.
   Include only if such cases exist and are interesting.
"""

def select_similar_pair(
    ged_matrix: np.ndarray,
    lev_matrix: np.ndarray,
    metadata: dict,
    target_ged: int = 1,
) -> tuple[int, int]:
    """Select a pair with small GED and small Levenshtein.
    
    Prefers same-class pairs (for IAM Letter).
    Among candidates, selects the pair with the most "visual" graphs
    (intermediate node count, not too trivial).
    """
    ...

def select_dissimilar_pair(
    ged_matrix: np.ndarray,
    lev_matrix: np.ndarray,
    metadata: dict,
) -> tuple[int, int]:
    """Select a pair with large GED and large Levenshtein.
    
    Prefers different-class pairs with moderate graph sizes.
    """
    ...

def select_knn_example(
    ged_matrix: np.ndarray,
    lev_matrix: np.ndarray,
    k: int = 5,
) -> int:
    """Select a query graph whose k-NN overlap is illustrative.
    
    Returns the graph index with the highest k-NN overlap (for a positive example)
    or the lowest overlap (for a failure case analysis).
    """
    ...
```

### 4.5 Figure Generation

#### Figure 1: Example Pairs (`fig_example_pairs.pdf`)

**Layout**: 3 rows × 2 columns + alignment below each row.

Each row shows one pair:
- Row 1: **Similar pair** (small GED, small Levenshtein) from IAM Letter LOW.
  Left: Graph G₁ with canonical string. Right: Graph G₂ with canonical string. Below: alignment.
  Caption: "GED = 1, d_L = 2"
  
- Row 2: **Dissimilar pair** (large GED, large Levenshtein) from IAM Letter LOW.
  Caption: "GED = 8, d_L = 12"

- Row 3: **Cross-domain example** from LINUX or ALKANE.

**Size**: Full page width (7.5"), variable height.

#### Figure 2: String Alignment Detail (`fig_string_alignment.pdf`)

Close-up of the alignment from the similar pair in Figure 1. Character-by-character, color-coded operations. This shows that the Levenshtein edit operations correspond to meaningful graph modifications.

#### Figure 3: Heatmap Comparison (`fig_heatmap_comparison.pdf`)

**Layout**: 1 row × 2 columns.

Left: GED distance matrix for a subset (e.g., 60 graphs from IAM Letter LOW, 4 per class × 15 classes). Right: Levenshtein distance matrix for the same subset. Same color scale, same row/column ordering (sorted by class label, then by GED from centroid within class).

Use `matplotlib.pyplot.imshow()` with a perceptually uniform colormap (`"viridis"` or `"cividis"`). Draw thin black lines between classes.

#### Figure 4: k-NN Overlap (`fig_nearest_neighbors.pdf`)

**Layout**: 2 rows.

Row 1: Query graph (center) with its 5-NN by GED (top) and 5-NN by Levenshtein (bottom). Shared neighbors highlighted.
Row 2: Same for a different query graph.

This directly shows that "looking up similar graphs" works equally well with Levenshtein.

#### Figure 5: Class Clustering (`fig_class_clustering.pdf`)

**Layout**: 1 row × 2 columns.

Left: 2D MDS of GED distances, colored by class label (IAM Letter). Right: 2D MDS of Levenshtein distances, same coloring. If clusters align, the string space preserves class structure.

**Note**: This overlaps with Agent 2's MDS figures, but the styling here is specifically for publication as a "graphical example" with selected class labels annotated.

---

## 5. CLI Interface

```bash
python -m benchmarks.eval_visualization.eval_visualization \
    --data-root data/eval \
    --output-dir results/eval_visualization \
    --iam-letter-path data/eval/datasets/iam_letter_raw \
    --seed 42

# Specific figures only
python -m benchmarks.eval_visualization.eval_visualization \
    --data-root data/eval \
    --output-dir /tmp/viz_test \
    --figures example_pairs,heatmaps \
    --datasets iam_letter_low
```

---

## 6. Local Testing Plan

### 6.1 Smoke Test

```python
def test_graph_drawing():
    # Draw a simple triangle
    G = nx.cycle_graph(3)
    fig, ax = plt.subplots()
    draw_graph(G, ax, title="Triangle", canonical_string="VNC")
    fig.savefig("/tmp/test_triangle.png")
    assert os.path.exists("/tmp/test_triangle.png")

def test_alignment():
    alignment = levenshtein_alignment("VVC", "VNVC")
    # Should produce match(V), insert(N), match(V), match(C)
    assert any(op == "insert" for op, _, _ in alignment)

def test_alignment_drawing():
    alignment = levenshtein_alignment("VVC", "VNVC")
    fig, ax = plt.subplots(figsize=(8, 2))
    draw_alignment(alignment, ax)
    fig.savefig("/tmp/test_alignment.png")
    assert os.path.exists("/tmp/test_alignment.png")
```

### 6.2 Integration Test

```bash
# Generate all figures for IAM Letter LOW
python -m benchmarks.eval_visualization.eval_visualization \
    --data-root data/eval \
    --output-dir /tmp/viz_test \
    --datasets iam_letter_low \
    --iam-letter-path data/eval/datasets/iam_letter_raw

# Check outputs
ls /tmp/viz_test/figures/
```

---

## 7. Design Principles

### 7.1 Color Scheme

Use a consistent color scheme throughout all figures:
- **Blue** (#3498db): Nodes, GED-related elements
- **Green** (#2ecc71): Matches, positive results
- **Red** (#e74c3c): Substitutions, highlighted differences
- **Orange** (#f39c12): Deletions, warnings
- **Purple** (#9b59b6): Levenshtein-related elements

### 7.2 Reproducibility

- All layouts use `seed=42` for `nx.spring_layout()`.
- All example selections are deterministic (documented in `selected_examples.json`).
- Export both PDF (for paper) and PNG (for presentations).

### 7.3 What NOT to Include

- No t-SNE or UMAP plots (methodologically incorrect for distance preservation, as discussed in the Experimental Evaluation Design, Section 4).
- No 3D visualizations (hard to read in print).
- No interactive plots (not suitable for paper submission).

---

## 8. Acceptance Criteria

1. ✅ `fig_example_pairs.pdf` exists with at least 2 illustrative pairs.
2. ✅ `fig_string_alignment.pdf` exists with colored character alignment.
3. ✅ `fig_heatmap_comparison.pdf` exists with side-by-side GED/Levenshtein heatmaps.
4. ✅ `fig_nearest_neighbors.pdf` exists with k-NN overlap example.
5. ✅ `fig_class_clustering.pdf` exists with 2D MDS colored by class.
6. ✅ All figures use consistent style (publication quality, serif fonts).
7. ✅ `selected_examples.json` documents which examples were chosen and why.
8. ✅ Figures render correctly (no clipping, readable text, correct colors).
9. ✅ Code passes `ruff check`.
