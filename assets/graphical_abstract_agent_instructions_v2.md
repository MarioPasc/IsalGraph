# Graphical Abstract Generation — Agent Instructions (v2)

## Context

We are preparing a graphical abstract for the IsalGraph paper targeting **Pattern Recognition** (Elsevier). IsalGraph is a bijective encoding between graphs and instruction strings over the alphabet Σ = {N, n, P, p, V, v, C, c, W}, mediated by a Circular Doubly Linked List (CDLL) with two mobile pointers. Levenshtein distance between canonical strings serves as a fast proxy for Graph Edit Distance (GED).

### Elsevier Specifications
- **Dimensions**: minimum 531 × 1328 pixels (h × w), aspect ratio ~1:2.5
- **Physical size**: readable at 5 × 13 cm at 96 dpi
- **Preferred formats**: TIFF, EPS, PDF, or MS Office files
- **Reading direction**: left to right
- **Content**: single panel; minimal text; no "Graphical Abstract" heading in the image
- **NO generative AI imagery** (matplotlib/programmatic figures are fine)

### Advisor's directive (Ezequiel López-Rubio)
> "Para el graphical abstract puedes poner una cadena IsalGraph, a continuación una flecha bidireccional, y después el grafo correspondiente; también puedes mostrar un panel con la mejora respecto al GED."

Translation: Show an IsalGraph string → bidirectional arrow → corresponding graph; also show a panel with the improvement compared to GED.

### Results priority (from strongest to softest)
1. **Message length**: IsalGraph encoding is more compact than GED constructive encoding (β = 0.537, R² = 0.947, Wins = 99.6%)
2. **Computational speedup**: Orders-of-magnitude faster (365× to 25,100×)
3. **GED correlation**: Levenshtein distance approximates GED (ρ = 0.691, β = 0.80)

---

## Design Layout (v2)

Two panels, left to right. Panel A shows the *method*, Panel B shows the *results* in priority order.

```
+--------- Panel A (~38%) ---------+--+------------ Panel B (~62%) ----------------+
|                                   |  |                                             |
|  [Colored instruction string w*]  |  |  B1: Message length (LARGEST)               |
|            ↕                      |  |  [Scatter: IsalGraph bits vs GED bits]      |
|     [CDLL ring (auxiliary)]       |  |  β=0.537, R²=0.947, Wins=99.6%             |
|       ↙ S2G    G2S ↘             |  +---------------------------------------------+
|                                   |  |  B2: Speedup         |  B3: GED proxy       |
|       [House graph G]             |  |  [Line plot           |  d_Lev ≈ d_GED      |
|                                   |  |   365× → 25,100×]    |  ρ=0.691, β=0.80    |
+-----------------------------------+--+---------------------------------------------+
```

---

## Output Structure

All outputs go in `paper_figures/graphical_abstract/`:

```
paper_figures/graphical_abstract/
├── panel_a_encoding.pdf       # Left panel (string ↔ CDLL ↔ graph)
├── panel_a_encoding.svg
├── panel_a_encoding.png
├── panel_b_results.pdf        # Right panel (message length + speedup + correlation)
├── panel_b_results.svg
├── panel_b_results.png
├── graphical_abstract.pdf     # Full composite
├── graphical_abstract.svg
├── graphical_abstract.png
└── graphical_abstract.tiff    # Elsevier preferred format
```

---

## Script 1: `panel_a_encoding.py`

### Location
`benchmarks/real_data/eval_visualizations/graphical_abstract/panel_a_encoding.py`

### Purpose
Generate the left panel showing the IsalGraph bijective encoding concept with the CDLL as the mediating auxiliary structure.

### Conceptual Layout

The panel uses a **triangular arrangement** (top → center → bottom):

```
         [Colored instruction string w*]
                    ↕
           [CDLL ring (auxiliary)]
          ↙ S2G           G2S ↘
    [House graph G]
```

The key visual message: the string and graph are connected through the CDLL, which serves as the shared state machine for both S2G and G2S conversions.

### Implementation Details

1. **Use the house graph** (same example as `fig_algorithm_overview`, the paper's running example):
   ```python
   import networkx as nx
   G = nx.house_graph()  # 5 nodes, 6 edges
   ```

2. **Get the G2S string** by running G2S from node 0:
   ```python
   from isalgraph.adapters.networkx_adapter import NetworkXAdapter
   from isalgraph.core.graph_to_string import GraphToString

   adapter = NetworkXAdapter()
   sg = adapter.from_external(G, directed=False)
   g2s = GraphToString(sg)
   w, _ = g2s.run(initial_node=0)
   # Expected: w = "VVpvpvPCnC" (10 characters)
   ```

3. **Run S2G to get the final CDLL state** (needed to draw the ring):
   ```python
   from isalgraph.core.string_to_graph import StringToGraph

   s2g = StringToGraph(w, False)
   result_graph, trace = s2g.run(trace=True)
   # trace[-1] gives: (SparseGraph, CDLL, primary_ptr, secondary_ptr, full_string)
   final_graph, final_cdll, final_pri, final_sec, _ = trace[-1]
   ```

4. **Figure layout** — Three vertical zones with matplotlib:
   ```python
   fig = plt.figure(figsize=(2.0, 1.77))  # ~38% of 4.5" width, full height
   # Use manual axes placement or GridSpec with 3 rows:
   # Row 0 (top, ~20%): Instruction string (horizontal colored boxes)
   # Row 1 (center, ~40%): CDLL ring + bidirectional arrows
   # Row 2 (bottom, ~40%): House graph drawing
   ```

5. **Top zone — Instruction string**:
   - Draw using the horizontal heatmap approach from `_render_instruction_heatmap_horizontal()` in `benchmarks/real_data/eval_visualizations/illustrative/algorithm_figures.py`.
   - Each character is a rounded rectangle filled with its color from `INSTRUCTION_COLORS` (see `benchmarks/plotting_styles.py`), with the character rendered in white bold monospace.
   - Colors reference:
     ```python
     INSTRUCTION_COLORS = {
         "N": "#4477AA",  "P": "#004488",  "n": "#66CCEE",  "p": "#88CCEE",
         "V": "#228833",  "v": "#117733",  "C": "#EE6677",  "c": "#CC6677",
         "W": "#BBBBBB",
     }
     ```
   - Label: small "w*" annotation to the left of the string.

6. **Center zone — Simplified CDLL ring**:
   - Draw a **simplified** version of the CDLL ring from `benchmarks/real_data/eval_visualizations/cdll_drawing.py`. Specifically:
     - A dashed circle (the ring backbone)
     - 5 small filled circles (indigo, #332288) at evenly-spaced positions on the ring, with node payload numbers (0–4) in white
     - Two colored labels indicating the pointer positions: "π" in red (#EE6677) near the primary pointer node, "σ" in blue (#4477AA) near the secondary pointer node
     - **Do NOT draw the full pointer arrows** — too cluttered at this scale. Just the colored labels are enough.
   - Below the ring: label "CDLL (auxiliary)" in italic, smaller font.
   - Use `_extract_cdll_order()` from `algorithm_figures.py` to get the correct node ordering from the final CDLL state.

7. **Bidirectional arrows connecting the three elements**:
   - **String ↔ CDLL**: A single vertical bidirectional arrow between the string row and the CDLL ring.
   - **CDLL ↔ Graph**: Two diagonal bidirectional arrows from the CDLL ring down to the graph area, labeled "S2G" (left arrow) and "G2S" (right arrow) in small gray text.
   - Arrow style: Use `matplotlib.patches.FancyArrowPatch` with `arrowstyle='<->'`, color `"0.5"`, linewidth 1.0.

8. **Bottom zone — House graph**:
   - Draw using `draw_graph()` from `benchmarks/real_data/eval_visualizations/graph_drawing.py`.
   - Use `nx.spring_layout(G, seed=42)` for deterministic layout (same as the paper figures).
   - Node color: #332288 (Paul Tol muted indigo, the project's default).
   - Node size: Smaller than paper figures since space is limited (~120–150 pt²).
   - Label: small "G" annotation.

9. **Style requirements**:
   - Call `apply_ieee_style()` from `benchmarks/plotting_styles.py`
   - Save with `save_figure(fig, path, formats=("pdf", "svg", "png"))`
   - White background, tight bbox
   - All text readable at the target print size (5 × 5 cm for this panel)

### Testing
```python
# Verify round-trip: S2G(w) should produce a graph isomorphic to G
from isalgraph.core.string_to_graph import StringToGraph
s2g = StringToGraph(w, False)
result, _ = s2g.run()
adapter = NetworkXAdapter()
G_result = adapter.to_external(result)
assert nx.is_isomorphic(G, G_result), "Round-trip failed!"
```

---

## Script 2: `panel_b_results.py`

### Location
`benchmarks/real_data/eval_visualizations/graphical_abstract/panel_b_results.py`

### Purpose
Generate the right panel showing the three main results in priority order.

### Layout

The panel has a **2-row structure**:

```
+------------------------------------------------+
|  B1: Message length comparison (LARGEST)       |
|  [Scatter plot + statistics box]               |
|                            ~55% of height      |
+------------------------+-----------------------+
|  B2: Speedup           |  B3: GED proxy        |
|  [Simplified line      |  [Formula block:       |
|   chart with shading]  |   d_Lev ≈ d_GED       |
|       ~45% height      |   ρ, β values]         |
+------------------------+-----------------------+
     ~55% width                ~45% width
```

### Implementation Details

#### B1: Message Length Comparison (Top, largest sub-panel)

This is the most important result. Shows that IsalGraph encoding is consistently more compact than GED constructive encoding.

1. **Data source**: Load from `data/eval/eval_message_length/raw/` CSV files. The relevant columns are `isalgraph_bits_uniform` and `ged_bits_standard` (or similar — check the actual column names in the message length CSVs). If the raw data isn't easily loadable, use the summary from `fig_message_length_scatter.png` (Image 7/8).

   Check these paths:
   - `data/eval/eval_message_length/raw/message_lengths_*.csv`
   - The loading function `_load_message_length_csvs()` in `benchmarks/real_data/eval_visualizations/fig_message_length.py`

2. **Plot**: 2D density heatmap (GED bits on x-axis, IsalGraph bits on y-axis), matching the style of `fig_message_length_scatter.png` (Image 7) but simplified:
   - Use `ax.hist2d()` or `ax.hexbin()` with a viridis-like colormap
   - Draw the **parity line** (y = x) as a dashed gray diagonal — points below this line mean IsalGraph wins
   - Draw the **OLS regression line** in red (#EE6677), slope β ≈ 0.537
   - Statistics annotation box (upper-left): `β = 0.537`, `R² = 0.947`, `Wins: 99.6%` — matching the style from Image 7/8

3. **Fallback** (if raw data is not available): Create a schematic version:
   - Draw the parity diagonal (dashed)
   - Draw the regression line (solid red, slope ~0.54)
   - Add a shaded region between the two lines labeled "IsalGraph more compact"
   - Add the statistics box

4. **Axes labels**: "GED bits" (x), "IsalGraph bits" (y). Keep fonts readable.

#### B2: Computational Speedup (Bottom-left)

1. **Data source**: Load from `data/eval/eval_computational/` (the same source as `composite_method_tradeoff_v2.png`, Image 1a). Check:
   - `data/eval/eval_computational/stats/` for summary JSONs
   - `data/eval/eval_computational/raw/` for timing CSVs
   - The loading code in `benchmarks/eval_visualizations/composite_method_tradeoff.py`

2. **Plot**: Simplified line chart (log scale y-axis):
   - **Show only Greedy-rnd(v₀)** — the fastest method — as a single green line with triangle markers (#228833)
   - Fill the area between the line and y=1 (breakeven) with light green (alpha=0.12)
   - Breakeven dashed line at y=1
   - Annotate extreme values: "365×" at n=3, "25,100×" at n=11
   - X-axis: "Number of nodes n" with values 3, 5, 7, 9, 11
   - Y-axis: "Speedup" (log scale)

3. **Fallback values** (if data files aren't loadable):
   ```python
   nodes = [3, 5, 7, 9, 11]
   # Approximate values from Image 1a for Greedy-rnd(v₀):
   greedy_rnd_speedup = [365, 40, 130, 1500, 25100]
   ```

4. **Simplifications vs. the full paper figure**:
   - Only one line (Greedy-rnd), not three — keeps it clean
   - No error bars
   - Larger markers and annotations for readability at small size

#### B3: GED Proxy Quality (Bottom-right)

This is presented as a **formula block**, not a plot (correlation is the weakest result and gets the least visual weight).

1. **Content** (rendered with LaTeX if `text.usetex=True`, or with mathtext):
   ```
   d_Lev(w*_A, w*_B) ≈ d_GED(G_A, G_B)
   ──────────────────────────────────────
   Spearman ρ = 0.691
   OLS slope β = 0.80
   ```

2. **Implementation**: Use `ax.text()` with LaTeX formatting:
   ```python
   ax.text(0.5, 0.65,
       r"$d_{\mathrm{Lev}}(w^*_A, w^*_B) \approx d_{\mathrm{GED}}(G_A, G_B)$",
       transform=ax.transAxes, ha="center", va="center",
       fontsize=9)
   ax.axhline(0.45, xmin=0.1, xmax=0.9, color="0.7", linewidth=0.5)
   ax.text(0.5, 0.30,
       r"Spearman $\rho = 0.691$" + "\n" + r"OLS slope $\beta = 0.80$",
       transform=ax.transAxes, ha="center", va="center",
       fontsize=8, color="0.4")
   ```

3. **Style**: Clean, centered text. No axes, no ticks. Light border or box around the text block. Possibly a small label "GED proxy quality" at the top.

### Overall Script Structure

```python
fig = plt.figure(figsize=(2.8, 1.77))  # ~62% of 4.5" width

# GridSpec: 2 rows, 2 columns
# Row 0 spans both columns (B1)
# Row 1: left=B2, right=B3
gs = GridSpec(2, 2, figure=fig,
             height_ratios=[1.1, 1.0],
             width_ratios=[1.1, 0.9],
             hspace=0.3, wspace=0.25)

ax_msg = fig.add_subplot(gs[0, :])     # B1: message length (full width)
ax_speed = fig.add_subplot(gs[1, 0])   # B2: speedup (left)
ax_corr = fig.add_subplot(gs[1, 1])    # B3: correlation formula (right)
```

### Testing
- Verify the scatter/heatmap matches the general pattern of Image 7
- Verify the speedup values are on the correct order of magnitude vs Image 1a
- Check all text is readable at 5 × 8 cm print size

---

## Script 3: `compose_graphical_abstract.py`

### Location
`benchmarks/real_data/eval_visualizations/graphical_abstract/compose_graphical_abstract.py`

### Purpose
Compose both panels into the final graphical abstract at Elsevier dimensions.

### Implementation Details

1. **Figure dimensions**:
   ```python
   # Elsevier minimum: 531 × 1328 px
   # At 300 DPI: 1.77" × 4.43"
   # Use slightly larger for quality margin:
   fig_width_inches = 4.5
   fig_height_inches = 1.8
   dpi = 300
   ```

2. **Approach A (recommended): Generate everything inline**

   Create a single matplotlib figure with two main axes areas:
   ```python
   fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
   gs = GridSpec(1, 2, figure=fig,
                width_ratios=[0.38, 0.62],
                wspace=0.06,
                left=0.02, right=0.98, top=0.96, bottom=0.04)
   ```

   Then call the drawing functions from panel_a and panel_b as sub-routines (not as separate figure-saving operations).

3. **Approach B (for Inkscape workflow): Generate panels separately**

   Save Panel A and Panel B as high-resolution PDFs/SVGs. Mario imports them into an Inkscape document at 1328 × 531 px and arranges them manually. This is the safest approach for pixel-perfect control.

   **Both approaches should be implemented.** The composite script generates the inline version. The individual panel scripts generate the separate versions.

4. **Vertical separator**: Draw a thin line (0.5pt, gray, 60% height) between the two panels.

5. **Save in all formats**:
   ```python
   from benchmarks.plotting_styles import save_figure

   save_figure(fig, output_path, formats=("pdf", "svg", "png"))
   # Also TIFF for Elsevier:
   fig.savefig(f"{output_path}.tiff", format="tiff", dpi=300,
               bbox_inches="tight", pad_inches=0.02)
   ```

6. **Verification**:
   ```python
   from PIL import Image
   img = Image.open(f"{output_path}.png")
   w, h = img.size
   assert w >= 1328, f"Width {w} < 1328 minimum"
   assert h >= 531, f"Height {h} < 531 minimum"
   print(f"Graphical abstract: {w} × {h} px — OK")
   ```

---

## Module Structure

Create `benchmarks/real_data/eval_visualizations/graphical_abstract/__init__.py`:
```python
"""Graphical abstract generation for the IsalGraph paper.

Generates publication-quality panels for the Elsevier graphical abstract:
  - Panel A: Bijective string ↔ CDLL ↔ graph encoding concept
  - Panel B: Three main results (message length, speedup, GED correlation)
  - Composite: Full graphical abstract at Elsevier dimensions (531×1328 px)

Results are presented in priority order:
  1. Message length compactness (strongest: β=0.537, R²=0.947, Wins=99.6%)
  2. Computational speedup (365× to 25,100× over exact GED)
  3. GED proxy quality (ρ=0.691, formula-only display)
"""
```

---

## CLI Entry Points

```bash
conda activate isalgraph

# Generate individual panels (for Inkscape workflow)
python -m benchmarks.real_data.eval_visualizations.graphical_abstract.panel_a_encoding \
    --output-dir paper_figures/graphical_abstract

python -m benchmarks.real_data.eval_visualizations.graphical_abstract.panel_b_results \
    --output-dir paper_figures/graphical_abstract \
    --data-dir data/eval

# Generate full composite
python -m benchmarks.real_data.eval_visualizations.graphical_abstract.compose_graphical_abstract \
    --output-dir paper_figures/graphical_abstract \
    --data-dir data/eval
```

---

## Style Requirements (applies to ALL scripts)

1. **Import `apply_ieee_style()` and call it at the start** of every script
2. **Use `save_figure()` from `benchmarks/plotting_styles.py`** for all saves (ensures 300 DPI, tight bbox)
3. **Add SVG to the format tuple**: `save_figure(fig, path, formats=("pdf", "svg", "png"))`
4. **Use the project's color constants**: `INSTRUCTION_COLORS`, `PAUL_TOL_BRIGHT`, `PAUL_TOL_MUTED`
5. **Use project drawing utilities**:
   - `render_colored_string()` from `plotting_styles.py` for instruction strings
   - `_render_instruction_heatmap_horizontal()` from `algorithm_figures.py` for horizontal colored boxes
   - `draw_graph()` from `graph_drawing.py` for graph visualization
   - `draw_cdll_ring()` from `cdll_drawing.py` for CDLL (or write a simplified version)
6. **Logging**: `logging.getLogger(__name__)`
7. **Type hints**: Full typing on all function signatures
8. **Docstrings**: Module-level and function-level

---

## Displayed Values — Provenance and Verification

Before hardcoding any value, verify against the latest pipeline output:

| Value | Meaning | Source | Verification |
|-------|---------|--------|-------------|
| β = 0.537 | OLS regression slope (IsalGraph bits ~ GED bits) | `fig_message_length_scatter` stats box | Check `data/eval/eval_message_length/stats/` |
| R² = 0.947 | Coefficient of determination | Same | Same |
| Wins = 99.6% | Fraction of graphs where IsalGraph bits < GED bits | Same | Same |
| 365× | Speedup at n=3, Greedy-rnd(v₀) | `composite_method_tradeoff_v2` | Check `data/eval/eval_computational/stats/` |
| 25,100× | Speedup at n=11, Greedy-rnd(v₀) | Same | Same |
| ρ = 0.691 | Spearman correlation, Canonical (Pruned) | `fig_aggregated_density_correlation` | Check `data/eval/eval_correlation/stats/` |
| β = 0.80 | OLS slope (Lev ~ GED) | Same | Same |

**IMPORTANT**: These values come from Image 2 and Image 7/8 as provided by Mario. The agent MUST verify them against the actual data files if available, and print warnings if discrepancies are found. If the data files give different values, use the data file values and report the difference.

---

## What Mario Does After Generation

1. Open individual panel SVGs in Inkscape
2. Create a new Inkscape document at 1328 × 531 px (w × h)
3. Import/paste the two panels side by side
4. Fine-tune spacing, add a thin vertical separator if needed
5. Verify all text renders correctly (especially LaTeX formulas if any font issues)
6. Export as TIFF at 300 DPI for Elsevier submission
7. Verify final pixel dimensions ≥ 531 × 1328

This is why generating clean, self-contained SVGs is critical. Embedded fonts in SVG should use the same serif family as the paper (Times New Roman / STIX).
