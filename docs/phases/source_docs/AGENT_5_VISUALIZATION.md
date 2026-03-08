# Agent 5: Unified Paper Figure Generation

**Priority**: Critical — all publication figures, tables, and individual examples for the paper.
**Dependencies**: Agents 0–4 must complete first. This agent reads **all** result directories.
**Parallelizable with**: None. This is the final aggregation step.
**Estimated local time**: ~45 min.
**Estimated Picasso time**: ~20 min.

---

## 0. Design Philosophy

### 0.1 Scope

This agent replaces the previous Agent 5 (which focused only on illustrative graphical examples). The new scope is: **for every hypothesis or claim tested across Agents 1–4, generate a coherent set of publication-quality artifacts**. The output is a self-contained `paper_figures/` directory that can be directly included in the LaTeX manuscript.

### 0.2 The Three-Artifact Rule

For each hypothesis or experimental claim, the agent generates exactly **three artifacts**:

| Artifact | Purpose | Audience |
|----------|---------|----------|
| **(A) Population-level figure** | Aggregated results with statistical significance annotations, effect sizes, 95% CI, and cross-dataset comparisons. Answers: *"Does the hypothesis hold in general?"* | Reviewer evaluating the statistical evidence. |
| **(B) Individual-level figure** | One to five concrete graph examples that illustrate the hypothesis at the instance level. Answers: *"What does this look like for a specific graph?"* | Reader building visual intuition. |
| **(C) LaTeX table** | Complete numerical results with best/worst highlighting, significance markers (`***`, `**`, `*`, `n.s.`), and effect sizes. Two versions: color (using `\forestgreen` / `\darkred`) and grayscale (using `\textbf{}` / `\underline{}`). | Reviewer comparing exact numbers. |

### 0.3 Design Standards

**Target venue**: Q1 journal of theoretical computer science (e.g., *Information Sciences*, *Pattern Recognition*, *Artificial Intelligence*).

**Figure format**: IEEE double-column compliant.
- Single-column width: 3.39 inches (86 mm).
- Full-width: 7.0 inches (178 mm).
- Maximum height: 9.0 inches (229 mm).
- Font: serif (Times New Roman / DejaVu Serif), minimum 8 pt.
- Export: PDF (vector, for paper) + PNG (300 dpi, for presentations).

**Color palette**: Paul Tol colorblind-safe schemes (already standardized in `benchmarks/plotting_styles.py`).

**Statistical annotation conventions**:
- `***` : p < 0.001
- `**` : p < 0.01
- `*` : p < 0.05
- `n.s.` : p ≥ 0.05
- Effect size interpretation (Cohen's d): negligible < 0.2, small 0.2–0.5, medium 0.5–0.8, large > 0.8.
- All p-values are permutation-based (Mantel, Procrustes) or bootstrap-derived, never parametric on non-independent pairs.

**Reproducibility**:
- All graph layouts use `seed=42` for `nx.spring_layout()`.
- All example selections are deterministic and documented in `selected_examples.json`.
- All random seeds are fixed and reported.

### 0.4 What NOT to Include

- **No t-SNE or UMAP** (methodologically incorrect for distance preservation; see Experimental Evaluation Design, Section 4).
- **No 3D visualizations** (not readable in print).
- **No interactive plots** (not suitable for static paper submission).
- **No spaghetti plots** with >6 overlapping lines without panel separation.

---

## 1. Input Data

This agent reads outputs from **all** previous agents.

### 1.1 From Agent 0 (Setup)

| Path | Content |
|------|---------|
| `data/eval/ged_matrices/{dataset}.npz` | Pairwise exact GED matrix |
| `data/eval/levenshtein_matrices/{dataset}_{method}.npz` | Pairwise Levenshtein matrix (method ∈ {exhaustive, greedy}) |
| `data/eval/canonical_strings/{dataset}_{method}.json` | Canonical strings with metadata |
| `data/eval/graph_metadata/{dataset}.json` | Node/edge counts, class labels, graph IDs |
| `data/eval/method_comparison/{dataset}_comparison.json` | Exhaustive vs greedy per-graph comparison |
| `data/eval/filtering_report.json` | Dataset filtering statistics |

### 1.2 From Agent 1 (Correlation)

| Path | Content |
|------|---------|
| `results/eval_correlation/stats/{dataset}_correlation_stats.json` | Per-dataset Spearman ρ, Pearson r, Kendall τ, CCC, P@k, Mantel p, bootstrap CI |
| `results/eval_correlation/stats/cross_distortion_analysis.json` | H₂ Jonckheere-Terpstra test, H₃ density-stratified analysis |
| `results/eval_correlation/stats/summary_table.json` | Cross-dataset summary |
| `results/eval_correlation/raw/{dataset}_pair_data.csv` | All (i, j, GED, Lev) pairs |

### 1.3 From Agent 2 (Embedding)

| Path | Content |
|------|---------|
| `results/eval_embedding/stats/{dataset}_{method}_embedding_stats.json` | Stress-1, NEV ratio, Procrustes m², Shepard R² |
| `results/eval_embedding/raw/{dataset}_cmds_eigenvalues.npz` | Full eigenspectra |
| `results/eval_embedding/raw/{dataset}_{method}_smacof_coords_{dim}d.npz` | MDS coordinates |
| `results/eval_embedding/stats/summary.json` | Cross-dataset embedding summary |

### 1.4 From Agent 3 (Computational)

| Path | Content |
|------|---------|
| `results/eval_computational/stats/{dataset}_timing_stats.json` | Per-pair timing summaries |
| `results/eval_computational/stats/crossover_analysis.json` | Crossover point results |
| `results/eval_computational/stats/scaling_regression.json` | Log-log regression fits |
| `results/eval_computational/raw/{dataset}_encoding_times.csv` | Per-graph encoding times |
| `results/eval_computational/raw/{dataset}_levenshtein_times.csv` | Per-pair Levenshtein times |
| `results/eval_computational/raw/{dataset}_ged_times.csv` | Per-pair GED times |
| `results/eval_computational/raw/{dataset}_amortized_comparison.csv` | Pipeline totals |

### 1.5 From Agent 4 (Encoding)

| Path | Content |
|------|---------|
| `results/eval_encoding/stats/scaling_exponents.json` | Per-family α for greedy, greedy-min, canonical |
| `results/eval_encoding/stats/density_analysis.json` | Density dependence |
| `results/eval_encoding/raw/synthetic_greedy_times.csv` | Timing measurements |
| `results/eval_encoding/raw/synthetic_canonical_times.csv` | Timing measurements |
| `results/eval_encoding/raw/real_timing_validation.csv` | Real vs predicted times |

### 1.6 Original Graphs (for individual-level drawing)

The agent reloads original graph structures for drawing:
- IAM Letter: Re-parse GXL files via `benchmarks.eval_setup.iam_letter_loader`.
- LINUX / AIDS: Load via `benchmarks.eval_setup.graphedx_loader`.

---

## 2. Output Specification

### 2.1 Directory Structure

```
paper_figures/
    # ── Agent 1: Correlation ──────────────────────────────────
    H1_1_global_correlation/
        population_correlation_overview.pdf        # (A) Multi-panel: scatter + bar + CI
        individual_concordant_pairs.pdf            # (B) 2–3 graph pairs with strings
        table_correlation_summary_color.tex        # (C) Color table
        table_correlation_summary_gray.tex         # (C) Grayscale table

    H1_2_monotone_degradation/
        population_distortion_trend.pdf            # (A) ρ vs distortion with JT p-value
        individual_distortion_examples.pdf         # (B) Same-class pair at LOW/MED/HIGH
        table_monotone_degradation_color.tex       # (C)
        table_monotone_degradation_gray.tex        # (C)

    H1_3_density_stratification/
        population_density_vs_rho.pdf              # (A) ρ binned by edge density
        individual_sparse_vs_dense.pdf             # (B) Sparse tree pair vs dense pair
        table_density_stratification_color.tex     # (C)
        table_density_stratification_gray.tex      # (C)

    H1_4_class_discrimination/
        population_within_vs_between.pdf           # (A) Violin/box: within vs between class
        individual_class_pair_examples.pdf         # (B) Same-class vs different-class pair
        table_class_discrimination_color.tex       # (C)
        table_class_discrimination_gray.tex        # (C)

    H1_5_exhaustive_vs_greedy/
        population_method_comparison.pdf           # (A) Grouped bar: ρ per method ± CI
        individual_method_divergence.pdf           # (B) Graph where strings differ
        table_method_comparison_color.tex          # (C)
        table_method_comparison_gray.tex           # (C)

    # ── Agent 2: Embedding ────────────────────────────────────
    H2_1_low_distortion/
        population_stress_by_dimension.pdf         # (A) Stress-1 vs d, all datasets
        individual_mds_scatter.pdf                 # (B) 2D MDS colored by class (IAM LOW)
        table_embedding_quality_color.tex          # (C)
        table_embedding_quality_gray.tex           # (C)

    H2_2_geometric_agreement/
        population_procrustes_comparison.pdf       # (A) m² bar chart + significance
        individual_procrustes_overlay.pdf          # (B) Superimposed MDS configs
        table_procrustes_color.tex                 # (C)
        table_procrustes_gray.tex                  # (C)

    H2_3_shepard_fidelity/
        population_shepard_r2_summary.pdf          # (A) R² per dataset × dimension
        individual_shepard_diagrams.pdf            # (B) Shepard plot: best vs worst dataset
        table_shepard_color.tex                    # (C)
        table_shepard_gray.tex                     # (C)

    # ── Agent 3: Computational ────────────────────────────────
    C3_1_speedup/
        population_speedup_bar.pdf                 # (A) Speedup factor per dataset
        individual_timing_trace.pdf                # (B) One graph pair: GED vs Lev timing
        table_speedup_color.tex                    # (C)
        table_speedup_gray.tex                     # (C)

    C3_2_crossover/
        population_crossover_curves.pdf            # (A) Time vs n: GED, Lev(exh), Lev(greedy)
        individual_crossover_example.pdf           # (B) Mark the crossing point on log-log
        table_crossover_color.tex                  # (C)
        table_crossover_gray.tex                   # (C)

    C3_3_amortized_pipeline/
        population_amortized_comparison.pdf        # (A) Stacked bar: encode + compare
        individual_pipeline_breakdown.pdf          # (B) One dataset: pie chart of time budget
        table_amortized_color.tex                  # (C)
        table_amortized_gray.tex                   # (C)

    # ── Agent 4: Encoding ─────────────────────────────────────
    C4_1_scaling_exponents/
        population_scaling_loglog.pdf              # (A) Log-log: time vs N per family
        individual_family_examples.pdf             # (B) Path, star, complete — graphs + times
        table_scaling_exponents_color.tex          # (C)
        table_scaling_exponents_gray.tex           # (C)

    C4_2_density_dependence/
        population_density_heatmap.pdf             # (A) Heatmap: log(time) vs (N, density)
        individual_density_pair.pdf                # (B) G(10,0.1) vs G(10,0.9) + their times
        table_density_dependence_color.tex         # (C)
        table_density_dependence_gray.tex          # (C)

    C4_3_greedy_vs_canonical/
        population_overhead_ratio.pdf              # (A) Canonical/greedy ratio per family
        individual_overhead_example.pdf            # (B) One graph: greedy trace vs canonical
        table_overhead_color.tex                   # (C)
        table_overhead_gray.tex                    # (C)

    # ── Illustrative (visual intuition, no standalone hypothesis) ──
    illustrative/
        fig_example_pairs.pdf                      # Similar + dissimilar graph pairs
        fig_string_alignment.pdf                   # Character-level Levenshtein alignment
        fig_heatmap_comparison.pdf                 # GED vs Lev(exh) vs Lev(greedy) heatmaps
        fig_nearest_neighbors.pdf                  # k-NN overlap visualization
        fig_class_clustering.pdf                   # 2D MDS colored by class

    # ── Metadata ──
    selected_examples.json                         # Deterministic example selection log
    figure_manifest.json                           # Every figure: path, hypothesis, caption
```

### 2.2 File Naming Convention

- Population figures: `population_{descriptive_name}.pdf`
- Individual figures: `individual_{descriptive_name}.pdf`
- Tables: `table_{descriptive_name}_{color|gray}.tex`
- All figures also exported as `.png` at 300 dpi alongside the PDF.

---

## 3. Hypothesis-by-Hypothesis Specification

### Legend for all tables

All LaTeX tables use the following conventions:

```latex
% Color version preamble
\usepackage{xcolor}
\definecolor{forestgreen}{RGB}{34,139,34}
\definecolor{darkred}{RGB}{139,0,0}
\newcommand{\best}[1]{\textcolor{forestgreen}{\textbf{#1}}}
\newcommand{\worst}[1]{\textcolor{darkred}{\underline{#1}}}

% Grayscale version preamble
\newcommand{\best}[1]{\textbf{#1}}
\newcommand{\worst}[1]{\underline{#1}}

% Significance markers (common to both)
% *** p < 0.001, ** p < 0.01, * p < 0.05, n.s. p >= 0.05
```

---

### 3.1 H1.1 — Global Positive Correlation (Levenshtein ↔ GED)

**Claim**: For each dataset, Spearman ρ(Levenshtein, GED) > 0 with Mantel permutation p < 0.001.

**Source**: Agent 1 — `{dataset}_correlation_stats.json`.

#### (A) Population Figure: `population_correlation_overview.pdf`

**Layout**: 2 × 3 panels, full width (7.0").

| Panel | Content |
|-------|---------|
| (a) | Hexbin scatter: Lev vs GED for IAM Letter LOW. OLS line. Annotate ρ, r, n_pairs. |
| (b) | Hexbin scatter: Lev vs GED for LINUX. Same format. |
| (c) | Hexbin scatter: Lev vs GED for AIDS. Same format. |
| (d) | Horizontal bar chart: Spearman ρ per dataset (all 5) with BCa 95% CI error bars. Vertical dashed line at ρ = 0. Color by dataset domain. Significance stars annotated to the right of each bar. |
| (e) | Horizontal bar chart: Pearson r per dataset, same format. Side-by-side comparison with ρ. |
| (f) | Grouped bar: Precision@k for k ∈ {5, 10, 20}, grouped by dataset. Include whisker for CI. |

**Design rationale**: The scatter panels give the reviewer an immediate visual impression of the correlation strength. The bar charts provide the formal quantification. P@k demonstrates practical retrieval utility.

**Statistical annotations on panels (a)–(c)**:
- Top-right: `ρ = X.XX [CI_lo, CI_hi]***`
- Below: `n = Y pairs`
- OLS regression line in red, 45° identity line in gray dashed.

#### (B) Individual Figure: `individual_concordant_pairs.pdf`

**Layout**: 3 rows × 3 columns, full width.

| Row | Left col | Center col | Right col |
|-----|----------|------------|-----------|
| Row 1: Concordant (similar) | Graph G₁ drawn | Graph G₂ drawn | Levenshtein alignment below both |
| Row 2: Concordant (dissimilar) | Graph G₃ drawn | Graph G₄ drawn | Levenshtein alignment below both |
| Row 3: Best failure case | Graph G₅ drawn | Graph G₆ drawn | Levenshtein alignment below both |

**Example selection** (deterministic):
1. **Row 1 — Similar**: Select the pair (i, j) from IAM Letter LOW with GED = 1 and the smallest Levenshtein distance, preferring pairs with 5–8 nodes (visually interesting). Same class preferred.
2. **Row 2 — Dissimilar**: Select the pair with GED ≥ 75th percentile and Levenshtein ≥ 75th percentile. Different class preferred.
3. **Row 3 — Discordant**: Select the pair with the largest |rank_GED(i,j) − rank_Lev(i,j)| — this is the pair where the two metrics disagree most. This is honest reporting of a failure case.

Each graph is drawn with `nx.spring_layout(seed=42)`, node labels, and the canonical string `w*` displayed below in monospace font.

Annotations per row:
- `GED = X, d_L = Y`
- Below: color-coded alignment strip showing match (green), substitute (red), insert (blue), delete (orange).

#### (C) LaTeX Table: `table_correlation_summary_{color|gray}.tex`

```
| Dataset       | N    | Pairs   | ρ [95% CI]           | r [95% CI]           | τ      | CCC    | P@10   | Mantel p |
|---------------|------|---------|----------------------|----------------------|--------|--------|--------|----------|
| IAM LOW       | XXX  | XXX     | \best{0.XX [a, b]}   | 0.XX [a, b]          | 0.XX   | 0.XX   | 0.XX   | ***      |
| IAM MED       | XXX  | XXX     | 0.XX [a, b]          | 0.XX [a, b]          | 0.XX   | 0.XX   | 0.XX   | ***      |
| IAM HIGH      | XXX  | XXX     | 0.XX [a, b]          | 0.XX [a, b]          | 0.XX   | 0.XX   | 0.XX   | ***      |
| LINUX         | XXX  | XXX     | 0.XX [a, b]          | 0.XX [a, b]          | 0.XX   | 0.XX   | 0.XX   | ***      |
| AIDS          | XXX  | XXX     | \worst{0.XX [a, b]}  | 0.XX [a, b]          | 0.XX   | 0.XX   | 0.XX   | **       |
```

`\best{}` marks the highest ρ across datasets; `\worst{}` marks the lowest. Significance column uses `***`/`**`/`*`/`n.s.`.

---

### 3.2 H1.2 — Monotone Degradation Across Distortion Levels

**Claim**: Spearman ρ decreases monotonically LOW → MED → HIGH. Tested via the Jonckheere-Terpstra ordered-alternative test.

**Source**: Agent 1 — `cross_distortion_analysis.json`.

#### (A) Population Figure: `population_distortion_trend.pdf`

**Layout**: Single-column (3.39"), 1 × 1 panel.

- X-axis: Distortion level (LOW, MED, HIGH) — ordinal, evenly spaced.
- Y-axis: Spearman ρ.
- Two lines: exhaustive (solid) and greedy (dashed), both with 95% CI band (shaded).
- Annotate: Jonckheere-Terpstra test statistic T_JT and p-value for each method.
- If both methods show the same monotone pattern, annotate: "Both methods confirm H₂."
- Reference dashed line at ρ = 0.

**Statistical rigor**: The confidence bands come from BCa bootstrap (10,000 resamples). The JT test is non-parametric and appropriate for ordered alternatives in independent samples (Jonckheere, 1954, *Biometrika* 41:133–145).

#### (B) Individual Figure: `individual_distortion_examples.pdf`

**Layout**: 3 columns × 2 rows, full width.

| | LOW | MED | HIGH |
|--|-----|-----|------|
| Row 1: same-class pair | G₁ᴸ, G₂ᴸ drawn | G₁ᴹ, G₂ᴹ drawn | G₁ᴴ, G₂ᴴ drawn |
| Row 2: alignment | Lev alignment strip | Lev alignment strip | Lev alignment strip |

Select the **same letter class** (e.g., "A") across all three distortion levels. Pick a pair with GED = 1 from each level (or the minimum available GED). Show that:
- At LOW: strings are nearly identical (few edit operations).
- At MED: more substitutions appear.
- At HIGH: the alignment shows substantial divergence.

Annotate each column: `ρ_dataset = X.XX`, `GED = Y`, `d_L = Z`.

#### (C) LaTeX Table: `table_monotone_degradation_{color|gray}.tex`

```
| Distortion | ρ_exh [95% CI]       | ρ_greedy [95% CI]    | Δρ (exh − prev) | JT test  |
|------------|----------------------|----------------------|-----------------|----------|
| LOW        | \best{0.XX [a, b]}   | \best{0.XX [a, b]}   | —               | —        |
| MED        | 0.XX [a, b]          | 0.XX [a, b]          | −0.XX           | —        |
| HIGH       | \worst{0.XX [a, b]}  | \worst{0.XX [a, b]}  | −0.XX           | —        |
| JT p-value |                      |                      |                 | ***      |
```

---

### 3.3 H1.3 — Density Stratification

**Claim**: Sparser graphs yield higher ρ (consistent with the theoretical expectation that tree-like structures are better served by string-based distances; cf. Akutsu et al., 2010).

**Source**: Agent 1 — `cross_distortion_analysis.json` (density-stratified sub-analysis).

#### (A) Population Figure: `population_density_vs_rho.pdf`

**Layout**: Single-column, 1 panel.

- Compute per-pair density as `(|E_i| + |E_j|) / (binom(|V_i|,2) + binom(|V_j|,2))` (or per-graph average density, binned).
- Bin pairs into 4–5 density quartiles.
- Box plot or violin plot of pairwise Levenshtein–GED rank agreement per density bin.
- Overlay: Spearman ρ computed within each density bin, connected by a line, with CI bands.
- Annotate: regression slope of ρ vs density, p-value from permutation.

#### (B) Individual Figure: `individual_sparse_vs_dense.pdf`

**Layout**: 2 rows × 2 columns, single-column width.

- Top row: A pair of sparse graphs (tree-like) from LINUX with low density. Show graphs + strings + alignment. Annotate GED and d_L.
- Bottom row: A pair of dense graphs (high density) from AIDS or IAM HIGH. Same format.
- Caption argues: "Sparse pair shows high agreement (GED = X, d_L = Y); dense pair shows degraded agreement (GED = X', d_L = Y')."

#### (C) LaTeX Table: `table_density_stratification_{color|gray}.tex`

```
| Density bin      | N pairs | ρ [95% CI]           | Mean |GED − d_L| | Concordance |
|------------------|---------|----------------------|--------------------|-------------|
| Q1 (sparsest)    | XXX     | \best{0.XX [a, b]}   | X.XX               | X.XX        |
| Q2               | XXX     | 0.XX [a, b]          | X.XX               | X.XX        |
| Q3               | XXX     | 0.XX [a, b]          | X.XX               | X.XX        |
| Q4 (densest)     | XXX     | \worst{0.XX [a, b]}  | X.XX               | X.XX        |
| Trend test p     |         |                      |                    | ***         |
```

---

### 3.4 H1.4 — Within-Class vs Between-Class Discrimination

**Claim**: For labeled datasets (IAM Letter), within-class Levenshtein distances are significantly smaller than between-class distances. Measured by Cohen's d effect size.

**Source**: Agent 1 — `{dataset}_correlation_stats.json` (H₄ sub-analysis).

#### (A) Population Figure: `population_within_vs_between.pdf`

**Layout**: 1 × 3 panels (one per IAM distortion level), single-column width stacked or full-width side-by-side.

Each panel:
- Split-violin plot or paired box plot: left = within-class Lev distances, right = between-class Lev distances.
- Annotate: Cohen's d, Mann-Whitney U p-value, and the number of pairs in each group.
- Horizontal bracket with significance stars.

**Why Cohen's d and not just p-value**: With hundreds of thousands of pairs, any difference will be "significant." Cohen's d quantifies the magnitude. This follows Demšar (2006), *JMLR* 7:1–30.

#### (B) Individual Figure: `individual_class_pair_examples.pdf`

**Layout**: 2 rows × 2 columns.

- Top row (within-class): Two "A" graphs from IAM Letter LOW. Show graphs + canonical strings. Low Levenshtein distance.
- Bottom row (between-class): One "A" graph, one "M" graph. Show graphs + canonical strings. High Levenshtein distance.
- Annotate distances on each row.

#### (C) LaTeX Table: `table_class_discrimination_{color|gray}.tex`

```
| Dataset  | d_L within (μ ± σ) | d_L between (μ ± σ) | Cohen's d         | U test p |
|----------|--------------------|---------------------|-------------------|----------|
| IAM LOW  | X.X ± X.X          | X.X ± X.X           | \best{X.XX}       | ***      |
| IAM MED  | X.X ± X.X          | X.X ± X.X           | X.XX              | ***      |
| IAM HIGH | X.X ± X.X          | X.X ± X.X           | \worst{X.XX}      | ***      |
```

---

### 3.5 H1.5 — Exhaustive vs Greedy Canonical Strings

**Claim**: The greedy-min approximation produces correlations statistically indistinguishable from the exhaustive canonical. Tested via paired bootstrap CI on Δρ = ρ_exhaustive − ρ_greedy.

**Source**: Agent 1 — per-method stats in `{dataset}_correlation_stats.json`, method comparison in `cross_distortion_analysis.json`.

#### (A) Population Figure: `population_method_comparison.pdf`

**Layout**: Full width, 1 × 2 panels.

- Panel (a): Grouped bar chart — for each dataset, two bars (exhaustive blue, greedy orange) showing Spearman ρ with 95% CI error bars. If CI overlap, they are statistically indistinguishable.
- Panel (b): Scatter of ρ_exhaustive vs ρ_greedy across all 5 datasets + identity line. Points near the identity line confirm equivalence. Annotate each point with the dataset name.

#### (B) Individual Figure: `individual_method_divergence.pdf`

**Layout**: 3 rows, full width.

| Row | Content |
|-----|---------|
| Row 1 | Graph G drawn (choose a graph where exhaustive ≠ greedy) |
| Row 2 | Two string blocks: `w*_exhaustive` (blue box) and `w'_greedy_min` (orange box) |
| Row 3 | Levenshtein alignment between the two strings, showing where they differ |

Select from `method_comparison/{dataset}_comparison.json` — pick a graph where `exhaustive ≠ greedy` and the length gap is representative (near the median gap among discordant cases).

Additionally, include a second example where `exhaustive = greedy` (the majority case) to show that agreement is typical.

#### (C) LaTeX Table: `table_method_comparison_{color|gray}.tex`

```
| Dataset  | ρ_exh [CI]          | ρ_greedy [CI]        | Δρ [CI]             | Agreement % |
|----------|---------------------|----------------------|---------------------|-------------|
| IAM LOW  | X.XX [a, b]         | X.XX [a, b]          | X.XXX [a, b] n.s.   | XX.X%       |
| ...      |                     |                      |                     |             |
```

`Agreement %` = fraction of graph pairs where the greedy and exhaustive Levenshtein distances are identical.

---

### 3.6 H2.1 — Low-Distortion MDS Embeddings

**Claim**: MDS embeddings of Levenshtein distance matrices achieve Stress-1 < 0.20 in 2D and < 0.10 in 5D (Kruskal, 1964, *Psychometrika* 29:1–27).

**Source**: Agent 2 — `{dataset}_{method}_embedding_stats.json`.

#### (A) Population Figure: `population_stress_by_dimension.pdf`

**Layout**: Single-column, 1 panel.

- X-axis: Embedding dimension d ∈ {2, 3, 5, 10}.
- Y-axis: Stress-1 (Kruskal formula).
- Lines: One per dataset (colored), solid = Levenshtein (exhaustive), dashed = GED reference.
- Horizontal reference lines at 0.20 (fair), 0.10 (good), 0.05 (excellent) in gray dotted.
- Annotate each threshold line with its Kruskal interpretation label.

#### (B) Individual Figure: `individual_mds_scatter.pdf`

**Layout**: 1 × 2 panels, full width.

- Left: 2D MDS of Levenshtein distances for IAM Letter LOW, colored by class label (9 classes: A, E, F, H, I, L, M, N, W). Paul Tol muted palette.
- Right: 2D MDS of GED distances, same dataset, same coloring, same aspect ratio.
- Visual comparison: if class clusters appear in both, the string space preserves structure.

**This is the advisor's primary requested visualization**: "grafos parecidos se representan mediante cadenas parecidas."

#### (C) LaTeX Table: `table_embedding_quality_{color|gray}.tex`

```
| Dataset  | Method     | NEV ratio | Stress-1 (2D)      | Stress-1 (5D)      | Var 2D (%) |
|----------|------------|-----------|--------------------|--------------------|------------|
| IAM LOW  | Exhaustive | X.XXX     | \best{X.XXXX}      | X.XXXX             | XX.X       |
| IAM LOW  | GED (ref)  | X.XXX     | X.XXXX             | X.XXXX             | XX.X       |
| ...      |            |           |                    |                    |            |
```

`\best{}` marks the lowest Stress-1 (best embedding quality).

---

### 3.7 H2.2 — Geometric Agreement (Procrustes)

**Claim**: Procrustes m² between MDS(Levenshtein) and MDS(GED) is significantly lower than random permutation (p < 0.001).

**Source**: Agent 2 — Procrustes results.

#### (A) Population Figure: `population_procrustes_comparison.pdf`

**Layout**: Single-column, 1 panel.

- Grouped bar chart: For each dataset, bars showing Procrustes m² at d = 2 and d = 5.
- Horizontal reference line: median m² from permutation null distribution (gray dashed).
- Significance stars above each bar.
- Whiskers: 95% CI from bootstrap on m² (if available; otherwise just the permutation p-value annotation).

#### (B) Individual Figure: `individual_procrustes_overlay.pdf`

**Layout**: Single-column, 1 panel.

- Superimpose MDS(Levenshtein, d=2) and MDS(GED, d=2) for IAM Letter LOW after Procrustes alignment.
- Lev points: circles (Paul Tol blue), GED points: triangles (Paul Tol red).
- Thin lines connecting corresponding points (same graph index).
- Shorter lines = better agreement. Color lines by residual magnitude (viridis).

#### (C) LaTeX Table: `table_procrustes_{color|gray}.tex`

```
| Dataset  | Method     | m² (2D)             | p (2D) | m² (5D)             | p (5D) |
|----------|------------|---------------------|--------|---------------------|--------|
| IAM LOW  | Exhaustive | \best{X.XXXX}       | ***    | X.XXXX              | ***    |
| IAM LOW  | Greedy     | X.XXXX              | ***    | X.XXXX              | ***    |
| ...      |            |                     |        |                     |        |
```

---

### 3.8 H2.3 — Shepard Fidelity

**Claim**: Shepard diagrams show a monotone relationship between original and embedded distances.

**Source**: Agent 2 — Shepard R² from `{dataset}_{method}_embedding_stats.json`.

#### (A) Population Figure: `population_shepard_r2_summary.pdf`

**Layout**: Single-column, 1 panel.

- Grouped bar chart: For each dataset, bars for Shepard R² at d=2 and d=5, both for Lev and GED.
- Horizontal reference at R² = 0.80 (high fidelity threshold).

#### (B) Individual Figure: `individual_shepard_diagrams.pdf`

**Layout**: 2 × 2 panels, full width.

| | Levenshtein | GED |
|--|-------------|-----|
| Best dataset (highest R²) | Shepard hexbin + isotonic regression | Shepard hexbin + isotonic regression |
| Worst dataset (lowest R²) | Shepard hexbin + isotonic regression | Shepard hexbin + isotonic regression |

Annotate each panel with R² and Spearman ρ².

#### (C) LaTeX Table: `table_shepard_{color|gray}.tex`

```
| Dataset  | Source     | R² (2D)             | ρ² (2D)  | R² (5D)             | ρ² (5D) |
|----------|-----------|---------------------|----------|---------------------|---------|
| IAM LOW  | Lev (exh) | \best{X.XXXX}       | X.XXXX   | X.XXXX              | X.XXXX  |
| IAM LOW  | GED       | X.XXXX              | X.XXXX   | X.XXXX              | X.XXXX  |
| ...      |           |                     |          |                     |         |
```

---

### 3.9 C3.1 — Speedup Factor

**Claim**: IsalGraph pipeline (encode + Levenshtein) is faster than exact GED for pairwise comparison.

**Source**: Agent 3 — `{dataset}_timing_stats.json`.

#### (A) Population Figure: `population_speedup_bar.pdf`

**Layout**: Single-column, 1 panel.

- Horizontal bar chart: Speedup factor = median(T_GED) / median(T_IsalGraph) per dataset.
- Two bars per dataset: exhaustive pipeline and greedy pipeline.
- Vertical reference line at speedup = 1 (breakeven).
- Log scale on x-axis (speedups can span orders of magnitude).
- Annotate geometric mean speedup across all datasets.
- Error bars: IQR-derived CI (since timing is skewed, report median ± IQR).

#### (B) Individual Figure: `individual_timing_trace.pdf`

**Layout**: Single-column, 1 panel.

- Select one representative pair (i, j) from LINUX (moderate size).
- Horizontal timeline:
  - Top bar: GED computation time (red bar, label "GED = X.XXX s").
  - Bottom bar: Encoding time for G_i + encoding time for G_j + Levenshtein time (stacked blue/green/purple segments).
- Visual ratio makes the speedup immediately intuitive.
- Annotate: "Speedup = X.Xx".

#### (C) LaTeX Table: `table_speedup_{color|gray}.tex`

```
| Dataset  | T_GED (ms) [med ± IQR] | T_encode (ms) | T_Lev (ms) | T_total (ms) | Speedup          |
|----------|------------------------|---------------|------------|---------------|------------------|
| IAM LOW  | X.XX ± X.XX            | X.XX          | X.XX       | X.XX          | \best{XXx}       |
| ...      |                        |               |            |               |                  |
```

---

### 3.10 C3.2 — Crossover Point

**Claim**: There exists a graph size n* at which IsalGraph becomes faster than exact GED.

**Source**: Agent 3 — `crossover_analysis.json`.

#### (A) Population Figure: `population_crossover_curves.pdf`

**Layout**: Full width, 1 panel.

- Log-log plot: X = graph size (number of nodes), Y = time per pair (seconds).
- Three curves: GED (exact A*), IsalGraph (exhaustive), IsalGraph (greedy).
- Fitted power-law / exponential regression lines (dashed) with R² annotation.
- Vertical dashed lines at n*_exhaustive and n*_greedy (crossover points).
- Shaded regions: "GED faster" (left of crossover) and "IsalGraph faster" (right).
- Reference complexity lines: O(n²), O(n³), O(n⁴), O(2^n) in light gray.

#### (B) Individual Figure: `individual_crossover_example.pdf`

**Layout**: Single-column, 1 panel.

Zoomed-in view of the crossover region. Show the actual data points (not just regression lines) near the crossing. Mark the exact crossover with a circle. Annotate: "At n = X, IsalGraph becomes faster."

#### (C) LaTeX Table: `table_crossover_{color|gray}.tex`

```
| Method       | Scaling model  | Exponent α | R²   | n* (crossover) | Speedup at n=12 |
|--------------|----------------|------------|------|----------------|-----------------|
| GED (exact)  | Exponential    | —          | X.XX | —              | 1.0x (ref)      |
| Exh + Lev    | Polynomial     | X.XX       | X.XX | X              | \best{X.Xx}     |
| Greedy + Lev | Polynomial     | X.XX       | X.XX | X              | \best{X.Xx}     |
```

---

### 3.11 C3.3 — Amortized Pipeline Comparison

**Claim**: For datasets of N graphs, the amortized IsalGraph pipeline (one-time encoding + O(N²) Levenshtein) is orders of magnitude faster than O(N²) GED computations.

**Source**: Agent 3 — `{dataset}_amortized_comparison.csv`.

#### (A) Population Figure: `population_amortized_comparison.pdf`

**Layout**: Full width, 1 panel.

- Stacked bar chart per dataset.
- Each bar decomposes into: encoding time (blue), Levenshtein time (green).
- Side-by-side: GED total time (red, single block).
- Y-axis: total wall-clock time (log scale).
- Annotate the speedup ratio above each pair of bars.

**Key insight to visualize**: The encoding cost is amortized once, and then Levenshtein for N(N−1)/2 pairs is fast. GED must do the full computation for every pair.

#### (B) Individual Figure: `individual_pipeline_breakdown.pdf`

**Layout**: Single-column, 1 panel.

- For the largest dataset (AIDS or LINUX), show a pie chart:
  - Encoding: X% of total IsalGraph time.
  - Levenshtein: Y% of total IsalGraph time.
- Next to it: a single-color pie showing GED time (100%).
- Size of pies proportional to absolute time.

#### (C) LaTeX Table: `table_amortized_{color|gray}.tex`

```
| Dataset  | N    | Pairs   | T_encode (s) | T_Lev (s) | T_IsalGraph (s) | T_GED (s) | Speedup          |
|----------|------|---------|--------------|------------|-----------------|-----------|------------------|
| IAM LOW  | XXX  | XXX,XXX | X.XX         | X.XX       | X.XX            | X.XX      | \best{XXx}       |
| ...      |      |         |              |            |                 |           |                  |
```

---

### 3.12 C4.1 — Scaling Exponents per Graph Family

**Claim**: Greedy encoding scales polynomially (α ≈ 2–4) for sparse families; canonical encoding is exponential for dense graphs.

**Source**: Agent 4 — `scaling_exponents.json`.

#### (A) Population Figure: `population_scaling_loglog.pdf`

**Layout**: Full width, 2 × 2 panels.

| Panel | Content |
|-------|---------|
| (a) | Log-log: Greedy single-start time vs N. One line per family (colored). Reference dashed lines: O(N²), O(N³), O(N⁴). Fit lines overlaid. |
| (b) | Log-log: Canonical time vs N for feasible families. Exponential families (complete, gnp_05) in dashed. |
| (c) | Bar chart: Scaling exponent α per family for greedy (blue) and canonical (orange). Reference dashed lines at α = 2, 3, 4. Annotate R² above each bar. |
| (d) | Scatter: Observed vs predicted time (synthetic prediction vs real dataset validation). Identity line. R² annotation. |

#### (B) Individual Figure: `individual_family_examples.pdf`

**Layout**: 1 × 3 panels, full width.

Show three graph families at n = 8:
- Path P₈ (sparse): draw the graph, show canonical string, show encoding time.
- BA(8, m=2) (moderate): same format.
- Complete K₈ (dense): same format.

Visual contrast: the path has a short string and fast encoding; the complete graph has a long string and slow encoding.

#### (C) LaTeX Table: `table_scaling_exponents_{color|gray}.tex`

```
| Family       | Density  | α_greedy | R²   | α_canonical         | R²   | Model       |
|--------------|----------|----------|------|---------------------|------|-------------|
| Path         | Sparse   | X.XX     | X.XX | X.XX                | X.XX | Polynomial  |
| Star         | Sparse   | X.XX     | X.XX | X.XX                | X.XX | Polynomial  |
| Complete     | Dense    | X.XX     | X.XX | exp (b=X.XX)        | X.XX | Exponential |
| ...          |          |          |      |                     |      |             |
```

`\best{}` marks the lowest α (fastest scaling); `\worst{}` marks exponential.

---

### 3.13 C4.2 — Density Dependence of Encoding

**Claim**: Encoding time increases super-linearly with edge density.

**Source**: Agent 4 — `density_analysis.json`.

#### (A) Population Figure: `population_density_heatmap.pdf`

**Layout**: Single-column, 1 panel.

- Heatmap: X-axis = density p ∈ {0.1, ..., 0.9}, Y-axis = node count N.
- Color = log₁₀(canonical encoding time in seconds).
- Contour lines at 1s, 10s, 60s, 300s (timeout thresholds).
- Cividis colormap (perceptually uniform, print-safe).
- This is a feasibility map: it shows the region where canonical encoding is practical.

#### (B) Individual Figure: `individual_density_pair.pdf`

**Layout**: 1 × 2 panels, single-column.

- Left: A sparse G(10, 0.2) graph drawn + canonical string + encoding time.
- Right: A dense G(10, 0.8) graph drawn + canonical string + encoding time.
- Annotate: "10× density increase → XXx time increase."

#### (C) LaTeX Table: `table_density_dependence_{color|gray}.tex`

```
| N  | p=0.1        | p=0.3        | p=0.5        | p=0.7        | p=0.9        |
|----|--------------|--------------|--------------|--------------|--------------|
| 5  | X.XX ms      | X.XX ms      | X.XX ms      | X.XX ms      | X.XX ms      |
| 8  | X.XX ms      | X.XX ms      | X.XX s       | X.XX s       | \worst{X.XX s} |
| 10 | X.XX ms      | X.XX s       | X.XX s       | \worst{TO}   | \worst{TO}   |
| 12 | X.XX s       | \worst{TO}   | \worst{TO}   | \worst{TO}   | \worst{TO}   |
```

`TO` = timeout (>300s). `\best{}` on fastest cells, `\worst{}` on timeout/slowest.

---

### 3.14 C4.3 — Greedy vs Canonical Overhead

**Claim**: Canonical encoding is orders of magnitude slower than greedy-min but produces lexicographically minimal strings.

**Source**: Agent 4 — `scaling_exponents.json` (both methods).

#### (A) Population Figure: `population_overhead_ratio.pdf`

**Layout**: Single-column, 1 panel.

- Bar chart: Ratio T_canonical / T_greedy per family (log scale y-axis).
- Horizontal reference at ratio = 1 (breakeven).
- Color bars by density class (sparse/moderate/dense).
- Annotate: median ratio across all families.

#### (B) Individual Figure: `individual_overhead_example.pdf`

**Layout**: Full width, 1 × 3 panels.

For one selected graph (e.g., BA(8, m=2)):
- Panel (a): Graph drawn.
- Panel (b): Greedy string from each starting node displayed in a table. Best greedy string highlighted.
- Panel (c): Canonical string. If it differs from greedy-min, show the alignment.
- Annotate times: "Greedy-min: X.XX ms, Canonical: X.XX ms, Overhead: XXx".

#### (C) LaTeX Table: `table_overhead_{color|gray}.tex`

```
| Family       | T_greedy_min (ms) | T_canonical (ms) | Overhead ratio     | String match % |
|--------------|-------------------|------------------|--------------------|----------------|
| Path         | X.XX              | X.XX             | \best{X.Xx}        | 100%           |
| Complete     | X.XX              | X,XXX.XX         | \worst{X,XXXx}     | XX%            |
| ...          |                   |                  |                    |                |
```

`String match %` = fraction of instances where greedy-min produces the same string as canonical.

---

### 3.15 Illustrative Figures (Visual Intuition, No Standalone Hypothesis)

These figures do not test a specific hypothesis but provide the visual evidence the advisor explicitly requested: *"grafos parecidos se representan mediante cadenas parecidas."*

#### `fig_example_pairs.pdf`

Already described in Section 3.1(B). This is the paper's "hero figure" — the first thing a reviewer sees.

#### `fig_string_alignment.pdf`

**Layout**: Full width, 2 rows.

- Row 1: Large-format alignment of a similar pair (from H1.1(B), Row 1). Each character is a colored cell: match (green), substitute (red), insert (blue), delete (orange). Two-row format (source above, target below). Monospace font, ~9pt.
- Row 2: Same for a dissimilar pair.
- Legend below.

#### `fig_heatmap_comparison.pdf`

**Layout**: Full width, 1 × 3 panels.

Three heatmaps for a **subset** of ~60 graphs from IAM Letter LOW (randomly sample ~7 per class for 9 classes):
1. GED distance matrix.
2. Levenshtein (exhaustive) distance matrix.
3. Levenshtein (greedy) distance matrix.

All three share: same row/column ordering (sorted by class label, then by within-class centroid distance under GED), same color scale (viridis, after min-max normalization), thin black lines between class boundaries.

#### `fig_nearest_neighbors.pdf`

**Layout**: 2 rows, full width.

Row 1:
- Center: Query graph drawn.
- Top 5: 5-NN by GED, drawn in a row.
- Bottom 5: 5-NN by Levenshtein, drawn in a row.
- Shared neighbors outlined with a green border.
- Annotate: "Overlap = X/5".

Row 2: Same for a different query graph from a different dataset.

Select query graphs with high P@5 (positive example) and moderate P@5 (realistic example).

#### `fig_class_clustering.pdf`

**Layout**: 1 × 2 panels, full width.

Identical to H2.1(B) but with specific publication styling:
- Larger point size.
- Class name labels placed at cluster centroids.
- Convex hulls drawn per class (light shading).
- This is specifically designed for the section of the paper discussing "topological similarity preservation."

---

## 4. Implementation Plan

### 4.1 Module Structure

```
benchmarks/eval_visualization/
    __init__.py
    eval_visualization.py          # CLI orchestrator
    figure_registry.py             # Maps hypothesis → figure generation function
    result_loader.py               # Loads all Agent 0–4 outputs
    graph_drawing.py               # NetworkX graph drawing utilities
    string_alignment.py            # Levenshtein DP backtrace + visualization
    example_selector.py            # Deterministic example selection
    table_generator.py             # LaTeX table generation (color + gray)
    population_figures/
        __init__.py
        correlation_figures.py     # H1.1–H1.5 population figures
        embedding_figures.py       # H2.1–H2.3 population figures
        computational_figures.py   # C3.1–C3.3 population figures
        encoding_figures.py        # C4.1–C4.3 population figures
    individual_figures/
        __init__.py
        correlation_examples.py    # H1.1–H1.5 individual figures
        embedding_examples.py      # H2.1–H2.3 individual figures
        computational_examples.py  # C3.1–C3.3 individual figures
        encoding_examples.py       # C4.1–C4.3 individual figures
    illustrative/
        __init__.py
        hero_figures.py            # fig_example_pairs, fig_string_alignment, etc.
    README.md
```

### 4.2 Result Loader (`result_loader.py`)

```python
"""Unified loader for all Agent 0–4 results.

Loads every JSON, CSV, and NPZ file produced by the evaluation
pipeline and provides typed access to all results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetArtifacts:
    """All artifacts for a single dataset."""

    name: str
    ged_matrix: np.ndarray
    lev_matrices: dict[str, np.ndarray]          # method -> matrix
    canonical_strings: dict[str, dict]            # method -> {graph_id: string}
    labels: np.ndarray | None
    node_counts: np.ndarray
    edge_counts: np.ndarray
    graph_ids: list[str]


@dataclass
class CorrelationStats:
    """Agent 1 results for one dataset × method."""

    spearman_rho: float
    spearman_ci: tuple[float, float]
    pearson_r: float
    pearson_ci: tuple[float, float]
    kendall_tau: float
    ccc: float
    mantel_p: float
    precision_at_k: dict[int, float]             # k -> P@k


@dataclass
class EmbeddingStats:
    """Agent 2 results for one dataset × method."""

    nev_ratio: float
    stress_by_dim: dict[int, float]              # dim -> Stress-1
    procrustes_m2: dict[int, float]              # dim -> m²
    procrustes_p: dict[int, float]               # dim -> p-value
    shepard_r2: dict[int, float]                 # dim -> R²


@dataclass
class ComputationalStats:
    """Agent 3 results for one dataset."""

    median_ged_time_ms: float
    median_encode_time_ms: dict[str, float]      # method -> time
    median_lev_time_ms: float
    speedup: dict[str, float]                    # method -> speedup
    crossover_n: dict[str, int]                  # method -> n*


@dataclass
class EncodingStats:
    """Agent 4 results (cross-family)."""

    scaling_exponents: dict[str, dict]           # family -> {alpha, R2, model}
    density_analysis: dict                       # density dependence results


@dataclass
class AllResults:
    """Complete results from all agents."""

    datasets: dict[str, DatasetArtifacts] = field(default_factory=dict)
    correlation: dict[str, dict[str, CorrelationStats]] = field(default_factory=dict)
    embedding: dict[str, dict[str, EmbeddingStats]] = field(default_factory=dict)
    computational: dict[str, ComputationalStats] = field(default_factory=dict)
    encoding: EncodingStats | None = None
    cross_distortion: dict | None = None         # H1.2, H1.3


def load_all_results(
    data_root: str,
    correlation_dir: str,
    embedding_dir: str,
    computational_dir: str,
    encoding_dir: str,
) -> AllResults:
    """Load all agent outputs into a unified structure.

    Args:
        data_root: Root of Agent 0 output (data/eval/).
        correlation_dir: Root of Agent 1 output.
        embedding_dir: Root of Agent 2 output.
        computational_dir: Root of Agent 3 output.
        encoding_dir: Root of Agent 4 output.

    Returns:
        AllResults with every agent's data loaded.
    """
    # Implementation loads JSON/NPZ/CSV from each directory
    # and populates the dataclass hierarchy.
    ...
```

### 4.3 Table Generator (`table_generator.py`)

```python
"""LaTeX table generation with color and grayscale variants.

Generates tables with:
- forestgreen/bold for best values
- darkred/underline for worst values
- Significance markers: ***, **, *, n.s.
- booktabs formatting
- Two output files per table: *_color.tex, *_gray.tex
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# LaTeX preamble fragments for each mode
COLOR_PREAMBLE = r"""
\usepackage{xcolor}
\definecolor{forestgreen}{RGB}{34,139,34}
\definecolor{darkred}{RGB}{139,0,0}
\newcommand{\best}[1]{\textcolor{forestgreen}{\textbf{#1}}}
\newcommand{\worst}[1]{\textcolor{darkred}{\underline{#1}}}
"""

GRAY_PREAMBLE = r"""
\newcommand{\best}[1]{\textbf{#1}}
\newcommand{\worst}[1]{\underline{#1}}
"""


def _format_significance(p: float) -> str:
    """Map p-value to significance marker."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def _highlight_best_worst(
    series: pd.Series,
    higher_is_better: bool = True,
    mode: str = "color",
) -> pd.Series:
    """Apply best/worst highlighting to a pandas Series.

    Args:
        series: Numeric series to highlight.
        higher_is_better: If True, max is best; if False, min is best.
        mode: 'color' or 'gray'.

    Returns:
        Series with LaTeX formatting applied.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    best_idx = numeric.idxmax() if higher_is_better else numeric.idxmin()
    worst_idx = numeric.idxmin() if higher_is_better else numeric.idxmax()

    result = series.copy()
    if pd.notna(best_idx):
        result[best_idx] = f"\\best{{{series[best_idx]}}}"
    if pd.notna(worst_idx) and worst_idx != best_idx:
        result[worst_idx] = f"\\worst{{{series[worst_idx]}}}"
    return result


def generate_dual_table(
    df: pd.DataFrame,
    output_dir: str | Path,
    basename: str,
    caption: str,
    label: str,
    highlight_cols: dict[str, bool] | None = None,
    note: str = "",
) -> tuple[Path, Path]:
    """Generate both color and grayscale LaTeX tables.

    Args:
        df: DataFrame with table content.
        output_dir: Output directory.
        basename: Base filename (without extension).
        caption: Table caption.
        label: LaTeX label.
        highlight_cols: Dict mapping column names to higher_is_better.
        note: Optional footnote below the table.

    Returns:
        Tuple of (color_path, gray_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("color", "gray"):
        df_styled = df.copy()
        if highlight_cols:
            for col, higher in highlight_cols.items():
                if col in df_styled.columns:
                    df_styled[col] = _highlight_best_worst(
                        df_styled[col], higher, mode
                    )

        preamble = COLOR_PREAMBLE if mode == "color" else GRAY_PREAMBLE
        latex_body = df_styled.to_latex(
            index=False,
            escape=False,
            column_format="l" + "c" * (len(df.columns) - 1),
        )

        # Wrap in booktabs table environment
        full_latex = (
            f"% Preamble (include in document header):\n"
            f"% {preamble.strip()}\n\n"
            f"\\begin{{table}}[htbp]\n"
            f"\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"\\small\n"
            f"{latex_body}"
        )
        if note:
            full_latex += f"\\\\[2pt]\n\\footnotesize {note}\n"
        full_latex += "\\end{table}\n"

        path = output_dir / f"{basename}_{mode}.tex"
        path.write_text(full_latex, encoding="utf-8")
        logger.info("Saved %s", path)

    return (
        output_dir / f"{basename}_color.tex",
        output_dir / f"{basename}_gray.tex",
    )
```

### 4.4 Example Selector (`example_selector.py`)

```python
"""Deterministic example selection for individual-level figures.

All selection criteria are documented and reproducible.
Results are logged to selected_examples.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SelectedExample:
    """Record of a selected example with justification."""

    hypothesis: str
    role: str                  # e.g., "concordant_similar", "discordant"
    dataset: str
    indices: tuple[int, ...]   # graph indices in the dataset
    graph_ids: tuple[str, ...]
    criterion: str             # e.g., "min Lev among GED=1 pairs"
    ged: float | None
    lev: float | None


def select_concordant_similar(
    ged: np.ndarray,
    lev: np.ndarray,
    node_counts: np.ndarray,
    labels: np.ndarray | None = None,
    target_ged: int = 1,
    min_nodes: int = 5,
    max_nodes: int = 8,
) -> tuple[int, int]:
    """Select a concordant similar pair (small GED, small Lev).

    Strategy:
        1. Filter pairs with GED == target_ged.
        2. Among those, prefer same-class pairs (if labels available).
        3. Among those, prefer node counts in [min_nodes, max_nodes].
        4. Break ties by smallest Levenshtein distance.
    """
    n = ged.shape[0]
    best_pair = None
    best_lev = float("inf")

    for i in range(n):
        for j in range(i + 1, n):
            if ged[i, j] != target_ged:
                continue
            if not (min_nodes <= node_counts[i] <= max_nodes):
                continue
            if not (min_nodes <= node_counts[j] <= max_nodes):
                continue
            if labels is not None and labels[i] != labels[j]:
                continue
            if lev[i, j] < best_lev:
                best_lev = lev[i, j]
                best_pair = (i, j)

    # Fallback: relax constraints if no pair found
    if best_pair is None:
        for i in range(n):
            for j in range(i + 1, n):
                if ged[i, j] != target_ged:
                    continue
                if lev[i, j] < best_lev:
                    best_lev = lev[i, j]
                    best_pair = (i, j)

    return best_pair


def select_discordant(
    ged: np.ndarray,
    lev: np.ndarray,
) -> tuple[int, int]:
    """Select the most discordant pair (largest rank disagreement).

    Uses |rank_GED(i,j) - rank_Lev(i,j)| as the discordance measure.
    """
    from scipy.stats import rankdata

    triu = np.triu_indices_from(ged, k=1)
    ged_flat = ged[triu]
    lev_flat = lev[triu]

    valid = np.isfinite(ged_flat) & np.isfinite(lev_flat)
    ged_ranks = rankdata(ged_flat[valid])
    lev_ranks = rankdata(lev_flat[valid])

    discordance = np.abs(ged_ranks - lev_ranks)
    worst_idx = np.argmax(discordance)

    # Map back to matrix indices
    valid_indices = np.where(valid)[0]
    flat_idx = valid_indices[worst_idx]
    i = triu[0][flat_idx]
    j = triu[1][flat_idx]
    return (int(i), int(j))


def save_selection_log(
    selections: list[SelectedExample],
    output_path: str | Path,
) -> None:
    """Save the selection log as JSON for reproducibility."""
    records = []
    for sel in selections:
        records.append({
            "hypothesis": sel.hypothesis,
            "role": sel.role,
            "dataset": sel.dataset,
            "indices": list(sel.indices),
            "graph_ids": list(sel.graph_ids),
            "criterion": sel.criterion,
            "ged": sel.ged,
            "lev": sel.lev,
        })
    Path(output_path).write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    logger.info("Saved selection log to %s", output_path)
```

### 4.5 Graph Drawing (`graph_drawing.py`)

```python
"""Graph drawing utilities for publication figures.

Uses NetworkX spring layout with fixed seed for reproducibility.
Consistent styling aligned with the project's plotting_styles.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


def draw_graph(
    G: nx.Graph,
    ax: plt.Axes,
    title: str = "",
    canonical_string: str = "",
    highlight_edges: set | None = None,
    highlight_color: str = "#EE6677",
    node_color: str = "#4477AA",
    edge_color: str = "#333333",
    node_size: int = 300,
    layout_seed: int = 42,
) -> None:
    """Draw a graph on a matplotlib Axes with publication styling.

    Args:
        G: NetworkX graph to draw.
        ax: Target Axes object.
        title: Title above the graph.
        canonical_string: IsalGraph string, displayed below the graph.
        highlight_edges: Set of (u, v) edges to highlight.
        highlight_color: Color for highlighted edges.
        node_color: Default node fill color (Paul Tol bright blue).
        edge_color: Default edge color.
        node_size: Marker area for nodes.
        layout_seed: Seed for spring_layout reproducibility.
    """
    k = 1.5 / np.sqrt(max(G.number_of_nodes(), 1))
    pos = nx.spring_layout(G, seed=layout_seed, k=k)

    # Separate edge colors
    e_colors, e_widths = [], []
    for u, v in G.edges():
        if highlight_edges and ((u, v) in highlight_edges or (v, u) in highlight_edges):
            e_colors.append(highlight_color)
            e_widths.append(2.5)
        else:
            e_colors.append(edge_color)
            e_widths.append(1.0)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=e_colors, width=e_widths, alpha=0.7)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_color, node_size=node_size,
        edgecolors="#2c3e50", linewidths=0.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white", font_weight="bold")

    ax.set_title(title, fontsize=10, pad=8)
    if canonical_string:
        ax.text(
            0.5, -0.08, f"$w^* = $ {canonical_string}",
            transform=ax.transAxes, ha="center", fontsize=8,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#bdc3c7"),
        )
    ax.set_axis_off()
```

### 4.6 String Alignment Visualization (`string_alignment.py`)

```python
"""Levenshtein alignment via DP backtrace and visual rendering.

Computes the optimal edit alignment and renders it as a
two-row colored character grid.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

MATCH_COLOR = "#228833"      # Paul Tol green
SUBSTITUTE_COLOR = "#EE6677"  # Paul Tol red
INSERT_COLOR = "#4477AA"      # Paul Tol blue
DELETE_COLOR = "#CCBB44"      # Paul Tol yellow


def levenshtein_alignment(
    s: str,
    t: str,
) -> list[tuple[str, str | None, str | None]]:
    """Compute Levenshtein alignment via DP backtrace.

    Args:
        s: Source string.
        t: Target string.

    Returns:
        List of (operation, char_from_s, char_from_t).
        Operations: 'match', 'substitute', 'insert', 'delete'.
    """
    n, m = len(s), len(t)
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
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Backtrace
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s[i - 1] == t[j - 1]:
            alignment.append(("match", s[i - 1], t[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            alignment.append(("substitute", s[i - 1], t[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            alignment.append(("insert", None, t[j - 1]))
            j -= 1
        else:
            alignment.append(("delete", s[i - 1], None))
            i -= 1

    return list(reversed(alignment))


def draw_alignment(
    alignment: list[tuple[str, str | None, str | None]],
    ax: plt.Axes,
    s_label: str = "w*₁",
    t_label: str = "w*₂",
) -> None:
    """Render a Levenshtein alignment as a colored two-row grid.

    Args:
        alignment: Output of levenshtein_alignment().
        ax: Matplotlib Axes to draw on.
        s_label: Label for the source row.
        t_label: Label for the target row.
    """
    colors = {
        "match": MATCH_COLOR,
        "substitute": SUBSTITUTE_COLOR,
        "insert": INSERT_COLOR,
        "delete": DELETE_COLOR,
    }
    cell_w, cell_h, gap = 0.6, 0.4, 0.15
    n_cols = len(alignment)

    ax.text(-0.8, cell_h + gap / 2, s_label, ha="right", va="center",
            fontsize=8, fontweight="bold")
    ax.text(-0.8, -gap / 2, t_label, ha="right", va="center",
            fontsize=8, fontweight="bold")

    for col, (op, cs, ct) in enumerate(alignment):
        x = col * cell_w
        color = colors[op]

        # Source row
        rect_s = FancyBboxPatch(
            (x, gap / 2), cell_w - 0.05, cell_h,
            boxstyle="round,pad=0.02", facecolor=color, alpha=0.3,
            edgecolor=color, linewidth=1,
        )
        ax.add_patch(rect_s)
        ax.text(x + cell_w / 2, gap / 2 + cell_h / 2,
                cs if cs else "—", ha="center", va="center",
                fontsize=9, fontfamily="monospace", fontweight="bold")

        # Target row
        rect_t = FancyBboxPatch(
            (x, -cell_h - gap / 2), cell_w - 0.05, cell_h,
            boxstyle="round,pad=0.02", facecolor=color, alpha=0.3,
            edgecolor=color, linewidth=1,
        )
        ax.add_patch(rect_t)
        ax.text(x + cell_w / 2, -gap / 2 - cell_h / 2,
                ct if ct else "—", ha="center", va="center",
                fontsize=9, fontfamily="monospace", fontweight="bold")

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

---

## 5. Figure Registry and Orchestration

### 5.1 Figure Registry (`figure_registry.py`)

A mapping from hypothesis IDs to generator functions. The CLI iterates over this registry:

```python
"""Registry mapping hypothesis IDs to figure generator functions."""

from __future__ import annotations

from typing import Callable

FIGURE_REGISTRY: dict[str, dict[str, Callable]] = {
    "H1_1": {
        "population": "population_figures.correlation_figures.fig_h1_1_population",
        "individual": "individual_figures.correlation_examples.fig_h1_1_individual",
        "table": "table_generator.table_h1_1",
    },
    "H1_2": {
        "population": "population_figures.correlation_figures.fig_h1_2_population",
        "individual": "individual_figures.correlation_examples.fig_h1_2_individual",
        "table": "table_generator.table_h1_2",
    },
    # ... etc for H1_3, H1_4, H1_5, H2_1, H2_2, H2_3, C3_1, C3_2, C3_3, C4_1, C4_2, C4_3
    "illustrative": {
        "example_pairs": "illustrative.hero_figures.fig_example_pairs",
        "string_alignment": "illustrative.hero_figures.fig_string_alignment",
        "heatmap_comparison": "illustrative.hero_figures.fig_heatmap_comparison",
        "nearest_neighbors": "illustrative.hero_figures.fig_nearest_neighbors",
        "class_clustering": "illustrative.hero_figures.fig_class_clustering",
    },
}
```

### 5.2 CLI Interface

```bash
# Generate ALL paper figures
python -m benchmarks.eval_visualization.eval_visualization \
    --data-root data/eval \
    --results-root results \
    --output-dir paper_figures \
    --seed 42

# Generate figures for a single hypothesis
python -m benchmarks.eval_visualization.eval_visualization \
    --data-root data/eval \
    --results-root results \
    --output-dir paper_figures \
    --hypotheses H1_1,H2_1 \
    --seed 42

# Generate only population figures (skip individual + tables)
python -m benchmarks.eval_visualization.eval_visualization \
    --data-root data/eval \
    --results-root results \
    --output-dir paper_figures \
    --artifacts population \
    --seed 42

# Generate only tables
python -m benchmarks.eval_visualization.eval_visualization \
    --data-root data/eval \
    --results-root results \
    --output-dir paper_figures \
    --artifacts table \
    --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root` | `data/eval` | Root of Agent 0 output |
| `--results-root` | `results` | Root containing `eval_correlation/`, `eval_embedding/`, etc. |
| `--output-dir` | `paper_figures` | Output directory for all artifacts |
| `--seed` | `42` | Random seed for layouts and selections |
| `--hypotheses` | `all` | Comma-separated hypothesis IDs or `all` |
| `--artifacts` | `all` | Comma-separated: `population`, `individual`, `table`, or `all` |
| `--datasets` | `all` | Restrict to specific datasets |
| `--dpi` | `300` | DPI for PNG export |

---

## 6. Design Principles

### 6.1 Color Scheme

All figures use the Paul Tol colorblind-safe palettes, standardized in `benchmarks/plotting_styles.py`:

| Purpose | Color | Hex | Palette |
|---------|-------|-----|---------|
| Dataset: IAM LOW | Blue | `#4477AA` | Bright |
| Dataset: IAM MED | Cyan | `#66CCEE` | Bright |
| Dataset: IAM HIGH | Green | `#228833` | Bright |
| Dataset: LINUX | Red | `#EE6677` | Bright |
| Dataset: AIDS | Purple | `#AA3377` | Bright |
| Method: Exhaustive | Blue | `#004488` | High Contrast |
| Method: Greedy | Yellow | `#DDAA33` | High Contrast |
| Method: GED (reference) | Red | `#BB5566` | High Contrast |
| Alignment: Match | Green | `#228833` | — |
| Alignment: Substitute | Red | `#EE6677` | — |
| Alignment: Insert | Blue | `#4477AA` | — |
| Alignment: Delete | Yellow | `#CCBB44` | — |

### 6.2 Typography

```python
PUBLICATION_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
}
```

Use `apply_ieee_style()` from `benchmarks/plotting_styles.py` for all figures.

### 6.3 Panel Labels

All multi-panel figures use bold lowercase labels: **(a)**, **(b)**, **(c)**, etc., positioned at the top-left of each panel using `ax.text(-0.15, 1.05, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold")`.

### 6.4 Significance Annotations

For bar charts with significance comparisons, use bracket annotations:

```python
def annotate_significance(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p: float,
    height: float = 0.02,
) -> None:
    """Draw a significance bracket between two bars.

    Args:
        ax: Axes to annotate.
        x1, x2: X positions of the two bars.
        y: Y position of the bracket.
        p: p-value for the comparison.
        height: Bracket height as fraction of y range.
    """
    sig_text = _format_significance(p)
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
            color="black", linewidth=0.8)
    ax.text((x1 + x2) / 2, y + height, sig_text,
            ha="center", va="bottom", fontsize=9)
```

### 6.5 Reproducibility

- All graph layouts: `seed=42`.
- All bootstrap: `seed=42`.
- All selections documented in `selected_examples.json`.
- `figure_manifest.json` records every generated file with its hypothesis, caption, and generation timestamp.

### 6.6 What NOT to Include

- **No t-SNE / UMAP** (violate distance metric properties; see Experimental Evaluation Design, Section 4).
- **No 3D plots** (unreadable in print).
- **No interactive visualizations** (incompatible with static PDF submission).
- **No pie charts for multi-category data** (only for 2–3 category budget breakdowns in C3.3).
- **No dual y-axes** (misleading; use separate panels instead).

---

## 7. Acceptance Criteria

### 7.1 Completeness

1. ✅ All 14 hypothesis directories exist under `paper_figures/`.
2. ✅ Each directory contains: `population_*.pdf`, `individual_*.pdf`, `table_*_color.tex`, `table_*_gray.tex`.
3. ✅ The `illustrative/` directory contains all 5 hero figures.
4. ✅ `selected_examples.json` documents every example selection with justification.
5. ✅ `figure_manifest.json` catalogs every generated file.

### 7.2 Quality

6. ✅ All figures render at 300 DPI without clipping, overlapping text, or truncated labels.
7. ✅ All figures use serif fonts ≥ 8pt.
8. ✅ All figures use Paul Tol colorblind-safe palettes.
9. ✅ All multi-panel figures have panel labels (a), (b), etc.
10. ✅ All statistical annotations use correct significance notation.

### 7.3 Tables

11. ✅ All `_color.tex` tables use `\forestgreen` for best and `\darkred` for worst.
12. ✅ All `_gray.tex` tables use `\textbf{}` for best and `\underline{}` for worst.
13. ✅ All tables include significance markers (`***`, `**`, `*`, `n.s.`).
14. ✅ All tables compile without errors in a standard LaTeX environment with `booktabs`, `xcolor`.

### 7.4 Code

15. ✅ Code passes `ruff check`.
16. ✅ All functions have docstrings with type annotations.
17. ✅ No hardcoded paths (all configurable via CLI flags).

---

## 8. Summary: Paper Figure Map

This table maps each paper section to the figures and tables generated by this agent.

| Paper section | Hypothesis | Population fig | Individual fig | Table |
|---------------|-----------|----------------|----------------|-------|
| §5.1 Correlation | H1.1 | `population_correlation_overview.pdf` | `individual_concordant_pairs.pdf` | `correlation_summary` |
| §5.1 Distortion | H1.2 | `population_distortion_trend.pdf` | `individual_distortion_examples.pdf` | `monotone_degradation` |
| §5.1 Density | H1.3 | `population_density_vs_rho.pdf` | `individual_sparse_vs_dense.pdf` | `density_stratification` |
| §5.1 Discrimination | H1.4 | `population_within_vs_between.pdf` | `individual_class_pair_examples.pdf` | `class_discrimination` |
| §5.1 Methods | H1.5 | `population_method_comparison.pdf` | `individual_method_divergence.pdf` | `method_comparison` |
| §5.2 Embedding | H2.1 | `population_stress_by_dimension.pdf` | `individual_mds_scatter.pdf` | `embedding_quality` |
| §5.2 Procrustes | H2.2 | `population_procrustes_comparison.pdf` | `individual_procrustes_overlay.pdf` | `procrustes` |
| §5.2 Shepard | H2.3 | `population_shepard_r2_summary.pdf` | `individual_shepard_diagrams.pdf` | `shepard` |
| §5.3 Speedup | C3.1 | `population_speedup_bar.pdf` | `individual_timing_trace.pdf` | `speedup` |
| §5.3 Crossover | C3.2 | `population_crossover_curves.pdf` | `individual_crossover_example.pdf` | `crossover` |
| §5.3 Pipeline | C3.3 | `population_amortized_comparison.pdf` | `individual_pipeline_breakdown.pdf` | `amortized` |
| §5.4 Scaling | C4.1 | `population_scaling_loglog.pdf` | `individual_family_examples.pdf` | `scaling_exponents` |
| §5.4 Density | C4.2 | `population_density_heatmap.pdf` | `individual_density_pair.pdf` | `density_dependence` |
| §5.4 Overhead | C4.3 | `population_overhead_ratio.pdf` | `individual_overhead_example.pdf` | `overhead` |
| §4 Visual | (illus.) | `fig_example_pairs.pdf`, `fig_heatmap_comparison.pdf`, `fig_class_clustering.pdf`, `fig_nearest_neighbors.pdf` | `fig_string_alignment.pdf` | — |

**Total artifacts**: 14 population figures + 14 individual figures + 28 LaTeX tables (14 × 2 modes) + 5 illustrative figures = **61 files** (plus PNG copies = ~94 total).
