# Cascading Changes: Dual-Mode Impact on Agents 1–5

**Context**: Agent 0 now produces TWO sets of canonical strings and TWO Levenshtein matrices per dataset (`_exhaustive` and `_greedy` suffixes). This document specifies the modifications required in each downstream agent. **Each agent's original `.md` file remains the authoritative reference for its scope, methods, and output formats** — this document only describes what changes.

---

## Naming Convention

Throughout all agents, the string method is referenced as:

```python
METHODS = ["exhaustive", "greedy"]
```

File paths follow the pattern:
```
canonical_strings/{dataset}_{method}.json
levenshtein_matrices/{dataset}_{method}.npz
```

The GED matrix, graph metadata, and filtering report are method-independent (single file per dataset).

---

## Agent 1: Correlation Analysis — Changes

### New Hypothesis

Add **H₅ (Method equivalence)**: Spearman ρ(Lev_exhaustive, GED) and Spearman ρ(Lev_greedy, GED) are not significantly different.

Tested via a **paired comparison**: for each dataset, compute ρ_exhaustive and ρ_greedy, then test equality using bootstrap confidence intervals on the **difference** Δρ = ρ_exhaustive − ρ_greedy. If the 95% CI for Δρ includes 0, the methods are statistically equivalent for proxy purposes.

### Execution Change

The main analysis loop becomes:

```python
for dataset in DATASETS:
    ged_matrix = load_ged(dataset)
    for method in METHODS:
        lev_matrix = load_levenshtein(dataset, method)
        stats = compute_all_correlations(ged_matrix, lev_matrix)
        save_stats(dataset, method, stats)
    
    # Method comparison
    compare_methods(dataset)
```

### Output Changes

- **Raw CSVs**: Now `{dataset}_{method}_pair_data.csv` (10 files total instead of 5).
- **Stats JSON**: Now `{dataset}_{method}_correlation_stats.json` (10 files).
- **New file**: `method_comparison_correlation.json` — per-dataset Δρ, bootstrap CI, and conclusion.
- **Summary table**: Adds a `Method` column. Rows: 5 datasets × 2 methods = 10 rows.

### Figure Changes

- **Panel (d)** of the main figure: Grouped bar chart with two bars per dataset (exhaustive vs greedy), both with 95% CI. Color-coded by method.
- **New supplementary figure**: `method_correlation_comparison.pdf` — scatter of ρ_exhaustive vs ρ_greedy across all 5 datasets, with identity line. Points near the line = methods equivalent.

### Cross-distortion analysis (H₂)

Run the Jonckheere-Terpstra test **separately** for exhaustive and greedy. If both show the same monotone degradation pattern, it strengthens the case that greedy is sufficient.

---

## Agent 2: Embedding Quality — Changes

### Execution Change

Run SMACOF + cMDS + Procrustes for both methods:

```python
for dataset in DATASETS:
    for method in METHODS:
        lev_matrix = load_levenshtein(dataset, method)
        eigenvalues = classical_mds_eigenvalues(lev_matrix)
        coords, stress = smacof_embedding(lev_matrix, ...)
        save_results(dataset, method, ...)
```

Procrustes is run **three ways** per dataset:
1. MDS(Lev_exhaustive) vs MDS(GED) — original analysis.
2. MDS(Lev_greedy) vs MDS(GED) — new.
3. MDS(Lev_exhaustive) vs MDS(Lev_greedy) — how similar are the two embeddings?

### Output Changes

- SMACOF coordinates: `{dataset}_{method}_smacof_coords_{dim}d.npz` (20 files for 5 datasets × 2 methods × multiple dims).
- Stats JSON: `{dataset}_{method}_embedding_stats.json`.
- Summary table: adds `Method` column.

### Figure Changes

- Main figure panel (a)/(b): Show MDS of exhaustive Levenshtein (primary) and greedy as supplementary.
- New supplementary: Procrustes overlay of exhaustive vs greedy MDS embeddings, showing how close the two geometries are.

---

## Agent 3: Computational Advantage — Changes

### Minimal Impact

Agent 3 already measures encoding time for both methods (exhaustive vs greedy is its core comparison). The dual-mode output from Agent 0 provides precomputed strings, but Agent 3 performs its **own** timing measurements.

### One Adjustment

The amortized comparison now has **three** regimes:
1. **GED (exact A*)**: Exponential — the baseline.
2. **IsalGraph (exhaustive + Levenshtein)**: Encode all graphs via exhaustive canonical + all-pairs Levenshtein.
3. **IsalGraph (greedy-min + Levenshtein)**: Encode all graphs via greedy-min + all-pairs Levenshtein.

The crossover analysis identifies two crossover points:
- n*_exhaustive: where exhaustive IsalGraph beats GED.
- n*_greedy: where greedy IsalGraph beats GED.

We expect n*_greedy < n*_exhaustive (greedy crosses over earlier since it's faster).

### Output Change

- `amortized_comparison.csv`: Add a `method` column with values `exhaustive`, `greedy`, `ged`.

---

## Agent 4: Encoding Complexity — Changes

### Minimal Impact

Agent 4 already measures both greedy and exhaustive scaling on synthetic graphs. The dual-mode output from Agent 0 provides real-dataset timing data for validation.

### One Enhancement

Use the per-graph timing data from Agent 0's `method_comparison/{dataset}_comparison.json` as a **real-world validation set** for the synthetic scaling regressions. Specifically:
- Plot observed canonical times from real datasets against predicted times from the synthetic scaling model.
- Report R² of the prediction.

This replaces the generic "real dataset validation" in the original Agent 4 with concrete data from Agent 0.

---

## Agent 5: Visualization — Changes

### New Figure: Exhaustive vs Greedy String Comparison

Add `fig_method_comparison.pdf`:

**Layout**: 2 rows × 3 columns.

| Position | Content |
|----------|---------|
| Row 1, Col 1 | Graph G drawn |
| Row 1, Col 2 | Exhaustive string w*_G (highlighted) |
| Row 1, Col 3 | Greedy-min string w'_G (highlighted) |
| Row 2 (spanning) | Alignment between w*_G and w'_G showing where they differ |

Select 2 example graphs:
1. One where exhaustive = greedy (majority case, ~68%).
2. One where exhaustive ≠ greedy (interesting case, ~32%) — show that the difference is typically small.

### Heatmap Enhancement

For `fig_heatmap_comparison.pdf`, now show **three** heatmaps side-by-side:
1. GED matrix
2. Levenshtein (exhaustive)
3. Levenshtein (greedy)

Same subset, same ordering, same color scale. Visual comparison of how similar the three matrices are.

### k-NN Enhancement

For `fig_nearest_neighbors.pdf`, show k-NN overlap for BOTH methods vs GED. This visually demonstrates whether the greedy proxy finds the same neighbors as the exhaustive proxy.

---

## Summary of Changes per Agent

| Agent | Files affected | New outputs | New figures | Effort |
|-------|---------------|-------------|-------------|--------|
| 1 (Correlation) | Main loop, stats, figures | +5 CSVs, +5 JSONs, 1 comparison JSON | 1 new panel, 1 supplementary | Medium |
| 2 (Embedding) | Main loop, Procrustes | +5 eigenvalue files, +5 stats | 1 supplementary | Low-Medium |
| 3 (Computational) | Amortized comparison | Adds `method` column | Minor panel adjustment | Low |
| 4 (Encoding) | Validation step | Uses Agent 0 timing data | No new figures | Low |
| 5 (Visualization) | 3 figure modifications | 1 new figure | `fig_method_comparison.pdf` | Medium |

---

## Execution DAG (Updated)

```
Agent 0 (SETUP: dual-mode) ─────┬──→ Agent 1 (CORRELATION: ×2 methods + comparison)
                                 ├──→ Agent 2 (EMBEDDING: ×2 methods + Procrustes³)
                                 ├──→ Agent 3 (COMPUTATIONAL: 3-way amortized)
                                 ├──→ Agent 4 (ENCODING: real-data validation)
                                 └──→ Agent 5 (VIZ: method comparison figures)
```

All agents 1–5 remain parallelizable after Agent 0 completes.
