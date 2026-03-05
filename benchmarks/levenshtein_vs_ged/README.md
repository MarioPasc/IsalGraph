# Levenshtein vs Graph Edit Distance Benchmark

## Purpose

Tests the **topological locality** claim from the preprint (Lopez-Rubio 2025, Section 2.3):

> A minimal change in the adjacency matrix corresponds to a small modification in the representing string, as measured by the Levenshtein distance between strings.

This validates that the IsalGraph representation preserves graph similarity structure: similar graphs produce similar strings.

## Method

Three experiments:

1. **Controlled edge-edits**: Start from base graphs, perturb by k=1,2,3,4 edges, measure Levenshtein distance between canonical strings. Tests monotonic scaling.
2. **Family pairs**: Compare structurally different graphs (path vs cycle, star vs path, complete vs cycle) of the same size. Tests cross-family discrimination.
3. **Random GNP pairs**: Compare random connected graphs at various densities. Tests general correlation.

For each pair: compute canonical strings (exhaustive backtracking), Levenshtein distance, and exact GED (via NetworkX, NP-hard).

## Statistical Tests

- **Pearson & Spearman correlation** with bootstrap 95% CI (10,000 resamples).
- **Jonckheere-Terpstra trend test** for monotonic increase of Levenshtein with k.
- **Cohen's d** for effect size: k=1 vs k=4 Levenshtein distances.

## Figure: `levenshtein_vs_ged_figure.pdf`

**Layout**: 1x3, double-column width.

| Panel | Content |
|-------|---------|
| (a) | Scatter: Levenshtein distance vs GED, colored by experiment type. OLS regression line with Pearson r annotation. |
| (b) | Line plot with error bands: mean Levenshtein distance vs k (edge-edit count). Shows locality property. |
| (c) | Box plot: Levenshtein/GED ratio by experiment type. |

## Table: `levenshtein_vs_ged_table.tex`

| Column | Description |
|--------|-------------|
| Experiment | Experiment type (edge_edit, family_pair, random_pair) |
| N_pairs | Number of graph pairs |
| Pearson_r | Pearson correlation coefficient |
| Pearson_95CI | Bootstrap 95% confidence interval |
| Spearman_r | Spearman rank correlation |
| Mean_Lev | Mean Levenshtein distance |
| Mean_GED | Mean graph edit distance |
| Mean_time_s | Mean computation time per pair |

## Expected Results

From local testing (seed=42, 85 pairs, max_nodes=7):
- **Overall**: Pearson r = 0.83 (p < 10^-22)
- **Locality confirmed**: k=1 -> Lev 4.2; k=4 -> Lev 6.1 (monotonically increasing)
- **Family pairs**: Pearson r = 0.88 (excellent structural correlation)

## Running

```bash
# Local (quick)
python -m benchmarks.levenshtein_vs_ged.levenshtein_vs_ged \
    --seed 42 --max-nodes 6 --num-random-pairs 20 --output-dir /tmp/lvg \
    --csv --plot --table

# Local (parallel)
python -m benchmarks.levenshtein_vs_ged.levenshtein_vs_ged \
    --seed 42 --max-nodes 8 --num-random-pairs 100 --n-workers 8 \
    --csv --plot --table

# Picasso (via SLURM)
bash slurm/launch.sh --benchmark levenshtein_vs_ged
```
