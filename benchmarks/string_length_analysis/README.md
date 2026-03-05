# String Length Analysis Benchmark

## Purpose

Tests the **compactness claim** from the preprint (Lopez-Rubio 2025, Section 2.2):

> For sparse graphs, the IsalGraph string length |w| is significantly shorter than the N^2 binary adjacency representation.

The preprint derives (Eq. 9) for the matrix-pointer algorithm:
E[|I_G|] ~ (sqrt(pi) / (2*sqrt(2))) * N^2 * sqrt(rho)

Our CDLL-based algorithm uses a different instruction set and may exhibit different scaling.

## Method

1. **Graph families**: Trees, paths, stars, cycles, complete graphs, Barabasi-Albert (m=1,2,3), GNP at controlled densities, Watts-Strogatz, grids, ladders, wheels, Petersen.
2. **Sizes**: N = 4 to 200 (configurable).
3. **Metrics**: Greedy string length (min over starting nodes), canonical string length (N <= 8), compression ratio |w|/N^2, theoretical Eq. 9 prediction.
4. **Timing**: Per-graph computation time tracks encoding performance.

## Statistical Tests

- **OLS regression**: log(|w|) ~ log(N) per family to extract scaling exponents.
- **Pearson correlation**: compression ratio vs edge density.
- **Wilcoxon signed-rank test**: greedy_best vs canonical length for N <= 8.

## Figure: `string_length_analysis_figure.pdf`

**Layout**: 2x2, double-column width.

| Panel | Content |
|-------|---------|
| (a) | Log-log scatter: \|w\| vs N, colored by family. Reference lines: y=N, y=N^2. |
| (b) | Scatter: compression ratio \|w\|/N^2 vs density rho, colored by family. |
| (c) | Bar chart: mean compression ratio by family (fixed N or aggregated). |
| (d) | Scatter: canonical vs greedy string length (N<=8) with y=x reference line. |

## Table: `string_length_analysis_table.tex`

| Column | Description |
|--------|-------------|
| Family | Graph family name |
| N_range | Range of graph sizes tested |
| Mean_\|w\| | Mean greedy string length |
| Mean_ratio | Mean compression ratio \|w\|/N^2 |
| Best_ratio | Best (minimum) compression ratio |
| Mean_time_s | Mean computation time per graph |

## Expected Results

From local testing (seed=42, 105 graphs):
- **Stars**: optimal compression (N-1 chars for N nodes, ratio ~4%)
- **Trees**: 4-8% of N^2 (excellent compression)
- **Complete graphs**: ~1.9x N^2 (expansion, not compression)
- **Greedy ~ Canonical**: only 3 total chars saved by exhaustive search

## Running

```bash
# Local (quick)
python -m benchmarks.string_length_analysis.string_length_analysis \
    --seed 42 --max-nodes 50 --output-dir /tmp/sla --csv --plot --table

# Local (large, no canonical)
python -m benchmarks.string_length_analysis.string_length_analysis \
    --seed 42 --max-nodes 200 --no-canonical --csv --plot --table

# Picasso (via SLURM)
bash slurm/launch.sh --benchmark string_length_analysis
```
