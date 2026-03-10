# Canonical String Invariance Benchmark

## Purpose

Validates that the **canonical string** is a **complete graph invariant**:

> For isomorphic graphs G ~ H: canonical_string(G) == canonical_string(H)
> For non-isomorphic graphs G !~ H: canonical_string(G) != canonical_string(H)

The canonical string is computed via exhaustive backtracking search over all valid neighbor choices at V/v branch points, selecting the lexicographically minimal shortest string.

## Method

1. **Invariance tests** (~70%): Generate isomorphic pairs via random node relabeling, verify canonical strings match.
2. **Discrimination tests** (~30%): Generate structurally different graph pairs (cycle vs path, star vs path, complete vs cycle, different GNP densities), verify canonical strings differ.
3. **Graph families**: Trees, cycles, complete, star, wheel, Petersen, GNP, Barabasi-Albert, ladder.

## Statistical Tests

- **Clopper-Pearson 95% CI** for invariance and discrimination rates (separately).
- **OLS regression**: log(time) ~ N to characterize scaling (expected: exponential in N).

## Figure: `canonical_invariance_figure.pdf`

**Layout**: 1x2, double-column width.

| Panel | Content |
|-------|---------|
| (a) | Grouped bar chart: pass rate by family, grouped by test type (invariance/discrimination), with 95% CI error bars |
| (b) | Box/violin plot: computation time vs graph size N (log scale y-axis) |
| Inset | Colored canonical strings of an isomorphic pair showing equality |

## Table: `canonical_invariance_table.tex`

| Column | Description |
|--------|-------------|
| Family | Graph family name |
| Test_type | "invariance" or "discrimination" |
| N_tests | Number of tests |
| Pass_rate | Success rate |
| 95% CI | Clopper-Pearson confidence interval |
| Mean_time_s | Mean computation time per test |
| Max_N | Largest graph size tested |

## Expected Results

From local testing (seed=42, 71 tests, max_nodes=8):
- **100% invariance rate**: canonical strings always match for isomorphic pairs
- **100% discrimination rate**: always differ for non-isomorphic pairs
- Computation time scales exponentially with N (~2.5s average at N=8)

## Running

```bash
# Local (quick)
python -m benchmarks.canonical_invariance.canonical_invariance \
    --num-tests 50 --max-nodes 7 --seed 42 --output-dir /tmp/ci --csv --plot --table

# Local (parallel)
python -m benchmarks.canonical_invariance.canonical_invariance \
    --num-tests 200 --max-nodes 8 --n-workers 8 --csv --plot --table

# Picasso (via SLURM)
bash slurm/launch.sh --benchmark canonical_invariance
```
