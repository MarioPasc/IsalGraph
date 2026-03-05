# Random Round-Trip Benchmark

## Purpose

Validates the **round-trip property** of IsalGraph:

> For any valid instruction string w, S2G(w) is isomorphic to S2G(G2S(S2G(w), v0)).

This is the fundamental correctness guarantee: encoding a graph as a string and decoding it back recovers the original graph (up to isomorphism).

## Method

1. **Random strings**: Generate random valid instruction strings (length 1-100) and verify the round-trip.
2. **NetworkX families**: Convert graphs from diverse families (trees, GNP, Barabasi-Albert, cycles, complete, star, wheel, ladder, Watts-Strogatz, Petersen) to IsalGraph strings and back.
3. **Cross-validation**: Every isomorphism check is verified by both `SparseGraph.is_isomorphic()` and `nx.is_isomorphic()`.
4. **Both directions**: Tests run with `directed=True` and `directed=False`.

## Statistical Tests

- **Clopper-Pearson 95% CI** for pass rate per graph family (exact binomial).
- **Kruskal-Wallis test** for execution time differences across families.

## Figure: `random_roundtrip_figure.pdf`

**Layout**: 1x2, double-column width.

| Panel | Content |
|-------|---------|
| (a) | Bar chart: pass rate (%) by graph family with 95% CI error bars |
| (b) | Box plot: execution time per test (log scale) by family |
| Inset | Colored IsalGraph string from longest successful round-trip |

## Table: `random_roundtrip_table.tex`

| Column | Description |
|--------|-------------|
| Family | Graph family name |
| N_tests | Number of tests for this family |
| Pass_rate | Percentage of successful round-trips |
| 95% CI | Clopper-Pearson confidence interval |
| Mean_time_ms | Mean execution time in milliseconds |
| Median_time_ms | Median execution time in milliseconds |
| Max_nodes | Largest graph tested in this family |
| Max_edges | Maximum edge count in this family |

## Expected Results

From local testing (seed=42, 945 tests):
- **100% pass rate** across all families
- Trees and stars: fastest (< 1ms)
- Complete graphs: slowest (dense adjacency)
- Directed GNP: some graphs skipped (reachability constraint -- expected)

## Running

```bash
# Local (quick)
python -m benchmarks.random_roundtrip.random_roundtrip \
    --num-tests 100 --seed 42 --output-dir /tmp/rr --csv --plot --table

# Local (parallel)
python -m benchmarks.random_roundtrip.random_roundtrip \
    --num-tests 1000 --seed 42 --n-workers 8 --csv --plot --table

# Picasso (via SLURM)
bash slurm/launch.sh --benchmark random_roundtrip
```
