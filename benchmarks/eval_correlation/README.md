# Eval Correlation: Levenshtein-GED Correlation Analysis

Statistical analysis validating IsalGraph's core claim that Levenshtein distance
between canonical strings approximates Graph Edit Distance (GED).

## Hypotheses

- **H1**: Spearman rho(Lev, GED) > 0 with Mantel permutation p < 0.001
- **H2**: rho decreases LOW -> MED -> HIGH (monotone degradation with noise)
- **H3**: Sparser graphs yield higher rho (density-stratified analysis)
- **H4**: Within-class Lev < between-class Lev (Cohen's d, IAM Letter only)
- **H5**: Exhaustive vs greedy comparison with bootstrap CI on delta-rho

## Usage

```bash
# Local smoke test (fast)
python -m benchmarks.eval_correlation.eval_correlation \
    --data-root /media/mpascual/Sandisk2TB/research/isalgraph/data/eval \
    --output-dir /tmp/eval_correlation_test \
    --datasets iam_letter_low \
    --n-bootstrap 100 --n-permutations 99 \
    --csv --plot --table

# Full local run
python -m benchmarks.eval_correlation.eval_correlation \
    --data-root /media/mpascual/Sandisk2TB/research/isalgraph/data/eval \
    --output-dir /tmp/eval_correlation_full \
    --n-bootstrap 500 --n-permutations 499 \
    --csv --plot --table

# Picasso submission
bash slurm/launch.sh --benchmark eval_correlation
```

## Output Structure

```
results/eval_correlation/
    raw/           # Pair-level CSVs
    stats/         # Per-dataset JSON statistics + cross-dataset analysis
    figures/       # Publication-quality PDF+PNG figures
    tables/        # LaTeX tables
```

## Modules

- `correlation_metrics.py`: Pure statistical functions (Mantel test, bootstrap CI,
  Lin's CCC, Precision@k, Jonckheere-Terpstra, Holm-Bonferroni)
- `eval_correlation.py`: CLI orchestrator, data loading, figure/table generation
