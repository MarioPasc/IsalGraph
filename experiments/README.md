# IsalGraph Experiment Registry

Single authoritative map from **paper artifact** to **the code that produces it**.
If a figure, table or number appears in the manuscript, its provenance is in this file.

Manuscript sources: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199`
(`main.tex`, `introduction.tex`, `methodology.tex`, `computational_experiments.tex`,
`results.tex`, `conclusion.tex`).

---

## 1. Layout: orchestration vs. routines

```
experiments/            ORCHESTRATION -- what runs, where, with which resources
  paper_pipeline/         Real-data pipeline for the paper (steps 1-4). CANONICAL.
    config.yaml             Frozen experiment config; copied into each run dir.
    launch.sh               SLURM dependency chain / --local / --dry-run.
    workers/step*.sh        One worker per pipeline step.
    generate_figures.py     Step 4: turns run-dir data into figures and tables.
  synthetic_suite/        Synthetic validation benchmarks (NOT in the paper).
    config.yaml             Per-benchmark blocks.
    launch.sh               Per-benchmark submission.
    workers/*_{slurm,login}.sh

benchmarks/             ROUTINES -- the Python that does the science
  real_data/              Invoked by paper_pipeline
  synthetic_data/         Invoked by synthetic_suite
  <name> -> real_data/<name> or synthetic_data/<name>   (symlinks)
```

**The symlinks are load-bearing.** Every worker invokes `python -m benchmarks.<name>.<name>`,
which resolves through `benchmarks/<name>` into `real_data/` or `synthetic_data/`. Do not
delete them and do not turn them into real directories.

**`experiments/paper_pipeline/` is canonical.** `experiments/synthetic_suite/` (formerly the
top-level `slurm/`) predates it and contains five older near-duplicates of pipeline steps
(`eval_setup`, `eval_correlation`, `eval_computational`, `eval_encoding`,
`topology_complexity_figs`). Those duplicates are retained only so old run directories remain
interpretable. **Never launch a paper step from `synthetic_suite/`** -- it reads a different
config schema (`benchmarks.<name>` rather than `steps.<name>`) and does not inject
`ISALGRAPH_RUN_DIR`.

---

## 2. Pipeline DAG

Run directory: `<runs_dir>/<TIMESTAMP>_<GIT_HASH>[_dirty]`, with `config.yaml` frozen into it
(stamped with `meta.git_commit`) and `conda list --export` recorded alongside.

| Job | Worker | Module invoked | Consumes | Produces | SLURM dependency |
|---|---|---|---|---|---|
| step1 | `step1_eval_setup.sh` | `benchmarks.eval_setup.eval_setup` | `paths.source_dir` (raw IAM / GraphEdX) | `<run>/data/`: metadata, `canonical/`, `levenshtein/`, `wl/`, `comparison/`, `filtering_report.json` | none |
| step2a | `step2a_eval_correlation.sh` | `benchmarks.eval_correlation.eval_correlation` | `<run>/data` | `<run>/correlation/stats/` | `afterok:step1` |
| step2b | `step2b_eval_computational.sh` | `benchmarks.eval_computational.eval_computational` | `<run>/data`, `source_dir` | `<run>/computational/` | `afterok:step1` |
| step2c | `step2c_eval_encoding.sh` | `benchmarks.eval_encoding.eval_encoding` | synthetic (generated in-process) | `<run>/encoding/raw/` | none |
| step2d | `step2d_eval_message_length.sh` | `benchmarks.eval_message_length.eval_message_length` | `<run>/data` | `<run>/message_length/{raw,stats}/` | `afterok:step1` |
| step3a | `step3a_algorithm_figures.sh` | `benchmarks.eval_visualizations.illustrative.algorithm_figures` | none | `figures/_intermediate/algorithm/` | none |
| step3b | `step3b_topology_figs.sh` | `benchmarks.eval_visualizations.illustrative.topology_and_complexity` | none | `figures/_intermediate/topology/` | none |
| step4 | `step4_generate_figures.sh` | `experiments/paper_pipeline/generate_figures.py --run-dir <RUN_DIR>` | all of the above | `<run>/figures/*.pdf`, `*.tex`, `*_caption.txt` | `afterok:` 2a,2b,2c,2d,3a,3b |

Launcher flags: `--dry-run`, `--local` (sequential, `runs/` under the repo), `--step <name>`,
`--config <path>`.

**Failure semantics to be aware of:** `generate_figures.py::generate_all` wraps each figure in
`try/except` and appends to an `errors` list. A figure that fails to render does **not** fail the
job. Always read the step-4 log, never just its exit code.

---

## 2b. Competitor representations (`isalgraph.competitors`, T-04)

Added for the *Pattern Recognition* revision. **T-04 ships the machinery; it does not run
the science** — T-04a runs the grid, T-06 the production matrices, T-17 the AE.3 table.
Full API: [`src/isalgraph/competitors/README.md`](../src/isalgraph/competitors/README.md).

| Competitor | Serves | Feeds | Entry point |
|---|---|---|---|
| `adjacency`, `graph6`, `nauty_graph6`, `agm_cam` | AE.3, R1.2a/b — the `n²` family, which isolates **canonicity at fixed format** | Claim A **one row + footnote**, not four identical columns; Claim B | `grid`, `f5` |
| `sparse6`, `sparse6_nauty` | R3.6a — IsalGraph's only genuine rival on message length | Claim A; `sparse6_nauty` is **supplementary**, not a preregistered comparator | `grid`, `f5` |
| `min_dfs` | R1.1, R1.2 — gSpan, named by R1 by name | Claim A, Claim B, Fig. 2 (**language-matched**) | `grid`, `f5` |
| `wl_subtree` | R1.2b — the completeness witness | Claim B **only**; no bit count, cell empty with the reason printed | `grid`, `f5` |
| `isalgraph_pruned`, `isalgraph_canonical` | the reference arm | every table | `grid`, `f5` |
| `size_null` | **finding 1** — `ρ(\|n₁−n₂\|, GED)` = 0.71–0.93 | a **baseline row beside every printed ρ**; outside the confirmatory family (decision 23) | `f5` |

| Artefact | Produced by | Consumed by |
|---|---|---|
| `repro_artefacts.json` | `reproduce --mode artefacts` | the provenance record — all 40 Suite-1 cells at delta `0.00e+00` |
| `corrected_rho_table.json` | `reproduce --mode table` | **T-06, T-17, T-20** — supersedes `competitors/README.md` §4.1/§4.2 |
| `smoke_picasso_suite{1,2}.json` | `smoke` on loginexa | the `pynauty` from-source gate |
| `agm_ceiling_B.json` | track B | AGM's 76 % / 82 % ceiling, `agm.md` §2.2b |

> **Fig. 2 must be language-matched.** Timing a pure-Python min-DFS against the C++ engine
> reproduces R1.1's own complaint inside our answer to it. Every smoke header records
> `isalgraph_engine`, so a timing cannot be quoted without it. Measured on Picasso with both
> arms in Python, GREC: `min_dfs` 1.03 ms/graph against `isalgraph_pruned` 17.6 ms/graph.

---

## 3. Artifact registry (figures and tables in the submitted PDF)

| Paper artifact | Section | Generating code | Upstream step | Controlling config |
|---|---|---|---|---|
| `fig:message_length_scatter` (`fig_message_length_scatter_log.pdf`) | results §res-info-content | `eval_visualizations/fig_message_length.py::generate_scatter_figure(log_counts=True)` | step2d | `steps.eval_message_length.{ged_schemes,isal_schemes}`, `experiment.{datasets,algorithms}` |
| `tab:information-content` | results | `eval_visualizations/fig_message_length.py::generate_information_content_table` | step2d | as above |
| `fig:empirical-complexity` (`fig_complexity_ratio_combined.pdf`) | results §res-complexity | `eval_visualizations/fig_empirical_complexity.py` (combined figure; power-law fit `_fit_polynomial`) | step2c + step2b + step2d | `steps.eval_encoding.{n_instances,n_reps,max_n_greedy,max_n_canonical}`, `steps.eval_setup.timeout_per_graph`, `steps.eval_computational.n_timing_reps` |
| `fig:heatmap-correlation-ged-lev` (`fig_aggregated_density_correlation.pdf`) | results §res-correlation | `eval_visualizations/population_figures/central_heatmap.py` | step2a | `steps.eval_correlation.{n_bootstrap,n_permutations}`, `experiment.seed` |
| `tab:performance-summary` | results | `eval_visualizations/table_performance_summary.py::generate_performance_table` | step2a | `steps.eval_correlation.*`, `experiment.algorithms` |
| `fig:g2s-method-comparison` (`fig_composite_method_tradeoff_v2.pdf`) | results §res-tradeoff | `eval_visualizations/composite_method_tradeoff.py::generate_composite_method_tradeoff_v2` | step2a + step2b | `steps.eval_correlation.*`, `steps.eval_computational.{n_timing_reps,n_pairs_per_bin}` |
| `tab:instructions` | methodology | **hand-written LaTeX** -- no generating code | -- | -- |

**Present in the repo but commented out of the submitted PDF:**
`fig_algorithm_overview` (`illustrative/algorithm_figures.py`),
`fig_shortest_path_comparison` (`illustrative/shortest_path_comparison.py`),
`fig_neighborhood_topology` (`illustrative/topology_and_complexity.py`),
`graphical_abtract` (`eval_visualizations/graphical_abstract/compose_graphical_abstract.py`).
These are candidates for reinstatement in the revision.

### Known reproducibility discrepancies

These are real defects. They are recorded rather than silently patched, because fixing a
filename changes what a rebuild emits and must be done deliberately.

1. **Figure basename mismatch.** The manuscript includes `fig_composite_method_tradeoff_v2.pdf`;
   `composite_method_tradeoff.py` emits `composite_method_tradeoff_v2.pdf` (no `fig_` prefix).
   The submitted PDF's copy was renamed by hand.
2. **`config.yaml:paper_outputs` is out of sync.** It lists `fig_empirical_complexity`, but the
   paper uses `fig_complexity_ratio_combined`; and it omits `table_information_content`, which is
   the source of Table 1. Corrected in §6 below.
3. **Tables were hand-edited after generation.** `table_performance_summary.py` emits a caption
   reading "Dataset properties and Spearman rho correlation ..."; the printed caption drops the
   properties block and adds the `Delta rho` column. Table 1 as printed merges dataset properties
   into the message-length table. Values are traceable to code; the LaTeX is not byte-reproducible.
4. **Paper-side derivations.** The "53%-74% of the bits" range and the alpha-exponent narrative are
   arithmetic performed in the text, not emitted by any script.

---

## 4. Reported numbers and their source

| Claim | Value | Computed by |
|---|---|---|
| Shorter bit representation for 98.8%-99.6% of graphs | 98.8-99.6% | `fig_message_length.py::generate_information_content_table` |
| Graph counts 1,180 / 1,253 / 2,059 / 89 / 769 | -- | `eval_setup/dataset_filter.py::filter_graphs` -> `filtering_report.json` |
| Pair counts 695,610 / 784,378 / 2,118,711 / 3,916 / 295,296 | -- | `table_performance_summary.py::_load_dataset_props` |
| Mean edge counts 3.07 / 3.17 / 4.56 / 8.35 / 10.70 | -- | `table_performance_summary.py::_load_dataset_props` |
| OLS slopes beta = 0.537 / 0.538 / 0.590, R^2 >= 0.940 | -- | `generate_information_content_table` (pooled fit) |
| Median ratios r-tilde 1.348-1.893, all significant | -- | same fn; `scipy.stats.wilcoxon(ratios - 1, alternative="greater")` |
| Complexity exponents alpha = 3.1 / 4.5 / 4.9 / 10.2 (n = 3-20) | -- | `fig_empirical_complexity.py::_fit_polynomial` (OLS on log-log) |
| Spearman rho grid (0.934 ... 0.251) | -- | `eval_correlation/correlation_metrics.py::bootstrap_correlation`, `mantel_test` |
| Delta rho = 0.027 / 0.014 / 0.057 / 0.228 | -- | `table_performance_summary.py::generate_performance_table` |
| Aggregate 3,424,764 pairs | -- | `population_figures/central_heatmap.py` via `result_loader.load_all_results` |
| Joint-density OLS slopes beta = 0.80 / 0.78 / 0.82 | -- | `central_heatmap.py` per-panel OLS |
| Speedups 48x (n=3) to >14,000x (n=11); Greedy-Min 3x-901x; Canonical 2.1x-563x | -- | `eval_computational.py::_analyze_crossover`, `_compute_amortized`; aggregated in `composite_method_tradeoff.py::_draw_panel_a_v2` |
| Feasible up to ~12 nodes | 12 | `steps.eval_setup.n_max` |
| Scales to 50 nodes within 600 s | 50 / 600 | `steps.eval_encoding.max_n_greedy`, `steps.eval_setup.timeout_per_graph` |
| log2(9) ~ 3.17 bits; `B_GED = (N-1+M) + 2M*ceil(log2 N)` | -- | `eval_message_length/message_length_computer.py` (analytic) |
| 25 timing reps, 5 instances, seed 42 | -- | `steps.eval_computational.n_timing_reps`, `steps.eval_encoding.{n_reps,n_instances}`, `experiment.seed`; `eval_computational/timing_utils.py::time_function` uses `time.process_time()` |

---

## 5. Datasets

| Dataset | Source | Loader | Filter | Graphs / mean edges / pairs |
|---|---|---|---|---|
| IAM Letter LOW | Riesen & Bunke (2008), GXL/CXL | `eval_setup/iam_letter_loader.py::load_iam_letter(level='LOW')` | `dataset_filter.py::filter_graphs(n_max=12, require_connected=True)` | 1,180 / 3.07 / 695,610 |
| IAM Letter MED | same | `load_iam_letter(level='MED')` | same | 1,253 / 3.17 / 784,378 |
| IAM Letter HIGH | same | `load_iam_letter(level='HIGH')` | same | 2,059 / 4.56 / 2,118,711 |
| LINUX | Bai et al. (2019), via GraphEdX (Jain et al. 2024) | `eval_setup/graphedx_loader.py::load_graphedx_dataset('LINUX')` | same | 89 / 8.35 / 3,916 |
| AIDS | NCI DTP, topology-only variant via GraphEdX | `load_graphedx_dataset('AIDS')` | same | 769 / 10.70 / 295,296 |
| Synthetic (complexity only) | BA m in {1,2}; ER p in {0.3,0.5} | `eval_encoding/synthetic_generator.py::generate_graph_family` | n = 3..50 greedy, 3..20 canonical | not tabulated |

**GED ground truth is not homogeneous across datasets.** IAM uses NetworkX A* with unit
node/edge insert-delete costs (node substitution 0), computed in `eval_setup/ged_computer.py`.
LINUX and AIDS use GraphEdX precomputed **topology-only** matrices with zero node-operation cost.
Reviewer 3 raises this directly; aggregating the three families into one correlation figure is
therefore only defensible with the dataset-level correlations treated as primary evidence.

Node and edge attributes are discarded by `graphedx_loader._strip_node_attributes`. For AIDS this
means atom and bond types are dropped -- Reviewer 1's point about label loss on molecular graphs
applies exactly here, and the code confirms it.

The authoritative record of graph counts is `<data_root>/filtering_report.json`, written by
`eval_setup.py`, not the manuscript.

---

## 6. Corrected `paper_outputs`

`experiments/paper_pipeline/config.yaml:paper_outputs` is the checklist step 4 is expected to
produce. The list below is the set actually consumed by the submitted manuscript:

```
composite_method_tradeoff_v2        # NOTE: paper file is fig_composite_method_tradeoff_v2.pdf
fig_aggregated_density_correlation
fig_complexity_ratio_combined
fig_message_length_scatter_log
table_information_content
table_performance_summary
```

Emitted but commented out of the submission (retain for the revision):
`fig_algorithm_overview`, `fig_algorithm_overview_full`, `fig_shortest_path_comparison`,
`fig_neighborhood_topology`, `graphical_abtract`.

---

## 7. Code that produces no paper artifact

Kept because it is validation or exploratory work, but it must not be mistaken for a paper
routine. Nothing here appears in the submitted PDF.

**Synthetic validation suite** (all of `benchmarks/synthetic_data/`, driven by
`experiments/synthetic_suite/`): `random_roundtrip`, `canonical_invariance`,
`string_length_analysis`, `levenshtein_vs_ged`, `greedy_optimality_gap`,
`starting_node_sensitivity`, `roundtrip_fixed_point`, `alphabet_entropy`, `string_pipeline`.
These verify the properties in `src/isalgraph/core/README.md` §6 and are the natural place to
extend scale-related evidence for the revision.

**Embedding track** (unused): `benchmarks/real_data/eval_embedding/` plus
`eval_visualizations/generate_embedding_figures.py`, `population_figures/embedding_figures.py`,
`individual_figures/embedding_examples.py`, `embedding_loader.py`.

**Superseded figure drivers** (not imported by `generate_figures.py`):
`generate_correlation_figures.py`, `generate_computational_figures.py`,
`generate_encoding_figures.py`, `generate_illustrative_figures.py`.

**Unused emitters:** `fig_message_length.py::generate_ratio_figure`,
`fig_empirical_complexity.py`'s standalone `fig_empirical_complexity`,
`topology_and_complexity.py`'s `fig_distance_field`,
`algorithm_figures.py::{fig_s2g_walkthrough, fig_g2s_walkthrough}`,
`composite_method_tradeoff.py::generate_composite_method_tradeoff` (v1),
`eval_message_length.py`'s in-module plotting (shadowed by `fig_message_length.py`),
`table_generator.py::generate_dual_table`, `graphical_abstract/*`,
`individual_figures/{correlation_examples,derived_examples}.py`,
`population_figures/{correlation_figures,derived_figures}.py`.

**Computed but never reported:** the WL-kernel baseline
(`eval_setup/wl_kernel_computer.py`, `experiment.distance_metrics: [levenshtein, wl_kernel]`,
`wl_kernel.n_iter: 5`), and the unpruned `canonical` algorithm -- `experiment.algorithms` lists
it, but the paper reports only `canonical_pruned`, `greedy_min` and `greedy_single`.
The WL-kernel results are an existing, already-computed comparison baseline; Reviewer 1 asks for
comparison against alternative approaches, so this is low-hanging fruit for the revision.

**Drawing primitives now superseded by `isalgraph.viz`:**
`eval_visualizations/{cdll_drawing,graph_drawing,string_alignment}.py` and
`benchmarks/plotting_styles.py`. New figure code must import from `isalgraph.viz`; see
`src/isalgraph/viz/README.md`.

---

## 8. Reproducing the paper

```bash
conda activate isalgraph

# Dry run: print the SLURM chain without submitting.
bash experiments/paper_pipeline/launch.sh --dry-run

# Full submission on Picasso.
bash experiments/paper_pipeline/launch.sh

# One step, locally.
bash experiments/paper_pipeline/launch.sh --local --step eval_setup
```

Synthetic validation benchmarks are launched separately and are not part of the paper build:

```bash
bash experiments/synthetic_suite/launch.sh --dry-run
bash experiments/synthetic_suite/launch.sh --benchmark canonical_invariance
```
