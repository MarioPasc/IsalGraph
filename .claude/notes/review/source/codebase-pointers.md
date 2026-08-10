# Codebase and results pointers

Where the implementation, the datasets and the result artifacts live, for anyone who has to re-run
or re-measure something in response to a reviewer comment.

**Repo**: `/home/mpascual/research/code/IsalGraph`
**Manuscript**: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/article/69b82c5859ed47c5468ca199`
**Data and results root**: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/`
**Review material**: `.claude/notes/review/` (`source/` = this package, `tasks/` = work items)

## Read this first, do not duplicate it

**`/home/mpascual/research/code/IsalGraph/experiments/README.md`** (243 lines) is the
reproducibility registry: the single authoritative map from paper artifact to producing code. It is
maintained, it is more detailed than anything here, and this file does not restate it.

| Section | Line | What it answers |
|---|---|---|
| 1. Layout: orchestration vs. routines | `:12` | why `experiments/` and `benchmarks/` are split |
| 2. Pipeline DAG | `:46` | step1 -> step4, inputs and outputs per step |
| **3. Artifact registry** | `:71` | **which script emits which figure or table** |
| **Known reproducibility discrepancies** | `:90` | four real defects, deliberately unpatched |
| **4. Reported numbers and their source** | `:110` | **every number in the paper, traced to a function** |
| 5. Datasets | `:133` | provenance, loaders, filters, counts |
| 6. Corrected `paper_outputs` | `:159` | `config.yaml`'s list is out of sync |
| 7. Code that produces no paper artifact | `:179` | dead code |
| 8. Reproducing the paper | `:223` | how to re-run |

Two of its entries bear directly on reviewer comments: `:144–148` (heterogeneous GED cost models,
R3.5b) and `:150–152` (label stripping on AIDS, R1.3). Both were reached independently of the
review.

## Environments

Two conda environments, both Python 3.11.15.

| Env | Path | Contents |
|---|---|---|
| `isalgraph` | `/home/mpascual/.conda/envs/isalgraph` | pure Python, no compiled extension |
| **`isalgraph-cpp`** | `/home/mpascual/.conda/envs/isalgraph-cpp` | the same packages **plus the built C++ engine** |

```bash
PY=~/.conda/envs/isalgraph-cpp/bin/python
$PY -m pip install -e ".[dev,native]"   # builds the extension
$PY -m pytest tests/ -q                 # ~2.5 min with the engine
```

**Use `isalgraph-cpp` for anything that runs the encoder.** Reference test state (integration of
wave 2026-08-10): 726 passed / 271 skipped with the engine, 561 passed / 276 skipped without.

## Forcing an engine

The mechanism lives entirely in `src/isalgraph/core/backends.py`.

| Control | Where | Behaviour |
|---|---|---|
| `ISALGRAPH_ENGINE` env var | read at `backends.py:111`; values `cpp` \| `python` | falls back to `DEFAULT_BACKEND` at `:123` when unset |
| `backend=` keyword | on every dispatching function; type alias `backends.py:75` | **always wins over the env var** — `backends.py:172`, documented `:161–162` |
| Which engine is live | `isalgraph.engine()` -> `'cpp'` or `'python'` | |
| Build provenance | `isalgraph.build_info()` | compiler, ISA level, `build_hash` — detects a stale `.so` |

**It never degrades silently.** `BackendError` (`src/isalgraph/errors.py:106`) is raised when
`backend='cpp'` is requested and the extension is absent (`backends.py:175`, comment at `:174`:
"An explicit request never silently degrades"), when `ISALGRAPH_ENGINE=cpp` and `_native` will not
import (`:114`), and on an unrecognised value (`:122`).

Two traps for anyone producing timings:

- **Never `export PYTHONPATH=$REPO/src`.** A src-first path shadows the installed package, the
  extension silently disappears, and the benchmark measures the pure-Python reference. Assert
  `isalgraph.engine() == "cpp"` in any script whose numbers you intend to report.
- `from isalgraph.core.canonical import canonical_string` gets the **pure-Python reference** and
  bypasses dispatch. Import `from isalgraph import canonical_string` to run on the active engine.

## The C++ engine

`isalgraph.core._native`, a nanobind extension built from `src/isalgraph/core/native/`
(`bindings.cpp` 268 L, `canonical.cpp` 420 L, `graph_to_string.cpp` 156 L, `sparse_graph.cpp` 137 L,
plus `string_to_graph`, `pairs`, `cdll`, `levenshtein`, `probe`). It postdates the submission and
appears nowhere in the manuscript.

**Byte-exact against the Python reference**, `docs/engineering/CPP_OPTIMIZATION_LOG.md`:
3,079 graphs and 4,000 string pairs, first run, zero mismatches (`:67`); canonical strings
1,180 compared / 0 mismatches (`:278`); Levenshtein 250 pairs / 0 mismatches (`:279`).

**End-to-end speedup** (`:224–231`, Erdos-Renyi `p = 0.35`, connected):

| nodes | canonical | pruned |
|---|---|---|
| 3 | 23.1x | 26.9x |
| 5 | 253.1x | 211.1x |
| 7 | 596.9x | 474.4x |
| 9 | **1024.7x** | 829.9x |
| 10 | 937.0x | 868.4x |

**Displacement-pair memoisation, isolated by A/B** (`:84–87`) — this is the direct measurement of
the recomputation R3.4b asks about:

| n | memo on | memo off | gain |
|---|---|---|---|
| 6 | 0.00073 s | 0.0186 s | 25.5x |
| 8 | 0.0182 s | 0.762 s | 41.9x |
| 9 | 0.0620 s | 3.573 s | 57.6x |
| 10 | 0.219 s | 23.83 s | 108.6x |

`:91` — "This single change accounts for most of the speedup." Toggle with
`_native.set_pairs_memo(bool)`; outputs are identical either way.

**Threading defaults to 1, deliberately**: 4 threads are 0.55x (slower) at `n = 6` and peak at
1.49x at `n = 9` on a 4x core budget, ~34% efficiency (`:180–188`). The paper's graphs average under
4 nodes.

Build flags are `-march=x86-64-v3`, never `-march=native` — Picasso is heterogeneous and `native`
produces SIGILL on a fraction of nodes. The `.so` installs into site-packages, so **it does not
rsync**; build it on the cluster as part of environment setup.

## Core implementation — `src/isalgraph/core/`

| Path | Lines | Role | Reviewer relevance |
|---|---|---|---|
| `graph_to_string.py` | 398 | G2S encoder | **R3.4a** `C`/`c` branches at `:208–221`, `:223–238`; **R3.4b** pair recompute at `:155`; **R3.3a** `_check_reachability` at `:305–340`; `generate_pairs_sorted_by_sum` at `:41` |
| `canonical.py` | 350 | exhaustive canonical | **R3.4b** pair recompute at `:223` inside `_step` (`:202`); `C`/`c` at `:313–346` |
| `canonical_pruned.py` | 365 | triplet-pruned canonical | **R3.4b** pair recompute at `:226` inside `_pruned_step` (`:204`); `C`/`c` at `:326–361` |
| `string_to_graph.py` | 306 | S2G decoder | **R3.3c** `directed_graph` / `directed` constructor args at `:57–61` |
| `sparse_graph.py` | 209 | adjacency-set graph | `__init__(max_nodes: int, directed_graph: bool)` — the flag again |
| `cdll.py` | 129 | circular doubly-linked list | |
| `backends.py` | 603 | engine dispatch | see above |
| `trace.py` | 310 | `StepSnapshot` / `AlgorithmTrace`, stdlib only | |
| `README.md` | 673 | full mathematics and architecture | |
| `native/` | ~1,400 C++ | the engine | |

**The Python reference in `{canonical, canonical_pruned, cdll, sparse_graph, string_to_graph,
graph_to_string}.py` is frozen.** It is what the differential suite compares the C++ engine
against; changing it means re-proving parity.

## Experiment orchestration — `experiments/`

```
experiments/
├── README.md              THE REGISTRY -- read it first
├── paper_pipeline/        the real-data paper pipeline, steps 1-4. CANONICAL.
│   ├── config.yaml        121 lines -- every knob in the paper
│   ├── launch.sh
│   ├── generate_figures.py
│   └── workers/           step1_eval_setup.sh, step2a..2d, step3a/3b, step4
└── synthetic_suite/       synthetic validation. NOT in the paper.
```

`experiments/paper_pipeline/config.yaml` — the values reviewers ask about:

| Key | Value | Line | Comment |
|---|---|---|---|
| `experiment.seed` | `42` | `:30` | matches `computational_experiments.tex:255` |
| `experiment.algorithms` | `[canonical, canonical_pruned, greedy_min, greedy_single]` | `:31` | the paper reports 3 of these 4 |
| `experiment.distance_metrics` | `[levenshtein, wl_kernel]` | `:32` | **WL is computed and never reported** |
| `experiment.wl_kernel.n_iter` | `5` | `:34` | |
| `experiment.datasets` | `[iam_letter_low, iam_letter_med, iam_letter_high, linux, aids]` | `:35` | |
| **`eval_setup.n_max`** | **`12`** | `:40` | **AE.1, R3.7 — the graph-size ceiling** |
| `eval_setup.timeout_per_graph` | `600` s | `:41` | |
| `eval_correlation.n_bootstrap` | `10000` | `:48` | **R3.5c — the undescribed bootstrap** |
| `eval_correlation.n_permutations` | `9999` | `:49` | Mantel test, never reported |
| `eval_computational.n_timing_reps` | `25` | `:56` | |
| `eval_computational.n_pairs_per_bin` | `50` | `:57` | |
| `eval_encoding.max_n_greedy` | `50` | `:66` | **E3 — fits declared over n=3..20** |
| `eval_encoding.max_n_canonical` | `20` | `:67` | |
| `topology_figs.exhaustive_timeout` | `60.0` s | `:92` | |

**`config.yaml:paths.*` (`:20–22`) point at Picasso**, not local disk. `config.yaml:paper_outputs`
is out of sync with the manuscript — corrected list at `experiments/README.md:159`.

## Routines — `benchmarks/`

`benchmarks/<name>` are **symlinks** into `real_data/` or `synthetic_data/`; every worker invokes
`python -m benchmarks.<name>.<name>` through them. Do not delete them or convert them to real
directories. The only real file at that level is `benchmarks/plotting_styles.py`, which re-exports
from `isalgraph.viz.style` so the published palette cannot drift (a test asserts byte-identity with
the submitted PDF).

### `benchmarks/real_data/eval_setup/` — step 1, where the data enters

| File | Role | Reviewer relevance |
|---|---|---|
| `eval_setup.py` | CLI orchestrator; `IAM_DATASETS` at `:81`; GED provenance tags at `:244–245` | |
| `iam_letter_loader.py` | IAM GXL/CXL loader, `load_iam_letter(level=)` | **R1.3 / D7** — `:4`, `:60` "Node attributes (x, y coordinates) are stripped" |
| `graphedx_loader.py` | LINUX/AIDS loader | **R1.3 / D7** — `_strip_node_attributes` at `:82–88` drops atom and bond types |
| `ged_computer.py` | all-pairs exact GED, NetworkX A*, **IAM only** | **R3.5b / D9** — the uniform-cost half |
| **`wl_kernel_computer.py`** | **Weisfeiler-Lehman subtree kernel distances** | **R1.1 / AE.4 / E10 — computed, never reported** |
| `levenshtein_computer.py` | all-pairs Levenshtein matrices | |
| `canonical_computer.py` | exhaustive + greedy-min canonical | |
| `pruned_exhaustive_computer.py` | triplet-pruned canonical | |
| `greedy_single_computer.py`, `generate_greedy_single.py` | Greedy-rnd(`v_0`) | |
| `dataset_filter.py` | `filter_graphs(n_max=12, require_connected=True)` | **AE.1 / D17** — writes `filtering_report.json` |
| `method_comparator.py`, `validator.py` | cross-method comparison, artifact validation | |

### Other `benchmarks/real_data/` modules

| Directory | Entry point | Produces |
|---|---|---|
| `eval_correlation/` | `eval_correlation.py`; `correlation_metrics.py` holds **`bootstrap_correlation`** and **`mantel_test`** | Spearman grid — **R3.5c** |
| `eval_computational/` | `eval_computational.py` (1,533 L); `_analyze_crossover`, `_compute_amortized`; `timing_utils.py::time_function` uses `time.process_time()` | speedups — **R1.1** |
| `eval_encoding/` | `eval_encoding.py` (1,355 L); `synthetic_generator.py`; `compute_synthetic_ged.py` | complexity data — **R3.4c**, **E3** |
| `eval_message_length/` | `eval_message_length.py`; `message_length_computer.py` (analytic `B_GED`) | **R3.6a** |
| `eval_embedding/` | `eval_embedding.py`, `embedding_methods.py` | no paper artifact |

### `benchmarks/real_data/eval_visualizations/` — figures and tables

| Producer | Artifact |
|---|---|
| `fig_message_length.py::generate_scatter_figure(log_counts=True)` | Figure 1 |
| `fig_message_length.py::generate_information_content_table` | **Table 2** |
| `fig_empirical_complexity.py` (790 L), **`_fit_polynomial`** | **Figure 2** and the exponents of **D1** |
| `population_figures/central_heatmap.py` (814 L) | **Figure 3**, and the 3,424,764 aggregate of **E2** |
| `table_performance_summary.py` (412 L), `_load_dataset_props` | **Table 3**, and the pair counts of **E2** |
| `composite_method_tradeoff.py::generate_composite_method_tradeoff_v2` | Figure 4 |
| `illustrative/` | the three figures commented out of the PDF |
| `graphical_abstract/` | graphical abstract |

Table 1 (`tab:instructions`) is **hand-written LaTeX with no generating code**
(`experiments/README.md:81`) — relevant to D5, since the Table-1 side of the mismatch has no source
of truth outside the `.tex`.

`grep -rn wl_kernel benchmarks/real_data/eval_visualizations/` returns nothing: the WL baseline
reaches no figure and no table.

## Visualization — `src/isalgraph/viz/`

All figures go through this package; do not hand-roll matplotlib in a figure script. Full API in
`src/isalgraph/viz/README.md`.

Directly relevant: **`src/isalgraph/viz/search_tree.py`**, `canonical_search_tree_figure` — a
canonical search-space schematic, described in `.claude/CLAUDE.md` as built for Reviewer 3. This is
the renderer for the figure **R3.7c** asks for. Recorded as an existing asset.

Contracts that matter: every third-party import lives inside a function body (a test enforces that
`import isalgraph.viz` succeeds with matplotlib uninstalled); a backend never creates a figure, it
paints on a caller-supplied `Axes`; traces carry graph node ids already resolved from CDLL indices.
G2S traces are produced by **replaying the emitted string**, not by instrumenting the encoder — so
a G2S step figure cannot show tentative pointer walks, rejected displacement pairs or the priority
cascade. For decision structure use `search_tree`.

## Data on disk

```
/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/
├── article/69b82c5859ed47c5468ca199/   manuscript sources (own git repo)
├── data/source/                        AIDS/  Letter/  LINUX/
├── data/eval/                          canonical_strings/ datasets/ ged_matrices/
│                                       graph_metadata/ levenshtein_matrices/
│                                       method_comparison/ filtering_report.json
│                                       validation_report.json
├── results/article/                    final_artifacts/  runs/
├── results/preprint/
└── paper_figures/graphical_abstract/
```

**`data/eval/filtering_report.json` is the authoritative record of graph counts**, not the
manuscript (`experiments/README.md:154–156`). It is where the answer to R3.5a's "how many pairs were
removed" should be checked first.

**IAM is present on this machine.** `data/source/Letter/` — it is *not* Picasso-only, contrary to
what the config paths suggest. AIDS and LINUX are present too. A Picasso mirror is declared at
`config.yaml:21` (`.../fscratch/datasets/isalgraph_source`), and the pipeline as configured runs
there, but nothing about the data blocks local re-measurement.

**Stale-path warning.** Several `benchmarks/real_data/` modules default to
`/media/mpascual/Sandisk2TB/research/isalgraph/...` — e.g. `eval_correlation/eval_correlation.py:59–60`,
`eval_message_length/eval_message_length.py:36–37`, `eval_setup/generate_greedy_single.py:113,118`.
**That directory does not exist**; the live tree is the `ISAL/completed/isalgraph/` path above.
Any re-run must pass the root explicitly.

## Dataset provenance

From `experiments/README.md:133–157`. Counts match `results.tex:36–38` exactly.

| Dataset | Source | Loader | Graphs / mean edges / pairs |
|---|---|---|---|
| IAM Letter LOW | Riesen & Bunke (2008), GXL/CXL | `load_iam_letter(level='LOW')` | 1,180 / 3.07 / 695,610 |
| IAM Letter MED | same | `level='MED'` | 1,253 / 3.17 / 784,378 |
| IAM Letter HIGH | same | `level='HIGH'` | 2,059 / 4.56 / 2,118,711 |
| LINUX | Bai et al. (2019) via GraphEdX | `load_graphedx_dataset('LINUX')` | 89 / 8.35 / 3,916 |
| AIDS | NCI DTP, topology-only via GraphEdX | `load_graphedx_dataset('AIDS')` | 769 / 10.70 / 295,296 |
| Synthetic | BA `m in {1,2}`; ER `p in {0.3,0.5}` | `synthetic_generator.py::generate_graph_family` | `n = 3..50` greedy, `3..20` canonical |

All five filtered by `dataset_filter.py::filter_graphs(n_max=12, require_connected=True)`.
Those pair counts are **pre-filter** — see `verified-discrepancies.md` E2.

## Tests

| Directory | `test_*.py` |
|---|---|
| `tests/unit/` | 11 |
| `tests/eval_validation/` | 7 |
| `tests/integration/` | 5 |
| `tests/native/` | 5 — C++ parity and differential suite |
| `tests/viz/` | 4 |
| `tests/property/` | 2 — hypothesis |

Fixtures in `tests/conftest.py`. `tests/property/` is the natural home for a directedness-collision
regression (D3b); `tests/native/` is where parity is proven.

## HPC

Runs execute on **Picasso** (SCBI, UMA) — `computational_experiments.tex:252–254`. Paths under
`/mnt/home/users/tic_163_uma/mpascual/fscratch/`. SLURM time limits are declared per step in
`config.yaml:42–95` (setup `2-00:00:00`, correlation `12:00:00`, computational `08:00:00`, encoding
`04:00:00`, message length `00:30:00`, topology `12:00:00`).

Before writing or editing any SLURM script, use the **`picasso-sbatch`** skill — it is the source of
truth for partitions, constraint flags and wallclock limits, and it also covers building the native
extension on the cluster. For a fast pre-flight, **`test-picasso-loginexa`** validates a smoke run
on the V100 login node with no queue.

Remember the `.so` does not rsync: build the extension on the cluster, and never with
`-march=native`.

## Scale reference for re-measurement

- The correlation study is already 3.9M raw pairs; the largest single dataset, IAM HIGH, is
  2,118,711 pairs from 2,059 graphs.
- The canonical encoder is the bottleneck, not the pairwise distances: `n_max = 12` exists because
  canonicalisation is `n^{4.9}` empirically with an exponential worst case.
- **The C++ engine changes the feasible envelope by 23x–1025x.** Anything that was infeasible at
  submission time under pure Python should be re-costed before being called infeasible now — the
  submitted numbers were produced without it.
