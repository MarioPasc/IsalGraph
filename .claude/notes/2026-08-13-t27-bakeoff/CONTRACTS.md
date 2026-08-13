# CONTRACTS — wave `2026-08-13-t27-bakeoff` (T-27 GED bound bake-off)

Written and owned by the **orchestrator**. Agents code against this file, never against each other.
If you believe a contract is wrong, **message the orchestrator** — do not negotiate with a peer and
do not silently deviate. A contract defect found early is a success.

Base commit: see the wave prompt. Design note: `.claude/notes/review/tasks/T-27-design.md`.

---

## 0. Paths — verified present 2026-08-13

```
DATA   = /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data
OUT    = /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-27-ged-bound-bakeoff
GEDLIB = ~/opt/build_gedlib/graphkit-learn          # in-place build; export PYTHONPATH to this
PY     = ~/.conda/envs/isalgraph-cpp/bin/python
```

| Input | Path | Keys used |
|---|---|---|
| Ground truth | `$DATA/source/GED_PRECOMPUTED/extended_merged_exact_ged/computed/{ds}.npz` | `ged_matrix`, `lb_matrix`, `ub_matrix`, `certified_mask`, `node_counts`, `edge_counts`, `graph_ids`, `metadata` |
| Graph topology | `$DATA/exported/{ds}.npz` | `graph_ids`, `n_nodes`, `n_edges`, `edge_offsets`, `edges` (shape `(2, sum n_edges)`) |
| Levenshtein (M6) | `$DATA/eval/levenshtein_matrices/{ds}_{variant}.npz` | `levenshtein_matrix` (int32, `n×n`), `graph_ids` |

`{ds}` ∈ `linux`, `aids`, `iam_letter_low`, `iam_letter_med`, `iam_letter_high`.
`{variant}` ∈ `exhaustive` (**primary**, the canonical string), `greedy`, `greedy_single`.

**Verified by the orchestrator, do not re-litigate**: for all five datasets the exported
`graph_ids` are element-wise identical to the ground-truth `graph_ids` **in the same order**, and
`n_nodes == node_counts`, `n_edges == edge_counts`. Levenshtein `graph_ids` match too.
**Assert this anyway at load time** — it is the single silent-corruption risk in the wave.

---

## 1. Canonical pair order — the spine of the whole wave

For a dataset with `n` graphs:

```python
pair_i, pair_j = np.triu_indices(n, k=1)     # int32; i < j; row-major
```

**Every array in every file below is in this order and has this length.** Never reorder, never
filter — masks select, they do not compact. `len == n*(n-1)//2`, which is 3,916 / 295,296 /
695,610 / 784,378 / 2,118,711 for the five datasets in the order above.

---

## 2. `$OUT/data/index/{ds}.npz` — written once by **Track A**, read by **Track B**

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `pair_i`, `pair_j` | int32 | `(P,)` | §1 order |
| `exact` | float64 | `(P,)` | ground-truth GED; **`inf` where censored** |
| `exact_lb`, `exact_ub` | float64 | `(P,)` | solver bracket; `exact_lb == exact_ub == exact` where certified |
| `certified` | bool | `(P,)` | from `certified_mask` |
| `n_max` | int32 | `(P,)` | `max(n_i, n_j)` |
| `lev_exhaustive`, `lev_greedy`, `lev_greedy_single` | int32 | `(P,)` | Levenshtein, §1 order |
| `graph_ids` | `<U…` | `(n,)` | graph order |
| `node_counts`, `edge_counts` | int32 | `(n,)` | per graph |
| `meta` | `<U…` | `()` | JSON, §4 |

## 3. `$OUT/data/cells/{ds}__{METHOD}.npz` — one per cell, **Track A** writes, **Track B** reads

`{METHOD}` is the GEDLIB name verbatim, upper case: `BRANCH`, `BRANCH_FAST`, `BRANCH_TIGHT`,
`STAR`, `IPFP`, `REFINE`, `BIPARTITE`, `BP_BEAM`, and `HED` if it yields a finite bound.

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `value` | float64 | `(P,)` | **the reported bound.** LB: the single-orientation value. UB: `min(value_fwd, value_rev)` |
| `value_fwd` | float64 | `(P,)` | orientation `(i, j)` |
| `value_rev` | float64 | `(P,)` | orientation `(j, i)`. **UB cells only**; LB cells omit this key entirely |
| `meta` | `<U…` | `()` | JSON, §4 |

**Rules that are not negotiable.**

- **Assert `0 <= value < inf` on every read from GEDLIB, per pair, per orientation.** `0` is legal
  only when `exact == 0`. A value of exactly `0.0` where `exact > 0` means the wrong accessor was
  called and **must raise**, not warn. This is the trap that silently fills a matrix with zeros.
- `end == "lower"` reads `get_lower_bound()`. `end == "upper"` reads `get_upper_bound()`. Never both.
- Import order: `importlib.import_module("gklearn.gedlib.libraries_import")` **before**
  `gklearn.gedlib.gedlibpy_gxl`. Use `importlib.import_module`, never a plain `from … import` —
  ruff/isort reorder those and break the `dlopen`.
- Edit costs: `set_edit_cost("CONSTANT", edit_cost_constant=[1, 1, 0, 1, 1, 0])`, every cell, no exceptions.
- `init(init_option="EAGER_WITHOUT_SHUFFLED_COPIES")`.
- `add_nx_graph` needs **string** node and edge attributes — attach a constant dummy label.
- A cell that fails writes **no `.npz`** and instead `$OUT/data/cells/{ds}__{METHOD}.failed.json`
  with `{"dataset","method","reason","traceback","options"}`. Failures are reported, never omitted.

## 4. `meta` JSON — same schema in index and cell files

```json
{
  "schema_version": 1,
  "wave": "2026-08-13-t27-bakeoff",
  "dataset": "linux",
  "n_graphs": 89,
  "n_pairs": 3916,
  "method": "BRANCH_FAST",
  "end": "lower",
  "options": "--threads 1",
  "deterministic": true,
  "cost_model": [1, 1, 0, 1, 1, 0],
  "gedlib_commit": "<git rev-parse HEAD of the graphkit-learn checkout>",
  "code_commit": "<git rev-parse HEAD of IsalGraph>",
  "host": "<platform.node()>",
  "wall_seconds": 12.3,
  "created_utc": "2026-08-13T00:00:00+00:00"
}
```

Index files set `method`, `end`, `options`, `deterministic` to `null`.

## 5. `$OUT/data/timing/{ds}__{METHOD}.json` — **Track A**, serial pass only

```json
{"dataset": "...", "method": "...", "options": "...", "n_pairs_timed": 2000, "seed": 42,
 "n_bar": 8.71, "us_per_pair_mean": 41.2, "us_per_pair_median": 38.9, "us_per_pair_p95": 77.0,
 "clock": "time.process_time", "parallel": false}
```

Plus `$OUT/data/timing/probe_n30__{METHOD}.json`, same schema, `"source": "iam_gxl:GREC+Protein",
"n_range": [25, 35]`. **Timings come from a single-process pass and never from the parallel pass.**

## 6. `$OUT/data/determinism/{ds}__{METHOD}.json` — **Track A**, design §3.11

```json
{"dataset": "...", "method": "...", "n_pairs": 5000, "seed": 42, "repetitions": 5,
 "defaults":  {"options": "", "frac_varying": 0.31, "max_spread": 4.0},
 "pinned":    {"options": "--threads 1 --randomness PSEUDO --seed 42 --initial-solutions 1",
               "frac_varying": 0.0, "max_spread": 0.0},
 "deterministic_under_pinned": true}
```

The exact option keys GEDLIB accepts per method are **discovered and reported**, not assumed. If a
method rejects an option, record the rejection rather than dropping the option silently.

## 7. `$OUT/data/analysis/…` — **Track B** writes, orchestrator reads

| File | Content |
|---|---|
| `metrics.json` | M1–M8 per cell. Every M1/M2/M3 figure appears twice: `all_certified` and `exact_gt_zero` |
| `validity.json` | M4: `{"cell": {"n_checked", "n_two_sided", "n_one_sided", "violations", "examples": [...]}}`. **`violations` must be 0 everywhere** |
| `bootstrap.json` | M5/M6 point estimates + 95 % percentile CI, graph-level, 2,000 replicates, seed 42; D7 paired differences |
| `significance.json` | Wilcoxon signed-rank + Holm within each end, rank-biserial effect sizes; Friedman + CD over 5 datasets |
| `selection.json` | Frozen §5 rule outcome, branch taken, margin; plus the §3.1 absolute-error and §3.2 corpus-collapsed companions and whether they agree |

## 8. Figures — **Track B**

`$OUT/figures/T27_lower_bound.pdf` + `.png`, `$OUT/figures/T27_upper_bound.pdf` + `.png`.
Two panels each: (a) mean relative error vs `max(n₁,n₂)`, ribbon = IQR, line per method, by dataset;
(b) forest plot of mean relative error with bootstrap CI, methods on y, grouped by dataset.

**Rendered through `isalgraph.viz`.** Palettes, IEEE sizes and `save_figure` come from
`isalgraph.viz.style`; a new view module `src/isalgraph/viz/bound_bakeoff_view.py` is the right
place for drawing code. **No bare `import matplotlib` in a benchmark script** — a test enforces
that third-party imports live inside function bodies in `isalgraph.viz`.

---

## 9. Ownership — disjoint, enforced

| Track | May create or edit | Everything else is read-only |
|---|---|---|
| **A** | `benchmarks/real_data/eval_setup/ged_bound_bakeoff.py`, `tests/unit/test_ged_bound_bakeoff.py` | ✔ |
| **B** | `benchmarks/real_data/eval_setup/ged_bakeoff_analysis.py`, `src/isalgraph/viz/bound_bakeoff_view.py`, `tests/unit/test_ged_bakeoff_analysis.py`, `tests/viz/test_bound_bakeoff_view.py` | ✔ |
| **C** | `.claude/notes/review/tasks/T-27-literature.md` | ✔ |

Plus each agent's own work log at `.claude/notes/2026-08-13-t27-bakeoff/<track>.md`.

**Nobody** edits: `.claude/notes/review/plan/**`, `tickets.md`, this file, `src/isalgraph/core/**`,
`benchmarks/real_data/eval_setup/ged_bounds.py`, or anything under `scratchpad/`.
**Nobody** runs the campaign, `ssh`, `rsync`, or `sbatch`. The orchestrator does all of that.
