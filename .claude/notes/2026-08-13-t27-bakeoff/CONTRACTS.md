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

`{CELL}` is the column name. **Amended 2026-08-13 mid-wave: 12 cells per dataset, 60 total** — not
the 8 first written here. `meta` carries `"cell"` (the column) beside `"method"` (the bare GEDLIB
name), so `IPFP_MS` and `IPFP_DET` both record `"method": "IPFP"`.

| End | Cells | Note |
|---|---|---|
| **Lower** (5) | `BRANCH`, `BRANCH_FAST`, `BRANCH_TIGHT`, `STAR`, `HED` | `HED` runs with `--edge-set-distances OPTIMAL`; its default is vacuous under D6 because edge substitution is free, and `hed.ipp` sets only a lower bound, so `UB = inf` is by design |
| **Upper** (7) | `BIPARTITE`, `IPFP_MS`, `REFINE_MS`, `BP_BEAM_MS`, `IPFP_DET`, `REFINE_DET`, `BP_BEAM_DET` | GEDLIB's `LSBasedMethod` defaults to one random start under `REAL` randomness, which is why IPFP returned 3.00 on P₄/C₄ |

**`_MS` is the arm that enters the §5 selection** — multi-start is the configuration the published
tightness claim was measured under and what a production matrix would use, so Holm within the upper
end stays C(4,2) = 6 over `BIPARTITE` + the three `_MS`. **`_DET` is a companion**: it quantifies how
much of IPFP's advantage is contingent on multi-start, and it is self-checking, since a monotone
local search started from BIPARTITE can never exceed BIPARTITE. Frozen before any tightness result
was visible.

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `value` | float64 | `(P,)` | **the reported bound.** LB: the single-orientation value. UB: `min(value_fwd, value_rev)` |
| `value_fwd` | float64 | `(P,)` | orientation `(i, j)` |
| `value_rev` | float64 | `(P,)` | orientation `(j, i)`. **UB cells only**; LB cells omit this key entirely |
| `meta` | `<U…` | `()` | JSON, §4 |

**Rules that are not negotiable.**

- **Assert `0 <= value < inf` on every read from GEDLIB, per pair, per orientation** — raises.

  > ### ⚠ AMENDED 2026-08-13, mid-wave — the original rule was FALSE and would have halted the run
  >
  > This section first said *"a value of exactly `0.0` where `exact > 0` means the wrong accessor
  > was called and must raise"*. **That is wrong.** A valid lower bound legitimately returns 0.0
  > whenever two non-isomorphic graphs share a degree sequence: under cost model D6 both node *and*
  > edge substitution are free, so a degree-preserving assignment costs nothing.
  >
  > Verified: C₆ versus two disjoint triangles has `networkx` exact GED **4.0**, and BRANCH,
  > BRANCH_FAST, BRANCH_TIGHT and STAR all return **0.00** — all valid. Measured at **1.0 %** of
  > certified LINUX pairs, and far higher on Letter, where n̄ = 4.7 makes degree collisions common.
  >
  > **The replacement, which catches the real failure without false positives:**
  > - **capability probe per cell**, before the pair loop, on a fixed synthetic pair with *differing*
  >   degree sequences — star K₁,₄ vs P₅, exact = 4.0 (BRANCH gives 2.00, HED-OPTIMAL 1.25). Require
  >   `0 < lb <= 4.0` for a lower-bound cell, `ub >= 4.0` for an upper-bound cell — raises;
  > - **all-zero guard**: if any `exact > 0` in the cell and the entire `value` vector is 0.0 — raises;
  > - **M4 per pair**, two-sided on certified and one-sided on censored, reported not clipped. A UB
  >   method read through `get_lower_bound()` returns 0 everywhere and is refuted by M4 on
  >   essentially every pair, which is the real backstop.
  >
  > Design §3.3 says "every valid lower bound returns 0 on an exact-GED-0 pair". That stays true;
  > **the converse it appears to imply is false.**
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
 "pinned":    {"options": "--threads 1 --randomness PSEUDO --initial-solutions 10",
               "frac_varying": 0.0, "max_spread": 0.0},
 "deterministic_under_pinned": true}
```

The exact option keys GEDLIB accepts per method are **discovered and reported**, not assumed. If a
method rejects an option, record the rejection rather than dropping the option silently.

> **Amended 2026-08-13**: the example above originally read `--randomness PSEUDO --seed 42`.
> **There is no `--seed` option** — GEDLIB raises `RuntimeError: Invalid option "seed"` rather than
> ignoring it. The pinned strings established empirically are
> `--threads 1 --randomness PSEUDO --initial-solutions 10` for the `_MS` arm and
> `--threads 1 --randomness PSEUDO --initialization-method BIPARTITE --initial-solutions 1` for
> `_DET`. Both measured fully deterministic — 0.0000 varying, 0.0 spread — against **91–94 %** of
> pairs varying at GEDLIB defaults, with spreads up to 10 edit operations.

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
