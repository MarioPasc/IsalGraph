# track-D-figures — the T-13 figure and table suite

Branch `worktree-agent-af3af62821a7b99f2`, base `b65ec21`.

---

## Mission

Build `benchmarks/real_data/eval_t13_figures/`, the figure and table suite for T-13, modelled on
`eval_t06_figures/` and consuming the frozen `t13.1` record. T-13 answers R3.7d by replacing the
manuscript's unqualified *"exponential worst case"* with a characterised one:

> The unpruned canonical search's cost is governed by the degree sequence; the pruned search's cost
> is governed by the automorphism group.

The evidence is a set of ladders holding `n`, `m` and the whole degree sequence fixed while `|Aut|`
falls. The figures exist to make that contrast legible at a glance.

---

## What I built

### `benchmarks/eval_t13_figures` → `real_data/eval_t13_figures`

The symlink every `python -m benchmarks.<name>.<module>` invocation goes through, matching the
twelve that already exist.

### `__init__.py`

Package docstring, module inventory, the four CLI entry points. `__all__` only; no re-exports, which
is `eval_t06_figures/__init__.py`'s shape.

### `design.py` — the registry, with T-06's two silent-failure modes closed

- `Representation` frozen dataclass: `key, short, long, tex, colour, marker, search_class, is_ours,
  is_focus`, with `linestyle` / `linewidth` / `zorder` derived.
- Thirteen entries. **Every colour is transcribed from `eval_t06_figures.design`** so a backend keeps
  one colour across the whole paper.
- `SearchClass.SEARCH_BASED` / `SEARCH_FREE`, drawn as a dash pattern because the search-free arms
  are the null of the cost law (design note §2.3) and that split has to survive a greyscale print.
- **`_check_registry()` runs at import.** It compares the styled keys against
  `measure.REPRESENTATIONS` in *both* directions and checks the search-class assignment against
  `measure.SEARCH_BASED` / `measure.SEARCH_FREE`. An arm added to the campaign and not styled here is
  an `ImportError`, not a missing legend entry.
- **`present()` raises** `UnknownRepresentationError` on an unregistered key. A deliberate omission
  must be named in `omit=`, which puts it at the call site where a reviewer can see it.
- `absent()` reports the other half: a styled arm with no data. Figures log it.
- `tex_name()` raises rather than echoing the key back (T-06 echoes, and a raw key in a table reads
  as a typo).
- Lifecycle: `style() / text_width() / column_width() / save() / finish_axes() / panel_letter() /
  note_box() / censor_arrow() / shared_legend()`. Every matplotlib import is inside a function body.

### `data.py` — loading, ladder addressing, censoring-aware statistics

- `Records`, `Ladder`, `LadderGraph`, `TimeSummary`, `SignTest`, `PowerLawFit`, `GraphResolution`,
  `CounterRecords` — all frozen and slotted.
- `load(paths)` globs, validates every row through `schema.validate_mapping` (propagated, never
  caught), keeps the headers, and **raises `MixedBuildError`** unless every shard declares one
  `build_info.build_hash`.
- `load_counters(paths)` validates the `t13c.1` field set against
  `dataclasses.fields(OperationCounts)` rather than a transcription, and rejects `parity_ok is not
  True` by default.
- `ladders(records, arm=...)` groups on `(family, n, base)` exactly as `families.ladder_span` does,
  orders graphs by `(rung, replicate)` where the rung comes from `params`, and raises
  `LadderIntegrityError` if `n`, `m` or `max_degree` moves inside a ladder.
- `resolutions(records)` collapses the nine repeated symmetry fields to one record per graph and
  raises if two rows of one graph disagree.
- Censoring: `is_completed / is_censored / is_observation / completion_rate /
  completions_only_median_seconds / km_median_seconds / summarise_times`.
- Statistics, plain Python: `spearman` (tie-corrected via mid-ranks), `sign_test` (exact two-sided
  binomial), `fit_power_law_completions_only` (OLS on logs + percentile bootstrap).

### `fig_cost_law.py` — the main-text figure

One panel per ladder, `x = log10_aut`, `y = seconds` on a log axis, thirteen series styled from the
registry, non-focus arms muted but still drawn and labelled. Each panel annotates `n`, `m`, `Δ`, the
fixed degree sequence, the `|Aut|` span, and the **measured** exhaustive/pruned variation ratios.
`caption()` quotes only numbers from the returned summary.

### `fig_resolution.py`

Panel (a): `n_wl_classes` and `n_triplet_classes` against `n_orbits` with the `y = x` ceiling drawn
and labelled. Panel (b): ECDF of the deficit below the ceiling. `check_ceiling()` runs **before
anything is drawn** and raises `CeilingViolationError` naming every offending field.

### `fig_operations.py`

Four panels — pair trials, pointer steps, neighbour checks, backtrack nodes — against `n`, with the
§2.1 bound overlaid. Raises on an unstyled encoder.

### `tables.py`

`tab_t13_ladder_spearman.tex` (per-ladder ρ over completions, with completion rate beside it, and a
sign-test footer across ladders), `tab_t13_scaling_exponent.tex` (α with a 2000-resample percentile
bootstrap, seed 13), `tab_t13_completion.tex` (completed / censored / unsupported / error, the
censoring mechanism named, KM median and completions-only median side by side). LaTeX is a
`list[str]` of raw strings joined at the end, which is `eval_t06_figures/tables.py`'s shape.
`_tex_escape` handles the underscores in `symmetry_ladder` / `complete_bipartite`.

### `tests/` — 5 files, 75 tests, and the fixture

`tests/__init__.py` builds the fixture and materialises it to
`tests/records_constructed_0of1.jsonl` (144 lines: 1 header + 143 rows) and
`tests/counters_fixture.jsonl` (16 rows). Every row is passed through `schema.validate_mapping` at
construction, so a broken fixture is a fixture bug and not a mysterious test failure.

---

## Acceptance criteria

| # | Criterion | Command | Actual output | Verdict |
|---|---|---|---|---|
| 1 | Every module imports with matplotlib uninstalled | `pytest .../tests/test_import_without_matplotlib.py` | 8 passed — a meta-path blocker for `matplotlib, networkx, igraph, numpy, scipy, pandas` plus a per-module AST check of module-scope imports | ✅ |
| 2 | `design.py` raises on an unknown key | `test_present_raises_on_an_unknown_key` | passed; `UnknownRepresentationError: unregistered representation(s) ['not_a_backend']` | ✅ |
| 3 | `data.py` raises on a mixed-`build_hash` load | `test_load_rejects_a_mixed_build` | passed; `MixedBuildError: the shards declare 2 different engine builds and cannot be pooled` | ✅ |
| 4 | Rungs order by the `params` index, not `log10_aut` | `test_rungs_order_by_the_params_index_and_not_by_log10_aut` | passed; the hypercube ladder's `log10_aut` is `[2.9823, 0.9031, 1.2041, 0.3010]` in rung order `(0,1,2,4)` — neither ascending nor descending, so an ordering by the measurement is visibly different | ✅ |
| 5 | No summary pools censored with completed rows | `test_the_naive_pool_and_the_censoring_aware_path_disagree` | passed; on `min_dfs` / spider ladder the naive pooled median is **below** the completions-only median (the 4.1 ms cap-censored row is the smallest number in the series), completion rate 0.75 | ✅ |
| 6 | `fig_resolution` raises above the ceiling | `test_resolution_raises_above_the_invariance_ceiling` | passed; `CeilingViolationError`, and the `.pdf` is **not** written | ✅ |
| 7 | All three figures generate `.pdf` + `.png` from a fixture | `test_*_writes_pdf_and_png` + the three CLI tests | passed; fixture at `benchmarks/real_data/eval_t13_figures/tests/records_constructed_0of1.jsonl` and `counters_fixture.jsonl` — 3 ladders, 7 censored rows, 3 unsupported rows, 1 error row, 1 non-monotone ladder | ✅ |
| 8 | `pytest .../tests/ -q` passes | `$PY -m pytest benchmarks/real_data/eval_t13_figures/tests/ -q` | `75 passed in 6.87s` | ✅ |
| 9 | `ruff` and `mypy` clean | `$PY -m ruff check ...` / `$PY -m mypy --explicit-package-bases ...` | `All checks passed!` / `Success: no issues found in 13 source files` | ✅ |

`src/isalgraph/` is untouched: `git status --porcelain -- src/` is empty and `git diff --stat HEAD --
src/` is empty.

The repository suite was **not** re-run. `pyproject.toml` sets `testpaths = ["tests"]`, so work under
`benchmarks/` cannot move the 2,618 reference figure, and this track registers no backend and edits
no `src/` file — the two things `.claude/CLAUDE.md` names as able to move it.

---

## Design decisions

### How censoring is rendered, and why

`status = "censored"` means *the completion time is greater than this*. Four decisions follow.

1. **Open marker, no fill, no connecting line, plus an upward arrow.** No fill because nothing
   completed; the arrow because the true value lies above and the reader must not read the plotted
   height as the measurement. `design.censored_kwargs` and `design.censor_arrow`; the arrow's length
   is a fixed fraction of the axes height in *display* space so it reads the same anywhere on a log
   axis, and it is stamped after the axis limits settle.
2. **Censored rows enter no median and no fit.** `_rung_medians` filters to completions, and
   `fit_power_law_completions_only` is named for its rule. The concrete failure this prevents:
   `min_dfs` at the most symmetric spider rung is `max_projections`-censored at **4.1 ms**. Pooled
   with its three completions it is the fastest observation in the series and drags the median down
   by more than an order of magnitude — the sentence *"min-DFS is fast"* when the data says *"min-DFS
   did not finish"*. `test_the_naive_pool_and_the_censoring_aware_path_disagree` pins it.
3. **Kaplan–Meier for anything that must estimate a central time.** `km_median_seconds` returns
   `(median, reached)`; when the survival curve never reaches 0.5 it returns `(None, False)` and the
   table prints *"not reached (> max observed)"* rather than substituting the completions-only
   median, which is exactly the bias the estimator exists to avoid. Reference: Kaplan and Meier,
   *JASA* 53(282):457–481, 1958.
4. **The mechanism is named, never pooled.** `schema` keeps `TIME_CENSORING_KINDS` and
   `CAP_CENSORING_KINDS` separable and so does the completion table: a wall-clock kill at 300 s and a
   projection cap that fires in 40 ms are different observations. `design.CENSORING_DISPLAY`.

`unsupported` and `error` observe no duration at all and are excluded from every time estimate, but
counted and printed.

### The registry raises instead of dropping

`.claude/CLAUDE.md` records that `eval_t06_figures.design.present()` drops unknown keys silently and
that this produced figures which regenerated successfully with an arm absent. Two changes: `present()`
raises, and `_check_registry()` runs at import so the *inverse* failure (campaign arm, no style)
cannot survive an import either. `absent()` covers the third case — style with no data — as a warning,
because a ladder on which one backend is `unsupported` throughout is a legitimate result that must
nonetheless be said out loud.

### Rung ordering is the design's, position is the measurement's

`ladders()` sorts by the `params` index. The figure then *places* each point at its own `log10_aut`
and joins per-rung medians in rung order. Those are not the same operation: ordering by the abscissa
would make the variable on the x axis decide the order of the correlation computed on it, which is
circular; placing a point at its own abscissa is just a scatter plot.

### `isalgraph.viz.style` directly, not `benchmarks.plotting_styles`

The brief's prohibition names `isalgraph.viz.style` first, and `plotting_styles` re-exports it, so
calling the source is one hop shorter and cannot drift. It also matters for criterion 9:
`benchmarks/plotting_styles.py` carries **15 pre-existing `mypy --strict` errors** (verified on the
untouched file — its last commit is `a368fdf`, well before this base), and routing this package
through it drags all 15 into `mypy --explicit-package-bases benchmarks/real_data/eval_t13_figures/`.
`test_geometry_comes_from_the_published_source_of_truth` asserts the geometry helpers really read
`viz.style`; the `plotting_styles` ↔ `viz.style` identity is not re-asserted here because the
repository already covers it and importing it would reintroduce those 15 errors.

### Legibility fixes made after looking at the rendered PNGs

Three defects only visible in the render, all fixed and all noted in code comments: the panel titles
overran a third-width panel and collided with the panel letter (`Ladder.title` is now short and
underscore-free, because matplotlib mathtext reads a bare `_` as a subscript); a five-line annotation
box hid the entire `complete_bipartite` series behind itself (now three lines, upper-left, with the
log y axis lifted by `NOTE_HEADROOM` to clear the censored row); and the first counter fixture drew
measured curves *above* their own bounds, which reads as a refuted derivation (counts are now
expressed as a fraction of the §2.1 bound, guarded by
`test_counter_fixture_stays_under_its_own_bounds`).

---

## Defects found in the brief

1. **The ownership list has no slot for the fixture data files that criterion 7 requires.** It permits
   `tests/__init__.py` and `tests/test_*.py`, but criterion 7 says *"build the fixture yourself,
   commit it under your `tests/`"*. I put the builder in `tests/__init__.py` (inside the list) and
   also committed the materialised `tests/records_constructed_0of1.jsonl` and
   `tests/counters_fixture.jsonl`, on the reading that criterion 7 is explicit permission and the
   glob simply did not anticipate data files. Flagging it rather than assuming.

2. **`Δ` is not in the `t13c.1` counter schema, so two of the four §2.1 bounds cannot be drawn as
   stated.** The counter record carries `n`, `m`, the counts and `parity_ok` — no maximum degree. I
   draw the `O(m Δ)` and `n·Δ^{n−1}` bounds at `Δ = n − 1`, which is `Δ`'s own worst case on a simple
   graph, making them looser than §2.1's bound and never tighter. It is labelled that way on the
   figure and in the caption. The alternative — pulling `max_degree` from the `t13.1` records and
   pairing it to a counter row — has no join key that survives a cohort row and I did not attempt it.

3. **§2.1 bounds search *leaves*; the brief names `backtrack_nodes`, which counts recursion frames.**
   Frames ≥ leaves, so attaching the leaf bound to the frame count would be a category error. Panel
   (d) draws `search_leaves` beside `backtrack_nodes` and attaches the bound to the series it
   actually bounds. Both are `0` for a greedy encode by construction
   (`instrumented.OperationCounts`), and a log axis cannot show zero, so the greedy series are absent
   from that panel rather than drawn at a fabricated floor.

4. **The `T ~ n^α` bounds are asymptotic, so a *measured* count exceeding one is not a defect.**
   Unlike the invariance ceiling — an exact inequality, which I do assert — the §2.1 bounds carry
   hidden constants. `fig_operations` therefore does **not** assert `measured ≤ bound`; only the
   fixture is held to it, and only so the supplementary figure does not look like a refutation.

5. **The single CLI form `--records <glob> --counters <glob> --out-dir <dir>` does not fit all four
   modules.** Each parser accepts both flags and requires only the one it reads, logging that it is
   ignoring the other, so the documented command line works verbatim on any module.

---

## What I did not do

- **No real data was plotted.** The T-13 campaign has not run; everything here is exercised against
  the synthetic fixture, and the fixture's numbers are the §6.3 pilot's on the spider ladder and
  plausible-but-invented elsewhere. Every invented value is confined to `tests/__init__.py` and its
  docstring says so. **No number in this package is quotable.**
- **The repository suite was not re-run** (see the acceptance table for why).
- **`--arm` beyond `default` is supported but untested against real ablation data.** `fig_cost_law
  --arm no_bnb` is the §6.3 consequence-2 check — the unpruned arm's flatness must survive
  `set_branch_and_bound(False)` — and the code path exists, but the fixture carries only the
  `default` arm, so it is exercised only structurally.
- **No cohort (`source = "cohort"`) handling in `ladders()`.** Cohort rows carry `family = None` and
  are skipped by design; the real-cohort external-validity arm (§2.5 / §6.1) has no figure here
  because the brief did not ask for one.
- **No `__main__.py`.** `eval_t06_figures` does not have one either, and its docstring's
  `python -m ...eval_t06_figures --report <dir>` is stale as a result. Each module is run
  individually, and this package's docstring says so rather than repeating the stale line.
- **Nothing under `eval_t13_complexity/`, `eval_t06_figures/`, `src/isalgraph/`, the plan files, the
  ticket board or `CONTRACTS.md` was touched.**
