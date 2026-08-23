# T-04a track A — the metric-feasibility grid

**Branch** `ticket/T-04a` · **base commit** `7e96f4a` · **workstation, `isalgraph-cpp`,
`isalgraph.engine() == "cpp"`** · no cluster, no SLURM.

---

## 1. What I changed

Three files, and only these three.

### `src/isalgraph/competitors/datasets.py`

| Added / removed | What |
|---|---|
| **+ `STRATA`** | `((2,5),(6,9),(10,12),(13,20),(21,40),(41,10**9))`, module-level and frozen. The deleted sampler kept them as a local named `bounds`, so nothing else could reach them. |
| **+ `SampleRecord`** | `frozen=True, slots=True`; `dataset`, `index`, `n_nodes`, `stratum`, `suite`. Carries the suite so the grid's F0 block can be split per suite without reloading a cohort. |
| **+ `stratum_of(n)`** | index into `STRATA`, first match wins, `None` below `n = 2`. |
| **+ `stratum_quotas(k, n_populated)`** | `divmod` with the remainder on the **last** (largest) strata. `(200,6) -> (33,33,33,33,34,34)`; `(50,6) -> (8,8,8,8,9,9)`. |
| **+ `stratified_subsample(records, k, *, seed, order=ALL_DATASETS)`** | the allocation itself, over an existing record list. `S50` is this call on `S200`, so the two draws share one implementation and cannot drift. |
| **+ `pooled_stratified_sample(names, k, *, seed)`** | CONTRACTS §1. Pools every graph of `names`, bins, allocates, draws with one `random.Random(seed)` consumed stratum by stratum ascending, each stratum ordered by `(names.index(dataset), index)`. Returns sorted by `(stratum, names.index(dataset), index)`. Shortfall is reported, never redistributed. |
| **− `stratified_sample`** | deleted, and removed from `__all__`. Nothing outside `grid.main` called it (checked by grep over `tests/`, `src/`, `experiments/`, `benchmarks/`). |

`__all__` now exports `STRATA`, `SampleRecord`, `pooled_stratified_sample`,
`stratified_subsample`, `stratum_of`, `stratum_quotas`. The import closure still reaches no GED
loader — `test_datasets_import_closure_reaches_no_ged_loader` is green.

### `src/isalgraph/competitors/grid.py`

Rewritten around a **cache per backend** rather than a measurement per cell.

- **F0/F1 split.** `encode_sample(backend, graphs, suites) -> EncodeCache` encodes every sample
  graph **once per backend** and is reused by all six metrics. It records, per backend, per
  `overall`/`suite1`/`suite2`: `attempted`, `encodable`, `frac`, and a `Counter` of
  `type(exc).__name__`. `except Exception`, not `except CompetitorError` — `min_dfs` raises a plain
  `ValueError` on a disconnected graph and a narrower catch would abort the run.
  F1 is now the fraction of pairs **among encodable graphs**, with `f1_n_pairs` as its denominator.
- **F3.** `encode_f3(backend, graphs, *, seed) -> F3Cache` draws 20 `fixtures.shuffled_copy`
  relabellings per graph and encodes **each copy once** (the ancestor encoded each twice, once for
  `is_defined` and once for `distance`, inside a loop that itself ran per cell). The copies are
  drawn *before* any encode, so the RNG stream is identical across backends whatever any of them
  raises on. A graph the backend raises on increments `skipped` and never `attempted`;
  `attempted + skipped == len(f3_graphs)` always. The F3 sample is
  `stratified_subsample(records, 50, seed)` — §3.2's `S50` — not `graphs[:50]`.
- **The candidate rule.** `_candidate_status(capabilities, metric)`, evaluated from the
  declarations alone **before anything is measured**, so eligibility cannot depend on an outcome.
  `metric.consumes not in {"symbols","frame","features"}` ⇒ `candidate = False`;
  `Capability.BASELINE` on the backend keeps its meaning and its existing message. Every cell is
  still measured and printed in full.
- **F6.** `f6_ms_per_pair` over all defined pairs, `f6_ms_per_pair_large` over pairs where **both**
  graphs have `n >= 21`, `f6_over_advisory_limit` a reported flag that `_apply_selection_rule` never
  reads. The ancestor's console line printed `1e3 * f6_ms_per_pair` under a header that said
  `us/pair`, i.e. ms rendered as µs; the header now says `ms/pair` and the number is ms.
- **F2 is now measured for the `VectorBackend` too.** The ancestor set `f2_violations = None` for
  `wl_subtree` and `_apply_selection_rule` reads `None` as "no violation", so an *unmeasured*
  criterion was passing selection by default. Measured on the real cohort: `wl_subtree × kernel`
  has zero violations, so the selection outcome is unchanged — but it is now measured rather than
  assumed. **This is beyond the four defects in the brief; flagging it rather than doing it
  silently.**
- **JSON** is CONTRACTS §2 exactly, and the `dryrun` path now emits the *same* block shape as
  `pooled`, so track C needs no special case.
- **Back-compat, deliberate.** `measure_cell` still accepts a plain graph sequence as its third
  argument and `_apply_selection_rule` still accepts a backend as its second, because two tests in
  the read-only `tests/unit/test_competitors_core.py` call them with the T-04 signatures. See §4.2.

### `tests/unit/test_competitors_grid.py` (new)

40 tests, each keyed in the module docstring to the defect it covers. Synthetic where the
assertion allows it; the four that need the real cohort carry `@pytest.mark.integration` and a
`skipif` on `datasets.available_datasets()`.

**Files touched, exhaustively:**

```
src/isalgraph/competitors/datasets.py
src/isalgraph/competitors/grid.py
tests/unit/test_competitors_grid.py
.claude/notes/2026-08-16-t04a-metric-feasibility/grid.md   (this file)
```

Nothing else. `git status` showed `.claude/notes/review/tasks/T-04a-design.md` modified by someone
else mid-session; I left it alone.

---

## 2. Which of my tests fail on the base commit `7e96f4a`

**All 40.** Verified by checking the two source files out at the base commit, running the file, and
restoring:

```bash
git checkout 7e96f4a -- src/isalgraph/competitors/grid.py src/isalgraph/competitors/datasets.py
$PY -m pytest tests/unit/test_competitors_grid.py -q     # 40 failed in 2.59s
git checkout HEAD   -- src/isalgraph/competitors/grid.py src/isalgraph/competitors/datasets.py
```

Named against the four defects, with the base-commit failure mode:

| Defect | Test | Fails on `7e96f4a` with |
|---|---|---|
| 1 — wrong sample | `test_stratified_sample_is_deleted` | `assert not hasattr(datasets, "stratified_sample")` — the per-dataset sampler is still importable |
| 1 | `test_stratum_quotas_send_the_remainder_to_the_largest_strata` | `AttributeError: module 'isalgraph.competitors.datasets' has no attribute 'stratum_quotas'` |
| 1 | `test_stratum_of_bins_on_the_first_match` (13 params) | `AttributeError: ... has no attribute 'stratum_of'` |
| 1 | `test_stratified_subsample_*` (3) | `AttributeError: ... has no attribute 'stratified_subsample'` |
| 1 | `test_pooled_stratified_sample_draws_exactly_k_balanced_over_strata` | `AttributeError: ... has no attribute 'pooled_stratified_sample'` |
| 2 — `size_null` wins | `test_size_null_metric_would_win_on_f6_and_is_refused_anyway` | `AttributeError: module 'isalgraph.competitors.grid' has no attribute 'encode_sample'` |
| 2 | `test_no_non_candidate_metric_is_ever_selected_anywhere` | `AttributeError: ... has no attribute 'run_grid'` |
| 2 | `test_baseline_backend_keeps_its_existing_exclusion` | `AttributeError: ... has no attribute 'BASELINE_EXCLUSION'` |
| 2 | `test_selection_reason_names_the_failing_criterion_for_every_candidate` | `AttributeError: ... has no attribute 'run_grid'` |
| 3 — F0 discarded | `test_encode_failures_are_counted_per_suite_with_their_type` | `AttributeError: ... has no attribute 'encode_sample'` |
| 3 | `test_a_non_competitor_exception_is_still_counted_not_raised` | same |
| 3 | `test_all_suite_keys_are_present_even_when_empty` | same |
| 3 | `test_f1_denominator_is_pairs_among_encodable_graphs` | same |
| 3 | `test_encoding_is_shared_across_the_metrics_of_a_row` | same |
| 4 — F3 on `graphs[:50]` | `test_f3_subsample_is_stratum_balanced` | `AttributeError: ... 'stratified_subsample'` |
| 4 | `test_f3_subsample_of_the_real_s200_is_stratum_balanced` | `AttributeError: ... 'pooled_stratified_sample'` |
| 4 | `test_f3_attempted_plus_skipped_is_the_sample_size` | `AttributeError: ... 'encode_f3'` |
| 4 | `test_f3_is_evaluated_on_one_encoding_per_copy_shared_by_every_metric` | same |
| 4 | `test_f3_relabelling_can_actually_fail_for_an_order_dependent_format` | `AttributeError: ... 'encode_sample'` |
| grid as a whole | `test_every_registered_cell_is_measured_and_printed`, `test_f6_*` (2), `test_grid_run_is_reproducible_*`, `test_an_unavailable_backend_*`, `test_sample_block_*`, `test_dryrun_cli_writes_the_frozen_payload` | missing `run_grid` / `Cell.f6_ms_per_pair_large` / `sample_block` / the CONTRACTS §2 keys |

**Honest caveat on this evidence.** Most of these fail on the base commit with `AttributeError`
rather than with a *wrong number*, because the repair required new entry points. Two are stronger
and worth reading as the real proof:

- `test_stratified_sample_is_deleted` fails on a live assertion, not a missing symbol.
- `test_size_null_metric_would_win_on_f6_and_is_refused_anyway` asserts the **premise** separately
  from the **rule**: on 11 path graphs `P2…P12` under `isalgraph_canonical`, `size_null` satisfies
  every merit criterion §3.4 names (F1 = 1.0, zero F2 violations, F3 = 11/11, zero-mass 0.0,
  CV 0.61) and costs `9.02e-5 ms/pair` against `levenshtein`'s `9.82e-4` — **10.9× cheaper**, so the
  ancestor's `min` on `(F6, name)` names it. The repaired rule refuses it on `consumes` with
  CONTRACTS §4's exact string. That is defect 2 measured on both sides.

---

## 3. Measurements

### 3.1 `--sample pooled-30`, wall time **74.9 s**

```
$ time $PY -m isalgraph.competitors.grid --sample pooled-30 --seed 42 --out <scratch>/t04a_pooled30.json
real 1m14.879s   user 1m14.493s   sys 0m2.161s   exit 0
```

Single-threaded (`user ≈ real`), 66 cells. Extrapolating the binding cost — F3 is
`k × 20` encodes per backend and the sample sweep is `k` encodes per backend, both linear in `k`,
while the pair sweeps go as `k²` but are ≤ 1.2 s at `k = 200` — the 200-graph run should land in the
**8–12 min** band, comfortably inside §1.5's "well under an hour".

### 3.2 F0 on `pooled-30`

| backend | overall | suite1 | suite2 | errors |
|---|---|---|---|---|
| adjacency | 30/30 | 10/10 | 20/20 | — |
| **agm_cam** | **14/30** | 10/10 | **4/20** | `SuiteScopeError` 15, `AGMBudgetExceeded` 1 |
| graph6 | 30/30 | 10/10 | 20/20 | — |
| **isalgraph_canonical** | **15/30** | 10/10 | **5/20** | `SuiteScopeError` 15 |
| **isalgraph_pruned** | **29/30** | 10/10 | 19/20 | `CanonicalizationTimeoutError` 1 |
| min_dfs | 30/30 | 10/10 | 20/20 | — |
| nauty_graph6 | 30/30 | 10/10 | 20/20 | — |
| size_null | 30/30 | 10/10 | 20/20 | — |
| sparse6 | 30/30 | 10/10 | 20/20 | — |
| sparse6_nauty | 30/30 | 10/10 | 20/20 | — |
| wl_subtree | 30/30 | 10/10 | 20/20 | — |

The shape matches §1.3.3's 200-graph measurement (`agm_cam` ~51 %, `isalgraph_canonical` ~50 %,
`isalgraph_pruned` ~96 %, `min_dfs` ~96 %) except that `min_dfs` loses nothing at `k = 30` — its 8
`MinDfsBudgetExceeded` failures are in the tail this smaller draw did not reach.

**A4 is discharged**: `agm_cam` reports `SuiteScopeError` on the Suite-2 graphs, by name and by
count.

### 3.3 Selection on `pooled-30`

| representation | primary | why |
|---|---|---|
| adjacency | **none** | F3 = 0/30 for every candidate |
| graph6 | **none** | F3 = 0/30 |
| sparse6 | **none** | F3 = 0/30 |
| agm_cam | `levenshtein` | passes with `padded_hamming`; F6 0.0007 vs 0.0061 ms/pair |
| nauty_graph6 | `levenshtein` | passes with `padded_hamming`; F6 **0.0013 vs 0.0848** ms/pair |
| sparse6_nauty | `levenshtein` | only candidate passing |
| min_dfs | `levenshtein` | only candidate passing |
| isalgraph_canonical | `levenshtein` | only candidate passing |
| isalgraph_pruned | `levenshtein` | only candidate passing |
| wl_subtree | `kernel` | only candidate |
| size_null | **none** | `Capability.BASELINE` |

This is a 30-graph rehearsal, not the ticket's result, but it already reproduces two of §1.6's
pre-committed predictions: the **non-canonical family has no admissible distance**, which puts
`k ≈ 3` over the Claim-B comparator set; and **`levenshtein` beats `padded_hamming` on F6 by 65×
for `nauty_graph6`**, which means `competitors/README` §3's provisional "padded Hamming" column for
`nauty→graph6` and `AGM CAM` is wrong and needs correcting. **I have not edited that plan file** —
reporting it for you to make the edit.

### 3.4 What surprised me

1. **`levenshtein_char` is cheaper than `levenshtein`** — 0.000290 vs 0.000982 ms/pair on the
   `P2…P12` fixture, 3.4×, and 0.0056 vs 0.0080 on `adjacency` in the `pooled-30` run. `rapidfuzz`
   on a `str` is bit-parallel per character; on a `tuple[str, ...]` it is not. So if
   `consumes == "text"` were a candidate, **`levenshtein_char` would win the F6 tie-break for
   essentially every serialisation** and the paper would report the character-level unit — the
   exact "wrong unit" trap `base.py`'s module docstring is written against. See §4.1: this is why I
   implemented the candidate rule as set membership rather than as an `order`-only check.
2. **`padded_hamming` scales badly and it is now visible.** `nauty_graph6 × padded_hamming` costs
   0.0848 ms/pair overall but **0.2032 ms/pair on the `n >= 21` pairs** — 2.4× worse where the
   §3.3 threshold is actually about. `levenshtein` on the same row goes 0.0013 → 0.0032, the same
   2.4×. `f6_ms_per_pair_large` earns its place.
3. **F3 = 0/30 for `adjacency`, `graph6` and `sparse6` at `k = 30`, and 2/20 on the dryrun.** The
   dryrun's non-zero count is graphs with `n = 2`, where every relabelling is an automorphism. On
   the stratified draw those are a sixth of the sample rather than the whole of it, so the count
   collapses. That is a *reason to prefer the stratified sample*, not an anomaly.
4. **`min_dfs` cost nothing at `k = 30`.** §1.5 measured 0.53 s/graph on the 200-graph sample;
   the 30-graph draw reaches `n = 83` too, yet the whole run is 75 s. The 0.53 s figure is
   dominated by a handful of large graphs, so it does not extrapolate linearly and the 200-graph
   estimate in §3.1 should be read as a band, not a point.

---

## 4. Where the brief turned out to be wrong

Two things, both real, neither worked around silently.

### 4.1 The candidate rule cannot be `consumes == "order"` alone — `"text"` must go too

**The brief says**: "`metric.consumes == "order"` ⇒ `candidate = False` with the exact
`excluded_because` string given there."

**But CONTRACTS §4's own first sentence says**: "A metric is a **candidate** for a representation
iff `metric.consumes in {"symbols", "frame", "features"}`." Design note §3.4 states it identically,
and `levenshtein_char`'s own docstring says "It is never a primary distance." Three sources give
the rule as set membership; only the code snippet narrows it to `order`.

**The evidence that this matters** is §3.4.1 above: `levenshtein_char` is measurably *cheaper* than
`levenshtein` on every serialisation, so under an `order`-only rule it wins the F6 tie-break
essentially everywhere and the manuscript reports the character-level edit unit — which charges
four edits for one deleted min-DFS tuple and is a twofold error `competitors/README` §6 lists as a
documented trap.

**What I implemented**: `CANDIDATE_CONSUMES = frozenset({"symbols", "frame", "features"})`, the
membership rule. `order` gets CONTRACTS §4's string **verbatim**:

```
baseline: consumes 'order'; not a candidate distance (competitors.md 3.2)
```

`text` gets a parallel but distinct string, so track C can tell the two apart:

```
supplementary: consumes 'text'; not a candidate distance (competitors.md 3.2)
```

**No schema change** — `candidate` and `excluded_because` are the CONTRACTS §3 fields, unchanged in
name and type. But track C should know that `excluded_because` now carries a second
`not a candidate distance` variant. **Please relay that to track C.**

### 4.2 "Everything else in the repository is read-only" collides with two existing tests

`tests/unit/test_competitors_core.py` calls two `grid.py` functions with their T-04 signatures:

- line 394: `measure_cell("agm_cam", "levenshtein", graphs, seed=42, suite="suite2")` — third
  argument is a **graph list**, and there is no `f3cache`.
- line 461: `_apply_selection_rule(cell, get_repr_backend("size_null"))` — second argument is a
  **backend**, and the function is expected to apply the `BASELINE` check itself.

The F0/F1 split needs `measure_cell` to take a cache, and the candidate rule needs the *metric*,
which `_apply_selection_rule` does not have. Acceptance criterion 2 says that file must still pass
and I may not edit it, so I kept both call shapes working:

- `measure_cell(backend, metric, cache_or_graphs, f3cache=None, *, seed, suite=None, suites=None)`
  — a graph sequence builds the caches on the spot. Documented as the ad-hoc path; `run_grid` never
  uses it.
- `_apply_selection_rule(cell, backend=None)` — when a backend is given, the `BASELINE` half of the
  candidate rule is applied there. `measure_cell` calls it with no backend, having already set
  `cell.candidate` and `cell.excluded_because` from `_candidate_status`.

Both are compatibility surface, not design. **If you would rather those two tests were updated to
the new signatures, say so and I will send you the two-line diff** — I did not edit that file.

### 4.3 One field the contract does not pin down

CONTRACTS §2 says "A suite key is present even when its count is 0" but its example never shows
`attempted == 0`. I emit `frac = 0.0` in that case, which reads as "0 % encodable" rather than "not
measured". `attempted` disambiguates it and the docstring says so. If track C would rather have
`null`, that is a one-line change — tell me before the 200-graph run, since the JSON is the artifact.

---

## 5. Acceptance criteria

`PY=~/.conda/envs/isalgraph-cpp/bin/python`

**1. `$PY -m pytest tests/unit/test_competitors_grid.py -q`**

```
40 passed in 3.93s
```

**2. `$PY -m pytest tests/unit/test_competitors_core.py -q`**

```
39 passed in 0.29s
```

including `test_grid_import_closure_reaches_no_ged_loader` and
`test_datasets_import_closure_reaches_no_ged_loader`. Decision 24's structural F5-blindness is
intact: neither file imports `ged_reference`, `scipy` or anything reaching a GED value.

**3. `$PY -m ruff check src/isalgraph/competitors/ tests/unit/test_competitors_grid.py`**

```
All checks passed!
```

**4. `$PY -m mypy src/isalgraph/`** (repo strict settings, whole package, not only my two files)

```
Success: no issues found in 71 source files
```

**5.**

```
$ $PY -c "from isalgraph.competitors import datasets; s=datasets.pooled_stratified_sample(datasets.ALL_DATASETS,200,seed=42); import collections; print(len(s), collections.Counter(r.stratum for r in s))"
200 Counter({4: 34, 5: 34, 0: 33, 1: 33, 2: 33, 3: 33})
```

The draw also reproduces design note §1.4 exactly: `n = 2 … 83`, mean `20.92`, and
`mutagenicity 50, coil_del 46, aids_iam 35, iam_letter_high 24, iam_letter_low 16, grec 10,
protein 8, iam_letter_med 6, aids 5, linux 0`.

**6.**

```
$ $PY -m isalgraph.competitors.grid --sample dryrun-20 --dataset iam_letter_low --seed 42 --out <scratch>/t04a_dryrun.json
exit 0
$ python -c "import json; d=json.load(open(...)); print(len(d['cells']), len(d['backends']))"
66 11
```

All CONTRACTS §2 top-level keys present; `f0` carries `overall`/`suite1`/`suite2` for all eleven
backends; `size_null` appears in no `primary_distance` value.

**7. Full 200-graph grid NOT run** — that is the orchestrator's. `--sample pooled-30` proves the
path: **74.9 s, exit 0, 66 cells**, output at
`/tmp/claude-1000/-home-mpascual-research-code-IsalGraph/06276acb-2ee7-4615-b7c9-201b86d8c2ee/scratchpad/t04a_pooled30.json`
with the console log beside it as `pooled30.log`.

---

## 6. For the orchestrator

Three things needing a decision or an edit I am not allowed to make:

1. **Relay §4.1 to track C**: `excluded_because` now carries two `not a candidate distance`
   variants, `order` and `text`. The `order` string is CONTRACTS §4 verbatim.
2. **`competitors/README` §3's provisional primary-distance column is wrong** for `nauty→graph6`
   and `AGM CAM` — measured, `levenshtein` beats `padded_hamming` on F6 by 65× and 9× respectively.
   §3.7 says this run wins and §3 is corrected in place. Your edit.
3. **§4.2**: two tests in the read-only `test_competitors_core.py` pin the T-04 signatures. I kept
   compatibility shims. Say the word and I will hand you the diff that removes them.
