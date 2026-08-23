# T-04a — metric feasibility: design note

**Ticket**: T-04a, *Metric feasibility — every (representation × distance) cell on a fixed 200-graph
sample; select each primary distance by the pre-declared rule.*
**Depends**: T-04 (closed 2026-08-15). **Gates**: T-06's distance matrices, T-17's AE.3 table.
**Serves**: AE.4a (requirement modal), AE.3, R1.1, R1.2a/b, R3.6a.
**Owns the pre-registration parameter `k`** ([preregistration](../plan/preregistration.md) §7).

Plan sources: [competitors](../plan/competitors.md) §3, [competitors/README](../plan/competitors/README.md),
[preregistration](../plan/preregistration.md) §5 and §7, [decisions](../plan/decisions.md) 18 / 23 / 24.

**Written and committed 2026-08-16, before any agent started.** Everything in §3 is frozen by that
commit; §4 is what closing the ticket requires.

---

## 1. State, measured now

Measured on the workstation, 2026-08-16, `isalgraph.engine() == "cpp"`, conda env `isalgraph-cpp`.
Base commit `f7ad283`. Working tree clean.

### 1.1 What is installed and available

| Item | Measured |
|---|---|
| Backends registered | **11** — `adjacency`, `agm_cam`, `graph6`, `isalgraph_canonical`, `isalgraph_pruned`, `min_dfs`, `nauty_graph6`, `size_null`, `sparse6`, `sparse6_nauty`, `wl_subtree` |
| Metrics registered | **6** — `hamming`, `kernel`, `levenshtein`, `levenshtein_char`, `padded_hamming`, `size_null` |
| Grid cells | **66** |
| Cohorts on disk | **10/10** under `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data` |
| Certified exact GED | **5/5** Suite-1 datasets, `source/GED_PRECOMPUTED/extended_merged_exact_ged/computed/` |
| Suite-2 bounds (T-05) | **10/10** `LB/` and `UB/` under `source/APPROX_GED/`; `graph_ids` verified to align element-wise with `datasets.load(<ds>)` (checked on `mutagenicity`, 4,040 ids, exact match) |

### 1.2 The cohort `datasets.ALL_DATASETS` actually pools

| dataset | N | n min / mean / max | | dataset | N | n min / mean / max |
|---|---:|---|---|---|---:|---|
| iam_letter_low | 1,180 | 2 / 4.07 / 7 | | grec | 650 | 4 / 11.45 / 24 |
| iam_letter_med | 1,253 | 2 / 4.11 / 8 | | aids_iam | 1,811 | 2 / 14.02 / 85 |
| iam_letter_high | 2,059 | 2 / 4.58 / 9 | | coil_del | 3,900 | 3 / 21.54 / 77 |
| linux | 89 | 4 / 8.71 / 10 | | mutagenicity | 4,040 | 4 / 28.53 / 98 |
| aids | 769 | 2 / 10.56 / 12 | | protein | 569 | 2 / 31.68 / 96 |

**Pooled: 16,320 graphs.** This is *not* T-01's Suite-2 count of 16,370 and is not meant to be:
`ALL_DATASETS` takes Letter / LINUX / AIDS from the **Suite-1** export (`n ≤ 12` filter) and the five
new datasets from the Suite-2 export, and excludes `aids_graphedx`. "The ten locked datasets" of
§3.1 is this union, and it is the set the shipped code defines.

### 1.3 Four defects in the shipped harness, all measured

Every one of these would have produced a plausible number and no error.

1. **`--sample stratified-200` does not implement §3.1.** `datasets.stratified_sample(ALL, 200,
   seed=42)` returns a `k`-graph draw **per dataset** — measured **1,889 graphs**, and
   `C(1889,2) = 1,783,516` pairs against §3.1's frozen `C(200,2) = 19,900`. Its allocation is also
   *proportional within a dataset*, which reproduces each dataset's own size distribution instead of
   spreading the draw over the six node-count strata §3.1 names.

2. **The `size_null` *metric* wins the selection for every representation.** It declares
   `consumes = "order"`, reads `Encoding.n_nodes` and never touches `symbols`. Measured: defined on
   **100 %** of pairs for all eleven backends, a true metric on ℤ (so zero F2 violations),
   relabelling-invariant (so 50/50 on F3), non-degenerate on a size-spread sample, and the cheapest
   cell in the grid. §3.4 selects *the cheapest that passes F1 at 100 %, F2, F3 and F4* and
   `_apply_selection_rule` guards only `Capability.BASELINE` **on the backend**. So the rule as
   implemented would name *count the nodes and subtract* the primary distance of all eleven
   representations, and every Claim-B row would then measure the size null and nothing else. This is
   finding 1 of [competitors/README](../plan/competitors/README.md) §5 reappearing inside the tool
   built to prevent it.

3. **`measure_cell` drops encode failures.** `encoded, _ = _encode_all(backend, graphs)` discards the
   failure count, and F1 is then computed over the pairs that *did* encode. `grid.py:313` claims the
   opposite in a comment ("a `SUITE1_ONLY` backend shows up as F1 < 1.0 instead"). Measured encode
   failures on the frozen sample: `agm_cam` **102/200** (101 `SuiteScopeError`, 1
   `AGMBudgetExceeded`), `isalgraph_canonical` **101/200** (`SuiteScopeError`), `isalgraph_pruned`
   **9/200** (`CanonicalizationTimeoutError`), `min_dfs` **8/200** (`MinDfsBudgetExceeded`); zero for
   the other seven.

4. **F3 is measured on `graphs[:50]`.** On a pooled sample ordered by stratum that is the whole of
   stratum 1 plus part of stratum 2 — F3 would be measured on the smallest graphs only. `_f3_cell`
   also encodes each relabelled copy **twice** (once for `is_defined`, once for `distance`), and
   returns no skip count, so a backend that raises on all 50 graphs reports `0/50` — indistinguishable
   from a genuine invariance failure. `src/isalgraph/competitors/README.md:139` states that `_f3`
   reports a skip count; it does not.

### 1.4 The frozen sample, measured

Pooled, stratum-balanced, 200 graphs, seed 42 (§3.1 as written):

```
n ∈ [2,5]     pool 4,016    n ∈ [13,20]   pool 2,106
n ∈ [6,9]     pool 2,249    n ∈ [21,40]   pool 4,292
n ∈ [10,12]   pool 2,557    n ∈ [41,∞)    pool 1,100
```

Draw: **200 graphs, n = 2 … 83, mean n 20.92**, from nine of the ten datasets
(`mutagenicity` 50, `coil_del` 46, `aids_iam` 35, `iam_letter_high` 24, `iam_letter_low` 16,
`grec` 10, `protein` 8, `iam_letter_med` 6, `aids` 5, `linux` 0). LINUX draws zero because its
`n ∈ [4,10]` sits inside strata where it is 89 graphs against thousands; that is the honest
consequence of stratifying on node count and is reported, not repaired.

### 1.5 Cost, measured — the grid is a workstation job, not a cluster job

Distance evaluation is free; **encoding is the whole cost**. Extrapolated to 19,900 pairs from a
60-graph timing run:

| | cost |
|---|---|
| `min_dfs` encode | **0.53 s/graph** on the frozen sample (106 s / 200) |
| `isalgraph_pruned` encode | **0.108 s/graph** (21.6 s / 200) |
| every other backend encode | ≤ 0.18 s **total** for 200 graphs |
| every (backend × metric) distance sweep over 19,900 pairs | **≤ 1.2 s**, and ≤ 0.3 s for all but `padded_hamming` |

So the binding cost is F3: 50 graphs × 20 relabellings = 1,000 encodes per backend, which is ~530 s
for `min_dfs` and ~110 s for `isalgraph_pruned` **if each copy is encoded once**. Encoding once per
backend and reusing across the six metrics turns the shipped harness's ~6× redundancy into 1×.
**Whole grid: well under an hour. No Picasso, no SLURM, no queue.**

### 1.6 What the measurement predicts, recorded before the run

Written down now so that a matching outcome is a confirmation rather than a description:

- `hamming` is undefined on **89–97 %** of pairs for every representation → F1 fails everywhere.
  That is §3's founding argument, measured.
- `padded_hamming` is defined on 100 % of pairs for `adjacency`, `graph6`, `nauty_graph6`, `agm_cam`
  and on **0 %** for `sparse6`, `sparse6_nauty`, `min_dfs`, `isalgraph_*` (no positional frame).
- `levenshtein` is defined on 100 % of pairs for all ten serialisations.
- Therefore the non-canonical formats (`adjacency`, `graph6`, `sparse6`) should have **no admissible
  distance**: their surviving candidates fail F3. That is §4 outcome 1, generalised from graph6 to
  the whole non-canonical family, and it puts `k ≈ 3`.
- Where two candidates survive, the tie breaks on F6, and **`levenshtein` is measurably cheaper than
  `padded_hamming`** (0.0 s vs 1.2 s per 19,900 pairs on `nauty_graph6`). If that holds,
  [competitors/README](../plan/competitors/README.md) §3's provisional "padded Hamming" for
  `nauty→graph6` and `AGM CAM` is **wrong** and gets corrected.

---

## 2. Approach

Four pieces, in order. The first two are code; the third is the run; the fourth is the report.

1. **Repair the harness** (`grid.py`, `datasets.py`) against §1.3, and add the F0/F1 split §3.3
   below freezes. Tests for each defect, each of which must fail on the current code.
2. **Extend F5** (`f5.py`) to take the *selected* primary distance per representation from the grid's
   output, to emit graph-level bootstrap CIs, and to add the descriptive Suite-2 arm on T-05's
   bounds.
3. **Run** the grid on the frozen sample, then F5 under the selected distances.
4. **Report**: the full 66-cell supplementary table, the selection table, `k`, the partial-
   computability term, and the four pre-committed outcomes.

### 2.1 Rejected alternatives

| Rejected | Why |
|---|---|
| Run `--sample stratified-200` as shipped | It is 1,889 graphs and the wrong stratification (§1.3.1). §3.1's text is frozen and says 200 / 19,900. |
| Proportional allocation across the six strata | Reproduces the pooled cohort's size distribution, which is 25 % Letter at `n ≤ 5`. §3.1's stated purpose is that "the unequal-`n` case dominates exactly as it does in production", which needs the strata *spread*, not weighted. |
| Guard `size_null` by name | A name check breaks the moment a second order-only metric is registered. The guard is on `consumes`, which is the property that makes the metric ineligible. |
| Let the encode failure drive F1 to < 1.0 (the comment's intent) | It conflates *this representation cannot be built for this graph* with *this distance is undefined on this pair*. §3.4's criteria are properties of a **distance**; preregistration §5 already separates computability and charges it differently (10 rows, not 15). Chosen instead: §3.3's F0/F1 split. **User decision, 2026-08-16.** |
| Run the grid twice, once per suite | More information, but it abandons §3.1's single pooled sample and doubles a run whose value is that one draw fixes one answer. The F0-per-suite report carries the same content. |
| Defer Suite-2 ρ to T-06 entirely | Correct by the letter of §3.3, but T-05's matrices are on disk and aligned, and the arm costs hours. **User decision, 2026-08-16: add it, descriptive only.** |
| Any cluster submission | §1.5 — the whole grid is under an hour on the workstation. A SLURM job here would violate the 2-hour floor and buy nothing. |

---

## 3. Frozen before the run

Everything in this section is fixed by the commit that adds this file. A change after that point
needs a dated entry in §6.

### 3.1 The sample `S200`

- Pool every graph of `datasets.ALL_DATASETS` (16,320) and bin by node count into
  `[2,5] [6,9] [10,12] [13,20] [21,40] [41,∞)`.
- Quotas: `divmod(200, 6) = (33, 2)` → **`[33, 33, 33, 33, 34, 34]`**, the remainder to the *largest*
  strata, preserving `stratified_sample`'s existing stated rationale (keep the tail, where the
  `m`-scaling argument lives, from being rounded away).
- One `random.Random(42)`, consumed stratum by stratum in ascending order, each stratum's members
  ordered by `(dataset, index)` with `ALL_DATASETS`' own dataset order.
- **`S200` is a function of `(ALL_DATASETS, 200, 42)` alone** and of nothing that ran before it.

### 3.2 The F3 sub-sample `S50`, and the F3 protocol

- 50 graphs drawn **from `S200`**, `divmod(50, 6) = (8, 2)` → **`[8, 8, 8, 8, 9, 9]`** per stratum,
  remainder to the largest strata, one `random.Random(42)`.
- 20 relabellings per graph via `fixtures.shuffled_copy` — **never** `nx.relabel_nodes(copy=True)`,
  which preserves insertion order (finding 13).
- Each copy is encoded **once**.
- F3 reports `invariant / attempted` **and** a skip count. A backend that raises on a graph
  contributes to `skipped`, never to `attempted`, so `0/50` and "never ran" are distinguishable.

### 3.3 F0 and F1 — the two axes, split

| | Definition | Grain |
|---|---|---|
| **F0 — encodability** | fraction of `S200` graphs the *representation* encodes without raising, with the exception type counted | per **representation × suite** |
| **F1 — well-definedness** | fraction of pairs **among encodable graphs** on which the *distance* is computable | per cell |

A cell is eligible to be a primary distance only if **F1 = 1.0**; a representation may carry a
printed row **for a suite** only if its **F0 = 1.0 on that suite**. The two exclusions are counted
separately and charged separately (§3.6).

### 3.4 The candidate set — §3.2 of the plan, made executable

A metric is a **candidate** for a representation iff it reads the representation:
`consumes ∈ {"symbols", "frame", "features"}`. A metric with `consumes == "order"` reads no part of
the encoding and is **never eligible as a primary distance**, whatever it scores.

This is not a new rule. [competitors](../plan/competitors.md) §3.2's grid table enumerates the
candidate distances per representation and `size_null` appears in none of them; decision 23 already
holds `size_null` outside the confirmatory family. §1.3.2 is the measurement showing the
implementation did not encode either statement.

**Every cell is still attempted and printed**, including the ineligible ones — §3.2's "a cell that
fails is a result" is unchanged. Ineligibility is recorded as
`excluded_because = "baseline: consumes 'order'; not a candidate distance (competitors.md §3.2)"`.

### 3.5 The selection rule — unchanged, and F5-blind

§3.4 verbatim: *for each representation the primary distance is the cheapest that passes F1 at
100 %, F2, F3 and F4; ties are broken by F6, never by F5.* Implemented as: among candidate cells
(§3.4) with F1 = 1.0, zero observed F2 violations, F3 invariant on every attempted graph, F4
zero-mass ≤ 0.5 and coefficient of variation ≥ 1e-6 — take `min` on `(f6_ms_per_pair, metric_name)`.

F5-blindness stays structural: `grid.py`'s import closure must not reach a GED loader, and the
existing test asserting that must still pass. **F6 is the tie-break and never a gate** — §3.3's
"> 1 ms/pair" line is reported as an advisory flag, because §3.4, which is the operative rule, does
not gate on it.

### 3.6 `k`, and the partial-computability term `p`

`k` is defined over [preregistration](../plan/preregistration.md) §4.1's **Claim-B comparator set**
— `graph6`, `sparse6`, `nauty_graph6`, `adjacency`, `agm_cam`, `min_dfs`, `wl_subtree` — and counts
those with **no admissible distance on any suite**. `sparse6_nauty`, `isalgraph_*` and `size_null`
are not comparators: the first is a T-04 addition outside the frozen family, the second is the
reference arm, the third is a baseline.

> ⚠ **preregistration §5 states `k ∈ 0–6`. The Claim-B comparator set has 7 members.** The range
> should be 0–7. Recorded now; it is moot unless `wl_subtree` is excluded.

`p` — representations with an admissible distance that cannot be **computed** on one suite — is
reported separately as `(representation, suite, rows lost)`. It is
[competitors/README](../plan/competitors/README.md) finding 6's unowned hole, and it requires a
dated amendment to preregistration §5 adding the case. **No p-value has been computed under the
current version**, so the changelog entry is clean.

### 3.7 Supersession, fixed before the outcome is known

- Where this run disagrees with [competitors/README](../plan/competitors/README.md) §3's provisional
  "primary distance" column, **this run wins** and §3 is corrected in place. That column was set by
  inspection during the T-04 scout and §3's own header says T-04a re-runs the grid under its own
  protocol.
- Where this run disagrees with `corrected_rho_table.json`, **that file wins on ρ** — it is a
  per-dataset 200-graph draw and T-04a's F5 is a different draw under different distances. The two
  are reported side by side and the difference is attributed to the draw, never averaged.
- F5 is computed **after** selection is written to disk, from a separate entry point, and is never
  fed back. If F5 contradicts the selection, the selection stands and the contradiction is printed.

### 3.8 F5 protocol

- **Suite-1 arm (primary, §3.3 as frozen)**: Spearman ρ against T-03's certified exact GED, per
  Suite-1 dataset, 200-graph draw per dataset at seed 42 (`Cohort.sample`), certified pairs only,
  each representation under **its selected primary distance**.
- **Suite-2 arm (descriptive, added by user decision 2026-08-16)**: ρ against T-05's `BRANCH_FAST`
  lower bound **and** `BIPARTITE` upper bound, per Suite-2 dataset, same draw protocol. Reported as
  two values, **never interpolated** ([approx_ged](../plan/approx_ged.md) §4's no-interpolation
  rule). Censored and un-encodable graphs are reported, not dropped silently.
- **Every ρ carries the size null `ρ(|n₁−n₂|, GED)`** in the same record, and a **graph-level
  bootstrap CI** (D2): 2,000 resamples of *graphs*, seed 42, percentile interval, the same resamples
  reused across representations within a dataset (D7). This discharges the T-04 board warning that
  graph-level bootstrap CIs are a precondition for printing any ρ.
- Both the all-pairs and the equal-`n` view are emitted (§4.2's argument).

---

## 4. Acceptance criteria

Each is checkable and names its artifact. Re-run by the orchestrator, not taken from a work log.

| # | Criterion | Proof |
|---|---|---|
| **A1** | `S200` is 200 graphs, reproducible from `(ALL_DATASETS, 200, 42)`, with the stratum quotas `[33,33,33,33,34,34]` | a test asserting the count, the quotas and byte-identical `graph_ids` across two calls |
| **A2** | All **66** cells attempted and present in the grid JSON, failures included | `len(payload["cells"]) == 66`; `padded_hamming × sparse6` present and marked undefined |
| **A3** | No metric with `consumes == "order"` is ever selected | a test asserting `select_primary` returns no `size_null` on a fixture where it would otherwise win on F6 |
| **A4** | Encode failures are recorded per representation per suite with the exception type; `agm_cam` reports `SuiteScopeError` on the Suite-2 graphs | the `f0` block of the grid JSON |
| **A5** | F3 runs on `S50` (stratified), encodes each copy once, and reports a skip count | a test asserting the F3 sample's stratum composition and that `attempted + skipped == 50` |
| **A6** | `grid.py`'s import closure still reaches no GED loader | the existing `test_competitors_core.py` closure test still passes |
| **A7** | The selection table names a primary distance, or "none admissible" with the failing criterion, for all eleven backends | the `primary_distance` block plus `excluded_because` on every non-selected cell |
| **A8** | `k` is stated with its membership, and `p` with `(representation, suite, rows lost)` | the report |
| **A9** | F5 emits, for every ρ, the size null and a graph-level bootstrap CI, in both the all-pairs and equal-`n` views, for Suite 1 (exact) and Suite 2 (LB and UB, uninterpolated) | the F5 JSON |
| **A10** | The four pre-committed outcomes ([competitors](../plan/competitors.md) §4 plus the size null) are each stated with the measurement that settles them, whichever way they fall | the report |
| **A11** | Full suite ≥ **2,106 passed / 321 skipped** (T-04's reference state), `ruff check` and `mypy --strict` clean | `$PY -m pytest tests/ -q`, `$PY -m ruff check src/ tests/`, `$PY -m mypy src/isalgraph/` |
| **A12** | Every plan file this run contradicts is corrected **in place**, not only in this note | `git diff` on `.claude/notes/review/plan/` |

---

## 5. Stop and ask

- The selection excludes a representation the manuscript already reports a number for, in a way that
  removes a table the paper needs.
- `k ≥ 5` — that removes ≥ 75 of F2's 182 tests and is a change to what the paper can claim.
- F5 under the selected distances moves any ρ in `corrected_rho_table.json` by more than the
  between-draw variability finding 14 records (0.07), which would mean the distance choice, not the
  draw, is doing the work.
- The `grid.py` import-closure test cannot be kept green — decision 24 stops being defensible and
  that is a PI decision, not a coding one.
- Any representation's F3 result contradicts the exhaustive `n = 2…6` proof in
  [competitors/README](../plan/competitors/README.md) §3.

---

## 6. Changelog

| Date | Change | Anything already computed under the old version? |
|---|---|---|
| 2026-08-16 | Initial freeze. §3.1–§3.8 fixed by this commit, before any agent started | no |
| 2026-08-16 | **§3.6: a suite with `attempted == 0` is not charged.** CONTRACTS §2 permits a zero-count suite and `frac` is then `0.0`, which would charge 5 or 10 rows for a property of the **draw** rather than of the representation. Live, not hypothetical: `S200` contains **zero LINUX graphs** — LINUX is 89 graphs at `n ∈ [4,10]` competing against thousands in the strata it falls in. Raised by track C, accepted | no |
| 2026-08-16 | **§3.6: a comparator absent from `primary_distance` aborts rather than raising `k`.** An unregistered backend is *not computable at all* — preregistration §5 charges that −25 and "records it separately" — so folding it into `k` undercharges by 10 and mislabels the case. Cannot arise here (all eleven backends registered and available, checked), so the tool refuses to guess. Raised by track C, changed on my ruling | no |
| 2026-08-16 | **§3.6's flagged `k ∈ 0–6` discrepancy is resolved upstream: `k` is now 0–7** in `preregistration.md`, matching §4.1's seven-member Claim-B comparator set. Fixed by T-06's owner on my report | no |
| 2026-08-16 | **§1.3's `isalgraph_pruned` F0 of 191/200 is a 2 s-budget artefact, not an encoder ceiling.** T-06's owner re-ran both canonical encoders on the C++ engine (`build_hash 298fc1188bf1b051`), 15 graphs/dataset, seed 42, killed-subprocess budget **30 s**: `pruned` completed **150/150 across all ten datasets, zero kills** (medians COIL-DEL 18.2 ms, Mutagenicity 126 ms, Protein 305 ms; worst single graph 28.3 s). **Every F0 this ticket prints must name its budget.** `isalgraph_canonical`'s 99/200 is unaffected — that is a `SuiteScopeError` capability refusal, not a timeout. Consequence recorded upstream: `competitors/README` finding 7b is a statement about the budget, not about the encoder | F0 measured under the backends' own budgets; reported as such |
| 2026-08-16 | **§3.8's `ρ − size_null` was computed against the wrong pair set, and two Mutagenicity numbers are retracted.** The null was taken over the cohort's **whole** pair set while the arm was taken over the representation's **own** pair set; wherever a representation loses graphs those are two correlations over two different samples and the difference is not a comparison. Mutagenicity is the only live case in this run — IsalGraph loses 14/200 to the 2.0 s budget, and **every censored graph is larger than every kept one** (mean 75.8 nodes, max 97, against 25.4 and 48), which collapses sd(\|n₁−n₂\|) from 16.4 to 8.0 and moves the *null* while leaving the arm at 0.8322 exactly. **RETRACTED: `::ub` +0.078 and `::lb` −0.295. The measured values are +0.196 and −0.289.** The restriction must be **per representation, not per dataset**: `min_dfs` is also censored 14/200 on Mutagenicity but on a different 14 graphs, restricted null 0.6817 against IsalGraph's 0.6363. Every cell now carries `size_null_on_my_pairs`. Fixed in `1651d6b`; recorded in [f5 log](../2026-08-16-t04a-metric-feasibility/f5.md) §9. **This is D14's censoring bias appearing inside the baseline rather than the arm — the direction nobody checks.** No verdict changes: all 15 differences still exclude zero and the Suite-2 flip is unchanged | **yes — and both were superseded.** The retracted pair survives in the closing brief and in `competitors/README` §7's range endpoint; both corrected 2026-08-23 |
| 2026-08-23 | **Closed.** Board row struck; `competitors.md` §9 RESULT appended; six plan files corrected in place (`competitors/README` §3 and §7, `nauty.md`, `agm.md`, `gspan-mdfsc.md`, plus RESULT blocks in `adjacency-matrix.md`, `graph6.md`, `sparse6.md`, `wl-subtree-kernel.md`); article notes and letter fragment written. **Acceptance criteria A1–A12 all met**, with A11 exceeded — the reference state moved from T-04's 2,106/321 to **2,322 passed / 321 skipped**. `k = 3` reported to T-06, which owns applying it to `preregistration.md` §7 | no |
