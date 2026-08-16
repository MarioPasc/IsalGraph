# stats — work log

**Branch** `t06/stats` · **Base** `8afa59e` (worktree cut at `863217b`, merged orchestrator `e960fa8`) · **Head** `87e7f36`

## What I built

A new package `benchmarks/real_data/eval_stats/` holding the T-06 statistics engine and the frozen
confirmatory-family runner. The resampling unit is the graph everywhere, which is the whole content
of the R3.5c fix: D2's cluster bootstrap carries all uncertainty, D3's Mantel test carries all
significance, and the pair-level `bootstrap_correlation` is unreachable by object identity rather
than by convention. `family.py` declares F0/F1/F2 as a fixed-sequence gatekeeping design and defines
`N_actual` by **enumerating** the admissible cell set, with the closed form printed beside it as a
check; that enumeration is what caught the `+ k·d` defect in the frozen pre-registration.

| File | Purpose | Lines |
|---|---|---:|
| `benchmarks/real_data/eval_stats/__init__.py` | Package docstring; names the replaced defect | 30 |
| `benchmarks/real_data/eval_stats/resampling.py` | D2 cluster bootstrap, D7 differences, D15 tier lookup, percentile intervals, bootstrap p-values | 432 |
| `benchmarks/real_data/eval_stats/association.py` | D1 Spearman/Kendall, condensed pair variables, D3 Mantel wrapper, D4 MRM + partial Mantel | 750 |
| `benchmarks/real_data/eval_stats/multiplicity.py` | D9 Benjamini–Hochberg + FCR levels, D8 Friedman/Wilcoxon–Holm/critical difference | 532 |
| `benchmarks/real_data/eval_stats/family.py` | F0/F1/F2 declaration, cell enumeration, `N_actual`, gate rules, F2 runner | 885 |
| `benchmarks/real_data/eval_stats/matrices.py` | Read-only `.npz` loaders for T-05 GED and T-06 distance files; identifier join | 250 |
| `tests/unit/test_t06_stats.py` | 65 tests covering every acceptance criterion | 968 |

### What I reused, and what I rewrote

**Reused verbatim, by import.**

- `ged_bakeoff_analysis.replicate_selection`, `.induced_pairs`, `.pair_flat_index` — the graph
  resample and the closed-form inverse of `np.triu_indices`. The seeding rule
  `SeedSequence([seed, replicate])` is the repository convention; a test asserts my entry point
  produces byte-identical draws to **both** existing implementations.
- `ged_bakeoff_analysis.factorize`, `.midranks`, `.spearman_from_ranks` — the counting-sort ranking
  path that makes D15's "re-rank inside every replicate" affordable.
- `ged_pair_index.pairs_from_indices_searchsorted` — tier-3 slot-pair decoding.
- `correlation_metrics.mantel_test` (D3) and `correlation_metrics.holm_bonferroni` (D8 post-hoc).
  Both are correct; `mantel_test` already permutes graph labels jointly on rows and columns. E10
  noted it had never been reported — now it is wired into the family.

**Rewritten, with the reason.**

- The **bootstrap driver** itself. `bootstrap_dataset` is welded to the bake-off's `IndexData` /
  `CellData` shape and to its fixed statistic list; `bootstrap_slope_ci` is welded to an OLS slope.
  `resampling.cluster_bootstrap` takes an arbitrary `statistic(flat_pair_indices) -> {name: value}`
  and is the single tested entry point the ticket asked for. Same mechanism, same seeds, no third
  algorithm.
- The **tier-3 subsample**, reimplemented over pair indices instead of over a regression fit, but on
  the same `SeedSequence([seed, replicate, 1])` substream. A test asserts it reproduces
  `approx_ged_analysis._bootstrap_slopes_subsampled`'s draw exactly.

**Deliberately not reused: `approx_ged_analysis`'s `np.bincount` / `np.einsum` weighted-sums
identity.** It computes a matrix's sum over the induced pair multiset without materialising it,
which is exact for an OLS slope and **inapplicable to Spearman** — a rank is not a sum, and D15
requires re-ranking inside every replicate. Using it would have silently reintroduced full-sample
ranks.

**Never imported: `correlation_metrics.bootstrap_correlation`.** It resamples pairs
(`correlation_metrics.py:242`). `statistics.md` §11 lists it as *replaced, not supplemented*.

## Acceptance criteria

| # | Criterion | Command | Result (verbatim) |
|---|---|---|---|
| 1 | Graph-level CI wider than pair-level | `pytest -k graph_level_interval` | `PASSED`. On **real LINUX**: graph-level width `0.11580`, pair-level width `0.02879`, **ratio 4.02×**. 89 graphs, not 3,916 independent observations. |
| 2 | Pair-level bootstrap unreachable | `pytest -k pair_level_bootstrap` | `PASSED`. Checked by **object identity** over every loaded `benchmarks.*` module, not by grepping source. |
| 3 | Tier 3 resamples graphs first | `pytest -k tier3` | 3 tests `PASSED`. Budget = all slot pairs reproduces the tier-1 multiset exactly; the graph draw is bit-identical across budgets; the subsample matches `approx_ged_analysis`'s. |
| 4 | BH correct | `pytest -k benjamini or fcr` | 5 tests `PASSED`. Hand example `p=(0.001,0.008,0.039,0.041,0.042)`, `m=5` → `(0.005,0.020,0.042,0.042,0.042)` via `assert_allclose`; the step-up `cummin` case (raw scaled values `0.065`, `0.05125` are non-monotone) is the one asserted. Cross-checked against `scipy.stats.false_discovery_control` on 20 random families of 17. |
| 5 | `N_actual` arithmetic | `pytest -k n_actual or removed_by_two or c_never or c_charges` | 8 parametrised cases + 6 targeted `PASSED`. `k=d=c=0` → **182**. Double-removal case `k={min_dfs}, d={coil_del}`: enumeration **160**, corrected closed form **160**, uncorrected **159**. |
| 6 | F0/F1 branch rule | `pytest -k gate or f0_ or f1_` | 9 parametrised boundary cases + 5 family tests `PASSED`. Includes CI excluding 0 at `|estimate| = 0.04` → **does not fire**, and `|estimate| = 0.05` exactly → **does not fire** (`>` not `>=`). |
| 7 | No omnibus on the exact regime | `pytest -k friedman or f2_runner` | `PASSED`. `friedman_omnibus(..., Regime.EXACT)` returns `ran=False`, `statistic=nan`, the locked reason, and **keeps the average ranks** so the regime stays descriptive rather than silent. `run_f2` emits only `A2` and `B2`, both `n_datasets=10`; `exact_regime_omnibus is None`. |
| 8 | MRM recovers a planted β₁ | `pytest -k mrm or partial_mantel` | 3 tests `PASSED`. Planted β₁ recovered to `rtol=0.15` and inside the bootstrap CI, permutation `p ≤ 0.01`. Collapse case: marginal ρ `> 0.4` while `|β₁| < 0.15`. |
| 9 | End-to-end on real matrices | `pytest -k end_to_end` | `PASSED`. See the block below. |
| 10 | Suite green | `pytest tests/unit/ -q` | `1789 passed, 50 skipped, 1 warning in 374.74s`. `ruff check benchmarks/ tests/` → my files clean (29 pre-existing errors in files I do not own; baseline at `e960fa8` is 28, and the delta is one auto-fix ruff applied to `starting_node_sensitivity.py`). `mypy src/isalgraph/` → `Success: no issues found in 69 source files`. |

### Criterion 9 — real LINUX, verbatim

```
LINUX G=89 pairs=3916 LB min=0.0 certified=247
  rho_lb: rho=0.7917 CI=[0.7260,0.8418] width=0.1158 tau_b=0.6364
  rho_ub: rho=0.3237 CI=[0.2280,0.4147] width=0.1868 tau_b=0.2304
  F1 diff: +0.46791 CI=[+0.39952,+0.54035] p=0.0010
  gate: (ci_excludes_zero=True, exceeds_threshold=True, fails=True)
  graph-level width=0.11580  pair-level width=0.02879  ratio=4.02x
  MRM betas=[0.3371, 0.7082, -0.0703] beta1_CI=[0.2637,0.4152] perm_p=0.0020 R2=0.8903
  k=0 d=0 c=0: N_actual=182 closed=182 disc=0 dbl=0
  k=1 d=2 c=0: N_actual=153 closed=153 disc=0 dbl=2
  k=2 d=3 c=10: N_actual=124 closed=124 disc=0 dbl=6
```

2,000 replicates, tier 1 as frozen. `LB min = 0.0`: **GED is legitimately 0** and nothing here
rejects it.

> ⚠ **The Levenshtein matrix in that run is a SURROGATE, not the real one.** The distance track owns
> the real one and had not produced it. I synthesised a conforming CONTRACTS §4 `.npz`
> (`distance_matrix`, `graph_ids`, `node_counts`, `defined_mask`, `metadata`) as `1.3·LB + N(0, 2)`,
> symmetrised, and loaded it through the real loader. **None of the numbers above is a scientific
> result.** The `F1 diff = +0.468` is an artefact of the surrogate being built from LB; the
> `MRM β₁ = 0.34 < β₂ = 0.71` likewise. What the run demonstrates is that the loader, the identifier
> join, the D7 shared-resample difference, the FCR interval and the F1 branch all execute on real
> 89-graph matrices and produce finite, ordered, in-range values. Re-run every one of these against
> the real Levenshtein matrix before quoting any of it.

## Decisions I made, and why

1. **`c` is keyed by `(suite, dataset, representation)`, not `(dataset, representation)`.** Suite 1
   and Suite 2 are different cohorts even where the name matches (`aids` 769 graphs vs
   `aids_graphedx` 819). Keying on the dataset name alone would apply a Suite-2 completion rate to a
   Suite-1 cell. *Rejected*: a single flat dataset key.
2. **Kendall τ-b is a full-sample point estimate, not bootstrapped.** D1 assigns it the role of a
   tie-robustness *check*, not an inference target, and re-ranking it inside 2,000 replicates over
   2 M pairs buys nothing the Spearman CI does not already carry. **This is not an omission** — the
   missing τ-b interval is deliberate and should be described as such in the manuscript. *Rejected*:
   bootstrapping both, which would have roughly tripled the D15 budget for a robustness check.
3. **The MRM carries two forms of inference, not one.** D4 says "permutation inference"; D2 says all
   uncertainty comes from the graph-level bootstrap. A permutation distribution is a **null**, not an
   interval, so I emit the permutation p-value (Legendre, Lapointe & Casgrain, *Evolution*
   48(5):1487–1499, 1994 — response-matrix labels permuted jointly on rows and columns) **and** a
   graph-level bootstrap percentile CI on standardised β₁. *Rejected*: calling a permutation quantile
   range a "CI", which is what a literal reading of D4 would have produced.
4. **Suite-1 D15 tiers live in their own table.** `statistics.md` §5 states that in the exact regime
   only IAM Letter HIGH reaches tier 2 and no Suite-1 dataset is subsampled. Sharing one table with
   Suite 2 would have silently applied `coil_del`'s tier 3 to a Suite-1 cohort. Confirmed by the
   orchestrator. *Rejected*: one shared lookup.
5. **The exact-regime refusal keeps the average ranks.** `friedman_omnibus(..., Regime.EXACT)`
   returns `ran=False` with the locked reason *and* the per-method mean ranks, because §4 says the
   exact regime is *descriptive*, not silent. *Rejected*: raising, which would have made the caller
   choose whether to report anything.
6. **`run_f2` warns rather than shrinks when a cell has no p-value.** An admissible cell missing its
   p-value stays in the BH denominator. Dropping it would shrink `N_actual` below what the data
   forces, which is the anti-conservative direction §5.1 rejects.
7. **`percentile_interval` never returns the bootstrap mean as the point estimate.** The resample
   carries uncertainty; the estimate is the full-sample value.

## Assumptions I recorded rather than blocking on

| Assumption | Told `main`? | Status |
|---|---|---|
| "BH-adjusted CI" means the FCR-adjusted interval of Benjamini & Yekutieli, *JASA* 100(469):71–81, 2005, at coverage `1 − Rq/m` | yes, item 2 of my defect message | **Accepted and written into `preregistration.md` §2.** Orchestrator added that widening the interval makes the gate *harder* to trip, i.e. the conservative direction |
| Suite-1 D15 tiers are a separate frozen table | yes | Accepted |
| Kendall τ-b is not bootstrapped | yes | Accepted; instructed to state it explicitly here, done above |
| The end-to-end Levenshtein matrix is a synthesised surrogate | stated here and in the test docstring | Unblocked by design — the ticket authorised it |

## What I could NOT do, and why

1. **No real Levenshtein distances anywhere.** The distance track owns them and had not emitted any
   when I finished. Every ρ, every β₁ and every gate decision in this log is on a surrogate or on
   synthetic data. **The engine is validated; no scientific claim is.**
2. **No production campaign.** Prohibition 3. Every bootstrap here runs at 200–2,000 replicates on
   one dataset. The orchestrator runs the family.
3. **No timings.** Prohibition 4. Three agents share this workstation.
4. **`c` is not determined.** It is an *input* to my runner and arrives from the encoding track via
   the orchestrator. Likewise `k`, from T-04a. Only `d` is computed here, by F1.
5. **F2's per-cell p-values are not produced.** `run_f2` consumes a `{Cell: p}` mapping; producing
   the 182 p-values needs the real encodings (A1), the real distances (B1e/B1a) and the real GED
   references. That is the orchestrator's production run, and the machinery for it is complete.
6. **The critical-difference *figure* is not drawn.** `multiplicity.critical_difference` returns the
   diagram data (sorted ranks, CD, cliques); drawing goes through `isalgraph.viz`, which I am
   forbidden to edit (`src/isalgraph/` is read-only to me).

## Contract defects found

**1. `preregistration.md` §5 closed form double-charged `k·d` cells. ⇒ FIXED by the orchestrator in
`e960fa8`.** This is the significant one and it was going to reach the paper.

`k` removes `(B1a, R, D)` for **all ten** Suite-2 datasets; `d` removes `(B1a, R', D)` for **all
seven** comparators. The intersection is exactly the `k·d` cells with `R` excluded and `D`
uninformative, and the frozen `182 − 15k − 8d − c` charged each of them twice. §5.2's no-double-count
rule was written for `c` alone and was silent on the `k`/`d` overlap. The overlap is *complete* —
`A1` is untouched by either term, `B1e` is indexed by Suite-1 datasets so `d` cannot reach it, and
`B3a` carries no representation index so `k` cannot.

The direction is what made it urgent rather than cosmetic: over-charging reports an `N_actual`
**below** the admissible count, which **lowers the BH burden on every surviving test** — the same
anti-conservative failure that got the per-representation `s` term rejected the same morning,
reappearing in the arithmetic instead of in the rule. Worked example `k=1, d=2`: frozen form 151,
truth 153. Closed form is now `182 − 15k − 8d + k·d − c`.

**2. "BH-adjusted CI" (§2 and §3 gate rules) named no computable object. ⇒ FIXED, §2.** A percentile
interval carries no multiplicity control, so the gate was unimplementable as written. Resolved as
the FCR-adjusted interval at coverage `1 − Rq/m`.

**3. F0's majority branch had no coefficient (§2: "N_actual drops accordingly"). ⇒ FIXED, new §5.3.**
The orchestrator froze it at **81 cells** demoted (B1a 70 + B2 1 + B3a 10), leaving **101**, and
resolved two things my report left open: `d` is **not applied at all** on that branch (F1 tests the
bracket *within* a regime that is now descriptive), and `k` charges **5, not 15** (the ten B1a rows
are already gone). Closed form on that branch: `101 − 5k − c`. Full precedence is now
**F0-demotion → `k` → `d` → `c`**, implemented in `family.admissible_cells` and covered by a
six-case parametrised test including one asserting `d` is ignored.

**Kept deliberately, now that the formula is right:** the enumeration stays authoritative, both
values print, `FamilyCardinality.discrepancy` is emitted on every run, and `double_charged` names the
overlapping cells by row/suite/dataset/representation. `discrepancy` is now 0 on every case I test —
which is the point. The machinery is what caught this, and it must survive to catch the next one.

**4. Not a defect, a note for whoever writes the response letter.** `matrices.align` widens every
`graph_ids` array to `<U16` before any set operation, per the orchestrator's mid-task amendment. A
test drives a join where the two files disagree on order, on dtype width and on membership, and
asserts a positional join would have read a different number.
