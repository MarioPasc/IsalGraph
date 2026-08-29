# Worklog — T-28: alternative similarity references for §5.4

**Ticket** T-28 · **Opened** 2026-08-29 · **Deadline** revision due 2026-08-31
**Design note (frozen)** `.claude/notes/review/tasks/T-28-design.md`
**Driver** PI request 2026-08-29: §5.4's GED correlation is weak enough to risk rejection;
add WL kernel and spectral λ-distance as alternative similarity references.

**Acceptance criterion (user):** the T-04a representation distances are *maintained*
(`levenshtein` for all competitors except `wl_subtree`, which uses `kernel`); only the
**reference** against which correlation is measured changes.

---

## 2026-08-29 — session 1

### Context secured

The Sandisk 2TB holding every T-06 artifact was about to be unplugged. Copied
`research/ISAL/completed/isalgraph/` (2.8 GB, 44,882 files) to
`/home/mpascual/research/data/isalgraph_archive/`; verified by file count and byte count on
both sides. Nothing else from the drive was copied.

### The decisive structural finding

**The representation side of the correlation is already computed and cached.**

```
data/source/T06/distances/{suite}/{dataset}__{rep}__{metric}.npz
    distance_matrix  float64 (G, G)     <- the expensive artifact
    graph_ids        <U16    (G,)
    node_counts      int32   (G,)
    defined_mask     bool    (G, G)
    metadata         0-d JSON (code_commit, build_hash, seed, ...)
```

515 MB across 15 (suite, dataset) cells × 7 arms. The reference matrices (`exact`, `lb`, `ub`)
live in separate trees and are joined on `graph_ids` at analysis time — the two sides are
**never** stored joined. Swapping the reference therefore costs **no re-encoding and no
re-computation of any representation distance**, which is exactly what the acceptance
criterion requires, and it is enforced structurally rather than by discipline.

`t06_f2.py::run_correlation_group` already takes `references: dict[str, ndarray]` and loops
over its keys, so adding `wl` and `spectral` is an extension, not a rewrite.

### Cost consequence

The 12 h `distances` SLURM stage is **skipped entirely**. Only the F2 statistics stage
(4 CPU / 24 G / 12 h wall, 15 shards) needs re-running, against ~2× the references. The
2-day budget is comfortable.

### Design decisions taken with the user before they left

1. **WL circularity.** The `wl_subtree` arm's own distance *is* the WL kernel distance, so
   under a WL reference it scores ρ = 1.0 by construction. Decision: report the row, mark it
   degenerate, **exclude it from win/loss counts**. Not silently dropped.
2. **Reporting.** Compute all references; produce the full grid **and** both manuscript
   variants (full-grid, winners-only); the PI decides at writing time.
3. **Spectral primary.** Normalised Laplacian `L_sym`, sorted spectrum, zero-padded,
   Euclidean — frozen before results, citing Wilson & Zhu, *Pattern Recognition*
   41(9):2833-2841 (2008).
4. **IsalChem p.7949 fallback.** Prepared in parallel on its own branch as insurance.

### Fast probe, and what it changed

Wrote `benchmarks/real_data/eval_reference_metrics/{spectral,probe_t28,summarise_t28}.py` —
point estimates only, no bootstrap — to answer the PI's question in minutes rather than
waiting on the full campaign. Smoke pass on AIDS + LINUX (`all_pairs`):

| reference | arm ρ (AIDS) | best competitor | **size null** | arm clears null? |
|---|---:|---:|---:|:--|
| GED exact | +0.2718 | +0.8213 `agm_cam` | **+0.8163** | no |
| **WL kernel** | +0.3154 | +0.4127 `min_dfs` | **+0.2127** | **yes** |
| spectral (norm L, padded) | +0.1359 | +0.8499 `agm_cam` | **+0.8717** | no |
| spectral (comb L, padded) | +0.3612 | +0.5990 `agm_cam` | +0.6265 | no |
| spectral (adjacency) | +0.1477 | +0.9312 `agm_cam` | +0.9592 | no |

Two results, and the second is the one that matters:

- **The zero-padded spectral distance is *more* size-dominated than GED** (null 0.87 vs 0.82).
  Mechanism, exact: `tr(L_sym) = n`, so `‖λ‖²` scales with `n`, and padding to `n_max` turns
  the Euclidean distance into a size proxy. **The padding convention carries the confound,
  not the normalisation.** This was not anticipated when the primary was frozen. It motivates
  `spectral_esd` (1-Wasserstein between eigenvalue *measures*), added as a declared,
  mechanism-driven variant — reported win or lose.
- **Under the WL kernel reference the size null collapses to ≈ 0.16–0.21 and IsalGraph
  clears it on 3/3 cells.** The WL kernel is a genuinely *structural* reference, not a size
  proxy. This directly addresses §5.4's headline concession — *"where the trivial baseline
  beats the representation, which competitor wins is second-order"* — which is the single
  most damaging sentence in the subsection. IsalGraph still loses the head-to-head to
  `min_dfs` on these two cells.

**Caveat, not yet resolved:** the probe uses per-arm pair masks, so it does not reproduce
T-06's `n_pairs` (234,258 vs 131,148 on AIDS) or its ρ (0.2718 vs 0.3266). T-06 intersects the
defined masks across the whole comparison group. The production run must use the group
intersection and reproduce T-06's `exact` column to 4 decimals — that is gate **G1**.

### Full probe — 15 cells, 1,070 records, point estimates

Raw records: `docs/worklogs/T-28-artifacts/probe_point_estimates.json`.

**Verdict against the *best* competitor** (the strictest reading), `all_pairs`:

| reference | win | loss | clears size null |
|---|---:|---:|---:|
| GED exact | 0 | 5 | 1/5 |
| GED lower bound | 0 | 10 | 0/10 |
| GED upper bound | 0 | 10 | 7/10 |
| **WL kernel** | **3** | 12 | **12/15** |
| spectral (norm L, padded) | 0 | 15 | 0/15 |
| spectral (comb L, padded) | 2 | 13 | 4/15 |
| spectral (adjacency) | 0 | 15 | 0/15 |

**Verdict per competitor** — the reading that answers the PI's *"gane a sus competidores"*.
Cells where the arm's ρ is higher, out of all cells; mean Δρ in brackets. `all_pairs`:

| reference | `agm_cam` | `min_dfs` | `nauty_graph6` | `sparse6_nauty` |
|---|---|---|---|---|
| GED exact | 1/5 (−0.225) | 0/5 (−0.154) | 3/5 (+0.053) | 4/5 (+0.136) |
| GED lower bound | 0/10 (−0.168) | 1/10 (−0.082) | 4/10 (+0.009) | 9/10 (+0.101) |
| GED upper bound | 6/10 (+0.106) | 1/10 (−0.042) | 2/10 (−0.034) | 2/10 (−0.024) |
| **WL kernel** | **15/15 (+0.177)** | 3/15 (−0.024) | **12/15 (+0.149)** | **15/15 (+0.206)** |
| spectral (norm L) | 0/15 (−0.268) | 1/15 (−0.126) | 7/15 (+0.015) | 11/15 (+0.114) |
| spectral (comb L) | 5/15 (−0.011) | 4/15 (−0.069) | 13/15 (+0.213) | 15/15 (+0.221) |
| spectral (adjacency) | 0/15 (−0.326) | 1/15 (−0.139) | 5/15 (−0.120) | 7/15 (−0.005) |

`wl_subtree` is excluded under the `wl` reference (degenerate, ρ ≡ 1.0 by construction).

### What this means

**The WL kernel reference carries the result; the spectral family does not.**

1. **Against the four non-degenerate competitors, IsalGraph wins three outright** — `agm_cam`
   15/15, `sparse6_nauty` 15/15, `nauty_graph6` 12/15 — and loses only to `min_dfs`, by a mean
   Δρ of **−0.024**, which is inside the interval width T-06 reported for comparable
   contrasts. This is the first reference under which the arm is not last.
2. **It clears its own `|Δn|` size null on 12 of 15 cells**, against 8 of 25 under GED and 1 of
   5 under *exact* GED. The mechanism is measurable: the size null against the WL kernel runs
   ρ ≈ 0.16–0.87 where against GED it runs 0.71–**0.997**. The WL kernel distance is a
   *structural* reference; graph edit distance on these cohorts is largely a size proxy.
   This is what repairs §5.4's most damaging sentence — *"where the trivial baseline beats the
   representation, which competitor wins is second-order"*.
3. **The spectral family is weaker and the primary variant fails outright.** The pre-declared
   `spectral` (normalised L, zero-padded) wins 0/15 against the best competitor and clears the
   size null 0/15. The padding artifact identified in the smoke pass is confirmed at full
   scale. `spectral_comb` is middling. `spectral_esd`, the size-controlled variant, is being
   built on track C and is not yet measured.

**Caveats, and they are not small.** These are point estimates with no confidence intervals,
computed on **per-arm** pair masks rather than T-06's group intersection, so they do not
reproduce T-06's `n_pairs` or ρ (AIDS: 234,258 pairs and ρ = 0.2718 here vs 131,148 and
0.3266 in T-06). **No verdict above is a significance verdict.** The production campaign must
use the group intersection, reproduce T-06's `exact` column to 4 decimals (gate G1), and
supply paired graph-level bootstrap intervals. A mean Δρ of −0.024 against `min_dfs` may well
be a tie; that is for the bootstrap to say, not the point estimate.

### 🔴 A guard the reference swap needed, and would not have got

`t06_f2._comparator_record` decided confirmatory-family membership from the
**representation alone**:

```python
in_family = comparator.representation in FAMILY_COMPARATORS
row = "B1e" if reference == "exact" else "B1a"
```

`N_actual = 79` and the Benjamini–Hochberg correction over it are pre-registered and
frozen. With that code, **every reference added by T-28 would have entered the family as a
`B1a` row and inflated `N_actual` past 79, with no error raised** — the pre-registered
correction silently invalidated, exactly the failure class this project has hit before
(the stale 726-test floor; the `padded_hamming` primary column; the `dgx` constraint).

Family membership is now a property of the **(representation, reference) pair**:

```python
CONFIRMATORY_REFERENCES: Final[frozenset[str]] = frozenset({"exact", "lb", "ub"})
in_family = (comparator.representation in FAMILY_COMPARATORS
             and reference in CONFIRMATORY_REFERENCES)
```

and `_reference_regime` returns a third label, `structural`, so a T-28 reference can never
merge into either GED regime's omnibus. `tests/unit/test_t28_plumbing.py` pins it with 15
tests, including an end-to-end check that the *same* comparator yields row `B1e` under
`exact` and no row at all under `wl`. The SLURM merge stage **asserts** `N_actual == 79`
rather than reporting it: T-28 adds only references, so a moved value is the guard failing,
not a finding.

Full unit suite after the change: **1,984 passed / 275 skipped**, no regressions.

### Reference matrices — built and independently verified

75 matrices (15 cells × 5 keys) in the dense CONTRACTS §4 schema, verified by the
orchestrator rather than taken from the building agent's log:

| gate | result |
|---|---|
| **G3** symmetric, zero-diagonal, finite, non-negative, joins on `graph_ids` | **75/75 clean** |
| **G4** `wl` matrix is the cached `wl_subtree__kernel` matrix | **15/15 byte-identical** |
| **G5** off-diagonal exact-zero fraction < 0.99 | max **0.155**, mean 0.042 |

G4 matters: the WL degeneracy is exact by construction, not approximate, so `ρ = 1.0` for
that arm is provably the identity rather than a bug hiding behind a near-miss.

### Picasso

The campaign reuses everything and recomputes nothing on the representation side, so
**there is no distance stage** — the 12 h that cost T-06 is skipped outright. Picasso
already held a complete `distances` tree (`T06_exhaustive/distances`, a superset of T-06's
arms) and `APPROX_GED`; only the exact GED and the new reference matrices had to be staged.

- `slurm/t28_metrics/{launcher.sh,f2_worker.sh}` — the worker is a **copy** of T-06's, so
  that campaign's frozen script stays byte-identical, plus `FAM_ROOT` (T-28 must not
  overwrite the families it reads distances beside), a validated `T28_REFERENCE_ROOT` (unset,
  every shard would run a successful GED-only recompute and emit no structural row at all),
  and the `N_actual` assertion above.
- Staged to a **separate checkout** `repos/IsalGraph-t28`: the shared `repos/IsalGraph`
  carries uncommitted local edits and is the target of the env's editable install, and
  clobbering someone's uncommitted work on a shared cluster is not a thing to do
  unprompted. The only `import isalgraph` in the F2 chain is inside `_metadata()` for
  provenance, so the package resolving from the older tree changes no computed number — but
  the manifest's `src_commit` will describe that tree, which is recorded here rather than
  discovered later.
- Environment verified live: numpy 2.4.6, scipy 1.17.1, the guard importable, and
  `T28_REFERENCE_ROOT` picked up at module import.

### Track B (IsalChem p.7949 fallback) — reasoned negative, no code

Page 7949 is the *Molecular Similarity* subsection and defines **seven binary-fingerprint
metrics** (Tanimoto, Dice, Cosine, Kulczynski, McConnaughey, Russel, Sokal) over RDKit
Morgan fingerprints — there is no explicit subgraph repertoire; the "repertoire" is the
fingerprint's hash-implicit circular-subgraph vocabulary.

All seven transfer at the formula level (they are pure bit-vector operations, no chemistry),
and the label-stripping is *not* the barrier. **Size domination is.** Graphlet counts scale
linearly with `n` for sparse graphs, so a graphlet-fingerprint Tanimoto obeys
`Tani(G,H) ≤ n_min/n_max` — a pure size ratio, which on COIL-DEL (size null 0.9971) buys
nothing; frequency normalisation collapses to near-1 uniformly instead. Recommendation:
**do not implement**. The repertoire specification is recorded for a future revision.

This is a useful negative: it means WL is the answer, not a way-station.

### Status

- [x] Archive secured off the Sandisk (2.8 GB, verified by file count and bytes)
- [x] Design note frozen and committed before results
- [x] Fast probe written; smoke pass done; full 15-cell run done
- [x] Full probe results reviewed — **WL kernel is the load-bearing reference**
- [x] Production reference matrices built and gate-verified (track C)
- [x] `t06_f2` reference plumbing + the family guard + 15 tests
- [x] SLURM launcher/worker; repo and exact GED staged on Picasso
- [x] IsalChem p.7949 fallback: reasoned **do not implement** (track B)
- [x] `spectral_esd` measured — does **not** rescue the spectral family
- [x] Agent branches verified and merged; suite **2,019 passed / 275 skipped**
- [x] Campaign submitted on Picasso (`2132238` shards → `2132239` merge)
- [x] `rho_vs_size` figure for the WL kernel reference
- [ ] Campaign completes; results copied back
- [ ] §5.4 rewrite drafted

---

## 2026-08-29 — session 2

### The `min_dfs` gap is a TIE, and that is now measured rather than inferred

The point estimate put the arm 0.024 below `min_dfs` under the WL reference, which is not a
verdict. The Picasso array was still sitting on `Priority/` with 0 of 15 partials, so the
shards were run **locally** off the same cached matrices to get the paired graph-level
bootstrap. On both LINUX cells the difference's 95 % interval covers zero:

| reference | comparator | arm − comparator [95 % CI] | verdict |
|---|---|---|---|
| GED exact | `min_dfs` | −0.1691 [−0.2785, −0.0784] | **LOSS** (p = 0.002) |
| **WL kernel** | `min_dfs` | **−0.0235 [−0.1470, +0.0923]** | **TIE** (p = 0.70) |
| **WL kernel** | `min_dfs` (suite 2) | **−0.0413 [−0.1326, +0.0508]** | **TIE** (p = 0.37) |
| WL kernel | `agm_cam` | +0.1831 [+0.0459, +0.3073] | WIN |
| WL kernel | `nauty_graph6` | +0.2237 [+0.0905, +0.3563] | WIN |
| WL kernel | `sparse6_nauty` | +0.2466 [+0.1378, +0.3524] | WIN |

**So under the WL kernel reference the arm is beaten by nothing**: three significant wins and
one tie, against a significant loss to the same comparator under exact GED. `verdicts_t28.py`
computes this tally from `difference_vs_reference_arm` — the paired difference on identical
pairs under one graph-level resample — never from two overlapping marginal intervals.

**Scope, stated plainly:** two of fifteen cells. The Picasso campaign is the record and will
extend it; the tie claim above is asserted only over what has been measured.

### The figure

`docs/worklogs/T-28-artifacts/fig1_rho_vs_size_wl.{pdf,png}`, from
`size_profile_wl.json` (1,199 stratum rows, 326 aggregate points, all 15 cells).

**One panel, one x axis, every competitor together** — not the two-regime layout of the GED
figure. That split exists because graph edit distance is exact only to `n = 12` and is a
*bracket* above it, and a bracket cannot share an axis with an exact value without inviting
the reader to read one as the other. The WL kernel distance is computed exactly at every
size, so there is no bracket, no ceiling and no reason to split. Same styling, same
Fisher-z aggregation, same local BH treatment.

`wl_subtree` draws as a flat line at ρ ≡ 1.0 and is **annotated on the figure**. It is the
reference, so this is the identity, not a competitor solving the problem. Hiding it would
have been a silent exclusion; leaving it unlabelled would have been worse.

Two implementation notes worth keeping:

- **The published `size_profile.py` could not produce this in time.** It re-derives every
  Levenshtein block per stratum from the encodings and managed **2 of ~120 (dataset,
  representation) units in fifteen minutes**. The matrices it was recomputing were already on
  disk. `size_profile_cached.py` reads them and imports `MIN_PAIRS`, `N_BOOTSTRAP`, `SEED`
  and `_bootstrap_ci` from the original module so the statistics cannot drift.
- **The per-stratum bootstrap is not read by the figure.** `figures.aggregate` derives its
  own interval from the Fisher-z weighted mean of `rho` and `n_graphs` and never looks at
  `ci_lo`/`ci_hi`. Skipping it took the full 15-cell profile from *hours* to **21 seconds**.
  `--no-bootstrap` is therefore correct for a figure run and wrong for any table quoting a
  per-stratum interval, and the docstring says so.

### ⚠ A width defect the new figure inherits, and did not fix

`05_results.tex` carries an open BUILD RISK: `rho_vs_size.pdf` is rendered **7.03 in** wide
and placed in Pattern Recognition's **4.72 in** text block, so `width=\textwidth` scales it
by 0.67 and its declared 5.5–6.5 pt labels reach the page at 3.7–4.4 pt. **The WL figure is
7.03 in too** — it matches its sibling, which is what was asked, and inherits the same defect.

A `--width` option was added, and **measuring it showed the option is not sufficient on its
own**: `save_figure` writes with `bbox_inches="tight"`, so the output box is the *content*
box, and at 4.72 in the seven-column legend and the title overflow and the box expands back
to 7.03 in — with nothing in the output to say so. Both renders came back 7.03 in. A genuine
narrow render also needs a narrower legend and a shorter title, which is a different figure,
so the misleading `_patrec` variant was deleted rather than shipped. The trap is documented
in the docstring. **`IEEE_TEXT_WIDTH_INCHES = 7.0` was not touched** — a test pins it to the
submitted PDF.

### `spectral_esd`, measured

The size-controlled variant works *as a size control* — its size null on LINUX is **0.3078**
against 0.8296 for the padded normalised spectrum — but the canonical string does not track
it: arm ρ = 0.1524 there. Across all 15 cells it beats `sparse6_nauty` 15/15 and
`nauty_graph6` 12/15, but only `agm_cam` 6/15 and `min_dfs` 3/15, which is strictly worse
than WL. **Adding it changed nothing about the conclusion, and it is reported anyway** —
that was the condition under which it was added (§5 of the design note).

Running the probe a second time against the *built* production matrices also cross-validates
track C's independent implementation against the prototype: the WL row is identical
(15/15, 3/15, 12/15, 15/15), as it must be, since both read the same cached kernel matrix.

### Submission, and one blocker the smoke test caught

A login-node smoke run on `suite1/linux` failed with

```
FamilyError: c names an undeclared representation 'isalgraph_exhaustive'
```

`T06_exhaustive/encodings` carries `isalgraph_exhaustive` and `isalgraph_greedy` on top of
T-06's arms. Under `T06_REFERENCE_ARM=isalgraph_pruned` those two are undeclared, the
regenerated completion table names them, and `family.validate()` rejects it — **all fifteen
shards, after a queue wait `sbatch --test-only` was then estimating at ~10 hours.** Fixed by
pointing `OUT_ROOT` at T-06's own distances and encodings, which is what gate G1 compares
against anyway. Those two trees (190 + 166 files, 527 MB) were staged to Picasso.

Smoke re-run through the real worker: **green, 49 s**, 72 rows over 6 references. It
confirmed three gates live before anything queued:

| gate | evidence |
|---|---|
| **G1** retained GED column unchanged | `arm/exact ρ = +0.4850` on **1,685** pairs — T-06's published value and pair count exactly |
| **G4** WL degeneracy is the identity | `wl/wl_subtree ρ = 1.0` exactly |
| family guard | all **60** structural rows carry `in_family=False`, `row=None`; the 12 GED rows still carry 8 `True` |

And the headline, already visible on LINUX with bootstrap intervals rather than point
estimates — the excess of the arm's ρ over its own size null:

| reference | arm ρ | size null | **excess [95 % CI]** |
|---|---:|---:|---|
| GED exact | +0.4850 | +0.7097 | **−0.2247 [−0.3492, −0.0922]** — significantly *below* |
| **WL kernel** | +0.4798 | **+0.1609** | **+0.3189 [+0.1699, +0.4454]** — significantly *above* |
| spectral (norm L) | +0.2384 | +0.8296 | −0.5912 [−0.7004, −0.4645] |
| spectral (comb L) | +0.6060 | +0.5780 | +0.0280 [−0.1373, +0.1968] |
| spectral ESD | +0.1524 | +0.3078 | −0.1554 [−0.3146, +0.0058] |

**Campaign**: array `2132238` (15 shards, one per task, 10 h wall, 4 CPU / 32 G, CPU
partition) → merge `2132239` on a verified `afterok`. The first submission used 8 tasks and a
20 h wall and sat on `Priority/`; one shard per task with a 10 h wall backfills far better,
which matters against a 2026-08-31 deadline.

**No distance stage ran.** Every representation distance is T-04a's, read from cache.

Results land at `T28_metrics/families/rho_table.json` on Picasso. Pull them from either
workstation with `bash slurm/t28_metrics/fetch_results.sh` (`--status` to look without
fetching); it is safe to run mid-campaign and names the missing shards.
