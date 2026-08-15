# Exact GED — Suite 1, the calibration anchor

**Owner**: T-03 (production) and T-05 (gates) · **Serves**: R3.5a, R3.5b, R1.3a, AE.1, E2/F2
**Status**: method LOCKED (decision 11); **execution structure revised 2026-08-12** to two stages.

Related: [gedlib](gedlib.md) (how to run it) · [approx_ged](approx_ged.md) (what happens above n=12) ·
[data](data.md) (the cohort) · [statistics](statistics.md) (what is done with the numbers)

---

## 1. Method assignment

| Role | Method | Source |
|---|---|---|
| **Primary** | GEDLIB **`ANCHOR_AWARE_GED`** | Blumenthal & Gamper; exact when run to completion, `LB == UB` certifies optimality |
| **Cross-check** | `networkx.graph_edit_distance` (A*) | the submitted study's solver |

**Benchmark the two before the production run.** GEDLIB is specialised C++ and `networkx` is Python;
if `ANCHOR_AWARE_GED` is materially faster it **raises the exact-GED ceiling above n = 12** and
enlarges the calibration regime — the single cheapest way to strengthen the whole design. This is
now a required step of the calibration ladder ([approx_ged](approx_ged.md) §3), not an opportunistic
extra.

Cost model: `[1, 1, 0, 1, 1, 0]` — see [gedlib](gedlib.md) §6.

---

## 2. Scope and cost

Every pair of every connected graph in the **five original datasets**. No subsampling of the graph
set, no split structure, no reliance on GraphEdX's within-split coverage. Exact GED is unobtainable
on the extension cohort, so Suite 1 is the whole exact-GED story.

| Dataset | connected | **all pairs** | ~s/pair | core-hours |
|---|---:|---:|---:|---:|
| Letter LOW | 1,180 | 695,610 | 0.004 | 0.8 |
| Letter MED | 1,253 | 784,378 | 0.004 | 0.9 |
| Letter HIGH | 2,059 | 2,118,711 | 0.008 | 4.7 |
| LINUX | 89 | 3,916 | 2.17 | 2.4 |
| **AIDS (GraphEdX)** | 769 | **295,296** | 12–20 | **985–1,640** |
| **Total** | | **3.90 M** | | **≈ 1,000–1,650** |

**16–26 h on 64 cores. AIDS is 98 % of it.** These counts use the correct `KEPT_ge2` population —
do **not** re-cost from the superseded `data.md` §3.1 table, which mixes n≥1 and n≥2 and is inflated
by 22,698 pairs (audit I-03).

### Why recompute at all

Two drivers, **both self-found, neither a reviewer demand**:

1. **E2 / F2 — GraphEdX ships GED only *within* train/val/test splits.**

   | Dataset | splits | within-split pairs | all pairs | coverage |
   |---|---|---:|---:|---:|
   | LINUX | 53 / 17 / 19 | **1,685** | 3,916 | 43.0 % |
   | AIDS | 546 / 182 / 183 | **181,909** | 414,505 | 43.9 % |

   The published LINUX ρ = 0.433 and AIDS ρ = 0.349 are **within-split figures, undisclosed**.
   Missing AIDS pairs = 295,296 − 181,909 = 113,387 → **378–630 core-h**, independent of any cost
   model. This corrects the earlier attribution of the LINUX 3,916 → 1,685 drop to the
   `GED > 0` / `Lev > 0` filter: it is **missing ground truth, not filtering**.
2. **D6 — one cost model.** The submission mixes IAM unit costs with GraphEdX topology-only costs on
   one axis. Under the unified model every AIDS and LINUX value must be recomputed regardless.

R3.5b's *literal* ask is answered for free by D5 (per-dataset primary, pooled demoted). The
recompute is a deliberate choice to **retire** the objection rather than caveat it.

---

## 3. Run it in two stages — the structure that keeps T-03 off the critical path

> **The census is disproportionate to every demand that drives it, and this plan contains the proof.**
> [statistics](statistics.md) **D2** — locked, and the literal answer to R3.5c — resamples **graphs**,
> not pairs, because pairs are dyadically dependent. Risk R1 draws the consequence: effective sample
> size is governed by the **number of graphs**. AIDS contributes **769 graphs** whether we compute
> 131,148 pairs or 295,296.
>
> **If D2 is right, the census buys no precision. If the census buys precision, D2 is wrong and R3.5c
> is unanswered.** Both cannot hold.

| Stage | Content | Cost | Role |
|---|---|---|---|
| **1** | Stratified sample over **all 769 AIDS graphs**, spanning every density and size stratum [statistics](statistics.md) §7 needs | **~100 core-h**, hours | **Pre-declared as the reported analysis.** Unblocks T-06 |
| **2** | The full 295,296-pair census, submitted at the same time, unattended | 985–1,640 core-h | **Supersedes stage 1 only if it lands before the T-20 text freeze** |

**The supersession rule is fixed now, before either runs**, so the choice between two ρ values cannot
be made after seeing which is more favourable. Letter wording either way: *"GED was recomputed under a
single unit cost model over a stratified sample of N pairs spanning all 769 graphs / over all 295,296
pairs."*

What the census genuinely buys, stated fairly: a ladder that says "all pairs" rather than "a
stratified sample" (rhetorically stronger against R3.5a); no sampling design to defend; more pairs
per density stratum — though the independent-unit count per stratum is again the number of graphs
in it.

**Saving: ~900–1,550 core-hours off the critical path, 2–5 elapsed days, census kept.**

---

## 4. Validation gates — before any production run

### Gate 0 — GraphEdX agreement (gates T-03, runs first)

Recompute ~500 **within-split** AIDS pairs under **GraphEdX's own** cost model and assert exact
agreement with the published matrix. If they disagree, our solver or our configuration is wrong and
everything downstream is invalid.

> ## ⚠ CORRECTED 2026-08-13 — T-03's retraction had not reached this file
>
> **GraphEdX charges UNIT node costs, the same model as D6.** T-03 measured its published AIDS values
> against both models: **unit-node 4/4, zero-node 0/4**, the published value exceeding the zero-node
> value by exactly `|n₁ − n₂|` every time. Commit `041a70c`.
>
> **The gate's configuration is therefore `[1, 1, 0, 1, 1, 0]` — the same as production.** Running it
> under `[0, 0, 0, 1, 1, 0]` is what produced gate 0's 150-below / 58-equal / 0-above result, which
> was read as evidence that the published matrix is approximate and **was not**: it was the arithmetic
> of the wrong cost model. Measured like-for-like over the full 131,148-pair overlap, ours never
> exceeds theirs and the two agree on all but two pairs.
>
> **This is the inherited-premise trap in its exact form**: the plan asserted zero node cost, the gate
> was configured from the assertion, the clean one-sided result was read as a fact about the data, and
> the "independent verification" shared the same premise. **Anyone configuring a gate from a plan
> assertion should test the assertion against the data first.**
>
> What survives: the gate's *purpose*. Agreement with a published matrix **validates the solver**, not
> our cost model — that is justified separately, by D6, whose own argument is untouched.

~~**The gate runs under a different cost model from production and the configuration must be written
down.** GraphEdX charges **zero for node operations**, so the gate is
`edit_cost_constant=[0, 0, 0, 1, 1, 0]` — **not** the D6 production model `[1, 1, 0, 1, 1, 0]`.
Running the gate under the production model produces a guaranteed mismatch that looks exactly like
a solver bug, at the worst possible point on the critical path.~~

### Gate 1 — bracket validity

`LB ≤ exact ≤ UB` on every calibration pair. Our own implementation gave **0 violations in 400
pairs**; GEDLIB must match. A single violation is a cost-model mismatch.

### Gate 2 — cross-implementation agreement · **EXECUTABLE, T-25 CLOSED 2026-08-12**

The missing implementation was **written rather than retired** (S-e option A, not the recommended
C+B). It is tracked in the repository, not in a scratchpad:

| Artifact | Path |
|---|---|
| The bounds | `benchmarks/real_data/eval_setup/ged_bounds.py` |
| The gate runner | `benchmarks/real_data/eval_setup/validate_ged_bounds.py` |
| Invariant tests | `tests/unit/test_ged_bounds.py` — **35 tests, all passing** |

`branch_lower_bound` is the BRANCH lower bound (Blumenthal & Gamper 2018): branch costs, incident
edges halved, minimised by `scipy.optimize.linear_sum_assignment`. `bipartite_upper_bound` is the
Riesen–Bunke assignment, and **returns the exact cost of the induced node mapping, not the assignment
objective** — that recomputation is what makes it a *proven* upper bound; the objective itself
double-counts edges and is not achievable.

**Reproduces the GEDLIB Picasso smoke test exactly**: P₄ vs C₄ → LB 1.00 / exact 1.00 / UB 1.00.

#### Gate 2 result — 400 LINUX pairs, unit costs, seed 42

```
GATE PASSED: 0 bracket violations on 400 pairs
```

| Quantity | **Measured 2026-08-12** | Retired H4 claim |
|---|---:|---:|
| bracket violations `LB ≤ exact ≤ UB` | **0 / 400** | 0 / 400 ✓ |
| ρ(exact, LB) | **0.859** | 0.966 |
| ρ(exact, UB) | **0.522** | 0.840 |
| mean relative bias, LB | **−26.3 %** | −11 % |
| mean relative bias, UB | **+135.2 %** | +78 % |
| certification rate `LB = UB` | **1.5 %** | 9.8–11.3 % |

> ### The decision survives; its numbers do not
>
> **"BRANCH-FAST is the primary large-`n` reference" is confirmed** and now has a reproducible
> artifact behind it: the lower bound tracks exact GED far better than the upper bound (**ρ 0.859 vs
> 0.522**) and is far tighter (**−26 % vs +135 %**). That is the same conclusion, on the same side,
> by a wide margin.
>
> **But not one of the retired numbers reproduces, and all six miss in the same direction** — the
> retired figures are uniformly more flattering. The most likely explanation is that H4 was measured
> on IAM Letter (n̄ ≈ 4.1, density 0.54) and quoted as if it characterised the cohort, while this
> sample is LINUX (n̄ = 8.71, density 0.255). Larger and sparser is harder for both bounds. **That is
> `gap-audit.md` MF1's defect class once more: a statistic measured on one population and printed
> under another's header.**
>
> **Do not quote 0.966 / 0.840 / −11 % / +78 % / 9.8–11.3 % anywhere.** Re-derive them **per dataset**
> in T-05's calibration ladder, which already stratifies by `n`, and print each with the population
> it was measured on. The certification rate in particular is a *reported* quantity
> ([approx_ged](approx_ged.md) §4) and 1.5 % versus 9.8–11.3 % is the difference between "GED is
> exact for free on a tenth of pairs" and "essentially never".

#### The finding the gate produced — the upper bound is not symmetric

`bipartite_upper_bound(g1, g2) ≠ bipartite_upper_bound(g2, g1)` — measured 12 vs 14 and 5 vs 7 on
small connected pairs. The star costs driving the assignment are asymmetric in the two graphs' roles.
Both values are valid upper bounds, but **a pairwise matrix filled in one orientation is not a
distance matrix**, and Levenshtein would be correlated against an asymmetric reference.

**The exposed bound therefore takes the minimum of both orientations**, which is symmetric, still
provably an upper bound (each orientation is an achievable edit path), and never worse. Measured
gain: ~~tighter on **33.2 %** of pairs, mean **1.15 edit operations**~~ → **see the correction
below**; ρ(exact, UB) from 0.479 → 0.522.

> ## ⚠ CORRECTED 2026-08-15 (T-05) — **33.2 % is not a cohort-level rate, and no single number is.**
> The decision to symmetrise survives untouched and is now better supported.
>
> The 33.2 % / 1.15-operation figure comes from **our own BP implementation on 400 LINUX pairs at
> n̄ = 8.71**. T-05 measured orientation asymmetry across the frozen `IPFP_MS` subsample, which spans
> `n = 2…98` over all ten Suite-2 datasets:
>
> | bin of `max(n₁,n₂)` | `[2,4)` | `[8,10)` | `[12,15)` | `[20,25)` | `[40,50)` | `[60,80)` |
> |---|---:|---:|---:|---:|---:|---:|
> | asymmetric | **0.0 %** | **3.7 %** | 12.0 % | 32.4 % | 52.8 % | **59.5 %** |
>
> **In the bin containing n̄ = 8.71 the rate is 3.7 %; 33.2 % is not reached until `[20,25)`.** So the
> figure does not merely lack precision — **it names a rate belonging to graphs roughly three times
> larger than the ones it was measured on.**
>
> **And the two upper bounds move in opposite directions in `n`**, which is why one number was never
> going to work: `BIPARTITE`'s asymmetry rate **falls** from 22.8 % (LINUX, n̄ 8.71) to 11.2 %
> (Mutagenicity, n̄ 28.5), while `IPFP_MS`'s **rises** from 3.7 % to 59.5 % over the same range.
> **Any restatement must name the method and the size range; neither alone is a fact about "the
> upper bound".**
>
> **What survives — all of it, and more firmly.** Symmetrisation is doing real work: it improves the
> bound on **28.1 %** of subsample pairs. And there is **no systematic orientation bias** (reverse
> tighter on 14.2 %, forward on 13.9 %), which is the check that distinguishes a needed `min` from a
> mis-ordered input. The instruction to T-05 below was followed: every upper bound is computed in
> both orientations and minimised, and `BRANCH_FAST`'s symmetry was **measured** on 9,406 pairs
> across five datasets and two size strata — identically equal, not equal within tolerance.
>
> **Retire the 400-pair BP figure from any cohort-level claim.** Owner of the rewrite: **T-20**.

> **This applies to GEDLIB and it is T-05's problem, not just ours.** `BIPARTITE`, `IPFP`, `REFINE`
> and `BP_BEAM` all construct an edit path from a directed assignment and have the same property.
> **T-05 must either fill both triangles and symmetrise with `min`, or assert symmetry and fail
> loudly.** The lower bound needs no such treatment — its cost matrix depends only on
> `|deg(u) − deg(v)|` and its assignment optimum is invariant under transposition (verified by test).

#### What still runs on Picasso

Gate 2 is now a **two-sided** check: the numbers above are our side. **T-05 replays the same seeded
sample through GEDLIB and compares.** `validate_ged_bounds.py --out` writes the per-pair records for
exactly that purpose. Disagreement is a bug in one of the two implementations and we need to know
which before either is trusted.

### Gate 3 — exact-solver agreement

`ANCHOR_AWARE_GED` and `networkx` A* must agree exactly on a shared sample under the same cost model.

---

## 5. Job configuration

- cost model: `[1, 1, 0, 1, 1, 0]` (D6)
- **GED timeout unchanged from the submission** (author decision). Record it explicitly and report
  the censoring rate **per stratum** — censoring is symmetry-correlated, never pool it
- non-computable pairs are **interval-censored `[LB, UB]`**, not dropped (D11)
- checkpoint every 5,000 pairs (`ged_computer.py` already does)
- one `cpu` job, 64–128 cores, `1-00:00:00`, 128 GB, written with the **`picasso-sbatch`** skill

### 5.1 Output footprint — T-03 is not what threatens the quota

**Measured from `ged_computer.py`, 2026-08-12.** `save_ged_matrix` writes **one
`np.savez_compressed` per dataset**, and `_save_checkpoint` writes **a single `.npz` that is
overwritten in place** — it does not accumulate files.

| | Files | Raw bytes |
|---|---:|---:|
| Suite 1 finals (5 datasets, one N×N matrix each) | 5 | 60 MB |
| Suite 1 checkpoints (one per dataset, overwritten) | 5 | — |
| Suite 2 finals (10 datasets, LB + UB in one `.npz` each) | 10 | 1,222 MB |
| Suite 2 checkpoints | 10 | — |
| **Total** | **30** | **1.25 GB raw, ~130–260 MB compressed** |

**30 files is 0.0075 % of the 400k hard limit.** T-23's stated rationale — "T-03 checkpoints
frequently and fails partway if it hits the limit" — **is false**, and T-03 does **not** need the
quota cleared before it can write its output.

> **What actually consumes the quota is the GEDLIB build tree: 50,000–90,000 files, 12–22 % of the
> hard limit.** That is a one-time *install* artifact, not a *run* artifact. The fix is not to delete
> another project's data — it is to prune the build tree once `build_ext` has produced the shared
> objects, keeping the `gklearn/` package and its `.so` files and discarding
> `include/gedlib-master/` and `ext/`. See [gedlib](gedlib.md) §2.
>
> **Verify before assuming**, since the pruning has not been executed:
> ```bash
> quota -s
> find $BUILD/graphkit-learn -type f | wc -l          # before
> find $BUILD/graphkit-learn -name '*.so*' | wc -l    # what must survive
> ```

If headroom is genuinely tight, two further tightenings are available and neither has been needed so
far: store the upper triangle as `uint16` rather than a full `float64` matrix (**8× smaller**, and
GED values are small non-negative integers), and write to node-local `$TMPDIR` during the run,
copying one file out at the end.

### 5.2 No format conversion is needed

The `.pt` files under `GED_PRECOMPUTED/{AIDS,LINUX}` are **GraphEdX's input distribution**, read by
`graphedx_loader.py` and nothing else. Our computed GED is `.npz` with key `ged_matrix`, which is
already what `eval_correlation.py`, `method_comparator.py`, `dataset_filter.py` and `validator.py`
consume. **Bring the `.npz` files home and use them directly** — there is no conversion step, and
writing our recomputed values into GraphEdX's `.pt` layout would misrepresent their provenance.

---

## 6. Expected consequence

**LINUX ρ = 0.433 and AIDS ρ = 0.349 will both change.** The pair sets grow 2.3× and 2.25× and the
cost model changes. Every downstream number must be re-derived, and published GraphEdX values will no
longer match ours — expected, and stated in the text.

> Do not restate the AIDS gain as 1.62×. `295,296 / 131,148 = 2.25×` is the population-matched ratio;
> 181,909 is a within-split count on the **raw** 911-graph set and is not the comparator. See
> [data](data.md) §5.

> **131,148 is now confirmed by measurement.** T-03's overlap between our all-pairs matrix and
> GraphEdX's published within-split entries is **exactly 131,148 pairs (44.4 %)** on the filtered
> 769-graph set. That closes [data](data.md) §5's instruction to "record 131,148's provenance when
> T-03 reproduces the run", and confirms **2.25×** rather than 1.62× as the population-matched gain.

---

## 7. RESULT — T-03 closed 2026-08-13

Twelve SLURM jobs, all `COMPLETED`. **≈ 2,081 core-hours, ≈ 6.5 h wall, zero requeues.**

| Dataset | graphs | pairs | certified exact | censored | core-h |
|---|---:|---:|---:|---:|---:|
| IAM Letter LOW | 1,180 | 695,610 | 695,610 (100 %) | 0 | 1.2 |
| IAM Letter MED | 1,253 | 784,378 | 784,378 (100 %) | 0 | 1.5 |
| IAM Letter HIGH | 2,059 | 2,118,711 | 2,118,711 (100 %) | 0 | 11.5 |
| LINUX | 89 | 3,916 | 3,870 (98.83 %) | 46 | 5.8 |
| AIDS | 769 | 295,296 | 234,258 (79.33 %) | 61,038 | 2,060.6 |
| **Total** | **5,350** | **3,897,911** | **3,836,827 (98.43 %)** | **61,084 (1.57 %)** | **≈ 2,081** |

Every matrix symmetric, diagonal zero, gate 4 passed. **The pair total reproduces §2's cohort
exactly.** AIDS is 99 % of the cost, as predicted; the total is 26 % above the 1,650 core-h upper
estimate because the `sr` nodes are ~2× slower per core than the machine §2's per-pair figures came
from.

**Both stages ran, and they agree.** Stage 2 seeded from stage 1 and the merge asserts no
conflicting value on any pair index present in more than one shard. It passed — so the pre-declared
stratified sample and the census agree exactly on their 22,051-pair overlap. Under the §3
supersession rule the **census is the reported analysis**, since it landed well before the T-20
freeze; stage 1 is retained as a methodological consistency check and both ρ values are printed.

**Artifacts**: `GED_PRECOMPUTED/extended_merged_exact_ged/` (computed + reference + `PROVENANCE.md`),
mirrored at `results/exact_ged/` and in `execs/isalgraph/exact_ged` on Picasso.

### Three findings the run produced, all of which change something

1. **The exact solver changed.** `ANCHOR_AWARE_GED` is **non-deterministic and not exact** — 14/15
   real AIDS pairs gave different answers across fresh environments, 4/18 wrong against brute force,
   and it reports `LB == UB` on wrong values. **Decision 11's exact half is void**; exact GED comes
   from `networkx` A* run to completion. GEDLIB keeps the `BRANCH_FAST` / `IPFP` bound roles.
2. **GraphEdX uses unit node costs, not zero** — see the correction in [gedlib](gedlib.md) §6 and
   [statistics](statistics.md) D6. This retracts T-03's own earlier "their reference is approximate"
   finding.
3. **The censoring rate is hardware-dependent** — §5's `[LB, UB]` interval-censoring is doing real
   work, and its rate is a property of *(cohort, timeout, machine)*. See the article note below.

### For the manuscript

`.claude/notes/review/tasks/T-03-article-notes.md` collects what belongs in the paper: the censoring
protocol, the timeout's status as a reported parameter, the `nx.graph_edit_distance` timeout defect
in the submitted pipeline, and the two provably non-optimal GraphEdX entries.
