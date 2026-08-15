# T-05 — article notes

**Bounded GED over Suite 2.** Closed 2026-08-15. Ordered by consequence: items that change what the
paper may claim first, reporting obligations after, and a **not claimable** section at the end that
prevents more damage than the rest of the file.

Artifacts: `$SANDISK/data/source/APPROX_GED/` (48 files, 1.2 GB, `manifest.json` + `PROVENANCE.md`) ·
`results/reports/T-05-bounded-ged/REPORT.md` and its `data/*.json` ·
[design + 15 amendments](T-05-design.md) · [plan RESULT](../plan/approx_ged.md).

**Reproduction parameters that must travel with every number below.** Solver **GEDLIB** via
`gklearn.gedlib` (jajupmochi fork, built 2026-08-11). Cost model **D6**: `CONSTANT` edit cost,
`[node_ins, node_del, node_rel, edge_ins, edge_del, edge_rel] = [1, 1, 0, 1, 1, 0]` — node and edge
insertion/deletion cost 1, substitutions free, one model for every dataset. Options strings are part
of the method names and are listed in the plan RESULT. Hardware **AMD EPYC 7H12** (`sr`), single
process per task. Exact-GED ladder: `networkx.graph_edit_distance`, **1,200 s per-pair budget**.
Bootstrap: graph-level cluster resampling, **D15 frozen tiers**, percentile CI, **seed 42**.

---

## 1. ⚠ The size-scaling result, and the metric correction under it — **T-20, results §, answers AE.1**

**Measured, all 21,710,892 pairs.** Fitting bracket width on `max(n₁, n₂)` **within each dataset**:

| | absolute gap `UB − LB` | relative width `(UB − LB)/UB` |
|---|---|---|
| slope sign | **positive in 10 of 10 datasets** | negative in 6 of 10 |
| `coil_del` (n 3–77) | **+1.52** ops/node | −0.0031 |
| `protein` (n 3–96) | **+1.46** | −0.0030 |
| `mutagenicity` (n 5–98) | **+0.52** | −0.0054 |
| `aids_iam` (n 2–85) | **+0.25** | −0.0127 |

All ten absolute CIs exclude zero. **The bracket gets absolutely looser as graphs grow**; `UB` simply
grows faster than the gap, which is why the ratio can fall over the same range.

> **The correction, and it belongs in the paper's method text, not only here.** `(UB − LB)/UB` is a
> **ratio whose denominator grows with `n`**. A bound whose absolute gap is merely *constant* already
> yields a falling relative width. **In 4 of 10 datasets the two measures carry opposite signs**, so
> reporting the relative width alone would have told a reviewer the bracket tightens at scale — the
> opposite of the truth, on exactly the point AE.1 disputes. Report the **absolute gap first**, the
> relative width beside it with its denominator named. `LB/UB` is *not* independent evidence: it is
> exactly `1 − (UB−LB)/UB`.

This **confirms** T-27 §5.4's prediction that the upper bound degrades fastest in `n`. Nothing was
ever in tension with it.

## 2. How much of the widening is our reference bound's fault — **T-20, results §; T-14, AE.1 answer**

The disclosed `BP_BEAM_DET` arm ran on **all** of Suite 2 so this is separable rather than asserted.
The primary arm's absolute slope exceeds the arm's in **10/10 datasets**; the ratio by slope role:

| slope role | mean `n` | ratio range | **gate's share of the widening** |
|---|---|---|---|
| small-`n` constraint only (Letter ×3, LINUX) | 4.07–8.71 | 2.70×–8.20× | **63–88 %** |
| intermediate (`aids_graphedx`, `grec`) | 11.0–11.5 | 2.18×–2.28× | 54–56 % |
| **unconfounded** (`aids_iam`, `coil_del`, `mutagenicity`, `protein`) | 14.0–31.7 | **1.54×–2.06×** | **35–51 %** |

The small-`n` and unconfounded ranges **do not overlap**. So at the sizes AE.1 disputes, a better
upper bound buys only about a third to a half off the slope: **most of the large-`n` looseness is not
attributable to the frozen gate and would survive replacing `BIPARTITE`.**

> **Not a trend in `n`.** The fall is **not monotone** — `mutagenicity` (n̄ 28.53, ratio 1.87×) sits
> above `coil_del` (n̄ 21.54, 1.66×). The grouped ranges are the claim; an ordered fall is not, and no
> regression is fitted to ten points whose provenance moves with size.

**`BIPARTITE` remains primary by PI ruling; `BP_BEAM_DET` is a disclosed sensitivity arm and was
never substituted after the fact.**

## 3. The exact-GED ceiling moved from 12 to 17 — **T-20, §exp; answers R3.7a's "with its cause"**

Ladder rungs `n = 13…18`, 250 pairs each, pooled across datasets, seed 42, 1,200 s budget. The
pre-declared 25 % truncation rule fired at rung 18:

| rung | 13 | 14 | 15 | 16 | **17** | 18 |
|---|---:|---:|---:|---:|---:|---:|
| certified | 81.2 % | 54.8 % | 48.0 % | 42.0 % | **28.4 %** | **20.8 %** |

**Measured exact-GED ceiling `n = 17`** — five nodes above T-03's 12, and two above the 15–16 the
25-pair pilot projected. This is a *measurement*, not an assertion: the rule was frozen before the
run and is the only thing that selected the rung.

> **Caveats that must travel with any rung number.** (a) The ladder is **six datasets, not ten** —
> Letter and LINUX cap at `n ≤ 10` and contribute at no rung; neither AIDS cohort has a 14-node
> connected graph, so rung 14 draws from four. Composition then shifts across rungs. **Never a bare
> rung-to-rung slope.** (b) Per-rung quantities conditioned on certified pairs sit on an increasingly
> biased subset as `n` grows — at rung 18 that is the 20.8 % A\* could finish in 1,200 s.

## 4. Certification rate per dataset — **T-20, results table.** §4 of the plan forbade promising one

| dataset | certified | mean gap | | dataset | certified | mean gap |
|---|---:|---:|---|---|---:|---:|
| `iam_letter_med` | **28.46 %** | 1.85 | | `grec` | 1.32 % | 14.46 |
| `iam_letter_low` | 28.05 % | 1.82 | | `coil_del` | 0.87 % | 45.95 |
| `iam_letter_high` | 23.61 % | 2.23 | | `aids_iam` | 0.67 % | 13.16 |
| `linux` | 2.02 % | 6.94 | | `aids_graphedx` | 0.41 % | 12.26 |
| `protein` | 0.16 % | 76.60 | | `mutagenicity` | **0.03 %** | 32.25 |

**A factor of 949 across the cohort.** T-27 measured 1.2–40.2 % at `n ≤ 12`; six of ten Suite-2
datasets are under 1 %. **No sentence may state a certification rate without naming its dataset.**

## 5. Validation, for the reproducibility statement — **T-21**

- **G2**, the strong gate: on the four datasets whose Suite-2 cohort is *identical* to Suite 1, the
  new pipeline reproduces T-27's recorded census **element-wise on 10,807,845 pairs across three
  method arms**, every `sha256` byte-identical. One comparison covers loader, cost model, options
  string, symmetrisation and pair ordering.
- **G3**: **0** bracket violations over all 21,710,892 pairs; **0** `lb ≤ exact ≤ ub` violations over
  3,836,827 T-03-certified pairs, joined on `graph_ids` and never positionally.
- **G4**: all 30 matrices symmetric, zero-diagonal, finite, non-negative, off-diagonal zero fraction
  below 0.99.
- Cohort reproduces exactly: **16,370 graphs, 21,710,892 pairs**.
- Every gate was **re-verified by the orchestrator with independent code**, not by re-running the
  gate module — routing the check through the machinery it checks would be circular.

## 6. Reporting obligations inherited by the manuscript

1. **Symmetrisation must name its method and size range.** Orientation asymmetry is a function of
   size and the two upper bounds move in **opposite** directions: `BIPARTITE` falls 22.8 % → 11.2 %
   from n̄ 8.71 to 28.5, while `IPFP_MS` rises 3.7 % → 59.5 %. `decisions.md` §6's **33.2 %** figure
   comes from our own BP on 400 LINUX pairs at n̄ 8.71 and **names a rate belonging to graphs ~3×
   larger than the ones it was measured on**. Retire it from any cohort-level claim. → **T-20**
2. **Class counts are false of the filtered cohort.** Letter LOW retains **9 of 15**, GREC **17 of
   22**; LINUX and AIDS-GraphEdX carry **no class label at all**. → **T-18, T-06**
3. **The D11 censored-interval upper ends are heuristic and run-dependent.** T-03's `ub_matrix` left
   `IPFP` on `--randomness REAL`, so 74–82 % of values change between runs. Exposure is bounded and
   verified — `ub_matrix == ged_matrix` on all 234,258 certified AIDS and 3,870 certified LINUX pairs
   — so it is **exactly the 61,084 interval upper ends**. Lower ends unaffected (5/5 identical).
   Accepted without repair by PI decision; **stating it is therefore obligatory**, or a reader
   re-running the script gets different numbers, which is the R3.5a failure mode. → **T-20**
4. **Size and provenance are confounded and no sampling design removes it** — it is a property of
   which real datasets contain large connected graphs. The size-scaling curve is reported
   **within-dataset primary**; any pooled curve is descriptive and carries the confound in its
   caption. → **T-20**

---

## 7. ⚠ NOT CLAIMABLE — read before quoting anything above

- **Not "the bracket tightens at large `n`".** The relative width falls; the absolute gap rises in
  all ten datasets. See item 1.
- **Not a per-pair cost of any method from `seconds_matrix`.** It records *in-worker solver time*,
  and the datasets ran at different worker counts under a parallelisation later measured to be
  **negative-scaling** (1 worker 36 core-s, 4 → 212, 15 → 928, 32 → 5,260 on identical work). The
  values are **not comparable across datasets** and under-report job consumption.
- **Amendment 7's "every method costs ×13–14 more than T-27's gate implied" is WITHDRAWN** — it
  compared T-27's single-process probe against pool-inflated production numbers. T-27 limitation 3
  may still be directionally right; **this ticket has not measured it** and must not be cited as
  having done so.
- **No collision or bracket figure without its enumeration window.** Same discipline as the
  directedness collision rate.
- **Not a canonicalisation censoring rate at D14's 300 s.** T-05 measured at a **15 s** budget on 10
  random graphs per dataset (`protein` 5/10 killed, `coil_del` 5/10, `mutagenicity` 1/10, `grec`/AIDS
  0/10). That establishes censoring is a **bulk property of two or three datasets rather than a
  marginal tail**; it does **not** license a 300 s number. Incidence at 300 s is unmeasured. → T-06
- **Nothing in T-05 measures IsalGraph.** The bracket is a property of GEDLIB's
  `BRANCH_FAST`/`BIPARTITE`/`BP_BEAM_DET` under D6. It bounds the *resolution* any later correlation
  against GED can have at a given `n`; it is not evidence for or against the encoding. **§7.5 — the
  deliverable that would connect the two — is deferred in full to T-06.**
- **`UB_TIGHT/` is a size-stratified sample, not a random sample of Suite-2 pairs.** It deliberately
  over-weights large `n`. Every figure from it is reported **per bin**, never pooled into a
  cohort-level mean.
- **The ceiling `n = 17` is not a ceiling for IsalGraph.** It is where exact GED becomes infeasible
  under a 1,200 s budget. IsalGraph encodes to `n = 98` with no timeout.
