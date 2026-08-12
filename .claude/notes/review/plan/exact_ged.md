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

> **The gate runs under a different cost model from production and the configuration must be written
> down.** GraphEdX charges **zero for node operations**, so the gate is
> `edit_cost_constant=[0, 0, 0, 1, 1, 0]` — **not** the D6 production model `[1, 1, 0, 1, 1, 0]`.
> Running the gate under the production model produces a guaranteed mismatch that looks exactly like
> a solver bug, at the worst possible point on the critical path.
>
> Note what the gate does *not* establish: agreement under GraphEdX's pseudometric model does not
> validate our metric model. **It validates the solver.** The cost-model change is justified
> separately, by D6.

### Gate 1 — bracket validity

`LB ≤ exact ≤ UB` on every calibration pair. Our own implementation gave **0 violations in 400
pairs**; GEDLIB must match. A single violation is a cost-model mismatch.

### Gate 2 — cross-implementation agreement · **NOT EXECUTABLE**

> Specified as: GEDLIB's `BRANCH_FAST` and `BIPARTITE` must reproduce `scratchpad/ged_bounds.py` on
> the same 300–400 pairs. **`ged_bounds.py` does not exist and never did.**
>
> **Collateral**: ρ(exact, LB) = 0.966 vs ρ(exact, UB) = 0.840 and the +78 % / −11 % biases — the
> evidence for "BRANCH-FAST is the primary large-`n` reference" — are **unreproducible from any
> surviving artifact**. The decision may well be right; it is currently unsupported.
>
> **Owner T-25, decision S-e.** Recommended: spot-check 20 pairs against `networkx` under the unit
> model (~1 h), then **retire gate 2 on the record**. Gate 1 already catches the failure mode that
> matters most — a bracket violation *is* a cost-model mismatch. Do not quietly drop it; strike it
> with the reason recorded.

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
- **T-23 must clear the fscratch file-count quota first** — T-03 checkpoints frequently and fails
  partway if it hits the limit

---

## 6. Expected consequence

**LINUX ρ = 0.433 and AIDS ρ = 0.349 will both change.** The pair sets grow 2.3× and 2.25× and the
cost model changes. Every downstream number must be re-derived, and published GraphEdX values will no
longer match ours — expected, and stated in the text.

> Do not restate the AIDS gain as 1.62×. `295,296 / 131,148 = 2.25×` is the population-matched ratio;
> 181,909 is a within-split count on the **raw** 911-graph set and is not the comparator. See
> [data](data.md) §5.
