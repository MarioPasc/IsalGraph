# T-02 — Statistics lock: design and decision record

**Status**: **CLOSED 2026-08-13.** Deliverable: [preregistration](../plan/preregistration.md).
**Serves**: R3.5a, R3.5c, AE.4c · **Signed**: [decisions](../plan/decisions.md) 23, 24, 25, 26

---

## 1. What the ticket found

D1–D15 were already locked and needed no re-litigation. The ticket's job was the one item
[statistics](../plan/statistics.md) §9 deferred — *"write the explicit list … with its cardinality,
and freeze it before T-06 runs"* — and in enumerating it, four defects surfaced in the locked
protocol itself.

| # | Defect | Where | Fix |
|---|---|---|---|
| **1** | §9's family table put a Friedman omnibus on **both** regimes; §4 locks the omnibus to the **approximate regime only**, because at `N = 5` the critical difference separates almost nothing | `statistics.md` §9 vs §4 | §4 wins. F2 carries **one** omnibus per claim |
| **2** | The calibration gate and D13 sat **inside** the family they gate. A test whose outcome removes rows from a family cannot be a member of it — the cardinality becomes a function of a result | `statistics.md` §9 | split into **F0** and **F1**, prior and separate: fixed-sequence gatekeeping |
| **3** | The labels row (`L`) was conditional on **S-d, open until 2026-08-18**, making the cardinality indeterminate on the day it was supposed to be frozen | `statistics.md` §9 | excluded. If Tier 2 is promoted it forms its **own** pre-declared family |
| **4** | D15's subsample validation drew **2 × 10⁶ pairs from Letter HIGH's 2,118,711** — a 94.4 % sample — to validate a procedure that runs at **7.72 %** in production | `statistics.md` §5 rule 2 | validate at the production **ratio**, plus a structure-matched arm on Mutagenicity itself |

Defect 4 is the one a reviewer would find: the arithmetic is one division away, and "we validated the
subsample" would have been a sentence supported by a comparison that could not fail.

---

## 2. The decisions taken, and what each rejected

### D13 promoted to confirmatory (decision 25)

**Rejected**: leaving D13 as a reporting rule, with `ρ(Lev, UB)` given its own 70 per-dataset rows in
the primary family.

Two independent arguments against those 70 rows, and they point the same way:

1. **Statistical.** `ρ(Lev, LB)` and `ρ(Lev, UB)` are computed on the *same pairs* from two bounds on
   the *same quantity*. They are near-duplicates by construction. BH-FDR assumes independence or PRDS
   and degrades on near-duplicates: the 70 tests inflate `N` without adding evidence, making every
   genuinely independent test in the family harder to detect.
2. **Scientific.** The claim worth registering is not "ρ against the upper bound is also large" — it
   is **"the conclusion does not depend on where inside the proven bracket the truth lies."** That is
   exactly D13's difference test, and it is one test per dataset rather than seven.

**The upper bound is not demoted.** [approx_ged](../plan/approx_ged.md) §4's reporting obligations are
untouched: both ρ printed per dataset, no interpolation, bracket width per stratum, certification
rate, symmetrisation. What changed is that its confirmatory weight is carried by the invariance claim
instead of by duplicate correlations — which is the stronger version of the bracket argument, not a
weaker one.

### BH over `N_actual` with an `N_max` sensitivity column (decision 24)

**Rejected**: fixing the denominator at `N_max` regardless of what T-04a excludes.

The reduction is defensible because [competitors](../plan/competitors.md) §3.4's selection rule is
**F5-blind** — ties break on cost (F6), never on correlation with GED — so removing a representation
is independent of the hypotheses in F2. The sensitivity column costs nothing (it is a re-threshold of
stored p-values) and removes the objection entirely, so there is no reason not to print it.

`N_actual(F2) = 182 − 15k − 8d`. **15, not 25**, because F1–F4 are properties of a *distance*: a
representation with no admissible distance loses its Claim B rows and keeps its Claim A rows, since a
bit count needs no distance.

### Both bracket ends re-selected by measurement (decision 26 → T-27)

**Not a T-02 deliverable — a ticket.** Raised here because freezing the family forced the question
"primary against *which* reference?", and the answer turned out to be unsupported at both ends:
`IPFP` has **never been measured against exact GED**, and `BRANCH_FAST` rests on **400 LINUX pairs at
n̄ = 8.71** licensing a regime to `n = 98`. T-03's 3.9 M certified exact values make the full
proven-method grid cost ≈ 5 core-hours. Spec: `T-27-spec.md`.

---

## 3. Measured / inherited / predicted

| Number | Provenance |
|---|---|
| `N_max = 197`; F0 5, F1 10, F2 182 | **derived by this ticket** from the comparator sets and the locked cohort. Arithmetic in [preregistration](../plan/preregistration.md) §4.2 |
| D15 tier assignment | **derived** from [data](../plan/data.md) §1's pair counts — which are **verified for Suite 1, unverified for Suite 2** (see T-01). Tiers may move if T-01's re-derivation moves the counts |
| Subsample ratios 7.72 % / 24.51 % | **computed** from the same pair counts, same caveat |
| Comparator sets: 6 (A) / 7 (B) | **inherited** from [competitors](../plan/competitors.md) §2 and §6. Subject to `k` |
| ρ(exact, LB) = 0.859 vs ρ(exact, UB) = 0.522 | **measured, on 400 LINUX pairs only** — see decision 26. Quote only with its population |
| "IPFP should do much better than BIPARTITE" | **predicted, never checked.** T-27 |

---

## 4. Debt

| Item | Owner | Due |
|---|---|---|
| `k` — representations excluded from Claim B | **T-04a** | before T-06 |
| `d` — Suite-2 datasets with an uninformative bracket | **T-06**, via F1 | during T-06 |
| The primary bound at each end | **T-27** | before T-05 |
| Whether the D15 tiers survive T-01's re-derivation | **T-01** | before T-06 |
| Whether labels Tier 2 becomes its own family | **S-d**, PI | 2026-08-18 |
