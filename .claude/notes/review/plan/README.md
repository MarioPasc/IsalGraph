# Revision plan — PR-D-26-03293 (IsalGraph)

**Manuscript**: *Representation of Graphs by Sequences of Instructions*, Pattern Recognition
(Elsevier). **Major revision, due 2026-08-31.**
**Status**: v1.0, 2026-08-12. Refactored from the single-file plan into one file per edge of the
proposal.

---

## The plan in six sentences

Recompute every GED ourselves under **one unit cost model** via **GEDLIB**, exact below `n = 12` and
as a **proven bracket** above it, over a cohort extended from 5 to **10 IAM datasets** reaching
**n = 98**. Put **six competitor representations** beside IsalGraph in three experiments, with each
one's distance chosen by **measurement** rather than assertion. Replace the statistical protocol
wholesale: **graph-level** bootstrap and Mantel instead of pair-level asymptotics, per-dataset
results primary, a pre-declared confirmatory family. **Decline** the sequential-model experiment and
bring the language-model claims down with it. Fix the formal statements, the pseudocode, the claim
scoping and twelve self-found defects. Deliver it inside **35 pages**, which is the constraint that
actually binds.

---

## The files

Each is atomic and cross-references the others. **Start with [tickets](tickets.md)** — it names, per
ticket, exactly which of these to read.

| File | The edge it owns |
|---|---|
| **[demands](demands.md)** | **The coverage contract.** Every demand in `mail.txt` → decision → ticket → artifact. Also the response letter's index |
| **[decisions](decisions.md)** | What is signed (1–21), what is still open (S-d…S-h) with owners and dates, and what is closed by measurement |
| **[tickets](tickets.md)** | The board, in brief, with per-ticket reading lists and the dependency structure |
| **[schedule](schedule.md)** | Calendar with gates, the board arithmetic, the risk register, and the cut order in **days and pages** |
| **[data](data.md)** | Cohort, filter, measured counts, what was dropped and why, encoding cost, and the four integrity defects that reach a printed number |
| **[gedlib](gedlib.md)** | The GED engine: install on Picasso, API, method capability matrix, the **two traps that fail silently**, and the cost model |
| **[exact_ged](exact_ged.md)** | Suite 1 — `ANCHOR_AWARE_GED`, the **two-stage T-03**, and the four validation gates (one of which is unexecutable) |
| **[approx_ged](approx_ged.md)** | Suite 2 — the `BRANCH_FAST ≤ GED ≤ IPFP` bracket, the calibration ladder, and the **no-interpolation** reporting rule |
| **[competitors](competitors.md)** | The six representations, the backend architecture, and **T-04a**, which selects each distance by measurement |
| **[statistics](statistics.md)** | The locked protocol D1–D15, stratification, and the confirmatory family that must be frozen before T-06 |
| **[preregistration](preregistration.md)** | **The frozen confirmatory family** — `N_max = 197` in three fixed-sequence families, the reduction rule, and what is deliberately excluded. Authoritative over `statistics.md` §9 |
| **[labels](labels.md)** | R1.3 / AE.4b, tiered, with the PI decision that is still open |
| **[corrections](corrections.md)** | Claim scoping B1–B6, the formal-statement audit, the twelve manuscript defects, and the [28]/[29] delta |
| **[manuscript](manuscript.md)** | Section rewrite map, artifact inventory, the **page budget**, the response-letter architecture |
| **[compliance](compliance.md)** | What the Editor-in-Chief checks **pass/fail**: bibliography count, recency, venue, arXiv, citation groups, 35 pages, submission package |

---

## The five things most likely to go wrong

1. **The schedule does not fit.** 93.5 days upper / 54.8 lower, on a critical path of 27.5–28.0 days
   serial, in a 19-day window. **S-f is open and is the most valuable decision on the board.**
2. **Pages, not days, are the binding constraint.** `main.pdf` is exactly 35 of 35 and the revision
   adds ≈ 12–13 gross against ≈ 4.75 recoverable. An extension returns days; nothing returns pages.
   The day-1 supplementary query to patcog@elsevier.com is free insurance on ~8 of them.
3. ~~**Validation gate 2 cannot be run.**~~ **Closed 2026-08-12** — `ged_bounds.py` written, tracked
   and passing (0 bracket violations / 400 LINUX pairs). The decision it supported survives; **its
   published numbers do not** and must be re-derived per dataset. It also produced a live finding:
   **GEDLIB's upper bounds are direction-dependent**, so the production matrix must be symmetrised.
4. **Two GEDLIB accessors return garbage rather than raising.** An upper-bound method returns
   `get_lower_bound() = 0.00`. A whole GED matrix can fill silently with zeros. **Assert
   `0 < value < inf` on every read.**
5. **The delta table can be turned against us.** It documents that both predecessors ran a sequence
   model and this paper does not. **R3.1a(ii) — the sufficiency paragraph — is what stops it, and it
   was unowned until 2026-08-12.**

---

## Provenance

Built from `../source/mail.txt` and the manuscript sources, audited three times:

| Pass | What it did | Where |
|---|---|---|
| Coverage audit | produced the traceability matrix and tickets T-17…T-24 | `../source/gap-audit.md` |
| Over-scope and integrity audit | rebuilt the demand inventory from `mail.txt` alone; 41 demands, 1 coverage hole, 2 over-scope items, 24 integrity defects | `../../audit-2026-08-11b/` |
| Third-auditor pass | audited that audit; corrected the board total, downgraded one misdirected finding, and staged T-03 | `../../audit-2026-08-11b/third-auditor.md` |

**Raw inputs and audit history stay in [`../source/`](../source/).** This folder is the plan; that
folder is what the plan was derived from.

Numbers here are **measured, not quoted**. Where a measurement is unreproducible because its script
is gone, it says so.
