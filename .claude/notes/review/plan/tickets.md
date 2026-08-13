# Ticket board

**A brief index, not a specification.** Each row names the ticket in one line and lists the files an
agent must read before starting it. The *content* lives in those files — do not duplicate it here.

**Board: 92.1 days upper / 53.9 lower. Critical path: 27.0 days serial against 19.**
See [schedule](schedule.md). **T-25 closed and T-23 rescoped 2026-08-12. T-03 CLOSED 2026-08-13 —
the long pole is off the critical path. T-02 CLOSED 2026-08-13. T-27 opened.**

> ⚠ **T-01 changed the cohort size. The pair count is 21,710,892, not 40,024,242.**
> COIL-DEL contributes **3,900** graphs — the IAM split index's own definition, 100 classes × 39 —
> not the 7,200 `.gxl` files that ship beside it, of which 3,300 carry no class label
> (decision 27). Suite 2 is **16,370 graphs / 21,710,892 pairs**, so the extension is **3.1× graphs
> and 5.6× pairs**, not 3.7× and 10.3×. **`n_max = 98`, `n̄ = 31.68` and the density span 0.094–0.607
> are unchanged, so AE.1's evidence is untouched.** Nine of ten rows and all three discard ratios
> reproduced exactly, and Suite 1 reproduces `export_graphs.py` to the pair.
> Inherits: **T-05, T-06, T-13, T-18, T-20**. Record: [data](data.md) §1.3 and §7.
>
> Two further findings: **the size-biased discard is cohort-wide** — Letter discards at 1.23–1.32×
> and is 84 % of Suite 1 — and **LINUX carries no node or edge attribute at all**, which settles E6
> by measurement and hands T-18 its Tier-0 label column.

> ⚠ **T-02 found that neither bracket end was selected by measurement.** `IPFP` has **never been
> measured against exact GED** — [approx_ged](approx_ged.md) §2 says so in its own words —
> and `BRANCH_FAST` rests on **400 LINUX pairs at n̄ = 8.71**, licensing a regime to `n = 98`.
> **T-27** runs the full proven-method grid against T-03's 3,897,911 certified exact values for
> ≈ 5 core-hours. **T-27 gates T-05.** Until it closes, both are *defaults*, not selections, and
> ρ(exact, LB) = 0.859 / ρ(exact, UB) = 0.522 may be quoted **only with "on 400 LINUX pairs"**
> attached. Inherits: **T-05, T-06, T-20**.

> ⚠ **T-03 invalidated a premise that T-05, T-06 and T-22 all read.** GraphEdX's published GED uses
> **unit node costs**, not the zero node cost asserted in [gedlib](gedlib.md) §6 and
> [statistics](statistics.md) D6. Measured 4/4 unit, 0/4 zero. Anything derived from "the submission
> mixes IAM unit costs with GraphEdX topology-only costs" needs re-checking before it is printed.
> D6's *metric* argument is unaffected. Full record: `.claude/notes/review/tasks/T-03-design.md`
> amendment 4.

**Read for every ticket**: [decisions](decisions.md) (do not re-litigate a signed decision) and
[demands](demands.md) (what the ticket is answering, and to whom).

---

## Board

| ID | Ticket | Depends | Days | Pri | **Read first** |
|---|---|---|---|---|---|
| ~~**T-01**~~ | ~~Data lock — audit tables, cohorts, merge splits, port surviving scripts into `tests/`~~ → **DONE 2026-08-13.** Re-derived, not ported: 15 of 16 scripts were gone and Suite 2 had no loader. `iam_gxl_loader.py` + `cohort_audit.py` + **34 tests**. **Suite 1 reproduces `export_graphs.py` exactly** (3,897,911 pairs). **Suite 2 = 16,370 graphs / 21,710,892 pairs / `n_max` 98** — COIL-DEL corrected 7,200 → **3,900** (decision 27). Nine of ten rows and all three discard ratios reproduced exactly. Four findings: the **size-biased discard is cohort-wide** (Letter 1.23–1.32×); **LINUX is unlabelled**, settling E6; the **density convention** matters (up to 27 %); **I-05 closed** at 1.19× | — | **done** | — | [data](data.md) §1, §7, [T-01 design](../tasks/T-01-design.md) |
| ~~**T-02**~~ | ~~Statistics lock — graph-level bootstrap, Mantel, pair-accounting ladder, and the frozen confirmatory family with its cardinality~~ → **DONE 2026-08-13.** Family enumerated and frozen at **`N_max = 197`** in three fixed-sequence families — F0 calibration gate 5, F1 bracket gate 10, F2 primary 182 — BH-FDR q = 0.05 within each; `N_actual = 182 − 15k − 8d`. **Four defects fixed in the locked protocol**: §9's exact-regime omnibus contradicted §4; two gates sat inside the family they gate; the labels row made the cardinality indeterminate; **D15 validated a 7.72 % subsample by drawing 94.4 % of a smaller dataset**. **D13 promoted to confirmatory**; ρ(Lev, UB) gets no primary rows. Raised **T-27** | T-01 | **done** | — | [preregistration](preregistration.md), [T-02 design](../tasks/T-02-design.md), [statistics](statistics.md) §12 |
| ~~**T-03**~~ | ~~Exact GED on Picasso~~ → **DONE 2026-08-13.** All five Suite-1 datasets: **3,897,911 pairs, 98.43 % certified exact, 1.57 % interval-censored, ≈ 2,081 core-h.** Both stages ran and **agree on their 22,051-pair overlap**. Three findings carried: the **exact solver changed** (`ANCHOR_AWARE_GED` is non-deterministic and non-exact), **GraphEdX uses UNIT node costs, not zero** (retracts a T-03 finding *and* contradicts [gedlib](gedlib.md) §6 / D6), and **censoring is hardware-dependent** | T-01 | **done** | — | [T-03 log](../tasks/../2026-08-12-exact-ged/summary.md), [exact_ged](exact_ged.md) §7 |
| **T-04** | **Competitor backends** — `src/isalgraph/competitors/` in the IsalHG idiom: graph6, sparse6, nauty, AGM, **gSpan min-DFS** | — | 3–8 | **P0** | [competitors](competitors.md) |
| **T-04a** | **Metric feasibility** — every (representation × distance) cell on a fixed 200-graph sample; select each primary distance by the pre-declared rule. **Must close before any production distance matrix** | T-04 | 0.5–1 | **P0** | [competitors](competitors.md) §3 |
| **T-05** | **Bounded GED via GEDLIB** — wire `BRANCH_FAST` + `IPFP`, pass the validation gates, run the **calibration ladder**, then all 40 M Suite-2 pairs | T-01, T-03 | 5–10 | **P0** | [approx_ged](approx_ged.md), [gedlib](gedlib.md), [exact_ged](exact_ged.md) §4 |
| **T-06** | **Full recompute** — all experiments, C++ engine, new cohorts, competitor columns, new statistics | T-02…T-05 | 10–14 | **P0** | [statistics](statistics.md), [data](data.md), [competitors](competitors.md), [labels](labels.md) |
| **T-07** | **Read [28] and [29]**; inherited/modified/new delta table **plus the sufficiency paragraph**; resolve D19 | — | 1–4 | **P0** | [corrections](corrections.md) §4, [decisions](decisions.md) 9 |
| **T-08** | **Related-work section** (§1.x) + bibliography to ≤ 55 | T-07 | 4–10 | P1 | [compliance](compliance.md), [manuscript](manuscript.md) §1 |
| **T-09** | **Explanatory figures** — the canonical search-space schematic (**R3.7c, requested**) and the S2G/G2S worked example (**author preference, first page cut**). Both regenerate the graphical abstract | — | 1.5 | P1 | [manuscript](manuscript.md) §2, [decisions](decisions.md) S-g |
| **T-11** | **Manuscript defects** — Alg. 2, Example 2.3, equivariance→invariance, **and E7's float fix, which must precede any trim** | — | 2 | P1 | [corrections](corrections.md) §3 |
| **T-12** | **Claim scoping** — B1…B6, and E5/E6 | T-06 | 2 | P1 | [corrections](corrections.md) §1 |
| **T-13** | **Complexity section** — `P(M)` recomputation, four costed operations, three-way separation, the `\|Aut(G)\|` worst case | — | 2 | P1 | [corrections](corrections.md) §5, [data](data.md) §4 |
| **T-14** | **Response letter** — assembles the fragments each ticket emits; **not written from scratch at the end** | all | 3 | **P0** | [manuscript](manuscript.md) §4, [demands](demands.md) |
| **T-15** | **Page trim to 35** + supplementary split | all | 2 | **P0** | [manuscript](manuscript.md) §3, [compliance](compliance.md) §7 |
| **T-17** | **AE.3 comparison table** as a paper artifact — properties, strengths, limitations of each, on R1.2's five axes. **Rows populated from T-04's measurements, not asserted** | T-04, T-07 | 2–3 | **P0** — the AE endorsed this in their own voice | [competitors](competitors.md), [demands](demands.md) AE.3/R1.2b |
| **T-18** | **Labels** — tiered; Tier 0 not optional, Tier 1 recommended | T-05, T-06 | 0.3–1 | **P0** (Tier 0) | [labels](labels.md) |
| **T-19** | **Bibliography recency and venue audit** — classify all 43 references; add **≥ 4 from 2025–26**, ≥ 3 at PR venues other than the PR journal, **self-citations excluded** | T-08 | 1–2 | **P0** — EiC checks independently | [compliance](compliance.md) §2–§4 |
| **T-20** | **Manuscript rewrite** — §3.1, §3.2, §3.3, §4, §5, abstract. The largest single writing task | T-06 | 5–7 | **P0** | [manuscript](manuscript.md) §1, [statistics](statistics.md), [data](data.md) |
| **T-21** | **Implementation, reproducibility and artifact release** — C++ engine and GEDLIB in §3.3; versions; the `-march` and non-rsyncing-`.so` traps; data-availability statement | T-06 | 1–2 | P1 | [compliance](compliance.md) §8, [gedlib](gedlib.md) |
| **T-22** | **Formal-statement audit** — restate Thm 2.12 within a fixed directedness class, move the flag hypothesis into the statement, **re-verify all three proof steps**, propagate to **Cor. 2.13** | — | 1–2 | **P0** | [corrections](corrections.md) §2, [statistics](statistics.md) D6 |
| ~~**T-23**~~ | ~~Clear the Picasso `fscratch` file-count quota~~ → **RESCOPED 2026-08-12, no longer blocking.** T-03 + T-05 output is **30 files** (0.0075 % of the hard limit); the pressure is the GEDLIB **build tree** (50–90k files), pruned after `build_ext`. Folded into T-05's environment setup | — | 0.1 | P2 | [gedlib](gedlib.md) §2, [exact_ged](exact_ged.md) §5.1 |
| **T-24** | **Submission package and Elsevier compliance** — source files, AI declaration, biographies, acknowledgements, highlights, graphical abstract (**fix the misspelt filename**), competing-interest and data-availability statements | T-15 | 1 | **P0** | [compliance](compliance.md) §8 |
| ~~**T-25**~~ | ~~Restore validation gate 2, or retire it on the record~~ → **CLOSED 2026-08-12 by option A.** `ged_bounds.py` written and **tracked in the repo**; gate 2 executable and **passing** (0 violations / 400 LINUX pairs); 35 unit tests. Two findings carried to T-05: the upper bound is **not symmetric**, and **the retired H4 numbers do not reproduce** | — | **done** | — | [exact_ged](exact_ged.md) §4 |
| **T-26** | **Bibliography-slot and page-budget reconciliation** — the two arithmetics the EiC checks independently and no other ticket owns end to end. **Runs after T-08 and T-19, before T-15** | T-08, T-19 | 0.5 | **P0 — EiC pass/fail** | [compliance](compliance.md) §2, [manuscript](manuscript.md) §2–§3 |

| **T-27** | **GED bound bake-off** — select **both** bracket ends by measurement, not by default. LB ∈ {BRANCH, BRANCH_FAST, BRANCH_TIGHT, STAR}, UB ∈ {IPFP, REFINE, BIPARTITE, BP_BEAM}, per dataset, against T-03's **3,897,911 certified exact** values. Deliver **one plot + one significance table + one literature table per end**. ≈ 5–20 core-h. **Gates T-05** | T-03 | 1–2 | **P0** | [T-27 spec](../tasks/T-27-spec.md), [approx_ged](approx_ged.md) §2 |

**Retired**: ~~T-01b~~ (new-dataset audit — **superseded**: its numbers are among the unverified ones,
see the header warning) · ~~T-10~~ (merged into T-09) ·
~~T-16~~ (`wl_pruned_canonical` — **rejected**, [decisions](decisions.md) §2).

---

## Dependency structure

**Critical path** — **T-27 → T-05 → T-06 → T-20 → T-15 → T-24**, with T-14 accruing throughout.
**T-01, T-02, T-03 are all done and off the path.** **T-27 is the only remaining gate on T-05**: it
decides which bound T-05 computes. It is 1–2 days and can start immediately.

**T-02 is closed and no longer gates T-06** — but three of its parameters are still open and each has
a named owner: `k` (T-04a), `d` (T-06's own F1 run), and the primary bound at each end (T-27). See
[preregistration](preregistration.md) §7.

**Parallel off it** — T-04 → T-04a → T-17 · T-07 → T-08 → T-19 → T-26 · T-22 · T-13 · T-09 · T-11.

**T-04a gates T-06's distance matrices**, so it is on the path for everything downstream of the
competitors even though it is half a day.

**Ordering constraints that cost rework if violated**: [manuscript](manuscript.md) §5.

---

## Closing a ticket

**Use the `review-close` skill.** It is the counterpart to `review-ticket`: that one drives a ticket
to completion, this one writes it up. It standardises the board entry, the plan-file RESULT section,
the article notes and the letter fragment — and, most importantly, it enforces the rule that cost
T-03 real time to learn:

> **A finding that contradicts a plan file must be written INTO that file, not only into the ticket
> log.** The log is for whoever audits the ticket; the plan files are the instruction set for whoever
> runs the next one. A correction that lives only in the log is one the next agent will not read.

It also names the **inherited-premise trap** — configuring a check from a plan assertion, getting a
clean one-sided result, concluding something about the *data*, then "independently verifying" it with
a second script that shares the same assertion. That is how T-03 briefly concluded GraphEdX's matrix
was approximate when the premise about its cost model was what was wrong.

## Response-letter fragments

**Every ticket emits its response fragment when it closes.** T-14 assembles, harmonises the register
and writes part 0 — three days is enough for that and is not enough for writing 41 answers from
scratch. [demands](demands.md) is the index; an empty fragment cell is a visible hole.
